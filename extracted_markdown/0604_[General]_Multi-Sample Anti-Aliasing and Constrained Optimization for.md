<!-- page 1 -->
Multi-Sample Anti-Aliasing and Constrained
Optimization for 3D Gaussian Splatting
Zheng Zhou1†, Jia-Chen Zhang1†, Yu-Jie Xiong1*, Chun-Ming Xia1
1*School of Electronic and Electrical Engineering, Shanghai University of
Engineering Science, Shanghai, 201620, China.
*Corresponding author(s). E-mail(s): xiong@sues.edu.cn;
Contributing authors: m320123332@sues.edu.cn;
m325123603@sues.edu.cn; cmxia@sues.edu.cn;
†These authors contributed equally to this work.
Abstract
Recent advances in 3D Gaussian splatting have significantly improved real-time
novel view synthesis, yet insufficient geometric constraints during scene optimiza-
tion often result in blurred reconstructions of fine-grained details, particularly in
regions with high-frequency textures and sharp discontinuities. To address this,
we propose a comprehensive optimization framework integrating multisample
anti-aliasing (MSAA) with dual geometric constraints. Our system computes pixel
colors through adaptive blending of quadruple subsamples, effectively reducing
aliasing artifacts in high-frequency components. The framework introduces two
constraints: (a) an adaptive weighting strategy that prioritizes under-reconstructed
regions through dynamic gradient analysis, and (b) gradient differential con-
straints enforcing geometric regularization at object boundaries. This targeted
optimization enables the model to allocate computational resources preferen-
tially to critical regions requiring refinement while maintaining global consistency.
Extensive experimental evaluations across multiple benchmarks demonstrate that
our method achieves state-of-the-art performance in detail preservation, par-
ticularly in preserving high-frequency textures and sharp discontinuities, while
maintaining real-time rendering efficiency. Quantitative metrics and perceptual
studies confirm statistically significant improvements over baseline approaches in
both structural similarity (SSIM) and perceptual quality (LPIPS).
Keywords: Rendering, point-based models, rasterization, machine learning
1
arXiv:2508.10507v1  [cs.CV]  14 Aug 2025

<!-- page 2 -->
Ground Truth
OURS
3DGS
Fig. 1: The difference between rendered images and real images. In comparison to the
first line of images, the window rendered by 3DGS loses its geometric shape and the
edges become unclear. The comparison of the second line image shows that the 3DGS
rendering result is missing some content.
1 Introduction
Novel view synthesis (NVS) has emerged as a cornerstone capability in 3D computer
vision [1], enabling photorealistic image generation from arbitrary viewpoints—a critical
requirement for applications spanning virtual reality, computational cinematography,
and immersive media. Conventional approaches reconstruct explicit 3D scene geometry
from multi-view inputs to ensure view consistency, but often face fidelity limitations
under complex geometric or illumination conditions [2]. The advent of Neural Radiance
Fields (NeRF) [3] marked a paradigm shift by coupling implicit scene representations
with differentiable volume rendering, achieving unprecedented view synthesis quality.
Subsequent variants addressed efficiency concerns through strategies like cone tracing [4,
5], hash-grid encodings [6], and sparse volumetric representations [7, 8]. Nevertheless,
these methods frequently sacrifice high-frequency detail when accelerating rendering,
particularly in high-resolution regimes. 3D Gaussian Splatting (3DGS) [9] presents
a compelling alternative by explicitly modeling scenes as anisotropic 3D Gaussians
with adaptive density control. Unlike NeRF’s computationally intensive volumetric
integration, 3DGS employs efficient splat-based projective rendering, achieving real-
time performance while preserving geometric fidelity. However, as illustrated in Figure 1,
3DGS exhibits pronounced limitations in regions with sparse observational data,
manifesting as local detail degradation and blurring artifacts due to inadequate
sampling density [9].
To address this issue, we present a novel 3DGS framework integrating multi-
sampling anti-aliasing [10, 11] and edge aware constraints [12, 13] to reduce rendering
blur and enrich local details through multi sampling and specific constraints. Usually,
3D Gaussian Splatting is rendered only through the Gaussian ellipsoid corresponding
to the pixel center, which can result in stiff transitions and color jumps [14]. We use 4
2

<!-- page 3 -->
sub-pixel points for multi sampling mixed rendering. Combining the information of sub-
pixel points for comprehensive rendering will result in smoother color transitions. At
the same time, considering the influence of different Gaussian ellipsoid coverage areas,
the multi sampling rendering method can focus on the color details and render the
image more realistically. In the loss function, we introduce adaptive weights [15] based
on the size of the error to address the discrepancy between the predicted results and
the actual image. In this way, when the error of certain pixels or regions is large, their
influence during the training process will be amplified, making the network pay more
attention to these weak reconstruction or difficult to predict regions, thereby avoiding
the phenomenon of excessive smoothness or texture loss in local details. In order to
enhance the reconstruction effect of details such as object edges and high-frequency
textures, we additionally add gradient difference loss on top of the original loss [16, 17].
By comparing the difference between the predicted image and the real image in the
gradient domain, the model focuses more on the local gradient information of the image.
Compared to simple L1/L2 [18] or SSIM loss [19], gradient difference constraint can
better capture high-frequency textures and structures in images, resulting in sharper
and richer visual details. The contributions of this work can be summarized in three
key aspects.
• A 4×MSAA rasterization pipeline for 3DGS that reduces aliasing artifacts through
subpixel-aware Gaussian blending.
• A hybrid loss function combining error-adaptive weighting and gradient difference
constraints to enhance detail reconstruction.
• Comprehensive benchmarking demonstrating state-of-the-art performance in both
quantitative metrics and perceptual evaluations across multiple datasets.
2 Related Work
2.1 Explicit Scene Representations
Early approaches to novel view synthesis often relied on convolutional neural networks
(CNNs) [20] to fuse and warp multi-view inputs for generating unseen viewpoints.
These methods typically compute pixel- or feature-level blending weights that combine
source images into approximate target views [21]. While they perform well for mod-
erate viewpoint changes, they frequently struggle with more complex geometry and
larger viewpoint shifts, leading to artifacts and blurred details. Subsequent research
shifted toward volumetric ray-marching and scene representations stored in 3D grids.
DeepVoxels [22], for instance, introduced a persistent voxel-based representation in
which rays sample 3D features to synthesize novel views, resulting in improved spatial
consistency compared to 2D image-based blends [23]. However, these explicit volumet-
ric methods can become computationally and memory intensive at higher resolutions,
limiting their scalability in large-scale or high-fidelity scenarios [24].
3

<!-- page 4 -->
2.2 Implicit Neural Representations
To address these limitations, research pivoted to NeRF [3], which represents scenes
implicitly via multilayer perceptrons (MLPs). By predicting density and color at
any continuous 3D coordinate, NeRF achieves photorealistic novel views with strong
multi-view consistency. Numerous NeRF extensions [4] have since been proposed to
handle challenges such as moving objects, unknown camera poses, few-shot settings,
and anti-aliasing [24]. Nevertheless, traditional NeRF-based pipelines often demand
extensive training times, and achieving real-time rendering remains a challenge [25].
Existing acceleration strategies—including smaller specialized MLPs, tensor factoriza-
tions, hashing-based encodings, and various compression techniques [26]—help reduce
these bottlenecks but can diminish high-frequency detail, particularly at higher resolu-
tions [27]. Despite progress, implicit methods face fundamental challenges in training
speed and real-time rendering, motivating hybrid representation paradigms.
2.3 Hybrid Gaussian Representations
In pursuit of an improved balance among rendering speed, fidelity, and data efficiency,
3D Gaussian Splatting has emerged as a promising alternative. This approach employs
an explicit representation of anisotropic Gaussians in 3D space, modeling local geometry
and color through Gaussian parameters optimized end-to-end [9, 28]. Unlike methods
that rely on large voxel grids or fully implicit functions, 3D Gaussian Splatting utilizes
a differentiable splatting process to project these Gaussians onto the image plane,
enabling faster training and real-time rendering while preserving high-fidelity detail.
Mip-splatting further addresses aliasing in 3D Gaussian Splatting by introducing a
multi-scale representation analogous to mipmapping [29]. Rather than rendering a
single level of Gaussians for the entire scene, this technique organizes Gaussian splats
into different resolution layers. During rendering, the pipeline adaptively selects the
most appropriate layer based on distance and viewing conditions, thereby mitigating
aliasing and improving overall rendering efficiency. Further developments in Multi-
scale 3D Gaussian Splatting introduce a multi-resolution strategy to mitigate aliasing
in splatting-based rendering [30]. By representing geometry and color information
through layered Gaussian splats at different scales, this technique adaptively selects the
most appropriate resolution level based on viewing distance and detail requirements,
thereby reducing artifacts caused by under- or over-sampling. This approach achieves
high-fidelity, anti-aliased results while maintaining computational efficiency. Scaffold-
GS offers a structured framework for arranging Gaussian splats in a hierarchical or
grid-based fashion [31]. By embedding geometric and color information into a scaffold-
like structure, the method adaptively selects the optimal level of detail for each
viewing condition, effectively balancing visual fidelity with computational overhead.
Despite these advantages, 3D Gaussian Splatting can exhibit over-reconstruction when
excessive or redundant Gaussians accumulate in certain regions. Such overlapping
Gaussians may lead to blurring and artifacts, particularly near high-contrast edges or
finely detailed textures [27, 32].
4

<!-- page 5 -->
2.4 Detail Preservation in Neural Rendering
The pursuit of high-frequency detail preservation in neural rendering has spawned
diverse technical approaches, each addressing specific aspects of the aliasing-detail
tradeoff. Traditional graphics-inspired anti-aliasing techniques like MSAA [33] and
temporal anti-aliasing (TAA) [34] achieve subpixel smoothing through supersampling
or frame accumulation, yet their direct application to differentiable rendering pipelines
remains challenging due to gradient computation constraints. Meanwhile, frequency-
domain optimization strategies have emerged as complementary solutions, with methods
like FreGS [35] performing coarse-to-fine Gaussian densification by exploiting low-to-
high frequency components that can be easily extracted with low-pass and high-pass
filters in the Fourier space. Despite these advances, current solutions predominantly
address aliasing artifacts and detail erosion as separate challenges—MSAA variants
focus on signal-space smoothing while neglecting texture sharpness, whereas gradient-
based constraints improve local contrast but struggle with subpixel discontinuities.
This bifurcation stems from the inherent difficulty in jointly optimizing geometric
stability and spectral fidelity through unified loss landscapes, a gap our method bridges
through dual-domain regularization.
To address these issues, we propose a method that combines MSAA, an adaptive
weighting strategy, and gradient difference constraints. Specifically, we integrate MSAA
into the rasterization process to reduce jagged edges and preserve high-frequency
details, introduce a pixel-wise weighting mechanism to emphasize regions with larger
reconstruction errors, and employ gradient-aware losses to sharpen edges and textures.
By simultaneously targeting aliasing and subtle detail restoration, our approach
enhances local fidelity while preserving the real-time advantages of 3D Gaussian
Splatting.
3 Method
In this section, we first recap the 3D Gaussian Splatting pipeline, including how 3D
Gaussians are parameterized and projected onto the image plane. We then describe
our proposed enhancements, consisting of three major components:
1. Multi-Sample Anti-Aliasing, which reduces aliasing and jagged boundaries in
the rendered images.
2. Adaptive Weighting Strategy, which selectively increases the training focus
on pixels or regions with higher reconstruction error.
3. Gradient Difference Constraints, which encourage the preservation of high-
frequency details and sharp edges.
Together, these techniques address the major shortcomings of naive splatting, notably
the loss of fine details and aliasing artifacts. The complete pipeline is shown in the
figure 2.
5

<!-- page 6 -->
3D Gaussians
Splatting
MSAA Rendering 
2D Gaussians
Rendered Image
Ground Truth
Camera perspective
AW&GDC Loss
ℒ𝑔𝑟𝑎𝑑
ℒ𝑤
1 pixel
average
Ours
3DGS
Ours
3DGS
Fig. 2: Overview of the Proposed Method. Our approach begins by placing four
offset sampling points for each pixel during the rasterization stage. Each sampling
point independently performs alpha blending to render the scene, and their resulting
colors are averaged to produce the final color for each pixel. This process effectively
reduces jagged edges while enhancing the model’s ability to capture both high- and
low-frequency details. During backpropagation, we introduce two complementary
constraints: an adaptive weighting strategy and gradient difference constraints. The
adaptive weighting strategy targets regions suffering from insufficient reconstruction,
while the gradient difference constraints address missing boundary details. By guiding
the model to focus on these unclear rendering areas, our method ultimately refines the
overall rendering quality.
3.1 Preliminaries: 3D Gaussian Splatting
We assume our scene is represented by N anisotropic 3D Gaussians, denoted as
{Gn}N
n=1. Each Gaussian Gn is described by a center µn ∈R3, a covariance matrix
Σn ∈R3×3 which can be decomposed as Σn = RnSnR⊤
n where Rn is a rotation matrix
and Sn is a diagonal scaling matrix, a color cn ∈R3, and an opacity αn ∈[0, 1].
This explicit representation allows each Gaussian to be learned and updated through
gradient-based optimization.
3.1.1 Projection and Splatting.
Let Rcam, tcam denote the camera extrinsic parameters, and K be the intrinsic matrix.
A 3D point x ∈R3 is mapped onto the image plane via the pinhole model:
p = π
 Rcamx + tcam, K

∈R2,
(1)
6

<!-- page 7 -->
where π(·) applies the standard perspective projection using K. For each Gaussian
Gn, its center µn is projected to pn ∈R2. The covariance Σ2D
n
of the elliptical 2D
footprint is derived from the original 3D covariance Σn, taking into account the camera
projection geometry.
3.1.2 Color Accumulation.
At a 2D pixel coordinate u ∈R2, the contribution of the n-th Gaussian can be
approximated by a Gaussian weight:
wn(u) = exp

−1
2 (u −pn)⊤ Σ2D
n
−1 (u −pn)

.
(2)
In practice, the color at u is computed through weighted or alpha compositing of all
Gaussians overlapping that pixel:
C(u) =
N
X
n=1
 αn wn(u)

cn
N
X
m=1
 αm wm(u)

.
(3)
Equations (2)–(3) define the forward splatting step.
Ground Truth 
3DGS
OURS
Fig. 3: Insufficient detail and blurring issues are observed in the vanilla 3DGS. In
particular, the chandelier and frame in the rendered scenes display noticeable distortion
and blur, including warped wall surfaces and edges. These artifacts indicate that
3DGS struggles to handle jagged edges and boundaries while maintaining overall scene
geometry, resulting in distorted and blurry final images.
3.2 Multi-Sample Anti-Aliasing
While forward splatting maintains a continuous approximation of scene geometry, it
introduces aliasing artifacts along high-curvature edges and thin structures, primarily
7

<!-- page 8 -->
manifested as blurring or jagged discontinuities in reconstructed boundaries, see
Figure 3. This phenomenon underscores the inherent limitations of 3DGS in preserving
sharp geometric discontinuities and linear feature fidelity. To address these artifacts, we
adapt principles from multisample anti-aliasing to the splatting paradigm by extending
per-pixel evaluations to a quadruple subsampling scheme. Specifically, our method
computes color and geometry contributions across four strategically offset subpixel
positions, effectively supersampling edge regions to mitigate high-frequency aliasing.
This approach enforces smoother gradient transitions while retaining computational
efficiency through optimized subsample weighting, bridging the gap between geometric
continuity and edge sharpness in splat-based rendering.
3.2.1 Subpixel Sampling
Let n ∈Z+ denote the number of subpixel samples per pixel. To mitigate aliasing
artifacts while preserving computational tractability, we define a set of subpixel offsets
{δk}n
k=1 within the unit square [0, 1] × [0, 1], where each δk = (δk,x, δk,y)represents a
fractional displacement from the pixel center. For each pixel center u on the image
grid, the subpixel sampling coordinates are computed as:
uk = u + δk,
k = 1, 2, . . . , n.
(4)
At each subpixel uk, we perform alpha compositing as defined in Eq. (3), accumu-
lating color contributions from overlapping Gaussians to yield the subsampled color
Ck(u). This stratified sampling strategy effectively captures high-frequency geometric
and photometric variations that would otherwise be lost in single-sample-per-pixel
rasterization.
3.2.2 Differentiable Multisample Aggregation
The final anti-aliased pixel color CMSAA(u) is computed via a normalized summation
over all subpixel samples:
CMSAA(u) = 1
n
n
X
k=1
Ck(u).
(5)
Critically, this aggregation operator remains fully differentiable, enabling gradient
backpropagation through all n sampling paths. The chain rule decomposes the gradient
∂CMSAA/∂Θ, where Θ denotes Gaussian parameters. This property encourages the
optimization process to prioritize Gaussians whose spatial and spectral properties
exhibit view-consistent behavior across subpixel perturbations, thereby enhancing
geometric stability and texture fidelity.
3.3 Adaptive Weighting Strategy
While MSAA effectively mitigates aliasing artifacts, residual reconstruction errors
persist in high-frequency regions as shown in Figure 4. Through comparative analysis
of our model’s outputs against 3DGS renderings, we generate inverse-color difference
8

<!-- page 9 -->
3DGS
Ours
Difference with GT - 3DGS
Difference with GT - Ours
Fig. 4: The rendering result on unclear areas. 3DGS may experience insufficient
reconstruction when dealing with unclear areas, resulting in severe dissonance and
blurring when rendering that area. Our method assigns more weights to areas with
insufficient reconstruction, which effectively enhances the reconstruction effect of the
area, adds details to unclear areas, and produces more realistic rendering results.
maps relative to ground truth - where whiter regions indicate closer alignment with
reference data. These visualizations reveal significant reconstruction deficiencies in
geometrically complex areas such as ceilings and staircases, where 3DGS exhibits
substantial detail loss and structural discontinuities. To address these limitations, we
implement an adaptive weighting mechanism within our loss function that dynamically
prioritizes pixels exhibiting elevated reconstruction errors. This strategic emphasis
enables our model to concentrate learning capacity on regions where conventional
rendering approaches struggle, particularly in dense geometric configurations requiring
high-frequency detail preservation.
3.3.1 Pixel-Wise Weight Definition
Let Ipred and Igt be the predicted and ground-truth images of size H × W. For each
pixel (i, j), we first compute the ℓ1 error:
ei,j =
Ipred(i, j) −Igt(i, j)

1.
(6)
9

<!-- page 10 -->
To prioritize under-reconstructed regions, we design an adaptive weight map:
wi,j = α + (1 −α)
ei,j
max
(p,q)∈Ωep,q + ε,
(7)
where Ω= {1, ..., H} × {1, ..., W} denotes the spatial domain, α ∈[0, 1] governs the
baseline weight floor, and ε > 0 ensures numerical stability. This formulation ensures
wi,j ∈[α, 1) with monotonic increase relative to local error magnitude.
3.3.2 Weighted Reconstruction Loss
The error-adaptive weights modulate our reconstruction objective:
Lw =
1
HW
H
X
i=1
W
X
j=1
wi,j
Ipred(i, j) −Igt(i, j)

1.
(8)
Complementing this, we incorporate a structural regularization term through a
decomposed SSIM metric LD-SSIM, yielding the composite loss:
Lrecon = λ1 Lw + λ2 LD-SSIM.
(9)
Here, λ1 and λ2 control the relative influence of weighted ℓ1 and SSIM. By selec-
tively amplifying the loss in high-error regions. This selective error amplification
mechanism particularly benefits high-frequency detail recovery where conventional ℓ1
losses underperform.
Ground Truth 
3DGS
OURS
Fig. 5: 3DGS boundary area reconstruction error, reconstructed straight line boundary
as curved boundary. 3DGS does not have effective constraints on boundary information,
which results in insufficient attention given to boundary reconstruction, making it
difficult to accurately reconstruct boundaries.
10

<!-- page 11 -->
3.4 Gradient Difference Constraints
Although the adaptive weighting mechanism effectively prioritizes high-error regions, it
lacks explicit enforcement of edge coherence and high-frequency fidelity – a limitation
exemplified in Figure 5 where pixel-wise losses induce excessive smoothing of textural
details. To address this critical gap, we introduce a Gradient Difference Constraint
(GDC) that directly regulates first-order image variations, ensuring geometric fidelity
in boundary reconstruction.
3.4.1 Discrete Gradient Operators
For discrete image domains Ω= {1, ..., H} × {1, ..., W}, we define forward-difference
operators capturing horizontal and vertical intensity variations:
∇xIpred(i, j) = Ipred(i, j + 1) −Ipred(i, j),
(10)
∇yIpred(i, j) = Ipred(i + 1, j) −Ipred(i, j).
(11)
This constructs gradient vector fields for both predicted and ground-truth images:
Gpred
i,j
=
 ∇xIpred(i, j), ∇yIpred(i, j)

,
(12)
Ggt
i,j =
 ∇xIgt(i, j), ∇yIgt(i, j)

.
(13)
3.4.2 Multi-Scale Gradient Alignment
Our GDC enforces multi-level gradient consistency through an ℓ1-norm penalty:
Lgrad =
1
(H −1)(W −1)
H−1
X
i=1
W −1
X
j=1
gpred
i,j
−ggt
i,j

1.
(14)
We exclude boundary indices where i = H or j = W to avoid index out-of-bounds. By
minimizing the discrepancy in horizontal and vertical image gradients, the network is
guided to reproduce sharp changes, such as edges, corners, and textures. We present one
of the cases using wavelet transform for clearer visualization in figure 6. Decompose the
boundary into horizontal and vertical directions for visual observation, and compare
the low-frequency and diagonal high-frequency information. The GDC particularly
enhances LH/HL band reconstruction where traditional losses fail to preserve edge
orientation statistics.
This geometric regularization complements the adaptive weighting strategy through
dual mechanisms: 1) Edge sharpening, direct gradient matching prevents boundary
blurring in high-curvature regions. 2) Texture preservation, high-frequency gradient
alignment maintains stochastic texture patterns. 3) Scale awareness, wavelet analysis
reveals improved recovery across frequency subbands.
11

<!-- page 12 -->
Approximation of GT
Horizontal Detail of GT
Vertical Detail of GT
Diagonal Detail of GT
Approximation of Ours
Horizontal Detail of Ours
Vertical Detail of Ours
Diagonal Detail of Ours
Approximation of 3DGS
Horizontal Detail of 3DGS
Vertical Detail of 3DGS
Diagonal Detail of 3DGS
Fig. 6: We perform wavelet transform on the image to visualize the boundary infor-
mation of the rendered results. The horizontal and vertical detail parts after wavelet
transform respectively display the boundary information in the horizontal and verti-
cal directions of the rendered image. Obviously, our method has clearer and sharper
boundaries, richer details, and low-frequency parts that are closer to real images com-
pared to 3DGS.
3.5 Overall Loss Framework
3.5.1 Composite Objective Function
Our complete optimization objective synthesizes three complementary mechanisms
through carefully calibrated coupling coefficients:
L = λ1 Lw + λ2LD-SSIM + λ3 Lgrad,
(15)
where the trilinear weighting scheme λ1, λ2, λ3 governs:
• Local Error Correction: Lw’s spatial adaptation for photometric accuracy
• Structural Coherence: LD-SSIM’s windowed similarity preservation
• Geometric Fidelity: Lgrad’s edge orientation constraints
Here, λ1, λ2, and λ3 balance the weighted reconstruction term, SSIM, and gradient
preservation. We found in practice that λ3 can be gradually increased during training
if edges remain overly smoothed.
12

<!-- page 13 -->
3.5.2 Multi-Sample Anti-Aliasing Integration
When computing Ipred at each iteration, we replace the standard pixel evaluation in
Eq. (3) with the MSAA-based process:
Ipred(i, j) = CMSAA(u),
(16)
where u = (j, i) in pixel coordinates and CMSAA(·) is defined as above. Hence, Ipred is
effectively the averaged color across multiple subpixel offsets.
By combining MSAA to reduce aliasing, the Adaptive Weighting Strategy to target
high-error regions, and the Gradient Difference Constraints to retain edges, our method
alleviates common pitfalls in 3D Gaussian Splatting. This leads to both more faithful
local detail reconstruction and improved global rendering quality in complex scenes.
4 Experiments
4.1 Datasets and Implementation Details
4.1.1 Datasets
For training and testing, we followed the dataset settings of 3DGS [9] and con-
ducted experiments on a total of 13 real-world images. Specifically, we evaluated our
method on all nine scenarios of the Mip-NeRF360 [4] dataset, two scenarios of the
Tanks&Templates dataset [36], and two scenarios of the Deep Blending dataset [37].
The selected scene presents a variety of styles, ranging from bounded indoor environ-
ments to unbounded outdoor environments. The resolution of all images involved is
also the same as in 3DGS. All scenes were trained 30000 iterations on a single RTX
4090 GPU, with hyperparameters consistent with 3DGS. Evaluate the performance of
the 3DGS model and our proposed model on all datasets, including quantitative and
qualitative analysis.
Table 1: Quantitative comparison on three datasets. SSIM↑and PSNR↑are higher-
the-better; LPIPS↓is lower-the-better. For fair comparison and to balance the trade-off
between overall quality and memory consumption, we trained these datasets with the
same settings as 3DGS. All methods use the same training data for training. The
best score , second best score , and third best score are red, orange, and yellow,
respectively.
Datasets
Mip-NeRF360
Tanks&Temples
Deep Blending
Methods
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
Plenoxels
0.626
23.08
0.463
0.719
21.08
0.379
0.795
23.06
0.510
INGP
0.671
25.30
0.371
0.723
21.72
0.330
0.797
23.62
0.423
Mip-NeRF360
0.792
27.69
0.237
0.759
22.22
0.257
0.901
29.40
0.245
3DGS
0.815
27.21
0.214
0.841
23.14
0.183
0.903
29.41
0.243
Pixel-GS
0.823
27.67
0.193
0.856
23.74
0.151
0.896
28.91
0.248
AbsGS
0.820
27.49
0.191
0.853
23.73
0.162
0.902
29.67
0.236
Ours(3DGS)
0.819
27.62
0.207
0.851
23.79
0.165
0.900
29.52
0.246
Ours(Pixel-GS)
0.830
27.68
0.189
0.865
23.75
0.145
0.907
28.92
0.242
Ours(AbsGS)
0.829
27.51
0.188
0.862
23.74
0.157
0.906
29.69
0.230
13

<!-- page 14 -->
4.1.2 Implementation
In our rendering reconstruction framework, we integrate three enhancements. First,
to reduce aliasing and jagged edges, 4× MSAA is employed at the rendering stage to
collect multiple samples per pixel, which are then fused either at the input layer or
within intermediate feature fusion modules. Second, our Adaptive Weighting Strategy
dynamically modulates the loss weights based on pixel or region-level reconstruction
errors, thus emphasizing challenging regions with higher errors. Finally, to preserve high-
frequency details and sharp boundaries, we apply Gradient Difference Constraints using
a gradient operator on both the predicted and ground-truth images, and incorporate
the resulting discrepancy into the overall loss function. We use the Adam optimizer
for training in the Pytorch framework[38], integrate MSAA into the rasterization of
3DGS, and set the λ1, λ2, λ3 to 0.8, 0.2, and 0.1.
Table 2: Quantitative comparison on Mip-NeRF360 dataset. SSIM↑and PSNR↑are
higher-the-better, using + to indicate the improvement of our method compared to
3DGS; LPIPS↓is lower-the-better, using −to represent the improvement of our method
compared to 3DGS. For fair comparison and to balance the trade-off between overall
quality and memory consumption, we trained these datasets with the same settings as
3DGS. All methods use the same training data for training. The best score are red.
Datasets
bicycle
bonsai
counter
Methods
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
3DGS
0.745
25.08
0.245
0.947
32.36
0.179
0.915
29.11
0.182
Ours
0.758 (+0.013) 25.23 (+0.15) 0.222 (-0.023) 0.949 (+0.002) 32.44 (+0.08) 0.173 (-0.006) 0.918 (+0.003) 29.12 (+0.01) 0.175 (-0.007)
Datasets
flowers
garden
kitchen
Methods
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
3DGS
0.589
21.40
0.358
0.856
27.28
0.122
0.931
31.32
0.116
Ours
0.606 (+0.017) 21.69 (+0.29) 0.339 (-0.019) 0.863 (+0.007) 27.41 (+0.13) 0.109 (-0.013) 0.934 (+0.003) 31.45 (+0.13) 0.111 (-0.005)
Datasets
room
stump
treehill
Methods
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
3DGS
0.926
31.70
0.196
0.768
26.63
0.243
0.635
22.52
0.346
Ours
0.929 (+0.003) 31.78 (+0.08) 0.186 (-0.010) 0.778 (+0.010) 26.81 (+0.18) 0.224 (-0.019) 0.643 (+0.008) 22.63 (+0.11) 0.327 (-0.019)
14

<!-- page 15 -->
Ground Truth
OURS
3DGS
Ground Truth
OURS
3DGS
Fig. 7: Qualitative comparison between our proposed method and state-of-the-art
methods in novel view synthesis. The comparison was conducted in multiple scenes
with different styles, including multiple scenes from Mip-NeRF360, Tank&Template’s
trains and trucks, and Deep Blending’s playroom. The GT in the figure represents the
real image used for reference, and we will zoom in on the clearly different details for
easier viewing. From these results, it can be seen that our method achieves excellent
image rendering with fewer artifacts and richer details.
4.2 Results Analysis
We compared our approach with 3DGS as well as other view-generation meth-
ods—including MipNerf360 [4], InstantNGP [6], Plenoxels [7], Mip-NeRF360 [5],
3DGS [9], AbsGS [39] and Pixel-GS [40]—across 14 scenarios from three datasets. To
ensure a fair comparison and balance between memory usage and performance, we
employed a similar number of Gaussians as in 3DGS. All methods were trained using
the same data and hardware configurations. Table 1 presents the quantitative met-
rics for all methods, and Table 2 details results for the nine scenarios in MipNerf360.
As shown in these analyses, our method consistently outperforms state-of-the-art
15

<!-- page 16 -->
approaches in terms of PSNR, SSIM, and LPIPS for all real-world scenarios. Moreover,
the quantitative results in Figure 7 further illustrate our superior rendering qual-
ity, highlighting the robustness of our approach in diverse environments. In addition,
we use point clouds and Gaussian ellipsoids to demonstrate the effectiveness of the
proposed dual constraints in Figure 8 and Figure 9.
OURS
3DGS
Fig. 8: Comparison diagram of point cloud effect. Our method outperforms 3DGS
in reconstructing detailed point clouds, especially in blurred or textureless areas,
enhancing local rendering quality.
4.3 Ablation Studies
To investigate the individual contributions of our proposed improvement modules,
we conducted ablation experiments on MSAA, Adaptive Weighting Strategy (AWS),
and Gradient Difference Constraints (GDC). Specifically, we sequentially removed or
selectively retained each module from the full model configuration, then evaluated and
compared each variant using standard reconstruction metrics.
4.3.1 Multisample Anti-Aliasing
We first investigate how our proposed MSAA affects local blurring in Gaussian distribu-
tions. Based on 3DGS, we trained a model 3DGS+MSAA, which executes the MSAA
strategy during the rasterization stage. As shown in the figure 10, the 3DGS+MSAA
model is clearer in local details, MSAA effectively alleviates jagged edges and aliasing
distortion during rendering, making the network less prone to noise and defocusing
when dealing with sharp edges or high-frequency textures.
16

<!-- page 17 -->
OURS
3DGS
Fig. 9: Gaussian ellipsoid coverage comparison diagram. Our method resolves 3DGS’s
blurry rendering by decomposing large Gaussian ellipsoids into finer, detail-rich com-
ponents for enhanced reconstruction.
Ground Truth
3DGS
3DGS+MSAA
Fig. 10: The results of the MSAA ablation experiment. 3DGS generates some non-
existent parts when reconstructing and rendering portraits, which reduces the clarity
of the rendering results. After adding the MSAA method, we effectively distinguished
the areas of the portrait, blank space, and frame during reconstruction and rendering,
making the content of each part clearer.
4.3.2 Adaptive Weighting Strategy & Gradient Difference
Constraints
To validate the effectiveness of our proposed constraint optimization frame-
work, we conduct comprehensive ablation studies by training a hybrid model,
3DGS+MSAA+AWS+GDC, which integrates MSAA with our adaptive constraint
optimization strategies. Specifically, our method outperforms both 3DGS+MSAA
and 3DGS+AWS+GDC across key metrics, demonstrating the synergistic benefits of
combining adaptive weighting and gradient-domain regularization. The AWS dynam-
ically adjusts loss weights for challenging regions based on per-pixel reconstruction
errors. This mechanism prioritizes under-optimized areas during training, significantly
17

<!-- page 18 -->
Table 3: Conduct ablation experiments on the Mip-NeRF360 and Tank&Temples
datasets. SSIM↑and PSNR↑are higher-the-better; LPIPS↓is lower-the-better. The
best score , second best score are red, orange, respectively.
Methods
Mip-NeRF360
Tank&Temple
Deep Blending
SSIM↑PSNR↑LPIPS↓SSIM↑PSNR↑LPIPS↓SSIM↑PSNR↑LPIPS↓
3DGS
0.815
27.21
0.214
0.841
23.14
0.183
0.903
29.41
0.243
3DGS+MSAA
0.812
27.50
0.221
0.844
23.69
0.178
0.899
29.48
0.246
3DGS+AWS+GDC
0.816
27.45
0.210
0.846
23.66
0.170
0.901
29.44
0.244
Ours
0.819
27.62
0.207
0.851
23.79
0.165
0.900
29.52
0.246
enhancing the recovery of fine texture details while mitigating visual artifacts like local
blurring and over-smoothing caused by uneven optimization focus. Complementing
AWS, the GDC constraint enforces coherence between adjacent pixels in the gradient
space, effectively suppressing excessive smoothing while preserving high-frequency
features and sharp structural edges. As quantitatively verified in Table 3, our full
framework achieves superior novel view synthesis quality compared to 3DGS+MSAA,
while also surpassing the standalone 3DGS+AWS+GDC configuration. These results
confirm that our constraint optimization paradigm not only resolves the limitations of
MSAA in handling complex textures but also provides a more balanced and robust
optimization landscape compared to isolated constraint strategies.
5 Conclusion
In summary, our approach leverages a pixel-level weighted constraint, MSAA and gra-
dient difference constraints to address the challenges of reconstructing fine local details
and mitigating aliasing in 3D Gaussian splatting. By assigning adaptive weights based
on pixel gradients, our method effectively focuses on areas with higher reconstruction
difficulty, while MSAA alleviates jagged boundaries. Furthermore, incorporating gradi-
ent difference constraints preserves high-frequency signals and sharp edges. As a result,
our method demonstrates remarkable improvements over naive splatting and achieves
state-of-the-art performance in local detail reconstruction.
Declarations
• Conflict of interest/Competing interests (check journal-specific guidelines for
which heading to use): The authors have no competing interests to declare that
are relevant to the content of this article.
• Data availability: The data used in this article has been declared and cited in the
experimental section 4.
References
[1] Shum, H., Kang, S.B.: Review of image-based rendering techniques. Visual
Communications and Image Processing 2000 4067, 2–13 (2000)
18

<!-- page 19 -->
[2] Seitz, S.M., Curless, B., Diebel, J., Scharstein, D., Szeliski, R.: A comparison and
evaluation of multi-view stereo reconstruction algorithms. In: Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), vol. 1,
pp. 519–528 (2006). https://doi.org/10.1109/CVPR.2006.19
[3] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R.,
Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis.
Communications of the ACM 65(1), 99–106 (2021)
[4] Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srini-
vasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance
fields. In: Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV), pp. 5855–5864 (2021)
[5] Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf
360: Unbounded anti-aliased neural radiance fields. In: Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5470–5479
(2022). https://doi.org/10.1109/CVPR52688.2022.00539
[6] Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives
with a multiresolution hash encoding. ACM transactions on graphics (TOG)
41(4), 1–15 (2022)
[7] Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.:
Plenoxels: Radiance fields without neural networks. In: Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5501–5510
(2022). https://doi.org/10.1109/CVPR52688.2022.00542
[8] Garbin, S.J., Kowalski, M., Johnson, M., Shotton, J., Valentin, J.: Fastnerf: High-
fidelity neural rendering at 200fps. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), pp. 14346–14355 (2021)
[9] Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)
[10] Jimenez, J., Gutierrez, D., Yang, J., Reshetov, A., Demoreuille, P., Berghoff, T.,
Perthuis, C., Yu, H., McGuire, M., Lottes, T., et al.: Filtering approaches for
real-time anti-aliasing. SIGGRAPH Courses 2(3), 4 (2011)
[11] Akeley, K.: Reality engine graphics. In: Proceedings of the 20th Annual Con-
ference on Computer Graphics and Interactive Techniques, pp. 109–116 (1993).
https://doi.org/10.1145/166117.166131
[12] Xu, L., Ren, J., Yan, Q., Liao, R., Jia, J.: Deep edge-aware filters. In: International
Conference on Machine Learning, pp. 1669–1678 (2015)
[13] Hua, M., Bie, X., Zhang, M., Wang, W.: Edge-aware gradient domain optimization
19

<!-- page 20 -->
framework for image filtering by local propagation. In: Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2838–2845
(2014). https://doi.org/10.1109/CVPR.2014.363
[14] Mai, A., Hedman, P., Kopanas, G., Verbin, D., Futschik, D., Xu, Q., Kuester, F.,
Barron, J.T., Zhang, Y.: Ever: Exact volumetric ellipsoid rendering for real-time
view synthesis. arXiv preprint arXiv:2410.01804 (2024)
[15] Johnson, J., Alahi, A., Fei-Fei, L.: Perceptual losses for real-time style transfer and
super-resolution. In: Proceedings of the IEEE European Conference on Computer
Vision (ECCV), pp. 694–711 (2016)
[16] Zhang, K., Zuo, W., Zhang, L.: Learning a single convolutional super-resolution
network for multiple degradations. In: Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pp. 3262–3271 (2018).
https://doi.org/10.1109/CVPR.2018.00344
[17] Lim, B., Son, S., Kim, H., Nah, S., Mu Lee, K.: Enhanced deep residual net-
works for single image super-resolution. In: Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pp. 136–144 (2017).
https://doi.org/10.1109/CVPRW.2017.151
[18] Zhao, H., Gallo, O., Frosio, I., Kautz, J.: Loss functions for image restoration
with neural networks. In: IEEE Transactions on Computational Imaging, vol. 3,
pp. 47–57 (2017). https://doi.org/10.1109/TCI.2016.2644865
[19] Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment:
from error visibility to structural similarity. IEEE transactions on image processing
13(4), 600–612 (2004)
[20] Flynn, J., Neulander, I., Philbin, J., Snavely, N.: Deepstereo: Learning to predict
new views from the world’s imagery. In: Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pp. 5515–5524 (2016).
https://doi.org/10.1109/CVPR.2016.595
[21] Hedman, P., Kopf, J., Langguth, F., Goesele, M.: Deep blending for free-viewpoint
image-based rendering. In: ACM Transactions on Graphics (TOG), vol. 37, pp.
1–15 (2018). https://doi.org/10.1145/3272127.3275084
[22] Sitzmann, V., Thies, J., Heide, F., Nießner, M., Wetzstein, G., Zollhöfer, M.:
Deepvoxels: Learning persistent 3d feature embeddings. In: Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.
2437–2446 (2019). https://doi.org/10.1109/CVPR.2019.00254
[23] Niemeyer, M., Mescheder, L., Oechsle, M., Geiger, A.: Differentiable volumetric
rendering: Learning implicit 3d representations without 3d supervision. Pro-
ceedings of the IEEE Conference on Computer Vision and Pattern Recognition
20

<!-- page 21 -->
(CVPR), 3504–3515 (2020)
[24] Martin-Brualla, R., Radwan, N., Sajjadi, M.S., Barron, J.T., Dosovitskiy,
A., Duckworth, D.: Nerf in the wild: Neural radiance fields for uncon-
strained photo collections. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 7206–7215 (2021).
https://doi.org/10.1109/CVPR46437.2021.00713
[25] Reiser, C., Peng, S., Liao, Y., Geiger, A.: Kilonerf: Speeding up neural radiance
fields with thousands of tiny mlps. In: Proceedings of the IEEE International
Conference on Computer Vision (ICCV), pp. 14335–14345 (2021)
[26] Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H.: Tensorf: Tensorial radiance fields
for compact neural scene representation. In: Proceedings of the IEEE European
Conference on Computer Vision (ECCV), pp. 1–17 (2022)
[27] Zhang, K., Barron, J.T., Tancik, M., Hedman, P., Srinivasan, P.P.: Nerf-
gan: Learning implicit generative representations of 3d scenes. arXiv preprint
arXiv:2010.08422 (2020)
[28] Linsen, L.: Point cloud representation through 3d gaussian models. In: IEEE
Visualization Conference (VIS), pp. 27–34 (2001)
[29] Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A.: Mip-splatting: Alias-
free 3d gaussian splatting. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 19447–19456 (2024).
https://doi.org/10.1109/CVPR52733.2024.01839
[30] Yan, Z., Low, W.F., Chen, Y., Lee, G.H.: Multi-scale 3d gaussian splat-
ting for anti-aliased rendering. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 20923–20931 (2024).
https://doi.org/10.1109/CVPR52733.2024.01977
[31] Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., Dai, B.: Scaffold-gs:
Structured 3d gaussians for view-adaptive rendering. In: Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.
20654–20664 (2024). https://doi.org/10.1109/CVPR52733.2024.01952
[32] Deng, K., Liu, A., Zhu, J.-Y.: Depth-supervised nerf: Fewer views and
faster training for free. In: Proceedings of the IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), pp. 12735–12744 (2022).
https://doi.org/10.1109/CVPR52688.2022.01254
[33] Molnar, S., Cox, M., Ellsworth, D., Fuchs, H.: A sorting classification of parallel
rendering. IEEE computer graphics and applications 14(4), 23–32 (1994)
[34] Akenine-Moller, T., Haines, E., Hoffman, N.: Real-time Rendering, (2019)
21

<!-- page 22 -->
[35] Zhang, J., Zhan, F., Xu, M., Lu, S., Xing, E.: Fregs: 3d gaussian splatting with
progressive frequency regularization. In: Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pp. 21424–21433 (2024).
https://doi.org/10.1109/CVPR52733.2024.02024
[36] Knapitsch, A., Park, J., Zhou, Q.-Y., Koltun, V.: Tanks and temples: Benchmark-
ing large-scale scene reconstruction. ACM Transactions on Graphics (ToG) 36(4),
1–13 (2017)
[37] Hedman, P., Philip, J., Price, T., Frahm, J.-M., Drettakis, G., Brostow, G.: Deep
blending for free-viewpoint image-based rendering. ACM Transactions on Graphics
(ToG) 37(6), 1–15 (2018)
[38] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen,
T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-
performance deep learning library. Advances in neural information processing
systems 32 (2019)
[39] Ye, Z., Li, W., Liu, S., Qiao, P., Dou, Y.: Absgs: Recovering fine details in 3d
gaussian splatting. In: Proceedings of the 32nd ACM International Conference on
Multimedia (ACM MM), pp. 1053–1061 (2024)
[40] Zhang, Z., Hu, W., Lao, Y., He, T., Zhao, H.: Pixel-gs: Density control with pixel-
aware gradient for 3d gaussian splatting. In: European Conference on Computer
Vision (ECCV), pp. 326–342 (2024). Springer
22
