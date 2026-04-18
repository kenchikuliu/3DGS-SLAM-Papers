# ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction With Fewer Primitives

BartÅomiej Baranowski Stefano Esposito Patricia GschoÃmann

Anpei Chenâ  Andreas Geiger

University of Tubingen, T Â¨ ubingen AI Center Â¨

baranowskibrt.github.io/conegs

<!-- image-->

<!-- image-->

1  
<!-- image-->

2  
<!-- image-->

3  
<!-- image-->  
Figure 1. ConeGS replaces cloning-based densification with a novel method that generates pixel-cone-sized primitives in regions of high image-space error. By improving placement and removing reliance on existing scene structure thanks to a flexible iNGP-based exploration, it achieves higher reconstruction quality than baselines using the same number of primitives. Results are averaged over Mip-NeRF 360 [3] and OMMO [33] datasets, with a visual comparison on the truck scene from Tanks & Temples [26].

## Abstract

3D Gaussian Splatting (3DGS) achieves state-of-the-art image quality and real-time performance in novel view synthesis but often suffers from a suboptimal spatial distribution of primitives. This issue stems from cloning-based densification, which propagates Gaussians along existing geometry, limiting exploration and requiring many primitives to adequately cover the scene. We present ConeGS, an image-space-informed densification framework that is independent of existing scene geometry state. ConeGS first creates a fast Instant Neural Graphics Primitives (iNGP) reconstruction as a geometric proxy to estimate per-pixel depth. During the subsequent 3DGS optimization, it identifies high-error pixels and inserts new Gaussians along the

corresponding viewing cones at the predicted depth values, initializing their size according to the cone diameter. A preactivation opacity penalty rapidly removes redundant Gaussians, while a primitive budgeting strategy controls the total number of primitives, either by a fixed budget or by adapting to scene complexity, ensuring high reconstruction quality. Experiments show that ConeGS consistently enhances reconstruction quality and rendering performance across Gaussian budgets, with especially strong gains under tight primitive constraints where efficient placement is crucial.

## 1. Introduction

Neural Radiance Fields (NeRF) [37] have significantly advanced novel view synthesis, achieving remarkable fidelity in scene reconstruction. However, representing scenes with neural networks makes NeRF slow to train and render, though it provides smooth parameterization and flexibility to handle changes in scene structure. Recently, 3D Gaussian Splatting (3DGS) [24] has gained attention as a faster, more practical alternative to NeRF, explicitly modeling scenes with sets of 3D Gaussians to achieve interactive rendering speeds while maintaining competitive visual fidelity. However, 3DGS increases expressiveness through cloning and splitting, which offer limited exploration, rely on hard-todefine densification rules, and generate many unnecessary primitives. As a result, primitives often accumulate in suboptimal regions, leaving large parts of the scene underrepresented or mispredicted.

<!-- image-->  
Figure 2. Densification comparison. Cloning-based methods are difficult to tune, and the resulting primitives may require many iterations to fit correctly into the scene. ConeGS, by contrast, places primitives precisely using the pixel viewing cone size, enabling faster scene integration without reliance on the existing geometry.

To address these issues, we propose ConeGS, which replaces cloning-based densification with a novel strategy that targets pixels exhibiting high photometric error. By sampling these pixels and using depth estimates from a fast Instant Neural Graphics Primitives (iNGP) [39] reconstruction, new Gaussians are placed precisely in regions where the current representation is insufficient. This targeted placement increases expressiveness in areas requiring higher primitive density, improving reconstruction quality while avoiding redundant primitives. To determine the size of new Gaussians, we draw inspiration from Mip-NeRF [2]. During densification, each Gaussian is initialized according to the size of the viewing cone of the pixel from which it is generated at the specified depth. Their initial size is thus defined directly by their image-space coverage, eliminating the need for local size analysis or adjustments to reconstructed regions. Figure 2 illustrates the effectiveness of the proposed approach. Combined with a pre-activation opacity penalty that quickly removes redundant Gaussians, this enables scene representation with fewer primitives while preserving high reconstruction quality. We further incorporate two primitive budgeting strategies to regulate the total number of primitives, either through a fixed budget or by adapting to scene complexity. ConeGS outperforms baseline methods across diverse datasets and a wide range of primitive budgets. The advantage is most pronounced under tight primitive budgets. At higher budgets, it matches the quality of cloning-based methods, where efficient primitive placement is less critical, while still rendering faster than the baselines. In summary, our contributions are:

â¢ A densification strategy that places new Gaussians in regions of high photometric error in image space, guided by depth estimates from an iNGP-based geometric proxy.

â¢ An approach that determines the size of new Gaussians from the viewing cones of the pixels from which they are generated.

â¢ An improved opacity penalty that promptly removes lowopacity Gaussians, combined with a budgeting strategy that balances scene complexity and primitive count.

Finally, our method is also compatible with other 3DGS improvements, making it straightforward to integrate with existing approaches for greater efficiency, or with methods where cloning strategies are ambiguous or hard to formalize [20, 34, 49].

## 2. Related work

Neural Radiance Fields: NeRFs [37] represent scenes as continuous volumetric radiance fields, enabling highquality novel view synthesis. This is achieved by parameterizing the scene with a neural network (typically an MLP), whose weights encode the scene globally. Despite producing photorealistic results, these methods rely on costly volumetric rendering and remain computationally inefficient. Extensions such as Mip-NeRF [2] and Mip-NeRF360 [3] reduce aliasing via conical frustum integration, while Zip-NeRF [4] improves view consistency with hierarchical sampling and multi-scale supervision. Hybrid approaches [7, 45, 46, 56] mitigate this by combining explicit data structures with compact neural representations, enabling faster optimization and real-time rendering. Instant Neural Graphics Primitives (iNGP) [39] further accelerate training through multi-resolution hash-grid encoding and shallow MLPs.

Primitive-based Differentiable Rendering: 3D Gaussian Splatting (3DGS) [24] has emerged as an efficient alternative to Neural Radiance Fields (NeRF) [37]. Rather than modeling the scene as a global volume, 3DGS represents it with local explicit 3D Gaussians and uses differentiable rasterization, resulting in significantly faster rendering. Its balance of fidelity and efficiency has attracted significant attention and spurred a wide range of follow-up research. Prior works have focused on tackling anti-aliasing [53, 57], reconstructing dynamic scenes [52, 54], enabling generative content creation [48, 66], reducing rendering artifacts [43], substituting alpha composition with volumetric rendering [35, 47], extracting geometry [16, 21, 58], levelof-detail reconstruction [44], frequency-based regularization [59, 60], and introducing new primitives or kernel functions [18, 20, 31, 49]. Recent efforts have also targeted reducing computational and memory costs, often through feature quantization or code-book encoding [10, 15, 34, 41], or scene simplification [63]. [11] reduces computation by lowering the number of primitives through an aggressive densification and pruning strategy, while [11, 12, 64] insert new Gaussians at the currently estimated depths using reinitialization. Unlike our method, this approach overwrites existing structures instead of adding new points, and further depends on the scene already being well reconstructed. [25] improves primitive distribution and exploration by incorporating positional errors and applying penalties to opacity and scaling. Closely related to our approach, several works focus on improving densification to reduce redundancy and better capture fine details. Strategies include refining cloning heuristics [5, 22, 25], per-Gaussian propertyor saliency-based cues [36], geometry- and volume-aware criteria [1, 23, 65], addressing gradient collision [55], perceptual sensitivity [64], learnable schemes [32, 40], and based on Gaussian Processes [17]. Some works target densification in challenging settings [38], filling holes in the representation [9, 29], though typically adding only a few primitives. PixelSplat [6] models dense probability distributions for more robust Gaussian placement, influencing later approaches [8]. Recent work [27] suggests that densification may be unnecessary for high-quality reconstruction given strong initialization. Like our method, they start by estimating scene geometry, but rely only on correspondences from a pretrained dense matching network, without enhancing densification, and at higher GPU memory cost than our approach. Concurrent work [62], employs Gaussians with spatially varying texture colors, improving finedetail reconstruction and reducing the number of primitives needed. Other methods use neural radiance fields for depth supervision [14, 28] or point cloud extraction [14, 42, 50] to initialize a scene, but not to improve densification directly. Concurrent work [13] applies NeRF for initialization and limited densification, constrained by existing Gaussian locations, and does not explore varying Gaussian sizes, which we find beneficial for reconstruction quality.

## 3. Preliminaries

3D Gaussian Splatting: 3DGS [24] represents a scene as an unordered set of 3D Gaussian primitives $\{ \mathcal { G } _ { i } | i ^ { \mathrm { ~ \scriptsize ~ = ~ } }$ $1 , \ldots , M \}$ . Each primitive $\mathcal { G } _ { i } = \left( \mathbf { p } _ { i } , \mathbf { s } _ { i } , \mathbf { R } _ { i } , o _ { i } , \mathbf { c } _ { i } \right)$ is defined by its position $\mathbf { p } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ , scaling vector $\mathbf { s } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ , rotation matrix ${ \bf R } _ { i } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ , opacity $o _ { i } \in \mathbb { R } .$ , and viewdependent color $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ . The color $\mathbf { c } _ { i }$ is represented by spherical harmonics (SH) coefficients $\mathbf { k } _ { i } ~ \in ~ \mathbb { R } ^ { 3 L }$ , where L is the number of coefficients determined by the chosen SH order. The 3D covariance matrix is given by $\Sigma _ { i } ~ =$ ${ \bf R } _ { i } { \bf S } _ { i } { \bf S } _ { i } ^ { T } { \bf R } _ { i } ^ { T }$ where $\mathbf { S } _ { i } \ = \ \mathrm { d i a g } ( \mathbf { s } _ { i } )$ is the scaling matrix. The color CË of a pixel is computed by Î±-blending over a set of N Gaussians, sorted by depth, whose projections overlap the pixel:

$$
\begin{array} { r } { \hat { C } = \sum _ { i \in N } \mathbf { c } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , } \end{array}\tag{1}
$$

$$
\alpha _ { i } = o _ { i } K ( \mathbf { p } _ { c } , \pmb { \mu } _ { i } ^ { \mathrm { 2 D } } , \pmb { \Sigma } _ { i } ^ { \mathrm { 2 D } } ) ,\tag{2}
$$

where $\alpha _ { i }$ is the blending weight of the i-th Gaussian, $\mathbf { p } _ { c }$ is the pixel center in image coordinates, $\mu _ { i } ^ { \mathrm { 2 D } }$ and $\Sigma _ { i } ^ { \mathrm { 2 D } }$ are the 2D projected mean and covariance of Gi, and $K ( \cdot )$ is a Gaussian filter response in screen space. The exact form of K depends on the chosen filter [24, 57, 67]. Gaussians are traditionally initialized from an SfM point cloud, with each component of $\mathbf { s } _ { i }$ set equal to the mean Euclidean distance to the three nearest neighbors $\mathcal { N } _ { 3 } ( i )$ of Gaussian i:

$$
\mathbf { s } _ { i } = ( s _ { i } , s _ { i } , s _ { i } ) , \quad s _ { i } = \frac 1 3 \sum _ { k \in \mathcal { N } _ { 3 } ( i ) } \left. \mathbf { p } _ { k } - \mathbf { p } _ { i } \right. \quad .\tag{3}
$$

During training, the Gaussian parameters are optimized with the loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G S } } = \left( 1 - \lambda \right) \mathbf { M } \mathbf { A } \mathrm { E } ( I , I ^ { * } ) + \lambda \mathcal { L } _ { D - S S I M } , } \end{array}\tag{4}
$$

where $\lambda = 0 . 2$ , MAE is the mean absolute error between the rendered image I and the ground-truth image Iâ, and $\mathcal { L } _ { D \cdot S S I M } = 1 - \mathrm { S S I M } ( I , I ^ { \ast } )$ [51].

Neural Radiance Fields: NeRFs [37] model a scene as a continuous 3D field that maps a 3D location along a camera ray and the viewing direction of the corresponding pixel to a density $\sigma \in \mathbb { R }$ and color $\mathbf { c } \in \mathbb { R } ^ { 3 }$ . A camera ray is parameterized as $\mathbf { r } ( t ) = \mathbf { p } _ { \mathrm { c a m } } + t \mathbf { d } .$ , where $\mathbf { p } _ { \mathrm { c a m } }$ is the camera position and d is a unit direction vector pointing toward the center of a pixel. Each ray is discretized into N intervals defined by distances $\{ t _ { i } , \bar { t _ { i + 1 } } \} _ { i = 1 } ^ { N }$ . For each sample position $\mathbf { r } ( t _ { i } )$ along the ray, the NeRF is queried to predict the sampleâs color $\mathbf { c } _ { i }$ and density $\sigma _ { i } .$ . Using volumetric rendering [37], the corresponding pixel color is approximated as:

$$
\hat { C } = \sum _ { i = 1 } ^ { N } \mathbf { c } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{5}
$$

$$
\alpha _ { i } = 1 - \exp ( - \sigma _ { i } \delta _ { i } ) \quad \mathrm { w i t h ~ } \delta _ { i } = t _ { i + 1 } - t _ { i } .\tag{6}
$$

Here, $\alpha _ { i }$ is the opacity of the i-th sample, $\delta _ { i }$ is the length of its ray segment, and the product term represents the transmittance $\begin{array} { r } { \tau _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } \end{array}$

Sampling only a single ray per pixel can lead to blur and aliasing. Mip-NeRF [2] addresses this by replacing the ray with a cone that models the pixel footprint, i.e. the 3D volume a pixel covers in world space. The cone is divided into frustums, and integration is performed over these volumes rather than along a 1D line. The coneâs radius $r _ { \mathrm { c o n e } } ( t )$ defines the cross-section of the pixel cone at distance t and is computed from the directions of rays passing through the pixel and its neighbors:

$$
r _ { \mathrm { c o n e } } ( t ) = t \frac { \| \mathbf { d } _ { x } - \mathbf { d } \| + \| \mathbf { d } _ { y } - \mathbf { d } \| } { 2 } ,\tag{7}
$$

where d is the direction of the ray through the center of the pixel, and $\mathbf { d } _ { x } , \mathbf { d } _ { y }$ are the directions of rays through the neighboring pixels in the x and y directions, respectively.

## 4. Method

This section outlines our ray-based densification approach for 3DGS. First, we explain how we use an iNGP model as a geometric proxy to initialize the 3D Gaussian scene (Section 4.1). Next, we detail our ray-based densification strategy, which uses the iNGP to place pixel-cone-sized Gaussians in high-error regions, along with associated optimization changes (Section 4.2). Finally, we provide additional implementation details in Section 4.3. An overview of the complete pipeline is shown in Figure 3.

## 4.1. Initialization

We use a trained iNGP model [39] as a geometric proxy to initialize the 3DGS scene and guide densification. Trained briefly on input images, it provides accurate depth estimates, that position both the initial Gaussian primitives and those added later during densification, with minimal impact on training time. Additionally, the depths can be evaluated on the fly during optimization, reducing both memory usage and computation compared to precomputing all depth maps. We initialize the scene with $\mathcal { P } _ { \mathrm { i n i t } }$ Gaussians, set to one million as in [42], or fewer if a smaller budget is specified (see Section 4.2). To construct this set, we uniformly sample ${ \mathcal { P } } _ { \mathrm { i n i t } }$ image-pixel pairs $( I , u , v )$ from the training set pixel domain $\mathcal { T } _ { \mathrm { t r a i n } }$ . Each sampled image-pixel pair defines exactly one Gaussian in the initialized scene. For each sample, we define its associated camera ray

$$
\mathbf { r } _ { I } ( u , v , t ) = \mathbf { p } _ { I } + t \mathbf { d } _ { I } ( u , v ) ,\tag{8}
$$

where I is an image index, $( u , v )$ are pixel coordinates, ${ \mathbf p } _ { I } \in \mathbb { R } ^ { 3 }$ is the camera center, and $\mathbf { d } _ { I } ( u , v ) \in \mathbb { R } ^ { 3 }$ is the normalized ray direction. We query the iNGP along this ray to obtain discrete transmittance values $\{ \tau _ { k } \}$ , from which the median depth $t _ { \mathrm { m e d } }$ is computed as:

$$
t _ { \mathrm { m e d } } = t _ { k } \quad \mathrm { w h e r e } \quad \tau _ { k - 1 } > 0 . 5 \geq \tau _ { k } .\tag{9}
$$

The center of the j-th Gaussian primitive is then set to:

$$
\mathbf { p } _ { j } = \mathbf { p } _ { I } + t _ { \mathrm { m e d } } \mathbf { d } _ { I } ( u , v ) ,\tag{10}
$$

yielding the set of initial centers $\{ \mathbf { p } _ { j } \} _ { j \in \mathbb { Z } \mathrm { s a m p l e } }$ with $\mathcal { T } _ { \mathrm { s a m p l e } } \subset \mathcal { T } _ { \mathrm { t r a i n } }$ . The scale ${ \bf s } _ { j }$ is initialized isotropically using the average distance to the three nearest neighbors $\mathcal { N } _ { 3 } ( j )$ , following Eq. (3). The rotation is set to identity $\mathbf { R } _ { j } = \mathbf { I }$ , the opacity to $o _ { j } = 0 . 1$ , and the SH coefficients to:

$$
\mathbf { k } _ { j } = \left( \mathbf { k } _ { 1 : 3 , j } , \mathbf { k } _ { 4 : L , j } \right) , \quad \mathbf { k } _ { 1 : 3 , j } = \mathbf { c } _ { j , 0 } , \quad \mathbf { k } _ { 4 : L , j } = \mathbf { 0 } ,\tag{11}
$$

where $\mathbf { c } _ { j , 0 }$ is the RGB color for the sampled pixel $( I , u , v )$ rendered with iNGP using a zeroed-out view direction. Although our densification strategy can achieve high-quality reconstructions without scene initialization, we retain this step to ensure consistently strong performance across all metrics (see Section 5.2).

## 4.2. Optimization

We fully replace the standard 3DGS cloning-based densification with an error-guided strategy that adapts the iNGPbased ray-depth rendering procedure from Section 4.1 to position new Gaussian primitives. Below, we outline the sampling, scaling, budgeting, and pruning stages of our densification pipeline.

Error-Weighted Gaussian Densification: To limit the number of primitives while targeting poorly reconstructed regions, we add new Gaussians at pixels with high photometric error. At iteration j, we render an image $I _ { j }$ and compute the per-pixel absolute error $( L _ { 1 }$ loss) $E ( \mathbf { p } ) \mathbf { \tau } =$ $| I _ { j } ( \mathbf { p } ) - I ^ { * } ( \mathbf { p } ) |$ with respect to the ground-truth image $I ^ { * }$ We then sample $N _ { \mathrm { s a m p l e } }$ pixels without replacement according to a multinomial distribution M with probabilities proportional to the normalized error map:

$$
\left\{ \mathbf { p } _ { s } \right\} _ { s = 1 } ^ { N _ { \mathrm { s a m p l e } } } \sim \mathcal { M } \left( N _ { \mathrm { s a m p l e } } , \frac { E ( \mathbf { p } ) } { \sum _ { \mathbf { p ^ { \prime } } \in I _ { j } } E ( \mathbf { p ^ { \prime } } ) } \right) ,\tag{12}
$$

While $L _ { 1 }$ loss does not always indicate possible improvements and can also arise from noise or difficult-tooptimize reflections, we found it to be a reliable indication of lacking expressiveness, especially at low primitive budgets. For each sampled pixel $\mathbf { p } _ { s }$ , a new Gaussian $\mathcal { G } _ { s }$ is created, with its center placed along the corresponding ray at the median depth $t _ { \mathrm { m e d } , s }$ given by the iNGP. Newly spawned Gaussians are appended to an accumulation set:

$$
\mathcal { G } _ { \mathrm { a c c u m } }  \mathcal { G } _ { \mathrm { a c c u m } } \cup \{ \mathcal { G } _ { s } \} _ { s = 1 } ^ { N _ { \mathrm { s a m p l e } } } .\tag{13}
$$

Every 100 iterations, low-opacity Gaussians in the scene $\mathcal { G } _ { \mathrm { s c e n e } }$ are pruned, and Gaussians in $\mathcal { G } _ { \mathrm { a c c u m } }$ are merged into the scene:

$$
\mathcal { G } _ { \mathrm { s c e n e } }  \mathcal { G } _ { \mathrm { s c e n e } } \cup \mathcal { G } _ { \mathrm { a c c u m } } , \quad \mathcal { G } _ { \mathrm { a c c u m } }  \emptyset .\tag{14}
$$

The newly inserted Gaussians are then jointly optimized with existing primitives. Unlike cloning-based methods, which constrain new primitives to the vicinity of existing ones and thus hinder exploration of unseen regions, our approach places Gaussians directly in high-error areas, enabling effective scene coverage even far from existing geometry. Moreover, we preserve the integrity of wellreconstructed areas, since new Gaussians are added on top of the existing structure rather than created through splitting or cloning. Although occasional iNGP depth inaccuracies may introduce misplaced Gaussians, diverse viewpoint coverage ensures that inconsistent ones are quickly corrected or pruned, while multiview-consistent ones are retained. The full densification pipeline is shown in Figure 4.

<!-- image-->  
Figure 3. Overview of the ConeGS pipeline. (a) First, an iNGP reconstruction is obtained to serve as a geometric proxy for object surfaces, guiding the placement of Gaussians both during scene initialization and throughout the 3DGS optimization process. (b) During 3DGS optimization, ConeGS performs error-guided densification by sampling a subset of pixels with high $L _ { 1 }$ error. For each sampled pixel, a new Gaussian G is created along the pixelâs viewing cone at the depth estimated by iNGP and scaled to match the coneâs size. New Gaussians are accumulated and, every 100 iterations, inserted into the scene after pruning those with low opacity. Blue arrows indicate gradient updates to Gaussian parameters, and the red arrow marks scene updates.

Pixel-Footprint-Aligned Scaling: Selecting an appropriate scale for newly added Gaussians during densification is crucial. If primitives are too large, they may obscure fine details and be pruned prematurely, whereas overly small ones contribute little to the rendered image, yielding weak gradients and slowing convergence. Although k-NNâbased scaling is effective for initialization, recomputing nearestneighbor distances at every densification step is computationally expensive and sensitive to outliers. Large distances can produce inflated scales, causing new Gaussians to overlap well-reconstructed regions and hinder further optimization (see Section 5.2). To avoid these issues, we set the initial scale of each newly added Gaussian directly from the pixel footprint at the median depth $t _ { \mathrm { m e d } , i }$ along its corresponding camera ray, as defined in Eq. 7:

$$
\begin{array} { r } { \mathbf { s } _ { i } = \lambda _ { \mathrm { s c a l e } } r _ { \mathrm { c o n e } } ( t _ { \mathrm { m e d } , i } ) ( 1 , 1 , 1 ) , } \end{array}\tag{15}
$$

where $\lambda _ { \mathrm { s c a l e } } = 2$ converts the cone radius $r _ { \mathrm { c o n e } } ( t _ { \mathrm { m e d } , i } )$ to the diameter of its cross-section. This ensures that, from the spawning viewpoint, the Gaussianâs projection onto the image plane approximately matches the pixel width, independent of scene depth. The assigned scale is only an initial value, with subsequent optimization steps jointly updating all primitives to allow newly added Gaussians to adjust to the existing scene. Our pixel-aligned, depth-aware scaling provides three key benefits: (1) it is independent of the current primitive distribution, avoiding the structural biases of cloning-based methods that replicate and reinforce local geometry, (2) pixel size allows Gaussians to contribute to optimization immediately and efficiently fit fine details while minimizing overlap with existing structure, and (3) its isotropic shape promotes stable multi-view integration, without shapes that the cloned Gaussians inherit.

Primitive Budgeting: We consider two budgeting strategies for controlling the number of Gaussians in the scene. The first enforces a hard upper bound, as in [25], ensuring that densification never exceeds the prescribed budget. This constraint regulates memory and computation while preventing uncontrolled growth of the primitive set. At each densification step, we set the number of sampled pixels $N _ { \mathrm { s a m p l e } }$ so that newly added Gaussians replace those pruned, avoiding excess primitives that would otherwise be discarded under the budget. This is computed as:

$$
N _ { \mathrm { s a m p l e } } = \frac { \operatorname* { m a x } ( 0 . 2 N _ { \mathrm { G S } } , 1 . 2 N _ { \mathrm { l a s t } } ) } { 1 0 0 } ,\tag{16}
$$

<!-- image-->  
Figure 4. Densification overview. Illustration of the proposed error-guided strategy. We render an image with 3DGS, compute the per-pixel $L _ { 1 }$ error, sample pixels proportionally to their error magnitude, and place new Gaussians at the iNGP-predicted depth along the corresponding viewing rays.

where $N _ { \mathrm { G S } }$ is the current total number of Gaussians, and $N _ { \mathrm { l a s t } }$ is the number of primitives inserted in the previous densification step, and the division by 100 reflects the densification interval. This formulation keeps $N _ { \mathrm { G S } }$ close to the budget limit even under aggressive pruning, maintaining consistent scene coverage throughout optimization.

The second strategy adapts the number of primitives to the sceneâs complexity, enabling controlled growth without imposing a fixed upper bound:

$$
N _ { \mathrm { s a m p l e } } = { \frac { \beta N _ { \mathrm { G S } } } { 1 0 0 } } .\tag{17}
$$

Here, $\beta$ controls the growth rate of the primitive set. Smaller values balance the number of Gaussians added with those pruned, maintaining a relatively stable primitive count, whereas larger values yield higher primitive counts, increasing geometric detail at the cost of memory and computing power. With scene initialization at 1M primitives and the application of the opacity penalty, we balance Gaussians added and pruned, unlike [5], which requires a predefined upper limit on primitives.

Opacity-Regularized Pruning: Following [24, 25], Gaussians with opacity below 0.005 are pruned every 100 iterations to remove primitives with negligible contribution to the rendered image. Earlier work has promoted sparsity through different strategies: periodically resetting opacities [24], which can destabilize training [5], reducing opacities by a constant amount after each densification [5], or introducing a post-activation opacity penalty $\mathcal { L } _ { \mathbf { o } } ^ { \mathrm { p o s t } }$ = $\left\| { \boldsymbol { \sigma } } ( \mathbf { o } _ { \mathrm { p r e } } ) \right\| _ { 1 }$ [25], where $\mathbf { o } _ { \mathrm { p r e } }$ denotes the opacity logits before the sigmoid and $\sigma$ is the sigmoid function. This applies the strongest constraint around 0.5 and only a weak penalty near the pruning threshold. In contrast, we employ a pre-activation opacity penalty $\mathcal { L } _ { \mathbf { o } } ^ { \mathrm { p r e } } = \Vert \mathbf { o } _ { \mathrm { p r e } } \Vert .$ 1. It provides a steady constraint across the full opacity range, including very low values, gradually reducing under-contributing primitives. The penalty acts throughout training, and our densification strategy can freely add new primitives, allowing any structure lost through pruning to be recovered more easily than with cloning-based approaches. In all experiments, this loss is scaled by $\lambda _ { \mathbf { o } } = 0 . 0 0 0 2$

## 4.3. Implementation Details

For the iNGP model, we use the proposal-based implementation from NerfAcc [30], trained for 20k iterations with the original setup and architecture. Gaussian optimization runs for 30k iterations, with our densification active for the first 25k. Unlike 3DGS, all SH components are optimized from the start, enabled by the stable initialization that removes the need for gradual SH introduction.

## 5. Evaluation

Dataset and Metrics: We evaluate our method on publicly available scenes from Mip-NeRF360 [3] and OMMO dataset [33], with 01 scene from OMMO resized to have 1600 pixels width. Following [24, 25], we also include the train and truck scene from Tanks & Temples [26], as well as Dr Johnson and playroom from the Deep-Blending [19] dataset. We report PSNR, SSIM [51], and LPIPS [61], with rendering speeds averaged across the full test set. All FPS measurements were recorded on an NVIDIA RTX 2080 Ti, whereas training speeds are reported on an NVIDIA A100, since EDGS requires more memory. These training speeds do not include later-added speed improvements [36].

Baselines: We primarily compare our method against 3DGS [24], itâs extension with iNGP point cloud initialization [14], MCMC [25] using different initialization types (random, SfM, iNGP point clouds), as well as, GaussianPro [9], Perceptual-GS [64], EDGS (with densification) [27], with the densification stopped for all of them if the primitive budget is reached. If the number of primitives at initialization would be higher than the specified budget, the number of primitives is sampled uniformly to fit below it. In the random initialization settings we follow the process described in MCMC [25]. We additionally test on Mini-Splatting2 [11] by matching their final number of primitives instead of a specific budget, due to their method relying on generating a high number of initial Gaussians.

## 5.1. Results

We observe improvements over the baselines across a wide range of specified budgets in Table 1, with plots comparing the most important methods on the budget and no-budget scenario in Figure 6. For a lower limit on Gaussians, we outperform the benchmarks across all datasets and metrics, while providing a competitive reconstruction quality compared to the best performing baselines on the high budget scenarios. In Table 2 we additionally show that even on a high number of primitives of 1M and including the iNGP training, our method provides competitive speed to other methods. The qualitative results in Figure 5 show significant improvement using a wide range of primitive budgets, demonstrating that for the same limit of primitives our method is able to produce a much better reconstruction, especially in areas that are challenging to properly capture on a low budget, such as isolated or high frequency structures. We provide additional qualitative and quantitative results, along with further scene analysis, in the appendix.

<table><tr><td rowspan="2"></td><td colspan="3">Mip-NeRF360 [3]</td><td colspan="3">OMMO [33]</td><td colspan="3">Tanks &amp; Temples [26]</td><td colspan="3">DeepBlending [19]</td></tr><tr><td>PSNR â</td><td>SSIM â â</td><td>LLPIS </td><td>PSNRâ</td><td>SSIMâ â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNRâ</td><td>SSIM â</td><td>LPIPS </td></tr><tr><td colspan="10">Number of Gaussians limited to 100k</td></tr><tr><td>3DGS [24] (SfM init.)</td><td>23.61</td><td>0.693</td><td>0.413</td><td>26.45</td><td>0.820</td><td>0.296</td><td>22.38</td><td>0.774</td><td>0.333</td><td>24.65</td><td>0.827</td><td>0.412</td></tr><tr><td>Foroutan et al. [14]â </td><td>26.64</td><td>0.781</td><td>0.318</td><td>26.89</td><td>0.829</td><td>0.276</td><td>22.48</td><td>0.766</td><td>0.341</td><td>25.31</td><td>0.822</td><td>0.418</td></tr><tr><td>MCMC [25] (rand. init.)</td><td>25.72</td><td>0.730</td><td>0.369</td><td>25.92</td><td>0.808</td><td>0.313</td><td>21.45</td><td>0.750</td><td>0.365</td><td>27.94</td><td>0.859</td><td>0.369</td></tr><tr><td>MCMC [25] (SfM init.)</td><td>27.06</td><td>0.800</td><td>0.303</td><td>27.01</td><td>0.841</td><td>0.266</td><td>22.50</td><td>0.780</td><td>0.332</td><td>28.94</td><td>0.876</td><td>0.333</td></tr><tr><td>MCMC [25] (iNGP init.)</td><td>27.35</td><td>0.797</td><td>0.299</td><td>26.95</td><td>0.837</td><td>0.265</td><td>22.69</td><td>0.775</td><td>0.326</td><td>29.02</td><td>0.872</td><td>0.337</td></tr><tr><td>GaussianPro [9]</td><td>25.57</td><td>0.766</td><td>0.338</td><td>26.14</td><td>0.822</td><td>0.289</td><td>20.59</td><td>0.757</td><td>0.348</td><td>28.15</td><td>0.870</td><td>0.342</td></tr><tr><td>Perceptual-GS [64]</td><td>25.66</td><td>0.774</td><td>0.320</td><td>26.13</td><td>0.819</td><td>0.292</td><td>20.76</td><td>0.759</td><td>00.347</td><td>28.32</td><td>0.874</td><td>0..338</td></tr><tr><td>E EDGS [27]</td><td>227.09</td><td>0.798</td><td>0.296</td><td>26.99</td><td>0.838</td><td>00.261</td><td>22.32</td><td>0.777</td><td>0.324</td><td>28.43</td><td>0.872</td><td>0.337</td></tr><tr><td> Ours</td><td>27.74</td><td>0.809</td><td>0.285</td><td> 27.59</td><td>0.855</td><td>0.243</td><td>23.12</td><td>0.791</td><td>0.310</td><td>29.44</td><td>0.880</td><td>0.328</td></tr><tr><td colspan="10">Number of Gaussians limited to 500k</td></tr><tr><td>3DGS [24] (SfM init.)</td><td>28.22</td><td>0.821</td><td>0.260</td><td>28.96</td><td>0.883</td><td>0.196</td><td>23.54</td><td>0.816</td><td>0.265</td><td>29.25</td><td>0.882</td><td>0.302</td></tr><tr><td>Foroutan et al. [14]â </td><td>28.88</td><td>0.862</td><td>0.204</td><td>28.80</td><td>0.884</td><td>0.185</td><td>23.52</td><td>0.816</td><td>0.257</td><td>29.61</td><td>0.886</td><td>0.297</td></tr><tr><td>MCMC [25] (rand. init.)</td><td>28.38</td><td>0.844</td><td>0.237</td><td>28.23</td><td>0.874</td><td>0.212</td><td>22.96</td><td>0.808</td><td>0.279</td><td>28.84</td><td>0..875</td><td>0.315</td></tr><tr><td>MCMC [25] (SfM init.)</td><td>28.82</td><td>0.861</td><td>0.214</td><td>28.72</td><td>0.885</td><td>0.194</td><td>23.68</td><td>0.825</td><td>0.259</td><td>29.44</td><td>0.887</td><td>0.308</td></tr><tr><td>MCMC [25] (iNGP init.)</td><td>29.02</td><td>0.867</td><td>0.198</td><td>28.75</td><td>0.885</td><td>0.186</td><td>23.57</td><td>0.823</td><td>0.243</td><td>229.61</td><td>0.885</td><td>0.288</td></tr><tr><td>GaussianPro [9]</td><td>28.02</td><td>0.827</td><td>0.253</td><td>28.32</td><td>0.876</td><td>0.206</td><td>22.31</td><td>0.802</td><td>0.286</td><td>29.33</td><td>0.886</td><td>0.301</td></tr><tr><td>Perceptual-GS [64]</td><td>28.66</td><td>0.856</td><td>0.211</td><td>28.42</td><td>0.876</td><td>0.203</td><td>22.77</td><td>0.813</td><td>0.267</td><td>29.44</td><td>0.888</td><td>0.296</td></tr><tr><td> EGS [27]</td><td>28.82</td><td>0.865</td><td>0.193</td><td>29.01</td><td>0.89</td><td>0.168</td><td>23.47</td><td>0.831</td><td>0.229</td><td>29.33</td><td>0.89</td><td>0.286</td></tr><tr><td>Ours</td><td>29.08</td><td>0.870</td><td>0.190</td><td>29.14</td><td>0.892</td><td>0.170</td><td>23.69</td><td>0.829</td><td>0.229</td><td>29..86</td><td>0.891</td><td>0.285</td></tr><tr><td colspan="10">Number Gusis o  MiSplat []</td></tr><tr><td>Mini-Splatting2 [11]</td><td>28.89</td><td>0.875</td><td>0.183</td><td>28.06</td><td>0.875</td><td>0.198</td><td>22.79</td><td>0.823</td><td>0.239</td><td>29.99</td><td>0.898</td><td>0.279</td></tr><tr><td>Ours</td><td>29.26</td><td>0.875</td><td>00.179</td><td>228.90</td><td>0.887</td><td>00.179</td><td>23.66</td><td>0.829</td><td>0.231</td><td>29..82</td><td>0.891</td><td>0.280</td></tr></table>

â  Original code was not publicly available. Our implementation uses iNGP initialization and does not include the additional depth-based loss.

Table 1. Quantitative results with a Gaussian number limit of 100k and 500k. Mini-Splatting2 [11] does not support constraining the number of Gaussians during reconstruction, so we match its Gaussian count. We highlight the best , second best and third best results among methods with the same Gaussian counts. Per-scene metrics for selected methods are provided in the supplementary material.  
EDGS [27]  
MCMC [25]  
Ours  
GT  
<!-- image-->  
Figure 5. Qualitative results comparing our method with MCMC [25] (with SfM point cloud initialization) and EDGS [27] on the Mip-NeRF 360 [3], OMMO [33], and DeepBlending [19] datasets, with varying Gaussian budgets (given in parentheses).

<!-- image-->

Figure 6. PSNR, LPIPS and FPS plots with (left) and without (right) a primitive budget, where counts correspond to Î² values. Averaged across Mip-NeRF360 [3] and OMMO [33]. Numerical results are provided in the appendix.
<table><tr><td></td><td>Ours 10k iters</td><td>Ours 20k iters</td><td>Ours 40k iters</td><td>EDGS [27]</td><td>3DGS [24] (SfM init.)</td><td>MCMC [25] (rand. init.)</td><td>MCMC [25] (SfM init.)</td></tr><tr><td>PSNR â</td><td>29.33</td><td>29.37</td><td>29.37</td><td>29.18</td><td>28.71</td><td>28.98</td><td>29.23</td></tr><tr><td>SSIM</td><td>0.877</td><td>0.880</td><td>0.881</td><td>0.877</td><td>0.846</td><td>0.865</td><td>0.875</td></tr><tr><td>LPIPS </td><td>0.171</td><td>0.168</td><td>0.166</td><td>0.168</td><td>0.222</td><td>0.204</td><td>0.190</td></tr><tr><td>FPS â</td><td>134</td><td>137</td><td>137</td><td>131</td><td>112</td><td>92</td><td>95</td></tr><tr><td>3DGS time â</td><td>20.7</td><td>20.5</td><td>20.5</td><td>19.4</td><td>22.6</td><td>26.3</td><td>25.1</td></tr><tr><td>Init. time â</td><td>1.6</td><td>3.1</td><td>6.1</td><td>2.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Overall time â</td><td>22.3</td><td>23.6</td><td>26.6</td><td>21.7</td><td>22.6</td><td>26.3</td><td>25.1</td></tr></table>

Table 2. Quantitative results averaged over Mip-NeRF360 [3] showing reconstruction timings for 3DGS [24], MCMC [25], EDGS [27], and our method (with varying iNGP durations), capped at 1M Gaussians. 3DGS time reports target scene optimization and densification, while init. time shows iNGP reconstruction for our method and initial matching for EDGS [27].

## 5.2. Ablations

We analyze each componentâs impact through ablation studies in Table 3. (a), (b) Longer iNGP reconstruction only slightly improves reconstruction quality. (c) continuing training the iNGP model also during optimization, (d) initializing Gaussians with the ground truth pixel color, or (e) predicting spherical harmonics with iNGP, leads to marginally worse results. Demonstrating the strength of our densification method, using (f) pixel-cone-sized primitives during initialization, or even not using any initialization (g), results in worse PSNR but maintains low LPIPS and considerably improves rendering speed, thanks to less overlap between primitives, reducing blending. (h) Sampling pixels uniformly instead of guiding Gaussian creation using the $L _ { 1 }$ loss from the training set produces lower reconstruction quality, although due to uniform sampling in image space still focusing more on parts of the scene seen most across views, the drop is not drastic. Densifying with

<table><tr><td>Ablation</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>FPS â</td></tr><tr><td>Ours</td><td>27.74</td><td>0.810</td><td>0.285</td><td>328</td></tr><tr><td>(a) 10k iNGP iter. (b) 40k iNGP iter. (c) Train iNGP during 3DGS</td><td>27.74 27.72 27.73</td><td>0.808 0.811 0.809</td><td>0.287 0.284 0.285</td><td>313 333</td></tr><tr><td>(d) Color from GT image (e) Prediction of SH with iNGP (f) Cone-sized initialization</td><td>27.73 27.74 27.38</td><td>0.807 0.805 0.806</td><td>0.287 0.290 0.287</td><td>310 320 320</td></tr><tr><td>(g) Without initialization (h) Uniform image-space sampling (i) Densify with 3DGS depth</td><td>27.46 27.51 27.43</td><td>0.811 0.806 0.797</td><td>0.285 0.286 0.296</td><td>437 415 307 332</td></tr><tr><td>() SfM initialization + 3DGS depth dens. (k) Densify with k-NN scaling (1) No opacity penalty (m) Post-densification opacity decrease [5]</td><td>27.15 27.54 27.31 27.46</td><td>0.790 0.802 0.794 0.798</td><td>0.302 0.295 0.301 0.297</td><td>329 299 294</td></tr><tr><td>(n) MCMC-style opacity penalty [25] (0) Î»scale = 1 (p) Î»scale = 4</td><td>27.49 27.70 27.63</td><td>0.803 0.810</td><td>0.293 0.285</td><td>256 239 329</td></tr></table>

Table 3. Ablation study of our method with 100k Gaussians, averaged over the Mip-NeRF360 [3] dataset. We highlight the best , second best , and third best results.

3DGS depth (i), also without using iNGP even for initialization (j), strongly affects the results. Similarly, k-NN sizing of newly added primitives based on their closest neighbors (k), or changing the opacity penalty (l), (m), (n), has a large effect on reconstruction quality, reinforcing the benefits of our densification approach. (o), (p) Altering Gaussian sizes from their default pixel-width cone size during densification slightly reduces quality, yet the small difference suggests Gaussians quickly resize to fit the scene.

## 6. Conclusion

We introduce ConeGS, a reconstruction pipeline replacing cloning-based densification with a method guided by photometric error and a coarse iNGP proxy, where new primitives are sized by pixel cones. Together with an improved opacity penalty, this allows creating primitives independently of existing structures through more flexible exploration. ConeGS consistently improves reconstruction quality and rendering performance across Gaussian budgets, with strong gains under tight primitive constraints. It achieves up to 0.6 PSNR increase and 20% speedup over cloning-based baselines.

Limitations: ConeGS performs well on standard scenes but may struggle with large-scale environments, inaccurate poses, or sparse views, occasionally creating floaters (see appendix). Benefits are also reduced at high Gaussian budgets, where dense coverage limits the impact of errorguided placement, primarily offering faster rendering.

## Acknowledgments

Stefano Esposito acknowledges travel support from the European Unionâs Horizon 2020 research and innovation program under ELISE Grant Agreement No. 951847.

## References

[1] Mohamed Abdul Gafoor, Marius Preda, and Titus Zaharia. Refining gaussian splatting: A volumetric densification approach. In Computer Science Research Notes. University of West Bohemia, Czech Republic, 2025. 3

[2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. ICCV, 2021. 2, 3, 15

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022. 1, 2, 6, 7, 8, 12, 13, 14, 15, 17, 18

[4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. ICCV, 2023. 2, 15

[5] Samuel Rota Bulo, Lorenzo Porzi, and Peter Kontschieder. \` Revising densification in gaussian splatting. ArXiv, abs/2404.06109, 2024. 3, 6, 8

[6] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, 2024. 3

[7] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In ECCV, 2022. 2

[8] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. arXiv preprint arXiv:2403.14627, 2024. 3

[9] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin, Yuexin Ma, Wenping Wang, and Xuejin Chen. GaussianPro: 3d gaussian splatting with progressive propagation. In International Conference on Machine Learning (ICML), 2024. 3, 6, 7, 13

[10] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps, 2023. 3

[11] Guangchi Fang and Bing Wang. Mini-splatting2: Building 360 scenes within minutes via aggressive gaussian densification. ArXiv, abs/2411.12788, 2024. 3, 6, 7, 12, 13

[12] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians, 2024. 3

[13] Shuangkang Fang, I-Chao Shen, Takeo Igarashi, Yufeng Wang, ZeSheng Wang, Yi Yang, Wenrui Ding, and Shuchang Zhou. Nerf is a valuable assistant for 3d gaussian splatting, 2025. 3

[14] Yalda Foroutan, Daniel Rebain, Kwang Moo Yi, and Andrea Tagliasacchi. Evaluating alternatives to sfm point cloud initialization for gaussian splatting. 2024. 3, 6, 7, 13

[15] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. Eagles: Efficient accelerated 3d gaussians with lightweight encodings, 2024. 3

[16] Antoine Guedon and Vincent Lepetit. Sugar: Surface- Â´ aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. CVPR, 2024. 3

[17] Zhihao Guo, Jingxuan Su, Shenglin Wang, Jinlong Fan, Jing Zhang, Li Hong Han, and Peng Wang. Gp-gs: Gaussian processes for enhanced gaussian splatting. ArXiv, abs/2502.02283, 2025. 3

[18] Abdullah Hamdi, Luke Melas-Kyriazi, Jinjie Mai, Guocheng Qian, Ruoshi Liu, Carl Vondrick, Bernard Ghanem, and Andrea Vedaldi. Ges : Generalized exponential splatting for efficient radiance field rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19812â19822, 2024. 3

[19] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. Deep blending for free-viewpoint image-based rendering. 2018. 6, 7, 13, 14, 17, 18

[20] Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi, Bernard Ghanem, and Marc Van Droogenbroeck. 3D convex splatting: Radiance field rendering with 3D smooth convexes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 3

[21] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. 2024. 3

[22] Binxiao Huang, Zhengwu Liu, and Ngai Wong. Decomposing densification in gaussian splatting for faster 3d scene reconstruction, 2025. 3

[23] Hanqing Jiang, Xiaojun Xiang, Han Sun, Hongjie Li, Liyang Zhou, Xiaoyu Zhang, and Guofeng Zhang. Geotexdensifier: Geometry-texture-aware densification for high-quality photorealistic 3d gaussian splatting, 2024. 3

[24] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 2, 3, 6, 7, 8, 13

[25] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splatting as markov chain monte carlo. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 1, 3, 5, 6, 7, 8, 12, 13, 14

[26] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017. 1, 6, 7, 12, 13, 14, 17, 18

[27] Dmytro Kotovenko, Olga Grebenkova, and Bjorn Ommer. Â¨ EDGS: eliminating densification for efficient convergence of 3dgs. arXiv, 2504.13204, 2025. 1, 3, 6, 7, 8, 12, 13, 15

[28] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. arXiv preprint arXiv:2403.06912, 2024. 3

[29] Mingrui Li, Shuhong Liu, Tianchen Deng, and Hongyu Wang. Densesplat: Densifying gaussian splatting slam with neural radiance prior, 2025. 3

[30] Ruilong Li, Hang Gao, Matthew Tancik, and Angjoo Kanazawa. Nerfacc: Efficient sampling accelerates nerfs. arXiv preprint arXiv:2305.04966, 2023. 6

[31] Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and Andrew Feng. Deformable beta splatting, 2025. 3

[32] Yueh-Cheng Liu, Lukas Hollein, Matthias NieÃner, and Â¨ Angela Dai. Quicksplat: Fast 3d surface reconstruction via learned gaussian initialization. ArXiv, abs/2505.05591, 2025. 3

[33] Chongshan Lu, Fukun Yin, Xin Chen, Tao Chen, Gang YU, and Jiayuan Fan. A large-scale outdoor multi-modal dataset and benchmark for novel view synthesis and implicit scene reconstruction, 2023. 1, 6, 7, 8, 12, 13, 14, 15, 17, 18

[34] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In CVPR, 2024. 2, 3

[35] Alexander Mai, Peter Hedman, George Kopanas, Dor Verbin, David Futschik, Qiangeng Xu, Falko Kuester, Jon Barron, and Yinda Zhang. Ever: Exact volumetric ellipsoid rendering for real-time view synthesis, 2024. 3

[36] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3dgs: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers, New York, NY, USA, 2024. Association for Computing Machinery. 3, 6

[37] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1, 2, 3

[38] Mahmud A. Mohamad, Gamal Elghazaly, Arthur Hubert, and Raphael Frank. Denser: 3d gaussians splatting for scene reconstruction of dynamic urban environments, 2024. 3

[39] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 2022. 2, 4

[40] Seungtae Nam, Xiangyu Sun, Gyeongjin Kang, Younggeun Lee, Seungjun Oh, and Eunbyung Park. Generative densification: Learning to densify gaussians for high-fidelity generalizable 3d reconstruction. arXiv preprint arXiv:2412.06234, 2024. 3

[41] Simon Niedermayr, Josef Stumpfegger, and Rudiger West- Â¨ ermann. Compressed 3d gaussian splatting for accelerated novel view synthesis. In CVPR, 2024. 3

[42] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari. Radsplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+ fps. arXiv.org, 2024. 3, 4

[43] Lukas Radl, Michael Steiner, Mathias Parger, Alexander Weinrauch, Bernhard Kerbl, and Markus Steinberger. StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering. ACM Transactions on Graphics, 4 (43), 2024. 3

[44] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. IEEE transactions on pattern analysis and machine intelligence, PP, 2024. 3

[45] Sara Fridovich-Keil and Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2022. 2

[46] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In CVPR, 2022. 2

[47] Chinmay Talegaonkar, Yash Belhe, Ravi Ramamoorthi, and Nicholas Antipa. Volumetrically consistent 3d gaussian rasterization, 2025. 3

[48] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023. 2

[49] Nicolas von Lutzow and Matthias NieÃner. Linprim: Lin- Â¨ ear primitives for differentiable volumetric rendering. ArXiv, abs/2501.16312, 2025. 2, 3

[50] Zipeng Wang and Dan Xu. Pygs: Large-scale scene representation with pyramidal 3d gaussian splatting, 2024. 3

[51] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 2004. 3, 6

[52] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, 2024. 2

[53] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee. Multi-scale 3d gaussian splatting for anti-aliased rendering. In CVPR, 2024. 2

[54] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint arXiv:2309.13101, 2023. 2

[55] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou. AbsGS: Recovering fine details in 3d gaussian splatting. In ACM MM, 2024. 3

[56] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. PlenOctrees for real-time rendering of neural radiance fields. In ICCV, 2021. 2

[57] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. CVPR, 2024. 2, 3

[58] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient high-quality compact surface reconstruction in unbounded scenes. arXiv:2404.10772, 2024. 3

[59] Zhaojie Zeng, Yuesong Wang, Lili Ju, and Tao Guan. Frequency-aware density control via reparameterization for high-quality rendering of 3d gaussian splatting. ArXiv, abs/2503.07000, 2025. 3

[60] Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and Eric P. Xing. Fregs: 3d gaussian splatting with progressive frequency regularization. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21424â21433, 2024. 3

[61] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 2018. 6

[62] Xin Zhang, Anpei Chen, Jincheng Xiong, Pinxuan Dai, Yujun Shen, and Weiwei Xu. Neural shell texture splatting: More details and fewer primitives. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 3

[63] Yangming Zhang, Wenqi Jia, Wei Niu, and Miao Yin. Gaussianspa: An âoptimizing-sparsifyingâ simplification framework for compact and high-quality 3d gaussian splatting. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26673â26682, 2024. 3

[64] Hongbi Zhou and Zhangkai Ni. Perceptual-gs: Sceneadaptive perceptual densification for gaussian splatting. ArXiv, abs/2506.12400, 2025. 3, 6, 7, 13

[65] Zheng Zhou, Yu-Jie Xiong, Chun-Ming Xia, Jia-Chen Zhang, and Hong-Jian Zhan. Gradient-direction-aware density control for 3d gaussian splatting. 2025. 3

[66] Yuanchen Guo Yangguang Li Ding Liang Yanpei Cao Songhai Zhang Zixin Zou, Zhipeng Yu. Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. arXiv preprint arXiv:2312.09147, 2023. 2

[67] M. Zwicker, H. Pfister, J. van Baar, and M. Gross. Ewa volume splatting. In Proceedings Visualization, 2001. VIS â01., 2001. 3

## Appendix

This appendix introduces additional results (Section A1) and ablations (Section A2). We then discuss our reconstructed scene structure (Section A3), possible failure cases (Section A4), and provide visualizations for the different types of initializations mentioned in the main paper (Section A5). Finally, we discuss the parallels between 3DGS and NeRF rendering, which allows training a radiance field using the rendering from 3DGS (Section A6).

## A1. Detailed results

Table A1 and Table A2 present results for the no-budget scenario, which were used for plots in Figure 6. We use various values of the $\beta$ parameter without specifying a budget for our method, except for the last cell, which uses the budget set by the number of Gaussians generated by 3D Gaussian Splatting. For each cell, we also compare MCMC [25] and EDGS [27] where they are set to match the number of Gaussians produced in each of the cells. The comparisons show that our method is able to produce high-quality results even without specifying a budget, instead adjusting the number of primitives based on the scene complexity, while still remaining sparse in the number of primitives. Notably, even when setting $\beta = 0 ,$ , which effectively disables densification, our method still performs well due to a dense initialization and effective filtering of unnecessary primitives enforced by the opacity penalty, consistent with

<table><tr><td>Ablation</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FPS â</td><td># Gaussians</td></tr><tr><td>Ours (Î² = 0) MCMC (SfM)</td><td>29.12</td><td>0.873</td><td>0.183</td><td>185</td><td>542k</td></tr><tr><td>EDGS</td><td>28.86 28.87</td><td>0.865 0.868</td><td>0.209 0.187</td><td>136</td><td></td></tr><tr><td>Ours (Î² = 0.01)</td><td></td><td></td><td></td><td>194</td><td></td></tr><tr><td>MCMC (SfM)</td><td>29.25 29.06</td><td>0.877</td><td>0.174</td><td>169</td><td>674k</td></tr><tr><td>EDGS</td><td>28.97</td><td>0.870</td><td>0.199</td><td>119</td><td></td></tr><tr><td>Ours (Î² = 0.02)</td><td></td><td>0.873</td><td>0.178</td><td>167</td><td></td></tr><tr><td>MCMC (SfM)</td><td>29.34</td><td>0.880</td><td>0.166</td><td>146</td><td>942k</td></tr><tr><td>EDGS</td><td>29.15</td><td>0.876</td><td>0.189</td><td>101</td><td></td></tr><tr><td></td><td>29.10</td><td>0.878</td><td>0.167</td><td>134</td><td></td></tr><tr><td>Ours (Î² = 0.04)</td><td>29.38</td><td>0.881</td><td>0.161</td><td>105</td><td>1.66M</td></tr><tr><td>MCMC (SfM) EDGS</td><td>29.32</td><td>0.882</td><td>0.174</td><td>77</td><td></td></tr><tr><td></td><td>29.19</td><td>0.881</td><td>0.159</td><td>102</td><td></td></tr><tr><td>Ours</td><td>29.35</td><td>0.879</td><td>0.159</td><td>80</td><td></td></tr><tr><td>MCMC (SfM)</td><td>29.56</td><td>0.887</td><td>0.164</td><td>55</td><td>2.57M</td></tr><tr><td>EDGS</td><td>29.37</td><td>0.884</td><td>0.153</td><td>67</td><td></td></tr><tr><td>3DGS</td><td>29.03</td><td>0.870</td><td>0.184</td><td>69</td><td></td></tr></table>

Table A1. No-budget Mip-NeRF360. Comparison of reconstruction quality without a fixed limit on the number of primitives. For each cell, our method is run with a different Î² parameter, which determines the number of Gaussians generated, and the other methods are limited to this number. In the last cell, the number of Gaussians is set to match the amount produced by 3DGS, with all other methods constrained accordingly. Results are averaged over the Mip-NeRF360 [3] dataset.

<table><tr><td>Ablation</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FPS â</td><td># Gaussians</td></tr><tr><td rowspan="2">Ours (Î² = 0) MCMC (SfM)</td><td>28.54</td><td>0.879</td><td>0.195</td><td>253</td><td rowspan="2">352k</td></tr><tr><td>28.37</td><td>0.877</td><td>0.207</td><td>199</td></tr><tr><td rowspan="2">EDGS Ours (Î² = 0.01)</td><td>28.62</td><td>0.885</td><td>0.183</td><td>219</td><td rowspan="2"></td></tr><tr><td>29.02 28.55</td><td>0.890</td><td>0.176</td><td>237</td></tr><tr><td rowspan="2">MCMC (SfM) EDGS</td><td>28.84</td><td>0.882 0.890</td><td>0.199 0.173</td><td>175 211</td><td rowspan="2"></td></tr><tr><td>29.19</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Ours (Î² = 0.02) MCMC (SfM) EDGS</td><td>28.79</td><td>0.894 0.888</td><td>0.166 0.190</td><td>217 157</td><td rowspan="2">581k</td></tr><tr><td>29.07</td><td>0.896</td><td>0.163</td><td>188</td></tr><tr><td rowspan="2">Ours (Î² = 0.04) MCMC (SfM)</td><td>29.32</td><td>0.898</td><td>0.157</td><td>116</td><td rowspan="2">927k</td></tr><tr><td>29.05</td><td>0.895</td><td>0.177</td><td>105</td></tr><tr><td rowspan="2">EDGS Ours</td><td>29.35</td><td>0.902</td><td>0.151</td><td>134</td><td rowspan="2"></td></tr><tr><td>29.60</td><td>0.904</td><td>0.144</td><td>114</td></tr><tr><td rowspan="2">MCMC EDGS</td><td>29.46</td><td>0.904</td><td>0.157</td><td>89</td><td rowspan="2">1.75M</td></tr><tr><td>29.85</td><td>0.909</td><td>0.137</td><td>99</td></tr><tr><td>3DGS</td><td>29.30</td><td>0.896</td><td>0.171</td><td>88</td><td></td></tr></table>

Table A2. No-budget OMMO. Comparison of reconstruction quality without a fixed limit on the number of primitives. For each cell, our method is run with a different $\beta$ parameter, which determines the number of Gaussians generated, and the other methods are limited to this number. In the last cell, the number of Gaussians is set to match the amount produced by 3DGS, with all other methods constrained accordingly. Results are averaged over the OMMO [33] dataset.

<!-- image-->

<!-- image-->

<!-- image-->  
Ours

<!-- image-->  
Mini-Splatting2 [11]  
Figure A1. Qualitative comparison between our approach and Mini-Splatting2 [11] on the garden scene from Mip-NeRF360 [3] and on the train scene from Tanks & Temples [26].

findings in [27]. However, this configuration does not allow explicit control over the number of primitives and may limit the achievable reconstruction quality.

If an even lower number of Gaussians is desired while still using a no-budget scenario, the number of initialized Gaussians can be reduced, effectively lowering the number of primitives generated at the end.

We show in Table A3 the results on the selected benchmarks for the budget of 1M primitives. Additionally, we present per-scene results for 100k Gaussians in Table A8,

<table><tr><td rowspan="2"></td><td colspan="3">Mip-NeRF360 [3]</td><td colspan="3">OMMO [33]</td><td colspan="3">Tanks &amp; Temples [26]</td><td colspan="3">DeepBlending [19]</td></tr><tr><td>PSNR â</td><td>SSIMâ</td><td>PIPS </td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIS </td><td>PSNR â</td><td>SSIM â</td><td>LLPIPS </td></tr><tr><td colspan="10">Number of Gaussians limited to 1M</td><td></td></tr><tr><td>3DGS [24] (SfM init.)</td><td>28.71</td><td>0.846</td><td>0.222</td><td>29.40</td><td>0.894</td><td>0.178</td><td>23.60</td><td>0.826</td><td>0.245</td><td>28.93</td><td>0.881</td><td>0.292</td></tr><tr><td>Foroutan et al. [14]</td><td>29.22</td><td>0.876</td><td>0.177</td><td>29.32</td><td>0.896</td><td>0.163</td><td>23.75</td><td>0.829</td><td>0..230</td><td>29.60</td><td>0.886</td><td>0.282</td></tr><tr><td>MCMC [25] (rand. init.)</td><td>28.98</td><td>0.865</td><td>0.204</td><td>28.83</td><td>0.888</td><td>0.185</td><td>23.41</td><td>0.824</td><td>0.249</td><td>28.94</td><td>0.877</td><td>0.303</td></tr><tr><td>MCMC [25] (SfM init.)</td><td>29.23</td><td>0.875</td><td>0.190</td><td>29.18</td><td>0.895</td><td>0.174</td><td>23.93</td><td>0.836</td><td>0.233</td><td>29.67</td><td>0.887</td><td>0.288</td></tr><tr><td>MCMC [25] (iNGP init.)</td><td>29.32</td><td>0.879</td><td>0.173</td><td>28.76</td><td>0.887</td><td>0.181</td><td>23.37</td><td>0.804</td><td>0.268</td><td>29.87</td><td>0.892</td><td>0.276</td></tr><tr><td>GaussianPro [9]</td><td>28.56</td><td>0.845</td><td>0.226</td><td>28.79</td><td>0.885</td><td>0.189</td><td>23.01</td><td>0.818</td><td>0.256</td><td>29.27</td><td>0.887</td><td>0..291</td></tr><tr><td>Perceptual-GS [64]</td><td>29.27</td><td>0.876</td><td>0.176</td><td>29.00</td><td>0.888</td><td>0.179</td><td>23.26</td><td>0.828</td><td>0.240</td><td>29.41</td><td>0.889</td><td>0.286</td></tr><tr><td> ES [27]</td><td>29.18</td><td>0.87</td><td>0..168</td><td>29.58</td><td>0.9</td><td>0..149</td><td>23.66</td><td>0.8</td><td>0.20</td><td>29.43</td><td>0..889</td><td>0.271</td></tr><tr><td>Ours</td><td>229.37</td><td>0..880</td><td>0.168</td><td>29.44</td><td>0.899</td><td>0.154</td><td>23.70</td><td>0.835</td><td>0.207</td><td>29.6</td><td>0..889</td><td>0.273</td></tr></table>

â  Original code was not publicly available. Our implementation uses iNGP initialization and does not include the additional depth-based loss.

Table A3. Quantitative results with a Gaussian number limit of 1M. We highlight the best , second best and third best results among methods with comparable numbers of Gaussians.
<table><tr><td></td><td>Ours</td><td>3DGS (SfM)</td><td>MCMC (SfM)</td><td>EDGS</td><td>Perceptual-GS</td></tr><tr><td># Memory (MiB)</td><td>9545</td><td>9049</td><td>8671</td><td>14517</td><td>12403</td></tr></table>

Table A4. Peak GPU memory usage on the 15 scene from the OMMO dataset [33] on the maximum budget of 500k primitives. We highlight the best , second best and third best results among all.

<table><tr><td></td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FPS â</td><td>Train (min) â</td></tr><tr><td>ConeGS</td><td>29.14</td><td>0.892</td><td>0.170</td><td>217</td><td>25</td></tr><tr><td>ConeGS (iNGP retrain)</td><td>29.27</td><td>0.893</td><td>0.168</td><td>212</td><td>29</td></tr></table>

Table A5. Ablation on additional iNGP training on the OMMO [33] dataset with 500k primitives, evaluating the effect of continuing training iNGP in parallel with the full 3DGS optimization. We highlight the best , second best and third best results among all.

500k Gaussians in Table A9, 1M Gaussians in Table A10, and 2M Gaussians in Table A11. We also show additional qualitative results with Mini-Splatting2 [11] in Figure A1.

## A2. Additional ablations and comparisons

We evaluate the peak GPU memory usage during optimization for several methods in Table A4. The results show that our method is very close to 3DGS [24] and MCMC [25] in terms of memory consumption, while remaining considerably lower than EDGS [27] and Perceptual-GS [64]. This efficiency allows our method to run on a wider range of GPUs, making it more accessible to devices with limited memory.

We expand on ablation (c) from the main paper, where iNGP continues training in parallel with the full 3DGS optimization. Since iNGP training, like densification, is guided by the L1 error from 3DGS, the iNGP model can better focus on regions where 3DGS reconstruction may fall short, potentially improving densification. The original ablation was tested on a budget of 100k primitives, which may not fully reveal this effect. To explore further, we run experiments with larger budgets. On the 500k budget for challenging OMMO [33] scenes (Table A5), we observe additional performance gains, though at the cost of longer training time. We also test this setup with a budget of 1M primitives across all datasets (Table A6), finding minimal improvements on difficult scenes and slight decreases on Mip-NeRF360, which may be caused by overfitting to certain areas.

<!-- image-->  
Figure A2. Number of primitives pruned, added, accumulated, as well as the total number, during each densification, on different budget scenarios. Results obtained from the garden scene from Mip-NeRF360 [3].

Figure A2 illustrates the number of primitives added and removed during each densification and pruning step. It also tracks the accumulation buffer of primitives. When no budget is specified, all accumulated primitives are added to the scene (see Eq. 17). Under a fixed budget, however, not all of them are used. This is because accumulation happens every iteration and must remain available for pruning and densification, while the exact number that can be added is unknown beforehand. As a result, the system accumulates more primitives than are usually required to keep the total count close to the budget after pruning (see Eq. 16).

<table><tr><td></td><td colspan="3">Mip-NeRF360 [3]</td><td colspan="3">OMMO [33]</td><td colspan="3">Tanks &amp; Temples [26] SSIMâ</td><td colspan="3">DeepBlending [19]</td></tr><tr><td>PSNR â</td><td>SSIMâ</td><td></td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNRâ</td><td></td><td>LPIPS â</td><td></td><td>PSNR âSSIMâ</td><td>LPIPS â</td></tr><tr><td colspan="9">Number of Gaussians limited to 1M</td><td colspan="3"></td></tr><tr><td>Ours</td><td>29.37</td><td>0.880</td><td>0.168</td><td>29.44</td><td>0.899</td><td>0.154</td><td>23.70</td><td>0.835</td><td>0.207</td><td>29.69 29.73</td><td>0.889</td><td>0.273</td></tr><tr><td>Ours, iNGP training during 3DGS</td><td>29.25</td><td>0.879</td><td>0.168</td><td>29.54</td><td>0.900</td><td>0.152</td><td>23.72</td><td>0.836 0.207</td><td></td><td>0.889</td><td></td><td>0.273</td></tr></table>

Table A6. Ablation on additional iNGP training for the budget of 1M primitives. We highlight the best, second best and third best results among methods with comparable numbers of Gaussians.

<!-- image-->  
Figure A3. The histogram of perceived image-space size of pixel-sized Gaussians rendered from different viewpoints. Size expressed in pixel widths. Analysis doesnât include the low-pass filter from 3DGS rasterization.

<table><tr><td></td><td>Ours</td><td>MCMC (SfM)</td><td>MCMC</td><td>3DGS (SfM)</td></tr><tr><td># Gaussians per pixel</td><td>30.72</td><td>49.55</td><td>48.45</td><td>34.74</td></tr></table>

Table A7. Blending analysis. Mean number of Gaussians contributing to alpha blending per pixel, averaged across the Mip-NeRF360 [3] dataset using a budget of 1M Gaussians. We highlight the best , second best and third best results among all.

The primitive size is defined to be approximately one pixel from a single viewpoint. From other viewpoints, this size is not exactly one pixel, but should remain close. To confirm this, we measure the size of newly added pixelsized primitives from multiple viewpoints and report the results in Figure A3. The distribution shows that the apparent size remains near one pixel, with very few cases above four pixels or below 0.2 pixels.

## A3. Scene structure

To confirm that our method produces reconstructions with desirable characteristics, such as a balanced distribution of scaling values and placement close to surfaces, which are often important for downstream tasks, we analyze the scenes created with our method in comparison to MCMC [25].

<!-- image-->

<!-- image-->  
Figure A4. Histograms of the Gaussian scaling values for our method and MCMC with SfM initialization on 2M Gaussians. The red lines indicate the minimum scaling required for a Gaussian to cover at least one pixel, disregarding the low-pass filter.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Ours

<!-- image-->  
MCMC  
Figure A5. Shrunk Gaussians. Visual comparison between the proposed method and MCMC, rendered using Gaussians scaled to half their original size. Both methods were trained with a limit of 1 million primitives on the bicycle and bonsai scenes from Mip-NeRF360 [3], and the 10 scene from OMMO [33].

Although Gaussian Splatting with the MCMC densification strategy explores the scene effectively, it also has clear shortcomings. Its cloning strategy and scaling penalty often cause Gaussians to shrink strongly along certain dimensions. The low-pass filter used by 3DGS renderer can hide this effect visually, but it prevents Gaussians from expanding in those directions and interferes with tasks that depend on accurate scales, such as MCMC position error. As shown in Figure A4, many of its Gaussians shrink below the pixel radius, in contrast to the more balanced distribution produced by our method.

Figure A5 shows that our approach produces Gaussians that are more uniform in size and placed closer to object surfaces. This avoids an overreliance on oversized background primitives located far from the surface. This primitive distribution not only improves geometric alignment but also increases rendering speed by reducing blending and sorting overhead during rasterization. To support this hypothesis, Table A7 reports the mean number of Gaussians blended per pixel. Our method consistently requires less blending compared to the selected benchmarks. The difference is especially large compared to MCMC, which blends over 60% more Gaussians per pixel on average.

## A4. Failure cases

While our method generally produces strong reconstructions on almost all tested scenes, its reliance on iNGP can make it susceptible to floaters in challenging scenarios such as noisy poses, very large or sparse-view scenes, or other cases where reliable iNGP reconstruction is difficult. One such example is shown in Figure A6, where the scene contains distortions and lacks sufficient viewpoint coverage near the cameras. This leads to spurious high-density regions in iNGP close to the cameras, which in turn degrades densification quality by placing Gaussians in incorrect regions. Although this can reduce performance, the method still produces high-quality reconstructions overall (see perscene results).

## A5. Initialization visualizations

The choice of initialization has a strong impact on the performance of 3DGS reconstruction. A sufficiently good initialization can even remove the need for additional densification ([27], Table A1, Table A2). We visualize different initialization types in Figure A7. Initialization from a sparse SfM point cloud often leaves large gaps that are difficult to fill correctly, while our initialization produces a more uniform coverage. Another important strategy is initialization based on pixel width. This approach does not provide a useful inductive bias of larger primitives and can lead to more gaps in unobserved regions, although it may offer faster rendering due to reduced blending (Section 5.2). Finally, our formulation allows scaling the initial primitives in image space, producing a balance between these two types of initialization.

GT  
ConeGS  
<!-- image-->  
iNGP  
Figure A6. Depth maps. Depth maps for ConeGS and the iNGP reconstruction model on the 01 scene from the OMMO [33] dataset. Since the Gaussians are generated from an iNGP reconstruction, the presence of floaters in it leads to floaters also appearing in the Gaussian Splatting results.

SfM point cloud init.  
Our init.  
<!-- image-->  
Figure A7. Initialization Methods Comparison. Visual comparison of four initialization methods for 3DGS reconstruction on the bicycle scene from Mip-NeRF 360 dataset [3]: (1) SfM point cloud initialization, (2) iNGP initialization with kNN-based sizing, (3) iNGP initialization with 1Ã sizing, and (4) iNGP initialization using the smaller of 10Ã pixel size or kNN-based sizing.

## A6. Gaussian-Based Radiance Field training

To establish a more direct connection between the volumetric ray marching procedure in Neural Radiance Fields (NeRF) and the rendering in 3D Gaussian Splatting (3DGS), the set of conical frustums sampled along a cone in Mip-NeRF and its derivatives [2â4] can be reinterpreted as a set of 3D Gaussian primitives, covering approximately the same area. The purpose of this reformulation is to enable training a neural radiance field model using only the 3DGS renderer, without using the traditional volumetric ray integration.

<!-- image-->  
Figure A8. Training equivalence. Illustration of the equivalence between training an implicit radiance field model using NeRFstyle ray marching and 3D Gaussian Splatting rasterization.

To align the 3DGS and NeRF rendering formulations, it is necessary to demonstrate that an entire ray can be equivalently represented as a sequence of Gaussians such that, when rendered, the result is equal to the volumetric integration process. This is achieved by casting cones along pixel directions and subdividing each into a set of conical frustums. Each frustum is then mapped to a Gaussian primitive, where the midpoint

$$
t _ { \mu , i } = \frac { t _ { i } + t _ { i + 1 } } { 2 }\tag{A1}
$$

is used to define the Gaussian center $\mathbf { p } _ { i }$ . The viewdependent RGB color $\mathbf { c } _ { i }$ and scalar density $\sigma _ { i }$ , which is converted to opacity $o _ { i }$ using Eq. 6, are predicted by a neural network. As detailed in Section 4.1, the predicted RGB color $\mathbf { c } _ { i }$ is encoded as spherical harmonics coefficients, which are compatible with the 3DGS rasterizer.

Since the Gaussians lie along the cone and their centers are co-linear with the pixel center and the camera origin, their projected 2D means fall directly on the target pixel. Consequently, the maximum opacity contribution aligns with the pixel center, and the kernel evaluation satisfies

$$
K ( { \bf p } _ { c } , { \mu } _ { i } ^ { \mathrm { 2 D } } , \Sigma _ { i } ^ { \mathrm { 2 D } } ) = 1 .\tag{A2}
$$

This condition ensures that the Gaussian opacity $o _ { i }$ corresponds directly to the opacity $\alpha _ { i }$ used in blending operations, as defined in $\operatorname { E q . }$ 6 and $\operatorname { E q . }$ . The opacity is therefore determined by the predicted density and the length of the corresponding frustum.

To maintain consistent 2D projection footprints across different depths, each Gaussianâs 3D covariance must be adjusted based on its location along the cone. Because elongating Gaussians along the ray direction does not affect the resulting 2D projection, the scaling can be derived using the pixel footprint size and set isotropically using the formulation in Eq. 15, while the quaternion can be set to the identity quaternion.

Given these assignments for position, color, opacity, scale, and orientation, each Gaussian can be rendered using the 3DGS renderer to produce the same pixel color as that produced by volumetric rendering. Assuming no lowpass filtering is applied and that only a single image is rendered at a time, it becomes possible to train an Instant-NGPstyle model using only the 3DGS rendering pipeline. Although this approach is marginally slower than traditional ray marching, it yields equivalent radiance field representations. A visual comparison of both approaches is shown in Figure A8.

This reinterpretation also provides a more principled foundation for the proposed densification strategy. Instead of using depth, the strategy can be understood as selecting a Gaussian that corresponds to a surface-level conical frustum of pixels with high photometric error. This establishes a clearer theoretical link between cone-based densification and neural implicit models such as NeRF.

<table><tr><td rowspan=1 colspan=3>Scene</td><td rowspan=1 colspan=2>Ours                  3DGS (SfM)              MCMC (SfM)                 EDGS</td></tr><tr><td rowspan=1 colspan=3>Bicycle</td><td rowspan=2 colspan=2>24.05/0.6520.379/368 20.99 / 0.470 / 0.543 / 466 23.34 / 0.630 / 0.400 / 339 23.68/0.635/0.393/44424.69//0.709/0.336/431 22.87 / 0.589 / 0.445 / 430 24.51/0.710I0.350/569 24.47/0.706/â²0.341493</td></tr><tr><td rowspan=1 colspan=3>Garden</td></tr><tr><td rowspan=1 colspan=3>Stump</td><td rowspan=1 colspan=1>25.710.70710.351/456 22.79 / 0.547 / 0.498 /506 25.16/0.680</td><td rowspan=1 colspan=1>0.370/ 402 25.08/0.67330.377/588</td></tr><tr><td rowspan=2 colspan=3>RoomCounter</td><td rowspan=1 colspan=1>31.09/0.90330.254/275 27.02 / 0.859 / 0.328 /267 30.37/0.900</td><td rowspan=1 colspan=1>/0.270/ 224 30.08/0.8960.265i317</td></tr><tr><td rowspan=1 colspan=2>28.22/0.879/0.251/ 22925.64 / 0.845 /0.307 /252 27.83/0.8800.260/ 19827.77/0.8770.251b245</td></tr><tr><td rowspan=1 colspan=3>KitchenBonsaiAverage</td><td rowspan=1 colspan=2>29.73/0.896//0.183/ 239 20.53 / 0.671 /0.440 /297 28.37I0.890/0.210 / 246 28.84/0.891/|0.186/26930.67/0.920/0.241// 276 25.41 / 0.866 / 0.331 /326 29.84/0.910/0.260/28129.7310.90610.256/29827.74/0.809/0.2857325 23.61 / 0.692 / 0.413 /363 27.06/0.8000.303/323 27.09I0.79880.296/379</td></tr><tr><td rowspan=2 colspan=3>0103</td><td rowspan=2 colspan=2>22.58 / 0.625/ 0.458 / 212 20.53 / 0.543 / 0.575 / 159 22.66 / 0.620 / 0.470 / 18022.50 / 0.608 / 0.475 / 22225.39/0.8420.244/356 24.09 / 0.803 / 0.295 / 112 24.30/0.810/0.290/ 217 24.85/0.827/b0.259/313</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>25.39/0.8420.244/356 24.09 / 0.803 / 0.295 / 112</td></tr><tr><td rowspan=1 colspan=2>05</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>27.95/0.863310.246/415 26.87 / 0.841 / 0.296 /354</td><td rowspan=1 colspan=1>27.70/0.8600.270/ 318 27.6910.85660.259/398</td></tr><tr><td rowspan=3 colspan=3>[EÎµ] OWWO  0610131415Average</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>27.30/0.913/0.202321 26.53 / 0.899 / 0.229 / 107</td></tr><tr><td rowspan=1 colspan=1>29.31/0.853/0.252338 28.86/ 0.833 / 0.291 / 125</td><td rowspan=1 colspan=1>28.70 /0.840/0.280/337 28.78/0.839/0.275i341</td></tr><tr><td rowspan=1 colspan=2>30.36/0.8990.211/44929.79 / 0.889 / 0.247 / 171 29.45/0.88010.250 /378 28.98 / 0.878 //0.234/46829.73/0.924)40.145/394 27.09 / 0.874 / 0.220 / 237 29.26/0.9200.160/36028.92 / 0.909/ 0.169 /37428.10/0.893/0.182405 27.80 / 0.877 / 0.213 / 153 27.720.8900.200/427 27.58 /0.8777/0.215/42027.59/0.8520.242/361 26.45 / 0.820 / 0.296 / 177 27.00/0.841/0.266/320 26.99/0.838/0.261/359</td></tr><tr><td rowspan=1 colspan=3>Ce] ecee TrucksxTrainAverage</td><td rowspan=1 colspan=2>24.54/0.8222/0.291/184 23.59 / 0.803 / 0.318 / 128 23.86 / 0.810 / 0.316 / 155 23.33 / 0.799 / 0.314 / 18321.70/0.7590.328/204 21.17/ 0.744 / 0.349 / 118 21.13/0.749/0.347/160 21.31/0.756/0.334/18023.12/0.790/0.309194 22.38/ 0.774 / 0.334 / 123 22.49/0.780/0.332158 22.32 /0.778/0.324182</td></tr><tr><td rowspan=1 colspan=3>[1]BuDr John-bdeesonPlayroomAverage</td><td rowspan=1 colspan=2>28.78 / 0.875 / 0.338 / 447 23.58 / 0.808 / 0.440 / 521 28.16 / 0.869 / 0.343 / 550 27.66 / 0.865 / 0.347 / 44430.09/0.88510.318/489 25.72 / 0.845 / 0.384 /489 29.72/0.883/0.323/552 29.1940.87940.327/ 48629.44/0.8800.328/468 24.65 / 0.827 / 0.412 /505 28.94/0.876/0.333/551 28.43/0.872I0.337/ 465</td></tr></table>

Table A8. Detailed results on a selection of datasets and methods with the number of Gaussians limited to 100k. Each field contains PSNR, SSIM, LPIPS and FPS respectively. We highlight the best, second best and third best results among all. Slight discrepancies from the main table are due to rounding.
<table><tr><td rowspan=1 colspan=2>Scene</td><td rowspan=1 colspan=1>Ours                  3DGS (SfM)              MCMC (SfM)                 EDGS</td></tr><tr><td rowspan=2 colspan=2>BicycleGarden</td><td rowspan=1 colspan=1>25.25/ 0.758/ 0.242/ 234 24.06 / 0.651 / 0.370 /174 24.90 / 0.740 / 0.282// 162 24.95/0.751/0.251/248</td></tr><tr><td rowspan=1 colspan=1>26.87/0.837/ 0.157/ 246 25.16 / 0.743 / 0.296 /297  26.36/0.823/0.189/ 221 26.49/0.833/ 0.161I252</td></tr><tr><td rowspan=1 colspan=2>Stump</td><td rowspan=1 colspan=1>27.03/0.793//0.215/255 25.27 / 0.688 / 0.348 /226 26.81/0.779/0.250/ 175 26.50/0.768 /0.237296</td></tr><tr><td rowspan=2 colspan=2>RoomCounter</td><td rowspan=1 colspan=1>32.02/0.924H0.205/192 31.47/ 0.913 / 0.234 / 112 31.72/0.921/0.221 / 114 31.36/0.923/0.203/168</td></tr><tr><td rowspan=1 colspan=1>28.87/0.906/0.194/148 28.86 / 0.901 / 0.213 /113  28.95 / 0.907 / 0.205 / 93 29.0510.91210.1841137</td></tr><tr><td rowspan=1 colspan=2>Kitchen</td><td rowspan=1 colspan=1>31.38/0.929/0.124/172 30.76 / 0.918 / 0.143 /111 31.04 / 0.921 / 0.141 / 111 31.35/0.928/0.122154</td></tr><tr><td rowspan=1 colspan=2>BonsaiAverage</td><td rowspan=1 colspan=1>32.16/0.94/0.191/190 31.95 / 0.936 / 0.218 /135  31.97/0.939/0.210/ 128 32.02I0.942/0.191/17029.08/0.870/0.190 /205 28.22 / 0.821 / 0.260 /167 28.82/ 0.861/0.214 / 143 28.82/0.865/0.1933/204</td></tr><tr><td rowspan=1 colspan=2>01</td><td rowspan=1 colspan=1>23.46 // 0.690 / 0.356/140 23.81/ 0.682 / 0.385 / 102  23.73 / 0.685 / 0.378 / 90  23.63/ 0.686/0.363/ 84</td></tr><tr><td rowspan=1 colspan=2>03</td><td rowspan=1 colspan=1>27.01/0.8870.181/197 26.18 / 0.868 / 0.220 /111 26.50 / 0.875/ 0.212/ 105 27.16/0.8920.176/ 93</td></tr><tr><td rowspan=1 colspan=2>05</td><td rowspan=1 colspan=1>28.61/0.877/0.199/232 28.53 / 0.874 / 0.235 /162 28.71/0.877710.233I154 28.79//0.881/0.196/ 122</td></tr><tr><td rowspan=6 colspan=2>[EÎµ] OWNO 0610131415Average</td><td rowspan=1 colspan=1>27.790.932/0.159/195 27.59/ 0.931 / 0.161 /120 26.99 /0.934 / 0.159/ 152 27.38/0.941/0.135/ 105</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.160.905/0.165/235 31.04/ 0.894/ 0.194 / 143 30.56 / 0.892 /0.192/168 31.110.90670.1657/ 129</td></tr><tr><td rowspan=1 colspan=1>33.02/0..949/0.115128032.40/0.939/0.149 /165 32.18/ 0.934 / 0.155 /175 31.95 /0.944/0.123/151</td></tr><tr><td rowspan=1 colspan=1>31.57/0.9500.096/22531.570.946/0.107/135 31.14 / 0.946 / 0.108 /130 31.51/0.9500.095/112</td></tr><tr><td rowspan=1 colspan=1>30.50/0.944/0.090/231 30.59/0.934/ 0.114 /149 29.92 / 0.933 / 0.113 /162 30.52I0.942/0.095/125</td></tr><tr><td rowspan=1 colspan=1>29.140.89270.170/217 28.96/ 0.884 / 0.196 /136 28.72 / 0.885 /0.194/142 29.010.893/0.1697/115</td></tr><tr><td rowspan=2 colspan=2>Ce] secee Truck sxuTrainAverage</td><td rowspan=1 colspan=1>25.41/0.8583 /0.204/ 122 24.80 / 0.842 / 0.253 / 93  25.24 / 0.851 / 0.242 / 68 24.66 / 0.850 / 0.212/11421.97/0.799/0.253/140  22.28/ 0.790 / 0.277 / 72  22.12/0.798/0.275/ 86 22.27 / 0.812/0.246/104</td></tr><tr><td rowspan=1 colspan=1>23.69/0.82970.229/131 23.54/ 0.816 / 0.265 / 82  23.68/0.825/0.259/ 77 23.47 /0.831/0.229/109</td></tr><tr><td rowspan=1 colspan=2>Dr John-</td><td rowspan=2 colspan=1>29.20 / 0.890 / 0.291 / 270 28.85 / 0.881 / 0.307 / 151 28.71 / 0.884 / 0.313 / 180 28.60 / 0.888 / 0.292 / 26930.51/0.892/0.279/287 29.66 / 0.883 / 0.298/ 161 30.16/0.889/ 0.303 / 209 30.06/ 0.890/ 0.281/ 27729.86/0.891/0.285/ 278 29.26 / 0.882 /0.303/ 156 29.44/0.887/ 0.308 /194 29.33 /0.889/0.287 /273</td></tr><tr><td rowspan=1 colspan=2>[6]ubdeesonPlayroomAverage</td></tr></table>

Table A9. Detailed results on a selection of datasets and methods with the number of Gaussians limited to 500k. Each field contains PSNR, SSIM, LPIPS and FPS respectively. We highlight the best, second best and third best results among all. Slight discrepancies from the main table are due to rounding.

<table><tr><td></td><td>Scene</td><td>Ours</td><td>3DGS (SfM)</td><td>MCMC (SfM)</td><td>EDGS</td></tr><tr><td>E] 9-</td><td>Bicycle Garden Stump Room Counter Kitchen Bonsai</td><td>25.45/ 0.778/ 0.198/ 156 27.42 / 0.861 / 0.115 / 162 27.23/0.802/ 0.184/164 32.30/0.928 10.196I 139 29.21/0.911 10.181/107 31.66 0.932 0.117 118 / / 32.33/0.945/ /0.183/132</td><td>24.40 / 0.688 / 0.325 / 129 26.70 / 0.827 / 0.172 1142 25.64 / 0.719 / 0.301 /162 31.62 / 0.918 / 0.221 / 82 29.11 /0.907 / 0.201 /75 31.18 /0.924 /0.131/78 32.34 / 0.941 / 0.205 / 95 28.71 / 0.846 / 0.222 /109</td><td>25.34/ 0.768/ 0.238 / 3 / 106 26.83 / /0.847 I0.147 / 135 27.21/0.798/0.213/110 32.09/ 0.926/ 0.208/ / 79 29.24/0.913/0.191/ /61 31.4410.92710.130/72 32.46/0.94410.199/ 81</td><td>25.26/ 0.778/ 0.199 / /161 27.05 / 0.857 0.118 158 / 26.71/0.78310.203/179 31.71 I0.928/ 0.191 111 29.27 7/ 0.917 / 0.171/ 94 31.83 3/0.933/ 0.114 |101 32.44/0.947 0.181 I115</td></tr><tr><td>[EÎµ] OWWO</td><td>Average 01 03 05 06 10 13 14 15</td><td>29.37/0.880 0.168/140 23.73 /0.711/ /0.315/104 27.39/0.89610.167 /127 28.56/0.877 0.185/157 27.87 /0.936/0.150/ 136 31.6010.914/0.147 / 170 33.5510.957 10.098I 190 31.8010.953/0.090 â²146 / 30.99 / 0.950 /0.081 / 151</td><td>24.18/ 0.703 / 0.350 / 73 26.96 / 0.886 / 0.197 / 64 28.72/ 0.877 /0.227 / 115 27.68/ 0.933 / 0.158 /110 31.47/ 0.906/ 0.171 / 104 33.13/0.949/ 0.129 / 116 31.93/ 0.951/ 0.097/89 31.13/ 0.943/ 0.096/ 100</td><td>29.23/0.875/ 0.189/92 24.18 / 0.709 / 0.341 / 63 27.20/0.890/0.189/67 28.98/0.883/0.217/98 27.46 / 0.939/ 0.146/103 30.89 / 0.904 /0.169/120 32.83/ 0.945 / 0.131 /124 31.44 / 0.950 / 0.099 /96 30.42 / 0.942 / 0.097 /104</td><td>29.18/0.878/ 0.168/131 24.02 / 0.709/ 0.321/ /84 27.72 0.905/ /0.157/ 85 / 29.15/0.885/ 0.181/ 101 27.75 0.945 /0.127 / 102 31.82/0.920/ b0.138/16 32.8/0.95510.100/125 32.08 / 0.955/ 0.086 / 92</td></tr><tr><td>e] sedee su</td><td>Average Truck Train Average</td><td>29.44/0.8997 /0.154/ 148 25.34 / 0.864 /0.181 /97 22.05 /0.805/ 0.232 /109 23.70/0.835 /0.207 7103</td><td>29.40 / 0.894 / 0.178 / 96 25.06/ 0.851 / 0.234 / 70 22.13/ 0.801 / 0.255 /51 23.60 / 0.826 / 0.245 /60</td><td>29.18 /0.895 / 0.174 / 97 25.57 / 0.862 / 0.217 / 45 22.29/0.811/0.249/47 23.93/0.837/ 0.233/ 46</td><td>31.25 5/0.950/ 0.079 /104 29.5810.903/ 0.149/101 25.02 / 0.862 / 0.181 / 82 22.29 /0.823 3/0.220/788</td></tr><tr><td>[]u</td><td>Dr John-</td><td>28.98 / 0.886/ 0.282 / 194</td><td>28.85 / 0.884 / 0.293 / 106</td><td>29.06 / 0.884 / 0.291 / 164</td><td>23.66/0.84/0.201/80 28.75 / 0.890 / 0.277 / 180</td></tr><tr><td>Pde</td><td>son Playroom Average</td><td>30.39/ 0.891/ 0.264/200 29.69/0.889/0.273/ 197</td><td>29.02 / 0.877 / 0.290 / 112 28.94 / 0.881 / 0.292 / 109</td><td>30.28/ 0.890 / 0.284 /173 29.67/0.887/0.288/168</td><td>30.11/ 0.889/0.265/182 29.43/0.890/0.271/181</td></tr></table>

Table A10. Detailed results on a selection of datasets and methods with the number of Gaussians limited to 1M. Each field contains PSNR, SSIM, LPIPS and FPS respectively. We highlight the best, second best and third best results among all. Slight discrepancies from the main table are due to rounding.
<table><tr><td></td><td>Scene</td><td>Ours</td><td>3DGS (SfM)</td><td>MCMC (SfM)</td><td>EDGS</td></tr><tr><td>[E 9-</td><td>Bicycle Garden Stump Room Counter Kitchen</td><td>25.43 / 0.781 / 0.175/ 99 27.5910.870/0.097 7/97 27.03 / 0.796 / 0.177/ 100 32.3210.929/0.189/92 29.23/0.912 0.175/70 31.80/0.934/0.114 1/72 32.63/0.948/0.177/84</td><td>24.85 / 0.729 / 0.267 / 79 27.22 / 0.852 / 0.129 /84 26.18 / 0.749 / 0.255 /100 31.80 / 0.919 / 0.218 / 63 29.07 / 0.907 / 0.201 /68 31.60 / 0.927 / 0.126 /54</td><td>25.56 / 0.786 / 0.205 / 71 27.33/0.862/0.122/82 27.39/0.808/0.189/ 69 32.1410.92910.199/ 53 29.41/0.917/0.181/ 42 32.00/0.931/ I0.122 / 47</td><td>25.48/ 0.792/ 0.168/93 27.46/ 0.870/ 0.097/89 26.79 / 0.788 / 0.186 / 100 31.89/ 0.930/ 0.184/ 81 29.37/ 0.918 0.164I65 32.06/0.935 0.11165</td></tr><tr><td>[EÎµ] ONWO</td><td>Bonsai Average 01 03 05 06 10 13</td><td>29.43/0.881/0.158/88 23.92 /0.725 /0.283/73 27.73/0.902 0.155/70 28.65 / 0.877 70.176/91 27.93/0.937 710.143/ 79 31.81/ 0.920/0.135/104 33.88/0.961/0.088/112</td><td>32.34 / 0.941 / 0.204 /84 29.01 / 0.861 / 0.200 /76 24.51 / 0.721 / 0.321 /49 27.05 0.888 / 0.193 /47 28.69 / 0.877 / 0.227 /112 27.67 / 0.933 0.157 /109 31.77/ 0.915/ 0.153 / 70 33.79/0.957 / 0.112 / 73</td><td>32.80/0.948/0.189/ 53 29.52/0.883/ 0.172 / 60 24.51 / 0.731 / 0.303/ 45 27.73/0.900/0.171/45 29.20/0.887 0.201/67 27.58 /0.942/0.138/ 63 31.59 / 0.913 /0.151/ 80 33.30 / 0.953 / 0.111/80</td><td>32.62/0.947 0.175/85 / 29.38/0.883/0.155/83 24.24 / 0.725 / 0.285/ 64 27.77/0.909/ 0.149/82 29.31 / 0.887 70.169/90 27.93 / 0.947 0.123/97 32.26/0.928 /0.123189</td></tr><tr><td></td><td>14 15 Average Truck</td><td>32.02 /0.955/0.086/83 31.25/0.953 30.075I86 29.65/0.90470.143/87 25.43/0.865 /0.167 7/69</td><td>31.95 / 0.951 / 0.096 /82 31.3510.947/0.090 /83 29.60/0.899 / 0.169 /78 25.39 / 0.859 / 0.217 /53</td><td>31.75 / 0.953/ 0.093/ 62 30.80 /0.947 70.086/ 67 29.56/0.903/0.157/ 64</td><td>33.5510.96110.087/90 31.7610.95610.084/86 31.63/0..95 /0.072/87 29.81 /0.908 370.137/86 /</td></tr><tr><td>e] seee sau</td><td>Train Average</td><td>21.95/0.809 0.217/77 23.69/0.837/0.192 I73</td><td>22.29/ 0.802 / 0.253 /55 23.84/ 0.831 / 0.235 /54</td><td>25.86/0.869 /0.193/38 22.35/0.822/0.228/ 40 24.11/0.846/0.211/ 39</td><td>25.09 /0.865/ 0.162 /55 21.84 /0.827 70.204/58 23.47 /0.846/0.183/56</td></tr><tr><td>[1u dee</td><td>Dr John- son</td><td>29.01 / 0.884 / 0.274 / 127</td><td>28.75 / 0.886 / 0.282 / 77</td><td>28.83 / 0.882 / 0.285 / 104</td><td>28.65 / 0.887 / 0.268 / 113</td></tr></table>

Table A11. Detailed results on a selection of datasets and methods with the number of Gaussians limited to 2M. Each field contains PSNR, SSIM, LPIPS and FPS respectively. We highlight the best, second best and third best results among all. Slight discrepancies from the main table are due to rounding.