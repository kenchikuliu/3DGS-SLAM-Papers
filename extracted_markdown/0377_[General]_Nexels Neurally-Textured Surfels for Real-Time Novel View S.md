# Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries

Victor Rongâ1,2 Jan Heldâ3,4 Victor Chu1,2 Daniel Rebain5 Marc Van Droogenbroeck4 Kiriakos N. Kutulakos1,2 Andrea Tagliasacchiâ 1,3 David B. Lindellâ 1,2

1University of Toronto 2Vector Institute 3Simon Frasier University 4University of Liege \` 5University of British Columbia â,â equal contribution

https://lessvrong.com/cs/nexels

<!-- image-->  
Figure 1. Teaser. Our nexels match the quality of 3D Gaussian splats with a fraction of the number of primitives, and at faster rendering speeds than prior textured primitive works. We show test-set averages of the LPIPSâ, # of primitives, and frames-per-second for two scenes: the STUMP scene from the Mip-NeRF360 dataset [2] and the GROCERY scene from our dataset.

## Abstract

Though Gaussian splatting has achieved impressive results in novel view synthesis, it requires millions of primitives to model highly textured scenes, even when the geometry of the scene is simple. We propose a representation that goes beyond point-based rendering and decouples geometry and appearance in order to achieve a compact representation. We use surfels for geometry and a combination of a global neural field and per-primitive colours for appearance. The neural field textures a fixed number of primitives for each pixel, ensuring that the added compute is low. Our representation matches the perceptual quality of 3D Gaussian splatting while using 9.7Ã fewer primitives and 5.5Ã less memory on outdoor scenes and using 31Ã fewer primitives and 3.7Ã less memory on indoor scenes. Our representation also renders twice as fast as existing textured primitives while improving upon their visual quality.

## 1. Introduction

In computer graphics, a sceneâs geometry governs how light is blocked, while its appearance determines the color contributed to each rendered pixel. Point-based representations like 3D Gaussian splats (3DGS) merge these roles, since each primitive jointly encodes geometry and appearance [22]. This coupling stands in contrast to traditional surface-based methods, where coarse geometry (e.g., a quad) can still capture fine-grained appearance (e.g., a detailed painting) through texturing. As illustrated in Figure 2, Gaussian splats therefore require a large number of primitives to represent high-frequency appearance, leading to substantial memory usage.

A more compact representation becomes possible by decoupling geometry from appearance. When appearance is delegated to textures, mesh-based models can depict highly realistic scenes, as commonly seen in professionally produced games and films. Yet meshes have lagged behind in the task of novel view synthesis, where the objective is to generate new viewpoints from a collection of input images of a 3D scene. In the dense-pose setting we consider, the dominant strategy is differentiable rendering, which optimizes scene parameters (geometry, light, and materials) so that rendered views match the training images. Within this framework, meshes face difficulties due to their discrete connectivity, potential for self-intersections, and complications in computing silhouette gradients [26, 29, 31, 37].

<!-- image-->  
Figure 2. Limitation of point-based representations. Our captured TABLE scene contains planar geometries with complex textures such as the sheet music above. An ideal representation would have very few parameters devoted to geometry, and much more devoted to appearance. Gaussian splatting needs millions of Gaussians to represent the fine details, each of which has a fixed ratio of geometry-to-appearance parameters.

To mitigate the need for an excessive number of primitives without resorting to the complexities of mesh optimization, recent work has proposed decoupling geometry and appearance in Gaussian splatting by introducing textured splats, most commonly via per-primitive image textures [7, 38, 42, 49]. However, the memory footprint of such textures grows quadratically with resolution, which constrains their ability to represent high-frequency appearance in large-scale scenes. Zhang et al. [59] attempt to overcome this limitation by employing a neural field to provide appearance in a deferred manner: each primitive samples neural features from a multi-resolution hash grid, which are then alpha-composited and passed through a feed-forward network. While this design requires only a single network evaluation, hash-grid lookups are similarly expensive, and performing several of them for every rayâprimitive intersection results in slow rendering (<30 FPS), as well as slow training.

Although these methods take inspiration from textured meshes, they miss a crucial observation: in standard mesh rasterization with depth buffering, texture fetches are performed only for a limited set of opaque surface fragments [1, 21]. Motivated by this, we introduce a new representation that preserves the optimization advantages of point-based approaches while leveraging the lightweight appearance modeling of mesh-based methods â all without sacrificing real-time rendering performance.

Contributions. We introduce nexels, a novel neurallytextured primitive that overcomes the coupling of geometry and appearance in Gaussian splatting. In particular: (i) For geometry, we introduce a differentiable quad indicator, enabling us to better reconstruct surfaces and sharp boundaries; (ii) For appearance, we learn a world-space neural field that provides view-dependent texture only for the most relevant primitives, capturing fine details efficiently while keeping computation low; (iii) We introduce a new dataset that highlights the limitations of point-based representations in accurately reconstructing regions with high--frequency textures; (iv) We achieve perceptual parity with 3DGS using 9.7Ã fewer primitives on outdoor scenes and 31Ã fewer primitives on indoor scenes; (v) Compared to concurrent texture methods, we achieve better photometric quality, while rendering more than twice as fast.

## 2. Related Work

## 2.1. Neural Fields

A neural field implicitly represents a quantity over a region, such as 3D space, through neural network queries. In novel view synthesis, Mildenhall et al. [34] use neural fields to parameterize per-point radiance and density for volumetric rendering. These neural radiance fields (NeRFs) have slow rendering speed due to both the large number of samples needed for volumetric rendering as well as expensive persample calculations. To reduce the number of samples used for rendering volumes, early NeRF papers employ stratified sampling and auxiliary proposal networks to improve the sampling efficiency [2, 3, 34]. Later works use empty space skipping and other acceleration techniques to avoid unnecessary queries [28, 36, 46]. The number of queries can be further constrained using explicit representations such as shells and meshes [45, 52]. In parallel, follow-up methods reduce the computational cost of each query by moving much of the neural networkâs representational power into spatial features backed by voxel grids [13, 48], hierarchical grids [33, 36, 50, 57], point sets [9, 10, 53], and other structures [6, 8]. Another line of work avoids the slow runtime of neural fields by only using them in an initial stage of training before baking in the appearance and geometry into a fast mesh representation [14, 40, 51, 55]. In contrast, we use surfels to obtain sample locations which avoids the added complication of mesh extraction while seamlessly integrating into the differentiable rendering scheme.

## 2.2. Gaussian Splatting

Gaussian splatting completely forgoes the neural network and instead represents radiance fields as a set of explicit primitives which have decaying opacities according to a Gaussian distribution [22]. It leverages fast point-based rasterization schemes to achieve real-time performance. There has been a deluge of follow-up works, of which we will only focus on those that present new geometry or appearance representations. Several works find that using flattened 3D primitives also results in high-quality novel view synthesis results [11, 20] and Ye et al. [56] use these to explicitly model opaque surfaces. Other non-Gaussian geometries have also been found to be effective: beta kernels, polynomial kernels, triangles, and more have been proposed [15â 18, 30, 35]. Beta splats have been particularly effective at achieving higher rendering quality with fewer parameters. They achieve this reduction through a leaner set of parameters for view-dependent appearance, which is orthogonal to our own work [30]. A number of Gaussian splatting works use neural fields in various ways, but unlike our work, the splats in these works have no spatial variation across them and cannot represent details smaller than themselves [32, 41].

<table><tr><td>Method</td><td>Scene</td><td>Ada</td><td>Speed</td><td>Mem</td></tr><tr><td>Texture-GS [54]</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>GStex [42]</td><td>â</td><td>X</td><td>â</td><td></td></tr><tr><td>Textured Gaussians [7]</td><td>â</td><td>Ã</td><td>X</td><td></td></tr><tr><td>BBSplat [49]</td><td>â</td><td>â</td><td>X</td><td>ÃÃÃÃ</td></tr><tr><td>NeST-Splatting [59]</td><td>â</td><td>X</td><td>X</td><td>â</td></tr><tr><td>GS-Texturing [38]</td><td>â</td><td>â</td><td>â</td><td>X</td></tr><tr><td>Nexels</td><td>â</td><td>â</td><td>â</td><td>â</td></tr></table>

Table 1. Prior and concurrent textured works. We summarize the limitations of recent representations with textured primitives. The criteria are scene-level reconstructions, adaptive density control with textures, real-time rendering speeds (30+ FPS), and ability to capture texture details in a memory-efficient manner.

A recent line of work applies textures to Gaussian splats to increase the appearance capacity of the representation. This has been done through per-primitive image textures [7, 38, 42, 49] and neural fields [54, 59]. Despite promising results, there are three key issues which hinder these methods, summarized in Tab. 1. First, the majority of these works initialize from a trained dense Gaussian splatting model, which limits their ability to optimize for sparse geometries. Furthermore, the texturing in all prior methods is applied for every ray-primitive interaction. For BBSplat and Textured Gaussians, whose alpha textures further increase overdraw, renders of scene-level reconstructions typically take over fifty milliseconds. Finally, an ideal texture should be able to capture multiscale details within a reasonable amount of memory. Works which use per-primitive image textures are limited by their resolution, which scales poorly with memory. As a partial solution, concurrent work from Papantonakis et al. [38] use image textures with adaptive resolutions. Only NeST-Splatting fully overcomes the issue by using a neural field, but this results in much slower renders. Our representation uses a neural field at a fixed number of samples per-pixel, leading to real-time renders.

## 3. Preliminaries

Our work builds on both Gaussian splatting and neural field methods. In particular, we make use of the differentiable surfel rasterization of 2D Gaussian splatting (2DGS) [20] and the neural field architecture of Instant-NGP [36].

<!-- image-->  
Figure 3. Kernel. We use a generalized Gaussian kernel in order to model near-opaque primitives. Values below 0.1 are set to 0 for visualization purposes. (Left) For Î³=1, our nexels correspond to a Gaussian distribution. (Right) As Î³ââ, they converge toward the indicator of a quad.

2D Gaussian Splatting. Each surfel in 2D Gaussian splatting (2DGS) is a 2D Gaussian in 3D space parameterized by its opacity $\textbf { \textit { o } } \in \mathbb { \textit { \textbf { R } } }$ , rotation matrix R = $[ \mathbf { v _ { 1 } } , \mathbf { v _ { 2 } } , \mathbf { v _ { 3 } } ] \in \mathbb { S } \mathbb { O } _ { 3 }$ , mean $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , and scale $\pmb { \sigma } \in \mathbb { R } ^ { 2 }$ . The first two axes of the rotation, $\mathbf { v _ { 1 } }$ and $\mathbf { v _ { 2 } } ,$ determine the axes of the Gaussian, while $\mathbf { v _ { 3 } }$ functions as the surfel normal. Additionally, each primitive has a view-dependent radiance $\mathbf { c } ^ { \odot } \in \mathbb { R } ^ { 3 S }$ parameterized by channel-wise spherical harmonics coefficients, where $S$ is the number of coefficients for a predetermined degree.

Following 3DGS, the surfels are preprocessed and sorted by the depth of their centers. For a ray $\mathbf { r } : = \mathbf { o } + t \mathbf { d }$ emanating from the camera position, the intersection in primitive space (u, v) can be computed differentiably from its geometry parameters $[ 2 0 , 4 7 ]$ . This is used to compute its alpha value, Î± := o exp $\left( - ( u ^ { 2 } + v ^ { 2 } ) / 2 \right)$ . The primitives are alpha-composited in a tile-based rasterizer. The final color is

$$
\mathbf { c } ( \mathbf { r } ) : = \sum _ { i } T _ { i } \alpha _ { i } \mathbf { c } _ { i } ^ { \odot } ( \mathbf { r } ) , \quad T _ { i } : = \prod _ { j } ^ { i - 1 } ( 1 - \alpha _ { i } ) .\tag{1}
$$

Instant-NGP. Instant-NGP is a neural field architecture composed of a multiresolution hash-grid H and a tiny multilayer perceptron (MLP) network G. The multiresolution hash-grid has L levels, each defining a hash-table $\mathcal { H } _ { \ell }$ at given scale $s _ { \ell } .$ . In our implementation, each hash-table has exactly T entries. Finally, it has a feature dimension of F . Altogether it has $L \times T \times F$ parameters.

Given an input coordinate $x \in \mathbb { R } ^ { 3 }$ , an Instant-NGP F can be configured to return a vector whose size is the MLP output dimension. First, a hash $\mathbf { h } _ { \ell } \in \{ 0 , 1 , \dots , T - 1 \}$ is calculated for each hash-grid level â based on $s _ { \ell } \cdot x .$ . This hash is indexed into the grid at that level to retrieve a feature vector. These are concatenated to obtain

$$
\begin{array} { r } { \ \mathcal { H } ( x ) : = \left( \mathcal { H } _ { 0 , { \bf h } _ { 0 } } , \ldots , \mathcal { H } _ { L - 1 , { \bf h } _ { L - 1 } } \right) \in \mathbb { R } ^ { L \times F } . } \end{array}\tag{2}
$$

Finally, the MLP is applied to yield $\mathcal { F } ( \boldsymbol { x } ) : = \mathcal { G } ( \mathcal { H } ( \boldsymbol { x } ) )$

<!-- image-->  
Figure 4. Rendering pipeline. (a) We iterate over the surfels and determine the weights at each intersection while also returning a nontextured render. (b) We collect the K intersections which have the largest weights. (c) We pass their K positions into a neural field to obtain textured appearance. (d) We composite the textures with the non-textured render to get the final output.

## 4. Nexels

This section provides an overview of the main components of our method. Section 4.1 introduces the geometry and appearance representations, while Section 4.2 describes the rendering process. Finally, Section 4.3 outlines the optimization framework, including adaptive density control and the loss terms used during training. For notational convenience, we present our method for a single pixel p with ray r and use view-independent radiance in equations.

## 4.1. Representation

Geometry. We represent the geometry of nexels using a set of N surfels defined by a kernel governing their opacities. Following 2DGS, the $i ^ { \mathrm { { t h } } }$ primitive has a mean $\mu _ { i } \in { \mathbb { R } } ^ { 3 }$ , rotation $\mathbf { R } _ { i } \in \mathbb { S } \mathbb { O } _ { 3 }$ , scale $\sigma _ { i } \in \mathbb { R } ^ { 2 }$ , and opacity $o _ { i } \in \mathbb { F }$ . In addition, we have a gamma parameter $\gamma _ { i } \in \mathbb { R } ^ { 2 }$ that controls how rect-like the primitive is. The opacity at an intersection $( u , v )$ in primitive space is defined by the generalized Gaussians,

$$
\alpha : = o _ { i } \exp \left( - \frac { | u | ^ { 2 \gamma _ { i , 1 } } } { 2 } \right) \exp \left( - \frac { | v | ^ { 2 \gamma _ { i , 2 } } } { 2 } \right) .\tag{3}
$$

Figure 3 shows the nexel shape for different values of $\gamma .$ As Î³â1, the kernel converges to a standard Gaussian. As Î³ââ it converges toward a quad with a sharp transition. This kernel is well suited for modeling opaque and flat surfaces: for large o and $\gamma _ { : }$ the function remains close to 1 over most of the support, unlike a conventional Gaussian which has a long tail. Our kernel behaves like a Gaussian where smoothness is needed, while still being able to produce hard edges in regions with sharp transitions. We constrain $\gamma { \geq } 1$ , which guarantees that our kernel is $C _ { \infty }$ . Taking inspiration from the separable filters of signal processing, we choose to take the outer product between the u and v axes. As Î³ grows, the kernel approaches a rectangle rather than an ellipse as in prior work [15, 30].

Appearance. Each surfel has two modes of appearance: non-textured and textured. During rendering, we select which to use on a per-pixel basis. The non-textured appearance of surfel i is the same as 2DGS [20], $\mathbf { c } _ { i } ^ { \odot } \in \mathbb { R } ^ { 3 S }$

The textured appearance of each primitive is represented using a shared neural field $\mathcal { F }$ backed by an Instant-NGP architecture [36]. The field takes as input any 3D position $x \in \mathbb { R } ^ { 3 }$ and outputs a view-dependent radiance $\mathcal { F } ( x )$ as a set of spherical harmonics coefficients. The texture at a given point on any of the surfels can be computed by querying the neural field at that position.

For a primitive i, we define $t _ { i } ^ { * } ( \mathbf { r } )$ as the intersection depth and $x _ { i } ^ { * } ( \mathbf { r } ) : = \mathbf { o } + t ^ { * } \mathbf { d }$ as the world-space intersection. Naively, the textured appearance for pixel p would be $\mathcal { F } ( x _ { i } ^ { * } ( \mathbf { r } ) )$ . Using point samples for differentiable rendering leads to inaccuracies in the fine levels of the hash-grid. Following the down-weighting analysis of Barron et al. [4], we multiply the hash-grid features of level â by

$$
\Delta _ { i , \ell } ( \mathbf { r } ) : = 1 - \exp \left( - \frac { 1 } { 2 \pi } \left( \frac { f } { s _ { \ell } t _ { i } ^ { \ast } ( \mathbf { r } ) } \right) ^ { 2 } \right) ,\tag{4}
$$

where $f$ is the focal length. This requires the same amount of computation as the naive sampling and removes noticeable aliasing artifacts. Ultimately, we define the filtered texture radiance along r for primitive i as

$$
\mathbf { c } _ { i } ^ { \oplus } ( \mathbf { r } ) : = \mathcal { G } \left( \Delta _ { i } ( \mathbf { r } ) \cdot \mathcal { H } ( x _ { i } ^ { * } ( \mathbf { r } ) ) \right) .\tag{5}
$$

Note that the only primitive information needed to compute $\mathbf { c } _ { i } ^ { \oplus } ( \mathbf { r } )$ is the intersection depth $t _ { i } ^ { * } ( \mathbf { r } )$

## 4.2. Rendering

Our rendering step is composed of two passes over the image pixels. The first pass computes the initial render using only the non-textured appearance and collects information on which primitives should be textured. The second pass applies the neural texture at the collected ray-primitive intersections and updates the initial render. Figure 4 provides an overview of the different rendering passes.

Collection Pass. In the first pass, we compute the alpha $\alpha _ { i }$ and transmittance values $T _ { i }$ across the M surfels which intersect a pixelâs ray. We use eq. (1) to compute the initial render N for each pixel, assuming no primitive is textured.

When rendering a single pixel, we limit the number of primitives which are textured to a hyperparameter $K \ll M$ We determine the subset of primitives which are textured based on their blending weights which we denote as $\mathbf { w } _ { i } : =$ $\alpha _ { i } T _ { i }$ . For a given pixel, the K primitives with the highest weights are selected. Inspired by fragment buffer techniques [5], we maintain a per-pixel buffer of size K. Each buffer entry contains a primitive id, its weight, and its depth, totalling to 3K 32-bit registers. Initially, the buffer is empty. When iterating through the M primitives and computing the weights, we update the buffer to store the running primitives with the highest weights. The final id, depth, and weight buffers are written into $H \times W \times K$ images I, D, and W, respectively. These are returned along with the non-textured render N. Additionally, for the K primitives in I, we remove their non-textured colour from the render N. The final value of the non-textured render at pixel $p$ is

$$
\mathbf { N } _ { p } : = \sum _ { i = 0 } ^ { M - 1 } \left[ i \notin \mathbf { I } _ { p } \right] \mathbf { w } _ { i } \mathbf { c } _ { i } ^ { \odot } .\tag{6}
$$

Texturing Pass. The texturing pass first computes worldspace intersection positions from the depth-map buffer D, yielding $\boldsymbol { x } ~ \in ~ \dot { \mathbb { R } } ^ { H \times W \times K \times 3 }$ . Following eq. (5), we query the neural field to compute the filtered texture $\textbf { T } \in$ $\dot { \mathbb { R } } ^ { H \times W \times K \times 3 }$ where $\mathbf { T } _ { \boldsymbol { p } , j } : = \mathbf { c } _ { \mathbf { I } _ { \boldsymbol { v } , j } } ^ { \oplus } ( \mathbf { r } )$ for pixel p with corresponding ray r. Extending eq. (1) to include the choice between texturing and not texturing, we have

$$
\begin{array} { r } { \displaystyle \sum _ { i = 0 } ^ { M - 1 } T _ { i } \alpha _ { i } \mathbf { c } _ { i } ( \mathbf { r } ) = \sum _ { i = 0 } ^ { M - 1 } \mathbf { w } _ { i } \left( \left[ i \notin \mathbf { I } _ { p } \right] \mathbf { c } _ { i } ^ { \odot } + \left[ i \in \mathbf { I } _ { p } \right] \mathbf { c } _ { i } ^ { \oplus } ( \mathbf { r } ) \right) } \\ { = \mathbf { N } _ { p } + \displaystyle \sum _ { j = 0 } ^ { K - 1 } \mathbf { W } _ { p , j } \mathbf { T } _ { p , j } . \qquad ( 7 } \end{array}
$$

Hence, we compute the final render from N, W, and T.

## 4.3. Optimization

We follow the scheme of 3DGS [22] and optimize against training views for 30,000 iterations with regularizations. We also interleave density control operations.

Adaptive Density Control. We introduce a hyperparameter P to control the number of primitives. Similar to BBSplat [49], we sample the point cloud output of COLMAP for initialization [44]. Specifically, we use farthest point sampling to reduce the initial point cloud to 0.5P points [27]. We initialize the scale at each point based on its distance to its predecessors during farthest point sampling.

We also prune unnecessary primitives and add new ones in regions where the current geometry is insufficient. We perform a densification and pruning step every 100 iterations from iteration 500 to 25,000. We select 5% of primitives to split evenly along their longest axis. The selection is stochastic with probabilities defined using blended errors similar to Rota Bulo et al. [ \` 43]âs method. We cap the number of primitives added in the splitting by P . Afterward, we prune all primitives whose opacity is below 0.005.

Losses. At each iteration, we render the nexels from a given training view to produce a predicted image. Following Kerbl et al. [22], we compute a photometric loss between the prediction and ground truth using a mixture of an $L _ { 1 }$ loss $\mathcal { L } _ { 1 }$ and D-SSIM loss ${ \mathcal { L } } _ { \mathrm { D - S S I M } }$ [22],

$$
\mathcal { L } _ { \mathrm { i m a g e } } : = ( 1 - \lambda _ { \mathrm { D - S S I M } } ) \mathcal { L } _ { 1 } + \lambda _ { \mathrm { D - S S I M } } \mathcal { L } _ { \mathrm { D - S S I M } } .\tag{8}
$$

We also include a texture loss, $\mathcal { L } _ { \mathrm { t e x t u r e } }$ , so that the nonpremultiplied texture fits to the ground truth image $I ^ { \mathrm { G T } }$

$$
\frac { 1 } { 3 H W } \sum _ { y = 0 } ^ { H - 1 } \sum _ { x = 0 } ^ { W - 1 } \sum _ { c = 0 } ^ { 2 } \left| I _ { y , x , c } ^ { \mathrm { G T } } - \frac { \sum _ { j = 0 } ^ { K - 1 } \mathbf { W } _ { y , x , j } \mathbf { T } _ { y , x , j , c } } { \sum _ { j = 0 } ^ { K - 1 } \mathbf { W } _ { y , x , j } } \right| .\tag{9}
$$

To penalize non-textured rendering, we minimize the total blending weights of the non-textured appearance,

$$
\mathcal { L } _ { \mathrm { a l p h a } } : = \frac { 1 } { H W } \sum _ { y = 0 } ^ { H - 1 } \sum _ { x = 0 } ^ { W - 1 } \left( 1 - \sum _ { j = 0 } ^ { K - 1 } \mathbf { W } _ { y , x , j } \right) .\tag{10}
$$

We include an $L _ { 1 }$ loss on the opacity to induce sparsity [23],

$$
\mathcal { L } _ { \mathrm { o p a c i t y } } : = \frac { 1 } { N } \sum _ { i = 0 } ^ { N - 1 } o _ { i } .\tag{11}
$$

Finally, we add a grid weight regularization term,

$$
\mathcal { L } _ { \mathrm { g r i d } } : = \sum _ { \ell = 0 } ^ { L - 1 } \sum _ { i = 0 } ^ { T - 1 } \sum _ { j = 0 } ^ { F - 1 } s _ { i } ^ { - 3 } \mathcal { H } _ { \ell , i , j }\tag{12}
$$

which encourages zero-mean grid values that justify the down-weighting done for grid anti-aliasing [4]. The final loss is

$$
\begin{array} { r l } & { { \mathscr { L } } : = { \mathscr { L } } _ { \mathrm { i m a g e } } + \lambda _ { \mathrm { a l p h a } } { \mathscr { L } } _ { \mathrm { a l p h a } } \quad + \lambda _ { \mathrm { t e x t u r e } } { \mathscr { L } } _ { \mathrm { t e x t u r e } } } \\ & { \qquad + \lambda _ { \mathrm { o p a c i t y } } { \mathscr { L } } _ { \mathrm { o p a c i t y } } + \lambda _ { \mathrm { g r i d } } { \mathscr { L } } _ { \mathrm { g r i d } } . } \end{array}\tag{13}
$$

We set $\lambda _ { \mathrm { a l p h a } } = 0 . 0 0 5 , \lambda _ { \mathrm { t e x t u r e } } = 0 . 5 , \lambda _ { \mathrm { o p a c i t y } } = 0 . 0 1 , \lambda _ { \mathrm { g r i d } } =$ 0.01, and $\lambda _ { \mathrm { D - S S I M } } = 0 . 2$ for all experiments.

## 5. Results

Our method is implemented in Python and CUDA on top of the INRIA 3DGS codebase [22]. We use PyTorch for performing parameter activations and composing the render passes with autograd and use the Adam optimizer for all parameters [24, 39]. We implement custom CUDA kernels for our multiresolution hash-grid operations and rasterization passes. We use the NVIDIA tinycudann library for the fully-fused MLP pass [36]. For all scenes, we use a texture limit of just K = 2. The MLP has two hidden layers without bias terms, each with a width of 64, and ReLU activations. We use degree 3 spherical harmonics coefficients for both appearance modes, corresponding to S = 16 and an MLP output dimension of 48. For the hash-grid, L = 16 and F = 2. Unless otherwise stated, we use $T = 2 ^ { 2 0 }$

<!-- image-->

Figure 5. Quality versus representation size. We show how the perceptual quality of our representation varies depending on the number of primitives (top row) and memory (bottom row). We evaluate the LPIPSâ across multiple settings for several methods and display separate graphs averaged over the four Mip-NeRF360 indoor scenes (left column), five Mip-NeRF360 outdoor scenes (middle column), and the four highly textured scenes of our custom dataset (right column). Nexels consistently has the lowest LPIPS for any number of primitives under 106, for all three of the datasets. When plotted against memory, nexels is always in the top three at any amount.
<table><tr><td></td><td colspan="5">Mip-NeRF360</td><td colspan="5">Tanks &amp; Temples</td><td colspan="5">Custom</td></tr><tr><td></td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Memory</td><td>#Prim.</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Memory</td><td>#Prim.</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Memory</td><td>#Prim.</td></tr><tr><td>3DGS [22]</td><td>27.21</td><td>0.815</td><td>0.214</td><td>640MB</td><td>2.7M</td><td>23.80</td><td>0.853</td><td>0.169</td><td>348MB</td><td>1.5M</td><td>26.65</td><td>0.858</td><td>0.224</td><td>1630MB</td><td>7.0M</td></tr><tr><td>2DGS [20]</td><td>27.04</td><td>0.805</td><td>0.252</td><td>480MB</td><td>2.0M</td><td>23.15</td><td>0.831</td><td>0.212</td><td>200MB</td><td>0.8M</td><td>25.98</td><td>0.831</td><td>0.267</td><td>450MB</td><td>1.8M</td></tr><tr><td>DBS [30]</td><td>28.10</td><td>0.829</td><td>0.192</td><td>340MB</td><td>3.1M</td><td>24.82</td><td>0.871</td><td>0.144</td><td>260MB</td><td>2.0M</td><td>26.21</td><td>0.842</td><td>0.259</td><td>270MB</td><td>2.5M</td></tr><tr><td>TriSplat [16]</td><td>27.16</td><td>0.814</td><td>0.191</td><td>790MB</td><td>3.5M</td><td>23.14</td><td>0.857</td><td>0.143</td><td>400MB</td><td>2.0M</td><td>24.28</td><td>0.780</td><td>0.275</td><td>620MB</td><td>2.7M</td></tr><tr><td>NeST-Splat [59]</td><td>26.68</td><td>0.795</td><td>0.212</td><td>440MB</td><td>1.0M</td><td>23.02</td><td>0.824</td><td>0.181</td><td>260MB</td><td>0.5M</td><td>25.40</td><td>0.824</td><td>0.246</td><td>900MB</td><td>3.2M</td></tr><tr><td>BBSplat [49]</td><td>26.98</td><td>0.783</td><td>0.231</td><td>260MB</td><td>0.4M</td><td>23.83</td><td>0.854</td><td>0.147</td><td>260MB</td><td>0.4M</td><td>26.57</td><td>0.844</td><td>0.225</td><td>260MB</td><td>0.4M</td></tr><tr><td>Ours</td><td>27.35</td><td>0.802</td><td>0.201</td><td>160MB</td><td>0.4M</td><td>23.55</td><td>0.841</td><td>0.155</td><td>160MB</td><td>0.4M</td><td>27.04</td><td>0.851</td><td>0.222</td><td>160MB</td><td>0.4M</td></tr></table>

Table 2. Quantitative results. We compare nexels with 400K primitives to baselines at their maximum recommended number of primitives, corresponding to each baselineâs rightmost point on Figure 5.

## 5.1. Datasets and Metrics

We perform our evaluation over fifteen real-world scenes across three datasets: all nine scenes from Mip-NeRF360 [3], the TRAIN and TRUCK scenes from the Tanks & Temples dataset [25], and four scenes which we collect involving high amounts of texture detail. Existing datasets seldom have close-up views despite these being important for photorealism. Fine appearance details within these scenes barely show up in their metrics. Our scenes contain high-frequency appearance, such as text and patterns, observed at a range of distances, including close-up shots. We describe the capture and preprocessing details in the supplementary. The full dataset will be released publicly.

We compare against point-based representations [16, 20, 22, 30] and textured representations [49, 59], which are state-of-the-art for novel view synthesis. We refer to the supplementary for experimental details for our baselines. In addition, we consider Gaussian splat variants which can produce sparser point sets than the original 3DGS [12, 23].

We evaluate photometric quality with the standard

<!-- image-->  
Figure 6. Sparse geometries. We show the BICYCLE and TABLE scenes reconstructed with 40K primitives across different textured representations. Only nexels are able to maintain a high quality in this extreme setting.

<!-- image-->  
Figure 7. Low memory comparisons. We adjust the settings of baseline methods designed to support low memory representations to get a model around 50MB in size. We show the average LPIPSâ across test views, number of primitives, and total memory for the GARDEN and TRIPOD scenes. With less memory and far fewer primitives, our nexels are able to capture fine appearance details and sparsely-observed background structures which point-based methods struggle with. We use $T = 2 ^ { 1 9 }$ for the hash grid here.

PSNR, SSIM, and LPIPS metrics used in novel view synthesis and differentiable rendering [34, 58]. The LPIPS metric across all baselines is computed with the same normalization as the original 3DGS paper [22, 43]. All timings are performed on a single RTX 6000 Ada GPU using the CUDA implementation of each methodâs forward pass.

## 5.2. Evaluation

Novel-view synthesis. We compare all baselines at each of their maximum recommended settings in Tab. 2. On the Mip-NeRF360 dataset, nexels are competitive with pointbased representations with only 400K primitives, achieving better PSNR than all methods except Deformable Beta Splatting (DBS [30]), which has over 7Ã the number of primitives and twice the memory. On our custom dataset, we achieve the highest PSNR and LPIPS and come second in SSIM to 3DGS at 10Ã our memory. We achieve better scores than NeST-Splatting and BBSplat on the Mip-NeRF360 and custom datasets.

Visual quality under primitive and memory budgets. In Figure 5, we plot the perceptual quality (LPIPS) against the number of primitives and memory across three scene groups. For baselines which can specify a number of primitives, we include data points corresponding to appropriate subsets of {40K, 100K, 400K, 1M} primitives. Details are included in the supplementary material.

Our representation exceeds virtually all other models at equal numbers of primitives. For example, 400K nexels obtain an LPIPS score of 0.164 across Mip-NeRF360 indoor scenes, while BBSplat and DBS obtain LPIPS scores of 0.171 and 0.192, respectively. Conversely, to reach a given LPIPS, nexels require far fewer primitives than other methods. The best result of 3DGS on the Mip-NeRF360 outdoor scenes is 0.234 LPIPS with 3.9 million Gaussians. We achieve 0.230 LPIPS with just 400K nexels. Similarly, 3DGS obtains 0.190 LPIPS on indoor scenes with 1.2 million Gaussians. We reach the same score at 40K nexels. Our method is also competitive when measured against memory. At 40MB on MipNeRF-360 outdoor scenes, we achieve 0.294 LPIPS while DBS achieves 0.312 LPIPS. We include qualitative results echoing these trends in Figures 6 and 7.

<table><tr><td rowspan="2">Method</td><td colspan="2">Indoor</td><td colspan="2">Outdoor</td><td colspan="2">Mean</td></tr><tr><td>FPS</td><td>Train.</td><td>FPS</td><td>Train.</td><td>FPS</td><td>Train.</td></tr><tr><td>NeST-Splat</td><td>27</td><td>4.4h</td><td>19</td><td>5.2h</td><td>23</td><td>4.9h</td></tr><tr><td>BBSplat</td><td>20</td><td>2.4h</td><td>20</td><td>2.1h</td><td>20</td><td>2.2h</td></tr><tr><td>Nexels</td><td>40</td><td>1.5h</td><td>58</td><td>0.9h</td><td>50</td><td>1.1h</td></tr></table>

Table 3. Timing comparisons. We show the average FPS of testviews across the Mip-NeRF360 scenes for the textured representations as well as training times.

For the custom scenes with high amounts of texture, nexels are clearly better than non-textured representations. We achieve an LPIPS of 0.240 with just 40K nexels, outperforming DBS with 2.5 million splats, which obtains 0.259 LPIPS. BBSplat comes near us at 400K primitives but struggles at low primitive counts.

Training time and rendering speed. Table 3 compares training time and rendering speed for different texturing works. We separate between indoor and outdoor scenes as they have different image sizes. NeST-Splat averages five hours of training and only renders at 23 FPS across Mip-NeRF360 scenes. This is due to its texturing, where hashgrid lookups are done for each rayâprimitive intersection, totaling thousands of random accesses per ray. BBSplat has a faster training time, but their use of alpha textures results in high overdraw, leading to a rendering speed of 20 FPS. In contrast, nexels train 4.2 and 1.9 times faster than NeST-Splatting and BBSplat, respectively, and render at 50 FPS.

## 5.3. Ablations

The results in Tab. 4 ablate individual components of our method to demonstrate their importance. We evaluate each individual experiment across the Mip-NeRF360 scenes trained with 400K primitives.

First, we verify that the main component, the neural texture, is useful. We find that all visual metrics, especially LPIPS, worsen without the neural texture. Qualitatively, the non-textured renders lose background details. Removing the Î³ parameter and using 2D Gaussians as the kernel results in slight degradations overall as a Gaussian kernel leads to lower texture weights. Dropping the higher-order per-primitive spherical harmonics coefficients and only relying on the neural field for view-dependent effects is a possibility if memory is the main priority, as it removes 45 parameters for each primitive. The increase in LPIPS is very small, although there is a trade-off in PSNR. Finally, omitting the hash-grid downweighting factor in eq. (4) leads to noticeable aliasing artifacts in test views, as the fine levels overfit at the intersections seen from training poses.

<table><tr><td></td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Mem.</td><td>FPS</td></tr><tr><td>No texture</td><td>27.20</td><td>0.787</td><td>0.261</td><td>100MB</td><td>85</td></tr><tr><td>No gamma</td><td>27.10</td><td>0.797</td><td>0.210</td><td>160MB</td><td>48</td></tr><tr><td>No prim. SH</td><td>27.00</td><td>0.797</td><td>0.205</td><td>90MB</td><td>47</td></tr><tr><td>No down.</td><td>27.03</td><td>0.797</td><td>0.211</td><td>160MB</td><td>49</td></tr><tr><td>Full</td><td>27.35</td><td>0.802</td><td>0.201</td><td>160MB</td><td>50</td></tr></table>

Table 4. Ablations. We show how dropping individual elements of nexels affects the photometric quality, memory, and FPS of the representation averaged across all Mip-NeRF360 scenes.

## 5.4. Limitations

The speed of the Instant-NGP model is reliant on tensor core operations, which are only available on high-end GPUs. While nexels are real-time on the machinery we use, real-time rendering on mobile GPUs or low-end laptop GPUs is currently not possible. In addition, nexels produce noise in unseen regions, which is likely less appealing for end users than the blurry artifacts of Gaussian splats. Our work also does not support the full complexity of the imaging process, including motion blur and depth-of-field blur. These effects are particularly relevant for differentiable rendering of fine details, as even small deviations from the expected pinhole camera model can damage reconstructions.

## 6. Conclusion

We present the first representation for novel view synthesis that removes the reliance on dense primitives while achieving real-time rendering at high quality. Our results show that far fewer geometries are needed to model scene-level datasets than recent research would suggest. Future work could incorporate level-of-detail structures to model cityscale scenes. Sparse surfaces also work well for ray-tracing applications. Finally, the top-K scheme can be extended to support textures combining the neural field and perprimitive features. In conclusion, nexels break new ground in the exploration between volumetric and surface representations, capturing high-frequency appearance at a low memory and compute cost.

Acknowledgements. JH acknowledges support from the F.R.S.-FNRS. DBL and KNK acknowledge support from NSERC under the RGPIN program. DBL also acknowledges support from the Canadian Foundation for Innovation and the Ontario Research Fund.

## References

[1] Tomas Akenine-Moller, Eric Haines, Naty Hoffman, Angelo Â¨ Pesce, MichaÅ Iwanicki, and Sebastien Hillaire. Â´ Real-Time Rendering. AK Peters/CRC Press, Boca Raton, Florida, USA, 4 edition, 2018. 2

[2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-NeRF: A multiscale representation for antialiasing neural radiance fields. In IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pages 5835â5844, Montreal, Can., 2021.Â´ Inst. Electr. Electron. Eng. (IEEE). 1, 2

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 5460â5469, New Orleans, LA, USA, 2022. Inst. Electr. Electron. Eng. (IEEE). 2, 6, 1

[4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-NeRF: Anti-aliased gridbased neural radiance fields. In IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pages 19640â19648, Paris, Fr., 2023. Inst. Electr. Electron. Eng. (IEEE). 4, 5, 1

[5] Louis Bavoil, Steven P. Callahan, Aaron Lefohn, Joao L. D. Ë Comba, and Claudio T. Silva. Multi-fragment effects on the Â´ GPU using thek-buffer. In ACM Symp. Interact. 3D Graph. Games, pages 97â104, Seattle, Washington, 2007. ACM. 5

[6] Ang Cao and Justin Johnson. HexPlane: A fast representation for dynamic scenes. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 130â141, Vancouver, Can., 2023. IEEE. 2

[7] Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao, Tuotuo Li, Qinbo Li, Ayush Saraf, Jia-Bin Huang, Johannes Kopf, Gordon Wetzstein, and Changil Kim. Textured Gaussians for enhanced 3D scene appearance modeling. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 8964â8974, Nashville, TN, USA, 2025. IEEE. 2, 3

[8] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. TensoRF: Tensorial radiance fields. In Eur. Conf. Comput. Vis. (ECCV), pages 333â350, Tel Aviv, Israel, 2022. Â¨ Springer Nat. Switz. 2

[9] Hanyu Chen, Bailey Miller, and Ioannis Gkioulekas. 3D reconstruction with fast dipole sums. ACM Trans. Graph., 43 (6):1â19, 2024. 2

[10] Zhang Chen, Zhong Li, Liangchen Song, Lele Chen, Jingyi Yu, Junsong Yuan, and Yi Xu. NeuRBF: A neural fields representation with adaptive radial basis functions. In IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pages 4159â 4171, Paris, Fr., 2023. IEEE. 2

[11] Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu, Huamin Wang, and Weiwei Xu. High-quality surface recon-

struction using Gaussian surfels. In ACM SIGGRAPH Conf. Pap., pages 1â11, Denver, CO, USA, 2024. ACM. 2

[12] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of Gaussians. In Eur. Conf. Comput. Vis. (ECCV), pages 165â181. Springer Nat. Switz., 2024. 6, 4

[13] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 5491â5500, New Orleans, LA, USA, 2022. Inst. Electr. Electron. Eng. (IEEE). 2

[14] Yuan-Chen Guo, Yan-Pei Cao, Chen Wang, Yu He, Ying Shan, and Song-Hai Zhang. VMesh: Hybrid volume-mesh representation for efficient view synthesis. In ACM SIG-GRAPH Asia Conf. Pap., pages 1â11, Sydney, New South Wales, Aust., 2023. ACM. 2

[15] Abdullah Hamdi, Luke Melas-Kyriazi, Jinjie Mai, Guocheng Qian, Ruoshi Liu, Carl Vondrick, Bernard Ghanem, and Andrea Vedaldi. GES: Generalized exponential splatting for efficient radiance field rendering. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 19812â19822, Seattle, WA, USA, 2024. Inst. Electr. Electron. Eng. (IEEE). 3, 4

[16] Jan Held, Renaud Vandeghen, Adrien Deliege, Daniel \` Hamdi, Abdullah Rebain, Silvio Giancola, Anthony Cioppa, Andrea Vedaldi, Bernard Ghanem, Andrea Tagliasacchi, and Marc Van Droogenbroeck. Triangle splatting for real-time radiance field rendering. arXiv, abs/2505.19175, 2025. 6

[17] Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi, \` Bernard Ghanem, and Marc Van Droogenbroeck. 3D convex splatting: Radiance field rendering with 3D smooth convexes. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 21360â21369, Nashville, TN, USA, 2025. Inst. Electr. Electron. Eng. (IEEE).

[18] Jan Held, Renaud Vandeghen, Sanghyun Son, Daniel Rebain, Matheus Gadelha, Yi Zhou, Ming C. Lin, Marc Van Droogenbroeck, and Andrea Tagliasacchi. Triangle splatting+: Differentiable rendering with opaque triangles. arXiv, abs/2509.25122, 2025. 3

[19] Jan Held, Renaud Vandeghen, Adrien Deliege, Daniel \` Hamdi, Abdullah Rebain, Silvio Giancola, Anthony Cioppa, Andrea Vedaldi, Bernard Ghanem, Andrea Tagliasacchi, and Marc Van Droogenbroeck. Triangle splatting for real-time radiance field rendering. In Int. Conf. 3D Vis. (3DV), pages 1â10, Vancouver, Can., 2026. 2

[20] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D Gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH Conf. Pap., pages 1â11, Denver, CO, USA, 2024. ACM. 2, 3, 4, 6

[21] John Hughes, Andries van Dam, Morgan McGuire, David Sklar, James D. Foley, Steven Feiner, and Kurt Akeley. Computer Graphics: Principles and Practice. Addison-Wesley, 3 edition, 2014. 2

[22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):1â14, 2023. 1, 2, 5, 6, 7

[23] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Jeff Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3D Gaussian splatting as Markov chain Monte Carlo. In Adv. Neural Inf. Process. Syst. (NeurIPS), pages 80965â80986, Vancouver, Can., 2024. Curran Assoc. Inc. 5, 6, 2

[24] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv, abs/1412.6980, 2014. 6

[25] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: benchmarking large-scale scene reconstruction. ACM Trans. Graph., 36(4):1â13, 2017. 6

[26] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Trans. Graph., 39(6):1â14, 2020. 2

[27] Jingtao Li, Jian Zhou, Yan Xiong, Xing Chen, and Chaitali Chakrabarti. An adjustable farthest point sampling method for approximately-sorted point cloud data. In IEEE Work. Signal Process. Syst. (sips), pages 1â6, Rennes, France, 2022. IEEE. 5, 1, 2, 4

[28] Ruilong Li, Hang Gao, Matthew Tancik, and Angjoo Kanazawa. NerfAcc: Efficient sampling accelerates NeRFs. In IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pages 18491â 18500, Paris, Fr., 2023. IEEE. 2

[29] Tzu-Mao Li, Miika Aittala, Fredo Durand, and Jaakko Lehti- Â´ nen. Differentiable Monte Carlo ray tracing through edge sampling. ACM Trans. Graph., 37(6):1â11, 2018. 2

[30] Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and Andrew Feng. Deformable beta splatting. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Paper, pages 1â11. ACM, 2025. 3, 4, 6, 7

[31] Shichen Liu, Weikai Chen, Tianye Li, and Hao Li. Soft rasterizer: A differentiable renderer for image-based 3D reasoning. In IEEE Int. Conf. Comput. Vis. (ICCV), pages 7707â 7716, Seoul, South Korea, 2019. Inst. Electr. Electron. Eng. (IEEE). 2

[32] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-GS: Structured 3D Gaussians for view-adaptive rendering. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 20654â 20664, Seattle, WA, USA, 2024. IEEE. 3

[33] Julien N. P. Martel, David B. Lindell, Connor Z. Lin, Eric R. Chan, Marco Monteiro, and Gordon Wetzstein. Acorn: : adaptive coordinate networks for neural scene representation. ACM Trans. Graph., 40(4):1â13, 2021. 2

[34] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Commun. ACM, 65(1):99â106, 2021. 2, 7

[35] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3D Gaussian ray tracing: Fast tracing of particle scenes. ACM Trans. Graph., 43(6):1â19, 2024. 3

[36] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a mul-

tiresolution hash encoding. ACM Trans. Graph., 41(4):1â15, 2022. 2, 3, 4, 6, 1

[37] Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob. Large steps in inverse rendering of geometry. ACM Trans. Graph., 40(6):1â13, 2021. 2

[38] Panagiotis Papantonakis, Georgios Kopanas, Fredo Durand, and George Drettakis. Content-aware texturing for gaussian splatting. arXiv preprint arXiv:2512.02621, 2025. 2, 3

[39] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Adv. Neural Inf. Process. Syst. (NeurIPS), pages 8026â8037, Vancouver, Can., 2019. Curran Assoc. Inc. 6

[40] Christian Reiser, Stephan Garbin, Pratul Srinivasan, Dor Verbin, Richard Szeliski, Ben Mildenhall, Jonathan Barron, Peter Hedman, and Andreas Geiger. Binary opacity grids: Capturing fine geometric detail for mesh-based view synthesis. ACM Trans. Graph., 43(4):1â14, 2024. 2

[41] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-GS: Towards consistent real-time rendering with LOD-structured 3D Gaussians. IEEE Trans. Pattern Anal. Mach. Intell., pages 1â15, 2025. 3

[42] Victor Rong, Jingxiang Chen, Sherwin Bahmani, Kiriakos N. Kutulakos, and David B. Lindell. GStex: Perprimitive texturing of 2D Gaussian splatting for decoupled appearance and geometry modeling. In IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), pages 3508â3518, Tucson, AZ, USA, 2025. IEEE. 2, 3

[43] Samuel Rota Bulo, Lorenzo Porzi, and Peter Kontschieder. \` Revising densification in Gaussian splatting. In Eur. Conf. Comput. Vis. (ECCV), pages 347â362. Springer Nat. Switz., 2024. 5, 7, 2

[44] Johannes L. Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 4104â4113, Las Vegas, NV, USA, 2016. Inst. Electr. Electron. Eng. (IEEE). 5

[45] Gopal Sharma, Daniel Rebain, Kwang Moo Yi, and Andrea Tagliasacchi. Volumetric rendering with baked quadrature fields. In Eur. Conf. Comput. Vis. (ECCV), pages 275â292. Springer Nat. Switz., 2024. 2

[46] Nicholas Sharp and Alec Jacobson. Spelunking the deep. ACM Trans. Graph., 41(4):1â16, 2022. 2

[47] Christian Sigg, Tim Weyrich, Mario Botsch, and Markus Gross. GPU-based ray-casting of quadratic surfaces. In Symposium on Point-Based Graphics, Boston, MA, USA, 2006. Eurographics Assoc. 3

[48] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 5449â5459, New Orleans, LA, USA, 2022. Inst. Electr. Electron. Eng. (IEEE). 2

[49] David Svitov, Pietro Morerio, Lourdes Agapito, and Alessio Del Bue. Billboard splatting (BBSplat): Learnable textured primitives for novel view synthesis. In IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pages 25029â25039, Honolulu, HI, USA, 2025. IEEE. 1, 2, 3, 5, 6

[50] Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles Loop, Derek Nowrouzezahrai, Alec Jacobson, Morgan McGuire, and Sanja Fidler. Neural geometric level of detail: Real-time rendering with implicit 3D shapes. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 11353â11362, Nashville, TN, USA, 2021. IEEE. 2

[51] Zian Wang, Tianchang Shen, Jun Gao, Shengyu Huang, Jacob Munkberg, Jon Hasselgren, Zan Gojcic, Wenzheng Chen, and Sanja Fidler. Neural fields meet explicit geometric representations for inverse rendering of urban scenes. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 8370â8380, Vancouver, Can., 2023. IEEE. 2

[52] Zian Wang, Tianchang Shen, Merlin Nimier-David, Nicholas Sharp, Jun Gao, Alexander Keller, Sanja Fidler, Thomas Muller, and Zan Gojcic. Adaptive shells for efficient neural Â¨ radiance field rendering. ACM Trans. Graph., 42(6):1â15, 2023. 2

[53] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-NeRF: Point-based neural radiance fields. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 5428â5438, New Orleans, LA, USA, 2022. IEEE. 2

[54] Tian-Xing Xu, Wenbo Hu, Yu-Kun Lai, Ying Shan, and Song-Hai Zhang. Texture-GS: Disentangling the geometry and texture for 3D Gaussian splatting editing. In Eur. Conf. Comput. Vis. (ECCV), pages 37â53. Springer Nat. Switz., 2024. 3

[55] Lior Yariv, Peter Hedman, Christian Reiser, Dor Verbin, Pratul P. Srinivasan, Richard Szeliski, Jonathan T. Barron, and Ben Mildenhall. BakedSDF: Meshing neural SDFs for real-time view synthesis. In ACM SIGGRAPH Conf. Proc., pages 1â9, Los Angeles, CA, USA, 2023. ACM. 2

[56] Keyang Ye, Tianjia Shao, and Kun Zhou. When Gaussian meets surfel: Ultra-fast high-fidelity radiance field rendering. ACM Trans. Graph., 44(4):1â15, 2025. 2

[57] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. PlenOctrees for real-time rendering of neural radiance fields. In IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pages 5732â5741, Montreal, Can., 2021. IEEE.Â´ 2

[58] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pages 586â595, Salt Lake City, UT, USA, 2018. IEEE. 7, 5

[59] Xin Zhang, Anpei Chen, Jincheng Xiong, Pinxuan Dai, Yujun Shen, and Weiwei Xu. Neural shell texture splatting: More details and fewer primitives. In IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pages 25229â25238, Honolulu, HI, USA, 2025. IEEE. 1, 2, 3, 6, 4

# Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries

Supplementary Material

## 7. Overview

We include additional implementation details, dataset details, experiment details, and discussion below.

## 8. Implementation Details

## 8.1. Instant-NGP

Architecture. The original Instant-NGP implementation assumes inputs $x \in [ 0 , 1 ] ^ { d }$ Later works apply it to largescale scenes by contracting space to this hypercube $[ 3 , 4 ,$ 59]. We instead make minor modifications to the hash-grid in order to accept unbounded inputs. First, we use a hashgrid, even for the coarse levels. Second, the original hash function used by Muller et al. [ Â¨ 36] has structured collisions when evaluated over both positive and negative inputs. We modify the hash to accept negative integers by applying the function

$$
\mathsf { M a p P o s i t i v e } ( x ) = \left\{ \begin{array} { l l } { 2 x - 1 \qquad } & { x > 0 , } \\ { - 2 x \qquad } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

to the hash function inputs.

Anti-aliasing. As mentioned in the main text, we additionally perform anti-aliasing on the hash-grid using the down-weighting strategy with grid weight decay proposed in ZipNeRF [4]. We describe how we derive our downweighting factor, which differs slightly from that of Barron et al. [4].

Let tâ be the depth along the ray and $f$ to be the focal length. We model the projected pixel footprint as a 2D isotropic normal distribution $\mathcal { N }$ in world-space, with mean $x ^ { * } : = \mathbf { o } + t ^ { * } \mathbf { d }$ and standard deviation $\textstyle { \frac { t ^ { * } } { f } }$ , that is parallel to the image plane. The integrated texture which we seek to approximate is $\mathbb { E } _ { x \sim \mathcal { N } } \left[ \mathcal { F } ( x ) \right]$

Estimating the MLP as a linear function, we have

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { N } } \left[ \mathcal { F } ( \boldsymbol { x } ) \right] \approx \mathcal { G } \left( \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { N } } \left[ \mathcal { H } ( \boldsymbol { x } ) \right] \right) . } \end{array}
$$

The hash-grid at level $\ell , \mathcal { H } _ { \ell }$ , is queried by hashing $s _ { \ell } x$ and using trilinear interpolation. For a coarse but efficient estimate of $\mathbb { E } _ { x \sim \mathcal { N } } \left[ \mathcal { H } _ { \ell } ( s _ { \ell } x ) \right]$ , we only query $\mathcal { H } _ { \ell }$ at $s _ { \ell } x ^ { * }$ and approximate the rest. More formally, we model

$$
\mathcal { H } _ { \ell } ( s _ { \ell } x ^ { * } + \epsilon ) \approx w ( \epsilon ) \mathcal { H } _ { \ell } ( s _ { \ell } x ^ { * } ) + ( 1 - w ( \epsilon ) ) \mathcal { H } _ { \ell } ( s _ { \ell } x ^ { * } + \epsilon ) ,
$$

where $w ( \epsilon )$ indicates a unit cube, i.e. $w ( \epsilon ) = 1 \mathrm { i f } | | \epsilon | | _ { \infty } \leq$ $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ and $w ( \epsilon ) = 0$ otherwise. We assume that the unknown

values are zero in expectation. Hence,

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { N } } \left[ \mathcal { H } _ { \ell } ( s _ { \ell } x ) \right] \approx \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { N } } \left[ w ( s _ { \ell } ( \boldsymbol { x } - \boldsymbol { x } ^ { * } ) ) \mathcal { H } _ { \ell } ( s _ { \ell } x ^ { * } ) \right] } \\ & { \qquad \approx \mathrm { e r f } \left( \displaystyle \frac { f } { 2 \sqrt { 2 } s _ { \ell } t ^ { * } } \right) ^ { 2 } \mathcal { H } _ { \ell } ( s _ { \ell } x ^ { * } ) . } \end{array}
$$

Using the approximation er $\begin{array} { r } { \dot { \mathbf { \rho } } ( x ) ^ { 2 } \approx 1 - \exp \left( \frac { - 4 x ^ { 2 } } { \pi } \right) } \end{array}$ yields our down-weighting factor

$$
\Delta _ { \ell } : = 1 - \exp \left( - \frac { 1 } { 2 \pi } \left( \frac { f } { s _ { \ell } t ^ { * } } \right) ^ { 2 } \right) .
$$

The above derivation makes several very crude approximations. Itâs likely that many down-weighting factors are adequate. More high-powered anti-aliasing techniques could be used, but we found that this simple downweighting was already enough to remove noticeable artifacts.

Note that the MLP in the tinycudann library does not use bias terms. As we use ReLU activations, a natural consequence is that $\mathcal { G } ( \vec { 0 } ) = \vec { 0 } .$ . We choose to preserve this property as our derivation approximated $\mathcal { G }$ to be linear, but we observe that the neural texture is mostly grey in the background of large scenes, particularly the TRAIN scene of the Tanks & Temples dataset.

Other details. Unlike prior work in which the sample positions are assumed to be non-differentiable [36, 59], we compute the positional gradient from the bilinear interpolation into each hash-grid level.

Both the hash-grid and MLP are computed using 16-bit floating point numbers. The latterâs quantization is particularly important in order to leverage tensor core operations on recent NVIDIA GPUs, which greatly accelerates the MLP pass. The gradients are accumulated within 32- bit floating point tensors to maintain accuracy during optimization. We use the Instant-NGP codebase for the MLP pass and implement our own hash-grid following the design specified above. When performing the backward pass, we use the same default loss scale of 128 as Instant-NGP to reduce underflow. We report the nexels storage memory using two bytes for each Instant-NGP parameter and four bytes for all other parameters.

## 8.2. Adaptive Density Control

Initialization. Given N points $x _ { 0 } , \ldots , x _ { N - 1 }$ , farthest point sampling iteratively samples the point which is farthest from the ones already sampled [27]. In other words,

it returns an indexing $I _ { 0 } , I _ { 1 } , \ldots , I _ { N - 1 }$ , such that for any $0 < i \le N - 1$ ,

$$
I _ { i } : = \underset { k \notin \{ I _ { 0 } , \ldots , I _ { i - 1 } \} } { \arg \operatorname* { m a x } } \underset { 0 \leq j < i } { \operatorname* { m i n } } d \left( x _ { k } , x _ { I _ { j } } \right) .
$$

When initializing our model, we use farthest point sampling to sample the positions. For the scale, the nearest-neighbors initialization suggested by Kerbl et al. [22] is not ideal for our representation as it leads to small primitives. Instead, we run nearest-neighbors across prefixes of the farthest point sampling order so that the initial scale for point i with position $x _ { I _ { i } }$ is approximately mi $\mathrm { n } _ { 0 \leq j < i } d \left( x _ { I _ { i } } , x _ { I _ { j } } \right)$ . This implicitly leads to a hierarchy where points sampled early are larger while points sampled later are smaller. These smaller primitives can potentially be removed if the optimization finds that the larger primitives can replace them with texture alone.

Densification. In the backwards pass of each iteration, we return the blended $L _ { 1 }$ errors as defined in the work of Rota Bulo et al. [ \` 43]. These blended errors are a perprimitive heuristic quantity where a higher error suggests that the primitive should be split. We accumulate these errors across iterations by adding them, rather than performing primitive-wise maximums. Inspired by 3DGS-MCMC [23], we normalize the blended errors and use them as probability weights to sample 5% of the current primitives for splitting.

Our splitting operation aims to preserve the textured surface. Say we are splitting surfel i. We will assume $\sigma _ { i , 1 } \geq \sigma _ { i , 2 }$ for convenience. The surfel is then split into two primitives both with scales $\left( \frac { \sigma _ { i , 1 } } { 2 } , \sigma _ { i , 2 } \right)$ and with means $\begin{array} { r } { \mu _ { i } - \frac { \sigma _ { i , 1 } } { 2 } \mathbf { v } _ { i , 1 } } \end{array}$ and $\begin{array} { r } { \mu _ { i } + \frac { \sigma _ { i , 1 } } { 2 } \mathbf { v } _ { i , 1 } } \end{array}$ , respectively. All other properties of the new primitives, including the Î³ values, are copied from the original one. For $\gamma _ { i } \to \infty$ , this splits the quad perfectly in half along its long axis.

## 9. Dataset Details

We capture four scenes, GRAFFITI, GROCERY, TABLE, and TRIPOD. All images were captured using a Canon EOS Rebel T7 equipped with an EF-S 18-55mm f/3.5-5.6 IS II zoom lens set to a focal length of approximately 35 mm.

To ensure photometric consistency, we configured camera settings by capturing preliminary reference shots from multiple angles at each scene. We then locked these settings for the capture of each scene, fixing ISO, white balance, zoom level, and exposure per scene. Auto-focus was enabled. These scenes were selected specifically to evaluate reconstruction performance on high-frequency textures. A similar dataset was captured by Chao et al. [7], but the data was not made public for licensing reasons.

For the indoor scenes (GROCERY, TABLE, TRIPOD), we captured 360-degree concentric trajectories centered on the object of interest. For the outdoor scene (GRAFFITI), which lacks a singular central subject, we densely captured the region of interest. It took approximately 30 minutes to capture up to 300 images for each scene. After capturing, we manually removed images with any noticeable motion blur. We include brief descriptions of each scene below. The dataset will be made public.

â¢ GRAFFITI (254 images). This outdoor scene depicts an alleyway where a concrete driveway separates a fence from a graffiti-covered building.

â¢ GROCERY (278 images). Captured indoors, this scene centers on a $0 . 5 \times 1$ meter dining table covered in kitchen rags. On top of the rags, the table features a variety of grocery items with text-heavy labels as well as an open book.

â¢ TABLE (165 images). This scene was taken inside a single room, with the camera focusing on a small table around 1 meter in length. Two open books, a newspaper, and a deck of cards are laid out on the table, along with miscellaneous other objects.

â¢ TRIPOD (210 images). Arranged inside a small room, this scene consists of magazines, markers, and various boxes surrounding a large poster propped up on a tripod.

## 10. Experiment Details

## 10.1. Baseline Details

We evaluated the baselines at various settings in order to obtain data points at a varying amount of primitives and memory. For 3DGS [22], 2DGS [20], and TriSplat [19], there is no fine control over the number of primitives. We use their default densification settings with no notable failures. For methods that provide separate hyperparameter sets for outdoor and indoor scenes, we used the indoor settings for our custom scenes. Additional details are provided below.

3DGS-MCMC [23]. The original paper uses the bicubic filter from the PIL library to downscale the images, specifically through the resolution flag. This produces smoother images than the ImageMagick downscaling which was used by Barron et al. [2] on the Mip-NeRF360 dataset. As a result, scripts yield slightly better photometric results when using the resolution flag for downscaling rather than using the images flag to specify the image directories downscaled by ImageMagick. For fairness, we re-evaluated 3DGS-MCMC using the same downscaling strategy as in the original 3D Gaussian Splatting work. When training for a low amount of splats, we initialize using farthest point sampling with the same number of points as our initialization [27].

BBSplat [49]. For BBSplat, we found that their scripts are run at a different resolution than the majority of Gaussian splatting works. We re-ran the baseline using the same resolution as in 3D Gaussian Splatting. Additionally, we set a maximum primitive cap of 400K and limited the number of SfM points for initialization to 150K. For all scenes, we use their sky-box initialization that adds another 10K points in the background. For BBSplat, we were unable to reproduce the results for the STUMP scene when run at the standard resolution, as it achieved a significantly lower PSNR. For this scene only, we instead ran the script at their resolution and rescaled the renders to our resolution with Lanczos resampling. We evaluate the storage memory of BBSplat after it performs dictionary-based compression on the textures.

<table><tr><td></td><td colspan="5">Mip-NeRF360 Indoor</td><td colspan="5">Mip-NeRF360 Outdoor</td><td colspan="5">Custom</td></tr><tr><td></td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Memory</td><td>#Prim.</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Memory</td><td>#Prim.</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Memory</td><td>#Prim.</td></tr><tr><td>3DGS</td><td>30.40</td><td>0.920</td><td>0.190</td><td>290MB</td><td>1.25M</td><td>24.64</td><td>0.731</td><td>0.234</td><td>900MB</td><td>3.88M</td><td>26.65</td><td>0.858</td><td>0.224</td><td>1630MB</td><td>7.03M</td></tr><tr><td>2DGS</td><td>30.41</td><td>0.917</td><td>0.192</td><td>180MB</td><td>0.78M</td><td>24.34</td><td>0.725</td><td>0.246</td><td>710MB</td><td>3.06M</td><td>25.98</td><td>0.831</td><td>0.267</td><td>420MB</td><td>1.80M</td></tr><tr><td>MiniSplatting</td><td>30.43</td><td>0.922</td><td>0.188</td><td>90MB</td><td>0.39M</td><td>24.72</td><td>0.741</td><td>0.242</td><td>130MB</td><td>0.57M</td><td>24.88</td><td>0.803</td><td>0.297</td><td>100MB</td><td>0.42M</td></tr><tr><td>TriSplat</td><td>30.80</td><td>0.928</td><td>0.160</td><td>530MB</td><td>2.25M</td><td>24.24</td><td>0.722</td><td>0.217</td><td>1080MB</td><td>4.56M</td><td>24.28</td><td>0.780</td><td>0.275</td><td>640MB</td><td>2.73M</td></tr><tr><td>NeST-Splat</td><td>30.35</td><td>0.909</td><td>0.176</td><td>280MB</td><td>0.33M</td><td>23.74</td><td>0.703</td><td>0.242</td><td>580MB</td><td>1.64M</td><td>25.39</td><td>0.824</td><td>0.246</td><td>900MB</td><td>3.24M</td></tr><tr><td>3DGS-MCMC (400K)</td><td>30.13</td><td>0.907</td><td>0.225</td><td>90MB</td><td>0.40M</td><td>24.03</td><td>0.672</td><td>0.341</td><td>90MB</td><td>0.40M</td><td>25.61</td><td>0.807</td><td>0.327</td><td>93MB</td><td>0.40M</td></tr><tr><td>3DGS-MCMC (1M)</td><td>31.16</td><td>0.926</td><td>0.189</td><td>230MB</td><td>1.00M</td><td>24.74</td><td>0.722</td><td>0.272</td><td>230MB</td><td>1.00M</td><td>25.96</td><td>0.823</td><td>0.295</td><td>232MB</td><td>1.00M</td></tr><tr><td>3DGS-MCMC</td><td>31.34</td><td>0.929</td><td>0.181</td><td>340MB</td><td>1.45M</td><td>25.04</td><td>0.744</td><td>0.232</td><td>830MB</td><td>3.57M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DBS (400K)</td><td>31.25</td><td>0.924</td><td>0.192</td><td>40MB</td><td>0.40M</td><td>24.11</td><td>0.692</td><td>0.312</td><td>40MB</td><td>0.40M</td><td>26.39</td><td>0.828</td><td>0.305</td><td>40MB</td><td>0.40M</td></tr><tr><td>DBS (1M)</td><td>31.98</td><td>0.933</td><td>0.174</td><td>110MB</td><td>1.00M</td><td>24.56</td><td>0.727</td><td>0.257</td><td>110MB</td><td>1.00M</td><td>26.72</td><td>0.844</td><td>0.273</td><td>110MB</td><td>1.00M</td></tr><tr><td>DBS</td><td>32.18</td><td>0.935</td><td>0.169</td><td>160MB</td><td>1.50M</td><td>24.83</td><td>0.745</td><td>0.209</td><td>470MB</td><td>4.40M</td><td>26.22</td><td>0.842</td><td>0.259</td><td>270MB</td><td>2.50M</td></tr><tr><td>BBSplat (40K)</td><td>28.85</td><td>0.886</td><td>0.228</td><td>30MB</td><td>0.04M</td><td>22.39</td><td>0.592</td><td>0.359</td><td>30MB</td><td>0.04M</td><td>24.98</td><td>0.794</td><td>0.321</td><td>40MB</td><td>0.04M</td></tr><tr><td>BBSplat (100K)</td><td>30.34</td><td>0.918</td><td>0.184</td><td>80MB</td><td>0.10M</td><td>23.17</td><td>0.647</td><td>0.304</td><td>80MB</td><td>0.10M</td><td>25.84</td><td>0.829</td><td>0.266</td><td>80MB</td><td>0.10M</td></tr><tr><td>BBSplat (400K)</td><td>31.16</td><td>0.926</td><td>0.170</td><td>260MB</td><td>0.40M</td><td>23.71</td><td>0.675</td><td>0.273</td><td>260MB</td><td>0.40M</td><td>26.57</td><td>0.844</td><td>0.225</td><td>260MB</td><td>0.40M</td></tr><tr><td>Nexels (40K,  $T = 2 ^ { 1 9 } )$ </td><td>29.19</td><td>0.894</td><td>0.199</td><td>40MB</td><td>0.04M</td><td>23.26</td><td>0.641</td><td>0.294</td><td>40MB</td><td>0.04M</td><td>26.20</td><td>0.825</td><td>0.261</td><td>40MB</td><td>0.04M</td></tr><tr><td>Nexels (40K)</td><td>29.38</td><td>0.900</td><td>0.190</td><td>80MB</td><td>0.04M</td><td>23.25</td><td>0.647</td><td>0.282</td><td>80MB</td><td>0.04M</td><td>26.50</td><td>0.836</td><td>0.240</td><td>80MB</td><td>0.04M</td></tr><tr><td>Nexels (100K)</td><td>29.94</td><td>0.906</td><td>0.183</td><td>90MB</td><td>0.10M</td><td>23.81</td><td>0.675</td><td>0.258</td><td>90MB</td><td>0.10M</td><td>26.62</td><td>0.844</td><td>0.237</td><td>90MB</td><td>0.10M</td></tr><tr><td>Nexels (400K)</td><td>30.94</td><td>0.921</td><td>0.164</td><td>160MB</td><td>0.40M</td><td>24.48</td><td>0.707</td><td>0.230</td><td>160MB</td><td>0.40M</td><td>27.04</td><td>0.851</td><td>0.222</td><td>160MB</td><td>0.40M</td></tr><tr><td>Nexels (1M)</td><td>31.32</td><td>0.926</td><td>0.155</td><td>310MB</td><td>1.00M</td><td>24.77</td><td>0.722</td><td>0.215</td><td>310MB</td><td>1.00M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 5. Quantitative results. We show full results of the data points from Figure 5 of the main paper, as well as additional experiment configurations.
<table><tr><td rowspan="2"></td><td colspan="4">Mip-NeRF360 Indoor</td><td colspan="4">Mip-NeRF360 Outdoor</td><td colspan="2">Tanks &amp; Temples</td><td colspan="4"></td></tr><tr><td>BNS</td><td>CNT</td><td>Ktc</td><td>RM</td><td>Bcc</td><td>FLW</td><td>GRD</td><td>STM</td><td>TRH</td><td>TRN</td><td>TRC</td><td>GRF GRC</td><td>TBL</td><td>TRP</td></tr><tr><td>Nexels (40K,  $T = 2 ^ { 1 9 } )$ </td><td>29.67</td><td>27.44</td><td>29.41</td><td>30.23</td><td>23.36</td><td>19.85</td><td>25.48</td><td>25.08 22.54</td><td></td><td></td><td>24.86</td><td>26.93</td><td>27.88</td><td>25.14</td></tr><tr><td>Nexels (40K)</td><td>29.98</td><td>27.61</td><td>29.62</td><td>30.32</td><td>23.44</td><td>19.79</td><td>25.59 25.11</td><td>22.33</td><td></td><td></td><td>25.38</td><td>27.16</td><td>27.95</td><td>25.50</td></tr><tr><td>Nexels (100K)</td><td>30.81</td><td>28.13</td><td>30.32</td><td>30.51</td><td>24.05</td><td>20.38</td><td>26.23 25.88</td><td>22.53</td><td></td><td>-</td><td>25.50</td><td>27.51</td><td>27.73</td><td>25.74</td></tr><tr><td>Nexels (400K)</td><td>31.92</td><td>28.95</td><td>31.22</td><td>31.66</td><td>24.72</td><td>21.19</td><td>26.96 26.56</td><td>22.99</td><td>21.50</td><td>25.61</td><td>25.98</td><td>28.07</td><td>27.98</td><td>26.12</td></tr><tr><td>Nexels (1M)</td><td>32.47</td><td>29.26</td><td>31.59</td><td>31.97</td><td>25.04</td><td>21.48</td><td>27.36 27.01</td><td>22.97</td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Nexels (40K,  $T = 2 ^ { 1 9 } )$ </td><td>0.907</td><td>0.875</td><td>0.895</td><td>0.899</td><td>0.645</td><td>0.488</td><td>0.784 0.688</td><td>0.600</td><td></td><td>-</td><td>0.743</td><td>0.853</td><td>0.901</td><td>0.802</td></tr><tr><td>Nexels (40K)</td><td>0.913</td><td>0.880</td><td>0.901</td><td>0.903</td><td>0.657</td><td>0.492</td><td>0.793 0.691</td><td>0.600</td><td></td><td></td><td>0.767</td><td>0.857</td><td>0.905</td><td>0.815</td></tr><tr><td>Nexels (100K)</td><td>0.923</td><td>0.892</td><td>0.910</td><td>0.899</td><td>0.690</td><td>0.530</td><td>0.817</td><td>0.728 0.607</td><td></td><td></td><td>0.776</td><td>0.864</td><td>0.904</td><td>0.831</td></tr><tr><td>Nexels (400K)</td><td>0.935</td><td>0.905</td><td>0.921</td><td>0.922</td><td>0.730</td><td>0.577</td><td>0.842 0.764</td><td>0.624</td><td>0.802</td><td>0.880</td><td>0.789</td><td>0.878</td><td>0.906</td><td>0.829</td></tr><tr><td>Nexels (1M)</td><td>0.942</td><td>0.911</td><td>0.927</td><td>0.925</td><td>0.749</td><td>0.595</td><td>0.855 0.781</td><td>0.629</td><td></td><td></td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Nexels (40K,  $T = 2 ^ { 1 9 } )$ </td><td>0.211</td><td>0.214</td><td>0.154</td><td>0.218</td><td>0.304</td><td>0.376</td><td>0.178</td><td>0.290 0.322</td><td></td><td>-</td><td>0.338</td><td>0.264</td><td>0.164</td><td>0.277</td></tr><tr><td>Nexels (40K)</td><td>0.199</td><td>0.205</td><td>0.146</td><td>0.210</td><td>0.288</td><td>0.364</td><td>0.164</td><td>0.282 0.309</td><td></td><td></td><td>0.302</td><td>0.248</td><td>0.151</td><td>0.261</td></tr><tr><td>Nexels (100K)</td><td>0.188</td><td>0.192</td><td>0.136</td><td>0.215</td><td>0.261</td><td>0.334</td><td>0.145 0.249</td><td>0.302</td><td></td><td>-</td><td>0.318</td><td>0.234</td><td>0.146</td><td>0.251</td></tr><tr><td>Nexels (400K)</td><td>0.170</td><td>0.176</td><td>0.123</td><td>0.186</td><td>0.227</td><td>0.298</td><td>0.124 0.216</td><td>0.286</td><td>0.197</td><td>0.112</td><td>0.274</td><td>0.222</td><td>0.146</td><td>0.245</td></tr><tr><td>Nexels (1M)</td><td>0.161</td><td>0.167</td><td>0.114</td><td>0.179</td><td>0.208</td><td>0.281</td><td>0.110 0.197</td><td>0.277</td><td></td><td></td><td>-</td><td></td><td></td><td>,</td></tr></table>

Table 6. Per-scene results. We provide per-scene PSNR, SSIM, and LPIPS scores for our nexels at various settings. Scene names are abbreviated to their first three consonants.

DBS [30]. Like 3DGS-MCMC, the results in the original DBS paper were obtained using downsampling with the resolution flag. We re-ran the method using Mip-NeRF360âs downscaled images. Furthermore, the original DBS uses a scene-specific number of iterations that explicitly checks the test set as a stopping condition. For fairness, we use the same number of iterations for all scenes (i.e. 30K). On our custom scenes, we set the primitive cap for DBS to 3 million primitives for GRAFFITI, GROCERY, and TRIPOD, and to 1 million for TABLE. We use 1 million primitives for TABLE because using 3 million resulted in unstable behavior, and the PSNR dropped significantly.

<!-- image-->

<!-- image-->  
(b)

Figure 8. Artifacts. (a) We show a render of the GRAFFITI scene from a view outside of both the training and test sets. The neural field produces noise in unseen regions. (b) We show a zoom-in of the ground truth (top) and render (bottom) of the ROOM scene. Even small amounts of motion blur are destructive for fine details, and our reconstruction cannot properly capture the text.
<table><tr><td>K</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>0</td><td>24.81</td><td>0.663</td><td>0.415</td></tr><tr><td>1</td><td>25.80</td><td>0.750</td><td>0.254</td></tr><tr><td>2</td><td>25.98</td><td>0.759</td><td>0.241</td></tr><tr><td>3</td><td>25.76</td><td>0.756</td><td>0.247</td></tr><tr><td>4</td><td>25.44</td><td>0.747</td><td>0.258</td></tr></table>

Table 7. Texture limit. We train 40K primitives at different texture limits, K, on the Mip-NeRF360 scenes.

MiniSplatting [12]. We ran MiniSplatting with their recommended indoor and outdoor settings for each respective scene. To obtain the two low memory comparisons shown in Figure 7, we set the sampling factor to 0.2 and 0.25, respectively.

NeST-Splatting [59]. The densification strategy in NeST-Splatting does not allow setting a maximum number of primitives. On our custom GRAFFITI scene, this resulted in an out-of-memory error even with a 48GB GPU. The results of NeST-Splatting on the custom dataset reported in the paper were obtained by doubling the interval between densification steps for the GRAFFITI scene alone. To attain the models shown in Figure 6 of the main paper, with exactly 40K primitives, we added a maximum number to their densification. We initialize from a COLMAP point cloud sampled using farthest point sampling [27] to 20K points, matching our initialization. The storage memory was computed using 16-bit floating point numbers for the hash-grid parameters.

## 10.2. Additional Results

We list the data points from Figure 5 of the main paper in Table 5. We include per-scene metrics for our method at

<!-- image-->  
(a)

<!-- image-->  
(b)

Figure 9. Geometric inaccuracies. (a) At a low number of primitives (40K) trained on the BONSAI scene, the nexels miss thin structures. (b) With 400K primitives trained on the BICYCLE scene, the background surfels optimize to an incorrect geometry and the textures learn to blur the details.
<table><tr><td>Routine</td><td>Indoor</td><td>Outdoor</td></tr><tr><td>Collection Pass</td><td>11.8 ms</td><td>8.1 ms</td></tr><tr><td>Texturing Pass</td><td>12.7 ms</td><td>8.4 ms</td></tr><tr><td>Total</td><td>24.5 ms</td><td>16.4 ms</td></tr></table>

Table 8. Detailed timings. We show timings for each pass averaged across the Mip-NeRF360 indoor and outdoor scenes with 400K primitives.

different settings in Table 6.

We also conduct an experiment to determine the best value of K in Table 7. The metrics sharply improve when adding in any texture at all (i.e. going from K = 0 to K = 1) and reach a peak at K = 2. Surprisingly, increasing K beyond 2 leads to degradation in quality. Empirically, we observe noise in the background of test renders at higher values of K, suggesting that primitives which were never textured at training views are textured at test views.

## 11. Further Discussion

## 11.1. Texturing Strategy

Limiting the number of textured primitives to K has a number of consequences which we outline. Firstly, the rendering speed becomes a compromise between that of rendering with per-primitive colours and that of texturing every primitive. The runtime of the texturing step is determined by the number of pixels and the number of per-pixel primitives that are textured, which is fixed.

In the forwards pass, the textured surfels determine the majority of the appearance. However, the non-textured surfels still make small improvements to the render in cases where the textureâs appearance is not sufficient. This is useful around areas with strong reflections and transparent materials, as our rendering formulation does not encapsulate

full lighting effects.

In the backwards pass, allowing non-textured primitives to contribute to the render allows them to receive gradient flow during back-propagation. This is important for early training iterations where the gradient moves primitives towards the underlying scene geometry, regardless of whether the primitive was textured. The neural texture does not need to be rendered at multiple overlapping primitives to guide its optimization. Thanks to the volumetric nature of the neural field, its gradient flow is the same whether it is rendered by several coinciding surfels or by a single surfel with the same total weight.

## 11.2. Metrics

We primarily show LPIPS throughout the paper as we find that it best corresponds to the reconstruction quality of fine texture details, which is a weakness of Gaussian splatting. It also generally corresponds better to human preference, as that was its original purpose [58]. We note that the LPIPS formula used by virtually all Gaussian splatting works has a minor discrepancy from the one used by Zhang et al. [58]. Specifically, the official 3DGS repository scales its images to [0, 1] before computing the LPIPS, while the LPIPS library used expects images scaled to [â1, 1]. Nonetheless, the scores returned using this appear to be reasonable for comparison purposes so long as all baselines are calculated in the same way. To be consistent with the rest of the field, we use the same LPIPS calculation as 3DGS.

## 11.3. Limitations

As mentioned in the main text, nexels are not able to run at interactive rates on mobile devices or low-end GPUs. We show timings of the individual rendering passes in Table 8. The neural texture step currently uses tensor core operations and would be much slower if implemented in WebGL or OpenGL.

Figure 8 includes qualitative examples of the limitations mentioned in the main paper, particularly unseen regions and motion blur. One more limitation is the geometric accuracy of the representation, particularly in background regions. If the geometry does not optimize or densify correctly, then the neural texture is of little help. We show two failure cases in Figure 9. In the first, the sparse geometries are not able to optimize towards thin structures. In the second, the background geometries optimize to a local minimum, resulting in blurry textures.