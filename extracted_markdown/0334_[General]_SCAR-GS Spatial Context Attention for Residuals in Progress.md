# SCAR-GS: Spatial Context Attention for Residuals in Progressive Gaussian Splatting

Revilla Diego1,2 Suresh Pooja1 Bhojan Anand1 Wei Tsang Ooi1

1National University of Singapore, Singapore

2University of Deusto, Spain

diego.r@opendeusto.es

dcsab@nus.edu.sg, dcsooiwt@nus.edu.sg

poojasuresh@u.nus.edu.sg

## Abstract

Recent advances in 3D Gaussian Splatting have allowed for real-time, high-fidelity novel view synthesis. Nonetheless, these models have significant storage requirements for large and medium-sized scenes, hindering their deployment over cloud and streaming services. Some of the most recent progressive compression techniques for these models rely on progressive masking and scalar quantization techniques to reduce the bitrate of Gaussian attributes using spatial context models. While effective, scalar quantization may not optimally capture the correlations of high-dimensional feature vectors, which can potentially limit the ratedistortion performance.

In this work, we introduce a novel progressive codec for 3D Gaussian Splatting that replaces traditional methods with a more powerful Residual Vector Quantization approach to compress the primitive features. Our key contribution is an auto-regressive entropy model, guided by a multi-resolution hash grid, that accurately predicts the conditional probability of each successive transmitted index, allowing for coarse and refinement layers to be compressed with high efficiency.

## 1. Introduction

Gaussian Splatting [19] marks a significant advancement in real-time computer graphics and scene reconstruction, enabling real-time photorealistic rendering and novel-view synthesis over traditional Neural Radiance Fields (NeRFs) [29], as representing scenes as a collection of scattered Gaussians offers an alternative, discretized approach to deep neural network inference. However, this explicit representation comes at a cost: the storage required for the millions of Gaussian attributes can be substantial, often reaching hundreds of megabytes per scene [1, 2]. This large memory footprint presents a significant barrier to the widespread deployment of these models, particularly on resource-constrained platforms such as mobile devices or web browsers.

To address this challenge, state-of-the-art compression techniques have been developed, [9, 10, 14, 16, 26, 30, 42, 47], However, most methods optimize purely for maximum compression efficiency at the expense of streamability or prioritize progressiveness at the expense of reconstruction quality and coding efficiency. Only a select few works [8, 11, 33, 34, 49] explore how to make these methods suitable for progressiveness in order not to bottleneck transmission over the RAM or network.

While context modeling has demonstrated remarkable results in compressing 3D Gaussian Splats [9, 10, 27, 38], these approaches predominantly yield single-rate representations that are ill-suited for progressive streaming. For instance, HAC [9] and its successor HAC++ [10] leverage the neural anchors introduced in Scaffold-GS [27] to learn sparse hash grids that capture spatial contextual relationships. Similarly, ContextGS [38] utilizes an autoregressive model to reuse decoded anchors for predicting finer details, and CAT-3DGS [44] employs multiscale triplanes to model inter-anchor correlations. Recently, HEMGS [24] proposed a hybrid entropy model combining variable-rate predictors with hyperpriors for flexible rate control. However, these methods primarily address spatial redundancy within a static reconstruction, neglecting a hierarchical quality representation required for progressive transmission.

Conversely, methods explicitly designed for progressivity often sacrifice reconstruction quality or compression efficiency. LapisGS [34] constructs a layered structure of cumulative Gaussians to incrementally increase rendering resolution, while GoDe [33] organizes primitives into hierarchical layers based on visibility heuristics. A critical limitation of these approaches is their reliance on limiting the quantity of Gaussians to enable progressiveness, rather than refining the quality of existing features. PCGS [8], however, does introduce a progressive encoding framework that utilizes trit-plane encoding and masking to transmit Gaussian attributes in importance order, allowing for both quantity and quality control over successive layers. However, because it uses a scalar approach to quantization, it might not harness the whole potential of inter-related dimensionality and feature-level quantization.

<!-- image-->  
Figure 1. Visual overview of the SCAR-GS pipeline. The system uses a hierarchical neural representation (Left) encoded via dualcodebook Residual Vector Quantization (Bottom). A spatially-aware autoregressive entropy model (Top) predicts indices for arithmetic coding. The decoder reconstructs features progressively to render Gaussians of increasing fidelity (Right).

Parallel to these advancements, Vector Quantization has also been used in different techniques for 3DGS compression [14, 23, 26, 32, 37]. Most notably, CompGS [26] reduces model size by quantizing Gaussian attributes into codebooks. However, this approach doesnât make use of spatial-level similarities that allow for further compression, and is not suitable for progressive encoding either.

SCAR-GS refines feature quality rather than attribute precision or primitive quantity. In this work, we propose a novel progressive codec that fundamentally changes the feature representation, moving from independent scalar quantization to a more powerful, holistic Residual Vector Quantization (RVQ) [25] scheme. As depicted in figure 1, our key contribution is an auto-regressive entropy model that operates on the sequence of RVQ indices. Guided by both a spatial hash-grid context and the previously decoded feature information, our model predicts the conditional probability of each successive codebook index. This architecture allows a scene to be represented as a base layer of coarse features followed by a series of increasingly detailed refinement layers.

In summary, our contributions are as follows:

1. We propose SCAR-GS: Spatial Context Attention for Residuals, a progressive 3DGS codec that attends to residual hierarchy to conditionally minimize the en-

tropy of the sent residuals.

2. We introduce hierarchical vector quantization for 3D Gaussian Splatting and propose data-limiting strategies to prevent RVQ-VAE parameter overhead.

3. We demonstrate through extensive evaluation on standard benchmarks that our RVQ-VAE based approach achieves a similar rate-perception-distortion trade-off in perceptual metrics (SSIM [40] and LPIPS [45]) compared to the state of the art, while requiring significantly reduced storage for comparable perceptual quality.

## 2. Related Work

## 2.1. Gaussian Splatting

3D Gaussian Splatting represents the scene using a myriad of 3D Gaussians, with each its own shape and color attributes. Unlike NeRFs, 3DGS enables high-speed rasterization [50] as it doesnât require network evaluation; however, it results in a significant memory footprint, often reaching hundreds of megabytes per scene. To mitigate this, initial compression methods such as LightGaussian [14], LP-3DGS [47] employ significance metrics to remove Gaussians that contribute minimally to the final image. Likewise, compaction methods such as GaussianSpa [46] try to sparsify Gaussian scenes in order to reduce unnecesary duplicates. However, both approaches do have an impact on the final image quality.

## 2.2. Encoded Representations

Encoded Representations allow us to represent information into a compact, latent format from which we can recover the previous information [39]. Deep representation learning has evolved from continuous latent variable models, such as Variational Autoencoders (VAEs) [20], to discrete counterparts like Vector Quantized VAEs (VQ-VAEs) [36]. By mapping inputs to a finite codebook of learnable embeddings, VQ-VAEs facilitate efficient storage in the latent space. To address the limited expressivity of a single discretized pass, Residual Vector Quantization (RVQ) [43] extends this paradigm by recursively quantizing the residual errors across multiple stages, effectively decomposing the signal into a coarse base and a series of high-frequency refinements. Finally, RVQ-VAEs [22] introduce latent space representation to Residual Vector Quantization.

In the case of Gaussian Splatting, this technique allows us to represent the attributes of one or several Gaussians in a dimensionality-reduced representation that may be regressed into fully reconstructed attributes, while greatly reducing the memory footprint of the representation. In Gaussian Splatting Compression, Scaffold-GS [27] was the first to introduce anchor-level representation learning for nearby Gaussians, which was a keystep in compression development. On top of that, ContextGS [38] provides a logical step up, deducing higher-order, fine-detail anchors by regressing a coarse set.

## 2.3. Autoregressive Entropy Modeling

The efficiency of any neural codec relies heavily on reducing the entropy distribution used by the arithmetic coder. In 3DGS compression, context modeling has proven effective for reducing redundancy [9, 10, 27, 38]. However, these existing context models are typically designed for static, single-rate decoding, and they do not account for the hierarchical aspect of progressive streaming, where the context must evolve as new refinement layers are received. While PCGS [8] does indeed use entropy modelling, its strategy is non-regressive, as it doesnât carry the previous layer information to reduce entropy in layers of increasing detail.

## 3. Methodology

## 3.1. Preliminaries

Based on previous works [8, 10, 27], we represent the volumetric scene S as a sparse point cloud of reduced N neural anchors which cluster nearby Gaussians, denoted as $\mathcal { A } = \{ \mathbf { x } _ { i } , \mathbf { o } _ { i } , \mathbf { s } _ { i } , \mathbf { f } _ { i } , \} _ { i = 1 } ^ { N }$ . Here, $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ represents the anchor position, $\mathbf { o } _ { i }$ represents position offsets, $\mathbf { s } _ { i }$ denotes the scaling factors, and $\mathbf { f } _ { i } \in \mathbb { R } ^ { \bar { D } }$ is a high-dimensional latent feature vector encapsulating the local appearance and geometry.

Unlike traditional explicit representations, the covariance, scaling, and rotation matrices and opacity values are not explicitly stored. Instead, we employ a set of lightweight MLPs to expand the Gaussian attributes of color and opacity values, and scale and rotation matrices, given the viewing direction v and the camera distance d,These learnt functions expand the feature $\mathbf { f } _ { i }$ into the attributes required for Gaussian rasterization:

$$
\begin{array} { c } { \alpha = \Phi _ { \alpha } ( \mathbf { f } _ { i } , \mathbf { v } , d ) } \\ { \mathbf { c } = \Phi _ { \mathbf { c } } ( \mathbf { f } _ { i } , \mathbf { v } , d ) } \\ { ( \mathbf { S } , \mathbf { R } ) = \Phi _ { c o v } ( \mathbf { f } _ { i } , \mathbf { v } , d ) } \end{array}
$$

where Î± is opacity, c is view-dependent color, and S, R are the covariance scaling and rotation matrices, respectively.

## 3.2. Residual Vector Quantization

We propose to quantize the latent features in the anchors fi using Residual Vector Quantization (RVQ) [43]. RVQ decomposes feature complexity into a sequence of progressively lower-entropy distributions, making each stage more amenable to accurate conditional probability estimation than a single large codebook. Concretely, RVQ progressively minimizes the reconstruction error across M quantization stages using multiple codebooks.

Standard RVQ implementations often use a single codebook or distinct codebooks for every layer. However, we observed that the initial quantization step captures a highvariance, sparse signal; while subsequent steps capture residual approximations that tend to be similar. Therefore, we adopt a dual Codebook strategy comprising two distinct codebooks to reduce parameter count while maintaining high fidelity: a coarse Codebook $\mathcal { C } _ { c o a r s e }$ and a Shared Residual Codebook $\mathcal { C } _ { r e s i d u a l }$ .The quantization process for a feature vector z proceeds iteratively. For the first stage $( m = 1 )$ , we utilize the base quantizer:

$$
\mathbf { z } _ { 1 } = \underset { \mathbf { e } \in \mathcal { C } _ { c o a r s e } } { \arg \operatorname* { m i n } } \| \mathbf { z } - \mathbf { e } \| , \quad \mathbf { r } _ { 1 } = \mathbf { z } - \mathbf { z } _ { 1 }
$$

For all subsequent stages $m \in \{ 2 , \ldots , M \}$ , we utilize the same shared residual quantizer to approximate the error from the previous step:

$$
\mathbf { z } _ { m } = \underset { \mathbf { e } \in \mathcal { C } _ { r e s } } { \arg \operatorname* { m i n } } \| \mathbf { r } _ { m - 1 } - \mathbf { e } \| , \quad \mathbf { r } _ { m } = \mathbf { r } _ { m - 1 } - \mathbf { z } _ { m }
$$

The reconstructed feature zË is the summation of the quantized vectors: $\begin{array} { r } { \hat { \mathbf { z } } = \mathbf { z } _ { 1 } + \sum _ { m = 2 } ^ { M } \mathbf { z } _ { m } } \end{array}$ . We can think of this approach as first approaching the coarse materials of the Gaussians and adding local details progressively on successive stages.

## 3.3. The Rotation Trick for gradient propagation

Since VQ is non-differentiable, we typically rely on the Straight-Through Estimator (STE) [6] where gradients bypass the discretization layer. However, this approach discards critical information about the reconstructed feature locality with respect to the original, potentially leading to poor semantic representation after quantization. To address this, we made use of the Rotation Trick [15] for gradient propagation. Instead of simply passing the gradients from the decoder output to encoder input, we model the relationship between them as a smooth linear transformation involving a rotation and rescaling. During the forward pass, we identify the transformation R such that e = Rz, where e is the quantized feature, and z is the encoder latent output. During backpropagation, this transformation R is treated as a constant. Consequently, the gradients flowing back to the encoder are modulated by the relative magnitude and angle between the encoder output and the codebook vector. This method injects information about the quantization geometry into the backward pass, improving codebook utilization and reducing quantization error compared to standard STE.

<!-- image-->  
Figure 2. Comparisons of the different progressive layers on the Flower scene from the Mip-NeRF360 dataset [4].

## 3.4. Spatially-Aware Autoregressive Entropy Modeling

To compress the stream of discrete indices k = $\{ k _ { 1 } , \dots , k _ { M } \}$ resulting from the RVQ, we perform lossless arithmetic coding [28]. The compression ratio is bounded by the cross-entropy between the true distribution of indices, which is a one-hot encoding of the real index over N possible codewords, and the predicted distribution P (k). We propose a hybrid entropy model that conditions the probability of the current index $k _ { m }$ on a fused context of local spatial geometry and the sequence of previously decoded residuals.

## 3.5. Spatial-Query Attention Mechanism

We model the dependency between quantization levels using a multi-layer Gated Recurrent Unit (GRU) [12] to predict the probability distribution of the next codebook entry based on the history of past indices $\left( k _ { < m } \right)$ . To account for local variations, we introduce a Spatial-Query Attention module. We treat the static spatial embedding as the Query (Q) and the sequence of GRU hidden states as the Keys (K) and Values (V ). The attention [3, 17] context $\mathbf { c } _ { a t t n }$ is computed as:

$$
Q = W _ { Q } { \bf h } _ { s p a t i a l } , \quad K = W _ { K } { \bf G } _ { < m } , \quad V = W _ { V } { \bf G } _ { < m }
$$

$$
\mathbf { c } _ { a t t n } = \mathrm { S o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { m o d e l } } } } \right) V
$$

where $\mathbf { G } _ { < m }$ represents the sequence of GRU hidden states corresponding to the previous indices. The final probability distribution is predicted via an MLP which receives as input the spatially-aware context:

$$
P ( k _ { m } | k _ { < m } , \mathbf { x } ) = \operatorname { S o f t m a x } \big ( \mathbf { M L P } ( \mathbf { c } _ { a t t n } ) \big )
$$

To model spatial embeddings, we employ a multi-resolution learnable spatial hash grid [31]. For an anchor at position x, we retrieve a spatial embedding $\mathbf { h } _ { s p a t i a l }$ by employing bicubic interpolation on the grid at xË, where zË is the anchor position in world space, as proposed by HAC [9].

## 3.6. Optimization Objective

## 3.6.1 Rendering Loss

The main objective of any 3D Gaussian training framework is to minimize the rendering error between the rendered image and the ground truth.

$$
\mathcal { L } _ { s c e n c e } = ( 1 - \lambda _ { s s i m } ) \mathcal { L } _ { 1 } ( I _ { r e n d e r } , I _ { g t } ) + \lambda _ { s s i m } \mathrm { S S I M } ( I _ { r e n d e r } , I _ { g t } ) ^ { \mathrm { I } }
$$

## 3.6.2 Entropy Loss

In the final fine-tuning stages, we enable the autoregressive entropy model. The rate loss $\mathcal { L } _ { r a t e }$ is added to the scene optimization objective to minimize the total bit-cost. This includes the loss for the history-conditioned residual index entropy auto-regression. Given the sequence of groundtruth quantization indices $\mathbf { k } = \{ k _ { 1 } , k _ { 2 } , \dots , k _ { M } \}$ , the loss minimizes the negative log-likelihood of each index $k _ { m }$ conditioned on its history $k _ { < m }$ and the spatial context:

$$
\mathcal { L } _ { f e a t } = \mathbb { E } \left[ - \sum _ { m = 1 } ^ { M } \log _ { 2 } P _ { \psi } ( k _ { m } \mid k _ { < m } , \mathbf { h } _ { s p a t i a l } ) \right]
$$

where $P _ { \psi }$ is the probability distribution predicted by the GRU model. For the geometry attributes, we adopt the bitrate loss formulation proposed in PCGS [8], which employs trit-plane quantization for progressive encoding.

$$
\mathcal { L } _ { r a t e } = \mathcal { L } _ { f e a t } + \mathcal { L } _ { s c a l e } + \mathcal { L } _ { o f f s e t }
$$

## 3.6.3 Quantization Loss

The VQ-VAE parameters are updated by a separate, dedicated optimizer. The goal of this optimizer is to minimize the distance between the continuous feature and the quantized one. The RVQ-VAE objective $\mathcal { L } _ { V Q }$ is the sum of a Feature Reconstruction Loss and a Codebook Commitment Loss:

$$
\begin{array} { r } { \begin{array} { c } { \mathcal { L } _ { r e c } = \mathcal { L } _ { 1 } ( f _ { c o n t } , f _ { q } ) } \\ { \mathcal { L } _ { c o m m i t } = \beta \| \mathbf { z } _ { e } ( \mathbf { x } ) - \mathbf { s g } ( \mathbf { e } ) \| _ { 2 } ^ { 2 } } \end{array} } \\ { \mathcal { L } _ { V Q } = \mathcal { L } _ { r e c } + \lambda _ { c o m m i t } \mathcal { L } _ { c o m m i t } } \end{array}
$$

## 3.7. Curriculum Learning

Training Vector Quantized networks can be unstable due to its non-differentiable nature, and a cold-start with hard quantization often leads to codebook collapse and suboptimal rendering quality [48], as itâs significantly harder to converge. To mitigate this and ensure high-fidelity reconstruction, we implement a multi-stage curriculum learning strategy that gradually transitions the network from continuous to discrete representations.

## 3.7.1 Phase 1: Continuous Feature Warm-up

In the initial training phase (Steps 0 to $T _ { s t a r t } = 1 0 \mathbf { k } )$ , we disable quantization entirely. The network optimizes the anchor features $\mathbf { f } _ { c o n t }$ directly. To prepare the features for the distribution shift, we add small uniform noise to the scaling and offset parameters to simulate quantization error and improve decoder robustness [5].

## 3.7.2 Phase 2: Soft Quantization Injection

Between steps $T _ { s t a r t }$ and $T _ { e n d } = 3 0 \mathbf { k }$ , we linearly interpolate between the continuous features and their quantized counterparts. Let $\mathbf { f } _ { q }$ be the output of the VQ-VAE. The feature used for rendering, ${ \bf f } _ { r e n d e r }$ , is computed as:

$$
\mathbf { f } _ { r e n d e r } = ( 1 - \beta ) \cdot \mathbf { f } _ { c o n t } + \beta \cdot \mathbf { s g } [ \mathbf { f } _ { q } + ( \mathbf { f } _ { c o n t } - \mathbf { s g } [ \mathbf { f } _ { c o n t } ] ) ]
$$

where $\beta$ is a time-dependent warmup factor that linearly increases from 0 to 1. This transition allows the set of Î¦ MLPs to progressively adapt to an increasingly quantized signal.

## 3.7.3 Phase 3: Hard Quantization and Entropy Minimization

After $T _ { e n d } ,$ the network switches to Hard Quantization $( \beta =$ 1). On top of that, we enable the entropy model and add the rate loss $\mathcal { L } _ { r a t e }$ to the objective.

## 3.8. Progressive Transmission

## 3.8.1 Header and Base Layer (m = 1)

The initial transmission block consists of:

â¢ Binarized Spatial Hash Grid: To minimize the memory footprint of the context model, we binarize the parameters of the multi-resolution spatial hash grid, as proposed by HAC and HAC++. [9, 10]

â¢ MLP Decoder Compression: The weights of the lightweight decoding MLPs (Î¦) are compressed using Zstandard [13].

â¢ Base Visibility Mask: We explicitly encode the binary visibility state of each anchor and its associated Gaussian primitives for the base level, determining which primitives contribute to the coarse rendering.

â¢ Anchors and Base Features: We encode the sparse anchor positions using Geometry Point Cloud Compression [7]. Alongside the geometry, we transmit the first quantization index $k _ { 1 }$ for each active anchor, which will be decoded into the coarse latent feature on decoding.

$$
\mathbf { f } _ { i } ^ { 1 } = \operatorname { D e c o d e r } ( k _ { 1 } )
$$

## 3.8.2 Refinement Layers (m > 1)

Subsequent data chunks transmit both feature residuals and geometry updates.

â¢ Incremental Visibility: Rather than re-transmitting the full visibility mask at every level, we employ Differential Mask Encoding. We compute the difference between the binary mask at level m and level $m - 1$ transmitting only the indices of newly activated Gaussians. This ensures zero redundancy for primitives that were already visible.

â¢ Feature Refinement: For active anchors, the bitstream provides the residual indices $k _ { m }$ . The client then updates the latent features:

$$
\mathbf { f } _ { i } ^ { ( m ) } = \operatorname { D e c o d e r } ( \sum _ { j = 1 } ^ { m } k _ { j } )
$$

## 4. Experiments and Results

## 4.1. Experimental Setup

## 4.1.1 Datasets

We evaluate SCAR-GS on standard benchmarks for neural rendering and Gaussian Splatting to demonstrate its effectiveness across diverse scene types and scales.

NeRF Synthetic [29] contains eight object-centric scenes with complex view-dependent effects rendered at $8 0 0 \times 8 0 0$ resolution, providing a controlled environment for evaluating reconstruction quality.

For real-world performance evaluation, we use Tanks & Temples [21], MipNeRF360 [4], and Deep Blending [18] datasets, which feature large-scale, unbounded scenes that better highlight the advantages of progressive compression due to their substantial storage requirements.

Unbounded outdoor performance is further evaluated using BungeeNeRF [41], which includes challenging largescale scenes: Amsterdam, Bilbao, Hollywood, Pompidou, and Quebec.

## 4.1.2 Baselines

We compare SCAR-GS against state-of-the-art progressive compression methods for 3DGS: PCGS [8] and GoDe [33].

PCGS achieves progressivity through trit-plane quantization with incremental mask transmission and entropy modeling, training once to obtain multiple quality levels. PCGS refines attribute precision through scalar quantization of Gaussian attributes.

GoDe organizes Gaussians into hierarchical layers based on visibility heuristics, achieving progressivity through layer-wise primitive replication. GoDe increases primitive quantity at each Level of Detail (LOD) rather than refining quality.

<!-- image-->  
Figure 3. R-D curve of our method over different $\lambda _ { s s i m }$ (0.1, ..., 0.4) values in the Bycicle scene from the MipNeRF360 dataset. Benchmarked against PCGS and GoDE.

Our RVQ-based approach introduces a third paradigm: refining learned feature representations through residual vector quantization. This comparison evaluates whether vector quantization can match scalar quantization efficiency (PCGS) and whether feature-level refinement outperforms primitive-level replication (GoDe).

## 4.1.3 Implementation Details

SCAR-GS is trained for 40k iterations. The RVQ-VAE uses N = 4 quantization stages with a dual-codebook design consisting of a base codebook and a shared residual codebook, each containing 1024 entries, as $l o g _ { 2 } ( 1 0 2 4 ) = 1 0 .$ All experiments are conducted on NVIDIA A100 and H100 GPUs.

Table 1. Quantitative Evaluation on NeRF Synthetic Dataset. Comparison against PCGS [8] at Low/Mid/High bitrates.
<table><tr><td>Scene</td><td>Method</td><td>Size (MB) â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FPS â</td><td>Enc (s) â</td><td>Dec (s) â</td></tr><tr><td rowspan="6">Chair</td><td>Ours (ss0)</td><td>4.64</td><td>32.45</td><td>0.9759</td><td>0.0238</td><td>13.87</td><td>3.74</td><td>6.91</td></tr><tr><td>Ours (ss1)</td><td>4.83</td><td>33.64</td><td>0.9806</td><td>0.0189</td><td>13.87</td><td>1.50</td><td>2.12</td></tr><tr><td>Ours (ss2)</td><td>5.02</td><td>33.94</td><td>0.9819</td><td>0.0175</td><td>13.71</td><td>1.54</td><td>2.17</td></tr><tr><td>PCGS (Low)</td><td>1.23</td><td>34.61</td><td>0.9833</td><td>0.0157</td><td></td><td>1.10</td><td>1.40</td></tr><tr><td>PCGS (Mid)</td><td>1.57</td><td>35.29</td><td>0.9856</td><td>0.0141</td><td></td><td>0.3</td><td>0.3</td></tr><tr><td>PCGS (High)</td><td>2.05</td><td>35.45</td><td>0.9861</td><td>0.0136</td><td>â</td><td>0.3</td><td>0.4</td></tr><tr><td rowspan="6">Drums</td><td>Ours (ss0)</td><td>4.97</td><td>24.75</td><td>0.9298</td><td>0.0685</td><td>119.93</td><td>2.29</td><td>3.91</td></tr><tr><td>Ours (ss1)</td><td>5.23</td><td>25.52</td><td>0.9414</td><td>0.0569</td><td>117.04</td><td>0.86</td><td>1.18</td></tr><tr><td>Ours (ss2)</td><td>5.49</td><td>25.74</td><td>0.9447</td><td>0.0538</td><td>121.27</td><td>0.89</td><td>1.16</td></tr><tr><td>PCGS (Low)</td><td>1.68</td><td>26.31</td><td>0.9504</td><td>0.0424</td><td>â</td><td>1.40</td><td>2.00</td></tr><tr><td>PCGS (Mid)</td><td>2.15</td><td>26.47</td><td>0.9522</td><td>0.0407</td><td>-</td><td>0.30</td><td>0.30</td></tr><tr><td>PCGS (High)</td><td>2.69</td><td>26.49</td><td>0.9524</td><td>0.0405</td><td>â</td><td>0.30</td><td>0.40</td></tr><tr><td rowspan="6">Ficus</td><td>Ours (ss0)</td><td>4.50</td><td>31.80</td><td>0.9709</td><td>0.0294</td><td>69.81</td><td>2.29</td><td>4.03</td></tr><tr><td>Ours (ss1)</td><td>4.71</td><td>33.33</td><td>0.9787</td><td>0.0208</td><td>125.43</td><td>0.89</td><td>2.32</td></tr><tr><td>Ours (ss2)</td><td>4.94</td><td>34.26</td><td>0.9822</td><td>0.0170</td><td>122.36</td><td>0.94</td><td>2.59</td></tr><tr><td>PCGS (Low)</td><td>1.18</td><td>34.78</td><td>0.9844</td><td>0.0144</td><td>â</td><td>0.90</td><td>1.20</td></tr><tr><td>PCGS (Mid)</td><td>1.47</td><td>35.45</td><td>0.9864</td><td>0.0129</td><td>â</td><td>0.20</td><td>0.20</td></tr><tr><td>PCGS (High)</td><td>1.82</td><td>35.53</td><td>0.9866</td><td>0.0127</td><td>â</td><td>0.20</td><td>0.30</td></tr><tr><td rowspan="6">Hotdog</td><td>Ours (ss0)</td><td>4.15</td><td>32.57</td><td>0.9629</td><td>0.0515</td><td>19.08</td><td>3.10</td><td>6.62</td></tr><tr><td>Ours (ss1)</td><td>4.33</td><td>35.52</td><td>0.9762</td><td>0.0337</td><td>19.04</td><td>1.43</td><td>2.05</td></tr><tr><td>Ours (ss2)</td><td>4.51</td><td>36.85</td><td>0.9808</td><td>0.0274</td><td>18.68</td><td>1.49</td><td>2.15</td></tr><tr><td>PCGS (Low)</td><td>0.99</td><td>37.18</td><td>0.9817</td><td>0.0277</td><td>â</td><td>0.70</td><td>0.80</td></tr><tr><td>PCGS (Mid)</td><td>1.20</td><td>37.77</td><td>0.9834</td><td>0.0257</td><td>â</td><td>0.20</td><td>0.20</td></tr><tr><td>PCGS (High)</td><td>1.48</td><td>37.88</td><td>0.9838</td><td>0.0250</td><td>â</td><td>0.20</td><td>0.30</td></tr><tr><td rowspan="6">Lego</td><td>Ours (ss0)</td><td>4.78</td><td>32.41</td><td>0.9673</td><td>0.0341</td><td>132.51</td><td>1.84</td><td>3.33</td></tr><tr><td>Ours (ss1)</td><td>4.99</td><td>33.60</td><td>0.9732</td><td>0.0273</td><td>133.56</td><td>0.69</td><td>0.96</td></tr><tr><td>Ours (ss2)</td><td>5.20</td><td>34.04</td><td>0.9751</td><td>0.0252</td><td>133.07</td><td>0.70</td><td>0.96</td></tr><tr><td>PCGS (Low)</td><td>1.45</td><td>35.08</td><td>0.9790</td><td>0.0207</td><td></td><td>1.20</td><td>1.70</td></tr><tr><td>PCGS (Mid)</td><td>1.85</td><td>35.60</td><td>0.9811</td><td>0.0190</td><td>â</td><td>0.30</td><td>0.30</td></tr><tr><td>PCGS (High)</td><td>2.36</td><td>35.70</td><td>0.9814</td><td>0.0186</td><td></td><td>0.30</td><td>0.40</td></tr></table>

Table 1 shows that SCAR-GS achieves competitive perceptual quality across progressive stages while enabling feature-level refinement rather than scalar precision tuning. Although SCAR-GS operates at higher bitrates than PCGS, quality improves smoothly across refinement stages, demonstrating the effectiveness of residual feature refinement for progressive transmission.

Table 2. Evaluation Results on Deep Blending Dataset. Comparison against PCGS [8] and GoDE [33] at various Levels of Detail.
<table><tr><td>Scene</td><td>Method</td><td>Size (MB) â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FPS â</td><td>Enc (s) â</td><td>Dec (s) â</td></tr><tr><td rowspan="10">DrJohnson</td><td>Ours (ss0)</td><td>9.73</td><td>28.51</td><td>0.8940</td><td>0.2760</td><td>227.02</td><td>10.04</td><td>16.10</td></tr><tr><td>Ours (ss1)</td><td>10.69</td><td>29.02</td><td>0.8995</td><td>0.2658</td><td>229.18</td><td>3.16</td><td>3.57</td></tr><tr><td>Ours (ss2)</td><td>11.70</td><td>29.19</td><td>0.9015</td><td>0.2623</td><td>227.52</td><td>3.34</td><td>3.88</td></tr><tr><td>Ours (ss3)</td><td>12.82</td><td>29.24</td><td>0.9024</td><td>0.2605</td><td>227.16</td><td>3.62</td><td>4.33</td></tr><tr><td>PCGS (Low)</td><td>3.73</td><td>29.70</td><td>0.9045</td><td>0.2620</td><td>â</td><td>4.00</td><td>5.60</td></tr><tr><td>PCGS (Mid)</td><td>5.02</td><td>29.83</td><td>0.9069</td><td>0.2584</td><td>â</td><td>0.90</td><td>1.00</td></tr><tr><td>PCGS (High)</td><td>6.70</td><td>29.85</td><td>0.9074</td><td>0.2576</td><td>â</td><td>1.00</td><td>1.20</td></tr><tr><td>GoDE (LOD 0)</td><td>3.70</td><td>28.56</td><td>0.875</td><td>0.391</td><td>970</td><td>â</td><td>â</td></tr><tr><td>GoDE (LOD 2)</td><td>7.90</td><td>29.15</td><td>0.891</td><td>0.361</td><td>767</td><td>-</td><td>â</td></tr><tr><td>GoDE (LOD 4)</td><td>16.60</td><td>29.26</td><td>0.897</td><td>0.342</td><td>570</td><td>â</td><td>â</td></tr><tr><td rowspan="10"></td><td>GoDE (LOD 7)</td><td>47.90</td><td>29.28</td><td>0.899</td><td>0.332</td><td>316</td><td>â</td><td>â</td></tr><tr><td>Ours (ss0)</td><td>7.24</td><td>29.41</td><td>0.8990</td><td>0.2772</td><td>284.41</td><td>7.45</td><td>11.48</td></tr><tr><td>Ours (ss1) Ours (ss2)</td><td>7.99</td><td>29.71</td><td>0.9026</td><td>0.2719</td><td>286.79</td><td>2.61</td><td>2.76</td></tr><tr><td></td><td>8.75</td><td>29.86</td><td>0.9040</td><td>0.2698</td><td>284.62</td><td>2.67</td><td>2.91</td></tr><tr><td>Ours (ss3)</td><td>9.55 2.80</td><td>29.91</td><td>0.9046</td><td>0.2688</td><td>285.65</td><td>2.78</td><td>3.15</td></tr><tr><td>PCGS (Low)</td><td>3.68</td><td>30.69</td><td>0.9091</td><td>0.2657</td><td>â</td><td>2.90</td><td>4.20</td></tr><tr><td>PCGS (Mid)</td><td>4.92</td><td>30.85</td><td>0.9113</td><td>0.2620</td><td>-</td><td>0.60</td><td>0.70</td></tr><tr><td>PCGS (High)</td><td>3.80</td><td>30.91 29.89</td><td>0.9119 0.9010</td><td>0.2609 0.3540</td><td>â</td><td>0.80</td><td>1.00</td></tr><tr><td>GoDE (LOD 0)</td><td>7.00</td><td>30.25</td><td>0.9090</td><td>0.3340</td><td>658</td><td>â</td><td>â</td></tr><tr><td>GoDE (LOD 2)</td><td></td><td></td><td></td><td></td><td>477</td><td>â</td><td></td></tr><tr><td></td><td>GoDE (LOD 4) GoDE (LOD 7)</td><td>13.00 31.7</td><td>30.29 30.27</td><td>0.9110 0.911</td><td>0.3240 0.316</td><td>406 224</td><td></td><td></td></tr></table>

Table 2 demonstrates that SCAR-GS provides consistent and monotonic improvements in perceptual quality as refinement layers are added. Compared to GoDe, which relies on primitive replication for level-of-detail control, SCAR-GS achieves smoother quality gains with substantially lower storage growth, highlighting the advantage of feature refinement over primitive-based LOD strategies.

Table 3. Evaluation Results on Tanks and Temples Dataset
<table><tr><td>Scene</td><td>Step</td><td>SSIM</td><td>PSNR</td><td>LPIPS</td><td>FPS</td><td>Train (s)</td><td>Enc (s)</td><td>Dec (s)</td></tr><tr><td rowspan="4">truck</td><td>ss0</td><td>0.8604</td><td>24.91</td><td>0.1802</td><td>189.67</td><td rowspan="4">18355.42</td><td>15.03</td><td>25.31</td></tr><tr><td>ss1</td><td>0.8729</td><td>25.45</td><td>0.1641</td><td>190.70</td><td>4.85</td><td>5.13</td></tr><tr><td>ss2</td><td>0.8776</td><td>25.63</td><td>0.1578</td><td>188.20</td><td>4.76</td><td>5.33</td></tr><tr><td>ss3</td><td>0.8795</td><td>25.71</td><td>0.1549 188.38</td><td></td><td>4.77</td><td>5.79</td></tr><tr><td rowspan="4">train</td><td>ss0</td><td>0.8008</td><td>21.54</td><td>0.2359</td><td>182.68</td><td rowspan="4">24854.57</td><td>8.85</td><td>13.38</td></tr><tr><td>ss1</td><td>0.8168</td><td>22.11</td><td>0.2169</td><td>179.86</td><td>2.99</td><td>3.52</td></tr><tr><td>ss2</td><td>0.8224</td><td>22.31</td><td>0.2089</td><td>178.19</td><td>3.46</td><td>4.04</td></tr><tr><td>ss3</td><td>0.8246</td><td>22.43</td><td>0.2052</td><td>173.77</td><td>3.80</td><td>4.78</td></tr></table>

As shown in Table 3 , SCAR-GS progressively improves reconstruction quality across refinement stages while maintaining stable rendering performance. This confirms that residual feature refinement generalizes effectively to complex real-world scenes without introducing rendering instability.

Table 4 illustrates that SCAR-GS enables fine-grained quality control on large, unbounded scenes. Progressive feature refinement yields steady gains in SSIM and LPIPS with moderate bitrate increases, contrasting with GoDeâs stepwise quality changes driven by increasing primitive counts.

Table 4. Evaluation Results on MipNeRF360 Dataset.
<table><tr><td>Scene</td><td>Method</td><td>Size (MB) â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FPS â</td><td>Enc (s) â</td><td>Dec (s) â</td></tr><tr><td rowspan="8">Bonsai</td><td>Ours (ss0)</td><td>7.30</td><td>29.84</td><td>0.9193</td><td>0.2173</td><td>207.06</td><td>8.14</td><td>12.38</td></tr><tr><td>Ours (ss1)</td><td>8.24</td><td>30.94</td><td>0.9331</td><td>0.2031</td><td>206.17</td><td>3.14</td><td>3.50</td></tr><tr><td>Ours (ss2)</td><td>9.20</td><td>31.30</td><td>0.9372</td><td>0.1978</td><td>202.22</td><td>3.28</td><td>3.80</td></tr><tr><td>Ours (ss3)</td><td>10.28</td><td>31.45</td><td>0.9390</td><td>0.1954</td><td>201.45</td><td>3.67</td><td>4.20</td></tr><tr><td>GoDE (LOD 0)</td><td>3.70</td><td>29.69</td><td>0.906</td><td>0.3300</td><td>434</td><td>â</td><td>â</td></tr><tr><td>GoDE (LOD 2)</td><td>6.40</td><td>31.39</td><td>0.9300</td><td>0.2920</td><td>338</td><td>-</td><td></td></tr><tr><td>GoDE (LOD 4)</td><td>11.30</td><td>31.77</td><td>0.9370</td><td>0.2730</td><td>276</td><td>â</td><td>â</td></tr><tr><td>GoDE (LOD 7)</td><td>25.80</td><td>31.89</td><td>0.9390</td><td>0.2660</td><td>211</td><td>â</td><td>â</td></tr><tr><td rowspan="8">Flowers</td><td>Ours (ss0)</td><td>14.71</td><td>20.71</td><td>0.5428</td><td>0.4028</td><td>183.14</td><td>22.23</td><td>35.50</td></tr><tr><td>Ours (ss1)</td><td>17.26</td><td>21.14</td><td>0.5669</td><td>0.3812</td><td>184.92</td><td>8.18</td><td>9.21</td></tr><tr><td>Ours (ss2)</td><td>20.17</td><td>21.32</td><td>0.5774</td><td>0.3711</td><td>183.77</td><td>8.89</td><td>10.23</td></tr><tr><td>Ours (ss3)</td><td>23.44</td><td>21.41</td><td>0.5827</td><td>0.3660</td><td>182.31</td><td>9.96</td><td>11.87</td></tr><tr><td>GoDE (LOD 0)</td><td>3.90</td><td>19.76</td><td>0.4700</td><td>0.5110</td><td>703</td><td>â</td><td>â</td></tr><tr><td>GoDE (LOD 2)</td><td>9.50</td><td>20.89</td><td>0.5430</td><td>0.4530</td><td>496</td><td>â</td><td>â</td></tr><tr><td>GoDE (LOD 4)</td><td>23.10</td><td>21.35</td><td>0.5840</td><td>0.4080</td><td>358</td><td>-</td><td>â</td></tr><tr><td>GoDE (LOD 7)</td><td>80.70</td><td>21.44</td><td>0.5960</td><td>0.3780</td><td>231</td><td>â</td><td>â</td></tr><tr><td rowspan="6">Stump</td><td>Ours (ss0)</td><td>11.42</td><td>25.85</td><td>0.7333</td><td>0.3089</td><td>213.91</td><td>15.93</td><td>25.70</td></tr><tr><td>Ours (ss1)</td><td>13.19</td><td>26.47</td><td>0.7565</td><td>0.2808</td><td>208.61</td><td>5.61</td><td>6.41</td></tr><tr><td>Ours (ss2)</td><td>14.97</td><td>26.73</td><td>0.7666</td><td>0.2682</td><td>207.62</td><td>5.88</td><td>6.77</td></tr><tr><td>Ours (ss3)</td><td>16.94</td><td>26.83</td><td>0.7711</td><td>0.2623</td><td>207.21</td><td>6.29</td><td>7.67</td></tr><tr><td>PCGS (Low)</td><td>4.23</td><td>26.67</td><td>0.7626</td><td>0.2711</td><td></td><td>2.40</td><td>2.70</td></tr><tr><td>PCGS (Mid)</td><td>4.64</td><td>26.67</td><td>0.7627</td><td>0.2707</td><td>â</td><td>2.80</td><td>3.40</td></tr><tr><td rowspan="8">Room</td><td>Ours (ss0)</td><td>10.32</td><td>26.16</td><td>0.8427</td><td>0.3389</td><td>174.45</td><td>15.65</td><td>22.79</td></tr><tr><td>Ours (ss1)</td><td>11.66</td><td>26.39</td><td>0.8462</td><td>0.3345</td><td>170.76</td><td>4.62</td><td>4.88</td></tr><tr><td>Ours (ss2)</td><td>12.96</td><td>26.40</td><td>0.8465</td><td>0.3333</td><td>168.27</td><td>4.59</td><td>4.91</td></tr><tr><td>Ours (ss3)</td><td>14.28</td><td>26.32</td><td>0.8452</td><td>0.3347</td><td>169.41</td><td>4.69</td><td>4.93</td></tr><tr><td>PCGS (Low)</td><td>5.00</td><td>32.07</td><td>0.9232</td><td>0.2094</td><td>â</td><td>5.30</td><td>7.90</td></tr><tr><td>PCGS (Mid)</td><td>6.79</td><td>32.24</td><td>0.9262</td><td>0.2043</td><td>-</td><td>1.10</td><td>1.20</td></tr><tr><td>PCGS (High)</td><td>8.85</td><td>32.28</td><td>0.9271</td><td>0.2021</td><td>-</td><td>1.20</td><td>1.40</td></tr><tr><td>PCGS (Ultra)</td><td>11.10</td><td>32.30</td><td>0.9274</td><td>0.2013</td><td></td><td>1.30</td><td>1.60</td></tr></table>

Table 5. Evaluation Results on BungeeNeRF Dataset. Comparison against PCGS [8].
<table><tr><td>Scene</td><td>Method</td><td>Size (MB) â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FPS â</td><td>Enc (s) â</td><td>Dec (s) â</td></tr><tr><td rowspan="7">Amsterdam</td><td>Ours (ss0)</td><td>9.70</td><td>24.32</td><td>0.8017</td><td>0.2490</td><td>234.81</td><td>10.61</td><td>17.98</td></tr><tr><td>Ours (ss1)</td><td>10.65</td><td>25.35</td><td>0.8343</td><td>0.2211</td><td>233.32</td><td>3.16</td><td>3.60</td></tr><tr><td>Ours (ss2)</td><td>11.59</td><td>25.79</td><td>0.8473</td><td>0.2097</td><td>232.88</td><td>3.17</td><td>3.66</td></tr><tr><td>Ours (ss3)</td><td>12.56</td><td>25.98</td><td>0.8529</td><td>0.2046</td><td>233.04</td><td>3.32</td><td>3.89</td></tr><tr><td>PCGS (Low)</td><td>15.49</td><td>27.03</td><td>0.8793</td><td>0.2028</td><td>-</td><td>16.30</td><td>23.90</td></tr><tr><td>PCGS (Mid)</td><td>21.08</td><td>27.23</td><td>0.8861</td><td>0.1942</td><td></td><td>3.30</td><td>3.70</td></tr><tr><td>PCGS (High)</td><td>27.74</td><td>27.28</td><td>0.8888</td><td>0.1891</td><td>â</td><td>4.00</td><td>5.00</td></tr><tr><td rowspan="7">Bilbao</td><td>Ours (ss0)</td><td>8.31</td><td>25.42</td><td>0.8294</td><td>0.2218</td><td>276.30</td><td>8.01</td><td>13.55</td></tr><tr><td>Ours (ss1)</td><td>9.07</td><td>26.58</td><td>0.8571</td><td>0.1973</td><td>279.38</td><td>2.49</td><td>2.77</td></tr><tr><td>Ours (ss2)</td><td>9.80</td><td>27.03</td><td>0.8677</td><td>0.1881</td><td>277.57</td><td>2.49</td><td>2.80</td></tr><tr><td>Ours (ss3)</td><td>10.55</td><td>27.21</td><td>0.8722</td><td>0.1839</td><td>275.36</td><td>2.55</td><td>2.94</td></tr><tr><td>PCGS (Low)</td><td>12.18</td><td>27.91</td><td>0.8818</td><td>0.1988</td><td>-</td><td>12.50</td><td>17.90</td></tr><tr><td>PCGS (Mid)</td><td>16.53</td><td>28.09</td><td>0.8872</td><td>0.1903</td><td>â</td><td>2.50</td><td>2.90</td></tr><tr><td>PCGS (High)</td><td>21.77</td><td>28.11</td><td>0.8891</td><td>0.1856</td><td>â</td><td>3.10</td><td>4.00</td></tr><tr><td rowspan="7">Hollywood</td><td>Ours (ss0)</td><td>8.79</td><td>23.37</td><td>0.7080</td><td>0.3535</td><td>298.59</td><td>8.70</td><td>14.57</td></tr><tr><td>Ours (ss1)</td><td>9.61</td><td>24.17</td><td>0.7471</td><td>0.3248</td><td>299.29</td><td>2.65</td><td>3.00</td></tr><tr><td>Ours (ss2)</td><td>10.39</td><td>24.54</td><td>0.7639</td><td>0.3117</td><td>299.82</td><td>2.67</td><td>3.03</td></tr><tr><td>Ours (ss3)</td><td>11.21</td><td>24.70</td><td>0.7717</td><td>0.3050</td><td>296.47</td><td>2.77</td><td>3.22</td></tr><tr><td>PCGS (Low)</td><td>12.35</td><td>24.43</td><td>0.7657</td><td>0.3319</td><td>-</td><td>12.30</td><td>16.90</td></tr><tr><td>PCGS (Mid)</td><td>16.33</td><td>24.58</td><td>0.7736</td><td>0.3254</td><td>â</td><td>2.30</td><td>2.50</td></tr><tr><td>PCGS (High)</td><td>20.95</td><td>24.64</td><td>0.7774</td><td>0.3210</td><td>â</td><td>2.70</td><td>3.30</td></tr><tr><td rowspan="7">Pompidou</td><td>Ours (ss0)</td><td>10.48</td><td>23.16</td><td>0.8230</td><td>0.2121</td><td>229.89</td><td>11.76</td><td>19.73</td></tr><tr><td>Ours (ss1)</td><td>11.56</td><td>24.06</td><td>0.8515</td><td>0.1861</td><td>231.53</td><td>3.57</td><td>4.03</td></tr><tr><td>Ours (ss2)</td><td>12.62</td><td>24.45</td><td>0.8625</td><td>0.1757</td><td>231.16</td><td>3.58</td><td>4.10</td></tr><tr><td>Ours (ss3)</td><td>13.71</td><td>24.62</td><td>0.8675</td><td>0.1709</td><td>230.25</td><td>3.71</td><td>4.31</td></tr><tr><td>PCGS (Low)</td><td>13.87</td><td>25.63</td><td>0.8517</td><td>0.2347</td><td>â</td><td>14.40</td><td>20.60</td></tr><tr><td>PCGS (Mid)</td><td>18.72</td><td>25.81</td><td>0.8570</td><td>0.2293</td><td>â</td><td>2.9</td><td>3.2</td></tr><tr><td>PCGS (High)</td><td>24.56</td><td>25.85</td><td>0.8585</td><td>0.2270</td><td>â</td><td>3.4</td><td>4.3</td></tr><tr><td rowspan="7">Quebec</td><td>Ours (ss0)</td><td>8.39</td><td>35.11</td><td>0.8268</td><td>0.2341</td><td>267.19</td><td>8.35</td><td>14.18</td></tr><tr><td>Ours (ss1)</td><td>9.18</td><td>26.22</td><td>0.8588</td><td>0.2025</td><td>266.34</td><td>2.58</td><td>2.90</td></tr><tr><td>Ours (ss2)</td><td>9.93</td><td>26.70</td><td>0.8716</td><td>0.1898</td><td>266.41</td><td>2.56</td><td>2.94</td></tr><tr><td>Ours (ss3)</td><td>10.71</td><td>26.90</td><td>0.8771</td><td>0.1842</td><td>266.51</td><td>2.69</td><td>3.02</td></tr><tr><td>PCGS (Low)</td><td>10.94</td><td>30.13</td><td>0.9338</td><td>0.1610</td><td>â</td><td>11.2</td><td>16.1</td></tr><tr><td>PCGS (Mid)</td><td>14.72</td><td>30.43</td><td>0.9380</td><td>0.1562</td><td>â</td><td>2.2</td><td>2.5</td></tr><tr><td>PCGS (High)</td><td>19.18</td><td>30.49</td><td>0.9388</td><td>0.1546</td><td>â</td><td>2.6</td><td>3.2</td></tr></table>

Results in Table 5 show that SCAR-GS maintains consistent perceptual improvements across extremely largescale outdoor scenes. Despite operating at lower bitrates than PCGS for comparable quality levels, SCAR-GS provides smoother progressive refinement, making it better suited for adaptive streaming scenarios.

Entropy decoding using the GRU and spatial-query attention model is performed once per progressive transmission step and is not part of the per-frame rendering loop.

The decoding cost scales linearly with the number of active anchors and refinement layers and is amortized over subsequent rendering, such that runtime FPS is unaffected once decoding is completed.

Since refinement layers only add residual feature information and do not introduce new primitives, decoder memory usage grows linearly with refinement depth and remains bounded by the final representation size.

## 5. Ablation Studies

To validate the effectiveness of our architectural choices, we conducted ablation studies on the Bicycle scene from MipNeRF360 [4] with $\lambda _ { s s i m } = 0 . 2$ , isolating specific components to evaluate their individual contributions to compression efficiency and reconstruction quality.

## 5.1. Architecture of the Entropy Model

We evaluate the impact of the context model architecture on compression efficiency by comparing our proposed GRU with Spatial-Query Attention against three baselines: a standard MLP, a Branched MLP (separate heads for spatial and feature context), and a vanilla GRU without the spatial attention mechanism.

Unlike prior entropy models that apply generic attention, our spatial-query attention conditions residual history asymmetrically, using spatial embeddings as queries over decoded residual sequences to enable geometry-aware sequential probability estimation.

Table 6. Ablation study on the architecture of the entropy model. Our proposed GRU with Spatial-Query Attention achieves the best compression rate (smallest size) and reconstruction quality.
<table><tr><td>Architecture</td><td>Size â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNR â</td></tr><tr><td>MLP</td><td>19.3</td><td>0.71</td><td>0.32</td><td>24.5</td></tr><tr><td>Branched MLP</td><td>22.4</td><td>0.72</td><td>0.31</td><td>24.6</td></tr><tr><td>GRU</td><td>21.9</td><td>0.71</td><td>0.30</td><td>24.4</td></tr><tr><td>GRU + Attn.</td><td>18.0</td><td>0.73</td><td>0.29</td><td>24.7</td></tr></table>

As shown in Table 6, our proposed architecture significantly outperforms all baselines. Simple MLPs cannot model sequential dependencies between residual codes, treating each quantization stage independently. The Branched MLP improves slightly by processing spatial and feature contexts separately, but fails to effectively fuse these modalities: the separate heads optimize independently without capturing their interaction.

The vanilla GRU successfully models temporal structure across residual layers but lacks spatial conditioning, achieving 21.9 MB at 0.71 SSIM. Without geometric context, probability predictions cannot adapt to local scene characteristics. Our Spatial-Query Attention mechanism bridges this gap by treating spatial embeddings as queries attending over GRU hidden states, allowing the network to dynamically weight residual history based on local geometry. This achieves 18.0 MB at 0.73 SSIM: a 7% size reduction and 0.02 SSIM improvement over vanilla GRU, demonstrating that spatially-conditioned autoregressive modeling is essential for efficient entropy coding.

## 5.2. Residual vs. Standard Vector Quantization

A key design choice in SCAR-GS is using RVQ (progressive) over VQ (single-rate). We compared our RVQ approach against single-stage VQ with a larger 4096-entry codebook to match capacity.

Table 7. Comparison between standard single-rate Vector Quantization (VQ) and our progressive Residual Vector Quantization (RVQ). RVQ yields superior rate-distortion performance.
<table><tr><td>Architecture</td><td>Size â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td></tr><tr><td>VQ</td><td>20.8</td><td>0.72</td><td>0.29</td><td>24.4</td></tr><tr><td>RVQ</td><td>18.0</td><td>0.73</td><td>0.29</td><td>24.7</td></tr></table>

Table 7 confirms that RVQ is superior for both streaming capability and compression efficiency. By decomposing the feature space into âcoarseâ base signals and âfineâ residuals, RVQ enables the entropy model to learn more distinct, lower-entropy distributions for each stage. The base layer captures high-variance global structure with a broad probability distribution, while residual layers model progressively lower-entropy refinements with peaked distributions.

Standard VQ requires 20.8 MB to achieve 0.72 SSIM: 15% larger than our RVQ at better quality (0.73 SSIM). This reflects the difficulty of optimizing entropy for large singlestage codebooks where the model must capture all feature complexity in one distribution without sequential context. Our autoregressive conditioning on previous quantizations produces inherently more compressible probability distributions.

## 5.3. Gradient Propagation: Rotation Trick vs. STE

We analyzed the impact of the gradient estimator used for the non-differentiable quantization step. We compared the STE [6] against the Rotation Trick [15] implemented in our pipeline.

Table 8 shows that while STE produces 6% smaller files, the Rotation Trick achieves marginally better PSNR (24.7 vs. 24.6) with identical perceptual metrics. More importantly, we observed significantly more stable training dynamics with the Rotation Trick across different scenes, random seeds, and initialization strategies. The Rotation Trick injects geometric information about quantization error magnitude and direction into gradients by modeling the encoder-to-codebook relationship as a smooth linear transformation. This leads to more balanced codebook utilization and avoids local minima where certain entries dominate. The 6% size increase likely reflects more conservative probability estimation when gradients carry geometric information, but improved training stability and consistent convergence justify this tradeoff for robust deployment across diverse scenes. Across ablation studies, we observe that architectural choices improving stability and representational robustness may incur modest increases in bitrate. These increases reflect tighter entropy modelling and improved generalisation rather than reduced compression effectiveness, and consistently result in superior perceptual quality and convergence behaviour.

Table 8. Impact of the gradient estimator on training stability and final quality. The Rotation Trick allows for better geometric capture on the gradient flow, leading to better reconstruction fidelity compared to the STE.
<table><tr><td>Estimator</td><td>Size â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td></tr><tr><td>STE</td><td>16.9</td><td>0.73</td><td>0.29</td><td>24.6</td></tr><tr><td>Rotation Trick</td><td>18.0</td><td>0.73</td><td>0.29</td><td>24.7</td></tr></table>

## 5.4. Impact of Curriculum Learning

As mentioned in the methodology section,training RVQ-VAEs with hard quantization from initialization can be unstable. We evaluate our three-phase curriculum learning strategy against a cold-start approach.

Table 9. Evaluation of the curriculum learning strategy. A âcold startâ without warm-up leads to significant quality degradation, while our curriculum schedule ensures robust convergence.
<table><tr><td>Training Strategy</td><td>Size â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td></tr><tr><td>Cold Start</td><td>13.7</td><td>0.70</td><td>0.34</td><td>24.4</td></tr><tr><td>Curriculum Learning</td><td>18.0</td><td>0.73</td><td>0.29</td><td>24.7</td></tr></table>

Table 9 demonstrates the disadvantage of cold-start training. Without gradual adaptation, the network prematurely commits to suboptimal codebook entries, causing codebook collapse where only a small subset of entries are actively used. The feature encoder learns to map all inputs to this limited subset, destroying representational capacity. Additionally, the scene decoder receives discrete inputs from initialization without the opportunity to learn smooth interpolation between codebook vectors.

Our curriculum learning (continuous warm-up (0-10k), soft quantization injection (10k-30k), and hard quantization refinement (30k-40k)) achieves 18.0 MB at 0.73 SSIM and 0.29 LPIPS. The 31% storage increase versus cold start is necessary to avoid catastrophic quality loss: 0.03 SSIM improvement and 17% LPIPS reduction. This validates that stable VQ training requires (1) warm initialization with continuous features, (2) gradual introduction of quantization constraints, and (3) progressive commitment to discrete representations.

## 5.5. Spatial Context Representation

We validated the design of our spatial hash grid. We compared a pure 3D Hash Grid against the Hybrid 2D+3D Grid, proposed by HAC++ [10].

Table 10. Effectiveness of the spatial hash grid representation. The hybrid 2D+3D grid captures anisotropic correlations better than a pure 3D grid.
<table><tr><td>Spatial Grid</td><td>Size â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td></tr><tr><td>3D Grid</td><td>14.9</td><td>0.70</td><td>0.33</td><td>24.5</td></tr><tr><td>Hybrid Grid</td><td>18.0</td><td>0.73</td><td>0.29</td><td>24.7</td></tr></table>

Table 10 shows that the hybrid grid achieves 18.0 MB at 0.73 SSIM versus the pure 3D gridâs 14.9 MB at 0.70 SSIM. The 21% size increase is justified by substantial quality improvements: 0.03 SSIM gain and 12% LPIPS reduction.

The hybrid design captures anisotropic spatial correlations: directional dependencies in scene structure. Many real-world scenes exhibit ground-plane dominated structure where lateral context (neighboring buildings, terrain features) differs fundamentally from vertical context (sky, height variations). Pure 3D grids treat all directions equally, failing to model these directional patterns.

When predicting residual indices, the hybrid grid allows the entropy model to distinguish high-correlation directions (lateral neighbors) from low-correlation directions (vertical), achieving tighter probability distributions. This directional modeling translates to more accurate probability estimation despite the additional hash grid parameters, improving both compression efficiency and reconstruction fidelity.

## 6. Conclusion

While SCAR-GS improves progressive quality refinement through feature-level residuals, it incurs higher baselayer storage than scalar-quantization approaches, which prioritize perceptual fidelity and refinement consistency over extreme base-layer compactness.

In this paper, we presented SCAR-GS, a spatially-aware vector-quantized autoregressive progressive codec for 3D Gaussian Splatting. Extensive experiments demonstrate that the proposed approach enables smooth and consistent perceptual quality improvement across refinement stages, making it well-suited for adaptive rendering scenarios that require on-demand transmission of visual content at variable quality levels.

One limitation of SCAR-GS arises in scenes with extremely sparse geometry or under very aggressive baselayer bitrate constraints, where RVQ base features may lack sufficient structural information, leading to slower perceptual convergence during refinement.

In future work, we would like to explore how we can propose a network streaming framework suited for SCAR-GS dynamic streaming, such as DASH [35] for LapisGS [34]. Additionally, exploring improved RVQ-VAE training objectives that more tightly preserve original feature structure may further enhance reconstruction fidelity and perceptual quality.

## References

[1] Muhammad Salman Ali, Maryam Qamar, Sung-Ho Bae, and Enzo Tartaglione. Trimming the fat: Efficient compression of 3d gaussian splats through pruning. ArXiv, abs/2406.18214, 2024. 1

[2] M. T. Bagdasarian, P. Knoll, Y. Li, F. Barthel, A. Hilsmann, P. Eisert, and W. Morgenstern. 3dgs.zip: A survey on 3d gaussian splatting compression methods. Computer Graphics Forum, page e70078, 2025. https://wm.github.io/3dgs-compression-survey/. 1

[3] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014. 4

[4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5460â5469, 2021. 4, 6, 8

[5] Chaim Baskin, Natan Liss, Yoav Chai, Evgenii Zheltonozhskii, Eli Schwartz, Raja Giryes, Avi Mendelson, and Alexander M. Bronstein. Nice: Noise injection and clamping estimation for neural network quantization. ArXiv, abs/1810.00162, 2018. 5

[6] Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Â´ Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013. 3, 8

[7] Anthony Chen, Shiwen Mao, Zhu Li, Minrui Xu, Hongliang Zhang, Dusit Niyato, and Zhu Han. An introduction to point cloud compression standards. GetMobile: Mobile Comp. and Comm., 27(1):11â17, May 2023. 5

[8] Yihang Chen, Mengyao Li, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Pcgs: Progressive compression of 3d gaussian splatting. In The 40th Annual AAAI Conference on Artificial Intelligence, 2026. 1, 2, 3, 5, 6, 7

[9] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Hac: Hash-grid assisted context for 3d gaussian splatting compression. In European Conference on Computer Vision, 2024. 1, 3, 4, 5

[10] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Hac++: Towards 100x compression of 3d gaussian splatting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1, 3, 5, 9

[11] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin, Yuexin Ma, Wenping Wang, and Xuejin Chen. Gaussianpro: 3d gaussian splatting with progressive propagation. In Proceedings of the 41st International Conference on Machine Learning, ICMLâ24. JMLR.org, 2024. 1

[12] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Â¨ Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoderâdecoder for statistical machine translation. In Alessandro Moschitti, Bo Pang, and Walter Daelemans, editors, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724â1734, Doha, Qatar, Oct. 2014. Association for Computational Linguistics. 4

[13] Yann Collet and Murray Kucherawy. Zstandard Compression and the âapplication/zstdâ Media Type. RFC 8878, Feb. 2021. 5

[14] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian: unbounded 3d gaussian compression with 15x reduction and 200+ fps. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS â24, Red Hook, NY, USA, 2024. Curran Associates Inc. 1, 2

[15] Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iyengar, Jerry W. Liu, Ehsan Amid, Sebastian Thrun, and Christopher Re. Restructuring vector quantization with theÂ´ rotation trick. In Proceedings of the International Conference on Learning Representations (ICLR), 2025. 4, 8

[16] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. Eagles: Efficient accelerated 3d gaussians with lightweight encodings. ArXiv, abs/2312.04564, 2023. 1

[17] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Softmax Units for Multinoulli Output Distributions, chapter 6.2.2.3, pages 180â184. MIT Press, Cambridge, MA, 2016. 4

[18] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. Deep blending for free-viewpoint image-based rendering. 37(6):257:1â257:15, 2018. 6

[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering, 2023. 1

[20] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Yoshua Bengio and Yann LeCun, editors, 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. 3

[21] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017. 6

[22] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11513â11522, 2022. 3

[23] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for

radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21719â21728, 2024.

[24] Lei Liu, Zhenghao Chen, Wei Jiang, Wei Wang, and Dong Xu. Hemgs: A hybrid entropy model for 3d gaussian splatting data compression, 2025. 1

[25] Shicong Liu, Junru Shao, and Hongtao Lu. Generalized residual vector quantization for large scale data. In 2016 IEEE International Conference on Multimedia and Expo (ICME), pages 1â6, 2016. 2

[26] Xiangrui Liu, Xinju Wu, Pingping Zhang, Shiqi Wang, Zhu Li, and Sam Kwong. Compgs: Efficient 3d scene representation via compressed gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, MM â24, page 2936â2944, New York, NY, USA, 2024. Association for Computing Machinery. 1, 2

[27] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 1, 3

[28] G. Nigel N. Martin, Glen G. Langdon, and Stephen J. P. Todd. Arithmetic codes for constrained channels. IBM Journal of Research and Development, 27(2):94â106, 1983. 4

[29] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Computer Vision â ECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part I, page 405â421, Berlin, Heidelberg, 2020. Springer-Verlag. 1, 6

[30] Wieland Morgenstern, Florian Barthel, Anna Hilsmann, and Peter Eisert. Compact 3d scene representation via selforganizing gaussian grids. In Computer Vision â ECCV 2024: 18th European Conference, Milan, Italy, September 29âOctober 4, 2024, Proceedings, Part LXXXV, page 18â34, Berlin, Heidelberg, 2024. Springer-Verlag. 1

[31] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4), July 2022. 4

[32] Simon Niedermayr, Josef Stumpfegger, and Rudiger West- Â¨ ermann. Compressed 3d gaussian splatting for accelerated novel view synthesis. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10349â10358, 2024. 2

[33] Francesco Di Sario, Riccardo Renzulli, Marco Grangetto, Akihiro Sugimoto, and Enzo Tartaglione. Gode: Gaussians on demand for progressive level of detail and scalable compression. ArXiv, abs/2501.13558, 2025. 1, 6, 7

[34] Yuang Shi, Simone Gasparini, Geraldine Morin, and Â´ Wei Tsang Ooi. LapisGS: Layered progressive 3D Gaussian splatting for adaptive streaming. In International Conference on 3D Vision, 3DV 2025, Singapore, March 25-28, 2025. IEEE, 2025. 1, 10

[35] Yuan-Chun Sun, Yuang Shi, Cheng-Tse Lee, Mufeng Zhu, Wei Tsang Ooi, Yao Liu, Chun-Ying Huang, and Cheng-Hsin Hsu. LTS: A DASH streaming system for dynamic

multi-layer 3D Gaussian splatting scenes. In The 16th ACM Multimedia Systems Conference, MMSys 2025, 2025. ACM, 2025. 10

[36] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPSâ17, page 6309â6318, Red Hook, NY, USA, 2017. Curran Associates Inc. 3

[37] Henan Wang, Hanxin Zhu, Tianyu He, Runsen Feng, Jiajun Deng, Jiang Bian, and Zhibo Chen. End-to-end ratedistortion optimized 3d gaussian representation. In Computer Vision â ECCV 2024: 18th European Conference, Milan, Italy, September 29âOctober 4, 2024, Proceedings, Part LVIII, page 76â92, Berlin, Heidelberg, 2024. Springer-Verlag. 2

[38] Yufei Wang, Zhihao Li, Lanqing Guo, Wenhan Yang, Alex C. Kot, and Bihan Wen. Contextgs: compact 3d gaussian splatting with anchor level context model. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS â24, Red Hook, NY, USA, 2024. Curran Associates Inc. 1, 3

[39] Yasi Wang, Hongxun Yao, and Sicheng Zhao. Auto-encoder based dimensionality reduction. Neurocomputing, 184:232â 242, 2016. RoLoD: Robust Local Descriptors for Computer Vision 2014. 2

[40] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600â612, 2004. 2

[41] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, and Dahua Lin. Bungeenerf: Progressive neural radiance field for extreme multi-scale scene rendering. In Computer Vision â ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23â27, 2022, Proceedings, Part XXXII, page 106â122, Berlin, Heidelberg, 2022. Springer-Verlag. 6

[42] Runyi Yang, Zhenxin Zhu, Zhou Jiang, Baijun Ye, Xiaoxue Chen, Yifei Zhang, Yuantao Chen, Jian Zhao, and Hao Zhao. Spectrally pruned gaussian fields with neural compensation, 2024. 1

[43] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An endto-end neural audio codec, 2021. 3

[44] Yu-Ting Zhan, Cheng-Yuan Ho, Hebi Yang, Yi-Hsin Chen, Jui Chiu Chiang, Yu-Lun Liu, and Wen-Hsiao Peng. CAT-3DGS: A context-adaptive triplane approach to ratedistortion-optimized 3DGS compression. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR), 2025. 1

[45] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586â595, 2018. 2

[46] Yangming Zhang, Wenqi Jia, Wei Niu, and Miao Yin. Gaussianspa: An âoptimizing-sparsifyingâ simplification framework for compact and high-quality 3d gaussian splatting. In

Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 26673â26682, June 2025. 2

[47] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang, Cheng Peng, Rama Chellappa, and Deliang Fan. Lp-3dgs: learning to prune 3d gaussian splatting. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS â24, Red Hook, NY, USA, 2024. Curran Associates Inc. 1, 2

[48] Wenhao Zhao, Qiran Zou, Rushi Shah, and Dianbo Liu. Representation collapsing problems in vector quantization. In Neurips Safe Generative AI Workshop 2024, 2024. 5

[49] Brent Zoomers, Maarten Wijnants, Ivan Molenaers, Joni Vanherck, Jeroen Put, Lode Jorissen, and Nick Michiels. PRoGS: Progressive Rendering of Gaussian Splats . In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3118â3127, Los Alamitos, CA, USA, Mar. 2025. IEEE Computer Society. 1

[50] M. Zwicker, H. Pfister, J. van Baar, and M. Gross. Ewa volume splatting. In Proceedings Visualization, 2001. VIS â01., pages 29â538, 2001. 2