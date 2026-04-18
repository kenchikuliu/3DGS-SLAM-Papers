# Region-Adaptive Learned Hierarchical Encoding for 3D Gaussian Splatting Data

Shashank N. Sridharaâ, Birendra Kathariyaâ , Fangjun Puâ , Peng Yinâ , Eduardo Pavezâ, Antonio Ortegaâ

âUniversity of Southern California Los Angeles, CA, USA

â Dolby Laboratories, Inc. Sunnyvale, CA, USA

## Abstract

We introduce Region-Adaptive Learned Hierarchical Encoding (RALHE) for 3D Gaussian Splatting (3DGS) data. While 3DGS has recently become popular for novel view synthesis, the size of trained models limits its deployment in bandwidth-constrained applications such as volumetric media streaming. To address this, we propose a learned hierarchical latent representation that builds upon the principles of âoverfittedâ learned image compression (e.g., Cool-Chic and C3) to efficiently encode 3DGS attributes. Unlike images, 3DGS data have irregular spatial distributions of Gaussians (geometry) and consist of multiple attributes (signals) defined on the irregular geometry. Our codec is designed to account for these differences between images and 3DGS. Specifically, we leverage the octree structure of the voxelized 3DGS geometry to obtain a hierarchical multi-resolution representation. Our approach overfits latents to each Gaussian attribute under a global rate constraint. These latents are decoded independently through a lightweight decoder network. To estimate the bitrate during training, we employ an autoregressive probability model that leverages octreederived contexts from the 3D point structure. The multi-resolution latents, decoder, and autoregressive entropy coding networks are jointly optimized for each Gaussian attribute. Experiments demonstrate that the proposed RALHE compression framework achieves a rendering PSNR gain of up to 2dB at low bitrates (â¤ 1 MB) compared to the baseline 3DGS compression methods.

## 1 Introduction

3D Gaussian Splatting (3DGS) has recently emerged as a state-of-the-art approach for image-based 3D scene reconstruction [1]. Compared to Neural Radiance Fields (NeRFs) [2] and Plenoxels [3], 3DGS enables much faster rendering while preserving high visual quality. In 3DGS, a scene or object is modeled explicitly as a collection of 3D Gaussians, each parameterized by a mean vector (position) and covariance matrix, along with view-dependent color and opacity as attributes [1]. Due to its efficiency in training and rendering, 3DGS has gained rapid adoption and is expected to play a central role in future 3D content creation and immersive applications [4]. Despite its efficiency, the large storage requirements for the trained 3DGS model remain a critical bottleneck in applications such as volumetric media streaming, underscoring the need for effective compression techniques [5].

Compression methods for 3DGS can be grouped into two categories. First, 3DGS model compaction approaches integrate compression into model training, reducing the size of the trained 3DGS model by using techniques such as Gaussian pruning, vector quantization, and entropy-constrained optimization of Gaussian attributes [6, 7, 8, 9].

<!-- image-->  
Figure 1: Overview of the proposed 3DGS compression framework. We first voxelize the Gaussian mean positions $( \pmb { \mu } _ { i } \in \mathbb { R } ^ { 3 } )$ and construct an octree to provide a hierarchical representation of the geometry. We encode the voxelized mean positions $( \tilde { \pmb { \mu } } _ { i } \in \mathbb { R } ^ { 3 } )$ using GPCC in lossless mode and covariances $( \tilde { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 } )$ using vector quantization. We encode the attribute dataâopacity and spherical harmonic coefficientâusing RALHE, where we jointly train latents, decoder networks, and entropy models for each 3DGS attribute.

These methods do not use explicit rate-distortion (RD) optimization or any form of spatial transform, limiting their ability to exploit the spatial correlations of 3DGS attributes. In addition, these methods train an entirely new 3DGS model for each RD point. Second, the post-training approaches decouple the compression pipeline from the 3DGS model training [5]. Unlike 3DGS model compaction methods, post-training approaches do not require retraining. Instead, they encode a pre-trained 3DGS model at different bitrates. This includes encoding the attributes mapped to the 2D plane using standard video codecs [10] (for which fast implementations are available), and applying fixed transform coding tools such as geometry-based point cloud compression (GPCC) and graph Fourier transforms (GFT) [11, 12, 13]. These methods achieve competitive RD performance with lower encoding and decoding complexities than model compaction techniques but rely on fixed, data-independent transforms, limiting their ability to fully exploit the distinct spatial correlation of different Gaussian attributes. Our approach follows the post-training compression paradigm, but similar to [13], includes voxelization and lightweight retraining before attribute encoding. The key advantage of our method is that latent representations (which play a similar role as transform domain representations in conventional methods) are learned for each 3DGS attribute, and thus can be adapted to their different spatial correlation characteristics, which is not possible if a fixed transform is used for all attributes.

In this work, we propose a learning-based codec for efficient 3DGS compression, aiming to: (i) learn data-dependent latent representations for each 3DGS attribute, rather than applying the same fixed transform (e.g., RAHT as in [13]) for all attributes; (ii) encode these representations efficiently at different bitrates with rateconstrained optimization; and (iii) achieve low decoder complexity, comparable to that of fixed transform coding methods such as GPCC. While autoencoder-based frameworks are widely used in learned compression [14] and have been extended to unstructured 3D data such as point clouds [15] and 3DGS data [16], their high decoder complexity limits deployment on resource-constrained devices. Thus, to achieve our objectives, we introduce an âoverfittedâ compression framework for 3DGS, inspired by learned image codecs such as Cool-Chic and C3 [17, 18], which achieve low decoder complexity comparable to conventional video codecs such as HEVC. Our framework adopts the same principle based on jointly learning a lightweight decoder, the multiresolution latent representation, and an entropy coding network for the target signal.

The main challenges in extending approaches such as [17, 18] to 3DGS data are: 1) Gaussians are irregularly placed in 3D space, resulting in no explicit spatial ordering or regular multi-resolution representation, and 2) instead of 3 colors per pixel, 3DGS consists of 17 attributes, including direction-dependent color attributes, i.e., the spherical harmonic (SH) coefficients, as well as opacity, which exhibit distinct spatial correlations and have different effects on rendering quality. To address the first challenge, we leverage the octree structure [19], which offers a natural multiresolution representation for the attributes that can be used to learn the latents. In addition, traversing an octree following the Morton order can exploit the spatial relations embedded in the octree [19, 20] to provide causal contexts for the autoregressive entropy model used in rate-constrained optimization and entropy coding of latents. To address the second challenge, we design separate latent representations, decoders, and entropy coders for each attribute and optimize them jointly by defining a single rate-distortion function. Although our method introduces higher encoding complexity compared to fixed transform coding tools (e.g., GPCC-GS [13]), it achieves superior rateâdistortion performance while maintaining comparable decoding complexity.

The rest of the paper is organized as follows. 3DGS preliminaries and proposed RALHE compression framework in Section 2. Experimental results and conclusions are in Section 3 and Section 4, respectively.

## 2 Proposed RALHE compression framework

## 2.1 3DGS preliminaries

3DGS data represents a scene as a set of N Gaussians, each defined by a mean position $\mu _ { i } \in \mathbb { R } ^ { 3 }$ , covariance matrix $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ , 16 spherical harmonic (SH) coefficients $\mathbf { C } _ { i } \in$ $\mathbb { R } ^ { 1 6 \times 3 }$ for view-dependent color, and opacity $\alpha _ { i } \colon \mathsf { M } = \{ \pmb { \mu } _ { i } , \pmb { \Sigma } _ { i } , \mathbf { C } _ { i } , \alpha _ { i } \} _ { i = 1 } ^ { N }$ . The model M is optimized using training images $\{ \tilde { \mathbf { I } } _ { i } \} _ { i = 1 } ^ { l }$ and camera parameters $\{ \pmb { \theta } _ { i } \} _ { i = 1 } ^ { l }$ . Quality is evaluated on test images $\{ { \bf I } _ { i } \} _ { i = 1 } ^ { m }$ with corresponding views $\{ \pmb { \theta } _ { i } \} _ { i = 1 } ^ { m }$ , where $R _ { \theta }$ renders the scene from view Î¸ [1]. The training and testing losses are give by:

$$
\mathcal { L } _ { \mathrm { t r a i n } } ( \mathsf { M } ) = \frac { 1 } { l } \sum _ { i = 1 } ^ { l } \| R _ { \bar { \theta } _ { i } } ( \mathsf { M } ) - \tilde { \mathbf { I } } _ { i } \| _ { F } ^ { 2 } , \quad \mathcal { L } _ { \mathrm { t e s t } } ( \mathsf { M } ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| R _ { \theta _ { i } } ( \mathsf { M } ) - \mathbf { I } _ { i } \| _ { F } ^ { 2 } .\tag{1}
$$

## 2.2 Overview of the proposed codec

We categorize the 3DGS data into geometry, comprising mean positions and covariances $\mathsf { G } = \{ \mu _ { i } , \Sigma _ { i } \} _ { i = 1 } ^ { N }$ , and attributes, which consist of SH coefficients and opacities $\mathsf { A } = \{ \mathbf { C } _ { i } , \alpha _ { i } \} _ { i = 1 } ^ { N }$ . Since the SH coefficients $\mathbf { C } _ { i } \in \mathbb { R } ^ { 3 \times 1 6 }$ contain 16 RGB triplets for an order-3 representation, there are a total of 17 attributes.

<!-- image-->  
Figure 2: (left) hierarchical representation of 3D geometry using octree and corresponding latents, (right) reconstructing the attributes from quantized latents at different resolutions. The quantized latents are upsampled and fed into the decoder.

Our proposed 3DGS compression framework consists of three stages: (i) preprocessing, which involves voxelization and finetuning, (ii) geometry compression, and (iii) attribute compression. Voxelization allows us to use the octree data structure to encode the positions efficiently. Therefore, similar to [13], we first voxelize the Gaussian mean positions $( \pmb { \mu } _ { i } )$ to a resolution determined by the maximum octree depth L. Gaussians mapped to the same voxel are merged, yielding new positions $\{ \tilde { \pmb { \mu } } _ { i } \} _ { i = 1 } ^ { M }$ where $M \leq N$ . Following [13], we perform a lightweight constrained retraining, by minimizing $\mathcal { L } _ { \mathrm { t r a i n } }$ in (1) while fixing $\{ \tilde { \pmb { \mu } } _ { i } \} _ { i = 1 } ^ { M }$ , to obtain $\tilde { \mathsf { M } } = \{ \tilde { \pmb { \mu } } _ { i } , \tilde { \Sigma } _ { i } , \tilde { \mathbf { C } } _ { i } , \tilde { \alpha } _ { i } \} _ { i = 1 } ^ { M }$ , ensuring that the rendering quality of MË is close to that of M. The retraining is performed only once as a pre-processing step, and the resulting model MË is compressed at various bitrates. The voxelized positions are encoded losslessly using GPCC [21]. The covariances $\{ \tilde { \Sigma } _ { i } \} _ { i = 1 } ^ { M }$ are compressed using vector quantization with a codebook size chosen to preserve rendering quality [22].

We denote the finetuned 3DGS attributes as $\tilde { \mathsf { A } } = \{ \tilde { \mathbf { C } } _ { i } ^ { ( 1 ) } , \tilde { \mathbf { C } } _ { i } ^ { ( 1 ) } , \hdots , \tilde { \mathbf { C } } _ { i } ^ { ( 1 6 ) } , \tilde { \alpha } _ { i } \} _ { i = 1 } ^ { M }$ and the matrix containing the n-th attribute of all points as $\mathbf { A } _ { n }$ , where $n \in \{ 1 , \ldots , 1 7 \}$ ; $\mathbf { A } _ { n }$ is $M \times 3$ for $n \in \{ 1 , \ldots , 1 6 \}$ (there are three color components) and $\mathbf { A } _ { 1 7 }$ is $M \times 1$ â¢ The i-th row of each $\mathbf { A } _ { n }$ contains the features for the i-th Gaussian when scanning the points following the Morton order. We compress $\tilde { \mathsf { A } }$ using the proposed RALHE. For each attribute $\mathbf { A } _ { n } ,$ we have (i) latents ${ \bf y } _ { n }$ , (ii) an entropy model $f _ { \Psi n }$ , and (iii) a decoder network $f _ { \theta _ { n } }$ . Latents, entropy models and decoder networks are jointly trained, for all attributes via backpropagation by minimizing the rate-distortion cost:

$$
\operatorname* { m i n } \sum _ { n = 1 } ^ { 1 7 } \mathcal { D } ( \mathbf { A } _ { n } , \hat { \mathbf { A } } _ { n } ) + \lambda R ( \hat { \mathbf { y } } _ { n } ) .\tag{2}
$$

Here D is the distortion between the original $\left( \mathbf { A } _ { n } \right)$ and the reconstructed $( \hat { \mathbf { A } } _ { n } )$ attributes, and R is the estimated bitrate of the quantized latents ${ \hat { \mathbf { y } } } _ { n }$ . The overview of our proposed codec is shown in Figure 1. Overall, we train a latent representation, a decoder network, and an entropy coding network for each of the 17 3DGS attributes. This approach exploits the distinct spatial correlation of each 3DGS attribute effectively. In what follows, we discuss each component of RALHE in detail.

## 2.3 Multiresolution latent representation

In images, multi-resolution representations are typically obtained using a pyramid structure, where coarser versions of the image are progressively produced by applying low-pass filtering followed by downsampling by a factor of two [23, 17]. For 3DGS data, we instead exploit the octree structure, which inherently provides a hierarchical multi-resolution organization of the 3D space, as illustrated in Figure 2a. We leverage this property to define latent representations at multiple resolutions. We consider k resolutions, from the finest octree resolution L to the coarse resolution $L - ( k - 1 )$ If the bounding box of the 3D points has a volume of $W \times W \times W$ , the first level of the octree divides the volume into $2 ^ { 3 }$ cubes, each with a volume $W / 2 \times W / 2 \times W / 2$ At each subsequent level, only the occupied cubes are divided [19, 24]. For the finest resolution with L levels of partitioning, the resulting cubes have a volume of $W / 2 ^ { L } \times W / 2 ^ { L } \times W / 2 ^ { L }$ , which represents the resolution of the voxelized 3D points. The coarse resolution voxels will have a volume of $W / 2 ^ { L - ( k - 1 ) } \times W / 2 ^ { L - ( k - 1 ) } \times \mathrm { \hat { W } / 2 ^ { L - ( k - 1 ) } }$ This multi-resolution depends solely on the voxelized positions $\{ \tilde { \pmb { \mu } } _ { i } \} _ { i = 1 } ^ { M }$ and it is the same for all attributes [20].

The matrix $\mathbf { A } _ { n } ,$ containing the nth attribute for all Gaussians, is represented by a set of k latent vectors at different resolutions $\mathbf y _ { n } = \{ \mathbf y _ { n } ^ { L } , \mathbf y _ { n } ^ { L - 1 } , \cdot \cdot \cdot \mathbf y _ { n } ^ { L - \left\{ k - 1 \right\} } \}$ , where ${ \bf y } _ { n } ^ { L - j }$ is the latent vector obtained from voxels at resolution $L - j$ . The dimension of each latent vector depends on the number of voxels at its respective resolution.

## 2.4 RALHE decoder module

The RALHE decoder module is depicted in Figure 2b. To recover $\mathbf { A } _ { n } .$ each of the quantized latent representations is upsampled to the finest resolution: $\hat { \mathbf { v } } _ { n } ^ { L - j } =$ $\mathsf { U } _ { j } ( \hat { \mathbf { y } } _ { n } ^ { L - j } )$ , where $\mathsf { U } _ { j } ( \cdot )$ denotes the upsampling operation needed to go from resolution $L - j$ to resolution L, therefore each $\hat { \mathbf { v } } _ { n } ^ { L - j } \in \mathbb { R } ^ { \bar { M } }$ . The upsampled latents corresponding to the n-th attribute of the i-th Gaussian $\hat { \mathbf { z } } _ { n , i } = ( \hat { \mathbf { v } } _ { n } ^ { L } ( i ) , \hat { \mathbf { v } } _ { n } ^ { L - 1 } ( i ) , \cdot \cdot \cdot \hat { \mathbf { v } } _ { n } ^ { L - ( k - 1 ) } ( i ) ) ^ { \top }$ are then passed to the decoder network $f _ { \theta _ { n } }$ to reconstruct the target attribute $\hat { \mathbf { A } } _ { n } ( i ) =$ $f _ { \theta _ { n } } ( \hat { \mathbf { z } } _ { n , i } )$ The decoder $f _ { \theta _ { n } }$ is a lightweight 4-layer neural network, with 3D sparse convolutions in the last two layers. While image upsampling can be carried out using bilinear or trilinear interpolation techniques [17], 3D upsampling is not straightforward due to the irregular geometry. In this work, we adopt a simple octree-based strategy based on copying, where in order to upsample the latents $\hat { \mathbf { y } } _ { n } ^ { L - j }$ by a factor of $2 ^ { j } \times 2 ^ { j } \times 2 ^ { j }$ , we partition the 3D points at the finest resolution into blocks of size $2 ^ { j } \times 2 ^ { j } \times 2 ^ { j }$ and copy the latent values from the resolution $L - j$ to all points within the corresponding block of resolution L to obtain $\hat { \mathbf { v } } _ { n } ^ { L - j } = \mathsf { U } _ { j } ( \hat { \mathbf { y } } _ { n } ^ { L - j } )$

## 2.5 Autoregressive probability model for entropy coding

The quantized hierarchical latents are encoded auto-regressively by learning a probability model $p _ { \Psi _ { n } }$ for each 3DGS attribute. In the case of images, we typically use a rectangular context window to auto-regressively encode the pixel latents. In contrast, our method obtains causal contexts by traversing 3D space using the Morton-order (Z-order). Morton order scanning is equivalent to performing a breadth-first traversal of the octree [19], as illustrated in Figure 3, and it ensures that neighboring attributes are spatially close to each other in 3D space. Therefore, for a voxel i in level $( L - j )$ , the causal context for the attribute sorted in Morton order is given by $\mathbf { x } _ { n , i } ^ { L - j } = ( \hat { \mathbf { y } } _ { n } ^ { L - j } ( i - 1 ) , \hat { \mathbf { y } } _ { n } ^ { L - j } ( i - 2 ) \cdot \cdot \cdot \hat { \mathbf { y } } _ { n } ^ { L - j } ( i - w ) ) ^ { \top }$ , where w is the context window size. We rely on a factorized model to obtain the joint distribution of the quantized latents $p _ { \psi _ { n } } ( \hat { \mathbf { y } } _ { n } )$ , as widely used in learned image compression [14, 17, 18], where each factor is the distribution of the latent $\hat { \mathbf { y } } _ { n } ( i ) ^ { L - j }$ conditioned on its context $\mathbf { x } _ { n , i } ^ { L - j }$ , thus

<!-- image-->  
Figure 3: Autoregressive probability model for entropy coding

$$
p _ { \psi _ { n } } ( \hat { \mathbf { y } } _ { n } ) = \prod _ { i , j } p _ { \psi _ { n } } ( \hat { \mathbf { y } } _ { n } ( i ) ^ { L - j } | \mathbf { x } _ { n , i } ^ { L - j } ) .\tag{3}
$$

Once the factorized probability model above is defined, we follow [17] to estimate the MLP $f _ { \Psi _ { n } }$ . Finally, the rate term in (2) is given by,

$$
R ( \hat { \mathbf { y } } _ { n } ) = - \log _ { 2 } p _ { \Psi _ { n } } ( \hat { \mathbf { y } } _ { n } ) = - \log _ { 2 } \prod _ { i , j } p _ { \Psi _ { n } } ( \hat { \mathbf { y } } _ { n } ( i ) ^ { L - j } | \mathbf { x } _ { n , i } ^ { L - j } ) .\tag{4}
$$

The learned entropy models $f _ { \Psi _ { n } }$ serve two purposes: during training, it is used to estimate and constrain the rate of the latent representation, and during encoding it is provided as parameters for entropy coding.

## 2.6 Joint optimization of RALHE for 3DGS attributes

The optimization in (2) can now be written as:

$$
\operatorname* { m i n } _ { \mathbf { y } _ { n } , \theta _ { n } , \Psi _ { n } } \sum _ { n = 1 } ^ { 1 7 } \mathscr { D } \big ( \mathbf { A } _ { n } , ~ f _ { \theta _ { n } } \big ( \mathsf { U } ( \hat { \mathbf { y } } _ { n } ) \big ) - \lambda \sum _ { n } ^ { 1 7 } \log _ { 2 } p _ { \Psi _ { n } } ( \hat { \mathbf { y } } _ { n } ) ,\tag{5}
$$

where, $\mathsf { U } ( \hat { \mathbf { y } } _ { n } ) = [ \hat { \mathbf { v } } _ { n } ^ { L } , \hat { \mathbf { v } } _ { n } ^ { L - 1 } , \cdot \cdot \cdot , \hat { \mathbf { v } } _ { n } ^ { L - ( k - 1 ) } ]$ Since quantization is non differentiable, we follow standard practice and the optimization is made quantization-aware during training by adding uniform noise to the latents, $\hat { \mathbf { y } } _ { n } = ( \mathbf { y } _ { n } + \mathbf { n } )$ , where $\begin{array} { r } { \mathbf { n } \sim \mathcal { U } \left( - \frac { 1 } { 2 } , \frac { 1 } { 2 } \right) } \end{array}$ as described in [17, 14]. After training, the latents are uniformly quantized resulting in $\hat { \mathbf { y } } _ { n } = \mathcal { Q } ( \mathbf { y } _ { n } )$ . The weights and biases of the trained decoder network $\theta _ { n }$ and entropy coder network $\Psi _ { n }$ are quantized and encoded using an arithmetic coder. Subsequently, the quantized latents of all attributes are autoregressively encoded under a Laplace distribution, whose mean and scale parameters are estimated by the quantized entropy model $f _ { \Psi _ { n } }$ . The same quantized entropy model is employed during both encoding and decoding of latents. Table 1 provides a summary of all encoded quantities.

Table 1: Details of overhead parameters that are transmitted to the decoder.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>decoder network $f _ { \theta _ { n } }$ </td><td rowspan=1 colspan=2>entropy coding network $f _ { \Psi _ { n } }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { \ o p a c i t y } { : } \alpha _ { i } \in \mathbb { R } } }$ </td><td rowspan=1 colspan=1> $\mathbf { \overline { { S H } } } { : } mathbf { C } _ { i } \in \mathbb { R } ^ { 1 6 \times 3 }$ </td><td rowspan=1 colspan=1> ${ \mathrm { o p a c i t y } } { : } \alpha _ { i } \in \mathbb { R }$ </td><td rowspan=1 colspan=1> $\overline { { \mathrm { S H } { : } \mathbf { C } _ { i } } } \in \mathbb { R } ^ { 1 6 \times 3 }$ </td></tr><tr><td rowspan=1 colspan=1># parameters</td><td rowspan=1 colspan=1>639</td><td rowspan=1 colspan=1> $\overline { { 6 3 9 \mathrm { ~ x ~ } 1 6 } }$ </td><td rowspan=1 colspan=1>578</td><td rowspan=1 colspan=1> $5 7 8 \mathrm { ~ x ~ } 1 6$ </td></tr></table>

<!-- image-->  
(a) Low bitrate (color): 0.034 bpp  
(b) High bitrate (color): 2.68bpp  
Figure 4: Visualization of learned latents for color attributes i.e., $0 ^ { \mathrm { t h } }$ order SH: $\mathbf { C } _ { i } ^ { ( 0 ) }$

## 3 Experiments

We evaluate the proposed RALHE compression framework on 3DGS models (mic, ficus, and materials) from the synthetic-NeRF dataset [2], each containing more than 200K Gaussians. Following [13], we use uniform voxelization and lightweight finetuning as preprocessing steps. The voxelization depth $L _ { \mathrm { v o x } }$ is selected heuristically for each model based on the point distribution and the sensitivity of rendering PSNR to quantization. Specifically, we set $L _ { \mathrm { v o x } } = 1 4$ for mic and materials, and $L _ { \mathrm { v o x } } = 1 0$ for ficus based on the rendering quality achieved after finetuning. The latent representation ${ \bf y } _ { n }$ is organized using the octree into a multi-resolution hierarchy of 5 latent grids $\{ \mathbf { y } _ { n } ^ { L } , \mathbf { y } _ { n } ^ { L - 1 } , \ldots , \mathbf { y } _ { n } ^ { L - 4 ) } \}$ . Finally, the context window size of the autoregressive entropy coding network is fixed to $w = 1 6$ . The RALHE framework was trained on an Nvidia RTX-2080 GPU for 10K iterations.

We first provide a visualization of the learned latents for the color attribute at different bitrates for the ficus model in Figure 4. For visualization, low-resolution latents are upsampled to match the full resolution. At high bitrates, the hierarchical latents exhibit a clear separation of roles: low-resolution latents capture coarse semantic structure (e.g., distinguishing leaves from the pot and branches), while highresolution latents provide fine details. In contrast, at low bitrates, the representational capacity is limited, and even high-resolution latents primarily encode low-frequency information, resulting in the loss of fine details.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 5: RD cuve comparison with state-of-the-art 3DGS compression methods.

## 3.1 Comparison with state-of-the-art 3DGS compression methods

We compare the proposed RALHE codec with state-of-the-art 3DGS compression approaches. For model compaction baselines, we consider RDO-GS [8], Compact-GS [25], and HAC++ [7]. For post-training baselines, we include MesonGS [11] and GPCC-GS with adaptive voxelization[13]. As shown in Figure 5, our proposed RALHE codec outperforms the baselines for mic and ficus models by a large margin. For materials, however, our method achieves RâD performance comparable to existing model compaction methods. This is because the materials model contains multiple spatially separated objects. In such cases, the octree structure groups Gaussians from different objects into the same blocks, which reduces reconstruction accuracy and leads to less reliable bitrate estimation.

Finally, we compare the rendering PSNR of the proposed RALHE at a fixed bitrate of 1 MB with RDO-GS [8] and GPCC-GS [13]1. As shown in Table 2, RALHE achieves a PSNR gain of 0.3â0.7 dB over RDO-GS and up to 2.0 dB over GPCC-GS at the same bitrate. We also report the decoding time per RâD point in Table 2. The GPCC-GS decoder performs entropy decoding and inverse transforms of the coefficients, while RDO-GS reconstructs attributes from the codebook index. For fairness, all baseline decoders run on the CPU. RALHEâs decoding time lies between the two, with GPCC-GS being the fastest due to its optimized C++ implementation.

## 4 Conclusion

In this work, we proposed a 3DGS compression framework, where the geometry (positions and Gaussians) is encoded first, followed by 3DGS attribute encoding, conditioned on the decoded geometry. Our main contribution is RALHE to encode 3DGS attributes, which âoverfitsâ a multi-resolution latent representation, a lightweight decoder, and an entropy coder for a given target signal. Since 3DGS consists of multiple attributes (signals), we optimize a separate latent representation, decoder, and entropy coder jointly for each attribute. We leverage the octree structure to obtain a multi-resolution representation for latents and to derive contexts for the autoregressive probability model used for entropy coding. The proposed RALHE decoder reconstructs the original attribute using quantized latents that are upsampled to the finest resolution using the octree. The proposed framework, combined with existing geometry coding solutions, achieves state-of-the-art coding performance.

Table 2: Comparison of rendering PSNR at a fixed bitrate of 1MB and decoding time of RALHE with 3DGS model compaction and post-training compression method
<table><tr><td rowspan="2">model</td><td colspan="3">rendering psnr (dB) at 1MB</td><td colspan="3">decoding time (seconds)</td></tr><tr><td>RDO-GS[8]</td><td>GPCC-GS [13]</td><td>RALHE</td><td>RDO-GS[8]</td><td>GPCC-GS[ [13]</td><td>RALHE</td></tr><tr><td>mic</td><td>35.25</td><td>32.06</td><td>36.03</td><td>5.92</td><td>2.38</td><td>4.78</td></tr><tr><td>ficus</td><td>34.64</td><td>33.58</td><td>34.93</td><td>6.38</td><td>3.24</td><td>5.05</td></tr><tr><td>materials</td><td>29.47</td><td>27.16</td><td>29.63</td><td>4.85</td><td>2.73</td><td>4.43</td></tr></table>

## 5 References

[1] B. Kerbl, G. Kopanas, T. LeimkÂ¨uhler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[3] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, âPlenoxels: Radiance fields without neural networks,â in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5491â5500.

[4] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, and Y. He, â3d gaussian splatting as new era: A survey,â IEEE Transactions on Visualization and Computer Graphics, pp. 1â20, 2024.

[5] M. T. Bagdasarian, P. Knoll, Y.-H. Li, F. Barthel, A. Hilsmann, P. Eisert, and W. Morgenstern, â3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods,â Computer Graphics Forum, 2025.

[6] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 654â20 664.

[7] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, âHac: Hash-grid assisted context for 3d gaussian splatting compression,â in European Conference on Computer Vision. Springer, 2024, pp. 422â438.

[8] H. Wang, H. Zhu, T. He, R. Feng, J. Deng, J. Bian, and Z. Chen, âEnd-to-end ratedistortion optimized 3d gaussian representation,â in European Conference on Computer Vision. Springer, 2024, pp. 76â92.

[9] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, Z. Wang et al., âLightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps,â Advances in neural information processing systems, vol. 37, pp. 140 138â140 158, 2024.

[10] S. Lee, F. Shu, Y. Sanchez, T. Schierl, and C. Hellge, âCompression of 3d gaussian splatting with optimized feature planes and standard video codecs,â arXiv preprint arXiv:2501.03399, 2025.

[11] S. Xie, W. Zhang, C. Tang, Y. Bai, R. Lu, S. Ge, and Z. Wang, âMesongs: Post-training compression of 3d gaussians via efficient attribute transformation,â in European Conference on Computer Vision. Springer, 2024, pp. 434â452.

[12] H. Huang, W. Huang, Q. Yang, Y. Xu, and Z. Li, âA hierarchical compression technique for 3d gaussian splatting compression,â in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1â5.

[13] C. Wang, S. N. Sridhara, E. Pavez, A. Ortega, and C. Chang, âAdaptive voxelization for transform coding of 3d gaussian splatting data,â in 2025 IEEE International Conference on Image Processing (ICIP), 2025, pp. 2414â2419.

[14] J. BallÂ´e, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, âVariational image compression with a scale hyperprior,â in International Conference on Learning Representations, 2018.

[15] X. Sheng, L. Li, D. Liu, Z. Xiong, Z. Li, and F. Wu, âDeep-pcac: An end-to-end deep lossy compression framework for point cloud attributes,â IEEE Transactions on Multimedia, vol. 24, pp. 2617â2632, 2022.

[16] Y. Chen, Q. Wu, M. Li, W. Lin, M. Harandi, and J. Cai, âFast feedforward 3d gaussian splatting compression,â arXiv preprint arXiv:2410.08017, 2024.

[17] T. Ladune, P. Philippe, F. Henry, G. Clare, and T. Leguay, âCool-chic: Coordinatebased low complexity hierarchical image codec,â in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 13 469â13 476.

[18] H. Kim, M. Bauer, L. Theis, J. R. Schwarz, and E. Dupont, âC3: High-performance and low-complexity neural compression from a single image or video,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 9347â9358.

[19] D. Meagher, âGeometric modeling using octree encoding,â Computer Graphics and Image Processing, vol. 19, no. 2, pp. 129 â 147, 1982.

[20] E. Pavez, P. A. Chou, R. L. de Queiroz, and A. Ortega, âDynamic polygon clouds: representation and compression for vr/ar,â APSIPA Transactions on Signal and Information Processing, vol. 7, p. e15, 2018.

[21] H. Liu, H. Yuan, Q. Liu, J. Hou, and J. Liu, âA comprehensive study and comparison of core technologies for mpeg 3-d point cloud compression,â IEEE Transactions on Broadcasting, vol. 66, no. 3, pp. 701â717, 2020.

[22] S. Niedermayr, J. Stumpfegger, and R. Westermann, âCompressed 3d gaussian splatting for accelerated novel view synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10 349â10 358.

[23] P. Burt and E. Adelson, âThe laplacian pyramid as a compact image code,â IEEE Transactions on Communications, vol. 31, no. 4, pp. 532â540, 1983.

[24] S. N. Sridhara, E. Pavez, and A. Ortega, âCylindrical coordinates for lidar point cloud compression,â in 2021 IEEE International Conference on Image Processing (ICIP). IEEE, 2021, pp. 3083â3087.

[25] K. Navaneet, K. Pourahmadi Meibodi, S. Abbasi Koohpayegani, and H. Pirsiavash, âCompgs: Smaller and faster gaussian splatting with vector quantization,â in European Conference on Computer Vision. Springer, 2024, pp. 330â349.