# Structured Image-based Coding for Efficient Gaussian Splatting Compression

Pedro Martin, AntÃ³nio Rodrigues, JoÃ£o Ascenso, and Maria Paula Queluz

Instituto de TelecomunicaÃ§Ãµes, Instituto Superior TÃ©cnico, University of Lisbon, 1049-001 Lisbon, Portugal {pedro.martin, antonio.rodrigues, joao.ascenso, paula.queluz}@lx.it.pt

AbstractâGaussian Splatting (GS) has recently emerged as a state-of-the-art representation for radiance fields, combining realtime rendering with high visual fidelity. However, GS models require storing millions of parameters, leading to large file sizes that impair their use in practical multimedia systems. To address this limitation, this paper introduces GS Image-based Compression (GSICO), a novel GS codec that efficiently compresses pre-trained GS models while preserving perceptual fidelity. The core contribution lies in a mapping procedure that arranges GS parameters into structured images, guided by a novel algorithm that enhances spatial coherence. These GS parameter images are then encoded using a conventional image codec. Experimental evaluations on Tanks and Temples, Deep Blending, and Mip-NeRF360 datasets show that GSICO achieves average compression factors of 20.2Ã with minimal loss in visual quality, as measured by PSNR, SSIM, and LPIPS. Compared with state-ofthe-art GS compression methods, the proposed codec consistently yields superior rate-distortion (RD) trade-offs.

Index TermsâGaussian splatting, radiance fields, compression, multimedia coding.

## I. INTRODUCTION

he rise of 3D visual representation has revolutionized multimedia consumption, enabling immersive and interactive experiences. From virtual reality (VR) gaming and virtual museum tours to augmented reality (AR) based surgical procedures for medical training, 3D content has expanded the boundaries of multimedia applications. In recent years, neural radiance fields (NeRFs) [1] have established themselves as a breakthrough paradigm for 3D scene representation and novel view synthesis, offering unprecedented levels of visual realism unattainable with traditional image-based approaches. By modeling the volumetric scene as a radiance function using neural networks (NNs) trained to map spatial coordinates and viewing directions to color and opacity values, NeRF achieves compelling results across synthetic and real-world datasets. However, the computational cost associated with NeRF training and rendering pose critical challenges for practical deployment. To address these challenges, several extensions and alternative representations have been introduced, with particular emphasis on improving training and rendering efficiency while preserving high-quality visual synthesis. Gaussian Splatting (GS) [2] has recently emerged as one of the most promising alternatives to NeRF. Instead of encoding a scene into a neural volumetric function, GS represents 3D content using a set of

anisotropic 3D Gaussian primitives, parameterized by spatial position, scale, rotation, opacity, and spherical harmonic (SH) coefficients for view-dependent color representation. Rendering is performed via a tile-based rasterization process, avoiding the expensive ray-marching approach used in NeRF. This design allows GS to achieve real-time frame rates, while reducing training times and avoiding unnecessary computations in empty 3D space. The seminal 3D Gaussian Splatting (3DGS) method [2], has become a new baseline in radiance field research, inspiring a growing number of variants. GS representations have already been applied to several domains, including immersive VR/AR applications, telepresence, cultural heritage preservation, and large-scale scene reconstruction [3].

Although GS methods offer significant advantages, they demand large amounts of data to represent a scene. A single GS model size may achieve hundreds of megabytes, limiting its use in bandwidth-constrained or storage-limited settings. Existing GS compression methods have primarily combined pruning, quantization, and entropy modeling within joint trainingcompression pipelines. However, such approaches often require retraining or fine-tuning the model, which may not be feasible in many real-world scenarios where the original images or videos used to train the model are no longer available. In fact, in many practical settings, the creator of the 3D content is not the party responsible for preparing and compressing it for distribution. This paper addresses such cases, where compression is applied after the 3D model has already been generated (i.e., post-training), by proposing a solution that is broadly applicable and agnostic to the underlying GS model.

In this context, the main objective of this work is to provide an efficient and versatile solution for compressing GS models without requiring any post-training optimization or fine-tuning of the model. To validate the approach, two widely-used GS baselines are considered, 3DGS [2] and Scaffold-GS [4], demonstrating wide applicability. The proposed compression solution, referred to as GS Image-based Compression (GSICO), converts GS parameters into structured image representations, which are then compressed by a conventional image codec. To further improve compressibility, the design incorporates a dedicated GS mapping algorithm, using a novel Nearest-Neighbor-based Sorting (NNS) that increases spatial coherence within the parameter images, prior to encoding. This imagebased paradigm not only leverages decades of progress in image compression, but also embodies the novelty of treating GS compression through a codec-like perspective; this contrasts with existing GS compression methods that rely heavily on retraining or parameter fine-tuning. The main contributions of this paper can be summarized as follows:

Novel GS Codec: The proposed GSICO codec introduces a novel training-free and model-agnostic compression framework capable of handling quite different GS representations, such as 3DGS and Scaffold-GS. Operating map Sf map strictly post-training, it remains independent of any model optimization stage, enabling seamless integration into a wide range of GS pipelines. Extensive evaluation on widely used radiance field datasets and comparison withvoxel state-of-the-art GS compression benchmarks demonstrates its effectiveness. GSICO achieves average compression factors of 20.2Ã for 3DGS inputs and 8.5Ã for Scaffold-GS inputs, with negligible degradation in rendering quality.

2D GS Mapping Algorithm: A second key contribution is the introduction of a novel parameter mapping algorithm, that clusters and spatially arrange GS parameters in the 2D image domain (using a novel NNS algorithm), to improve local coherence. This strategy substantially reduces entropy, thereby improving compression efficiency when paired with a standard image codec. Furthermore, NNS enforces consistent Gaussian-to-pixel alignment across all parameter maps, ensuring coherent and fast reconstruction during decoding.

The link for the GSICO implementation will be available after acceptance.

## II. RELATED WORK

Research on radiance field representations and their compression has evolved rapidly in recent years. This section reviews the most relevant prior work, focusing on three main directions: i) compression of NeRF-based representations, ii) compression of GS-based representations, and iii) current challenges in GS compression.

## A. NeRF Representation and Compression

NeRF [1] established a new paradigm for novel view synthesis by learning implicit volumetric scene representations using multilayer perceptrons (MLPs). While NeRF methods achieve highly realistic view synthesis, their dependence on dense ray sampling and large NNs makes both training and rendering computationally demanding. To attenuate these costs, several explicit extensions have been proposed, including hashbased encodings, voxel grids, and tensor factorization [5]-[7]. These designs significantly accelerate rendering but often lead to large model sizes. Consequently, compression of NeRF representations has become an active research direction. Early works employed parameter pruning, vector quantization, and entropy coding to reduce the NeRF model size [8], [9], while more recent approaches explored transforms such as Fourier and wavelet bases [10], [11].

## B. GS Representation and Compression

Another step forward in radiance field representations was the introduction of the GS representation. The seminal 3DGS method [2], explicitly models a scene using millions of 3D

<!-- image-->

<!-- image-->  
(a) 3DGS  
(b) Scaffold-GS  
Fig. 1: Illustration of the model representation of the 3DGS and Scaffold-GS methods.

Gaussians, each parametrized by its position, $\mathbf { x } _ { \mathbf { G } } =$ $( x _ { G } , y _ { G } , z _ { G } )$ , scale, $\mathbf { \boldsymbol { s } } = \left( { \boldsymbol { s } } _ { x } , { \boldsymbol { s } } _ { y } , { \boldsymbol { s } } _ { z } \right)$ , rotation (defined by the quaternion representation), $\mathbf { r } = \left( r _ { u } , r _ { x } , r _ { y } , r _ { z } \right)$ , opacity, $\sigma ,$ and SH (for view-dependent color), ${ \bf k } = ( k _ { 1 } , \dots , k _ { 4 8 } )$ . Rendering is performed via differentiable rasterization, enabling real-time visualization and substantially faster training than NeRF, while preserving high visual quality. However, the cost of storing millions of Gaussians, each described by several parameters (many of them with more than one element), results in a large amount of data posing practical challenges for storage and transmission. Scaffold-GS [4] mitigates this limitation by introducing a novel GS representation built on voxel anchors. Instead of storing each Gaussian explicitly, the scene is represented using a sparse set of voxels distributed in 3D space. Each voxel has a learnable anchor whose position, $\mathbf { x } _ { \mathrm { A } } =$ $( x _ { A } , y _ { A } , z _ { A } )$ , defines its spatial location, and a scale factor, $S _ { f }$ determines the voxel size, allowing the representation to adapt to local geometric complexity. Multiple neural Gaussians are associated to each anchor, to model fine-grained geometry and appearance within that anchor voxel. These Gaussians are called neural because their attributes are not stored explicitly but predicted through small MLPs. The spatial position of each neural Gaussian is computed from a vector of learned offset features, $\mathbf { 0 } = ( O _ { 1 } , \dots , O _ { 3 0 } )$ , that encodes displacements relative to the anchor position. Each voxel also includes a vector of anchor features, $\mathbf { A } = ( A _ { 1 } , \dots , A _ { 3 2 } )$ , that encodes the geometry and appearance of its associated neural Gaussians. Small MLPs take as input the offset and anchor features to predict each neural Gaussian scale, rotation, opacity, and color. This twolevel organization forms a hierarchical structure: anchors represent the coarse information of the scene, while the neural Gaussians represent fine scene details. As a result, Scaffold-GS achieves a more compact and efficient scene representation than 3DGS. Fig. 1 illustrates the contrast between 3DGS, which uses individual and independent Gaussians, and Scaffold-GS, which groups them around latent anchors. Together, 3DGS and Scaffold-GS represent the two dominant architectures in GS.

Research on GS compression can be broadly categorized into three strategies: i) pruning-based techniques that reduce model size by discarding redundant Gaussians or SH coefficients [12]- [15]. Pruning is typically guided by heuristics or learned metrics that evaluate each Gaussian perceptual relevance. While pruning can achieve substantial reductions, aggressive removal may considerably degrade the rendering quality; ii) quantization and codebook methods, that represent GS parameters with a limited set of representative symbols (codewords) through some mapping [12]-[14], [16]-[18]. This significantly reduces the model size and can achieve high compression ratios with limited perceptual loss, but requires careful codebook design to avoid visual artifacts; and iii) entropy modeling, that exploits statistical redundancies in Gaussian parameters [12]-[14], [19]. By learning probability distributions across GS parameters, entropy-based methods can produce compact bitstream sizes.

<!-- image-->  
Fig. 2: GSICO encoder (blue) and decoder (orange) framework pipeline.

## C. Limitations of Existing GS Compression Methods

Despite their effectiveness, most existing GS compression methods exhibit a major limitation; they are training-dependent, requiring either full retraining or fine-tuning after compression. This severely constrains their applicability, since it is necessary to have the original images used for training, and significant computational resources to compress the model. In many areas (e.g., point cloud compression), the model is traditionally obtained first and compressed as it is, maintaining as much fidelity to the original model as possible. In addition, many compression methods are tailored to a specific GS representation (e.g., 3DGS or Scaffold-GS), resulting in limited use across different baseline models.

A promising yet underexplored direction is to leverage advanced image coding technologies by mapping Gaussian parameters into 2D layouts compatible with standard image/video codecs [15], [16]. By exploiting spatial correlations through established image/video codecs, such strategy can achieve substantial reductions while avoiding the need for retraining or fine-tuning. Actually, this was explored for other 3D representations such as point clouds (e.g., within the MPEG V-PCC standard [20]). Image/video codecs bring additional advantages: they benefit from decades of optimization, have standardized decoders, and are hardwareaccelerated on most platforms. Further exploration of this direction is crucial to bridge the gap between GS compression research and broader multimedia coding systems. The proposed GSICO codec directly addresses this need by introducing a compression strategy compatible with both 3DGS and Scaffold-GS, the two currently most representative GS model baselines.

## III. GSICO FRAMEWORK AND MODULES

This section presents a comprehensive overview of the GSICO framework, detailing each module of the encoder and decoder pipelines. Fig. 2 illustrates the overall processing flow, whose architecture supports both 3DGS-based and Scaffold-GS-based representations (except for the color-space conversion module, which is specific to 3DGS). The color figures shown beneath each module correspond to the 3DGS representation and are included to offer an intuitive visual interpretation of how GSICO operates on explicit Gaussian primitives.

## A. Walkthrough

The GSICO encoder accepts models produced by both 3DGS-based and Scaffold-GS-based methods, offering flexibility and broad applicability. Table I summarizes the parameters associated with each model. Although Scaffold-GS introduces additional model specific parameters, it generally requires far fewer voxels than the number of Gaussians used in 3DGS, leading to a more compact and efficient representation.

3DGS-based models store, for each Gaussian, its position, rotation, scale, opacity, and SH coefficients. The latter spans four degrees (0 to 3), resulting in 16 coefficients per RGB color channel: one DC coefficient (for degree-0), and three, five, and seven AC coefficients, for degrees 1, 2, and 3, respectively. SH are a set of base functions defined on the sphere surface and, in the context of GS, are used to efficiently model view-dependent color variations for each Gaussian. Their perceptual contribution decreases with increasing degree, as higher-order coefficients correspond to higher frequency variations, analogous to the role of high-frequency components in DCTbased transforms.

GS MODEL REPRESENTATION IN THE GS FILE DATA STRUCTURE  
TABLE I  
(a) 3DGS
<table><tr><td rowspan=1 colspan=1>GS parameter</td><td rowspan=1 colspan=1>#param perGaussian</td></tr><tr><td rowspan=1 colspan=1>Position, $( x , y , z )$ </td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>SH DC, $( k _ { I } , k _ { 2 } , k _ { 3 } )$ </td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1> $\operatorname { S H A C } , ( k _ { 4 } , . . . , k _ { 4 8 } )$ </td><td rowspan=1 colspan=1>45</td></tr><tr><td rowspan=1 colspan=1>Scale, $\underline { { ( s _ { x } , s _ { y } , s _ { z } ) } }$ </td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Rotation, $\underline { { ( r _ { u } , r _ { x } , r _ { y } , r _ { z } ) } }$ </td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Opacity, (Ï)</td><td rowspan=1 colspan=1>1</td></tr></table>

(b) Scaffold-GS
<table><tr><td rowspan=1 colspan=1>GS parameter</td><td rowspan=1 colspan=1>#paramper voxel</td></tr><tr><td rowspan=1 colspan=1>Position (anchor) $( x , y , z )$ </td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Offset features, $( O _ { I } , . . . , O _ { 3 0 } )$ </td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>Anchor features, $\underline { { ( A _ { I } , . . . , A _ { 3 2 } ) } }$ </td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>Scale factor, ()</td><td rowspan=1 colspan=1>1</td></tr></table>

Scaffold-GS-based models store, for each voxel, anchor positions, scale factor, offset and anchor features, and three small MLPs that predict Gaussian attributes. Although MLP parameters must also be transmitted, their size is relatively small (7086 weights and biases, in total), when compared to the remaining model parameters. For this reason, GSICO does not compressed them; instead, they are transmitted directly in the bitstream (though a standard like MPEG NNR [21] could be used instead).

The key idea behind GSICO is to convert the rich 3D representation used in GS into a series of 2D parameter maps (or a structured 3D volume), where each map (or slice) stores one parameter of the GS model. Under this approach, the number of maps equals the number of parameters, P (with P = 59 for 3DGS and P = 66 for Scaffold-GS), and the total number of pixels of each map matches the number of Gaussians or voxels, in the underlying 3D representation. As shown in Fig. 3, in red color, a given pixel location, $( x , y )$ , is consistently associated with the same Gaussian (or voxel) across all parameter maps. In this sense, (?, ?) acts as a lookup index that retrieves the full set of GS parameters stored across the maps.

Simply arranging GS parameters into 2D maps is not sufficient to achieve efficient compression. Each map stores parameters from different Gaussians (or voxels) with limited correlation between them. To enable efficient compression with a conventional image/video codec, spatial redundancy within each map must be exploited. For this purpose, a novel strategy is introduced to generate the structured 3D volume: Gaussians (or voxels) are first grouped into fixed-size clusters based on similarity and then assigned to 3D blocks; each block is filled with the Gaussians (or voxels) parameters of one cluster, as illustrated with orange color in Fig. 3. A dedicated similarity criterion, specifically designed for this task, guides this process. This criterion relies on SH coefficients (for 3DGS-based inputs), as these dominate the data volume, or on anchor positions and offset features (for Scaffold-GS-based inputs), as this combination empirically showed to correlate well with compression performance. This approach enhances local similarity within each parameter map, enabling significantly more efficient compression than naÃ¯ve approaches such as random ordering. The GSICO encoder modules are described next:

Color space conversion: The SH coefficients represent the view-dependent color of each Gaussian and are originally defined in the RGB color space. To reduce redundancy across color components, the linear RGB-to-YUV transform specified in ITU-R BT.601 [22] (used without any clipping) is directly applied to the SH coefficients. This concentrates most of the signal energy in the luminance component, enabling the pruning of chrominance AC SH coefficients with negligible perceptual impact. As both the RGB-to-YUV transform and the subsequent GS mapping of SH coefficients are linear operations, applying the color transform before (in the pixel domain) or after (in the SH coefficients domain) the GS mapping is mathematically equivalent. However, since GS mapping uses the SH coefficients within its similarity criterion, applying the color transform beforehand ensures that it operates on already decorrelated color components, improving its effectiveness. This color space transform operation is applied only to 3DGS-based inputs, since Scaffold-GS does not employ SH-based color parametrization.

<!-- image-->

(a) 3DGS-based input files  
<!-- image-->  
(b) Scaffold-GS-based input files  
Fig. 3: Illustration of the GS parameter mapping for the 3DGS-based and Scaffold-GS-based input files.

Pruning: This step ensures that Gaussian (or voxel) parameters can be organized into rectangular-shaped 2D maps, while slightly reducing the data volume and preserving rendering quality. Since the resulting maps are composed of 16Ã16 blocks (the block size used in the next steps), both width and height must be multiples of 16 pixels. The chosen block size aligns with those commonly employed in standard image codecs, facilitating better compatibility with the subsequent image coding step. Other common block sizes (such as 4Ã4, 8Ã8, 32Ã32, 64Ã64, and 128Ã128 pixels) were evaluated but did not provide a better trade-off between computational efficiency and compression performance. In particular, larger block sizes yield only marginal compression gains while significantly increasing runtime. The width and height map sizes, $W _ { m a p }$ and $H _ { m a p } .$ , respectively, are determined based on the total number of blocks that can be formed by all Gaussians (or voxels). Thus, if the total number of Gaussians (or voxels) is not divisible by $^ { 1 6 , }$ the excess elements are pruned to satisfy this requirement. Furthermore, if the resulting number of blocks that can be formed from all Gaussians (or voxels) is a prime number, 256 Gaussians (equivalent to one 16Ã16 block) are pruned to make the block count factorable. The final width and height are then chosen as the two closest integer factors (both multiples of 16) whose product matches the remaining number of elements. Whenever pruning is required, opacity is used as the selection criterion, as it has proven to be a reliable indicator of the visual importance of each Gaussian [15]. Gaussians with lower opacities (or voxels with lower average Gaussian opacities) contribute less to the rendered quality, making them suitable candidates for pruning.

Clustering: In this step, Gaussians (or voxels) are grouped into fixed-size clusters to enable the block-based organization of their parameters in the 2D maps. Each cluster contains 256 elements, allowing their parameters to be later arranged as a 16Ã16 block on the corresponding map (cf. Fig. 3). Clustering is performed in a feature space (using a similarity criterion) defined by GS parameters that were found most relevant to compression efficiency. The specific feature space depends on the baseline model: for 3DGS-based inputs, clustering relies on the luminance SH AC coefficients, while for Scaffold-GS-based inputs it consists of anchor positions concatenated with offset features. This step ensures that Gaussians (or voxels) with similar parameters are placed together in the same block, increasing spatial coherence within each parameter map. Additional implementation details are provided in Section III.B.

Cluster-to-block assignment: This step receives as input the set of clusters computed in the previous step and assigns each one to a 3D block on the set of 2D maps (or 3D volume). The goal is to place clusters with similar characteristics near one another, improving global spatial coherence; the actual placement of parameter values within each block is performed in the next step. A novel NNS algorithm defines the association between clusters and blocks, using the same similarity criterion employed during clustering. For this purpose, each cluster is represented by a feature vector obtained by averaging the relevant parameters (namely, the luminance SH AC coefficients for 3DGS-based inputs or the anchor positions concatenated with offset features for Scaffold-GS inputs) across all elements in that cluster. Additional implementation details are provided in Section III.C.

Block filling: This step takes as input the clusters and their assigned block locations, and fills each 3D block with the corresponding GS parameter values. Note that each slice of the 3D block corresponds to one parameter. This filling is done block-by-block in snake scan order, starting with the top-left block and proceeding horizontally. The same NNS algorithm used for cluster-to-block assignment is now applied for block filling to determine the placement of each Gaussian (or voxel) within the 3D block, using the same similarity criterion of the clustering step. This enforces that neighboring positions on a given 2D map correspond to elements with similar parameter values, thereby enhancing local spatial coherence. Further details are provided in Section III.C.

Quantization: This step receives as input the set of 2D maps produced previously, which are in floating-point, and applies uniform mid-tread quantization. The goal is to reduce parameter precision, thereby lowering the bitrate after image compression while preserving the quality of the reconstructed scene. Furthermore, for 3DGS models, the SH AC chrominance components are discarded to further reduce the bitrate, with negligible perceptual impact. For quantization, each ?-th parameter (and thus each 2D map) is assigned a fixed bit depth, $b _ { i } ,$ determined empirically based on extensive experiments. Since the value ranges differ across parameters, the quantization step size is computed independently for each map based on the minimum, $x _ { \mathrm { m i n } } .$ , and maximum, $x _ { \mathrm { m a x } } ,$ observed values. The quantization step size is then given by:

$$
Q _ { \mathrm { s t e p } } ^ { i } = { \frac { x _ { \mathrm { m a x } } ^ { i } - x _ { \mathrm { m i n } } ^ { i } } { 2 ^ { b _ { i } } } }\tag{1}
$$

Both the minimum and maximum values of each parameter are included in the bitstream, enabling the decoder to compute the step size, $Q _ { s \mathrm { t e p } } ^ { i } .$ . The bit depth assigned to each parameter is assumed to be known to both the encoder and decoder. Finally, each 2D map is independently quantized according to:

$$
Q _ { \mathrm { i n d e x } } ^ { i } = \mathrm { r o u n d } \left( \frac { x _ { i } - x _ { \mathrm { m i n } } ^ { i } } { Q _ { \mathrm { s t e p } } ^ { i } } \right)\tag{2}
$$

JPEG XL encoding: This step encodes the set of quantized 2D maps from the previous step using the JPEG XL image codec [23]. JPEG XL was selected for its high compression efficiency, especially for structured data such as the GS parameters maps. JPEG XL provides a quality parameter that regulates the balance between compression efficiency and fidelity, ranging from negative infinity to 100, where 100 denotes lossless coding. For 3DGS-based models, the SH parameter maps are encoded in lossy mode, while all other parameter maps are encoded in lossless mode. This leverages the lower sensitivity of SH coefficients to small distortions, while preserving geometric parameters. In its lossy configuration, JPEG XL employs 8Ã8 DCT-based predictive coding. As discussed earlier, this characteristic influenced the selection of the block size used for GS parameter maps creation. For Scaffold-GS-based models, all maps are encoded using JPEG XL in lossless mode. This choice reflects the voxelbased nature of Scaffold-GS, where even small losses introduced by image compression can propagate and result in noticeable rendering artifacts. Since each parameter corresponds to a dedicated 2D map, encoding is performed on a map-by-map basis, with each map treated as a singlechannel luminance image. This step constitutes the final stage of the GSICO encoder. The resulting bitstream includes the JPEG XL-encoded 2D maps along with the associated quantization metadata (minimum and maximum parameter values), and, in Scaffold-GS-based models, also includes the MLP parameters.

The GSICO decoder performs the inverse operations of the encoder to reconstruct a GS file from the codec bitstream. It first decodes the compressed GS parameter maps, after which inverse quantization is applied to get their floating-point values. Specifically, each integer index is mapped back to its reconstruction level using $x ^ { \prime } { } _ { i } = Q _ { \mathrm { i n d e x } } ^ { i } Q _ { \mathrm { s t e p } } ^ { i } + x _ { \mathrm { m i n } } ^ { i }$ , where the step size is derived from the minimum and maximum parameter values transmitted in the bitstream. Finally, the Gaussians (or voxels) are reconstructed by reading the same pixel position across all reconstructed 2D maps, since each position across the parameter images corresponds to a single Gaussian (or voxel). This synchronized organization enables fast decoding. For Scaffold-GS-based models, this is the last process of GSICO, enabling the restored file to be used directly for rendering novel views. For 3DGS-based models, the inverse color space conversion (from YUV-to-RGB) is further applied to the SH coefficients. At the end of the codec pipeline, the decoded GS file can be used for view synthesis from camera positions requested by a user.

The GSICO operating points were selected empirically by assessing how quantization and JPEG XL coding impact the rendering quality, leading to the configuration presented in Table II. For 3DGS-based inputs, different quantization settings are applied to the SH coefficients; therefore, Table II distinguishes between luminance (Y) and chrominance (UV) coefficients, as well as between first, second, and third-degree SH coefficients (1Âº, 2Âº, and 3Âº). Five operating points were defined for each representation, namely 3DGS-based and Scaffold-GS-based, ensuring that the resulting RD curves reflect a meaningful trade-off between rate and quality. For 3DGS, the operating points are most effectively tuned by varying the JPEG XL quality level in lossy mode, as this provides the dominant RD behavior. In contrast, for Scaffold-GS (where the voxel structure is more sensitive to compression), JPEG XL is used exclusively in lossless mode, and the operating points are instead determined by varying the quantization bit depth configuration.

While the overview above outlines the full GSICO pipeline, its core contribution lies in the GS mapping encoder, which organizes GS parameters into a set of 2D maps in a compression-efficient manner. Section III.B and Section III.C detail the key modules involved in this process.

## B. Clustering

A random mapping of Gaussians (or voxels) parameters onto 2D maps (or 3D volume) is inefficient, since it leads to very limited small spatial correlation and therefore poor coding efficiency. On the other hand, exhaustively testing all possible combinations of Gaussians (or voxels) parameters would be computationally expensive and impractical due to the typically large number of elements involved (in the order of millions). Clustering provides an efficient intermediate step by first grouping similar elements, that can later be organized into 3D blocks at much lower computational cost.

The key idea is to perform fixed-size clustering of Gaussians (or voxels), grouping them according to the similarity of their parameter values. A fixed cluster size allows the resulting data to be arranged into 2D maps using a regular grid of 16Ã16-pixel blocks, which aligns well with the design characteristics of standard image/video codecs.

TABLE II  
GSICO OPERATING POINTS DEFINED BY THE QUANTIZATIONBIT DEPTH AND JPEG XL QUALITY LEVEL  
(a) 3DGS-based input files
<table><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>RDpoint</td><td rowspan=1 colspan=8>GS parameters</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>P3</td><td rowspan=1 colspan=1>P4</td><td rowspan=1 colspan=1>P5</td><td rowspan=1 colspan=1>P6</td><td rowspan=1 colspan=1>P7</td><td rowspan=1 colspan=1>P8</td></tr><tr><td rowspan=1 colspan=1>Quantization(bit depth)</td><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=5 colspan=1>JPEG XL(quality level)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>100</td></tr></table>

(P1 - position, P2 - scale, P3 - rotation, P4 - SH DC, P5 - SH AC Y 1Âº, P6 - SH AC Y 2Âº/3Âº, P7 - SH AC UV, P8 â opacity)

(b) Scaffold-GS-based input files
<table><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>RDpoint</td><td rowspan=1 colspan=4>GS parameters</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>P3</td><td rowspan=1 colspan=1>P4</td></tr><tr><td rowspan=5 colspan=1>Quantization(bit depth)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>JPEG XL(quality level)</td><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td></tr></table>

(P1 - position, P2 - scale factor, P3 - offset features, P4 - anchor features)

Given its optimality in minimizing intra-cluster variance, the K-Means algorithm [24] was adopted as the basis of the clustering process. Since K-Means does not inherently support fixed-size clusters, a K-Means-based fixed-size clustering algorithm was developed. This algorithm receives as input a set of ? Gaussians (or voxels) with the respective parameter values, $\mathbf { G } = \{ \mathbf { g } _ { 1 } , \ldots , \mathbf { g } _ { N } \}$ and outputs a set of ? fixed-size clusters of 256 elements (to enable the creation of 16Ã16-pixel blocks), $\mathbf { C } = \{ \mathbf { C } _ { 1 } , \dots , \mathbf { C } _ { L } \}$ , with $L = N / 2 5 6$ . The pruning step already ensured that the total number of Gaussians (or voxels) to be clustered are multiple of 256. The clustering algorithm consists of five steps:

1. Initialization: Initialize ? to an empty set;

2. Clustering: Apply K-Means to ?, with K-Means++ initialization, to obtain $K = { \mathrm { c a r d } } ( \mathbf G ) / 2 5 6$ intermediate clusters (initially, $K = L ) , \mathbf { C } _ { \mathrm { I } } = \{ \mathbf { C } _ { 1 } ^ { \mathrm { I } } , \dots , \mathbf { C } _ { K } ^ { \mathrm { I } } \}$ . In K-Means, similarity is measured using the Euclidean distance:

$$
d \big ( \mathbf { f } ( \mathbf { g } _ { i } ) , \mathbf { c } _ { j } \big ) = \big \| \mathbf { f } ( \mathbf { g } _ { i } ) - \mathbf { c } _ { j } \big \| _ { 2 } ,\tag{3}
$$

where $\mathbf { f } ( \mathbf { g } _ { i } )$ denotes the feature vector of the ?-th element from ? and $\mathbf { c } _ { j }$ is the centroid of the ?-th cluster. For

3DGS-based inputs, ?(â) consists of the luminance SH AC coefficients; for Scaffold-GS-based inputs, it is formed by the concatenation of anchor positions and offset features. Each centroid $\mathbf { c } _ { j }$ is computed as the element-wise average of the feature vectors in the respective intermediate cluster, $\mathbf { f } _ { \mathrm { a v g } } ( \mathbf { C } _ { j } ^ { \mathrm { I } } )$ .

3. Extract fixed-size clusters: Each cluster in $\mathbf { C _ { I } }$ with more than 256 elements is randomly partitioned into disjoint sub-clusters of size 256, with each resulting sub-cluster added to ?. Other splitting strategies were evaluated without leading to significant differences in performance, as K-Means already groups elements that are similar between them.

4. Update the remaining set: Remove from ? the elements assigned to clusters of ? in the previous step. Any leftover elements (resulting from the clusters $\mathbf { C } _ { j } ^ { \mathbf { I } }$ whose cardinal is not a multiple of 256) remain in ?.

5. Stopping criteria: $\operatorname { I f } \mathbf { G } \neq \varnothing$ , repeat from the second step using K-Means on the remaining elements of ?.

## C. Cluster-to-Block Assignment and Filling

The fixed-size clusters previously obtained are assigned to blocks within a set of 2D maps (or 3D volume). However, a spatial organization is required (both across blocks and within each block) to ensure that similar clusters and their elements are placed close to each other. This spatial coherence is essential to improve coding performance. To this end, a novel Nearest-Neighbor-based Sorting (NNS) algorithm was developed to spatially organize Gaussians (or voxels), or their clusters, within the set of 2D maps. NNS ensures that elements with similar characteristics are placed at neighboring spatial locations, thereby inducing strong local correlations. This spatial organization leads to smooth variations across the parameter maps, which are well aligned with the prediction and transform mechanisms exploited by standard image codecs. As a consequence, the resulting 2D maps exhibit reduced entropy and can be compressed more efficiently. In the cluster-to-bock assignment procedure, the NNS algorithm is used to determine the placement of clusters within the set of blocks; in the block filling procedure, it defines the placement of the cluster Gaussians (or voxels) inside the block.

The NNS algorithm takes as input a set of ? elements, $\mathbf { E } = \{ \mathbf { e } _ { 1 } , \dots , \mathbf { e } _ { s } \}$ and two integers, ? and ?, such that $S = W \times H ;$ for cluster-to-block assignment, $\mathbf { e } _ { i }$ represents the centroid of a cluster obtained in the clustering step, $W = W _ { m a p } / 1 6$ , and $H = ~ H _ { m a p } / 1 6$ , where $W _ { m a p }$ and $H _ { m a p }$ are, respectively, the width and height of the 2D maps (computed during the pruning step described in Section III.A); thus, in this case, ? corresponds to the total number of clusters (i.e., 16Ã16 blocks to be formed); for block filling, $\mathbf { e } _ { i }$ represents a Gaussian (or voxel), ? = 16 and ? = 16. The output is a 2D matrix, ?, with size ? Ã ?, that contains in each position the index of an element of ?.	Accordingly,	the NNS algorithm can be defined as $\mathbf { M } = \mathrm { N N S } ( \mathbf { E } , W , H )$ being composed of four steps:

1. Initialization and first assignment: Compute the element-wise median feature vector of the entire set ?, denoted as $\mathbf { f } _ { \mathrm { m e d } } ( \mathbf { E } )$ ; the feature vector of each element, $\mathbf f ( \cdot )$ , is the same considered for the clustering step. Select the element $\mathbf { e } _ { k }$ from ? whose feature vector $\mathbf { f } ( \mathbf { e } _ { k } )$ is closest to $\mathbf { f } _ { \mathrm { m e d } } ( \mathbf { E } )$ , in terms of Euclidean distance:

<!-- image-->  
Fig. 4: Illustration of the NNS algorithm concept.

$$
\mathbf { e } _ { k } = \arg \operatorname* { m i n } _ { \mathbf { e } _ { l } \in \mathbf { E } } \| \mathbf { f } ( \mathbf { e } _ { l } ) - \mathbf { f } _ { \mathrm { m e d } } ( \mathbf { E } ) \| _ { 2 }\tag{4}
$$

Assign the index of element $\mathbf { e } _ { k }$ to the first matrix position, $\mathbf { M } [ 1 , 1 ] = k$ , and remove $\mathbf { e } _ { k }$ from ?.

2. Transversal filling: Fill the remaining positions of ? following a snake scan order (left to right on one row and then right to left on the next) and according to the following assignment rule: for each new matrix position, $( i , j )$ , determine its set of already assigned neighboring elements, $\mathcal { N } _ { i j }$ . Neighbors include any assigned elements to the positions that are directly to the left, right, top, topleft, or top-right of $( i , j )$ . The positions below (bottomleft, bottom, and bottom-right) are not considered, as they remain unfilled during the snake scanning. Compute the element-wise average of the feature vectors of these neighbors, $\mathbf { f } _ { \mathrm { a v g } } ( \mathcal { N } _ { i j } )$ . Select from set ? the element $\mathbf { e } _ { k }$ whose feature vector is closest to this average:

$$
\mathbf { e } _ { k } = \arg \operatorname* { m i n } _ { \mathbf { e } _ { l } \in \mathrm { E } } \left\| \mathbf { f } ( \mathbf { e } _ { l } ) - \mathbf { f } _ { \mathrm { a v g } } \big ( \mathcal { N } _ { i j } \big ) \right\| _ { 2 }\tag{5}
$$

Assign the index of element $\mathbf { e } _ { k }$ to $\mathbf { M } [ i , j ]$ and remove $\mathbf { e } _ { k }$ from ?.

3. Stopping criterion: Repeat the second step until all positions of ? are assigned and the input set is empty (i.e., $\mathbf { E } = \varnothing )$ ).

Each element of the NNS output matrix, ?, represents either the index of a cluster (for the cluster-to-bock assignment procedure) or the index of a Gaussian (or voxel) within a cluster (for the block filling procedure). Fig. 4 illustrates the key concept of NNS algorithm, where the red cell indicates the current matrix position, $( i , j )$ , being analyzed at each iteration of the NNS algorithm, blue cells correspond to the neighboring positions considered during that iteration, $\mathscr { N } _ { i j }$ , and grey cells denote positions that have already been assigned to a cluster or Gaussian (or voxel). The NNS algorithm scans the matrix in a snake order, progressively filling elements in each position considering the feature similarity with its already assigned neighbors.

The first time NNS is used on the GSICO encoder is for deciding the cluster-to-block assignment, i.e., to define the position, at the block level, of the formed fixed-size clusters. This procedure receives as input the set of ? fixed-size clusters, of 256 Gaussians (or voxels) each, that resulted from the clustering procedure, $\mathbf { C } = \{ \mathbf { C } _ { 1 } , \dots , \mathbf { C } _ { L } \}$ , and the width, $W _ { m a p }$ and height, $H _ { m a p } ,$ of the 2D parameter maps. The output is a 2D matrix, $\mathbf { M } _ { b l o c k s }$ , with dimension $W _ { m a p } / 1 6 \times H _ { m a p } / 1 6$ The cluster-to-block assignment follows two major steps:

1. Initialization: Compute the set of cluster centroids, $\mathbf { C } _ { c } =$ $\{ \mathbf { c } _ { 1 } , \mathbf { \Gamma } \_ { 1 } , \mathbf { c } _ { L } \}$ , where the centroid of each cluster is given by the element-wise average of its elements feature vectors, $\mathbf { c } _ { i } = \mathbf { f } _ { \mathbf { a v g } } ( \mathbf { C } _ { i } )$

2. NNS algorithm: Run the NNS algorithm on $\mathbf { C } _ { c } ,$ $\mathrm { N N S } ( \mathbf { C } _ { c } , \ W _ { m a p } / 1 6 , H _ { m a p } / 1 6 )$ , to obtain a 2D matrix, $\mathbf { M } _ { b l o c k s }$ , where every position of $\mathbf { M } _ { b l o c k s }$ contains the index of a cluster.

After the cluster-to-block assignment, NNS is used again for block filling, i.e., to define the position, at the pixel level, of the Gaussians (or voxels) of each cluster in the respective block. This procedure receives as input the set of ? fixed-size clusters of 256 Gaussians (or voxels) elements, $\mathbf { C } = \{ \mathbf { C } _ { 1 } , \dots , \mathbf { C } _ { L } \}$ $\mathbf { M } _ { b l o c k s }$ , and the 2D maps width, $W _ { m a p } .$ , and height, $H _ { m a p }$ . The output is a 2D matrix, ${ { \mathbf { M } } _ { a l l } }$ , with size $W _ { m a p } \times H _ { m a p } .$ . The block filling can be described in the steps:

1. First block filling: Let $\mathbf { C } _ { i }$ denote the cluster previously assigned to the first block (specified in $\mathbf { M } _ { b l o c k s } [ 1 , 1 ] ,$ ). The NNS algorithm is applied to the Gaussians (or voxels) in $\mathbf { C } _ { i }$ with a target block size of 16Ã16 pixels, $\mathsf { N N S } ( \mathbf { C } _ { i } , 1 6 , 1 6 )$ . The resulting matrix is used to fill the corresponding block region in ${ { \mathbf { M } } _ { a l l } }$ , the global matrix that stores the spatial assignments of all Gaussians (or voxels) across the full set of 2D parameter maps.

2. Local block filling: Following a block-by-block snake scan order, run $\mathrm { N N S } ( \mathbf { C } _ { i } , 1 6 , 1 6 )$ with the Gaussians (or voxels) corresponding to the cluster $\mathbf { C } _ { i }$ assigned to the current block of the 2D maps according to $\mathbf { M } _ { b l o c k s }$ . The resulting matrix is used to fill the corresponding block region in ${ { \mathbf { M } } _ { a l l } }$

3. Stopping criterion: Repeat the second step until all positions of ${ \bf M } _ { a l l }$ are filled.

After cluster-to-block assignment and filling is completed, ${ { \mathbf { M } } _ { a l l } }$ is used for a final step where all Gaussian (or voxel) parameter values are mapped into the set of 2D maps according to their defined index positions. The resulting parameter maps are stored as image files, whose bit depth depends on the applied quantization (cf. Table II). All maps are stored in 8-bit PNG format, except the position maps, which require higher precision and are therefore stored as a 16-bit PNG. Fig. 5 shows a randomly organized parameter map of the truck scene trained (from Tanks and Temples dataset [25]) with the 3DGS model, alongside the same map produced using the proposed NNSbased GS mapping. In this example, only a subset of 2304 Gaussians was considered (for visualization purposes) and only one of the luminance SH AC parameters (one of the GS parameters used in the NNS similarity criterion for 3DGSbased inputs) is represented. Notably, the random map results in a PNG file with 1558 bytes (using only the lossless PNG compression), whereas the NNS-based map leads to 995 bytes (using the the NNS algorithm followed by lossless PNG compression). This highlights the importance of NNS in the GS parameter mapping step of the GSICO encoder.

<!-- image-->  
(a) Random map

<!-- image-->  
(b) NNS-based map  
Fig. 5: Impact of the NNS-based GS mapping on the spatial organization of GS parameters.

## IV. PERFORMANCE ASSESSMENT

To evaluate the GSICO performance, a comprehensive RD analysis was conducted. This section first describes the experimental setup, including the test material, rate and distortion metrics, and selected benchmarks. It then presents and discusses the experimental results, followed by an ablation study to measure the impact of each GSICO module on its overall performance.

## A. Test conditions

The test conditions were defined based on previous work, with the objective of ensuring reproducibility and comparability with existing solutions.

## 1) Test material

The Tanks and Temples (T&T) [25], Deep Blending (DB) [26], and Mip-NeRF360 [27] were selected to ensure a comprehensive evaluation. The T&T dataset includes two complex real-world outdoor scenes, train and truck, exhibiting varied geometry and texture richness. The DB dataset contains two complex real-world indoor scenes, drjohnson and playroom, with varying lighting conditions and fine geometric details. The Mip-NeRF360 dataset comprises four real-world indoor scenes, bonsai, counter, kitchen, and room, and five realworld outdoor scenes, bicycle, flowers, garden, stump, and treehill, with significant variation in depth and texture. Each dataset is organized into training and test image sets. Training images are used by the GS model to reconstruct the scene (i.e., for training), while test images are reserved for rendering novel views and evaluating the performance of the model reconstruction and its compression. The two sets are disjoint, ensuring that the reported RD results are fairly obtained from viewpoints not seen during training.

TABLE III  
GSICO RD RESULTS COMPARING WITH THE BASELINE MODELS
<table><tr><td rowspan="2">Method</td><td colspan="4">Tanks and Temples</td><td colspan="4">Deep Blending</td><td colspan="4">Mip-NeRF360</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Size (MB)â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Size (MB)â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Size (MB)</td></tr><tr><td>3DGS</td><td>23.88</td><td>0.848</td><td>0.176</td><td>306.9</td><td>29.27</td><td>0.899</td><td>0.260</td><td>223.2</td><td>27.52</td><td>0.815</td><td>0.214</td><td>556.2</td></tr><tr><td>Scaffold-GS</td><td>24.13</td><td>0.854</td><td>0.176</td><td>80.0</td><td>30.33</td><td>0.909</td><td>0.253</td><td>57.4</td><td>27.73</td><td>0.815</td><td>0.221</td><td>187.6</td></tr><tr><td>GSICO (w/ 3DGS)</td><td>23.50</td><td>0.837</td><td>0.189</td><td>15.0</td><td>29.15</td><td>0.890</td><td>0.265</td><td>11.3</td><td>26.70</td><td>0.798</td><td>0.232</td><td>27.6</td></tr><tr><td>GSICO (w/ Scaffold-GS)</td><td>24.03</td><td>0.849</td><td>0.179</td><td>11.2</td><td>30.18</td><td>0.907</td><td>0.256</td><td>6.3</td><td>27.27</td><td>0.800</td><td>0.232</td><td>20.8</td></tr></table>

## 2) Rate and quality metrics

The full bitstream size, expressed in megabytes (MB), was adopted as the rate measure. For quality evaluation, five widely recognized full-reference image quality metrics were considered. DISTS [28] and FSIM [29] metrics, both shown to exhibit high correlation with human perception [30], were selected for the RD characterization of GSICO and the ablation study. DISTS combines deep feature representations with texture and structure similarity, while FSIM computes a normalized similarity measure based on salient image features. To facilitate RD curve visualization, both DISTS and FSIM results were transformed into decibel (dB) scales using $- 2 0 \log _ { 1 0 } ( m )$ (for DISTS) and $- 2 0 \log _ { 1 0 } ( 1 - m )$ (for FSIM), where ? denotes the original metric value. Additionally, PSNR, SSIM [31], and LPIPS (VGG) [32] were included to enable direct comparison with existing GS compression methods, as these three metrics are consistently reported in prior works. PSNR evaluates per-pixel fidelity, SSIM measures structural consistency and luminance-contrast similarity, and LPIPS assesses perceptual similarity based on deep NN features. All metrics were computed in the RGB color space, except FSIM which operates in the LAB color space. SSIM and FSIM range from 0 (lowest quality) to 1 (highest quality) while LPIPS and DISTS ranges from 0 (highest quality) to 1 (lowest quality). For all metrics, the reference corresponds to the test images provided in the datasets, in line with common practice in radiance field evaluation [33]. Metric values were first computed per test image (of a scene) and then averaged to obtain a single score per scene. The score of a dataset corresponds to the average of the scores of its scenes.

## 3) Selected benchmarks

GSICO performance was compared against the baseline models (3DGS [2] and Scaffold-GS [4]) and several state-ofthe-art GS compression methods, selected for their diversity in terms of technical approach and competitive performance. These include: Compact3D [19], employing vector quantization, compressing the codebooks via index sorting and run-length encoding; Reduced3DGS [12], implementing resolution-aware Gaussian pruning, adapting SH coefficient maximum degree per Gaussian, and applying adaptive visual importance-aware quantization; LightGaussian [13], pruning Gaussians based on scene spatial coverage, distilling SH coefficients to lower degrees, and applying vector quantization; CompGS [14], using a learnable mask to remove Gaussian with low visual impact, employing vector quantization, and building an entropy model based on GS parameter hyperpriors and interparameter priors; EAGLES [34], integrating entropy-aware coding with spatial Gaussian attributes prediction; SOG [15], pruning low-opacity Gaussians and implementing an imagebased Gaussian parameters compression; CodecGS [16], using vector quantization built upon tri-plane image representation; RDO Gaussian [35], applying RD optimization to allocate bit depths dynamically across parameters; and FCGS [36], focusing on streamlined data structures and transforms optimized for fast coding. All benchmarks were evaluated using the parameter settings recommended by the authors in their works to ensure a fair comparison. Although recent GS methods such as HAC [17], HAC++ [18], HEMGS [37], and ContextGS [38] have reported competitive GS compression performances, they were excluded from the comparison: HEMGS due to the lack of a publicly available implementation, while the remaining methods did not provide a functional decoder at the time of this work.

<!-- image-->

<!-- image-->  
Fig. 6: RD performance of GSICO codec and the 3DGS and Scaffold-GS baseline models.

<!-- image-->  
Fig. 7: Qualitative comparison between ground-truth (GT) images, baseline model renderings, and GSICO-compressed renderings for selected scenes across all datasets, shown with the corresponding average PSNR values and bitstream sizes.

## B. Experimental Results

This section presents the RD performance of GSICO, compares its compression gains with baseline models and GS compression state-of-the-art benchmarks, and reports an ablation study on the GSICO components.

## 1) RD characterization of the proposed GSICO codec

To fully characterize the RD performance of the proposed GSICO codec, five operating points were defined for each input representation model (cf. Section III.A). Fig. 6 shows the GSICO performance results obtained for the T&T dataset. The RD curves exhibit excellent rate scalability and stability across operating points. For both 3DGS-based and Scaffold-GS-based configurations, GISCO covers a wide operating range, from highly compressed to near-lossless regimes (i.e., same quality as the baseline without any compression), while maintaining smooth and monotonic quality progression. Notably, no abrupt quality drops are observed between adjacent operating points, demonstrating the robustness and consistency of the quantization strategy across bitrate levels.

## 2) Improvements over the GS baseline models

A comparison between GSICO and the corresponding uncompressed baseline models is shown in Table III, considering only the highest-quality GSICO operating point, which closely matches the quality of baseline models. These results demonstrate that GSICO achieves substantial reductions in model size while preserving high rendering quality.

For 3DGS-based inputs, GSICO reduces the model size by an average factor of 20.2Ã (from 306.6 MB to 15.0 MB on T&T, from 223.2 MB to 11.3 MB on DB, and from 556.2 MB to 27.6 MB on Mip-NeRF360). The quality degradation is minimal: average PSNR drops are limited to 0.45 dB (from 23.88 dB to 23.50 dB on T&T, from 29.27 dB to 29.15 dB on DB, and from 27.52 dB to 26.70 dB on Mip-NeRF360), SSIM variations remain below 0.01, and LPIPS degradation does not exceed 2%. These results indicate that GSICO efficiently compresses GS parameters without introducing significant perceptual degradations.

<!-- image-->  
Fig. 8: GSICO RD curves compared with selected benchmarks.

For Scaffold-GS inputs, GSICO achieves an average compression factor of 8.5Ã (from 80.0 MB to 11.2 MB on T&T, from 57.4 MB to 6.3 MB on DB, and from 187.6 MB to 20.8 MB on Mip-NeRF360). Average PSNR reduction is 0.38 dB (from 24.13 dB to 24.03 dB on T&T, from 30.33 dB to 30.18 dB on DB, and from 27.73 dB to 27.27 dB on Mip-NeRF360), SSIM variations remain below 0.01, and LPIPS values do not degrade more than 1.5%, further supporting the efficiency of the codec in preserving rendering quality. Although lower than the compression gains observed for 3DGS, these reductions are particularly significant given the already more compact nature of Scaffold-GS representation.

A qualitative evaluation of the proposed codec was also conducted on representative scenes from all the selected datasets. Fig. 7 presents visual comparisons (after synthesis) between ground-truth (GT) images, uncompressed GS baseline models, and GSICO. Even under substantial compression, GSICO preserves the quality of the rendered views, with no noticeable degradations such as blurring, color shifts or structural distortions. Both 3DGS-based and Scaffold-GSbased configurations maintain visual consistency across complex indoor and outdoor scenes, reinforcing GSICOâs ability to significantly reduce model size while preserving perceptual quality.

## 3) Benchmark evaluation

For comparison with state-of-the-art GS compression methods, only the two highest-quality GSICO operating points were considered, ensuring a fair and balanced comparison with benchmarks that typically operate over a narrower RD range. Fig. 8 illustrates the RD performance across all datasets.

On the T&T dataset, the Scaffold-GS-based GSICO configuration emerged as the top-performing solution, with the best trade-off between quality and compression efficiency. At a rate of 10 MB, GSICO outperforms CompGS by approximately 0.3 dB in PSNR. The GSICO 3DGS-based configuration, although less competitive than its Scaffold-GS counterpart, still outperforms the image-based compression method SOG, requiring approximately 5.9 MB less bitrate at a PSNR of 23.5 dB.

On the DB dataset, the Scaffold-GS-based GSICO configuration again delivers the best performance, surpassing all benchmarks. Relative to the strongest competing solutions, Compact3D and CodecGS, at a SSIM of 0.906, GSICO requires approximately 5.85 MB and 2.85 MB less rate, respectively. Although the GSICO 3DGS-based configuration is less competitive in this case, it still surpasses SOG in terms of LPIPS, the quality metric with highest correlation with human perception from the three selected quality metrics [30].

On the Mip-NeRF360 dataset, which contains large-scale and geometrically complex indoor and outdoor scenes, GSICO maintains strong performance. The Scaffold-GS-based configuration achieves competitive quality, particularly in terms of LPIPS: at 19.1 MB (corresponding to its low-rate configuration), it achieves a quality level similar to the highrate configurations of both Compact3D and RDO-Gaussian. GSICO 3DGS-based configuration, while achieving more modest compression gains, it still provides a superior RD performance compared to SOG for both SSIM and LPIPS curves. The challenging nature of the Mip-NeRF360 dataset, where GS methods tend to produce a higher number of Gaussians than in other datasets, makes compression particularly difficult. In this case, joint GS training and compression methods tend to perform better.

Across all datasets, the RD performances depicted in Fig. 8 confirm that GSICO consistently has the best RD tradeoff, with its Scaffold-GS-based configuration emerging as the best solution. Overall, GSICO distinguishes itself from prior work through its training-free and model-agnostic design, ensuring immediate applicability across diverse GS models and thus supporting practical deployment in real-world multimedia systems.

## 4) Ablation study

The ablation study selectively disabled individual GSICO components to provide insight into their respective contribution to the overall system behavior. RD performance is evaluated for each supported baseline model (3DGS and Scaffold-GS) using the T&T scenes. The study begins to assess the impact of the first GSICO component, namely pruning. Since pruning is required in GSICO to enable a rectangular 2D map arrangement of GS parameters, its impact is evaluated relative to the uncompressed model. Subsequently, the full GSICO configuration is used as the reference to analyze the contributions of the NNS-based mapping and the JPEG XL coding components. Specifically, GSICO is compared against: i) a variant employing random GS mapping, which emulates the absence of the proposed NNS-based mapping, and ii) a variant in which JPEG XL coding is disabled and the 2D maps are instead encoded using a default PNG format. Finally, to gain insight into the impact of the quantization component (whose behavior depends on the 2D map creation) a third variant is introduced in which both NNS-based mapping and JPEG XL coding are disabled. This variant employs random mapping for 2D map creation and relies primarily on the GS parameter quantization.

TABLE IV  
(a) 3DGS-based input files
<table><tr><td>Technique</td><td>FSIMâ</td><td>DISTSâ</td><td>Size [MB]â</td></tr><tr><td>3DGS (no compression) 3DGS (w/ pruning)</td><td>0.925 0.925</td><td>0.071 0.071</td><td>306.2</td></tr><tr><td>GSICO</td><td>0.921</td><td>0.074</td><td>306.1 15.0</td></tr><tr><td>GSICO (w/o NNS-based mapping)</td><td>0.920</td><td>0.075</td><td>21.6</td></tr><tr><td>GSICO (w/o JPEG XL coding) GSICO (w/o NNS and JPEG XL)</td><td>0.922 0.922</td><td>0.072</td><td>21.3</td></tr></table>

(b) Scaffold-GS-based input files
<table><tr><td>Technique</td><td>FSIMâ</td><td>DISTSâ</td><td>Size [MB]â</td></tr><tr><td>Scaffold-GS (no compression) Scaffold-GS (w/ pruning)</td><td>0.929 0.929</td><td>0.068 0.068</td><td>80.0 79.9</td></tr><tr><td>GSICO</td><td>0.926</td><td>0.070</td><td>9.9</td></tr><tr><td>GSICO (w/o NNS-based mapping)</td><td>0.926</td><td>0.070</td><td>10.7</td></tr><tr><td>GSICO (w/o JPEG XL coding)</td><td>0.926</td><td>0.070</td><td>11.3</td></tr><tr><td>GSICO (w/o NNS and JPEG XL)</td><td>0.926</td><td>0.070</td><td>12.5</td></tr></table>

For both baseline model configurations, the pruning component has a negligible impact on RD performance, as expected, since its role is solely to impose the structural constraints required to arrange GS parameters into 2D maps. Regarding the GS mapping, the proposed NNS-based strategy consistently improves compression efficiency compared to random mapping, especially for the 3DGS-based configuration. This behavior can be attributed to the higher parameter variability exhibited by Scaffold-GS, which limits the exploitation of spatial coherence, leading to less smooth NNSbased maps. Regarding the image coding, JPEG XL outperforms PNG in terms of compression efficiency for the evaluated variant. In the 3DGS-based configuration, the lossy JPEG XL coding applied to some maps introduces a residual quality degradation, whereas the Scaffold-GS-based configuration employs fully lossless JPEG XL coding and therefore does not exhibit any quality degradation. Finally, quantization yields the most significant RD gains, particularly in the 3DGS-based configuration, where the combination of SH color space conversion and the removal of the chrominance AC coefficients lead to a significant reduction in model size.

## V. FINAL REMARKS AND FUTURE WORK

The main objective of this work was to address the challenge of efficiently compressing GS models for practical deployment in real-world multimedia scenarios. GS methods, notably 3DGS and Scaffold-GS, have emerged as leading representations for radiance fields due to their favorable trade-off between rendering quality and computational efficiency. However, their high storage requirements limit their applicability in bandwidth-constrained or storage-limited environments. This paper addresses this limitation by introducing GSICO, an efficient, versatile, and post-training GS codec. GSICO maps GS parameters into structured 2D images, which are subsequently compressed using a standard image codec. A novel 2D GS parameter mapping strategy is proposed to enhance spatial coherence within these images, thereby significantly improving their compressibility.

Comprehensive experiments across three widely used radiance field datasets demonstrates the effectiveness of the proposed approach. When applied to 3DGS-based models, GSICO achieves an average compression factor of 20.2Ã with negligible perceptual degradation, while for Scaffold-GS-based models it yields an average reduction of 8.5Ã. RD performance evaluations further show that the GSICO Scaffold-GS-based configuration consistently outperforms existing benchmarks. Unlike most existing GS compression approaches, which tightly integrate compression mechanisms into the training process and therefore require access to the original training data, GSICO operates entirely as a post-training codec. This design choice enables immediate adoption in practical multimedia scenarios where training data are unavailable. Moreover, GSICOâs compatibility with both 3DGS-based and Scaffold-GS-based representations enhances its versatility, ensuring applicability across the two most representative GS model baselines.

Future work will focus on three main directions: i) implementing a learning-based quantization strategy, where quantization step sizes are adaptively determined based on both the statistical properties and the perceptual relevance of each GS parameter; ii) while JPEG XL was selected for its outstanding performance in coding structured parameter images, emerging learning-based image codecs, such as JPEG AI [39], offer promising opportunities for even greater efficiency, provided they are properly fine-tuned to the statistical characteristics of GS parameters; and iii) extending the proposed codec framework to support dynamic GS models represents another compelling research direction.

## REFERENCES

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: representing scenes as neural radiance fields for view synthesis,â in Commun. ACM, vol. 65, no. 1, pp. 99â106, Dec. 2021.

[2] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, â3D Gaussian Splatting for Real-Time Radiance Field Rendering,â in ACM Trans. Graph., vol. 42, no. 4, p. 139:1-139:14, Jul. 2023.

[3] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, and Y. He, â3D Gaussian Splatting as a New Era: A Survey,â in IEEE Trans. Vis. Comput. Graph., vol. 31, no. 8, pp. 4429â4449, Aug. 2025.

[4] T. Lu et al., âScaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering,â in IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Seattle, WA, USA, Jun. 2024, pp. 20654â20664.

[5] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, âTensoRF: Tensorial Radiance Fields,â in Eur. Conf. Comput. Vis. (ECCV), Tel Aviv, Israel, Oct. 2022, pp. 333â350.

[6] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, âPlenoxels: Radiance Fields Without Neural Networks,â in

IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), New Orleans, LA, USA, Jun. 2022, pp. 5501â5510.

[7] T. MÃ¼ller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics primitives with a multiresolution hash encoding,â in ACM Trans. Graph., vol. 41, no. 4, p. 102:1-102:15, Jul. 2022.

[8] S. Li, H. Li, Y. Liao, and L. Yu, âNeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for Memory-Efficient Scene Representation,â in IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Seattle, WA, USA, Jun. 2024, pp. 21274â21283.

[9] Y. Xing, Q. Yang, K. Yang, Y. Xu, and Z. Li, âExplicit-NeRF-QA: A Quality Assessment Database for Explicit NeRF Model Compression,â Jul. 11, 2024, arXiv:2407.08165 [eess.IV].

[10] R. Khatib and R. Giryes, âTriNeRFLet: A Wavelet Based Triplane NeRF Representation,â in Eur. Conf. Comput. Vis. (ECCV), Milan, Italy, Nov. 2024, pp. 358â374.

[11] D. Rho, B. Lee, S. Nam, J. C. Lee, J. H. Ko, and E. Park, âMasked Wavelet Representation for Compact Neural Radiance Fields,â in IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Vancouver, BC, Canada, Jun. 2023, pp. 20680â20690.

[12] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Drettakis, âReducing the Memory Footprint of 3D Gaussian Splatting,â in ACM Comput. Graph. Interact. Tech., vol. 7, no. 1, p. 16:1-16:17, May 2024.

[13] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, âLightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS,â Mar. 29, 2024, arXiv:2311.17245 [cs.CV].

[14] X. Liu, X. Wu, P. Zhang, S. Wang, Z. Li, and S. Kwong, âCompGS: Efficient 3D Scene Representation via Compressed Gaussian Splatting,â in ACM Int. Conf. Multimed., New York, NY, USA: Association for Computing Machinery, Oct. 2024, pp. 2936â2944.

[15] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, âCompact 3D Scene Representation via Self-Organizing Gaussian Grids,â in Eur. Conf. Comput. Vis. (ECCV), Milan, Italy, Nov. 2024, pp. 18â34.

[16] S. Lee, F. Shu, Y. Sanchez, T. Schierl, and C. Hellge, âCompression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs,â Jan. 06, 2025, arXiv:2501.03399 [cs.CV].

[17] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, âHAC: Hash-Grid Assisted Context for 3D Gaussian Splatting Compression,â in Eur. Conf. Comput. Vis. (ECCV), Milan, Italy, Sep. 2024, pp. 422â438.

[18] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, âHAC++: Towards 100X Compression of 3D Gaussian Splatting,â Feb. 11, 2025, arXiv: arXiv:2501.12255 [cs.CV].

[19] K. L. Navaneet, K. P. Meibodi, S. A. Koohpayegani, and H. Pirsiavash, âCompact3D: Smaller and Faster Gaussian Splatting with Vector Quantization,â Jun. 11, 2024, arXiv:2311.18159 [cs.CV].

[20] MPEG 3DG Subgroup, âPCC Test Model Category 2 v0,â in ISO/IEC JTC1/SC29/WG11/N17248. Macau, China, Oct. 2017.

[21] H. Kirchhoffer et al., âOverview of the Neural Network Compression and Representation (NNR) Standard,â in IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 5, pp. 3203â3216, May 2022.

[22] ITU Recommendation BT.601, âStudio encoding parameters of digital television for standard 4:3 and wide-screen 16:9 aspect ratios,â Int. Telecommun. Union, Mar. 2021.

[23] J. Alakuijala et al., âJPEG XL next-generation image compression architecture and coding tools,â in SPIE Appl. Digit. Image Process. XLII, Sep. 2019, pp. 112â124.

[24] J. Macqueen, âMultivariate Observations: Some Methods for Classification and Analysis of Multivariate Observations,â in Berkeley Symp. Math. Stat. Probab., Oakland, CA, USA, Jan. 1967, pp. 281â298.

[25] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, âTanks and temples: benchmarking large-scale scene reconstruction,â in ACM Trans. Graph., vol. 36, no. 4, p. 78:1-78:13, Jul. 2017.

[26] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Brostow, âDeep blending for free-viewpoint image-based rendering,â in ACM Trans. Graph., vol. 37, no. 6, p. 257:1-257:15, Dec. 2018.

[27] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields,â in IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), New Orleans, LA, USA, Jun. 2022, pp. 5470â5479.

[28] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, âImage Quality Assessment: Unifying Structure and Texture Similarity,â in IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 5, pp. 2567â2581, May 2022.

[29] L. Zhang, L. Zhang, X. Mou, and D. Zhang, âFSIM: A Feature Similarity Index for Image Quality Assessment,â in IEEE Trans. Image Process., vol. 20, no. 8, pp. 2378â2386, Aug. 2011.

[30] P. Martin, A. Rodrigues, J. Ascenso, and M. P. Queluz, âGS-QA: Comprehensive Quality Assessment Benchmark for Gaussian Splatting View Synthesis,â Jun. 16, 2025, arXiv: arXiv:2502.13196 [cs.MM].

[31] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â in IEEE Trans. Image Process., vol. 13, no. 4, pp. 600â612, Apr. 2004.

[32] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe Unreasonable Effectiveness of Deep Features as a Perceptual Metric,â in IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Salt Lake City, UT, USA, Jun. 2018, pp. 586â595.

[33] A. S. A. Rabby and C. Zhang, âBeyondPixels: A Comprehensive Review of the Evolution of Neural Radiance Fields,â Aug. 2023, arXiv:2306.03000 [cs.CV].

[34] S. Girish, K. Gupta, and A. Shrivastava, âEAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS,â Dec. 07, 2023, arXiv:2312.04564 [cs.CV].

[35] H. Wang et al., âEnd-to-End Rate-Distortion Optimized 3D Gaussian Representation,â in Eur. Conf. Comput. Vis. (ECCV), Milan, Italy, Nov. 2024, pp. 76â92.

[36] Y. Chen, Q. Wu, M. Li, W. Lin, M. Harandi, and J. Cai, âFast Feedforward 3D Gaussian Splatting Compression,â in Int. Conf. Learn. Rep. (ICLR), Singapore, Singapore, Oct. 2024.

[37] L. Liu, Z. Chen, W. Jiang, W. Wang, and D. Xu, âHEMGS: A Hybrid Entropy Model for 3D Gaussian Splatting Data Compression,â Apr. 22, 2025, arXiv: arXiv:2411.18473 [cs.CV].

[38] Y. Wang, Z. Li, L. Guo, W. Yang, A. C. Kot, and B. Wen, âContextGS: compact 3D Gaussian splatting with anchor level context model,â in Int. Conf. Neural Info. Process. Syst. (NIPS), in NIPS â24, vol. 37. Red Hook, NY, USA: Curran Associates Inc., Dec. 2024, pp. 51532â51551.

[39] S. Esenlik et al., âAn overview of the JPEG AI learning-based image coding standard,â in IEEE Trans. Circuits Syst. Video Technol., Sep. 2025.

<!-- image-->

Pedro Martin (Member, IEEE) received the B.S. and M.S. degrees in electrical and computer engineering from Instituto Superior TÃ©cnico (IST), University of Lisbon (UL), Portugal, in 2019 and 2021, respectively. He is currently pursuing the Ph.D. degree with the Department of Electrical and Computer Engineering, 1 D :th stit 1

IST/UL. He has been a Researcher with Instituto de TelecomunicaÃ§Ãµes, since 2021, and a Teaching Assistant with the Department of Electrical and Computer Engineering, IST/UL, since 2020. His main research interests include visual quality assessment and coding, with a particular focus on neural radiance fields.

<!-- image-->

AntÃ³nio Rodrigues (Member, IEEE) received the B.S. and M.S. degrees in electrical and computer engineering from Instituto Superior TÃ©cnico (IST), Technical University of Lisbon, Lisbon, Portugal, in 1985 and 1989, respectively, and the Ph.D. degree from the Catholic University of Louvain, Louvain-la-Neuve, Belgium, in 1997. Since 1985, he has been with the

Department of Electrical and Computer Engineering, IST, where he is currently an Associate Professor. He is also a Senior Research Member of Instituto de TelecomunicaÃ§Ãµes, Lisbon. His current research interests include mobile and satellite communications, wireless networks, modulation, coding, and multiple access techniques.

<!-- image-->

JoÃ£o Ascenso (Senior Member, IEEE) received the E.E., M.Sc., and Ph.D. degrees in electrical and computer engineering from Instituto Superior TÃ©cnico (IST), Universidade TÃ©cnica de Lisboa, Lisbon, Portugal, in 1999, 2003, and 2010, respectively. He is currently an Associate Professor with the Department of Electrical

and Computer Engineering, IST, and a member of Instituto de TelecomunicaÃ§Ãµes.

He has authored more than 100 papers in international conferences. His current research interests include visual coding, quality assessment, light field and point cloud processing, and indexing and searching of multimedia content. He was an Associate Editor of IEEE TRANSACTIONS ON IMAGE PROCESSING and IEEE TRANSACTIONS ON MULTIMEDIA.

<!-- image-->

Maria Paula Queluz received the B.S. and M.S. degrees in electrical and computer engineering from Instituto Superior TÃ©cnico (IST), University of Lisbon, Portugal, and the Ph.D. degree from the Catholic University of Louvain, Louvainla-Neuve, Belgium. She is currently an Associate Professor with the Department of

Electrical and Computer Engineering, IST, and a Senior Research Member of Instituto de TelecomunicaÃ§Ãµes, Lisbon, Portugal. Her main scientific and research interests include image/video quality assessment, image/video processing, and wireless communications.