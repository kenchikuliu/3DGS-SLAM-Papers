# Image-Conditioned 3D Gaussian Splat Quantization

Xinshuang Liu Runfa Blark Li Keito Suzuki Truong Nguyen

University of California, San Diego

San Diego, CA, USA

xil235@ucsd.edu rul002@ucsd.edu k3suzuki@ucsd.edu tqn001@ucsd.edu https://XinshuangL.github.io/ICGS-Quantizer

## Abstract

3D Gaussian Splatting (3DGS) has attracted considerable attention for enabling high-quality real-time rendering. Although 3DGS compression methods have been proposed for deployment on storage-constrained devices, two limitations hinder archival use: (1) they compress medium-scale scenes only to the megabyte range, which remains impractical for large-scale scenes or extensive scene collections; and (2) they lack mechanisms to accommodate scene changes after long-term archival. To address these limitations, we propose an Image-Conditioned Gaussian Splat Quantizer (ICGS-Quantizer) that substantially enhances compression efficiency and provides adaptability to scene changes after archiving. ICGS-Quantizer improves quantization efficiency by jointly exploiting inter-Gaussian and inter-attribute correlations and by using shared codebooks across all training scenes, which are then fixed and applied to previously unseen test scenes, eliminating the overhead of per-scene codebooks. This approach effectively reduces the storage requirements for 3DGS to the kilobyte range while preserving visual fidelity. To enable adaptability to post-archival scene changes, ICGS-Quantizer conditions scene decoding on images captured at decoding time. The encoding, quantization, and decoding processes are trained jointly, ensuring that the codesâquantized representations of the sceneâare effective for conditional decoding. We evaluate ICGS-Quantizer on 3D scene compression and 3D scene updating. Experimental results show that ICGS-Quantizer consistently outperforms state-of-the-art methods in compression efficiency and adaptability to scene changes. Our code, model, and data will be publicly available on GitHub.

## 1. Introduction

3D Gaussian Splatting (3DGS) [25] has gained considerable attention for its real-time rendering capabilities and highquality visual outputs. However, 3DGS representations typically require substantial storage, which limits their applicability to large-scale scenes and extensive scene collections. This storage overhead is particularly problematic on storage-constrained devices such as robots, AR/VR headsets, and smartphones. A promising solution is to compress 3D Gaussians before storage. Recent state-of-the-art methods [27, 38] achieve compression by quantizing 3D Gaussians into discrete codes and storing associated codebooks for decoding. These codebooks are learned individually per scene via vector quantization [26, 45]. While effective, these approaches face two fundamental limitations, which we address with ICGS-Quantizerâan Image-Conditioned Gaussian Splat Quantizer.

Limitation 1: Inefficient storage at scale. Existing 3DGS quantizers typically reduce medium-scale scenes only to the megabyte range, which remains impractical for large-scale environments or extensive collections. Two key factors drive this inefficiency: (1) they train separate codebooks for each scene, incurring considerable storage overhead because each scene must store its own floatingpoint codebooks in addition to the quantized codes; and (2) they quantize each Gaussian and its attributes (e.g., rotation, scale) independently, missing correlations both across Gaussians and among attributes. To address this, we train ICGS-Quantizer on large-scale data to learn shared codebooks across all training scenes, then fix and apply them to previously unseen test scenes, eliminating per-scene codebook storage. Furthermore, we jointly compress all Gaussians and their attributes to capture inter-Gaussian and interattribute correlations. Precisely, the scene is partitioned into sparse 3D blocks, each containing multiple Gaussians, and each block is compressed into residual discrete codes. Within each blockâwhich preserves local spatial structureâthe Gaussians and their attributes are jointly encoded into latent representations before quantization. This design reduces 3DGS storage requirements to the kilobyte range while preserving high visual fidelity.

Limitation 2: No post-archival scene adaptation. In real-world applications, achieving a high compression ratio for 3D scenes is not the sole objective. Over time, scenes may change after archivalâfor example, lighting can shift from day to night, or colors can fade. Therefore, a critical goal during scene decoding is to adapt scenes to their current illumination and appearance. We achieve this by conditioning scene decoding on one or a few current images (Figure 1). Specifically, our conditional decoder extracts multiscale image features from a pre-trained DINOv2 model [40] and fuses them into sparse 3D latents via visibility-aware, coarse-to-fine aggregation. This design enables the decoded scene to adapt to its current illumination and appearance. Furthermore, the encoding, quantization, and decoding processes are trained jointly, so that the quantized codes are optimized for conditional decoding. When no scene update is needed, our ICGS-Quantizer can still decode the archived scene without requiring any conditioning images.

<!-- image-->  
Figure 1. Image-conditioned scene quantization. (a) At time t0, the scene is encoded and quantized as discrete codes. However, after a prolonged period, the scene may have changed. (b) Conventional methods decode the scene from the codes, but can only recover the original scene at time t0<t. (c) Our method decodes the scene from the codes conditioned on its image(s) captured at time t, adapting the scene to its current illumination and appearance. We recommend watching our videos to observe the dynamic results.

We evaluate ICGS-Quantizer on both 3D scene compression and 3D scene updating. For compression, our shared-codebook and joint quantization strategies achieve higher compression ratios and superior rendering quality compared to state-of-the-art methods; for fairness, we do not use conditioning images for our method in this setting. For scene updating, conditioning on current views provides strong adaptability to changes, achieving the highest rendering quality for updated scenes. Additional experiments validate the effectiveness of our coarse-to-fine imageconditioning strategy and demonstrate that the quantizer effectively exploits multiple views while maintaining robustness with only a few input views.

Our contribution can be summarized as follows:

â¢ Shared codebooks for all scenes. We learn codebooks that are shared across all training scenes and fix them at test time, eliminating per-scene codebook storage and improving generalization.

â¢ Joint Gaussianâattribute quantization. We jointly quantize multiple Gaussians and their attributes to exploit both inter-Gaussian and inter-attribute correlations, enabling highly efficient compression.

â¢ Image-conditioned scene decoding. We condition scene decoding on a fewâor even a singleâcurrent view of the scene to adapt it to its current state, in a coarse-tofine manner. The encoding, quantization, and decoding processes are trained jointly, so that the quantized codes are optimized for conditional decoding.

## 2. Related Work

## 2.1. Vector Quantization

Vector quantization (VQ) [2, 16] approximates a vector with its nearest entry in a learned codebook of representative vectors. The objective of VQ is to minimize the discrepancy between the original vectors and their quantized counterparts. Various extensions have been proposed to enhance quantization accuracy and efficiency. Residual quantization [4, 24, 34] iteratively quantizes residuals across multiple stages, with each stage quantizing the difference between the original vector and its reconstruction from previous stages. Product quantization [14, 23, 44] partitions vectors into sub-vectors, quantizing each independently. However, traditional vector quantization methods remain inherently non-differentiable, restricting their compatibility with end-to-end neural network training. VQ-VAE [45] addressed this by copying gradients from the quantizerâs outputs to its inputs. To enhance accuracy, subsequent work extended this approach as residual quantization [26] and proposed implicit codebooks that are dynamically generated based on the outputs of preceding quantization stages [21]. We adopt residual quantization with shared codebooks across all training scenes, fixing them when applied to unseen test scenes. This design achieves high compression by avoiding per-scene codebook storage.

## 2.2. 3D Gaussian Splat Compression

Neural Radiance Fields (NeRFs) [36] deliver high-quality renderings and have supported many applications [30, 35, 47]. Subsequent methods enhanced NeRFâs computational efficiency by using ensembles of smaller MLPs [37, 42], each operating efficiently. Other methods enhanced sampling efficiency through grid-based [12] or point-based [53] representations. Combining both computational and sampling efficiency, 3D Gaussian Splatting (3DGS) [25] enables fast 3D reconstruction and real-time rendering, forming a computationally efficient foundation for various tasks [17, 28, 41]. However, unlike NeRF-based methods, 3DGS incurs substantial storage due to the large number of Gaussians. To mitigate this, recent methods [6, 33, 48] proposed compact 3DGS representations by leveraging anchor levels that capture shared features among Gaussians. Further compression methods [11, 15, 27, 38, 39] applied vector quantization (VQ) [26, 45] to quantize Gaussians into discrete codes alongside necessary floating-point data, reducing storage demands to the megabyte range. However, these approaches typically (i) train unshared, perscene codebooks, adding floating-point storage overhead, and (ii) quantize each Gaussian and its attributes independently, overlooking inter-Gaussian and inter-attribute correlations. In contrast, we learn shared codebooks across training scenes and jointly quantize multiple Gaussians with their attributes, enabling kilobyte-level compression while preserving visual fidelity, thus enhancing scalability for large or numerous scenes on storage-limited devices.

<!-- image-->  
Figure 2. An overview of our proposed ICGS-Quantizer. (a) Quantization module: geometry and texture are decoupled and quantized independently. (b) Joint quantization of 3D Gaussians and their associated attributes. (c) Image-conditioned decoding of 3D Gaussians. Auxiliary neural network branches and training objectives are omitted here for simplicity.

## 2.3. 3D Scene Updating

Several scene editing methods have been proposed for 3D Gaussian Splatting (3DGS) [25], enabling applications such as object removal [49], scene relighting [1, 13, 18], and natural language-driven edits [3, 5, 50, 52]. While these methods produce high-quality results, they do not address our objective: adapting scenes to their current views when decoding them from previously quantized codes. The most relevant work lies in style transfer [7, 9, 19], which adjusts the style of one image to match reference images. Deng et al. [9] utilized transformer architectures for high-quality style transfer. Hong et al. [19] introduced a pattern repeatability metric to quantify the rhythm and repetition of local patterns in the style image, thereby enhancing style transfer efficacy. Chung et al. [7] leveraged pre-trained diffusion models to enhance style transfer performance. Recent works [22, 29] incorporated style transfer into 3DGS for multi-view consistency. Unlike these methods, which focus on style manipulation, we perform image-conditioned decoding of previously quantized scene codes, enabling scene updates to their current views without re-optimization.

## 3. Methodology

Figure 2 provides an overview of our method. We first encode each 3DGS scene into a latent representation that is decoupled into geometry and texture features (Sec. 3.2), with this decoupling reinforced by the training objectives in Sec. 3.4. The resulting features are quantized into discrete geometry and texture codes via residual vector quantizers (RVQs) whose codebooks are shared across all training scenes and frozen at test time, eliminating per-scene codebook storage. Storing only discrete codes rather than floating-point latent vectors significantly reduces storage. To reconstruct a scene (Sec. 3.3), we recover the latent representation from the codes and decode Gaussian attributes, optionally conditioning on current images to adapt the scene to its current illumination and appearance.

## 3.1. Preliminary

3D Gaussian Splatting. Kerbl et al. [25] proposed representing a scene using a collection of 3D Gaussians. Each 3D Gaussian G is parameterized as follows: (1) a mean vector $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ indicating its center; (2) spherical harmonics (SH) coefficients $\boldsymbol { y } \in \mathbb { R } ^ { k }$ modeling view-dependent appearance, with k denoting the degrees of freedom; (3) a quaternion $\pmb { r } \in \mathbb { R } ^ { 4 }$ for rotation; (4) a scaling factor $\boldsymbol { s } \in \mathbb { R } ^ { 3 }$ ; and (5) an opacity value $\sigma \in \mathbb { R } _ { + }$ . A scene is thus defined by a collection $\{ \mathcal { G } _ { i } = ( \pmb { \mu _ { i } } , \pmb { y _ { i } } , \pmb { r _ { i } } , \pmb { s _ { i } } , \sigma _ { i } ) \} _ { i \in [ N ] }$ , where N denotes the number of Gaussians. Using the rotation matrix R derived from r, the Gaussian covariance $\pmb { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ is computed as:

$$
\Sigma = R \mathrm { d i a g } ( s ) ^ { 2 } R ^ { T } .\tag{1}
$$

Given a viewing transformation W and the Jacobian J of the affine approximation of the projective transformation, Zwicker et al. [55] compute the covariance in camera coordinates as:

$$
\pmb { \Sigma ^ { \prime } } = \pmb { J } \pmb { W } \pmb { \Sigma } \pmb { W } ^ { T } \pmb { J } ^ { T } .\tag{2}
$$

The color of each rendered pixel p is computed as:

$$
C ( \pmb { p } ) = \sum _ { i = 1 } ^ { N } \pmb { c } _ { i } \sigma _ { i } G _ { i } ^ { 2 D } ( \pmb { p } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \sigma _ { j } G _ { j } ^ { 2 D } ( \pmb { p } ) \right) ,\tag{3}
$$

where $c _ { i }$ denotes the view-dependent color computed from the SH coefficients $\mathbf { \nabla } _ { \mathbf { { y } } _ { i } } .$ and $G _ { i } ^ { 2 D } ( \pmb { p } )$ is defined as:

$$
G _ { i } ^ { 2 D } ( \pmb { p } ) = e ^ { - \frac { 1 } { 2 } ( \pmb { p } - \pmb { \mu } _ { i } ) ^ { T } \left( \pmb { \Sigma } ^ { 2 D } \right) ^ { - 1 } ( \pmb { p } - \pmb { \mu } _ { i } ) } ,\tag{4}
$$

where $\Sigma ^ { 2 D }$ is derived by removing the third row and column of $\Sigma ^ { \prime }$

Vector Quantization. Vector quantization (VQ) uses a codebook $\mathcal { C } = \{ ( i , e _ { i } ) \} _ { i \in [ N ] }$ , containing N codes i and associated embeddings $e _ { i } \in \mathbb { R } ^ { d _ { \epsilon } }$ . For an input vector x â $\mathbb { R } ^ { d _ { \epsilon } }$ e, VQ outputs the nearest code embedding:

$$
\begin{array} { r } { \mathcal { Q } ( \pmb { x } ; \mathcal { C } ) = \underset { i \in [ N ] } { \arg \operatorname* { m i n } } \left\| \pmb { x } - \pmb { e } _ { i } \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{5}
$$

To reduce the quantization error, residual vector quantization (RVQ) quantizes the input vector using D codebooks. Starting with $\mathbf { \boldsymbol { r } } _ { 0 } = \mathbf { \boldsymbol { x } }$ , RVQ quantizes the residual value $\mathbf { \Delta } \mathbf { r } _ { d }$ at each iteration d:

$$
\begin{array} { r l } & { i _ { d } = \mathcal { Q } ( \pmb { r } _ { d - 1 } ; \mathcal { C } _ { d } ) , } \\ & { r _ { d } = \pmb { r } _ { d - 1 } - e _ { i _ { d } } , } \end{array}\tag{6}
$$

for $d = 1 , \ldots , D$ . The final approximation xË is obtained by summing the selected code embeddings:

$$
\hat { \pmb x } = \sum _ { d = 1 } ^ { D } \pmb e _ { i _ { d } } .\tag{7}
$$

RVQ requires $D \log _ { 2 } N$ bits of storage per input and offers exponentially increasing representational capacity with respect to the number of codebooks, serving as the foundation of our method.

## 3.2. Joint Quantization of Gaussians and Attributes

We encode a 3DGS scene into latent features and quantize them into discrete codes for efficient storage.

Grid 3DGS representation. Optimizing 3DGS typically requires Gaussian pruning/splitting to avoid poor local minima, making the number of Gaussians difficult to control and complicating storage planning. To address this, we discretize space into a sparse voxel grid; each non-empty cell holds one Gaussian and empty cells are omitted. This removes the need to store floating-point Gaussian centers Âµ and enables direct control of storage via grid resolution.

Grid-block feature encoding. As illustrated in Figure 2(b), we encode the per-grid Gaussian attributesâSH coefficients y, quaternion r, scale s, and opacity Ïâinto vectors $\pmb { f } _ { y } , \pmb { f } _ { r } , \pmb { f } _ { s } , \pmb { f } _ { \sigma } \in \mathbb { R } ^ { d _ { g } }$ , where $d _ { g } = 3 2$ is the gridlevel feature dimension. These vectors are concatenated and linearly projected to form a grid feature vector $\pmb { f } _ { q } \in \mathbb { R } ^ { d _ { g } }$ . To capture local structure, we group each $K \times K \times K$ neighboring grids into a block (with $K { = } 4 )$ . Within each block, grid feature vectors are processed by 3D convolutional layers to produce a block feature vector $\pmb { f } _ { b } \in \mathbb { R } ^ { d _ { b } } ( d _ { b } { = } 1 2 8 )$ padding empty grids with a learnable vector. The feature vectors from all non-empty blocks form a sparse 3D feature map, which a 3D U-Net [43] further processesâagain with learned paddingâto produce the sceneâs sparse latent.

Scene latent quantization. We project the latent into geometry and texture features and quantize each with a RVQ to obtain geometry and texture codes (Figure 2(a)). The training objectives in Sec. 3.4 enforce the intended decoupling. To eliminate per-scene codebook storage and enhance generalizability, we use shared codebooks across all training scenes, and fix them for unseen scenes at test time.

Our method reduces 3DGS storage to the kilobyte range while preserving high visual fidelity by (i) jointly encoding Gaussians to exploit correlations and (ii) sharing codebooks across scenes to avoid per-scene codebook storage.

## 3.3. Image-Conditioned Scene Decoding

We decode a scene from its codes and optionally condition the process on its current images to adapt to scene changes.

Scene latent recovery. The decoding process begins by reconstructing geometry and texture latents from their codes via the RVQs and fusing them with a linear projection to form a scene latent (Figure 2(a)).

Image feature extraction. Optionally, in order to adapt to scene changes, we condition the scene decoding on features extracted from one or more current images (Figure 2(c)). Image features come from a frozen DINOv2 model [40] augmented by a lightweight 2D U-Net (trained from scratch). Grid-level features are obtained by projecting non-empty grid centers onto image planes to query the feature maps, then averaging the retrieved features for each grid cell, weighted by view-dependent visibility; to obtain robust weights, we estimate visibility using non-empty cells only and share the same visibility across cells in a block. The resulting grid-level features are then aggregated to form block-level image features.

Coarse-to-fine conditioning. When images are available, scene decoding is conditioned at two levels. Coarse image condition: block-level image features are injected into the scene latent via a 3D U-Net to produce a block latent, with empty blocks padded by a learnable vector. Fine image condition: the block latent is upsampled to the grid cells and fused with grid-level image features to produce a grid latent. This coarse-to-fine design first imparts global semantics and then refines local details. The effectiveness of both conditioning stages is validated in Sec. 4.4.

Gaussian attribute decoding. Finally, the Gaussian attributes are decoded from the grid latent using geometry and texture decoders. Geometry attributes include the rotation quaternion r, scaling factor s, and opacity Ï; texture attributes include spherical harmonics coefficients y.

In summary, scene decoding can be optionally conditioned on image features to adapt to current illumination and appearance. This conditioning follows a coarse-to-fine strategy: the coarse stage imparts global semantics, while the fine stage refines local details.

## 3.4. Training Objective

We supervise the model using both Gaussian attributes, which provide direct guidance early in training, and images, which enhance the final visual quality.

Loss functions. For Gaussian attribute supervision, we use a Huber loss [20] on rotations and scaling factors and a binary cross-entropy on opacity. All terms use equal weights of 1 without dataset-specific tuning, minimizing hyperparameters and enhancing reproducibility. The Huber loss stabilizes training by minimizing sensitivity to outliers. For image supervision, we adopt the photometric loss from 3DGS [25] and add a Mean Squared Error (MSE) term for training stability. To ensure compatibility of vector quantization within the deep neural network, we additionally include a commitment loss term [45].

Training objective design. We supervise the model both before image conditioning (to preserve the archived scene) and after conditioning (to adapt to the current scene), thereby optimizing the quantized codes for both archival fidelity and post-archival adaptability. Geometryâtexture decoupling is enforced through (i) separate quantizers and decoders for geometry and texture features, and (ii) a stopgradient from non-geometry objectives into the geometry quantizer outputs, encouraging geometry codes to specialize in geometric signals.

Optimization. We train the model for 100 epochs using AdamW [32] $( \beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 5 )$ with gradient clipping and a batch size of 8. In epochs 1â10, vector quantization is disabled and the learning rate is linearly warmed up [46] to $1 \times 1 0 ^ { - 4 }$ . In epochs 11â90, the learning rate follows a cosine decay schedule [31] down to $1 \times 1 0 ^ { - 5 }$ . In the same interval, the Gaussian-attribute loss weight is reduced from 0.1 to 0, while the commitment loss weight is increased from 0.01 to 0.1. Codebooks are updated via k-means clustering every two epochs. In the final 10 epochs, both the learning rate and codebooks are fixed.

This subsection presents the training objective; the complete formulation of the losses and additional implementation details are provided in the supplementary material.

## 4. Experiments

This section presents a thorough evaluation of ICGS-Quantizer. Sec. 4.1 details the experiment setup; Sec. 4.2 and Sec. 4.3 report results for 3D scene compression and 3D scene updating, respectively; Sec. 4.4 analyzes the effect of image conditioning and the number of input images.

## 4.1. Experiment Setup

We evaluate our method on the Google Scanned Objects (GSO) dataset [10], which contains real-world scanned objects. After quality filtering, a total of 360 objects are used to create 36,000 training scenes, while the remaining 100 unseen objects form 100 test scenes for evaluation. Each scene has two states: an archived version at time $t _ { 0 }$ and a changed version at time $t > t _ { 0 }$ . Details of the dataset construction are provided in the supplementary material. For quantitative evaluation, we render 50 novel views per scene by uniformly sampling camera poses over a full 360â¦ rotation. We report Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) [51], and Learned Perceptual Image Patch Similarity (LPIPS) [54]. All methods operate with 3 degrees of freedom for each Gaussian spherical harmonic. Our method uses residual codebooks with a depth of 4 and a size of 1024, shared across all training scenes. The codebooks remain fixed during evaluation on test scenes. To contextualize performance under varying storage constraints, we include grid-based 3DGS variants as baselines for both 3D scene compression and 3D scene updating, benefiting from their convenient storage management: 3DGS-small, 3DGS-medium, and 3DGS-large, corresponding to grid resolutions of 32, 64, and 128, respectively.

## 4.2. 3D Scene Compression

In addition to the 3DGS baselines, we compare against state-of-the-art compression methods for 3DGS representations, including CompGS [38] and C3DGS [39], using a codebook size of 256 per scene. For a fair comparison, both our method and the baseline compression methods are applied to well-optimized 3DGS without image-based reoptimization, with Gaussians having negative opacity logits discarded to reduce storage. Our method employs a block resolution of 32 for storage, whereas the compared compression methods use a grid resolution of 64 for storage, which is 2Ã finer than ours and thus requires more storage.

Table 1. Comparison with state-of-the-art methods for 3D scene compression. #Images denotes the number of images used during testing. The results are highlighted as best , second-best , and third-best , respectively.
<table><tr><td>Method</td><td>#Images</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>Storage (KB)</td></tr><tr><td>3DGS-small</td><td>0</td><td>25.23</td><td>0.9345</td><td>0.1546</td><td>121.0</td></tr><tr><td>3DGS-medium</td><td>0</td><td>28.79</td><td>0.9641</td><td>0.0729</td><td>561.9</td></tr><tr><td>3DGS-large</td><td>0</td><td>30.92</td><td>0.9777</td><td>0.0292</td><td>2661.1</td></tr><tr><td rowspan="2">CompGS C3DGS</td><td>0</td><td>27.47</td><td>0.9545</td><td>0.0830</td><td>21.9</td></tr><tr><td>0</td><td>27.90</td><td>0.9582</td><td>0.0785</td><td>21.1</td></tr><tr><td rowspan="2">ICGS-Quantizer</td><td>0</td><td>28.50</td><td>0.9634</td><td>0.0405</td><td>16.9</td></tr><tr><td>1</td><td>29.52</td><td>0.9681</td><td>0.0339</td><td>16.9</td></tr></table>

<!-- image-->  
Figure 3. Qualitative results of 3D scene compression. Zoom in to examine details. âLargeâ, âMediumâ, and âSmallâ indicate the number of Gaussians, balancing quality and storage.

Experimental results are presented in Table 1 and Figure 3. As expected, the performance of 3DGS improves with increased storage capacity, underscoring the importance of evaluating compression methods under comparable storage budgets. Under such constraints, our method significantly outperforms the state-of-the-art scene compression approaches even without the aid of conditioning images. For example, C3DGS achieves 27.90 PSNR, 0.9582 SSIM, and 0.0785 LPIPS with 21.1 KB of storage, whereas ICGS-Quantizer, at 16.9 KB, attains 28.50 PSNR, 0.9634 SSIM, and 0.0405 LPIPS. Remarkably, in this setting (no conditioning images), ICGS-Quantizer reaches performance comparable to uncompressed 3DGS-medium, which requires 33Ã the storage. With a single conditioning image, ICGS-Quantizer further approaches uncompressed 3DGSlarge, which demands 157Ã the storage.

These results demonstrate the storage efficiency of our method, attributable to (i) exploiting inter-Gaussian and inter-attribute correlations to reduce redundancy and (ii) using shared codebooks across all scenes, which eliminates per-scene codebook storage.

## 4.3. 3D Scene Updating

We evaluate the capability of ICGS-Quantizer to adapt the scene to post-archival changes through a 3D scene updating task. Each scene has two states: one archived at time t0 and one current at time t. The archived scene serves as input, while the current scene serves as ground truth. For all methods, six images are used to update the scene. Since the task requires adapting illumination and appearance from the provided images, we compare ICGS-Quantizer with style transfer methods, including StyleID [7], AesPA-Net [19], and StyTR2 [9]. We average their outputs from each reference to use multiple reference images, and refine the results using masks rendered by 3DGS. As 3D scene updating relates to continual learning, we further include comparisons with 3DGS under three settings: (1) using the archived 3DGS (a lower-end performance reference), (2) fine-tuning the archived 3DGS with current images (a continual-learning baseline), and (3) using the current 3DGS (an upper-end performance reference).

Table 2. Comparison with state-of-the-art methods for 3D scene updating. The results are highlighted as best , second-best , and third-best , respectively.
<table><tr><td>Storage (KB)</td><td>Method</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td></tr><tr><td rowspan="6">121.0</td><td>Archived 3DGS</td><td>24.01</td><td>0.9210</td><td>0.1655</td></tr><tr><td>Fine-tuned 3DGS</td><td>23.13</td><td>0.9238</td><td>0.1325</td></tr><tr><td>Current 3DGS</td><td>25.23</td><td>0.9345</td><td>0.1546</td></tr><tr><td>StyleID</td><td>23.75</td><td>0.9177</td><td>0.1188</td></tr><tr><td>AesPA-Net</td><td>23.91</td><td>0.9266</td><td>0.1141</td></tr><tr><td>StyTR2</td><td>23.57</td><td>0.9252</td><td>0.1177</td></tr><tr><td rowspan="6">561.9</td><td>Archived 3DGS</td><td>26.13</td><td>0.9421</td><td>0.0895</td></tr><tr><td>Fine-tuned 3DGS</td><td>26.52</td><td>0.9495</td><td>0.0598</td></tr><tr><td>Current 3DGS</td><td>28.79</td><td>0.9641</td><td>0.0729</td></tr><tr><td>StyleID</td><td>25.46</td><td>0.9320</td><td>0.0603</td></tr><tr><td>AesPA-Net</td><td>26.40</td><td>0.9435</td><td>0.0647</td></tr><tr><td>StyTR2</td><td>26.27</td><td>0.9458</td><td>0.0635</td></tr><tr><td rowspan="6">2661.1</td><td>Archived 3DGS</td><td>26.92</td><td></td><td></td></tr><tr><td>Fine-tuned 3DGS</td><td></td><td>0.9490</td><td>0.0502</td></tr><tr><td>Current 3DGS</td><td>28.46 30.92</td><td>0.9599 0.9777</td><td>0.0312 0.0292</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>StyleID AesPA-Net</td><td>25.78</td><td>0.9346 0.9494</td><td>0.0383 0.0455</td></tr><tr><td>StyTR2</td><td>27.49 27.93</td><td>0.9552</td><td>0.0414</td></tr><tr><td>16.9</td><td>ICGS-Quantizer</td><td>30.45</td><td>0.9772</td><td>0.0232</td></tr></table>

Experimental results are presented in Table 2 and Figure 4. Our method is compared against baselines with varying storage budgets, all of which require more storage than ours. Notably, the 3DGS of the archived scene (Archived 3DGS) performs poorly, despite utilizing substantial storage resources (2661.1 KB). This is primarily due to the appearance gap between the archived and current scenes, which cannot be bridged by increasing storage without incorporating information from the current scene. In contrast, the 3DGS of the current scene (Current 3DGS) performs well when provided with sufficient storage, highlighting the importance of incorporating updated visual information. Finetuning the archived 3DGS with current images (Fine-tuned 3DGS) improves performance over the Archived 3DGS, but still falls short of Current 3DGS. This indicates that while reference images contribute positively, the continual learning paradigm is constrained by a limited number of inputs. Among style transfer methods, StyTR2 at 2661.1 KB achieves the highest PSNR (27.93), comparable to Finetuned 3DGS (28.46) while requiring no re-optimization. This demonstrates the effectiveness of style transfer in lowdata scenarios. Finally, our ICGS-Quantizer significantly outperforms all baselines and is comparable to the upperend performance reference (Current 3DGS at 2661.1 KB), while surpassing it on LPIPS, yet requiring the least storage and no re-optimization.

Trained on large-scale scenes, ICGS-Quantizer learns to effectively extract and store the core information needed to decode scenes conditioned on current images. As a result, it demonstrates superior adaptability with minimal storage overhead, successfully updating scenes to reflect post-archival changes.

## 4.4. Evaluation of Image Conditioning

We begin with an ablation study assessing the effectiveness of image conditioning. As shown in Table 3, incorporating a coarse image condition yields a substantial improvement, highlighting the benefit of using up-to-date images during decoding. Further gains are achieved when using a fine image condition, with consistent improvements across PSNR, SSIM, and LPIPS for both tasks, thereby validating the effectiveness of integrating image information at a finer level.

We also examine the effect of varying the number of conditioning images. As shown in Figure 5, performance improves steadily as the number of input images increases, as reflected by PSNR, SSIM, and LPIPS. Remarkably, even with a single conditioning image, the model maintains strong performance. These observations demonstrate both the modelâs capacity to exploit multi-view information and its robustness in scenarios with limited input views.

<!-- image-->  
Figure 4. Qualitative results of 3D scene updating. For the baseline methods, âLargeâ and âSmallâ refer to configurations utilizing varying numbers of Gaussians, balancing quality and storage.

Table 3. Ablation study of image conditioning on 3D scene compression and 3D scene updating. Six images are used for conditioning.
<table><tr><td rowspan="2">Coarse image condition</td><td rowspan="2">Fine image condition</td><td colspan="3">3D scene compression</td><td colspan="3">3D scene updating</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td></tr><tr><td></td><td></td><td>28.50</td><td>0.9634</td><td>0.0405</td><td>26.33</td><td>0.9466</td><td>0.0524</td></tr><tr><td>&gt;&gt;</td><td></td><td>30.34</td><td>0.9752</td><td>0.0261</td><td>30.34</td><td>0.9750</td><td>0.0262</td></tr><tr><td></td><td>â</td><td>30.44</td><td>0.9773</td><td>0.0232</td><td>30.45</td><td>0.9772</td><td>0.0232</td></tr></table>

<!-- image-->  
Figure 5. Experimental results of ICGS-Quantizer for 3D scene compression (âcompressionâ) and 3D scene updating (âupdatingâ) using varying numbers of input images. Since the conditioning images are optional, â0â indicates no image is used.

## 5. Conclusion

We introduce ICGS-Quantizer, an Image-Conditioned Gaussian Splat Quantizer for archiving large-scale scenes and extensive scene collections. ICGS-Quantizer enhances quantization efficiency by (i) exploiting inter-Gaussian and inter-attribute correlations to remove redundancy and (ii) learning shared codebooks across training scenes that are fixed at test time, eliminating per-scene codebook storage. This design drives storage down to the kilobyte range while preserving visual fidelity. To enable adaptability to post-archival scene changes, ICGS-Quantizer conditions scene decoding on images captured at decoding time. The encoding, quantization, and decoding processes are trained jointly, so that the quantized codes are optimized for conditional decoding. Experimental results demonstrate that ICGS-Quantizer outperforms state-of-the-art methods in both compression efficiency and adaptability to scene changes.

Future Work. This work presents a promising approach for archiving data as discrete codes that are particularly well-suited for conditional decoding. Future research may explore extending this approach to other data modalities, such as images and audio. Additionally, incorporating pretrained foundation models into the method offers a potential direction for further enhancing compression efficiency.

## References

[1] Zoubin Bi, Yixin Zeng, Chong Zeng, Fan Pei, Xiang Feng, Kun Zhou, and Hongzhi Wu. GS3: Efficient relighting with triple gaussian splatting. In SIGGRAPH Asia 2024 Conference Papers, SA 2024, Tokyo, Japan, December 3-6, 2024, pages 12:1â12:12. ACM, 2024. 3

[2] Andres Buzo, A Gray, R Gray, and John Markel. Speech cod- Â´ ing based upon vector quantization. IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(5):562â574, 1980. 2

[3] Minghao Chen, Iro Laina, and Andrea Vedaldi. DGE: direct gaussian 3D editing by consistent multi-view editing. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXIV, pages 74â92. Springer, 2024. 3

[4] Yongjian Chen, Tao Guan, and Cheng Wang. Approximate nearest neighbor search by residual vector quantization. Sensors, 10(12):11259â11273, 2010. 2

[5] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3D editing with gaussian splatting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 21476â21485. IEEE, 2024. 3

[6] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. HAC: hash-grid assisted context for 3D gaussian splatting compression. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part VII, pages 422â438. Springer, 2024. 3

[7] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style injection in diffusion: A training-free approach for adapting largescale diffusion models for style transfer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 8795â 8805. IEEE, 2024. 3, 6

[8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit S. Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, IlaÂ¨Ä± Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Rame,Â´ Sagar Waghmare, Helen Miller, Nathan Byrd, Ashrith Sheshan, Raia Hadsell Sangnie Bhardwaj, Pawel Janus, Tero Rissa, Dan Horgan, Sharon Silver, Ayzaan Wahid, Sergey Brin, Yves Raimond, Klemen Kloboves, Cindy Wang, Nitesh Bharadwaj Gundavarapu, Ilia Shumailov, Bo Wang, Mantas Pajarskas, Joe Heyward, Martin Nikoltchev, Maciej Kula, Hao Zhou, Zachary Garrett, Sushant Kafle, Sercan Arik, Ankita Goel, Mingyao Yang, Jiho Park, Koji Kojima, Parsa Mahmoudieh, Koray Kavukcuoglu, Grace Chen, Doug

Fritz, Anton Bulyenov, Sudeshna Roy, Dimitris Paparas, Hadar Shemtov, Bo-Juen Chen, Robin Strudel, David Reitter, Aurko Roy, Andrey Vlasov, Changwan Ryu, Chas Leichner, Haichuan Yang, Zelda Mariet, Denis Vnukov, Tim Sohn, Amy Stuart, Wei Liang, Minmin Chen, Praynaa Rawlani, Christy Koh, JD Co-Reyes, Guangda Lai, Praseem Banzal, Dimitrios Vytiniotis, Jieru Mei, and Mu Cai. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025. 1

[9] Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, and Changsheng Xu. Stytr2: Image style transfer with transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 11316â 11326. IEEE, 2022. 3, 6

[10] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas Barlow McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3D scanned household items. In 2022 International Conference on Robotics and Automation, ICRA 2022, Philadelphia, PA, USA, May 23-27, 2022, pages 2553â2560. IEEE, 2022. 5

[11] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian: Unbounded 3D gaussian compression with 15x reduction and 200+ FPS. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. 3

[12] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 5491â5500. IEEE, 2022. 2

[13] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. Relightable 3D gaussians: Realistic point cloud relighting with BRDF decomposition and ray tracing. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLV, pages 73â89. Springer, 2024. 3

[14] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. Optimized product quantization. IEEE Trans. Pattern Anal. Mach. Intell., 36(4):744â755, 2014. 2

[15] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. EA-GLES: efficient accelerated 3D gaussians with lightweight encodings. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXIII, pages 54â71. Springer, 2024. 3

[16] Robert Gray. Vector quantization. IEEE Assp Magazine, 1 (2):4â29, 1984. 2

[17] Antoine Guedon and Vincent Lepetit. Sugar: Surface- Â´ aligned gaussian splatting for efficient 3D mesh reconstruction and high-quality mesh rendering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR

2024, Seattle, WA, USA, June 16-22, 2024, pages 5354â 5363. IEEE, 2024. 3

[18] Yijia Guo, Yuanxi Bai, Liwen Hu, Ziyi Guo, Mianzhi Liu, Yu Cai, Tiejun Huang, and Lei Ma. PRTGS: precomputed radiance transfer of gaussian splats for real-time high-quality relighting. In Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024, pages 5112â 5120. ACM, 2024. 3

[19] Kibeom Hong, Seogkyu Jeon, Junsoo Lee, Namhyuk Ahn, Kunhee Kim, Pilhyeon Lee, Daesik Kim, Youngjung Uh, and Hyeran Byun. Aespa-net: Aesthetic pattern-aware style transfer networks. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1- 6, 2023, pages 22701â22710. IEEE, 2023. 3, 6

[20] Peter J Huber. Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution, pages 492â518. Springer, 1992. 5

[21] Iris A. M. Huijben, Matthijs Douze, Matthew J. Muckley, Ruud van Sloun, and Jakob Verbeek. Residual quantization with implicit neural codebooks. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. 2

[22] Sahil Jain, Avik Kuthiala, Prabhdeep Singh Sethi, and Prakanshul Saxena. Stylesplat: 3D object style transfer with gaussian splatting. arXiv preprint arXiv:2407.09473, 2024. 3

[23] Herve J Â´ egou, Matthijs Douze, and Cordelia Schmid. Prod- Â´ uct quantization for nearest neighbor search. IEEE Trans. Pattern Anal. Mach. Intell., 33(1):117â128, 2011. 2

[24] Biing-Hwang Juang and Augustine H. Gray Jr. Multiple stage vector quantization for speech coding. In IEEE International Conference on Acoustics, Speech, and Signal Processing, ICASSP â82, Paris, France, May 3-5, 1982, pages 597â600. IEEE, 1982. 2

[25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139:1â 139:14, 2023. 1, 2, 3, 4, 5

[26] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 11513â11522. IEEE, 2022. 1, 2, 3

[27] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3D gaussian representation for radiance field. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 21719â21728. IEEE, 2024. 1, 3

[28] Runfa Blark Li, Keito Suzuki, Bang Du, Ki Myung Brian Lee, Nikolay Atanasov, and Truong Nguyen. Splatsdf: Boosting neural implicit sdf via gaussian splatting fusion. arXiv preprint arXiv:2411.15468, 2024. 3

[29] Kunhao Liu, Fangneng Zhan, Muyu Xu, Christian Theobalt, Ling Shao, and Shijian Lu. Stylegaussian: Instant 3D style

transfer with gaussian splatting. In SIGGRAPH Asia 2024 Technical Communications, SA 2024, Tokyo, Japan, December 3-6, 2024, pages 21:1â21:4. ACM, 2024. 3

[30] Xinshuang Liu, Siqi Li, and Yue Gao. Image matting and 3D reconstruction in one loop. International Journal of Computer Vision, pages 1â21, 2025. 2

[31] Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. 5

[32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. 5

[33] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3D gaussians for view-adaptive rendering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 20654â 20664. IEEE, 2024. 3

[34] Julieta Martinez, Holger H Hoos, and James J Little. Stacked quantizers for compositional vector compression. arXiv preprint arXiv:1411.2173, 2014. 2

[35] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3D shapes and textures. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 12663â12673. IEEE, 2023. 2

[36] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM, 65(1):99â106, 2022. 2

[37] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1â 102:15, 2022. 2

[38] K. L. Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi Koohpayegani, and Hamed Pirsiavash. Compgs: Smaller and faster gaussian splatting with vector quantization. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XXXII, pages 330â349. Springer, 2024. 1, 3, 5, 2

[39] Simon Niedermayr, Josef Stumpfegger, and Rudiger West-Â¨ ermann. Compressed 3D gaussian splatting for accelerated novel view synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 10349â10358. IEEE, 2024. 3, 5, 2

[40] Maxime Oquab, Timothee Darcet, Th Â´ eo Moutakanni, Huy V. Â´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J Â´ egou, Â´

Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024, 2024. 2, 4

[41] Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, and Siyu Tang. 3DGS-Avatar: Animatable avatars via deformable 3D gaussian splatting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 5020â 5030. IEEE, 2024. 3

[42] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 14315â14325. IEEE, 2021. 2

[43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015 - 18th International Conference Munich, Germany, October 5 - 9, 2015, Proceedings, Part III, pages 234â241. Springer, 2015. 4

[44] ML Sabin and R Gray. Product code vector quantizers for waveform and voice coding. IEEE transactions on acoustics, speech, and signal processing, 32(3):474â488, 2003. 2

[45] Aaron van den Oord, Oriol Vinyals, and Koray Â¨ Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 6306â6315, 2017. 1, 2, 3, 5

[46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998â6008, 2017. 5

[47] Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and Lingjie Liu. Neus2: Fast learning of neural implicit surfaces for multi-view reconstruction. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 3272â3283. IEEE, 2023. 2

[48] Yufei Wang, Zhihao Li, Lanqing Guo, Wenhan Yang, Alex C. Kot, and Bihan Wen. Contextgs : Compact 3D gaussian splatting with anchor level context model. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. 3

[49] Yuxin Wang, Qianyi Wu, Guofeng Zhang, and Dan Xu. Learning 3D geometry and feature consistent gaussian splatting for object removal. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part III, pages 1â17. Springer, 2024. 3

[50] Yuxuan Wang, Xuanyu Yi, Zike Wu, Na Zhao, Long Chen, and Hanwang Zhang. View-consistent 3D editing with gaus-

sian splatting. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XXXV, pages 404â420. Springer, 2024. 3

[51] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process., 13(4): 600â612, 2004. 5

[52] Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian D. Reid, Philip Torr, and Victor Adrian Prisacariu. Gaussctrl: Multi-view consistent text-driven 3D gaussian splatting editing. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XIV, pages 55â71. Springer, 2024. 3

[53] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf: Point-based neural radiance fields. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 5428â5438. IEEE, 2022. 2

[54] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 586â 595. Computer Vision Foundation / IEEE Computer Society, 2018. 5

[55] Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and Markus H. Gross. Surface splatting. In Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 2001, Los Angeles, California, USA, August 12-17, 2001, pages 371â378. ACM, 2001. 4

# Image-Conditioned 3D Gaussian Splat Quantization

Supplementary Material

## A. Model Architecture Details

This section provides implementation details to facilitate reproduction of our results, including the encoding process to obtain the scene latent and the visibility estimation used for image conditioning.

Encoding process to obtain scene latent. The Methodology section in the main paper describes the overall architecture; here we provide a detailed illustration of how we obtain the scene latent (Figure A.1). The input to the model is a scene represented by $\{ { \mathcal { G } } _ { i } \} _ { i \in [ N ] } .$ . For each Gaussian, we use the attribute scaling factor $s ^ { i }$ , quaternion $r ^ { i }$ opacity $\sigma ^ { i }$ , and spherical harmonics (SH) coefficients $y ^ { i }$ as input. These attributes are individually projected to feature vectors $\boldsymbol { f } _ { s } ^ { i } , \boldsymbol { f } _ { r } ^ { i } , \boldsymbol { f } _ { \sigma } ^ { i }$ , and $\pmb { f } _ { y } ^ { i }$ through linear layers. The Gaussian center serves as the grid position but is not used as a feature input. We exclude Gaussians whose opacity logits are below the threshold $\tau _ { o } = 0$ . The four feature vectors are concatenated and linearly projected to a grid feature vector $\pmb { f } _ { g } \in \mathbb { R } ^ { d _ { g } }$ with $d _ { g } = 3 2 .$ , where $d _ { g } = 3 2$ denotes the grid-level feature dimensionality. To capture local spatial structure, the 3D grid is partitioned into blocks of size $K { \times } K { \times } K$ (with $K { = } 4 )$ , and only non-empty blocks are retained. Within each block, grid features are processed and downsampled by 3D convolutional layers to produce a block feature vector $\pmb { f } _ { b } \in \mathbb { R } ^ { d _ { b } } ( d _ { b } { = } 1 2 8 )$ . Empty grids are padded with a learnable vector. The set of non-empty blocks forms a sparse 3D feature map, which is further processed by a block-level 3D U-Net with learnable padding to produce a sparse latent scene representation.

Visibility estimation. For image conditioning, grid-level image features are obtained by projecting non-empty grid centers onto the image plane(s) and querying the corresponding feature maps. The retrieved features are then averaged per grid cell, weighted by view-dependent visibility. To robustly estimate visibility, we assign to each non-empty grid a virtual sphere centered at the grid position with diameter equal to twice the grid size. For each view, occlusion is computed using this spherical proxy. The visibility of a grid is defined as the visibility of the sphere center for that view. We further propagate visibility at the block level: if any grid within a block is visible from a given view, all grids in that block are marked visible for that view. This strategy reduces sensitivity to local errors in Gaussian shape or opacity.

## B. Training Loss Function Details

In the Method section of the main paper, we described the training objectives and schedules. This section provides additional details of the loss functions that help reproduce our results. The losses include both Gaussian-attribute losses, which provide direct guidance early in training, and image losses, which enhance final visual quality.

Gaussian-attribute loss. To develop a practical model, we train ICGS-Quantizer on scenes containing real-world scanned objects. However, such scans often contain noise and high-frequency textures $( e . g .$ , small printed text), which can introduce noisy artifacts in the optimized Gaussians. In addition to automated data cleaning with a vision-language model (Gemini 2.5 Pro [8]), we further use robust Gaussian-attribute objectives that provide straightforward guidance early in geometry learning. Specifically, the Gaussian-attribute loss $\mathcal { L } _ { G }$ is

$$
\mathcal { L } _ { G } = \mathcal { L } _ { q } + \mathcal { L } _ { s } + \mathcal { L } _ { \sigma } ,\tag{8}
$$

where $\mathcal { L } _ { q }$ and $\mathcal { L } _ { s }$ are Huber losses for quaternions and scales, respectively, and $\mathcal { L } _ { \sigma }$ is a binary cross-entropy (BCE) loss on opacity (from logits). For scales, we apply the Huber objective both in log-space and linear space to balance training stability and accuracy. All terms are weighted equally to minimize the number of hyperparameters and enhance reproducibility. The Huber threshold is set dynamically to the 90th percentile of the current loss values and updated via a moving average.

Image loss function. For image loss $\mathcal { L } _ { I }$ , we adopt the photometric loss from 3DGS [25] and add a mean squared error (MSE) term for training stability:

$$
\mathcal { L } _ { I } = \lambda \mathcal { L } _ { \mathrm { D - S S I M } } + \frac { 1 - \lambda } { 2 } ( \mathcal { L } _ { 1 } + \mathcal { L } _ { 2 } ) ,\tag{9}
$$

with Î»=0.2 as in [25].

Final loss. We supervise the model at three decoding stages: (i) before image conditioning (to preserve the archived scene), (ii) after coarse image conditioning (to adapt to the current scene), and (iii) after fine image conditioning (to further refine adaptation). Auxiliary decoders are used accordingly. This design optimizes the quantized codes for both archival fidelity and post-archival adaptability. Using the original Gaussian attributes as targets, the final Gaussian-attribute loss is:

$$
\mathcal { L } _ { G } ^ { \mathrm { f i n a l } } = \mathcal { L } _ { G } ^ { \mathrm { s c e n e } } + \mathcal { L } _ { G } ^ { \mathrm { c o a r s e f u s i o n } } + \mathcal { L } _ { G } ^ { \mathrm { f i n e f u s i o n } } ,\tag{10}
$$

where $\mathcal { L } _ { G } ^ { \mathrm { s c e n e } }$ uses the output before image conditioning as the prediction, $\mathcal { L } _ { G } ^ { \mathrm { c o } }$ arse fusion uses the output after coarse image conditioning as the prediction, and $\mathcal { L } _ { G } ^ { \mathrm { f i n e f u s i o n } }$ uses the output after fine image conditioning as the prediction. Similarly, the final image loss is:

$$
\mathcal { L } _ { I } ^ { \mathrm { f i n a l } } = \mathcal { L } _ { I } ^ { \mathrm { s c e n e } } + \frac { 1 } { 2 } \left( \mathcal { L } _ { I } ^ { \mathrm { c o a r s e f u s i o n } } + \mathcal { L } _ { I } ^ { \mathrm { f i n e f u s i o n } } \right) ,\tag{11}
$$

<!-- image-->  
Figure A.1. Scene encoding process in ICGS-Quantizer. For clarity, we visualize a toy configuration with 2Ã2Ã2 grids and blocks. For the i-th Gaussian, $s ^ { i } , r ^ { i } , \sigma ^ { i }$ , and $y ^ { i }$ denote the scaling factor, quaternion, opacity, and spherical harmonics (SH) coefficients, respectively. $\textstyle f _ { s } ^ { i } , f _ { r } ^ { i } , f _ { \sigma } ^ { i }$ iÏ , and $\pmb { f } _ { y } ^ { i }$ are their corresponding projected feature vectors.

where ${ \mathcal { L } } _ { I } ^ { \mathrm { s c e n e } }$ uses the output before image conditioning as the prediction and the original sceneâs images as targets, $\mathcal { L } _ { I } ^ { \mathrm { c o a r s e f u s i o n } }$ uses the output after coarse image conditioning as the prediction and the current sceneâs images as targets, and ${ \mathcal { L } } _ { I } ^ { \mathrm { f i n e } }$ e fusion uses the output after fine image conditioning as the prediction and the current sceneâs images as targets. Finally, the total loss is:

$$
\mathcal { L } _ { \mathrm { a l l } } = \mathcal { L } _ { I } ^ { \mathrm { f i n a l } } + w _ { G } \mathcal { L } _ { G } ^ { \mathrm { f i n a l } } + w _ { \mathrm { v q } } \mathcal { L } _ { \mathrm { v q } } ,\tag{12}
$$

where $w _ { G }$ is the weight of the Gaussian-attribute loss, $\mathcal { L } _ { \mathrm { v q } }$ is the commitment loss term [45] for vector quantization and $w _ { \mathrm { v q } }$ is its weight. As detailed in the main paper, during epochs 11â90 we linearly reduce $w _ { G }$ from 0.1 to 0 while increasing $w _ { \mathrm { v q } }$ from 0.01 to 0.1.

## C. Implementation Details

In addition to the planned release of our code, we provide additional implementation details to support the reproducibility of our research.

Codebook learning. We adopt both exponential moving average (EMA) and k-means clustering for codebook learning. EMA updates codebook vectors incrementally at each training step. Every two epochs, k-means clustering is applied to update the codebooks using input vectors computed from the dataset. The codebooks are shared across all training scenes and frozen at test time.

Simulating appearance changes. To evaluate the capability of our model to adapt scenes to their post-archival changes, we consider each scene to have two states: an archived version at time $t _ { 0 }$ and a changed version at time $t > t _ { 0 }$ . We consider scene changes in two aspects: (i) the illumination changes and (ii) the appearance changes. To simulate illumination changes, we randomly sample lighting directions and colors. To simulate appearance changes, we randomly rotate the scene texture colors in RGB space around a unit vector. In our scene updating task, the original scene at time $t _ { 0 }$ serves as the input and the current scene at time t as the target.

Number of conditioning images. Our imageconditioned Gaussian splat quantizer can use an arbitrary number of conditioning images. To make the method both robust in scenarios with limited input views and capable of exploiting multi-view information, we train it with a varying number of conditioning images, ranging from 1 to 6 with equal probability. Our training objectives also include supervision for scene recovery without conditioning images, thereby ensuring that the model remains effective even in the absence of input images.

Regularization of grid-based 3DGS. To promote sparse and accurate 3DGS representations, we penalize Gaussian sizes and encourage larger absolute values of Gaussian opacity logits.

## D. Computation of Storage

This section details how storage is computed for our method and the baselines.

Ours. Our method stores each sparse 3D block as D geometry and D texture codes, where D is the number of codebooks used in residual vector quantization. As indices, each code requires $\lceil \log _ { 2 } N \rceil$ bits of storage, where N is the codebook size. In addition to the codes, we also store the block index of each block, which requires $\lceil \log _ { 2 } ( B ^ { 3 } ) \rceil { = } \lceil 3 \log _ { 2 } B \rceil$ bits, where B is the block resolution (number of blocks per axis). Thus, each block requires storage of $2 D \lceil \log _ { 2 } N \rceil + \lceil 3 \log _ { 2 } B \rceil$ bits. The total storage of one scene is therefore M $( 2 D \lceil \log _ { 2 } N \rceil + \lceil 3 \log _ { 2 } B \rceil )$ , where M is the number of sparse blocks.

Baseline methods. For 3DGS [25], we count all variables as 32-bit floating-point numbers, except for the Gaussian centers, which are stored as indices of the grid cells, resulting in less storage. For C3DGS [39], we use the official implementation to compute storage; for each variable, we automatically set the threshold to keep 50% of them at their original values and approximate the remaining 50% by vector quantization. For CompGS [38], we use storage computation similar to ours, adding the additional storage of their per-scene codebooks.

## E. Limitations

We acknowledge existing limitations of this research. While these aspects lie beyond the primary scope of this work, future improvements could be achieved by exploring additional data compression strategies and optimizing the weighting of loss terms used for training.

Comprehensive compression settings. To store a scene efficiently, we quantize it into discrete codes represented as integers. Each code requires $\lceil \log _ { 2 } N \rceil$ bits when stored as binary data, where N denotes the codebook size. For simplicity and fairness in comparison, we do not apply additional compression techniques to the code data for either our method or the CompGS baseline. However, there remains potential for further improving compression efficiency. One possible enhancement is Run-Length Encoding (RLE) [38], which orders the Gaussians according to one of their attributes and stores the frequency of each code value for that attribute, rather than every individual code. Another potential improvement is Huffman coding [39], which exploits redundancy in the binary representation of the codes by assigning shorter binary sequences to more frequently occurring patterns.

Weight of loss functions. In this work, we introduce loss terms to supervise the learning of each Gaussian attribute. The Gaussian attributes for supervision include: (1) a quaternion $\pmb r \in \mathbb R ^ { 4 }$ representing rotation, (2) a scaling factor $s \in \mathbb { R } ^ { 3 }$ , and (3) an opacity value $\sigma \in \mathbb { R } _ { + }$ . In our experiments, we assign each loss term an equal weight of 1, without performing dataset-specific tuning. This design choice minimizes the number of hyperparameters, thereby enhancing reproducibility. While the overall loss function performs well, there remains potential for further improvement through more effective weighting strategies for the loss terms. One possible direction is to determine optimal weights via hyperparameter searching. Another direction is to automatically balance the loss terms using methods such as Nash Multi-Task Learning.