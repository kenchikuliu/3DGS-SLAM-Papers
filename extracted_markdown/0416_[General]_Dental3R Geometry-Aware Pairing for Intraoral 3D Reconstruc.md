# Dental3R: Geometry-Aware Pairing for Intraoral 3D Reconstruction from Sparse-View Photographs

Yiyi Miao1,2â , Taoyu Wu3,4,â , Tong Chen1,2, Ji Jiang5, Zhe Tang6,

Zhengyong Jiang1, Angelos Stefanidis1, Limin Yu3â¡, Jionglong Su1â¡

1School of AI and Advanced Computing, Xiâan Jiaotong-Liverpool University, China

2School of Electrical Engineering, Electronics and Computer Science, University of Liverpool, United Kingdom

3School of Advanced Technology, Xiâan Jiaotong-Liverpool University, China

4School of Physical Sciences, University of Liverpool, Liverpool, United Kingdom

6Institute of Artificial Intelligence Innovation, Zhejiang University of Technology, China

AbstractâIntraoral 3D reconstruction is fundamental to digital orthodontics, yet conventional methods like intraoral scanning are inaccessible for remote tele-orthodontics, which typically relies on sparse smartphone imagery. While 3D Gaussian Splatting (3DGS) shows promise for novel view synthesis, its application to the standard clinical triad of unposed anterior and bilateral buccal photographs is challenging. The large view baselines, inconsistent illumination, and specular surfaces common in intraoral settings can destabilize simultaneous pose and geometry estimation. Furthermore, sparse-view photometric supervision often induces a frequency bias, leading to over-smoothed reconstructions that lose critical diagnostic details. To address these limitations, we propose Dental3R, a pose-free, graph-guided pipeline for robust, high-fidelity reconstruction from sparse intraoral photographs. Our method first constructs a Geometry-Aware Pairing Strategy (GAPS) to intelligently select a compact subgraph of high-value image pairs. The GAPS focuses on correspondence matching, thereby improving the stability of the geometry initialization and reducing memory usage. Building on the recovered poses and point cloud, we train the 3DGS model with a waveletregularized objective. By enforcing band-limited fidelity using a discrete wavelet transform, our approach preserves fine enamel boundaries and interproximal edges while suppressing highfrequency artifacts. We validate our approach on a large-scale dataset of 950 clinical cases and an additional video-based test set of 195 cases. Experimental results demonstrate that Dental3R effectively handles sparse, unposed inputs and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art methods.

Index TermsâOrthodontics, 3D Reconstruction, Wavelet.

## I. INTRODUCTION

Reliable intraoral 3D reconstruction underpins digital orthodontics: it informs diagnosis, treatment planning, appliance design, and longitudinal outcome assessment. While CBCT and intraoral scanning (IOS) deliver accurate 3D models and occlusal relationships, they require clinic-grade hardware, trained operators, and controlled environments, which limits accessibility for remote follow-ups and increases costs. In contrast, smartphone-assisted image capture has become common in tele-orthodontics; however, the resulting data are typically sparse, unposed, and of varying qualityâmaking robust 3D reconstruction particularly challenging.

Recent progress in neural rendering and feed-forward geometry offers a promising alternative. 3D Gaussian Splatting (3DGS) [1] achieves photorealistic synthesis with efficient differentiable rasterization over explicit Gaussian primitives. To enhance robustness in challenging scenarios, a key trend is to solve for camera poses and scene structure simultaneously, thus bypassing the often brittle and slow SfM pre-processing step. CF-3DGS [2] introduces an end-to-end pipeline for simultaneous camera tracking and novel view synthesis directly from video, eliminating any reliance on SfM pre-processing. DUSt3R [3] regresses dense pointmaps from two unposed views, bypassing SfM initialization. However, directly coupling DUSt3R with 3DGS in intraoral settings exposes three practical failure modes: (i) dense or one-reference star pairing inflates memory/compute when propagating DUSt3R features across many pairs while still under-constraining large baselines; (ii) the clinical triad of frontal plus bilateral buccal images exhibits large view baselines, inconsistent illumination, and specular enamel, destabilizing camera recovery and radiance optimization and leading to drift, floating artifacts, and over-reconstruction under suboptimal pair selection; and (iii) with sparse viewpoints, purely photometric supervision induces frequency bias that smooths enamel edges and interproximal details, producing texture smearing even when geometry is roughly correct.

We present Dental3R, a pose-free, graph-guided pipeline tailored to sparse intraoral photographs. We construct a Hybrid Sequential Geometric View-Graph that arranges images on a cycle and proposes a bounded, multi-scale set of chords as pairing candidates. A range-aware importance model ranks candidate edges based on their expected geometric overlap, and a degree-bounded selection yields a compact subgraph that balances local reliability and global rigidity. Plugging this subgraph into DUSt3R focuses correspondence on highvalue pairs, thereby improving pose stability and reducing memory usage. Building on the DUSt3R-initialized geometry, we train 3DGS with a wavelet-regularized objective. A twolevel discrete wavelet transform enforces band-limited fidelity to preserve fine enamel boundaries and interproximal edges, while suppressing high-frequency artifacts under sparse views. Extensive experimental results demonstrate that Dental3R effectively handles sparse, unposed inputs and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art techniques.

## II. RELATED WORK

## A. Dense Input 3D Representations for Novel View Synthesis

In past years, 3D reconstruction from images has been dominated by classical pipelines combining SfM and Multi-View Stereo (MVS). SfM systems, such as the widely-used COLMAP [4], first recover a sparse 3D point cloud and camera poses by matching local features across multiple views and performing bundle adjustment. Subsequently, MVS algorithms densify this sparse representation by leveraging photometric consistency across views. A paradigm shift occurred with the introduction of Neural Radiance Fields (NeRF) [5], which represents a scene as a continuous 5D function learned by a Multi-Layer Perceptron (MLP). Given a dense set of input images with known camera poses, NeRF can produce high-fidelity new viewpoints, albeit with time-consuming per-scene training and rendering. However, the standard NeRF pipeline assumes rigid scenes and ample multi-view coverage, which are often violated in orthodontics, where anatomy is deformable and camera motion is limited. More recent approaches, such as 3D Gaussian Splatting(3DGS) [1], [6], address NeRFâs efficiency limitations by representing scenes as explicit 3D Gaussian primitives. It initializes a set of 3D Gaussian splats from sparse SfM points and optimizes their color, opacity, and anisotropic covariance, thereby preserving the fidelity of volumetric fields without incurring heavy neural network inference in empty space. The result is dramatically faster training and rendering, making dense novel view synthesis more practical for interactive applications.

## B. Camera Pose-Free Reconstruction

Another key trend has been lifting the requirement of known camera poses. Traditional pipelines run Structure-from-Motion (e.g. COLMAP) to estimate poses before reconstruction, but this can be slow or brittle for challenging scenes (e.g. texturepoor surgical images). Recent methods instead solve for camera poses simultaneously with scene reconstruction. These methods aim to jointly optimize the scene representation and camera parameters. NoPe-NeRF [7] is a seminal method for optimizing Neural Radiance Fields without known camera poses. By leveraging monocular depth priors and introducing novel consistency losses, it jointly learns the scene representation and a consistent camera trajectory. Compared to prior joint optimization methods (e.g. BARF [8], SC-NeRF [9]) limited to forward-facing scenes, NoPe-NeRF can handle casual handheld videos, yielding both accurate novel view rendering and improved camera pose estimates. In parallel, researchers have integrated pose estimation into the 3DGS framework. CF-3DGS [2] introduces an end-to-end pipeline for simultaneous camera tracking and novel view synthesis directly from video, eliminating any reliance on SfM pre-processing. The method progressively builds a global 3D Gaussian representation while estimating camera poses by registering this evolving model against each new incoming frame. By eliminating reliance on COLMAP, CF-3DGS, and NoPe-NeRF, 3D reconstruction becomes more robust and accessible, a direction that is especially pertinent where calibrations or fiducial markers might otherwise be needed. Indeed, the push towards self-contained (pose-free) 3D vision is now informing surgical applications as well, as discussed next.

## C. Sparse Input Novel View Synthesis

Capturing dozens of images per scene is impractical in many settings, motivating NVS approaches that work with sparse inputs [10]. MVSNeRF [11] pioneered fast generalizable radiance field reconstruction from as few as three views. By leveraging plane-sweep cost volumes (a multi-view stereo technique) to provide geometry cues, MVSNeRFâs network can infer a neural radiance field from just a handful of images. It achieves realistic novel views using minimal inputs and can be fine-tuned with additional images for higher quality, a significant speedup over the original NeRF, which required hours of per-scene optimization. More recently, MVSplat [12] introduces a feed-forward method that reconstructs a scene as a set of 3D Gaussians from a few wide-baseline images, using a plane-sweep cost volume to establish initial geometry in a single pass. Trained end-to-end with only photometric supervision, this approach achieves state-of-the-art novel view quality with significantly greater parameter efficiency, faster inference speed, and superior generalization compared to previous methods.

## D. End-to-End Reconstruction from Unposed Images

DUSt3R [3] introduced the idea of regressing dense 3D structure and camera poses directly from images in a feedforward manner. DUSt3R predicts a pair of pointmaps given two input images, effectively solving pairwise Structure-from-Motion via neural regression. Building on this, MUSt3R [13] extended the architecture to handle multi-view input by introducing a symmetric design and a latent memory that maintains a global frame of reference. In parallel, VGGT [14] introduces a versatile Transformer-based model that directly infers a comprehensive suite of 3D attributes, including camera parameters and scene geometry, from a widely variable number of input views. This approach achieves state-of-the-art results on multiple geometric tasks within a single forward pass, surpassing traditional pipelines by eliminating the need for iterative post-optimization, such as bundle adjustment. Along the same lines, MapAnything [15] is a universal feed-forward 3D reconstruction model that unifies diverse tasks within one transformer-based architecture. MapAnything accepts flexible inputs â ranging from one or more images to additional cues such as intrinsics or partial reconstructions â and directly outputs the metric 3D scene structure and camera parameters.

<!-- image-->  
Fig. 1. Overview of Dental3R. Given a set of sparse and unposed input images, we first employ our GAPS strategy to generate image pairs. Subsequently, we leverage a stereo-dense reconstruction model to regress a dense point cloud in a global coordinate system, while concurrently obtaining the corresponding relative camera poses. The resulting point cloud is then used to initialize the 3D Gaussians. During the optimization process, we incorporate wavelet constraints to ensure geometric consistency and frequency details.

## III. METHODOLOGY

## A. Preliminary

1) 3D Gaussian Splatting.: 3DGS [1] is a high-fidelity radiance field method that represents a 3D scene as an explicit collection of anisotropic Gaussian primitives. Each Gaussian primitive $G _ { i }$ is defined by a set of optimizable attributes: a center position $\pmb { \mu } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ , an opacity $o _ { i } ~ \in ~ [ 0 , 1 ]$ , a set of Spherical Harmonic (SH) coefficients for modeling viewdependent color, and a $3 \times 3$ covariance matrix $\Sigma _ { i }$ . The spatial density of each Gaussian is given by:

$$
G _ { i } ( \mathbf { X } ) = \exp \left\{ - \frac { 1 } { 2 } ( \mathbf { X } - \pmb { \mu } _ { i } ) ^ { \top } \pmb { \Sigma } _ { i } ^ { - 1 } ( \mathbf { X } - \pmb { \mu } _ { i } ) \right\} ,\tag{1}
$$

where $\textbf { X } \in \ \mathbb { R } ^ { 3 }$ denotes a point in 3D space. To ensure the covariance matrix remains positive semi-definite and to facilitate efficient optimization, $\Sigma _ { i }$ is parameterized by a 3D scaling vector $\mathbf { s } _ { i }$ and a rotation quaternion $\mathbf { q } _ { i }$

The final color ${ \hat { C } } ( \mathbf { p } )$ and depth DË (p) for each pixel p are then synthesized by alpha-blending the contributions of all overlapping Gaussians, which are sorted from front to back along the camera ray. This volumetric rendering process is formulated as:

$$
\hat { C } ( \mathbf { p } ) = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , \quad \hat { D } ( \mathbf { p } ) = \sum _ { i \in \mathcal { N } } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $\mathcal { N }$ is the ordered set of Gaussians that along the pixelâs viewing ray. For each Gaussian $i , c _ { i }$ is the color derived from its SH coefficients based on the current viewing direction, and $d _ { i }$ is its depth, corresponding to the z-coordinate of its center $\pmb { \mu } _ { i }$ in camera coordinates.

2) DUSt3R: DUSt3R [3] introduces a novel paradigm for dense 3D reconstruction from unconstrained image collections, operating without prior knowledge of camera intrinsic or extrinsic parameters. The framework circumvents traditional Structure-from-Motion (SfM) pipelines by directly regressing a dense scene representation called a pointmap. A pointmap, denoted as $\mathbf { X } \in \dot { \mathbb { R } } ^ { W \times H \times 3 } ,$ , establishes a direct mapping from the pixel coordinates $( u , v )$ of an image I to 3D points in a specific coordinate system. Given a depth map D and camera intrinsics K, the pointmap is defined as:

$$
\begin{array} { r } { \mathbf { X } _ { u , v } = D _ { u , v } \cdot \mathbf { K } ^ { - 1 } \left[ u , v , 1 \right] ^ { T } , } \end{array}\tag{3}
$$

where X is expressed in the cameraâs local coordinate frame. The core network architecture takes a pair of images, ${ \bf { I } } _ { 1 }$ and I2, and outputs two corresponding pointmaps, $\mathbf { X } ^ { 1 , \bar { 1 } }$ and $\mathbf { X } ^ { 2 , 1 }$ , that are implicitly aligned to a common reference frame.

The model is trained using a 3D regression loss designed to minimize the Euclidean distance between predicted and ground-truth pointmaps. To address the inherent scale ambiguity in uncalibrated stereo reconstruction, both the predictions and the ground truth are normalized. The normalized regression loss for a pixel i in view v is formulated as:

$$
\mathcal { L } _ { \mathrm { r e g } } ( v , i ) = \left. \frac { 1 } { s } \mathbf { X } _ { i } ^ { v , 1 } - \frac { 1 } { \bar { s } } \bar { \mathbf { X } } _ { i } ^ { v , 1 } \right. ,\tag{4}
$$

where X and XÂ¯ are the predicted and ground-truth pointmaps, respectively. The scale factors, s and sÂ¯, are computed as the mean distance of all valid 3D points from the origin, ensuring scale-invariant comparison.

Furthermore, to handle regions that are inherently difficult to reconstruct, such as textureless surfaces, sky, or transparent objects, DUSt3R employs a confidence-aware training objective. The network jointly predicts a per-pixel confidence map C, which modulates the regression loss. The final confidenceaware loss function is:

$$
\mathcal { L } _ { \mathrm { c o n f } } = \sum _ { v \in \{ 1 , 2 \} } \sum _ { i \in \mathcal { D } ^ { v } } \left( C _ { i } ^ { v , 1 } \mathcal { L } _ { \mathrm { r e g } } ( v , i ) - \alpha \log C _ { i } ^ { v , 1 } \right) ,\tag{5}
$$

where $\mathcal { D } ^ { v }$ is the set of valid pixels in view $v ,$ and the logarithmic term regularizes the confidence prediction . This objective enhances robustness against geometric ambiguities and yields a per-pixel confidence that is valuable for downstream tasks.

6 views  
9 views  
<!-- image-->  
Fig. 2. Novel View Synthesis Comparisons with 6 views and 9 views input. We qualitatively compare the quality of novel view synthesis with 3DGS [1], CF-3DGS [2], and InstantSplat [16], and show that our method achieves better quality and more accurate texture details.

TABLE I  
QUANTITATIVE EVALUATION ON THE VIDEO TEST SET, USING DIFFERENT INPUT VIEWPOINTS. THE BEST RESULTS ARE BOLDED.
<table><tr><td rowspan="2">Algorithm</td><td colspan="3">3 Training views</td><td colspan="3">6 Training views</td><td colspan="3">9 Training views</td><td colspan="3">12 Training views</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3DGS [1]</td><td></td><td></td><td></td><td></td><td></td><td>â</td><td></td><td></td><td></td><td>11.51</td><td>0.53</td><td>0.574</td></tr><tr><td>InstantSplat [16]</td><td>23.81</td><td>0.826</td><td>0.304</td><td>27.01</td><td>0.863</td><td>0.268</td><td>28.66</td><td>0.890</td><td>0.241</td><td>29.32</td><td>0.898</td><td>0.235</td></tr><tr><td>CF-3DGS [2]</td><td>15.32</td><td>0.748</td><td>0.443</td><td>18.01</td><td>0.795</td><td>0.277</td><td>21.29</td><td>0.812</td><td>0.374</td><td>23.02</td><td>0.853</td><td>0.337</td></tr><tr><td>Ours</td><td>23.87</td><td>0.824</td><td>0.300</td><td>27.97</td><td>0.869</td><td>0.251</td><td>29.161</td><td>0.892</td><td>0.237</td><td>29.947</td><td>0.884</td><td>0.221</td></tr></table>

## B. Hybrid Sequential Geometric Scene Graph

a) Motivation: In multi-view 3D reconstruction, the selection of image pairs is a critical determinant of both computational efficiency and model accuracy. Dense or complete graphs (pairing every view) are computationally prohibitive for large datasets and, when used with modern Transformer-based priors such as DUSt3R, consume large amounts of memory and require high-performance GPUs, making them unsuitable for practical medical scenarios. Conversely, overly sparse graphs or suboptimal strategies like the oneref method from DUST3R may lack sufficient geometric constraints, leading to inaccurate camera poses and a degradation in the final reconstruction quality.

Therefore, to address these limitations, we propose the Geometry-Aware Pairing Strategy (GAPS) to construct a sparse measurement graph that balances local reliability and global rigidity. GAPS operates on a cycle of views, defining multi-scale neighborhoods to efficiently enforce both robust local and global constraints, thereby ensuring accurate modeling. We formalize this by letting the view indices form the vertex set $V = \{ 0 , 1 , \ldots , n - 1 \}$ with $n \in \mathbb { N } ,$ , arranged on the cycle graph $C _ { n } . \mathrm { ~ A ~ }$ small, fixed set of positive offsets K encodes a multi-scale neighborhood on $C _ { n }$ . Throughout, $\{ i , j \}$ denotes an undirected edge between vertices $i , j \in V$ , and all index arithmetic is taken modulo n.

b) Sequential Prior and Candidate Chords: Multi-scale sequential structure proposes a bounded set of chords as candidates:

$$
E _ { \mathrm { c a n d } } = \bigcup _ { k \in { \mathcal { K } } } \{ \{ i , ( i + k ) { \bmod { n } } \} : i \in V \} ,\tag{6}
$$

where $E _ { \mathrm { c a n d } }$ is the candidate edge set obtained by uniting the k-ring chords on $C _ { n }$ across $k \in \mathcal { K }$

c) Geometric Overlap and Edge Importance: Expected image overlap is modeled by a wrap-around index distance on the cycle:

$$
d ( i , j ) = \mathrm { m i n } \big ( | i - j | , n - | i - j | \big ) ,\tag{7}
$$

where d : $V \times V  \{ 0 , 1 , \ldots , \lfloor n / 2 \rfloor \}$ is the circular distance on $C _ { n }$ and âÂ·â denotes the floor operator.

A monotone decay $\phi$ maps this distance to a surrogate overlap, which is converted to a range-adaptive importance:

$$
w ( i , j ) = \alpha _ { r ( i , j ) } \phi \bigl ( d ( i , j ) \bigr ) \ + \ \beta _ { r ( i , j ) } ,\tag{8}
$$

where $w : E _ { \mathrm { c a n d } } \to \mathbb { R } _ { \geq 0 }$ is the edge-importance function, $\phi : [ 0 , \lfloor n / 2 \rfloor ] \to [ 0 , 1 ]$ is a fixed monotone decay, $r ( i , j ) \in$ {local, medium, long} is a coarse range class derived from $d ( i , j )$ , and $\alpha . , \beta . \in \mathbb { R }$ are range-dependent constants chosen once. Edges with low importance are discarded, yielding a weighted candidate graph.

d) Degree-Bounded High-Weight Subgraph: To control complexity and prevent hubs while favoring informative connections, we seek a degree-constrained, high-weight subgraph:

$$
\operatorname* { m a x } _ { E ^ { \prime } \subseteq E _ { \mathrm { c a n d ~ } } } \sum _ { \{ i , j \} \in E ^ { \prime } } w ( i , j ) \quad \mathrm { s . t . } \quad \deg _ { E ^ { \prime } } ( v ) \leq b \forall v \in V ,\tag{9}
$$

where $E ^ { \prime }$ is the selected edge set, $\deg _ { E ^ { \prime } } ( v )$ is the degree of vertex v in $( V , E ^ { \prime } )$ , and $b \in \mathbb { N }$ is a uniform degree cap.

By adopting the proposed pairing strategy in the initial stage of Dust3R [3], we effectively reduced memory usage while maintaining training results.

## C. Wavelet Decomposition

Wavelet analysis provides a mathematical framework for the multi-resolution representation of signals. Its principal advantage over Fourier analysis is time-frequency localization [17]. Unlike the Fourier transform, which utilizes globally supported sinusoidal basis functions, the wavelet transform employs basis functions that are localized in both the time and frequency domains. For two-dimensional signals, such as images, this property enables a decomposition into sub-bands of varying spatial frequency and orientation, while preserving crucial spatial information.

The two-dimensional Discrete Wavelet Transform (DWT) is applied to a discrete image, represented as a matrix $I \in$ $\mathbb { R } ^ { H \times W }$ . The transform utilizes a pair of Quadrature Mirror Filters (QMFs), a low-pass filter h and a high-pass filter $^ { g , }$ which are applied separably to the imageâs rows and columns. Each filtering stage is followed by dyadic downsampling. This process yields four coefficient sub-bands:

$$
\begin{array} { r } { L L = \left( I * _ { \mathrm { r o w } } g \right) * _ { \mathrm { c o l } } g } \\ { L H = \left( I * _ { \mathrm { r o w } } g \right) * _ { \mathrm { c o l } } h } \\ { H L = \left( I * _ { \mathrm { r o w } } h \right) * _ { \mathrm { c o l } } g } \\ { H H = \left( I * _ { \mathrm { r o w } } h \right) * _ { \mathrm { c o l } } h } \end{array}\tag{10}
$$

Here, $* _ { \mathrm { r o w } }$ and $^ * \mathrm { c o l }$ denote one-dimensional convolution along the respective image dimensions. The resulting coefficient maps represent the approximation (LL), horizontal details (LH), vertical details (HL), and diagonal details (HH).

Our framework incorporates a wavelet loss term to regularize the optimization process in the frequency domain, based on a two-level DWT. First, we define the residual map $\Delta _ { x }$ for each frequency component x as the difference between the ground truth image $I _ { \mathrm { g t } }$ and the rendered output $\hat { I } { : }$

$$
\Delta _ { x } = W _ { x } ( I _ { \mathrm { g t } } ) - W _ { x } ( \hat { I } ) ,\tag{11}
$$

where $W _ { x }$ denotes the operation that extracts the sub-band $x \in$ $\{ L L , L H , H L , H H \}$ . The total wavelet loss is then computed as the weighted sum of the squared $L _ { 2 }$ norms of these residual maps:

$$
\mathcal { L } _ { \mathrm { w a v e l e t } } = \sum _ { x \in \{ L L , L H , H L , H H \} } \lambda _ { x } \left. \Delta _ { x } \right. _ { 2 } ^ { 2 } .\tag{12}
$$

The weights $\lambda _ { x }$ control the influence of each frequency component on the wavelet loss.

## IV. EXPERIMENT

## A. Implementation Details

1) Dataset Description: To rigorously assess accuracy and robustness, we assembled a clinical intraoral dataset in partnership with specialist dental hospitals. All imagery was acquired by certified orthodontists using a Canon EOS 700D with a 100 mm macro lens under forced-flash settings, ensuring consistent illumination and minimizing lighting variability. The dataset comprises two subsets. (i) Video subset: 195 clinical cases, each a short intraoral sequence spanning a continuous sweep from right buccal to frontal occlusion to left buccal. From every sequence we uniformly sampled 24 frames, then split them evenly into 12 training and 12 test images. Training was performed using only the training images and their corresponding camera poses. We evaluated four sparse-view regimes with 3, 6, 9, and 12 input views. After optimization, novel views were rendered at the test poses to assess the quality of synthesis. (ii) Three-image subset: 950 clinical cases, each containing exactly three photographs (anterior occlusal, left buccal, right buccal), used to probe reconstruction and novel-view synthesis under extremely sparse observations.

2) Experimental Setup: All experiments were conducted on a desktop workstation equipped with an Intel Core i9-13900KF processor and an NVIDIA GeForce RTX 4090 GPU. To ensure a fair and consistent comparison across all cases, a uniform set of hyperparameters was applied throughout the experiments. For the optimization of the 3D Gaussian attributes, we adhered to the default training parameters established in the original Gaussian Splatting implementation [1]. The parameters were updated using the Adam optimizer [19]. To balance rendering efficiency and quality, we set the number of training iterations to 3000.

## B. Evaluation results

1) Comparative Experiments on Video Test Dataset: To assess our frameworkâs novel view synthesis capabilities, we benchmarked it against the original 3DGS and other state-ofthe-art methods, including CF-3DGS and InstantSplat, using our comprehensive video dataset. The quantitative results, summarized in Table I. Averaged across 195 distinct cases, our method consistently outperforms all baselines across the three standard metrics: Peak Signal-to-Noise Ratio (PSNR) [20], Structural Similarity Index Measure (SSIM) [21], and Learned Perceptual Image Patch Similarity (LPIPS) [22].

Notably, the performance of standard 3DGS is of particular interest. The model fails to converge entirely when initialized with sparse multi-view inputs, as indicated by the â-â entries in the table. Even in the more data-rich 12-view setting, where it is trainable, its rendering quality remains substantially inferior. This behavior highlights a fundamental limitation of conventional approaches in sparse-observation scenarios, which are typical of dental imaging.

TABLE II  
ABLATION STUDY AND PAIRING-STRATEGY COMPARISON ON THE VIDEO TEST SET. BEST IS IN BOLD, THE SECOND BEST IS UNDERLINED.
<table><tr><td rowspan="2">Method</td><td colspan="5">3 Training views</td><td colspan="5">6 Training views</td></tr><tr><td>Pairs</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>GPU Memory (MB)â</td><td>Pairs</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>GPU Memory (MB)â</td></tr><tr><td>Cosine [18]</td><td>2</td><td>0.762</td><td>20.20</td><td>0.401</td><td>3746</td><td>10</td><td>0.809</td><td>22.817</td><td>0.360</td><td>2984</td></tr><tr><td>Complete [16]</td><td>6</td><td>0.834</td><td>25.65</td><td>0.286</td><td>7848</td><td>30</td><td>0.873</td><td>28.77</td><td>0.247</td><td>10158</td></tr><tr><td>Oneref [16]</td><td>4</td><td>0.790</td><td>24.01</td><td>0.340</td><td>3542</td><td>10</td><td>0.839</td><td>26.72</td><td>0.295</td><td>4710</td></tr><tr><td>Ours w.o. Wavelet</td><td>6</td><td>0.819</td><td>24.41</td><td>0.324</td><td>2414</td><td>9</td><td>0.857</td><td>28.27</td><td>0.264</td><td>6479</td></tr><tr><td>Ours</td><td>6</td><td>0.820</td><td>24.55</td><td>0.323</td><td>2744</td><td>9</td><td>0.860</td><td>28.35</td><td>0.264</td><td>6420</td></tr><tr><td rowspan="3">Method</td><td></td><td colspan="3">9 Training views</td><td></td><td colspan="4">12 Training views</td></tr><tr><td>Pairs SSIMâ</td><td></td><td>PSNRâ</td><td>LPIPSâ</td><td>GPU Memory (MB)â</td><td>Pairs</td><td>SSIMâ</td><td>PSNRâ LPIPSâ</td><td>GPU Memory (MB)â</td></tr><tr><td>16</td><td>0.819</td><td>23.17</td><td>0.324</td><td>4144</td><td>22</td><td>0.871</td><td>25.46</td><td>0.275</td><td>4320</td></tr><tr><td>Cosine [18] Complete [16]</td><td>72</td><td>0.893</td><td>29.26</td><td>0.229</td><td>12888</td><td>132</td><td>0.904</td><td>29.76</td><td>0.224</td><td>16038</td></tr><tr><td>Oneref [16]</td><td>16</td><td>0.871</td><td>28.27</td><td>0.258</td><td>4880</td><td>22</td><td>0.886</td><td>29.27</td><td>0.241</td><td>5068</td></tr><tr><td>Ours w.o. Wavelet</td><td>27</td><td>0.892</td><td>29.61</td><td>0.232</td><td>7427</td><td>38</td><td>0.899</td><td>29.81</td><td>0.227</td><td>7664</td></tr><tr><td>Ours</td><td>27</td><td>0.894</td><td>29.62</td><td>0.231</td><td>7636</td><td>38</td><td>0.900</td><td>29.84</td><td>0.221</td><td>7623</td></tr></table>

<!-- image-->  
Fig. 3. Novel View Synthesis Results with Different Pair Strategy. We perform a qualitative comparison against the complete and oneref graph strategies from InstantSplat [16], as well as the cosine graph strategy from EasySplat [18]. The results demonstrate that our proposed GAPS strategy achieves novel view synthesis performance competitive with the exhaustive complete strategy. Furthermore, GAPS surpasses the other two sparse methods (oneref and cosine), yielding superior rendering quality and more accurate textural details across various input views.

Figure 2 illustrates the qualitative evaluations of novel view synthesis. Under the challenging 6-view condition, outputs from CF-3DGS are plagued by noticeable blurring and floating artifacts, while InstantSplat exhibits pronounced geometric distortions in the lower teeth. Although increasing the inputs to 9 views allows CF-3DGS to mitigate major artifacts, the resulting images are still of low resolution and suffer from clear overfitting, such as hallucinated geometry in the right buccal area. InstantSplat also struggles with over-reconstruction in the right molars, leading to texture and shape degradation. Such artifacts could critically undermine the reliability of clinical assessments, particularly in applications such as remote orthodontic monitoring. In stark contrast, our reconstructions maintain exceptional geometric fidelity and remain artifactfree across both input conditions, demonstrating robust generalization for high-quality novel view synthesis.

TABLE III  
QUANTITATIVE RESULTS WITH THE OTHER METHODS USING 3 VIEWS.
<table><tr><td>Methods</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Times (Seconds)â</td></tr><tr><td>3DGS [1]</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InstantSplat [16]</td><td>32.78</td><td>0.945</td><td>0.160</td><td>57</td></tr><tr><td>CF-3DGS [2]</td><td>18.37</td><td>0.803</td><td>0.32</td><td>372</td></tr><tr><td>Ours</td><td>33.89</td><td>0.949</td><td>0.137</td><td>64</td></tr></table>

2) Comparative Experiments on 3 Views images Dataset: The frameworkâs robustness was further challenged under extremely sparse input conditions using a large-scale image dataset of 950 clinical cases. For this experiment, occlusal reconstruction was performed from a minimal set of three intra-oral images: the anterior, left buccal, and right buccal views. Given the absence of novel test views in this dataset, our quantitative evaluation in Table III assesses reconstruction fidelity on the training views themselves, averaged over 956 cases.

## C. Ablation Study

To validate the contribution of each key component in our framework, we conduct an ablation study on the proposed graph strategy GAPS and the wavelet constraint, with results summarized in Table II. We compare the Complete and oneref graph strategy from InstantSplat [16] and the cosine graph strategy from EasySplat [18], as shown in the Figure 3. The Ours w.o. Wavelet shows the results of our graph strategy with the wavelet constraint removed. The analysis first highlights the efficacy of our graph strategy. We observe that it produces image pairs that are more amenable to optimization, which is crucial for achieving high-quality novel view synthesis. Moreover, this strategy achieves significantly lower GPU memory usage than the Complete graph strategy baseline in InstantSplat [16], while maintaining comparable performance. The quantitative results in the table, averaged over 20 randomly sampled cases from the video test dataset. The best result is bolded, and the second result is underlined, showing the effectiveness and efficiency of our work.

## V. CONCLUSION

In this paper, we introduced Dental3R, a novel, graphguided pipeline designed for high-fidelity 3D reconstruction of dental occlusion from sparse and unposed intraoral photographs. We addressed the challenges in the tele-orthodontic scene: unstable camera pose estimation arising from large view baselines and the loss of fine diagnostic details due to photometric supervision under sparse views. Our proposed

GAPS strategy stabilizes the reconstruction process by intelligently selecting optimal image pairs for a robust and efficient geometry initialization. Furthermore, our wavelet-regularized objective for 3DGS training effectively counteracts frequency bias, preserving the sharp enamel edges and interproximal details that are essential for clinical assessment. Extensive experiments on our large-scale clinical dataset demonstrate that Dental3R significantly outperforms state-of-the-art models in terms of novel view synthesis quality.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[2] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, âColmapfree 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 796â20 805.

[3] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, âDust3r: Geometric 3d vision made easy,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 697â20 709.

[4] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4104â4113.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[6] T. Wu, Y. Miao, Z. Li, H. Zhao, K. Dang, J. Su, L. Yu, and H. Li, âEndoflow-slam: Real-time endoscopic slam with flow-constrained gaussian splatting,â arXiv preprint arXiv:2506.21420, 2025.

[7] W. Bian, Z. Wang, K. Li, J.-W. Bian, and V. A. Prisacariu, âNope-nerf: Optimising neural radiance field with no pose prior,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4160â4169.

[8] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, âBarf: Bundle-adjusting neural radiance fields,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5741â5751.

[9] L. Song, G. Wang, J. Liu, Z. Fu, Y. Miao et al., âSc-nerf: Selfcorrecting neural radiance field with sparse views,â arXiv preprint arXiv:2309.05028, 2023.

[10] Y. Miao, T. Wu, T. Chen, S. Li, J. Jiang, Y. Yang, A. Stefanidis, L. Yu, and J. Su, âDentalsplatdental occlusion novel view synthesis from sparse intra-oral photographs,â arXiv preprint arXiv:2511.03099, 2025.

[11] A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su, âMvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 14 124â14 133.

[12] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.- J. Cham, and J. Cai, âMvsplat: Efficient 3d gaussian splatting from sparse multi-view images,â in European Conference on Computer Vision. Springer, 2024, pp. 370â386.

[13] Y. Cabon, L. Stoffl, L. Antsfeld, G. Csurka, B. Chidlovskii, J. Revaud, and V. Leroy, âMust3r: Multi-view network for stereo 3d reconstruction,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1050â1060.

[14] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, âVggt: Visual geometry grounded transformer,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294â 5306.

[15] N. Keetha, N. Muller, J. Sch Â¨ onberger, L. Porzi, Y. Zhang, T. Fischer, Â¨ A. Knapitsch, D. Zauss, E. Weber, N. Antunes et al., âMapanything: Universal feed-forward metric 3d reconstruction,â arXiv preprint arXiv:2509.13414, 2025.

[16] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos et al., âInstantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds,â arXiv preprint arXiv:2403.20309, vol. 2, no. 3, p. 4, 2024.

[17] T. Wu, Y. Miao, J. Guo, Z. Chen, S. Zhao, Z. Li, Z. Tang, B. Huang, and L. Yu, âEndowave: Rational-wavelet 4d gaussian splatting for endoscopic reconstruction,â arXiv preprint arXiv:2510.23087, 2025.

[18] A. Gao, L. Guo, T. Chen, Z. Wang, Y. Tai, J. Yang, and Z. Zhang, âEasysplat: View-adaptive learning makes 3d gaussian splatting easy,â arXiv preprint arXiv:2501.01003, 2025.

[19] D. P. Kingma, âAdam: A method for stochastic optimization,â arXiv preprint arXiv:1412.6980, 2014.

[20] A. Hore and D. Ziou, âImage quality metrics: Psnr vs. ssim,â in 2010 20th international conference on pattern recognition. IEEE, 2010, pp. 2366â2369.

[21] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE transactions on image processing, vol. 13, no. 4, pp. 600â612, 2004.

[22] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.