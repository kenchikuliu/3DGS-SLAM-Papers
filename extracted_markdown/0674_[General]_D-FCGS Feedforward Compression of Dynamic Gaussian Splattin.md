# D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos

Wenkang Zhang1, 2, Yan Zhao1, Qiang Wang3, Zhixin Xu1, Li Song1, Zhengxue Cheng1\*

1Shanghai Jiao Tong University

2Zhiyuan College, Shanghai Jiao Tong University

3Visionstar Information Technology (Shanghai) Co., Ltd

{conquer.wkzhang, zhaoyanzy, zhixin.xu, song li, zxcheng}@sjtu.edu.cn, wq@sightp.com

## Abstract

Free-Viewpoint Video (FVV) enables immersive 3D experiences, but efficient compression of dynamic 3D representation remains a major challenge. Existing dynamic 3D Gaussian Splatting methods couple reconstruction with optimization-dependent compression and customized motion formats, limiting generalization and standardization. To address this, we propose D-FCGS, a novel Feedforward Compression framework for Dynamic Gaussian Splatting. Key innovations include: (1) a standardized Group-of-Frames (GoF) structure with I-P coding, leveraging sparse control points to extract inter-frame motion tensors; (2) a dual prior-aware entropy model that fuses hyperprior and spatialtemporal priors for accurate rate estimation; (3) a controlpoint-guided motion compensation mechanism and refinement network to enhance view-consistent fidelity. Trained on Gaussian frames derived from multi-view videos, D-FCGS generalizes across diverse scenes in a zero-shot fashion. Experiments show that it matches the rate-distortion performance of optimization-based methods, achieving over 17Ã compression compared to the baseline while preserving visual quality across viewpoints. This work advances feedforward compression of dynamic 3DGS, facilitating scalable FVV transmission and storage for immersive applications.

Code â https://github.com/Mr-Zwkid/D-FCGS

## 1 Introduction

Our world is inherently dynamic, with 3D scenes evolving over time and observable from arbitrary viewpoints. Capturing and representing such 4D dynamics has long been a fundamental challenge in computer vision and graphics. Freeviewpoint video (FVV), which enables immersive 6-DoF experiences, has emerged as a promising solution with applications in virtual reality, telepresence, and remote education. However, realizing practical FVV systems demands efficient solutions for reconstruction, compression, transmission, and rendering. In this work, we focus on the critical problem of compression for dynamic 3D scenes.

Recent advances of 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) have revolutionized 3D scene representation, offering unparalleled rendering quality and real-time performance. Extending to 4D, dynamic forms of 3DGS (Yang et al. 2024; Wu et al. 2024a; Li et al. 2024; Yang et al. 2023; Sun et al. 2024) gradually garner attention. Analogous to the temporal expansion of images facilitated by videos, frameby-frame 3D Gaussians can naturally serve as a temporal expansion of 3DGS, forming the basis for FVV. Based on this per-frame idea, 3DGStream (Sun et al. 2024) pioneers dynamic scene reconstruction via on-the-fly training, while subsequent works improve compression efficiency through techniques like rate-aware training (Hu et al. 2025), vectorquantized residuals (Girish et al. 2024), and static-dynamic decomposition (Wu et al. 2025).

<!-- image-->  
Figure 1: Left: Illustration of differences between previous optimization-based methods and our D-FCGS. Right: R-D curve and storage size comparison on N3V dataset.

Despite these advances, existing methods remain constrained by their coupled optimization of reconstruction and compression. By representing frame-wise Gaussian motions through specialized formats (e.g., neural networks), these approaches require per-scene optimization and meticulous hyperparameter tuning. This optimization-dependent paradigm creates two critical barriers to practical deployment: (1) it severely limits generalization to unseen scenes; (2) it hinders the development of standardized compression schemes that could enable widespread adoption of FVV.

To address these challenges, we propose D-FCGS, a novel feedforward compression framework for dynamic Gaussian Splatting. Our key insight is that temporal coherence in Gaussian point clouds can be efficiently modeled via the Group-of-Frames (GoF) representation, where inter-frame motions can be compressed in a feedforward and scene-agnostic manner. Specifically, we adopt the standard GS format as input and leverage sparse control points to efficiently extract motion tensors, optimal for both compression efficiency and computational performance. These motion tensors are then processed through our feedforward motion compression pipeline, which incorporates a tailored dual prior-aware entropy model to enhance probability estimation accuracy. Following decompression, the sparse motions are propagated across the entire Gaussian frame under the guidance of the control points. Finally, lightweight color refinement is applied to improve view-consistent fidelity.

Our D-FCGS is trained end-to-end on frame-wise GS sequences constructed from both real-world and synthetic multi-view video datasets. Once trained, it serves as a general-purpose inter-frame compression codec, requiring no scene-specific optimization or access to multi-view images during zero-shot inference. Extensive experiments show that D-FCGS exhibits strong generalization across diverse dynamic scenes and achieves state-of-the-art ratedistortion performance, surpassing 17Ã compression over 3DGStream while maintaining comparable fidelity.

Our contributions can be summarized as follows:

â¢ We present D-FCGS, a novel feedforward compression framework for dynamic 3DGS that enables zero-shot inter-frame coding of Gaussian sequences.

â¢ We propose a sparse motion representation with control points for I-P coding, coupled with a dual prior-aware entropy model for efficient compression, and develop a decoder with motion compensation and refinement for enhanced fidelity.

â¢ Experiments across various scenes show the effectiveness and robustness of D-FCGS, achieving over 17Ã compression over 3DGStream while preserving fidelity, outperforming most optimization-based methods.

## 2 Related Work

## 2.1 3D Gaussian Splatting Compression

High storage demands of 3DGS have motivated compression efforts. Optimization-based compression methods include value-based and structure-based ones. The former involve the use of pruning (Girish, Gupta, and Shrivastava 2024; Ali et al. 2024), masking (Lee et al. 2024b; Wang et al. 2024a), vector quantization (Niedermayr, Stumpfegger, and Westermann 2024; Navaneet et al. 2023) and distillation (Fan et al. 2024) to reduce insignificant Gaussians or redundancy of Gaussian parameters. The latter utilize structural modeling such as anchors (Lu et al. 2024), tri-planes (Lee et al. 2025) and 2D grids (Morgenstern et al. 2024) to address the sparsity and unorganized nature of Gaussian Splatting. When combined with learned entropy models (Balle,Â´

Laparra, and Simoncelli 2016; Cheng et al. 2020), they achieve excellent rate-distortion performance (Chen et al. 2024b, 2025; Zhan et al. 2025).

However, these optimization-based methods require perscene fine-tuning, limiting their generalization. Recent works have introduced feedforward pipelines (Chen et al. 2024a; Huang et al. 2025; Yang et al. 2025b) that decouples reconstruction and compression. These pre-trained codecs can directly compress arbitrary Gaussian clouds without multi-view supervision, offering greater practicality. We aim to extend this paradigm to dynamic GS compression.

## 2.2 Dynamic GS and its Compression

Dynamic GS methods can be categorized by their motion representation approaches. Implicit/explicit motion fields (Yang et al. 2024; Wu et al. 2024a; Li et al. 2024; Lin et al. 2024) and 4D Gaussian formulations (Duan et al. 2024; Lee et al. 2024a) often face challenges in real-time streaming, variable resolutions, or long durations. In contrast, per-frame approaches (Luiten et al. 2024; Sun et al. 2024; Wang et al. 2024b; Yan et al. 2025) demonstrate superior practicality through incremental frame updates, making them ideal for free-viewpoint video streaming. Our work focuses on this line of Per-Frame Gaussian methods.

For compression of Per-Frame Gaussian, HiCoM (Gao et al. 2024) employs hierarchical grid-wise motion representation and QUEEN (Girish et al. 2024) leverages a latentdecoder for quantization of attribute residuals. 4DGC (Hu et al. 2025) instills rate-aware training into frame reconstruction, achieving decent compression. While effective, they fundamentally couple reconstruction with compression, requiring scene-specific tuning. Our work breaks this constraint by introducing a novel optimization-free compression framework for dynamic GS, enabling zero-shot compression to arbitrary 4D scenes.

## 3 Preliminary

Our feedforward compression method takes frame-wise standard 3DGS as input, denoted as Gaussian frames, and each frame is pre-optimized from multi-view images at the respective timestamp. Within a given Gaussian frame, each Gaussian primitive is characterized by: (1) geometry parameters including position $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ and covariance matrix $ { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ ; (2) appearance parameters including opacity $o \in \mathbb { R } ^ { 1 }$ and SH-based color $c _ { S H } ~ \in ~ \mathbb { R } ^ { 4 8 }$ . The covariance matrix can be further represented as $\pmb { \Sigma } = \pmb { R } \pmb { S } \pmb { S } ^ { T } \pmb { R } ^ { T }$ where $\boldsymbol { R } \in \mathbb { R } ^ { 3 \times 3 }$ is the rotation matrix parameterized by the quaternion $\pmb q \in \mathbb { R } ^ { 4 }$ , and the scale matrix $\pmb { S } \in \mathbb { R } ^ { 3 \times 3 }$ is a diagonal matrix with elements $s \in \mathbb { R } ^ { 3 }$ . The geometry of a Gaussian primitive can be formulated as:

$$
G ( { \pmb x } ) = e ^ { - \frac { 1 } { 2 } ( { \pmb x } - { \pmb \mu } ) ^ { T } { \pmb \Sigma } ^ { - 1 } ( { \pmb x } - { \pmb \mu } ) } ,\tag{1}
$$

where x $\in \mathbb { R } ^ { 3 }$ is any random 3D location within the scene.

Given a viewpoint, 3D Gaussians are projected into a 2D plane, and the color of a pixel $C \in \mathbb { R } ^ { 3 }$ is derived by alphablending of overlapping 2D Gaussians:

$$
C = \sum _ { i } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

<!-- image-->  
Figure 2: Overview of D-FCGS framework. Our feedforward pipeline processes sequential Gaussian frames in GoF through three stages: (1) sparse motion extraction (Section 4.2), (2) feedforward motion compression (Section 4.3), and (3) motion compensation and refinement (Section 4.4). Once trained with rate-distortion loss, D-FCGS can infer on brand-new GS sequences in a zero-shot manner.

where $c _ { i } \in \mathbb { R } ^ { 3 }$ is the view-dependent color calculated from SH-based color $c _ { S H }$ , and $\boldsymbol { \alpha _ { i } } ~ \in \mathbb { R } ^ { 1 }$ is the blending weight derived from opacity o.

With the differentiable rasterizer, training can be supervised by 2D images from varied views in an end-to-end way. The rendering loss of vanilla 3DGS is:

$$
L _ { \mathrm { r e n d e r } } = \lambda L _ { \mathrm { D - S S I M } } + ( 1 - \lambda ) L _ { 1 } .\tag{3}
$$

## 4 Method

## 4.1 Overview

As shown in Fig. 2, our D-FCGS framework comprises three key components: (1) sparse motion extraction (Section 4.2), (2) feedforward motion compression (Section 4.3) and (3) motion compensation and refinement (Section 4.4). Training and inference procedures will be mentioned in Section 4.5.

Given two adjacent Gaussian frames $\mathbf { \Delta } \mathbf { x } _ { t }$ and $\hat { x } _ { t - 1 }$ , we first sample sparse control points from dense Gaussians and extract motion features. These motion tensors are compressed using our dual prior-aware entropy model for efficient rate estimation. During decoding, we reconstruct $\hat { \mathbf { x } } _ { t }$ via control-point-guided motion compensation and color refinement, while storing it into the buffer for future reference.

## 4.2 Sparse Motion Extraction via Control Points

Building on observations that most Gaussians exhibit local motion coherence, we propose an efficient sparse representation. Inspired by previous optimization-based 4D reconstruction methods (He et al. 2024; Yan et al. 2025; Huang et al. 2024), we employ Farthest Point Sampling (FPS) to select $\begin{array} { r } { N ^ { c } = \frac { N } { M } } \end{array}$ control points from the N Gaussian primitives, and derive corresponding geometry parameters, for-

mulated as:

$$
\pmb { \mu } ^ { c } = F P S ( \{ \mu _ { i } \} _ { i \in N } , \frac { N } { M } ) ,\tag{4}
$$

$$
\begin{array} { r } { { \pmb x } ^ { c } = I n d e x ( { \pmb x } , { \pmb \mu } ^ { c } ) , } \end{array}\tag{5}
$$

where M denotes the downscale factor, and $\pmb { x } ^ { c } \ \left( \pmb { x } ^ { c } \ \right. =$ $\{ \mu ^ { c } , q ^ { c } \} )$ represent geometry parameters of control points. This process efficiently reduces storage demand and processing costs, while preserving motion characteristics.

After control point sampling, we derive the native sparse geometry attributes of the current frame and reference Gaussian frame, denoted as $\boldsymbol { x } _ { t } ^ { c }$ and $\hat { x } _ { t - 1 } ^ { c }$ . We encode these attributes using frequency encoding (Mildenhall et al. 2021) and MLP projection:

$$
\pmb { y _ { t } ^ { c } } = M L P ( F r e q E n c \{ \pmb { x _ { t } ^ { c } } \} ) ,\tag{6}
$$

$$
\hat { \pmb { y } } _ { t - 1 } ^ { c } = M L P ( F r e q E n c \{ \hat { \pmb { x } } _ { t - 1 } ^ { c } \} ) .\tag{7}
$$

Motion tensors are then derived in the feature domain:

$$
m _ { t } ^ { c } = C o n v e r t e r ( y _ { t } ^ { c } - \hat { y } _ { t - 1 } ^ { c } ) ,\tag{8}
$$

where $\boldsymbol { m } _ { t } ^ { c }$ denotes the motion tensors at time t, which keeps the same dimension as $\boldsymbol { x } _ { t } ^ { c }$ and $\hat { x } _ { t - 1 } ^ { c }$ . Converter is realized by MLP at the feature level.

## 4.3 Feedforward Motion Compression

End-to-End Motion Compression. The obtained sparse motion tensors $\boldsymbol { m } _ { t } ^ { c }$ are then fed into our end-to-end compression module. This process begins with data encoding, followed by differentiable quantization simulated by additive uniform noise (Balle, Laparra, and Simoncelli 2016) to Â´ enable gradient backpropagation :

<!-- image-->  
Figure 3: Illustration of the dual prior-aware entropy model. Spatial-temporal context priors extracted via multiscale hashgrids and hyperpriors generated through the factorized model are fused by a lightweight fusion network for bitrate estimation.

$$
\begin{array} { r } { \hat { y } _ { t } ^ { m } = Q ( y _ { t } ^ { m } ) = y _ { t } ^ { m } + \mathcal { U } ( - \cfrac { q ^ { \prime } } { 2 } , \cfrac { q ^ { \prime } } { 2 } ) , \mathrm { ~ f o r ~ t r a i n i n g ~ } } \\ { = R o u n d ( \cfrac { y _ { t } ^ { m } } { q ^ { \prime } } ) \cdot q ^ { \prime } , \mathrm { ~ f o r ~ t e s t i n g } } \end{array}\tag{9}
$$

where $q ^ { \prime } \in \mathbb { R } ^ { 1 }$ is the quantization step size, and $y _ { t } ^ { m } , \hat { y } _ { t } ^ { m }$ denote the encoded latent motion before and after quantization, respectively. Next to that, arithmetic coding (AC) converts the quantized data into a compact bitstream for efficient transmission and storage. On the decoder side, the bitstream is decompressed back into motion tensors using arithmetic decoding (AD) for frame reconstruction.

Dual Prior-Aware Entropy Model. According to Shannonâs theory (Shannon 1948), the cross-entropy between the estimated and true latent distributions provides a tight lower bound on the achievable bitrate:

$$
R ( \hat { \pmb y } _ { t } ^ { m } ) \geq \mathbb { E } _ { \hat { \pmb y } _ { t } ^ { m } \sim q _ { \hat { \pmb { y } } _ { t } ^ { m } } } [ - \log _ { 2 } p _ { \hat { \pmb y } _ { t } ^ { m } } ( \hat { \pmb y } _ { t } ^ { m } ) ] ,\tag{10}
$$

where $p _ { \hat { y } _ { t } ^ { m } }$ and $q _ { \hat { y } _ { t } ^ { m } }$ are respectively estimated and true probability mass functions (PMFs) of the quantized latent codes $\hat { y } _ { t } ^ { m }$ . Since arithmetic coding can achieve a bitrate close to this bound, our goal is to devise an entropy model that accurately estimates $p _ { \hat { y } _ { t } ^ { m } }$

Fig. 3 shows the proposed dual prior-aware entropy model that combines hyperprior and spatial-temporal context prior for precise distribution estimation. Following (Balle et al. Â´ 2018), we use a factorized model to learn the hyperprior and estimate its PMF $p ( \hat { z } _ { t } ^ { m } )$ , which is common in deep image and video compression. However, for GS-based FVV, the latent codes also exhibit strong 3D-spatial and temporal correlations. Assuming adjacent Gaussian frames share similar features, we strive to extract spatial-temporal priors from the reference frame $\hat { x } _ { t - 1 }$ through a multi-resolution hash grid encoding scheme. Specifically, at each grid intersection, we store learnable feature vectors that capture position-specific information. For each Gaussian positioned at $\pmb { \mu } _ { t }$ , we retrieve multi-scale features by performing tri-linear interpolation across different levels $\dot { G } ^ { l }$ of the voxel grid, where l indicates the resolution level. The interpolated features from all levels are then concatenated and processed by a lightweight

MLP to generate comprehensive positional contexts:

$$
c o n t e x t = M L P ( \bigcup _ { l = 1 } ^ { L } I n t e r p ( \pmb { \mu _ { t } } , G ^ { l } ) ) ,\tag{11}
$$

where Interp(Â·) denotes the grid interpolation operation. These positional contexts are subsequently combined with the transformed appearance parameters as the final spatialtemporal priors. The prior fusion network is then to integrate these spatial-temporal priors with the hyperpriors, estimating the mean $\mu _ { t } ^ { m }$ and scale $\pmb { \sigma } _ { t } ^ { m }$ of the latent code distribution (assumed as normal distribution). The probability mass for each quantized latent value $\hat { y } _ { t , i } ^ { m }$ is computed by integrating the PMF over the quantization bin:

$$
p ( \hat { y } _ { t , i } ^ { m } ) = \int _ { \hat { y } _ { t , i } ^ { m } - \frac { q ^ { \prime } } { 2 } } ^ { \hat { y } _ { t , i } ^ { m } + \frac { q ^ { \prime } } { 2 } } \mathcal { N } ( y | \mu _ { t , i } ^ { m } , \sigma _ { t , i } ^ { m } ) d y ,\tag{12}
$$

where i corresponds to the index of a certain control point. The rate loss combines contributions from both the motion and hyper latent distributions:

$$
L _ { \mathrm { r a t e } } = \frac { 1 } { N ^ { c } } \sum _ { i = 1 } ^ { N ^ { c } } ( - \log _ { 2 } ( p ( \hat { y } _ { t , i } ^ { m } ) ) - \log _ { 2 } ( p ( \hat { z } _ { t , i } ^ { m } ) ) ) ,\tag{13}
$$

where $\begin{array} { r } { N ^ { c } = \frac { N } { M } } \end{array}$ is the total number of control points.

## 4.4 Motion Compensation and Refinement

Control Point Guided Motion Compensation. The decoded control point motions $\hat { m } _ { t } ^ { c }$ are propagated to the entire Gaussian set in a distance-aware compensation way. For the $i ^ { t h }$ control point, we first identify its K-nearest Gaussians $\kappa ( i )$ via KNN search. The motion vectors are then distributed to neighboring Gaussians using an exponentially decaying weight function based on spatial distance. The closer the distance is, the more influence we consider the control point can exert on the motion of this neighboring point. The motion that $i ^ { t h }$ control point assigns to its $j ^ { t h }$ neighbor can be written as:

$$
m _ { i , j } = \frac { e ^ { - d _ { i , j } } m _ { i } } { \sum _ { k \in \mathcal { K } ( i ) } e ^ { - d _ { i , k } } } ,\tag{14}
$$

where $d _ { i , j }$ represents the Euclidean distance between the $i ^ { t h }$ control point and its $j ^ { t h }$ neighbor. The aggregated motion vectors are finally applied to adjust the geometry parameters of the reference frame via addition.

This approach provides two key benefits: (1) enhanced motion estimation accuracy through localized correlation modeling that suppresses error propagation, and (2) inherent parallelizability due to the independent processing of control points, ensuring computational efficiency. More discussions on frame matching during motion compensation can be seen in the supplementary.

Color Refinement. For image and video compression, post-compression refinement plays a critical role in mitigating visual artifacts such as color banding and blurring. In our GS-based framework, we specifically target refinement at SH coefficients $c _ { S H }$ while preserving sensitive geometry and opacity parameters (Chen et al. 2024a; Girish et al. 2024; Papantonakis et al. 2024). The spatial-temporal priors from the entropy model module are repurposed to predict color residuals $\Delta c _ { S H }$ , which are dynamically added to $c _ { S H }$ during decoding without additional storage. This onthe-fly refinement is fully differentiable, allowing gradient backpropagation through the entropy model for joint ratedistortion optimization.

## 4.5 Training and Inference Pipeline of D-FCGS

Training Process and Loss. Following established practices (Wang et al. 2023; Zheng et al. 2024a,b), we adopt a Group-of-Frames (GoF) paradigm with the structure:

$$
\underbrace { \left( I - P \mathrm { - } \cdots \cdots P \right) } _ { \mathrm { G e r } _ { 1 } ( L ) } \underbrace { \left( I - P \mathrm { - } \cdots \cdots P \right) } _ { \mathrm { G o F } _ { 2 } ( L ) } \cdots \underbrace { \left( I - P \mathrm { - } \cdots \cdots P \right) } _ { \mathrm { G o F } _ { k } ( L ) } ,\tag{15}
$$

where $\mathrm { G o F } _ { k } ( L )$ denotes the $k ^ { t h }$ group containing one intracoded (I) frame followed by $L - 1$ predictively-coded (P) frames. The model is trained end-to-end with a composite loss function:

$$
L _ { \mathrm { t o t a l } } = L _ { \mathrm { r e n d e r } } + \lambda _ { \mathrm { s i z e } } L _ { \mathrm { r a t e } } ,\tag{16}
$$

where $L _ { \mathrm { r e n d e r } }$ replicates the original 3D Gaussian Splatting objective, and $\lambda _ { \mathrm { s i z e } }$ controls the rate-distortion trade-off.

Encoding and Decoding Process. The encoder extracts sparse motion from control points and compresses motion tensors $\hat { y } _ { t } ^ { m }$ and hyperpriors $\hat { z } _ { t } ^ { m }$ via arithmetic coding. During decoding, the system first reconstructs $\hat { z } _ { t } ^ { m }$ , then combines it with spatial-temporal priors to estimate distribution parameters $( \sigma _ { t } ^ { m } , \mu _ { t } ^ { m } )$ for decoding $\hat { y } _ { t } ^ { m }$ . These decoded motion features enable subsequent motion compensation.

## 5 Experiments

We rigorously evaluate our D-FCGS model by addressing three critical research questions:

â¢ Generalization and Effectiveness: Can D-FCGS generalize across new scenes from widely used datasets while achieving competitive rate-distortion performance compared to optimization-based methods? (Section 5.2)

â¢ Robustness and Stability: How does D-FCGS perform under diverse and high-dynamic scenes, and is the system stable under varying hyperparameters? (Section 5.3)

â¢ Module Efficacy: Do individual modules (e.g. control points) contribute meaningfully? (Section 5.4)

## 5.1 Experimental Setup

Datasets and Implementation Details. We derive sequential Gaussian frames from six multi-view video datasets: (1) N3V (Li et al. 2022b) (2) MeetRoom (Li et al. 2022a) (3) WideRange4D (Yang et al. 2025a) (4) Google Immersive (Broxton et al. 2020) (5) Self-Cap (Xu et al. 2024) (6) VRU (Wu et al. 2025). For training, we use 3 scenes from MeetRoom and 28 scenes from WideRange4D. For evaluation, we reserve the âdiscussionâ scene from MeetRoom for in-domain testing and six scenes from N3V for outof-domain benchmarking. Additional robustness tests are performed on diverse and high-dynamic scenes (see Section 5.3). All experiments are conducted on NVIDIA RTX 4090 GPU. Key hyperparameters include: (1) the downscale factor M = 70, (2) $K = 3 0$ for KNN, (3) quantization step $q ^ { \prime } = 1$ , (4) GoF size $L = 1 0$ during inference, and (4) $\lambda _ { \mathrm { s i z e } }$ = 1e-3. Further details of datasets and implementation are provided in the supplementary material.

<table><tr><td>Method</td><td>|PSNR (dB) â</td><td>Size SSIMâ (MB) â</td><td>Render (FPS) â</td><td>Feedforward Compression</td></tr><tr><td>K-Planes</td><td>31.63</td><td>0.920</td><td>1.0 0.15</td><td>X</td></tr><tr><td>HyperReel</td><td>31.10</td><td>0.931</td><td>1.7 16.7</td><td>X</td></tr><tr><td>NeRFPlayer</td><td>30.69</td><td>0.931</td><td>18.4 0.05</td><td>X</td></tr><tr><td>StreamRF</td><td>30.61</td><td>0.930</td><td>7.6 8.3</td><td>X</td></tr><tr><td>ReRF</td><td>29.71</td><td>0.918</td><td>0.77 2.0</td><td>X</td></tr><tr><td>TeTriRF</td><td>30.65</td><td>0.931</td><td>0.76 2.7</td><td>X</td></tr><tr><td>D-3DG</td><td>30.67</td><td>0.931</td><td>9.2 460</td><td>X</td></tr><tr><td>3DGStream*</td><td>32.20</td><td>0.953</td><td>7.75 215</td><td>X</td></tr><tr><td>HiCoM</td><td>31.17</td><td>-</td><td>0.70 274</td><td>X</td></tr><tr><td>QUEEN</td><td>32.19</td><td>0.946</td><td>0.75 248</td><td>X</td></tr><tr><td>4DGC</td><td>31.58</td><td>0.943</td><td>0.49 168</td><td>X</td></tr><tr><td>D-FCGS (ours)</td><td>31.91</td><td>0.952</td><td>0.46 215</td><td>â</td></tr></table>

Table 1: Quantitative results on N3V (Li et al. 2022b) dataset, averaged over 300 frames across six scenes. \* denotes results reproduced by our implementation. Bold and underlined values indicate the best and second-best performance, respectively. Detailed per-scene results are reported in supplementary material.
<table><tr><td>Method</td><td>PSNR (dB) â</td><td>SSIMâ</td><td>Size (MB) â</td><td>Render (FPS) â</td></tr><tr><td>StreamRF</td><td>26.71</td><td>0.913</td><td>8.23</td><td>10</td></tr><tr><td>ReRF</td><td>26.43</td><td>0.911</td><td>0.63</td><td>2.9</td></tr><tr><td>TeTriRF</td><td>27.37</td><td>0.917</td><td>0.61</td><td>3.8</td></tr><tr><td>3DGStream*</td><td>31.74</td><td>0.957</td><td>7.66</td><td>288</td></tr><tr><td>HiCoM</td><td>29.61</td><td>-</td><td>0.40</td><td>284</td></tr><tr><td>4DGC</td><td>28.08</td><td>0.922</td><td>0.42</td><td>213</td></tr><tr><td>D-FCGS (ours)</td><td>30.97</td><td>0.950</td><td>0.38</td><td>288</td></tr></table>

Table 2: Quantitative results on MeetRoom (Li et al. 2022a) dataset, averaged over 300 frames.

Evaluation Metrics. We evaluate D-FCGS using three kinds of metrics: (1) PSNR and SSIM (Wang et al. 2004) for reconstruction quality, (2) compressed size (MB/frame) for rate efficiency, and (3) rendering speed (FPS) plus encoding/decoding time (sec) for computational performance.

## 5.2 Benchmark Comparison

Benchmark Methods. As the first feedforward interframe codec for Gaussian point clouds (to our knowledge), D-FCGS faces unique evaluation challenges due to the absence of directly comparable optimization-free approaches. Our analysis therefore encompasses two categories of benchmark methods in optimization-based dynamic scene reconstruction: (1) NeRF-based techniques including K-Planes (Fridovich-Keil et al. 2023), Hyper-Reel (Attal et al. 2023), NeRFPlayer (Song et al. 2023),

Time

D

<!-- image-->

<!-- image-->  
(a) Flame Steak

<!-- image-->  
(b) Flame Salmon

<!-- image-->  
(c) Discussion

<!-- image-->  
(d) Cave

<!-- image-->  
(e) Bar

Figure 4: Qualitative comparison of rendering results. Visual comparisons between (1) Ground Truth, (2) 3DGStream, and (3) our D-FCGS. D-FCGS significantly reduces storage size while maintaining comparable high fidelity to 3DGStream.
<table><tr><td rowspan="2">Method</td><td colspan="3">Google Immersive</td><td colspan="3">Self-Cap</td><td colspan="3">VRU</td></tr><tr><td>PSNR(dB)â</td><td>SSIMâ</td><td>Size(MB)â</td><td>PSNR(dB)â</td><td>SSIMâ</td><td>Size(MB)â</td><td>PSNR(dB)â</td><td>SSIMâ</td><td>Size(MB)â</td></tr><tr><td>3DGStream*</td><td>25.84</td><td>0.872</td><td>7.6</td><td>26.43</td><td>0.858</td><td>7.6</td><td>26.18</td><td>0.892</td><td>7.6</td></tr><tr><td>D-FCGS(ours)</td><td>25.58</td><td>0.879</td><td>0.34</td><td>26.00</td><td>0.854</td><td>0.90</td><td>25.13</td><td>0.876</td><td>0.43</td></tr></table>

Table 3: Robustness test on diverse scenes from extensive datasets. Averaged PSNR, SSIM and P-frame size are reported. Perscene results are provided in the appendix.

<table><tr><td>K M L</td><td>30 60 70 70 300 300</td><td>120 70 300</td><td>30 140 300</td><td>30 280 300</td><td>30 70 32</td><td>30 70 4</td></tr><tr><td>Sizeâ PSNRâ</td><td>8.9 9.0 33.596 33.596</td><td>9.0</td><td>4.0 33.596 33.599</td><td>2.0 33.599</td><td>9.0 33.604 33.598</td><td>9.2</td></tr></table>

Table 4: Robustness test on hyperparameter settings (K for KNN, downscale factor M, and GoF size L). Averaged P-frame size (Ã0.1 MB) and PSNR (dB) are reported on âcut beefâ scene.

StreamRF (Li et al. 2022a), ReRF (Wang et al. 2023), and TeTriRF (Wu et al. 2024b); and (2) Per-Frame Gaussian methods including D-3DG (Luiten et al. 2024), 3DGStream (Sun et al. 2024), HiCoM (Gao et al. 2024), QUEEN (Girish et al. 2024), and 4DGC (Hu et al. 2025). While this comparison inherently favors optimization-based approaches that benefit from scene-specific tuning, it provides critical insights into D-FCGSâs performance relative to current paradigms in dynamic scene compression.

<table><tr><td>Method</td><td>Encoding(sec)</td><td>Decoding(sec)</td><td>Total(sec)</td></tr><tr><td>proposed</td><td>0.61</td><td>0.72</td><td>1.33</td></tr><tr><td>w/o control points</td><td>1.33</td><td>2.88</td><td>4.21</td></tr></table>

Table 5: Average encoding and decoding time for P-frames.

Quantitative Results. Our compression results demonstrate significant improvements over existing methods. As shown in Section 5.1 and Section 5.1, D-FCGS achieves remarkable average sizes of 0.46MB (N3V) and 0.38MB (MeetRoom) per frame, namely a 17Ã reduction compared to 3DGStreamâs 7.75MB and 7.66MB. Note that our codec is designed for inter-frame compression, and our P-frame compression ratio exceeds 32Ã in most cases (Fig. 1). While I-frames could be further compressed using existing static methods (e.g., FCGS (Chen et al. 2024a)), we maintain their original size for fair comparison. Additionally, the entire encoding/decoding pipeline operates efficiently, with both processes completing in under 1 second (Section 5.1).

<!-- image-->  
D-FCGS (proposed) D-FCGS (w/o ST-Prior) D-FCGS (w/o refinement) D-FCGS (w/o control points) 3DGStream 4DGC  
Figure 5: Rate-Distortion comparison of the proposed method and ablations.

Qualitative Results. We visualize the qualitative comparisons with 3DGStream (Sun et al. 2024) in Fig. 4, presenting results on âflame steakâ, âflame salmonâ (N3V dataset) and âdiscussionâ (MeetRoom dataset). From scene details, we can tell that D-FCGS achieves comparable fidelity to 3DGStream, even better in some cases.

## 5.3 Robustness Test

Robustness to Diverse Scenes. We evaluate D-FCGS on three additional datasets: Google Immersive, Self-Cap, and VRU, covering scenarios such as indoor painting, cave exploration, and basketball games. As summarized in Section 5.1, for Gaussian sequences reconstructed by 3DGStream, our approach achieves unprecedented P-frame compression while maintaining near-identical SSIM and small PSNR drop (< 0.5 dB for Google Immersive and Self-Cap). While PSNR gaps exist for VRU (full of blurry motions), our method offers a more practical and efficient pipeline for real-world applications, sacrificing acceptable visual fidelity for great improvements in bitrates. Fig. 4 highlights visual results on the outdoor âcaveâ (Google Immersive) and high-dynamic âbarâ (Self-Cap).

Robustness to Hyperparameter Settings. Here, we apply varied parameter configurations (KNN neighborhood size K, downscale factor M, and GoF size L) to the âcut beefâ scene (N3V). As presented in Section 5.1, D-FCGS maintains stable rendering quality across most parameter combinations, showing strong parametric robustness. Notably, increasing the downscale factor M effectively reduces control point counts, leading to a linear decrease in P-frame compression size, which aligns with expectations.

## 5.4 Ablation Study

In this section, we evaluate key components of D-FCGS through systematic ablation studies.

Effect of Control Points. The sparse motion representation via control points forms the foundation of our efficient compression pipeline. Removing control points and predicting motions for all Gaussians increases storage substantially (Fig. 5) and slows encoding/decoding by 3.2Ã (Section 5.1).

This validates our sparse motion representationâs efficiency in rate saving and computational cost.

Effect of Dual Prior-Aware Entropy Model. Our entropy model employs a novel dual-prior architecture. While hyperpriors effectively capture global dependencies (BalleÂ´ et al. 2018), our key innovation lies in the hash-grid-based spatial-temporal prior that models local correlations. Removing the spatial-temporal prior branch leads to noticeable R-D performance degradation (Fig. 5), confirming its importance to optimal compression.

Effect of Color Refinement. Our online color refinement module improves rendering quality (PSNR) by 0.1 â¼ 0.5 dB (Fig. 5), while requiring no additional storage and negligible decoding overhead. Per-scene results on N3V dataset are shown in the supplementary.

## 6 Conclusion

In this paper, we propose Feedforward Compression of Dynamic Gaussian Splatting (D-FCGS), a novel feedforward framework for zero-shot dynamic Gaussian sequence compression. Our contributions are threefold. First, we adopt the I-P coding profile for standard GS compression and introduce sparse inter-frame motion extraction via control points. Second, we present an end-to-end motion compression framework with a dual prior-aware entropy model, fully leveraging hyperpriors and spatial-temporal context to improve rate estimation. Third, control-point-guided motion compensation is combined with a color refinement network to guarantee high-fidelity and view-consistent reconstruction. Experimental results show that D-FCGS achieves superior compression efficiency (over 17Ã compression) on two benchmark datasets (MeetRoom and N3V) and remarkable robustness across diverse scenes from extensive datasets, significantly enhancing transmission and storage efficiency for free-viewpoint video applications.

## Acknowledgments

This work was partly supported by the NSFC62431015, Science and Technology Commission of Shanghai Municipality No.24511106200, the Fundamental Research Funds for the Central Universities, Shanghai Key Laboratory of Digital Media Processing and Transmission under Grant 22DZ2229005, 111 project BP0719010, Okawa Research Grant and Explore-X Research Fund.

## References

Ali, M. S.; Qamar, M.; Bae, S.-H.; and Tartaglione, E. 2024. Trimming the fat: Efficient compression of 3d gaussian splats through pruning. arXiv preprint arXiv:2406.18214.

Attal, B.; Huang, J.-B.; Richardt, C.; Zollhoefer, M.; Kopf, J.; OâToole, M.; and Kim, C. 2023. HyperReel: High-fidelity 6-DoF video with ray-conditioned sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16610â16620.

Balle, J.; Laparra, V.; and Simoncelli, E. P. 2016. End- Â´ to-end optimized image compression. arXiv preprint arXiv:1611.01704.

Balle, J.; Minnen, D.; Singh, S.; Hwang, S. J.; and Johnston, Â´ N. 2018. Variational image compression with a scale hyperprior. arXiv preprint arXiv:1802.01436.

Broxton, M.; Flynn, J.; Overbeck, R.; Erickson, D.; Hedman, P.; Duvall, M.; Dourgarian, J.; Busch, J.; Whalen, M.; and Debevec, P. 2020. Immersive light field video with a layered mesh representation. ACM Transactions on Graphics (TOG), 39(4): 86â1.

Chen, Y.; Wu, Q.; Li, M.; Lin, W.; Harandi, M.; and Cai, J. 2024a. Fast feedforward 3d gaussian splatting compression. arXiv preprint arXiv:2410.08017.

Chen, Y.; Wu, Q.; Lin, W.; Harandi, M.; and Cai, J. 2024b. Hac: Hash-grid assisted context for 3d gaussian splatting compression. In European Conference on Computer Vision, 422â438. Springer.

Chen, Y.; Wu, Q.; Lin, W.; Harandi, M.; and Cai, J. 2025. HAC++: Towards 100X Compression of 3D Gaussian Splatting. arXiv preprint arXiv:2501.12255.

Cheng, Z.; Sun, H.; Takeuchi, M.; and Katto, J. 2020. Learned image compression with discretized gaussian mixture likelihoods and attention modules. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 7939â7948.

Duan, Y.; Wei, F.; Dai, Q.; He, Y.; Chen, W.; and Chen, B. 2024. 4d-rotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes. In ACM SIGGRAPH 2024 Conference Papers, 1â11.

Fan, Z.; Wang, K.; Wen, K.; Zhu, Z.; Xu, D.; Wang, Z.; et al. 2024. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural information processing systems, 37: 140138â140158.

Fridovich-Keil, S.; Meanti, G.; Warburg, F. R.; Recht, B.; and Kanazawa, A. 2023. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12479â12488.

Gao, Q.; Meng, J.; Wen, C.; Chen, J.; and Zhang, J. 2024. Hicom: Hierarchical coherent motion for dynamic streamable scenes with 3d gaussian splatting. Advances in Neural Information Processing Systems, 37: 80609â80633.

Girish, S.; Gupta, K.; and Shrivastava, A. 2024. Eagles: Efficient accelerated 3d gaussians with lightweight encodings. In European Conference on Computer Vision, 54â71. Springer.

Girish, S.; Li, T.; Mazumdar, A.; Shrivastava, A.; De Mello, S.; et al. 2024. QUEEN: QUantized Efficient ENcoding of Dynamic Gaussians for Streaming Free-viewpoint Videos. Advances in Neural Information Processing Systems, 37: 43435â43467.

He, B.; Chen, Y.; Lu, G.; Wang, Q.; Gu, Q.; Xie, R.; Song, L.; and Zhang, W. 2024. S4d: Streaming 4d real-world reconstruction with gaussians and 3d control points. arXiv preprint arXiv:2408.13036.

Hu, Q.; Zheng, Z.; Zhong, H.; Fu, S.; Song, L.; Xiaoyun-Zhang; Zhai, G.; and Wang, Y. 2025. 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video. arXiv:2503.18421.

Huang, H.; Huang, W.; Yang, Q.; Xu, Y.; and Li, Z. 2025. A hierarchical compression technique for 3d gaussian splatting compression. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1â5. IEEE.

Huang, Y.-H.; Sun, Y.-T.; Yang, Z.; Lyu, X.; Cao, Y.-P.; and Qi, X. 2024. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4220â4230.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Lee, J.; Won, C.; Jung, H.; Bae, I.; and Jeon, H.-G. 2024a. Fully explicit dynamic gaussian splatting. Advances in Neural Information Processing Systems, 37: 5384â5409.

Lee, J. C.; Rho, D.; Sun, X.; Ko, J. H.; and Park, E. 2024b. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21719â21728.

Lee, S.; Shu, F.; Sanchez, Y.; Schierl, T.; and Hellge, C. 2025. Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs. arXiv preprint arXiv:2501.03399.

Li, L.; Shen, Z.; Wang, Z.; Shen, L.; and Tan, P. 2022a. Streaming radiance fields for 3d video synthesis. Advances in Neural Information Processing Systems, 35: 13485â 13498.

Li, T.; Slavcheva, M.; Zollhoefer, M.; Green, S.; Lassner, C.; Kim, C.; Schmidt, T.; Lovegrove, S.; Goesele, M.; Newcombe, R.; et al. 2022b. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5521â 5531.

Li, Z.; Chen, Z.; Li, Z.; and Xu, Y. 2024. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8508â8520.

Lin, Y.; Dai, Z.; Zhu, S.; and Yao, Y. 2024. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21136â21145.

Lu, T.; Yu, M.; Xu, L.; Xiangli, Y.; Wang, L.; Lin, D.; and Dai, B. 2024. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20654â20664.

Luiten, J.; Kopanas, G.; Leibe, B.; and Ramanan, D. 2024. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV), 800â809. IEEE.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Morgenstern, W.; Barthel, F.; Hilsmann, A.; and Eisert, P. 2024. Compact 3d scene representation via self-organizing gaussian grids. In European Conference on Computer Vision, 18â34. Springer.

Navaneet, K.; Meibodi, K. P.; Koohpayegani, S. A.; and Pirsiavash, H. 2023. Compact3d: Compressing gaussian splat radiance field models with vector quantization. arXiv preprint arXiv:2311.18159, 4.

Niedermayr, S.; Stumpfegger, J.; and Westermann, R. 2024. Compressed 3d gaussian splatting for accelerated novel view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10349â10358.

Papantonakis, P.; Kopanas, G.; Kerbl, B.; Lanvin, A.; and Drettakis, G. 2024. Reducing the memory footprint of 3d gaussian splatting. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 7(1): 1â17.

Shannon, C. E. 1948. A mathematical theory of communication. The Bell system technical journal, 27(3): 379â423.

Song, L.; Chen, A.; Li, Z.; Chen, Z.; Chen, L.; Yuan, J.; Xu, Y.; and Geiger, A. 2023. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 29(5): 2732â2742.

Sun, J.; Jiao, H.; Li, G.; Zhang, Z.; Zhao, L.; and Xing, W. 2024. 3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20675â20685.

Wang, H.; Zhu, H.; He, T.; Feng, R.; Deng, J.; Bian, J.; and Chen, Z. 2024a. End-to-end rate-distortion optimized 3d gaussian representation. In European Conference on Computer Vision, 76â92. Springer.

Wang, L.; Hu, Q.; He, Q.; Wang, Z.; Yu, J.; Tuytelaars, T.; Xu, L.; and Wu, M. 2023. Neural residual radiance fields for streamably free-viewpoint videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 76â87.

Wang, P.; Zhang, Z.; Wang, L.; Yao, K.; Xie, S.; Yu, J.; Wu, M.; and Xu, L. 2024b. VË 3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians. ACM Transactions on Graphics (TOG), 43(6): 1â13.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to

structural similarity. IEEE transactions on image processing, 13(4): 600â612.

Wu, G.; Yi, T.; Fang, J.; Xie, L.; Zhang, X.; Wei, W.; Liu, W.; Tian, Q.; and Wang, X. 2024a. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 20310â20320.

Wu, J.; Peng, R.; Wang, Z.; Xiao, L.; Tang, L.; Yan, J.; Xiong, K.; and Wang, R. 2025. Swift4D: Adaptive divide-and-conquer Gaussian Splatting for compact and efficient reconstruction of dynamic scene. arXiv preprint arXiv:2503.12307.

Wu, M.; Wang, Z.; Kouros, G.; and Tuytelaars, T. 2024b. Tetrirf: Temporal tri-plane radiance fields for efficient freeviewpoint video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6487â 6496.

Xu, Z.; Xu, Y.; Yu, Z.; Peng, S.; Sun, J.; Bao, H.; and Zhou, X. 2024. Representing Long Volumetric Video with Temporal Gaussian Hierarchy. ACM Transactions on Graphics, 43(6).

Yan, J.; Peng, R.; Wang, Z.; Tang, L.; Yang, J.; Liang, J.; Wu, J.; and Wang, R. 2025. Instant gaussian stream: Fast and generalizable streaming of dynamic scene reconstruction via gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, 16520â16531.

Yang, L.; Zhu, K.; Tian, J.; Zeng, B.; Lin, M.; Pei, H.; Zhang, W.; and Yan, S. 2025a. WideRange4D: Enabling High-Quality 4D Reconstruction with Wide-Range Movements and Scenes. arXiv preprint arXiv:2503.13435.

Yang, Q.; Yang, L.; Van Der Auwera, G.; and Li, Z. 2025b. HybridGS: High-Efficiency Gaussian Splatting Data Compression using Dual-Channel Sparse Representation and Point Cloud Encoder. arXiv preprint arXiv:2505.01938.

Yang, Z.; Gao, X.; Zhou, W.; Jiao, S.; Zhang, Y.; and Jin, X. 2024. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 20331â20341.

Yang, Z.; Yang, H.; Pan, Z.; and Zhang, L. 2023. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642.

Zhan, Y.-T.; Ho, C.-Y.; Yang, H.; Chen, Y.-H.; Chiang, J. C.; Liu, Y.-L.; and Peng, W.-H. 2025. CAT-3DGS: A Context-Adaptive Triplane Approach to Rate-Distortion-Optimized 3DGS Compression. arXiv preprint arXiv:2503.00357.

Zheng, Z.; Zhong, H.; Hu, Q.; Zhang, X.; Song, L.; Zhang, Y.; and Wang, Y. 2024a. HPC: Hierarchical Progressive Coding Framework for Volumetric Video. In Proceedings of the 32nd ACM International Conference on Multimedia, 7937â7946.

Zheng, Z.; Zhong, H.; Hu, Q.; Zhang, X.; Song, L.; Zhang, Y.; and Wang, Y. 2024b. JointRF: End-to-End Joint Optimization for Dynamic Neural Radiance Field Representation and Compression. In 2024 IEEE International Conference on Image Processing (ICIP), 3292â3298. IEEE.