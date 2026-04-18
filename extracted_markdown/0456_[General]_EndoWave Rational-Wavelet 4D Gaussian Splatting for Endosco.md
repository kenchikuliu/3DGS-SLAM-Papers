# EndoWave: Rational-Wavelet 4D Gaussian Splatting for Endoscopic Reconstruction

Taoyu Wu1,2, Yiyi Miao1,2, Jiaxin Guo3, Ziyan Chen1, Sihang Zhao1,

Zhuoxiao Li4, Zhe Tang5, Baoru Huang2, Limin Yu1

1Xiâan Jiaotong-Liverpool University, China

2University of Liverpool, United Kingdom

3The Chinese University of Hong Kong, Hong Kong, China

4The Hong Kong University of Science and Technology (Guangzhou), China

5Zhejiang University of Technology, China

AbstractâIn robot-assisted minimally invasive surgery, accurate 3D reconstruction from endoscopic video is vital for downstream tasks and improved outcomes. However, endoscopic scenarios present unique challenges, including photometric inconsistencies, non-rigid tissue motion, and view-dependent highlights. Most 3DGS-based methods that rely solely on appearance constraints for optimizing 3DGS are often insufficient in this context, as these dynamic visual artifacts can mislead the optimization process and lead to inaccurate reconstructions. To address these limitations, we present EndoWave, a unified spatiotemporal Gaussian Splatting framework by incorporating an optical flow-based geometric constraint and a multi-resolution rational wavelet supervision. First, we adopt a unified spatiotemporal Gaussian representation that directly optimizes primitives in a 4D domain. Second, we propose a geometric constraint derived from optical flow to enhance temporal coherence and effectively constrain the 3D structure of the scene. Third, we propose a multi-resolution rational orthogonal wavelet as a constraint, which can effectively separate the details of the endoscope and enhance the rendering performance. Extensive evaluations on two real surgical datasets, EndoNeRF [1] and StereoMIS [2], demonstrate that our method EndoWave achieves state-of-theart reconstruction quality and visual accuracy compared to the baseline method.

Index Termsâ3D Reconstruction, Gaussian Splatting, Wavelet.

## I. INTRODUCTION

Three-dimensional reconstruction in endoscopy is a crucial technology for enhancing surgeon perception and guidance during minimally invasive surgery. After collecting multiframe endoscopic video, reconstructing the operative scene improves spatial understanding and supports downstream tasks [3]. In robotic-assisted procedures, accurate modelling of dynamic anatomy benefits surgical planning and intraoperative navigation. However, the endoscopic environment introduces unique challenges for vision algorithms. The confined field of view, frequent occlusions such as instruments or tissue folds, specular highlights, and continuous non-rigid deformations undermine the static-scene assumptions and dense feature correspondences required by conventional stereo, Simultaneous Localization and Mapping (SLAM), and Structure-from-Motion pipelines [4]. These factors directly violate the core assumptions of static scenery and abundant feature correspondences upon which conventional 3D mapping approaches are built, often leading to incomplete or inaccurate reconstructions. To overcome these fundamental limitations, the field has increasingly turned towards learning-based methods capable of modeling complex, dynamic scenes.

<!-- image-->  
Fig. 1. Wavelet decomposition visualization on EndoNeRF dataset. We propose a rational wavelet decomposition that is effective for endoscopic scenarios. Our proposed rational wavelet is compared against the standard Orthogonal Haar and Biorthogonal bior6.8, visualizing their respective LL, LH, and HL components.

Among these learning-based solutions, Neural Radiance Fields (NeRF) [5], [6] have emerged as a paradigm, demonstrating exceptional fidelity in novel view synthesis and 3D reconstruction by representing scenes implicitly with neural networks. Early adaptations of this method to endoscopy, such as EndoNeRF [1], introduced dual-field representations to model deformable tissue through a canonical radiance field paired with a time-varying deformation field. Subsequent methods sought to refine this approach; for instance, EndoSurf [7] employed a neural signed distance function to explicitly enforce surface consistency, thereby improving geometric accuracy for dynamic tissues. Despite their rendering quality, NeRF-based methods are still hindered by slow inference and extensive training requirements, which make them impractical for realtime surgical use [8], [9]. Moreover, purely photometric NeRF optimizations can struggle with localization accuracy and temporal consistency in monocular endoscopy scenarios.

Recently, 3D Gaussian Splatting (3DGS) has emerged as a highly efficient alternative for real-time scene representation and rendering [10], [11]. In contrast to the implicit neural fields of NeRF, 3DGS models a scene as an explicit collection of anisotropic 3D Gaussians, each defined by properties like position, orientation, color, and opacity. This explicit structure permits rasterization at extremely high frame rates, making it a compelling candidate for surgical applications. Initial efforts to adapt 3DGS to the surgical domain, EndoGaussian [12] integrated depth priors to achieve near real-time reconstruction of deformable anatomy. Building on this, subsequent research has focused on extending 3DGS to model dynamic, 4D scenes. A foundational 4D Gaussian Splatting framework [13] proposed a holistic spatio-temporal representation, where a canonical set of Gaussians is deformed over time using a lightweight neural network. Deform3DGS [14] further refines the concept with a flexible deformation model that tracks tissue motion. Endo-4DGS [15] couples 3DGS with a learned deformation field to achieve monocular reconstruction. SurgicalGS [16] refines the reconstruction by fusing multi-frame depth cues and imposing geometric constraints to better capture fine anatomical details.

Despite this progress, significant gaps remain for 4D reconstruction in challenging endoscopic scenes. First, many dynamic 3DGS methods rely on a two-stage paradigm that first learns a static canonical representation and then trains a separate network to model its deformation over time. This separation can be suboptimal for complex non-rigid motion and complicates the optimization process. Second, existing approaches are predominantly guided by photometric consistency, lacking explicit constraints to ensure that the reconstructed motion is geometrically consistent with the observed pixel-level dynamics. This can lead to temporal artifacts and inaccurate deformation tracking. Third, the unique visual characteristics of endoscopic scenes, which contain smooth, low-frequency surfaces and sharp, high-frequency specular highlights, are difficult for standard loss functions to model accurately, often resulting in blurred details or noisy artifacts [17].

To address these limitations, we present EndoWave, a 4D Gaussian Splatting (4DGS) framework guided by optical flow and multi-resolution rational wavelets for high-fidelity reconstruction of dynamic endoscopic scenes. First, we adopt a unified temporal representation that directly optimizes Gaussians in a 4D spatio-temporal domain, avoiding the need for a canonical model and a separate deformation network. This approach inherently captures complex tissue motion and simplifies the training pipeline. Second, we introduce a geometric constraint by leveraging optical flow. We enforce consistency between the projected 2D motion of our 4D Gaussians and the optical flow estimated by off-the-shelf methods, such as RAFT [18] or GMFlow [19], ensuring that the reconstructed scene flow accurately reflects the observed pixel dynamics. Third, we propose a multi-resolution constraint using rational wavelets, which are better suited for the unique frequency characteristics of endoscopic imagery than traditional wavelets. By decomposing the rendered and ground-truth images into distinct frequency bands, we can preserve the global tissue structure while reconstructing high-frequency details, such as vessel boundaries and specular reflections. Extensive experiments on the EndoNeRF and StereoMIS datasets demonstrate that our method achieves state-of-the-art reconstruction quality, surpassing NeRF-based and 3DGS-based baselines in both geometric accuracy and visual fidelity, while maintaining interactive rendering rates.

Our contributions can be summarized in threefold:

â¢ Unified Spatio-Temporal Gaussian Representation. We represent the scenes as spatio-temporal Gaussians optimized directly over time, avoiding the conventional twostage canonical and deformation pipeline.

â¢ Flow-induced Geometric Constraint. We introduce an optical flow-derived loss to serve as a geometric constraint.

â¢ Multi-Resolution Wavelet Supervision. We propose a plug-in rational wavelet component that supervises the reconstruction across multiple frequency bands.

## II. RELATED WORK

## A. Neural Radiance Fields in Surgical Reconstruction

NeRF achieves impressive fidelity in novel view synthesis by learning a continuous function that maps 3D coordinates and view directions to color and density. Its adoption in medical imaging, while exploratory, has shown considerable promise [20]. The pioneering work of EndoNeRF first adapted this paradigm to endoscopic environments, introducing a dual neural field approach to separately model tissue deformation and the canonical scene representation. This decomposition enables the reconstruction of dynamic surgical scenes. Building upon this, EndoSurf enhanced geometric fidelity by incorporating signed distance functions and self-consistency constraints, which improved surface reconstruction accuracy at the cost of increased computational complexity. A primary challenge for these methods, however, is their computational intensity. To accelerate rendering for dynamic scenes, researchers have explored factorized or hybrid representations. Inspired by lowrank decomposition techniques, such as Instant-NGP, Ler-Plane [21] accelerates the modeling of deformable tissue by introducing a 4D grid representation that factorizes the scene into compact spatial and temporal components. Despite these advances, NeRF-based methods still suffer from fundamental computational limitations. Training times can extend for hours, and rendering speeds remain far from real-time requirements. This performance gap has motivated the exploration of alternative, explicit representations that promise comparable quality with superior computational efficiency.

## B. 3D Gaussian Splatting and Surgical Scene Modelling

In contrast to implicit volumetric approaches, 3DGS employs an explicit set of anisotropic Gaussian primitives whose properties are optimized through differentiable rasterization, enabling both direct geometric control and real-time rendering. In surgical environments, EndoGaussian [12] proposes holistic Gaussian initialization and spatio-temporal Gaussian tracking to cope with narrow baselines and rapid soft tissue motion. SurgicalGS [16] improves geometric accuracy via depth guided initialization and normalized depth regularization to counteract depth compression in inverse depth objectives. Deform3DGS [14] replaces heavy feature planes with learnable Gaussian bases that parameterize flexible deformations, preserving visual fidelity while cutting training time to approximately one minute per scene. Free-SurGS [8] introduces the first SfM-free 3D Gaussian Splatting framework for surgical scene reconstruction, which jointly optimizes camera poses and scene representation by utilizing optical flow as a geometric constraint.

<!-- image-->  
Fig. 2. Overall of the proposed framework. We take RGB-D frames as input to initialize a set of 4D Gaussian Splatting primitives G. Each primitive is decomposed into a conditional 3D Gaussian for its spatial representation and a marginal 1D Gaussian to model its temporal dynamics. The primitives are then jointly optimized using a composite loss function with RGB, depth, optical flow, and multi-scale wavelet supervision. After training, the model can render high-fidelity, time-evolving color and depth maps from novel viewpoints at any given time t.

## C. SLAM based Method

NeRF-based SLAM systems such as ENeRF-SLAM [22] and Endo-Depth-and Motion [23] adapt implicit fields to online tracking and mapping in endoscopy, but their computational cost limits real-time use. In contrast, EndoGSLAM [24] leverages the Gaussian representation with differentiable rasterization to maintain frame rates exceeding one hundred frames per second during joint tracking and mapping. EndoFlow-SLAM [9] further introduces flow-constrained optimization, in which the Gaussian map and camera motion are guided by both photometric consistency and optical flow, thereby strengthening temporal alignment and geometric stability in highly deformable sequences.

## III. METHODOLOGY

## A. Preliminary

1) 3D Gaussian Splatting: 3DGS [10] represents a scene using an explicit collection of anisotropic Gaussian primitives defined in 3D space. Each primitive $G _ { i }$ is parametrized by a mean position $\pmb { \mu } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ , an opacity $o _ { i } ~ \in ~ [ 0 , 1 ]$ , and a covariance matrix $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ . The spatial density of each Gaussian is given as:

$$
G _ { i } ( \mathbf { X } ) = o _ { i } \cdot \exp \left\{ - \frac { 1 } { 2 } ( \mathbf { X } - \pmb { \mu } _ { i } ) ^ { \top } \pmb { \Sigma } _ { i } ^ { - 1 } ( \mathbf { X } - \pmb { \mu } _ { i } ) \right\} ,\tag{1}
$$

where $\mathbf { X } \in \mathbb { R } ^ { 3 }$ denotes an arbitrary 3D point. The covariance matrix $ { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ can be decomposed into a scaling matrix and a rotation quaternion for efficient optimization.

2) Wavelet Decomposition: Wavelet analysis provides a mathematical framework for representing signals through a multi-resolution decomposition. Its fundamental strength lies in achieving a joint time-frequency representation, which allows for the localization of transient features across various scales. In contrast to the Fourier transform, which characterizes a signalâs constituent frequencies on a global basis, wavelet transforms employ a family of basis functions called wavelets that are localized in both time and frequency.

The practical implementation for a discrete image $I \in { \mathbf { \Omega } }$ $\mathbb { R } ^ { H \times W }$ is accomplished via the two-dimensional separable Discrete Wavelet Transform (DWT). This process entails the sequential application of a low-pass filter h and a high-pass filter g along the imageâs rows and columns. A single level of decomposition has four distinct sub-bands:

$$
\begin{array} { l } { { \displaystyle L L ( u , v ) = \sum _ { m } \sum _ { n } I ( m , n ) h ( u { - } m ) h ( v { - } n ) , } } \\ { { \displaystyle L H ( u , v ) = \sum _ { m } \sum _ { n } I ( m , n ) h ( u { - } m ) g ( v { - } n ) , } } \\ { { \displaystyle H L ( u , v ) = \sum _ { m } \sum _ { n } I ( m , n ) g ( u { - } m ) h ( v { - } n ) , } } \\ { { \displaystyle H H ( u , v ) = \sum _ { m } \sum _ { n } I ( m , n ) g ( u { - } m ) g ( v { - } n ) . } } \end{array}\tag{2}
$$

The LL sub-band represents a coarse, down-sampled approximation of the original image I. In contrast, the set $\{ L H , H L , H H \}$ captures high-frequency details corresponding to horizontal, vertical, and diagonal features, respectively.

Recursing on the approximation sub-band produces a multilevel pyramid.

## B. Spatio-temporal Modeling with 4D Gaussian Splatting

To capture the complex spatio-temporal dynamics inherent to endoscopic procedures, our work adapts the 4DGS framework [25], [26], which models the scene by optimizing a collection of 4D primitives within a unified spacetime volume to represent both spatial structure and temporal evolution simultaneously.

a) 4D primitive and time conditioning.: We leverage a set of 4D Gaussian primitives to represent the dynamic scene over time [25]. Each primitive is represented by an unnormalized Gaussian density over a spatio-temporal coordinate $( \mathbf { x } , t ) \in \mathbb { R } ^ { 3 } \times \mathbb { R }$ , defined as:

$$
\begin{array} { r } { p ( \mathbf { x } , t ) = \exp \Big ( - \frac { 1 } { 2 } \big [ ( \mathbf { x } , t ) - \pmb { \mu } \big ] ^ { \top } \pmb { \Sigma } ^ { - 1 } \big [ ( \mathbf { x } , t ) - \pmb { \mu } \big ] \Big ) , } \end{array}\tag{3}
$$

where $\boldsymbol { \mu } = \left( \mu _ { x } , \mu _ { t } \right)$ is the 4D mean, and $ { \Sigma } \in  { \mathbb { R } } ^ { 4 \times 4 }$ is 4D covariance . We partition Î£ into spatial, cross, and temporal blocks as $\Sigma _ { x , x } \in \bar { \mathbb { R } } ^ { 3 \times 3 } , \ \Sigma _ { x , t } \in \bar { \mathbb { R } } ^ { \bar { 3 } \times 1 }$ , and $\Sigma _ { t , t } \in \mathbb { R }$

b) Time-Evolved Appearance Spherindrical Harmonics: In 4DGS, view-dependent appearance is expanded in 4D spherindrical harmonics that couple spherical harmonics over viewing direction with a cosine-based temporal component, which implicitly fixes each temporal atom to a zero-phase origin. Building on this formulation, we introduce Time-Evolved Appearance Spherindrical Harmonics(TEASH), which incorporates a learnable phase for each temporal frequency. This approach produces a phase-adaptive Fourier atom while preserving the original spatial structure of the SH. Consequently, 4D TEASH $Z _ { n l } ^ { m } ( t , \theta , \phi )$ can be expressed as:

$$
Z _ { n l } ^ { m } ( t , \theta , \phi ) ~ = ~ Y _ { l } ^ { m } ( \theta , \phi ) \cos { \left( \omega _ { n } t + \varphi _ { n } \right) } ,\tag{4}
$$

where $Y _ { l } ^ { m }$ denotes the real spherical harmonic for direction $( \theta , \phi ) , \stackrel { . . } { \omega _ { n } } = 2 \pi n / T$ is the n-th temporal angular frequency over period $T ,$ , and $\varphi _ { n }$ is a learnable phase. This parameterization aligns each temporal component with the observed phase of the signal, since $\cos ( \omega t + \varphi )$ spans the sineâcosine pair at frequency Ï without introducing extra coefficients. In practice, TEASH retains the orthogonal angular basis while improving temporal fit to generate a concise and phaseconsistent appearance model for dynamic content.

## C. Flow induced geometric constraint

a) 3D Scene Flow and Estimated 2D Optical Flow: Our 4D Gaussian Splatting representation models a dynamic scene by defining the trajectory of each Gaussian primitive over time. For any given Gaussian i, its 3D center position is a function of time, denoted as ${ \bf P } _ { i } ( t ) \in \mathbb { R } ^ { 3 }$ according to Eq. (3). The instantaneous 3D velocity field of the scene is known as scene flow. We define the discrete scene flow vector $\mathbf { S } _ { i }$ for Gaussian i between two time steps, $t _ { 1 }$ and $t _ { 2 } ,$ , as the difference in its 3D position:

$$
\mathbf { S } _ { i } = \mathbf { P } _ { i } ( t _ { 2 } ) - \mathbf { P } _ { i } ( t _ { 1 } ) .\tag{5}
$$

To obtain a 2D optical flow representation from this 3D scene flow, we project the 3D Gaussian centers at both time steps onto their respective 2D image planes. Let Î (Â·) be the projection function that maps a 3D world point to 2D pixel coordinates, using the cameraâs intrinsic matrix K and extrinsic matrix E. The 2D projection of Gaussian i at time t is

$$
\mathbf { p } _ { i } ( t ) = \Pi \bigl ( \mathbf { P } _ { i } ( t ) , \mathbf { K } _ { t } , \mathbf { E } _ { t } \bigr ) .\tag{6}
$$

The estimated 2D optical flow vector $\mathbf { f } _ { e s t , i }$ for an individual Gaussian is the displacement of its projected center:

$$
\mathbf { f } _ { e s t , i } = \mathbf { p } _ { i } ( t _ { 2 } ) - \mathbf { p } _ { i } ( t _ { 1 } ) .\tag{7}
$$

For a 2D pixel p, the per-pixel flow is obtained by standard front-to-back alpha compositing:

$$
{ \hat { \mathbf { f } } } ( \mathbf { p } ) \ = \ \mathbf { F } _ { \mathrm { e s t } } ( \mathbf { p } ) \ = \ \sum _ { i \in \mathcal { G } ( \mathbf { p } ) } w _ { i } ( \mathbf { p } ) \mathbf { f } _ { \mathrm { e s t } , i } ,\tag{8}
$$

where $\mathcal G ( \mathbf { p } )$ denotes the set of Gaussians contributing to pixel p, sorted by the visibility ordering.

b) Ground Truth Optical Flow Supervision: To supervise the estimated dense flow $\hat { \mathbf { f } } \left( \mathbf { p } \right)$ , we generate ground-truth optical flow. Specifically, we first render the scene at times $t _ { 1 }$ and $t _ { 2 }$ with the 4D Gaussian Splatting pipeline to generate two consecutive frames $\mathbf { I } ( t _ { 1 } )$ and $\mathbf { I } ( t _ { 2 } )$ . We use the off-theshelf method GMFlow to get optical flow $\mathbf { f } \left( \mathbf { p } \right)$ as pseudo GT.

To enhance robustness, we filter unreliable pixel correspondences using a bi-directional consistency check on the pseudo ground-truth optical flow between frames $\mathbf { I } ( t _ { 1 } )$ and $\mathbf { I } ( t _ { 2 } )$ . This process generates a binary validity mask, ${ \bf M } _ { v a l i d }$ , to identify pixels with consistent flow. The final flow loss is subsequently computed only on the regions validated by this ${ { \bf { M } } _ { v a l i d } }$ mask:

$$
\mathcal { L } _ { \mathrm { f l o w } } = \left. \mathbf { M } _ { v a l i d } \odot \hat { \mathbf { f } } ( \mathbf { p } ) - \mathbf { M } _ { v a l i d } \odot \mathbf { f } ( \mathbf { p } ) \right. _ { 2 } .\tag{9}
$$

## D. Rational Wavelet Decomposition

1) Limitations of traditional dyadic wavelets.: Dyadic wavelet systems (orthogonal or biorthogonal) partition scale in octaves $2 ^ { \bar { j } }$ , producing a regular logarithmic tiling amenable to fast filter banks. For endoscopic imagesâwhere narrow specular spikes overlay low-frequency albedoâthese octave steps from $2 ^ { j }$ to $2 ^ { j + 1 }$ are often too coarse to isolate sharp peaks while preserving smooth content. Linear-phase biorthogonal variants do not resolve this, since their scale quantization is still dyadic; hence, dyadic analysis can be ill-suited when key features lie between octaves.

To address this limitation, a rational wavelet transform offers a more flexible multiresolution analysis by employing a rational dilation factor, allowing for a denser tiling of the time-frequency plane.

<!-- image-->  
Fig. 3. Qualitative results for novel view synthesis. Left: Results on the StereoMIS [2] dataset, with magnified details. Center and Right: Comparison of the Cutting and Pulling sequences from the EndoNeRF [1] dataset, respectively. The last column of each sequence is the error map, dark purple indicates low error, and yellow indicates high error.

a) Rational Scaling Principle: The core of the rational wavelet transform is the rational scale factor a, defined as:

$$
a = \frac { p } { q } = \frac { q + 1 } { q } , \qquad q \in \mathbb { N } , q \geq 1\tag{10}
$$

where $p$ and $q$ are coprime integers. This construction generates a sequence of scales $a ^ { j }$ that are more closely spaced than their dyadic counterparts.

b) Time-Domain Filter Design: Formal rational wavelet constructions, such as those based on the Meyer wavelet, are defined by their complex spectra in the frequency domain. Consequently, their corresponding time-domain filter coefficients lack a simple analytical form and must be computed numerically via an Inverse Fourier Transform. For a computationally efficient implementation, particularly within a deep learning loss function, we adopt a pragmatic approach in the time domain. We design a set of Meyer-like filters directly in the discrete time domain. These filters are inspired by the formal wavelet analysis [27] but are constructed from Gaussian functions for numerical stability and straightforward implementation. The filter design is parameterized by the rational scale factor a, from which we define a scale-adaptive bandwidth parameter $\sigma = 1 / a$ . The filters are defined over a discrete time index t.

Low-Pass Scaling Filter $( h _ { 0 } ) \colon \mathbf { A }$ smooth, zero-phase low-pass filter, analogous to a scaling function filter, is constructed from a normalized Gaussian function. This filter $h _ { 0 }$ captures the low-frequency approximation of the signal.

$$
h _ { 0 } ( t ) = \frac { \exp { \left( - \frac { 1 } { 2 } \left( \frac { t } { \sigma } \right) ^ { 2 } \right) } } { \sum _ { u } \exp { \left( - \frac { 1 } { 2 } \left( \frac { u } { \sigma } \right) ^ { 2 } \right) } } .\tag{11}
$$

High-Pass Wavelet Filters $( h _ { 1 } , g ) \colon$ Two distinct filters are defined to capture signal details at different orientations, similar to wavelet function filters. The first, $h _ { 1 } ( t )$ , is a derivative-of-

Gaussian (DoG) type filter, which is zero-mean and effective at emphasizing edges.

$$
h _ { 1 } ( t ) = \frac { - t \cdot \exp \left( - \frac { 1 } { 2 } \left( \frac { t } { \sigma } \right) ^ { 2 } \right) } { \sum _ { u } \left| - u \cdot \exp \left( - \frac { 1 } { 2 } \left( \frac { u } { \sigma } \right) ^ { 2 } \right) \right| } .\tag{12}
$$

The second, $g ( t )$ , is a smooth, Meyer-like band-pass filter constructed as a modulated Gaussian. Its center frequency depends on the rational factor a.

$$
g ( t ) = \frac { \sin ( \pi t / a ) \cdot \exp \left( - \frac { 1 } { 2 } \left( \frac { t } { 2 \sigma } \right) ^ { 2 } \right) } { \sum _ { u } \left| \sin ( \pi u / a ) \cdot \exp \left( - \frac { 1 } { 2 } \left( \frac { u } { 2 \sigma } \right) ^ { 2 } \right) \right| } .\tag{13}
$$

In the above definitions, u is the summation over all the discrete sample points of the filter kernel.

c) Separable 2D Transform: For a 2D signal image I, a separable wavelet transform is constructed from the tensor product of 1D analysis filters applied sequentially to the signalâs rows and columns. This decomposition is performed iteratively over J levels, indexed by $j \in \{ 0 , 1 , \ldots , J - 1 \}$ , to produce a multi-level signal representation.

The input to the transform at level j is denoted by $I ^ { ( j ) }$ with the original signal serving as the base level, $I ^ { ( 0 ) } \stackrel { \cdot } { = } I . \mathrm { A t }$ each level of analysis, the input $I ^ { ( j ) }$ is decomposed into four distinct frequency sub-bands. This decomposition is achieved by first performing 1D convolutions along the rows $( * _ { r } )$ and columns $( * _ { c } )$ , followed by a rational downsampling. The subbands are computed according to Eq. (2) as follow:

$$
\begin{array} { l } { { L L ^ { ( j ) } = ( ( I ^ { ( j ) } * _ { r } h _ { 0 } ) * _ { c } h _ { 0 } ) } } \\ { { L H ^ { ( j ) } = ( ( I ^ { ( j ) } * _ { r } h _ { 0 } ) * _ { c } h _ { 1 } ) } } \\ { { H L ^ { ( j ) } = ( ( I ^ { ( j ) } * _ { r } h _ { 1 } ) * _ { c } h _ { 0 } ) } } \\ { { H H ^ { ( j ) } = ( ( I ^ { ( j ) } * _ { r } g ) * _ { c } g ) } } \end{array}\tag{14}
$$

<!-- image-->  
Fig. 4. Qualitative 3D Model comparison on the EndoNeRF dataset

The approximation sub-band $L L ^ { ( j ) }$ is propagated to the next level to serve as the input for the subsequent decomposition, $I ^ { ( j + 1 ) } = L L ^ { ( j ) }$

d) Wavelet-domain loss.: We utilize the rational orthogonal wavelet transform with two-level decomposition. We utilize $L _ { 2 }$ loss based on the frequency map from the rendered image and ground truth image. The discrepancies of each frequency between rendered image ËI and ground truth I can be expressed as:

$$
\mathcal { L } _ { \mathrm { w a v e l e t } } = \sum _ { x \in \{ L L , L H , H L , H H \} } \lambda _ { x } \left. W _ { x } ( I _ { t } ) - W _ { x } \left( \hat { I } _ { t } \right) \right. ,\tag{15}
$$

where $W _ { x }$ represents the wavelet transformation that extract component x from an image and the $\lambda _ { x }$ denotes the weight for each wavelet component.

## E. Overall Training Objective function

With all defined loss terms, the overall training objective can be formulated as follows:

$$
{ \mathcal { L } } ~ = ~ \lambda _ { \mathrm { r g b } } { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { d e p t h } } { \mathcal { L } } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { f l o w } } { \mathcal { L } } _ { \mathrm { f l o w } } + \lambda _ { \mathrm { w a v e l e t } } { \mathcal { L } } _ { \mathrm { w a v e l e t } } ,\tag{16}
$$

where $\mathcal { L } _ { \mathrm { r g b } }$ and ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ denote the original losses in 3DGS [10], while ${ \mathcal { L } } _ { \mathrm { f l o w } }$ and $\mathcal { L } _ { \mathrm { w a v e l e t } }$ correspond to the proposed flow constraint and wavelet constraint, respectively.

## IV. EXPERIMENTS

## A. Implementation Details

1) Experimental Setup: All experiments were conducted on a single NVIDIA RTX 4090 GPU. We employed the Adam optimizer [29] for optimizing Gaussian primitives. For the attributes of the Gaussians, we follow the hyperparameter settings from Kerbl et al. [10], such as the learning rates and the threshold for density control. To ensure a fair comparison, the training iteration strategy was kept consistent with [25]. To validate the generalizability of our method, we applied the fixed set of hyperparameters, along with a consistent loss function and initialization strategy, across all datasets without any per-scene tuning for a fixed number of iterations.

2) Datasets: We quantitatively and qualitatively evaluate our proposed method on two publicly available datasets for dynamic endoscopic scene reconstruction: EndoNeRF and StereoMIS. (1) The EndoNeRF dataset [1] consists of stereo endoscopic sequences from two human prostatectomy procedures. Captured from a stationary viewpoint, this dataset provides challenging scenarios involving significant non-rigid tissue deformation and frequent tool occlusions. Each sequence is supplied with estimated depth maps, which are pre-calculated using stereo matching algorithms, and manually annotated masks to segment surgical tools from the scene. (2) The StereoMIS dataset [2] features eleven stereo video sequences from live porcine surgeries, recorded with the da Vinci Xi surgical system. This dataset is particularly characterized by large-scale and complex tissue deformations throughout the procedures. Similar to EndoNeRF, the dataset is annotated with corresponding tool segmentation masks. For our experiments, we extract the continuous 800 to 1000 frames from the first scene following [15]. For all scenes from the EndoNeRF and StereoMIS datasets, the frames were partitioned into training and testing sets following the 7:1 ratio.

TABLE I  
QUANTITATIVE RESULTS ON THE ENDONERF DATASET [1]. BEST RESULTS ARE IN BOLD.
<table><tr><td rowspan="2">Models</td><td colspan="3">EndoNeRF-Cutting</td><td colspan="3">EndoNeRF-Pulling</td><td rowspan="2">FPS â</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>EndoNeRF [1]</td><td>35.84</td><td>0.942</td><td>0.057</td><td>35.43</td><td>0.939</td><td>0.064</td><td>0.2</td></tr><tr><td>EndoSurf [7]</td><td>34.89</td><td>0.952</td><td>0.107</td><td>34.91</td><td>0.955</td><td>0.120</td><td>0.04</td></tr><tr><td>LerPlane-32k [21]</td><td>34.66</td><td>0.923</td><td>0.071</td><td>31.77</td><td>0.910</td><td>0.071</td><td>1.5</td></tr><tr><td>Endo-4DGS [15]</td><td>36.56</td><td>0.955</td><td>0.032</td><td>37.85</td><td>0.959</td><td>0.043</td><td>100</td></tr><tr><td>EndoGS [28]</td><td>37.16</td><td>0.953</td><td>0.045</td><td>36.19</td><td>0.941</td><td>0.041</td><td>70</td></tr><tr><td>SurgicalGS [16]</td><td>38.31</td><td>0.962</td><td>0.062</td><td>38.05</td><td>0.959</td><td>0.062</td><td>194</td></tr><tr><td>Ours</td><td>38.93</td><td>0.981</td><td>0.010</td><td>38.51</td><td>0.969</td><td>0.021</td><td>86</td></tr></table>

3) Evaluation Metrics: We evaluate the quality of our novel view synthesis using three image quality metrics: Peak Signalto-Noise Ratio (PSNR), Learned Perceptual Image Patch Similarity (LPIPS), and Structural Similarity Index Measure (SSIM). In addition to accuracy metrics, we also report practical performance indicators, including inference speed measured in Frames Per Second (FPS).

## B. Quantitative and qualitative results

We evaluated our method, EndoWave, against six representative reconstructing surgical scenes methods on the EndoNeRF and StereoMIS datasets. The compared methods include NeRF-based approaches and Gaussian Splatting relevant frameworks. The performance metrics, including PSNR, SSIM, LPIPS, and rendering speed (FPS), are detailed in Table I and Table II.

TABLE II  
QUANTITATIVE RESULTS ON THE STEREOMIS DATASET [2]. BEST RESULTS ARE IN BOLD.
<table><tr><td>Models</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FPS â</td></tr><tr><td>EndoNeRF [1]</td><td>21.49</td><td>0.622</td><td>0.360</td><td>0.2</td></tr><tr><td>EndoSurf [7]</td><td>29.87</td><td>0.809</td><td>0.303</td><td>0.04</td></tr><tr><td>LerPlane-32k [21]</td><td>30.80</td><td>0.826</td><td>0.174</td><td>1.7</td></tr><tr><td>Endo-4DGS [15]</td><td>32.69</td><td>0.850</td><td>0.148</td><td>100</td></tr><tr><td>SurgicalGS [16]</td><td>31.60</td><td>0.854</td><td>0.263</td><td>183</td></tr><tr><td>Ours</td><td>33.26</td><td>0.9126</td><td>0.075</td><td>77</td></tr></table>

On the EndoNeRF dataset [1], our method demonstrates superior reconstruction quality while operating at a real-time frame rate of 86 FPS. In contrast, existing methods, such as EndoNeRF [1] and EndoSurf [7], are limited to 0.2 and 0.04 FPS, respectively, which hinders their practical deployment. Leveraging the efficient scene representation of 3D Gaussian Splatting (3DGS), our approach not only preserves this computational advantage but also achieves high-fidelity novel view synthesis. Specifically, on the Cutting and Pulling sequences, our method attains PSNR scores of 38.93 and 38.51, respectively. As illustrated in Figure 3, we provide ground-truthmasked error maps where dark purple and yellow represent low and high reconstruction error, respectively. To further validate the generalization and robustness of our approach, we conducted evaluations on the StereoMIS dataset [2]. As reported in Table II, our method achieves state-of-the-art performance, with a PSNR of 33.26, an SSIM of 0.9126, and an LPIPS of 0.0554. For qualitative analysis, we present magnified views of regions with significant discrepancies in the error maps, as illustrated in the left of Figure 3.

This significant performance improvement is a direct result of our core contributions. Firstly, the direct time-conditioned 4DGS representation more accurately models complex nonrigid tissue deformations compared to approaches that rely on a canonical space and a deformation field. Secondly, the integration of an optical flow-based geometric constraint ensures temporal coherence across frames. Finally, the lower LPIPS score is attributed to our proposed rational orthogonal wavelets, which effectively separate the high and lowfrequency components of endoscopic scenes.

## C. Abaltion Study

We conducted an ablation study on the Pulling sequence of the EndoNeRF dataset to further analyze the contribution of our design. Starting from the full model, we progressively removed the proposed components and loss terms and performed extensive ablations. In addition to disabling our wavelet constraint $\mathcal { L } _ { \mathrm { w a v e l e t } }$ and flow constraint ${ \mathcal { L } } _ { \mathrm { f l o w } }$ , we also examined the effect of removing the depth constraint inherited from the original framework. As shown in Table III, excluding any single component consistently degrades performance, highlighting the critical role each plays in improving reconstruction quality, accuracy, and overall robustness.

TABLE III  
ABLATION EXPERIMENTS OF THE PROPOSED METHOD ON ENDONERF DATASET [1]. THE BEST RESULTS ARE IN BOLD.
<table><tr><td rowspan="2">Flow Constraint</td><td rowspan="2">Wavelet Constraint</td><td rowspan="2">Depth Constraint</td><td colspan="3">EndoNeRF-Pulling</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td></td><td></td><td></td><td>37.44</td><td>0.948</td><td>0.027</td></tr><tr><td></td><td></td><td></td><td>37.25</td><td>0.948</td><td>0.026</td></tr><tr><td></td><td>xx&gt;</td><td></td><td>36.93</td><td>0.946</td><td>0.030</td></tr><tr><td></td><td></td><td></td><td>37.47</td><td>0.947</td><td>0.029</td></tr><tr><td></td><td></td><td></td><td>37.20</td><td>0.946</td><td>0.029</td></tr><tr><td>****&gt;</td><td></td><td></td><td>37.56</td><td>0.950</td><td>0.027</td></tr><tr><td></td><td>*&gt;&gt;&gt;</td><td>&gt;*xÃ**&gt;</td><td>38.51</td><td>0.969</td><td>0.021</td></tr></table>

TABLE IV  
EFFECT OF WAVELET CONSTRAINT ON STEREOMIS DATASET (SECOND SEQUENCE) WITH ENDO-4DGS [15]. BEST RESULTS ARE IN BOLD.
<table><tr><td>Method</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>Endo-4DGS [15]</td><td>31.49</td><td>0.837</td><td>0.211</td></tr><tr><td> $\mathrm { E n d o { - } 4 D G S } + \mathcal { L } _ { \mathrm { w a v e l e t } }$ </td><td>31.72</td><td>0.839</td><td>0.203</td></tr></table>

We further evaluated generalization on the StereoMIS [2] dataset by incorporating the proposed wavelet constraint directly into Endo-4DGS [15]. The results Table IV show effective performance when only $\mathcal { L } _ { \mathrm { w a v e l e t } }$ is integrated, without modifying other parts.

## V. CONCLUSION

In this paper, we introduced EndoWave, a spatio-temporal Gaussian Splatting method for endoscopic reconstruction. Operating directly within a 4D domain, our approach enhances standard photometric training by incorporating supervision sensitive to both motion and frequency. To achieve this, the method estimates per-primitive scene flow from temporal evolution, projects this flow into the image space, and enforces consistency with externally computed optical flow. Furthermore, a two-level rational orthogonal wavelet loss constrains both low-frequency global appearance and highfrequency details, effectively mitigating artifacts from sources like specular reflections. Quantitative evaluations on the EndoNeRF and StereoMIS datasets show the effectiveness of our approach. EndoWave achieves a PSNR of 38.93 dB on the EndoNeRF Cutting sequence and 38.51 dB on the Pulling sequence, with interactive rendering at 86 FPS. On StereoMIS, it obtains 33.26 dB PSNR and 0.9126 SSIM at 77 FPS. Further analysis through ablation studies validates that the optical flow guidance, wavelet supervision, and depth regularization each provide complementary performance benefits.

This work relies on pseudo ground-truth optical flow and a fixed two-level wavelet configuration. Future work will investigate self-supervised motion cues that reduce dependence on external estimators, as well as adaptive scale selection in rational wavelets, including robustness under heavy instrument occlusion and rapid camera motions. We also plan to explore appearance models with richer temporal bases and to validate the method on larger multi-institutional datasets to assess generalization.

## REFERENCES

[1] Y. Wang, Y. Long, S. H. Fan, and Q. Dou, âNeural rendering for stereo 3d reconstruction of deformable tissues in robotic surgery,â in MICCAI. Springer, 2022, pp. 431â441.

[2] M. Hayoz, C. Hahne, M. Gallardo, D. Candinas, T. Kurmann, M. Allan, and R. Sznitman, âLearning how to robustly estimate camera pose in endoscopic videos,â International journal of computer assisted radiology and surgery, vol. 18, no. 7, pp. 1185â1192, 2023.

[3] M. Xu, Z. Guo, A. Wang, L. Bai, and H. Ren, âA review of 3d reconstruction techniques for deformable tissues in robotic surgery,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 157â167.

[4] K. Wang, C. Yang, Y. Wang, S. Li, Y. Wang, Q. Dou, X. Yang, and W. Shen, âEndogslam: Real-time dense reconstruction and tracking in endoscopic surgeries using gaussian splatting,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 219â229.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[6] J. Guo, J. Wang, R. Wei, D. Kang, Q. Dou, and Y.-h. Liu, âUc-nerf: Uncertainty-aware conditional neural radiance fields from endoscopic sparse views,â IEEE Transactions on Medical Imaging, 2024.

[7] R. Zha, X. Cheng, H. Li, M. Harandi, and Z. Ge, âEndosurf: Neural surface reconstruction of deformable tissues with stereo endoscope videos,â in International conference on medical image computing and computer-assisted intervention. Springer, 2023, pp. 13â23.

[8] J. Guo, J. Wang, D. Kang, W. Dong, W. Wang, and Y.-h. Liu, âFreesurgs: Sfm-free 3d gaussian splatting for surgical scene reconstruction,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 350â360.

[9] T. Wu, Y. Miao, Z. Li, H. Zhao, K. Dang, J. Su, L. Yu, and H. Li, âEndoflow-slam: Real-time endoscopic slam with flow-constrained gaussian splatting,â arXiv preprint arXiv:2506.21420, 2025.

[10] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[11] Z. Li, S. Yao, T. Wu, Y. Yue, W. Zhao, R. Qin, A. F. Garcia-Fernandez, A. Levers, and X. Zhu, âUlsr-gs: Ultra large-scale surface reconstruction gaussian splatting with multi-view geometric consistency,â arXiv preprint arXiv:2412.01402, 2024.

[12] Y. Liu, C. Li, C. Yang, and Y. Yuan, âEndogaussian: Real-time gaussian splatting for dynamic endoscopic scene reconstruction,â arXiv preprint arXiv:2401.12561, 2024.

[13] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20 310â20 320.

[14] S. Yang, Q. Li, D. Shen, B. Gong, Q. Dou, and Y. Jin, âDeform3dgs: Flexible deformation for fast surgical scene reconstruction with gaussian splatting,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 132â142.

[15] Y. Huang, B. Cui, L. Bai, Z. Guo, M. Xu, M. Islam, and H. Ren, âEndo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 197â207.

[16] J. Chen, X. Zhang, M. I. Hoque, F. Vasconcelos, D. Stoyanov, D. S. Elson, and B. Huang, âSurgicalgs: Dynamic 3d gaussian splatting for accurate robotic-assisted surgical scene reconstruction,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2025, pp. 572â582.

[17] T. Guo, T. Zhang, E. Lim, M. Lopez-Benitez, F. Ma, and L. Yu, âA review of wavelet analysis and its applications: Challenges and opportunities,â IEEe Access, vol. 10, pp. 58 869â58 903, 2022.

[18] Z. Teed and J. Deng, âRaft: Recurrent all-pairs field transforms for optical flow,â in Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part II 16. Springer, 2020, pp. 402â419.

[19] H. Xu, J. Zhang, J. Cai, H. Rezatofighi, and D. Tao, âGmflow: Learning optical flow via global matching,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 8121â 8130.

[20] Y. Long, Z. Li, C. H. Yee, C. F. Ng, R. H. Taylor, M. Unberath, and Q. Dou, âE-dssr: efficient dynamic surgical scene reconstruction with transformer-based stereoscopic depth perception,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2021, pp. 415â425.

[21] C. Yang, K. Wang, Y. Wang, X. Yang, and W. Shen, âNeural lerplane representations for fast 4d reconstruction of deformable tissues,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 46â56.

[22] J. Shan, Y. Li, T. Xie, and H. Wang, âEnerf-slam: a dense endoscopic slam with neural implicit representation,â IEEE Transactions on Medical Robotics and Bionics, 2024.

[23] D. Recasens, J. Lamarca, J. M. Facil, J. Montiel, and J. Civera, âEndo-Â´ depth-and-motion: Reconstruction and tracking in endoscopic videos using depth networks and photometric constraints,â IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 7225â7232, 2021.

[24] K. Wang, C. Yang, Y. Wang, S. Li, Y. Wang, Q. Dou, X. Yang, and W. Shen, âEndogslam: Real-time dense reconstruction and tracking in endoscopic surgeries using gaussian splatting,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 219â229.

[25] Z. Yang, H. Yang, Z. Pan, and L. Zhang, âReal-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting,â arXiv preprint arXiv:2310.10642, 2023.

[26] F. Li, J. He, J. Ma, and Z. Wu, âReal-time spatio-temporal reconstruction of dynamic endoscopic scenes with 4d gaussian splatting,â in 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). IEEE, 2025, pp. 1â5.

[27] L. Yu, F. Ma, E. Lim, E. Cheng, and L. B. White, âRational-orthogonalwavelet-based active sonar pulse and detector design,â IEEE Journal of Oceanic Engineering, vol. 44, no. 1, pp. 167â178, 2018.

[28] L. Zhu, Z. Wang, J. Cui, Z. Jin, G. Lin, and L. Yu, âEndogs: Deformable endoscopic tissues reconstruction with gaussian splatting,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 135â145.

[29] D. P. Kingma, âAdam: A method for stochastic optimization,â arXiv preprint arXiv:1412.6980, 2014.