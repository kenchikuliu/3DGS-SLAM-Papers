# Where, What, Why: Toward Explainable 3D-GS Watermarking

Mingshu Cai1 Jiajun Li2 Osamu Yoshie1 Yuya Ieiri1 Yixuan Li3\*

1Waseda University 2Southeast University 3Nanyang Technological University

mignshucai@fuji.waseda.jp, yoshie@waseda.jp, ieyuharu@ruri.waseda.jp

jiajun li@seu.edu.cn, yixuan.li@ntu.edu.sg

## Abstract

As 3D Gaussian Splatting becomes the de facto representation for interactive 3D assets, robust yet imperceptible watermarking is critical. We present a representationnative framework that separates where to write from how to preserve quality. A Trio-Experts module operates directly on Gaussian primitives to derive priors for carrier selection, while a Safety and Budget Aware Gate (SBAG) allocates Gaussians to watermark carriersâoptimized for bit resilience under perturbation and bitrate budgetsâand to visual compensators that are insulated from watermark loss. To maintain fidelity, we introduce a channel-wise group mask that controls gradient propagation for carriers and compensators, thereby limiting Gaussian parameter updates, repairing local artifacts, and preserving highfrequency details without increasing runtime. Our design yields view-consistent watermark persistence and strong robustness against common image distortions such as compression and noise, while achieving a favorable robustnessâquality trade-off compared with prior methods. In addition, the decoupled finetuning provides per-Gaussian attributions that reveal where the message is carried and why those carriers are selected, enabling auditable explainability. Compared with state-of-the-art methods, our approach achieves a PSNR improvement of +0.83 dB and a bit-accuracy gain of +1.24%.

## 1. Introduction

3D Gaussian Splatting (3D-GS) [10], with its explicit parameterization, real-time performance, and high fidelity, is emerging as the mainstream paradigm for 3D content creation and deploymentâsucceeding NeRF [22]âand is being widely adopted across film, gaming, autonomous driving, digital humans, and world models [5, 12, 14, 28, 39, 48].

However, with the large-scale generation and widespread distribution of 3D assets, their susceptibility to copying and tampering has made copyright protection increasingly criticalâonce a model is illicitly obtained or redistributed, the original creatorâs rights become extremely difficult to enforce. This challenge is particularly acute in 3D Gaussian Splatting (3D-GS): its core strengthâexplicit, directly editable Gaussian parametersâwhile enhancing representational expressiveness and rendering efficiency, also introduces significant security risks. Attackers can easily copy the model, tamper with its content, strip away authorship information, and illegally redistribute it, thereby severely undermining copyright attribution and provenance tracking.

<!-- image-->  
Figure 1. Usage of our model. The owner embeds a secret message into a 3D-GS model using our method; even if the model is stolen and undergoes distortion attacks, the rendered views can still be decoded to verify copyright.

Effective watermarking for 3D-GS is therefore urgent. Prior work on radiance-field watermarking, such as WateRF and 3DGSW [3, 7, 8], leaves two core gaps for explicit, discretized Gaussians:

(i) Carrier selection: From a large, heterogeneous set of Gaussian primitives, select watermark carriers by jointly considering multi-view visibility, frequency-domain cues, and the stability of geometry and appearance, with an emphasis on visual stability and information security.

(ii) Robust & imperceptible embedding: Embed a robust watermark without degrading visual or rendering quality, and ensure it remains extractable after common distortions such as cropping, compression, and format conversion.

Motivated by these considerations, we aim to build a framework unifying where to write, what to write, and why it matters: selecting stable 3D Gaussian carriers (where), embedding a distortion-resilient yet invisible signal (what), and achieving interpretable, auditable watermarking (why). A brief usage example is shown in Fig. 1

To address carrier selection, we introduce Trio-Experts, which analyzes intrinsic 3D parameters rather than rendered images. It extracts representation aligned priors: the Geometry expert scores structural stability, the Appearance expert gauges frequency characteristics for imperceptible edits, and the Redundancy expert estimates spatial replaceability for robustness. An uncertainty-aware fusion yields robustness-optimized Gaussian priors. These priors feed the Safety and Budget Aware Gate (SBAG), which finalizes where to write by routing Gaussians to the watermark set (WM) only when multi-view stability and safety budgets are met, and by expanding/densifying WM within the permitted visual-quality envelope. Non-carriers are assigned as Visual Compensators (VIS) for watermark embedding.

To minimize visual degradation from both WM and VIS Gaussians and improve embedding efficiency, we introduce a channel-wise Group Mask that specifies what to write by selecting watermark-eligible parameter channels while constraining gradient flow and per-channel update magnitudes.

Crucially, we completely decouple WM and VIS during training. VIS points are excluded from the watermark loss to avoid conflicts between carrier optimization and rendering fidelity, which explains why the system remains stable under adversarial procedures such as EOT; if VIS were coupled they would counter WM updates, harming visual quality and destabilizing extraction. Decoupling preserves image quality while maintaining watermark accuracy. In essence, this dual-role architecture enables robust & imperceptible watermark embedding.

Extensive experiments show that our method embeds watermarks consistently across all rendered views of a 3D-GS model while remaining robust to attacks on both images and the underlying representation. Compared with stateof-the-art approaches [3, 7, 8], it achieves superior results across all major metrics. Our core contributions are summarized as follows:

â¢ We propose a highly interpretable, attack oriented Decoupled Finetuning framework that fully separates carriers and compensators updating, with a channel wise Group Mask to route WM and VIS gradients, suppress harmful changes, preserve high frequency details, and achieve secure, robust embedding.

â¢ We introduce Trio-Experts, which operate directly on large 3D-GS point sets to extract high-quality geometry/appearance/redundancy priors for densification, channel-mask construction and watermark-carrier selection.

â¢ We present the Safety and Budget-Aware Gate (SBAG) that, under an adaptive budget, uses representation priors and lightweight rendering to select and densify watermark carriers, while cleanly separating them from visual compensators to enable decoupled optimization.

â¢ Our method achieves state-of-the-art performance and remains robust under diverse image distortion attacks.

## 2. Related Work

## 2.1. 3D Gaussian Splatting

3D Gaussian Splatting (3D-GS) [10] pairs an explicit point representation with differentiable splatting for realtime neural radiance-field rendering. Recent progress spans static reconstructionâshading/reflectance, largescale scenes, structured/view-adaptive models, multi-scale anti-aliasing [9, 16, 34, 44]âand dynamic/editable settings, including sparsely controllable editing, geometryprior deformation, real-time 4D, and pose-conditioned human avatars [6, 15, 32, 41]. On the systems side, volumetrically consistent rasterization, 4K-scale efficiency, and training-time point dropping enhance physical fidelity, throughput, and regularization [27, 33, 37]; under weak constraints, depthâGS coupling and unposed joint estimation enable feed-forward reconstruction and cross-scene generalization [4, 42, 46], while 3D-GS also probes 2D foundation features and unifies 2D/3D tone mapping for HDR [2, 17]. These trends sharpen the need for copyright protection and trustworthy use of 3D-GS assets.

## 2.2. Discrete Wavelet Transform

Discrete Wavelet Transform (DWT) captures spatialâfrequency locality via multiscale decomposition and has long supported image denoising/detail recovery [24]. In NeRFs and related 3D pipelines, wavelet-space priors improve sparsity, generalization, and training stabilityâcovering coefficient sparsification and generalizable synthesis [31, 43], direction-aware dynamics [18], and sparse-view wavelet losses plus 3D-GS frequency regularization [8, 25, 26]. We therefore enforce multiscale and local high-frequency consistency in the wavelet domain to boost quality and robustness while preserving global structure.

## 2.3. Digital Watermarking

Digital watermarking protects digital assets by identifying copyrights. The main difference lies in the priority of data embedding: watermarking prioritizes robustnessâensuring detection after distortionsâwhereas steganography prioritizes invisibility. To achieve robustness, traditional methods embed data in DWT subbands [1, 30, 35, 38]. HiDDeN [47] introduces an end-to-end deep watermarking framework with a noise layer. For radiance fields, CopyRNeRF [19] embeds messages into images rendered from implicit NeRFs. StegaNeRF [13] focuses on hiding information within rendered views with minimal visual impact, emphasizing imperceptibility. NeRFProtector [36] approaches the problem from a protection and authentication perspective, adding verifiable information to NeRFs without altering scene structure. WateRF [7], on the other hand, leverages DWT-based frequency embedding to enhance both fidelity and robustness. In the 3D-GS representation, 3DGSW [8] improves robustness through joint regularization based on rendering contribution and wavelet-domain constraints, while GuardSplat [3] focuses on real-world attack scenarios, achieving robust watermark embedding and verification with a strong pretrained CLIP [29] decoder. Unlike prior frequency-cue methods, we watermark directly in 3D-GS parameter space with gated, decoupled, attack-aware training for secure, robust, visually lossless embedding.

<!-- image-->  
Figure 2. Pipeline of Our Method. During initialization, we prune redundant Gaussians based on their rendering contribution. The Trio-Experts module extracts geometry/appearance/redundancy priors and aggregates them into an evidence package $E _ { k } ( i )$ . The SBAG decouples ranking and budgeting and uses this evidence and a one-shot render to select, expand, and densify watermark carriers. In finetuning, a channel-wise group mask enforces disjoint gradient routing for watermark carriers $( G S _ { \mathrm { w m } } )$ and visual compensators $( G S _ { \mathrm { v i s } } )$ EOT attacks render both clean and attacked views: $\dot { \mathcal { L } } _ { \mathrm { v i s } }$ preserves appearance, while low-frequency subbands form the watermark loss ${ \mathcal { L } } _ { \mathrm { w m } } .$ with $\mathcal { L } _ { \mathrm { w a v } } ^ { \mathrm { l o w } }$ penalizing over-editing. The separate optimization of ${ \mathcal { L } } _ { \mathrm { v i s } }$ and ${ \mathcal { L } } _ { \mathrm { w m } }$ improves both fidelity and robustness.

## 3. Method

## 3.1. Preliminary

3D Gaussian Splatting. We work under the standard   
3D Gaussian Splatting (3D-GS) formulation [10], where a

scene is represented by a set of 3D Gaussian primitives parameterized by mean $\pmb { \mu }$ and covariance Î£:

$$
\begin{array} { r } { G ( \mathbf { x } ; \pmb { \mu } , \pmb { \Sigma } ) = \mathrm { e x p } \big ( - \frac { 1 } { 2 } ( \mathbf { x } - \pmb { \mu } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \mathbf { x } - \pmb { \mu } ) \big ) . } \end{array}\tag{1}
$$

For rendering, each 3D Gaussian is projected to the image plane of a viewpoint Ï. Given depth ordering, pixels are composited by front-to-back alpha blending:

$$
I _ { \pi } [ x , y ] = \sum _ { i \in { \cal N } _ { G } } c _ { i } \alpha _ { i } T _ { i } , \quad T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $\mathcal { N } _ { G }$ is the set of visible 2D Gaussians at Ï sorted by depth, $c _ { i }$ denotes color, $\alpha _ { i }$ is the per-pixel contribution, and $T _ { i }$ is accumulated transmittance.

Prune by Contribution of Rendering Quality. Pre-trained 3D-GS models often contain redundant 3D Gaussians. We adopt the contribution-based pruning method from 3D-GSW [8], which quantifies contribution using an auxiliary loss with temporary color parameters $C ^ { \prime } .$ . The gradient $V _ { \pi } = \partial L _ { \pi } ^ { a u x } / \partial C ^ { \prime }$ serves as the contribution score to identify and prune low-impact Gaussians.

## 3.2. 3D Feature-Aware Trio-Experts

Prior methods often select carrier locations in the image or frequency domain based on gradient or high-frequency heuristics, but this evidence tends to drift with viewpoint changes. To address this issue, we propose a representationnative Trio-Experts system, whose decision evidence is fully anchored in the 3D-GS parameter space.

For the i-th Gaussian, its parameters are denoted as $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ (position), $\mathbf { s } _ { i } ~ \in ~ \mathbb { R } _ { + } ^ { 3 }$ (scale), $\mathbf { q } _ { i } ~ \in ~ \mathbb { H }$ (rotation), $\alpha _ { i } \in ( 0 , 1 )$ (opacity), and $\mathbf { h } _ { i } ~ = ~ [ \mathbf { h } _ { i } ^ { ( 0 ) } , \mathbf { h } _ { i } ^ { ( \ge 1 ) } ]$ (SH coefficients). We only extract features from these native parameters, without relying on pixel-domain gradients, thereby ensuring view-consistent evaluation and reducing computational overhead. We group the native 3D-GS parameters by semantics: $\mathcal { C } _ { \mathrm { g e o } } = \{ \mathbf { x } , \mathbf { s } , \mathbf { q } \}$ , $\mathcal { C } _ { \mathrm { a p p } } = \{ \alpha , \mathbf { h } ^ { ( 0 ) } , \mathbf { h } ^ { ( \ge 1 ) } \}$ , and $\mathcal { C } _ { \mathrm { r e d } } = \{ \mathbf { x } , \mathbf { s } , \mathbf { q } , \mathbf { h } ^ { ( 0 ) } \}$ . After quantile-based minâmax normalization, we construct a k-NN neighborhood $\mathcal { N } _ { k } ( i )$ in the 3D position space of x to model local density, and compute:

Geometric Features $z _ { 1 }$ . Based on ${ \mathcal { C } } _ { \mathrm { g e o } } ,$ we capture structural decomposition and boundary cues in 3D space via scale isotropy, rotational consistency, and a compact footprint, to measure geometric stability:

$$
z _ { 1 } ( i ) = \mathrm { N o r m } \big ( \mathrm { I s o } _ { i } , \mathrm { R o t C o n s } _ { i } , 1 - \overline { { \mathrm { f p } } } _ { i } \big ) ,\tag{3}
$$

where $\mathrm { I s o } _ { i } = \mathrm { m i n } ( \mathbf { s } _ { i } ) / \mathrm { m a x } ( \mathbf { s } _ { i } )$ measures scale isotropy, $\begin{array} { r } { \operatorname { R o t } \mathrm { C o n s } _ { i } \ = \ ( 1 / k ) \sum _ { j \in \mathcal { N } _ { k } ( i ) } | \langle \mathbf { q } _ { i } , \mathbf { q } _ { j } \rangle | } \end{array}$ is neighborhood quaternion consistency, and $\overline { { \mathrm { f p } } } _ { i } = \mathrm { e x p } ( ( 1 / 3 ) \sum _ { d } \log s _ { i , d } )$ is the geometric-mean footprint.

Appearance Features $z _ { 2 } .$ . Based on $\mathcal { C } _ { \mathrm { a p p } } ,$ we measure cross-view appearance consistency from color and opacity cues via DC band-pass, opacity gating, and high-frequency suppression:

$$
z _ { 2 } ( i ) = \mathrm { N o r m } \big ( 1 - \rho _ { i } ^ { \mathrm { h f } } , ~ g ( \alpha _ { i } ) , ~ c _ { i } \big ) ,\tag{4}
$$

where $\rho _ { i } ^ { \mathrm { h f } }$ is the AC high-frequency energy ratio, $g ( \alpha )$ is a double-sided opacity gate, and $c _ { i }$ is a Gaussian band-pass on DC strength.

Redundancy Features $z _ { 3 }$ . Based on $\mathcal { C } _ { \mathrm { r e d } }$ , we characterize distributional density among 3D Gaussians and estimate substitutability via overlap-weighted neighborhood similarity in DC color and shape:

$$
z _ { 3 } ( i ) = \mathrm { N o r m } \Big ( \frac { 1 } { k } \sum _ { j \in \mathcal { N } _ { k } ( i ) } w _ { i j } r _ { i j } \Big ) ,\tag{5}
$$

where $r _ { i j }$ combines color and shape similarity, and $w _ { i j } =$ $\exp ( - \bar { d _ { i j } ^ { 2 } } / ( \sigma _ { o } ^ { 2 } ( \bar { s } _ { i } ^ { 2 } + \bar { s } _ { j } ^ { 2 } ) ) )$ approximates projected overlap using spatial distance and scale.

Each expert k maps its features $z _ { k } ( i )$ into an evidence package $E _ { k } ( i ) \ = \ [ U _ { k } ( i ) , S _ { k } ( i ) ]$ , separating quality from

certainty:

$$
\begin{array} { r l } & { U _ { k } ( i ) = \mathrm { N o r m } \bigl ( \mathrm { D i s p } _ { \mathcal { N } ( i ) } ( z _ { k } ) + \mathrm { P e n a l t y } _ { k } ( i ) \bigr ) , } \\ & { S _ { k } ( i ) = \mathrm { N o r m } ( z _ { k } ( i ) ) , k \in \{ 1 , 2 , 3 \} . } \end{array}\tag{6}
$$

where $U _ { k } ~ \in ~ [ 0 , 1 ]$ measures uncertainty from neighborhood dispersion (Disp) and expert-specific penalties, and $S _ { k } ~ \in ~ [ 0 , 1 ]$ is the quality score; this decoupling enables confidence-aware expert gating.

## 3.3. Safety and Budget-Aware Gate (SBAG)

To select robust watermark carriers from a large 3D-GS set, we decouple the pipeline into ranking and budgeting. Ranking relies solely on Trio-Experts evidence packages $E _ { k } ( i ) \ = \ [ U _ { k } ( i ) , S _ { k } ( i ) ]$ , mapped to proxy scores aligned with $\{ \mathcal { C } _ { \mathrm { g e o } } , \mathcal { C } _ { \mathrm { a p p } } , \mathcal { C } _ { \mathrm { r e d } } \}$

$$
R _ { k } ( i ) = \mathrm { c l i p } \big ( S _ { k } ( i ) - \beta U _ { k } ( i ) , 0 , 1 \big ) , \quad k \in \{ 1 , 2 , 3 \} .\tag{7}
$$

where $\beta$ is a fixed constant shared across experts. We interpret $R _ { 1 }$ as geometric stability, $R _ { 2 }$ as appearance safety, and $R _ { 3 }$ as redundancy certainty; in all cases, higher scores are preferred. To avoid assuming expert dominance, we define a symmetric point-wise utility:

$$
u _ { i } = { \Big ( } R _ { 1 } ( i ) \cdot R _ { 2 } ( i ) \cdot R _ { 3 } ( i ) { \Big ) } ^ { \frac { 1 } { 3 } } .\tag{8}
$$

We render all training views once using DC+opacity to obtain view-corrected visibility and distribution priors. The rasterizer provides the non-negative compositing weight $w _ { i , p } ^ { ( t ) } \geq 0$ of Gaussian i at pixel p in view t. We estimate a scene-level crowding factor $\eta \in ( 0 , 1 ]$ as

$$
\eta = \frac { 1 } { V } \sum _ { t = 1 } ^ { V } \frac { \sum _ { p } \operatorname* { m i n } \Bigl ( 1 , \sum _ { i } w _ { i , p } ^ { ( t ) } \Bigr ) } { \sum _ { p } \sum _ { i } w _ { i , p } ^ { ( t ) } + \epsilon } , \qquad w _ { i , p } ^ { ( t ) } \geq 0 ,\tag{9}
$$

and obtain per-Gaussian visibility $v _ { i } \in [ 0 , 1 ]$ by accumulating its screen-space contribution and normalizing. We further compute the scene-average visibility $\begin{array} { r } { \bar { v } = \frac { 1 } { N } \bar { \sum } _ { i = 1 } ^ { N } v _ { i } } \end{array}$

Given message length M (bits), we model the effective bits contributed by one carrier as a scene-adaptive coefficient

$$
\kappa _ { \mathrm { e f f } } = \kappa _ { 0 } \cdot \bar { v } \cdot \eta , \qquad B = \biggl \lceil \frac { M } { \kappa _ { \mathrm { e f f } } } \biggr \rceil ,\tag{10}
$$

where $\kappa _ { 0 }$ is a constant determined by the embedding design.

We define a generic feasible set using quantile-based constraints:

$$
\begin{array}{c} \mathcal { F } = \Big \{ i \Big | _ { R _ { 1 } ( i ) } ^ { R _ { 1 } ( i ) } \geq \mathrm { Q } _ { q } ( R _ { 1 } ) , R _ { 2 } ( i ) \geq \mathrm { Q } _ { q } ( R _ { 2 } ) ,  \\ { R _ { 3 } ( i ) \geq \mathrm { Q } _ { q } ( R _ { 3 } ) , v _ { i } \geq \mathrm { Q } _ { q } ( v ) } \end{array} \Big \} ,\tag{11}
$$

and perform deterministic water-level selection within F by $u _ { i } \colon$

$$
\mathcal { W M } _ { 0 } = \mathrm { t o p } { - } B \ \{ u _ { i } \ | \ i \in \mathcal { F } \} .\tag{12}
$$

To bridge viewpoint-induced coverage gaps, we retain a Prototype-based Proximity Extension. We build a compact evidence vector where $c _ { i } = \| \mathbf h _ { i } ^ { ( 0 ) } \| _ { 2 }$ is DC strength and $h _ { i } = \rho _ { i } ^ { \mathrm { h f } }$ is the AC high-frequency ratio:

$$
\begin{array} { r } { \mathbf { e } _ { i } = \mathrm { N o r m } \big ( R _ { 1 } ( i ) , R _ { 2 } ( i ) , R _ { 3 } ( i ) , v _ { i } , h _ { i } , c _ { i } \big ) , } \end{array}\tag{13}
$$

compute the prototype $\begin{array} { r } { \pmb { \mu } = \frac { 1 } { | \mathcal { W } \mathcal { M } _ { 0 } | } \sum _ { i \in \mathcal { W } \mathcal { M } _ { 0 } } \mathbf { e } _ { i } , } \end{array}$ , recruit proximal neighbors $\boldsymbol { \ w } \mathcal { M } _ { \mathrm { p r o x } }$ by cosine similarity, and form

$$
\mathcal { W } \mathcal { M } _ { \mathrm { p a r e n t } } = \mathcal { W } \mathcal { M } _ { 0 } \cup \mathcal { W } \mathcal { M } _ { \mathrm { p r o x } } .\tag{14}
$$

Finally, each parent is split into $N _ { s }$ visually equivalent children, routing one child to the watermark branch:

$$
\mathcal { W M } _ { \star } = \bigcup _ { i \in \mathcal { W M } _ { \mathrm { p a r e n t } } } \mathcal { C } _ { \mathrm { w m } } ( i ) ,\tag{15}
$$

while the remaining children act as visual compensators during finetuning to neutralize embedding artifacts. The resulting WMâ and its complement VIS are used for subsequent channel-wise mask routing.

## 3.4. Channel-wise Group Mask

To avoid visible degradation, we assign channel-wise masks to watermark carriers and visual compensators, and optimize a visual loss and a watermark loss under separate gradient routes. For Gaussian $i ,$ we precompute per-group masks $m _ { g } ^ { \mathrm { w m } } ( i ) , m _ { g } ^ { \mathrm { v i s } } ( i ) \ \in \ [ 0 , 1 ]$ from Trio evidence and one-shot priors, We derive the two masks from point-wise channel weightsw $i _ { g } ^ { ( \mathrm { v i s } ) } ( j )$ and $w _ { g } ^ { ( \mathrm { w m ) } } ( j )$ as

$$
\begin{array} { r l } & { m _ { g } ^ { \mathrm { v i s } } = \operatorname* { m a x } \bigl ( \mathrm { c l i p } ( \mathrm { m e a n } ( w _ { g } ^ { \mathrm { v i s } } [ { \mathcal V } { \mathcal Z } S ] ) , 0 , c a p _ { g } ) , f l o o r _ { g } \bigr ) , } \\ & { m _ { g } ^ { \mathrm { w m } } = \mathrm { c l i p } \bigl ( \mathrm { m e d } ( w _ { g } ^ { \mathrm { w m } } [ { \mathcal W } { \mathcal M } _ { \star } ] ) , 0 , c a p _ { g } \bigr ) . } \end{array}\tag{16}
$$

where $Q _ { 0 . 5 }$ is the median, $\mathcal { V } = \{ 1 , . . . , N \} \setminus \mathcal { W } \mathcal { M } _ { \star }$ , and $\mathrm { c a p } _ { g } , \mathrm { f l o o r } _ { g }$ are per-channel bounds.

and gate gradients as

$$
\nabla _ { \theta _ { i } ^ { g } } \mathcal { L } = \left\{ \begin{array} { l l } { m _ { g } ^ { \mathrm { w m } } ( i ) \nabla _ { \theta _ { i } ^ { g } } \mathcal { L } _ { \mathrm { w m } } , } & { i \in \mathcal { W M } _ { \star } , } \\ { m _ { g } ^ { \mathrm { v i s } } ( i ) \nabla _ { \theta _ { i } ^ { g } } \mathcal { L } _ { \mathrm { v i s } } , } & { i \in \mathcal { V } \mathbb { Z } S , } \end{array} \right.\tag{17}
$$

where $g \ \in \ \{ \delta _ { \mathrm { d c } } , \rho _ { \mathrm { r e s t } } , \omega _ { \mathrm { o p a } } , \theta _ { \mathrm { r o t } } , \sigma _ { \mathrm { s c a } } \}$ denotes five parameter channels.

To ensure complete gradient separation, we perform two passes of ${ \mathcal { L } } _ { \mathrm { v i s } }$ and ${ \mathcal { L } } _ { \mathrm { w m } } .$ , so that $\mathcal { W M } _ { \star }$ â and VIS receive their gradients in an orthogonal manner.

## 3.5. Decoupled Watermark Finetuning

To resolve the gradient conflict between rendering fidelity and watermark robustness, we implement a decoupled finetuning strategy. It isolates gradient propagation by ensuring that watermark carriers WMâ are exempt from visual reconstruction constraints, while visual compensators VIS remain unaffected by watermark-related objectives. Specifically, the visual objective is

$$
\mathcal { L } _ { \mathrm { v i s } } = \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { l p i p s } } \mathcal { L } _ { \mathrm { l p i p s } } + \lambda _ { \mathrm { w a v } } ^ { \mathrm { h i g h } } \mathcal { L } _ { \mathrm { w a v } } ^ { \mathrm { h i g h } } ,\tag{18}
$$

where $\mathcal { L } _ { \mathrm { r e c } }$ is the L1 reconstruction loss, $\mathcal { L } _ { \mathrm { l p i p s } }$ is LPIPS [45], and $\mathcal { L } _ { \mathrm { w a v } } ^ { \mathrm { h i g h } }$ penalizes multi-level DWT highfrequency subbands $( S \in \{ L H , H L , H H \} )$ with an L1 distance.

Simultaneously, we optimize the watermark branch using Expectation Over Transformation (EOT). Let $\begin{array} { r l } { \mathbf { b } _ { i } } & { { } \in \mathbf { \pi } } \end{array}$ $\{ 0 , 1 \} ^ { \bar { B } }$ denote the target bits of carrier i, $\nu _ { i }$ its visible views, and $D _ { \psi } ( \cdot )$ a decoder producing logits. For clean and transformed renderings, we average logits over visible views:

$$
\bar { \mathbf { z } } _ { i } ^ { \mathrm { c l } } = \frac { 1 } { | \mathcal { V } _ { i } | } \sum _ { v \in \mathcal { V } _ { i } } D _ { \psi } ( \hat { I } _ { v } ) , \quad \bar { \mathbf { z } } _ { i } ( \mathbf { t } ) = \frac { 1 } { | \mathcal { V } _ { i } | } \sum _ { v \in \mathcal { V } _ { i } } D _ { \psi } ( \mathbf { t } ( \hat { I } _ { v } ) ) ,\tag{19}
$$

where $\mathsf { \varepsilon } \sim p ( \mathsf { t } )$ is sampled from a standard degradation family (e.g., blur, rotation, scaling, crop, noise, JPEG). We decode bits by $\hat { \mathbf { y } } _ { i } ^ { \mathrm { c l } } = \sigma ( \bar { \mathbf { z } } _ { i } ^ { \mathrm { c l } } )$ and $\hat { \mathbf { y } } _ { i } ^ { \mathrm { e o t } } ( \mathbf { t } ) ~ = ~ \sigma ( \bar { \mathbf { z } } _ { i } ( \mathbf { t } ) )$ . We embed watermarks only in the DWT low-frequency $( L L )$ subband and regularize low-frequency distortion with $\mathcal { L } _ { \mathrm { w a v } } ^ { \mathrm { l o w } }$ (multi-level DWT LL subband with an L1 distance; see supp.). The clean and EOT watermark losses are

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { w m } } ^ { \mathrm { c l e a n } } = \frac { 1 } { | \mathcal { W } _ { \star } | } \displaystyle \sum _ { i \in \mathcal { W } _ { \star } } \mathrm { B C E } \bigl ( \hat { \mathbf { y } } _ { i } ^ { \mathrm { c l } } , \mathbf { b } _ { i } \bigr ) , } \\ & { \mathcal { L } _ { \mathrm { w m } } ^ { \mathrm { e o t } } = \mathbb { E } _ { \mathsf { t } \sim p } \left[ \frac { 1 } { | \mathcal { W } _ { \star } | } \displaystyle \sum _ { i \in \mathcal { W } _ { \star } } \mathrm { B C E } \bigl ( \hat { \mathbf { y } } _ { i } ^ { \mathrm { e o t } } ( \mathsf { t } ) , \mathbf { b } _ { i } \bigr ) \right] , } \end{array}\tag{20}
$$

and the total watermark objective is

$$
\mathcal { L } _ { \mathrm { w m } } = \lambda _ { \mathrm { w m } } ^ { \mathrm { c l e a n } } \mathcal { L } _ { \mathrm { w m } } ^ { \mathrm { c l e a n } } + \lambda _ { \mathrm { w m } } ^ { \mathrm { e o t } } \mathcal { L } _ { \mathrm { w m } } ^ { \mathrm { e o t } } + \lambda _ { \mathrm { w a v } } ^ { \mathrm { l o w } } \mathcal { L } _ { \mathrm { w a v } } ^ { \mathrm { l o w } } .\tag{21}
$$

Finally, although we report the joint scalar objective ${ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { v i s } } { \mathcal { L } } _ { \mathrm { v i s } } + \lambda _ { \mathrm { w m } } { \mathcal { L } } _ { \mathrm { w m } }$ , the actual parameter updates follow the masked routing rule in Eq. (17): watermark gradients act only on $\mathcal { W } _ { \star }$ , and visual gradients act only on VIS. This disjoint gradient routing effectively eliminates the optimization interference in joint finetuning.

## 4. Experiments

## 4.1. Experimental Setting

Dataset & Pre-trained 3D-GS. We evaluate on the standard benchmarks in the NeRF [22] and 3D-GS [10] literatureâBlender [21], LLFF [20], and Mip-NeRF 360 [23]. Following common practice, we report results on 25 scenes drawn from the full versions of these datasets.

Baseline. We compare our method against three strategies for fairness: (i) WateRF [7], an innovative watermarking method deployable in both the NeRF representations; for fairness, we instantiate and evaluate it in the 3D-GS setting. (ii) GuardSplat [3], a CLIP-guided and SH-aware robust watermarking model for 3DGS, enabling invisible, secure, and distortion-resistant embedding with high visual fidelity. (iii) 3D-GSW [8], a state-of-the-art watermark embedding model operating in the frequency domain.

<!-- image-->  
Figure 3. Rendering-quality comparison. We compare our method with all baselines using 32-bit messages. The difference maps are shown at 10Ã scale. Our approach achieves higher bit accuracy and better visual fidelity than competing methods.

<table><tr><td>Methods</td><td colspan="4">32 bits</td><td colspan="4">48 bits</td><td colspan="4">64 bits</td></tr><tr><td></td><td>Bit Accâ</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>Bit Accâ</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>Bit Accâ</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td></tr><tr><td>WateRF [7]+3D-GS [10]</td><td>93.28</td><td>30.57</td><td>0.954</td><td>0.052</td><td>84.39</td><td>30.06</td><td>0.949</td><td>0.056</td><td>74.92</td><td>25.73</td><td>0.887</td><td>0.105</td></tr><tr><td>GuardSplat [3]</td><td>95.58</td><td>35.32</td><td>0.978</td><td>0.043</td><td>93.29</td><td>33.36</td><td>0.969</td><td>0.045</td><td>90.14</td><td>32.25</td><td>0.963</td><td>0.048</td></tr><tr><td>3D-GSW [8]</td><td>97.22</td><td>35.15</td><td>0.977</td><td>0.044</td><td>93.59</td><td>33.26</td><td>0.972</td><td>0.047</td><td>91.31</td><td>32.52</td><td>0.966</td><td>0.050</td></tr><tr><td>Ours</td><td>98.46</td><td>35.98</td><td>0.982</td><td>0.041</td><td>94.29</td><td>33.45</td><td>0.973</td><td>0.044</td><td>91.65</td><td>32.71</td><td>0.969</td><td>0.047</td></tr></table>

Table 1. Bit accuracy and rendering quality compared with baselines. We report results for 32, 48, and 64-bit messages, averaged over the Blender, LLFF, and Mip-NeRF 360 datasets. The best scores are shown in bold.

Implementation Details. We train all models on a single NVIDIA A800 GPU for 2â10 epochs using Adam [11]. The decoder is a frozen HiDDeN [47] model (32/48/64 bits). We prune Gaussians with $V _ { \pi } { < } 1 0 ^ { - 8 }$ . All experiments are repeated with three random seeds.

Evaluation. We evaluate our watermarking framework along three key dimensions.(1) Invisibility. Visual fidelity is measured using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) [40], and Learned Perceptual Image Patch Similarity (LPIPS) [45]. (2) Robustness. We assess robustness by reporting bit accuracy under a variety of image-space degradations: Gaussian noise (Ï = 0.1), rotation (randomly sampled from Â±Ï/6), scaling to 75% of the original size, Gaussian blur (Ï = 0.1), cropping to 40%, JPEG compression (quality 50%), and a composite attack that combines Gaussian noise, crop, and JPEG compression. We also evaluate model-space distortions by removing 20% of Gaussians, cloning 20% of Gaussians, and adding Gaussian noise (Ï = 0.1) to 3D-GS parameters. (3) Capacity. We further investigate bit accuracy under different payload sizes, with message lengths

$$
M _ { b } \in \{ 3 2 , 4 8 , 6 4 \} .
$$

## 4.2. Results

Rendering Quality and Bit Accuracy. We compare our approach with existing methods in terms of visual fidelity and watermark extraction. As shown in Fig. 3, our results are closest to the original views while still achieving high bit accuracy. This is especially notable on real-world scenes, where complex geometry makes faithful rendering harder. Unlike prior methods that struggle to trade off image quality and watermark reliability, our approach maintains both. Quantitative results in Tab. 1 further confirm that our method consistently preserves rendering quality and bit accuracy across all datasets.

Robustness to Image Distortions. In this section, we evaluate how well our method withstands post-processing on rendered images, where such operations may alter or weaken the embedded watermark. We measure bit accuracy under various distortion types. As shown in Table. 2, most existing methods fail to maintain high robustness. Moreover, WateRF+3D-GS and 3D-GSW without EOT-based adversarial training still struggle to reliably preserve embedded messages in rendered views. GuardSplat does introduce EOT to improve robustness, but its heavy reliance on the CLIP decoder and SH-space embedding makes it less effective under more complex or compound distortions. In contrast, our method performs strict selection and decoupled embedding, writing watermarks only into safe 3D Gaussians, thereby achieving consistently robust extraction across all distortion settings.

<!-- image-->

Figure 4. Rendering-quality comparison under different message capacities. With 32-bit, 48-bit, and 64-bit embedded messages, (differences Ã10), our method maintains high bit-acc across different capacities while preserving perceptual quality.
<table><tr><td rowspan="2">Methods</td><td colspan="8">Bit Accuracy(%) â</td></tr><tr><td>No Distortion</td><td>Gaussian Noise (Ï = 0.1)</td><td>Rotation (Â±Ï/6)</td><td>Scaling (75%)</td><td>Gaussian Blur (Ï = 0.1)</td><td>Crop (40%)</td><td>JPEG Compression (50% quality)</td><td>Combined</td></tr><tr><td>WateRF [7]+3D-GS [10]</td><td>93.28</td><td>78.12</td><td>81.47</td><td>84.63</td><td>87.09</td><td>84.58</td><td>82.03</td><td>64.73</td></tr><tr><td>GuardSplat [3]</td><td>95.58</td><td>90.11</td><td>95.87</td><td>94.93</td><td>97.16</td><td>95.05</td><td>89.92</td><td>88.64</td></tr><tr><td>3D-GSW [8]</td><td>97.22</td><td>83.71</td><td>88.05</td><td>94.58</td><td>95.94</td><td>92.73</td><td>92.54</td><td>90.96</td></tr><tr><td>Ours</td><td>98.46</td><td>91.22</td><td>96.18</td><td>95.06</td><td>97.75</td><td>95.88</td><td>92.95</td><td>91.30</td></tr></table>

Table 2. Quantitative robustness comparison under different attacks against baseline methods. Results are averaged over the Blender, LLFF, and Mip-NeRF 360 datasets using 32-bit messages. The best scores are marked in bold.

<table><tr><td rowspan="2">Methods</td><td colspan="4">Bit Accuracy(%) â</td></tr><tr><td>No Distortion</td><td>Adding Gaussian Noise 3D Gaussians 3D Gaussians (Ï = 0.1)</td><td>Removing (20 %)</td><td>Cloning (20 %)</td></tr><tr><td>WateRF [7]+3D-GS [10]]</td><td>93.28</td><td>62.35</td><td>60.91</td><td>76.16</td></tr><tr><td>GuardSplat [3]</td><td>95.58</td><td>79.96</td><td>88.49</td><td>92.30</td></tr><tr><td>3D-GSW [8]</td><td>97.22</td><td>89.93</td><td>97.23</td><td>96.85</td></tr><tr><td>Ours</td><td>98.46</td><td>90.52</td><td>98.16</td><td>97.48</td></tr></table>

Table 3. Robustness under model-level distortions with a 32-bit payload. Best results are shown in bold.

Robustness to Model-level Distortions. To evaluate robustness under malicious model-level tampering, we perturb the 3D-GS representation by adding parameter noise and randomly removing or cloning 3D Gaussians. As reported in Tab. 3, our method consistently outperforms prior works across these distortions, indicating that the embedded message is not tied to a fragile subset of primitives. Notably, it maintains reliable decoding even when the underlying model is partially corrupted, demonstrating robust protection for both the 3D-GS model and its rendered outputs.

Message Capacity. Since there is an inherent trade-off among bit accuracy, rendering quality, and payload size, we investigate different message lengths 32, 48, 64. As shown in Table. 1, increasing the message length consistently leads to a slight drop in both bit accuracy and rendering quality. Nevertheless, our method maintains a favorable balance between invisibility and capacity, and the performance gap over other methods becomes more evident as the message length grows. Fig. 4 further illustrates that our approach achieves a better compromise between bit accuracy and visual fidelity.

## 4.3. Ablation Study

In this section, we ablate the SBAG, Group Mask, and Decoupled Finetuning to clarify their necessity. Fig. 5 and Tab. 4 show that removing any one of them leads to either lower bit accuracy or degraded rendering, and removing all of them breaks the qualityârobustness balance. This indicates that all three are required to stabilize watermark embedding.

<!-- image-->  
Figure 5. Comparison of rendering quality among the full method (ours), without SBAG, without group mask, without decoupled finetuning, and the baseline model. All images are embedded with 32-bit messages.

<table><tr><td colspan="3">Methods</td><td colspan="4">Ours</td></tr><tr><td>SBAG</td><td>Group Mask</td><td>Decoupling</td><td>Bit Acc(%)â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>â</td><td>â</td><td>â</td><td>94.70</td><td>30.00</td><td>0.952</td><td>0.073</td></tr><tr><td>â</td><td>â</td><td></td><td>96.80</td><td>34.10</td><td>0.974</td><td>0.048</td></tr><tr><td>â</td><td></td><td>â</td><td>97.10</td><td>33.70</td><td>0.972</td><td>0.051</td></tr><tr><td></td><td>â</td><td>â</td><td>95.10</td><td>33.10</td><td>0.968</td><td>0.056</td></tr><tr><td></td><td>Â¸</td><td>â</td><td>97.80</td><td>35.20</td><td>0.979</td><td>0.042</td></tr></table>

Table 4. Quantitative ablation results. The best performance is obtained when all components are activated, with evaluation conducted on 32-bit message embedding.

<table><tr><td>Budget</td><td>Bit Acc â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>Storage â</td></tr><tr><td>1%</td><td>97.19</td><td>34.98</td><td>0.980</td><td>0.046</td><td>164MB</td></tr><tr><td>Adaptive</td><td>98.46</td><td>35.98</td><td>0.982</td><td>0.041</td><td>178MB</td></tr><tr><td>10%</td><td>98.12</td><td>34.24</td><td>0.975</td><td>0.048</td><td>195MB</td></tr></table>

Table 5. Ablation study on Budget for Watermark Carrier Gaussians. Results represent the average score across Blender, LLFF, and Mip-NeRF 360 datasets using 32-bit messages.

SBAG. To jointly improve bit accuracy and rendering quality during watermark embedding, we introduce the SBAG to select, expand and create high-quality carrier Gaussians. The Gaussians selected by SBAG lie in visually insensitive regions and have multiple channels that can be safely modified. Combined with decoupled training, this effectively alleviates the conflict between visual fidelity and watermark embedding. As shown in Tab. 4 and Fig. 5, SBAG-based selection consistently boosts both rendering quality and bit accuracy. Moreover, Tab. 5 shows that controlled densification within an adaptive budget can further improve watermark performance with lower storage cost.

<!-- image-->

<!-- image-->  
Figure 6. Training dynamics of coupled vs. decoupled finetuning. Decoupled optimization converges faster and consistently achieves higher PSNR and BitAcc across finetuning steps.

Group mask. Before watermark embedding, the pretrained 3D-GS already achieves high rendering quality, so we want to avoid large updates on the majority of non-carrier (i.e., visual) Gaussians. In contrast, carrier Gaussians should have more freedom to be perturbed across channels to encode the watermark. Our gradient group mask enforces this asymmetry during backpropagation: gradients on visual Gaussians are suppressed below those of prior methods, while gradients on carrier Gaussians are adaptively scaled per channel. This preserves rendering quality while maintaining watermark accuracy. As shown in Fig. 7, our group-aware mask achieves a better balance between visual fidelity and watermark performance than alternative masking strategies.

<!-- image-->  
Figure 7. Qualitative comparison of our group mask. For background-free objects, it confines 3D Gaussian updates to object boundaries, preserving rendering quality.(difference Ã8)

Decoupled Finetuning. To verify the necessity of decoupled finetuning, we analyze the optimization conflict between the visual and watermark objectives by running coupled and decoupled finetuning under identical settings and periodically evaluating PSNR and BitAcc on the same validation set. As shown in Fig. 6, coupled finetuning exhibits a transient degradation in BitAcc while PSNR keeps improving, revealing a clear trade-off, whereas decoupled optimization converges faster and maintains consistently higher PSNR and BitAcc throughout finetuning.

We provide additional ablation results in the supplementary material.

## 5. Conclusion

We conclude with a representation-native framework that unifies where to write, what to write, and why it matters. Where: Trio-Experts derive geometry, appearance, and redundancy priors from Gaussian parameters, and together with a safety and budget aware SBAG plus single-pass rendering cues, select carriers that are robust yet visually safe, training them separately from visual Gaussians. What: A channel wise group mask constrains gradients to designated parameter channels, preserving rendering quality while encoding the signal. Why: This separation makes watermarking controllable and interpretable, sustains high bit accuracy under image-space and model-space attacks, and yields per Gaussian attribution for auditable verification; the design naturally extends to dynamic scenes and multimodal payloads.

Limitations. Our frequency decoupled training requires careful tuning of loss weights to balance quality and robustness; we provide empirically validated defaults, but extreme configurations can degrade performance. In addition, as with other neural watermarking methods, the approach depends on a pretrained decoder, and its robustness bounds the overall system performance.

## Acknowledgements

This work was supported by JST BOOST (Japan), Grant Number JPMJBS2429.

## References

[1] Mauro Barni, Franco Bartolini, and Alessandro Piva. Improved wavelet-based watermarking through pixel-wise masking. IEEE transactions on image processing, 10(5): 783â791, 2001. 2

[2] Yue Chen, Xingyu Chen, Anpei Chen, Gerard Pons-Moll, and Yuliang Xiu. Feat2gs: Probing visual foundation models with gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6348â6361, 2025. 2

[3] Zixuan Chen, Guangcong Wang, Jiahao Zhu, Jianhuang Lai, and Xiaohua Xie. Guardsplat: Efficient and robust watermarking for 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16325â16335, 2025. 1, 2, 3, 6, 7

[4] Yuanyuan Gao, Hao Li, Jiaqi Chen, Zhengyu Zou, Zhihang Zhong, Dingwen Zhang, Xiao Sun, and Junwei Han. Citygsx: A scalable architecture for efficient and geometrically accurate large-scale scene reconstruction, 2025. 2

[5] Georg Hess, Carl Lindstrom, Maryam Fatemi, Christoffer Â¨ Petersson, and Lennart Svensson. Splatad: Real-time lidar and camera rendering with 3d gaussian splatting for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11982â11992, 2025. 1

[6] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4220â4230, 2024. 2

[7] Youngdong Jang, Dong In Lee, MinHyuk Jang, Jong Wook Kim, Feng Yang, and Sangpil Kim. Waterf: Robust watermarks in radiance fields for protection of copyrights. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12087â12097, 2024. 1, 2, 3, 5, 6, 7

[8] Youngdong Jang, Hyunje Park, Feng Yang, Heeju Ko, Euijin Choo, and Sangpil Kim. 3d-gsw: 3d gaussian splatting for robust watermarking. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5938â5948, 2025. 1, 2, 3, 6, 7

[9] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5322â5332, 2024. 2

[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 3, 5, 6, 7

[11] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. 6

[12] Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel, Oncel Tuzel, and Anurag Ranjan. Hugs: Human gaussian splats. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 505â515, 2024.

[13] Chenxin Li, Brandon Y. Feng, Zhiwen Fan, Panwang Pan, and Zhangyang Wang. Steganerf: Embedding invisible information within neural radiance fields. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 441â453, 2023. 3

[14] Wanhua Li, Yujie Zhao, Minghan Qin, Yang Liu, Yuanhao Cai, Chuang Gan, and Hanspeter Pfister. Langsplatv2: Highdimensional 3d language gaussian splatting with 450+ fps. arXiv preprint arXiv:2507.07136, 2025. 1

[15] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19711â19722, 2024. 2

[16] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, et al. Vastgaussian: Vast 3d gaussians for large scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5166â5175, 2024. 2

[17] Jinfeng Liu, Lingtong Kong, Bo Li, and Dan Xu. Gausshdr: High dynamic range gaussian splatting via learning unified 3d and 2d local tone mapping. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5991â6000, 2025. 2

[18] Ange Lou, Benjamin Planche, Zhongpai Gao, Yamin Li, Tianyu Luan, Hao Ding, Terrence Chen, Jack Noble, and Ziyan Wu. Darenerf: Direction-aware representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5031â 5042, 2024. 2

[19] Ziyuan Luo, Qing Guo, Ka Chun Cheung, Simon See, and Renjie Wan. Copyrnerf: Protecting the copyright of neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22401â22411, 2023. 3

[20] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG), 2019. 5

[21] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), pages 405â421. Springer, 2020. 5

[22] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 5

[23] Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, Peter Hedman, Ricardo Martin-Brualla, and Jonathan T. Barron.

MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and RawNeRF, 2022. 5

[24] S Kother Mohideen, S Arumuga Perumal, and M Mohamed Sathik. Image de-noising using discrete wavelet transform. International Journal of Computer Science and Network Security, 8(1):213â216, 2008. 2

[25] Hung Nguyen, Blark Runfa Li, and Truong Nguyen. Dwtnerf: Boosting few-shot neural radiance fields via discrete wavelet transform. arXiv preprint arXiv:2501.12637, 2025. 2

[26] Hung Nguyen, Runfa Li, An Le, and Truong Nguyen. Dwtgs: Rethinking frequency regularization for sparse-view 3d gaussian splatting. arXiv preprint arXiv:2507.15690, 2025. 2

[27] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian: Structural regularization for sparse-view gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21600â21609, 2025. 2

[28] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051â20060, 2024. 1

[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 3

[30] Mehul S Raval and Priti P Rege. Discrete wavelet transform based multiple watermarking scheme. In TENCON 2003. Conference on Convergent Technologies for Asia-Pacific Region, pages 935â938. IEEE, 2003. 2

[31] Daniel Rho, Byeonghyeon Lee, Seungtae Nam, Joo Chan Lee, Jong Hwan Ko, and Eunbyung Park. Masked wavelet representation for compact neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20680â20690, 2023. 2

[32] Fei Shen and Jinhui Tang. Imagpose: A unified conditional framework for pose-guided person generation. Advances in neural information processing systems, 37:6246â 6266, 2024. 2

[33] Fei Shen, Xin Jiang, Xin He, Hu Ye, Cong Wang, Xiaoyu Du, Zechao Li, and Jinhui Tang. Imagdressing-v1: Customizable virtual dressing. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6795â6804, 2025. 2

[34] Fei Shen, Jian Yu, Cong Wang, Xin Jiang, Xiaoyu Du, and Jinhui Tang. Imaggarment-1: Fine-grained garment generation for controllable fashion design. arXiv preprint arXiv:2504.13176, 2025. 2

[35] Mark J Shensa. The discrete wavelet transform: wedding the a trous and mallat algorithms. IEEE Transactions on signal processing, 40(10):2464â2482, 2002. 2

[36] Qi Song, Ziyuan Luo, Ka Chun Cheung, Simon See, and Renjie Wan. Protecting nerfsâ copyright via plug-and-play watermarking base model. In ECCV, 2024. 3

[37] Chinmay Talegaonkar, Yash Belhe, Ravi Ramamoorthi, and Nicholas Antipa. Volumetrically consistent 3d gaussian ras-

terization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10953â10963, 2025. 2

[38] Peining Tao and Ahmet M Eskicioglu. A robust multiple watermarking scheme in the discrete wavelet transform domain. In Internet Multimedia Management Systems V, pages 133â144. SPIE, 2004. 2

[39] Hanzhang Tu, Zhanfeng Liao, Boyao Zhou, Shunyuan Zheng, Xilong Zhou, Liuxin Zhang, QianYing Wang, and Yebin Liu. Gbc-splat: Generalizable gaussian-based clothed human digitalization under sparse rgb cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26377â26387, 2025. 1

[40] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600â612, 2004. 6

[41] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310â20320, 2024. 2

[42] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16453â16463, 2025. 2

[43] Muyu Xu, Fangneng Zhan, Jiahui Zhang, Yingchen Yu, Xiaoqin Zhang, Christian Theobalt, Ling Shao, and Shijian Lu. Wavenerf: Wavelet-based generalizable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18195â18204, 2023. 2

[44] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee. Multi-scale 3d gaussian splatting for anti-aliased rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20923â20931, 2024. 2

[45] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5, 6

[46] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21936â21947, 2025. 2

[47] Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. Hidden: Hiding data with deep networks. In Proceedings of the European conference on computer vision (ECCV), pages 657â672, 2018. 3, 6

[48] Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, and Jiwen Lu. Gaussianworld: Gaussian world model for streaming 3d occupancy prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6772â6781, 2025. 1