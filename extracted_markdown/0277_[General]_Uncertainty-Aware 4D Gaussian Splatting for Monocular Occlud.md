# Uncertainty-Aware 4D Gaussian Splatting for Monocular Occluded Human Rendering

Weiquan Wang1, Feifei Shao1, Lin Li2, Zhen Wang2, Jun Xiao1â , and Long Chen2

1 Zhejiang University 2 The Hong Kong University of Science and Technology wqwangcs@zju.edu.cn; junx@cs.zju.edu.cn

Abstract. High-fidelity rendering of dynamic humans from monocular videos typically degrades catastrophically under occlusions. Existing solutions incorporate external priorsâeither hallucinating missing content via generative models, which induces severe temporal flickering, or imposing rigid geometric heuristics that fail to capture diverse appearances. To this end, we reformulate the task as a Maximum A Posteriori estimation problem under heteroscedastic observation noise. In this paper, we propose U-4DGS, a framework integrating a Probabilistic Deformation Network and a Double Rasterization pipeline. This architecture renders pixel-aligned uncertainty maps that act as an adaptive gradient modulator, automatically attenuating artifacts from unreliable observations. Furthermore, to prevent geometric drift in regions lacking reliable visual cues, we enforce Confidence-Aware Regularizations, which leverage the learned uncertainty to selectively propagate spatial-temporal validity. Extensive experiments on ZJU-MoCap and OcMotion demonstrate that U-4DGS achieves SOTA rendering fidelity and robustness.

Keywords: Occluded Human Rendering Â· 4D Gaussian Splatting Â· Uncertainty Modeling

## 1 Introduction

High-fidelity reconstruction of dynamic humans from monocular videos stands as a foundational technology for immersive applications, spanning telepresence, metaverse interaction, sports analysis, and virtual reality [31, 38]. Recent years have witnessed remarkable progress in this domain, enabling the synthesis of photorealistic digital avatars from monocular inputs [33,34]. However, these successes hinge on the idealized assumption of full visibility and isolation from the background. This assumption rarely holds in unconstrained in-the-wild scenarios, where subjects inevitably interact with their surroundings, such as sitting on chairs, navigating behind obstacles, or being partially obscured by environmental objects. When standard pipelines are applied to occluded scenarios, their performance degrades catastrophically [28, 37]. As illustrated in Fig. 1(e), the model erroneously interprets the occluder as part of the human geometry, manifesting as incomplete geometry or severe texture artifacts. Consequently, enabling robust human rendering from occluded observations is imperative to bridge the gap between laboratory prototypes and real-world applications.

<!-- image-->  
Fig. 1: Performance, Fidelity, and Stability. (a) Our U-4DGS achieves the best trade-off between rendering quality and training efficiency. (b) Visualizes the occluded input and the corresponding uncertainty map predicted by our method. (c)-(f) Visual comparison. While our method (d) recovers fine details consistent with the Reference (c), Gauhuman (e) fails catastrophically, fusing occlusion artifacts into the body. SymGaussian (f ) fails to recover asymmetric details (see red circle), as the model erroneously propagates features from the unadorned opposite side. (g) Temporal consistency. GTU suffers from severe texture drifting, where the shirt color inconsistently shifts over time. In contrast, our uncertainty-guided aggregation ensures physical consistency.

To address these specific challenges, pioneering NeRF-based approaches, such as OccNeRF [37] and Wild2Avatar [36], explore the task of rendering humans from object-occluded monocular videos. By incorporating geometry priors or decoupling scene components, these methods demonstrate the feasibility of recovering occluded surfaces. However, despite their visual fidelity, they inherit the intrinsic computational bottlenecks of implicit neural representations. Specifically, the reliance on expensive volumetric rendering incurs prohibitive training and inference costs, effectively precluding their deployment in real-time applications.

Recently, the advent of 3D Gaussian Splatting (3DGS) [13] shifted the paradigm, offering real-time rendering capabilities [4, 5]. To adapt 3DGS for occluded human rendering, concurrent works incorporate external priors to compensate for missing information. Methods relying on geometric heuristics enforce rigid rules, such as left-right symmetry [12] or feature aggregation [40]. However, these handcrafted spatial priors typically lack flexibility in modeling diverse human appearances. As shown in Fig. 1(f), the symmetry prior fails to recover the chest logo, erroneously copying unadorned features from the opposite side. Alternatively, methods leveraging generative priors utilize 2D diffusion models to hallucinate invisible regions [16,28]. Yet, the inherent stochasticity of frame-by-frame generation often induces severe temporal flickering. This is evident in Fig. 1(g), where the shirt color drifts inconsistently across frames, violating physical consistency. Crucially, these methods rely on imperfect external priors to infer missing con-

tent, neglecting the intrinsic reliability of the input data, where valid cues are embedded across the temporal sequence.

To bridge this gap, our insight stems from the videoâs intrinsic temporal redundancy: since occlusions are transient, parts obscured in one frame are likely visible in others. This motivates adopting a 4D representation to consolidate fragmented observations into a unified canonical space. However, naive 4D Gaussian Splatting faces a supervision dilemma [19, 35]. Indiscriminately treating all pixels as valid constraints forces the deformation field to establish erroneous correspondences, compelling the human geometry to adhere to the occluderâs surface texture. Conversely, strictly limiting supervision to visible regions leaves the occluded parts mathematically unconstrained; without gradient feedback, the geometry in these blind spots fails to densify or maintain coherence, resulting in incomplete structures. Thus, the reconstruction process demands a dynamic discrimination mechanism to assess the validity of the observation, differentiating between visible human details and misleading observational noise (i.e., occluded regions).

To this end, we reformulate the task as a Maximum A Posteriori (MAP) estimation problem under heteroscedastic observation noise. Specifically, instead of assuming constant variance across all observations, we model the likelihood of each pixel as a Laplacian distribution with a learnable scale. This formulation naturally derives a probabilistic objective where the predicted scaleârepresenting the aleatoric uncertaintyâemerges as an inverse weighting term in the loss function. Consequently, this uncertainty functions as an adaptive gradient modulator : low uncertainty permits RGB supervision to dominate the optimization for fine-grained refinement, while high uncertainty automatically attenuates the gradients from unreliable observations. As visualized in Fig. 1(b), the network explicitly captures this reliability distribution, autonomously assigning high uncertainty (bright regions) to occluded areas.

Building upon this formulation, we present U-4DGS, a robust uncertaintyaware framework for monocular occluded human rendering. To parameterize the heteroscedastic noise model, we propose a Probabilistic Deformation Network. Beyond standard geometric warping, this module explicitly predicts the per-primitive aleatoric uncertainty, serving as the confidence metric for our MAP objective. To enable the adaptive gradient modulation, we introduce a Double Rasterization pipeline. By rendering pixel-aligned uncertainty maps alongside color, this mechanism enforces the inverse weighting scheme, ensuring that gradients from occluded regions are effectively attenuated. Finally, to prevent geometric drift in areas lacking visual cues (i.e., occluded regions), we design Confidence-Aware Regularizations. These constraints leverage the learned uncertainty to selectively propagate validity from confident regions to unreliable areas, ensuring coherent completion. Consequently, as illustrated in Fig. 1(a), U-4DGS achieves high-fidelity rendering even under severe occlusion while maintaining training efficiency.

In summary, our main contributions are as follows:

We reformulate occluded human rendering as a MAP estimation problem. By modeling heteroscedastic observation noise, we equip the optimization with a mechanism to discriminate between valid supervision and occlusion artifacts.

â We propose U-4DGS, a framework integrating a Probabilistic Deformation Network with a Double Rasterization pipeline. This design renders pixelaligned uncertainty maps that act as adaptive gradient modulators, effectively shielding the canonical geometry from corruption.

â We introduce Confidence-Aware Regularizations to prevent geometric drift in regions lacking reliable visual cues. These constraints leverage learned uncertainty to enforce spatial-temporal consistency, ensuring coherent completion.

We demonstrate that U-4DGS achieves state-of-the-art rendering fidelity on ZJU-Mocap and OcMotion datasets, successfully recovering clean avatars from severe occlusions.

## 2 Related Works

3D Human Avatar Reconstruction. Traditional human reconstruction methods typically rely on dense camera arrays or depth sensors, limiting their application in in-the-wild scenarios [2, 11, 27, 41]. To address this, NeRF-based methods have enabled high-fidelity reconstruction from monocular inputs by conditioning radiance fields on SMPL priors [18, 30, 42]. However, the reliance on volumetric ray-marching incurs prohibitive computational costs, precluding real-time applicability. Recently, 3D Gaussian Splatting (3DGS) [13] has introduced a paradigm shift with its real-time rasterization capabilities. State-of-theart methods [9, 15, 17, 24, 26, 45] adapt this representation to dynamic humans by representing the body in a canonical space and transforming it via Linear Blend Skinning. Critically, while performant under full visibility, these standard pipelines implicitly assume valid supervision for all pixels. Thus, they lack mechanisms to distinguish the human subject from occlusions, leading to catastrophic degradation where occluder artifacts are erroneously fused into the avatar geometry.

Occlusion-Aware Human Rendering. Existing methodologies generally fall into three paradigms. Early strategies focused on scene decoupling, employing visibility priors to isolate the subject from obstacles [36, 37]; however, these implicit frameworks often suffer from high computational overhead. To synthesize unobserved content, recent approaches leverage generative priors via pretrained 2D diffusion models [3, 16, 28]. Yet, the inherent stochasticity of such processes frequently induces identity drift and temporal flickering. Alternatively, other 3DGS-based methods employ geometric heuristics such as symmetry constraints [12, 40], which often falter under asymmetric clothing or motions. Distinctively, U-4DGS eschews unstable hallucinations and rigid heuristics. Instead, we formulate the reconstruction as an MAP estimation problem. By strictly aggregating valid temporal information via an adaptive gradient modulation mechanism, we ensure reconstruction that is both physically consistent and faithful to the subject.

Dynamic Reconstruction and Uncertainty Modeling. Recent strides in dynamic Gaussian Splatting have achieved high-fidelity 4D reconstruction via deformation fields [1,21,35,39,44]. However, these methods predicate their success on a deterministic correspondence between canonical and observation spaces. While valid for clean sequences, this premise disintegrates under occlusion, where the absence of valid motion cues causes the solver to overfit to occluders [7]. To address this, uncertainty modeling has been explored to enhance robustness against sparse views or transient noise [6, 25, 29]. Notably, concurrent work like USPLAT4D [7] employs an uncertainty-aware motion graph for general dynamic scenes. Yet, treating uncertainty solely as a smoothing weight or edge attribute proves insufficient for articulated human reconstruction, where structural integrity is paramount. In contrast, U-4DGS elevates uncertainty to a central arbitration role. Beyond simple loss attenuation, we leverage learned confidence to actively modulate optimization dynamics: shielding canonical geometry from misleading gradients while simultaneously triggering confidence-aware regularizations to enforce physical plausibility within regions lacking reliable visual cues.

## 3 Preliminaries

## 3.1 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) [13] explicitly models the scene using a set of anisotropic 3D Gaussians. Each primitive i is parameterized by a mean $\mu _ { i } \in { \mathbb { R } } ^ { 3 }$ covariance $\Sigma _ { i } ,$ opacity $\alpha _ { i } \in [ 0 , 1 ]$ ], and view-dependent color $\mathbf { c } _ { i }$ (via Spherical Harmonics). To enforce positive semi-definiteness during differentiable optimization, the covariance $\Sigma _ { i }$ is decomposed into a rotation quaternion $\mathbf { q } _ { i } \in \mathbb { R } ^ { 4 }$ and a scaling vector $\mathbf { s } _ { i } \in \mathbb { R } ^ { 3 }$

$$
\begin{array} { r } { \pmb { \Sigma } _ { i } = \pmb { \mathrm { R } } ( \pmb { \mathrm { q } } _ { i } ) \pmb { \mathrm { S } } ( \pmb { \mathrm { s } } _ { i } ) \pmb { \mathrm { S } } ( \pmb { \mathrm { s } } _ { i } ) ^ { \top } \pmb { \mathrm { R } } ( \pmb { \mathrm { q } } _ { i } ) ^ { \top } , } \end{array}\tag{1}
$$

where R and S denote the rotation and scaling matrices, respectively.

For rendering, Gaussians are projected onto the image plane to compute the pixel color C(u) via differentiable front-to-back Î±-blending of N sorted primitives:

$$
\mathbf { C } ( \mathbf { u } ) = \sum _ { i \in \mathcal { N } } T _ { i } \alpha _ { i } G _ { i } ^ { 2 D } ( \mathbf { u } ) \mathbf { c } _ { i } , \quad \mathrm { w i t h ~ } T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } G _ { j } ^ { 2 D } ( \mathbf { u } ) ) ,\tag{2}
$$

where $G _ { i } ^ { 2 D } ( { \mathbf { u } } )$ represents the probability density of the projected 2D Gaussian at pixel u, and $T _ { i }$ denotes the accumulated transmittance.

## 3.2 Articulated Human Gaussians

Following recent works [9, 24], we leverage the SMPL model [20] to establish a canonical coordinate system. SMPL provides a skinned vertex mesh parametrized by pose $\theta \in \mathbb { R } ^ { 7 2 }$ and shape $\beta \in \mathbb { R } ^ { 1 0 }$ . To model dynamic humans, we instantiate 3D Gaussians within this canonical T-pose space. For a specific frame t with pose $\theta _ { t } ,$ any canonical point $\mathbf { x } _ { c a n }$ is transformed to the observation space $\mathbf { x } _ { o b s }$ via Linear Blend Skinning:

$$
\mathbf { x } _ { o b s } = W ( \mathbf { x } _ { c a n } , \mathbf { J } ( \beta ) , \theta _ { t } , \mathcal { W } ) ,\tag{3}
$$

where $W ( \cdot )$ denotes the LBS function, J represents the joint locations, and W contains the blend weights. In U-4DGS, we bind Gaussian primitives to their nearest SMPL vertices. This strategy ensures that coarse body movements are explicitly driven by skeletal articulation, while a learned neural field subsequently models residual non-rigid deformations (e.g., clothing dynamics).

## 3.3 Problem Formulation: A Probabilistic Perspective

Given a monocular video sequence ${ \boldsymbol \nu } = \{ { \bf I } _ { t } \} _ { t = 1 } ^ { T }$ capturing a dynamic human, our objective is to reconstruct a high-fidelity human avatar. We assume access to camera parameters and coarse SMPL tracking. From a probabilistic standpoint, standard 3DGS optimization can be viewed as a Maximum Likelihood Estimation (MLE) problem. Minimizing the standard photometric loss $( \mathrm { e . g . } , \mathcal { L } _ { 1 }$ in Eq. 2) implicitly assumes that the observation noise across all pixels follows a fixed, homoscedastic distribution (i.e., constant variance). While effective for clean sequences, this assumption disintegrates under environmental occlusion. In such scenarios, the observation noise is inherently heteroscedastic: visible regions exhibit low noise, whereas occluded regions suffer from gross corruption. Applying a homoscedastic MLE objective to such data forces the model to overfit these outliers, compelling the human geometry to adhere to the occluderâs surface erroneously.

To resolve this, we move beyond deterministic fitting and reformulate the reconstruction as a Maximum A Posteriori (MAP) estimation problem. Our goal is to jointly estimate the canonical parameters Î and the per-pixel aleatoric uncertainty (representing the noise scale). This formulation allows us to dynamically down-weight unreliable observations while incorporating structural priors to constrain the solution in regions lacking reliable visual cues. We detail this uncertainty-aware framework in Sec. 4.

## 4 Methodology

## 4.1 Overview

Building upon the probabilistic formulation derived in Sec. 3, we propose U-4DGS, a framework designed to realize the MAP estimation for robust occluded human rendering. U-4DGS explicitly models the heteroscedastic nature of observation noise, enabling an autonomous arbitration mechanism between valid supervision and occlusion artifacts. As illustrated in Fig. 2, the pipeline consists of three integral components: (1) Probabilistic Deformation (Sec. 4.2): We introduce a network Î¦ that serves a dual role: it models the temporal geometry evolution via deformation offsets and parameterizes the input-dependent aleatoric uncertainty Ï for each Gaussian, explicitly quantifying the confidence of the temporal correspondence. (2) Double Rasterization (Sec. 4.3): To propagate this 3D confidence to the 2D observation space, we employ a dualbranch rasterizer. This module renders the deformed geometry into a photometric image CË while simultaneously accumulating per-primitive uncertainty into a pixel-aligned Uncertainty Map UË . (3) Optimization via MAP Estimation (Sec. 4.4): These outputs drive our MAP-based objective. Here, UË functions as an adaptive gradient modulator within the probabilistic photometric loss, attenuating gradients from unreliable observations. Simultaneously, to prevent geometric drift in regions lacking reliable visual cues, we enforce Confidence-Aware Regularizations that leverage the learned uncertainty to impose spatial-temporal priors.

<!-- image-->  
Fig. 2: The framework of U-4DGS. Left: The Probabilistic Deformation Network conditions Canonical Gaussians on time embedding $\gamma ( t )$ and pose $\theta _ { t }$ to predict geometric offsets $( \varDelta \mathbf { r } , \varDelta \mu ,$ âs) alongside per-primitive aleatoric uncertainty Ï. Middle: The deformed Gaussians are transformed via LBS and rendered through a Double Rasterization pipeline, simultaneously producing a photometric image and a pixel-aligned Uncertainty Map (where bright regions indicate high uncertainty). Right: During optimization, the Uncertainty Map functions as an adaptive gradient modulator (symbolized by Ã·) in the $\mathcal { L } _ { N L L }$ objective, effectively attenuating gradients from unreliable observations. Simultaneously, Confidence-Aware Regularizations $( \mathcal { L } _ { s p a } , \mathcal { L } _ { t e m p } )$ leverage the learned uncertainty to enforce spatial-temporal constraints in regions lacking reliable visual cues.

## 4.2 Probabilistic Deformation Network

To instantiate the MAP framework, we require a mechanism that not only recovers time-varying geometry but also quantifies the reliability of each correspondence. Unlike standard deterministic fields [9, 39] that indiscriminately fit all observations, we propose a Probabilistic Deformation Network $\varPhi _ { \psi }$ . This module serves a dual purpose: (1) it models high-frequency non-rigid residuals beyond the SMPL prior; (2) it explicitly predicts the aleatoric uncertainty $\sigma ,$ providing the input-dependent noise scale required for our probabilistic objective.

We parameterize $\varPhi _ { \psi }$ as a Multi-Layer Perceptron (MLP). To capture highfrequency details in both spatial and temporal domains, we apply sinusoidal positional encoding $\gamma ( \cdot ) = \bigl ( \sin ( 2 ^ { k } \pi \cdot ) , \cos ( 2 ^ { k } \pi \cdot ) \bigr ) _ { k = 0 } ^ { L - 1 }$ to both the input coordinates and time. The network conditions the canonical Gaussians on these embeddings and the 3D skeleton pose $\theta _ { t }$ to predict geometry offsets and uncertainty:

$$
( \varDelta \mu , \varDelta \mathbf { r } , \varDelta \mathbf { s } , \sigma ) = \varPhi _ { \psi } ( \gamma ( \mathrm { s g } ( \mu _ { c a n } ) ) , \gamma ( t ) , \theta _ { t } ) ,\tag{4}
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. The offsets $( \varDelta \mu , \varDelta \mathbf { r } , \varDelta \mathbf { s } )$ explicitly model the time-varying geometry, enabling the fusion of features across frames. For the uncertainty $\sigma _ { \mathrm { { : } } }$ we employ a Softplus activation to ensure strictly positive values $( \sigma \in \mathbb { R } ^ { + } )$ ). We then apply the predicted offsets to the canonical Gaussians to obtain the deformed state:

$$
\mu _ { d e f } = \mu _ { c a n } + \varDelta \mu , \quad \mathbf { s } _ { d e f } = \mathbf { s } _ { c a n } + \varDelta \mathbf { s } , \quad \mathbf { q } _ { d e f } = \mathbf { q } _ { c a n } \cdot \varDelta \mathbf { r } .\tag{5}
$$

These deformed primitives are subsequently transformed to the observation space via LBS (Eq. 3).

Crucially, since Ï is predicted in the canonical space, it is defined as an intrinsic property of the Gaussian primitive. It remains bound to the surface during the LBS transformation, invariant to the global body movement. This design allows the network to learn specific spatial-temporal failure modes, such as assigning consistently high uncertainty to body parts undergoing rapid deformation or frequent self-occlusion, regardless of their global position in the camera view.

## 4.3 Double Rasterization

To bridge the gap between the 3D aleatoric uncertainty Ï and the 2D observation space required for our MAP objective, we propose a Double Rasterization scheme. Since occlusion is an inherently view-dependent phenomenon, the reliability of a pixel is determined by the accumulated uncertainty along its corresponding optical ray. Therefore, we perform two pixel-aligned rasterization passes in parallel, generating both the photometric appearance CË and the uncertainty map UË .

The first pass renders the predicted color image CË using the standard splatting formulation (Eq. 2). Crucially, this pass utilizes the deformed geometry derived from the Probabilistic Deformation Network, ensuring that the rendering faithfully reflects the current temporal dynamics.

The second pass renders the uncertainty map $\hat { U } \in \mathbb { R } ^ { H \times W }$ . To ensure strict geometric consistency between the rendered appearance and its associated confidence, we accumulate the uncertainty using the same Î±-blending weights used for color:

$$
{ \hat { U } } ( \mathbf { u } ) = \sum _ { i \in \mathcal { N } } T _ { i } \alpha _ { i } \sigma _ { i } ,\tag{6}
$$

where $T _ { i }$ and $\alpha _ { i }$ are identical to those computed in the color pass. This formulation has a clear physical interpretation: $\hat { U } ( { \bf u } )$ represents the expected aleatoric uncertainty of the visible surface at pixel u. By compositing Ï with the opacity $\alpha ,$ , the resulting map accurately reflects the confidence of the foremost visible surface, automatically disregarding occluded primitives hidden behind. Consequently, $\hat { U }$ serves as the spatially varying noise scale, functioning as the adaptive gradient modulator in the subsequent optimization.

## 4.4 Optimization via MAP Estimation

To reconstruct high-fidelity humans from monocular videos, we jointly optimize the canonical Gaussian parameters Î and the deformation network weights $\psi .$ Following the MAP formulation introduced in Sec. 3, our objective function maximizes the posterior $P ( \Theta | \mathcal { D } ) \propto P ( \mathcal { D } | \Theta ) P ( \Theta )$ . This decomposes the optimization into two synergistic components: a Likelihood term $P ( \mathcal { D } | \theta )$ that models data fidelity under heteroscedastic noise, and a Prior term $P ( \Theta )$ that enforces physical constraints via Confidence-Aware Regularizations.

Uncertainty-Weighted Photometric Loss $\big ( P ( \mathcal { D } | \theta ) \big )$ . The primary challenge in occluded rendering is the mismatch between the reconstructed geometry and occluder-corrupted observations. Standard objectives (e.g., L1 loss) assume homoscedastic noise, treating all pixels equally and thus forcing the model to fit occlusion artifacts. To resolve this, we model the observation likelihood $P ( \mathcal { D } | \theta )$ by assuming the pixel-wise residual follows a Laplacian distribution, where the location parameter is the predicted color $\hat { \mathbf { C } } ( \mathbf { u } )$ and the scale parameter is the predicted uncertainty $\hat { U } ( { \bf u } )$ . The probability density function is given by $\begin{array} { r } { p ( x | \mu ) = \frac { 1 } { 2 \hat { U } } \exp ( - \frac { | x - \mu | } { \hat { U } } ) } \end{array}$ . Minimizing the negative log-likelihood (NLL) of this distribution yields our uncertainty-weighted objective:

$$
\mathcal { L } _ { N L L } = \sum _ { \mathbf { u } \in \varOmega } \left( \frac { \| \mathbf { C } _ { g t } ( \mathbf { u } ) - \hat { \mathbf { C } } ( \mathbf { u } ) \| _ { 1 } } { \hat { U } ( \mathbf { u } ) + \epsilon } + \lambda _ { r e g } \log ( \hat { U } ( \mathbf { u } ) + \epsilon ) \right) ,\tag{7}
$$

where $\epsilon = 1 0 ^ { - 7 }$ ensures numerical stability. This objective functions as an adaptive gradient modulator: in visible regions, the solver minimizes the numerator $( \mathcal { L } _ { 1 }$ error) and reduces ${ \hat { U } } ;$ in occluded regions, it autonomously increases $\hat { U }$ to attenuate the gradient magnitude. This mechanism effectively shields the canonical geometry from misleading supervision.

Confidence-Aware Regularizations $( P ( \Theta ) )$ . While the uncertainty-weighted likelihood effectively attenuates gradients from artifacts, it leaves regions lacking reliable visual cues mathematically unconstrained. Without valid supervision, the human Gaussians in these blind spots are prone to the ill-posed geometric drift mentioned in Sec. 1. To resolve this, we impose the prior $P ( \Theta )$ via Confidence-Aware Regularizations. These constraints utilize the learned uncertainty Ï to selectively enforce physical plausibility: applying strong regularization only where observations are unreliable, while allowing data-driven deformation in confident regions.

(1) Confidence-Guided Spatial Consistency: To maintain geometric integrity, we enforce local rigidity weighted by uncertainty. We construct a KNN graph in the canonical space. For each Gaussian i and its neighbor $j \in \mathcal { N } _ { c a n } ( i )$ , we constrain their deformation attributes to be consistent:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { s p a } = \sum _ { i } \mathrm { s g } ( \sigma _ { i } ) \sum _ { j } \omega _ { i j } \Big ( \| \boldsymbol { \Delta \mu } _ { i } - \boldsymbol { \Delta \mu } _ { j } \| _ { 2 } + \lambda _ { r o t } \| \boldsymbol { \Delta \mathbf { r } } _ { i } - \boldsymbol { \Delta \mathbf { r } } _ { j } \| _ { 2 } } } \\ { ~ + ~ \lambda _ { s c l } \| \boldsymbol { \Delta \mathbf { s } } _ { i } - \boldsymbol { \Delta \mathbf { s } } _ { j } \| _ { 2 } \Big ) . }  \end{array}\tag{8}
$$

Here, $\omega _ { i j }$ weights neighbors by canonical distance. The term $\operatorname { s g } ( \sigma _ { i } )$ acts as an attention mechanism: When a Gaussian is occluded $( \sigma _ { i }$ is high), the $\mathcal { L } _ { N L L }$ gradients vanish, and $\mathcal { L } _ { s p a }$ penalty dominates, forcing the Gaussian to synchronize with its neighbors. Crucially, since visible neighbors are firmly anchored by the photometric loss, this constraint effectively propagates geometric validity from the confident boundary inward to the occluded regions.

(2) Uncertainty-Weighted Temporal Inertia: To prevent jittering in the absence of visual cues, we enforce a smoothness prior on the trajectory of uncertain Gaussians. Considering that we train on monocular video sequences with a frame interval k, we formulate this constraint as a second-order difference over the sampled timestamps:

$$
\mathcal { L } _ { t e m p } = \frac { 1 } { | \mathcal { T } ^ { \prime } | } \sum _ { t \in \mathcal { T } ^ { \prime } } \sum _ { i } \operatorname { s g } ( \sigma _ { i , t } ) \Vert \mathcal { F } _ { i , t - k } - 2 \mathcal { F } _ { i , t } + \mathcal { F } _ { i , t + k } \Vert _ { 2 } ,\tag{9}
$$

where $\mathcal { F } = \{ \varDelta \mu , \varDelta \mathbf { r } , \varDelta \mathbf { s } \}$ denotes the set of deformation parameters, and $\tau ^ { \prime }$ represents the set of valid frames with available temporal neighbors. This term acts as an inertial prior. For high-uncertainty Gaussians, it penalizes acceleration, effectively interpolating their dynamics based on the trajectory established by confident frames, ensuring smooth motion transition across occlusion gaps.

Note that in both $\mathcal { L } _ { s p a }$ and $\mathcal { L } _ { t e m p }$ , we apply a stop-gradient operator sg(Â·) to the uncertainty weights $\sigma$ . This ensures that Ï serves purely as an attention mechanism for the regularization strength, preventing the optimizer from trivially reducing Ï to minimize the regularization penalty.

Total Loss. The final objective function consolidates the MAP formulation, combining the uncertainty-weighted likelihood (data fidelity), the confidenceaware priors (regularization), and auxiliary perceptual constraints:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { n l l } + \lambda _ { s p a } \mathcal { L } _ { s p a } + \lambda _ { t e m p } \mathcal { L } _ { t e m p } + \lambda _ { m a s k } \mathcal { L } _ { m a s k } + \mathcal { L } _ { i m g } ,\tag{10}
$$

where $\mathcal { L } _ { i m g } = \lambda _ { \mathrm { s s i m } } \mathcal { L } _ { \mathrm { s s i m } } + \lambda _ { \mathrm { l p i p s } } \mathcal { L } _ { \mathrm { l p i p s } }$ ensures perceptual quality using standard metrics [32, 43]. Additionally, $\mathcal { L } _ { m a s k } = \Vert \hat { \mathbf { O } } - \mathbf { M } _ { s m p l } \Vert _ { 2 }$ imposes a global

Table 1: Quantitative comparison on ZJU-MoCap and OcMotion datasets. LPIPS values are scaled by Ã1000. The best results are bolded and the second-best results are underlined.
<table><tr><td>Method</td><td>Category</td><td>ZJU-MoCap PSNR â SSIM â LPIPS â</td><td></td><td>OcMotion PSNR â SSIM â LPIPS â</td></tr><tr><td colspan="5">Standard Human Rendering</td></tr><tr><td>HumanNeRF [34]</td><td></td><td>20.67 0.9509</td><td>-</td><td>-</td></tr><tr><td>GaussianAvatar [8]</td><td></td><td>18.01 0.9512</td><td>60.33 - 55.88</td><td>- - -</td></tr><tr><td>GauHuman [9]</td><td></td><td>21.55 0.9430</td><td>15.09</td><td>0.8525 107.1</td></tr><tr><td colspan="5">Occlusion-Aware Approaches</td></tr><tr><td>OccNeRF [37] Wild2Avatar [36]</td><td>Scene Decoupling</td><td>22.40 0.9562 43.01 - - -</td><td>15.71 14.09</td><td>0.8523 82.90 0.8484 93.31</td></tr><tr><td>OccGaussian [40] SymGaussian [12]</td><td>Geometric Heristics</td><td>23.29 0.9482 23.22 0.9535</td><td>41.93 - 39.02 -</td><td>-</td></tr><tr><td>GTU [16]</td><td>Generative Priors</td><td>22.89 0.9503 40.78</td><td>- 15.83 0.8437</td><td>- 83.46</td></tr><tr><td>OccFusion [28]</td><td></td><td>23.96 0.9548</td><td>32.34 18.28</td><td>0.8875 82.42</td></tr><tr><td colspan="2">U-4DGS (Ours)</td><td>24.62 0.9606 31.72</td><td>20.11 0.9030</td><td>79.17</td></tr></table>

silhouette constraint, aligning the accumulated opacity $\hat { \bf O }$ with the projected SMPL mask $\mathbf { M } _ { s m p l }$ to guide coarse geometry. The hyperparameters Î» balance the contribution of each term.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. We evaluate U-4DGS on two standard benchmarks representing both synthetic and real-world occlusion scenarios. First, we utilize the ZJU-MoCap dataset [23], which contains multi-view sequences of 6 subjects. Following the protocol of OccNeRF [37], we simulate occlusions by masking the central 50% of the human region in the first 80% of the frames. We adopt a monocular setting for training, using only Camera 1 with 100 frames subsampled at an interval of 5, while the remaining 22 cameras are reserved for novel view evaluation. Second, to assess robustness in natural environments, we employ the OcMotion dataset [10], which features varying degrees of real-world human-object interactions. Consistent with prior works [28, 36], we select 6 representative sequences and train on sparse sub-sequences of 50 frames to challenge the modelâs capability under limited observations.

Baselines. We benchmark our method against comprehensive state-of-the-art approaches, categorized into two distinct groups: (1) Standard Human Rendering. To demonstrate the impact of occlusion on conventional pipelines, we evaluate HumanNeRF [34], GaussianAvatar [8], and GauHuman [9]. These methods are trained directly on the occluded sequences without specific handling

<!-- image-->

<!-- image-->  
Input View  
GH  
OF  
Ours  
Ref. View  
Input View  
GH  
OF  
Ours  
Ref. View

Fig. 3: Qualitative comparisons on novel view synthesis. Left: Results on the ZJU-MoCap dataset with synthetic occlusions. Right: Results on the OcMotion dataset with real-world occlusions. GH denotes GauHuman [9] and OF denotes Occ-Fusion [28]. GauHuman fails to disentangle occluders from the human body. OccFusion tends to produce blurry textures or hallucination artifacts in heavily occluded regions. Our U-4DGS recovers high-fidelity geometry and appearance consistent with the reference view.

mechanisms. (2) Occlusion-Aware Approaches. We compare against representative methods covering three mainstream technical paradigms: a) Scene Decoupling: OccNeRF [37] and Wild2Avatar [36], which utilize implicit fields or layer-wise parameterization to separate the human subject from environmental obstacles. b) Geometric Heuristics: OccGaussian [40] and SymGaussian [12], which rely on rigid priors such as feature aggregation or left-right symmetry to infer missing geometry. c) Generative Priors: OccFusion [28] and GTU [16], which leverage 2D diffusion models to hallucinate the unobserved regions. For a fair comparison, all methods are evaluated using identical segmentation masks and SMPL pose priors.

Implementation Details: Our framework is implemented in PyTorch [22] and trained on a single NVIDIA A100 PCIe GPU. The Probabilistic Deformation

Network is instantiated as an 8-layer MLP (hidden dimension 256) with a skip connection at the 4-th layer. To capture high-frequency spatial-temporal dynamics, we apply sinusoidal positional encoding to input coordinates and time with frequency bands $L _ { x y z } = 1 0$ and $L _ { t } = 6$ , respectively.

We optimize the model for 10k iterations using the Adam optimizer [14]. Learning rates for canonical Gaussian attributes follow the schedule of GauHuman [9], while the deformation network utilizes a rate of $1 . 6 \times 1 0 ^ { - 4 }$ . Crucially, to preclude trivial solutions (where the model simply predicts infinite uncertainty to minimize the NLL), we adopt a progressive training strategy. For the initial 2k iterations, we deactivate the uncertainty branch and supervise geometry solely via standard $L _ { 1 }$ loss. Subsequently, we activate the uncertainty head and switch to the MAP objective (Eq. 7) to jointly optimize for reconstruction fidelity and reliability. The regularization weights are set as: $\lambda _ { r o t } = \lambda _ { s c l } = 0 . 5 , \lambda _ { m a s k } = 0 . 1$ , and $\lambda _ { s s i m } = \lambda _ { l p i p s } = \lambda _ { s p a } = \lambda _ { t e m p } = 0 . 0 1$

## 5.2 Comparisons with State-of-the-Arts

Quantitative Results. Tab. 1 reports the performance on ZJU-MoCap and OcMotion benchmarks. On the synthetic ZJU-MoCap dataset, U-4DGS sets a new SOTA, achieving a PSNR of 24.62 dB. Notably, it surpasses the leading generative baseline, OccFusion (23.96 dB), validating that our uncertainty-guided aggregation yields superior fidelity without relying on potentially unstable diffusion priors. Crucially, on the real-world OcMotion dataset, our framework demonstrates exceptional robustness, outperforming the runner-up by a substantial margin of 1.83 dB. Unlike standard rendering or scene decoupling methods that degrade under occlusions, U-4DGS effectively leverages Confidence-Aware Regularizations to shield the canonical geometry, proving that modeling heteroscedastic noise is decisive for handling unconstrained interactions.

Qualitative Results. Fig. 3 visualizes the rendering quality. Standard methods like GauHuman (GH) [9] lack occlusion awareness, erroneously fusing environmental obstacles (e.g., bars or boxes) directly onto the human body. While generative approaches like OccFusion (OF) [28] can remove occluders, they suffer from inherent stochasticity, often yielding over-smoothed textures or hallucinations inconsistent with the subjectâs identity. In contrast, U-4DGS achieves superior fidelity. By functioning as an adaptive gradient modulator, our uncertainty mechanism effectively attenuates gradients from artifacts. By strictly aggregating valid temporal information, our method faithfully restores clean, sharp, and physically consistent avatars, even under severe occlusions.

## 5.3 Ablation and Analysis

Ablation Studies. We validate the efficacy of each component by incrementally integrating it into the baseline to instantiate our complete MAP framework. Quantitative metrics on ZJU-MoCap are reported in Tab. 2, and visual comparisons are presented in Fig. 4. (1) Exp. A (Baseline): We establish a deterministic baseline OccGauHuman, structurally inheriting the pipeline from

Table 2: Ablation study on the ZJU-Mocap dataset. $\mathcal { L } _ { s p a } \mathrm { : }$ : Confidence-guided spatial constraint; $\mathcal { L } _ { t e m p } \colon$ Uncertainty-weighted temporal inertia. constraint
<table><tr><td>Exp.</td><td>Configuration</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS</td></tr><tr><td>A</td><td>Baseline</td><td>22.85</td><td>0.9490</td><td>45.12</td></tr><tr><td>B</td><td>+ Uncertainty Modeling</td><td>24.15</td><td>0.9575</td><td>35.20</td></tr><tr><td>C</td><td>+ Spatial Constraint  $( \mathcal { L } _ { s p a } )$ </td><td>24.48</td><td>0.9592</td><td>32.50</td></tr><tr><td>D</td><td>+ Temporal Constraint  $\left( \mathcal { L } _ { t e m p } \right)$ </td><td>24.62</td><td>0.9606</td><td>31.72</td></tr></table>

GauHuman [9]. It employs a standard deformation field optimized via $L _ { 1 }$ loss, effectively assuming homoscedastic observation noise. As shown in Fig. 4 (Exp. A), limited by this rigid deterministic fitting, the model fails to distinguish between the human body and environmental obstacles. Consequently, it indiscriminately minimizes geometric error, leading to severe artifact adhesion at occlusion boundaries. (2) Exp. B (+ Uncertainty Modeling): Replacing the deterministic field with our Probabilistic Deformation Network and enabling Double Rasterization yields a substantial quality leap. This step introduces the Likelihood term of our MAP objective. By functioning as an adaptive gradient modulator, the learned uncertainty effectively attenuates gradients from unreliable observations, successfully removing occlusion artifacts. However, without regularization priors, the geometry in regions lacking reliable visual cues remains noisy and ill-defined (see Exp. B), as the probabilistic loss solely down-weights error without explicitly guiding completion. (3) Exp. $c _ { \ ( + \ } . _ { \mathscr { L } _ { s p a } } ) { : }$ The introduction of Confidence-Guided Spatial Consistency imposes the first Prior term. By leveraging uncertainty to selectively propagate spatial validity from confident neighbors, this constraint effectively regularizes the noisy geometry, restoring a coherent and complete body shape. (4) Exp. D $( + \mathcal { L } _ { t e m p } ) \colon$ Finally, incorporating Uncertainty-Weighted Temporal Inertia completes the U-4DGS framework. This temporal prior ensures trajectory smoothness and further refines surface details, achieving the most faithful reconstruction with high physical plausibility.

Visualization of Uncertainty Maps. To validate the efficacy of our heteroscedastic noise modeling, we visualize the rendered uncertainty map UË in Fig. 5. In our MAP framework, UË represents the estimated scale of observation noise, tasked with autonomously distinguishing between valid signals and occlusion artifacts. As evident in Fig. 5, the network assigns low uncertainty (depicted in dark purple) to visible body parts, allowing the photometric supervision to drive the optimization in these trusted regions. Crucially, for anatomical regions that are occluded in the current frame but belong to the canonical topology, the network autonomously predicts significantly high uncertainty (depicted in bright yellow). This visualization empirically confirms that UË functions as an adaptive gradient modulator. By automatically increasing the denominator in the NLL objective (Eq. 7), it effectively attenuates gradients in these regions lacking reliable visual cues, preventing the canonical geometry from adhering to environmental occlusions.

<!-- image-->  
Fig. 4: Qualitative ablation study. Exp. A (Baseline): The deterministic baseline overfits the occlusion, resulting in severe artifacts. Exp. B (+ Uncertainty): Explicit uncertainty modeling removes the artifacts but leaves the geometry noisy and ill-defined in the occluded region. Exp. C $( + \mathcal { L } _ { s p a } )$ : Spatial regularization smooths out the noise, restoring a complete body shape. Exp. D $( + \ { \mathcal { L } } _ { t e m p } )$ : The full model further refines details and ensures physical plausibility.

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 5: Visualization of the learned Uncertainty Map. Left: The input image showing only the visible regions of the current frame (occluded regions are black). Right: The predicted uncertainty map UË rendered by our method. As indicated by the color bar, dark blue represents high confidence (low uncertainty), while bright red indicates high uncertainty. The network correctly assigns high uncertainty to the "missing" occluded regions, effectively creating a soft mask to ignore unreliable supervision during training.

Temporal Stability Analysis. To validate the temporal coherence of our framework, we conduct a longitudinal comparison against the generative baseline GTU [16] across a long sequence (Fig. 6). As observed in the top row, GTU suffers from severe identity drift due to the inherent stochasticity of its diffusion priors. Since the method hallucinates missing regions frame-by-frame without strict temporal coupling, the synthesized textures fluctuate drastically. This instability becomes particularly pronounced in later frames (e.g., t = 90 to t = 150), where the clothing consistency collapses, degenerating into severe dark artifacts and incoherent noise on the subjectâs back. In contrast, U-4DGS achieves rigorous stability. By anchoring appearance in a unified canonical space and enforcing Uncertainty-Weighted Temporal Inertia, our method effectively constrains the optimization trajectory. This ensures that the reconstructed human maintains physical consistency and sharp details throughout the sequence, eliminating the texture flickering observed in generative approaches.

<!-- image-->  
Fig. 6: Evaluation of temporal stability. Top: The diffusion-based method GTU [16] suffers from severe texture drift and identity inconsistency. Note how the shirtâs color and pattern hallucinated in the occluded regions fluctuate randomly over time, degenerating into dark noise in the later frames. Bottom: Our U-4DGS maintains superior temporal coherence. By leveraging the canonical representation and temporal constraints, our method recovers a stable and consistent appearance.

## 6 Conclusion

In this paper, we present U-4DGS, a robust framework that reformulates monocular human rendering as an MAP estimation problem to handle severe environmental occlusion. Departing from paradigms that rely on stochastic hallucinations or rigid priors, we explicitly model the heteroscedastic nature of observation noise. By integrating a Probabilistic Deformation Network with a Double Rasterization pipeline, our method establishes an adaptive gradient modulator, enabling the optimization to autonomously distinguish between valid geometric cues and occlusion artifacts. Furthermore, we leverage the learned uncertainty to drive Confidence-Aware Regularizations, preventing geometric drift and ensuring structural integrity in regions lacking reliable visual cues. Extensive evaluations on ZJU-MoCap and OcMotion demonstrate that U-4DGS significantly outperforms state-of-the-art approaches, achieving superior robustness and photorealism.

## References

1. Bae, J., Kim, S., Yun, Y., Lee, H., Bang, G., Uh, Y.: Per-gaussian embeddingbased deformation for deformable 3d gaussian splatting. In: European Conference on Computer Vision. pp. 321â335. Springer (2024)

2. Collet, A., Chuang, M., Sweeney, P., Gillett, D., Evseev, D., Calabrese, D., Hoppe, H., Kirk, A., Sullivan, S.: High-quality streamable free-viewpoint video. ACM Transactions on Graphics (ToG) 34(4), 1â13 (2015)

3. Fan, J., Zhao, S., Zheng, L., Zhang, J., Yang, Y., Gong, M.: Inpainthuman: Reconstructing occluded humans with multi-scale uv mapping and identity-preserving diffusion inpainting. arXiv preprint arXiv:2601.02098 (2026)

4. Fei, B., Xu, J., Zhang, R., Zhou, Q., Yang, W., He, Y.: 3d gaussian splatting as new era: A survey. IEEE Transactions on Visualization and Computer Graphics (2024)

5. Feng, G., Chen, S., Fu, R., Liao, Z., Wang, Y., Liu, T., Hu, B., Xu, L., Pei, Z., Li, H., et al.: Flashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 26652â26662 (2025)

6. Goli, L., Reading, C., SellÃ¡n, S., Jacobson, A., Tagliasacchi, A.: Bayesâ rays: Uncertainty quantification for neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20061â20070 (2024)

7. Guo, F., Hsu, C.C., Ding, S., Zhang, C.: Uncertainty matters in dynamic gaussian splatting for monocular 4d reconstruction. arXiv preprint arXiv:2510.12768 (2025)

8. Hu, L., Zhang, H., Zhang, Y., Zhou, B., Liu, B., Zhang, S., Nie, L.: Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 634â644 (2024)

9. Hu, S., Hu, T., Liu, Z.: Gauhuman: Articulated gaussian splatting from monocular human videos. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20418â20431 (2024)

10. Huang, B., Shu, Y., Ju, J., Wang, Y.: Occluded human body capture with selfsupervised spatial-temporal motion prior. arXiv preprint arXiv:2207.05375 (2022)

11. Huang, Z., Xu, Y., Lassner, C., Li, H., Tung, T.: Arch: Animatable reconstruction of clothed humans. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3093â3102 (2020)

12. Jiang, Z., Duan, T., Zhang, D.: Symgaussian: Occluded human rendering with multi-scale symmetry feature from monocular video. In: ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1â5. IEEE (2025)

13. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139â1 (2023)

14. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Bengio, Y., LeCun, Y. (eds.) 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings (2015), http://arxiv.org/abs/1412.6980

15. Kocabas, M., Chang, J.H.R., Gabriel, J., Tuzel, O., Ranjan, A.: Hugs: Human gaussian splats. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 505â515 (2024)

16. Lee, I., Kim, B., Joo, H.: Guess the unseen: Dynamic 3d scene reconstruction from partial 2d glimpses. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1062â1071 (2024)

17. Lei, J., Wang, Y., Pavlakos, G., Liu, L., Daniilidis, K.: Gart: Gaussian articulated template models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19876â19887 (2024)

18. Li, C., Lin, J., Lee, G.H.: Ghunerf: Generalizable human nerf from a monocular video. In: 2024 International Conference on 3D Vision (3DV). pp. 923â932. IEEE (2024)

19. Li, D., Huang, S.S., Lu, Z., Duan, X., Huang, H.: St-4dgs: Spatial-temporally consistent 4d gaussian splatting for efficient dynamic scene rendering. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1â11 (2024)

20. Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J.: Smpl: a skinned multi-person linear model. ACM Transactions on Graphics (TOG) 34(6), 1â16 (2015)

21. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In: 2024 International Conference on 3D Vision (3DV). pp. 800â809. IEEE (2024)

22. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems 32 (2019)

23. Peng, S., Zhang, Y., Xu, Y., Wang, Q., Shuai, Q., Bao, H., Zhou, X.: Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9054â9063 (2021)

24. Qian, Z., Wang, S., Mihajlovic, M., Geiger, A., Tang, S.: 3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5020â5030 (2024)

25. Ren, W., Zhu, Z., Sun, B., Chen, J., Pollefeys, M., Peng, S.: Nerf on-the-go: Exploiting uncertainty for distractor-free nerfs in the wild. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8931â 8940 (2024)

26. Shao, Z., Wang, Z., Li, Z., Wang, D., Lin, X., Zhang, Y., Fan, M., Wang, Z.: Splattingavatar: Realistic real-time human avatars with mesh-embedded gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1606â1616 (2024)

27. Su, Z., Xu, L., Zheng, Z., Yu, T., Liu, Y., Fang, L.: Robustfusion: Human volumetric capture with data-driven visual cues using a rgbd camera. In: European Conference on Computer Vision. pp. 246â264. Springer (2020)

28. Sun, A., Xiang, T., Delp, S., Fei-Fei, L., Adeli, E.: Occfusion: Rendering occluded humans with generative diffusion priors. Advances in neural information processing systems 37, 92184â92209 (2024)

29. SÃ¼nderhauf, N., Abou-Chakra, J., Miller, D.: Density-aware nerf ensembles: Quantifying predictive uncertainty in neural radiance fields. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). pp. 9370â9376. IEEE (2023)

30. Te, G., Li, X., Li, X., Wang, J., Hu, W., Lu, Y.: Neural capture of animatable 3d human from monocular video. In: European Conference on Computer Vision. pp. 275â291. Springer (2022)

31. Tian, Y., Zhang, H., Liu, Y., Wang, L.: Recovering 3d human mesh from monocular images: A survey. IEEE transactions on pattern analysis and machine intelligence 45(12), 15406â15425 (2023)

32. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600â612 (2004)

33. Wen, J., Zhao, X., Ren, Z., Schwing, A.G., Wang, S.: Gomavatar: Efficient animatable human modeling from monocular video using gaussians-on-mesh. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2059â2069 (2024)

34. Weng, C.Y., Curless, B., Srinivasan, P.P., Barron, J.T., Kemelmacher-Shlizerman, I.: Humannerf: Free-viewpoint rendering of moving people from monocular video. In: Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition. pp. 16210â16220 (2022)

35. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20310â 20320 (2024)

36. Xiang, T., Sun, A., Delp, S., Kozuka, K., Fei-Fei, L., Adeli, E.: Rendering humans behind occlusions. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

37. Xiang, T., Sun, A., Wu, J., Adeli, E., Fei-Fei, L.: Rendering humans from objectoccluded monocular videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3239â3250 (2023)

38. Yang, S., Gu, X., Kuang, Z., Qin, F., Wu, Z.: Innovative ai techniques for photorealistic 3d clothed human reconstruction from monocular images or videos: a survey. The Visual Computer 41(6), 3973â4000 (2025)

39. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20331â 20341 (2024)

40. Ye, J., Zhang, Z., Liao, Q.: Occgaussian: 3d gaussian splatting for occluded human rendering. In: Proceedings of the 2025 International Conference on Multimedia Retrieval. pp. 1710â1719 (2025)

41. Yu, T., Zheng, Z., Guo, K., Liu, P., Dai, Q., Liu, Y.: Function4d: Real-time human volumetric capture from very sparse consumer rgbd sensors. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5746â5756 (2021)

42. Yu, Z., Cheng, W., Liu, X., Wu, W., Lin, K.Y.: Monohuman: Animatable human neural field from monocular video. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16943â16953 (2023)

43. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586â595 (2018)

44. Zhang, X., Liu, Z., Zhang, Y., Ge, X., He, D., Xu, T., Wang, Y., Lin, Z., Yan, S., Zhang, J.: Mega: Memory-efficient 4d gaussian splatting for dynamic scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 27828â27838 (2025)

45. Zhao, Y., Wu, C., Huang, B., Zhi, Y., Zhao, C., Wang, J., Gao, S.: Surfel-based gaussian inverse rendering for fast and relightable dynamic human reconstruction from monocular videos. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)