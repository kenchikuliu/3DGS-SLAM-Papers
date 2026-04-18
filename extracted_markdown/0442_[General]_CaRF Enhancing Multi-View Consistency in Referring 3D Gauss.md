# Camera-Aware Cross-View Alignment for Referring 3D Gaussian Splatting Segmentation

Yuwen Tao1, Kanglei Zhou2,â , Xin Tan1, Yuan Xie1 1 East China Normal University 2 Tsinghua University

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 1: Comparison between ReferSplat [1] and our CaRF on Ramen and Waldo Kitchen. CaRF produces more geometrically consistent and accurate segmentations across views, accompanied by smaller Hessian traces, indicating improved optimization stability enabled by camera-aware cross-view alignment.

AbstractâReferring 3D Gaussian Splatting Segmentation (R3DGS) aims to ground free-form language queries in 3D Gaussian fields. However, existing methods rely on single-view pseudo supervision, leading to viewpoint drift and inconsistent predictions across views. We propose CaRF (Camera-aware Referring Field), a camera-aware cross-view alignment framework for view-consistent referring in 3D Gaussian splatting. CaRF introduces Camera-conditioned Alignment Modulation (CAM) to inject camera geometry into Gaussianâtext interactions, and Gaussian-level Cross-view Logit Alignment (GCLA) to explicitly align referring responses of the same Gaussians across calibrated views during training. By turning cross-view discrepancy into an optimizable objective, CaRF enables geometry-aware and viewconsistent reasoning directly in the Gaussian space. Extensive experiments on three benchmarks demonstrate that CaRF achieves state-of-the-art performance, improving mIoU by 16.8%, 4.3%, and 2.0% on Ref-LERF, LERF-OVS, and 3D-OVS, respectively. Our code is available at https://github.com/eR3R3/CaRF.

Index Termsâ3D Gaussian Splatting, Semantic Segmentation, Language Grounding, Multi-View Consistency.

## I. INTRODUCTION

Referring 3D Gaussian Splatting Segmentation (R3DGS) [1], built on 3D Gaussian Splatting (3DGS) [2], aims to ground free-form language queries in 3D Gaussian fields by learning a per-scene referring field over Gaussians. It enables open-vocabulary 3D retrieval and segmentation from calibrated images and text, which is crucial for embodied AI [3], autonomous driving [4], [5], and AR/VR [6], [7]. Unlike conventional methods [8]â[11], R3DGS must interpret free-form expressions with complex attributes and spatial relations, often involving occlusions or invisible targets, making view-consistent reasoning essential and challenging.

Recent methods extend 3DGS toward open-vocabulary understanding by distilling 2D visionâlanguage features or lifting 2D masks into 3D space, including LangSplat [8], Feature3DGS [9], GaussianGrouping [12], and OpenGaussian [13]. ReferSplat [1] further introduces position-aware cross-modal interaction and Gaussianâtext contrastive learning for referring segmentation. While multi-view consistency is broadly recognized as beneficial, explicitly enforcing crossview agreement in R3DGS is inherently challenging: pseudo masks are view-dependent and noisy, visibility varies under occlusion, and referring queries depend on fine-grained attributes and spatial relations. As a result, optimizing with viewspecific supervision can easily induce viewpoint drift and lead to inconsistent predictions across views (see Fig. 1).

To further analyze this challenge, we observe that singleview pseudo supervision in current pipelines tends to overfit view-specific artifacts, without explicitly constraining how the same Gaussians should respond across calibrated views. Several works attempt to improve coherence via pre-/postprocessing, such as FMLGS [14] and OmniSeg3D [15], but their non-differentiable and heuristic designs make them fragile for fine-grained referring scenarios [16]. Alternatively, differentiable clustering methods [12], [13], [17], [18] promote cross-view stability through feature regularization, yet still rely mainly on 2D image-level cues and underexploit native 3D geometry [15]. These observations suggest that achieving robust cross-view consistency in R3DGS is non-trivial, calling for a fully differentiable and geometry-aware formulation directly in the 3D Gaussian space.

To this end, we propose CaRF (Camera-aware Referring Field), a novel camera-aware cross-view alignment paradigm for referring 3D Gaussian splatting. Unlike prior pipelines that optimize view-specific supervision independently, CaRF formulates referring segmentation as Gaussian-level crossview alignment in the 3D space. Specifically, CaRF introduces Camera-conditioned Alignment Modulation (CAM) to inject camera geometry into Gaussianâtext interactions, disentangling view-dependent evidence from view-invariant semantics, and Gaussian-level Cross-view Logit Alignment (GCLA) to explicitly align the referring responses of the same Gaussians across calibrated views during training. By turning crossview discrepancy into an optimizable objective, CaRF enables geometry-aware and view-consistent reasoning, providing a principled solution to viewpoint drift in R3DGS.

Extensive experiments on three benchmarks show that CaRF improves mIoU by 16.8%, 4.3%, and 2.0% over state-of-theart methods on Ref-LERF, LERF-OVS, and 3D-OVS, respectively, while significantly enhancing cross-view consistency.

Our contributions are three-fold:

â¢ We identify viewpoint drift in R3DGS as a key challenge, showing that view-dependent pseudo supervision leads to inconsistent referring across calibrated views.

â¢ We propose CaRF, a camera-aware referring field that reformulates referring segmentation as Gaussian-level cross-view alignment in the underlying 3D representation.

â¢ We develop a fully differentiable, geometry-aware alignment mechanism that enables robust and view-consistent 3D language grounding in Gaussian fields.

## II. RELATED WORK

3D Neural Representations. Neural Radiance Fields (NeRF) [19] enable high-quality novel view synthesis but suffer from slow training and rendering due to implicit representations. Recent explicit representations, such as voxels and point clouds, alleviate this issue. 3D Gaussian Splatting (3DGS) [2] represents scenes with anisotropic Gaussians and achieves real-time rendering via fast differentiable rasterization. Since then, 3DGS has been extended to various downstream tasks, including Gaussian editing [20] and 3D semantic segmentation [8], [11]. In this work, we build upon 3DGS to study language-guided understanding at the Gaussian level.

3D Segmentation in Gaussian Splatting. Large visionâ language models (VLMs) such as CLIP [21] have enabled open-vocabulary 2D segmentation, motivating their extension to 3D neural representations. Most 3DGS-based segmentation methods follow a common paradigm: extracting semantic cues from multi-view images using foundation models and distilling or lifting them into Gaussian fields. One branch focuses on 2D feature distillation [8], [22], leveraging models such as SAM, LSeg, and related approaches [23]â[25]. Another branch lifts 2D masks to supervise 3D Gaussians, including GaussianGrouping [12], SAGA [26], GaussianCut [27], Click-Gaussian [10], and OpenGaussian [13]. Although effective for category-level open-vocabulary segmentation, these methods rely heavily on single-view 2D supervision and struggle with free-form language grounding and cross-view consistency.

<!-- image-->  
Fig. 2: Overview of CaRF: CaRF integrates language interaction and cameraaware cross-view alignment in 3D Gaussian space, learning a view-consistent referring field for 3D Gaussian splatting.

Referring 3D Gaussian Splatting Segmentation. Referring 3D Gaussian Splatting Segmentation (R3DGS) extends referring expression segmentation in 2D [28]â[30] and pointbased 3D referring segmentation [31]â[33] to Gaussian fields. ReferSplat [1] pioneers this direction by learning per-Gaussian referring features with confidence-weighted pseudo-mask supervision and Gaussianâtext contrastive learning, achieving state-of-the-art performance on Ref-LERF. However, its supervision remains view-specific, making it prone to inconsistent predictions across views. Our work targets this limitation by enforcing cross-view consistency in the 3D Gaussian space.

## III. PRELIMINARIES: NOTATIONS & TASK DEFINITION

3D Gaussian Splatting (3DGS). 3DGS [2] represents a scene as a set of N anisotropic Gaussians ${ \mathcal { G } } = \{ G _ { i } =$ $( \mu _ { i } , \Sigma _ { i } , c _ { i } , \alpha _ { i } ) \} _ { i = 1 } ^ { N }$ , where $\mu _ { i } , \Sigma _ { i } , c _ { i } .$ , and $\alpha _ { i }$ denote the center, covariance, color, and opacity. Given calibrated cameras $\left( \mathbf { K } , \left[ \mathbf { R } | t \right] \right)$ , Gaussians are projected to the image plane and rendered by alpha compositing $\begin{array} { r } { C ( \pmb { p } ) \ = \ \sum _ { i } T _ { i } ( \pmb { p } ) \alpha _ { i } ^ { \prime } ( \pmb { p } ) \pmb { c } _ { i } , } \end{array}$ with $\begin{array} { r } { T _ { i } ( \pmb { p } ) = \prod _ { j < i } ( 1 - \alpha _ { j } ^ { \prime } ( \pmb { p } ) ) } \end{array}$ ). The parameters are optimized by a photometric loss $\bar { \mathcal { L } } _ { \mathrm { { p h o t o } } }$ for differentiable reconstruction.

Referring 3DGS (R3DGS). R3DGS [1] augments each Gaussian with a semantic feature $f _ { i } \in \mathbb { R } ^ { d }$ , forming a language field over Gaussians. Given a query q, a pretrained language encoder produces token embeddings $\textbf { E } \in \mathbb { R } ^ { L \times d }$ . A crossinteraction module $\phi ( \cdot , \cdot )$ fuses geometry and language to obtain enhanced features ${ \bf { g } } _ { i } = \phi ( { \bf { f } } _ { i } , { \bf { E } } )$ , and the referring score is computed as $\begin{array} { r } { m _ { i } ~ = ~ \sum _ { i } { g _ { i } ^ { \top } } e _ { j } } \end{array}$ . These scores are rendered into a 2D mask $\mathbf { M } _ { \mathrm { p r e d } } .$ , supervised by pseudo groundtruth $\mathbf { M } _ { \mathrm { g t } }$ using $\mathcal { L } _ { \mathrm { B C E } } = \mathrm { B C E } ( \mathbf { M } _ { \mathrm { p r e d } } , \mathbf { M } _ { \mathrm { g t } } )$ . ReferSplat further applies an object-wise contrastive loss ${ \mathcal L } _ { \mathrm { c o n } } = \mathrm { C o n } ( { \pmb f } _ { g } , { \pmb e } _ { t } )$ where $f _ { g }$ aggregates top-Ï Gaussians and $e _ { t }$ is the sentence embedding. The total loss is $\mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { c o n } }$

## IV. CARF: CAMERA-AWARE REFERRING FIELD

## A. Motivation and Framework Overview

Challenges. Despite promising progress, achieving viewconsistent referring in R3DGS remains non-trivial. Existing methods [1] rely on single-view pseudo supervision, which is inherently noisy and view-dependent, making models prone to overfitting view-specific artifacts and inducing viewpoint drift. As a result, the same Gaussians may yield inconsistent predictions across calibrated views (see Fig. 1), undermining reliable 3D spatial reasoning for free-form queries.

Core Idea. We rethink referring segmentation in 3DGS as a camera-aware cross-view alignment problem in 3D Gaussian space, where camera geometry is not used as auxiliary cues but co-defines the referring field, and cross-view consistency is enforced during feature formation.

Framework Overview. Fig. 2 shows the framework of CaRF. Given calibrated multi-view images and a free-form query, we first obtain a pseudo mask from 2D foundation models and use it as a supervisory signal. Each Gaussian is associated with a learnable referring feature, which interacts with word embeddings through cross-modal interaction [1] to produce Gaussianâtext similarity scores. CAM then conditions these features on camera intrinsics and extrinsics, making the similarity computation explicitly view-aware. During training, GCLA samples a pair of overlapping views, rasterizes the same Gaussian-level responses into two 2D masks, and jointly supervises them with paired-view losses, so that gradients from both views are coupled at shared Gaussians.

## B. Camera-conditioned Alignment Modulation (CAM)

View-consistent referring in 3D Gaussian Splatting is intrinsically challenging: the same Gaussian $G _ { i }$ may correspond to heterogeneous visual evidence under different camera poses due to occlusion, projection distortion, and depth-dependent appearance changes. If camera geometry is ignored, the model is forced to explain such variations purely in semantic space, which easily entangles view-dependent evidence with viewinvariant semantics and leads to semantic drift.

To explicitly model this dependency, we condition the referring field on camera geometry at the feature formation stage. Each calibrated camera is parameterized by intrinsics K and extrinsics [R|t]. We construct a pose descriptor

$$
\boldsymbol { c } = \Gamma ( \mathbf { K } , \mathbf { R } , t ) \in \mathbb { R } ^ { d _ { c } } ,\tag{1}
$$

where $\Gamma ( \cdot )$ concatenates the vectorized rotation vec(R), translation t, and normalized intrinsic parameters (focal lengths and principal point), followed by linear normalization to ensure numerical stability. This descriptor is embedded via

$$
f _ { \mathrm { c a m } } = \mathcal { E } _ { \mathrm { c a m } } ( \boldsymbol { c } ) = \mathrm { M L P } _ { \mathrm { c a m } } ( \boldsymbol { c } ) \in \mathbb { R } ^ { d } ,\tag{2}
$$

where ${ \mathrm { M L P } } _ { \mathrm { c a m } }$ is a lightweight network shared across views that maps continuous camera parameters to a coarse poseconditioned embedding, providing view-aware cues without modeling precise camera geometry.

Given a Gaussian $G _ { i }$ with referring feature $f _ { i }$ and token embeddings $\mathbf { E } = \{ e _ { j } \} _ { j = 1 } ^ { L }$ , cross-modal interaction produces a language-aware representation

$$
\pmb { g } _ { i } = \phi ( \pmb { f } _ { i } , \mathbf { E } ) \in \mathbb { R } ^ { d } .\tag{3}
$$

We then apply a camera-conditioned transform:

$$
\tilde { \pmb { g } } _ { i } ^ { ( v ) } = \mathcal { T } _ { \mathrm { c a m } } \Big ( \pmb { g } _ { i } , \pmb { f } _ { \mathrm { c a m } } ^ { ( v ) } \Big ) = \pmb { g } _ { i } + \pmb { f } _ { \mathrm { c a m } } ^ { ( v ) } ,\tag{4}
$$

yielding a view-aware Gaussian representation for camera v. This modulation introduces a pose-dependent bias in feature space, serving as a minimal and stable way to inject geometric priors without altering the underlying semantic structure. It allows the same Gaussian to express view-specific evidence while preserving a shared semantic backbone across views.

The referring logit of $G _ { i }$ under view v is computed as

$$
m _ { i } ^ { ( v ) } = \psi \left( \tilde { g } _ { i } ^ { ( v ) } , { \bf E } \right) = \sum _ { j = 1 } ^ { L } \left( \tilde { g } _ { i } ^ { ( v ) } \right) ^ { \top } e _ { j } ,\tag{5}
$$

which explicitly makes the referring response a function of both language and camera geometry. Importantly, as our ablation shows, CAM alone may amplify view-dependent variations and degrade performance if not properly constrained. This highlights that camera-aware expressiveness must be jointly regularized by cross-view supervision, which is achieved by GCLA in the following section.

## C. Gaussian-level Cross-view Logit Alignment (GCLA)

Although CAM equips each Gaussian with a view-aware representation, view consistency cannot be guaranteed without coupling supervision across views. Under conventional singleview training, the learning objective provides no constraint that links the responses of the same Gaussians observed from different cameras. As a result, for a Gaussian $G _ { i } ,$ , the predicted logits $\{ m _ { i } ^ { ( v ) } \}$ may drift across views, making cross-view consistency an ill-posed property to emerge from optimization.

We therefore formulate cross-view consistency as a Gaussian-level alignment problem and realize it through paired-view training. For a given query $q ,$ we sample two calibrated views $( v _ { a } , v _ { b } )$ that share overlapping visible regions. Using CAM, we obtain view-conditioned logits $\{ m _ { i } ^ { ( v _ { a } ) } \}$ and $\{ m _ { i } ^ { ( v _ { b } ) } \}$ for the same underlying Gaussians. These logits are rasterized into predicted masks via

$$
\begin{array} { r } { \mathbf { M } _ { \mathrm { p r e d } } ^ { ( v ) } = \mathcal { R } \left( \{ m _ { i } ^ { ( v ) } , \alpha _ { i } ^ { ( v ) } , \mu _ { i } , \boldsymbol { \Sigma } _ { i } \} _ { i = 1 } ^ { N _ { v } } \right) , \quad v \in \{ v _ { a } , v _ { b } \} , } \end{array}\tag{6}
$$

where the alpha compositing naturally down-weights occluded Gaussians, making the supervision implicitly visibility-aware.

To couple the optimization of the two views, we define

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { 2 v i e w } } = \alpha \mathrm { B C E } \Big ( \mathbf { M } _ { \mathrm { p r e d } } ^ { ( v _ { a } ) } , \mathbf { M } _ { \mathrm { g t } } ^ { ( v _ { a } ) } \Big ) + ( 1 - \alpha ) \mathrm { B C E } \Big ( \mathbf { M } _ { \mathrm { p r e d } } ^ { ( v _ { b } ) } , \mathbf { M } _ { \mathrm { g t } } ^ { ( v _ { b } ) } \Big ) } \end{array}\tag{7}
$$

where $\mathbf { M } _ { \mathrm { g t } } ^ { ( v ) }$ denotes the pseudo ground-truth mask for view v. Importantly, both terms are functions of the same Gaussian parameters through $\mathcal { R } ( \cdot )$ . Thus, gradients from the two views are accumulated on shared Gaussians during back-propagation, which implicitly drives the logits to satisfy

$$
m _ { i } ^ { ( v _ { a } ) } \approx m _ { i } ^ { ( v _ { b } ) } , \qquad \forall G _ { i } \in \mathcal { G } _ { v _ { a } } \cap \mathcal { G } _ { v _ { b } } ,\tag{8}
$$

for Gaussians that are jointly visible in both views. Rather than imposing an explicit logit matching term, this formulation ties cross-view agreement to the shared optimization of Gaussianlevel responses under paired-view supervision.

By turning viewpoint discrepancy into a directly optimizable objective in Gaussian space, GCLA stabilizes training under noisy pseudo supervision and complements the camera-aware expressiveness introduced by CAM. The overall objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { 2 v i e w } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { c o n } } ,\tag{9}
$$

TABLE I: Comparison on the Ref-LERF dataset with state-of-the-art methods. The last row reports the relative improvement (%) of CaRF over ReferSplat.
<table><tr><td>Method</td><td>Publisher Ramen Figurines Teatime Kitchen</td><td></td><td></td><td></td><td></td><td>Average</td></tr><tr><td>SPIn-NeRF [34]</td><td>ICCV&#x27;23</td><td>7.3</td><td>9.7</td><td>11.7</td><td>10.3</td><td>9.8</td></tr><tr><td>Grounded SAM [35]</td><td>arXiv&#x27;24</td><td>14.1</td><td>16.0</td><td>16.9</td><td>16.2</td><td>15.8</td></tr><tr><td>LangSplat [8]</td><td>CVPR&#x27;24</td><td>12.0</td><td>17.9</td><td>7.6</td><td>17.9</td><td>13.9</td></tr><tr><td>GS-Grouping [12]</td><td>ECCV&#x27;24</td><td>27.9</td><td>8.6</td><td>14.8</td><td>6.3</td><td>14.4</td></tr><tr><td>GOI [36]</td><td>MM&#x27;24</td><td>27.1</td><td>16.5</td><td>22.9</td><td>15.7</td><td>20.6</td></tr><tr><td>ReferSplat [1]</td><td>ICML&#x27;25</td><td>28.3</td><td>24.3</td><td>27.2</td><td>20.1</td><td>25.0</td></tr><tr><td>CaRF (Ours)</td><td></td><td>33.5</td><td>28.7</td><td>29.7</td><td>24.7</td><td>29.2</td></tr><tr><td>Improvement (%)</td><td></td><td>- +18.4</td><td>+18.1</td><td>+9.2</td><td>+22.9</td><td>+16.8</td></tr></table>

TABLE II: Comparison on the LERF-OVS dataset. The last row reports the relative improvement (%) of CaRF over ReferSplat.
<table><tr><td>Method</td><td>Publisher Ramen</td><td></td><td>Figurines Teatime Kitchen Average</td><td></td><td></td><td></td></tr><tr><td>Feature-3DGS [9]</td><td>CVPR&#x27;24</td><td>43.7</td><td>58.8</td><td>40.5</td><td>39.6</td><td>45.6</td></tr><tr><td>LEGaussians [37]</td><td>CVPR&#x27;24</td><td>46.0</td><td>60.3</td><td>40.8</td><td>39.4</td><td>46.6</td></tr><tr><td>LangSplat [8]</td><td>CVPR&#x27;24</td><td>51.2</td><td>65.1</td><td>44.7</td><td>44.5</td><td>51.4</td></tr><tr><td>GS-Grouping [12]</td><td>ECCV&#x27;24</td><td>45.5</td><td>60.9</td><td>40.0</td><td>38.7</td><td>46.3</td></tr><tr><td>GOI [36]</td><td>MM&#x27;24</td><td>52.6</td><td>63.7</td><td>44.5</td><td>41.4</td><td>50.6</td></tr><tr><td>ReferSplat [1]</td><td>ICML&#x27;25</td><td>53.1</td><td>64.1</td><td>50.1</td><td>43.3</td><td>52.6</td></tr><tr><td>CaRF (Ours)</td><td></td><td>55.2</td><td>67.1</td><td>51.0</td><td>46.3</td><td>54.9</td></tr><tr><td>Improvement (%)</td><td></td><td>+4.0</td><td>+4.7</td><td>+1.8</td><td>+6.9</td><td>+4.3</td></tr></table>

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are weighting coefficients.

## V. EXPERIMENTS

## A. Experimental Setting

Datasets. We evaluate CaRF on three representative benchmarks: Ref-LERF, LERF-OVS, and 3D-OVS, which collectively cover diverse challenges in 3D language grounding. Ref-LERF focuses on scene-specific referring expressions with complex spatial relations and occlusions; we adopt its official data splits and follow the confidence-weighted IoU pseudomasking protocol introduced by ReferSplat [1]. LERF-OVS extends 3D Gaussian Splatting to open-vocabulary segmentation across multiple scenes; we align our evaluation with LangSplat [8] and LangSplat-V2 by querying with classlevel textual phrases. 3D-OVS targets large-scale, categoryand room-level 3D open-vocabulary segmentation, as used in recent works such as GAGS and OpenGaussian [13].

Evaluation Metric (mIoU). We adopt the mean Intersection-over-Union (mIoU) as the primary evaluation metric, following [1]. For each queryâview pair, we compute the Intersection-over-Union between the predicted mask YË and the ground-truth mask Y as IoU $\begin{array} { r } { \operatorname { J } ( Y , \hat { Y } ) = \frac { | Y \cap \hat { Y } | } { | Y \cup \hat { Y } | } } \end{array}$

Implementation Details. We first pretrain an RGB-only 3DGS model and freeze its geometry parameters before learning the referring field, following standard practice for accurate visibility supervision. Text embeddings are extracted by BERT, and Gaussianâtext interaction follows ReferSplat [1]. During training, we sample paired views $( v _ { a } , v _ { b } )$ with at least 30% overlap and use Î± = 0.5 in Eq. (7). Pseudo masks are generated from K SAM candidates via confidence-weighted IoU aggregation. We optimize the model with Adam for 30k iterations, using learning rates of $2 . 5 \times 1 0 ^ { - 3 }$ for the referring field and contrastive head, and $1 \times 1 0 ^ { - 4 }$ for the camera MLP. The feature dimension is $d \ = \ 1 2 8 ,$ , with mixed-precision training and gradient clipping of 1.0. All experiments are conducted on RTX A6000 GPUs. We reproduce ReferSplat under the same settings, and results are averaged over five runs with $\lambda _ { 1 } = \lambda _ { 2 } = 1$ for balanced weighting.

TABLE III: Comparison on the 3D-OVS dataset. The last row reports the relative improvement (%) of CaRF over ReferSplat.
<table><tr><td>Method</td><td>Publisher</td><td>Bed</td><td>Bench</td><td>Room</td><td>Sofa</td><td>Lawn</td><td>Average</td></tr><tr><td>Feature-3DGS [9]</td><td>CVPR 2024</td><td>83.5</td><td>90.7</td><td>84.7</td><td>86.9</td><td>93.4</td><td>87.8</td></tr><tr><td>LEGaussians [37]</td><td>CVPR 2024 84.9</td><td></td><td>91.1</td><td>86.0</td><td>87.8</td><td>92.5</td><td>88.5</td></tr><tr><td>LangSplat [8]</td><td>CVPR 2024</td><td>92.5</td><td>94.2</td><td>94.1</td><td>90.0</td><td>96.1</td><td>93.4</td></tr><tr><td>GS-Grouping [12]</td><td>ECCV&#x27;24 83.0</td><td></td><td>91.5</td><td>85.9</td><td>87.3</td><td>90.6</td><td>87.7</td></tr><tr><td>GOI [36]</td><td>MM&#x27;24 89.4</td><td></td><td>92.8</td><td>91.3</td><td>85.6</td><td>94.1</td><td>90.6</td></tr><tr><td>ReferSplat [1]</td><td>ICML&#x27;25 90.2</td><td></td><td>93.8</td><td>94.1</td><td>90.8</td><td>95.5</td><td>92.9</td></tr><tr><td>CaRF (Ours)</td><td></td><td>92.1</td><td>94.2</td><td>96.8</td><td>93.2</td><td>97.3</td><td>94.7</td></tr><tr><td>Improvement (%)</td><td></td><td>â +2.1</td><td>+0.4</td><td>+2.9</td><td>+2.6</td><td>+1.9</td><td>+2.0</td></tr></table>

TABLE IV: Unified ablation study of CaRF on Ref-LERF.
<table><tr><td>Setting</td><td>GCLA CAM Selection Cam Fusion #Views Ramen</td><td></td><td></td><td></td><td></td><td></td><td>Kitchen</td></tr><tr><td>Baseline (ReferSplat)</td><td>Ã</td><td>X</td><td>LERF</td><td>â</td><td> 2</td><td>28.3</td><td>20.1</td></tr><tr><td>+ GCLA</td><td>â</td><td>Ã</td><td>LERF</td><td>â</td><td></td><td>31.6</td><td>22.4</td></tr><tr><td>+ CAM</td><td>X</td><td>â</td><td>LERF</td><td>MLP</td><td>1</td><td>24.3</td><td>13.5</td></tr><tr><td>Ours (Cosine)</td><td>;</td><td></td><td>Cosine</td><td>MLP</td><td>2</td><td>33.5</td><td>24.7</td></tr><tr><td>Ours (LERF)</td><td></td><td>&gt;&gt;</td><td>LERF</td><td>MLP</td><td>2</td><td>31.2</td><td>23.2</td></tr><tr><td>Ours (Post-fusion)</td><td>;</td><td>&gt;&gt;</td><td>Cosine</td><td>Post</td><td>2</td><td>25.6</td><td>18.3</td></tr><tr><td>Ours (Lang-enc)</td><td></td><td></td><td>Cosine</td><td>Lang</td><td>2</td><td>28.3</td><td>22.4</td></tr><tr><td>Ours (3-view)</td><td></td><td></td><td>Cosine</td><td>MLP</td><td>3</td><td>33.7</td><td>23.1</td></tr><tr><td>Ours (4-view)</td><td>&gt;&gt;</td><td>&gt;</td><td>Cosine</td><td>MLP</td><td>4</td><td>32.4</td><td>24.1</td></tr><tr><td>Ours (2-view)</td><td>â</td><td>â</td><td>Cosine</td><td>MLP</td><td>2</td><td>33.5</td><td>24.7</td></tr></table>

## B. Results on the Ref-LERF Dataset

As shown in Tab. I, CaRF achieves new state-of-theart performance on Ref-LERF, improving the average mIoU from 25.0 (ReferSplat) to 29.2 (+16.8%). Consistent gains are observed across all scenes, including Ramen (+18.4%), Figurines (+18.1%), Teatime (+9.2%), and Kitchen (+22.9%). The largest improvements appear in Kitchen and Ramen, where heavy occlusion, clutter, and fine structures make single-view pseudo supervision particularly brittle. Overall, these results demonstrate that camera-aware cross-view alignment substantially stabilizes multi-view reasoning for referring 3DGS.

## C. Results on 3D Open-Vocabulary Segmentation Datasets

As shown in Tabs. II and III, CaRF consistently outperforms prior methods on both LERF-OVS and 3D-OVS. On LERF-OVS, CaRF achieves 54.9 mIoU, surpassing ReferSplat by 4.3%, with larger gains in cluttered scenes such as Kitchen (+6.9%) and Figurines (+4.7%), where viewpoint drift is more severe. On 3D-OVS, CaRF reaches 94.7 mIoU, improving over ReferSplat by 2.0%; the smaller margin is expected since the baseline is already near-saturated, leaving limited room for improvement. Overall, these results demonstrate that CaRF enables robust open-vocabulary segmentation across scenes.

## D. Ablation Study

We conduct a unified ablation study on Ref-LERF to analyze key design choices of CaRF, including paired-view supervision (GCLA), camera-aware encoding (CAM), Gaussian selection, camera fusion design, and the number of training views. As summarized in Tab. IV, introducing GCLA brings consistent gains over the baseline, while using CAM alone degrades performance, indicating that camera-conditioned features require cross-view constraints to form stable correspondences. Combining both yields the best results, confirming their complementarity. For Gaussian selection, cosine similarity outperforms LERF-style relevancy scoring, providing more stable and accurate responses. Among camera fusion strategies, the MLP-based design achieves the highest performance, while post-fusion and language-level fusion lead to clear drops, showing that directly conditioning Gaussian features is most effective. Finally, using two views offers the best accuracyâefficiency trade-off, as more views bring marginal gains but higher cost. Overall, these results validate that CaRFâs improvements stem from jointly modeling camera geometry and Gaussian-level cross-view alignment.

<!-- image-->

<!-- image-->

<!-- image-->

Fig. 3: Performanceâcomputation comparison between CaRF and ReferSplat [1]. (a) Parameter increase (âParams). (b) FLOPs difference (âFLOPs). (c) Performance improvement (âmIoU) across three benchmarks.  
<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 4: Analysis of CaRF on Ref-LERF. We report (a) GV-IoU, (b) cross-view agreement, (c) gains under increasing language complexity, and (d) Hessian trace ratio (lower is better) on Ramen and Kitchen.

## E. Qualitative and Quantitative Analysis

Balance Between Computation and Accuracy As shown in Fig. 3, CaRF achieves a favorable accuracyâefficiency tradeoff. The only extra parameters come from the lightweight CAM MLP, adding merely 25.7K parameters to map camera poses to a 128-d embedding. The cross-view alignment loss (GCLA) is used only during training and introduces moderate overhead (approximately 2Ã training time with negligible VRAM increase due to dual-view rasterization), while inference follows the rendering pipeline of ReferSplat with almost no additional FLOPs. Despite this small training overhead, CaRF improves mIoU by 16.8%, 4.3%, and 2.0% on Ref-LERF, LERF-OVS, and 3D-OVS, respectively, demonstrating that camera-aware alignment significantly enhances 3D language grounding while keeping inference lightweight.

Cross-View Consistency. In Fig. 4(a), we report the Paired-View IoU in Gaussian space, $\mathrm { G V - I o U } = \frac { | S ^ { ( v _ { a } ) } \cap S ^ { ( v _ { b } ) } | } { | S ^ { ( v _ { a } ) } \cup S ^ { ( v _ { b } ) } | }$ , where $S ^ { ( v ) } = \{ i \mid m _ { i } ^ { ( v ) }$ in top-k}, which measures the overlap of foreground Gaussians predicted from two calibrated views. CaRF achieves the highest GV-IoU, reaching 0.60 on Ramen and 0.54 on Kitchen, surpassing ReferSplat (0.40/0.35).

<!-- image-->

Fig. 5: Qualitative comparisons on Ref-LERF across two scenes and two views: compared with GS-Grouping [38] and ReferSplat [1], our CaRF produces view-consistent masks that closely match the GT.  
<!-- image-->

<!-- image-->  
Fig. 6: Failure cases on the Ref-LERF dataset.

Cross-View Agreement. In Fig. 4(b), we compute the Pearson correlation $\rho = \mathrm { C o r r } ( m ^ { ( v _ { a } ) } , m ^ { ( v _ { b } ) } )$ between per-Gaussian logits rendered from paired views to assess continuous response consistency. CaRF again performs best with Ï = 0.86 on Ramen and 0.76 on Kitchen.

Language Complexity. In Fig. 4(c), we group queries into noun-only, attribute plus noun, and spatial relation, and report the mIoU gains over ReferSplat. CaRF shows increasingly larger improvements as language becomes more complex, achieving up to +6.5 and +5.4 mIoU for spatial queries.

Optimization Stability. In Fig. 4(d), we estimate the Hessian trace ratio of the training loss using Hutchinson approximation, normalized by ReferSplat. CaRF yields the lowest ratios, 0.73 on Ramen and 0.80 on Kitchen, indicating the most stable optimization among all variants.

Case Study. In Fig. 5, CaRF produces more accurate and complete masks that better match the referring expressions, while Gaussian Grouping often confuses nearby regions due to category-level clustering, and ReferSplat exhibits viewinconsistent or fragmented predictions. Failure cases in Fig. 6 mainly stem from noisy or ambiguous pseudo masks rather than model errors. Even under such imperfect supervision, CaRF still yields smoother and more consistent segmentations than ReferSplat, demonstrating robustness to label noise.

## VI. CONCLUSION AND DISCUSSION

We presented CaRF, a camera-aware R3DGS method that tackles multi-view inconsistency. By combining cameraconditioned alignment modulation with Gaussian-level crossview logit alignment, CaRF enables geometry-aware and viewconsistent language grounding directly in Gaussian space. Experiments demonstrate consistent mIoU gains over prior methods, establishing CaRF as a strong baseline for R3DGS and highlighting the importance of coupling language grounding with camera geometry for robust 3D understanding.

CaRF still has limitations. It relies on pseudo masks for supervision and requires per-scene optimization, which may limit scalability to dynamic scenes. Future work will explore stronger 3D priors and more lightweight designs to extend CaRF toward dynamic 3D perception.

## REFERENCES

[1] S. He, G. Jie, C. Wang, Y. Zhou, S. Hu, G. Li, and H. Ding, âRefersplat: Referring segmentation in 3d gaussian splatting,â in ICML, 2025.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaus- Â¨ sian splatting for real-time radiance field rendering,â arXiv preprint arXiv:2308.04079, 2023.

[3] O. Shorinwa, J. Tucker, A. Smith, A. Swann, T. Chen, R. Firoozi, M. K. III, and M. Schwager, âSplat-mover: Multi-stage, open-vocabulary robotic manipulation via editable gaussian splatting,â arXiv preprint arXiv:2405.04378, 2024.

[4] K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen, A. Maalouf, S. Li, G. Iyer, S. Saryazdi, N. Keetha, A. Tewari, J. B. Tenenbaum, C. M. de Melo, M. Krishna, L. Paull, F. Shkurti, and A. Torralba, âConceptfusion: Open-set multimodal 3d mapping,â arXiv preprint arXiv:2302.07241, 2023.

[5] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. M. de Melo, J. B. Tenenbaum, A. Torralba, F. Shkurti, and L. Paull, âConceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,â arXiv preprint arXiv:2309.16650, 2023.

[6] Y. Jiang, C. Yu, T. Xie, X. Li, Y. Feng, H. Wang, M. Li, H. Lau, F. Gao, Y. Yang, and C. Jiang, âVr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality,â arXiv preprint arXiv:2401.16663, 2024.

[7] K. Liu, F. Zhan, J. Zhang, M. Xu, Y. Yu, A. E. Saddik, C. Theobalt, E. Xing, and S. Lu, âWeakly supervised 3d open-vocabulary segmentation,â arXiv preprint arXiv:2305.14093, 2024.

[8] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, âLangsplat: 3d language gaussian splatting,â in CVPR, pp. 20051â20060, 2024.

[9] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, âFeature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields,â in CVPR, pp. 21676â21685, 2024.

[10] S. Choi, H. Song, J. Kim, T. Kim, and H. Do, âClick-gaussian: Interactive segmentation to any 3d gaussians,â arXiv preprint arXiv:2407.11793, 2024.

[11] H. Li, R. Qin, Z. Zou, D. He, B. Li, B. Dai, D. Zhang, and J. Han, âLangsurf: Language-embedded surface gaussians for 3d scene understanding,â arXiv preprint arXiv:2412.17635, 2024.

[12] M. Ye, M. Danelljan, F. Yu, and L. Ke, âGaussian grouping: Segment and edit anything in 3d scenes,â in ECCV, pp. 162â179, 2024.

[13] Y. Wu, J. Meng, H. Li, C. Wu, Y. Shi, X. Cheng, C. Zhao, H. Feng, E. Ding, J. Wang, et al., âOpengaussian: Towards point-level 3d gaussian-based open vocabulary understanding,â in NeurIPS, vol. 37, pp. 19114â19138, 2024.

[14] X. Tan, Y. Ji, H. Zhu, and Y. Xie, âFmlgs: Fast multilevel language embedded gaussians for part-level interactive agents,â arXiv preprint arXiv:2504.08581, 2025.

[15] H. Ying, Y. Yin, J. Zhang, F. Wang, T. Yu, R. Huang, and L. Fang, âOmniseg3d: Omniversal 3d segmentation via hierarchical contrastive learning,â in CVPR, pp. 20612â20622, 2024.

[16] R. Zhu, S. Qiu, Z. Liu, K.-H. Hui, Q. Wu, P.-A. Heng, and C.-W. Fu, âRethinking end-to-end 2d to 3d scene segmentation in gaussian splatting,â in CVPR, pp. 3656â3665, 2025.

[17] M. C. Silva, M. Dahaghin, M. Toso, and A. D. Bue, âContrastive gaussian clustering: Weakly supervised 3d scene segmentation,â arXiv preprint arXiv:2404.12784, 2024.

[18] S. Choi, H. Song, J. Kim, T. Kim, and H. Do, âClick-gaussian: Interactive segmentation to any 3d gaussians,â in ECCV, vol. 15061, pp. 289â305, 2024.

[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â arXiv preprint arXiv:2003.08934, 2020.

[20] Y. Chen, Z. Chen, C. Zhang, F. Wang, X. Yang, Y. Wang, Z. Cai, L. Yang, H. Liu, and G. Lin, âGaussianeditor: Swift and controllable 3d editing with gaussian splatting,â arXiv preprint arXiv:2311.14521, 2023.

[21] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, âLearning transferable visual models from natural language supervision,â arXiv preprint arXiv:2103.00020, 2021.

[22] Y. Ji, H. Zhu, J. Tang, W. Liu, Z. Zhang, X. Tan, and Y. Xie, âFastlgs: Speeding up language embedded gaussians with feature grid mapping,â in AAAI, vol. 39, pp. 3922â3930, 2025.

[23] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick, Â´ âSegment anything,â arXiv preprint arXiv:2304.02643, 2023.

[24] B. Li, K. Q. Weinberger, S. Belongie, V. Koltun, and R. Ranftl, âLanguage-driven semantic segmentation,â arXiv preprint arXiv:2201.03546, 2022.

[25] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, Â¨ N. Carion, C.-Y. Wu, R. Girshick, P. Dollar, and C. Feichtenhofer, Â´ âSam 2: Segment anything in images and videos,â arXiv preprint arXiv:2408.00714, 2024.

[26] J. Cen, J. Fang, C. Yang, L. Xie, X. Zhang, W. Shen, and Q. Tian, âSegment any 3d gaussians,â in AAAI, vol. 39, pp. 1971â1979, 2025.

[27] U. Jain, A. Mirzaei, and I. Gilitschenski, âGaussiancut: Interactive segmentation via graph cut for 3d gaussian splatting,â NeurIPS, vol. 37, pp. 89184â89212, 2024.

[28] H. Ding, C. Liu, S. Wang, and X. Jiang, âVision-language transformer and query generation for referring segmentation,â arXiv preprint arXiv:2108.05565, 2021.

[29] C. Liu, H. Ding, and X. Jiang, âGres: Generalized referring expression segmentation,â arXiv preprint arXiv:2306.00968, 2023.

[30] H. Ding, C. Liu, S. He, X. Jiang, and C. C. Loy, âMevis: A largescale benchmark for video segmentation with motion expressions,â arXiv preprint arXiv:2308.08544, 2023.

[31] S. He, H. Ding, X. Jiang, and B. Wen, âSegpoint: Segment any point cloud via large language model,â in ECCV, pp. 349â367, 2024.

[32] C. Wang, M. Wu, S.-K. Lam, X. Ning, S. Yu, R. Wang, W. Li, and T. Srikanthan, âGpsformer: A global perception and local structure fitting-based transformer for point cloud understanding,â arXiv preprint arXiv:2407.13519, 2024.

[33] C. Wang, S. He, X. Fang, M. Wu, S.-K. Lam, and P. Tiwari, âTaylor series-inspired local structure fitting network for few-shot point cloud semantic segmentation,â arXiv preprint arXiv:2504.02454, 2025.

[34] A. Mirzaei, T. Aumentado-Armstrong, K. G. Derpanis, J. Kelly, M. A. Brubaker, I. Gilitschenski, and A. Levinshtein, âSpin-nerf: Multiview segmentation and perceptual inpainting with neural radiance fields,â in CVPR, pp. 20669â20679, 2023.

[35] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, et al., âGrounded sam: Assembling open-world models for diverse visual tasks,â arXiv preprint arXiv:2401.14159, 2024.

[36] Y. Qu, S. Dai, X. Li, J. Lin, L. Cao, S. Zhang, and R. Ji, âGoi: Find 3d gaussians of interest with an optimizable open-vocabulary semanticspace hyperplane,â in ACM MM, pp. 5328â5337, 2024.

[37] J.-C. Shi, M. Wang, H.-B. Duan, and S.-H. Guan, âLanguage embedded 3d gaussians for open-vocabulary scene understanding,â in CVPR, pp. 5333â5343, 2024.

[38] M. Ye, M. Danelljan, F. Yu, and L. Ke, âGaussian grouping: Segment and edit anything in 3d scenes,â in ECCV, pp. 162â179, 2024.