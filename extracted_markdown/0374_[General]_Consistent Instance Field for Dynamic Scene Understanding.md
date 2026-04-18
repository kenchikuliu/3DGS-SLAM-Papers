# Consistent Instance Field for Dynamic Scene Understanding

Junyi Wu1,2,â  Changchang Sun1

Gengyu Zhang1 Meng Zheng2 Feiran Wang1

Terrence Chen2 Yan Yan1 Ziyan Wu2

1University of Illinois Chicago, Chicago, IL, USA 2United Imaging Intelligence, Boston, MA, USA

## Abstract

We introduce Consistent Instance Field, a continuous and probabilistic spatio-temporal representation for dynamic scene understanding. Unlike prior methods that rely on discrete tracking or view-dependent features, our approach disentangles visibility from persistent object identity by modeling each spaceâtime point with an occupancy probability and a conditional instance distribution. To realize this, we introduce a novel instance-embedded representation based on deformable 3D Gaussians, which jointly encode radiance and semantic information and are learned directly from input RGB images and instance masks through differentiable rasterization. Furthermore, we introduce new mechanisms to calibrate per-Gaussian identities and resample Gaussians toward semantically active regions, ensuring consistent instance representations across space and time. Experiments on HyperNeRF and Neu3D datasets demonstrate that our method significantly outperforms state-of-the-art methods on novel-view panoptic segmentation and open-vocabulary 4D querying tasks.

## 1. Introduction

Dynamic scenes reveal not only how the world changes, but also how its entities persist and interact over time. Understanding them remains a central challenge in computer vision, which goes beyond reconstructing geometry and appearance, and seeks to know what is moving while maintaining temporally consistent semantics. This capability forms the foundation for a wide range of applications, including augmented/virtual reality [13, 18, 49], autonomous driving [12, 60, 65], and robotics [28, 48, 52, 54, 68].

Dynamic scene understanding has advanced rapidly with the encouraging progress in 3D representations [20, 37,

<!-- image-->  
Figure 1. Comparisons with prior work SA4D [16]. Previous methods like SA4D often rely on view-dependent features with RGB modulation, leading to semantic inconsistencies in dynamic scenes: unstable under cross-view instance supervision, confusing color opacity with object occupancy, and underrepresenting semantically meaningful regions. Our approach formulates a continuous probabilistic field over existence and identity in space-time, enabling identity modeling beyond visibility cues and adaptive redistribution of Gaussian capacity. This results in a coherent instance field across deformation and changing viewpoints.

38] and large foundation models [24, 47]. Deformable representations, including NeRF-based [40, 41, 45] and Gaussian-based variants [1, 8, 14], achieve photorealistic reconstructions of motion and appearance, yet they omit explicit modeling of object identity. Recent works attempted to address this gap by adding vision-language features into 3D representation [21, 27, 46, 51] or by incorporating 2D mask supervision [9, 29, 44, 50, 58]. However, as shown in Figure 1, these approaches have inherent limitations: their supervision is mediated through RGB rendering, making them inherently view-dependent. Without explicitly modeling persistent object existence in space-time, they tie identity to radiance and remain vulnerable to visibility bias, which can lead to drifting and underrepresented semantics when objects deform or change appearance across views.

To address this issue, we propose Consistent Instance

Field (CIF), a novel framework modeling dynamic scenes using spatio-temporal functions that jointly encode object existence and identity. In our framework, a dynamic scene is represented as a continuous, object-centric field, where each point in space-time is attributed to a persistent entity. This perspective shifts the focus from tracking changing appearances to modeling the persistent composition of objects in 4D: evaluating the field at any 4D location reveals both whether an entity exists and which one it is. To realize this field, we extend deformable Gaussians [35, 62], with each Gaussian acting as a local carrier of geometric, radiometric information, and encodes two probabilistic quantities: an occupancy probability, indicating physical existence, and a conditional identity distribution, specifying the associated instance. These quantities are jointly optimized through differentiable rasterization [20] of both RGB and semantic fields in our Field-Aware Splatting, enabling coherent photometric and semantic supervision across views and time.

To further align the discrete Gaussian representation with the underlying continuous field, we introduce two key mechanisms. First, Instance Identity Estimation maps input 2D instance masks into our representation via aggregation across views and time. We then incorporate calibration factors to alleviate the visibility bias and refine the identity distributions, converting view-dependent labels into temporally consistent distributions and allowing each Gaussian to maintain stable instance identities. Second, Instance-Guided Resampling adaptively redistributes Gaussian density according to the instance-field signal, concentrating representational capacity around semantically relevant regions while preserving local volumetric balance. Together, these mechanisms enable the representation to self-organize around meaningful entities, achieving coherent geometry, appearance, and identity throughout the 4D scene.

We evaluate our method on standard benchmarks of dynamic scenes, achieving state-of-the-art performance on novel-view panoptic segmentation and open-vocabulary 4D querying tasks. In summary, our contributions are:

1. We propose Consistent Instance Field (CIF), a continuous and probabilistic spatio-temporal formulation for dynamic scenes that jointly encodes occupancy and instance identity, which is realized in a Gaussian representation and optimized via differentiable rasterization.

2. We introduce two mechanisms, Instance Identity Estimation and Instance-Guided Resampling, that align the discrete Gaussian representation with the continuous field, enabling temporally consistent and semantically coherent instance modeling.

3. We demonstrate that CIF outperforms prior methods, improving average mIoU by +11.4 on HyperNeRF and +5.8 on Neu3D for novel-view panoptic segmentation, while producing sharper boundaries and more accurate instance separation for open-vocabulary 4D querying.

## 2. Related Work

Dynamic Scene Representation. Modeling dynamic scenes is a long-standing challenge in 3D vision, often approached by extending static neural representations with deformation fields [4, 11]. Neural Radiance Fields (NeRF) [37] represent a scene as a continuous volumetric function, later generalized to dynamic settings through deformation fields that map canonical coordinates to their time-dependent counterparts [10, 17, 40, 41, 53]. However, these implicit volumetric models remain computationally heavy and offer limited structural interpretability. 3D Gaussian Splatting [20] offers an efficient, explicit alternative with real-time rendering [2, 55, 66]. Its dynamic variants introduce deformation [56, 62], trajectories [30, 57, 64], or time-conditioned primitives [7], achieving high photometric fidelity but neglecting explicit modeling of instance persistence. Our work bridges this gap through an instanceconsistent formulation that unifies geometry, motion, and identity in a coherent 4D representation.

Semantic Scene Understanding. Beyond reconstruction, recent research has sought to endow 3D representations with semantic understanding. Point-based approaches [6, 15, 32, 44, 61] directly embed semantics in point cloud, providing explicit category cues but lacking photometric grounding. NeRF-based approaches [9, 21, 25, 31] learn semantic and instance fields from vision foundation models [24, 39], enabling novel-view rendering but remaining implicit and computationally heavy. More recently, Gaussian-based representations [20] have emerged as an efficient and interpretable alternative, supporting fine-grained understanding through explicit primitives. Languageaugmented extensions [19, 27, 43, 46, 51, 58] further incorporate visionâlanguage features [47] to enable openvocabulary querying. However, most approaches rely on view-dependent semantic cues, resulting in semantics biased toward visible regions and prone to fragmentation under occlusion. To improve cross-view consistency, recent works [3, 16, 29, 36, 50, 63, 67] leverage 2D masks or video trackers [5] to constrain Gaussian semantics, but ambiguous boundaries and occlusions still require heuristic filtering. In contrast, we introduce a continuous, temporally consistent instance field, where semantics are grounded in the persistence of physical entities, enabling a principled connection with geometry and appearance in dynamic scenes.

## 3. Method

In this section, we first introduce our Consistent Instance Field (CIF), including the instance-embedded Gaussian representation, which jointly encodes geometry, radiance, and semantics (Sec. 3.1). We then explain how input 2D masks are propagated through Instance Identity Estimation to establish coherent instance identities (Sec. 3.2), and how

<!-- image-->  
Figure 2. Overview of our Consistent Instance Field. Our method models each dynamic scene as a continuous 4D Consistent Instance Field that encodes existence and identity distributions (Sec. 3.1.1). We realize the field as an Instance-Embedded Gaussian Representation, which jointly models geometry, appearance, occupancy, and instance identity (Sec. 3.1.2). (Bottom) Instance Identity Estimation. Per-Gaussian identity distributions are inferred by aggregating 2D observations over time and views. A learnable calibration then corrects visibility-induced biases (Eqs. (6), (7), (8)), yielding stable identity under occlusion and appearance changes (Sec. 3.2). (Right) Instance-Guided Resampling. To align representational capacity with semantic signals, Gaussians with weak instance responses are adaptively relocated toward semantically active regions. A volume-conserving adjustment further refines opacity and occupancy (Eqs. (10), (11)), forming dense object-aligned clusters while preserving radiance and spatial continuity (Sec. 3.3). (Top) Field-Aware Splatting. During rendering, each Gaussian contributes occupancy and identity to per-pixel distributions, supervised with a cross-entropy loss to align Gaussians with the underlying 4D instance field (Sec. 3.1.3).

Instance-Guided Resampling adaptively reallocates capacity to preserve the fidelity of the evolving continuous field (Sec. 3.3). Finally, we describe the joint optimization of all components via differentiable rendering (Sec. 3.4).

## 3.1. Consistent Instance Field

## 3.1.1. Formulation of Instance Consistency

Dynamic scenes can be represented as a continuous 4D field where entities persist, move, and interact through time. To describe this perspective, we introduce the Consistent Instance Field, a spatioâtemporal function to capture both their existence and identity over space and time. Formally, let $E \in \{ 0 , 1 \}$ be a binary variable indicating whether a 4D location (x, t) is occupied by any physical entity, and $K \in \kappa$ a categorical variable representing the instance identity. The field is expressed as the joint distribution:

$$
\gamma ( \mathbf { x } , t , k ) = P ( E { = } 1 , K { = } k \mid \mathbf { x } , t )
$$

$$
\mathbf { \xi } = \underbrace { P ( E { = } 1 \mid \mathbf { x } , t ) } _ { \pi ( \mathbf { x } , t ) } \underbrace { P ( K { = } k \mid E { = } 1 , \mathbf { x } , t ) } _ { p ( \mathbf { x } , t , k ) } .\tag{1}
$$

(2)

Here, $\pi ( \mathbf { x } , t ) \in [ 0 , 1 ]$ is the probability that the location $\mathbf { \Psi } ( \mathbf { x } , t )$ is occupied by objects, while $p ( \mathbf { x } , t , k )$ is a conditional distribution over instance identities with $\scriptstyle \sum _ { k } p ( \mathbf { x } , t , k ) = 1$ Each value $\gamma ( \mathbf { x } , t , k )$ therefore quantifies the probability that instance k occupies the space-time coordinate (x, t).

This decomposition separates the existence of matter from the persistence of identity: the occupancy Ï models spatial-temporal continuity of physical presence, while the conditional identity p maintains consistent affiliation across deformation and motion. This allows our formulation to capture not only where geometry resides but also which entity it belongs to. The Consistent Instance Field thus defines an object-centric partition of space-time, where low-entropy regions signify stable ownership and higher-entropy zones near interactions represent softly shared boundaries.

## 3.1.2. Instance-Embedded Gaussian Representation

To ground the Consistent Instance Field into an explicit framework, we adopt an Instance-Embedded Gaussian Representation, which approximates the continuous field $\gamma ( \mathbf { x } , t , k )$ through a finite set of spatially localized primitives. Building upon Gaussian-based dynamic scene representations [35, 56], each primitive is extended to not only encode geometry and radiance but also instance semantics. Through this integration, the scene can be represented as a set of Gaussians

$$
\mathcal { G } { = } \{ g _ { i } { = } ( \mathbf { x } _ { i } , \mathbf { R } _ { i } , \mathbf { s } _ { i } , \mathbf { c } _ { i } , \alpha _ { i } , \pi _ { i } , p _ { i } ^ { 1 } , \ldots , p _ { i } ^ { K } ) \} _ { i = 1 } ^ { N } ,
$$

where $\mathbf { x } _ { i } { \in } \mathbb { R } ^ { 3 }$ denotes the center, $\mathbf { R } _ { i } { \in } S O ( 3 )$ the rotation, $\mathbf { s } _ { i } \in \mathbb { R } _ { + } ^ { 3 }$ the scale, $\mathbf { c } _ { i } { \in } \mathbb { R } ^ { 3 }$ the color, $\alpha _ { i } \in [ 0 , 1 ]$ the opacity, and $( { \dot { \pi } } _ { i } , p _ { i } ^ { 1 } , \dots , p _ { i } ^ { K } )$ the probability of existence and identity distribution over K instances defined in Sec. 3.1.1. Representation of dynamic scenes is modeled by an additional time-conditioned MLP [56, 62], which produces smooth trajectories for each Gaussian and modulates its parameters over time. Each primitive carries instance coefficients:

$$
\pi _ { i } \approx \pi ( { \bf x } _ { i } ( t ) , t ) , ~ p _ { i } ^ { k } \approx p ( { \bf x } _ { i } ( t ) , t , k ) .\tag{3}
$$

These parameters connect the continuous field to discrete primitives to track both object existence and identity. As each Gaussian evolves under the deformation field, it preserves its semantic identity, propagating instance information in the dynamic scene while maintaining local consistency across objects.

## 3.1.3. Field-Aware Splatting

With the Instance-Embedded Gaussian Representation established, we detail how the Consistent Instance Field guides the rendering process. In our formulation, splatting [20, 43] can marginalize the 4D occupancyidentity distribution $\gamma ( \mathbf { x } , t , k )$ along camera rays, converting the geometric and semantic attributes of Gaussians into view-dependent color and instance maps. For appearance, we adopt the standard alpha-compositing rule [20]. The color at pixel $( u , v )$ and time t is given by: $\begin{array} { r } { \mathbf { C } ( u , v , t ) { = } \sum _ { i } T _ { i } ( u , v , t ) \alpha _ { i } ( t ) P _ { i } ( u , v , t ) \mathbf { c } _ { i } ( t ) } \end{array}$ , where $P _ { i } ( u , v , t )$ is the Gaussian weight obtained by projecting the mean and covariance of $g _ { i } ,$ , and $\begin{array} { r } { T _ { i } ( u , v , t ) { = } \prod _ { j < i } ( 1 - } \end{array}$ $\alpha _ { j } ( t ) P _ { j } ( u , v , t ) \big )$ is the accumulated transmittance.

To incorporate instance information, we extend this process to render the marginal instance identity map:

$$
{ \bf M } _ { k } ( u , v , t ) = \sum _ { i } T _ { i } ^ { \mathrm { i n s t } } ( u , v , t ) \pi _ { i } P _ { i } ( u , v , t ) p _ { i } ^ { k } ,\tag{4}
$$

with instance transmittance $\begin{array} { l r } { T _ { i } ^ { \mathrm { i n s t } } ( u , v , t ) } \end{array} = \prod _ { j < i } ( 1 \ - \begin{array} { r } { \begin{array} { r c l r } \end{array} } \end{array}$ $\pi _ { j } P _ { j } ( u , v , t ) )$ . Intuitively, $\pi _ { i }$ governs the spatial support of each Gaussian in the 4D field, while $p _ { i } ^ { k }$ encodes its semantic affiliation to instance $k .$ The rendered map ${ { \bf { M } } _ { k } }$ therefore represents a soft assignment of each pixel to instance k, jointly shaped by geometry, occupancy, and identity.

## 3.2. Instance Identity Estimation

The Consistent Instance Field defines a conditional distribution over instance identities, which can be empirically grounded in real observations and adapt to scene dynamics. We achieve this through Instance Identity Estimation, which links 2D input instance masks to the Gaussian representation through probabilistic inference. This step converts pixel-level supervision into consistent identity associations, as illustrated in Figure 2 (Bottom).

Inferring Gaussian Identities from 2D Masks. We first obtain per-frame instance masks using DEVA [5], which provides temporally consistent instance segmentation. To relate these masks to the Gaussian primitives, we exploit the splatting process to infer the fractional contribution of each Gaussian to image formation [19, 59]. For each pixel $( u , v , t )$ , we define the normalized rendering weight:

$$
w _ { i } ( u , v , t ) = \frac { T _ { i } ( u , v , t ) \alpha _ { i } ( t ) P _ { i } ( u , v , t ) } { \sum _ { j } T _ { j } ( u , v , t ) \alpha _ { j } ( t ) P _ { j } ( u , v , t ) } ,\tag{5}
$$

representing the fraction of the pixelâs color that is explained by $g _ { i }$ from a posterior perspective. Aggregating these weights over all pixels and time frames yields an empirical estimate of how often each Gaussian participates in explaining instance k:

$$
\tilde { p } _ { i } ^ { k } = \frac { \sum _ { t , ( u , v ) } \mathbf { 1 } [ \mathbf { M } _ { t } ( u , v ) = k ] ~ w _ { i } ( u , v , t ) } { \sum _ { t , ( u , v ) } w _ { i } ( u , v , t ) } ,\tag{6}
$$

$$
\hat { p } _ { i } ^ { k } = \frac { \tilde { p } _ { i } ^ { k } } { \sum _ { k ^ { \prime } } \tilde { p } _ { i } ^ { k ^ { \prime } } } .\tag{7}
$$

The resulting $\hat { p } _ { i } ^ { k }$ serves as an initialization of the Gaussianâs identity distribution, providing a visibility-weighted estimate of instance affiliation.

Visibility Bias Calibration. Since the rendered weights depend on photometric transmittance, frequently visible or well-illuminated regions may dominate supervision, while occluded or low-contrast regions are underrepresented. To compensate for this imbalance, we introduce learnable calibration factors $m _ { i } ^ { k } > 0$ that rescale the initial distribution:

$$
p _ { i } ^ { k } = \frac { \hat { p } _ { i } ^ { k } m _ { i } ^ { k } } { \sum _ { k ^ { \prime } } \hat { p } _ { i } ^ { k ^ { \prime } } m _ { i } ^ { k ^ { \prime } } } .\tag{8}
$$

These factors can absorb residual discrepancies between the visibility-biased estimate and the underlying 4D instance field. They are optimized jointly with all other Gaussian parameters through the instance-field rendering in Eq. (4). During training, gradients propagate through both occupancy $\pi _ { i }$ and calibrated identity $p _ { i } ^ { k }$ , enabling the model to refine instance assignments from purely appearance-driven initialization toward temporally consistent, geometry-aware identity estimation across the dynamic scene.

## 3.3. Instance-Guided Resampling

While the Consistent Instance Field provides a continuous description of spatial occupancy and instance identity, its discretization through a finite set of Gaussians can lead to suboptimal capacity allocation. Regions carrying strong semantic signals may be underrepresented, whereas uninformative or background areas may retain redundant primitives [22], deviating from the underlying field. To address this imbalance, we introduce Instance-Guided Resampling, an adaptive refinement mechanism that reallocates Gaussians according to the strength of their instance affiliation. We illustrate this process in Figure 2 (right).

Adaptive Redistribution. For a given instance $k ,$ we define the instance response of each Gaussian as $\gamma _ { i } ^ { k } = \pi _ { i } p _ { i } ^ { k }$ , which jointly measures its space-time occupancy $\pi _ { i }$ and semantic affinity $p _ { i } ^ { k }$ . We leverage this signal to construct two complementary sampling distributions:

$$
P _ { \mathrm { w e a k } } ( i | k ) \propto ( \gamma _ { i } ^ { k } ) ^ { - 1 } , \qquad P _ { \mathrm { s t r o n g } } ( i | k ) \propto \gamma _ { i } ^ { k } ,\tag{9}
$$

where $\gamma _ { i } ^ { k }$ is clamped to [Ïµ, 1] for numerical stability before being normalized into probability distributions.

Intuitively, $P _ { \mathrm { s t r o n g } }$ favors Gaussians that strongly support instance k, while $P _ { \mathrm { w e a k } }$ emphasizes those contributing weakly or redundantly. We then sample a weak-strong pair $( w , s )$ from $P _ { \mathrm { w e a k } }$ and $P _ { \mathrm { s t r o n g } }$ within the same instance, and treat $g _ { s }$ as a source primitive from which a new replica is spawned by reinitializing $g _ { w }$ near $g _ { s }$ and inheriting its geometric and semantic attributes. This operation transfers representational capacity to semantically active regions, encouraging the discrete ensemble to align more closely with the underlying 4D field distribution.

Volume-Conserving Adjustment. Naively replicating Gaussians without regulation may lead to local oversaturation, where multiple primitives overlap and artificially inflate both photometric contribution and semantic confidence. Let $g _ { \mathrm { s r c } }$ be the Gaussian selected for replication and n the number of replicas it has already produced. To prevent such optimization instabilities [33], we apply a volumeconserving adjustment to both opacity and occupancy for the source Gaussian and all of its new replicas:

$$
\alpha _ { \mathrm { s r c } } ^ { \mathrm { n e w } } = \alpha ^ { \mathrm { n e w } } = 1 - \left( 1 - \alpha _ { \mathrm { s r c } } \right) ^ { 1 / ( n + 1 ) } ,\tag{10}
$$

$$
\pi _ { \mathrm { s r c } } ^ { \mathrm { n e w } } = \pi ^ { \mathrm { n e w } } = 1 - \left( 1 - \pi _ { \mathrm { s r c } } \right) ^ { 1 / ( n + 1 ) } .\tag{11}
$$

This adjustment locally preserves the effective volumetric contribution of the Gaussian cluster after redistribution, preventing semantic drift or radiance inflation, and enabling the representation to refine adaptively while maintaining visual fidelity and semantic coherence.

## 3.4. Training Objective

All components of our framework are jointly optimized through differentiable field rendering. At each training iteration, both RGB images and instance maps are rendered as described in Sec. 3.1.3. The overall training objective combines photometric loss $\mathcal { L } _ { \mathrm { r g b } }$ and semantic loss ${ \mathcal { L } } _ { \mathrm { i n s t } }$ encouraging the model to produce temporally consistent, geometry-aware instance assignments. Formally, the loss is defined as:

$$
{ \mathcal { L } } = { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { i n s t } } { \mathcal { L } } _ { \mathrm { i n s t } } ,\tag{12}
$$

where $\lambda _ { \mathrm { i n s t } }$ balances the contribution of two different losses. For photometric loss ${ \mathcal { L } } _ { \mathrm { r g b } } ,$ we use $\ell _ { 1 }$ between the rendered and ground-truth RGB images: $\mathcal { L } _ { \mathrm { r g b } } { = } | | \mathbf { C } ^ { \mathrm { r e n d e r e d } } { - } \mathbf { C } ^ { \mathrm { g t } } | | _ { 1 }$ . For semantic supervision, the instance loss ${ \mathcal { L } } _ { \mathrm { i n s t } }$ is defined as the cross-entropy between the rendered and ground-truth instance masks: $\begin{array} { r l } { \mathcal { L } _ { \mathrm { i n s t } } } & { { } = } \end{array}$ $\begin{array} { r l } { - \sum _ { u , v , t } \sum _ { k } \mathbf { M } _ { k } ^ { \mathrm { g t } } ( \bar { u } , v , t ) \log \mathbf { M } _ { k } ^ { \mathrm { r e n d e r e d } } ( u , v , t ) } \end{array}$

## 4. Experiments

In this section, we first describe the experimental setup for two dynamic scene understanding tasks: novel-view panoptic segmentation, which jointly evaluates spatial accuracy and temporal consistency of instance identities, and openvocabulary 4D querying, which evaluates the ability to retrieve instances in space-time based on textual descriptions (Sec. 4.1). We then compare our proposed method, CIF, with the state-of-the-art [16, 19, 43, 50] on standard benchmarks, HyperNeRF [41] and Neu3D [26] (Sec. 4.2). Finally, we present ablation studies to evaluate the impact of different design choices in our method (Sec. 4.3).

## 4.1. Experimental Setup

Evaluation Datasets. We evaluate our proposed method on both monocular and multi-view dynamic scene datasets: HyperNeRF [41], which provides monocular videos of complex human-object interactions captured by a moving camera; and Neu3D [26], which contains synchronized multi-view recordings of complex scenes. As in previous work [16, 27], we use DEVA [5] to obtain the ground-truth instance masks as these are not provided in either the original dataset. In the multi-view dataset, to synchronize the instance identities across view, we treat the spatial multiview dimension as inter-temporal and merge all views into a single pseudo-monocular sequence. Nonetheless, segmentation can still be inconsistent due to occlusions, i.e., some objects may be fully occluded in certain views, which is inherently ill-posed. Therefore, to avoid cross-view inconsistencies in the ground truth, we only consider instances that remain visible across all camera views. More details are provided in the supplementary material.

Evaluation Metrics. To evaluate the novel-view panoptic segmentation task, we follow previous works [16, 19, 27, 29, 63], rendering the Gaussians from novel views and computing three standard metrics to assess the quality of the rendered instance masks, including: (i) mAcc-pix: the mean pixel accuracy within instance masks, computed as the average ratio of correctly labeled pixels to total pixels across all frames; (ii) mAcc-inst: the mean instance accuracy, obtained by averaging per-instance pixel-wise prediction accuracies across views, thus treating each instance equally regardless of size; (iii) mIoU: the mean Intersection-over-Union between predicted and reference instance masks, averaged over all instances and frames. For the openvocabulary 4D querying task, we follow prior work [29, 63] and, given a text prompt, use Grounded DINO [34] to generate 2D masks that are reprojected into 3D to obtain the corresponding Gaussians, which we render in the test views. Baselines. We evaluate against recent state-of-the-art methods for scene understanding. Dr. Splat [19] and VLGS [43] incorporate semantics into 3DGS, while Trace3D [50] and SA4D [16] perform instance segmentation in 3D and 4D domains, respectively. We also include 4D LangSplat [27], which is the recent state-of-the-art for open-vocabulary querying in dynamic scenes. All results on HyperNeRF and Neu3D are reproduced under our unified evaluation protocol, as the original implementations are either unavailable or limited to static 3D scenes. We adapt Dr. Splat and Trace3D for 4D data and re-implement SA4D and VLGS based on their papers.

Table 1. Quantitative comparison of our method with the state-of-the-art on novel-view panoptic segmentation using the HyperNeRF [41] dataset. We report mAcc-pix, mAcc-inst, and mIoU metrics. The best , second best , and third best results are highlighted.
<table><tr><td></td><td colspan="3">americano</td><td colspan="3">split-cookie</td><td colspan="3">chickchicken</td></tr><tr><td>Method</td><td></td><td>mAcc-pix mAcc-inst</td><td>mIoU</td><td></td><td>mAcc-pix mAcc-inst</td><td>mIoU</td><td>mAcc-pix mAcc-inst</td><td></td><td>mIoU</td></tr><tr><td>Dr. Splat [19]</td><td>87.22</td><td>62.84</td><td>56.13</td><td>89.44</td><td>63.10</td><td>56.85</td><td>84.96</td><td>57.45</td><td>52.86</td></tr><tr><td>Trace3D [50]</td><td>95.81</td><td>77.96</td><td>72.33</td><td>93.79</td><td>75.87</td><td>68.80</td><td>92.29</td><td>63.88</td><td>56.68</td></tr><tr><td>VLGS [43]</td><td>97.56</td><td>86.38</td><td>82.50</td><td>96.35</td><td>80.67</td><td>76.08</td><td>95.38</td><td>68.68</td><td>63.39</td></tr><tr><td>SA4D [16]</td><td>96.45</td><td>79.27</td><td>74.54</td><td>95.50</td><td>76.95</td><td>72.60</td><td>94.94</td><td>68.55</td><td>62.52</td></tr><tr><td>Ours</td><td>98.40</td><td>91.73</td><td>87.48</td><td>97.93</td><td>90.40</td><td>86.03</td><td>96.50</td><td>82.31</td><td>75.07</td></tr><tr><td></td><td colspan="3">espresso</td><td colspan="3">keyboard</td><td colspan="3">torchocolate</td></tr><tr><td>Method</td><td>mAcc-pix</td><td>mAcc-inst</td><td>mIoU</td><td>mAcc-pix</td><td>mAcc-inst</td><td>mIoU</td><td>mAcc-pix</td><td>mAcc-inst</td><td>mIoU</td></tr><tr><td>Dr. Splat [19]</td><td>87.22</td><td>62.84</td><td>56.13</td><td>89.44</td><td>63.10</td><td>56.85</td><td>84.96</td><td>57.45</td><td>52.86</td></tr><tr><td>Trace3D [50]</td><td>85.11</td><td>60.03</td><td>51.59</td><td>86.40</td><td>66.00</td><td>55.43</td><td>87.54</td><td>64.15</td><td>57.39</td></tr><tr><td>VLGS [43]</td><td>91.71</td><td>70.25</td><td>62.16</td><td>93.79</td><td>72.80</td><td>64.51</td><td>91.09</td><td>64.70</td><td>59.66</td></tr><tr><td>SA4D [16]</td><td>91.97</td><td>73.00</td><td>64.76</td><td>93.70</td><td>73.29</td><td>64.94</td><td>92.87</td><td>72.10</td><td>65.44</td></tr><tr><td>Ours</td><td>94.73</td><td>84.03</td><td>75.80</td><td>94.73</td><td>80.17</td><td>71.87</td><td>96.10</td><td>85.52</td><td>80.55</td></tr></table>

<!-- image-->  
Figure 3. Qualitative comparison of our method with the state-of-the-art on novel-view panoptic segmentation using the Hyper-NeRF [41] dataset. For clarity, we crop and slightly zoom in on representative regions around the manipulated objects. Our approach produces noticeably sharper and more coherent segmentations, even under occlusion and appearance variations.

Implementation Details. The Field-Aware Splatting module is implemented in CUDA, while other components are in PyTorch [42]. All experiments are run on a single NVIDIA A40 GPU. Each scene is trained for 10,000 iterations for reconstruction and 3,000 iterations for instance segmentation using Adam [23]. Learning rates are 0.01 for occupancy and instance identity calibration, while other parameters use the default values from Deformable Gaussian

Table 2. Quantitative comparison of our method with the state-of-the-art on novel-view panoptic segmentation using the Neu3D [26] dataset. We report mAcc-pix, mAcc-inst, and mIoU metrics. The best , second best , and third best results are highlighted.
<table><tr><td rowspan="2">Method</td><td colspan="3">coffee martini</td><td colspan="3">cook spinach</td><td colspan="3">cut roasted beef</td></tr><tr><td></td><td>mAcc-pix mAcc-inst</td><td>mIoU</td><td>mAcc-pix mAcc-inst</td><td></td><td>mIoU</td><td>mAcc-pix</td><td>mAcc-inst</td><td>mIoU</td></tr><tr><td>Dr. Splat [19]</td><td>88.37</td><td>73.84</td><td>70.74</td><td>78.46</td><td>76.07</td><td>69.55</td><td>53.55</td><td>30.03</td><td>25.96</td></tr><tr><td>Trace3D [50]</td><td>82.05</td><td>85.97</td><td>64.31</td><td>83.84</td><td>74.10</td><td>60.30</td><td>62.59</td><td>58.81</td><td>39.63</td></tr><tr><td>VLGS [43]</td><td>94.80</td><td>94.08</td><td>86.03</td><td>95.60</td><td>92.36</td><td>85.74</td><td>93.56</td><td>83.36</td><td>78.09</td></tr><tr><td>SA4D [16]</td><td>81.39</td><td>76.34</td><td>64.12</td><td>84.70</td><td>74.19</td><td>56.16</td><td>74.26</td><td>61.77</td><td>54.61</td></tr><tr><td>Ours</td><td>96.07</td><td>95.04</td><td>91.50</td><td>96.63</td><td>93.82</td><td>87.61</td><td>95.12</td><td>85.78</td><td>80.24</td></tr><tr><td></td><td colspan="3">flame salmon</td><td colspan="3">flame steak</td><td colspan="3">sear steak</td></tr><tr><td>Method</td><td>mAcc-pix mAcc-inst</td><td></td><td>mIoU</td><td>mAcc-pix mAcc-inst</td><td></td><td>mIoU</td><td>mAcc-pix mAcc-inst</td><td></td><td>mIoU</td></tr><tr><td>Dr. Splat [19]</td><td>81.22</td><td>78.25</td><td>74.32</td><td>69.13</td><td>62.97</td><td>57.34</td><td>73.72</td><td>74.88</td><td>66.79</td></tr><tr><td>Trace3D [50]</td><td>86.25</td><td>76.20</td><td>67.97</td><td>73.62</td><td>89.63</td><td>77.04</td><td>86.99</td><td>90.30</td><td>78.14</td></tr><tr><td>VLGS [43]</td><td>79.66</td><td>82.83</td><td>70.83</td><td>87.71</td><td>95.28</td><td>89.68</td><td>90.20</td><td>96.22</td><td>84.60</td></tr><tr><td>SA4D [16]</td><td>78.89</td><td>75.26</td><td>60.79</td><td>81.54</td><td>80.36</td><td>66.29</td><td>86.93</td><td>94.57</td><td>80.12</td></tr><tr><td>Ours</td><td>91.31</td><td>92.16</td><td>87.69</td><td>95.31</td><td>95.74</td><td>91.97</td><td>95.36</td><td>96.61</td><td>90.83</td></tr></table>

<!-- image-->  
Figure 4. Qualitative comparison of our method with the state-of-the-art on novel-view panoptic segmentation using the Neu3D [26] dataset. As described in Sec. 4.1, to avoid the inherent inconsistencies in the ground truth, we consider only instances that are visible across all camera views. Our method produces smoother boundaries, cleaner backgrounds, and more consistent object identities.

Splatting [56]. Instance-Guided Resampling uses a sampling rate of 1% of all Gaussians for HyperNeRF and 5% for Neu3D, and the instance loss weight (Î»inst) is set to 0.01 and 0.005, respectively. For mask rendering, we use argmax to obtain predicted instances without applying confidence thresholds.

## 4.2. Comparison with the State-of-the-Art

Novel-View Panoptic Segmentation. Tables 1 and 2 compare our method with prior works [16, 19, 43, 50]. Across all scenes of both datasets, our method consistently outperforms existing approaches by a large margin. On the HyperNeRF [41] dataset, our method achieves an average of 96.40 mAcc-pix, 85.69 mAcc-inst, and 79.47 mIoU, surpassing the second-best method (VLGS) by +2.09, +11.78, and +11.42, respectively. Notably, on the âtorchocolateâ scene, our method improves mIoU by +15.11 over SA4D and +20.89 over VLGS, demonstrating superior instance consistency and boundary accuracy. Similarly, on the Neu3D [26] dataset, our method achieves 94.97 mAcc-pix, 93.19 mAcc-inst, and 88.31 mIoU, outperforming VLGS by +4.72, +2.50, and +5.82, respectively. These consistent improvements across diverse dynamic scenes of both monocular and multi-view videos demonstrate the robustness and generalization ability of our proposed method.

Figures 3 and 4 further provide qualitative comparisons, which highlight the strength of the Consistent Instance Field in preserving spatial precision and semantic coherence in dynamic scenes. As illustrated in Figure 3, on the HyperNeRF dataset, our method produces finer segmentation with clearer instance separation, especially around hand-object interactions and partial occlusions, where prior methods yield fragmented or flickering masks. Similarly, as shown in Figure 4, on the Neu3D dataset, our method maintains precise object boundaries and consistent identities, even in cluttered environments. More qualitative video results are provided in the supplementary material.

<!-- image-->  
Figure 5. Qualitative comparison of our method with the state-of-the-art on open-vocabulary 4D querying using the HyperNeRF [41] dataset. For clarity, we crop and zoom in on the central regions. Our method produces clearer boundaries and more accurate instance separation, even under transparent and reflective materials such as the glass cup and steel jug.

Table 3. Ablation study. We evaluate our method under different configurations on the âsplit-cookieâ scene from HyperNeRF [41].
<table><tr><td>Method</td><td>mAcc-pix</td><td>mAcc-inst</td><td>mIoU</td><td>PSNR</td></tr><tr><td>(i) Const. Occ.</td><td>96.26</td><td>85.57</td><td>80.80</td><td>31.76</td></tr><tr><td>(ii) Opa. Occ.</td><td>96.60</td><td>87.20</td><td>82.34</td><td>32.16</td></tr><tr><td>(iii) w/o Calib.</td><td>95.99</td><td>82.65</td><td>78.16</td><td>26.73</td></tr><tr><td>(iv) w/o Resamp.</td><td>96.78</td><td>87.98</td><td>82.82</td><td>32.34</td></tr><tr><td>(v) Full</td><td>97.93</td><td>90.40</td><td>86.03</td><td>32.42</td></tr></table>

<!-- image-->  
Figure 6. Ablation study. We present the corresponding qualitative results for each configuration shown in Table 3.

Open-Vocabulary 4D Querying. Figure 5 presents qualitative comparisons on open-vocabulary 4D querying against the recent state-of-the-art method for dynamic scene understanding, 4D LangSplat [27]. For comprehensive evaluation, we also extend SA4D [16] to this task by using Grounded DINO [34] for text-mask correspondence. Our approach achieves more accurate instance localization and sharper boundaries, even under challenging visual conditions involving transparent or reflective objects (e.g., the glass cup and steel jug). In contrast, both baselines suffer from boundary leakage or semantic confusion. Additional video and quantitative results are provided in the supplementary material.

## 4.3. Ablation Studies

We conduct controlled ablations to assess the impact of each component in our proposed method in Table 3. Each variant modifies a single module while keeping all other settings identical to the full method, ensuring fair and isolated comparisons. We consider five configurations: (i) Constant Occupancy, where occupancy (Ïi) is fixed to 0.02, removing spatial adaptivity; (ii) Opacity as Occupancy, using RGB opacity as a surrogate for occupancy, which blurs the distinction between existence and visibility; (iii) w/o Identity Calibration, removing the calibration term $( m _ { i } ^ { k } )$ , allowing 2D visibility bias to propagate into the 4D instance distribution; (iv) w/o Instance-Guided Resampling, disabling resampling so Gaussian capacity may be redundant in background but insufficient in active regions; and (v) Full, our complete approach. The results show that removing calibration (iii) or resampling (iv) significantly degrades performance, producing inconsistent and noisy segmentation. Using opacity as occupancy (ii) retains coarse geometry but fails to preserve instance consistency, while constant occupancy (i) also reduces performance due to lack of spatial adaptivity. In contrast, the full method (v), which jointly learns adaptive occupancy, calibrated identities, and resampled Gaussians, achieves the best balance of geometric fidelity and semantic consistency. Figure 6 qualitatively illustrates that only the full method maintains coherent segmentation through complex hand-object interactions.

## 5. Conclusion

We presented the Consistent Instance Field, a unified probabilistic framework for dynamic scene understanding that jointly models geometry, motion, and semantics in a continuous 4D representation. CIF disentangles visibility from identity and leverages instance-field signals to adaptively organize Gaussians around semantically meaningful entities, improving both spatial accuracy and temporal coherence. We demonstrate that CIF significantly outperforms existing methods on both novel-view panoptic segmentation and open-vocabulary 4D querying tasks. For future work, we aim to introduce a standardized 4D evaluation benchmark for open-vocabulary 4D querying, enabling more consistent assessment and driving further advances in 4D instance-level modeling.

## References

[1] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun Bang, and Youngjung Uh. Per-gaussian embedding-based deformation for deformable 3d gaussian splatting. In ECCV, 2024. 1

[2] Yanqi Bao, Tianyu Ding, Jing Huo, Yaoli Liu, Yuxin Li, Wenbin Li, Yang Gao, and Jiebo Luo. 3d gaussian splatting: Survey, technologies, challenges, and opportunities. In IEEE TCSVT, 2025. 2

[3] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3d gaussians. In AAAI, 2025. 2

[4] Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890, 2024. 2

[5] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In ICCV, 2023. 2, 4, 5, 12, 13

[6] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. Pla: Language-driven openvocabulary 3d scene understanding. In CVPR, 2023. 2

[7] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes. In SIGGRAPH, 2024. 2

[8] Bardienus P Duisterhof, Zhao Mandi, Yunchao Yao, Jia-Wei Liu, Mike Zheng Shou, Shuran Song, and Jeffrey Ichnowski. Md-splatting: Learning metric deformation from 4d gaussians in highly deformable scenes. arXiv preprint arXiv:2312.00583, 2023. 1

[9] Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, and Federico Tombari. Opennerf: open set 3d neural scene segmentation with pixelwise features and rendered novel views. arXiv preprint arXiv:2404.03650, 2024. 1, 2

[10] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias NieÃner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia, 2022. 2

[11] Kyle Gao, Yina Gao, Hongjie He, Dening Lu, Linlin Xu, and Jonathan Li. Nerf: Neural radiance field in 3d vision, a comprehensive review. arXiv preprint arXiv:2210.00379, 2022. 2

[12] Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In ICRA, 2024. 1

[13] Mohamed Amine Guerroudji, Kahina Amara, Mohamed Lichouri, Nadia Zenati, and Mostefa Masmoudi. A 3d visualization-based augmented reality application for brain tumor segmentation. Computer Animation and Virtual Worlds, 2024. 1

[14] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In CVPR, 2024. 1

[15] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, et al. Conceptfusion: Open-set multimodal 3d mapping. arXiv preprint arXiv:2302.07241, 2023. 2

[16] Shengxiang Ji, Guanjun Wu, Jiemin Fang, Jiazhong Cen, Taoran Yi, Wenyu Liu, Qi Tian, and Xinggang Wang. Segment any 4d gaussians. arXiv preprint arXiv:2407.04504, 2024. 1, 2, 5, 6, 7, 8, 12, 13

[17] Wei Jiang, Kwang Moo Yi, Golnoosh Samei, Oncel Tuzel, and Anurag Ranjan. Neuman: Neural human radiance field from a single video. In ECCV, 2022. 2

[18] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin Yang, et al. Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In ACM SIG-GRAPH 2024 Conference Papers, pages 1â1, 2024. 1

[19] Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. splat: Directly referring 3d gaussian splatting via direct language embedding registration. In CVPR, 2025. 2, 4, 5, 6, 7, 13

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. In ACM TOG, 2023. 1, 2, 4

[21] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In ICCV, 2023. 1, 2

[22] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splatting as markov chain monte carlo. In NeurIPS, 2024. 4

[23] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 6

[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 1, 2

[25] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via feature field distillation. In NeurIPS, 2022. 2

[26] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe,

et al. Neural 3d video synthesis from multi-view video. In CVPR, 2022. 5, 7, 12, 13

[27] Wanhua Li, Renping Zhou, Jiawei Zhou, Yingwei Song, Johannes Herter, Minghan Qin, Gao Huang, and Hanspeter Pfister. 4d langsplat: 4d language gaussian splatting via multimodal large language models. In CVPR, 2025. 1, 2, 5, 8, 12, 13

[28] Yulong Li and Deepak Pathak. Object-aware gaussian splatting for robotic manipulation. In ICRA Workshop, 2024. 1

[29] Yun-Jin Li, Mariia Gladkova, Yan Xia, and Daniel Cremers. Sadg: Segment any dynamic gaussian without object trackers. arXiv preprint arXiv:2411.19290, 2024. 1, 2, 5

[30] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In CVPR, 2024. 2

[31] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, and Shijian Lu. Weakly supervised 3d openvocabulary segmentation. In NeurIPS, 2023. 2

[32] Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, and Hao Su. Partslip: Low-shot part segmentation for 3d point clouds via pretrained image-language models. In CVPR, 2023. 2

[33] Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and Andrew Feng. Deformable beta splatting. In ACM SIGGRAPH, 2025. 5

[34] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024. 5, 8, 13

[35] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 3DV, 2024. 2, 3

[36] Weijie Lyu, Xueting Li, Abhijit Kundu, Yi-Hsuan Tsai, and Ming-Hsuan Yang. Gaga: Group any gaussians via 3d-aware memory bank. arXiv preprint arXiv:2404.07977, 2024. 2

[37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[38] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 2022. 1

[39] Maxime Oquab, Timothee Darcet, Th Â´ eo Moutakanni, Huy Â´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2

[40] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In ICCV, 2021. 1, 2

[41] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higher-

dimensional representation for topologically varying neural radiance fields. In SIGGRAPH Asia, 2021. 1, 2, 5, 6, 7, 8, 13

[42] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019. 6

[43] Qucheng Peng, Benjamin Planche, Zhongpai Gao, Meng Zheng, Anwesa Choudhuri, Terrence Chen, Chen Chen, and Ziyan Wu. 3d vision-language gaussian splatting. arXiv preprint arXiv:2410.07577, 2024. 2, 4, 5, 6, 7, 13

[44] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In CVPR, 2023. 1, 2

[45] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In CVPR, 2021. 1

[46] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In CVPR, 2024. 1, 2

[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 2

[48] Adam Rashid, Satvik Sharma, Chung Min Kim, Justin Kerr, Lawrence Yunliang Chen, Angjoo Kanazawa, and Ken Goldberg. Language embedded radiance fields for zero-shot taskoriented grasping. In CoRL, 2023. 1

[49] Hannah Schieber, Jacob Young, Tobias Langlotz, Stefanie Zollmann, and Daniel Roth. Semantics-controlled gaussian splatting for outdoor scene reconstruction and rendering in virtual reality. In 2025 IEEE Conference Virtual Reality and 3D User Interfaces (VR), 2025. 1

[50] Hongyu Shen, Junfeng Ni, Yixin Chen, Weishuo Li, Mingtao Pei, and Siyuan Huang. Trace3d: Consistent segmentation lifting via gaussian instance tracing. In ICCV, 2025. 1, 2, 5, 6, 7, 13

[51] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for openvocabulary scene understanding. In CVPR, 2024. 1, 2

[52] Ola Shorinwa, Johnathan Tucker, Aliyah Smith, Aiden Swann, Timothy Chen, Roya Firoozi, Monroe Kennedy III, and Mac Schwager. Splat-mover: Multi-stage, openvocabulary robotic manipulation via editable gaussian splatting. arXiv preprint arXiv:2405.04378, 2024. 1

[53] Liangchen Song, Xuan Gong, Benjamin Planche, Meng Zheng, David Doermann, Junsong Yuan, Terrence Chen, and Ziyan Wu. Pref: Predictability regularized neural motion fields. In ECCV, 2022. 2

[54] Yuhao Su, Anwesa Choudhuri, Zhongpai Gao, Benjamin Planche, Van Nguyen Nguyen, Meng Zheng, Yuhan Shen, Arun Innanje, Terrence Chen, Ehsan Elhamifar, et al. Medgrpo: Multi-task reinforcement learning for heterogeneous medical video understanding. In NeurIPS, 2025. 1

[55] Feiran Wang, Jiachen Tao, Junyi Wu, Haoxuan Wang, Bin Duan, Kai Wang, Zongxin Yang, and Yan Yan. X-field: A

physically grounded representation for 3d x-ray reconstruction. In NeurIPS, 2025. 2

[56] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, 2024. 2, 3, 7, 13

[57] Junyi Wu, Jiachen Tao, Haoxuan Wang, Gaowen Liu, Ramana Rao Kompella, and Yan Yan. Orientation-anchored hyper-gaussian for 4d reconstruction from casual videos. In NeurIPS, 2025. 2

[58] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, et al. Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. In NeurIPS, 2024. 1, 2

[59] Butian Xiong, Rong Liu, Kenneth Xu, Meida Chen, and Andrew Feng. Splat feature solver. arXiv preprint arXiv:2508.12216, 2025. 4

[60] Jiawei Xu, Kai Deng, Zexin Fan, Shenlong Wang, Jin Xie, and Jian Yang. Ad-gs: Object-aware b-spline gaussian splatting for self-supervised autonomous driving. In ICCV, 2025. 1

[61] Jihan Yang, Runyu Ding, Weipeng Deng, Zhe Wang, and Xiaojuan Qi. Regionplc: Regional point-language contrastive learning for open-world 3d scene understanding. In CVPR, 2024. 2

[62] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In CVPR, 2024. 2, 3, 13

[63] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In ECCV, 2024. 2, 5

[64] Jihwan Yoon, Sangbeom Han, Jaeseok Oh, and Minsik Lee. Splinegs: Learning smooth trajectories in gaussian splatting for dynamic scene reconstruction. In ICLR, 2025. 2

[65] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In CVPR, 2024. 1

[66] Jiaxuan Zhu and Hao Tang. Dynamic scene reconstruction: Recent advance in real-time rendering and streaming. arXiv preprint arXiv:2503.08166, 2025. 2

[67] Ruijie Zhu, Mulin Yu, Linning Xu, Lihan Jiang, Yixuan Li, Tianzhu Zhang, Jiangmiao Pang, and Bo Dai. Objectgs: Object-aware scene reconstruction and scene understanding via gaussian splatting. In ICCV, 2025. 2

[68] Siting Zhu, Guangming Wang, Xin Kong, Dezhi Kong, and Hesheng Wang. 3d gaussian splatting in robotics: A survey. arXiv preprint arXiv:2410.12262, 2024. 1

# Consistent Instance Field for Dynamic Scene Understanding

Supplementary Material

Overview. In the supplementary material, we first present our proposed multi-view instance segmentation to get crossview consistent instance masks on the Neu3D dataset (Sec. A.1). We then present additional qualitative video results (Sec. B.1) and quantitative results (Sec. B.2). Finally, we discuss the limitation of our method (Sec. C.1) and the societal impact (Sec. C.2).

## A. Implementation Details

## A.1. Multi-View Consistent Instance Segmentation

Pre-processing (merging multi-view videos). As discussed in Sec. 4.1 of the main paper, the multi-view benchmark Neu3D [26] provides synchronized videos but does not include ground-truth instance annotations. Therefore, we follow prior works [16, 27] and use DEVA [5], a video object tracking model, to generate input masks. However, because DEVA processes each video independently, the resulting masks are inconsistent across views. To address this limitation, we reinterpret the spatial multi-view dimension as an inter-temporal one and merge all views into a single pseudo-monocular sequence. More precisely, we concatenate each video with the reversed video of its adjacent view. Formally, given N spatially adjacent camera views and T frames per video corresponding to each view, we first reorder frames to maximize temporal continuity:

$$
\begin{array} { r } { { S } _ { n } ^ { \mathrm { o r d e r e d } } = \left\{ \begin{array} { l l } { ( I _ { 1 } , I _ { 2 } , \ldots , I _ { T } ) , } & { \mathrm { i f } n \mathrm { i s ~ o d d } , } \\ { \left( I _ { T } , I _ { T - 1 } , \ldots , I _ { 1 } \right) , } & { \mathrm { i f } n \mathrm { i s ~ e v e n } , } \end{array} \right. } \end{array}\tag{S13}
$$

where $I _ { t }$ denotes the t-th frame from that n-th view. The input to DEVA is then constructed by concatenating all reordered view sequences:

$$
\begin{array} { r } { { S } _ { 1 : N } ^ { \mathrm { o r d e r e d } } = \left[ S _ { 1 } ^ { \mathrm { o r d e r e d } } , S _ { 2 } ^ { \mathrm { o r d e r e d } } , \dots , S _ { N } ^ { \mathrm { o r d e r e d } } \right] . } \end{array}\tag{S14}
$$

This merged pseudo-monocular sequence ensures that adjacent frames vary smoothly across both time and viewpoint. Video instance segmentation. Given a concatenated video $S _ { 1 : N } ^ { \mathrm { o r d e r e d } }$ , DEVA then produces per-frame instance masks with temporally propagated instance IDs.

Post-processing (visibility filtering). Due to occlusions and limited camera overlap, some instances may disappear entirely in certain views, leading to conflicting identities across cameras. To prevent incomplete or inconsistent masks, we retain only instances that remain visible in all views. Specifically, an instance k is considered valid if its mask has non-empty support in every view of the concatenated videos. Instances that do not meet this criterion are discarded.

Figure S1 compares the default DEVA results, where each video is segmented independently, with our multi-view

<!-- image-->

<!-- image-->  
Default DEVA

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Our Multi-View Consistent Instance Segmentation

Figure S1. Results of multi-view video segmentation on Neu3D [26] dataset. (Left) DEVA [5] produces instance masks independently per camera view, leading to inconsistent instance identities across synchronized camera streams, (Right) our multi-view video segmentation results by merging multi-videos into a single pseudo-monocular sequence to produce multi-view consistent instance masks.

Table S1. Quantitative comparison of our method with the state-of-the-art on open-vocabulary 4D querying using the HyperNeRF [41] dataset. We report mAcc and mIoU metrics. The best , second best , and third best results are highlighted. \* indicates failure of localizing the objects based on the text queries, as also demonstrated in Figure 5 of the main paper.
<table><tr><td></td><td colspan="4">americano</td><td colspan="4">espresso</td></tr><tr><td></td><td colspan="2">&quot;glass cup&quot;</td><td colspan="2"> $\ " m a t \prime \prime$ </td><td colspan="2">&quot;steel jug&quot;</td><td colspan="2">&quot;coaster&quot;</td></tr><tr><td>Method</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td></tr><tr><td>4D LangSplat [27]</td><td>98.02</td><td>78.17</td><td>72.90</td><td>7.55*</td><td>96.33</td><td>35.61</td><td>86.69</td><td>0.43*</td></tr><tr><td>SA4D [16]</td><td>88.91</td><td>40.05</td><td>71.65</td><td>39.67</td><td>99.25</td><td>70.49</td><td>99.56</td><td>81.12</td></tr><tr><td>Ours</td><td>99.02</td><td>88.52</td><td>94.68</td><td>77.66</td><td>99.72</td><td>87.93</td><td>99.73</td><td>85.47</td></tr></table>

consistent instance segmentation. This strategy effectively converts the multi-view videos into a single coherent identity sequence, enabling cross-view consistent pseudo-labels for both supervision and evaluation.

## B. Experimental Results

## B.1. Additional Qualitative Results

To further demonstrate that our Consistent Instance Field provides coherent instance understanding across both space and time, we include additional qualitative video results comparing our method with recent state-of-the-art approaches SA4D [16], Trace3D [50], Dr.Splat [19], and VLGS [43] on the standard HyperNeRF [41] and Neu3D [26] benchmarks for both novel-view panoptic segmentation and open-vocabulary 4D querying tasks:

â¢ Results on the monocular benchmark HyperNeRF [41] for novel-view panoptic segmentation: panoptic segmentation hypernef.mp4

â¢ Results on the multi-view benchmark Neu3D [26] for novel-view panoptic segmentation: panoptic segmentation neu3d.mp4;

â¢ Results on the monocular benchmark HyperNeRF [41] for open-vocabulary 4D querying:

open vocabulary 4d querying hypernerf.mp4

## B.2. Additional Quantitative Results

To further assess the capability of the Consistent Instance Field in open-vocabulary 4D querying, we report additional quantitative results using standard metrics, mAcc and mIoU. We follow the experimental setting described in the main paper and use Grounded DINO [34] to obtain 2D masks for each text query as pseudo-ground-truth annotations.

As shown in Table S1, our method consistently achieves higher retrieval accuracy across various text prompts. On average, our method achieves 98.29 mAcc and 84.90 mIoU, surpassing the second-best approach (SA4D) by 8.45 and 27.07, respectively. In contrast, previous methods like 4D LangSplat [27] often struggle to localize the objects based on the text query, as demonstrated in Figure 5 of the main paper, thus yielding extremely low performance (e.g., 7.55 mIoU on the âamericanoâ scene with the âmatâ query and 0.43 on the âespressoâ scene with the âcoasterâ query). These results highlight the strength of our method in finegrained and coherent instance modeling in complex dynamic scenes.

## C. Discussions

## C.1. Limitations

While the proposed Consistent Instance Field provides a principled formulation for identity modeling in dynamic scenes, several limitations remain. First, our formulation is instantiated via a deformable Gaussian representation [56, 62], which inherits its representational constraints. Scenes involving amorphous or continuously evolving materials (e.g., smoke or liquids) lack stable structure and may not be faithfully represented through persistent Gaussian primitives. In such cases, identity assignments become less interpretable, as the Gaussians fail to maintain consistent spatial support or correspond to physically persistent entities. Future advances in dynamic scene representations may help extend our method to broader scene types. Second, although we construct pseudo-monocular sequences to synchronize multi-view pseudo-labels from video object tracking models [5], residual cross-view inconsistencies or missing annotations under severe occlusion can still bias identity estimation. Exploring more rigorous multi-view pseudolabel harmonization strategies represents a promising direction for enhancing robustness in such scenarios.

## C.2. Societal Impact

Our approach provides structured, instance-consistent scene representations that extend beyond visual reconstruction to support simulation, prediction, and interaction within dynamic environments. These advances could improve the safety, efficiency, and interpretability of autonomous and interactive systems, provided that ethical and privacy standards are respected.