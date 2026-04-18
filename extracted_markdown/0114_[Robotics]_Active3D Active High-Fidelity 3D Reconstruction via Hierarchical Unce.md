# Active3D: Active High-Fidelity 3D Reconstruction via Hierarchical Uncertainty Quantification

Yan Li1, Yingzhao Li2, Gim Hee Lee1

1National University of Singapore   
2Harbin Institute of Technology

## Abstract

In this paper, we present an active exploration framework for high-fidelity 3D reconstruction that incrementally builds a multi-level uncertainty space and selects next-best-views through an uncertainty-driven motion planner. We introduce a hybrid implicitâexplicit representation that fuses neural fields with Gaussian primitives to jointly capture global structural priors and locally observed details. Based on this hybrid state, we derive a hierarchical uncertainty volume that quantifies both implicit global structure quality and explicit local surface confidence. To focus optimization on the most informative regions, we propose an uncertainty-driven keyframe selection strategy that anchors high-entropy viewpoints as sparse attention nodes, coupled with a viewpoint-space sliding window for uncertainty-aware local refinement. The planning module formulates next-best-view selection as an Expected Hybrid Information Gain problem and incorporates a risk-sensitive path planner to ensure efficient and safe exploration. Extensive experiments on challenging benchmarks demonstrate that our approach consistently achieves state-ofthe-art accuracy, completeness, and rendering quality, highlighting its effectiveness for real-world active reconstruction and robotic perception tasks.

Website â https://yanyan-li.github.io/project/vlx/active3d

## 1 Introduction

Visual-based 3D reconstruction (Newcombe et al. 2011; Whelan et al. 2015; Dai et al. 2017; Li et al. 2020) aims to infer the geometry and appearance of previously unseen scenes from 2D imagery, making it a fundamental problem in both computer vision and robotics. Depending on how the sensor moves, reconstruction methods can be clustered into two categories: passive and active. Passive systems process streams of RGB (Schonberger and Frahm 2016) or RGB-D (Li and Tombari 2022) frames to jointly estimate six-degree-of-freedom (6-DoF) camera motions and fuse the measurements into sparse or dense 3D models, under the assumption of a fixed, user-driven path. In contrast, active reconstruction frameworks (Aloimonos, Weiss, and Bandyopadhyay 1988; Chen, Li, and Kwok 2011) integrate next-best-view (NBV) planning (Peralta et al. 2020) to autonomously select subsequent viewpoints to maximize information gain (Isler et al. 2016; Kirsch, Van Amersfoort, and Gal 2019) and ensure comprehensive surface coverage. In addition to accurate geometry, next-generation intelligent robots demand dense 3D models with high fidelity and photometric consistency for downstream tasks.

<!-- image-->  
(a) Rendering-reconstruction Curve (b) Our Model's Output Figure 1: Performance on the Replica dataset. Left: Comparison of rendering quality (PSNR) versus reconstruction completeness (C.R.) across state-of-the-art methods. Right: Qualitative outputs of our method including reconstructed mesh, Gaussian map, and estimated depth.

The conventional active reconstruction problem (Isler et al. 2016; Huang et al. 2018) is typically cast as an exploration task: select the sequence of viewpoints that will most effectively reveal detailed scene geometry and appearance. Early approaches leverage occupancy-grid (Elfes 2013) or voxel-based (Wu et al. 2014) maps and frontierdriven exploration to push the boundary between known and unknown space, ensuring that new measurements continually reduce map uncertainty (Lee et al. 2022). However, since these approaches focus solely on geometric uncertainty, the resulting reconstructions are ill-suited for highquality novel-view rendering and often lack the photometrically consistent details required for downstream tasks. Recent advances in scene representation have revealed two complementary paradigms: implicit neural fields (Mildenhall et al. 2021; Barron et al. 2022) and explicit parameterizations such as 3D Gaussian Splatting (Kerbl et al. 2023; Li et al. 2024), both achieving impressive performance in novel-view synthesis and surface reconstruction. Implicit models encode continuous neural fields that excel at capturing global structure, while explicit Gaussians faithfully preserve observed geometry and fine details. However, existing active frameworks typically adopt only one of these paradigms. Implicit-based active methods (Yan, Yang, and Zha 2023; Kuang et al. 2024) leverage neural priors for view planning, but their continuous fields tend to hallucinate missing surfaces (e.g., transparent or mirrored areas), leading to persistent high uncertainty and planner oscillation. Conversely, GS-based active approaches (Li et al. 2025; Jin et al. 2025) directly reflect observations into the map, providing reliable local geometry but lacking the ability to reason about occluded or unseen regions, resulting in suboptimal exploration coverage.

These complementary strengths and limitations motivate a hybrid implicitâexplicit formulation for active reconstruction, unifying global priors and local textured surface within a single information-theoretic planning framework. First, given a posed RGB-D stream, Active3D constructs a hybrid implicitâexplicit scene state and derives a hierarchical uncertainty map to jointly quantify global structural entropy and local surface uncertainty. Based on this hybrid uncertainty, the planner is further proposed to formulate next-best-view selection as an Expected Hybrid Information Gain (EHIG) optimization and executes viewpointaware trajectory planning. Keyframes are promoted via a dual-uncertainty intersection criterion, selecting viewpoints that observe regions where both implicit and explicit uncertainties are high. This establishes a sparse attention mechanism over the hybrid scene state. A viewpoint-space sliding window then performs uncertainty-aware local refinement of Gaussian primitives with respect to implicit priors, maintaining globalâlocal consistency throughout the reconstruction process. Our contributions are summarized as follows:

â¢ We propose a hybrid implicitâexplicit scene representation for active 3D reconstruction, unifying neural fields and Gaussian primitives into a joint entropy minimization framework and introducing the Hybrid Scene State Entropy.

â¢ We design a hierarchical uncertainty map that fuses global implicit variance, local depth residuals, local photometric residuals, and temporal SDF changes via Bayesian fusion, providing a principled multi-scale signal to drive exploration and refinement.

â¢ We formulate next-best-view planning as an Expected Hybrid Information Gain (EHIG) problem, combining global structural exploration and local detail preservation with risk-aware path optimization.

â¢ We introduce a viewpoint-aware keyframe selection strategy driven by the intersection of implicit and explicit uncertainties, anchoring high-information regions as sparse attention nodes in the hybrid map. Integrated with a spatial (non-temporal) sliding window, this enables uncertaintyaware local refinement and consistent reconstruction of the hybrid scene state.

## 2 Related Work

Neural Implicit and Explicit Representation. Traditionally, 3D reconstructed models have been represented using various geometric formats, including meshes (Kazhdan, Bolitho, and Hoppe 2006; Li et al. 2021), surfels (Whelan et al. 2015; Stuckler and Behnke 2014), and truncated Â¨ signed distance fields (TSDF) (Osher, Fedkiw, and Piechor 2004; Izadi et al. 2011). With the advent of differentiable radiance fields, these representations have been significantly extended to support high-quality novel view synthesis. In particular, NeRF (Mildenhall et al. 2021) have emerged as a powerful paradigm for photorealistic rendering and scene understanding. Specifically, iMAP (Sucar et al. 2021) utilizes MLP as the only scene representation for both tracking and mapping. To address the over smoothed reconstruction problem of only-MLP representation in large-scale environments, NeuralRecon (Sun et al. 2021) integrates neural TSDF volumes with learned features to enhance 3D reconstruction quality in indoor scenes. Similarly, ConvONet (Peng et al. 2020) predicts occupancy probabilities in 3D space using 3D convolutional architectures (CÂ¸ icÂ¸ek et al. 2016; Ronneberger, Fischer, and Brox 2015; Niemeyer et al. 2020), combining the strengths of spatially aware feature encoding and implicit shape modeling.

In contrast to implicit and hybrid approaches, explicit representations directly encode scene geometry and appearance in structured forms such as voxel grids (Muller et al. 2022) Â¨ or Gaussian primitives (Kerbl et al. 2023), enabling efficient rendering and fast optimization. Plenoxels (Fridovich-Keil et al. 2022) replace MLPs with a sparse voxel grid that stores density and spherical harmonics coefficients. TensoRF (Chen et al. 2022) further improves scalability and memory efficiency by applying low-rank tensor decomposition. More recently, 3D Gaussian Splatting (Kerbl et al. 2023; Li et al. 2024) introduces a point-based explicit method where each Gaussian encodes position, orientation, scale, and radiance attributes, supporting high-fidelity rendering with real-time performance and continuous surfaces.

Active High-quality 3D Modeling. Active reconstruction methods (Yan, Yang, and Zha 2023; Kuang et al. 2024; Pan et al. 2022; Li et al. 2025; Jin et al. 2024; Feng et al. 2024; Chen et al. 2025) autonomously select viewpoints during iterative mapping to maximize coverage and reconstruction quality. NeRF-based NBV strategies (Lee et al. 2022; Pan et al. 2022) use pixel-wise rendering variance as uncertainty cues, while FisherRF (Jiang, Lei, and Daniilidis 2024) introduces Fisher information for view planning. ANM (Yan, Yang, and Zha 2023) maintains weight-space uncertainty in a continually learned neural field, and NARUTO (Feng et al. 2024) extends this paradigm to 6-DoF exploration in largescale scenes.

Recently, Gaussian primitives have been adopted for active scene modeling. ActiveGAMER (Chen et al. 2025) incorporates rendering quality into the information gain metric. GS-Planner (Jin et al. 2024) detects unobserved regions in the Gaussian map and employs a sampling-based NBV policy. HGS (Xu et al. 2024) proposes an adaptive hierarchical planning strategy balancing global and local refinement. ActiveSplat (Li et al. 2025) extends Gaussian-based SLAM to active mapping with decoupled viewpoint orientation.

Uncertainty estimation plays a central role in NBV selection. NeRF-based methods typically derive voxel or pixelwise variance from density fields (Pan et al. 2022; Lee et al. 2022), while Gaussian-based methods rely on observation completeness or visibility priors (Jin et al. 2024; Li et al. 2025). In contrast, we fuse global implicit variance, local surface residuals, and temporal SDF variation, constructing a hierarchical uncertainty map that simultaneously guides global exploration and local refinement.

## 3 Methodology

In the active reconstruction task, the core of the problem is to decide the position and orientation of the $i ^ { t h }$ viewpoint based on the information captured by the previous posed RGB-D stream $\begin{array} { r c l } { S _ { i - 1 } } & { = } & { \{ { \bf S } _ { k } \} , { \bf S } _ { k } } & { = } \end{array}$ $\{ I _ { k } , D _ { k } , \hat { \mathbf { T } _ { c _ { k } , w } } , \mathbf { K } \} , k \in \mathsf { ~ [ 0 , 1 , \ldots , } i - 1 ]$ . Therefore, the problem can be defined as determining how to leverage the previously posed RGB-D stream to guide the selection of the current viewpoint in order to achieve high-quality reconstruction. This process first involves the data organization of the previous RGB-D stream, followed by quantifying the historical information to evaluate the current reconstruction state and predicting potential information gain. By modeling the scene coverage, uncertainty distribution, and geometric consistency from $\mathcal { S } _ { i - 1 }$ , the system can actively plan the next viewpoint that maximizes scene completeness and reconstruction fidelity. Fig. 2 depicts the algorithmâs workflow.

## Hybrid Implicit-explicit Space

To simultaneously capture continuous global priors and high-quality local surface, we construct a hybrid implicitâexplicit space that integrates implicit neural fields with explicit Gaussian primitives. Given a posed RGB-D observation $\mathbf { S } _ { k } = [ I _ { k } , D _ { k } ^ { - } , \mathbf { T } _ { c _ { k } , w } , \mathbf { K } ]$ , this hybrid space provides a unified state representation for incremental active reconstruction.

Definition of Hybrid Scene State. We introduce a state formulation for the incremental active reconstruction task, where the state $\mathcal { M } _ { k }$ at step k is designed to represent the currently reconstructed portion of the scene:

$$
\boldsymbol { \mathcal { M } } _ { k } = \{ \boldsymbol { \mathcal { F } } _ { \boldsymbol { \theta } } , \boldsymbol { \mathcal { G } } _ { k } \} ,\tag{1}
$$

where $\mathcal { F } _ { \theta } : \mathbb { R } ^ { 3 }  \mathrm { S D F }$ is the implicit neural field, and $\mathcal { G } _ { k } =$ $\{ G _ { i } \} _ { i = 1 } ^ { N _ { k } }$ is the set of 3D Gaussian primitives. Each primitive $G _ { i }$ is parameterized as $G _ { i } = ( \mu _ { i } , \Sigma _ { i } , \alpha _ { i } , c _ { i } )$ , where $\mu _ { i } ~ \in$ $\mathbb { R } ^ { 3 }$ is the mean position, $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ the covariance, $\alpha _ { i } \in$ [0, 1] the opacity, and $c _ { i } \in \mathbb { R } ^ { 3 }$ the color vector. And for the implicit neural field, we employ a One-blob encoder (Wang, Wang, and Agapito 2023; Muller et al. 2019) to extract deep Â¨ features from input point clouds. The implicit representation subsequently maps world coordinates $\mathbf { x } \in \mathbb { R } ^ { 3 }$ to SDF values and color attributes via the MLP:

$$
s = f _ { \tau } \bigl ( \gamma ( \mathbf { x } ) , \mathcal { V } _ { \alpha } ( \mathbf { x } ) \bigr )\tag{2}
$$

where $\gamma ( \mathbf { x } )$ denotes tri-plane decomposition of spatial coordinates, and $\mathcal { V } _ { \alpha } ( \mathbf { x } )$ represents position feature vectors obtained through volumetric trilinear interpolation. The function $f _ { \tau } ( \cdot )$ corresponds to the geometry decoder.

Hybrid State Quantification. At state k, the key objective is to quantify the current scene knowledge and guide the next-best-view selection. This hybrid formulation bridges global structural exploration driven by $\mathcal { F } _ { \theta }$ and local highfidelity surface enabled by $\mathcal { G } _ { k }$ . Casting NBV planning as an expected hybrid information gain optimization, we formalize active reconstruction in a probabilistic informationtheoretic context.

We define the voxel-wise hybrid entropy as:

$$
H _ { \mathrm { h y b r i d } } ( v ) = \lambda _ { \mathrm { i m p } } H [ p _ { \mathcal F _ { \theta } } ( v ) ] + \lambda _ { \mathrm { e x p } } H [ p _ { \mathcal G _ { k } } ( v ) ] ,\tag{3}
$$

where $H [ p ]$ denotes Shannon entropy and $\lambda _ { \mathrm { i m p } } , \lambda _ { \mathrm { e x p } }$ balance global priors and local observations.

The NBV reward for c is accumulated over all visible voxels:

$$
R ( \mathbf { c } ) = \sum _ { v \in \mathcal { V } _ { \mathbf { c } } } w ( v | \mathbf { c } ) ( 1 - O ( v ) ) ,\tag{4}
$$

where $\nu _ { \mathbf { c } }$ is the set of voxels visible from c, and $O ( v )$ is the occupancy probability used to discount free-space ambiguity.

## Hierarchical Uncertainty Map Construction

To drive the hybrid NBV objective in Eq. 3, we construct a hierarchical uncertainty volume $\mathcal { V } _ { u } \in \mathbb { R } ^ { \mathbf { \tilde { L } } \times \mathbf { \tilde { W } } \times H }$ that fuses global implicit priors, local view-dependent surface, and temporal consistency cues. Each voxel v stores a scalar $u ( v ) \in \mathbb { R } ^ { + }$ representing the hybrid reconstruction confidence.

Global Structure Uncertainty. The implicit branch $\mathcal { F } _ { \theta }$ encodes a continuous SDF-based representation that provides global structural entropy. We approximate per-voxel variance using an uncertainty head $f _ { \delta } ( \cdot ) \colon$

$$
u _ { \mathrm { i m p } } ( v ) = \phi \big ( f _ { \delta } ( \gamma ( \mathbf x _ { v } ) , \mathcal V _ { \alpha } ( \mathbf x _ { v } ) ) \big ) ,\tag{5}
$$

where $\mathbf { x } _ { v }$ denotes the voxel center, $\gamma ( \cdot )$ is the tri-plane encoder, and $\phi ( \cdot )$ applies a softplus normalization. Upon receiving new observations, the structural uncertainty is updated, encouraging coverage-driven exploration and mitigating local greedy behavior during the early stages of mapping.

View-dependent Local Uncertainty. The explicit Gaussian map $\mathcal { G } _ { k }$ provides local observation entropy through photometric and geometric residuals. At each step, we select top-K high-uncertainty candidate viewpoints $\mathcal { C } _ { \mathrm { h i g h } }$ and compute depth and color errors:

$$
E _ { t } ^ { \mathrm { d e p t h } } = \left. D _ { t } ^ { \mathrm { r e n d e r } } - D _ { t } ^ { \mathrm { g t } } \right. \odot M _ { t } ,\tag{6}
$$

$$
E _ { t } ^ { \mathrm { r g b } } = \sum _ { c \in \{ R , G , B \} } \left\| I _ { t , c } ^ { \mathrm { r e n d e r } } - I _ { t , c } ^ { \mathrm { g t } } \right\| \odot M _ { t } ,\tag{7}
$$

where $M _ { t }$ masks valid pixels. The 2D errors are backprojected into the 3D voxel space to estimate the uncertainty of the local surface using the following formulation:

$$
u _ { \mathrm { e x p } } ( v ) = \frac { 1 } { | \mathcal { C } _ { \mathrm { h i g h } } | } \sum _ { t \in \mathcal { C } _ { \mathrm { h i g h } } } \mathcal { P } ( E _ { t } ^ { \mathrm { d e p t h } } , E _ { t } ^ { \mathrm { r g b } } ; v ) ,\tag{8}
$$

where $\mathcal { P } ( \cdot )$ denotes voxel-wise backprojection with bilinear interpolation.

<!-- image-->  
Figure 2: Our method processes the RGB-D stream through dual explicit and implicit reconstruction branches. The explicit branch projects data into a 3D Gaussian model, while the implicit branch employs an encoder-decoder architecture to regress RGB values and SDF. Subsequently, the discrepancy between the rendered RGB-D and the GT RGB-D is computed. Another mlp predicts global uncertainty, while temporal variations on the SDF surface are characterized to derive uncertainty for the hybrid explicit-implicit representation. This representation then drives NBV selection and path planning. Finally, keyframes are selected within a sliding window for joint optimization of the explicit and implicit maps.

Temporal Variation Uncertainty. To detect emerging surfaces and inconsistencies, we evaluate SDF changes between consecutive keyframes: $\Delta S _ { t } = S _ { t } - S _ { t - 1 }$ . According to the varying states of surfaces, define masks for new surfaces, geometry changes, and novel free space:

$$
\left\{ \begin{array} { l l } { \mathrm { ~ } M _ { \mathrm { n e w } } = \mathbb { I } ( 0 \leq S _ { t } \leq \tau _ { s } ) \odot \mathbb { I } ( \Delta S _ { t } > \tau _ { n } ) , } \\ { \mathrm { ~ } M _ { \mathrm { c h a n g e } } = \mathbb { I } ( | \Delta S _ { t } | > \tau _ { c } ) , } \\ { \mathrm { ~ } M _ { \mathrm { f r e e } } = \mathbb { I } ( S _ { t } > \tau _ { f } ) \odot \mathbb { I } ( S _ { t - 1 } < - \tau _ { f } ) . } \end{array} \right.\tag{9}
$$

The temporal uncertainty term is:

$$
u _ { \mathrm { t i m e } } ( v ) = \beta _ { 1 } | \Delta S _ { t } ( v ) | + \beta _ { 2 } \cdot \mathbb { I } ( v \in M _ { \mathrm { f o c u s } } ) ,\tag{10}
$$

where $M _ { \mathrm { f o c u s } } = M _ { \mathrm { n e w } } \cup M _ { \mathrm { c h a n g e } } \cup M _ { \mathrm { f r e e } } .$

Then, the final hierarchical uncertainty is fused as:

$$
u _ { \mathrm { f i n a l } } ( v ) = \alpha _ { 1 } u _ { \mathrm { i m p } } ( v ) + \alpha _ { 2 } u _ { \mathrm { e x p } } ( v ) + \alpha _ { 3 } u _ { \mathrm { t i m e } } ( v ) ,\tag{11}
$$

where $\alpha _ { i }$ are weights estimated via evidence maximization, interpreted as a fusion of global priors, local observations, and temporal consistency. This hierarchical map directly links to the NBV reward in Eq. 4, providing a multi-scale uncertainty signal that balances exploration coverage and model fidelity.

## Next-Best-View Searching

With the hybrid scene state $\mathcal { M } _ { k } = \{ \mathcal { F } _ { \theta } , \mathcal { G } _ { k } \}$ and hierarchical uncertainty map $u _ { \mathrm { f i n a l } } ( v )$ defined in Eq. 11, the goal of active planning is to select the next viewpoint $\mathbf { c } _ { i }$ that maximizes the expected hybrid information gain.

EHIG Objective. Based on the final hierarchical uncertainty, we cast NBV selection as:

$$
\mathbf { c } _ { i } ^ { * } = \arg \operatorname* { m a x } _ { \mathbf { c } \in \mathcal { C } } \mathbb { E } \left[ \Delta \mathcal { T } _ { \mathrm { h y b r i d } } ( \mathbf { c } ) \right] ,\tag{12}
$$

where C is the candidate viewpoint set, and $\Delta \mathcal { T } _ { \mathrm { h y b r i d } }$ measures the reduction of hybrid entropy:

$$
\Delta \mathcal { T } _ { \mathrm { h y b r i d } } ( \mathbf { c } ) = \Delta H [ \mathcal { F } _ { \boldsymbol { \theta } } ] + \Delta H [ \mathcal { G } _ { k } ] ,\tag{13}
$$

corresponding to global implicit and local explicit uncertainty reduction, respectively.

Voxel-wise Information Weighting. For a voxel v visible from candidate c, we define its contribution as:

$$
w ( v | \mathbf { c } ) = \alpha U ( v ) + \beta H _ { \mathrm { h y b r i d } } ( v ) ,\tag{14}
$$

where $U ( v )$ is the hierarchical uncertainty estimate from Eq. 11 and $H _ { \mathrm { h y b r i d } } ( v )$ is the hybrid entropy in Eq. 3. Î± and Î² are weights. This formulation unifies multi-scale uncertainty into a single information-theoretic weight.

NBV Reward. Given the information weight of the voxel, the expected reward of candidate c is obtained via Eq. 4.

Risk-Aware Path Planning. After obtaining the next goal, we employs an enhanced RRT\* algorithm (LaValle and Kuffner 2001) for active path planning. To generate physically feasible trajectories, we integrate the NBV reward into a risk-aware cost function:

$$
{ \bf p } ^ { * } = \arg \operatorname* { m i n } _ { { \bf p } } \int _ { \bf p } \Big ( C _ { \mathrm { t r a v e l } } ( x ) - \eta R ( { \bf c } _ { x } ) + \lambda C _ { \mathrm { r i s k } } ( x ) \Big ) d x ,\tag{15}
$$

where p is the planned path, $C _ { \mathrm { t r a v e l } }$ the navigation cost, $C _ { \mathrm { { r i s k } } }$ the collision probability, and $R ( \mathbf { c } _ { x } )$ the NBV reward at pose x.

This proposed NBV searching bridges hybrid scene representation, multi-scale uncertainty, and active trajectory optimization into a single expected information gain framework. By combining global implicit entropy reduction and local explicit observation gain, the planner achieves coverageaware and detail-preserving exploration.

## Uncertainty-driven Keyframe Selection

Unlike conventional keyframe strategies that are tightly coupled with temporal ordering, we propose a Uncertaintydriven selection criterion that anchors high-information observations in the hybrid scene state $\mathcal { M } _ { k } .$ Rather than merely ensuring temporal coverage, the proposed keyframes act as a sparse attention mechanism, focusing optimization on regions where the hybrid uncertainty is maximized.

Viewpoint-Based Keyframe Selection. By decoupling keyframe selection from temporal sampling and binding it to viewpoint-space information gain, our method avoids redundant observations and focuses optimization capacity on spatially complementary views, which is crucial for active reconstruction. For a newly acquired RGB-D frame $\mathbf { S } _ { c }$ with camera pose $\mathbf { T } _ { c , w } ,$ we compute its viewpoint divergence relative to the active keyframe set $\mathcal { S } _ { \mathrm { K F } }$

$$
\delta _ { c } = \operatorname* { m i n } _ { \mathbf { S } _ { j } \in S _ { \mathrm { K F } } } d _ { \mathrm { v i e w } } \big ( \mathbf { T } _ { c , w } , \mathbf { T } _ { j , w } \big ) ,\tag{16}
$$

where $d _ { \mathrm { v i e w } }$ measures the viewpoint baseline in SE(3) space, combining angular separation and projected frustum overlap.

Aggressive active motion planning may cause an agent to overskip salient textural structures, we introduce a dual-uncertainty intersection criterion. Define the $h i g h -$ uncertainty intersection set as:

$$
\mathcal { V } _ { \mathrm { h i g h } } = \left\{ v \in \mathcal { V } _ { u } \ \middle \vert \ u _ { \mathrm { e x p } } ( v ) > \tau _ { h } \wedge u _ { \mathrm { i m p } } ( v ) > \tau _ { h } \right\} .\tag{17}
$$

For frame $\mathbf { S } _ { c } ,$ we compute its uncertainty coverage ratio $\rho _ { c }$ as the fraction of $\mathcal { V } _ { \mathrm { h i g h } }$ visible in its frustum:

$$
\rho _ { c } = \vert \mathcal { V } _ { \mathrm { h i g h } } \cap \mathcal { V } _ { \mathrm { v i s } } ( \mathbf { S } _ { c } ) \vert / \vert \mathcal { V } _ { \mathrm { v i s } } ( \mathbf { S } _ { c } ) \vert ,
$$

with $\mathcal { V } _ { \mathrm { v i s } }$ being the visible voxel set. A frame is promoted to keyframe if:

$$
( \delta _ { c } > \tau _ { \mathrm { v i e w } } ) \wedge ( \Delta \mathcal { T } _ { \mathrm { h y b r i d } } ( \mathbf { S } _ { c } ) > \tau _ { \mathrm { i n f o } } ) \wedge ( \rho _ { c } > \tau _ { \rho } ) ,\tag{18}
$$

where $\tau _ { \mathrm { v i e w } } ,$ , Ïinfo, $\tau _ { \rho }$ are viewpoint, information-gain and coverage threshold, respectively. The uncertainty-driven keyframe selection scheme actually establishes a sparse attention mechanism toward scene structures. This ensures selection of frames observing regions where both geometric and neural uncertainties are high.

Viewpoint-Space Sliding Window. Employing all keyframes for joint optimization still incurs excessive computational burden, prior approaches maintained a sliding window over continuous time. However, this strategy exhibits significant viewpoint redundancy as agent approaches the target, while failing to establish sufficient covisibility constraints upon revisiting similar locations. We maintain a local optimization window $\mathcal { W } _ { k } = \{ \mathbf { S } _ { c _ { 1 } } , \dots , \mathbf { S } _ { c _ { m } } \}$ indexed by spatially selected keyframes, not constrained by temporal adjacency. The hybrid state $\mathcal { M } _ { k }$ is jointly refined via:

$$
E _ { \mathrm { t o t a l } } = E _ { \mathrm { p h o t o } } + E _ { \mathrm { g e o } } + \lambda E _ { \mathrm { r e g } } ,\tag{19}
$$

where $E _ { \mathrm { p h o t o } }$ enforces multi-view photometric consistency on $\mathcal { G } _ { k } , E _ { \mathrm { g e o } } ^ { \mathrm { i } }$ aligns Gaussian primitives with the implicit SDF ${ \mathcal { F } } _ { \theta } .$ , and $\mathrm { \Delta } \dot { E } _ { \mathrm { r e g } }$ prevents overfitting across non-overlapping viewpoints.

## 4 Experiments

## Implementation and Simulator

We implement the proposed method within the Habitat simulator (Savva et al. 2019) as an active exploration system. The agent captures posed RGB-D observations along planned viewpoints. The camera field-of-view is set to $6 0 ^ { \bar { \circ } }$ vertically and $\mathsf { \Pi } _ { 9 0 ^ { \circ } }$ horizontally, and the system processes sequences online with on-policy planning and incremental reconstruction. Further implementation details are presented in the supplementary material.

## Datasets, Metrics, and Baselines

Following prior active mapping benchmarks (Yan, Yang, and Zha 2023), we evaluate on two widely used datasets:(i) Replica (Straub et al. 2019) with 8 indoor scenes, and (ii) Matterport3D (MP3D) (Chang et al. 2017) with 5 large-scale scenes exhibiting significant occlusion and spatial complexity. All methods are run for 2000 exploration steps on Replica and MP3D.

We report metrics targeting the critical objectives of active reconstruction: accuracy (Acc, cm), completion (Com, cm), and completion ratio (C.R., %), where Acc/Com are computed with a 5cm threshold. To evaluate rendering quality, we report PSNR, SSIM, and LPIPS on held-out viewpoints. For additional geometric consistency analysis, we compute the Mean Absolute Distance (MAD) between the reconstructed SDF and ground-truth surfaces.

We compare our method against state-of-the-art active reconstruction frameworks: ActiveNR (Yan, Yang, and Zha 2023), ANM-S (Kuang et al. 2024), NARUTO (Feng et al. 2024), and ActiveSplat (Li et al. 2025). We further compare passive baselines in the supplemental material. All baselines are re-trained and evaluated locally for fair comparison.

## Evaluation on Replica

Table 1 reports 3D reconstruction and view synthesis metrics on the Replica dataset. Our method consistently achieves the best or second-best performance across all metrics. For reconstruction, it yields the highest completion ratio (C.R.) and lowest Acc/Com error, reaching 98.09% C.R. on R1 and 98.18% on R2. For view synthesis, it achieves the highest PSNR (up to 40.51) and SSIM (0.980) while maintaining the lowest LPIPS, demonstrating sharp textures and photometric consistency.

HxpK

<!-- image-->

<!-- image-->  
pLe4

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
gZ6f

<!-- image-->  
ANM-S

<!-- image-->  
NARUTO

<!-- image-->  
ActiveSplat

<!-- image-->  
OURS

<!-- image-->  
GT

Figure 3: Qualitative comparison of 3D reconstruction results on representative MP3D sequences. Additional results and detailed comparisons for all Replica and MP3D sequences are provided in the supplementary material.
<table><tr><td>Method</td><td>Metric</td><td>Off0</td><td>Off1</td><td>Off2</td><td>Off3</td><td>Off4</td><td>R0</td><td>R1</td><td>R2</td></tr><tr><td rowspan="3">ANM-S</td><td>Acc (cm) â</td><td>1.44</td><td>1.03</td><td>1.60</td><td>1.80</td><td>1.50</td><td>1.47</td><td>1.29</td><td>1.28</td></tr><tr><td>Com. (cm) â</td><td>1.98</td><td>1.55</td><td>6.65</td><td>1.13</td><td>1.08</td><td>0.91</td><td>1.02</td><td>0.85</td></tr><tr><td>C.R. (%) â</td><td>95.43</td><td>92.66</td><td>79.20</td><td>94.98</td><td>95.35</td><td>96.71</td><td>95.66</td><td>96.79</td></tr><tr><td rowspan="6">NARUTO</td><td>Acc (cm) â</td><td>1.26</td><td>1.04</td><td>Ã</td><td>34.84</td><td>1.67</td><td>1.75</td><td>Ã</td><td>1.50</td></tr><tr><td>Com. (cm) â</td><td>1.41</td><td>1.30</td><td>Ã</td><td>2.96</td><td>2.01</td><td>1.56</td><td>Ã</td><td>1.49</td></tr><tr><td>C.R. (%) â</td><td>97.63</td><td>96.88</td><td>Ã</td><td>91.27</td><td>95.14</td><td>94.58</td><td>Ã</td><td>97.56</td></tr><tr><td>SNR</td><td>31.01</td><td>31.43</td><td>Ã</td><td>26.63</td><td>28.57</td><td>26.55</td><td>Ã</td><td>25.56</td></tr><tr><td>SSâ</td><td>0.892</td><td>0.897</td><td>Ã</td><td>0.831</td><td>0.882</td><td>0.782</td><td>Ã</td><td>0.818</td></tr><tr><td>LPIS </td><td>0.299</td><td>0.283</td><td>Ã</td><td>0.283</td><td>0.284</td><td>0.354</td><td>Ã</td><td>0.367</td></tr><tr><td rowspan="6">ActiveSplat</td><td>Acc (cm) â</td><td>1.16</td><td>1.11</td><td>1.47</td><td>1.70</td><td>1.50</td><td>1.67</td><td>1.43</td><td>1.36</td></tr><tr><td>Com. (cm) â</td><td>0.63</td><td>0.94</td><td>5.59</td><td>1.83</td><td>1.06</td><td>0.84</td><td>0.74</td><td>1.04</td></tr><tr><td>C.R. (%)</td><td>97.54</td><td>94.54</td><td>80.69</td><td>91.49</td><td>95.34</td><td>97.04</td><td>96.84</td><td>95.65</td></tr><tr><td>SNR</td><td>24.487</td><td>26.955</td><td>22.728</td><td>20.965</td><td>27.887</td><td>26.163</td><td>29.005</td><td>28.865</td></tr><tr><td>SIM</td><td>0.857</td><td>0.871</td><td>0.888</td><td>0.804</td><td>0.878</td><td>0.823</td><td>0.877</td><td>0.894</td></tr><tr><td>LPIPS â</td><td>0.145</td><td>0.130</td><td>0.113</td><td>0.232</td><td>0.147</td><td>0.199</td><td>0.136</td><td>0.113</td></tr><tr><td rowspan="6">OURS</td><td>Acc (cm) â</td><td>1.12</td><td>1.02</td><td>1.34</td><td>1.56</td><td>1.38</td><td>1.59</td><td>1.13</td><td>1.26</td></tr><tr><td>Com. (cm) </td><td>1.34</td><td>1.17</td><td>1.66</td><td>1.97</td><td>1.87</td><td>1.75</td><td>1.32</td><td>1.52</td></tr><tr><td>C.R. (%) â</td><td>97.76</td><td>98.21</td><td>96.86</td><td>94.70</td><td>96.80</td><td>97.28</td><td>98.09</td><td>98.18</td></tr><tr><td>PSN</td><td>40.51</td><td>40.54</td><td>33.72</td><td>34.14</td><td>37.37</td><td>33.80</td><td>34.63</td><td>36.00</td></tr><tr><td> SIM</td><td>0.980</td><td>0.979</td><td>0.951</td><td>0.949</td><td>0.964</td><td>0.948</td><td>0.954</td><td>00.962</td></tr><tr><td>LPPIPS </td><td>0.030</td><td>0.034</td><td>0.067</td><td>0.075</td><td>0.054</td><td>0.072</td><td>0.056</td><td>0.053</td></tr></table>

Table 1: Quantitative comparison of 3D reconstruction and view synthesis quality between the proposed method and state-of-the-art approaches on the Replica dataset. The symbol Ã indicates that the method fails to complete exploration within five trials.

## Evaluation on MP3D

<table><tr><td>Method</td><td>Metric</td><td>Gdvg</td><td>gZ6f</td><td>HxpK</td><td>pLe4</td><td>YmJk</td><td>Avg.</td></tr><tr><td rowspan="3">ActiveINR</td><td>Acc (cm) â</td><td>5.09</td><td>4.15</td><td>15.60</td><td>5.56</td><td>8.61</td><td>7.80</td></tr><tr><td>Com. (cm) â</td><td>5.69</td><td>7.43</td><td>15.96</td><td>8.03</td><td>8.46</td><td>9.11</td></tr><tr><td>C.R. (%) â</td><td>80.99</td><td>80.68</td><td>48.34</td><td>76.41</td><td>79.35</td><td>73.15</td></tr><tr><td rowspan="3">ANM-S</td><td>Acc (cm) â</td><td>5.52</td><td>1.62</td><td>2.13</td><td>4.54</td><td>4.50</td><td>3.66</td></tr><tr><td>Com. (cm) â</td><td>3.95</td><td>2.01</td><td>12.49</td><td>2.51</td><td>3.53</td><td>4.90</td></tr><tr><td>C.R.(%) â</td><td>91.00</td><td>94.58</td><td>60.39</td><td>95.02</td><td>88.65</td><td>85.93</td></tr><tr><td rowspan="6">NARUTO</td><td>Acc (cm) â</td><td>2.34</td><td>3.57</td><td>7.29</td><td>4.46</td><td>9.52</td><td>5.44</td></tr><tr><td>Com. (cm) â</td><td>4.93</td><td>2.47</td><td>2.84</td><td>3.14</td><td>5.68</td><td>3.81</td></tr><tr><td>C.R. (%) â</td><td>84.88</td><td>93.26</td><td>92.15</td><td>82.67</td><td>78.99</td><td>86.39</td></tr><tr><td>PSNR â</td><td>23.42</td><td>23.84</td><td>23.32</td><td>27.15</td><td>23.64</td><td>24.27</td></tr><tr><td>SSIM â</td><td>0.742</td><td>0.719</td><td>0.734</td><td>0.767</td><td>0.735</td><td>0.739</td></tr><tr><td>LPIPS â</td><td>0.416</td><td>0.523</td><td>0.492</td><td>0.554</td><td>0.517</td><td>0.500</td></tr><tr><td rowspan="6">ActiveSplat</td><td>Acc (cm) â</td><td>2.39</td><td>1.74</td><td>2.53</td><td>4.09</td><td>9.52</td><td>4.05</td></tr><tr><td>Com. (cm) â</td><td>3.76</td><td>1.34</td><td>24.28</td><td>1.07</td><td>2.84</td><td>6.66</td></tr><tr><td>C.R.(%) â</td><td>92.11</td><td>97.61</td><td>44.45</td><td>99.10</td><td>90.78</td><td>84.81</td></tr><tr><td>PSNR â</td><td>22.77</td><td>16.40</td><td>18.33</td><td>23.49</td><td>24.57</td><td>21.12</td></tr><tr><td>SSIMâ</td><td>0.700</td><td>0.601</td><td>0.776</td><td>0.667</td><td>0.852</td><td>0.719</td></tr><tr><td>LPIPS â</td><td>0.264</td><td>0.342</td><td>0.236</td><td>0.345</td><td>0.156</td><td>0.269</td></tr><tr><td rowspan="6">OURS</td><td>Acc (cm) â</td><td>1.68</td><td>1.90</td><td>1.61</td><td>2.68</td><td>2.66</td><td>2.11</td></tr><tr><td>Com. (cm) â</td><td>1.59</td><td>1.96</td><td>2.09</td><td>2.38</td><td>2.81</td><td>2.27</td></tr><tr><td>C.R.(%)</td><td>98.23</td><td>97.94</td><td>98.12</td><td>94.55</td><td>91.73</td><td>96.11</td></tr><tr><td>PSNR â</td><td>31.12</td><td>32.43</td><td>29.53</td><td>33.14</td><td>30.93</td><td>31.43</td></tr><tr><td>SSIMâ</td><td>0.912</td><td>0.939</td><td>0.905</td><td>0.920</td><td>0.923</td><td>0.920</td></tr><tr><td>LPIPS â</td><td>0.160</td><td>0.168</td><td>0.176</td><td>0.222</td><td>0.179</td><td>0.181</td></tr></table>

Table 2 evaluates our method on the MP3D dataset. Compared to ActiveSplat, our approach significantly improves both geometry and rendering fidelity. We achieve the highest combined reconstruction score in nearly all scenes, exceeding 98% on three out of five sequences. For photometric metrics, our method delivers the best PSNR and SSIM in four out of five cases, while maintaining the lowest LPIPS, reflecting perceptually consistent rendering.

Table 2: Quantitative comparison on the MP3D dataset for 3D reconstruction and novel view synthesis.

Figure 3 visualizes reconstructions on MP3D. Compared to NARUTO and ActiveSplat, our method produces sharper edges, fewer ghosting artifacts, and consistent textures under dynamic occlusion.

<!-- image-->  
(a) Cavity Region

<!-- image-->  
(b) MLP Uncertainty

<!-- image-->  
(c) Depth Uncertainty

<!-- image-->

<!-- image-->  
(f) Clean Mesh

(g) SDF Surface  
<!-- image-->

<!-- image-->

<!-- image-->  
(h) Combined Uncertainties

(d) Photometric Uncertainty (e) SDF Variation Uncertainty  
<!-- image-->  
(i) Uncertainty with SDF

<!-- image-->  
(g) Total Uncertainty Heatmap

<!-- image-->  
(k) Occlusion  
Figure 4: Visualization of uncertainties and their spatial relationship to real scene. Our proposed hybrid strategy not only endows the agent with global optimization capabilities, but also enables it to perceive intricate structures and textures while handling occlusions.

## Ablation Study

As summarized in Table 3, ablation studies are conducted on the challenging MP3D YmJk sceneâcharacterized by significant occlusion and complex geometry.

Uncertainty Setting. Removing multi-resolution tri-plane encoding causes system failure due to complete loss of spatial perception. Eliminating the MLP-predicted uncertainty volume severely degrades reconstruction completeness (Com: 4.37 cm vs. 2.81 cm) by impeding global scene understanding. Exclusion of depth uncertainty induces erratic reconstruction (Acc: 4.75 cm vs. 2.66 cm) due to compromised surface fidelity estimation, which destabilizes optimization. Omission of RGB uncertainty substantially deteriorates rendering metrics (PSNR: 28.35 dB vs. 30.93 dB), attributable to degraded color/texture perception. Disabling the time-varying SDF representation markedly decreases reconstruction completeness.

Searching and Planning. Replacing the risk-aware path planner with naive uncertainty-volume aggregation degrades reconstruction coverage (C.R.: 89.23% vs. 91.73%), as this suboptimal strategy prompts excessive surface proximity, reducing global observability while increasing collision risk. Finally, disabling keyframe management guided by spatial co-visibility and uncertainty underutilizes historical observations upon revisit, leading to rendering degradation.

Advantages of Hierarchical Uncertainties. Fig. 4 visualizes the Hierarchical Uncertainty Map. The fully implicit uncertainty (b, e) provides the agent with global optimization capability. However, as the MLP-predicted SDF tends to generate redundant structures (f, g), it induces excessively high uncertainty in void regions (a) and redundant structure areas (g). This results in the agent allocating excessive attention to non-existent uncertainties (h). Conversely, the fully explicit uncertainty (c, d) aids the agent in identifying complex structures and textures. Nevertheless, due to its inability to perceive occluded regions (k) via Î±-blending, it leads the agent to prematurely conclude optimization completeness and initiate subsequent planning. Our hybrid approach synergistically combines the strengths of both explicit and implicit representations. By adaptively weighting the explicit and implicit uncertainties, it enhances the agentâs perceptual awareness across all local and global regions (g).

<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>MADâ</td><td>Accâ</td><td>Comâ</td><td>C.R.â</td></tr><tr><td>final</td><td>30.93</td><td>0.923</td><td>0.179</td><td>1.53</td><td>2.66</td><td>2.81</td><td>91.73</td></tr><tr><td>w.o. Tri-plane Encoder</td><td>Ã</td><td>Ã</td><td>Ã</td><td>Ã</td><td>Ã</td><td>Ã</td><td>Ã</td></tr><tr><td>W.o. MLP Uncert</td><td>30.89</td><td>0.917</td><td>0.191</td><td>1.80</td><td>2.65</td><td>4.37</td><td>84.84</td></tr><tr><td>w.o. Depth Uncert</td><td>29.28</td><td>0.907</td><td>0.218</td><td>1.88</td><td>4.75</td><td>5.12</td><td>83.97</td></tr><tr><td>W.O. RGB Uncert</td><td>28.35</td><td>0.901</td><td>0.201</td><td>1.78</td><td>2.69</td><td>4.02</td><td>86.18</td></tr><tr><td>w.o. SDF Temp</td><td>31.23</td><td>0.921</td><td>0.187</td><td>1.58</td><td>2.71</td><td>3.11</td><td>90.83</td></tr><tr><td>w.o. Risk Planning</td><td>30.78</td><td>0.916</td><td>0.179</td><td>1.61</td><td>2.67</td><td>3.40</td><td>89.23</td></tr><tr><td>w.o. Uncert Keyframe</td><td>29.43</td><td>0.917</td><td>0.182</td><td>1.54</td><td>2.77</td><td>2.88</td><td>88.91</td></tr><tr><td>w. Temporal Sliding Window</td><td>28.69</td><td>0.910</td><td>0.186</td><td>1.62</td><td>2.69</td><td>3.72</td><td>87.02</td></tr></table>

Table 3: Ablation study on MP3D dataset. The best results are highlighted in the table.

## 5 Conclusion

We have introduced Active3D, an active 3D reconstruction framework that unifies implicit neural fields and explicit Gaussian primitives into a hybrid information-theoretic formulation. By deriving a hierarchical uncertainty volume from this hybrid scene state, our method simultaneously captures global structural priors and local observation confidence, enabling principled next-best-view selection. An uncertainty-driven keyframe selection strategy anchors high-entropy viewpoints as sparse attention nodes, while a viewpoint-space sliding window performs uncertaintyaware local refinement to maintain globalâlocal consistency. Formulating NBV planning as an Expected Hybrid Information Gain problem with a risk-aware path planner further ensures efficient and safe exploration.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 5: Novel view synthesis results on the MP3D dataset. The tested viewpoints were not present in any training trajectories of the evaluated methods. PSNR values are indicated in the top-left corner. Challenging regions are highlighted with red boxes.

<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>MADâ</td><td>Accâ</td><td>Comâ</td><td>C.R.â</td></tr><tr><td>Hybrid</td><td>32.43</td><td>0.939</td><td>0.168</td><td>1.22</td><td>1.90</td><td>1.96</td><td>97.94</td></tr><tr><td>Impliit-only</td><td>30.64</td><td>0.922</td><td>0.187</td><td>1.25</td><td>1.91</td><td>2.34</td><td>96.24</td></tr><tr><td>Explicit-only</td><td>32.29</td><td>0..932</td><td>0.171</td><td>1.28</td><td>1.92</td><td>3.85</td><td>94.87</td></tr></table>

Table 4: Numerical Results of Hybrid Representation on MP3D dataset. The best results are highlighted in the table.

## Acknowledgments

This research was supported by the Tier 2 Grant (MOE-T2EP20124-0015) from the Singapore Ministry of Education.

## A Supplementary Details

This section elaborates on algorithmic details and experimental results omitted from the main text. We begin by introducing loss functions for both explicit and implicit reconstruction. Subsequently, we present computational efficiency metrics for each submodule and analyze the convergence point of reconstruction completeness during iterative refinement. Finally, extensive comparative evaluations against SOTA methods are provided, assessing reconstruction performance and rendering quality.

## Hybrid-map Optimization

Within BA optimization, we compute gradients of the loss function with respect to both Gaussian parameters $( G _ { i } \ =$ $( \mu _ { i } , \Sigma _ { i } , \alpha _ { i } , c _ { i } ) )$ and the implicit weights of the MLP. Explicit Loss. In the explicit branch, each Gaussian primitiveâs parameters are optimized by minimizing photometric $( \mathcal { L } _ { \mathrm { p h o } } )$ and geometric $( \bar { \mathcal { L } } _ { \mathrm { g e o } } )$ residuals between rendered and observed data:

$$
\begin{array} { l } { { \mathcal { L } _ { \mathrm { p h o } } = \left\| I ( \mathbf { \mathcal { M } } , \mathbf { T } _ { c , w } , \mathbf { K } ) - \bar { I } \right\| _ { 2 } } } \\ { { \mathcal { L } _ { \mathrm { g e o } } = \left\| D ( \mathbf { \mathcal { M } } , \mathbf { T } _ { c , w } , \mathbf { K } ) - \bar { D } \right\| _ { 2 } } } \end{array}\tag{20}
$$

<!-- image-->  
Figure 6: Convergence curves of completeness vs. iterations on the MP3D dataset. Our method achieves a faster convergence rate and higher final reconstruction completeness compared to other SOTA approaches.

where Â¯I and DÂ¯ represent the observed RGB image and depth map, while $I ( \cdot )$ and $D ( \cdot )$ denote the rendered images synthesized from the static Gaussian map M, camera pose $\mathbf { T } _ { c , w } \in \mathrm { S E } ( 3 )$ , and intrinsic matrix $\mathbf { K } \in \mathbb { R } ^ { 3 \times 3 }$

In the implicit branch, we employ four core loss functions to jointly optimize geometry, appearance, and uncertainty: RGB Loss. For ray i with rendered color $\hat { C } _ { i }$ and ground truth $\bar { C } _ { i } ,$ we compute:

$$
\mathcal { L } _ { \mathrm { r g b } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \omega _ { i } \left\| \hat { C } _ { i } - \bar { C } _ { i } \right\| _ { 2 } ^ { 2 } ,\tag{21}
$$

where N is the total ray count, and $\omega _ { i }$ weights rays based on

<table><tr><td></td><td colspan="3">Implicit Branch</td><td colspan="2">Gaussian Splatting</td><td colspan="3">Uncertainty Construction</td><td colspan="4">Planning</td></tr><tr><td></td><td>Ray Samping</td><td>MLP Inference</td><td>MLP Backward</td><td>Rendering</td><td>Mapping</td><td>Depth Uncert</td><td>Photometric Uncert</td><td>SDF Uncert</td><td>Uncert Aggre</td><td>NBV</td><td>Risk Field</td><td>RRT Planning</td></tr><tr><td>Time (ms)</td><td>6.56</td><td>7.38</td><td>14.46</td><td>7.03</td><td>35.42</td><td>6.34</td><td>7.55</td><td>2.55</td><td>16.34</td><td>1.98</td><td>2.12</td><td>3.07</td></tr></table>

Table 5: Computational Time Breakdown for System Components. The data represent the mean values derived from 2000 test frames in the MP3D dataset.

<!-- image-->  
Figure 7: Novel view synthesis results on the Replica dataset. The tested viewpoints were not present in any training trajectories of the evaluated methods. PSNR values are indicated in the top-left corner. Challenging regions are highlighted with red boxes. The office2 and room1 sequences are not exhibited due to NARUTOâs complete failure.

depth validity.

Depth Loss. For valid depth rays $( \bar { D } _ { j } \leq D _ { \mathrm { t r u n c } } ) \colon$

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \frac { 1 } { N _ { v } } \sum _ { j = 1 } ^ { N _ { v } } \left( \hat { D } _ { j } - \bar { D } _ { j } \right) ^ { 2 } ,\tag{22}
$$

where $N _ { v }$ is the count of valid depth measurements, and $\hat { D } _ { j }$ is the rendered depth for ray j.

SDF Loss. Given sampled depths $\mathbf { z } = \{ z _ { i } \} _ { i = 1 } ^ { N _ { s } }$ along a ray with $N _ { s }$ samples, ground truth depth $d ^ { * }$ , and predicted SDF values $\mathbf { s } = \{ \bar { s } _ { i } \} _ { i = 1 } ^ { N _ { s } }$

$$
\mathcal { L } _ { \mathrm { s d f } } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { i \in \mathcal { M } } \left( s _ { i } - \left( d ^ { * } - z _ { i } \right) \right) ^ { 2 } ,\tag{23}
$$

where $\mathcal { M } = \left\{ i : | z _ { i } - d ^ { * } | < \tau \right\}$ defines the truncation region with threshold Ï .

Uncertainty Loss. For valid depth rays:

$$
\mathcal { L } _ { \mathrm { u n c e r t } } = \frac { 1 } { N _ { v } } \sum _ { j = 1 } ^ { N _ { v } } \left( \frac { ( \hat { D } _ { j } - \bar { D } _ { j } ) ^ { 2 } } { 2 \sigma _ { j } ^ { 2 } } + \frac { 1 } { 2 } \log { \sigma _ { j } ^ { 2 } } \right) ,\tag{24}
$$

where $\sigma _ { j } ^ { 2 }$ is the variance from the uncertainty grid for ray $j .$ The Uncertainty Loss incorporates two regularization terms. The first term reflects depth prediction uncertainty, promoting uncertainty amplification when depth estimates deviate significantly from ground truth to enhance agent attentiveness, while attenuating uncertainty under accurate predictions. The second term serves to curb excessive uncertainty expansion.

Total Loss. The unified objective combines all loss components:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \sum _ { k \in \{ \mathrm { p h o , g e o , r g b , d e p t h , s d f , u n c e r t } \} } \lambda _ { k } \mathcal { L } _ { k } ,\tag{25}
$$

where $\lambda _ { k }$ are balancing weights for each loss term.

## Computational Efficiency and Convergence

Owing to our dual-branch framework comprising both implicit and explicit reconstructions, we demonstrate the realtime capability by separately measuring the forward inference and backward optimization latency per RGB-D frame, further analyzing the computational cost of four key uncertainty components, and presenting the efficiency of path planning.

Time Analysis. Table 5 presents the average computation time for key modules. For the implicit branch, the primary computational cost lies in coordinate point sampling, model forward inference, and back-propagation. Our MLP is lightweight due to its shallow depth and minimal number of hidden neurons. Within the Gaussian reconstruction branch, significant time is consumed by Î±-blending and Gaussian map optimization, which is performed only on keyframes within the sliding window to enhance computational efficiency. For the uncertainty voxels, implicit uncertainty voxels are directly obtained via MLP inference, while the construction of other uncertainty voxels completes within milliseconds. Finally, the path planning module aggregates uncertainty voxels, identifies the NBV, constructs a risk field, and performs RRT planning. Note that planning is triggered only when the agent reaches its target position and initiates the next planning cycle; the system primarily operates in motion execution, thereby enhancing overall efficiency.

C.R. Convergence. Fig. 6 illustrates the reconstruction completeness curve versus iteration count. In challenging scenario YmJk, ActiveSplat (Li et al. 2025) fails to plan effective trajectories, significantly hindering its per-

<!-- image-->  
Figure 8: Reconstruction results on all 5 sequences of the MP3D dataset. The first row of each group illustrates local details, while the second row demonstrates global completeness. Our method achieves reconstructions with higher-fidelity local geometric details and superior completeness, while maintaining robustness across diverse scenarios. Scene appearance may exhibit variations due to method-specific simulator lighting configurations.

<!-- image-->  
office0  
office1  
office2  
office3

Figure 9: Reconstruction results on the first 4 sequences of the Replica dataset. While MonoGS and LoopSplat are passive reconstruction approaches, the others represent active reconstruction schemes. Our method reconstructs more complete scene structures and demonstrates robustness across all sequences. Scene appearance may exhibit variations due to method-specific simulator lighting configurations.

<!-- image-->  
office4  
room0  
room1  
room2

Figure 10: Reconstruction results on the last 4 sequences of the Replica dataset. While MonoGS and LoopSplat are passive reconstruction approaches, the others represent active reconstruction schemes. Our method reconstructs more complete scene structures and demonstrates robustness across all sequences. Scene appearance may exhibit variations due to method-specific simulator lighting configurations.

formance. While Active-INR (Yan, Yang, and Zha 2023) and ANM-S (Kuang et al. 2024) exhibit rapid initial exploration, they ultimately converge to low completeness levels. NARUTO (Feng et al. 2024) demonstrates superior convergence and completeness, yet it frequently predicts large-scale redundant structures in invalid regions (refer to reconstruction results in the main text). In contrast, our method robustly plans trajectories and reconstructs highfidelity meshes.

## Rendering Performance

Compared to NeRF, Gaussian Splatting exhibits superior novel view synthesis. Quantitative evaluations in the main text confirm our approach outperforms SOTA methods across all metrics. This subsection provides qualitative comparisons on all MP3D and Replica sequences. Fig. 5 and Fig. 7 show results for MP3D and Replica, respectively. Notably, test viewpoints were excluded from all training trajectories. Our method renders sharp boundaries and clear textures.

## Reconstruction Results

Mesh reconstructions for three MP3D sequences are compared with SOTA in the main text. Fig. 8-10 show full qualitative comparisons across five MP3D and eight Replica sequences. MonoGS (Matsuki et al. 2023) and LoopSplat (Zhu et al. 2024) (passive schemes) exhibit extensive fragmentation due to incomplete scene observation. Neural methods ANM-S (Kuang et al. 2024) and NARUTO (Feng et al. 2024) partially fill holes but generate superfluous structures, causing agent over-focus as shown in the text. ActiveSplat (Li et al. 2025) produces only sparse point clouds. In contrast, our approach ensures superior global integrity and enhanced local geometric accuracy.

## B Hybrid Representation Analysis

To validate the hybrid implicit-explicit formulation, we compare against two variants: (i) implicit-only, using only FÎ¸, and (ii) explicit-only, using only Gk.

## Implicit-only

For the implicit-only evaluation, we utilize the MLPpredicted SDF and uncertainty for active exploration. We simultaneously capture RGB-D images along the planned trajectory, perform Gaussian projection, and optimize the Gaussian branch using the loss defined in Eq. 20. However, the explicit and implicit branches remain entirely independent, yielding no mutual enhancement or interference.

## Explicit-only

For the explicit-only evaluation, we employ both Depth Uncertainty and Photometric Uncertainty to construct an uncertainty voxel map, computing the discrepancy between the ground-truth depth and the observed depth as the SDF map, which is updated in real-time during subsequent tracking. During incremental updates, voxel grids are projected into camera coordinates and validated against depth bounds and image constraints. Valid voxels sample depth values via bilinear interpolation, enforcing local depth continuity to reject outliers. The SDF observations are truncated to a narrow band around surfaces. Each voxel update includes depth adaptation and consistency weight between images, which is maintained by an exponential weighted moving average.

Although uncertainty predicted by the implicit MLP is not utilized, we retain its presence while optimizing the MLP for convergence analysis.

## Complementarity Analysis

Completeness Analysis. Fusing implicit and explicit branches yields a statistically significant acceleration in reconstruction convergence speed and elevates final reconstruction completeness. This synergistic integration facilitates mutual enhancement between the two map representation paradigms. Table 4 shows the quantitative comparison of different representation modes.

## Risking Filed and Planning

Conventional path planners rely on explicit representations (e.g., occupancy grids, octrees), yet suffer from degraded navigation accuracy in complex environments due to erroneous occupancy estimation. SDF fields conversely exhibit initial exploration instability, frequently guiding paths into obstacles and causing collisions. To address these limitations, we propose a hybrid implicit-explicit planner. Our framework constructs an occupancy field via Gaussian processes, integrates it with an SDF to generate a risk field, enabling efficient global exploration and planning.

Fig. 12 visualizes cross-sections of the occupancy field, SDF field, and risk field on MP3D sequence gZ6f. Initial exploration shows the Gaussian-based occupancy map is incomplete, misleading planners into boundary violations. The SDF map generates redundant structures. Our risk field synergistically preserves distinct object boundaries while eliminating redundant structures.

Uncertainty Convergence Analysis. For the three schemes (Hybrid, Implicit-only, Explicit-only), we statistically analyzed the temporal evolution of the MLP-predicted uncertainty throughout the tracking and reconstruction pipeline. Fig. 11 juxtaposes these trends against the convergence/- divergence of reconstruction accuracy (Comp) and completeness rate (C.R.). The uncertainty convergence curves reveal that the Implicit-only scheme culminates in persistently elevated uncertainty values. This stems from its tendency to over-prioritize redundant structures (as analyzed in the main text), thereby diminishing exploratory coverage in structurally rich regions and degrading reconstruction quality and completeness. Furthermore, gradient optimization of the uncertainty MLP exhibits direct correlation with depth prediction fidelity. Suboptimal reconstruction accuracy consequently amplifies uncertainty, establishing a detrimental bidirectional feedback loop.

Reconstruction Accuracy Analysis. The Explicit-only scheme demonstrates consistently inferior accuracy across all iterations. This limitation arises from its fundamental inability to address occlusions, causing premature abandonment of under-optimized regions. In contrast, the integration of implicit uncertainty imbues the agent with heightened attentional focus on occluded areas, thereby enhancing reconstruction precision.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 11: Convergence-divergence analysis of hybrid representation on MP3D dataset. The left figure depicts the evolution of predictive uncertainty from the MLP during the iterative process. The center figure plots the reconstruction accuracy against iteration count. The right figure illustrates the progression of reconstruction completeness throughout the iterations.

<!-- image-->  
Figure 12: Visualization of occupancy field, sdf field, and risk field for MP3D gZ6f. The risk field completes the gaps within the occupancy field and eliminates the inherent redundant structures in the SDF field.

## References

Aloimonos, J.; Weiss, I.; and Bandyopadhyay, A. 1988. Active vision. International journal of computer vision, 1: 333â 356.

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2022. Mip-nerf 360: Unbounded antialiased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5470â5479.

Chang, A.; Dai, A.; Funkhouser, T.; Halber, M.; Niessner, M.; Savva, M.; Song, S.; Zeng, A.; and Zhang, Y. 2017. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158.

Chen, A.; Xu, Z.; Geiger, A.; Yu, J.; and Su, H. 2022. Ten-

sorf: Tensorial radiance fields. In European conference on computer vision, 333â350. Springer.

Chen, L.; Zhan, H.; Chen, K.; Xu, X.; Yan, Q.; Cai, C.; and Xu, Y. 2025. ActiveGAMER: Active GAussian Mapping through Efficient Rendering. arXiv preprint arXiv:2501.06897.

Chen, S.; Li, Y.; and Kwok, N. M. 2011. Active vision in robotic systems: A survey of recent developments. The International Journal of Robotics Research, 30(11): 1343â 1377.

CÂ¸ icÂ¸ek, O.; Abdulkadir, A.; Lienkamp, S. S.; Brox, T.; and Â¨ Ronneberger, O. 2016. 3D U-Net: learning dense volumetric segmentation from sparse annotation. In Medical Image Computing and Computer-Assisted InterventionâMICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19, 424â432. Springer.

Dai, A.; NieÃner, M.; Zollhofer, M.; Izadi, S.; and Theobalt, Â¨ C. 2017. Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration. ACM Transactions on Graphics (ToG), 36(4): 1.

Elfes, A. 2013. Occupancy grids: A stochastic spatial representation for active robot perception. arXiv preprint arXiv:1304.1098.

Feng, Z.; Zhan, H.; Chen, Z.; Yan, Q.; Xu, X.; Cai, C.; Li, B.; Zhu, Q.; and Xu, Y. 2024. Naruto: Neural active reconstruction from uncertain target observations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21572â21583.

Fridovich-Keil, S.; Yu, A.; Tancik, M.; Chen, Q.; Recht, B.; and Kanazawa, A. 2022. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5501â 5510.

Huang, R.; Zou, D.; Vaughan, R.; and Tan, P. 2018. Active image-based modeling with a toy drone. In 2018 IEEE International Conference on Robotics and Automation (ICRA), 6124â6131. IEEE.

Isler, S.; Sabzevari, R.; Delmerico, J.; and Scaramuzza, D. 2016. An information gain formulation for active volumetric 3D reconstruction. In 2016 IEEE International Conference on Robotics and Automation (ICRA), 3477â3484. IEEE.

Izadi, S.; Kim, D.; Hilliges, O.; Molyneaux, D.; Newcombe, R.; Kohli, P.; Shotton, J.; Hodges, S.; Freeman, D.; Davison, A.; et al. 2011. Kinectfusion: real-time 3d reconstruction and interaction using a moving depth camera. In Proceedings of the 24th annual ACM symposium on User interface software and technology, 559â568.

Jiang, W.; Lei, B.; and Daniilidis, K. 2024. Fisherrf: Active view selection and mapping with radiance fields using fisher information. In European Conference on Computer Vision, 422â440. Springer.

Jin, L.; Zhong, X.; Pan, Y.; Behley, J.; Stachniss, C.; and Popovic, M. 2025. Activegs: Active scene reconstruction Â´ using gaussian splatting. IEEE Robotics and Automation Letters.

Jin, R.; Gao, Y.; Wang, Y.; Wu, Y.; Lu, H.; Xu, C.; and Gao, F. 2024. Gs-planner: A gaussian-splatting-based planning framework for active high-fidelity reconstruction. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 11202â11209. IEEE.

Kazhdan, M.; Bolitho, M.; and Hoppe, H. 2006. Poisson surface reconstruction. In Proceedings of the fourth Eurographics symposium on Geometry processing, volume 7.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph., 42(4): 139â1.

Kirsch, A.; Van Amersfoort, J.; and Gal, Y. 2019. Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning. Advances in neural information processing systems, 32.

Kuang, Z.; Yan, Z.; Zhao, H.; Zhou, G.; and Zha, H. 2024. Active neural mapping at scale. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 7152â7159. IEEE.

LaValle, S. M.; and Kuffner, J. J. 2001. Rapidly-exploring random trees: Progress and prospects: Steven m. lavalle, iowa state university, a james j. kuffner, jr., university of tokyo, tokyo, japan. Algorithmic and computational robotics, 303â307.

Lee, S.; Chen, L.; Wang, J.; Liniger, A.; Kumar, S.; and Yu, F. 2022. Uncertainty guided policy for active robotic 3d reconstruction using neural radiance fields. IEEE Robotics and Automation Letters, 7(4): 12070â12077.

Li, Y.; Brasch, N.; Wang, Y.; Navab, N.; and Tombari, F. 2020. Structure-slam: Low-drift monocular slam in indoor environments. IEEE Robotics and Automation Letters, 5(4): 6583â6590.

Li, Y.; Kuang, Z.; Li, T.; Hao, Q.; Yan, Z.; Zhou, G.; and Zhang, S. 2025. ActiveSplat: High-Fidelity Scene Reconstruction Through Active Gaussian Splatting. IEEE Robotics and Automation Letters, 10(8): 8099â8106.

Li, Y.; Lyu, C.; Di, Y.; Zhai, G.; Lee, G. H.; and Tombari, F. 2024. Geogaussian: Geometry-aware gaussian splatting for scene rendering. In European Conference on Computer Vision, 441â457. Springer.

Li, Y.; and Tombari, F. 2022. E-graph: Minimal solution for rigid rotation with extensibility graphs. In European Conference on Computer Vision, 306â322. Springer.

Li, Y.; Yunus, R.; Brasch, N.; Navab, N.; and Tombari, F. 2021. RGB-D SLAM with structural regularities. In 2021 IEEE international conference on Robotics and automation (ICRA), 11581â11587. IEEE.

Matsuki, H.; Murai, R.; Kelly, P. H.; and Davison, A. J. 2023. Gaussian splatting slam. arXiv preprint arXiv:2312.06741.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Muller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In-Â¨ stant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1â 15.

Muller, T.; McWilliams, B.; Rousselle, F.; Gross, M.; and Â¨ Novak, J. 2019. Neural importance sampling. Â´ ACM Transactions on Graphics (ToG), 38(5): 1â19.

Newcombe, R. A.; Izadi, S.; Hilliges, O.; Molyneaux, D.; Kim, D.; Davison, A. J.; Kohi, P.; Shotton, J.; Hodges, S.; and Fitzgibbon, A. 2011. Kinectfusion: Real-time dense surface mapping and tracking. In 2011 10th IEEE international symposium on mixed and augmented reality, 127â136. Ieee.

Niemeyer, M.; Mescheder, L.; Oechsle, M.; and Geiger, A. 2020. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 3504â3515.

Osher, S.; Fedkiw, R.; and Piechor, K. 2004. Level set methods and dynamic implicit surfaces. Appl. Mech. Rev., 57(3): B15âB15.

Pan, X.; Lai, Z.; Song, S.; and Huang, G. 2022. Activenerf: Learning where to see with uncertainty estimation. In European Conference on Computer Vision, 230â246. Springer.

Peng, S.; Niemeyer, M.; Mescheder, L.; Pollefeys, M.; and Geiger, A. 2020. Convolutional occupancy networks. In Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part III 16, 523â540. Springer.

Peralta, D.; Casimiro, J.; Nilles, A. M.; Aguilar, J. A.; Atienza, R.; and Cajote, R. 2020. Next-best view policy for 3d reconstruction. In Computer VisionâECCV 2020 Workshops: Glasgow, UK, August 23â28, 2020, Proceedings, Part IV 16, 558â573. Springer.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted interventionâMICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, 234â241. Springer.

Savva, M.; Kadian, A.; Maksymets, O.; Zhao, Y.; Wijmans, E.; Jain, B.; Straub, J.; Liu, J.; Koltun, V.; Malik, J.; et al. 2019. Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF international conference on computer vision, 9339â9347.

Schonberger, J. L.; and Frahm, J.-M. 2016. Structure-frommotion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4104â4113.

Straub, J.; Whelan, T.; Ma, L.; Chen, Y.; Wijmans, E.; Green, S.; Engel, J. J.; Mur-Artal, R.; Ren, C.; Verma, S.; et al. 2019. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797.

Stuckler, J.; and Behnke, S. 2014. Multi-resolution surfel Â¨ maps for efficient dense 3D modeling and tracking. Journal of Visual Communication and Image Representation, 25(1): 137â147.

Sucar, E.; Liu, S.; Ortiz, J.; and Davison, A. J. 2021. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF international conference on computer vision, 6229â6238.

Sun, J.; Xie, Y.; Chen, L.; Zhou, X.; and Bao, H. 2021. Neuralrecon: Real-time coherent 3d reconstruction from monocular video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 15598â15607.

Wang, H.; Wang, J.; and Agapito, L. 2023. Co-slam: Joint coordinate and sparse parametric encodings for neural realtime slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13293â13302.

Whelan, T.; Leutenegger, S.; Salas-Moreno, R. F.; Glocker, B.; and Davison, A. J. 2015. ElasticFusion: Dense SLAM without a pose graph. In Robotics: science and systems, volume 11, 3. Rome, Italy.

Wu, S.; Sun, W.; Long, P.; Huang, H.; Cohen-Or, D.; Gong, M.; Deussen, O.; and Chen, B. 2014. Quality-driven poisson-guided autoscanning. ACM Trans. Graph., 33(6): 203â1.

Xu, Z.; Jin, R.; Wu, K.; Zhao, Y.; Zhang, Z.; Zhao, J.; Gao, F.; Gan, Z.; and Ding, W. 2024. Hgs-planner: Hierarchical planning framework for active scene reconstruction using 3d gaussian splatting. arXiv preprint arXiv:2409.17624.

Yan, Z.; Yang, H.; and Zha, H. 2023. Active neural mapping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10981â10992.

Zhu, L.; Li, Y.; Sandstrom, E.; Huang, S.; Schindler, K.; and Â¨ Armeni, I. 2024. Loopsplat: Loop closure by registering 3d gaussian splats. arXiv preprint arXiv:2408.10154.