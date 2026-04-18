# PhysMorph-GS: Differentiable Shape Morphing via Joint Optimization of Physics and Rendering Objectives

Chang-Yong Song1 David Hyde1

1Vanderbilt University

chang-yong.song@vanderbilt.edu

david.hyde.1@vanderbilt.edu

## Abstract

Shape morphing with physics-based simulation offers natural handling of large deformations and topology changes, but suffers from a fundamental ârendering gapâ: nondifferentiable surface extraction prevents pixel-space objectives from directly informing physics optimization. We introduce PhysMorph-GS, which bridges differentiable Material Point Method (MPM) with 3D Gaussian Splatting (3DGS) through a deformation-aware, stable upsampling bridge that maps sparse physics states (x, F) to dense Gaussian parameters (Âµ, Î£). Our key idea is to enable bidirectional gradient flow. Multi-modal rendering losses (silhouette ${ \mathcal { L } } _ { \alpha }$ and depth Ld) backpropagate along two parallel paths: from covariances Î£ to deformation gradients F via a stretch-based mapping ${ \pmb { \Sigma } } ^ { \prime } = { \bf S } { \pmb { \Sigma } } { \bf S } ^ { \top }$ , and from Gaussian means Âµ to particle positions x. Through the MPM adjoint, these gradients directly update deformation controls while mass is conserved at a compact set of anchor particles. A multi-pass interleaved optimization scheme repeatedly injects rendering gradients into successive physics steps, avoiding collapse to purely physicsdriven solutions. On challenging morphing sequences, PhysMorph-GS improves boundary fidelity and temporal stability over a physics-only differentiable MPM baseline while maintaining physical plausibility through fixed constitutive models and material parameter constraints. Quantitatively, our depth-supervised variant reduces Chamfer distance by 2.5% relative to a physics-only baseline, and our full model better reconstructs thin structures (e.g., ears, tails) at the cost of a mild trade-off in Chamfer. By providing a differentiable particle-to-Gaussian bridge, PhysMorph-GS closes a gap in existing physicsârendering pipelines and enables inverse design under direct visual supervision.

## 1. Introduction

Integrating physics-based simulation with high-fidelity rendering in a single differentiable pipeline remains challenging. The material point method (MPM) [14, 34, 35] robustly handles large strain and topology changes, but represents states as sparse particles (N ), whereas rendering typically expects dense, continuous surfaces (M â« N ). In practice, users simulate and cache particle states (x, F), extract surfaces with VDB [29] or marching cubes [24], then smooth and render. This unidirectional post-process blurs detail and, more importantly, breaks the computational link between physics and pixels.

Differentiable physics [12] enables gradient-based optimization of internal states or controls, but most MPMbased systems still optimize purely in the physics domain using proxy objectives on grid mass or positions. Nondifferentiable surface extraction leaves the pixel â physics feedback loop weak: image-space losses cannot directly shape the deformation field.

In parallel, 3D Gaussian Splatting (3DGS) [17] has emerged as a powerful particle-based representation for differentiable rendering. Recent work drives Gaussians with explicit simulators [42] or neural surrogates [23], or fuses physics with neural radiance fields, but typically in a one-way fashion (physicsârendering) or via learned priors. Pixel-space objectives seldom backpropagate all the way to physically meaningful quantities such as deformation gradients while respecting conservation laws.

We address this gap with PhysMorph-GS, a bidirectional coupling between differentiable MPM and 3DGS designed for physics-guided shape morphing [43]. Morphing provides a stringent testbed: large deformations and thin structures demand accurate boundaries, while physically plausible trajectories require temporal coherence and mass preservation. PhysMorph-GS builds a differentiable upsampling bridge that maps sparse MPM states (x, F) to dense Gaussian parameters (Âµ, Î£(F)) via deformationaware subdivision [1, 47] and multi-scale F-field interpolation [30, 50]. Upsampled Gaussians are render-only virtual samples, so mass remains defined at a compact set of anchor particles. Multi-modal lossesâprimarily silhouette ${ \mathcal { L } } _ { \alpha }$ and depth $\mathcal { L } _ { \mathrm { d } }$ with optional edge and shrinkage termsâsupervise the rendered images. Gradients from these losses flow back along two paths (via Âµ to positions and via Î£ to deformation gradients), and are fused with grid-based physics gradients in a multi-pass, interleaved optimization loop.

Our main contributions are:

â¢ An end-to-end MPM â3DGS optimization pipeline in which pixel-space gradients jointly update particle positions and deformation gradients, while mass conservation is enforced at a sparse set of anchor particles.

â¢ A deformation-aware upsampling bridge that uses adaptive particle subdivision and multi-scale F-field interpolation to construct anisotropic Gaussian covariances directly from the physical deformation field.

â¢ Render-informed physics control that combines gridbased mass objectives with multi-modal silhouette/depth supervision, enabling inverse design of morphing trajectories under first-principles dynamics.

â¢ A multi-pass interleaved optimization strategy that repeatedly injects rendering gradients into the MPM solver and stabilizes convergence on challenging morphing sequences.

## 2. Related Work

## 2.1. Differentiable Physics and Shape Morphing

Shape morphing algorithms range from purely geometric interpolation [2, 37] to physics-based methods, including elasticity-based deformation [4] and fluid control [25, 26]. Geometric approaches achieve visually smooth transitions but lack physical realism and struggle with complex topological changes. Physics-based methods offer greater realism but have traditionally been optimized independently from rendering, using physics-domain proxy losses (e.g., vertex positions or mass distributions) rather than visual or artist-provided targets.

Classical fluid control casts keyframe guidance as adjoint or spacetime optimization [8, 31, 36], backpropagating visual objectives through simulators. Unlike these methods, which tune external forces or boundary conditions in Eulerian solvers, we propagate pixel-space gradients through a particle-to-Gaussian bridge to update per-particle deformation gradients in MPM, better aligning with topology changes and mass conservation.

Differentiable physics simulation [6, 7, 22], particularly MPM [14, 34], has emerged as a robust framework for handling large deformations and topology changes. Building on MLS-MPM [11], ChainQueen [13] and DiffTaichi [12] enabled gradient-based control and differentiable programming for physics simulation. Xu et al. [43] proposed a differentiable MPM framework for shape morphing, optimizing per-particle deformation gradients under grid-based mass losses. However, their optimization remains confined to the physics domain due to the absence of a differentiable renderer and depends on non-differentiable surface extraction [24], creating a ârendering gapâ where pixel-space objectives cannot directly inform physics.

## 2.2. Neural Rendering and Gaussian Splatting

Neural Radiance Fields (NeRF) [27] represent scenes as continuous volumetric functions optimized via differentiable volume rendering. Extensions such as Mip-NeRF [3] and Instant-NGP [28] improve anti-aliasing and training efficiency through multi-scale representations and hash encodings, while implicit-surface methods like NeuS [38] and VolSDF [40] refine reconstruction quality by learning signed distance functions.

3D Gaussian Splatting (3DGS) [17] has recently emerged as a powerful alternative, achieving real-time, high-quality rendering using anisotropic 3D Gaussians and a visibility-aware differentiable rasterizer. This success has spurred extensions to dynamic scenes. Methods such as 4DGS [39], Deformable 3D Gaussians [44], Superpoint Gaussian Splatting [10], and Ex4DGS [20] learn deformation fields or per-Gaussian trajectories, paralleling dynamic NeRF approaches like D-NeRF [33] and HyperNeRF [32]. Other works such as SplaTAM [16] and 4DGF [9] focus on SLAM or large-scale scene reconstruction. However, motion in these approaches is purely kinematicâthey represent observed dynamics but lack explicit awareness of forces, mass, or material properties.

## 2.3. Integrating Physics with Neural Rendering

Integrating first-principles physics with neural rendering remains challenging. Early attempts using meshes [19, 45] or implicit fields [41] encounter difficulties with topological changes and computational cost.

PAC-NeRF [21] augments NeRF with continuum mechanics constraints via a hybrid EulerianâLagrangian formulation and excels at system identificationârecovering geometry and physical parameters (e.g., Youngâs modulus, density) from videos. However, it primarily tackles the forward problem of parameter estimation rather than the inverse design problem we consider, i.e., optimizing deformation trajectories toward target shapes, and relies on volumetric rendering rather than explicit Gaussians.

PhysGaussian [42] drives 3DGS with forward MPM simulations, producing high-fidelity visualizations but remaining fundamentally one-way (physicsârender): rendering gradients cannot flow back to correct physics states. PhysGaussian is well suited for visualizing pre-computed simulations but cannot use pixel-space feedback to refine trajectories toward target geometry.

OmniPhysGS [23] replaces explicit physics with neural surrogates, enabling efficient dynamics generation across diverse phenomena. This comes at the cost of potentially imperfect conservation of mass or momentum and limited robustness outside the training distribution. By contrast, our MPM-based approach preserves first-principles conservation through its variational formulation while still exposing gradients to visual objectives.

PhysDreamer [48] combines differentiable MPM with Gaussian rendering and video diffusion priors to learn neural material fields and initial velocities from pixel observations. This yields bidirectional coupling between physics and rendering but introduces dependence on learned priors, which may bias solutions toward training data and complicate generalization to unseen morphing scenarios. Our method instead optimizes physics parameters directly under pixel-space supervision, avoiding external priors.

Our Contribution. We address the ârendering gapâ in differentiable MPM morphing [43] through an MPMâ3DGS bridge that propagates rendering gradients $( \nabla { \mathcal { L } } _ { \alpha } , \ \nabla { \mathcal { L } } _ { d } )$ directly to per-particle deformation gradients F without explicit surface extraction. Unlike visualization-focused particle-to-Gaussian methods [49] or surrogate-based dynamics [23], our bidirectional coupling combines first-principles conservation with the gradient accessibility of explicit Gaussian rendering, enabling inverse morphing under direct visual supervision. Direct quantitative comparison is challengingâprior works target different problems (forward visualization, system identification, or prior-driven interaction)âso we instead validate our design via ablations of the proposed components (Â§4.2) and analyses of physical consistency (Â§4.1).

## 3. Method

Although the core ideas of this paper are applicable to other problems in vision and graphics, we focus on the task of shape morphing for the remainder of our paper. In shape morphing, the goal is to transform a source shape $S _ { 0 }$ into a target shape ${ \boldsymbol { S } } ^ { * }$ through physically plausible intermediate states while achieving high-quality rendering. We achieve this by coupling differentiable MPM with 3DGS through a deformation-aware subdivision bridge that maps sparse physics states to dense render-ready Gaussians.

## 3.1. Overview

Figure 1 illustrates our iterative optimization loop. Starting from a source shape (e.g., a sphere), each training episode consists of: (1) forward physics simulation with differentiable MPM (Â§3.3), (2) deformation-aware subdivision upsampling (Â§3.4), (3) multi-scale F-field interpolation (Â§3.5), (4) covariance construction $\Sigma ^ { \prime } = { \bf S } \Sigma { \bf S } ^ { T } ( \ S 3 . 6 ) , ( 5 )$ differentiable rendering with multi-modal losses (Â§3.7), and (6) gradient fusion and multi-pass optimization (Â§3.8).

Key design principle. Child particles are render-only virtual samples â they never participate in MPMâs particleâgrid transfers (P2G/G2P). Mass and momentum are conserved exclusively at anchor particles, preventing physics violations while still enabling dense rendering.

## 3.2. Deformation Gradient Control

To enable direct control, we augment standard MPM evolution with an explicit, learnable control deformation gradient update $\tilde { \mathbf { F } } _ { p }$ applied at selected timesteps:

$$
\mathbf { F } _ { p } ^ { n + 1 } = ( \mathbf { I } + \Delta t \mathbf { C } _ { p } ^ { n + 1 } ) ( \mathbf { F } _ { p } ^ { n } + \tilde { \mathbf { F } } _ { p } ^ { n + 1 } ) ,\tag{1}
$$

where $\mathbf { C } _ { p } ^ { n + 1 }$ denotes the particle affine velocity tensor (approximating the local velocity gradient updated from the grid). The control term $\tilde { \mathbf { F } } _ { p }$ is optimized by backpropagating loss gradients (from mass, rendering, or geometric constraints) through the forward simulation. This allows the simulation to achieve complex tracking or morphing by directly adjusting the local deformation field, bypassing the need for manual force design or surface tracking.

## 3.3. Physics-Based Loss: Grid Mass Conservation

To guide morphing toward the target while preserving physical consistency, following the work of Xu et al. [43], we supervise using a log-based grid mass loss:

$$
\mathcal { L } _ { \mathrm { m a s s } } = \sum _ { i \in \mathcal { G } } \left[ \ln ( m _ { i } + 1 + \epsilon ) - \ln ( m _ { i } ^ { * } + 1 + \epsilon ) \right] ^ { 2 } .\tag{2}
$$

This formulation operates on the Eulerian grid, eliminating the need for particle-to-particle correspondence. The loss compares total grid mass distributions, providing scale invariance and well-behaved gradients. By matching spatial mass rather than explicit positions, the approach robustly drives material toward the target shape.

To ensure simulation stability and prevent particle loss, we additionally incorporate a minimum mass penalty ${ \mathcal { L } } _ { \operatorname* { m i n } }$ The total physics objective is a weighted sum $\mathcal { L } _ { \mathrm { p h y s i c s } } ~ =$ $\mathcal { L } _ { \mathrm { m a s s } } + w _ { \mathrm { m i n } } \mathcal { L } _ { \mathrm { m i n } }$ . We refer to the Supplementary Material for the detailed formulation of ${ \mathcal { L } } _ { \operatorname* { m i n } }$ and hyperparameters. Gradients from this physics objective are backpropagated through G2P transfer via adjoint MPM, ultimately updating deformation gradients F and positions x.

## 3.4. Deformation-Aware Subdivision Upsampling

To bridge the gap between coarse physics anchors and dense render particles, we use adaptive subdivision to add child particles where deformation is strongest. The local deformation magnitude for each anchor particle i is

$$
d _ { i } = | \operatorname* { d e t } ( \mathbf { F } _ { i } ) - 1 | ,\tag{3}
$$

so both compression and expansion yield high values. The number of children $n _ { i }$ assigned to anchor i is proportional to its deformation:

$$
n _ { i } = \left\lfloor \frac { d _ { i } } { \sum _ { j } d _ { j } } \cdot M _ { \mathrm { c h i l d } } \right\rfloor , \quad \mathrm { w i t h ~ } n _ { \mathrm { m a x } } = 2 0 ,\tag{4}
$$

<!-- image-->  
Figure 1. Pipeline overview. (1) Deformation control: Differentiable MPM evolves anchor particles (blue spheres) under log-based grid mass loss ${ \mathcal { L } } _ { \mathrm { m a s s } }$ to maintain material conservation. (2) Surface-aware upsampling: Subdivision upsampling spawns child particles (cyan) in proportion to local deformation magnitude | det(F) â 1|, upsampling from sparse anchors to a dense set of render particles. A multiscale F-field then interpolates deformation gradients from coarse anchors to fine render particles with edge-aware regularization, and polar decomposition constructs anisotropic Gaussian covariances $\Sigma ^ { \prime } = \mathbf { S } \Sigma \mathbf { S } ^ { T }$ . (3) Differentiable rendering and gradient fusion: 3DGS renders from the Gaussian parameters and is supervised by silhouette and depth losses $( \mathcal { L } _ { \alpha } , \mathcal { L } _ { \mathrm { d } } )$ . These rendering gradients are combined with physics gradients via PCGrad and magnitude normalization, and are injected back into the next physics pass.

where $M _ { \mathrm { c h i l d } } ~ = ~ M - N$ is the total number of children to allocate, capped per anchor to avoid clustering. When the average deformation magnitude is negligible, uniform subdivision is used instead.

Child particles are placed by jittering around their parent anchors:

$$
\begin{array} { r } { { \bf x } _ { \mathrm { c h i l d } } = { \bf x } _ { i } + \epsilon \cdot h _ { i } \cdot { \bf e } , ~ { \bf e } \sim \mathcal { N } ( { \bf 0 } , { \bf I } ) , } \end{array}\tag{5}
$$

where $h _ { i }$ is the average local spacing computed from the nearest neighboring anchors, and Ïµ is a jitter scale hyperparameter $( \mathrm { e . g . , 0 . 1 } )$ .

This deformation-aware subdivision allocates sampling density where physics induces deformation, concentrating render capacity near thin structures and high-strain regions (Fig. 2 and Fig. 3).

Differentiability note. The allocation of $n _ { i }$ and the parentâchild assignment are discrete operations and thus nondifferentiable in a strict sense. In practice, we follow common practice in 3DGS densification: the subdivision pattern is treated as piecewise constant within each optimization episode, and gradients from child particles are accumulated at their parents through the continuous jittered positions. Empirically, this piecewise-differentiable approximation was sufficient for stable optimization in all our experiments.

## 3.5. Multi-Scale F-field Interpolation

A critical challenge in coupling sparse physics (N anchors) with dense rendering $( M \gg N$ particles) is determining how pixel-level gradients should influence the physical simulation. Our multi-scale interpolation acts as a differentiable deformation-gradient distributor, ensuring that highfrequency rendering signals are effectively scattered back to the low-frequency physics skeleton.

Two-scale estimation. For each render particle $j ,$ we compute two candidate deformation gradients using k-Nearest Neighbor (k-NN) [5] interpolation from the anchor particles $\{ \mathbf { x } _ { i } , \mathbf { F } _ { i } \}$ :

$$
\mathbf { F } _ { \mathrm { c o a r s e } } ^ { ( j ) } = \sum _ { i \in \mathcal { N } _ { 6 4 } ( j ) } w _ { j i } ^ { \mathrm { c } } \mathbf { F } _ { i } , \quad \mathbf { F } _ { \mathrm { f i n e } } ^ { ( j ) } = \sum _ { i \in \mathcal { N } _ { 1 6 } ( j ) } w _ { j i } ^ { \mathrm { f } } \mathbf { F } _ { i } ,\tag{6}
$$

where $\mathcal { N } _ { k } ( j )$ denotes the set of k nearest anchors $( k = 6 4$ for coarse, $k = 1 6$ for fine), and weights $w _ { j \ i }$ i are inversedistance weights normalized to sum to one. For scalability, all k-NN queries are accelerated using the FAISS library [15], enabling efficient interpolation even for dense point sets $( > 1 0 ^ { 5 } )$ ). $\mathbf { F } _ { \mathrm { c o a r s e } }$ captures global shape trends, while $\mathbf { F } _ { \mathrm { f i n e } }$ preserves local geometric features.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 2. Subdivision upsampling process (left to right). (a) Original shape: Initial anchors (8,918 particles) represent the coarse geometry. (b) Parent selection: 99 parents are chosen based on high local deformation, visualized by color. (c) Parentâchild relationships: For clarity, we show only the top 20 high-deformation parents (red) and their associated children (orange), highlighting how subdivision concentrates new particles around deformation hotspots. (d) Upsampled children: All 8,811 children are distributed near their parent anchors according to local spacing and jitter. (e) After upsampling (combined): The final point cloud merges anchors and children (17,729 total), resulting in adaptive spatial resolution concentrated at regions of greatest deformation.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 3. Particle distribution after subdivision (left: $x ,$ middle: $y , \mathrm { r i g h t } : z )$ . The adaptive scheme generates a spatially continuous and dense particle field. By concentrating resolution in high-deformation regions, the method ensures smooth coverage without clustering or sparsity artifacts (see Fig. 2).

Adaptive blending. We blend these scales adaptively, prioritizing the coarse scale in regions of large deformation (to maintain stability) and the fine scale in regions of small deformation (to preserve detail). The blending weight $\alpha _ { j }$ is determined by the deviation of the determinant from unity (volume conservation) and is clamped to $[ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ] \colon$

$$
d _ { \mathrm { s c a l e } } ^ { ( j ) } = | \operatorname* { d e t } ( \mathbf { F } _ { \mathrm { s c a l e } } ^ { ( j ) } ) - 1 | , \quad \mathrm { s c a l e } \in \{ \mathrm { c o a r s e , f n e } \} ,\tag{7}
$$

$$
\alpha _ { j } = \mathrm { c l a m p } \left( \sigma \left( \frac { d _ { \mathrm { c o a r s e } } ^ { ( j ) } - d _ { \mathrm { f i n e } } ^ { ( j ) } } { \tau } \right) , \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } \right) ,\tag{8}
$$

where $\sigma ( \cdot )$ is the sigmoid function, Ï is a temperature parameter, and $\alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } }$ are user-defined blending limits. The final deformation gradient is

$$
\mathbf { F } _ { \mathrm { f i n a l } } ^ { ( j ) } = \alpha _ { j } \mathbf { F } _ { \mathrm { c o a r s e } } ^ { ( j ) } + ( 1 - \alpha _ { j } ) \mathbf { F } _ { \mathrm { f i n e } } ^ { ( j ) } .\tag{9}
$$

This formulation ensures that gradients $\nabla _ { \mathbf { F _ { \mathrm { f i n a l } } } } \mathcal { L }$ backpropagate to the anchor states $\mathbf { F } _ { i }$ through the weighted sum, allowing multi-scale rendering feedback to guide the physics simulation.

Differentiability note. The k-NN neighborhoods $\mathcal { N } _ { k } ( j )$ are recomputed as particle positions evolve, but the neighbor indices themselves are discrete. We therefore do not differentiate through changes in neighborhood membership; gradients flow only through the continuous inverse-distance weights and the interpolated deformation gradients. As in subdivision, this yields a piecewise-differentiable mapping that we found sufficient for stable training.

<!-- image-->

<!-- image-->  
Figure 4. Multi-scale F-field statistics. Anisotropy and total deformation distributions for the morphing task are set by stability-driven loss design and by the physical material parameters (Youngâs modulus $E = 1 . 4 \times 1 0 ^ { 5 }$ , Poissonâs ratio $\nu = 0 . 3 )$ The control deformation gradient induces pronounced anisotropy, illustrating how surface-driven constraints directly shape the deformation.

## 3.6. Covariance from Deformation Gradient

Unlike standard 3DGS, which learns covariance via per-Gaussian scaling and rotation parameters, we compute Î£ deterministically from the physics state F. This ensures anisotropy is driven solely by physical deformation without extra learnable parameters. For each particle, we construct the covariance as

$$
{ \boldsymbol { \Sigma } } ^ { \prime } = \mathbf { S } { \boldsymbol { \Sigma } } \mathbf { S } ^ { T } , \quad { \boldsymbol { \Sigma } } = s ^ { 2 } \mathbf { I } ,\tag{10}
$$

where S is the symmetric stretch tensor from the polar decomposition ${ \bf F } = { \bf R } { \bf S }$ . By using only S (excluding rotation

R), we ensure that Gaussian shapes reflect pure deformation (elongation/compression) independent of orientation. To maintain stability, we apply differentiable soft-clamping to the singular values of S to prevent degenerate or excessive stretching.

This formulation defines a smooth gradient path from the rendering loss to the deformation gradients: gradients $\nabla _ { \Sigma ^ { \prime } } \mathcal { L }$ flow through Eq. 10 and the SVD-based polar decomposition back to F, given a fixed subdivision and neighborhood structure.

## 3.7. Rendering Supervision

We supervise the morphing process using multi-channel rendering losses computed at key timesteps. The total rendering loss is a weighted sum:

$$
\mathcal { L } _ { \mathrm { r e n d e r } } = w _ { \alpha } \mathcal { L } _ { \alpha } + w _ { d } \mathcal { L } _ { \mathrm { d } } + w _ { e } \mathcal { L } _ { \mathrm { e } } + w _ { s } \mathcal { L } _ { \mathrm { s h r i n k } } .\tag{11}
$$

The silhouette loss $( \mathcal { L } _ { \alpha } )$ and depth loss $( \mathcal { L } _ { \mathrm { d } } )$ act as primary drivers for implicit geometric guidance, enforcing 2D shape boundaries and 3D structure alignment between rendered outputs and ground truth. The edge loss $( \mathcal { L } _ { \mathrm { e } } )$ , computed as a difference of Sobel-filtered outputs, emphasizes boundary sharpness and prevents blurred or ambiguous silhouettes, which commonly arise when morphing thin structures or regions of high curvature. Optionally, an opacity shrinkage loss $( \mathcal { L } _ { \mathrm { s h r i n k } } )$ penalizes superfluous interior particles, promoting the emergence of crisp, watertight object boundaries by pruning points that do not contribute to the visible surface.

All loss channels are applied differentiably at the pixel or voxel level, ensuring that gradients can flow back through the entire simulation-to-rendering chain. Figure 5 visualizes the spatial distribution and evolution of these losses over the morphing sequence, showing how combined multi-channel supervision focuses optimization effort on silhouettes, surface thinning, and interior pruning.

## 3.8. Gradient Fusion and Optimization

Optimization involves two competing gradient sources: physics gradients $\mathbf { g } _ { \mathrm { p h y s } }$ (from mass conservation) and rendering gradients $\mathbf { g } _ { \mathrm { r e n d e r } }$ (from visual losses). We employ Project Conflicting Gradients (PCGrad) [46] to resolve conflicts: if $\mathbf { g } _ { \mathrm { p h y s } } \cdot \mathbf { g } _ { \mathrm { r e n d e r } } < 0$ , the rendering gradient is projected onto the normal plane of $\mathbf { g } _ { \mathrm { p h y s } } ,$ , ensuring rendering objectives do not violate physical consistency. Given the scale disparity between gradient sources, we apply magnitude normalization before fusion.

The combined gradients are injected as external forces into the next simulation pass, closing the loop for renderinformed physics. We use a multi-pass interleaved strategy (Alg. B, Supplementary), iterating physicsârender updates within each episode to accelerate convergence while maintaining stability.

MORPHING SEQUENCES

<!-- image-->  
Figure 5. Rendering loss evolution. Top: $\mathcal { L } _ { \alpha }$ localizes errors along shape boundaries $( t = 0 , 1 0 , 4 0 )$ . Bottom: depth hits visualize $\mathcal { L } _ { \mathrm { s h r i n k } }$ activation for interior pruning. The combined supervision ensures clean silhouettes and removes internal artifacts as morphing progresses.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
high stiffness

Figure 6. Impact of Material Stiffness on Optimization Behavior. Optimization results under varying constitutive parameters. (Left) Hard Material $( E \approx 1 . 4 \times 1 0 ^ { 5 } , \nu \approx 0 . 2 0 ) $ : High stiffness restricts deformation, resulting in low anisotropy (mean â 1.02) and rigid-body motion. (Center) Medium Material $( E \approx 2 . 8 \times 1 0 ^ { 4 } , \nu \approx 0 . 4 0 ) $ : Intermediate stiffness shows transitional behavior with moderate stretching. (Right) Soft Material $( E \approx 3 . 0 \times 1 0 ^ { 3 } , \nu \approx 0 . 4 8 )$ : Low stiffness allows significant elastic deformation, showing high anisotropy (mean â 1.12) and volumetric force propagation.

## 4. Experiments

## 4.1. Analysis of Physical Consistency

To validate the bidirectional coupling between physics and rendering, we analyzed how material properties constrain the Gaussian covariance evolution. We tested three distinct material settings:

â¢ Hard $( E ~ \approx ~ 1 . 4 ~ \times ~ 1 0 ^ { 5 } , ~ \nu ~ \approx ~ 0 . 2 0 ) \colon$ Stiff material resists rendering gradients, maintaining near-spherical Gaussians (low anisotropy â 1.02) via rigid translation.

â¢ Medium (E â 2.8 Ã 104, Î½ â 0.40): Intermediate stiffness balances shape preservation with elastic deformation, showing moderate boundary anisotropy.

â¢ Soft (E â 3.0 Ã 103, Î½ â 0.48): Low shear modulus allows rendering gradients to propagate internally, inducing large stretches (high anisotropy â 1.12) while preserving volume.

This stiffness-dependent behavior confirms that PhysMorph-GS respects the underlying constitutive model, producing physically plausible deformations rather than arbitrary geometric warping.

## 4.2. Ablation Study

To rigorously validate our contributions, we performed an ablation study comparing the Physics-Only baseline against our Render-Informed models. The results, summarized in Table 2, reveal a compelling trade-off between numerical safety (conservative shapes) and structural fidelity (detailed reconstruction).

Quantitative Impact of Rendering Loss. Our Depth-Only configuration achieves the lowest Chamfer Distance (CD â 0.0546), quantitatively outperforming the Physics-Only baseline (CD â 0.0560). While both methods performed well (the baseline already had a rather low CD), this 2.5% improvement demonstrates that our differentiable bridge adds meaningful improvements by propagating pixel-space gradients to guide physical simulation toward target geometry.

Density-Fidelity Trade-off. The Full Model (CD â 0.0595) incorporates additional silhouette loss (LÎ±) and edge loss $( \mathcal { L } _ { e } ) .$ , triggering aggressive subdivision in highcurvature regions. This results in a 4.3Ã increase in particle density (315k â 1.37M), enabling reconstruction of thin features (ears, legs) entirely absent in the baselineâs smooth âblobâ shape.

The modest CD increase reflects an inherent tradeoff: sparse conservative shapes minimize point-wise error, while dense detailed reconstruction incurs local misalignments in challenging thin regions. For applications prioritizing visual fidelity over metric optimality, the Full Model

# Timestep t 0 0 0 \* \*

Figure 7. High resolution results with Physics-Guided Shape Morphing. Snapshots from source sphere into targets. PhysMorph-GS successfully guides the physical simulation to match the target geometry. Starting from a coarse MPM initialization, the pipeline progressively optimizes deformation gradients and positions. Note how the system captures fine geometric details (ears, tail) while maintaining physical connectivity and mass conservation throughout the sequence.

provides superior structural authenticity. For geometric accuracy, Depth-Only configuration is recommended.

Optimization Dynamics. Figure 8 demonstrates effective convergence: physics loss decreases 96.5% and depth loss reduces 75.4%. Alpha loss maintains low values (â¼0.06), functioning as a boundary regularizer that sharpens silhouettes and triggers subdivision rather than serving as a primary optimization target. This design choice explains the configurations in Table 2: Depth-Only prioritizes geometric accuracy, while Full Model leverages alphadriven subdivision for structural density.

## 4.3. Positioning Relative to Prior Work

As summarized in Table 1, existing physicsârendering pipelines differ fundamentally in their ability to support inverse, target-driven shape design. PhysGaussian [42] uses 3DGS for forward visualization of MPM simulations and remains one-way (physicsârender), so rendering errors cannot be used to update the underlying physics states. OmniPhysGS [23] and PhysDreamer [48] obtain differentiability through neural surrogates or diffusion-based priors, which may relax strict conservation of physical quantities (e.g., mass or momentum) and limit generalization outside their training distribution. PAC-NeRF [21] tackles a complementary inverse problemâsystem identification of physical parameters from videosârather than optimizing deformation trajectories from given target shapes.

Table 1. Methodological Comparison. Existing MPM-3DGS works focus on forward simulation (visualization) or rely on learned priors. Our PhysMorph-GS is distinct as it enables inverse optimization for shape morphing by establishing a bidirectional gradient flow while strictly enforcing physical conservation.
<table><tr><td>Method</td><td>Core Task</td><td>Gradient Flow (Phys ââ Render)</td><td>Target-Driven Optimization</td><td>Physics Integrity</td></tr><tr><td>PhysGaussian [42] OmniPhysGS [23]</td><td>Fwd. Simulation Generation</td><td>No (One-way) Yes (Surrogate)</td><td>No No</td><td>MPM (Explicit) Learned Prior</td></tr><tr><td>PhysDreamer [48] PAC-NeRF [21]</td><td>Interaction Sys. ID</td><td>Yes (Diff. Prior) No (Param. Est.)</td><td>No No</td><td>Learned Prior Hybrid</td></tr><tr><td>Ours</td><td>Shape Morphing</td><td>Yes (Bidirectional)</td><td>Yes</td><td>MPM (Explicit)</td></tr></table>

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 8. Training convergence. Physics loss (main objective) decreases 96.5%, while depth loss (geometric guidance) reduces 75.4%. Alpha loss functions as a boundary regularizer, maintaining low values (â¼0.06) to preserve silhouette sharpness without dominating optimization.

Table 2. Quantitative Ablation. Comparison of geometry metrics (CD â) and structural density (Particle Count â). Ours (Depth) achieves the best geometric alignment, while Ours (Full) maximizes structural density to capture fine details.
<table><tr><td>Method</td><td>Chamfer Dist. (â)</td><td>Particle Count (â)</td></tr><tr><td>(a) Physics Only (Baseline)</td><td>0.0560</td><td>315,211</td></tr><tr><td>(b) Ours (Alpha Only)</td><td>0.0553</td><td>1,368,756</td></tr><tr><td>(c) Ours (Depth Only)</td><td>0.0546</td><td>1,372,736</td></tr><tr><td>(d) Ours (Full Model)</td><td>0.0595</td><td>1,372,406</td></tr></table>

In contrast, PhysMorph-GS provides a differentiable, bidirectional MPMâ3DGS bridge. Pixel-space gradients (from $\mathcal { L } _ { \alpha } , \mathcal { L } _ { \mathrm { d } } )$ propagate directly to per-particle deformation controls while explicit mass conservation is enforced at a compact set of anchor particles. Because prior methods do not expose such a renderâphysics path for morphing, we focus our evaluation on ablations of this bridge and on material-consistency analyses in Â§4.1 rather than direct metric comparisons.

## 5. Limitations and Future Work

Our method, while effective, presents limitations. High computation and memory demands pose scalability challenges as resolution increases. The log-based physics loss prioritizes stability, potentially limiting extreme deformations without targeted tuning, and reconstructing watertight surfaces or thin features remains difficult. Additionally, conservation laws are enforced only at anchor particles, so secondary effects like collisions may not fully propagate. Finally, our current validation is limited to morphing tasks.

Future work will address scalability through GPU optimization and adaptive sampling. We also plan to enhance surface fidelity via differentiable mesh or implicit reconstruction and expand to richer multiphysics models. Investigating end-to-end learning for control and validating the pipeline in broader dynamic scenariosâsuch as articulated bodies or mixed soft/rigid systemsâremains an open topic.

## Acknowledgements

We thank Simon Kassman, Manish Acharya, and Misa Viveiros for helpful discussions and insightful feedback during the early stages of this work. This material is based upon work supported by the National Science Foundation under Grant No. 2450401.

## A. Constitutive model

We use fixed-corotational elasticity [14] for stable large deformations. The PiolaâKirchhoff stress is

$$
\mathbf { P } ( \mathbf { F } ) = 2 \mu ( \mathbf { F } - \mathbf { R } ) + \lambda ( J - 1 ) J \mathbf { F } ^ { - T } ,\tag{12}
$$

where $\mu$ and Î» are Lame parameters, Â´ ${ J } = \operatorname* { d e t } ( \mathbf { F } )$ , and R is the rotational component from the polar decomposition ${ \bf F } = { \bf R } { \bf S }$ This formulation removes rotational artifacts while preserving stretch/compression.

## A.1. Minimum mass penalty

To prevent particle loss during simulation, we add a soft constraint:

$$
\mathcal { L } _ { \mathrm { m i n } } = \sum _ { m _ { i } < m _ { \mathrm { m i n } } } ( m _ { \mathrm { m i n } } - m _ { i } ) ^ { 2 } ,\tag{13}
$$

where $m _ { \mathrm { m i n } } = 1 0 ^ { - 3 }$ is the minimum allowed grid mass. The total physics loss is

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \mathcal { L } _ { \mathrm { m a s s } } + w _ { \mathrm { m i n } } \mathcal { L } _ { \mathrm { m i n } } ,\tag{14}
$$

with $w _ { \mathrm { m i n } } = 5 . 0$ . Gradients $\nabla _ { m } \mathcal { L }$ are stored in grid nodes and backpropagated through G2P transfer to particles via adjoint MPM, ultimately updating deformation gradients F and positions x.

## B. Algorithm Details

Our optimization loop interleaves physics-only and renderinformed passes within each training episode. Figure S1 provides a high-level overview. Pass 1 runs a forward MLS-MPM simulation and evaluates the physics objective ${ \mathcal { L } } _ { \mathrm { p h y s i c s } } = { \mathcal { L } } _ { \mathrm { m a s s } } + w _ { \mathrm { m i n } } { \mathcal { L } } _ { \mathrm { m i n } }$ Passes 2 and 3 restart from the same initial state but inject rendering gradients obtained from the previous pass, optimizing the combined loss $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { p h y s i c s } } + \mathcal { L } _ { \mathrm { r e n d e r } }$ via adjoint MPM.

The bottom panel highlights how gradients from the two objectives are propagated. Rendering loss $\mathcal { L } _ { \mathrm { r e n d e r } }$ backpropagates through the differentiable renderer to Gaussian parameters $( \mu , \Sigma ^ { \prime } )$ , and further to particle positions and deformation gradients via the subdivision bridge. Physics loss $\mathcal { L } _ { \mathrm { p h y s i c s } }$ produces grid-mass gradients that are backpropagated to the same deformation controls. We combine these signals with PCGrad and magnitude normalization before updating the control deformation gradients FË in Algorithm S2.

## B.1. Implementation Details

Simulation Settings. We employ a moving least-squares material point method (MLS-MPM) solver [11, 43] on a $3 2 \times 3 2 \times 3 2$ background grid with a fixed time step $\Delta t =$ 1/120. We initialize the source shape with N â 1,000 anchor particles, which are upsampled to $\mathcal { O } ( 1 0 ^ { 6 } )$ render particles through our adaptive subdivision bridge to capture fine geometric details. The main simulation and optimization hyperparameters shared across morphing sequences are summarized in Table S1.

Table S1. Simulation and optimization hyperparameters. Values given as ranges indicate the minimum and maximum used in our sweeps. For all morphing sequences in this supplementary and in the main paper, we used $w _ { \mathrm { m a s s } } \in [ 1 . 0 , 5 . 0 ]$
<table><tr><td>Block</td><td>Parameter</td><td>Value</td></tr><tr><td colspan="3">Simulation</td></tr><tr><td></td><td>Grid resolution</td><td>323 (bounding box [â16, 16]3)</td></tr><tr><td></td><td>Grid spacing dx</td><td>1.0</td></tr><tr><td></td><td>Time step ât</td><td>1/120 s</td></tr><tr><td></td><td>Density Ï</td><td>60-75</td></tr><tr><td></td><td>LamÃ© Î»</td><td> $2 . 0 \times 1 0 ^ { 4 } â 3 . 9 \times 1 0 ^ { 4 }$ </td></tr><tr><td></td><td>LamÃ© Âµ</td><td> $1 . 0 \times 1 0 ^ { 3 } â 5 . 8 3 \times 1 0 ^ { 4 }$ </td></tr><tr><td></td><td>Drag</td><td>0.5</td></tr><tr><td></td><td>External force</td><td>(0, 0, 0)</td></tr><tr><td></td><td>Smoothing factor Î³</td><td>0.955</td></tr><tr><td colspan="3">Optimization</td></tr><tr><td></td><td>Episodes / animations</td><td>4050</td></tr><tr><td></td><td>Timesteps per episode</td><td>10</td></tr><tr><td></td><td>Control stride</td><td>1-3</td></tr><tr><td></td><td>Max GD / LS iterations</td><td>13 (GD) / 15 (LS)</td></tr><tr><td></td><td>Initial step size Î±</td><td>0.01</td></tr><tr><td></td><td>Adaptive step size</td><td>enabled</td></tr><tr><td colspan="3">Loss weights</td></tr><tr><td></td><td>Physics loss weight wmass</td><td>1.0 (fixed; sweeps in [1, 5])</td></tr><tr><td></td><td>Minimum-mass weight wmin</td><td>5.0</td></tr><tr><td></td><td> $( w _ { \alpha } , w _ { \mathrm { d e p t h } } , w _ { \mathrm { e d g e } } , w _ { s } )$ </td><td>(1.5, 4.0, 3.0, 0.5)</td></tr><tr><td></td><td> $( w _ { \mathrm { p h o t o } } , w _ { \mathrm { c o v - a l i g n } } , w _ { \mathrm { c o v - r e g } } , w _ { \mathrm { d e t } } )$ </td><td>(0.8, 6.0, 0.02, 0.15)</td></tr></table>

Auxiliary rendering regularizers. In addition to the primary terms in Eq. (11) (silhouette ${ \mathcal { L } } _ { \alpha } ,$ depth $\mathcal { L } _ { d } ,$ edge $\mathcal { L } _ { e } ,$ and shrinkage $\mathcal { L } _ { \mathrm { s h r i n k } } )$ , we use weak auxiliary losses for coverage/photometric consistency and covariance conditioning. Their weights $( w _ { \mathrm { p h o t o } } , w _ { \mathrm { c o v - a l i g n } } , w _ { \mathrm { c o v - r e g } } , w _ { \mathrm { d e t } } )$ are listed in Table S1. These auxiliary terms act as stabilizers and are kept fixed across all experiments; the ablations in Sec. 4.2 of the main paper vary only the primary rendering channels while leaving these regularizers unchanged.

Neighborhood sizes. Eq. (6) in the main paper uses $( k _ { F } ^ { \mathrm { c o a r s e } } , k _ { F } ^ { \mathrm { f i n e } } ) = ( 6 4 , 1 6 )$ as our default setting. The sphereâheart example in Table S2 uses a slightly larger coarse neighborhood $( k _ { F } ^ { \mathrm { c o a r s e } } = 9 6 )$ due to its higher target resolution; all other sequences use comparable values in the range 64â96 for $k _ { F } ^ { \mathrm { c o a r s e } }$ with $k _ { F } ^ { \mathrm { f i n e } } = 1 6$

Optimization Hyperparameters. We optimize the control deformation gradients $\tilde { \mathbf { F } } _ { p }$ with Adam [18] using a learning rate of $1 \times 1 0 ^ { - 2 }$ . In practice, we run 40â50 episodes per sequence, and training typically converges within this budget. All morphing sequences in the main paper and this supplementary fix the physics loss weight at $w _ { \mathrm { m a s s } } = 1 . 0 ;$ internal sweeps with $w _ { \mathrm { m a s s } } \in [ 1 , 5 ]$ produced qualitatively similar behavior. The loss weights for physics and rendering terms are summarized in Table S1. To ensure rendering stability and prevent degenerate Gaussians $( \mathrm { e . g . }$ , needle artifacts), singular values of the deformation gradient are softly clamped, and base Gaussian scales are chosen as in Table S2.

<!-- image-->  
Figure S1. Multi-pass interleaved optimization and gradient flow. Top: Each training episode performs multiple passes. Pass 1 runs a forward MPM simulation and evaluates the physics loss ${ \mathcal { L } } _ { \mathrm { p h y s i c s } } .$ . Passes 2 and 3 re-run the simulation while injecting rendering gradients from the previous pass and descending the combined loss $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { p h y s i c s } } + \mathcal { L } _ { \mathrm { r e n d e r } }$ with adjoint MPM. Bottom: Gradient paths for both objectives. Rendering loss $\mathcal { L } _ { \mathrm { r e n d e r } }$ backpropagates through the differentiable renderer to Gaussian means and covariances, and further to particle positions and deformation gradients, yielding $\nabla _ { F } \mathcal { L } _ { \mathrm { r e n d e r } }$ . Physics loss $\mathcal { L } _ { \mathrm { p h y s i c s } }$ produces grid-mass gradients that backpropagate via adjoint MPM to $\nabla _ { F } \mathcal { L } _ { \mathrm { p h y s i c s } }$ . We fuse these signals using PCGrad to obtain the combined control gradient $\nabla _ { F } \mathcal { L } _ { \mathrm { t o t a l } }$ used in the next pass.

Table S2. Upsampling, camera, and rendering settings for a representative sphere-to-heart morphing example.
<table><tr><td>Block</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="6">Upsample</td><td>Target samples M Subdivision target</td><td> $1 . 2 \times 1 0 ^ { 5 }$  2.5 Ã 106</td></tr><tr><td>Micro jitter scale</td><td>0.18</td></tr><tr><td></td><td>0.02</td></tr><tr><td>Uniform mix ratio</td><td></td></tr><tr><td> $( k _ { F } ^ { \mathrm { c o a r s e } } , k _ { F } ^ { \mathrm { f i n e } } )$ </td><td>(96, 16)</td></tr><tr><td>Base scale 0 Isotropic scale iso</td><td>0.055 0.036</td></tr><tr><td rowspan="6">Camera</td><td>SV clamp range</td><td>[0.35, 2.5]</td></tr><tr><td></td><td>3840 Ã 2160</td></tr><tr><td>Resolution (W, H) Focal  $( f _ { x } , \dot { f } _ { y } )$ </td><td>(1425, 1425)</td></tr><tr><td></td><td></td></tr><tr><td> $\begin{array} { l } { { \mathrm { P r i n c i p a l ~ p o i n t } \left( c _ { x } , c _ { y } \right) } } \\ { { \mathrm { N e a r } / \mathrm { f a r } } } \end{array}$ </td><td>(1920, 1080)</td></tr><tr><td>Eye â target</td><td>0.01/100  $( 2 0 , - 2 5 , 1 2 . 5 )  ( 0 , 0 , 0 )$ </td></tr><tr><td rowspan="6">Rendering</td><td>Background color</td><td>white (1, 1, 1)</td></tr><tr><td>Training res. scale</td><td></td></tr><tr><td></td><td>1.0</td></tr><tr><td>Particle color</td><td>(0.27, 0.51, 0.71)</td></tr><tr><td>Surface mask ratio</td><td>0.25 (edge-based)</td></tr><tr><td>Gradient focus min. ratio Lighting model</td><td>0.25 (visible only) Phong, directional</td></tr></table>

Camera-aware Rendering Gradients. We restrict render gradients to camera-visible particles to avoid diluting physics updates. An edge+depth visibility mask is computed per pass, and both image-space losses $( { \mathcal { L } } _ { \alpha } , { \mathcal { L } } _ { d } ,$ , and optional coverage/photometric losses grouped under ${ \mathcal { L } } _ { \mathrm { p h o t o } } )$ and backpropagated gradients are scaled by the visible ratio. If the mask falls below 5% or exceeds 60%, the system automatically disables the mask or subsamples it to keep magnitudes stable.

Morphing Sequences. All morphing sequences in Sec. 4.2 share the global simulation and optimization settings summarized in Table S1. Table S2 reports one representative upsampling and rendering configuration (sphereâheart); the remaining sequences use comparable values with minor adjustments to target sample count and camera pose. Per-example parameters (start/end mesh pair, number of timesteps, density, smoothing factor Î³, and initial/target particle counts) are provided in the accompanying configuration files used to generate the main-paper results.

Algorithm 7: Multi-Pass Interleaved Optimization   
Input: Anchor particles $( x ^ { 0 } , F ^ { 0 } )$ , target shape $S ^ { * }$ , number of episodes $N _ { \mathrm { e p i s o d e s } }$ (40â50 in our experiments)   
Output: Optimized trajectory $\{ ( \boldsymbol { x } ^ { t } , \bar { F } ^ { t } ) \} _ { t = 0 } ^ { T }$   
1: for $e = 1$ to $N _ { \mathrm { e p i s o d e s } }$ do   
2: for $p = 1$ to 3 do // Multi-pass strategy   
3: // Forward Physics Simulation   
4: $( x , F ) \gets \mathbf { M P M } \mathbf { \mathrm { F o r w a r d } } ( \tilde { F } , \Delta t , T )$   
5: $\mathcal { L } _ { \mathrm { p h y s } }  \mathcal { L } _ { \mathrm { m a s s } } + w _ { \mathrm { m i n } } \mathcal { L } _ { \mathrm { m i n } } \quad / / E q . ( 2 ) ,$ (13)   
6: // Subdivision Upsampling   
7: $d _ { i } \gets$ | det(Fi) â 1| for all $i \in [ 1 , N ]$ // Eq. (3)   
8: $n _ { i } \gets \left\lfloor \frac { d _ { i } } { \sum _ { j } d _ { j } } \cdot M _ { \mathrm { c h i l d } } \right\rfloor$ , capped at 20 // Eq. (4)   
9: Spawn children: $\boldsymbol { x _ { c } } = \overline { { \boldsymbol { x } } } _ { i } + \epsilon \boldsymbol { h } _ { i } \cdot \boldsymbol { e } , \boldsymbol { e } \sim \mathcal { N } ( 0 , I )$ // Eq. (5)   
10: // Multi-Scale F-field Interpolation   
11: for each render particle $j \in [ 1 , M ]$ do   
12: $\begin{array} { r } { F _ { \mathrm { c o a r s e } } ^ { ( j ) }  \sum _ { i \in \mathcal { N } _ { 6 4 } ( j ) } w _ { i } ^ { c } F _ { i } , \quad F _ { \mathrm { f i n e } } ^ { ( j ) }  \sum _ { i \in \mathcal { N } _ { 1 6 } ( j ) } w _ { i } ^ { f } F _ { i } \quad / / E q . ( 6 ) } \end{array}$   
13: $\alpha _ { j } \gets \mathrm { c l a m p } \left( \sigma \left( \frac { d _ { \mathrm { c o a r s e } } ^ { ( j ) } - d _ { \mathrm { f i n e } } ^ { ( j ) } } { \tau } \right) , \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } \right)$ // Eq. (8)   
14: $\begin{array} { r l } { F ^ { ( j ) }  \alpha _ { j } F _ { \mathrm { c o a r s e } } ^ { ( j ) } + ( 1 - \alpha _ { j } ) F _ { \mathrm { f n e } } ^ { ( j ) } } & { { } / / E q . ~ ( 9 ) } \end{array}$   
15: end for   
16: // Covariance Construction   
17: $F ^ { ( j ) } = R ^ { ( j ) } S ^ { ( j ) }$ via polar decomposition   
18: $\Sigma ^ { \prime ( j ) }  S ^ { ( j ) } \Sigma _ { 0 } S ^ { ( j ) \bar { T } }$ where $\Sigma _ { 0 } = s ^ { 2 } I$ // Eq. (10)   
19: // Differentiable Rendering   
20: I â GaussianSplatting( $\{ \mu _ { j } , \Sigma _ { j } ^ { \prime } \} _ { j = 1 } ^ { M } )$   
21: ${ \mathcal { L } } _ { \mathrm { r e n d e r } }  w _ { \alpha } { \mathcal { L } } _ { \alpha } + w _ { d } { \mathcal { L } } _ { d } + w _ { e } { \mathcal { L } } _ { e } + w _ { s } { \mathcal { L } } _ { \mathrm { s h r i n k } }$ // Eq. (11)   
22: // Gradient Fusion (PCGrad)   
23: $\mathbf { g } _ { \mathrm { p h y s } }  \nabla _ { \tilde { F } } \mathcal { L } _ { \mathrm { p h y s } }$ via adjoint MPM   
24: grender $ \nabla _ { \tilde { F } } \mathcal { L } _ { \mathrm { r e n d e r } }$ via $\nabla _ { \Sigma }  \nabla _ { F }  \nabla _ { \tilde { F } }$   
25: if $\mathbf { g } _ { \mathrm { p h y s } } \cdot \mathbf { g } _ { \mathrm { r e n d e r } } < 0$ then   
26: grender $ \mathbf { g } _ { \mathrm { r e n d e r } } - \frac { \mathbf { g } _ { \mathrm { r e n d e r } } \cdot \mathbf { g } _ { \mathrm { p h y s } } } { \vert \vert \mathbf { g } _ { \mathrm { p h y s } } \vert \vert ^ { 2 } } \mathbf { g } _ { \mathrm { p h y s } }$   
27: end if   
28: $\mathbf { g }  \frac { \mathbf { g } _ { \mathrm { p h y s } } } { \vert \vert \mathbf { g } _ { \mathrm { p h y s } } \vert \vert } + \frac { \mathbf { g } _ { \mathrm { r e n d e r } } } { \vert \vert \mathbf { g } _ { \mathrm { r e n d e r } } \vert \vert }$ // Magnitude normalization   
29: // Update Controls   
30: $\tilde { F }  \tilde { F } - \alpha \cdot \mathbf { g }$ // Adam optimizer, base lr $\alpha = 0 . 0 1$   
31: end for   
32: end for   
33: return $\{ \tilde { F } ^ { t } \} _ { t = 0 } ^ { T }$  
Figure S2. Complete multi-pass interleaved optimization procedure. Each episode performs three passes that progressively inject rendering gradients into physics simulation. The multi-scale F-field interpolation (lines 11â15) ensures high-frequency rendering signals effectively backpropagate to the low-frequency physics skeleton, while PCGrad (lines 22â28) prevents rendering objectives from violating physical consistency.

<!-- image-->  
Figure S3. Representative success cases. We visualize two morphing sequences that start from the same nearly spherical rest configuration. The top row shows a sphere-to-heart morph and the bottom row a sphere-to-pillar morph. In both examples, the proposed PhysMorph-GS pipeline produces smooth deformations without particle loss or visible numerical artifacts, while approximately preserving volume and avoiding self-intersection. These results indicate that our method can reliably handle large but topologically simple deformations.

<!-- image-->  
Figure S4. Failure case under extreme topological changes. Starting from a single-body source geometry (bob), the optimizer is asked to morph the object into a quadruped-like target with multiple thin legs and a protruding head. Strong rendering supervision encourages the emergence of new limbs and head structures and prevents the simulation from collapsing. However, the resulting surface exhibits noisy, high-frequency artifacts and locally over-stretched regions in later frames (right). The underlying MLSâMPM simulation remains numerically stable, but this example highlights a limitation of our current design: when topology changes are very severe and poorly supported by the coarse physics skeleton, the rendering loss can dominate and produce visually rough surfaces.

Evaluation Metrics. We report Chamfer distance (CD) for all morphing sequences. For CD, we first uniformly sample $N _ { \mathrm { p t s } } = 1 0 0 { , } 0 0 0$ points on the surface of both the predicted and target meshes using area-weighted sampling. CD is computed as the symmetric bidirectional $L _ { 2 }$ distance between the two point sets and is normalized by the diagonal length of the joint bounding box of the pair:

$$
\begin{array} { r } { \mathrm { C D } = \frac { 1 } { \| d _ { \mathrm { b b o x } } \| _ { 2 } } \Bigg [ \frac { 1 } { N _ { \mathrm { p t s } } } \sum _ { p \in P } \displaystyle \operatorname* { m i n } _ { q \in Q } \| p - q \| _ { 2 } ^ { 2 } ~ + } \\ { \frac { 1 } { N _ { \mathrm { p t s } } } \sum _ { q \in Q } \displaystyle \operatorname* { m i n } _ { p \in P } \| p - q \| _ { 2 } ^ { 2 } \Bigg ] , } \end{array}\tag{15}
$$

where P and Q denote the sampled point sets and $d _ { \mathrm { b b o x } }$ is the diagonal of the axis-aligned bounding box enclosing both meshes. All numbers reported in Sec. 4.2 are averaged over timesteps and morphing sequences.

## C. Additional Examples

## C.1. Success Cases

As representative success cases of our pipeline, we highlight the two morphing sequences shown in Fig. S3. Both sequences start from the same almost-spherical rest shape and are optimized using the full three-pass interleaved scheme with render supervision. In the first row, the material gradually forms a heart-shaped target with a pronounced groove while remaining smooth and approximately volume preserving. In the second row, the object is stretched into a tall pillar-like target without developing grid-aligned artifacts or numerical instabilities. These examples demonstrate that, when topology is preserved, PhysMorph-GS can realize large deformations while maintaining stable dynamics and visually clean surfaces.

Table S3. Runtime/Memory snapshot (single NVIDIA RTX A4000 GPU, 64GB Memory).
<table><tr><td>Stage</td><td>Time / episode</td><td>Peak Memory</td></tr><tr><td>Forward MPM (323, T=10, â¼600K)</td><td>0.91.2 min</td><td>1.5 GB</td></tr><tr><td>Subdivision + kNN (â¼1.3M)</td><td>4.05.0 min</td><td>3.2 GB</td></tr><tr><td>3DGS render + losses</td><td>2.03.0 min</td><td>8.7 GB</td></tr><tr><td>Total (3 passes)</td><td>21-30 min</td><td>18-25 GB</td></tr></table>

## C.2. Failure Case: Extreme Topological Changes

To probe the limits of PhysMorph-GS, we consider a challenging morphing sequence with substantial topological changes, visualized in Fig. S4. The source is a singlecomponent duck float, whereas the target resembles a quadruped with four legs and a bulky head. Achieving this transformation requires material to split and reattach in ways that are not explicitly represented by the coarse MPM skeleton. Strong rendering supervision successfully drives the formation of leg- and head-like structures and prevents the simulation from collapsing, but it also introduces high-frequency surface roughening and small-scale artifacts in later frames. In contrast, a physics-only baseline typically fails to break symmetry and cannot form distinct limbs. This experiment shows that our method can prioritize target topology and silhouette accuracy under extreme deformations, at the expense of local surface smoothness and strict physical realism.

## References

[1] Bart Adams, Mark Pauly, Richard Keiser, and Leonidas J Guibas. Adaptively sampled particle fluids. In ACM SIG-GRAPH, page 48, 2007. 1

[2] Marc Alexa, Daniel Cohen-Or, and David Levin. As-rigidas-possible shape interpolation. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 165â172. 2023. 2

[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Int. Conf. Comput. Vis., pages 5855â 5864, 2021. 2

[4] Sofien Bouaziz, Sebastian Martin, Tiantian Liu, Ladislav Kavan, and Mark Pauly. Projective dynamics: Fusing constraint projections for fast simulation. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 787â797. 2023. 2

[5] T. Cover and P. Hart. Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1):21â 27, 1967. 4

[6] Jonas Degrave, Michiel Hermans, Joni Dambre, and Francis Wyffels. A differentiable physics engine for deep learning in robotics. Frontiers in Neurorobotics, 13:6, 2019. 2

[7] Tao Du, Kui Wu, Pingchuan Ma, Sebastien Wah, Andrew Spielberg, Daniela Rus, and Wojciech Matusik. Diffpd: Differentiable projective dynamics. ACM Trans. Graph., 41(2): 1â21, 2021. 2

[8] Raanan Fattal and Dani Lischinski. Target-driven smoke animation. ACM Trans. Graph., 23(3):441â448, 2004. 2

[9] Tobias Fischer et al. Dynamic 3d gaussian fields for urban areas. In Int. Conf. Comput. Vis., 2024. 2

[10] Diwen Gao, Jingdong Xu, Ziwei Qiao, et al. Superpoint gaussian splatting for real-time high-fidelity dynamic scene reconstruction. In Int. Conf. Mach. Learn., 2024. 2

[11] Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana, and Chenfanfu Jiang. A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph., 37(4):150, 2018. 2, 9

[12] Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, and Fredo Durand. Difftaichi:Â´ Differentiable programming for physical simulation. arXiv preprint arXiv:1910.00935, 2019. 1, 2

[13] Yuanming Hu, Jiancheng Liu, Andrew Spielberg, Joshua B. Tenenbaum, William T. Freeman, Jiajun Wu, Daniela Rus, and Wojciech Matusik. Chainqueen: A real-time differentiable physical simulator for soft robotics. In IEEE Int. Conf. Robot. Autom., pages 6265â6271, 2019. 2

[14] Chenfanfu Jiang, Craig Schroeder, Joseph Teran, Alexey Stomakhin, and Andrew Selle. The material point method for simulating continuum materials. In ACM SIGGRAPH 2016 Courses, pages 1â52. 2016. 1, 2, 9

[15] Jeff Johnson, Matthijs Douze, and Herve J Â´ egou. Billion- Â´ scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535â547, 2019. 4

[16] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In IEEE Conf. Comput. Vis. Pattern Recog., pages 21357â21366, 2024. 2

[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139:1â 139:14, 2023. 1, 2

[18] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. 9

[19] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Trans. Graph., 39(6):1â14, 2020. 2

[20] Junoh Lee, Changyeon Won, Hyunjun Jung, Inhwan Bae, and Hae-Gon Jeon. Fully explicit dynamic gaussian splatting. In Adv. Neural Inform. Process. Syst., 2024. 2

[21] Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, et al. Pacnerf: Physics augmented continuum neural radiance fields for geometry-agnostic system identification. In Int. Conf. Learn. Represent., 2023. 2, 8

[22] Yifei Li, Tao Du, Kui Wu, Jie Xu, and Wojciech Matusik. Diffcloth: Differentiable cloth simulation with dry frictional contact. ACM Trans. Graph., 42(1):2:1â2:20, 2022. 2

[23] Yuchen Lin, Chenguo Lin, Jianjin Xu, and Yadong Mu. Omniphysgs: 3d constitutive gaussians for general physics-based dynamics generation. arXiv preprint arXiv:2501.18982, 2025. 1, 2, 3, 8

[24] William E Lorensen and Harvey E Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field, pages 347â353. 1998. 1, 2

[25] Pierre-Luc Manteaux, Ulysse Vimont, Chris Wojtan, Damien Rohmer, and Marie-Paule Cani. Space-time sculpting of liquid animation. In Int. Conf. on Motion in Games (MIG), pages 61â71, 2016. 2

[26] Antoine McNamara, Adrien Treuille, Zoran Popovic, and Jos Â´ Stam. Fluid control using the adjoint method. ACM Trans. Graph., 23(3):449â456, 2004. 2

[27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Eur. Conf. Comput. Vis., 2020. 2

[28] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1â 102:15, 2022. 2

[29] Ken Museth. Vdb: High-resolution sparse volumes with dynamic topology. ACM Trans. Graph., 32(3):27:1â27:22, 2013. 1

[30] Y. Ohtake, A. Belyaev, and H.P. Seidel. A multi-scale approach to 3d scattered data interpolation with compactly supported basis functions. In Shape Modeling Int. (SMI), pages 153â161, 2003. 1

[31] Zherong Pan and Dinesh Manocha. Efficient solver for spacetime control of smoke. ACM Trans. Graph., 36(5): 162:1â162:13, 2017. 2

[32] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021. 2

[33] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In IEEE Conf. Comput. Vis. Pattern Recog., 2021. 2

[34] Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph Teran, and Andrew Selle. A material point method for snow simulation. ACM Trans. Graph., 32(4):1â10, 2013. 1, 2

[35] Deborah Sulsky, Zhen Chen, and Howard L Schreyer. A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng., 118(1-2):179â196, 1994. 1

[36] Adrien Treuille, Antoine McNamara, Zoran Popovic, and Jos Â´ Stam. Keyframe control of smoke simulations. ACM Trans. Graph., 22(3):716â723, 2003. 2

[37] Greg Turk and James F Oâbrien. Shape transformation using variational implicit functions. In ACM SIGGRAPH 2005 Courses, pages 13âes. 2005. 2

[38] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In Adv. Neural Inform. Process. Syst., 2021. 2

[39] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering, 2024. 2

[40] Haoyu Wu, Alexandros Graikos, and Dimitris Samaras. Svolsdf: Sparse multi-view stereo regularization of neural implicit surfaces. In Int. Conf. Comput. Vis., pages 3556â3568, 2023. 2

[41] Sean Wu, Shamik Basu, Tim Broedermann, Luc Van Gool, and Christos Sakaridis. Pbr-nerf: Inverse rendering with physics-based neural fields. In IEEE Conf. Comput. Vis. Pattern Recog., pages 10974â10984, 2025. 2

[42] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physicsintegrated 3d gaussians for generative dynamics. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4389â4398, 2024. 1, 2, 8

[43] Michael Xu, Chang-Yong Song, David Levin, and David Hyde. A differentiable material point method framework for shape morphing. IEEE Trans. Vis. Comput. Graph., 31(10): 9140â9153, 2025. 1, 2, 3, 9

[44] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction, 2023. 2

[45] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Oztireli, and Â¨ Olga Sorkine-Hornung. Differentiable surface splatting for point-based geometry processing. ACM Trans. Graph., 38 (6):1â14, 2019. 2

[46] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Adv. Neural Inform. Process. Syst., 2020. 6

[47] Pingping Zhang, Xu Wang, Lin Ma, Shiqi Wang, Sam Kwong, and Jianmin Jiang. Progressive point cloud upsampling via differentiable rendering. IEEE Trans. Circuits Syst. Video Technol., 31(12):4673â4685, 2021. 1

[48] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y. Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T. Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation, 2024. 3, 8

[49] Jingqiu Zhou, Lue Fan, Xuesong Chen, et al. Gaussianpainter: Painting point cloud into 3d gaussians with normal guidance. In AAAI Conf. Artif. Intell., 2025. 3

[50] Tobias Zirr and Anton S Kaplanyan. Real-time rendering of procedural multiscale materials. In ACM SIGGRAPH Symp. Interact. 3D Graph. Games, pages 139â148, 2016. 1