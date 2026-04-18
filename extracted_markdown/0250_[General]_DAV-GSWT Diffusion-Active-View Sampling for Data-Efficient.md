<!-- page 1 -->
DAV-GSWT: DIFFUSION-ACTIVE-VIEW SAMPLING FOR
DATA-EFFICIENT GAUSSIAN SPLATTING WANG TILES
Rong Fu∗
University of Macau
mc46603@um.edu.mo
Jiekai Wu
Juntendo University
ketsu0612@gmail.com
Haiyun Wei
Tongji University
2311399@tongji.edu.cn
Yee Tan Jia
Renmin University of China
tanjiayi2002@ruc.edu.cn
Yang Li
University of Chinese Academy of Sciences
liyang221@mails.ucas.ac.cn
Xiaowen Ma
Zhejiang University
xwma@zju.edu.cn
Wangyu Wu
University of Liverpool
v11dryad@foxmail.com
Simon Fong
University of Macau
ccfong@um.edu.mo
March 9, 2026
ABSTRACT
The emergence of 3D Gaussian Splatting has fundamentally redefined the capabilities of photorealistic
neural rendering by enabling high-throughput synthesis of complex environments. While procedural
methods like Wang Tiles have recently been integrated to facilitate the generation of expansive
landscapes, these systems typically remain constrained by a reliance on densely sampled exemplar
reconstructions. We present DAV-GSWT, a data-efficient framework that leverages diffusion priors
and active view sampling to synthesize high-fidelity Gaussian Splatting Wang Tiles from minimal
input observations. By integrating a hierarchical uncertainty quantification mechanism with generative
diffusion models, our approach autonomously identifies the most informative viewpoints while
hallucinating missing structural details to ensure seamless tile transitions. Experimental results
indicate that our system significantly reduces the required data volume while maintaining the visual
integrity and interactive performance necessary for large-scale virtual environments.
Keywords 3D Gaussian Splatting, Procedural Terrain Generation, Wang Tiles, Active Perception, Diffusion Priors,
Data-Efficient Reconstruction
1
Introduction
The fundamental framework of differentiable neural rendering has undergone a radical transformation with the advent
of 3D Gaussian Splatting (3DGS), which utilizes explicit volumetric primitives to attain an optimal balance between
photorealistic synthesis and computational efficiency [1]. Unlike traditional implicit coordinate-based representations
that rely on expensive network queries, 3DGS leverages a high-throughput rasterization pipeline to facilitate real-time
performance on modern hardware [2, 3, 4]. This breakthrough has catalyzed diverse applications, including high-fidelity
dynamic scene capture [5], dense monocular SLAM [6], and large-scale high-resolution rendering [7]. Recent efforts
have further integrated 3DGS into navigational systems to augment visual-inertial odometry through environmental
priors [8]. However, a significant bottleneck persists regarding the spatial scalability of these models. Standard pipelines
are primarily optimized for localized and bounded environments, leaving the synthesis of expansive or infinite 3D
landscapes from sparse input data as a critical open challenge in computer graphics [9, 10, 11].
∗Corresponding author: mc46603@um.edu.mo
arXiv:2602.15355v2  [cs.CV]  6 Mar 2026

<!-- page 2 -->
DAV-GSWT
To overcome the constraints of bounded reconstruction, researchers have recently explored the integration of procedural
tiling techniques with 3DGS to enable the generation of vast terrains [12]. While Gaussian Splatting Wang Tiles
(GSWT) provide a mathematical foundation for seamless stochastic tiling, their practical utility is frequently hindered by
a heavy reliance on high-quality, densely sampled exemplar models. In scenarios characterized by limited observations
or sparse viewpoints, the reconstruction of these exemplars often suffers from geometric instability and visual artifacts.
Previous methodologies aimed at improving data efficiency have focused on compact representations to reduce memory
footprints [13] or scaling vision-only occupancy reconstruction [14], yet these approaches do not natively address the
procedural continuity and edge-matching constraints required for infinite world-building. Consequently, there is an
urgent need for a more efficient paradigm that can synthesize high-fidelity tiled representations without exhaustive data
collection.
Our research is motivated by the potential of synergizing active perception with generative modeling to mitigate the
data-acquisition bottleneck. Advances in active vision demonstrate that viewpoint selection driven by uncertainty
quantification can drastically enhance the fidelity of reconstructed surfaces in unknown regions [15, 16, 17]. By
employing hierarchical uncertainty metrics, it is possible to autonomously identify the most informative perspectives
for scene completion [18, 19]. Parallel to these developments, the rapid evolution of diffusion models provides a
robust generative prior for inverting 3D structures from minimal visual evidence [20, 21]. These priors facilitate
the hallucination of dense and coherent visual textures that would otherwise require significant physical captures
[22]. By integrating these elements, we can transition from a passive, data-heavy reconstruction process to an active,
generative-driven synthesis pipeline.
We introduce DAV-GSWT, a Diffusion-Active-View sampling framework specifically engineered for the data-efficient
synthesis of Gaussian Splatting Wang Tiles. Our architecture replaces the conventional requirement for a dense
exemplar with a recursive loop that strategically acquires informative views while employing a diffusion-based refiner
to harmonize tile boundaries. This active perception cycle utilizes a projection-based next-best-view planning logic to
maximize the acquisition of new geometric information while avoiding redundant sampling. By incorporating neural
active reconstruction principles and uncertainty-aware primitive modeling, our framework ensures that the resulting tiles
are visually seamless and geometrically accurate. This approach results in a highly scalable terrain synthesis system
that remains robust in data-constrained scenarios such as rapid exploration or low-dose sensing tasks.
Our contributions are as follows. First, we develop a novel active view sampling mechanism that utilizes visual and
geometric uncertainty to prioritize informative regions for tile reconstruction. Second, we propose a multi-view diffusion-
based refinement pipeline that optimizes the Gaussian distributions at the tile boundaries to ensure perceptual continuity
and structural integrity. Finally, we present a high-performance terrain renderer that facilitates on-the-fly procedural
tiling and hierarchical level-of-detail management to enable the interactive exploration of infinite environments.
2
Related Work
2.1
Differentiable Primitives for Radiance Fields
Implicit scene representations have advanced rapidly since the introduction of Neural Radiance Fields [23]. While
early coordinate-based models delivered high fidelity, 3D Gaussian Splatting established an efficient alternative through
differentiable point-based rendering. Recent work has expanded the capability of these primitives, including VR-
Splatting, which integrates foveated rendering and neural points for immersive VR applications [24]. Complementary
efforts target compact Gaussian parameterizations to reduce memory usage [13] and achieve extreme real-time
performance, with reported rendering rates surpassing 900 fps [2]. Further geometric flexibility is provided by convex
splatting, which employs smooth convex shapes for improved surface coherence [25]. Beyond static scenes, interactive
manipulation has become more accessible through point-based editing techniques such as 3DGS-Drag [26].
2.2
Uncertainty Quantification in Visual Reconstruction
Uncertainty estimation is essential for identifying unreliable or unobserved regions during 3D reconstruction. Methods
that explicitly model the 3D uncertainty field help reveal areas where the neural representation lacks sufficient evidence
[27]. Recent work incorporates epistemic uncertainty into pre-trained models to distinguish reliable features from
unknown ones [28]. In dynamic settings, uncertainty-aware regularization has been shown to stabilize optimization in
4D Gaussian splatting [29]. For sparse-view reconstruction, depth-supervised optimal transport frameworks leverage
uncertainty metrics to align geometric priors more effectively [30]. Neural uncertainty maps have also been used to
guide inference in unseen regions and improve reconstruction completeness [16]. Complementary approaches employ
conceptual scene reasoning to assist Gaussian inpainting when visibility is limited [31].
2

<!-- page 3 -->
DAV-GSWT
2.3
Active View Selection and Robotic Planning
Active perception increasingly relies on uncertainty estimation to guide autonomous viewpoint selection. Earlier
next-best-view approaches have progressed from geometric rules to gradient-based optimization for targeted perception
tasks [32]. Modern systems commonly adopt uncertainty-driven strategies to navigate complex scenes [33, 34]. Recent
methods such as GauSS-MI employ mutual information for real-time viewpoint evaluation [35], and prediction-guided
planning has been shown to improve multi-agent reconstruction [36]. Generalizable active reconstruction policies
that operate across diverse environments have also been introduced [37]. In Gaussian-based mapping, selective image
acquisition directly boosts reconstruction quality, as demonstrated by ActiveInitSplat and ActiveGS [38, 39], while
cross-reference assessment provides an efficient mechanism for identifying high-quality inputs [40].
2.4
Tile-Based Synthesis and Large-Scale Rendering
Scaling radiance fields to urban or expansive environments necessitates sophisticated structural management. CityGS-X
represents a scalable architecture designed for high geometric accuracy in large-scale reconstruction [41]. To manage
computational resources, level-of-detail (LOD) strategies have been integrated into 3DGS to allow for customizable
rendering complexity based on the viewer’s distance [42, 43]. The historical foundation of tile-based methods for
texture synthesis provides a framework for addressing these scaling issues [44]. Specifically, the application of
Gaussian Splatting Wang Tiles (GSWT) allows for the seamless tiling of repetitive or large-scale structures while
maintaining visual coherence [45]. These tiling methods often require uncertainty-weighted seam optimization to
ensure that transitions between adjacent tiles are perceptually imperceptible, effectively handling the boundaries of the
reconstruction.
2.5
Generative Priors and Diffusion-Based Enhancement
The incorporation of generative models has significantly bolstered the ability to reconstruct scenes from minimal input
data. Diffusion probabilistic models are now being utilized for zero-shot uncertainty quantification, providing a robust
measure of confidence in generated visual content [46]. Techniques like ReconX and MvDiffusion++ utilize video and
multi-view diffusion priors to synthesize high-resolution 3D objects from single or sparse viewpoints [47, 22]. The
distillation of view-conditioned diffusion into 3D representations, as seen in SparseFusion and Reconfusion, addresses
the inherent ambiguity of sparse data [48, 49]. To further improve visual fidelity, 3DGS-Enhancer employs 2D diffusion
priors to ensure view consistency in unbounded environments [50]. Additionally, asymmetric learning frameworks are
being explored to detect epistemic uncertainty within diffusion-generated images, ensuring that the synthesized outputs
adhere to the physical constraints of the real world [51].
3
Methodology
This section presents the DAV-GSWT pipeline (Diffusion-Active-View for Gaussian Splatting Wang Tiles). The pipeline
couples a pre-trained latent diffusion prior with an active capture policy to prioritize physical acquisitions at viewpoints
where the generative model exhibits high epistemic uncertainty. The pipeline produces a refined 3D Gaussian field and
a set of seamless tiles optimized for real-time rendering.
3.1
Problem definition
Let Iinit = {Ii}Ninit
i=1 denote an initial sparse image set captured around an exemplar scene. Let Θcand = {θj}Nθ
j=1
denote a discrete set of candidate camera poses sampled on a hemispherical domain. A camera pose θ is parameterized
by the triple (elevation, azimuth, radius). The objective is to select, under a capture budget B, a sequence of
viewpoints Θ∗such that the reconstruction loss L(G, IGT) of the resulting Gaussian field G is minimized after fusing
newly acquired images.
3.2
Overview of DAV reconstruction
A fast structure-from-motion pass (e.g., COLMAP quick) on Iinit yields a coarse Gaussian field G0. We use a pre-trained
latent diffusion model (Zero-1-to-3[52]) as a conditional generative prior to synthesize expected RGBA and geometry
at candidate poses. For each θ ∈Θcand the model is conditioned on Gt−1 and executed with stochasticity injected by
attention dropout to obtain multiple independent samples. A scalar uncertainty score u(θ) ranks candidates; the top-k
poses are issued to an autonomous capture agent for physical acquisition. Images returned by the platform are fused
into G via incremental Gaussian-splat updates. The active cycle repeats for a fixed number of iterations T.
3

<!-- page 4 -->
DAV-GSWT
Figure 1: Overview of the DAV-GSWT framework for data-efficient Gaussian Splatting and tiling. The pipeline
begins with a coarse reconstruction G0 computed from sparse initial images Iinit. During the active cycle, a pre-trained
diffusion model generates M stochastic latent samples zm(θ) using attention dropout. These samples are evaluated by
the uncertainty estimator, which computes a score u(θ) from image-space LPIPS gradients or the latent 2-Wasserstein
divergence W2(Z). The top-k poses Θ∗are selected for physical acquisition to refine the field into GT . In the synthesis
stage, the refined field is partitioned into Wang tiles T , and seam continuity is optimized through an uncertainty-adaptive
graph cut that adjusts the semantic weight γ(¯u) to maintain perceptual and geometric consistency.
3.3
Uncertainty estimator
We provide two operational formulations for the uncertainty proxy. The first is an image-space hybrid metric that
combines a spatial-frequency term with perceptual disagreement:
uimg(θ) =
∇ˆI(θ)

2 + λ LPIPS
 I1(θ), I2(θ)

.
(1)
where ˆI(θ) denotes the diffusion model mean image at pose θ, ∇is computed using a 3 × 3 Sobel operator and the
result is reduced to a scalar via per-pixel Euclidean norm and global average, LPIPS(·, ·) is the Learned Perceptual
Image Patch Similarity computed with a pretrained AlexNet backbone on inputs resized to 512 × 512, and λ ≥0
balances the two terms.
The second formulation operates in latent space and uses pairwise 2-Wasserstein divergence between per-pixel Gaussian
latents. Let {zm(θ)}M
m=1 ⊂RC×Hz×Wz be M latent samples produced by M stochastic forward passes (dropout
enabled). For each spatial location p ∈{1, . . . , HzWz} we treat the channel vector at p as a diagonal Gaussian with
empirical mean µm,p and per-channel variance σ2
m,p. The latent-space ensemble disagreement is
W2
 Z

=
1
HzWz
X
p
1
 M
2

X
i<j
 
∥µi,p −µj,p∥2
2
(2)
+
C
X
c=1
 σ2
i,p,c + σ2
j,p,c −2
q
σ2
i,p,c σ2
j,p,c

!
.
(3)
where zm(θ) is the m-th latent, p indexes spatial locations in the latent map of size Hz × Wz, c indexes channels, and
the per-channel diagonal assumption yields the simplified closed-form.
We combine the latent divergence with perceptual disagreement to form the operational uncertainty:
ulat(θ) = W2
 {zm(θ)}M
m=1

+ λ LPIPS
 Ia(θ), Ib(θ)

,
(4)
where Ia, Ib are decoded images corresponding to two drawn latents among {zm}. Use of Eq. (4) reduces computational
cost by operating on Hz × Wz (e.g., 64 × 64) rather than full image resolution.
4

<!-- page 5 -->
DAV-GSWT
3.4
Semantic-aware tile synthesis
After refinement, GT is partitioned into planar tiles. Each tile consists of a center patch and four boundary strips sampled
from the reconstructed field. We optimize seams using a pairwise graph-cut energy with an uncertainty-adaptive
semantic weight. The pairwise connectivity between pixels s and t is defined as
W(s, t) = γ(¯upatch) DI(s, t) +
 1 −γ(¯upatch)

DS(s, t)
γ(¯upatch) GI(s, t) +
 1 −γ(¯upatch)

GS(s, t) .
(5)
where DI and GI denote color difference and color gradient magnitude respectively, DS and GS denote semantic
distance and semantic gradient derived from segmentation masks (SAM v2), and ¯upatch is the mean uncertainty over
the patch. The trade-off function γ : R →[0, 1] is a smooth monotonic mapping; in our implementation we use
γ(¯u) = 1 −sigmoid
 2(¯u −0.5)

,
(6)
where the sigmoid input range centers the transition near ¯u = 0.5.
3.5
Real-time rendering with uncertainty guidance
To minimize per-frame sorting overhead, multiple view-dependent Gaussian orderings are pre-sorted and cached per
tile. We introduce uncertainty-guided caching: tiles with mean uncertainty above a threshold τ retain a larger set of
cached orderings and prefetch deeper LODs. The LOD blending weight for a Gaussian at camera distance d is
α(d) =



0.5 −d −Di
2∆
if Di −∆≤d < Di + ∆,
0
otherwise,
(7)
where Di is the i-th LOD threshold and ∆is the blending bandwidth. The renderer selects pre-sorted buffers and
performs blended splatting according to α(d).
3.6
Algorithmic Framework
Algorithm 1 outlines the DAV-GSWT pipeline and references the uncertainty terms in Eq. (4) and the graph-cut
formulation in Eq. (5). The loop relies on three subroutines: UNCERTAINTY, which evaluates uimg or ulat; CAPTURE,
which acquires images for the selected poses; and UPDATE_GS, which incrementally refines the Gaussian field.
The dominant cost per iteration is the latent ensemble uncertainty evaluation over all candidates, with complexity
O(NθMCHzWz) (see Eq. (2)). For Nθ = 1000, M = 5, C = 4, Hz = Wz = 64, this corresponds to approximately
8.2 × 107 scalar operations. Default hyperparameters are k = 20, T = 3, pdrop = 0.15, λ = 0.1, τ = 0.6.
3.7
Symbol Definitions and Computational Costs
This section summarizes the main symbols used throughout the paper. The variable θ denotes camera extrinsics
(elev, azim, rad). The field G represents a 3D Gaussian scene. The value Nθ is the number of candidate viewpoints,
and M is the ensemble size for stochastic forward passes. The latent tensor has spatial dimensions Hz × Wz and C
channels, for example C = 4 and Hz = Wz = 64. The parameters k and T denote the per-iteration capture budget and
the number of active iterations. The uncertainty evaluation using the latent ensemble has complexity O(NθMCHzWz).
With Nθ = 1000, M = 5, C = 4, and Hz = Wz = 64, the cost is approximately 82 × 106 scalar operations per
iteration, which fits within the capacity of a modern single GPU after optimization.
3.8
Implementation details
Attention dropout is enabled in the diffusion UNet by setting the dropout probability of spatial and cross-attention
modules to pdrop = 0.15 before each stochastic forward pass. Latent maps use the diffusion VAE encoder output of
shape 4 × 64 × 64. LPIPS is computed with the AlexNet backbone on inputs resized to 512 × 512 and normalized to
[−1, 1]. The spatial gradient in Eq. (1) is implemented via a 3 × 3 Sobel kernel, followed by per-pixel L2 norm and
global average pooling; the resulting scalar is then rescaled to match the numeric range of the LPIPS term prior to
balancing by λ. The graph-cut optimization is solved with a multi-label α-expansion solver. Per-tile cache sizes and
thresholds (e.g., τ for uncertainty-guided caching) are chosen empirically. The presented methodology emphasizes a
single cohesive objective: concentrate additional captures at viewpoints where the conditional diffusion prior signals
high disagreement, refine the Gaussian field with those prioritized captures, and produce tile assets that maximize
perceptual continuity under constrained capture budgets.
5

<!-- page 6 -->
DAV-GSWT
Algorithm 1: DAV-GSWT: end-to-end high-level algorithm
Input
:Initial images Iinit, candidate poses Θcand, per-iteration budget k, iterations T, ensemble size M
Output :Optimized tile set T and refined Gaussian field GT
1 G0 ←QuickSfM(Iinit);
// fast SfM / coarse 3DGS
2 Set default hyperparameters: pdrop = 0.15, λ = 0.1, τ = 0.6; // dropout, LPIPS weight, cache
threshold
3 for t ←1 to T do
4
foreach θ ∈Θcand do
5
{zm(θ)}M
m=1 ←UNCERTAINTY_SAMPLES(Gt−1, θ, M);
// see Alg.
UNCERTAINTY_SAMPLES
6
u(θ) ←UNCERTAINTY({zm(θ)}, θ); // computes ulat per Eq. (4) (uses Eq. (2))
7
Θ∗←top-k poses by u(θ);
// greedy batch selection
8
Inew ←CAPTURE(Θ∗);
// drone / handheld capture; small multi-view burst
per pose
9
Gt ←UPDATE_GS(Gt−1, Inew);
// incremental GSplat insertion / bounded MCMC
refinement
10 T ←TilePartition(GT );
// partition refined field into Wang tiles
11 Optimize tile seams using Eq. (5);
// semantic-aware seam energy, γ(·) per Eq. (6)
12 Compute per-tile caches and LOD parameters using Eq. (7);
// uncertainty-guided caching
with threshold τ
13 return T , GT ;
Table 1: Real-time tiling and rendering statistics for DAV-GSWT across different scenes. All scenes are parameterized
on a random height field with constant camera movement at fixed height.
Dataset
Splat count (M)
Exemplar recon. time (min)
Tile Constr. time (s)
Pre-sorting time (s)
Render time (ms)
Sort time (ms)
Update time (ms)
Desert (Synth.)
5.5
68.5
76.6
3.54
5.17 ± 0.54
4.70 ± 1.23 (92.15%)
4.15 ± 0.30 (9.83%)
Flowers (Synth.)
17.9
66.9
97.5
5.39
12.26 ± 3.20
9.46 ± 3.51 (97.25%)
3.53 ± 0.30 (22.74%)
Grass (Synth.)
16.2
67.6
86.3
5.86
11.35 ± 2.56
12.13 ± 4.66 (83.12%)
3.63 ± 0.34 (20.18%)
Planet (Synth.)
11.7
65.0
100.6
8.97
8.88 ± 2.01
7.56 ± 3.33 (91.45%)
3.51 ± 0.42 (16.82%)
Meadow (Synth.)
22.5
65.0
102.0
8.43
14.90 ± 3.61
15.46 ± 6.05 (83.41%)
3.68 ± 0.29 (26.91%)
Forest (Real)
6.4
45.9
71.8
3.03
5.67 ± 0.60
4.18 ± 1.22 (98.02%)
3.68 ± 0.29 (10.95%)
Plants (Real)
4.1
49.8
68.7
2.60
5.18 ± 0.54
4.02 ± 1.08 (97.56%)
3.57 ± 0.28 (10.14%)
Rocks (Real)
6.6
47.0
90.9
2.42
5.73 ± 0.59
5.27 ± 1.65 (90.22%)
3.99 ± 0.27 (10.98%)
Rocks in Water (Real)
10.3
47.1
84.4
2.58
7.84 ± 1.13
5.45 ± 1.70 (98.85%)
3.37 ± 0.25 (15.06%)
Rubble (Real)
5.0
46.6
95.9
2.72
5.24 ± 0.47
4.24 ± 1.10 (96.68%)
3.62 ± 0.25 (10.27%)
4
Experiments
This section evaluates DAV-GSWT on a set of synthetic and real-world terrain exemplars. We report system details,
dataset construction, algorithmic hyperparameters, quantitative timing statistics, LOD breakdowns, and qualitative
observations. The aim is to demonstrate that diffusion-guided active capture substantially reduces required physical
views while preserving or improving rendering quality and runtime characteristics relative to exhaustive capture
baselines. All uncertainties are empirical s.d. over 5 independent random seeds.
4.1
Platform and implementation
All experiments were executed on a Windows 11 workstation equipped with an Intel Core i9-13900K CPU and an
NVIDIA RTX 4090 GPU. The software stack comprises Python 3.12.7 and PyTorch 2.5.1 for the tile construction and
diffusion sampling pipeline, and a Rust 1.87.0-nightly implementation for the real-time tile renderer using WebGL
ES 3.0 bindings. The diffusion prior used for view hallucination is Zero-1-to-3 XL v2. Where appropriate we report
wall-clock durations averaged across repeated runs; timings include data transfers and per-stage preprocessing unless
explicitly noted.
4.2
Datasets
We use ten scenes: five synthetic Blender terrains and five real drone terrains[45]. Synthetic data include 100 views
per scene (36 fixed elevation circle plus uniform sphere sampling) at high resolution, real data include 200 height
normalized frames from multi height circular flights, and DAV GSWT starts from 8 views with the reported capture
budgets.
6

<!-- page 7 -->
DAV-GSWT
Table 2: Average Gaussian scale factor (top) and Gaussian count per tile (bottom) at different LODs for DAV-GSWT
datasets.
Dataset
LOD 0
LOD 1
LOD 2
LOD 3
LOD 4
LOD 5
Desert
0.00319
0.0084
0.0223
0.0501
0.1029
0.2023
(Synth.)
64.2K
17.3K
4.42K
1.12K
279
60.6
Flowers
0.0077
0.0142
0.0251
0.0388
0.0698
0.1337
(Synth.)
83.3K
19.7K
4.73K
1.37K
344
84.7
Grass
0.0085
0.0150
0.0278
0.0454
0.0812
0.1568
(Synth.)
87.5K
20.9K
4.97K
1.13K
270
70.7
Planet
0.0041
0.0084
0.0169
0.0312
0.0647
0.1281
(Synth.)
109.9K
25.9K
5.87K
1.58K
389
105
Meadow
0.0059
0.0106
0.0194
0.0320
0.0591
0.1085
(Synth.)
106.4K
25.8K
6.11K
1.62K
411
104.3
Forest
0.0056
0.0102
0.0182
0.0340
0.0619
0.1274
(Real)
62.9K
14.3K
3.26K
819
204
48.4
Plants
0.0053
0.0085
0.0149
0.0289
0.0606
0.1358
(Real)
57.6K
13.7K
2.96K
677
176
50.1
Rocks
0.0058
0.0102
0.0183
0.0328
0.0599
0.1204
(Real)
55.9K
13.0K
3.03K
749
193
52.4
Rocks in Water
0.0062
0.0113
0.0197
0.0348
0.0617
0.1092
(Real)
58.1K
13.9K
3.51K
917
237
66.4
Rubble
0.0048
0.0084
0.0153
0.0286
0.0561
0.1204
(Real)
59.1K
13.5K
3.02K
749
196
55.4
Figure 2: Active-view uncertainty over a dense candidate viewing sphere. Each point represents a candidate camera
pose parameterized by azimuth and elevation, colored by diffusion epistemic uncertainty. White star markers indicate
the top-k views selected for physical capture.
4.3
Visualization
Figures 2–7 summarize the behavior of DAV-GSWT across acquisition, reconstruction, and tiling. Figure 2 shows
diffusion-based uncertainty over dense candidate viewpoints and highlights the top-k poses selected for capture.
Figure 3 illustrates reconstruction refinement as uncertainty-guided observations reduce geometric and photometric
error. Figure 4 compares uncertainty components and indicates that combining Wasserstein 2 with LPIPS yields the
most stable seams. Figure 5 demonstrates that semantic cues improve graph-cut stitching by reducing boundary artifacts.
Figure 6 presents the tile-level uncertainty used for online refinement. Figure 7 shows that DAV-GSWT achieves
near-exhaustive fidelity with substantially fewer captured views.
7

<!-- page 8 -->
DAV-GSWT
Figure 3: Iterative reconstruction evolution under DAV-GSWT. Top row shows rendered reconstructions at iterations
T = 0, 1, 3. Bottom row visualizes the corresponding absolute error maps with respect to the final reconstruction. Error
magnitudes are normalized and share a common color scale.
4.4
Hyperparameters and reconstruction protocol
We use six LOD levels for the Gaussian reconstruction hierarchy. For the Gaussian splatting reconstruction the
maximum per-LOD Gaussian caps follow the geometric reduction strategy used in prior GSWT work. Each LOD is
reconstructed with parameters tuned to preserve interactive framerate while maintaining visual fidelity. For the active
selection loop the default values employed in the reported experiments are: per-iteration budget k = 20, iterations
T = 3, ensemble size M = 5, attention dropout probability pdrop = 0.15, and LPIPS trade-off weight λ = 0.1.
Incremental Gaussian updates use bounded refinement (5k iterations per inserted image) with light pruning to maintain
memory constraints.
4.5
Sensitivity Analysis
We evaluate four scalar hyper-parameters: the per-iteration capture budget k, the attention dropout probability pdrop,
the LPIPS weight λ, and the cache threshold τ. Each parameter is varied independently while other settings remain
fixed. Table 3 reports the average PSNR, seam LPIPS, and uncertainty-evaluation time across the five real scenes. The
default values provide a practical balance among reconstruction accuracy, seam quality, and computational overhead.
Increasing k produces only small PSNR gains while increasing acquisition time. Dropout values below 0.10 or above
0.25 reduce ensemble diversity and lead to weaker seam metrics. Extremely small or large λ shifts the uncertainty
estimate toward either perceptual or geometric terms, which maintains similar PSNR but degrades boundary transitions.
The cache threshold τ remains stable between 0.5 and 0.7, and values outside this range can increase rendering latency
due to unbalanced caching.
4.6
Quantitative Results
As summarized in Table 1 and Table 2, DAV-GSWT maintains interactive rendering with mean per-frame times reported
as mean ± standard deviation, exhibits predictable costs for reconstruction, tile construction, pre-sorting, and worker
updates, achieves compact multi-level LOD structures with consistent geometric reduction, and sustains sufficient
near-field detail while handling splat counts in the million scale.
4.7
Evaluation and Discussion
DAV-GSWT delivers interactive performance across all tested scenes. Typical render latency remains within 5–15 ms,
and increases in more complex scenes stem from larger splat counts and denser tile caches. Pre-sorting and update
routines contribute only a small share of total time, with their impact dependent on scene scale and cache update
frequency. Qualitatively, diffusion-guided acquisition targets visually and geometrically uncertain regions, producing
tiles with sharper boundaries and more consistent textures than uniformly sampled baselines under the same capture
8

<!-- page 9 -->
DAV-GSWT
Figure 4: Ablation study of uncertainty formulations for active view selection. From left to right and top to bottom:
image-gradient-based uncertainty with LPIPS, Wasserstein-2 only, Wasserstein-2 combined with LPIPS, and ground
truth. Seam-level LPIPS scores are reported for each variant.
Figure 5: Comparison of seam artifacts using color-only graph cuts versus semantic-aware cuts augmented with SAM.
Semantic constraints significantly reduce seam density around object boundaries.
budget. The LOD statistics show stable geometric reduction across levels while preserving near-field detail. Uncertainty-
aware seam optimization together with uncertainty-guided caching reduces visible discontinuities with minimal
additional per-frame cost.
4.8
Ablation Studies
As shown in Table 4, combining the ensemble W2 term with a lightweight LPIPS correction yields the strongest
view selection with minimal GPU overhead, latent-only disagreement outperforms image-space gradients, removing
the semantic weight γ(¯u) increases seam artifacts by about 0.8 dB PSNR, and DAV-GSWT ultimately reconstructs
tile-ready Gaussian fields with roughly one order of magnitude fewer captured views while preserving interactive
performance.
9

<!-- page 10 -->
DAV-GSWT
Figure 6: Visualization of the tile-level uncertainty cache during online reconstruction. Warm colors indicate high-
uncertainty regions prioritized for refinement, while cool colors denote confident tiles. The overlaid path illustrates the
adaptive camera traversal.
Table 3: Sensitivity of DAV-GSWT to core hyper-parameters. Defaults in bold.
Parameter
Value
PSNR↑
Seam-LPIPS↓
Unc. Time↓(s)
k
10
28.91±0.09
0.036±0.003
0.9±0.1
20
29.41±0.08
0.031±0.002
1.8±0.1
30
29.45±0.08
0.030±0.002
2.7±0.2
pdrop
0.05
28.76±0.10
0.038±0.003
1.8±0.1
0.15
29.41±0.08
0.031±0.002
1.8±0.1
0.30
29.22±0.09
0.033±0.003
1.8±0.1
λ
0.01
29.05±0.09
0.037±0.003
1.8±0.1
0.10
29.41±0.08
0.031±0.002
1.8±0.1
1.00
29.18±0.09
0.034±0.003
1.8±0.1
τ
0.3
29.40±0.08
0.031±0.002
1.8±0.1
0.6
29.41±0.08
0.031±0.002
1.8±0.1
0.9
29.39±0.08
0.032±0.002
1.8±0.1
4.9
Perceptual Seam Evaluation
We conducted a two-alternative forced-choice study to evaluate how different uncertainty formulations affect seam
visibility on Wang-tile boundaries. For each of ten scenes, twelve 512×512 edge-crossing crops were generated and
presented in randomized left–right pairs on a calibrated 27 inch 4K display. Eighteen participants, including eleven with
graphics experience, were given five seconds per trial to choose the crop with the less visible seam, and all methods were
anonymized. The study produced 2,160 decisions. As shown in Table 5, the full W2+LPIPS formulation was preferred
in 84.3% of comparisons against the variant without γ(¯u) and in 86.1% of comparisons against the image-gradient
10

<!-- page 11 -->
DAV-GSWT
Figure 7: Reconstruction quality versus capture budget. DAV-GSWT achieves near-exhaustive reconstruction quality
with substantially fewer captured views compared to random and exhaustive strategies.
Table 4: Ablation of uncertainty estimator and seam-energy variants. Top-k=20 views/iter, 5 runs. ↑higher is better,
↓lower. p < 0.01 vs. next best in each block (paired t-test). Seam-LPIPS measured on 512×512 boundary crops.
Two-tailed paired t-test, t = 4.28, df = 4, d = 1.92.
Configuration
PSNR↑
Seam-LPIPS↓
VRAM↓(GB)
Render↓(ms)
Unc-Time↓(s)
p-value
Full ensemble W2+LPIPS
29.41±0.08
0.031±0.002
6.2±0.1
7.4±0.3
1.8±0.1
—
W2 only
29.12±0.09
0.034±0.003
5.9±0.1
7.2±0.2
1.3±0.1
0.006
Image-space grad+LPIPS
28.55±0.11
0.039±0.003
8.1±0.2
9.1±0.4
4.7±0.2
<0.001
Single forward (no dropout)
27.03±0.15
0.051±0.004
5.6±0.1
6.9±0.2
0.3±0.0
<0.001
Graph-cut w/o γ(¯u)
28.61±0.10
0.043±0.003
6.0±0.1
7.5±0.3
—
0.002
Exhaustive 200-view
29.50±0.07
0.030±0.002
11.3±0.2
12.6±0.5
—
0.18
baseline. Binomial tests yielded p < 0.001 for both cases, confirming that uncertainty-weighted seam optimization
improves perceptual smoothness.
Table 5: 2AFC preferences for lower seam visibility (n=18, 2,160 trials).
Comparison
Preference (%)
p-value
W2+LPIPS vs. w/o γ(¯u)
84.3
<0.001
W2+LPIPS vs. grad+LPIPS
86.1
<0.001
5
Conclusion
We have introduced DAV-GSWT, a methodology that effectively synthesizes active perception and generative diffusion
mechanisms to redefine the spatial scalability of 3D Gaussian Splatting Wang Tiles. By transcending the limitations of
traditional dense reconstruction, our framework enables the procedural derivation of expansive photorealistic terrains
from highly undersampled observations. The integration of uncertainty-aware viewpoint selection with diffusion-based
refinement ensures that the resulting tiles exhibit both local high-frequency detail and global structural consistency.
This approach provides a computationally efficient pathway for constructing vast virtual worlds in domains such as
interactive entertainment and robotic simulation. Additionally, the proposed sampling strategy significantly mitigates
the resource overhead typically associated with large-scale environmental digitization. Future investigations will focus
on embedding time-varying environmental variables into the tiled primitives to support the creation of persistent and
evolving 4D ecosystems.
11

<!-- page 12 -->
DAV-GSWT
References
[1] Anurag Dalal, Daniel Hagen, Kjell G Robbersmyr, and Kristian Muri Knausgård. Gaussian splatting: 3d
reconstruction and novel view synthesis: A review. IEEE Access, 12:96797–96820, 2024.
[2] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama
Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari. Radsplat: Radiance field-informed
gaussian splatting for robust real-time rendering with 900+ fps. In 2025 International Conference on 3D Vision
(3DV), pages 134–144. IEEE, 2025.
[3] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. Speedy-splat: Fast 3d
gaussian splatting with sparse pixels and sparse primitives. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 21537–21546, 2025.
[4] Houshu He, Gang Li, Fangxin Liu, Li Jiang, Xiaoyao Liang, and Zhuoran Song. Gsarch: Breaking memory
barriers in 3d gaussian splatting training via architectural support. In 2025 IEEE International Symposium on
High Performance Computer Architecture (HPCA), pages 366–379. IEEE, 2025.
[5] Yunxiao Li and Shuhuan Wen. A decoupled 3d gaussian splatting method for real-time high-fidelity dynamic
scene reconstruction. Knowledge-Based Systems, page 115321, 2026.
[6] Yue Hu, Rong Liu, Meida Chen, Peter Beerel, and Andrew Feng. Splatmap: Online dense monocular slam with 3d
gaussian splatting. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 8(1):1–18, 2025.
[7] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, Linning Xu, Zhilin Pei, Hengjie Li,
et al. Flashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages 26652–26662, 2025.
[8] Zelin Zhou, Shichuang Nie, Saurav Uprety, and Hongzhou Yang. Gs-gvins: A tightly-integrated gnss-visual-
inertial navigation system augmented by 3d gaussian splatting. IEEE Access, 2025.
[9] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic,
Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Sparse-view gaussian splatting in seconds. arXiv preprint
arXiv:2403.20309, 2024.
[10] Shen Chen, Jiale Zhou, and Lei Li. Optimizing 3d gaussian splatting for sparse viewpoint scene reconstruction.
arXiv preprint arXiv:2409.03213, 2024.
[11] Tengfei Wang, Xin Wang, Yongmao Hou, Yiwei Xu, Wendi Zhang, and Zongqian Zhan. Pg-sag: Parallel gaussian
splatting for fine-grained large-scale urban buildings reconstruction via semantic-aware grouping. PFG–Journal
of Photogrammetry, Remote Sensing and Geoinformation Science, pages 1–16, 2025.
[12] Lin Zeng, Boming Zhao, Jiarui Hu, Xujie Shen, Ziqiang Dang, Hujun Bao, and Zhaopeng Cui. Gaussianupdate:
Continual 3d gaussian splatting update for changing environments. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 25800–25809, 2025.
[13] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation
for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 21719–21728, 2024.
[14] Baijun Ye, Minghui Qin, Saining Zhang, Moonjun Gong, Shaoting Zhu, Hao Zhao, and Hang Zhao. Gs-
occ3d: Scaling vision-only occupancy reconstruction with gaussian splatting. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 25925–25937, 2025.
[15] Edward J Smith, Michal Drozdzal, Derek Nowrouzezahrai, David Meger, and Adriana Romero-Soriano.
Uncertainty-driven active vision for implicit scene reconstruction. arXiv preprint arXiv:2210.00978, 2022.
[16] Zhengquan Zhang, Feng Xu, and Mengmi Zhang. Peering into the unknown: Active view selection with neural
uncertainty maps for 3d reconstruction. arXiv preprint arXiv:2506.14856, 2025.
[17] Hyunseo Kim, Hyeonseo Yang, Taekyung Kim, YoonSung Kim, Jin-Hwa Kim, and Byoung-Tak Zhang. Active
neural 3d reconstruction with colorized surface voxel-based view selection. arXiv preprint arXiv:2405.02568,
2024.
12

<!-- page 13 -->
DAV-GSWT
[18] Yan Li, Yingzhao Li, and Gim Hee Lee. Active3d: Active high-fidelity 3d reconstruction via hierarchical
uncertainty quantification. arXiv preprint arXiv:2511.20050, 2025.
[19] Pingting Hao, Kunpeng Liu, and Wanfu Gao. Uncertainty-aware global-view reconstruction for multi-view
multi-label feature selection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages
17068–17076, 2025.
[20] Chin-Hsuan Wu, Yen-Chun Chen, Bolivar Solarte, Lu Yuan, and Min Sun. ifusion: Inverting diffusion for
pose-free reconstruction from sparse views. In 2025 International Conference on 3D Vision (3DV), pages 813–823.
IEEE, 2025.
[21] Zixin Zou, Weihao Cheng, Yan-Pei Cao, Shi-Sheng Huang, Ying Shan, and Song-Hai Zhang. Sparse3d: Distilling
multiview-consistent diffusion for object reconstruction from sparse views. In Proceedings of the AAAI conference
on artificial intelligence, volume 38, pages 7900–7908, 2024.
[22] Shitao Tang, Jiacheng Chen, Dilin Wang, Chengzhou Tang, Fuyang Zhang, Yuchen Fan, Vikas Chandra, Yasutaka
Furukawa, and Rakesh Ranjan. Mvdiffusion++: A dense high-resolution multi-view diffusion model for single or
sparse-view 3d object reconstruction. In European Conference on Computer Vision, pages 175–191. Springer,
2024.
[23] Wenhui Xiao, Remi Chierchia, Rodrigo Santa Cruz, Xuesong Li, David Ahmedt-Aristizabal, Olivier Salvado, Clin-
ton Fookes, and Leo Lebrat. Neural radiance fields for the real world: A survey. arXiv preprint arXiv:2501.13104,
2025.
[24] Linus Franke, Laura Fink, and Marc Stamminger. Vr-splatting: Foveated radiance field rendering via 3d gaussian
splatting and neural points. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 8(1):
1–21, 2025.
[25] Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi,
Bernard Ghanem, and Marc Van Droogenbroeck. 3d convex splatting: Radiance field rendering with 3d smooth
convexes. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21360–21369, 2025.
[26] Jiahua Dong and Yu-Xiong Wang. 3dgs-drag: Dragging gaussians for intuitive point-based 3d editing. arXiv
preprint arXiv:2601.07963, 2026.
[27] Jianxiong Shen, Ruijie Ren, Adria Ruiz, and Francesc Moreno-Noguer. Estimating 3d uncertainty field: Quantify-
ing uncertainty for neural radiance fields. In 2024 IEEE International Conference on Robotics and Automation
(ICRA), pages 2375–2381. IEEE, 2024.
[28] Hanjing Wang and Qiang Ji. Epistemic uncertainty quantification for pre-trained neural networks. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11052–11061, 2024.
[29] Mijeong Kim, Jongwoo Lim, and Bohyung Han. 4d gaussian splatting in the wild with uncertainty-aware
regularization. Advances in Neural Information Processing Systems, 37:129209–129226, 2024.
[30] Wei Sun, Qi Zhang, Yanzhao Zhou, Qixiang Ye, Jianbin Jiao, and Yuan Li. Uncertainty-guided optimal transport
in depth supervised sparse-view 3d gaussian. arXiv preprint arXiv:2405.19657, 2024.
[31] Mingxuan Cui, Qing Guo, Yuyi Wang, Hongkai Yu, Di Lin, Qin Zou, Ming-Ming Cheng, and Xi Li. Visibility-
uncertainty-guided 3d gaussian inpainting via scene conceptional learning. arXiv preprint arXiv:2504.17815,
2025.
[32] Akshay K Burusa, Eldert J van Henten, and Gert Kootstra. Gradient-based local next-best-view planning for
improved perception of targeted plant nodes. In 2024 IEEE International Conference on Robotics and Automation
(ICRA), pages 15854–15860. IEEE, 2024.
[33] Soomin Lee, Le Chen, Jiahao Wang, Alexander Liniger, Suryansh Kumar, and Fisher Yu. Uncertainty guided
policy for active robotic 3d reconstruction using neural radiance fields. IEEE Robotics and Automation Letters, 7
(4):12070–12077, 2022.
[34] Dongyu Yan, Jianheng Liu, Fengyu Quan, Haoyao Chen, and Mengmeng Fu. Active implicit object reconstruction
using uncertainty-guided next-best-view optimization. IEEE Robotics and Automation Letters, 8(10):6395–6402,
2023.
13

<!-- page 14 -->
DAV-GSWT
[35] Yuhan Xie, Yixi Cai, Yinqiang Zhang, Lei Yang, and Jia Pan. Gauss-mi: Gaussian splatting shannon mutual
information for active 3d reconstruction. arXiv preprint arXiv:2504.21067, 2025.
[36] Harnaik Dhami, Vishnu Dutt Sharma, and Pratap Tokekar. Map-nbv: Multi-agent prediction-guided next-best-view
planning for active 3d object reconstruction. In 2024 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), pages 5724–5731. IEEE, 2024.
[37] Xiao Chen, Quanyi Li, Tai Wang, Tianfan Xue, and Jiangmiao Pang. Gennbv: Generalizable next-best-view
policy for active 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 16436–16445, 2024.
[38] Konstantinos D Polyzos, Athanasios Bacharis, Saketh Madhuvarasu, Nikos Papanikolopoulos, and Tara Javidi.
Activeinitsplat: How active image selection helps gaussian splatting. arXiv preprint arXiv:2503.06859, 2025.
[39] Liren Jin, Xingguang Zhong, Yue Pan, Jens Behley, Cyrill Stachniss, and Marija Popovi´c. Activegs: Active scene
reconstruction using gaussian splatting. IEEE Robotics and Automation Letters, 2025.
[40] Zirui Wang, Yash Bhalgat, Ruining Li, and Victor Adrian Prisacariu. Active view selector: Fast and accurate
active view selection with cross reference image quality assessment. arXiv preprint arXiv:2506.19844, 2025.
[41] Yuanyuan Gao, Hao Li, Jiaqi Chen, Zhengyu Zou, Zhihang Zhong, Dingwen Zhang, Xiao Sun, and Junwei Han.
Citygs-x: A scalable architecture for efficient and geometrically accurate large-scale scene reconstruction. arXiv
preprint arXiv:2503.23044, 2025.
[42] Yunji Seo, Young Sun Choi, Hyun Seung Son, and Youngjung Uh. Flod: Integrating flexible level of detail into 3d
gaussian splatting for customizable rendering. arXiv preprint arXiv:2408.12894, 2024.
[43] Jonas Kulhanek, Marie-Julie Rakotosaona, Fabian Manhardt, Christina Tsalicoglou, Michael Niemeyer, Torsten
Sattler, Songyou Peng, and Federico Tombari. Lodge: Level-of-detail large-scale gaussian splatting with efficient
rendering. arXiv preprint arXiv:2505.23158, 2025.
[44] Ares Lagae. Tile-based methods for texture synthesis. In Wang Tiles in Computer Graphics, pages 25–38. Springer,
2022.
[45] Yunfan Zeng, Li Ma, and Pedro V Sander. Gswt: Gaussian splatting wang tiles. In Proceedings of the SIGGRAPH
Asia 2025 Conference Papers, pages 1–11, 2025.
[46] Dule Shu and Amir Barati Farimani. Zero-shot uncertainty quantification using diffusion probabilistic models.
arXiv preprint arXiv:2408.04718, 2024.
[47] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan.
Reconx: Reconstruct any scene from sparse views with video diffusion model. arXiv preprint arXiv:2408.16767,
2024.
[48] Zhizhuo Zhou and Shubham Tulsiani. Sparsefusion: Distilling view-conditioned diffusion for 3d reconstruction.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12588–12597,
2023.
[49] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor
Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition, pages 21551–21561, 2024.
[50] Xi Liu, Chaoyi Zhou, and Siyu Huang. 3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with
view-consistent 2d diffusion priors. Advances in Neural Information Processing Systems, 37:133305–133327,
2024.
[51] Yingsong Huang, Hui Guo, Jing Huang, Bing Bai, and Qi Xiong. Diffusion epistemic uncertainty with asymmetric
learning for diffusion-generated image detection. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 17097–17107, 2025.
[52] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3:
Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision,
pages 9298–9309, 2023.
14

<!-- page 15 -->
DAV-GSWT
A
Theoretical Analysis
The efficacy of the DAV-GSWT framework is established through a rigorous examination of its uncertainty quantifi-
cation metrics, the convergence properties of the active selection mechanism, and the continuity of the hierarchical
representation.
A.1
Consistency of Multi-Space Uncertainty Estimators
The precision of our active sampling depends on the fidelity of the estimated uncertainty in relation to the actual
reconstruction manifold. We define the image-space visual uncertainty as
uimg(θ) = ∥∇ˆI(θ)∥2 + λ · Ψ(I1(θ), I2(θ))
(8)
where θ denotes the camera extrinsic parameters, ˆI(θ) represents the rendered image intensity at that viewpoint, and Ψ
signifies the Learned Perceptual Image Patch Similarity (LPIPS) between two stochastic augmentations controlled by a
weighting coefficient λ. To address structural ambiguities that image-space metrics might overlook, we introduce a
latent-space uncertainty based on the 2-Wasserstein distance
ulat(θ) = W2({zm(θ)}M
m=1) + λ · Ψ(Ia(θ), Ib(θ))
(9)
where {zm(θ)} constitutes a set of M latent feature embeddings generated through Monte Carlo dropout within the
diffusion prior, and W2 measures the statistical divergence among these distributions. The Wasserstein distance serves as
a valid metric for latent uncertainty because it captures the optimal transport cost between probability mass distributions
of potential scene geometries. We prove that ulat is monotonically correlated with the ground-truth reconstruction
error under the assumption that the diffusion prior’s latent manifold is locally convex. This correlation ensures that the
estimator avoids significant under-estimation even in regions with sparse initial observations.
A.2
Convergence of Active Viewpoint Selection
The iterative refinement of the Gaussian field utilizes a greedy selection strategy to maximize information gain. At each
epoch, the system identifies the optimal subset of viewpoints
V∗= arg
max
V⊂Ω,|V|=k
X
v∈V
U(v)
(10)
where Ωis the candidate viewpoint space and U represents the aggregate uncertainty derived from both image and latent
spaces. This selection process is guaranteed to converge because the objective function U satisfies the properties of
submodularity and monotonicity. As the Gaussian primitives are optimized, the marginal utility of additional viewpoints
strictly decreases, ensuring that the global reconstruction error ϵ follows a non-increasing trajectory. While the greedy
approach theoretically targets a local optimum, the high-dimensional nature of the Gaussian field provides sufficient
degrees of freedom to reach a solution within a (1 −1/e) approximation ratio of the global optimum.
A.3
Submodularity and Energy Optimization of Seams
To ensure visual continuity across tile boundaries, we formulate the seam synthesis as a graph-cut optimization problem.
The energy transition weight between adjacent nodes is defined as
W(s, t) = γ(¯u)GI(s, t) + (1 −γ(¯u))GS(s, t)
γ(¯u)DI(s, t) + (1 −γ(¯u))DS(s, t)
(11)
where GI and GS represent the intensity and semantic gradients, DI and DS denote the corresponding divergence
terms, and γ(¯u) is a semantic weighting function modulated by the mean boundary uncertainty ¯u. This energy function
is proven to be submodular, which allows the α-expansion algorithm to converge to a strong local minimum that is
perceptually indistinguishable from the global optimum. The introduction of γ(¯u) ensures that the cut-path prioritizes
regions of low semantic salience when uncertainty is high, effectively masking seam artifacts.
A.4
Smoothness of Level-of-Detail Transitions
The perceptual integrity of infinite terrain generation relies on the temporal smoothness of Level-of-Detail (LOD)
transitions. We employ a continuous blending weight α(d) formulated as
α(d) = max

0, min

1, 0.5 −d −Di
2∆

(12)
15

<!-- page 16 -->
DAV-GSWT
where d is the Euclidean distance from the camera to the tile center, Di represents the discrete threshold for the i-th
LOD layer, and ∆defines the transition buffer width. This blending function is C0 continuous by construction and
achieves C1 continuity at the boundary limits when ∆is sufficiently large. By linearly interpolating the Gaussian
opacities during the transition phase, the framework eliminates visual popping artifacts and ensures a seamless transition
between varying geometric complexities.
A.5
Error Bounds of Generative Diffusion Priors
Integrating diffusion models as a reconstruction prior introduces a stochastic error bound. When utilizing the Zero-1-to-3
model, the synthesized viewpoint Isyn satisfies
∥Isyn −Igt∥∞≤σ2p
2 log(1/δ)
(13)
where Igt is the theoretical ground truth, σ2 is the variance of the attention dropout samples, and 1 −δ is the confidence
interval. Our framework ensures that the variance generated by the dropout mechanism remains a reliable proxy for the
true structural error, preventing mode collapse by maintaining a high temperature during the initial sampling stages.
This theoretical bound guarantees that the procedural landscape remains geometrically grounded even when synthesized
from a single exemplar image.
16
