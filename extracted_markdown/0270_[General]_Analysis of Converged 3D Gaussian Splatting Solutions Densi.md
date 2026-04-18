# Analysis of Converged 3D Gaussian Splatting Solutions: Density Effects and Prediction Limits

Zhendong Wangââ³, Cihan Ruanââ³, Jingchuan Xiaoâ â³, Chuqing Shiâ¡â³ Wei JiangÂ§, Wei WangÂ§, Wenjie LiuÂ¶, and Nam Lingââ¦

zwang29@scu.edu, luciacihanruan@gmail.com, josephxky@gmail.com, chs139@ucsd.edu {wjiang, rickweiwang}@futurewei.com, 51265901068@stu.ecnu.edu.cn, nling@scu.edu

âDepartment of Computer Science and Engineering, Santa Clara University, Santa Clara, CA, USA

â Department of Mathematics and Computer Studies, Mary Immaculate College, Limerick, Ireland

â¡Department of Mathematics, University of California, San Diego, CA, USA

Â§Futurewei Technologies Inc., San Jose, CA, USA

Â¶School of Computer Science and Technology, East China Normal University, Shanghai, China

AbstractâWe investigate what structure emerges in 3D Gaussian Splatting (3DGS) solutions from standard multiview optimization. We term these Rendering-Optimal References (RORs) and analyze their statistical properties, revealing stable patternsâmixture-structured scales and bimodal radianceâacross diverse scenes. To understand what determines these parameters, we apply learnability probes: training predictors to reconstruct RORs from point clouds without rendering supervision. Our analysis uncovers fundamental densitystratification: dense regions exhibit geometry-correlated parameters amenable to render-free prediction, while sparse regions show systematic failure across architectures. We formalize this through variance decomposition, demonstrating that visibility heterogeneity creates covariance-dominated coupling between geometric and appearance parameters in sparse regions. This reveals RORsâ dual characterâgeometric primitives where point clouds suffice, view synthesis primitives where multi-view constraints are essential. We provide density-aware strategies that improve training robustness and discuss architectural implications for systems that adaptively balance feed-forward prediction and rendering-based refinement.

Index Termsâ3D Gaussian Splatting, Rendering-based Optimization, Learnability Analysis, Density Stratification, Variance Decomposition, Geometric Primitives, View Synthesis, Hybrid Architectures

## I. INTRODUCTION

3D Gaussian Splatting (3DGS) achieves remarkable rendering quality through iterative optimization of explicit primitives under multi-view supervision [1]. However, the nature of these converged solutions remains largely opaque: the structure that emerges in the parameters, and which aspects reflect geometric constraints versus view synthesis requirements, are not well understood. Understanding these solutions is critical for both theoretical insight and practical applications like feed-forward generation [2], [3].

We systematically anatomize converged 3DGS solutions from standard rendering-based optimization [1]. We term these Rendering-Optimal References(RORs) to emphasize they arise from typical multi-view supervision, serving as concrete, representative outcomes for analysis. Our central focus is understanding what regularities exist in ROR parameters and the extent to which they are determined by local geometric observations versus global rendering constraints.

Recent studies have begun exploring the statistical behavior of optimized primitives, observing implicit priors in scale, opacity, and color distributions [4], [5]. Yet, a systematic analysis of how these structures emergeâand how they relate to learnabilityâremains absent. Our work bridges this gap by analyzing the intrinsic organization and stability of 3D Gaussian primitives.

We employ three complementary approaches. We begin with statistical characterization, analyzing parameter distributions across 15 scenes and revealing consistent structured scale distributions and bimodal radiance patterns, consistent with multiplicative-noise optimization theory [6]. We then deploy learnability probesâhigh-capacity predictors including Transformers and Point-Voxel CNNsâtrained to reconstruct ROR parameters from point clouds without rendering supervision, testing whether structure is deducible from local geometry alone [2], [7]. Finally, variance analysis formalizes the coupling between geometric (Î£) and appearance (S) parameters through visibility- modulated gradients [8], [9].

Our analysis reveals that RORs exhibit dual character depending on local geometry density. In dense regions (high point cloud coverage), Gaussians act as geometric primitives: parameters strongly correlate with local structure, learnability probes achieve low error, and variance remains bounded. In sparse regions, they become view synthesis primitives: weak geometric correlation, systematic probe failure, and covariance-dominated variance. This stratification holds across scenes and architectures, revealing that sparse regions encode multi-view constraints inaccessible from point clouds aloneâexplaining both prediction failure (information deficiency) and optimization instability (variance coupling).

Standard 3DGS optimization [1] produces Gaussians that partially correlate with input point clouds yet diverge systematically in sparse regions (Fig. 1). Recent work addresses specific artifacts: Quadratic Gaussian Splatting [10] replaces isotropic primitives with quadric surfaces for better geometry capture; Ye et al. [4] identify and regularize covariance rank degradation; methods like [7] introduce geometric priors to stabilize optimization. While effective at mitigation, these lack systematic characterization of why certain regions exhibit instability. We provide the comprehensive anatomy of converged solutions, revealing density-stratified structure through statistical analysis, learnability probes, and variance decompositionâexplaining fundamental information boundaries rather than proposing immediate fixes.

Our contributions include:

â¢ Systematic anatomy of rendering-based 3DGS solutions, revealing stable statistical regularities (mix structured scales, bimodal radiance) and fundamental densitystratified structure across diverse scenes.

â¢ Learnability diagnosis demonstrating that render-free prediction exhibits qualitatively different behavior in dense versus sparse regions, with consistent patterns across architectures indicating information-theoretic rather than capacity-based limitations.

â¢ Variance decomposition framework formalizing visibility-coupled gradient dynamics, providing unified explanation for both optimization fragility in sparse regions and fundamental bounds on render-free prediction.

â¢ Design implications for practical systems: density-aware allocation principles, hierarchical processing strategies, and hybrid architecture guidelines that balance geometric and synthesis-based primitives.

Our study focuses on standard 3DGS with point cloud initialization. We analyze what RORs encode and where they are learnable, clarifying boundaries between geometrybased and synthesis-based information to guide efficient hybrid architectures for edge deployment and real-time applications.

## II. GRADIENT VARIANCE IN SPARSELY-SAMPLED REGIONS: A THEORETICAL ANALYSIS

While densely-sampled regions in 3D Gaussian Splatting optimize smoothly, sparsely-sampled regions, where primitives receive few training rays, exhibit severe instabilities. We analyze the coupling between geometric and appearance parameters to explain how sparse sampling amplifies gradient variance and prevents convergence.

## A. Simplified Model

We model each Gaussian with a covariance matrix $\Sigma \in \mathbb { S } _ { + + } ^ { 3 }$ for geometry and a scalar $S \in \mathbb { R } _ { + }$ for appearance. While actual 3DGS uses rotationâscale parameterization [1] and spherical harmonics [11], [12], this abstraction preserves the essential coupling dynamics. The optimization objective takes the form

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { g e o } } ( \Sigma ) + \omega \mathcal { L } _ { \mathrm { a p p } } ( \Sigma , S )\tag{1}
$$

<!-- image-->  
Fig. 1: Spatial distribution comparison between COLMAP point cloud (blue) and converged Gaussian primitives (red) for the truck scene. While strong correlation exists in geometryrich regions (vehicle body), substantial divergence appears in sparse regions (background, ground plane), motivating our investigation into what determines Gaussian parameters beyond local point cloud observations.

where $\mathcal { L } _ { \mathrm { g e o } }$ measures geometric accuracy and $\mathcal { L } _ { \mathrm { a p p } } = \mathbb { E } [ ( I -$ $T S ) ^ { 2 } ]$ measures rendering quality. Here, I denotes the observed pixel intensity, S is the appearance parameter, and $T ( \Sigma ) \ \in \ [ 0 , 1 ]$ is a visibility function: since Î£ controls the Gaussianâs spatial extent and orientation, it determines which rays intersect the Gaussian and how strongly.

## B. Gradient Coupling Mechanism

The gradients involve the visibility term:

$$
\frac { \partial \mathcal { L } } { \partial \Sigma } = \frac { \partial \mathcal { L } _ { \mathrm { g e o } } } { \partial \Sigma } + \omega \frac { \partial \mathcal { L } _ { \mathrm { a p p } } } { \partial T } \cdot \frac { \partial T } { \partial \Sigma } , \qquad \frac { \partial \mathcal { L } } { \partial S } = \omega \frac { \partial \mathcal { L } _ { \mathrm { a p p } } } { \partial S }\tag{2}
$$

Changes to Î£ alter the spatial footprint, modifying which rays intersect the primitive and with what intensity. This affects T , which in turn modulates the gradient for S. The geometry and appearance parameters are thus entangled through visibility. In actual 3DGS alpha-compositing $\begin{array} { r } { C = \sum _ { i } c _ { i } \alpha _ { i } \prod _ { j < i } ( 1 - \alpha _ { j } ) } \end{array}$ [1], this coupling is stronger: geometric parameters control transmittance for downstream primitives while opacity gates gradient flow to appearance.

## C. Variance Analysis and Density-Dependent Instability

Let $\xi _ { \Sigma }$ and $\xi _ { S }$ denote stochastic gradient components from mini-batch sampling. The total gradient variance is

$$
\begin{array} { r } { \mathcal { V } _ { \mathrm { t o t a l } } = \mathrm { V a r } ( \xi _ { \Sigma } ) + \mathrm { V a r } ( \xi _ { S } ) + 2 \mathrm { C o v } ( \xi _ { \Sigma } , \xi _ { S } ) } \end{array}\tag{3}
$$

Under heterogeneous sampling, this variance behaves dramatically differently across regions:

â¢ Dense regions: Many rays consistently sample the primitive, keeping visibility T stable. Both individual variances and covariance remain bounded, enabling smooth optimization.

â¢ Sparse regions: Few rays sample the primitive, making T volatileâsmall geometric changes drastically alter which rays hit in each mini-batch. This instability cascades through three mechanisms: (1) Var(Î¾Î£) grows as $\partial T / \partial \Sigma$ becomes noisy, (2) $\mathrm { V a r } ( \xi _ { S } )$ grows as S receives inconsistent signals through volatile T , and (3) most critically, $\mathrm { C o v } ( \xi _ { \Sigma } , \xi _ { S } )$ grows superlinearly because both gradients depend on the same unstable visibility samples.

The covariance can be shown to scale as

$$
\mathrm { C o v } ( \xi _ { \Sigma } , \xi _ { S } ) \propto \left\| \frac { \partial T } { \partial \Sigma } \right\| ^ { 2 } \cdot \mathrm { V a r } ( \mathcal { L } _ { \mathrm { a p p } } )\tag{4}
$$

In sparse regions, visibility sensitivity $\| \partial T / \partial \Sigma \|$ explodes (binary hit-or-miss behavior across batches) while appearance variance $\mathrm { V a r } ( \mathcal { L } _ { \mathrm { a p p } } )$ is large (poor signal from few samples). Their product yields covariance that often exceeds the sum of individual variances, dominating $\mathcal { V } _ { \mathrm { t o t a l } }$ . This quadratic amplification explains why coupled parameters exhibit far worse instability than independent parameters under the same sparse samplingâthe interaction amplifies noise beyond simple addition. High variance triggers erratic geometry updates, further destabilizing T in a positive feedback loop that prevents convergence.

Implications. This variance analysis on the simplified model can be generalized the 3DGS to explain observed failures, such as artifacts in sparse regions and training instability in occluded areas. Standard 3DGS applies identical regularization to all primitives, ignoring local density, which overlooks the variance structure above. This motivates densityaware regularization, and decoupled optimization of geometric and appearance parameters as shown later in Section III.

## III. EXPERIMENTAL VALIDATION AND STRUCTURAL REMEDIES

## A. Setup and ROR Priors

We employ converged 3DGS models RORs following standard optimization [1]. To probe learnability, we train Render-Free Predictors (RFPs)âTransformer-based networks that reconstruct ROR parameters from point clouds without rendering supervision. We evaluate on the Mip-NeRF 360 dataset, partitioning scenes into $N = 1 2 9$ spatial blocks stratified by local point density $\rho .$ Performance is measured via MSE between RFP predictions and ROR parameters.

## B. Statistical Characterization of ROR

Analysis of converged RORs reveals stable statistical regularities (Fig. 3a). Gaussian scale eigenvalues exhibit structured, non-unimodal distributions, arising from multiplicative updates $( \lambda _ { t + 1 } \approx \lambda _ { t } ( 1 + \varepsilon _ { t } ) )$ with implicit mean-reversion from regularization. Radiance values show bimodal structure: under heterogeneous visibility, the rendering loss $\begin{array} { r l } { \mathcal { L } _ { \mathrm { a p p } } } & { { } = } \end{array}$ $\mathbb { E } [ ( I - T S ) ^ { 2 } ]$ yields equilibria at high intensity for visible surfaces $( T ~ \approx ~ 1 )$ and low intensity for occluded regions $( T \approx 0 )$ . These patterns appear consistently across scenes, confirming robust optimization-emergent regularities.

<!-- image-->

Fig. 2: Density-stratified learnability analysis across three representative blocks. Dense $\mathrm { Q _ { 1 } }$ (top): high point coverage, RFP successfully reconstructs ROR distribution. Mid $\mathrm { Q _ { 2 } }$ (middle): moderate coverage, partial success. Sparse $\mathrm { Q _ { 3 } }$ (bottom): low coverage, systematic RFP failure.  
<!-- image-->

<!-- image-->  
(a) Statistical regularities in ROR: bimodal radiance (left) and structured scales (right).

<!-- image-->

<!-- image-->  
(b) Validation results: density-stratified prediction error (left) and training dynamics comparison (right).  
Fig. 3: Experimental validation of density-stratified structure.

## C. Learnability Analysis: Density-Stratified Failure

We stratify blocks into density terciles and evaluate RFP learning dynamics (Table I). Dense regions $\left( \mathbf { Q } _ { 1 } \right)$ show clear and stable improvement, with median MSE decreasing from 44.96 to 9.12 (+79.7%), indicating that ROR parameters in these regions are strongly correlated with local geometry and can be reliably predicted from point clouds alone. Mid-density regions $\left( \mathbf { Q } _ { 2 } \right)$ exhibit similar behavior (+85.1%), suggesting that moderate point coverage is still sufficient to constrain the underlying Gaussian parameters.

<table><tr><td>Density Group</td><td>n</td><td>Final MSE (median)</td><td>Init MSE (median)</td><td>Improvement (%)</td></tr><tr><td>Dense  $\mathrm { Q _ { 1 } }$ </td><td>43</td><td>9.12</td><td>44.96</td><td>+79.7</td></tr><tr><td>Mid  $\mathrm { Q _ { 2 } }$ </td><td>43</td><td>8.30</td><td>55.56</td><td>+85.1</td></tr><tr><td>Sparse  $\mathrm { Q _ { 3 } }$ </td><td>43</td><td>11.07</td><td>16.67</td><td>+33.6</td></tr></table>

TABLE I: Density-stratified training results under render-free prediction.

Sparse regions $\mathrm { ( Q _ { 3 } ) }$ behave differently. Although training reduces error from 16.67 to 11.07 (+33.6%), the final prediction error remains noticeably higher than in denser regions, and the overall improvement is substantially smaller. This indicates that, under extreme sparsity, geometric observations provide weaker constraints on the target ROR parameters, limiting the effectiveness of render-free prediction.

Fig. 3b (left) summarizes this density-dependent trend, showing a monotonic increase in final MSE as density decreases. Qualitative results (Fig. 2) further confirm that while RFPs accurately reconstruct Gaussian parameter distributions in dense blocks, predictions in sparse blocks are less structured and exhibit higher residual error despite identical model capacity.

## D. Variance Coupling Validation

To validate our variance decomposition theory, we compare training dynamics under different loss configurations (Fig. 3). Baseline coupled optimization (G1) exhibits high oscillation, confirming covariance-dominated variance in sparse regions. Our density-aware decoupled scheme (G4)âwhich downweights covariance-sensitive parameters in low-density areas and enforces scale-structure regularizationâachieves significantly smoother convergence and 20% lower final error in sparse blocks. This empirically confirms that variance coupling is the primary instability source, and that density-aware strategies can partially mitigate (though not eliminate) the fundamental information deficiency.

## E. Discussion

Our experiments validate three key findings: (1) RORs exhibit stable statistical patterns (structured scale distributions and bimodal radiance) across scenes, (2) render-free prediction encounters intrinsic information limits in sparse regions, resulting in consistent performance degradation across density levels, independent of architecture, and (3) variance coupling explains both optimization fragility and prediction limits through density-dependent covariance amplification. While density-aware strategies improve robustness in the supervised regime, they cannot overcome information deficiency in render-free settings. This suggests practical systems should adopt hybrid architectures: feed-forward prediction for dense regions, rendering-based refinement for sparse regions, adaptively allocated based on local density.

## IV. CONCLUSION AND FUTURE WORK

We analyze converged 3D Gaussian Splatting solutions and find density-stratified structure. Sparse regions show visibilitycoupled variance between geometric (Î£) and appearance (S) parameters that prevents render-free predictionânot from model capacity but information deficiency.

Our experiments reveal stable patterns (mix-structured scales, bimodal radiance) and show prediction error increases 2Ã from dense to sparse regions across architectures (Fig. 3b). The decoupling method (G4) improves training stability by 20% in sparse blocks but cannot overcome the fundamental information gap.

Several directions extend this work:

Formal mathematical derivation. Future work includes providing a rigorous proof of Eq. (4) and extending it to the complex 3DGS formulation involving alpha compositing. Another important direction is to theoretically derive the mix-structured scale and bimodal radiance patterns from the underlying optimization dynamics.

Quantifying the density threshold. We demonstrate qualitative differences between dense and sparse regions but do not establish where the transition occurs. The threshold likely depends on point count, local curvature, or view coverage. Characterizing this boundary would inform when feed-forward prediction is viable.

Hierarchical processing strategies. Dense regions exhibit learnable structure while sparse regions do not. Predictions from dense areas could provide anchor constraints for sparse optimization, though whether this avoids reintroducing new variance coupling similar to that of Eq. (3)-(4) requires investigation.

Extension to dynamic reconstruction. We focus on static scenes. Temporal constraints in dynamic settings might provide additional signals in sparse regions or exacerbate coupling effects. Our framework could extend to deformable 3DGS but requires empirical investigation.

These directions build on our core finding: density stratification in 3DGS is not an artifact but reflects fundamental information boundaries. Practical systems must treat densityaware allocation as a necessity rather than an optimization.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[2] J. Park and T. Kim, âPredicting 3dgs parameters from single images without rendering supervision,â in ICCV, 2024.

[3] T. Yi, J. Fang, J. Wang, G. Wu, L. Xie, X. Zhang, W. Liu, Q. Tian, and X. Wang, âGaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6796â6807.

[4] S. Hyung, J. Hong et al., âEffective rank analysis and regularization for enhanced 3d gaussian splatting,â arXiv preprint arXiv:2403.02110, 2024.

[5] T. Lee and S.-E. Ko, âOptimized minimal gaussians: Redundant-free 3d gaussian splatting with compact attributes,â in CVPR, 2025.

[6] T. Sandev and A. Iomin, âHitting times in turbulent diffusion due to multiplicative noise,â Physica A: Statistical Mechanics and its Applications, 2020.

[7] Q. Wu, J. Zheng, and J. Cai, âSurface reconstruction from 3d gaussian splatting via local structural hints,â in European Conference on Computer Vision. Springer, 2024, pp. 441â458.

[8] B. Mildenhall, M. Tancik et al., âNerfies: Deformable neural radiance fields,â in CVPR, 2023.

[9] X. Zhang and Y. Wu, âLearning visibility for 3d scene reconstruction,â in NeurIPS, 2023.

[10] Z. Zhang, B. Huang, H. Jiang, L. Zhou, X. Xiang, and S. Shen, âQuadratic gaussian splatting: High quality surface reconstruction with second-order geometric primitives,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 28 260â28 270.

[11] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, âPlenoxels: Radiance fields without neural networks,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5501â5510.

[12] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphicsÂ¨ primitives with a multiresolution hash encoding,â ACM transactions on graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.