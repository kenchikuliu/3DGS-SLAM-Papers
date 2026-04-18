# Gradient-Driven Natural Selection for Compact 3D Gaussian Splatting

Xiaobin Deng Qiuli Yu Changyu Diao\* Min Li Duanqing Xu\* Zhejiang University

<!-- image-->  
Figure 1. Our methodâs pipeline consists of two parts. The upper section shows the full workflow: starting from sparse SfM point clouds, we densify and optimize to obtain a high-quality dense scene, which is then refined via a natural selection mechanism to produce a compact, high-fidelity representation. The lower section details this natural selection framework: a globally consistent regularization gradient field (dashed lines) is applied to all Gaussian opacities, guiding optimization gradients (solid lines) to identify and prune Gaussians whose opacity falls below a survival threshold. Those Gaussians with smaller optimized gradients will gradually decay in opacity until reaching the death threshold, after which they are permanently removed. The result is a high-quality, compact scene.

## Abstract

3DGS employs a large number of Gaussian primitives to fit scenes, resulting in substantial storage and computational overhead. Existing pruning methods rely on manually designed criteria or introduce additional learnable parameters, yielding suboptimal results. To address this, we propose an natural selection inspired pruning framework that models survival pressure as a regularization gradient field applied to opacity, allowing the optimization gradientsâdriven by the goal of maximizing rendering qualityâto autonomously determine which Gaussians to retain or prune. This process is fully learnable and requires no human intervention. We further introduce an opacity decay technique with a finite opacity prior, which accelerates the selection process without compromising pruning effectiveness. Compared to 3DGS, our method achieves over

0.6 dB PSNR gain under 15% budgets, establishing stateof-the-art performance for compact 3DGS. Project page https://xiaobin2001.github.io/GNS-web .

## 1. Introduction

Novel view synthesis is a fundamental yet challenging task in computer vision, with the core objective of reconstructing scene images from novel viewpoints given a limited set of input views. This technology demonstrates significant value in critical domains such as virtual reality (VR), digital twins, and autonomous driving. Neural Radiance Fields (NeRF) [12], based on implicit neural representations, achieve a breakthrough in high-fidelity view synthesis from sparse input views by parameterizing the light transport function using Multilayer Perceptron (MLP). However, this method struggles to achieve real-time rendering. In contrast, 3D Gaussian Splatting (3DGS) [7], as a reconstruction technique utilizing explicit representation, simultaneously ensures both rendering quality and real-time performance. Initialized from sparse point clouds obtained through SfM (Structure from Motion) [14], 3DGS models the scene as a collection of 3D Gaussian primitives with optimizable parameters. Each primitive is characterized by its center position Âµ, covariance matrix Î£, opacity Î±, and color attributes encoded via spherical harmonics.

<!-- image-->  
Figure 2. Qualitative comparison between our method and 3DGS on the garden scene

3DGS typically employs millions of Gaussians to fit a scene for high-quality rendering. However, the large number of Gaussian ellipsoids significantly increases rendering and storage costs, hindering the adoption of downstream applications. Existing compact 3DGS variants reduce the number of Gaussians through manually designed pruning criteria or by introducing additional learnable masks. Inspired by natural selection, we propose a learnable pruning technique that requires neither manually designed pruning criteria nor additional parameters, thereby enhancing the performance of compact 3DGS.

Figure 1 illustrates the overall pipeline of our proposed method. Inspired by natural selection, where environmental pressures screen for the fittest genes, we model a sparsity metric as a uniform survival pressure applied to all Gaussians, gradually reducing their opacity. Concurrently, optimization gradients aimed at maximizing rendering quality counteract the negative gradients induced by this pressure. This process favors Gaussians that contribute more significantly to rendering quality, as those receiving stronger optimization gradients survive longer under high survival pressure. To confine the natural selection process within a limited number of training iterations, we design a minimalprior opacity decay technique to simulate environmental pressure. Under the same training iterations, our method uses only 15% of the budget and one-third of the training time compared to the original 3DGS, while achieving a 0.6 dB improvement in PSNR. Figure 2 compares our method with the original 3DGS.

In summary, our contributions are as follows:

â¢ We propose a learnable pruning technique that requires neither manually designed pruning criteria nor additional parameters, yielding highly competitive performance.

â¢ We design a opacity decay technique with finite that accelerates the pruning process without compromising the resulting quality.

â¢ Our method achieves state-of-the-art performance in compact 3DGS and is highly portable.

## 2. Related Works

Quality Optimization of 3DGS: 3DGS already achieves a balance between rendering quality and real-time performance, yet there remains room for optimization. AbsGS [16] addresses the inherent reconstruction blur in 3DGS by introducing an absolute gradient-based densification evaluation criterion. TamingGS [11] incorporates multiple evaluation metrics for the densification process and optimizes the rendering kernel of 3DGS, significantly improving training speed. Improved-GS [3] reconstructs the densification process of 3DGS from three perspectives, significantly enhancing rendering quality. ScaffoldGS [10] combines implicit and explicit representations by using neural networks to implicitly store the parameters of Gaussian primitives. Mip-Splatting [17] introduces a 3D smoothing filter and a 2D mipmap filter to mitigate aliasing artifacts that may occur during magnification in 3DGS. GaussianPro [2] utilizes optimized depth and normal maps to initialize new Gaussians via reprojection. It further enhances geometric reconstruction through planar regularization. 2DGS [6] employs 2D Gaussian primitives to represent scenes, improving the application of Gaussian Splatting in geometric reconstruction. Spec-Gaussian [15] adopts an anisotropic spherical Gaussian appearance field for Gaussian color modeling, significantly enhancing rendering quality in complex scenes with specular reflections and anisotropic surfaces.

Compact 3DGS: The approaches to achieving compact 3DGS primarily fall into two categories. One builds upon the neural representation introduced by ScaffoldGS, utilizing octrees [13] or hash grids [1] for compression. This approach relies on specific neural representations and cannot be generalized to standard 3DGS. The other approach involves pruning techniques to eliminate redundant Gaussians, thereby reducing the total number of Gaussians in the scene to achieve compactness.

Compact3DGS [8] aims to reduce both rendering and storage overhead in 3DGS. It proposes a learnable masking strategy to decrease the number of Gaussians, while further compressing storage through neural color representations and vector quantization. LightGaussian [4] employs a global rendering weight combined with Gaussian volume to assist in pruning. LP-3DGS introduces the Gumbel-Sigmoid activation function into 3DGS to replace the Sigmoid function for masking or pruning scoring. Gumbel-Sigmoid pushes values closer to 0 or 1, providing a good approximation for binary masks. Compact3DGS uses the straight-through estimator to address the non-differentiability issue in the binarization process, whereas MaskGS [9] directly employs probabilistic masks and derives gradients for mask parameters, achieving better mask pruning results. MaskGS also utilizes Gumbel-Sigmoid for activating mask parameters. Mini-Splatting [5] introduces depth reinitialization to address the uneven spatial distribution of Gaussians and uses maximum rendering contribution area and global rendering weight as criteria for pruning. GaussianSPA [18] employs the Alternating Direction Method of Multipliers (ADMM) to gradually attenuate the opacity of Gaussians scheduled for pruning to zero during optimization. This replaces explicit pruning with a smooth optimization process, contributing to improved rendering quality. Unlike previous methods, we avoid manual pruning rules and extra parameters.

Based on the different technical approaches adopted, we select the following works as comparative benchmarks: Compact3DGS, as a representative of binary masking; Mini-Splatting, as a representative of scene reorganization for sparsification; MaskGS, as a representative of probabilistic masking; and GaussianSPA, as a representative of progressive sparsification. LP-3DGS [19] only introduces a new activation function, which is also used in MaskGS, and is therefore not included in the comparison. The rendering importance pruning employed by LightGaussian will be compared in the ablation experiments.

## 3. Methods

## 3.1. Preliminaries

In 3DGS framework, a scene is represented by a collection of anisotropic 3D Gaussian primitives:

$$
G ( x ) = \exp { \left( - \frac { 1 } { 2 } ( x ) ^ { T } \Sigma ^ { - 1 } ( x ) \right) } ,\tag{1}
$$

where x denotes the offset from the Gaussianâs mean position, and Î£ is its 3D covariance matrix. To guarantee that Î£ remains positive semi-definite, 3DGS expresses it through a decomposition involving a rotation matrix R and a scaling matrix S:

$$
\Sigma = R S S ^ { T } R ^ { T } .\tag{2}
$$

Here, the scaling matrix S is parameterized by a 3D vector $s ,$ and the rotation matrix R is derived from a unit quaternion $q .$ When rendering an image from a given camera viewpoint, the final color of a pixel p is computed by alphacompositing $N$ Gaussians $\{ G _ { i } \mid i = 1 , \ldots , N \}$ that project onto $p ,$ ordered from front to back, according to:

$$
C = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{3}
$$

where $\alpha _ { i }$ is the opacity contribution of $G _ { i }$ at pixel $p$ (obtained by evaluating the projected Gaussian and scaling by its intrinsic opacity), and $c _ { i }$ is the color of $G _ { i }$ , encoded using spherical harmonics (SH) coefficients.

## 3.2. Compact 3DGS Objective

Background and Challenges: Under a constrained Gaussian budget, the core objective of compact 3DGS remains to achieve high-fidelity rendering. Traditional 3DGS densification strategies identify blurry regions and insert additional Gaussians, which are inherently unconstrained local operations. However, when the total number of Gaussians (budget) is limited, global balance of rendering quality becomes essential. Simply restricting densification growth is therefore insufficient to meet the requirement of compact 3DGS (see Sec. 4.3).

Current approaches typically adopt a two-stage strategy: first, generate a high-quality scene representation through unconstrained densification; second, perform pruning to balance quality and budget.

Bi-Objective Decomposition: In 3DGS, scene optimization relies on back-propagation, where each Gaussian only receives gradient signals from pixels it contributes to. Consequently, the retained subset of Gaussians after pruning, denoted as $G _ { C }$ , fundamentally determines the achievable rendering upper bound. Once a Gaussian representing unique local details is mistakenly pruned, its contribution is almost unrecoverable.

Based on this, we decompose the overall objective of compact 3DGS into two sub-goals:

1. Selecting a superior retained subset $G _ { C }$ to maximize the potential quality upper bound;

2. Ensuring stable convergence of the pruning-optimization process within a limited number of iterations.

The following sections address these two goals respectively. Sec. 3.3 introduces a natural selectionâbased adaptive pruning framework for achieving Goal 1, while Sec. 3.4 presents a regularization gradient field with finite priors to accelerate convergence (Goal 2).

## 3.3. Adaptive Pruning via Natural Selection

To fulfill Goal 1âselecting a superior subset $G _ { C }$ âwe draw inspiration from natural selection in biological evolution (excluding genetic inheritance and mutation) and propose an adaptive pruning framework. This framework simulates the competition between environmental pressure and individual fitness, progressively eliminating low-contribution Gaussians and retaining high-importance ones to maximize rendering quality under a strict budget.

Problem Formulation: Let the complete Gaussian set be $G ~ = ~ \{ g _ { i } \} _ { i = 1 } ^ { N }$ , where each Gaussian $g _ { i }$ is parameterized as $\theta _ { i } = ( \alpha _ { i } , \mu _ { i } , \Sigma _ { i } , c _ { i } )$ (opacity, mean, covariance, and color). The pruning objective is formulated as:

$$
\operatorname* { m i n } _ { G _ { C } \subset G } \mathcal { L } _ { \mathrm { r e n d e r } } ( G _ { C } ) \quad \mathrm { s . t . } \quad | G _ { C } | \leq B ,\tag{4}
$$

where $\mathcal { L } _ { \mathrm { r e n d e r } }$ denotes the rendering loss and B is the Gaussian budget.

Natural Selection Mechanism: We model the biological analogy through the following definitions:

â¢ Vitality: the opacity $\alpha _ { i } \in [ 0 , 1 ]$ represents the existence strength of each Gaussian. This is because opacity directly determines the rendering contribution of each primitive.

â¢ Environmental Pressure: a globally consistent regularization gradient field ${ \mathcal { L } } _ { \mathrm { r e g } } ( \alpha )$ applied to all Gaussians (see Sec. 3.4 for details).

â¢ Fitness: the rendering gradient $\nabla _ { \alpha _ { i } } \mathcal { L } _ { \mathrm { r e n d e r } }$ received by Gaussian $g _ { i } ,$ representing its dynamic importance.

The overall loss is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r e n d e r } } ( \Theta ) + \mathcal { L } _ { \mathrm { r e g } } ( \alpha ) , } \end{array}\tag{5}
$$

and the net gradient on the opacity of each Gaussian is:

$$
\nabla _ { \alpha _ { i } } ^ { \mathrm { n e t } } = \nabla _ { \alpha _ { i } } \mathcal { L } _ { \mathrm { r e n d e r } } + \nabla _ { \alpha } \mathcal { L } _ { \mathrm { r e g } } .\tag{6}
$$

Here, the first term is Gaussian-specific, while the second term is a global constant. In practice, the regularization gradient is applied every $N = 5 0$ rendering iterations. This allows fitness signals to accumulate from multiple training views, producing more stable survival decisions before applying environmental pressure.

If the rendering gradient $\nabla _ { \alpha _ { i } } \mathcal { L } _ { \mathrm { r e n d e r } }$ is opposite in direction to the regularization gradient $\nabla _ { \alpha } \mathcal { L } _ { \mathrm { r e g } }$ , increasing opacity improves rendering quality, and the Gaussian tends to survive. Conversely, when both gradients are aligned, the Gaussian contributes negatively to the rendering and is rapidly suppressed. Since the regularization gradient magnitude is globally constant, the relative differences between rendering gradients dominate the competitionâonly highfitness Gaussians maintain strong vitality and survive.

At the end of densification, when the scene has largely converged, the natural selection process is activated.

## Selection Process and Survival Criterion:

During optimization, a Gaussian is permanently removed if its opacity falls below a survival threshold: $\alpha _ { i } <$ Ï. We empirically set $\tau = 0 . 0 0 1$ , smaller than the 0.005 threshold used in 3DGS, as Gaussians below this level contribute negligibly to rendering.

Strong environmental pressure gradually removes weak Gaussians until $| G _ { C } | \le B$ . The resulting subset $G _ { C }$ thus consists solely of high-fitness, high-contribution primitives.

Importantly, this framework requires no pre-defined importance score; instead, global competition among gradients naturally yields an optimal subset.

Key Innovations: Prior works regularize opacity in two common ways, both fundamentally different from ours:

<!-- image-->  
Figure 3. The figure illustrating the optimization speed improvement of finite prior over no prior.

1. Weak Regularization: introduces low-intensity penalties to reduce average opacity and encourage color aggregation from multiple primitives.

2. Auxiliary Pruning Regularization: applies regularization as a post-processing tool for numerical stability, but it does not participate in pruning decisions.

In contrast, our method applies a strong, globally consistent regularization gradient that explicitly competes with the rendering gradient. This introduces a novel gradientcompetition-driven pruning mechanism, enabling:

1. Automatic Selection: weak Gaussians are eliminated via gradient competition;

2. Smooth Optimization: opacity decays continuously, avoiding instability from discrete pruning.

As shown in Section 4.3, this mechanism consistently outperforms hand-designed pruning heuristics and learnable-mask approaches across multiple datasets.

## 3.4. Gradient Field with Finite Priors

Introduction of Finite Priors: In the natural selection mechanism, the gradient of the optimization loss with respect to opacity parameters is used to counteract the environmental pressure, while other parameters continue to be optimized to minimize the rendering loss. To achieve faster optimization convergence (Goal 2 in Section 3.2), it is essential to design an appropriate environmental pressure that ensures the natural selection process does not hinder the optimization of opacity parameters for surviving Gaussians.

In the natural-selection framework, environmental pressure should be applied uniformly to all Gaussian opacities to guarantee fairness of the selection process. We define the opacity decay ratio:

$$
R _ { o } = { \frac { \alpha _ { t - 1 } - \alpha _ { t } } { \alpha _ { t - 1 } } }\tag{7}
$$

to quantify the degree of influence that the environmental pressure exerts on an individual. Ideally, applying the same decay ratio to every Gaussian accurately models the environmental pressure. Because the regularization gradient received by every Gaussian has the same magnitude and direction, it can be regarded as a gradient field. However, if, by the end of the natural-selection stage, the opacities of surviving Gaussians are generally suppressed too strongly, this will substantially degrade subsequent optimization convergence speed and numerical stability.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 4. Performance under varying Gaussian budgets. Results for the remaining scenes can be found in appendix.

To satisfy Goal 2, we introduce a finite prior designed to accelerate the overall process without breaking the fairness of natural selection. There exists a correlation between a Gaussianâs current opacity and its fitness: high-opacity Gaussians tend to contribute more to rendering and are more likely to be important primitives; moreover, in the late stage of natural selection opacity already reflects fitness. We exploit this weak correlation as a limited prior to speed up both selection and the optimization recovery (see Figure 3). Concretely, we introduce the finite prior by applying the gradient field to the pre-activation opacity parameter v instead of to alpha directly,

Opacity Î± is activated by v through the sigmoid function:

$$
\alpha = S ( v ) = \frac { 1 } { 1 + e ^ { - v } } .\tag{8}
$$

Let the global regularization gradient be âv and the learning rate be lr. Ignoring rendering gradients, the update rule is:

$$
\boldsymbol { v } _ { t + 1 } = \boldsymbol { v } _ { t } + \boldsymbol { \nabla } \boldsymbol { v } \cdot \boldsymbol { l r } ,\tag{9}
$$

yielding the decay ratio:

$$
R _ { o } ^ { t + 1 } = \frac { S ( v _ { t } ) - S ( v _ { t } + \nabla v \cdot l r ) } { S ( v _ { t } ) } .\tag{10}
$$

When $\nabla v \cdot l r  0 .$

$$
R _ { o } ^ { t + 1 } \approx ( 1 - \alpha ) \cdot | \nabla v \cdot l r | .\tag{11}
$$

Thus, the decay ratio $R _ { o }$ is linearly proportional to $1 - \alpha \mathrm { : }$ â¢ As $\alpha  1 , R _ { o } ^ { t + 1 }  0$ , protecting consistently highfitness Gaussians;

â¢ As $\alpha  0 , R _ { o } ^ { t + 1 }  | \nabla v \cdot l r |$ , meaning the decay ratio approaches its maximum value. For example, compared to Gaussians with average opacity $\alpha = 0 . 5$ (whose decay ratio is $0 . 5 \cdot | \nabla v \cdot l r | )$ , low-opacity Gaussians $( \alpha  0 )$ exhibit up to twice the decay magnitude. This accelerates the elimination of less fit individuals.

This limited prior speeds up convergence while preserving the fairness and adaptivity of natural selection.

Implementation Details: To prevent Gaussians with extremely high opacity from evading selection during the initial phase, we apply prior-free regularization under a low learning rate in the early stages of natural selection. After introducing the prior, a higher learning rate is subsequently used to accelerate the screening process. The regularization gradient âv is derived from:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e g } } = ( \mathbb { E } [ v ] - T ) ^ { 2 } , } \end{array}\tag{12}
$$

where E[v] denotes the mean value of all v parameters and $T$ is the regularization target. Its gradient is:

$$
\nabla v = 2 ( \mathbb { E } [ v ] - T ) ,\tag{13}
$$

When $T \ \gg \ \mathbb { E } [ v ]$ , âv is primarily dominated by T , remaining stable throughout the process. In practice, we set $T = - 2 0$ . Since the natural selection process terminates when the number of surviving Gaussians meets the budget requirement, T here primarily serves to provide a gradient direction and a stable magnitude. Because $\nabla \boldsymbol { v } \propto T$ (when $| T | \gg | { \mathbb { E } } [ v ] | ,$ ), increasing |T | is equivalent to linearly amplifying the strength of the gradient field, an effect similar to increasing the learning rate lr.

## 4. Experiments

## 4.1. Datasets and Metrics

Following 3DGS and other compact 3DGS baselines, we select 13 scenes from the Mip-NeRF 360, Deep Blending, and Tanks and Temples datasets for evaluation. In each experiment, every 8th image is used as the test set, and the remaining images form the training set.

For quantitative evaluation, we adopt three widely used metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). All metrics are computed using the official evaluation framework provided by 3DGS.

<table><tr><td>Dataset</td><td colspan="5">Mip-NeRF360</td><td colspan="5">Deep Blending</td><td colspan="5">Tanks&amp;Temples</td></tr><tr><td>MethodâMetric</td><td>SSIMâ</td><td>PSNRâ LPIPSâ¡ Num+ Time+]</td><td></td><td></td><td></td><td>SSIMâ</td><td></td><td></td><td>PSNRâ LPIPSâNum</td><td>Time</td><td>SSIMâ</td><td></td><td>PSNRâ LPIPSâ</td><td>Num</td><td>Time</td></tr><tr><td>3DGS</td><td>0.816</td><td>27.50</td><td>0.216</td><td>3320453</td><td>35.9</td><td>0.904</td><td>29.56</td><td>0.244</td><td>2819180</td><td>32.5</td><td>0.849</td><td>23.72</td><td>0.177</td><td>1835092</td><td>19.9</td></tr><tr><td>Compact-3DGS</td><td>0.807</td><td>27.33</td><td>0.227</td><td>1516172</td><td>31.3</td><td>0.904</td><td>29.61</td><td>0.249</td><td>1251516</td><td>28.9</td><td>0.847</td><td>23.67</td><td>0.180</td><td>933650</td><td>18.0</td></tr><tr><td>Mini-splatting</td><td>0.822</td><td>27.36</td><td>0.217</td><td>493466</td><td>26.8</td><td>0.910</td><td>30.07</td><td>0.241</td><td>554179</td><td>23.2</td><td>0.847</td><td>23.47</td><td>0.180</td><td>300481</td><td>17.3</td></tr><tr><td>MaskGS</td><td>0.815</td><td>27.43</td><td>0.218</td><td>1582926</td><td>31.9</td><td>0.907</td><td>29.76</td><td>0.245</td><td>910162</td><td>27.6</td><td>0.847</td><td>23.73</td><td>0.180</td><td>740683</td><td>17.4</td></tr><tr><td>GaussianSPA</td><td>0.817</td><td>27.31</td><td>0.229</td><td>421427</td><td>34.3</td><td>0.913</td><td>30.00</td><td>0.242</td><td>443740</td><td>28.8</td><td>0.850</td><td>23.40</td><td>0.171</td><td>424801</td><td>23.3</td></tr><tr><td>ImprovedGS</td><td>0.814</td><td>27.66</td><td>0.233</td><td>466667</td><td>8.1</td><td>0.911</td><td>30.23</td><td>0.244</td><td>450000</td><td>6.8</td><td>0.856</td><td>24.39</td><td>0.179</td><td>450000</td><td>5.7</td></tr><tr><td>Ours</td><td>0.833</td><td>28.13</td><td>0.207</td><td>466667</td><td>12.5</td><td>0.914</td><td>30.15</td><td>0.233</td><td>450000</td><td>9.4</td><td>0.871</td><td>24.63</td><td>0.154</td><td>450000</td><td>8.5</td></tr></table>

Table 1. Quantitative results on the Mip-NeRF 360, Deep Blending, and Tanks and Temples datasets. Cells are highlighted as follows: best , and second best . Per-scene detailed data can be found in the appendix.

## 4.2. Implementation Details

Our method is independent of specific CUDA kernels and introduces no additional trainable parameters, making it easily transferable to advanced 3DGS variants. We adopt Improved-GS as the base model, since it provides faster and higher-quality densification. To reduce the influence of initial opacity before natural selection, we increase the opacity learning rate to 4 times the original value, ensuring that survival is primarily determined by dynamic gradient competition. The opacity learning rate will be restored to its original value after natural selection is completed.

The natural selection stage begins after densification (at 15K iterations) and continues until the number of remaining Gaussians meets the target budget. Empirically, this process converges within 5Kâ8K iterations, achieving the best rendering quality. Based on this observation, we tune the regularization learning rate per scene, while all other hyperparameters remain shared across scenes. A detailed discussion of hyperparameter choices is provided in appendix.

All experiments are conducted on a single RTX A5000 GPU, with a total of 30K training iterations for all methods. Competing methods are reimplemented using the official configurations provided by their authors. During the reproduction of GaussianSPA, several non-trivial issues were observed and are discussed in the appendix. As noted in Section 3.2, densification alone cannot meet the compact 3DGS requirement. Thus, we also include a sparse variant of Improved-GSâobtained by lowering the peak densification budgetâas an additional baseline.

## 4.3. Quantitative Comparison

The quantitative results are summarized in Table 1. Despite using a minimal Gaussian budget, our method achieves the best overall performance across PSNR, SSIM, and LPIPS. Compared with 3DGS, our approach attains a > 0.6dB gain in PSNR while using only 15% of the Gaussian budget. Relative to Improved-GS, our method shows comparable training time but significantly better rendering quality.

Mini-Splatting reaches a level similar to 3DGS with low budget usage, whereas GaussianSPA, due to its ADMMbased optimization, requires substantially longer iterations and performs pruning only at the end, resulting in longer training time. Compared to CompactGS, MaskGS achieves comparable or superior rendering quality with a similar or smaller budget. Overall, our method achieves SOTA performance in compact 3DGS.

<table><tr><td>Dataset MethodâMetric</td><td>Outdoors</td><td>PSNRâ LPIPSâ</td><td>SSIMâ  PSNRâ LPIPSâ</td><td>Indoors</td><td></td></tr><tr><td>ImprovedGS (sparse)</td><td>SSIMâ 0.728</td><td>24.85</td><td>0.260</td><td>0.922 31.18</td><td>0.198</td></tr><tr><td>SPA Pruning</td><td>0.733</td><td>24.95</td><td>0.261</td><td>0.928 31.70</td><td>0.184</td></tr><tr><td>MaskGS Pruning</td><td>0.749</td><td>25.11</td><td>0.242 0.927</td><td>31.67</td><td>0.184</td></tr><tr><td>Opacity Pruning</td><td>0.722</td><td>24.75</td><td>0.275</td><td>0.924 31.52</td><td>0.190</td></tr><tr><td>Render Pruning Edge Pruning</td><td>0.726</td><td>24.67</td><td>0.261</td><td>0.923 31.39</td><td>0.191</td></tr><tr><td>Natural Selection (Ours)</td><td>0.728 0.753</td><td>24.81 25.20</td><td>0.263 0.234</td><td>0.924 31.50 0.930 31.78</td><td>0.193 0.178</td></tr></table>

Table 2. Ablation Study of unified base model.

<table><tr><td rowspan=2 colspan=1>DatasetMethodâMetric</td><td rowspan=2 colspan=5>OutdoorsSSIMâ  PSNRâ LPIPSâ</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=4>IndoorsSSIMâ  PSNRâ LPIPSâ </td></tr><tr><td rowspan=1 colspan=1>No Prior</td><td rowspan=1 colspan=1>0.751</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>25.18</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.242</td><td rowspan=1 colspan=4>0.928 31.70  0.183</td></tr><tr><td rowspan=1 colspan=1>Strong Prior</td><td rowspan=1 colspan=1>0.747</td><td rowspan=1 colspan=2>25.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.245</td><td rowspan=1 colspan=1>0.929</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>31.75</td><td rowspan=1 colspan=1>0.181</td></tr><tr><td rowspan=1 colspan=1>Finite Prior (Ours)</td><td rowspan=1 colspan=1>0.753</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>25.20</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.234</td><td rowspan=1 colspan=1>0.930</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>31.78</td><td rowspan=1 colspan=1>0.178</td></tr></table>

Table 3. Ablation Study of prior.

Figure 4 illustrates performance curves under different budget settings. Notably, in some scenes, our approach surpasses the original 3DGS using only 5% of its budget.

## 4.4. Qualitative Comparison

Representative qualitative results are shown in Figure 5.

Garden: Our method achieves higher rendering quality than 3DGS while using only 10% of the Gaussian budget. Although Mini-Splatting and GaussianSPA consume less budget, their reconstruction quality is far inferior. Compared to ImprovedGS, our approach restores finer scene details and avoids large blurry regions.

Room: mong all methods, our approach produces fewer artifacts and is the only one that successfully restores the details of the globe in the upper left corner. Train: Our method delivers superior reconstruction quality while maintaining the lowest budget among all methods. Except for ImprovedGS-sparse, all other baselines exhibit visible artifacts and fail to reconstruct the distant mountain contours accurately. Our approach produces more faithful geometry and textures.

<!-- image-->  
Figure 5. Qualitative comparison results among scenes garden, room, train. Num is final Gaussian count after training. The results for the remaining scenes can be found in appendix.

## 4.5. Ablation Studies

Unified Base Model: To eliminate the influence of inherent model differences (e.g., Improved-GS, 3DGS, MiniGS-D) and ensure a fair comparison of pruning techniques, we apply all pruning methods on a unified Improved-GS (15K iteration) base model. We choose MaskGS as the representative of mask-based pruning and SPA for smooth pruning. Additionally, we test three non-learnable pruning weights: (1) opacity-based weight (used in LightGaussian and Mini-

Splatting), (2) rendering-weight, and (3) edge-weight from Improved-GS densification. These weights are used as relative retention probabilities to prevent structural voids.

As shown in Table 2, our method consistently outperforms all others. Among the baselines, MaskGS performs better in outdoor scenes, while SPA is slightly superior in indoor scenes. This indicates that in complex scenes, selecting the right Gaussians is critical, whereas in simpler scenes, smoothness of pruning plays a larger role. Our approach effectively balances both aspects, achieving the best results. We also observe that single-weight heuristics differ little among themselves but remain substantially worse than

<!-- image-->  
Figure 6. Comparison of the point cloud distributions in scenes trained with 3DGS and our method. For the bicycle scene, the background has been removed to more clearly highlight the differences between the two approaches.

learnable methods.

Finite Prior Ablation: To validate the effectiveness of the finite prior design, we conduct two ablation variants:

1. No Prior: By manually compensating gradients, the opacity attenuation magnitude of all Gaussians is maintained strictly consistent, thereby eliminating prior influence. Since direct parameter modification invalidates the optimizer, manual gradient compensation serves as an approximation.

2. Strong Prior: Using opacity as the sampling probability for exemption from natural selection in the current round.

As shown in Table 3, the No Prior variant fails to achieve efficient convergence (Goal 2) and performs poorly in indoor scenes. The Strong Prior variant disrupts the fairness and adaptability of natural selection (Goal 1), leading to significant performance drops in outdoor scenes. Our finite prior strikes the optimal balance between fairness and efficiency. More ablation study results can be found in appendix.

## 4.6. Spatial Distribution of Point Cloud

Figure 6 compares the spatial distribution of Gaussians between 3DGS and our method after training. Apart from a significant reduction in the total number of Gaussians, our method yields more uniform distributions. In 3DGS, Gaussian points often form dense clusters, as observed in the bicycle scene, where many Gaussians are concentrated in lowfrequency areas such as the bicycle frame, resulting in redundancy. Mini-Splatting mitigates this problem via depth reinitialization, whereas our method eliminates redundancy without any external intervention. During training, the positions of Gaussians in edge regions undergo repeated finetuning, causing these areas to receive high gradients and form excessively dense distributions (e.g., along table edges and carpet boundaries in the Playroom scene). Our method avoids such over-clustering effectively.

## 5. Conclusion

We presented a biologically inspired simplification framework for 3D Gaussian Splatting that leverages natural selection principles to autonomously prune redundant Gaussians. By applying uniform survival pressure through opacity regularization and using optimization gradients as fitness indicators, our method eliminates the need for manual criteria or additional parameters. Accelerated by finite-prior opacity decay, it achieves SOTA rendering quality with only 15% of the original Gaussian budget and one-third of the training time, while improving PSNR by over 0.6 dB. This efficient, portable solution significantly lowers computational barriers, enabling broader adoption of 3DGS in resourceconstrained applications. Future work will explore integrating our natural-selection framework into dynamic or sparseview reconstruction.

## References

[1] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Hac: Hash-grid assisted context for 3d gaussian splatting compression. In European Conference on Computer Vision, pages 422â438. Springer, 2024. 2

[2] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin, Yuexin Ma, Wenping Wang, and Xuejin Chen. Gaussianpro: 3d gaussian splatting with progressive propagation. In Fortyfirst International Conference on Machine Learning, 2024. 2

[3] Xiaobin Deng, Changyu Diao, Min Li, Ruohan Yu, and Duanqing Xu. Improving densification in 3d gaussian splatting for high-fidelity rendering. arXiv preprint arXiv:2508.12313, 2025. 2

[4] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural information processing systems, 37: 140138â140158, 2024. 2

[5] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In European Conference on Computer Vision, pages 165â181. Springer, 2024. 3

[6] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1â11, 2024. 2

[7] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1

[8] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21719â 21728, 2024. 2

[9] Yifei Liu, Zhihang Zhong, Yifan Zhan, Sheng Xu, and Xiao Sun. Maskgaussian: Adaptive 3d gaussian representation from probabilistic masks. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 681â690, 2025. 3

[10] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 2

[11] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3dgs: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers, pages 1â11, 2024. 2

[12] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1

[13] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent

real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024. 2

[14] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104â4113, 2016. 2

[15] Ziyi Yang, Xinyu Gao, Yang-Tian Sun, Yihua Huang, Xiaoyang Lyu, Wen Zhou, Shaohui Jiao, Xiaojuan Qi, and Xiaogang Jin. Spec-gaussian: Anisotropic view-dependent appearance for 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:61192â61216, 2024. 2

[16] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou. Absgs: Recovering fine details in 3d gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 1053â1061, 2024. 2

[17] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024. 2

[18] Yangming Zhang, Wenqi Jia, Wei Niu, and Miao Yin. Gaussianspa: Anâ optimizing-sparsifyingâ simplification framework for compact and high-quality 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26673â26682, 2025. 3

[19] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang, Cheng Peng, Rama Chellappa, and Deliang Fan. Lp-3dgs: Learning to prune 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:122434â122457, 2024. 3

## A. Appendix

## A.1. Quantitative Comparison per Scene

<table><tr><td rowspan=1 colspan=4>SceneâMethods</td><td rowspan=1 colspan=1>3DGS</td><td rowspan=1 colspan=1>CompactGS</td><td rowspan=1 colspan=1>MiniGS</td><td rowspan=1 colspan=1>MaskGS</td><td rowspan=1 colspan=1>SPA</td><td rowspan=1 colspan=1>ImprovedGS-s</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=4>bicycle</td><td rowspan=1 colspan=1>0.765</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.773</td><td rowspan=1 colspan=1>0.765</td><td rowspan=1 colspan=1>0.761</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.792</td></tr><tr><td rowspan=1 colspan=4>flowers</td><td rowspan=1 colspan=1>0.606</td><td rowspan=1 colspan=1>0.592</td><td rowspan=1 colspan=1>0.625</td><td rowspan=1 colspan=1>0.604</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.640</td></tr><tr><td rowspan=1 colspan=4>garden</td><td rowspan=1 colspan=1>0.867</td><td rowspan=1 colspan=1>0.855</td><td rowspan=1 colspan=1>0.848</td><td rowspan=1 colspan=1>0.866</td><td rowspan=1 colspan=1>0.840</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.867</td></tr><tr><td rowspan=1 colspan=4>stump</td><td rowspan=1 colspan=1>0.773</td><td rowspan=1 colspan=1>0.756</td><td rowspan=1 colspan=1>0.805</td><td rowspan=1 colspan=1>0.774</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.813</td></tr><tr><td rowspan=1 colspan=4>treehill</td><td rowspan=1 colspan=1>0.632</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=1>0.654</td><td rowspan=1 colspan=1>0.634</td><td rowspan=1 colspan=1>0.656</td><td rowspan=1 colspan=1>0.648</td><td rowspan=1 colspan=1>0.664</td></tr><tr><td rowspan=1 colspan=4>bonsai</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.940</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.947</td></tr><tr><td rowspan=2 colspan=2>counterkitchen</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.906</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.916</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.928</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.923</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.931</td></tr><tr><td rowspan=1 colspan=4>room</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.918</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.922</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.929</td></tr><tr><td rowspan=1 colspan=4>playroom</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.914</td><td rowspan=1 colspan=1>0.910</td><td rowspan=1 colspan=1>0.916</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.916</td></tr><tr><td rowspan=1 colspan=4>drjohnson</td><td rowspan=1 colspan=1>0.901</td><td rowspan=1 colspan=1>0.901</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.904</td><td rowspan=1 colspan=1>0.910</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.911</td></tr><tr><td rowspan=2 colspan=4>traintruck</td><td rowspan=2 colspan=1>0.8150.882</td><td rowspan=1 colspan=1>0.813</td><td rowspan=1 colspan=1>0.812</td><td rowspan=1 colspan=1>0.812</td><td rowspan=1 colspan=1>0.813</td><td rowspan=1 colspan=1>0.819</td><td rowspan=1 colspan=1>0.843</td></tr><tr><td rowspan=1 colspan=1>0.880</td><td rowspan=1 colspan=1>0.883</td><td rowspan=1 colspan=1>0.881</td><td rowspan=1 colspan=1>0.887</td><td rowspan=1 colspan=1>0.892</td><td rowspan=1 colspan=1>0.899</td></tr></table>

Table 4. The SSIM scores for all works in each scene.

<table><tr><td rowspan=1 colspan=1>SceneâMethods</td><td rowspan=1 colspan=1>3DGS</td><td rowspan=1 colspan=1>CompactGS</td><td rowspan=1 colspan=1>MiniGS</td><td rowspan=1 colspan=1>MaskGS</td><td rowspan=1 colspan=1>SPA</td><td rowspan=1 colspan=1>ImprovedGS-s</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>bicycle</td><td rowspan=1 colspan=1>25.22</td><td rowspan=1 colspan=1>24.92</td><td rowspan=1 colspan=1>25.22</td><td rowspan=1 colspan=1>25.21</td><td rowspan=1 colspan=1>25.07</td><td rowspan=1 colspan=1>25.43</td><td rowspan=1 colspan=1>25.76</td></tr><tr><td rowspan=1 colspan=1>flowers</td><td rowspan=1 colspan=1>21.62</td><td rowspan=1 colspan=1>21.39</td><td rowspan=1 colspan=1>21.53</td><td rowspan=1 colspan=1>21.55</td><td rowspan=1 colspan=1>21.56</td><td rowspan=1 colspan=1>21.59</td><td rowspan=1 colspan=1>21.86</td></tr><tr><td rowspan=1 colspan=1>garden</td><td rowspan=1 colspan=1>27.41</td><td rowspan=1 colspan=1>27.13</td><td rowspan=1 colspan=1>26.91</td><td rowspan=1 colspan=1>27.39</td><td rowspan=1 colspan=1>26.80</td><td rowspan=1 colspan=1>27.17</td><td rowspan=1 colspan=1>27.79</td></tr><tr><td rowspan=1 colspan=1>stump</td><td rowspan=1 colspan=1>26.64</td><td rowspan=1 colspan=1>26.31</td><td rowspan=1 colspan=1>27.24</td><td rowspan=1 colspan=1>26.66</td><td rowspan=1 colspan=1>27.05</td><td rowspan=1 colspan=1>27.05</td><td rowspan=1 colspan=1>27.38</td></tr><tr><td rowspan=1 colspan=1>treehill</td><td rowspan=1 colspan=1>22.44</td><td rowspan=1 colspan=1>22.48</td><td rowspan=1 colspan=1>22.73</td><td rowspan=1 colspan=1>22.52</td><td rowspan=1 colspan=1>23.03</td><td rowspan=1 colspan=1>23.00</td><td rowspan=1 colspan=1>23.16</td></tr><tr><td rowspan=1 colspan=1>bonsai</td><td rowspan=1 colspan=1>32.26</td><td rowspan=1 colspan=1>32.14</td><td rowspan=1 colspan=1>31.49</td><td rowspan=1 colspan=1>32.08</td><td rowspan=1 colspan=1>31.40</td><td rowspan=1 colspan=1>32.22</td><td rowspan=1 colspan=1>32.77</td></tr><tr><td rowspan=2 colspan=1>counterkitchen</td><td rowspan=1 colspan=1>29.03</td><td rowspan=1 colspan=1>28.98</td><td rowspan=1 colspan=1>28.65</td><td rowspan=1 colspan=1>28.96</td><td rowspan=1 colspan=1>28.56</td><td rowspan=1 colspan=1>29.31</td><td rowspan=1 colspan=1>29.68</td></tr><tr><td rowspan=1 colspan=1>31.48</td><td rowspan=1 colspan=1>31.15</td><td rowspan=1 colspan=1>31.26</td><td rowspan=1 colspan=1>31.14</td><td rowspan=1 colspan=1>30.99</td><td rowspan=1 colspan=1>31.06</td><td rowspan=2 colspan=1>32.2532.52</td></tr><tr><td rowspan=1 colspan=1>room</td><td rowspan=1 colspan=1>31.43</td><td rowspan=1 colspan=1>31.49</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>31.40</td><td rowspan=1 colspan=1>31.36</td><td rowspan=1 colspan=1>32.14</td></tr><tr><td rowspan=1 colspan=1>playroom</td><td rowspan=1 colspan=1>29.96</td><td rowspan=1 colspan=1>30.06</td><td rowspan=1 colspan=1>30.57</td><td rowspan=1 colspan=1>30.15</td><td rowspan=1 colspan=1>30.49</td><td rowspan=1 colspan=1>30.66</td><td rowspan=1 colspan=1>30.63</td></tr><tr><td rowspan=1 colspan=1>drjohnson</td><td rowspan=1 colspan=1>29.16</td><td rowspan=1 colspan=1>29.17</td><td rowspan=1 colspan=1>29.57</td><td rowspan=1 colspan=1>29.37</td><td rowspan=1 colspan=1>29.51</td><td rowspan=1 colspan=1>29.79</td><td rowspan=1 colspan=1>29.66</td></tr><tr><td rowspan=2 colspan=1>traintruck</td><td rowspan=2 colspan=1>22.0025.44</td><td rowspan=2 colspan=1>22.0125.33</td><td rowspan=1 colspan=1>21.61</td><td rowspan=1 colspan=1>22.09</td><td rowspan=1 colspan=1>21.34</td><td rowspan=2 colspan=1>22.4326.35</td><td rowspan=2 colspan=1>22.7426.52</td></tr><tr><td rowspan=1 colspan=1>25.33</td><td rowspan=1 colspan=1>25.36</td><td rowspan=1 colspan=1>25.47</td></tr></table>

Table 5. The PSNR scores for all works in each scene.

<table><tr><td rowspan=1 colspan=1>SceneâMethods</td><td rowspan=1 colspan=1>3DGS</td><td rowspan=1 colspan=1>CompactGS</td><td rowspan=1 colspan=1>MiniGS</td><td rowspan=1 colspan=1>MaskGS</td><td rowspan=1 colspan=1>SPA</td><td rowspan=1 colspan=1>ImprovedGS-s</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>bicycle</td><td rowspan=1 colspan=1>0.210</td><td rowspan=1 colspan=1>0.235</td><td rowspan=1 colspan=1>0.225</td><td rowspan=1 colspan=1>0.212</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.213</td></tr><tr><td rowspan=1 colspan=1>flowers</td><td rowspan=1 colspan=1>0.335</td><td rowspan=1 colspan=1>0.351</td><td rowspan=1 colspan=1>0.327</td><td rowspan=1 colspan=1>0.337</td><td rowspan=1 colspan=1>0.346</td><td rowspan=1 colspan=1>0.337</td><td rowspan=1 colspan=1>0.307</td></tr><tr><td rowspan=1 colspan=1>garden</td><td rowspan=1 colspan=1>0.106</td><td rowspan=1 colspan=1>0.124</td><td rowspan=1 colspan=1>0.150</td><td rowspan=1 colspan=1>0.108</td><td rowspan=1 colspan=1>0.168</td><td rowspan=1 colspan=1>0.164</td><td rowspan=1 colspan=1>0.127</td></tr><tr><td rowspan=1 colspan=1>stump</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>0.237</td><td rowspan=1 colspan=1>0.199</td><td rowspan=1 colspan=1>0.216</td><td rowspan=1 colspan=1>0.225</td><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>0.197</td></tr><tr><td rowspan=1 colspan=1>treehill</td><td rowspan=1 colspan=1>0.327</td><td rowspan=1 colspan=1>0.335</td><td rowspan=1 colspan=1>0.313</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=1>0.331</td><td rowspan=1 colspan=1>0.333</td><td rowspan=1 colspan=1>0.306</td></tr><tr><td rowspan=1 colspan=1>bonsai</td><td rowspan=1 colspan=1>0.203</td><td rowspan=1 colspan=1>0.205</td><td rowspan=1 colspan=1>0.200</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.199</td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1>0.193</td></tr><tr><td rowspan=3 colspan=1>counterkitchenroom</td><td rowspan=1 colspan=1>0.200</td><td rowspan=2 colspan=1>0.2040.128</td><td rowspan=1 colspan=1>0.198</td><td rowspan=1 colspan=1>0.203</td><td rowspan=1 colspan=1>0.198</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.189</td></tr><tr><td rowspan=1 colspan=1>0.126</td><td rowspan=1 colspan=1>0.129</td><td rowspan=1 colspan=1>0.129</td><td rowspan=1 colspan=1>0.134</td><td rowspan=2 colspan=1>0.1500.221</td><td rowspan=1 colspan=1>0.126</td></tr><tr><td rowspan=1 colspan=1>0.218</td><td rowspan=1 colspan=1>0.223</td><td rowspan=1 colspan=1>0.212</td><td rowspan=1 colspan=1>0.222</td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1>0.202</td></tr><tr><td rowspan=1 colspan=1>playroom</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.248</td><td rowspan=1 colspan=1>0.238</td><td rowspan=1 colspan=1>0.247</td><td rowspan=1 colspan=1>0.239</td><td rowspan=1 colspan=1>0.248</td><td rowspan=1 colspan=1>0.235</td></tr><tr><td rowspan=1 colspan=1>drjohnson</td><td rowspan=1 colspan=1>0.244</td><td rowspan=1 colspan=1>0.249</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.245</td><td rowspan=1 colspan=1>0.240</td><td rowspan=1 colspan=1>0.230</td></tr><tr><td rowspan=2 colspan=1>traintruck</td><td rowspan=2 colspan=1>0.2070.146</td><td rowspan=2 colspan=1>0.2100.151</td><td rowspan=2 colspan=1>0.2220.139</td><td rowspan=2 colspan=1>0.2120.149</td><td rowspan=1 colspan=1>0.216</td><td rowspan=2 colspan=1>0.2190.139</td><td rowspan=2 colspan=1>0.1900.119</td></tr><tr><td rowspan=1 colspan=1>0.126</td></tr></table>

Table 6. The LPIPS scores for all works in each scene.

<table><tr><td rowspan=1 colspan=1>SceneâMethods</td><td rowspan=1 colspan=1>ImprovedGS-s</td><td rowspan=1 colspan=1>SPA</td><td rowspan=1 colspan=1>MaskGS</td><td rowspan=1 colspan=1>Opacity Pruning</td><td rowspan=1 colspan=1>Render Pruning</td><td rowspan=1 colspan=1>Edge Pruning</td><td rowspan=1 colspan=1>Natural Selection(Ours)</td></tr><tr><td rowspan=1 colspan=1>bicycle</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.759</td><td rowspan=1 colspan=1>0.781</td><td rowspan=1 colspan=1>0.751</td><td rowspan=1 colspan=1>0.755</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.792</td></tr><tr><td rowspan=1 colspan=1>flowers</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.618</td><td rowspan=1 colspan=1>0.636</td><td rowspan=1 colspan=1>0.606</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.613</td><td rowspan=1 colspan=1>0.637</td></tr><tr><td rowspan=1 colspan=1>garden</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.844</td><td rowspan=1 colspan=1>0.861</td><td rowspan=1 colspan=1>0.842</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>0.867</td></tr><tr><td rowspan=1 colspan=1>stump</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>0.788</td><td rowspan=2 colspan=1>0.8130.657</td></tr><tr><td rowspan=1 colspan=1>treehill</td><td rowspan=1 colspan=1>0.648</td><td rowspan=1 colspan=1>0.647</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=1>0.637</td><td rowspan=1 colspan=1>0.636</td></tr><tr><td rowspan=1 colspan=1>bonsai</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.945</td><td rowspan=1 colspan=1>0.944</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.947</td></tr><tr><td rowspan=1 colspan=1>counter</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.916</td></tr><tr><td rowspan=1 colspan=1>kitchen</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.928</td><td rowspan=1 colspan=1>0.925</td><td rowspan=1 colspan=1>0.925</td><td rowspan=1 colspan=1>0.924</td><td rowspan=1 colspan=1>0.930</td></tr><tr><td rowspan=1 colspan=1>room</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.924</td><td rowspan=1 colspan=1>0.923</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.922</td><td rowspan=1 colspan=1>0.929</td></tr></table>

Table 7. The SSIM scores for all ablation studies in each scene.

<table><tr><td rowspan=1 colspan=1>SceneâMethods</td><td rowspan=1 colspan=1>ImprovedGS-s</td><td rowspan=1 colspan=1>SPA</td><td rowspan=1 colspan=1>MaskGS</td><td rowspan=1 colspan=1>Opacity Pruning</td><td rowspan=1 colspan=1>Render Pruning</td><td rowspan=1 colspan=1>Edge Pruning</td><td rowspan=1 colspan=1>Natural Selection(Ours)</td></tr><tr><td rowspan=1 colspan=1>bicycle</td><td rowspan=1 colspan=1>25.43</td><td rowspan=1 colspan=1>25.38</td><td rowspan=1 colspan=1>25.63</td><td rowspan=1 colspan=1>25.25</td><td rowspan=1 colspan=1>25.25</td><td rowspan=1 colspan=1>25.39</td><td rowspan=2 colspan=1>25.8021.85</td></tr><tr><td rowspan=1 colspan=1>flowers</td><td rowspan=1 colspan=1>21.59</td><td rowspan=1 colspan=1>21.74</td><td rowspan=1 colspan=1>21.85</td><td rowspan=1 colspan=1>21.46</td><td rowspan=1 colspan=1>21.41</td><td rowspan=1 colspan=1>21.55</td></tr><tr><td rowspan=1 colspan=1>garden</td><td rowspan=1 colspan=1>27.17</td><td rowspan=1 colspan=1>27.33</td><td rowspan=1 colspan=1>27.68</td><td rowspan=1 colspan=1>27.29</td><td rowspan=1 colspan=1>27.23</td><td rowspan=1 colspan=1>27.39</td><td rowspan=1 colspan=1>27.80</td></tr><tr><td rowspan=1 colspan=1>stump</td><td rowspan=1 colspan=1>27.05</td><td rowspan=1 colspan=1>27.19</td><td rowspan=1 colspan=1>27.26</td><td rowspan=1 colspan=1>26.90</td><td rowspan=1 colspan=1>26.74</td><td rowspan=1 colspan=1>26.95</td><td rowspan=2 colspan=1>27.4223.16</td></tr><tr><td rowspan=1 colspan=1>treehill</td><td rowspan=1 colspan=1>23.00</td><td rowspan=1 colspan=1>23.12</td><td rowspan=1 colspan=1>23.15</td><td rowspan=1 colspan=1>22.87</td><td rowspan=1 colspan=1>22.71</td><td rowspan=1 colspan=1>22.78</td></tr><tr><td rowspan=1 colspan=1>bonsai</td><td rowspan=1 colspan=1>32.22</td><td rowspan=1 colspan=1>32.74</td><td rowspan=1 colspan=1>32.63</td><td rowspan=1 colspan=1>32.40</td><td rowspan=1 colspan=1>32.19</td><td rowspan=1 colspan=1>32.46</td><td rowspan=1 colspan=1>32.74</td></tr><tr><td rowspan=1 colspan=1>counter</td><td rowspan=1 colspan=1>29.31</td><td rowspan=1 colspan=1>29.61</td><td rowspan=1 colspan=1>29.59</td><td rowspan=1 colspan=1>29.42</td><td rowspan=1 colspan=1>29.47</td><td rowspan=1 colspan=1>29.55</td><td rowspan=1 colspan=1>29.69</td></tr><tr><td rowspan=1 colspan=1>kitchen</td><td rowspan=1 colspan=1>31.06</td><td rowspan=1 colspan=1>32.08</td><td rowspan=1 colspan=1>32.08</td><td rowspan=1 colspan=1>31.96</td><td rowspan=1 colspan=1>31.81</td><td rowspan=1 colspan=1>31.74</td><td rowspan=1 colspan=1>32.22</td></tr><tr><td rowspan=1 colspan=1>room</td><td rowspan=1 colspan=1>32.14</td><td rowspan=1 colspan=1>32.36</td><td rowspan=1 colspan=1>32.40</td><td rowspan=1 colspan=1>32.30</td><td rowspan=1 colspan=1>32.10</td><td rowspan=1 colspan=1>32.25</td><td rowspan=1 colspan=1>32.45</td></tr></table>

Table 8. The PSNR scores for all ablation studies in each scene.

<table><tr><td rowspan=1 colspan=1>SceneâMethods</td><td rowspan=1 colspan=1>ImprovedGS-s</td><td rowspan=1 colspan=1>SPA</td><td rowspan=1 colspan=1>MaskGS</td><td rowspan=1 colspan=1>Opacity Pruning</td><td rowspan=1 colspan=1>Render Pruning</td><td rowspan=1 colspan=1>Edge Pruning</td><td rowspan=1 colspan=1>Natural Selection(Ours)</td></tr><tr><td rowspan=1 colspan=1>bicycle</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.260</td><td rowspan=1 colspan=1>0.236</td><td rowspan=1 colspan=1>0.265</td><td rowspan=1 colspan=1>0.253</td><td rowspan=1 colspan=1>0.256</td><td rowspan=2 colspan=1>0.2140.311</td></tr><tr><td rowspan=1 colspan=1>flowers</td><td rowspan=1 colspan=1>0.337</td><td rowspan=1 colspan=1>0.330</td><td rowspan=1 colspan=1>0.311</td><td rowspan=1 colspan=1>0.339</td><td rowspan=1 colspan=1>0.332</td><td rowspan=1 colspan=1>0.330</td></tr><tr><td rowspan=1 colspan=1>garden</td><td rowspan=1 colspan=1>0.164</td><td rowspan=1 colspan=1>0.161</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>0.155</td><td rowspan=1 colspan=1>0.161</td><td rowspan=1 colspan=1>0.128</td></tr><tr><td rowspan=1 colspan=1>stump</td><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.236</td><td rowspan=1 colspan=1>0.229</td><td rowspan=1 colspan=1>0.225</td><td rowspan=2 colspan=1>0.1970.318</td></tr><tr><td rowspan=1 colspan=1>treehill</td><td rowspan=1 colspan=1>0.333</td><td rowspan=1 colspan=1>0.338</td><td rowspan=1 colspan=1>0.320</td><td rowspan=1 colspan=1>0.364</td><td rowspan=1 colspan=1>0.336</td><td rowspan=1 colspan=1>0.344</td></tr><tr><td rowspan=1 colspan=1>bonsai</td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1>0.196</td><td rowspan=1 colspan=1>0.198</td><td rowspan=1 colspan=1>0.205</td><td rowspan=1 colspan=1>0.207</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.194</td></tr><tr><td rowspan=1 colspan=1>counter</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.194</td><td rowspan=1 colspan=1>0.196</td><td rowspan=1 colspan=1>0.204</td><td rowspan=1 colspan=1>0.203</td><td rowspan=1 colspan=1>0.205</td><td rowspan=1 colspan=1>0.189</td></tr><tr><td rowspan=1 colspan=1>kitchen</td><td rowspan=1 colspan=1>0.150</td><td rowspan=1 colspan=1>0.135</td><td rowspan=1 colspan=1>0.130</td><td rowspan=1 colspan=1>0.137</td><td rowspan=1 colspan=1>0.135</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.126</td></tr><tr><td rowspan=1 colspan=1>room</td><td rowspan=1 colspan=1>0.221</td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>0.216</td><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>0.218</td><td rowspan=1 colspan=1>0.203</td></tr></table>

Table 9. The LPIPS scores for all ablation studies in each scene.

<!-- image-->  
Figure 7. Performance under varying Gaussian budgets of 2 scenes.

## A.2. Performance Under Varying Gaussian Budgets

Figure 7 and 8 present the Performance Under Varying Gaussian Budgets across 8 scenes from the Mip-NeRF 360 dataset.   
Due to the inherent high volatility of the T & T and DB datasets, similar comparisons are not conducted for these.

<!-- image-->  
Figure 8. Performance under varying Gaussian budgets of 6 scenes.

## A.3. More Qualitative Comparisons

Figure 9 and 10 present qualitative comparisons of the remaining 10 scenes. Our method achieves optimal rendering results across all scenes.

<!-- image-->  
Figure 9. Qualitative comparison results among scenes bicycle, flowers, stump, treehill, bonsai.

<!-- image-->  
Figure 10. Qualitative comparison results among scenes counter, kitchen, drjohnson, playroom, truck.

## A.4. Derivation of the Finite Prior

This section derives the linear relation between the decay ratio and the current opacity under the finite prior.

Opacity is activated from the pre-activation variable v by the sigmoid function:

$$
\alpha = S ( v ) = \frac { 1 } { 1 + e ^ { - v } } .\tag{14}
$$

Ignoring rendering gradients, the update under environmental pressure is:

$$
\boldsymbol { v } _ { t + 1 } = \boldsymbol { v } _ { t } + \boldsymbol { \nabla } \boldsymbol { v } \cdot \boldsymbol { l r } ,\tag{15}
$$

which yields the decay ratio:

$$
R _ { o } ^ { t + 1 } = \frac { S ( v _ { t } ) - S ( v _ { t } + \nabla v \cdot l r ) } { S ( v _ { t } ) } .\tag{16}
$$

Let $\Delta \boldsymbol { v } = \nabla \boldsymbol { v } \cdot \boldsymbol { l r }$ and assume it is small, i.e., $| \Delta v | \ll 1 .$ . A first-order Taylor expansion of $S ( v _ { t } + \Delta v )$ at $v _ { t }$ gives:

$$
S ( v _ { t } + \Delta v ) \approx S ( v _ { t } ) + S ^ { \prime } ( v _ { t } ) \Delta v .\tag{17}
$$

Since the derivative of the sigmoid is

$$
S ^ { \prime } ( v ) = S ( v ) [ 1 - S ( v ) ] = \alpha ( 1 - \alpha ) ,\tag{18}
$$

we obtain

$$
\begin{array} { r } { S ( v _ { t } + \Delta v ) \approx \alpha _ { t } + \alpha _ { t } ( 1 - \alpha _ { t } ) \Delta v . } \end{array}\tag{19}
$$

Substituting into the decay ratio:

$$
\begin{array} { c } { R _ { o } ^ { t + 1 } = \frac { \alpha _ { t } - S ( v _ { t } + \Delta v ) } { \alpha _ { t } } } \\ { \approx - ( 1 - \alpha _ { t } ) \Delta v . } \end{array}\tag{20}
$$

Environmental pressure enforces $\Delta v < 0$ , so we take the magnitude:

$$
\begin{array} { r } { R _ { o } ^ { t + 1 } \approx \left( 1 - \alpha _ { t } \right) | \Delta \boldsymbol { v } | = \left( 1 - \alpha \right) | \nabla \boldsymbol { v } \cdot \boldsymbol { l r } | . } \end{array}\tag{21}
$$

When $| \Delta v |  0$ , the approximation error is $\mathcal { O } ( | \Delta v | ^ { 2 } )$ , and the first-order expression becomes accurate.

Thus, the decay ratio is linearly proportional to $( 1 - \alpha )$

$\mathbf { A s } \alpha  1$ , the decay ratio vanishes, preserving consistently high-fitness Gaussians;

$\mathrm { { A s } } \alpha  0$ , the decay ratio approaches its maximum $| \boldsymbol { \nabla } \boldsymbol { v } \cdot \boldsymbol { l r } |$ , accelerating the removal of low-fitness individuals.

This finite prior therefore accelerates convergence while maintaining the fairness and adaptivity of natural selection.

## A.5. Pruning based on 3DGS

Our method employs Improved-GS as the densification strategy to achieve the best compact 3DGS rendering quality, but this does not imply that our approach relies on specific prior work. Table 10 also presents the results of our method on 3DGS, with the budget uniformly set to 1/4 of the peak densification budget in 3DGS. Even within the 3DGS framework, our method achieves highly competitive pruning performance.

<table><tr><td>Dataset</td><td colspan="3">Mip-NeRF360</td></tr><tr><td>Method- âMetric</td><td>SSIMâ</td><td> $P S N R ^ { \uparrow }$   $L P I P S ^ { \downarrow }$ </td><td> $N u m ^ { \downarrow }$ </td></tr><tr><td>3DGS</td><td>0.816</td><td>27.50 0.216</td><td>3320453</td></tr><tr><td>Compact-3DGS</td><td>0.807</td><td>27.33 0.227</td><td>1516172</td></tr><tr><td>MaskGS</td><td>0.815</td><td>27.43 0.218</td><td>1582926</td></tr><tr><td>Ours(3DGS)</td><td>0.815</td><td>27.45 0.222</td><td>830000</td></tr></table>

Table 10. Quantitative results based on 3DGS densification method.

## A.6. Opacity Learning Rate Scaling

Necessity. To complete the natural selection process within a limited number of training iterations, a sufficiently strong regularization gradient is required. If the original opacity learning rate is maintained, the selection outcome is heavily influenced by the initial opacity distribution, since the final opacity can be decomposed as:

$$
\alpha _ { \mathrm { f i n a l } } = \alpha _ { \mathrm { i n i t } } + \mathrm { c u m u l a t i v e ~ r e c o v e r y } - \mathrm { c u m u l a t i v e ~ d e c a y } .
$$

Increasing the opacity learning rate during natural selection proportionally amplifies both the recovery and decay terms, thereby increasing the relative influence of dynamic optimization gradients and reducing the dependence on initial opacity.

<table><tr><td>Factors</td><td>SSIMâ PSNRâ</td><td>LPIPSâ</td></tr><tr><td>1Ã</td><td>0.786</td><td>25.71 0.227</td></tr><tr><td>2Ã</td><td>0.790</td><td>25.72 0.217</td></tr><tr><td>3Ã</td><td>0.792</td><td>25.80 0.214</td></tr><tr><td>4Ã</td><td>0.793</td><td>25.80 0.214</td></tr><tr><td>5Ã</td><td>0.792</td><td>25.76 0.213</td></tr></table>

Table 11. The impact of different opacity learning rate scaling factors on the rendering quality of the bicycle scene.

On the bicycle scene, we evaluate different scaling factors (Table 11). A scale of 4Ã already approaches the upper quality bound. However, excessively large learning rates also increase per-step decay, which may violate the assumption that gradients are approximately zero under the finite prior. Thus, we adopt a scaling factor of 4Ã as a balanced choice. After natural selection concludes, we allow the scene an additional 1,000 iterations for opacity recovery, after which the opacity learning rate will revert to its original value.

## A.7. Learning Rate of the Regularization Gradient Field

<table><tr><td>br</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>0.002</td><td>0.773</td><td>25.51</td><td>0.239</td></tr><tr><td>0.0025</td><td>0.792</td><td>25.78</td><td>0.213</td></tr><tr><td>0.003</td><td>0.793</td><td>25.78</td><td>0.213</td></tr><tr><td>0.004</td><td>0.791</td><td>25.78</td><td>0.217</td></tr><tr><td>0.006</td><td>0.787</td><td>25.72</td><td>0.226</td></tr><tr><td>0.008</td><td>0.784</td><td>25.71</td><td>0.232</td></tr></table>

Table 12. The impact of regularization gradient field learning rate on the rendering quality of the bicycle scene.

Effect of the learning rate. The learning rate of the regularization gradient field directly controls the rate at which natural selection proceeds. After selection completes, a short period of fine-tuning is still required. Empirically, allowing natural selection to run for approximately 5Kâ8K iterations yields the highest pruning quality.

We set the latest possible termination point at 23K iterations. If the budget is not met by that point, a one-shot pruning based on opacity is applied. In the bicycle scene, six learning rates are tested (Table 12, Fig. 11). The second-smallest learning rateâused in our main experimentsâenables natural selection to finish just before the latest allowed iteration.

Results show:

â¢ Natural selection lasting 5Kâ8K iterations consistently achieves the best rendering quality.

â¢ Insufficient selection time degrades pruning quality, though moderately.

â¢ If the process does not finish on time and must fall back to one-shot pruning, rendering quality drops substantially due to the harshness and instability of one-shot removal.

Learning rate selection. Across scenes, we observe that higher redundancy corresponds to lower required learning rates. Redundancy depends on scene complexity (resolution, detail density) and the extent to which high Gaussian counts actually improve rendering quality. Indoor scenes usually exhibit lower complexity and redundancy, while higher-resolution scenes contain richer details and thus lower redundancy.

<!-- image-->  
Figure 11. In the bicycle scene, pruning curves corresponding to different regularization learning rates.

## A.8. Automated Learning Rate

By tracking the opacity change curve of Gaussians ranked at the final budget under the optimal learning rate, we observe that the overall trend is linear. Consequently, we can dynamically adjust the learning rate by comparing the opacity of the current target Gaussian with the preset curve. This version was completed after the paper submission and has been included in the code provided in Supplementary Material. Additionally, this version sets the densification budget to three times the final budget, meaning that for each scene, only the final budget parameter needs to be specified. This parameter can be easily automated by adjusting the gradient threshold for densification. However, manual control of the final budget remains a desirable feature, so we have chosen to retain it.

The automated learning rate version achieves rendering quality nearly identical to that of manual parameter tuning. However, since the main text version still employs manual parameter tuning, the results provided in the appendix are also based on manually tuned parameters.

## A.9. FPS Evaluation

<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>ImprovedGS-s</td><td rowspan=1 colspan=1>SPA</td><td rowspan=1 colspan=1>MaskGS</td><td rowspan=1 colspan=1>Opacity Pruning</td><td rowspan=1 colspan=1>Render Pruning</td><td rowspan=1 colspan=1>Edge Pruning</td><td rowspan=1 colspan=1>Natural Selection(Ours)</td></tr><tr><td rowspan=1 colspan=1>FPS</td><td rowspan=1 colspan=1>153</td><td rowspan=1 colspan=1>174</td><td rowspan=1 colspan=1>177</td><td rowspan=1 colspan=1>197</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>153</td><td rowspan=1 colspan=1>193</td></tr></table>

Table 13. FPS under the same budget and with Improved-GS as the shared pre-stage.

Table 13 reports FPS under the same budget and with Improved-GS as the shared pre-stage. Our method achieves slightly higher FPS than Improved-GS.

FPS is primarily determined by the average length of the per-pixel rendering queue. MiniGS significantly improves FPS by employing depth reinitialization and a âmax-contribution-onlyâ retention rule, both of which shorten rendering queues. However:

â¢ depth reinitialization incurs considerable extra optimization time, and

â¢ retaining only the maximum-contribution Gaussian contradicts the objective of preserving the highest rendering quality.

Since our focus is on improving quality in compact 3DGS rather than maximizing FPS, direct comparison with MiniGS is not entirely appropriate. Compared with other pruning methods, our approach yields no noticeable FPS disadvantage while achieving superior quality.

<table><tr><td>Dataset</td><td colspan="4">Mip-NeRF360</td></tr><tr><td>Method- âMetric</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td> $N u m ^ { \downarrow }$ </td></tr><tr><td>MiniGS</td><td>0.822</td><td>27.36</td><td>0.217</td><td>493466</td></tr><tr><td>GaussianSPA-30K</td><td>0.817</td><td>27.31</td><td>0.229</td><td>421427</td></tr><tr><td>GaussianSPA-40K</td><td>0.821</td><td>27.58</td><td>0.224</td><td>421086</td></tr></table>

Table 14. Quantitative results based on 3DGS densification method.

## A.10. Reproduction Issue of GaussianSPA

The GaussianSPA paper states that pruning is based solely on opacity. However, the official implementation instead ranks Gaussians using:

$$
\mathrm { c u r r e n t o p a c i t y } + \mathrm { a c c u m u l a t e d d e c a y o f f s e t } .
$$

In each pruning step, Gaussians within the top budget are exempted, whereas the others undergo decay. The accumulated offset corresponds to past decay, and due to the implemented ranking rule, Gaussians that have undergone multiple decays (i.e., originally low-opacity Gaussians) gradually move forward in the ranking, making them increasingly likely to be retained.

Thus, the implementation effectively becomes:

## opacity pruning + partial random pruning,

introducing unintended stochasticity. When we correct this logic, performance noticeably decreasesâlikely because the stochastic component implicitly mimics the gradient-competition behavior of our natural selection, thereby improving performance âby accident.â

SPAâs true contribution lies in its ADMM-based smooth attenuation mechanism, which is independent of any specific pruning weight. Although a MiniGS-based pruning weight is provided, it requires computing per-view maximum contribution counts, making training several times slower. For fairness, we report the original SPA results in the main paper and explain the discrepancy in the appendix.

Additionally, SPA uses 40K iterations in its paper while all comparison methods use 30K. According to the authors, SPA continues to improve beyond 30K whereas other methods overfit. For fair comparison, we report SPA at 30K in the main paper, and its 40K results are included in Table 14.

<table><tr><td>Method- -Metric</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>Num</td></tr><tr><td>base</td><td>0.792</td><td>25.78</td><td>0.213</td><td></td></tr><tr><td>T2.5 Ã andlr0.4Ã</td><td>0.792</td><td>27.78</td><td>0.212</td><td></td></tr></table>

Table 15. Different combinations of T and learning rate in quality.

## A.11. Effect of the Finite Prior Parameter T

Table 15 presents results obtained using different combinations of T and learning rate. Increasing T is theoretically equivalent to increasing the learning rate because, under the finite-prior formulation:

$$
\nabla v = 2 ( \mathbb { E } [ v ] - T ) .
$$

When $T \gg \mathbb { E } [ v ]$ , the gradient magnitude is dominated by T , making |âv| proportional to |T |.

When T is scaled to 2.5 times its original value, the effective gradient scaling factor becomes:

$$
\frac { 2 . 5 T - \mathbb { E } [ v ] } { T - \mathbb { E } [ v ] } = 2 . 5 + \frac { 1 . 5 \mathbb { E } [ v ] } { T - \mathbb { E } [ v ] } .
$$

Since $\mathbb { E } [ v ] < 0$ in practice (as described in the main text), the second term is positive, meaning the effective growth factor is greater than 2.5.

Thus, if T increases by 2.5Ã while the learning rate is reduced by 2.5Ã, the actual natural-selection speed becomes slightly fasterâconsistent with the observed results in Fig. 12.

<!-- image-->  
Figure 12. Different combinations of T and learning rate in pruning speed.