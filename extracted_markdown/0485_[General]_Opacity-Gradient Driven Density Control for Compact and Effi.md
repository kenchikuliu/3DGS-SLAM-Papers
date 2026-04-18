# Opacity-Gradient Driven Density Control for Compact and Efficient Few-Shot 3D Gaussian Splatting

Abdelrhman Elrawy and Emad A. Mohammed

Department of Computer Science and Physics, Wilfrid Laurier University

75 University Ave W, Waterloo, ON N2L 3C5

Email: elra6860@mylaurier.ca, emohammed@wlu.ca

Abstractâ3D Gaussian Splatting (3DGS) struggles in few-shot scenarios, where its standard adaptive density control (ADC) can lead to overfitting and bloated reconstructions. While stateof-the-art methods like FSGS improve quality, they often do so by significantly increasing the primitive count. This paper presents a framework that revises the core 3DGS optimization to prioritize efficiency. We replace the standard positional gradient heuristic with a novel densification trigger that uses the opacity gradient as a lightweight proxy for rendering error. We find this aggressive densification is only effective when paired with a more conservative pruning schedule, which prevents destructive optimization cycles. Combined with a standard depth-correlation loss for geometric guidance, our framework demonstrates a fundamental improvement in efficiency. On the 3-view LLFF dataset, our model is over 40% more compact (32k vs. 57k primitives) than FSGS, and on the Mip-NeRF 360 dataset, it achieves a reduction of approximately 70%. This dramatic gain in compactness is achieved with a modest trade-off in reconstruction metrics, establishing a new state-of-the-art on the quality-vsefficiency Pareto frontier for few-shot view synthesis.

Index TermsâNovel View Synthesis, 3D Gaussian Splatting, Few-Shot Learning, Adaptive Density Control, Image Reconstruction.

## I. INTRODUCTION

Novel View Synthesis (NVS) aims to generate photorealistic images of a scene from novel camera perspectives. This technology is a cornerstone of 3D computer vision, with applications spanning from virtual and augmented reality to digital content creation. The introduction of Neural Radiance Fields (NeRF) [4] marked a significant leap forward, achieving unprecedented quality. However, the high computational cost of NeRF, which often required hours of training per scene, was a major bottleneck. More recently, 3D Gaussian Splatting (3DGS) [2] revolutionized the field by introducing an explicit, point-based representation that enables real-time rendering without sacrificing quality.

Despite its success, 3D Gaussian Splatting (3DGS) degrades significantly when trained from only a sparse set of input views, a few-shot setting commonly encountered in practical capture scenarios [2], [30], [31].

This regime exposes fundamental limitations in the core 3DGS optimization: the representation readily overfits the observed views, producing floating Gaussians (âfloatersâ) and other artifacts [32], [33].

Furthermore, the standard Adaptive Density Control (ADC) responsible for adding and pruning Gaussian primitives can become unreliable under sparse supervision, leading to under-/over-densification and additional artifacts [2], [13], [12], [34].

To address these challenges, state-of-the-art methods such as FSGS [3] introduced novel geometric regularization and densification strategies. However, these improvements often came at the cost of high model compactness, resulting in a large number of Gaussian primitives. The framework proposed in this paper, illustrated in Figure 1, takes a different approach. Instead of adding new complex modules, the core optimization algorithm of 3DGS is fundamentally revised to prioritize efficiency while maintaining high quality.

This paper proposes a comprehensive framework that revises the densification and pruning logic within 3DGS. The primary contribution of this work is a novel Error-Driven Densification that uses the opacity gradient as a simple and efficient proxy for rendering error. Crucially, this change to densification necessitates a corresponding adjustment to the pruning logic. A critical inefficiency is identified and resolved where the proposed aggressive densifier is nullified by an equally aggressive standard pruning schedule. The solution is to pair the proposed error-driven densification with a more conservative pruning schedule. When combined with a standard depth correlation loss, the proposed framework achieves a dramatic improvement in efficiency over state-ofthe-art methods such as FSGS. The main contributions are:

1) A novel Error-Driven Adaptive Density Control (ADC) framework is introduced that uses the opacity gradient as a simple and efficient proxy for rendering error, avoiding the need for complex auxiliary losses.

2) A critical optimization conflict is identified and resolved, demonstrating through rigorous ablation that the aggressive densifier must be paired with a more conservative pruning schedule to be effective.

3) Establishes a new state-of-the-art on the efficiencyquality Pareto frontier, reducing model compactness by over 40% (32k vs. 57k primitives) against FSGS on the LLFF dataset for a modest trade-off in image quality metrics.

## II. RELATED WORK

## A. Neural Representations for 3D Reconstruction

The recent advancement of neural rendering techniques, such as Neural Radiance Fields (NeRFs) [4], showed encouraging progress for novel view synthesis. NeRF learned an implicit neural scene representation that utilized a Multi-Layer Perceptron (MLP) to map 3D coordinates and viewdependency to color and density through a volume rendering function. A tremendous body of work focused on improving its efficiency [9], [10], quality [1], generalizing to unseen scenes [19], [28], [22], applying artistic effects [21], [29], and 3D generation [24], [23]. In particular, Reiser et al. [26] proposed a method to accelerate NeRFâs training by splitting a large MLP into thousands of tiny MLPs. MVSNeRF [19] constructed a 3D cost volume and rendered high-quality images from novel viewpoints. Moreover, Mip-NeRF [18] adopted conical frustums rather than single rays to mitigate aliasing, and Mip-NeRF 360 [17] further extended this to unbounded scenes. While these NeRF-like models presented strong performance on various benchmarks, they generally required several hours of training time. Muller et al. [10] Â¨ adopted a multiresolution hash encoding technique that reduced training time significantly. Kerbl et al. [2] proposed a 3D Gaussian Splatting pipeline that achieved real-time rendering for both objects and unbounded scenes.

<!-- image-->  
Fig. 1: An overview of the proposed optimization framework. Starting from a sparse set of input views and an initial point cloud from SfM, our method iteratively refines a set of 3D Gaussians. The core optimization loop consists of rendering, loss computation (photometric and geometric), and our revised Adaptive Density Control (ADC). Crucially, densification is driven by the opacity gradient, a direct proxy for rendering error, while a multi-stage pruning strategy removes transparent or redundant primitives. This cycle of rendering, error-driven adaptation, and pruning produces an efficient scene representation.

## B. Novel View Synthesis from Sparse Inputs

The original Neural Radiance Fields (NeRF) was evaluated with approximately 100 input views for synthetic scenes and around 50 images per scene on LLFF, making practical deployment challenging in settings where only a handful of views are available [4], [20].

To reduce the reliance on large numbers of training views, numerous methods aim at few-shot or sparse-view NeRF via learned priors and regularization, including pixelNeRF, RegNeRF, DS-NeRF, and FreeNeRF [28], [5], [20], [7]. When views were sparse, the ill-posed nature of the problem became severe, and NeRF-based methods often resorted to regularization to prevent degenerate solutions. DepthNeRF [20] applied additional depth supervision to improve rendering quality. RegNeRF [5] proposed a depth smoothness loss and appearance regularization by constraining patch renderings from unobserved viewpoints. DietNeRF [8] added supervision on the CLIP embedding space to constrain rendered unseen views. FreeNeRF [7] identified the negative impact of highfrequency signals in positional encodings and proposed a dynamic frequency controlling module for few-shot NeRF. PixelNeRF [28] trained a convolutional encoder to capture context information and learn to predict a 3D representation from sparse inputs. More recently, SparseNeRF [6] proposed a new spatial continuity loss to distill spatial coherence from monocular depth estimators. Concurrent work ReconFusion [27] employed diffusion models to synthesize additional views, though these may not always adhere to view consistency and can be time-consuming. These methods demonstrated that adding external constraints or priors was a dominant strategy for regularizing NeRF in the low-data regime.

## C. Advancements in Few-Shot 3D Gaussian Splatting

The transition to 3DGS brought real-time rendering but did not solve the fundamental few-shot problem; in fact, the reliance on an initial point cloud from Structure-from-Motion (SfM), which was often extremely sparse with few views, exacerbated the issue [5], [11]. Consequently, a new line of research focused on adapting 3DGS for sparse inputs. Many approaches followed the NeRF-based trend of incorporating external priors. Several works used monocular depth estimators to provide geometric supervision and regularize the 3D shape of the scene [5], [11]. FSGS [3], a state-ofthe-art method, combined depth-based regularization with a novel âProximity-guided Gaussian Unpoolingâ densification strategy. Instead of splitting or cloning based on gradients, FSGS inserted new Gaussians between existing ones based on a geometric proximity score, effectively filling gaps in the sparse initial geometry. To further mitigate overfitting, it synthesized pseudo-views and applied a depth correlation loss on both real and virtual views. The proposed framework targets the goal of improving few-shot 3DGS but differs fundamentally in its approach. While FSGS introduces a new geometric densification mechanic, this framework focuses on improving the core optimization algorithm by reformulating the densification trigger and pruning logic to be directly driven by rendering error, aiming for a more efficient and compact scene representation.

## D. Revisiting Adaptive Density Control in 3DGS

The core of 3DGS optimization was its Adaptive Density Control (ADC), a mechanism that dynamically managed the set of Gaussian primitives through densification (adding) and pruning (removing) [2]. This process was critical for transforming an initial sparse point cloud into a high-fidelity scene representation.

The Standard Heuristic and its Limitations. The standard 3DGS method [2] triggered densification when a Gaussianâs average view-space positional gradient magnitude exceeded a threshold. The intuition was that a primitive that moved frequently was likely trying to cover an under-reconstructed area. Pruning was performed concurrently, removing Gaussians whose opacity fell below a near-zero threshold. While effective for dense inputs, this gradient-based heuristic was identified as a key failure point in more challenging scenarios [7]. A recent study critiqued this approach, noting that the positional gradient was âblind to the absolute value of the errorâ and could remain low in high-error regions (e.g., blurry textures), leading to âsubstantial scene underfittingâ [7]. This work directly addresses this failure by proposing a densification mechanism that uses opacity gradients as a more direct proxy for rendering error.

Error-Driven Densification. A more principled line of research aimed to replace the positional gradient heuristic with a direct error signal. One prominent example proposed a pixelerror-driven formulation [1]. This method introduced an auxiliary loss function to attribute per-pixel rendering error back to individual primitives, using this error score as the densification criterion [1]. While this established a direct link between error and densification, it required an auxiliary loss and an extra, non-trainable parameter per Gaussian to facilitate the error attribution [1]. The work presented in this paper falls within this category but proposes using the opacity gradient from the primary photometric loss as a computationally lightweight proxy for error, avoiding implementation overhead.

Geometric and Mechanical Modifications. A complementary research direction focused not on when to densify, but how. FSGS is the key few-shot baseline for the framework presented here. FSGS used âProximity-guided Gaussian Unpooling,â which inserted new primitives based on geometric proximity to neighbors rather than gradients [3]. Other approaches have explored localized point management techniques for optimizing Gaussian distributions [14]. This mechanical improvement is largely orthogonal to this work, which focuses on the densification trigger. The error-driven approach presented here could potentially be combined with such geometric strategies in future work.

In summary, the literature on few-shot view synthesis revealed a clear trend: a reliance on external geometric priors and the introduction of novel, often complex, densification mechanics to regularize the ill-posed problem. While effective, state-of-the-art methods like FSGS often achieved higher quality at the cost of significant model compactness, creating dense models that compromise the efficiency gains of 3DGS. Concurrently, a separate line of inquiry identified the core ADC of 3DGS, particularly its reliance on positional gradients, as a weakness in sparse settings. However, a research gap remained in developing a solution that fundamentally improved the core optimization algorithm for efficiency without introducing significant computational overhead. Furthermore, the critical interplay between densification and pruning in these modified frameworks remained underexplored. This paper aims to fill this gap by proposing a lightweight, error-driven densification mechanism that, when paired with a revised pruning strategy, creates a more synergistic and efficient optimization cycle, leading to highly compact yet high-fidelity scene representations.

## III. METHODOLOGY

The literature review revealed a critical research gap: while state-of-the-art methods like FSGS improved few-shot reconstruction quality, they often did so at the cost of significant model compactness, creating dense models that undermined the efficiency of 3DGS. This led to the central research problem: to formulate a 3DGS optimization framework that prioritizes model compactness under the primary constraint of sparse input views, while maintaining high visual fidelity. This motivated the central research question: How can the core 3DGS optimization algorithm be reformulated to generate geometrically efficient representations from sparse inputs without introducing significant computational overhead?

The hypothesis of this work is that the key lies in creating a more synergistic and intelligent optimization cycle. It is posited that by replacing the unreliable positional gradient heuristic with a direct, error-driven densification trigger (using opacity gradients) and, critically, pairing this aggressive densification with a deliberately conservative pruning schedule, it is possible to prevent the inefficient âcreate-destroyâ cycles that lead to model bloat. The objective, therefore, is to establish a new state-of-the-art on the efficiency-quality Pareto frontier.

To validate this hypothesis, a framework embodying these principles is constructed and evaluated quantitatively. Success is measured by a dual-metric approach: the number of Gaussian primitives as a proxy for model complexity and standard image reconstruction metrics (PSNR, SSIM, LPIPS). An essential component of the validation is a rigorous ablation study. By systematically isolating each component of the revised ADC, its individual contribution is analyzed, proving that the synergy between the proposed densification and pruning strategies is the primary driver of the frameworkâs performance. This analysis justifies the final proposed configuration and validates its effectiveness in achieving the research objective. The section begins with a brief review of the 3DGS framework.

3D Gaussian Splatting Preliminaries. 3DGS [2] represents a scene with a collection of 3D Gaussians, each defined by a position (mean) $\mu \in \mathbb { R } ^ { 3 }$ , a covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ , a color defined by Spherical Harmonics (SH) coefficients, and an opacity $\alpha .$ To render an image, the 3D Gaussians are projected into 2D and then blended together using alpha-blending:

Algorithm 1 Proposed Few-Shot 3DGS Optimization   
1: Input: Training views {I}, camera poses $\{ P \}$ , initial   
Gaussians G   
2: Output: Optimized Gaussians $\mathcal { G } _ { f i n a l }$   
3: for $k = 1$ to max iterations do   
4: Render images $\{ \hat { I } \}$ from $\mathcal { G }$   
5: Compute total loss L using Eq. 3   
6: Accumulate gradients and update $\mathcal { G }$ via Adam   
7: Track max opacity gradient $\nabla _ { \alpha } ^ { m a x }$ for each Gaussian   
8: if k is a densification step then   
9: $\mathcal { G } _ { d e n s i f y }  \{ g \in \mathcal { G } \mid \nabla _ { \alpha } ^ { m a x } ( g ) > \tau _ { d e n s i f y } \}$   
10: Densify primitives in $\mathcal { G } _ { d e n s i f y }$ (Sec. III-A)   
11: Reset $\bar { \nabla } _ { \alpha } ^ { m a x }$ for all Gaussians   
12: end if   
13: if k is a pruning step then   
14: Prune Gaussians with $\alpha < \tau _ { p r u n e }$ (Sec. III-B)   
15: Prune Gaussians to enforce budget $N _ { m a x }$ (Sec. III-B)   
16: end if   
17: end for   
18: return $\mathcal { G }$

$$
C = \sum _ { i \in N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )\tag{1}
$$

where C is the final pixel color, and the product is over the ordered list of Gaussians that overlap the pixel. The optimization is driven by a photometric loss, typically a combination of L1 and D-SSIM (a structural dissimilarity metric calculated as $\frac { 1 - S S I M } { 2 } )$ , between the rendered image and the ground truth.

The proposed framework revises the Adaptive Density Control (ADC) mechanism within this pipeline. The complete optimization process is outlined in Algorithm 1. It consists of two main components, detailed in the following sections: 1) a novel error-driven densification mechanism that uses opacity gradients as a direct proxy for reconstruction error, and 2) a multi-stage pruning strategy that combines a delayed, conservative schedule with a hard budget on the total number of primitives.

## A. Error-Driven Densification via Opacity Gradient

The standard 3DGS pipelineâs reliance on view-space positional gradients for densification is a key limitation in sparse settings, as this metric is an indirect and often unreliable proxy for rendering error. To address this, a more direct, error-driven mechanism is proposed.

Error Attribution via Opacity Gradient: Instead of introducing complex auxiliary losses, an existing signal is leveraged: the gradient of each Gaussianâs opacity with respect to the photometric loss. The intuition is that if a Gaussianâs opacity is suboptimal for minimizing rendering error, the loss gradient with respect to its opacity, $\frac { \partial \mathcal { L } } { \partial \alpha _ { k } }$ , will have a large magnitude. This gradient serves as an efficient proxy for the primitiveâs contribution to the reconstruction error. During optimization, the maximum magnitude of this gradient is tracked for each Gaussian between densification cycles. A primitive is marked for densification if this maximum accumulated error score surpasses a defined threshold. This single error metric replaces the positional gradient heuristic for both cloning and splitting operations.

Principled Opacity Correction for Cloning: When a small Gaussian is cloned, the opacity of both the original and the new primitive must be adjusted to maintain their combined transparency and avoid biasing the alpha-compositing. The correction proposed in [1] is adopted, where the opacity Î± is adjusted to $\alpha _ { \mathrm { n e w } }$ such that $( 1 - \alpha ) \ : = \ : ( 1 - \alpha _ { \mathrm { n e w } } ) ^ { 2 }$ . This yields the corrected update rule:

$$
\alpha _ { \mathrm { n e w } } = 1 - \sqrt { 1 - \alpha }\tag{2}
$$

Both the original and the cloned Gaussian are assigned this new opacity value.

## B. A Multi-Stage Pruning Strategy for Compact Models

Aggressive, error-driven densification can be counterproductive if not paired with a compatible pruning schedule. The standard 3DGS pruning strategy, which starts early and is aggressive, can remove newly created primitives before they are optimized, leading to a destructive âcreate-destroyâ cycle. It was found that a more deliberate, multi-stage pruning strategy is required to achieve a compact and high-quality final model.

Delayed and Conservative Pruning: The first component of the presented strategy addresses a critical issue where newly densified primitives are removed before they can be properly optimized. When new Gaussians are created, they may initially have low opacity values. The standard pruning schedule starts early in training (at iteration 500) and uses a relatively high opacity threshold (0.005), which can inadvertently remove these potentially useful new primitives. To prevent this, a delayed and conservative schedule is employed. First, the onset of pruning is delayed until iteration 2,000. This provides newly created Gaussians with sufficient optimization steps to adjust their parameters and contribute meaningfully to the scene representation. Second, a more conservative opacity threshold of 0.001 is used, ensuring that only Gaussians with extremely low opacity are considered for removal. This combined approach ensures that the densification process is not undermined by premature pruning. The specific values for the delay and threshold were determined empirically; as shown in the ablation study (Table III), reverting to a more aggressive, standard pruning schedule significantly degrades reconstruction quality.

Enforcing a Primitive Budget: To ensure a compact final model and prevent uncontrolled growth, the second component of the strategy is to enforce a hard budget on the total number of Gaussians. If the number of primitives exceeds this budget after a densification step, an additional pruning pass is performed. In this pass, the number of excess primitives is identified and that many Gaussians with the lowest opacity values are pruned. This ensures the model remains within the predefined complexity budget throughout training.

This two-pronged pruning strategy, combining a delayed, conservative schedule with a hard primitive budget, works in synergy with the aggressive densifier presented in this work. The primary goal of this approach is to produce a more efficient and compact geometric representation, quantified by a significant reduction in the total number of Gaussian primitives. A lower primitive count directly translates to tangible performance benefits, including faster rendering speeds (FPS) and a smaller memory footprint. The central hypothesis of this paper is that by fundamentally improving the core optimization algorithm, this dramatic gain in compactness can be achieved with only a modest and perceptually acceptable trade-off in standard image quality metrics. While metrics like Peak Signal-to-Noise Ratio (PSNR) provide a useful quantitative measure, they do not always perfectly correlate with human perception of visual fidelity. Therefore, a small decrease in PSNR can be a highly favorable exchange for substantial improvements in model efficiency, establishing a more practical operating point on the quality-versus-efficiency Pareto frontier.

## C. Geometric Regularization and Overall Loss

To guide the reconstruction in the data-sparse regime, a geometric prior is incorporated using estimated depth maps. The total loss function combines a photometric loss with a depth correlation term:

$$
\mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } + w _ { \mathrm { d e p t h } } \mathcal { L } _ { \mathrm { d e p t h } }\tag{3}
$$

where $\mathcal { L } _ { 1 }$ and ${ \mathcal { L } } _ { \mathrm { D - S S I M } }$ are the standard L1 and D-SSIM photometric losses from 3DGS [2], and $w _ { \mathrm { d e p t h } }$ is a weighting hyperparameter for the depth regularization term ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ . This depth loss encourages consistency between the rendered depth $\mathbf { d } _ { \mathrm { { r e n d e r } } }$ and an estimated monocular depth map $\mathbf { d } _ { \mathrm { { e s t } } }$ . The Pearson correlation coefficient is used, which is robust to scale and shift differences. By minimizing a loss of 1 â correlation, the optimization is driven to maximize the correlation between the two maps. This forces the rendered depth to adopt the same relative geometric structure as the estimateâif one pixel is further than another in the estimated map, the loss penalizes the model if this relationship is not preserved in the rendered map.

$$
\mathcal { L } _ { \mathrm { d e p t h } } = 1 - \frac { \mathrm { C o v } ( \mathbf { d } _ { \mathrm { r e n d e r } } , \mathbf { d } _ { \mathrm { e s t } } ) } { \sqrt { \mathrm { V a r } ( \mathbf { d } _ { \mathrm { r e n d e r } } ) \mathrm { V a r } ( \mathbf { d } _ { \mathrm { e s t } } ) } }\tag{4}
$$

where Cov is the covariance and Var is the variance. The optimization proceeds by interleaving steps of gradient descent on this total loss with the revised ADC mechanism.

## IV. EXPERIMENTAL RESULTS

A series of experiments was conducted to validate the proposed framework, focusing on both reconstruction quality and model efficiency.

## A. Experimental Setup

Datasets. The framework is evaluated on two standard fewshot NVS benchmarks. For the Local Light Field Fusion (LLFF) [16] dataset, which consists of eight forward-facing real-world scenes, the standard evaluation protocol from prior work is adopted. The test set split from RegNeRF [5] is used, which selects every eighth image for testing. Following FSGS [3], 3 of the remaining images are used for training and evaluated on image resolutions downsampled by 4x (1008x756) and 8x (504x378). The Mip-NeRF 360 [17] dataset consists of nine complex outdoor scenes. For this benchmark, 24 training views are used and the same testing split as for LLFF is followed.

Comparison to Baseline Methods. The proposed framework is compared against FSGS [3], a state-of-the-art method for few-shot 3DGS. To provide context, results from the original 3DGS [2] implementation using its standard denseview training configuration are also included. The proposed framework is designed to be competitive with other leading few-shot methods such as DietNeRF [8], RegNeRF [5], FreeNeRF [7], and SparseNeRF [6].

Evaluation Metrics. Performance is evaluated using standard image quality metrics: Peak Signal-to-Noise Ratio (PSNR) â, Structural Similarity Index (SSIM) â, and Learned Perceptual Image Patch Similarity (LPIPS) â [15]. Frames Per Second (FPS) â is also reported as a measure of rendering speed.

Implementation Details. The initial point cloud is computed from Structure-from-Motion (SfM) using only the training views. For geometric regularization, depth maps estimated from a pre-trained DPT model [25] are used. The total optimization is set to 10,000 iterations for all experiments, which were run on a single NVIDIA RTX A6000 GPU.

## B. Main Quantitative Results

A comprehensive quantitative comparison is presented against state-of-the-art few-shot view synthesis methods on the LLFF and Mip-NeRF 360 datasets. The results, detailed in Table I and Table II, demonstrate that the proposed framework establishes a new, highly competitive operating point on the quality-vs-efficiency Pareto frontier.

LLFF Dataset. The results on the 3-view LLFF dataset validate the central hypothesis of this work. As shown in Table I, the revised ADC yields a model that is over 40% more compact than the state-of-the-art FSGS (32k vs. 57k primitives). This efficiency gain stems from the error-driven densifier, which avoids model bloat by placing primitives more judiciously, and the conservative pruner, which allows them to mature. Crucially, the proposed framework improves perceptual quality over FSGS (e.g., a 10.8% improvement in LPIPS) with a comparable PSNR (20.00 vs. 20.31). This result establishes a new, more efficient position on the qualityvs-efficiency Pareto frontier, confirming that the proposed framework is an effective strategy for generating lightweight yet high-quality representations in data-sparse regimes.

TABLE I: Quantitative Comparison on the LLFF Dataset (3 Training Views). The proposed framework achieves results competitive with the state-of-the-art across both resolutions while using a significantly more compact scene representation and enabling faster rendering.
<table><tr><td rowspan="2">Method</td><td colspan="4">1/8 Resolution (504x378)</td><td colspan="4">1/4 Resolution (1008x756)</td></tr><tr><td>FPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>FPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Mip-NeRF</td><td>0.21</td><td>16.11</td><td>0.401</td><td>0.460</td><td>0.14</td><td>15.22</td><td>0.351</td><td>0.540</td></tr><tr><td>DietNeRF</td><td>0.14</td><td>14.94</td><td>0.370</td><td>0.496</td><td>0.08</td><td>13.86</td><td>0.305</td><td>0.578</td></tr><tr><td>RegNeRF</td><td>0.21</td><td>19.08</td><td>0.587</td><td>0.336</td><td>0.14</td><td>18.06</td><td>0.535</td><td>0.411</td></tr><tr><td>FreeNeRF</td><td>0.21</td><td>19.63</td><td>0.612</td><td>0.308</td><td>0.14</td><td>18.73</td><td>0.562</td><td>0.384</td></tr><tr><td>SparseNeRF</td><td>0.21</td><td>19.86</td><td>0.624</td><td>0.328</td><td>0.14</td><td>19.07</td><td>0.564</td><td>0.401</td></tr><tr><td>3DGS</td><td>385</td><td>17.43</td><td>0.522</td><td>0.321</td><td>312</td><td>16.94</td><td>0.488</td><td>0.402</td></tr><tr><td>FSGS</td><td>458</td><td>20.31</td><td>0.652</td><td>0.288</td><td>351</td><td>19.88</td><td>0.612</td><td>0.340</td></tr><tr><td>Proposed</td><td>719</td><td>20.00</td><td>0.680</td><td>0.257</td><td>321</td><td>19.55</td><td>0.652</td><td>0.336</td></tr></table>

TABLE II: Quantitative Comparison on the Mip-NeRF 360 Dataset (24 Training Views). The proposed framework continues to demonstrate a strong balance of quality and efficiency in more complex, large-scale scenes.
<table><tr><td rowspan="2">Method</td><td colspan="4">1/8 Resolution</td><td colspan="4">1/4 Resolution</td></tr><tr><td>FPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>FPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Mip-NeRF 360</td><td>0.12</td><td>21.23</td><td>0.613</td><td>0.351</td><td>0.07</td><td>19.78</td><td>0.530</td><td>0.431</td></tr><tr><td>DietNeRF</td><td>0.05</td><td>20.21</td><td>0.557</td><td>0.387</td><td>0.03</td><td>19.11</td><td>0.482</td><td>0.452</td></tr><tr><td>RegNeRF</td><td>0.07</td><td>22.19</td><td>0.643</td><td>0.335</td><td>0.04</td><td>20.55</td><td>0.546</td><td>0.398</td></tr><tr><td>FreeNeRF</td><td>0.07</td><td>22.78</td><td>0.689</td><td>0.323</td><td>0.04</td><td>21.04</td><td>0.587</td><td>0.377</td></tr><tr><td>SparseNeRF</td><td>0.07</td><td>22.85</td><td>0.693</td><td>0.315</td><td>0.04</td><td>21.13</td><td>0.600</td><td>0.389</td></tr><tr><td>3DGS</td><td>223</td><td>20.89</td><td>0.633</td><td>0.317</td><td>145</td><td>19.93</td><td>0.588</td><td>0.401</td></tr><tr><td>FSGS</td><td>290</td><td>23.70</td><td>0.745</td><td>0.220</td><td>203</td><td>22.82</td><td>0.693</td><td>0.293</td></tr><tr><td>Proposed</td><td>475</td><td>23.26</td><td>0.715</td><td>0.284</td><td>464</td><td>22.72</td><td>0.694</td><td>0.338</td></tr></table>

Mip-NeRF 360 Dataset. The Mip-NeRF 360 dataset, with its higher view count (24), tests the frameworkâs generalizability beyond extreme few-shot scenarios. On these largescale, unbounded scenes, the proposed framework remains competitive with the state-of-the-art, achieving a PSNR of 23.26, comparable to the FSGS baseline (Table II). This result is significant, as it shows the efficiency-focused ADC does not hinder performance when more data is available. Our framework reduces model compactness by approximately 70% on average across these scenes, demonstrating that the compactness gains are consistent beyond the extreme few-shot LLFF setting.

## C. Rendering Efficiency (FPS)

Frames Per Second (FPS) directly reflects rendering throughput and is the practical payoff of a compact representation. On the 3-view LLFF dataset at 1/8 resolution, the proposed method attains 719 FPS, a 1.57 Ã improvement over FSGS (458 FPS) and 1.87 Ã over 3DGS (385 FPS) (Table I). At 1/4 resolution, our FPS remains competitive (321 vs. 351 for FSGS and 312 for 3DGS). The smaller margin at higher resolution suggests that the pipeline becomes increasingly pixel-bound, so the advantage from fewer primitives translates less directly to throughput; nevertheless, we preserve the most compact model while maintaining real-time speed.

On the larger Mip-NeRF 360 dataset, the efficiency gains are more pronounced. At 1/8 resolution we reach 475 FPS (1.64 Ã over FSGS at 290 FPS; 2.13 Ã over 3DGS at 223 FPS), and at 1/4 resolution we achieve 464 FPS (2.29 Ã over FSGS at 203 FPS; 3.20 Ã over 3DGS at 145 FPS) (Table II). These results corroborate the central claim that the substantial reduction in Gaussian primitives translates to significantly higher rendering throughput and lower memory footprint. All timings were measured on a single NVIDIA RTX A6000; absolute values vary with hardware, but the relative trends are consistent.

## D. Ablation Study

To validate the frameworkâs design, a rigorous ablation study was conducted on the LLFF dataset. This dataset was chosen for its status as a standard benchmark in few-shot view synthesis and its challenging real-world scenes, providing a robust testbed to evaluate performance under sparse supervision. The study tests the central hypothesis that the synergy between error-driven densification and conservative pruning is critical for achieving both model compactness and high reconstruction quality. The results in Table III substantiate this claim.

The analysis is as follows: The most critical comparison is between the full framework and the Proposed w/o Conservative Pruning configuration. The latter yields the worst reconstruction quality across all metrics, despite producing the most compact model. This confirms the hypothesis of an inefficient âcreate-destroyâ cycle, where well-placed new primitives are removed before they can be optimized, catastrophically harming the final reconstruction. Conversely, the Proposed w/o Error-Driven Densification configuration, which applies conservative pruning to the standard 3DGS densifier, results in a massive increase in the number of primitives without a proportional improvement in quality. This highlights that a simple change in pruning is not enough without a more intelligent densification strategy. Furthermore, removing the depth-correlation loss (Proposed w/o Depth Loss) results in a noticeable degradation in quality, demonstrating that while the core ADC improvements provide a strong foundation, geometric guidance is crucial for achieving the best results in a few-shot setting. Ultimately, the full framework demonstrates the power of all components working in synergy. By pairing the error-driven densifier with the enabling conservative pruning schedule and depth-correlation loss, a strong PSNR is maintained while achieving a highly compact and efficient representation. These results prove that the components of the proposed framework are synergistic. The error-driven densifier is highly effective at identifying regions needing refinement, but it is the conservative pruning schedule that provides the necessary âgrace periodâ for these new primitives to mature.

Ours  
Ground Truth  
FSGS  
<!-- image-->  
(a) LLFF Flower (Best Case)

<!-- image-->  
(b) Mip-NeRF 360 Garden (Best Case)  
Fig. 2: Best-case qualitative results. Our framework produces high-fidelity reconstructions on diverse scenes while maintaining a highly compact model, demonstrating the effectiveness of the efficiency-focused optimization cycle.

TABLE III: Ablation study on the LLFF (3-view) dataset. The impact of each component of the proposed framework is analyzed. The full framework demonstrates the best balance of quality and efficiency.
<table><tr><td>Configuration</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td># Points (Avg) â</td></tr><tr><td>Proposed w/o Conservative Pruning</td><td>19.51</td><td>0.638</td><td>0.348</td><td>28,487</td></tr><tr><td>Proposed w/o Error-Driven Densification</td><td>20.33</td><td>0.709</td><td>0.199</td><td>978,937</td></tr><tr><td>Proposed w/o Depth Loss</td><td>19.84</td><td>0.662</td><td>0.272</td><td>30,337</td></tr><tr><td>Proposed (Full Framework)</td><td>20.00</td><td>0.680</td><td>0.257</td><td>32,000</td></tr></table>

## E. Qualitative Results

To provide a comprehensive understanding of our frameworkâs performance, we present a detailed qualitative analysis across a spectrum of scenes, categorized into best-case, marginal, and poor outcomes. This analysis directly connects the visual results to our core contributions: the error-driven densification mechanism, the conservative pruning schedule, and the trade-offs between model complexity and image fidelity.

1) Best-Case Performance: Figure 2 illustrates our frameworkâs performance in ideal scenarios. In scenes with good initialization from SfM and distinct geometric features, such as the âflowerâ scene from LLFF and the âgardenâ scene from Mip-NeRF 360, our framework excels. The error-driven densification correctly identifies regions of high reconstruction error, such as object silhouettes and textured surfaces, allocating new primitives efficiently. The conservative pruning schedule allows these new Gaussians to optimize their positions, shapes, and colors, contributing to a sharp, detailed final render. The resulting model is highly compact, containing only the necessary primitives to represent the scene, which is a direct consequence of our efficiency-focused optimization cycle. The low error maps in these cases confirm that high visual fidelity is achieved with a fraction of the primitive count compared to FSGS.

2) Marginal-Case Performance: Figure 3 shows more challenging scenarios. In scenes characterized by very fine, repetitive textures (âroomâ) or complex, semi-transparent surfaces (âcounterâ), our framework produces results that are quantitatively strong but may lack the high-frequency detail of more complex models like FSGS. This outcome is a direct tradeoff made by our optimization strategy. The opacity-gradientdriven densification prioritizes larger areas of error, and with a strict primitive budget, it may not allocate enough Gaussians to capture every minute detail. The error maps highlight these subtle differences, often localized in areas of complex material properties. This demonstrates a core principle of our framework: we accept a modest and often imperceptible decrease in fidelity on challenging textures in exchange for a significant reduction in model complexity and rendering time.

3) Failure Cases and Limitations: Figure 4 presents failure cases, which are critical for understanding the limitations of our framework. Our frameworkâs performance is fundamentally dependent on the quality of the initial geometric priors from SfM and the monocular depth estimator. In scenes with severe challenges, such as the ambiguous geometry in the âleavesâ scene or reflective surfaces in the âcounterâ scene, the initial point cloud can be sparse and inaccurate. In these situations, our error-driven densification can be misled. If the depth priors are incorrect, the opacity gradients may point to empty space, causing the framework to allocate primitives (âfloatersâ) that do not correspond to real-world geometry, as shown in the error maps. While our conservative pruning mitigates this to some extent, it cannot fully compensate for a fundamentally flawed geometric initialization. This demonstrates that while our optimization is more efficient, it is not a complete solution for the ill-posed nature of few-shot reconstruction and remains sensitive to the quality of its inputs.

Ours  
<!-- image-->

Ground Truth  
<!-- image-->  
(a) LLFF Room (Marginal Case)

FSGS  
<!-- image-->

Ours  
<!-- image-->

Ground Truth  
<!-- image-->

FSGS  
<!-- image-->  
(b) Mip-NeRF 360 Counter (Marginal Case)

Fig. 3: Marginal-case qualitative results. In scenes with challenging textures or materials, our framework makes a direct tradeoff, accepting minor losses in high-frequency detail for significant gains in model compactness and rendering efficiency.  
<!-- image-->  
(a) LLFF Leaves (Poor Case)

Ours  
<!-- image-->

Ground Truth  
<!-- image-->

FSGS  
<!-- image-->  
(b) Mip-NeRF 360 Counter (Poor Case)  
Fig. 4: Poor-performance (failure case) results. These examples illustrate the frameworkâs limitations. When faced with highly ambiguous geometry or poor initial priors from SfM, our method can produce artifacts such as floaters or geometric inaccuracies.

## V. CONCLUSION, LIMITATION AND FUTURE WORK

This paper addressed the research question of how to reformulate the core 3DGS algorithm for geometric efficiency in sparse-view scenarios. We hypothesized that a synergistic redesign of the Adaptive Density Control, pairing an error-driven densifier with a conservative pruner, could prevent model bloat and produce more compact representations. Our results validate this hypothesis. The key contribution is the empirical demonstration of the critical interplay between densification and pruning; our proposed ADC creates an optimization cycle that is fundamentally more efficient in data-sparse regimes. By integrating this improved core algorithm with a standard depthcorrelation loss, our final framework establishes a new stateof-the-art on the efficiency-quality Pareto frontier, achieving a model compactness reduction of over 40% on the LLFF dataset and approximately 70% on the Mip-NeRF 360 dataset, with only a modest trade-off in image quality. This work provides a more principled and efficient foundation for fewshot novel view synthesis.

Limitations. The frameworkâs most dramatic efficiency gains are observed in extremely sparse settings like the 3- view LLFF dataset. While it remains competitive in less sparse scenarios, the compactness benefits may be less pronounced. Furthermore, the framework relies on geometric priors from an external monocular depth estimator, which is a common dependency but one that prevents a fully end-to-end solution.

Finally, the optimal ADC hyperparameters (e.g., pruning delay and thresholds) were determined empirically and may require tuning for different datasets or application requirements.

Future Work. The findings of this paper open several promising avenues for future research. A primary direction is the integration of the error-driven densification trigger with alternative densification mechanics, such as the proximitybased unpooling from FSGS. This could potentially combine the benefits of intelligent placement with geometrically aware growth. Another key area is the development of a dynamic, or even learned, pruning schedule that can adapt to the scene complexity and training progress, removing the need for fixed hyperparameters. Finally, exploring methods to reduce or eliminate the reliance on external geometric priors would be a significant step towards a more robust and self-contained system for few-shot view synthesis.

## REFERENCES

[1] S. Rota Bulo, L. Porzi, and P. Kontschieder, âRevising Densification in\` Gaussian Splatting,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D GaussianÂ¨ Splatting for Real-Time Radiance Field Rendering,â ACM Trans. Graph., vol. 42, no. 4, 2023.

[3] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, âFSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.

[4] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020.

[5] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. M. Sajjadi, A. Geiger, and N. Radwan, âRegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[6] G. Wang, Z. Chen, C. C. Loy, and Z. Liu, âSparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis,â in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2023.

[7] J. Yang, M. Pavone, and Y. Wang, âFreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[8] A. Jain, M. Tancik, and P. Abbeel, âPutting NeRF on a Diet: Semantically Consistent Few-shot View Synthesis,â in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2021.

[9] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, âTensoRF: Tensorial Radiance Fields,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022.

[10] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant Neural Graphics Â¨ Primitives with a Multiresolution Hash Encoding,â ACM Trans. Graph., vol. 41, no. 4, 2022.

[11] A. Paliwal, W. Ye, J. Xiong, D. Kotovenko, R. Ranjan, V. Chandra, and N. K. Kalantari, âCoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.

[12] G. Grubert, F. Barthel, A. Hilsmann, and P. Eisert, âImproving Adaptive Density Control for 3D Gaussian Splatting,â in Proc. Int. Conf. on Computer Vision Theory and Applications (VISAPP), 2025.

[13] Z. Zhang, W. Hu, Y. Lao, T. He, and H. Zhao, âPixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.

[14] H. Yang, C. Zhang, W. Wang, M. Volino, A. Hilton, L. Zhang, and X. Zhu, âImproving Gaussian Splatting with Localized Points Management,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025.

[15] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe Unreasonable Effectiveness of Deep Features as a Perceptual Metric,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018.

[16] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ramamoorthi, R. Ng, and A. Kar, âLocal Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines,â ACM Trans. Graph., vol. 38, no. 4, 2019.

[17] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[18] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields,â in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2021.

[19] A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su, âMVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo,â in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2021.

[20] K. Deng, A. Liu, J. Y. Zhu, and D. Ramanan, âDepth-supervised NeRF: Fewer views and faster training for free,â arXiv:2107.02791, 2021.

[21] Z. Fan, Y. Jiang, P. Wang, X. Gong, D. Xu, and Z. Wang, âUnified Implicit Neural Stylization,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022.

[22] M. M. Johari, Y. Lepoittevin, and F. Fleuret, âGeoNeRF: Generalizing NeRF with Geometry Priors,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[23] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, âMagic3D: High-Resolution Text-to-3D Content Creation,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[24] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, âDreamFusion: Textto-3D using 2D Diffusion,â arXiv:2209.14988, 2022.

[25] R. Ranftl, A. Bochkovskiy, and V. Koltun, âVision Transformers for Dense Prediction,â in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2021.

[26] C. Reiser, S. Peng, Y. Liao, and A. Geiger, âKiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs,â in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2021.

[27] R. Wu, B. Mildenhall, P. Henzler, K. Park, R. Gao, D. Watson, P. P. Srinivasan, D. Verbin, J. T. Barron, B. Poole, and A. Holynski, âRecon-Fusion: 3D Reconstruction with Diffusion Priors,â arXiv:2312.02981, 2023.

[28] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, âpixelNeRF: Neural Radiance Fields from One or Few Images,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021.

[29] K. Zhang, N. Kolkin, S. Bi, F. Luan, Z. Xu, E. Shechtman, and N. Snavely, âARF: Artistic Radiance Fields,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022.

[30] H. Xiong, S. Muttukuru, R. Upadhyay, P. Chari, and A. Kadambi, âSparseGS: Real-Time 360Â° Sparse View Synthesis using Gaussian Splatting,â arXiv:2312.00206, 2023.

[31] S. Chen, J. Zhou, and L. Li, âOptimizing 3D Gaussian Splatting for Sparse Viewpoint Scene Reconstruction,â arXiv:2409.03213, 2024.

[32] J. Chung, J. Oh, and K. M. Lee, âDepth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images,â arXiv:2311.13398, 2023.

[33] L. Wang, Q. Ren, K. Liao, H. Wang, Z. Chen, and Y. Tang, âStableGS: A Floater-Free Framework for 3D Gaussian Splatting,â arXiv:2503.18458, 2025.

[34] P. Wang, Y. Wang, D. Wang, S. Mohan, Z. Fan, L. Wu, R. Cai, Y.-Y. Yeh, Z. Wang, Q. Liu, and R. Ranjan, âSteepest Descent Density Control for Compact 3D Gaussian Splatting,â arXiv:2505.05587, 2025.