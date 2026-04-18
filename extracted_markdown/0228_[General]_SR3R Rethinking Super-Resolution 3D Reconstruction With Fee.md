# SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting

Xiang Feng1 2â \* Xiangbo Wang1\* Tieshi Zhong 1 Chengkai Wang1 Yiting Zhao 1 Tianxiang Xu 4 Zhenzhong Kuang 1 â¡ Feiwei Qin 1 Xuefei Yin 3 Yanming Zhu 3 â¡ 1Hangzhou Dianzi University 2ShanghaiTech University 3Griffith University 4Peking University https://xiangfeng66.github.io/SR3R/

<!-- image-->  
(a) Novel Paradigm: directly predict HR 3DGS parameters

<!-- image-->

<!-- image-->  
(b) Novel Objective: learn generalized mapping function ??

<!-- image-->

<!-- image-->

<!-- image-->

DepthSplat  
<!-- image-->

<!-- image-->  
Figure 1. We reformulate 3DGS-based 3DSR as a feed-forward mapping problem from sparse LR views to HR 3DGS representation. (a) Unlike existing methods that rely on dense multi-view inputs and per-scene 3DGS self-optimization, our method directly predicts HR 3DGS by a learned network from as few as two LR views. (b) This reformulation fundamentally changes how 3DSR acquires high-frequency knowledge. Instead of inheriting the limited priors embedded in 2DSR models, our SR3R learns a generalized crossscene mapping function from large-scale multi-scene data, enabling the network to autonomously acquire the 3D-specific high-frequency structures required for accurate HR 3DGS reconstruction. The bottom row illustrates that our SR3R produces significantly sharp and faithful reconstructions.

## Abstract

3D super-resolution (3DSR) aims to reconstruct highresolution (HR) 3D scenes from low-resolution (LR) multiview images. Existing methods rely on dense LR inputs and per-scene optimization, which restricts the highfrequency priors for constructing HR 3D Gaussian Splatting (3DGS) to those inherited from pretrained 2D superresolution (2DSR) models. This severely limits reconstruction fidelity, cross-scene generalization, and real-time usability. We propose to reformulate 3DSR as a direct feedforward mapping from sparse LR views to HR 3DGS representations, enabling the model to autonomously learn

3D-specific high-frequency geometry and appearance from large-scale, multi-scene data. This fundamentally changes how 3DSR acquires high-frequency knowledge and enables robust generalization to unseen scenes. Specifically, we introduce SR3R, a feed-forward framework that directly predicts HR 3DGS representations from sparse LR views via the learned mapping network. To further enhance reconstruction fidelity, we introduce Gaussian offset learning and feature refinement, which stabilize reconstruction and sharpen high-frequency details. SR3R is plug-and-play and can be paired with any feed-forward 3DGS reconstruction backbone: the backbone provides an LR 3DGS scaffold, and SR3R upscales it to an HR 3DGS. Extensive experiments across three 3D benchmarks demonstrate that SR3R surpasses state-of-the-art (SOTA) 3DSR methods and achieves strong zero-shot generalization, even outperforming SOTA per-scene optimization methods on unseen scenes.

## 1. Introduction

3D super-resolution (3DSR) aims to reconstruct highresolution (HR) 3D representations from low-resolution (LR) multi-view observations. This task has become increasingly critical because state-of-the-art 3D Gaussian Splatting (3DGS)âbased reconstruction methods [11] typically require dense and high-resolution input views to recover fine geometric and appearance details. However, in real-world scenarios, obtaining such high-quality observations is often infeasible due to sensor resolution limits, constrained capture conditions, and storage or bandwidth restrictions [9, 23]. These practical limitations motivate the development of 3DSR methods capable of lifting sparse and LR inputs to high-fidelity 3D representations.

Current 3DSR methods [7, 12, 21, 36] typically employ pretrained 2D image or video super-resolution (2DSR) models to generate pseudo-HR images from dense multiview LR inputs, which are then used as supervision for perscene optimization of HR 3DGS. Although this strategy injects high-frequency cues into the HR 3DGS reconstruction, it suffers from several fundamental limitations. First, perscene optimization isolates each scene as an independent problem and restricts the source of high-frequency knowledge to the priors embedded in pretrained 2DSR models. This prevents leveraging large-scale cross-scene data to learn 3D-specific SR priors and to train a generalized 3DSR model, thereby inherently limiting reconstruction fidelity, cross-scene generalization, and real-time usage. Second, reliance on 2DSR-generated pseudo-HR labels inherently caps the achievable reconstruction fidelity. Third, dense multi-view synthesis and iterative optimization introduce substantial computational and data overhead.

To address these limitations, we propose SR3R, a feedforward 3DSR framework that directly predicts HR 3DGS from sparse LR views via a learned mapping network. The key idea behind SR3R is to reformulate 3DSR as a direct mapping from LR views to HR 3DGS representation, enabling the model to autonomously learn high-frequency geometric and texture details from large-scale, multi-scene data. This reformulation replaces the conventional 2DSR prior injection with data-driven 3DSR prior learning, marking a fundamental paradigm shift from per-scene HR 3DGS optimization to generalized HR 3DGS prediction (Fig. 1). Concretely, SR3R first employs any feed-forward 3DGS reconstruction model to estimate an LR 3DGS scaffold from sparse LR views, and then upscales it to HR 3DGS via the learned mapping network. The framework is fully plugand-play and compatible with existing feed-forward 3DGS pipelines. To further enhance reconstruction fidelity, we introduce Gaussian offset learning and feature refinement that sharpen high-frequency details and stabilize reconstruction. Extensive experiments demonstrate that SR3R outperforms state-of-the-art (SOTA) 3DSR methods and achieves strong zero-shot generalization, even surpassing per-scene optimization baselines on unseen scenes.

The main contributions are as follows.

â¢ A novel formulation of 3DSR. We reformulate 3DSR as a direct feed-forward mapping from LR views to HR 3DGS representations, eliminating the need for 2DSR pseudo-supervision and per-scene optimization. This shifts 3DSR from a 3DGS self-optimization paradigm to a generalized, feed-forward prediction.

â¢ A plug-and-play feed-forward framework for sparseview 3DSR. We propose SR3R, a feed-forward framework that directly reconstructs HR 3DGS from as few as two LR views through a learned mapping network. SR3R is plug-and-play with any feed-forward 3DGS reconstruction backbone and supports scalable cross-scene training.

â¢ Gaussian offset learning with feature refinement. We propose learning Gaussian offsets instead of directly regressing HR Gaussian parameters, which improves learning stability and reconstruction fidelity. In addition, we incorporate a feature refinement to further enhance highfrequency texture details.

â¢ SOTA performance and robust generalization. Extensive experiments on three 3D benchmarks demonstrate that SR3R surpasses SOTA 3DSR methods and exhibits strong zero-shot generalization, even outperforming perscene optimization baselines on unseen scenes.

## 2. Related Work

## 2.1. 3D Reconstruction

3DGS [11] has shown remarkable success in 3D scene reconstruction, offering real-time, high-fidelity rendering via Gaussian representations [5, 16, 37]. However, standard 3DGS reconstruction pipelines rely on dense multi-view inputs [22, 39] and per-scene optimization [2], severely limiting their scalability and applicability in real-time or openworld settings. To overcome these constraints, feed-forward 3DGS [2, 4, 26, 28] reconstruction models directly infer Gaussian parameters from input views using neural networks, enabling fast, end-to-end reconstruction. Recent extensions have even removed the need for known camera poses [32], further improving their practicality. This framework has been gradually applied in fields such as stylization [19, 24] and scene understanding [30]. Despite these advances, the current 3D reconstruction quality remains highly sensitive to input image resolution, resulting in significant loss of geometric and texture details under LR conditions. Our proposed SR3R addresses this challenge, enabling high-quality 3D reconstruction from as few as two LR views in a fully feed-forward manner.

<!-- image-->  
Figure 2. Overview of the SR3R framework. Given two LR input views, a feed-forward 3DGS backbone produces an LR 3DGS, which is then densified via Gaussian Shuffle Split to form a structural scaffold. The LR views are upsampled and processed by our mapping network: a ViT encoder with feature refinement integrates LR 3DGS-aware cues, and a ViT decoder performs cross-view fusion. The Gaussian offset learning module then predicts residual offsets to the dense scaffold, yielding the final HR 3DGS for high-fidelity rendering.

## 2.2. 2D Super-Resolution

2DSR aims to reconstruct HR images or video frames from their LR counterparts by learning an LR-to-HR image mapping. Over the past decade, the field has seen significant advances driven by model architectures, evolving from early convolutional networks [1, 6, 17, 41] to transformerbased architectures [14, 15] and, more recently, to generative approaches based on adversarial [13, 25, 31] and diffusion models [8, 20, 38, 44]. The availability of large-scale datasets has further fueled the success of 2DSR. However, 2DSR models face fundamental limitations when applied to 3D scene reconstruction. Since they operate solely in the image domain, they cannot enforce cross-view consistency [7], often leading to texture artifacts and geometric ambiguity when used to supervise 3D representations. Moreover, domain gaps between natural 2D images and multi-view 3D data further reduce the reliability of 2DSR priors. These limitations raise a central question: instead of relying on 2DSR, can we learn a direct mapping from LR views to HR 3D scene representations? This motivates us to propose SR3R, which directly addresses this problem.

## 2.3. 3D Super-Resolution

3DSR aims to reconstruct HR 3D scene representations from LR multi-view images [12, 35]. Recent 3DGS-based 3DSR methods [7, 12, 21, 27, 36] address this by injecting high-frequency information derived from pretrained 2DSR models. Typically, pseudo-HR images are generated from dense multi-view LR inputs to supervise the selfoptimization of HR 3DGS, while additional regularization, such as confidence-guided fusion [27] or radiance field correction [7], is applied to reduce view inconsistency caused by 2D pseudo-supervision. However, these pipelines suffer from critical limitations. Reconstruction fidelity is bounded by the quality of pseudo-HR labels, and per-scene optimization is computationally expensive and prevents cross-scene learning, limiting scalability. Inspired by recent advances in feed-forward 3DGS reconstruction, we propose SR3R, a feed-forward 3DSR framework that directly maps from LR views to HR 3DGS representations, enabling high-quality 3D reconstruction from as few as two LR input views while supporting efficient, cross-scene generalization.

## 3. Methodology

## 3.1. Problem Formulation

We reformulate 3DGS-based 3DSR as a feed-forward mapping problem from LR multi-view images to an HR 3DGS representation. Unlike prior methods that rely on dense inputs and per-scene optimization supervised by pseudo-HR 2D labels, our formulation enables direct HR 3DGS reconstruction from as few as two LR views, without any perscene optimization. This removes the reliance on 2DSR pseudo-supervision, allows learning from large-scale multiscene data, and enables cross-scene generalization, substantially improving scalability and efficiency.

Formally, given a set of V LR input views with camera intrinsics $\{ ( I _ { l r } ^ { v } , K ^ { v } ) \} _ { v = 1 } ^ { V }$ , our goal is to learn a feedforward mapping function fÎ¸ that predicts an HR 3DGS representation $\mathcal { G } ^ { \mathrm { H R } }$ . Each 3D Gaussian primitive is parameterized by its center $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , opacity $\alpha \in \mathbb { R }$ , quaternion rotation $\pmb { r } \in \mathbb { R } ^ { 4 }$ , scale $\boldsymbol { s } \in \mathbb { R } ^ { 3 }$ , and spherical harmonics (SH) appearance coefficients $\boldsymbol { c } \in \mathbb { R } ^ { k }$ , where k is the number of SH components. For simplicity, we omit the superscript for all Gaussian parameters. The mapping is defined as:

$$
f _ { \pmb \theta } : \{ ( I _ { l r } ^ { v } , K ^ { v } ) \} _ { v = 1 } ^ { V } \mapsto \mathcal G ^ { \mathrm { H R } }\tag{1}
$$

where $\mathcal { G } ^ { \mathrm { H R } } \ = \ \{ \cup ( \pmb { \mu } _ { i } ^ { v } , \pmb { \alpha } _ { i } ^ { v } , \pmb { r } _ { i } ^ { v } , \pmb { s } _ { i } ^ { v } , \pmb { c } _ { i } ^ { v } ) \} _ { i = 1 , \dots , N } ^ { v = 1 , \dots , V }$ , Î¸ denotes the learnable parameters of the neural network, and N is the number of Gaussian primitives in $\mathcal { G } ^ { \mathrm { H R } }$ . We omit the view index v hereafter for brevity.

## 3.2. Overall Framework

An overview of the proposed SR3R framework is illustrated in Figure 2. Given two LR input views, SR3R first reconstructs their LR 3DGSs $\mathcal { G } ^ { \mathrm { L R } }$ using any pretrained feedforward 3DGS reconstruction model, highlighting the plugand-play nature of our design. Each $\mathbf { \bar { \boldsymbol { g } } ^ { L R } }$ is then densified via a Gaussian Shuffle Split operation [23] to produce $\mathcal { G } ^ { \mathrm { D e n s e } }$ , which provides a structural scaffold for highfrequency geometry and texture recovery.

The LR input images are upsampled to the target resolution and processed by our mapping network, which consists of a ViT encoder, a feature refinement module, a ViT decoder, and a Gaussian offset learning module. The ViT encoder extracts mid-level feature tokens $t _ { \mathrm { e n } } ,$ which are refined through cross-attention with intermediate features from the feed-forward 3DGS backbone to produce corrected feature tokens $\pmb { t } _ { \mathrm { c a } }$ . The ViT decoder then performs cross-view fusion to generate ${ \pmb t } _ { \mathrm { d e } }$ , integrating complementary information from both views and mitigating misalignment or ghosting caused by pose inaccuracies or limited overlap. Finally, the Gaussian offset learning module predicts residual offsets from $\mathcal { G } ^ { \mathrm { D e n s e } }$ to the target HR 3DGS $\mathcal { G } ^ { \mathrm { H R } }$ . Learning offsets rather than directly regressing HR Gaussian parameters yields more stable training and significantly improves high-frequency texture fidelity, substantially enhancing overall reconstruction quality (Table 1).

## 3.3. LR 3DGS Reconstruction and Densification

LR 3DGS $\mathcal { G } ^ { \mathrm { L R } }$ for each input LR view can be obtained by any feed-forward 3DGS model. We then densify them via the Gaussian Shuffle Split operation [23] to produce $\mathcal { G } ^ { \mathrm { D e n s e } }$ , which serves as a finer structural scaffold for capturing high-frequency geometry and texture details and forms the basis for subsequent Gaussian offset learning.

Each Gaussian primitive $G _ { j } ^ { \mathrm { L R } } = ( \pmb { \mu _ { j } } , \pmb { \alpha _ { j } } , \pmb { r _ { j } } , \pmb { s _ { j } } , \pmb { c _ { j } } )$ in $\mathcal { G } ^ { \mathrm { L R } }$ is replaced by six smaller sub-Gaussians distributed along the positive and negative directions of its three principal axes. The sub-Gaussian centers are shifted from $\mu _ { j }$ by offsets proportional to the scale $\begin{array} { r } { \boldsymbol { s } _ { j } ~ = ~ [ s _ { j , 1 } , s _ { j , 2 } , s _ { j , 3 } ] , } \end{array}$ controlled by a factor Î² (set to 0.5 by default):

$$
\pmb { \mu } _ { j , k } = \pmb { \mu } _ { j } + \beta R _ { j } \pmb { e } _ { k } \odot \pmb { s } _ { j } , \quad k = 1 , \ldots , 6 ,\tag{2}
$$

where $R _ { j }$ is the rotation matrix derived from the quaternion $\mathbf { \nabla } _ { \mathbf { r } _ { j } , \mathbf { \eta } }$ , and $e _ { k }$ denotes the unit direction vectors along each positive and negative principal axis. Each sub-Gaussian inherits $r _ { j } , \alpha _ { j }$ , and $c _ { j }$ from the original, while its scale along the offset axis is reduced to $\textstyle { \frac { 1 } { 4 } }$ of its original to preserve spatial coverage. For stability, this operation is applied only to Gaussians with opacity above 0.5, focusing densification on structurally significant regions. The final densified 3DGS is obtained by aggregating all sub-Gaussians:

$$
\mathcal { G } ^ { \mathrm { { D e n s e } } } = \bigcup _ { j = 1 } ^ { M } \bigcup _ { k = 1 } ^ { 6 } G _ { j , k } ^ { \mathrm { { D e n s e } } } , \ G _ { j , k } ^ { \mathrm { { D e n s e } } } = ( \pmb { \mu } _ { j , k } , \alpha _ { j } , \pmb { r } _ { j } , s _ { j , k } , \pmb { c } _ { j } ) ,\tag{3}
$$

where M is the number of Gaussian primitives in $\mathcal { G } ^ { \mathrm { L R } }$ , and $\mathcal { G } ^ { \mathrm { D e n s e } }$ contains $N = 6 M$ primitives after densification.

## 3.4. LR Image to HR 3DGS Mapping

The mapping network is the core of SR3R, learning a viewconsistent transformation from LR input images to feature representations used for HR 3DGS reconstruction. It adopts a transformer-based architecture composed of a ViT encoder, a feature refinement module, a ViT decoder, and a Gaussian offset learning module. This design enables a view-aware mapping from the 2D LR image domain to the 3D Gaussian domain and leverages large-scale multi-scene training to achieve strong cross-scene generalization.

ViT Encoder. Each input LR image is first upsampled to the target resolution and, together with its camera intrinsics, is projected into a sequence of patch embeddings before being processed by the ViT encoder to produce midlevel feature tokens $t _ { \mathrm { e n } } .$ The encoder learns locally contextualized representations capturing essential texture and geometric cues. Trained across diverse scenes, these tokens remain reasonably aligned across views with minimal geometric priors, facilitating subsequent cross-view fusion.

Feature Refinement Module. Upsampled LR images often contain ambiguous or hallucinated high-frequency patterns due to interpolation, which may mislead the mapping network and introduce geometric or texture artifacts in 3D. To correct these unreliable 2D features, we introduce a feature refinement module that aligns the encoder tokens $\pmb { t } _ { \mathrm { e n } } \in \mathbb { R } ^ { N \times C }$ with geometry-aware tokens $t _ { \mathrm { p r e } } \in \mathbb { R } ^ { N \times C } \ { \mathrm { e x } }$ - tracted from the pretrained feed-forward 3DGS backbone used to obtain $\mathcal { G } ^ { \tt L \bar { R } }$ . Here, N denotes the number of tokens, and $C$ is the feature embedding dimension. Two crossattentions are computed in opposite directions:

$$
\mathbf { U } _ { o  p } = \mathrm { s o f t m a x } ( \frac { ( t _ { \mathrm { e n } } W _ { Q } ^ { o } ) ( t _ { \mathrm { p r e } } W _ { K } ^ { p } ) ^ { \top } } { \sqrt { d } } ) ( t _ { \mathrm { p r e } } W _ { V } ^ { p } ) ,
$$

$$
\mathbf { U } _ { p  o } = \mathrm { s o f t m a x } ( \frac { ( t _ { \mathrm { p r e } } W _ { Q } ^ { p } ) ( t _ { \mathrm { e n } } W _ { K } ^ { o } ) ^ { \top } } { \sqrt { d } } ) ( t _ { \mathrm { e n } } W _ { V } ^ { o } ) ,\tag{4}
$$

<!-- image-->  
Figure 3. Qualitative comparison with SOTA feed-forward 3DGS reconstruction methods on Re10k (top three) and ACID (bottom three) datasets. SR3R delivers significantly sharper details and more stable geometry than DepthSplat, NoPoSplat, and their upsampled variants, consistently improving reconstruction quality across different 3DGS backbones under sparse LR inputs.

where o and p denote our encoder and the pretrained encoder, respectively, $W _ { Q } ^ { ( \cdot ) } , W _ { K } ^ { ( \cdot ) }$ , and ${ \pmb W } _ { V } ^ { ( \cdot ) } \in \mathbb { R } ^ { C \times d }$ are learnable projection matrices, and d is the feature dimension per attention head. The two attention outputs $\mathbf { U } _ { o  p }$ and $\mathbf { U } _ { p  o }$ are then concatenated and fused through a fully connected layer to generate the refined feature token $\mathbf { \delta } _ { t _ { c a } } .$ This refinement process transfers reliable 3D geometric priors from the pretrained 3DGS encoder into our 2D feature space, suppressing upsampling-induced ambiguities and producing features that are better aligned with the underlying Gaussian structure and more consistent across views.

ViT Decoder. The refined features $\pmb { t } _ { \mathrm { c a } }$ from both views are fed into a ViT decoder, which performs intra-view selfattention to aggregate global contextual information and inter-view cross-attention to fuse cross-view features. This produces the decoded features $\pmb { t } _ { \mathrm { d e } } \in \mathbb { R } ^ { N \times C }$ , which integrate multi-view geometry and reduce inconsistencies caused by pose inaccuracy or limited view overlap. The decoded features are then provided to the Gaussian offset learning module (Section 3.5) to estimate residual corrections from the densified representation $\mathcal { G } ^ { \mathrm { D e n s e } }$ to the target HR 3DGS $\mathcal { G } ^ { \mathrm { H R } }$

## 3.5. Gaussian Offset Learning

Given the non-linear and scene-dependent relationship between 2D appearance and 3D geometry, directly regressing absolute Gaussian parameters from image features is often inefficient and unstable, as the resulting prediction space is large and multi-modal. In contrast, the densified representation $\mathcal { G } ^ { \mathrm { D e n s e } }$ already provides a reliable structural scaffold, meaning that the remaining discrepancy to HR is primarily local and high-frequency. Motivated by this, we proposed to learn a Gaussian offset field that predicts residual corrections to $\mathcal { G } ^ { \mathrm { D e n s e } }$ rather than regressing full HR parameters. This formulation constrains the learning target to local geometric and photometric offset, leading to more stable optimization and sharper reconstruction quality (Table 1).

Table 1. Quantitative comparison of 4Ã 3DSR on the large-scale RE10K and ACID datasets. SR3R consistently and substantially outperforms all baselines and their upscaled-input versions across PSNR, SSIM, and LPIPS, with only moderate Gaussian complexity and training memory. Bold indicates the best results and underline the second best.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">Metrics</td><td rowspan="2">Gaussian Param. â</td><td rowspan="2">Gaussian Num. â</td><td rowspan="2">Training Mem. â</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td rowspan="6">RE10K 64Ã64 â 256Ã256</td><td>NoPoSplat [33]</td><td>21.326</td><td>0.612</td><td>0.307</td><td>2.7M</td><td>8,192</td><td>4.82GB</td></tr><tr><td>Up-NoPoSplat</td><td>23.374</td><td>0.771</td><td>0.251</td><td>44.5M</td><td>131,072</td><td>21.36GB</td></tr><tr><td>Ours (NoPoSplat)</td><td>24.794</td><td>0.827</td><td>0.188</td><td>16.5M</td><td>49,152</td><td>12.92GB</td></tr><tr><td>DepthSplat [29]</td><td>23.147</td><td>0.699</td><td>0.281</td><td>2.3M</td><td>8,192</td><td>7.25GB</td></tr><tr><td>Up-DepthSplat</td><td>24.712</td><td>0.793</td><td>0.244</td><td>38.3M</td><td>131,072</td><td>26.17GB</td></tr><tr><td>Ours (DepthSplat)</td><td>26.250</td><td>0.856</td><td>0.165</td><td>14.2M</td><td>49,152</td><td>17.43GB</td></tr><tr><td rowspan="6">ACID 64Ã64 â 256Ã256</td><td>NoPoSplat [33]</td><td>21.451</td><td>0.606</td><td>0.531</td><td>2.7M</td><td>8,192</td><td>4.82GB</td></tr><tr><td>Up-NoPoSplat</td><td>23.911</td><td>0.692</td><td>0.384</td><td>44.5M</td><td>131,072</td><td>21.36GB</td></tr><tr><td>Ours (NoPoSplat)</td><td>25.541</td><td>0.746</td><td>0.283</td><td>16.5M</td><td>49,152</td><td>12.92GB</td></tr><tr><td>DepthSplat [29]</td><td>23.801</td><td>0.624</td><td>0.437</td><td>2.3M</td><td>8,192</td><td>7.25GB</td></tr><tr><td>Up-DepthSplat</td><td>25.315</td><td>0.721</td><td>0.322</td><td>38.3M</td><td>131,072</td><td>26.17GB</td></tr><tr><td>Ours (DepthSplat)</td><td>27.018</td><td>0.797</td><td>0.261</td><td>14.2M</td><td>49,152</td><td>17.43GB</td></tr></table>

Specifically, for each Gaussian primitive $\begin{array} { r l } { G _ { i } ^ { \mathrm { D e n s e } } } & { { } = } \end{array}$ $( \mu _ { i } , \alpha _ { i } , r _ { i } , s _ { i } , c _ { i } )$ in $\mathcal { G } ^ { \mathrm { D e n s e } }$ , we project its 3D center $\pmb { \mu } _ { i }$ onto the image plane to obtain the 2D coordinate $\mathbf { \nabla } _ { \pmb { p } _ { i } }$ . The corresponding local feature $\mathbf { \nabla } F _ { i }$ is then sampled from the reshaped decoded feature map $t _ { d e }$ at location $\mathbf { \Delta } _ { \pmb { p } _ { i } ^ { \prime } \mathbf { \Delta } ^ { s } }$ patch. These queried features are aggregated together with the Gaussian center and camera intrinsics K, and passed into a PointTransformerV3 network for spatial reasoning and multi-scale feature encoding:

$$
\begin{array} { r } { \pmb { F } = \Phi _ { \mathrm { P T v 3 } } \big ( \left[ \pmb { \mu } _ { i } ; \{ \pmb { F } _ { i } \} _ { i = 1 } ^ { N } ; \pmb { K } \right] \big ) , } \end{array}\tag{5}
$$

where $\Phi _ { \mathrm { P T v 3 } }$ denotes the PointTransformerV3 encoder that captures geometric relations and contextual dependencies among neighboring Gaussians. The encoded feature $\pmb { F }$ is then fed into a Gaussian Head $\Psi _ { \mathrm { G H } }$ , a lightweight MLP that predicts residual offsets for the Gaussian parameters:

$$
\begin{array} { r } { \Delta G = ( \Delta \mu , \Delta \alpha , \Delta r , \Delta s , \Delta c ) = \Psi _ { \mathrm { G H } } ( F ) . } \end{array}\tag{6}
$$

The final HR 3DGS is obtained via residual composition:

$$
\mathcal { G } ^ { \mathrm { H R } } = \mathcal { G } ^ { \mathrm { D e n s e } } + \Delta \mathcal { G } , \quad \Delta \mathcal { G } = \Delta G _ { i i = 1 } ^ { \ N }\tag{7}
$$

This residual formulation naturally focuses the network on high-frequency refinements while preserving the coarse structure encoded by $\mathcal { G } ^ { \mathrm { D e n s e } }$ . Compared with direct parameter regression, it improves convergence stability, reduces artifacts, and consistently yields sharper textures and more accurate geometry.

## 3.6. Training Objective

The predicted HR 3DGS $\mathcal { G } ^ { \mathrm { H R } }$ is rendered into novel-view images and supervised using the corresponding groundtruth RGB observations. The entire SR3R is trained end-toend through differentiable Gaussian rasterization. Following [33], we adopt a combination of pixel-wise reconstruction loss (MSE) and perceptual consistency loss (LPIPS) to jointly preserve geometric accuracy and visual fidelity.

## 4. Experimental Results

## 4.1. Experimental Setup

Datasets. We evaluate SR3R on three widely used 3D datasets: RealEstate10K (RE10K) [42], ACID [18], and DTU [10]. RE10K and ACID are two large-scale datasets, containing indoor real estate walkthrough videos and outdoor natural scenes captured by aerial drones, respectively. For fair comparison, we follow the official trainâtest splits used in prior works [28, 32]. To further assess generalization, we perform zero-shot 3DSR experiments on the DTU dataset, which features object-centric scenes with different camera motion and scene types from the RE10K.

Baselines and Metrics. We compare SR3R with two state-of-the-art feed-forward 3DGS reconstruction models, NoPoSplat [32] and DepthSplat [28], as well as the perscene optimization methods SRGS [7] and FSGS [43]. This setup allows us to evaluate large-scale 3DSR performance and demonstrate SR3Râs superior zero-shot capability without scene-specific optimization. Following prior work [7, 12], we assess novel-view synthesis quality using PSNR, SSIM, and LPIPS [40].

Implementation Details. We implement SR3R in Py-Torch and evaluate its plug-and-play compatibility with two 3DGS reconstruction backbones, NoPoSplat [32] and DepthSplat [28]. Input images are preprocessed by rescaling and center cropping, where the LR inputs are downsampled to 64 Ã 64 and the ground-truth (GT) targets to 256 Ã 256 using the LANCZO resampling filter. SwinIR [15] is used as the upsampling backbone, while simpler op-

<!-- image-->  
Noposplat (Base)

<!-- image-->  
+ Upsampling

<!-- image-->  
+ Cross Attention

<!-- image-->  
+ G. Offset w/o PTv3

<!-- image-->  
+ PTv3 (Ours)

Figure 4. Qualitative ablation results of SR3R components. Each component of SR3R progressively improves reconstruction quality, with upsampling reducing coarse blur, cross-attention improving feature alignment, Gaussian offset learning enhancing local geometry, and PTv3 yielding the sharpest and most consistent results.

erators such as Bicubic yield comparable results (Table 4). The ViT encoderâdecoder follows a vanilla configuration with a patch size of 16 and 8 attention heads. The MSE and LPIPS loss weights follow [32] and are set to 1 and 0.05. Both the backbone and our mapping network are trained for 75,000 iterations with a batch size of 8 and a learning rate of 2.5Ã10â5. All experiments are conducted on four NVIDIA RTX 5090 GPUs.

## 4.2. Comparison with State-of-the-Art

We evaluate SR3R through 4Ã 3DSR experiments on the large-scale RE10K and ACID datasets, and compare it against the SOTA feed-forward 3DGS reconstruction models NoPoSplat and DepthSplat. In addition to their standard version, we further evaluate their upsampled-input variants (Up-NoPoSplat and Up-DepthSplat), where LR inputs are first upsampled before direct HR Gaussian regression.

Table 1 shows that SR3R consistently outperforms both original and upsampled-input baselines across all metrics on both datasets. These results highlight the advantage of learning Gaussian offsets over direct parameter regression, enabling more accurate high-frequency recovery under sparse LR inputs. We also report complexity and training cost, showing that SR3R achieves these substantial gains with moderate computational overhead, demonstrating its practicality for scalable feed-forward 3DSR.

Figure 3 provides qualitative comparisons. Both baselines exhibit blurring, texture flattening, and geometric instability, while their upsampled variants remain unable to recover reliable high-frequency details and often introduce hallucinated edges or ghosting artifacts. In contrast, SR3R reconstructs sharper textures, cleaner boundaries, and more consistent geometry across views. These improvements hold for both 3DGS backbones, confirming that our offsetbased refinement and cross-view fusion effectively restore 3D-specific high-frequency structures that 2D upsampling and direct HR regression cannot recover.

## 4.3. Zero-Shot Generalization

We further evaluate the zero-shot generalization ability of SR3R on the DTU dataset, a challenging object-centric benchmark with unseen geometries and illumination conditions. All feed-forward models, including SR3R and baselines, are trained on RE10K and directly tested on DTU without any fine-tuning. We additionally include two SOTA per-scene optimization methods, SRGS [7] and FSGS [43], a sparse-view-specific model that we combine with SRGS (denoted as FSGS+SRGS) to provide a stronger baseline.

As shown in Table 2, SR3R achieves substantially higher accuracy than all feed-forward baselines in the zero-shot setting, demonstrating strong cross-scene generalization. Notably, SR3R also surpasses the per-scene optimization methods SRGS and FSGS+SRGS, despite requiring no scene-specific fitting at test time. This indicates that SR3R effectively preserves geometric and photometric fidelity even on completely unseen scenes. In terms of efficiency, SR3R is significantly faster than optimization-based methods, enabling practical real-time inference. Although its inference cost is slightly higher than that of other feedforward models, the clear performance gains make SR3R a compelling choice for scalable 3DSR.

Table 2. Zero-shot generalization results from RE10K to DTU. Feed-forward models are trained on RE10K and tested on DTU without fine-tuning. SRGS and FSGS+SRGS use per-scene optimization. SR3R delivers the best reconstruction quality while remaining significantly faster than optimization-based methods. Bold indicates the best results and underline the second best.
<table><tr><td rowspan="2">Method</td><td colspan="3">RE10K â DTU</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â Rec. Time â</td></tr><tr><td>SRGS [7]</td><td>12.420</td><td>0.327 0.598</td><td>300s</td></tr><tr><td>FSGS+SRGS [43]</td><td>13.720</td><td>0.444 0.481</td><td>420s</td></tr><tr><td>NopoSplat [33]</td><td>12.628</td><td>0.343 0.581</td><td>0.01s</td></tr><tr><td>Up-Noposplat</td><td>16.643</td><td>0.598 0.369</td><td>0.16s</td></tr><tr><td>Ours (NopoSplat)</td><td>17.241</td><td>0.607</td><td>1.69s</td></tr></table>

## 4.4. Ablation Study

Component Analysis. To assess the contribution of each component in SR3R, we perform a component-wise ablation using NoPoSplat as the baseline and evaluate 4Ã 3DSR performance on RE10K. As reported in Table 3, all proposed modules bring consistent and significant improvements. Adding the upsampling module provides a stronger initial estimate and yields clear improvements. Incorporating bidirectional cross-attention further enhances structural consistency by injecting geometric priors from the pretrained 3DGS encoder. Gaussian Offset Learning yields the largest performance gain. Even without PTv3 (G. Offset w/o PTv3), it significantly improves reconstruction quality while reducing the number of learnable Gaussian parameters, demonstrating its efficiency. Adding PointTransformerV3 further boosts accuracy through multi-scale spatial reasoning, producing the full SR3R model with the best performance. These results confirm that all components are necessary and complementary, collectively enabling SR3R to achieve high-fidelity HR 3D reconstruction.

Figure 4 presents the qualitative ablation results. The NoPoSplat baseline produces severe blurring and geometric degradation under sparse LR inputs. Applying 2D upsampling reduces excessive softness but still fails to recover reliable high-frequency structures, often introducing ambiguous or hallucinated textures. Adding cross-attention feature refinement improves feature alignment across views and suppresses texture drift. Gaussian Offset Learning further sharpens local geometry and appearance, yielding clearer object boundaries and more stable surface details. Integrating PTv3 completes the model and produces the sharpest textures, most accurate geometry, and highest overall fidelity. These results confirm that each SR3R component contributes progressively and that refinement, offset learning, and PTv3 together are essential for high-quality 3DSR.

Table 3. Component-wise ablation on RE10K (4Ã 3DSR). Modules are added cumulatively to the NoPoSplat baseline. Each component improves performance, and Gaussian Offset Learning yields the largest gain with fewer learnable Gaussians. The full SR3R achieves the best results.
<table><tr><td rowspan="2">Component</td><td colspan="4">RE10K (64 â 256)</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>Gauss. Param. â</td></tr><tr><td>Noposplat(Base)</td><td>21.326</td><td>0.612</td><td>0.307</td><td>2.7M</td></tr><tr><td>+ Upsampling</td><td>23.374</td><td>0.771</td><td>0.251</td><td>44.5M</td></tr><tr><td>+ Cross Attention</td><td>23.504</td><td>0.784</td><td>0.237</td><td>44.5M</td></tr><tr><td>+ G. Offset w/o PTv3</td><td>24.447</td><td>0.808</td><td>0.211</td><td>16.5M</td></tr><tr><td>+ PTv3 (Ours)</td><td>24.794</td><td>0.827</td><td>0.188</td><td>16.5M</td></tr></table>

Robustness to Upsampling Strategy. We evaluate the robustness of SR3R to different upsampling strategies used before the ViT encoder. Four commonly used methods are tested, including two interpolation-based approaches (Bilinear, Bicubic) and two learning-based SR models (SwinIR [15] and HAT [3]). As shown in Table 4, SR3R delivers consistently strong performance across all metrics, with only minor variation across different upsampling choices. Notably, even Bilinear interpolation already surpasses all feed-forward baselines (Table 1), indicating that SR3R does not depend on a particular upsampling design.

Table 4. Ablation on upsampling strategies on RE10K (4Ã 3DSR). SR3R maintains consistently strong performance across all interpolation and learning-based upsampling methods.
<table><tr><td rowspan="2">Upsampling</td><td colspan="4">RE10K (64 â 256)</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>Rec. Time â</td></tr><tr><td>Bilinear</td><td>24.586</td><td>0.795</td><td>0.204</td><td>1.59s</td></tr><tr><td>Bicubic</td><td>24.663</td><td>0.817</td><td>0.193</td><td>1.53s</td></tr><tr><td>SwinIR [15]</td><td>24.794</td><td>0.827</td><td>0.188</td><td>1.69s</td></tr><tr><td>HAT [3]</td><td>24.782</td><td>0.819</td><td>0.183</td><td>1.75s</td></tr></table>

## 5. Conclusion

We reformulate 3DSR as a feed-forward mapping from sparse LR views to HR 3DGS, enabling the learning of 3Dspecific high-frequency priors from large-scale multi-scene data. Based on this new paradigm, SR3R combines feature refinement and Gaussian offset learning to achieve highquality HR reconstruction with strong generalization. Experiments show that SR3R surpasses prior methods and provides an efficient, scalable solution for feed-forward 3DSR.

# SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting

Supplementary Material

## A. More Details for Gaussian Offset Learning

Figure S1 presents the detailed workflow of the proposed Gaussian Offset Learning, complementing the description in Section 3.5 of the main paper. Given the densified 3DGS template $\mathcal { G } ^ { \mathrm { D e n s e } } = \{ G _ { i } ^ { \mathrm { D e n s e } } \} _ { i = 1 } ^ { \hat { N } }$ and the decoded ViT feature tensor $\mathbf { t } _ { d e }$ , our Gaussian Offset Learning pipeline refines each Gaussian primitive through a sequence of geometryappearance fusion operations. For each Gaussian $G _ { i } ^ { \mathrm { { D e n s e } } } =$ $( \mu _ { i } , \alpha _ { i } , r _ { i } , s _ { i } , c _ { i } )$ , we first project its 3D center $\pmb { \mu } _ { i }$ onto the image plane. Let $\tilde { { \pmb \mu } } _ { i } = [ { \pmb \mu } _ { i } ^ { \top } , 1 ] ^ { \top } \in \mathbb { R } ^ { 4 }$ denote the homogeneous center, and let the camera extrinsic matrix be $\mathbf { P } \overset { ^ { \mathbf { \hat { \mathbf { \mu } } } } } { = } \left[ \mathbf { R } \mid \mathbf { t } \right] \in \mathbb { R } ^ { 3 \times 4 }$ with rotation R and translation t, and intrinsic matrix $\mathbf { K } \in \mathbb { R } ^ { 3 \times 3 }$ . The homogeneous image coordinate $\tilde { \pmb { p } } _ { i } \in \mathbb { R } ^ { 3 }$ is obtained by

$$
\tilde { p } _ { i } = \mathbf { K P } \tilde { \mu } _ { i } = \left[ \begin{array} { l } { \tilde { u } _ { i } } \\ { \tilde { v } _ { i } } \\ { \tilde { w } _ { i } } \end{array} \right] ,\tag{8}
$$

where $\tilde { u } _ { i } , \tilde { v } _ { i } ,$ and $\tilde { w } _ { i }$ denote the homogeneous pixel coordinates. The final 2D pixel position $\mathbf { \Delta } \mathbf { p } _ { i } = ( u _ { i } , v _ { i } ) ^ { \top }$ on the image plane is obtained by inhomogeneous normalization:

$$
u _ { i } = \frac { \tilde { u } _ { i } } { \tilde { w } _ { i } } , \qquad v _ { i } = \frac { \tilde { v } _ { i } } { \tilde { w } _ { i } } .\tag{9}
$$

These 3D centers are also fed into a position embedding network to generate the corresponding Gaussian position tokens, providing geometry-aware descriptors for each primitive. In parallel, the feature map $\mathbf { t } _ { d e } \in \mathbb { R } ^ { 4 \times 4 \times 7 6 8 } ~ \mathrm { i s }$ reshaped into a grid of local descriptors, from which we extract the feature $\mathbf { F } _ { i }$ corresponding to $\mathbf { \nabla } _ { \mathbf { p } _ { i } . }$ This queried feature serves as the queried token shown in the diagram. The Gaussian position token and queried image token are then fused and passed through a stack of M PointTransformerV3 (PTv3) blocks, which model geometric relations, neighborhood context, and long-range interactions among Gaussians. This produces an enhanced latent representation for each primitive. Finally, the encoded features are fed into a lightweight Gaussian Head, implemented as a small MLP, which predicts the residual parameter offsets $\Delta G _ { i } = ( \Delta \pmb { \mu _ { i } } , \Delta \pmb { \alpha _ { i } } , \Delta \pmb { r _ { i } } , \Delta \pmb { s _ { i } } , \Delta \pmb { c _ { i } } )$

## B. Additional Zero-Shot Visualizations on DTU

The main paper reports quantitative zero-shot results on the DTU dataset, demonstrating that SR3R achieves the highest accuracy among both feed-forward and per-scene optimization methods. To complement these quantitative findings, Figure S2 presents additional qualitative comparisons on DTU. As can be seen, both feed-forward and optimization-based baselines struggle under sparse LR inputs. SRGS and FSGS+SRGS exhibit strong geometric distortions and severe texture degradation, while NoPoSplat and its upsampled variant produce blurry or unstable highfrequency details. In contrast, SR3R reconstructs sharper textures, clearer boundaries, and substantially more stable geometry, consistent with the improvements observed on other datasets. These visualizations further validate SR3Râs strong cross-scene generalization and its ability to recover fine 3D structure on completely unseen scenes.

<!-- image-->  
Figure S1. Detailed Gaussian Offset Learning pipeline. Each Gaussian center is projected to the image plane to query local ViT features. The queried token is fused with a geometry-aware position embedding and processed by PTv3 blocks for spatial reasoning. A lightweight Gaussian Head predicts residual offsets to refine the initial 3DGS template.

## C. Additional Zero-shot Evaluation on Scan-Net++

To further validate the generalization ability of SR3R, we perform an additional zero-shot experiment on the Scan-Net++ dataset [34], which contains indoor scenes with different camera motion and scene types from the RE10K. The experimental setup follows the same protocol as in the main paper: all feed-forward models, including SR3R and the baselines, are trained on RE10K and directly tested on ScanNet++ without any fine-tuning. The per-scene optimization methods SRGS and FSGS+SRGS are evaluated using scene-specific optimization.

Table S1 shows that SR3R achieves the highest performance across all metrics, outperforming both feed-forward baselines and per-scene optimization methods. This experiment further demonstrates the strong cross-scene generalization of SR3R and its ability to recover high-frequency geometry and appearance on completely unseen datasets.

<!-- image-->  
Figure S2. Zero-shot qualitative comparison on the DTU dataset. Per-scene optimization and feed-forward baselines show blurring and geometric artifacts, while SR3R recovers significantly sharper textures and consistent geometry, highlighting its strong generalization to unseen scenes.

Table S1. Zero-shot generalization results from RE10K to Scanet++. Feed-forward models are trained on RE10K and tested on Scanet++ without fine-tuning. SRGS and FSGS+SRGS use per-scene optimization. SR3R delivers the best reconstruction quality while remaining significantly faster than optimizationbased methods. Bold indicates the best results.
<table><tr><td rowspan="2">Method</td><td colspan="3">RE10K â Scanet++</td></tr><tr><td>PSNR â</td><td>SSIM â LPIPS â</td><td>Rec. Time â</td></tr><tr><td>SRGS [7]</td><td>12.542</td><td>0.455</td><td>0.502 240s</td></tr><tr><td>FSGS+SRGS [43]</td><td>16.514</td><td>0.596 0.409</td><td>280s</td></tr><tr><td>NopoSplat [33]</td><td>18.284</td><td>0.578 0.421</td><td>0.01s</td></tr><tr><td>Up-Noposplat</td><td>20.870</td><td>0.696 0.303</td><td>0.16s</td></tr><tr><td>Ours (NopoSplat)</td><td>21.743</td><td>0.739 0.256</td><td>1.69s</td></tr></table>

Figure S3 presents the qualitative comparisons on Scan-Net++. As shown, the per-scene optimization methods SRGS and FSGS+SRGS exhibit strong geometric distortions and unstable shading artifacts under sparse LR inputs. Feed-forward baselines, including NoPoSplat and its upsampled variant, remain overly smooth and fail to recover high-frequency textures such as fine surface patterns or sharp edges. In contrast, SR3R reconstructs clearer textures, cleaner boundaries, and more stable geometry, closely matching the ground-truth appearance. These results further validate the strong cross-dataset generalization of SR3R.

## D. Additional Qualitative Comparisons

To complement the qualitative comparisons in Figure 3 of the main paper, we provide additional visual results in Figures S4 and S5. These examples follow the same evaluation protocol and compare SR3R with NoPoSplat, Depth-Splat, and their upsampled-input variants. Across a wide range of scenes, the same patterns observed in the main paper consistently hold: feed-forward baselines exhibit noticeable blurring, texture flattening, and geometric instability, while their upsampled variants still fail to recover reliable high-frequency structure. In contrast, our SR3R produces sharper textures, clearer boundaries, and more stable geometry across views. The improvements are consistent for both backbones, demonstrating that our offsetbased refinement and cross-view fusion robustly enhance 3D-specific high-frequency reconstruction under sparse LR inputs. These extended visualizations further substantiate the conclusions drawn in the main paper and highlight the reliability of SR3R across diverse scenes.

<!-- image-->  
Figure S3. Zero-shot qualitative comparison on the ScanNet++ dataset. Per-scene optimization and feed-forward baselines show blurring and geometric artifacts, while SR3R recovers significantly sharper textures and consistent geometry, highlighting its strong generalization to unseen scenes.

## References

[1] Kelvin CK Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Basicvsr: The search for essential components in video super-resolution and beyond. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4947â4956, 2021. 3

[2] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19457â19467, 2024. 3

[3] Xiangyu Chen, Xintao Wang, Wenlong Zhang, Xiangtao Kong, Yu Qiao, Jiantao Zhou, and Chao Dong. Hat: Hybrid attention transformer for image restoration. arXiv preprint arXiv:2309.05239, 2023. 8

[4] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei

Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, pages 370â386. Springer, 2024. 3

[5] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin, Yuexin Ma, Wenping Wang, and Xuejin Chen. Gaussianpro: 3d gaussian splatting with progressive propagation. In Fortyfirst International Conference on Machine Learning, 2024. 2

[6] Chao Dong, Chen Change Loy, and Xiaoou Tang. Accelerating the super-resolution convolutional neural network. In European Conference on Computer Vision (ECCV), pages 391â407, Cham, 2016. Springer International Publishing. 3

[7] Xiang Feng, Yongbo He, Yubo Wang, Yan Yang, Wen Li, Yifei Chen, et al. Srgs: Super-resolution 3d gaussian splatting. arXiv preprint arXiv:2404.10318, 2024. 2, 3, 6, 7, 8

[8] Sicheng Gao, Xuhui Liu, Bohan Zeng, Sheng Xu, Yanjing Li, Xiaoyan Luo, Jianzhuang Liu, Xiantong Zhen, and Baochang Zhang. Implicit diffusion models for continuous super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10021â10030, 2023. 3

[9] Yuqi Han, Tao Yu, Xiaohang Yu, Di Xu, Binge Zheng, Zonghong Dai, Changpeng Yang, Yuwang Wang, and Qionghai Dai. Super-nerf: View-consistent detail generation for nerf super-resolution. IEEE Transactions on Visualization and Computer Graphics, 2024. 2

[10] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik AanÃ¦s. Large scale multi-view stereopsis evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 406â413, 2014. 6

[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 2

[12] Hyun-kyu Ko, Dongheok Park, Youngin Park, Byeonghyeon Lee, Juhee Han, and Eunbyung Park. Sequence matters: Harnessing video models in 3d super-resolution. arXiv preprint arXiv:2412.11525, 2024. 2, 3, 6

[13] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Â´ Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photorealistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4681â4690, 2017. 3

[14] Wenbo Li, Xin Lu, Shengju Qian, Jiangbo Lu, Xiangyu Zhang, and Jiaya Jia. On efficient transformer and image pre-training for low-level vision. arXiv preprint arXiv:2112.10175, 2021. 3

[15] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In IEEE/CVF International Conference on Computer Vision Workshops, pages 1833â1844, 2021. 3, 6, 8

[16] Zhihao Liang, Qi Zhang, Wenbo Hu, Lei Zhu, Ying Feng, and Kui Jia. Analytic-splatting: Anti-aliased 3d gaussian splatting via analytic integration. In European Conference on Computer Vision (ECCV), pages 281â297. Springer, 2024. 2

[17] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 3

[18] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14458â14467, 2021. 6

[19] Hanzhou Liu, Jia Huang, Mi Lu, Srikanth Saripalli, and Peng Jiang. Stylos: Multi-view 3d stylization with single-forward gaussian splatting. arXiv preprint arXiv:2509.26455, 2025. 3

[20] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image superresolution via iterative refinement. arXiv:2104.07636, 2021. 3

[21] Yuan Shen, Duygu Ceylan, Paul Guerrero, Zexiang Xu, Niloy J. Mitra, Shenlong Wang, and Anna Frâuhstâuck. Supergaussian: Repurposing video models for 3d super resolution. In European Conference on Computer Vision (ECCV), 2024. 2, 3

[22] Changyue Shi, Chuxiao Yang, Xinyuan Hu, Yan Yang, Jiajun Ding, and Min Tan. Mmgs: Multi-model synergistic gaussian splatting for sparse view synthesis. Image and Vision Computing, page 105512, 2025. 3

[23] Yecong Wan, Mingwen Shao, Yuanshuo Cheng, and Wangmeng Zuo. S2gaussian: Sparse-view super-resolution 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 711â721, 2025. 2, 4

[24] Peng Wang, Xiang Liu, and Peidong Liu. Styl3r: Instant 3d stylized reconstruction for arbitrary scenes and styles. arXiv preprint arXiv:2505.21060, 2025. 3

[25] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy. Esrgan: Enhanced super-resolution generative adversarial networks. In European Conference on Computer Vision (ECCV), 2018. 3

[26] Yijia Weng, Zhicheng Wang, Songyou Peng, Saining Xie, Howard Zhou, and Leonidas J Guibas. Gaussianlens: Localized high-resolution reconstruction via on-demand gaussian densification. arXiv preprint arXiv:2509.25603, 2025. 3

[27] Shiyun Xie, Zhiru Wang, Yinghao Zhu, and Chengwei Pan. Supergs: Super-resolution 3d gaussian splatting via latent feature field and gradient-guided splitting. arXiv preprint arXiv:2410.02571, 2024. 3

[28] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16453â16463, 2025. 3, 6

[29] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16453â16463, 2025. 6

[30] Qi Xu, Dongxu Wei, Lingzhe Zhao, Wenpu Li, Zhangchi Huang, Shunping Ji, and Peidong Liu. Siu3r: Simultaneous scene understanding and 3d reconstruction beyond feature alignment. arXiv preprint arXiv:2507.02705, 2025. 3

[31] Yiran Xu, Taesung Park, Richard Zhang, Yang Zhou, Eli Shechtman, Feng Liu, Jia-Bin Huang, and Difan Liu. Videogigagan: Towards detail-rich video super-resolution. 2024. 3

[32] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, et al. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207, 2024. 3, 6, 7

[33] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, et al. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. In The Thirteenth International Conference on Learning Representations, 2025. 6, 8, 2

[34] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias NieÃner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12â22, 2023. 1

[35] Youngho Yoon and Kuk-Jin Yoon. Cross-guided optimization of radiance fields with multi-view image superresolution for high-resolution novel view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12428â12438, 2023. 3

[36] Xiqian Yu, Hanxin Zhu, Tianyu He, and Zhibo Chen. Gaussiansr: 3d gaussian super-resolution with 2d diffusion priors. arXiv preprint arXiv:2406.10111, 2024. 2, 3

[37] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19447â 19456, 2024. 2

[38] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image superresolution by residual shifting. Advances in Neural Information Processing Systems, 36, 2024. 3

[39] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In European Conference on Computer Vision (ECCV), pages 335â352. Springer, 2024. 3

[40] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 586â595, 2018. 6

[41] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. Image super-resolution using very deep residual channel attention networks. In European Conference on Computer Vision (ECCV), 2018. 3

[42] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM Transactions on Graphics (TOG), 37(4):1â12, 2018. 6

[43] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian

splatting. In European Conference on Computer Vision, pages 145â163, 2024. 6, 7, 8, 2

[44] Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, and Tianfan Xue. Flashvsr: Towards real-time diffusion-based streaming video super-resolution, 2025. 3

<!-- image-->  
GT  
DepthSplat Up-DepthSplat Ours (DepthSplat)  
Figure S4. Qualitative comparison with SOTA feed-forward 3DGS reconstruction methods on the ACID dataset. SR3R delivers significantly sharper details and more stable geometry than DepthSplat, NoPoSplat, and their upsampled variants, consistently improving reconstruction quality across different 3DGS backbones under sparse LR inputs.

<!-- image-->  
DepthSplat Up-DepthSplat Ours (DepthSplat)  
NoPoSplat Up-NoPoSplat Ours (NoPoSplat)

Figure S5. Qualitative comparison with SOTA feed-forward 3DGS reconstruction methods on the RE10k dataset. SR3R delivers significantly sharper details and more stable geometry than DepthSplat, NoPoSplat, and their upsampled variants, consistently improving reconstruction quality across different 3DGS backbones under sparse LR inputs.