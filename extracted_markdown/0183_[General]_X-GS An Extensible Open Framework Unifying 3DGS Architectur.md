# X-GS: An Extensible Open Framework for Perceiving and Thinking via 3D Gaussian Splatting

Yueen Ma13, Zenglin Xu23, Irwin King1

The Chinese University of Hong Kong1, Fudan University2, Shanghai Academy of AI for Science3 {yema21, king}@cse.cuhk.edu.hk, zenglinxu@fudan.edu.cn

## Abstract

3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains such as pose-free 3DGS, online SLAM, and semantic enrichment. In this paper, we introduce X-GS, an extensible open framework consisting of two major components: the X-GS-Perceiver, which unifies a broad range of 3DGS techniques to enable real-time online SLAM and distill semantic features; and the X-GS-Thinker, which interfaces with downstream multimodal models. In our implementation of the Perceiver, we integrate various 3DGS methods through three novel mechanisms: an online Vector Quantization (VQ) module, a GPUaccelerated grid-sampling scheme, and a highly parallelized pipeline design. The Thinker accommodates vision-language models and utilizes the resulting 3D semantic Gaussians, enabling downstream applications such as object detection, caption generation, and potentially embodied tasks. Experimental results on realworld datasets demonstrate the efficiency and newly unlocked multimodal capabilities of the X-GS framework.

## 1 Introduction

3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) has achieved groundbreaking success in real-time novel view synthesis, catalyzing rapid advancements across several distinct research directions. To eliminate the reliance on computationally expensive offline Structure-from-Motion (SfM) pipelines like COLMAP, recent methods (Fu et al., 2024) have successfully reconstructed scenes directly from unposed image sequences by relaxing camera pose requirements. Furthermore, the advent of

<!-- image-->  
Figure 1: X-GS is an extensible open framework that unifies previously isolated domains in 3D-GS. X-GS-Perceiver achieves real-time 3DGS-based online SLAM with semantics using an online VQ module, GPU-accelerated grid-sampling, and highly parallelized scheduling. X-GS-Thinker bridges the resulting 3D semantic Gaussians with diverse downstream multimodal models and tasks.

3DGS-based SLAM systems (Matsuki et al., 2024) has improved upon traditional point-cloud-based methods, allowing the mapped scene to retain significantly richer visual information. Extending beyond pure appearance, Semantic-GS (Li et al., 2025c) endows the 3D Gaussians with explicit semantic feature channels alongside their standard parameters to enable complex scene understanding. Finally, bridging the gap between spatial representations and multimodal reasoning, the latest works have developed Vision-Language Models (VLMs) (Thai et al., 2025) that natively utilize 3D Gaussians as direct inputs rather than relying solely on 2D images.

Despite this rapid progress, these advancements typically operate in isolation. To combine their respective strengths, we propose the first work to unify these disparate 3DGS techniques from a comprehensive framework perspective: the X-GS framework, as illustrated in Figure 1. Standing for an eXtensible open framework for 3DGS, the âXâ serves as a placeholder prefix denoting distinct 3DGS paradigms, such as RGBD-3D-GS, posefree-GS, online-GS, SLAM-GS, semantic-GS, and VLM-for-GS. This design ensures that future developments in these different directions can be assimilated into X-GS for continued improvement, directly showcasing its extensibility. Built upon this unified foundation, X-GS achieves real-time 3DGS-based online SLAM with semantic distillation while simultaneously bridging the gap to downstream VLMs.

Although X-GS is built upon existing 3DGS algorithms, integrating them for real-time performance is a non-trivial process. To achieve this, we introduce X-GS-Perceiver, our dedicated semantic mapping module, which is driven by three key efficiency techniques: an online vector quantization (VQ) module, a grid-sampling scheme, and a meticulously designed pipeline that fully leverages parallelism. The VQ module drastically reduces the dimensionality of the semantic channels stored in each 3D Gaussian. While previous works have utilized VQ modules in 3DGS, they were limited to offline processing. We overcome this by introducing a VQ module with Exponential Moving Average (EMA) updates to support continuous online learning. Additionally, because 3D Gaussians usually project to areas rather than single pixels on the rendered image, we introduce a gridsampled semantic supervision scheme. This is paired with a custom GPU kernel that executes only the minimal calculations required for the subsampled pixels. This technique maintains semantic map quality while achieving significant speedup and memory load reduction. At a systemic level, X-GS-Perceiver schedules these individual components using highly parallelized strategies, such as early VQ codebook updates and grid-sampled target prefetching.

Because X-GS can accommodate various vision foundation models, it readily supports multiple downstream multimodal tasks through the X-GS-Thinker component. By combining SAM (Kirillov et al., 2023) and CLIP (Radford et al., 2021), X-GS-Perceiver receives object-level semantics. The resulting 3D Gaussians thus encode these semantics, enabling X-GS-Thinker, which in this case is based on the CLIP text encoder, to perform textprompted 3D object detection in the mapped scene. If we integrate the vision tower of a VLM (Liu et al., 2023), the 3DGS semantics can also be fed into the modelâs backbone, acting as the X-GS-Thinker, to generate scene captions. Potentially, by connecting these semantic features to a vision-language-action (VLA) model (Brohan et al., 2023), its action generation module can serve as the X-GS-Thinker to tackle embodied tasks. This suggests the additional extensibility of the X-GS framework to downstream multimodal models.

In summary, the main contributions are:

â¢ We propose X-GS, an extensible open framework that unifies previously isolated domainsâsuch as 3DGS-based online SLAM, pose-free 3DGS, semantic 3DGS, and 3DGSbased VLMsâinto a cohesive system.

â¢ We introduce X-GS-Perceiver, equipped with three core optimization techniques: an online VQ module with EMA updates, a GPUaccelerated grid-sampling scheme, and a highly parallelized scheduling design, collectively ensuring real-time performance.

â¢ We demonstrate its additional extensibility by introducing the X-GS-Thinker component, which bridges the resulting semantic 3DGS with downstream multimodal models to enable tasks such as 3D object detection, scene captioning, and embodied tasks.

## 2 Related Work

## 2.1 3DGS-based Online SLAM

Standard 3D Gaussian Splatting (3DGS) algorithms (Kerbl et al., 2023) rely on offline multiview image sequences and COLMAP initialization. While subsequent methods like CF-3DGS (Fu et al., 2024) eliminate the need for pre-computed poses, they still require offline processing. Recently, approaches such as MonoGS (Matsuki et al., 2024), GS-SLAM (Yan et al., 2024), and Gaussian-SLAM (Yugay et al., 2023) have achieved online, real-time SLAM using 3DGS. However, these systems focus exclusively on 3D reconstruction and camera tracking, ignoring high-level semantic scene understanding. To enrich 3DGS representations, Feature 3DGS (Zhou et al., 2024) and LangSplat (Qin et al., 2024; Li et al., 2025c) distill dense semantic features from vision foundation models (e.g., SAM or CLIP) directly into the 3D Gaussian field, enabling language-driven queries and zero-shot 3D object detection. Nevertheless, these semantic methods rely heavily on precise, precomputed camera poses and are strictly designed for offline mapping, inherently limiting their applicability in dynamic or autonomous environments.

Table 1: Comparison of X-GS with representative 3DGS methods. Gen-VLM: integration with generative VLMs.
<table><tr><td>Method</td><td>RGB-only</td><td>RGB-D</td><td></td><td>Pose-free Online-SLAM Real-time</td><td></td><td>Semantics</td><td> Gen-VLM</td></tr><tr><td>Offline 3DGS &amp; Semantic Fields</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3DGS (Kerbl et al., 2023)</td><td>â</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>CF-3DGS (Fu et al., 2024)</td><td>â</td><td></td><td></td><td>X</td><td></td><td></td><td></td></tr><tr><td>Feature 3DGS (Zhou et al., 2024)</td><td>â</td><td>Ã Ã</td><td>&gt; Ã</td><td>X</td><td>Ã Ã</td><td></td><td></td></tr><tr><td>LangSplat (Qin et al., 2024)</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>&gt;&gt;</td><td>ÃÃÃ</td></tr><tr><td>LangSplatV2 (Li et al., 2025c)</td><td></td><td>X</td><td>X</td><td>X</td><td>â</td><td>â</td><td>X</td></tr><tr><td>Online 3DGS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MonoGS (Matsuki et al., 2024) (SLAM)</td><td>â</td><td></td><td></td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>LEGO-SLAM (Lee et al., 2025)</td><td>X</td><td></td><td>â</td><td>â</td><td>â</td><td>â</td><td></td></tr><tr><td>OpenMonoGS-SLAM (Yoo et al., 2025)</td><td>â</td><td>&gt;Ãx</td><td>â</td><td>â</td><td></td><td>â</td><td>Ã Ãx</td></tr><tr><td>EA3D (Zhou et al., 2025) (SLAM)</td><td>â</td><td></td><td></td><td>â</td><td></td><td>â</td><td></td></tr><tr><td>EmbodiedSplat (Lee et al., 2026)</td><td>â</td><td>â</td><td>X</td><td>â</td><td></td><td>â</td><td>X</td></tr><tr><td>VLMs for 3DGS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UniGS (Li et al., 2025b)</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td></td><td></td></tr><tr><td>SplatTalk (Thai et al., 2025)</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td></td><td></td></tr><tr><td>X-GS (Ours)</td><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td><td></td><td></td></tr></table>

## 2.2 Semantic 3DGS-based SLAM

Existing semantic 3DGS SLAM systems, such as LEGO-SLAM (Lee et al., 2025) and OpenMonoGS-SLAM (Yoo et al., 2025), attempt online semantic mapping but suffer from rigid architectural bottlenecks. LEGO-SLAM strictly requires RGB-D input, whereas OpenMonoGS-SLAM operates solely on monocular videoâlacking the flexibility to utilize depth maps even when available. Furthermore, OpenMonoGS-SLAM fails to achieve real-time performance. X-GS natively overcomes all of these limitations.

## 2.3 VLMs for 3DGS

Recent literature (Li et al., 2024) demonstrates the immense value of explicitly integrating 3D representations with Vision-Language Models (VLMs) for downstream reasoning. For instance, UniGS (Li et al., 2025b) aligns optimized 3D Gaussians with textual spaces for multimodal contrastive learning, while SplatTalk (Thai et al., 2025) and Chat-Splat (Chen et al., 2024) build VLMs that take 3DGS as visual input. However, current VLMs for 3DGS are restricted entirely to static, offline scenes. X-GS unifies these disjointed paradigms into a single online framework.

Table 1 provides a comparison of key capabilities among representative 3DGS methods. To the best of our knowledge, X-GS is the first framework to fulfill all the listed requirements online.

## 3 Method

We propose X-GS, an extensible open framework designed to unify previously isolated 3DGS advancements. As illustrated in Figure 2, the framework is divided into two primary components. First, the perception component, X-GS-Perceiver, ingests unposed monocular or RGB-D video streams to continuously co-optimize the 3D Gaussians and camera poses, while simultaneously distilling highdimensional foundation model embeddings into the Gaussians as semantic features. To ensure real-time performance, it utilizes online vector quantization, grid sampling, and parallel scheduling. Second, the X-GS-Thinker component accommodates various multimodal models. By leveraging the 3D semantic Gaussians produced by the Perceiver, the Thinker unlocks advanced capabilities such as textprompted 3D object detection, zero-shot scene captioning, and potentially embodied tasks.

## 3.1 Preliminaries

3D Gaussian Splatting and Feature Rendering. 3DGS represents scene radiance (geometry and appearance) using a collection of 3D Gaussians. Each Gaussian is parameterized by $\Theta = \{ \mu , \Sigma , \alpha , c \}$ , representing its 3D center Âµ, covariance Î£, base opacity Î±, and color c. To compute the expected color $C ( x )$ for a pixel x from a specific camera pose $\mathbf { T } \in \mathrm { S E } ( 3 )$ , a set of N overlapping 3D Gaussians are transformed by T into camera space, projected into 2D screen coordinates, sorted by depth, and composited via front-to-back Î±-blending:

<!-- image-->  
Figure 2: Overview of the X-GS framework. X-GS-Perceiver synergizes a memory-efficient Vector Quantization (VQ) module, grid-based semantic supervision, and an asynchronous parallelized pipeline to perform SLAM and distill semantics simultaneously in an online fashion, executing in real time at â¼15 FPS. As an open framework, it accommodates both RGB-only and RGB-D inputs, and can flexibly integrate various Vision Foundation Models (VFMs). Furthermore, the X-GS-Thinker component is extensible to different multimodal models, enabling a wide range of downstream tasks.

$$
C ( x \mid \mathbf { T } , \Theta ) = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } ) ,\tag{1}
$$

where $c _ { i }$ is the color and $\alpha _ { i } ^ { \prime }$ is the evaluated 2D footprint opacity of the i-th Gaussian. Crucially, $\alpha _ { i } ^ { \prime }$ and the depth-sorting order are explicit functions of the camera pose T via the view transformation and perspective projection, rendering the entire rasterization process differentiable with respect to both T and Î.

Beyond RGB appearance, this rasterization process natively extends to high-dimensional semantic features. By substituting the 3-dimensional color $c _ { i }$ with a D-dimensional feature vector, the exact same Î±-blending formulation in (1) renders a dense 2D semantic feature map $F ( x \mid \mathbf { T } , \Theta )$ . This computationally pairs the discrete 3D semantic states of the Gaussians with their corresponding 2D spatial observations.

3DGS-based SLAM. The online SLAM pipeline (Matsuki et al., 2024) operates via two continuous, concurrent threads: tracking and mapping. During tracking, camera poses are estimated through direct optimization against the explicit 3D Gaussian representation. The pose $\mathbf { T } _ { t }$ for an incoming frame t is initialized using a constant velocity kinematic model. Leveraging the differentiability of (1), $\mathbf { T } _ { t }$ is then iteratively refined via gradient descent by holding the map parameters Î fixed and minimizing the tracking loss ${ \mathcal { L } } _ { \mathrm { t r a c k } }$ against the captured groundtruth color image $I _ { t }$ and optional depth map $D _ { t }$

$$
\mathbf { T } _ { t } ^ { * } = \underset { \mathbf { T } _ { t } } { \arg \operatorname* { m i n } } \mathcal { L } _ { \mathrm { t r a c k } } ( I _ { t } , D _ { t } \mid \mathbf { T } _ { t } , \Theta ) .\tag{2}
$$

Concurrently, the mapping thread refines the global scene representation by holding the camera poses fixed and optimizing the Gaussian parameters Î. This optimization is performed over a sliding window of keyframes K, which is dynamically maintained by inserting new frames that provide sufficient novel viewpoints, while evicting redundant older views to preserve a maximum capacity of $N _ { w i n }$ . The mapping thread refines Î by minimizing the mapping loss ${ \mathcal { L } } _ { \mathrm { m a p } }$ across this active window of ground-truth images $I _ { k }$ and depth maps $D _ { k }$

$$
\Theta ^ { * } = \underset { \Theta } { \operatorname { a r g m i n } } \sum _ { k \in \mathcal { K } } \mathcal { L } _ { \operatorname* { m a p } } ( I _ { k } , D _ { k } \mid \Theta , \mathbf { T } _ { k } ) .\tag{3}
$$

Dynamic mapping heuristics, such as geometryguided Gaussian insertion and opacity-based pruning, are utilized alongside these objectives to maintain a structurally clean topological map.

## 3.2 X-GS-Perceiver

We develop a set of modules that add semantic enrichment capabilities to the online geometric and photometric optimization pipeline. Naively attaching semantic features to 3D Gaussians has proven to be heavily memory- and compute-intensive. To ensure our pipeline still fulfills the real-time requirement, we devise three key speedup techniques: an online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a meticulously designed pipeline that fully leverages highly parallelized scheduling. Specifically, the VQ module reduces the feature dimensionality of each pixel within the rasterized semantic map, while the grid-sampling scheme reduces its resolution (height and width). Finally, parallelized scheduling optimizes the overall execution at the module level.

Online Vector Quantization Module. We represent the semantic state of each Gaussian by logits over a shared codebook, inspired by (Li et al., 2025c). Let $N _ { g }$ denote the number of Gaussians, K the number of codewords, and D the feature dimension. The shared codebook parameters are

$$
\mathbf { E } \in \mathbb { R } ^ { K \times D } .\tag{4}
$$

Each Gaussian i stores a learnable logit vector

$$
\mathbf { z } _ { i } \in \mathbb { R } ^ { K } .\tag{5}
$$

We convert these logits into mixture weights using a softmax over all codebook entries:

$$
\begin{array} { r l } & { { \bf w } _ { i } = \mathrm { s o f t m a x } ( { \bf z } _ { i } ) , } \\ & { w _ { i k } = \frac { \exp ( z _ { i k } ) } { \sum _ { k ^ { \prime } = 1 } ^ { K } \exp ( z _ { i k ^ { \prime } } ) } . } \end{array}\tag{6}
$$

The decoded semantic feature of Gaussian i is then obtained by a weighted sum over the shared codebook:

$$
\begin{array} { l } { { \displaystyle { \hat { \mathbf { f } } } _ { i } = { \mathbf { E } } ^ { \top } { \mathbf { w } } _ { i } } } \\ { { \displaystyle \quad = \sum _ { k = 1 } ^ { K } { w _ { i k } \mathbf { e } _ { k } \in \mathbb { R } ^ { D } } . } } \end{array}\tag{7}
$$

This parameterization separates Gaussian-specific semantic coefficients from the shared semantic atoms in the codebook, yielding a compact representation that can be optimized jointly with the Gaussian scene.

We update the shared codebook online using vector quantization with Exponential Moving Averages (EMA). Given a set of observed semantic features $\{ { \bf x } _ { n } \} _ { n = 1 } ^ { N }$ , each feature is assigned to its nearest codeword:

$$
a _ { n } = \arg \operatorname* { m i n } _ { k \in \{ 1 , . . . , K \} } \left\| \mathbf { x } _ { n } - \mathbf { e } _ { k } \right\| _ { 2 } .\tag{8}
$$

We then accumulate the assignment counts and feature sums,

$$
\begin{array} { l } { \displaystyle c _ { k } = \sum _ { n = 1 } ^ { N } \mathbb { I } [ a _ { n } = k ] , } \\ { \displaystyle \mathbf { s } _ { k } = \sum _ { n = 1 } ^ { N } \mathbb { I } [ a _ { n } = k ] \mathbf { x } _ { n } , } \end{array}\tag{9}
$$

and update the EMA buffers

$$
\begin{array} { c } { { N _ { k }  \lambda N _ { k } + ( 1 - \lambda ) c _ { k } , } } \\ { { \mathbf { M } _ { k }  \lambda \mathbf { M } _ { k } + ( 1 - \lambda ) \mathbf { s } _ { k } , } } \end{array}\tag{10}
$$

where $\lambda \in [ 0 , 1 )$ is the EMA decay. The updated codeword is

$$
\mathbf { e } _ { k }  { \frac { \mathbf { M } _ { k } } { N _ { k } + \varepsilon } } ,\tag{11}
$$

with a small constant Îµ for numerical stability. In this way, the per-Gaussian logits are learned by gradient descent, while the shared codebook tracks the evolving feature distribution through EMA continuous online learning.

Grid-Sampled Semantic Supervision. Applying dense semantic supervision at every image pixel is unnecessarily expensive during mapping, because 3D Gaussians project to areas rather than single pixels on the rendered image. We therefore supervise the rendered semantic features on a regular stride-offset grid in the image plane. Let the full image domain be

$$
\Omega = \{ 0 , \ldots , H - 1 \} \times \{ 0 , \ldots , W - 1 \} ,\tag{12}
$$

and let the grid stride be $s \geq 1$ with offset ${ \textbf { o } } =$ $\left( o _ { h } , o _ { w } \right)$ , where $0 \leq o _ { h } , o _ { w } < s$ . The sampled grid is

$$
\begin{array} { r } { \Omega _ { s , \mathbf { o } } = \left. ( u , v ) \in \Omega \mid u \equiv o _ { h } ( \mathrm { m o d } s ) , \right. \phantom { x x x x x x x x x x x x x x x x x } } \\ { v \equiv o _ { w } ( \mathrm { m o d } s ) \left. . \right. } \end{array}\tag{13}
$$

Equivalently, each sampled location can be written as

$$
( u , v ) = ( o _ { h } + m s , ~ o _ { w } + n s ) ,\tag{14}
$$

for integer indices $m , n .$ . The resulting compact feature-map resolution is

$$
\begin{array} { l } { { \displaystyle H _ { s } = \operatorname* { m a x } \biggl ( 0 , \ : 1 + \left\lfloor \frac { H - 1 - o _ { h } } { s } \right\rfloor \biggr ) \ : , } } \\ { { \displaystyle W _ { s } = \operatorname* { m a x } \biggl ( 0 , \ : 1 + \left\lfloor \frac { W - 1 - o _ { w } } { s } \right\rfloor \biggr ) \ : . } } \end{array}\tag{15}
$$

Let $\hat { \mathbf { F } } \in \mathbb { R } ^ { D \times H \times W }$ denote the dense rendered semantic feature field. Conceptually, grid sampling restricts this field to the grid $\Omega _ { s , \mathbf { o } }$ , yielding the compact prediction

$$
\begin{array} { r } { \hat { \mathbf { G } } _ { : , m , n } = \hat { \mathbf { F } } _ { : , o _ { h } + m s , o _ { w } + n s } \in \mathbb { R } ^ { D } . } \end{array}\tag{16}
$$

To maximize efficiency, we pair this scheme with a custom GPU kernel optimized to execute only the minimal pixel-level calculations required for the subsampled grid locations. The rasterizer directly outputs this compact prediction, entirely avoiding the computational and memory burdens of forming a dense semantic feature map before subsampling.

We construct the target feature map ${ \bf G } ^ { \star } \in { \bf \Xi }$ $\mathbb { R } ^ { D \times H _ { s } \times W _ { s } }$ on the same grid, together with a binary validity mask

$$
\mathbf { V } \in \{ 0 , 1 \} ^ { 1 \times H _ { s } \times W _ { s } } .\tag{17}
$$

The semantic supervision is then applied only on valid sampled locations:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g r i d } } = \lambda _ { \mathrm { s e m } } \left[ \mathcal { L } _ { \mathrm { c o s } } \Big ( \mathbf { V } \odot \hat { \mathbf { G } } , \mathbf { V } \odot \mathbf { G } ^ { \star } \Big ) \right. } \\ { \left. + \mathcal { L } _ { 1 } \Big ( \mathbf { V } \odot \hat { \mathbf { G } } , \mathbf { V } \odot \mathbf { G } ^ { \star } \Big ) \right] , } \end{array}\tag{18}
$$

where â denotes element-wise masking. The validity mask V serves two vital purposes. Primarily, it acts as a filter for unannotated or background regions that yield meaningless zero-vectors in the target grid $\mathbf { G } ^ { \star }$ , preventing the 3D Gaussians from incorrectly learning empty features. Secondarily, it masks out padded, out-of-bounds coordinates that arise from maintaining rigid batched tensor shapes at large grid offsets, ensuring the loss is strictly evaluated on robust semantic data.

When $s = 1$ , the formulation reduces to standard dense semantic supervision. However, for a generic stride $s > 1$ , the custom kernel correctly reduces the number of supervised pixels by a factor of $s ^ { 2 }$ . This explicit reduction yields a proportional $s ^ { 2 } \times$ savings in both the memory bandwidth required for rasterizing the deep semantic features and the computational overhead of the loss evaluation, keeping the entire optimization comfortably within real-time constraints.

Parallel Pipeline Architecture. To integrate these individual components into a highly efficient workflow, we employ a heavily parallelized system architecture, as illustrated in Figure 2. The VQ codebook update is executed as soon as the Vision Foundation Model (VFM) completes encoding an incoming keyframe. Simultaneously, we initiate a âGrid-Sampled Target Prefetchingâ operation on the full-resolution semantic map generated by the VFM. Because operations across different spatial grid offsets are mutually independent, we allocate multiple background workers to extract and buffer the grid-sampled targets well in advance of the actual optimization loop.

Crucially, to maintain system stability and computational efficiency, we strictly decouple the geometry and appearance updates from the semantic updates. During the semantic optimization phase, all foundational parameters of the 3D Gaussians (i.e., position, scale, rotation, opacity, and color) remain frozen. Conversely, the learned semantic logits and VQ codebooks are strictly excluded from the base tracking and mapping steps. By synergizing the memory-efficient VQ module, the reduced computational footprint of our grid-based semantic supervision, and this multi-threaded alternating optimization schedule, our system achieves comprehensive semantic enrichment while consistently maintaining real-time performance at â¼15 FPS.

## 3.3 X-GS-Thinker

While the X-GS-Perceiver handles the efficient construction of 3D semantic Gaussians, the X-GS-Thinker operates atop this framework to interpret and leverage these learned representations for downstream multimodal tasks. Because the semantic fields mapped onto the 3D Gaussians are distilled directly from prominent vision foundation models (e.g., CLIP, SAM, or SigLIP), the X-GS-Thinker remains highly extensible, adapting seamlessly to various multimodal network architectures.

Contrastive VLM. For text-prompted 3D object detection, the X-GS-Thinker relies on a contrastive vision-language model, such as OpenCLIP. Crucially, rather than rendering 2D semantic feature maps, we localize user-specified concepts by querying the 3D scene representation directly. First, the text query and a standard set of generic negative phrases (e.g., âobjectâ, âthingsâ, âstuffâ, âtextureâ) are encoded and $L _ { 2 }$ -normalized to construct a contrastive subspace. Then, for each 3D Gaussian in the scene, we decode its continuous semantic embedding by multiplying its learned mixture weights with the shared codebook. We compute a promptconditioned relevance score natively in 3D by contrasting each Gaussianâs decoded vector against the encoded text phrases, applying a temperaturescaled softmax to penalize features that strongly correlate with negative structural noise. By thresholding these explicit 3D relevancy queries, we generate a prompt-specific Gaussian mask. The surviving subset of Gaussians can then be rasterized from any arbitrary viewpoint to isolate the target object, effectively performing deterministic, view-independent open-vocabulary 3D segmentation without requiring explicit bounding box inferences.

Generative VLM. To achieve higher-level scene understanding, the decoded 3D Gaussian semantic features can be routed directly into the vision tower of a Generative Vision-Language Model (VLM). However, passing the full set of $N _ { g }$ Gaussians into a generative model is computationally prohibitive and yields highly redundant tokens. Thus, we condense the scene into a compact sequence of M informative tokens using an Entropy-Adaptive Gaussian Sampling strategy.

Rather than relying on naive spatial downsampling, this strategy leverages the informationtheoretic uncertainty inherent in each Gaussianâs learned semantic state. Specifically, we evaluate the Shannon entropy of the semantic assignment probabilities produced by the VQ mixture logits. Gaussians exhibiting high entropyâoften corresponding to ambiguous object boundaries, interactive structural frontiers, or geometrically dense regionsâare actively prioritized. In contrast, Gaussians with low entropy tightly map to a single categorical cluster, indicating redundant, homogeneous backgrounds (like large flat floors or empty walls). We run a continuous Top-M sorting algorithm over these calculated entropy scores, explicitly extracting only the subset of Gaussians exhibiting peak semantic ambiguity. This deterministic sampling naturally distills out redundant backgrounds while preserving critical semantic boundaries and object structures. The resulting token sequence seamlessly bridges the continuous geometric scene representation with the VLMâs discrete context window for complex reasoning tasks, such as 3D visual question answering (VQA) and zero-shot scene captioning.

Potential Embodied AI Application. Finally, as a future direction for embodied AI applications, the X-GS-Thinker can be configured to interface with a Vision-Language-Action (VLA) model (Brohan et al., 2023). By feeding language-aligned 3D geometric features directly into a VLA, our system provides the real-time spatial information required to support embodied tasks.

## 4 Experiments

## 4.1 Implementation Details

Our 3DGS-based online SLAM pipeline for geometry and appearance builds upon MonoGS (Matsuki et al., 2024), which executes online SLAM via two concurrent threads: a tracking thread for camera pose estimation and a mapping thread for optimizing the geometry and appearance of the 3D Gaussians. During the online codebook learning stage (Li et al., 2025c), the EMA VQ codebook is instantiated with $K = 2 5 6$ independent semantic codes. The EMA update incorporates a decay momentum of $\lambda = 0 . 9 6$ and a temperature of $\tau = 0 . 7$ to smoothly adapt the cluster centers to the sequential frame stream.

To extract object boundaries, we employ SAM (Kirillov et al., 2023), utilizing its base backbone to obtain semantic region masks. To inject openvocabulary capabilities into the 3D Gaussians, we utilize CLIP (ViT-B/16) (Cherti et al., 2023) to extract full-resolution semantic maps. For generative tasks (e.g., scene captioning), we adopt SigLIP (Zhai et al., 2023) as its visual encoder. Specifically, we extract sparse, patch-level visual embeddings from SigLIP and align them with our grid-sampled pixels using nearest-neighbor interpolation.

At runtime, we maintain a sliding window of $N _ { w i n } = 8$ keyframes, allocating 150 optimization iterations for the radiance (geometry and appearance) phase, followed by 50 iterations dedicated solely to the semantic phase. The entire X-GS framework operates efficiently on a single NVIDIA V100 GPU.

## 4.2 X-GS-Perceiver Results

To evaluate the visual and semantic fidelity of our system, we present qualitative comparisons of scene reconstruction and open-vocabulary object localization in Figure 3. First, we observe that the proposed X-GS-Perceiver module maintains exceptional geometry and appearance tracking; the rendered RGB reconstructions exhibit high visual fidelity and closely match the ground truth (GT) images. Second, we evaluate the distilled semantic field. Despite running in an online fashion at realtime speeds, our framework successfully distills semantic information from the VFMs of SAM and CLIP. The integration of the VQ module and gridsampled supervision effectively balances speed and quality. Finally, we demonstrate the practical utility of our semantically enriched 3D scenes. By querying the distilled semantic field using text prompts, our framework computes per-Gaussian similarity scores to accurately localize specific objects in 3D space. This confirms that X-GS operates not merely as an efficient SLAM system, but successfully constructs a deeply comprehensible, queryable 3D semantic field ready for complex downstream tasks. We also include results for when X-GS receives additional depth map inputs in Figure 4. A quantitative comparison of these settings is provided in Table 2.

<!-- image-->  
Figure 3: Qualitative results of X-GS on scene reconstruction and semantic distillation. From left to right: Ground Truth (GT) RGB, Rendered RGB, GT Semantic Map (from VFMs, SAM + CLIP), Rendered Semantic Map, and an open-vocabulary Object Detection example.

Table 2: Comparison of RGB-only and RGB-D settings on TUM fr3/office. Rendering metrics are reported after color refinement.
<table><tr><td>Method</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>ATE RMSE â</td></tr><tr><td>RGB-only</td><td>27.039</td><td>0.8651</td><td>0.1698</td><td>0.01368</td></tr><tr><td>RGB-D</td><td>27.304</td><td>0.8797</td><td>0.1280</td><td>0.00952</td></tr></table>

## 4.3 X-GS-Thinker Results

To further demonstrate the multimodal generation capabilities of our framework, Figure 5 illustrates qualitative results for 3D scene captioning. By instantiating LLaVA-OneVision (Li et al., 2025a)â backed by the Qwen2-7B-Instruct LLM (Yang et al., 2024)âas the X-GS-Thinker, our system directly ingests the semantically enriched 3D Gaussians reconstructed by the X-GS-Perceiver. As shown in the figure, the Thinker successfully leverages these spatial and semantic features to generate coherent, natural language descriptions of the environment, accurately capturing both individual object properties and complex global scene layouts.

<!-- image-->  
Figure 4: Qualitative results of X-GS using RGB-D inputs. Depth map is optional and not the required input by X-GS-Perceiver.

## 4.4 Computational Footprint Analysis

To evaluate the efficiency and resource requirements of our system, we present a computational footprint analysis of the X-GS-Perceiver in Table 3. The table breaks down the processing duration across key pipeline components during keyframe optimization, including the time allocated for geometry, appearance, and semantic updates. Additionally, we report the overall system throughput and GPU memory load. Thanks to our highly parallelized scheduling, the heavy optimization tasks run efficiently in the background, allowing the framework to maintain real-time tracking performance on a single consumer-grade GPU.

<!-- image-->  
Figure 5: Qualitative results of X-GS for 3D scene caption generation. The VLM generates captions in a zero-shot setting by taking 3D semantic Gaussians as input. Note that only a portion of the generated content is shown.

## 5 Conclusion

In this paper, we introduced X-GS, an extensible framework that unifies previously isolated 3DGS domainsâincluding pose-free GS, online SLAM, and semantic GSâinto a single cohesive system. To achieve real-time performance for X-GS-Perceiver, we overcome computation bottlenecks using an online VQ module with EMA updates, GPU-accelerated grid sampling, and highly parallelized scheduling. Furthermore, we demonstrated the frameworkâs capability for multimodal reasoning through X-GS-Thinker, a module that bridges our 3D semantic Gaussians with downstream VLMs. By natively supporting multimodal tasks such as open-vocabulary object detection, scene captioning, and embodied tasks, X-GS establishes a robust, modular foundation for spatial reasoning.

Table 3: Computational footprint analysis of X-GS-Perceiver. We employ SigLIP as its VFM and evaluate the pipeline on a single NVIDIA V100 GPU.
<table><tr><td>Main Component</td></tr><tr><td>Geometry &amp; Appearance Optimization 14.0s Semantic Supervision Target Preparation Vision Encoding 0.2s</td></tr><tr><td>VQ Codebook Update 2.3s Grid-Sampled Target Prefetching 4.3s Semantic Optimization 11.2s</td></tr><tr><td>Average Time per Keyframe 25.2s</td></tr><tr><td>Average Time per Frame 2.8s</td></tr><tr><td>Average FPS 21.4 GPU Memory Load ~9 GB</td></tr></table>

## Limitations and Future Work

While X-GS establishes a highly extensible framework, its multimodal reasoning capabilities require more comprehensive evaluation across diverse embodied and 3D vision-language benchmarks. First, the interface with the X-GS-Thinker currently operates in a modular fashion; future work will explore end-to-end fine-tuning of the downstream VLMs natively within the 3DGS representations (Thai et al., 2025). Second, as demonstrated by recent systems like FreeSplat (Wang et al., 2024b), SplatTalk (Thai et al., 2025), EA3D (Zhou et al., 2025), and EmbodiedSplat (Lee et al., 2026), there is a growing trend toward feed-forward architectures for 3DGS. Integrating such feed-forward mechanisms into X-GS could significantly reduce the current optimization overhead, but such models also require significant training cost for different VFMs. Third, leveraging emerging 3D vision foundation modelsâsuch as DUSt3R (Wang et al., 2024a), MASt3R (Leroy et al., 2024), or VGGT (Wang et al., 2025)âcould provide stronger geometric priors for mapping, further enhancing the robustness of our pose-free tracking pipeline. Finally, accommodating dynamic scenes using techniques such as 4DGS (Wu et al., 2024; Gao et al., 2025) represents a highly promising direction for real-world embodied AI applications.

## References

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, and 35 others. 2023. RT-2: vision-languageaction models transfer web knowledge to robotic control. CoRR, abs/2307.15818.

Hanlin Chen, Fangyin Wei, and Gim Hee Lee. 2024. Chatsplat: 3d conversational gaussian splatting. CoRR, abs/2412.00734.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. 2023. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818â2829.

Yang Fu, Xiaolong Wang, Sifei Liu, Amey Kulkarni, Jan Kautz, and Alexei A. Efros. 2024. Colmap-free 3d gaussian splatting. In CVPR, pages 20796â20805. IEEE.

Zhongpai Gao, Benjamin Planche, Meng Zheng, Anwesa Choudhuri, Terrence Chen, and Ziyan Wu. 2025. 7dgs: Unified spatial-temporal-angular gaussian splatting. CoRR, abs/2503.07946.

Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139:1â139:14.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, ChloÃ© Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr DollÃ¡r, and Ross B. Girshick. 2023. Segment anything. CoRR, abs/2304.02643.

Seungjun Lee, Zihan Wang, Yunsong Wang, and Gim Hee Lee. 2026. Embodiedsplat: Online feedforward semantic 3dgs for open-vocabulary 3d scene understanding. arXiv preprint arXiv:2603.04254.

Sibaek Lee, Seongbo Ha, Kyeongsu Kang, Joonyeol Choi, Seungjun Tak, and Hyeonwoo Yu. 2025. LEGO-SLAM: language-embedded gaussian optimization SLAM. CoRR, abs/2511.16144.

Vincent Leroy, Yohann Cabon, and JÃ©rÃ´me Revaud. 2024. Grounding image matching in 3d with mast3r. In ECCV (72), volume 15130 of Lecture Notes in Computer Science, pages 71â91. Springer.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2025a. Llavaonevision: Easy visual task transfer. Trans. Mach. Learn. Res., 2025.

Fei-Fei Li, Justin Johnson, Christoph Lassner, and Ben Mildenhall. 2024. What is spatial intelligence?

Haoyuan Li, Yanpeng Zhou, Tao Tang, Jifei Song, Yihan Zeng, Michael Kampffmeyer, Hang Xu, and Xiaodan Liang. 2025b. Unigs: Unified language-image-3d pretraining with gaussian splatting. In ICLR. Open-Review.net.

Wanhua Li, Yujie Zhao, Minghan Qin, Yang Liu, Yuanhao Cai, Chuang Gan, and Hanspeter Pfister. 2025c. Langsplatv2: High-dimensional 3d language gaussian splatting with 450+ FPS. CoRR, abs/2507.07136.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. CoRR, abs/2304.08485.

Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. 2024. Gaussian splatting SLAM. In CVPR, pages 18039â18048. IEEE.

Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. 2024. Langsplat: 3d language gaussian splatting. In CVPR, pages 20051â20060. IEEE.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In ICML, volume 139 of Proceedings of Machine Learning Research, pages 8748â8763. PMLR.

Anh Thai, Songyou Peng, Kyle Genova, Leonidas J. Guibas, and Thomas A. Funkhouser. 2025. Splattalk: 3d VQA with gaussian splatting. CoRR, abs/2503.06271.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David NovotnÃ½. 2025. VGGT: visual geometry grounded transformer. In CVPR, pages 5294â5306. Computer Vision Foundation / IEEE.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and JÃ©rÃ´me Revaud. 2024a. Dust3r: Geometric 3d vision made easy. In CVPR, pages 20697â20709. IEEE.

Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. 2024b. Freesplat: Generalizable 3d gaussian splatting towards free view synthesis of indoor scenes. In NeurIPS.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 2024. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, pages 20310â20320. IEEE.

Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. 2024. GS-SLAM: dense visual SLAM with 3d gaussian splatting. In CVPR, pages 19595â19604. IEEE.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and 43 others. 2024. Qwen2 technical report. CoRR, abs/2407.10671.

Jisang Yoo, Gyeongjin Kang, Hyun-kyu Ko, Hyeonwoo Yu, and Eunbyung Park. 2025. Openmonogs-slam: Monocular gaussian splatting SLAM with open-set semantics. CoRR, abs/2512.08625.

Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Oswald. 2023. Gaussian-slam: Photorealistic dense SLAM with gaussian splatting. CoRR, abs/2312.10070.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In ICCV, pages 11941â11952. IEEE.

Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. 2024. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In CVPR, pages 21676â21685. IEEE.

Xiaoyu Zhou, Jingqi Wang, Yuang Jia, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. 2025. EA3D: online open-world 3d object extraction from streaming videos. CoRR, abs/2510.25146.

## A 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) represents a 3D scene using a collection of anisotropic 3D Gaussians, operating as a point-based alternative to continuous neural implicit representations. Each Gaussian i in the comprehensive map parameter set Î is explicitly defined by the following attributes:

â¢ Mean vector (Position): $\mu _ { i } \in \mathbb { R } ^ { 3 }$

â¢ Covariance matrix: $\Sigma _ { i }$

â¢ Opacity: $\alpha _ { i } \in [ 0 , 1 ]$

â¢ Color features: $c _ { i }$ (normally represented as Spherical Harmonics for view-dependence, though often simplified to base RGB values in dense SLAM configurations).

The unnormalized spatial distribution of a 3D Gaussian evaluated at a spatial point x is defined as:

$$
G ( x ) = \exp \left( - { \frac { 1 } { 2 } } ( x - \mu _ { i } ) ^ { T } \Sigma _ { i } ^ { - 1 } ( x - \mu _ { i } ) \right)\tag{19}
$$

To maintain positive semi-definiteness during gradient-based optimization, the covariance matrix $\Sigma _ { i }$ is parameterized by a scaling matrix $S _ { i }$ and a rotation matrix $R _ { i }$ :

$$
\Sigma _ { i } = R _ { i } S _ { i } S _ { i } ^ { T } R _ { i } ^ { T }\tag{20}
$$

In order to render these primitives onto the image plane from a given camera pose, the 3D Gaussians are "splatted" to 2D. Given a learned viewing transformation W and the Jacobian J of the affine approximation of the projective alignment, the resulting 2D covariance matrix $\Sigma _ { i } ^ { \prime }$ is calculated as:

$$
\Sigma _ { i } ^ { \prime } = J W \Sigma _ { i } W ^ { T } J ^ { T }\tag{21}
$$

This continuous state update sets a framework perfectly suited for dense tracking and mapping.

## B 3DGS-SLAM Objectives

Continuous optical depth models compute the expected color C along a camera ray by integrating transmittance T (t), volume density Ï(t), and color $c ( t )$ . 3DGS discretizes this continuous integral using a point-based approximation. For a set of N sorted 3D Gaussians projected onto the 2D image plane given a camera pose $\mathbf { T } \in \mathrm { S E } ( 3 )$ , the color at pixel x evaluates to:

$$
C ( x \mid \mathbf { T } , \Theta ) = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{22}
$$

where $\alpha _ { i } ^ { \prime }$ is the 2D footprint opacity. Depth $D ( x )$ is similarly rendered by substituting color $c _ { i }$ with the ray depth $d _ { i }$

Camera tracking is performed via Analysis-by-Synthesis. For an incoming frame at timestamp t, the pose is initialized via a constant velocity model. Because the 3DGS rasterizer is natively differentiable, precise tracking is achieved by freezing the map parameters Î and running gradient descent on the camera extrinsic parameters to minimize the geometric and photometric depth losses (if depth inputs are available):

$$
\mathcal { L } _ { \mathrm { t r a c k } } = \mathcal { L } _ { \mathrm { p h o t o } } ( I _ { t } , \hat { I } _ { t } ) + \lambda _ { d } \| D _ { t } - \hat { D } _ { t } \| _ { 1 }\tag{23}
$$

The global mapping thread optimizes the scene parameters Î over a sliding window of keyframes K using the same combined loss, augmented with an isotropic regularization term $\mathcal { L } _ { \mathrm { i s o } }$ to prevent heavily elongated artifacts:

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { m a p } } = \sum _ { k \in \mathcal { K } } \Big ( \mathcal { L } _ { \mathrm { p h o t o } } ( I _ { k } , \hat { I } _ { k } ) + \lambda _ { d } \| D _ { k } - \hat { D } _ { k } \| _ { 1 } \Big ) } \quad } & { } \\ & { + \lambda _ { \mathrm { i s o } } \mathcal { L } _ { \mathrm { i s o } } } \end{array}\tag{24}
$$

## C Additional Details about Online EMA Codebook Updates

To stabilize the online EMA updates introduced in the main text, we implement active dead-code management and a warm-start initialization process.

Instead of starting from uniform counts, we initialize the EMA statistics using a confidencefiltered assignment pass over an initial buffer of semantic features. We transform the observed feature distances into confidence scores using a temperature parameter Ï :

$$
p _ { n k } = \frac { \exp ( - \| \mathbf { x } _ { n } - \mathbf { e } _ { k } \| _ { 2 } / \tau ) } { \sum _ { j = 1 } ^ { K } \exp ( - \| \mathbf { x } _ { n } - \mathbf { e } _ { j } \| _ { 2 } / \tau ) }\tag{25}
$$

Only samples where the maximum confidence exceeds a threshold $\gamma$ are allowed to contribute to the initialization of the counts $N _ { k }$ and sums ${ { \bf { M } } _ { k } }$ :

$$
m _ { n } = \mathbb { I } \bigg [ \operatorname* { m a x } _ { k } p _ { n k } \ge \gamma \bigg ]\tag{26}
$$

During the continuous online tracking stage, the confidence score masks unreliable incoming samples, while the actual codeword assignments remain hard nearest-neighbor to ensure discrete latent separation.

Furthermore, to handle dead codes escaping the tracked field of view, we explicitly monitor the EMA accumulated mass. If a codewordâs utilization falls below a dead-code threshold $( N _ { k } < \delta )$ , it is immediately reinitialized:

$$
\mathbf { M } _ { k } \gets \tilde { \mathbf { x } } , \qquad N _ { k } \gets 1 + \varepsilon\tag{27}
$$

where xË is sampled directly from a small reservoir of recently observed historical semantic features.

## D Extraction of Grid-Sampled Semantic Targets

As formulated in the main text, the grid-sampled supervision relies on a target map $\mathbf { G } ^ { \star }$ and validity mask M. We populate these tensors through two distinct pathways depending on the downstream semantic source:

1. Precomputed Region-Indexed Annotations: For explicit segmentation masks $S \in$ $\{ - 1 , 0 , \dotsc , R - 1 \} ^ { H \times W }$ (where â1 denotes background) and a corresponding region-feature table $\bar { \Phi } \in \mathbb { R } ^ { R \times D }$ , we directly query the table at the sampled grid location $( u , v )$

$$
\begin{array} { r l } & { \quad r _ { m , n } = S ( u , v ) , } \\ & { \quad M _ { m , n } = \mathbb { I } [ r _ { m , n } \neq - 1 ] , } \\ & { \quad \mathbf { G } _ { : , m , n } ^ { \star } = \{ \Phi _ { r _ { m , n } } , \quad r _ { m , n } \neq - 1 ,  } \\ & { \quad  \mathbf { G } _ { : , m , n } ^ { \star } = \{ \mathbf { 0 } , \qquad r _ { m , n } = - 1 .  } \end{array}\tag{28}
$$

2. Online Continuous Feature Maps: For dense semantic features streaming from an online VLM at abstract lower resolutions $\mathrm { ~ { ~ \bf ~ P ~ } ~ } \in$ $\mathbb { R } ^ { D \times H _ { f } \times W _ { f } }$ , we dynamically align the image coordinates to the low-resolution grid coordinates:

$$
\begin{array} { r l r } & { } & { \tilde { u } _ { m , n } = \mathrm { r o u n d } \left( u \cdot \cfrac { H _ { f } - 1 } { \operatorname* { m a x } ( H - 1 , 1 ) } \right) , } \\ & { } & { \tilde { v } _ { m , n } = \mathrm { r o u n d } \left( v \cdot \cfrac { W _ { f } - 1 } { \operatorname* { m a x } ( W - 1 , 1 ) } \right) , } \end{array}\tag{29}
$$

yielding the target $\mathbf { G } _ { : , m , n } ^ { \star } = \mathbf { P } _ { : , \tilde { u } _ { m , n } , \tilde { v } _ { m , n } }$ with an unconstrained mask $M _ { m , n } = 1$

Finally, to prevent the optimized point-cloud from overfitting to a static grid lattice, the offset $\mathbf { o } = \left( o _ { h } , o _ { w } \right)$ is dynamically randomized over optimization steps. By drawing the offset from the valid domain of phase shifts $\mathbf { o } _ { t } \in \{ 0 , \ldots , s - 1 \} ^ { 2 }$ the sampled grid smoothly traverses and covers all $s ^ { 2 }$ possible pixel phases over the course of sequential mapping.

## E Additional Details on Object Detection

Given a text prompt p, we encode it with OpenCLIP and normalize the resulting embedding:

$$
\mathbf { t } ^ { + } = \frac { E _ { \mathrm { t e x t } } ( p ) } { \Vert E _ { \mathrm { t e x t } } ( p ) \Vert _ { 2 } } .\tag{30}
$$

We also encode a small set of generic negative phrases

$$
\mathcal { N } = \{ \mathrm { o b j e c t , ~ t h i n g s , ~ s t u f f , ~ t e x t u r e } \} ,\tag{31}
$$

with normalized embeddings

$$
\mathbf { t } _ { n } ^ { - } = \frac { E _ { \mathrm { t e x t } } ( n ) } { \| E _ { \mathrm { t e x t } } ( n ) \| _ { 2 } } , \qquad n \in \mathcal { N } .\tag{32}
$$

For semantic level â, the renderer reconstructs the feature at pixel $( u , v )$ from the rendered code weights and the learned codebook:

$$
\mathbf { f } _ { u v } ^ { ( \ell ) } = \sum _ { j = 1 } ^ { C _ { \ell } } w _ { u v , j } ^ { ( \ell ) } \mathbf { c } _ { j } ^ { ( \ell ) } , \qquad \hat { \mathbf { f } } _ { u v } ^ { ( \ell ) } = \frac { \mathbf { f } _ { u v } ^ { ( \ell ) } } { \left. \mathbf { f } _ { u v } ^ { ( \ell ) } \right. _ { 2 } + \varepsilon } .\tag{33}
$$

The prompt relevance at that pixel is computed by contrasting the positive query against each negative phrase:

$$
R _ { u v } ^ { ( \ell ) } ( p ) = \operatorname* { m i n } _ { n \in \mathcal { N } } \mathrm { s o f t m a x } \left( \tau \left[ \hat { \mathbf { f } } _ { u v } ^ { ( \ell ) \top } \mathbf { t } ^ { + } \right] \right) _ { 1 } ,\tag{34}
$$

where $( \cdot ) _ { 1 }$ denotes the probability assigned to the positive prompt and $\tau = 1 0$ in the implementation. The final per-pixel relevance is obtained by taking the strongest response across levels:

$$
R _ { u v } ( p ) = \operatorname* { m a x } _ { \ell } R _ { u v } ^ { ( \ell ) } ( p ) .\tag{35}
$$

In the prompt-rendering variant, the same idea is also applied directly to individual Gaussians. If Gaussian i has decoded semantic feature

$$
\mathbf { g } _ { i } = \sum _ { j = 1 } ^ { C } a _ { i j } \mathbf { c } _ { j } , \qquad \hat { \mathbf { g } } _ { i } = \frac { \mathbf { g } _ { i } } { \lVert \mathbf { g } _ { i } \rVert _ { 2 } + \varepsilon } ,\tag{36}
$$

its prompt relevance score is

$$
r _ { i } ( p ) = \operatorname* { m i n } _ { n \in \mathcal { N } } \mathrm { s o f t m a x } \bigg ( \tau \left[ \frac { \hat { \mathbf { g } } _ { i } ^ { \top } \mathbf { t } ^ { + } } { \hat { \mathbf { g } } _ { i } ^ { \top } \mathbf { t } _ { n } ^ { - } } \right] \bigg ) _ { 1 } .\tag{37}
$$

A prompt-specific Gaussian mask is then obtained by thresholding

$$
m _ { i } ( p ) = \mathbb { I } [ r _ { i } ( p ) > \delta ] .\tag{38}
$$

If no Gaussian survives the threshold, the implementation falls back to keeping the top-scoring subset.

## F Entropy-Adaptive Gaussian Sampling for Generative VLMs

As demonstrated by recent works such as SplatTalk (Thai et al., 2025), projecting 3D Gaussian features into discrete tokens enables Large Language Models (LLMs) to perform zero-shot 3D visual question answering and scene understanding. However, directly passing all $N _ { g }$ Gaussians from an uncompressed SLAM map into the vision tower of a Generative VLM is computationally prohibitive and yields highly redundant tokens, particularly in homogeneous regions like walls and floors.

To bridge our dense semantic 3DGS map with the downstream Generative VLM, we condense the scene into a compact sequence of M informative tokens $( M \ll N _ { g } )$ using an Entropy-Adaptive Gaussian Sampling strategy. This approach leverages the continuous-to-discrete probability distribution generated by the online Vector Quantization (VQ) module in our X-GS-Perceiver.

Let $\mathbf { f } _ { i } \in \mathbb { R } ^ { D }$ represent the semantic feature logits of the i-th Gaussian over the D discrete semantic clusters. We first convert these features into a semantic assignment probability distribution $p _ { i , k }$ using a standard softmax operator across the feature dimension:

$$
p _ { i , k } = \frac { \exp ( f _ { i , k } ) } { \sum _ { j = 1 } ^ { D } \exp ( f _ { i , j } ) }\tag{39}
$$

The semantic entropy $H _ { i }$ of the i-th Gaussian is then calculated as the Shannon entropy of this assignment distribution:

$$
H _ { i } = - \sum _ { k = 1 } ^ { D } p _ { i , k } \log p _ { i , k }\tag{40}
$$

Gaussians with a low entropy $H _ { i }$ confidently match a single semantic cluster, typically representing redundant, homogeneous background surfaces. Conversely, Gaussians with a high entropy $H _ { i }$ denote semantic ambiguityâindicating object boundaries, transitions between distinct instances, or complex structural information holding higher contextual value for the VLM.

To select the final M tokens, we deploy a deterministic Top-M sorting mechanism over the computed entropy scores. We directly extract the subset of Gaussians S that exhibit the highest semantic ambiguity:

$$
S = \arg \tan _ { i \in \{ 1 , . . . , N _ { g } \} } H _ { i }\tag{41}
$$

The resulting M Gaussians are then projected into the VLMâs embedding space. This strategy guarantees that the generative VLM receives an informative and tightly packed token sequence, actively filtering out redundant backgrounds while preserving critical object boundaries and structural frontiers required for robust multimodal reasoning.