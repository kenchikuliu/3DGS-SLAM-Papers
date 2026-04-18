# SPFSplatV2: Efficient Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views

Ranran Huang, Krystian Mikolajczyk

AbstractâWe introduce SPFSplatV2, an efficient feed-forward framework for 3D Gaussian splatting from sparse multi-view images, requiring no ground-truth poses during training and inference. It employs a shared feature extraction backbone, enabling simultaneous prediction of 3D Gaussian primitives and camera poses in a canonical space from unposed inputs. A masked attention mechanism is introduced to efficiently estimate target poses during training, while a reprojection loss enforces pixel-aligned Gaussian primitives, providing stronger geometric constraints. We further demonstrate the compatibility of our training framework with different reconstruction architectures, resulting in two model variants. Remarkably, despite the absence of pose supervision, our method achieves stateof-the-art performance in both in-domain and out-of-domain novel view synthesis, even under extreme viewpoint changes and limited image overlap, and surpasses recent methods that rely on geometric supervision for relative pose estimation. By eliminating dependence on ground-truth poses, our method offers the scalability to leverage larger and more diverse datasets. Code and pretrained models will be available on our project page: https://ranrhuang.github.io/spfsplatv2/.

Index TermsâGaussian Splatting, novel view synthesis, selfsupervised, pose-free, efficiency.

## I. INTRODUCTION

R Ecent advancements in 3D reconstruction and novel viewsynthesis (NVS) have been driven by Neural Radiance synthesis (NVS) have been driven by Neural Radiance Fields (NeRFs) [1] and 3D Gaussian splatting (3DGS) [2]. A standard training pipeline for novel view synthesis reconstructs a 3D scene from input views and optimizes it by aligning rendered novel views with ground-truth images [3]â[8].

State-of-the-art methods typically employ geometry-aware architectures by constructing cost volumes [3], [4], [6], leveraging epipolar transformers [5], or encoding camera poses using Plucker ray embeddings [ Â¨ 9]â[11]. These approaches rely on camera poses estimated with Structure-from-Motion (SfM) [12] to reconstruct 3D scenes, as illustrated in Fig. 1 (a). However, acquiring camera poses from SfM is computationally expensive and often unreliable in sparse-view scenarios due to insufficient correspondences, limiting the applicability of these pose-required methods. To address this, recent research has focused on novel view synthesis under pose-free settings.

Existing pose-free methods reconstruct 3D scenes from unposed images by learning in a canonical space [7], [8], [13], [14], leveraging latent scene representations [15], [16], or jointly optimizing both context-view camera poses and

<!-- image-->  
Fig. 1. Comparison of three typical training pipelines for sparse-view 3D reconstruction in novel view synthesis. For simplicity, the image rendering loss on the rendered target view is omitted. (a) Pose-required methods rely on ground-truth poses for both 3D scene reconstruction and target-view rendering. (b) Supervised pose-free methods requires no ground-truth poses for reconstruction but still rely on ground-truth poses for rendering loss. (c) Our self-supervised pose-free pipeline instead leverages estimated target poses to optimize 3D scene reconstruction from unposed images, thereby removing dependence on ground-truth poses during both training and inference.

3D scene representations [17]â[19]. Although these methods do not require accurate poses at inference, their training is still supervised by rendering losses given ground-truth poses at novel viewpoints, as shown in Fig. 1 (b). We therefore categorize these approaches as supervised pose-free methods. As a result, their reliance on training datasets with known camera poses restricts the scalability to large-scale real-world data without pose annotations.

This raises a critical question: Are ground-truth poses truly indispensable for optimizing 3D scenes during training? One solution is to use estimated poses at novel viewpoints, referred to as the self-supervised pose-free paradigm in Fig. 1 (c). However, this presents an inherent challenge: since the rendering loss intrinsically couples the learning of 3D scene geometry and camera poses, pose errors can degrade reconstruction quality, which further hampers pose estimation. Such mutual dependency creates a feedback loop that can potentially lead to unstable training or even divergence. Recent self-supervised pose-free approaches [20], [21] struggle to mitigate this issue primarily due to their use of separate and cascading modules for scene reconstruction and pose estimation, discouraging the learning of consistent feature representations across the two tasks and impairing geometric consistency. Consequently, these methods exhibit poor training stability, particularly under large viewpoint changes, and still lag far behind state-of-theart pose-required and supervised pose-free methods [5]â[7].

<!-- image-->  
Fig. 2. Training pipeline of SPFSplatV2. A shared backbone with three specialized heads simultaneously predicts Gaussian centers, additional Gaussian parameters, and camera poses from unposed images in a canonical space, with the first input view as the reference. Encoder tokens, concatenated with a learnable pose token and an optional embedding of ground-truth intrinsics, are fed into the decoder, which employs masked attention to prevent context tokens from attending to target tokens, ensuring Gaussian reconstruction remains independent of target-view information. The 3D Gaussians are optimized via a rendering loss using the predicted target poses, while a reprojection loss enforces alignment between Gaussian centers and their corresponding pixels using the predicted context poses. By jointly optimizing Gaussians and camera poses, the pipeline enhances geometric consistency and improves reconstruction quality.

To address the challenge, we introduce SPFSplatV2, a selfsupervised pose-free approach for 3D Gaussian splatting from unposed sparse views. As shown in Fig. 2, SPFSplatV2 employs a shared backbone for feature extraction with dedicated heads for predicting 3D Gaussian primitives and camera poses relative to a reference view. The unified backbone improves computational efficiency and facilitates joint feature learning for scene reconstruction and pose estimation, thereby enhancing geometric consistency and mitigating feedback instability. This is achieved by enabling 3D geometry to benefit from context-aware camera alignment and allowing pose predictions to leverage global scene context.

During training, in addition to context images, the target images are also incorporated as input for target pose estimation, enabling rendering losses at target views. To prevent information leakage from target images into the Gaussian reconstruction of context views, we introduce a masked attention mechanism, as shown in Fig. 2. In this design, context tokens attend only to context tokens, ensuring that 3D Gaussian reconstruction remains independent of target-view information. Conversely, target tokens attend to both context and target tokens, allowing the model to exploit global scene context for accurate target pose estimation. Finally, we complement the rendering loss with a reprojection loss that explicitly enforces alignment between the predicted Gaussians and their corresponding image pixels, imposing stronger geometric constraints and further enhancing training stability. In conclusion, we make the following key contributions:

â¢ We propose SPFSplatV2, a feed-forward framework with masked attention that enables efficient and stable joint optimization of scene reconstruction and pose estimation from sparse unposed views, requiring no ground-truth poses during training and inference.

â¢ SPFSplatV2 outperforms state-of-the-art pose-required, supervised pose-free, and self-supervised pose-free methods on both in-domain and out-of-domain novel view synthesis, demonstrating robustness under limited view overlap and extreme viewpoint changes. Despite relying solely on image supervision, its efficient feed-forward relative pose estimation surpasses most approaches that depend on geometric supervision.

â¢ By eliminating the reliance on ground-truth poses during training, our method offers the scalability needed to leverage larger and more diverse datasets. Its effectiveness across different architectures further demonstrates the paradigmâs broad compatibility.

This work substantially extends our previous method, SPF-Splat [22], with the key novelties summarized as follows:

â¢ Methodological Improvements: Different from SPFSplat, which employs separate context-only and context-withtarget input branches to avoid target information leakage, we introduce a unified architecture with masked attention mechanism that reduces computational overhead and pose misalignment. Pose estimation is further improved with learnable pose tokens, which selectively attend to relative multi-view cues for more accurate camera inference. In addition, a multi-view dropout strategy enhances generalization across varying numbers and spatial distributions of context views.

â¢ Architectural Compatibility: We demonstrate that our training paradigm is compatible with state-of-the-art reconstruction models. To this end, we develop two variants: SPFSplatV2, which follows a MASt3R-style [23] architecture (consistent with SPFSplat), and SPFSplatV2- L, which adopts the VGGT [24] architecture.

â¢ Superior Performance: Extensive experiments show that SPFSplatV2 and SPFSplatV2-L achieve significant improvements over SPFSplat [22] and other state-of-the-art methods in novel view synthesis, cross-domain generalization and relative pose estimation.

## II. RELATED WORK

## A. Novel View Synthesis

NeRF [1] and 3DGS [2] have demonstrated strong performance in 3D reconstruction and novel view synthesis. Early methods rely on dense input views for per-scene optimization [25]â[28], whereas recent approaches focus on generalizable reconstruction from sparse-view images [3]â[11]. Typical NVS pipelines reconstruct 3D scenes from input views and optimize them by aligning synthesized images to ground-truth targets. Based on their dependence on ground-truth camera poses during training and inference, existing methods can be grouped into pose-required, supervised pose-free, and selfsupervised pose-free approaches, as illustrated in Fig. 1.

Pose-Required Methods reconstruct 3D scenes from images given accurate poses using geometry-aware architectures [3]â [6], [9]â[11]. For example, MVSNeRF [3] and MuRF [4] construct cost volumes for multi-view aggregation to reconstruct radiance fields, while MVSplat [6] uses cost volumes for depth estimation to reconstruct Gaussian primitives. Other strategies include epipolar transformers in pixelSplat [5] or encoding camera poses with Plucker ray embeddings [ Â¨ 9]â [11]. Despite their effectiveness, these methods depend on SfM for precise camera poses, which is computationally expensive and often unreliable in sparse-view scenarios. Recent pose estimation techniques [23], [29]â[32] mitigate some issues but still struggle in low-overlap or texture-less settings. Consequently, pose-required methods remain impractical for unposed reconstruction during both training and inference.

Supervised Pose-Free Methods enable 3D reconstruction from unposed images, relaxing the need for camera poses at inference. Methods such as UpSRT [16] and UpFusion [15] encode unposed images into latent scene representations, while BARF [19], SPARF [33], DBARF [18], and CoPoNeRF [17] jointly optimize poses and NeRF representations. LEAP [13] and PF-LRM [14] leverage ViT architectures to define neural volumes in canonical camera coordinates. More recently, Splatt3R [8] predicts 3D Gaussians in a canonical space by regressing offsets to pointmaps from a frozen MASt3R [23], but requires depth supervision. NoPoSplat [7] removes this depth reliance and refines this pipeline by fine-tuning MASt3R and incorporating intrinsics to mitigate scale ambiguity. However, despite removing pose requirements at inference, these methods still depend on ground-truth poses during training via rendering losses [7], [8], [13]â[16], explicit pose supervision [17], or coarse pose initialization [19], [33], therefore limiting scalability to large-scale unposed real-world data.

Self-Supervised Pose-Free Methods completely eliminate the reliance on ground-truth poses during training by enabling rendering losses at novel viewpoints using estimated poses, as shown in Fig. 1 (c). For instance, Nope-NeRF [34], CF-3DGS [35], and FlowCam [36] reconstruct 3D scenes and estimate camera poses incrementally by re-rendering dense video sequences. However, they are limited to continuous video frames and do not generalize well to sparse views. Recent self-supervised pose-free methods, such as PF3plat [20] and SelfSplat [21], attempt to estimate both input- and target-view poses from sparse views. PF3plat relies on off-the-shelf feature descriptors [37] with RANSAC-based initialization, yielding a pipeline that is inefficient and not end-to-end trainable, thereby limiting representational capacity. In contrast, SelfSplat employs cross-view U-Nets [38], [39] for pose prediction, but its performance remains weak particularly under large viewpoint changes due to the lack of geometric priors. Beyond the limitations in pose estimation, both methods separate pose prediction and Gaussian reconstruction into distinct modules, resulting in unshared features, weaker geometric alignment, and higher computational overhead. Moreover, both follow a local-to-global strategy: per-pixel depth is first predicted for each view, then lifted into world coordinates using the estimated poses. Pose errors at this stage can directly corrupt the lifted 3D points, degrading reconstruction and amplifying instability through feedback. Consequently, these approaches suffer from unstable training and exhibit a large performance gap compared to state-of-the-art methods.

SPFSplat [22], our previous approach, also adopts a selfsupervised, pose-free paradigm. This is accomplished by jointly optimizing 3D Gaussians and camera poses through a shared backbone in a canonical space, guided by both image rendering and reprojection losses. The unified backbone ensures that pose estimation is informed by the same scene geometry that drives Gaussian prediction, thereby promoting geometric consistency and improving training stability. Building upon SPFSplat, we introduce masked attention to reduce computational cost and alleviate potential pose misalignment, enhance pose estimation through learnable pose tokens, and further incorporate a multi-view dropout strategy, which together lead to improved overall performance and generalization across multiple views.

## B. Structure-from-Motion (SfM)

Structure-from-Motion (SfM) [40], [41] is a core problem in computer vision that jointly estimates camera parameters and reconstructs sparse 3D maps from image collections. Classical SfM pipelines typically involve local feature detection and matching [42]â[44], geometric verification via epipolar geometry or homographies with RANSAC [45], triangulation [46] to recover 3D points, and bundle adjustment [47] to refine poses and structure. Recent advances have incorporated learning-based components into SfM, including robust feature descriptors [48]â[51], improved image matching [37], [52], detector-free matching [53], and neural bundle adjustment [19], [54]. However, the sequential design of SfM pipelines remains prone to error propagation. To overcome this, fully differentiable pipelines have been introduced [23], [24], [29], [55]â[57]. For example, VGGSfM [55] enables endto-end sparse reconstruction, while DUSt3R [29], following the architecuture of CroCo [58], performs dense 3D reconstruction without camera parameters. MASt3R [23] further enhances feature matching and local representations but, like DUSt3R, remains constrained by pairwise architectures and costly global optimization, which often fail in multi-view settings. Extensions such as MV-DUSt3R+ [56] and Fast3R [57] address multi-view reconstruction, while FLARE [59] employs cascaded learning with pose as the central bridge. Recently, VGGT [24] introduces a feed-forward transformer that jointly infers camera parameters, depth, point maps, and 3D tracks, achieving state-of-the-art results.

Similar to these SfM methods, our method jointly predicts 3D points and poses, with rendering and reprojection losses acting as a differentiable form of bundle adjustment to refine both geometry and poses. Unlike prior work, it requires no ground-truth geometric priors during training. In addition, the training paradigm is naturally compatible with advanced reconstruction backbones such as MASt3R [23] and VGGT [24], giving rise to two variants: SPFSplatV2 and SPFSplatV2-L.

## III. METHOD

We aim to learn a feed-forward network that reconstructs 3D Gaussians from unposed images while simultaneously estimating the camera poses. During training, the 3D Gaussians are optimized by rendering photorealistic images from the estimated poses at target views, thereby eliminating the need for ground-truth poses.

## A. Problem Formulation

Consider N context images $\lbrace I ^ { v } \rbrace _ { v = 1 , } ^ { N } \ \mathbf { a }$ s input. During training, additional M target images $\{ \bar { I } ^ { v } \} _ { v = N + 1 } ^ { \tilde { N } + M }$ are provided, resulting in a total of $V = N { + } M$ views.

3D Gaussian Reconstruction: Following [7], [8], we predict 3D Gaussians from context images in a canonical 3D space where the first input view $I ^ { 1 }$ serves as the global coordinate frame. The reconstruction network is formulated as:

$$
f _ { \pmb \theta } : \{ \pmb { I } ^ { v } \} _ { v = 1 } ^ { N } \mapsto \{ \pmb { \mathscr G } ^ { v \to 1 } \} _ { v = 1 } ^ { N } ,\tag{1}
$$

where $\begin{array} { r l r } { \mathscr { G } ^ { v \to 1 } } & { = } & { \{ ( \mu _ { j } ^ { v \to 1 } , r _ { j } ^ { v \to 1 } , s _ { j } ^ { v } , c _ { j } ^ { v \to 1 } , \alpha _ { j } ^ { v } ) \} _ { j = 1 , \dots , H \times W } } \end{array}$ represents the pixel-aligned Gaussians for $I ^ { v } ,$ , represented in the coordinate frame of $I ^ { 1 }$ . Each Gaussian is parameterized by center $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , rotation quaternion $\pmb { r } \in \mathbb { R } ^ { 4 }$ , scale $\boldsymbol { s } \in \mathbb { R } ^ { 3 }$ , opacity $\alpha \in \mathbb { R }$ , and spherical harmonics (SH) $\mathbf { \boldsymbol { c } } \in \mathbb { R } ^ { k }$ with k degrees of freedom.

Pose Estimation: We introduce a pose network $f _ { \phi }$ to estimate the relative transformation from each view $\pmb { I } ^ { v }$ to the reference view $I ^ { 1 }$ , which is denoted as $P ^ { v  1 } = [ R ^ { v  1 } | T ^ { v  1 } ]$ where $R ^ { v  1 } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ represents the rotation matrix, and $\pmb { T } ^ { v  1 } \in \mathbb { R } ^ { 3 \times 1 }$ represents the translation vector. This can be formulated as:

$$
\begin{array} { r } { \pmb { P } ^ { v  1 } = f _ { \phi } ( \pmb { I } ^ { v } , \dots , \pmb { I } ^ { 1 } ) , v \in [ 1 , \dots , V ] . } \end{array}\tag{2}
$$

Novel View Synthesis: Novel views are then rendered using the estimated target poses and reconstructed Gaussians:

$$
\hat { \pmb { I } } ^ { t } = \mathcal { R } ( \pmb { P } ^ { t  1 } , \{ \pmb { \mathcal { G } } ^ { v  1 } \} _ { v = 1 } ^ { N } ) , t \in [ N + 1 , \ldots , V ] .\tag{3}
$$

## B. Architecture

Following state-of-the-art large reconstruction models such as MASt3R [23] and VGGT [24], our framework consists of three main components: an encoder, a decoder, and taskspecific prediction heads, as illustrated in Fig. 2. Both the encoder and decoder follow Vision Transformer (ViT) architectures [60]. As shown in Fig. 3, we develop two model variants: SPFSplatV2, which adopts the MASt3R-style architecture, and SPFSplatV2-L, which follows the VGGT design. In the following, we introduce both variants in detail.

Encoder: The RGB image $\pmb { I } ^ { v }$ for each input view v is first patchified and flattened into a sequence of image tokens. These tokens are independently processed by a shared-weight ViT encoder, which extracts view-specific feature representations $\pmb { F } ^ { v } \in \mathbb { R } ^ { L \times C }$ , where L denotes the number of tokens. This is formulated as follows:

$$
\begin{array} { r } { \pmb { F } ^ { v } = \mathrm { E n c o d e r } ( \pmb { I } ^ { v } ) , v \in [ 1 , \ldots , V ] . } \end{array}\tag{4}
$$

Learnable Pose Token: The earlier SPFSplat [22] encodes camera poses by applying global average pooling to decoded image tokens, enforcing uniform feature aggregation and thereby diluting critical geometric cues. In contrast, SPFSplatV2 introduces a learnable pose token $\pmb { g } \in \mathbb { R } ^ { 1 \times C }$ , which is replicated for each view $v \in [ 1 , V ]$ as $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } ^ { v } }$ . Unlike SPFSplatV2 which uses an asymmetrical decoder to distinguish the reference frame from the other views, SPFSplatV2-L introduces two separate learnable pose tokens gÂ¯ and $\bar { \bar { \pmb { g } } } \in \mathbb { R } ^ { 1 \times C }$ , following VGGT. Specifically, gÂ¯ is assigned to the reference frame $( \pmb { g } ^ { 1 } \ : = \ \bar { \pmb { g } } )$ , while all other views share $\bar { \bar { \pmb g } } ( { \pmb g } ^ { v } \ : = \ \bar { \bar { \pmb g } } , { \pmb v } \in$ $[ 2 , \ldots , V ] )$ . The pose tokens are concatenated with the encoder tokens, yielding ${ \bf F } ^ { v } : = [ { \bf g } ^ { v } , { \bf F } ^ { v } ]$ . In the decoding stage, the learnable pose tokens can selectively attend to the most informative features, enabling more accurate pose estimation.

<!-- image-->  
Fig. 3. Architecture comparison of SPFSplatV2 and SPFSplatV2-L. SPF-SplatV2 $\left( \mathbf { a } ~ + ~ \mathbf { b } \right)$ uses asymmetrical decoders and heads to distinguish the reference view $\dot { I } ^ { 1 }$ from other views, whereas SPFSplatV2-L $\left( \mathrm { a } + \mathrm { c } \right)$ employs a unified decoder and head for all views.

Intrinsics Embedding: Following [7], we encode the camera intrinsics of each view into a token $k ^ { v }$ via a linear layer and concatenate it with the pose token and encoder tokens, forming the decoder input $F ^ { v } : = [ k ^ { v } , g ^ { v } , F ^ { v } ]$ . This explicitly injects calibration information, helping to resolve scale ambiguity and improve alignment of predicted poses and 3D Gaussians, particularly under large focal length variations. Importantly, the intrinsic token is optional, and SPFSplatV2 maintains strong performance without it (Sec. IV-D), underscoring the robustness and flexibility of the design.

Masked Multi-view Decoder: To effectively aggregate information across multiple views, we adopt a ViT-based decoder with cross-view attention, enabling joint reasoning over token representations across input views and facilitating cross-view information exchange to capture spatial relationships and the global 3D scene geometry.

During training, since target views are also provided as input, the original SPFSplat [22] adopts a dual-branch design to avoid information leakage from target views into the Gaussian reconstruction of context views. One branch processes only context images for Gaussian prediction, while the other takes both context and target images for pose estimation. However, this design introduces two drawbacks: (i) higher computational cost in cross-attention caused by two forward passes during training, as shown in Fig. 4 (a), and (ii) redundant pose predictions for context views, since each branch produces its own estimates, potentially causing misalignment.

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 4. Comparison of cross-attention in (a) SPFSplat and (b) SPFSplatV2/ V2-L. $I ^ { 1 } , \cdots , I ^ { N }$ denote context images, while $I ^ { N + 1 } , \cdots , I ^ { V }$ denote target images. SPFSplat relies on dual input branches (context-only and context+target) to block target leakage into context reconstruction, leading to higher cross-attention cost. SPFSplatV2 replaces this with masked attention, enforcing the same separation within a single branch while reducing computation by removing redundant contextâtarget interactions.

To address these issues, we introduce a masked attention mechanism as shown in Fig. 3 and Fig. 4 (b). In this approach, context and target images are jointly processed in a single forward pass, with cross-attention selectively masked to control information flow. Specifically, context tokens attend only to context tokens, ensuring Gaussian reconstruction remains independent of target-view information. Meanwhile, target tokens attend to both context and target tokens, allowing the network to leverage global cues for accurate pose estimation.

For SPFSplatV2, we extend MASt3Râs pairwise asymmetric decoder to a multi-view setting, which scales efficiently with the number of views while avoiding excessive memory overhead, following similar implementations in [7], [56]. Each decoder block first performs intra-view self-attention, followed by masked cross-attention. Tokens from the first (reference) view are processed with DecoderBlock1, while tokens from the remaining views use DecoderBlock2. The two decoders share the same architecture but maintain independent weights. Formally, the masked decoder block is defined as:

$$
\begin{array} { r } { G _ { i } ^ { v } = \left\{ \begin{array} { l l } { \mathrm { D e c o d e r B l o c k } _ { i } ^ { 1 } ( G _ { i - 1 } ^ { v } , G _ { i - 1 } ^ { 1 : K } ) , } & { v = 1 , } \\ { \mathrm { D e c o d e r B l o c k } _ { i } ^ { 2 } ( G _ { i - 1 } ^ { v } , G _ { i - 1 } ^ { 1 : K } ) , } & { v \in [ 2 , \ldots , V ] , } \end{array} \right. } \end{array}\tag{5}
$$

for $i = 1 , \ldots , B$ , where B is the number of decoder blocks and $G _ { 0 } ^ { v } = F ^ { v }$ are the initial tokens for view v. Here, $K = N$ for context views $( v \in [ 1 , \ldots , N ] )$ ]), and $K = V$ for target views $( v \in [ N + 1 , \ldots , V ] )$

For SPFSplatV2-L, we adopt the VGGT architecture, which alternates between intra-frame and inter-frame attention. Different from MASt3Râs asymmetric design, the decoder here is unified across all views, which can be expressed as:

$$
\begin{array} { r } { \pmb { G } _ { i } ^ { v } = \mathrm { D e c o d e r B l o c k } _ { i } \left( \pmb { G } _ { i - 1 } ^ { v } , \pmb { G } _ { i - 1 } ^ { 1 : K } \right) , \quad v \in [ 1 , \ldots , V ] , } \end{array}\tag{6}
$$

for $i = 1 , \ldots , B $ , where B is the number of decoder blocks, and K follows the same definition as in SPFSplatV2.

Overall, the masked multi-view decoder achieves three key advantages: (i) it preserves generalization to novel viewpoints by strictly preventing target-specific information from contaminating the Gaussian representation, (ii) it significantly reduces computational overhead by avoiding redundant forward passes, and (iii) it eliminates inconsistent context pose estimates, thereby improving geometric alignment. Together, these improvements lead to more efficient and stable training, as well as more accurate reconstructions.

Gaussian Prediction Heads: Following [7], [8], we employ two DPT-based heads [61] to infer Gaussian parameters. The first head processes decoder tokens of context views and predicts 3D coordinates for each pixel, defining Gaussian centers. The second head estimates rotation, scale, opacity, and SH coefficients for each Gaussian primitive.

For SPFSplatV2, the Gaussian center head extends MASt3Râs pairwise asymmetric pointmap head to a multi-view setting by assigning decoder tokens from the first view to the reference head PointHead1, and tokens from all remaining views to the non-reference head PointHead2. The Gaussian parameter head follows the same structure as the Gaussian center head. As proposed in [5]â[7], we incorporate highresolution skip connections by feeding the original context images into the Gaussian parameter heads, preserving finegrained spatial details. These heads can be formulated as:

$$
\mu ^ { v  1 } = \{ \begin{array} { l l } { \mathrm { P o i n t H e a d } ^ { 1 } ( \{ G _ { i } ^ { v } \} _ { i = 0 } ^ { B } ) , } & { v = 1 , } \\ { \mathrm { P o i n t H e a d } ^ { 2 } ( \{ G _ { i } ^ { v } \} _ { i = 0 } ^ { B } ) , } & { v \in [ 2 , \ldots , N ] , } \end{array} \tag{7}
$$

$$
\overline { { \mathscr { G } } } ^ { v  1 } = \{ \begin{array} { l l } { \mathrm { G S H e a d } ^ { 1 } ( \{ G _ { i } ^ { v } \} _ { i = 0 } ^ { B } , I ^ { v } ) , } & { v = 1 , } \\ { \mathrm { G S H e a d } ^ { 2 } ( \{ G _ { i } ^ { v } \} _ { i = 0 } ^ { B } , I ^ { v } ) , } & { v \in [ 2 , \ldots , N ] , } \end{array} \tag{8}
$$

where $\{ G _ { i } ^ { v } \} _ { i = 0 } ^ { B }$ denotes the set of decoder tokens taken from different blocks, $\mu ^ { v \to 1 }$ denotes Gaussian centers, and $\overline { { { \pmb { \mathscr { G } } } } } ^ { v  1 } =$ $\{ ( \boldsymbol { r } _ { j } ^ { v  1 } , \pmb { c } _ { j } ^ { v  1 } , \alpha _ { j } ^ { v } , \pmb { s } _ { j } ^ { v } ) \}$ represents rotation, scale, opacity, and SH coefficients for each Gaussian primitive.

For SPFSplatV2-L, we adopt the VGGT design for the pointmap head, which serves as both the Gaussian center head and the Gaussian parameter head. Similar to SPFSplatV2, the Gaussian parameter head additionally also incorporates the original context images as an auxiliary input. Unlike the asymmetric design in SPFSplatV2, the Gaussian prediction heads in SPFSplatV2-L are unified across all views:

$$
\pmb { \mu } ^ { v  1 } = \mathrm { P o i n t H e a d } \big ( \{ \pmb { G } _ { i } ^ { v } \} _ { i = 0 } ^ { B } \big ) , \qquad v \in [ 1 , \dots , N ]\tag{9}
$$

$$
\overline { { \pmb { \mathscr { G } } } } ^ { v \to 1 } = \mathrm { G S H e a d } \bigl ( \{ { \pmb { G } } _ { i } ^ { v } \} _ { i = 0 } ^ { B } , { \pmb { I } } ^ { v } \bigr ) , \quad v \in [ 1 , \ldots , N ] .\tag{10}
$$

Pose Head: After decoding, the attended pose tokens $\hat { \mathbf { \chi } } _ { \hat { \mathbf { \chi } } ^ { v } } ^ { v }$ are fed into the pose head and further processed by a 3-layer MLP to predict the camera pose as a 10-dimensional representation [62]. The predicted pose representation is decomposed into translation and rotation for each view. The translation is represented using four homogeneous coordinates [62], while the rotation is encoded in a 6D format, capturing two unnormalized coordinate axes. These axes are normalized and combined via a cross-product operation to construct a full rotation matrix [63]. To compute the relative pose with respect to the reference view, the 10D pose representation is converted into a homogeneous transformation matrix $P ^ { v \to 1 } \in \mathbb { R } ^ { 4 \times 4 }$ Following MASt3R, we make the pose head asymmetrical:

$$
\begin{array} { r } { \pmb { P } ^ { v  1 } = \{ \begin{array} { l l } { \mathrm { P o s e H e a d } ^ { 1 } ( \hat { \pmb { g } ^ { v } } ) , } & { v = 1 , } \\ { \mathrm { P o s e H e a d } ^ { 2 } ( \hat { \pmb { g } ^ { v } } ) , } & { v \in [ 2 , . . . , V ] , } \end{array}  } \end{array}\tag{11}
$$

where $P ^ { v  1 }$ is the estimated relative pose from $I ^ { v }$ to $I ^ { 1 }$

For SPFSplatV2-L, the pose head follows the original VGGT design: the refined pose tokens $\hat { \pmb { g } } ^ { v }$ are subsequently processed by four additional self-attention layers and a linear projection to predict the camera parameters.

$$
\begin{array} { r } { { P } ^ { v  1 } = \mathrm { P o s e H e a d } ( \hat { g ^ { v } } ) , v \in [ 1 , . . . , V ] . } \end{array}\tag{12}
$$

We normalize the camera poses by assigning the first input view the canonical pose [U|0], where U represents the identity matrix, and 0 denotes the zero translation vector.

## C. Loss Functions

Image Rendering Loss: Our model is trained using groundtruth target RGB images as supervision. The training loss is formulated as a weighted combination of the $L _ { 2 }$ loss and the LPIPS loss [64], formulated as:

$$
\mathcal { L } _ { \mathrm { r e n d e r } } = \Vert \pmb { I } ^ { t } - \hat { \pmb { I } } ^ { t } \Vert _ { 2 } + \gamma \mathrm { L P I P S } ( \pmb { I } ^ { t } , \hat { \pmb { I } } ^ { t } ) ,\tag{13}
$$

where $I ^ { t }$ and $\hat { \boldsymbol { I } } ^ { t }$ denote the ground-truth and rendered target images for $t \in [ N + 1 , V ]$ , and $\gamma$ is a weighting factor that balances pixel-level accuracy and perceptual similarity.

Reprojection Loss: Existing approaches enforce pixelaligned Gaussian prediction by constraining Gaussian locations along the input viewing rays [5], [6], [9], [11], [20], [21]. Meanwhile, canonical-space-based methods [7], [8] rely on ground-truth camera poses to guide the canonical 3D points (Gaussian centers). Both strategies ensure alignment between each pixel and its corresponding 3D point. However, since our model learns 3D Gaussian centers in a canonical space without known camera poses, the network lacks explicit geometric constraints to enforce pixel-aligned Gaussian representation.

A naive solution is to include context views in the image rendering loss (Eq. 13) by synthesizing images from them and computing the loss against their ground-truth counterparts. However, this leads to unstable training due to overfitting. Specifically, the network prioritizes improving the rendering quality of the first context view, as the 3D Gaussian space is defined in its camera coordinate, making its rendering independent of the learnable poses. Since the Gaussians from this view already captures sufficient scene information, the model suppresses the contribution of other context views by shifting their Gaussian centers away and adjusting camera poses, ultimately causing training collapse.

To address this issue, we instead employ a pixel-wise reprojection loss to jointly optimize 3D points and camera poses [12], [65]. Unlike purely image-based supervision, this reprojection loss enforces explicit geometric constraints, thereby reducing overfitting to the context views. Concretely, for each pixel $\mathbf { p } _ { j } ^ { v }$ in view $\begin{array} { r l r } { v } & { { } \in } & { [ 1 , N ] } \end{array}$ , we project the corresponding 3D Gaussian center $\mu _ { j } ^ { v \to 1 }$ from the canonical coordinate frame into the 2D pixel space using the estimated pose of view v, and minimize the reprojection error:

$$
\mathcal { L } _ { \mathrm { r e p r o j } } = \sum _ { v = 1 } ^ { N } \sum _ { j = 1 } ^ { H \times W } \left| \mathbf { p } _ { j } ^ { v } - \pi ( K ^ { v } , P ^ { v \to 1 } , \pmb { \mu } _ { j } ^ { v \to 1 } ) \right| ,\tag{14}
$$

where $\pi$ denotes the camera projection function, $\pmb { K } ^ { v }$ the camera intrinsics of view v, and $P ^ { v  1 }$ the relative pose from view v to the canonical frame.

Different from SPFSplat, which applies reprojection loss to both context-only and context-with-target branches, SPF-SplatV2 predicts a single set of context poses, avoiding redundant supervision and potential pose misalignment. This streamlined design leverages reprojection loss to enable more stable training and efficient optimization of pixel-aligned 3D Gaussians, without requiring ground-truth camera poses.

## D. Multi-View Dropout

Unlike SPFSplat, which trains separate models for different numbers of context views, we use a single unified model. To improve generalization, we introduce a multi-view dropout strategy: for more than two context views, the leftmost and rightmost views are retained while intermediate views are randomly dropped during training. This encourages the network to handle flexible input configurations, improves robustness to varying numbers and spatial distributions of views at test time, and implicitly regularizes feature aggregation, leading to more stable training and higher-quality reconstructions.

## IV. EXPERIMENTS

We report evaluation results on novel view synthesis quality, cross-dataset generalization, and relative pose estimation across several datasets. In addition, we conduct comprehensive ablation studies to analyze the effectiveness of our method.

## A. Experimental Settings

Dataset: We train and evaluate our method on RealEstate10K (RE10K) [66], which contains large-scale real estate videos from YouTube, and ACID [67], a dataset of nature scenes captured by aerial drones. Camera poses for both datasets are obtained via SfM, and we follow the official train-test splits used in prior work [5]â[7]. Following [7], evaluations on RE10K and ACID are conducted under varying camera overlaps, where input pairs are grouped by overlap ratios: small (0.05%â0.3%), medium (0.3%â0.55%), and large (0.55%â0.8%), determined using a pretrained dense matcher [68]. To analyze the impact of training scale, we also use DL3DV [69], an outdoor dataset with 10K videos and diverse camera motions beyond RE10K. For cross-dataset generalization, we evaluate on ACID, the object-centric DTU dataset [70], DL3DV, and ScanNet++ [71], which contains indoor scenes with camera trajectories distinct from RE10K.

Baselines: For novel view synthesis, we compare to three groups of baselines: pose-required methods (pixelSplat [5], MVSplat [6]), supervised pose-free methods (CoPoNeRF [17], Splatt3R [8], NoPoSplat [7]), and self-supervised pose-free methods (PF3plat [20], SelfSplat [21], SPFSplat [22]). For camera pose estimation, we compare against SfM-based methods (SuperPoint [48] + SuperGlue [52], DUSt3R [29], MASt3R [23], VGGT [24]) and splatting-based methods (No-PoSplat, SelfSplat, PF3plat, SPFSplat).

Evaluation Protocol: For novel view synthesis, we adopt standard metrics: pixel-level PSNR, patch-level SSIM [72], and feature-level LPIPS [64]. For pose estimation, following prior works [7], [52], we report the area under the cumulative pose error curve (AUC) at thresholds of $5 ^ { \circ } , ~ 1 0 ^ { \circ }$ , and 20â¦, where the pose error is defined as the maximum of the angular errors in rotation and translation.

TABLE I  
PERFORMANCE COMPARISON OF NOVEL VIEW SYNTHESIS ON RE10K [66] WITH DIFFERENT IMAGE OVERLAP. OUR METHOD OUTPERFORMS STATE-OF-THE-ART APPROACHES. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED, AND â DENOTES EVALUATION WITH POSE ALIGNMENT.
<table><tr><td rowspan="2">Method</td><td colspan="3">Small</td><td colspan="3">Medium</td><td colspan="3">Large</td><td colspan="3">Average</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td colspan="10">Pose-Required</td><td colspan="3"></td></tr><tr><td>pixelSplat</td><td>20.277</td><td>0.719</td><td>0.265</td><td>23.726</td><td>0.811</td><td>0.180</td><td>27.152</td><td>0.880</td><td>0.121</td><td>23.859</td><td>0.808</td><td>0.184</td></tr><tr><td>MVSplat</td><td>20.371</td><td>0.725</td><td>0.250</td><td>23.808</td><td>0.814</td><td>0.172</td><td>27.466</td><td>0.885</td><td>0.115</td><td>24.012</td><td>0.812</td><td>0.175</td></tr><tr><td colspan="10">Supervised Pose-Free</td><td colspan="3"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CoPoNeRF Spplatt3R</td><td>17.393 17.789</td><td>0.585 0.582</td><td>0.462 0.375</td><td>18.813</td><td>0.616</td><td>0.392</td><td>20.464</td><td>0.652</td><td>0.318</td><td>18.938</td><td>0.619</td><td>0.388 0.596</td></tr><tr><td>NoPoSplat*</td><td>22.514</td><td>0.784</td><td>0.210</td><td>18.828 24.899</td><td>0.607 0.839</td><td>0.330 0.160</td><td>19.243 27.411</td><td>0.593 0.883</td><td>0.317 0.119</td><td>18.688 25.033</td><td>0.337 0.838</td><td>0.160</td></tr><tr><td colspan="10">Self-Supervised Pose-Free</td><td colspan="3"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SelfSplat</td><td>14.828</td><td>0.543</td><td>0.469</td><td>18.857</td><td>0.679</td><td>0.328</td><td>23.338</td><td>0.798</td><td>0.208</td><td>19.152</td><td>0.680</td><td>0.328</td></tr><tr><td>PFplat</td><td>18.358</td><td>0.668</td><td>0.298</td><td>20.953</td><td>0.741</td><td>0.231</td><td>23.491</td><td>0.795</td><td>0.179</td><td>21.042</td><td>0.739</td><td>0.233</td></tr><tr><td>SPFSplat</td><td>22.897</td><td>0.792</td><td>0.201</td><td>25.334</td><td>0.847</td><td>0.153</td><td>27.947</td><td>0.894</td><td>0.110</td><td>25.484</td><td>0.847</td><td>0.153</td></tr><tr><td>SPFSplat*</td><td>23.178</td><td>0.796</td><td>0.200</td><td>25.695</td><td>0.853</td><td>0.151</td><td>28.377</td><td>0.899</td><td>0.111</td><td>25.845</td><td>0.852</td><td>0.152</td></tr><tr><td>SPFSplatV2</td><td>23.123 23.456</td><td>0.800</td><td>0.195 0.193</td><td>25.542</td><td>0.853</td><td>0.149</td><td>28.143 28.682</td><td>0.897 0.905</td><td>0.110</td><td>25.693 26.157</td><td>0.853</td><td>0.149 0.146</td></tr><tr><td>SPFSplatV2* SPSpatvV2-L</td><td>23.138</td><td>0.806 0.804</td><td>0.184</td><td>26.030 25.518</td><td>0.862 0.856</td><td>0.145 0.136</td><td>28.081</td><td>0.899</td><td>0.107 0.099</td><td>2668</td><td>0.861 0.855</td><td>0..137</td></tr><tr><td>SPFSplatV2-L*</td><td>23.329</td><td>0.804</td><td>0.183</td><td>25.863</td><td>0.861</td><td>00.134</td><td>28.456</td><td>0.903</td><td>098</td><td>25.983</td><td>0.859</td><td>0.136</td></tr></table>

TABLE II

PERFORMANCE COMPARISON OF NOVEL VIEW SYNTHESIS ON ACID [67]. THE BEST AND SECOND BEST RESULTS ARE HIGHLIGHTED.
<table><tr><td rowspan="2">Method</td><td colspan="3">Small</td><td colspan="3">Medium</td><td colspan="3">Large</td><td colspan="3">Average</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td colspan="10">Pose-Required</td><td colspan="3"></td></tr><tr><td>pixelSplat</td><td>22.088</td><td>0.655</td><td>0.284</td><td>25.525</td><td>0.777</td><td>0.197</td><td>28.527</td><td>0.854</td><td>0.139</td><td>25.889</td><td>0.780</td><td>0.194</td></tr><tr><td>MVSplat</td><td>21.412</td><td>0.640</td><td>0.290</td><td>25.150</td><td>0.772</td><td>0.198</td><td>28.457</td><td>0.854</td><td>0.137</td><td>25.561</td><td>0.775</td><td>0.195</td></tr><tr><td colspan="10">Supervised Pose-Free</td><td colspan="3"></td></tr><tr><td>oOPONeRF</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Splatt3R</td><td>18.651 17.419</td><td>0.551 0.501</td><td>0.485 0.434</td><td>20.654 18.257</td><td>0.595 0.514</td><td>0.418 0.405</td><td>22.654</td><td>0.652</td><td>0.343</td><td>20.950</td><td>0.606</td><td>0.406 0.407</td></tr><tr><td>NoOPOSplat*</td><td>23.087</td><td>0.685</td><td>0.258</td><td>25.624</td><td>0.77</td><td>0.193</td><td>18.134 28.043</td><td>0.508 0.841</td><td>0.395 0.144</td><td>18.060 25.1</td><td>0.510 0.781</td><td>0.189</td></tr><tr><td colspan="10">Self-Supervised Pose-Free</td><td colspan="3"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SelfSplat</td><td>18.301</td><td>0.568</td><td>0.408</td><td>21.375</td><td>0.676</td><td>0.314</td><td>25.219</td><td>0.792</td><td>0.214</td><td>22.089</td><td>0.694</td><td>0.298 0.293</td></tr><tr><td>PF3plat</td><td>18.112</td><td>0.537</td><td>0.376</td><td>20.732</td><td>0.615</td><td>0.307</td><td>23.607</td><td>0.710</td><td>0.228</td><td>21.206</td><td>0.632</td><td>0.186</td></tr><tr><td>SPFSplat</td><td>22.667</td><td>0.665</td><td>0.262</td><td>25.620</td><td>0.773</td><td>0.192</td><td>28.607</td><td>0.856</td><td>0.136</td><td>26.070</td><td>0.781</td><td>0.176</td></tr><tr><td>SPPFSplat*</td><td>23.676 22.44</td><td>0.708 0.679</td><td>0.243</td><td>26.351</td><td>0.801</td><td>0.182</td><td>29.170 28.766</td><td>0.870 00.862</td><td>0.131 0.133</td><td>26.796 26.284</td><td>0.807</td><td>0.182</td></tr><tr><td>SPFSpplatV2 SPFSpatV2*</td><td>23.635</td><td>0.700</td><td>0.255 0.247</td><td>25.849 26..356</td><td>0.784 0.798</td><td>0.187 0.182</td><td>29.223</td><td>0.871</td><td>0.129</td><td>26.809</td><td>0.791 0.804</td><td>0.176</td></tr><tr><td>SPFSpatv2-L</td><td>23.640</td><td>0.706</td><td>0.225</td><td>26.272</td><td>0.801</td><td>0.166</td><td>28.938</td><td>0.868</td><td>0.120</td><td>26.674</td><td>0.806</td><td>0.162</td></tr><tr><td>SPFSplatV2-L*</td><td>23.937</td><td>0.710</td><td>0.224</td><td>26.489</td><td>00.803</td><td>0.165</td><td>29.188</td><td>0.871</td><td>0.118</td><td>26.917</td><td>0.809</td><td>00.160</td></tr></table>

During evaluation of novel view synthesis, target images are typically rendered with ground-truth poses [5], [6], [8], [17]. An alternative is to render using estimated target poses, as in PF3plat [20] and SelfSplat [21]. NoPoSplat [7] instead adopts an evaluation-time pose alignment (EPA) strategy, which optimizes the target pose during evaluation while keeping the reconstructed Gaussians fixed, so that the rendered image best matches the ground truth. This alignment decouples rendering quality from pose estimation accuracy, enabling direct assessment of Gaussian reconstruction. In contrast, rendering with estimated poses jointly evaluates reconstruction fidelity and the consistency between estimated poses and the learned Gaussians. Unless otherwise noted, we render with estimated poses for comprehensive evaluation and additionally report results with pose alignment for fair comparison to NoPoSplat.

## B. Implementation Details.

Our method is implemented in PyTorch and leverages a CUDA-based 3DGS renderer with gradient support for camera poses. All models are trained on a single NVIDIA A100 GPU. Each training sample corresponds to a scene with context and target views, with the frame distance between context views gradually increased during training. The initial learning rate is set to $1 \times 1 0 ^ { - 5 }$ for the backbone and $1 \times 1 0 ^ { - 4 }$ for all other parameters, and LPIPS and reprojection losses are weighted 0.05 and 0.001, respectively. For SPFSplatV2, the encoder adopts a ViT-Large architecture with a patch size of 16, while the decoder is based on ViT-Base. The encoder, decoder, and Gaussian center head are initialized from pretrained MASt3R [23], while the pose head is initialized to approximate the identity rotation matrix for stable convergence. For SPFSplatV2-L, the encoder is a ViT-Large from DINOv2 [73] with a patch size of 14. The encoder, decoder, pose head, and Gaussian center head are initialized from pretrained VGGT [24] weights. All remaining layers are randomly initialized. Training is performed at a resolution of 256 Ã 256 and 224 Ã 224 for V2 and V2-L, respectively.

<!-- image-->  
Fig. 5. Qualitative comparison on RE10K (top three rows) and ACID (bottom three rows). Our method 1) better handles extreme viewpoint changes and minimal input overlap (e.g., Row 1 and Row 2), 2) preserves finer details (e.g., Row 3) and more accurate geometric structure (e.g., Row 4 and Row 5), and 3) reduces misaligned blending artifacts and ghosting effect (e.g. Row 5 and Row 6).

## C. Results

Novel View Synthesis: Quantitative results on RE10K and ACID are reported in Tab. I and Tab. II. We make the following observations: 1) Despite being trained without ground-truth poses, SPFSplatV2 and SPFSplatV2-L consistently outperform state-of-the-art methods. Notably, both variants outperform our earlier SPFSplat baseline in most cases across different image overlap settings, with or without pose alignment, underscoring the effectiveness of the improved architecture. 2) While evaluation pose alignment generally improves performance, SPFSplatV2 without alignment still outperforms NoPoSplat with alignment, indicating that the jointly optimized poses are well aligned with the reconstructed Gaussians. 3) Between our variants, SPFSplatV2 achieves slightly higher PSNR on RE10K, while SPFSplatV2-L attains better LPIPS scores. On ACID, SPFSplatV2-L delivers the strongest overall results, likely benefiting from VGGTâs superior multi-view reconstruction capabilities and feature representations.

Qualitative comparisons in Fig. 5 further demonstrate that our models reduce misalignment and recover more accurate geometry than baselines, even in challenging scenarios such as minimal input overlap or extreme viewpoint changes. Specifically, SPFSplatV2 improves structural accuracy and visual clarity compared to the original SPFSplat, while SPFSplatV2- L produces the highest overall rendering quality, capturing fine geometric details and textures more faithfully.

Cross-Dataset Generalization: To evaluate zero-shot generalization, we train on RE10K (indoor scenes) and test on ACID (outdoor), DTU (object-centric), DL3DV (outdoor), and ScanNet++ (indoor). As shown in Tab. III, both SPFSplatV2 variants generalize robustly across these diverse domains, consistently outperforming prior approaches. These datasets exhibit substantially different camera motions and scene types compared to RE10K, highlighting the strong out-of-domain generalization capability of our models, even under minimal image overlaps. Notably, in the RE10KâACID setting, SPFSplatV2 surpasses NoPoSplat and SPFSplat trained directly on ACID (Tab. II) when evaluated with pose alignment. SPFSplatV2-L consistently outperforms SPFSplatV2 both with and without pose alignment.

Qualitative results in Fig. 6 show that both variants produce sharper and more geometrically accurate reconstructions than prior methods, with SPFSplatV2-L achieving the highest visual quality. These results demonstrate that, even without ground-truth poses, our framework effectively aligns

TABLE III  
PERFORMANCE COMPARISON OF CROSS-DATASET GENERALIZATION. ALL METHODS ARE TRAINED ON RE10K AND EVALUATED IN A ZERO-SHOT SETTING ON ACID, DTU, DL3DV AND SCANNET++. OUR METHOD DEMONSTRATES SUPERIOR GENERALIZATION COMPARED TO STATE-OF-THE-ART APPROACHES. â DENOTES EVALUATION WITH POSE ALIGNMENT.
<table><tr><td rowspan="2">Method</td><td colspan="3">ACID</td><td colspan="3">DTU</td><td colspan="3">DL3DV</td><td colspan="3">ScanNet++</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td colspan="10">Pose-Required</td><td colspan="3"></td></tr><tr><td>pixelSplat</td><td>25.477</td><td>0.770</td><td>0.207</td><td>15.067</td><td>0.539</td><td>0.341</td><td>18.688</td><td>0.582</td><td>0.354</td><td>18.422</td><td>0.720</td><td>0.278</td></tr><tr><td>MVSplat</td><td>25.525</td><td>0.773</td><td>0.199</td><td>14.542</td><td>0.537</td><td>0.324</td><td>17.786</td><td>0.545</td><td>0.357</td><td>17.138</td><td>0.687</td><td>0.297</td></tr><tr><td colspan="10">Supervised Pose-Free</td><td colspan="3"></td></tr><tr><td>NoPoSplat*</td><td>25.764</td><td>0.776</td><td>0.199</td><td>17.899</td><td>0.629</td><td>0.279</td><td>19.974</td><td>0.612</td><td>0.305</td><td>22.136</td><td>0.798</td><td>0.232</td></tr><tr><td colspan="10">Self-Supervised Pose-Free</td><td colspan="3"></td></tr><tr><td></td><td>22.204</td><td>0.686</td><td>0.316</td><td>13.249</td><td>0.434</td><td>0.441</td><td>15.047</td><td>0.410</td><td>0.498</td><td>13.277</td><td>0.538</td><td>0.534</td></tr><tr><td colspan="10">SelfSplat</td><td colspan="3"></td></tr><tr><td>PF3plat</td><td>20.726</td><td>0.610</td><td>0.308</td><td>12.972</td><td>0.407</td><td>0.464</td><td>15.773</td><td>0.458</td><td>0.417</td><td>16.471</td><td>0.688</td><td>0.303</td></tr><tr><td>SPFSplat</td><td>25.965</td><td>0.781</td><td>0.190</td><td>16.550</td><td>0.579</td><td>0.270</td><td>19.172</td><td>0.573</td><td>0.315</td><td>19.971</td><td>0.738</td><td>0.265</td></tr><tr><td>SPFSplat*</td><td>26.697</td><td>0.806</td><td>0.181</td><td>18.297</td><td>0.660</td><td>0.255</td><td>19.494</td><td>0.574</td><td>0.319</td><td>22.312</td><td>0.793</td><td>0.243</td></tr><tr><td>SPFSplatV2</td><td>26.220</td><td>0.789</td><td>0.185</td><td>16.793</td><td>0.584</td><td>0.265</td><td>19.439</td><td>0.584</td><td>0.304</td><td>20.919</td><td>0.771</td><td>0.243</td></tr><tr><td>SPFSplatV2*</td><td>26.802</td><td>0.805</td><td>0.179</td><td>18.506</td><td>0.663</td><td>0.246</td><td>19.978</td><td>0.607</td><td>0.302</td><td>22.776</td><td>0.812</td><td>0.227</td></tr><tr><td>SPFSplatV2-L</td><td>26.361</td><td>0.796</td><td>0.169</td><td>17.739</td><td>0.653</td><td>0.228</td><td>19.743</td><td>0.613</td><td>0.277</td><td>21.796</td><td>0.811</td><td>0.200</td></tr><tr><td>SPFSplatV2-L*</td><td>26.680</td><td>0.802</td><td>0.166</td><td>19.316</td><td>0.671</td><td>0.229</td><td>20.108</td><td>0.615</td><td>0.279</td><td>23.072</td><td>0.820</td><td>0.199</td></tr></table>

Ref.  
pixelSplat  
MVSplat  
NoPoSplat

PF3plat  
SelfSplat  
SPFSplat  
GT  
SPFSplatV2 SPFSplatV2-L  
<!-- image-->  
Fig. 6. Qualitative comparison on cross-dataset generalization. All methods are trained on RE10K and evaluated on ACID and DTU, DL3DV, and ScanNet++ (from top to bottom). Both SPFSplatV2 and SPFSplatV2-L yield more geometrically accurate reconstructions than prior methods.

3D Gaussians with predicted camera poses, enabling robust generalization to out-of-distribution scenes.

Relative Pose Estimation. We evaluate relative pose estimation between input image pairs on RE10K, ACID, DL3DV, and ScanNet++, with results in Tab. IV. All splat-based methods are trained on RE10K to evaluate generalization. Since VGGT does not natively support 224 Ã 224 inputs, we resize and center-crop its input images to 224 Ã 224 and pad the width to 518, as specified in [24]. SPFSplatV2-L uses 224Ã224 inputs, while all other methods operate at 256Ã256. SuperPoint + SuperGlue derives relative poses from Essential Matrices estimated from feature correspondences. DUSt3R, MASt3R, and NoPoSplat use PnP [40] with RANSAC [45], while PF3plat and VGGT directly predict poses. Our SPFSplat variants support two strategies: (i) direct regression through the pose head, and (ii) PnP with RANSAC applied to predicted 3D Gaussian centers.

As shown in Tab. IV, both regression and PnP yield similarly strong performance, reflecting consistent alignment between estimated poses and reconstructed 3D points. Despite no geometry priors during training, SPFSplatV2 substantially outperforms MASt3R, its initialization model, and SPFSplatV2- L improves over VGGT in most cases. This demonstrates our frameworkâs ability to jointly optimize camera poses and 3D structure using only image-level supervision. Both SPFSplatV2 variants also significantly surpass the original SPFSplat, primarily due to masked attention improving pose alignment. On RE10K and ACID, our models achieve stateof-the-art results. On DL3DV and ScanNet++, which exhibit challenging camera motions, NoPoSplat achieves better performance, benefiting from ground-truth pose supervision during training. As shown in Sec. IV-D, this gap can be eliminated by scaling our approach to larger training datasets.

TABLE IV  
PERFORMANCE COMPARISON OF POSE ESTIMATION IN AUC WITH VARIOUS THRESHOLDS ON RE10K, ACID, DL3DV AND SCANNET++ DATASETS.
<table><tr><td rowspan="2">Method</td><td colspan="3">RE10K</td><td colspan="3">ACID</td><td colspan="3">DL3DV</td><td colspan="3">ScanNet++</td></tr><tr><td>5oâ</td><td>10Â° â</td><td>20Â° â</td><td>5Â° â$</td><td>10Â° â</td><td>20Â° â</td><td>5oâ$</td><td>10Â° â</td><td>20Â° â</td><td>5o â$</td><td>10Â° â</td><td>20Â° â</td></tr><tr><td>SfM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SP + SG</td><td>0.234</td><td>0.406</td><td>0.569</td><td>0.228</td><td>0.363</td><td>0.500</td><td>0.224</td><td>0.372</td><td>0.492</td><td>0.087</td><td>0.151</td><td>0.248</td></tr><tr><td>DUSt3R</td><td>0.336</td><td>0.541</td><td>0.702</td><td>0.118</td><td>0.279</td><td>0.470</td><td>0.275</td><td>0.490</td><td>0.686</td><td>0.109</td><td>0.284</td><td>0.500</td></tr><tr><td>MASt3R</td><td>0.281</td><td>0.494</td><td>0.672</td><td>0.138</td><td>0.312</td><td>0.507</td><td>0.332</td><td>0.593</td><td>0.772</td><td>0.139</td><td>0.336</td><td>0.549</td></tr><tr><td>VGGT</td><td>0.257</td><td>0.474</td><td>0.658</td><td>0.142</td><td>0.304</td><td>0.486</td><td>0.356</td><td>0.609</td><td>0.784</td><td>0.156</td><td>0.311</td><td>0.514</td></tr><tr><td>Pose-Free View Synthesis</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NoPoSplat</td><td>0.571</td><td>0.727</td><td>0.833</td><td>0.335</td><td>0.496</td><td>0.644</td><td>0.470</td><td>0.646</td><td>0.762</td><td>0.207</td><td>0.403</td><td>0.641</td></tr><tr><td>PF3plat</td><td>0.187</td><td>0.398</td><td>0.613</td><td>0.060</td><td>0.165</td><td>0.340</td><td>0.118</td><td>0.281</td><td>0.479</td><td>0.058</td><td>0.204</td><td>0.415</td></tr><tr><td>SPFSplat</td><td>0.617</td><td>0.755</td><td>0.845</td><td>0.364</td><td>0.520</td><td>0.662</td><td>0.283</td><td>0.461</td><td>0.622</td><td>0.098</td><td>0.188</td><td>0.374</td></tr><tr><td>SPFSplat (PnP)</td><td>0.613</td><td>0.754</td><td>0.845</td><td>0.355</td><td>0.516</td><td>0.658</td><td>0.279</td><td>0.464</td><td>0.626</td><td>0.120</td><td>0.226</td><td>0.408</td></tr><tr><td>SPFSplatV2</td><td>0.638</td><td>0.776</td><td>0.863</td><td>0.387</td><td>0.541</td><td>0.672</td><td>0.369</td><td>0.534</td><td>0.694</td><td>0.144</td><td>0.281</td><td>0.487</td></tr><tr><td>SPFSplatV2 (PnP)</td><td>0.641</td><td>0.777</td><td>0.864</td><td>0.374</td><td>0.533</td><td>0.667</td><td>0.375</td><td>0.542</td><td>0.700</td><td>0.111</td><td>0.250</td><td>0.463</td></tr><tr><td>SPFSplatV2-L</td><td>0.645</td><td>0.780</td><td>0.864</td><td>0.379</td><td>0.539</td><td>0.671</td><td>0.420</td><td>0.582</td><td>0.711</td><td>0.184</td><td>0.400</td><td>0.630</td></tr><tr><td>SPFSplatV2-L (PnP)</td><td>0.657</td><td>0.786</td><td>0.867</td><td>0.375</td><td>00.535</td><td>0.668</td><td>0.429</td><td>0.587</td><td>0.716</td><td>0.183</td><td>0.400</td><td>0.627</td></tr></table>

<!-- image-->  
Fig. 7. Comparison of 3D Gaussians and rendered results. Red and green denote context and target camera poses, respectively. Rendered images and depth maps at the target views are shown on the right. Our method produces higher-quality 3D Gaussians and better rendering over baselines.

Geometry Reconstruction: As illustrated in Fig. 7, SPFSplatV2 and SPFSplatV2-L produce substantially higher-quality 3D Gaussian primitives than prior methods, even under large viewpoint changes between input pairs. Previous approaches often exhibit distorted structures or ghosting artifacts, whereas our models, trained without ground-truth poses, reconstruct more accurate 3D geometry and yield sharper renderings, reflecting improved Gaussian alignment across views. This improvement arises from the joint optimization of Gaussians and poses, which strengthens geometric consistency. Compared to SPFSplat, SPFSplatV2 achieves more precise structural reconstruction, particularly visible in the left windows, while SPFSplatV2-L further enhances overall Gaussian quality.

Extension to Multiple Views: Our method naturally extends to multiple input views. As shown in Tab. V, novel view synthesis performance consistently improves with more context views. Both SPFSplatV2 and SPFSplatV2-L outperform the NoPoSplat and original SPFSplat, benefiting from enhanced architectures and the multi-view dropout strategy. With denser inputs, SPFSplatV2-L shows a clearer advantage, as its VGGT backbone, pretrained on multi-view data, provides stronger representations than MASt3R, which is limited to pairwise training. These results show that SPFSplatV2-L can better leverage additional views to enhance geometric consistency and reconstruction fidelity. Overall, the consistent gains with increasing context views highlight the flexibility and scalability of our framework for multi-view scenarios.

Efficiency: We compare the efficiency of our method to other approaches in Tab. VI and Tab. VII.

1) Inference Efficiency: Tab. VI reports parameter size, FLOPs, and runtime during inference, measured for reconstructing 3D Gaussians from two input images on an A6000 GPU. The input resolution is 256 Ã 256, except for SPFSplatV2-L which uses 224 Ã 224. SPFSplatV2 achieves comparable model size, FLOPs, and runtime to NoPoSplat and SPFSplat, while providing substantial speedups of approximately 3.5Ã, 1.4Ã, 2.3Ã, and 27Ã over pixelSplat, MVSplat, SelfSplat, and PF3plat, respectively. These gains stem from architectural differences: pixelSplat relies on a time-consuming epipolar transformer; MVSplat requires costly cost-volume construction; and both SelfSplat and PF3plat depend on separate pose-estimation modules to lift predicted depth into Gaussians, with PF3plat further incurring heavy local feature matching costs. In contrast, SPFSplatV2 reconstructs Gaussians directly in a canonical space using a feed-forward network, avoiding explicit geometric operations such as cost-volume construction. Compared to SPFSplatV2, SPFSplatV2-L introduces additional computational overhead, as a trade-off for superior reconstruction quality.

TABLE V  
NOVEL VIEW SYNTHESIS WITH VARYING INPUT VIEW NUMBERS. FOR NOPOSPLAT, ONLY RESULTS REPORTED IN [7] ARE SHOWN.
<table><tr><td rowspan="2">Method</td><td colspan="3">2 Views</td><td colspan="3">3 Views</td><td colspan="3">5 Views</td><td colspan="3">10 Views</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>NoPoSplat*</td><td>25.033</td><td>0.838</td><td>0.160</td><td>26.619</td><td>0.872</td><td>0.127</td><td>â</td><td>â</td><td>â</td><td></td><td>â</td><td>â</td></tr><tr><td>SPFSplat</td><td>25.484</td><td>0.847</td><td>0.153</td><td>26.724</td><td>0.871</td><td>0.128</td><td>26.891</td><td>0.875</td><td>0.122</td><td>27.159</td><td>0.880</td><td>0.115</td></tr><tr><td>SPPSplatV2</td><td>25.693</td><td>0.853</td><td>0.149</td><td>27.262</td><td>0.884</td><td>0.120</td><td>27.585</td><td>0.890</td><td>0.115</td><td>28.188</td><td>0.901</td><td>0.106</td></tr><tr><td>SPFSplatV2-L</td><td>25.668</td><td>0.855</td><td>0.137</td><td>27.685</td><td>0.898</td><td>0.101</td><td>28.141</td><td>0.907</td><td>0.094</td><td>28.973</td><td>0.922</td><td>0.083</td></tr></table>

TABLE VI

COMPARISON OF INFERENCE EFFICIENCY ON AN NVIDIA A6000 GPU.
<table><tr><td>Methods</td><td>Params (B)</td><td>Inference FLOPs (T)</td><td>Inference Time (s)</td></tr><tr><td>pixelSplat</td><td>0.119</td><td>0.764</td><td>0.152</td></tr><tr><td>MVSplat</td><td>0.012</td><td>0.170</td><td>0.059</td></tr><tr><td>NoPoSplat</td><td>0.612</td><td>0.405</td><td>0.042</td></tr><tr><td>SelfSplat</td><td>0.081</td><td>0.491</td><td>0.101</td></tr><tr><td>PF3plat</td><td>0.394</td><td>2.164</td><td>1.171</td></tr><tr><td>SPFSplat</td><td>0.616</td><td>0.405</td><td>0.044</td></tr><tr><td>SPFSplatV2</td><td>0.613</td><td>0.405</td><td>0.043</td></tr><tr><td>SPFSplatV2-L</td><td>1.223</td><td>0.610</td><td>0.075</td></tr></table>

2) Training Efficiency: For a fair comparison, we evaluate the training efficiency of SPFSplat variants only against PF3plat and SelfSplat, as self-supervised pose-free methods require both context and target images during training, whereas other methods use only context images. The results are summarized in Tab. VII. For all methods, two images are used as context views and one as the target view. Training time and GPU memory usage are measured on an NVIDIA A100 and averaged per sample. PF3plat requires significantly larger FLOPs, time, and GPU memory during training. Thanks to masked attention, SPFSplatV2 reduces training FLOPs of SPFSplat by 12%, resulting in a 25% speedup and a 13% reduction in memory consumption. In contrast, SPFSplatV2-L incurs higher computational and memory costs, with a 46% increase in FLOPs, 30% longer training time, and 25% higher memory usage compared to SPFSplat, reflecting the trade-off for its superior reconstruction performance.

## D. Ablation Analysis

Scaling to Larger Training Data: Since our approach does not rely on ground-truth poses, it scales efficiently to larger datasets with minimal annotation. To evaluate the effect of training data size, we train SPFSplatV2 and SPFSplatV2-L on a combination of RE10K and DL3DV, and assess pose estimation on RE10K and DL3DV (in-domain), as well as ACID and ScanNet++ (out-of-domain). As shown in Tab. VIII, enlarging the training set consistently improves both direct regression and PnP-based pose estimation, driven by the greater diversity of camera trajectories and scene appearances introduced by DL3DV. Among the models, SPFSplatV2-L achieves the strongest results across most benchmarks. Compared to Tab. IV, incorporating DL3DV during training enables both variants to surpass all prior methods, including NoPoSplat, across different benchmarks, highlighting the scalability and effectiveness of our framework without relying on any groundtruth pose supervision.

TABLE VII  
COMPARISON OF TRAINING EFFICIENCY ON AN NVIDIA A100 GPU.
<table><tr><td>Methods</td><td>Training FLOPs (T)</td><td>Training Time (s)</td><td>Mem. (GB)</td></tr><tr><td>SelfSplat</td><td>0.491</td><td>0.122</td><td>6.602</td></tr><tr><td>PF3plat</td><td>2.164</td><td>0.633</td><td>15.043</td></tr><tr><td>SPFSplat</td><td>0.582</td><td>0.110</td><td>5.634</td></tr><tr><td>SPFSplatV2</td><td>0.515</td><td>0.082</td><td>4.891</td></tr><tr><td>SPFSplatV2-L</td><td>0.849</td><td>0.143</td><td>7.044</td></tr></table>

Ablation on Different Components: We conduct an ablation study to assess the contribution of individual components in our framework, as summarized in Tab. IX. Setting (a) corresponds to SPFSplatV2, while setting (c) corresponds to SPFSplat. Compared to (a), setting (b) replaces the learnable pose token with global average pooling over feature maps, leading to a slight performance drop. This highlights the effectiveness of the improved pose estimation enabled by the learnable token. Compared with (c), setting (b) substitutes SPFSplatâs two-branch input design with masked attention yields consistent improvements in both novel view synthesis and pose estimation, demonstrating the benefits of masked attention. From (a) to (d), removing intrinsic embeddings in the backbone slightly reduces accuracy, primarily due to increased scale ambiguity in both 3D Gaussian learning and pose estimation. Nevertheless, even without intrinsic embeddings, our method still surpasses NoPoSplat with intrinsic embeddings (Tab. I), achieving improvements of 2.25% in PSNR, 1.07% in SSIM, and 4.38% in LPIPS under pose alignment evaluation. Finally, comparing (a) with (e), removing the reprojection loss while retaining only the image rendering loss on target views leads to a substantial degradation in both novel view synthesis and pose estimation accuracy. This underscores the crucial role of geometric constraints between 3D points and camera poses for accurate reconstruction.

In Tab. X, we evaluate our frameworkâs ability to reconstruct geometry without ground-truth pose supervision by analyzing the effect of incorporating ground-truth poses during training in two settings: (b) rendering novel views using ground-truth poses, as in NoPoSplat, and (c) introducing a pose loss that penalizes the discrepancy between predicted and ground-truth poses, while still rendering with predicted poses. The pose loss combines a geodesic loss [74] for rotation and an $L _ { 2 }$ loss for translation. Since ground-truth poses are used only during training, the framework remains pose-free at inference.

TABLE VIII  
POSE ESTIMATION PERFORMANCE USING AN AUGMENTED TRAINING SET (RE10K + DL3DV). THE IMPROVEMENT PERCENTAGE IS CALCULATED RELATIVE TO THE PERFORMANCE SHOWN IN TAB. IV.
<table><tr><td rowspan="2">Method</td><td colspan="3">RE10K</td><td colspan="3">ACID</td><td colspan="3">DL3DV</td><td colspan="3">ScanNet++</td></tr><tr><td> $5 ^ { \circ } \uparrow$ </td><td> $1 0 ^ { \circ } ~ \uparrow$ </td><td> $2 0 ^ { \circ } ~ \uparrow$ </td><td> $5 ^ { \circ } \uparrow$ </td><td> $1 0 ^ { \circ } ~ \uparrow$ </td><td> $2 0 ^ { \circ } ~ \uparrow$ </td><td> $5 ^ { \circ } \uparrow$ </td><td>10Â° â</td><td> $2 0 ^ { \circ } ~ \uparrow$ </td><td> $5 ^ { \circ } \uparrow$ </td><td>10Â° â</td><td> $2 0 ^ { \circ } ~ \uparrow$ </td></tr><tr><td>SPFSplatV2</td><td>0.652 +2.19%</td><td>0.785 +1.16%</td><td>0.867 +0.46%</td><td>0.390 +0.78%</td><td>0.543 +0.37%</td><td>0.675 +0.45%</td><td>0.560 +51.76%</td><td>0.711 +33.15%</td><td>0.806 +16.14%</td><td>0.251 +74.31%</td><td>0.478 +70.11%</td><td>0.698 +43.33%</td></tr><tr><td>SPFSplatV2 (PnP)</td><td>0.652 +1.72%</td><td>0.784 +0.90%</td><td>0.867</td><td>0.383</td><td>0.538</td><td>0.672</td><td>0.559</td><td>0.714</td><td>0.809</td><td>0.288</td><td>0.493</td><td>0.702</td></tr><tr><td>SPFSplatV2-L</td><td>0.654</td><td>0.788</td><td>+0.35% 0.870</td><td>+2.41% 0.405</td><td>+0.94% 0.558</td><td>+0.75% 0.687</td><td>+49.07% 0.568</td><td>+31.73% 0.713</td><td>+15.57% 0.809</td><td>+159.46% 0.268</td><td>+97.20% 0.473</td><td>+51.62% 0.670</td></tr><tr><td></td><td>+1.40%</td><td>+1.03%</td><td>+0.69%</td><td>+6.86%</td><td>+3.53%</td><td>+2.38%</td><td>+35.24%</td><td>+22.51%</td><td>+13.78%</td><td>+45.65%</td><td>+18.25%</td><td>+6.35%</td></tr><tr><td>SPFSplatV2-L (PnP)</td><td>0.668</td><td>0.794</td><td>0.873</td><td>0.404</td><td></td><td></td><td></td><td></td><td></td><td>0.261</td><td>0.469</td><td>0.667</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.556</td><td>0.685</td><td>0.568</td><td>0.717</td><td>0.811</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>+32.40%</td><td></td><td></td><td></td><td>+17.25%</td><td></td></tr><tr><td></td><td>+1.67%</td><td>+1.02%</td><td>+0.69%</td><td>+7.73%</td><td>+3.93%</td><td>+2.54%</td><td></td><td>+22.15%</td><td>+13.27%</td><td>+42.62%</td><td></td><td>+6.38%</td></tr></table>

TABLE IX

COMPONENT ABLATIONS ON RE10K. NVSâ DENOTES NOVEL VIEW SYNTHESIS EVALUATED WITH POSE ALIGNMENT.
<table><tr><td rowspan="2">#</td><td rowspan="2">M.</td><td rowspan="2">P.</td><td rowspan="2">I.</td><td rowspan="2">R.</td><td colspan="3">NVS</td><td colspan="3">NVS*</td><td colspan="3">Pose</td><td colspan="3">Pose (PnP)</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td> $5 ^ { \circ } \uparrow$ </td><td> $1 0 ^ { \circ } ~ \uparrow$ </td><td> $2 0 ^ { \circ } ~ \uparrow$ </td><td> $5 ^ { \circ } \uparrow$ </td><td>10Â° â</td><td> $2 0 ^ { \circ } ~ \uparrow$ </td></tr><tr><td>(a)</td><td>â</td><td>â</td><td>â</td><td>â</td><td>25.693</td><td>0.853</td><td>0.149</td><td>26.157</td><td>0.861</td><td>0.146</td><td>0.638</td><td>0.776</td><td>0.863</td><td>0.641</td><td>0.777</td><td>0.864</td></tr><tr><td>(b)</td><td>â</td><td>X</td><td>â</td><td>â</td><td>25.636</td><td>0.851</td><td>0.150</td><td>26.098</td><td>0.859</td><td>0.147</td><td>0.634</td><td>0.772</td><td>0.859</td><td>0.632</td><td>0.770</td><td>0.858</td></tr><tr><td>(c)</td><td>X</td><td>Ã</td><td>&gt;Ã</td><td>â</td><td>25.484</td><td>0.847</td><td>0.153</td><td>25.845</td><td>0.852</td><td>0.152</td><td>0.617</td><td>0.755</td><td>0.845</td><td>0.613</td><td>0.754</td><td>0.845</td></tr><tr><td>(d)</td><td>â â</td><td>â</td><td>â</td><td>â</td><td>24.998</td><td>0.834 0.651</td><td>0.157 0.280</td><td>25.597 22.013</td><td>0.847 0.751</td><td>0.153 0.244</td><td>0.546 0.023</td><td>0.716</td><td>0.829 0.393</td><td>0.581 0.015</td><td>0.738</td><td>0.841</td></tr><tr><td>(e)</td><td></td><td>â</td><td></td><td>Ã</td><td>19.818</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.144</td><td></td><td></td><td>0.064</td><td>0.197</td></tr></table>

âM.â indicates masked attention. âP.â indicates learnable pose token. âI.â indicates intrinsics embedding. âR.â indicates reprojection loss.

TABLE X  
GROUND-TRUTH POSES ABLATIONS ON RE10K.
<table><tr><td rowspan="2">Method</td><td colspan="3">NVS*</td><td colspan="3">Pose</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td> $5 ^ { \circ } \uparrow$ </td><td>10Â° â</td><td> $2 0 ^ { \circ } ~ \uparrow$ </td></tr><tr><td>(a) SPFSplatV2 (Ours)</td><td>26.157</td><td>0.861</td><td>0.146</td><td>0.638</td><td>0.776</td><td>0.863</td></tr><tr><td>(b) render with gt pose</td><td>25.033</td><td>0.838</td><td>0.160</td><td>0.571</td><td>0.727</td><td>0.833</td></tr><tr><td>(c) w/ gt pose loss</td><td>25.910</td><td>0.860</td><td>0.150</td><td>0.693</td><td>0.814</td><td>0.889</td></tr></table>

Relative to setting (b), (a) SPFSplatV2 achieves improvements in both novel view synthesis and pose estimation. This can be attributed to the joint optimization of Gaussians and poses, which encourages better geometric alignment and more consistent feature learning. From (a) to (c), adding a pose loss improves pose accuracy but yields only marginal gains in novel view synthesis, underscoring the modelâs capacity to reconstruct geometry without explicit pose supervision. These findings also suggest that high-quality novel view synthesis depends on factors beyond pose accuracy, such as occlusion, textureless regions, and extreme viewpoint changes, which may require generative priors or explicit 3D supervision.

Ablation on Initialization: In our main experiments, SPFSplatV2 is initialized with MASt3R weights, while SPFSplatV2-L uses VGGT weights. Tab. XI further analyzes the effect of different initialization strategies, showing that MASt3R slightly outperforms DUSt3R, likely due to its feature-matching pretraining, which yields stronger local features that benefit both pose estimation and 3D Gaussian reconstruction. For random initialization, we employ a warmup phase with a point cloud distillation loss from DUSt3R during the first 10,000 steps. This additional supervision is essential, as noted in [7], because training solely with a photometric loss, especially without ground-truth geometric supervision, makes it difficult for the network to learn Gaussians in the canonical space. While random initialization leads to a clear performance drop, the results still demonstrate the modelâs ability to reconstruct Gaussians. Notably, performance under random initialization remains significantly higher than SelfSplat (Tab. I), which also avoids pretrained 3D priors but relies on CroCoV2 [75] weights for feature extraction.

TABLE XI  
COMPARISON OF DIFFERENT INITIALIZATION STRATEGIES.
<table><tr><td>Method</td><td>Initialization</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td></td><td>Random</td><td>22.394</td><td>0.737</td><td>0.230</td></tr><tr><td>SPFSplatV2</td><td>DUSt3R</td><td>25.800</td><td>0.854</td><td>0.150</td></tr><tr><td></td><td>MASt3R</td><td>26.157</td><td>0.861</td><td>0.146</td></tr><tr><td></td><td>Random</td><td>22.226</td><td>0.724</td><td>0.224</td></tr><tr><td>SPFSplatV2-L</td><td>VGGT</td><td>25.983</td><td>0.859</td><td>0.136</td></tr></table>

Evaluation on In-the-Wild Data: We highlight the effectiveness of our model on mobile phone photos using SPFSplatV2 without intrinsic embeddings. The 3D geometry and rendered results in Fig. 8 demonstrate strong out-of-domain generalization, even under large viewpoint changes.

Failure Cases: As shown in Fig. 9, our method can produce blurred outputs or artifacts in occluded or texture-less regions, or under extreme viewpoint changes. Addressing these limitations may require stronger generative capabilities or larger training data.

<!-- image-->  
Fig. 8. 3D Gaussians from smartphone without intrinsics and rendered image.  
Ref.

GT  
Ref.  
<!-- image-->  
Fig. 9. Failure cases of SPFSplatV2. Blurriness and artifacts occur in occluded or texture-less regions and under extreme viewpoint changes.

## V. LIMITATIONS AND FUTURE WORK

Our method can be trained without ground-truth poses and scales effectively to large datasets, opening the possibility for future work to exploit more diverse data to further improve pose estimation and generalization. Nonetheless, it still benefits from the priors provided by supervised models such as MASt3R and VGGT, as evidenced by the performance drop when training from random initialization. Furthermore, since our approach is not generative, it cannot reconstruct unseen regions with high-fidelity textures. Incorporating generative models is a promising direction to address this limitation.

## VI. CONCLUSION

This paper presents SPFSplatV2, a self-supervised pose-free framework for 3D Gaussian splatting from sparse unposed views. By jointly optimizing camera poses and 3D Gaussian primitives through a unified backbone with masked attention, our approach achieves efficient and stable training as well as strong geometric consistency without requiring ground-truth poses. A reprojection loss is also incorporated with the conventional rendering loss to enforce pixel-aligned Gaussians. Extensive experiments on multiple datasets demonstrate that SPFSplatV2 and its larger variant SPFSplatV2-L establish new state-of-the-art results in novel view synthesis, crossdataset generalization, and relative pose estimation, even under challenging conditions of extreme viewpoint change and limited overlap. Importantly, the frameworkâs independence from ground-truth poses underscores its scalability to large and diverse real-world datasets, paving the way for future advances in scalable and generalizable 3D reconstruction.

## REFERENCES

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[3] A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su, âMvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 14 124â14 133.

[4] H. Xu, A. Chen, Y. Chen, C. Sakaridis, Y. Zhang, M. Pollefeys, A. Geiger, and F. Yu, âMurf: multi-baseline radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 041â20 050.

[5] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, âpixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 457â19 467.

[6] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.- J. Cham, and J. Cai, âMvsplat: Efficient 3d gaussian splatting from sparse multi-view images,â in European Conference on Computer Vision. Springer, 2024, pp. 370â386.

[7] B. Ye, S. Liu, H. Xu, X. Li, M. Pollefeys, M.-H. Yang, and S. Peng, âNo pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images,â in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=P4o9akekdf

[8] B. Smart, C. Zheng, I. Laina, and V. A. Prisacariu, âSplatt3r: Zeroshot gaussian splatting from uncalibrated image pairs,â arXiv preprint arXiv:2408.13912, 2024.

[9] K. Zhang, S. Bi, H. Tan, Y. Xiangli, N. Zhao, K. Sunkavalli, and Z. Xu, âGs-lrm: Large reconstruction model for 3d gaussian splatting,â in European Conference on Computer Vision. Springer, 2024, pp. 1â19.

[10] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu, âLgm: Large multi-view gaussian model for high-resolution 3d content creation,â in European Conference on Computer Vision. Springer, 2024, pp. 1â18.

[11] Y. Xu, Z. Shi, W. Yifan, H. Chen, C. Yang, S. Peng, Y. Shen, and G. Wetzstein, âGrm: Large gaussian reconstruction model for efficient 3d reconstruction and generation,â in European Conference on Computer Vision. Springer, 2024, pp. 1â20.

[12] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4104â4113.

[13] H. Jiang, Z. Jiang, Y. Zhao, and Q. Huang, âLEAP: Liberate sparseview 3d modeling from camera poses,â in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=KPmajBxEaF

[14] P. Wang, H. Tan, S. Bi, Y. Xu, F. Luan, K. Sunkavalli, W. Wang, Z. Xu, and K. Zhang, âPF-LRM: Pose-free large reconstruction model for joint pose and shape prediction,â in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=noe76eRcPC

[15] B. R. Nagoor Kani, H.-Y. Lee, S. Tulyakov, and S. Tulsiani, âUpfusion: Novel view diffusion from unposed sparse view observations,â in European Conference on Computer Vision (ECCV), 2024.

[16] M. S. Sajjadi, H. Meyer, E. Pot, U. Bergmann, K. Greff, N. Radwan, S. Vora, M. LuciË c, D. Duckworth, A. Dosovitskiy Â´ et al., âScene representation transformer: Geometry-free novel view synthesis through set-latent scene representations,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 6229â 6238.

[17] S. Hong, J. Jung, H. Shin, J. Yang, S. Kim, and C. Luo, âUnifying correspondence pose and nerf for generalized pose-free novel view synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 196â20 206.

[18] Y. Chen and G. H. Lee, âDbarf: Deep bundle-adjusting generalizable neural radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24â34.

[19] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, âBarf: Bundle-adjusting neural radiance fields,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5741â5751.

[20] S. Hong, J. Jung, H. Shin, J. Han, J. Yang, C. Luo, and S. Kim, âPf3plat: Pose-free feed-forward 3d gaussian splatting,â arXiv preprint arXiv:2410.22128, 2024.

[21] G. Kang, J. Yoo, J. Park, S. Nam, H. Im, S. Shin, S. Kim, and E. Park, âSelfsplat: Pose-free and 3d prior-free generalizable 3d gaussian splatting,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 22 012â22 022.

[22] R. Huang and K. Mikolajczyk, âNo pose at all: Self-supervised pose-free 3d gaussian splatting from sparse views,â arXiv preprint arXiv:2508.01171, 2025.

[23] V. Leroy, Y. Cabon, and J. Revaud, âGrounding image matching in 3d with mast3r,â in European Conference on Computer Vision. Springer, 2024, pp. 71â91.

[24] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, âVggt: Visual geometry grounded transformer,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294â 5306.

[25] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM transactions on graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[26] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, âK-planes: Explicit radiance fields in space, time, and appearance,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 479â12 488.

[27] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, âTensorf: Tensorial radiance fields,â in European conference on computer vision. Springer, 2022, pp. 333â350.

[28] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, âPlenoxels: Radiance fields without neural networks,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5501â5510.

[29] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, âDust3r: Geometric 3d vision made easy,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 697â20 709.

[30] J. Y. Zhang, D. Ramanan, and S. Tulsiani, âRelpose: Predicting probabilistic relative rotation for single objects in the wild,â in European Conference on Computer Vision. Springer, 2022, pp. 592â611.

[31] A. Lin, J. Y. Zhang, D. Ramanan, and S. Tulsiani, âRelpose++: Recovering 6d poses from sparse-view observations,â in 2024 International Conference on 3D Vision (3DV). IEEE, 2024, pp. 106â115.

[32] J. Wang, C. Rupprecht, and D. Novotny, âPosediffusion: Solving pose estimation via diffusion-aided bundle adjustment,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9773â9783.

[33] P. Truong, M.-J. Rakotosaona, F. Manhardt, and F. Tombari, âSparf: Neural radiance fields from sparse and noisy poses,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4190â4200.

[34] W. Bian, Z. Wang, K. Li, J.-W. Bian, and V. A. Prisacariu, âNope-nerf: Optimising neural radiance field with no pose prior,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4160â4169.

[35] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, âColmapfree 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 796â20 805.

[36] C. Smith, Y. Du, A. Tewari, and V. Sitzmann, âFlowcam: Training generalizable 3d radiance fields without camera poses via pixel-aligned scene flow,â Advances in Neural Information Processing Systems, vol. 36, pp. 1476â1488, 2023.

[37] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, âLightglue: Local feature matching at light speed,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17 627â17 638.

[38] O. Ronneberger, P. Fischer, and T. Brox, âU-net: Convolutional networks for biomedical image segmentation,â in Medical image computing and computer-assisted interventionâMICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer, 2015, pp. 234â241.

[39] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, âHighresolution image synthesis with latent diffusion models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684â10 695.

[40] R. Hartley and A. Zisserman, Multiple view geometry in computer vision. Cambridge university press, 2003.

[41] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4104â4113.

[42] D. G. Lowe, âDistinctive image features from scale-invariant keypoints,â International journal of computer vision, vol. 60, pp. 91â110, 2004.

[43] H. Bay, A. Ess, T. Tuytelaars, and L. Van Gool, âSpeeded-up robust features (surf),â Comput. Vis. Image. Und., vol. 110, no. 3, pp. 346â 359, 2008.

[44] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, âOrb: An efficient alternative to sift or surf,â in Proc. IEEE Int. Conf. Comput. Vision. (ICCV). Ieee, 2011, pp. 2564â2571.

[45] M. FISCHLER AND, âRandom sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,â Commun. ACM, vol. 24, no. 6, pp. 381â395, 1981.

[46] R. I. Hartley and P. Sturm, âTriangulation,â Computer vision and image understanding, vol. 68, no. 2, pp. 146â157, 1997.

[47] B. Triggs, P. F. McLauchlan, R. I. Hartley, and A. W. Fitzgibbon, âBundle adjustmentâa modern synthesis,â in International workshop on vision algorithms. Springer, 1999, pp. 298â372.

[48] D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperpoint: Selfsupervised interest point detection and description,â in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 224â236.

[49] M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler, âD2-Net: A Trainable CNN for Joint Detection and Description of Local Features,â in Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[50] R. Huang, J. Cai, C. Li, Z. Wu, X. Liu, and Z. Chai, âDrkf: Distilled rotated kernel fusion for efficient rotation invariant descriptors in local feature matching,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 1885â1892.

[51] Z. Luo, L. Zhou, X. Bai, H. Chen, J. Zhang, Y. Yao, S. Li, T. Fang, and L. Quan, âAslfeat: Learning local features of accurate shape and localization,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 6589â6598.

[52] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperglue: Learning feature matching with graph neural networks,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4938â4947.

[53] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, âLoftr: Detectorfree local feature matching with transformers,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 8922â8931.

[54] C. Tang and P. Tan, âBa-net: Dense bundle adjustment network,â arXiv preprint arXiv:1806.04807, 2018.

[55] J. Wang, N. Karaev, C. Rupprecht, and D. Novotny, âVggsfm: Visual geometry grounded deep structure from motion,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 21 686â21 697.

[56] Z. Tang, Y. Fan, D. Wang, H. Xu, R. Ranjan, A. Schwing, and Z. Yan, âMv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5283â5293.

[57] J. Yang, A. Sax, K. J. Liang, M. Henaff, H. Tang, A. Cao, J. Chai, F. Meier, and M. Feiszli, âFast3r: Towards 3d reconstruction of 1000+ images in one forward pass,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 924â21 935.

[58] P. Weinzaepfel, V. Leroy, T. Lucas, R. Bregier, Y. Cabon, V. Arora, Â´ L. Antsfeld, B. Chidlovskii, G. Csurka, and J. Revaud, âCroco: Selfsupervised pre-training for 3d vision tasks by cross-view completion,â Advances in Neural Information Processing Systems, vol. 35, pp. 3502â 3516, 2022.

[59] S. Zhang, J. Wang, Y. Xu, N. Xue, C. Rupprecht, X. Zhou, Y. Shen, and G. Wetzstein, âFlare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 936â 21 947.

[60] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, âAn image is worth 16x16 words: Transformers for image recognition at scale,â in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[61] R. Ranftl, A. Bochkovskiy, and V. Koltun, âVision transformers for dense prediction,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 12 179â12 188.

[62] E. Brachmann, T. Cavallari, and V. A. Prisacariu, âAccelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 5044â5053.

[63] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, âOn the continuity of rotation representations in neural networks,â in Proceedings of the

IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 5745â5753.

[64] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.

[65] E. Brachmann, J. Wynn, S. Chen, T. Cavallari, A. Monszpart, D. Tur- Â´ mukhambetov, and V. A. Prisacariu, âScene coordinate reconstruction: Posing of image collections via incremental learning of a relocalizer,â in European Conference on Computer Vision. Springer, 2024, pp. 421â 440.

[66] T. Zhou, R. Tucker, J. Flynn, G. Fyffe, and N. Snavely, âStereo magnification: learning view synthesis using multiplane images,â ACM Transactions on Graphics (TOG), vol. 37, no. 4, pp. 1â12, 2018.

[67] A. Liu, R. Tucker, V. Jampani, A. Makadia, N. Snavely, and A. Kanazawa, âInfinite nature: Perpetual view generation of natural scenes from a single image,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 14 458â14 467.

[68] J. Edstedt, Q. Sun, G. Bokman, M. Wadenb Â¨ ack, and M. Felsberg, âRoma:Â¨ Robust dense feature matching,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 790â19 800.

[69] L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo, Z. Yu, Y. Lu et al., âDl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 160â22 169.

[70] R. Jensen, A. Dahl, G. Vogiatzis, E. Tola, and H. AanÃ¦s, âLarge scale multi-view stereopsis evaluation,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2014, pp. 406â413.

[71] C. Yeshwanth, Y.-C. Liu, M. NieÃner, and A. Dai, âScannet++: A highfidelity dataset of 3d indoor scenes,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 12â22.

[72] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE transactions on image processing, vol. 13, no. 4, pp. 600â612, 2004.

[73] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., âDinov2: Learning robust visual features without supervision,â arXiv preprint arXiv:2304.07193, 2023.

[74] S. S. M. Salehi, S. Khan, D. Erdogmus, and A. Gholipour, âReal-time deep pose estimation with geodesic loss for image-to-template rigid registration,â IEEE transactions on medical imaging, vol. 38, no. 2, pp. 470â481, 2018.

[75] P. Weinzaepfel, T. Lucas, V. Leroy, Y. Cabon, V. Arora, R. Bregier, Â´ G. Csurka, L. Antsfeld, B. Chidlovskii, and J. Revaud, âCroco v2: Improved cross-view completion pre-training for stereo matching and optical flow,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17 969â17 980.