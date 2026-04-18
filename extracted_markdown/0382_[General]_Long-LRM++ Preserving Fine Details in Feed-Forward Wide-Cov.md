# Long-LRM++: Preserving Fine Details in Feed-Forward Wide-Coverage Reconstruction

Chen Ziwen1 Hao Tan1 Peng Wang2 Zexiang Xu3 Li Fuxin4 1Adobe Research 2Tripo AI 3Hillbot 4Oregon State University

## Abstract

Recent advances in generalizable Gaussian splatting (GS) have enabled feed-forward reconstruction of scenes from tens of input views. Long-LRM notably scales this paradigm to 32 input images at 960Ã540 resolution, achieving 360Â° scene-level reconstruction in a single forward pass. However, directly predicting millions of Gaussian parameters at once remains highly error-sensitive: small inaccuracies in positions or other attributes lead to noticeable blurring, particularly in fine structures such as text. In parallel, implicit representation methods such as LVSM and LaCT have demonstrated significantly higher rendering fidelity by compressing scene information into model weights rather than explicit Gaussians, and decoding RGB frames using the full transformer or TTT backbone. However, this computationally intensive decompression process for every rendered frame makes real-time rendering infeasible. These observations raise key questions: Is the deep, sequential âdecompressionâ process necessary? Can we retain the benefits of implicit representations while enabling real-time performance? We address these questions with Long-LRM++, a model that adopts a semi-explicit scene representation combined with a lightweight decoder. Long-LRM++ matches the rendering quality of LaCT on DL3DV while achieving real-time 14 FPS rendering on an A100 GPU, overcoming the speed limitations of prior implicit methods. Our design also scales to 64 input views at the 950Ã540 resolution, demonstrating strong generalization to increased input lengths. Additionally, Long-LRM++ delivers superior novel-view depth prediction on ScanNetv2 compared to direct depth rendering from Gaussians. Extensive ablation studies validate the effectiveness of each component in the proposed framework. Project page: http: //arthurhero.github.io/projects/llrm2/

## 1. Introduction

In recent years, novel-view synthesis has emerged as a prominent approach for 3D reconstruction, enabling the generation of photorealistic renderings of real-world scenes from arbitrary viewpoints. The milestone work NeRF [26] encodes scene information implicitly in the weights of a multilayer perceptron (MLP), which predicts color and density for any 3D location given a viewing direction. While this implicit formulation offers a compact representation, NeRF requires hours of optimization per scene and remains far from realtime during rendering. In contrast, 3D Gaussian Splatting (3D GS) [14] introduces a fully explicit representation using a set of 3D Gaussians parameterized by position, color, and opacity. This explicit formulation enables efficient optimizationâtypically within minutes per sceneâand achieves real-time rendering without any neural network inference.

<!-- image-->  
Figure 1. We present Long-LRM++, a feed-forward novel-view synthesis method for high-resolution scene reconstruction. By leveraging a semi-explicit feature-Gaussian representation, Long-LRM++ substantially reduces the blurriness issues observed in Long-LRM while preserving real-time rendering speed.

Following the success of 3D GS, several generalizable Gaussian Splatting (GS) methods [2, 5, 21, 43, 47, 52] have been developed. By training on large-scale multiview datasets, these approaches enable instant, feed-forward GS-based reconstruction. Among them, Long-LRM [52] achieves, for the first time, wide-coverage scene-level reconstruction at 950Ã540 resolution. However, its reconstructions often appear blurry, particularly in regions with fine textures or high-frequency details such as text. In parallel, implicit representationâbased feed-forward methods [12, 48] have emerged with better rendering quality. LVSM [12] (decoderonly) feeds the source images together with the target camera embedding as inputs to its pure transformer backbone. LaCT [48] further introduces a large-chunk test-time training layer, encoding reconstruction information in the âfast weightsâ, successfully scaling to large input sizes comparable to Long-LRM while delivering superior rendering quality. Nevertheless, these implicit representation methods sacrifice a key advantage of GS-based approachesâreal-time rendering, because each target frame rendering requires a full (often 24-layer) network forward pass. This trade-off is inherent: the more implicitly scene information is stored, the longer and more complex the decoding process must be.

Is it possible to overcome this trade-off and achieve better reconstruction while maintaining real-time rendering speed? We argue that a key limitation of feed-forward GS methods lies in the color splatting mechanism. Since each Gaussian represents only a single color, faithfully reproducing fine details in a scene requires predicting tens of millions of Gaussiansâoften one per pixelâto achieve high-resolution rendering. This imposes a heavy burden on feed-forward models that must predict all of the Gaussian parameters instantaneously. Moreover, even slight inaccuracies in the input camera poses or Gaussian positions can propagate and manifest as noticeable blurriness in the rendered results.

With this observation, we propose Long-LRM++, which leverages a semi-explicit representation to enhance Gaussian splatting for scene reconstruction. Similar to standard 3D GS, our approach maintains a set of 3D Gaussians positioned in an explicit 3D space with real-world metric correspondence. However, unlike conventional Gaussian splatting, Long-LRM++ relaxes the strict alignment between Gaussian positions/colors and the actual scene surfaces. In Long-LRM++, each Gaussian carries a feature vector rather than a fixed color, similar in spirit to neural point clouds. Unlike prior neural point cloudâbased NVS methods [1, 44], which initialize points using depth sensors or off-the-shelf multiview stereo algorithms, the Gaussians in Long-LRM++ are free to distribute anywhere in 3D space. During rendering, we first render a feature map at the target view using standard Gaussian splatting. This feature map is then decoded by a lightweight five-layer decoder into RGB color or depth. A key component of the decoder is the translation-invariant local-attention block, which significantly improves rendering quality compared to global attention with absolute positional embeddings. We further introduce a multi-space partitioning step that enables independent rendering and decoding of mutually exclusive Gaussian subsets, followed by feature merging before the final output layer. This design again substantially improves rendering quality.

Long-LRM++ combines the best of both worlds: it achieves rendering quality boost similar to LaCT while maintaining real-time rendering speed. We evaluate Long-LRM++ on two tasks: novel-view color rendering on DL3DV [19] at 960Ã540 resolution with up to 64 input views, and novelview color and depth rendering on ScanNetv2 [6] at 448Ã336 resolution with up to 128 input views. In both settings, Long-LRM++ achieves state-of-the-art results in terms of quality and efficiency. Its rendering quality significantly surpasses Long-LRM, despite using only 1/4 of its Gaussians. Compared to LaCT, Long-LRM++ achieves comparable rendering fidelity while being substantially faster in both reconstruction and rendering, maintaining a real-time 14 FPS rendering speed on a single A100 GPU. We further conduct extensive ablation studies on key design choices, including feature Gaussian prediction and rendering, the target-frame decoder architecture, and the multi-space partitioning design. In summary, Long-LRM++ demonstrates that feed-forward novel-view synthesis models can achieve both quality and efficiency for high-resolution, wide-coverage, scene-level reconstruction.

## 2. Related Work

Reconstruction representations in novel-view synthesis. A wide range of scene representations have been explored for novel-view synthesis, spanning different levels of explicitness to balance rendering quality and efficiency. Before NeRF [26], there have been methods employing neural point clouds [1], neural textures on meshes [39], neural volumes [23, 31, 34], multiplane images [25], and MLP-based implicit fields [35]. Neural Point-Based Graphics [1], for instance, initializes a point cloud from depth sensors, associates each point with a learnable descriptor, and renders novel views by decoding the projected features with a U-Net. Building upon NeRF, NSVF [20] learns a sparse voxel octree via differentiable ray marching, Plenoxels [7] optimizes a sparse 3D grid with spherical harmonics, Instant-NGP [27] accelerates NeRF with multiresolution hash encodings, TensoRF [4] factorizes the scene into low-rank tensors, and Point-NeRF [44] leverages neural point clouds initialized from off-the-shelf MVS methods. Among Gaussian-based methods, Scaffold-GS [24] predicts anchor Gaussians with learnable offsets, while SAGS [40] applies graph neural networks over Gaussian neighborhoods. Octree-GS [30] introduces octree-structured anchors to enable large-scale reconstruction. All these approaches, however, rely on perscene optimization rather than feed-forward inference.

Feed-forward novel-view synthesis. Both NeRF and 3D Gaussian Splatting (3D GS) have inspired feed-forward variants trained on large-scale datasets for instant reconstruction. Both NeRF-based approaches [3, 9, 13, 17, 22, 36, 41, 42, 46, 49] and Gaussian-based methods [2, 5, 11, 21, 43, 47, 52] typically involve volume rendering in their inference step. Among them, Long-LRM [52] is the first to achieve highresolution, wide-coverage scene-level reconstruction. In contrast, recent implicit representation methods [12, 48] bypass volume rendering and directly predict RGB colors. LVSM [12] employs a pure transformer backbone, achieving much higher fidelity than prior feed-forward models, but is still limited to low resolution and few input views. LaCT [48] scales to dozens of inputs by storing scene information in the fast weights of TTT layers, yielding substantially better rendering quality than Long-LRM. However, this implicit representation requires running the entire 24-layer backbone to render each frame, significantly limiting rendering speed. Long-LRM++ gets the best of both worlds by retaining the fast, explicit nature of 3D Gaussians while replacing the error-prone color splatting with the more robust feature splatting, coupled with a novel, lightweight yet powerful decoder.

<!-- image-->  
Figure 2. Overview of the Long-LRM++ architecture. Long-LRM++ takes up to 64 input images at 950Ã540 resolution along with their camera poses, and processes them using a backbone composed of interleaved Mamba2 and Transformer blocks, similar to Long-LRM. Each image token predicts K free-moving feature Gaussians (visualized with originating pixelâs color). During rendering, we introduce a multi-space partitioning step that divides the Gaussians into multiple subsets, each rendered and decoded independently. The target-frame decoder incorporates translation-invariant local-attention blocks to improve robustness and rendering quality. Finally, the decoded feature maps are merged and passed through a linear layer to produce the novel-view color or depth rendering.

## 3. Method

Feature rendering with 3D Gaussians. Several works also extend 3D Gaussians with feature vectors and render feature maps, but their goals differ from ours. Feature 3DGS [50] and Feature Splatting [28] augment each Gaussian with semantic features extracted from pre-trained 2D vision-language models such as CLIP [29], enabling downstream tasks like novel-view segmentation and scene decomposition. Spacetime Gaussian Feature Splatting [16] is an optimization-based 4D reconstruction method that replaces spherical harmonics with a 9-dimensional feature vector encoding RGB, view direction, and time, to more compactly represent radiance. BBSplat [38] and its feed-forward extension LGTM [15] print color texture maps on 2D Gaussians.

Long-LRM++ is a feed-forward novel-view synthesis model that takes a set of RGB input images and produces the RGB color or depth map at a target camera pose. The overall architecture consists of three main components: an input processing backbone, feature Gaussian prediction and rendering, and a target-frame decoder. We describe the input processing backbone in Sec. 3.1, the feature Gaussians prediction and rendering in Sec. 3.2, the target-frame decoding in Sec. 3.3, and the training objectives in Sec. 3.4.

## 3.1. Input Processing Backbone

Long-LRM++ adopts an input processing backbone similar to Long-LRM [52]. The input RGB images Iinput â $[ 0 , 1 ] ^ { V \times H \times \mathbf { \breve { W } } \times 3 }$ are first patchified through a linear projection into features of shape $\mathbb { R } ^ { V \times \frac { H } { p } \times \frac { W } { p } \times D }$ , and then reshaped into a row-major token sequence RLÃD, where $\begin{array} { r } { L \stackrel { \cdot } { = } V \times \frac { H } { p } \times \frac { W } { p } } \end{array}$ . The tokens are processed by a backbone composed of a mixture of Mamba2 and transformer blocks, following the design of [52], which has demonstrated better scalability than pure transformer backbones and higher rendering quality than pure Mamba-based ones. Different from [52], we employ a {1T7M}Ã3 block sequence, reduce the hidden dimension D to 768, and remove the token merging module. Since Long-LRM++ uses a semi-explicit scene representationâfeature Gaussians, it no longer requires dense, pixel-aligned Gaussians to capture high-frequency details for high-quality rendering, and thus only needs 1/4 number of Gaussians compared to Long-LRM, which significantly lowers GPU memory usage. The reduced memory footprint allows us to train at our highest-resolution setting without the token merging module, which, although computationally efficient, slightly degrades performance.

## 3.2. Feature Gaussian prediction and splatting

Instead of predicting one Gaussian per pixel as in [52], Long-LRM++ predicts K Gaussians per token through a linear projection. Each Gaussian is associated with a feature vector $\mathbf { \bar { f } } \in \mathbb { R } ^ { F }$ , a 2D offset $\mathbf { o } \in \mathbb { R } ^ { 2 }$ from the patch center, a camera-rayâaligned depth, and other Gaussian attributes including rotation, opacity, and scale. The 2D offsets allow the Gaussians to move freely from the patch centers, providing great spatial flexibility. The Gaussiansâ 3D positions can be calculated combining the 2D offsets and the ray-aligned depths. Given the predicted feature Gaussians and a target camera pose, we splat the Gaussian features onto a canvas of size $\textstyle { \frac { H } { r } } \times { \frac { W } { r } }$ , where $H \times W$ denotes the input image resolution and r is a hyperparameter. Different from color rendering with spherical harmonics, the feature vectors are view-independent. To incorporate view-dependent information, we compute the Plucker rays of the target camera and Â¨ concatenate them with the rendered feature map, resulting in a final tensor of shape $\begin{array} { r } { \textrm { p } { \frac { H } { r } } \times { \frac { W } { r } } \times ( F { + } 6 ) } \end{array}$ )

## 3.3. Target-frame decoder

Similar to the input images, the rendered feature map is patchified through a linear layer into shape $\mathbb { R } ^ { \frac { H } { q } \times \frac { W } { q } ^ { \mathbf { \lambda } } \times E } .$ where E denotes the hidden dimensionality of the decoder. The decoder is composed of several transformer-based attention blocks. If we were to employ only global self-attention, the same object rendered in different spatial regions would receive distinct positional embeddings, leading to inconsistent computation. To address this, the initial layers of the decoder adopt translation-invariant local-attention blocks, ensuring that the same object is processed consistently regardless of its location on the feature map. These local-attention layers are followed by several global-attention blocks, which aggregate global context across the entire frame. We provide an ablation study of the decoder design in Table 4. Finally, the decoded feature map is passed through a linear projection head to produce the final RGB color or depth output.

Translation-invariant local-attention block. In the localattention layer, each token attends only to its k nearest neighbors, gathering features locally. Instead of using global absolute positional embeddings, we adopt relative positional embeddings. Let $\Delta \mathbf { x } = ( \Delta x , \Delta y )$ denote the relative position between two tokens. Following [51], we compute

$$
\mathbf { r } = ( \Delta x , \Delta y , \| \Delta \mathbf { x } \| _ { 2 } , \frac { \Delta x } { \| \Delta \mathbf { x } \| _ { 2 } } , \frac { \Delta y } { \| \Delta \mathbf { x } \| _ { 2 } } )\tag{1}
$$

where the Euclidean distance $\| \Delta \mathbf { x } \| _ { 2 }$ is rotation-invariant, and the cosine and sine values are scale-invariant. Then, r is passed through a small MLP and the resulting embeddings are added to the $Q K$ attention scores before the softmax. In our experiments, we set the neighborhood size to $k = 6 4$ .

Multi-space partitioning and merging. Inspired by [45], we propose a Gaussian partitioning step that divides the set of feature Gaussians into multiple subsets, which are rendered and processed by the decoder independently, and merged before the final linear layer. One motivating example for this design is mirrors: by rendering the virtual scene inside a mirror independently of the real world and performing adaptively weighted merging, the virtual Gaussians need not adhere to the physical geometry of the real world, e.g., no virtual objects appear when viewing the back of the mirror. The partitioning and merging work simply. Each Gaussianâs feature vector is passed through a linear layer to produce S softmax-normalized weights. The Gaussians are then duplicated into S copies, with each copyâs opacity values suppressed by the corresponding weights. The S Gaussian splits are then rendered and decoded independently. The resulting S feature maps are passed through another linear layer to produce $S$ maps of merging weights. The final feature map is obtained via a weighted-sum operator and is fed into the output linear layer to produce RGB color or depth. Empirically, $S = 2$ provides a substantial performance boost compared to $S = 1$ (no partitioning) with minimal impact on rendering speed. Ablation results are reported in Table 5.

## 3.4. Training objectives

To supervise RGB color rendering at novel camera poses, Long-LRM++ adopts the same training objectives as [47, 52], combining Mean Squared Error (MSE) loss and Perceptual loss:

$$
{ \mathcal { L } } _ { \mathrm { c o l o r } } = \mathbf { M S E } \left( \mathbf { I } ^ { \mathrm { g t } } , \mathbf { I } ^ { \mathrm { p r e d } } \right) + { \boldsymbol { \lambda } } \cdot \mathbf { P e r c e p t u a l } \left( \mathbf { I } ^ { \mathrm { g t } } , \mathbf { I } ^ { \mathrm { p r e d } } \right)\tag{2}
$$

where Î» is empirically set to 0.5. For depth rendering at novel camera poses, Long-LRM++ adopts the log-L1 loss

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \mathrm { S m o o t h . L 1 } \left( \log \mathbf { D } ^ { \mathrm { g t } } , \log \mathbf { D } ^ { \mathrm { p r e d } } \right)\tag{3}
$$

. To further improve depth map training, we adopt two auxiliary losses from [32] that enforce consistency between order-1 gradients and normals derived from the depth maps. For both losses, the predicted and ground-truth depth maps are first downsampled to 1/4 of the original resolution, yielding $\mathbf { D } _ { 4 \downarrow } ^ { \mathrm { p r e d } }$ and $\mathbf { D } _ { 4 \downarrow } ^ { \mathrm { g i } }$ . The gradient loss is then computed as

$$
\mathcal { L } _ { \mathrm { g r a d } } = \sum _ { l = 1 } ^ { 4 } \mathrm { L } 1 \left( \mathrm { S o b e l . f i l t e r } \left( \mathbf { D } _ { 4 l \downarrow } ^ { \mathrm { g t } } \right) , \mathrm { S o b e l . f i l t e r } \left( \mathbf { D } _ { 4 l \downarrow } ^ { \mathrm { p r e d } } \right) \right)\tag{4}
$$

where the Sobel filter is applied along both vertical and horizontal directions. The normal loss is computed by first unprojecting the downsampled depth maps into 3D point maps, $\mathbf { P } _ { 4 \downarrow } ^ { \mathrm { p r e d } }$ and $\mathbf { P } _ { 4 \downarrow } ^ { \mathrm { g t } }$ . Sobel filters are then applied in both directions to approximate the normal maps, $\mathbf { N } _ { 4 \downarrow } ^ { \mathrm { p r e d } }$ and $\mathbf { N } _ { 4 \downarrow } ^ { \mathrm { g t } }$ Finally, a cosine similarity loss is applied:

$$
{ \mathcal { L } } _ { \mathrm { n o r m a l } } = 1 - \mathrm { C o s i n e \_ s i m i l a r i t y } \left( \mathbf { N } _ { 4 \downarrow } ^ { \mathrm { g t } } , \mathbf { N } _ { 4 \downarrow } ^ { \mathrm { p r e d } } \right)\tag{5}
$$

. The ablation results for the two auxiliary losses are reported in Table 6. The overall training objective for depth rendering is then defined as:

$$
{ \mathcal { L } } _ { \mathrm { d e p t h \mathrm { { - } t o t a l } } } = { \mathcal { L } } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { g r a d } } \cdot { \mathcal { L } } _ { \mathrm { g r a d } } + \lambda _ { \mathrm { n o r m a l } } \cdot { \mathcal { L } } _ { \mathrm { n o r m a l } }\tag{6}
$$

where we empirically set $\lambda _ { \mathrm { g r a d } } = \lambda _ { \mathrm { n o r m a l } } = 0 . 1$

We omit the opacity loss from [52], as our semi-explicit representation already compresses scene information into a much smaller set of Gaussians, making Gaussian pruning unnecessary for training at our highest-resolution setup. We also remove the soft depth supervision from [52], which is used to regularize 3D Gaussian positions. In Long-LRM++, Gaussians are free to distribute anywhere in 3D spaceârather than being constrained to object surfacesâas long as they enable accurate color and depth rendering at the target camera views.

## 4. Experiments

We train and evaluate Long-LRM++ on two tasks: novelview color rendering on the DL3DV [19] dataset, and novelview color and depth rendering on ScanNetv2 [6]. Common implementation details are provided in Sec. 4.1, while the training stage configuration and evaluation results for the two tasks are presented in Secs. 4.2 and 4.3, respectively.

## 4.1. Implementation Details

Unless otherwise specified, we follow most training and architecture settings of [52], including learning rate, optimizer, and data augmentation. The input processing backbone consists of 24 blocks arranged in a {1T7M}Ã3 sequence with hidden dimensionality 768. The target-frame decoder comprises 5 transformer blocks in an âLLGGGâ sequence (âLâ for local attention, âGâ for global attention) with hidden dimensionality 256. Following [12], we apply QK-Norm [8] in all transformer blocks. Input images are patchified with patch size $p = 8 .$ . For each image token, we predict $K = 1 6$ Gaussians with feature dimensionality $F = 3 2$ . Feature Gaussians are rendered onto a canvas of size ${ \frac { H } { r } } \times { \frac { W } { r } }$ with r = 2. Before the target-frame decoder, the rendered feature map is patchified into shape R $\begin{array} { r } { \frac { H } { q } \times \frac { W } { q } \times E } \end{array}$ with $q \ : = \ : 8$ and $E = 2 5 6$ . For multi-space partitioning and merging, we set $S = 2 .$

## 4.2. Novel-view color rendering on DL3DV

Dataset. DL3DV [19] is a large-scale, scene-level novelview synthesis dataset containing diverse indoor and outdoor scenes. Camera poses are obtained via COLMAP [33]. We train on the DL3DV-10K split, which includes over 10K scenes (excluding those in DL3DV-140), and evaluate on the DL3DV-140 benchmark, which consists of 140 representative scenes. For each scene, we sample every 8th frame as targe views and sample input views from the remaining frames using K-means clustering as in Long-LRM.

<table><tr><td>Input Views</td><td>Method</td><td>PSNRâ SSIMâLPIPSâ</td><td></td><td>Reconstruction Rendering Time</td><td>FPS</td></tr><tr><td rowspan="5">16</td><td> $3 \mathrm { D } \mathrm { G S } _ { 3 0 k }$ </td><td>21.20</td><td>0.708 0.264</td><td>13min</td><td>50+</td></tr><tr><td>Long-LRM</td><td>22.66 0.740</td><td>0.292</td><td>0.4sec</td><td>50+</td></tr><tr><td>LaCT</td><td>24.70 0.793</td><td>0.224</td><td>14.6sec</td><td>1.8</td></tr><tr><td>Ours</td><td>24.40 0.795</td><td>0.231</td><td>1.6sec</td><td>14</td></tr><tr><td> $3 \mathrm { D } \mathrm { G S } _ { 3 0 k }$ </td><td>23.60 0.779</td><td>0.213</td><td>13min</td><td>50+</td></tr><tr><td rowspan="6">32</td><td>Scaffold-GS30k</td><td>24.77</td><td>0.805</td><td></td><td></td><td>50+</td></tr><tr><td>Long-LRM</td><td></td><td></td><td>0.205</td><td>16min</td><td>50+</td></tr><tr><td></td><td>24.10</td><td>0.783</td><td>0.254</td><td>1sec</td><td></td></tr><tr><td>Long-LRM10 LaCT</td><td>25.60</td><td>0.826</td><td>0.233</td><td>37sec</td><td>50+</td></tr><tr><td></td><td>26.90 26.43</td><td>0.837</td><td>0.185</td><td>29.3sec 4.7sec</td><td>1.8</td></tr><tr><td>Ours</td><td></td><td>0.846</td><td>0.180</td><td></td><td>14</td></tr><tr><td rowspan="4">64</td><td> $3 \mathrm { D } \mathrm { G S } _ { 3 0 k }$ </td><td>26.43</td><td>0.854</td><td>0.167</td><td>14min</td><td>50+</td></tr><tr><td>Long-LRM</td><td>24.77</td><td>0.804</td><td>0.239</td><td>3sec</td><td>50+</td></tr><tr><td> $\mathrm { L a C T }$ </td><td>28.30</td><td>0.857</td><td>0.169</td><td>59sec</td><td>1.8</td></tr><tr><td>Ours</td><td>27.30</td><td>0.869</td><td>0.161</td><td>16.3sec</td><td>14</td></tr></table>

Table 1. Quantitative comparison of novel-view synthesis quality on the DL3DV-140 at 960Ã540 resolution. Subscripts denote the number of (post-prediction) optimization steps. âReconstruction Timeâ measures the latency of converting input images into the scene representation, while âRendering $\mathrm { F P S } ^ { \prime }$ reports the frame rate for generating RGB images from the representation. All timing results are measured on a single A100 GPU. Long-LRM++ is trained using 32 input views, and the results for 16- and 64-view settings are obtained in a zero-shot manner.

Training stages. Following [52], we adopt a curriculum training schedule divided into four stages, gradually increasing the image resolution from 256Ã256 to 960Ã540. Unlike [52], which starts directly with 32 input views, we begin with 8 input frames and scale to 32 only in the final stage to reduce the computation. See Sec. 10 for details.

Evaluation results. Quantitative evaluation results are shown in Table 1. Our baselines include the optimizationbased 3D GS [14], Long-LRM [52], which predicts an explicit set of Gaussians, and LaCT [48], which stores scene information implicitly in the fast weights of TTT [37] layers. Long-LRM++ is trained only on the 32-view setup and evaluated zero-shot on 16- and 64-view settings, demonstrating strong generalization to different input lengths. Compared to 3D GS and Long-LRM, Long-LRM++ achieves superior rendering quality across all input lengths and metrics. Notably, for 32 inputs, Long-LRM++ surpasses Long-LRMâs result even after 10 post-prediction optimization steps. For 64 inputs, Long-LRM underperforms the optimization-based 3D GS, whereas Long-LRM++ surpasses 3D GS by 0.9dB in PSNR. While slower than color-based Gaussians (50+ FPS), Long-LRM++ still maintains real-time rendering at 14 FPS. Compared to the fully implicit LaCT, Long-LRM++ achieves comparable rendering quality across all input sequencesâespecially excelling in SSIMâwhile rendering 8Ã faster, highlighting the benefits of our semi-explicit representation design.

<!-- image-->  
Long-LRM (1sec)  
Long-LRM++ (4.7sec)

Figure 3. Qualitative comparison on reconstruction details of DL3DV (32-input, 960Ã540-resolution) with baselines, including optimizationbased methods. Reconstruction time in parentheses. Long-LRM++ notably produces fine details such as readable text in just 5 seconds.
<table><tr><td rowspan="2">Input Views</td><td rowspan="2">Method</td><td colspan="2">Color rendering</td><td colspan="3">Depth rendering</td></tr><tr><td>PSNRâ SSIMâ LPIPSâ Abs Diffâ Abs Relâ Sq Relâ Î´ &lt; 1.25â</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">128</td><td>Long-LRM</td><td>22.36 0.727</td><td>0.359</td><td>0.184</td><td>0.113 0.044</td><td>0.870</td></tr><tr><td>Ours</td><td>26.96 0.807</td><td>0.279 0.131</td><td>0.080</td><td>0.023</td><td>0.935</td></tr></table>

Table 2. Quantitative comparison of novel-view color and depth rendering quality on the ScanNetv2 test split at 448Ã336 resolution. For each scene, we sample every 8th frame from the first 1280 frames as target views, and uniformly sample 128 frames from the remaining sequence as input views.

## 4.3. Novel-view rendering on ScanNetv2

Dataset. ScanNetv2 [6] is a large-scale indoor dataset widely used for 3D reconstruction. Each RGB stream is paired with corresponding depth maps and ground-truth camera poses. The training split contains 1,201 rooms, while the test split includes 100 rooms.

Training stages. We adopt a curriculum, multi-stage training strategy, gradually increasing the image resolution from 256Ã256 to 448Ã336 and the number of input frames from 8 to 128. See Sec. 10 for details.

Evaluation results. Table 2 presents quantitative evaluation results comparing Long-LRM++ with Long-LRM for novel-view color and depth rendering on ScanNetv2. We train Long-LRM using the same training stages and training objectives as Long-LRM++, directly supervising depth maps rendered from the 3D Gaussians. During evaluation, we use the first 1,280 frames of each scene, uniformly sampling every 8th frame as target views and 128 frames from the remaining frames as input. Long-LRM++ demonstrates superior performance for both color and depth rendering. For color, it achieves a +4.6dB PSNR improvement, while for depth, it reduces the Absolute Difference metric by 0.053.

## 5. Analysis

<table><tr><td>#Gaussian Gaussian Feature splat per token feature dim canvas size</td><td></td><td>PSNRâ</td></tr><tr><td>8</td><td>32 0.5</td><td>27.94</td></tr><tr><td>8 16</td><td>64 0.5 32 0.25</td><td>28.20 27.81</td></tr><tr><td>16</td><td>32 0.5</td><td>28.37</td></tr><tr><td>16</td><td>32 1.0</td><td>28.41</td></tr><tr><td>16</td><td>64 0.5</td><td>28.44</td></tr><tr><td>32</td><td>16 0.5</td><td>26.71</td></tr></table>

<table><tr><td>Splat type</td><td>Decoder Decoder blocks</td><td>dim</td><td>PSNRâ</td></tr><tr><td>Color</td><td>I</td><td>I</td><td>27.32</td></tr><tr><td>Feature</td><td>GGG</td><td>256</td><td>27.78</td></tr><tr><td>Feature</td><td>LLL</td><td>256</td><td>28.23</td></tr><tr><td>Feature</td><td>LLG</td><td>256</td><td>28.37</td></tr><tr><td>Feature LLGGG</td><td></td><td>256</td><td>28.74</td></tr><tr><td>Feature LLGGG</td><td></td><td>768</td><td>29.09</td></tr></table>

Table 3. Ablation studies on number of Gaussians per token, Gaussian feature dimensionality, and feature-splat canvas size (ratio to input size), evaluated on DL3DV-140 under the 4-view, 256- resolution setting with the âLLGâ decoder. row is the chosen setup.

Table 4. Ablation studies on decoder block architecture and dimensionality. âLâ denotes a local-attention layer and âGâ denotes a globalattention layer, evaluated on DL3DV-140 under the 4-view, 256-resolution setting.
<table><tr><td rowspan="2">#Partition</td><td colspan="3">8-input,256-res</td><td colspan="3">32-view, 960Ã540</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>PSNRâ SSIMâ LPIPSâ PSNRâ SSIMâ LPIPSâ Recon Timeâ Render FPSâ</td></tr><tr><td>1</td><td>28.74</td><td>0.906</td><td>0.093 26.08</td><td>0.836</td><td>0.191 4.7sec</td><td>18</td></tr><tr><td>2</td><td>29.73 0.919</td><td></td><td>0.084</td><td>26.43 0.846 0.180</td><td>4.7sec</td><td>14</td></tr><tr><td>3</td><td>29.82 0.922 0.074</td><td></td><td></td><td>-</td><td>-</td><td>-</td></tr></table>

Table 5. Ablation studies on the number of Gaussian partitions, evaluated on DL3DV-140.

## 5.1. Hyperparameter studies for feature rendering

In Table 3, we present the rendering quality under different hyperparameter configurations for the feature Gaussian design, including the number of Gaussians K predicted by each image token, the feature vector dimensionality F of each Gaussian, and the size of the feature rendering canvas. Intuitively, increasing either K or F allows the model to store more information about the scene, thereby improving rendering quality. A larger K enables finer spatial resolution and more flexible distribution of features in 3D space, whereas a larger F allows each Gaussian to encode more complex local information, and also increases the âthicknessâ of the rendered feature map. However, indiscriminately increasing K or F can result in substantially higher memory consumption. Therefore, we need to strike a balance between the two. To investigate this trade-off, we compare three configurations: âK8, F 64â, âK16, F 32â, âK32, F 16â.

<!-- image-->  
Figure 4. More qualitative comparison on DL3DV (32-input, 960Ã540-resolution). PSNR is shown below each rendered image. Long-LRM++ notably sharpens fine details compared to Long-LRM, and produces more faithful light reflections than LaCT.

<table><tr><td rowspan="2">Method</td><td colspan="2">Color rendering</td><td colspan="2">Depth rendering</td></tr><tr><td>PSNRâ SSIMâLPIPSâ Abs Diffâ Î´ &lt; 1.25â</td><td></td><td></td><td></td></tr><tr><td>Long-LRM2DGS</td><td>23.89 0.744</td><td>0.344</td><td>0.227</td><td>0.821</td></tr><tr><td>Long-LRM3DGS</td><td>24.30</td><td>0.759 0.321</td><td>0.243</td><td>0.818</td></tr><tr><td>OurSw/o aux losses</td><td>28.28</td><td>0.835 0.226</td><td>0.157</td><td>0.911</td></tr><tr><td>Ours</td><td>27.86</td><td>0.826 0.234</td><td>0.135</td><td>0.916</td></tr></table>

Table 6. Ablation studies on novel-view color and depth rendering on ScanNetv2 under the 8-input, 256- resolution setup. Methods are trained for 10K steps.

These settings have roughly similar memory footprint, while âK16, F 32â achieves the best rendering quality, suggesting that this configuration provides the optimal balance between spatial flexibility (controlled by K) and feature complexity (controlled by F ). The rendering canvas size determines the resolution of the rendered feature map. Increasing it generally improves rendering quality, but the benefit diminishes as the resolution grows. Taking both rendering quality and efficiency into account, we select the highlighted row as our final configuration. Note that F = 32 is smaller than the 48 channels used by color Gaussians with SH degree 3.

## 5.2. Ablation studies for feature decoding

Target-frame decoder. Table 4 presents the rendering quality under different decoder block configurations and hidden dimensionalities. Comparing âGGGâ and âLLLâ, we observe that using purely local-attention blocks yields better performance than purely global-attention ones. Further, the hybrid âLLGâ, where local-attention blocks are followed by global attention, achieves even higher quality, suggesting that global feature aggregation is beneficial. As expected, increasing the hidden dimensionality from 256 to 768 also improves performance. Balancing rendering quality and efficiency, we adopt the highlighted row as our final configuration.

<!-- image-->  
Figure 5. Qualitative comparison of novel-view color and depth rendering on ScanNetv2 (128-input, 448Ã336 resolution). Long-LRM++ outperforms Long-LRM and produces high-quality depth maps despite using a sparse set of free-moving feature Gaussians.

Multi-space partitioning and merging. Table 5 demonstrates the effectiveness of the proposed partitioning and merging design. Increasing the partition count S from 1 to 2 consistently improves performance across all three rendering quality metrics, while only slightly affecting rendering speed. However, further increasing S to 3 yields diminishing returns in performance under the low-resolution setup, so we opt not to adopt it in our final configuration.

## 5.3. Ablation studies for novel-view depth rendering

Table 6 presents ablation studies for the novel-view depth rendering task. We compare the performance of Long-LRM with a 3D Gaussian head, Long-LRM with a 2D Gaussian head [10], and Long-LRM++. The results show that Long-LRM3DGS achieves higher color rendering quality but slightly worse depth accuracy than Long-LRM2DGS, while Long-LRM++ outperforms both by a substantial margin on both color and depth rendering metrics. We further ablate the two auxiliary losses used for depth supervisionâthe gradient loss and the normal loss. The results indicate that these auxiliary signals effectively enhance depth rendering quality, though they slightly degrade color performance. We speculate this trade-off arises because these losses encourage the decoder to allocate more capacity toward depth reconstruction rather than color decoding.

## 6. Conclusion

We introduced Long-LRM++, a feed-forward novel-view synthesis framework for high-resolution scene reconstruction. In contrast to Long-LRM, which predicts pixel-aligned color Gaussians, Long-LRM++ leverages a set of free-moving feature Gaussians that provide greater robustness and stronger representational capacity. Together with a lightweight decoder that incorporates translation-invariant local-attention blocks, as well as a novel multi-space partitioningâandâmerging mechanism, Long-LRM++ effectively resolves the blurriness artifacts observed in Long-LRM and delivers a substantial boost in rendering quality. Our method achieves performance competitive with fully implicit approaches such as LaCT, yet maintains real-time rendering speed, whereas LaCT suffers from slow inference. Furthermore, the proposed semi-explicit representation reduces the number of required Gaussians, lowers storage cost, and improves robustness for long input sequences. Overall, Long-LRM++ demonstrates an efficient and scalable path toward high-quality, high-resolution feed-forward 3D scene reconstruction.

## References

[1] Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lempitsky. Neural point-based graphics. In European conference on computer vision, pages 696â712. Springer, 2020. 2

[2] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19457â19467, 2024. 1, 2

[3] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14124â14133, 2021. 2

[4] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision (ECCV), 2022. 2

[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. arXiv preprint arXiv:2403.14627, 2024. 1, 2

[6] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828â5839, 2017. 2, 5, 6

[7] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5501â5510, 2022. 2

[8] Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers. arXiv preprint arXiv:2010.04245, 2020. 5

[9] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. In The Twelfth International Conference on Learning Representations, 2024. 2

[10] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1â11, 2024. 8

[11] Guichen Huang, Ruoyu Wang, Xiangjun Gao, Che Sun, Yuwei Wu, Shenghua Gao, and Yunde Jia. Longsplat: Online generalizable 3d gaussian splatting from long sequence images. arXiv preprint arXiv:2507.16144, 2025. 2

[12] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242, 2024. 2, 3, 5

[13] Mohammad Mahdi Johari, Yann Lepoittevin, and FrancÂ¸ois Fleuret. Geonerf: Generalizing nerf with geometry priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18365â18375, 2022. 2

[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 5

[15] Yixing Lao, BAI Xuyang, Xiaoyang Wu, Nuoyuan Yan, Zixin Luo, Tian Fang, Jean-Daniel Nahmias, Yanghai Tsin, Shiwei Li, and Hengshuang Zhao. Less gaussians, texture more: 4k feed-forward textured splatting. In The Fourteenth International Conference on Learning Representations. 3

[16] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8508â8520, 2024. 3

[17] Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Efficient neural radiance fields for interactive free-viewpoint video. In SIGGRAPH Asia 2022 Conference Papers, pages 1â9, 2022. 2

[18] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 1

[19] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learningbased 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160â 22169, 2024. 2, 5

[20] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651â15663, 2020. 2

[21] Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen, Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, and Ziwei Liu. Fast generalizable gaussian splatting reconstruction from multi-view stereo. arXiv preprint arXiv:2405.12218, 2024. 1, 2

[22] Yuan Liu, Sida Peng, Lingjie Liu, Qianqian Wang, Peng Wang, Christian Theobalt, Xiaowei Zhou, and Wenping Wang. Neural rays for occlusion-aware image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7824â7833, 2022. 2

[23] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural volumes: Learning dynamic renderable volumes from images. arXiv preprint arXiv:1906.07751, 2019. 2

[24] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 2

[25] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1â14, 2019. 2

[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[27] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022. 2

[28] Ri-Zhao Qiu, Ge Yang, Weijia Zeng, and Xiaolong Wang. Feature splatting: Language-driven physics-based scene synthesis and editing. arXiv preprint arXiv:2404.01223, 2024. 3

[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748â8763. PmLR, 2021. 3

[30] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024. 2

[31] Gernot Riegler and Vladlen Koltun. Stable view synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12216â12225, 2021. 2

[32] Mohamed Sayed, John Gibson, Jamie Watson, Victor Prisacariu, Michael Firman, and Clement Godard. Simplere- Â´ con: 3d reconstruction without 3d convolutions. In European Conference on Computer Vision, pages 1â19. Springer, 2022. 4

[33] Johannes L Schonberger, Enliang Zheng, Jan-Michael Frahm, Â¨ and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In Computer VisionâECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, pages 501â518. Springer, 2016. 5

[34] Vincent Sitzmann, Justus Thies, Felix Heide, Matthias NieÃner, Gordon Wetzstein, and Michael Zollhofer. Deepvoxels: Learning persistent 3d feature embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2437â2446, 2019. 2

[35] Vincent Sitzmann, Michael Zollhofer, and Gordon Wetzstein. Â¨ Scene representation networks: Continuous 3d-structureaware neural scene representations. Advances in neural information processing systems, 32, 2019. 2

[36] Mohammed Suhail, Carlos Esteves, Leonid Sigal, and Ameesh Makadia. Generalizable patch-based neural rendering. In European Conference on Computer Vision, pages 156â174. Springer, 2022. 2

[37] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024. 5

[38] David Svitov, Pietro Morerio, Lourdes Agapito, and Alessio Del Bue. Billboard splatting (bbsplat): Learnable textured

primitives for novel view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 25029â25039, 2025. 3

[39] Justus Thies, Michael Zollhofer, and Matthias NieÃner. De- Â¨ ferred neural rendering: Image synthesis using neural textures. Acm Transactions on Graphics (TOG), 38(4):1â12, 2019. 2

[40] Evangelos Ververas, Rolandos Alexandros Potamias, Jifei Song, Jiankang Deng, and Stefanos Zafeiriou. Sags: Structure-aware 3d gaussian splatting. In European Conference on Computer Vision, pages 221â238. Springer, 2024. 2

[41] Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan, Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai Zhang. Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction. In The Twelfth International Conference on Learning Representations, 2023. 2

[42] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690â4699, 2021. 2

[43] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16453â16463, 2025. 1, 2

[44] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf: Pointbased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5438â5448, 2022. 2

[45] Ze-Xin Yin, Jiaxiong Qiu, Ming-Ming Cheng, and Bo Ren. Multi-space neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12407â12416, 2023. 4

[46] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578â4587, 2021. 2

[47] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. arXiv preprint arXiv:2404.19702, 2024. 1, 2, 4

[48] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025. 2, 3, 5

[49] Xiaoshuai Zhang, Sai Bi, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Nerfusion: Fusing radiance fields for largescale scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5449â5458, 2022. 2

[50] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Pro-

ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676â21685, 2024. 3

[51] Chen Ziwen, Kaushik Patnaik, Shuangfei Zhai, Alvin Wan, Zhile Ren, Alexander G Schwing, Alex Colburn, and Li Fuxin. Autofocusformer: Image segmentation off the grid. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18227â18236, 2023. 4

[52] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. arXiv preprint 2410.12781, 2024. 1, 2, 3, 4, 5

# Long-LRM++: Preserving Fine Details in Feed-Forward Wide-Coverage Reconstruction

Supplementary Material

<!-- image-->  
Figure 6. Qualitative comparison with GS-LRM for object reconstruction on the GSO dataset.  
Figure 7. Qualitative comparison with Long-LRM on the Tanks&Temples dataset.

## 7. More implementation details

Due to its semi-explicit formulation, Long-LRM++ exhibits a stronger tendency to overfit to input frames when training on mixed sets of input and unseen target frames. This effect becomes more pronounced on datasets such as DL3DV, where neighboring frames have relatively large pose differencesâthat is, the effective frame density per unit scene coverage is lower. To mitigate this overfitting, we manually reduce the probability that input frames are selected as target frames during training. Concretely, during the random sampling (without replacement) of target frame indices, we decrease the selection weight of input frames to 0.1 while keeping all other frames at 1. For denser datasets such as ScanNetv2, this adjustment is unnecessary because the number of unseen frames significantly exceeds the number of input frames.

<table><tr><td>Method</td><td>PSNRâ</td></tr><tr><td>GS-LRMdim768</td><td>31.39</td></tr><tr><td>GS-LRM Long-LRM++</td><td>31.95 32.52</td></tr></table>

<table><tr><td>Method PSNRâ</td></tr><tr><td>GS-LRM 28.10</td></tr><tr><td>Long-LRM 28.54</td></tr><tr><td>Long-LRM++ 29.31</td></tr></table>

<table><tr><td>Method</td><td>PSNRâ</td></tr><tr><td>3D GS</td><td>18.10</td></tr><tr><td>Long-LRM</td><td>18.38</td></tr><tr><td>Long-LRM++</td><td>19.30</td></tr></table>

Table 7. Object recon- Table 8. Scene recon- Table 9. Scene reconstruction on GSO. struction on RE10K. struction on T&T.

## 8. More evaluation results

We evaluate Long-LRM++ on three additional datasets: object reconstruction on GSO (Table 7), scene reconstruction on RealEstate10K (Table 8), and zero-shot scene reconstruction on Tanks&Temples (Table 9). For GSO, we train Long-LRM++ on Objaverse for 80K steps and evaluate on GSO, comparing against GS-LRM and GS-LRMdim768, which matches Long-LRM++âs backbone dimension. As shown in Fig. 6, Long-LRM++ achieves superior detail fidelity. For RealEstate10K, we train on the training split for 100K steps and evaluate on the test split, achieving stateof-the-art performance. For Tanks&Temples, we conduct zero-shot evaluation using a model trained on DL3DV, obtaining a +1 dB PSNR improvement over Long-LRM (see qualitative results in Fig. 7).

## 9. Comparison with Depth Anything 3 (DA3)

We compare Long-LRM++ (110M param) with the Gaussian prediction feature of DA3-GIANT [18] (1.15B param) on DL3DV using both COLMAP poses and DA3 poses. To obtain DA3 poses, we run the pose predictor of DA3 on all frames of a scene. During inference, we feed the obtained poses of selected input frames to the Gaussian prediction models. Quantitative comparison is shown in Table 10 and qualitative in Fig. 8. Both Long-LRM and Long-LRM++ show better rendering quality than DA3-GIANT.

<table><tr><td>Pose Source</td><td>Method</td><td>PSNRâSSIMâLPIPSâ</td><td></td></tr><tr><td rowspan="2">COLMAP</td><td>DA3-GIANT</td><td>17.52</td><td>0.562 0.382</td></tr><tr><td>Long-LRM</td><td>24.10</td><td>0.783 0.254 0.180</td></tr><tr><td rowspan="2">DA3</td><td>Long-LRM++ DA3-GIANT</td><td>26.43 17.27</td><td>0.846</td></tr><tr><td>Long-LRM</td><td>22.98</td><td>0.540 0.395 0.731 0.277</td></tr><tr><td rowspan="2"></td><td>Long-LRM++</td><td></td><td></td></tr><tr><td></td><td>24.43</td><td>0.773 0.212</td></tr></table>

Table 10. Quantitative comparison with DA3 on DL3DV-140 (32 input views, 960Ã540 resolution).

## 10. Training stage configuration

Table 11 summarizes the detailed setup for each stage of training of Long-LRM++ on DL3DV, including training iterations, number of GPUs, and total GPU hours. Table 12 summarizes the stage setup for the ScanNetv2 training.

<table><tr><td colspan="5">Stage #Input #Target Resolution Time/Step #Step</td><td rowspan="2"></td><td rowspan="2">Batch size</td><td rowspan="2">#GPU</td><td rowspan="2">GPU Hours</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>8</td><td>8</td><td>256Ã256</td><td>9.6sec</td><td>60K</td><td>256</td><td>16</td><td>2560</td></tr><tr><td>2</td><td>8</td><td>8</td><td>512Ã512</td><td>7.7sec</td><td>10K</td><td>64</td><td>16</td><td>342</td></tr><tr><td>3</td><td>8</td><td>8</td><td>960x540</td><td>17.6sec</td><td>10K</td><td>64</td><td>16</td><td>782</td></tr><tr><td>4</td><td>32</td><td>8</td><td>960x540</td><td>27.4sec</td><td>10K</td><td>64</td><td>64</td><td>4871</td></tr></table>

Table 11. Training stage configuration of Long-LRM++ for the DL3DV10K novel-view synthesis task. GPU Hours is calculated as Time/Step Ã #Steps Ã #GPU.

<!-- image-->  
Ground truth

<!-- image-->  
DA3-GIANT

<!-- image-->  
Long-LRM

<!-- image-->  
Long-LRM++

Figure 8. Qualitative comparison with DA3 on DL3DV (32-input, 960Ã540-resolution) using DA3 poses.
<table><tr><td colspan="5">Stage #Input #Target Resolution Time/Step #Step</td><td rowspan="2"></td><td rowspan="2">Batch size</td><td rowspan="2">#GPU</td><td rowspan="2">GPU Hours</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>8</td><td>8</td><td>256X256</td><td>6.7sec</td><td>20K</td><td>128</td><td>8</td><td>298</td></tr><tr><td>2</td><td>8</td><td>8</td><td>448Ã336</td><td>3.9sec</td><td>5K</td><td>128</td><td>32</td><td>173</td></tr><tr><td>3</td><td>128</td><td>8</td><td>448Ã336</td><td>30.6sec</td><td>2K</td><td>64</td><td>64</td><td>1088</td></tr></table>

Table 12. Training stage configuration of Long-LRM++ for the ScanNetv2 novel-view color+depth rendering task.