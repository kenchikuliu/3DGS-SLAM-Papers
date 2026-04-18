# OG-Mapping: Octree-based Structured 3D Gaussians for Online Dense Mapping

Meng Wang1,2, Junyi Wang3, Changqun Xia2, Chen Wang4, Yue Qi1

1State Key Laboratory of Virtual Reality Technology and Systems, Beihang University. 2 PengCheng Laboratory. 3 School of Computer Science and Technology, Shandong University. 4 Beijing Technology and Business University.

<!-- image-->  
Figure 1. In this work, we introduce OG-Mapping, a novel online dense mapping framework with an octree-based structured 3D Gaussian representation. By combining our proposed anchor-based progressive map refinement strategy and dynamic keyframe window, OG-Mapping achieves fast, high-fidelity online reconstruction with efficient memory usage, and demonstrates superior realism in novel view synthesis compared to other existing RGB-D online mapping methods. The rendering FPS is indicated to the right of each method.

## Abstract

3D Gaussian splatting (3DGS) has recently demonstrated promising advancements in RGB-D online dense mapping. Nevertheless, existing methods excessively rely on per-pixel depth cues to perform map densification, which leads to significant redundancy and increased sensitivity to depth noise. Additionally, explicitly storing 3D Gaussian parameters of room-scale scene poses a significant storage challenge. In this paper, we introduce OG-Mapping, which leverages the robust scene structural representation capability of sparse octrees, combined with structured 3D Gaussian representations, to achieve efficient and robust online dense mapping. Moreover, OG-Mapping employs an anchor-based progressive map refinement strategy to recover the scene structures at multiple levels of detail. Instead of maintaining a small number of active keyframes with a fixed keyframe window as previous approaches do, a dynamic keyframe window is employed to allow OG-Mapping to better tackle false local minima and forgetting issues. Experimental results demonstrate that OG-Mapping delivers more robust and superior realism mapping results than existing Gaussian-based RGB-D online mapping methods with a compact model, and no additional post-processing is required.

## 1. Introduction

Constructing highly detailed dense maps in real-time holds significant importance in AR/VR, robotics, and digital twins applications. For the past several decades, research on mapping has extensively centered around the scene representation, resulting in various representations such as occupancy grids [9], TSDF [3, 44], surfels [1, 37] and point clouds [7]. Although systems utilizing these map representations have reached a production-ready standard over the past years, there remain notable deficiencies that demand attention and resolution. These methods exhibit deficiencies in rendering realism of reconstructed results, and finedetailed maps based on the above representations require massive amounts of storage space.

In recent years, implicit representations [2, 26, 35] have demonstrated promising outcomes in various domains following the advent of Neural Radiance Fields (NeRF) [25]. Numerous studies [6, 12, 14, 30, 34, 46] employ these implicit representations to enhance mapping methodologies and exhibit strengths in generating high-quality dense maps with low memory consumption. Nevertheless, volumetric sampling significantly limits these methodsâ efficiency. Consequently, they opt for optimization over a sparse set of pixels instead of dense per-pixel photometric error, resulting in the reconstruction dense maps lack the richness and intricacy of texture details.

More recently, several works [23, 41] based on the 3D Gaussian representation [16] and tile-based splatting technique have shown great superiority in the efficiency of differentiable rendering. However, these methods are meticulously designed for offline reconstruction scenarios. Concurrent works [11, 15, 24, 27, 38, 42] attempt to incorporate 3D Gaussians representation into RGB-D online dense mapping to deliver high-fidelity rendering, overcoming the limitation of Nerf-based methods. While promising results are demonstrated, these methods neglect the scene structure and rely excessively on pixel-level depth information to expand 3D Gaussians, resulting in significant redundancy and being sensitive to depth noise effects.

To overcome these challenges, in this paper, we introduce an innovative framework named OG-Mapping to perform efficient and highly detailed online dense mapping. Our solution comprises three main components. First, we incrementally build a sparse voxel octree with Morton coding [33] for fast allocation and retrieval of anchors to dynamically expand the map. Each anchor tethers a set of 3D Gaussians with learnable offsets. The attributes (color, opacity, quaternion, and scale) of these Gaussians are predicted based on the viewing direction and anchor feature encodings. By leveraging this structured representation, we can effectively mitigate pixel-level depth noise effects and avoid the substantial memory consumption associated with explicitly storing large amounts of 3D Gaussian attributes. Second, as the anchors organized by the sparse octree can only provide a rough description of the scene structure, we further design an anchor-based progressive map refinement strategy to recover the scene structures at different levels of detail by adaptively adding finer-level anchors. Unlike the previous method [23] utilizes an error-based policy, when an area is marked as under-optimized, we grow anchors according to the level hierarchy of anchors already assigned to that area, resulting in a more compact map. Finally, we develop a dynamic keyframe window to alleviate the false local minima and forgetting issue to improve mapping quality. Through extensive experiments, we empirically demonstrate that our method achieves superior mapping results with approximately a 5dB enhancement in PSNR on realworld scenes [4, 27], with a more compact model, while maintaining a more compact model and operating at high processing speed.

To summarize, our contributions are as follows:

â¢ We introduce a novel framework that naturally integrates the sparse octree and structured 3D Gaussian representation to perform efficient and detailed online dense mapping.

â¢ We design an anchor-based progressive map refinement strategy for better scene coverage and mapping quality.

â¢ We develop a dynamic keyframe window to mitigate the false local minima and forgetting issue.

## 2. Related Work

Classical RGB-D online dense mapping. For online 3D reconstruction of scenes, various explicit representations have been employed to store scene information, including point clouds [7], surface elements(surfels) [1, 37] and truncated signed distance functions(TSDF) [3, 44]. Several works [17, 31] leverage deep learning to improve the accuracy and robustness of above mentioned representations, and even achieve dense reconstruction with monocular input [32, 39]. These methods are renowned for their rapid processing speed, which is attributable to the inherent physical properties of explicit representations. However, they demand significant memory resources to manage high-detail mapping [45], and are incapable of realistically rendering from novel views.

NeRF-based RGB-D online dense mapping. Following the significant success of neural radiance fields (NeRF) [25], several studies leveraged latent features and neural networks as implicit representations to integrated NeRF with RGB-D dense mapping. iMAP [30] presents the first NeRF-style online dense mapping, using Multi-Layer Perceptrons (MLP) as the scene representation. NICE-SLAM [46] represents scenes as hierarchical feature grids, utilizing pre-trained MLPs for decoding. To enhance mapping speed and expand representational capacity, various representations have been investigated, including multi-resolution hash grids [12, 13, 34], factored grids [14], sparse octree grids [40] and neural point clouds [28]. Nevertheless, the aforementioned methods struggle to achieve fine-detailed rendering results and maintain fast rendering speed. These limitations arise from their reliance on time-consuming volumetric rendering techniques. In contrast, our approach leverages fast rasterization, enabling complete use of perpixel dense photometric errors.

Gaussian-based RGB-D online dense mapping. The high-fidelity and rapid rasterization capabilities of 3D Gaussian Splatting (3DGS) [16] facilitate superior quality and efficiency in scene reconstruction. Recently, many works [11, 15, 24, 42] have attempted to apply 3DGS in online dense mapping. SplaTAM [15] adopts an explicit volumetric approach using isotropic Gaussians, enabling precise map densification. However, this method necessitates projecting and densifying each pixel in the depth image to perform gaussian densification, resulting in substantial map storage usage. MonoGS [24] randomly select a subset of pixels for projection, which necessitates more optimization time for map densification. Concurrent to our work, CG-SLAM [10] introduces a depth uncertainty model to select more valuable Gaussian primitives during optimization. RTG-SLAM [27] treats each opaque Gaussian as an ellipsoid disc on the dominant plane of Gaussian to maintain a compact representation. NGM-SLAM [18] and Gaussian-SLAM [42] focuse on building submap for 3D Gaussian represention. More recently MG-SLAM [21] leverages Manhattan World hypothesis and additional semantic informations to refine and complete scene geometries. Different from these methods, our approach is based on the structured 3D Gaussian representation [23] and utilizes the structured information of the octree to guide the distribution of Gaussian kernels, resulting in better robustness to depth noise and smaller map size.

<!-- image-->  
Figure 2. Overview of OG-Mapping. Given a set of sequential RGB-D frames and camera poses, we utilize an octree-based structured 3D Gaussians as the scene representation to perform efficient online dense mapping. When a new keyframe is detected, we employ a sparse octree to swiftly capture the rough structure of the new observed region to guide anchor densification (Sec. 3.1) . During the map update process, we perform anchor-based progressive map refinement to enhance the geometry and appearance quality (Sec. 3.2), and construct a dynamic keyframe window to effectively mitigate false local minima and forgetting issues (Sec. 3.3).

## 3. Method

The overview of OG-Mapping is shown in Fig. 2. Taking RGB-D images from sensors and poses from other tracking modules, we utilize a structured 3D Gaussians representation [23] managed by a sparse octree(Sec. 3.1) to depict the scene geometry and appearance. During mapping process, we employ an anchor-based progressive map refinement strategy to enhance map reconstruction quality(Sec. 3.2). In Sec. 3.3, we introduce how to build our dynamic keyframe window to mitigate the fasle local minima and forgetting problem. Sec. 3.4 elaborate the online optimization details.

## 3.1. Structured Gaussian Representation for Online Mapping

Scene Representation. We represent the underlying map of the scene as a set of anchors [23]. Specifically, we first voxelize the scene using the provided camera pose and depth image. For each voxel, the center v is treated as an anchor point, equipped with a level mark $l _ { v } \in  { \mathbb { N } } _ { 0 } .$ , a scaling factor $s _ { v } \in \mathbb { R } ^ { 3 }$ , n learnable offsets $\{ o _ { v } ^ { i } \} _ { i = 1 } ^ { n } \in \mathbb { R } ^ { n \times 3 }$ and a feature vector $f _ { v } = \mathbf { e n c o d i n g } ( v )$ (In our dense version, the encoding function is the multi-hash encoding [26], while in the sparse version, it represents local context encoding. Implementation details are in supplementary).

For each visible anchor within the viewing frustum, n 3D Gaussians are generated. The positions $\{ \bar { \mu _ { v } ^ { i } } \} _ { i = 1 } ^ { n }$ of these 3D Gaussians are calculated as:

$$
\left\{ \mu _ { v } ^ { i } \right\} _ { i = 1 } ^ { n } = p _ { v } + s _ { v } \cdot \left\{ o _ { v } ^ { i } \right\} _ { i = 1 } ^ { n }\tag{1}
$$

The other attributes (color $c _ { v } ^ { i } \in \mathbb { R } ^ { 3 }$ , opacity $\alpha _ { v } ^ { i } \in \mathbb { R } _ { > \mathcal { Y } }$ quaternions $q _ { v } ^ { i } \in \mathbb { R } ^ { 4 }$ , and scale $\varphi _ { v } ^ { i } \in \mathbb { R } ^ { 3 } )$ of n 3D Gaussians are decode from the relative distance $\delta _ { c v }$ , view direction $\mathbf { d } _ { c v }$ , and the anchor feature $f _ { v }$ using individual multilayer perceptron(MLP) decoders, denoted as $F _ { c o l o r } .$ $F _ { o p a c i t y }$ and $F _ { c o v }$ :

$$
\begin{array} { r l } & { \left\{ c _ { v } ^ { i } \right\} _ { i = 1 } ^ { n } = \mathbf { F } _ { c o l o r } \big ( \delta _ { c v } , \mathbf { d } _ { c v } , f _ { v } \big ) , } \\ & { \left\{ \alpha _ { v } ^ { i } \right\} _ { i = 1 } ^ { n } = \mathbf { F } _ { o p a c i t y } \big ( \delta _ { c v } , \mathbf { d } _ { c v } , f _ { v } \big ) , } \\ & { \left\{ q _ { v } ^ { i } , \varphi _ { v } ^ { i } \right\} _ { i = 1 } ^ { n } = \mathbf { F } _ { c o v } \big ( \delta _ { c v } , \mathbf { d } _ { c v } , f _ { v } \big ) , } \end{array}\tag{2}
$$

where the relative distance $\delta _ { c v }$ and the viewing direction $\mathbf { d } _ { c v }$ between camera position $p _ { c }$ and anchor point position $p _ { v }$ are calculated as follows :

$$
\delta _ { c v } = \| p _ { c } - p _ { v } \| _ { 2 } , \quad \mathbf { d } _ { c v } = \frac { p _ { c } - p _ { v } } { \delta _ { c v } } .\tag{3}
$$

Then, these generated N 3D Gaussians are used for fast rasterization rendering to produce color and depth maps. Given a viewpoint, the rendered color of each pixel p can be written as:

$$
C ( \mathbf { p } ) = \sum _ { i \in \mathcal { N } } c ^ { i } \sigma ^ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma ^ { j } ) , \sigma ^ { i } = \alpha ^ { i } G _ { 2 d } ^ { i } ( \mathbf { p } ) ,\tag{4}
$$

where the 2D Gaussians $G _ { 2 d } ( \mathbf { p } )$ are transformed from 3D Gaussian $G ( \mathbf { p } )$ introduced by [16]. Similarly, per-pixel depth is rendered via alpha-blending:

$$
{ \cal D } ( { \bf p } ) = \sum _ { i \in \mathcal { N } } z ^ { i } \sigma ^ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma ^ { j } ) ,\tag{5}
$$

where $z ^ { i }$ is the distance of the $i ^ { t h }$ 3D Gaussianâs position $u ^ { i }$ along the camera ray.

Map Densification. To achieve a comprehensive representation of the environment, we need to add new anchors to the scene during online scanning to cover newly observed regions. The adaptive map densification approach in [15, 27], which is based on pixel-level projection error, is sensitive to depth noise and is unreliable when handling edges. Upon receiving a new keyframe, we dynamically allocate new voxels using the provided pose and depth image, incrementally updating a sparse octree to roughly encompass all visible regions.

## 3.2. Anchor-based Progressive Map Refinement

A progressive optimization strategy can better shape the loss landscape, reducing the risk of the algorithm becoming trapped in misleading local minima. Such a strategy has been successfully applied in various computer vision applications like registration [20] and surface reconstruction [19]. OG-Mapping also utilizes a coarse-to-fine optimization scheme to reconstruct the scene with progressive levels of detail, by adaptively adding finer-level anchors to underoptimized regions.

To ensure system efficiency, the sparse octree only maintains the coarsest granularity of anchors to quickly reveal scene structure information. The 3D Gaussians inferred from these newly added anchors have a large scale, which makes it difficult to fit areas with high-frequency texture changes. Therefore, anchors with finer granularity need to be added to these under-optimized areas. Scaffold-GS [23] employs an error-based anchor growth method, which allows the 3D Gaussians of fine-level anchors to participate in the growth of coarser ones. Nevertheless, during online reconstruction, it fails to provide sufficient optimization iterations to stabilize the gradients of such fine-level 3D Gaussians, resulting in the generation of numerous unnecessary anchors.

<!-- image-->  
Figure 3. Anchor-based Progressive Map Refinement. We grow new anchors in under-optimized regions based on the level of 3D Gaussians and their gradients. If the regions already contain anchors at the same level, the granularity of these new anchors will be increased.

To address the aforementioned issues, we develop an anchor-based progressive map refinement strategy as illustrated in Fig. 3. Specifically, for each level l, we predefine the corresponding voxel size $\gamma _ { l }$ and significance threshold $\tau _ { l } .$ . To determine under-optimized areas, the gradients of 3D Gaussians belonging to anchor v are denoted as $\{ \bigtriangledown _ { v } ^ { i } \} _ { i = 1 } ^ { n } .$ Subsequently, the 3D Gaussian satisfying $\nabla _ { v } ^ { i } \geqslant \tau _ { l _ { \tau } }$ marks its respective region as candidate region. A new anchor $v ^ { \prime }$ with level $l _ { v ^ { \prime } } = l _ { v }$ is deployed at the position $u _ { v } ^ { i }$ . If the candidate area already has anchor with level ${ { l } _ { { { v } ^ { \prime } } } }$ , indicating that the current level of detail is still insufficient to meet the optimization requirements, the granularity of anchor $v ^ { \prime }$ will be increased. Specifically, the 3D Gaussians that have already participated in growth will not be used for further anchor growth. In practice, the variations of voxel size and significant threshold between different levels are defined as:

$$
\gamma _ { l + 1 } = \gamma _ { l } / 4 , \quad \tau _ { l + 1 } = \tau _ { l } * 2 .\tag{6}
$$

This coarse-to-fine strategy allows us to effectively suppress the expansion of anchors caused by gradient instability.

## 3.3. Dynamic Keyframe Window Construction

Keyframe Insert Selection. The keyframe set is constructed based on the overlap between the visible coarsest anchors of the current frame and the last keyframe. A new input RGB-D frame is added to the keyframe set if the ratio $N _ { o v e r l a p } / N _ { u n i o n }$ smaller than a threshold, where $N _ { o v e r l a p }$ is the number of coarsest granularity anchors visible in both the current frame and the last keyframe, and $N _ { u n i o n }$ represents the total number of visible coarsest anchors of them. By this insertion strategy, we ensure the views in the keyframe set have relatively little overlap.

Dynamic Keyframe Window. When a new keyframe is added, it typically indicates that new unexplored areas require optimization. Relying solely on the data from this single keyframe for map optimization can lead to severe forgetting and overfitting issues, resulting in poor final reconstruction quality. Existing online mapping methods usually select a specific number of keyframes from the keyframe set, along with the newly added keyframe(some works additionally add the most recent keyframe), to form a fixed keyframe window. Only the keyframes within this fixed window are used for map optimization. However, these methods face

<!-- image-->  
Figure 4. Illustration of the differences between Fixed Keyframe Window and our Dynamic Keyframe Window. Left: Fixed Keyframe Window maintains a static window content across all optimization iterations. Right: our dynamic Keyframe Window updates the keyframe window with new data each optimization iteration.

challenges in balance optimization iterations with mapping quality. Suppose the new keyframe $K _ { n e w }$ observes a new region with distribution Î¶. Since the information for Î¶ is only observed by $K _ { n e w } ,$ it is necessary to optimize $K _ { n e w }$ with k times to accurately fit the new region. Let a represent the number of keyframes in current window, the total number of inference operations required is $k \times ( a { + } 1 )$ . A diminutive b is prone to causing significant overfitting and forgetting problem, which hampers the modelâs performance on previously visited area. Conversely, optimize $K _ { n e w }$ with a larger b can lead to an excessively high number of inference operations, potentially causing the model to become computationally intensive and time-consuming.

Further considering this issue, it becomes apparent that only the $K _ { n e w }$ necessitates multiple optimization iterations.

The role of the other keyframes is to alleviate local overfitting and forgetting issues. Thus, having diverse sources for these keyframes is advantageous. Based on the above analysis, instead of using a fixed keyframe window, we develop a simple yet effective dynamic keyframe window. Specifically, we first calculate the overlap weight Ï between the existing keyframes in the keyframe set KF and the $K _ { n e w }$ KF is subsequently divided into two sets with threshold $\varrho \colon$ the local set $( \{ K _ { i } | K _ { i } \in \mathbf { K } \mathbf { F } , \varsigma _ { i } \geq \varrho \} )$ and global set $( \{ K _ { i } | K _ { i } \in \mathbf { K } \mathbf { F } , \varsigma _ { i } < \varrho \} )$ . During each new optimization iteration, we clear the keyframe window of all keyframes except $K _ { n e w }$ . Then, we select $b _ { 1 }$ keyframes from the local set and $b _ { 2 }$ keyframes from the global set without replacement and add them to the keyframe window $( a = b _ { 1 } + b _ { 2 } )$ . By employing the aforementioned method, we ensure that the contents of the keyframe window are always dynamically changing. This approach enables effective adaptation to newly observed regions while incorporating past experiences, thereby addressing the limitations of the fixed keyframe window method. A more intuitive comparison between the two methods can be found in Fig. 4.

## 3.4. Losses Design

The learnable parameters and MLPs are optimized with respect to the L1 loss over rendered pixel colors , denoted as $\mathcal { L } _ { c } ,$ and depths, denoted as $\mathcal { L } _ { d }$ . The loss function is further augmented by SSIM term [8] $\mathcal { L } _ { S S I M }$ and scale term [22] $\mathcal { L } _ { s } \mathrm { : }$

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { c } \mathcal { L } _ { c } + \lambda _ { S S I M } \mathcal { L } _ { S S I M } + \lambda _ { d } \mathcal { L } _ { d } + \lambda _ { s } \mathcal { L } _ { s } . } \end{array}\tag{7}
$$

where the Gaussians scale term $\mathcal { L } _ { s }$ is:

$$
\mathcal { L } _ { s } = \sum _ { i = 1 } ^ { N } P r o d ( \varphi ^ { i } ) .\tag{8}
$$

The $P r o d ( . )$ is the product computation function and $\mathcal { N }$ is the the number of all 3D Gaussians.

## 4. Experiment

## 4.1. Experimental Setup

Dataset. To evaluate the performance of our proposed method, we compare its mapping accuracy and time consumption with other RGB-D mapping systems currently open-source on both the synthetic Replica dataset [29] and the real-world ScanNet dataset [4] in addition to a selfscanned scene provided by [27].

Metrics. Following the evaluation protocol of mapping results used in SplaTAM [15], for measuring RGB rendering performance, we report standard photometric rendering quality metrics:PSNR [36], SSIM [8] and LPIPS(AlexNet) [43]. Depth rendering performance is measured by Depth L1 loss(cm). All rendering metrics are computed on each frame with valid pose to evaluate the map quality. We report the average across five runs for all our evaluations.

Baseline Methods. We select several advanced Gaussianbased dense RGB-D SLAM methods currently opensource, MonoGS [24], SplaTAM [15] and RTG-SLAM [27] for comparison. We also benchmark our method against other RGB-D based approaches [12â14, 28] that, like ours, do not have explicit loop closure and submap.

Implementation Details. All methods are benchmarked on a desktop computer with an intel i7-14700K CPU and an Nvidia RTX 4090 GPU. As we exclusively focus on incremental mapping, we omit the tracking component of [14, 15, 24, 28] and instead employ the ground truth pose. All methods are evaluated using their single-threaded versions. Specifically, the scene refinement module used after online mapping process in MonoGS [24] was removed to ensure a fair comparison. An anchor is pruned if all the opacity of its 3D Gaussians is less than $\rho = 0 . 0 1$ . To provide a more comprehensive comparison with other methods, in addition to the full version(marked as âOursâ), we also offer the âOurs-sparseâ version (using a larger Ï to prune more anchors) and the âOurs-sparseÃâ version (based on the last version but using only half the number of iterations for mapping). More details of hyperparameters are provided in the supplementary material.

<table><tr><td>Methods</td><td>Metrics</td><td>R0</td><td>R1</td><td>R2</td><td>O0</td><td>O1</td><td>O2</td><td>O3</td><td>O4</td><td>Avg</td></tr><tr><td rowspan="3">H2-Mapping [12] (RAL23)</td><td>PSNR â</td><td>29.67</td><td></td><td></td><td>32.34 32.23 37.87</td><td>38.93</td><td></td><td>31.02 30.64 33.07 33.22</td><td></td><td></td></tr><tr><td>SSIM â</td><td>0.927</td><td></td><td>0.956 0.963</td><td>0.982</td><td>0.984</td><td>0.958</td><td>0.95</td><td></td><td>0.968 0.932</td></tr><tr><td>LPIPS â</td><td>0.217</td><td>0.168</td><td>0.151</td><td>0.09</td><td>0.0945</td><td>0.203</td><td>0.221</td><td>0.166</td><td>0.164</td></tr><tr><td rowspan="2">ESLAM [14] (CVPR23)</td><td>PSNR â SSIM â</td><td>26.40</td><td>28.34 0.874 0.923</td><td>30.25</td><td>35.10</td><td>34.76</td><td>29.07</td><td>28.83</td><td>31.15</td><td>30.49</td></tr><tr><td>LPIPS â</td><td>0.763 0.300</td><td>0.292</td><td>0.232</td><td>0.928 0.180</td><td>0.921 0.205</td><td>0.879</td><td></td><td>0.8760.908</td><td>0.871</td></tr><tr><td rowspan="3">Point-SLAM [28] (ICCV23)</td><td>PSNR â</td><td>34.38</td><td>35.05</td><td>36.80</td><td>39.35</td><td>40.29</td><td>0.235 34.95</td><td>0.191 34.54</td><td>0.199 35.66</td><td>0.230 36.38</td></tr><tr><td>SSIM â</td><td>0.937</td><td>0.942</td><td>0.955</td><td>0.962</td><td>0.960</td><td>0.916</td><td>0.917</td><td>0.942</td><td></td></tr><tr><td>LPIPS â</td><td></td><td></td><td>0.093 0.106 0.102 0.088</td><td></td><td>0.103</td><td>0.147</td><td>0.116 0.129</td><td></td><td>0.941 0.111</td></tr><tr><td rowspan="2">H3-Mapping [13] (arXiv24)</td><td>PSNR â</td><td>33.16</td><td>34.99</td><td>35.24</td><td>39.85</td><td>40.12</td><td></td><td>33.89 34.10 35.99 35.92</td><td></td><td></td></tr><tr><td>SSIM â LPIPS</td><td>0.921</td><td></td><td></td><td>0.939 0.948 0.969</td><td>0.966</td><td></td><td>0.936 0.933 0.950 0.945</td><td></td><td></td></tr><tr><td rowspan="2">MonoGS [24] (CVPR24)</td><td>PSNR â</td><td>â</td><td></td><td></td><td>33.16 34.99 35.24 39.85 â40.12</td><td></td><td>34.59 34.3236.50 35.68</td><td></td><td></td><td></td></tr><tr><td>SSIM â</td><td></td><td></td><td></td><td>0.928 0.928 0.942 0.972</td><td>0.968</td><td></td><td>0.941 0.939 0.952 0.946</td><td></td><td></td></tr><tr><td rowspan="3">SplaTAM [15] (CVPR24)</td><td>LPIPS PSNR â</td><td>â 0.128</td><td>0.153</td><td>0.129</td><td>0.085</td><td>0.084</td><td>0.129</td><td>0.107</td><td>0.129</td><td>0.118</td></tr><tr><td></td><td>33.57</td><td></td><td></td><td>34.40 36.36 40.37</td><td>40.51</td><td></td><td>33.38 32.56 35.07</td><td></td><td>35.78</td></tr><tr><td>SSIMâ LPIPS â</td><td>0.978</td><td>0.973</td><td>0.984</td><td>0.985</td><td>0.980</td><td>0.974</td><td></td><td>0.968 0.970</td><td>0.976</td></tr><tr><td rowspan="3">RTG-SLAM [27] (SIGGRAPH24)</td><td>PSNR â</td><td>0.052 31.30</td><td>0.064 33.95</td><td>0.053 34.95</td><td>0.049</td><td>0.067 39.63</td><td>0.073 32.90</td><td>0.081 32.86 36.12</td><td>0.106</td><td>0.068</td></tr><tr><td>SSIM â</td><td></td><td></td><td>0.983</td><td>39.28</td><td></td><td></td><td></td><td></td><td>35.12</td></tr><tr><td>LPIPSâ</td><td>0.967</td><td>0.979</td><td></td><td>0.988</td><td>0.990</td><td>0.981</td><td>0.982</td><td>0.985</td><td>0.982</td></tr><tr><td rowspan="3">Ours</td><td>PSNR â</td><td>0.144</td><td>0.111</td><td>0.116 38.73</td><td>0.080 42.07</td><td>0.097 42.26</td><td>0.140</td><td>36.15 36.34 38.58 38.56</td><td>0.133 0.118</td><td>0.117</td></tr><tr><td>SSIM â</td><td>36.24</td><td>38.13 0.974</td><td></td><td>0.976 0.986</td><td>0.981</td><td></td><td>0.973 0.971 0.975 0.976</td><td></td><td></td></tr><tr><td>LPIPS â</td><td>0.971</td><td></td><td></td><td>0.044 0.049 0.055 0.036</td><td>0.049</td><td></td><td>0.054 0.047 0.050 0.048</td><td></td><td></td></tr></table>

Table 1. Quantitative comparison of our method against baselines for rendering results on the Replica [29] dataset.

## 4.2. Experiments Results.

Evaluation on Replica [29]. In Tab. 1, we evaluate our methodâs rendering quality results on Replica dataset [29]. Our approach achieves the best PSNR and LPIPS results, while maintaining a highly competitive SSIM result. Fig. 5 provides a qualitative comparison of the rendering of ours and baseline method. Thanks to our hierarchical representation optimization method, our approach is able to better capture details. Tab. 4 provides the performance analysis of MonoGS [24], SplaTAM [15], RTG-SLAM [27] and our method. Since we donât need to explicitly store all 3D Gaussian parameters, our method is more storage-efficient. Despite using fewer iterations and a more sparse distribution of 3D Gaussians, our approach still achieves superior mapping results.

<table><tr><td>Methods</td><td>Metrics 0000</td><td>0059</td><td>0106</td><td>0169</td><td>0181</td><td>0207</td><td>Avg</td></tr><tr><td rowspan="3">H2-Mapping [12] (RAL23)</td><td>PSNR â 21.02</td><td>17.60</td><td>15.59</td><td>20.65</td><td>18.83</td><td>21.04</td><td>19.12</td></tr><tr><td>SSIMâ 0.809</td><td>0.765</td><td>0.742</td><td>0.819</td><td>0.844</td><td>0.822</td><td>0.800</td></tr><tr><td>LPIPS â</td><td>0.475 0.445</td><td>0.522</td><td>0.419</td><td>0.492</td><td>0.472</td><td>0.471</td></tr><tr><td>ESLAM [14] (CVPR23)</td><td>PSNR â SSIMâ LPIPSâ</td><td>19.10 17.84 0.636 0.630</td><td>16.92 0.616</td><td>20.42 0.690</td><td>17.60 0.708</td><td>19.03 0.664</td><td>18.64 0.657</td></tr><tr><td rowspan="3">Point-SLAM [28] (ICCV23)</td><td>PSNR â</td><td>0.560 0.520</td><td>0.590</td><td>0.542</td><td>0.566</td><td>0.591</td><td>0.561</td></tr><tr><td>24.09</td><td>22.14</td><td>21.33</td><td>22.74</td><td>22.29</td><td>24.36</td><td>22.83</td></tr><tr><td>SSIMâ</td><td>0.715 0.683</td><td></td><td>0.619 0.630</td><td>0.746</td><td>0.717</td><td>0.685</td></tr><tr><td rowspan="3">MonoGS [24] (CVPR24)</td><td>LPIPS â SR</td><td>0.471 0.480</td><td></td><td>0.554 0.560</td><td>0.520</td><td>0.512</td><td>0.516</td></tr><tr><td></td><td>17.2015.44</td><td></td><td>16.9218.79</td><td></td><td>13.7618.10</td><td>16.72</td></tr><tr><td>SSIMâ LPIPS â 0.560</td><td>0.636 0.574</td><td>0.657</td><td>0.704</td><td>0.594</td><td>0.705</td><td>0.645</td></tr><tr><td rowspan="3">SplaTAM [15] (CVPR24)</td><td>PSNR â</td><td>0.628 19.70 19.65</td><td>0.571 19.11</td><td>0.553</td><td>0.723</td><td>0.542</td><td>0.596</td></tr><tr><td>SSIMâ 0.654</td><td></td><td></td><td>23.23</td><td>18.58</td><td>20.64</td><td>20.15</td></tr><tr><td>LPIPS â 0.422</td><td>0.807 0.240</td><td>0.747 0.315</td><td>0.787</td><td>0.749</td><td>0.749</td><td>0.749</td></tr><tr><td rowspan="3">RTG-SLAM [27] (SIGGRAPH24)</td><td>PSNR â</td><td>18.98 17.44</td><td>16.87</td><td>0.276 19.39</td><td>0.351 17.51</td><td>0.278 18.97</td><td>0.313 18.19</td></tr><tr><td>SSIMâ 0.804</td><td>0.705</td><td>0.684</td><td>0.778</td><td></td><td></td><td></td></tr><tr><td>LPIPS â 0.488</td><td>0.534</td><td>0.579</td><td>0.499</td><td>0.716</td><td>0.757</td><td>0.741</td></tr><tr><td rowspan="3">Ours</td><td>PSNR â</td><td></td><td></td><td></td><td>0.613</td><td>0.538 26.13</td><td>0.542</td></tr><tr><td>SSIMâ 0.807</td><td>25.87 24.04 0.836</td><td>24.86 0.864</td><td>26.84 0.838</td><td>25.91 0.873</td><td></td><td>25.61</td></tr><tr><td>LPIPS â 0.345</td><td>0.270</td><td>0.271</td><td>0.285</td><td>0.322</td><td>0.819 0.340</td><td>0.840 0.306</td></tr></table>

Table 2. Quantitative comparison of our method against baselines for rendering results on the ScanNet [4] dataset.

<table><tr><td>Methods</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Depth L1(cm)â</td><td>Rendering FPSâ</td><td>Model Size(MB)â</td></tr><tr><td>MonoGS [24]</td><td>25.31</td><td>0.808</td><td>0.543</td><td>3.94</td><td>862.0</td><td>5.6</td></tr><tr><td>SplaTAM [15]</td><td>22.99</td><td>0.723</td><td>0.378</td><td>1.60</td><td>54.9</td><td>540.3</td></tr><tr><td>RTG-SLAM [27]</td><td>24.40</td><td>0.823</td><td>0.415</td><td>2.72</td><td>316.5</td><td>270.8</td></tr><tr><td>Ours-sparse</td><td>28.91</td><td>0.846</td><td>0.423</td><td>1.10</td><td>647.8</td><td>5.4</td></tr><tr><td>Ours</td><td>30.32</td><td>0.874</td><td>0.300</td><td>0.89</td><td>559.7</td><td>30.5</td></tr></table>

Table 3. Quantitative comparison in terms of rendering, Depth L1, and memory performance on the real scanned scene provided by [27].

Evaluation on Real-world Scene. Since the camera trajectory provided by the ScanNet [4] originates from BundleFusion [5] rather than ground truth, and the depth images contain significant noise, conducting online mapping on Scan-Net is extremely challenging. Tab. 2 presents the rendering evaluation results on six scenes of ScanNet. Due to the lack of structured organization, previous Gaussian-based methods [15, 24, 27] are highly sensitive to depth noise and camera trajectory inaccuracies. In contrast, our method effectively filters out these noises, resulting in more realistic rendering outcomes. In Tab. 5, we compare the performance of OG-Mapping with other Gaussian-based methods. The results demonstrate that our method can construct more compact maps at a faster processing speed. Fig. 1 and Fig. 6 also illustrate that our method can achieve better visual quality with more reliable geometry and texture details. The rendering viewpoints of these results are selected outside of the training views, with random perturbations added to the poses. Since the depth information provided by ScanNet [4] contains excessive noise, the Depth-L1 method used in SplaTAM [15] is not suitable for measuring geometric accuracy. In Tab. 3, we provide additional experimental results on the high-quality large indoor scene [27] to further validate the effectiveness of our method in real-world scenarios.

Office_3  
<!-- image-->  
Figure 5. Qualitative comparison of rendering results across three scenes from the Replica dataset. Key details are highlighted by colored boxes. The average PSNR metric for each scene is indicated in the lower-left corner.

<table><tr><td>Methods</td><td>PSNR â</td><td>Depth L1 [cm] â</td><td>Processing FPSâ</td><td>Model Size[MB]â</td></tr><tr><td>MonoGS [24]</td><td>35.68</td><td>2.12</td><td>1.2</td><td>9.7</td></tr><tr><td>SplaTAM [15]</td><td>35.78</td><td>0.61</td><td>0.4</td><td>275.1</td></tr><tr><td>RTG-SLAM [27]</td><td>35.12</td><td>1.69</td><td>15.6</td><td>71.5</td></tr><tr><td>Ours-sparse</td><td>36.75</td><td>0.72</td><td>16.1</td><td>8.9</td></tr><tr><td>Ours-sparse</td><td>37.00</td><td>0.69</td><td>8.5</td><td>8.8</td></tr><tr><td>Ours</td><td>38.56</td><td>0.46</td><td>5.6</td><td>34.6</td></tr></table>

Table 4. Comparison of key performance metrics on the Replica [29] dataset between ours and baseline Gaussian-based methods [15, 24, 27].

<table><tr><td>Methods</td><td>PSNR â</td><td>Processing FPSâ</td><td>Model Size[MB]â</td></tr><tr><td>MonoGS [24]</td><td>16.72</td><td>2.9</td><td>5.6</td></tr><tr><td>SplaTAM [15]</td><td>19.99</td><td>1.8</td><td>160.7</td></tr><tr><td>RTG-SLAM [27]</td><td>18.19</td><td>11.4</td><td>153.2</td></tr><tr><td>Ours-sparse</td><td>23.87</td><td>12.5</td><td>3.1</td></tr><tr><td>Ours</td><td>25.61</td><td>10.1</td><td>36.1</td></tr></table>

Table 5. Comparison of key performance metrics on ScanNet [4] between ours and baseline Gaussian-based methods [15, 24, 27].
<table><tr><td>Methods</td><td>PSNRâ</td><td>Processing FPSâ</td><td>Model Size(MB)â</td></tr><tr><td>Projection-based [15]</td><td>36.18</td><td>2.95</td><td>71.8</td></tr><tr><td>Naive Unique</td><td>36.24</td><td>3.37</td><td>39.9</td></tr><tr><td>Octree-based(ours)</td><td>36.24</td><td>5.16</td><td>39.9</td></tr></table>

Table 6. Ablation study of map densification methods quantitative results on room0 [29].

<table><tr><td>Dataset</td><td>Methods</td><td>PSNRâ</td><td>Model Size(MB)â</td></tr><tr><td rowspan="2">Replica</td><td>w/o Growing</td><td>36.85</td><td>29.4</td></tr><tr><td>Error-based [23] Anchor-based(ours)</td><td>38.54 38.56</td><td>40.5</td></tr><tr><td rowspan="2">ScanNet</td><td>w/o Growing</td><td>24.89</td><td>34.6 19.9</td></tr><tr><td>Error-based [23] Anchor-based(ours)</td><td>25.27 25.61</td><td>78.4 36.1</td></tr></table>

Table 7. Ablation study of our progressive map refinement strategy on Replica [29] dataset and ScanNet [4] dataset.

<table><tr><td>Type</td><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td rowspan="3">Fixed Window</td><td>Random [14]</td><td>37.14</td><td>0.969</td><td>0.057</td></tr><tr><td>Overlap [15, 24]</td><td>35.85</td><td>0.963</td><td>0.063</td></tr><tr><td>Coverage- Maximizing [12]</td><td>37.75</td><td>0.972</td><td>0.050</td></tr><tr><td>Dynamic Window</td><td>Global(ours)</td><td>38.56</td><td>0.976</td><td>0.048</td></tr></table>

Table 8. Ablation study of our dynamic keyframe window methods on Replica [29] dataset.

<!-- image-->  
Figure 6. Qualitative comparison of rendering results across three scenes from Scannet. The average rendering FPS are indicated at the bottom of the images.

Map Densification Ablations. To prove the effectiveness of our octree-based structured map densification method, we compare the mapping performance of Depth error-based method [15], Naive Unique method (Using the unique function to determine whether the anchors are newly observed), and our octree-based method on the room0 of Replica. As shown in Tab. 6, The projection-based method inevitably adds extra anchors at object edges incorrectly, resulting in larger map size and worse rendering results. While the Naive Unique method can handle anchor growth correctly like ours, the inefficient unique computation results in an intolerable time overhead. Our octree-based method leverages the efficiency of sparse octrees, combining both processing accuracy and high efficiency.

Progressive Map Refinement Ablations. We evaluated our progressive map refinement described in Sec. 3.2. Tab. 7 shows the results of disabling growing operation and employing the error-based method [23] on the Replica and the ScanNet datasets. The results show the growing operation is crucial for accurately reconstructing details. Compared to the error-based method [23], our anchor-based method makes better use of structured information, resulting in smaller map occupancy while maintaining comparable rendering accuracy.

Keyframe Window Ablations. In Tab. 8, we compared our dynamic keyframe window method with other existing keyframe window construction approaches. The overlap method, which only uses keyframes that overlap with the newly added keyframe, is not suitable for addressing the problem of forgetting. Consequently, it results in the poorest rendering quality. The Coverage-Maximizing method [12] uses the most recently unoptimized keyframes to construct the keyframe window. However, the reconstruction process tends to overfit these areas as this window is fixed. In contrast, our dynamic keyframe window overcomes these issues, achieving optimal results.

## 5. Conclusion

In this work, we introduce OG-Mapping, a novel framework for effective online dense mapping. By utilizing an octree-based structured 3D Gaussians representation, OG-Mapping achieves efficient map densification and compaction. Additionally, we propose an anchor-based progressive map refinement strategy to to enhance the capture of finer details.. Furthermore, we develop a dynamic keyframe window to mitigate the issues of local overfitting and forgetting problems encountered during the reconstruction process. Experiment results demonstrate that this approach, leveraging a more compact map, outperforms existing algorithms. The advantages of our structural representation and dynamic keyframe window are particularly evident in challenging real scenes where existing Gaussian-based online mapping methods typically falter.

Limitation. OG-Mapping relies on inputs from an RGB-D sensor to build a sparse octree. Future research is anticipated to explore the use of monocular image input alone. Our method employs MLPs as feature decoders, successfully constructing compact maps. However, in particularly large scenes, it may still encounter forgetting issues. Further research on submap construction [18, 42] could further reduce frame processing time.

## References

[1] Yan-Pei Cao, Leif Kobbelt, and Shi-Min Hu. Real-time highaccuracy three-dimensional reconstruction with consumer rgb-d cameras. ACM Transactions on Graphics (TOG), 37 (5):1â16, 2018. 1, 2

[2] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16123â16133, 2022. 1

[3] Jiawen Chen, Dennis Bautembach, and Shahram Izadi. Scalable real-time volumetric surface reconstruction. ACM Trans. Graph., 32(4):113â1, 2013. 1, 2

[4] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828â5839, 2017. 2, 5, 6, 7, 8

[5] Angela Dai, Matthias NieÃner, Michael Zollhofer, Shahram Â¨ Izadi, and Christian Theobalt. Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration. ACM Transactions on Graphics (ToG), 36(4): 1, 2017. 6

[6] Tianchen Deng, Guole Shen, Tong Qin, Jianyu Wang, Wentao Zhao, Jingchuan Wang, Danwei Wang, and Weidong Chen. Plgslam: Progressive neural scene represenation with local to global bundle adjustment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19657â19666, 2024. 1

[7] Hao Du, Peter Henry, Xiaofeng Ren, Marvin Cheng, Dan B Goldman, Steven M Seitz, and Dieter Fox. Interactive 3d modeling of indoor environments with a consumer depth camera. In Proceedings of the 13th international conference on Ubiquitous computing, pages 75â84, 2011. 1, 2

[8] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366â2369. IEEE, 2010. 5

[9] Armin Hornung, Kai M Wurm, Maren Bennewitz, Cyrill Stachniss, and Wolfram Burgard. Octomap: An efficient probabilistic 3d mapping framework based on octrees. Autonomous robots, 34:189â206, 2013. 1

[10] Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li, Liangjing Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. Cg-slam: Efficient dense rgb-d slam in a consistent uncertainty-aware 3d gaussian field. arXiv preprint arXiv:2403.16095, 2024. 2

[11] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. Photo-slam: Real-time simultaneous localization and photorealistic mapping for monocular, stereo, and rgb-d cameras. arXiv preprint arXiv:2311.16728, 2023. 2

[12] Chenxing Jiang, Hanwen Zhang, Peize Liu, Zehuan Yu, Hui Cheng, Boyu Zhou, and Shaojie Shen. H {2}-mapping: Real-time dense mapping using hierarchical hybrid representation. IEEE Robotics and Automation Letters, 2023. 1, 2, 6, 8

[13] Chenxing Jiang, Yiming Luo, Boyu Zhou, and Shaojie Shen. H3-mapping: Quasi-heterogeneous feature grids for realtime dense mapping using hierarchical hybrid representation. arXiv preprint arXiv:2403.10821, 2024. 2, 6

[14] Mohammad Mahdi Johari, Camilla Carta, and FrancÂ¸ois Fleuret. Eslam: Efficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17408â17419, 2023. 1, 2, 6, 8

[15] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2, 4, 5, 6, 7, 8

[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4):1â14, 2023. 1, 2, 4

[17] Lukas Koestler, Nan Yang, Niclas Zeller, and Daniel Cremers. Tandem: Tracking and dense mapping in real-time using deep multi-view stereo. In Conference on Robot Learning, pages 34â45. PMLR, 2022. 2

[18] Mingrui Li, Jingwei Huang, Lei Sun, Aaron Xuxiang Tian, Tianchen Deng, and Hongyu Wang. Ngm-slam: Gaussian splatting slam with radiance field submap. arXiv preprint arXiv:2405.05702, 2024. 2, 8

[19] Zhaoshuo Li, Thomas Muller, Alex Evans, Russell H Tay- Â¨ lor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. Neuralangelo: High-fidelity neural surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8456â8465, 2023. 4

[20] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5741â5751, 2021. 4

[21] Shuhong Liu, Heng Zhou, Liuzhuozheng Li, Yun Liu, Tianchen Deng, Yiming Zhou, and Mingrui Li. Structure gaussian slam with manhattan world hypothesis. arXiv preprint arXiv:2405.20031, 2024. 2

[22] Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. Mixture of volumetric primitives for efficient neural rendering. ACM Transactions on Graphics (ToG), 40(4):1â13, 2021. 5

[23] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 1, 2, 3, 4, 8

[24] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian Splatting SLAM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2, 6, 7, 8

[25] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[26] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1â15, 2022. 1, 3

[27] Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time 3d reconstruction at scale using gaussian splatting. arXiv preprint arXiv:2404.19706, 2024. 2, 4, 5, 6, 7

[28] Erik Sandstrom, Yue Li, Luc Van Gool, and Martin R Os-Â¨ wald. Point-slam: Dense neural point cloud-based slam. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18433â18444, 2023. 2, 6

[29] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019. 5, 6, 7, 8

[30] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6229â6238, 2021. 1, 2

[31] Keisuke Tateno, Federico Tombari, Iro Laina, and Nassir Navab. Cnn-slam: Real-time dense monocular slam with learned depth prediction. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6243â6252, 2017. 2

[32] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558â16569, 2021. 2

[33] Emanuele Vespa, Nikolay Nikolov, Marius Grimm, Luigi Nardi, Paul HJ Kelly, and Stefan Leutenegger. Efficient octree-based volumetric slam supporting signed-distance and occupancy mapping. IEEE Robotics and Automation Letters, 3(2):1144â1151, 2018. 2

[34] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. In Proceedings of the IEEE international conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 2

[35] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021. 1

[36] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 5

[37] Thomas Whelan, Stefan Leutenegger, Renato F Salas-Moreno, Ben Glocker, and Andrew J Davison. Elasticfusion: Dense slam without a pose graph. In Robotics: science and systems, page 3. Rome, Italy, 2015. 1, 2

[38] Chi Yan, Delin Qu, Dong Wang, Dan Xu, Zhigang Wang, Bin Zhao, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting. 2024. 2

[39] Nan Yang, Lukas von Stumberg, Rui Wang, and Daniel Cremers. D3vo: Deep depth, deep pose and deep uncertainty for monocular visual odometry. In Proceedings of

the IEEE/CVF conference on computer vision and pattern recognition, pages 1281â1292, 2020. 2

[40] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages 499â507. IEEE, 2022. 2

[41] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19447â19456, 2024. 1

[42] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Oswald. Gaussian-slam: Photo-realistic dense slam with gaussian splatting, 2023. 2, 8

[43] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 5

[44] Yizhong Zhang, Weiwei Xu, Yiying Tong, and Kun Zhou. Online structure analysis for real-time indoor scene reconstruction. ACM Transactions on Graphics (TOG), 34(5):1â 13, 2015. 1, 2

[45] Xingguang Zhong, Yue Pan, Jens Behley, and Cyrill Stachniss. Shine-mapping: Large-scale 3d mapping using sparse hierarchical implicit neural representations. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 8371â8377. IEEE, 2023. 2

[46] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12786â12796, 2022. 1, 2