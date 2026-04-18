# CaRtGS: Computational Alignment for Real-Time Gaussian Splatting SLAM

Dapeng Feng, Zhiqiang Chen, Yizhen Yin, Shipeng Zhong, Yuhua Qi, and Hongbo Chen

AbstractâSimultaneous Localization and Mapping (SLAM) is pivotal in robotics, with photorealistic scene reconstruction emerging as a key challenge. To address this, we introduce Computational Alignment for Real-Time Gaussian Splatting SLAM (CaRtGS), a novel method enhancing the efficiency and quality of photorealistic scene reconstruction in real-time environments. Leveraging 3D Gaussian Splatting (3DGS), CaRtGS achieves superior rendering quality and processing speed, which is crucial for scene photorealistic reconstruction. Our approach tackles computational misalignment in Gaussian Splatting SLAM (GS-SLAM) through an adaptive strategy that enhances optimization iterations, addresses long-tail optimization, and refines densification. Experiments on Replica, TUM-RGBD, and VECtor datasets demonstrate CaRtGSâs effectiveness in achieving highfidelity rendering with fewer Gaussian primitives. This work propels SLAM towards real-time, photorealistic dense rendering, significantly advancing photorealistic scene representation. For the benefit of the research community, we release the code and accompanying videos on our project website: https://dapengfeng. github.io/cartgs.

Index TermsâMapping, Gaussian Splatting SLAM, SLAM.

## I. INTRODUCTION

IMULTANEOUS Localization and Mapping (SLAM) is a cornerstone of robotics and has been a subject of extensive research over the past few decades [1]â[5]. The rapid evolution of applications such as autonomous driving, virtual and augmented reality, and embodied intelligence has introduced new challenges that extend beyond the traditional scope of real-time tracking and mapping. Among these challenges is the need for photorealistic scene reconstruction, which necessitates precise spatial understanding coupled with high-fidelity visual representation.

In response to these challenges, recent research has explored the use of implicit volumetric scene representations, notably Neural Radiance Fields (NeRF) [6]. While promising, integrating NeRF into SLAM systems has encountered several obstacles, including high computational demands, lengthy optimization times, limited generalizability, an over-reliance on visual cues, and a susceptibility to catastrophic forgetting [7].

In a significant breakthrough, a novel explicit scene representation method utilizing 3D Gaussian Splatting (3DGS) [8] has emerged as a potent solution. This method not only rivals the rendering quality of NeRF but also excels in processing speed, offering an order-of-magnitude improvement in both rendering and optimization tasks.

The advantages of this representation make it a strong candidate for incorporation into online SLAM systems that require real-time performance. It has the potential to transform the field by enabling photorealistic dense SLAM, thereby expanding the horizons of scene understanding and representation in dynamic environments.

However, existing Gaussian Splatting SLAM (GS-SLAM) methods [9]â[17] struggle to achieve superior rendering performance under real-time constraints when dealing with a limited number of Gaussian primitives. These issues stem from the misalignment between the computational demands of the algorithm and the available processing resources, which can lead to insufficient optimization and optimization processes. Addressing these challenges is crucial for enhancing the performance and applicability of GS-SLAM in real-time environments.

In this paper, we scrutinize the computational misalignment phenomenon and propose the Computational Alignment for Real-Time Gaussian Splatting SLAM (CaRtGS) to address these challenges. Our approach aims to optimize the computational efficiency of GS-SLAM, ensuring that it can meet the demands of real-time applications while achieving high rendering quality with fewer Gaussian primitives.

Our contributions are listed as follows:

â¢ We provide an analysis of the computational misalignment phenomenon present in GS-SLAM.

â¢ We introduce an adaptive computational alignment strategy that effectively tackles insufficient optimization, longtail optimization, and weak-constrained densification, achieving high-fidelity rendering with fewer Gaussian primitives under real-time constraints.

â¢ We conduct comprehensive experiments and ablation studies to demonstrate the effectiveness of our proposed method on three popular datasets with three distinct camera types.

## II. RELATED WORKS

GS-SLAM leverages the benefits of 3DGS [8] to achieve enhanced performance in terms of rendering speed and photo-

<!-- image-->  
Figure 1: Performance on TUM-RGBD. We provide a comparison of most of the available open-source GS-SLAM methods.

realism. In this section, we conduct a concise review of both 3D Gaussian Splatting and Gaussian Splatting SLAM.

## A. 3D Gaussian Splatting

3DGS [8] is a cutting-edge real-time photorealistic rendering technology that employs differentiable rasterization, eschewing traditional volume rendering methods. This groundbreaking method represents the scene as explicit Gaussian primitivies and enables highly efficient rendering, achieving a remarkable 1080p resolution at 130 frames per second (FPS) on contemporary GPUs, and has substantially spurred research advancements.

In response to the burgeoning interest in 3DGS, a variety of extensions have been developed with alacrity. Accelerating the acquisition of 3DGS scene representations is a key area of focus, with various strategies being explored. One prominent research direction is the reduction of Gaussians through the refinement of densification heuristics [18]â[20]. Moreover, optimizing runtime performance has become a priority, with several initiatives concentrating on enhancing the differentiable rasterizer and optimizer implementations [20]â[23].

Motivated by these advancements, our work addresses the challenge of insufficient optimization in photorealistic rendering within real-time SLAM by utilizing splat-wise backpropagation [20]. In parallel, recent methodologies have concentrated on sparse-view reconstruction and have sought to compact the scene representation. This is achieved by training a neural network to serve as a data-driven prior, which is capable of directly outputting Gaussians in a single forward pass [24]â[27]. In contrast, our research zeroes in on real-time dense-view and per-scene visual SLAM. This targeted focus demands an incremental photorealistic rendering output that is tailored to the unique characteristics of each scene.

## B. Gaussian Splatting SLAM

3DGS [8] has also quickly gained attention in the SLAM literature, owing to its rapid rendering capabilities and explicit scene representation. MonoGS [9] and SplaTAM [10] are seminal contributions to the coupled GS-SLAM algorithms, pioneering a methodology that simultaneously refines Gaussian primitives and camera pose estimates through gradient backpropagation. Gaussian-SLAM [11] introduces the concept of sub-maps to address the issue of catastrophic forgetting. Furthermore, LoopSplat [12], which extends the work of Gaussian-SLAM [11], employs a Gaussian splatbased registration for loop closure to enhance pose estimation accuracy. However, the reliance on the intensive computations of 3DGS [8] for estimating the camera pose of each frame presents challenges for these methods in achieving real-time performance.

To overcome this, decoupled GS-SLAM methods have been proposed [13]â[17]. Splat-SLAM [13] and IG-SLAM [14] utilize pre-trained dense bundle adjustment [1] for camera pose tracking and proxy depth maps for map optimization. RTG-SLAM [15] incorporates frame-to-model ICP for tracking and renders depth by focusing on the most prominent opaque Gaussians. GS-ICP-SLAM [16] achieve remarkably high speeds (up to 107 FPS) by leveraging the shared covariances between G-ICP [2] and 3DGS [8], with scale alignment of Gaussian primitives. Photo-SLAM [17] employs ORB-SLAM3 [3] for tracking and introduces a coarse-to-fine map optimization for robust performance.

These methods achieve state-of-the-art PSNR with a large number of Gaussian primitives, as presented in Figure 1, which will limit the application of real-time GS-SLAM in large-scale scenarios due to increased computational demands. In this paper, we delve into the limitations of existing GS-SLAM and propose an innovative computational alignment technique to enhance PSNR while reducing the number of Gaussian primitives required, all within the constraints of realtime SLAM operations.

## III. METHODS

In this section, we delve into the photorealistic rendering aspect of GS-SLAM. Initially, we scrutinize the computational misalignment phenomenon inherent to GS-SLAM. This misalignment can significantly impair computational efficiency and hinder the swift convergence of photorealistic rendering, adversely affecting the performance of real-time GS-SLAM. To overcome these obstacles, we propose a novel adaptive computational alignment strategy. This strategy aims to accelerate the 3DGS process, optimize computational resource allocation, and efficiently control model complexity, thereby enhancing the overall effectiveness and practicality of 3DGS in real-time SLAM applications.

## A. Computational Misalignment

The computational misalignment encountered in photorealistic rendering within the context of SLAM can be attributed to three primary aspects: insufficient optimization, long-tail optimization, and weak-constrained densification, which reduces rendering quality and increases map size. These factors significantly hinder the real-time applications of GS-SLAM, limiting its applicability in resource-constrained devices.

<!-- image-->  
Figure 2: The Effect of Adaptive Optimization on Replica. Dashed lines depict performance without adaptive optimization, while solid lines show results with it. Blue represents keyframe iterations, and red indicates PSNR. The horizontal line marks average PSNR and iterations. Our method significantly improves low-PSNR keyframe processing through enhanced iterative optimization, as evident from the trend comparison between dashed and solid lines.

1) Insufficient Optimization: In contrast to typical 3DGS [8], which is not constrained by real-time considerations, online rendering within the realm of SLAM necessitates the concurrent execution of localization, mapping, and rendering at a speed that is synchronized with the frequency of incoming sensor data. To achieve this, the majority of current realtime GS-SLAM methods [15]â[17] rely on keyframes for both mapping and rendering. However, these methods typically achieve only a few thousand iterations in rendering optimization in total, which significantly lags behind the tens of thousands of iterations achieved by 3DGS [8]. Due to insufficient optimization, the optimization process has not fully converged, adversely affecting the quality of online rendering.

Recent observations by several researchers indicate that pixel-wise backpropagation in 3DGS presents a significant computational challenge [20], [21]. This process becomes a bottleneck due to the contention among multiple GPU threads for access to shared Gaussian primitives, which necessitates serialized atomic operations, thereby limiting parallelization efficiency. Unfortunately, this drawback is integrated into the previous implementations of GS-SLAM [15]â[17]. In this paper, we utilize a fast splat-wise backpropagation [20] to reduce thread contention. This approach not only achieves a $3 \times$ increase in the number of iterations compared to the baseline [17], but also maintains the same runtime. This advancement significantly mitigates the problem of insufficient optimization, substantially improving the rendering quality of real-time GS-SLAM.

2) Long-Tail Optimization: To mitigate the issue of catastrophic forgetting, a common approach in GS-SLAM is to randomly select a keyframe from the keyframe pool for periodic reoptimization [15]â[17]. However, this method can result in suboptimal long-tail optimization, which overfits the oldest keyframe and underfits the newest one, as depicted in Figure 2. Specifically, the reoptimization frequency of the earliest keyframes tends to exceed that of the most recently added ones. This disparity arises because the keyframe pool is continuously expanded as the camera moves through the environment, which can result in an uneven distribution of reoptimization efforts and and a declining trend in the PSNR for newly incoming keyframes.

In this paper, we propose an innovative adaptive optimization strategy that selects reoptimization keyframes from the pool based on their optimization loss to counteract longtail effect. By employing this approach, we aim to increase the reoptimization frequency of keyframes with lower PSNR values. This targeted approach has been demonstrated to significantly enhance the rendering quality, as evidenced by an improvement from 34.9 dB to 36.4 dB in the Replica Room2 scenario, as depicted in Figure 2. By doing so, our adaptive strategy ensures a more equitable distribution of reoptimization efforts across the keyframe pool, optimizing each keyframeâs contribution to the systemâs overall performance. This innovative approach not only improves the quality of the rendered output but also enhances the efficiency and effectiveness of the reoptimization process.

3) Weak-constrained Densification: Densification is a critical component of photorealistic rendering in the context of GS-SLAM, encompassing both geometry densification and adaptive densification [9]â[17]. Geometric densification involves the conversion of a color point cloud into initialized Gaussian primitives for each newly identified keyframe, providing a foundational geometric structure for the environment. Adaptive densification, on the other hand, refines the Gaussian primitives using operations such as splitting and cloning, which are guided by gradients and the size of the primitives themselves [8]. These densifications are solely constrained by a simplistic pruning strategy that eliminates Gaussian primitives with low opacity. However, emerging research [25]â [27] suggests that this approach is insufficient for managing the modelâs size within an optimal range. In this paper, we introduce an opacity regularization loss to encourage the Gaussian primitives to learn a low opacity, thereby not only facilitating the pruning process to eliminate less significant primitives but also preserving high-fidelity rendering.

## B. System Overview

As delineated in Figure 3, we take the modular designs, which are easy to integrate into existing real-time decoupled GS-SLAM, e.g., GS-ICP-SLAM [16] and Photo-SLAM [17]. Given a sequence of observations $\{ \gamma _ { 1 } , . . . , \gamma _ { N } \}$ , we employ a state-of-the-art front-end tracker [2], [3], which estimates the 6-DoF pose for each frame and identifies keyframes $\{ v _ { 1 } , . . . , v _ { k } \}$ based on criteria related to translation and rotation. Once a keyframe $v _ { i }$ is identified, the frontend tracker transforms the corresponding observation $\nu _ { i }$ into the global coordinate system and integrates it into the global Point Cloud ${ \mathcal { P } } .$

In the photorealistic rendering phase, we utilize 3DGS [8] as the backend render. Firstly, we convert $\mathcal { P }$ into a set of Gaussian primitives ${ \mathcal { G } } .$ Each primitive is characterized by its posistion $ { \mathbf { p } } \in \mathbb { R } ^ { 3 }$ , orientation represented as quaternion $\mathbf { q } \in$ $\mathbb { R } ^ { 4 }$ , scaling factor $\textbf { s } \in \mathbb { R } ^ { 3 }$ , opacity $\sigma \in \mathbb { R } ^ { 1 }$ , and spherical harmonic coefficients $\mathbf { S H } \in \mathbb { R } ^ { 4 8 }$ . By employing Î±âblending rendering [8], we achieve the high-fidelity rendering $\hat { \mathcal { T } }$ for a selected keyframe $v _ { i } { \cdot }$

<!-- image-->  
Figure 3: The overview of CaRtGS. We adopt a real-time cutting-edge SLAM system as a front-end tracker, severing for localization and geometry mapping. In the photorealistic rendering back-end, we apply the proposed adaptive computational alignment strategy to enhance the 3DGS optimization process, including fast splat backward, adaptive optimization, and opacity regularization.

$$
\hat { \mathcal { T } } = \sum _ { k \in \mathcal { G } } c _ { k } \alpha _ { k } \prod _ { j = 1 } ^ { k - 1 } \left( 1 - \alpha _ { k } \right) ,\tag{1}
$$

where $c _ { k }$ denotes the color derived from $\mathbf { S H } , \alpha _ { k }$ is determined by evaluating a projected 2D Gaussian multipied with the learned opacity $\sigma _ { k }$ . To refine the Gaussian primitives G, we take both $\mathcal { L } _ { 1 }$ and Structural Similarity Index (SSIM) Loss $\mathcal { L } _ { s s i m }$ to supervise the optimization process. These losses are crucial for enhancing the quality of our photorealistic renderings. Additionally, we incorporate opacity regularization into our comprehensive loss function to control the model size, which is detailed in Sec. III-C3.

## C. Adaptive Computational Alignment

To address the computational misalignment of photorealistic rendering in real-time GS-SLAM, we propose an adaptive computational alignment strategy termed CaRtGS. Below, we outline the key steps of this strategy in detail.

1) Fast Splat-wise Backpropagation: In the conventional 3DGS optimization pipeline, the backpropagation phase is computationally demanding as it entails the propagation of gradient information from pixels to Gaussian primitives. This process necessitates the calculation of gradients for each splatpixel pair $( i , j )$ , followed by an aggregation step. In our notation, i denotes the index of the i-th splat, and j denotes the index of the $j \mathrm { - t h }$ pixel. To parallelize the execution, we assign thread i to process the i-th splat, and thread j to process the j-th pixel. In the forward pass, GPU thread $i + 1$ applies the standard Î±-blending logic to transition from the received state $\mathcal { X } _ { i , j }$ to $\mathcal X _ { i + 1 , j }$ , integrating this updated information into the gradient computation. In the backward pass, the gradients associated with the i-th splat, denoted as $\nabla { \mathcal { X } } _ { i } .$ , are accumulated across the pixels that are influenced by this splat. This process can be mathematically represented as:

$$
\mathcal { X } _ { i + 1 , j } = \mathcal { F } ( \mathcal { X } _ { i , j } ) ,
$$

$$
\nabla \mathcal { X } _ { i , j } = \nabla \mathcal { F } \cdot \nabla \mathcal { X } _ { i + 1 , j } ,\tag{2}
$$

(3)

$$
\nabla { \mathcal X } _ { i } = \sum _ { j } \nabla { \mathcal X } _ { i , j } ,\tag{4}
$$

where $\mathcal { F }$ presents the Î±-blending function.

Pixel-wise propagation is widely used in GS-SLAM [9]â [17], mapping threads to pixels and processing splats in reverse depth order. Thread j computes partial gradients for the splats in the order they are blended, updating the cumulative gradient for each splat through atomic operations. However, this method can lead to contention among threads for shared memory access, resulting in serialized operations that impede performance.

To address this challenge, we utilize a novel parallelization strategy [20] that shifts the focus from pixel-based to splatbased processing. This strategy allows each thread to independently maintain the state of a splat and to efficiently exchange pixel state information. Thread i can compute the gradient contribution for the i-th splat, requiring the pixel j state after the first i splats have been blended.

During the forward pass, threads archive transmittance T and accumulated color RGB for pixels every N splats, preparing for the backward pass. These stored states include initial conditions $\mathcal { X } _ { 0 , j } , \mathcal { X } _ { N , j } , \cdot \cdot \cdot \forall j$ . At the commencement of the backward pass, each thread in a tile generates the pixel state $\mathcal { X } _ { i , j }$ . Threads then engage in rapid collaborative sharing to exchange pixel states.

For further details, please refer to Figure 4a. The data presented in Figure 4b clearly show that the splat-wise backpropagation method significantly enhances the total number of optimization iterations by a factor of 3, increasing from an average of 4.6k to 15.4k. This improvement effectively addresses the issue of insufficient optimization compared to Photo-SLAM [17] equipped with pixel-wise propagation.

2) Adaptive Optimization: Although splat-wise propagation achieves sufficient optimization in total, the long-tail distribution of iterations per keyframe is a challenge. To address this, we recommend augmenting the splat-wise approach with an adaptive optimization based on traininig loss L to ensure a more equitable distribution of iterations across the keyframe pool K.

<!-- image-->  
(a) Gradient Backpropagation

<!-- image-->  
Figure 4: The Effect of Different Gradient Backpropagation. (a) The original 3DGS employs pixel-wise parallelism for backpropagation, which is prone to frequent contentions, leading to slower backward passes. We introduce a splat-centric parallelism, where each thread handles one Gaussian splat at a time, significantly reducing contention. The gradient computation relies on a set of per-pixel, per-splat values, effectively traversing a splat â pixel relationship table. During the forward pass, we save pixel states for every $3 2 ^ { \mathrm { n d } }$ splat. For the backward pass, splats are grouped into buckets of 32, each processed by a CUDA warp. Warps utilize intra-warp shuffling to efficiently construct their segment of the state table. (b) We provide a comparison of total iteration on Replica with monocular camera.

Given a keyframe pool $\kappa _ { k }$ containing keyframes $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { k } \}$ , we maintain two sets: $\mathcal { R } _ { k } = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { k } \}$ which tracks the remaining optimization iterations for each keyframe, and $\mathcal { L } _ { k } = \{ l _ { 1 } , l _ { 2 } , \ldots , l _ { k } \}$ which records the last optimization loss value for each keyframe. Upon the detection of a new keyframe $v _ { k + 1 }$ , we update our pools as follows:

$$
\begin{array} { r } { K _ { k + 1 } = K _ { k } \cup \left\{ v _ { k + 1 } \right\} , } \end{array}\tag{5}
$$

$$
\begin{array} { r } { \mathcal { R } _ { k + 1 } = \mathcal { R } _ { k } \cup \left\{ r _ { k + 1 } ^ { 0 } \right\} , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { \mathcal { L } _ { k + 1 } = \mathcal { L } _ { k } \cup \left\{ l _ { k + 1 } \right\} , } \end{array}\tag{7}
$$

where $r _ { k + 1 } ^ { 0 }$ is the initial optimization iteration count assigned to the new keyframe, and $l _ { k + 1 }$ is its initial optimization loss value. We then select a keyframe $v ^ { \prime }$ randomly from the subset of keyframes with remaining iterations, defined as $\{ v _ { i } | r _ { i } > 0 , \forall r _ { i } \in \mathcal { R } _ { k } \}$ , to train the 3D Gaussians Map G. Postoptimization, we decrement the optimization iteration count for the selected keyframe by one, adjusting $r ^ { \prime }$ to $r ^ { \prime } - 1$ , and also update the corresponding optimization loss value lâ².

When $\{ v _ { i } | r _ { i } > 0 , \forall r _ { i } \in \mathcal { R } _ { k } \}$ is empty, we update $\mathcal { R } _ { k }$ based on $\mathcal { L } _ { k }$ as follows:

$$
r _ { i } = \left\{ { \begin{array} { l l } { 1 } & { l _ { i } \notin { \prod _ { i } ^ { d _ { k } } ( \mathcal { L } _ { k } ) } , } \\ { 2 } & { l _ { i } \in { \prod _ { i } ^ { d _ { k } } ( \mathcal { L } _ { k } ) } , } \end{array} } \right.\tag{8}
$$

where $\prod ( \cdot )$ donates top $d _ { k }$ largest elements, $d _ { k } = \operatorname* { m a x } ( 1 , \frac { k } { d } )$ and d is a hyperparameter. This method prioritizes keyframes with higher optimization loss values for the photorealistic rendering module, effectively tackling the long-tail optimization as demonstrated in Figure 2.

3) Opacity Regularization: In the typical application of 3DGS, the rendered loss $\mathcal { L } _ { r e n d e r e d }$ is utilized to refine the 3D Gaussian primitives [8]. To efficiently manage memory usage and model size, we have devised a strategy that encourages the elimination of Gaussians in areas where they do not contribute to the rendering process. Since the presence of a Gaussian is primarily indicated by its opacity $^ { O , }$ we impose a regularization term $\mathcal { L } _ { o }$ on this attribute. The complete formulation of our optimization loss $\mathcal { L }$ is as follows:

$$
\mathcal { L } _ { r e n d e r e d } = ( 1 - \lambda _ { s s i m } ) \mathcal { L } _ { 1 } + \lambda _ { s s i m } \mathcal { L } _ { s s i m } ,\tag{9}
$$

$$
\mathcal { L } _ { o } = \frac { 1 } { N } \sum _ { i } | o _ { i } | ,\tag{10}
$$

$$
\mathcal { L } = \mathcal { L } _ { \mathit { r e n d e r e d } } + \lambda _ { o } \mathcal { L } _ { o } ,\tag{11}
$$

where $\lambda _ { s s i m }$ is the weighting factor, $\lambda _ { o }$ is the regularization coefficient, and N denotes the total count of Gaussian primitives.

## IV. EXPERIMENTS

In this section, we present a comparative analysis of CaRtGS against state-of-the-art GS-SLAM systems [9]â[11], [16], [17] and Loopy-SLAM [28], a state-of-the-art NeRFbased SLAM system. This evaluation spans multiple scenarios, including those captured using monocular, RGB-D, and stereo cameras. Furthermore, we perform an ablation study to substantiate the efficacy of the novel techniques introduced in our approach.

## A. Setup

Dataset. We conducted evaluations on three distinct camera systems: monocular, RGB-D, and stereo. These assessments were carried out on three renowned datasets: Replica [29], TUM-RGBD [30], and VECtor [31]. Replica [29] is a highquality reconstruction dataset at room and building scale, including high-resolution high-dynamic-range (HDR) textures. TUM-RGBD [30] is a well-known RGB-D dataset that contains color and depth images captured by a Microsoft Kinect sensor, along with the ground-truth trajectory obtained from a high-accuracy motion-capture system. VECtor [31] is a SLAM benchmark dataset that covers the full spectrum of motion dynamics, environmental complexities, and illumination conditions. To ensure data consistency, we employed a soft time synchronization to align the sensor data and ground truth with a precision of $\Delta t = 0 . 0 8 s$

Table I: Quantitative Results on Replica.
<table><tr><td>Cam</td><td>Method</td><td>Metric</td><td>office0</td><td>office1</td><td>office2</td><td>office3</td><td>office4</td><td>room0</td><td>room1</td><td>room2</td></tr><tr><td rowspan="6">Moooorr</td><td></td><td>ATE FPS</td><td>0.20 Â± 0.02 36.91 Â± 0.75</td><td>2.95 Â± 6.23 36.41 Â± 0.66</td><td>0.91 Â± 0.39 34.48 Â± 0.52</td><td>0.11 Â± 0.01 34.60 Â± 0.36</td><td>0.17 Â± 0.00 35.98 Â± 0.49</td><td>0.15 Â± 0.00 34.40 Â± 0.29</td><td>0.24 Â± 0.04 36.37 Â± 0.66</td><td>0.10 Â± 0.02 33.32 Â± 0.28</td></tr><tr><td>Photo-SLAM [17]</td><td>IPF</td><td>2.66 Â± 0.11</td><td>2.31 Â± 0.05</td><td>2.30 Â± 0.06</td><td>2.29 Â± 0.05</td><td>2.30 Â± 0.04</td><td>2.03 Â± 0.02</td><td>2.22 Â± 0.02</td><td>2.30 Â± 0.08</td></tr><tr><td></td><td>PSNR</td><td>35.02 Â± 0.45</td><td>32.75 Â± 5.37</td><td>31.19 Â± 0.65</td><td>31.13 Â± 0.53</td><td>32.94 Â± 0.18</td><td>28.74 Â± 0.39</td><td>30.56 Â± 0.38</td><td>31.69 Â± 0.25</td></tr><tr><td></td><td></td><td></td><td>97.04k Â± 31.16k</td><td></td><td></td><td>75.98k Â± 3.39k</td><td></td><td></td><td></td></tr><tr><td></td><td>Points</td><td>78.40k Â± 2.94k</td><td></td><td>99.40k Â± 1.67k</td><td>76.36k Â± 3.19k</td><td></td><td>0.11m Â± 6.27k</td><td>0.12m Â± 5.56k</td><td>81.10k Â± 1.81k</td></tr><tr><td>Ours (Photo-SLAM)</td><td>ATE FPS</td><td>0.22 Â± 0.06 36.65 Â± 0.46</td><td>2.97 Â± 6.24 36.08 Â± 0.47</td><td>1.53 Â± 1.37 33.90 Â± 00.28</td><td>0.12 Â± 0.01 34.88 Â± 0.68</td><td>0.17 Â± 0.01 35.96 Â± 0.54</td><td>0.17 Â± 0.00 33.58 Â± 0.20</td><td>0.52 Â± 0.48 36.65 Â± 0.29</td><td>0.09 Â± 0.00 33.73 Â± 0.26</td></tr><tr><td rowspan="6">Photo-SLAM [17]</td><td></td><td>8.10 Â± 0.21</td><td>7.76 Â± 0.23</td><td></td><td></td><td></td><td>7.40 Â± 0.11</td><td></td><td></td></tr><tr><td>IPF PSNR</td><td></td><td></td><td>8.05 Â± 0.13</td><td>7.15 Â± 0.09</td><td>7.35 Â± 0.13</td><td></td><td>7.33 Â± 0.24</td><td>7.67 Â± 0.04</td></tr><tr><td>Points</td><td>34.58 Â± 0.31</td><td>34.97 Â± 4.96</td><td>33.52 Â± 0.12</td><td>33.26 Â± 0.08</td><td>35.22 Â± 0.23</td><td>31.92 Â± 0.26</td><td>31.99 Â± 1.15</td><td>34.39 Â± 0.16</td></tr><tr><td></td><td>38.32k Â± 1.97k</td><td>48.37k Â± 11.77k</td><td>64.07k Â± 1.03k</td><td>54.93k Â± 0.91k</td><td>53.67k Â± 1.13k</td><td>87.49k Â± 2.99k</td><td>73.44k Â± 2.84k</td><td>58.92k Â± 1.30k</td></tr><tr><td>ATE</td><td>0.45 Â± 0.05</td><td>0.35 Â± 0.04</td><td>1.13 Â± 0.14</td><td>0.37 Â± 0.02</td><td>0.44 Â± 0.05</td><td>0.30 Â± 0.02</td><td>0.33 Â± 0.04</td><td>0.18 Â± 0.00</td></tr><tr><td rowspan="10">RORD</td><td>FPS</td><td>31.61 Â± 0.53 3.43 Â± 0.09</td><td>31.96 Â± 0.32</td><td>30.43 Â± 0.81</td><td>29.33 Â± 0.52</td><td>27.87 Â± 0.54</td><td>27.49 Â± 0.52</td><td>29.87 Â± 0.91</td><td>27.37 Â± 0.52</td></tr><tr><td>IPF</td><td></td><td>3.04 Â± 0.12</td><td>3.18 Â± 0.04</td><td>3.28 Â± 0.04</td><td>3.10 Â± 0.05</td><td>3.17 Â± 0.05</td><td></td><td></td></tr><tr><td>PSNR</td><td>36.83 Â± 0.32</td><td>36.79 Â± 0.29</td><td>32.45 Â± 0.38</td><td>33.38 Â± 0.07</td><td></td><td></td><td>3.12 Â± 0.05</td><td>3.20 Â± 0.05</td></tr><tr><td>Points</td><td>81.34k Â± 2.95k</td><td>79.24k Â± 1.71k</td><td>0.12m Â± 4.04k</td><td>93.03k Â± 3.79k</td><td>35.13 Â± 0.39 0.12m Â± 1.61k</td><td>30.13 Â± 2.14 0.19m Â± 2.70k</td><td>33.80 Â± 0.36</td><td>34.53 Â± 0.87</td></tr><tr><td>ATE FPS</td><td>0.48 Â± 0.04</td><td>0.38 Â± 0.06</td><td>1.10 Â± 0.19</td><td>0.38 Â± 0.02</td><td>0.56 Â± 0.10</td><td>0.31 Â± 0.01</td><td>0.16m Â± 8.84k 0.34 Â± 0.03</td><td>0.14m Â± 2.09k 0.18 Â± 0.00</td></tr><tr><td>Ours (Photo-SLAM)</td><td>30.84 Â± 0.37</td><td>31.49 Â± 0.31</td><td>30.04 Â± 0.43</td><td>28.76 Â± 0.58</td><td>28.64 Â± 0.66</td><td>27.81 Â± 0.62</td><td>29.55 Â± 0.55</td><td>26.87 Â± 0.31</td></tr><tr><td>IPF</td><td>10.45 Â± 0.30</td><td>9.90 Â± 0.26</td><td>10.06 Â± 0.21</td><td>10.40 Â± 0.40</td><td>10.71 Â± 0.35</td><td>9.95 Â± 0.66</td><td>9.25 Â± 0.40</td><td>9.97 Â± 0.06</td></tr><tr><td>PSNR</td><td>35.54 Â± 0.28</td><td>37.74 Â± 0.41</td><td>33.40 Â± 0.29</td><td>33.84 Â± 0.27</td><td>35.64 Â± 0.41</td><td>29.38 Â± 3.70</td><td>34.30 Â± 0.64</td><td></td></tr><tr><td>Points</td><td>39.74k Â± 1.11k</td><td>54.61k Â± 2.58k</td><td>79.29k Â± 3.24k</td><td>68.03k Â± 2.06k</td><td>75.58k Â± 4.31k</td><td>0.11m Â± 3.74k</td><td>0.10m Â± 1.21k</td><td>36.54 Â± 0.19</td></tr><tr><td>ATE</td><td>0.19 Â± 0.00</td><td>0.13 Â± 0.00</td><td>0.18 Â± 0.00</td><td>0.19 Â± 0.01</td><td></td><td></td><td></td><td>0.10m Â± 2.72k 0.11 Â± 0.01</td></tr><tr><td rowspan="6">GS-ICP-SLAM [16] Ours</td><td>FPS</td><td>30.00 Â± 0.00</td><td>30.00 Â± 0.00</td><td></td><td></td><td>0.22 Â± 0.01</td><td>0.16 Â± 0.00</td><td>0.16 Â± 0.00</td><td></td></tr><tr><td></td><td>2.88 Â± 0.00</td><td></td><td>30.00 Â± 0.00</td><td>30.00 Â± 0.00</td><td>30.00 Â± 0.00</td><td>30.00 Â± 0.00</td><td>30.00 Â± 0.00</td><td>30.00 Â± 0.00</td></tr><tr><td>IPF</td><td>40.57 Â± 0.03</td><td>2.37 Â± 0.01 40.96 Â± 0.11</td><td>2.88 Â± 0.01</td><td>2.87 Â± 0.00</td><td>2.91 Â± 0.01</td><td>2.90 Â± 0.07</td><td>2.84 Â± 0.07</td><td>2.67 Â± 0.01</td></tr><tr><td>PSNR Points</td><td>1.57m Â± 0.85k</td><td>1.57m Â± 7.30k</td><td>32.77 Â± 0.16</td><td>31.60 Â± 0.07</td><td>38.84 Â± 0.04</td><td>35.54 Â± 0.06</td><td>37.81 Â± 0.06</td><td>38.54 Â± 0.05</td></tr><tr><td>ATE</td><td></td><td>0.12 Â± 0.00</td><td>1.54m Â± 2.51k</td><td>1.55m Â± 9.54k</td><td>1.60m Â± 10.33k</td><td>1.55m Â± 2.86k</td><td>1.55m Â± 0.70k</td><td>1.54m Â± 3.78k</td></tr><tr><td>FPS (GS-ICP-SLAM) IPF</td><td>0.25 Â± 0.14 30.00 Â± 0.00</td><td>30.00 Â± 0.00</td><td>0.28 Â± 0.10 30.00 Â± 0.00</td><td>0.19 Â± 0.02 30.00 Â± 0.00</td><td>0.24 Â± 0.01 30.00 Â± 0.00</td><td>0.16 Â± 0.00 30.00 Â± 0.00</td><td>0.16 Â± 0.00 30.00 Â± 0.00</td><td>0.11 Â± 0.00 30.00 Â± 0.00 11.68 Â± 0.08 10.46 Â± 0.05</td></tr></table>

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 5: Qualitative results on TUM-RGBD with RGBD Camera. Qualitative assessments demonstrate that our approach significantly improves rendering quality and effectively mitigates visual artifacts. Furthermore, our method achieves precise localization accuracy. In contrast, Gaussian-SLAM exhibits substantial drift, as indicated by the red dashed line.

Implementation Detail. All experimental evaluations were conducted on a desktop with an Nvidia RTX 4090 GPU, an AMD Ryzen 9 7950X CPU, and 128 GB RAM. We retained most of the original hyperparameters from the 3DGS [8]. However, we densify every 500 iterations with a positional gradients threshold $\tau _ { p } = 0 . 0 0 1$ and remove the transparent Gaussians with a threshold $\epsilon _ { \alpha } ~ = ~ 0 . 0 2$ . By default, we set $d = 4$ and $\lambda _ { o } = 0 . 0 0 1$ . On Replica, we use $r _ { k + 1 } ^ { 0 } = 8 .$ , whileas $r _ { k + 1 } ^ { 0 } = 2$ on TUM-RGBD and VECtor.

Evaluation. We performed all experiments 5 times to ensure statistical robustness and rendered original-resolution images for each estimated camera pose. To measure performance, we utilized the evo toolkit1 and the torchmetrics toolkit2. We recorded various performance indicators, including Absolute Trajectory Error (ATE) to assess the accuracy of localization,

Peak Signal-to-Noise Ratio (PSNR) to assess the quality of the photorealistic renderings, and the number of 3D Gaussian points to assess the model size. To assess the sufficiency of the Gaussian primitivesâ optimization, we introduced a metric known as Iterations Per Frame (IPF), defined as the ratio of total iterations to the total number of frames $\begin{array} { r } { ( \mathrm { I P F } = \frac { \mathrm { I t e r a t i o n s } } { \mathrm { F r a m e s } } ) } \end{array}$ . All performance indicators are reported in the format of mean Â± standard deviation.

## B. Results

The quantitative comparison presented in Table I, Table II, and Table III illustrates the performance of various methods. The best resutls of the PSNR and the count of Gaussian primitives are distinctively highlighted as $\mathbf { 1 ^ { s t } } , \ 2 ^ { \mathrm { n d } }$ , and $3 ^ { \mathrm { r d } }$ In summary, our approach consistently delivers superior rendering performance, utilizing a reduced number of Gaussian primitives, while adhering to real-time constraints of over 22 frames per second. Specifically, on the Replica dataset [29] with monocular camera, compared with Photo-SLAM [17], and under similar localization accuracy, our approach significantly improves the average PSNR by more than 2 dB and halves the number of Gaussian primitives. As shown in Table I and Table II, our method can be easily integrated into

Table II: Quantitative Results on TUM-RGBD.
<table><tr><td>Cam</td><td>Method</td><td>Metric</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td></tr><tr><td rowspan="18">Joor</td><td rowspan="4">MonoGS [9]</td><td>ATE FPS</td><td>4.93 Â± 0.16  $1 . 8 7 \pm 0 . 0 5$ </td><td>4.66 Â± 0.13  $3 . { \overset { \cdot } { 3 } } { \overset { \cdot } { 7 } } \pm { \overset { \cdot } { 0 } } . 0 { \overset { \cdot } { 6 } }$ </td><td>3.35 Â± 0.45 2.26 Â± 0.01</td></tr><tr><td>IPF</td><td> $8 4 . 0 7 \pm 0 . 2 5$ </td><td> $5 1 . 6 4 \pm 0 . 2 6$ </td><td>60.5 Â± 0.43</td></tr><tr><td>PSNR</td><td> $1 7 . 6 5 \pm 0 . 4 0$ </td><td> $1 5 . 5 6 \pm 0 . 0 2$ </td><td>19.35 Â± 0.31</td></tr><tr><td>Points</td><td>26.64k Â± 1.58k</td><td>43.59k Â± 2.09k</td><td>35.24k Â± 3.24k</td></tr><tr><td rowspan="4">Photo-SLAM [17]</td><td>ATE</td><td>1.55 Â± 0.06</td><td> $\overline { { 0 . 6 3 \pm 0 . 1 8 } }$ </td><td>1.10 Â± 0.70</td></tr><tr><td>FPS</td><td> $2 5 . 1 8 \pm 0 . 3 0$ </td><td> $2 5 . 8 3 \pm 0 . 1 2$ </td><td>24.74 Â± 0.25</td></tr><tr><td>IPF</td><td> $7 . 0 8 \pm 0 . 0 8$ </td><td> $6 . 6 6 \pm 0 . 0 8$ </td><td>7.77 Â± 0.20</td></tr><tr><td>PSNR</td><td></td><td></td><td></td></tr><tr><td rowspan="4">Ours (Photo-SLAM)</td><td>Points</td><td> $1 9 . 6 9 \pm 0 . 0 4$ </td><td> $2 0 . 1 9 \pm 0 . 5 2$  0.10m Â± 7.50k</td><td> $1 8 . 3 2 \pm 1 . 3 6$   $8 1 . 1 6 \mathbf { k } \pm 3 . 4 4 \mathbf { k }$ </td></tr><tr><td>ATE</td><td> $4 0 . 0 0 \mathrm { k } \pm 0 . 7 9 \mathrm { k }$   $\overline { { 1 . 5 5 \pm 0 . 0 6 } }$ </td><td> $\overline { { 0 . 7 0 \pm 0 . 0 8 } }$ </td><td> $\overline { { 0 . 5 7 \pm 0 . 3 3 } }$ </td></tr><tr><td>FPS</td><td> $2 4 . 9 5 \pm 0 . 4 6$ </td><td> $2 6 . 1 6 \pm 0 . 1 2$ </td><td></td></tr><tr><td>IPF</td><td>17.88 Â± 0.02</td><td></td><td> $2 5 . 0 3 \pm 0 . 1 1$ </td></tr><tr><td rowspan="6">Loopy-SLAM [28]</td><td>PSNR</td><td> $2 0 . 5 1 \pm 0 . 0 8$ </td><td>14.41 Â± 0.26  $2 1 . 5 4 \pm 0 . 8 5$ </td><td>16.06 Â± 0.32 19.38 Â± 1.47</td></tr><tr><td>Points</td><td> $3 8 . 6 5 \mathbf { k } \pm 1 . 8 2 \mathbf { k }$  â</td><td> $6 6 . 5 1 \mathbf { k } \pm 1 . 7 1 \mathbf { k }$ </td><td>51.71k Â± 3.46k</td></tr><tr><td>ATE</td><td> $\overline { { 3 . 9 3 \pm 1 . 1 3 } }$ </td><td> $\overline { { 1 . 4 3 \pm 0 . 1 6 } }$ </td><td>4.65 Â± 1.63</td></tr><tr><td>FPS</td><td> $0 . 2 3 \pm 0 . 0 0$ </td><td> $0 . 2 1 \pm 0 . 0 0$ </td><td>0.20 Â± 0.00</td></tr><tr><td>IPF</td><td></td><td></td><td></td></tr><tr><td>PSNR Points</td><td> $1 3 . 6 6 \pm 0 . 1 2$ </td><td> $1 7 . 9 5 \pm 0 . 4 1$ </td><td>17.43 Â± 0.15</td></tr><tr><td rowspan="6">Gaussian-SLAM [11]</td><td>ATE SplaTAM [10]</td><td> $\overline { { 2 . 5 1 \pm 0 . 0 1 } }$ </td><td> $\overline { { 0 . 5 0 \pm 0 . 0 0 } }$ </td><td>4.52 Â± 0.21</td></tr><tr><td>FPS</td><td> $0 . 2 7 \pm 0 . 0 1$ </td><td> $0 . 0 3 \pm 0 . 0 2$ </td><td> $0 . 2 5 \pm 0 . 0 0$ </td></tr><tr><td>IPF</td><td> $4 6 0 . 3 2 \pm 0 . 0 0$ </td><td> $4 6 0 . 8 8 \pm 0 . 0 0$ </td><td>460.84 Â± 0.00</td></tr><tr><td>PSNR</td><td> $2 1 . 0 3 \pm 0 . 1 0$ </td><td> $2 3 . 1 9 \pm 0 . 1 3$  Q</td><td> $2 0 . 1 0 \pm 0 . 0 5$ </td></tr><tr><td>Points</td><td> $0 . 9 6 \mathrm { m } \pm 3 . 9 6 \mathrm { k }$ </td><td> $6 . 3 6 \mathrm { { m } \pm 8 1 . 3 7 \mathrm { { k } } }$ </td><td>0.79m Â± 5.89k</td></tr><tr><td>ATE</td><td>2.74 Â± 0.11</td><td>0.96 Â± 0.44</td><td>8.42 Â± 1.19</td></tr><tr><td rowspan="6">MonoGS [9]</td><td>FPS</td><td> $0 . 5 7 \pm 0 . 0 6$ </td><td> $0 . 4 8 \pm 0 . 0 3$ </td><td></td></tr><tr><td>IPF</td><td></td><td></td><td>0.59 Â± 0.02</td></tr><tr><td></td><td> $3 0 9 . 3 7 \pm 4 . 2 9$ </td><td> $3 0 8 . 4 4 \pm 0 . 0 4$ </td><td> $3 1 0 . 6 6 \pm 0 . 1 1$ </td></tr><tr><td>PSNR</td><td> ${ \bf 2 3 . 7 1 \pm 0 . 1 0 }$ </td><td> ${ \bf 2 3 . 9 5 \pm 0 . 3 9 }$ </td><td>25.80 Â± 0.09</td></tr><tr><td>Points</td><td> $0 . 7 6 \mathrm { { m } \pm 1 2 . 1 2 \mathrm { { k } } }$ </td><td> $0 . 6 9 \mathrm { { m } \pm 2 6 . 0 7 \mathrm { { k } } }$ </td><td> $1 . 4 7 \mathrm { m } \pm 6 . 7 5 \mathrm { k }$ </td></tr><tr><td>ATE FPS</td><td>1.84 Â± 0.09</td><td>1.71 Â± 0.08</td><td>1.74 Â± 0.10</td></tr><tr><td rowspan="6">RCGP Photo-SLAM [17]</td><td></td><td> $2 . 1 8 \pm 0 . 0 2$ </td><td> $3 . 2 3 \pm 0 . 0 7$ </td><td>2.48 Â± 0.03</td></tr><tr><td>IPF</td><td> $7 7 . 7 7 \pm 0 . 0 6$ </td><td> $5 1 . 2 3 \pm 0 . 1 8$ </td><td>63.20 Â± 0.06</td></tr><tr><td>PSNR</td><td> $1 9 . 0 0 \pm 0 . 0 9$ </td><td> $1 5 . 8 1 \pm 0 . 0 3$ </td><td>19.11 Â± 0.25</td></tr><tr><td>Points</td><td> $\underline { { 4 3 . 0 1 \mathbf { k } \pm 1 . 9 5 \mathbf { k } } }$ </td><td> $\mathbf { 3 7 . 2 0 k \pm 4 . 7 8 k }$ </td><td>52.67k Â± 2.00k</td></tr><tr><td>ATE</td><td> $\overline { { 1 . 4 9 \pm 0 . 0 3 } }$ </td><td> $\overline { { 0 . 3 2 \pm 0 . 0 2 } }$ </td><td>1.17 Â± 0.34</td></tr><tr><td>FPS</td><td> $2 3 . 4 5 \pm 0 . 1 8$ </td><td> $2 3 . 4 4 \pm 0 . 0 1$ </td><td> $2 2 . 6 3 \pm 0 . 2 2$ </td></tr><tr><td rowspan="6">Ours</td><td>IPF</td><td> $8 . 8 8 \pm 0 . 1 4$ </td><td> $7 . 6 8 \pm 0 . 2 8$ </td><td>8.54 Â± 0.26</td></tr><tr><td>PSNR</td><td></td><td></td><td></td></tr><tr><td></td><td> $1 9 . 9 8 \pm 0 . 0 3$ </td><td> $2 1 . 9 2 \pm 0 . 4 2$ </td><td>22.18 Â± 1.20</td></tr><tr><td>Points</td><td> $\underline { { 4 5 . 6 4 \mathbf { k } \pm 1 . 1 8 \mathbf { k } } }$ </td><td> $\underline { { 6 8 . 6 8 \mathbf { k } \pm 1 0 . 0 0 \mathbf { k } } }$ </td><td>67.69k Â± 1.75k</td></tr><tr><td>ATE FPS</td><td> $\overline { { 1 . 5 2 \pm 0 . 0 3 } }$ </td><td> $\overline { { 0 . 3 0 \pm 0 . 0 1 } }$ </td><td>0.90 Â± 0.03</td></tr><tr><td>IPF</td><td> $2 3 . 0 6 \pm 0 . 2 2$ </td><td> $2 3 . 3 6 \pm 0 . 0 7$ </td><td> $2 2 . 7 8 \pm 0 . 1 0$ </td></tr><tr><td rowspan="5">GS-ICP-SLAM [16]</td><td>PSNR</td><td> $2 0 . 6 0 \pm 0 . 4 6$   $2 0 . 5 4 \pm 0 . 0 6$ </td><td> $1 8 . 0 5 \pm 0 . 3 1$   $2 2 . 7 5 \pm 0 . 2 2$ </td><td> $1 7 . 6 6 \pm 0 . 3 2$  22.95 Â± 0.79</td></tr><tr><td>Points</td><td> $3 8 . 6 5 \mathrm { k } \pm 0 . 7 6 \mathrm { k }$ </td><td> $4 9 . 8 0 \mathrm { k \pm 2 . 6 3 k }$ </td><td> $7 1 . 3 3 \mathrm { k } \pm 6 . 7 9 \mathrm { k }$ </td></tr><tr><td>ATE</td><td></td><td> $2 . 2 6 \pm 0 . 0 4$ </td><td></td></tr><tr><td>FPS</td><td> $3 . 2 6 \pm 0 . 2 8$ </td><td></td><td>3.07 Â± 0.41</td></tr><tr><td>IPF</td><td> $3 0 . 0 0 \pm 0 . 0 0$ </td><td> $3 0 . 0 0 \pm 0 . 0 0$ </td><td>30.00 Â± 0.00</td></tr><tr><td rowspan="5"></td><td>PSNR</td><td>6.10 Â± 0.05  $1 5 . 6 2 \pm 0 . 0 7$ </td><td>3.69 Â± 0.05  $1 8 . 4 3 \pm 0 . 1 9$ </td><td>3.96 Â± 0.08 19.20 Â± 0.05</td></tr><tr><td>Points</td><td> $\underline { { 0 . 5 3 \mathrm { m } \pm 6 . 8 2 \mathrm { k } } }$ </td><td> $1 . 9 1 \mathrm { m } \pm 1 1 . 3 7 \mathrm { k }$ </td><td> $\underline { { 2 . 0 9 \mathrm { m } \pm 2 1 . 0 4 \mathrm { k } } }$ </td></tr><tr><td>ATE</td><td> $\overline { { 3 . 9 2 \pm 0 . 7 1 } }$ </td><td> $\overline { { 2 . 4 4 \pm 0 . 0 6 } }$ </td><td>4.11 Â± 1.28</td></tr><tr><td>FPS</td><td></td><td></td><td>30.00 Â± 0.00</td></tr><tr><td>IPF</td><td> $3 0 . 0 0 \pm 0 . 0 0$ </td><td> $3 0 . 0 0 \pm 0 . 0 0$ </td><td></td></tr><tr><td rowspan="5">Ours (GS-ICP-SLAM)</td><td></td><td> $2 0 . 0 2 \pm 0 . 1 0$ </td><td> $1 8 . 4 3 \pm 0 . 1 5$ </td><td> $1 2 . 1 7 \pm 0 . 1 3$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>PSNR</td><td> $1 7 . 5 4 \pm 0 . 0 7$ </td><td> $2 1 . 3 5 \pm 0 . 2 0$ </td><td>20.84 Â± 0.06</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Points</td><td> $0 . 1 8 \mathrm { m } \pm 3 . 6 5 \mathrm { k }$ </td><td> $0 . 1 3 \mathrm { { m } \pm 1 2 . 3 2 \mathrm { { k } } }$ </td><td> $0 . 3 4 \mathrm { { m } \pm 1 9 . 2 4 \mathrm { { k } } }$ </td></tr></table>

Table III: Quantitative Results on VECtor.
<table><tr><td>Cam</td><td>Method</td><td>Metric</td><td>corner-slow</td><td>robot-normal</td><td>corridors-dolly</td></tr><tr><td rowspan="10">Mroor</td><td rowspan="5">Photo-SLAM [17]</td><td>ATE FPS</td><td> $\overline { { 0 . 6 6 \pm 0 . 0 1 } }$ </td><td> $\overline { { 2 . 2 0 \pm 1 . 6 6 } }$ </td><td> $\overline { { 9 . 5 6 \pm 6 . 0 8 } }$ </td></tr><tr><td></td><td> $2 3 . 2 7 \pm 0 . 2 1$ </td><td> $2 1 . 9 0 \pm 0 . 3 2$ </td><td> $2 0 . 1 8 \pm 0 . 2 6$ </td></tr><tr><td>IPF</td><td> $3 . 1 1 \pm 0 . 0 3$ </td><td>3.37 Â± 0.17</td><td>3.11 Â± 0.03</td></tr><tr><td>PSNR Points</td><td> $2 4 . 6 3 \pm 0 . 0 5$ </td><td> $1 9 . 5 8 \pm 0 . 1 8$ </td><td> $1 5 . 3 1 \pm 0 . 6 9$ </td></tr><tr><td></td><td> $0 . 1 2 \mathrm { m } \pm 1 7 . 0 2 \mathrm { k }$ </td><td> $0 . 1 6 \mathrm { { m } \pm 7 2 . 3 8 \mathrm { { k } } }$ </td><td> $0 . 3 8 \mathrm { m } \pm 3 . 9 9 \mathrm { k }$ </td></tr><tr><td rowspan="5">Ours (Photo-SLAM)</td><td>ATE</td><td> $\overline { { 0 . 6 8 \pm 0 . 0 2 } }$ </td><td> $\overline { { 2 . 3 5 \pm 1 . 1 7 } }$ </td><td> $\overline { { 1 0 . 0 6 \pm 6 . 2 0 } }$ </td></tr><tr><td>FPS</td><td> $2 1 . 5 6 \pm 0 . 3 3$ </td><td> $1 8 . 3 0 \pm 1 . 2 0$ </td><td> $1 8 . 0 0 \pm 0 . 2 6$ </td></tr><tr><td>IPF</td><td> $7 . 6 9 \pm 0 . 1 7$ </td><td> $1 0 . 7 8 \pm 0 . 6 8$ </td><td> $1 1 . 1 1 \pm 0 . 1 8$ </td></tr><tr><td>PSNR</td><td> ${ \bf 2 5 . 3 7 \pm 0 . 1 2 }$ </td><td> $\mathbf { 2 2 . 1 6 \pm 1 . 4 6 }$ </td><td> ${ \bf 2 3 . 0 2 \pm 5 . 6 7 }$ </td></tr><tr><td>Points</td><td> $7 . 3 1 \mathbf { k } \pm 0 . 2 5 \mathbf { k }$ </td><td> $\underline { { 8 . 2 4 \mathbf { k } \pm 2 . 0 6 \mathbf { k } } }$ </td><td> $3 6 . 9 6 \mathbf { k } \pm 1 . 5 9 \mathbf { k }$ </td></tr><tr><td rowspan="8">Sreo</td><td rowspan="5">Photo-SLAM [17]</td><td>ATE</td><td> $\overline { { 1 . 1 5 \pm 0 . 0 0 } }$ </td><td> $\overline { { 1 . 5 2 \pm 0 . 0 0 } }$ </td><td> $\overline { { 1 1 . 9 1 \pm 0 . 0 4 } }$ </td></tr><tr><td>FPS</td><td> $2 0 . 4 3 \pm 0 . 3 2$ </td><td> $1 7 . 7 7 \pm 0 . 3 1$ </td><td> $1 9 . 3 1 \pm 0 . 0 1$ </td></tr><tr><td>IPF</td><td>1.68 Â± 0.08</td><td>2.58 Â± 0.04</td><td> $2 . 7 6 \pm 0 . 0 2$ </td></tr><tr><td>PSNR</td><td> $1 9 . 3 4 \pm 0 . 0 2$ </td><td> $1 6 . 5 9 \pm 0 . 0 1$ </td><td> $1 4 . 5 1 \pm 0 . 3 4$ </td></tr><tr><td>Points</td><td> $3 8 . 9 8 \mathbf { k } \pm 4 . 2 9 \mathbf { k }$ </td><td> $4 7 . 3 6 \mathbf { k } \pm 0 . 6 4 \mathbf { k }$ </td><td> $0 . 2 4 \mathrm { { m } \pm 2 . 9 2 \mathrm { { k } } }$ </td></tr><tr><td rowspan="5">Ours (Photo-SLAM)</td><td>ATE</td><td> $\overline { { 1 . 1 5 \pm 0 . 0 0 } }$ </td><td> $\overline { { 1 . 5 2 \pm 0 . 0 0 } }$ </td><td> $\overline { { 1 1 . 5 1 \pm 0 . 2 3 } }$ </td></tr><tr><td>FPS</td><td> $2 0 . 7 5 \pm 0 . 3 7$ </td><td> $1 4 . 6 4 \pm 0 . 2 3$ </td><td> $1 6 . 6 4 \pm 0 . 8 3$ </td></tr><tr><td>IPF</td><td>9.23 Â± 0.02</td><td>12.24 Â± 0.20</td><td>11.21 Â± 0.16</td></tr><tr><td>PSNR</td><td> $1 9 . 5 6 \pm 0 . 0 4$ </td><td> $1 6 . 7 7 \pm 0 . 0 5$ </td><td> $1 9 . 3 4 \pm 0 . 0 6$ </td></tr><tr><td>Points</td><td> $\mathbf { 6 . 4 5 k \pm 0 . 2 0 k }$ </td><td> $\mathbf { 7 . 6 8 k \pm 0 . 2 4 k }$ </td><td> $\mathbf { 3 0 . 8 1 \mathrm { k } \pm 2 . 2 1 \mathrm { k } }$ </td></tr></table>

Photo-SLAM [17] and GS-ICP-SLAM [16]. In Table II, our approach achieves high rendering quality using a comparable number of Gaussian primitives to MonoGS [9]. In Table III, we present the results on VECtor [31], specifically using a monocular camera. Our method improves the average PSNR by more than 3 dB with only one-tenth of the Gaussian primitives. Furthermore, the qualitative results depicted in Figure 5 corroborate that our approach achieves high-fidelity rendering.

Figure 6 depicts our ablation studies on the monocular Replica dataset [29], rigorously validating our design choices and highlighting their contributions to system performance. Key findings include:

Splat-wise backpropagation enhances the rendering quality by refining the iterative process efficiently. The integration of splat-wise backpropagation has significantly improved average total iterations from 4.6k to 15.4k and average PSNR from 32.1 dB to 33.8 dB.

<!-- image-->

Figure 6: The Radar Chart of Ablation Study. Radial axis presents the PSNR.  
<!-- image-->  
Figure 7: The Effect of Opacity Regularization. The left side illustrates the value of PSNR. The right side depicts the count of Gaussian points.

Adaptive optimization strategically allocates computational resources to enhance rendering quality. Integrating splat-wise backpropagation with adaptive optimization has continuously boosted average PSNR from 33.8 dB to 34.6 dB. Furthermore, as illustrated in Figure 2, this approach equitably distributes computational resources across keyframes, efficiently addressing long-tail optimization challenges.

Opacity regularization is instrumental in reducing the model size without compromising the superior rendering quality. Our opacity regularization technique, as shown in Figure 7, can halve the model size with a regularization coefficient of $\lambda _ { o } = 0 . 0 0 1$ , with minimal PSNR performance loss. Increasing the coefficient to 0.01 further reduces less critical Gaussian primitives, which results in a more efficient model at the expense of some rendering quality.

## V. LIMITATIONS AND FEATURE WORK

CaRtGS is an adaptive optimization technology that leverages 3D Gaussian models for high-quality rendering and environmental reconstruction in real-time GS-SLAM systems.

Despite its potential, several limitations and challenges are identified below, structured into categories for clarity:

1) Dynamic Environment Challenges. CaRtGS assumes static environments, limiting real-world use and causing tracking failures with dynamic objects.

2) Localization Robustness. CaRtGS focuses on improving the rendering quality of GS-SLAM. However, localization accuracy affects rendering quality, especially in some degeneracy scenarios. Therefore, a robustness localization module is essential for GS-SLAM.

3) Geometry Accuracy. Effective geometry mapping is vital in GS-SLAM. As shown in Table III, the stereo modelâs inferior rendering quality stems from the stereo cameraâs suboptimal geometry mapping.

Looking forward, we envision further improvements by integrating advanced machine learning models to predict and handle dynamic objects.

## VI. CONCLUSION

In this work, we introduced CaRtGS, a novel framework that integrates computational alignment with Gaussian Splatting SLAM to achieve real-time photorealistic dense rendering. Our key contribution lies in the development of an adaptive computational alignment strategy that optimizes the rendering process by addressing the computational misalignment inherent in GS-SLAM systems. Through fast splat-wise backpropagation, adaptive optimization, and opacity regularization, we significantly enhanced the rendering quality and computational efficiency of the SLAM process.

## REFERENCES

[1] Z. Teed and J. Deng, âDroid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras,â Advances in neural information processing systems, vol. 34, pp. 16 558â16 569, 2021. 1, 2

[2] A. Segal, D. Haehnel, and S. Thrun, âGeneralized-icp.â in Robotics: science and systems, vol. 2, no. 4. Seattle, WA, 2009, p. 435. 1, 2, 3

[3] C. Campos, R. Elvira, J. J. G. RodrÃ­guez, J. M. Montiel, and J. D. TardÃ³s, âOrb-slam3: An accurate open-source library for visual, visualâ inertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021. 1, 2, 3

[4] S. Zhong, H. Chen, Y. Qi, D. Feng, Z. Chen, J. Wu, W. Wen, and M. Liu, âColrio: Lidar-ranging-inertial centralized state estimation for robotic swarms,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 3920â3926. 1

[5] D. Feng, Y. Qi, S. Zhong, Z. Chen, Q. Chen, H. Chen, J. Wu, and J. Ma, âS3e: A multi-robot multimodal dataset for collaborative slam,â IEEE Robotics and Automation Letters, 2024. 1

[6] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021. 1

[7] F. Tosi, Y. Zhang, Z. Gong, E. SandstrÃ¶m, S. Mattoccia, M. R. Oswald, and M. Poggi, âHow nerfs and 3d gaussian splatting are reshaping slam: a survey,â arXiv preprint arXiv:2402.13255, vol. 4, 2024. 1

[8] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023. 1, 2, 3, 4, 5, 6

[9] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048. 1, 2, 3, 4, 5, 7

[10] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366. 1, 2, 3, 4, 5, 7

[11] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, âGaussian-slam: Photo-realistic dense slam with gaussian splatting,â arXiv preprint arXiv:2312.10070, 2023. 1, 2, 3, 4, 5, 7

[12] L. Zhu, Y. Li, E. SandstrÃ¶m, K. Schindler, and I. Armeni, âLoopsplat: Loop closure by registering 3d gaussian splats,â arXiv preprint arXiv:2408.10154, 2024. 1, 2, 3, 4

[13] E. SandstrÃ¶m, K. Tateno, M. Oechsle, M. Niemeyer, L. Van Gool, M. R. Oswald, and F. Tombari, âSplat-slam: Globally optimized rgb-only slam with 3d gaussians,â arXiv preprint arXiv:2405.16544, 2024. 1, 2, 3, 4

[14] F. A. Sarikamis and A. A. Alatan, âIg-slam: Instant gaussian slam,â arXiv preprint arXiv:2408.01126, 2024. 1, 2, 3, 4

[15] Z. Peng, T. Shao, Y. Liu, J. Zhou, Y. Yang, J. Wang, and K. Zhou, âRtgslam: Real-time 3d reconstruction at scale using gaussian splatting,â in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1â11. 1, 2, 3, 4

[16] S. Ha, J. Yeon, and H. Yu, âRgbd gs-icp slam,â in European Conference on Computer Vision. Springer, 2024, pp. 180â197. 1, 2, 3, 4, 5, 6, 7

[17] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Realtime simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â21 593. 1, 2, 3, 4, 5, 6, 7

[18] S. Kheradmand, D. Rebain, G. Sharma, W. Sun, J. Tseng, H. Isack, A. Kar, A. Tagliasacchi, and K. M. Yi, â3d gaussian splatting as markov chain monte carlo,â arXiv preprint arXiv:2404.09591, 2024. 2

[19] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 654â20 664. 2

[20] S. S. Mallick, R. Goel, B. Kerbl, M. Steinberger, F. V. Carrasco, and F. De La Torre, âTaming 3dgs: High-quality radiance fields with limited resources,â in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1â 11. 2, 3, 4

[21] S. Durvasula, A. Zhao, F. Chen, R. Liang, P. K. Sanjaya, and N. Vijaykumar, âDistwar: Fast differentiable rendering on raster-based rendering pipelines,â arXiv preprint arXiv:2401.05345, 2023. 2, 3

[22] L. HÃ¶llein, A. BoÅ¾ic, M. ZollhÃ¶fer, and M. NieÃner, â3dgs-lm: Ë Faster gaussian-splatting optimization with levenberg-marquardt,â arXiv preprint arXiv:2409.12892, 2024. 2

[23] G. Feng, S. Chen, R. Fu, Z. Liao, Y. Wang, T. Liu, Z. Pei, H. Li, X. Zhang, and B. Dai, âFlashgs: Efficient 3d gaussian splatting for largescale and high-resolution rendering,â arXiv preprint arXiv:2408.07967, 2024. 2

[24] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos et al., âInstantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds,â arXiv preprint arXiv:2403.20309, 2024. 2

[25] S. Niedermayr, J. Stumpfegger, and R. Westermann, âCompressed 3d gaussian splatting for accelerated novel view synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10 349â10 358. 2, 3

[26] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, âCompact 3d scene representation via self-organizing gaussian grids,â in European Conference on Computer Vision. Springer, 2024, pp. 18â34. 2, 3

[27] H. Wang, H. Zhu, T. He, R. Feng, J. Deng, J. Bian, and Z. Chen, âEnd-toend rate-distortion optimized 3d gaussian representation,â in European Conference on Computer Vision. Springer, 2024, pp. 76â92. 2, 3

[28] L. Liso, E. SandstrÃ¶m, V. Yugay, L. Van Gool, and M. R. Oswald, âLoopy-slam: Dense neural slam with loop closures,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 363â20 373. 5, 7

[29] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019. 5, 6, 7

[30] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of rgb-d slam systems,â in 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012, pp. 573â580. 5

[31] L. Gao, Y. Liang, J. Yang, S. Wu, C. Wang, J. Chen, and L. Kneip, âVector: A versatile event-centric benchmark for multi-sensor slam,â IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 8217â8224, 2022. 5, 6, 7