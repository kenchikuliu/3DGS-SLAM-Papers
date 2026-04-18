# LOBE-GS: LOAD-BALANCED AND EFFICIENT 3D GAUSSIAN SPLATTING FOR LARGE-SCALE SCENE RE-CONSTRUCTION

Sheng-Hsiang Hung1â Ting-Yu Yen1 Wei-Fang Sun2 Simon See2

Shih-Hsuan Hung1 Hung-Kuo Chu1

1 National Tsing Hua University 2 NVIDIA AI Technology Center (NVAITC)

## ABSTRACT

3D Gaussian Splatting (3DGS) has established itself as an efficient representation for real-time, high-fidelity 3D scene reconstruction. However, scaling 3DGS to large and unbounded scenes such as city blocks remains difficult. Existing divide-and-conquer methods alleviate memory pressure by partitioning the scene into blocks, but introduce new bottlenecks: (i) partitions suffer from severe load imbalance since uniform or heuristic splits do not reflect actual computational demands, and (ii) coarse-to-fine pipelines fail to exploit the coarse stage efficiently, often reloading the entire model and incurring high overhead. In this work, we introduce LoBE-GS, a novel Load-Balanced and Efficient 3D Gaussian Splatting framework, that re-engineers the large-scale 3DGS pipeline. LoBE-GS introduces a depth-aware partitioning method that reduces preprocessing from hours to minutes, an optimization-based strategy that balances visible Gaussiansâa strong proxy for computational loadâacross blocks, and two lightweight techniques, visibility cropping and selective densification, to further reduce training cost. Evaluations on large-scale urban and outdoor datasets show that LoBE-GS consistently achieves up to 2Ã faster end-to-end training time than state-of-theart baselines, while maintaining reconstruction quality and enabling scalability to scenes infeasible with vanilla 3DGS.

## 1 INTRODUCTION

Recent advances in 3D scene reconstruction and novel view synthesis have shifted from classical photogrammetry and Neural Radiance Fields (NeRFs) toward explicit, real-time representations. While photogrammetry offers geometric precision but poor rendering efficiency, NeRFs achieve photorealism but remain computationally expensive. 3D Gaussian Splatting (3DGS) addresses these limitations by representing scenes with millions of anisotropic Gaussian primitives optimized through a GPU-friendly rasterization pipeline, delivering both high fidelity and real-time performance. Its efficiency has quickly established 3DGS as a leading representation for scalable 3D content creation.

Despite its success in bounded scenes, scaling 3DGS to large and unbounded environments, such as city-scale reconstructions, remains an open challenge. The memory and computational costs scale with the number of Gaussian primitives, leading to optimization times and GPU usage that quickly become prohibitive. To mitigate this, recent works such as CityGaussian (CityGS) (Liu et al., 2025), VastGaussian (VastGS) (Lin et al., 2024), and DOGS (Chen & Lee, 2024) adopt a divide-andconquer strategy, partitioning large scenes into spatial blocks that can be processed in parallel. While effective in reducing raw memory pressure, this paradigm introduces new bottlenecks as follows.

â¢ Lack of load balancing: Current partitioning strategies do not explicitly account for computational load balance. Heuristics such as uniform grid splits or block size normalization often yield sub-regions with highly uneven optimization demands. As a result, the slowest block dominates the total training time, creating a long-tail bottleneck.

â¢ Inefficient coarse-to-fine pipelines: Methods employing a coarse-to-fine pipeline, such as CityGS (Liu et al., 2025), fail to fully exploit the coarse stage for accelerating fine-level optimization. The coarse model is typically reloaded in full, incurring heavy computational overhead.

To overcome these limitations, we introduce LoBE-GS, a novel framework that fundamentally reengineers the large-scale 3DGS pipeline for load-balanced and efficient parallel training. LoBE-GS addresses the inefficiency of heuristic partitioning, improves the utilization of coarse models, and establishes a standardized evaluation protocol. We first introduce a novel partitioning approach that radically reduces the data partitioning time. Existing methods can result in a complex O(M Ã N) projection problem, where M is the number of blocks and N is the number of camera views, requiring up to several hours. Our method leverages depth information from a coarse model to assign each camera to its corresponding block with a single, highly efficient projection per camera. This reduces the projection complexity to a linear O(N) time and shortens the preprocessing time from hours to minutes.

To avoid unbalanced loading in each block for the parallel training, we employee an optimization to scene partitioning that directly addresses the load-balancing problem. Our experiments revealed a strong correlation between the initial number of visible Gaussians in the blocks and the subsequent optimization time. We therefore adopt the number of visible Gaussians as a reliable proxy for computational load. By explicitly balancing this metric across blocks, our framework eliminates long-tailed training bottlenecks and ensures more uniform computational demands. Moreover, we propose two complementary techniques to reduce the computational load of each block. First, we introduce visibility cropping, a technique applied after scene partitioning to prune irrelevant Gaussians from each block, which reduces the training time without sacrificing the quality of the final reconstruction. Second, we propose selective densification to further reduce the computational load of each block by strategically adding or cloning Gaussians only when needed.

We evaluate LoBE-GS on diverse large-scale datasets, including urban and outdoor scenes spanning hundreds of meters. Experimental results show that our method consistently delivers faster training and more balanced computation than prior approaches, while maintaining or improving reconstruction quality. In particular, LoBE-GS reduces end-to-end training time by up to 2Ã over baselines that use coarse models and achieves stable scalability on scenes that are otherwise infeasible for vanilla 3DGS. The main contributions of this work are summarized as follows:

â¢ We identify load-balancing limitations in prior approaches and introduce a proxy that more closely correlates with fine-training runtime, enabling improved load balancing.

â¢ We present LoBE-GS, featuring (i) load balance-aware scene partitioning for evenly distributed computational workloads, (ii) fast camera selection to minimize partition overhead, and (iii) visibility cropping and selective densification for accelerated fine-training.

â¢ Extensive experiments show that LoBE-GS achieves a 2Ã training speedup over existing methods while preserving rendering quality.

## 2 RELATED WORK

## 2.1 NOVEL VIEW SYNTHESIS

Given a set of captured images, novel view synthesis seeks to render photorealistic 3D scenes from previously unseen viewpoints. Neural Radiance Fields (NeRF) (Mildenhall et al., 2020) model radiance fields with an MLP and use volumetric ray marching to integrate color along camera rays. NeRF delivers high fidelity but incurs substantial training time and inference latency due to dense sampling and repeated neural evaluations. In contrast, 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) adopts Gaussian primitives, enabling differentiable rasterization for real-time rendering and training that often converges within minutes. While 3DGS yields strong quality, open issues include aliasing under wide baselines, semi-transparent geometry leakage, memory growth from millions of primitives, and robustness under sparse views or imperfect calibration. These advances and limitations motivate our design choices and evaluation, specially for large scale reconstruction.

## 2.2 LARGE-SCALE SCENE RECONSTRUCTION

For decades, reconstructing large-scale 3D scenes has been a central goal for researchers and engineers (Snavely et al., 2006; Agarwal et al., 2011). At city and regional scales, such reconstruction, especially for aerial views (Jiang et al., 2025; Tang et al., 2025), faces prohibitive memory demands and computational performance, motivating scalable training and rendering strategies.

Distributed training approaches train a unified model jointly across multiple GPUs. NeRF-XL (Li et al., 2024) shares NeRF parameters and activations across devices to maintain a single global model, executing multi-GPU volume rendering and loss computation with low inter-GPU communication, while DOGS (Chen & Lee, 2024) and Grendel-GS (Zhao et al., 2024) distribute Gaussian primitives via consensus or sparse all-to-all routing. However, such systems typically require customized multi-GPU infrastructure to support frequent synchronization and communication, which limits their practicality on standard hardware setups.

Divide-and-conquer approaches partition a large scene into subregions, train submodels in parallel with multiple GPUs, and compose their outputs. Block-NeRF (Tancik et al., 2022) partitions a city into spatial blocks and assigns training views by camera position; Mega-NeRF (Turki et al., 2022a) decomposes space into grids and routes each pixel to the grids intersected by its ray; Switch-NeRF (Mi & Xu, 2023) learns the decomposition and routing end-to-end via a mixture-of-NeRFexperts. Within 3DGS representations, VastGS (Lin et al., 2024) introduces a progressive spatial partitioning strategy that divides a large scene into blocks and assigns training cameras and point clouds using an airspace-aware visibility criterion. Each block is optimized in parallel and subsequently fused to a seamless global 3DGS reconstruction. CityGS (Liu et al., 2025) leverages a coarse 3DGS prior to guide scene partitioning and parallel 3DGS submodel training, improving coherence and reconstruction quality across spatial partitions. They map unbounded scenes into a normalized unit cube and then partition the contracted scenes with a uniform grid for parallel training. However, most of the aforementioned works underemphasize load balancing of the submodels during partitioning, which limits parallel scalability. Moreover, CityGS loads the entire coarse model during the parallel stage, which is inefficient. To address these, LoBE-GS balances the 3DGS prior across submodels within each subregion and trains them efficiently in parallel.

## 2.3 EFFICIENT GAUSSIAN SPLATTING RECONSTRUCTION

As new 3DGS methods emerge, many research efforts target efficient 3D Gaussian Splatting reconstruction and rendering. With limited resources, 3DGS compression (Navaneet et al., 2024; Papantonakis et al., 2024) reduces on-disk storage, while Taming 3DGS (Mallick et al., 2024) addresses budget-constrained training and rendering via guided, purely constructive densification that steers growth toward high-contribution Gaussians. For large-scale scenes, level-of-detail (LoD) 3DGS representations enable efficient rendering (Ren et al., 2024; Kerbl et al., 2024). CityGaussianV2 (Liu et al., 2024) builds on CityGS (Liu et al., 2025) with an optimized parallel training pipeline that incorporates 2DGS for accurate geometric modeling. Momentum-GS (Fan et al., 2024) extends Scaffold-GS (Lu et al., 2024) to large-scale scenes by introducing scene momentum self-distillation and reconstruction-guided block weighting, allowing scalable parallel training with improved reconstruction quality. CityGS-X (Gao et al., 2025) proposes a scalable hybrid hierarchical representation with multitask batch rendering and training, eliminating mergeâpartition overhead while achieving efficient and geometrically accurate large-scale reconstruction. In this work, we focus on an efficient 3DGS reconstruction for large-scale scenes with coarse 3DGS prior and load-balanced parallel training.

## 3 ANALYSIS OF SCENE PARTITIONING AND LOAD BALANCING

In this section, we first show that existing scene partitioning strategies fail to achieve satisfactory load balancing during the fine-training stage. We then provide a principled analysis to identify a reliable proxy for estimating the per-block fine-training runtime.

<!-- image-->  
Figure 1: Illustration of per-block training time under different partitioning strategies. (Left) The uniform area partitioning in CityGS. (Right) The load-balanced partitioning in LoBE-GS.

## 3.1 IMPACT OF SCENE PARTITION ON LOAD BALANCING

Large-scale 3DGS pipelines typically adopt a partitionâandâmerge paradigm: the scene is divided into B spatial blocks, each optimized independently in parallel, and then merged into a complete model. Some methods further employ a coarse-to-fine strategy, where a coarse model is trained first, followed by scene partitioning and parallel fine training before the final merging stage. Let $T _ { \mathrm { c o a r s e } }$ denote the coarse-stage optimization time, $T _ { \mathrm { p a r t i t i o n } }$ the partitioning time, and $T _ { \mathrm { f i n e } } ^ { ( b ) }$ the fine-stage runtime of block $b \in { \bf \bar { \{ 1 , \dots , B \} } }$ . Assuming sufficient computational resources to run all fine-stage processes in parallel, the end-to-end runtime is defined as:

$$
T _ { \mathrm { E 2 E } } = T _ { \mathrm { c o a r s e } } + T _ { \mathrm { p a r t i t i o n } } + \operatorname* { m a x } _ { b \in \{ 1 , \ldots , B \} } T _ { \mathrm { f i n e } } ^ { ( b ) } .\tag{1}
$$

Thus, an effective partitioning strategy must balance the workloads across blocks to minimize the runtime of the slowest block while maintaining reconstruction quality. Prior work has relied on heuristics such as equalizing area, camera counts, or point counts, yet their ability to predict finestage runtime were underexplored.

As a motivational example, consider the CityGS pipeline, which partitions the scene by equalizing block areas in contracted space. Figure 1 illustrates the fine-stage runtime per block on the Building dataset. Figure 1(a) shows that the strategy adopted by CityGS leads to significant load imbalance in fine-stage training. In contrast, Figure 1(b) shows that LoBE-GS achieves a more balanced runtime distribution by employing a different proxy. Similar patterns are observed across other datasets (see Appendix A.2), suggesting that existing heuristics are often suboptimal for actual fine-stage runtimes. As a result, they lead to skewed per-block runtimes and longer end-to-end runtime $T _ { \mathrm { E 2 E } }$

## 3.2 RUNTIME CORRELATION WITH PER-BLOCK PREDICTORS

To address this, we analyze the correlation between candidate proxy variables and observed finestage runtimes to determine which predictors most accurately reflect the computational cost of each block. For each block b, we computed the Pearson correlation between its fine-stage runtime $T _ { \mathrm { f i n e } } ^ { ( b ) }$ (in minutes) and the following quantities, all available prior to fine-stage optimization:

$A ^ { ( b ) }$ : area of block b in contracted space.

$C ^ { ( b ) }$ : number of cameras assigned to block b.

$G _ { \mathrm { b l k } } ^ { ( b ) }$ : initial number of Gaussians inside block b at the start of fine-stage optimization.

$G _ { \mathrm { v i s } } ^ { ( b ) }$ : initial number of Gaussians visible across all cameras assigned to block b.

$G _ { \mathrm { a v g \mathrm { - } v i s } } ^ { ( b ) } = G _ { \mathrm { v i s } } ^ { ( b ) } / C ^ { ( b ) }$ : initial average number of visible Gaussians per assigned camera.

$$
{ \begin{array} { r l r l r l r l r l } { \cdots \Phi \cdots } & { A ^ { ( b ) } \ ( r = - 0 . 2 5 ) } & & { \cdots \bullet \cdots } & { C ^ { ( b ) } \ ( r = 0 . 1 2 ) } & & { \cdots \bullet \cdots } & { G _ { \ast b \ast } ^ { ( b ) } \ ( r = 0 . 8 8 ) } & & { \qquad } & & { G _ { \ast b \ast } ^ { ( b ) } \ ( r = 0 . 5 2 ) } & & { \cdots \bullet \cdots } & { G _ { \ast \ast } ^ { ( b ) } \ ( r = 0 . 1 1 ) } \end{array} }
$$

<!-- image-->

<!-- image-->  
Figure 2: Correlation between per-block training time and block-level statistics under CityGSâs partitioning. (a) Plots of block area $A ^ { ( b ) }$ and camera count $C ^ { ( b ) }$ . (b) Plots of Gaussian-based measures $( \bar { G } _ { \mathrm { b l k } } ^ { ( b ) } , G _ { \mathrm { v i s } } ^ { ( b ) } , G _ { \mathrm { a v g - v i s } } ^ { ( b ) } ) . G _ { \mathrm { v i s } } ^ { ( b ) }$ yields the strongest and most consistent correlation across datasets.

Figure 2 presents scatter plots across five representative datasets, Building, Rubble, Residence, Sci-Art, and MatrixCity, evaluated under fixed hardware and hyperparameters. Each point is color-coded by a candidate proxy variable, with fine-stage runtime on the x-axis and the corresponding proxy value on the y-axis. For each proxy, a dashed line of the same color hue is fit using linear regression. The legend also reports the Pearson correlation coefficients (r) between $T _ { \mathrm { f i n e } } ^ { ( b ) }$ and the respective block-level quantities.

The results indicate that the area proxy $A ^ { ( b ) }$ , commonly adopted in prior works (Liu et al., 2025; 2024; Fan et al., 2024), exhibits relatively weak correlation with fine-stage runtime. Similarly, the per-block Gaussian count $G _ { \mathrm { b l k } } ^ { ( b ) }$ shows minimal correlation, implying that considering only Gaussians physically contained within a block underestimates the effective optimization load. In contrast, the visibility-augmented measure $G _ { \mathrm { v i s } } ^ { ( b ) }$ achieves the strongest and most consistent correlation across datasets, confirming its suitability as a reliable predictor of per-block training cost. Normalizing this quantity by camera count, resulting in $G _ { \mathrm { a v g - v i s } } ^ { ( b ) } .$ , weakens the correlation, while the camera count alone, $C ^ { ( b ) }$ , also used in previous studies (Chen & Lee, 2024; Yuan et al., 2025), exhibits only weak correlation. Overall, these findings suggest that balancing partitions by the number of initial visible Gaussians $G _ { \mathrm { v i s } } ^ { ( b ) }$ , as implemented in the proposed LoBE-GS, provides a more principled strategy than traditional equal-area or equal-camera approaches.

## 4 METHODOLOGY

Prior large-scale 3DGS systems have demonstrated strong results but continue to face challenges with load imbalance and training efficiency. To address these limitations, we propose LoBE-GS, a coarse-to-fine training framework where each block is fine-trained independently, following prior works (Liu et al., 2025; 2024). The overall pipeline is illustrated in Figure 3. Section 4.1 introduces load balance-aware scene partition that iteratively refines initial uniform cuts to minimize a proxy for fine-stage runtime. In Section 4.2, fast camera selection is proposed to improve efficiency over existing camera selection strategies. Finally, Section 4.3 describes visibility cropping and selective densification, two techniques that further reduce memory and computation costs during fine-training.

## 4.1 LOAD BALANCE-AWARE SCENE PARTITION

To mitigate load imbalance, we propose load balance-aware scene partition that minimizes maximum fine-training time $\operatorname* { m a x } _ { b } T _ { \mathrm { f i n e } } ^ { ( b ) }$ by leveraging proxy metrics $\operatorname* { m a x } _ { b } G _ { \mathrm { v i s } } ^ { ( b ) } .$ , which exhibit strong correlation with fine-stage runtimes as analyzed in Section 3.2. For a grid partition with $B = m \times n$ blocks, given a coarse model $\mathcal { G } _ { \mathrm { c o a r s e } }$ and a set of c camera views, the objective is to optimize vertical

<!-- image-->  
Figure 3: Overview of our framework. Our approach begins with training a coarse 3DGS model. Using our load balanceâaware data partition, we optimize the grid cuts to achieve a more balanced division of the scene. We then apply visibility cropping and selective densification before and during the parallel fine-training stage, enabling faster and more efficient training. Finally, we prune regions outside each block and merge the results into a unified, high-quality model.

and horizontal cut positions $( v , h )$ such that:

$$
( \pmb { v } ^ { * } , \pmb { h } ^ { * } ) = \arg \operatorname* { m i n } _ { \pmb { v } , \pmb { h } } \operatorname* { m a x } _ { b } G _ { \mathrm { v i s } } ^ { ( b ) } ( \pmb { v } , \pmb { h } ) ,\tag{2}
$$

where $\pmb { v } = ( v _ { 1 } , \dots , v _ { m - 1 } ) \in ( 0 , 1 ) ^ { m - 1 }$ and $\pmb { h } = ( h _ { 1 } , \dots , h _ { n - 1 } ) \in ( 0 , 1 ) ^ { n - 1 }$ denotes monotonically increasing cut positions in contracted space. The proxy $G _ { \mathrm { v i s } } ^ { ( b ) } ( \pmb { v } , \pmb { h } )$ denotes the number of visible Gaussians in block $b = ( i - 1 ) ( n + 1 ) + j$ for $i \in \{ 1 , \ldots , m \}$ and $j \in \{ 1 , \ldots , n \}$ , as defined by the corresponding cut boundaries $\dot { B } ^ { ( b ) }$ for the i-th row, j-th column block.

Since the computation of $G _ { \mathrm { v i s } } ^ { ( b ) } ( \pmb { v } , \pmb { h } )$ is non-differentiable, we adopt Bayesian Optimization (BO) with a Gaussian Process (GP) surrogate for iterative cut refinement. The process begins with an initial uniform partition $( \pmb { v } ^ { [ 0 ] } , \pmb { h } ^ { [ 0 ] } )$ , where $\begin{array} { r } { ( v _ { i } ^ { [ 0 ] } , h _ { j } ^ { [ 0 ] } ) = ( \frac { i } { m } , \frac { j } { n } ) } \end{array}$ . To preserve ordering, each cut is constrained to move at most halfway toward its neighbors, i.e., $\begin{array} { r } { v _ { i } \in [ \frac { 1 } { 2 } ( v _ { i - 1 } ^ { [ 0 ] } + v _ { i } ^ { [ 0 ] } ) , \frac { 1 } { 2 } ( v _ { i } ^ { [ 0 ] } + v _ { i + 1 } ^ { [ 0 ] } ) ] } \end{array}$ with $v _ { 0 } = 0$ and $v _ { m } = 1$ (defined analogously for $h _ { j } )$ . At each iteration l, BO proposes candidate cuts $( { \pmb v } ^ { [ l ] } , h ^ { [ l ] } )$ . The corresponding block regions are set to slightly enlarged grid cell, $\begin{array} { r l } { \boldsymbol { B } ^ { ( b ) } } & { { } = } \end{array}$ $[ v _ { i - 1 } ^ { [ l ] } - \delta _ { v } , v _ { i } ^ { [ l ] } + \delta _ { v } ] \times [ h _ { j - 1 } ^ { [ l ] } - \delta _ { h } , h _ { j } ^ { [ l ] } + \delta _ { h } ]$ , following prior works. Each block is then assigned a camera set ${ \mathcal { C } } ^ { ( b ) }$ using standard view assignment strategies, and the number of visible Gaussians $G _ { \mathrm { v i s } } ^ { ( b ) }$ is calculated. The GP surrogate is updated to fit the observed maxb $G _ { \mathrm { v i s } } ^ { ( b ) } ( { \pmb v } ^ { [ l ] } , { \pmb h } ^ { [ l ] } )$ , and the best solution is tracked. After L iterations, the best solution is returned. In practice, $L = 1 0 0$ and $\begin{array} { r } { ( \delta _ { v } , \delta _ { h } ) \ = \ ( \frac { 0 . 1 } { m } , \frac { 0 . 1 } { n } ) } \end{array}$ yield satisfactory results, eliminating the need for reinitialization or nested search-space refinements.

## 4.2 FAST CAMERA SELECTION

Camera selection is performed to assign a subset of views $\mathcal { C } ^ { ( b ) }$ to each block for fine-training. The goal is to reduce per-block fine-training cost by discarding views with negligible coverage of the corresponding block region. This ensures that each block is optimized with only the most relevant views, improving efficiency without compromising reconstruction quality.

Despite its importance, prior studies often overlook the computational burden of this process, which can account for nearly half of the overall end-to-end runtime (see Section 5.3). For instance, given M partitioned blocks and N camera views, CityGS assigns cameras by computing the SSIM between the full coarse render and each per-block render, where the latter is obtained by filtering out Gaussians outside the block boundaries. This requires rendering every view for every block, resulting in at least $( M + 1 ) \times N$ projections, which constitutes the main computational bottleneck.

To eliminate this overhead, we introduce fast camera selection, which reduces the computation to only N projections. First, for each camera view, we compute the per-pixel depth D using the Î±- blending equation: $\begin{array} { r } { D = \sum _ { i \in \mathcal { N } } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } \end{array}$ , where $\mathcal { N }$ is the ordered set of points along the ray, $d _ { i }$ the depth of point i, and $\alpha _ { i }$ its opacity determined by covariance and opacity. The resulting depth map is then back-projected into 3D space, forming a dense point cloud $\mathcal { P } ^ { ( c ) } = \{ p _ { c , k } \ | \ k = \ $

$1 , \ldots , K \}$ , with $\pmb { p } _ { c , k } \in \mathbb { R } ^ { 3 }$ and K denotes the total number of points for camera c. Next, for each camera c and block b, we compute the visibility ratio of points inside the block:

$$
V _ { c , b } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { 1 } [ { \pmb p } _ { c , k } \in { \pmb B } ^ { ( b ) } ] ,\tag{3}
$$

where 1 denotes the indicator function, and $B ^ { ( b ) }$ is the spatial region of block b. Finally, the assigned camera set for block b is defined as $\mathcal { C } ^ { ( b ) } = \{ c \ | \ V _ { c , b } \geq \tau \}$ , where Ï is a predefined threshold (with $\tau = 0 . 1 5 )$ to prune views with negligible block coverage. This makes the procedure substantially faster, even enabling its use as a subroutine in BO, where the back-projection is computed once and reused throughout all iterations.

## 4.3 VISIBILITY CROPPING AND SELECTIVE DENSIFICATION

Prior coarse-to-fine 3DGS pipelines load the entire coarse model during per-block fine-training. This introduces both memory and runtime overhead. The memory overhead arises from storing the entire coarse model in GPU memory, while the runtime overhead arises from the Adam optimizer rather than rendering, as frustum culling already excludes non-visible points. Since Adam maintains momentum terms, it still updates parameters of all Gaussians, including those not observed by any camera in ${ \mathcal { C } } ^ { ( b ) }$ . Similar effects have also been observed in Mallick et al. (2024).

As fine-trained models are cropped before being merged into the final model, one naive solution is to retain only Gaussians strictly within each block $\mathcal { G } _ { \mathrm { b l k } } ^ { ( b ) } = \{ \pmb { g } \in \mathcal { G } _ { \mathrm { c o a r s e } } \ | \ \pmb { g } \in \mathcal { B } ^ { ( b ) } \}$ . However, this leads to degraded results due to over-pruning of Gaussians that lie outside block boundaries that remain visible in some views. To address this, we introduce visibility cropping that retain the visible Gaussians $\mathcal { G } _ { \mathrm { v i s } } ^ { ( b ) } = \{ \pmb { g } \in \mathcal { G } _ { \mathrm { c o a r s e } } \ | \ g$ visible from some $c \in \mathcal { C } ^ { ( b ) } \}$ for each block prior to fine-training. This visibility-based filtering substantially reduces the number of Gaussians involved in optimization. In addition, since $G _ { \mathrm { v i s } } ^ { ( b ) } = | \mathcal { G } _ { \mathrm { v i s } } ^ { ( b ) } |$ must be recomputed at every BO iteration, we implement its evaluation entirely in NVIDIA Warp, achieving near-native CUDA performance and significantly reducing partition time. More implementation details are presented in Appendix A.1.

While visibility cropping preserves all visible Gaussians necessary for fine-training, it also includes those outside the block, $\mathrm { i . e . , } \mathcal { G } _ { \mathrm { v i s } } ^ { ( b ) } \setminus \mathcal { G } _ { \mathrm { b l k } } ^ { ( b ) }$ , which are ultimately discarded prior to merging. Although retaining these Gaussians is essential to prevent quality degradation, they need not participate in densification. Motivated by this observation, we introduce selective densification, which restricts densification to Gaussians strictly within the block. This approach reduces the number of new Gaussians created during training, thereby lowering memory consumption and improving optimization efficiency, while maintaining per-block fidelity.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Datasets. We conducted experiments on five large-scale scenes, including four real-world datasets and one synthetic dataset. For the real-world datasets, we used Building and Rubble from Mill19 (Turki et al., 2022b), and Residence and Sci-Art from UrbanScene3D (Lin et al., 2022). For the synthetic dataset, we adopted Aerial, which represents a small city region from MatrixCity (Li et al., 2023). Following prior work (Liu et al., 2025), all images in MatrixCity were resized to a width of 1600 pixels. For a fair comparison on real-world datasets, we downsampled all images by a factor of four, consistent with previous methods.

Baselines. We compare our framework against state-of-the-art large-scale 3DGS methods, including CityGS (Liu et al., 2025), VastGS (Lin et al., 2024), and DOGS (Chen & Lee, 2024). We also include 3DGSâ , which follows the original 3DGS pipeline but extends training to 60k iterations, sets the densification interval to 200 iterations, and applies densification until 30k iterations. For VastGS and DOGS, we directly adopt the metrics reported in DOGS paper, where VastGS was evaluated without appearance modeling. For runtime analysis, we use an unofficial implementation of VastGS (also without appearance modeling) to enable a fairer comparison of training efficiency. For consistency, we denote both variants as VastGSâ  throughout our experiments. We do not report runtime results for DOGS, as its distributed training setup involves interconnect communication overhead, which is not directly comparable to our parallel but independent runtime setting.

<table><tr><td rowspan="2">Methods</td><td colspan="3">MatrixCity-Aerial</td><td colspan="3">Mill19</td><td colspan="3">UrbanScene3D</td></tr><tr><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td></tr><tr><td>3DGSâ </td><td>23.67</td><td>0.735</td><td>0.384</td><td>22.97</td><td>0.749</td><td>0.291</td><td>21.25</td><td>0.814</td><td>0.239</td></tr><tr><td>CityGS</td><td>27.46</td><td>0.865</td><td>0.204</td><td>23.66</td><td>0.796</td><td>0.237</td><td>21.70</td><td>0.825</td><td>0.221</td></tr><tr><td>Ours</td><td>27.74</td><td>0.875</td><td>0.186</td><td>23.87</td><td>0.797</td><td>0.240</td><td>21.33</td><td>0.826</td><td>0.213</td></tr></table>

Table 1: Quantitative comparison. Results on MatrixCity-Aerial, Mill19 (average of Rubble and Building), and UrbanScene3D (average of Residence and Sci-Art).

<table><tr><td rowspan="2">Methods</td><td colspan="2">MatrixCity-Aerial</td><td colspan="3">Mill19</td><td colspan="3">UrbanScene3D</td></tr><tr><td>C-PSNR (â) C-SSIM (â)</td><td>C-LPIPS (â)</td><td>C-PSNR (â)</td><td>C-SSIM (â)</td><td>C-LPIPS (â)</td><td>C-PSNR (â)</td><td>C-SSIM (â)</td><td>C-LPIPS (â)</td></tr><tr><td>VastGsâ </td><td>28.33</td><td>0.835</td><td>0.220 23.50</td><td>0.735</td><td>0.245</td><td>21.83</td><td>0.730</td><td>0.261</td></tr><tr><td>DOGS</td><td>28.58</td><td>0.847</td><td>0.219</td><td>24.26 0.762</td><td>0.231</td><td></td><td>23.18 0.772</td><td>0.232</td></tr><tr><td>Ours</td><td>28.91</td><td>0.879</td><td>0.187</td><td>24.68 0.795</td><td>0.241</td><td></td><td>23.55 0.832</td><td>0.213</td></tr></table>

Table 2: Quantitative comparison with color-corrected metrics (denoted by the âC-â prefix). Results on MatrixCity-Aerial, Mill19 (average of Rubble and Building), and UrbanScene3D (average of Residence and Sci-Art).

Metrics. We evaluate reconstruction quality using PSNR, SSIM, and LPIPS. Since some prior works, such as DOGS and VastGS, apply color correction before computing these metrics, we also adopt the color-corrected versions to ensure fair comparison. In contrast, when comparing against 3DGS and CityGS, which do not apply color correction, we report the standard PSNR, SSIM, and LPIPS values.

Efficiency metrics & runtime protocol. We use $T _ { \mathrm { c o a r s e } } , T _ { \mathrm { p a r t i t i o n } }$ , max $T _ { \mathrm { f i n e } } .$ , and $T _ { \mathrm { E 2 E } }$ (as defined in Equation 1) as our efficiency metrics. For all runtime analysis presented in this paper, we adopt the same block configurations as CityGS: 36 blocks for MatrixCity-Aerial, 20 for Building, 20 for Residence, 9 for Rubble, and 9 for Sci-Art. All runtimes are measured on identical compute hardware, with detailed specifications provided in Appendix A.1.

## 5.2 QUANTITATIVE RESULTS

From Table 1 and Table 2, our method achieves competitive or superior reconstruction quality across datasets. Compared to CityGS, performance is largely on par, with modest gains (â 1.0â1.02Ã) in PSNR/SSIM where applicable and consistently better LPIPS, at the cost of a slight PSNR drop on one dataset in exchange for improved perceptual quality. Compared to 3DGSâ , we observe consistent improvements, typically â 1.05â1.2Ã higher PSNR/SSIM and up to â¼ 2Ã lower LPIPS. With color-corrected metrics, our method also surpasses VastGSâ  and DOGS on most datasets, leading in C-PSNR and C-SSIM. Overall, these results demonstrate parity with CityGS while clearly outperforming VastGSâ , DOGS, and 3DGSâ . Additional quantitative results are provided in Appendix A.4.

## 5.3 LOAD BALANCE AND RUNTIME ANALYSIS

As shown in Table 3, our method consistently achieves the lowest coarse-stage runtime and slowestblock fine-stage runtime across all datasets, yields the best partition time on two of three datasets, and achieves the best end-to-end runtime on MatrixCity-Aerial and UrbanScene3D; on Mill19, Notably, although our $T _ { \mathrm { E 2 E } }$ on Mill19 is slightly longer than the reported VastGSâ  runtime (which omits $T _ { \mathrm { c o a r s e } } ) _ { \cdot }$ , our method delivers higher reconstruction qualityâsurpassing VastGSâ  on PSNR, SSIM, and LPIPS (see Table 2)âhighlighting a favorable qualityâlatency trade-off.

<table><tr><td rowspan="2">Methods</td><td colspan="4">MatrixCity-Aerial</td><td colspan="4"></td><td colspan="4">UrbanScene3D</td></tr><tr><td> $T _ { \mathrm { c o a r s e } }$ </td><td> $T _ { \mathrm { p a r t i t i o n } }$ </td><td> $\operatorname* { m a x } T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { E 2 E } }$ </td><td> $T _ { \mathrm { c o a r s e } }$ </td><td> $T _ { \mathrm { p a r t i t i o n } }$ </td><td> $\operatorname* { m a x } T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { E 2 E } }$ </td><td> $T _ { \mathrm { c o a r s e } }$ </td><td> $T _ { \mathrm { p a r t i t i o n } }$ </td><td> $\operatorname* { m a x } T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { E 2 E } }$ </td></tr><tr><td>3DGSâ </td><td>01:50</td><td></td><td></td><td>01:50</td><td>01:20</td><td></td><td></td><td>01:20</td><td>01:01</td><td></td><td></td><td>01:01</td></tr><tr><td>VastGSâ </td><td>â</td><td>00:48</td><td>01:13</td><td>02:01</td><td>â</td><td>00:05</td><td>00:42</td><td>00:47</td><td></td><td>00:17</td><td>00:40</td><td>00:57</td></tr><tr><td>CityGS</td><td>00:52</td><td>01:39</td><td>01:00</td><td>03:31</td><td>01:03</td><td>00:15</td><td>01:10</td><td>02:28</td><td>00:43</td><td>00:20</td><td>01:04</td><td>02:07</td></tr><tr><td>Ours</td><td>00:38</td><td>00:16</td><td>00:30</td><td>01:24</td><td>00:24</td><td>00:07</td><td>00:36</td><td>01:07</td><td>00:21</td><td>00:07</td><td>00:28</td><td>00:55</td></tr></table>

Table 3: End-to-end runtime comparison. A value of âââ indicates that the method does not include the corresponding stage.

<table><tr><td>FCS</td><td>LB-SP</td><td>VC</td><td>SD</td><td>MatrixCity-Aerial Max  $T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { p a r t i t i o n } }$ </td><td>Residence Max  $T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { p a r t i t i o n } }$ </td><td>Building Max  $T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { p a r t i t i o n } }$ </td></tr><tr><td>&gt;&gt;</td><td></td><td></td><td></td><td>01:00</td><td>00:14</td><td>01:01</td><td>00:04</td><td>01:06</td><td>00:03</td></tr><tr><td></td><td></td><td>â</td><td></td><td>00:52</td><td>00:14</td><td>00:47</td><td>00:04</td><td>00:45</td><td>00:03</td></tr><tr><td>â</td><td>â</td><td>â</td><td></td><td>00:47</td><td>00:16</td><td>00:36</td><td>00:07</td><td>00:34</td><td>00:08</td></tr><tr><td>â</td><td></td><td>â</td><td>â</td><td>00:32</td><td>00:14</td><td>00:33</td><td>00:04</td><td>00:39</td><td>00:03</td></tr><tr><td>V</td><td>â</td><td>â</td><td>V</td><td>00:30</td><td>00:16</td><td>00:30</td><td>00:07</td><td>00:30</td><td>00:08</td></tr></table>

Table 4: Ablation on model components. Evaluate the effectiveness of individual components: Fast Camera Selection (FCS), Load Balance-aware Scene Partition (LB-SP), Visibility Cropping (VC), and Selective Densification (SD).

## 5.4 ABLATION STUDIES

To assess the contribution of each component in our framework, we conduct ablation experiments on three representative datasets: MatrixCity-Aerial, Residence, and Building. We evaluate different combinations of four components: (1) Fast Camera Selection (FCS), which accelerates camera-toblock assignment with negligible accuracy loss; (2) Load Balance-aware Scene Partition (LB-SP), which redistributes Gaussians across blocks based on proxy load metrics to mitigate imbalance; (3) Visibility Cropping (VC), which prunes invisible Gaussians to reduce optimization time; and (4) Selective Densification (SD), which restricts densification to block regions. As shown in Table 4, LB-SP consistently reduces the worst-block fine-stage runtime max $T _ { \mathrm { f i n e } } \mathrm { : }$ configurations with LB-SP always outperform otherwise identical ones without it. Moreover, enabling all four components halves the worst-block fine-stage runtime compared to the FCS-only baseline $( \sim 0 1 { : } 0 0 \to \sim 0 0 { : } 3 0 )$ , corresponding to a â¼ 2Ã speedup in max $T _ { \mathrm { { f i n e } } }$ and substantially improved end-to-end efficiency. These results highlight that $\mathrm { L B } { - } \mathrm { S P } ^ { \prime } \mathrm { s }$ workload rebalancing complements the per-block reductions of VC and SD, yielding the largest cumulative runtime gains when combined.

## 6 CONCLUSION

In this paper, we present LoBE-GS, which addresses load balancing and efficiency in the parallel training of 3DGS models. At the core of LoBE-GS is a computational-load proxy that enables an optimization for the scene partition of a coarse 3DGS model. We further introduce fast camera selection to accelerate the scene partitioning, as well as visibility cropping and selective densification to reduce loading in each block. LoBE-GS achieves up to 2Ã training speedup over existing methods using coarse models for large-scale scene reconstruction while preserving the quality of the 3DGS models. In future work, we plan to experiment with larger and more complex scenes that would benefit from partitioning into a greater number of blocks for fine-training, and to explore the integration of level-of-detail (LoD) and 2DGS representations. We also plan to evaluate the framework on more diverse datasets, including those with sparse camera views in specific regions, and to investigate alternative partitioning strategies beyond the current grid-based approach.

## REFERENCES

Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54(10):105â112, 2011.

Maximilian Balandat, Brian Karrer, Daniel R. Jiang, Samuel Daulton, Benjamin Letham, Andrew Gordon Wilson, and Eytan Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. In Advances in Neural Information Processing Systems 33, 2020. URL http://arxiv.org/abs/1910.06403.

Yu Chen and Gim Hee Lee. DOGS: Distributed-oriented gaussian splatting for large-scale 3d reconstruction via gaussian consensus. Advances in Neural Information Processing Systems, 37: 34487â34512, 2024.

Jixuan Fan, Wanhua Li, Yifei Han, and Yansong Tang. Momentum-GS: Momentum gaussian selfdistillation for high-quality large scene reconstruction. arXiv preprint arXiv:2412.04887, 2024.

Yuanyuan Gao, Hao Li, Jiaqi Chen, Zhengyu Zou, Zhihang Zhong, Dingwen Zhang, Xiao Sun, and Junwei Han. CityGS-X: A scalable architecture for efficient and geometrically accurate largescale scene reconstruction, 2025. URL https://arxiv.org/abs/2503.23044.

Lihan Jiang, Kerui Ren, Mulin Yu, Linning Xu, Junting Dong, Tao Lu, Feng Zhao, Dahua Lin, and Bo Dai. Horizon-GS: Unified 3d gaussian splatting for large-scale aerial-to-ground scenes. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26789â26799, 2025.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splat- Â¨ ting for real-time radiance field rendering. ACM Transactions on Graphics (SIGGRAPH), 42(4): 1â14, 2023.

Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics, 43(4), July 2024. URL https://repo-sam. inria.fr/fungraph/hierarchical-3d-gaussians/.

Ruilong Li, Sanja Fidler, Angjoo Kanazawa, and Francis Williams. NeRF-XL: Scaling nerfs with multiple GPUs. In European Conference on Computer Vision (ECCV), 2024.

Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. MatrixCity: A large-scale city dataset for city-scale neural rendering and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3205â3215, 2023.

Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, and Wenming Yang. VastGaussian: Vast 3d gaussians for large scene reconstruction. In CVPR, 2024.

Liqiang Lin, Yilin Liu, Yue Hu, Xingguang Yan, Ke Xie, and Hui Huang. Capturing, reconstructing, and simulating: the urbanscene3d dataset. In ECCV, 2022.

Yang Liu, Chuanchen Luo, Zhongkai Mao, Junran Peng, and Zhaoxiang Zhang. CityGaussianV2: Efficient and geometrically accurate reconstruction for large-scale scenes. arXiv preprint arXiv:2411.00771, 2024.

Yang Liu, Chuanchen Luo, Lue Fan, Naiyan Wang, Junran Peng, and Zhaoxiang Zhang. CityGaussian: Real-time high-quality large-scale scene rendering with gaussians. In European Conference on Computer Vision, pp. 265â282. Springer, 2025.

Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-GS: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20654â20664, 2024.

Miles Macklin. Warp: A high-performance python framework for gpu simulation and graphics. https://github.com/nvidia/warp, March 2022. NVIDIA GPU Technology Conference (GTC).

Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3DGS: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers, SA â24, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400711312. doi: 10.1145/3680528.3687694. URL https://doi.org/10.1145/3680528.3687694.

Zhenxing Mi and Dan Xu. Switch-NeRF: Learning scene decomposition with mixture of experts for large-scale neural radiance fields. In International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id=PQ2zoIZqvm.

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), pp. 405â421. Springer, 2020.

KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi Koohpayegani, and Hamed Pirsiavash. CompGS: Smaller and faster gaussian splatting with vector quantization. ECCV, 2024.

Miles Olson, Elizabeth Santorella, Louis C. Tiao, Sait Cakmak, David Eriksson, Mia Garrard, Sam Daulton, Maximilian Balandat, Eytan Bakshy, Elena Kashtelyan, Zhiyuan Jerry Lin, Sebastian Ament, Bernard Beckerman, Eric Onofrey, Paschal Igusti, Cristian Lara, Benjamin Letham, Cesar Cardoso, Shiyun Sunny Shen, Andy Chenyuan Lin, and Matthew Grange. Ax: A Platform for Adaptive Experimentation. In AutoML 2025 ABCD Track, 2025.

Panagiotis Papantonakis, Georgios Kopanas, Bernhard Kerbl, Alexandre Lanvin, and George Drettakis. Reducing the memory footprint of 3d gaussian splatting. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 7(1), May 2024. URL https://repo-sam. inria.fr/fungraph/reduced_3dgs/.

Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-GS: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024.

Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In ACM siggraph 2006 papers, pp. 835â846. 2006.

Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-NeRF: Scalable large scene neural view synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8248â8258, 2022.

Jiadong Tang, Yu Gao, Dianyi Yang, Liqi Yan, Yufeng Yue, and Yi Yang. Dronesplat: 3d gaussian splatting for robust 3d reconstruction from in-the-wild drone imagery. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 833â843, 2025.

Haithem Turki, Deva Ramanan, and Mahadev Satyanarayanan. Mega-NERF: Scalable construction of large-scale nerfs for virtual fly-throughs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12922â12931, June 2022a.

Haithem Turki, Deva Ramanan, and Mahadev Satyanarayanan. Mega-NeRF: Scalable construction of large-scale nerfs for virtual fly-throughs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12922â12931, June 2022b.

Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1â17, 2025.

Zhensheng Yuan, Haozhi Huang, Zhen Xiong, Di Wang, and Guanghua Yang. Robust and efficient 3d gaussian splatting for urban scene reconstruction. arXiv preprint arXiv:2507.23006, 2025.

Hexu Zhao, Haoyang Weng, Daohan Lu, Ang Li, Jinyang Li, Aurojit Panda, and Saining Xie. On scaling up 3d gaussian splatting training, 2024. URL https://arxiv.org/abs/2406. 18533.

## A APPENDIX

## A.1 IMPLEMENTATION DETAILS

Scene Partition. Bayesian Optimization (BO) with Gaussian Process (GP) surrogate modeling (Section 4.1) is implemented using Ax (Olson et al., 2025) with BoTorch (Balandat et al., 2020) backend for GPU-accelerated optimization. Block load computation (Section 4.1), fast camera selection (Section 4.2), and visibility cropping (Section 4.3), are implemented in NVIDIA Warp (Macklin, 2022), which enables kernel-based programming in Python with performance comparable to native CUDA. In preliminary benchmark on the MatrixCity-Aerial scene, the Warp implementation on a single GPU achieves speedups of approximately 450Ã over sequential CPU code and 5Ã over Numba-parallelized CPU code executed on 128 logical CPU cores. These performance improvements are enabled by low-level optimizations not exposed in PyTorch, including bitsets, atomics, and Warp tiles, with the latter providing functionality analogous to shared memory and cooperative groups in CUDA C++.

3DGS Training. The coarse-training stage employs the Sparse Adam optimizer to accelerate training, which has minimal impact on final performance. In contrast, the fine-training stage continues to use the standard Adam optimizer, as Sparse Adam was found to degrade performance in this setting. Aside from selective densification, fine-training details follows the standard vanilla 3DGS procedure (as in CityGS), with additional code-level optimizations through the gsplat library (Ye et al., 2025) and fused-ssim (Mallick et al., 2024) for SSIM loss evaluation.

Experimental Setup. For consistency, all CityGS runtimes reported in Section 3 are measured using a modified version of CityGS with gsplat, fused-ssim, and visibility cropping enabled. Moreover, since selective densification shortens per-block fine-training time, we disable it in LoBE-GS when reporting results in Section 3. A comparison against the unmodified CityGS with the full LoBE-GS pipeline (including selective densification) is provided in Figure A.1. In Section 5, since the official implementation of VastGSâ  is unavailable, we report performance results based on an unofficial implementation available at https://github.com/kangpeilun/VastGaussian.

System Configuration. All experiments are conducted on a cluster consisting of 10 compute nodes, each equipped with 8 NVIDIA L40 GPUs and 128 logical CPU cores (Intel Xeon Platinum 8362), amounting to a total of 80 GPUs across the cluster. The fine-training stage is parallelized across blocks with one GPU per block, whereas all other stages are executed on a single GPU.

Reproducibility. Source code along with a pre-built Docker image will be released upon paper acceptance to ensure reproducibility. All reported runtimes are measured within the Docker environment to eliminate potential discrepancies caused by library mismatches or system-level variations.

Declaration of LLM usage. Large Language Models (LLMs) are only used for editing grammar.

## A.2 LOAD BALANCE ACROSS DATASETS

As shown in Figure A.1, our method yields a noticeably more uniform per-block workload distribution across the evaluated datasets. In particular, the load balance-aware partitioning combined with visibility cropping and selective densification systematically reduces the worst-case per-block fine-stage runtime, i.e., the slowest straggler blocks are much faster than under the baselines. This reduction in the tail of the runtime distribution leads to fewer stragglers and improved end-to-end efficiency. These gains are consistent across datasets, demonstrating the robustness of our partitioning strategy in mitigating workload skew.

<!-- image-->  
Figure A.1: Comparison of load balance and partitioning between CityGS (Left) and LoBE-GS (Right) across five datasets: Building, Rubble, Residence, Sci-Art, and MatrixCity-Aerial.

## A.3 ADDITIONAL CORRELATION ANALYSIS ACROSS DATASETS

In Section 3, we observed a strong correlation with $G _ { \mathrm { v i s } } ^ { ( b ) }$ when using the original CityGS pipeline combined with visibility cropping. In the fine-training stage, however, both visibility cropping and selective densification were enabled to further reduce the per-block load in LoBE-GS. To ensure that the correlation still remains strong under these settings, we additionally conducted experiments with both visibility cropping and selective densification enabled.

$$
\begin{array} { c c c c c c c c c c c c c c c c c } { { \cdots \bullet \cdots } } & { { C ^ { ( b ) } ( r = 0 . 2 2 ) } } & { { \cdots \bullet \cdots } } & { { G _ { \mathrm { { v i s } } } ^ { ( b ) } ( r = 0 . 9 1 ) } } & { { } } & { { } } & { { } } & { { } } & { { G _ { \mathrm { { b l k } } } ^ { ( b ) } ( r = 0 . 7 6 ) } } & { { \cdots \bullet \cdots } } & { { G _ { \mathrm { { u v g } } } ^ { ( b ) } ( r = 0 . 0 7 ) } } \end{array}
$$

<!-- image-->

<!-- image-->  
Figure A.2: Correlation between per-block training time and block-level statistics under CityGSâs partitioning with both visibility cropping and selective densification enabled. (a) Plots camera count C(b). (b) Plots of Gaussian-based measures $( G _ { \mathrm { b l k } } ^ { ( b ) } , G _ { \mathrm { v i s } } ^ { ( b ) } , G _ { \mathrm { a v g - v i s } } ^ { ( b ) } ) . G _ { \mathrm { v i s } } ^ { ( b ) }$ yields the strongest and most consistent correlation across datasets even when selective densification is enabled.

## A.4 EXTENDED QUANTITATIVE COMPARISON

This appendix collects the full numerical results omitted from the main text for space reasons. Table A.1 and Table A.2 report per-dataset qualitative results and their color-corrected counterparts to ensure fair comparison with baselines that apply post-hoc color alignment. Table A.3 summarizes end-to-end timing $T _ { \mathrm { E 2 E } } ( T _ { \mathrm { c o a r s e } } , T _ { \mathrm { p a r t i t i o n } } , \operatorname* { m a x } T _ { \mathrm { f i n e } } )$ and Table A.4 compares load balance aware data partition times for CPU vs GPU implementations across five datasets.

<table><tr><td rowspan="2">Methods</td><td colspan="3">Residence</td><td colspan="3">Rubble</td><td colspan="3">Building</td><td colspan="3">Sci-Art</td></tr><tr><td>PSNR ()</td><td>SSIM (â)LPIPS (â)</td><td></td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td></tr><tr><td>3DGS</td><td>21.44</td><td>0.791</td><td>0.236</td><td>25.47</td><td>0.777</td><td>0.277</td><td>20.46</td><td>0.720</td><td>0.305</td><td>21.05</td><td>0.837</td><td>0.242</td></tr><tr><td>CityGS</td><td>22.00</td><td>0.813</td><td>0.211</td><td>25.77</td><td>0.813</td><td>0.228</td><td>21.55</td><td>0.778</td><td>0.246</td><td>21.39</td><td>0.837</td><td>0.230</td></tr><tr><td>Ours</td><td>21.41</td><td>0.808</td><td>0.206</td><td>25.78</td><td>0.811</td><td>0.234</td><td>21.96</td><td>0.783</td><td>0.245</td><td>21.24</td><td>0.843</td><td>0.219</td></tr></table>

Table A.1: Quantitative comparison on Mill19 and UrbanScene3D datasets. We report PSNR, SSIM, and LPIPS.

<table><tr><td rowspan="2">Methods</td><td colspan="3">Residence</td><td colspan="3">Rubble</td><td colspan="3">Building</td><td colspan="3">Sci-Art</td></tr><tr><td>C-PSNR (â)</td><td>C-SSIM (â)</td><td>C-LPIPS (â)</td><td>C-PSNR (â)</td><td>C-SSIM (â)</td><td>C-LPIPS (â)</td><td>C-PSNR (â)</td><td>C-SSIM (â)</td><td>C-LPIPS (â)</td><td>C-PSNR (â) C-SSIM (â)</td><td></td><td>C-LPIPS (â)</td></tr><tr><td>VastGS</td><td>21.01</td><td>0.699</td><td>0.261</td><td>25.20</td><td>0.742</td><td>0.264</td><td>21.80</td><td>0.728</td><td>0.225</td><td>22.64</td><td>0.761</td><td>0.261</td></tr><tr><td>DoGS</td><td>21.94</td><td>0.740</td><td>0.244</td><td>25.78</td><td>0.765</td><td>0.257</td><td>22.73</td><td>0.759</td><td>0.204</td><td>24.42</td><td>0.804</td><td>0.219</td></tr><tr><td>Ours</td><td>22.94</td><td>0.822</td><td>0.206</td><td>26.55</td><td>0.810</td><td>0.235</td><td>22.80</td><td>0.779</td><td>0.247</td><td>24.71</td><td>0.853</td><td>0.217</td></tr></table>

Table A.2: Quantitative comparison on Mill19 and UrbanScene3D datasets. We report colorcorrected (denoted by the âC-â prefix) PSNR, SSIM, and LPIPS.

<table><tr><td rowspan="2">Methods</td><td rowspan="2"> $T _ { \mathrm { c o a r s e } }$ </td><td colspan="3">Residence</td><td colspan="4">Rubble</td><td colspan="4">Building</td><td colspan="4">Sci-Art</td></tr><tr><td> $T _ { \mathrm { p a r i t i o n } }$ </td><td> $\mathbf { M a x } T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { E 2 E } }$ </td><td> $T _ { \mathrm { c o a r s e } }$ </td><td> $T _ { \mathrm { p a r i t i o n } }$ </td><td> $\mathbf { M a x } T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { E 2 E } }$ </td><td> $T _ { \mathrm { c o a r s e } }$ </td><td> $T _ { \mathrm { p a r i t i o n } }$ </td><td> $\mathbf { M a x } T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { E 2 E } }$ </td><td> $T _ { \mathrm { c o a r s e } }$ </td><td> $T _ { \mathrm { p a r i t i o n } }$ </td><td> $\mathbf { M a x } T _ { \mathrm { f i n e } }$ </td><td> $T _ { \mathrm { E 2 E } }$ </td></tr><tr><td>3DGS</td><td>01:22</td><td></td><td>â</td><td>01:22</td><td> $0 1 { : } 1 0$ </td><td>â</td><td>â</td><td>01:10</td><td> $0 1 { : } 3 0$ </td><td>â</td><td></td><td>01:30</td><td> $0 0 { : } 4 0$ </td><td></td><td>â</td><td>00:40</td></tr><tr><td>VastGS</td><td>â</td><td>00:08</td><td>00:49</td><td>00:57</td><td></td><td>00:04</td><td>00:39</td><td>00:43</td><td></td><td>00:05</td><td>00:44</td><td>00:49</td><td></td><td>00:25</td><td>00:31</td><td>00:56</td></tr><tr><td>CityGS</td><td>00:43</td><td>00:31</td><td>01:22</td><td>02:36</td><td>01:06</td><td>00:09</td><td>01:14</td><td>02:29</td><td>00:59</td><td>00:21</td><td>01:06</td><td>02:26</td><td>00:42</td><td>00:08</td><td>00:45</td><td>01:35</td></tr><tr><td>Ours</td><td>00:26</td><td>00:08</td><td>00:30</td><td>01:04</td><td>00:23</td><td>00:05</td><td>00:41</td><td>01:09</td><td>00:25</td><td>00:08</td><td>00:30</td><td>01:03</td><td>00:16</td><td>00:05</td><td>00:26</td><td>00:47</td></tr></table>

Table A.3: End-to-end runtime comparison on Mill19 and UrbanScene3D dataset. For each dataset we report coarse time $T _ { \mathrm { c o a r s e } } ,$ partition time $T _ { \mathrm { p a r t i t i o n } } ,$ , max fine time (Max $T _ { \mathrm { f i n e } } ) _ { \mathrm { \ell } }$ , and total $T _ { \mathrm { E 2 E } }$ . A value $\mathrm { o f } \ ^ { 6 6 } - \mathrm { } ^ { 5 }$ indicates that the method does not include the corresponding stage.

<table><tr><td>Methods</td><td>| MatrixCity-Aerial | Residence</td><td></td><td>Rubble</td><td>Building</td><td>SciArt</td></tr><tr><td>LB-SP (CPU)</td><td>00:47</td><td>00:18</td><td>00:03</td><td>00:15</td><td>00:10</td></tr><tr><td>LB-SP (GPU)</td><td>00:16</td><td>00:06</td><td>00:05</td><td>00:06</td><td>00:05</td></tr></table>

Table A.4: Partition time (hh:mm) comparison across CPU and GPU methods for five datasets.