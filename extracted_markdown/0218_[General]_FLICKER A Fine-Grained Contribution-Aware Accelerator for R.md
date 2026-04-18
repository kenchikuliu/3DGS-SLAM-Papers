# FLICKER: A Fine-Grained Contribution-Aware Accelerator for Real-Time 3D Gaussian Splatting

Wenhui Ou1, Zhuoyu Wu2, Yipu Zhang1, Dongjun Wu1, Freddy Ziyang Hong1, C. Patrick Yue1,\*

1Department of Electronic and Computer Engineering, The Hong Kong University of Science and Technology 2School of IT, Monash University, Malaysia Campus

AbstractâRecently, 3D Gaussian Splatting (3DGS) has become a mainstream rendering technique for its photorealistic quality and low latency. However, the need to process massive noncontributing Gaussian points makes it struggle on resource-limited edge computing platforms and limits its use in next-gen AR/VR devices. A contribution-based prior skipping strategy is effective in alleviating this inefficiency, but the associated contributiontesting workload becomes prohibitive when it is further applied to the edge. In this paper, we present FLICKER, a contributionaware 3DGS accelerator that leverages a hardwareâsoftware co-design framework, including adaptive leader pixels, pixelrectangle grouping, hierarchical Gaussian testing, and mixedprecision architecture, to achieve near-pixel-level, contributiondriven rendering with minimal overhead. Experimental results show that our design achieves up to 1.5Ã speedup, 2.6Ã energy efficiency improvement, and 14% area reduction over a state-ofthe-art accelerator. Meanwhile, it also achieves 19.8Ã speedup and 26.7Ã energy efficiency compared with a common edge GPU. Index Termsâ3DGS, Contribution-Aware, Accelerator

## I. INTRODUCTION

Advances in photorealistic novel view synthesis (NVS) have significantly enhanced the immersive experience in augmented and virtual reality (AR/VR) applications [1]. Recently, 3D Gaussian Splatting (3DGS) [2], [3] has emerged as a leading NVS technique for its outstanding rendering speed. This makes it a promising solution for next-generation AR/VR systems. However, such real-time capability is largely limited to powerful cloud or desktop-level GPUs, while edge devices still suffer from severe performance loss and energy constraints [4].

The limitation stems from its inherent principle. Specifically, 3DGS explicitly represents a scene with a set of anisotropic Gaussians, whose number often exceeds millions in real-world scenarios [5]. To avoid excessive memory footprint, the rendering of each frame is typically split into tiles, so that only the Gaussians that might contribute to the current tile are accessed instead of the entire set. However, the large tile size and over-inclusive Gaussian testing cause pixels to process a substantial number of unnecessary Gaussians [7]. This not only wastes computational resources, but more importantly, disrupts the efficient alignment of dataflow across parallel hardware units, leading to low hardware utilization. For edge devices [6] with limited computing units, such inefficiency is particularly detrimental to both performance and energy efficiency.

To address the aforementioned challenge, several works [7]â [10] try to refine the identification of contributing Gaussians by considering more Gaussian features. These features help narrow the candidate contribution region of Gaussians, thereby reducing the pixels that unnecessarily process them. Nevertheless, striving to make the candidate region exactly match the true contribution area will inevitably incurs significant computational complexity. Instead, [11] adopts a GPU-based Contribution-Aware Test (CAT) by directly evaluating each Gaussianâs contribution to a leader pixel within a pixel group before rendering. If the contribution is negligible, the entire pixel group can skip processing that Gaussian. This method alone achieves a 1.3Ã speedup for the overall system.

<!-- image-->

<!-- image-->  
(a) Rendering Speed  
() Kernel Hard. Util.  
Fig. 1: (a) Rendering speed and (b) hardware utilization of the rendering kernel in vanilla 3DGS, profiled on RTX 3090 [13] and Jetson Xavier NX [14]. In (b), CU denotes the utilization of compute units (i.e., GPU SMs), reflecting overall computation activity, while FP indicates the achieved FP32 performance relative to the device peak performance.

Despite its effectiveness, applying this contribution-aware strategy to edge designs faces several challenges. First, the overhead of CAT is substantial. For instance, the 2 Ã 2 pixel group size adopted in [11] incurs significant computational overhead, as each four-pixel group requires a contribution test of a leader pixel, and the total number of such tests scales linearly with image resolution, making it prohibitively expensive for edge devices. Second, enabling independent Gaussian skipping for each pixel group requires dedicated memory allocation per group, which will lead to significant onchip memory overhead. Third, the leader-pixel-based approach makes rendering quality highly sensitive to pixel group size. Simply reducing the number of leader pixels by adopting larger pixel groups will significantly increase the risk of missing contributing Gaussians, resulting in noticeable image degradation.

To address the above challenge, this work make the following contribution:

â¢ We propose FLICKER, a fine-grained contribution-aware accelerator that leverages hardware-software co-design to achieve accurate skipping of non-contributing Gaussians at nearly pixel-scale granularity, facilitating real-time 3DGS rendering on the edge.

â¢ We introduce an adaptive leader-pixel scheme that dynamically reduces the number of leader pixels based on Gaussian shapes. Furthermore, we propose a novel batch processing technique that organizes leader pixels into rectangular groups. By sharing intermediate results within each group, the overhead is nearly halved without compromising image quality (Sec. III).

â¢ We introduce a two-stage hierarchical testing flow that effectively filters Gaussians with reduced computation and memory overhead. Moreover, we design a mixedprecision CAT engine, tightly integrated with the rendering pipeline, to minimize area overhead and effectively hide CAT latency (Sec. IV).

â¢ Experimental results show that FLICKER achieves up to 1.5Ã speedup and 2.6Ã energy efficiency compared to the baseline design, while requiring 14% less area (Sec. V).

## II. BACKGROUND AND MOTIVATION

## A. 3DGS with Contribution-Aware Rendering

3DGS Rendering Pipeline. The 3DGS rendering process begins with a set of Gaussian ellipsoids defined by differentiable parameters. For a given camera pose, generating an image frame from these Gaussians involves three main steps, as shown in Fig. 2(a). In Step(1), Gaussians within the view frustum are projected onto the image plane, generating 2D features such as mean $( \mu ^ { \prime } )$ , covariance $( \Sigma ^ { \prime } )$ , and color (câ²). Since rendering proceeds tile by tile, an intersection test is performed for each tile to identify Gaussians that may contribute. These Gaussians are then copied as needed to form a dedicated list for each tile. In Step (2), the Gaussians in each list are then sorted by their distance from the camera (i.e., depth), arranged from near to far. With the sorted list (containing N Gaussians), all pixels within a tile are rendered in a uniform manner by iterating over the Gaussians in the list (Step (3)). For each Gaussian $G _ { i }$ , the âcontributionâ to pixels is first computed:

$$
\alpha _ { i } = o _ { i } \cdot e ^ { - \frac { 1 } { 2 } ( p - \mu _ { i } ^ { \prime } ) ^ { \top } \Sigma _ { i } ^ { ' - 1 } ( p - \mu _ { i } ^ { \prime } ) }\tag{1}
$$

where $o _ { i } \in [ 0 , 1 ]$ is the opacity and $p$ is the pixel coordinate. Gaussians with $\begin{array} { l l l } { \alpha _ { i } } & { < } & { { \frac { 1 } { 2 5 5 } } } \end{array}$ are considered no contribution and skipped. Otherwise, Î±i is further used to compute the pixel color $\begin{array} { r } { c = \sum _ { i } ^ { N } T _ { i } c _ { i } ^ { \prime } \alpha _ { i } } \end{array}$ . With transmittance defined as $\begin{array} { r } { T _ { i } = \prod _ { i = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } \end{array}$ , rendering of the current tile can terminate early if the transmittance of all pixels falls below a predefined threshold.

Bounding Box Intersection Test. Over-inclusive intersection tests impose significant rendering overhead. As shown in Fig. 2(b), vanilla 3DGS employs a simple Axis-Aligned Bounding Box (AABB) test [2]. For Gaussians with anisotropic shapes, the 3Ï rule defines their effective boundaries, which are then replaced by a bounding box aligned with the coordinate axes. In this toy example, the AABB test marks all tiles as intersected, leading to substantial redundant computation. GSCore [7], an ASIC-level 3DGS accelerator, adopts an Oriented Bounding Box (OBB) technique to better fit the anisotropic shapes of Gaussians. Moreover, by dividing tiles into subtiles with $8 \times 8$ pixels, the intersected region is significantly reduced. Nevertheless, the region still does not precisely match the Gaussianâs actual contribution.

<!-- image-->  
Fig. 2: (a) Overall rendering pipeline of 3DGS [2], and (b) comparison of three intersection methodsâAABB in vanilla 3DGS, OBB in GSCore [7] and proposed Mini-Tile CAT.

Mini-Tile Contribution-Aware Test. Building upon the CAT [11] discussed in Sec. I, we propose Mini-Tile CAT, which uses multiple leader pixels to accurately capture Gaussian contributions within a $, ~ 4 \times 4$ mini-tile. A mini-tile is marked intersected if any leader pixels is contributed by the Gaussian. Combined with customized optimization, this method can enable accurate intersection while largely reducing CAT overhead (details in Sec. III). As shown in Fig. 2(b), the intersected region closely aligns with the Gaussianâs true contribution boundary, thereby ensuring higher rendering efficiency.

## B. 3DGS Profiling and Strategy Analysis

We adopt a two-stage analysis method to investigate performance bottlenecks and identify optimization opportunities: First, we profile the 3DGS rendering pipeline on two GPUs: a desktop-level GPU, RTX 3090 [13], and an edge GPU, XNX [14]. Since XNX shares similar hardware specifications to edge GPUs used in advanced VR headsets [15], its profiling results can better reflect the bottlenecks in practical edge scenarios. We obtain a detailed GPU performance breakdown using Nsight Compute [12] and profiling is conducted with datasets from Mip-NeRF360 [16]. Second, we further conduct an in-depth analysis of the introduced intersection methods to quantify the expected benefits and overheads of the Mini-Tile CAT, thereby guiding the subsequent design.

3DGS Profiling Result. As shown in Fig. 1(a), 3DGS achieves real-time rendering on desktop GPUs, with average FPS exceeding 100. In contrast, on edge GPUs, the FPS drops sharply to around 5 per scene, highlighting such edge devices struggle to handle the 3DGSâs workload. To understand this discrepancy, we further analyzed the hardware utilization of the rendering step in 3DGS, which accounts for a significant portion of GPU kernel execution time, often exceeding 60% [7], [17], [18]. As shown in Fig. 1(b), the overall compute unit utilization on the GPU, calculated via SM Core throughput [23], reaches an average of 85%. However, the average floatingpoint utilization is merely 29%, which corresponds to the core computations in the rendering step. The result indicates that the already limited computation resources on edge GPUs are underutilized, exacerbating performance degradation. This inefficiency stems from the fact that, during rendering, certain pixels within the same tile are skipped due to their negligible Gaussian contributions, which causes warp divergence. Moreover, the over-inclusive intersection test further amplifies this effect, ultimately leading to poor performance.

<!-- image-->  
Fig. 3: Mini-Tile CAT algorithm optimization: (a) adaptive leader pixels, and (b) pixel-rectangle grouping. In (a), the PSNR of vanilla 3DGS is 25.56, while the Uniform-Dense mode shows negligible loss. Although smooth Gaussians account for 43%, the Smooth-Focused mode achieves higher PSNR, indicating that the contribution of smooth Gaussians is more significant in this case. In (b), the Mini-Tile CAT of a Pixel Rectangle (PR) can be simplified by exploiting its coordinate symmetry.

Strategy Analysis. Based on profiling, we further compare introduced intersection methods on a real-world scene to guide Mini-Tile CAT implementation on edge hardware. In this analysis, each mini-tile is initially assigned 4 leader pixels. The results, shown in Fig. 4, lead to the following observations: First, Mini-Tile CAT demonstrates a clear advantage in filtering Gaussians at the finegrained 4 Ã 4 tile level, i.e., mini-tile. Compared to AABB on a 16 Ã 16 tile, Mini-Tile CAT reduces the number of Gaussians each pixel must process to only 10%, the lowest among all tested methods. This underscores its potential to substantially cut rendering workload and alleviate underutilization of parallel hardware units caused by pixel skipping. Second, although Mini-Tile CAT reduces unnecessary Gaussian processing, it incurs significant computational overhead because leader pixels must be tested on many Gaussians. Moreover, since Mini-Tile CAT evaluates far more Gaussians than are ultimately forwarded to the rendering stage, it can easily become a pipeline bottleneck, causing downstream stalls. This highlights the need for an efficient architectural support of Mini-Tile CAT. Third, smaller tile sizes more effectively reduce redundant Gaussian processing but at the cost of higher memory overhead due to increased duplicated Gaussians. For instance, reducing the tile size from 16 Ã 16 to 4 Ã 4 pixels increases the total number of duplicated Gaussians to 4Ã the original, highlighting the need for an efficient hierarchical intersection strategy and memory allocation.

<!-- image-->  
Fig. 4: Per-pixel processed Gaussians across intersection strategies and duplicate Gaussians across tile sizes.

## III. ALGORITHM OPTIMIZATION

In this section, we introduce two algorithmic strategies to effectively reduce the computation overhead of Mini-Tile CAT from two perspectives: reducing (1) the number of required leader pixels, and (2) the per-leader-pixel CAT overhead.

## A. Adaptive Leader Pixels

Basic Mode. We begin by defining two sampling settings for Mini-Tile CAT: Dense Sampling, using four corner pixels per mini-tile, and Sparse Sampling, using two diagonal pixels. From these, we can derive two modes for Mini-Tile CAT: Uniform-Dense, where all Gaussians use Dense Sampling, and Uniform-Sparse, where all use Sparse Sampling. As shown in Fig. 3(a), Uniform-Dense captures most contributing Gaussians with only a negligible PSNR drop (0.08 dB) compared to vanilla 3DGS, while Uniform-Sparse, though reducing leader pixels by half, causes noticeable quality degradation.

Adaptive Mode. To combine the strengths of both uniform modes, we propose an adaptive strategy that dynamically switches the sampling based on Gaussian shape. All Gaussians are first classified as Smooth (axis ratio < 3) or Spiky (axis ratio â¥ 3). Smooth Gaussians use Dense Sampling for broader coverage (Smooth-Focused mode), while spiky Gaussians use Sparse Sampling to save leader pixels. As shown in Fig. 3(a), this adaptive mode reduces PSNR loss by 73% compared to Uniform-Sparse, while retaining 57% of its leaderpixel savings. Notably, if spiky Gaussians carry more critical visual details, the strategy can be switched to a Spiky-Focused mode, in which Dense Sampling is applied to spiky Gaussians.

## B. Pixel-Rectangle Grouping

CAT for Single Pixels. As discussed in Sec. II, the contribution of a Gaussian to a leader pixel is calculated using Eq. 1. The result is then compared to a threshold $\textstyle ( \mathbf { e . g . , } \alpha < { \frac { 1 } { 2 5 5 } } )$ to determine whether the Gaussian can be skipped. A straightforward way for CAT is to design a dedicated Alpha Culling Unit (ACU) [7], [17], [18] to test each pixel individually. However, this approach incurs significant computation overhead, especially when testing multiple leader pixels across dense mini-tiles.

CAT for Pixel Rectangles. To reduce the average CAT overhead for per leader pixel, we first simplify the Eq. 1 as follows:

$$
\ln ( 2 5 5 \cdot o ) > - \frac { 1 } { 2 } ( p - \mu ^ { \prime } ) ^ { \top } \Sigma ^ { ' - 1 } ( p - \mu ^ { \prime } )\tag{2}
$$

In the inequality, the left-hand side term, ln(255 Â· o), is identical for all leader pixels tested against the same Gaussian, and therefore only needs to be computed once and shared. To reduce the overhead for computing the right-hand side, i.e., the Gaussian Weight E, we propose a Pixel-Rectangle Test Unit (PRTU), which evaluates the contribution of a Gaussian to a group of four leader pixels arranged in a rectangular Pixel Rectangle (PR). Within each PR, the two off-diagonal corner pixels have symmetric coordinates relative to the main-diagonal pixels, allowing the intermediate results from the main-diagonal pixels to be reused for computing the off-diagonal pixels. The pseudocode for processing a PR is provided in Alg. 1.

Algorithm 1 Pixel-Rectangle Gaussian Weight Computation   
Input: Gaussian mean $\mu ^ { \prime } ,$ conic entries $\Sigma _ { x x } ^ { \prime - 1 } , \Sigma _ { y y } ^ { \prime - 1 } , \Sigma _ { x y } ^ { \prime - 1 } ,$   
Main diagonal pixel coordinates ptop, pbot (correspond to   
p0 and $p _ { 3 }$ in a PR).   
Output: Gaussian weight $ { E _ { 0 } } ,  { E _ { 1 } } ,  { E _ { 2 } } ,  { E _ { 3 } }$   
1: $\Delta _ { \mathrm { t o p } }  \mathbf { p } _ { \mathrm { t o p } } - \mu ^ { \prime } \Delta _ { \mathrm { b o t } }  \mathbf { p } _ { \mathrm { b o t } } - \mu ^ { \prime }$   
2: $s _ { \mathrm { t o p } } ^ { x } \dot { \overline { { \mathbf { \alpha } } } } = 0 . 5 \cdot \Delta _ { \mathrm { t o p } , x } ^ { 2 } \cdot \Sigma _ { x x } ^ { \prime - 1 } \quad s _ { \mathrm { t o p } } ^ { y } = 0 . 5 \cdot \Delta _ { \mathrm { t o p } , y } ^ { 2 } \cdot \Sigma _ { y y } ^ { \prime - 1 }$   
3: $\begin{array} { r l } { s _ { \mathrm { b o t } } ^ { \dddot { x } ^ { \star } } = 0 . 5 \cdot \Delta _ { \mathrm { b o t } , x } ^ { \dddot { 2 } ^ { \star \star } } \cdot \Sigma _ { x x } ^ { \dddot { \prime } - 1 } } & { { } s _ { \mathrm { b o t } } ^ { y ^ { \star } } = 0 . 5 \cdot \Delta _ { \mathrm { b o t } , y } ^ { \dddot { 2 } ^ { \star \star } , y } \cdot \Sigma _ { y y } ^ { \dddot { \prime } - 1 } } \end{array}$   
4: $t _ { 0 } = \Delta _ { \mathrm { t o p } , x } \cdot \Delta _ { \mathrm { t o p } , y } \cdot \Sigma _ { x y } ^ { \prime - 1 } \quad t _ { 1 } = \Delta _ { \mathrm { b o t } , x } \cdot \Delta _ { \mathrm { t o p } , y } ^ { \sim } \cdot \Sigma _ { x y } ^ { \prime - 1 } .$   
5: $t _ { 2 } = \Delta _ { \mathrm { t o p } , x } \cdot \Delta _ { \mathrm { b o t } , y } \cdot \Sigma _ { x y } ^ { \prime - 1 } ~ t _ { 3 } = \Delta _ { \mathrm { b o t } , x } \cdot \Delta _ { \mathrm { b o t } , y } \cdot \Sigma _ { x y } ^ { \prime - 1 }$   
6: $\begin{array} { r } { E _ { 0 } = s _ { \mathrm { t o p } } ^ { x } + s _ { \mathrm { t o p } } ^ { y } + t _ { 0 } \quad E _ { 1 } = s _ { \mathrm { b o t } } ^ { x } + s _ { \mathrm { t o p } } ^ { y } + t _ { 1 } } \end{array}$   
7: $E _ { 2 } = s _ { \mathrm { t o p } } ^ { x } + s _ { \mathrm { b o t } } ^ { y } + t _ { 2 }$ $E _ { 3 } = s _ { \mathrm { b o t } } ^ { x } + s _ { \mathrm { b o t } } ^ { y } + t _ { 3 }$

Compared to the ACU that tests pixels individually, our pixel-rectangle grouping method nearly halves the computation cost. Most importantly, it can be effectively combined with our adaptive leader-pixel strategy. As shown in Fig. 3(b), a sub-tile composed of four mini-tiles typically includes multiple PRs: in Dense Sampling, each mini-tile contributes one PR, resulting in four PRs per sub-tile, whereas Sparse Sampling still can form two valid PRs across mini-tiles.

## IV. HARDWARE ARCHITECTURE

We begin with an overview of the FLICKER architecture, then detail the hierarchical Gaussian testing and contributionaware rendering pipeline, which enables high-throughput Mini-Tile CAT through dedicated architectural support. Finally, we introduce the mixed-precision contribution-aware test unit (CTU) for applying Mini-Tile CAT with minimal hardware overhead.

## A. Overall Architecture

Main Components. As shown in Fig. 5, the architecture consists of four main components: preprocessing core, sorting unit, CTU, and rendering core. The preprocessing core projects 3D Gaussian features into 2D, determines whether Gaussians fall within the frustum, classifies them as spiky or smooth, and performs AABB tests for sub-tile intersections. The sorting unit fetchs the converted features, sorts them by depth, and forwards them to the CTU. The CTU applies Mini-Tile CAT to filter sorted Gaussians according to their contribution. Finally, the rendering core completes the rendering step using the Gaussians that pass CTU.

Memory Access Optimization. Since the number of Gaussians is extremely large, most parameters must be stored offchip. To reduce DDR traffic, we adopt a clustering method that groups multiple Gaussians into larger âbig Gaussiansâ [18]. Frustum culling is then performed on these big Gaussians instead of on individual ones, significantly reducing the number of DDR accesses for the preprocessing core. Moreover, the bandwidth efficiency is further improved by loading only geometric features (10 parameters) during culling, while color features (45 parameters) and other parameters are fetched only for Gaussians that pass frustum culling and intersection test.

<!-- image-->  
Fig. 5: Overall hardware architecture of FLICKER. The key component, the contribution-aware test unit (CTU), is highlighted in purple and will be detailed in Sec. IV-C.

## B. Hierarchical Gaussian Testing and Contribution-Aware Rendering Pipeline

Hierarchical Testing. As discussed in Sec. III, Mini-Tile CAT reduces per-pixel overhead, but the number of Gaussians to test remains high. To handle this, we introduce a two-stage hierarchical testing strategy (Fig. 6). Stage 1: In the preprocessing core, a sub-tile AABB test is performed. Gaussians are duplicated into feature buffers according to their sub-tile intersection mask, enabling efficient skipping at the sub-tile level while reducing the CTU workload (by 30%, as shown in Fig. 4). Stage 2: The CTU processes Gaussians that pass Stage 1 by applying Mini-Tile CAT to generate fine-grained masks. Based on these masks, the contributing Gaussians are then duplicated into the corresponding FIFOs in the rendering core. Each FIFO drives two VRUs, which together render 16 pixelsâexactly one mini-tile. Four such channels within a rendering core cover one sub-tile, while the four rendering cores in FLICKER collectively span a full tile. Since hierarchical testing reduces the Gaussian count to about 10% of the original, the required FIFO capacity is small, which in turn lowers memory overhead. Overall, this organization enables efficient and fine-grained mini-tile skipping under the tile level.

<!-- image-->  
Fig. 6: Hierarchical Gaussian testing.

Contribution-Aware Rendering Pipeline. Beyond hierarchical testing, we further optimize the runtime pipeline to ensure smooth execution. With the dedicated CTU for Mini-Tile CAT, most of its latency is hidden by overlapping with VRU rendering. To further reduce FIFO capacity, we design a stall-resilient pipeline. When any FIFO inside the rendering core becomes full, a FIFO monitor detects the stall and notifies the CTU (Fig. 5). Upon receiving the stall signal, the CTU halts the intake of new Gaussians, while the in-flight pipeline results are safely stored in a small built-in CTU FIFO, ensuring no data loss despite its fully pipelined nature. As validated in Sec. V-B, this design allows very shallow FIFOs to achieve most of the speedup provided by mini-tile skipping. Instead, if CTU throughput falls behind the VRUs, the system can switch to Uniform-Sparse mode, boosting Mini-Tile CAT throughput.

## C. Mixed-Precision Contribution-Aware Test Unit

As shown in Fig. 7(a), the mixed-precision CTU architecture consists of two PRTUs, a Mask Merge Unit (MMU), and units for computing the shared term ln(255 Â· o). The CTU is fully pipelined and can process two PRs (total 8 leader pixels) per cycle, with each PRTU handling one PR. The controller dynamically adjusts the sampling mode based on the Gaussian spiky flag. For Sparse Sampling, the two PRTUs directly generate test masks for two PRs, which are then merged by the MMU and output. For Dense Sampling, four PRs are processed in two batches: the mask from the first batch is stored in registers, and after the second batch completes, the MMU merges both batches to produce the final output, as illustrated in Fig. 7(b).

To reduce the hardware overhead of the CTU, we employ a mixed-precision PRTU: differences between pixel and Gaussian coordinates (line 1 in Alg. 1) are computed in FP16, and the results are then converted to FP8 for subsequent calculations in the Quarda Accumulation Unit (lines 2â7). We evaluate three precision schemes: Full FP16, Full FP8, and mixed precision. As shown in Fig. 7(c), the mixed precision scheme maintains high image quality, whereas Full FP8 suffers from severe PSNR degradation and noticeable blocky artifacts. The degradation primarily arises from the compression of relative positional information between pixels and Gaussians, which leads to the loss of interpolation details. In contrast, our mixed precision scheme preserves critical interpolation information and leverages the inherent error tolerance of Mini-Tile CAT, finally ensuring quality with low hardware cost.

<!-- image-->

<!-- image-->  
Fig. 7: Mixed-precision contribution test unit: (a) microarchitecture, (b) dataflow for the adaptive leader pixel test, and (c) comparison across different precision schemes.

TABLE I: Evaluation of rendering quality (PSNRâ and SSIMâ) across different approaches
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>Tanks &amp;Temples [19]</td><td rowspan=1 colspan=2>MipNeRF360(outdoor) [16]</td><td rowspan=1 colspan=2>DeepBlending [20]</td><td rowspan=1 colspan=1>Average</td></tr><tr><td rowspan=1 colspan=1>PSNR</td><td rowspan=1 colspan=1>SSIM</td><td rowspan=1 colspan=1>PSNR</td><td rowspan=1 colspan=1>SSIM</td><td rowspan=1 colspan=1>PSNR</td><td rowspan=1 colspan=1>SSIM</td><td rowspan=1 colspan=1>PSNR</td></tr><tr><td rowspan=1 colspan=1>Base.</td><td rowspan=1 colspan=1>24.08</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>25.88</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>29.72</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>26.56</td></tr><tr><td rowspan=1 colspan=1>Prun.</td><td rowspan=1 colspan=1>23.61</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>24.71</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>29.64</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>25.99</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>23.51</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>24.52</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>29.62</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>25.88</td></tr></table>

## V. EVALUATION

## A. Experimental Settings

Algorithm Setup. We evaluate on eight real-world scenes: two outdoor scenes from Tanks & Temples, four outdoor scenes from Mip-NeRF 360, and two indoor scenes from Deep Blending. Each scene is first trained with vanilla 3DGS [2] for 30K iterations to obtain baseline models. To produce more compact models, we apply a pruning technique [21], which removes Gaussians with negligible contribution, followed by an additional 3K fine-tuning iterations. After pruning, we adopt the clustering method [18] to group Gaussians into clusters. Training is performed in FP32, and parameters are then quantized for full FP16 rendering on FLICKER.

Hardware Setup. The proposed accelerator comprises 4 Ã (4 Ã 2) VRUs (4 rendering cores), 4 CTUs, 4 sorting units, and 4 preprocessing cores. The design is developed in Verilog and synthesized with Synopsys Design Compiler using the TSMC 28nm process, with SRAMs generated via the memory compiler. To evaluate performance, we build a cycle-accurate simulator of FLICKER, including an LPDDR4 memory with 51.2 GB/s bandwidth, and estimate DRAM energy following [22] [24]. For comparison, we use GSCore [7] and the Jetson XNX GPU [14] as baselines. In addition, we build a simplified version of FLICKER without the CTU to assess its impact.

## B. Critical Component Analysis

Fig. 8 presents the normalized speedup and energy savings obtained from employing the CTU. To highlight its contribution, we evaluate on the baseline model without other optimizations and focus solely on the rendering stage. In terms of speedup, the simplified version of FLICKER is 4Ã slower than GSCore, primarily because it only adopts a basic AABB test, whereas GSCore employs an OBB test [7] and doubles the number of VRUs (32 vs. 64). Integrating the CTU improves performance to 4Ã over the baseline by enabling accurate minitile Gaussian skipping. Even with fewer VRUs, FLICKER still matches the rendering speedup of GSCore. Further configuring the CTU in Uniform-Sparse mode yields an additional 1.1Ã speedup, as it helps efficiently skip large groups of noncontributing Gaussians that would otherwise stall the VRUs. For energy efficiency, our design achieves up to 1.6Ã energy savings over GSCore, as it prevents the massive VRUs from wasting energy on non-contributing Gaussians.

<!-- image-->

<!-- image-->

Fig. 8: Comparison of (a) speedup and (b) energy efficiency for the rendering stage. Note that GSCore [7] is configured with 64 VRUs, while ours uses 32 VRUs for smaller area. The evaluation is performed on the scene Garden only.  
<!-- image-->  
Fig. 9: The sensitivity of speedup and CTU stall rate to the depth of the feature FIFO in the rendering stage. The evaluation is performed on the scene Garden only.

To quantify the impact of FIFO depth on CTU stalls (when FIFO is full), we evaluate the rendering-stage speedup of FLICKER across depths from 1 to 128 (Fig. 9) and the corresponding CTU stall rates. Results show that increasing FIFO depth reduces stalls and improves speedup, reaching a maximum of 1.36Ã at depth 128. However, returns diminish beyond a depth of 16, which already achieves 96% of the maximum speedup while using only 12.5% of the memory compared to depth 128. Therefore, we select a FIFO depth of 16 for configuration. This highlights the effectiveness of hierarchical testing, which enables mini-tile skipping with shallow FIFOs rather than large sub-tile buffers and achieves most of the performance gain with minimal memory overhead.

## C. Overall System Evaluation

Tbl. I compares the rendering image quality across different methods. Ours incurs only an average PSNR loss of 0.11 dB compared to the pruning model, demonstrating that our adaptive leader pixel strategy is effective in capturing most contributing Gaussians and preserving visual quality.

<!-- image-->  
Fig. 10: Overall (a) speedup and (b) energy efficiency, normalized to the GPU baseline.

(a)
<table><tr><td>Component</td><td>Config.</td><td>Area [mm2]</td></tr><tr><td>Preprocessing Core</td><td>4</td><td>0.76</td></tr><tr><td>Sorting Unit</td><td>4</td><td>0.16</td></tr><tr><td>Contri. Test Unit</td><td>4</td><td>0.09</td></tr><tr><td>Rendering Core</td><td>4Ã(4x2)</td><td>0.96</td></tr><tr><td>Fea. Buffer+Others</td><td>288KB</td><td>1.50</td></tr><tr><td>Total</td><td></td><td>3.47</td></tr></table>

(b)  
<!-- image-->  
TABLE II: (a) Hardware configuration and area breakdown. (b) Area comparison between baseline and our design.

Fig. 10 shows the system performance over the baseline, with all values normalized to the XNX. Integrating CTU while adopting existing optimizations (pruning and clustering), FLICKER achieves average 1.1Ã speedup over GSCore and 14.4Ã over XNX. Furthermore, FLICKER consistently achieves the highest efficiency across all dataset, with maximum 2.6Ã up to GSCore and 26.7Ã compared to XNX. This demonstrates FLICKERâs capability to enable real-time 3DGS rendering for edge applications, and its compatibility with existing optimization techniques.

Tbl. II(a) reports the area breakdown of FLICKER. Thanks to the mixed-precision architecture and pixel-rectangle grouping, the CTU occupies less than 10% of the VRUs area (rendering core), yet delivers up to 2.3Ã overall speedup, which is difficult to achieve by merely adding more VRUs. We further extend the simplified version from 32 VRUs to 64 VRUs to emulate GSCoreâs configuration as baseline in terms of VRU count. As shown in Tbl. II(b), although the CTU and feature FIFO introduce minor additional area, they enable a more efficient area allocation than using more VRUs, ultimately achieving 14% total area savings.

## VI. CONCLUSION

This paper introduces FLICKER, a contribution-aware accelerator that focuses on reducing unnecessary Gaussian processing over merely scaling parallel hardware. It performs prior contribution test to accurately skip Gaussians at finegrained pixel blocks before rendering, while leveraging softwareâhardware co-design to alleviate its associated overheads. Experimental results show that our design outperforms a stateof-the-art 3DGS accelerator and an edge GPU device across most real-world datasets, while incurring less area overhead.

[1] Meta, âIntroducing orion, our first true augmented reality glasses,â https://about.fb.com/news/2024/09/introducing-orion-our-first-trueaugmented-reality-glasses/, 2024.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[3] T. Wu, Y.-J. Yuan, L.-X. Zhang, J. Yang, Y.-P. Cao, L.-Q. Yan, and L. Gao, âRecent advances in 3d gaussian splatting,â Computational Visual Media, vol. 10, no. 4, pp. 613â642, 2024.

[4] W. Lin, Y. Feng, and Y. Zhu, âMetasapiens: Real-time neural rendering with efficiency-aware pruning and accelerated foveated rendering,â in Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, 2025, pp. 669â682.

[5] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Drettakis, âReducing the memory footprint of 3d gaussian splatting,â Proceedings of the ACM on Computer Graphics and Interactive Techniques, vol. 7, no. 1, pp. 1â17, 2024.

[6] Meta, âMeta quest 3 mixed reality headset,â https://www.meta.com/quest/quest-3/, 2023.

[7] J. Lee, S. Lee, J. Lee, J. Park, and J. Sim, âGscore: Efficient radiance field rendering via architectural support for 3d gaussian splatting,â in Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, 2024, pp. 497â511.

[8] G. Feng, S. Chen, R. Fu, Z. Liao, Y. Wang, T. Liu, B. Hu, L. Xu, Z. Pei, H. Li et al., âFlashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 652â26 662.

[9] X. Wang, R. Yi, and L. Ma, âAdr-gaussian: Accelerating gaussian splatting with adaptive radius,â in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1â10.

[10] A. Hanson, A. Tu, G. Lin, V. Singla, M. Zwicker, and T. Goldstein, âSpeedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 537â21 546.

[12] NVIDIA Corporation, âNvidia nsight compute,â https://developer.nvidia.com/nsight-compute, 2025.

[13] NVIDIA, âGeforce rtx 3090 family,â https://www.nvidia.com/engb/geforce/graphics-cards/30-series/rtx-3090-3090ti/, 2025.

[11] X. Huang, H. Zhu, Z. Liu, W. Lin, X. Liu, Z. He, J. Leng, M. Guo, and Y. Feng, âSeele: A unified acceleration framework for real-time gaussian splatting,â arXiv preprint arXiv:2503.05168, 2025.

[14] ââ, âJetson xavier series,â https://www.nvidia.com/en-gb/autonomousmachines/embedded-systems/jetson-xavier-series/, 2025.

[15] Y. K. Zhao, S. Wu, J. Zhang, S. Li, C. Li, and Y. C. Lin, âInstant-nerf: Instant on-device neural radiance field training via algorithm-accelerator co-designed near-memory processing,â in 2023 60th ACM/IEEE Design Automation Conference (DAC). IEEE, 2023, pp. 1â6.

[16] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5470â5479.

[17] Y. Wang, Y. Li, J. Chen, J. Yu, and K. Wang, âFamers: An fpga accelerator for memory-efficient edge-rendered 3d gaussian splatting,â in 2025 Design, Automation & Test in Europe Conference (DATE). IEEE, 2025, pp. 1â7.

[18] J. Jo and J. Park, âPs-gs: Group-wise parallel rendering with stagewise complexity reductions for real-time 3d gaussian splatting,â in 2025 Design, Automation & Test in Europe Conference (DATE). IEEE, 2025, pp. 1â7.

[19] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, âTanks and temples: Benchmarking large-scale scene reconstruction,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1â13, 2017.

[20] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Brostow, âDeep blending for free-viewpoint image-based rendering,â ACM Transactions on Graphics (ToG), vol. 37, no. 6, pp. 1â15, 2018.

[21] M. S. Ali, M. Qamar, S.-H. Bae, and E. Tartaglione, âTrimming the fat: Efficient compression of 3d gaussian splats through pruning,â arXiv preprint arXiv:2406.18214, 2024.

[22] P. Dong, Y. Tan, X. Liu, P. Luo, Y. Liu, L. Liang, Y. Zhou, D. Pang, M.-T. Yung, D. Zhang et al., âA 28nm 0.22 Âµj/token memory-computeintensity-aware cnn-transformer accelerator with hybrid-attention-based

layer-fusion and cascaded pruning for semantic-segmentation,â in 2025 IEEE International Solid-State Circuits Conference (ISSCC), vol. 68. IEEE, 2025, pp. 01â03.

[23] Z. Jia, M. Maggioni, B. Staiger, and D. P. Scarpazza, âDissecting the nvidia volta gpu architecture via microbenchmarking,â arXiv preprint arXiv:1804.06826, 2018.

[24] K. Song, S. Lee, D. Kim, Y. Shim, S. Park, B. Ko, D. Hong, Y. Joo, W. Lee, Y. Cho et al., âA 1.1 v 2y-nm 4.35 gb/s/pin 8 gb lpddr4 mobile device with bandwidth improvement techniques,â IEEE Journal of Solid-State Circuits, vol. 50, no. 8, pp. 1945â1959, 2015.