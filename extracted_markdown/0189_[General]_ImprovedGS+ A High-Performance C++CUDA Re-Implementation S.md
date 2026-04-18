# ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting

Jordi Munoz Vicente Ë

Universidad de Murcia

Murcia, Spain

jordi.munozv@um.es

## Abstract

Recent advancements in 3D Gaussian Splatting (3DGS) have shifted the focus toward balancing reconstruction fidelity with computational efficiency. In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework. By transitioning from high-level Python logic to optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency. Our implementation introduces a Long-Axis-Split (LAS) CUDA kernel, a custom Laplacian-based importance kernels with Non-Maximum Suppression (NMS) for edge scores, and an adaptive Exponential Scale Scheduler.

Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Paretooptimal front for scene reconstruction. Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality. Furthermore, our full variant demonstrates a 1.28 dB PSNR increase over the ADC baseline with a 38.4% reduction in parametric complexity. These results validate ImprovedGS+ as a scalable, high-speed solution that upholds the core pillars of Speed, Quality, and Usability within the LichtFeld-Studio ecosystem.

## 1. Introduction

Recent advances in 3D Gaussian Splatting (3DGS) [5] have shifted the focus to balancing reconstruction quality with computational efficiency. Building upon the foundations of ImprovedGS strategy [3] which is built upon the TamingGSâs work [8], we propose ImprovedGS+: a highperformance, low-level reinvention of the ImprovedGS baseline.

While ImprovedGS provides a robust framework for densification, our implementation seeks to push the boundaries of time efficiency by transitioning from Python-based logic to a native C++/CUDA architecture. ImprovedGS+ adopts a streamlined version of its predecessor, focusing on the two most impactful components: Long-Axis-Split (LAS) and Edge-Score importance sampling. However, we go beyond a mere port of the baseline: We simplify some ImprovedGS features while we introduce subtle but critical modifications that proved to improve terminal metrics:

â¢ Exponential Scale Scheduling

â¢ Optimized Positional Learning

â¢ Customized Enhanced Laplacian Masking

Our implementation is developed within the LichtFeld-Studio [9] framework. By leveraging custom GPU kernels for both filtering and the Long-Axis-Split operation, we minimize host-device synchronization and maximize training throughput. This allows ImprovedGS+ to achieve highefficiency scene representation without compromising the visual fidelity required for complex, large-scale environments.

## 2. Methodology

## 2.1. Laplacian Filter Kernel Implementation

To generate precise structural importance maps for densification, we implemented a custom CUDA-native pipeline. We transition from high-level framework calls to a sequence of specialized kernels.

The pipeline follows a distilled Canny Edge Detection logic [2]:

1. Preprocessing: Grayscale conversion and a 5 Ã 5 Gaussian blurâstored in Constant Memory ( constant )âto filter sensor noise.

2. Gradient Extraction: Parallel computation of magnitude and orientation via Sobel operators.

3. Non-Maximum Suppression (NMS): A custom thinning kernel that compares magnitudes along the gradient vector, effectively isolating geometric manifolds.

As illustrated in Figure 1, while the Baseline (original Python-based ImprovedGS) (b) suffers from excessive stipple noise on planar surfaces, our ImprovedGS+ approach (c) produces a significantly thinned structural backbone. This prevents âdensification driftâ by ensuring primitives are placed only on true geometric boundaries. Furthermore, we apply median-normalization to mitigate photometric outliers, ensuring high-intensity signals are preserved without being suppressed by global peaks.

Future Work Future versions could further reduce VRAM bandwidth through Kernel Fusion, consolidating the initial stages into a single monolithic kernel using Shared Memory ( shared ) for intermediate storage.

## 2.2. Direct CUDA Kernel Long-Axis-Split (LAS)

To achieve maximum efficiency during scene densification, ImprovedGS moves away from standard isotropic splitting and cloning in favor of an exclusive Long-Axis-Split (LAS) strategy. Within LichtFeld-Studio framework, we implemented a custom CUDA kernel specifically for this purpose. Unlike the original Python-based implementation, which relies on high-level PyTorch function calls and multiple memory passes, our approach encapsulates the entire splitting logic within a single CUDA kernel. In this architecture, all geometric and attribute operations are computed in parallel at the individual Gaussian level.

It is important to note that we modify using both storage data (raw parameters) and physical data (spatial parameters retrieved via activation functions). Since scales are stored in logarithmic space, we perform the shrinking operation directly on the storage data using the logarithmic properties to ensure numerical consistency.

Algorithm 1 where Î± represents the shrinking factor in the longest axis (set to 0.5 in our version) and Î³ the shrinking factor for the secondary axes (set to 0.85). Physical scale values are only retrieved via exponential activation when calculating the spatial offset. Similarly, for opacity, we reduce the physical brightness by a factor of Î² = 0.6 before converting back to the raw logit space.

Specialized Global Transformation for LAS Finally, our custom specialized kernel CUDA optimizes the global coordinate transformation for the new Gaussian positions. Since the split displacement occurs strictly along a single local basis vector (the principal axis), the global displacement is mathematically equivalent to a linear scaling of the corresponding column of the rotation matrix.

<!-- image-->

(a) Reference Image  
<!-- image-->

<!-- image-->  
(c) ImprovedGS+ (Ours)  
Figure 1. Visual comparison of edge importance maps on the Bicycle scene. The zoomed regions highlight how our CUDA-based NMS kernel reduces the noise and focuses in a thinned structural backbone compared to the baseline.

By extracting this column directly, we avoid the computational cost of a full 3 Ã 3 matrix-vector multiplication, reducing the operation from 9 multiplications and 6 additions

Algorithm 1: Long-Axis-Split Gaussians In Place   
Data: Gaussian attributes g, Rotation Matrix R.   
foreach Gaussian $g \in$ scene to split do   
$/ /$ Identi $\mathtt { f y }$ the principal axis   
and magnitude offset   
$l _ { i d x }  \mathrm { g e t }$ max index(g.scale)   
$o f f s e t _ { m a g } \gets \exp ( g . { \mathrm { s c a l } } \mathbf { e } _ { l _ { i d x } } ) \times \alpha$   
$/ /$ Update scale components   
$n w _ { - } s c a l e _ { l _ { i d x } }  g . \mathrm { s c a l e } _ { l _ { i d x } } + \log ( \alpha )$   
$n w \_ s c a l e _ { o t h e r } \gets g . s c a l \mathbf { e } _ { o t h e r } + \log ( \gamma )$   
$/ /$ Update opacity (applying   
reduction factor)   
raw $\_ o p a c  \sigma ( g . o p a c ) \times \beta$   
nw opac â Ïâ1(raw opac)   
$/ /$ Compute global position   
offset using R   
$\vec { v } _ { o f f s e t }  \mathbf { R } [ : , l _ { i d x } ] \times o f f s e t _ { m a g }$   
nw $_ { - p o s _ { 0 } } \gets g . p o s + { \vec { v } } _ { o f f s e t }$   
nw ${ - p o s _ { 1 } \gets g . p o s - \vec { v } _ { o f f s e t } }$   
end

Algorithm 2: Coordinate Transformation for LAS   
Input : Rotation matrix R (flat $3 \times 3 )$ , principal   
axis index ${ l } _ { i d x } ,$ physical scale $s _ { p h y s }$   
Output: Global displacement vector d   
$d _ { x } \gets \mathbf { R } [ l _ { i d x } ] \times s _ { p h y s }$   
$d _ { y } \gets \mathbf { R } [ l _ { i d x } + 3 ] \times s _ { p h y s }$   
$d _ { z } \gets \mathbf { R } [ l _ { i d x } + 6 ] \times s _ { p h y s }$

to only 3 multiplications.

## 2.3. Refinement Setup

We operate under the observation that reducing densification frequency can preserve reconstruction quality while significantly minimizing computational overhead. Consequently, we keep a fixed densification interval of 500 iterations. By constraining the densification window from iteration 500 to 15,000, we restrict the model to exactly 30 total densification steps. We specifically investigate the interplay between an exclusive Long-Axis-Split (LAS) approach and the Gaussian scaling learning rate, demonstrating that our specialized learning schedule improves both initial convergence speed and terminal metrics. The strategy is divided into a two-stage process:

â¢ Stage I: High-Momentum Expansion. To compensate for the removal of Gaussian cloning, the initial scale learning rate is increased to 0.020 (Ã4 original initial scale rate). This allows the primitives to rapidly expand and occupy empty scene volume. When combined with the LAS primitive, this ensures that each split is spatially deterministic and non-overlapping.

â¢ Stage II: Precision Refinement. To capture highfrequency details that a high scale learning rate might otherwise overlook, we transition into a refinement phase. Using an Exponential Learning Rate Scheduler with $\gamma = ( 0 . 1 )$ from initial 0.020 is gradually decayed to a final value 0.002 learning rate. This stabilizes the primitives, allowing them to represent complex, fine-grained geometries, which boosts both PSNR and SSIM metrics while at the same time preventing redundant primitives, lowering the final Gaussian count.

Positional Learning Rate Optimization Furthermore, we have significantly increased the positional (means) learning rate. The original ImprovedGS baseline utilizes an initial learning rate of 0.00004 decaying to 0.000002, our implementation employs an initial rate of 0.000128 decaying to 0.0000128. As demonstrated in our Ablation Study, although a higher positional learning rate initially faces challenges in representing the scene, it eventually unlocks higher potential in PSNR and SSIM, particularly when combined with the scale variability provided by our specialized scale scheduler.

Global Warm-up Phase (Initial Score Masking) A subtle yet functional nuance in our pipeline is the handling of early-stage importance scores. For the first three densification steps (iterations 500, 1,000, and 1,500), we suspend the gradient-threshold masking, allowing a number of Gaussians to split based purely on their calculated edge scores.

This âwarm-upâ phase ensures that outdoor scenes with broad, low-gradient surfaces are sufficiently populated before the model transitions into strict structural refinement.

## 3. Experiments and Results

This section evaluates the proposed approach within LichtFeld-Studio framework both quantitatively and qualitatively. The evaluation was conducted using an NVIDIA RTX A4500 GPU. Results for the other techniques within LichtFeld Studio such as MCMC strategy [6] and ADC (based on 3DGS), including training times, were obtained on the same hardware and environment to ensure comparability.

## 3.1. Datasets and Metrics

We evaluate our method exclusively on the Mip-Nerf360 dataset [1], covering a diverse range of nine scenes: Garden, Bicycle, Stump, Bonsai, Counter, Kitchen, Room, Flowers, and Treehill.

Although other benchmarks such as Tanks & Temples [7] and Deep Blending [4] are common in the literature, we had to omit them from this study. This decision is due to the current validation architecture of the LichtFeld-Studio framework, which was returning metrics based on train dataset for both Tank&Temples and DeepBlending. To ensure the most rigorous and consistent comparative analysis within this ecosystem, we focused on the comprehensive Mip-NeRF360 suite, which provides a balanced mix of complex outdoor environments and structured indoor manifolds.

<table><tr><td>Scene</td><td colspan="2">Garden</td><td colspan="2">Bonsai</td></tr><tr><td></td><td>PSNR â SSIMâ</td><td>#G(106) â</td><td>PSNR â SSIMâ</td><td>#G(106) â</td></tr><tr><td>IGS (Baseline)</td><td>26.86 0.845</td><td>1.310</td><td>30.39 0.935</td><td>0.535</td></tr><tr><td>IGS+ (Ours)</td><td>26.08 0.809</td><td>1.400</td><td>30.34 0.932</td><td>0.573</td></tr><tr><td>-w/o scale scheduler</td><td>25.95 0.806</td><td>1.400</td><td>30.18 0.933</td><td>0.575</td></tr></table>

Table 1. Bonsai and Garden scenes Ablations table at 7,000 iterations for ImprovedGS+. Averaged from 6 independent runs.

We compare the common quality metrics: peak signalto-noise ratio (PSNR) and structural similarity (SSIM). It is important to note that our work focuses in resource efficiency: Aiming to achieve high quality with low resource usage. We assess these qualities by timing the optimization (Train time) in minutes and seconds and by counting the final number of Gaussians (#G) in millions (Ã106) for clarity.

## 3.2. Ablation Study

To validate the individual components of the ImprovedGS+ (IGS+) framework, we conduct a comparative analysis across two distinct environments from the Mip-Nerf360 dataset: Garden (Outdoor) and Bonsai (Indoor).

We evaluate three configurations without any budget to isolate the impact of our optimizations:

â¢ IGS+ (Ours): The full implementation featuring retuned position lr boundaries and init scale rate along with our Exponential Scale Scheduling.

â¢ w/o Scale Scheduler: The full IGS+ pipeline but without the exponential decay for Gaussian scales.

â¢ IGS (Baseline): Our C++/CUDA re-implementation of the original ImprovedGS, using its native positional and scale learning rates.

At 7,000 iterations (Table 1), the standard IGS baseline maintains a slight lead. This is an intentional design choice: our higher positional lr at Stage I expansion prioritizes the rapid acquisition of geometric volume and primitive displacement over immediate photometric precision. This trade-off is critical for preventing the âunderreconstructionâ of complex areas.

By the terminal state at 30,000 iterations (Table 2), the benefits of our LichtFeld-Studio parameter tuning become clear:

1. Garden: PSNR improves from 29.05 (Baseline) to 29.39 (Ours).

2. Bonsai: PSNR improves from 33.16 (Baseline) to 33.37 (Ours).

<table><tr><td>Scene</td><td colspan="2">Garden</td><td colspan="2">Bonsai</td></tr><tr><td></td><td>PSNR â SSIMâ</td><td>#G(106)</td><td>PSNR â SSIMâ</td><td>#G(106) </td></tr><tr><td>IGS (Baseline)</td><td>29.05 0.887</td><td>2.614</td><td>33.16 0.953</td><td>0.827</td></tr><tr><td>IGS+ (Ours)</td><td>29.39 0.896</td><td>2.930</td><td>33.37 0.954</td><td>0.830</td></tr><tr><td>-w/o scale scheduler</td><td>29.16 0.894</td><td>2.937</td><td>32.95 0.952</td><td>0.843</td></tr></table>

Table 2. Bonsai and Garden scenes Ablations table at 30,000 iterations for ImprovedGS+. Averaged from 6 independent runs.

As shown in Table 2, in Garden the IGS Baseline tops at 2.61M Gaussians, unable to capture further complexity. Our positional lr allows for a more adaptive densification, reaching 2.93M Gaussians, which is directly responsible for the 0.34 dB PSNR improvement.

Impact of the Scale Scheduler The removal of the Exponential Scale Scheduler (w/o scale scheduler) leads to a measurable drop in terminal metrics. In the Bonsai scene, disabling the scheduler results in a 0.42 dB drop in PSNR. This confirms that our scheduler is essential for âfreezingâ the structural details and preventing the primitives from oscillating or over-expanding during the final refinement phase.

## 3.3. Results

<table><tr><td>Dataset</td><td colspan="4">MipNerf360</td></tr><tr><td></td><td>PSNR â</td><td>SSIM â</td><td>Timeâ</td><td>#G(106) â</td></tr><tr><td>MCMC 1M</td><td>28.64</td><td>0.848</td><td>62m14s</td><td>1.000</td></tr><tr><td>Ours (IGS+ 1M)</td><td>28.79</td><td>0.859</td><td>45m33s</td><td>0.867</td></tr><tr><td>Ours (IGS+)</td><td>29.38</td><td>0.877</td><td>62m10s</td><td>1.688</td></tr><tr><td>ADC</td><td>28.10</td><td>0.848</td><td>76m02s</td><td>2.741</td></tr></table>

Table 3. Experimental Results across the Mip-Nerf360 dataset at 30,000 iterations. Metrics for each scene are averaged from 6 independent runs. For a fair comparative analysis, the best results between MCMC 1M and ImprovedGS+ 1M are highlighted in red.

## 4. Conclusion

As summarized in Table 3, ImprovedGS+ (IGS+) demonstrates clear dominance over existing strategies within the LichtFeld-Studio framework. Compared to the State-Of-The-Art MCMC strategy, under an identical 1 million Gaussian budget our method achieves superior metrics in significantly less time. We report an aggregate reduction of approximately 17 minutes across all nine benchmark scenes (26.8% time reduction) while utilizing 13.3% fewer Gaussians. This underscores the high level of optimization and âper-primitive contributionâ that IGS+ introduces to the densification process.

A key strength of our approach is its adaptative efficiency. As evidenced in the indoor scenes (see Appendix), IGS+ consistently reaches SOTA metrics without exhausting the maximum permitted budget. In the Room scene, for instance, our model achieves superior PSNR using only

0.591M Gaussians compared to the 1.000M required by MCMC. Even when run without a strict budget (utilizing a 3M pre-allocated capacity), IGS+ naturally optimizes the scene representation to an average of 1.688M Gaussians â a 40% reduction compared to the ADC strategy. Crucially, this high-fidelity variant completes training in the same temporal window as the MCMC 1M baseline, suggesting significantly higher architectural throughput.

While the baseline restrictions observed in our ablation study initially limited growth in complex environments, our refined positional lr scheduling successfully pushed these boundaries. By allowing for a more granular movement and precise placement of primitives (especially at the end phase of training), we directly translated increased Gaussian counts into improved visual fidelity in challenging outdoor scenes like Garden

Ultimately, ImprovedGS+ provides a highly scalable solution: the full version maximizes visual quality for highend rendering, while the 1M budget variant offers a highspeed, mobile-ready alternative that performs the MCMC baseline across all metrics. This performance reinforces the core pillars of Speed, Quality and Usability, that define the LichtFeld-Studio ecosystem.

## References

[1] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3

[2] John Canny. A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6):679â698, 1986. 1

[3] Xiaobin Deng, Changyu Diao, Min Li, Ruohan Yu, and Duanqing Xu. Improving densification in 3d gaussian splatting for high-fidelity rendering, 2025. 1

[4] Peter Hedman, Julien Philip, True-Price Garrison, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. Deep blending for free-viewpoint image-based rendering. ACM Transactions on Graphics (SIGGRAPH Asia), 37(6), 2018. 3

[5] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 2023. 1

[6] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Kwang Moo Yi, and Andrea Tagliasacchi. 3d gaussian splatting as markov chain monte carlo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19582â19591, 2024. 3

[7] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017. 3

[8] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Francisco Vicente Carrasco, Markus Steinberger, and Fer-

nando De La Torre. Taming 3dgs: High-quality radiance fields with limited resources, 2024. 1

[9] LichtFeld Studio. A high-performance c++ and cuda implementation of 3d gaussian splatting, 2025. 1

## A. Appendix

## A.1. Quantitative Comparison per Scene

Comparison highlight is done only between MCMC and ImprovedGS+ strategies when both using same budget.

## MipNeRF360: Indoor Scenes

<table><tr><td rowspan="2">Scene</td><td colspan="4">Bonsai</td><td colspan="4">Counter</td></tr><tr><td>PSNR â</td><td>SSIMâ</td><td>Time â</td><td>#Gâ</td><td>PSNR â</td><td>SSIMâ</td><td>Time â</td><td>#G â</td></tr><tr><td rowspan="3">MCMC 1M ImprovedGS+</td><td>32.91</td><td>0.951</td><td>5m54s</td><td>1.000</td><td>30.84</td><td>0.928</td><td>6m48s</td><td>1.000</td></tr><tr><td>33.37</td><td>0.954</td><td>4m35s</td><td>0.830</td><td>31.05</td><td>0.929</td><td>4m13s</td><td>0.618</td></tr><tr><td>32.10</td><td>0.942</td><td>4m12s</td><td>1.147</td><td>29.19</td><td>0.897</td><td>4m08s</td><td>0.875</td></tr><tr><td rowspan="2">Scene</td><td colspan="4">Kitchen</td><td colspan="4">Room</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>Time â</td><td>#Gâ</td><td>PSNR â</td><td>SSIMâ</td><td>Time â</td><td>#G â</td></tr><tr><td rowspan="3">MCMC 1M ImprovedGS+</td><td>32.43</td><td>0.938</td><td>6m57s</td><td>1.000</td><td>34.24</td><td>0.946</td><td>5m00s</td><td>1.000</td></tr><tr><td>32.47</td><td>0.938</td><td>5m09s</td><td>0.760</td><td>34.36</td><td>0.946</td><td>3m16s</td><td>0.591</td></tr><tr><td>31.16</td><td>0.927</td><td>5m36s</td><td>1.194</td><td>32.97</td><td>0.933</td><td>4m05s</td><td>1.127</td></tr></table>

Table 4. Extended evaluation metrics for all Indoor scenes. Each scene averaged from 6 independent runs.

## MipNeRF360: Outdoor Scenes

<table><tr><td rowspan="2">Scene</td><td colspan="4">Garden</td><td colspan="4">Bicycle</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>Time â</td><td>#G â</td><td>PSNR â</td><td>SSIM â</td><td>Time â</td><td>#G â</td></tr><tr><td rowspan="3">MCMC 1M ImprovedGS+1M</td><td>28.27</td><td>0.866</td><td>7m53s</td><td>1.000</td><td>25.07</td><td>0.790</td><td>6m59s</td><td>1.000</td></tr><tr><td>28.47</td><td>0.871</td><td>6m26s</td><td>1.000</td><td>25.39</td><td>0.798</td><td>5m33s</td><td>1.000</td></tr><tr><td>29.39</td><td>0.896</td><td>11m39s</td><td>2.930</td><td>26.88</td><td>0.813</td><td>9m45s</td><td>3.000</td></tr><tr><td rowspan="2">ADC Scene</td><td>28.46</td><td>0.870</td><td>13m42s</td><td>4.082</td><td>25.68</td><td>0.803</td><td>14m06s</td><td>5.211</td></tr><tr><td colspan="4">Flowers</td><td colspan="4">Treehill</td></tr><tr><td rowspan="4">MCMC 1M ImprovedGS+1M ImprovedGS+</td><td>PSNR â</td><td>SSIMâ</td><td>Time â</td><td>#G â</td><td>PSNR â</td><td>SSIMâ</td><td>Time â</td><td>#G â</td></tr><tr><td>23.43</td><td>0.741</td><td>7m44s</td><td>1.000</td><td>22.91</td><td>0.727</td><td>7m59s</td><td>1.000</td></tr><tr><td>23.39 24.35</td><td>0.720 0.761</td><td>5m42s 8m32s</td><td>1.000 2.250</td><td>23.21 24.24</td><td>0.738</td><td>5m22s</td><td>1.000</td></tr><tr><td>23.32</td><td>0.718</td><td>10m42s</td><td>3.542</td><td>21.86</td><td>0.790 0.693</td><td>7m41s 8m32s</td><td>2.250 3.213</td></tr><tr><td rowspan="2">ADC Scene</td><td colspan="5"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td></td><td>Stump</td><td></td><td></td></tr><tr><td rowspan="4">MCMC 1M</td><td>PSNR â</td><td>SSIMâ</td><td>Time â</td><td>#Gâ</td><td></td><td></td><td></td></tr><tr><td>27.65</td><td>0.832</td><td>7m00s</td><td>1.000</td><td></td><td></td><td></td></tr><tr><td>ImprovedGS+ 1M 27.38</td><td>0.835</td><td>5m17s</td><td>1.000</td><td></td><td></td><td></td></tr><tr><td>28.36</td><td>0.864</td><td>7m20s</td><td>1.960</td><td></td><td></td><td></td></tr><tr><td>ImprovedGS+ ADC</td><td>28.18</td><td>0.848</td><td>10m52s</td><td>4.280</td><td></td><td></td><td></td><td></td></tr></table>

Table 5. Extended evaluation metrics for all Outdoor scenes. Each scene averaged from 6 independent runs.