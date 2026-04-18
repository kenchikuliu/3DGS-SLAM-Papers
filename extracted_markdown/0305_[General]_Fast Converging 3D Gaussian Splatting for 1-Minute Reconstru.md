# Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction

Ziyu Zhang zhangziyu2021@ia.ac.cn University of Chinese Academy of Sciences Institute of Automation, Chinese Academy of Sciences Beijing, Beijing, China

Diantao Tu diantao.tu@ia.ac.cn University of Chinese Academy of Sciences Institute of Automation, Chinese Academy of Sciences Beijing, Beijing, China

## Abstract

We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge. The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP [10] poses (highly accurate). To robustly handle these heterogeneous settings, we develop a two-stage solution.

In the first round, we use (1) reverse per-Gaussian parallel optimization and compact forward splatting based on Taming-GS [8] and Speedy-splat [1], (2) load-balanced tiling, (3) an anchor-based Neural-Gaussian representation [7] enabling rapid convergence with fewer learnable parameters, (4) initialization from monocular depth and partially from feed-forward 3DGS models [4], and (5) a global pose refinement module for noisy SLAM trajectories.

In the final round, the accurate COLMAP poses change the optimization landscape; we (1) disable pose refinement, revert from Neural-Gaussians back to standard 3DGS to eliminate MLP inference overhead (2) introduce multi-view consistency-guided Gaussian splitting inspired by Fast-GS [9] (3) introduce a depth estimator [2] to supervise the rendered depth. Together, these techniques enable high-fidelity reconstruction under a strict one-minute budget. Our method achieved the top performance with a PSNR of 28.43 and ranked first in the competition. Code available at https://github.com/will-zzy/siggraph_asia.

## CCS Concepts

â¢ Computing methodologies â Reconstruction; Rendering;   
Neural networks; Computer vision problems.

## Keywords

3D Gaussian Splatting, Fast Reconstruction, Neural Rendering, Scene Representation

## 1 Introduction

Efficient reconstruction with 3D Gaussian Splatting [5] has become a key research direction. While existing methods demonstrate impressive visual fidelity, achieving real-time convergence remains challenging, primarily due to (i) the limited efficiency of current

Tianle Liu tianleliu@whu.edu.cn Wuhan University Wuhan, Hubei, China

Shuhan Shen shshen@nlpr.ia.ac.cn University of Chinese Academy of Sciences Institute of Automation, Chinese Academy of Sciences Beijing, Beijing, China

rasterization pipelines, (ii) the critical dependence on high-quality initialization, and (iii) the lack of an effective densification and pruning strategy. The 3DGS Challenge targets these issues by imposing a strict 1-minute time budget for training. Our solution is engineered specifically for this constraint, and we describe it in two phases:

First Round (SLAM poses): Noisy poses, sparse point clouds, and challenging frame sampling. Final Round (COLMAP [10] poses): Accurate poses, more stable training, and larger allowable Gaussian counts. Below, we summarize the architecture and optimization strategy for each stage.

## 2 First Round: Fast Reconstruction with Noisy SLAM Poses

## 2.1 Forward: Compact BBox and Load Balance

GS-like methods [1, 3, 5, 11, 12] involve two main steps during forward rendering. The first step, preprocessCUDA, can be viewed as analogous to the vertex shader stage in rasterization. It projects each 3D Gaussian onto the image plane using an approximate affine transform, computes the 2D Gaussian centroid and covariance, and then determines the tiles that need to be shaded. Each tile then writes the tile id and the depth of the Gaussian centroid in parallel, which will later be used for depth sorting.

The second step, renderCUDA, resembles the fragment shader stage. It processes each pixel in parallel, iterates through the depthsorted Gaussians, computes the Gaussian weights, performs alpha blending, and outputs the final rendered image. However, the tile size computed for each Gaussian in preprocessCUDA significantly impacts the efficiency of renderCUDA. Many of the generated tiles are irrelevant for rendering, resulting in substantial computational waste, especially for elongated Gaussians as shown in Fig. 1. Therefore, we adopt the analytical approach introduced in speedysplat [1] to prune redundant tiles. Specifically, given the 2D ellipse parameters $C o v 2 D = \{ a , b , c \}$ , the ellipse centroid $\mu ,$ and Gaussian opacity ??, the ellipse can be parameterized by:

$$
2 \ln ( 2 5 5 \cdot o ) = t = a x _ { d } ^ { 2 } + 2 b x _ { d } y _ { d } + c y _ { d } ^ { 2 }\tag{1}
$$

Using the extremum condition $\partial y _ { d } / \partial x _ { d } = 0 .$ , we can compute the tangent points of the ellipse along the ??-axis and ??-axis, i.e. $x _ { m i n } / x _ { m a x } , y _ { m i n } / y _ { m a x }$ . This yields a compact bounding box (the

<!-- image-->

Figure 1: The illustration of the compact bounding strategy, adapted from [1]. The blue thin boxes denote the bounding boxes, and the yellow regions indicate the tiles being written. The first column shows the original 3DGS. The second column our method. The third column presents the pruned bounding box after applying our compact tiling strategy.  
<!-- image-->  
Figure 2: Illustration of sequential writing (left) versus loadbalanced writing (right). The blue boxes denote the initially recorded tiles, while the red regions indicate the tiles that are finally written.

âSnugBoxâ in Fig. 1):

$$
y _ { m i n / m a x } = \frac { - b x _ { d } \pm \sqrt { ( b ^ { 2 } - a c ) x _ { d } ^ { 2 } + t c } } { c } , x _ { d } = \pm \sqrt { \frac { b ^ { 2 } t } { ( b ^ { 2 } - a c ) a } }\tag{2}
$$

Within the rectangle tiles covered by the SnugBox, we iterate over each column tile and analytically compute its intersection with the ellipse using the left and right tile boundary coordinates $( \mathrm { e . g . , } x = x _ { t m i n } , x = x _ { t m a x } )$ . This allows us to identify which tiles intersect the ellipse and sequentially write out their tile id and depth value.

Since the BBox sizes vary significantly across Gaussians, the number of tiles processed by each thread also differs greatly, which can lead to thread divergence. Therefore, we also explore using load balance to accelerate this stage. Specifically, we employ cooperative groups to distribute work evenly among threads within the same warp. The difference between this approach and sequential tile writing is illustrated in Fig. 2. In the sequential write mode, a single thread scans through each column or row and analytically computes the intersections between the ellipse and the column/row. In contrast, with load balancing, all threads within the same warp (the blue boxed region) simultaneously evaluate whether their assigned tile intersects with the ellipse.

## 2.2 Backward Propagation: Per-Gaussian Parallelism

GS-like methods also adopt a two-stage procedure during backpropagation. In preprocessCUDA, gradients are propagated with a per-Gaussian parallel strategy, while renderCUDA uses a per-pixel parallel strategy. However, a single tile (16Ã16 pixels) often contains hundreds or even thousands of Gaussians. In such cases, the available parallelism becomes smaller than the serial workload, causing performance degradation. Moreover, during per-pixel backpropagation, multiple pixels may simultaneously write gradients into the same memory region, leading to thread contention and reduced efficiency. To address these issues, we adopt the per-Gaussian backward pass introduced in Taming-GS [8]. Specifically, during the forward pass we record, every 32 splats, the accumulated transmittance ?? , the blended color ??, and the blended depth ??. In the backward pass, each warp can then independently perform recursive gradient updates for the splats within its group. The comparison between this strategy and the original per-pixel backward pass is shown in Fig. 3. Using the cached ?? and ??, each warp iterates over all pixels within its tile and recursively accumulates the gradient for each splat. Only after all pixels have been processed does it issue a single atomic write to global memory, significantly reducing thread contention. Quantitative comparisons between these acceleration strategies are presented in the experimental section.

## 2.3 Representation: Neural-Gaussians

After applying the aforementioned acceleration strategies, we observed that the reconstruction still could not converge within one minute on an RTX 4090. Even with an early stopping schedule (20,000 iterations), the runtime remained above the one-minute limit. This motivated us to explore alternative scene representations that enable faster convergence, as shown in Fig. 4.

Our analysis indicates that in the original formulation, each splat is treated as an independent leaf node, preventing splats from sharing optimization signals. This leads to inefficient learning. Moreover, scenes containing millions of Gaussian primitives introduce a massive number of parameters, further hindering rapid convergence. To address these limitations, we adopt the Neural-Gaussians representation proposed in Scaffold-GS [7] , where each splat is inferred from an anchor feature rather than optimized directly. This significantly reduces the number of learnable parameters and enables faster convergence.

Concretely, we initialize the sparse point cloud as anchors equipped with learnable features and maintain a shallow global MLP. During each forward pass, we feed the anchor features into the MLP to infer the attributes of their associated child Gaussians. These Gaussians are then rendered in the standard manner, and gradients are propagated back to update only the anchor features and the shared MLP. Quantitative comparisons between these initialization strategies are presented in the experimental section.

<!-- image-->  
Figure 3: Illustration of per-pixel parallelism (left) and per-Gaussian parallelism (right). In per-pixel parallelism, each lane is responsible for processing one pixel, whereas in per-Gaussian parallelism, each lane handles one Gaussian splat.

<!-- image-->

<!-- image-->  
Figure 4: Comparison between using Scaffold-GS as the scene representation (left) and the original 3DGS formulation (right). The Neural-Gaussian representation leads to faster convergence and captures finer scene details.

<!-- image-->

<!-- image-->  
Figure 5: Comparison between non-densified initial point clouds (left) and densified initial point clouds (right). Densifying the initialization introduces more geometric details and improves scene coverage.

## 2.4 Feedforward Initial Points

We observed that the sparse point clouds provided in the first round dataset were insufficient, leading to slow convergence. To address this issue, we explored two strategies for improving anchor initialization: (1) We applied Metric3D-v2 [2] to estimate monocular depth, aligned the predicted depth scale to the SLAM point cloud using RANSAC, and randomly sampled the back-projected 3D points as initial anchor positions. (2) We randomly sampled Gaussian points inferred by the feedforward 3DGS model AnySplat [4] and used them as anchor initialization. Fig. 5 shows that appropriately increasing the density of the initial point cloud leads to faster convergence. Quantitative comparisons between these initialization strategies are presented in the experimental section.

<!-- image-->  
w/o pose optimization

<!-- image-->  
w/ pose optimization  
Figure 6: Comparison between results without pose optimization (left) and with pose optimization (right). Applying pose optimization yields visibly sharper and more accurate renderings.

## 2.5 Pose Optimization

We observed that the camera poses in the first-round dataset were highly inaccurate, which motivated us to optimize the poses during training. Specifically, we maintain a learnable transform delta $\{ \Delta _ { R } , \Delta _ { t } \}$ , representing a corrective rotation and translation applied to all cameras. During training, we accumulate the gradient of the rendering loss with respect to this transform delta at every iteration, and update all camera poses using the optimized correction every 300 iterations. For simplicity and ease of testing, we maintain a single global transform delta rather than an individual correction for each camera. The same global delta estimated during training is then applied to the test cameras as well. As shown in Fig. 6, pose optimization significantly improves the rendering quality.

## 3 Final Round: Fast Reconstruction with Accurate SfM Poses

The final round dataset contains monocular RGB images together with COLMAP [10] poses. We observed that with these much more accurate camera poses, enabling pose optimization actually degraded the rendering quality, as shown in Table 3. Therefore, the pose refinement module used in the first round was disabled in the final round. Meanwhile, we found that integrating anySplat provides high-quality Gaussian initialization, whose attributes are already close to a good solution. This reduces the necessity of Neural-Gaussians. Moreover, the MLP inference overhead in Scaffold-GS introduces a substantial runtime bottleneck, significantly slowing down training. For these reasons, we reverted to the original 3DGS ellipsoidal representation in the final stage. Additionally, we introduced several new modules tailored for the final round. We describe these components in detail in the following sections.

## 3.1 Depth Regularization

We supervise the rendered depth using the monocular depth predicted by Metric3D-v2 [2], which is aligned to the COLMAP scale. At the beginning of training, we assign this depth loss a weight of 0.1 to guide the reconstruction toward the correct scene surface more quickly. For efficiency reasons, our blended depth corresponds to the centroid depth of each Gaussian rather than the rayâprimitive intersection depth, which makes the rendered depth less accurate. Therefore, we gradually reduce the weight of this depth loss to zero as training progresses.

Additionally, we supervise disparity (the inverse depth) instead of raw depth values, which helps mitigate the ambiguity and instability of depth estimation in distant regions.

## 3.2 Multi-view Score Guided Densification

To compensate for the reduced convergence speed after removing the Neural-Gaussian representation, we adopt the multi-view scoreguided densification and pruning strategy introduced in Fast-GS [9]. This strategy provides an efficient mechanism for guiding Gaussian primitives toward faster and more stable convergence.

Specifically, during each densify-and-prune stage, we randomly sample ?? = 10 training views and render their corresponding RGB images. For densification, we first compute the absolute photometric error between the ??-th rendered image and the ground-truth image, followed by normalization:

$$
e _ { u , v } ^ { j } = \mathrm { n o r m a l i z e } ( | | R ^ { j } ( u , v ) - G ^ { j } ( u , v ) | | _ { 1 } )\tag{3}
$$

We then identify pixels whose photometric error exceeds a threshold, forming a binary mask $\mathcal { M } _ { \sf m a s k } ^ { j } ( u , v ) = \mathbb { I } ( e _ { u , v } ^ { j } > \tau )$ , and collect all Gaussian primitives contributing to these masked pixels, denoted as ??. The multi-view consistency score for each primitive ?? â ?? is then computed as:

$$
s _ { i } ^ { + } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \sum _ { u , v } \mathcal { M } _ { \mathrm { m a s k } } ^ { j } ( u , v )\tag{4}
$$

If $s _ { i } ^ { + }$ exceeds a predefined threshold, we clone or split the ??-th Gaussian primitive to increase representational capacity in regions of high error. For pruning, we compute the photometric consistency loss for each view:

$$
E _ { \mathrm { p h o t o } } ^ { j } = ( 1 - \lambda ) L _ { 1 } ^ { j } + \lambda ( 1 - L _ { \mathrm { S S I M } } ^ { j } )\tag{5}
$$

and use the error mask to compute a pruning score:

$$
s _ { i } ^ { - } = \mathrm { n o r m a l i z e } ( \sum _ { j = 1 } ^ { K } \sum _ { u , v } \mathcal { M } _ { \mathrm { m a s k } } ^ { j } ( u , v ) \cdot E _ { \mathrm { p h o t o } } ^ { j } )\tag{6}
$$

If $s _ { i } ^ { - }$ exceeds its threshold, the i-th Gaussian primitive is pruned. Together, these multi-view guided densification and pruning rules allow the model to rapidly allocate representational capacity to higherror regions while eliminating redundant primitives, significantly accelerating convergence in the final round.

## 4 Experiments

We conduct all experiments on an RTX 4090 GPU. The maximum number of training iterations is 6,000 (with Neural-Gaussians) for round 1 and 15,000 for round 2 (without Neural-Gaussians). If the model fails to converge within the 1-minute time limit, we immediately terminate training, save the current Gaussian point cloud, and record the elapsed training time.

For the first round, Table 1 compares the training time (30k iterations on the TNT dataset) under four configurations: (1) baseline

<table><tr><td rowspan=1 colspan=1>TNT(/s â)</td><td rowspan=1 colspan=1>Barn</td><td rowspan=1 colspan=1>Truck</td><td rowspan=1 colspan=1>Ignatius</td><td rowspan=1 colspan=1>Meeting</td><td rowspan=1 colspan=1>Caterp</td></tr><tr><td rowspan=4 colspan=1>3DGS (baseline)w/Bw/B&amp;F&amp;SqW/B&amp;F&amp;LB</td><td rowspan=1 colspan=1>638</td><td rowspan=1 colspan=1>611</td><td rowspan=1 colspan=1>618</td><td rowspan=1 colspan=1>574</td><td rowspan=1 colspan=1>615</td></tr><tr><td rowspan=1 colspan=1>191</td><td rowspan=1 colspan=1>173</td><td rowspan=1 colspan=1>181</td><td rowspan=1 colspan=1>145</td><td rowspan=2 colspan=1>183171</td></tr><tr><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>159</td><td rowspan=1 colspan=1>173</td><td rowspan=1 colspan=1>137</td></tr><tr><td rowspan=1 colspan=1>176</td><td rowspan=1 colspan=1>163</td><td rowspan=1 colspan=1>177</td><td rowspan=1 colspan=1>141</td><td rowspan=1 colspan=1>171</td></tr></table>

Table 1: Comparison of training time on the TNT dataset [6] over 30,000 iterations using three configurations: per-Gaussian parallel backpropagation (w/ B), sequential writing of compact bounding boxes (w/ B & F & Sq), and load-balanced writing of compact bounding boxes (w/ B & F & LB).

<table><tr><td></td><td>| w/o Pose Opt |</td><td>w/o densify</td><td>w/o NG</td><td>Full</td></tr><tr><td>PSNR â</td><td>21.15</td><td>24.89</td><td>23.57</td><td>25.48</td></tr><tr><td>Time(s)â</td><td>58.5</td><td>51.3</td><td>31.7</td><td>60.0</td></tr></table>

Table 2: Ablation study of the used components on the first round challenge dataset.

<table><tr><td></td><td>w/NG</td><td>w/o Depth Prior | w/ Pose Opt</td><td></td><td>Full</td></tr><tr><td>PSNR â</td><td>26.25</td><td>28.61</td><td>28.37</td><td>28.72</td></tr><tr><td>Time(s) â</td><td>60.0</td><td>52.4</td><td>56.3</td><td>56.2</td></tr></table>

Table 3: Ablation study of the used components on the final round challenge dataset.

3DGS, (2) per-Gaussian parallel backpropagation (w/ B), (3) per-Gaussian backpropagation combined with sequential writing of compact bounding boxes (w/ B & F & Sq), and (4) per-Gaussian backpropagation combined with load-balanced writing of compact bounding boxes (w/ B & F & LB).

Table 2 reports the ablation study of all components used in the first round with 6,000 training iterations. We observe that removing pose optimization (w/o Pose Opt) causes a significant drop in rendering quality. Removing Neural-Gaussians (w/o NG) enables reconstruction within 30 seconds but leads to incomplete convergence, resulting in degraded image quality. Without densifying the initial point cloud (w/o densify), background regions cannot be effectively represented, as foreground Gaussians fail to split sufficiently into those areas. With all components enabled (Full), the reconstruction achieves the highest rendering quality.

Table 3 reports the ablation study conducted in the final round with 15,000 training iterations. When the initial camera poses are already sufficiently accurate, enabling pose optimization instead degrades rendering quality (w/ Pose Opt). Removing monocular depth supervision (w/o Depth Prior) slows down early-stage convergence, which negatively impacts the final rendering quality. Using Neural-Gaussians without the multi-view scoreâguided strategy (w/ NG) often fails to complete the full 15,000 iterations, as training is terminated at the 60-second limit, resulting in inferior reconstruction quality. Finally, removing Neural-Gaussians while enabling multi-view score guidance, disabling pose optimization, and incorporating monocular depth supervision (Full) achieves the best overall performance.

## References

[1] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. 2025. Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). 21537â21546. https://speedysplat.github.io/

[2] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. 2024. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

[3] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery. doi:10.1145/ 3641519.3657428

[4] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. 2025. AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views. arXiv preprint arXiv:2505.23716 (2025).

[5] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics 42, 4 (July 2023). https://repo-sam.inria.fr/fungraph/3dgaussian-splatting/

[6] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG) 36, 4 (2017), 1â13.

[7] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. 2024. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20654â20664.

[8] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. 2024. Taming 3DGS: High-Quality Radiance Fields with Limited Resources. In SIGGRAPH Asia 2024 Conference Papers (SA â24). Association for Computing Machinery, New York, NY, USA, Article 2, 11 pages. doi:10.1145/3680528.3687694

[9] Shiwei Ren, Tianci Wen, Yongchun Fang, and Biao Lu. 2025. FastGS: Training 3D Gaussian Splatting in 100 Seconds. arXiv preprint arXiv:2511.04283 (2025).

[10] Johannes Lutz SchÃ¶nberger and Jan-Michael Frahm. 2016. Structure-from-Motion Revisited. In Conference on Computer Vision and Pattern Recognition (CVPR).

[11] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024. Mip-Splatting: Alias-free 3D Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 19447â19456.

[12] Ziyu Zhang, Binbin Huang, Hanqing Jiang, Liyang Zhou, Xiaojun Xiang, and Shunhan Shen. 2025. Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives. In IEEE International Conference on Computer Vision (ICCV).