# Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering

Xiaobin Deng, Changyu Diao\*, Min Li, Ruohan Yu, Duanqing Xu\*

Zhejiang University

## Abstract

Although 3D Gaussian Splatting (3DGS) has achieved impressive performance in real-time rendering, its densification strategy often results in suboptimal reconstruction quality. In this work, we present a comprehensive improvement to the densification pipeline of 3DGS from three perspectives: when to densify, how to densify, and how to mitigate overfitting. Specifically, we propose an Edge-Aware Score to effectively select candidate Gaussians for splitting. We further introduce a Long-Axis Split strategy that reduces geometric distortions introduced by clone and split operations. To address overfitting, we design a set of techniques, including Recovery-Aware Pruning, Multi-step Update, and Growth Control. Our method enhances rendering fidelity without introducing additional training or inference overhead, achieving state-of-theart performance with fewer Gaussians.

Code: https://xiaobin2001.github.io/improved-gs-web

## Introduction

Novel view synthesis (NVS) is a classical problem in computer vision, with widespread applications in virtual reality, cultural heritage preservation, autonomous driving, and other fields. Neural Radiance Field (NeRF) (Mildenhall et al. 2021) introduced the use of neural networks to learn the structure and features of a scene, requiring only multiview 2D images as training data to synthesize novel views. However, NeRF suffers from long synthesis times for individual views (Muller et al. 2022; Fridovich-Keil et al. 2022), Â¨ making real-time rendering challenging. Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) has attracted attention due to its explicit representation and real-time rendering performance.

3DGS represents a scene using a large number of 3D Gaussian ellipsoids. The properties of these Gaussians include position, shape, opacity, and color, all of which can be optimized through differentiable rendering. 3DGS generates an initial set of Gaussians from the sparse points obtained through Structure from Motion (SfM) (Schonberger and Frahm 2016), and subsequently refines the scene representation by adding Gaussians via adaptive density control (ADC).

The densification of 3D Gaussian Splatting has a significant impact on rendering quality, However, the ADC yields unsatisfactory performance. To achieve high-fidelity rendering, we conduct a comprehensive analysis and improvement of the densification strategy in 3DGS, addressing three key aspects: when to densify, how to densify, and how to mitigate overfitting. When to densify: In 3DGS, view-averaged coordinate gradients are used to select Gaussians for densification. However, due to gradient conflicts(Ye et al. 2024), this strategy fails to identify certain large Gaussians that contribute to blurred reconstructions. To select higher-quality candidates, we introduce an Edge-Aware Score (EAS) approach, which combines edge information with pixel-level loss. How to densify: In 3DGS, the shape of Gaussians gradually adapts to the scene through backpropagationbased optimization. However, the clone and split operations induce sudden geometric perturbations in the regions previously represented by the original Gaussians. Minimizing such geometric discrepancies can clearly improve optimization efficiency. Inspired by this insight, we propose Long-Axis Split(LAS), which carefully designs the relative positions, shapes, and opacities of child Gaussians to minimize the geometric difference before and after splitting. Mitigating overfitting: We introduce three techniques to alleviate overfitting: 1. Recovery-Aware Pruning(RAP): This method removes potentially overfitted Gaussians early during the densification phase. 2. Multi-step Update(MU): During training after the densification process, we perform parameter updates every N (N > 1) iterations. 3. Growth Control(GC): A smooth curve is introduced to constrain the growth rate of Gaussian counts.

In summary, our method significantly improves the rendering quality of 3DGS without introducing additional training or rendering overhead. Compared to 3DGS and the current state-of-the-art works, our approach achieves better rendering quality while utilizing fewer Gaussians. Our contributions are summarized as follows:

â¢ We propose an Edge-Aware Score approach to better identify candidates for densification.

â¢ We propose Long-Axis Split, which is designed to minimize geometric inconsistencies introduced during the densification process.

â¢ We present multiple techniques to mitigate overfitting.

â¢ Our work significantly improves the rendering quality of 3DGS without introducing additional computational overhead.

## Related Works

3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) demonstrates outstanding performance in both rendering quality and speed, representing the current state-of-the-art in novel view synthesis. 3DGS has been widely adopted across fields including dynamic scenes (Wu et al. 2024; Lin et al. 2024), simultaneous localization and mapping (SLAM) (Matsuki et al. 2024), 3D content generation (Chen et al. 2024), autonomous driving (Zhou et al. 2024), and high-fidelity human avatars (Shao et al. 2024).

Numerous studies have focused on improving 3DGS rendering quality. For instance, Mip-Splatting (Yu et al. 2024) introduces a 3D smoothing filter and a 2D mipmap filter to eliminate aliasing artifacts present in 3DGS during scaling. To mitigate the impact of defocus blur on reconstruction quality, Deblurring 3DGS (Lee et al. 2024) applies a small multi-layer perceptron (MLP) to the covariance matrix, learning spatially varying blur effects. GaussianPro (Cheng et al. 2024) leverages optimized depth and normal maps to guide densification, filling gaps in areas initialized via SfM. Spec-Gaussian (Yang et al. 2024) employs anisotropic spherical Gaussian appearance fields for Gaussian color modeling, enhancing 3DGS rendering quality in complex scenes with specular and anisotropic surfaces. Notably, all these enhancements rely on the original density control and could benefit from our proposed work.

There are many works dedicated to improving the densification strategies of 3DGS. Mini-Splatting (Fang and Wang 2024) addresses this by generating depth maps for trained scenes to reinitialize the sparse points, and identifies blurred Gaussians with large rendering areas during training, splitting them as needed. Pixel-GS (Zhang et al. 2024) reduces blur through pixel-area-weighted averaging of gradients across views. AbsGS (Ye et al. 2024) attribute blur in reconstructions to conflicts in gradient direction across pixels when computing Gaussian coordinate gradients. This conflict leads to larger Gaussians, which represent blur, receiving insufficient average gradients. To resolve this, they compute Gaussian coordinate gradients by taking the modulus of pixel coordinate gradients before summing. TamingGS (Mallick et al. 2024) proposes a densification judgment condition that employs a weighted combination of multiple scores. RevisingGS (Rota Bulo, Porzi, and Kontschieder \` 2024) optimizes the opacity bias of Gaussians after cloning, and also uses pixel loss as a criterion for selecting candidate points. 3DGS-MCMC (Kheradmand et al. 2024) treats the insertion and optimization of Gaussians as a Stochastic Gradient Langevin Dynamics procedure. SteepGS (Wang et al. 2025) designs an algorithm to compute a splitting matrix that determines whether a Gaussian should be split and where the child Gaussians should be placed. Perceptual-GS (Zhou and Ni 2025) introduces multi-view perceptual sensitivity to guide densification as well as parameter optimization.

We compare against all the above-mentioned works except RevisingGS, which has not made its code publicly accessible. Experiments demonstrate that our method surpasses all the aforementioned works in rendering quality.

## Methods

## Preliminaries

3DGS defines the scene as a set of anisotropic 3D Gaussian primitives:

$$
G ( x ) = \exp { \left( - \frac { 1 } { 2 } ( x ) ^ { T } \Sigma ^ { - 1 } ( x ) \right) } ,\tag{1}
$$

where $\Sigma$ is the 3D covariance matrix and $x$ represents the position relative to the Gaussian mean coordinates. To ensure the semi-definiteness of the covariance matrix, 3DGS reparameterizes it as a combination of a rotation matrix R and a scaling matrix S:

$$
\Sigma = R S S ^ { T } R ^ { T } .\tag{2}
$$

The scaling matrix $S$ can be represented using a 3D vector $s ,$ while the rotation matrix R is obtained from the quaternion $q .$ To render an image from a specified viewpoint, the color of each pixel p is obtained by blending N ordered Gaussians $\{ G _ { i } \mid i = 1 , . . . , N \}$ that cover pixel $p ,$ with the following formula:

$$
C = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{3}
$$

where $\alpha _ { i }$ is the value obtained by projecting $G _ { i }$ onto p and multiplying by the opacity of $G _ { i }$ , while $c _ { i }$ represents the color of $G _ { i } ,$ expressed by SH coefficients.

3DGS initializes the scene using sparse points generated by SfM, and then increases the number and density of Gaussians in the scene through adaptive density control. Specifically, 3DGS calculates the cumulative average view-space positional gradients of Gaussians every 100 iterations, with each iteration training a single viewpoint. The formula for calculating the average gradient is as follows:

$$
\frac { { \sum _ { k = 1 } ^ { M } \sqrt \left( \frac { \partial { \cal L } _ { k } } { \partial \mu _ { \mathbf { k } , \mathbf { x } } ^ { i } } \right) ^ { 2 } + \left( \frac { \partial { \cal L } _ { k } } { \partial \mu _ { \mathbf { k } , \mathbf { y } } ^ { i } } \right) ^ { 2 } } } { { \cal M } ^ { i } } > \tau _ { \mathrm { p o s } } ,\tag{4}
$$

where $M ^ { i }$ represents the number of viewpoints in which the Gaussian participates during a cycle, $\tau _ { \mathrm { p o s } }$ is the given densification threshold, $\frac { \partial L _ { k } } { \partial \mu _ { \bf k , x } ^ { i } }$ and $\frac { { \dot { \sigma } } L _ { k } } { \partial \mu _ { \mathbf { k } , \mathbf { y } } ^ { i } }$ represent the gradients of the Gaussian with respect to the x and $y$ for the current viewpoint, obtained by summing the gradients of each pixel with respect to the coordinates:

$$
\frac { \partial L _ { k } } { \partial \mu _ { \mathrm { k , x } } ^ { i } } = \sum _ { j = 1 } ^ { m } \frac { \partial L _ { j } } { \partial \mu _ { i , x } } , \frac { \partial L _ { k } } { \partial \mu _ { \mathrm { k , y } } ^ { i } } = \sum _ { j = 1 } ^ { m } \frac { \partial L _ { j } } { \partial \mu _ { i , y } } .\tag{5}
$$

Gaussians with average gradients exceeding a predefined threshold undergo densification using either clone or split, depending on their size.

The ADC strategy mainly includes three operations. Clone: Duplicate a small Gaussian with parameters (including position) identical to the parent. Since the clone operation occurs after the rendering step, the cloned Gaussian does not receive gradients during the current iteration. In subsequent parameter updates, only the parent Gaussianâs parameters are modified. Split: A large Gaussian is replaced by two smaller Gaussians, which retain the same shape, opacity, and color as the original. Each smaller Gaussian is scaled down to 1/1.6 of the parentâs size. The coordinates of the two smaller Gaussians are generated through Gaussian sampling, using the parentâs position and covariance matrix as parameters. Reset Opacity: During densification, an opacity reset operation is performed every 3000 iterations. Specifically, the opacity of Gaussians with opacity greater than 0.01 is reset to 0.01.

## Edge-Aware Score

The densification process aims to enhance reconstruction quality by increasing the density of Gaussians. Human vision is particularly sensitive to blurring in edge regions, yet conventional pixel-wise losses fail to adequately capture these areas. To address this limitation, we propose an edgeaware scoring mechanism.

First, we apply the Laplacian operator to each training image for edge detection, generating per-pixel edge weights. Then, for each Gaussian, we compute its edge-aware score under a given view using the following formula:

$$
S _ { i , j } = \sum _ { p \in \mathcal { P } } \omega _ { \mathrm { p , j } } ^ { i } \cdot \alpha _ { \mathrm { p , j } } ^ { i }\tag{6}
$$

where $\omega _ { \mathrm { p , j } } ^ { i }$ denotes the edge weight of pixel p in view $j ,$ and $\alpha _ { \mathrm { p , j } } ^ { i }$ represents the rendering weight of Gaussian i at pixel $p$ in view j. Before each densification step, we randomly sample $N _ { s }$ views from the training set and compute the average edge-aware score $\bar { S } _ { i }$ of each Gaussian across these views:

$$
\bar { S } _ { i } = \frac { 1 } { N _ { s } } \sum _ { j = 1 } ^ { N } S _ { i , j }\tag{7}
$$

The edge-aware score represents the contribution of a Gaussian to image edge regions, but it is not sufficient to identify the rendering quality of the region in which it resides. To address this issue, we use the view-averaged absolute coordinate gradients as the second criterion for evaluation. The absolute coordinate gradient is obtained by replacing Equation (4) with:

$$
{ \frac { \partial L _ { k } } { \partial \mu _ { { \bf k } , { \bf x } } ^ { i } } } = \sum _ { j = 1 } ^ { m } \left| { \frac { \partial L _ { j } } { \partial \mu _ { i , x } } } \right| , \quad { \frac { \partial L _ { k } } { \partial \mu _ { { \bf k } , { \bf y } } ^ { i } } } = \sum _ { j = 1 } ^ { m } \left| { \frac { \partial L _ { j } } { \partial \mu _ { i , y } } } \right| .\tag{8}
$$

The absolute gradient effectively approximates the pixelwise loss and can be used to evaluate the rendering quality of the region occupied by the Gaussian. Gaussians with an average absolute gradient greater than a specified threshold are selected as candidates. The probability of splitting a candidate Gaussian is proportional to its average edge-aware score.

## Long-Axis Split

Minimizing the geometric disturbance introduced during the densification process can accelerate optimization speed and sometimes even improve the final rendering quality. The split operation in 3DGS generates the relative positions of child Gaussians using probabilistic sampling, which introduces randomness and limits fine-grained shape control. In our approach, we fix the positions of the child Gaussians to be symmetric along the longest axis of the original Gaussian, with the original center as the midpoint. Under this configuration, the position of each child Gaussian is solely determined by the distance from its center to the original Gaussian center. The three axes of the child Gaussians are initialized with the same values and then gradually optimized to become anisotropic. When a rendering ray is parallel to one of the axes, that axis will not contribute to the rendering, so the longest axis has the minimal probability of being aligned with the rendering rays during projection. Therefore, we choose the longest axis as the splitting direction to maximize the rendering contribution.

<!-- image-->

<!-- image-->  
Figure 1: Compare the geometric differences before and after splitting between split and LAS. The outer dashed line represents the Gaussian shape before splitting.

Let d denote the distance from the center of a child Gaussian to the original Gaussian center, and let $L _ { 0 }$ be the semilength of the original Gaussianâs longest axis. The semilength $L _ { s }$ of the childâs longest axis should be adjusted to $L _ { 0 } - d ,$ ensuring that the two child Gaussians are tangent to the original Gaussian at its endpoints. As d decreases from $L _ { 0 }$ to $0 . 5 L _ { 0 } ,$ the shape difference between before and after the split persists. Further reducing d from $0 . 5 L _ { 0 }$ to 0 reduces the shape difference even more but increases the overlapping area between the two child Gaussians.

To determine d (with the constraint $d ~ \leq ~ 0 . 5 L _ { 0 } )$ while minimizing the shape difference, the lengths of the other two axes of the child Gaussians, denoted $R _ { s }$ , should satisfy:

$$
R _ { s } = R _ { 0 } \cdot { \sqrt { 1 - { \frac { d ^ { 2 } } { L _ { 0 } ^ { 2 } } } } }\tag{9}
$$

This ensures that the endpoints of the two minor axes of the child Gaussians lie exactly on the surface of the original Gaussian. A detailed proof is provided in the Appendix A.4. Gaussian overlap may interfere with their individual optimization. However, the rendering weights near the edges of Gaussians are significantly lower than those at the centers, making small overlaps at the edges acceptable. In practice, we set $d = 0 . 4 5 L _ { 0 }$ to balance the trade-off between minimizing overlap and maintaining low shape difference (see Figure 1). See Appendix A.8 for the ablation study.

After split, the coverage area of the original Gaussian transitions from a single-center distribution to a dual-center distribution, which introduces a discrepancy from the perspective of density distribution. Reducing the opacity of the child Gaussians appropriately can help mitigate this discrepancy. In practice, we set the opacity of the child Gaussians to 60% of the original opacity. 3DGS uses clone and split to address under-reconstruction and over-reconstruction. However, during optimization, the size of Gaussians tends to converge to a value that balances under-reconstruction and over-reconstruction to minimize loss. Therefore, we only use Long-Axis Split as the densification operation. Figure 2 shows that compared to split, the splitting error introduced by LAS is smaller.

<!-- image-->  
Figure 2: Evaluate the drop in PSNR after splitting using different splitting strategies, test scene is bicycle, a smaller drop indicates less geometric error introduced by the splitting.

## Recovery-Aware Pruning

During training, 3DGS may generate some overfitted Gaussians that contribute less from the training viewpoints but negatively impact generalization performance. The original 3DGS resets the opacity at iterations 3K, 6K, 9K, and 12K. By leveraging the difference in opacity recovery speed, we can eliminate some of these overfitted Gaussians. Specifically, we prune the bottom 20% Gaussians in terms of opacity at iteration 3300 and 6300. Since the densification process continues until iteration 15K and RAP is only applied in the early stages, it does not affect the final number of Gaussians or the training speed.

## Muti-view Update

In 3DGS, each training iteration processes and updates parameters based on a single view, i.e., the batch size is 1. Increasing the batch size can improve generalization performance; however, training on multiple views per iteration significantly increases the training cost. In contrast, indirectly increasing the effective batch size by extending the parameter update interval (N > 1) can reduce training overhead. Unfortunately, this approach leads to a notable degradation in final rendering quality (see Appendix A.6). This is due to two main reasons:

â¢ The densification phase requires rapid optimization after splitting. Otherwise, Gaussians that have not been sufficiently optimized may be incorrectly split again.

â¢ Extending the update interval reduces the frequency of gradient updates, thereby slowing down the convergence of the parameters.

We observe that the optimization speed naturally slows down after the densification phase, which reduces the dependency on frequent updates. At this stage, extending the update interval no longer causes erroneous splits. Therefore, we propose a two-stage training strategy:

â¢ In the early training phase, we maintain single-view updates to meet the optimization requirements during densification.

â¢ After densification is complete (specifically, between 15,000 and 22,500 iterations, we use N=5, followed by N=20 afterward), we switch to multi-view batching. This enables joint optimization of computational efficiency and generalization performance.

## Growth Control

When using the Edge Awareness Score to select split candidates, we observed that the number of Gaussians reaches its peak early in the densification stage. An early peak in Gaussian count increases the risk of overfitting and prolongs the overall training time. To address this, we employ a smooth convex curve as the control curve for the number of Gaussians, ensuring it meets two requirements: rapid densification in the early stages to maintain training efficiency, and a peak in Gaussian count only near the end of the densification phase. Specifically, the Gaussian budget N for the current training iteration is calculated using the following formula:

$$
N = N _ { m a x } \cdot \sqrt { \frac { I - I _ { s t a r t } } { I _ { e n d } - I _ { s t a r t } } }\tag{10}
$$

where I is the current round, with $I _ { s t a r t }$ and $I _ { e n d }$ representing the start and end iteration of the calculation, respectively, and $N _ { m a x }$ is the user-defined final budget. Appendix A.9 shows the shape of the growth curves.

## Experiments

## Datasets and metrics

We evaluated our method on real-world scenes from the Mip-NeRF 360 (Barron et al. 2022), Tanks and Temples (Knapitsch et al. 2017), and Deep Blending (Hedman et al. 2018) datasets. As with 3DGS, we selected all nine scenes from the Mip-NeRF 360 dataset, including five outdoor scenes and four indoor scenes. For the Tanks and Temples dataset, we chose the train and truck scenes, and for the Deep Blending dataset, we selected the drjohnson and playroom scenes. In each experiment, every 8th image was used as the validation set. We report peak signal-to-noise ratio (PSNR), structural similarity (SSIM), and perceptual metric (LPIPS) from (Zhang et al. 2018) as quality evaluation metrics. All three scores were calculated using the methods provided by 3DGS.

## Implementation

We built our code upon the open-source repository of TamingGS. According to the scene size, we divided the 13 scenes into three categories, with each category having the same Gaussian budget. The budget for bicycle, garden, and stump is 3M. The budget for flowers, treehill, drjohnson, and truck is 1.5M. The remaining scenes have a budget of 1M.

<!-- image-->  
Figure 3: Qualitative comparison results among scenes garden, drjohnson, train.

We reproduced 3DGS, AbsGS, PixelGS, MiniSplatting-D (MiniGS), TamingGS, 3DGS-MCMC, SteepGS, and Perceptual-GS as baseline methods, using all default parameter settings from their respective original papers. All results represent the best performance obtained over three runs. All experiments were conducted on a single 4090D GPU.

## Qualitative Analysis

The results of the comprehensive qualitative comparison are shown in Figure 3, the results for the remaining 10 scenes can be found in Appendix A.3.

In the garden scene, our method significantly outperforms other works in terms of wall brick texture details and overall image cleanliness. TamingGS achieves a relatively clean result but fails to recover detailed textures. AbsGS is able to reconstruct most textures but suffers from poor image cleanliness. 3DGS-MCMC and SteepGS perform the worst: the former produces an overall blurry reconstruction, while the latter results in an unacceptably messy output. Notably, only our method successfully recovers the white bricks on the right side of the wall.

In the drjohnson scene, our method delivers the highest quality in capturing fine details such as ceiling bulbs and sensors, while also maintaining the cleanest visual output. MiniGS performs well in detail reconstruction; however, it uses several times more Gaussian budget than ours and shows inferior performance in image cleanliness. PerceptualGS ranks second overall but struggles with large straightline regions. Other methods perform poorly, with 3DGS-MCMC completely missing many detailed areas.

In the train scene, only our method fully reconstructs the objects on the hillside. All other methods exhibit noticeable blurred regions. TamingGS ranks second in this case but still lags significantly behind our method. Although SteepGS uses the least Gaussian budget, its reconstruction quality is the worst.

Overall, our method leads over other approaches in terms of image cleanliness, detail recovery, and budget efficiency.

<table><tr><td rowspan="2">Dataset MethodâMetric</td><td colspan="4">Mip-NeRF360 PSNRâ  LPIPSâ</td><td colspan="4">Deep Blending</td><td colspan="4">Tanks&amp;Temples</td></tr><tr><td>SSIMâ</td><td></td><td></td><td>Num</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>Num</td><td>SSIMâ</td><td></td><td>PSNRâ LPIPSâ</td><td>Num</td></tr><tr><td>3DGS</td><td>0.815</td><td>27.48</td><td>0.216</td><td>3337659</td><td>0.904</td><td>29.57</td><td>0.244</td><td>2832494</td><td>0.848</td><td>23.69</td><td>0.177</td><td>1847041</td></tr><tr><td>AbsGS (ACMMM24)</td><td>0.820</td><td>27.52</td><td>0.198</td><td>3194225</td><td>0.905</td><td>29.49</td><td>0.243</td><td>2054043</td><td>0.857</td><td>23.83</td><td>0.164</td><td>1442984</td></tr><tr><td>PixelGS (ECCV24)</td><td>0.824</td><td>27.62</td><td>0.189</td><td>5619828</td><td>0.897</td><td>28.98</td><td>0.248</td><td>4644260</td><td>0.857</td><td>23.84</td><td>0.149</td><td>4519926</td></tr><tr><td>MiniSplatting-D (ECCV24)</td><td>0.832</td><td>27.57</td><td>0..176</td><td>4685127</td><td>0.906</td><td>29.93</td><td>0.211</td><td>4627579</td><td>0.855</td><td>23.36</td><td>0.140</td><td>4260423</td></tr><tr><td>TamingGS (SIGGRAPHAsia24)</td><td>0.822</td><td>27.96</td><td>0.207</td><td>3182444</td><td>0.907</td><td>29.93</td><td>0.236</td><td>2799868</td><td>0.860</td><td>24.42</td><td>0.163</td><td>1849918</td></tr><tr><td>3DGS-MCMC (NeurIPS24)</td><td>0.835</td><td>28.01</td><td>0.186</td><td>3227778</td><td>0.912</td><td>29.78</td><td>0.237</td><td>2950000</td><td>0.869</td><td>24.40</td><td>0.149</td><td>1850000</td></tr><tr><td>SteepGS (CVPR25)</td><td>0.795</td><td>27.05</td><td>0.247</td><td>2193808</td><td>0.905</td><td>29.74</td><td>0.251</td><td>1605267</td><td>0.838</td><td>23.42</td><td>0.193</td><td>1310323</td></tr><tr><td>Perceptual-GS (ICML25)</td><td>0.829</td><td>27.77</td><td>0.187</td><td>2685908</td><td>0.906</td><td>29.88</td><td>0.231</td><td>2892183</td><td>0.857</td><td>23.88</td><td>0.150</td><td>1721090</td></tr><tr><td>Ours</td><td>0.836</td><td>28.19</td><td>0.186</td><td>1777778</td><td>0.913</td><td>30.19</td><td>0.226</td><td>1250000</td><td>0.872</td><td>24.59</td><td>0.145</td><td>1250000</td></tr></table>

Table 1: Quantitative results on the Mip-NeRF 360, Deep Blending, and Tanks and Temples datasets. Cells are highlighted as follows: best , and second best . All parameters for the compared methods follow the default settings from their respective original papers.

<!-- image-->  
bicycle

<!-- image-->  
Ours Full

<!-- image-->  
w/o EAS

<!-- image-->  
w/o EAS (abs gradient)  
Figure 4: Qualitative comparison for evaluating the effectiveness of EAS.

## Quantitative Analysis

The quantitative comparison results are shown in Table 1. Our method achieves the best results on all three datasets. Among all compared methods, our approach uses the least Gaussian budget across all datasets, yet leads in both SSIM and PSNR metrics. In terms of the LPIPS metric, our method is only slightly outperformed by MiniGS, which however employs nearly three times as many Gaussians as ours. Although TamingGS and 3DGS-MCMC perform well in SSIM and PSNR, they both suffer from noticeable rendering quality issues: TamingGS struggles with certain blurred regions, while 3DGS-MCMC performs poorly in distant details. Perceptual-GS is the most balanced among the baseline methods, but still lags significantly behind our approach.

Compared to works that achieve rendering quality better than 3DGS, we consistently achieve the best performance under the smallest budget in all scenes. Detailed per-scene metrics can be found in the Appendix A.1.

Due to differences in CUDA kernels used, it is not possible to directly compare training speed across different methods (e.g., the kernel used in TamingGS is significantly faster than that of 3DGS). Under the same CUDA kernel, 3DGS-MCMC, PerceptualGS, and SteepGS require additional computational overhead, whereas other methods, including our approach, have computational costs proportional to the allocated Gaussian budget. Compared to TamingGS, the fastest method in terms of training speed among the baselines (14.2 minutes), our approach further reduces the training time by half (6.7 minutes).

<table><tr><td>MethodâMetric</td><td>|SSIMâ PSNRâ</td><td>LPIPSâ</td><td>Time</td><td>FPS</td></tr><tr><td>Ours Full</td><td>0.854 27.95</td><td>0.186</td><td>6.7</td><td>289</td></tr><tr><td>w/o EAS</td><td>0.837 27.77</td><td>0.220</td><td>6.4</td><td>346</td></tr><tr><td>w/o EAS(abs gradient)</td><td>0.852 27.95</td><td>0.192</td><td>7.5</td><td>306</td></tr><tr><td>w/o LAS</td><td>0.846 27.81</td><td>0.195</td><td>7.8</td><td>239</td></tr><tr><td>w/o RAP</td><td>0.851 27.79</td><td>0.187</td><td>7.8</td><td>267</td></tr><tr><td>w/o MU</td><td>0.852</td><td>27.80</td><td>0.187 8.4</td><td>292</td></tr><tr><td>w/o GC</td><td>0.853</td><td>27.94</td><td>0.188</td><td>7.8 301</td></tr><tr><td>w/o RAP&amp;MU&amp;GC</td><td>0.848</td><td>27.67</td><td>0.193</td><td>9.1 286</td></tr></table>

Table 2: Results of the ablation study on all datasets, where all configurations share the same budget.

## Ablation Experiments

The overall ablation study results are summarized in Table 2. Since the ablation experiments do not change the CUDA kernel or the Gaussian budget, the training time and FPS metrics can be directly compared.

Effect of EAS: Refer to Rows 1, 2, and 3 in Table 2, EAS improves rendering quality, especially significantly in terms of the LPIPS metric. The improvement in quality comes from two aspects: first, EAS performs well in identifying blurry regions; second, it enhances reconstruction quality at edge regions. As shown in Figure 4, EAS helps eliminate reconstruction blur. Compared to using only the absolute gradient, EAS achieves better reconstruction at leaf edges. Although EAS increases rendering overhead, this is a natural trade-off for eliminating reconstruction blur.

Effect of LAS: Refer to Rows 1 and 4 in Table 2, LAS improves rendering quality while reducing rendering overhead. LAS reduces rendering cost because, compared to clone and split, the resulting child Gaussians after LAS exhibit lower overlap ratios, thereby shortening the sequence of Gaussians involved in rendering each pixel. The impact of Gaussian overlap on FPS can be found in Appendix A.8. The improvement in rendering quality is attributed to the minimization of geometric discrepancies before and after splitting. As illustrated in Figure 5, LAS accelerates optimization, which aligns with our hypothesis. As shown in Figure 6, LAS also improves detail reconstruction.

<!-- image-->  
Figure 5: Evaluating the impact of LAS on optimization speed.

<!-- image-->  
Figure 6: Qualitative comparison for evaluating the effectiveness of LAS. Only LAS enables the recovery of the white brick patch on the wall in the garden scene.

Mitigating Overfitting: Referring to the lower part of Table 2, we observe that RAP, MU, and GC all moderately enhance generalization performance and reduce training cost, with RAP also lowering rendering overhead.

RAP mainly improves performance by eliminating certain overfitted Gaussians, some of which contribute to floating artifacts that severely degrade rendering quality. As seen in Figure 7, RAP effectively reduces the occurrence probability of such artifacts. The reduction in rendering overhead may result from the removal of overfitted Gaussians and their descendants, which otherwise elongate the rendering sequence.

<!-- image-->  
Ours Full

<!-- image-->  
w/o RAP

Figure 7: Qualitative comparison for evaluating the effectiveness of RAP.  
<!-- image-->  
drjohnson

Figure 8: Qualitative comparison for evaluating the effectiveness of overfitting mitigation strategies.

MU accelerates training by reducing the total number of parameter updates, while GC speeds up training by lowering the peak time of Gaussian count.

Comparing the first and last rows of Table 2, it is evident that the combined effect of these three overfitting mitigation components leads to a significant overall improvement. The qualitative comparison can be referred to in Figure 8.

## Conclusion

We optimized the densification process of 3D Gaussian Splatting (3DGS) from three perspectives: when to densify, how to densify, and how to alleviate overfitting. The proposed method significantly improves rendering quality without introducing additional computational overhead. Compared to current state-of-the-art densification approaches, our method achieves better reconstruction quality with fewer Gaussians. Additionally, our work does not rely on specific CUDA kernels and can be easily integrated into existing 3DGS-based methods, making a significant contribution to the broader adoption of 3DGS.

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2022. Mip-nerf 360: Unbounded antialiased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5470â5479.

Chen, Z.; Wang, F.; Wang, Y.; and Liu, H. 2024. Text-to-3d using gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21401â21412.

Cheng, K.; Long, X.; Yang, K.; Yao, Y.; Yin, W.; Ma, Y.; Wang, W.; and Chen, X. 2024. Gaussianpro: 3d gaussian splatting with progressive propagation. In Forty-first International Conference on Machine Learning.

Fang, G.; and Wang, B. 2024. Mini-splatting: Representing scenes with a constrained number of gaussians. In European Conference on Computer Vision, 165â181. Springer.

Fridovich-Keil, S.; Yu, A.; Tancik, M.; Chen, Q.; Recht, B.; and Kanazawa, A. 2022. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5501â 5510.

Hedman, P.; Philip, J.; Price, T.; Frahm, J.-M.; Drettakis, G.; and Brostow, G. 2018. Deep blending for free-viewpoint image-based rendering. ACM Transactions on Graphics (ToG), 37(6): 1â15.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph., 42(4): 139â1.

Kheradmand, S.; Rebain, D.; Sharma, G.; Sun, W.; Tseng, Y.-C.; Isack, H.; Kar, A.; Tagliasacchi, A.; and Yi, K. M. 2024. 3d gaussian splatting as markov chain monte carlo. Advances in Neural Information Processing Systems, 37: 80965â80986.

Knapitsch, A.; Park, J.; Zhou, Q.-Y.; and Koltun, V. 2017. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4): 1â 13.

Lee, B.; Lee, H.; Sun, X.; Ali, U.; and Park, E. 2024. Deblurring 3d gaussian splatting. In European Conference on Computer Vision, 127â143. Springer.

Lin, Y.; Dai, Z.; Zhu, S.; and Yao, Y. 2024. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21136â21145.

Mallick, S. S.; Goel, R.; Kerbl, B.; Steinberger, M.; Carrasco, F. V.; and De La Torre, F. 2024. Taming 3dgs: High-quality radiance fields with limited resources. In SIG-GRAPH Asia 2024 Conference Papers, 1â11.

Matsuki, H.; Murai, R.; Kelly, P. H.; and Davison, A. J. 2024. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18039â18048.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Muller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In-Â¨ stant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1â 15.

Rota Bulo, S.; Porzi, L.; and Kontschieder, P. 2024. Revising \` densification in gaussian splatting. In European Conference on Computer Vision, 347â362. Springer.

Schonberger, J. L.; and Frahm, J.-M. 2016. Structure-frommotion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4104â4113.

Shao, Z.; Wang, Z.; Li, Z.; Wang, D.; Lin, X.; Zhang, Y.; Fan, M.; and Wang, Z. 2024. Splattingavatar: Realistic realtime human avatars with mesh-embedded gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1606â1616.

Wang, P.; Wang, Y.; Wang, D.; Mohan, S.; Fan, Z.; Wu, L.; Cai, R.; Yeh, Y.-Y.; Wang, Z.; Liu, Q.; et al. 2025. Steepest Descent Density Control for Compact 3D Gaussian Splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, 26663â26672.

Wu, G.; Yi, T.; Fang, J.; Xie, L.; Zhang, X.; Wei, W.; Liu, W.; Tian, Q.; and Wang, X. 2024. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20310â20320.

Yang, Z.; Gao, X.; Sun, Y.-T.; Huang, Y.; Lyu, X.; Zhou, W.; Jiao, S.; Qi, X.; and Jin, X. 2024. Spec-gaussian: Anisotropic view-dependent appearance for 3d gaussian splatting. Advances in Neural Information Processing Systems, 37: 61192â61216.

Ye, Z.; Li, W.; Liu, S.; Qiao, P.; and Dou, Y. 2024. Absgs: Recovering fine details in 3d gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, 1053â1061.

Yu, Z.; Chen, A.; Huang, B.; Sattler, T.; and Geiger, A. 2024. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19447â19456.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586â595.

Zhang, Z.; Hu, W.; Lao, Y.; He, T.; and Zhao, H. 2024. Pixelgs: Density control with pixel-aware gradient for 3d gaussian splatting. In European Conference on Computer Vision, 326â342. Springer.

Zhou, H.; and Ni, Z. 2025. Perceptual-GS: Scene-adaptive Perceptual Densification for Gaussian Splatting. arXiv preprint arXiv:2506.12400.

Zhou, X.; Lin, Z.; Shan, X.; Wang, Y.; Sun, D.; and Yang, M.-H. 2024. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21634â21643.

## Quantitative Comparison per Scene

<table><tr><td rowspan=1 colspan=1>SceneâMethods</td><td rowspan=1 colspan=1>3DGS</td><td rowspan=1 colspan=1>AbsGS</td><td rowspan=1 colspan=1>PixelGS</td><td rowspan=1 colspan=1>MiniGS</td><td rowspan=1 colspan=1>TamGS</td><td rowspan=1 colspan=1>MCMC</td><td rowspan=1 colspan=1>SteepGS</td><td rowspan=1 colspan=1>PercepGS</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>bicycle</td><td rowspan=1 colspan=1>0.76561</td><td rowspan=1 colspan=1>0.78243</td><td rowspan=1 colspan=1>0.77857</td><td rowspan=1 colspan=1>0.79877</td><td rowspan=1 colspan=1>0.77766</td><td rowspan=1 colspan=1>0.79896</td><td rowspan=1 colspan=1>0.73151</td><td rowspan=1 colspan=1>0.79283</td><td rowspan=2 colspan=1>0.803690.63902</td></tr><tr><td rowspan=1 colspan=1>flowers</td><td rowspan=1 colspan=1>0.60522</td><td rowspan=1 colspan=1>0.62010</td><td rowspan=1 colspan=1>0.63628</td><td rowspan=1 colspan=1>0.64182</td><td rowspan=1 colspan=1>0.61695</td><td rowspan=1 colspan=1>0.64659</td><td rowspan=1 colspan=1>0.54756</td><td rowspan=1 colspan=1>0.63873</td></tr><tr><td rowspan=1 colspan=1>garden</td><td rowspan=1 colspan=1>0.86670</td><td rowspan=1 colspan=1>0.86803</td><td rowspan=1 colspan=1>0.87039</td><td rowspan=1 colspan=1>0.87853</td><td rowspan=1 colspan=1>0.87378</td><td rowspan=1 colspan=1>0.87676</td><td rowspan=1 colspan=1>0.85584</td><td rowspan=1 colspan=1>0.87014</td><td rowspan=1 colspan=1>0.88139</td></tr><tr><td rowspan=1 colspan=1>stump</td><td rowspan=1 colspan=1>0.77057</td><td rowspan=1 colspan=1>0.78059</td><td rowspan=1 colspan=1>0.78454</td><td rowspan=1 colspan=1>0.80582</td><td rowspan=1 colspan=1>0.77425</td><td rowspan=1 colspan=1>0.81188</td><td rowspan=1 colspan=1>0.73186</td><td rowspan=1 colspan=1>0.79787</td><td rowspan=1 colspan=1>0.80930</td></tr><tr><td rowspan=1 colspan=1>treehill</td><td rowspan=1 colspan=1>0.63369</td><td rowspan=1 colspan=1>0.62100</td><td rowspan=1 colspan=1>0.63534</td><td rowspan=1 colspan=1>0.64207</td><td rowspan=1 colspan=1>0.64464</td><td rowspan=1 colspan=1>0.65881</td><td rowspan=1 colspan=1>0.61224</td><td rowspan=1 colspan=1>0.64299</td><td rowspan=1 colspan=1>0.65686</td></tr><tr><td rowspan=1 colspan=1>bonsai</td><td rowspan=1 colspan=1>0.94217</td><td rowspan=1 colspan=1>0.94456</td><td rowspan=1 colspan=1>0.94657</td><td rowspan=1 colspan=1>0.94805</td><td rowspan=1 colspan=1>0.94532</td><td rowspan=1 colspan=1>0.94784</td><td rowspan=1 colspan=1>0.93827</td><td rowspan=1 colspan=1>0.94786</td><td rowspan=1 colspan=1>0.94911</td></tr><tr><td rowspan=2 colspan=1>counterkitchen</td><td rowspan=1 colspan=1>0.90860</td><td rowspan=1 colspan=1>0.91116</td><td rowspan=1 colspan=1>0.91465</td><td rowspan=1 colspan=1>0.91044</td><td rowspan=1 colspan=1>0.91327</td><td rowspan=1 colspan=1>0.91709</td><td rowspan=1 colspan=1>0.90180</td><td rowspan=2 colspan=1>0.915200.93130</td><td rowspan=2 colspan=1>0.919090.93617</td></tr><tr><td rowspan=1 colspan=1>kitchen</td><td rowspan=1 colspan=1>0.92771</td><td rowspan=1 colspan=1>0.93075</td><td rowspan=1 colspan=1>0.93117</td><td rowspan=1 colspan=1>0.93344</td><td rowspan=1 colspan=1>0.93180</td><td rowspan=1 colspan=1>0.93339</td><td rowspan=1 colspan=1>0.92294</td></tr><tr><td rowspan=1 colspan=1>room</td><td rowspan=1 colspan=1>0.91915</td><td rowspan=1 colspan=1>0.92565</td><td rowspan=1 colspan=1>0.92242</td><td rowspan=1 colspan=1>0.92808</td><td rowspan=1 colspan=1>0.92460</td><td rowspan=1 colspan=1>0.92954</td><td rowspan=1 colspan=1>0.91558</td><td rowspan=1 colspan=1>0.92826</td><td rowspan=1 colspan=1>0.93194</td></tr><tr><td rowspan=1 colspan=1>playroom</td><td rowspan=1 colspan=1>0.90743</td><td rowspan=1 colspan=1>0.90871</td><td rowspan=1 colspan=1>0.90571</td><td rowspan=1 colspan=1>0.90753</td><td rowspan=1 colspan=1>0.90612</td><td rowspan=1 colspan=1>0.91606</td><td rowspan=1 colspan=1>0.90630</td><td rowspan=1 colspan=1>0.90715</td><td rowspan=2 colspan=1>0.915870.91055</td></tr><tr><td rowspan=1 colspan=1>drjohnson</td><td rowspan=1 colspan=1>0.90096</td><td rowspan=1 colspan=1>0.90194</td><td rowspan=1 colspan=1>0.88791</td><td rowspan=1 colspan=1>0.90505</td><td rowspan=1 colspan=1>0.90801</td><td rowspan=1 colspan=1>0.90809</td><td rowspan=1 colspan=1>0.90279</td><td rowspan=1 colspan=1>0.90454</td></tr><tr><td rowspan=2 colspan=1>traintruck</td><td rowspan=1 colspan=1>0.81321</td><td rowspan=1 colspan=1>0.82877</td><td rowspan=1 colspan=1>0.82760</td><td rowspan=1 colspan=1>0.82125</td><td rowspan=1 colspan=1>0.82614</td><td rowspan=1 colspan=1>0.83955</td><td rowspan=1 colspan=1>0.80120</td><td rowspan=1 colspan=1>0.82613</td><td rowspan=1 colspan=1>0.84640</td></tr><tr><td rowspan=1 colspan=1>0.88184</td><td rowspan=1 colspan=1>0.88549</td><td rowspan=1 colspan=1>0.88693</td><td rowspan=1 colspan=1>0.88939</td><td rowspan=1 colspan=1>0.89293</td><td rowspan=1 colspan=1>0.89926</td><td rowspan=1 colspan=1>0.87559</td><td rowspan=1 colspan=1>0.88798</td><td rowspan=1 colspan=1>0.89837</td></tr></table>

Table 3: The SSIM scores for all works in each scene.

<table><tr><td rowspan=1 colspan=1>SceneâMethods</td><td rowspan=1 colspan=1>3DGS</td><td rowspan=1 colspan=1>AbsGS</td><td rowspan=1 colspan=1>PixelGS</td><td rowspan=1 colspan=1>MiniGS</td><td rowspan=1 colspan=1>TamGS</td><td rowspan=1 colspan=1>MCMC</td><td rowspan=1 colspan=1>SteepGS</td><td rowspan=1 colspan=1>PercepGS</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>bicycle</td><td rowspan=1 colspan=1>25.213</td><td rowspan=1 colspan=1>25.372</td><td rowspan=1 colspan=1>25.279</td><td rowspan=1 colspan=1>25.581</td><td rowspan=1 colspan=1>25.504</td><td rowspan=1 colspan=1>25.681</td><td rowspan=1 colspan=1>24.798</td><td rowspan=1 colspan=1>25.527</td><td rowspan=1 colspan=1>25.866</td></tr><tr><td rowspan=1 colspan=1>flowers</td><td rowspan=1 colspan=1>21.539</td><td rowspan=1 colspan=1>21.368</td><td rowspan=1 colspan=1>21.580</td><td rowspan=1 colspan=1>21.526</td><td rowspan=1 colspan=1>21.871</td><td rowspan=1 colspan=1>22.010</td><td rowspan=1 colspan=1>20.713</td><td rowspan=1 colspan=1>21.494</td><td rowspan=1 colspan=1>21.768</td></tr><tr><td rowspan=1 colspan=1>garden</td><td rowspan=1 colspan=1>27.361</td><td rowspan=1 colspan=1>27.408</td><td rowspan=1 colspan=1>27.493</td><td rowspan=1 colspan=1>27.693</td><td rowspan=1 colspan=1>27.898</td><td rowspan=1 colspan=1>27.811</td><td rowspan=1 colspan=1>27.125</td><td rowspan=1 colspan=1>27.628</td><td rowspan=1 colspan=1>28.098</td></tr><tr><td rowspan=1 colspan=1>stump</td><td rowspan=1 colspan=1>26.539</td><td rowspan=1 colspan=1>26.726</td><td rowspan=1 colspan=1>26.843</td><td rowspan=1 colspan=1>27.140</td><td rowspan=1 colspan=1>26.632</td><td rowspan=1 colspan=1>27.384</td><td rowspan=1 colspan=1>25.832</td><td rowspan=1 colspan=1>27.030</td><td rowspan=1 colspan=1>27.211</td></tr><tr><td rowspan=1 colspan=1>treehill</td><td rowspan=1 colspan=1>22.495</td><td rowspan=1 colspan=1>22.094</td><td rowspan=1 colspan=1>22.296</td><td rowspan=1 colspan=1>22.234</td><td rowspan=1 colspan=1>23.024</td><td rowspan=1 colspan=1>22.944</td><td rowspan=1 colspan=1>22.204</td><td rowspan=4 colspan=1>22.40832.60729.28831.909</td><td rowspan=3 colspan=1>22.83932.96629.798</td></tr><tr><td rowspan=1 colspan=1>bonsai</td><td rowspan=1 colspan=1>32.242</td><td rowspan=1 colspan=1>32.145</td><td rowspan=1 colspan=1>32.547</td><td rowspan=1 colspan=1>32.163</td><td rowspan=1 colspan=1>32.889</td><td rowspan=1 colspan=1>32.646</td><td rowspan=1 colspan=1>31.806</td></tr><tr><td rowspan=1 colspan=1>counter</td><td rowspan=1 colspan=1>29.016</td><td rowspan=1 colspan=1>29.096</td><td rowspan=1 colspan=1>29.181</td><td rowspan=1 colspan=1>28.560</td><td rowspan=1 colspan=1>29.486</td><td rowspan=1 colspan=1>29.348</td><td rowspan=2 colspan=1>28.74330.919</td></tr><tr><td rowspan=1 colspan=1>kitchen</td><td rowspan=1 colspan=1>31.474</td><td rowspan=1 colspan=1>31.852</td><td rowspan=1 colspan=1>31.752</td><td rowspan=1 colspan=1>31.704</td><td rowspan=1 colspan=1>32.131</td><td rowspan=1 colspan=1>32.040</td><td rowspan=1 colspan=1>32.554</td></tr><tr><td rowspan=1 colspan=1>room</td><td rowspan=1 colspan=1>31.446</td><td rowspan=1 colspan=1>31.634</td><td rowspan=1 colspan=1>31.588</td><td rowspan=1 colspan=1>31.528</td><td rowspan=1 colspan=1>32.199</td><td rowspan=1 colspan=1>32.188</td><td rowspan=1 colspan=1>31.330</td><td rowspan=1 colspan=1>32.040</td><td rowspan=1 colspan=1>32.652</td></tr><tr><td rowspan=1 colspan=1>playroom</td><td rowspan=1 colspan=1>30.019</td><td rowspan=1 colspan=1>30.051</td><td rowspan=1 colspan=1>29.876</td><td rowspan=1 colspan=1>30.447</td><td rowspan=1 colspan=1>30.186</td><td rowspan=1 colspan=1>30.449</td><td rowspan=1 colspan=1>30.099</td><td rowspan=1 colspan=1>30.200</td><td rowspan=2 colspan=1>30.62729.763</td></tr><tr><td rowspan=1 colspan=1>drjohnson</td><td rowspan=1 colspan=1>29.119</td><td rowspan=1 colspan=1>28.933</td><td rowspan=1 colspan=1>28.084</td><td rowspan=1 colspan=1>29.404</td><td rowspan=1 colspan=1>29.670</td><td rowspan=1 colspan=1>29.112</td><td rowspan=1 colspan=1>29.385</td><td rowspan=1 colspan=1>29.551</td></tr><tr><td rowspan=2 colspan=1>traintruck</td><td rowspan=2 colspan=1>21.95825.414</td><td rowspan=2 colspan=1>22.12625.535</td><td rowspan=2 colspan=1>22.14525.543</td><td rowspan=1 colspan=1>21.320</td><td rowspan=2 colspan=1>22.78026.062</td><td rowspan=2 colspan=1>22.46326.338</td><td rowspan=2 colspan=1>21.67825.169</td><td rowspan=2 colspan=1>22.18025.572</td><td rowspan=2 colspan=1>22.64126.542</td></tr><tr><td rowspan=1 colspan=1>25.409</td></tr></table>

Table 4: The PSNR scores for all works in each scene.

<table><tr><td rowspan=1 colspan=3>SceneâMethods</td><td rowspan=1 colspan=1>3DGS</td><td rowspan=1 colspan=1>AbsGS</td><td rowspan=1 colspan=1>PixelGS</td><td rowspan=1 colspan=1>MiniGS</td><td rowspan=1 colspan=1>TamGS</td><td rowspan=1 colspan=1>MCMC</td><td rowspan=1 colspan=1>SteepGS</td><td rowspan=1 colspan=1>PercepGS</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=3>bicycle</td><td rowspan=1 colspan=1>0.20921</td><td rowspan=1 colspan=1>0.18275</td><td rowspan=1 colspan=1>0.17956</td><td rowspan=1 colspan=1>0.15755</td><td rowspan=1 colspan=1>0.19210</td><td rowspan=1 colspan=1>0.16815</td><td rowspan=1 colspan=1>0.26239</td><td rowspan=1 colspan=1>0.17383</td><td rowspan=2 colspan=1>0.165100.28529</td></tr><tr><td rowspan=1 colspan=3>flowers</td><td rowspan=1 colspan=1>0.33536</td><td rowspan=1 colspan=1>0.28512</td><td rowspan=1 colspan=1>0.26170</td><td rowspan=1 colspan=1>0.25465</td><td rowspan=1 colspan=1>0.33052</td><td rowspan=1 colspan=1>0.28063</td><td rowspan=1 colspan=1>0.39758</td><td rowspan=1 colspan=1>0.26712</td></tr><tr><td rowspan=1 colspan=3>garden</td><td rowspan=1 colspan=1>0.10651</td><td rowspan=1 colspan=1>0.10663</td><td rowspan=1 colspan=1>0.09873</td><td rowspan=1 colspan=1>0.09011</td><td rowspan=1 colspan=1>0.09846</td><td rowspan=1 colspan=1>0.09575</td><td rowspan=1 colspan=1>0.12678</td><td rowspan=1 colspan=1>0.10273</td><td rowspan=1 colspan=1>0.09222</td></tr><tr><td rowspan=1 colspan=3>stump</td><td rowspan=1 colspan=1>0.21659</td><td rowspan=1 colspan=1>0.20442</td><td rowspan=1 colspan=1>0.18780</td><td rowspan=1 colspan=1>0.16817</td><td rowspan=1 colspan=1>0.20476</td><td rowspan=1 colspan=1>0.17079</td><td rowspan=1 colspan=1>0.27371</td><td rowspan=1 colspan=1>0.18501</td><td rowspan=1 colspan=1>0.17718</td></tr><tr><td rowspan=1 colspan=3>treehill</td><td rowspan=1 colspan=1>0.32461</td><td rowspan=1 colspan=1>0.29297</td><td rowspan=1 colspan=1>0.27505</td><td rowspan=1 colspan=1>0.26107</td><td rowspan=1 colspan=1>0.31206</td><td rowspan=1 colspan=1>0.27054</td><td rowspan=1 colspan=1>0.37381</td><td rowspan=1 colspan=1>0.28248</td><td rowspan=1 colspan=1>0.28717</td></tr><tr><td rowspan=1 colspan=3>bonsai</td><td rowspan=1 colspan=1>0.20380</td><td rowspan=1 colspan=1>0.19269</td><td rowspan=1 colspan=1>0.19116</td><td rowspan=1 colspan=1>0.17372</td><td rowspan=1 colspan=1>0.19937</td><td rowspan=1 colspan=1>0.18995</td><td rowspan=1 colspan=1>0.21073</td><td rowspan=1 colspan=1>0.18098</td><td rowspan=1 colspan=1>0.18503</td></tr><tr><td rowspan=2 colspan=1>counterkitchen</td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.19967</td><td rowspan=1 colspan=1>0.19324</td><td rowspan=1 colspan=1>0.18275</td><td rowspan=1 colspan=1>0.17333</td><td rowspan=1 colspan=1>0.19408</td><td rowspan=1 colspan=1>0.18346</td><td rowspan=1 colspan=1>0.21453</td><td rowspan=2 colspan=1>0.176340.11704</td><td rowspan=2 colspan=1>0.178650.11389</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.12570</td><td rowspan=1 colspan=1>0.12130</td><td rowspan=1 colspan=1>0.11883</td><td rowspan=1 colspan=1>0.11396</td><td rowspan=1 colspan=1>0.12117</td><td rowspan=1 colspan=1>0.12040</td><td rowspan=1 colspan=1>0.13437</td></tr><tr><td rowspan=1 colspan=3>room</td><td rowspan=1 colspan=1>0.21834</td><td rowspan=1 colspan=1>0.20404</td><td rowspan=1 colspan=1>0.20970</td><td rowspan=1 colspan=1>0.18750</td><td rowspan=1 colspan=1>0.20872</td><td rowspan=1 colspan=1>0.19769</td><td rowspan=1 colspan=1>0.22849</td><td rowspan=1 colspan=1>0.19437</td><td rowspan=1 colspan=1>0.19277</td></tr><tr><td rowspan=2 colspan=3>playroomdrjohnson</td><td rowspan=1 colspan=1>0.24304</td><td rowspan=1 colspan=1>0.24314</td><td rowspan=1 colspan=1>0.24037</td><td rowspan=1 colspan=1>0.20347</td><td rowspan=1 colspan=1>0.23718</td><td rowspan=1 colspan=1>0.23554</td><td rowspan=1 colspan=1>0.25294</td><td rowspan=2 colspan=1>0.231220.23090</td><td rowspan=2 colspan=1>0.225600.22588</td></tr><tr><td rowspan=1 colspan=1>0.24406</td><td rowspan=1 colspan=1>0.24378</td><td rowspan=1 colspan=1>0.25469</td><td rowspan=1 colspan=1>0.21834</td><td rowspan=1 colspan=1>0.23513</td><td rowspan=1 colspan=1>0.23779</td><td rowspan=1 colspan=1>0.24825</td></tr><tr><td rowspan=1 colspan=3>traintruck</td><td rowspan=1 colspan=1>0.207410.14644</td><td rowspan=1 colspan=1>0.189480.13846</td><td rowspan=1 colspan=1>0.177830.12024</td><td rowspan=1 colspan=1>0.179080.10031</td><td rowspan=1 colspan=1>0.200800.12577</td><td rowspan=1 colspan=1>0.185020.11215</td><td rowspan=1 colspan=1>0.227660.15890</td><td rowspan=1 colspan=1>0.183060.11761</td><td rowspan=1 colspan=1>0.169470.12091</td></tr></table>

Table 5: The LPIPS scores for all works in each scene.

## Implementation Details

We built our code upon the TamingGS repository. In addition to the proposed method, we made the following modifications:

â¢ Regarding hyperparameters, we adjusted position lr init to 0.00004 and position lr final to 0.000002. The gradient threshold used in EAS is set to 0.0003 and the opacity reset threshold is set 0.05.

â¢ We removed the pruning of large Gaussians, as our method can correctly identify these Gaussians and optimize their covered regions through splitting.

## More Qualitative Comparisons

<!-- image-->  
Figure 9: Qualitative comparison results among scenes bicycle, flowers, stump, treehill, bonsai.

<!-- image-->  
Figure 10: Qualitative comparison results among scenes counter, kitchen, room, playroom, truck.

<!-- image-->  
Figure 11: Schematic illustration of the split operation when simplified to a 2D ellipse.

We can first simplify the shape change of a 3D Gaussian after splitting into a 2D ellipse form (since the scaling ratios along the two shorter axes are identical, excluding the longest axis). Through intuitive geometric comparison(see Figure 11), it is straightforward to prove that when the condition in the formula(red lines):

$$
R _ { s } = R _ { 0 } \cdot { \sqrt { 1 - { \frac { d ^ { 2 } } { L _ { 0 } ^ { 2 } } } } } ,\tag{11}
$$

the absolute difference between the overlapping and non-overlapping areas of the child and parent Gaussians is minimized.

In this case, if we increase the length of $R _ { s }$ (green lines), the increment in the overlapping area is smaller than the increment in the non-overlapping area. The regions marked with green X in the figure represent the additional non-overlapping area compared to the overlapping area. Conversely, if we decrease the length of $R _ { s }$ (yellow lines), the reduction in the overlapping area exceeds the reduction in the non-overlapping area. The regions marked with yellow âXâ indicate the extra area lost in the overlap relative to the non-overlap. The actual introduced area difference is twice the size of these marked regions.

In practice, the geometric error introduced by splitting also depends on factors such as color, opacity, and other Gaussians covering the same region. Therefore, while the geometric area-optimal solution may not be absolutely optimal, it is very close to the global optimum. Figure 12 provides a comparison of results on the âstumpâ scene using different values of $R _ { s }$

<!-- image-->  
Figure 12: The trend of LPIPS with respect to Rs in the stump scene. The red dot indicates the value we chose in practice.

## Quantitative Comparison Under the Same Budget

TamingGS and 3DGS-MCMC are the only two methods among the compared approaches that allow manually setting a budget for the number of Gaussians. Here, we also provide quantitative results for these two methods under the same Gaussian budget as our method.

<table><tr><td>Dataset</td><td colspan="4">Average</td></tr><tr><td>Method- Metric</td><td>SSIMâ</td><td>PSN Râ</td><td>LPIPS</td><td>Num</td></tr><tr><td rowspan="2">TamingGS(SIGGRAPHAsia24) 3DGS-MCMC(NeurIPS24)</td><td>0.83777</td><td>27.673</td><td>0.21659</td><td>1615312</td></tr><tr><td>0.84969</td><td>27.706</td><td>0.20075</td><td>1615385</td></tr><tr><td>Ours</td><td>0.85367</td><td>27.948</td><td>0.18609</td><td>1615385</td></tr></table>

Table 6: Quantitative comparison across all scenes.

## Ablation Study of MU

Figure 13 shows the impact of enabling MU at different times. According to the figure, enabling MU from the beginning significantly degrades rendering quality. The negative impact of MU on rendering quality gradually diminishes as the densification process progresses. Enabling MU at the end of the densification process can improve rendering quality. Our two-stage MU approach performs slightly better than the single-stage MU.

<!-- image-->  
Figure 13: The impact of different MU starting iterations on LPIPS, test scene is bicycle. The update interval for parameters is uniformly set to 5. The red dot indicates the value we chose in practice.

## Ablation Study of Opacity Reduction Rate

Figure 14 shows the impact of different Opacity Reduction Rates on rendering quality. As can be seen from the figure, reducing opacity after splitting improves rendering quality, regardless of the reduction magnitude. When the Rate varies between 0.4 and 0.8, the change in rendering quality is relatively small. We choose 0.6 as an empirically determined optimal value.

## Ablation Study of Split Distance

Figure 15 shows the impact of the distance (d) between child Gaussians and the original Gaussian center after splitting, on both rendering quality and FPS. As d gradually decreases from 0.5, the geometric discrepancy caused by splitting slightly reduces, but the overlapping area among Gaussians increases significantly. This overlap increases the pixel rendering queue length, leading to higher rendering costs. Moreover, we observe that as d decreases, the negative effect of Gaussian overlap on rendering quality gradually outweighs the positive effect from reduced geometric discrepancy. The value of 0.45 we choose is an empirically determined trade-off that balances FPS and rendering quality.

<!-- image-->  
Figure 14: The trend of LPIPS with respect to opacity reduction rate in the stump scene. The red dot indicates the value we chose in practice.

<!-- image-->  
Figure 15: Testing the impact of the distance d between the sub-Gaussian and the original Gaussian centers on quality and rendering frame rate, test scene is bicycle.

## Ablation Study of Growth Control

Figure 16 shows the growth curves of Gaussians with and without GC. Without GC, the number of Gaussians reaches its peak early in the densification process, which clearly increases rendering overhead. In contrast, with GC applied, the growth of Gaussians follows a smoother trend, reaching its peak precisely at the end of densification.

<!-- image-->  
Figure 16: Growth curves of Gaussians with and without GC, test scene is bicycle.