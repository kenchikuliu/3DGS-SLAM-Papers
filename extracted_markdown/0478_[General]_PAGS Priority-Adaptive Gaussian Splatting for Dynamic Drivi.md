# PAGS: PRIORITY-ADAPTIVE GAUSSIAN SPLATTING FOR DYNAMIC DRIVING SCENES

Ying A1,3, Wenzhang Sun2, Chang Zeng2, Chunfeng Wang2, Hao Li2, Jianxun Cui1,3

1 Harbin Institute of Technology, China

2 Li Auto, China

3Chongqing Research Institute of HIT, China

## ABSTRACT

Reconstructing dynamic 3D urban scenes is crucial for autonomous driving, yet current methods face a stark trade-off between fidelity and computational cost. This inefficiency stems from their semantically agnostic design, which allocates resources uniformly, treating static backgrounds and safety-critical objects with equal importance. To address this, we introduce Priority-Adaptive Gaussian Splatting (PAGS), a framework that injects task-aware semantic priorities directly into the 3D reconstruction and rendering pipeline. PAGS introduces two core contributions: (1) Semantically-Guided Pruning and Regularization strategy, which employs a hybrid importance metric to aggressively simplify non-critical scene elements while preserving fine-grained details on objects vital for navigation. (2) Priority-Driven Rendering pipeline, which employs a prioritybased depth pre-pass to aggressively cull occluded primitives and accelerate the final shading computations. Extensive experiments on the Waymo and KITTI datasets demonstrate that PAGS achieves exceptional reconstruction quality, particularly on safety-critical objects, while significantly reducing training time and boosting rendering speeds to over 350 FPS.

Index Termsâ Gaussian Splatting, Real-Time Rendering, Autonomous Driving

## 1. INTRODUCTION

The reconstruction of dynamic, large-scale urban environments is a cornerstone of modern autonomous driving systems, providing the foundation for critical applications like simulation testing, synthetic data generation, and the creation of digital twins [1, 2]. With the advent of 3D Gaussian Splatting [3], the field has seen a significant leap forward, enabling real-time, photorealistic synthesis of novel views. To contend with the complexity of bustling cityscapes, a dominant paradigm has emerged: decomposing the scene into a static background and multiple, independently modeled dynamic foregrounds [4, 5, 6, 7]. This strategy has been adeptly adopted by contemporary 3DGS-based methods such as StreetGS [8] and DrivingGaussian [9].

However, the pursuit of universal high fidelity reveals a fundamental conflict when applied to autonomous driving. While recent works such as Speedy-Splat [12], FlashGS [13], and Mini-Splatting [14] have made significant strides in acceleration, they operate under a paradigm of uniform optimization. These methods remain semantically agnostic, failing to distinguish between functionally critical and non-critical scene components. This semantic blindness leads to a profound misallocation of resources. Significant computational budget is expended on perfecting the texture of a distant, static building or the foliage of a roadside treeâelements that have negligible impact on driving decisions. Every computation cycle spent refining a non-critical background element is a cycle not spent capturing the high-frequency details of a safety-critical foreground object. The finite representational capacity, when stretched thin across an entire scene, inevitably leads to compromised fidelity where it matters most: on pedestrians, cyclists, and other vehicles [15]. Consequently, subtle but vital visual cues risk being smoothed over or lost in a blur of generalized detail.

<!-- image-->  
Fig. 1: Qualitative comparisons on the Waymo [10] dataset. Our method achieves a sharper and more detailed reconstruction with just over 1 hour of training. This quality surpasses both StreetGS [8] and EmerNeRF [11], which produce less detailed results despite requiring significantly longer training times of over 3h and 11h.

To bridge this critical gap, we introduce Priority-Adaptive Gaussian Splatting (PAGS), a framework that embeds task-oriented semantic importance throughout the reconstruction and rendering pipeline. PAGS materializes this concept through two synergistic contributions. First, Semantically-Guided Pruning and Regularization strategy employs a hybrid metricâbalancing a static, foundation model-derived semantic score with a dynamic, gradientbased contribution. This allows for aggressive simplification of non-critical backgrounds while preserving high-fidelity details on key objects like vehicles and pedestrians. Second, the Priority-Driven Rendering pipeline first uses high-importance primitives to generate a coarse depth map. The GPUâs hardware-accelerated Early-Z test then leverages this map to cull occluded fragments before they undergo expensive shading. This significantly boosts rendering speed without compromising the perceptual quality of critical scene elements. Extensive experiments on the Waymo and KITTI [16] datasets demonstrate the efficacy of our approach. PAGS achieves a superior balance between reconstruction fidelity and computational efficiency, outperforming existing methods. Specifically, our framework not only enhances the visual quality of safety-critical objects but also drastically reduces training time while achieving real-time rendering speeds exceeding 350 FPS.

<!-- image-->  
Fig. 2: Overview of our proposed framework. Our pipeline embeds semantic importance throughout the reconstruction and rendering process. (Left) We begin with an offline semantic scene decomposition, using foundation models to assign a semantic prior to each 3D Gaussian. (Center) During training, this semantic prior guides two synergistic optimization strategies: Adaptive Stochastic Dropout, which applies regularization based on semantic importance scores in each iteration, and Semantically-Guided Pruning, which uses a hybrid importance metric to strategically remove Gaussians at specific intervals. This yields a set of priority-optimized 3D Gaussians. (Right) At inference, a priority-driven rendering pipeline first generates a depth map using high-importance occluders, then leverages it to accelerate the final color pass through efficient hardware-based culling.

## 2. METHODS

## 2.1. Compositional gaussian representation

We decompose the scene into distinct static and dynamic components. The static background is represented by a set of 3D Gaussian fixed in the world coordinate, initialized from a combination of LiDAR scans and Structure-from-Motion (SfM) [17] point clouds to ensure comprehensive coverage. Each primitive is defined by its position $( \mu _ { b } )$ , covariance $( \Sigma _ { b } ) .$ , opacity (Î±b), and view-dependent appearance modeled by Spherical Harmonics (SH). For unbounded regions like the sky, we employ a high-resolution cubemap [6].

Concurrently, each moving object is modeled as an independent set of Gaussian primitives within its own local coordinate frame. These primitives share similar attributes to their static counterparts, such as opacity $( \alpha _ { o } )$ and local scale $( S _ { o } )$ . To capture motion, we associate each object with a series of optimizable, time-varying poses, each comprising a rotation $R _ { c }$ and translation $T _ { c } .$ . To realistically model the time-varying appearance of dynamic objects under changing illumination, we utilize a four-dimensional Spherical Harmonics model, where each SH coefficient is represented as a function of time through a set of Fourier coefficients [8].

## 2.2. Semantically-guided pruning and regularization

## 2.2.1. Offline semantic scene decomposition

The foundation of our priority-adaptive framework is a clear semantic understanding of the scene. As a one-time preprocessing step, we first employ the Segment Anything Model (SAM) [18] to generate class-agnostic instance masks for all images. To assign semantic labels to these masks, we integrate results from an off-theshelf semantic segmentation model [19]. We group these semantic labels into a principled binary partition: categories such as vehicles, pedestrians, and cyclists are designated as Critical, while all others (e.g., buildings, roads, vegetation) are classified as Non-Critical.

## 2.2.2. Semantically-guided pruning

To intelligently allocate model capacity, our framework employs a semantically-guided pruning strategy based on a Hybrid Importance Metric $( S _ { \mathrm { h y b r i d } } )$ . This metric balances a stable, top-down semantic prior with a dynamic, data-driven contribution score, ensuring that pruning decisions are aligned with task priorities.

The metric consists of two components. The first is a Semantic Importance Score $( S _ { \mathrm { s e m } } )$ , calculated once during initialization. For each gaussian primitive, we project it into all views and compute the average overlap with pre-defined semantic masks. This yields a score $\bar { S _ { \mathrm { s e m } } } \in [ \bar { 0 , 1 } ]$ that acts as a strong semantic prior throughout training [22]. The second component is a Dynamic Contribution Score $( S _ { \mathrm { g r a d } } )$ , which quantifies a primitiveâs instantaneous importance to the reconstruction. This score is derived by aggregating the squared gradients of the final pixel color with respect to the primitiveâs contribution, which are readily available during the backward pass. The two scores are combined via a hyperparameter Î±:

$$
S _ { \mathrm { h y b r i d } } = \alpha \cdot S _ { \mathrm { s e m } } + ( 1 - \alpha ) \cdot S _ { \mathrm { g r a d } }\tag{1}
$$

This formulation allows us to apply a more lenient pruning threshold to primitives with high semantic importance, preserving fine details on critical objects even when their immediate gradient contribution is low, while non-critical elements are pruned more aggressively.

Operationally, we use $S _ { \mathrm { h y b r i d } }$ to rank and prune primitives at specific training intervals. During the densification stage (at 10k, 15k, and 20k iterations), we prune 60% of the Gaussians with the lowest scores. Subsequently, during fine-tuning (starting at 25k iterations), we continue to prune 30% of primitives every 5k iterations. As demonstrated in our ablation study (Section 3.3.2), these specific rates are critical for achieving an optimal balance between training efficiency and final reconstruction quality.

## 2.2.3. Adaptive stochastic dropout

Reconstructing moving objects is challenging due to sparse views, which risks overfitting and degrading reconstruction quality. To mitigate this, we introduce Adaptive Stochastic Dropout, a regularization technique that respects the sceneâs semantic hierarchy. The dropout probability for each primitive is inversely modulated by its semantic importance, as defined by the formula:

$$
D _ { t , i } = ( 1 - \beta \cdot S _ { \mathrm { s e m } , i } ) \cdot \gamma \cdot \frac { t } { t _ { \mathrm { t o t a l } } }\tag{2}
$$

This approach applies gentler regularization to critical, sparselyobserved objects, preventing overfitting while preserving their fine details. This method works synergistically with our pruning strategy, as primitives that are frequently dropped become natural candidates for removal. To maintain color fidelity, the opacities of surviving primitives are amplified by a compensation factor:

$$
C ( i ) = \frac { 1 } { 1 - D _ { t , i } }\tag{3}
$$

Table 1: Quantitative results on the Waymo and KITTI datasets.
<table><tr><td>Dataset</td><td colspan="5">Waymo Open Dataset</td><td colspan="5">KITTI Dataset</td></tr><tr><td>Method</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>FPS â</td><td>Train Time â</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>FPS â</td><td>Train Time â</td></tr><tr><td>EmerNeRF[11]</td><td>28.11</td><td>0.786</td><td>0.377</td><td>0.23</td><td>11h48m</td><td>26.95</td><td>0.828</td><td>0.219</td><td>0.28</td><td>11h51m</td></tr><tr><td>PVG[20]</td><td>30.46</td><td>0.910</td><td>0.229</td><td>52</td><td>4h23m</td><td>31.82</td><td>0.937</td><td>0.070</td><td>59</td><td>4h37m</td></tr><tr><td>StreetGS[8]</td><td>32.21</td><td>0.907</td><td>0.073</td><td>136</td><td>3h39m</td><td>32.76</td><td>0.922</td><td>0.067</td><td>141</td><td>3h54m</td></tr><tr><td>DeSiRe-GS[21]</td><td>33.52</td><td>0.917</td><td>0.204</td><td>36</td><td>5h34m</td><td>33.75</td><td>0.934</td><td>0.044</td><td>41</td><td>5h47m</td></tr><tr><td>Ours</td><td>34.63</td><td>0.933</td><td>0.073</td><td>353</td><td>1h22m</td><td>34.58</td><td>0.947</td><td>0.032</td><td>365</td><td>1h31m</td></tr></table>

<!-- image-->  
Fig. 3: Qualitative results on the Waymo dataset. The bottom row shows the final, fully converged reconstruction results for all methods. Our approach produces significantly sharper details on both dynamic objects and the static background. The top row presents a time-equalized comparison, showing the output of each method after an identical, fixed training duration. This highlights the superior convergence speed of our method, which achieves a high-fidelity result while competitors still exhibit noticeable blur and artifacts.

Table 2: Inference efficiency and memory footprint comparison.
<table><tr><td>Method</td><td>Speed (FPS) â</td><td>Size (MB) â</td><td>VRAM (GB) â</td></tr><tr><td>EmerNeRF</td><td>0.23</td><td>1217</td><td>10.5</td></tr><tr><td>PVG</td><td>52</td><td>959</td><td>8.2</td></tr><tr><td>StreetGS</td><td>136</td><td>853</td><td>7.8</td></tr><tr><td>DeSiRe-GS</td><td>36</td><td>984</td><td>8.5</td></tr><tr><td>Ours</td><td>353</td><td>530</td><td>6.1</td></tr></table>

## 2.3. Priority-driven rendering

To translate semantic importance into rendering efficiency, we introduce a Priority-Driven Rendering pipeline that leverages hardware occlusion culling. The first pass, an Occluder Depth Pre-Pass, rapidly establishes a coarse depth map of the scene. This is achieved by rendering a filtered subset of primitives, termed the Occluder Set, which are selected based on high semantic scores $( S _ { \mathrm { s e m } > } 0 . 5 )$ and opacity. This pass utilizes a minimal shader that discards all color and view-dependent calculations, writing only depth values to the Z-buffer with negligible computational overhead.

The subsequent Color Pass renders the final, high-fidelity image for all primitives, capitalizing on the depth information from the pre-pass to achieve significant acceleration. Specifically, the GPUâs hardware-accelerated Early-Z test compares the depth of each incoming fragment against the pre-populated Z-buffer, instantly culling any occluded fragments before expensive shading computations. To ensure correct alpha blending for the semi-transparent Gaussian model, a per-tile, back-to-front sort is performed using a composite key derived from both semantic priority and depth. By eliminating the redundant workload of rendering occluded primitives, this pipeline focuses computation exclusively on visible surfaces, which is the primary driver for the substantial increase in rendering speed to over 350 FPS in our results.

## 3. EXPERIMENTS

## 3.1. Experimental setup

We conduct evaluations on the Waymo and KITTI benchmarks. For Waymo, we use three frontal cameras, while for KITTI, we use the stereo camera pair. We evaluate reconstruction fidelity and perceptual quality using PSNR, SSIM, and LPIPS [23], and assess efficiency via total training time and rendering speed (FPS). Our method is compared against advanced approaches including EmerNeRF [11], PVG [20], StreetGS [8], and DeSiRe-GS [21]. All experiments are performed on a NVIDIA RTX 4090 GPU. For our framework, key hyperparameters are set to Î±=0.4 for the hybrid importance metric, Î³=0.25 and Î²=0.5 for adaptive stochastic dropout.

## 3.2. Quantitative and qualitative analysis

Quantitatively, as shown in Table 1, our method achieves a PSNR of 34.63 and an SSIM of 0.933 on the Waymo dataset, surpassing all compared methods while significantly accelerating rendering speed to 353 FPS. Notably, our training time of 1h22m is significantly shorter than most competing approaches. Similar performance gains are observed on the KITTI dataset. In terms of inference efficiency and memory footprint, as detailed in Table 2, our framework demonstrates a significant advantage crucial for practical deployment. PAGS achieves a real-time rendering speed of 353 FPS while maintaining a compact model size of just 530 MB and a low VRAM footprint of 6.1 GB, significantly outperforming all competing methods. This superior performance is a direct outcome of our core designs; the Semantically-Guided Pruning strategy is instrumental in creating a highly compact model, while the Priority-Driven Rendering pipeline is the primary driver behind the substantial leap in rendering speed.

<!-- image-->  
Fig. 4: Qualitative ablation of key components on the Waymo dataset. Our Semantically-Guided Pruning and Regularization (SPR) strategy yields a markedly sharper reconstruction of the vehicle than the baseline combining Staged Pruning (SP) and Stochastic Dropout (SD).

Table 3: Quantitative ablation of key components.
<table><tr><td>SP</td><td>SD</td><td>SPR</td><td>PDR</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>FPS</td><td>Time</td></tr><tr><td>â</td><td></td><td></td><td></td><td>29.83</td><td>0.892</td><td>0.083</td><td>135</td><td>1h30m</td></tr><tr><td rowspan="4"></td><td>V</td><td></td><td></td><td>32.87</td><td>0.951</td><td>0.041</td><td>130</td><td>3h35m</td></tr><tr><td>â</td><td></td><td></td><td>32.35</td><td>0.953</td><td>0.039</td><td>134</td><td>1h36m</td></tr><tr><td></td><td>â</td><td></td><td>34.95</td><td>0.958 0.035</td><td></td><td>134</td><td>1h24m</td></tr><tr><td></td><td></td><td>â</td><td>32.02</td><td>0.942</td><td>0.031</td><td>324</td><td>3h20m</td></tr><tr><td></td><td></td><td></td><td>V</td><td>34.63</td><td>0.933</td><td>0.073</td><td>353</td><td>1h22m</td></tr></table>

Qualitatively, as shown in Figure 3, the visual comparison further underscores our methodâs superiority. In the final results, our approach produces reconstructions with significantly sharper and clearer details on both vehicles and background elements. After just one hour of training, our method yields a high-fidelity result that is already much clearer than the blurry and artifact-laden outputs from competing methods at the same training duration. This combined evidence demonstrates the effectiveness and efficiency of our proposed framework.

## 3.3. Ablation studies

## 3.3.1. Analysis of component contributions

We validates the distinct contributions of our Semantically-Guided Pruning and Regularization (SPR) and Priority-Driven Rendering (PDR) modules. We first benchmarked our SPR module against a non-semantic baseline that combines standard Staged Pruning (SP) and Stochastic Dropout (SD). As detailed in Table 3, the baseline achieves a PSNR of 32.35 in 1h36m of training. In contrast, our SPR module elevates the PSNR to 34.95 while reducing training time to just 1h24m. This quantitative superiority is further supported by our qualitative results in Figure 4, which shows that the SPR module yields a markedly sharper reconstruction of critical objects compared to the baseline. This demonstrates that embedding semantic importance breaks the conventional trade-off, yielding a model that is both significantly more accurate and faster to train.

Furthermore, our Priority-Driven Rendering pipeline is designed for inference acceleration. By applying PDR to the SPR-trained model, rendering speed is boosted by 2.6x from 134 FPS to 353 FPS. This real-time performance is enabled by a controlled tradeoff, which results in a final global PSNR of 34.63. Consequently, the complete PAGS framework provides a highly optimized solution, achieving a superior balance of reconstruction quality, training efficiency, and rendering speed.

<!-- image-->

<!-- image-->  
Fig. 5: Impact of pruning rates on PSNR and training time.

Table 4: Validation of the hybrid importance metric.
<table><tr><td>Methods</td><td>Î±</td><td>PSNR-C</td><td>PSNR-NC</td><td>PSNR</td><td>Gauss.(M)</td></tr><tr><td>Gradient</td><td>0.0</td><td>28.15</td><td>34.54</td><td>33.38</td><td>58.8</td></tr><tr><td>Semantic</td><td>1.0</td><td>30.98</td><td>27.82</td><td>29.15</td><td>48.9</td></tr><tr><td>Hybrid</td><td>0.4</td><td>35.97</td><td>33.20</td><td>34.63</td><td>52.4</td></tr></table>

## 3.3.2. Analysis of pruning rate sensitivity

For Semantically-Guided Pruning, We conducted a sensitivity analysis to determine the optimal configuration for our key hyperparameters. We investigated the impact of densification and fine-tuning pruning rates on quality and training time. As visualized in Figure 5, we identified rates of 0.6 for densification and 0.3 for fine-tuning as the optimal balance, achieving a competitive PSNR of 34.63 while reducing training time to 82 minutes.

## 3.3.3. Analysis of hybrid importance metric

To validate our core innovationâthe hybrid importance metric âwe compare it against two specialized variants: a Gradient-Only strategy and a Semantic-Only strategy. The results in Table 4 validate our approach. The Gradient-Only approach neglects the fidelity of smaller, critical elements, resulting in a low PSNR-Critical score. Conversely, the Semantic-Only variant preserves critical objects at the expense of background integrity, leading to the lowest overall PSNR. Our hybrid approach emerges as the optimal solution, achieving the highest overall PSNR (34.63) and the best PSNR-Critical (35.97) while producing a compact model (52.4M Gaussians).

## 4. CONCLUSION

We presented Priority-Adaptive Gaussian Splatting (PAGS), a framework designed to overcome the semantic-agnostic limitations of current 3DGS methods for dynamic driving scenes. By synergizing a semantically-guided pruning and and Regularization strategy with a priority-driven rendering pipeline, PAGS effectively concentrates computation on safety-critical elements. Our results on the Waymo and KITTI datasets confirm the benefits of this approach, achieving rendering speeds over 350 FPS and reduced training times without compromising reconstruction quality on key objects. PAGS demonstrates that task-aware optimization is a key enabler for deploying high-fidelity, real-time 3D Gaussian Splatting in practical autonomous driving applications.

## 5. REFERENCES

[1] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel Urtasun, âUnisim: A neural closed-loop sensor simulator,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1389â1399.

[2] Zirui Wu, Tianyu Liu, Liyi Luo, Zhide Zhong, Jianteng Chen, Hongmin Xiao, Chao Hou, Haozhe Lou, Yuantao Chen, Runyi Yang, et al., âMars: An instance-aware, modular and realistic simulator for autonomous driving,â in CAAI International Conference on Artificial Intelligence. Springer, 2023, pp. 3â15.

[3] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis, â3d gaussian splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â 1, 2023.

[4] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide, âNeural scene graphs for dynamic scenes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 2856â2865.

[5] Abhijit Kundu, Kyle Genova, Xiaoqi Yin, Alireza Fathi, Caroline Pantofaru, Leonidas J Guibas, Andrea Tagliasacchi, Frank Dellaert, and Thomas Funkhouser, âPanoptic neural fields: A semantic object-aware neural scene representation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12871â12881.

[6] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar, âBlock-nerf: Scalable large scene neural view synthesis,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 8248â8258.

[7] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20310â20320.

[8] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng, âStreet gaussians: Modeling dynamic urban scenes with gaussian splatting,â in European Conference on Computer Vision. Springer, 2024, pp. 156â173.

[9] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang, âDrivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 21634â 21643.

[10] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al., âScalability in perception for autonomous driving: Waymo open dataset,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 2446â2454.

[11] Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, et al., âEmernerf: Emergent spatial-temporal scene decomposition via self-supervision,â arXiv preprint arXiv:2311.02077, 2023.

[12] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein, âSpeedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21537â21546.

[13] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, Linning Xu, Zhilin Pei, Hengjie Li, et al., âFlashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26652â 26662.

[14] Guangchi Fang and Bing Wang, âMini-splatting: Representing scenes with a constrained number of gaussians,â in European Conference on Computer Vision. Springer, 2024, pp. 165â181.

[15] Huaize Liu, Wenzhang Sun, Qiyuan Zhang, Donglin Di, Biao Gong, Hao Li, Chen Wei, and Changqing Zou, âHi-vae: Efficient video autoencoding with global and detailed motion,â arXiv preprint arXiv:2506.07136, 2025.

[16] Andreas Geiger, Philip Lenz, and Raquel Urtasun, âAre we ready for autonomous driving? the kitti vision benchmark suite,â in 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012, pp. 3354â3361.

[17] Johannes L Schonberger and Jan-Michael Frahm, âStructurefrom-motion revisited,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4104â 4113.

[18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al., âSegment anything,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015â4026.

[19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PmLR, 2021, pp. 8748â8763.

[20] Yurui Chen, Chun Gu, Junzhe Jiang, Xiatian Zhu, and Li Zhang, âPeriodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering,â arXiv preprint arXiv:2311.18561, 2023.

[21] Chensheng Peng, Chengwei Zhang, Yixiao Wang, Chenfeng Xu, Yichen Xie, Wenzhao Zheng, Kurt Keutzer, Masayoshi Tomizuka, and Wei Zhan, âDesire-gs: 4d street gaussians for static-dynamic decomposition and surface reconstruction for urban driving scenes,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 6782â6791.

[22] Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Norman Â´ Muller, Matthias NieÃner, Angela Dai, and Peter Kontschieder, Â¨ âPanoptic lifting for 3d scene understanding with neural fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9043â9052.

[23] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.