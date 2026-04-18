# GaussianTrimmer: Online Trimming Boundaries for 3DGS Segmentation

Liwei Liao

Peking University Shenzhen Graduate School

levio@pku.edu.cn

Ronggang Wang Peking University Shenzhen Graduate School rgwang@pkusz.edu.cn

AbstractâWith the widespread application of 3D Gaussians in 3D scene representation, 3D scene segmentation methods based on 3D Gaussians have also gradually emerged. However, existing 3D Gaussian segmentation methods basically segment on the basis of Gaussian primitives. Due to the large variation range of the scale of 3D Gaussians, large-sized Gaussians that often span the foreground and background lead to jagged boundaries of segmented objects. To this end, we propose an online boundary trimming method, GaussianTrimmer, which is an efficient and plugand-play post-processing method capable of trimming coarse boundaries for existing 3D Gaussian segmentation methods. Our method consists of two core steps: 1. Generating uniformly and well-covered virtual cameras; 2. Trimming Gaussian at the primitive level based on 2D segmentation results on virtual cameras. Extensive quantitative and qualitative experiments demonstrate that our method can improve the segmentation quality of existing 3D Gaussian segmentation methods as a plug-and-play method.

Index Termsâ3DGS Segmentation, Boundary Trimming, 3D Gaussian Splatting

## I. INTRODUCTION

In recent years, 3D Gaussian splatting (3DGS) [1] has gradually occupied an important position in 3D scene representation due to its high-quality rendering and real-time rendering speed. With the widespread application of 3D Gaussians, 3D scene segmentation methods [2]â[14] based on 3D Gaussians have also gradually emerged. However, most existing 3DGS-based segmentation methods suffer from two key limitations: (1) they treat individual Gaussian primitives as the atomic unit for segmentation, and (2) boundary-straddling Gaussians always lie on object boundaries (as illustrated in Fig. 1 (a)). These straddling Gaussians, which overlap both foreground and background regions, hinder the attainment of sharp segmentation boundaries, often resulting in jagged boundaries. Such irregularities not only degrade quantitative metrics like mIoU and mAcc but also compromise visual quality. These shortcomings severely constrain the utility of 3DGS in downstream tasks, including 3D scene editing (e.g., object relocation or compositing) and robotic applications (e.g., grasping simulations). The jagged boundaries of the segmented objects are refered to as âboundary challengeâ.

To resolve the boundary challenge, we tackle it from the perspective of decomposing straddling primitives, by further splitting each straddling Gaussian into two smaller Gaussians that respectively belong to the object and the background, thereby achieving training-free trimming of 3D Gaussians. Inspired by SAGD [7], we adapt an online method for splitting straddling Gaussians, by decomposing the 2D covariance of the projected Gaussians using segmentation boundaries from the 2D view to guide the decomposition ratio of the 3D covariance, and separately computing the new positions corresponding to the decomposed Gaussians. Unlike previous methods, we treat boundary trimming as a form of postprocessing and refinement of the segmentation results (as shown in Fig. 1 (b)). To this end, we devise a boundary trimming strategy based on virtual camera planning, thereby enabling segmentation refinement with extremely low latency (approximately 1s). Our key contributions can be summarized as:

<!-- image-->  
(a) The boundary challenge in 3DGS Segmentation

<!-- image-->  
(b) Illustration of usage of GaussianTrimmer  
Fig. 1. (a) Straddling Gaussians make it difficult to achieve precise segmentation boundaries; (b) Our GaussianTrimmer effectively trims these boundaries for improved segmentation quality within only approximately 1 second latency.

â¢ We propose GaussianTrimmer, an online boundary trimming method for 3DGS segmentation, which effectively addresses the boundary challenge within approximately 1 second latency.

â¢ We design an effective Virtual Camera Planning (VCP) module for more precise 3D Gaussian decomposition through the 2D masks.

â¢ Extensive experiments demonstrate our GaussianTrimmer can work as a plug-and-play module to improve the quantitative and qualitative results.

## II. RELATED WORKS

## A. 3D Neural Scene Segmentation

Since 3DGS was established as both training-efficient and capable of real-time rendering, a range of exceptional 3DGSspecific segmentation methods have emerged [2]â[14]. Specifically, GaussianGrouping [2] links masks across views using a tracker, then applies joint learning to optimize reconstruction and identity encodings simultaneously. ClickGaussian [4], a post-processing approach, enhances pre-trained 3D Gaussians with two-level granularity features and introduces Global Feature-guided Learning to mitigate mask inconsistencies across views. Certain studies [7], [11] focus on refining boundaries in 3DGS segmentation. Meanwhile, Flashsplat [5] accelerates the optimization process by approximately 50 times, though it still requires 30 seconds for optimization, classifying it as an offline method.

## B. Boundary Refinement

Decomposing 3D primitives has been explored in various contexts. In 3D reconstruction, methods like SAGD [7] decompose Gaussians based on 2D segmentation boundaries to improve object delineation. Other works have focused on decomposing meshes or point clouds to enhance segmentation quality. Our approach builds upon these ideas by introducing an online boundary trimming method specifically designed for 3DGS segmentation, addressing the challenges posed by straddling Gaussians. COB-GS [11] also addresses boundary issues but focuses on a different aspect of Gaussian representation. In contrast, our method emphasizes efficient and effective trimming of boundaries as a post-processing step, enhancing existing segmentation results with minimal latency.

Unlike previous methods that require retraining or complex optimization, our GaussianTrimmer operates as a plug-andplay solution, making it easy to integrate into existing 3DGS segmentation pipelines. Moreover, our method focuses on realtime performance, ensuring that boundary refinement can be achieved with minimal computational overhead.

## III. METHOD

In this section, we introduce GaussianTrimmer, a method for online trimming boundaries in 3DGS segmentation. As illustrated in Fig. 2, GaussianTrimmer is a post-processing module designed to refine the segmentation results from existing 3DGS segmentation methods. The input to GaussianTrimmer is a jagged segmentation result, and the output result has smoother boundaries. This step can be formulized as follows:

$$
\begin{array} { r } { \hat { \Theta _ { \mathcal { O } } } = \mathsf { G a u T r i m n e r } ( \Theta _ { \mathcal { O } } ; \Theta , \vec { U } ) , } \end{array}\tag{1}
$$

where $\Theta \in \mathbb { R } ^ { N \times C }$ represents total Gaussians of the scene, $\Theta _ { \mathcal { O } } \in \mathbb { R } ^ { K \times C }$ is a subset of Î, denoting the jagged coarse segmentation result obtained from an existing 3DGS segmentation method, and $\vec { U }$ denotes the up vector used for virtual camera orientation. The output $\hat { \Theta _ { \mathcal { O } } } \in \mathbb { R } ^ { K \times C }$ is the refined segmentation result with improved boundary quality. Note that Î is optional and can be used for background augmentation during virtual view segmentation.

## A. Virtual Camera Planning (VCP)

Corresponding to Fig. 2 (A), the first step of our pipeline is Virtual Camera Planning (VCP). Different from previous boundary refinement methods that utilize 2D masks from training views as cues, we employ virtual views that offer better alignment and more comprehensive coverage (see Fig. 3). The VCP module generates a set of virtual cameras $\textbf { c } = \ \{ c _ { 0 } , c _ { 1 } , . . . , c _ { n - 1 } \}$ that uniformly cover the segmented object $\Theta _ { \mathcal { O } }$ . To achieve this, we adopt a spherical coordinate system as follows: First, we determine the centroid of the object, which serves as the center of the spherical coordinate system. The centroid L is computed using the positions of the Gaussians in the segmented object $\Theta _ { \mathcal { O } }$ as

$$
L = { \frac { 1 } { | p | } } \sum _ { \mathbf { x } \in p } \mathbf { x } ,\tag{2}
$$

where ${ \pmb p } = \{ { \bf x } _ { i } | { \bf x } _ { i } \in \Theta _ { \mathcal { O } } \}$ represents the set of positions of Gaussians in the segmented object, and $| p |$ is the total number of these Gaussians. Next, we determine suitable distance r from the centroid L to position the virtual cameras. Let $\{ l , w , h \}$ denote length, width and height of $\Theta _ { \mathcal { O } }$ . We set the distance r as

$$
r = \tt m a x ( \lambda _ { 1 } \cdot m a x ( \lambda , \it w ) , \lambda _ { 2 } \cdot \it h ) ,\tag{3}
$$

where $\lambda _ { 1 }$ and Î»2 are hyperparameters to control the distance and keep the whole object visible in virtual views. In our setting, we set $\lambda _ { 1 } = 2 0$ and $\lambda _ { 2 } = 6$

Then, we construct a spherical coordinate system Y with L as the center, r as the radius and $\vec { U }$ as the up axis. In Y, we can place the virtual cameras based on yaw and pitch angles $\{ \psi , \theta \}$ . The $\vec { U }$ can be obtained by manual calibration or normal vector estimation of the ground plane via RANSAC [15]. We fix the pitch angle Î¸ to ensure all virtual cameras lie on the same plane, and then uniformly sample the yaw angle Ï within a specific range. For each camera position $P _ { i } ,$ , it can be represented as:

$$
P _ { i } = ( \psi _ { i } , \theta , r ) .\tag{4}
$$

Then, we orient each virtual camera to face the target centroid $L ,$ generating the final camera poses. Finally, we set the intrinsic parameters for each virtual camera to ensure that its field of view covers the entire target. For intrinsic matrix generation, we set:

$$
\mathbf { { K } } = \left( \begin{array} { c c c } { \frac { w } { 2 \tan \left( \frac { F O V } { 2 } \right) } } & { 0 } & { \frac { w } { 2 } } \\ { 0 } & { \frac { h } { 2 \tan \left( \frac { F O V } { 2 } \right) } } & { \frac { h } { 2 } } \\ { 0 } & { 0 } & { 1 } \end{array} \right)\tag{5}
$$

where w and h are the width and height of the rendered image, respectively. $F O V$ is the field of view, which is set to $6 0 ^ { \circ }$ in our experiments.

<!-- image-->  
Fig. 2. GaussianTrimmer Pipeline : Guided by user interaction, Virtual Camera Generation (VCG) module generates smooth and object-center virtual views, followed by Rendering-Tracking-Pruning (RTP) loop on the generated virtual views to identify Gaussians belonging to the target, which are represented as a Gaussian-level Boolean 3D mask.

<!-- image-->  
Fig. 3. Our virtual cameras vs. real cameras. Our planned virtual cameras (top row) provide better object coverage and alignment compared to real cameras (bottom row), leading to more accurate trimming results.

## B. Virtual View Segmentation (VVS)

After planning the virtual cameras, we proceed to the Virtual View Segmentation (VVS) step, as depicted in Fig. 2 (B). In this step, we render the scene from the perspective of each virtual camera to generate a series of virtual views. Each virtual view captures a 2D projection of the 3D scene, allowing us to leverage 2D segmentation techniques for boundary refinement. The Segment Anything (SAM) series [16], [17], especially SAM2, is employed to perform segmentation on these virtual views. SAM2 is chosen for its robustness and accuracy in cross-view tracking. For cross-view tracking, we design two strategies to better suit the Gaussian boundary processing: 1) automatic mask generation; 2) background augmentation.

Automatic Mask Generation. To bootstrap the segmentation process in SAM2, we automatically generate an initial mask for the first virtual view. Subsequently, we leverage SAM2âs cross-view tracking capabilities to propagate the segmentation to the remaining virtual views. For the initial mask generation, we render the coarse segmentation output $\Theta _ { \mathcal { O } }$ . Since the background in the first virtual view $I _ { 0 }$ is blank, the bounding box of the target object can be directly extracted. This bounding box is then provided as a prompt to SAM2 to produce a refined initial mask using

$$
m _ { 0 } = { \tt S A M 2 } ( I _ { 0 } ; \mathrm { b o x } ( I _ { 0 } ) ) .\tag{6}
$$

Background Augmentation. Due to the blank background in the virtual views, SAM2 may mistakenly classify the jagged boundaries as part of the foreground object. To address this issue, we employ a background augmentation strategy. Specifically, we augment the coarse segmentation result $\Theta _ { \mathcal { O } }$ by incorporating additional background Gaussians to obtain $\Theta _ { \mathcal { O } } ^ { \prime }$ as:

$$
\Theta _ { \mathcal { O } } ^ { \prime } = \Theta _ { \mathcal { O } } \cup \Theta _ { n } ,\tag{7}
$$

where $\Theta _ { n }$ represents the set of neighboring Gaussians of $\Theta _ { \mathcal { O } }$ Then we render $\Theta _ { \mathcal { O } } ^ { \prime }$ to obtain virtual views for 2D masks.

## C. Boundary Gaussian Decomposition (BGD)

BGD module is the core module of GaussianTrimmer. As shown in Fig. 2 (C), the BGD module operates on the straddling Gaussians located at the boundaries. By decomposing overly long Gaussians into two shorter ones, it effectively addresses the boundary jaggedness issue. Thus, the goal of the BGD module is to: (1) detect these straddling Gaussians and (2) perform decomposition processing on them.

Boundary Gaussians Detection. To efficiently identify the straddling boundary Gaussians, we employ a reverse Î±- blending strategy. Specifically, we trace back from a single pixel in the 2D virtual view to the set of Gaussians that contribute to the color value of that pixel. Since the number of Gaussians contributing to each pixelâs rendering varies, we implement a top-k selection scheme, selecting the topk Gaussians based on their contribution to the pixelâs color rendering. By constructing a one-to-k pixel-to-Gaussian index mapping, we can efficiently index all straddling Gaussians for the contour points of the 2D mask in the virtual view. Thus, we can quickly identify all straddling Gaussians $\Theta _ { b }$ that may cause jagged boundaries, which is a small subset compared to the entire set of object Gaussians $\Theta _ { \mathcal { O } }$ â¢

<!-- image-->  
(a) Pixel-to-Gaussian Tracing

<!-- image-->  
(b) Illustration of boundary detection  
Fig. 4. (a) Illustration of pixel-to-Gaussian mapping; (b) Ilustration of boundary Gaussian detection.

Gaussian Decomposition. Once the straddling Gaussians $\Theta _ { b }$ are identified, we proceed to decompose them into two shorter Gaussians. For each straddling Gaussian, we first determine its principal axis based on its covariance matrix. We then split the Gaussian along this axis at its centroid, creating two new Gaussians with adjusted means and covariances. The weights and colors of the new Gaussians are inherited from the original Gaussian. This decomposition effectively reduces the length of the straddling Gaussians, leading to smoother boundaries in the segmentation result. For a single Gaussian decomposition whose long axis and position are denoted as {l, x}, we follow the SAGD [7] to figure out one of the new Gaussians $( \{ l ^ { \prime } , { \bf x } ^ { \prime } \} )$ as

$$
{ \mathit { l } } ^ { \prime } = \lambda { \mathit { l } } ,\tag{8}
$$

$$
{ \bf x } ^ { \prime } = { \bf x } + \frac { 1 } { 2 } \left( l - \lambda l \right) { \bf e } ,\tag{9}
$$

where Î» is the decomposition ratio, and e is the unit vector along the principal axis. Finally, we prune the Gaussians that are not consistently identified as part of the target object across multiple views, resulting in a refined segmentation with smoother boundaries.

## IV. EXPERIMENTS

## A. Experimental Setup

To demonstrate that our GaussianTrimmer is an effective post-processing method for 3DGS segmentation results, we primarily evaluate the improvement in segmentation quality brought by GaussianTrimmer to existing 3DGS segmentation methods in our quantitative experiments. We conduct comparative experiments with existing 3DGS segmentation methods on multiple benchmark datasets and perform both quantitative and qualitative analyses. For quantification, we utilize the NVOS dataset [18], which is derived from the

LLFF [19] dataset and provides ground truth masks with precise object edges. For qualitative evaluation, we employ scenes from various datasets, including IN2N [20], Mipnerf-360 [21], PKU-DyMVHuman [22], and LERF-Mask [23].

Implementation Details. We first obtain an initial 3DGS segmentation result using existing 3DGS segmentation methods, then apply GaussianTrimmer for boundary trimming on the initial segmentation result, and finally evaluate the improvement in segmentation quality before and after trimming. In all quantitative experiments, we set the number of virtual cameras to 5 for each target object.

## B. Quantitative Results

We evaluate the effectiveness of our GaussianTrimmer on the NVOS dataset [18] by applying it to enhance the segmentation results of several state-of-the-art 3DGS segmentation methods, including SA3D [24], OmniSeg3D [25], SAGA [3], FlashSplat [5], SAGD [7], and COB-GS [11]. The quantitative results are summarized in Table I. Our method consistently improves the mIoU and mAcc metrics across all evaluated methods, demonstrating its effectiveness as a plug-and-play post-processing technique for enhancing 3DGS segmentation quality.

TABLE I  
QUANTITATIVE SEGMENTATION RESULTS ON NVOS DATASET.
<table><tr><td>Method</td><td>mIoU (%)</td><td>mAcc (%)</td></tr><tr><td>SA3D [24] + GaussianTrimmer</td><td>90.3 91.9 (+1.6)</td><td>98.2 98.5 (+0.3)</td></tr><tr><td>OmniSeg3D [25]</td><td>91.7</td><td>98.4</td></tr><tr><td>+ GaussianTrimmer</td><td>92.4 (+0.7)</td><td>98.7 (+0.3)</td></tr><tr><td>SAGA [3]</td><td>90.9</td><td>98.3</td></tr><tr><td>+ GaussianTrimmer</td><td>92.1 (+1.2)</td><td>98.5 (+0.2)</td></tr><tr><td>FlashSplat [5] + GaussianTrimmer</td><td>91.8</td><td>98.6</td></tr><tr><td></td><td>92.2 (+0.4)</td><td>98.7 (+0.1)</td></tr><tr><td>SAGD [7]</td><td>90.4</td><td>98.2</td></tr><tr><td>+ GaussianTrimmer</td><td>92.0 (+1.6)</td><td>98.5 (+0.3)</td></tr><tr><td>COB-GS [11]</td><td>92.1</td><td></td></tr><tr><td>+ GaussianTrimmer</td><td>92.5 (+0.4)</td><td>98.6 98.7 (+0.1)</td></tr></table>

TABLE II

QUANTITATIVE VISUAL RESULTS ON NVOS DATASET.
<table><tr><td rowspan="2">Method</td><td colspan="3">CLIP-IQA [26] (%) â</td></tr><tr><td>Boundary</td><td>Clear I Unclear Smooth / Noisy Boundary</td><td>Complete / Mutilated Object</td></tr><tr><td>SAGD [7]</td><td>0.621</td><td>0.631</td><td>0.788</td></tr><tr><td>+ GauTrimmer</td><td>0.688</td><td>0.738</td><td>0.835</td></tr><tr><td>COB-GS [11]</td><td>0.682</td><td>0.731</td><td>0.859</td></tr><tr><td>+ GauTrimmer</td><td>0.695</td><td>0.742</td><td>0.867</td></tr></table>

Moveover, we follow Zhang et al. [11] to assess the visual quality of segmentation results before and after applying GaussianTrimmer using the CLIP-IQA [26] metric, which evaluates the clarity of object boundaries and the completeness of segmented objects. As shown in Table II, GaussianTrimmer significantly enhances the visual quality of segmentation results, yielding higher CLIP-IQA scores across all evaluated aspects.

<!-- image-->  
Fig. 5. Qualitative Results. This figure showcases the direct improvement effects of GaussianTrimmer on the segmentation results of existing methods. From a visual perspective, GaussianTrimmer effectively refines jagged edges and enhances the clarity of segmentation boundaries.

## C. Qualitative Results

We further demonstrate the effectiveness of GaussianTrimmer through qualitative comparisons on various real-world scenes from multiple datasets, including IN2N [20], Mipnerf-360 [21], PKU-DyMVHuman [22]. As illustrated in Figure 5, GaussianTrimmer effectively refines the segmentation boundaries and recovers missing object parts, leading to more accurate and visually appealing segmentation results. These qualitative results further validate the capability of GaussianTrimmer to enhance 3DGS segmentation quality across diverse scenarios.

## D. Ablation Study

We conduct ablation studies on three aspects: virtual cameras, background augmentation, and the number of virtual viewpoints. Firstly, we replace the virtual viewpoints with real training viewpoints to verify the effectiveness of the VCP module. Then we disable the background augmentation module to validate its contribution to segmentation quality improvement. Finally, we evaluate the performance and latency of GaussianTrimmer under different numbers of virtual viewpoints to balance segmentation quality and computational overhead (see Table IV).

TABLE III  
ABLATION STUDY ON NVOS DATASET TAKING SAGA AS BASELINE.
<table><tr><td>Method</td><td>mIoU (%)</td><td>mAcc (%)</td></tr><tr><td>SAGA [3] (baseline)</td><td>90.9</td><td>98.3</td></tr><tr><td>No VCP</td><td>91.3</td><td>98.3</td></tr><tr><td>No Bg. Augmentation</td><td>91.5</td><td>98.4</td></tr><tr><td>No VCP and BA</td><td>91.3</td><td>98.3</td></tr><tr><td>Full GaussianTrimmer</td><td>92.1 (+1.2)</td><td>98.5 (+0.2)</td></tr></table>

TABLE IV

ABLATION ON VARYING VIRTUAL CAMERA NUMBER.
<table><tr><td>n =</td><td>1</td><td>2</td><td>3</td><td>5</td><td>7</td><td>10</td></tr><tr><td>Latency(ms)</td><td>256</td><td>409</td><td>521</td><td>766</td><td>1037</td><td>1566</td></tr><tr><td>mIOU(%) Gain</td><td>0.8</td><td>0.9</td><td>1.0</td><td>1.2</td><td>1.2</td><td>1.2</td></tr></table>

## V. CONCLUSION

In this paper, we have presented GaussianTrimmer, an efficient and plug-and-play online boundary trimming method designed to enhance the segmentation quality of existing 3D Gaussian Splatting (3DGS) segmentation methods. By generating uniformly distributed virtual cameras and leveraging 2D segmentation results to trim straddling Gaussians, our approach effectively addresses the boundary challenge inherent in 3DGS segmentation. Extensive experiments on benchmark datasets demonstrate that GaussianTrimmer significantly improves segmentation accuracy, particularly in refining object boundaries. Our method serves as a valuable post-processing step that can be seamlessly integrated into various 3DGS segmentation pipelines, paving the way for more precise and reliable 3D scene understanding.

## REFERENCES

[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Â¨ Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM TOG, vol. 42, no. 4, pp. 1â14, 2023.

[2] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke, âGaussian grouping: Segment and edit anything in 3d scenes,â in ECCV. Springer, 2024, pp. 162â179.

[3] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian, âSegment any 3d gaussians,â arXiv preprint arXiv:2312.00860, 2023.

[4] Seokhun Choi, Hyeonseop Song, Jaechul Kim, Taehyeong Kim, and Hoseok Do, âClick-gaussian: Interactive segmentation to any 3d gaussians,â in ECCV. Springer, 2024, pp. 289â305.

[5] Qiuhong Shen, Xingyi Yang, and Xinchao Wang, âFlashsplat: 2d to 3d gaussian splatting segmentation solved optimally,â in ECCV. Springer, 2024, pp. 456â472.

[6] Runsong Zhu, Shi Qiu, Zhengzhe Liu, Ka-Hei Hui, Qianyi Wu, Pheng-Ann Heng, and Chi-Wing Fu, âRethinking end-to-end 2d to 3d scene segmentation in gaussian splatting,â arXiv preprint arXiv:2503.14029, 2025.

[7] Xu Hu, Yuxi Wang, Lue Fan, Junsong Fan, Junran Peng, Zhen Lei, Qing Li, and Zhaoxiang Zhang, âSagd: Boundary-enhanced segment anything in 3d gaussian via gaussian decomposition,â arXiv preprint arXiv:2401.17857, 2024.

[8] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister, âLangsplat: 3d language gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20051â20060.

[9] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, et al., âOpengaussian: Towards point-level 3d gaussian-based open vocabulary understanding,â arXiv preprint arXiv:2406.02058, 2024.

[10] Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao Zhang, Ronggang Wang, and Jian Zhang, âInstancegaussian: Appearancesemantic joint gaussian representation for 3d instance-level perception,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 14078â14088.

[11] Jiaxin Zhang, Junjun Jiang, Youyu Chen, Kui Jiang, and Xianming Liu, âCob-gs: Clear object boundaries in 3dgs segmentation based on boundary-adaptive gaussian splitting,â arXiv preprint arXiv:2503.19443, 2025.

[12] Liwei Liao, Xufeng Li, Xiaoyun Zheng, Boning Liu, Feng Gao, and Ronggang Wang, âZero-shot visual grounding in 3d gaussians via view retrieval,â arXiv preprint arXiv:2509.15871, 2025.

[13] Yian Zhao, Wanshi Xu, Ruochong Zheng, Pengchong Qiao, Chang Liu, and Jie Chen, âisegman: Interactive segment-and-manipulate 3d gaussians,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 661â670.

[14] Guibiao Liao, Jiankun Li, Zhenyu Bao, Xiaoqing Ye, Jingdong Wang, Qing Li, and Kanglin Liu, âClip-gs: Clip-informed gaussian splatting for real-time and view-consistent 3d semantic understanding,â arXiv preprint arXiv:2404.14249, 2024.

[15] Martin A Fischler and Robert C Bolles, âRandom sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,â Communications of the ACM, vol. 24, no. 6, pp. 381â395, 1981.

[16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al., âSegment anything,â in ICCV, 2023, pp. 4015â4026.

[17] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Â¨ Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and ChristophÂ´ Feichtenhofer, âSam 2: Segment anything in images and videos,â arXiv preprint arXiv:2408.00714, 2024.

[18] Zhongzheng Ren, Aseem Agarwala, Bryan Russell, Alexander G Schwing, and Oliver Wang, âNeural volumetric object selection,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 6133â6142.

[19] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar, âLocal light field fusion: Practical view synthesis with prescriptive sampling guidelines,â ACM Transactions on Graphics (ToG), vol. 38, no. 4, pp. 1â14, 2019.

[20] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa, âInstruct-nerf2nerf: Editing 3d scenes with instructions,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 19740â19750.

[21] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in CVPR, 2022, pp. 5470â5479.

[22] Xiaoyun Zheng, Liwei Liao, Xufeng Li, Jianbo Jiao, Rongjie Wang, Feng Gao, Shiqi Wang, and Ronggang Wang, âPku-dymvhumans: A multi-view video benchmark for high-fidelity dynamic human modeling,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22530â22540.

[23] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik, âLerf: Language embedded radiance fields,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19729â19739.

[24] Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, et al., âSegment anything in 3d with nerfs,â Advances in Neural Information Processing Systems, vol. 36, pp. 25971â25990, 2023.

[25] Haiyang Ying, Yixuan Yin, Jinzhi Zhang, Fan Wang, Tao Yu, Ruqi Huang, and Lu Fang, âOmniseg3d: Omniversal 3d segmentation via hierarchical contrastive learning,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20612â20622.

[26] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy, âExploring CLIP for Assessing the Look and Feel of Images,â in AAAI, 2023.