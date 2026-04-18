# DRIVINGSCENE: A MULTI-TASK ONLINE FEED-FORWARD 3D GAUSSIAN SPLATTING METHOD FOR DYNAMIC DRIVING SCENES

Qirui Hou1,2, Wenzhang Sun3, Chang Zeng3, Chunfeng Wang3, Hao Li3, Jianxun Cui1,2\*

1 Harbin Institute of Technology, China 2 Chongqing Research Institute of HIT, China 3 Li Auto, China

## ABSTRACT

Real-time, high-fidelity reconstruction of dynamic driving scenes is challenged by complex dynamics and sparse views, with prior methods struggling to balance quality and efficiency. We propose DrivingScene, an online, feed-forward framework that reconstructs 4D dynamic scenes from only two consecutive surround-view images. Our key innovation is a lightweight residual flow network that predicts the non-rigid motion of dynamic objects per camera on top of a learned static scene prior, explicitly modeling dynamics via scene flow. We also introduce a coarse-to-fine training paradigm that circumvents the instabilities common to end-to-end approaches. Experiments on nuScenes dataset show our image-only method simultaneously generates high-quality depth, scene flow, and 3D Gaussian point clouds online, significantly outperforming state-of-the-art methods in both dynamic reconstruction and novel view synthesis.

Index Termsâ Autonomous Driving, Novel view Synthesis, Multi task Learning

## 1. INTRODUCTION

Accurate, real-time 4D (3D space + time) environmental perception and reconstruction form the bedrock of safety and reliability for autonomous driving systems. Modern autonomous vehicles are typically equipped with multiple cameras for 360-degree surroundview perception. Compared to fusion-based approaches that rely on multi-modal sensors like LiDAR or RaDAR[1, 2, 3], vision-only methods[4, 5] offer a more cost-effective and computationally efficient pathway for complex online perception tasks. However, reconstructing a large-scale, geometrically accurate, and photorealistic dynamic scene in real-time, solely from sparse and dynamic surroundview images, remains a significant and unresolved challenge.

The pursuit of higher reconstruction fidelity has seen tremendous success with neural rendering techniques like NeRF [6] and 3DGS [7]. However, the majority of these methods, whether for static scenes like StreetGaussian [8], DrivingGaussian [9] or dynamic scenes like EmerNeRF [10], are bound by a per-scene optimization paradigm. This reliance on time-consuming offline training is incompatible with the real-time requirements of autonomous driving downstream tasks, necessitating a paradigm shift towards âfeed-forwardâ reconstruction[11, 12, 13, 14]. This online approach has matured for static scenes, with methods like pixelSplat [15] and MVSplat [16] demonstrating its viability, and culminating in works like DrivingForward [17] which successfully handle sparse driving contexts. Yet, their foundational static world assumption inevitably leads to severe artifacts when confronted with moving vehicles. To address this, methods like Driv3R [18] have attempted to model dynamic scenes end-to-end. However, this monolithic design not only imposes a heavy computational burden but, more importantly, fails to explicitly decouple the inherently distinct static and dynamic components of a scene, leaving room for improvement in reconstruction detail and fidelity.

<!-- image-->  
Fig. 1. Example predictions by our method on nuScenes [2]. Top to bottom: input image (one of the sequence), depth map and optical flow. Our model is fully self-supervised and can handle dynamic objects and occlusions explicitly.

To address these challenges, we introduce DrivingScene, an efficient online, feed-forward framework designed specifically for online dynamic driving scene reconstruction. The key to our approach is a two-stage, static-to-dynamic learning strategy that decouples the complex 4D reconstruction problem into two more tractable subtasks: robust static scene modeling and subsequent dynamic refinement. Specifically, in the first stage, we focus on training a network to learn a powerful static scene prior from large-scale data. This initial phase establishes a high-fidelity and geometrically consistent foundation for the static components of the world, such as buildings and road infrastructure. Upon convergence, we freeze this static backbone and introduce a lightweight residual flow network[19, 20]. This network is uniquely trained to predict only the non-rigid motion residuals corresponding to independently moving objects, rather than the entire motion field. This progressive, static-to-dynamic paradigm offers several advantages: it effectively circumvents the training instabilities common to monolithic end-to-end approaches, and by decomposing the motion, it allows our model to generate temporally coherent and detailed high-fidelity dynamic scenes with computational efficiency necessary for real-time performance.

The main contributions of this paper are summarized as follows: 1) We proposed DrivingScene, an online, feed-forward framework that achieves state-of-the-art 4D dynamic scene reconstruction from only two surround view images and generates valuable intermediate representations, it operates in real-time and trained entirely with selfsupervised objectives. 2) We design a residual flow network with a hybrid-shared architecture. It features a shared backbone to learn a generalized motion prior and lightweight, per-camera heads to adapt to varying camera extrinsics and intrinsics , which keeps consistent scale prediction and computational efficiency across all views. 3) We introduce a coarse-to-fine, two-stage training paradigm. In Stage 1, DrivingScene learns a robust static scene prior. In Stage 2, with the static backbone frozen, a residual flow network is trained to refine the scene by modeling only the non-rigid motion of dynamic objects, ensuring both training stability and high-fidelity results.

<!-- image-->  
Fig. 2. Overview of DrivingScene. Given two consecutive surround-view frames, our framework first predicts a static scene composed of 3D Gaussian primitives using a depth and a Gaussian parameter network. A residual flow network then computes the non-rigid motion field between the frames. This motion is combined with the rigid flow derived from ego-motion and applied as temporal displacements to the static Gaussians, resulting in a complete, dynamic 4D scene representation.

## 2. METHODOLOGY

We introduce DrivingScene, an online, feedforward framework for reconstructing spatio-temporally consistent 4D dynamic scenes from two consecutive, sparse surround view images. Figure 2 illustrates the overall framework of DrivingScene. To effectively learn both static and dynamic scene properties, we devise a coarse-to-fine training paradigm. The first stage provides a robust prior for the sceneâs rigid layout but ignores dynamic motion. Instead of a generic flow network, which would disregard these learned rigid constraints, we introduce the residual flow network, which is trained specifically to predict only the residual, non-rigid motion of dynamic objects on top of the frozen static backbone. This progressive, static-to-dynamic approach enables DrivingScene to explicitly model dynamics via scene flow and perform online, high-fidelity reconstruction.

## 2.1. Static scene geometry and appearance modeling

We ground our scene representation in 3D Gaussian Splatting (3DGS), which explicitly models a scene with a set of Gaussian primitives $\mathcal { G } = \{ \bar { G _ { k } } = \{ \mu _ { k } , \Sigma _ { k } , \alpha _ { k } , \mathbf { c } _ { k } \} \} _ { k = 1 } ^ { K }$ , parameterized by a 3D mean $\pmb { \mu _ { k } }$ , a covariance matrix $\Sigma _ { k }$ , an opacity $\alpha _ { k }$ , and Spherical Harmonic (SH) coefficients $\mathbf { c } _ { k }$ . To enable feed forward inference, we design a depth network $D ,$ , and a Gaussian parameter network P , to directly predict these properties from images.

Given a pair of consecutive surround view image sets and their poses, the depth network D first predicts a per-pixel depth map for each image, which provides the 3D means $( \mu _ { k } )$ for the Gaussian primitives. Subsequently, the network P takes image and depth features as input to infer the remaining attributes.

<!-- image-->  
Fig. 3. The architecture of residual flow network

The Gaussian primitives predicted from each of the six camera views are transformed into a common world coordinate system using the known extrinsic parameters. These individual point clouds are then concatenated to form a single unified scene representation. In this feed-forward paradigm, we do not perform explicit de-duplication or fusion in 3D space. Instead, we rely on the differentiable renderer to handle potential redundancies and inconsistencies during the view synthesis process, where Gaussians that are occluded or inconsistent with the target view will naturally contribute minimally to the final rendered pixel color.

## 2.2. Dynamic modeling via residual scene flow

The static model established in Stage 1 is inherently incapable of capturing independently moving objects. To model these dynamics, we introduce a residual flow network, R. The central principle is to decompose the total motion field into a rigid component $\mathbf { F } _ { \mathrm { r i g i d } }$ and a non-rigid residual component $\mathbf { F } _ { \mathrm { r e s i d u a l } }$ . This allows the network to focus on learning challenging, object-specific motion.

To achieve this efficiently online across multiple views, we introduce a hybrid architecture for R (as depicted in Figure 3). It follows a coarse-to-fine principle, featuring a shared deep encoder backbone to extract a multiscale pyramid of generic motion features and dedicated per-camera pyramidal decoders. During decoding, the flow is iteratively refined from the lowest resolution upward, with each levelâs prediction serving as an initial estimate for the next. This pyramidal refinement strategy is critical for handling large displacements, while the hybrid design ensures consistent scale and a compact parameter footprint.

<!-- image-->  
Fig. 4. Qualitative results of surrounding views. Details from surrounding views are present for easy comparison.

The process is as follows: we first compute the rigid flow field $\mathbf { F } _ { \mathrm { r i g i d } }$ using the predicted depth and known camera poses. The network R then takes a rich set of inputs, including the warped source image, the target image, and the rigid flow, to predict the final residual component $\mathbf { F } _ { \mathrm { r e s i d u a l } }$ . The complete motion field, $\mathbf { F } _ { \mathrm { t o t a l } } = \mathbf { F } _ { \mathrm { r i g i d } } +$ $\mathbf { F } _ { \mathrm { r e s i d u a l } } .$ , is then applied to the means of Gaussian primitives to model their temporal evolution.

## 2.3. Two-stage training and objectives

We propose a two-stage coarse-to-fine training strategy that decouples the learning of static and dynamic scene properties, mitigating the challenges of joint end-to-end optimization.

In the first stage, we exclusively train the depth D and Gaussian parameter P networks. The training is guided by a self-supervised composite loss function $\mathcal { L } _ { \mathrm { s t a g e l } }$ , which combines geometric and rendering objectives:

$$
{ \mathcal { L } } _ { \mathrm { s t a g e 1 } } = \lambda _ { \mathrm { l o c } } { \mathcal { L } } _ { \mathrm { l o c } } + \lambda _ { \mathrm { s m o o t h } } { \mathcal { L } } _ { \mathrm { s m o o t h } } + \lambda _ { \mathrm { r e n d e r } } { \mathcal { L } } _ { \mathrm { r e n d e r } }
$$

Here, the geometric loss, $\mathcal { L } _ { \mathrm { l o c } } .$ , introduced from DrivingForward[17], enforces multiview consistency through a photometric reprojection objective. The smoothness loss $\mathcal { L } _ { \mathrm { { s m o o t h } } }$ , is a regularization term that penalizes large gradients in the disparity map. Finally, the rendering loss $\mathcal { L } _ { \mathrm { r e n d e r } } .$ , ensures visual fidelity by minimizing the difference between the rendered image $I _ { \mathrm { r e n d e r } }$ and the ground truth image $I _ { \mathrm { { g t } } }$ using a combination of L2 photometric loss and perceptual LPIPS losses [21], with the weight $\lambda _ { \mathrm { p } }$ set to 0.05:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e n d e r } } = \mathcal { L } _ { \mathrm { L 2 } } ( I _ { \mathrm { r e n d e r } } , I _ { \mathrm { g t } } ) + \lambda _ { \mathrm { p } } \mathcal { L } _ { \mathrm { L P I P S } } ( I _ { \mathrm { r e n d e r } } , I _ { \mathrm { g t } } ) } \end{array}
$$

Upon convergence of the static model, we freeze the weights of D and P and exclusively train the residual flow network R. The total loss $\mathcal { L } _ { \mathrm { s t a g e } 2 }$ is a weighted sum of three self-supervised components:

$$
{ \mathcal { L } } _ { \mathrm { s t a g e 2 } } = \lambda _ { \mathrm { w a r p } } { \mathcal { L } } _ { \mathrm { w a r p } } + \lambda _ { \mathrm { c o n s i s t } } { \mathcal { L } } _ { \mathrm { c o n s i s t } } + \lambda _ { \mathrm { r e n d e r } } { \mathcal { L } } _ { \mathrm { r e n d e r } }
$$

The flow consistency loss $\mathcal { L } _ { \mathrm { c o n s i s t } }$ provides geometric regularization through a forward-backward check. The Gaussian rendering loss $\mathcal { L } _ { \mathrm { r e n d e r } }$ uses the same formulation as in Stage 1 to provide end-to-end supervision. The flow warping loss ${ \mathcal { L } } _ { \mathrm { w a r p } }$ enforces photometric consistency on the warped image $\hat { I } _ { t + 1 } = W ( I _ { t } , { \bf F } _ { \mathrm { t o t a l } } )$ . It is a composite objective combining three distinct error metrics:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { w a r p } } = \mathcal { L } _ { \mathrm { L } 1 } ( I _ { \mathrm { t + 1 } } , \hat { I } _ { \mathrm { t + 1 } } ) + \lambda _ { \mathrm { s } } \mathcal { L } _ { \mathrm { S S I M } } ( I _ { \mathrm { t + 1 } } , \hat { I } _ { \mathrm { t + 1 } } ) + \lambda _ { \mathrm { w p } } \mathcal { L } _ { \mathrm { L P I P S } } ( I _ { \mathrm { t + 1 } } , \hat { I } _ { \mathrm { t + 1 } } ) } \end{array}
$$

Table 1. Quantitative comparison for novel view synthesis on the nuScenes validation set.
<table><tr><td>Method</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>MVSplat</td><td>22.83</td><td>0.629</td><td>0.327</td></tr><tr><td>DepthSplat</td><td>24.21</td><td>0.732</td><td>0.271</td></tr><tr><td>StreetGaussian</td><td>25.59</td><td>0.765</td><td>0.212</td></tr><tr><td>DrivingForward</td><td>26.06</td><td>0.781</td><td>0.215</td></tr><tr><td>Driv3R</td><td>26.10</td><td>0.808</td><td>0.084</td></tr><tr><td>DrivingScene</td><td>28.76</td><td>0.895</td><td>0.113</td></tr></table>

where $\mathcal { L } _ { \mathrm { L 1 } } , \mathcal { L } _ { \mathrm { S S I M } } .$ , and ${ \mathcal { L } } _ { \mathrm { L P I P S } }$ denote the L1 photometric loss, the Structural Similarity (SSIM) loss [22], and the perceptual LPIPS loss, with weights set to $\lambda _ { \mathrm { s } } = 0 . 1$ and $\lambda _ { \mathrm { w p } } = 0 . 0 5$ , respectively.

## 3. EXPERIMENTS

## 3.1. Experimental setup

Our model is implemented in PyTorch and trained on NVIDIA RTX5090 GPUs (32GB). We use the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 1. Our two-stage training proceeds as follows: Stage 1 (6 epochs) uses loss weights $\lambda _ { \mathrm { { r e n d e r } } } = 0 . 0 1 , \lambda _ { \mathrm { { l o c } } } = 0 . 1$ , and $\lambda _ { \mathrm { s m o o t h } } = 0 . 0 0 1$ . Stage 2 (6 epochs) uses weights $\lambda _ { \mathrm { r e n d e r } } = 0 . 0 1 , \lambda _ { \mathrm { c o n s i s t } } = 1 0 ^ { - 5 }$ , and $\lambda _ { \mathrm { w a r p } } ~ = ~ 0 . 0 2 .$ We evaluate on the official split of the nuScenes dataset (700/150 scenes) at 352 Ã 640 resolution. The primary task is novel view synthesis of the intermediate temporal frame between two keyframes, evaluated using PSNR, SSIM, and LPIPS.

We compare against leading online, feed-forward methods. Our primary baseline is DrivingForward [17], a static reconstruction method whose limitations with dynamic objects we directly address. We also provide extensive comparisons against the dynamic method Driv3R [18], other static approaches (DepthSplat [23], MVSplat), and the per-scene optimization method StreetGaussian, aligning our setup with their protocols for a fair comparison.

## 3.2. Quantitative and qualitative comparison

Quantitative results for novel view synthesis are presented in Table 1. DrivingScene achieves state-of-the-art performance, outperforming all feed-forward baselines across all metrics. This demonstrates the superior quality of our 4D reconstructions.

Table 2. Quantitative comparison for depth comprasion
<table><tr><td>Method</td><td>Abs Rel â</td><td>Sq Rel â</td><td>RMSE â</td></tr><tr><td>Driv3R</td><td>0.234</td><td>2.279</td><td>7.298</td></tr><tr><td>DrivingScene</td><td>0.227</td><td>2.195</td><td>7.254</td></tr></table>

<!-- image-->  
Fig. 5. The comparison of rigid flow with full flow

Table 3. Efficiency analysis
<table><tr><td rowspan="2">Method</td><td colspan="2">Inference</td><td colspan="2">Training</td></tr><tr><td>Time â</td><td>VRAMâ</td><td>Time â</td><td>VRAM â</td></tr><tr><td>DrivingForward</td><td>0.34S</td><td>7.58GB</td><td>â 3 days</td><td>40.0GB</td></tr><tr><td>Driv3R</td><td>0.71s</td><td>5.04GB</td><td>â 7.5days</td><td>175.5GB</td></tr><tr><td>DrivingScene</td><td>0.21S</td><td>6.48GB</td><td>â 5 days</td><td>27.3GB</td></tr></table>

The qualitative comparisons in Figure 4 further highlight the advantages of our approach. In particular, the comparison with DrivingForward showcases the critical importance of our dynamic modeling. While DrivingForward achieves strong results in static parts of the scene, its static assumption leads to significant ghosting and blurring artifacts for moving objects, such as vehicles and pedestrians. DrivingScene effectively resolves these dynamic elements, producing sharp and temporally consistent reconstructions that faithfully capture the sceneâs motion. Compared to Driv3R, our method demonstrates superior fine-grained detail reconstruction and overall visual fidelity.

A key advantage of our framework is the generation of highquality intermediate representations. We compare our predicted depth maps with Driv3R[18] in Table 2. The results show that our method produces more accurate and geometrically coherent depth, validating the effectiveness of our explicit, multitask prediction approach. This superior geometric understanding is a key factor contributing to our higher rendering fidelity. Furthermore, we visualize the decomposed flow fields in Figure 5. The rigid flow component correctly captures the global scene motion induced by the ego-vehicle, while the learned residual flow successfully isolates and highlights non-rigidly moving objects. This provides clear evidence for the efficacy of our residual motion modeling strategy.

## 3.3. Efficiency analysis

We evaluated the computational efficiency of our method against DrivingForward and Driv3R in terms of training time, inference speed, and GPU memory consumption. As detailed in Table 3, when synthesizing a full surround-view scene (6 images at 352 Ã 640), our method not only achieves a faster inference frame rate. The reported memory usage further confirms that our approach is more resource-efficient during both training and inference. Furthermore, as shown in Table 4, DrivingScene maintains a compact model size with significantly fewer parameters compared to Driv3R and DrivingForward. This highlights the efficiency of our hybrid-shared architecture and residual learning approach.

Table 4. Model complexity comparison.
<table><tr><td>Method</td><td>DrivingForward</td><td>Driv3R</td><td>DrivingScene</td></tr><tr><td>Params â</td><td>0.173GB</td><td>2.512GB</td><td>0.117GB</td></tr></table>

Table 5. Ablation studies on the key components of our method.
<table><tr><td>Configuration</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>Full Model</td><td>28.76</td><td>0.895</td><td>0.113</td></tr><tr><td>1. w/o Residual Flow</td><td>26.40</td><td>0.780</td><td>0.201</td></tr><tr><td>2. Single-Stage Training</td><td>13.69</td><td>0.334</td><td>0.731</td></tr><tr><td>3. w/o Flow Warping Loss</td><td>27.32</td><td>0.872</td><td>0.145</td></tr></table>

## 3.4. Ablation studies

To systematically validate the key design choices of our method, we conduct a series of ablation studies. The results are summarized in Table 5.

Efficacy of Residual Flow. To verify the necessity of our dynamic modeling, we train a static-only variant of our model by disabling the residual flow network. This configuration is conceptually similar to the DrivingForward framework. The significant performance drop observed in the results confirms that explicitly modeling scene dynamics via our residual flow strategy is crucial for highquality reconstruction in realistic driving scenarios.

Efficacy of Two-Stage Training. We compare our two-stage paradigm with a single-stage, end-to-end training alternative, where all loss functions are activated from the beginning. This joint training approach leads to a substantial degradation in performance. We observe that it impairs the modelâs ability to learn scale-aware geometry, underscoring the importance of establishing a robust static prior before refining with dynamic information.

Efficacy of Flow Warping Loss. Finally, we investigate the contribution of flow warping loss ${ \mathcal { L } } _ { \mathrm { w a r p } }$ by removing it from the Stage 2 objective. The results show a noticeable decline in rendering quality, confirming that this loss provides a critical supervisory signal that tightly couples our motion estimation with the final rendering task, thereby enhancing multitask consistency.

## 4. CONCLUSION

In this paper, we introduced DrivingScene, an online, feed-forward framework for high-fidelity 4D reconstruction of dynamic driving scenes. Our key innovation is a two-stage, static-to-dynamic training strategy that decouples the learning of static geometry from dynamic motion, proving to be both effective and stable. DrivingScene surpasses key baselines like DrivingForward and Driv3R in rendering quality and computational efficiency, while concurrently generating high-quality intermediate outputs like depth and scene flow. While DrivingScene shows significant progress, future work could explore integrating information over longer temporal windows to enhance robustness, or adopting more expressive, per-Gaussian deformation models to handle a wider range of dynamic phenomena.

## 5. REFERENCES

[1] Andreas Geiger, Philip Lenz, and Raquel Urtasun, âAre we ready for autonomous driving? the kitti vision benchmark suite,â in 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012, pp. 3354â3361.

[2] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom, ânuscenes: A multimodal dataset for autonomous driving,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11621â11631.

[3] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al., âScalability in perception for autonomous driving: Waymo open dataset,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 2446â2454.

[4] Jonah Philion and Sanja Fidler, âLift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d,â in European conference on computer vision. Springer, 2020, pp. 194â210.

[5] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai, âBevformer: learning birdâs-eye-view representation from lidar-camera via spatiotemporal transformers,â IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[6] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[7] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis, â3d gaussian splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â 1, 2023.

[8] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng, âStreet gaussians: Modeling dynamic urban scenes with gaussian splatting,â in European Conference on Computer Vision. Springer, 2024, pp. 156â173.

[9] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang, âDrivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 21634â 21643.

[10] Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, et al., âEmernerf: Emergent spatial-temporal scene decomposition via self-supervision,â arXiv preprint arXiv:2311.02077, 2023.

[11] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser, âIbrnet: Learning multi-view image-based rendering,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 4690â4699.

[12] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa, âpixelnerf: Neural radiance fields from one or few images,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 4578â4587.

[13] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny, âVggt: Visual geometry grounded transformer,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294â5306.

[14] Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard Newcombe, and Zhaoyang Lv, â4dgt: Learning a 4d gaussian transformer using real-world monocular videos,â arXiv preprint arXiv:2506.08015, 2025.

[15] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann, âpixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 19457â19467.

[16] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai, âMvsplat: Efficient 3d gaussian splatting from sparse multi-view images,â in European Conference on Computer Vision. Springer, 2024, pp. 370â386.

[17] Qijian Tian, Xin Tan, Yuan Xie, and Lizhuang Ma, âDrivingforward: Feed-forward 3d gaussian splatting for driving scene reconstruction from flexible surround-view input,â in Proceedings of the AAAI Conference on Artificial Intelligence, 2025, vol. 39, pp. 7374â7382.

[18] Xin Fei, Wenzhao Zheng, Yueqi Duan, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Jiwen Lu, âDriv3r: Learning dense 4d reconstruction for autonomous driving,â arXiv preprint arXiv:2412.06777, 2024.

[19] Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, and Thomas Brox, âFlownet 2.0: Evolution of optical flow estimation with deep networks,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2462â2470.

[20] Zhichao Yin and Jianping Shi, âGeonet: Unsupervised learning of dense depth, optical flow and camera pose,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 1983â1992.

[21] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.

[22] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE transactions on image processing, vol. 13, no. 4, pp. 600â612, 2004.

[23] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys, âDepthsplat: Connecting gaussian splatting and depth,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 16453â16463.