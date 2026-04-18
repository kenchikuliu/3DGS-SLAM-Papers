# DQO-MAP: Dual Quadrics Multi-Object mapping with Gaussian Splatting

Haoyuan Li1, Ziqin Ye1, Yue Hao1, Weiyang Lin1, Chao Ye1,â 

Abstractâ Accurate object perception is essential for robotic applications such as object navigation. In this paper, we propose DQO-MAP, a novel object-SLAM system that seamlessly integrates object pose estimation and reconstruction. We employ 3D Gaussian Splatting for high-fidelity object reconstruction and leverage quadrics for precise object pose estimation. Both of them management is handled on the CPU, while optimization is performed on the GPU, significantly improving system efficiency. By associating objects with unique IDs, our system enables rapid object extraction from the scene. Extensive experimental results on object reconstruction and pose estimation demonstrate that DQO-MAP achieves outstanding performance in terms of precision, reconstruction quality, and computational efficiency. The code and dataset are available at: https://github.com/LiHaoy-ux/DQO-MAP.

## I. INTRODUCTION

VISION-based Simultaneous Localisation and Mapping (SLAM) plays a crucial role in the field of AR/VR and robotics. Previous works concentrated to provide accurate ego estimation and environment maps for navigation. However, these maps consisted of discrete points are only contain metric information, which limits their application in more complex tasks, such as object navigatior that require scene understanding. With deep learning method, it is possible to introduce semantic information into the map. And Object-SLAM, which is aimed at constructing object-level map with semantic information, has attracted the interest of many researchers.

Unlike point-level maps, Object-SLAM systems [1], [2] utilize detections or semantic information to construct a map that includes objectsâ poses and locations in the scene, which can be leveraged for downstream applications. However, a crucial challenge is how to make objects representation more effective. Some researchers replace original objects with geometric primitives (e.g., cubes, quadrics). These compact representations serve as higher-dimensional features for navigation, including key information, such as category, size, pose, and location. Object-level maps offer advantages in long-term relocalization compared to point-level maps. However, geometric primitives lack valuable information about shape and texture features, posing a challenge for widespread application.

The reconstruction of objectsâ shape has been extensively studied. Some researchers have explored depth-based methods to resolve geometry, such as surfels and signed distance functions (SDFs). More recently, neural networks have gained popularity for inferring and optimizing geometry in latent spaces, which can then be decoded into voxel grids or implicit functions. While these approaches can reconstruct complete objects from sparse observations, they are often constrained by prior assumptions and struggle to handle arbitrary geometric shapes. A natural question arises: Can we reconstruct objects without prior knowledge? Neural Radiance Fields (NeRF) [3] and 3D Gaussian Splatting (3DGS) [4] offer promising solutions, enabling geometry reconstruction from RGB and depth images alone. Compared to NeRF, 3DGS achieves higher efficiency through explicit representation and rasterization, and has recently shown excellent performance in SLAM applications.

<!-- image-->  
Fig. 1: DQO-MAP simultaneously reconstructs objects using Gaussian Splatting and estimates their poses with quadrics.Each object is assigned a unique ID for association and extraction.

In this paper, we present an online multi-object reconstruction system with two tightly integrated components: object pose estimation and object reconstruction. Built on ORB-SLAM2 [5], the system initializes objects and Gaussians on the CPU while optimizing them in parallel on the GPU. Guided by object IDs, it enables fast object extraction from the scene.

The contributions of this work are summarized as follows:

â¢ We propose a pose-free 3D multi-object mapping system that can simultaneously perform pose estimation and reconstruction of scene objects.

â¢ We present an object-level association framework that effectively aggregates different object measurements to enhance 3D-2D correspondence accuracy.

â¢ We introduce an efficient object loss function and an incremental update strategy, enabling real-time performance.

â¢ Comprehensive experiments across synthetic and realworld datasets demonstrate the systemâs superior object perception performance.

## II. RELATED WORK

## A. Object SLAM

As the pioneering object-based SLAM system, SLAM++ [6] utilized depth data to match object models and optimized object maps through pose-graph optimization. However, the dependence on prior object knowledge limits its real applications. Subsequently, researcher focus on online object reconstruction. Fusion++ [7], MaskFusion [8] employed Mask R-CNN for instance segmentation and fused multi-frame observations to estimate TSDF, but its implementation is dependent on computational resources.

In contrast to dense reconstruction approaches, some studies explored lightweight object maps using geometric primitives. Cube SLAM [1] reconstructed 3D cuboids from 2D bounding boxes (BBox), assuming coplanar object placement to simplify the optimization. Quadric SLAM [2] parameterized objects as differentiable quadrics and optimized them via reprojection errors, but it requires manual association. To address this issue, VOOM [9] established data association relationships using ORB feature and quadric distance metrics.

Volume rendering and Gaussian splatting provide alternative methods for localization and reconstruction. vMAP [10] encoded objects via MLPs, prioritizing rendering fidelity over geometric precision. RO-MAP [11] decoupled construction of object maps into reconstruction with NeRF and object pose estimation used cuboid. However, it is inherited NeRFâs computational limitations and it is still a loosely coupled system. Our work tightly integrates object pose estimation and reconstruction by combining quadric parameterization with 3DGS for efficient simultaneous optimization.

## B. Gaussian Splatting in SLAM

The 3DGS [4], as an explicit radiance field representation, outperforms NeRF in rendering speed and geometric interpretability, consequently gaining widespread adoption. However, 3DGS still faces challenges to use directly for Object SLAM. Existing 3DGS frameworks prioritize global scene reconstruction over object-level modeling. SiLVR [12] optimized scenes via submap stitching without isolating dynamic objects, while Mip-Splatting [13] enhanced rendering quality but lacks instance-level editing. Although these methods reconstruct entire scenes, extracting specific objects requires additional post-processing.

Another obstacle to the further utilization of 3DGS are resources and efficiency. As explicit expressions, numerous variables need to be stored and optimized. For instance, MonoGS [14] and Splatam [15] added and optimized Gaussian at each frame, which incurs significant overhead in the scene. To overcome the problem, RTG-SLAM [16] classifie and managed Gausssians into different categories, which increased speed and reduced memory consumption.

Our object-centric 3DGS-SLAM framework decouples scene and object Gaussians based on object associations, enabling parallel instance-level reconstruction. By limiting Gaussian types and refining the update strategy, our approach retains 3DGSâs real-time rendering benefits while enhancing scene robustness through object-geometric constraints.

## III. METHOD

The proposed methodâs architecture is illustrated in Fig.2. Our system combines object reconstruction with object-level mapping. Given RGB-D and instance frames, it performs simultaneously object pose estimation and reconstruction. Guided by the association results, the system enables decouple objects from the scene. Subsequently, we employ Tsdf-Fusion [17] to generate a 3D mesh of the object.

## A. Objectsâ Pose and Size Estimation

Unlike previous methods [9], [11] that estimate only the yaw angle, our approach estimates full 6-DOF object poses. While prior work [18] showed that instance segmentation improves orientation accuracy, it is computationally expensive and often unstable (e.g., when using OpenCV), complicating data association. We employ YOLOv10 [19] as the detector to generate 2D bounding boxes (BBoxes).

We use dual quadrics as a geometric primitive to estimate an objectâs pose. For the kth object $O _ { k }$ , itâs quadric $Q _ { k } ^ { * }$ could be calculate as:

$$
\pi ^ { T } Q _ { k } ^ { * } \pi = 0\tag{1}
$$

where $\pi$ is formed by the back projection of any edge of 2D BBox to the world coordinates. As the shape of dual quadrics is the same as 3D Gaussian, the $Q _ { k } ^ { * }$ could be simply decoupled as follows:

$$
\begin{array} { r l } & { \boldsymbol { \mu } = \left( x _ { k } \quad y _ { k } \quad z _ { k } \right) ^ { T } } \\ & { \Sigma ^ { - 1 } = R ( \boldsymbol { \theta } ) ^ { T } S ^ { T } S R ( \boldsymbol { \theta } ) } \end{array}\tag{2}
$$

where $R ( \theta ) , \mu$ represents the rotation and location of object, respectively. And S is formed by the scale $[ a , b , c ]$ , which can be expressed as:where $R ( \theta ) , \mu$ represents the rotation and location of object, respectively. And $S$ is formed by the scale $[ a , b , c ] .$ , which can be expressed as:

$$
S = { \left( \begin{array} { l l l } { { \frac { 1 } { a ^ { 2 } } } } & { 0 } & { 0 } \\ { 0 } & { { \frac { 1 } { b ^ { 2 } } } } & { 0 } \\ { 0 } & { 0 } & { { \frac { 1 } { c ^ { 2 } } } } \end{array} \right) }\tag{3}
$$

## B. Object-Level Association

The object association is a key problem for the multiobject system. Unlike only using object id known in advance, we design a coarse-to-fine association strategy for 3D-2D object information association, as shown in Fig.3. We first project all 3D objects to the image as follows:

$$
C _ { k } ^ { * } = P ^ { T } Q _ { k } ^ { * } P\tag{4}
$$

where $P = K \cdot R t \in \mathbb { R } ^ { 3 \times 4 }$

<!-- image-->  
Fig. 2: Overview of our proposed system. DQO-MAP tightly integrate object pose estimation and reconstruction, leveraging quadrics for object estimation and 3DGS for reconstruction.

Then, we use the Intersection over Union (IoU) between objectsâ projection and 2D BBoxes of each frame(in Fig.3(a)) to filter the coarse results.

<!-- image-->  
Fig. 3: Different dtypes of data association

All observed BBoxes matched are used for initialization object $O ^ { o b s }$ , then we use a standard distance between quadrics (QD) to filter out wrong matched(in Fig.3(b)). The QD is defined as :

$$
Q D = e x p ( - \tau \cdot ( | | u ^ { o b s } - u k | | _ { 2 } + | | S ^ { o b s } - S | | _ { F } ) )\tag{5}
$$

where Ï is a constant and $| | \cdot | | _ { F }$ is the Frobenius norm. If the $Q D >$ thre (thre is a threshold), the result will be filtered out.

For occlusion cases(in Fig.3(c)), an object will be observed into pieces. The objectâs parameters will be updated to estimate the whole object. When there are two objects has a same id and $A r e a ( B B o x _ { j } ) < A r e a ( B B o x _ { i } )$ , the score t is calculated as follos:

$$
t = O v e r L a p ( B B o x _ { i } , B B o x _ { j } ) / B B o x _ { j }\tag{6}
$$

If $t > t _ { t h r e } ( t _ { t h r e } = 0 . 8 5 )$ )and $Q D < d ( d = 0 . 1 ) \quad$ , the two objects are from the whole object, update the parameters of quadrics by pop the $O _ { j }$

## C. Gaussains Incremental Update

Training all Gaussians per frame is inefficient and does not always enhance geometric stability. Following prior work [16], we adopt an incremental update strategy for classifying Gaussians as opaque Gaussians (OG) for geometry fitting and transparent Gaussians (TG) for color correction. Instead of updating all Gaussians, we optimize only those linked to unstable masks, determined by color, depth, and instance errors.

$$
\begin{array} { r l } & { M a s k _ { g e o } = \left\{ u _ { g e o } \vert \mathrm { i n s } < \theta _ { \alpha } , \quad \mathrm { o r } \quad \vert d - \hat { d } \vert > \theta _ { d } \right\} } \\ & { M a s k _ { r g b } = \left\{ u _ { c } \vert \vert c - \hat { c } \vert > \theta _ { c } \right\} } \end{array}\tag{7}
$$

where M as $k _ { g e o }$ indicates the object should be added new geometry. ins $< ~ \theta _ { \alpha }$ means the area may not remain an object, and $| d - \hat { d } | > \theta _ { d }$ means that the object is not reconstructed completely, and $\theta _ { d } = 0 . 1 . \ M a s k _ { r g b }$ represents wrong area, and $\theta _ { c } = 0 . 1$

With objectsâ 3D-2D association across frames, we assign object IDs to every Gaussian, which is a simple but efficient way to separate objects from the scene. Through this way, the Gaussians belonging to $O _ { K }$ need to be added or optimized could be written as:

$$
M a s k ^ { O _ { k } } = \left\{ M a s k _ { g e o } ^ { i d } , M a s k _ { r g b } ^ { i d } \mid i d = k \right\}\tag{8}
$$

## D. Object Pose and Reconstruction Training Loss

Object pose and Gaussians are optimized trained parallel on GPUs.

1) Object Pose Traing Loss: the parameters of quadrics will be optimized by minimizing the IOU loss.

$$
J = \sum _ { i \in F _ { O _ { k } } } I O U ( B B o x ( C _ { k } ) , B B o x _ { i } ^ { o b s } )\tag{9}
$$

where $F _ { O _ { k } }$ is a set of frame observe object $O _ { k }$

2) Reconstruction Training Loss:: For a set of Gaussian, $G ^ { k }$ belonging to object $O ^ { k }$ , the RGB loss is written as:

$$
L _ { r g b } ^ { o b j } = \sum _ { i \in G ^ { k } } | | c _ { i } - \hat { c _ { i } } | | _ { 2 }\tag{10}
$$

For geometry, we use a depth a loss:

$$
L _ { d e p t h } ^ { o b j } = \sum _ { i \in G ^ { k } } | | d _ { i } - \hat { d } _ { i } | | _ { 2 }\tag{11}
$$

As Gaussian splatting uses Î± blending to render RGB and depth, the edges between objects and the background are

blurred. We use instance information ins $\in \ [ 0 , 1 ]$ to limit the outline of the object. The insi is calculated as:

$$
i n s = \sum _ { i = 1 } ^ { N } \alpha _ { i } , \alpha _ { i } \in O G\tag{12}
$$

where $\alpha _ { i }$ means Gaussainâs opacity rendered on the pixel. And the instance loss $L _ { i n s }$ is defined as follows:

$$
L _ { i n s } ^ { o b j } = \sum | | i n s _ { i } - i \hat { n s } _ { i } | | _ { 2 }\tag{13}
$$

The overall training losses are accumulated for object $O _ { K } \mathrm { { : } }$

$$
L _ { t o t a l } ^ { O _ { k } } = L _ { r g b } + L _ { d e p t h } + \lambda L _ { i n s }\tag{14}
$$

where Î» is a constant value.

## IV. EXPERIMENTS

We evaluate the performance of the proposed system on both synthetic and real-world datasets. We first evaluate the quality of the objectsâ reconstruction and then test the objectsâ pose. Finally, an ablation study of the state of the data association is performed.

Implementation Detail: The propsed system in implemented on a desktop with a 2.10Hz Intel(R) 5218R CPU, and a NVIDIA 3090 24GB GPU. We set $O G \ = \ 0 . 9 .$ $T G = 0 . 1$ , and the Tsdf-Fusion is used to extract mesh, with the $s c a l e = 0 . 8$ . We implement tracking, mapping and joint optimization parts with Pytorch framework, and leverage CUDA kernels for rasterization and back propagation. And we used Azure Kinect RGBD camera for real-world test.

Baselines: For reconstruction, we compare to the classical offline method, COLMAP [20], Tsdf-Fusion [17], and Nerfbased object reconstruction method, vMAP [10], RO-MAP [11]. Addtionally, we also compoared our method with a classical GS-based approach, MonoGS [14]. For pose estimation, we compared to RGBD object-SLAM, VOOM [9].

Datasets and Matrics: For reconstruction evaluation, we evaluate the Cube-Diorama dataset and the Replica dataset, which provide ground truth (GT) instance segment and mesh. Accuracy, Completion are used for quantitative evaluation of object reconstruction. Subsequently, we qualitatively evaluate the system on a self-collected real-world dataset. For object pose, we use IoU and distance to evaluate on our own simulation dataset, which is collected by Ai2-THOR [21] with GT.

## A. Evaluation of Online Object Reconstruction

1) Cube-Diorama: As other methods are focus on scene reconstruction, we use post-processing to extract objects from the scene. The results, presented in Fig.4, indicate that the TSDF-Fusion and MonoGS struggle with detailed texture preservation, while RO-MAP exhibits insufficient surface smoothness and partial surface incompleteness. In contrast, our method improves boundary clarity and overall integrity while maintaining fusion-level reconstruction quality. To quantify performance, we analyzed four objects in the Room scene with Accuracy (Acc.), Completion (comp), and the results are shown in Table I. Our method achieves high completeness, especially for errors under 1 cm, surpassing others in detail preservation. While RO-MAP, as a NeRFbased approach, excels in completeness, our method better preserves fine details. Overall, our results outperform other object reconstruction methods.

<!-- image-->

<!-- image-->  
Fig. 4: Object reconstruction results in Room

TABLE I: Quantitative Evaluation of Object Reconstruction on ROOM. Bold and underline indicate the best and the second-best respectively.
<table><tr><td>Method</td><td>Acc. [cm]â</td><td>Comp. [cm]â</td><td>Comp.Ratio [&lt;1cm%]â</td></tr><tr><td>COLMAP</td><td>3.10</td><td>0.36</td><td>91.48</td></tr><tr><td>Tsdf-Fusion</td><td>3.12</td><td>0.56</td><td>89.95</td></tr><tr><td>RO-MAP</td><td>2.23</td><td>0.42</td><td>94.34</td></tr><tr><td>MonoGS</td><td>3.69</td><td>1.04</td><td>66.72</td></tr><tr><td>Ours</td><td>1.92</td><td>0.34</td><td>92.54</td></tr></table>

2) Replica: In order to further test the performance of the proposed method in multiple objects and large scenes, we tested our method on Replica, with results shown in Fig. 5. TSDF-Fusion struggles with detail, texture, and structure, while vMAP and MonoGS achieve higher structural accuracy. Our method delivers superior geometric accuracy and texture fidelity while operating efficiently. In Table II, our method achieves high accuracy in indoor environments, with a notably high reconstruction rate for errors <5 cm, significantly outperforming others across metrics.

<!-- image-->  
Fig. 5: Object reconstruction results in Replica

TABLE II: Quantitative Evaluation of Object Reconstruction on Replica.
<table><tr><td>Scene</td><td>Method</td><td>Acc. [cm]â</td><td>Comp. [cm]â</td><td>Comp.Ratio [&lt;5cm%]â</td></tr><tr><td rowspan="5">room0</td><td>vMAP</td><td>5.27</td><td>4.08</td><td>36.74</td></tr><tr><td>Tsdf-Fusion</td><td>16.03</td><td>3.89</td><td>86.46</td></tr><tr><td>MonoGS</td><td>0.62</td><td>0.92</td><td>89.54</td></tr><tr><td>Ours</td><td>1.21</td><td>1.07</td><td>90.42</td></tr><tr><td>vMAP</td><td>3.15</td><td>15.55</td><td>49.09</td></tr><tr><td rowspan="4">office0</td><td>Tsdf-Fusion</td><td>8.91</td><td>10.01</td><td>82.57</td></tr><tr><td>MonoGS</td><td>0.61</td><td>8.64</td><td>82.02</td></tr><tr><td></td><td></td><td>4.43</td><td></td></tr><tr><td>Ours</td><td>0.82</td><td></td><td>86.21</td></tr></table>

3) Real World: To evaluate the robustness of our method, we conducted experiments on a real-world scene captured using the Azure Kinect RGBD camera, as shown in Fig. 6. Given the restricted camera viewpoints and the presence of irregular objects, we conducted a qualitative analysis on the scene and single object. The results demonstrate that our method successfully extracted detailed object meshes, even in conditions with sparse data.

4) Runtime and Memory: In addition to reconstruction quality, we compared the average runtime and memory costs of our method with other online methods, as detailed in Table III. Our method achieves faster tracking and mapping than NeRF and 3DGS methods. By incremental update strategy, we accelerated the runtime while maintaining quality.

<!-- image-->  
Fig. 6: Object reconstruction results in real dataset

TABLE III: Runtime and Memory Cost of Object Reconstruction on ROOM
<table><tr><td>Method</td><td>Tracking(s)</td><td>Mapping(s)</td><td>FPS(fps)</td><td>Model Size (MB)</td></tr><tr><td>RO-MAP</td><td>0.142</td><td>0.143</td><td>6.99</td><td>806.09</td></tr><tr><td>MonoGS</td><td>1.2365</td><td>0.7773</td><td>0.49</td><td>2498.52</td></tr><tr><td>Ours</td><td>0.152</td><td>0.074</td><td>13.46</td><td>907</td></tr></table>

## B. Evaluation of Object Pose Estimation

We evaluate the pose estimation with RGBD quadrics SLAM on synthetic dataset colllected by Ai2-THOR [21] as shown in Fig.7, which provided GT of objects. We use Center Distance Error (CDE, cm) and IOU (2D and 3D IOU) between estimated and GT to evaluate the accuracy, and the results are shown in Table IV. The experimental results show that our method achieves competitive accuracy among the scene.

TABLE IV: ACCURACY OF OBJECT POSE ESTIMA-TION
<table><tr><td>Scene</td><td>Metrics</td><td>VOOM</td><td>Ours</td></tr><tr><td rowspan="3">ROOM</td><td>3D IoUâ</td><td>0.561</td><td>0.572</td></tr><tr><td>2D IoUâ</td><td>0.650</td><td>0.752</td></tr><tr><td>CDEâ</td><td>0.93</td><td>0.90</td></tr><tr><td rowspan="3">Ai2-THOR1</td><td>3D IoUâ</td><td>0.304</td><td>0.467</td></tr><tr><td>2D IoU</td><td>0.722</td><td>0.791</td></tr><tr><td>CDTâ</td><td>2.5</td><td>1.1</td></tr><tr><td rowspan="3">Ai2-THOR2</td><td>3D IoUâ</td><td>0.467</td><td>0.510</td></tr><tr><td>2D IoUâ</td><td>0.667</td><td>0.723</td></tr><tr><td>CDEâ</td><td>1.4</td><td>1.3</td></tr></table>

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
(c)

<!-- image-->  
(d)  
Fig. 7: Evaluation of object pose in self-collected Ai2-THOR

## C. Ablation Study

In this section, we validate the effectiveness of the proposed data associdation strategy. As Table V and Fig.8 show, the performance of different data association strategies, only

TABLE V: DATA ASSOCIATION RESULTS
<table><tr><td>Scene</td><td>Only IoU</td><td>Only QD</td><td>QD+IoU</td><td>GT</td></tr><tr><td>ROOM</td><td>10</td><td>6</td><td>4</td><td>4</td></tr><tr><td>room0</td><td>30</td><td>27</td><td>23</td><td>23</td></tr><tr><td>office0</td><td>18</td><td>15</td><td>16</td><td>17</td></tr></table>

IoU, only QD, and a QD+IoU, are systematically compared across three distinct scenes. The results indicate that the QD+IoU approach outperforms the individual metrics, demonstrating that integrating both metrics enhances the accuracy of association.

<!-- image-->

<!-- image-->

<!-- image-->  
(c)

<!-- image-->  
Fig. 8: Qualitative comparison of data association results. (a) Only IoU method. (b)Only QD method. (c)IOU conbined QD method.

## V. CONCLUSIONS

We present DQO-MAP, a novel object SLAM system tightly integrate object pose estimation and reconstruction. Our approach employs 3D Gaussian Splatting for object reconstruction and leverages quadrics for precise pose estimation. While Gaussian management and object association are handled on the CPU, all components are optimized in parallel on the GPU, significantly enhancing system efficiency. Comprehensive experiments demonstrate that our system excels in both object reconstruction and pose estimation. In the future, we plan to focus on leveraging object maps for downstream tasks such as object navigation, robotic manipulation and scene understanding.

## REFERENCES

[1] Shichao Yang and Sebastian Scherer. Cubeslam: Monocular 3-d object slam. IEEE Transactions on Robotics, 35(4):925â938, 2019.

[2] Lachlan Nicholson, Michael Milford, and Niko Sunderhauf. Quadric- Â¨ slam: Dual quadrics from object detections as landmarks in objectoriented slam. IEEE Robotics and Automation Letters, 4(1):1â8, 2018.

[3] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

[4] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Â¨ Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023.

[5] Raul Mur-Artal and Juan D. Tard Â´ os. ORB-SLAM2: an open-source Â´ SLAM system for monocular, stereo and RGB-D cameras. IEEE Transactions on Robotics, 33(5):1255â1262, 2017.

[6] Renato F Salas-Moreno, Richard A Newcombe, Hauke Strasdat, Paul HJ Kelly, and Andrew J Davison. Slam++: Simultaneous localisation and mapping at the level of objects. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1352â1359, 2013.

[7] John McCormac, Ronald Clark, Michael Bloesch, Andrew Davison, and Stefan Leutenegger. Fusion++: Volumetric object-level slam. In 2018 international conference on 3D vision (3DV), pages 32â41. IEEE, 2018.

[8] Martin Runz, Maud Buffier, and Lourdes Agapito. Maskfusion: Real-time recognition, tracking and reconstruction of multiple moving objects. In 2018 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages 10â20. IEEE, 2018.

[9] Yutong Wang, Chaoyang Jiang, and Xieyuanli Chen. Voom: Robust visual object odometry and mapping using hierarchical landmarks. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 10298â10304. IEEE, 2024.

[10] Xin Kong, Shikun Liu, Marwan Taher, and Andrew J Davison. vmap: Vectorised object mapping for neural field slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 952â961, 2023.

[11] Xiao Han, Houxuan Liu, Yunchao Ding, and Lu Yang. Ro-map: Realtime multi-object mapping with neural radiance fields. IEEE Robotics and Automation Letters, 8(9):5950â5957, 2023.

[12] Yifu Tao, Yash Bhalgat, Lanke Frank Tarimo Fu, Matias Mattamala, Nived Chebrolu, and Maurice Fallon. Silvr: Scalable lidar-visual reconstruction with neural radiance fields for robotic inspection. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 17983â17989. IEEE, 2024.

[13] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024.

[14] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18039â18048, 2024.

[15] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21357â21366, 2024.

[16] Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time 3d reconstruction at scale using gaussian splatting. In ACM SIGGRAPH 2024 Conference Papers, pages 1â11, 2024.

[17] Jiaxin Wei and Stefan Leutenegger. Gsfusion: Online rgb-d mapping where gaussian splatting meets tsdf fusion. IEEE Robotics and Automation Letters, 2024.

[18] Yutong Wang, Bin Xu, Wei Fan, and Changle Xiang. Qiso-slam: Object-oriented slam using dual quadrics as landmarks based on instance segmentation. IEEE Robotics and Automation Letters, 8(4):2253â2260, 2023.

[19] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, et al. Yolov10: Real-time end-to-end object detection. Advances in Neural Information Processing Systems, 37:107984â108011, 2025.

[20] Alex Fisher, Ricardo Cannizzaro, Madeleine Cochrane, Chatura Nagahawatte, and Jennifer L Palmer. Colmap: A memory-efficient occupancy grid mapping framework. Robotics and Autonomous Systems, 142:103755, 2021.

[21] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474, 2017.