# HGS-Planner: Hierarchical Planning Framework for Active Scene Reconstruction Using 3D Gaussian Splatting

Zijun Xu1, Rui Jin2, Ke Wu1, Yi Zhao1, Zhiwei Zhang1, Jieru Zhao3, Fei Gao2 Zhongxue Gan1 and Wenchao Ding1â

<!-- image-->  
Fig. 1: In a simulated complex house scene, we implemented our high-fidelity active reconstruction system on a mobile robot equipped with an RGB-D sensor. The colored curves represent the robotâs executed trajectories. We showcase the reconstruction results, which include the entire rendered scene, detailed renderings from three different views, and the variation in information gain at a specific view.

Abstractâ In complex missions such as search and rescue, robots must make intelligent decisions in unknown environments, relying on their ability to perceive and understand their surroundings. High-quality and real-time reconstruction enhances situational awareness and is crucial for intelligent robotics. Traditional methods often struggle with poor scene representation or are too slow for real-time use. Inspired by the efficacy of 3D Gaussian Splatting (3DGS), we propose a hierarchical planning framework for fast and high-fidelity active reconstruction. Our method evaluates completion and quality gain to adaptively guide reconstruction, integrating global and local planning for efficiency. Experiments in simulated and realworld environments show our approach outperforms existing real-time methods.

## I. INTRODUCTION

In tasks such as search and rescue or target finding, which rely on active exploration, robots must preserve as much geometric and texture information from the environment as possible to support effective decision-making [1, 2]. Online active reconstruction plays a crucial role in these missions by enabling robots to construct and update environmental models in real time, allowing them to navigate and adapt more efficiently in complex and unknown environments.

However, conventional active reconstruction [3]â[5] that fuse sensor data across space and time only capture coarse structures and struggle with rich scene details and novel view evaluation. Recently, Neural Radiance Field (NeRF) [6]-based methods [7]â[10] have gained popularity for their high-fidelity scene representation and efficient memory usage. However, NeRFâs inherent volumetric rendering process requires dense sampling of every pixel, resulting in long training times and poor real-time performance [11]. Additionally, its use of implicit neural representations makes it challenging to evaluate reconstruction quality accurately in real time. In fact, active reconstruction systems demand quick responses and the ability to make decisions based on real-time reconstruction quality dynamically. NeRFâs computational bottlenecks make it unsuitable for scene representation in active reconstruction, especially in scenarios that require real-time responses.

Compared to NeRF, 3D Gaussian Splatting (3DGS) [12] offers a more efficient explicit representation, reducing computational complexity and better suiting online active reconstruction [13]. Additionally, the Gaussian mapâs realtime integration of new data gives it the potential to provide immediate feedback on the rendering quality of new views. However, despite these notable advantages, the application of 3DGS for active reconstruction in unknown environments is still largely unexplored.

Though high-quality reconstruction can be achieved using 3DGS, the task with 3D Gaussian representation faces three main challenges. First, efficiently and accurately evaluating novel view quality without ground truth is crucial for guiding robot motion planning, but it remains challenging. Second, while efficiency is critical to active reconstruction, Gaussian maps can only represent occupied areas, posing a challenge for efficiently reconstructing unobserved regions. Third, effectively integrating Gaussian map data into closed-loop motion planning is essential for active reconstruction, yet how to do this effectively is still an open question.

To address the above problems, we propose an efficient 3D Gaussian-based real-time planning framework for active reconstruction. To the best of our knowledge, our framework is the pioneering work exploring 3DGS representation for online active reconstruction. Firstly, we introduce Fisher Information, which represents the expectation of observation information and is independent of ground truth [14], to evaluate novel view quality gain in online reconstruction. Secondly, we improve exploration efficiency in 3D Gaussian representation by integrating unknown voxels into the splatting-based rendering process, allowing us to assess new viewpointsâ coverage of unexplored areas. Thirdly, we use Gaussian map data to adaptively select viewpoints, which balances reconstruction quality and efficiency, and integrate it into an active planning framework. Our experimental results confirm that our framework supports efficient and high-quality online reconstruction.

To summarize, our contributions are:

â¢ To the best of our knowledge, we propose the first online adaptive hierarchical autonomous reconstruction system using 3DGS.

â¢ We design a novel viewpoint selection strategy based on reconstruction coverage and quality and implement it within an autonomous reconstruction framework.

â¢ We conduct extensive simulation and real-scene experiments to validate the effectiveness of the proposed system.

## II. RELATED WORK

## A. High-fidelity Reconstruction Representation

Various scene representations are used for reconstruction, including meshes, planes, and surfel clouds. Recently, Neural

Radiance Field (NeRF) [6] has gained prominence due to its photorealistic rendering capabilities. NeRF methods can be categorized into three types: implicit, hybrid representation, and explicit. Implicit method [15] is memory-efficient but faces challenges such as catastrophic forgetting and significant computational overhead in larger scenes. Hybrid representation methods [16]â[18] integrate the benefits of implicit MLPs with structural features, significantly improving scene scalability and precision. The explicit method introduced in [19] directly embeds map features within voxels, bypassing the use of MLPs, which allows for faster optimization.

Although NeRF excels in photorealistic reconstruction [7], its ray sampling approach leads to high computational costs, making it impractical for real-time autonomous reconstruction [13]. In contrast, 3DGS [12] facilitates real-time rendering of novel views through its fully explicit representation and innovative differential splatting rendering, which has been utilized in real-time SLAM, allowing the scene reconstruction from RGB-D images [11, 20, 21].

## B. Active Reconstruction System

The active reconstruction system integrates data acquisition into the decision-making loop, guiding robots in data collection tasks [13]. Scene representations can categorize these systems: voxel-based methods [4, 5, 22], surface-based methods [22]â[24], neural network-based methods [7, 25] and 3D Gaussian-based methods [13].

Voxel methods [4, 5, 22] use compact grids for efficient space representation, while surface-based methods [22]â[24] focus on geometric details. However, both largely neglect color and texture details. Neural network-based methods, such as NeurAR [7] and Naruto [25], combine NeRF with Bayesian models for view planning but are computationally intensive, causing frequent delays. 3D Gaussian Splatting (3DGS) offers high-fidelity scene representation and fast data fusion, but its application in active reconstruction is still rare. GS-Planner [13] combines 3DGS with voxel maps but lacks effective information gain evaluation and relies on random sampling, reducing efficiency and risking local optima.

## III. METHOD

## A. Problem Statement and System Overview

This study aims to efficiently explore unknown and spatially constrained 3D environments and reconstruct high-quality 3D models using a mobile robot by generating a trajectory composed of a sequence of paths and viewpoints [7]. In previous greedy-based NBV methods [8, 25], the path design seeks to identify the trajectory leading to the next optimal view. However, from a global perspective, this approach always converges on local optima, reducing reconstruction efficiency significantly. We design a hierarchical autonomous reconstruction framework through a novel viewpoint selection criterion, selecting a series of optimal viewpoints for global and local path planning, enabling rapid and high-fidelity reconstruction. As illustrated in Fig. 2, our proposed hierarchical autonomous reconstruction framework consists of two main components. The 3D Gaussian Representation module reconstructs high-fidelity scenes and offers real-time evaluations of potential future viewpoints by leveraging 3DGSâs efficient data fusion and online rendering capabilities. These evaluations encompass gains in both coverage information and reconstruction quality. The Active Reconstruction Planning module is divided into two subcomponents: global planning and local planning. Global planning generates a path that enhances exploration efficiency and avoids local optima, while local planning identifies optimal viewpoints through view sampling and adaptive selection, developing a local path. Finally, the global and local paths are merged into an exploration path that guides the robotâs movement.

<!-- image-->  
Fig. 2: An overview of our efficient autonomous reconstruction system with high-fidelity. Utilizing 3DGS for scene representation, the unobserved areas and the Fisher Information from the GS map are provided in real-time to evaluate the quality and completeness of the online reconstruction. Our proposed active reconstruction planning framework efficiently guides the robot to acquire new scene data, ensuring a comprehensive and high-fidelity 3DGS reconstruction.

## B. 3D Gaussian Representation

We use SplaTam [26], a 3D Gaussian-based SLAM method, for online 3D Gaussian Splatting reconstruction. The scene is represented as numerous isotropic 3D Gaussian, each characterized by eight parameters: center position $\xi \in \mathbb { R } ^ { 3 }$ RGB color $r \in \mathbb { R } ^ { 3 }$ , radius $\mu \in \mathbb { R }$ , and opacity $\rho \in \mathbb { R }$ . The opacity function Ï of a point $\alpha \in \mathbb { R } ^ { 3 }$ , computed from each 3D Gaussian, is defined as follows:

$$
\pi ( \alpha , \rho ) = \rho \exp \left( - \frac { | \alpha - \xi | ^ { 2 } } { 2 \mu ^ { 2 } } \right) .\tag{1}
$$

We adopt a differentiable approach to render the images to optimize the Gaussian parameters for scene representation. The final rendered RGB color $R _ { p i x }$ and depth $D _ { p i x }$ can be mathematically formulated as the alpha blending of N sequentially ordered points that overlap the pixel,

$$
\begin{array} { l } { { \displaystyle R _ { p i x } = \sum _ { i = 1 } ^ { N } r _ { i } \pi _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \pi _ { j } \right) } } \\ { { \displaystyle D _ { p i x } = \sum _ { i = 1 } ^ { N } d _ { i } \pi _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \pi _ { j } \right) . } } \end{array}\tag{2}
$$

where $d _ { i }$ is the depth of the i-th 3D Gaussian center, corresponding to the z-coordinate of its center position in the camera coordinate system.

<!-- image-->  
Fig. 3: A 3D illustration of pixel-level coverage gain evaluation. Given a set of reconstructed Gaussians and a viewpoint, the coverage gain is rendered by weighting unobserved voxel Gaussians with the transmittance of reconstructed Gaussians along the optical ray.

## C. Reconstruction Coverage Gain Evaluation

To improve the efficiency and completeness of scene reconstruction, we implemented an evaluation of reconstruction coverage gain for candidate viewpoints. Calculating the increase in reconstructed areas from a new viewpoint requires considering occupied and unobserved regions. Yet, under the 3D Gaussian representation, we can only recognize the former, making it difficult to determine which regions have yet to be observed. To solve this problem, similar to GS-Planner [13], we maintain a voxel map to represent unobserved volume and integrate it into the splatting rendering. However, unlike GS-Planner, we employ a more streamlined calculation method that leverages uniform voxel volumes to achieve modelconsistent pixel-level reconstruction coverage gain within the 3DGS rendering process. Specifically, given a set of 3D Gaussians and a viewpoint pose, we first sort the Gaussians from front to back by depth. Then, using the ordered 3D Gaussians, we can efficiently render depth images by alphacompositing the splatted 2D projection of each Gaussian sequentially in pixel space. During rendering, by integrating the unobserved voxels from the maintained voxel map into the Gaussian map, we can determine whether an unobserved region exists between adjacent Gaussians. Considering both the uniform volume of each voxel and the inherent opacity attribute of the Gaussians, we can evaluate the visibility gain of unobserved regions for each viewpoint by utilizing a transmittance weight, which can be expressed as:

<!-- image-->  
Fig. 4: The top images are depth maps: the left is the ground truth, and the right is the rendered depth. Below, the left image shows the squared error, and the right illustrates the quality gain.

$$
V _ { p i x } = \sum _ { i = 1 } ^ { n } V \prod _ { j = 1 } ^ { m _ { i } } ( 1 - \alpha _ { j } ) .\tag{3}
$$

where n is the number of unobserved volumes along the ray, mi is the number of the related 3D Gaussians before the i-th unobserved voxel Gaussian, $\textstyle \prod _ { j = 1 } ^ { m _ { i } } { \big ( } 1 - \alpha _ { j } { \big ) }$ is the transmittance weight, V represents the same unobserved voxel volume. Leveraging the fast splatting-based rendering, the Reconstruction Coverage evaluation process runs in parallel with the reconstruction process, resulting in highly efficient overall computation. To illustrate the coverage evaluation process more intuitively, we provide Fig. 3.

## D. Reconstruction Quality Gain Evaluation

To enhance reconstruction quality and accuracy, we employ Fisher Information to quantify the quality gains from novel viewpoints, leveraging its independence from ground truth [14]. The primary goal of neural rendering is to minimize the negative log-likelihood (NLL) between rendered and ground truth images, described by:

$$
- \log \mathbb { P } ( \pmb { \Psi } | \mathbf { x } , \mathbf { w } ) = \left( \pmb { \Psi } - \pmb { f } ( \mathbf { x } , \mathbf { w } ) \right) ^ { T } \left( \pmb { \Psi } - \pmb { f } ( \mathbf { x } , \mathbf { w } ) \right) .\tag{4}
$$

where x is the camera pose, Î¨ the corresponding image, w the model parameters, and $f ( \mathbf { x } , \mathbf { w } )$ the rendering model. Under the regularity conditions [27], Fisher Information for Eq. 4 is defined as the Hessian of the log-likelihood function concerning w:

$$
\mathcal { I } ( \mathbf { w } ) = - \mathbb { E } _ { \mathbb { P } ( \Psi \mid \mathbf { x } , \mathbf { w } ) } \left[ \frac { \partial ^ { 2 } \log \mathbb { P } ( \Psi \mid \mathbf { x } , \mathbf { w } ) } { \partial \mathbf { w } ^ { 2 } } \middle | \mathbf { w } \right] = \mathbf { H } ^ { \prime \prime } [ \Psi \mid \mathbf { x } , \mathbf { w } ] ,\tag{5}
$$

where $\mathbf { H } ^ { \prime \prime } [ \Psi | \mathbf { x } , \mathbf { w } ]$ is the Hessian of Eq. 4. In the evaluation process, we can obtain the initial estimation of parameters $\mathbf { w } ^ { * }$ by using $\{ \Psi _ { i } ^ { a c q } \}$ as the training set $D ^ { t r a i n }$ . Our quality evaluation purpose is to identify the viewpoints that can maximize the Information Gain [28]â[30] among the viewpoints $\mathbf { x } _ { i } ^ { a c q } { \mathbf { \Psi } } \in D ^ { c a n d i d a t e }$ in comparison to $D ^ { t r a i n }$ , where $D ^ { c a n d i d \bar { a } t e }$ represents the collection of candidate viewpoints:

$$
\begin{array} { r l } & { \mathcal { T } [ \mathbf { w } ^ { * } ; \{ \Psi _ { i } ^ { a c q } \} | \{ \mathbf { x } _ { i } ^ { a c q } \} , D ^ { t r a i n } ] } \\ & { \ = H [ \mathbf { w } ^ { * } | D ^ { t r a i n } ] - H [ \mathbf { w } ^ { * } | \{ \Psi _ { i } ^ { a c q } \} , \{ \mathbf { x } _ { i } ^ { a c q } \} , D ^ { t r a i n } ] , } \end{array}\tag{6}
$$

where $H [ \cdot ]$ is the entropy [29]. Considering the log-likelihood form in Eq. 4, specifically the rendering loss, the entropy difference in the R.H.S. of Eq. 6 only depends on $H [ \mathbf { \bar { w } } ^ { * } | \{ \Psi _ { i } ^ { a c q } \} , \{ \mathbf { x } _ { i } ^ { a c q } \} , D ^ { t r a i n } ]$ , then the Hessian can be approximated using just the Jacobian matrix of $f ( \mathbf { x } , \mathbf { w } )$ [14]:

$$
\mathbf { H } ^ { \prime \prime } [ \Psi | \mathbf { x } , \mathbf { w } ^ { * } ] = \nabla _ { \mathbf { w } } f ( \mathbf { x } ; \mathbf { w } ^ { * } ) ^ { T } \nabla _ { \mathbf { w } } f ( \mathbf { x } ; \mathbf { w } ^ { * } ) .\tag{7}
$$

As expected, the trace of Eq. 7 can be computed without ground truths $\{ \Psi _ { i } ^ { a c q } \}$ , as Fisher Information is independent of observations. Furthermore, with the Laplace approximation [31, 32], Eq. 7 can be approximated by considering only diagonal elements and adding a log-prior regularizer Î»I:

$$
\mathbf { H } ^ { \prime \prime } [ \Phi | \mathbf { x } , \mathbf { w } ^ { * } ] \approx \mathrm { d i a g } ( \nabla _ { \mathbf { w } } f ( \mathbf { x } , \mathbf { w } ^ { * } ) ^ { T } \nabla _ { \mathbf { w } } f ( \mathbf { x } , \mathbf { w } ^ { * } ) ) + \lambda I .\tag{8}
$$

Like coverage reconstruction, we integrate quality evaluation into splatting-based rendering for computational efficiency.

## E. Adaptive Hierarchical Planning

To avoid local optima in the exploration path, inspired by TARE [5], we propose an adaptive hierarchical planning framework, which combines global planning with adaptive local planning to improve the efficiency of scene reconstruction. The entire scene is divided into two regions: the local space C for local planning and the space outside C which is partitioned into evenly cuboid subspaces for global planning.

1) Global planning: Each cuboid subspace is classified into three states based on the voxel map mentioned in Sec. III-C: âreconstructedâ (only observed voxels), âreconstructingâ (both observed and unobserved voxels), and âunreconstructedâ (only unobserved voxels). In global planning, only âreconstructingâ subspaces are taken into account. The global is to find a global path $\Gamma _ { \mathrm { g l o b a l } }$ that traverses all âreconstructing subspaces, connecting their centers and the robotâs current location. To achieve this, similar to [5], we construct a sparse random roadmap in the traversable space expanded from the past trajectory. Then we apply $\mathbf { A } ^ { * }$ search on the roadmap to find the shortest paths among the subspaces and the current pose followed by solving a Traveling Salesman Problem (TSP) [33] to get $\Gamma _ { \mathrm { g l o b a l } }$

2) Adaptive local planning: Due to the trade-off between efficiency and efficacy in reconstruction, we design adaptive local planning to adjust the weights of these two aspects dynamically. Similar to the global planning approach, we use the $\mathbf { A } ^ { * }$ algorithm combined with a TSP solver to perform local path planning after selecting the best views. The whole best views selecting algorithm is listed as Alg. 1.

Specifically, we first calculate the intersection points between the global path and the local horizon and uniformly sample viewpoints within the local region (Lines 1-2). Then, we combine these intersections and sampled points and assess a comprehensive 360-degree information gain for each (Line 3-4). This information gain comprises two components: coverage gain and quality gain, which are weighted relative to the proportion of observed areas within the local region:

$$
G = \operatorname { G } ( C ) + \lambda _ { o } \operatorname { G } ( Q )\tag{9}
$$

where G is the final information gain, G(C) is the coverage gain, G(Q) is the quality gain, $\lambda _ { o }$ is the proportion of observed voxels within the local region. Subsequently, leveraging the 360-degree information gain, we select viewpoints that exceed a threshold of information gain and use a slidingwindow technique to identify the optimal yaw angles(Lines 5-12). Finally, we obtain the exploration path by connecting the global and local paths. The reconstruction is complete when all cuboid subspaces are âreconstructedâ and no more viewpoints are selected.

Algorithm 1 Adaptive local views selection   
Require: Global Path $\Gamma _ { \mathrm { g l o b a l } }$ , Local Horizon L, Current Pose   
$\mathbf { P } _ { C }$ , Gaussian Map $\mathbf { M } _ { G }$ , Voxel map $\mathbf { M } _ { V }$   
1: Intersection Points $\mathbf { P } _ { I } \gets$ CalIntersection $( \Gamma _ { \mathrm { g l o b a l } } , \mathcal { L } )$   
2: Sampling Points $\mathbf { P } _ { S } \gets$ SamplingViewpoints $( \mathbf { P } _ { C } )$   
3: $\mathbf { P } _ { A L L } = \mathbf { P } _ { S } \cup \mathbf { P } _ { I }$   
4: Gain $\mathbf { G } _ { A L L }  \mathbf { A }$ daptiveEvaluation $\left( \mathbf { P } _ { A L L } \right)$   
5: for $( g _ { i } , p _ { i } ) \in ( \mathbf { G } _ { A L L } , \mathbf { P } _ { A L L } )$ do   
6: if $g _ { i } < g _ { \mathrm { t h r e s } }$ and $p _ { i } \in \mathbf { P } _ { S }$ then   
7: $\mathbf { G } _ { A L L }  \mathbf { G } _ { A L L } \setminus g _ { i }$   
8: $\mathbf { P } _ { A L L }  \mathbf { P } _ { A L L } \setminus p _ { i }$   
9: continue   
10: end if   
11: end for   
12: Best Views $\mathbf { V } _ { b }  \mathrm { S e l e c t B e s t Y a w s } ( \mathbf { G } _ { A L L } , \mathbf { P } _ { A L L } )$   
13: Result local views: $\mathbf { V } _ { b }$   
14: Return $\mathbf { V } _ { b }$

## IV. RESULTS

## A. Implementation details

We run our active reconstruction system on a desktop PC with a 2.9 GHz Intel i7-10700 CPU and an NVIDIA RTX 3090 GPU, using the Autonomous Exploration Development Environment [34] for simulation. The systemâs car is equipped with an RGB-D sensor and a Lidar VLP-16, providing realtime RGB-D images at 1200 Ã680 resolution with a 5-meter range, and uses LOAM [35] for localization. The maximum velocity limit is $1 . 0 ~ \mathrm { ~ m ~ } / \mathrm { s } ,$ and the depth data includes a uniform noise of 2 cm.

We validate our method through simulations in three complex Matterport3D (MP3D) [36] scenes: 17DRP, 2t7WU, and Gdvg with the local planner range of $6 \mathrm { ~ m ~ } \times \mathrm { ~ 6 ~ }$ m and the resolution of voxel map integrated into Gaussian map to 0.1 m. Viewpoints are sampled with a minimum distance of 1.5 m to avoid excessive overlap.

Similar to [8] and [7], we evaluate our method in terms of effectiveness and efficiency. We adopt scene quality metrics from NARUTO [25]: Accuracy (cm), Completion (cm), and Completion Ratio (the percentage of points in the reconstructed mesh with Completion under 5 cm). We extract geometric centroids from Gaussian spheres to simulate mesh vertices due to the absence of a standard method for converting 3DGS into mesh. In these metrics, about 300k points are sampled from the surfaces. For efficiency, similar to [9], we evaluate each step planning time (second) $T _ { P }$ and the path length (meter) P.L.. For each planning cycle during the reconstruction, $T _ { P }$ is divided into viewpoints sampling and evaluation time $T _ { V E } ,$ local path planning time $T _ { L P }$ and global planning time $T _ { G P }$ , with average $T _ { G P }$ times of approximately 0.017 s (17DRP), 0.018 s (2t7WU) and 0.0 15s (Gdvg). i.e $. T _ { P } = T _ { V E } + T _ { L P } + T _ { G P }$ . In TARE, the time taken to evaluate viewpoints corresponds to the time required to update the information about the areas they cover.

<!-- image-->  
Fig. 5: Trajectories and the reconstruction results from the top view. Left: Ours, Right: GS-Planner

## B. Efficacy of the Method

Following [9], we evaluate our methodâs efficacy based on its validity and efficiency. We create variants of our method with 3D Gaussian representation: V1 (TARE [5]), V2 (Coverage evaluation only), V3 (Quality evaluation only), and V4 (On both without an adaptive strategy). Our method proves highly effective and more efficient than other approaches.

1) Quality evaluation in real-time reconstruction: Fig. 4 shows that quality evaluation closely matches actual losses, even without ground truth. Highlighted areas on the loss map indicate regions with lower reconstruction quality, aligning with our quality gain evaluation.

2) Novel view evaluation criterion: We make V1 as a baseline applying hierarchical planning, V2 (for coverage), V3 (for quality), and V4 (on both) to verify our evaluation criterionâs efficacy. Metrics in Table I show that evaluating coverage and quality improves reconstruction. However, V2 results in low-quality reconstruction as it overlooks complex details, while V3 yields poor completeness by only refining already-covered areas and neglecting unobserved regions.

3) Adaptive hierarchical planning: To validate the adaptive hierarchical planning, we establish V4 as our baseline. Combining these two tasks noticeably hampers the speed of scene exploration and may result in local optima, especially when dealing with intricate reconstruction details. The introduction of adaptive hierarchical planning (Ours) ensures efficient exploration while maintaining reconstruction quality, preventing the process from getting stuck in local optima.

## C. Comparison with existing reconstruction methods

We benchmark two recent works: NARUTO [25] based on view information gain fields, and GS-Planner [13] using 3D Gaussian reconstruction. The metrics in Table II show our framework outperforms both planning efficiency and reconstruction quality. NARUTO neglects uncovered areas reducing the exploration efficiency, while GS-Planner often gets stuck in local optima during exploration. Fig. 6 and metrics in Table II highlight our methodâs superior reconstruction. We refer readers to the supplementary video for more visual results and the reconstruction process. We implemented the GS-Planner algorithm on a mobile vehicle. Fig. 5 compares the trajectories of our method and GS-Planner in scene 2t7WU, showing GS-Plannerâs focus on smaller areas reduces overall efficiency. Our framework achieves more efficient scene reconstruction.

TABLE I: Evaluations of the effectiveness and efficiency with 3DGS representation.
<table><tr><td></td><td colspan="2">Variant</td><td colspan="2"></td><td colspan="2">Scene_17DRP</td><td colspan="2">Scene_2t7WU</td><td colspan="3"></td></tr><tr><td>Method</td><td> $C _ { o v e . } Q _ { u a . } H _ { i e r . } \mathrm { A d a p }$ </td><td></td><td></td><td>Accâ (cm)</td><td>Compâ (cm)</td><td>C.R.â</td><td>Accâ (cm)</td><td>Compâ (cm)</td><td>C.R.â Accâ (cm)</td><td>Scene_Gdvg Compâ (cm)</td><td>C.R.â</td></tr><tr><td>V1(TARE)</td><td></td><td>â</td><td></td><td>2.86</td><td>57.23</td><td>0.47</td><td>3.61</td><td>28.40</td><td>0.65</td><td>3.03</td><td>15.51</td></tr><tr><td>V2(Coverage)</td><td>â</td><td>â</td><td></td><td>2.84</td><td>7.16</td><td>0.81</td><td>3.21</td><td>10.07</td><td>0.79 3.01</td><td>12.97</td><td>0.71 0.79</td></tr><tr><td>v(Quality)</td><td></td><td>â</td><td></td><td>2.82</td><td>54.67</td><td>0.49</td><td>3.19</td><td>22.13</td><td>0.69 2.94</td><td>11.27</td><td>0.75</td></tr><tr><td>V4(On both)</td><td>â</td><td>â</td><td></td><td>2.81</td><td>6.52</td><td>0.85</td><td>3.12</td><td>9.63</td><td>0.82 2.98</td><td>9.85</td><td>0.85</td></tr><tr><td>V5(Ours full)</td><td>â</td><td>â</td><td>â</td><td>2.80</td><td>2.66</td><td>0.90</td><td>3.09</td><td>2.63</td><td>0.91 1.97</td><td>2.63</td><td>0.90</td></tr><tr><td>Variant</td><td>Cov.  $Q _ { u a }$ </td><td></td><td>Hier. Adap</td><td>TV E (s)</td><td>TP (s)</td><td>P.L. (m)</td><td>TV E (s)</td><td>TP (s)</td><td>P.L. (m)  $T _ { V E } ~ ( \mathrm { s } )$ </td><td>TP (s)</td><td>P.L. (m)</td></tr><tr><td>V1(TARE)</td><td></td><td>â</td><td></td><td>0.075</td><td>0.098</td><td>57.71</td><td>0.091</td><td>0.115</td><td>25.97 0.096</td><td>0.119</td><td>26.15</td></tr><tr><td>V2(Coverage</td><td>â</td><td>â</td><td></td><td>0.054</td><td>0.077</td><td>83.27</td><td>0.065</td><td>0.088</td><td>48.76 0.059</td><td>0.081</td><td>27.33</td></tr><tr><td>(Quality)</td><td></td><td>â</td><td></td><td>0.052</td><td>0.075</td><td>60.23</td><td>0.057</td><td>0.082</td><td>30.05 0.049</td><td>0.069</td><td>26.46</td></tr><tr><td>V(On both)</td><td>â</td><td>â</td><td></td><td>0.103</td><td>0.126</td><td>98.09</td><td>0.118</td><td>0.144</td><td>58.62 0.116 0.122</td><td>0.137</td><td>28.17</td></tr><tr><td>V5(Ours full)</td><td>â</td><td>â</td><td>â</td><td>0.105</td><td>0.129</td><td>90.35</td><td>0.126</td><td>0.151</td><td>60.52</td><td>0.142</td><td>30.15</td></tr></table>

TABLE II: Evaluations of the effectiveness and efficiency with existing planning methods.
<table><tr><td></td><td colspan="4">Scene_17DRP c(cmC.R.TP ( PL. )cm(C.R.</td><td colspan="8">Scene_2t7WU</td><td colspan="4">Scene_Gdvg</td></tr><tr><td>Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>TP (s) P.L. (m)|Accâ(cm) Compâ (cm) C.R.â Tp (s) P.L. (m)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GS-Planner</td><td>2.88</td><td>4.06</td><td>0.84</td><td>0.147</td><td>95.04</td><td>3.19</td><td>2.85</td><td>0.88</td><td>0.179</td><td>74.98</td><td>3.09</td><td></td><td>4.76</td><td>0.89</td><td>0.161</td><td>30.78</td></tr><tr><td>NARUTO</td><td>10.29</td><td>2.87</td><td>0.89</td><td>0.368</td><td>120.66</td><td>23.94</td><td>7.38</td><td>0.69</td><td>0.349</td><td>108.01</td><td>6.97</td><td></td><td>4.61</td><td>0.91</td><td>0.315</td><td>90.33</td></tr><tr><td>Ours</td><td>2.80</td><td>2.66</td><td>0.90</td><td>0.129</td><td>90.35</td><td>3.09</td><td>2.63</td><td>0.91</td><td>0.151</td><td>60.52</td><td>1.97</td><td></td><td>2.63</td><td>0.90</td><td>0.142</td><td>30.15</td></tr></table>

Ground Truth

GS-Planner  
<!-- image-->  
Fig. 6: Comparison of the reconstruction scenes with different methods

## D. Robot experiments in real scene

We implemented our proposed framework on an UGV equipped with Realsense Depth Camera D435i and Ouster Lidar to perform the real scene reconstruction. FAST-LIO [37] provides the localization. Since we use an Ackermannsteering vehicle, we replace the A\* algorithm with Kino-A\* to ensure the path meets kinematic constraints. The detailed process will be shown in the supplementary video.

## V. CONCLUSIONS

In this paper, we developed a hierarchical planning framework for efficient and high-fidelity active reconstruction with 3DGS. We introduced Fisher Information to evaluate reconstruction quality and assessed coverage gain by integrating the Voxel and Gaussian maps. We also designed a novel viewpoint selection strategy within hierarchical planning. Extensive experiments show our methodâs superior performance. For future work, we aim to extend our research to swarm robotics in large-scale scenes.

[1] Ross D Arnold, Hiroyuki Yamaguchi, and Toshiyuki Tanaka. Search and rescue with autonomous flying robots through behavior-based cooperative intelligence. Journal of International Humanitarian Action, 3(1):1â18, 2018.

[2] Mingyang Lyu, Yibo Zhao, Chao Huang, and Hailong Huang. Unmanned aerial vehicles for search and rescue: A survey. Remote Sensing, 15(13), 2023.

[3] Stefan Isler, Reza Sabzevari, Jeffrey Delmerico, and Davide Scaramuzza. An information gain formulation for active volumetric 3d reconstruction. In IEEE International Conference on Robotics and Automation (ICRA), pages 3477â3484, 2016.

[4] Boyu Zhou, Yichen Zhang, Xinyi Chen, and Shaojie Shen. Fuel: Fast uav exploration using incremental frontier structure and hierarchical planning. IEEE Robotics and Automation Letters, 6(2):779â786, 2021.

[5] Chao Cao, Hongbiao Zhu, Fan Yang, Yukun Xia, Howie Choset, Jean Oh, and Ji Zhang. Autonomous exploration development environment and the planning algorithms. In 2022 International Conference on Robotics and Automation (ICRA), pages 8921â8928, 2022.

[6] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

[7] Yunlong Ran, Jing Zeng, Shibo He, Jiming Chen, Lincheng Li, Yingfeng Chen, Gimhee Lee, and Qi Ye. Neurar: Neural uncertainty for autonomous 3d reconstruction with implicit neural representations. IEEE Robotics and Automation Letters, 8(2):1125â1132, 2023.

[8] Jing Zeng, Yanxu Li, Yunlong Ran, Shuo Li, Fei Gao, Lincheng Li, Shibo He, Jiming Chen, and Qi Ye. Efficient view path planning for autonomous implicit reconstruction. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 4063â4069. IEEE, 2023.

[9] Jing Zeng, Yanxu Li, Jiahao Sun, Qi Ye, Yunlong Ran, and Jiming Chen. Autonomous implicit indoor scene reconstruction with frontier exploration. arXiv preprint arXiv:2404.10218, 2024.

[10] Ke Wu, Kaizhao Zhang, Mingzhe Gao, Jieru Zhao, Zhongxue Gan, and Wenchao Ding. Swift-mapping: Online neural implicit dense mapping in urban scenes. Proceedings of the AAAI Conference on Artificial Intelligence, 38(6):6048â6056, Mar. 2024.

[11] Ke Wu, Kaizhao Zhang, Zhiwei Zhang, Shanshuai Yuan, Muer Tie, Julong Wei, Zijun Xu, Jieru Zhao, Zhongxue Gan, and Wenchao Ding. Hgs-mapping: Online dense mapping using hybrid gaussian representation in urban scenes, 2024.

[12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Â¨ Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 2023.

[13] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu, and Fei Gao. Gsplanner: A gaussian-splatting-based planning framework for active high-fidelity reconstruction, 2024.

[14] Wen Jiang, Boshu Lei, and Kostas Daniilidis. Fisherrf: Active view selection and uncertainty quantification for radiance fields using fisher information. arXiv, 2023.

[15] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6229â6238, 2021.

[16] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12786â12796, 2022.

[17] Muer Tie, Julong Wei, Zhengjun Wang, Ke Wu, Shansuai Yuan, Kaizhao Zhang, Jie Jia, Jieru Zhao, Zhongxue Gan, and Wenchao Ding. O2v-mapping: Online open-vocabulary mapping with neural implicit representation. arXiv preprint arXiv:2404.06836, 2024.

[18] Chenxing Jiang, Hanwen Zhang, Peize Liu, Zehuan Yu, Hui Cheng, Boyu Zhou, and Shaojie Shen. H2-mapping: Real-time dense mapping using hierarchical hybrid representation. IEEE Robotics and Automation Letters, 8(10):6787â6794, 2023.

[19] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5501â5510, 2022.

[20] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[21] Chi Yan, Delin Qu, Dong Wang, Dan Xu, Zhigang Wang, Bin Zhao, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting. arXiv preprint arXiv:2311.11700, 2023.

[22] Lukas Schmid, Michael Pantic, Raghav Khanna, Lionel Ott, Roland Siegwart, and Juan Nieto. An efficient sampling-based method for online informative path planning in unknown environments. IEEE Robotics and Automation Letters, 5(2):1500â1507, 2020.

[23] Rui Huang, Danping Zou, Richard Vaughan, and Ping Tan. Active image-based modeling with a toy drone. In IEEE International Conference on Robotics and Automation (ICRA), pages 6124â6131, 2018.

[24] Yuman Gao, Yingjian Wang, Xingguang Zhong, Tiankai Yang, Mingyang Wang, Zhixiong Xu, Yongchao Wang, Yi Lin, Chao Xu, and Fei Gao. Meeting-merging-mission: A multi-robot coordinate framework for large-scale communication-limited exploration. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 13700â13707, 2022.

[25] Ziyue Feng, Huangying Zhan, Zheng Chen, Qingan Yan, Xiangyu Xu, Changjiang Cai, Bing Li, Qilun Zhu, and Yi Xu. Naruto: Neural active reconstruction from uncertain target observations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21572â21583, June 2024.

[26] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[27] M.J. Schervish. Theory of Statistics. Springer Series in Statistics. Springer New York, 2012.

[28] D. V. Lindley. On a Measure of the Information Provided by an Experiment. The Annals of Mathematical Statistics, 27(4):986 â 1005, 1956.

[29] Andreas Kirsch and Yarin Gal. Unifying approaches in active learning and active sampling via fisher information and information-theoretic quantities. Transactions on Machine Learning Research, 2022. Expert Certification.

[30] Neil Houlsby, Ferenc Huszar, Zoubin Ghahramani, and MatÂ´ e Lengyel. Â´ Bayesian active learning for classification and preference learning. CoRR, abs/1112.5745, 2011.

[31] David J. C. MacKay. Bayesian Interpolation. Neural Computation, 4(3):415â447, 05 1992.

[32] Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace reduxâeffortless Bayesian deep learning. In NeurIPS, 2021.

[33] Christos H. Papadimitriou. The complexity of the linâkernighan heuristic for the traveling salesman problem. SIAM Journal on Computing, 21(3):450â465, 1992.

[34] Chao Cao, Hongbiao Zhu, Fan Yang, Yukun Xia, Howie Choset, Jean Oh, and Ji Zhang. Autonomous exploration development environment and the planning algorithms. CoRR, abs/2110.14573, 2021.

[35] Ji Zhang and Sanjiv Singh. Loam: Lidar odometry and mapping in real-time. In Robotics: Science and Systems, 2014.

[36] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D data in indoor environments. International Conference on 3D Vision (3DV), 2017.

[37] Wei Xu and Fu Zhang. FAST-LIO: A fast, robust lidar-inertial odometry package by tightly-coupled iterated kalman filter. CoRR, abs/2010.08196, 2020.