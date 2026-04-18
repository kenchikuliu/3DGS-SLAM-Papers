# HGSLoc: 3DGS-based Heuristic Camera Pose Refinement

Zhongyan Niu1, Zhen Tan1, Jinpu Zhang1, Xueliang Yang1, Dewen Huâ1

Abstractâ Visual localization refers to the process of determining camera poses and orientation within a known scene representation. This task is often complicated by factors such as changes in illumination and variations in viewing angles. In this paper, we propose HGSLoc, a novel lightweight plugand-play pose optimization framework, which integrates 3D reconstruction with a heuristic refinement strategy to achieve higher pose estimation accuracy. Specifically, we introduce an explicit geometric map for 3D representation and high-fidelity rendering, allowing the generation of high-quality synthesized views to support accurate visual localization. Our method demonstrates higher localization accuracy compared to NeRFbased neural rendering localization approaches. We introduce a heuristic refinement strategy, its efficient optimization capability can quickly locate the target node, while we set the steplevel optimization step to enhance the pose accuracy in the scenarios with small errors. With carefully designed heuristic functions, it offers efficient optimization capabilities, enabling rapid error reduction in rough localization estimations. Our method mitigates the dependence on complex neural network models while demonstrating improved robustness against noise and higher localization accuracy in challenging environments, as compared to neural network joint optimization strategies. The optimization framework proposed in this paper introduces novel approaches to visual localization by integrating the advantages of 3D reconstruction and the heuristic refinement strategy, which demonstrates strong performance across multiple benchmark datasets, including 7Scenes and Deep Blending dataset. The implementation of our method has been released at https://github.com/anchang699/HGSLoc.

## I. INTRODUCTION

Visual localization is a research direction that aims to determine the pose and orientation of a camera within a known scene by analyzing and processing image data. This technique has significant applications in various fields, such as augmented reality (AR), robotic navigation, and autonomous driving. By enabling devices to accurately identify their spatial location in complex 3D environments, visual localization facilitates autonomous navigation, environmental awareness, and real-time interaction. The core objective of visual localization is to estimate the cameraâs absolute pose. However, this task is challenging due to factors such as illumination changes, dynamic occlusions, and variations in viewing angles, necessitating the development of robust and efficient algorithms to address these complexities.

Two major categories of visual location methods are Absolute Pose Regression (APR) [1]â[8] and Scene Coordinate Regression (SCR) [9]â[11]. APR is an end-to-end deep learning approach that directly regresses the cameraâs pose from the input image. The key advantages of APR lie in its simplicity and computational efficiency. However, APR exhibits notable limitations, particularly in complex or previously unseen environments, where its generalization capability is weak [12]. In contrast, SCR adopts an indirect strategy for pose estimation. It first predicts the 3D scene coordinates of each image pixel using a deep learning model, followed by the computation of the camera pose through the spatial transformation of these coordinates. Although SCR demonstrates high accuracy and robustness in familiar scenes, it incurs substantial computational costs due to the need to predict a large number of pixel-wise coordinates.

<!-- image-->  
Fig. 1. HGSLoc significantly reduces the error between the coarse pose and the GT, and exhibits strong noise resistance.

In this paper, we propose a novel visual localization paradigm that enhances pose estimation accuracy through the integration of 3D reconstruction. Neural Radiance Field (NeRF) [13] is capable of synthesizing and rendering highquality 3D scene images through neural network training. Some existing NeRF-based visual localization methods [14], [15] have gained widespread recognition. However, NeRFâs pixel-wise training and inference mechanism results in significant computational costs, limiting its practical applications. In contrast, 3D Gaussian Splatting (3DGS) [16] represents scene points as Gaussian distributions, significantly reducing computational overhead and leveraging CUDA acceleration for efficient training and inference. In known or partially known static environments, several approaches, such as 3DGS-ReLoc [17] and GSLoc [18], have been developed. The 3DGS-ReLoc method relies on grid search with the normalized cross-correlation (NCC) [19], which affects the localization accuracy. The GSLoc method requires several rounds of iterations to achieve the desired result in the case of poor initial pose estimation, and each iteration utilizes MASt3R [20] for assisted localization. Whereas, our method is a lightweight framework that enables efficient positional optimization for any image. As shown in Fig. 1, by incorporating 3DGS, richer geometric information is available for pose estimation, and through heuristic optimization of coarse pose estimates, the accuracy of localization can be significantly enhanced in complex scenes.

Absolute Pose Regression (APR) and Scene Coordinate Regression (SCR) provide coarse pose estimates for further refinement. To enhance scene rendering, we introduce 3D Gaussian Splatting (3DGS), which constructs a dense point cloud for high-fidelity reconstruction. Building on this, we employ a heuristic refinement algorithm [21] that efficiently adjusts the rendered view to better align with the query image, improving pose accuracy. This modular approach reduces reliance on computationally expensive neural network training, offering a more efficient alternative to deep learning-based pose optimization. Our method demonstrates strong generalization capabilities, maintaining high accuracy even in the presence of noisy pose data, making it adaptable across various platforms. Experimental results on benchmark datasets, including 7Scenes and Deep Blending, validate its effectiveness in visual localization tasks. The contributions of our approach are summarized as follows:

â¢ We propose a lightweight, plug-and-play pose optimization framework that facilitates efficient pose refinement for any query image.

â¢ We design a heuristic refinement strategy and set the step-level optimization step to adapt various complex scenes.

â¢ Our proposed framework achieves higher localization accuracy than NeRF-based neural rendering localization approach [22] and outperforms neural network joint pose optimization strategy in noisy conditions.

## II. RELATED WORK

In this section, we introduce visual localization methods and 3D Gaussian Splatting.

## A. Visual localization

PoseNet pioneered Absolute Pose Regression (APR) [1]â [8] by employing a convolutional neural network (CNN) to directly regress camera pose from images, bypassing traditional feature extraction and geometric computations. This end-to-end approach simplifies visual localization across diverse environments. MS-Transformer [7] improves APR by incorporating global context modeling and multi-head selfattention, enhancing scene understanding and pose accuracy. DFNet [6] further extends APR by integrating multimodal sensor data, increasing robustness. However, APR methods remain sensitive to noise and environmental variability, with accuracy degrading under poor lighting, adverse weather, or occlusions.

Scene Coordinate Regression (SCR) [9]â[11] estimates camera pose by mapping image pixels to 3D scene coordinates, eliminating the need for complex feature matching and improving efficiency. DSAC\* [9] enhances SCR with a differentiable hypothesis selection mechanism and supports both RGB and RGB-D inputs, leveraging depth information for improved scene interpretation. ACE [10] optimizes image coordinate encoding and decoding to accelerate feature matching while enhancing robustness to noise and lighting variations, ensuring reliable pose estimation in challenging environments.

## B. 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) [16], an emerging method in 3D reconstruction, has rapidly gained prominence since its introduction. This method significantly accelerates the synthesis of new views by modeling the scene with Gaussian ellipsoids and utilizing advanced rendering methods. Within the realm of 3DGS research, various techniques have enhanced and optimized 3DGS in different aspects, such as quality improvement [23], compression and regularization [24], navigation [25], dynamic 3D reconstruction [26], and handling challenging inputs [27]. The advancement of 3DGS methods not only enhances the quality of scene reconstruction but also speeds up rendering, offering novel and improved approaches for visual localization tasks. For instance, GSLoc [18] leverages rendered images from new viewpoints for matching and pose optimization, while the InstantSplat [28] method, utilizing DUSt3R [29], achieves rapid and high-quality scene reconstruction by jointly optimizing poses with 3D Gaussian parameters. Our proposed method builds upon 3DGS reconstructed scenes and employs heuristic pose optimization to enhance pose accuracy in specific scenarios while preserving the original pose accuracy.

## III. METHOD

In this section, we outline the fundamental principles of the 3D Gaussian Splatting (3DGS) and heuristic refinement strategy, along with their integrated implementation. An overview of our framework is depicted in Fig. 2.

## A. Explicit Geometric Map

3D Gaussian Splatting (3DGS) [16] is a method for representing and rendering three-dimensional scenes. It models the distribution of objects within a scene using 3D Gaussian functions and approximates object surface colors through spherical harmonic coefficients. This method not only delivers an accurate depiction of scene geometry but also effectively captures and renders the lighting and color variations. In 3DGS, each primitive is characterized by a three-dimensional covariance matrix $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ and mean value $\mu _ { i } \in \mathbb { R } ^ { 3 }$ :

$$
g _ { i } ( \mathbf { x } ) = e ^ { - \frac { 1 } { 2 } ( \mathbf { x } - \boldsymbol { \mu } _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( \mathbf { x } - \boldsymbol { \mu } _ { i } ) }\tag{1}
$$

where $\boldsymbol { \Sigma } = \mathbf { R } \mathbf { S } \mathbf { S } ^ { \top } \mathbf { R } ^ { \top } , \mathbf { R } \in \mathbb { R } ^ { 3 \times 3 }$ represents the rotation, $\mathbf { S } \in$ $\mathbb { R } ^ { 3 }$ represents the anisotropy scale.

When projecting onto the viewing plane, 3D Gaussian Splatting (3DGS) utilizes a 2D Gaussian directly, rather than performing the axial integral of a 3D Gaussian. This approach addresses the computational challenge of requiring a large number of samples by limiting the computation to the number of Gaussians, thereby enhancing efficiency. The projected 2D covariance matrix and means are $\Sigma ^ { \prime } = \bar { \bf J } \bf W \Sigma W ^ { \bar { T } } J ^ { T }$ and $\mu ^ { \prime } = \mathbf { J } \mathbf { W } \mu$ , respectively, where W represents the transformation from the world coordinate system to the camera coordinate system and J denotes the radial approximation of the Jacobian matrix for the projection transformation.

<!-- image-->  
Fig. 2. Overview of HGSLoc. Coarse pose estimates are generated by a pre-trained pose estimator, while high-quality reconstructed scenes are obtained through Gaussian densification. The rendered image of the coarse pose in the scene differs significantly from the query image. After applying the heuristic optimization algorithm, the rendered image aligns much more closely with the query image, resulting in a more accurate pose estimate.

During the rendering phase, spatial depth and tile ID are utilized as key values to sort the Gaussian primitives using GPU-based ordering. Subsequently, the color of each pixel is computed based on the volume rendering formula:

$$
C = \sum _ { i \in \mathcal { N } } \mathbf { c } _ { i } p _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - p _ { i } \alpha _ { j } )\tag{2}
$$

Where:

$$
p _ { i } = e ^ { - \frac { 1 } { 2 } ( x - \mu _ { i } ^ { \prime } ) ^ { T } \Sigma ^ { \prime - 1 } ( x - \mu _ { i } ^ { \prime } ) }\tag{3}
$$

$$
\alpha _ { 2 d } = 1 - \exp \left( - \frac { \alpha _ { 3 d } } { \sqrt { \operatorname * { d e t } ( \Sigma _ { 3 d } ) } } \right)\tag{4}
$$

A major advantage of 3D Gaussian Splatting (3DGS) is its efficient rendering speed. By leveraging CUDA kernel functions for pixel-level parallel processing, 3DGS achieves rapid training and rendering. Additionally, 3DGS employs adaptive control strategies to accommodate objects of various shapes, enhancing both the accuracy and efficiency of rendering. This results in high-quality reconstructed scenes and more realistic new-view images, which provide opportunities for further advancements in pose accuracy.

## B. Heuristic Algorithm Implementation

Heuristic approaches [21] are often implemented to path planning and graph search, combining the strengths of depthfirst search (DFS) and breadth-first search (BFS). It has been widely applied to various real-world problems, including game development, robotic navigation, and geographic information systems (GIS). The primary goal of the heuristic algorithm is to efficiently find the optimal path from an initial node to a goal node, where each node represents a state within the search space. The algorithm relies on an evaluation function, $f ( n )$ , to prioritize nodes for expansion. This function typically consists of two components:

$$
f ( n ) = g ( n ) + h ( n )\tag{5}
$$

Where $g ( n )$ function is the actual cost from the start node to the current node; $h ( n )$ function is the estimated cost from the current node to the target node.

The core idea of the heuristic algorithm is to minimize the number of expanded nodes by guiding the search direction using a heuristic function, $h ( n )$ , while ensuring the least costly path. The heuristic function must satisfy two important properties: Admissibility and Consistency. Admissibility ensures that $h ( n )$ never overestimates the cost of traveling from node n to the target node. Consistency requires that for any node n and its neighboring node $n ^ { \prime }$ , the heuristic function satisfies the following condition:

$$
h ( n ) \leq g ( n , n ^ { \prime } ) + h ( n ^ { \prime } )\tag{6}
$$

Where $g ( n , n ^ { \prime } )$ denotes the actual cost from n to $n ^ { \prime } ,$ which ensures that the algorithm does not repeatedly return to an already expanded node. The algorithm has Optimality and Completeness, i.e., it is guaranteed to find the most optimal path from the start node to the goal node, and for a finite search space, the algorithm always finds a solution.

We use 3DGS as a new-viewpoint image renderer with the goal of finding a more suitable pose within a certain range around the initial pose. A pose is characterized by $( q _ { w } , q _ { x } , q _ { y } , q _ { z } , t _ { x } , t _ { y } , t _ { z } )$ , where $q _ { i }$ represent quaternion of a rotation and ti represent translation. We set the rotation and translation variations $\delta _ { q _ { i } }$ and $\delta _ { t i } .$ , and the current node is transformed to other neighboring nodes by different variations. The pose can be viewed as nodes in the search space, while the transitions between different pose correspond to edges in the graph, and this process can be viewed as expanding nodes in the search graph. In this application, the key to the heuristic algorithm is to design a reasonable cost function. We design the actual cost of a child node as the sum of the actual cost of the current node and the length of the path to the child node, and the estimated cost as the difference value between the rendered image and the query image corresponding to the pose of the current node:

$$
g ( n _ { c h i l d } ) = g ( n _ { c u r r e n t } ) + 1\tag{7}
$$

$$
h ( n _ { c h i l d } ) = \Sigma | I _ { q } - I _ { n c h i l d } |\tag{8}
$$

Where $I _ { q }$ represents the current query image and $I _ { n c h i l d }$ represents the rendering image of current child node.

The heuristic function effectively guides the algorithm toward the optimal pose, ultimately identifying the pose that produces a rendered image most similar to the query image. We provide the pseudo-code for the algorithmâs implementation in Tab. I. In this pseudo-code, OpenList is used to store nodes awaiting expansion, while CloseList contains nodes that have already been expanded.

TABLE I

## HEURISTIC POSE OPTIMIZATION STRATEGY

Heuristic Algorithm   
while openList is not empty:   
1. pop top node with min( f (n)) from openList.   
2. if top is destination node:   
break   
3. closeList.push(top)   
4. for each child node of top:   
if child in closeList:   
continue   
computes the costtentative from the start node to child.   
if child not in openList:   
g(nchild ) = g(ncurrent ) + 1   
$h ( n _ { c h i l d } ) = \Sigma | I _ { q } - I _ { n c h i l d } |$   
openList.push(child)   
elif costtentative < g(nchild ) :   
g(nchild ) = tentative cost   
heap adjustments

## IV. EXPERIMENT

In this section, we compare and analyze the coarse pose with the optimized pose, including pose accuracy and preci-

sion.

## A. Implementation

The deep learning framework employed in this work is PyTorch [31]. Each scene is reconstructed using 3D Gaussian Splatting (3DGS) with 30,000 training iterations, running on RTX 4090 GPUs. For the 7Scenes dataset, we adopt the SfM ground truth (GT) provided by [32].

## B. Datasets, Metrics and Baselines

a) Datasets: We evaluated our method on two public datasets: 7Scenes and Deep Blending. In the case of the 7Scenes datasets [33], [34], the official test lists were used as query images, while the remaining images were utilized for training. For the Deep Blending dataset [35], we selected four scenes and constructed a test image set following the 1-out-of-8 approach suggested by Mip-NeRF [36].

b) Evaluation Metrics: We show the median rotation and translation errors, and also provide the ratio of pose error within 1cm/1Â°.

c) Benchmark: Our approach is built upon an initial coarse pose estimation. For the APR [1]â[8] framework, we have selected the widely recognized Marepo [8] method as the benchmark for comparison. Similarly, for the SCR [9]â [11] framework, we have chosen the classical ACE [10] method as the benchmark for comparison.

## C. Analysis of results

a) 7Scenes dataset: For the 7Scenes dataset, we evaluate the performance of Marepo [8] and ACE [10] after incorporating HGSLoc. Tab. II demonstrates that our method effectively reduces the error in the coarse pose estimates obtained from both Marepo and ACE. Compared to other NRP methods, our approach achieves results with smaller relative pose errors. Furthermore, Tab. III presents the ratio of query images with relative pose errors of up to 1 cm and 1Â°, showing significant improvements after applying the HGSLoc framework. This indicates that our method efficiently optimizes cases involving small relative pose errors, further enhancing accuracy.

b) Deep Blending dataset: We selected two indoor scenes, âPlayroomâ and âDrJohnsonâ, and two outdoor scenes, âBoatsâ and âNightSnowâ, for testing. For both the Marepo [8] and ACE [10] methods, we observed that the coarse pose errors were significantly large. This may be attributed to the higher complexity of the Deep Blending dataset compared to the 7Scenes datasets, as well as the limited training data, which may have prevented model convergence. Consequently, we utilized an alternative method (HLoc [37]) that leverages point clouds to obtain an initial pose estimate and compared the results. As shown in Tab. IV, the improvement from boosting is not pronounced, likely due to the high image quality of the Deep Blending dataset, which already provided relatively accurate preliminary poses with the HLoc framework. To better demonstrate the effectiveness of our pose optimization method, Tab. V introduces various levels of step noise, making the visualization results more intuitive.

TABLE II  
WE PRESENT THE RESULTS OF COMPARISON EXPERIMENTS ON THE 7SCENES DATASET, HIGHLIGHTING THE MEDIAN TRANSLATION AND ROTATION ERRORS (CM/Â°) OF THE POSE RELATIVE TO THE GROUND TRUTH (GT) POSE FOR VARIOUS METHODS ACROSS SEVEN SCENES. THE BEST RESULTS ARE INDICATED IN BOLD. âNRPâ REFERS TO NEURAL RENDER POSE ESTIMATION.
<table><tr><td></td><td>Method</td><td>chess</td><td>fire</td><td>heads</td><td>office</td><td>pumpkin</td><td>redkitchen</td><td>stairs</td><td>Avg.â[cm/]</td></tr><tr><td>APR</td><td>Marepo [8]</td><td>1.9/0.83</td><td>2.3/0.91</td><td>2.2/1.27</td><td>2.8/0.93</td><td>2.5/0.88</td><td>3.0/0.99</td><td>5.8/1.50</td><td>2.9/1.04</td></tr><tr><td>SCR</td><td>ACE [10]</td><td>0.6/0.18</td><td>0.8/0.31</td><td>0.6/0.33</td><td>1.1/0.28</td><td>1.2/0.22</td><td>0.8/0.20</td><td>2.9/0.81</td><td>1.1/0.33</td></tr><tr><td rowspan="3">NRP</td><td>HR-APR [30]</td><td>2.0/0.55</td><td>2.0/0.75</td><td>2.0/1.45</td><td>2.0/0.64</td><td>2.0/0.62</td><td>2.0/0.67</td><td>5.0/1.30</td><td>2.4/0.85</td></tr><tr><td>NeRFMatch [22]</td><td>0.9/0.3</td><td>1.1/0.4</td><td>1.5/1.0</td><td>3.0/0.8</td><td>2.2/0.6</td><td>1.0/0.3</td><td>10.1/0.7</td><td>2.8/0.73</td></tr><tr><td>Marepo+HGSLoc</td><td>0.7/0.33</td><td>1.4/0.62</td><td>1.5/0.92</td><td>2.2/0.70</td><td>1.8/0.46</td><td>2.2/0.63</td><td>4.8/1.34</td><td>2.1/0.71</td></tr><tr><td></td><td>ACE+HGSLoc</td><td>0.5/0.17</td><td>0.6/0.25</td><td>0.5/0.29</td><td>1.0/0.25</td><td>1.1/0.21</td><td>0.7/0.20</td><td>2.8/0.69</td><td>1.0/0.29</td></tr></table>

TABLE III

WE PRESENT THE AVERAGE PERCENTAGE OF POSE ERRORS WITHIN 1 CM AND 1Â° ON THE 7SCENES DATASET. âNRPâ DENOTES NEURAL RENDER POSE ESTIMATION.
<table><tr><td></td><td>Methods</td><td>Avg.â[1cm,1Â°]</td></tr><tr><td>APR</td><td>Marepo [8]</td><td>6.2</td></tr><tr><td>SCR</td><td>ACE [10]</td><td>53.7</td></tr><tr><td>NRP</td><td>Marepo+HGSLoc</td><td>25.8</td></tr><tr><td>NRP</td><td>ACE+HGSLoc</td><td>59.2</td></tr></table>

TABLE IV

WE PRESENT THE MEDIAN TRANSLATION AND ROTATION ERRORS (CM/Â°) FOR BOTH THE INITIAL ESTIMATED POSE AND THE OPTIMIZED POSE RELATIVE TO THE GT POSE.
<table><tr><td></td><td>init error</td><td>refine error</td></tr><tr><td>Playroom</td><td>0.7/0.060</td><td>0.6/0.059</td></tr><tr><td>DrJohnson</td><td>0.3/0.055</td><td>0.3/0.054</td></tr><tr><td>Boats</td><td>0.5/0.016</td><td>0.4/0.013</td></tr><tr><td>NightSnow</td><td>0.2/0.024</td><td>0.2/0.019</td></tr></table>

As shown in Tab. VI, to further demonstrate the effectiveness of our method, we compare it with an alternative joint optimization strategy [28]. For this comparison, a noise level of $1 \times 1 0 ^ { - 3 }$ granularity is introduced to the initial pose. Our method employs heuristic optimization based on highquality scene reconstruction obtained through the 3DGS [16] method, whereas the alternative strategy jointly optimizes both the scene reconstruction and the initial noisy pose [28].

c) Qualitative Analysis: By inputting the pose into the 3D reconstructed scene, we generate a rendered image that visualizes the pose. Each query image corresponds to the GT pose, and the discrepancy between the estimated pose and the GT pose is reflected in the rendered images from various viewpoints. To better analyze errors and optimization improvements, we select viewpoints where significant accuracy gains are observed. Fig. 3 demonstrate that, when using our framework on the 7Scenes datasets, the rendered images more closely match the GT images. Fig. 4 illustrates the results of applying our framework to noisy poses in the Deep Blending dataset, showing that our method effectively refines the original pose, resulting in rendered images that closely resemble the GT images. However, experiments on the Cambridge Landmarks dataset [1] yielded suboptimal results, likely due to its limited viewpoint coverage, which leads to gaps in the 3D reconstruction of the scenes. In contrast, the 360Â° closed-loop image structure of the Deep Blending dataset enhance the effectiveness of pose optimization.

TABLE V  
WE SHOW THE MEDIAN TRANSLATION AND ROTATION ERRORS (M/Â°) FOR THE POSES WITH NOISE AND FOR THE POSES AFTER OPTIMIZATION. (Q2, T1) DENOTES THE INTRODUCTION OF NOISE AT THE PERCENTILE OF QVEC, DECILE OF TVEC, AND THE REST IS THE SAME.  
(a) Playroom
<table><tr><td></td><td>noise error</td><td>refine error</td><td>tvecâ</td><td>qvecâ</td></tr><tr><td>q2, t1</td><td>0.81/7.79</td><td>0.29/2.72</td><td>64.2%</td><td>65.1%</td></tr><tr><td>q2, t2</td><td>0.31/8.42</td><td>0.16/1.81</td><td>48.4%</td><td>78.5%</td></tr><tr><td>q3, t3</td><td>0.03/0.81</td><td>0.02/0.26</td><td>33.3%</td><td>67.9%</td></tr></table>

(b) DrJohnson
<table><tr><td></td><td>noise error</td><td>refine error</td><td>tvecâ</td><td>qvecâ</td></tr><tr><td>q2, t1</td><td>0.68/7.81</td><td>0.15/1.87</td><td>77.9%</td><td>76.1%</td></tr><tr><td>q2, t2</td><td>0.33/7.86</td><td>0.13/2.21</td><td>60.6%</td><td>71.9%</td></tr><tr><td>q3, t3</td><td>0.03/0.72</td><td>0.01/0.21</td><td>66.7%</td><td>70.8%</td></tr></table>

(c) Boats
<table><tr><td></td><td>noise error</td><td>refine error</td><td>tvecâ</td><td>qvecâ</td></tr><tr><td>q2, t1</td><td>0.62/6.41</td><td>0.01/0.02</td><td>98.4%</td><td>99.7%</td></tr><tr><td>q2, t2</td><td>0.26/9.17</td><td>0.17/2.46</td><td>34.6%</td><td>73.2%</td></tr><tr><td>q3, t3</td><td>0.04/0.71</td><td>0.01/0.15</td><td>75.0%</td><td>78.9%</td></tr></table>

(d) NightSnow
<table><tr><td></td><td>noise error</td><td>refine error</td><td>tvecâ</td><td>qvecâ</td></tr><tr><td>q2, t1</td><td>0.49/7.79</td><td>0.01/0.03</td><td>98.0%</td><td>99.6%</td></tr><tr><td>q2, t2</td><td>0.29/7.62</td><td>0.07/0.58</td><td>75.9%</td><td>92.4%</td></tr><tr><td>q3, t3</td><td>0.03/0.69</td><td>0.01/0.10</td><td>66.7%</td><td>85.5%</td></tr></table>

TABLE VI

WE SHOW THE MEDIAN TRANSLATION AND ROTATION ERRORS (M/Â°) FOR HEURISTIC OPTIMIZATION AND JOINT OPTIMIZATION STRATEGIES.
<table><tr><td></td><td>init error</td><td>joint error</td><td>heuristic error</td></tr><tr><td>Playroom</td><td>0.03/0.81</td><td>0.02/0.42</td><td>0.02/0.26</td></tr><tr><td>DrJohnson</td><td>0.03/0.72</td><td>0.02/0.47</td><td>0.01/0.21</td></tr><tr><td>Boats</td><td>0.04/0.71</td><td>0.01/0.19</td><td>0.01/0.15</td></tr><tr><td>NightSnow</td><td>0.03/0.69</td><td>0.02/0.36</td><td>0.01/0.10</td></tr></table>

d) Ablation study: In our method, we use the sum of pixel-by-pixel differences as the heuristic function. To demonstrate the effectiveness of this heuristic function, Tab. VII compares the results obtained using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM)

<!-- image-->  
ACE

<!-- image-->  
ACE+HGSLoc (Ours)

<!-- image-->  
Marepo

<!-- image-->  
Marepo+HGSLoc (Ours)  
Fig. 3. HGSLoc demonstrates a significant optimization effect on the coarse poses obtained using the ACE and Marepo methods. Each subimage is divided by a diagonal line: the rendered image from the pose is shown in the bottom left part, while the GT image is shown in the top right part. The rendered images corresponding to the ACE and Marepo methods exhibit substantial misalignment with the GT images. To facilitate a clearer comparison, we provide a zoomed-in view of the image, highlighted within the red box.

as alternative heuristic functions. Higher values of PSNR and SSIM indicate better image quality and structural similarity, whereas we would like to see them take the opposite number as the value of the heuristic function is as small as possible. To illustrate the impact of different heuristic functions more clearly, we applied these comparisons to the Deep Blending dataset, which introduces a relatively large level of noise.

$$
h _ { 1 } ( n _ { c h i l d } ) = 1 0 0 - P S N R ( I _ { q } , I _ { n c h i l d } )\tag{9}
$$

$$
h _ { 2 } ( n _ { c h i l d } ) = 1 . 0 - S S I M ( I _ { q } , I _ { n c h i l d } )\tag{10}
$$

TABLE VII  
WE SHOW THE MEDIAN TRANSLATION AND ROTATION ERRORS (M/Â°) FOR POSES WITH NOISE AND FOR POSES AFTER OPTIMIZATION USING DIFFERENT HEURISTIC FUNCTIONS.
<table><tr><td></td><td>noise error</td><td>H(Sum of Diff)</td><td>H(PSNR)</td><td>H(SSIM)</td></tr><tr><td>Playroom</td><td>0.81/7.79</td><td>0.29/2.72</td><td>0.76/6.29</td><td>0.87/6.83</td></tr><tr><td>DrJohnson</td><td>0.68/7.81</td><td>0.15/1.87</td><td>0.60/6.61</td><td>0.65/7.59</td></tr><tr><td>Boats</td><td>0.62/6.41</td><td>0.01/0.02</td><td>0.44/3.93</td><td>0.59/5.97</td></tr><tr><td>NightSnow</td><td>0.49/7.79</td><td>0.01/0.03</td><td>0.49/7.03</td><td>0.46/7.25</td></tr></table>

## V. LIMITATIONS AND FUTURE WORK

While our method successfully achieves the desired results, it still has certain limitations. During the optimization process, typically 300â400 nodes need to be expanded, with multiple Gaussian renderings performed for each expansion, which significantly constrains the methodâs real-time performance. Future work will focus on optimizing pose representation to reduce the number of Gaussian renderings required per node expansion, thereby improving computational efficiency and enhancing the methodâs applicability to real-time scenarios.

<!-- image-->  
noisy pose (playroom)

<!-- image-->  
noise pose+HGSLoc (Ours)

<!-- image-->  
noisy pose (drjohnson)

<!-- image-->  
noise pose+HGSLoc (Ours)

(a) Indoor Scenes  
<!-- image-->  
noisy pose (Boats)

<!-- image-->

<!-- image-->

noise pose+HGSLoc (Ours)  
<!-- image-->  
noisy pose (NightSnow)  
noise pose+HGSLoc (Ours)  
(b) Outdoor Scenes  
Fig. 4. Each subimage is divided by a diagonal line, with the image rendered by the estimated pose on the lower left and the GT image on the upper right. The diagonal lines in the optimized comparison image appear less distinct, reflecting improved alignment with the GT image. HGSLoc demonstrates its effectiveness in refining pose estimation, achieving precise values while mitigating the impact of noise.

## VI. CONCLUSIONS

This study introduces a lightweight, plug-and-play visual localization framework that integrates heuristic refinement with 3D reconstruction to enhance pose accuracy, achieving state-of-the-art performance on two datasets. Compared to NeRF-based methods [22], it achieves superior localization accuracy by efficiently refining coarse estimations using heuristic functions. The modular approach not only reduces dependence on complex neural network training, but also demonstrates robust performance in noisy environments. By combining heuristic refinement with a 3D Gaussian distribution, this approach offers a novel and effective solution, providing valuable reference for the future development of visual localization systems.

## REFERENCES

[1] A. Kendall, M. Grimes, and R. Cipolla, âPosenet: A convolutional network for real-time 6-dof camera relocalization,â in Proceedings of the IEEE international conference on computer vision, 2015, pp. 2938â2946.

[2] A. Kendall and R. Cipolla, âModelling uncertainty in deep learning for camera relocalization,â in 2016 IEEE international conference on Robotics and Automation (ICRA). IEEE, 2016, pp. 4762â4769.

[3] , âGeometric loss functions for camera pose regression with deep learning,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5974â5983.

[4] B. Wang, C. Chen, C. X. Lu, P. Zhao, N. Trigoni, and A. Markham, âAtloc: Attention guided camera localization,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 06, 2020, pp. 10 393â10 401.

[5] S. Chen, Z. Wang, and V. Prisacariu, âDirect-posenet: Absolute pose regression with photometric consistency,â in 2021 International Conference on 3D Vision (3DV). IEEE, 2021, pp. 1175â1185.

[6] S. Chen, X. Li, Z. Wang, and V. A. Prisacariu, âDfnet: Enhance absolute pose regression with direct feature matching,â in European Conference on Computer Vision. Springer, 2022, pp. 1â17.

[7] Y. Shavit, R. Ferens, and Y. Keller, âLearning multi-scene absolute pose regression with transformers,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 2733â2742.

[8] S. Chen, T. Cavallari, V. A. Prisacariu, and E. Brachmann, âMaprelative pose regression for visual re-localization,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 665â20 674.

[9] E. Brachmann and C. Rother, âVisual camera re-localization from rgb and rgb-d images using dsac,â IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 9, pp. 5847â5865, 2021.

[10] E. Brachmann, T. Cavallari, and V. A. Prisacariu, âAccelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 5044â5053.

[11] F. Wang, X. Jiang, S. Galliani, C. Vogel, and M. Pollefeys, âGlace: Global local accelerated coordinate encoding,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 562â21 571.

[12] T. Sattler, Q. Zhou, M. Pollefeys, and L. Leal-Taixe, âUnderstanding the limitations of cnn-based absolute camera pose regression,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 3302â3312.

[13] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[14] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-Y. Lin, âinerf: Inverting neural radiance fields for pose estimation,â in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 1323â1330.

[15] D. Maggio, M. Abate, J. Shi, C. Mario, and L. Carlone, âLoc-nerf: Monte carlo localization using neural radiance fields,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 4018â4025.

[16] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[17] P. Jiang, G. Pandey, and S. Saripalli, â3dgs-reloc: 3d gaussian splatting for map representation and visual relocalization,â arXiv preprint arXiv:2403.11367, 2024.

[18] C. Liu, S. Chen, Y. Bhalgat, S. Hu, Z. Wang, M. Cheng, V. A. Prisacariu, and T. Braud, âGsloc: Efficient camera pose refinement via 3d gaussian splatting,â arXiv preprint arXiv:2408.11085, 2024.

[19] Y. Hiasa, Y. Otake, M. Takao, T. Matsuoka, K. Takashima, A. Carass, J. L. Prince, N. Sugano, and Y. Sato, âCross-modality image synthesis from unpaired data using cyclegan: Effects of gradient consistency loss and training data size,â in Simulation and Synthesis in Medical Imaging: Third International Workshop, SASHIMI 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Proceedings 3. Springer, 2018, pp. 31â41.

[20] V. Leroy, Y. Cabon, and J. Revaud, âGrounding image matching in 3d with mast3r,â arXiv preprint arXiv:2406.09756, 2024.

[21] V. Bulitko, N. Sturtevant, J. Lu, and T. Yau, âGraph abstraction in real-time heuristic search,â Journal of Artificial Intelligence Research, vol. 30, pp. 51â100, 2007.

[22] Q. Zhou, M. Maximov, O. Litany, and L. Leal-Taixe, âThe nerfect Â´ match: Exploring nerf features for visual localization,â arXiv preprint arXiv:2403.09577, 2024.

[23] X. Song, J. Zheng, S. Yuan, H.-a. Gao, J. Zhao, X. He, W. Gu, and H. Zhao, âSa-gs: Scale-adaptive gaussian splatting for training-free anti-aliasing,â arXiv preprint arXiv:2403.19615, 2024.

[24] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, âCompact 3d gaussian representation for radiance field,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 719â21 728.

[25] T. Chen, O. Shorinwa, J. Bruno, A. Swann, J. Yu, W. Zeng, K. Nagami, P. Dames, and M. Schwager, âSplat-nav: Safe real-time robot navigation in gaussian splatting maps,â arXiv preprint arXiv:2403.02751, 2024.

[26] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 310â20 320.

[27] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, âFsgs: Real-time few-shot view synthesis using gaussian splatting,â arXiv preprint arXiv:2312.00451, 2023.

[28] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos, Z. Wang, and Y. Wang, âInstantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds,â 2024.

[29] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, âDust3r: Geometric 3d vision made easy,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 697â20 709.

[30] C. Liu, S. Chen, Y. Zhao, H. Huang, V. Prisacariu, and T. Braud, âHr-apr: Apr-agnostic framework with uncertainty estimation and hierarchical refinement for camera relocalisation,â arXiv preprint arXiv:2402.14371, 2024.

[31] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., âPytorch: An imperative style, high-performance deep learning library,â Advances in neural information processing systems, vol. 32, 2019.

[32] E. Brachmann, M. Humenberger, C. Rother, and T. Sattler, âOn the limits of pseudo ground truth in visual camera re-localisation,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 6218â6228.

[33] B. Glocker, S. Izadi, J. Shotton, and A. Criminisi, âReal-time rgbd camera relocalization,â in 2013 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2013, pp. 173â179.

[34] J. Shotton, B. Glocker, C. Zach, S. Izadi, A. Criminisi, and A. Fitzgibbon, âScene coordinate regression forests for camera relocalization in rgb-d images,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2013, pp. 2930â2937.

[35] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Brostow, âDeep blending for free-viewpoint image-based rendering,â ACM Transactions on Graphics (ToG), vol. 37, no. 6, pp. 1â15, 2018.

[36] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-nerf: A multiscale representation for antialiasing neural radiance fields,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5855â5864.

[37] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, âFrom coarse to fine: Robust hierarchical localization at large scale,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 12 716â12 725.