# 3DGS-LSRï¼Large-Scale Relocation for Autonomous Driving Based on 3D Gaussian Splatting

Haitao Lu 1,2 , Haijier Chen 1,2 , Haoze Liu 1,2 , Shoujian Zhang1,\* , Bo Xu1, Ziao Liu1

1 The School of Geodesy and Geomatics, Wuhan University, No. 129 Luoyu Road, Wuhan 430079, Peopleâs Republic of China

E-mail: shjzhang@sgg.whu.edu.cn

## Abstract

In autonomous robotic systems, precise localization is a prerequisite for safe navigation. However, in complex urban environments, GNSS positioning often suffers from signal occlusion and multipath effects, leading to unreliable absolute positioning. Traditional mapping approaches are constrained by storage requirements and computational inefficiency, limiting their applicability to resource-constrained robotic platforms. To address these challenges, we propose 3DGS-LSR: a large-scale relocalization framework leveraging 3D Gaussian Splatting (3DGS), enabling centimeter-level positioning using only a single monocular RGB image on the client side. We combine multi-sensor data to construct highaccuracy 3DGS maps in large outdoor scenes, while the robot-side localization requires just a standard camera input. Using SuperPoint and SuperGlue for feature extraction and matching, our core innovation is an iterative optimization strategy that refines localization results through step-by-step rendering, making it suitable for real-time autonomous navigation. Experimental validation on the KITTI dataset demonstrates our 3DGS-LSR achieves average positioning accuracies of 0.026m, 0.029m, and 0.081m in town roads, boulevard roads, and traffic-dense highways respectively, significantly outperforming other representative methods while requiring only monocular RGB input. This approach provides autonomous robots with reliable localization capabilities even in challenging urban environments where GNSS fails.

Keywords: 3D Gaussian mapping, relocation, complex environmentï¼autonomous driving

## 1. Introduction

Accurate and reliable localization in complex environments is essential for enabling unmanned technologies, such as autonomous driving. Current localization methods primarily depend on sensors like GNSS, LiDAR, IMUs, and cameras. However,GNSS-based localization encounters significant challenges in urban areas [1].

Tall buildings can obstruct satellite signals, leading to unstable reception, while the multipath effect â caused by signals reflecting off building surfaces â introduces positioning errors. When GNSS fails to provide accurate positioning, the trajectories and positions obtained from IMUs, LiDAR, and cameras can accumulate errors over time [2,3,4].To overcome these challenges, high-precision maps that have been pre-collected are often used for localization in complex scenarios where GNSS fails to provide accurate positioning. These maps contain essential information about absolute positions and orientations, which aids in localization [5]. Specifically, environmental feature data,

<!-- image-->  
Figure 1 A Visual Relocation System for High-Precision Pose Estimationï¼The system first matches the images captured by the camera with a reference image database to identify the reference image with the highest similarity, which is determined as the coarse pose.Subsequently, feature points between the target image and the reference image are extracted and matched using the RGB and depth images under the coarse pose, establishing a 2D-3D relationship. The pose of the target image is computed using the Perspective-n-Point (PnP) algorithm, resulting in a relatively accurate pose. Finally, RGB and depth images are rendered under the relatively accurate pose, and an iterative refinement method is employed to determine the final refined pose of the target image.

along with their corresponding high-precision absolute positions and orientations, is gathered in advance. When the carrier returns to the environment, it can retrieve and match its sensor data with the pre-stored environmental feature data to accurately determine its current absolute position and orientation [6].

These maps are available in various formats, including point cloud maps, vector maps, and bird's-eye view (BEV) maps. Each format has its limitations when used for absolute positional assistance in practical navigation scenarios [7]. Point cloud maps enable high-precision environmental alignment by matching LiDAR data; however, they are computationally demanding and require significant storage capacity. Vector maps utilize road networks and landmark information for large-scale navigation, but their positioning accuracy may not always be sufficient, and the landmark information can become outdated over time [8]. BEV maps provide a comprehensive representation of the environment, making it easier to match with camera images. However, limitations in resolution and field of view can negatively affect matching performance and positioning accuracy [9].

3D Gaussian(3DGS) maps represent a state-of-the-art method for mapping environments, providing high-fidelity visual representations from various perspectives while meticulously preserving texture details. Furthermore, their rapid rendering capabilities make them ideal for meeting the real-time demands of autonomous driving applications.

We introduce a new relocation method based on 3D Gaussian maps.First,creating a 3D Gaussian map specifically designed for autonomous driving scenarios. Next, we use RGB images captured by the onboard camera to match them with RGB images rendered from the Gaussian map. Once the initial matching is complete, we combine the RGB image with the depth image to estimate the vehicle's approximate position. This approximate position is then used to let the 3DGS map to render the image from the corresponding viewpoint. By iteratively repeating this process, our method achieves high-speed, centimeter-level repositioning with excellent accuracy.

In this paper, we propose a high-precision relocation method based on 3D Gaussian maps, highlighting the following key contributions:

ï¼1ï¼By combining efficient matching algorithms with the rapid rendering capabilities of Gaussian maps, this method allows for real-time, iterative repositioning with high accuracy.

ï¼2ï¼The method removes reliance on unreliable GNSS for absolute positioning in urban environments. Instead, vehicles can achieve centimeter-level repositioning using only camera sensors.

Our proposed method was evaluated on various sequences from the KITTI dataset, which represent different scenarios, and successfully demonstrated its real-time performance and localization accuracy.

## 2.RELATED WORK

In the field of relocation, existing methods mainly include fingerprint relocation based on radio signals and relocation based on sensors, such as LIDAR sensors, cameras, and so on.

Radio signal-based relocation methods utilize signals such as Wi-Fi and Bluetooth. While these methods perform well in indoor environments, they are susceptible to interference, which poses some challenges [10]. These challenges include the need for frequent updates to the signal fingerprint information, limited accuracy in relocation, and a restricted application range primarily suited for small indoor areas. Consequently, these methods are not suitable for large-scale outdoor scenarios, such as automated driving [11].

Sensor-based relocation can be categorized into two main types: LiDAR-based relocation technology and camera-based relocation technology. Camera-based visual relocation has several significant advantages over LiDAR and other highcost equipment.

Firstly, visual relocation can be achieved using ordinary cameras, resulting in lower costs for sensors, easier deployment, and reduced maintenance expenses compared to LiDAR. Secondly, cameras not only capture geometric information but also provide rich texture and illumination data, which enhances scene understanding and localization in complex environments.

Various research methodologies exist for visual relocation, including feature matching-based methods, scene coordinate regression [12], pose regression [13], and direct image alignment [14]. Notably, 2D-3D feature-matching methods are predominant in this domain. These methodologies ascertain the camera's pose â comprising both position and orientation â by extracting feature points from twodimensional images and aligning them with pre-constructed three-dimensional scene feature points.The construction of 3D scene models usually relies on Structure-from-Motion (SfM) [15], Truncated Signed-Distance Function (TS) based on the range image, and the Signed-Distance Function (TSDF) [16] based on range images or LiDAR image building techniques [17]. However, these traditional methods have deficiencies in the ability to express lighting and texture information, and the generated models are mainly used for geometric reconstruction, which is difficult to support image rendering at new viewpoints, limiting their applicability in complex scenes.

As technology advances, the methods used to represent maps for relocation are also changing. Unlike traditional fixed formats such as vector maps or point cloud maps, newer representations like Neural Radiance Fields (NeRF) [18]and 3D Gaussian Splatting offer a more detailed depiction of scene information [19,20,21]. These innovative methods can generate high-fidelity images from any viewpoint, providing new opportunities for research in relocation based on these advanced map representations.

The introduction of Neural Radiation Fields (NeRF) and subsequent research has established a novel paradigm for viewpoint synthesis and camera poseestimation. NeRF is capable of providing precise and realistic representations of static scenes characterized by complex geometries and lighting conditions. It achieves this by modeling the scene as a continuous three-dimensional volume and learning the inherent properties of its radiation field. However, despite the high accuracy of NeRF-based camera pose estimation methods, they encounter several limitations. For example, the computational complexity associated with the inverse projection and light projection processes of the NeRF model results in prolonged training and inference times, which complicates the achievement of real-time performance. Furthermore, NeRF's limited capacity to accommodate dynamic scenes further restricts its applicability in specific scenarios.

In contrast, the 3D Gaussian map represents an advanced method of map representation that offers enhanced computational efficiency and superior dynamic scene representation [22,23]. This technique facilitates rapid rendering and allows for the seamless integration of multimodal data, making it particularly advantageous for real-time applications, such as autonomous driving. The SplatLoc [24] approach was the first to implement the 3DGS technique for visual localization by introducing 3D Gaussian primitives to represent scenes. This method has demonstrated performance that either exceeds or is comparable to leading implicit visual localization techniques in terms of rendering and localization efficacy. Subsequently, GauLoc [25] put forth an implicit feature metric alignment method that exhibits increased accuracy and robustness in complex scenarios. The GSLoc [26] method realized efficient image alignment and visual localization, particularly in texture-free environments, by employing a coarse-to-fine optimization strategy and integrating a fuzzy kernel to alleviate the nonconvex optimization challenges. Likewise, GsplatLoc [27] achieved efficient image alignment and visual localization in texture-free contexts by minimizing the disparity between the rendered depth map and the observed depth map, thereby attaining exceptionally accurate camera position estimation.

Current methods for position estimation using 3D Gaussian maps primarily focus on small indoor environments. However, these methods encounter challenges when applied to large-scale outdoor open scenes. To address this issue, the approach proposed in this paper utilizes 3DGS maps and camera sensors to achieve low-cost, high-precision, and rapid repositioning for users in extensive outdoor environments.

## 3.OUR APPROACH

In this section, we elaborate on our system, which is divided into three categories: 3DGS map rendering, enhanced feature extraction and matching, and refined visual relocation. The overall system framework is shown in Fig. 1.

## 3.1 3D Gaussian map construction

For the large-scale outdoor scenes faced by autonomous driving, due to the influence of a single viewing angle, high contrast, and an increase in the number of moving objects, the traditional 3DGS image construction method will produce a blind spot in the viewing angle, which affects the acquisition of depth information and the three-dimensional sense of the object, and also leads to overexposure or the loss of shadow details in the image, resulting in the phenomenon of artifacts, which reduces the accuracy of the threedimensional reconstruction. Therefore, we adopt OmniRe [28], a â panoramic â system capable of handling diverse participants, to render the image, which integrates laser data and image information, effectively adapts to the limitation of a single viewpoint, provides more credible depth information, and builds a clear and blur-free image. At the same time, by constructing a dynamic neural scene graph, using Gaussian representation to model various dynamic actors in the scene, and assigning dedicated local normative space to different dynamic entities, we achieve comprehensive capture and reconstruction of complex dynamic scenes, restore the scene with high fidelity, ensure the completeness of the details, and satisfy the demand for high-precision 3DGS map models required by our high-precision localization for autonomous driving.

Among them, in order to realize the effective representation of various dynamic objects, the Gaussian scene graph is composed of sky nodes, background nodes, rigid nodes, and non-rigid nodes; the background nodes are represented by a set of static Gaussians $G _ { b g }$ , which are initialized by accumulating LIDAR points and randomly generated extra points [29]; the rigid nodes are represented by G v in the local space, which are in turn transformed into the world space by the following formula:

$$
G _ { \nu } = T _ { \nu } \left( t \right) \otimes G _ { \nu }\tag{1}
$$

The non-rigid nodes are further subdivided into two categories: the SMPL nodes for walking or running pedestrians as well as the deformable nodes for out-ofdistribution non-rigid instances, and the two types of nodes are represented in the world space formulated as follows:

$$
G _ { \scriptscriptstyle S M P L } ( t ) = T _ { \scriptscriptstyle h } ( t ) \otimes L B S ( \theta ( t ) , \bar { G } _ { \scriptscriptstyle S M P L } )\tag{2}
$$

$$
G _ { d e f o r m } ( t ) = T _ { h } ( t ) \otimes ( G _ { d e f o r m } \otimes F _ { \ell } ( G _ { d e f o r m } , e _ { h } , t ) )\tag{3}
$$

The three nodes are transformed into a Gaussian representation in world space. This Gaussian is then rendered and stitched together by a rasterizer. Meanwhile, the sky node is depicted using an optimized environment texture map and is rendered separately. The rendering results $C _ { s k y }$ from the sky node are then combined with the overall rendering results of the other nodes to produce the final output, defined by the following formula [29,30]:

$$
C = C _ { \sc G } + ( 1 - O _ { \sc G } ) C _ { s k y }\tag{4}
$$

Optimize all of the above parameters simultaneously in one stage to achieve the best overall rendering, the optimization function is defined as follows:

$$
\begin{array} { r } { \zeta = \left( 1 - \lambda _ { r } \right) \zeta _ { 1 } + \lambda _ { r } \zeta _ { S S I M } + \lambda _ { d e p t h } \zeta _ { d e p t h } + \lambda _ { o p a c i t y } \zeta _ { o p a c i t y } + \zeta _ { r e g } } \end{array}\tag{5}
$$

The $\zeta _ { 1 }$ is the L1 loss used to measure the pixel-level difference between the rendered image and the target image, $\zeta _ { S S I M }$ is the structural similarity loss of the rendered image, $\zeta _ { d e p t h }$ is the difference between the rendered Gaussian depth values and the sparse depth signal from the LiDAR data, and denotes the regularization constraints applied to the different Gaussian representations.

## 3.2 Enhanced feature point matching method

The front-end task of our whole system is feature point extraction and matching, common feature point extraction and matching methods, such as SIFT+FLANN [31], ORB+BFMatcher [32], etc. The effect is limited by feature selection, there is the possibility of information loss, and more sensitive to noise and occlusion, although the speed of processing speed is faster, but in terms of matching accuracy is not accurate enough, so it is difficult to cope with the complex visual environment of large-scale scenes faced by automated driving, and it is impossible to find the best matching image in the sparse gallery and accurately find a sufficient number of feature matching points. Therefore, our system adopts the SuperPoint+SuperGlue [33,34] combination of feature extraction and matching, using SuperPoint to extract feature points and descriptors in the image, and SuperGlue to realize the matching between images, organically combining the three steps of feature point detection, feature description, and feature matching based on deep learning into a complete deep network architecture, so as to realize an end-to-end deep network architecture. architecture, thus realizing an end-to-end image feature point matching processing front-end module [35]. The method has good performance in dealing with lowquality images caused by fast vehicle movement and missing or obstructed light in the field of view, and at the same time, the number of extracted feature points and the accuracy of the matching results are much better than the traditional methods when facing large-scale changing images.

SuperPoint is a deep network trained by a self-supervised approach to recognize a large number of feature points and also generate high-dimensional fixed-length descriptors [36]. The SuperPoint network mainly consists of three parts: encoder, feature point decoder, and descriptor decoder. In this network, the feature point detector and descriptor in the encoding phase share the same encoding network; while in the decoding phase, the According to the requirements of specific tasks, feature point decoding and descriptor decoding adopt different structures, so as to learn their independent network parameters, the specific algorithms are shown in the following Algorithm1 [33].

Algorithm1: SuperPoint   
Input: An image of size $H \times W$   
Output: Feature point probabilities, descriptors   
1.Extraction of base features using the VGG-style encoder   
while reducing the image scale   
2.The feature point decoder converts the dimensions of the   
image output by the encoder into a tensor of the size of   
$H / 8 \times \bar { W } / 8 \times \bar { 6 } 4$ , Then processed by Softmax and   
Reshape module, a new $H \times W$ size tensor can be obtained,   
and after the encoder can finally get the probability that   
each pixel point is a feature point.   
3.The descriptor decoder first acquires a semi-dense   
descriptor and then transforms the size of the image output   
by the encoder into a tensor of $H / 8 \times W / 8 \times D$ size, The   
rest of the description is obtained by double cubic   
polynomial interpolation, and then the normalized   
descriptor processing module is introduced to finally obtain   
the tensor of $H \times W \times D$ size, and the local descriptor of   
each pixel is finally obtained by the descriptor decoder.

SuperGlue is a novel graph neural network algorithm [37] that can efficiently filter out outliers during feature matching. It can utilize the differentiable optimal transmission theory for feature matching, and combines the attention mechanism with the introduction of twodimensional feature points and aggregation strategy,so that the algorithm can efficiently process feature points and descriptors obtained from both traditional and deep learning methods to generate accurate matching pairs. The specific algorithm flow is shown in the following Algorithm2 [34]:

Algorithm2: SuperGlue   
Input: Detected feature points and descriptors   
Output: An accurate match   
1.Using the GNN (graph neural network) module, the   
feature points and their descriptors are converted into   
feature matching vectors, and the self-attention and cross  
attention mechanisms in the attention mechanism   
continuously reinforce the (L=7) vector F;   
2.At the optimal matching level, the inner product is used to   
obtain the score matrix, and after the Sinkhorn algorithm (T   
iterations) processes these points to solve the optimal   
feature assignment matrix and obtain the overall accurate   
matching results of the final image feature points.

## 3.3 High-precision attitude convergence

1ï¼ Initial Matching to Determine the Coarse Position:

when determining the initial position of the target location, the feature matching between the target query image and the RGB images in the reference image library is first performed by the SuperGlue algorithm in the reference image library, and the feature matching is optimized by using the spatial context and the global features, to identify the reference RGB image $I _ { G S }$ with the highest degree of similarity,The reference library here is the multiple RGB images generated from the 3DGS map. When we obtain the 6D position parameters of the reference image, it is used as the initial coarse position of the target point.

2ï¼PnP to determine the accurate positional attitude: in the process of determining the coarse positional attitude is matched to the reference image, which contains the RGB image and depth image generated by the 3DGS map, which for each spatial point we can obtain a set of threedimensional coordinates under the camera coordinate system, i.e., $\left( X _ { \cal G } , \frac { } { } Y _ { \cal G } , \frac { } { } Z _ { \cal G } \right)$ through the chi-square coordinates of the spatial point with the feature points obtained by the projection of the normalized planar chi-square coordinates, we adopt the Perspective-n-Point (PnP) method to obtain the relative accurate position of the target image location based on the initial rendering reference map obtained.

EPnP (efficient perspective n point) algorithm is an efficient way to solve the PnP conditional problem [38] ï¼ the method is small in computation and the complexity is , for the extraction of a larger number of feature points, has a faster computational efficiency. The core theory of the EPnP algorithm is to use the non-coplanar virtual control point linearly weighted representation of an arbitrary sign point in the camera coordinate system, so it is generally the first to determine the virtual control point and sign point in the camera coordinate system position, construct the relationship between the world and the camera coordinate system is defined as follows:

$$
\begin{array} { l } { \displaystyle { \left\{ P _ { i } ^ { k } = \sum _ { j = 1 } ^ { 4 } a _ { i j } C _ { j } ^ { k } \left( i = 1 , 2 , 3 \cdots , n \right) \right. } } \\ { \displaystyle { \left. \left[ P _ { i } ^ { m } = \sum _ { j = 1 } ^ { 4 } a _ { i j } C _ { j } ^ { m } \left( i = 1 , 2 , 3 \cdots , n \right) \right. \right. } } \end{array}\tag{6}
$$

$P _ { i } ^ { k }$ and ${ P _ { i } } ^ { m }$ are the positions of the marker points in the world coordinate system and the corresponding camera coordinate system, $C _ { j } ^ { k }$ and $C _ { j } ^ { m }$ are the positions of the virtual control points in the world coordinate system and the corresponding camera coordinate system, respectively, and are the weighting coefficients corresponding to each marker point, which sums up to 1. In order to obtain the positions of the virtual control points in the camera coordinate system, the construction equations are defined as follows:

$$
\begin{array} { r } { Z _ { c } \left[ \begin{array} { l } { u _ { i } } \\ { \nu _ { i } } \\ { 1 } \end{array} \right] = K \left[ \begin{array} { l } { R T } \\ { 0 1 } \end{array} \right] \left[ \begin{array} { l } { X _ { i } ^ { m } } \\ { Y _ { i } ^ { m } } \\ { Z _ { i } ^ { m } } \\ { 1 } \end{array} \right] } \end{array}\tag{7}
$$

Where K is the internal reference matrix of the camera and $\boldsymbol { C } _ { j } ^ { m } = \left[ \boldsymbol { X } _ { i } ^ { m } \quad Y _ { i } ^ { m } \quad Z _ { i } ^ { m } \quad 1 \right] ^ { T }$ are the unknown quantities to be solved, i.e. the coordinates of the virtual control point in the camera coordinate system. The 6D position of the target point is then solved using the ICP (Iterative Closest Point) absolute localization method [39].

In order to make use of more effective information and reduce the influence of noise between matching feature points, we proceed to construct a least squares optimization problem to adjust the estimates (Bundle Adjustment) [40], by constructing a least squares problem for the pixel and spatial point position relation matrix error to minimize it Li algebraically, the formula is defined as follows:

$$
{ T } ^ { * } = \arg \operatorname* { m i n } _ { T } \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } \lVert u _ { i } - \frac { 1 } { s _ { i } } K T P _ { i } \rVert _ { 2 } ^ { 2 } .\tag{8}
$$

Find the optimal solution as the exact solution for the 6D position of the target pointã

3ï¼Iterative calculations to determine the refined positionï¼

Considering the constraints of positioning accuracy under one computation, in order to further improve the accuracy, we iteratively refine the accurate positioning determined by one PnP, and refer to the following algorithm 3 for details.

The accurate positioning information is re-inputted to the 3DGS map, and the rendering generates the RGB image and depth image under the accurate position, and the latest rendering result and the target map are utilized to iteratively execute the positioning Determination, at the end of each execution a new target point pose is generated, the iterative process aims at determining a more accurate target point pose, and the latest result obtained after satisfying the iteration conditions is used as the final refined pose, i.e., it is considered to be the pose information of the target point.The detailed relocation iterative optimization structure is as follows: Algorithm 3.

## 4. Experimental Evaluation

In this section, we begin by introducing the dataset utilized for our study and outlining the implementation details of our method, along with the performance evaluation metrics. We then assess our proposed method in terms of its feature-matching capabilities and localization performance.

Algorithm3: Iterative optimization   
Input: relatively accurate pose $P _ { 1 }$   
Output: fine pose $P _ { t r u t h }$   
// determine an upper limit for the number of cycles $i _ { \mathrm { m a x } }$   
${ i _ { \operatorname* { m a x } } \leq 1 0 }$   
for $i \gets 1 \ : \mathbf { t o } i _ { \mathrm { m a x } }$ do   
//3DGS rendering   
$( C _ { r g b } ^ { i } , C _ { d e p t h } ^ { i } , C _ { p o s e } ^ { i } ) = C _ { r a n d e r } ( P _ { i } )$   
// PnP algorithm processes the target and renders the   
data   
$P _ { i + 1 } = C _ { P n P } ( ( C _ { r g b } ^ { i } , C _ { d e p t h } ^ { i } , C _ { p o s e } ^ { i } ) , C _ { o b j e c t i v e } )$   
$i = ( 1 , { i _ { \mathrm { m a x } } } )$   
// loop termination condition   
If $\left| P _ { n } - P _ { n - 1 } \right| \leq 0 . 0 1$ then   
$P _ { t r u t h } = P _ { n }$   
Else   
execute $C _ { r a n d e r } 0$ and $C _ { p _ { n } P } 0$ function   
end   
$P _ { t r u t h } = P _ { n } ^ { e n d }$

The evaluation demonstrates the superiority of our approach through comparative tests with various high-performance methods. The purpose of these test experiments is to validate the effectiveness of our proposed method, which aims to achieve fast and efficient re-localization based on 3DGS maps and visual information in large-scale autonomous driving scenarios

## 4.1 Experimental Setup

Datasets We used the Karlsruhe Institute of Technology and Toyota Technological Institute dataset(KITTI) dataset for 3DGS mapping as well as relocation tests to evaluate our approach.The KITTI dataset is a standard dataset developed by the Karlsruhe Institute of Technology and the Toyota Technological Institute in Chicago for a wide range of applications in autonomous driving research. The dataset contains a rich set of real-world driving scenarios, in which we use the camera, IMU inertial navigation device, Velodyne HDL-64E S2 LIDAR sensor data, and the MASK information of the objects to complete the construction of high-precision 3DGS maps of large-scale scenarios, and then we use a single image provided by the camera to complete the relocation, and the relocation of the true value of the position of the carrier is based on the dataset. The positional truth value of the repositioning is calculated based on the carrier positional truth value provided by the dataset and the external reference matrix of the camera.

<!-- image-->  
Figure 2 Comparison of feature matching effect. A comparison of the matching results using the traditional method and superglue in different scenarios is given in the figure. Superglue achieves better-matching results in each scenario.

Implementation Details Our localization process was implemented on a system with an Intel Core i9-13620H CPU, 16 GB of RAM, and an NVIDIA RTX 4090 GPU with 24 GB of graphics memory. The algorithm was developed in Python and PyTorch, utilizing a custom CUDA kernel to accelerate the rasterization and backpropagation processes inherent in our micro-renderable approach. This configuration ensures that our approach achieves real-time performance, which is critical for real-world applications of retargeting systems.

## 4.2 Image feature point matching comparison

In our approach, a portion of 2D RGB maps and their corresponding depth maps are first rendered based on the established outdoor large-scale 3DGS maps as the base map library for the initial matching of the relocation query maps. To ensure the lightness of the base map library and the timeliness of the query image matching, the base map library has to be rendered with a certain degree of sparseness, i.e., only one image will be rendered after a certain distance of being rendered densely. Instead of rendering it densely.

However, in large-scale outdoor scenes, the sparse rendering of images results in significant differences in the viewing angle between the query image and even the bestmatching RGB image in the gallery. Therefore, we must ensure that the query can accurately select the best matching image in the base gallery, even in the presence of large viewing angle differences. We adopts SuperPoint+SuperGlue combination method in the front-end feature point extraction and matching part, which performs well in the outdoor scene, especially in the scene with blurred image and insufficient light, and also can dynamically adjust its operation rate according to the complexity of the matching task, which greatly improves the number of feature points extracted and the accuracy of the matching result comparing with the traditional method.

Figure 2 gives a comparison of the feature extraction and matching effects of the traditional method (ORB+Brief) and our method (SuperPoint+SuperGlue) under multiple scenarios. Obviously, the feature point extraction and matching ability of the combined method is significantly better than the traditional method in large-scale scenarios. The first and second rows indicate the road environments with sparse features and the road environments with large steering angles, respectively, in which the combined ORB + Brief method suffers from matching errors in these two challenging environments. The third and fourth rows represent road environments with more features, and the SuperPoint + SuperGlue combination method can extract more high-quality matches to ensure the accuracy of the initial displacements calculated by pnp.

In addition to this, with the 3DGS iteration by iteration, the feature matching effect between the rendered image and the query image is also gradually improved, as shown in Fig. 3. In the iteration-by-iteration process; it can be found that the view angle gap between the new image rendered by the 3DGS map and the query image is gradually reduced, and the number of feature points matched between the images is gradually improved, which makes the accuracy of calculating the relative displacement also gradually improved, which is in line with the iterative progression of the relocation process.

Figure 3 visually reflects the improvement in feature point extraction and matching as the number of iterations increases. Table 1 quantitatively demonstrates the improvement effect of iterations on feature point extraction and matching using indicators such as the number of feature point matches, the average confidence of feature matches, and the uniformity of feature point distribution. The number of feature point matches reflects the number of feature points in two images. The average confidence of feature matches indicates the reliability of all feature matches. The uniformity of feature point distribution is the difference between the normalized standard deviation of feature point distribution and 1; the closer it is to 1, the more uniformly the feature points are distributed across the two images.

<!-- image-->  
Figure 3: Iterative matching graph. As the iteration proceeds, the rendered image gets closer and closer to the query graph, and the number of feature point matches and the confidence of feature point matches between the two gradually increase

Table 1: trajectory errors in different scenarios
<table><tr><td>Number of Iterations</td><td>Number of Feature Point Matches</td><td>Average Confidence of Feature Matches</td><td>Uniformity of Feature Point Distribution</td></tr><tr><td>1</td><td>58</td><td>0.735</td><td>0.354</td></tr><tr><td>2</td><td>126</td><td>0.826</td><td>0.466</td></tr><tr><td>3</td><td>257</td><td>0.892</td><td>0.598</td></tr><tr><td>4</td><td>383</td><td>0.957</td><td>0.701</td></tr></table>

## 4.3 Relocation experiments

We tested the relocation accuracy of our method in three typical scenarios in the KITTI dataset, namely, the town road, the boulevard road, and the vehicle-dense highway.

The distribution of the error between the relocation results and the reference truth value of some pictures in a selected sequence of each scene is counted as shown in the following figure 4.

Seq1, Seq2, and Seq3 in the above figure represent the localization error statistics on the town road, boulevard road , and traffic-intensive highways, respectively. Among them, the localization performance was best in the town road, while it was slightly weaker on boulevard roads. This may be due to the comparison shown in Figure 5, between the first and second rows. When rendering the complex and variable light and shadow scenes in the forest, compared to town roads, there is a certain degree of blurring in the randered images,as indicated by the red box in the right image of the second row.

<!-- image-->

<!-- image-->

Seq1  
<!-- image-->  
(c)

<!-- image-->  
(d)

Seq2  
<!-- image-->  
(e)

<!-- image-->  
Seq3  
(f)  
Figure 4: Histogram of position and angle localization errors in town roads, boulevard roads, and traffic-dense highway scenes

This affects the extraction of feature points and the accuracy of subsequent relative localization calculations. In the vehicle-intensive highway scene, the average positional error and average angular error of the relocation calculation are relatively poorer among the three scenes, which may be because, although our 3DGS construction method effectively avoids the ghosting effect of rendering dynamic objects, in feature-degraded environments like highways, there are fewer feature point matches between images, as shown in the third row of Figure 5, with poor distribution uniformity. This results in a decrease in the final average error.

Then we randomly selected a period of time in each of the three sequences respectively, each with a length of 300 meters. and arranged the pictures taken by the camera sensors in the vehicle trajectory during that time in order, and relocated them sequentially as query pictures, and statistically analyzed the obtained trajectory results.

The absolute trajectory errors of the localization results and the reference truth in each of the three scenarios are synthesized statistics, including the root mean square error, standard deviation, mean, median, minimum, and maximum values as shown in Table 2.

From the Table 2, it can be seen that the RMSE and Std value of the absolute trajectory error of the town road are the smallest, respectively 0.026935 and 0.018506, but the Mean value of the trajectory error of the boulevard road is the smallest, 0.01312, and the statistics of each of the vehicledense road absolute trajectory error are all on the large side, so that the RMSE in the town road scenario as well as in the boulevard scene is close to 0.02, and the accuracy is the The best, although the effect is slightly worse under the highway scene, but the RMSE is also close to 0.07, combined with the above trajectory diagram can be seen, the three scenes, the overall have higher positioning accuracy results. At the same time, the standard deviation of the errors in the three scenarios are 0.018506, 0.023182, 0.069639 respectively, which shows that the absolute trajectory error is very robust, and in summary, it shows that our system still meets the demand for high-precision localization in the case of continuous operation of vehicles.

<!-- image-->  
Figure 5: Analysis of reasons affecting positioning accuracy

Table 2: trajectory errors in different scenarios(m)
<table><tr><td>Seq</td><td>RMSE</td><td>Std</td><td>Mean</td><td>Median</td><td>Min</td><td>Max</td></tr><tr><td>1</td><td>0.026</td><td>0.018</td><td>0.019</td><td>0.015</td><td>0.003</td><td>0.209</td></tr><tr><td>2</td><td>0.027</td><td>0.023</td><td>0.014</td><td>0.013</td><td>0.008</td><td>0.246</td></tr><tr><td>3</td><td>0.079</td><td>0.069</td><td>0.038</td><td>0.018</td><td>0.006</td><td>0.522</td></tr></table>

At the same time, the trajectory obtained from the relocation calculation is compared with the ground truth as shown in Figure 6 .

We have applied special magnification processing to the point of maximum deviation between the estimated trajectory and the ground truth trajectory. It can be observed that, in terms of displacement error, the Z-axis shows a significant anomaly at the end of the sequence in the dense highway scene, reaching 0.7 meters. This is the reason why the average localization accuracy in this scene is noticeably lower than in the other two. At this moment, there are too many moving vehicles in the highway scene. Although our 3DGS mapping method can largely avoid motion blur during rendering, in extremely extreme situations, some blurring still occurs, affecting the accuracy of relocalization.

To validate the sophistication of our method, we compare it with five representative relocation methods, which include DSAC++, AS, NG-RANSAC, Regressiononly, and 3DGS-Reloc.DSAC++ is a deep learning-based relocation method, which estimates the camera position with high accuracy and generalization ability through differentiable RANSAC algorithm and CNN network to predict the scene coordinates and use PnP algorithm to

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
ï¼1ï¼

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
ï¼3ï¼  
Figure 6 Seq1, 2, and 3 predicted trajectories vs.ground truth

estimate the camera position with high localization accuracy and generalization ability.AS (Active Search) is a featurepoint matching based method that combines the F2P and P2F search mechanisms to optimize the matching process by using covariance relationship and solves the position by DLT and RANSAC algorithms, which is suitable for texture-rich scenes.NG- RANSAC is an improvement of the classical RANSAC algorithm, which optimizes the sampling strategy and computation process, and improves the operation efficiency and robustness.The Regression-only method predicts the camera position directly from the image by neural network, which does not rely on feature point extraction, and has fast computation and strong noise immunity.3DGS-Reloc utilizes 3D Gaussian rendering technique for re localization, which is similar to the idea of our constructed method, but its requires additional sensors such as GPS provision to provide a rough initial position. These comparisons cover a wide range of techniques, from traditional feature matching to deep learning, and demonstrate the advantages of our approach.

Table 3: Localization results on the KITTI, Waymo, and NuScenes datasets. The average absolute translation error (m) and absolute rotation error (Â°) are reported for each scene, and Acc. indicates the average accuracy for all scenes.
<table><tr><td rowspan="2">Method</td><td colspan="3">KITTI</td><td colspan="3">Waymo</td><td colspan="3">NuScenes</td></tr><tr><td>Townroad Boulevard</td><td></td><td> $\operatorname { A v g }$ </td><td>Townroad Boulevard</td><td></td><td>Avg</td><td>Townroad Boulevard</td><td></td><td>Avg</td></tr><tr><td>AS [41]</td><td>16.6/0.75</td><td>21.2/0.84</td><td>18.9/0.79</td><td>15.1/0.77</td><td>19.2/0.71</td><td>17.1/0.74</td><td>16.3/0.59</td><td>20.5/0.62</td><td>18.4/0.61</td></tr><tr><td>NG-RANSAC [42]</td><td>13.9/-</td><td>16.4/-</td><td>15.2/-</td><td>10.5/-</td><td>11.6/-</td><td>11.1/-</td><td>8.1/-</td><td>10.4/-</td><td>9.25/-</td></tr><tr><td>Regression-only</td><td>10.9/0.74</td><td>13.4/0.79</td><td>12.2/0.77</td><td>12.8/0.62</td><td>15.4/0.72</td><td>14.1/0.67</td><td>9.7/0.53</td><td>12.2/0.56</td><td>10.9/0.55</td></tr><tr><td>DSAC++ [43]</td><td>8.8/0.67</td><td>10.2/0.72</td><td>9.5/0.70</td><td>7.8/0.53</td><td>11.3/0.63</td><td>9.6/0.58</td><td>6.5/0.42</td><td>9.9/0.48</td><td>8.2/0.45</td></tr><tr><td>3DGS-Reloc [44]</td><td>4.8/0.77</td><td>5.2/0.75</td><td>5.0/0.76</td><td>3.0/0.64</td><td>4.2/0.71</td><td>3.6/0.68</td><td>4.4/0.53</td><td>3.8/0.57</td><td>4.1/0.55</td></tr><tr><td>Ours</td><td>2.9/0.67</td><td>3.5/0.71</td><td>3.2/0.69</td><td>3.1/0.55</td><td>3.5/0.61</td><td>3.3/0.58</td><td>2.6/0.42</td><td>3.1/0.47</td><td>2.9/0.45</td></tr></table>

<!-- image-->  
Figure 7: Percentage of High-Precision Relocalization Images

The results are reported in Table 3. Some results of competing methods were obtained from the respective papers, and others were derived from reproduced code. Overall, our method achieves excellent results. On the KITTI dataset, our method achieved localization errors of 2.9 cm/0.67Â° and 3.5 cm/0.71 o , significantly outperforming the other five methods. On the Waymo and NuScenes datasets, our method also achieved the best overall localization performance. Although our method did not achieve the best results on all datasets in the Townroad scenario, it maintained excellent localization accuracy in the more challenging Boulevard scenario. We also reported the percentage of test images with localization errors less than 10 cm and 1 degree, as shown in Figure 7. This metric is used to comprehensively compare the localization performance of different methods. It can be seen that our method achieved the best performance across the three different datasets. Overall, our method achieved the lowest localization error in all scenarios and demonstrated good stability and adaptability across different environments. This indicates that our method has significant advantages in various complex environments.

## 4.4 Timeliness analysis

We analyzed the time consumption at each stage of the proposed model during the relocalization task. The relocalization process mainly includes the following steps: 1)querying the nearest anchor image from the retrieval database; 2) calculating the relative displacement between the query image and the nearest anchor image using the PnP algorithm; 3) rendering a new anchor image based on this relative displacement using a Gaussian model; 4) iteratively updating the localization result until the stopping condition is met.

Table 4:The time spent on each step during the relocalization process.
<table><tr><td rowspan="2">stage</td><td rowspan="2">Running Speed</td><td colspan="3">Number of Executions</td></tr><tr><td>Seq1</td><td>Seq2</td><td>Seq3</td></tr><tr><td>Superpoint</td><td>10ms/image</td><td>25</td><td>35</td><td>45</td></tr><tr><td>Superglue</td><td>30ms/image</td><td>25</td><td>35</td><td>45</td></tr><tr><td>PnP</td><td>2ms/image</td><td>5</td><td>5</td><td>5</td></tr><tr><td>render</td><td>6ms/image</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Time Spent</td><td>-</td><td>1.04s</td><td>1.44s</td><td>1.84s</td></tr></table>

In Table 4, we analyzed the time consumption at each stage and reported the relocalization time for three KITTI scenarios. The entire relocalization process can be completed within 2 seconds, indicating that this method can meet the real-time requirements of relocalization tasks in most autonomous driving scenarios and has the potential for practical application. In all scenarios, the iterative rendering converges in about five iterations. However, as the scene size increases, the number of anchor points increases, leading to a linear increase in retrieval time and a decrease in relocalization speed. By designing a reasonable anchor point selection strategy to reduce the size of the query database, the efficiency of the relocalization algorithm can be effectively improved.

## 5. Conclusion and discussion

In this paper, we construct a 3DGS map of outdoor large-scale environment based on the gassuion splatting technique of multi-sensor data, and combine the Superpoint and Superglue algorithms to accomplish efficient retrieval of sparse basemap library for RGB query images under large viewing angle deviation and reliable feature point matching, and utilize the rendering characteristics of 3DGS map to design the relocated iterative approximation structure to achieve High-precision relocation effect. We tested our method on numbers of sequences data from a variety of representative scenarios in the KITTI dataset, and all of them achieved centimeter-level localization accuracy, which is significantly improved compared to existing methods.

However, during the experiment, we found that because the multi-sensor data in the KITTI dataset is unidirectional trajectory data collected along the vehicle's driving route, the range of viewpoints from which the 3DGS maps constructed in this case can be rendered with high fidelity is limited, and thus if there is a significant deviation of the angle in the iterative process, it will result in a significant degradation of the quality of the 3DGS rendered maps, which will affect the convergence of the iterations. accuracy of the iterative convergence [45,46,47]. In future research, we can further explore how to make 3DGS maps rendered with high fidelity from a more free viewing angle in the case of limited data acquisition, and how to remove the sensitivity of the repositioning iteration process to angular errors.

## Funding Information

This research was supported by the National Natural Science Foundation of China under Grant No. 42374016.

## References

[1] Bresson, G., Alsayed, Z., Yu, L., & Glaser, S. (2017). Simultaneous Localization and Mapping: A Survey of Current Trends in Autonomous Driving. IEEE Transactions on Intelligent Vehicles, 2, 194-220.

[2] Chalvatzaras, A., Pratikakis, I., & Amanatiadis, A. (2023). A Survey on Map-Based Localization Techniques for Autonomous Vehicles. IEEE Transactions on Intelligent Vehicles, 8, 1574-1596.

[3] Kuutti, S., Fallah, S., Katsaros, K., Dianati, M., Mccullough, F., & Mouzakitis, A. (2018). A Survey of the State-of-the-Art Localization Techniques and Their Potentials for Autonomous Vehicle Applications. IEEE Internet of Things Journal, 5, 829-846.

[4] Lu, Y., , H., Smart, E., & Yu, H. (2022). Real-Time Performance-Focused Localization Techniques for Autonomous Vehicle: A Review. IEEE Transactions on Intelligent Transportation Systems, 23, 6082-6100.

[5] Javanmardi, E., Gu, Y., Javanmardi, M., & Kamijo, S. (2019). Autonomous vehicle self-localization based on abstract map and multi-channel LiDAR in urban area. IATSS Research.

[6] Wang, K., Zhao, G., & Lu, J. (2024). A Deep Analysis of Visual SLAM Methods for Highly Automated and Autonomous Vehicles in Complex Urban Environment. IEEE Transactions on Intelligent Transportation Systems, 25, 10524-10541.

[7] Luo, L., Cao, S., Han, B., Shen, H., & Li, J. (2021). BVMatch: Lidar-Based Place Recognition Using Bird's-Eye View Images. IEEE Robotics and Automation Letters, 6, 6076- 6083.

[8] Hu, Y., Li, S., Weng, W., Xu, K., & Wang, G. (2023). NSAW: An Efficient and Accurate Transformer for Vehicle LiDAR Object Detection. IEEE Transactions on Instrumentation and Measurement, 72, 1-10.

[9] Andrew, Q. (2024). Adaptive bird's eye view description for long-term mapping and loop closure in 3D point clouds. Applied and Computational Engineering.

[10] Khoo, H., Ng, Y., & Tan, C. (2024). Optimized Received Signal Strength-Based Radio Map Interpolation for Indoor Positioning Systems. Journal of Cases on Information Technology.

[11] Zhou, C., Li, Z., Zeng, D., & Wang, Y. (2021). Mining Geometric Constraints From Crowd-Sourced Radio Signals and its Application to Indoor Positioning. IEEE Access, 9, 46686-46697.

[12] Brachmann, E., Krull, A., Nowozin, S., Shotton, J., Michel, F., Gumhold, S., & Rother, C. (2017). Dsac-differentiable ransac for camera localization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 6684-6692).

[13] Laskar, Z., Melekhov, I., Kalia, S., & Kannala, J. (2017). Camera relocalization by computing pairwise relative poses using convolutional neural network. In Proceedings of the IEEE international conference on computer vision workshops (pp. 929-938).

[14] Sarlin, P. E., Unagar, A., Larsson, M., Germain, H., Toft, C., Larsson, V., ... & Sattler, T. (2021). Back to the feature: Learning robust camera localization from pixels to pose. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 3247-3257).

[15] Saputra, M. R. U., Markham, A., & Trigoni, N. (2018). Visual SLAM and structure from motion in dynamic environments: A survey. ACM Computing Surveys (CSUR), 51(2), 1-36.

[16] Werner, D., Al-Hamadi, A., & Werner, P. (2014). Truncated signed distance function: experiments on voxel size. In Image Analysis and Recognition: 11th International Conference, ICIAR 2014, Vilamoura, Portugal, October 22-24, 2014, Proceedings, Part II 11 (pp. 357-364). Springer International Publishing.

[17] Wolcott, R. W., & Eustice, R. M. (2014, September). Visual localization within lidar maps for automated urban driving. In 2014 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 176-183). IEEE.

[18] Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2021). Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1), 99-106.

[19] Yen-Chen, L., Florence, P., Barron, J. T., Rodriguez, A., Isola, P., & Lin, T. Y. inerf: Inverting neural radiance fields for pose estimation. In 2021 IEEE. In RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 1323-1330).

[20] Maggio, D., Abate, M., Shi, J., Mario, C., & Carlone, L. (2023, May). Loc-nerf: Monte carlo localization using neural radiance fields. In 2023 IEEE International Conference on Robotics and Automation (ICRA) (pp. 4018-4025). IEEE.

[21] Sucar, E., Liu, S., Ortiz, J., & Davison, A. J. (2021). imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6229-6238).

[22] Matsuki, H., Murai, R., Kelly, P. H., & Davison, A. J. (2024). Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 18039-18048).

[23] Sun, Y., Wang, X., Zhang, Y., Zhang, J., Jiang, C., Guo, Y., & Wang, F. (2023). iComMa: Inverting 3D Gaussian Splatting for Camera Pose Estimation via Comparing and Matching. arXiv preprint arXiv:2312.09031.

[24] Zhai, H., Zhang, X., Zhao, B., Li, H., He, Y., Cui, Z., ... & Zhang, G. (2024). Splatloc: 3d gaussian splatting-based visual localization for augmented reality. arXiv preprint arXiv:2409.14067.

[25] Xin, Z., Dai, C., Li, Y., & Wu, C. (2024, October). GauLoc: 3D Gaussian Splattingâbased Camera Relocalization. In Computer Graphics Forum (Vol. 43, No. 7, p. e15256).

[26] Botashev, K., Pyatov, V., Ferrer, G., & Lefkimmiatis, S. (2024, October). GSLoc: Visual Localization with 3D Gaussian Splatting. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 5664-5671). IEEE.

[27] Zeller, A. J. (2024). GSplatLoc: Ultra-Precise Camera Localization via 3D Gaussian Splatting. arXiv preprint arXiv:2412.20056.

[28] Chen, Z., Yang, J., Huang, J., de Lutio, R., Esturo, J. M., Ivanovic, B., ... & Wang, Y. (2024). Omnire: Omni urban scene reconstruction. arXiv preprint arXiv:2408.16760.

[29] Chen, Y., Gu, C., Jiang, J., Zhu, X., & Zhang, L. (2023). Periodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering. arXiv preprint arXiv:2311.18561.

[30] Yang, J., Ivanovic, B., Litany, O., Weng, X., Kim, S. W., Li, B., ... & Wang, Y. (2023). Emernerf: Emergent spatialtemporal scene decomposition via self-supervision. arXiv preprint arXiv:2311.02077.

[31] Gupta, M., & Singh, P. (2021, July). An image forensic technique based on SIFT descriptors and FLANN based matching. In 2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT) (pp. 1-7). IEEE.

[32] Noble, F. K. (2016, November). Comparison of OpenCV's feature detectors and feature matchers. In 2016 23rd

International Conference on Mechatronics and Machine Vision in Practice (M2VIP) (pp. 1-6). IEEE.

[33] DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops (pp. 224- 236).

[34] Sarlin, P. E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4938-4947).

[35] Shi, Y., Cai, J. X., Shavit, Y., Mu, T. J., Feng, W., & Zhang, K. (2022). Clustergnn: Cluster-based coarse-to-fine graph neural network for efficient feature matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 12517-12526).

[36] Ma, J., Jiang, X., Fan, A., Jiang, J., & Yan, J. (2021). Image matching from handcrafted to deep features: A survey. International Journal of Computer Vision, 129(1), 23- 79.

[37] Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., ... & Bowman, S. (2019). Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32.

[38] Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). EP n P: An accurate O (n) solution to the P n P problem. International journal of computer vision, 81, 155-166.

[39] Besl, P. J., & McKay, N. D. (1992, April). Method for registration of 3-D shapes. In Sensor fusion IV: control paradigms and data structures (Vol. 1611, pp. 586-606). Spie.

[40] Wang, J., Rupprecht, C., & Novotny, D. (2023). Posediffusion: Solving pose estimation via diffusion-aided bundle adjustment. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 9773-9783).

[41] Sattler, T., Leibe, B., & Kobbelt, L. (2016). Efficient & effective prioritized matching for large-scale image-based localization. IEEE transactions on pattern analysis and machine intelligence, 39(9), 1744-1756.

[42] Brachmann, E., & Rother, C. (2019). Neural-guided RANSAC: Learning where to sample model hypotheses. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4322-4331).

[43] Brachmann, E., & Rother, C. (2018). Learning less is more-6d camera localization via 3d surface regression. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4654-4662).

[44] Jiang, P., Pandey, G., & Saripalli, S. (2024). 3dgs-reloc: 3d gaussian splatting for map representation and visual relocalization. arXiv preprint arXiv:2403.11367.

[45] Zhang, S., Liu, Z., Xu, B., Wang, J., & Li, Y. (2025). Fusion GNSS/INS/Vision with Path Planning Prior for High Precision Navigation in Complex Environment. IEEE Sensors Journal.

[46] Xu, B., Zhang, S., Kuang, K., & Li, X. (2023). A unified cycleslip, multipath estimation, detection and mitigation method for VIO-aided PPP in urban environments. GPS Solutions, 27(2), 59.

[47] Wang, J., Liu, J., Zhang, S., Xu, B., Luo, Y., & Jin, R. (2023). Sky-view images aided NLOS detection and suppression for tightly coupled GNSS/INS system in urban canyon areas. Measurement Science and Technology, 35(2), 025112.