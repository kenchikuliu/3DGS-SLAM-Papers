# 4DRadar-GS: Self-Supervised Dynamic Driving Scene Reconstruction with 4D Radar

Xiao Tang1, Guirong Zhuo1, Cong Wang2, Boyuan Zheng1, Minqing Huang1, Lianqing Zheng1, Long Chen2, Shouyi Lu1â

Abstractâ 3D reconstruction and novel view synthesis are critical for validating autonomous driving systems and training advanced perception models. Recent self-supervised methods have gained significant attention due to their cost-effectiveness and enhanced generalization in scenarios where annotated bounding boxes are unavailable. However, existing approaches, which often rely on frequency-domain decoupling or optical flow, struggle to accurately reconstruct dynamic objects due to imprecise motion estimation and weak temporal consistency, resulting in incomplete or distorted representations of dynamic scene elements. To address these challenges, we propose 4DRadar-GS, a 4D Radar-augmented self-supervised 3D reconstruction framework tailored for dynamic driving scenes. Specifically, we first present a 4D Radar-assisted Gaussian initialization scheme that leverages 4D Radarâs velocity and spatial information to segment dynamic objects and recover monocular depth scale, generating accurate Gaussian point representations. In addition, we propose a Velocity-guided PointTrack (VGPT) model, which is jointly trained with the reconstruction pipeline under scene flow supervision, to track fine-grained dynamic trajectories and construct temporally consistent representations. Evaluated on the OmniHD-Scenes dataset, 4DRadar-GS achieves state-of-the-art performance in dynamic driving scene 3D reconstruction.

## I. INTRODUCTION

Closed-loop simulation of driving scenarios is of significant importance for the testing of autonomous driving functions and the training of large driving models [1], [2], [3]. Recent advancements in 3D reconstruction and novel view synthesis techniques have introduced new research opportunities in the field of closed-loop simulation. Starting from 2D images of real-world scenes, these 3D reconstruction methods enable the reconstruction of realistic street views. This approach can substantially reduce the sim-to-real gap compared to simulators based on virtual engines.

3D Gaussian Splatting [4] is an efficient method for 3D reconstruction and novel view synthesis, utilizing Gaussian ellipsoids to represent the geometric structures of a 3D scene. It has demonstrated robust performance in reconstructing static indoor environments or objects. However, driving scenarios pose greater challenges due to their high dynamism, incorporating moving vehicles and pedestrians. To enhance the dynamic reconstruction of driving scenes, some approaches have proposed leveraging LiDAR to provide initial point cloud inputs and using 3D bounding boxes [5], [6], [7] to decouple moving actors from the scene for separate modeling. The dynamic-static decoupling strategy effectively mitigates motion blur issues. Nonetheless, obtaining accurate bounding boxes is challenging and costly, prompting exploration into self-supervised dynamic-static decoupling reconstruction methods. These methods attempt to reconstruct scenes using frequency domain decoupling [8], [9], semantic masks, or estimated optical flow [10] to enable dynamic driving scene reconstruction without requiring bounding boxes. However, these methods suffer from false or incomplete detections of dynamic targets (e.g., misclassifying stationary roadside vehicles as dynamic), which hinders effective dynamic-static decoupling. Furthermore, their motion modeling approach for dynamic points struggles to establish accurate inter-frame correspondence for dynamic actors, particularly in the presence of rapid ego or actor motion, leading to a suboptimal reconstruction of the dynamic elements within the scene.

Recently, 4D Radar has garnered significant attention from both academia and industry due to its exceptional capabilities in dynamic object perception. Leveraging its precise spatial localization and velocity measurement abilities, 4D Radar can effectively capture the motion information of dynamic objects. This is particularly advantageous in complex driving scenarios, where it can overcome the limitations faced by traditional sensors, such as LiDAR and cameras, in highly dynamic environments.

To address these challenges, we propose 4DRadar-GS, a dynamic reconstruction method assisted by 4D Radar. For the first challenge, we propose a 3D perception initialization scheme for the deformation field, used to directly learn a deformation field dedicated to dynamic objects from monocular driving videos. Specifically, we first utilize the dynamic perception capabilities of 4D Radar to propose a dynamic segmentation model, to accurately decouple the scene into a static background and a dynamic foreground. Secondly, utilizing the spatial localization capabilities of 4D Radar, we recover the scale of monocular depth estimation, and use the dynamic depth points as the initialization for the dynamic foreground. For the second challenge, we train a Velocity-guided PointTrack (VGPT) model to associate dynamic points across multiple frames. Existing methods often depend solely on two-dimensional flow estimation for supervision, which may deteriorate results due to inadequate multi-view consistency. By leveraging the radial velocity information from 4D Radar, we provide the VGPT network with additional radial supervision alongside optical flow supervision, significantly aiding the learning of subsequent four-dimensional representations and enhancing the rendering quality of dynamic actors. Additionally, to address frameto-frame inconsistencies in point cloud initialization, we devised a simple yet powerful regularization technique. During training, we randomly remove Gaussian points, allowing Gaussians that are obscured by nearby Gaussian points to receive attention under sparse view conditions, thereby alleviating the issue of overfitting to training perspectives.

In summary, our key contributions can be listed as follows:

â¢ We propose a novel self-supervised reconstruction framework, the first to systematically leverage 4D Radar to perform accurate dynamic decoupling and scale recovery, thereby achieving robust initialization for dynamic driving scenes.

â¢ We introduce a VGPT model to establish robust temporal correspondence for dynamic actors, training it with a dualsupervision scheme that complements optical flow with direct physical constraint from 4D Radarâs radial velocity.

â¢ We devised a regularization method to mitigate the issue of overfitting to training viewpoints. Our approach achieves SOTA performance on the OmniHD-Scenes dataset, which was selected due to its essential 4D Radar data.

## II. RELATED WORKS

## A. Static Scene Reconstruction with 3DGS

3D Gaussian Splatting [4] has emerged as a leading method for high-fidelity novel view synthesis, using millions of Gaussian primitives to achieve real-time rendering of static scenes. Following its success, initial works focused on improving its efficiency and scalability for large-scale static environments. For instance, GaussianPro [11] introduced a progressive training scheme to enhance quality, while Octree-GS [12] employed an octree-based structure to manage Gaussians more efficiently. Subsequent works have extended 3DGS to large-scale urban environments; for instance, Hierarchical-GS [13] achieves this by incorporating a hierarchical structure. Other approaches further optimize for large scenes by dividing the point cloud into cells and introducing Level-of-Detail representations [14], [15].

## B. Dynamic Scene Reconstruction with 3DGS

A significant research thrust involves extending 3DGS from static to dynamic scenes, typically by employing deformation fields to model temporal changes, as seen in methods like Deformable-GS [16] and 4D-GS [17]. In the challenging context of autonomous driving, these techniques have been successfully applied, often with the aid of manual supervision. StreetGS [5] and DrivingGaussian [18] first proposed a composite dynamic Gaussian framework to model multiple rigid moving objects, while OmniRe [6] extended this capability to non-rigid pedestrians using SMPL models [19]. A critical drawback of these methods is their heavy reliance on extensive manual annotations, which causes their reconstruction performance to degrade significantly in the presence of imprecise labels, thereby challenging their widespread application in-the-wild scenarios.

## C. Self-Supervised Scene Rendering

To eliminate this dependency on manual labels, a more challenging yet highly generalizable paradigm of selfsupervised reconstruction has been explored. S3Gaussian [20] implicitly models object trajectories through a spatiotemporal decomposition network and jointly optimizes the static background and dynamic objects using various selfsupervised signals, achieving photorealistic reconstruction of dynamic urban scenes in an annotation-free manner. Meanwhile, PVG [8] proposed a unified model of Periodically Vibrating Gaussians to represent diverse objects and elements, constructing long-term trajectories by linking segments that exhibit periodic motion. DeSiRe-GS [9] extracts 2D motion masks by exploiting the poor reconstruction quality of standard 3D Gaussians in dynamic regions and introduces a temporal cross-view consistency constraint. Nevertheless, these methods often fail to establish continuous temporal correspondence for moving objects, particularly during rapid motion, leading to severe artifacts in novel view synthesis. This is a key problem that our work aims to address by introducing novel physical priors from 4D Radar.

## III. METHODS

Our proposed framework, 4DRadar-GS, reconstructs dynamic driving scenes through the two-stage pipeline illustrated in Figure 1. We first detail our 4D Radar-assisted initialization strategy in Section III-A, where we leverage the dynamic perception and spatial localization capabilities of 4D Radar to decouple the scene into static and dynamic Gaussian primitives. We then describe the joint training network in Section III-B, which models the motion of these dynamic primitives using a VGPT model. Finally, the complete set of objective functions and regularization techniques used for optimization are presented in Section III-C.

## A. 4D Radar-Assisted Gaussian Initialization

For the self-supervised reconstruction of autonomous driving scenarios, we generate depth map and dynamic mask by leveraging the dynamic perception capabilities of 4D Radar for dynamic segmentation, and its spatial localization capabilities for scale recovery. This process enables efficient partitioning of all Gaussians into two categories: object Gaussians $\varPhi _ { o b j }$ and background Gaussians $\varPhi _ { b k g }$

4D Radar-Camera Dynamic Segmentation Model. Unlike methods based on LiDAR or SfM which struggle to directly decouple dynamic elements, 4D Radar offers the inherent advantage of providing velocity information. We leverage this by employing a 4D Radar-assisted initialization scheme that identifies dynamic objects efficiently without requiring annotated bounding box information.

As illustrated in Figure 2, our model predicts a dynamic segmentation mask $\tilde { M } \in \tilde { R _ { + } ^ { H \times W } }$ by fusing a single RGB image $I ~ \in ~ \mathbb { R } ^ { 3 \times H \times W }$ with its corresponding 4D Radar point cloud $P ~ = ~ \{ p _ { i } | p _ { i } ~ \in ~ \mathbb { R } ^ { 4 } , i ~ = ~ 0 , 1 , 2 , \cdot \cdot \cdot , k ~ - ~ 1 \}$ First, we estimate and compensate for the ego-vehicleâs motion using the RANSAC algorithm [21], enabling the precise identification of dynamic 4D Radar points. From these dynamic points, we first randomly sample a fixed number to serve as anchors. These anchors are then projected onto the image plane, and for each projected point, we define its Region of Interest (ROI) by extracting a square image patch $\bar { C } _ { i } \in \mathbb { R } ^ { 3 \times h \times w }$ , centered at that location. These patches are subsequently processed by a ResNet [22] backbone to extract multi-scale image features. Concurrently, the features of the 4D Radar points associated with each image patch, specifically their position and velocity attributes, are meanpooled into a single vector. This vector is then processed through a fully-connected layer to match the dimensionality of the corresponding image features. The features from both modalities are then deeply fused through several layers of self-attention and cross-attention modules [23]. Finally, the fused features are passed to a U-Net-style decoder [24] to generate a high-resolution dynamic probability map for each ROI, denoted as ${ \hat { y } } _ { i } = h _ { \theta } ( C _ { i } , p _ { i } ) \in [ 0 , 1 ] ^ { h \times w }$ . Finally, the definitive dynamic mask MË $\in \hat { R } _ { + } ^ { H \times \tilde { W } }$ is obtained by reassembling all predicted patches into their corresponding regions in the image and then classifying each region as dynamic or static based on the prediction scores and a predefined threshold Ï . This process can be summarized as:

<!-- image-->  
Fig. 1. Overview of 4DRadar-GS Framework. Our pipeline consists of two main stages. (Left) Gaussian Initialization Assisted by 4D Radar. This stage generates initial Gaussian points from 4D Radar-corrected depth and separates them into static and dynamic components using a 4D Radar-guided segmentation mask. (Right) 3DGS VGPT Joint Training Network. Here, a VGPT model maps dynamic Gaussians to a canonical space to model their motion. These, along with static Gaussians, form a complete Gaussian scene graph, jointly optimized for high-fidelity rendering, depth, and mask outputs.

<!-- image-->  
Fig. 2. Architecture of the 4D Radar-Camera Dynamic Segmentation Model. The model fuses features from image patches and 4D Radar points using an attention-based mechanism to produce a high-resolution dynamic object mask.

$$
\hat { M } ( u , v ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } \hat { y } _ { m a x } ( u , v ) > \tau } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{1}
$$

The final mask is obtained by compositing the probability maps from all ROIs into a global confidence map. Where these maps overlap, the resulting confidence is determined by the maximum value across all predictions covering a given pixel. Specifically, the value for each pixel $( u , v )$ is determined by the ROI identified by arg max $\hat { y } _ { i } ( u , v )$

Monocular Depth Estimation and Scale Recovery. Monocular depth estimation algorithms, such as DepthAnythingV2 [25] employed in this work, inherently recover the relative depth of a scene, thus lacking a true physical scale. To resolve this inherent scale ambiguity, we leverage 4D Radar to perform scale recovery. Our core idea is to perform cross-modal data association on the surface of a unit sphere by projecting two sets of points onto it: the visual 3D point cloud, obtained by back-projecting the single-frame depth map, and the static 4D Radar point cloud, isolated after ego-motion compensation. Given the more uniform spatial distribution of visual points on the unit sphere, we construct a KD-tree from them to efficiently search for the three nearest visual neighbors for each 4D Radar point on the sphereâs surface.

Subsequently, we use a geometric constraint to estimate the scale factor. For each 4D Radar point $p _ { i }$ , its three nearest visual neighbors in the camera coordinate system $p _ { a } , p _ { b } , p _ { c } ,$ define a local physical plane. The normal vector for this plane is given by $n = ( p _ { a } - p _ { b } ) \times ( p _ { b } - p _ { c } )$ . By projecting both the 4D Radar point and a visual neighbor onto this normal, we can compute the scale factor $s _ { i }$ for an individual 4D Radar point $p _ { i }$ as: $\begin{array} { r } { s _ { i } \ = \ \frac { n \cdot p _ { i } } { n \cdot p _ { a } } } \end{array}$ . To ensure the validity of this estimation, we introduce two constraints. First, the method presupposes that the visual points lie on a common physical plane. This assumption is considered valid only when these points are sufficiently close to each other in their spherical projection and also have similar depth values in the reference depth map. Only then is the corresponding scale estimate considered reliable. Second, since depth estimates for dynamic objects are often inaccurate, dynamic 4D Radar points are excluded from this scale calculation. Finally, we statistically aggregate the scale factors computed from all valid 4D Radar points. The globally optimal scale is then determined using a robust histogram-based voting method.

## B. Joint Training Framework for 3DGS and VGPT

To address the issue of inaccurate dynamic association, we introduce a direct and robust physical constraint for the deformation field by leveraging the radial velocity information from 4D Radar. Furthermore, to address the computational redundancy in existing methods [8], we depart from the strategy of training a global deformation field across the entire scene and instead propose a sparse, object-centric strategy, exclusively constructing and optimizing the deformation field within the dynamic regions identified by our segmentation module. Consequently, our model can be trained directly on high-resolution images, preserving the fine-grained details that are essential for high-fidelity scene reconstruction.

<!-- image-->  
Fig. 3. Overview of the VGPT Pipeline and its Dual Supervision. The pipeline (top) models motion by warping points from source time $T _ { i }$ to target time $\bar { T } _ { j }$ via canonical space. This process is supervised by two signals (bottom): a geometric consistency loss derived from optical flow, and a direct physical constraint based on 4D Radarâs radial velocity.

Deformation Field Modeling. For the spatio-temporal modeling of dynamic elements, we introduce a timedependent deformation field $\mathcal { D } _ { t } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 }$ , which warps point clouds from all timestamps into a common space to initialize a set of canonical Gaussian primitives. To render image at arbitrary time t, these canonical Gaussians are transformed by the forward deformation field to their respective spatial locations for that time. They are then rendered from any given viewpoint using the standard 3DGS pipeline.

Our deformation field $\mathcal { D } _ { t } .$ , inspired by Tracking everything [26], is implemented as an invertible Multilayer Perceptron network [27]. The network takes a 3D coordinate point $x \in \mathbb { R } ^ { 3 }$ and a normalized timestamp $t \in [ 0 , 1 ]$ as input to predict the pointâs new position $\boldsymbol { x } _ { t } ~ \in ~ \mathbb { R } ^ { 3 }$ Owing to its invertible architecture, both the forward deformation $\mathcal { D } _ { t }$ and its inverse $\mathcal { D } _ { t } ^ { - 1 }$ can be analytically derived from a single forward pass, obviating the need for extra networks or iterative optimization. A single set of MLP weights is shared across all timestamps to ensure temporal continuity.

Deformation Field Supervision. To optimize the deformation field, we use two complementary supervisory signals, as shown in Figure 3: (1) a pseudo-ground-truth 3D scene flow derived from optical flow, (2) a direct physical constraint based on the radial velocities provided by the 4D Radar.

a) Optical Flow-based 3D Scene Flow Supervision: Given a pair of images at timestamps $t _ { i }$ and $t _ { j } ,$ we first employ a pre-trained optical flow model [28] to estimate the dense 2D optical flow field between them. As depicted in Figure 3, this 2D optical flow field establishes pixel-level correspondences for the dynamic Gaussian points across the two frames. Combined with the depth information provided by our monocular depth estimation network, these 2D correspondences are then lifted to sparse 3D correspondences between the two point clouds. This process generates a pseudo-ground-truth 3D scene flow. We leverage this 3D flow to supervise the learning of the deformation field D. Specifically, for any given dynamic point $\boldsymbol { x } _ { t _ { i } }$ , its corresponding target position $\hat { x } _ { t _ { j } } = \mathcal { D } _ { t _ { j } } \circ \mathcal { D } _ { t _ { i } } ^ { - 1 } ( x _ { t _ { i } } )$ at time $t _ { j }$ should be close to $\boldsymbol { x } _ { t _ { j } }$ . The deformation field is then optimized by minimizing the following geometric consistency loss:

$$
\mathcal { L } _ { \mathrm { f l o w } } = \sum \left\| \mathcal { D } _ { t _ { j } } \circ \mathcal { D } _ { t _ { i } } ^ { - 1 } \left( x _ { t _ { i } } \right) - x _ { t _ { j } } \right\| ^ { 2 } .\tag{2}
$$

b) Physical Constraints Based on 4D Radar Radial Velocity: The ability of 4D Radar to directly measure the Radial Relative Velocity (RRV) of objects provides a physical prior that we leverage for direct supervision of our deformation field. First, we associate our segmented 3D dynamic Gaussians with the dynamic 4D Radar points at each timestamp using a K Nearest Neighbors (KNN) algorithm [29]. By setting K=1, we find the single nearest 4D Radar point for each Gaussian, from which the Gaussian inherits the ground-truth radial velocity measurement $v ^ { r } .$ Assuming uniform motion over a small time interval $\Delta t ,$ the radial displacement between consecutive frames can be described by the geometric relationship shown in Figure 3:

$$
v ^ { r } \Delta t = F _ { g t } ^ { \top } { \frac { \left( x _ { t _ { i } } - A _ { t _ { i } } \right) } { \left\| x _ { t _ { i } } - A _ { t _ { i } } \right\| _ { 2 } } } .\tag{3}
$$

Here, $A _ { t _ { i } }$ is the 4D Radar center, and $F _ { g t } = T ^ { - 1 } x _ { t _ { j } } - x _ { t }$ is the flow vector of the ground truth scene for the point $\boldsymbol { x } _ { t _ { i } }$ , where $T ^ { - 1 }$ is the transformation matrix from the coordinate frame at time $t _ { j }$ to the frame at time $t _ { i }$ . The projection of this vector onto the viewing direction must equal the radial displacement. Based on this physical constraint, we formulate a radial displacement loss $\mathcal { L } _ { r a d } .$ , to directly supervise the radial component of the flow vector $F _ { i } ,$ generated by the deformation field. The predicted flow is defined as $F _ { i } = T ^ { - 1 } \cdot \hat { x } _ { t _ { j } } - x _ { t _ { i } }$ . And the loss is:

$$
\mathcal { L } _ { r a d } = \sum \left\| F _ { i } ^ { \top } \frac { ( x _ { t _ { i } } - A _ { t _ { i } } ) } { \| ( x _ { t _ { i } } - A _ { t _ { i } } ) \| _ { 2 } } - v ^ { r } \Delta t \right\| .\tag{4}
$$

Despite certain unavoidable measurement errors, we empirically find that the RRV provides a strong supervisory signal, as demonstrated in Section IV-C.

## C. Training and Optimization Strategies

Gaussian Dropout Regularization. Potential inconsistencies in the 4D Radar-fused depth scales can lead to a critical reconstruction artifact: correctly positioned Gaussians are erroneously occluded by others. To mitigate this representational occlusion, we introduce a Gaussian dropout regularization method, inspired by the concept of Dropout in deep learning. During each training iteration, we stochastically set the opacity of a subset of Gaussian primitives to zero with a certain probability. This random dropping mechanism disrupts the established occlusion patterns in any given view, allowing Gaussians that would otherwise be occluded by foreground points to become exposed and receive supervision even under sparse view conditions. This approach effectively mitigates overfitting to specific training views and enhances the geometric consistency of novel view synthesis.

<!-- image-->  
PVG

<!-- image-->  
AD-GS

<!-- image-->  
DeSiRe-GS

<!-- image-->  
Ours

<!-- image-->  
Ground Truth  
Fig. 4. Qualitative comparison of novel view synthesis. The red boxes highlight the reconstruction quality of dynamic vehicles. While prior methods suffer from severe artifacts like motion blur and ghosting, our method produces sharp and coherent reconstructions that are highly consistent with the ground truth.

TABLE I  
QUANTITATIVE COMPARISON RESULTS ON OMNIHD-SCENES DATASET. THE IMAGE RESOLUTION IS 1920 Ã 1080. LPIPS UNIFORMLY ADOPTS THE VGG-NET. PSNR\* FOR DYNAMIC OBJECTS. BOLD: BEST. UNDERLINE: SECOND BEST.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Venue</td><td rowspan="2">Type</td><td rowspan="2">BBox</td><td rowspan="2">3D Sensor</td><td colspan="4">Scene Reconstruction</td><td colspan="4">Novel View Synthesis</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNR*â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNR*â</td></tr><tr><td>3D-GS [4]</td><td>SIGGRAPH&#x27;23</td><td>Static</td><td>Ã</td><td></td><td>30.15</td><td>0.911</td><td>0.199</td><td>25.83</td><td>20.10</td><td>0.667</td><td>0.390</td><td>14.91</td></tr><tr><td>GaussianPro [11]</td><td>ICML&#x27;24</td><td>Static</td><td>Ã</td><td></td><td>27.84</td><td>0.867</td><td>0.253</td><td>22.24</td><td>20.68</td><td>0.675</td><td>0.395</td><td>15.71</td></tr><tr><td>Octree-GS [12]</td><td>TPAMI&#x27;25</td><td>Static</td><td>Ã</td><td></td><td>32.34</td><td>0.941</td><td>0.146</td><td>26.38</td><td>21.97</td><td>0.696</td><td>0.345</td><td>16.46</td></tr><tr><td>4D-GS [17]</td><td>CVPR&#x27;24</td><td>Dynamic</td><td>Ã</td><td>-</td><td>29.97</td><td>0.874</td><td>0.242</td><td>26.75</td><td>20.63</td><td>0.691</td><td>0.396</td><td>15.83</td></tr><tr><td>Deformable-GS [16]</td><td>CVPR&#x27;24</td><td>Dynamic</td><td></td><td>-</td><td>31.87</td><td>0.929</td><td>0.169</td><td>27.33</td><td>19.93</td><td>0.662</td><td>0.393</td><td>14.96</td></tr><tr><td>StreetGS [5]</td><td>ECCV&#x27;24</td><td>Dynamic</td><td></td><td>LiDAR</td><td>33.58</td><td>0.947</td><td>0.139</td><td>28.04</td><td>26.55</td><td>0.774</td><td>0.270</td><td>23.01</td></tr><tr><td>OmniRe [6]</td><td>ICLR&#x27;25</td><td>Dynamic</td><td>Ã&gt;&gt;</td><td>LiDAR</td><td>34.87</td><td>0.954</td><td>0.120</td><td>29.22</td><td>27.43</td><td>0.783</td><td>0.249</td><td>23.98</td></tr><tr><td>S3Gaussian [20]</td><td>arXiv&#x27;24</td><td>Dynamic</td><td>Ã</td><td>LiDAR</td><td>34.77</td><td>0.953</td><td>0.121</td><td>28.24</td><td>20.24</td><td>0.678</td><td>0.357</td><td>16.75</td></tr><tr><td>PVG [8]</td><td>arXiv&#x27;23</td><td>Dynamic</td><td>Ã</td><td>LiDAR</td><td>34.40</td><td>0.952</td><td>0.127</td><td>28.28</td><td>25.15</td><td>0.767</td><td>0.282</td><td>21.48</td></tr><tr><td>DeSiRe-GS [9]</td><td>CVPR&#x27;25</td><td>Dynamic</td><td>Ã</td><td>LiDAR</td><td>33.98</td><td>0.949</td><td>0.135</td><td>29.60</td><td>25.11</td><td>0.767</td><td>0.288</td><td>21.46</td></tr><tr><td>AD-GS [30]</td><td>ICCV&#x27;25</td><td>Dynamic</td><td>Ã</td><td>LiDAR</td><td>32.51</td><td>0.936</td><td>0.153</td><td>27.37</td><td>25.42</td><td>0.770</td><td>0.277</td><td>21.95</td></tr><tr><td>Ours</td><td></td><td>Dynamic</td><td>Ã</td><td>4D Radar</td><td>34.96</td><td>0.958</td><td>0.119</td><td>29.81</td><td>26.68</td><td>0.790</td><td>0.265</td><td>23.33</td></tr></table>

Three-Stage Training Strategy. To ensure stable convergence and effective decoupling of our scene representation, we devise a three-stage training strategy. The first stage involves training the Gaussian network for the static background alongside the deformation field intended for the dynamic objects. In the second stage, we freeze the parameters of the static model and jointly train the dynamic Gaussians with the deformation field. The third stage is a global fine-tuning phase where all modules are unfrozen for end-to-end joint optimization of the static Gaussians, dynamic Gaussians, and the deformation field.

Total Loss. The total training loss function is formulated as:

$$
\begin{array} { r } { \mathcal { L } = ( 1 - \lambda _ { r } ) \mathcal { L } _ { 1 } + \lambda _ { r } \mathcal { L } _ { s s i m } + \lambda _ { d } \mathcal { L } _ { d } + \lambda _ { o b j } \mathcal { L } _ { o b j } + \lambda _ { s k y } \mathcal { L } _ { s k y } , } \\ { \mathcal { L } = ( 1 - \lambda _ { r } ) \mathcal { L } _ { 1 } + \lambda _ { r } \mathcal { L } _ { s s i m } + \lambda _ { d } \mathcal { L } _ { d } + \lambda _ { o b j } \mathcal { L } _ { o b j } + \lambda _ { s k y } \mathcal { L } _ { s k y } , } \end{array}\tag{5}
$$

where the Î» terms are hyperparameters that weight each loss component. $\mathcal { L } _ { 1 }$ and $\mathcal { L } _ { s s i m }$ are the L1 and SSIM losses on rendered images. $\mathcal { L } _ { d }$ is an inverse depth supervision loss following PVG [8], supervised by our scale-recovered depth maps. We adopt a mask constraint loss $\begin{array} { r l } { \mathcal { L } _ { o b j } } & { { } = } \end{array}$ $B C E ( O _ { d y n } , M _ { d y n } )$ similar to that of StreetGS. This loss employs a binary cross-entropy term to align the rendered dynamic object mask $O _ { d y n }$ , with the foreground mask $M _ { d y n } .$ provided by our dynamic segmentation model. Following PVG, we employ a learnable environment map to represent the sky and apply a cross-entropy loss $\mathcal { L } _ { s k y } = B C E ( 1 -$ $O _ { s k y } , M _ { s k y } )$ to enforce opacity in sky regions. Here, $O _ { s k y } =$ $\textstyle \prod _ { i = 1 } ^ { N } \left( 1 - \alpha _ { i } \right)$ represents the accumulated transparency of all rendered Gaussians, and $M _ { s k y }$ is sky mask predicted by Grounded-SAM [31] model.

## IV. EXPERIMENTS

## A. Experimental Setup

Datasets. As widely-used autonomous driving benchmarks such as the KITTI [32], Waymo [33], and nuScenes [34] datasets lack 4D Radar sensors, we evaluate our model on the OmniHD-Scenes [35] dataset. This dataset is recorded at 10 Hz and is particularly suitable for our evaluation as it encompasses a rich variety of scenarios, ranging from rainy weather to clear days and from daytime to nighttime lighting, in addition to high-speed, low-speed, urban, and suburban settings. This variety is crucial for rigorously evaluating our modelâs robustness and its ability to generalize across different real-world conditions. From this dataset, we select 8 representative sequences for our experiments. Each sequence contains 50 frames, and we utilize data from the front-facing camera and front-facing 4D Radar. The camera resolution is downsampled to 1920Ã1080. For our train-test split, we assign every fourth frame of each sequence to the test set and use the remaining frames for training.

<!-- image-->  
Fig. 5. Comparison of dynamic object segmentation under significant egomotion.  
TABLE II

QUANTITATIVE COMPARISON OF DIFFERENT MASK TYPES ON SCENE RECONSTRUCTION TASK.
<table><tr><td>Mask type</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR* â</td></tr><tr><td>Our</td><td>34.96</td><td>0.958</td><td>0.119</td><td>29.81</td></tr><tr><td>SAM2 [36]</td><td>34.52</td><td>0.952</td><td>0.120</td><td>28.73</td></tr></table>

Baselines. We evaluate our method against several stateof-the-art self-supervised approaches, including S3Gaussian [20], PVG [8], DeSiRe-GS [9], and AD-GS [30]. For a more comprehensive evaluation, we also compare our results with OmniRe [6] and StreetGS [5], which are methods that require additional bounding box information for supervision.

Implementation Details. All experiments in this study were conducted on a single NVIDIA RTX 4090 GPU.

Segmentation Model Training: We use dynamic 4D Radar points projected onto the image as prompts for Grounded-SAM [31] to generate initial segmentation results for dynamic objects. From these, we manually select 11,973 highquality results to serve as the ground truth for training. The input image resolution is 1920Ã1080, with a patch size of 256Ã256. The model is trained for 50 epochs using the Adam optimizer. The key hyperparameters are set as follows: a learning rate of 2e-4, a batch size of 6, $\beta _ { 1 } ~ = ~ 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . Total training time is approximately 12 hours.

Gaussian Model Training: Our Gaussian model is trained for 30,000 iterations per sequence. The second stage of this training commences at 15,000 iterations, and the third stage begins at 20,000 iterations. The entire training process for each sequence takes approximately one hour to complete. We use learning rates similar to the original 3DGS implementation and set the loss weights as follows: $\lambda _ { r } = 0 . 2 , \lambda _ { d } =$ $0 . 1 , \lambda _ { s k y } = 0 . 0 5 , \lambda _ { o b j } = 0 . 0 5$ . The $\lambda _ { o b j }$ is only enabled in the third training stage. Sky masks are generated using Grounded-SAM [31] with a âskyâ prompt. Regarding initialization, unlike the baseline methods which adhere to their respective protocols for supplementing background points, our approach directly initializes all Gaussians from the dense depth map. This eliminates the need for any additional point supplementation; the only filtering step is to discard points corresponding to the sky mask.

<!-- image-->  
Fig. 6. Loss ablation by gradually adding the losses.  
TABLE III

ABLATION STUDY ON THE CONTRIBUTION OF EACH COMPONENT ONNOVEL VIEW SYNTHESIS TASK.
<table><tr><td>obj&amp;sky</td><td>depth</td><td>drop</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR* â</td></tr><tr><td></td><td></td><td></td><td>23.46</td><td>0.750</td><td>0.299</td><td>17.44</td></tr><tr><td>â</td><td></td><td></td><td>25.04</td><td>0.767</td><td>0.280</td><td>20.63</td></tr><tr><td>â</td><td>â</td><td></td><td>26.13</td><td>0.783</td><td>0.269</td><td>21.70</td></tr><tr><td>â</td><td>â</td><td>V</td><td>26.68</td><td>0.790</td><td>0.265</td><td>23.33</td></tr></table>

## B. Comparison

Comparison with Reconstructed Models. Following the evaluation protocol of PVG [8], we assess our method on two tasks: image reconstruction and novel view synthesis. The results are summarized in Table I, with the LPIPS metric computed using a VGG [37] backbone. In the self-supervised category, our method achieves SOTA performance across all rendering metrics for both reconstruction and synthesis tasks. The qualitative comparisons in Figure 4 further corroborate these findings: previous methods like PVG and DeSiRe-GS exhibit severe ghosting and artifacts on dynamic vehicles, while AD-GSâs reconstructions suffer from motion blur and noticeable noise. In contrast, by leveraging our effective point-wise tracking mechanism, our method generates sharp and coherent dynamic scenes, significantly outperforming existing self-supervised models in visual quality.

Notably, when compared to supervised methods that rely on additional bounding box annotations, our approach achieves highly competitive results without depending on any manual 3D labels. Its performance is on par with StreetGS [5] and approaches that of OmniRe [6].

Comparison with Segmentation Models. Our segmentation method demonstrates superior robustness to significant ego-motion compared to vision-only approaches. As illustrated in Figure 5, large camera displacements cause Grounded-SAM2 [36] to misclassify static vehicles, leading to over-segmentation. In contrast, our method leverages the physical velocity priors from 4D Radar, demonstrating exceptional robustness to this issue. It successfully isolates only genuinely moving vehicles, thereby enabling more accurate parsing of the dynamic scene.This improved accuracy is quantitatively validated by the results presented in Table II.

<!-- image-->  
Fig. 7. Ablation Study on the supervision for the VGPT Model.  
TABLE IV

ABLATION STUDY ON THE SUPERVISION OF OUR VGPT MODEL ON NOVEL VIEW SYNTHESIS TASK.
<table><tr><td></td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR* â</td></tr><tr><td>VGPT</td><td>25.02</td><td>0.762</td><td>0.305</td><td>18.38</td></tr><tr><td>VGPT+flow</td><td>25.97</td><td>0.775</td><td>0.273</td><td>22.29</td></tr><tr><td>VGPT+rad</td><td>25.58</td><td>0.770</td><td>0.285</td><td>20.93</td></tr><tr><td>all</td><td>26.68</td><td>0.790</td><td>0.265</td><td>23.33</td></tr></table>

## C. Ablation Study

Ablation of Gaussian Loss. We conduct an ablation study to validate the effectiveness of our modelâs components and their contributions to the novel view synthesis task. This study involves the progressive incorporation of several components atop a baseline model: the combined object and sky mask losses $\mathcal { L } _ { \mathrm { o b j } } + \mathcal { L } _ { \mathrm { s k y } }$ , the depth supervision loss $\mathcal { L } _ { d } ,$ and our dropout regularization. As demonstrated by the results in Table III and Figure 6, each component provides a distinct benefit: the 2D supervision from the mask losses is crucial for facilitating a clean spatial separation of object and background Gaussians; the 3D geometric constraints from the depth loss further enhance the accuracy of structural reconstruction; and finally, the Gaussian dropout regularization improves geometric robustness by effectively suppressing artifacts and ensuring greater cross-view consistency.

Ablation of VGPT Loss. To validate the effectiveness of the deformation field supervision in our proposed VGPT model, we incrementally add the network, $\mathcal { L } _ { f l o w }$ and ${ \mathcal { L } } _ { r a d } ,$ corresponding to âtrackâ, âflowâ, and âradâ, respectively. As demonstrated by the results in Table IV and Figure 7, without the VGPT model, the model fails to account for the objectâs motion, resulting in extreme motion blur that renders the vehicleâs form nearly indecipherable. Introducing the tracking network supervised solely by the optical flow-based loss substantially reduces this blur and recovers the vehicleâs general shape, yet significant ghosting artifacts and a lack of sharp details remain. Conversely, using only the 4D Radarbased radial displacement loss is insufficient to constrain the complex 3D motion, leading to severe geometric distortion and an uninterpretable reconstruction. By combining both supervisory signals, our full model synergistically leverages the dense 2D guidance from optical flow and the precise physical constraints from 4D Radar to generate a sharp, coherent, and artifact-free reconstruction.

Ablation of Dynamic Associations. We compare the association strategies of PVG [8] and AD-GS [30], conducting all experiments across four representative sequences under identical initialization and masking conditions to ensure a fair comparison. The results are presented in Table V and Figure 8. Our findings reveal that in reconstructing dynamic driving scenes, the novel view synthesis results from the two methods exhibit distinct characteristics and trade-offs. PVGâs reliance on a periodic vibration model makes it susceptible to correspondence errors, which in turn introduces dynamic artifacts. In contrast, AD-GS employs B-splines to enforce the temporal smoothness of keypoint trajectories. While this approach effectively mitigates the issue of association failure, its strong smoothing prior hinders the precise capture of subtle dynamics, leading to a loss of fine detail in the final reconstruction.

<!-- image-->  
Fig. 8. Qualitative comparison of different association strategies.  
TABLE V

QUANTITATIVE COMPARISON WITH BASELINE METHODS ON NOVEL VIEW SYNTHESIS TASK (EVALUATED ON 4 SEQUENCES).
<table><tr><td></td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR* â</td></tr><tr><td>PVG-based</td><td>25.92</td><td>0.800</td><td>0.240</td><td>19.03</td></tr><tr><td>AD-GS-based</td><td>27.04</td><td>0.808</td><td>0.231</td><td>21.48</td></tr><tr><td>Our</td><td>27.68</td><td>0.820</td><td>0.225</td><td>23.74</td></tr></table>

TABLE VI  
QUANTITATIVE COMPARISON OF INITIALIZATION METHODS ON SCENE RECONSTRUCTION TASK, EVALUATED WITHIN THE PVG FRAMEWORK.
<table><tr><td>initialize type</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>4D Radar</td><td>34.4011</td><td>0.9517</td><td>0.1274</td></tr><tr><td>LiDAR</td><td>34.3979</td><td>0.9521</td><td>0.1268</td></tr></table>

Ablation of Sensor. To fairly evaluate and compare the performance differences between 4D Radar and LiDAR as data sources for initialization, we conducted a set of controlled experiments within the PVG [8] framework. While keeping other experimental variables, such as the segmentation masks, strictly consistent, we initialized the Gaussian primitives using two different approaches: one based on pure LiDAR point clouds, and the other using our proposed method which fuses 4D Radar data with monocular depth estimates. A quantitative comparison of the two initialization strategies is presented in Table VI. As the results indicate, 4D Radar-based initialization achieves a level of accuracy comparable to that of the denser LiDAR point clouds.

## V. CONCLUSIONS

We present a self-supervised 3D Gaussian Splatting framework that, for the first time, leverages 4D Radar priors for high-fidelity dynamic driving scene reconstruction without manual annotations. Our method uses 4D Radar velocity for robust dynamic segmentation and its radial velocity as a direct physical constraint to supervise a deformation field alongside optical flow for accurate motion modeling. Experiments demonstrate state-of-the-art performance among selfsupervised dynamic driving scene reconstruction methods.

## REFERENCES

[1] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, âCarla: An open urban driving simulator,â in Conference on robot learning, 2017.

[2] R. Li, T. Li, H. Li, S. Li, and Z. Pan, âPerception data generation via synthesized 3d scenes and closed-loop optimization in autonomous driving simulation,â in Proceedings of the 2025 International Conference on Generative Artificial Intelligence and Digital Media, 2025, pp. 41â46.

[3] Y. Xiong, X. Zhou, Y. Wan, D. Sun, and M.-H. Yang, âDrivinggaussian++: Towards realistic reconstruction and editable simulation for surrounding dynamic driving scenes,â arXiv preprint arXiv:2508.20965, 2025.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics (TOG), vol. 42, no. 4, pp. 1â14, 2023.

[5] Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and S. Peng, âStreet gaussians for modeling dynamic urban scenes,â arXiv preprint arXiv:2401.01339, 2024.

[6] Z. Chen, J. Yang, J. Huang, R. de Lutio, J. M. Esturo, B. Ivanovic, O. Litany, Z. Gojcic, S. Fidler, M. Pavone et al., âOmnire: Omni urban scene reconstruction,â in The Thirteenth International Conference on Learning Representations, 2025.

[7] C. Wang, X. Guo, W. Xu, W. Tian, R. Song, C. Zhang, L. Li, and L. Chen, âDrivesplat: Decoupled driving scene reconstruction with geometry-enhanced partitioned neural gaussians,â arXiv preprint arXiv:2508.15376, 2025.

[8] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, âPeriodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering,â arXiv preprint arXiv:2311.18561, 2023.

[9] C. Peng, C. Zhang, Y. Wang, C. Xu, Y. Xie, W. Zheng, K. Keutzer, M. Tomizuka, and W. Zhan, âDesire-gs: 4d street gaussians for static-dynamic decomposition and surface reconstruction for urban driving scenes,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 6782â6791.

[10] S. Sun, C. Zhao, Z. Sun, Y. V. Chen, and M. Chen, âSplatflow: Selfsupervised dynamic gaussian splatting in neural motion flow field for autonomous driving,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 27 487â27 496.

[11] K. Cheng, X. Long, K. Yang, Y. Yao, W. Yin, Y. Ma, W. Wang, and X. Chen, âGaussianpro: 3d gaussian splatting with progressive propagation,â in International Conference on Machine Learning, 2024, pp. 8123â8140.

[12] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, âOctreegs: Towards consistent real-time rendering with lod-structured 3d gaussians,â IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[13] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and G. Drettakis, âA hierarchical 3d gaussian representation for real-time rendering of very large datasets,â ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1â15, 2024.

[14] J. Lin, Z. Li, X. Tang, J. Liu, S. Liu, J. Liu, Y. Lu, X. Wu, S. Xu, Y. Yan et al., âVastgaussian: Vast 3d gaussians for large scene reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5166â5175.

[15] Y. Liu, C. Luo, L. Fan, N. Wang, J. Peng, and Z. Zhang, âCitygaussian: Real-time high-quality large-scale scene rendering with gaussians,â in European Conference on Computer Vision, 2025, pp. 265â282.

[16] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, âDeformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20 331â20 341.

[17] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20 310â20 320.

[18] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, âDrivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 634â21 643.

[19] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, âSmpl: a skinned multi-person linear model,â ACM Transactions on Graphics (TOG), vol. 34, no. 6, pp. 1â16, 2015.

[20] N. Huang, X. Wei, W. Zheng, P. An, M. Lu, W. Zhan, M. Tomizuka, K. Keutzer, and S. Zhang, âS3gaussian: Self-supervised street gaussians for autonomous driving,â arXiv preprint arXiv:2405.20323, 2024.

[21] S. Lu, G. Zhuo, H. Wang, Q. Zhou, H. Zhou, R. Huang, M. Huang, L. Zheng, and Q. Shu, âTdfanet: Encoding sequential 4d radar point clouds using trajectory-guided deformable feature aggregation for place recognition,â arXiv preprint arXiv:2504.05103, 2025.

[22] K. He, X. Zhang, S. Ren, and J. Sun, âDeep residual learning for image recognition,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770â778.

[23] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, âLoftr: Detectorfree local feature matching with transformers,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 8922â8931.

[24] O. Ronneberger, P. Fischer, and T. Brox, âU-net: Convolutional networks for biomedical image segmentation,â in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234â241.

[25] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, âDepth anything v2,â Advances in Neural Information Processing Systems, vol. 37, pp. 21 875â21 911, 2024.

[26] Q. Wang, Y.-Y. Chang, R. Cai, Z. Li, B. Hariharan, A. Holynski, and N. Snavely, âTracking everything everywhere all at once,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 795â19 806.

[27] L. Dinh, J. Sohl-Dickstein, and S. Bengio, âDensity estimation using real nvp,â arXiv preprint arXiv:1605.08803, 2016.

[28] Z. Teed and J. Deng, âRaft: Recurrent all-pairs field transforms for optical flow,â in European conference on computer vision. Springer, 2020, pp. 402â419.

[29] L. E. Peterson, âK-nearest neighbor,â Scholarpedia, vol. 4, no. 2, p. 1883, 2009.

[30] J. Xu, K. Deng, Z. Fan, S. Wang, J. Xie, and J. Yang, âAd-gs: Objectaware b-spline gaussian splatting for self-supervised autonomous driving,â arXiv preprint arXiv:2507.12137, 2025.

[31] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., âGrounded sam: Assembling open-world models for diverse visual tasks,â arXiv preprint arXiv:2401.14159, 2024.

[32] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, âVision meets robotics: The kitti dataset,â The international journal of robotics research, vol. 32, no. 11, pp. 1231â1237, 2013.

[33] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine et al., âScalability in perception for autonomous driving: Waymo open dataset,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 2446â2454.

[34] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, ânuscenes: A multimodal dataset for autonomous driving,â in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun 2020. [Online]. Available: http://dx.doi.org/10.1109/cvpr42600. 2020.01164

[35] L. Zheng, L. Yang, Q. Lin, W. Ai, M. Liu, S. Lu, J. Liu, H. Ren, J. Mo, X. Bai et al., âOmnihd-scenes: A next-generation multimodal dataset for autonomous driving,â arXiv preprint arXiv:2412.10734, 2024.

[36] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson Â¨ et al., âSam 2: Segment anything in images and videos,â arXiv preprint arXiv:2408.00714, 2024.

[37] K. Simonyan and A. Zisserman, âVery deep convolutional networks for large-scale image recognition,â International Conference on Learning Representations,International Conference on Learning Representations, Jan 2015.