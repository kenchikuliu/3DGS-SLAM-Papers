# SplatPose: Geometry-Aware 6-DoF Pose Estimation from Single RGB Image via 3D Gaussian Splatting

Linqi Yang, Xiongwei Zhao, Qihao Sun, Ke Wang, Ao Chen, Peng Kang

Abstractâ 6-DoF pose estimation is a fundamental task in computer vision with wide-ranging applications in augmented reality and robotics. Existing single RGB-based methods often compromise accuracy due to their reliance on initial pose estimates and susceptibility to rotational ambiguity, while approaches requiring depth sensors or multi-view setups incur significant deployment costs. To address these limitations, we introduce SplatPose, a novel framework that synergizes 3D Gaussian Splatting (3DGS) with a dual-branch neural architecture to achieve high-precision pose estimation using only a single RGB image. Central to our approach is the Dual-Attention Ray Scoring Network (DARS-Net), which innovatively decouples positional and angular alignment through geometry-domain attention mechanisms, explicitly modeling directional dependencies to mitigate rotational ambiguity. Additionally, a coarseto-fine optimization pipeline progressively refines pose estimates by aligning dense 2D features between query images and 3DGSsynthesized views, effectively correcting feature misalignment and depth errors from sparse ray sampling. Experiments on three benchmark datasets demonstrate that SplatPose achieves state-of-the-art 6-DoF pose estimation accuracy in single RGB settings, rivaling approaches that depend on depth or multiview images.

## I. INTRODUCTION

6 -DOF pose estimation, which calculates the 3D positionand orientation of a camera in relation to objects or scenes, is fundamental in fields such as robotics and augmented reality. Although methods based on RGB-D data or point clouds [1], [2] have garnered significant attention, their reliance on depth sensors introduces significant limitations, including high hardware costs and sensitivity to challenging material properties (e.g., transparent or reflective surfaces). In contrast, monocular RGB-based approaches [3], [4] offer broader applicability but face inherent trade-offs between scalability, accuracy, and computational efficiency.

A critical challenge in existing RGB-based pose estimation frameworks lies in their dependency on resource-intensive data representations. For instance, instance-level object pose estimation methods [5], [6] require textured 3D models of target objects during training, limiting scalability. Neural Radiance Fields (NeRF) [7] pioneered scene reconstruction through differentiable volume rendering, enabling pose estimation via photometric optimization. However, NeRF-based methods suffer from two critical limitations: (1) Their implicit volumetric representation requires computationally intensive ray marching, making real-time applications impractical; (2) They demand dense multi-view images for training, which contradicts the single-image inference requirement in most pose estimation scenarios. Although variants like iNeRF [8] and Parallel iNeRF [9] attempt to address these issues, they remain heavily dependent on the initial pose and are susceptible to local minima.

Recent advances in 3D Gaussian Splatting (3DGS) [10], [11] have emerged as a paradigm shift, offering explicit scene modeling through anisotropic 3D Gaussian primitives. Unlike NeRFâs implicit volumetric approach, 3DGS supports real-time rendering by utilizing rasterization and retains photorealistic visual fidelity, making it particularly attractive for pose estimation tasks. However, methods like SplatLoc [12] and 3DGS-ReLoc [13] rely on dense depth data and multiview images to reconstruct scenes and retrieve the initial pose, resulting in substantial storage and data collection costs. Conversely, single RGB image-based methods, such as 6DGS [14], eliminate the need for depth sensors or multiframe databases by directly leveraging 3DGSâs differentiable rendering through rendering inversion. However, 6DGSâs Gaussian ellipsoid-based ray sampling strategy introduces rotational ambiguity due to its bias toward rays with minimal perpendicular distance to the cameraâs optical center, while neglecting angular offsets. These shortcomings highlight a fundamental trade-off: single RGB-based methods rely on initial pose estimation and introduce rotational ambiguity, while methods that incorporate depth or multiple views incur prohibitive storage and data acquisition costs.

To address these challenges, we introduce SplatPose, a novel framework aimed at solving the problem of 6-DoF pose estimation using a single RGB image. First, SplatPose proposes the Dual-Attention Ray Scoring Network (DARS-Net), which introduces a refined geometry scoring mechanism by decomposing the ray score into two independent components: position score and orientation score. By leveraging high-position-scoring rays and high-orientation-scoring rays to independently determine the cameraâs position and orientation, DARS-Net effectively overcomes the rotational ambiguity, achieving significant improvements in both translational and rotational accuracy, as illustrated in Fig. 1. Second, SplatPose designs an innovative 6-DoF pose estimation pipeline within a coarse-to-fine framework, which employs an effective feature point matching technique to refine the coarse pose initially derived from 3DGS rays. It represents a robust and scalable solution that pushes the boundaries of pose estimation based on 3D models. In summary, the key contributions of our proposed method are outlined as follows:

<!-- image-->  
Fig. 1. Comparison of pose estimation between 6DGS [14] and SplatPose. 6DGS selects high-scoring rays solely based on proximity to the cameraâs optical center, while SplatPose, via DARS-Net, refines pose estimation by incorporating both high-position-scoring rays (closer to the optical center) and high-orientation-scoring rays (aligned with the camera orientation), ultimately achieving smaller rotational errors compared to 6DGS.

â¢ We propose the Dual-Attention Ray Scoring Network (DARS-Net), which leverages attention mechanisms to decompose ray scoring into position and orientation components, effectively mitigating rotational ambiguity in 6-DoF pose estimation from a single RGB image.

â¢ We introduce a novel coarse-to-fine pipeline that employs an efficient keypoint matching technique to refine the coarse pose estimated from 3DGS rays.

â¢ The proposed SplatPose achieves state-of-the-art 6-DoF pose estimation results on three public Novel View Synthesis benchmarks, outperforming existing single RGBbased pose estimation methods while achieving performance comparable to depth- and multi-view-based approaches.

## II. RELATED WORKS

## A. Pose estimation based on Neural Radiance Fields

iNeRF [8] presented a framework employing NeRF to estimate 6-DoF poses by matching rendered images to target images through minimizing photometric errors. However, it tends to become trapped in local minima, leading to advancements like Parallel iNeRF [9], which optimizes multiple candidate poses in parallel. NeMo+VoGE [15], [16] uses volumetric Gaussian reconstruction kernels but relies on ray marching and iterative optimization, requiring training on multiple objects. In comparison, our approach utilizes a single-object 3DGS model, making it more efficient. CROSSFIRE [17] incorporates learned local features to mitigate local minima but still relies on accurate initial pose priors. IFFNeRF [18] proposes NeRF model inversion to rerender images matching a target view but overlooks unique 3DGS characteristics, such as ellipsoid elongation, rotation, and non-uniform spatial distribution, which our approach effectively addresses.

## B. Pose estimation based on 3D Gaussian Splatting

3DGS-ReLoc [13] pioneers LiDAR-camera fused 3DGS mapping using KD-trees and 2D voxel grids, employing NCC for coarse alignment and PnP for pose refinement. GSLoc [19] tackles photometric loss non-convexity via coarse-to-fine optimization, backpropagating gradients through 3DGS to refine sparse feature-based initializations. Meanwhile, SplatLoc [12] proposes a hybrid framework merging explicit 3DGS maps with learned descriptors, using saliency-driven landmark selection and anisotropic regularization to ensure accurate 2D-3D matching with compactness. These methods predominantly rely on depth information for 3D Gaussian scene reconstruction or necessitate multi-view image sequences. In contrast, 6DGS [14] eliminates dependency on pose initialization by inverting the 3DGS rendering process, thereby achieving single-RGBimage-based 6-DoF camera pose estimation. However, its Gaussian ellipsoid-based ray sampling strategy introduces rotational ambiguity, a limitation fundamentally addressed in our approach by the Dual-Attention Ray Scoring Network (DARS-Net), which resolves this geometric ambiguity through a meticulously designed dual-branch attention mechanism.

## C. Correspondence matching

Conventional methods for 6-DoF image matching predominantly rely on feature-based approaches, including both classical hand-crafted descriptors such as SIFT [20] and modern deep learning techniques like SuperGlue [21] and TransforMatcher [22]. SuperGlue utilizes a Graph Neural Network (GNN) to enhance feature focus and applies the Sinkhorn algorithm [23] for establishing correspondences. TransforMatcher [22] incorporates global match-to-match attention, facilitating accurate localization of correspondences. Additionally, feature equivariance techniques [24], [25] have been developed to enhance robustness by ensuring features remain consistent under transformations. These approaches, however, presume that the two feature ensembles intrinsically reside within a homogeneous data modality, usually derived directly from image data. In 3DGS-based approaches, the matching challenge differs, as it entails associating pixels with rays originating from the Ellicells. While OnePose++ [26] employs point cloud-image matching and CamNet [27] directly regresses poses, both require extensive multi-scene training (â¥500 images). To overcome these limitations, we utilize the proposed attention model to efficiently manage associations between rays and pixels, and achieve greater data efficiency by operating with significantly fewer images (approximately 100 or less) used exclusively during training.

<!-- image-->  
Fig. 2. An overview of our SplatPose pipeline. Our framework is composed of three key stages: (1) 3D Gaussian Scene Representation, where a 3DGS scene map is constructed from sparse point clouds to initialize the scene representation; (2) DARS-Net and Coarse Estimation, which decouples ray scoring into translation and rotation attention mechanisms, independently computing position and orientation scores for cast rays, selecting top-k rays based on these scores, and leveraging them to estimate the cameraâs position and orientation; and (3) Pose Refinement, where a synthetic scene view is rendered using the coarse pose, and keypoints are matched between the rendered view and the query image to refine the camera pose.

## III. METHODOLODY

In this part, we present the pipelines for reconstruction and pose estimation in our method, as illustrated in Fig. 2. In Section III-A, we begin by presenting the 3D Gaussian scene representation. Then, we introduce the Dual-Attention Ray Scoring Network( Section III-B). Finally, the coarse pose estimation and pose refinement are shown in Section III-C and Section III-D.

## A. 3D Gaussian Scene Representation

3D Gaussian Splatting represents a scene explicitly by employing a set of anisotropic 3D Gaussian primitives. Each Gaussian primitive is characterized in the world space by a mean vector $\mu \in \mathbb { R } ^ { 3 }$ and a covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ , as described by:

$$
G ( \mu , \Sigma ) = e ^ { - \frac { 1 } { 2 } ( x - \mu ) ^ { T } \Sigma ( x - \mu ) } .\tag{1}
$$

To ensure the covariance matrix Î£ retains its physical validity during optimization, it is expressed as the decomposition of a

scaling matrix S and a rotation matrix $R ,$ as proposed in [28]:

$$
\Sigma = R S S ^ { T } R ^ { T } ,\tag{2}
$$

where the scaling matrix S is derived from a 3D scale vector s, $S = \mathrm { d i a g } ( [ \mathbf { s } ] )$ , and the rotation matrix R is parameterized using a quaternion.

Following the method in [29], the 3D Gaussians are projected into the 2D image plane for rendering. The covariance matrix in the camera coordinate system is computed utilizing the viewing matrix W alongside the Jacobian J derived from the affine approximation of the projective transformation, as follows:

$$
\widetilde { \Sigma } = J W \Sigma W ^ { T } J ^ { T } .\tag{3}
$$

The corresponding 2D Gaussian distribution $\hat { G } ( \widetilde { \mu } , \widetilde { \Sigma } )$ is then derived from the 2D pixel location $\widetilde { \mu }$ eof the 3D Gaussian center and the projected covariance matrix $\widetilde { \Sigma }$

For novel view synthesis and fast rasterization-based rendering, each 3D Gaussian primitive is associated with an opacity value $\sigma ~ \in ~ \mathbb { R }$ and a color $c \in \mathbb { R } ^ { 3 }$ , represented using spherical harmonics (SH) coefficients. To achieve photorealistic rendering, the differentiable rasterizer employs alpha blending [30], which accumulates Gaussian properties and opacity values Ï for each pixel by traversing the ordered primitives. Specifically, the color properties are computed as follows:

$$
\hat { I } = \sum _ { i = 1 } ^ { N } c _ { i } \cdot \alpha _ { i } \cdot \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{4}
$$

where $\hat { I }$ is the rendered color. Here, $\alpha _ { i } = \hat { G } ( \widetilde { \mu } , \widetilde { \Sigma } )$ represents ethe opacity contribution of the i-th Gaussian to the pixel, $\textstyle \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )$ denotes the accumulated transmittance, and N is the total number of Gaussian primitives contributing to the pixel during the splatting process.

## B. Dual-Attention Ray Scoring Network

The primary challenge of rotational ambiguity in monocular 6-DoF pose estimation arises from the undifferentiated treatment of spatial and angular information in conventional ray scoring. To address this, we propose the Dual-Attention Ray Scoring Network (DARS-Net), which independently estimates camera position and orientation by leveraging dual attention mechanisms to model position and orientation scores for cast rays. Specifically, we generate multiple cast rays r for each Gaussian ellipsoid and determine a subset of r that corresponds to the target image It. Two attention maps, $A _ { p }$ and $A _ { o } ,$ compute position scores $\hat { s } _ { p }$ and orientation scores $\hat { s } _ { o }$ by evaluating the correlation between rays and image pixels. The top K rays from $A _ { p }$ estimate the camera position, while the top K rays from $A _ { o }$ determine the orientation.

Ray features $\mathbf { R } \in \mathbb { R } ^ { N \times C }$ , where C indicates the feature dimension, while N represents the total count of rays, are extracted using an augmented Multi-Layer Perceptron (MLP) architecture incorporating spatial coordinate embedding [31], boosting the networkâs ability to differentiate features. Image features are extracted from $\mathbf { I } _ { t }$ using the pre-trained DINOv2 [32] backbone, producing feature sets $\mathbf { \bar { F } } \in \mathbb { R } ^ { M \times C }$ , where $M = W \times H ,$ W represents the image width, and H represents the image height. These are processed through attention modules $\mathbf { \bar { \Phi } } _ { A _ { p } \left( \mathbf { R } , \mathbf { \bar { F } } \right) } \in \mathbb { R } ^ { M \times N }$ and $A _ { o } ( \mathbf { R } , \mathbf { F } ) \in \mathbb { R } ^ { \bar { M } \times N }$ , where ray features function as queries, while image features operate as keys. The resulting attention maps are optimized by performing row-wise summation and transforming them into per-ray correlation scores, respectively defined as position scores $\begin{array} { r } { \hat { s } _ { p } = \sum _ { i = 1 } ^ { M } A _ { p i } } \end{array}$ and orientation scores $\begin{array} { r } { \hat { s } _ { o } = \sum _ { i = 1 } ^ { \mathbf { \bar { M } } } A _ { o i } . } \end{array}$ During inference, the top K rays with the highest position scores predict the camera position, while those with the highest orientation scores determine the orientation.

The predicted scores $\hat { s } _ { p }$ and $\hat { s } _ { o }$ are supervised employing identical images from the 3DGS training set, under supervision based on the distance from the camera origin to its projection on the corresponding ray, along with the angle between the cameraâs orientation and the rayâs direction. The projection is computed as $L = \operatorname* { m a x } ( ( \mathbf { P } - \mathbf { r } _ { o } ) \mathbf { r } _ { d } , 0 )$ , where P is the camera position, $\mathbf { r } _ { o }$ the ray origin, and $\mathbf { r } _ { d }$ the ray direction. The distance is given by $d = | ( \mathbf { r } _ { o } + L \mathbf { r } _ { d } ) - \mathbf { P } | _ { 2 }$ , with d = 0 indicating that the ray intersects the optical center. The angle is calculated as $\begin{array} { r } { \theta = \operatorname { a r c c o s } \left( \frac { - \mathbf { Q } \cdot \mathbf { r } _ { d } } { | \mathbf { Q } | \cdot | \mathbf { r } _ { d } | } \right) } \end{array}$ , where Q is the camera orientation. These distances and angles are mapped to attention map scores for supervision. We map these distances and angels to attention map scores using:

$$
\begin{array} { c c } { { \alpha = 1 - \operatorname { t a n h } \left( \displaystyle \frac { d } { \gamma } \right) ; } } & { { s _ { p } = \alpha \displaystyle \frac { { \cal M } } { \sum \alpha } ; } } \\ { { \beta = 1 - \operatorname { t a n h } \left( \displaystyle \frac { \theta } { \gamma } \right) ; } } & { { s _ { o } = \beta \displaystyle \frac { { \cal M } } { \sum \beta } . } } \end{array}\tag{5}
$$

Here, Î³ regulates the allocation of rays to a given camera. Additionally, to compute the attention maps, the ground truth scores must be normalized due to the softmax operation. To optimize the predicted position scores $\hat { s } _ { p }$ and orientation scores $\hat { s } _ { o }$ against the computed ground truth position scores $s _ { p }$ and orientation scores ${ \mathit { s } } _ { o } ,$ we employ the $L _ { 2 }$ loss, formulated as:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { p } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \hat { \boldsymbol s } _ { p _ { i } } - \boldsymbol s _ { p _ { i } } \| _ { 2 } , } \\ { \displaystyle \mathcal { L } _ { o } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \big \| \hat { \boldsymbol s } _ { o _ { i } } - \boldsymbol s _ { o _ { i } } \big \| _ { 2 } , } \\ { \displaystyle \mathcal { L } = \mathcal { L } _ { p } + \mathcal { L } _ { o } . } \end{array}\tag{6}
$$

Where ${ \mathcal { L } } _ { p }$ represents the positional score loss, while $\mathcal { L } _ { o }$ denotes the orientation score loss. In every training iteration, a predicted image and pose are utilized to estimate the 3DGS model.

## C. Coarse Pose Estimation

The predicted position scores $\hat { s } _ { p }$ and orientation scores $\hat { s } _ { o }$ are independently used to determine the top K most relevant rays for position and direction respectively, with the selection restricted to a maximum of one ray per ellipsoid. The camera position is computed at the intersection of the selected rays and formulated as a weighted least-squares optimization problem. Due to discretization noise from the DARS-Net, 3D rays rarely converge at a single point. Therefore, the problem is addressed by minimizing the summation of squared normal distances. For each selected ray $\mathbf { r } _ { i }$ with $i = 1 . . . K$ , the error is defined as the squared distance between the predicted camera position PË and its orthogonal projection onto $\mathbf { r } _ { i }$

$$
\sum _ { i = 1 } ^ { K } \left( ( \hat { \mathbf { P } } - \mathbf { r } _ { o , i } ) ^ { T } ( \hat { \mathbf { P } } - \mathbf { r } _ { o , i } ) - ( ( \hat { \mathbf { P } } - \mathbf { r } _ { o , i } ) ^ { T } \mathbf { r } _ { d , i } ) ^ { 2 } \right) ,\tag{7}
$$

where $\mathbf { r } _ { o , i }$ represents the origin of the i-th ray, and $\mathbf { r } _ { d , i }$ denotes its corresponding direction. To minimize Eq. (7), the equation is differentiated with respect to PË, yielding:

$$
\hat { \mathbf { P } } = \sum _ { i = 1 } ^ { N } \hat { s } _ { p , i } \big ( \mathbb { I } - \mathbf { r } _ { d , i } \mathbf { r } _ { d , i } ^ { T } \big ) \mathbf { r } _ { o , i } ,\tag{8}
$$

where I denotes the identity matrix, and $\hat { s } _ { p , i }$ represent the predicted position scores. This formulation can be resolved as a weighted linear system.

The camera orientation is computed as the negative weighted sum of the direction vectors of the selected rays, with the weights determined by their predicted orientation scores $\hat { \boldsymbol { s } } _ { o }$ . The resulting orientation vector QË is expressed as:

$$
\hat { \mathbf { Q } } = - \frac { \sum _ { i = 1 } ^ { N } \hat { s } _ { o , i } \mathbf { r } _ { d , i } } { \left\| \sum _ { i = 1 } ^ { N } \hat { s } _ { o , i } \mathbf { r } _ { d , i } \right\| } ,\tag{9}
$$

where $\hat { s } _ { o , i }$ denotes predicted orientation scores. The normalization ensures that the computed orientation vector has unit magnitude.

## D. Pose Refinement

The coarse pose estimation process relies on ray sampling; however, the presence of noisy rays often prevents even high-scoring rays from precisely traversing the optical center of the camera, leading to inaccuracies in position and orientation estimation. This limitation imposes an inherent upper bound on the accuracy of pose estimation, necessitating further refinement. The refinement process begins with the extraction and matching of 2D feature points using LoFTR [33], a transformer-based feature matching method. Note that LoFTR here can be replaced with any other feature matcher. Using the coarse pose estimate, the 3D Gaussian primitives are mapped onto the 2D image plane to generate a synthetic rendering of the scene. LoFTR then computes high-quality 2D-2D correspondences between the query image and the rendered view, resulting in a set of matched keypoints. These 2D-2D correspondences are leveraged to compute 2D-3D correspondences by back-projecting the 2D keypoints in the rendered view to their corresponding 3D coordinates using the depth information of the 3D Gaussian primitives and the camera intrinsics. The resulting 2D-3D correspondences are then used to estimate the refined camera pose through a Perspective-n-Point (PnP) algorithm.

TABLE I  
THE MEAN ANGULAR ERROR (MAE) AND MEAN TRANSLATION ERROR (MTE) FOR 6-DOF POSE ESTIMATION ARE EVALUATED ON Mip-NeRF 360â¦ IN DEGREES AND UNITS u, WHERE 1u IS THE OBJECTâS LARGEST DIMENSION. LOWER VALUES INDICATE BETTER PERFORMANCE. [UP]: FIXED POSE PRIOR (FROM [8]). [MIDDLE]: RANDOM POSE PRIOR. [BOTTOM]: NO POSE PRIOR. RED: BEST. BLUE: SECOND BEST.
<table><tr><td>Method</td><td>Avg â</td><td>Bicycle</td><td>Bonsai</td><td>Counter</td><td>Garden</td><td>Kitchen</td><td>Room</td><td>Stump</td></tr><tr><td>iNeRF [8]</td><td>37.3/0.172</td><td>39.5/0.116</td><td>51.3/0.228</td><td>40.7/0.324</td><td>31.0/0.121</td><td>38.2/0.113</td><td>38.8/0.274</td><td>21.4/0.030</td></tr><tr><td>NeMo + VoGE [16]</td><td>40.9/0.036</td><td>43.8/0.015</td><td>52.5/0.036</td><td>45.6/0.072</td><td>31.8/0.026</td><td>41.6/0.042</td><td>44.9/0.045</td><td>26.3/0.016</td></tr><tr><td>Parallel iNeRF [9]</td><td>28.9/0.146</td><td>35.9/0.116</td><td>41.1/0.223</td><td>24.7/0.212</td><td>18.2/0.090</td><td>37.3/0.109</td><td>30.7/0.257</td><td>14.8/0.016</td></tr><tr><td>iNeRF [8]</td><td>85.0/0.292</td><td>76.6/0.217</td><td>96.7/0.385</td><td>70.3/0.487</td><td>72.8/0.210</td><td>100.2/0.266</td><td>91.6/0.444</td><td>86.9/0.035</td></tr><tr><td>NeMo + VoGE [16]</td><td>103.8/0.058</td><td>111.8/0.038</td><td>98.9/0.073</td><td>98.1/0.139</td><td>89.2/0.038</td><td>122.2/0.082</td><td>110.0/0.010</td><td>96.3/0.025</td></tr><tr><td>Parallel iNeRF [9]</td><td>58.0/0.218</td><td>44.4/0.150</td><td>58.2/0.298</td><td>42.1/0.435</td><td>60.0/0.144</td><td>65.0/0.193</td><td>63.5/0.271</td><td>72.6/0.033</td></tr><tr><td>6DGS [14]</td><td>24.3/0.022</td><td>12.1/0.010</td><td>10.5/0.038</td><td>19.6/0.043 37.8/0.015</td><td></td><td>23.2/0.018</td><td>38.3/0.019</td><td>28.3/0.009</td></tr><tr><td>Ours(Only DARS-Net)</td><td>11.1/0.012</td><td>9.14/0.010</td><td>5.79/0.020</td><td></td><td>9.65/0.022 21.9/0.008</td><td>7.91/0.009</td><td>8.79/0.013</td><td>14.3/0.005</td></tr><tr><td>Ours(DARS-Net + Pose Refinement)</td><td>1.06/0.007</td><td>0.17/0.003</td><td></td><td>0.73/0.006 0.52/0.015 2.55/0.005 0.47/0.008</td><td></td><td></td><td>2.44/0.010</td><td>0.53/0.002</td></tr></table>

TABLE II

THE MEAN ANGULAR ERROR (MAE) AND MEAN TRANSLATION ERROR (MTE) FOR 6-DOF POSE ESTIMATION ARE EVALUATED ON Tanks&Temples IN DEGREES AND UNITS u, WHERE 1u IS THE OBJECTâS LARGEST DIMENSION. LOWER VALUES INDICATE BETTER PERFORMANCE. [UP]: FIXED POSE PRIOR (FROM [8]). [MIDDLE]: RANDOM POSE PRIOR. [BOTTOM]: NO POSE PRIOR. RED: BEST. BLUE: SECOND BEST.
<table><tr><td>Method</td><td>Avg â</td><td>Barn</td><td>Caterpillar</td><td>Family</td><td>Ignatius</td><td>Truck</td></tr><tr><td>iNeRF [8]</td><td>35.0/0.452</td><td>26.5/0.208</td><td>42.9/0.166</td><td>42.8/0.794</td><td>31.4/0.723</td><td>31.6/0.370</td></tr><tr><td>NeMo + VoGE [16]</td><td>53.6/0.965</td><td>51.2/0.752</td><td>52.6/0.516</td><td>58.4/1.130</td><td>51.2/1.193</td><td>54.6/1.236</td></tr><tr><td>Parallel iNeRF [9]</td><td>24.7/0.346</td><td>22.9/0.131</td><td>25.2/0.138</td><td>22.9/0.507</td><td>23.4/0.604</td><td>29.4/0.351</td></tr><tr><td>iNeRF [8]</td><td>90.2/1.455</td><td>89.2/0.682</td><td>89.3/2.559</td><td>93.9/1.505</td><td>84.1/1.489</td><td>94.4/1.042</td></tr><tr><td>NeMo + VoGE [16]</td><td>92.6/1.457</td><td>92.5/0.684</td><td>90.5/2.559</td><td>97.0/1.506</td><td>85.4/1.491</td><td>97.7/1.045</td></tr><tr><td>Parallel iNeRF [9]</td><td>91.1/1.130</td><td>85.2/0.572</td><td>86.8/0.843</td><td>99.0/2.028</td><td>86.9/1.326</td><td>97.6/0.883</td></tr><tr><td>6DGS [14]</td><td>21.7/0.268</td><td>30.3/0.162</td><td>14.5/0.027</td><td>20.6/0.468</td><td>15.5/0.441</td><td>27.5/0.242</td></tr><tr><td>Ours(Only DARS-Net)</td><td>5.36/0.257</td><td>5.13/0.147</td><td>4.91/0.025</td><td>4.52/0.460</td><td>5.90/0.412</td><td>6.35/0.239</td></tr><tr><td>Ours(DARS-Net + Pose Refinement)</td><td>2.97/0.211</td><td>3.86/0.122</td><td>2.00/0.023</td><td>3.16/0.413</td><td>1.92/0.273</td><td>3.90/0.227</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## IV. EXPERIMENTS

## A. Experimental Setup

We compare SplatPose with the 3DGS-based method 6DGS [14] and Nerf-based approaches for 6-DoF pose estimation with single RGB image, including iNeRF [8], Parallel iNeRF [9], and NeMo+VoGE [16]. Following the evaluation protocol in [14], experiments are conducted on Mip-NeRF 360â¦ [35] and Tanks&Temples [36] datasets using their predefined training-test splits. We test under two pose initialization scenarios: (i) iNeRF initialization, with uniformly sampled errors in [â40â¦, +40â¦] for rotation and [â0.1, +0.1] for translation; and (ii) a realistic initialization, where the starting pose is randomly chosen from those used to create the 3DGS model. The second setting evaluates methods under more practical conditions. Ablation studies are also performed to validate each system component. Pose estimation performance is measured using mean angular error (MAE) and mean translational error (MTE) (see Table I and Table II).

Additionally, to verify SplatPoseâs robustness in intricate physical environments, we follow the experimental setup of [12], selecting two Depth- and Multi-View-Based approaches for comparative analysis, including scene coordinate regression approach SCRNet [34] and recent 3DGSbased visual localization approach SplatLoc [12], using the 12Scenes dataset [37].

TABLE III  
THE MEDIAN ANGULAR ERROR AND MEDIAN TRANSLATION ERROR (MAE, MTE) FOR 6-DOF POSE ESTIMATION ARE EVALUATED ON 12Scenes IN DEGREES AND cm. LOWER VALUES INDICATE BETTER PERFORMANCE. RED: BEST. BLUE: SECOND BEST.
<table><tr><td rowspan="2">Method</td><td colspan="2">Apartment 1</td><td colspan="3">Apartment 2</td><td colspan="4">Office 1</td><td colspan="2">Office 2</td></tr><tr><td>kitchen</td><td>living</td><td>kitchen</td><td>living</td><td>luke</td><td>gates362</td><td>gates381</td><td>lounge</td><td>manolis</td><td>5a</td><td>5b</td></tr><tr><td>SCRNet [34]</td><td>1.3/2.3</td><td>0.8/2.4</td><td>1.0/2.1</td><td>1.8/4.2</td><td>1.4/4.4</td><td>0.8/2.6</td><td>1.4/3.4</td><td>0.9/2.7</td><td>1.0/1.8</td><td>1.5/3.6</td><td>1.2/3.4</td></tr><tr><td>SplatLoc [12]</td><td>0.4/0.8</td><td>0.4/1.1</td><td>0.5/1.0</td><td>0.5/1.2</td><td>0.6/1.5</td><td>0.5/1.1</td><td>0.5/1.2</td><td>0.5/1.6</td><td>0.5/1.1</td><td>0.6/1.4</td><td>0.5/1.5</td></tr><tr><td>Ours</td><td>0.4/5.3</td><td>0.4/3.6</td><td>0.5/2.9</td><td>0.5/2.8</td><td>0.7/4.2</td><td>0.3/1.9</td><td>0.7/4.0</td><td>0.5/5.2</td><td>0.6/3.1</td><td>0.6/5.0</td><td>0.5/3.0</td></tr></table>

<!-- image-->  
(a) room

<!-- image-->  
(b) counter

<!-- image-->  
(c) Barn

<!-- image-->  
(d) Caterpillar  
Fig. 3. The illustration presents qualitative results from the Mip-NeRF 360Â° dataset ((a) and (b)) and the Tanks & Temples dataset ((c) and (d)). From top to bottom, there are results of 6DGS [14], ours, and ground truth. For each scene, the images are rendered based on the estimated camera poses utilizing the provided 3DGS model.

TABLE IV  
MEMORY USAGE, TRAINING TIME AND INFERENCE TIME OF DIFFERENT METHODS ON SCENE MANOLIS FROM 12-SCENES DATASET.
<table><tr><td>Method</td><td>Memory â</td><td>Training time â</td><td>Inference time â</td></tr><tr><td>SCRNet [34]</td><td>165MB</td><td>2days</td><td>1min</td></tr><tr><td>SplatLoc [12]</td><td>737MB</td><td>25mins</td><td>9mins13s</td></tr><tr><td>Ours</td><td>264MB</td><td>45mins</td><td>6mins20s</td></tr></table>

Implementation Details. The SplatPose framework is implemented using PyTorch, with the attention map trained for 1,500 iterations (approximately 45 minutes) on an NVIDIA GeForce RTX 3090 GPU. This optimization employs the Adafactor algorithm [38], with a weight decay coefficient of 10â3. To accelerate the training process, 2,000 3DGS ellipsoids are uniformly sampled at each iteration.

## B. Datasets

Mip-NeRF 360â¦ [35] includes seven scenes (two outdoor, five indoor) with structured settings and consistent backgrounds. We use the original 1:8 train-test splits from [35]. Following [9], all objects are scaled to a unit box, and translation errors are normalized by object size.

Tanks&Temples [36] is a benchmark for 3D reconstruction on real-world objects of varying scales. Objects were captured from human-like perspectives under challenging illumination, shadows, and reflections. We evaluate five scenes (Barn, Caterpillar, Family, Ignatius, Truck) using dataset splits from [39], with 247 training images (87%) and 35 test images (12%) per split.

12Scenes [37] provides RGB-D imagery from 12 rooms across four scenes, captured with depth sensors and iPad cameras. Following standard protocols, the first sequence is used for evaluation, and the others for training.

## C. Experimental Analysis

Comparison with Single RGB-Based Methods: Table I and Table II present quantitative comparisons across Mip-NeRF 360Â° and Tanks & Temples benchmarks, demonstrating SplatPoseâs superior accuracy in all environments. For Mip-NeRF 360Â° evaluations, our framework with only DARS-Net obtains mean angular errors of 11.1Â° and positional errors of 0.012, surpassing 6DGSâs metrics (24.3Â°/0.022). With the full pipeline (DARS-Net + Pose Refinement), the errors are further reduced to 1.06Â° and 0.007. Similarly, on the Tanks & Temples dataset, using only DARS-Net achieves an angular error of 5.36Â° and a translation error of 0.257, surpassing 6DGS (21.7Â°/0.268). The full pipeline reduces these errors to 2.97Â° and 0.211. The full pipeline of our method achieves the best pose estimation performance on both benchmark datasets.

TABLE V  
THE IMPACT OF DIFFERENT STAGES IN SPLATPOSE ON POSE ESTIMATION PERFORMANCE. WE REPORT THE MEAN ANGULAR AND TRANSLATION ERRORS (DEGREE, U) ON MIP-NERF 360Â° DATASET, WHERE 1U IS THE OBJECTâS LARGEST DIMENSION. A: DARS-NET. B:POSE REFINEMENT. BOLD: BEST IN COL.
<table><tr><td>Method</td><td>Baseline</td><td>A</td><td>B</td><td>Avg â</td><td>Bicycle</td><td>Bonsai</td><td>Counter</td><td>Garden</td><td>Kitchen</td><td>Room</td><td>Stump</td></tr><tr><td>Exp1</td><td>â</td><td>X</td><td>X</td><td>24.3/0.022</td><td>12.1/0.010</td><td>10.5/0.038</td><td>19.6/0.043</td><td>37.8/0.015</td><td>23.2/0.018</td><td>38.3/0.019</td><td>28.3/0.009</td></tr><tr><td>Exp2</td><td>â</td><td></td><td>â r x</td><td>11.1/0.012</td><td>9.14/0.010</td><td>5.79/0.020</td><td>9.65/0.022</td><td>21.9/0.008</td><td>7.91/0.009</td><td>8.79/0.013</td><td>14.3/0.005</td></tr><tr><td>Exp3</td><td>â</td><td></td><td>x â</td><td>10.1/0.009</td><td>3.58/0.004</td><td>1.10/0.008</td><td>3.67/0.018</td><td>25.54/0.007</td><td>4.24/0.009</td><td>30.9/0.014</td><td>1.45/0.002</td></tr><tr><td>Exp4</td><td>â</td><td>â</td><td>â</td><td>1.06 / 0.007</td><td>0.17 / 0.003</td><td>0.73 / 0.006</td><td>0.52 / 0.015</td><td>2.55 / 0.005</td><td>0.47 / 0.008</td><td>2.44 / 0.010</td><td>0.53 / 0.002</td></tr></table>

Fig. 3 presents qualitative results across various scenes, highlighting the effectiveness of our method. Our approach consistently produces results closely aligned with the ground truth (GT), while 6DGS demonstrates significant camera pose deviation, particularly in cluttered or complex scenes. These findings corroborate the quantitative results, highlighting the robustness and adaptability of our method across varying scene complexities and environments.

Comparison with Depth- and Multi-View-Based Methods: As shown in Table III, SplatPose achieves angular accuracy comparable to SplatLoc using only a single RGB image, whereas SplatLoc requires multiple views and depth information. Table IV compares memory usage, training time, and inference time for different methods on the Manolis scene. Our method achieves a memory footprint of 264 MB, markedly lower than SplatLocâs 737 MB, by eliminating the need to store consecutive frames and leveraging a compact 3D Gaussian map representation. While our approach and SplatLoc exhibit comparable training times (45 minutes vs. 25 minutes) due to similar Gaussian optimization processes, both significantly outperform SCRNetâs 2-day training duration, as neither relies on complex network architectures. Furthermore, our inference time of 6 minutes and 20 seconds surpasses SplatLocâs 9-minute runtime by bypassing initial pose retrieval. Although SCRNet achieves faster inference (1 minute) and lower memory usage(165 MB), this advantage is offset by its intensive computational training requirements and limited scene generalization, stemming from its datadependent regression paradigm.

## D. Ablation Study

As shown in Table V, we analyze the impact of each stage of SplatPose on pose estimation performance. The baseline method (Exp1), which lacks both DARS-Netâs scoring mechanism and Pose Refinement, shows the highest average errors (24.3Â°, 0.022), performing poorly in sequences like Garden and Room (37.8Â° and 38.3Â° angular errors, respectively).

Exp2 introduces DARS-Net (A), reducing the average angular error by 54.4% (to 11.1Â°) and the translation error by 45.5% (to 0.012) compared to the baseline. Notably, in the most challenging sequence, Room, the rotation error decreases by 77.0% (from 38.3Â° to 8.79Â°). Exp3 replaces DARS-Net with Pose Refinement (B), achieving comparable results, with improved performance in indoor sequences like Counter (angular error reduced to 3.67Â°) but higher error in Garden (25.54Â°).

Exp4, combining DARS-Net and Pose Refinement, achieves the best performance across all sequences, reducing the average angular error by 95.6% (to 1.06Â°) and the translation error by 68.2% (to 0.007). In Garden, angular error drops by 93.3% (from 37.8Â° to 2.55Â°), while simpler sequences like Bonsai achieve near-perfect results (0.73Â°, 0.006). These results demonstrate the complementary strengths of DARS-Net and Pose Refinement, enabling robust and precise 6-DoF pose estimation.

## V. CONCLUSIONS

This work introduces SplatPose, an advanced 6-DoF pose estimation system that builds on 3DGS with a Dual-Attention Ray Scoring Network (DARS-Net) and a coarse-to-fine pose estimation pipeline. By leveraging DARS-Net, our approach decouples positional and angular alignment in the geometric domain, effectively addressing rotational ambiguity and achieving state-of-the-art accuracy in single-image RGB pose estimation. Experiments on three public datasetsâMip-NeRF 360Â°, Tanks&Temples, and 12Scenesâdemonstrate SplatPoseâs superiority over existing single RGB-based methods and depth- or multi-view-based approaches. On Mip-NeRF 360Â°, SplatPose achieves 10â20Ã lower angular errors and 3Ã lower translation errors than 6DGS. On Tanks&Temples, it attains angular and translation errors of 2.97Â° and 0.211, outperforming prior methods in real-world conditions. On 12Scenes, SplatPose matches the accuracy of depth-dependent methods like SplatLoc while avoiding reliance on large image databases or depth data. Additionally, it reduces memory usage by over 64% compared to SplatLoc and offers faster inference, enhancing practicality.

## REFERENCES

[1] B. Zhao, L. Yang, M. Mao, H. Bao, and Z. Cui, âPnerfloc: Visual localization with point-based neural radiance fields,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7450â7459.

[2] Y. He, W. Sun, H. Huang, J. Liu, H. Fan, and J. Sun, âPvn3d: A deep point-wise 3d keypoints voting network for 6dof pose estimation,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11 632â11 641.

[3] J. Tremblay, T. To, B. Sundaralingam, Y. Xiang, D. Fox, and S. Birchfield, âDeep object pose estimation for semantic robotic grasping of household objects,â arXiv preprint arXiv:1809.10790, 2018.

[4] Y. Lin, J. Tremblay, S. Tyree, P. A. Vela, and S. Birchfield, âSinglestage keypoint-based category-level object pose estimation from an rgb image,â in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 1547â1553.

[5] M. Sundermeyer, Z.-C. Marton, M. Durner, M. Brucker, and R. Triebel, âImplicit 3d orientation learning for 6d object detection from rgb images,â in Proceedings of the european conference on computer vision (ECCV), 2018, pp. 699â715.

[6] S. Peng, Y. Liu, Q. Huang, X. Zhou, and H. Bao, âPvnet: Pixelwise voting network for 6dof pose estimation,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4561â4570.

[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[8] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-Y. Lin, âinerf: Inverting neural radiance fields for pose estimation,â in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 1323â1330.

[9] Y. Lin, T. Muller, J. Tremblay, B. Wen, S. Tyree, A. Evans, P. A. Â¨ Vela, and S. Birchfield, âParallel inversion of neural radiance fields for robust pose estimation,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 9377â9384.

[10] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, âDeformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 331â20 341.

[11] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, âMip-splatting: Alias-free 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 447â19 456.

[12] H. Zhai, X. Zhang, B. Zhao, H. Li, Y. He, Z. Cui, H. Bao, and G. Zhang, âSplatloc: 3d gaussian splatting-based visual localization for augmented reality,â arXiv preprint arXiv:2409.14067, 2024.

[13] P. Jiang, G. Pandey, and S. Saripalli, â3dgs-reloc: 3d gaussian splatting for map representation and visual relocalization,â arXiv preprint arXiv:2403.11367, 2024.

[14] M. Bortolon, T. Tsesmelis, S. James, F. Poiesi, and A. Del Bue, â6dgs: 6d pose estimation from a single image and a 3d gaussian splatting model,â arXiv preprint arXiv:2407.15484, 2024.

[15] A. Wang, A. Kortylewski, and A. Yuille, âNemo: Neural mesh models of contrastive features for robust 3d pose estimation,â arXiv preprint arXiv:2101.12378, 2021.

[16] A. Wang, P. Wang, J. Sun, A. Kortylewski, and A. Yuille, âVoge: a differentiable volume renderer using gaussian ellipsoids for analysisby-synthesis,â arXiv preprint arXiv:2205.15401, 2022.

[17] A. Moreau, N. Piasco, M. Bennehar, D. Tsishkou, B. Stanciulescu, and A. de La Fortelle, âCrossfire: Camera relocalization on selfsupervised features from an implicit representation,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 252â262.

[18] M. Bortolon, T. Tsesmelis, S. James, F. Poiesi, and A. Del Bue, âIffnerf: Initialisation free and fast 6dof pose estimation from a single image and a nerf model,â arXiv preprint arXiv:2403.12682, 2024.

[19] Z. Niu, Z. Tan, J. Zhang, X. Yang, and D. Hu, âHgsloc: 3dgs-based heuristic camera pose refinement,â arXiv preprint arXiv:2409.10925, 2024.

[20] D. G. Lowe, âObject recognition from local scale-invariant features,â in Proceedings of the seventh IEEE international conference on computer vision, vol. 2. Ieee, 1999, pp. 1150â1157.

[21] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperglue: Learning feature matching with graph neural networks,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4938â4947.

[22] S. Kim, J. Min, and M. Cho, âTransformatcher: Match-to-match attention for semantic correspondence,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8697â8707.

[23] M. Cuturi, âSinkhorn distances: Lightspeed computation of optimal transport,â Advances in neural information processing systems, vol. 26, 2013.

[24] J. Lee, B. Kim, and M. Cho, âSelf-supervised equivariant learning for oriented keypoint detection,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 4847â4857.

[25] J. Lee, B. Kim, S. Kim, and M. Cho, âLearning rotation-equivariant features for visual correspondence,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21 887â21 897.

[26] X. He, J. Sun, Y. Wang, D. Huang, H. Bao, and X. Zhou, âOnepose++: Keypoint-free one-shot object pose estimation without cad models,â Advances in Neural Information Processing Systems, vol. 35, pp. 35 103â35 115, 2022.

[27] M. Ding, Z. Wang, J. Sun, J. Shi, and P. Luo, âCamnet: Coarse-to-fine retrieval for camera re-localization,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 2871â2880.

[28] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[29] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, âEwa volume splatting,â in Proceedings Visualization, 2001. VISâ01. IEEE, 2001, pp. 29â538.

[30] N. Max, âOptical models for direct volume rendering,â IEEE Transactions on Visualization and Computer Graphics, vol. 1, no. 2, pp. 99â108, 1995.

[31] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, and R. Ng, âFourier features let networks learn high frequency functions in low dimensional domains,â Advances in neural information processing systems, vol. 33, pp. 7537â7547, 2020.

[32] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., âDinov2: Learning robust visual features without supervision,â arXiv preprint arXiv:2304.07193, 2023.

[33] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, âLoftr: Detectorfree local feature matching with transformers,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 8922â8931.

[34] X. Li, S. Wang, Y. Zhao, J. Verbeek, and J. Kannala, âHierarchical scene coordinate classification and regression for visual localization,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 983â11 992.

[35] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5470â5479.

[36] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, âTanks and temples: Benchmarking large-scale scene reconstruction,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1â13, 2017.

[37] J. Valentin, A. Dai, M. NieÃner, P. Kohli, P. Torr, S. Izadi, and C. Keskin, âLearning to navigate the energy landscape,â in 2016 Fourth International Conference on 3D Vision (3DV). IEEE, 2016, pp. 323â 332.

[38] N. Shazeer and M. Stern, âAdafactor: Adaptive learning rates with sublinear memory cost,â in International Conference on Machine Learning. PMLR, 2018, pp. 4596â4604.

[39] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, âTensorf: Tensorial radiance fields,â in European conference on computer vision. Springer, 2022, pp. 333â350.