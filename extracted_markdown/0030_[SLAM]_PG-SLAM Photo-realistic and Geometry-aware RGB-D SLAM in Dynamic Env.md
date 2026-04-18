# PG-SLAM: Photo-realistic and Geometry-aware RGB-D SLAM in Dynamic Environments

Haoang Li, Xiangqi Meng, Xingxing Zuo, Zhe Liu, Hesheng Wang, and Daniel Cremers

AbstractâSimultaneous localization and mapping (SLAM) has achieved impressive performance in static environments. However, SLAM in dynamic environments remains an open question. Many methods directly filter out dynamic objects, resulting in incomplete scene reconstruction and limited accuracy of camera localization. The other works express dynamic objects by point clouds, sparse joints, or coarse meshes, which fails to provide a photo-realistic representation. To overcome the above limitations, we propose a photo-realistic and geometryaware RGB-D SLAM method by extending Gaussian splatting. Our method is composed of three main modules to 1) map the dynamic foreground including non-rigid humans and rigid items, 2) reconstruct the static background, and 3) localize the camera. To map the foreground, we focus on modeling the deformations and/or motions. We consider the shape priors of humans and exploit geometric and appearance constraints of humans and items. For background mapping, we design an optimization strategy between neighboring local maps by integrating appearance constraint into geometric alignment. As to camera localization, we leverage both static background and dynamic foreground to increase the observations for noise compensation. We explore the geometric and appearance constraints by associating 3D Gaussians with 2D optical flows and pixel patches. Experiments on various real-world datasets demonstrate that our method outperforms state-of-the-art approaches in terms of camera localization and scene representation. Source codes will be publicly available upon paper acceptance.

Index TermsâRGB-D SLAM, dynamic environments, Gaussian splatting, optical flows.

## I. INTRODUCTION

Simultaneous localization and mapping (SLAM) is a crucial technique to help mobile agents autonomously navigate in unknown environments. It has various applications such as robotics, autonomous vehicles, and augmented reality [1]. Among different types of SLAM configurations, visual SLAM based on RGB and/or depth cameras gain popularity thanks to low cost and high compactness. Previous visual SLAM research develops various solutions based on the extended Kalman filter [2], feature correspondences [3], and direct pixel alignment [4]. They achieve reliable camera localization, but fail to provide photo-realistic scene representation based on point clouds. To improve scene representation, some new approaches based on the differentiable rendering [5]â[7] were proposed. They feature high-quality scene rendering from novel views and gained wide attention in recent years.

The above methods mainly focus on static environments, regardless of the strategies for scene representation. In practice, dynamic environments are very common, exemplified by crowded shopping malls and busy roads. A dynamic environment is typically composed of static background such as walls and floors, and dynamic foreground such as non-rigid humans and rigid items. SLAM in dynamic environments is challenging since dynamic foreground does not satisfy some basic geometric constraints such as epipolar geometry [8], and thus significantly affects the system robustness. To overcome this challenge, a straightforward solution is filtering out dynamic foreground and only using static background for camera localization [9]â[11]. This solution is not ideal for two main reasons. First, it fails to map the foreground, resulting in an incomplete scene reconstruction. Second, the accuracy of camera localization is affected by the loss of foreground information, especially when dynamic foreground is dominant in an image.

Contrary to the above foreground filtering-based solution, some works involve dynamic foreground in camera localization and scene mapping. Early methods [12], [13] only focus on rigid items like cars via exploring some geometric relationship. Recent approaches [14], [15] further model the motions and shapes of non-rigid humans based on prior constraints. However, the mapping results of the above methods lack fine details and textures. For example, the rigid itemoriented methods can only generate dynamic point clouds. The approaches targeted at non-rigid humans merely reconstruct sparse joints or coarse meshes. To overcome the above limitations, we propose a photo-realistic and geometry-aware RGB-D SLAM method named PG-SLAM in dynamic environments. As shown in Fig. 1, our method is composed of three main modules to 1) reconstruct the dynamic foreground including non-rigid humans and rigid items, 2) map the static background, and 3) localize the moving camera. All these modules are designed by extending Gaussian splatting (GS) [16], a recently popular differentiable rendering technique.

To achieve a photo-realistic mapping of dynamic foreground, we propose dynamic GS constrained by geometric priors and human/item motions. For non-rigid humans, we adapt GS in three aspects. First, we attach Gaussians to the skinned multi-person linear (SMPL) model [17], which satisfies the articulated constraints of humans. Second, we design a neural network to model the deformation of humans over time by considering the varied human pose. Third, to compute the motions of humans, we exploit the 3D root joint of the SMPL model and regularize the scales of Gaussians. As for the mapping of rigid items, we focus on computing the rigid motions of these items. To achieve this, we associate dynamic Gaussians at different times based on the optical flows, followed by aligning these Gaussinas. Moreover, we improve the completeness of mapping by effectively managing the previously and newly observed parts of items.

As to the mapping of static background, given that an area can be consistently observed by multiple sequential images, we introduce a local map to manage these images. Gaussian optimization within such a local map improves the accuracy thanks to multiple-view appearance constraints. Moreover, we propose an optimization strategy between neighboring local maps based on both geometric and appearance constraints. The geometric constraint is formulated as an iterative alignment between the centers of Gaussians from neighboring local maps. Moreover, we integrate appearance constraint into each iteration, leading to more reliable convergence. The above optimization strategy can reduce the accumulated error and ensure consistency between local maps. In addition, this strategy can be applied to loop closure when two local maps are not temporally continuous but spatially overlapping.

For camera localization, our method leverages information of not only static background but also dynamic foreground. Additional observations can compensate for noise and thus improve the localization accuracy. To achieve this, we propose a two-stage strategy to estimate the camera pose in a coarse-to-fine manner based on both appearance and geometric constraints. For one thing, we use the camera pose to render Gaussians as images, formulating the appearance constraints. For another, we use the camera pose to project Gaussians and blend them based on the rendering weights, generating the projected optical flows. By aligning the observed and projected optical flows, we define the geometric constraint. By combining the above constraints, our method achieves highaccuracy camera localization.

Overall, we propose a photo-realistic and geometry-aware RGB-D SLAM method for dynamic environments. Our main contributions are summarized as follows:

â¢ To the best of our knowledge, we propose the first Gaussian splatting-based SLAM method that can not only localize the camera and reconstruct the static background, but also map the dynamic humans and items.

â¢ Our method can provide photo-realistic representation of dynamic scenes. For foreground mapping, we consider shape priors of humans and exploit geometric and appearance constraints with respect to Gaussians. To map the background, we design an effective optimization strategy between neighboring local maps.

â¢ Our method simultaneously uses the geometric and appearance constraints to localize the camera by associating 3D Gaussians with 2D optical flows and pixel patches. We leverage information of both static background and dynamic foreground to compensate for noise, effectively improving the localization accuracy.

Experiments on various real-world datasets demonstrate that our method outperforms state-of-the-art approaches.

## II. RELATED WORK

Existing visual SLAM methods can be roughly classified into three categories in terms of representation strategies, i.e. the traditional representation-based, the implicit representation-based, and the GS-based approaches. We review these types of work for both static and dynamic environments.

## A. Traditional Representation-based Methods

Point cloud is one of the dominant strategies for 3D representation. To reconstruct point clouds, previous SLAM methods match 2D point features for triangulation [2], [3] or back-project depth images [18], [19]. These methods also use point clouds to localize the camera based on several geometric algorithms. Mesh and voxel are the other common 3D representation strategies. Some SLAM approaches [20], [21] consider these types of data to achieve a continuous representation, but partly increase the complexity of map update. The above SLAM methods can reliably work in static environments, but become unstable in the presence of dynamic objects due to lack of effective geometric constraints.

To improve the SLAM robustness in dynamic scenes, a straightforward strategy is detecting and filtering out dynamic objects. To achieve this, FlowFusion [22] utilizes the optical flows, DSLAM [23] considers the temporal correlation of points, and DS-SLAM [10] leverages the semantic information. While these methods can provide satisfactory estimation of camera trajectory, they are unable to reconstruct dynamic objects. By contrast, some methods were designed to map dynamic objects. For example, VDO-SLAM [12] and DynaSLAM II [13] explore geometric constraints with respect to the motions of rigid items. AirDOS [14] extends the rigid items to non-rigid humans by expressing humans with a set of articulated joints. By contrast, Body-SLAM [15] can additionally provide coarse human shapes based on the SMPL model. The main limitation of the above methods is that they neglect the details and textures of dynamic objects and thus fail to provide a photo-realistic scene representation.

## B. Implicit Representation-based Methods

Signed distance function fields [24] and neural radiance fields (NeRF) [25] are representative implicit scene representation methods. They express a 3D environment based on neural networks whose inputs are 3D coordinates of arbitrary positions and outputs are geometric and appearance attributes of these positions. They can provide differentiable and photorealistic rendering of 3D scenes from novel views. These techniques have already been used for SLAM to jointly represent the 3D scene and optimize the camera poses [5], [6], [26]â [30]. The seminal work iMAP [5] adapts the original NeRF by neglecting view directions to efficiently express a 3D space. A subsequent method NICE-SLAM [6] introduces the hierarchical voxel features as inputs to improve the generalization of scene representation. The state-of-the-art work ESLAM [27] leverages the multi-scale axis-aligned feature planes where the interpolated features are passed through a decoder to predict the truncated signed distance field. This method can generate more accurate 3D meshes than iMAP and NICE-SLAM.

The above methods are mainly designed for static scenes and can hardly handle dynamic environments. To solve this problem, Rodyn-SLAM [11] was recently proposed. This method estimates the masks of dynamic objects and further eliminates these entities. While it achieves a relatively decent camera tracking performance, it cannot reconstruct dynamic objects. Moreover, the object filtering-based strategy results in the loss of useful information, affecting the accuracy of camera localization.

## C. Gaussian Splatting-based Methods

While the above NeRF has shown impressive performance in 3D scene representation, there is still room for improvement in quality and efficiency of rendering. GS improves NeRF by replacing the implicit neural network with a set of explicit Gaussians. Several concurrent works [7], [31]â[33] integrated this technique into SLAM. They render Gaussians as RGB and depth images, and employ the photometric and depth losses to jointly optimize Gaussians and camera pose. They also tailor GS to special configurations of SLAM, such as sequential images with relatively similar views. For example, MonoGS [31] introduces the isotropic loss to avoid the ellipsoids with too long axes. SplatSLAM [32] proposes a silhouette-guided pixel selection strategy to exploit reliable pixels for optimization. These methods typically achieve more accurate 3D reconstruction and/or faster camera localization than the above NeRF-based approaches.

Despite high reliability in static environments, the performance of the above methods is unsatisfactory in dynamic scenes. The reason is that they use static Gaussians to express dynamic object movement, resulting in blurry rendered images that drastically affect SLAM optimization. To solve this problem, a very recent work [34] follows the above Rodyn-SLAM to filter out dynamic objects. However, it leads to an incomplete scene reconstruction.

Overall, existing visual SLAM methods fail to achieve a photo-realistic and complete scene representation, as well as accurate camera localization in dynamic environments. We overcome these limitations by proposing dynamic GS constrained by geometric priors and human/item motions. Our optimization strategies based on both appearance and geometric constraints effectively improve the accuracy of SLAM.

## III. PROBLEM FORMULATION

In this section, we first introduce preliminary knowledge of GS and SMPL model, and then present an overview of our SLAM system.

## A. Preliminary

1) Gaussian Splatting: GS is a differentiable rendering technique, featuring higher performance and geometric explainability. Briefly, a 3D space is expressed by a set of Gaussians $\mathcal { G } = \{ G _ { i } \}$ , each of which is defined by:

$$
G _ { i } ( \mathbf { P } ) = o _ { i } \cdot \exp \big \{ - \frac { 1 } { 2 } ( \mathbf { P } - \pmb { \mu } _ { i } ) ^ { \top } \pmb { \Sigma } _ { i } ^ { - 1 } ( \mathbf { P } - \pmb { \mu } _ { i } ) \big \} ,\tag{1}
$$

where P is an arbitrary position in 3D, $\mu _ { i } , o _ { i } ,$ i, and Î£i d $\Sigma _ { i }$ enote the center, opacity, and covariance matrix of the i-th Gaussian, respectively. A covariance matrix can be decomposed into the scale and rotation. Each Gaussian is also associated with spherical harmonics to encode different colors along various view directions.

As to the differentiable rendering, a pixel p and the camera center define a 3D projection ray. Along this ray, N 3D Gaussians are projected onto 2D image from close to far. The i-th projected 2D Gaussian is associated with the color $c _ { i } ,$ depth $d _ { i } ,$ , and covariance $\Sigma _ { i } ^ { \mathrm { 2 D } } \left( 1 \leq i \leq N \right)$ . These parameters are obtained by spherical harmonics, z-coordinate of the center, and covariance of the corresponding 3D Gaussian, respectively. The above N 2D Gaussians are blended to determine the color $c _ { \mathbf { p } }$ and depth $d _ { \mathbf { p } }$ of the pixel p:

$$
c _ { \mathbf { p } } = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) , \ d _ { \mathbf { p } } = \sum _ { i = 1 } ^ { N } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) ,\tag{2}
$$

where $\alpha _ { i }$ represents the blending weight of the i-th 2D Gaussian which is computed based on the opacity $o _ { i }$ and covariance $\Sigma _ { i } ^ { \mathrm { 2 D } }$ . Through the above pixel-wise computation, the rendered RGB image ËI and depth image DË can be obtained. For symbol simplification, in our context, we denote the above rendering process by $\tilde { I } , \tilde { D } = \pi [ \mathcal { G } ]$

To measure the appearance similarity between the rendered and observed RGB images, the photometric error is defined by the combination of $L _ { 1 }$ and SSIM errors [35]:

$$
L _ { \mathrm { P } } = \lambda \cdot \| I - \tilde { I } \| _ { 1 } + ( 1 - \lambda ) \cdot \big ( 1 - \mathrm { S S I M } ( I , \tilde { I } ) \big ) ,\tag{3}
$$

where $\| \cdot \| _ { 1 }$ denotes $L _ { 1 }$ error, Î» is the weight to control the trade-off between two terms. In addition, to measure the similarity between the rendered and observed depth images, we use the depth error $L _ { \mathrm { D } }$ defined by $L _ { 1 }$ error. The combination of photometric and depth errors forms the appearance constraint.

2) SMPL Model: SMPL model [17] can effectively express dynamic 3D humans. It is defined by a kinematic skeleton composed of 24 body joints, the pose parameter Î to drive the joints, and the shape parameter Î¦ to describe different heights and weights. Body joints include a root joint and other ordinary joints. The root joint is considered to be the pelvis, as this is the central part of the body from which the rest of the bodyâs kinematic chain (spine, legs, arms, etc.) originates. The origin of the SMPL frame is located at the root joint. The root joint is associated with the rotation and translation between the SMPL frame and the camera frame. The k-th ordinary joint is associated with a transformation $\theta ^ { k }$ that defines its position and orientation relative to its parent joint in the hierarchy. These transformations $\{ \theta ^ { k } \}$ constitute the human pose Î. The human shape Î¦ is defined as a low-dimensional vector.

## B. System Overview

We begin with introducing basic setups. We follow existing SLAM methods [3], [5] to treat the first camera frame as the world frame. The absolute camera pose $\mathbf { T } _ { t }$ represents the rotation and translation from the t-th camera frame to the world frame. Given sequential RGB images, we use Mask R-CNN [36] to segment non-rigid humans, rigid items, and static background. Based on segmentation results, we introduce several sets of Gaussians to express different parts of a dynamic environment. Given a camera pose, each set of Gaussians is rendered as an RGB patch and a depth patch, instead of complete RGB-D images. Without loss of generality, we consider the scene with a single human or item for illustration. Our method can be easily extended to environments with multiple dynamic objects, as will be shown in the experiments.

<!-- image-->  
Fig. 1. Overview of our SLAM method. Given sequential RGB-D images obtained in a dynamic environment, our method can not only reconstruct the static background and localize the camera, but also map the dynamic foreground. By optimizing Gaussians based on appearance and geometric constraints, our method can provide photo-realistic scene representation and accurate camera localization.

As shown in Fig. 1, our SLAM method is composed of three main modules to 1) map the dynamic foreground, 2) map the static background, and 3) localize the camera. Following dominant SLAM methods [31], [32], our mapping modules rely on the rough camera pose estimated by the localization module. The localization module exploits Gaussians reconstructed by the mapping modules at the previous times.

1) Mapping of Dynamic Foreground: For non-rigid humans, we attach Gaussians to the SMPL model and design a neural network to express the human deformation over time. Then for human motion expression, we exploit the 3D root joint of the SMPL model to transform humans to the camera frame, and further use the camera pose to transform humans to the world frame. For rigid items, we associate dynamic Gaussians at different times based on the optical flows, followed by aligning these Gaussians to compute the item motions. Moreover, we add new Gaussians based on a new observation mask. Details are available in Section IV.

2) Mapping of Static Background: We generate a local map to manage multiple images that partially observe the same area. We optimize Gaussians within such a local map based on multiple-view appearance constraints. Moreover, we propose an optimization strategy between neighboring local maps based on both geometric and appearance constraints. The geometric constraint is formulated as an iterative alignment between the centers of Gaussians from neighboring local maps. Moreover, we integrate appearance constraint into each iteration. Details are available in Section V.

3) Camera Localization: We propose a two-stage strategy to estimate the camera pose in a coarse-to-fine manner. In the first stage, we only use the static background. In the second stage, we exploit both static background and dynamic foreground. Our localization simultaneously leverages the appearance and geometric constraints with respect to the camera pose. For one thing, we use the camera pose to render Gaussians as RGB-D patches, formulating the appearance constraints. For another, we use the camera pose to project Gaussians, generating the projected optical flows. By aligning the observed and projected optical flows, we define the geometric constraint. Details are available in Section VI.

## IV. MAPPING OF DYNAMIC FOREGROUND

In this section, we present how we map dynamic foreground in the world frame. We aim to 1) model their 3D structures and 2) describe their motions and/or deformations. We first introduce non-rigid humans, followed by rigid items.

## A. Non-rigid Humans

Our 3D human representation is based on dynamic GS constrained by the SMPL model. We first introduce the initialization of Gaussians, given the first RGB-D images. Then we present how we model the deformation and motion of Gaussians based on two neighboring RGB-D images.

1) Initialization of Gaussians: As shown in Fig. 2, given the first RGB image, we employ ReFit [37] to estimate the pose and shape parameters Î and Î¦. We use these parameters to generate an SMPL mesh in the SMPL frame. Based on the rotation Rroot1 and translation troot1 associated with the root joint, we transform the generated SMPL mesh to the camera frame. Note that the translation $\mathbf { t } _ { 1 } ^ { \mathrm { r o o t } }$ is up-to-scale, i.e., the scale of SMPL mesh may not fit the real structure due to the inherent scale ambiguity of the perspective projection [37]. To determine the real scale of the SMPL mesh, we leverage the root joint. Specifically, ReFit provides the coordinates of the 2D root joint in the first RGB image. We back-project this joint into 3D using its associated depth provided by the depth image, obtaining the 3D root joint $\mathbf { r } _ { 1 }$ in the camera frame. Recall that 3D root joint is the origin of the SMPL frame. Therefore, $\mathbf { r } _ { 1 }$ approximates to the real-scale translation from SMPL frame to the camera frame. We compute the relative ratio between $\mathbf { r } _ { 1 }$ and the above up-to-scale translation $\mathbf { t } _ { 1 } ^ { \mathrm { r o o t } }$ by $\lVert \mathbf { r } _ { 1 } \rVert / \lVert \mathbf { t } _ { 1 } ^ { \mathrm { r o o t } } \rVert$ , and use this ratio to re-scale the SMPL mesh.

<!-- image-->  
Fig. 2. Initialization of human Gaussians. Given the first RGB-D images, we generate an SMPL mesh up to scale, followed by transforming it to the camera frame based on the estimated transformation of root joint. We further determine the real scale of SMPL mesh based on the depth of root joint. Then we attach a set of Gaussians to the re-scaled SMPL mesh, and optimize these Gaussians based on the appearance constraint.

Given the re-scaled SMPL mesh, we attach a set of Gaussians to its triangular facets. The center, scale, and rotation of a Gaussian can be roughly determined by the center, area, and normal of the corresponding facet, respectively. Since SMPL mesh is typically not associated with color information, we randomize the opacity and spherical harmonics of Gaussians. To optimize these human Gaussians denoted by $\mathcal { G } _ { 1 }$ , we render them as RGB and depth patches: $\tilde { I } _ { 1 } ( \mathcal { G } _ { 1 } ) , \tilde { D _ { 1 } } ( \mathcal { G } _ { 1 } ) = \pi [ \mathcal { G } _ { 1 } ]$ Then we use these rendered patches and their corresponding observed patches to define the photometric and depth losses, optimizing the Gaussian $\mathcal { G } _ { 1 }$

$$
\operatorname* { m i n } _ { \mathcal { G } _ { 1 } } \lambda _ { \mathrm { P } } \cdot L _ { \mathrm { P } } \Big ( \tilde { I } _ { 1 } ( \mathcal { G } _ { 1 } ) \Big ) + \lambda _ { \mathrm { D } } \cdot L _ { \mathrm { D } } \Big ( \tilde { D } _ { 1 } ( \mathcal { G } _ { 1 } ) \Big ) ,\tag{4}
$$

where $\lambda _ { \mathrm { P } }$ and $\lambda _ { \mathrm { D } }$ denote coefficients of the photometric and depth losses, respectively.

2) Update of Gaussians: For a dynamic human, its associated Gaussians initialized above will be continuously updated over time. Human typically exhibits two types of dynamics, i.e., deformation and transformation between times t and $t + 1$ $\left( t \geq 1 \right)$ . First, we estimate the deformation of Gaussians in the SMPL frame. We begin with using ReFit to estimate the human poses $\Theta _ { t }$ and $\Theta _ { t + 1 }$ based on the RGB images $I _ { t }$ and $I _ { t + 1 }$ respectively. Intuitively, the variation of human pose leads to the variation of geometric attributes of Gaussians. As shown in Fig. 3, to model this variation, we propose a neural network D. The input of this network is the variation of the human pose $\Delta \Theta _ { t , t + 1 }$ from time t to time t + 1. It encodes the transformation variation of each ordinary joint (except for the root joint), i.e., $\Delta \Theta _ { t , t + 1 } = [ \Delta \theta _ { t , t + 1 } ^ { 1 } , \cdot \cdot \cdot , \Delta \theta _ { t , t + 1 } ^ { i } , \cdot \cdot \cdot , \Delta \theta _ { t , t + 1 } ^ { 2 3 } ]$ The transformation variation $\Delta \theta _ { t , t + 1 } ^ { i }$ of the i-th ordinary joint is computed by

<!-- image-->  
Fig. 3. Update of human Gaussians. In the SMPL frame, we use a neural network D to deform Gaussians at time t into new Gaussians at time t + 1. Then we transform the deformed Gaussians to the (t + 1)-th camera frame using the transformation associated with the root joint. Finally, we optimize these Gaussians and the network D based on the appearance constraint.

$$
\Delta \theta _ { t , t + 1 } ^ { i } = \prod _ { j \in \Omega _ { i } } \theta _ { t + 1 } ^ { j } ( \prod _ { j \in \Omega _ { i } } \theta _ { t } ^ { j } ) ^ { - 1 } ,\tag{5}
$$

where $\Omega _ { i }$ represents the set of parent joints of the i-th joint, $\theta _ { t } ^ { j }$ and $\theta _ { t + 1 } ^ { j }$ represent the transformations of the j-th parent joint at times t and $t + 1$ respectively. Given the variation of the human pose $\Delta \Theta _ { t , t + 1 }$ , the network D can predict the variation in position $\Delta \mu _ { t , t + 1 }$ and rotation $\Delta \mathbf { R } _ { t , t + 1 }$ of a Gaussian located at the position $\pmb { \mu } _ { t }$ in the SMPL frame:

$$
\Delta \pmb { \mu } _ { t , t + 1 } , \Delta \mathbf { R } _ { t , t + 1 } = \mathscr { D } \big ( \mathscr { E } ( \pmb { \mu } _ { t } ) , \Delta \Theta _ { t , t + 1 } \big ) ,\tag{6}
$$

where $\mathcal { E } ( \cdot )$ denotes positional encoding. The architecture of the network D is based on the multi-layer perceptron (MLP). Through this network, we can obtain the geometric attributes of Gaussians at time t + 1 in the SMPL frame by

$$
\pmb { \mu } _ { t + 1 } = \pmb { \mu } _ { t } + \Delta \pmb { \mu } _ { t , t + 1 } , \ \mathbf { R } _ { t + 1 } = \Delta \mathbf { R } _ { t , t + 1 } \mathbf { R } _ { t } .\tag{7}
$$

$\pmb { \mu } _ { t + 1 }$ and ${ \bf R } _ { t + 1 }$ belong to the attributes of Gaussians $\mathcal { G } _ { t + 1 }$ at time $t + 1$ . Therefore, $\mathcal { G } _ { t + 1 }$ can be treated as the function with respect to the weights of the above network D, which is denoted by $\mathcal { G } _ { t + 1 } ( \mathcal { D } )$ . The other Gaussian attributes (color, opacity, and scale) are not changed by the network, i.e., they are shared by the Gaussians $\mathcal { G } _ { t }$ and $\mathcal { G } _ { t + 1 }$ at times t and $t + 1$ respectively.

Second, we model the transformation of Gaussians in the world frame. We begin with expressing the status of Gaussians with respect to each camera frame. Let us take the (k + 1)-th camera frame for example. Similar to the above Gaussian initialization, we use the RGB image $I _ { t + 1 }$ and depth image $D _ { t + 1 }$ to compute the rotation ${ \bf R } _ { t + 1 } ^ { \mathrm { r o o t } }$ and real-scale translation $\mathbf { r } _ { t + 1 }$ associated with the 3D root joint.1 We use these parameters to transform the above deformed Gaussians $\mathcal { G } _ { t + 1 } ( \mathcal { D } )$ from the SMPL frame to the (t + 1)-th camera frame. Then we render the transformed Gaussisans as RGB and depth patches: $\tilde { I } _ { t + 1 } \big ( \mathcal { G } _ { t + 1 } ( \mathcal { D } ) \big ) , \tilde { D } _ { t + 1 } \big ( \mathcal { G } _ { t + 1 } ( \mathcal { D } ) \big ) = \pi [ \mathcal { G } _ { t + 1 } ( \mathcal { D } ) ]$ . We use these rendered patches and their corresponding observed RGB-D patches to define the photometric and depth losses, optimizing the attributes (color, opacity, and scale) of Gaussians $\mathcal { G } _ { t + 1 }$ and the network D:

$$
\operatorname* { m i n } _ { \mathcal { G } _ { t + 1 } , \mathcal { D } } \lambda _ { \mathrm { P } } \cdot L _ { \mathrm { P } } \Big ( \tilde { I } _ { t + 1 } \big ( \mathcal { G } _ { t + 1 } ( \mathcal { D } ) \big ) \Big ) + \lambda _ { \mathrm { D } } \cdot L _ { \mathrm { D } } \Big ( \tilde { D } _ { t + 1 } \big ( \mathcal { G } _ { t + 1 } ( \mathcal { D } ) \big ) \Big ) .\tag{8}
$$

At the same time, since the above partial attributes of Gaussians $\mathcal { G } _ { t + 1 }$ are shared with Gaussians $\mathcal { G } _ { t } .$ , we additionally consider the appearance constraints at time t. We fix the exclusive parameters (centers and rotations) of Gaussians $\mathcal { G } _ { t }$ which have been optimized at the time t, and optimize the shared parameters by rendering $\mathcal { G } _ { t }$ in the t-th camera frame. Therefore, the shared attributes are simultaneously constrained by two views, which ensures the temporal continuity of the Gaussian attributes and improves the reliability of Gaussian optimization. After optimizing Gaussians $\mathcal { G } _ { k + 1 }$ in the (k + 1)- th camera frame, we use the camera pose $\mathbf { T } _ { t + 1 }$ to transform these Gaussians to the world frame. By repeating the above procedures, we can express the transformation of Gaussians in the world frame.

In addition, to improve the robustness of Gaussian optimization, we introduce a simple but effective human scale regularization loss. Specifically, the depth image is inevitably affected by noise. Accordingly, the optimized 3D humans at different times may render plausible RGB appearance but exhibit inconsistent sizes. To solve this problem, we compute the average scale sË of all the Gaussians initialized by the first image, and treat it as the constant standard scale. At any time t, we compute the average scale $\bar { s } _ { t }$ and force it to be similar to sË. Based on this constraint, the sizes of the optimized 3D humans at different times fall into the same range.

## B. Rigid Items

Our 3D item representation is based on dynamic GS constrained by rigid transformation. We first introduce the initialization of Gaussians, given the first RGB-D images. Then we present how we model the rigid transformation of Gaussians based on two neighboring images. Finally, given new RGB-D images, we introduce the addition of Gaussians.

1) Initialization of Gaussians: Given the first RGB-D images, we back-project the pixels associated with the rigid item into 3D, obtaining a 3D point cloud in the camera frame. Then we use the position and color of each point to initialize the center and color of a Gaussian. The other attributes of these Gaussisans are randomized. The optimization of these Gaussians is analogous to the above operations for humans. We render these Gaussians as RGB and depth patches, and define the photometric and depth losses for attribute optimization.

2) Rigid Transformation of Gaussians: Let us consider the t-th and (t + 1)-th images to illustrate how we estimate the rigid transformation of Gaussians in a coarse-to-fine manner (see Fig. 4). For the rough estimation, we first establish 2D-2D point correspondences in two RGB images by the optical flow estimation [38]. Then we back-project the matched 2D pixels into 3D using their associated depths, obtaining two point clouds in respective camera frames. Based on the established 2D-2D point correspondences, 3D-3D point correspondences of two point clouds are known. We further use the camera poses $\mathbf { T } _ { t }$ and $\mathbf { T } _ { t + 1 }$ to transform these two point clouds to the world frame, respectively. Finally, we compute the rigid transformation $\Delta \mathbf { M } _ { t , t + 1 }$ between two point clouds using 3D-3D correspondences via the singular value decomposition [39].

<!-- image-->  
Fig. 4. Rigid transformation and addition of item Gaussians. We first estimate the optical flow and back-project depth images to establish 3D-3D point correspondences. Then we use these correspondences to roughly estimate the transformation, followed by optimizing Gaussians and transformation based on appearance constraint. Finally, we estimate the new observation mask that guides the addition of Gaussians using appearance constraint.

To further improve the accuracy of the transformation $\Delta \mathbf { M } _ { t , t + 1 }$ , we leverage the appearance constraint. In the world frame, we first use $\Delta \mathbf { M } _ { t , t + 1 }$ to transform the centers of Gaussians from time t to the time t + 1:

$$
\pmb { \mu } _ { t + 1 } ^ { \mathcal { W } } = \Delta \mathbf { M } _ { t , t + 1 } ( \pmb { \mu } _ { t } ^ { \mathcal { W } } ) .\tag{9}
$$

We further transform these centers to the (k + 1)-th camera frame using the camera pose $\mathbf { T } _ { t + 1 }$ . Accordingly, Gaussians can be treated as a function with respect to the transformation $\Delta \mathbf { M } _ { t , t + 1 }$ , which is denoted by $\mathscr { G } _ { t + 1 } ( \Delta \mathbf { M } _ { t , t + 1 } )$ . The other attributes of these Gaussians (color, opacity, rotation, and scale) are shared with Gaussians at time t. We render Gaussians at time $t + 1$ as RGB and depth patches: $\begin{array} { r l } { \tilde { I } _ { t + 1 } \big ( \mathcal { G } _ { t + 1 } ( \Delta \mathbf { M } _ { t , t + 1 } ) \big ) , \tilde { D } _ { t + 1 } \big ( \mathcal { G } _ { t + 1 } ( \Delta \mathbf { M } _ { t , t + 1 } ) \big ) } & { { } = } \end{array}$ $\pi [ \mathcal { G } _ { t + 1 } ( \Delta \mathbf M _ { t , t + 1 } ) ]$ . Then we use these rendered patches and their corresponding observed patches to define the photometric and depth losses, optimizing the attributes of Gaussians $\mathcal { G } _ { t + 1 }$ (except for the center) and the transformation $\Delta \mathbf { M } _ { t , t + 1 } \colon$

$$
\begin{array} { r } { \underset { \mathcal { G } _ { t + 1 } , \Delta \mathbf { M } _ { t , t + 1 } } { \operatorname* { m i n } } \lambda _ { \mathrm { P } } \cdot L _ { \mathrm { P } } \Big ( \tilde { I } _ { t + 1 } \big ( \mathcal { G } _ { t + 1 } ( \Delta \mathbf { M } _ { t , t + 1 } ) \big ) \Big ) + } \\ { \lambda _ { \mathrm { D } } \cdot L _ { \mathrm { D } } \Big ( \tilde { D } _ { t + 1 } \big ( \mathcal { G } _ { t + 1 } ( \Delta \mathbf { M } _ { t , t + 1 } ) \big ) \Big ) . } \end{array}\tag{10}
$$

Moreover, similar to the above operation for humans, we additionally consider the appearance constraints at time t. We fix the centers of Gaussians $\mathcal { G } _ { t }$ which have been optimized at time t, and optimize the shared parameters by rendering $\mathcal { G } _ { t }$ in the t-th camera frame. Therefore, the shared attributes are simultaneously constrained by two views.

3) Addition of Gaussians: Due to viewpoint change and camera movement, some parts of a 3D item, which cannot be observed at time t, become visible at time $t + 1$ . We aim to map these 3D parts, i.e., add some Gaussians to express these parts. Given a new RGB image $I _ { t + 1 }$ , we first identify the pixels that correspond to the newly observed but not reconstructed 3D parts of the item. As shown in Fig. 4, we transform the above optimized Gaussians $\mathcal { G } _ { t + 1 }$ to the (t + 1)-th camera frame using the camera pose $\mathbf { T } _ { t + 1 }$ , and render these Gaussians into a depth patch $\tilde { D } _ { t + 1 }$ . If some pixels of the item in the RGB image $I _ { t + 1 }$ are not associated with the rendered depths, we consider that these pixels have not been associated with 3D Gaussians. Accordingly, these pixels constitute the ânew observation maskâ.

Then we back-project the pixels belonging to the mask into 3D based on their associated depths provided by the depth image $D _ { t + 1 }$ , obtaining a set of colored 3D points in the $( t + 1 )$ -th camera frame. Then we use the positions and colors of these 3D points to initialize the centers and colors of a set of new Gaussians. The optimization of these Gaussians is analogous with the above operation for initialization of item Gaussians. We render these Gaussians as RGB and depth patches, and define the photometric and depth losses for attribute optimization.

## V. MAPPING OF STATIC BACKGROUND

In this section, we present how we map static background. To guarantee high accuracy, we introduce local maps to reconstruct and manage Gaussians. Our main technical novelty lies in an optimization strategy between neighboring local maps. Different from related work [40], our method simultaneously leverages the geometric and appearance constraints, which can effectively reduce the accumulated error.

## A. Generation of A Single Local Map

Recall that to optimize the dynamic foreground at time t+1, we only use images obtained at times t + 1 and t. The reason is that the states of dynamic objects change quickly, and the previous images are not well-aligned to the current states. By contrast, the attributes of background Gaussians remain consistent within a wider time window. Therefore, we involve a larger number of sequential images in mapping a part of static background. A set of Gaussians optimized by these images is called a local map. In the following, let us consider a local map starting from time t for illustration.

1) Initialization: We initialize Gaussians following the operations for rigid items (see Section IV-B1). Briefly, given the RGB-D images $( I _ { t } , D _ { t } )$ that is the first element of a local map, we generate a colored 3D point cloud and use it to initialize the centers and colors of Gaussians. Then we render these Gaussians as RGB and depth patches, and exploit the photometric and depth losses for optimization. After that, we use the camera pose to transform the optimized Gaussians from the t-th camera frame to the world frame.

Given the new RGB-D images $( I _ { t + m } , D _ { t + m } ) \ ( m \geq 1 )$ , we identify whether these images belong to the current local map. If their associated camera pose does not significantly differ from that of the images $( I _ { t } , D _ { t } )$ , we treat $\left( I _ { t + m } , D _ { t + m } \right)$ as a new element of the current local map. Otherwise, we initialize a new local map.

<!-- image-->  
Fig. 5. Optimization between n-th and (n + 1)-th local maps. Here, we show the centers of Gaussians. We iteratively align Gaussians $\mathcal { G } _ { n + 1 }$ to Gaussians ${ \mathcal { G } } _ { n }$ based on geometric constraint. To improve the robustness of optimization, we integrate the appearance constraint into each iteration. Gaussians $\mathcal { G } _ { n + 1 }$ are rendered by multiple cameras associated with Gaussians $\mathcal { G } _ { n }$

2) Update: Assume that $\left( I _ { t + m } , D _ { t + m } \right)$ is a new element of a local map. From this view, some parts of 3D background become newly observable. We follow the operation for rigid items to determine the new observation mask and add Gaussians associated with this mask (see Section IV-B3). Then we transform these Gaussians to the world frame W and further append them to the local map. After generating a local map $\mathcal { G } ^ { \mathcal { W } }$ using M images, we optimize this map based on multiple-view constraints. For each of M cameras, we use the m-th camera pose $\mathbf { T } _ { m }$ to transform Gaussians ${ \mathcal { G } } ^ { \mathcal { W } }$ from the world frame to the m-th camera frame and then render these Gaussians as RGB and depth patches: $\tilde { I } _ { m } ( \mathcal { G } ^ { \nu } ) , \tilde { D } _ { m } ( \mathcal { G } ^ { \nu } ) =$ $\pi [ \mathcal { G } ^ { \mathcal { W } } , \mathbf { T } _ { m } ] \left( 1 \leq m \leq M \right)$ . We use these rendered patches and their corresponding observed patches to define the photometric and depth losses, optimizing the Gaussian ${ \mathcal { G } } ^ { \mathcal { W } }$

$$
\operatorname* { m i n } _ { \mathcal { G } ^ { \mathcal { W } } } \sum _ { m = 1 } ^ { M } \lambda _ { \mathrm { P } } \cdot L _ { \mathrm { P } } \Big ( \tilde { I } _ { m } ( \mathcal { G } ^ { \mathcal { W } } ) \Big ) + \lambda _ { \mathrm { D } } \cdot L _ { \mathrm { D } } \Big ( \tilde { D } _ { m } ( \mathcal { G } ^ { \mathcal { W } } ) \Big ) .\tag{11}
$$

## B. Optimization Between Neighboring Local Maps

In the above, we generate and optimize each local map independently. Neighboring local maps lack effective constraints to connect them. Accordingly, these local maps may exhibit appearance inconsistency and geometric non-alignment. To solve this problem, we leverage the co-visibility constraint between neighboring local maps. For ease of understanding, let us first consider the n-th and $( n + 1 )$ -th local maps for illustration.

For each local map, we use the centers of Gaussians to generate a 3D point cloud in the world frame. Given two neighboring point clouds that partly deviate from each other due to noise, a straightforward strategy is to use the iterative closest points (ICP) algorithm [39] to align them. However, in practice, the deviation between two point clouds may be relatively large. The pure ICP-based strategy is prone to get stuck in a local optimum. To align two local maps more robustly, we integrate the appearance constraint into ICP.

As shown in Fig. 5, we first follow ICP to establish tentative 3D-3D point correspondences based on the shortest distance, and then use these correspondences to roughly estimate the relative transformation $\Delta \mathbf { T } _ { n , n + 1 }$ . After that, instead of directly using $\Delta \mathbf { T } _ { n , n + 1 }$ to update point correspondences, we exploit the appearance losses to optimize $\Delta \mathbf { T } _ { n , n + 1 }$ . Specifically, in the world frame, we use $\Delta \mathbf { T } _ { n , n + 1 }$ to transform the $( n + 1 )$ )-th local map $\mathcal { G } _ { n + 1 } ^ { \mathcal { W } }$ . This map is further transformed to the m-th camera frame of the n-th local map based on the m-th camera pose $( 1 \leq m \leq M )$ Then we render these Gaussians into RGB and depth patches: $\tilde { I } _ { m } ( \Delta \mathbf { T } _ { n , n + 1 } ) , \tilde { D } _ { m } ( \Delta \mathbf { T } _ { n , n + 1 } ) \ =$ $\pi [ \mathcal { G } _ { n + 1 } ^ { \mathcal { W } } , \Delta \mathbf { T } _ { n , n + 1 } , \mathbf { T } _ { m } ]$ . We use these rendered patches and their corresponding observed patches to define the photometric and depth losses, optimizing the transformation $\Delta \mathbf { T } _ { n , n + 1 } \colon$

$$
\begin{array} { r } { \displaystyle \operatorname* { m i n } _ { \Delta \mathbf { T } _ { n , n + 1 } } \sum _ { m = 1 } ^ { M } \lambda _ { \mathrm { P } } \cdot L _ { \mathrm { P } } \Big ( \tilde { I } _ { m } ( \Delta \mathbf { T } _ { n , n + 1 } ) \Big ) + } \\ { \displaystyle \sum _ { m = 1 } ^ { M } \lambda _ { \mathrm { D } } \cdot L _ { \mathrm { D } } \Big ( \tilde { D } _ { m } ( \Delta \mathbf { T } _ { n , n + 1 } ) \Big ) . } \end{array}\tag{12}
$$

The optimized transformation $\Delta \mathbf { T } _ { n , n + 1 }$ is then fed back to the second round of iteration to establish more accurate 3D-3D point correspondences. We repeat the above process until convergence. The appearance and geometric losses complement each other, which can help optimization converge more reliably. Based on the optimized transformation $\Delta \mathbf { T } _ { n , n + 1 }$ , we can better align neighboring local maps.

In addition, we can easily extend the above optimization method to loop closure. Briefly, we can use an arbitrary loop detection algorithm to identify two local maps that partly overlap each other. Then our method can estimate the transformation between two local maps and use this transformation to perform trajectory correction.

## VI. CAMERA LOCALIZATION

In this section, we introduce how we localize the camera using not only static background, but also dynamic foreground. Our method leverages both geometric and appearance constraints. In the following, we consider the t-th and (t + 1)-th RGB-D images for illustration. We assume that both static background and dynamic foreground at time t have been reconstructed, and the camera pose $\mathbf { T } _ { t }$ has been estimated. We aim to estimate the camera pose at time t + 1.

## A. Two-stage Localization Strategy

We propose a two-stage strategy to estimate the camera pose in a coarse-to-fine manner. In the first stage, we only use the static background to obtain the initial estimation of the camera pose. In the second stage, we simultaneously exploit the static background and dynamic foreground to refine the camera pose. Note that we do not consider dynamic foreground in the first stage. The reason is that at time $t + 1$ we do not have a prior 3D map of dynamic foreground in the world frame which is consistent with the observed

RGB-D images. By contrast, this 3D map can be roughly obtained after the first-stage estimation (see below), and thus can be used in the second stage. Compared with the previous works [9]â[11] that only leverage static background, the above strategy can use more observations to compensate for noise and thus improve the accuracy of camera localization. As to the constraints for optimization, we introduce not only the appearance constraints of Gaussians, but also the geometric constraints of optical flows. In the following, we first consider the appearance constraints to illustrate the pipeline of our twostage localization strategy. Then we will present the integration of geometric constraints in the next subsection.

In the first stage, given the background Gaussians $B _ { t } ^ { \mathcal { W } }$ obtained at time t in the world frame (see Section V), we use the unknown-but-sought camera pose $\mathbf { T } _ { t + 1 }$ to transform them to the $( t + 1 )$ -th camera frame. We fix the attributes of the transformed Gaussians and render these Gaussians as RGB and depth patches: $\tilde { I } _ { t + 1 } ( \mathbf { T } _ { t + 1 } ) , \tilde { D } _ { t + 1 } ( \mathbf { T } _ { t + 1 } ) = \pi [ \mathcal { B } _ { t } ^ { \mathcal { W } } , \mathbf { T } _ { t + 1 } ]$ Then we use these rendered patches and their corresponding observed RGB-D patches to define the photometric and depth losses, optimizing the transformation $\mathbf { T } _ { k + 1 } \colon$

$$
\operatorname* { m i n } _ { \mathbf { T } _ { t + 1 } } \ \lambda _ { \mathrm { P } } \cdot L _ { \mathrm { P } } \Big ( \tilde { I } _ { t + 1 } ( \mathbf { T } _ { t + 1 } ) \Big ) + \lambda _ { \mathrm { D } } \cdot L _ { \mathrm { D } } \Big ( \tilde { D } _ { t + 1 } ( \mathbf { T } _ { t + 1 } ) \Big ) .\tag{13}
$$

For optimization, we initialize the camera pose $\mathbf { T } _ { t + 1 }$ based on the known camera pose $\mathbf { T } _ { t }$ and constant velocity motion model used in ORB-SLAM2 [3].

In the second stage, we simultaneously consider the constraints of static background and dynamic foreground to refine the coarse camera pose $\mathbf { T } _ { t + 1 }$ estimated above. The operation for static background is similar to that at the first stage. Therefore, we mainly introduce the usage of the dynamic foreground. Recall that given RGB-D images $( I _ { t + 1 } , D _ { t + 1 } )$ and coarse camera pose $\mathbf { T } _ { t + 1 }$ obtained in the first stage, we can roughly estimate a set of foreground Gaussians $\bar { \mathcal { F } } _ { t + 1 } ^ { \mathcal { W } }$ in the world frame (see Section IV). Compared with background Gaussians optimized by multiple-view constraints, foreground Gaussians may not be accurate enough due to fewer views, and thus are treated as variables to optimize. We use the coarse camera pose $\mathbf { T } _ { t + 1 }$ to transform these Gaussians to the $( t + 1 ) \cdot$ th camera frame, and render them as RGB and depth patches: $\tilde { I } _ { t + 1 } ( \mathcal { F } _ { t + 1 } ^ { \mathcal { W } } , \mathbf { T } _ { t + 1 } ) , \tilde { D } _ { t + 1 } ( \mathcal { F } _ { t + 1 } ^ { \mathcal { W } } , \mathbf { T } _ { t + 1 } ) \ = \ \pi [ \mathcal { F } _ { t + 1 } ^ { \mathcal { W } } , \mathbf { T } _ { t + 1 } ]$ . We use these rendered patches and corresponding observed RGB-D patches to define the photometric and depth losses, optimizing Gaussians $\mathbf { \mathcal { F } } _ { t + 1 } ^ { \mathcal { W } }$ and camera pose $\mathbf { T } _ { t + 1 }$

$$
\begin{array} { r } { \underset { \mathbf { T } _ { t + 1 } , \mathcal { F } _ { t + 1 } ^ { \mathcal { W } } } { \operatorname* { m i n } } \lambda _ { \mathrm { P } } \cdot L _ { \mathrm { P } } \Big ( \tilde { I } _ { t + 1 } ( \mathcal { F } _ { t + 1 } ^ { \mathcal { W } } , \mathbf { T } _ { t + 1 } ) \Big ) + } \\ { \lambda _ { \mathrm { D } } \cdot L _ { \mathrm { D } } \Big ( \tilde { D } _ { t + 1 } ( \mathcal { F } _ { t + 1 } ^ { \mathcal { W } } , \mathbf { T } _ { t + 1 } ) \Big ) . } \end{array}\tag{14}
$$

Since the above strategy uses as much information as possible, it can effectively improve the accuracy of localization, as will be shown in the experiments.

## B. Geometric Constraint of Optical Flows

The above appearance constraint-based localization can lead to accurate camera localization when the movements of cameras and dynamic objects are moderate. However, this strategy may be unreliable in the case of large movement. The reason is that the rendered and observed pixels are prone to be wrongly associated, resulting in a local optimum of optimization. To improve the robustness of camera localization, we further leverage the geometric constraints by associating 3D Gaussians and 2D optical flows. We consider the optical flows of both static background (for the above first- and second-stage localization) and dynamic foreground (for the above secondstage localization).

<!-- image-->  
Fig. 6. Geometric constraint of optical flows. We mainly consider the static background for illustration. In the image $I _ { k + 1 } .$ , we compute the projected optical flow Ëf of the pixel p based on the weighted combination of 2D vectors $\mathbf { q } _ { k } - \mathbf { p }$ (shown in black). The projected optical flow Ëf should overlap the observed optical flow f estimated using the image pair $( I _ { k } , I _ { k + 1 } )$ . For dynamic foreground, the generation of a projected optical flow additionally involves the centers $\{ \mu _ { k } ^ { \mathcal { W } } \}$ of dynamic Gaussians at time k + 1.

1) Static Background: Optical flows of static background are caused by the camera movement, which provides a clue for camera pose estimation. First, we establish the connection between optical flows and background Gaussians. As shown in Fig. 6, a pixel p in the image $I _ { t }$ corresponds to a set of 3D Gaussians. The color of this pixel is determined by a set of 2D Gaussians projected from these 3D Gaussians. Each 2D Gaussian is associated with a blending weight $\alpha _ { k }$ to encode its importance for rendering (see Section III-A1). We neglect 2D Gaussians whose weights are too small and assign the weights of the remaining K 2D Gaussians to their corresponding 3D Gaussians. Then we transform these weighted 3D Gaussians from the world frame to the (t + 1)-th camera frame using the unknown-but-sought camera pose $\mathbf { T } _ { t + 1 }$ . After that, we project the centers of these 3D Gaussians onto the (t + 1)-th image, obtaining K projected points $\{ \mathbf { q } _ { k } ( \mathbf { T } _ { t + 1 } ) \} _ { k = 1 } ^ { K }$ with respect to the camera pose $\mathbf { T } _ { t + 1 }$ . The pixel p and each projected point $\mathbf q _ { k }$ define a 2D vector by ${ \bf q } _ { k } ( { \bf T } _ { t + 1 } ) - { \bf p }$ . We compute the weighted sum of these vectors based on the blending weights $\alpha _ { k }$ of Gaussians, generating the optical flow Ëf of the pixel p:

$$
\tilde { \mathbf { f } } ( \mathbf { T } _ { t + 1 } ) = \sum _ { k = 1 } ^ { K } \alpha _ { k } \cdot \Bigl ( \mathbf { q } _ { k } ( \mathbf { T } _ { t + 1 } ) - \mathbf { p } \Bigr ) .\tag{15}
$$

Since $\tilde { \mathbf { f } }$ is generated by projecting 3D Gaussians, we call it the âprojectedâ optical flow. Ëf can be regarded as a function with respect to the camera pose $\mathbf { T } _ { t + 1 }$

The above generation of optical flow is in 3D. In the following, we compute the optical flow in 2D. Given the RGB images $I _ { t }$ and $I _ { t + 1 }$ , we utilize the RAFT [38] algorithm to track pixels. For the pixel p in the image $I _ { t } ,$ , we obtain its optical flow f and call it the âobservedâ optical flow. Ideally, the projected and observed flows should overlap each other. Based on this constraint, we minimize the difference between $\tilde { \mathbf { f } } ( \mathbf { T } _ { t + 1 } )$ in Eq. (15) and f to optimize the camera pose $\mathbf { T } _ { t + 1 }$ . In practice, we establish J pairs of projected and observed optical flows $\{ ( \tilde { \mathbf { f } } _ { j } , \mathbf { f } _ { j } ) \} _ { j = 1 } ^ { J }$ , formulating the camera pose optimization by

$$
\operatorname* { m i n } _ { \mathbf { T } _ { t + 1 } } \sum _ { j = 1 } ^ { J } \| \tilde { \mathbf { f } } _ { j } ( \mathbf { T } _ { t + 1 } ) - \mathbf { f } _ { j } \| _ { 2 } .\tag{16}
$$

2) Dynamic Foreground: Optical flows of dynamic foreground are caused by not only the camera movement, but also the motion of dynamic objects, compared with the above static foreground. As shown in Fig. 6, each point $\mathbf { q } _ { k }$ projected from a dynamic Gaussian is with respect to the camera pose $\mathbf { T } _ { t + 1 } ,$ as well as the Gaussian center $\mu _ { k } ^ { \mathcal { W } }$ in the world frame at time t + 1. Accordingly, we extend the computation of the projected optical flow of static background (see Eq. (15)) into

$$
\tilde { \mathbf { f } } ( \mathbf { T } _ { t + 1 } , \{ \pmb { \mu } _ { k } ^ { \mathcal { W } } \} _ { k = 1 } ^ { K } ) = \sum _ { k = 1 } ^ { K } \alpha _ { k } \cdot \Big ( \mathbf { q } _ { k } ( \mathbf { T } _ { t + 1 } , \pmb { \mu } _ { k } ^ { \mathcal { W } } ) - \mathbf { p } \Big ) .\tag{17}
$$

The projected optical flow $\tilde { \mathbf { f } }$ is additionally with respect to the Gaussian center $\mu _ { k } ^ { \mathcal { W } }$ , compared with Eq. (15). Then by analogy with the static foreground, we minimize the difference between the projected flow Ëf in Eq. (17) and the observed flow f extracted by RAFT algorithm. This achieves a joint optimization of the transformation $\mathbf { T } _ { t + 1 }$ and Gaussian centers $\{ \mu _ { k } ^ { \mathcal { W } } \} _ { k = 1 } ^ { K }$

## VII. EXPERIMENTS

In this section, we first introduce the experimental setup, and then compare our method with state-of-the-art approaches. After that, we conduct ablation study to validate the effectiveness of the proposed modules.

## A. Experimental Setup

1) Datasets: We follow dynamic SLAM methods [10], [11], [13], [22] to conduct experiments on Bonn RGB-D Dynamic dataset [43] and TUM RGB-D dataset [41]. Moreover, we consider NeuMan dataset [42] to additionally evaluate the accuracy of our human localization. For writing simplification, we denote the above datasets by Bonn, TUM, and NeuMan datasets, respectively. We provide basic information as follows.

â¢ Bonn dataset was obtained in indoor environments. It includes several sequences with one or more humans performing various actions and items exhibiting rigid transformations. It provides ground-truth camera trajectories to evaluate the accuracy of camera localization.

â¢ TUM dataset was established in indoor environments that involve both dynamic and static scenes. We use the sequences of dynamic scenes for evaluation. Similar to the above Bonn dataset, some sequences include multiple dynamic humans. This dataset also provides ground truth camera trajectories for evaluation of camera localization.

<!-- image-->  
(a) ESLAM [27]

<!-- image-->  
(b) MonoGS [31]

<!-- image-->

(c) Rodyn-SLAM [11]  
<!-- image-->  
(d) PG-SLAM (our)  
Fig. 7. Representative environment mapping comparison between various SLAM methods on Sequence walking_rpy of TUM dataset [41]. (a) ES-LAM and (b) MonoGS [31] are originally designed for static environments, and cannot handle this dynamic scene well. (c) Rodyn-SLAM only focuses on mapping static background. (d) Our PG-SLAM can reconstruct both static background and dynamic humans at different times.

â¢ NeuMan dataset was obtained in outdoor environments. The quality of the depth image is partly affected by the limited measurement distance of the depth camera. Each sequence contains one human with various poses. Different from the above datasets, this dataset not only offers ground-truth trajectories of cameras, but also the reference positions of dynamic humans.

2) Evaluation Metrics: As to 3D mapping, we mainly adopt the qualitative evaluation since some datasets lack ground truth 3D structures. To evaluate the accuracy of camera localization and human tracking, we first connect the camera centers and root joints of a human at different times to generate trajectories of camera and human, respectively. Then we adopt the widely-used absolute trajectory error [41], [44] to measure the difference between the estimated and ground truth trajectories. We report both root mean square error (RMSE) and standard deviation (SD) of this error. Unless otherwise specified, the unit of the reported values is centimeters.

<!-- image-->

<!-- image-->  
(b)

Fig. 8. Representative environment mapping results of our PG-SLAM on (a) Sequence bike of NeuMan dataset [42], and (b) Sequence moving_box of TUM dataset [41]. Our method can reconstruct not only static background, but also non-rigid human and rigid box at different times. The red and green dotted lines denote the trajectories of human and camera, respectively.  
<!-- image-->

<!-- image-->  
(b)  
Fig. 9. Representative human trajectories estimated by our PG-SLAM on (a) Sequence parking_lot and (b) Sequence seattle of NeuMan dataset [42]. The cyan and black lines denote the estimated and ground truth human trajectories, respectively. The centers of human Gaussians and the root joints of a human at some randomly selected times (such as $t _ { 1 } , t _ { 2 } , \cdots )$ are shown in gray and red points, respectively.

3) Implementation Details: We treat Gaussian-SLAM [33] as the baseline to develop our SLAM method. For appearance constraint, we set the weights of the photometric and depth losses $\lambda _ { \mathrm { P } }$ and $\lambda _ { \mathrm { D } }$ (see Sections IV, V, and VI) to 0.6 and $0 . 4 ,$ respectively. We minimize the loss based on Adam following [6], [32]. As introduced in Section IV-A2, the network D to model the human deformation is based on MLP. Its input is the concatenation of the positional encoding result and human pose variation. Eight layers map the input into a 256- dimensional feature vector. Then this feature is processed by two independent layers to predict the variation of position and rotation, respectively. We conduct experiments on a computer equipped with a CPU of E3-1226 and a GPU of RTX 4090.

TABLE I  
CAMERA LOCALIZATION COMPARISONS BETWEEN VARIOUS SLAM METHODS ON THREE DATASETS. WE REPORT THE ABSOLUTE TRAJECTORY ERROR OF CAMERA TRAJECTORY.
<table><tr><td></td><td colspan="10">Bonn dataset [43]</td><td colspan="3"></td></tr><tr><td>Sequences</td><td colspan="2">balloon</td><td colspan="2">balloon2</td><td colspan="2">ps_track</td><td colspan="2">ps_track2</td><td colspan="2">mv_box</td><td colspan="2">mv_box2</td><td colspan="2">Average</td></tr><tr><td></td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td></tr><tr><td>ESLAM [27]</td><td>22.6</td><td>12.2</td><td>36.2</td><td>19.9</td><td>48.0</td><td>18.7</td><td>51.4</td><td>23.2</td><td>8.4</td><td>3.5</td><td>17.7</td><td>7.5</td><td>31.4</td><td>14.7</td></tr><tr><td>MonoGS [31]</td><td>49.5</td><td>20.3</td><td>42.5</td><td>20.9</td><td>100.3</td><td>50.7</td><td>114.3</td><td>52.8</td><td>8.9</td><td>3.7</td><td>24.1</td><td>11.2</td><td>66.1</td><td>31.1</td></tr><tr><td>RoDyn-SLAM [11]</td><td>7.9</td><td>2.7</td><td>11.5</td><td>6.1</td><td>14.5</td><td>4.6</td><td>13.8</td><td>3.5</td><td>7.2</td><td>2.4</td><td>12.6</td><td>4.7</td><td>12.3</td><td>4.4</td></tr><tr><td>PG-SLAM (our)</td><td>6.4</td><td>2.2</td><td>7.3</td><td>3.4</td><td>5.0</td><td>1.9</td><td>8.5</td><td>2.8</td><td>4.6</td><td>1.3</td><td>7.0</td><td>2.0</td><td>6.5</td><td>2.2</td></tr><tr><td colspan="10">TUM dataset [41]</td><td colspan="7"></td></tr><tr><td>Sequences</td><td colspan="10">f3/wk_xyz f3/wk_hf</td><td colspan="5">f3/st_st f3/st_xyz</td></tr><tr><td></td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>RMSE SD</td><td>f3/st_rpy RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td><td>Average RMSE</td><td>SD</td></tr><tr><td>ESLAM [27]</td><td>45.7</td><td>28.5</td><td>60.8</td><td>27.9</td><td>93.6</td><td>20.7</td><td>5.3 3.3</td><td>8.4</td><td>5.7</td><td>0.9</td><td>0.5</td><td>5.4</td><td>3.9</td><td>31.4</td><td>12.9</td></tr><tr><td>MonoGS [31]</td><td>130.8</td><td>55.4</td><td>73.5</td><td>42.1</td><td>16.3</td><td>2.5</td><td>10.5 2.9</td><td>25.0</td><td>7.3</td><td>0.75</td><td>0.42</td><td>1.9</td><td>0.6</td><td>37.0</td><td>15.9</td></tr><tr><td>RoDyn-SLAM [11]</td><td>8.3</td><td>5.5</td><td>5.6</td><td>2.8</td><td>1.7</td><td>0.9</td><td>4.4 2.2</td><td>11.4</td><td>4.6</td><td>0.76</td><td>0.43</td><td>5.0</td><td>1.0</td><td>5.3</td><td>2.5</td></tr><tr><td>PG-SLAM (our)</td><td>6.8</td><td>2.9</td><td>11.7</td><td>4.4</td><td>1.4</td><td>0.6</td><td>4.0 1.5</td><td>5.4</td><td>2.4</td><td>0.72</td><td>0.39</td><td>1.5</td><td>0.5</td><td>4.5</td><td>1.8</td></tr><tr><td colspan="10">NeuMan dataset [42]</td><td colspan="7"></td></tr><tr><td colspan="10">citron jogging parkinglot</td><td colspan="7">seattle Average</td></tr><tr><td>Sequences</td><td colspan="10">bike</td><td colspan="5">RMSE</td></tr><tr><td>ESLAM [27]</td><td>RMSE 44.96</td><td>SD 20.87</td><td></td><td>RMSE 36.99</td><td>SD 20.22</td><td>RMSE 6.18</td><td>SD 3.12</td><td>RMSE 98.91</td><td></td><td>SD 41.75</td><td>RMSE 146.29</td><td>SD 57.41</td><td></td><td></td><td>SD 28.67</td></tr><tr><td>MonoGS [31]</td><td>1.41</td><td>0.53</td><td></td><td>30.94</td><td>15.35</td><td>16.47</td><td>6.52</td><td>3.03</td><td></td><td>1.42</td><td>0.86</td><td>0.34</td><td></td><td>66.66 10.52</td><td>4.83</td></tr><tr><td>RoDyn-SLAM [11]</td><td>1.38</td><td>0.54</td><td></td><td>4.62</td><td>3.51</td><td>1.02</td><td>0.46</td><td>3.07</td><td></td><td>1.20</td><td>2.65</td><td>1.02</td><td></td><td>2.54</td><td>1.35</td></tr><tr><td>PG-SLAM (our)</td><td>1.15</td><td>0.47</td><td></td><td>2.43</td><td>0.92</td><td>0.78</td><td>0.45</td><td>2.39</td><td></td><td>1.12</td><td>0.53</td><td>0.18</td><td></td><td>1.45</td><td>0.62</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## B. Comparison with State-of-the-art Approaches

1) Methods for Comparison: In the following, we denote our photo-realistic and geometry-aware SLAM method by PG-SLAM. We compare our method with state-of-the-art approaches introduced in Section II:

â¢ ESLAM [27]: The implicit representation-based method originally designed for static environments. It leverages the multi-scale axis-aligned feature planes.

â¢ MonoGS [31]: The GS-based approaches also targeted at static environments. It introduces several regularization strategies such as the isotropoic loss.

â¢ Rodyn-SLAM [11]: The implicit representation-based method suitable for dynamic environments. It directly eliminates dynamic foreground for robust estimation.

All the above methods are the differentiable rendering-based, which is helpful for a fair and unbiased comparison.

2) Environment Mapping: We compare different methods on multiple sequences of the above datasets. Fig. 7 shows a representative testing result on a sequence of TUM dataset, which involves both static background and two dynamic humans. For visual comparison, we render the same part of the scene from a novel view. ESLAM and MonoGS do not distinguish between foreground and background. They inappropriately map the dynamic humans based on the strategies only suitable for static background, leading to the fuzzy scene representation. Rodyn-SLAM filters out humans based on the estimated masks before mapping. While this method provides a satisfactory background reconstruction, it fails to model dynamic humans, and thus generates an incomplete map. Moreover, the loss of foreground information affects the accuracy of the map-based camera localization and path planning. Our PG-SLAM can reconstruct not only the static background but also dynamic foreground. We merge the reconstructed background and foreground at different times, and render the merged elements at a certain time. Our method achieves a complete and photo-realistic scene representation. All the elements exhibit fine textures and details, as well as reasonable spatial relationship.

We further evaluate our PG-SLAM on Bonn and Neu-Man datasets. As shown in Fig. 8, each sequence contains a single dynamic human. Similar to the above operations, on each sequence, we merge the reconstructed background and foreground at different times, and render the merged elements from a novel view at a certain time. We also present the estimated trajectories of humans and cameras. Our PG-SLAM can maintain the appearance consistency of both static background and dynamic foreground over time.

<!-- image-->  
Fig. 10. Representative camera trajectories estimated by various SLAM methods on the Sequence Walking_xyz of TUM dataset [41], Sequence tracking_person2 of Bonn dataset [43], and Sequence Citron of NeuMan dataset [42]. The cyan and black lines denote the estimated and ground truth trajectories, respectively. Pentagram represents the starting point of camera trajectory.

In addition, based on the ground truth human trajectories provided by NeuMan dataset, we evaluate the human localization of our PG-SLAM. We provide some qualitative results in Fig. 9 and also the quantitative results on all the sequences in the second part of Table IV. Our estimated human trajectory is well aligned to the ground truth trajectory and the absolute trajectory error is small, demonstrating high accuracy of our human localization.

3) Camera Localization: We compare different methods on multiple sequences of Bonn, TUM, and NeuMan datasets. We report quantitative results in Table I, and also present some qualitative results on partial sequences in Fig. 10. Overall, our PG-SLAM achieves state-of-the-art localization performance, demonstrating the effectiveness of our strategy that simultaneously leverages geometric and appearance constraints of both foreground and background. In the following, we analyze the results on each dataset.

On Bonn dataset, the high proportion of non-rigid humans and rigid items (i.e., boxes and balloons) in the RGB-D images brings considerable challenges for camera localization. ES-LAM and MonoGS are significantly affected, resulting in high trajectory errors. Rodyn-SLAM overcomes the challenges to some extent, demonstrating the advantages of disentangling dynamic foreground from static background. The accuracy of our PG-SLAM significantly exceeds that of Rodyn-SLAM on all the sequences.

<!-- image-->  
(a)

<!-- image-->

<!-- image-->  
(c)  
Fig. 11. Ablation study of camera localization using geometric constraint and dynamic foreground on Sequence person_track of Bonn dataset [43]. The colored and black lines denote the estimated and ground truth camera trajectories, respectively. Color bar indicates the magnitude of the absolute trajectory error. Pentagram represents the starting point of camera trajectory. (a) A version without using geometric constraint. (b) A version without considering foreground. (c) Our complete method that simultaneously leverages the geometric and appearance constraints of both foreground and background.

TABLE II  
ABLATION STUDY OF THE GEOMETRIC CONSTRAINT-BASED LOCALIZATION ON THREE DATASETS. WE REPORT THE ABSOLUTE TRAJECTORY ERROR OF CAMERA TRAJECTORY.
<table><tr><td>Dataset</td><td>Without Constraint</td><td></td><td>With Constraint</td></tr><tr><td></td><td>RMSE</td><td>SD</td><td>RMSE SD</td></tr><tr><td>Bonn [43]</td><td>8.0</td><td>3.1</td><td>4.8 1.6</td></tr><tr><td>NeuMan [42]</td><td>1.3</td><td>0.4 0.78</td><td>0.4</td></tr><tr><td>TUM [41]</td><td>8.6</td><td>3.5</td><td>6.8 2.9</td></tr></table>

On TUM dataset, some sequences such as st_st and st_xyz contain humans who sit on a chair and only exhibit slight motion variations. These cases approximate to the static environments, which allows ESLAM and MonoGS to achieve relatively good performance. However, both methods become unreliable on highly dynamic sequences such as wk_xyz and wk_hf. By contrast, Rodyn-SLAM can handle both cases more reliably by eliminating dynamic foreground. Our PG-SLAM further improves the accuracy on most of the sequences.

On NeuMan dataset, dynamic humans account for a relatively small ratio of the RGB-D images. Similar to the results on the above datasets, ESLAM and MonoGS perform unsatisfactorily, and Rodyn-SLAM partly improves the robustness. Our PG-SLAM achieves the best performance on all the sequences. In particular, the superiority of our PG-SLAM is significant on Sequence citron since human on this sequence exhibits a relatively large pose variation.

## C. Ablation Study

In this section, we conduct ablation studies of our proposed strategies and modules.

TABLE III  
ABLATION STUDY OF LOCALIZATION USING DYNAMIC FOREGROUND ON THREE DATASETS. WE REPORT THE ABSOLUTE TRAJECTORY ERROR OF CAMERA TRAJECTORY.
<table><tr><td>Dataset</td><td>Without Foreground</td><td></td><td>With Foreground</td></tr><tr><td></td><td>RMSE</td><td>SD</td><td>RMSE SD</td></tr><tr><td>Bonn [43]</td><td>5.4</td><td>2.2</td><td>4.8 1.6</td></tr><tr><td>NeuMan [42]</td><td>0.96</td><td>0.4 0.78</td><td>0.4</td></tr><tr><td>TUM [41]</td><td>7.6</td><td>3.0</td><td>6.8 2.9</td></tr></table>

1) Geometric Constraint-based Localization: Recall that for camera localization, we additionally leverage the geometric constraint of optical flows to complement the appearance constraint (see Section VI-B). To validate its effectiveness, we compare our complete method to the version without geometric constraint. We report the quantitative results in Table II and provide a qualitative comparison in Figs. 11(a) and 11(c). On all the datasets, our geometric constraint can significantly improve the accuracy of camera localization, especially on Bonn dataset where the proportion of dynamic objects is relatively high.

2) Localization Using Dynamic Foreground: Recall that our method leverages the dynamic foreground for camera localization (see Section VI-A). We compare our complete method using both foreground and background with the version that only considers background. As shown in Table III, by utilizing foreground information, the accuracy of camera trajectory can be effectively improved on all the datasets. In particular, Bonn dataset contains sufficient observations of dynamic foreground including both non-rigid humans and rigid items. Figs. 11(b) and 11(c) show a qualitative evaluation, demonstrating the usefulness of foreground information for camera localization.

3) Human Scale Regularization: Recall that we apply a human scale regularization loss to human reconstruction in Section IV-A. This loss is designed to address the issue of low-quality depth images, which is particularly serious on NeuMan dataset. For validation, we compare our complete method to the version without this loss on NeuMan dataset. As shown in Table IV, this loss can effectively reduce the error of human trajectory on all the sequences. A qualitative result in Fig. 12 illustrates that this regularization can avoid inappropriate human sizes and floating Gaussians, and also mitigate the discontinuity of the estimated human trajectory.

<!-- image-->

<!-- image-->  
(b)  
Fig. 12. Ablation study of human scale regularization on Sequence bike of NeuMan dataset [42]. (a) Without regularization (b) With regularization. The notations are the same as those in Fig. 9.

TABLE IV  
ABLATION STUDY OF HUMAN SCALE REGULARIZATION ON NEUMAN DATASET [42]. WE REPORT THE ABSOLUTE TRAJECTORY ERROR OF HUMAN TRAJECTORY.
<table><tr><td>Sequence</td><td colspan="2">Without Scale</td><td colspan="2">With Scale</td></tr><tr><td></td><td>RMSE</td><td>SD</td><td>RMSE</td><td>SD</td></tr><tr><td>bike</td><td>21.1</td><td>9.9</td><td>6.0</td><td>4.5</td></tr><tr><td>citron</td><td>19.0</td><td>11.2</td><td>14.2</td><td>8.7</td></tr><tr><td>jogging</td><td>76.0</td><td>56.9</td><td>8.2</td><td>4.9</td></tr><tr><td>parkinglot</td><td>18.6</td><td>7.3</td><td>15.2</td><td>6.7</td></tr><tr><td>seattle</td><td>23.5</td><td>13.3</td><td>9.4</td><td>5.5</td></tr><tr><td>Average</td><td>36.3</td><td>19.7</td><td>10.6</td><td>6.0</td></tr></table>

4) Optimization Between Neighboring Local Maps: Recall that we introduce a strategy to optimize local maps in Section V-B. We evaluate the effect of optimization on all the datasets, as shown in Table V. On sequences of TUM dataset, the camera typically moves within the same scene throughout. Accordingly, the overlapping regions between neighboring local maps are relatively large. In this case, the accuracy of both 3D map and camera trajectory is significantly improved based on our optimization strategy. A representative result is shown in Fig. 13. By contrast, on Bonn and NeuMan datasets, the camera gradually explores new areas, leading to relatively small overlapping regions between adjacent local maps. Consequently, the effectiveness of the proposed optimization algorithm decreases.

<!-- image-->  
(a) Gaussian-SLAM [33]

<!-- image-->

<!-- image-->

<!-- image-->  
(b) PG-SLAM (our)  
Fig. 13. Ablation study of optimization between neighboring local maps on sequence sitting_halfsphere of TUM dataset [41]. (a) Before optimization. (b) After optimization. Left: We report representative local maps shown in magenta and cyan (for visualization, we show the centers of Gaussians). Middle: The colored and black lines denote the estimated and ground truth trajectories, respectively. Color bar indicates the magnitude of the absolute trajectory error. Pentagram represents the starting point of camera trajectory. Right: We present the rendered background from a novel view.

TABLE V  
ABLATION STUDY OF OPTIMIZATION BETWEEN NEIGHBORING LOCAL MAPS ON THREE DATASETS. WE REPORT THE ABSOLUTE TRAJECTORY ERROR OF CAMERA TRAJECTORY.
<table><tr><td>Dataset</td><td>Without Correction</td><td></td><td>With Correction</td></tr><tr><td></td><td>RMSE</td><td>SD</td><td>RMSE SD</td></tr><tr><td>TUM [41]</td><td>6.2</td><td>2.3</td><td>4.5 1.8</td></tr><tr><td>Bonn [43]</td><td>6.6</td><td>2.4 6.5</td><td>2.2</td></tr><tr><td>NeuMan [42]</td><td>1.5</td><td>0.7</td><td>1.4 0.6</td></tr></table>

## VIII. CONCLUSIONS

In this paper, we propose a photo-realistic and geometryaware RGB-D SLAM method in dynamic environments. To the best of our knowledge, our method is the first Gaussian splatting-based approach that can not only localize the camera and reconstruct the static background, but also map the dynamic humans and items. For foreground mapping, we estimate the deformations and/or motions of dynamic objects by considering the shape priors of humans and exploiting both geometric and appearance constraints with respect to Gaussians. To map the background, we design an effective optimization strategy between neighboring local maps. To localize the camera, our method simultaneously uses the geometric and appearance constraints by associating 3D Gaussians with 2D optical flows and pixel patches. We leverage information of both static background and dynamic foreground to compensate for noise, effectively improving the localization accuracy. Experiments on various real-world datasets demonstrate that our method outperforms state-of-the-art approaches.

[1] C. Cadena, L. Carlone, H. Carrillo, Y. Latif, D. Scaramuzza, J. Neira, I. Reid, and J. J. Leonard, âPast, Present, and Future of Simultaneous Localization and Mapping: Toward the Robust-Perception Age,â IEEE Transactions on Robotics, vol. 32, no. 6, pp. 1309â1332, Dec. 2016.

[2] A. J. Davison, I. D. Reid, N. D. Molton, and O. Stasse, âMonoSLAM: Real-Time Single Camera SLAM,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 29, no. 6, pp. 1052â1067, Jul. 2007.

[3] R. Mur-Artal and J. D. TardÃ³s, âORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras,â IEEE Transactions on Robotics, vol. 33, no. 5, pp. 1255â1262, Oct. 2017.

[4] J. Engel, V. Koltun, and D. Cremers, âDirect Sparse Odometry,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 3, pp. 611â625, Mar. 2017.

[5] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, âiMAP: Implicit Mapping and Positioning in Real-Time,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 6229â6238.

[6] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNICE-SLAM: Neural Implicit Scalable Encoding for SLAM,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2022, pp. 12 786â12 796.

[7] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 595â19 604.

[8] R. I. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed. Cambridge University Press, 2004.

[9] Y. Sun, M. Liu, and M. Q.-H. Meng, âMotion removal for reliable RGB-D SLAM in dynamic environments,â Robotics and Autonomous Systems, vol. 108, pp. 115â128, Oct. 2018.

[10] C. Yu, Z. Liu, X.-J. Liu, F. Xie, Y. Yang, Q. Wei, and Q. Fei, âDS-SLAM: A Semantic Visual SLAM towards Dynamic Environments,â in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems, 2018, pp. 1168â1174.

[11] H. Jiang, Y. Xu, K. Li, J. Feng, and L. Zhang, âRoDyn-SLAM: Robust Dynamic Dense RGB-D SLAM with Neural Radiance Fields,â IEEE Robotics and Automation Letters, vol. 9, no. 9, pp. 7509â7516, Sept. 2024.

[12] J. Zhang, M. Henein, R. Mahony, and V. Ila, âVDO-SLAM: A visual dynamic object-aware SLAM system,â arXiv preprint arXiv:2005.11052, 2020.

[13] B. Bescos, C. Campos, J. D. TardÃ³s, and J. Neira, âDynaSLAM II: Tightly-Coupled Multi-Object Tracking and SLAM,â IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 5191â5198, Jul. 2021.

[14] Y. Qiu, C. Wang, W. Wang, M. Henein, and S. Scherer, âAirDOS: Dynamic SLAM benefits from Articulated Objects,â in Proceedings of the IEEE International Conference on Robotics and Automation, 2022, pp. 8047â8053.

[15] D. F. Henning, T. Laidlow, and S. Leutenegger, âBodySLAM: Joint Camera Localisation, Mapping, and Human Motion Tracking,â in Proceedings of the European Conference on Computer Vision, 2022, pp. 656â673.

[16] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, â3D Gaussian Splatting for Real-Time Radiance Field Rendering,â ACM Transactions on Graphics, vol. 42, no. 4, Jul. 2023.

[17] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, âSMPL: A Skinned Multi-Person Linear Model,â ACM Transactions on Graphics, vol. 34, no. 6, pp. 248:1â248:16, Oct. 2015.

[18] T. Whelan, R. F. Salas-Moreno, B. Glocker, A. J. Davison, and S. Leutenegger, âElasticFusion: Real-time dense SLAM and light source estimation,â The International Journal of Robotics Research, vol. 35, no. 14, pp. 1697â1716, Sept. 2016.

[19] F. Endres, J. Hess, J. Sturm, D. Cremers, and W. Burgard, â3-D Mapping With an RGB-D Camera,â IEEE Transactions on Robotics, vol. 30, no. 1, pp. 177â187, Feb. 2014.

[20] M. Bloesch, T. Laidlow, R. Clark, S. Leutenegger, and A. Davison, âLearning Meshes for Dense Visual SLAM,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 5854â5863.

[21] M. Muglikar, Z. Zhang, and D. Scaramuzza, âVoxel Map for Visual SLAM,â in Proceedings of the IEEE International Conference on Robotics and Automation, 2020, pp. 4181â4187.

[22] T. Zhang, H. Zhang, Y. Li, Y. Nakamura, and L. Zhang, âFlowfusion: Dynamic dense rgb-d slam based on optical flow,â in Proceedings of the IEEE International Conference on Robotics and Automation, 2020, pp. 7322â7328.

[23] W. Dai, Y. Zhang, P. Li, Z. Fang, and S. Scherer, âRGB-D SLAM in Dynamic Environments Using Point Correlations,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 1, pp. 373â 389, Jan. 2022.

[24] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, âDeepSDF: Learning Continuous Signed Distance Functions for Shape Representation,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 165â174.

[25] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,â in Proceedings of the European Conference on Computer Vision, 2020, pp. 405â421.

[26] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVox-Fusion: Dense Tracking and Mapping with Voxel-based Neural Implicit Representation,â in Proceedings of the IEEE International Symposium on Mixed and Augmented Reality, 2022, pp. 499â507.

[27] M. M. Johari, C. Carta, and F. Fleuret, âESLAM: Efficient Dense SLAM System Based on Hybrid Representation of Signed Distance Fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 408â17 419.

[28] H. Wang, J. Wang, and L. Agapito, âCo-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 293â13 302.

[29] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, âGO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3727â3737.

[30] E. SandstrÃ¶m, Y. Li, L. Van Gool, and M. R. Oswald, âPoint-SLAM: Dense Neural Point Cloud-based SLAM,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 433â18 444.

[31] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, âGaussian Splatting SLAM,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[32] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[33] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, âGaussian-SLAM: Photorealistic Dense SLAM with Gaussian Splatting,â 2024.

[34] Y. Xu, H. Jiang, Z. Xiao, J. Feng, and L. Zhang, âDG-SLAM: Robust Dynamic Gaussian Splatting SLAM with Hybrid Pose Optimization,â arXiv preprint arXiv:411.08373, 2024.

[35] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, Apr. 2004.

[36] K. He, G. Gkioxari, P. DollÃ¡r, and R. Girshick, âMask R-CNN,â in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 2980â2988.

[37] Y. Wang and K. Daniilidis, âReFit: Recurrent Fitting Network for 3D Human Recovery,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 14 598â14 608.

[38] Z. Teed and J. Deng, âRAFT: Recurrent All-Pairs Field Transforms for Optical Flow ,â in Proceedings of the European Conference on Computer Vision, 2021, pp. 4839â4843.

[39] K. S. Arun, T. S. Huang, and S. D. Blostein, âLeast-Squares Fitting of Two 3-D Point Sets,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-9, no. 5, pp. 698â700, Sept. 1987.

[40] L. Zhu, Y. Li, E. SandstrÃ¶m, S. Huang, K. Schindler, and I. Armeni, âLoopSplat: Loop Closure by Registering 3D Gaussian Splats,â arXiv preprint arXiv:2408.10154, 2024.

[41] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of RGB-D SLAM systems,â in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 573â580.

[42] W. Jiang, K. M. Yi, G. Samei, O. Tuzel, and A. Ranjan, âNeuMan: Neural Human Radiance Field from a Single Video,â in Proceedings of the European Conference on Computer Vision, 2022, pp. 402â418.

[43] E. Palazzolo, J. Behley, P. Lottes, P. GiguÃ¨re, and C. Stachniss, âReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals,â in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems, 2019, pp. 7855â7862.

[44] Z. Zhang and D. Scaramuzza, âA Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry,â in Proceedings of the

IEEE/RSJ International Conference on Intelligent Robots and Systems, 2018, pp. 7244â7251.