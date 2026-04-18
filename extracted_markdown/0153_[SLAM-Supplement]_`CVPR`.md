# SmallGS: Gaussian Splatting-based Camera Pose Estimation for Small-Baseline Videos

Yuxin Yao1, Yan Zhang2, Zhening Huang1, Joan Lasenby1 1University of Cambridge, 2Meshcapade {yy561,zh340,jl221}@cam.ac.uk; yan@meshcapade.com

<!-- image-->  
Figure 1. Camera pose estimation for small-baseline videos with SmallGS. Our method focuses on camera pose estimation in smallbaseline videos, updating the camera poses based on the rasterization of Gaussian splatting in the learned camera viewpoints. We achieved better and smoother results compared to the previous SOTA MonST3R.

## Abstract

Dynamic videos with small baseline motions are ubiquitous in daily life, especially on social media. However, these videos present a challenge to existing pose estimation frameworks due to ambiguous features, drift accumulation, and insufficient triangulation constraints. Gaussian splatting, which maintains an explicit representation for scenes, provides a reliable novel view rasterization when the viewpoint change is small. Inspired by this, we propose SmallGS, a camera pose estimation framework that is specifically designed for small-baseline videos. SmallGS optimizes sequential camera poses using Gaussian splatting, which reconstructs the scene from the first frame in each video segment to provide a stable reference for the rest. The temporal consistency of Gaussian splatting within limited viewpoint differences reduced the requirement of sufficient depth variations in traditional camera pose estimation. We further incorporate pretrained robust visual features, e.g. DINOv2, into Gaussian splatting, where high-

dimensional feature map rendering enhances the robustness of camera pose estimation. By freezing the Gaussian splatting and optimizing camera viewpoints based on rasterized features, SmallGS effectively learns camera poses without requiring explicit feature correspondences or strong parallax motion. We verify the effectiveness of SmallGS in small-baseline videos in TUM-Dynamics sequences, which achieves impressive accuracy in camera pose estimation compared to MonST3R and DORID-SLAM for small-baseline videos in dynamic scenes. Our project page is at: https://yuxinyao620.github.io/SmallGS

## 1. Introduction

With the advancement of deep learning and computer vision, camera pose estimation has improved significantly over the past decade. Structure-from-Motion (SfM) and SLAM-based methods represent mature approaches for estimating camera poses in static scenes with large camera baselines [9]. However, videos captured in everyday scenarios are typically recorded using stationary setups, exhibiting small baselines and limited viewpoint rotation. These characteristics result in substantial camera drift accumulation as trajectories extend. Although bundle adjustment requires sufficient geometric constraints to mitigate drift, videos with small baselines inherently lack diverse camera motion, which limits optimization effectiveness.

Furthermore, recent methods that aim to reconstruct global 3D point clouds thereby capture the entire sceneâs geometry for better camera pose estimation. Deep visual SLAM methods [27, 31] leverage visual features for robust camera tracking and scene reconstruction. However, most struggle with videos featuring small-baselines and unconstrained camera trajectories [20, 21].

Gaussian splatting as an explicit scene representation provides a feasible way to update camera poses while reconstructing the 3D scene. Compared to NeRF-based methods, Gaussian splatting offers a more efficient way to optimize camera pose, since affine transformations can be directly applied to 3D Gaussians. Camera pose estimation via 3D Gaussian splatting is less susceptible to small-baseline limitations, as it optimizes relative poses between adjacent frames. However, most methods [5, 6, 30] focus exclusively on static scenes, and their iterative pairwise camera pose estimation remains computationally expensive.

In this work, we focus on camera pose estimation in small-baseline videos, where the camera motion exhibits limited translations. Such videos are ubiquitous on social medias like TikTok and Instagram. Camera poses and intrinsic parameters can be estimated by leveraging advanced global visual features and bundle adjustment [27, 31]. While these approaches can produce realiable global scene geometry and camera intrinsics, the estimated poses exhibit low precision with significant jitter.

We introduce SmallGS for accurate camera tracking in in-the-wild dynamic videos with small camera movements, and leverage DINOv2 [18] features in the 3D Gaussians in order to improve robustness. Following CF-3DGS [6], we lift depth maps predicted by MonST3R to 3D dense point clouds to initialize Gaussian splatting. To address moving objects in the scene, we mask dynamic regions using MonST3Râs predicted confidence masks, which indicate the likelihood of an object being in motion.

Although MonST3R can also estimate camera poses, the results in our small-baseline scenarios are not sufficiently accurate and have artifacts like jittering. MonST3R estimates camera poses primarily through pointmap alignment, where pointmaps are generated by a neural network. However, in small-baseline videos, key constraintsâsuch as sufficient depth variationâare often insufficient, making it difficult for the neural network to produce consistent and robust pointmaps. This will lead to e.g. inaccurate estimates of the pointmaps that are far away from the camera, and hence will negatively influence the camera poses that are obtained via pointmap alignment, illustrated in Fig. 1

In contrast, Gaussian splatting leverages neural rendering and photometric losses to estimate the camera SE(3) transformations [6]. The splatting process will focus on the 3D Gaussians that are closer to the camera and have higher opacities. In this case, inaccuracies of scene geometries that are far away from the camera will not have significant impact. Small baselines that are with respect to the far objects are actually not small with respect to objects in the near field. In addition, since adjacent frames exhibit high similarity, frames can be represented using the Gaussian splatting reconstructed from the neighboring frame. To further exploit this consistency, we employ a sliding window to partition the image sequences into segments, where the first frame of each segment is the same as the last frame of the previous segment. The Gaussian splatting for the first frame is optimized to match the RGB image, reducing the reliance on depth predictions within the batch and improving pose stability(Fig. 1). We then jointly optimize a batch of adjacent camera poses using a single Gaussian splatting scene, significantly improving computational efficiency. Furthermore, we incorporate a camera trajectory smoothness constraint to encourage smooth camera motion and remove jittering artifacts.

We evaluate our method on the TUM-dynamics dataset [23], which consists of handheld camera recordings in dynamic environments. SmallGS estimates camera poses using MonST3Râs predicted depth maps, semantic masks, camera intrinsics, and DINOv2 features. Our approach significantly improves Absolute Trajectory Error (AT E) and Relative Pose Error (RP E). Additionally, we analyze the velocity difference between the estimated and ground-truth camera positions, providing insights into the smoothness of the estimated camera trajectory. The velocity difference also serves as a measure of the motion consistency between the estimated and ground-truth camera trajectories.

In summary, the main contributions of our work are as follows.

â¢ Establish a novel camera pose estimation framework, SmallGS, for small-baseline videos, leveraging Gaussian splatting to optimize camera poses while mitigating dynamic object interference via predicted semantic masks. SmallGS does not rely on 3D alignments or triangulation, alleviating the instability in camera pose estimation caused by limited parallax and weak geometric constraints.

â¢ Utilize visual features from DINOv2 in camera pose estimation, investigating the influence of geometry understanding and higher-dimensional features. This improves the accuracy of camera pose estimation.

## 2. Related Works

## 2.1. 3D Gaussian splatting

3D Gaussian splatting is a novel scene representation method, which is explicit, fast, and capable of high-quality rendering. A set of 3D Gaussians with different covarainces and means are used to represent static scenes [5â 7, 11, 35]. By incorporating time-varying parameters in each Gaussian, it can also effectively represent dynamic scenes [12, 13, 28]. Sparse points and camera poses for each image are required for scene reconstruction, which is commonly obtained using COLMAP [20, 21]. However, COLMAP relies on matching the extracted feature descriptor and triangulation to reconstruct 3D scenes [5, 11], which can lead to inaccuracies when estimating depth and camera pose in videos with small baseline disparity.

To resolve preprocessing requirements, CF-3DGS [6] utilized pre-trained depth estimation models like DPT [19] to lift image pixels into a dense point cloud. CF-3DGS progressively grows the 3D Gaussian set by processing one neighbouring frame at a time and estimating camera poses through optimization without relying on COLMAP. Similarly, InstantSplat [5] obtains relative camera poses by aligning dense pointmaps predicted using DUSt3r [5].

CF-3DGS and InstantSplat are primarily applied to static scenes. Dynamic 3DGS enables dynamic reconstruction by applying the dense tracking of Gaussians [12]. Splat-Fields employ DOMA [33] to learn the motion fields of the Gaussians, introducing spatial bias which stabilizes the optimization process. However, most dynamic scene methods require pre-established camera motion models to estimate camera poses incrementally [12].

## 2.2. Traditional Camera Pose Estimation Methods

Estimating camera pose is one of the most critical components in the pipelines of 3D reconstruction, robotics, and AR/VR systems. The estimation of camera extrinsic and intrinsic parameters involves techniques like SfM and SLAM, primarily used for cases of unstructured video sequences. Traditional SLAM methods rely on aligning image correspondences or utilizing feature descriptors to establish alignment between different viewpoints, followed by bundle adjustment to optimize 3D point cloud positions and camera parameters [2, 4, 14, 17]. Recent advances in deep learning have enabled SLAM and SfM approaches to learn higher-dimensional features for matching and scene reconstruction, which enhances robustness to blur and artifacts while reducing the need for extensive preprocessing [3, 24, 26]. DROID-SLAM [26] introduces a recurrent bundle adjustment layer using GRU, effectively integrating camera pose estimation with pixel-wise depth recovery. However, these methods mainly focus on static scenes and require large baselines to establish sufficient triangulation constraints. Consequently, their performance degrades in dynamic environments or scenarios with small baseline video sequences, which are prevalent in everyday applications.

To address these problems, SLAM and SfM methods incoorperated pretrained depth and semantic segmentation neural networks. The depth map, semantic segmentation, and camera poses are optimized jointly, improving the overall geometry of the scene and exploiting the epipolar constriant with respect to the static parts of the scene [8, 25, 32, 34]. Casual-SLAM [34] jointly optimized the depth and camera pose estimation. It fine-tuned a pretrained depth network to represent dense 3D correspondance. Robust-CVD [8] also jointly estimate camera parameters and depth maps from dynamic videos, while the optimization is based on the alignment of spatially-varying splines.

## 2.3. Learning-based Camera Pose Estimation

Predicting the camera parameters directly from an endto-end deep learning architecture is increasingly powerful. Learning-based visual odometry methods allow largescale training under different scenes, static and dynamic [1, 15, 16, 22, 27, 31]. For example, DytanVO [22] developed a unified neural network that estimates the motion of the camera and segments dynamic objects, which is robust in dynamic videos. Similarly, LEAP-VO [1] leverages the attention mechanism to learn long-term point tracking, inferring moving object masks for accurate camera pose estimation. However, the camera pose estimation of dynamic video with small baselines remains a problem. Drift accumulation and loop closure issues occur in learning-based camera parameter predictions.

DUSt3r [27] directly regresses the geometry of the scene by learning the pointmap representation, which estimates the 3D location of each associated pixel in a pair of images in a common coordinate shared by the pair of frames. Downstream applications like camera pose estimation and 3D reconstruction can be achieved using the pointmaps. MonST3R [31], fine-tuned from DUSt3r, achieves SOTA results of camera parameter estimation and 4D reconstruction in dynamic videos. The drift accumulation flaw is less serious in MonST3R.

The concurrent work, GS-CPR [10], utilized the fast rendering property of Gaussian splatting to refine camera poses. However, they pre-train a Gaussian splatting, followed by finetuning the prior camera pose based on the 2D-2D matching between query image and rendered image. Another concurrent work, ZeroGS [30], focuses on static scene reconstruction with unposed and unordered images. Both of them ignored the small-baseline video cases.

Our method further improves the camera pose estimation performances in small baseline videos by utilizing Gaussian splatting and DINOv2 visual features. We apply a method similar to CF-3DGS given the higher-dimensional visual features, leading to a more accurate and smooth camera pose in small baseline videos.

## 3. Method

Given a sequence of unposed images in a dynamic video with a small camera baseline, we aim to estimate the camera poses. MonST3R provides robust pairwise pointmap predictions, along with downstream intrinsic and relative camera pose estimation, depth estimation, etc. Building upon existing methods, we propose SmallGS to learn the relative camera pose. In this section, we detail our approach. Sect.3.1 reviews the rendering process and camera optimization process of CF-3DGS [6], while Sect. 3.2 introduces SmallGS, which optimizes camera poses in batches to improve efficiency. Additionally, we incorporate a pretrained visual feature extractor, DINOv2 [18], into the SmallGS pipeline. The entire pipeline is shown in Fig. 2.

## 3.1. Preliminary Colmap-Free 3D Gaussian splatting

3D Gaussian splatting [7] represents scenes using 3D Gaussians, each characterized by several parameters: mean point $( \mu \in \mathbb { R } ^ { 3 } )$ , rotation factor $( r \in \mathbb { R } ^ { 4 } )$ , scale factor $( s \in \mathbb { R } ^ { 3 } )$ ; opacity $( \alpha \in \mathbb { R } )$ , and spherical harmonics (SH) coefficients $( c \in \mathbb { R } ^ { k } )$ representing color or features with k degrees of freedom. The covariance matrix Î£ of a 3D Gaussian describes the corresponding 3D ellipsoid, which is determined by the scale factor and the rotation factor represented as a quaternion.

$$
\Sigma = R S S ^ { T } R ^ { T }\tag{1}
$$

where R is the rotation matrix converted from the quaternion r, and S is the diagonal matrix with the scale factor s, with ${ \cal S } = d i a g ( [ s _ { x } , s _ { y } , s _ { z } ] )$

In CF-3DGS, the parameters and camera poses are updated by applying supervision on the rendered images. The rendering process of Gaussian splatting is differentiable [7] with respect to the camera pose $W \in \mathbf { S E } ( \mathbf { 3 } )$ . The covariance of the Gaussians is projected onto the camera coordinates, resulting in 2D images at the camera viewpoint W [7].

$$
\Sigma _ { 2 D } = J W \Sigma W ^ { T } J ^ { T }\tag{2}
$$

J is the Jacobian of the affine approximation of the projective transformation. The color and opacity of each pixel in the rendered image are calculated by the alpha-blending of N ordered points that overlap the pixel [7]:

$$
C = \sum _ { i } ^ { N } c _ { i } \alpha _ { i } \prod _ { j } ^ { i - 1 } ( 1 - \alpha _ { j } )\tag{3}
$$

where C is the color for the pixel in rendered image. Similarly, rendering the higher-dimensional feature map F instead of an image, Eq.3 also applies [29, 35].

The camera pose estimation of the images is achieved by optimizing the local relative camera poses of the adjacent frames. 3DGS provides an explicit representation of the entire scene, and the projection of each Gaussian in the scene relies on the viewing transform W , which is the camera pose [6]. Thus, the 2D projection of the same Gaussian under a different camera pose $W ^ { \prime }$ is given by:

$$
\mu _ { 2 D } = K \frac { W \mu } { ( W \mu ) _ { z } } , \quad \mu _ { 2 D } ^ { \prime } = K \frac { W ^ { \prime } \mu } { ( W ^ { \prime } \mu ) _ { z } }\tag{4}
$$

where $( W \mu )$ z denotes the depth (z-axis) of the Gaussian. $\mu _ { 2 D } ^ { \prime }$ denotes the pixel position of the projected Gaussians from a different viewpoint. K denotes the camera intrinsic matrix. Hence, the camera pose can be optimized similarly to optimizing the center points of Gaussians [6].

In CF-3DGS, the sparse points used for initialization are replaced by lifting the predicted depth using an off-theshelf monocular depth estimation network, DPT [19]. With dense depth information, the geometry of Gaussian splatting closely approximates the target scene before optimization. The 3D Gaussians $G _ { t }$ are further updated by minimizing the photometric loss between the rendered image and the current frame t. After Gaussian splatting is initialized with the first frame of the video, the relative camera transformation between frame t and frame $t + 1$ is optimized instead of the parameters of $G _ { t }$

$$
T _ { t } ^ { * } = \arg \operatorname* { m i n } _ { T _ { t } } L _ { r g b } \big ( \mathcal { R } ( T _ { t } \odot G _ { t } ) , I _ { t + 1 } \big )\tag{5}
$$

where $T _ { t }$ is the relative camera transformation, including the rotation matrix and the translation vector, and $G _ { t + 1 } =$ $T _ { t } \odot G _ { t }$ . The â denotes the element-wise transformation. Since the two adjacent frames are close, the Gaussian splatting maintains robust visual features for camera pose optimization.

## 3.2. Batch camera pose optimization

To increase the efficiency of camera pose estimation in CF-3DGS for small-baseline videos, we introduce SmallGS, which optimizes batched camera poses. Small-baseline videos typically exhibit minimal camera movement over a few seconds, resulting in frames that remain highly similar to subsequent frames. A sliding window with size of b is used to divide the video into video segments, where the first frame of each segment is the same as the last frame of the previous segment. Gaussian splatting is reinitialized for each video segment, and the relative camera poses are estimated.

$$
T _ { t , \dots , t + b } ^ { * } = \arg \operatorname* { m i n } _ { T _ { t } , \dots , T _ { t + b } } \sum _ { i = t } ^ { t + b } L _ { r g b } ( \mathcal { R } ( T _ { i } \odot G _ { t } ) , I _ { i } )\tag{6}
$$

where the camera intrinsics in the rendering function are given by MonST3R. The loss function $L _ { r g b }$ is the mean squared error (MSE) loss between the rendered image and the ground-truth video frame. Upon estimating the relative camera poses for the entire sliding window, the relative camera pose for the full trajectory is obtained. However, in the presence of dynamic objects in the video, frame-toframe consistency is broken. To address this, we introduce semantic masks to remove dynamic objects. MonST3R provides a confidence map for static objects in each image, and can serve as a semantic mask M that masks out the dynamic objects. Additionally, we utilize the depth map predicted by MonST3R to initialize the point cloud for Gaussian splatting. Thus, each Gaussian splatting instance is initialized with both a depth map and a semantic mask. The image pixels, $p _ { 2 D }$ , of the first frame in the video segment are lifted to 3D camera coordinates using the corresponding depth map D. Then, the semantic mask M is applied to filter out dynamic points.

<!-- image-->  
Figure 2. Pipeline of SmallGS. Our method follows the CF-3DGS pipeline, estimating camera poses in video segments. The process is: (1) Use MonST3R to predict depth maps, confidence masks, and camera intrinsics; (2) Lift the first frameâs depth map into a dense point cloud, masking dynamic objects using the confidence mask as a semantic mask; (3) Initialize and update Gaussian splatting with the first frame; (4) Freeze the Gaussian parameters and optimize batched camera poses by minimizing the error between the rasterized feature maps (under the estimated poses) and the DINOv2 [18] feature maps, with semantic masks applied to both.

$$
\mathbf { p } = \Pi ^ { - 1 } \left( K ^ { - 1 } \tilde { \mathbf { p } } _ { 2 D } \cdot D \right) \cdot M\tag{7}
$$

where $\Pi ^ { - 1 }$ denotes the unprojection function that maps image coordinates and depth to 3D space, and pË2D represents the homogeneous image coordinates.

However, the batched camera poses within a video segment may have different scales, and the poses at the end of the window tend to exhibit more pronounced rotation and translation variations. To address this, a camera smoothness loss is applied during the batched optimization process to penalize large changes in camera pose.

$$
L _ { \mathrm { s m o o t h } } = \lambda _ { c } \sum _ { i = 1 } ^ { N - 2 } \| ( \mathbf { x } _ { i + 1 } - \mathbf { x } _ { i } ) - ( \mathbf { x } _ { i } - \mathbf { x } _ { i - 1 } ) \|\tag{8}
$$

where $\lambda _ { c }$ varies from 0 to 1, increasing with the number of Gaussian splatting optimization iterations. x denotes the position of the camera. In addition, we use SSIM loss to

measure the quality of the rendered image, with $\lambda _ { s }$ set to 0.2 in our experiments. Overall, the optimization of batched camera poses is:

$$
T _ { t } ^ { * } . . . T _ { t + b } ^ { * } = \arg \operatorname* { m i n } _ { T _ { t } , \dots , T _ { t + b } } \sum _ { i = t } ^ { t + b } L _ { r g b } ( \mathcal { R } ( T _ { i } \odot G _ { t } ) , I _ { i } )\tag{9}
$$

Using a robust visual feature instead of RGB images for Gaussian splatting enhances the overall scene geometry and semantics understanding [35]. Visual features are less affected by lighting or perspective changes. We utilize DINOv2 [18] features, which are generally robust in daily scenes, and select the $f$ most informative feature channels. Principal Component Analysis (PCA) is applied to rank the importance of features in each channel. The rendering process follows a similar approach to rendering color but with higher-dimensional features [29]. Thus, the optimization of batched camera poses is:

$$
\begin{array} { r l } { T _ { t } ^ { * } \ldots T _ { t + b } ^ { * } = \arg \underset { T _ { t } , \ldots , T _ { t + b } } { \operatorname* { m i n } } } & { { } \underset { i = t } { \sum } L _ { m s e } ( \mathcal { R } ( T _ { i } \odot G _ { t } ) , F _ { i } ^ { f } ) } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } & { { } } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + L _ { s m o o t h } + \lambda _ { s } L _ { s s i m } \quad } & { { } ( 1 0 } \end{array}\tag{0}
$$

where $F _ { i } ^ { f }$ represents the DINOv2 feature with $f$ selected channels. The loss function $L _ { m s e }$ in the optimization process is the channel-wise mean squared error (MSE) loss.

## 4. Experiment

In our SmallGS framework, we set the first frame in each video segment as the canonical frame and estimate the rela-

<table><tr><td></td><td>MonST3R</td><td>DROID</td><td>SmallGS</td><td>SmallGS w/ 3 DINOv2</td><td>SmallGS w/ 16 DINOv2</td></tr><tr><td>ATEâ</td><td>0.00294</td><td>0.00268</td><td>0.00263</td><td>0.00254</td><td>0.00228</td></tr><tr><td>RPEr â</td><td>0.316</td><td>0.320</td><td>0.301</td><td>0.335</td><td>0.340</td></tr><tr><td>RPEt â</td><td>0.00186</td><td>0.00197</td><td>0.00183</td><td>0.00135</td><td>0.00123</td></tr><tr><td>âv â</td><td>0.0258</td><td>0.0263</td><td>0.0230</td><td>0.00980</td><td>0.00090</td></tr></table>

Table 1. Comparison of baseline methods and SmallGS based on AT E, $R P E _ { r }$ , and $R P E _ { t }$ . âv represents the average difference in camera translation velocity between the predicted and ground-truth trajectory. Bold values indicate the best performance, underlined values the second-best. DINOv2 denotes that SmallGS learns visual features instead of RGB in Gaussian Splatting, with 3 and 16 indicating the top 3 and 16 most informative feature channels, respectively.

tive camera poses for the subsequent frames.

## 4.1. Dataset

We select 46 sequences from the TUM-dynamics dataset [23]. These sequences feature very small camera baselines and include dynamic objects within the scenes. The length of each sequence is 30-frame, and the image resolution is 512 Ã 384. The number of frames for each video segment in SmallGS is 15 for all experiments. We compare our estimated camera poses with the ground truth after alignment.

## 4.2. Evaluation Metric

We evaluate the camera pose estimation by reporting the Absolute Trajectory Error (AT E) and the Relative Pose Error (RP E) in both translation and rotation.

In addition, we compare the average difference in camera translation velocity (âv) between our predicted trajectory and the ground truth. Velocity represents the rate of change in camera position. A high velocity difference between frames indicates significant camera drift and jitter in pose estimation. Conversely, similar inter-frame velocities between the ground-truth and estimated camera poses imply motion consistency.

## 4.3. Baseline and ours

Camera pose estimation is a well-established topic, with existing methods generally achieving strong performance. SmallGS is developed as an improvement over CF-3DGS [6], which estimates camera poses simultaneously with scene reconstruction as a baseline. Unlike CF-3DGS, SmallGS focuses exclusively on camera pose estimation and introduces a batched optimization approach. We compare our method against learning-based visual odometry approaches for dynamic scenes, specifically DROID-SLAM [26] and MonST3R [31].

Our implementation is based on PyTorch and GSplat [29], following the default GSplat configuration unless otherwise stated. The Gaussian splatting of the canonical frame is removed after optimizing the camera pose for each sliding window. The depth and confidence masks predicted by MonST3R are used both for dense point cloud initialization in Gaussian splatting and as semantic masks during the learning process. We mask out the dynamic obejcts using the semantic masks in the Gaussian splatting. We initialize the camera pose using the identity matrix and use the same depth and confidence masks from MonST3R across all experiments to ensure fair comparisons. While our experiments depend on MonST3R outputs for consistency, the SmallGS framework is not restricted to MonST3R specifically; alternative sources for depth, semantic masks, and intrinsics can be easily incorporated under different scenarios.

Furthermore, to extract more informative features than RGB images, we learn feature fields using Gaussian splatting [29, 35]. We employ DINOv2 [18] to extract robust visual features, utilizing its distilled backbone in all subsequent experiments. Principal Component Analysis (PCA) is applied to filter the most informative channels from patched tokens, selecting the top sixteen features for comparison with the same model trained on RGB images.

## 5. Results and Discussion

## 5.1. Experiment Results

The results are shown in Tab.1. SmallGS improved trajectory accuracy without additional features. Both the relative pose error (RP E) and absolute trajectory error (AT E) outperform MonST3R and DROID-SLAM, demonstrating that batched Gaussian splatting effectively learns camera trajectories. Additionally, the lower âv indicates that our estimated trajectory closely aligns with the ground truth motion. This is because the Gaussian splatting is consistent over small viewpoint changes, providing a robust batched camera pose optimization process which is minimally affected by noisy depth map or point cloud initialization. The segment-wise camera pose estimation also reduces drift accumulation.

Incorporating DINOv2 visual features further enhances accuracy. DINOv2 extracts general visual features that encode semantic and depth information. Using 3-channel DINOv2 feature maps significantly improves AT E and $R P E _ { t }$ , highlighting the effectiveness of leveraging DI-NOv2 features. The experiment with 16-channel DINOv2 features achieves the best overall performance. Although

<!-- image-->  
(a) Comparison with DROID

<!-- image-->  
(b) Comparison with MonST3R  
Figure 3. Fig. 3b compares the estimated camera trajectories of MonST3R and SmallGS with 16-channel DINOv2 features. Fig. 3a shows the same SmallGS trajectory compared to DROID. The red dashed line represents the ground truth.

$R P E _ { r }$ is slightly higher than the baseline methods, improvements in $A T E$ and $R P E _ { t }$ indicate that SmallGS provides high-precision camera position estimation. Furthermore, the lowest $\Delta v$ suggests that SmallGS with 16- channel DINOv2 features yields the most accurate camera motion alignment with the ground truth. Thus, we conclude that SmallGS effectively learns meaningful 3D visual features, enhancing scene representation and improving camera pose estimation.

Our method achieves lower AT E, RP E, and $\Delta v$ compared to MonST3R and DROID-SLAM overall, particularly improving the accuracy of relative translation in camera pose estimation. This suggests that SmallGS is better suited for small-baseline videos, benefiting from the robust scene geometry provided by Gaussian splatting in small-baseline video and the strong visual features extracted by DINOv2.

Fig. 3 illustrates the comparison among DROID-SLAM, MonST3R, and SmallGS trained with 16-channel DINOv2 features. Fig. 4 illustrates the comparisons of camera trajectories plotted in the 3D scenes, further illustrating that SmallGS produces smoother trajectores that have more consistent camera motion as the ground truth. The estimated camera poses from our method are smoother and exhibit less drift compared to the others.

<table><tr><td></td><td>Time â</td><td>ATEâ</td><td> $R P E _ { r } \downarrow$ </td><td> $R P E _ { t } \downarrow$ </td><td> $\Delta v \downarrow$ </td></tr><tr><td>CF-3DGS</td><td>682s</td><td>0.00230</td><td>0.259</td><td>0.00189</td><td>0.0269</td></tr><tr><td>CF-3DGS +Mask</td><td>635s</td><td>0.00258</td><td>0.286</td><td>0.00182</td><td>0.0229</td></tr><tr><td>SmallGS</td><td>271s</td><td>0.00228</td><td>0.340</td><td>0.00123</td><td>0.0009</td></tr></table>

Table 2. Average training time and accuracy comparison between CF-3DGS [6] with and without semantic masks, and SmallGS w/ 16 DINOv2.

In Tab. 2, we present the time consumption of SmallGS and CF-3DGS [6] for estimating the camera poses of

30-frame video sequences. CF-3DGS estimates pairwise camera poses by reconstructing the Gaussian splatting for the previous frame at each estimation, making it timeconsuming. While CF-3DGS considers full scene reconstruction, reconstructing the 3D scene for each frame is unnecessary for camera pose estimation. We additionally applied the semantic masks, removing the dynamic object for CF-3DGS. However, AT E, $R P E _ { t } ,$ and $\Delta v$ are larger for CF-3DGS with and without semantic masks than for SmallGS, demonstrating that SmallGS outperforms CF-3DGS in estimating camera poses for small-baseline videos in a more efficient way.

## 5.2. Ablation

In this section, we present several ablation experiments to analyze the rationale behind our framework. Firstly, we use SmallGS to fine-tune the camera poses predicted by MonST3R, revealing that initializing camera poses with the identity matrix yields better results. Secondly, we utilize the pairwise pointmaps predicted by MonST3R to initialize Gaussian splatting. MonST3Râs pointmaps provide a dense point cloud that is temporally consistent within paired images. These pointmaps encode both geometric and potential semantic information, making them more informative and robust than point clouds lifted from depth maps.

MonST3R Pointmaps Instead of DINOv2 Feature Maps MonST3R is trained to predict pairwise pointmaps that maintain temporal consistency between paired images within a sliding window. These pointmaps, which serve as dense point clouds, preserve scene geometry more effectively within the sliding window. They incorporate richer geometric and potential semantic information than depth maps. The correlation of each 3D Gaussian is ensured to match the corresponding 3D Gaussian in the paired frame, reducing the complexity of estimating transformations between two Gaussian splatting representations. Hence, MonST3R pointmaps could provide a more reliable initialization for Gaussian splatting.

To select appropriate image pairs for generating pointmaps, we choose the pairs that are the furthest apart within the video segments in SmallGS. We compare the results of initializing Gaussian splatting using MonST3R pointmaps and depth maps, as shown in Tab. 3. SmallGS using pointmaps achieves the lowest relative rotation error, as their temporal consistency provides better initialization for point clouds, improving the geometry of Gaussian splatting. However, SmallGS with DINOv2 features achieves the best overall performance with more accurate camera position estimation, demonstrating that robust general visual features are more effective for small-baseline camera pose estimation.

<!-- image-->  
Figure 4. Comparison of ground-truth trajectories, MonST3R-predicted trajectories, and the SmallGS-learned trajectory with 16-channel DINOv2 feature maps. The trajectories predicted by MonST3R often exhibit jitter around the ground-truth trajectories. SmallGS efficiently learns the trajectories of small-baseline videos, improving camera pose accuracy.

<table><tr><td></td><td>ATEâ</td><td>RPEr â</td><td>RPEt â</td><td>âvâ</td></tr><tr><td>SmallGS</td><td>0.00263</td><td>0.301</td><td>0.00183</td><td>0.0230</td></tr><tr><td>SmallGS w/ pointmaps</td><td>0.00251</td><td>0.293</td><td>0.00181</td><td>0.0228</td></tr><tr><td>SmallGS w/DINO</td><td>0.00228</td><td>0.340</td><td>0.00123</td><td>0.0009</td></tr></table>

Table 3. Comparison of SmallGS initialized with depth maps vs. MonST3R pointmaps. SmallGS uses depth maps without DI-NOv2, SmallGS w/ 16 DINO incorporates 16-channel DINOv2 features, and SmallGS w/ pointmaps is initialized with MonST3R pointmaps.

Camera Pose Refinement Tab.4 presents a comparison between using SmallGS to refine the camera poses predicted by MonST3R and learning camera poses directly without a camera pose prior. In the refinement setting, we initialize the camera poses for each frame using the predicted camera poses from MonST3R.

The relative pose error of using SmallGS for refinement is lower. However, the AT E and RP Et of SmallGS refining MonST3Râs camera poses are worse than the non-refinement setting. Additionally, the larger difference in camera position velocity (âv) indicates that the refined camera poses lack smoothness. This is because MonST3R predicts jittery and non-smooth camera poses in small-baseline videos, making them more challenging for SmallGS to fine-tune. As shown in Fig. 3, the predicted trajectory of MonST3R exhibits noticeable jitter. Thus, initializing camera poses with the identity matrix proves to be more effective for small-baseline videos.

<table><tr><td></td><td> $A T E \downarrow$ </td><td> $R P E _ { r } \downarrow$ </td><td> $R P E _ { t }$  â</td><td> $\Delta v \downarrow$ </td></tr><tr><td>Refine w/ 16 DINO</td><td>0.00229</td><td>0.321</td><td>0.00136</td><td>0.00106</td></tr><tr><td>SmallGS w/ 16 DINO</td><td>0.00228</td><td>0.340</td><td>0.00123</td><td>0.00090</td></tr></table>

Table 4. Comparison of fine-tuning MonST3R-predicted camera poses vs. SmallGS-only estimation. Refine w/ 16 DINO fine-tunes poses using SmallGS with 16-channel DINOv2, while SmallGS w/ 16 DINO runs SmallGS without prior poses.

## 6. Conclusion and Future Work

We present SmallGS, a new approach to estimating temporally consistent camera motions in small-baseline videos with dynamic objects. Our approach improves the camera pose accuracy by incorporating self-supervised robust visual features besides RGB images. The experiment results on small-baseline videos in TUM-dynamics [23] show the effectiveness of SmallGS, and its superior performances to state-of-the-art baselines.

Although using prior camera poses from MonST3R results in slightly less accurate estimation, the difference between SmallGS with and without priors remains small. Future work could further explore the impact of incorporating prior camera poses in SmallGS.

Acknowledgement. We sincerely thank Michael J. Black for fruitful discussions on the small-baseline video scenario.

## References

[1] Weirong Chen, Le Chen, Rui Wang, and Marc Pollefeys. Leap-vo: Long-term effective any point tracking for visual odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19844â 19853, 2024. 3

[2] Andrew J Davison, Ian D Reid, Nicholas D Molton, and Olivier Stasse. Monoslam: Real-time single camera slam. IEEE transactions on pattern analysis and machine intelligence, 29(6):1052â1067, 2007. 3

[3] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 224â236, 2018. 3

[4] Jakob Engel, Thomas Schops, and Daniel Cremers. Lsd- Â¨ slam: Large-scale direct monocular slam. In European conference on computer vision, pages 834â849. Springer, 2014. 3

[5] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, Zhangyang Wang, and Yue Wang. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds, 2024. 2, 3

[6] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A. Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Com-

puter Vision and Pattern Recognition (CVPR), pages 20796â 20805, 2024. 2, 3, 4, 6, 7

[7] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 3, 4

[8] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1611â1621, 2021. 3

[9] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast, and robust structure and motion from casual dynamic videos. arXiv preprint arXiv:2412.04463, 2024. 2

[10] Changkun Liu, Shuai Chen, Yash Sanjay Bhalgat, Siyan HU, Ming Cheng, Zirui Wang, Victor Adrian Prisacariu, and Tristan Braud. GS-CPR: Efficient camera pose refinement via 3d gaussian splatting. In The Thirteenth International Conference on Learning Representations, 2025. 3

[11] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20654â20664, 2024. 3

[12] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713, 2023. 3

[13] Marko Mihajlovic, Sergey Prokudin, Siyu Tang, Robert Maier, Federica Bogo, Tony Tung, and Edmond Boyer. Splatfields: Neural gaussian splats for sparse 3d and 4d reconstruction. In European Conference on Computer Vision (ECCV). Springer, 2024. 3

[14] Anastasios I Mourikis and Stergios I Roumeliotis. A multistate constraint kalman filter for vision-aided inertial navigation. In Proceedings 2007 IEEE international conference on robotics and automation, pages 3565â3572. IEEE, 2007. 3

[15] Raul Mur-Artal and Juan D Tardos. Orb-slam2: An open- Â´ source slam system for monocular, stereo, and rgb-d cameras. IEEE transactions on robotics, 33(5):1255â1262, 2017. 3

[16] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: a versatile and accurate monocular slam system. IEEE transactions on robotics, 31(5):1147â1163, 2015. 3

[17] Richard A Newcombe, Steven J Lovegrove, and Andrew J Davison. Dtam: Dense tracking and mapping in real-time. In 2011 international conference on computer vision, pages 2320â2327. IEEE, 2011. 3

[18] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy V.Â´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bo-

janowski. Dinov2: Learning robust visual features without supervision, 2023. 2, 4, 5, 6

[19] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi- Â´ sion transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179â12188, 2021. 3, 4

[20] Johannes Lutz Schonberger and Jan-Michael Frahm. Â¨ Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2, 3

[21] Johannes Lutz Schonberger, Enliang Zheng, Marc Pollefeys, Â¨ and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV), 2016. 2, 3

[22] Shihao Shen, Yilin Cai, Wenshan Wang, and Sebastian Scherer. Dytanvo: Joint refinement of visual odometry and motion segmentation in dynamic environments. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 4048â4055. IEEE, 2023. 3

[23] Jurgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Â¨ Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573â580. IEEE, 2012. 2, 6, 9

[24] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6229â6238, 2021. 3

[25] Wei Tan, Haomin Liu, Zilong Dong, Guofeng Zhang, and Hujun Bao. Robust monocular slam in dynamic environments. In 2013 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages 209â218. IEEE, 2013. 3

[26] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558â16569, 2021. 3, 6

[27] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697â 20709, 2024. 2, 3

[28] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20310â 20320, 2024. 3

[29] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for Gaussian splatting. arXiv preprint arXiv:2409.06765, 2024. 4, 5, 6

[30] Evangelos Ververas Jifei Song Jiankang Deng Gim Hee Lee Yu Chen, Rolandos Alexandros Potamias. Zerogs: Training 3d gaussian splatting from unposed images. In arXiv, 2024. 2, 3

[31] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-

Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825, 2024. 2, 3, 6

[32] Tianwei Zhang, Huayan Zhang, Yang Li, Yoshihiko Nakamura, and Lei Zhang. Flowfusion: Dynamic dense rgbd slam based on optical flow. In 2020 IEEE international conference on robotics and automation (ICRA), pages 7322â 7328. IEEE, 2020. 3

[33] Yan Zhang, Sergey Prokudin, Marko Mihajlovic, Qianli Ma, and Siyu Tang. Degrees of freedom matter: Inferring dynamics from point trajectories. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 3

[34] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and William T Freeman. Structure and motion from casual videos. In European Conference on Computer Vision, pages 20â37. Springer, 2022. 3

[35] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields, 2024. 3, 4, 5, 6