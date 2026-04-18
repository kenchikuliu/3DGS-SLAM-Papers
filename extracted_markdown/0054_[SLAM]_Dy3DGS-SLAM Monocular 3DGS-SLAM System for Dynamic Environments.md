# Dy3DGS-SLAM: Monocular 3D Gaussian Splatting SLAM for Dynamic Environments

Mingrui Li1â , Yiming Zhou2,3,5â , Hongxing Zhou4, Xinggang Hu1, Florian Roemer3, Hongyu Wang1â, Ahmad Osman2,3

Abstractâ Current Simultaneous Localization and Mapping (SLAM) methods based on Neural Radiance Fields (NeRF) or 3D Gaussian Splatting excel in reconstructing static 3D scenes but struggle with tracking and reconstruction in dynamic environments, such as real-world scenes with moving elements. Existing NeRF-based SLAM approaches addressing dynamic challenges typically rely on RGB-D inputs, with few methods accommodating pure RGB input. To overcome these limitations, we propose Dy3DGS-SLAM, the first 3D Gaussian Splatting (3DGS) SLAM method for dynamic scenes using monocular RGB input. To address dynamic interference, we fuse optical flow masks and depth masks through a probabilistic model to obtain a fused dynamic mask. With only a single network iteration, this can constrain tracking scales and refine rendered geometry. Based on the fused dynamic mask, we designed a novel motion loss to constrain the pose estimation network for tracking. In mapping, we use the rendering loss of dynamic pixels, color, and depth to eliminate transient interference and occlusion caused by dynamic objects. Experimental results demonstrate that Dy3DGS-SLAM achieves state-of-the-art tracking and rendering in dynamic environments, outperforming or matching existing RGB-D methods.

## I. INTRODUCTION

Recently, dense SLAM systems expressed through NeRF [1] or 3DGS [2] have attracted significant attention. These systems have achieved photo-realistic rendering results in static scenes and are gradually expanding towards largescale or challenging scenarios. However, a practical issue for SLAM systems is evident: the real world contains a large number of dynamic objects, and current NeRF or 3DGSbased SLAM systems [3]â[7] perform poorly in addressing this challenge. Another issue that has gained attention is how to achieve better results without relying on RGB-D sensors and using only monocular RGB input, which is considered a more accessible sensor and a solution with greater potential.

Although some NeRF-based methods have attempted to address dynamic objects, such as DN-SLAM [8], DDN-SLAM [9], NID-SLAM [10], and RoDyn-SLAM [11], they often rely on predefined dynamic priors or heavily depend on depth priors to determine dynamic object masks, making them unsuitable for environments with only monocular RGB input. Furthermore, due to the limitations of NeRF [1] representation, there are constraints on rendering accuracy, often resulting in severe rendering artifacts. 3DGS-based SLAM systems, such as SplaTAM [12], Photo-SLAM [13], and MonoGS [14], perform well in static environments, but they tend to encounter tracking failures and mapping errors in dynamic scenes.

Therefore, we propose Dy3DGS-SLAM, the first RGBonly 3DGS-SLAM system designed for dynamic environments. We utilize optical flow to obtain dynamic masks without relying on predefined moving objects, though these masks can be noisy in regions with uniform textures or fast motion. To address this, we incorporate monocular depth estimation, providing complementary spatial cues, especially for occlusions and depth discontinuities. We then propose a depth-regularized mask fusion strategy that combines the strengths of both modalities, mitigating individual limitations and producing more precise, robust dynamic masks.

For tracking, we incorporate the estimated depth and fused mask into the motion loss, effectively recovering scale and pose in the pose estimation network, resulting in more accurate tracking outcomes. In terms of rendering, to address transient interference and occlusion, we penalize dynamic Gaussians based on the color and depth of dynamic pixels. Compared to baseline methods, our approach significantly reduces rendering artifacts and greatly improves geometric accuracy. In summary, our method has the following contributions:

â¢ We propose Dy3DGS-SLAM, the first RGB-only 3DGS-SLAM system for dynamic environments, capable of robust tracking and high-fidelity reconstruction in dynamic environments.

â¢ We propose a mask fusion method that accurately covers dynamic objects by combining motion cues from optical flow with geometric consistency from depth estimation. Based on the fused mask, we introduce novel motion and rendering losses to effectively mitigate dynamic object interference in tracking and rendering.

â¢ Our results on three real-world datasets demonstrate that our method achieves better tracking and rendering performance compared to baseline methods.

## II. RELATED WORK

Vision-based SLAM systems are essential technologies for addressing mapping challenges in robotics and scene reconstruction in VR/AR applications [15]â[17]. In real-world scenarios, dynamic objects pose a significant challenge to visual SLAM systems. There has been extensive exploration in the traditional visual SLAM field to tackle the interference caused by dynamic objects. Recently, deep learning-based methods have gained attention, mainly falling into two categories. One type focuses on semantic prior-based segmentation, represented by systems such as DS-SLAM [18], OVD-SLAM [19], and SG-SLAM [20]. These methods utilize deep learning frameworks to recognize semantic information and then use epipolar geometry or depth constraints to identify dynamic points and remove them. The second type relies on optical flow estimation methods, such as FlowFusion [21], DeflowSLAM [22], and DytanVO [23], using deep learningbased optical flow estimation frameworks to estimate camera poses. However, these methods often lack dense and stable reconstruction and fail to accurately recover depth or obtain precise dynamic object masks under monocular conditions.

With the advent of NeRF and 3DGS showing highfidelity reconstruction and fast rendering capabilities in 3D reconstruction, there has been a growing interest in RGB-D SLAM systems. However, these systems often perform poorly in real-world dynamic environments. Some NeRFbased RGB-D SLAM systems have explored this problem. For instance, DN-SLAM [8] uses optical flow estimation to remove dynamic points and employs the Instant NGPbased [24] rendering framework for view synthesis, although it struggles with artifacts. NID-SLAM [10] leverages an optical flow estimation system to obtain dynamic masks and completes background reconstruction, but its tracking accuracy is limited, and the rendering process lacks sufficient constraints. Rodyn-SLAM [11] uses a sliding window optical flow estimation method to acquire motion masks and proposes specific rendering losses, but it heavily depends on prior depth information provided by the sensors. Our approach utilizes the advantages of 3DGS for representation while addressing the reliance on depth sensors through a depth estimation system. By using optical flow estimation, we generate more accurate motion masks, enabling the reconstruction of static scenes effectively.

## III. METHOD

Our system pipeline is shown in Fig. 1. In Section III-A, we address the problem of fusing the dynamic mask obtained from optical flow with the depth map estimated from monocular input, resulting in an accurate dynamic fusion mask. In Section III-B, we propose the motion estimation network and introduce a motion loss incorporating depth estimation, enabling the network to iteratively refine accurate camera poses. In Section III-C, we penalize the Gaussians corresponding to pixels labeled as dynamic and apply an additional rendering loss based on monocular depth to optimize the scene details. Finally, we synthesize a static scene using multi-view consistency.

## A. Dynamic Mask Fusion

Our tracking network includes an optical flow estimation module, a depth estimation module, a mask fusion method, and a pose estimation module. Assume that we have two consecutive undistorted images, $I _ { t }$ and $I _ { t + 1 }$ , as input, and the output is the relative camera motion $E = ( R | T )$ , where $T \in$ $\mathbb { R } ^ { 3 }$ represents the 3D translation and $R \in { \bf S O } ( 3 )$ represents the 3D rotation.

To detect optical flow anomalies caused by dynamic objects, we employ a lightweight U-Net [25] motion segmentation network. It takes the original input frames as input and binarizes the result, setting all optical flow within the mask area to zero, and finally obtains the corresponding optical flow values F and the optical flow mask $F _ { m }$ . However, this estimation is relatively coarse and static areas are easily missegmented, affecting the accuracy of camera pose estimation. Therefore, we introduce additional depth supervision using the DepthanythingV2 [26] estimation network to provide estimated depth, obtaining the corresponding depth mask $D _ { m }$

Although we obtained dynamic masks generated by the optical flow estimation network, there are instances of incorrect mask estimation. Therefore, we need to use the depth mask to correct the optical flow mask. We combine depth and optical flow information using conditional probability to obtain a more accurate fusion mask MË for determining dynamic regions.

First, we aggregate optical flow pixels to separate multiple potential moving objects. We use the K-means clustering algorithm to segment the motion pixels. Let the set of pixels in the motion region be $P _ { \mathrm { d y n a m i c } } = \{ p \ | \ M ( p ) = 1 \}$ . We divide these motion pixels into k clusters, with each cluster representing an independent dynamic object. The goal of clustering is to minimize the sum of squared pixel distances within each cluster. The clustering objective function is defined as follows:

$$
\operatorname* { m i n } _ { \mu _ { 1 } , \mu _ { 2 } , \dots , \mu _ { k } } \sum _ { i = 1 } ^ { k } \sum _ { p \in C _ { i } } \| p - \mu _ { i } \| ^ { 2 } ,\tag{1}
$$

where $C _ { i }$ represents the set of pixels assigned to the i-th cluster, and $\mu _ { i }$ is the center of the i-th cluster, defined as:

$$
\mu _ { i } = { \frac { 1 } { | C _ { i } | } } \sum _ { p \in C _ { i } } p .\tag{2}
$$

Through the clustering result, we can separate moving objects and ensure the correctness of the subsequent fusion mask process. Finally, we obtain the set of moving objects $N = \{ N _ { 1 } , N _ { 2 } , \ldots , N _ { k } \}$ , where $N _ { i }$ represents the pixel set of the i-th moving object. We perform a separate depth probability search for each moving object to achieve mask fusion. Since the depth map and optical flow mask are independent, we propose a Bayesian model for estimating the probability:

$$
P ( D _ { m } , F _ { m } \mid M ( p ) ) = P ( D _ { m } \mid M ( p ) ) \cdot P ( F _ { m } \mid M ( p ) ) .\tag{3}
$$

<!-- image-->  
Fig. 1: Pipeline of Our Network: Our system workflow consists of two main threads: tracking and mapping. In the tracking thread, we use a segmentation optical flow network and a depth estimation network to generate the estimated motion optical flow mask and depth map mask. By applying a conditional probability approach, we create a fused mask MË . This fused mask is subsequently input into the pose estimation network to determine the estimated pose. In the mapping thread, we utilize the fused mask MË along with keyframes to construct the map. We impose color and depth penalties on the Gaussians corresponding to the moving pixels identified by the fused mask, which ultimately results in a multi-view rendering outcome.

We binarize the mask $M ( p )$ by setting it to 1, where $P ( M ( p ) = 1 \mid D _ { m } , F _ { m } )$ represents the posterior probability that a pixel belongs to a moving object. $P ( D _ { m } \mid M ( p ) = 1 )$ is the likelihood of observing the depth map $D _ { m }$ given that the pixel is part of a moving object, and $P ( F _ { m } \mid M ( p ) = 1 )$ is the likelihood of observing the optical flow mask $F _ { m }$ under the same assumption. $P ( M ( p ) = 1 )$ denotes the prior probability that a pixel is part of a moving object.

Finally, by combining all known information and calculating the posterior probability for each pixel, a new motion mask for each pixel is obtained:

$$
\begin{array} { r } { \hat { M } _ { i } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } ~ P ( M ( p ) = 1 | D _ { m } , F _ { m } ) > T } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right. , } \end{array}\tag{4}
$$

where $T$ is the pixel probability threshold, set to 0.95 in our method.

The final fused mask for all moving objects is given by $\hat { M } = M ( N _ { 1 } ) \cup M ( N _ { 2 } ) \cup \dots \cup M ( N _ { i } )$ . Our method does not require additional network iterations and is more generalized, allowing it to handle multiple moving objects simultaneously without the need for scene-specific parameter adjustments.

## B. Tracking for Monocular Dynamic Scenes

Unlike other visual odometry systems [27], our systems can provide accurate pixel depth constraints due to the use of estimated depth information. This is significant in addressing pose estimation errors caused by depth ambiguity. We obtain the static depth mask $M _ { d s } = \{ p \in \hat { M } | M ( p ) = 0 \}$ through the fusion of dynamic masks. The static depth mask is applied to the optical flow map, and the corresponding scale factor $S _ { n }$ is fused to obtain the formula:

$$
\tilde { F } = F \cdot M _ { d s } \cdot S _ { n } .\tag{5}
$$

In this formula, F is the optical flow map, $M _ { d s }$ filters the residual mask of untrustworthy dynamic regions, and $S _ { n }$ provides accurate scale information for the remaining static regions. Through this method, the network can apply more reliable depth information to the static areas, avoiding interference from dynamic regions while correcting scale errors in monocular estimation.

We fuse the acquired static mask $M _ { d s }$ with the optical flow mask $\tilde { F }$ and input them into the network to update the pose. To achieve a more accurate iterative pose estimation process, we introduce a camera motion loss $\mathcal { L } _ { \mathcal { M } }$ , adjusting the estimated pose distance from the ground truth. The loss function with the introduced scale constraint is expressed as:

$$
\mathcal { L } _ { \mathcal { M } } = \frac { \hat { T } } { \operatorname* { m a x } ( | \hat { T } \cdot S _ { n } | , \varepsilon ) } - \frac { T } { \operatorname* { m a x } ( | T \cdot S _ { n } | , \varepsilon ) } + ( \hat { R } - R ) \cdot M _ { d s } ,\tag{6}
$$

where $S _ { n }$ is used to adjust the scale of the translation vector to align it with the true scale value. We perform pose updates within a pose estimation network based on ResNet50 [28], following the training design in TartanVO [27]. The network jointly optimizes optical flow loss $\mathcal { L } _ { \mathcal { O } }$ , motion segmentation loss $\mathcal { L } _ { \mathcal { U } }$ , and camera motion loss $\mathcal { L } _ { \mathcal { M } }$ , which incorporates depth masking and scale constraints. DytanVO [23] improves camera pose estimation and dynamic mask segmentation through three iterations, but the final results still have limitations. In contrast, our method requires only a single iteration without incurring additional computational costs. The comprehensive tracking loss function is formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathcal { P } } = \lambda _ { 1 } \mathcal { L } _ { \mathcal { O } } + \lambda _ { 2 } \mathcal { L } _ { \mathcal { U } } + \mathcal { L } _ { \mathcal { M } } , } \end{array}\tag{7}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are weights that control the different loss terms, ensuring that the network can balance the tasks of optical flow, motion segmentation, and pose estimation during training.

For more accurate pose estimation, we generate a keyframe every 10 frames and create a keyframe group consisting of at least 4 keyframes, applying local Bundle Adjustment optimization to correct accumulated errors.

## C. Gaussian Rendering for Monocular Dynamic Scenes

In 3DGS [2], an explicit point-based scene representation is optimized. Each 3D Gaussian is parameterized by a set of 3D attributes, including position, opacity, scale, and rotation. The Gaussian ellipsoid is characterized by a full 3D covariance matrix Î£, which is defined (normalized) in world space. The Gaussian function is defined as:

$$
\begin{array} { l } { { \displaystyle g ( { \bf x } ) = o \exp \left( - \frac { 1 } { 2 } { \bf x } ^ { T } { \Sigma } ^ { - 1 } { \bf x } \right) , { \Sigma } = R S S ^ { T } R ^ { T } } , } \end{array}\tag{8}
$$

where Î£ is the covariance matrix, $o \in [ 0 , 1 ]$ represents the opacity value, S is the scale matrix, and R is the rotation matrix.

We use 3D Gaussian ellipsoids to render 2D images through splatting techniques, as described in [29], [30]. In the camera coordinate system, the covariance matrix $\Sigma ^ { \prime }$ is formulated as:

$$
\pmb { \Sigma ^ { \prime } } = \pmb { J } \pmb { W } \pmb { \Sigma } \pmb { W } ^ { T } \pmb { J } ^ { T } ,\tag{9}
$$

where W represents the viewing direction, and J is the Jacobian matrix of the affine approximation of the projection transformation. For each pixel, the color and opacity of all Gaussian ellipsoids are computed and blended using the following formula:

$$
C = \sum _ { i \in N } c _ { i } g _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - g _ { j } \right) ,\tag{10}
$$

where $c _ { i }$ represents the color of the i-th Gaussian ellipsoid. Additionally, we propose a similar formula for depth rendering:

$$
D = \sum _ { i = 1 } ^ { n } d _ { i } g _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - g _ { j } \right) ,\tag{11}
$$

where $d _ { i }$ is the z-axis depth of the center of the i-th 3D Gaussian.

Since our method is based on keyframe multi-view rendering, each Gaussian $g _ { i }$ is associated with a keyframe that anchors it to the map G. For the Gaussians produced by pixels marked as dynamic, we set their depth to infinity to perform pruning. However, this may cause artifacts that are hard to remove, so we apply photometric loss $L _ { c }$ and depth loss $L _ { d }$ to the pixel masking process to eliminate the artifactsâ impact.

The new photometric loss is:

$$
L _ { c } = \lambda _ { d } \cdot \frac { N _ { d } } { N _ { p i } } \left| C _ { k } - C _ { k } ^ { g t } \right| + \lambda _ { s } \cdot \frac { N _ { p i } - N _ { d } } { N _ { p i } } \left| C _ { k } - C _ { k } ^ { g t } \right| ,\tag{12}
$$

where $\lambda _ { d }$ is the penalty factor for dynamic pixel masks, $\lambda _ { s }$ is the penalty factor for static pixel masks, $N _ { p i }$ is the number

of pixels in each keyframe, and $N _ { d }$ represents the number of pixels corresponding to the dynamic mask.

The new depth loss is:

$$
L _ { d } = \lambda _ { t } \cdot \frac { D _ { d } } { D _ { p i } } \left| D _ { k } - D _ { k } ^ { e } \right| + \lambda _ { m } \cdot \frac { D _ { p i } - D _ { d } } { D _ { p i } } \left| D _ { k } - D _ { k } ^ { e } \right| ,\tag{13}
$$

where $\lambda _ { t }$ is the penalty factor for dynamic depth masks, $\lambda _ { m }$ is the penalty factor for static depth masks, $D _ { p i }$ is the number of pixels corresponding to the estimated depth in each keyframe, $D _ { d }$ represents the depth corresponding to the dynamic mask, and $D _ { k } ^ { e }$ represents the depth generated by monocular estimation.

The final rendering loss function $L _ { G }$ is:

$$
L _ { G } = L _ { c } + \lambda \cdot L _ { d } ,\tag{14}
$$

where Î» is a hyperparameter set to 1.

## IV. EXPERIMENTAL RESULTS

## A. Experimental Details and Metrics

Datasets and Implementation details. We evaluated our method on three public datasets from the real world: the TUM RGB-D dataset [31], AirDOS-Shibuya dataset [32] and the BONN RGB-D dynamic dataset [33], all of which capture real indoor environments. We conducted our SLAM experiments on a desktop equipped with a single RTX 3090 Ti GPU. We present results from our multiprocess implementation designed for real-time applications. Consistent with the 3DGS framework, time-critical rasterisation and gradient computation are implemented using CUDA.

Metrics and Baseline Methods. To evaluate camera tracking accuracy, we report the Root Mean Square Error (RMSE) of the Absolute Trajectory Error (ATE) for keyframes. For runtime performance and network iteration speed, we measure frames per second (FPS) and milliseconds (ms), respectively. GPU usage is assessed in megabytes (MB). We compare our Dy3DGS-SLAM method against traditional dynamic SLAM approaches, such as ORB-SLAM3 [34], Droid-SLAM [35], DynaSLAM [11], DytanVO [23] and ReFusion [33], as well as state-of-the-art NeRF-based methods utilizing RGB-D sensors, including NICE-SLAM [4], ESLAM [36], Co-SLAM [37], and NID-SLAM [10]. Furthermore, we consider SplaTAM [12], which is based on 3DGS.

## B. Evaluation on TUM and Bonn RGB-D

Tracking. As shown in Table II, we present results for three highly dynamic sequences, one mildly dynamic sequence, and two static sequences from the TUM dataset [31]. Thanks to our proposed dynamic mask fusion method, our system demonstrates advanced tracking performance compared to RGB-D-based methods and is even competitive with traditional SLAM methods. Furthermore, we evaluated the tracking performance on the more complex and challenging BONN dataset [33], as illustrated in Table I. Even in these more complicated and large-scale scenarios, our method achieved superior performance. Our method outperforms all other approaches, with NID-SLAM [10] being the only one achieving results close to ours. Additionally, our method demonstrates superior performance compared to traditional methods. This highlights our dynamic mask fusion can effectively remove the dynamic objects and enhance the tracking process.

<!-- image-->  
Fig. 2: Visual comparison of the reconstructed meshes on the BONN and TUM RGB-D datasets. Our results are more complete and accurate without the dynamic object floaters.

TABLE I: Tracking performance on BONN-RGB-D (ATE RMSE â [cm]). The best results are highlighted as first , second and third .
<table><tr><td>Methods</td><td>ballon</td><td>ballon2</td><td>ps_track</td><td>ps_track2</td><td>mv_box1</td><td>mv_box2</td><td>Avg.</td></tr><tr><td>Traditional SLAM methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ORB-SLAM3 [34]</td><td>5.8</td><td>17.7</td><td>70.7</td><td>77.9</td><td>3.1</td><td>3.7</td><td>29.8</td></tr><tr><td>Droid-SLAM [35]</td><td>5.4</td><td>4.6</td><td>21.3</td><td>46.0</td><td>8.9</td><td>5.9</td><td>15.4</td></tr><tr><td>DynaSLAM [38]</td><td>3.0</td><td>3.0</td><td>6.1</td><td>7.8</td><td>4.9</td><td>3.9</td><td>4.8</td></tr><tr><td>ReFusion [33]</td><td>17.5</td><td>25.4</td><td>28.9</td><td>46.3</td><td>30.2</td><td>17.9</td><td>27.7</td></tr><tr><td>DytanVO [23]</td><td>6.3</td><td>3.1</td><td>3.5</td><td>9.0</td><td>5.5</td><td>6.1</td><td>5.6</td></tr><tr><td colspan="8">NeRF or 3DGS based SLAM methods</td></tr><tr><td>NICE-SLAM [4]</td><td>80.3</td><td>66.8</td><td>54.9</td><td>45.3</td><td></td><td></td><td></td></tr><tr><td>Co-SLAM [37]</td><td>28.8</td><td>20.6</td><td>61.0</td><td>59.1</td><td>21.2 38.3</td><td>31.9 70.0</td><td>44.1</td></tr><tr><td>ESLAM [36]</td><td>22.6</td><td>36.2</td><td>48.0</td><td>51.4</td><td>12.4</td><td>117.7</td><td>46.3 31.4</td></tr><tr><td>NID-SLAM [10]</td><td>3.7</td><td>2.8</td><td>10.0</td><td>14.7</td><td>6.9</td><td>4.2</td><td>7.1</td></tr><tr><td>SplaTAM [12]</td><td>231.9</td><td>126.4</td><td>27.8</td><td>53.1</td><td>63.7</td><td>73.6</td><td>96.1</td></tr><tr><td>Dy3DGS-SLAM (Ours)</td><td>4.5</td><td>1.9</td><td>5.6</td><td>6.4</td><td>5.2</td><td>3.8</td><td>4.5</td></tr></table>

Mapping. To comprehensively evaluate the performance of our proposed system in dynamic scenes, we analyze the results from a qualitative perspective. We compare the rendered images with ground truth poses obtained from the generated Gaussian map, using the same viewpoint as other methods. Four challenging sequences were selected: crowd and person tracking from the BONN dataset, and f3 walk xyz val and f3 walk static from the TUM RGB-D dataset. As shown in Fig. 2, our method shows significant advantages in geometric and texture details, especially in reducing artifacts. Notably, our approach is based on a monocular system and has been validated on two realworld datasets, demonstrating its capability to accurately record dynamic scenes with just a simple camera. This highlights the potential of our method to effectively track and reconstruct indoor environments, making it a valuable tool for applications where depth sensors may not be available.

## V. ABLATION STUDY

## A. Fusion Mask Strategy Evaluation

To evaluate the effectiveness of the proposed methods in our system, we conduct ablation studies on five scenes from the AirDOS-Shibuya dataset, with all results being the average of five experiments, as shown in Table III. We calculate the average Absolute Trajectory Error (ATE) to assess the impact of each method on the overall system performance. The results in Table IV indicate that all the proposed methods contribute to improved camera tracking. Compared to using only the optical flow estimation strategy, our method improves ATE by 60.52%, indicating that integrating the optical flow mask with the depth mask can effectively enhance pose estimation.

TABLE II: Tracking performance on TUM-RGB-D (ATE RMSE â [cm]). â-â denotes the absence of mention. âXâ denotes the tracking failures. The best results are highlighted as first , second and third .
<table><tr><td rowspan="2">Methods</td><td colspan="4">High Dynamic</td><td colspan="2">Low Dynamic</td><td rowspan="2">Avg.</td></tr><tr><td>f3/wk_xyz</td><td>f3/wk_hf</td><td>f3/wk_st</td><td>f3/st_hf</td><td>f3/st_xyz</td><td>f1/st rpy</td></tr><tr><td colspan="8">Traditional SLAM methods</td></tr><tr><td>ORB-SLAM3 [34]</td><td>28.1</td><td>30.5</td><td>2.0</td><td>2.6</td><td>2.2</td><td>2.8</td><td>11.5</td></tr><tr><td>DVO-SLAM [39]</td><td>59.7</td><td>52.9</td><td>21.2</td><td>6.2</td><td>2.1</td><td>3.0</td><td>23.2</td></tr><tr><td>DynaSLAM [38]</td><td>1.7</td><td>2.6</td><td>0.7</td><td>2.8</td><td>1.6</td><td>5.1</td><td>2.7</td></tr><tr><td>ReFusion [33]</td><td>9.9</td><td>10.4</td><td>1.7</td><td>11.0</td><td>-</td><td>-</td><td>8.3</td></tr><tr><td>DytanVO [23]</td><td>8.7</td><td>9.8</td><td>9.5</td><td>14.7</td><td>12.4</td><td>12.3</td><td>11.2</td></tr><tr><td colspan="8">NeRF or 3DGS based SLAM methods</td></tr><tr><td>NICE-SLAM [4]</td><td>113.8</td><td>X</td><td>137.3</td><td>93.0</td><td>43.9</td><td>65.6</td><td>90.6</td></tr><tr><td>Co-SLAM [37]</td><td>51.8</td><td>105.1</td><td>49.5</td><td>4.7</td><td>9.3</td><td>8.9</td><td>36.3</td></tr><tr><td>ESLAM [36]</td><td>45.7</td><td>60.8</td><td>93.6</td><td>3.6</td><td>8.6</td><td>9.2</td><td>34.5</td></tr><tr><td>NID-SLAM [10]</td><td>6.4</td><td>7.1</td><td>6.2</td><td>10.9</td><td>7.5</td><td>8.6</td><td>7.8</td></tr><tr><td>SplaTAM [12]</td><td>41.3</td><td>72.6</td><td>45.7</td><td>75.9</td><td>32.8</td><td>40.5</td><td>53.2</td></tr><tr><td>Dy3DGS-SLAM (Ours)</td><td>5.8</td><td>7.0</td><td>6.5</td><td>3.4</td><td>2.0</td><td>3.8</td><td>4.7</td></tr></table>

<!-- image-->  
Fig. 3: Evaluation of Tracking Network Loss Methods

## B. Evaluation of Mask Fusion Accuracy

We compared the mask results obtained with different loss functions $\mathcal { L } _ { \mathcal { P } }$ as shown in Fig. 3. When using only the optical flow mask or only the motion loss similar to DytanVO, significant mask estimation errors appeared. Our strategy achieved the best results, closely approaching the ground truth.

## C. Operating Speed and Network Performance Evaluation

We evaluated the systemâs tracking, mapping, and network update speed using the AirDOS-Shibuya dataset as shown in Table IV. In comparison to DytanVO and the state-ofthe-art 3DGS-based SLAM system SplaTAM, our approach demonstrates a superior balance between runtime efficiency and performance. This advantage primarily stems from the absence of a multi-iteration network update strategy in our system.

TABLE III: Ablation study of the proposed method in our systems. The best result is highlighted as first .
<table><tr><td>Optical Flow</td><td>Depth</td><td>Optical Flow+Depth</td><td>ATE RMSE (cm) â</td></tr><tr><td></td><td></td><td></td><td>7.6</td></tr><tr><td>&gt; x</td><td>Ã &gt;</td><td>Ã x</td><td>94.8</td></tr><tr><td></td><td></td><td></td><td>3.0</td></tr></table>

TABLE IV: Operating speed and network performance. The best result is highlighted as first .
<table><tr><td>Method</td><td>Tracking (FPS) â</td><td>Mapping (ms) â</td><td>Network update (ms) â</td><td>GPU memory (MB) â</td></tr><tr><td>DytanVO</td><td>10.5</td><td>Ã</td><td>32.9</td><td>7.6</td></tr><tr><td>pplaTAM</td><td>3.8</td><td>390.4</td><td>Ã</td><td>14.6</td></tr><tr><td>Ours</td><td>17.0</td><td>430.5</td><td>10.3</td><td>12.8</td></tr></table>

## VI. CONCLUSIONS

We propose Dy3DGS-SLAM, the first 3DGS-based SLAM method designed for dynamic scenes using monocular RGB input. This method first generates dynamic object masks through optical flow estimation, combining these masks with monocular depth estimation to create a fused mask and recover scale, accurately capturing dynamic object masks. To further improve pose accuracy, we optimized the loss function based on the fused mask, reducing the computational cost associated with multiple iterations. Additionally, to enhance rendering performance, we applied additional photometric and depth losses to eliminate transient interference artifacts and improve geometric accuracy. Experimental results demonstrate that, compared to baseline methods, Dy3DGS-SLAM achieves state-of-the-art tracking and rendering performance in dynamic environments. In the future, we will focus on applying this approach to mobile devices with lower computational costs.

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Transactions on Graphics (TOG), vol. 42, no. 4, pp. 139â1, 2023.

[3] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, âimap: Implicit mapping and positioning in real-time,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 6229â6238.

[4] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 786â12 796.

[5] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, âPoint-Â¨ slam: Dense neural point cloud-based slam,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 433â18 444.

[6] F. Tosi, Y. Zhang, Z. Gong, E. Sandstrom, S. Mattoccia, M. R. Oswald, Â¨ and M. Poggi, âHow nerfs and 3d gaussian splatting are reshaping slam: a survey,â arXiv preprint arXiv:2402.13255, vol. 4, 2024.

[7] Y. Zhou, Z. Zeng, A. Chen, X. Zhou, H. Ni, S. Zhang, P. Li, L. Liu, M. Zheng, and X. Chen, âEvaluating modern approaches in 3d scene reconstruction: Nerf vs gaussian-based methods,â arXiv preprint arXiv:2408.04268, 2024.

[8] C. Ruan, Q. Zang, K. Zhang, and K. Huang, âDn-slam: A visual slam with orb features and nerf mapping in dynamic environments,â IEEE Sensors Journal, vol. 24, no. 4, pp. 5279â5287, 2024.

[9] M. Li, Y. Zhou, G. Jiang, T. Deng, Y. Wang, and H. Wang, âDdn-slam: Real-time dense dynamic neural implicit slam with joint semantic encoding,â arXiv preprint arXiv:2401.01545, 2024.

[10] Z. Xu, J. Niu, Q. Li, T. Ren, and C. Chen, âNid-slam: Neural implicit representation-based rgb-d slam in dynamic environments,â arXiv preprint arXiv:2401.01189, 2024.

[11] H. Jiang, Y. Xu, K. Li, J. Feng, and L. Zhang, âRodyn-slam: Robust dynamic dense rgb-d slam with neural radiance fields,â IEEE Robotics and Automation Letters, vol. 9, no. 9, pp. 7509â7516, 2024.

[12] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[13] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Real-time simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â 21 593.

[14] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[15] Y. Zhu, C. Honnet, Y. Kang, J. Zhu, A. J. Zheng, K. Heinz, G. Tang, L. Musk, M. Wessely, and S. Mueller, âDemonstration of chromocloth: Re-programmable multi-color textures through flexible and portable light source,â in Adjunct Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, 2023, pp. 1â3.

[16] Y. Song, P. Arora, R. Singh, S. T. Varadharajan, M. Haynes, and T. Starner, âGoing blank comfortably: Positioning monocular headworn displays when they are inactive,â in Proceedings of the ACM International Symposium on Wearable Computers, 2023, pp. 114â118.

[17] Y. Song, P. Arora, S. T. Varadharajan, R. Singh, M. Haynes, and T. Starner, âLooking from a different angle: Placing head-worn displays near the nose,â in Proceedings of the Augmented Humans International Conference, 2024, pp. 28â45.

[18] C. Yu, Z. Liu, X.-J. Liu, F. Xie, Y. Yang, Q. Wei, and Q. Fei, âDs-slam: A semantic visual slam towards dynamic environments,â in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 1168â1174.

[19] J. He, M. Li, Y. Wang, and H. Wang, âOvd-slam: An online visual slam for dynamic environments,â IEEE Sensors Journal, vol. 23, no. 12, pp. 13 210â13 219, 2023.

[20] S. Cheng, C. Sun, S. Zhang, and D. Zhang, âSg-slam: A realtime rgb-d visual slam toward dynamic scenes with semantic and geometric information,â IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1â12, 2022.

[21] T. Zhang, H. Zhang, Y. Li, Y. Nakamura, and L. Zhang, âFlowfusion: Dynamic dense rgb-d slam based on optical flow,â in IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 7322â7328.

[22] W. Ye, X. Yu, X. Lan, Y. Ming, J. Li, H. Bao, Z. Cui, and G. Zhang, âDeflowslam: Self-supervised scene motion decomposition for dynamic dense slam,â arXiv preprint arXiv:2207.08794, 2022.

[23] S. Shen, Y. Cai, W. Wang, and S. Scherer, âDytanvo: Joint refinement of visual odometry and motion segmentation in dynamic environments,â in IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 4048â4055.

[24] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[25] O. Ronneberger, P. Fischer, and T. Brox, âU-net: Convolutional networks for biomedical image segmentation,â in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.

[26] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, âDepth anything v2,â arXiv preprint arXiv:2406.09414, 2024.

[27] W. Wang, Y. Hu, and S. Scherer, âTartanvo: A generalizable learningbased vo,â in Conference on Robot Learning. PMLR, 2021, pp. 1761â1772.

[28] K. He, X. Zhang, S. Ren, and J. Sun, âDeep residual learning for image recognition,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016, pp. 770â778.

[29] G. Kopanas, J. Philip, T. Leimkuhler, and G. Drettakis, âPoint-based Â¨ neural rendering with per-view optimization,â in Computer Graphics Forum, vol. 40, no. 4. Wiley Online Library, 2021, pp. 29â43.

[30] W. Yifan, F. Serena, S. Wu, C. Oztireli, and O. Sorkine-Hornung, Â¨ âDifferentiable surface splatting for point-based geometry processing,â ACM Transactions on Graphics (TOG), vol. 38, no. 6, pp. 1â14, 2019.

[31] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of rgb-d slam systems,â in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012, pp. 573â580.

[32] Y. Qiu, C. Wang, W. Wang, M. Henein, and S. Scherer, âAirdos: Dynamic slam benefits from articulated objects,â in IEEE International Conference on Robotics and Automation (ICRA), 2022, pp. 8047â 8053.

[33] E. Palazzolo, J. Behley, P. Lottes, P. Giguere, and C. Stachniss, \` âRefusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals,â in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019, pp. 7855â7862.

[34] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âOrb-slam3: An accurate open-source library for visual, Â´ visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[35] Z. Teed and J. Deng, âDroid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras,â Advances in Neural Information Processing Systems, vol. 34, pp. 16 558â16 569, 2021.

[36] M. M. Johari, C. Carta, and F. Fleuret, âEslam: Efficient dense slam system based on hybrid representation of signed distance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 408â17 419.

[37] H. Wang, J. Wang, and L. Agapito, âCo-slam: Joint coordinate and sparse parametric encodings for neural real-time slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 293â13 302.

[38] B. Bescos, J. M. Facil, J. Civera, and J. Neira, âDynaslam: Tracking, Â´ mapping, and inpainting in dynamic scenes,â IEEE Robotics and Automation Letters, vol. 3, no. 4, pp. 4076â4083, 2018.

[39] C. Kerl, J. Sturm, and D. Cremers, âDense visual slam for rgb-d cameras,â in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2013, pp. 2100â2106.