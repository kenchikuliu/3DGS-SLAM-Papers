# GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting

Jongwon Lee1 and Timothy Bretl1

Abstractâ In this paper, we present a method for localizing a query image with respect to a precomputed 3D Gaussian Splatting (3DGS) scene representation. First, the method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP) problem to produce a final pose estimate. Results from evaluation across three existing datasets with 38 scenes and over 2,700 test images show that our method significantly reduces both inference time (by over two orders of magnitude, from more than 10 seconds to as fast as 0.1 seconds) and estimation error compared to baseline methods that use photometric loss minimization. Results also show that our method tolerates large errors in the initial pose estimate of up to $\mathbf { 5 5 } ^ { \circ }$ in rotation and 1.1 units in translation (normalized by scene scale), achieving final pose errors of less than $\mathbf { 5 } ^ { \circ }$ in rotation and 0.05 units in translation on 90% of images from the Synthetic NeRF and Mip-NeRF360 datasets and on 42% of images from the more challenging Tanks and Temples dataset.

## I. INTRODUCTION

Visual localization is the process of determining the pose (position and orientation) of a query image with respect to a previously reconstructed scene (i.e., a map). It is a key component of both low-level perception modules, such as relocalization in visual simultaneous localization and mapping (vSLAM) when tracking is lost, and highlevel systems for autonomous navigation, augmented reality, robotic manipulation, and humanârobot interaction [1].

In this paper, we restrict our attention to visual localization with respect to a 3D Gaussian Splatting (3DGS) scene representation, in particular [2]. This scene representation offers high-quality novel view synthesisâwhich alternatives like point clouds and meshes lackâwith significantly faster training and rendering than neural radiance fields (NeRF) [3], making it appealing for robotics applications [4].

Existing 3DGS-based visual localization approaches generally estimate the camera pose by defining and minimizing the photometric loss between a rendered image at the estimated pose and the query image [5-8]. However, these photometric loss minimization approachesâwhich rely on gradient-based optimization by iteratively rendering and comparing imagesâhave slow inference times that are insufficient for real-time applications (e.g., significantly slower than 10 frames per second). For instance, iComMa [5] requires over one second per image on an NVIDIA RTX

<!-- image-->  
(a) Query and Initial Pose

<!-- image-->

<!-- image-->  
(b) Rendered Dpth  
(c) Estimate  
Fig. 1: Our method estimates the pose of a query image on a 3D Gaussian Splatting (3DGS) scene by establishing feature correspondences between the query image and an image rendered at a rough initial pose (a). The matched points are then lifted to 3D using the depth map rendered from 3DGS (b), and the final pose is estimated by solving a PnP problem. The image rendered at the estimated pose (c) is also shown.

A6000 GPUâoutperforming its NeRF-based counterpart, iNeRF [9], which takes over ten secondsâyet still falls short of real-time performance.

To address these limitations, we propose a method of visual localization with respect to a 3DGS scene representation that needs to render only one synthetic image and that uses feature correspondence rather than photometric loss minimization (see Fig. 1 for a high-level overview and Fig. 2 for a detailed workflow). First, our method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP)

problem to produce a final pose estimate. Conceptually, our method is a feature-based alternative to iComMa [5], which uses photometric loss minimization with respect to a 3DGS scene representation, in the same way that the method of Chen et al. [10] is a feature-based alternative to iNeRF [9] or PiNeRF [11], which use photometric loss minimization with respect to a NeRF scene representation. This relationship is summarized by Table I.

Although we assume that an initial pose estimate is available, the error in this initial pose estimate can be large. In particular, the results we present in Section IV show that our method tolerates initial errors of up to 55â¦ in rotation and 1.1 units in translation (normalized by scene scale), which is the maximum error that would still allow some overlap between the query image and the synthetic image that is rendered at the initial pose estimate.

Our paper is structured as follows. Section II reviews existing literature on visual localization methods, particularly those based on NeRF or 3DGS as scene representations. Section III provides details on the proposed method based on feature correspondence. Section IV validates our method, in comparison with open-source baseline methods that use photometric loss minimization on NeRF or 3DGS. We conduct experiments under two different scenarios: when the initial pose is randomly sampled (Section IV-B and Section IV-C) and when it is provided by 6DGS [12] (Section IV-D). We also analyze the sensitivity of the method with respect to error in the initial pose estimate (Section IV-E), and assess the effect of the choice of feature extractor and matcher (Section IV-F). Finally, Section V concludes the paper and discusses several directions for future work.

## II. RELATED WORKS

The emergence of NeRF [3] and 3DGS [2], known for their realistic scene reconstruction and novel view synthesis capabilities, has introduced new directions in visual localization. While some recent approaches incorporate NeRF or 3DGS into localization pipelines without requiring any initial pose estimates (e.g., through fully end-to-end learning-based methods [12]), a more common setting assumes a rough initial poseâwithin a range around the ground-truth such that the corresponding view overlaps with the query imageâ typically obtained from image retrieval or auxiliary sensors, and seeks to refine it on a NeRF [9-11] or 3DGS [5-8] scene representation. Our focus is on this line of work, which explicitly performs localization given a rough initial pose.

With this scope in mind, we begin by reviewing advances in visual localization with a rough initial pose using NeRF, as the advent of NeRF preceded that of 3DGS. NeRF facilitates approach using photometric loss minimization thanks to its ability to render novel views that were not used during reconstruction. In this approach, the camera pose is estimated by minimizing the discrepancy between the query image and an image rendered at the estimated pose, as proposed by iNeRF [9] and piNeRF [11]. The former compares the query image with an image rendered from a single initial pose, while the latter, building upon iNeRF, renders multiple images from poses sampled around the initial pose in parallel, thereby better avoiding local minima. Meanwhile, methods that do not incorporate photometric loss fall into a separate category, namely those based on feature correspondence. Chen et al. [10] estimate camera pose by matching 2D feature points between a query image and an image rendered from an initial pose using a NeRF, lifting them into 3D using rendered depth, filtering out unreliable 3D points through a consistency check, and solving for pose via PnP. As a post-processing step, they further refine the pose by applying photometric loss minimization (e.g., iNeRF [9] or piNeRF [11]) with a reduced number of iterations.

Compared to NeRF, 3DGS is a more recent scene representation and has just begun to be explored for localization. Most existing methods rely on photometric loss [5-8]. A representative example in this category is iComMa [5], which estimates camera pose by minimizing both photometric and feature-matching losses between a query image and an image rendered from an initial pose using a pre-constructed 3DGS, where the feature-matching loss is defined as the Euclidean distance between corresponding feature points on the two images. The two losses are equally weighted in early iterations, with only photometric loss used in later iterations. However, such a method based on photometric loss minimization requires iterative rendering and comparisonâoften over hundreds of stepsâresulting in high computational cost and limited real-time applicability (e.g., reported inference times exceeding one second per image [5]).

In this work, we propose a visual localization method using 3DGS, under the assumption that a rough initial pose estimate is provided. Our approach is analogous to that of Chen et al. [10], which applies classical feature correspondence techniques to a NeRF-based scene representation. Specifically, our method uses a subset of their pipeline: unlike Chen et al., we do not perform the consistency-based filtering of 3D points or apply photometric loss minimization for post-pose refinement. By using feature correspondence instead of photometric loss minimization, our method offers a significantly faster alternative for 3DGS-based localization, while not compromising estimation accuracy and even allowing a wider range of initial pose estimates. Table I summarizes representative visual localization approaches that assume a rough initial pose and operate on NeRF or 3DGS, as discussed in this section.

## III. METHODOLOGY

Our method estimates the camera pose of a query image by first establishing 2Dâ2D correspondences with a rendered image at an initial pose, and then solving a PnP problem using the corresponding 3D points obtained from the rendered depth map of the 3DGS representation (see Fig. 2). Our method follows the approach introduced by Chen et al. [10], which is based on feature correspondence with NeRF as the scene representation. Notably, our method is a simplified version of theirs, omitting both the consistency-based filtering of 3D points and the photometric loss minimization used for post-pose refinement.

<!-- image-->  
Fig. 2: Overview of the proposed pipeline for visual localization using 3DGS as the scene representation, given a query image $I _ { q } ,$ an initial pose estimate $\bar { \mathbf { T } } ^ { 0 } \in \mathrm { S E } ( 3 )$ , and the 3DGS scene representation.

TABLE I: REPRESENTATIVE VISUAL LOCALIZATION METHODS WITH AN INITIAL POSE ESTIMATE, CATEGO-RIZED BY SCENE REPRESENTATION AND POSE ESTIMATION APPROACH
<table><tr><td rowspan=2 colspan=1>Approach</td><td rowspan=1 colspan=2>SceneRepresentation</td></tr><tr><td rowspan=1 colspan=1>NeRF</td><td rowspan=1 colspan=1>3DGS</td></tr><tr><td rowspan=1 colspan=1>Photometric LossMinimization</td><td rowspan=1 colspan=1>iNeRFâ [9],piNeRFâ [11]</td><td rowspan=1 colspan=1>iComMaâ  [5]</td></tr><tr><td rowspan=1 colspan=1>FeatureCorrespondence</td><td rowspan=1 colspan=1>Chen et al. [10]</td><td rowspan=1 colspan=1>Ours</td></tr></table>

â : Source code available

## A. Problem statement

Suppose we are given a query image $I _ { q } ,$ an initial estimate of its camera pose in the world coordinate frame $\mathbf { T } _ { C , \mathrm { i n i t } } ^ { W } \in$ SE(3), and a 3DGS representation of the scene. The goal of visual localization is to estimate the camera pose $\mathbf { T } _ { C } ^ { \breve { W } } \in$ SE(3) from which the query image was captured, relative to the world coordinate frame in which the 3DGS scene is represented. For simplicity, we omit the superscript and subscript in the coordinate notation and denote $\mathbf { T } _ { C , \mathrm { i n i t } } ^ { W }$ and $\mathbf { T } _ { C } ^ { W }$ as $\mathbf { T } ^ { 0 }$ and T, respectively.

## B. Step 1: Rendering the image and the depth map

The first step starts by rendering the image $I _ { r } : = \widehat { I } ( \mathbf { T } ^ { 0 } )$ and the depth map $\begin{array} { r } { D _ { r } : = \widehat { D } ( \mathbf { T } ^ { 0 } ) } \end{array}$ at the initial pose $\mathbf { T } ^ { 0 }$ respectively, where $\widehat { I } ( \cdot )$ and $\widehat { D } ( \cdot )$ denotes the rendering of the RGB and the depth image from the 3DGS.

## C. Step 2: Establishing 2D-2D feature correspondences

The next step is to extract feature points from both the query image $I _ { q }$ and the rendered image $I _ { r }$ , and to establish

correspondences between them. As a result, a set of 2D correspondences is obtained as

$$
\{ ( \mathbf { u } _ { q , i } , \mathbf { u } _ { r , i } ) \} _ { i = 1 } ^ { N } ,
$$

where $N$ denotes the number of matched feature pairs between $I _ { q }$ and $I _ { r }$ , and $\mathbf { u } _ { q , i } , \ \mathbf { u } _ { r , i } \ \in \ \mathbb { R } ^ { 2 }$ represent the corresponding pixel coordinates in the query and rendered images, respectively.

## D. Step 3: Establishing 2Dâ3D feature correspondences

The next step is to lift the 2D feature points $\mathbf { u } _ { r , i } ,$ , where $i \in \{ 1 , \ldots , N \}$ , on the rendered image $I _ { r }$ into 3D space. This process is conducted by projecting each 2D point into the camera coordinate frame using the rendered depth map $D _ { r }$ and the camera intrinsic matrix K, as follows:

$$
\mathbf { p } _ { i } ^ { C } = K ^ { - 1 } \bar { \mathbf { u } } _ { r , i } d ,
$$

where d denotes the depth at pixel location ${ \bf u } _ { r , i }$ , obtained from the depth map $D _ { r }$ , and $\bar { \mathbf { u } } _ { r , i }$ is the homogeneous representation of $\mathbf { u } _ { r , i }$

Each $\mathbf { p } _ { i } ^ { C }$ is transformed to the world coordinate frame as

$$
\bar { \mathbf { p } } _ { i } ^ { W } = \mathbf { T } ^ { 0 } \bar { \mathbf { p } } _ { i } ^ { C } ,
$$

where $\mathbf { T } ^ { 0 } \in \mathrm { S E } ( 3 )$ is the initial pose estimate from which $I _ { r }$ and $D _ { r }$ are rendered, and $\bar { \mathbf { p } } _ { i } ^ { W } , \bar { \bar { \mathbf { p } } } _ { i } ^ { C }$ denote the homogeneous representations of $\mathbf { p } _ { i } ^ { W }$ and $\mathbf { p } _ { i } ^ { C }$ , respectively.

The resulting 3D point $\bar { \mathbf { p } } _ { i } ^ { W }$ is thus associated with both the corresponding query feature point $\mathbf { u } _ { q , i }$ and the rendered feature point $\mathbf { u } _ { r , i } .$ , from which it was lifted.

## E. Step 4: Pose estimation

Once the 2Dâ3D correspondencesâi.e., the set of pairs {(uq,i, pWi )}Ni=1

are established, the camera pose T from which the query image was captured can be estimated using the PnP algorithm. We choose a PnP solver that minimizes the sum of squared reprojection errors using the LevenbergâMarquardt optimization [13]. A minimum of four 2Dâ3D correspondences is required to compute a valid solution. To increase robustness to outliers, we apply RANSAC and use only the inlier correspondences for the final pose estimation. This process yields the final estimate of the camera pose, denoted as $\mathbf { T } ^ { * }$ , for the query image.

## IV. EXPERIMENTS

We evaluated our method against several baselines for visual localization using either NeRF or 3DGS scene representations, as summarized in Table I. We selected piNeRF [11] and iComMa [5] as representative open-source methods based on photometric loss minimization using NeRF and 3DGS, respectively. We also compared our method to that of Chen et al. [10], which uses feature correspondence like we do but which uses NeRF instead of 3DGS as the scene representation. Since an open-source implementation of Chen et al. [10] was not available, the nature of our comparison in that case was different, and so is discussed separately in Section IV-C. All three methods address the same problem setting: estimating the camera pose from a single query image given a pre-constructed scene and an initial pose estimate, making them suitable for comparison. We also did experiments to assess the value of using 6DGS [12], an endto-end method of visual localization with respect to 3DGS that does not require initial pose estimates, as a source of such estimates for other methods in Section IV-D.

The evaluation was conducted on open-source datasets commonly used for scene reconstruction and localization with NeRF and 3DGS: Synthetic NeRF [3], Mip-NeRF360 [14], and Tanks and Temples [15]. These datasets contain hundreds of images captured in diverse indoor and outdoor scenes from varying viewpoints, either synthetic or realistic, with each image labeled with ground-truth 3D camera poses.

We chose commonly used evaluation metrics: mean rotation error (RE), mean translation error (TE), the percentage of results with $\mathrm { R E } ~ < ~ 5 ^ { \circ }$ and $\mathrm { T E } < 0 . 0 5$ , and the mean inference time per image. As scene scale varies across datasets, we normalized TE in each scene by setting the scale to the mean Euclidean distance of all camera poses from their centroid. Doing so is reasonable because all images in each scene were captured around a central object and is also reproducible since ground-truth camera poses are available for all datasets. This setup also enables fair comparisons of TE and of the condition $\mathrm { T E } < 0 . 0 5$ across scenes.

## A. Implementation details

1) Scene reconstruction: Visual localization requires a scene representation (e.g., 3DGS) as well as a query image. We used gsplat [16], an implementation of Gaussian splatting [2] that offers improved reconstruction efficiency and comparable rendering quality, while also supporting depth renderingâa feature not available in the original implementation. We used NerfBaselines [17], a framework for reconstructing and interfacing with gsplat [16]. We used the existing train and test splits from the Synthetic NeRF dataset for scene reconstruction and query images in visual localization. For the Mip-NeRF360 and Tanks and Temples datasets, we selected every eighth image as a query, using the remaining images for scene reconstruction.

All baseline methods also require a scene reconstruction, similar to ours. PiNeRF requires Instant-NGP [18], a NeRF variant with significantly improved reconstruction speed. iComMa and 6DGS require the original Gaussian splatting [2]. The reconstructions for all scenes, in each respective representation, were conducted in advance of the visual localization experiments.

2) Visual localization: We used SuperPoint [19] and SuperGlue [20], a learning-based feature point extractor and matcher, which shows robust performance compared to traditional methods $( \mathrm { e . g . }$ ., SIFT [21]), with comparable runtime when run on GPU. We used the pre-trained model trained on its indoor datasets and the default values for keypoint and match thresholds (0.005 and 0.2, respectively), without any additional parameter tuning. For pose estimation, we applied PnP with 50 iterations, followed by RANSAC with a a reprojection error threshold corresponding to approximately 1% of the image width. If the number of 2Dâ3D correspondences was insufficient for PnP to proceed, or if iterative PnP failed to converge, our method returned the initial pose estimate as the final result.

We used the default setups and hyperparameters for the baseline methodsâpiNeRF, iComMa, and 6DGSâwithout any additional tuning.

All experimentsâincluding scene reconstruction using either NeRF or 3DGS and the subsequent visual localizationâ were conducted on a system equipped with a 32-core 13th Gen Intel(R) Core(TM) i9-13900K CPU (up to 5.8 GHz) and a single NVIDIA GeForce RTX 4090 GPU.

## B. Comparison under randomized initial poses

First, we compare the results when the initial pose of the query image is generated by applying rotations and translations to the ground-truth pose, with rotation axes and translation directions sampled uniformly at random and magnitudes sampled from zero-mean normal distributions such that 95% of the samples fall within $\Delta \theta$ and $\Delta p ,$ respectively. Here, $\Delta \theta$ and $\Delta p$ were determined based on the cameraâs field of view, the radius of the camera trajectory in each scene, and the assumption that the centered object has a radius equal to half of the trajectory radius, such that the deviation results in the object leaving the image frame. This level of deviation reflects the typical roughness of initial pose estimates (e.g., from image retrieval, auxiliary sensors, or other methods), and aligns with the operating conditions of the methods being compared.

Table II shows the comparison between our method and the baseline methods. The aggregated results on the Synthetic NeRF, Mip-NeRF360, and Tanks and Temples datasets, are reported. Our method consistently achieves the best accuracy and efficiency across all datasets. Particularly, compared to iComMaâa 3DGS-based visual localization method by minimizing photometric loss between the query image and a rendered image at the estimated poseâour method shows up to about 12% improvement in $\mathrm { R E } < 5 ^ { \circ }$ and $\mathrm { T E } < 0 . 0 5$ on the Synthetic NeRF dataset, while also reducing inference time per image by more than two orders of magnitudeâ from more than 10 seconds to as fast as 0.1 seconds across all datasets.

TABLE II: COMPARISON OF VISUAL LOCALIZATION METHODS UNDER RANDOMIZED INITIAL POSES
<table><tr><td rowspan="2"></td><td colspan="3">Synthetic NeRF</td><td colspan="3">Mip-NeRF360</td><td colspan="3">Tanks and Temples</td></tr><tr><td>piNeRF</td><td>iComMa</td><td>Ours</td><td>piNeRF</td><td>iComMa</td><td>Ours</td><td>piNeRF</td><td>iComMa</td><td>Ours</td></tr><tr><td>RE ()</td><td>3.08</td><td>13.29</td><td>1.61</td><td>14.36</td><td>2.67</td><td>1.63</td><td>21.27</td><td>31.71</td><td>11.10</td></tr><tr><td>TE (unitless)</td><td>0.05</td><td>0.20</td><td>0.03</td><td>0.25</td><td>0.04</td><td>0.04</td><td>0.27</td><td>0.38</td><td>0.20</td></tr><tr><td> $\mathrm { R E } < 5 ^ { \circ }$  TE &lt; 0.05 (%)</td><td>71.88</td><td>79.00</td><td>90.94</td><td>1.22</td><td>86.99</td><td>90.65</td><td>6.89</td><td>32.45</td><td>42.42</td></tr><tr><td>Time / Image (s)</td><td>13.56</td><td>16.74</td><td>0.09</td><td>13.46</td><td>27.78</td><td>0.15</td><td>14.59</td><td>41.79</td><td>0.10</td></tr></table>

\* Synthetic NeRF (8 scenes, 1600 test images), Mip-NeRF360 (9 scenes, 221 test images), and Tanks and Temples (21 scenes, 943 test images); TE is normalized by the trajectory radius for each scene.

TABLE III: COMPARISON OF OUR METHOD WITH CHEN et al. [10] ON SYNTHETIC NERF DATASET
<table><tr><td></td><td>Chen (Full)</td><td>Chen (Lite)</td><td>Ours (Avg.)</td><td>Ours (Best)</td></tr><tr><td>RE ()</td><td>1.25</td><td>1.57</td><td>4.77</td><td>0.19</td></tr><tr><td>TE (unitless)</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.00</td></tr><tr><td> $\mathrm { T E } < 0 . 0 5 ~ ( \% )$ </td><td>95.00</td><td>94.50</td><td>87.78</td><td>99.69</td></tr><tr><td> $\mathrm { R E } < 5 ^ { \circ } ~ ( \% )$ </td><td>88.00</td><td>75.00</td><td>96.71</td><td>99.81</td></tr></table>

\* Chen (Full): Chen et al.âs full pipeline; Chen (Lite): Without post-pose refinement; Ours (Avg.): Our method, averaging the results of 5 random initial poses per test image; Ours (Best): Our method, reporting the best result among 5 random initial poses per test image.

## C. Comparison with a method based on feature correspondence using NeRF

We provide a comparison with Chen et al. [10], a method based on feature correspondence using NeRF as a scene representationâpart of which our method usesâin this separate subsection, as their open-source implementation is not available and their experimental setup does not align with that in Section IV-B. We use the exact numbers reported in their paper, evaluated on the Synthetic NeRF dataset, with only the translation error normalized by scene scale to enable a fair comparison with ours (Table III).

Chen (Full) refers to the results from their full pipeline, while Chen (Lite) uses the same setup but without post-pose refinement. Chen reports results using only five randomly selected test images per scene (out of 200), each evaluated under five random initial poses, generated by applying a translation uniformly sampled between 0 and 0.2 along a random direction, followed by a rotation with a magnitude uniformly sampled between 10â¦ and $4 0 ^ { \circ }$ around a random axis. However, it is not clearly stated whether their reported results show an average over the five initial poses or the best among them.

To ensure a fair comparison, we used the same protocol for generating initial poses as described by Chen. Due to the ambiguity regarding whether their results show an average or the best case, we report both: the results averaged over five random initial poses per test image (Ours (Avg.)) and the best results among the five in rotation error (Ours (Best)). Additionally, while Chen evaluates only five test images per scene, our method is evaluated on all 200 test images per scene, totaling 1,600 evaluations across all scenes.

## D. Comparison under initial poses provided by 6DGS

Next, we compare the results when the initial pose of the query image is provided by 6DGS [12]. This setup is reasonable, as 6DGS can estimate rough poses (e.g., approximately 20â¦ in rotation error and 0.2-0.5 in normalized translation error), and can therefore serve as a source of initial poses for the methods we compare.

Table IV shows comparisons between our method and baseline methods on the three datasets. The original 6DGS resultsâwhich do not require any initial pose but do require a model pretrained on each scene as a prior stepâ are included as a reference in the leftmost subcolumn of each dataset column. Our method consistently achieves the best accuracy and efficiency across all datasets. Particularly, compared to iComMa, our method shows over a 40% improvement in $\mathrm { R E } ~ < ~ 5 ^ { \circ }$ and $\mathrm { T E } ~ < ~ 0 . 0 5$ on the Tanks and Temples dataset, while also reducing inference time per image by more than two orders of magnitudeâfrom more than 10 seconds to as fast as 0.1 seconds across all datasets.

Figure 3 shows examples of both failure and success cases from our method on several scenes from the Tanks and Temples dataset, where success is defined as $\mathrm { R E } < 5 ^ { \circ }$ and $\mathrm { T E } < 0 . 0 5$ . Each triplet shows, from left to right, the query image, the image rendered at the initial pose provided by 6DGS, and the image rendered at the pose estimated by our method. Failures (Fig. 3(a)) occur either due to inaccurate pose estimation despite feature matching (top two rows), or due to PnP failure caused by insufficient feature matches between the query and the rendered imageâresulting in the initial pose being returned (bottom two rows). Nonetheless, even within the same scene, our method shows success cases as well (Fig. 3(b)) despite drastic appearance differences between the query and the rendered image.

## E. Sensitivity to error in the initial pose estimate

We also study the extent to which our method can reliably estimate the pose of a query image despite error in the initial pose estimate. We report the percentage of results with rotation and translation (normalized by scene scale) errors below threshold (i.e., $\mathrm { R E } < 5 ^ { \circ }$ , $\mathrm { T E } < 0 . 0 5 )$ as a function of error in the initial pose estimate. Specifically, we examine cases where the initial rotation error is between 0 and 1âÎ¸ in yaw, and the initial translation error is between 0 and $\Delta p$ along x-axis. Here, âÎ¸ and $\Delta p$ were determined based on the cameraâs field of view, the radius of the camera trajectory in each scene, and the assumption that the centered object has a radius equal to half of the trajectory radius, such that the deviation would result in the object leaving the image frame.

TABLE IV: COMPARISON OF VISUAL LOCALIZATION METHODS UNDER INITIAL POSES PROVIDED BY 6DGS
<table><tr><td rowspan="2"></td><td colspan="4">Synthetic NeRF</td><td colspan="4">Mip-NeRF360</td><td colspan="4">Tanks and Temples</td></tr><tr><td>6DGS</td><td>piNeRF</td><td>iComMa</td><td>Ours</td><td>6DGS</td><td>piNeRF</td><td>iComMa</td><td>Ours</td><td>6DGS</td><td>piNeRF</td><td>iComMa</td><td>Ours</td></tr><tr><td>RE ()</td><td>17.93</td><td>23.69</td><td>35.61</td><td>14.09</td><td>20.84</td><td>17.04</td><td>9.60</td><td>6.47</td><td>17.93</td><td>27.96</td><td>35.69</td><td>11.21</td></tr><tr><td>TE (unitless)</td><td>0.59</td><td>0.52</td><td>0.51</td><td>0.27</td><td>0.22</td><td>0.28</td><td>0.11</td><td>0.05</td><td>0.59</td><td>0.23</td><td>0.39</td><td>0.08</td></tr><tr><td> $\mathrm { R E } < 5 ^ { \circ } , \ \mathrm { T E } < 0 . 0 5 \ ( \% )$ </td><td>0.06</td><td>10.44</td><td>44.25</td><td>61.25</td><td>0.41</td><td>2.44</td><td>78.46</td><td>90.24</td><td>0.06</td><td>8.91</td><td>31.60</td><td>72.53</td></tr><tr><td>Time / Image (s)</td><td>0.04</td><td>14.62</td><td>29.92</td><td>0.09</td><td>0.02</td><td>14.18</td><td>30.65</td><td>0.15</td><td>0.04</td><td>14.88</td><td>45.68</td><td>0.11</td></tr></table>

\* Synthetic NeRF (8 scenes, 1600 test images), Mip-NeRF360 (9 scenes, 221 test images), and Tanks and Temples (21 scenes, 943 test images); TE is normalized by the trajectory radius for each scene; 6DGS results (which do not require an initial pose) are included for reference.

<!-- image-->  
(a) Failure $( \mathrm { R E } \geq 5 ^ { \circ }$ or TE â¥ 0.05)

<!-- image-->  
(b) Success $( \mathrm { R E } < 5 ^ { \circ }$ and $\mathrm { T E } < 0 . 0 5 )$

Fig. 3: Example pairs of failure and success cases from several scenes in the Tanks and Temples dataset, where success is defined as $\mathrm { R E } < 5 ^ { \circ }$ and $\mathrm { T E } < 0 . 0 5$ . Each triplet shows, from left to right, the query image, the image rendered at the initial pose provided by 6DGS, and the image rendered at the pose estimated by our method.  
<!-- image-->  
Rotation Perturbation (1ÎÎ¸ = 33.84 Â°)  
Fig. 4: Percentage of pose estimates (out of 200 test images) with rotation error $< ~ 5 ^ { \circ }$ and normalized translation error $< 0 . 0 5$ as a function of difference between the initial pose estimate and ground-truth in yaw on the Lego scene (unit: %). $\Delta \theta = 3 3 . 8 4 ^ { \circ }$ is determined as the minimum yaw error required for the object to move out of view.

<!-- image-->  
Translation Perturbation (1Îp = 2.44 m)  
Fig. 5: Percentage of pose estimates (out of 200 test images) with rotation error $< ~ 5 ^ { \circ }$ and normalized translation error $< 0 . 0 5$ as a function of difference between the initial pose and ground-truth in x-position on the Lego scene (unit: %). $\Delta p = 2 . 4 4$ m is determined as the minimum translation error required for the object to move out of view.

TABLE V: COMPARISON OF FEATURE EXTRACTOR AND MATCHER CHOICES ON THREE DATASETS
<table><tr><td colspan="2"></td><td>RE</td><td>TE</td><td>Acc</td><td>Time</td></tr><tr><td rowspan="3">Synthetic NeRF</td><td>SIFT</td><td>7.04</td><td>0.37</td><td>66.94</td><td>0.14</td></tr><tr><td>SP+SG</td><td>1.61</td><td>0.03</td><td>90.94</td><td>0.09</td></tr><tr><td>LoFTR</td><td>1.84</td><td>0.03</td><td>91.94</td><td>0.13</td></tr><tr><td rowspan="3">Mip-NeRF360</td><td>SIFT</td><td>1.44</td><td>0.05</td><td>87.80</td><td>0.21</td></tr><tr><td>SP+SG</td><td>1.63</td><td>0.04</td><td>90.65</td><td>0.15</td></tr><tr><td>LoFTR</td><td>1.47</td><td>0.04</td><td>90.24</td><td>0.18</td></tr><tr><td rowspan="3">Tanks and Temples</td><td>SIFT</td><td>11.85</td><td>0.59</td><td>34.47</td><td>0.14</td></tr><tr><td>SP+SG</td><td>11.10</td><td>0.20</td><td>42.42</td><td>0.10</td></tr><tr><td>LoFTR</td><td>12.27</td><td>0.21</td><td>41.68</td><td>0.13</td></tr></table>

$\ast R E \cdot$ rotation error $( ^ { \circ } ) ;$ TE: translation error (unitless); Acc: percentage of results with $R E < 5 ^ { \circ }$ and $T E < 0 . 0 5 ;$ Time: time per image (s). \*\* TE is normalized by the trajectory radius for each scene.

Figures 4 and 5 show results for the Lego scene in the Synthetic NeRF dataset, where âÎ¸ and $\Delta p$ are 33.84â¦ and 2.44 m, respectively. Our method achieves the best performance up to 1.0âÎ¸ in rotation and 0.8âp in translation among the three methods.

## F. Choice of feature point extractor and matcher

Lastly, we study how the choice of feature extractor and matcher influences the performance of the proposed approach. We compare SIFT [21], a commonly used traditional method; SuperPoint [19] and SuperGlue [20] (SP+SG), a learning-based method used as our default; and LoFTR [22], another newer learning-based method that has been reported to outperform SP+SG. For SIFT, we applied Loweâs ratio test [21] with a threshold of 0.7 to filter out outlier matches. For LoFTR, we used its official pre-trained model trained on the outdoor dataset, without any additional parameter tuning.

Table V shows the results of how the choice of feature extractor and matcher affects performance, aggregated over each of the three datasets. The initial pose estimate for each query image was generated by applying random rotations and translations to the ground-truth pose, with magnitudes sampled from normal distributions such that 95% of the samples fall within âÎ¸ and $\Delta p ,$ where $\Delta \theta$ and $\Delta p$ are scenespecific constants. Rotation axes and translation directions were sampled uniformly at random. These initial conditions are the same as those used in Section IV-B. The results show that learning-based methodsâSP+SG and LoFTRâ consistently outperform SIFT in terms of TE, RE, and the percentage of results with $\mathrm { R E } ~ < ~ 5 ^ { \circ }$ and $\mathrm { T E } ~ < ~ 0 . 0 5$ achieving improvements of over 20% on the Synthetic NeRF dataset. This is obtained with on par or even better inference time per image. Between SP+SG and LoFTR, the error metrics are comparable, but SP+SG consistently shows lower inference time. Based on this, we choose SP+SG as the default feature extractor and matcher in our pipeline.

## V. CONCLUSION

In this paper, we proposed a visual localization approach using 3DGS as a scene representation. We showed that our method outperforms existing approaches that estimate camera poses by minimizing the photometric difference between the query image and the rendered image at the estimated pose, either using NeRF and 3DGS as a scene representation. Our method yielded an improvement of more than two orders of magnitude in inference time along with significant gains in accuracy and better tolerance of errors in the initial pose estimate.

## A. Limitations of our method and of baseline methods

The performance of our method and of the baseline methods depends on the quality of the reconstructed scene, whether represented by 3DGS or NeRF. High-quality reconstruction yields better rendered images, which are essential for methods using feature correspondence or photometric loss minimization. Additionally, our method depends on the accuracy of rendered depth maps for correct PnP results, which, again, relies on scene quality.

## B. Future work

One direction for future work is to improve the robustness of our method. For instance, a failure mode may occur when the number of feature correspondences is too low (e.g., fewer than 10) to yield a reliable pose estimate. To address this, one could render an intermediate image from the pose estimate and perform feature matching between this image and the query image to obtain the final pose. This additional step may help eliminate such failure cases by increasing the number of valid correspondences between the rendered image at the intermediate pose and the query image, at the cost of a few extra iterations and slightly increased inference time.

Another direction is to integrate our method into a broader robotics pipeline. For example, most visual SLAM systems using 3DGS representations (e.g., Gaussian Splatting SLAM [23]) rely on photometric loss minimization to track incoming frames. While this approach works well when frame-to-frame motion is small, it may fail when the motion becomes large. Our method could serve as an alternative module in such scenarios, helping to accurately localize incoming frames that undergo abrupt motion. This role is analogous to relocalization in conventional visual SLAM systems (e.g., ORB-SLAM3 [24]) when tracking is lost.

A third possible direction is to apply our method to realworld robotics tasks. Since our method (as well as the baselines) is evaluated on scenes captured around a central object, one could envision applicationsâpresumably using the same pipeline with little or no modificationâwhere a robot localizes itself while observing the object, assuming a pre-constructed 3DGS scene of that object is available.

## REFERENCES

[1] J. Miao, K. Jiang, T. Wen, Y. Wang, P. Jia, B. Wijaya, X. Zhao, Q. Cheng, Z. Xiao, J. Huang et al., âA survey on monocular relocalization: From the perspective of scene map representation,â IEEE Transactions on Intelligent Vehicles, 2024.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[3] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[4] S. Zhu, G. Wang, D. Kong, and H. Wang, â3d gaussian splatting in robotics: A survey,â arXiv preprint arXiv:2410.12262, 2024.

[5] Y. Sun, X. Wang, Y. Zhang, J. Zhang, C. Jiang, Y. Guo, and F. Wang, âicomma: Inverting 3d gaussians splatting for camera pose estimation via comparing and matching,â arXiv preprint arXiv:2312.09031, 2023.

[6] P. Jiang, G. Pandey, and S. Saripalli, â3dgs-reloc: 3d gaussian splatting for map representation and visual relocalization,â arXiv preprint arXiv:2403.11367, 2024.

[7] K. Botashev, V. Pyatov, G. Ferrer, and S. Lefkimmiatis, âGsloc: Visual localization with 3d gaussian splatting,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 5664â5671.

[8] H. Jun, H. Yu, and S. Oh, âRenderable street view map-based localization: Leveraging 3d gaussian splatting for street-level positioning,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 5635â5640.

[9] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-Y. Lin, âinerf: Inverting neural radiance fields for pose estimation,â in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 1323â1330.

[10] R. Chen, Y. Cong, and Y. Ren, âMarrying nerf with feature matching for one-step pose estimation,â in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 7302â7309.

[11] Y. Lin, T. Muller, J. Tremblay, B. Wen, S. Tyree, A. Evans, P. A. Â¨ Vela, and S. Birchfield, âParallel inversion of neural radiance fields for robust pose estimation,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 9377â9384.

[12] B. Matteo, T. Tsesmelis, S. James, F. Poiesi, and A. Del Bue, â6dgs: 6d pose estimation from a single image and a 3d gaussian splatting model,â in European Conference on Computer Vision. Springer, 2024, pp. 420â436.

[13] K. Madsen, H. B. Nielsen, and O. Tingleff, âMethods for non-linear least squares problems,â 2004.

[14] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5470â5479.

[15] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, âTanks and temples: Benchmarking large-scale scene reconstruction,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1â13, 2017.

[16] V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye, J. Hu, M. Tancik et al., âgsplat: An open-source library for gaussian splatting,â Journal of Machine Learning Research, vol. 26, no. 34, pp. 1â17, 2025.

[17] J. Kulhanek and T. Sattler, âNerfbaselines: Consistent and reproducible evaluation of novel view synthesis methods,â arXiv preprint arXiv:2406.17345, 2024.

[18] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural Â¨ graphics primitives with a multiresolution hash encoding,â ACM Trans. Graph., vol. 41, no. 4, pp. 102:1â102:15, Jul. 2022. [Online]. Available: https://doi.org/10.1145/3528223.3530127

[19] D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperpoint: Selfsupervised interest point detection and description,â in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 224â236.

[20] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperglue: Learning feature matching with graph neural networks,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4938â4947.

[21] D. G. Lowe, âDistinctive image features from scale-invariant keypoints,â International journal of computer vision, vol. 60, pp. 91â110, 2004.

[22] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, âLoftr: Detectorfree local feature matching with transformers,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 8922â8931.

[23] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[24] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âOrb-slam3: An accurate open-source library for visual, Â´ visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.