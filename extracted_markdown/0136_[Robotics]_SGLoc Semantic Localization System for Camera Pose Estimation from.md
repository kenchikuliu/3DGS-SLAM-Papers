# SGLoc: Semantic Localization System for Camera Pose Estimation from 3D Gaussian Splatting Representation

Beining Xuâ 1, Siting Zhuâ 1, Hesheng Wang1

Abstractâ We propose SGLoc, a novel localization system that directly regresses camera poses from 3D Gaussian Splatting (3DGS) representation by leveraging semantic information. Our method utilizes the semantic relationship between 2D image and 3D scene representation to estimate the 6DoF pose without prior pose information. In this system, we introduce a multilevel pose regression strategy that progressively estimates and refines the pose of query image from the global 3DGS map, without requiring initial pose priors. Moreover, we introduce a semantic-based global retrieval algorithm that establishes correspondences between 2D (image) and 3D (3DGS map). By matching the extracted scene semantic descriptors of 2D query image and 3DGS semantic representation, we align the image with the local region of the global 3DGS map, thereby obtaining a coarse pose estimation. Subsequently, we refine the coarse pose by iteratively optimizing the difference between the query image and the rendered image from 3DGS. Our SGLoc demonstrates superior performance over baselines on 12scenes and 7scenes datasets, showing excellent capabilities in global localization without initial pose prior. Code will be available at https://github.com/IRMVLab/SGLoc.

## I. INTRODUCTION

Visual localization is a fundamental challenge in autonomous driving [1], [2] and robotics [3]. It enables estimation of 6DoF camera poses within a previously mapped environment. Existing traditional localization systems can be categorized into feature-based and regression-based methods. Feature-based methods typically extract 2D and 3D keypoints, then match 2D keypoints from query images with 3D keypoints of the scene [4]â[6] to regress the camera pose. Regression-based methods employ neural networks to extract image features and encode absolute poses or scene coordinates for direct 6DoF pose regression [7]â[9]. These methods rely on low-level visual features, such as textural and geometric features. However, low-level visual features are inherently sensitive to environmental variations, particularly in scenes with insufficient texture information or varying lighting conditions, which leads to decreased localization accuracy.

3D Gaussian Splatting (3DGS) [10] emerges as a promising scene representation. As 3DGS has demonstrated its effectiveness in scene modeling for robotics tasks [11], [12], enabling direct pose estimation from 3DGS maps becomes crucial. Existing works leverage the high-quality novel view synthesis capability of 3DGS representation to achieve visual localization from 3DGS maps. Among these approaches, [13] leverages the rendering process of 3DGS for pose estimation. However, these methods struggle when given poor pose priors, such as significant rotations and translations, leading to substantial discrepancies between the rendering and query views. Such discrepancies result in degraded accuracy of the pose regression. [14]â[16] directly follow the approach of traditional feature-based localization, where keypoints are extracted and matched between 3DGS maps and query images to regress poses. Consequently, these methods inherit the limitations of traditional localization approaches discussed above. Furthermore, existing methods overlook the consistency of semantic information between 2D query image and 3D scene representation, resulting in degraded localization performance in complex scenes.

To address these challenges, we propose a novel semanticbased visual localization framework. Our method introduces a multi-level pose regression strategy that integrates semantic-based global retrieval with rendering-based optimization, which enables precise localization of a query RGB image without requiring initial poses. We utilize semantic information to compensate for the inherent shortcomings of traditional feature-based methods. Specifically, we leverage semantic consistency to directly retrieve the closest match to query image from 3DGS map, thereby obtaining a coarse pose estimation. This strategy enables more reliable initial pose estimates for further pose refinement, even in scenes with sparse textural features. Subsequently, we iteratively optimize the initial pose by comparing rendered images and query images, achieving accurate localization results.

Overall, we provide the following contributions:

â¢ We present SGLoc, a novel semantic-based localization system that directly regresses camera poses from 3D Gaussian Splatting. We introduce a multi-level pose regression strategy that progressively estimates and refines the pose of 2D query image based on the global 3DGS map, without requiring initial pose priors.

â¢ We employ a semantic-based global retrieval method to establish correspondence between 2D query image and 3DGS semantic representation, thereby obtaining pose estimation of image.

â¢ Extensive evaluations are conducted on 12Scenes and 7Scenes datasets, to demonstrate the effectiveness of our method in localization performance.

## II. RELATED WORK

## A. Traditional Localization

Classical localization methods include feature-based methods and regression-based methods. Feature-based methods typically focus on matching keypoints from 2D images and 3D models, then apply Perspective-n-Point (PnP) [17] algorithm with RANSAC [18] for pose estimation [19]â[23]. But these methods are easily affected by noise. Regressionbased methods employ neural networks to extract image features and encode camera poses or scene coordinates for 6DoF pose regression [7], [24]â[27]. Although regressionbased methods are faster, they are not superior in accuracy and generalization. Nevertheless, their lack of geometric constraints leads to lower accuracy compared to featurebased methods.

However, due to the geometric ambiguities between 2D and 3D representation, few studies have focused on direct 2D-3D matching. Given the semantic consistency between 3D and 2D representations, we propose a semantic-based 2D image-to-3DGS map matching method. By aligning the semantic features of query images with known 3DGS map, this method provides reliable initial pose estimation. Our method fully exploits semantic consistency between images and 3D scenes, which significantly enhances the robustness of localization in complex scenarios.

## B. NeRF-based Localization

Neural Radiance Fields [28] has been utilized for localization tasks for its ability to synthesize novel view images. iNerf [29] introduces an inverse NeRF method to estimate camera poses. NeFeS [30] optimizes differences between rendered images and query images to obtain poses. Most approaches [31]â[34] follow traditional feature-based localization methods to match 2D and 3D features. PNeRFLoc [31] introduces warping loss to improve pose estimation. NeRFMatch [33] achieves 2D-3D matches with specialized feature extractors. However, those NeRF-based methods all suffer from poor rendering quality and extensive rendering time.

## C. 3DGS-based Localization

3D Gaussian Splatting [10] achieves high quality and real-time novel-view synthesis of the 3D scenes and has recently been employed for visual localization tasks. Some approaches design the pose estimation framework by combining the rendering process of 3DGS. iComMa [13] designs a gradient-based differentiable framework to adopt iterative optimization for camera pose regression. 6DGS [35] avoids the iterative process by inverting the 3DGS rendering process for direct 6-DoF pose estimation. However, both of them struggle when given poor initial poses, like large rotations and translations. Most 3DGS-based localization methods follow the classical feature-based visual localization framework. In particular, SplatLoc [15] uses minimal parameters to achieve localization with high-quality rendering. GSLoc [14] establishes 2D-3D correspondences via rendered RGB images and depth maps, enabling localization without training feature descriptors. GSplatLoc [16] aligns rendered images with query images by extracting features via XFeat [21] for 2D-3D matching during optimization iterations. However, like traditional feature-based methods, these methods still suffer from performance degradation in scenes with insufficient texture and structure information.

Our goal is to design a localization method capable of regressing camera poses from arbitrary query images without prior pose. We introduce a multi-level framework that progressively estimates and refines the pose of query image from the global 3DGS map. Considering the semantic consistency between 2D query image and 3DGS semantic representation, we propose a semantic-based 2D image-to-3DGS matching method. By matching the query image with pre-built 3DGS map, our method provides coarse initial pose estimations. Subsequently, we refine the initial pose via iterative rendering optimization, leveraging the novel view synthesis capability of 3DGS representation.

## III. METHOD

The overview of our method is shown in Fig. 1. We adopt a semantic 3DGS representation [39] to obtain 3DGS global map G. As the query image typically corresponds to a local 3D region rather than the entire scene, we divide the 3DGS map into submaps $G = \left\{ G _ { i } : i \in { 1 , \dots , N } \right\}$ . Given a query image $I _ { q } ,$ we first perform semantic segmentation and extract semantic descriptors from both the image and the 3DGS submaps. We define the ground truth camera pose of $I _ { q }$ as $P = \left[ T \mid R \right]$ , where $T \in \mathbb { R } ^ { 3 }$ is the translation vector and $R \in { \mathrm { S O } } ( 3 )$ is the rotation matrix. Then, to provide a reliable coarse initial pose $P ^ { * } = \left[ T ^ { * } \mid R ^ { * } \right]$ for pose refinement, we align the 3DGS submaps and the query image at the scene level by matching the semantic descriptors $F _ { I }$ of 2D query image and $F _ { G }$ of 3DGS representation. Finally, the coarse pose is further refined by comparing the query image and rendered image $I _ { r }$ from 3DGS representation, resulting in the final estimated pose ${ \hat { P } } = [ { \hat { T } } \mid { \hat { R } } ]$ . Sec. III-A describes our multi-level localization framework. Sec. III-B presents our semantic-based global place retrieval. Sec. III-C introduces details of rendering-based pose refinement.

## A. Multi-Level Localization Pipline

We obtain the pose of query image from 3DGS global map in a coarse-to-fine manner. In the coarse stage, we perform 2D-3D global place retrieval by aligning 2D and 3D scene semantic descriptors into a shared feature space, enabling direct similarity measurement. Through matching 2D and 3D scene semantic descriptors, we retrieve the top-k most similar 3D descriptors corresponding to the query image, which provides k initial poses for downstream optimization. In the fine stage, we perform rendering-based pose estimation to refine the coarse initial pose.

Coarse Stage. Following the retrieval-based localization strategies introduced in UniLoc [40], we adapt it to a semantic-guided retrieval framework between 2D images and 3DGS representation. To establish instance-level correspondences, we first perform semantic segmentation on both 2D query image and 3DGS representation. Each 3D submap and 2D image contains multiple object instances $G _ { i } = \left\{ g _ { j } ^ { i } \right.$ $j \in { 1 , \dots , n } \} , I = \{ p _ { j } : j \in { 1 , \dots , m } \}$ . The correspondence problem between query image and 3DGS representation is formulated as a retrieval task. Considering the semantic relationship between 2D query image and 3D scene representation, we extract scene semantic descriptors from the query image and the 3DGS submaps. Then, we map the semantic features into a shared feature space through contrastive learning. Moreover, by calculating the similarity scores between scene semantic descriptors of query image and the 3DGS submaps, we identify the top-k submaps that exhibit the highest similarity scores. The poses corresponding to the top-k candidate submaps are selected as coarse initial poses for subsequent pose refinement. Besides, we filter out obvious retrieval errors before pose refinement by calculating the similarity between the query image and rendered images that are generated from the initial coarse poses.

<!-- image-->  
Fig. 1. An overview of SGLoc. Our method takes a query image and 3DGS global map as input. We perform semantic segmentation on both query image and 3DGS representation. 2D and 3D instances are fed into CLIP model [36] and PointNet++ [37] to obtain semantic features respectively. Instance encoders are utilized to encode 2D and 3D instancesâ color, size, and position information. All features are aggregated as scene semantic descriptors through multi-head attention [38] with FFN layer. The semantic-based global retrieval model is guided by contrastive loss to align the 2D and 3D scene semantic descriptors. The top-k submaps are selected by cosine similarity, and corresponding poses are selected as coarse initial poses for pose refinement. Pose is refined through iterative optimization of matching loss between the rendered image and query image.

Fine Stage. Benefitting from the high-quality rendering capability of 3DGS representation [10], we leverage the initial coarse poses provided in the first stage to optimize the differences between the query image and the rendered image, obtaining precise pose estimation.

## B. Semantic-based Global Place Retrieval

Feature Extraction. For RGB images, we utilize SAM [41] to segment them into instance-level masks. For each segmented instance, we crop the corresponding RGB region based on its mask to obtain an instance-level RGB image.

Then, cropped instance images are fed into the CLIP model to extract semantic features $f _ { \mathrm { C L I P } } \in \mathbb { R } ^ { B \times N \times d _ { c } }$ , where B denotes the batch size, N is the number of instances, and $d _ { c }$ is the embedding dimension of CLIP model. fCLIP is projected into a unified latent space via 3-layer MLP. We utilize instance encoder to extract additional instance features. Specifically, for every instance, we encode the average color $\bar { \in \mathbb { R } ^ { 3 } }$ , normalized instance size $\in \mathbb { R } .$ , and relative position of each instance in UV coordinates $\in \mathbb { R } ^ { 2 }$ through different MLP $\mathcal { F } _ { c } ^ { I } , \mathcal { F } _ { s } ^ { I } , \mathcal { F } _ { p } ^ { I }$ . Then, we concatenate all features and pass through another three-layer MLP to obtain the feature descriptor $\mathbf { \bar { \rho } } _ { f _ { I } } \in \mathbb { R } ^ { B \times N \times d }$ for each 2D instance.

For 3DGS submaps, we use 3DGS representation proposed by Gaussian Grouping [39] to generate our 3DGS global map. [39] incorporates new identity encoding parameters to each Gaussian primitive, enabling semantic Gaussian representation. To extract instance-level 3D features, we employ a pre-trained PointNet++ [37] to process the point cloud of each object instance and obtain a semantic embedding $f _ { \mathrm { P N } } \in \mathbb { R } ^ { B \times \mathbf { \check { N } } \times d }$ The semantic feature is projected into a unified latent space via a learnable 3-layer MLP:

$$
f _ { \mathrm { g e o } } = \mathcal { F } _ { \mathrm { P N } } ^ { G } ( f _ { \mathrm { P N } } ) \in \mathbb { R } ^ { d }
$$

Since each 3D Gaussian primitive in the 3DGS model contains both coordinate and color information, we use different MLP $\mathcal { F } _ { c } ^ { G } , \mathcal { F } _ { s } ^ { G } , \mathcal { F } _ { p } ^ { G }$ to encode the average color $\in \mathbb { R } ^ { 3 }$ , the number of 3D Gaussian primitives $\in \mathbb { R }$ , and the relative position of each instance projected into the camera coordinate $\in \mathbb { R } ^ { 3 }$

All features are integrated through concatenation followed by a three-layer MLP to obtain the feature descriptor $f _ { G } \in$ $\check { \mathbb R } ^ { B \times N \times d }$

Feature Aggregation. To establish correspondence between 2D images and 3DGS map, we aggregate instancelevel features fI, fG into scene semantic descriptors $F _ { I } ,$ $F _ { G }$ and then align the scene semantic descriptors from 2D images and 3DGS submaps. Specifically, to interact with different instance features effectively and assign attention weights to them adaptively, we employ a Multi-Head Self-Attention mechanism [38] (Attr) with a feed-forward neural network (FFN) for feature aggregation. Attr and the FFN layer take both query (Q), key (K), and value (V) as input. Taking image features as an example, query (Q), key (K) and value (V) are all derived from instance features $f _ { I }$

$$
\begin{array} { r } { \hat { f } _ { I } = Q + A t t r ( Q , K , V ) , } \\ { \hat { F } _ { I } = \hat { f } _ { I } + F F N ( \hat { f } _ { I } ) , } \\ { W = s o f t m a x ( \mathcal { F } ( \hat { F } _ { I } ) ) . } \end{array}\tag{1}
$$

Subsequently, taking $\hat { f } _ { I }$ as input, we generate attention weights $\bar { W } \in \bar { \mathbb { R } } ^ { B \times N }$ through a three-layer MLP followed by softmax layer. These attention weights are utilized to aggregate instance descriptors into a scene semantic descriptor:

$$
F _ { \widehat { I } } = \sum _ { i = 1 } ^ { N } \widehat { F _ { I _ { i } } } \times W _ { i } .\tag{2}
$$

Here, $W _ { i }$ and $\hat { F } _ { I _ { i } }$ denote the attention weight and instance feature corresponding to the i-th instance, respectively.

Then, we use cosine similarity to match the 2D and 3D scene semantic descriptors. We select the top-k submaps as the result of place retrieval. And the poses corresponding to the top-k submaps are selected as coarse initial poses for pose refinement.

Since rendered images generated by poses with significant translation and rotation errors diverge substantially from the query image, we employ the Peak Signal-to-Noise Ratio (PSNR) [42] as the similarity metric to filter out mismatches. PSNR is denoted by the following formula:

$$
\begin{array} { r l r } & { } & { \mathrm { P S N R } = 1 0 \cdot \log _ { 1 0 } \left( \frac { \mathbf { M A X } _ { I } ^ { 2 } } { \mathbf { M S E } } \right) , } \\ & { } & { \mathrm { M S E } = \displaystyle \frac { 1 } { h w } \sum _ { i = 0 } ^ { h - 1 } \sum _ { j = 0 } ^ { w - 1 } \left[ I ( i , j ) - I _ { r } ( i , j ) \right] ^ { 2 } , } \end{array}\tag{3}
$$

where h and w represent the height and weight of the image, Ir and I represent the rendered image and query image. $\mathrm { M A X } _ { I } ^ { 2 }$ is the maximum possible pixel value of the image. If PSNR values are below a predefined threshold $\varepsilon = 5 5$ we will discard the corresponding initial pose. The filtered coarse initial pose is denoted as $P _ { i } ^ { * } = [ T _ { i } ^ { * } | R _ { i } ^ { * } ]$

Loss Functions. We utilize the contrastive learning loss [43] to align scene semantic descriptors from 3D representation and 2D images. For the i-th image and 3D submap pair $( I _ { i } , G _ { i } )$ , the contrastive loss function can be calculated

TABLE I  
ACCURACY COMPARISON ON 12SCENES DATASET FOR MEDIAN TRANSLATION AND ROTATION ERRORS (CM/Â°) METRICS.
<table><tr><td>Method</td><td colspan="2">Apartment 2 Bed Kitchen</td><td>Office 1 Lounge</td><td>Avg.â [cm/]</td></tr><tr><td>SCRNet [9]</td><td>3.3/1.5</td><td>2.1/1.0</td><td>2.7/0.9</td><td>2.7/1.1</td></tr><tr><td>SCRNet-ID [44]</td><td>2.0/0.8</td><td>1.8/0.9</td><td>3.4/1.1</td><td>2.4/0.9</td></tr><tr><td>NeRF-SCR [45]</td><td>1.6/0.7</td><td>1.2/0.5</td><td>1.8/0.6</td><td>1.5/0.6</td></tr><tr><td>PNeRFLoc [31]</td><td>1.2/0.5</td><td>0.8/0.4</td><td>2.3/0.8</td><td>1.5/0.6</td></tr><tr><td>SpaltLoc [15]</td><td>1.2/0.5</td><td>1.0/0.5</td><td>1.6/0.5</td><td>1.2/0.5</td></tr><tr><td>SGLoc (Ours)</td><td>0.5/0.4</td><td>01/0.1</td><td>0.3/0.1</td><td>0.3/0.2</td></tr></table>

using the following formula:

$$
\begin{array} { r } { l ( I _ { i } , G _ { i } ) = f ( I _ { i } , G _ { i } ) + f ( G _ { i } , I _ { i } ) , } \\ { f ( I _ { i } , G _ { i } ) = - \log \frac { \exp ( F _ { I _ { i } } \cdot F _ { G _ { i } } / \tau ) } { \sum _ { j \in N } \exp ( F _ { I _ { i } } \cdot F _ { G _ { j } } / \tau ) } , } \end{array}\tag{4}
$$

where $F _ { I _ { i } }$ and $F _ { G _ { i } }$ represent the image and 3D scene semantic descriptors respectively. Ï is the temperature parameter. N is the number of 3DGS submaps in the scene.

The batch loss is derived by averaging the contrastive loss terms.

## C. Rendering-based Pose Refinement

Given a coarse initial pose $P _ { i } ^ { * } = [ T _ { i } ^ { * } \ | \ R _ { i } ^ { * } ]$ , we adopt a training-free rendering-based method following [13] to refine pose. At each optimization step, the image is rendered from the current camera pose. Subsequently, the errors between the rendered and query images are calculated, and the camera pose is iteratively refined through gradient-based optimization to minimize this error. The problem is formulated as follows:

$$
\hat { P } = \arg \operatorname* { m i n } \mathcal { L } ( I _ { q } , I _ { r } | p )\tag{5}
$$

where $I _ { r }$ is the render image generated by the initial pose $P ^ { * }$ $\hat { P }$ presents the predicted pose. We optimize the camera poses by gradient descent. $\mathcal { L }$ is the loss function defined as [13], including pixel-level loss ${ \mathcal L } _ { \mathrm { p i x e l } }$ and matching loss ${ \mathcal { L } } _ { \mathrm { m a t c h } } { : }$

$$
\mathcal { L } = \lambda \mathcal { L } _ { \mathrm { m a t c h } } + ( 1 - \lambda ) \mathcal { L } _ { \mathrm { p i x e l } }\tag{6}
$$

Where $\lambda$ is the balancing coefficient.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p i x e l } } = \| I _ { q } - I _ { r } \| _ { 2 } ^ { 2 } } \end{array}\tag{7}
$$

$$
\mathcal { L } _ { \mathrm { m a t c h } } = \sum _ { k } \| x _ { k } ^ { q } - x _ { k } ^ { r } \| _ { 2 } ^ { 2 }\tag{8}
$$

where $x _ { k } ^ { q } , x _ { k } ^ { r }$ are matched keypoints identified by [20] in the query and rendered images. The total loss is defined as:

From the top-k initial poses selected by the first stage, the pose associated with the rendered image with the highest similarity to the query image is selected as the final pose $\hat { P } = [ \hat { T } | \check { R } ]$

7scenes, Chess

12scenes, Office1, Lounge

<!-- image-->  
Fig. 2. Qualitative comparison of localization accuracy on the 7Scenes/chess and 12Scenes/lounge scenes. Camera poses with distinct colors represent visualization of initial coarse poses estimated by ACE [49], Glace [50], Marepo [52], and our method. (Query Image) Query RGB image; (Estimated NVS) Rendered image using our final estimated pose; (Initial Pose NVS) Rendered image using the initial coarse pose estimated by our method.

## IV. EXPERIMENTS

## A. Evaluation Setup

Datasets. We evaluate the performance of our SGLoc on two public visual localization datasets, including 4 scenes on 7Scenes dataset [46], [47] and 3 scenes on 12Scenes dataset [43]. These datasets contain RGB-D image sequences of various indoor scenes for the evaluation of visual localization performance.

Baselines and Metrics. We use median translation error (cm) and rotation error (Â°) to evaluate the performance of our method. Avg. represents the average error. We compare the metrics with recent traditional localization [7], [9], [22], [25], [27], [44], [48]â[50], NeRF-based localization [30]â[33], [45], [51] and 3DGS-based localization [14], [15] methods. Implementation Details. We use Gaussian Grouping [39] to obtain 3DGS global map. Each submap is constructed as a cubic region centered around poses sampled from the training trajectories. Specifically, we sample a set of camera poses with a fixed spatial interval depending on the size and complexity of the scene. Overlaps between submaps naturally exist due to the fixed sampling interval. We train our semantic-based place retrieval using the Adam optimizer. In the coarse stage, we initialize the learning rate (LR) at 1e-3 and train 24 epochs with a batch size of 32. We utilize threelayer MLP and 4-head 2-layer Multi-Head Self-Attention. Besides, k = 5 and temperature parameter Ï = 0.1. We follow the default settings of all baseline methods to obtain the estimated pose for each query image.

TABLE II  
ACCURACY COMPARISON ON 7SCENES DATASET FOR MEDIAN TRANSLATION AND ROTATION ERRORS (CM/Â°) METRICS.
<table><tr><td>Method</td><td>Chess</td><td>Heads</td><td>Office</td><td>Redkitchen</td><td>Avg.â [cm/Â°]</td></tr><tr><td>PoseNet [7]</td><td>10/4.02</td><td>18/13.0</td><td>17/5.97</td><td>22/5.91</td><td>16.75/7.23</td></tr><tr><td>MS-Transformer [27]</td><td>11/6.38</td><td>13/13.0</td><td>18/8.14</td><td>16/8.92</td><td>14.5/9.11</td></tr><tr><td>DFNet [25]</td><td>3/1.12</td><td>4/2.29</td><td>6/1.54</td><td>7/1.74</td><td>5/1.67</td></tr><tr><td>Marepo [52]</td><td>1.9/0.83</td><td>2.1/1.24</td><td>2.9/0.93</td><td>2.9/0.98</td><td>2.45/1.0</td></tr><tr><td>DSAC* [53]</td><td>0.5/0.17</td><td>0.5/0.34</td><td>1.2/0.34</td><td>0.7/0.21</td><td>0.73/0.27</td></tr><tr><td>ACE [49]</td><td>0.5/0.18</td><td>0.5/0.33</td><td>1/0.29</td><td>0.8/0.20</td><td>0.7/0.5</td></tr><tr><td>GLACE [50]</td><td>0.6/0.18</td><td>0.6/0.34</td><td>1.1/0.29</td><td>0.8/0.20</td><td>0.78/0.25</td></tr><tr><td>FQN-MN [51]</td><td>4.1/1.31</td><td>9.2/2.45</td><td>3.6/2.36</td><td>16.1/4.42</td><td>8.25/2.64</td></tr><tr><td>CrossFire [32]</td><td>1/0.4</td><td>3/2.3</td><td>5/1.6</td><td>2/0.8</td><td>2.75/1.28</td></tr><tr><td>DFNet + NeFeS50 [30]</td><td>2/0.57</td><td>2/1.28</td><td>2/0.56</td><td>2/0.57</td><td>2.1/0.75</td></tr><tr><td>HR-APR [54]</td><td>2/0.55</td><td>2/1.45</td><td>2/0.64</td><td>2/0.67</td><td>2/0.82</td></tr><tr><td>NeRFMatch [33]</td><td>0.9/0.3</td><td>1.6/1.0</td><td>3.3/0.7</td><td>1.3/0.3</td><td>1.78/0.58</td></tr><tr><td>DFNet + GSLoc [14]</td><td>1.3/0.35</td><td>1.1/0.71</td><td>2.2/0.5</td><td>2.2/0.47</td><td>1.7/0.51</td></tr><tr><td>Marepo + GSLoc [14]</td><td>1.3/0.4</td><td>1.4/0.68</td><td>2.2/0.5</td><td>2.2/0.48</td><td>1.78/0.52</td></tr><tr><td>ACE + GSLoc [14]</td><td>0.5/0.15</td><td>0.5/0.28</td><td>1/0.25</td><td>0.8/0.17</td><td>0.7/0.21</td></tr><tr><td>SGLoc (Ours)</td><td>0.14/0.05</td><td>0.14/0.06</td><td>0.43/0.22</td><td>1.3/0.26</td><td>0.5/0.15</td></tr></table>

## B. Experimental Results

Localization Results. As shown in Tab. I, our method outperforms other baseline methods in 12Scenes dataset [43], as well as achieves up to 87.5% increase in translation accuracy and 80% increase in rotation accuracy. Tab. II demonstrates that our method achieves the highest average accuracy in 7scenes dataset [46], [47], with the lowest average translation (0.15cm) and rotation (0.05Â°) errors. Moreover, our method achieves 29% relative increase in average median translation and rotation errors. Such improvement is attributed to our semantic-based global retrieval method, which provides precise initial poses for pose regression. By leveraging semantic consistency to establish correspondence between query image and 3DGS map, our method achieves superior performance over other methods that are based on the traditional feature extraction.

TABLE III  
ABLATION STUDY OF USING DIFFERENT INITIAL POSE ESTIMATORS ON 12SCENES DATASET.
<table><tr><td rowspan="2">Method</td><td colspan="2">Apartment 2</td><td rowspan="2">Office 1 Lounge</td><td rowspan="2">Avg. â [cm/]</td></tr><tr><td>Bed</td><td>Kitchen</td></tr><tr><td>ACE [49]+ SGLoc2</td><td>610.44/73.77</td><td>152.27/150.40</td><td>147.48/97.62</td><td>303.40/107.26</td></tr><tr><td>GLACE [50]+ SGLoc2</td><td>500.557/76.49</td><td>139.38/85.55</td><td>118.22/113.83</td><td>252.72/91.96</td></tr><tr><td>Marepo [52]+ SGLoc2</td><td>518.10/79.70</td><td>97.70/47.24</td><td>257.61/174.91</td><td>299.14/100.62</td></tr><tr><td>SGLoc (Ours)</td><td>0.48/0.39</td><td>0.11/0.05</td><td>0.28/0.08</td><td>0.29/0.17</td></tr></table>

TABLE IV

ABLATION STUDY OF USING DIFFERENT INITIAL POSE ESTIMATORS ON 7SCENES DATASET.
<table><tr><td>Method</td><td>Chess</td><td>Heads</td><td>Office</td><td>Redkitchen</td><td>Avg.â [cm/Â°]</td></tr><tr><td>ACE [49]+ SGLoc2</td><td>186.29/68.13</td><td>67.41/70.80</td><td> $2 2 5 . 6 7 / 7 9 . 7 3$ </td><td>417.51/49.54</td><td>224.22/67.05</td></tr><tr><td>GLACE [50]+ SGLoc2</td><td>266.19/174.91</td><td>106.29/85.55</td><td>226.56/80.99</td><td>339.40/50.73</td><td>234.71/98.05</td></tr><tr><td>Marepo [52]+ SGLoc2</td><td>147.04/171.36</td><td>66.97/77.26</td><td>139.58/108.26</td><td>335.70/46.94</td><td>172.32/100.96</td></tr><tr><td>SGLoc (Ours)</td><td>0.14/0.05</td><td>0.14/0.06</td><td>0.43/0.22</td><td>1.3/0.26</td><td>0.5/0.15</td></tr></table>

Visualization Results. To further demonstrate the effectiveness of our approach, we visualize the localization comparison results of 2 scenes in Fig. 2. The visualization of each scene contains three components: (1) a subfigure of the query image and the rendered image using the estimated pose (bottom left), (2) visualization of initial coarse poses estimated by ACE [49], Glace [50], Marepo [52], and our method (top panel, with distinct colors), (3) a rendered image generated from the initial coarse pose estimated by our method (bottom right). In the subfigure (bottom left), a diagonal line divides into 2 parts: the bottom-left quadrant displays the query image, while the top-right quadrant shows the rendered image with our estimated pose. As is shown in Fig. 2, initial poses provided by our method are the closest to the ground truth, which fully demonstrates the accuracy of our designed coarse pose estimator. The initial poses estimated by other methods lead to large errors, especially in 12scenes/lounge scene. This improvement is attributed to our semantic-based global place retrieval strategy that leverages semantic consistency between 2D query image and 3DGS global map to directly obtain initial pose estimation.

## C. Ablation Studies

In this section, we validate the effectiveness of our semantic-based place retrieval module and demonstrate that our rendering-based optimization can effectively achieve pose refinement.

Effects of semantic-based global retrieval. To evaluate the effectiveness of our semantic-based global retrieval algorithm, we employ initial poses predicted by three state-ofthe-art pose estimators (ACE [49], Glace [50], Marepo [48]) as input for pose refinement. SGLoc2 denotes our renderingbased pose refinement module. The localization performance is evaluated by median rotation error (Â°) and translation error (cm) metrics. As shown in Tab. III and Tab. IV, we present localization results with different initialization strategies. Experimental results demonstrate that these coarse pose estimators followed by the same pose refinement module generally fail to accomplish localization tasks on two datasets. However, our global retrieval algorithm achieves superior performance. It also indicates that our semanticbased global location retrieval module has the most powerful matching capability and robustness in various scenes, which are attributed to full extraction and integration of global semantic features.

TABLE V  
ABLATION STUDY OF OUR SGLOC ON THE 12SCENES DATASET.âW/O SGLOC2â INDICATES WITHOUT OUR POSE REFINEMENT MODULE.
<table><tr><td>Method</td><td colspan="2">Apartment 2 Bed Kitchen</td><td>Office 1 Lounge</td><td>Avg.â [cm/]</td></tr><tr><td>w/o SGLoc2</td><td>4.26/1.82</td><td>1.58/5.35</td><td>2.96/5.21</td><td>4.24/4.52</td></tr><tr><td>SGLoc (Ours)</td><td>0.48/0.39</td><td>0.11/0.05</td><td>0.28/0.08</td><td>0.29/0.17</td></tr></table>

TABLE VI

ABLATION STUDY OF OUR SGLOC ON THE 7SCENES DATASET. âW/O SGLOC2â INDICATES WITHOUT OUR POSE REFINEMENT MODULE.
<table><tr><td>Method</td><td>Chess</td><td>Heads</td><td>Office</td><td>Redkitchen</td><td>Avg.â [cm/Â°]</td></tr><tr><td>w/o SGLoc2</td><td> $2 . 6 4 / 0 . 4 4$ </td><td> $5 . 4 / 0 . 5 2 $ </td><td> $1 . 5 7 / 3 . 1 2$ </td><td>6.26/5.42</td><td>3.97/2.38</td></tr><tr><td>SGLoc (Ours)</td><td>0.14/.05</td><td>0.14/0.06</td><td>0.43/0.22</td><td>1.3/0.26</td><td>0.5/0.15</td></tr></table>

Effects of rendering-based pose refinement. As shown in Tab. V and Tab. VI, rendering-based optimization can effectively reduce the translation error and rotation error by at least 5 times and can even reach the error level of 0.1 cm and 0.01Â°. It also demonstrates that, given a better coarse initial pose, rendering-based optimization can achieve accurate localization results without the need for designing a more complex pose refinement strategy.

## V. CONCLUSIONS

We propose SGLoc, a novel localization framework that estimates 6DoF pose from 3D Gaussian Splatting (3DGS) representation through semantic information. By designing a multi-level localization strategy guided by semantic consistency, our method achieves competitive global localization effects without prior pose information. We introduce a semantic-based global retrieval algorithm that aligns the image with the local region of the global 3DGS map to obtain a coarse pose estimation. Subsequently, we perform rendering-based pose refinement through iterative optimization of the differences between the query image and the rendered image from 3DGS. Experiments demonstrate that our SGLoc achieves superior performance over baselines on 12scenes and 7scenes datasets, showing excellent capabilities in global localization without initial pose prior.

## REFERENCES

[1] J. Liu, D. Zhuo, Z. Feng, S. Zhu, C. Peng, Z. Liu, and H. Wang, âDvlo: Deep visual-lidar odometry with local-to-global feature fusion

and bi-directional structure alignment,â in European Conference on Computer Vision. Springer, 2024, pp. 475â493.

[2] Y. Sha, S. Zhu, H. Guo, Z. Wang, and H. Wang, âTowards autonomous indoor parking: A globally consistent semantic slam system and a semantic localization subsystem,â arXiv preprint arXiv:2410.12169, 2024.

[3] S. Zhu, G. Wang, H. Blum, J. Liu, L. Song, M. Pollefeys, and H. Wang, âSni-slam: Semantic neural implicit slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 167â21 177.

[4] M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler, âD2-net: A trainable cnn for joint description and detection of local features,â in Proceedings of the ieee/cvf conference on computer vision and pattern recognition, 2019, pp. 8092â8101.

[5] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, âLightglue: Local feature matching at light speed,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17 627â 17 638.

[6] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, âFrom coarse to fine: Robust hierarchical localization at large scale,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 12 716â12 725.

[7] A. Kendall, M. Grimes, and R. Cipolla, âPosenet: A convolutional network for real-time 6-dof camera relocalization,â in Proceedings of the IEEE international conference on computer vision, 2015, pp. 2938â2946.

[8] S. Dong, S. Wang, Y. Zhuang, J. Kannala, M. Pollefeys, and B. Chen, âVisual localization via few-shot scene region classification,â in 2022 International Conference on 3D Vision (3DV). IEEE, 2022, pp. 393â 402.

[9] X. Li, S. Wang, Y. Zhao, J. Verbeek, and J. Kannala, âHierarchical scene coordinate classification and regression for visual localization,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 983â11 992.

[10] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[11] S. Zhu, G. Wang, D. Kong, and H. Wang, â3d gaussian splatting in robotics: A survey,â arXiv preprint arXiv:2410.12262, 2024.

[12] S. Zhu, R. Qin, G. Wang, J. Liu, and H. Wang, âSemgauss-slam: Dense semantic gaussian splatting slam,â arXiv preprint arXiv:2403.07494, 2024.

[13] Y. Sun, X. Wang, Y. Zhang, J. Zhang, C. Jiang, Y. Guo, and F. Wang, âicomma: Inverting 3d gaussian splatting for camera pose estimation via comparing and matching,â arXiv preprint arXiv:2312.09031, 2023.

[14] C. Liu, S. Chen, Y. Bhalgat, S. Hu, M. Cheng, Z. Wang, V. A. Prisacariu, and T. Braud, âGsloc: Efficient camera pose refinement via 3d gaussian splatting,â arXiv preprint arXiv:2408.11085, 2024.

[15] H. Zhai, X. Zhang, B. Zhao, H. Li, Y. He, Z. Cui, H. Bao, and G. Zhang, âSplatloc: 3d gaussian splatting-based visual localization for augmented reality,â arXiv preprint arXiv:2409.14067, 2024.

[16] G. Sidorov, M. Mohrat, K. Lebedeva, R. Rakhimov, and S. Kolyubin, âGsplatloc: Grounding keypoint descriptors into 3d gaussian splatting for improved visual localization,â arXiv preprint arXiv:2409.16502, 2024.

[17] X.-S. Gao, X.-R. Hou, J. Tang, and H.-F. Cheng, âComplete solution classification for the perspective-three-point problem,â IEEE transactions on pattern analysis and machine intelligence, vol. 25, no. 8, pp. 930â943, 2003.

[18] M. FISCHLER AND, âRandom sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,â Commun. ACM, vol. 24, no. 6, pp. 381â395, 1981.

[19] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperglue: Learning feature matching with graph neural networks,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4938â4947.

[20] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, âLoftr: Detectorfree local feature matching with transformers,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 8922â8931.

[21] G. Potje, F. Cadar, A. Araujo, R. Martins, and E. R. Nascimento, âXfeat: Accelerated features for lightweight image matching,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2682â2691.

[22] E. Brachmann and C. Rother, âVisual camera re-localization from rgb and rgb-d images using dsac,â IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 9, pp. 5847â5865, 2021.

[23] W. Zhou, C. Liu, J. Lei, L. Yu, and T. Luo, âHfnet: Hierarchical feedback network with multilevel atrous spatial pyramid pooling for rgb-d saliency detection,â Neurocomputing, vol. 490, pp. 347â357, 2022.

[24] P.-E. Sarlin, A. Unagar, M. Larsson, H. Germain, C. Toft, V. Larsson, M. Pollefeys, V. Lepetit, L. Hammarstrand, F. Kahl, et al., âBack to the feature: Learning robust camera localization from pixels to pose,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 3247â3257.

[25] S. Chen, X. Li, Z. Wang, and V. A. Prisacariu, âDfnet: Enhance absolute pose regression with direct feature matching,â in European Conference on Computer Vision. Springer, 2022, pp. 1â17.

[26] A. Kendall and R. Cipolla, âGeometric loss functions for camera pose regression with deep learning,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5974â5983.

[27] Y. Shavit, R. Ferens, and Y. Keller, âLearning multi-scene absolute pose regression with transformers,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 2733â2742.

[28] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[29] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-Y. Lin, âinerf: Inverting neural radiance fields for pose estimation,â in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 1323â1330.

[30] S. Chen, Y. Bhalgat, X. Li, J. Bian, K. Li, Z. Wang, and V. A. Prisacariu, âNeural refinement for absolute pose regression with feature synthesis,â arXiv preprint arXiv:2303.10087, 2023.

[31] B. Zhao, L. Yang, M. Mao, H. Bao, and Z. Cui, âPnerfloc: Visual localization with point-based neural radiance fields,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7450â7459.

[32] A. Moreau, N. Piasco, M. Bennehar, D. Tsishkou, B. Stanciulescu, and A. de La Fortelle, âCrossfire: Camera relocalization on selfsupervised features from an implicit representation,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 252â262.

[33] Q. Zhou, M. Maximov, O. Litany, and L. Leal-Taixe, âThe nerfect Â´ match: Exploring nerf features for visual localization,â in European Conference on Computer Vision. Springer, 2024, pp. 108â127.

[34] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, âPgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction,â IEEE Transactions on Visualization and Computer Graphics, 2024.

[35] B. Matteo, T. Tsesmelis, S. James, F. Poiesi, and A. Del Bue, â6dgs: 6d pose estimation from a single image and a 3d gaussian splatting model,â in European Conference on Computer Vision. Springer, 2024, pp. 420â436.

[36] M. Hafner, M. Katsantoni, T. Koster, J. Marks, J. Mukherjee, Â¨ D. Staiger, J. Ule, and M. Zavolan, âClip and complementary methods,â Nature Reviews Methods Primers, vol. 1, no. 1, p. 20, 2021.

[37] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, âPointnet++: Deep hierarchical feature learning on point sets in a metric space,â Advances in neural information processing systems, vol. 30, 2017.

[38] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Å. Kaiser, and I. Polosukhin, âAttention is all you need,â Advances in neural information processing systems, vol. 30, 2017.

[39] M. Ye, M. Danelljan, F. Yu, and L. Ke, âGaussian grouping: Segment and edit anything in 3d scenes,â in European Conference on Computer Vision. Springer, 2024, pp. 162â179.

[40] Y. Xia, Z. Li, Y.-J. Li, L. Shi, H. Cao, J. F. Henriques, and D. Cremers, âUniloc: Towards universal place recognition using any single modality,â arXiv preprint arXiv:2412.12079, 2024.

[41] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., âSegment anything,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015â4026.

[42] D. H. Johnson, âSignal-to-noise ratio,â Scholarpedia, vol. 1, no. 12, p. 2088, 2006.

[43] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., âLearning transferable

visual models from natural language supervision,â in International conference on machine learning. PmLR, 2021, pp. 8748â8763.

[44] T. Ng, A. Lopez-Rodriguez, V. Balntas, and K. Mikolajczyk, âReassessing the limitations of cnn methods for camera pose regression,â arXiv preprint arXiv:2108.07260, 2021.

[45] L. Chen, W. Chen, R. Wang, and M. Pollefeys, âLeveraging neural radiance fields for uncertainty-aware visual localization,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 6298â6305.

[46] B. Glocker, S. Izadi, J. Shotton, and A. Criminisi, âReal-time rgbd camera relocalization,â in 2013 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2013, pp. 173â179.

[47] J. Shotton, B. Glocker, C. Zach, S. Izadi, A. Criminisi, and A. Fitzgibbon, âScene coordinate regression forests for camera relocalization in rgb-d images,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2013, pp. 2930â2937.

[48] S. Chen, T. Cavallari, V. A. Prisacariu, and E. Brachmann, âMaprelative pose regression for visual re-localization,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 665â20 674.

[49] E. Brachmann, T. Cavallari, and V. A. Prisacariu, âAccelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 5044â5053.

[50] F. Wang, X. Jiang, S. Galliani, C. Vogel, and M. Pollefeys, âGlace: Global local accelerated coordinate encoding,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 562â21 571.

[51] H. Germain, D. DeTone, G. Pascoe, T. Schmidt, D. Novotny, R. Newcombe, C. Sweeney, R. Szeliski, and V. Balntas, âFeature query networks: Neural surface description for camera pose refinement,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5071â5081.

[52] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, âMasked-attention mask transformer for universal image segmentation,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 1290â1299.

[53] E. Brachmann, A. Krull, S. Nowozin, J. Shotton, F. Michel, S. Gumhold, and C. Rother, âDsac-differentiable ransac for camera localization,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 6684â6692.

[54] C. Liu, S. Chen, Y. Zhao, H. Huang, V. Prisacariu, and T. Braud, âHr-apr: Apr-agnostic framework with uncertainty estimation and hierarchical refinement for camera relocalisation,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 8544â8550.