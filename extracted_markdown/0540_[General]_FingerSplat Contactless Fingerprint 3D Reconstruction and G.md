# FingerSplat: Contactless Fingerprint 3D Reconstruction and Generation based on 3D Gaussian Splatting

Yuwei Jiaâ , Student Member, IEEE, Yutang Luâ , Zhe Cui\*, Member, IEEE, Fei Su, Member, IEEE,

AbstractâResearchers have conducted many pioneer researches on contactless fingerprints, yet the performance of contactless fingerprint recognition still lags behind contact-based methods primary due to the insufficient contactless fingerprint data with pose variations and lack of the usage of implicit 3D fingerprint representations. In this paper, we introduce a novel contactless fingerprint 3D registration, reconstruction and generation framework by integrating 3D Gaussian Splatting, with the goal of offering a new paradigm for contactless fingerprint recognition that integrates 3D fingerprint reconstruction and generation. To our knowledge, this is the first work to apply 3D Gaussian Splatting to the field of fingerprint recognition, and the first to achieve effective 3D registration and complete reconstruction of contactless fingerprints with sparse input images and without requiring camera parameters information. Experiments on 3D fingerprint registration, reconstruction, and generation prove that our method can accurately align and reconstruct 3D fingerprints from 2D images, and sequentially generates highquality contactless fingerprints from 3D model, thus increasing the performances for contactless fingerprint recognition.

Index TermsâContactless Fingerprint, Fingerprint Registration, 3D Reconstruction, Fingerprint Generation, 3D Gaussian Splatting.

## I. INTRODUCTION

F INGERPRINT is one of the most widely used bio-metric modalities for identity authentication. Compared metric modalities for identity authentication. Compared with conventional contact-based fingerprints, contactless fingerprints provide a different touchless acquisition approach and lead to specific researches on the contactless fingerprint. In recent years, many contactless fingerprint recognition methods have been proposed[1][2][3][4][5][6][7]. However, due to the issues such as imaging quality, lighting conditions, and finger pose variations, the recognition performances of contactless fingerprints still struggle to achieve practically useful recognition accuracy compared with contact-based fingerprints, especially contactless fingerprints with large pose variation and small overlapping area. Several methods [1][8] furtherly introduce the 3D model into contactless fingerprint recognition to solve pose issues, but their methods often need additional information for 3D fingerprint reconstruction, which introduces extra burdens.

Another main limitation for contactless fingerprint method is the insufficient training data, primarily due to specific acquisition equipment and biometric privacy issues, which further limits the research and application of contactless fingerprints. For data issues, synthetic methods [9][10][11][2][7] are introduced to enlarge the contactless fingerprint datasets. Some synthetic methods [9][10][11] are 2D-image based, while several methods [2][7] are 3D-fingerprint synthetic methods. But the synthetic methods are often data-dependent, and can not generate high-quality and high-fidelity fingerprint images.

<!-- image-->  
Fig. 1: The proposed method can reconstruct 3D Gaussian Splatting from 3 contactless fingerprint images and then generate contactless fingerprint images.

Recently, the advent of deep learning in fingerprint 3D reconstruction [8] and fingerprint alignment [12][13] has encouraged us to incorporate these achievements into fingerprint 3D reconstruction. Our method further demonstrates that, it is feasible to directly reconstruct and render the fingerprint in 3D space from sparse contactless fingerprint images without camera parameters, and that the resulting 3D fingerprint can be used to synthesize new viewpoints of contactless fingerprint images, thereby augmenting contactless fingerprint datasets to eventually improve recognition performances.

## A. Related Works

1) Contactless Fingerprint Recognition: A major challenge for contactless fingerprint recognition methods is the difficulty in effectively matching contactless fingerprints with large pose variance. To address this issue, Tan et al. [1], Cui et al. [8], Grosz et al. [5], and Yin et al.[4] focus mainly on correcting perspective distortions across different views of contactless fingerprints. Tan et al. [1] normalize the orientation of contactless fingerprints, while other methods [8][5][4] unify ridge frequencyâwhether in 3D or 2D spaceto reduce distortion. Dong et al. [2] adopt the network-based method to correct fingerprint distortion, and Shi et al. [3] attempt to learn unique contactless fingerprint features via graph neural networks. Although these methods address view variations through decreasing perspective distortion at some level, the pose variation issue is still not solved yet, as these methods match contactless fingerprints from different views in an independent way, neglecting the complementary information across different finger views to form a complete 3D fingerprint, which is beneficial for recognition.

2) Contactless Fingerprint 3D Reconstruction: A common approach for 3D fingerprint reconstruction is to perform multiview 3D reconstruction when the cameraâs intrinsic and extrinsic parameters are known [14][15][16]. Methods such as shape from shading [17][18] and shape from focus [19] can reconstruct 3D fingerprints, but they require expensive and complex acquisition equipment. There are also 3D fingerprint acquisition methods based on structured light [20][21], ultrasound imaging [22], and laser [23], but they also rely on dedicated acquisition devices. Some previous work has tried to mosaic contactless fingerprint images [24][25][26] to make a complete contactless fingerprint, but these methods have not been tested on publicly available contactless fingerprint datasets without camera pose or true depth. The mosaicked results heavily depend on the specifics of the camera capture setup, which is not practical.

3) Fingerprint Synthesis: Research on fingerprint synthesis has a long history, including early work that used traditional approaches to generate fingerprints [27][28], as well as studies based on GANs [9][10]. More recent research has adopted state-of-the-art diffusion models to synthesize fingerprints and even contactless fingerprint images [11]. However, these fingerprint synthesis methods can only generate two-dimensional fingerprint images. In recent years, several methods [29][2][7][11] have been proposed to synthesize contactless fingerprint data. Priesnitz et al. [29] mapped fingerprint textures onto skin images in 2D space, but their approach cannot generate fingerprints from different viewpoints. Dong et al. [2] addressed this limitation by reprojecting textures onto a 3D Bezier model. Nevertheless, these methods cannot Â´ synthesize new views from existing contactless fingerprints, and the quality of the generated images remains limited. Recently, some work has introduced implicit 3D model representations into finger biometrics [30]. This work builds its own finger-video dataset and are successfully trained a NeRF to reconstruct 3D fingers and generate contactless fingerprints. However, their experiments on the UWA dataset [31] rely on pretraining with dense multi-view inputs, which heavily relies on the large amount of additional fingerprint data.

## B. Objective and Key Contributions

Based on the recent achievements of 3D Gaussian Splatting [32] and 3D reconstruction [33], we propose a direct 3D fingerprint reconstruction method from contactless fingerprint images captured at different angles of the same fingerprint. The core idea is to first align a pair of contactless fingerprints in 3D space using 3D correspondences matching method [33]. Then, based on pairwise correspondences, we estimate global camera parameters to complete the initial 3D point-cloud reconstruction of the fingerprint. Next, we refine both the point cloud and camera poses using 3D Gaussian Splatting. Finally, we perform post-processing of the 3D Gaussian scene using SAM segmentation [34], resulting in a 3D Gaussian representation of the fingerprint that can be used for template construction or rendering new contactless fingerprints (see Fig. 2).

Our approach is able to preserve the original fingerprint image information while achieving high-quality 3D fingerprint results, avoiding the information loss during 3D-2D transition. Our method can further reconstruct and stitch a 3D fingerprint using 2D contactless fingerprint imagesâcaptured from poses with large variationsâwithout any annotated camera parameters, demonstrating strong generalizability.

In general, our work has four contributions:

1) We propose a fingerprint 3D reconstruction framework that can effectively reconstruct high-quality 3D models of contactless fingerprints from sparse-view 2D inputs (e.g. only three viewpoints).

2) We are the first to introduce 3D Gaussian Splatting into the domain of fingerprint recognition, demonstrating that it enables high-quality and photorealistic generation of multi-view contactless fingerprints.

3) We innovatively register contactless fingerprints in 3D space, achieving more accurate alignment than traditional 2D registration methods. Moreover, our registration approach adheres better to physical principles.

4) Experimental results demonstrate that our method can align, reconstruct, and synthesize contactless fingerprints in 3D space, and generates high-quality contactless fingerprint images. Therefore, our method can effectively broadens the variety of contactless fingerprint images and improve the recognition performance for contactless fingerprints.

## II. METHOD

This section details the methodology for reconstructing and synthesizing contactless 3D fingerprints of our method, including: A. Fingerprint Pairwise 3D Local Alignment, B. Fingerprint Groupwise 3D Global Alignment, C. 3D Gaussian Splatting Rendering. D. 3D Fingerprint Post-processing. The proposed method is able to generate 3D as well as 2D fingerprint through 3D fingerprint reconstruction and rendering by 3D Gaussian Splatting. Moreover, the proposed method is able to operate effectively even under unknown finger pose scenes and sparse input fingerprint images, which is especially critical and surpasses previous contactless fingerprint methods.

<!-- image-->  
Fig. 2: Overview structure of the proposed FingerSplat

<!-- image-->  
Fig. 3: Fingerprint Pairwise Local Alignment

## A. Fingerprint Pairwise 3D Local Alignment

In our framework, the first step is to align contactless fingerprint images in a 3D space to generate complete 3D fingerprint data. This 3D alignment is two-stage, we first perform fingerprint alignment in pairs, and then perform global optimization based on these pairwise alignment results to generate a full 3D fingerprint. Unlike previous fingeprint registration methods that match fingerprint features in 2D space, we directly match fingerprints utilizing 3D point cloud registration [33]. Since it is very expensive to manually annotate point correspondences on contactless fingerprints in 3D space, we adopt the original pretrained model of [33] in the first step of local alignment, and finetune the result in the following global alignment.

As shown in Fig. 3, the core of our pairwise local alignment step is an end-to-end deep network that directly regresses 3D fingerprint pointmap from unconstrained fingerprint images, which can be viewed as the position of each point in a 2D image within three-dimensional space. These pointmaps are representations that contain dense 3D geometric information, and all pointmaps are expressed in the same reference view, thereby implicitly encoding the geometric relationships between views. It is worth noting that our method directly uses a pre-trained model to obtain the depth of contactless fingerprints, achieving performance comparable to previous monocular fingerprint depth prediction approaches [8].

## B. Fingerprint Groupwise 3D Global Alignment

After local alignment, global alignment is performed to obtain the global positions and camera parameters of all fingerprint images through joint optimization of the pointmaps. By this step, we can obtain a complete 3D point cloud of the finger, which is then used for the subsequent contactless fingerprint synthesis rendering via 3D Gaussian Splatting.

## â¢ Camera and Pose Initialization

â Camera Initialization: Since the pointmap is expressed in the cameraâs coordinate frame, camera intrinsics (such as focal length) can be recovered by optimizing the reprojection error of the pointmap onto the image plane .

â Pose Initialization: The relative pose between two cameras can be recovered by Procrustes alignment to compare two pointmaps (e.g., $X ^ { 1 , 1 }$ and $X ^ { 2 , 2 } )$ , which yields the scaled relative pose $P ^ { * } = \sigma ^ { * } [ R ^ { * } | t ^ { * } ]$ . Its optimization objective is:

$$
P ^ { * } = \arg \operatorname* { m i n } _ { \sigma , R , t } \sum _ { i } C _ { i } ^ { 1 , 1 } C _ { i } ^ { 2 , 2 } \left\| \sigma ( R X _ { i } ^ { 1 , 1 } + t ) - X _ { i } ^ { 2 , 2 } \right\| ^ { 2 }\tag{1}
$$

. Where $C$ represents the confidence of the corresponding point X.

<!-- image-->  
Fig. 4: Comparison of rendered 3D fingerprints with or without 3D Gaussian Splatting.

## â¢ 3D Global Alignment

We adopt a Global Alignment optimization step which aims to align all pairwise predicted pointmaps into a globally consistent 3D space, forming a complete model of the scene. Its optimization problem is defined as minimizing the 3D projection error between each pairwise prediction and the final global model:

$$
\chi ^ { * } = \arg \operatorname* { m i n } _ { \chi , P , \sigma } \sum _ { e \in \mathcal { E } } \sum _ { v \in e } \sum _ { i = 1 } ^ { H W } C _ { i } ^ { v , e } | | \chi _ { i } ^ { v } - \sigma _ { e } P _ { e } X _ { i } ^ { v , e } | |\tag{2}
$$

Here, $e = ( n , m )$ represents a pair of images, $X _ { i } ^ { v , e }$ is the 3D point predicted in the reference frame of pair $e , P _ { e }$ , $C _ { i } ^ { v , e }$ is the corresponding confidence and $\sigma _ { e }$ are the pose and scale parameters for that pair, and $\chi _ { i } ^ { v }$ is the global 3D point to be optimized.

## C. 3D Gaussian Splatting Rendering

Due to illumination effects, directly stitching a 3D fingerprint from point-maps often shows obvious seams, which introduce substantial noise to the fingerprintâs surface texture. Therefore, we incorporate 3D Gaussian Splatting to further optimize the 3D model with ground-truth viewpoints to remove artifacts and generate fingerprints with higher quality, as shown in Fig. 4.

3D Gaussian Splatting is an explicit 3D scene representation technique that models scenes using a collection of 3D Gaussians. Each 3D Gaussian is defined by a mean vector $\mathbf { x } \in \mathbb { R } ^ { 3 }$ , an opacity $\alpha \in \mathbb { R }$ , and a covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$

$$
G ( \mathbf { p } , \alpha , \Sigma ) = \alpha \exp \left( - \frac { 1 } { 2 } ( \mathbf { p } - \mathbf { x } ) ^ { T } \Sigma ^ { - 1 } ( \mathbf { p } - \mathbf { x } ) \right)\tag{3}
$$

<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 5: Comparison of rendering point clouds without or with 3D segmentation post-processing. (a) Without post-processing, (b) With post-processing.

To handle view-direction-dependent effects, spherical harmonic (SH) coefficients are attached to each Gaussian, and the color is rendered using the view-dependent color and opacity. However, 3D fingerprints typically do not require complex lighting modeling. Moreover, in certain datasets, the data acquisition is not performed by capturing the same scene from different viewpoints, but rather by using a fixed camera position while varying the finger pose. Therefore, when processing colors, we only retain the base color representations without the addition for spherical harmonics. This simplifies the calculation process while ensuring the quality of the rendering results.

The training loss function of 3DGS rendering is defined as:

$$
\mathcal { L } = ( 1 - \lambda _ { \mathrm { S S I M } } ) \mathcal { L } _ { 1 } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } }\tag{4}
$$

where:

$\mathcal { L } _ { 1 }$ is the photometric (L1) loss:

$$
\mathcal { L } _ { 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| G _ { i } - R _ { i } \right|\tag{5}
$$

Here, $G _ { i }$ and $R _ { i }$ are the pixel values of the ground truth and rendered images, respectively, and N is the total number of pixels.

$\mathcal { L } _ { \mathrm { S S I M } }$ is the D-SSIM loss:

$$
\mathcal { L } _ { \mathrm { S S I M } } = 1 - \mathrm { S S I M } ( G , R )\tag{6}
$$

SSIM is the Structural Similarity Index Measure, which considers luminance, contrast, and structural information.

â¢ Î»SSIM is the weight for the D-SSIM term, typically set to 0.2.

## D. 3D Fingerprint Post-processing

The contactless fingerprint rendered by 3D-GS [32] is able to deliver high-fidelity, real-time novel-view synthesis. However, the direct output of 3D-GS uses boundary Gaussians to straddle multiple objects, leading to rough segmentation edges, which needs post-processing. Since the main issue of direct 3D-GS output is the unclear boundary, we conduct a SAGD [34] based 3D fingerprint segmentation step by combining multi-view mask generation, Gaussian decomposition, and label voting. As shown in Fig. 5, when rendering with 3D Gaussian Splatting (3D-GS) using a black background, a large amount of noise is introduced, which degrades the performance of novel view synthesis. The segmentation postprocessing outputs a clean boundary result.

## III. EXPERIMENT

Our method is able to reconstruct, register, and synthesize contactless fingerprints in 3D space, and the following experiments examine our method. First, we test the depth prediction capability of our proposed method for testing 3D reconstruction. Sceond, we test our contactless fingerprint 3D alignment procedure as a fingerprint registration task and conduct comparative experiments. Then, we evaluate the quality of our synthesized contactless fingerprints using image metrics and fingerprint quality assessment metrics. Moreover, we verify that our synthetic fingerprints can be used for fingerprint recognition through matching experiments between synthesized and original fingerprints. Finally, we apply the synthesized new fingerprints to practical applications, including training fingerprint classification networks for fingerprint matching and using them as templates for fingerprint retrieval.

## A. Datasets and Experiment Details

Most of the available contactless fingerprint datasets [18][35][36][37][38][5] lack significant pose variations, making it difficult to complete 3D fingerprints. The most suitable datasets for our experiments are UWA Benchmark 3D/2D Fingerprint Database [31] and CFPose Database [1], so we conduct our experiments primarily on these two datasets. In addition, we employed the PolyU 3D+ Database [18], which provides depth ground truth, to conduct depth prediction experiments.

UWA Benchmark 3D/2D Fingerprint Database contains 8,958 contactless fingerprints with pose variations. This dataset uses sensors to capture contactless fingerprints from three viewpoints simultaneously, eliminating variations in light source and finger position. We use the three contactless fingerprints from the first capture for all subsequent experiments. Meanwhile, we conducted depth prediction experiments using the PolyU 3D+ Database, which contains depth ground truth.

CFPose Database contains 1,400 contactless fingerprints with random pose variations. For this database, we predict the yaw angle of the fingerprints by [1] and select the image with the smallest angle and the two with the largest angle from ten different viewpoints to synthesize 3D fingerprints for all subsequent experiments. Since the data collection protocol of this dataset is different from UWA, we preprocess the dataset by upright-rotation using the method of Cui et al. [8]. We also rescale its image width to match the UWA dataset (1024 pixels), and cropp out the black background at the top, retaining only a height of 1280 pixels.

PolyU 3D+ Database [18] We utilize the 2,016 ground truth depth maps and corresponding images from the first session for depth estimation experiments.

Since the 3D Gaussian Splatting is able to generate any views of fingerprints, we extract 12 frames at equal intervals between the left, front, and right viewpoints in both datasets as our synthesized fingerprint images for subsequent experiments.

We generally adopt the default parameter settings of InstantSplat [39] and SAGD [34], with the sole exception of setting SH DEGREE = 0.

TABLE I: Weighted depth error (mm) compared with Cui et al [8].
<table><tr><td>Dataset</td><td>Cui et al. [8]</td><td>Proposed</td></tr><tr><td>PolyU 3D+</td><td>2.9577</td><td>1.7876</td></tr></table>

## B. 3D Reconstruction Accuracy

To validate the effectiveness of our method on 3D fingerprint reconstruction, we perform depth prediction experiments on the PolyU 3D+ Database [18] with real depth ground truth. Since the fingerprint alignment step requires at least two input images while most current fingerprint depth estimation methods are single-input, we feed two identical images during testing to obtain depth, effectively converting our method into a monocular depth estimation approach for comparison with existing monocular method [8]. As shown in Table I, our method significantly outperforms that of Cui et al. [8], indicating the reliability of our depth predictions. Since Cui et al.âs method was trained on the UWA Database where the ground truth is not real depth, we do not include depth error comparisons on this dataset for fairness.

## C. 3D Fingerprint Registration Accuracy

We verify that our method can correctly register contactless fingerprints. Previous 2D registration methods [40][41][42][12][13] cannot obtain 3D information of contactless fingerprints, and thus can only register contactless fingerprints at the 2D image level. Our method, however, can effectively register them in 3D space by predicting the depth of contactless fingerprints in 3D space, thereby significantly improving the accuracy of contactless fingerprint registration. In the registration experiment, only the results of the registered fingerprint point cloud are used for experiments, without involving the subsequent new view rendering and synthesis.

Previous fingerprint registration methods often use the correlation coefficient of binarized fingerprints as a metric for registration accuracy. However, since the binarization of contactless fingerprints is often inaccurate, we adopt the average distance of minutiae points in the test set after registration as the metric for registration accuracy. That is, given the pixel coordinates (x, y) and mated $( { \bf x } ^ { \prime } , \dot { \bf y } ^ { \prime } )$ of minutiae point pairs in two images, their pixel distance is

$$
D = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sqrt { ( \mathbf { x } _ { i } - \mathbf { x } _ { i } ^ { \prime } ) ^ { 2 } + ( \mathbf { y } _ { i } - \mathbf { y } _ { i } ^ { \prime } ) ^ { 2 } }
$$

We use Verifinger 13.0 [43] to extract minutiae points and match minutiae relationships on the UWA dataset, and manually adjust the minutiae points and matching relationships through manual adjustment to obtain more precise ground truth label. Considering the large of manual annotation, we randomly annotate minutiae points and matching relationships for only 46 fingers, and perform matching between them, obtaining 6 pairs of matches for each finger.

Since fingerprint textures with a 90-degree pose difference almost have no intersection, which are meaningless for evaluation, only fingerprints with a 45-degree pose difference are selected for testing, resulting in 46\*4=184 registration pairs.

<!-- image-->  
Fig. 6: Manually annotate ground truth minutiae points and matching relationships.

TABLE II: Comparisons of registration accuracy across different methods and datasets
<table><tr><td colspan="2"></td><td rowspan="2">UWA</td><td rowspan="2">UWA*</td><td rowspan="2">CFPose*</td></tr><tr><td colspan="2">Test Set Methods</td></tr><tr><td>2D</td><td>Rigid Transform TPS Based [40]</td><td>52.27 26.18</td><td>73.47 6.831</td><td>42.58 4.671</td></tr><tr><td>3D</td><td>SARNet [13] VGGT [44] Proposed</td><td>16.86  $1 0 . 4 5 ^ { 2 }$  13.94</td><td>39.09 28.692 30.24</td><td>25.57 303.02 40.07</td></tr></table>

\*: Represents the minutiae matching relationships predicted by Verifinger, which are not necessarily correct.  
1: The TPS uses the predicted minutiae pairs to conduct deformation, with the ground truth also being the predicted minutiae pairs, thus the minutiae errors are unrealistically low. Using TPS aligns the minutiae pairs as much as possible without considering image plausibility, which leads to artificially low errors.

2: Although VGGT performs well on UWA, it completely fails to work on CFPose.

Additionally, considering that manually annotated test sets might be suspected of cherry-picking good results, we also test registration accuracy on 6214 pairs from 1167 fingers where Verifinger could find matching minutiae points. However, due to the large angle difference between left and right views, Verifinger often makes incorrect minutiae matching, resulting in poor matching performance. We ultimately remove matches with a 90-degree pose difference and conduct registration experiments on the remaining 4312 pairs. We also conduct similar experiments on CFPose. As shown in Table II, for contactless fingerprint datasets with large pose spans like UWA, 3D registration methods have a clear advantage over 2D registration. However, for datasets like CFPose with smaller pose differences, 2D registration methods still have certain advantages, although the state-of-the-art 3D registration method VGGT [44] performs poorly on CFPose, our method can still achieve good results.

The experimental results show that 3D registration methods have clear advantages over 2D registration methods on datasets with large pose variations captured from different angles, such as UWA. We visualize the registration results of several methods for contactless fingerprints in Fig. 7, revealing that 2D methods produce registrations with obvious physically implausible deformations, while 3D registration methods can correctly register contactless fingerprints in accordance with physical laws. We do not conduct comparative experiments with dense registration methods [41][42][12] because these methods are based on TPS-based registration, and when TPSbased methods fail, these methods also fail, yielding results similar to TPS-based methods, making such experiments unnecessary. Since it is difficult to completely annotate the correct matching relationships of minutiae points manually, we only annotated the minutiae points that can be obviously observed manually when annotating the matching relationships, which is why there arenât many matching minutiae points in Fig. 7.

TABLE III: NFIQ2 on both real and synthetic contactless fingerprints.
<table><tr><td>Dataset</td><td>Real image</td><td>Mast3R [45]</td><td>VGGT [44]</td><td>Proposed</td></tr><tr><td>UWA</td><td>38.13</td><td>35.48</td><td>35.46</td><td>35.68</td></tr><tr><td>CFPose</td><td>40.19</td><td>38.43</td><td>-</td><td>39.57</td></tr></table>

Although the recently proposed VGGT[44] shows better performance on the UWA dataset, the method completely fails to handle the CFPose dataset as shown in Fig. 8. This is mainly because VGGTâs training set consists entirely of static objects. UWA captures fingers from three angles simultaneously, which is consistent with VGGTâs training set construction method. However, CFPose uses fixed camera positions to capture fingers in different poses, which doesnât align with VGGTâs training set, making it difficult to work effectively. Additionally, since VGGT is an end-to-end framework, itâs challenging to adapt it for CFPose.

## D. Synthesized Fingerprint Data Quality

We evaluate the quality of the synthesized images using NFIQ2 for fingerprint quality assessment by Verifinger. For real images, UWA use nearly 4,500 images from the three angles of the first capture to calculate the average NFIQ2, with missing images assigned a NFIQ2 of 0; CFPose calculate the average NFIQ2 for all images. As shown in Table III, the quality of our generated contactless fingerprints is relatively close to that of the real datasets. Since VGGT fails to perform effective registration on CFPose, we do not evaluate its NFIQ2. Fig. 9 shows the NFIQ2 distributions of the contactless fingerprint images synthesized by our method and the real fingerprint images. It can be observed that the distributions are relatively close on both datasets, indicating that our method can effectively synthesize contactless fingerprint data.

## E. Synthesized Fingerprints for Fingerprint Matching

Subsequently, we match synthesized images to real images for testing. For the UWA dataset, there are total of $1 5 0 0 \times 1 2 = 1 8 0 0 0$ synthesized images, and real images are $1 5 0 0 \times 3 = 4 5 0 0$ images each time. Missing images are assigned a matching score of 0. Following previous matching principles [1][2][8] on UWA, we match 3000 images from the first 100 fingers, comparing real images from the first and second captures, synthesized images with real images from the

I

<!-- image-->

<!-- image-->  
Fig. 7: Examples of 3D registration results of a pair of contactless fingerprints with a 45-degree difference, showing that 2D methods cannot handle fingerprint rotation effectively in three-dimensional space, and may even produce physically implausible distortions; meanwhile, our method can effectively handle contactless fingerprint registration with large angle difference, and remains physically plausible, while other methods cannot produce reasonable images.

<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 8: Registration comparison between VGGT and our method on CFPose. VGGT often fails to register completely. (a) VGGT, (b) Proposed.

TABLE IV: Matching performances between synthetic and real fingerprints using Verifinger on UWA database.
<table><tr><td>Dataset</td><td>shot</td><td>EER</td><td>FMR@1%</td><td>FMR_Zero</td></tr><tr><td>Real Image1</td><td>1&amp;2</td><td>32.30%</td><td>62.87%</td><td>69.41%</td></tr><tr><td rowspan="2">InstantSplat1</td><td>1</td><td>31.79%</td><td>61.68%</td><td>66.59%</td></tr><tr><td>2</td><td>34.17%</td><td>64.27%</td><td>70.38%</td></tr><tr><td rowspan="2">Proposed1</td><td>1</td><td>28.00%</td><td>61.47%</td><td>66.50%</td></tr><tr><td>2</td><td>30.00%</td><td>64.01%</td><td>70.22%</td></tr><tr><td>Real Image2</td><td>1&amp;2</td><td>31.48%</td><td>63.40%</td><td>69.84%</td></tr><tr><td rowspan="2">InstantSplat2</td><td>1</td><td>32.48%</td><td>65.24%</td><td>74.85%</td></tr><tr><td>2</td><td>34.75%</td><td>69.28%</td><td>79.26%</td></tr><tr><td rowspan="2">Proposed2</td><td>1</td><td>26.05%</td><td>64.64%</td><td>74.25%</td></tr><tr><td>2</td><td>27.17%</td><td>68.84%</td><td>78.80%</td></tr></table>

1: Following [8], the matching test is conducted with both real and synthetic images sized at 3000\*3000.  
2: For the matching test on the entire dataset, the real images are 4500\*4500 and the synthetic images are 18000\*4500.

first capture, and synthesized images with real images from the second capture. Each comparison involves 3000 Ã 3000 matches, resulting in 9000 genuine matches and 8,991,000 impostor matches. The synthesized images selected are the 1st, 7th, and 12th from 12 views. As shown in Table IV, the results of synthesized images are very close to the matching results between real images. The three angles from the first capture are used to synthesize 3D fingerprints, and the matching performance of synthesized fingerprint images are somewhat better when matched with the first capture compared to the second capture. We use the original InstantSplat [39] pipeline without any modifications as baseline method.

TABLE V: Matching performances between synthetic and real fingerprints using Verifinger on CFPose database.
<table><tr><td>Dataset</td><td>EER</td><td>FMR@1%</td><td>FMR_Zero</td></tr><tr><td>Real Image</td><td>11.38%</td><td>16.38%</td><td>31.35%</td></tr><tr><td>InstantSplat</td><td>15.98%</td><td>29.15%</td><td>49.18%</td></tr><tr><td>Proposed</td><td>13.40%</td><td>24.90%</td><td>44.47%</td></tr></table>

Subsequently, since our method does not involve a training set, we performed matching on the entire dataset. We conduct 18000 Ã 4500 matching experiments for both the first and second captures, resulting in 54,000 genuine matches and 80,946,000 impostor matches. It can be seen that the matching results between synthesized data and real data are also very close.

It is worth mentioning that our results using Verifinger matching are not as good as previous outcomes [8][1][2]. This is because we use the original fingerprint images rather than preprocessed ones (histogram equalization, rotation correction, frequency normalization, etc.). Whether fingerprints are preprocessed or not does not affect our demonstration that the proposed method can synthesize contactless fingerprints with quality comparable to the original images.

We also conduct similar experiments on CFPose. Specifically, the real image matching involved 1400\*1399/2 = 979,300 total pairs, including 6,300 genuine matches and 973,000 impostor pairs. For synthetic images, the matching involve $1 4 0 ^ { * } 1 2 ^ { * } 1 4 0 ^ { * } 1 0 = 2 , 3 5 2 , 0 0 0$ pairs, including 16,800 genuine matches and 2,335,200 impostor pairs. The experimental results demonstrate that our performance closely approaches that on real images, and our method effectively improves the matching performance.

<!-- image-->

<!-- image-->

Fig. 9: NFIQ2 distributions on UWA (left) and CFPose (right) databases of real images and our synthetic images  
<!-- image-->  
Fig. 10: Examples of synthetic contactless fingerprints of 12 different views. ji

TABLE VI: Matching Performances of Synthetic Fingerprints at Different Angles  
Matched with Center Pose
<table><tr><td>Angle Diff ()</td><td>EER</td><td>FMR@1%</td><td>FMR@0.01%</td><td>FMR @0%</td></tr><tr><td>-45</td><td>0.5287</td><td>0.7367</td><td>0.9487</td><td>0.9553</td></tr><tr><td>-36.82</td><td>0.4393</td><td>0.6213</td><td>0.8947</td><td>0.9113</td></tr><tr><td>-28.67</td><td>0.3353</td><td>0.5193</td><td>0.8207</td><td>0.8453</td></tr><tr><td>-20.45</td><td>0.1747</td><td>0.2453</td><td>0.4780</td><td>0.5047</td></tr><tr><td>-12.27</td><td>0.1147</td><td>0.1273</td><td>0.1740</td><td>0.1820</td></tr><tr><td>-4.091</td><td>0.0813</td><td>0.0847</td><td>0.0933</td><td>0.0953</td></tr><tr><td>4.091</td><td>0.0787</td><td>0.0813</td><td>0.0873</td><td>0.0873</td></tr><tr><td>12.27</td><td>0.1127</td><td>0.1220</td><td>0.1540</td><td>0.1600</td></tr><tr><td>20.45</td><td>0.1660</td><td>0.2093</td><td>0.3293</td><td>0.3487</td></tr><tr><td>28.67</td><td>0.3080</td><td>0.4573</td><td>0.7527</td><td>0.7693</td></tr><tr><td>36.82</td><td>0.4507</td><td>0.6387</td><td>0.8913</td><td>0.9060</td></tr><tr><td>45</td><td>0.5713</td><td>0.7460</td><td>0.9407</td><td>0.9507</td></tr></table>

We further detailedly test the synthesized fingerprint images with different pose angles. In the UWA dataset, each fingerprint is captured from the left, front, and right perspectives. But our method is able to synthesize any view angle of contactless fingerprints using images from three angles. In our experiment, we generate 12 different views in reality, and test those different viewsâ matching performances. We match the 12 synthesized new fingerprints with the real images from the left and front views. We consider the left angle to be -45 degrees, the front to be 0 degrees, and the right to be 45 degrees, and perform 1500\*1500 matching for each of the 12 angles. The final results, as shown in Table VI and Fig. 11, indicate that contactless fingerprint matching performs well when the angle difference is less than 20 degrees, but matching performance deteriorates as the angle difference increases. When the angle difference reaches 90 degrees, the performance becomes very poor.

Matched with Left Pose
<table><tr><td>Angle Diff ()</td><td>EER</td><td>FMR @1%</td><td>FMR@0.01%</td><td>FMR @0%</td></tr><tr><td>0</td><td>0.0678</td><td>0.0760</td><td>0.0860</td><td>0.0860</td></tr><tr><td>8.18</td><td>0.0893</td><td>0.1180</td><td>0.1647</td><td>0.1673</td></tr><tr><td>16.36</td><td>0.1387</td><td>0.3460</td><td>0.5540</td><td>0.5713</td></tr><tr><td>24.55</td><td>0.2024</td><td>0.6227</td><td>0.8633</td><td>0.8747</td></tr><tr><td>32.73</td><td>0.2229</td><td>0.6653</td><td>0.8967</td><td>0.9180</td></tr><tr><td>40.91</td><td>0.2367</td><td>0.7133</td><td>0.9380</td><td>0.9487</td></tr><tr><td>49.09</td><td>0.2833</td><td>0.8093</td><td>0.9740</td><td>0.9807</td></tr><tr><td>57.27</td><td>0.4153</td><td>0.9033</td><td>0.9913</td><td>0.9920</td></tr><tr><td>65.45</td><td>0.5587</td><td>0.9640</td><td>0.9973</td><td>0.9980</td></tr><tr><td>73.64</td><td>0.6547</td><td>0.9813</td><td>0.9973</td><td>0.9980</td></tr><tr><td>81.82</td><td>0.6647</td><td>0.9793</td><td>0.9980</td><td>0.9993</td></tr><tr><td>90.0</td><td>0.6813</td><td>0.9847</td><td>1.0000</td><td>1.0000</td></tr></table>

Fig. 10 qualitatively illustrates some synthesized results with different poses, demonstrating that our method can generate high-quality contactless fingerprint images with large pose variations.

<!-- image-->

<!-- image-->  
Fig. 11: Performance for synthetic fingerprint matching across different angles. Left from $9 0 Â°$ to $0 ^ { \circ }$ . Right from $- 4 5 ^ { \circ }$ to $4 5 ^ { \circ }$

TABLE VII: Matching performances between synthetic and real fingerprints by DeepPrint on UWA database.
<table><tr><td>Dataset</td><td>shot</td><td>EER</td><td>FMR@1%</td><td>FMR_Zero</td></tr><tr><td rowspan="2">DeepPrint [46]</td><td>Tex</td><td>31.70%</td><td>72.56%</td><td>90.24%</td></tr><tr><td>Tex + Minu</td><td>28.66%</td><td>66.90%</td><td>86.62%</td></tr><tr><td rowspan="2">Proposed</td><td> $\mathrm { T e x }$ </td><td>28.55%</td><td>67.06%</td><td>83.69%</td></tr><tr><td> $\mathrm { T e x } + \mathrm { M i n u }$ </td><td>26.37%</td><td>63.46%</td><td>83.41%</td></tr></table>

TABLE VIII: Matching performances between synthetic and real fingerprints by DeepPrint on CFPose database.
<table><tr><td>Dataset</td><td>shot</td><td>EER</td><td>FMR@1%</td><td>FMR_Zero</td></tr><tr><td rowspan="2">DeepPrint [46]</td><td>Tex</td><td>5.67%</td><td>24.59%</td><td>75.58%</td></tr><tr><td>Tex + Minu</td><td>5.56%</td><td>16.85%</td><td>63.08%</td></tr><tr><td rowspan="2">Proposed</td><td>Tex</td><td>5.52%</td><td>21.54%</td><td>70.87%</td></tr><tr><td>Tex + Minue</td><td>5.53%</td><td>17.61%</td><td>60.15%</td></tr></table>

## F. Synthesized data for Contactless Fingerprint Recognition

We further test our synthesized for training deep neural networks to exam its improvement for the recognition performance of contactless fingerprints. In this section, we use the open-source DeepPrint [46] for training contactless fingerprint recognition network. We use the last 50 fingers of the UWA dataset as the training set, training with both original images (about 3000) and original images + synthesized images (about 3000+18000), and test on the first 100 fingers of UWA (about 6000 images) and the first 120 fingers of CFPose (about 1200 images). Moreover, we do not include any real or synthetic data from CFPose in our training set; however, we conduct matching experiments on the first 120 fingers of CFPose.

We use Verifinger to extract contactless fingerprint minutiae as ground truth for the Minutiae branch. To avoid potential inaccuracies in the minutiae extracted by Verifinger, we train both a DeepPrint with only the Texture branch and a DeepPrint with both Texture and Minutiae branches. All trainings are conducted for 100 epochs. As shown in Table VII and VIII and Fig. VII, after adding our synthesized fingerprints, the matching performances of networks trained using both methods improved significantly.

<!-- image-->  
Fig. 12: DET curves for training a fingerprint matching network with or without our synthetic data

## IV. CONCLUSION

In this research, we innovatively propose a framework for contactless fingerprint 3D reconstruction, registration, and fingerprint generation. It is the first method to introduce 3D Gaussian Splatting (3D-GS) into the fingerprint recognition field, achieving high-quality 3D fingerprint synthesis on public contactless fingerprint datasets. Moreover, we also propose a 3D fingerprint registration method, which is more physically consistent and interpretable compared to traditional 2D registration methods when applied to contactless fingerprints. Most importantly, the newly synthesized contactless fingerprint can be used in training fingerprint recognition neural networks, effectively improving the performance of contactless fingerprint recognition.

## ACKNOWLEDGMENT

This work is supported in part by the National Natural Science Foundation of China under Grants 62206026.

## REFERENCES

[1] H. Tan and A. Kumar, âTowards more accurate contactless fingerprint minutiae extraction and pose-invariant matching,â IEEE Transactions on Information Forensics and Security, vol. 15, pp. 3924â3937, 2020. 1, 2, 5, 6, 7

[2] C. Dong and A. Kumar, âSynthesis of multi-view 3D fingerprints to advance contactless fingerprint identification,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 4, no. 11, pp. 13 134â 13 151, 2023. 1, 2, 6, 7

[3] Y. Shi, Z. Zhang, S. Liu, and M. Liu, âTowards more accurate matching of contactless fingerprints with a deep geometric graph convolutional network,â IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 5, no. 1, pp. 29â38, 2022. 1, 2

[4] X. Yin, Y. Zhu, and J. Hu, âContactless fingerprint recognition based on global minutia topology and loose genetic algorithm,â IEEE Transactions on Information Forensics and Security, vol. 15, pp. 28â41, 2019. 1, 2

[5] S. A. Grosz, J. J. Engelsma, E. Liu, and A. K. Jain, âC2CL: Contact to contactless fingerprint matching,â IEEE Transactions on Information Forensics and Security, vol. 17, pp. 196â210, 2022. 1, 2, 5

[6] X. Yin, Y. Zhu, and J. Hu, â3D fingerprint recognition based on ridge-valley-guided 3D reconstruction and 3D topology polymer feature extraction,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 3, pp. 1085â1091, 2021. 1

[7] C. Dong and A. Kumar, âBridging dimensions in fingerprints to advance distinctiveness: Recovering 3d minutiae from a single contactless 2d fingerprint image,â IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1, 2

[8] Z. Cui, J. Feng, and J. Zhou, âMonocular 3D fingerprint reconstruction and unwarping,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 8679â8695, 2023. 1, 2, 3, 5, 6, 7

[9] S. A. Grosz and A. K. Jain, âSpoofGAN: Synthetic fingerprint spoof images,â IEEE Transactions on Information Forensics and Security, vol. 18, pp. 730â743, 2022. 1, 2

[10] J. J. Engelsma, S. A. Grosz, and A. K. Jain, âPrintsGAN: Synthetic fingerprint generator,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 6111â6124, 2023. 1, 2

[11] S. A. Grosz and A. K. Jain, âUniversal fingerprint generation: Controllable diffusion model with multimodal conditions,â IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 1, 2

[12] X. Guan, J. Feng, and J. Zhou, âPhase-aggregated dual-branch network for efficient fingerprint dense registration,â IEEE Transactions on Information Forensics and Security, 2024. 1, 5, 6

[13] Y. Jia, Z. Cui, and F. Su, âA single-step accurate fingerprint registration method based on local feature matching,â arXiv preprint arXiv:2507.16201, 2025. 1, 5, 6

[14] F. Liu and D. Zhang, â3d fingerprint reconstruction system using feature correspondences and prior estimated finger model,â Pattern Recognition, vol. 47, no. 1, pp. 178â193, 2014. 2

[15] G. Parziale, E. Diaz-Santana, and R. Hauke, âThe surround imager: A multi-camera touchless device to acquire 3D rolled-equivalent fingerprints,â in International Conference on Biometrics (ICB), 2006, pp. 244â250. 2

[16] R. D. Labati, A. Genovese, V. Piuri, and F. Scotti, âToward unconstrained fingerprint recognition: A fully touchless 3-d system based on two views on the move,â IEEE transactions on systems, Man, and cybernetics: systems, vol. 46, no. 2, pp. 202â219, 2015. 2

[17] A. Kumar and C. Kwong, âTowards contactless, low-cost and accurate 3D fingerprint identification,â in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013, pp. 3438â3443. 2

[18] C. Lin and A. Kumar, âTetrahedron based fast 3D fingerprint identification using colored LEDs illumination,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 12, pp. 3022â3033, 2017. 2, 5

[19] G. Abramovich, K. Harding, S. Manickam, J. Czechowski, V. Paruchuru, R. Tait, C. Nafis, and A. Vemury, âMobile, contactless, single-shot, fingerprint capture system,â in Biometric Technology for Human Identification VII, vol. 7667. SPIE, 2010, pp. 70â81. 2

[20] Y. Wang, L. G. Hassebrook, and D. L. Lau, âData acquisition and processing of 3-d fingerprints,â IEEE Transactions on Information Forensics and Security, vol. 5, no. 4, pp. 750â760, 2010. 2

[21] J. Wang, Y. Ye, W. Cao, J. Zhao, and Z. Song, â3d fingerprint reconstruction and registration based on binocular structured light,â in Chinese Conference on Biometric Recognition. Springer, 2023, pp. 73â84. 2

[22] A. Baradarani, R. G. Maev, and F. Severin, âResonance based analysis of acoustic waves for 3d deep-layer fingerprint reconstruction,â in 2013 IEEE International Ultrasonics Symposium (IUS). IEEE, 2013, pp. 713â716. 2

[23] J. Galbally, G. Bostrom, and L. Beslay, âFull 3D touchless fingerprint recognition: Sensor, database and baseline performance,â in International Joint Conference on Biometrics (IJCB), 2017, pp. 225â233. 2

[24] M. Alkhathami, F. Han, and R. Van Schyndel, âA mosaic approach to touchless fingerprint image with multiple views,â in Proceedings of the International Conference on Distributed Smart Cameras, 2014, pp. 1â8. 2

[25] F. Liu, D. Zhang, C. Song, and G. Lu, âTouchless multiview fingerprint acquisition and mosaicking,â IEEE Transactions on Instrumentation and Measurement, vol. 62, no. 9, pp. 2492â2502, 2013. 2

[26] F. Liu, Q. Zhao, and D. Zhang, Advanced fingerprint recognition: from 3D shape to ridge detail. Springer, 2020. 2

[27] R. Cappelli, D. Maltoni, D. Maio, A. Jain, and S. Prabhakar, âSynthetic fingerprint generation,â Handbook of fingerprint recognition, pp. 203â 232, 2003. 2

[28] R. Cappelli, D. Maio, and D. Maltoni, âAn improved noise model for the generation of synthetic fingerprints,â in ICARCV 2004 8th Control, Automation, Robotics and Vision Conference, 2004., vol. 2. IEEE, 2004, pp. 1250â1255. 2

[29] J. Priesnitz, C. Rathgeb, N. Buchmann, and C. Busch, âSyncolfinger: Synthetic contactless fingerprint generator,â Pattern Recognition Letters, vol. 157, pp. 127â134, 2022. 2

[30] H. Xu, J. Huang, Y. Ma, Z. Li, and W. Kang, âImproving 3d finger traits recognition via generalizable neural rendering,â International Journal of Computer Vision, pp. 1â35, 2024. 2

[31] W. Zhou, J. Hu, I. Petersen, S. Wang, and M. Bennamoun, âA benchmark 3D fingerprint database,â in International Conference on Fuzzy Systems and Knowledge Discovery (FSKD), 2014, pp. 935â940. 2, 5

[32] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023. 2, 4

[33] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, âDust3r: Geometric 3d vision made easy,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 697â20 709. 2, 3

[34] X. Hu, Y. Wang, L. Fan, J. Fan, J. Peng, Z. Lei, Q. Li, and Z. Zhang, âSemantic anything in 3d gaussians,â arXiv preprint arXiv:2401.17857, 2024. 2, 4, 5

[35] C. Lin and A. Kumar, âMatching contactless and contact-based conventional fingerprint images for biometrics identification,â IEEE Transactions on Image Processing, vol. 27, no. 4, pp. 2008â2021, 2018. 5

[36] P. Birajadar, M. Haria, P. Kulkarni, S. Gupta, P. Joshi, B. Singh, and V. Gadre, âTowards smartphone-based touchless fingerprint recognition,â Sadhan Â¯ aÂ¯, vol. 44, pp. 1â15, 2019. 5

[37] A. Sankaran, A. Malhotra, A. Mittal, M. Vatsa, and R. Singh, âOn smartphone camera based fingerphoto authentication,â in International Conference on Biometrics Theory, Applications and Systems (BTAS), 2015, pp. 1â7. 5

[38] A. Malhotra, A. Sankaran, M. Vatsa, and R. Singh, âOn matching finger-selfies using deep scattering networks,â IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 2, no. 4, pp. 350â362, 2020. 5

[39] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos et al., âInstantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds,â arXiv preprint arXiv:2403.20309, vol. 2, no. 3, p. 4, 2024. 5, 7

[40] A. M. Bazen and S. H. Gerez, âFingerprint matching by thin-plate spline modelling of elastic deformations,â Pattern Recognition, vol. 36, no. 8, pp. 1859â1867, 2003. 5, 6

[41] X. Si, J. Feng, J. Zhou, and Y. Luo, âDetection and rectification of distorted fingerprints,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 3, pp. 555â568, 2015. 5, 6

[42] Z. Cui, J. Feng, and J. Zhou, âDense registration and mosaicking of fingerprints by training an end-to-end network,â IEEE Transactions on Information Forensics and Security, vol. 16, pp. 627â642, 2020. 5, 6

[43] Neurotechnology Inc., VeriFinger SDK 13.0. http://www. neurotechnology.com. 5

[44] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, âVggt: Visual geometry grounded transformer,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 6

[45] V. Leroy, Y. Cabon, and J. Revaud, âGrounding image matching in 3d with mast3r,â in European Conference on Computer Vision. Springer, 2024, pp. 71â91. 6

[46] T. Rohwedder, D. Osorio-Roig, C. Rathgeb, and C. Busch, âBenchmarking fixed-length fingerprint representations across different embedding sizes and sensor types,â in 2023 International Conference of the Biometrics Special Interest Group (BIOSIG). IEEE, 2023, pp. 1â6. 9