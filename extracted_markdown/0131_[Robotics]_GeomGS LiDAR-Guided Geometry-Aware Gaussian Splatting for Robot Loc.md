# GeomGS: LiDAR-Guided Geometry-Aware Gaussian Splatting for Robot Localization

Jaewon Lee1, Mangyu Kong1, Minseong Park1, and Euntai Kim1,â

Abstractâ Mapping and localization are crucial problems in robotics and autonomous driving. Recent advances in 3D Gaussian Splatting (3DGS) have enabled precise 3D mapping and scene understanding by rendering photo-realistic images. However, existing 3DGS methods often struggle to accurately reconstruct a 3D map that reflects the actual scale and geometry of the real world, which degrades localization performance. To address these limitations, we propose a novel 3DGS method called Geometry-Aware Gaussian Splatting (GeomGS). This method fully integrates LiDAR data into 3D Gaussian primitives via a probabilistic approach, as opposed to approaches that only use LiDAR as initial points or introduce simple constraints for Gaussian points. To this end, we introduce a Geometric Confidence Score (GCS), which identifies the structural reliability of each Gaussian point. The GCS is optimized simultaneously with Gaussians under probabilistic distance constraints to construct a precise structure. Furthermore, we propose a novel localization method that fully utilizes both the geometric and photometric properties of GeomGS. Our GeomGS demonstrates state-of-the-art geometric and localization performance across several benchmarks, while also improving photometric performance.

## I. INTRODUCTION

3D Gaussian Splatting (3DGS) has attracted significant interest from various fields, ranging from computer vision and robotics to AR/VR, and it is considered promising direction mapping and localization. Since the inception of 3DGS, numerous studies have been conducted to improve the performance of 3DGS, with most relying on Structurefrom-Motion (SfM) [1] to reconstruct the 3D structure [2], [3]. Recently, some works have leveraged range data from LiDAR [4], [5] or estimated depth [6] rather than solely relying on Structure-from-Motion (SfM) [1] to refine the scale and structure of 3DGS. In particular, [7] proposed to impose a simple distance constraint between initial LiDAR points and Gaussian points, aiming to maintain the structural consistency of 3DGS.

However, we believe there is still room for improvement in combining LiDAR with 3DGS to further enhance its quality. Specifically, the simple distance constraint developed in [7] focuses primarily on improving rendering quality, while relatively neglecting the geometric accuracy of points, potentially distorting the scale and structure of 3DGS depending on the environment. For instance, if this distance constraint is applied to the sky, enforcing the distances between LiDAR and 3DGS points to match could severely distort the geometric structure of the 3DGS, resulting in inaccurate 3D reconstruction.

<!-- image-->  
Fig. 1. The qualitative results of GeomGS on the KITTI-360 dataset are as follows: (a) 3DGS created with SfM points, (b) 3DGS created with LiDAR points, and (c) GeomGS. The proposed method allows for observing finer details and can address cases where the structure is largely disrupted.

To address the limitations of existing methods, we propose a novel Gaussian Splatting approach called Geometry-Aware Gaussian Splatting (GeomGS). GeomGS generates renderable maps that more accurately reflect real-world scales and structures. To achieve this, we introduce the Geometric Confidence Score (GCS), which evaluates the structural reliability of each Gaussian point. We incorporate probabilistic distance constraints [8] based on the GCS, enabling the generation of a more accurate structure by focusing on higher-confidence Gaussian points while minimizing the influence of points that primarily affect image quality, thus preserving overall rendering quality.

Using these accurate renderable maps and the confidence of each Gaussian, we propose a new localization method that fully utilizes the rendering properties of 3DGS and the precise structure of GeomGS. We use a weighted Iterative Closest Point (ICP) [9] algorithm to align the query LiDAR scan within GeomGS by leveraging GCS values. Then, we optimize the pose by comparing the rendered image at the current pose with the ground truth image. We iteratively update these two methods. Through this integration, we design a robust and accurate localization technique.

We demonstrate the effectiveness of our method on various autonomous driving datasets, producing superior image quality and structurally accurate maps of the environment. Also, our approach shows significant improvements in localization accuracy compared to existing methods. Fig. 1 presents qualitative results based on the initial points and shows that our method captures finer details and prevents scene degradation more effectively.

In conclusion, our proposed method makes the following contributions:

â¢ We introduce GeomGS, which uses a novel Geometric Confidence Score (GCS) and imposes probabilistic distance constraints between Gaussian and LiDAR points to generate geometrically accurate scenes. This enables the reconstruction of a map suitable for localization.

â¢ We propose a novel localization method integrating LiDAR-based localization with image-based pose optimization on a geometrically precise renderable map.

â¢ We comprehensively evaluate image quality, geometric accuracy, and localization performance, showing that our method achieves superior results. It outperforms existing techniques across various autonomous driving datasets.

## II. RELATED WORK

## A. Neural Scene Representation

Recent advancements in novel-view synthesis and high-fidelity rendering have emerged from various approaches. Starting with NeRF [10], improvements have been made in implicit representation through several notable works [11], [12], [13]. An alternative and more advanced approach, 3D Gaussian Splatting (3DGS) [14], allows for real-time, point-based rendering, achieving superior image quality. The optimization of Gaussians for rasterization has led to advanced developments in novel-view synthesis. Following these advancements, recent methods [3], [15], [16] leverage additional information such as image depth or image normals to achieve even higher-quality scene representations. Additionally, emerging works have explored the use of 2D Gaussians instead of 3D Gaussians to improve geometric accuracy in scene construction [17], [18]. Also, several works [19], [20], [21] introduce various appearance models, to construct scenes that are robust to changes in lighting and environmental conditions. Commonly, these approaches use Structure-from-Motion (SfM) [1] to obtain initial points and camera poses. In our method, we improve structural accuracy and image quality by initializing points with LiDAR data and applying constraints and auxiliary approaches to enhance the results.

## B. Scene Reconstruction with Priors

Recently, various NeRF and 3DGS works have introduced different types of priors or directly used point data to enhance scene reconstruction performance. In NeRF-based studies, notable works include S-NeRF [22], Point-NeRF [23], and Points2NeRF [24]. These studies focus on utilizing LiDAR point clouds or projecting LiDAR points onto images to build more accurate scenes. In 3DGS, which can directly manipulate point clouds, several works have been developed without relying on SfM. For instance, some approaches generate initial points using image information [15], while others use NeRF results as priors [25]. Further studies leverage

LiDAR point clouds as initial points or apply simple distance constraints to reconstruct scenes [4], [5], [7]. Additionally, some methods utilize 3D Diffusion Models to generate initial point clouds [26]. Our work uses LiDAR point clouds as the initial points, applying probabilistic distance constraints and evaluating the geometric accuracy. We also introduce a novel method to make our map suitable for localization, addressing a new challenge in this field.

## C. Localization in Radiance Field

Several works have explored localization and pose estimation using images within radiance fields, such as NeRF and 3DGS. In NeRF, numerous studies have demonstrated effective pose estimation by leveraging images [27], particles [28], [29], and different optimization techniques [30], [31]. Similarly to our work, LocNDF [32] defines a Distance Field for efficient localization with LiDAR data. Additionally, a learning-based method [33] is proposed for accurately registering NeRF blocks using surface fields. Recent works [34], [35] with 3DGS, also utilize imagebased pose estimation techniques. Building on these works, we propose a novel approach that fully utilizes rendering properties in radiance fields along with our point-based accurate map representation. This method demonstrates superior localization performance compared to existing approaches.

## III. METHODOLOGY

In this section, we introduce Geometry-Aware Gaussian Splatting (GeomGS). GeomGS integrates LiDAR data into conventional 3D Gaussian Splatting (3DGS) to significantly improve geometric accuracy and localization performance. Unlike existing methods, it efficiently utilizes LiDAR data to create an accurate map suitable for localization based on 3DGS. These improvements are particularly effective in applications such as autonomous driving and robotics. We start with a brief overview of 3DGS. We then introduce our Geometry-Aware Mapping method, which proposes a probabilistic distance loss based on the Geometric Confidence Score (GCS) to enhance structural and localization accuracy. Finally, we present our novel localization method, which leverages the precise geometry and photo-realistic rendering of 3DGS to improve localization in complex environments. Fig. 2 illustrates the overall system architecture, showing how LiDAR data is integrated throughout the process to enhance both mapping and localization.

## A. 3D Gaussian Splatting with Real-Time Rendering

In 3D Gaussian Splatting (3DGS) [14], each point in the scene is represented by 3D Gaussians, defined by its mean $\mu ,$ covariance matrix $\Sigma ,$ color $c ,$ and opacity $\alpha .$ . These Gaussians are flexible 3D primitives that are rendered efficiently by being rasterized into 2D. Each Gaussian primitive is represented as shown in Eq. 1:

$$
G ( x ) = e ^ { - \frac { 1 } { 2 } ( { \bf x } - { \pmb \mu } ) ^ { \top } { \pmb \Sigma } ^ { - 1 } ( { \bf x } - { \pmb \mu } ) }\tag{1}
$$

where x represents the 3D coordinates, and Î£ is the covariance matrix that defines the shape and orientation of the

<!-- image-->  
Fig. 2. Overall system of GeomGS. (a), (b) We start with forward-facing images and poses from a dataset. The accumulated LiDAR points, based on the pose, are used as the initial points. (c) We perform geometrically accurate mapping. The parameters of the Gaussian are defined by mean, quaternion, color, and opacity. Additionally, the Geometrically Consistent Score (GCS) is used to identify points that are more geometrically reliable while remaining close to the given LiDAR points. (d) Our localization module fully utilizes LiDAR-based localization and renderable properties of Gaussians to perform iterative localization processes.

Gaussian. The covariance matrix Î£ can be computed using a rotation matrix R and a scaling matrix S, as shown in Eq. 2:

$$
\pmb { \Sigma } = \mathbf { R } \mathbf { S } \mathbf { S } ^ { \top } \mathbf { R } ^ { \top }\tag{2}
$$

Where S represents the scale of the Gaussian along each axis, and R defines its orientation in 3D space. Each Gaussian has a color c and an opacity $\alpha ,$ which determine the appearance and transparency of the Gaussian during rendering. The final rendered image I can be represented by the following rendering function, as shown in Eq. 3:

$$
\mathcal { T } = \hat { C } ( \mathbf { G } \mid \mathbf { R } _ { c } , \mathbf { t } _ { c } ) = \sum _ { i = 1 } ^ { N } T _ { i } \alpha _ { i } c _ { i }\tag{3}
$$

$$
\alpha _ { i } = ( 1 - \exp ( - \sigma _ { i } \delta _ { i } ) ) , \quad T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { i } ) \qquad \tag{4}
$$

Here, $\hat { C } ( \mathbf { G } \mid \mathbf { R } _ { c } , \mathbf { t } _ { c } )$ represents the rendered image generated from a set of 3D Gaussians G under the camera pose defined by the rotation $\mathbf { R } _ { c }$ and translation $\mathbf { t } _ { c } .$ The function blends the contributions of all Gaussians overlapping each pixel by accounting for both color $c _ { i }$ and opacity $\alpha _ { i } .$ . The transmittance $T _ { i }$ makes each Gaussian visible properly, which helps make the rendered image look more realistic. 3DGS utilizes SfM to generate an initial set of points. After that, it goes through steps like densification and pruning to represent the entire scene.

## B. Geometric Mapping with Geometric Confidence Score

The primary goal of GeomGS is to create a highly accurate structural representation based on 3DGS. To achieve this, we first accumulate LiDAR point cloud data using the pose information provided in the dataset. This accumulated data serves as the initial point cloud. Compared to traditional methods like SfM, this approach can produce a denser point cloud. The accumulated point cloud is created by utilizing the transformation matrix $\mathbf { T } _ { i }$ corresponding to each pose. This transformation matrix converts the LiDAR scan $\mathbf { P } _ { i }$ into the world coordinate system. The transformed LiDAR scans from all poses are then combined into the final accumulated point cloud P, as shown in Eq. 5:

$$
\mathbf { P } = \bigcup _ { i } ( \mathbf { T } _ { i } \mathbf { P } _ { i } )\tag{5}
$$

To improve the structural accuracy of the map, first, we introduce the Geometric Confidence Score (GCS) Î³, a new Gaussian parameter that optimizes the identification of geometrically reliable points. For measuring the geometric confidence of a point, we compute using an asymmetric sigmoid function, $\sigma _ { \mathrm { a s y m } } ( x )$ , as shown in Eq. 6, where k controls the slope, and d determines the dividing point. We set k to 20 and d to 0.9 to distinguish confidence based on the distance between the LiDAR points and Gaussian primitives.

$$
\sigma _ { \mathrm { a s y m } } ( x ) = \frac { 1 } { 1 + e ^ { k ( x - d ) } } \in [ 0 , 1 ]\tag{6}
$$

We optimize the GCS $\gamma$ using the function in Eq. 6. To continuously quantify it during the optimization process, we introduce a new loss term, ${ \mathcal { L } } _ { \mathrm { g e o m } } .$ , as shown in Eq. 8. This GCS is fully utilized in the process of creating a more precise structure.

$$
d _ { i } = \operatorname* { m i n } _ { \mathbf { p } \in \mathbf { P } } \| \mathbf { g } _ { i } - \mathbf { p } \| ^ { 2 } , \quad \mathbf { g } _ { i } \in \mathbf { G }\tag{7}
$$

$$
\mathcal { L } _ { \mathrm { g e o m } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \gamma _ { i } - \sigma _ { \mathrm { a s y m } } ( d _ { i } ) ) ^ { 2 } , \quad \gamma _ { i } \in ( 0 , 1 )\tag{8}
$$

Where the term $d _ { i }$ as shown in Eq. 7 represents the distance between the G and its closest accumulated LiDAR point p in the P for all N Gaussian points.

Next, to construct an accurate structural map, we impose probabilistic distance constraints on the Gaussian primitives based on the GCS Î³. Rather than simply minimizing the Euclidean distance, we apply more robust constraints by incorporating GCS, as shown in Eq. 9:

$$
\mathcal { L } _ { p r o b } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \ln ( 1 - \gamma _ { i } ) + \frac { d _ { i } } { ( 1 - \gamma _ { i } ) } \right)\tag{9}
$$

This probabilistic distance optimization method assigns more weight to points that have higher GCS. Focusing more on structurally important points when generating an accurate map, reduces the influence of points that are not structurally crucial (e.g., sky, tall buildings), but are important for image rendering. This approach continuously generates and optimizes an accurate structure without compromising image quality.

In addition, we apply a scale loss [36] to prevent the overlapping of Gaussian points, a perceptual loss [37] to preserve feature-level details, and an appearance model [21] to enhance robustness against brightness variations. Similar to [21], $\mathcal { T } _ { a }$ is the image from the appearance model, and $\mathcal { T } _ { r }$ is the rendered image. The final loss design, which includes these components, is shown in Eq. 10 and Eq. 11:

$$
\mathcal { L } _ { r g b } = ( 1 - \lambda _ { \mathrm { r g b } } ) \mathcal { L } _ { 1 } ( \mathcal { T } _ { a } , \mathcal { T } _ { g t } ) + \lambda _ { \mathrm { r g b } } \mathcal { L } _ { \mathrm { D - S S I M } } ( \mathcal { T } _ { r } , \mathcal { T } _ { g t } )\tag{10}
$$

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { g e o m } } { \mathcal { L } } _ { \mathrm { g e o m } } + \lambda _ { \mathrm { p r o b } } { \mathcal { L } } _ { \mathrm { p r o b } } } \\ & { ~ + ~ \lambda _ { \mathrm { s c a l e } } { \mathcal { L } } _ { \mathrm { s c a l e } } + \lambda _ { \mathrm { p e r c } } { \mathcal { L } } _ { \mathrm { p e r c } } } \end{array}\tag{11}
$$

The values of the hyper-parameter Î» used in the loss function are set as follows: $\lambda _ { \mathrm { r g b } } = 0 . 2 , \lambda _ { \mathrm { g e o m } } = 0 . 1 , \lambda _ { \mathrm { p r o b } } = 0 . 1$ $\lambda _ { \mathrm { s c a l e } } = 1 0 0 . 0$ , and $\lambda _ { \mathrm { p e r c } } = 0 . 5$

## C. Gaussian Splatting-based Unified Localization

We propose a novel localization method based on a highly accurate map generated from GeomGS. Existing pose optimization techniques, such as iNeRF [27] estimate the pose by minimizing the loss between the rendered output from the current pose and the ground truth image over several iterations. While these methods have demonstrated effective pose estimation, they do not inherently operate within a coordinate system that reflects actual space or accounts for actual scale.

In contrast, our approach leverages the advantages of GeomGS, which allows the use of renderable properties, an accurate Gaussian map, and its confidence scores. The first key idea of our method is to apply Iterative Closest Point (ICP) [9] between the Gaussian points and the query LiDAR scan. Afterward, we iteratively refine the pose by comparing the rendered image at the pose R, t, obtained from the ICP results, with the ground truth image. Similar to the approach in iNeRF [27], as shown in Eq. 12 and Eq. 13. This refined pose is applied to the LiDAR scan $\mathbf { Q } _ { p }$ again. We repeat these two processes iteratively for a given number of iterations to refine the pose and align the points.

One of the most significant aspects of our approach is the use of the Geometric Confidence Score (GCS), developed within GeomGS, to perform Weighted ICP. GCS assigns reliability scores to each point, and these scores are used as weights for each point pair in the ICP process. This results in more accurate and robust pose estimation. The core of Weighted ICP is the use of a weight matrix in Eq. 14, which influences the transformation calculation between the source and target point clouds. The relationship between the weighted source points S and target points T is represented by the matrix H in Eq. 15, where Â¯s and Â¯t are the centroids (mean points) of the source and target point clouds, respectively. Using Singular Value Decomposition (SVD), the rotation matrix $\mathbf { R } _ { \mathrm { i c p } }$ and translation vector $\mathbf { t } _ { \mathrm { i c p } }$ are computed, as shown in Eq. 16:

$$
\Delta \mathbf { R } , \Delta \mathbf { t } = \underset { \Delta \mathbf { R } , \Delta \mathbf { t } } { \arg \operatorname* { m i n } } \left( \mathcal { L } \left( \hat { C } ( \mathbf { G } \mid \mathbf { R } , \mathbf { t } ) , \mathcal { T } _ { g t } \right) \right)\tag{12}
$$

$$
{ \bf Q } _ { p } ^ { \prime } = \Delta { \bf R } { \bf Q } _ { p } + \Delta { \bf t }\tag{13}
$$

$$
\mathbf { W } = \mathrm { d i a g } ( \gamma )\tag{14}
$$

$$
\mathbf { H } = ( \mathbf { S } - { \bar { \mathbf { s } } } ) ^ { \top } \mathbf { W } ( \mathbf { T } - { \bar { \mathbf { t } } } ) , \quad \mathbf { S } \mathbf { V } \mathbf { D } ( \mathbf { H } ) = \mathbf { U } \mathbf { S } \mathbf { V } ^ { \top }\tag{15}
$$

$$
\mathbf { R } _ { i c p } = \mathbf { V } \mathbf { U } ^ { \top } , \quad \mathbf { t } _ { i c p } = \bar { \mathbf { t } } - \mathbf { R } _ { i c p } \bar { \mathbf { s } }\tag{16}
$$

Weighted ICP and image refinement are combined in a way that they support each other as shown in Fig. 2, helping to overcome each methodâs weaknesses. If ICP fails to align the points correctly, image refinement can fix the pose or prevent the process from failing with pixel-level comparison. Likewise, if image refinement struggles with correcting large errors, ICP can help correct them based on accurately structured Gaussian points. This approach takes full advantage of both image rendering and LiDAR-based localization, ensuring reliable localization even in challenging environments.

## IV. EXPERIMENTS

## A. Experimental Setup

We designed our experiments to demonstrate the effectiveness of our system by evaluating the state-of-the-art (i) image rendering quality via GeomGS, (ii) the accuracy of geometric representations under the proposed constraints, which have not been previously evaluated, and (iii) the feasibility and accuracy of our proposed localization method in GeomGS. We selected 100 consecutive images from KITTI [38] and KITTI-360 [39] datasets, representing scenes that cover approximately 100 meters in actual space. Unlike existing methods that rely on pose information obtained from SfM, we used pose data provided by KITTI and KITTI-360. For a baseline comparison, we selected 3DGS, which has achieved SOTA in novel view synthesis.

TABLE I  
QUANTITAIVE RESULTS ON KITTI-360 AND KITTI
<table><tr><td rowspan="2"></td><td rowspan="2">Initial Points</td><td colspan="3">KITTI-360</td><td colspan="3">KITTI</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>Scaffold-GS [2]</td><td>SfM</td><td>18.9783</td><td>0.7678</td><td>0.3072</td><td>18.0842</td><td>0.6679</td><td>0.3214</td></tr><tr><td>Scaffold-GS [2]</td><td>LiDAR</td><td>20.5806</td><td>0.7796</td><td>0.2923</td><td>18.1341</td><td>0.6429</td><td>0.3384</td></tr><tr><td>GaussianSurfels [18]</td><td>SfM</td><td>21.0695</td><td>0.7765</td><td>0.3569</td><td>18.4256</td><td>0.5727</td><td>0.4791</td></tr><tr><td>GaussianSurfels [18]</td><td>LiDAR</td><td>23.0385</td><td>0.8291</td><td>0.2844</td><td>18.3671</td><td>0.5678</td><td>0.4796</td></tr><tr><td>2DGS [17]</td><td>SfM</td><td>23.3374</td><td>0.8388</td><td>0.2536</td><td>21.6865</td><td>0.7584</td><td>0.2724</td></tr><tr><td>2DGS [17]</td><td>LiDAR</td><td>23.6735</td><td>0.8437</td><td>0.2459</td><td>21.5642</td><td>0.7489</td><td>0.2845</td></tr><tr><td>3DGS [14]</td><td>SfM</td><td>23.1327</td><td>0.8231</td><td>0.2435</td><td>21.5168</td><td>0.7611</td><td>0.2316</td></tr><tr><td>3DGS [14]</td><td>LiDAR</td><td>23.5725</td><td>0.8332</td><td>0.2259</td><td>21.7329</td><td>0.7622</td><td>0.2368</td></tr><tr><td>Ours-S</td><td>LiDAR</td><td>24.2963</td><td>0.8533</td><td>0.2033</td><td>21.8820</td><td>0.7590</td><td>0.2367</td></tr><tr><td>Ours-P</td><td>LiDAR</td><td>24.2981</td><td>0.8555</td><td>0.1903</td><td>21.9829</td><td>0.7646</td><td>0.2329</td></tr></table>

Evaluate the quality of the rendered image in a conventional test scene.

## B. Image Quality Validation

We evaluated our method using the same approach as 3DGS, with standard metrics such as PSNR, SSIM [40], and LPIPS [41], testing on scenes sampled every 8 frames. However, unlike 3DGS, we set the initial position learning rate to 1.6e-5, instead of 1.6e-4. Additionally, we compared our approach with recent methods like 2DGS [17], Gaussian Surfels [18], and Scaffold-GS [2]. For a fair comparison, we used ground truth (GT) poses to evaluate these methods, testing them on both the initial SfM points and initial LiDAR points. For comparison with SfM points, we used COLMAPâs [1] triangulation to scale the SfM points to the actual size. This process required known camera parameters, which we extracted from the datasets.

Table I shows that using LiDAR points as initial points improves performance in most methods. While image quality often degrades in complex scenes, initializing with dense LiDAR points helps reduce this issue in most methods. However, the most significant performance improvement is achieved by applying our method, which includes the constraint between LiDAR points and Gaussian points.

Ours-S applies a simple Euclidean distance loss to our method, whereas Ours-P employs a probabilistic distance loss with GCS. The probabilistic distance loss can also achieve structural characteristics without degrading image performance. Fig. 1 and Fig. 3 illustrate that our method captures fine details and structural accuracy more effectively across various scenes. It also successfully addresses shape distortion in novel views.

TABLE II  
GEOMETRIC PERFORMACE ON KITTI-360
<table><tr><td></td><td>Initial Points</td><td>F-Score â @0.1</td><td>F-Score â @0.2</td><td>F-Score â @1.0</td><td>CD â</td></tr><tr><td>3DGS</td><td>SfM</td><td>0.4220</td><td>0.4912</td><td>0.6393</td><td>261.1804</td></tr><tr><td>3DGS</td><td>LiDAR</td><td>0.5639</td><td>0.6456</td><td>0.8132</td><td>6.6338</td></tr><tr><td>2DGS</td><td>SfM</td><td>0.4243</td><td>0.4953</td><td>0.6589</td><td>183.5640</td></tr><tr><td>2DGS</td><td>LiDAR</td><td>0.4783</td><td>0.5550</td><td>0.7153</td><td>91.7571</td></tr><tr><td>Ours-S</td><td>LiDAR</td><td>0.8230</td><td>0.8822</td><td>0.9534</td><td>2.8688</td></tr><tr><td>Ours-P</td><td>LiDAR</td><td>0.8948</td><td>0.9267</td><td>0.9629</td><td>2.6709</td></tr></table>

## C. Geometric Quality Validation

We impose initial LiDAR Points and apply constraints between Gaussian points and LiDAR points to enhance the accuracy of reconstruction. To evaluate the structural improvement over existing methods, we measure the similarity between the two point clouds, following the approach in Points2NeRF [24]. Specifically, we calculate Chamfer Distance (CD), as shown in Eq. 17, and the F-Score, as shown in Eq. 19. We then compare these metrics between the generated Gaussians G and the accumulated LiDAR Points P.

It is important to note that the reconstructed Gaussian points cannot perfectly match the initial points, especially in areas where LiDAR data lacks coverage, such as the sky or tall buildings. Nevertheless, we observe significant improvements in both CD and F-Score, as shown in Table II. Simply replacing the initial points with Gaussians leads to better results, but our approach using constraints between the points achieves superior reconstruction accuracy. Similarly, our approach using probabilistic distance loss (Ours-P) with GCS achieves an outstanding structure.

$$
\begin{array} { l } { { \displaystyle C D ( { \bf P _ { 1 } } , { \bf P _ { 2 } } ) = \frac { 1 } { | { \bf P _ { 1 } } | } \sum _ { { \bf p \in { \bf P _ { 1 } } } } \operatorname* { m i n } _ { { \bf q \in { \bf P _ { 2 } } } } \left\| { \bf p - q } \right\| ^ { 2 } } \ ~ } \\ { { \displaystyle ~ + \frac { 1 } { | { \bf P _ { 2 } } | } \sum _ { { \bf q \in { \bf P _ { 2 } } } } \operatorname* { m i n } _ { { \bf p \in { \bf P _ { 1 } } } } \left\| { \bf q - p } \right\| ^ { 2 } } } \end{array}\tag{17}
$$

$$
\mathrm { p r e c i s i o n } _ { 1 } = \frac { 1 } { | { \bf P } _ { 1 } | } \sum _ { { \bf p } \in { \bf P } _ { 1 } } \mathbb { I } \left( \operatorname* { m i n } _ { { \bf q } \in { \bf P } _ { 2 } } \| { \bf p } - { \bf q } \| ^ { 2 } < \tau \right) ,
$$

$$
\mathrm { p r e c i s i o n } _ { 2 } = \frac { 1 } { | { \bf P } _ { 2 } | } \sum _ { { \bf q } \in { \bf P } _ { 2 } } \mathrm { I } \left( \operatorname* { m i n } _ { { \bf p } \in { \bf P } _ { 1 } } \| { \bf q } - { \bf p } \| ^ { 2 } < \tau \right)\tag{18}
$$

$$
{ \mathrm { F \mathrm { - } s c o r e } } = { \frac { 2 \times { \mathrm { p r e c i s i o n } } _ { 1 } \times { \mathrm { p r e c i s i o n } } _ { 2 } } { \mathrm { p r e c i s i o n } _ { 1 } + { \mathrm { p r e c i s i o n } } _ { 2 } } }\tag{19}
$$

## D. 3D Localization Performance

We evaluate the 3D localization performance on our structurally accurate map. For the evaluation, we selected test cases from every 10th sequence out of 100 sequences and calculated the average localization performance. And we intentionally applied large initial errors to poses. Our method performs localization by iteratively leveraging the properties of the structurally accurate map and its image rendering capabilities. In particular, we utilize the Geometric Confidence Score (GCS) as the weight in the Weighted ICP (WICP) algorithm to enhance its robustness and accuracy. We compare our approach with existing methods such as ICP [9], iNeRF [27], and our WICP. Our approach uses 20 iterations in total. In each iteration of our method, WICP is applied once, followed by image refinement over 20 steps. To ensure a fair comparison, the same number of iterations is applied to the other methods. WICP generally performs well on our map, and overall, our proposed method shows strong performance. We evaluate both rotation and translation errors, and as shown in Table III, our method typically produces better results. Fig. 4 illustrates the error reduction across iterations, where our method generally demonstrates superior performances.

<!-- image-->

<!-- image-->  
(a) KITTI-360

<!-- image-->

<!-- image-->

<!-- image-->  
(b) KITTI

<!-- image-->  
Fig. 3. Qualitative comparison of GeomGS and 3DGS in (a) KITTI-360 & (b) KITTI datasets. Patches represent visually distinct regions, highlighting fine details and geometric variations. Our method performs better in various scenarios by incorporating finer details, improving geometric representation, and enhancing overall image quality. The notation adjacent to 3DGS denotes which specific initial point was utilized in the process.

TABLE III  
COMPARISON OF LOCALIZATION METHODS BASED ON ROTATION AND TRANSLATION ERRORS
<table><tr><td>Initial Error</td><td>ICP (SfM)</td><td>ICP</td><td>iNeRF [27]</td><td>WICP</td><td>Ours</td></tr><tr><td>20.0 / 2.0</td><td>2.8548 / 1.7532</td><td>0.7388 / 0.9853</td><td>15.4341 / 1.2635</td><td>4.0217 / 1.0654</td><td>0.8635 / 0.5396</td></tr><tr><td>20.0 / 3.0</td><td>6.7209 / 6.4538</td><td>6.2766 / 5.2994</td><td>22.5281 / 3.2002</td><td>3.9178 / 1.1412</td><td>3.8972 / 1.5464</td></tr><tr><td>30.0 / 2.0</td><td>11.9116 / 24.4708</td><td>11.1796 / 7.4975</td><td>37.1290 / 2.3754</td><td>6.9021 / 0.9786</td><td>7.0102 / 1.1784</td></tr><tr><td>30.0 / 3.0</td><td>15.0402 / 5.7775</td><td>16.0031 / 4.6866</td><td>34.6142 / 3.2958</td><td>9.8547 / 1.6552</td><td>3.2957 / 2.3108</td></tr><tr><td>40.0 / 4.0</td><td>22.2579 / 8.9544</td><td>20.4090 / 9.3045</td><td>45.4746 / 3.8619</td><td>15.2049 / 3.1879</td><td>14.6590 / 2.7048</td></tr></table>

The values in the table are presented as ârotation error [â¦] / translation error [m]â

(a)  
<!-- image-->

<!-- image-->

(b)  
<!-- image-->

<!-- image-->  
Fig. 4. Comparison of ICP [9], WICP, Image Refinement (iNeRF [27]), and Ours per Iteration. (a) Initial error : 20.0â¦ / 2.0m, (b) Initial error : 25.0â¦ / 5.0m

## V. CONCLUSIONS

We propose GeomGS, a method for representing environments with accurate structures and enabling precise localization. We introduce a Geometric Confidence Score (GCS) to identify the geometric reliability of each point. Using a probabilistic distance optimization approach based on GCS, we generate more precise structures without degrading image quality. For localization, we present a novel approach that leverages GCS, LiDAR-based localization, and 3DGS rendering within our accurate map. We evaluate both qualitatively and quantitatively how our maps preserve structural accuracy without compromising image quality and analyze localization performance using these maps.

[1] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â Â¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[2] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 654â20 664.

[3] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, âDngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 775â20 785.

[4] Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and S. Peng, âStreet gaussians: Modeling dynamic urban scenes with gaussian splatting,â in ECCV, 2024.

[5] S. Hwang, M.-J. Kim, T. Kang, J. Kang, and J. Choo, âVegs: View extrapolation of urban scenes in 3d gaussian splatting using learned priors,â in ECCV, 2024.

[6] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, âColmap-free 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 20 796â20 805.

[7] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, âDrivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 634â21 643.

[8] M. Tian, L. Pan, M. H. Ang, and G. H. Lee, âRobust 6d object pose estimation by learning rgb-d features,â in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 6218â6224.

[9] P. Besl and N. D. McKay, âA method for registration of 3-d shapes,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 14, no. 2, pp. 239â256, 1992.

[10] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â in ECCV, 2020.

[11] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â CVPR, 2022.

[12] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman., âZip-nerf: Anti-aliased grid-based neural radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 19 697â19 705.

[13] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan, J. T. Barron, and H. Kretzschmar, âBlock-nerf: Scalable large scene neural view synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8248â8258.

[14] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[15] M. Turkulainen, X. Ren, I. Melekhov, O. Seiskari, E. Rahtu, and J. Kannala, âDn-splatter: Depth and normal priors for gaussian splatting and meshing,â 2024.

[16] K. Cheng, X. Long, K. Yang, Y. Yao, W. Yin, Y. Ma, W. Wang, and X. Chen, âGaussianpro: 3d gaussian splatting with progressive propagation,â in Forty-first International Conference on Machine Learning, 2024.

[17] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024.

[18] P. Dai, J. Xu, W. Xie, X. Liu, H. Wang, and W. Xu, âHigh-quality surface reconstruction using gaussian surfels,â in ACM SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024.

[19] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, âNerf in the wild: Neural radiance fields for unconstrained photo collections,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 7210â7219.

[20] J. Kulhanek, S. Peng, Z. Kukelova, M. Pollefeys, and T. Sattler, âWildgaussians: 3d gaussian splatting in the wild,â arXiv preprint arXiv:2407.08447, 2024.

[21] J. Lin, Z. Li, X. Tang, J. Liu, S. Liu, J. Liu, Y. Lu, X. Wu, S. Xu, Y. Yan, et al., âVastgaussian: Vast 3d gaussians for large scene reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5166â5175.

[22] Z. Xie, J. Zhang, W. Li, F. Zhang, and L. Zhang, âS-nerf: Neural radiance fields for street views,â in ICLR 2023, 2023.

[23] Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, and U. Neumann, âPoint-nerf: Point-based neural radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5438â5448.

[24] D. Zimny, J. Waczynska, T. Trzci Â´ nski, and P. Spurek, âPoints2nerf: Â´ Generating neural radiance fields from 3d point cloud,â Pattern Recognition Letters, vol. 185, pp. 8â14, 2024.

[25] M. Niemeyer, F. Manhardt, M.-J. Rakotosaona, M. Oechsle, D. Duckworth, R. Gosula, K. Tateno, J. Bates, D. Kaeser, and F. Tombari, âRadsplat: Radiance field-informed gaussian splatting for robust realtime rendering with 900+ fps,â arXiv.org, 2024.

[26] T. Yi, J. Fang, J. Wang, G. Wu, L. Xie, X. Zhang, W. Liu, Q. Tian, and X. Wang, âGaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models,â in CVPR, 2024.

[27] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.- Y. Lin, âiNeRF: Inverting neural radiance fields for pose estimation,â in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021.

[28] D. Maggio, M. Abate, J. Shi, C. Mario, and L. Carlone, âLoc-nerf: Monte carlo localization using neural radiance fields,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 4018â4025.

[29] M. Kong, S. Lee, J. Lee, and E. Kim, âFast global localization on neural radiance field,â arXiv preprint arXiv:2406.12202, 2024.

[30] Y. Lin, T. Muller, J. Tremblay, B. Wen, S. Tyree, A. Evans, P. A. Â¨ Vela, and S. Birchfield, âParallel inversion of neural radiance fields for robust pose estimation,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 9377â9384.

[31] W. Bian, Z. Wang, K. Li, J. Bian, and V. A. Prisacariu, âNope-nerf: Optimising neural radiance field with no pose prior,â 2023.

[32] L. Wiesmann, T. Guadagnino, I. Vizzo, N. Zimmerman, Y. Pan, H. Kuang, J. Behley, and C. Stachniss, âLocndf: Neural distance field mapping for robot localization,â IEEE Robotics and Automation Letters, 2023.

[33] Y. Chen and G. H. Lee, âDreg-nerf: Deep registration for neural radiance fields,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 22 703â22 713.

[34] Y. Sun, X. Wang, Y. Zhang, J. Zhang, C. Jiang, Y. Guo, and F. Wang, âicomma: Inverting 3d gaussians splatting for camera pose estimation via comparing and matching,â arXiv preprint arXiv:2312.09031, 2023.

[35] M. Bortolon, T. Tsesmelis, S. James, F. Poiesi, and A. Del Bue, â6dgs: 6d pose estimation from a single image and a 3d gaussian splatting model,â in ECCV, 2024.

[36] S. Lombardi, T. Simon, G. Schwartz, M. Zollhoefer, Y. Sheikh, and J. Saragih, âMixture of volumetric primitives for efficient neural rendering,â ACM Transactions on Graphics (ToG), vol. 40, no. 4, pp. 1â13, 2021.

[37] J. Johnson, A. Alahi, and L. Fei-Fei, âPerceptual losses for real-time style transfer and super-resolution,â in Computer VisionâECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11- 14, 2016, Proceedings, Part II 14. Springer, 2016, pp. 694â711.

[38] A. Geiger, P. Lenz, and R. Urtasun, âAre we ready for autonomous driving? the kitti vision benchmark suite,â in Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

[39] Y. Liao, J. Xie, and A. Geiger, âKITTI-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d,â Pattern Analysis and Machine Intelligence (PAMI), 2022.

[40] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, 2004.

[41] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.