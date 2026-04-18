# VAD-GS: Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes

Yikang Zhang, Rui Fan

Tongji University

## Abstract

3D Gaussian splatting (3DGS) has demonstrated impressive performance in synthesizing high-fidelity novel views. Nonetheless, its effectiveness critically depends on the quality of the initialized point cloud. Specifically, achieving uniform and complete point coverage over the underlying scene structure requires overlapping observation frustums, an assumption that is often violated in unbounded, dynamic urban environments. Training Gaussian models with partially initialized point clouds often leads to distortions and artifacts, as camera rays may fail to intersect valid surfaces, resulting in incorrect gradient propagation to Gaussian primitives associated with occluded or invisible geometry. Additionally, existing densification strategies simply clone and split Gaussian primitives from existing ones, incapable of reconstructing missing structures. To address these limitations, we propose VAD-GS, a 3DGS framework tailored for geometry recovery in challenging urban scenes. Our method identifies unreliable geometry structures via voxel-based visibility reasoning, selects informative supporting views through diversityaware view selection, and recovers missing structures via patch matching-based multi-view stereo reconstruction. This design enables the generation of new Gaussian primitives guided by reliable geometric priors, even in regions lacking initial points. Extensive experiments on the Waymo and nuScenes datasets demonstrate that VAD-GS outperforms state-of-the-art 3DGS approaches and significantly improves the quality of reconstructed geometry for both static and dynamic objects. Source code will be released upon publication.

## Introduction

Realistic simulation is critical for the development and validation of autonomous driving systems (Bao et al. 2025). Traditional simulators rely on handcrafted assets, inherently limiting scene scalability and diversity (Dosovitskiy et al. 2017). Recent advances in neural scene representations have enabled data-driven, photorealistic novel view synthesis (NVS), which provides a more efficient and scalable alternative. Specifically, neural radiance field (NeRF)-based approaches (Mildenhall et al. 2021) represent scenes using neural networks and achieve high-fidelity 3D reconstruction. Nevertheless, volume rendering is typically computationally intensive, thereby limiting the practical applicability of NeRF and its variants. To enable real-time rendering, 3D Gaussian splatting (3DGS) explicitly represents scenes as

min max Depth Reference

Training Views

Test View

<!-- image-->

<!-- image-->  
Figure 1: A comparison between VAD-GS and Street-Gaussians. While both methods achieve comparable rendering quality, VAD-GS demonstrates superior recovery of incomplete or unreliable scene geometry, as evidenced by notable improvements in the rendered depth and normal maps.

anisotropic 3D Gaussian primitives with learnable geometry and appearance attributes. These primitives are jointly optimized to align with the underlying scene structure by enforcing photometric consistency across images captured from multiple views. Building on its impressive performance in object-level reconstruction, recent extensions of 3DGS to large-scale and dynamic environments (Kerbl et al. 2024; Wu et al. 2024) have demonstrated strong potential for reconstructing complex urban scenes.

Despite these advances, recovering complete and reliable geometry for unbounded environments from sparse observations remains a major challenge. In 3DGS, scene completeness and texture details are typically enhanced by splitting or cloning existing Gaussian primitives, which are initialized from point clouds obtained through structure from motion (SfM) or LiDAR scan accumulation (Bao et al. 2025). However, autonomous driving datasets present inherent limitations: (1) Unlike scene-centric reconstruction settings, multiple cameras mounted on a vehicle capture outward-facing views with limited overlap (typically less than 15%) (Wei, Li et al. 2025), which makes stereo matching between adjacent, synchronous images unreliable; (2) Although multiview stereo (MVS) methods can recover static scene structure from asynchronous video frames, they are ineffective in reconstructing dynamic objects; (3) Despite the incorporation of LiDAR points to enhance geometric consistency in pioneering studies (Yan et al. 2024; Zhou et al. 2024), substantial blind spots in scene structure persist due to the limited field of view. Consider a typical scenario where a low-texture traffic sign is positioned too high to be captured by LiDAR. It may also lack sufficient visual features for reliable correspondence matching across images. In such cases, insufficient visual cues make both geometry and appearance reconstruction particularly challenging. Furthermore, during Gaussian training, photometric errors caused by missing geometry are erroneously attributed to background structures, such as trees or buildings behind the sign. Consequently, gradient-based splitting and cloning operations may inadvertently be applied to invisible Gaussian primitives. Although this training process may improve rendering quality for specified views, it distorts the underlying scene geometry and ultimately degrades generalization to unseen perspectives.

Several recent studies have focused on enhancing scene completeness within the original 3DGS framework. For instance, GeoTexDensifier (Jiang et al. 2024) incorporates additional depth and normal priors to guide the splitting process for improved surface alignment, whereas DNGaussian (Li et al. 2024) identifies missing geometry by performing global-local normalization between the rendered depth and that estimated using DPT (Ranftl et al. 2021). Nonetheless, these methods are confined to regions with existing Gaussian primitives and are incapable of handling uninitialized areas. To overcome this drawback, GaussianPro (Cheng et al. 2024) introduces a patch matching-based geometry completion strategy, which leverages stereo constraints from a set of images with precomputed camera poses to generate additional point clouds independent of the rendering process. While GaussianPro greatly enhances geometry recovery, it remains limited to static scenes and struggles to handle dynamic objects. Moreover, it relies solely on adjacent frames captured using a single camera, thereby missing long-range temporal dependencies and cross-camera visual cues.

To address the aforementioned challenges, this paper introduces a visibility-aware densification framework for 3D Gaussian splatting (VAD-GS) tailored for dynamic, unbounded urban environments. Unlike previous approaches that passively react to photometric errors, VAD-GS actively evaluates structural completeness and selectively reconstructs incomplete regions by leveraging views that provide the most reliable stereo geometry. Specifically, we introduce a voxel-based object surface visibility reasoning approach that provides geometric priors for both static backgrounds and dynamic objects. Each voxel aggregates view-dependent visibility information of the corresponding 3D points, thereby enabling occlusion-aware modeling through depth rasterization with z-buffering. Furthermore, we propose a diversity-aware view sampling strategy that selects informative supporting views for each reference view, aiming to balance view frustum overlap and triangulation quality. The selected views are processed using a patch matching-based MVS algorithm to extract depth and normal information, which serves as reliable geometric priors for new Gaussian primitive initialization and scene consistency enforcement. VAD-GS is evaluated on the Waymo Open dataset (Sun et al. 2020) and the nuScenes dataset (Caesar et al. 2020), both of which contain complex urban dynamics and sparse multi-view observations. Extensive experiments demonstrate that VAD-GS achieves state-of-the-art (SoTA) rendering quality while producing more consistent geometry with fewer artifacts compared to previous SoTA methods (see Fig. 1). The main contributions of this study are summarized as follows:

â¢ A novel Gaussian splatting framework tailored for dynamic urban scenes, which actively completes missing geometry using multi-camera, cross-frame observations.

â¢ A voxel-based surface visibility reasoning approach that identifies unreliable static and dynamic object geometry.

â¢ A diversity-aware sampling strategy that improves MVS reconstruction quality by optimizing supporting views.

â¢ An extension of MVS reconstruction to dynamic, multicamera driving scenarios, enabling both Gaussian densification and scene consistency enforcement.

## Related Work

## Novel View Synthesis

NVS aims to generate photorealistic images of objects or scenes from previously unseen viewpoints, without the explicit modeling of 3D geometry or illumination. NeRF (Mildenhall et al. 2021), a pioneering work in this field, represents 3D scenes as learnable volumetric density fields, parameterized by a large multi-layer perceptron (MLP). Subsequent studies have primarily focused on improving both rendering quality and computational efficiency. For example, Instant-NGP (Muller et al. 2022) introduces a Â¨ multi-resolution hash encoding scheme that adaptively allocates higher representational capacity to geometrically complex regions, thereby significantly improving rendering efficiency. Mip-NeRF (Barron et al. 2021) improves point sampling in ray marching to mitigate aliasing artifacts caused by resolution mismatches, while its extension Mip-NeRF 360 (Barron et al. 2022) further adapts the approach to handle unbounded scenes. However, the substantial computational cost of volume rendering remains a major barrier to the practical deployment of NeRF models in real-world applications.

Recent 3DGS approaches (Kerbl et al. 2023) have introduced an alternative NVS paradigm, enabling real-time rendering of large-scale scenes. By projecting anisotropic 3D

<!-- image-->  
Figure 2: VAD-GS pipeline. For each static or dynamic instance with incomplete geometry, VAD-GS first performs voxel-based visibility reasoning to identify a set of potential observation views. It then incrementally selects diverse supporting views to perform MVS reconstruction. The resulting geometric priors are subsequently used for Gaussian densification and optimization.

Gaussian ellipsoids onto the 2D image plane using splattingbased rasterization and computing pixel colors through depth sorting and Î±-blending, these methods effectively circumvent the computational overhead of ray marching. Since then, several studies have extended 3DGS to dynamic urban scenes. Notably, StreetGaussians (Yan et al. 2024) models dynamic vehicles in the foreground as rigid groups of Gaussian primitives and employs a 4D spherical harmonics model to capture appearance variation over time. Similarly, DrivingGaussian (Zhou et al. 2024) uses a composite dynamic Gaussian graph to model multiple dynamic objects, while OmniRe (Chen et al. 2025) incorporates skinned multi-person linear Gaussians to represent non-rigid entities such as pedestrians and cyclists. Despite the appealing results achieved by this paradigm, two major limitations persist: (1) new Gaussian primitives are typically generated by splitting or cloning existing ones, making reconstruction quality highly dependent on the accuracy and completeness of the initialized point clouds; (2) recovering missing geometry based solely on photometric errors is inherently challenging, as gradient updates may be incorrectly propagated to view-proximal yet geometrically unrelated Gaussians, leading to the distortion of neighboring Gaussians and ultimately degrading the overall scene geometry. To address these issues, we explore the incorporation of additional geometric cues, particularly MVS constraints, to obtain more reliable structural information beyond gradient propagation.

## Multi-View Stereo

MVS is a fundamental computer vision technique that reconstructs dense 3D scene geometry from a set of images with known intrinsic parameters (Wang et al. 2024). Online MVS methods typically select keyframes from lowresolution video streams to perform real-time camera pose estimation and point cloud generation (Dai et al. 2017). In contrast, offline MVS approaches, typically following the SfM pipeline (Schonberger and Frahm 2016; AanÃ¦s et al. 2016; Schops et al. 2017), aim for high-resolution, largescale scene reconstruction at the expense of greater computational complexity. Despite their differing objectives, both pipelines face critical challenges that directly impact reconstruction quality, particularly in view selection and depth estimation. For view selection, GP-MVS (Hou, Kannala, and Solin 2019) employs a heuristic pose-distance measure function to select informative keyframes, while MVSNet (Yao et al. 2018) introduces a score function that ranks neighboring views based on frustum overlap. In terms of depth estimation, plane sweeping-based methods discretize depth candidates to construct cost volumes and measure feature similarity across warped views, thereby favoring reconstructions with higher resolution (Cheng et al. 2020; Yang et al. 2022). In contrast, patch matching-based approaches achieve high efficiency by randomly sampling depth guesses for individual pixels and iteratively propagating plausible estimates from neighboring pixels (Xu and Tao 2020; Wang et al. 2021). In this work, we exploit MVS consistency not only to guide the densification of Gaussian primitives for each object but also to provide geometric supervision that complements photometric gradient-based optimization.

## Preliminaries

3DGS-based approaches represent a scene using a set of anisotropic 3D Gaussian primitives (Kerbl et al. 2023). Each primitive x is modeled by a Gaussian distribution, defined as follows:

$$
G ( \pmb { x } ) = \exp \left( - \frac { 1 } { 2 } ( \pmb { x } - \pmb { \mu } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \pmb { x } - \pmb { \mu } ) \right) ,\tag{1}
$$

where $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ denotes the primitive center, and the covariance matrix Î£ is parameterized using a scaling factor and a rotation quaternion. Unlike volumetric representations such as NeRF, 3DGS avoids the computational overhead of volumetric ray marching by adopting a tile-based rasterization pipeline. The color C at each pixel is obtained by compositing the overlapping Gaussians along the pixel ray via frontto-back Î±-blending, as expressed as follows:

<!-- image-->  
Figure 3: Voxel-based visibility reasoning. (a) Red points are visible, whereas blue points, captured from other views, are invisible in the reference view. (b) The invisibility of blue points may result from occlusions or insufficient sampling rays in the reference view. (c) Rasterizing the distances and indices of visible voxels (in green) yields dense depth maps and accurate pixel-voxel mapping.

$$
C = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $\mathcal { N }$ denotes the set of Gaussian primitives intersected by the pixel ray, while $c _ { i }$ and $\alpha _ { i }$ represent the color and opacity of the i-th primitive, respectively. To improve scene geometry representation, 3DGS adopts a densification strategy by splitting existing Gaussians with large covariance into smaller primitives, thereby improving coverage and structural fidelity.

## Methodology

As illustrated in Fig. 2, the proposed VAD-GS framework identifies visible surfaces with unreliable geometry, selects informative supporting views, and complements missing structures via MVS-guided densification.

## Voxel-Based Visibility Reasoning

Accurate modeling of scene visibility is essential for training reliable Gaussian models, yet it remains underexplored. In the training phase, photometric-guided optimization and densification are typically performed via ray marching over semi-transparent Gaussian primitives. Nevertheless, when the initialized geometry is incomplete, these processes may erroneously update Gaussians associated with occluded or invisible surfaces, leading to geometric distortions and the emergence of floater artifacts (Zhang et al. 2024).

To address this issue, explicit view-dependent visibility and occlusion reasoning for scene objects must be incorporated into the reconstruction process. As illustrated in Fig. 3, a 3D point sampled along a pixel ray should correspond to the first intersected surface, visible from the given viewpoint. Existing methods either extract visible points independently from each single view or aggregate points from all available views without considering occlusion. While sampling points from a specific view guarantees valid surface intersections, it provides limited scene coverage due to observation constraints. In contrast, multi-view point cloud aggregation improves spatial coverage but lacks occlusion awareness, allowing rays to traverse occluded structures and resulting in erroneous updates on non-visible geometry.

To enable efficient reasoning about scene visibility and occlusion, VAD-GS first applies voxelization to the initialized point cloud to enforce uniform spatial density. The visibility of each voxel is defined as the union of the observation views associated with its constituent points. Specifically, Li-DAR points are sourced from individual frames, whereas SfM points are triangulated from at least two views. Reasoning about voxel visibility provides two key advantages: First, rasterizing the distances of visible voxel surfaces via classical z-buffering produces a denser and more reliable depth map, compared to conventional geometric supervision methods that rely on sparse point clouds and nearest-neighbor search. This rasterized depth map reduces missed surface intersections, constrains depth errors within the voxel resolution, and naturally excludes occluded voxels located behind incomplete foreground geometry, thereby preventing erroneous updates during model optimization. Second, rasterizing a 2D index map establishes a mapping between image pixels and their underlying 3D structures. By storing only the index of the first intersected and visible voxel along each pixel ray, this mapping ensures both validity and efficiency, enabling fast retrieval of geometric attributes such as 3D position, surface normals, and neighborhood connectivity.

Voxel visibility is further utilized to identify incomplete scene structures. Specifically, scene elements such as vehicles, trees, and buildings are individually extracted using an offline instance segmentation network (Kirillov et al. 2023). Pixels belonging to each segmented instance are then mapped to their corresponding voxels according to the rasterized index map. For each instance, two depth values are compared: one rasterized from visible voxels, while the other rendered from existing Gaussians. If the depth rendered from Gaussians is consistently smaller than the voxelderived depth, it indicates either successful completion of previously missing geometry or acceptable redundancy, both of which can be effectively handled via opacity adjustment. In contrast, if the Gaussian-rendered depth is absent or significantly larger than the voxel depth, it implies that the geometry is either partially initialized or distorted in earlier optimization stages. In such cases, the instance is flagged for re-initialization to improve reconstruction completeness.

## Diversity-Aware View Selection

After identifying instances with unreliable structures, the corresponding voxels along with available views can be retrieved using the rasterized index map. To ensure reliable geometry reconstruction via MVS, it is essential to select a representative subset of views that provides strong geometric constraints. Although consecutive video frames captured by a single camera can provide sufficient overlap for 3D reconstruction in static scenes (Cheng et al. 2024), their performance deteriorates in driving scenarios due to the existence of both dynamic objects and continuous egomotion. These factors necessitate sufficient frustum overlap and strong stereo constraints, which can only be achieved by selecting views from different cameras and timestamps. To this end, we define the following score:

<!-- image-->  
Figure 4: View Selection and MVS Reconstruction. Image patches are warped across views to check the consistency of depth, normal, and color. Only consistently matched patches (in red) are considered valid for MVS reconstruction, while inconsistent ones (in blue) are discarded. The reconstructed geometry is then used to guide Gaussian densification.

$$
s = \frac { N } { { d _ { R } ^ { \top } } { d _ { S } } } \frac { \sqrt { { t _ { x } } ^ { 2 } + { t _ { y } } ^ { 2 } } } { \left| t _ { z } \right| } \sin \theta ,\tag{3}
$$

to quantify the geometric diversity between a pair of views, where $d _ { R }$ and $d _ { S }$ are column vectors that store the distances from N voxels visible in both reference and supporting views to their respective viewpoints, $\mathbf { \Psi } _ { t } = ( t _ { x } , t _ { y } , \mathbf { \hat { { t } } } _ { z } ) ^ { \top }$ denotes the relative translation between the two views, and Î¸ represents the angular difference between their orientations. Higher scores are achieved when voxels are denser and closer to both views, lateral variations are greater, longitudinal displacements are minimal, and orientation differences are larger. In contrast, lower scores result under opposite conditions. Diverse supporting views selected based on this score are subsequently utilized for MVS reconstruction.

## MVS Reconstruction from Selected Views

By incorporating the selected supporting views that provide sufficient geometric constraints, reliable 3D points can be generated to complement incomplete scene structures. To achieve this goal, we adopt a multi-view patch matching approach, which has been widely used for dense 3D reconstruction in static scenes (Xu and Tao 2020). The method estimates local surface planes by matching small image patches across multiple views, under the assumption that the scene geometry is locally piecewise planar. Specifically, an image pixel located at $\mathbf { \widehat { \mathbf { \phi } } } \mathbf { \bar { \mathbf { \phi } } } \mathbf { \bar { \mathbf { \phi } } } \mathbf { \phi } ( \mathbf { \psi } _ { } \mathbf { \psi } _ { p } ) ^ { \top }$ is associated with a local 3D plane, expressed as: $\stackrel { \cdot } { z } { \pmb { n } } ^ { \top } { \pmb { K } } ^ { - 1 } \widetilde { \pmb { p } } + d = 0 .$ , where $z$ denotes the depth value at $^ { p , }$ n represents the corresponding surface normal, K denotes the camera intrinsic matrix, p represents the homogeneous coordinates of ${ \dot { \mathbf { \sigma } } } _ { p , \mathbf { \sigma } }$ and d represents the distance between the surface and the camera origin. In the patch matching process, p in the reference view, associated with a plane hypothesis (d, n), is projected to $\pmb { p } ^ { \prime }$ in a supporting view using the following expression:

$$
\widetilde { \pmb { p } } ^ { \prime } \simeq \pmb { K } \left( \pmb { R } - \frac { \pmb { t } { \pmb { n } } ^ { \top } } { d } \right) \pmb { K } ^ { - 1 } \widetilde { \pmb { p } } ,\tag{4}
$$

where $[ R , t ]$ denotes the relative pose from the reference view to the supporting view. Patches are warped across views to assess photometric consistency using RGB images and geometric consistency using depth and normal maps. As shown in Fig. 4, a pair of patches is deemed consistent if the reference patch and warped supporting patch exhibit similar image features, and their plane hypotheses (d, n) are well aligned. Patch hypotheses are initialized using real-world images and Gaussian-rendered results, and are iteratively refined by propagating candidates from neighboring pixels based on the assumption of local hypothesis similarity. Re-sampling is performed when no consistent matches are found. Through repeated updates and consistency checks, a set of patches that remain consistent across the majority of views is obtained for robust scene reconstruction.

Although reliable static structures can be recovered in this way, handling dynamic objects remains a significant challenge. In theory, a moving rigid vehicle can be treated as static by transforming all observation views into its local coordinate system. However, in practice, object masks derived from 3D bounding boxes cannot accurately delineate the boundaries between foreground and background. Including pixels from the static background may introduce misleading patch matches and disrupt the geometric consistency assumptions for the moving object. Although instance segmentation methods provide more precise, contour-aligned masks, their performance is highly sensitive to the quality of input prompts. A prompt derived from a single frame may fail to capture the entire object due to sparse point coverage, whereas one generated from a multi-frame aggregated point cloud often introduces occluding structures unrelated to the object. This challenge, nevertheless, can be effectively addressed by leveraging our rasterized visible voxels, which inherently incorporate occlusion cues and provide more accurate instance-level prompts for segmentation. Consequently, the method restricts the patch matching process within either static or dynamic regions across all relevant views, effectively minimizing cross-region interference and greatly enhancing reconstruction accuracy.

## Loss Function

We optimize the model by minimizing a weighted sum of four loss terms, as expressed as follows:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c o l o r } } + \lambda _ { \mathrm { n o r m a l } } \mathcal { L } _ { \mathrm { n o r m a l } } + \lambda _ { \mathrm { h a r d } } \mathcal { L } _ { \mathrm { h a r d } } + \lambda _ { \mathrm { s o f t } } \mathcal { L } _ { \mathrm { s o f t } } ,\tag{5}
$$

where $\scriptstyle { \mathcal { L } } _ { \mathrm { c o l o r } }$ quantifies the discrepancy between rendered and observed images, ${ \mathcal { L } } _ { \mathrm { n o r m a l } }$ quantifies the angular deviations between the rendered surface normals and those obtained via patch matching, and $\mathcal { L } _ { \mathrm { h a r d } }$ and ${ \mathcal { L } } _ { \mathrm { s o f t } }$ quantify depth errors under hard and soft Gaussian opacity settings, respectively. The weights $\lambda _ { \mathrm { n o r m a l } } , \lambda _ { \mathrm { h a r d } }$ , and $\lambda _ { \mathrm { s o f t } }$ , which control the three geometric losses, are also set following the studies (Yan et al. 2024). By incorporating our visibilityaware Gaussian densification strategy, the optimization of (5) achieves superior geometric reconstruction performance in complex dynamic urban settings.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

Figure 5: Qualitative comparisons between VAD-GS and other SoTA approaches on the nuScenes dataset.
<table><tr><td></td><td colspan="4">PVG</td><td colspan="4">OmniRe</td><td colspan="4">StreetGS</td><td colspan="4">VAD-GS (Ours)</td></tr><tr><td></td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G</td></tr><tr><td>Scene 00</td><td>22.77</td><td>0.63</td><td>0.22</td><td>195k</td><td>22.71</td><td>0.63</td><td>0.16</td><td>153k</td><td>22.87</td><td>0.65</td><td>0.13</td><td>146k</td><td>25.54</td><td>0.81</td><td>0.16</td><td>234k</td></tr><tr><td>Scene 01</td><td>22.91</td><td>0.66</td><td>0.24</td><td>196k</td><td>22.27</td><td>0.64</td><td>0.22</td><td>108k</td><td>22.12</td><td>0.66</td><td>0.23</td><td>109k</td><td>23.69</td><td>0.76</td><td>0.24</td><td>200k</td></tr><tr><td>Scene 03</td><td>24.30</td><td>0.77</td><td>0.14</td><td>172k</td><td>24.56</td><td>0.76</td><td>0.12</td><td>133k</td><td>24.18</td><td>0.77</td><td>0.13</td><td>128k</td><td>26.57</td><td>0.88</td><td>0.13</td><td>148k</td></tr><tr><td>Scene 04</td><td>24.37</td><td>0.65</td><td>0.15</td><td>184k</td><td>24.32</td><td>0.64</td><td>0.12</td><td>162k</td><td>24.37</td><td>0.67</td><td>0.13</td><td>140k</td><td>25.57</td><td>0.76</td><td>0.21</td><td>245k</td></tr><tr><td>Scene 05</td><td>20.76</td><td>0.50</td><td>0.37</td><td>249k</td><td>19.80</td><td>0.46</td><td>0.29</td><td>155k</td><td>20.20</td><td>0.50</td><td>0.28</td><td>146k</td><td>20.06</td><td>0.56</td><td>0.30</td><td>329k</td></tr><tr><td>Scene 06</td><td>22.68</td><td>0.64</td><td>0.17</td><td>173k</td><td>23.08</td><td>0.67</td><td>0.12</td><td>177k</td><td>23.58</td><td>0.70</td><td>0.12</td><td>169k</td><td>25.64</td><td>0.83</td><td>0.16</td><td>179k</td></tr></table>

Table 1: Quantitative comparisons between VAD-GS and other SoTA approaches on the nuScenes dataset. â#Gâ denotes the number of Gaussian primitives. Scene 02 is excluded due to the stationary ego vehicle, which provides no additional supporting views. Scenes 07â09 are also omitted, as their extreme nighttime illumination conditions fall beyond the scope of this study.

## Experiments

## Datasets and Implementation Details

We conduct extensive experiments on two large-scale autonomous driving datasets: Waymo Open (Sun et al. 2020) and nuScenes (Caesar et al. 2020). The Waymo Open dataset comprises 1,150 driving scenes recorded in suburban and urban environments. Each frame contains images captured using five cameras and fused point clouds collected using five LiDARs, with an average of 177k points per frame. Following the study (Yan et al. 2024), we select eight sequences, each containing around 100 frames under dynamic traffic conditions. For the nuScenes dataset, we follow the study (Chen et al. 2025), which also provides baseline implementations of PVG, OmniRe, and StreetGaussians. The data are collected using a 32-beam LiDAR, resulting in significantly sparser point clouds (with an average of 34k points per frame) and more uneven spatial coverage compared to the Waymo Open dataset. The increased sparsity poses greater challenges for Gaussian initialization and densification.

Our implementation is primarily based on the frameworks of StreetGaussians, GaussianPro, and DNGaussian. All models are trained for 30,000 iterations. Every fourth frame is used for model evaluation, while the remaining frames are used for model training. To alleviate photometric overfitting caused by imbalanced view sampling, training views are sampled without replacement. Voxel visibilitybased densification is performed every five complete sampling cycles. All experiments are conducted on a single NVIDIA GeForce RTX 4090 GPU with 24 GB of memory.

<table><tr><td></td><td>3D GS</td><td>NSG</td><td>MARS</td><td>EmerNeRF</td><td>StreetGS</td><td>Ours</td></tr><tr><td>PSNRâ</td><td>29.64</td><td>28.31</td><td>29.75</td><td>30.87</td><td>34.61</td><td>35.59</td></tr><tr><td>PSNR*â</td><td>21.25</td><td>24.32</td><td>26.54</td><td>0.346</td><td>30.23</td><td>31.31</td></tr><tr><td>SSIMâ</td><td>0.918</td><td>0.862</td><td>0.886</td><td>0.264</td><td>0.938</td><td>0.950</td></tr><tr><td>LPIPSâ</td><td>0.117</td><td>21.67</td><td>0.905</td><td>0.133</td><td>0.079</td><td>0.047</td></tr></table>

Table 2: Quantitative results on Waymo Open Dataset. PSNR\*: evaluated on dynamic objects only.

## Quantitative and Qualitative Results

We compare VAD-GS with several baseline approaches (Kerbl et al. 2023; Ost et al. 2021; Wu et al. 2023; Yang et al. 2023; Yan et al. 2024; Chen et al. 2024, 2025). Evaluation metrics, including the peak signal-to-noise ratio (PSNR), the structural similarity index measure (SSIM), and the learned perceptual image patch similarity (LPIPS), are used to quantify modelsâ performance. As shown in Table 2, VAD-GS consistently outperforms all baseline methods across all evaluation metrics on the Waymo Open dataset. In particular, VAD-GS improves PSNR by â¼2.8% and PSNR\* (evaluated on dynamic objects only) by â¼3.6%. These improvements can be primarily attributed to the complemented geometry, which effectively suppresses photometric distortions in erroneously exposed Gaussians. VAD-GS also outperforms the second-best method in terms of SSIM and LPIPS by 0.012 and 0.032, respectively, owing to its more complete and high-fidelity reconstruction of geometry and appearance, which in turn enhances photometric consistency. However, these performance gains reflect the potential of our densification strategy only to a limited extent, as existing baselines report single-camera results on the Waymo Open dataset. To ensure fair comparison, we have to adopt the same settings, which restrict the exploitation of crosscamera cues, a key advantage of VAD-GS.

<!-- image-->  
(a) Ground Truth

<!-- image-->  
(b) w/o Voxel Visibility Reasoning

<!-- image-->  
ew Selection  (c) w/o Vi

<!-- image-->  
(d) w/o Geometric Losses

<!-- image-->  
(e) Complete Model  
Figure 6: Qualitative ablation study results.

Completing missing geometry is less critical for the Waymo Open dataset, which provides high-quality point clouds for reliable Gaussian initialization, but becomes essential for the nuScenes dataset due to its significantly sparser LiDAR observations. Given that the difficulty of 3D reconstruction is highly scene-dependent and influenced by factors such as sampling trajectories, occlusions, and dynamic traffic behaviors, we additionally provide per-scene evaluation results on the nuScenes dataset, as shown in Table 1 and Fig. 5. Our method consistently outperforms baseline approaches in terms of SSIM, with improvements exceeding 0.06, and achieves a significant PSNR gain of over 0.78 dB across most scenes, except for Scene 05, where the ego vehicle follows a fast and nearly linear trajectory. This leads to sparse viewpoint sampling and limited overlap across camera views, making it challenging to observe objects from diverse perspectives. Our method does not achieve the lowest LPIPS values in several scenes, primarily because the dynamic objects in these scenes are mostly moving pedestrians. Since MVS-based reconstruction methods generally assume object rigidity, their effectiveness degrades when handling deformable or non-rigid objects such as pedestrians. While the number of Gaussian primitives is not a direct indicator of reconstruction quality, it does reflect modeling efficiency to some extent. The slightly higher Gaussian count observed in our method results from targeted densification in underrepresented regions, rather than uncontrolled growth or redundant duplication in well-initialized geometry. Additional details are given in the supplement.

## Ablation Study

To validate the efficacy of each component, we conduct a comprehensive ablation study on the nuScenes dataset. Given the interdependence among components (voxel visibility reasoning, view selection, and geometric losses), we train three variants of VAD-GS, each omitting one of these components, and compare their performance against the complete model. Fig. 6 and Table 3 present the qualitative and quantitative results, respectively. The first variant removes the voxel-based visibility reasoning component, which consequently disables all other components. Although photometric-based densification remains active, it fails to accurately recover unreliable geometry. As shown in Fig. 6(b), this leads to incorrect gradient updates that distort Gaussian primitives, ultimately causing significant performance degradation. The second variant disables diversityaware view selection and instead relies on fixed consecutive frames for patch matching. Although this improves densification in static regions, the recovery of missing geometry remains incomplete. Moreover, the absence of explicit separation between static and dynamic regions causes misleading matches between background and foreground geometries, leading to severe floater artifacts, as shown in Fig. 6(c). The third variant excludes geometric losses from the optimization objective. While it achieves comparable or even slightly better photometric metrics, attributed to the strong reliance on image similarity as the sole supervision signal, which encourages overfitting to visual appearance, this variant introduces noticeable artifacts under large viewpoint deviations, such as the rough vehicle surfaces observed in Fig. 6(d).

<table><tr><td>Configurations</td><td>PSNRâ</td><td>PSNR*â</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>w/o voxel visibility reasoning</td><td>23.79</td><td>22.75</td><td>0.753</td><td>0.215</td></tr><tr><td>w/o view selection</td><td>23.92</td><td>22.83</td><td>0.757</td><td>0.212</td></tr><tr><td>w/o geometric losses</td><td>24.59</td><td>22.78</td><td>0.764</td><td>0.194</td></tr><tr><td>Complete model</td><td>24.51</td><td>23.16</td><td>0.765</td><td>0.199</td></tr></table>

Table 3: Quantitative ablation study results.

## Conclusion and Future Work

This paper introduced VAD-GS, a novel 3DGS framework designed to enhance geometry recovery under sparse observations, particularly in dynamic, unbounded urban environments. Unlike prior Gaussian densification methods that exclusively clone or split existing Gaussians, VAD-GS reconstructed new Gaussians via MVS, which effectively recovers missing or uncertain geometry for both static and dynamic objects. The framework explicitly modeled the viewdependent voxel visibility, which enables the identification of regions requiring reconstruction. It then strategically selected supporting views based on a newly defined diversity score and generated additional point clouds that satisfy multi-view consistency, thereby improving structural completeness. Extensive experiments on public datasets demonstrate the superiority of VAD-GS. We plan to extend VAD-GS to model deformable objects in the future.

## References

AanÃ¦s, H.; et al. 2016. Large-scale data for multiple-view stereopsis. International Journal of Computer Vision, 120: 153â168.

Bao, Y.; et al. 2025. 3D Gaussian splatting: Survey, technologies, challenges, and opportunities. IEEE Transactions on Circuits and Systems for Video Technology, 35: 6832â 6852.

Barron, J. T.; et al. 2021. Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 5855â5864.

Barron, J. T.; et al. 2022. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5470â5479.

Caesar, H.; et al. 2020. nuScenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11621â11631.

Chen, Y.; et al. 2024. Periodic vibration Gaussian: Dynamic urban scene reconstruction and real-time rendering. arXiv:2311.18561.

Chen, Z.; et al. 2025. OmniRe: Omni urban scene reconstruction. In Proceedings of the International Conference on Learning Representations (ICLR), 85508â85527.

Cheng, K.; et al. 2024. GaussianPro: 3D Gaussian splatting with progressive propagation. In Proceedings of the International Conference on Machine Learning (ICML), 8123â 8140.

Cheng, S.; et al. 2020. Deep stereo using adaptive thin volume representation with uncertainty awareness. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2524â2534.

Dai, A.; et al. 2017. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 5828â5839.

Dosovitskiy, A.; et al. 2017. CARLA: An open urban driving simulator. In Conference on Robot Learning (CoRL), 1â16.

Hou, Y.; Kannala, J.; and Solin, A. 2019. Multi-view stereo by temporal nonparametric fusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2651â2660.

Jiang, H.; et al. 2024. GeoTexDensifier: Geometry-textureaware densification for high-quality photorealistic 3D Gaussian splatting. arXiv:2412.16809.

Kerbl, B.; et al. 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4): 1â14.

Kerbl, B.; et al. 2024. A hierarchical 3D Gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics, 43(4): 1â15.

Kirillov, A.; et al. 2023. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 4015â4026.

Li, J.; et al. 2024. DNGaussian: Optimizing sparse-view 3D Gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20775â 20785.

Mildenhall, B.; et al. 2021. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Muller, T.; et al. 2022. Instant neural graphics primitives Â¨ with a multiresolution hash encoding. ACM Transactions on Graphics, 41(4): 1â15.

Ost, J.; et al. 2021. Neural scene graphs for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2856â2865.

Ranftl, R.; et al. 2021. Vision Transformers for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 12179â12188.

Schonberger, J. L.; and Frahm, J.-M. 2016. Structure-frommotion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4104â 4113.

Schops, T.; et al. 2017. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3260â3269.

Sun, P.; et al. 2020. Scalability in perception for autonomous driving: Waymo Open dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2446â2454.

Wang, F.; et al. 2021. PatchmatchNet: Learned multi-view patchmatch stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14194â14203.

Wang, F.; et al. 2024. Learning-based Multi-View Stereo: A Survey. arXiv:2408.15235.

Wei, D.; Li, Z.; et al. 2025. Omni-Scene: Omni-Gaussian representation for ego-centric sparse-view scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 22317â 22327.

Wu, G.; et al. 2024. 4D Gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20310â20320.

Wu, Z.; et al. 2023. MARS: An instance-aware, modular and realistic simulator for autonomous driving. In Proceedings of the CAAI International Conference on Artificial Intelligence (CICAI), 3â15. Springer.

Xu, Q.; and Tao, W. 2020. Planar prior assisted patchmatch multi-view stereo. In Proceedings of the AAAI conference on artificial intelligence (AAAI), 12516â12523.

Yan, Y.; et al. 2024. Street Gaussians: Modeling dynamic urban scenes with Gaussian splatting. In Proceedings of the European Conference on Computer Vision (ECCV), 156â 173. Springer.

Yang, J.; et al. 2022. Non-parametric depth distribution modelling based depth inference for multi-view stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8626â8634.

Yang, J.; et al. 2023. EmerNeRF: Emergent spatialtemporal scene decomposition via self-supervision. arXiv:2311.02077.

Yao, Y.; et al. 2018. MVSNet: Depth inference for unstructured multi-view stereo. In Proceedings of the European Conference on Computer Vision (ECCV), 767â783.

Zhang, Z.; et al. 2024. Pixel-GS: Density control with pixelaware gradient for 3D Gaussian splatting. In Proceedings of the European Conference on Computer Vision (ECCV), 326â342. Springer.

Zhou, X.; et al. 2024. DrivingGaussian: Composite Gaussian splatting for surrounding dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 21634â 21643.

<!-- image-->  
(a)

<!-- image-->

<!-- image-->  
Figure 1: An illustration of multi-camera, cross-frame views. For both static and dynamic objects, informative observation views are typically captured by different cameras at different timestamps.

## Implementation Details

An Illustration of Multi-Camera, Cross-Frame Views As shown in Fig. 1, the outward-facing multi-camera views have limited overlaps. Prior methods such as (?) typically treat all views indiscriminately during Gaussian training, regardless of their spatial or temporal differences. Nonetheless, structural complexity varies significantly across regions, necessitating a selective reconstruction strategy that prioritizes critical objects over trivial or redundant structures. Object-centric reconstruction strategies generally assume sufficient overlap among views within a bounded range and minimal interference from unrelated perspectives. However, this assumption breaks down in dynamic, unbounded urban scenes. The failure case illustrated in Fig. 1 suggests that observations from the same camera fail to continuously capture a moving target vehicle.

View Selection The diversity score s introduced in the main paper quantifies the geometric dissimilarity between a pair of views. However, selecting an informative subset of supporting views for reconstruction requires more than simply maximizing diversity between view pairs, ensuring that the subset is collectively informative and non-redundant. Moreover, as the same reference object may appear repeatedly during training, a deterministic selection based solely on diversity may lead to overfitting or limited generalization. To address this issue, we propose to sample views via:

$$
\operatorname* { m a x } _ { \mathcal { V } _ { s } \subset \mathcal { V } _ { c } } \sum _ { v _ { i } \in \mathcal { V } _ { s } } s _ { i R } \xi _ { i R } + \lambda \sum _ { \{ v _ { i } , v _ { j } \} \subset \mathcal { V } _ { s } \atop | \mathcal { V } _ { s } | = k , \quad \xi \sim \mathcal { N } ( 1 , \epsilon ) , }\tag{1}
$$

where $\mathcal { V } _ { c }$ denotes the full set of all candidate views, $\gamma _ { s }$ represents the selected subset containing k supporting views, $s _ { i R }$ denotes the diversity score between each pair of candidate view and the reference view, $s _ { i j }$ represent the diversity score among views within the subset, and Ïµ represents a noise term introduced to encourage sampling diversity. This randomized selection strategy ensures relevance to the reference view while avoiding deterministic bias, resulting in a diverse yet non-redundant subset of supporting views.

## Additional Experiments

## Experimental Details

While many 3DGS methods adopt similar train/test splitting strategies, the specific details on these splits remain ambiguous for urban driving scenes. For example, statements such as ârandomly select every n-th image of different camerasâcan be interpreted in multiple ways: either as discarding specific frames with all associated camera views, or as selectively omitting individual views while retaining the full sequence of frames. Moreover, such random sampling schemes are misaligned with the practical goal of novel view synthesis, which aims to render intermediate views between consecutive video frames captured by multi-camera systems mounted on a moving vehicle.

While both strategies remove the same number of views, randomly selecting individual test views results in more uniform frustum coverage and visually cleaner outputs. However, this approach exploits temporal redundancy and overlooks the realistic constraint that multi-camera views are typically available or missing as a complete observation. In contrast, removing all views at specific timestamps significantly reduces scene coverage and degrades visual quality, particularly when the vehicle is moving rapidly. Despite being more challenging, this setting better reflects real-world deployment constraints and more effectively evaluates the modelâs generalizability.

Specifically, we select every fourth frame along with all associated camera views to construct the test set. As a result, spatial observations are entirely unavailable for approximately 25% of the ego vehicle poses. This setting poses significant challenges for models that rely on multi-view consistency or temporal cues, and serves as a rigorous benchmark for evaluating reconstruction robustness under sparse observational conditions.

## Additional Qualitative Comparisons

In this supplement, we provide additional comparative results against recent methods on large-scale driving scenes. Due to the page limitation, qualitative results on the Waymo Open dataset (?) are provided in Fig. 2. For fair comparison, we adopt the validation configuration of StreetGaussians (?) and use only a single forward-facing camera. This setup simplifies view-dependent appearance and geometry consistency constraints, as the forward-facing view undergoes relatively minor temporal changes. However, it inherently limits the acquisition of novel information and significantly reduces overall scene coverage. These minimal interframe variations result in highly similar and redundant observations, which can provide limited geometric diversity for triangulation or multi-view spatial-consistency reasoning, thus failing to fully unleash the potential of visibilityaware densification for complete geometry reconstruction. Consequently, high-fidelity rendering quality may not indicate accurate scene geometry recovery, but rather reflect overfitting to specific image observations.

To further demonstrate the high quality of our scene reconstruction, we present an additional example in Fig. 3. This comparison is performed by adopting a multi-camera configuration that utilizes cameras 0, 1, and 2 from the Waymo Open dataset. Although all methods achieve comparable rendering quality, the underlying geometry differs significantly. The traffic sign, highlighted by yellow circles, lies outside the LiDAR scanning range and is only partially visible from a limited number of viewpoints. In OmniRe (?), the sign is reconstructed as a set of scattered and unstructured Gaussians, indicating overfitting to appearance cues in the absence of reliable geometric constraints. As for StreetGaussians, the sign appears fragmented and discontinuous, with Gaussians erroneously updated to positions between the sign and the background trees. These artifacts stem from missing Gaussians caused by incomplete initialization, which in turn lead to erroneous gradient propagation toward trees that should be occluded. The misdirected gradients distort the initial Gaussians representing the leaves, altering their color, position, and shape, and unnaturally pull them toward the sign, ultimately resulting in fragmented and misaligned geometry.

<!-- image-->  
Figure 2: Additional qualitative results on the Waymo Open dataset. Due to the single-camera configuration, test views captured by the forward-facing camera exhibit substantial overlap with the training views. While all methods achieve highfidelity rendering results under this setting, such performance may not reliably indicate the quality of the underlying geometry.

Benefiting from visibility reasoning, view selection, and MVS-based reconstruction, VAD-GS densifies Gaussians beyond conventional photometric-based splitting and cloning strategies, greatly alleviating issues related to incomplete or distorted geometry. Notably, VAD-GS accurately recovers the planar structure of the traffic sign, with only minor artifacts at the top border due to limited observations. Additionally, the road surface, highlighted by the white box, demonstrates a more geometrically consistent reconstruction compared to other approaches.

## Additional Ablation Studies

In this supplementary material, we also present additional qualitative ablation study results, including rendered RGB images, depth maps, and normal maps, to further demonstrate the effectiveness of each module in VAD-GS. As shown in Fig. 4, the sparse point clouds provide limited surface coverage. Each LiDAR scan line in the ground-truth point typically contributes only two or three points to thin structures such as tree trunks or utility poles. Additionally, due to the limited scanning angle and sparse sampling intervals, the resulting point cloud distribution exhibits substantial gaps and covers only a narrow field of view. These limitations pose significant challenges for capturing complete geometry, particularly for large and distant surfaces such as buildings and walls.

Furthermore, we select several challenging test views to more clearly demonstrate the contribution of each component. A common issue during densification is the emergence of floaters, where Gaussians become misaligned with the actual scene geometry. While most floaters are often naturally pruned or corrected when they appear in regions wellcovered by training views, they tend to persist in sparsely observed areas. In selected test views where these floaters are prominent, our geometric loss effectively penalizes them, encouraging alignment with the correct underlying surfaces. This process significantly improves the final surface quality and substantially reduces visual artifacts. Moreover, objects that are only transiently visible, such as moving vehicles or structures primarily observed from side views, often suffer from sparse observations. Our view selection and MVSbased reconstruction modules improve the instance-level fidelity in these challenging regions, including dynamic vehicles, small trees, and complex landmarks such as the bottleshaped building.

## Failure Cases and Limitations

Despite achieving high-fidelity performance, VAD-GS still exhibits several known limitations. The primary challenge lies in its inability to effectively handle deformable objects.

<!-- image-->  
Figure 3: Qualitative comparison between VAD-GS and other SoTA methods on the Waymo Open dataset when a multicamera configuration is used.

Given that our objective is to recover geometry in complex urban scenes, the presence of walking pedestrians is inevitable. Nonetheless, these non-rigid objects violate the rigidity assumption required by MVS-based reconstruction. Future work will explore the integration of state-of-theart Gaussian-based deformable object modeling approaches, such as 4DGS (?) and SC-GS (?), to address this issue.

Second, our method assumes locally consistent visibility among neighboring points. While this assumption enables effective occlusion modeling and supports continuous surface reconstruction, it may fail in extreme cases involving complex structures such as wire fences or glass surfaces. These structures often reflect LiDAR beams, producing dense point clouds that resemble those from regular surfaces. Nevertheless, the simultaneously captured images may reveal background objects without occlusion, leading to discrepancies between geometric and visual observations. Accurately and efficiently modeling occlusion relationships in such challenging and visually ambiguous regions remains an important direction for future research.

<!-- image-->  
Figure 4: Additional qualitative ablation study results on the nuScenes dataset. The rendered RGB images, depth maps, and normal maps are visualized in (a), (b), and (c), respectively.