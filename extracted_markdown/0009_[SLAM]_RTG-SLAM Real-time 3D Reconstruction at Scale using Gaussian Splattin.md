# RTG-SLAM: Real-time 3D Reconstruction at Scale Using Gaussian Splatting

Zhexi Pengâ   
zhexipeng@zju.edu.cn   
State Key Lab of CAD&CG   
Zhejiang University   
Hangzhou, China   
Tianjia Shaoâ   
tjshao@zju.edu.cn   
State Key Lab of CAD&CG   
Zhejiang University   
Hangzhou, China

Yin Yang yin.yang@utah.edu University of Utah Salt Lake City, USA

Jingdong Wang welleast@gmail.com Baidu Research Beijing, China

Yong Liu   
Jingke Zhou   
zilae@zju.edu.cn   
zhoujk@zju.edu.cn   
State Key Lab of CAD&CG   
Zhejiang University   
Hangzhou, China   
Kun Zhouâ    
kunzhou@acm.org   
State Key Lab of CAD&CG   
Zhejiang University   
Hangzhou, China

<!-- image-->

<!-- image-->  
Ours: 16.28 fps; 7.3GB memory

<!-- image-->  
Co-SLAM: 8.77 fps; 17GB memory

<!-- image-->  
Point-SLAM: 0.22 fps; 9.4GB memory

Figure 1: A hotel room (about 56.3??2Ã1.7??) reconstructed by our system and the state-of-the-art NeRF-based RGBD SLAM techniques (Co-SLAM [Wang et al. 2023], Point-SLAM [SandstrÃ¶m et al. 2023]) without any post-processing. Compared with the state-of-the-art NeRF-based RGBD SLAM, our system achieves comparable high-quality reconstruction but with around twice the speed and half the memory cost, and shows higher realism in novel view synthesis.

## ABSTRACT

We present Real-time Gaussian SLAM (RTG-SLAM), a real-time 3D reconstruction system with an RGBD camera for large-scale environments using Gaussian splatting. The system features a compact

âJoint first authors â Corresponding author

Gaussian representation and a highly efficient on-the-fly Gaussian optimization scheme. We force each Gaussian to be either opaque or nearly transparent, with the opaque ones fitting the surface and dominant colors, and transparent ones fitting residual colors. By rendering depth in a different way from color rendering, we let a single opaque Gaussian well fit a local surface region without the need of multiple overlapping Gaussians, hence largely reducing the memory and computation cost. For on-the-fly Gaussian optimization, we explicitly add Gaussians for three types of pixels per frame: newly observed, with large color errors, and with large depth errors. We also categorize all Gaussians into stable and unstable ones, where the stable Gaussians are expected to well fit previously observed RGBD images and otherwise unstable. We only optimize the unstable Gaussians and only render the pixels occupied by unstable Gaussians. In this way, both the number of

Gaussians to be optimized and pixels to be rendered are largely reduced, and the optimization can be done in real time. We show real-time reconstructions of a variety of large scenes. Compared with the state-of-the-art NeRF-based RGBD SLAM, our system achieves comparable high-quality reconstruction but with around twice the speed and half the memory cost, and shows superior performance in the realism of novel view synthesis and camera tracking accuracy.

## CCS CONCEPTS

â¢ Computing methodologies â Reconstruction; Point-based models.

## KEYWORDS

SLAM, 3D reconstruction, Gaussian splatting, RGBD, scan

## ACM Reference Format:

Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. 2024. RTG-SLAM: Real-time 3D Reconstruction at Scale Using Gaussian Splatting. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers â24 (SIGGRAPH Conference Papers â24), July 27-August 1, 2024, Denver, CO, USA. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3641519.3657455

## 1 INTRODUCTION

Real-time 3D reconstruction at scale has been a long-studied problem in computer graphics and vision, and is crucial in many applications including VR/AR, autonomous robots, and interactive scanning with immediate feedback. With the ubiquity of RGBD cameras (e.g., Microsoft Kinect), different RGBD SLAM (Simultaneous Localization and Mapping) methods are proposed for real-time 3D reconstruction, using a variety of surface representations such as point clouds [Du et al. 2011], surfels [Keller et al. 2013; Whelan et al. 2015], and signed distance functions [Newcombe et al. 2011]. These methods are able to reconstruct large-scale scenes in real time with high-quality 3D surfaces [Dai et al. 2017; NieÃner et al. 2013; SteinbrÃ¼cker et al. 2013]. However, they mainly focus on the geometry accuracy of the 3D reconstruction, and rarely consider the rendering realism of reconstructed results.

Some works attempt to employ neural radiance fields (NeRF) as the implicit scene representation [Mildenhall et al. 2020] for dense RGBD SLAM in hopes of achieving high-quality reconstruction of both geometry and appearance. These methods typically represent the scene as an MLP network [Sucar et al. 2021a] or an implicit grid [Yang et al. 2022; Zhu et al. 2022a], optimizing the scene parameters and estimating the camera pose via differentiable volume rendering. However, due to the expensive cost of volume rendering, these methods have difficulties in reaching real-time performance. Besides, the high memory cost makes it hard for them to handle large scale scenes.

More recently, 3D Gaussians [Kerbl et al. 2023] have emerged as an alternative representation of radiance fields, which can achieve equal or better rendering quality than previous NeRFs while being much faster in rendering and training. However, the 3D Gaussian representation up until now is mainly used in offline reconstruction scenarios [Chung et al. 2024; Yang et al. 2023], not suitable for online reconstruction tasks with sequential RGBD inputs. To use it for real-time 3D reconstruction at scale, the core problem lies in how to represent the scene with low memory and computation cost, and how to perform online Gaussian optimization in real time. We noticed there are several concurrent works [Huang et al. 2023b; Keetha et al. 2023; Matsuki et al. 2023; Yan et al. 2023; Yugay et al. 2023] trying to incorporate Gaussians into RGBD SLAM systems, which use different scene representations with Gaussians as well as different online optimization strategies. While promising results are demonstrated, there is still a long way to realize real-time reconstruction of large-scale scenes.

In this paper, we introduce Real-time Gaussian SLAM (RTG-SLAM), a real-time 3D reconstruction system with an RGBD camera for large-scale environments using Gaussian splatting, featuring a compact Gaussian representation and a highly efficient on-the-fly Gaussian optimization scheme. In our compact Gaussian representation, we force each Gaussian to be either opaque or nearly transparent, with the opaque ones fitting the surface (i.e., depth map) and dominant colors, and transparent ones fitting residual colors. Our intention is to use a single opaque Gaussian to fit a local region of the surface without the need for multiple overlapping Gaussians. However, even for an opaque Gaussian, rendering its depth in the same way as rendering color would produce varying depth values declined from the Gaussian center, making it inaccurate to represent a local area using this Gaussian alone. To this end, we propose to render depth in a different way from color rendering. Following classical point rendering techniques [Zwicker et al. 2001], we treat each opaque Gaussian as an ellipsoid disc on the dominant plane of Gaussian, so that it can well fit a local region or a large flat area by itself. The depth rendering is very convenient under this setting. During color rendering, we already have the sorted Gaussians as well as their opacities for each pixel. By selecting from front to back the first Gaussian whose opacity for the pixel is larger than a given threshold, we consider the ray hits the ellipsoid disc and compute the intersection point using equations of the ray and disc plane. Then the depth for the pixel is equal to the depth of the intersection point. The whole process is differentiable, so Gaussians can be optimized by measuring the differences between the rendered and input depth maps through backpropagation. The compact Gaussian representation can fit the 3D surfaces with much fewer Gaussians, hence largely reducing the memory and computation cost.

We design a highly efficient on-the-fly Gaussian optimization scheme for the compact Gaussian representation. We first categorize all Gaussians into stable and unstable ones following classical pointbased reconstruction works [Keller et al. 2013], based on whether they have been sufficiently optimized. The stable Gaussians are expected to well fit previously observed RGBD images and otherwise unstable. Then given a new RGBD frame during scanning, instead of adaptively densifying Gaussians based on view space position gradients [Kerbl et al. 2023], we explicitly add Gaussians for three types of pixels with valid depths: newly observed pixels, pixels with large color errors after color re-rendering, and pixels with large depth errors after depth re-rendering. For newly observed pixels or pixels with large depth errors, which means new opaque Gausians are required to fit the surface, we uniformly sample a small portion of pixels to initialize opaque Gaussians. For the pixels with only large color errors, which means they already have opaque

Gaussians well fitting the surface but poorly fitting the appearance in the current view, we apply the same pixel sampling and check the states of associated opaque Gaussians. If unstable, we leave them to continue being optimized. Otherwise, we add a transparent Gaussian to provide a residual color to improve the color in the current view without breaking previous observation. Afterwards, we launch the optimization process based on the re-rendering losses of color and depth. Note we only optimize the unstable Gaussians and only render the pixels occupied by the unstable Gaussians. In this way, both the number of Gaussians to be optimized and pixels to be rendered are largely reduced, and the optimization can be done in real time. We also establish a state management mechanism that enables the mutual conversion between stable/unstable Gaussians, as well as the removal of long-term erroneous Gaussians. Finally, to achieve accurate tracking in complex real-world environment, we use the classical frame-to-model ICP as the front-end odometry, and maintain a set of landmarks for back-end graph optimization.

We show real-time reconstructions of a variety of real large scenes, including corridor, storeroom, hotel room, home and office, ranging from 43??2â¼100??2. All the results are scanned and reconstructed with a Microsoft Azure Kinect in real time (around 16 fps) without any post-processing. Comparisons demonstrate that RTG-SLAM runs at around twice the speed of the state-of-the-art NeRF-based SLAM, with around half the memory cost (e.g., 17.9 fps, 8.8 GB versus 8.65 fps, 17.3 GB [Wang et al. 2023] on the home scene). We also compare our method with the concurrent Gaussian SLAM work SplaTAM [Keetha et al. 2023] (the only one with code published). We also surpass SplaTAM in speed and memory, where SplaTAM runs at 0.31 fps and is out of memory during scanning of the home scene. We also conduct extensive experiments on three widely-used datasets: Replica [Straub et al. 2019], TUM-RGBD [Sturm et al. 2012] and ScanNet++ [Yeshwanth et al. 2023]. Compared with the state-of-the-art NeRF SLAM methods, our system achieves comparable high-quality reconstruction, and shows superior performance in time and memory performance, realism of novel view synthesis, and camera tracking accuracy.

## 2 RELATED WORK

Classical RGBD dense SLAM. There has been extensive work on 3D reconstruction with RGBD cameras over the past decade. We point the reader to the excellent state-of-the-art report [ZollhÃ¶fer et al. 2018] for detailed reviews. For online 3D reconstruction of scenes, numerous valuable works have emerged in the field of RGBD dense SLAM, with a variety of map representations, such as point clouds [Du et al. 2011], Hermite radial basis functions [Xu et al. 2022], surfels [Cao et al. 2018; Keller et al. 2013; Whelan et al. 2015], and signed distance functions (TSDF) [Chen et al. 2013; Dai et al. 2017; Huang et al. 2023a; Newcombe et al. 2011; NieÃner et al. 2013; Zhang et al. 2015]. For example, BundleFusion [Dai et al. 2017], the state-of-the-art TSDF method for online reconstructing large-scale scenes, presents real-time globally consistent 3D reconstruction using on-the-fly surface re-integration, which reconstructs high-quality 3D scenes at scale. ElasticFusion [Whelan et al. 2015] represents scenes as a collection of surfels, employing surfelrendered depth and color for tracking, also achieving high-quality results in real time. DI-Fusion [Huang et al. 2021] encodes scene priors considering both the local geometry and uncertainty parameterized by a deep neural network. These works mainly focus on the geometry reconstruction, while differently, our method simultaneously considers the surface reconstruction and photorealistic rendering.

NeRF-based RGBD dense SLAM. Recently, with the great success of neural radiance fields (NeRF) [Mildenhall et al. 2020], some works have integrated NeRF with RGBD dense SLAM systems. For example, iMap [Sucar et al. 2021b] is the first NeRF SLAM method using a single MLP as the scene representation. NICE-SLAM [Zhu et al. 2022b] represents scenes as hierarchical feature grids, utilizing pre-trained MLPs for decoding. Vox-fusion [Yang et al. 2022] represents scenes as voxel-based neural implicit surfaces and stores them using octrees. The state-of-the-art NeRF SLAM works include ESLAM [Johari et al. 2023] representing scenes as multi-resolution feature grids, and Co-SLAM [Wang et al. 2023] representing scenes as multi-resolution hash grids. An alternative approach is Point-SLAM [SandstrÃ¶m et al. 2023], which employs neural point clouds and performs volumetric rendering with feature interpolation. These methods have achieved impressive results. However, as they are based on time-consuming volume rendering, all these methods have difficulties to reach real-time performance on real scenes. Besides, the memory cost of these NeRF SLAM methods is high, prohibiting them from reconstructing large-scale scenes. In contrast, our method can reconstruct large scenes in real time, with much higher speed and lower memory cost.

Gaussian-based RGBD dense SLAM. There are some concurrent works aiming to integrate 3D Gaussians into dense RGBD SLAM. 3D Gaussians [Kerbl et al. 2023] can render high-quality images in real time, but the optimization is conducted offline typically requiring several minutes. To extend Gaussians to online reconstruction, [Yan et al. 2023] proposes an adaptive expansion strategy to add new or delete noisy 3D Gaussian and a coarse-to-fine technique to select reliable Gaussians for tracking. [Yugay et al. 2023] proposes novel strategies for seeding and optimizing Gaussian splats to extend their use to sequential RGBD inputs. SplaTAM [Keetha et al. 2023] tailors an online reconstruction pipeline to use an underlying Gaussian representation and silhouette-guided optimization via differentiable rendering. [Matsuki et al. 2023] unifies the Gaussian representation for accurate, efficient tracking, mapping, and high-quality rendering. [Huang et al. 2023b] introduces a Gaussian-Pyramid-based training method to progressively learn multi-level features and enhance mapping performance. While promising results are demonstrated, it is still difficult in reaching real-time reconstruction at scale. The reported fastest reconstruction speed is 8.34 fps on an NVIDIA RTX 4090 GPU [Matsuki et al. 2023] on the synthetic Replica dataset [Straub et al. 2019]. They did not present the complete reconstruction results on real large scenes either. Thanks to our compact Gaussian representation and highly efficient Gaussian optimization strategy, our method can reconstruct real large scenes in real time with low memory cost.

## 3 METHOD

The overview of our reconstruction pipeline is illustrated in Fig. 2. In Sec. 3.1, we first introduce our compact Gaussian representation

<!-- image-->  
Figure 2: Overview of our method. Left: we force each Gaussian to be either opaque or nearly transparent, and the depth is rendered differently from the color using the opaque Gaussian, so that a single opaque Gaussian can well fit a local region of the surface, yielding a compact Gaussian representation fitting 3D surfaces with much fewer Gaussians. Right: we compute the color error map, depth error map, and light transmission map to determine where to add opaque Gaussians or transparent Gaussians. we only optimize the unstable Gaussians, and only render the pixels occupied by them for optimization.

and the corresponding rendering process of color and depth (Fig. 2 left). Next, we describe in detail the entire online reconstruction process based on the compact Gaussian representation in Sec. 3.2.

## 3.1 Compact Gaussian Representation

We represent the scene S using a collection of 3D Gaussians $\left\{ { G } _ { i } \right\}$ Similar to [Kerbl et al. 2023], each Gaussian is associated with the position p?? , covariance matrix $\Sigma _ { i } ,$ opacity ???? and spherical harmonics (SH) coefficients $\mathrm { s H } _ { i }$ . The covariance matrix $\Sigma _ { i }$ is decomposed into a scale vector $\mathbf { \boldsymbol { s } } _ { i }$ and a quaternion q??. Each Gaussian is determined once after being added to be opaque $( \alpha = 0 . 9 9 )$ for fitting the 3D surface and dominant color, or to be nearly transparent $( \alpha = 0 . 1 )$ for fitting the residual color.

We also treat each Gaussian as an ellipsoid disc (or surfel), and record the surfel parameters including the normal $\mathbf { n } _ { i } ,$ , the confidence count ???? , and the initialization timestamp $t _ { i } .$ The normal vector is defined as the direction of the smallest eigenvector. The shape of surfel is defined as the region with Gaussian density larger than $\delta _ { \alpha } = e ^ { - 0 . 5 }$ on the dominant plane of Gaussian, which corresponds to the density range within the standard deviation of the Gaussian distribution. ?? records how often a Gaussian is optimized, and ?? records the time a Gaussian is created. We also divide the Gaussians into stable ones $\boldsymbol { S _ { s t a b l e } }$ and unstable ones $S _ { u n s t a b l e }$ based on the confidence count threshold $\delta _ { \eta }$ . All parameters are stored in a flat array indexed by the Gaussian index ??.

Image rendering. The core of optimizing Gaussians lies in rendering color and depth maps through differentiable splatting, calculating errors with input RGBD images, and updating the Gaussian parameters. Now we introduce the rendering process in detail. Given a camera pose $\mathrm { T } _ { q }$ and camera intrinsic matrix K, the ray through the center of each pixel u in the image is defined as:

$$
\mathbf { r } ( \mathbf { u } ) = ( \mathbf { R } _ { g } \mathbf { K } ^ { - 1 } \dot { \mathbf { u } } ) \boldsymbol { \theta } + \mathbf { t } _ { g } , \mathrm { w h e r e } \mathbf { T } _ { g } = \left[ \begin{array} { c c } { \mathbf { R } _ { g } } & { \mathbf { t } _ { g } } \\ { \mathbf { 0 } } & { 1 } \end{array} \right] \in \mathbb { S } \mathbb { E } ( 3 ) .
$$

Here ?? is the length parameter along the ray direction and uÂ¤ is the homogeneous vector $\dot { \mathbf { u } } : = ( \mathbf { u } ^ { \top } | \boldsymbol { 1 } ) ^ { \top }$ . Then the color image $\hat { \mathbf { C } }$ can be

rendered by alpha-blending proposed in [Kerbl et al. 2023] :

$$
\hat { \mathbf { C } } ( \mathbf { u } ) = \sum _ { i = 1 } ^ { n } \mathbf { c } _ { i } f _ { i } ( \mathbf { u } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \mathbf { u } ) \right) ,\tag{1}
$$

where $\mathbf { c } _ { i }$ represents the Gaussian color based on the view direction r?? and the SH coefficients SH?? . ???? (??) is computed by the center ???? and covariance matrix $\Sigma _ { 2 D , i }$ of the splatted 2D Gaussian in pixel space:

$$
f ( \mathbf { u } ) = \alpha _ { i } \exp ( - \frac { 1 } { 2 } ( \mathbf { u } - \boldsymbol { \mu } ) ^ { \top } \Sigma _ { 2 D , i } ^ { - 1 } ( \mathbf { u } - \boldsymbol { \mu } ) ) .\tag{2}
$$

Also a light transmission image $\hat { \bf T }$ to determine the visibility can be rendered as:

$$
{ \hat { \mathbf { T } } } ( \mathbf { u } ) = \prod _ { i = 1 } ^ { n } \left( 1 - f _ { i } ( \mathbf { u } ) \right) .\tag{3}
$$

TË represents the remaining energy of the light after it passes through a series of 3D Gaussians.

The depth rendering is the key for our compact Gaussian representation, where each single Gaussian can well fit a local region of surface without the need for multiple Gaussians. Note that all concurrent Gaussian SLAM works utilize the alpha blending methods to render the depth as the color. However, as illustrated in Fig. 3, in the alpha blending setting, a single Gaussian will present varying depth values declined from the Gaussian center, which is inappropriate to alone fit a local area that can typically be approximated as a plane. To this end, we render depth differently from rendering color. That is, for each pixel, we compute the intersection point of the view ray and the frontest opaque ellipsoid disc to obtain the pixelâs depth. Fortunately, we donât need to explicitly convert the Gaussians into ellipsoid discs and compute the intersections for each ray. During color rendering, all Gaussians $\{ G _ { j } ^ { \mathbf { r } } \}$ crossed by the ray $\mathbf { r } ( \mathbf { u } )$ are already sorted from front to back and the corresponding opacities $\{ \alpha _ { i } ^ { \mathbf { r } } \}$ along the ray are computed. The intersected Gaussian $G _ { j } ^ { \mathbf { r } }$ is the first Gaussian with $\alpha _ { j } ^ { \mathbf { r } } > \delta _ { \alpha } .$ . The intersection point p??r,r can be easily calculated by the ray plane intersection

<!-- image-->  
Figure 3: If the depth is rendered in the same way as the color, the opaque Gaussian would produce varying depth values declined from the Gaussian center, making it inaccurate to represent a local surface. In contrast, we treat the opaque Gaussian as an ellipsoid disc on the dominant plane, and can well fit the local region.

formula:

$$
\mathbf { p } _ { G _ { j } ^ { \mathrm { r } } , \mathbf { r } } = ( \mathbf { R } _ { g } \mathbf { K } ^ { - 1 } \dot { \mathbf { u } } ) \boldsymbol { \theta } _ { \mathbf { u } } + \mathbf { t } _ { g } , \mathrm { ~ w h e r e ~ } \boldsymbol { \theta } _ { \mathbf { u } } = \frac { ( \mathbf { p } _ { j } ^ { \mathrm { r } } - \mathbf { t } _ { g } ) \cdot \mathbf { n } _ { j } ^ { \mathrm { r } } } { ( \mathbf { R } _ { g } \mathbf { K } ^ { - 1 } \dot { \mathbf { u } } ) \cdot \mathbf { n } _ { j } ^ { \mathrm { r } } } .\tag{4}
$$

Here $\mathbf { p } _ { j } ^ { \mathbf { r } }$ and $\mathbf { n } _ { j } ^ { \mathbf { r } }$ are the position and normal of intersected Gaussian. If all $\{ \alpha _ { j } ^ { \mathbf { r } } \}$ are smaller than $\delta _ { \alpha } .$ the pixel depth is set to -1. When the disc normal and the ray are nearly perpendicular, the ray plane intersection could lead to aberrations and we use $\mathbf { p } _ { j } ^ { \mathbf { r } }$ to approximate the intersection. Finally, depth map $\hat { \bf D }$ is defined as:

$$
\begin{array} { r } { \hat { \mathbf { D } } ( \mathbf { u } ) = \left\{ \begin{array} { l l } { - 1 \quad \mathrm { i f ~ n o ~ i n t e r s e c t i o n , } } \\ { ( \mathbf { T } _ { g } ^ { - 1 } \mathbf { p } _ { G _ { j } ^ { \mathrm { r } } , \mathbf { r } } ) _ { z } \quad \mathrm { e l i f ~ } \langle \mathbf { n } _ { j } ^ { \mathrm { r } } , \mathbf { r } \rangle < 6 0 ^ { \circ } , } \\ { ( \mathbf { T } _ { g } ^ { - 1 } \mathbf { p } _ { j } ^ { \mathrm { r } } ) _ { z } \quad \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{5}
$$

Clearly, the depth rendered by the $\operatorname { E q . }$ (5) is linearly computed from the position and rotation of the Gaussian, and such calculation process is fully differentiable, so the Gaussians can be optimized from the depth image loss. In addition, based on the results of ray-ellipsoid intersection, we also obtain the normal map $\hat { \mathbf { N } } ( \mathbf { u } )$ and index map ËI(u), where the respective Gaussiansâ normals and indices are stored. NË (u) is employed to carry out the frame-tomodel ICP for camera tracking, while the ËI(u) provides a one-to-one mapping from pixels to Gaussians, utilized in subsequent processes such as Gaussian adding and state management.

## 3.2 Online Reconstruction Process

As shown in Fig. 2, our online reconstruction system is composed of the following stages.

Input pre-processing. Given an input color image $\mathbf { C } _ { k }$ and depth map $\mathbf { D } _ { k }$ , following [Newcombe et al. 2011], we compute the local vertex map $\mathbf { V } _ { k } ^ { l }$ and the local normal map $\mathbf { N } _ { k } ^ { l }$ . With the estimated camera pose $\mathbf { T } _ { \mathfrak { g } , k } , \mathbf { V } _ { k } ^ { l }$ and $\mathbf { N } _ { k } ^ { l }$ can be transformed into $\mathbf { V } _ { k } ^ { g }$ and $\mathbf { N } _ { k } ^ { g }$ in the global coordinate.

Gaussians adding. In order to obtain a complete representation of the environment, we need to add new Gaussians to the scene during online scanning to cover new observed regions. The adaptive control of Gaussians in [Kerbl et al. 2023] based on view-space positional gradients are inefficient for the online scanning. Therefore, we utilize a more efficient and reliable Gaussian adding strategy based on both geometry and appearance. Specifically, given the estimated camera pose $\mathbf { T } _ { { g , k } }$ , we first render the color map $\hat { \mathbf { C } } _ { k }$ , depth map $\hat { \mathbf { D } } _ { k }$ , light transmission map $\hat { \mathbf { T } } _ { k }$ and index map $\hat { \mathbf { I } } _ { k }$ using the existing Gaussians in the scene. Then a mask ?? is created to determine for which pixel a Gaussian should be added:

$$
\begin{array} { r l } & { M _ { s } = \{ \mathbf { u } _ { s } \Big | \hat { \mathbf { T } } _ { k } ( \mathbf { u } _ { s } ) > \delta _ { \mathrm { T } } , \mathrm { ~ o r ~ } | \hat { \mathbf { D } } _ { k } ( \mathbf { u } _ { s } ) - \mathbf { D } _ { k } ( \mathbf { u } _ { s } ) | > \delta _ { d } \} , } \\ & { M _ { c } = \{ \mathbf { u } _ { c } \Big | | \hat { \mathbf { C } } _ { k } ( \mathbf { u } ) - \mathbf { C } _ { k } ( \mathbf { u } ) | > \delta _ { \mathrm { c } } , \mathrm { ~ a n d ~ } \mathbf { u } _ { c } \notin M _ { s } \} . } \end{array}\tag{6}
$$

Here $M _ { s }$ represents regions where new geometry should be added. $\hat { \mathbf { T } } _ { k } ( \mathbf { u } _ { s } ) > \delta _ { \mathrm { T } }$ means the remaining energy of the ray is large without hitting any Gaussian or just hitting the Gaussian boundary, indicating newly observed areas, and $\delta _ { \mathrm { T } } ~ = ~ 0 . 5$ in our setting. $| \hat { \mathbf { D } } _ { k } ( \mathbf { u } _ { s } ) - \mathbf { D } _ { k } ( \mathbf { u } _ { s } ) | > \delta _ { d }$ means there exist large re-rendering errors of depth, indicating new surface appears different from the existing scene, and $\delta _ { d } = 0 . 1$ in our setting. $M _ { c }$ represents those areas that are geometrically accurate but with apparent color errors, and $\delta _ { c } = 0 . 1$ .

With $\mathbf { V } _ { k } ^ { g }$ and $M ,$ we have a good estimation of where those Gaussians should be added. However, adding Gaussians using all ?? will cause considerable GPU memory overhead and hinder realtime performance. Therefore, we uniformly sample 5% pixels on $M _ { s }$ and $M _ { c }$ to perform Gaussian adding. For the pixels sampled from $M _ { s } ,$ , we add Gaussians for them to fit the newly observed surface and we set the opacity to 0.99. For each pixel sampled from $M _ { c }$ it is already associated with an existing opaque Gaussian in the scene, and we query that Gaussian using the index map $\hat { \mathbf { I } } _ { k }$ and check its confidence state. If unstable, which means the Gaussian can be further optimized to fit the color, we do not add a new Gaussian for this pixel. Otherwise, we add a transparent $( \alpha = 0 . 1 )$ Gaussian to correct color errors together with the stable Gaussian. The advantage of using transparent Gaussians lies in that such Gaussians with low opacity do not cause a significant attenuation of light energy, and the color impact to other views is little. In addition, during depth rendering, they are automatically filtered out by $\delta _ { \alpha } .$ , therefore not affecting the depth rendering. All the added Gaussians are initialized as thin circle discs with the pixelsâ colors, positions, and normals, with confidence count $\eta = 0$ and timestamp $t = k ,$ . For opaque Gaussians, their sizes are initialized to cover the scene as much as possible with little overlapping, while the transparent oneâs radius is limited to below 0.01m to elliminate the disruption of the rendering results on other pixels.

Gaussian optimization. After adding Gaussians for the frames within a time window, we launch the Gaussian optimization based on the color loss and depth loss between the input and rendered RGBD images. We randomly sample a frame ?? within the window per iteration during optimization. We use the $L _ { 1 }$ loss for optimization:

$$
L _ { c o l o r } = | \mathbf { C } _ { k } - \hat { \mathbf { C } } _ { k } | , \quad L _ { d e p t h } = | \mathbf { D } _ { k } - \hat { \mathbf { D } } _ { k } | .\tag{7}
$$

The opacity learning rate $l r _ { \alpha }$ is set to 0 to fix opacity and we do not calculate depth loss on pixels with no intersection with Gaussians. At the same time, we hope the transparent Gaussians focus on refining the local color without affecting other areas. Hence, we design a regularization term $L _ { r e g } ,$ an $L _ { 2 }$ loss applied to all transparent Gaussians to constrain their geometry properties p, q, s remaining the same as their initial values. The overall loss function is defined as:

$$
L = w _ { c } L _ { c o l o r } + w _ { d } L _ { d e p t h } + w _ { r e g } L _ { r e g } .\tag{8}
$$

We use $w _ { c } = 1 , w _ { d } = 1 , w _ { r e g } = 1 0 0 0$ in all our tests. We use the regularization term instead of zero learning rate for transparent Gaussians simply because in our PyTorch implementation it is difficult to set different learning rates for different Gaussians. The Gaussians will be optimized through multiple iterations and the confidence count is incremented by 1 when SH is updated. We notice that after optimization, the Gaussians can fit the current time window well but the rendering quality of previous views will decline, making it challenging to obtain high realism rendering under all views. Therefore we use a weighted average method to fuse the current result with previous results. Denote each Gaussian after the optimization for the current window ?? as $G _ { o } ^ { \prime }$ , and the fused Gaussian properties are computed as:

$$
\begin{array} { l } { { G _ { o } = \big ( 1 - w _ { c u r r } \big ) G _ { o - 1 } + w _ { c u r r } G _ { o } ^ { \prime } , } } \\ { { \mathrm { } } } \\ { { \eta _ { o } = \eta _ { o } ^ { \prime } , w _ { c u r r } = \frac { \eta _ { o } ^ { \prime } - \eta _ { o - 1 } } { \eta _ { o } ^ { \prime } } . } } \end{array}\tag{9}
$$

The fusing strategy can effectively avoid the forgetting problem. Note optimization of all the Gaussians is still too time-consuming for real-time reconstruction. To this end, we consider Gaussians with $\eta _ { k } ~ > ~ \delta _ { \eta }$ stable and otherwise unstable. The stable Gaussians have well fit previous observations and will not be optimized, greatly reducing the number of Gaussians that need to be optimized. Meanwhile, we only need to focus on the pixels affected by the unstable Gaussians, avoiding the optimization on all pixels. In this way, the optimization can be done in real time. Please refer to the supplementary material for more details.

State management. Another key step is the mutual conversion between $\boldsymbol { S _ { s t a b l e } }$ and $S _ { u n s t a b l e }$ and the deletion of wrong Gaussians. We use the optimized scene $S ^ { * }$ to render the color image $\hat { \mathbf { C } } _ { k } ^ { * }$ , depth map $\hat { \mathbf { D } } _ { k } ^ { * } ,$ , normal map $\hat { \bf N } _ { k } ^ { * } ,$ , and index map $\hat { \mathbf { I } } _ { k } ^ { * }$ for frame ??. We also calculate the $L _ { 1 }$ difference for the color and depth compared with the RGBD input. For each stable Gaussian, if the corresponding color or depth error exceeds $\delta _ { \mathbf { c } }$ or $\delta _ { d } ,$ the error count $e _ { i }$ of this Gaussian is incremented by 1. Then we manage the Gaussian states according to the following conditions. The stable Gaussians with $e _ { i } > \delta _ { e }$ are converted to unstable while the unstable Gaussians with $\eta _ { i } > \delta _ { \eta }$ are converted to stable. The unstable Gaussians with $k - t _ { i } > \delta _ { t }$ are removed because they keep unstable for a long time, and are treated as outliers. Note the Gaussians have to be observed in certain views to be marked as stable. Even if they are occluded later all the time, they are not redundant.

Camera tracking. We utilize the frame-to-model ICP as the frontend odometry for camera tracking. Specifically, we use the optimized Gaussians in the previous frame to render the depth map $\hat { \mathbf { D } } _ { k - 1 } ^ { * }$ and normal map $\hat { \bf N } _ { k - 1 } ^ { * }$ , and convert $\hat { \mathbf { D } } _ { k - 1 } ^ { * }$ to the global space $\hat { \mathbf { V } } _ { k - 1 } ^ { \tilde { g } * }$ . Then given the current frame $\mathbf { v } _ { k } ^ { l } ,$ we aim to find the camera pose that minimizes the point-to-plane error between 3D backprojected vertices:

$$
E ( \pmb \xi ) = \sum \left\| \big ( \mathbf { T } _ { g , k } \mathbf { V } _ { k } ^ { l } ( \mathbf { u } ) - \hat { \mathbf { V } } _ { k - 1 } ^ { g * } ( \hat { \mathbf { u } } ) \big ) \cdot \hat { \mathbf { N } } _ { k - 1 } ^ { * } ( \hat { \mathbf { u } } ) \right\| .\tag{10}
$$

Here $\xi$ is the Lie algebra representation of $\mathbf { T } _ { g , k }$ . We run a multilevel ICP to solve the objective function as [Newcombe et al. 2011]. Meanwhile, in order to reduce the drift during the scanning of large scenes, we also run a back-end optimization thread similar to ORB-SLAM2 [2017]. While the pose estimation is finished, a set of 3D landmarks are also maintained. These landmarks are used for graph optimization in the back-end, enabling more accurate camera tracking.

Keyframes and global optimization. In the global optimization, we further optimize the Gaussians in the global scene. Our keyframe selection strategy is inspired by [Cao et al. 2018]. The keyframe list is constructed based on the camera motion. If the rotation angle relative to the last keyframe exceeds a threshold $\delta _ { a n g l e } ,$ or the relative translation exceeds $\delta _ { m o v e }$ , we add a new keyframe. We use $\delta _ { a n g l e } = 3 0 ^ { \circ }$ and $\delta _ { m o v e } = 0 . \vdots$ m in our experiments. Whenever a new keyframe is added, we optimize all the Gaussians in S using the latest keyframe and three randomly selected keyframes based on the same loss function as (8). In order to ensure the speed of global optimization, we only optimize the pixels with the top 40% color errors on each keyframe. Whatâs more, to avoid overfitting in the selected viewpoint, we do not update the position of the Gaussian during the global optimization and we use 0.1Ã the original learning rate to optimize the other parameters. When the scan is finished, we optimize S using all recorded keyframes with 10Ã the number of keyframes iterations.

## 4 EVALUATION

## 4.1 Experimental Setup

Implementation Details. We implemented our SLAM system on a desktop computer with an intel i9 13900KF CPU and an Nvidia RTX 4090 GPU. We implemented the mapping and tracking parts in Python using Pytorch framework and wrote custom CUDA kernels for rasterization and back propagation. We used an Azure Kinect as the RGBD camera for real-time scanning. Please refer to the supplementary material for more details.

Datasets. We conducted experiments on three public datasets: Replica [Straub et al. 2019], TUM-RGBD [Sturm et al. 2012], Scan-Net++ [Yeshwanth et al. 2023], and a self-scanned Azure dataset. Replica is the simplest benchmark as it contains synthetic, highly accurate and complete RGBD images. TUM-RGBD is a widely used dataset in the SLAM field for evaluating tracking accuracy because it provides accurate camera poses from an external motion capture system.

ScanNet++ is a large-scale dataset that couples together capture of high-quality and commodity-level geometry and color of indoor scenes. Its depth maps are rendered from models reconstructed from laser scanning. Different from other benchmarks, each camera pose in ScanNet++ is very far apart from one another. We also scanned real-world scenes by ourselves to build an Azure dataset, including corridor, storeroom, hotel room, home, office, ranging from $4 3 m ^ { 2 } { \sim } 1 0 0 m ^ { 2 }$

Baselines. We compare our method with existing state-of-the-art NeRF RGBD SLAM methods such as NICE-SLAM [Zhu et al. 2022b], Point-SLAM [SandstrÃ¶m et al. 2023], Co-SLAM [Wang et al. 2023], ESLAM [Johari et al. 2023] and a concurrent Gaussian SLAM work SplaTAM [Keetha et al. 2023] (the only one with code released). We reproduce the results using the official code and run all experiments on the same desktop computer. Most of the experimental parameters follow their settings on the Replica dataset and we only adjust the bounding box setting based on the size of the new scenes. For ScanNet++, we double the scene (or map in SLAM) update frequency for all methods to ensure a fair comparison because of the sparsity of viewpoints.

Table 1: Comparison of time and memory performance on Replica (Off 0) and Azure Dataset (Home). Here â means out of memory.
<table><tr><td>Method</td><td>Dataset</td><td>Tracking /Frame</td><td>Mapping /Iteration</td><td>Mapping /Frame</td><td>FPS</td><td>Model Size (MB)</td><td>Memory Cost (MB)</td></tr><tr><td>NICE-SLAM [2022b]</td><td>Replica</td><td>1.05s</td><td>60.9ms</td><td>1.03s</td><td>0.95</td><td>87</td><td>9890</td></tr><tr><td></td><td>Azure</td><td>0.68s</td><td>116.5ms</td><td>1.58s</td><td>0.63</td><td>136</td><td>10057</td></tr><tr><td>Co-SLAM [2023]</td><td>Replica</td><td>0.11s</td><td>7.8ms</td><td>0.10s</td><td>9.26</td><td>7</td><td>7899</td></tr><tr><td></td><td>Azure</td><td>0.11s</td><td>7.2ms</td><td>0.12s</td><td>8.65</td><td>7</td><td>17342</td></tr><tr><td>ESLAM [2023]</td><td>Replica</td><td>0.15s</td><td>16.7ms</td><td>0.10s</td><td>6.80</td><td>46</td><td>18777</td></tr><tr><td>Point-SLAM</td><td>Azure</td><td>0.13s</td><td>15.4ms</td><td>0.11s</td><td>7.54</td><td>139</td><td>Ã</td></tr><tr><td>[2023]</td><td>Replica</td><td>1.05s</td><td>38.1ms</td><td>2.27s</td><td>0.44</td><td>15431</td><td>9890</td></tr><tr><td>SplaTAM</td><td>Azure</td><td>4.54s</td><td>68.4ms</td><td>4.00s</td><td>0.22</td><td>42536</td><td>9950</td></tr><tr><td>[2023]</td><td>Replica</td><td>1.16s</td><td>32.1ms</td><td>1.96s</td><td>0.51</td><td>310</td><td>9166</td></tr><tr><td></td><td>Azure</td><td>2.00s</td><td>53.4ms</td><td>3.22s</td><td>0.31</td><td>520</td><td>Ã</td></tr><tr><td>Ours</td><td>Replica</td><td>0.02s</td><td>3.5ms</td><td>0.05s</td><td>17.24</td><td>71</td><td>2751</td></tr><tr><td></td><td>Azure</td><td>0.03s</td><td>4.3ms</td><td>0.05s</td><td>17.90</td><td>399</td><td>8782</td></tr></table>

## 4.2 Evaluation of Online Reconstruction

Time/memory performance. Following [SandstrÃ¶m et al. 2023], we report the time per iteration for mapping optimization (e.g., NeRF optimization and Gaussian optimization), the tracking and mapping time per frame, the whole reconstruction FPS, the maximum memory usage during the SLAM process, and the final size of reconstructed scene on Replica office 0 and the home scene (around 70??2) of our Azure dataset in Table 1.

We can see our reconstruction speed is around twice that of NeRF SLAM methods and about 46Ã that of SplaTAM which is also based on 3D Gaussians. Notably, the memory cost of our method is much smaller compared to other methods, which allows us to scan largescale environments. Note SplaTAM uses alpha blending to render depths as colors, thus yielding much more Gaussians (7155880 before out of memory in the home scene) than our compact Gaussian representation (987524). Even though they store the RGB values instead of spherical harmonics to reduce the memory overhead, their memory cost is still very high and runs out of memory in the home scene.

Tracking accuracy. The camera tracking accuracy on the realworld dataset TUM-RGBD is reported in Table 2, and we report the results on the synthetic Replica dataset in the supplementary material. Our method outperforms the NeRF SLAM methods and concurrent Gaussian SLAM method on both datasets, and achieves tracking accuracy comparable with classical SLAM methods on the real-world data.

Novel view synthesis. We qualitatively compare the rendering quality for novel view synthesis for all methods. The results are shown in Fig. 4. Note the NeRF-based methods require a depth map to synthesize high-quality images, so we use the reconstructed mesh to render depth maps for them. We also quantitatively compare the novel-view synthesis on the ScanNet++ testing views, where the ground truth depth is used for NeRF-based methods, and the results are reported in the supplementary material. We can see that our method and SplaTAM clearly produces higher quality images with much fewer artifacts and higher fidelity appearance. We also quantitatively compare the rendering quality with other methods on the training views of Replica, following Point-SLAM and all concurrent Gaussian SLAM works. Please see the table in the supplementary material. Our method achieves a rendering quality comparable with SplaTAM and Point-SLAM (which needs the ground-truth depth map as input), and consistently outperforms the other NeRF SLAM methods.

Table 2: Comparison of tracking accuracy (unit: ????) on TUM-RGBD.
<table><tr><td>Method</td><td>fr1_desk</td><td>fr2_xyz</td><td>fr3_office</td><td>Avg.</td></tr><tr><td>NICE-SLAM[2022b]</td><td>4.30</td><td>31.73</td><td>3.87</td><td>13.28</td></tr><tr><td>Co-SLAM[2023]</td><td>2.92</td><td>1.75</td><td>3.55</td><td>2.74</td></tr><tr><td>ESLAM[2023]</td><td>2.49</td><td>1.11</td><td>2.74</td><td>2.11</td></tr><tr><td>Point-SLAM[2023]</td><td>2.56</td><td>1.20</td><td>3.37</td><td>2.38</td></tr><tr><td>SplaTAM[2023]</td><td>3.33</td><td>1.55</td><td>5.28</td><td>3.39</td></tr><tr><td>Ours</td><td>1.66</td><td>0.38</td><td>1.13</td><td>1.06</td></tr><tr><td>ElasticFusion[2015]</td><td>2.53</td><td>1.17</td><td>2.52</td><td>2.07</td></tr><tr><td>ORB-SLAM2[2017]</td><td>1.60</td><td>0.40</td><td>1.00</td><td>1.00</td></tr><tr><td>BAD-SLAM[2019]</td><td>1.70</td><td>1.10</td><td>1.70</td><td>1.50</td></tr></table>

Reconstruction quality. Following NICE-SLAM [Zhu et al. 2022b], we use the metrics including Accuracy, Completion, Accuracy Ratio[<3cm] and Completion Ratio[<3cm] to evaluate the scene geometry on ScanNet++. We remove unseen regions that are not inside any cameraâs frustum. For the NeRF SLAM methods, the meshes produced by marching cubes with 1???? voxel size are used for evaluation. For Point-SLAM, as mentioned in the paper, we use the re-rendered depth maps for TSDF Fusion. For SplaTAM and ours, we uniformly sample an equal amount of points from the reconstructed Gaussians for evaluation. To eliminate the impact of tracking accuracy, we use the ground truth camera pose for reconstruction and the results are reported in Table 3. Please note that Point-SLAM does not optimize the locations of neural points, so in this experiment its depth is always correct, thus always obtaining accurate geometry. Nevertheless, our geometry accuracy outperforms other methods except Point-SLAM, and achieves comparable completion results. It demonstrates that our compact Gaussians can accurately fit surfaces with a small number of Gaussians. We also demonstrate a qualitative comparison of reconstruction results and novel view synthesis in Fig.5. Note SplaTAM and ESLAM run out of memory in the scene. The top-view scenes in the teaser and Fig. 5 are directly rendered from Gaussians without mesh extraction for SplaTAM and ours. We can see our method can achieve comparable high-quality reconstruction as the state-of-the-art NeRF SLAM methods, and surpass them in novel view synthesis. We further illustrate our reconstruction and novel view synthesis results on our real captured scenes in Fig. 8.

Table 3: Comparison of geometry accuracy on ScanNet++.
<table><tr><td>Method</td><td>Acc.â</td><td>Acc. Ratioâ</td><td>Comp.â</td><td>Comp. Ratioâ</td></tr><tr><td>NICE-SLAM[2022b]</td><td>4.45</td><td>74.49</td><td>2.04</td><td>86.63</td></tr><tr><td>Co-SLAM[2023]</td><td>5.26</td><td>78.86</td><td>1.06</td><td>96.25</td></tr><tr><td>ESLAM[2023]</td><td>4.43</td><td>74.51</td><td>1.05</td><td>97.42</td></tr><tr><td>Point-SLAM[2023]</td><td>0.67</td><td>99.12</td><td>0.68</td><td>98.94</td></tr><tr><td>SplaTAM[2023]</td><td>1.32</td><td>95.31</td><td>1.54</td><td>93.55</td></tr><tr><td>Ours</td><td>0.95</td><td>96.41</td><td>1.11</td><td>97.16</td></tr></table>

## 4.3 Ablation studies

We evaluate the compact Gaussian representation here. Please see the supplementary material for the evaluation on stable/unstable Gaussians, and the sampled pixel number for Gaussians.

Compact Gaussian Representation. To prove the effectiveness of our compact Gaussian representation, we randomly select 20 RGBD images on Replica, and uniformly sample a certain number of pixels to initialize and optimize Guassians to fit the RGBD images. We compare the fitting results between our compact Gaussians and the original Gaussians using alpha blending. As shown in Fig. 7 left, our compact Gaussian representation requires much fewer Gaussians than the original Gaussian representation to reach the same depth accuracy. Also our compact Gaussian representation can better fit surfaces with the same amount of Gaussians (Fig. 7 right).

We then assess the necessity of transparent Gaussians in the compact Gaussian representation. We show a reconstruction result versus the result trained using only opaque Gaussians in Figure 6. We can see that the pure opaque Gaussians will obscure the existing Gaussians during the color blending process, leading to significant color errors from new views.

## 5 CONCLUSION

We present a real-time 3D reconstruction system for large-scale environments using Gaussian splatting. We introduce a compact Gaussian representation to reduce the number of Gaussians needed to fit the surface, thereby greatly reducing the memory and computation cost. For on-the-fly Gaussian optimization, we explicitly add Gaussians for three types of pixels per frame: newly observed, with large color errors and with large depth errors, and only optimize the unstable Gaussians and only render the pixels occupied by unstable Gaussians. We reconstruct large-scale real scanning scenes, and achieve better performance than both the state-of-the-art NeRF SLAM method and the concurrent Gaussian SLAM methods. Because only opaque Gaussians and transparent Gaussians are used to represent the scene in order to reach real time reconstruction at scale, our rendering quality is inevitably degraded compared with original Gaussians. How to improve the rendering quality while keeping real-time performance is worth exploring in the future. Besides, the reflective or transparent materials can cause the surface color largely varying across different views, making some Gaussians frequently switch between two states and not optimized well. In the future we will also extend our system to handle outdoor scenes, dynamic objects, fast camera movement, and scenes with changing lightings.

## ACKNOWLEDGMENTS

The authors would like to thank the reviewers for their insightful comments. This work is supported by NSF China (No. U23A20311 & 62322209), the XPLORER PRIZE, and the 100 Talents Program of Zhejiang University. The source code and data are available at https://gapszju.github.io/RTG-SLAM.

## REFERENCES

Sebastien Bonopera, Jerome Esnault, Siddhant Prakash, Simon Rodriguez, Theo Thonat, Mehdi Benadel, Gaurav Chaurasia, Julien Philip, and George Drettakis. 2020. sibr: A System for Image Based Rendering. https://gitlab.inria.fr/sibr/sibr_core

Yan-Pei Cao, Leif Kobbelt, and Shi-Min Hu. 2018. Real-Time High-Accuracy Three-Dimensional Reconstruction with Consumer RGB-D Cameras. ACM Trans. Graph. 37, 5, Article 171 (sep 2018), 16 pages. https://doi.org/10.1145/3182157

Jiawen Chen, Dennis Bautembach, and Shahram Izadi. 2013. Scalable real-time volumetric surface reconstruction. ACM Trans. Graph. 32, 4 (2013), 113:1â113:16. https://doi.org/10.1145/2461912.2461940

Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee. 2024. Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images. arXiv:2311.13398 [cs.CV]

Angela Dai, Matthias NieÃner, Michael ZollhÃ¶fer, Shahram Izadi, and Christian Theobalt. 2017. BundleFusion: Real-Time Globally Consistent 3D Reconstruction Using On-the-Fly Surface Reintegration. ACM Trans. Graph. 36, 4, Article 76a (jul 2017), 18 pages. https://doi.org/10.1145/3072959.3054739

Hao Du, Peter Henry, Xiaofeng Ren, Marvin Cheng, Dan B. Goldman, Steven M. Seitz, and Dieter Fox. 2011. Interactive 3D modeling of indoor environments with a consumer depth camera. In UbiComp 2011: Ubiquitous Computing, 13th International Conference, UbiComp 2011, Beijing, China, September 17-21, 2011, Proceedings, James A. Landay, Yuanchun Shi, Donald J. Patterson, Yvonne Rogers, and Xing Xie (Eds.). ACM, 75â84. https://doi.org/10.1145/2030112.2030123

Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. 2023b. Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras. CoRR abs/2311.16728 (2023). https://doi.org/10.48550/ ARXIV.2311.16728 arXiv:2311.16728

Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, and Shi-Min Hu. 2021. DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Shi-Sheng Huang, Haoxiang Chen, Jiahui Huang, Hongbo Fu, and Shi-Min Hu. 2023a. Real-Time Globally Consistent 3D Reconstruction With Semantic Priors. IEEE Transactions on Visualization and Computer Graphics 29, 4 (2023), 1977â1991. https: //doi.org/10.1109/TVCG.2021.3137912

Mohammad Mahdi Johari, Camilla Carta, and FranÃ§ois Fleuret. 2023. ESLAM: Efficient Dense SLAM System Based on Hybrid Representation of Signed Distance Fields. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 17408â17419. https://doi.org/10. 1109/CVPR52729.2023.01670

Nikhil Varma Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian A. Scherer, Deva Ramanan, and Jonathon Luiten. 2023. SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM. CoRR abs/2312.02126 (2023). https://doi.org/10.48550/ARXIV.2312.02126 arXiv:2312.02126

Maik Keller, Damien Lefloch, Martin Lambers, Shahram Izadi, Tim Weyrich, and Andreas Kolb. 2013. Real-Time 3D Reconstruction in Dynamic Scenes Using Point-Based Fusion. In 2013 International Conference on 3D Vision, 3DV 2013, Seattle, Washington, USA, June 29 - July 1, 2013. IEEE Computer Society, 1â8. https: //doi.org/10.1109/3DV.2013.9

Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4 (2023), 139:1â139:14. https://doi.org/10.1145/3592433

Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. 2023. Gaussian Splatting SLAM. arXiv:2312.06741 [cs.CV]

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part I (Lecture Notes in Computer Science, Vol. 12346), Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm (Eds.). Springer, 405â421. https://doi.org/10.1007/978-3-030-58452-8_24

RaÃºl Mur-Artal and Juan D. TardÃ³s. 2017. ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras. IEEE Transactions on Robotics 33, 5 (2017), 1255â1262. https://doi.org/10.1109/TRO.2017.2705103

Richard A. Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J. Davison, Pushmeet Kohli, Jamie Shotton, Steve Hodges, and Andrew W. Fitzgibbon. 2011. KinectFusion: Real-time dense surface mapping and tracking. In 10th IEEE International Symposium on Mixed and Augmented Reality, ISMAR 2011, Basel, Switzerland, October 26-29, 2011. IEEE Computer Society, 127â136. https://doi.org/10.1109/ISMAR.2011.6092378

Matthias NieÃner, Michael ZollhÃ¶fer, Shahram Izadi, and Marc Stamminger. 2013. Real-time 3D reconstruction at scale using voxel hashing. ACM Trans. Graph. 32, 6 (2013), 169:1â169:11. https://doi.org/10.1145/2508363.2508374

Erik SandstrÃ¶m, Yue Li, Luc Van Gool, and Martin R. Oswald. 2023. Point-SLAM: Dense Neural Point Cloud-based SLAM. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Thomas SchÃ¶ps, Torsten Sattler, and Marc Pollefeys. 2019. BAD SLAM: Bundle Adjusted Direct RGB-D SLAM. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. Computer Vision Foundation / IEEE, 134â144. https://doi.org/10.1109/CVPR.2019.00022

Frank SteinbrÃ¼cker, Christian Kerl, and Daniel Cremers. 2013. Large-Scale Multiresolution Surface Reconstruction from RGB-D Sequences. In IEEE International Conference on Computer Vision, ICCV 2013, Sydney, Australia, December 1-8, 2013. IEEE Computer Society, 3264â3271. https://doi.org/10.1109/ICCV.2013.405

Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. 2019. The Replica Dataset: A Digital Replica of Indoor Spaces. arXiv preprint arXiv:1906.05797 (2019).

JÃ¼rgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. 2012. A benchmark for the evaluation of RGB-D SLAM systems. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. 573â580. https://doi. org/10.1109/IROS.2012.6385773

Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davison. 2021a. iMAP: Implicit Mapping and Positioning in Real-Time. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021. IEEE, 6209â6218. https://doi.org/10.1109/ICCV48922.2021.00617

Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davison. 2021b. iMAP: Implicit Mapping and Positioning in Real-Time. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021. IEEE, 6209â6218. https://doi.org/10.1109/ICCV48922.2021.00617

Hengyi Wang, Jingwen Wang, and Lourdes Agapito. 2023. Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 13293â13302. https://doi.org/10.1109/CVPR52729.2023.01277

Thomas Whelan, Stefan Leutenegger, Renato F Salas-Moreno, Ben Glocker, and Andrew J Davison. 2015. ElasticFusion: Dense SLAM without a pose graph.. In Robotics: science and systems, Vol. 11. Rome, Italy, 3.

Yabin Xu, Liangliang Nan, Laishui Zhou, Jun Wang, and Charlie C. L. Wang. 2022. HRBF-Fusion: Accurate 3D Reconstruction from RGB-D Data Using On-the-fly Implicits. ACM Trans. Graph. 41, 3, Article 35 (apr 2022), 19 pages. https://doi.org/ 10.1145/3516521

Chi Yan, Delin Qu, Dong Wang, Dan Xu, Zhigang Wang, Bin Zhao, and Xuelong Li. 2023. GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting. CoRR abs/2311.11700 (2023). https://doi.org/10.48550/ARXIV.2311.11700 arXiv:2311.11700

Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. 2022. Vox-Fusion: Dense Tracking and Mapping with Voxel-based Neural Implicit Representation. In IEEE International Symposium on Mixed and Augmented Reality, ISMAR 2022, Singapore, October 17-21, 2022, Henry B. L. Duh, Ian Williams, Jens Grubert, J. Adam Jones, and Jianmin Zheng (Eds.). IEEE, 499â507. https://doi.org/ 10.1109/ISMAR55827.2022.00066

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. 2023. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. arXiv preprint arXiv:2309.13101 (2023).

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias NieÃner, and Angela Dai. 2023. Scan-Net++: A High-Fidelity Dataset of 3D Indoor Scenes. In Proceedings of the International Conference on Computer Vision (ICCV).

Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Oswald. 2023. Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting. arXiv:2312.10070 [cs.CV]

Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo Poggi. 2023. GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Yizhong Zhang, Weiwei Xu, Yiying Tong, and Kun Zhou. 2015. Online Structure Analysis for Real-Time Indoor Scene Reconstruction. ACM Trans. Graph. 34, 5 (2015), 159:1â159:13. https://doi.org/10.1145/2768821

Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. 2022a. NICE-SLAM: Neural Implicit Scalable Encoding for SLAM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. 2022b. NICE-SLAM: Neural Implicit Scalable Encoding for SLAM. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE, 12776â12786. https://doi.org/10.1109/CVPR52688.2022.01245

Michael ZollhÃ¶fer, Patrick Stotko, Andreas GÃ¶rlitz, Christian Theobalt, Matthias NieÃner, Reinhard Klein, and Andreas Kolb. 2018. State of the Art on 3D Reconstruction with RGB-D Cameras. Comput. Graph. Forum 37, 2 (2018), 625â652. https://doi.org/10.1111/CGF.13386

Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and Markus Gross. 2001. Surface splatting. In Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH â01). Association for Computing Machinery, New York, NY, USA, 371â378. https://doi.org/10.1145/383259.383300

<!-- image-->  
Figure 4: Comparison of novel view synthesis on ScanNet++.

<!-- image-->  
Figure 5: Comparison of reconstruction quality and novel view synthesis in real scanned hotel room scene.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

Figure 6: Ablation study on transparent Gaussians.  
<!-- image-->  
Figure 7: Ablation study on compact Gaussian Representation.

<!-- image-->  
Figure 8: Reconstruction results and novel view synthesis results of real scenes using our system.

## SUPPLEMENTARY MATERIALS

## A GAUSSIAN INITIALIZATION

Here we introduce how we compute the covariance matrix $\Sigma _ { \mathbf { u } }$ for each newly added Gaussian for pixel u in detail. Each new Gaussian $G _ { \mathbf { u } }$ is initialized as a flat circle disc. For opaque Gaussians, our goal is to cover the surface as much as possible without largely impacting existing ones. For this, we calculate the distance from the pixelâs 3D position $\mathbf { V } _ { k } ^ { g } ( \mathbf { u } )$ to its three nearest Gaussians $G _ { 1 , 2 , 3 }$ in the scene, and initialize its scale based on the following formula:

$$
\begin{array} { l } { { \displaystyle { \bf s } _ { \bf u , 1 } = \sqrt { \frac { 1 } { 3 } \sum _ { i = 1 } ^ { 3 } \left( \vert \vert { \bf V } _ { k } ^ { g } ( { \bf u } ) - { \bf p } _ { i } \vert \vert - 0 . 5 ( a _ { i } + b _ { i } ) \right) } } , } \\ { { \displaystyle { \bf s } _ { { \bf u } , 2 } = { \bf s } _ { { \bf u } , 1 } \quad { \bf s } _ { { \bf u } , 3 } = 0 . 1 { \bf s } _ { { \bf u } , 1 } } . } \end{array}\tag{11}
$$

Here ?? is the biggest eigenvalue of the Gaussian covariance matrix and ?? is the second largest eigenvalue. For the transparent Gaussian, the ratios of its three axes are also set to 1:1:0.1. However, in order to reduce the disruption of the rendering results on other pixels, its maximum scale is limited to 0.01 ??. Finally, the Gaussian orientation qu is initialized such that its shortest axis aligns with the pixelâs global normal $\mathbf { N } _ { k } ^ { g } ( \mathbf { u } )$ . The above initialization method allows the newly added opaque Gaussians to cover the scene surface as much as possible, with little influence on the existing Gaussians in the scene S.

## B GAUSSIAN OPTIMIZATION

Since we only optimize the unstable Gaussians, we use $\boldsymbol { S _ { u n s t a b l e } }$ to render a light transmission map $\hat { \mathbf { T } } ^ { - - }$ ???????????????? before optimization, and the loss ?? is only computed on the pixels:

$$
\begin{array} { r } { M _ { u n s t a b l e } = \{ \mathbf { u } _ { u n s t a b l e } | \hat { \mathbf { T } } _ { u n s t a b l e } ( \mathbf { u } _ { u n s t a b l e } ) < 1 \} . } \end{array}\tag{12}
$$

However, in order to achieve fast rendering and optimization, [Kerbl et al. 2023] adopts a tile-based rasterizer for Gaussian splatting: the image is divided into $1 6 \times 1 6$ tiles, and each Gaussian is assigned a key that combines view space depth and tile ID and then sorted. In fact, when there are just a few pixels within a tile that need to be rendered, instantiating all Gaussians on this tile is quite inefficient. Therefore, we discard those tiles where the number of pixels that need to be rendered is less than 50%. This strategy drastically reduces inefficient computation, and increases the overall speed of the optimization process.

## C IMPLEMENTATION DETAILS

We accelerate our system by implementing it in parallel with three threads: one thread for Gaussians optimization, one for front-end online tracking, and the other for back-end graph optimization. The Gaussians optimization and front-end online tracking are implemented by python using the pytorch framework and we write custom CUDA kernels for our rendering process and back propagation. The back-end optimization part is inherited from ORB SLAM2 [Mur-Artal and TardÃ³s 2017] and implemented in C++. We also build an interactive viewer using the open-source SIBR [Bonopera et al. 2020; Kerbl et al. 2023] to visualize the SLAM process and the reconstructed model. In order to achieve free movement and scanning in the scene, we use a laptop with an intel i7 10750-H CPU and nvidia 2070 GPU connected to an Azure Kinect RGBD camera for data acquisition. The RGBD images captured by the camera are transmitted to the desktop computer through a wireless network and the desktop computer completes the SLAM computations, and then the results are sent back to our viewer on the laptop for visualization. In all our experiments, we sample 5% of pixels for the Gaussians adding. For small-scale synthetic dataset Replica [Straub et al. 2019], we set the optimization time window to 6 and optimize 50 iterations. For our large-scale real Azure dataset, we set the optimization time window to 8 and optimize 50 iterations. For TUM-RGBD [Sturm et al. 2012], we set the optimization time window to 4 and optimize 50 iterations. For ScanNet++ [Yeshwanth et al. 2023], we set the optimization time window to 3 and optimize 75 iterations due to the high-resolution (1752 Ã 1168) and sparse viewports. For learning rates, we set $l r _ { p o s i t i o n } = 0 . 0 0 1 , l r _ { S H 0 } = 0 . 0 0 0 5 , l r _ { \alpha } = 0 _ { \mathrm { { } } }$ $l r _ { s c a l e } = 0 . 0 0 4 , l r _ { r o t a t i o n } = 0 . 0 0 1$ on Replica and ScanNet++. And we set $l r _ { p o s i t i o n } = 0 . 0 0 1 , l r _ { S H 0 } = 0 . 0 0 1 , l r _ { \alpha } = 0 , l r _ { s c a l e } = 0 . 0 0 2 ,$ $l r _ { r o t a t i o n } = 0 . 0 0 1$ on our datset and TUM-RGBD. The learning rates of other SH coefficients is $0 . 0 5 \times l r _ { S H 0 }$ . For the confidence count threshold, we use $\delta _ { \eta } = 1 0 0$ for synthetic Replica, $\delta _ { \eta } = 2 0 0$ for TUM-RGBD and $\delta _ { \eta } = 4 0 0$ for large-scale Azure and ScanNet++ scenes.

Table 4: Statistics of Azure dataset
<table><tr><td>Statistic</td><td>corridor</td><td>storeroom</td><td>hotel room</td><td>home</td><td>office</td></tr><tr><td>Trajectory Length (m)</td><td>21.9</td><td>18.9</td><td>39.7</td><td>32.2</td><td>41.0</td></tr><tr><td>Scan Area (m2)</td><td>43.4</td><td>44.3</td><td>56.3</td><td>69.8</td><td>100.2</td></tr><tr><td>Frame Number</td><td>4890</td><td>3310</td><td>4838</td><td>6130</td><td>6889</td></tr></table>

## D DATASET DETAILS

For ScanNet++, we select 4 subsets (8b5caf3398, 39f36da05b, b20a261fdf, f34d532901) for evaluation. The statistics of Azure dataset are listed in Table 4. Note that we donât have the ground truth for Azure dataset, so it is mainly used for qualitative demonstration (except the evaluation of time & memory performance).

## E MORE COMPARISON RESULTS

## E.1 Quantitative results on different datasets

We add more evaluation on different datasets here. Please note that the low quality 3D model of TUM-RGBD makes it infeasible to evaluate geometry accuracy, as in previous papers. The cameras far apart in ScanNet++ result in tracking failure for classical, NeRF and our SLAM, so only geometry accuracy is evaluated. Our Azure dataset doesnât have the groundtruth camera or 3D model. The comparison of geometry accuracy on Replica is shown in Table 5. Similar to the paper, our geometry accuracy still outperforms the other methods except Point-SLAM which doesnât optimize point positions from the perfect depth. SplaTAM and ours degrade in completion because Gaussians cannot complete unscanned regions as NeRF. The comparison of tracking accuracy on Replica are shown in Table 6. Our method consistently outperforms the NeRF SLAM and the concurrent Gaussian SLAM method. We believe this is due to our back-end graph optimization based on ORB landmarks. Please note that in order to ensure fairness in comparison, although there is no noise in the depth images on Replica, we still used frame-to-model ICP, just without applying the bilateral filter to the depth input. We also report the time and memory performance on TUM-RGBD in Table 7. Our system achieves the highest scanning speed and lowest memory cost.

Table 5: Comparison of geometry accuracy on Replica.
<table><tr><td>Method</td><td>Acc.â</td><td>Acc. Ratioâ</td><td>Comp.â</td><td>Comp. Ratioâ</td></tr><tr><td>NICE-SLAM[2022b]</td><td>2.84</td><td>84.44</td><td>2.31</td><td>84.97</td></tr><tr><td>Co-SLAM[2023]</td><td>2.33</td><td>88.89</td><td>1.63</td><td>89.94</td></tr><tr><td>ESLAM[2023]</td><td>1.47</td><td>91.44</td><td>1.11</td><td>93.84</td></tr><tr><td>Point-SLAM[2023]</td><td>0.61</td><td>99.94</td><td>2.42</td><td>86.85</td></tr><tr><td>SplaTAM[2023]</td><td>2.88</td><td>73.89</td><td>3.57</td><td>71.68</td></tr><tr><td>Ours</td><td>0.75</td><td>98.87</td><td>2.81</td><td>82.76</td></tr></table>

Table 6: Comparison of tracking accuracy (unit: ????) on Replica.
<table><tr><td>Method</td><td>Rm 0</td><td>Rm 1</td><td>Rm 2</td><td>Off 0</td><td>Off 1</td><td>Off 2</td><td>Off 3</td><td>Off 4</td><td>Avg.</td></tr><tr><td>NICE-SLAM[2022b]</td><td>0.97</td><td>1.31</td><td>1.07</td><td>0.88</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>1.06</td></tr><tr><td>Co-SLAM[2023]</td><td>0.69</td><td>0.59</td><td>0.73</td><td>0.87</td><td>0.47</td><td>2.16</td><td>1.30</td><td>0.62</td><td>0.93</td></tr><tr><td>ESLAM[2023]</td><td>0.66</td><td>0.62</td><td>0.55</td><td>0.44</td><td>0.43</td><td>0.50</td><td>0.66</td><td>0.53</td><td>0.55</td></tr><tr><td>Point-SLAM[2023]</td><td>0.54</td><td>0.43</td><td>0.34</td><td>0.36</td><td>0.45</td><td>0.44</td><td>0.63</td><td>0.72</td><td>0.49</td></tr><tr><td>SplaTAM[2023]</td><td>0.47</td><td>0.42</td><td>0.32</td><td>0.46</td><td>0.24</td><td>0.28</td><td>0.39</td><td>0.56</td><td>0.39</td></tr><tr><td>Ours</td><td>0.20</td><td>0.18</td><td>0.13</td><td>0.22</td><td>0.12</td><td>0.22</td><td>0.20</td><td>0.19</td><td>0.18</td></tr></table>

Table 7: Comparison of time and memory performance on TUM-RGBD.
<table><tr><td>Method</td><td>FPSâ</td><td>Memory (MB)â</td></tr><tr><td>NICE-SLAM[2022b]</td><td>0.06</td><td>9930</td></tr><tr><td>CO-SLAM[2023]</td><td>7.18</td><td>18607</td></tr><tr><td>ESLAM[2023]</td><td>0.30</td><td>18617</td></tr><tr><td>Point-SLAM[2023]</td><td>0.26</td><td>11000</td></tr><tr><td>SplaTAM[2023]</td><td>0.14</td><td>12100</td></tr><tr><td>Ours</td><td>21.74</td><td>3563</td></tr></table>

Finally we compare the rendering quality. The results of training view synthesis quality on Replica are reported in Table 8. Please note that this comparison is actually unfair as Point-SLAM [2023] takes the ground-truth depth maps as input to help sampling the 3D volume for rendering. In contrast, our method and SplaTAM [2023] do not require any auxiliary input. Even so, our method still achieves a rendering quality comparable with Point-SLAM and SplaTAM, and consistently outperforms the other NeRF SLAM methods. The quantitative comparison of novel-view synthesis on ScanNet++ testing views is reported in Table 9. Our method is comparable to SplaTAM and outperforms the other NeRF-SLAM methods.

## E.2 Comparison with more SLAM systems

We show more comparisons with ElasticFusion [Whelan et al. 2015], BundleFusion [Dai et al. 2017], and GO-SLAM [Zhang et al. 2023]. The tracking accuracy is evaluated on Replica and TUM-RGBD and the results are listed in Table 10. we also evaluate the geometric accuracy on Replica as shown in Table 11. Our method achieves comparable geometry accuracy as BundleFusion. GO-SLAMâs completion numbers are worse than those of its paper due to considering unscanned regions for fair comparison.

Table 8: Comparison of train view synthesis on Replica.
<table><tr><td>Method</td><td>Metric</td><td>Rm 0</td><td>Rm 1</td><td>Rm 2</td><td>Off 0</td><td>Off 1</td><td>Off 2</td><td>Off 3</td><td>Off 4</td><td>Avg.</td></tr><tr><td rowspan="3">NICE-SLAM [2022b]</td><td>PSNRâ</td><td>24.72</td><td>26.79</td><td>27.06</td><td>30.21</td><td>32.78</td><td>26.59</td><td>26.22</td><td>24.74</td><td>27.39</td></tr><tr><td>SSIMâ</td><td>0.787</td><td>0.799</td><td>0.807</td><td>0.881</td><td>0.906</td><td>0.816</td><td>0.801</td><td>0.834</td><td>0.829</td></tr><tr><td>LPIPSâ</td><td>0.431</td><td>0.372</td><td>0.329</td><td>0.322</td><td>0.275</td><td>0.321</td><td>0.288</td><td>0.333</td><td>0.334</td></tr><tr><td rowspan="3">Co-SLAM [2023]</td><td>PSNRâ</td><td>28.88</td><td>28.51</td><td>29.37</td><td>35.44</td><td>34.63</td><td>26.56</td><td>28.79</td><td>32.16</td><td>30.54</td></tr><tr><td>SSIMâ</td><td>0.892</td><td>0.843</td><td>0.851</td><td>0.854</td><td>0.826</td><td>0.814</td><td>0.866</td><td>0.856</td><td>0.850</td></tr><tr><td>LPIPSâ</td><td>0.213</td><td>0.205</td><td>0.215</td><td>0.177</td><td>0.181</td><td>0.172</td><td>0.163</td><td>0.176</td><td>0.188</td></tr><tr><td rowspan="3">ESLAM [2023]</td><td>PSNRâ</td><td>26.96</td><td>28.98</td><td>29.80</td><td>35.04</td><td>33.81</td><td>30.08</td><td>30.01</td><td>31.34</td><td>30.75</td></tr><tr><td>SSIMâ</td><td>0.821</td><td>0.837</td><td>0.843</td><td>0.902</td><td>0.873</td><td>0.865</td><td>0.881</td><td>0.886</td><td>0.863</td></tr><tr><td>LPIPSâ</td><td>0.171</td><td>0.173</td><td>0.187</td><td>0.172</td><td>0.181</td><td>0.186</td><td>0.172</td><td>0.174</td><td>0.177</td></tr><tr><td rowspan="3">Point-SLAM [2023]</td><td>PSNRâ</td><td>32.40</td><td>34.08</td><td>35.50</td><td>38.26</td><td>39.16</td><td>33.99</td><td>33.48</td><td>33.49</td><td>35.17</td></tr><tr><td>SSIMâ</td><td>0.974</td><td>0.977</td><td>0.982</td><td>0.983</td><td>0.986</td><td>0.960</td><td>0.960</td><td>0.979</td><td>0.975</td></tr><tr><td>LPIPSâ</td><td>0.113</td><td>0.116</td><td>0.111</td><td>0.100</td><td>0.118</td><td>0.156</td><td>0.132</td><td>0.142</td><td>0.124</td></tr><tr><td rowspan="3">SplaTAM [2023]</td><td>PSNRâ</td><td>32.31</td><td>33.36</td><td>34.78</td><td>38.16</td><td>38.49</td><td>31.66</td><td>29.24</td><td>31.54</td><td>33.69</td></tr><tr><td>SSIMâ</td><td>0.974</td><td>0.966</td><td>0.983</td><td>0.982</td><td>0.980</td><td>0.962</td><td>0.948</td><td>0.946</td><td>0.968</td></tr><tr><td>LPIPSâ</td><td>0.072</td><td>0.101</td><td>0.073</td><td>0.084</td><td>0.095</td><td>0.102</td><td>0.123</td><td>0.157</td><td>0.101</td></tr><tr><td rowspan="3">Ours</td><td>PSNRâ</td><td>31.56</td><td>34.21</td><td>35.57</td><td>39.11</td><td>40.27</td><td>33.54</td><td>32.76</td><td>36.48</td><td>35.43</td></tr><tr><td>SSIMâ</td><td>0.967</td><td>0.979</td><td>0.981</td><td>0.990</td><td>0.992</td><td>0.981</td><td>0.981</td><td>0.984</td><td>0.982</td></tr><tr><td>LPIPSâ</td><td>0.131</td><td>0.105</td><td>0.115</td><td>0.068</td><td>0.075</td><td>0.134</td><td>0.128</td><td>0.117</td><td>0.109</td></tr></table>

Table 9: Comparison of novel view synthesis on ScanNet++.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>NICE-SLAM[2022b]</td><td>23.71</td><td>0.797</td><td>0.341</td></tr><tr><td>CO-SLAM[2023]</td><td>23.20</td><td>0.837</td><td>0.413</td></tr><tr><td>ESLAM[2023]</td><td>27.06</td><td>0.856</td><td>0.322</td></tr><tr><td>Point-SLAM[2023]</td><td>21.85</td><td>0.802</td><td>0.404</td></tr><tr><td>SplaTAM[2023]</td><td>27.77</td><td>0.864</td><td>0.233</td></tr><tr><td>Ours</td><td>27.27</td><td>0.872</td><td>0.295</td></tr></table>

Table 10: Comparison of tracking accuracy (unit: ????) with more SLAM systems.
<table><tr><td>Method</td><td>Replica</td><td>TUM-RGBD</td></tr><tr><td>ElasticFusion[2015]</td><td>1.13</td><td>2.07</td></tr><tr><td>BundleFusion[2017]</td><td>0.46</td><td>1.63</td></tr><tr><td>GO-SLAM[2023]</td><td>0.37</td><td>1.28</td></tr><tr><td>Ours</td><td>0.18</td><td>1.06</td></tr></table>

Table 11: Comparison of geometry accuracy on Replica with more SLAM systems.
<table><tr><td>Method</td><td>Acc.â</td><td>Acc. Ratioâ</td><td>Comp.â</td><td>Comp. Ratioâ</td></tr><tr><td>ElasticFusion[2015]</td><td>1.13</td><td>96.33</td><td>4.43</td><td>75.25</td></tr><tr><td>BundleFusion[2017]</td><td>0.77</td><td>99.88</td><td>5.35</td><td>76.69</td></tr><tr><td>GO-SLAM[2023]</td><td>2.51</td><td>76.93</td><td>5.11</td><td>65.10</td></tr><tr><td>Ours</td><td>0.75</td><td>98.87</td><td>2.81</td><td>82.76</td></tr></table>

<!-- image-->  
Figure 9: Comparison of novel view synthesis on Replica.

Table 12: Ablation study on sampled pixel number.  
Table 14: Ablation study on depth rendering.
<table><tr><td>Sample ratio</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>5%</td><td>39.01</td><td>0.965</td><td>0.072</td></tr><tr><td>10%</td><td>39.63</td><td>0.970</td><td>0.051</td></tr><tr><td>20%</td><td>39.31</td><td>0.972</td><td>0.044</td></tr></table>

Table 13: Ablation study on stable/unstable Gaussians.

<table><tr><td>Method</td><td>Acc.â</td><td>Acc. Ratioâ</td><td>Comp.â</td><td>Comp. Ratioâ</td><td>ATE (cm)â</td><td>Gaussian Number</td></tr><tr><td>Alpha-blending</td><td>2.48</td><td>70.54</td><td>3.32</td><td>75.01</td><td>1.24</td><td>468916</td></tr><tr><td>Ours</td><td>0.75</td><td>98.87</td><td>2.81</td><td>82.76</td><td>0.18</td><td>431692</td></tr></table>

<table><tr><td>Scene</td><td>Storeroom</td><td>Hotel room</td><td>Home</td></tr><tr><td> $s$ </td><td>9.8ms</td><td>8.4ms</td><td>8.9ms</td></tr><tr><td> $\boldsymbol { S _ { u n s t a b l e } } ,$  all pixels</td><td>7.4ms</td><td>6.5ms</td><td>6.4ms</td></tr><tr><td> $\boldsymbol { S _ { u n s t a b l e } } ,$  unstable pixels</td><td>5.2ms</td><td>4.7ms</td><td>4.3ms</td></tr></table>

Table 15: Ablation study on confidence count threshold.

## F MORE ABLATION STUDIES

<table><tr><td> $\delta _ { \eta }$ </td><td>PSNRâ</td><td>FPSâ</td></tr><tr><td>50</td><td>33.63</td><td>18.21</td></tr><tr><td>100</td><td>35.43</td><td>17.31</td></tr><tr><td>200</td><td>35.12</td><td>16.59</td></tr><tr><td>400</td><td>35.37</td><td>15.49</td></tr></table>

## F.1 Ablation study on sampled pixel number

## F.2 Ablation study on stable/unstable Gaussians

Here we evaluate the influence of the number of sampled pixels for adding Gaussians. We set the sampling ratio to 5%, 10%, and 20% for each frame to reconstruct Replica office0 and reported the image quality metrics. The results suggest that even if we sample a small number of images for reconstruction, the image quality is not significantly affected.

We test the impact of our proposed stable/unstable Gaussians on time performance. We report the optimization time per iteration, for optimizing all Gaussians using the whole image, optimizing only unstable Gaussians using the whole image, and optimizing only unstable Gaussians using only the pixels covered by them. As seen in Table 13, our strategy greatly improves the optimization speed.

## F.3 Ablation study on depth rendering

Our depth blending is tightly coupled with our Gaussian adding and state management, so in the paper we show that alpha blending yields much more Gaussians through the comparison with SplaTAM which uses alpha blending (987524 vs 7155880). Here for better ablation study, we first use our depth blending to determine the adding of opaque/transparent Gaussians as well as the states, and then replace our depth blending with alpha blending for optimization. The results are listed in Table 14. We can see with similar Gaussian numbers, our depth blending outperforms alpha blending in terms of geometry accuracy and tracking accuracy.

## F.4 Ablation study on confidence count threshold

The ablation study on confidence count threshold $\delta _ { \eta }$ on Replica is shown in Table 15. As $\delta _ { \eta }$ increases, the Gaussians will be in the unstable state for a longer time, resulting in a slower speed. On the other hand, if $\cdot _ { \delta _ { \eta } }$ is small, the Gaussians may not be fully optimized, yielding reduced rendering quality.

Table 16: Ablation study on backend pose optimization in terms of tracking accuracy .
<table><tr><td>Method</td><td>Replica</td><td>TUM-RGBD fr1_desk</td><td>TUM-RGBD fr2_xyz</td></tr><tr><td>ElasticFusion without backend</td><td>0.68</td><td>2.93</td><td>1.32</td></tr><tr><td>ElasticFusion</td><td>1.13</td><td>2.53</td><td>1.17</td></tr><tr><td>Ours without backend</td><td>0.22</td><td>5.39</td><td>1.63</td></tr><tr><td>Ours</td><td>0.18</td><td>1.66</td><td>0.38</td></tr></table>

Table 17: Ablation study on backend pose optimization in terms of geometry accuracy .
<table><tr><td>Method</td><td>Acc.â</td><td>Acc. Ratioâ</td><td>Comp.â</td><td>Comp. Ratioâ</td></tr><tr><td>ElasticFusion without backend</td><td>1.05</td><td>98.32</td><td>4.33</td><td>77.69</td></tr><tr><td>ElasticFusion</td><td>1.13</td><td>96.33</td><td>4.43</td><td>75.25</td></tr><tr><td>Ours without backend</td><td>0.95</td><td>97.57</td><td>2.74</td><td>83.17</td></tr><tr><td>Ours</td><td>0.75</td><td>98.87</td><td>2.81</td><td>82.76</td></tr></table>

Table 18: Ablation study on SH coefficients in terms of training view synthesis.
<table><tr><td>Color</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>RGB</td><td>33.90</td><td>0.951</td><td>0.113</td></tr><tr><td>SH</td><td>35.43</td><td>0.982</td><td>0.109</td></tr></table>

Table 19: Ablation study on SH coefficients in terms of novel view synthesis .
<table><tr><td>Color</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>RGB</td><td>25.67</td><td>0.855</td><td>0.304</td></tr><tr><td>SH</td><td>27.27</td><td>0.874</td><td>0.291</td></tr></table>

## F.5 Ablation study on backend pose optimization

Here we evaluate the effect of backend pose optimization. We test the tracking accuracy on Replica and TUM-RGBD, and report the results in Table 16. We also test the geometry accuracy on Replica and report the results in Table 17. On high-quality images on Replica, we achieve relatively high accuracy using only the frontend ICP. However, on low-quality images on TUM-RGBD, we rely more on the ORB backend, because the ICP may be performed on the Gaussians still under optimization.

## F.6 Ablation study on SH coefficients

Here we report the rendering quality using SHs and RGB colors. We report the results of training view synthesis on Replica in Table 18. We also report the results of novel view synthesis on ScanNet++ in Table 19. We can notice that using SH coefficients has better rendering quality.

Table 20: Ablation study on outlier pruning.
<table><tr><td>Frame ID</td><td>Ours</td><td>Ours without outlier pruning</td></tr><tr><td>1000#</td><td>151947</td><td>211390</td></tr><tr><td>2000#</td><td>268625</td><td>382736</td></tr><tr><td>3000#</td><td>507987</td><td>797242</td></tr><tr><td>4000#</td><td>675818</td><td>1073366</td></tr></table>

## F.7 Ablation study on outlier pruning

Here we evaluate the influence of our outlier pruning strategy. We report the number of Gaussians every 1000 frames in the Azure hotel room scene in Table 20 and the results show that the Gaussian number will increase significantly without outlier pruning.