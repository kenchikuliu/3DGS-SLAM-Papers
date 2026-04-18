# ROS-GS: Relightable Outdoor Scenes With Gaussian Splatting

Lianjun Liao, Chunhui Zhang, Tong Wu, Henglei Lv, Bailin Deng and Lin Gao

AbstractâImage data captured outdoors often exhibit unbounded scenes and unconstrained, varying lighting conditions, making it challenging to decompose them into geometry, reflectance, and illumination. Recent works have focused on achieving this decomposition using Neural Radiance Fields (NeRF) or the 3D Gaussian Splatting (3DGS) representation but remain hindered by two key limitations: the high computational overhead associated with neural networks of NeRF and the use of lowfrequency lighting representations, which often result in inefficient rendering and suboptimal relighting accuracy. We propose ROS-GS, a two-stage pipeline designed to efficiently reconstruct relightable outdoor scenes using the Gaussian Splatting representation. By leveraging monocular normal priors, ROS-GS first reconstructs the sceneâs geometry with the compact 2D Gaussian Splatting (2DGS) representation, providing an efficient and accurate geometric foundation. Building upon this reconstructed geometry, ROS-GS then decomposes the sceneâs texture and lighting through a hybrid lighting model. This model effectively represents typical outdoor lighting by employing a spherical Gaussian function to capture the directional, high-frequency components of sunlight, while learning a radiance transfer function via Spherical Harmonic coefficients to model the remaining lowfrequency skylight comprehensively. Both quantitative metrics and qualitative comparisons demonstrate that ROS-GS achieves state-of-the-art performance in relighting outdoor scenes and highlight its ability to deliver superior relighting accuracy and rendering efficiency.

Index TermsâGaussian splatting, inverse rendering, relighting.

## I. INTRODUCTION

O UTDOOR relighting plays a crucial role in a widerange of applications, including augmented reality, virtual range of applications, including augmented reality, virtual production, urban planning, and autonomous driving simulation. This capability enables the seamless integration of virtual objects with real-world environments and provides realistic visualization of infrastructure and objects under diverse and dynamic lighting scenarios. For instance, augmented reality applications rely on accurate relighting to maintain consistency between virtual elements and their surroundings, while urban planning and simulation tools require realistic lighting to assess the visual and functional impact of structures under varying conditions.

Despite its significance, outdoor relighting remains a challenging problem due to the inherent complexity of natural lighting and scene interactions. Outdoor environments experience continuously changing illumination conditions dictated by the sunâs position, weather variations, and atmospheric phenomena such as light scattering and absorption. These dynamic factors produce intricate illumination patterns, making it difficult to accurately replicate unknown lighting conditions from captured outdoor scenes. Furthermore, the vast scale of outdoor scenes, with their intricate details and wide-ranging features, poses a challenge in balancing efficiency and realism. Addressing these challenges requires techniques capable of handling multiple unknown lighting conditions and the largescale, computationally intensive nature of outdoor relighting, paving the way for advancements across multiple domains.

Current state-of-the-art methods [1]â[4] leverage neural implicit representations like Neural Radiance Fields (NeRF) [5], which encode scene geometry and appearance using neural networks. These methods have shown promising results in relighting by modeling light interactions and producing highquality visual outputs. While they can handle multiple unknown lighting conditions, they fall short in efficiency because of costly volume rendering in NeRF. Such inefficiency is particularly problematic for outdoor environments, where large-scale scenes and dynamic lighting demand both speed and flexibility.

Recently proposed Gaussian Splatting-based methods have improved efficiency largely. However, works like Relightable 3D Gaussians [6]â[8] are designed for single object relighting and can only handle a single unknown lighting condition. ReCap [9] can model multiple lighting conditions but only focus on the synthetic single object and does not account for self-shadowing effects. There are other works that aim to relight scenes, however, among them, GS-IR [10] is constrained to single lighting condition, and methods like GS3 [11], [12] are restricted to a known one-light-at-a-time (OLAT) setting. Apart from above, LumiGauss [13] aims to relight outdoor scenes with multiple unknown lighting, but its use of a low-frequency lighting representation solely based on Spherical Harmonics prevents it from accurately modeling high-frequency illumination effects, such as cast shadows and high-contrast light-dark boundaries, which are essential for outdoor realistic relighting. These limitations highlight the need for more advanced methods that integrate computational and storage efficiency while effectively handling multiple unknown

Training image  
<!-- image-->

Reconstruction  
<!-- image-->

Surface normal  
<!-- image-->  
Relighting

Diffuse albedo  
<!-- image-->  
Novel view

Figure 1: Given multi-view images with unconstrained lighting conditions, ROS-GS optimizes its Gaussian Splatting representation and decomposes the geometry, texture, and lighting components of the outdoor scene, which enables not only faithful novel view synthesis but also realistic and accurate relighting.

illumination conditions in outdoor captures. Additionally, such techniques must enable the accurate representation of finegrained lighting interactions in outdoor scenes.

To address these challenges, we propose a novel framework for outdoor relighting that leverages the fast rendering speed of 2D Gaussian Splatting and the realism of physically-based rendering to achieve photorealistic lighting manipulation for outdoor scenes. Our approach comprises two stages inspired by previous inverse rendering methods [2], [14]. In the first stage, we incorporate the compact 2DGS representation [15] and monocular normal estimation prior [16] to reconstruct outdoor scenesâ geometry from a set of unconstrained images. Then in the second stage, based on the reconstructed geometry, we introduce physically-based rendering into the 2DGSâs rendering process, where the Spherical Harmonics attribute of Gaussians is replaced with view-independent texture parameters. In addition, inspired by SOL-NeRF [2], a hybrid lighting representation is adopted for modeling high-frequency sun light and low-frequency sky light. By progressively optimizing texture and lighting, ROS-GS can decompose the geometry, texture, and lighting from multi-view images, making outdoor relighting and real-time rendering plausible. The contribution of our method can be summarized as follows:

â¢ ROS-GS leverages the 2DGS representation and the monocular normal estimation into the inverse rendering pipeline, producing more faithful geometry and texture estimation.

â¢ By incorporating the hybrid lighting representation with the Gaussian Splatting representation, ROS-GS enables realistic relighting and real-time dynamic shadow synthesis.

â¢ Both qualitative and quantitative results show that ROS-GS outperforms baseline methods in outdoor scene decomposition and relighting tasks.

## II. RELATED WORKS

## A. Novel View Synthesis and Geometry Reconstruction

Novel view synthesis has long been a core topic in computer graphics and vision. Early works used light field interpolation [17], [18], while later methods employed depth warping [19], [20] and Multi-Plane Images (MPIs) [21]â[24], though these were limited to small camera movements. More advanced representations, such as meshes, voxels, implicit representation [5], [25]â[28] and neural rendering [29], [30], enabled greater flexibility but often suffered from rendering inefficiencies. Recently, 3DGS [31] emerged, efficiently modeling scenes as Gaussian ellipsoids for fast and high-quality rendering.

Traditional methods use hand-crafted features to match pixels across views, reconstructing surface from extracted point clouds with Poisson surface reconstruction [32], [33] or Marching Cubes [34]. With the advent of Neural Radiance Fields (NeRF) [5] enabling photorealistic novel view synthesis, recent works have leveraged NeRF for geometry reconstruction. NeuS [35] introduced an unbiased, occlusion-aware formulation connecting signed distance fields to volumetric rendering, while UNISURF [36] linked occupancy networks with NeRF. Building on NeuS, SparseNeuS [37] enabled sparse-view reconstruction, and MonoSDF [38] improved geometry quality using monocular normal estimation. With 3DGS improving the rendering speed, SuGaR [39] introduces a self-regularization term for better surface alignment. 2DGS [15] and GS-Surfel [40] introduces more compact 2D Gaussian representation for better surface reconstruction.

## B. Relighting

Early relighting works usually capture scenes under varying illumination [41], [42] to estimate material properties and environmental lighting for relighting purposes. With neural rendering enabling realistic novel view synthesis, works start to recover geometry, texture, and lighting from casually captured multi-view images. The pioneering work PhySG [43] adopts signed distance function as geometry representation and uses spherical Gaussians (SG) to approximate lighting. NDR [44] and NDRMC [45] incorporate the deformable tetrahedra representation [46] for efficient geometry extraction and rendering. NeRFactor [14] further introduces prior knowledge from existing material datasets for more accurate texture and lighting decomposition. For more complex materials, implicit lighting representation [47]â[50] is proposed to better capture high-frequency lighting information. Regarding the outdoor scene setting, NeRF-OSR [1] utilizes a Spherical Harmonics (SH) function to represent lighting information to approximate visibility with a neural network. SOL-NeRF [2] instead introduces a hybrid lighting formulation, where a spherical Gaussian (SG) function models the sun light and a Spherical Harmonics (SH) function represents the sky light, which makes creating cast shadow possible. To improve the shadow calculation efficiency, NeuSky [3] proposes a spherical directional distance function to approximate the visibility information among the scene.

More recently, 3DGS-based relighting methods have emerged. Relightable 3D Gaussians [6] incorporate a raytracer to 3DGS in order to depict shadow effects. While this method can achieve continuously moving shadow effects, it fails to produce accurate self-shadows with sharp edges. Similarly, IRGS [8] implements shadow effects using a 2D Gaussian ray-tracer. However, it focuses solely on object relighting rather than whole-scene relighting and can only handle a single unknown lighting condition. $\mathrm { G S ^ { 3 } }$ [11] focused on single objects under known OLAT lighting condition. Recap [9] aimed to improve inverse rendering using multiple lighting conditions but was confined to synthetic objects, neglecting secondary effects like shadows. Its use of a learnable image-based cubemap for lighting is memory-intensive and computationally costly, proving inefficient for the numerous lighting conditions typical in outdoor captures. LumiGauss [13] integrates Precomputed Radiance Transfer (PRT) with 2DGS for outdoor scenes, but its omnidirectional lighting assumption limits high-frequency effects such as sharp shadows, and PRT overfitting often results in baked-in artifacts. Notably, the concurrent work GaRe [51] also tackles unconstrained lighting input, decomposes illumination into sun, sky and indirect component with small MLP mappings, and generates dynamic shadows via a ray-trace-based Gaussian visibility query method. In contrast, our method adopts a more compact lighting representation and computes shadows using a meshbased intersection detection strategy.

## III. METHOD

The overall pipeline of our method is illustrated in Fig. 2. Given multi-view images of outdoor scenes captured under unconstrained lighting conditions, ROS-GS decouples geometry reconstruction from texture-lighting decomposition through a two-stage design. This decomposition subsequently enables realistic relighting of the scenes using the Gaussian Splatting representation [31]. The first stage focuses on reconstructing the scene geometry utilizing the compact 2DGS representation [15] combined with an appearance transformation module, guided by monocular normal estimation priors. Afterward, the second stage introduces a hybrid sun-sky lighting model to effectively disentangle texture and illumination, enabling high-quality relighting with faithful shadow effects.

## A. Preliminaries

1) 2D Gaussian Splatting: ROS-GS utilizes 2D Gaussian Splatting [15], a compact geometric representation capable of reconstructing scene surfaces with smooth normals from multi-view images. A 2D Gaussian surfel is defined by:

$$
\mathcal { G } ( \boldsymbol { p } ) = \exp { ( - \frac { 1 } { 2 } ( \boldsymbol { p } - \boldsymbol { p } _ { k } ) ^ { \top } \Sigma ^ { - 1 } ( \boldsymbol { p } - \boldsymbol { p } _ { k } ) ) } ,\tag{1}
$$

where $p _ { k }$ is the center of a Gaussian surfel, and $\Sigma = R S S ^ { \top } R ^ { \top }$ is its covariance matrix, parameterized by a scaling matrix S and a rotation matrix R. In the rendering process, Gaussian surfels in world space are first transformed into camera coordinates. They are then projected onto the image plane using an explicit ray-splat intersection technique, which facilitates stable optimization. Then, a volumetric alpha blending process is employed to compute the color of a pixel by integrating contributions from front to back:

$$
c ( \mathbf { x } ) = \sum _ { i = 1 } ^ { } c _ { i } \alpha _ { i } \hat { \mathcal { G } } _ { i } ( \mathbf { u } ( \mathbf { x } ) ) \prod _ { j = 1 } ^ { i - 1 } \big ( 1 - \alpha _ { j } \hat { \mathcal { G } } _ { j } ( \mathbf { u } ( \mathbf { x } ) ) \big ) ,\tag{2}
$$

where $c _ { i }$ and $\alpha _ { i }$ are the view-dependent appearance and the alpha value of the i-th Gaussian. $\hat { \mathcal { G } } _ { k } ( \mathbf { u } ( \mathbf { x } ) ) ~ ( k { = } i , j )$ is the projected 2D Gaussian function, determined by its projected covariance matrix $\Sigma ^ { \prime }$ evaluated at the intersection point u(x) between a camera ray and the Gaussian. $c ( \mathbf { x } )$ is the alphablended color along the ray x.

2) Rendering Equation: To achieve relighting under novel illumination, it is crucial to model texture and lighting independently. ROS-GS incorporates physically-based rendering principles within the Gaussian Splatting framework, guided by the rendering equation [52]:

$$
L ( x , \omega _ { o } ) = \int _ { \Omega } f _ { r } ( x , \omega _ { o } , \omega _ { i } ) L _ { i n } ( x , \omega _ { i } ) ( \omega _ { i } \cdot n ) \mathrm { d } \omega _ { i } .\tag{3}
$$

Here, x denotes a point on a surface with normal n. $L _ { i n } ( x , \omega _ { i } )$ is the incident lighting at point x from direction $\omega _ { i }$ . The function $f _ { r } ( x , \omega _ { o } , \omega _ { i } )$ represents the BRDF at point x, which depends on both the light direction $\omega _ { i }$ and the viewing direction $\omega _ { o } .$ The outgoing radiance $L ( x , \omega _ { o } )$ is obtained by integrating these contributions over the hemisphere $\Omega = \{ \omega _ { i } : \omega _ { i } \cdot n > 0 \}$

3) Precomputed Radiance Transfer: Sloan et al. [53] propose that the rendering equation can be viewed as an integral over the product of the incident illumination $L _ { i n } ( x , \omega _ { i } )$ and the radiance transfer function $T ( x , \omega _ { i } , \omega _ { o } ) = f _ { r } ( x , \omega _ { o } , \omega _ { i } ) ( \omega _ { i } \cdot n )$ Using Spherical Harmonics (SH) basis functions approximates low-frequency illumination efficiently, as the integral reduces to a simple dot product of SH coefficients due to their orthonormality. For diffuse surfaces, the radiance transfer simplifies to a vector $T ( x )$ , yielding the outgoing light L(x) as:

<!-- image-->  
Figure 2: Pipeline of our method. Given multi-view images captured under unconstrained lighting, our ROS-GS introduces a two-stage relighting pipeline based on Gaussian splatting. (a) In the first stage, we reconstruct scene geometry using the 2DGS representation with an appearance transformation module, guided by monocular normal priors. (b) In the second stage, we employ a hybrid lighting representation to achieve textureâlighting decomposition. (c) After training, our model supports both novel view synthesis and realistic relighting of outdoor scenes.

$$
\begin{array} { r } { L ( x ) = \sum _ { k = 1 } ^ { ( n + 1 ) ^ { 2 } } T ( x ) ^ { k } L _ { i n } ^ { k } , } \end{array}\tag{4}
$$

where $T ( x ) ^ { k }$ and $L _ { i n } ^ { k }$ are the k-th SH coefficients of the transfer function and the incident lighting, respectively.

## B. Geometry Initialization

ROS-GS takes as input N multi-view images of an outdoor scene under unconstrained lighting conditions. The first stage of our pipeline focuses on geometry initialization. Reconstructing accurate geometry from such inputs is challenging due to complex appearance variations across views and weakly textured regions, often further exacerbated by hard shadows.

To address these challenges, we integrate an appearance transformation module, inspired by NeRF-in-the-Wild [54] and WildGaussians [55], into the standard 2DGS training. In this stage, each Gaussian is defined as:

$$
\mathcal { G } _ { I } = \{ p _ { k } , \sigma _ { k } , R _ { k } , o _ { k } , \rho _ { k } , \Lambda _ { k } \} , k \in K ,\tag{5}
$$

where $p _ { k } , \sigma _ { k } , R _ { k }$ , and $o _ { k }$ denote the position, scaling, rotation, and opacity of the k-th Gaussian, respectively. ${ \rho } _ { k } \in [ 0 , 1 ] ^ { 3 }$ denotes the base color of the k-th Gaussian, which provides a simpler and more compact representation than using Spherical Harmonics (SH). To model appearance variability, we introduce a per-Gaussian embedding $\Lambda _ { k }$ that captures local appearance variations, and a set of per-image embeddings $\{ \Xi _ { j } \} _ { j = 1 } ^ { N }$ that accounts for lighting changes across different views. The appearance transformation module is an MLP $f _ { a p p }$ that predicts the parameters of an affine transformation:

$$
( \gamma , \beta ) = f _ { a p p } ( \rho _ { k } , \Lambda _ { k } , \Xi _ { j } ) ,\tag{6}
$$

which adjusts the base color $\rho _ { k }$ to produce the transformed appearance color $\tilde { \rho } _ { k }$ :

$$
\tilde { \rho } _ { k } = \gamma \cdot \rho _ { k } + \beta .\tag{7}
$$

The resulting appearance color $\tilde { \rho } _ { k }$ is then rasterized using Eq. (2).

Although this formulation improves robustness, weakly textured and planar regions still hinder the optimization of smooth surface normals. To further enhance geometry quality, we incorporate a monocular normal prior. Since our inputs are captured under unconstrained conditions, moving foreground objects often cause multi-view inconsistencies. To ensure stable supervision, we identify these dynamic regions and exclude them using semantic masks generated by an off-the-shelf segmentation method [56]. In addition, we apply the normal prior only to non-edge regions. Specifically, we define non-edge regions as image areas that do not correspond to geometric discontinuities. To extract them, we use the classical Canny edge detector with thresholds set to [100, 200], followed by dilation with a 3Ã3 kernel. We focus on non-edge regions because planar areas benefit most from normal constraints, whereas applying the prior across the entire image leads to over-smoothed normals.

Finally, to prepare for shadow computation in the second stage, we extract a mesh from the reconstructed geometry. We first render depth maps from the training views, filter out sky, foliage, and other dynamic occluders using semantic masks, and then fuse the depth maps via the truncated signed distance function (TSDF) integration method to obtain clean surface meshes.

## C. Texture-Lighting Decomposition

In the second stage, building upon the geometry initialized in the first stage, ROS-GS focuses on decomposing the sceneâs texture and illumination. We employ a hybrid lighting representation, inspired by SOL-NeRF [2], to model the complex lighting information present in outdoor scenes. Specifically, we employ a single spherical Gaussian (SG) function to capture the directional, high-frequency components of sunlight. For the modeling of the skylight, SOL-NeRF employs a single set of SH functions combined with an MLP-predicted ambient occlusion term. However, this introduces significant computational overhead during inference. In contrast, our approach leverages Precomputed Radiance Transfer (PRT) to model skylight in a more comprehensive and spatially adaptive manner. By decoupling lighting transport from geometry and precomputing visibility interactions, our method achieves more realistic and detailed illumination without relying on expensive per-pixel MLP evaluations. We define globally shared SH coefficients $l _ { c } \in \mathbb { R } ^ { 3 \times ( n + 1 ) ^ { 2 } }$ to represent the low-frequency ambient illumination, where $c = 1 , \ldots , N$ indexing the input views. For a balance between efficiency and quality, we adopt second-order SH $( n = 2 ) \ [ 5 7 ]$ . Meanwhile, the per-Gaussian SH coefficients $t _ { k } \in \mathbb { R } ^ { 1 \times ( n + 1 ) ^ { 2 } }$ encode the radiance transfer function. Thus, each Gaussian in this stage is defined as:

$$
\begin{array} { r } { \mathcal { G } _ { I I } = \{ p _ { k } , \sigma _ { k } , R _ { k } , o _ { k } , \rho _ { k } , t _ { k } \} , k \in K , } \end{array}\tag{8}
$$

where $p _ { k } , \sigma _ { k } , R _ { k }$ and $o _ { k } ,$ as geometry attributes of Gaussians, are all inherited from the first-stage reconstruction. For $\rho _ { k } \in$ $[ 0 , 1 ] ^ { 3 }$ , we reuse the same parameter vector from the first stage, but reinterpret it as the albedo of the k-th Gaussian in the second stage.

Assuming the sun and sky are infinitely distant, incoming light rays can be treated as parallel, making the incident illumination primarily dependent on surface orientation rather than spatial position. Under this assumption, accurate and smooth surface normals are crucial for faithful estimation of lighting. While the final accumulated normals produced by 2D Gaussian Splatting are generally smooth, the local normals of individual Gaussians contributing to a ray can exhibit significant variance. To ensure more consistent multi-view geometry and shading, we adopt a deferred shading technique [58], [59], performing shading calculations in screen space at each pixel using the final accumulated normals rather than on a per-Gaussian basis. Besides, we primarily focus on diffuse surfaces and replace the BRDF $f _ { r } ( x , \omega _ { o } , \omega _ { i } )$ with a simplified diffuse model $\textstyle f _ { d } = { \frac { \rho } { \pi } }$ where $\rho$ is the diffuse albedo, which also makes the outgoing light independent of the viewing direction. Thus, for each pixel x, the rendered color is computed as:

$$
C ( x ) = \frac { \rho } { \pi } ( I _ { s u n } + I _ { a m b } ) .\tag{9}
$$

Here, $\rho$ is an albedo map generated by splatting the albedo attribute $\rho _ { k }$ of Gaussians, $I _ { s u n }$ is the irradiance contribution from the sun represented by an SG, and $I _ { a m b }$ is the irradiance from the ambient skylight represented by SH.

A spherical Gaussian is defined as $S G _ { \mu , \lambda , \xi } ( \nu ) = \mu \mathrm { e } ^ { \lambda ( \nu \cdot \xi - 1 ) }$ , where $\mu \in \mathbb { R } _ { + } ^ { 3 }$ is the lobe amplitude, $\lambda \in \mathbb { R } _ { + }$ controls the lobe sharpness, $\xi \in \mathbb { S } ^ { 2 }$ is a unit vector representing the sun direction, and Î½ is the input direction. The irradiance of the sunlight component is then computed as:

$$
I _ { s u n } = \int _ { \Omega } V _ { s u n } ( x , \omega _ { i } ) S G _ { \mu , \lambda , \xi } ( \omega _ { i } ) ( \omega _ { i } \cdot n ) \mathrm { d } \omega _ { i } ,\tag{10}
$$

where, $V _ { s u n } ( x , \omega _ { i } ) \in \{ 0 , 1 \}$ denotes the visibility of the surface point X with respect to the incident direction $\omega _ { i } .$ , X is the surface point corresponding to the pixel $x ,$ and n denotes the normal map. Visibility is calculated for each pixel in screen space in a deferred manner. We perform ray tracing against the mesh extracted in the first stage, accelerated by a Bounding Volume Hierarchy (BVH) tree. $V = 1$ if the light from direction $\omega _ { i }$ reaches the point x, and $V = 0$ if the ray intersects the mesh.

For the ambient light irradiance $I _ { a m b }$ , we first compute the ambient irradiance at each Gaussian by the dot product between the global and local SH coefficients, and then splat these per-Gaussian ambient contributions to produce the ambient irradiance map:

$$
I _ { a m b } = \mathcal { S } ( \sum _ { j = 1 } ^ { ( n + 1 ) ^ { 2 } } l _ { c } ^ { j } \cdot t _ { k } ^ { j } ) ,\tag{11}
$$

where $\boldsymbol { \mathcal { S } } ( \cdot )$ denotes the 2DGS rasterization process for splatting Gaussian attributes.

## D. Training losses

Our model utilizes the 2DGS rasterization process for imagebased scene reconstruction. The training is divided into two stages, each with a specific loss function. The loss function for the first stage, focused on geometry reconstruction, is defined as:

$$
\mathcal { L } _ { 1 s t } = \lambda _ { 1 } \mathcal { L } _ { r e n d e r } + \lambda _ { 2 } \mathcal { L } _ { r e g } + \lambda _ { 3 } \mathcal { L } _ { n p } + \lambda _ { 4 } \mathcal { L } _ { m a s k } .\tag{12}
$$

Here, $\mathcal { L } _ { \boldsymbol { r e n d e r } }$ denotes a rendering loss that measures the difference between the rendered images and the ground truth images, and is composed of an $L _ { 1 }$ term and a D-SSIM [60] term. $\mathcal { L } _ { r e g }$ is a term introduced in 2DGS for normal consistency and distortion regularization and retained in our pipeline. $\mathcal { L } _ { n p }$ and $\mathcal { L } _ { m a s k }$ are geometry prior losses for normal prior and semantic mask, respectively.

For $\mathcal { L } _ { n p } .$ , we obtain a prior normal map $n _ { p }$ from an off-theshelf monocular normal estimator [16], and compare it against both the alpha-blended rendered normal $\hat { n } _ { r }$ and the normal $\hat { n } _ { s }$ derived from the depth map gradient via:

$$
\mathcal { L } _ { n p } = ( \sum \left( 1 - n _ { p } ^ { \top } \hat { n } _ { r } \right) + \sum \left( 1 - n _ { p } ^ { \top } \hat { n } _ { s } \right) ) .\tag{13}
$$

This ensures consistency between the geometries defined by depth and normals.

For $\mathcal { L } _ { m a s k }$ , we generate semantic masks for the input images using MMSegmentation [56], and compute a binary cross entropy loss between the sky regions in the masks and the rendered alpha channel. Additionally, we use the semantic masks to filter out dynamic object regions when computing the rendering loss and the normal prior loss, thereby focusing the training on the static primary components of the scene.

In the second stage, we fix the geometry reconstructed in the first stage and focus on texture and lighting decomposition. The loss function is defined as:

$$
\mathcal { L } _ { 2 n d } = \lambda _ { 1 } \mathcal { L } _ { r e n d e r } + \lambda _ { 5 } \mathcal { L } _ { s p } + \lambda _ { 6 } \mathcal { L } _ { a m b } ,\tag{14}
$$

where we employ two loss terms $\mathcal { L } _ { s p }$ and $\mathcal { L } _ { a m b }$ for priors on the sunlight color and ambient light, respectively.

The sunlight color prior $\mathcal { L } _ { s p }$ constrains the intensity of the SG lobe within the typical color range of sunlight [2]. This prior is formulated as a piecewise polynomial function $f _ { s u n }$ that fits the result values of the modified Nishita model [61] for training efficiency, and the term is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { s p } = \| f _ { s u n } ( \theta ) - \mu \| _ { 1 } , } \end{array}\tag{15}
$$

where $\mu$ denotes the learned SG intensity and Î¸ is a simulated solar elevation angle parameter computed from the sun direction $\xi$ in our SG lighting model.

The ambient light loss $\mathcal { L } _ { a m b }$ encourages the ambient SH component to capture low-frequency illumination and prevents overfitting hard-edged baked-in shadows, and is defined as the total variation of the rendered ambient irradiance map $I _ { a m b } { : }$

$$
\begin{array} { c l l } { \mathcal { L } _ { a m b } = \displaystyle \sum _ { x } \Big ( \| I _ { a m b } ( x + { \bf e } _ { x } ) - I _ { a m b } ( x ) \| _ { 1 } } \\ { \displaystyle + \| I _ { a m b } ( x + { \bf e } _ { y } ) - I _ { a m b } ( x ) \| _ { 1 } \Big ) , } \end{array}\tag{16}
$$

where $\mathbf { e } _ { x } , \mathbf { e } _ { y }$ denote unit pixel shifts along the horizontal and vertical directions.

## IV. EXPERIMENTS AND RESULTS

## A. Implementation Details

For evaluation, we conduct experiments and ablation studies on two datasets: a synthetic dataset from SOL-NeRF [2] and a real dataset from NeRF-OSR [1]. Since the NeRF-OSR dataset does not provide ground-truth texture or normal maps, we report quantitative decomposition results only on the three synthetic scenes from the SOL-NeRF dataset. For quantitative relighting evaluation, we follow prior works and report results on the three scenes of the NeRF-OSR dataset. We adopt commonly used metrics for decomposition and relighting evaluation, including PSNR, SSIM [62], and Mean Squared Error (MSE). For geometry reconstruction evaluation, we use the Mean Absolute Error (MAE) between the decomposed normal maps and the ground truth normal maps.

We run all experiments using a single NVIDIA RTX 3090 GPU with 24GB VRAM. We set $\lambda _ { 1 } = 1 . 0$ for both $L _ { 1 }$ loss and D-SSIM loss for rendered images and set $\lambda _ { 2 } ~ = ~ 0 . 0 5$ for both normal consistency loss and distortion loss. We set $\lambda _ { 3 } = 0 . 1$ for the normal prior loss $\mathcal { L } _ { n p }$ and set $\lambda _ { 4 } = 0 .$ 1 for the semantic mask loss $\mathcal { L } _ { m a s k }$ . In addition, we set $\lambda _ { 5 } = 1 0 . 0$ decaying to 0.01 over 50k iterations, for the sunlight color prior loss. This ensures that $\mathcal { L } _ { s p }$ acts as a strong constraint in the early phase, stabilizing sunlight color learning. Finally, we set $\lambda _ { 6 } = 1 . 0$ for the loss $\mathcal { L } _ { a m b }$ on the ambient irradiance map. For appearance transformation, we use embeddings of size 24 for the per-Gaussian embedding $\Lambda _ { k }$ and 32 for the per-image embedding $\Xi _ { j }$ . For the appearance MLP $f _ { a p p } ,$ we use one hidden layer of size 128.

Table I: Quantitative comparisons of reconstruction and decomposition results using SSIM, PSNR, and MSE metrics for rendered image and albedo on the synthetic dataset with baseline methods. For the geometry, we compare MAE (Mean Absolute Error) between the rendered and ground truth normals.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Rendered</td><td colspan="3">Albedo</td><td rowspan="2">Normal</td></tr><tr><td>PSNR</td><td></td><td></td><td></td><td></td><td>âSSIM âMSE âPSNR âSSIM âMSE âMAE(Â°) â</td></tr><tr><td>NeRF-OSR 22.75</td><td></td><td>0.808</td><td>0.009</td><td>20.16</td><td>0.825</td><td>0.014</td><td>26.21</td></tr><tr><td>SOL-NeRF 23.44</td><td></td><td>0.863</td><td>0.006</td><td>24.89</td><td>0.827</td><td>0.006</td><td>16.33</td></tr><tr><td>ReCap</td><td>26.34</td><td>0.919</td><td>0.004</td><td>17.57</td><td>0.774</td><td>0.028</td><td>30.55</td></tr><tr><td>LumiGauss 24.97</td><td></td><td>0.897</td><td>0.004</td><td>24.30</td><td>0.884</td><td>0.005</td><td>33.88</td></tr><tr><td>Ours</td><td>26.75</td><td>0.933</td><td>0.003</td><td>26.80</td><td>0.869</td><td>0.003</td><td>13.97</td></tr></table>

We employ a progressive training strategy to effectively decompose scene geometry, appearance, and illumination. In the first stage, optimization is restricted to scene geometry. During the second stage, all geometry-related Gaussian attributes are frozen. Since the appearance transformation module in the first stage tends to overfit the scene appearance, the model does not inherit a well-defined albedo at the beginning of the second stage. Thus, during the initial 10,000 iterations of the second stage, shading is omitted to allow the model to learn an initial, refined albedo. For the following 40,000 iterations, all Gaussian attributes, including the newly learned albedo, are frozen to focus exclusively on optimizing the lighting parameters. Finally, the last 50,000 iterations involve the joint optimization of non-geometric Gaussian attributes and the lighting parameters, enabling all components to be fine-tuned together.

## B. Decomposition Results

ROS-GS performs decomposition of geometry, texture, and lighting from multi-view images. We compare the decomposition results of our approach with four baseline methods: LumiGauss [13], ReCap [9], NeRF-OSR [1], and SOL-NeRF [2]. We present qualitative results on three real scenes from the NeRF-OSR dataset in Fig.3 and on two synthetic scenes from SOL-NeRF in Fig.4. We further compare with NeuSky [3] on a real scene in Fig.5. We report quantitative results on synthetic scenes from SOL-NeRF in Table I. For geometry recovery, our approach produces normal maps with clearer structural details and smoother surfaces compared to the baseline methods. In terms of appearance decomposition, our decomposed albedo maps not only exhibit more natural colors but also contain fewer shadow artifacts, unlike those produced by LumiGauss and ReCap, which often contain bakedin shadows. Moreover, our approach demonstrates superior illumination expressiveness, as evidenced by the enhanced quality of reconstruction and the more accurately decoupled shadow maps. Compared to baseline methods, our model can handle high-frequency illumination effects, such as sharper cast shadows and high-contrast light-dark boundaries. Overall, our method outperforms these approaches in novel view synthesis and excels in both albedo and normal decomposition, leading to improved scene relighting results, as discussed in later sections.

<!-- image-->  
(a) Ground Truth  
(b) Reconstruction  
(c) Albedo  
(d) Shadow  
(e) Normal  
Figure 3: Qualitative comparisons of decomposition results on baseline methods and our method. For each scene, we show decomposed components (normal, albedo, and shadow) and the reconstructed image.

<!-- image-->  
(a) Ground Truth  
(b) Reconstruction  
(c) Albedo  
(d) Shadow  
(e) Normal  
Figure 4: Qualitative comparison of decomposition results on synthetic scenes. We present reconstruction, albedo, shadow and normal results of our method compared with SOL-NeRF [2], ReCap [9] and LumiGauss [13] on two synthetic scenes.

## C. Relighting Results

We present qualitative relighting results on three real scenes from the NeRF-OSR dataset in Fig. 6. For each scene, we show the input image, two relighting results from the same viewpoint, and their corresponding shadow maps. We report quantitative results in Table II. The shadow maps generated by LumiGauss [13], which models lighting using low-frequency Spherical Harmonics (SH), fail to capture illumination changes and yield weak cast shadows. NeRF-OSR [1], which also adopts SH-based lighting, produces similarly limited shadow effects. By contrast, ReCap [9] does not account for shadows, which results in baked-in shadows within the albedo map and consequently produces incorrect shadow effects. Moreover, its albedo maps also suffer from unnatural color distortions that degrade relighting quality. Additionally, the inaccurate and noisy normal maps predicted by LumiGauss result in uneven surface shading and further reduce realism. In comparison, our pipeline benefits from monocular geometry estimation, which improves geometric accuracy and supports more reliable decomposition and relighting. Moreover, our hybrid lighting formulation produces sharp and realistic shadow effects under varying illumination. We present a qualitative comparison of shadow movement under changing sunlight directions in Fig. 7.

<!-- image-->

(a) Ground Truth  
<!-- image-->  
(b) Reconstruction

<!-- image-->  
(c) Albedo

<!-- image-->  
(d) Shadow

<!-- image-->  
(e) Normal  
Figure 5: Additional qualitative comparison of decomposition results. We present reconstruction, albedo, shadow and normal results of our method compared with SOL-NeRF [2], NeuSky [3], ReCap [9] and LumiGauss [13] on a real scene.

Table II: Quantitative comparison of relighting results using PSNR, SSIM, MAE and MSE metrics on real scenes. u/s denotes using up-sampled images for evaluation. We use environment maps provided by the NeRF-OSR dataset. Results are averaged over five novel views in the test set.
<table><tr><td rowspan="2">Methods</td><td colspan="4">Ludwigskirche</td><td colspan="4">Staatstheater</td><td colspan="4">Landwehrplatz</td></tr><tr><td>PSNR â</td><td>MSE â</td><td>MAE â</td><td>SSIM â</td><td>PSNR â</td><td>MSE â</td><td>MAE â</td><td>SSIM â</td><td>PSNR â</td><td>MSE â</td><td>MAE â</td><td>SSIM â</td></tr><tr><td>Yu et al.u/s [63]</td><td>17.87</td><td>0.017</td><td>0.097</td><td>0.378</td><td>15.28</td><td>0.032</td><td>0.138</td><td>0.385</td><td>15.17</td><td>0.033</td><td>0.133</td><td>0.376</td></tr><tr><td>Philip et al. [64]</td><td>16.63</td><td>0.023</td><td>0.113</td><td>0.367</td><td>12.34</td><td>0.065</td><td>0.200</td><td>0.272</td><td>12.28</td><td>0.062</td><td>0.179</td><td>0.319</td></tr><tr><td>NeRF-OSR [1]</td><td>18.72</td><td>0.014</td><td>0.090</td><td>0.468</td><td>15.43</td><td>0.029</td><td>0.133</td><td>0.517</td><td>16.65</td><td>0.024</td><td>0.114</td><td>0.501</td></tr><tr><td>FEGR [65]</td><td>21.53</td><td>0.007</td><td>-</td><td>-</td><td>17.00</td><td>0.023</td><td>-</td><td></td><td>17.57</td><td>0.018</td><td>-</td><td>-</td></tr><tr><td>SOL-NeRF [2]</td><td>21.23</td><td>0.008</td><td>-</td><td>0.749</td><td>18.18</td><td>0.019</td><td>-</td><td>0.680</td><td>17.58</td><td>0.028</td><td>-</td><td>0.618</td></tr><tr><td>SR-TensoRF [4]</td><td>17.30</td><td>0.021</td><td>0.096</td><td>0.542</td><td>15.43</td><td>0.030</td><td>0.111</td><td>0.632</td><td>16.74</td><td>0.024</td><td>0.093</td><td>0.653</td></tr><tr><td>NeuSky [3]</td><td>22.50</td><td>0.005</td><td>-</td><td>-</td><td>16.66</td><td>0.023</td><td>-</td><td>-</td><td>18.31</td><td>0.016</td><td>-</td><td>-</td></tr><tr><td>LumiGauss [13]</td><td>19.59</td><td>0.012</td><td>0.085</td><td>0.700</td><td>17.02</td><td>0.021</td><td>0.107</td><td>0.729</td><td>18.01</td><td>0.017</td><td>0.096</td><td>0.778</td></tr><tr><td>ReCap [9]</td><td>24.11</td><td>0.004</td><td>0.041</td><td>0.808</td><td>22.70</td><td>0.006</td><td>0.051</td><td>0.819</td><td>21.55</td><td>0.008</td><td>0.051</td><td>0.837</td></tr><tr><td>Ours</td><td>25.33</td><td>0.003</td><td>0.027</td><td>0.833</td><td>23.62</td><td>0.005</td><td>0.036</td><td>0.833</td><td>21.87</td><td>0.007</td><td>0.043</td><td>0.847</td></tr></table>

## D. Efficiency

Table III compares the training time and FPS of our method with other approaches on the NeRF-OSR dataset. Our training time is comparable to ReCap, and significantly shorter than NeRF-OSR. We also achieve real-time rendering performance that is faster than LumiGauss and significantly outperforms NeRF-OSR.

## E. Ablation Study

In this subsection, we conduct ablation studies to evaluate the impact of several important design choices in our pipeline.

1) Appearance Transformation Module: As described in Sec. III-B, we integrate an appearance transformation module into the first stage of our pipeline to handle appearance variations across multi-view inputs. Its impact is illustrated in the ablation study in Fig. 8. Without its compensation for appearance variations, the estimated normal maps become fuzzy and distorted, thereby hindering the subsequent separation of texture and illumination. The quantitative results in Table IV further confirm its effectiveness, demonstrating improved robustness under varying appearance conditions.

<!-- image-->

<!-- image-->

<!-- image-->  
(a) Input  
(b) Relight #1  
(c) Shadow #1  
(d) Relight #2  
(e) Shadow #2  
Figure 6: Qualitative comparisons of relighting results on baseline methods and our method. For each input view, we relight it with two different lighting conditions and show rendered images and shadow maps.

<!-- image-->  
GT Input

Figure 7: Qualitative comparison of shadow movement effects under varying sunlight directions between our method and LumiGauss [13]. For LumiGauss, relighting results are generated by rotating its SH lighting representation.  
<!-- image-->  
Figure 8: Qualitative comparison of decomposed normal results with two variants. w/o AT refers to the variant that does not incorporate appearance transformation module. w/o NP refers to the variant that does not apply monocular normal estimation as prior. Note that ground truth is unavailable for real data.

Table III: Performance comparison with baselines.
<table><tr><td>Method</td><td>Training Time</td><td>FPS</td></tr><tr><td>NeRF-OSR</td><td>31h</td><td>0.003</td></tr><tr><td>ReCap</td><td>2.7h</td><td>71.1</td></tr><tr><td>LumiGauss</td><td>1.3h</td><td>29.6</td></tr><tr><td>Ours</td><td>2.9h</td><td>38.6</td></tr><tr><td>Ours SHÃAO</td><td>3.8h</td><td>10.6</td></tr></table>

2) Normal Prior Loss $\mathcal { L } _ { { n p } } .$ During the first stage of geometry reconstruction we introduce a normal prior loss, ${ \mathcal { L } } _ { n p } ,$ to encourage more accurate surface normal estimation.

Table IV: Quantitative comparison of ablation study results on the synthetic dataset. w/o AT: removal of the appearance transformation module; w/o NP: removal of the normal prior loss ${ \mathcal { L } } _ { n p } ;$ SHÃAO: SH-based ambient light with MLP-predicted AO; w/o SG: removal of spherical Gaussian (SG) sunlight modeling; Forward: forward shading; Joint: joint training instead of our two-stage strategy; Ours: our full model.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Rendered</td><td colspan="3">Albedo</td><td>Normal</td></tr><tr><td></td><td>PSNR âSSIM âMSE âPSNR âSSIM âMSE âMAE (Â°) â</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o AT</td><td>19.14</td><td>0.753</td><td>0.012</td><td>19.19</td><td>0.754</td><td>0.012</td><td>47.99</td></tr><tr><td>w/o NP</td><td>20.84</td><td>0.763</td><td>0.009</td><td>20.78</td><td>0.758</td><td>0.009</td><td>36.05</td></tr><tr><td>SHÃAO</td><td>20.33</td><td>0.905</td><td>0.014</td><td>20.37</td><td>0.729</td><td>0.010</td><td>13.97</td></tr><tr><td>w/o SG</td><td>23.87</td><td>0.907</td><td>0.005</td><td>21.89</td><td>0.756</td><td>0.007</td><td>13.97</td></tr><tr><td>Forward</td><td>23.32</td><td>0.877</td><td>0.005</td><td>20.27</td><td>0.722</td><td>0.009</td><td>13.97</td></tr><tr><td>Joint</td><td>20.76</td><td>0.791</td><td>0.010</td><td>20.37</td><td>0.729</td><td>0.010</td><td>21.84</td></tr><tr><td>Ours</td><td>26.75</td><td>0.933</td><td>0.003</td><td>26.80</td><td>0.869</td><td>0.003</td><td>13.97</td></tr></table>

To evaluate its effectiveness, we present an ablation study in Fig. 8. Without the normal prior loss, the normals of large planar regions tend to deviate toward incorrect directions and become distorted, leading to inaccurate decompositions of texture and illumination. The quantitative results provided in Table IV further confirm the effectiveness of the normal prior loss, showing clear improvements in the fidelity of the estimated normal maps.

3) SH-based Ambient Lighting with AO vs. PRT: As described in Sec. III-C, we employ a hybrid lighting representation that combines a spherical Gaussian (SG) sunlight with a PRT-based ambient component. For the ambient light, we compare our PRT-based model against the alternative approach used in SOL-NeRF [2], which represents ambient illumination using Spherical Harmonics (SH) together with an MLP-predicted ambient occlusion (AO). The qualitative results in Fig. 9 and the quantitative results in Table IV demonstrate that our model achieves higher-quality renderings and more accurate albedo estimation. The shadow map further reveals that the MLP-based AO tends to incorrectly bake material textures into the shadow map. Moreover, the performance comparison in Table III shows that our model is over three times faster than the SHÃAO baseline.

<!-- image-->  
Figure 9: Qualitative comparison of reconstruction results between our full model and two variants. SHÃAO denotes the variant adopting the same lighting model as SOL-NeRF [2], which uses an SH-based environment map with an MLPpredicted ambient occlusion. w/o SG indicates the variant without Spherical Gaussian (SG) sunlight modeling, leading to inaccurate shadows and degraded albedo quality.

4) Lighting Model without SG: We compare our full model against a baseline that relies solely on an omnidirectional environment light representation. As shown in Table IV and Fig. 9, our full model produces more accurate and welldecoupled albedo. In contrast, the baseline without SG tends to overfit illumination, causing shadows to be baked into the albedo or locked into the illumination map, which prevents them from adapting correctly to changes in lighting conditions.

5) Deferred vs. Forward Shading: We further conducted an ablation study to evaluate our deferred shading strategy. In the forward shading variant, shading is computed at each Gaussian before splatting, and shadows are estimated per Gaussian by testing visibility from the SG light to each Gaussian. As analyzed in Sec. III-C, while the final alpha-blended normal map appears smooth, the normals at individual Gaussians are often insufficiently accurate for high-quality shading. As a result, the forward shading approach produces noisy shadow maps and degraded rendering quality, as shown in Fig. 10. Quantitative comparisons in Table IV further confirm the superiority of our deferred shading design.

<!-- image-->

<!-- image-->  
GT

Figure 10: Qualitative comparison of decomposed diffuse albedo and shadow maps with two variants. âForwardâ denotes a variant based on forward shading, while âJointâ indicates a joint training strategy in contrast to our two-stage approach. Note that our two-stage method significantly improves reconstruction and shadow separation.  
<!-- image-->  
Figure 11: Qualitative comparison of a failure case with a reflective surface exhibiting strong specular highlights.

6) Two-Stage vs. Joint Training: Our method consists of two stages for constructing geometry and for decomposing texture and lighting, respectively. To assess the effectiveness of this two-stage pipeline, we introduce a baseline that jointly decomposes all three components in a single step. However, as shown in Fig. 10, the joint training baseline results in blurry reconstructions and less accurate decomposition. Further quantitative comparisons in Table IV confirm the superiority of our two-stage training strategy, highlighting its advantage in achieving more precise and faithful reconstruction and decomposition results.

## V. CONCLUSION

## A. Technical Summary

We introduced ROS-GS, a Gaussian Splatting framework for outdoor relighting, effectively combining rapid rendering with physically-based realism for photorealistic lighting manipulation. Its two-stage approach first reconstructs scene geometry using 2DGS and monocular normal priors from multi-view images. Subsequently, it employs physically-based rendering and a hybrid sun-sky lighting model to decompose view-independent texture, high-frequency sunlight, and lowfrequency skylight. Progressive optimization allows ROS-GS to disentangle these components from images captured under varying conditions, enabling plausible, efficient outdoor relighting. Experiments demonstrate ROS-GS surpasses current NeRF-based and 3DGS-based baselines.

## B. Limitations and Future works

Our current model assumes primarily diffuse surface materials, which may be suboptimal for highly reflective scenes as shown in Fig. 11. Additionally, its reliance on mesh-based ray tracing for visibility omits light interactions with dynamic scene elements. Future work will address these by incorporating more sophisticated material models and exploring advanced rendering techniques to handle complex illumination and dynamic scenes more robustly, thereby enhancing realism and applicability.

## REFERENCES

[1] V. Rudnev, M. Elgharib, W. A. P. Smith, L. Liu, V. Golyanik, and C. Theobalt, âNerf for outdoor scene relighting,â in Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XVI, ser. Lecture Notes in Computer Science, S. Avidan, G. J. Brostow, M. Cisse, G. M. Farinella, and Â´ T. Hassner, Eds., vol. 13676. Springer, 2022, pp. 615â631. [Online]. Available: https://doi.org/10.1007/978-3-031-19787-1 35

[2] J. Sun, T. Wu, Y. Yang, Y. Lai, and L. Gao, âSol-nerf: Sunlight modeling for outdoor scene decomposition and relighting,â in SIGGRAPH Asia 2023 Conference Papers, SA 2023, Sydney, NSW, Australia, December 12-15, 2023. ACM, 2023, pp. 31:1â31:11.

[3] J. A. D. Gardner, E. Kashin, B. Egger, and W. A. P. Smith, âThe skyâs the limit: Relightable outdoor scenes via a sky-pixel constrained illumination prior and outside-in visibility,â in Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LIV, ser. Lecture Notes in Computer Science, A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds., vol. 15112. Springer, 2024, pp. 126â143. [Online]. Available: https://doi.org/10.1007/978-3-031-72949-2 8

[4] Y. Chang, Y. Kim, S. Seo, J. Yi, and N. Kwak, âFast sun-aligned outdoor scene relighting based on tensorf,â in IEEE/CVF Winter Conference on Applications of Computer Vision, WACV 2024, Waikoloa, HI, USA, January 3-8, 2024. IEEE, 2024, pp. 3614â3624. [Online]. Available: https://doi.org/10.1109/WACV57701.2024.00359

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: Representing scenes as neural radiance fields for view synthesis,â in ECCV, 2020, pp. 405â421.

[6] J. Gao, C. Gu, Y. Lin, Z. Li, H. Zhu, X. Cao, L. Zhang, and Y. Yao, âRelightable 3d gaussians: Realistic point cloud relighting with BRDF decomposition and ray tracing,â in Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLV, ser. Lecture Notes in Computer Science, A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds., vol. 15103. Springer, 2024, pp. 73â89. [Online]. Available: https://doi.org/10.1007/978-3-031-72995-9 5

[7] Y. Jiang, J. Tu, Y. Liu, X. Gao, X. Long, W. Wang, and Y. Ma, âGaussianshader: 3d gaussian splatting with shading functions for reflective surfaces,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024. IEEE, 2024, pp. 5322â5332. [Online]. Available: https://doi.org/10.1109/CVPR52733.2024.00509

[8] C. Gu, X. Wei, Z. Zeng, Y. Yao, and L. Zhang, âIRGS: interreflective gaussian splatting with 2d gaussian ray tracing,â CoRR, vol. abs/2412.15867, 2024. [Online]. Available: https://doi.org/10.48550/ arXiv.2412.15867

[9] J. Li, Z. Wu, E. Zamfir, and R. Timofte, âRecap: Better gaussian relighting with cross-environment captures,â CoRR, vol. abs/2412.07534, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2412.07534

[10] Z. Liang, Q. Zhang, Y. Feng, Y. Shan, and K. Jia, âGS-IR: 3d gaussian splatting for inverse rendering,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024. IEEE, 2024, pp. 21 644â21 653. [Online]. Available: https://doi.org/10.1109/CVPR52733.2024.02045

[11] Z. Bi, Y. Zeng, C. Zeng, F. Pei, X. Feng, K. Zhou, and H. Wu, $^ { \mathrm { { \sc \circ } } } \mathrm { { G s } } ^ { 3 } \mathrm { { ; } }$ Efficient relighting with triple gaussian splatting,â in SIGGRAPH Asia 2024 Conference Papers, SA 2024, Tokyo, Japan, December 3-6, 2024, T. Igarashi, A. Shamir, and H. R. Zhang, Eds. ACM, 2024, pp. 12:1â 12:12. [Online]. Available: https://doi.org/10.1145/3680528.3687576

[12] J. Fan, F. Luan, J. Yang, M. Hasan, and B. Wang, âRNG: relightable neural gaussians,â CoRR, vol. abs/2409.19702, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2409.19702

[13] J. Kaleta, K. Kania, T. Trzcinski, and M. Kowalski, âLumigauss: Relightable gaussian splatting in the wild,â 2024.

[14] X. Zhang, P. P. Srinivasan, B. Deng, P. Debevec, W. T. Freeman, and J. T. Barron, âNeRFactor: Neural factorization of shape and reflectance under an unknown illumination,â ACM Trans. Graph., vol. 40, no. 6, pp. 1â18, 2021.

[15] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH 2024, Denver, CO, USA, 27 July 2024- 1 August 2024. ACM, 2024, p. 32.

[16] C. Ye, L. Qiu, X. Gu, Q. Zuo, Y. Wu, Z. Dong, L. Bo, Y. Xiu, and X. Han, âStablenormal: Reducing diffusion variance for stable and sharp normal,â ACM Trans. Graph., vol. 43, no. 6, pp. 250:1â250:18, 2024.

[17] M. Levoy and P. Hanrahan, âLight field rendering,â in Proceedings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 1996, New Orleans, LA, USA, August 4-9, 1996. ACM, 1996, pp. 31â42.

[18] A. Davis, M. Levoy, and F. Durand, âUnstructured light fields,â Comput. Graph. Forum, vol. 31, no. 2pt1, pp. 305â314, 2012.

[19] J. Song, X. Chen, and O. Hilliges, âMonocular neural image based rendering with continuous view control,â in 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019. IEEE, 2019, pp. 4089â4099.

[20] O. Wiles, G. Gkioxari, R. Szeliski, and J. Johnson, âSynsin: End-to-end view synthesis from a single image,â in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020. Computer Vision Foundation / IEEE, 2020, pp. 7465â7475.

[21] M. Shih, S. Su, J. Kopf, and J. Huang, â3d photography using contextaware layered depth inpainting,â in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020. Computer Vision Foundation / IEEE, 2020, pp. 8025â8035.

[22] Y. Han, R. Wang, and J. Yang, âSingle-view view synthesis in the wild with learned adaptive multiplane images,â in SIGGRAPH â22: Special Interest Group on Computer Graphics and Interactive Techniques Conference, Vancouver, BC, Canada, August 7 - 11, 2022. ACM, 2022, pp. 14:1â14:8.

[23] Q. Wang, Z. Li, D. Salesin, N. Snavely, B. Curless, and J. Kontkanen, â3d moments from near-duplicate photos,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE, 2022, pp. 3896â3905.

[24] D. C. Luvizon, G. S. P. Carvalho, A. A. dos Santos, J. S. ConceicÂ¸ao, J. L. Ë Flores-Campana, L. G. L. Decker, M. R. e Souza, H. Pedrini, A. Joia, and O. A. B. Penatti, âAdaptive multiplane image generation from a single internet picture,â in IEEE Winter Conference on Applications of Computer Vision, WACV 2021, Waikoloa, HI, USA, January 3-8, 2021. IEEE, 2021, pp. 2555â2564.

[25] Z. Chen and H. Zhang, âLearning implicit fields for generative shape modeling,â in CVPR, 2019, pp. 5939â5948.

[26] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, âDeepsdf: Learning continuous signed distance functions for shape representation,â in CVPR, 2019, pp. 165â174.

[27] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger, âOccupancy networks: Learning 3d reconstruction in function space,â in CVPR, 2019, pp. 4460â4470.

[28] M. Niemeyer, L. Mescheder, M. Oechsle, and A. Geiger, âDifferentiable volumetric rendering: Learning implicit 3d representations without 3d supervision,â in CVPR, 2020, pp. 3501â3512.

[29] J. Thies, M. Zollhofer, and M. NieÃner, âDeferred neural rendering: Â¨ image synthesis using neural textures,â ACM Trans. Graph., vol. 38, no. 4, pp. 66:1â66:12, 2019.

[30] M. Oechsle, L. M. Mescheder, M. Niemeyer, T. Strauss, and A. Geiger, âTexture fields: Learning texture representations in function space,â in ICCV, 2019, pp. 4530â4539.

[31] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Trans. Graph., vol. 42, no. 4, pp. 139:1â139:14, 2023.

[32] M. M. Kazhdan, M. Bolitho, and H. Hoppe, âPoisson surface reconstruction,â in Eurographics Symposium on Geometry Processing, 2006.

[33] M. M. Kazhdan and H. Hoppe, âScreened poisson surface reconstruction,â ACM Trans. Graph., vol. 32, no. 3, pp. 29:1â29:13, 2013.

[34] W. E. Lorensen and H. E. Cline, âMarching cubes: A high resolution 3d surface construction algorithm,â in Proceedings of the 14th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 1987, Anaheim, California, USA, July 27-31, 1987, M. C. Stone, Ed. ACM, 1987, pp. 163â169.

[35] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, âNeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction,â in Advances in Neural Information Processing Systems, 2021, pp. 27 171â27 183.

[36] M. Oechsle, S. Peng, and A. Geiger, âUnisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction,â in ICCV, 2021, pp. 5589â5599.

[37] X. Long, C. Lin, P. Wang, T. Komura, and W. Wang, âSparseneus: Fast generalizable neural surface reconstruction from sparse views,â in ECCV 2022, vol. 13692, 2022, pp. 210â227.

[38] Z. Yu, S. Peng, M. Niemeyer, T. Sattler, and A. Geiger, âMonosdf: Exploring monocular geometric cues for neural implicit surface reconstruction,â in Advances in Neural Information Processing Systems, 2022.

[39] A. Guedon and V. Lepetit, âSugar: Surface-aligned gaussian splatting Â´ for efficient 3d mesh reconstruction and high-quality mesh rendering,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024. IEEE, 2024, pp. 5354â5363.

[40] P. Dai, J. Xu, W. Xie, X. Liu, H. Wang, and W. Xu, âHigh-quality surface reconstruction using gaussian surfels,â in ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH 2024, Denver, CO, USA, 27 July 2024- 1 August 2024. ACM, 2024, p. 22.

[41] S. Bi, Z. Xu, K. Sunkavalli, M. Hasan, Y. Hold-Geoffroy, D. J. Kriegman, and R. Ramamoorthi, âDeep reflectance volumes: Relightable reconstructions from multi-view photometric images,â in Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part III, vol. 12348. Springer, 2020, pp. 294â311.

[42] S. Bi, Z. Xu, P. P. Srinivasan, B. Mildenhall, K. Sunkavalli, M. Hasan, Y. Hold-Geoffroy, D. J. Kriegman, and R. Ramamoorthi, âNeural reflectance fields for appearance acquisition,â CoRR, vol. abs/2008.03824, 2020.

[43] K. Zhang, F. Luan, Q. Wang, K. Bala, and N. Snavely, âPhySG: Inverse rendering with spherical gaussians for physics-based material editing and relighting,â in CVPR, 2021, pp. 5453â5462.

[44] J. Munkberg, W. Chen, J. Hasselgren, A. Evans, T. Shen, T. Muller, Â¨ J. Gao, and S. Fidler, âExtracting triangular 3d models, materials, and lighting from images,â in CVPR, 2022, pp. 8270â8280.

[45] J. Hasselgren, N. Hofmann, and J. Munkberg, âShape, light, and material decomposition from images using monte carlo rendering and denoising,â in Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

[46] T. Shen, J. Gao, K. Yin, M.-Y. Liu, and S. Fidler, âDeep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis,â in Advances in Neural Information Processing Systems, 2021, pp. 6087â 6101.

[47] Y. Liu, P. Wang, C. Lin, X. Long, J. Wang, L. Liu, T. Komura, and W. Wang, âNero: Neural geometry and BRDF reconstruction of reflective objects from multiview images,â ACM Trans. Graph., vol. 42, no. 4, pp. 114:1â114:22, 2023.

[48] T. Wu, J. Sun, Y. Lai, and L. Gao, âDe-nerf: Decoupled neural radiance fields for view-consistent appearance editing and high-frequency environmental relighting,â in ACM SIGGRAPH 2023 Conference Proceedings, SIGGRAPH 2023, Los Angeles, CA, USA, August 6-10, 2023. ACM, 2023, pp. 74:1â74:11.

[49] ââ, âVd-nerf: Visibility-aware decoupled neural radiance fields for view-consistent editing and high-frequency relighting,â IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 5, pp. 3344â3357, 2025. [Online]. Available: https://doi.org/10.1109/TPAMI.2025.3531417

[50] D. Verbin, P. Hedman, B. Mildenhall, T. E. Zickler, J. T. Barron, and P. P. Srinivasan, âRef-nerf: Structured view-dependent appearance for neural radiance fields,â in CVPR, 2022, pp. 5481â5490.

[51] H. Bai, J. Zhu, S. Jiang, W. Huang, T. Lu, Y. Li, J. Guo, R. Fu, Y. Guo, and L. Chen, âGare: Relightable 3d gaussian splatting for outdoor scenes from unconstrained photo collections,â CoRR, vol. abs/2507.20512, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2507.20512

[52] J. T. Kajiya, âThe rendering equation,â in Proceedings of the 13th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 1986, Dallas, Texas, USA, August 18-22, 1986. ACM, 1986, pp. 143â150.

[53] P. J. Sloan, J. Kautz, and J. M. Snyder, âPrecomputed radiance transfer for real-time rendering in dynamic, low-frequency lighting environments,â ACM Trans. Graph., vol. 21, no. 3, pp. 527â536, 2002. [Online]. Available: https://doi.org/10.1145/566654.566612

[54] R. Martin-Brualla, N. Radwan, M. S. M. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, âNerf in the wild: Neural radiance fields for unconstrained photo collections,â in IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021. Computer Vision Foundation / IEEE, 2021, pp. 7210â7219.

[55] J. Kulhanek, S. Peng, Z. Kukelova, M. Pollefeys, and T. Sattler, âWildgaussians: 3d gaussian splatting in the wild,â CoRR, vol. abs/2407.08447, 2024.

[56] M. Contributors, âMMSegmentation: Openmmlab semantic segmentation toolbox and benchmark,â https://github.com/open-mmlab/ mmsegmentation, 2020.

[57] R. Ramamoorthi and P. Hanrahan, âAn efficient representation for irradiance environment maps,â in Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 2001, Los Angeles, California, USA, August 12-17, 2001, L. Pocock, Ed. ACM, 2001, pp. 497â500. [Online]. Available: https://doi.org/10.1145/383259.383317

[58] T. Wu, J. Sun, Y. Lai, Y. Ma, L. Kobbelt, and L. Gao, âDeferredgs: Decoupled and editable gaussian splatting with deferred shading,â CoRR, vol. abs/2404.09412, 2024.

[59] K. Ye, Q. Hou, and K. Zhou, â3d gaussian splatting with deferred reflection,â in ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH 2024, Denver, CO, USA, 27 July 2024- 1 August 2024. ACM, 2024, p. 40.

[60] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE Trans. Image Process., vol. 13, no. 4, pp. 600â612, 2004. [Online]. Available: https://doi.org/10.1109/TIP.2003.819861

[61] T. Nishita, T. Sirai, K. Tadamura, and E. Nakamae, âDisplay of the earth taking into account atmospheric scattering,â in Proceedings of the 20th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 1993, Anaheim, CA, USA, August 2-6, 1993. ACM, 1993, pp. 175â182.

[62] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: From error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, 2004.

[63] Y. Yu, A. Meka, M. Elgharib, H. Seidel, C. Theobalt, and W. A. P. Smith, âSelf-supervised outdoor scene relighting,â in Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXII, ser. Lecture Notes in Computer Science, A. Vedaldi, H. Bischof, T. Brox, and J. Frahm, Eds., vol. 12367. Springer, 2020, pp. 84â101. [Online]. Available: https://doi.org/10.1007/978-3-030-58542-6 6

[64] J. Philip, M. Gharbi, T. Zhou, A. A. Efros, and G. Drettakis, âMulti-view relighting using a geometry-aware network,â ACM Trans. Graph., vol. 38, no. 4, pp. 78:1â78:14, 2019. [Online]. Available: https://doi.org/10.1145/3306346.3323013

[65] Z. Wang, T. Shen, J. Gao, S. Huang, J. Munkberg, J. Hasselgren, Z. Gojcic, W. Chen, and S. Fidler, âNeural fields meet explicit geometric representations for inverse rendering of urban scenes,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 2023, pp. 8370â8380. [Online]. Available: https://doi.org/10.1109/CVPR52729.2023.00809