# RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection

Yongyang Zhou yongyangzhou@bit.edu.cn Beijing Institute of Technology Beijing, China

Zichen Wang zichenwang@bit.edu.cn Beijing Institute of Technology Beijing, China

Fang-Lue Zhang fanglue.zhang@vuw.ac.nz Victoria University of Wellington Wellington, New Zealand

Lei Zhang leizhang@bit.edu.cn Beijing Institute of Technology Beijing, China

<!-- image-->  
Figure 1: We propose RTR-GS, a framework for geometry-light-material decomposition from multi-view images. Our method significantly enhances normal estimation and visual realism for reflective surfaces compared to GS-IR [31] and GShader [22]. Additionally, we achieve material and lighting decomposition while accounting for secondary lighting effects through physically-based deferred rendering. The material components include albedo, metallic, and roughness. This high-quality decomposition enables realistic relighting and material editing.

## Abstract

3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities in novel view synthesis. However, rendering reflective objects remains a significant challenge, particularly in inverse rendering and relighting. We introduce RTR-GS, a novel inverse rendering framework capable of robustly rendering objects with arbitrary reflectance properties, decomposing BRDF and lighting, and delivering credible relighting results. Given a collection of multi-view images, our method effectively recovers geometric structure through a hybrid rendering model that combines forward rendering for radiance transfer with deferred rendering for reflections. This approach successfully separates high-frequency and low-frequency appearances, mitigating floating artifacts caused by spherical harmonic overfitting when handling high-frequency details. We further refine BRDF and lighting decomposition using an additional physicallybased deferred rendering branch. Experimental results show that our method enhances novel view synthesis, normal estimation, decomposition, and relighting while maintaining efficient training inference process. https://github.com/ZyyZyy06/RTR-GS

## CCS Concepts

â¢ Computing methodologies â Rasterization; â¢ Point-based models; â¢ Machine learning approaches; â¢ Rendering;

## Keywords

Novel view synthesis, Gaussian Splatting, Relighting

## 1 Introduction

Inverse rendering is a long-standing challenge that seeks to decompose a 3D sceneâs physical attributesâgeometry, materials, and lightingâfrom captured images. This decomposition enables downstream tasks such as relighting and editing. The problem is particularly challenging due to the complex interplay of these attributes during rendering, especially under unknown illumination conditions, which make it inherently under-constrained. Neural Radiance Fields (NeRF) [36] have achieved remarkable success in novel view synthesis, laying the groundwork for inverse rendering. Methods such as [7, 32, 62, 64] use implicit neural representations, like Multi-Layer Perceptrons (MLPs), to decompose physical components. However, MLPs suffer from limited expressiveness and high computational costs, making it challenging to balance quality and efficiency. 3D Gaussian Splatting (3DGS) [25] improves both the speed and quality of learning-based volumetric rendering, and several methods [16, 31, 43] have integrated physically-based rendering into this framework. However, spherical harmonic functions lack the directional resolution needed to accurately represent specular reflections, and overfitting during Gaussian splatting and cloning can introduce floating artifacts.

Accurate geometry is crucial for decomposing materials and lighting from complex appearances. However, high-frequency details can cause overfitting, leading to floating artifacts that deviate from physically smooth surfaces and compromise geometric accuracy. To address this issue, we propose using a reflection map to store specular components, isolating high-frequency appearance details from the radiance component to mitigate overfitting. Additionally, we replace independent spherical harmonics with radiance transfer rendering, which imposes stronger global lowfrequency constraints when computing radiance components. By separating high-frequency and low-frequency appearances, our method enables accurate recovery of geometric structures with arbitrary reflectance properties. Following geometry reconstruction, we model occlusion and indirect illumination by baking visibility into 3D voxels and introducing indirect lighting parameters. This approach reduces aliasing artifacts in albedo, shadows, and lighting during decomposition. Finally, we achieve effective material and lighting decomposition by integrating an additional differentiable, physically-based deferred rendering branch.

The primary contribution of our work is the introduction of a Gaussian splatting-based inverse rendering framework, RTR-GS, which accurately estimates surface normals, bidirectional reflectance distribution functions (BRDF), and environmental lighting from multi-view images of both diffuse and specular objects. Specifically, it includes the following key aspects:

â¢ We propose a 3DGS-based hybrid rendering model that integrates reflection maps with radiance transfer, effectively separating high-frequency and low-frequency appearances. This enables efficient rendering of objects with arbitrary reflectance properties while reducing floating artifacts, thereby improving geometric structure recovery with high-quality normals.

â¢ We further enhance appearance decomposition through a dual-branch rendering approach, enabling efficient and accurate material and lighting decomposition via rational lighting modeling and occlusion data baked into 3D voxels.

â¢ Comprehensive experiments demonstrate that our method achieves state-of-the-art performance in novel view synthesis and relighting, producing credible results for both diffuse and specular objects.

## 2 RELATED WORK

## 2.1 Neural representations

Recent advancements in Neural Radiance Fields (NeRF) [36] have garnered significant attention. Subsequent research has focused on enhancing rendering quality [2, 4, 26], improving surface reconstruction [29, 47, 54], and advancing object generation [11, 40, 49, 59], among other areas. Additionally, some methods aim to balance speed and quality [10, 12, 15, 20, 37, 45], facilitating more efficient evaluations.

3D Gaussian Splatting [25] effectively combines radiance field rendering with rasterization by leveraging discrete Gaussian distributions and the splatting technique. Subsequent research has focused on enhancing rendering quality [33, 57], more accurate geometry reconstruction [21, 35, 58], expanding editability [34, 60], and increasing scalability [39]. However, these methods do not decompose appearance into materials and lighting, limiting their suitability for relighting and editing tasks.

## 2.2 Inverse rendering

Inverse rendering aims to decompose physically-based attributes from observations, including geometry, material, and lighting. A variety of methods simplify this problem by assuming controllable lighting conditions [1, 5, 6, 17, 41]. Some works relax these assumptions to consider direct lighting effects [7, 8, 62]. These works [13, 51, 53, 61, 64, 65] model secondary lighting effects using additional MLPs. To reduce computational overhead, some methods [23, 28] employ tensor decomposition techniques inspired by TensoRF [12]. For compatibility with existing rendering pipelines, NvDiffrec [38] and NvDiffrecMC [19] utilize differentiable rendering with rasterization or ray-tracing pipelines.

Methods based on 3D Gaussian Splatting (3DGS) have significantly accelerated training and rendering. GS-IR [31], GIR [43], and R3DG [16] constrain surface normals using pseudo normals derived from depth and model shadows and indirect lighting through baking or ray-tracing. By leveraging pre-computed radiance transfer, PRT-GS [18] enables relighting, including secondary lighting effects. Phys3DGS [14] integrates 3D Gaussian splats with mesh-based representations. Although these methods retain the high efficiency of 3DGS, using spherical harmonic functions as a radiance representation for geometry recovery often introduces floating artifacts on reflective surfaces, leading to geometric inaccuracies.

## 2.3 Reflective object reconstruction

Reconstructing reflective objects poses a significant challenge in inverse rendering tasks due to the high-frequency appearance changes that result in view inconsistencies. Ref-NeRF [46] tries to address this by using reflection directions instead of view directions and introducing Integrating Direction Encoding (IDE) to model reflections effectively. NeRO [32] explicitly models the reflection process. Spec-Gaussian [52] simulates reflections using anisotropic Gaussians. Deferred rendering approaches, such as DeferredGS [50], 3DGS-DR [55], GS-ROR [66], and GUS-IR[30] replace forward rendering to better handle reflections. GaussianShader [22] separates specular components and incorporates residual terms to capture secondary lighting effects. Additionally, PRD-GS [56] introduces progressive radiance distillation.

Inspired by these works, we adopt 3D Gaussians as the scene representation and develop an inverse rendering framework capable of effectively rendering object with arbitrary reflectance properties while also decomposing material and lighting components.

<!-- image-->  
Figure 2: RTR-GS Rendering Pipeline. Our rendering pipeline consists of a hybrid rendering branch and a physically-based rendering branch. The hybrid rendering branch computes the radiance color for each Gaussian using forward rendering through radiance transfer, which is then blended with the reflections from deferred rendering after splatting. The physically-based rendering branch is fully implemented during the deferred rendering phase. Initially, the hybrid rendering branch reconstructs the fundamental geometric structure and stores visibility in voxel grids. The physically-based rendering branch is then activated to further decompose material appearances.

## 3 Method

## 3.1 Overview

Figure 2 illustrates the overall framework of the proposed RTR-GS. We initialize 3D Gaussians using sparse point clouds generated randomly or estimated by COLMAP [42]. To model reflections, it is essential to define the normals for the Gaussians. We define normals as the shortest axis of each Gaussian, oriented toward the viewing direction, and optimize them synergistically using deferred rendering of reflections and pseudo-normals derived from a depth map (Sec. 3.2). Subsequently, we refine the Gaussians by introducing additional parameters and integrating key components into a hybrid rendering model (Sec. 3.3). This model combines radiance from forward rendering with reflections from deferred rendering, effectively separating high-frequency and low-frequency appearances to better represent complex materials and achieve high-quality scene reconstruction. Next, we decompose the appearance using differentiable physically-based deferred rendering, incorporating occlusion baking, indirect lighting modeling, and additional BRDF parameters. During this process, we employ two rendering branches simultaneously to refine the geometry (Sec. 3.4). Finally, we enhance the results through rendering losses and additional regularization terms (Sec. 3.5).

## 3.2 Deferred Rendering and Normal Modeling

In the 3DGS framework, the attributes of multiple Gaussians are blended in the image plane using splatting and alpha blending, as follows:

$$
I _ { f } = \sum _ { i = 0 } ^ { N } f _ { i } \alpha _ { i } T _ { i }\tag{1}
$$

where $\alpha _ { i }$ is the opacity, $\begin{array} { r } { T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } \end{array}$ represents the accumulated transmittance, $f _ { i }$ denotes the parameters of the ??-th Gaussian, and $I _ { f }$ represents the splatted screen-space attribute buffer. In vanilla 3DGS [25], outgoing radiance is computed per-Gaussian before blending. This process is referred to as forward rendering. Additionally, the attributes associated with each Gaussian can be transformed into screen space for subsequent shading, a process known as deferred rendering. The following section explains our normal design and optimization based on the deferred rendering implementation.

Accurate normals are essential for modeling reflection. We define the normal direction as the shortest axis of the Gaussian. During the optimization process, the Gaussian shape typically flattens as it aligns with the surface, causing the shortest axis to correspond to a larger area. Similar to GS-IR [31] and R3DG [16], we optimize normals by enforcing consistency between the pseudo-normal map $\hat { \mathbf { n } } _ { \mathbf { d } } ,$ derived from the depth map, and the Gaussian normals map n, as follows:

$$
\mathcal { L } _ { n } = \| \mathbf { n } - \hat { \mathbf { n } } _ { \bf d } \| _ { 2 }\tag{2}
$$

This constraint is effective in optimizing normals when the depth map is smooth enough. Additionally, normals are used to compute reflection directions and contribute to deferred rendering. This process enables rendering losses to be backpropagated to the normals, refining the Gaussian shape. When specular reflection is dominant, rendering losses from reflections primarily drive normal optimization. Conversely, in diffuse regions, depth-derived pseudo-normals impose a stronger constraint. Figure 3 illustrates the normal optimization process. Inspired by 3DGS-DR [55], we also introduce a simplified normal propagation mechanism that periodically enhances Gaussian opacity, improving the modelâs robustness against extreme specular reflections.

<!-- image-->  
Figure 3: By adjusting the shapes of the Gaussians using the pseudo normals and gradients from the reflection map, the normals are optimized.

## 3.3 Hybrid Rendering and Radiance Transfer

To effectively render appearances with diverse variations and to mitigate Gaussian floating artifacts caused by limited representation capability, we propose a hybrid rendering approach to replace the spherical harmonics-based forward rendering in 3DGS [25]. Our hybrid rendering model separates radiance and reflection to capture low-frequency and high-frequency components, respectively. Specifically, the radiance is computed using forward rendering, while the reflection is obtained through deferred rendering. The two components are then adaptively blended based on the reflection intensity as follows:

$$
I _ { r g b } = C _ { r } \cdot \left( 1 . 0 - R _ { i } \right) + C _ { r e f } \cdot R _ { i }\tag{3}
$$

where $C _ { r }$ is the radiance color, $C _ { r e f }$ is the reflection color, and $R _ { i }$ is the reflection intensity. The final blending is done in screen space. Further details on the reflection and radiance components are provided in the following sections.

Reflection. In forward rendering, BRDF lobes are computed individually using the respective normal of each Gaussian and are then blended after shading. However, this blending process broadens the final BRDF lobe, resulting in blurry rendering effects. In contrast, deferred rendering generates a single BRDF lobe based on the blended normal, providing higher precision and better preservation of BRDF sharpness. Similar observations have been analyzed in GUS-IR [30] and GS-ROR [66].

For each Gaussian, we introduce additional reflection attributes for deferred rendering: reflection tint $R _ { t }$ and reflection roughness $R _ { r }$ . We adopt a microfacet BRDF to simulate surfaces with varying roughness levels and achieve efficient computation using the splitsum approximation [24]. The final reflection color is computed as:

$$
C _ { r e f } = R _ { t } \cdot F _ { r e f } ( E _ { r } , R _ { r } , \mathbf { n } , \mathbf { v } )\tag{4}
$$

where $E _ { r }$ is a learnable reflection map, n and v denote the normal and the view direction, respectively. $F _ { r e f }$ represents the split-sum approximation [24], which will be explained in more detail in Section $3 . 4 .$

Radiance. Inspired by Precomputed Radiance Transfer (PRT) [44], we adopt radiance transfer instead of spherical harmonics to compute outgoing radiance. Firstly, we will describe how radiance transfer is used to shade each Gaussian, including both view-independent and view-dependent components. Then we will explain the motivation behind using radiance transfer.

The view-independent component is consistent with the radiance transfer rendering in PRT. This calculation approximates the diffuse part of rendering equation as a dot product of two vectors as follows:

$$
C _ { d } \approx \rho _ { d } \sum _ { j = 0 } ^ { n ^ { 2 } } c _ { j } c _ { j } ^ { t }\tag{5}
$$

where $\rho _ { d }$ represents the diffuse base color, $c _ { j }$ denotes the coefficients of the spherical harmonics lighting, and $c _ { j } ^ { t }$ represents the transfer vector. Notably, all Gaussians share the same spherical harmonics lighting $c _ { j }$ but use individual transfer vector $c _ { j } ^ { t } .$

For the view-dependent component, following the derivation in PRT [44], we need to compute a radiance transfer matrix to convert environmental lighting into transferred lighting. However, ??-order spherical harmonics lighting requires $n ^ { 2 ^ { \cdot } }$ parameters to store the transfer matrix, leading to rapidly increasing storage costs as the number of Gaussians grows. To address this issue, we adopt neural radiance transfer for the view-dependent component and compute it in a manner similar to the view-independent case. Specifically, for each Gaussian, we introduce a set of randomly initialized radiance transfer features $f _ { t }$ and a specular base color $\rho _ { s }$ . We decode $f _ { t }$ and the reflection direction o using a lightweight MLP ?? to obtain the neural radiance transfer vector $c _ { j } ^ { t } ( \mathbf { o } )$ . The view-dependent outgoing radiance is computed as:

$$
C _ { s } ( \mathbf { 0 } ) \approx \rho _ { s } \sum _ { j = 0 } ^ { n ^ { 2 } } c _ { j } c _ { j } ^ { t } ( \mathbf { 0 } ) , \quad w i t h \quad c _ { j } ^ { t } ( \mathbf { 0 } ) = G ( f _ { t } , \mathbf { 0 } )\tag{6}
$$

The total outgoing radiance is given by $C _ { r } = C _ { d } { + } C _ { s } ( \mathbf { o } )$ . After Gaussian splatting and blending, this radiance further participates in the blending process during deferred rendering. A detailed derivation of our radiance transfer implementation is provided in the supplementary materials.

Compared to spherical harmonics, radiance transfer allows us to maintain enougth representational capacity while providing stronger global low-frequency constraints. In the shading process, all Gaussians share two global components: the spherical harmonics lighting $c _ { j }$ and the MLP ??. This design enables shading across Gaussians to be connected through shared components, promoting the representation of overall low-frequency variations. Meanwhile, each Gaussian has its own independent transfer vector and transfer features, along with base color attributes. This enables our radiance transfer representation to better handle components that are difficult to recover in the reflection part, such as local reflections and shadows. Figure 4 illustrates the differences between our radiance transfer representation and spherical harmonics in modeling the radiance component. While the rendering results exhibit comparable visual quality, radiance transfer demonstrates better performance in low-frequency component fitting, prevents artifact generation, and maintains geometric smoothness.

Rendering  
Normal  
Depth  
Reflection  
Radiance  
<!-- image-->  
Figure 4: Radiance transfer provides a better representation of low-frequency appearances and helps prevent artifacts caused by overfitting high-frequency details.Such artifacts can degrade the smoothness of depth and normal estimations, reducing the quality of the reconstructed geometry, and adversely affect subsequent decomposition processes.

## 3.4 Illumination Modeling and Decomposition

We primarily use differentiable physically-based deferred rendering to decompose appearance into material and lighting components. To prevent aliasing artifacts in shadows, lighting, and albedo, we leverage the recovered geometric structure to bake occlusion information into a voxel grid, following the approach in GS-IR [31]. Specifically, we set the background color to white and assign black to the Gaussian regions. The scene is then projected to generate a cubemap texture, which is converted into spherical harmonics coefficients and stored in the voxel grid. In the following, we describe our material and illumination modeling in detail.

For materials, we assign BRDF attributes to each Gaussian, including albedo c, metallic ??, and roughness ?? . For illumination, we use an environmental cubemap to implement image-based lighting (IBL) for handling direct lighting. Additionally, we add a parameter $L _ { i n d } \in [ 0 , 1 ] ^ { 3 }$ for each Gaussian to represent diffuse indirect lighting. The rendering equation $\begin{array} { r } { L ( \mathbf { o } ) = \int _ { \Omega } { L _ { i } ( \mathbf { i } ) f ( \mathbf { i } , \mathbf { o } ) ( \mathbf { i } } } \end{array}$ Â· n)??i is separated into diffuse and specular components to simplify computation. The diffuse component $L _ { d }$ is computed as follows:

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { d } ( { \bf x } ) = \frac { c } { \pi } \int _ { \Omega } L _ { i } ( { \bf x , i } ) ( { \bf n \cdot i } ) d { \bf i } } \ ~ } \\ { { \displaystyle ~ = \frac { c } { \pi } [ \int _ { \Omega } L _ { i } ^ { d i r } ( { \bf x , i } ) ( { \bf n \cdot i } ) d { \bf i } + \int _ { \Omega } L _ { i } ^ { i n d } ( { \bf x , i } ) ( { \bf n , i } ) d { \bf i } ] } \ ~ } \\ { { \displaystyle ~ \approx \frac { c } { \pi } [ V ( { \bf x } ) L _ { d } ^ { d i r } ( { \bf x } ) + ( 1 - V ( { \bf x } ) ) L _ { d } ^ { i n d } ( { \bf x } ) ) ] } \ ~ } \end{array}\tag{7}
$$

where $L _ { d } ^ { d i r } ( { \bf x } )$ represents the direct environmental illumination, which depends only on the normal direction n. This value is precomputed for efficiency and stored in a 2D texture. The indirect illumination $L _ { d } ^ { i n d } ( \mathbf x )$ is derived through the splatting and blending of $L _ { i n d } .$ . The visibility term ?? (x) is determined by applying trilinear interpolation to the precomputed spherical harmonics stored in the baked voxel grid.

For the specular $L _ { s } ,$ we employ the split-sum approximation [24], treating it as the product of two independent integrals as follows:

$$
L _ { s } ( \mathbf { x } , \mathbf { o } ) \approx \int _ { \Omega } f _ { s } ( \mathbf { i } , \mathbf { o } ) ( \mathbf { n } \cdot \mathbf { i } ) d \mathbf { i } \int _ { \Omega } L _ { i } ( \mathbf { x } , \mathbf { i } ) D ( \mathbf { i } , \mathbf { o } ) ( \mathbf { n } \cdot \mathbf { i } ) d \mathbf { i }\tag{8}
$$

where $f ( \mathbf { i } , \mathbf { o } )$ represents the microfacet BRDF [9]. The first term of the integral represents the BRDF, which is independent of the lighting. It is precomputed and stored in a Look-Up Table (LUT). The second term accounts for the incoming radiance modulated by the normal distribution function (NDF) ??, which is pre-integrated and represented using a filtered cubemap. Finally, the outgoing radiance is expressed as:

$$
L _ { o } ( \mathbf { x } , \mathbf { o } ) = L _ { d } ( \mathbf { x } ) + L _ { s } ( \mathbf { x } , \mathbf { o } )\tag{9}
$$

After completing deferred rendering, we obtain the final PBR result $I _ { p b r }$

In the decomposition process, we use both the previously mentioned hybrid rendering and PBR branches simultaneously, rather than freezing the geometric parameters or enabling only the PBR branch. This approach is adopted for two main reasons. Firstly, different rendering models still require corresponding geometric adjustments for proper adaptation, so completely freezing the geometric parameters is undesirable. We need to locally optimize the geometric attributes of the Gaussian to accommodate the newly introduced PBR branch. Secondly, since the PBR-related parameters are initialized randomly, using only PBR can easily lead to drastic changes in the geometric structure, which may render the baked visibility inapplicable. These two points will be further elaborated in the experimental section.

## 3.5 Optimization

Throughout the training process, we optimize the geometric attributes of the Gaussian, as well as various rendering attributes closely related to the two rendering branches, as illustrated by the 3D Gaussians in Figure 2. In addition, we need to optimize the small MLP ??, which is a 3-layer network with 64 hidden units, used to decode the transfer feature and reflection direction, as well as two 6 Ã 128 Ã 128 cubemaps: the reflection map for hybrid rendering and the environment map for PBR. We first activate the hybrid rendering branch and optimize the corresponding parameters. After restoring the basic geometric structure, we then activate the PBR branch and optimize all parameters. Finally, we outline the primary loss function and the specialized regularization terms.

Rendering losses. As in 3DGS[25], we calculate the hybrid rendering loss $\mathcal { L } _ { H R }$ and PBR loss L?????? using the following equation:

$$
\mathcal { L } = \left( 1 - \lambda \right) \mathcal { L } _ { 1 } ( \hat { I } , I _ { g t } ) + \lambda \mathcal { L } _ { D - S S I M } ( \hat { I } , I _ { g t } )\tag{10}
$$

Light regularization. We apply a light regularization assuming a natural white incident light [32, 38] for optimizing environment map used in PBR as follows:

$$
\mathcal { L } _ { l i g h t } = \sum _ { c } ( L _ { c } - \frac { 1 } { 3 } \sum _ { c } L _ { c } ) , c \in \{ R , G , B \}\tag{11}
$$

Metal reflection prior. Due to the reflective properties of metals, we aim to make the metallic parameter ?? in the PBR model as close

Table 1: NVS quality, training time and FPS on TensoIR, Shiny Blender and Stanford ORB datasets. âHRâ represents our hybrid rendering branch.
<table><tr><td rowspan="2">Methods</td><td colspan="3">TensoIR</td><td colspan="3">Shiny Blender</td><td colspan="3">Stanford ORB</td><td rowspan="2">Training Time</td><td rowspan="2">FPS</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>NeRO</td><td>32.60</td><td>0.933</td><td>0.082</td><td>30.96</td><td>0.953</td><td>0.081</td><td>29.25</td><td>0.970</td><td>0.060</td><td>8h</td><td>&lt;1</td></tr><tr><td>TensoIR</td><td>35.18</td><td>0.976</td><td>0.040</td><td>27.95</td><td>0.896</td><td>0.159</td><td>34.81</td><td>0.983</td><td>0.029</td><td>5h</td><td>4</td></tr><tr><td>GS-IR</td><td>34.80</td><td>0.960</td><td>0.047</td><td>26.98</td><td>0.874</td><td>0.152</td><td>32.95</td><td>0.928</td><td>0.054</td><td>0.4h</td><td>189</td></tr><tr><td>R3DG</td><td>37.15</td><td>0.981</td><td>0.024</td><td>27.30</td><td>0.922</td><td>0.121</td><td>38.54</td><td>0.988</td><td>0.016</td><td>1h</td><td>16</td></tr><tr><td>3DGS-DR</td><td>38.15</td><td>0.979</td><td>0.031</td><td>32.03</td><td>0.960</td><td>0.084</td><td>39.80</td><td>0.987</td><td>0.015</td><td>0.4h</td><td>271</td></tr><tr><td>GShader</td><td>37.13</td><td>0.982</td><td>0.023</td><td>30.87</td><td>0.953</td><td>0.088</td><td>36.02</td><td>0.989</td><td>0.017</td><td>1h</td><td>65</td></tr><tr><td>Ours</td><td>39.17</td><td>0.985</td><td>0.021</td><td>33.99</td><td>0.971</td><td>0.061</td><td>39.81</td><td>0.990</td><td>0.016</td><td>0.5h</td><td>133</td></tr><tr><td>Ours(HR)</td><td>41.39</td><td>0.988</td><td>0.017</td><td>35.24</td><td>0.975</td><td>0.055</td><td>40.49</td><td>0.991</td><td>0.014</td><td>0.5h</td><td>96</td></tr></table>

Table 2: Relighting quality is evaluated on the TensoIR, Shiny Blender, and Stanford ORB datasets.
<table><tr><td rowspan="2">Methods</td><td colspan="3">TensoIR</td><td colspan="3">Shiny Blender</td><td colspan="3">Stanford ORB</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>TensoIR</td><td>28.55</td><td>0.945</td><td>0.080</td><td>22.30</td><td>0.842</td><td>0.184</td><td>26.22</td><td>0.947</td><td>0.049</td></tr><tr><td>GShader</td><td>26.86</td><td>0.930</td><td>0.063</td><td>19.20</td><td>0.874</td><td>0.131</td><td>26.23</td><td>0.952</td><td>0.043</td></tr><tr><td>GS-IR</td><td>25.98</td><td>0.897</td><td>0.092</td><td>21.18</td><td>0.846</td><td>0.160</td><td>28.44</td><td>0.960</td><td>0.038</td></tr><tr><td>R3DG</td><td>28.52</td><td>0.931</td><td>0.069</td><td>20.69</td><td>0.869</td><td>0.141</td><td>27.88</td><td>0.957</td><td>0.039</td></tr><tr><td>Ours</td><td>30.10</td><td>0.944</td><td>0.053</td><td>26.16</td><td>0.928</td><td>0.084</td><td>28.93</td><td>0.967</td><td>0.029</td></tr></table>

<!-- image-->  
Figure 5: Qualitative comparisons on a synthetic dataset. Our method retains more details, particularly in specular regions.

as possible to the reflection intensity $R _ { i }$ in hybrid rendering, as follows:

$$
\mathcal { L } _ { m } = \mathcal { L } _ { 1 } ( m , R _ { i } )\tag{12}
$$

which encourages our two rendering branches to maintain appearance consistency in high-frequency regions. The effectiveness of this regularization term is discussed in the following section. In addition, we incorporate a bilateral smoothness term $\mathcal { L } _ { s }$ and an object mask constraint $\mathcal { L } _ { o }$ . The final loss $\mathcal { L }$ is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { H R } + \lambda _ { P B R } \mathcal { L } _ { P B R } + \lambda _ { 0 } \mathcal { L } _ { l i g h t } + \lambda _ { 1 } \mathcal { L } _ { m } + \lambda _ { 2 } \mathcal { L } _ { n } + \mathcal { L } _ { s } + \mathcal { L } _ { o } } \end{array}\tag{13}
$$

where $\lambda _ { P B R } = 0 ~ \mathrm { o r } ~ 1 , \lambda _ { 0 } = 0 . 0 0 3 , \lambda _ { 1 } = 0 . 1 , \lambda _ { 2 } = 0 . 0 2 .$ Detailed descriptions of $\mathcal { L } _ { s }$ and $\mathcal { L } _ { o }$ are provided in the supplementary materials.

## 4 Experiments

## 4.1 Evaluation Setup

Dataset and Metrics. For synthetic objects in the TensoIR [23] and Shiny Blender [46] datasets, as well as real objects in the Stanford ORB dataset [27], we evaluate the performance of novel view synthesis and relighting using PSNR, SSIM [48], and LPIPS [63] metrics. For the ball object in the Shiny Blender dataset, only qualitative results are provided due to the absence of relighting ground truth (GT). In addition, we use mean angular error (MAE) to evaluate the quality of normal estimation. In addition, we have also provided the results of training duration and inference speed (FPS). We further evaluate novel view synthesis on the Ref-Real [46] and MipNeRF-360 [3] datasets. Numbers in bold represent the best performance, while underscored numbers indicate the second-best performance.

Methods for Comparison. We compared the quality of novel view synthesis against several NeRF-based methods [23, 32] and 3DGS-based methods [16, 22, 31, 55]. In addition, we evaluated the relighting quality between different inverse rendering methods. All methods were implemented and trained using their publicly available code and default configurations.

## 4.2 Comparison with previous works

Novel view synthesis. Table 1 presents the quantitative comparison results for novel view synthesis (NVS) on object-level datasets. Our PBR results show clear advantages over other methods. Additionally, we provide our Hybrid Rendering (HR) branch results to demonstrate the effectiveness of the hybrid rendering model. Visual comparisons are provided in Figure 5. Notably, our method preserves stable geometric structures even with high-frequency surface variations, producing clearer and more accurate novel views. Furthermore, Table 3 presents our results on the Ref-Real dataset [46] and the Mip-NeRF 360 dataset [3], where our method achieves competitive quantitative results.

Relighting. Table 2 presents the results of the relighting comparison. For the TensoIR and Shiny Blender datasets, albedo is aligned to the ground truth via channel-wise scaling before relighting as described in [27, 62]. For the Stanford ORB dataset, albedo scaling is disabled to more accurately evaluate absolute decomposition performance on real objects. Results for the TensoIR and Shiny Blender datasets are averaged over all viewpoints under five different environment maps. For the Stanford ORB dataset, relighting is evaluated using the provided 20 image-environment map pairs. Visual comparisons are provided in Figure 6. Our methodâs superior detail preservation and effectively suppresses aliasing artifacts in both albedo and lighting, leading to more realistic and visually consistent relighting results. Notably, our approach maintains credibility under different relighting conditions, without significant surface artifacts appearing on either rough or smooth objects.

Normal and materials estimation. Table 4 and Figure 7 present the results of our normal estimation. Notably, in the presence of high-frequency surface details, our method effectively prevents surface discontinuities caused by floating artifacts. In Figure 9, we visualize the estimated albedo, metallic, roughness, normal, and environmental lighting components. Our framework successfully decomposes both diffuse and specular objects. For specular objects, we achieve high-quality decomposition results with clearer environmental lighting. Additional albedo estimation results and more qualitative comparisons are provided in the supplementary materials.

## 4.3 Ablation Study

We specifically evaluated the effectiveness of radiance transfer compared to spherical harmonics. Additionally, we performed ablation studies on simplified normal propagation to validate the contribution of our proposed components. We also evaluate the impact of the metal reflection prior introduced in Sec. 3.5. For decomposition process, we further conducted experiments of using fixed geometric parameters and disabling the hybrid rendering branch (i.e., using only the PBR branch) during appearance decomposition, to demonstrate the advantages of our dual-branch rendering framework.

Table 3: Novel view synthesis quality evaluated using PSNR, SSIM, and LPIPS on the Ref-Real dataset and the Mip-NeRF 360 dataset.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Ref-Real</td><td colspan="3">Mip-NeRF 360</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>GS-IR</td><td>23.41</td><td>0.606</td><td>0.297</td><td>26.18</td><td>0.801</td><td>0.200</td></tr><tr><td>GShader</td><td>21.13</td><td>0.578</td><td>0.375</td><td>22.33</td><td>0.577</td><td>0.329</td></tr><tr><td>3DGS-DR</td><td>23.51</td><td>0.638</td><td>0.343</td><td>25.14</td><td>0.783</td><td>0.304</td></tr><tr><td>Ours</td><td>23.54</td><td>0.627</td><td>0.337</td><td>26.65</td><td>0.806</td><td>0.233</td></tr></table>

Table 4: Normal estimation quality with Gaussian-based methods evaluated using MAEâ on the TensoIR dataset and the Shiny Blender dataset.
<table><tr><td></td><td>GS-IR</td><td>R3DG</td><td>3DGS-DR</td><td>GShader</td><td>Ours</td></tr><tr><td>TensoIR</td><td>5.313</td><td>5.914</td><td>5.728</td><td>5.303</td><td>5.347</td></tr><tr><td>Shiny Blender</td><td>9.328</td><td>9.238</td><td>3.632</td><td>4.800</td><td>3.091</td></tr></table>

Table 5: Ablation study of key components on the Shiny Blender dataset. "w/o radiance transfer" represents using SHs to calculate the radiance part in hybrid rendering. "Propagation" denotes simplified normal propagation. "Frozen geometry" indicates freezing geometry attributes during decomposition. "w/o hybrid rendering" refers to disabling the hybrid rendering branch during decomposition.
<table><tr><td>Ablations</td><td>NVS PSNRâ</td><td>Relighting PSNRâ</td></tr><tr><td>ours</td><td>33.99</td><td>26.16</td></tr><tr><td>w/o radiance transfer</td><td>32.15</td><td>25.85</td></tr><tr><td>w/o propagation</td><td>33.26</td><td>26.09</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { m }$ </td><td>33.76</td><td>25.88</td></tr><tr><td>w/ frozen geometry</td><td>31.49</td><td>24.66</td></tr><tr><td>w/o hybrid rendering</td><td>32.90</td><td>25.18</td></tr></table>

Analysis on radiance transfer. As illustrated in Figure 8, using radiance transfer instead of spherical harmonics to represent the radiance component in hybrid rendering reduces floating artifacts and prevents normal and visibility errors caused by local geometric inaccuracies, particularly for specular objects. These improvements significantly enhance the quality of relighting. As shown in Table 5, radiance transfer also leads to notable improvements in quantitative results.

Analysis on decomposition process. When decomposing the appearance, we simultaneously enable hybrid rendering and PBR to fine-tune the geometry, making it compatible with both rendering models. We also evaluate the effects of freezing geometric

<!-- image-->  
Figure 6: Qualitative comparisons of relighting with different environment lighting conditions.

<!-- image-->  
Ours

<!-- image-->  
R3DG

<!-- image-->  
GS-IR  
3DGS-DR

<!-- image-->  
GShader

<!-- image-->

<!-- image-->  
TensoIR

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 7: Qualitative comparisons of normal produced by different methods. Our method provides robust normal estimation.

Rendering

Visibility

Normal

Radiance

w/ Radiance Transfer

<!-- image-->  
Relighting

<!-- image-->  
w/o Radiance Transfer  
Rendering  
Albedo  
Metallic  
Normal  
Roughness  
Environment Lighting

<!-- image-->

<!-- image-->  
Figure 9: Normal, albedo, roughness, metallic and environment lighing results on synthetic dataset.  
Figure 8: Radiance transfer can more effectively separate lowfrequency components of appearance, thereby preventing artifacts caused by overfitting. These artifacts compromise geometric smoothness and degrade the quality of rendering and relighting.

parameters or enabling only the PBR branch, which demonstrates the limitations of single-branch approaches. As shown in Table 5, both frozen geometry and enabling the PBR branch only lead to significant quality degradation. The former occurs because the geometric structure required for hybrid rendering does not fully meet PBRâs requirements, while the latter leads to geometric mutations, rendering the baked occlusion ineffective.

Limitation We assume that lighting originates from an infinite distance, which differs from actual lighting conditions in large-scale scenes. Additionally, our method does not consider more complex indirect lighting effects, such as inter-reflections. These limitations are shown in Figure 10.

<!-- image-->  
Figure 10: Limitation of our method.

## 5 Conclusions

We introduce RTR-GS, an inverse rendering framework that enables realistic novel view synthesis and relighting through Gaussian splatting and deferred rendering. By separating high-frequency and lowfrequency appearances using reflection maps and radiance transfer, we achieve high-quality hybrid rendering and normal estimation. Building on this, we further decompose material and lighting from the appearance by an additional PBR branch. Experimental results demonstrate that our method delivers competitive performance in novel view synthesis and relighting across various objects. In the future, we aim to explore more precise rendering techniques and incorporate more complex secondary lighting effects.

## References

[1] Dejan Azinovic, Tzu-Mao Li, Anton Kaplanyan, and Matthias NieÃner. 2019. Inverse path tracing for joint material and lighting estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2447â2456.

[2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. 2021. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5855â5864.

[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. 2022. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5470â5479.

[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. 2023. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19697â 19705.

[5] Sai Bi, Zexiang Xu, Kalyan Sunkavalli, MiloÅ¡ HaÅ¡an, Yannick Hold-Geoffroy, David Kriegman, and Ravi Ramamoorthi. 2020. Deep reflectance volumes: Relightable reconstructions from multi-view photometric images. In Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part III 16. Springer, 294â311.

[6] Sai Bi, Zexiang Xu, Kalyan Sunkavalli, David Kriegman, and Ravi Ramamoorthi. 2020. Deep 3d capture: Geometry and reflectance from sparse multi-view images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5960â5969.

[7] Mark Boss, Raphael Braun, Varun Jampani, Jonathan T Barron, Ce Liu, and Hendrik Lensch. 2021. Nerd: Neural reflectance decomposition from image collections. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12684â12694.

[8] Mark Boss, Varun Jampani, Raphael Braun, Ce Liu, Jonathan Barron, and Hendrik Lensch. 2021. Neural-pil: Neural pre-integrated lighting for reflectance decomposition. Advances in Neural Information Processing Systems 34 (2021), 10691â10704.

[9] Brent Burley and Walt Disney Animation Studios. 2012. Physically-based shading at disney. In Acm Siggraph, Vol. 2012. vol. 2012, 1â7.

[10] Ang Cao and Justin Johnson. 2023. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 130â141.

[11] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. 2021. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5799â5809.

[12] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. 2022. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision. Springer, 333â350.

[13] Hao Chen, Bo He, Hanyu Wang, Yixuan Ren, Ser Nam Lim, and Abhinav Shrivastava. 2021. Nerv: Neural representations for videos. Advances in Neural Information Processing Systems 34 (2021), 21557â21568.

[14] Euntae Choi and Sungjoo Yoo. 2024. Phys3DGS: Physically-based 3D Gaussian splatting for inverse rendering. arXiv preprint arXiv:2409.10335 (2024).

[15] Sara Fridovich-Keil, Giacomo Meanti, Frederik RahbÃ¦k Warburg, Benjamin Recht, and Angjoo Kanazawa. 2023. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12479â12488.

[16] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. 2025. Relightable 3D Gaussians: realistic point cloud relighting with BRDF decomposition and ray tracing. In European Conference on Computer Vision. Springer, 73â89.

[17] Kaiwen Guo, Peter Lincoln, Philip Davidson, Jay Busch, Xueming Yu, Matt Whalen, Geoff Harvey, Sergio Orts-Escolano, Rohit Pandey, Jason Dourgarian, et al. 2019. The relightables: Volumetric performance capture of humans with realistic relighting. ACM Transactions on Graphics (ToG) 38, 6 (2019), 1â19.

[18] Yijia Guo, Yuanxi Bai, Liwen Hu, Ziyi Guo, Mianzhi Liu, Yu Cai, Tiejun Huang, and Lei Ma. 2024. PRTGS: Precomputed radiance transfer of gaussian Splats for real-time high-quality relighting. In Proceedings of the 32nd ACM International Conference on Multimedia. 5112â5120.

[19] Jon Hasselgren, Nikolai Hofmann, and Jacob Munkberg. 2022. Shape, light, and material decomposition from images using monte carlo rendering and denoising. Advances in Neural Information Processing Systems 35 (2022), 22856â22869.

[20] Peter Hedman, Pratul P Srinivasan, Ben Mildenhall, Jonathan T Barron, and Paul Debevec. 2021. Baking neural radiance fields for real-time view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5875â 5884.

[21] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024. 2d Gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers. 1â11.

[22] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. 2024. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5322â5332.

[23] Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang, Songfang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu, and Hao Su. 2023. Tensoir: Tensorial inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 165â174.

[24] Brian Karis and Epic Games. 2013. Real shading in unreal engine 4. Proc. Physically Based Shading Theory Practice 4, 3 (2013), 1.

[25] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139â1.

[26] Simin Kou, Fang-Lue Zhang, Jakob Nazarenus, Reinhard Koch, and Neil A Dodgson. 2025. OmniPlane: A recolorable representation for dynamic scenes in omnidirectional videos. IEEE Transactions on Visualization and Computer Graphics (2025).

[27] Zhengfei Kuang, Yunzhi Zhang, Hong-Xing Yu, Samir Agarwala, Elliott Wu, Jiajun Wu, et al. 2023. Stanford-orb: a real-world 3d object inverse rendering benchmark. Advances in Neural Information Processing Systems 36 (2023), 46938â 46957.

[28] Jia Li, Lu Wang, Lei Zhang, and Beibei Wang. 2024. Tensosdf: Roughness-aware tensorial representation for robust geometry and material reconstruction. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1â13.

[29] Zhaoshuo Li, Thomas MÃ¼ller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. 2023. Neuralangelo: High-fidelity neural surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8456â8465.

[30] Zhihao Liang, Hongdong Li, Kui Jia, Kailing Guo, and Qi Zhang. 2024. GUS-IR: Gaussian splatting with unified shading for inverse rendering. arXiv preprint arXiv:2411.07478 (2024).

[31] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and Kui Jia. 2024. Gs-ir: 3d gaussian splatting for inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21644â21653.

[32] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng Wang, Lingjie Liu, Taku Komura, and Wenping Wang. 2023. Nero: Neural geometry and brdf reconstruction of reflective objects from multiview images. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1â22.

[33] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. 2024. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20654â20664.

[34] Guan Luo, Tian-Xing Xu, Ying-Tian Liu, Xiao-Xiong Fan, Fang-Lue Zhang, and Song-Hai Zhang. 2024. 3D Gaussian editing with a single image. In Proceedings of the 32nd ACM International Conference on Multimedia. 6627â6636.

[35] Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi. 2024. 3dgsr: Implicit surface reconstruction

with 3d gaussian splatting. arXiv preprint arXiv:2404.00409 (2024).

[36] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99â106.

[37] Thomas MÃ¼ller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (TOG) 41, 4 (2022), 1â15.

[38] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao, Wenzheng Chen, Alex Evans, Thomas MÃ¼ller, and Sanja Fidler. 2022. Extracting triangular 3d models, materials, and lighting from images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8280â8290.

[39] Jakob Nazarenus, Simin Kou, Fang-Lue Zhang, and Reinhard Koch. 2024. Arbitrary optics for Gaussian splatting using space warping. Journal of Imaging 10, 12 (2024), 330.

[40] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022).

[41] Carolin Schmitt, Simon Donne, Gernot Riegler, Vladlen Koltun, and Andreas Geiger. 2020. On joint estimation of pose, geometry and svbrdf from a handheld scanner. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3493â3503.

[42] Johannes L Schonberger and Jan-Michael Frahm. 2016. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition. 4104â4113.

[43] Yahao Shi, Yanmin Wu, Chenming Wu, Xing Liu, Chen Zhao, Haocheng Feng, Jingtuo Liu, Liangjun Zhang, Jian Zhang, Bin Zhou, et al. 2023. Gir: 3d gaussian inverse rendering for relightable scene factorization. arXiv preprint arXiv:2312.05133 (2023).

[44] Peter-Pike Sloan, Jan Kautz, and John Snyder. 2023. Precomputed radiance transfer for real-time rendering in dynamic, low-frequency lighting environments. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 339â348.

[45] Cheng Sun, Min Sun, and Hwann-Tzong Chen. 2022. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5459â5469.

[46] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. 2022. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5481â5490.

[47] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689 (2021).

[48] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13, 4 (2004), 600â612.

[49] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2024. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems 36 (2024).

[50] Tong Wu, Jiamu Sun, Yukun Lai, Yuewen Ma, Leif Kobbelt, and Lin Gao. 2024. DeferredGS: Decoupled and editable Gaussian splatting with deferred shading. arXiv preprint arXiv:2404.09412 (2024).

[51] Ziyi Yang, Yanzhen Chen, Xinyu Gao, Yazhen Yuan, Yu Wu, Xiaowei Zhou, and Xiaogang Jin. 2023. Sire-ir: Inverse rendering for brdf reconstruction with shadow and illumination removal in high-illuminance scenes. arXiv preprint arXiv:2310.13030 (2023).

[52] Ziyi Yang, Xinyu Gao, Yangtian Sun, Yihua Huang, Xiaoyang Lyu, Wen Zhou, Shaohui Jiao, Xiaojuan Qi, and Xiaogang Jin. 2024. Spec-gaussian: Anisotropic view-dependent appearance for 3d gaussian splatting. arXiv preprint arXiv:2402.15870 (2024).

[53] Yao Yao, Jingyang Zhang, Jingbo Liu, Yihang Qu, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. 2022. Neilf: Neural incident light field for physically-based material estimation. In European Conference on Computer Vision. Springer, 700â716.

[54] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume rendering of neural implicit surfaces. Advances in Neural Information Processing Systems 34 (2021), 4805â4815.

[55] Keyang Ye, Qiming Hou, and Kun Zhou. 2024. 3d gaussian splatting with deferred reflection. In ACM SIGGRAPH 2024 Conference Papers. 1â10.

[56] Keyang Ye, Qiming Hou, and Kun Zhou. 2024. Progressive radiance distillation for inverse rendering with Gaussian splatting. arXiv preprint arXiv:2408.07595 (2024).

[57] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19447â19456.

[58] Zehao Yu, Torsten Sattler, and Andreas Geiger. 2024. Gaussian opacity fields: Efficient and compact surface reconstruction in unbounded scenes. arXiv preprint arXiv:2404.10772 (2024).

[59] Yu-Jie Yuan, Xinyang Han, Yue He, Fang-Lue Zhang, and Lin Gao. 2024. Munerf: Robust makeup transfer in neural radiance fields. IEEE Transactions on Visualization and Computer Graphics (2024).

[60] Dingxi Zhang, Yu-Jie Yuan, Zhuoxun Chen, Fang-Lue Zhang, Zhenliang He, Shiguang Shan, and Lin Gao. 2024. Stylizedgs: Controllable stylization for 3d gaussian splatting. arXiv preprint arXiv:2404.05220 (2024).

[61] Jingyang Zhang, Yao Yao, Shiwei Li, Jingbo Liu, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. 2023. Neilf++: Inter-reflectable light fields for geometry and material estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3601â3610.

[62] Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, and Noah Snavely. 2021. Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5453â5462.

[63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 586â595.

[64] Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul Debevec, William T Freeman, and Jonathan T Barron. 2021. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. ACM Transactions on Graphics (ToG) 40, 6 (2021), 1â18.

[65] Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, and Xiaowei Zhou. 2022. Modeling indirect illumination for inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18643â 18652.

[66] Zuo-Liang Zhu, Beibei Wang, and Jian Yang. 2024. Gs-ror: 3d gaussian splatting for reflective object relighting via sdf priors. arXiv preprint arXiv:2406.18544 (2024).

## A radiance transfer

## A.1 View-independent part

We will derive the calculation formula for radiance transfer starting from the rendering equation as follows:

$$
L ( \mathbf { o } ) = \int L ( \omega ) f ( \omega , \mathbf { o } ) \mathrm { m a x } ( 0 , \mathbf { n } \cdot \omega ) d \omega\tag{14}
$$

We transform the integration domain from the upper hemisphere to the entire sphere by constraining the cosine value. Assuming that spherical harmonics are used to reconstruct the incident radiance $L ( \omega )$ , the radiance transfer term $T ( \omega , \mathbf { o } )$ accounts for the remaining components. This transformation allows for a more comprehensive representation of the radiance transfer and incident light interaction. The rendering equation can be approximated in the following form:

$$
L ( \mathbf { o } ) \approx \sum _ { j = 0 } ^ { n } c _ { j } \int B _ { j } ( \omega ) T ( \omega , \mathbf { o } ) d \omega\tag{15}
$$

where $B _ { j } ( \omega )$ is the corresponding basis function. For the viewindependent part, the radiance transfer term is decomposed into the diffuse albedo $\pmb { \rho } _ { d }$ and $T ^ { \prime } ( i )$ , as follows:

$$
L _ { d } \approx \rho _ { d } \sum _ { j = 0 } ^ { n ^ { 2 } } c _ { j } \int B _ { j } ( \omega ) T ^ { \prime } ( \omega ) d \omega\tag{16}
$$

where the integral part can be considered as a projection of $T ^ { \prime } ( \omega )$ on the basis functions. Through projection, the corresponding spherical harmonic coefficients can be calculated as follows:

$$
c _ { j } ^ { t } = \int B _ { j } ( \omega ) T ^ { \prime } ( \omega ) d \omega\tag{17}
$$

In the case of using a finite order of spherical harmonic functions, the outgoing radiance can be approximated as:

$$
L _ { d } \approx \rho _ { d } \sum _ { j = 0 } ^ { n ^ { 2 } } c _ { j } c _ { j } ^ { t }\tag{18}
$$

By using basis functions as an intermediate representation, we can quickly approximate the complex integral through a point multiplication of two sets of coefficient vectors.

## A.2 View-dependent part

For the view-dependent part, similar to Equation (16), the radiance transfer term is simultaneously related to both the incident and outgoing directions.

$$
L _ { s } ( \mathbf { o } ) \approx \rho _ { s } \sum _ { j = 1 } ^ { n ^ { 2 } } c _ { j } \int B _ { j } ( \omega ) T ^ { \prime } ( \omega , \mathbf { o } ) d \omega\tag{19}
$$

Similar to Equation (17), we project the radiance transfer term onto the spherical harmonics and obtain the corresponding coefficients as Eqution (20).

$$
c _ { j } ^ { t } ( \mathbf { 0 } ) = \int B _ { j } ( \omega ) T ^ { \prime } ( \omega , \mathbf { 0 } ) d \omega\tag{20}
$$

However, the coefficients in this case are related to the outgoing direction and can not be used directly as follows:

$$
L _ { s } ( \mathbf { o } ) \approx \rho _ { s } \sum _ { j = 1 } ^ { n ^ { 2 } } c _ { j } c _ { j } ^ { t } ( \mathbf { o } )\tag{21}
$$

Therefore, we continue to project $c _ { j } ^ { t } ( \mathbf { o } )$ onto the spherical harmonics $B _ { k } ( \mathbf { o } )$ according to the outgoing direction, obtaining the matrix $c _ { j k } ^ { t } ,$ , which corresponds to $B _ { j } ( \omega )$ and $B _ { k } ( \mathbf { o } )$ respectively as follows:

$$
c _ { j k } ^ { t } = \int B _ { k } ( \mathbf { o } ) c _ { j } ^ { t } ( \mathbf { o } ) d \mathbf { o }\tag{22}
$$

$\mathtt { B y }$ using the coefficients $c _ { j k } ^ { t }$ and the corresponding basis functions $B _ { k } ( \mathbf { o } )$ , it is also possible to approximate $c _ { j } ^ { t } ( \mathbf { o } )$ as shown below.

$$
c _ { j } ^ { t } ( \mathbf { 0 } ) \approx \sum _ { k = 1 } ^ { n ^ { 2 } } B _ { k } ( \mathbf { 0 } ) c _ { j k } ^ { t }\tag{23}
$$

Substituting Equation 23 into Equation 21, we can obtain an approximation of the outgoint radiance as follows:

$$
L _ { s } ( \mathbf { o } ) \approx \pmb { \rho _ { s } } \sum _ { j = 1 } ^ { n ^ { 2 } } \sum _ { k = 1 } ^ { n ^ { 2 } } c _ { j } c _ { j k } ^ { t } B _ { k } ( \mathbf { o } )\tag{24}
$$

However, computing the radiance transfer matrix, which stores the coefficients $c _ { j k } ^ { t } ,$ , is required for modeling view-dependent effects. For ??-th order spherical harmonics (SH) lighting, each transfer matrix requires $n ^ { 2 }$ parameters. Consequently, with increasing numbers of Gaussians, the storage cost grows rapidly and becomes impractical. To address this, our final computation adopts the formulation in Equ. 21, where $c _ { j } ^ { t } ( \mathbf { o } )$ is dynamically decoded from the radiance transfer features $f _ { t }$ and the reflection direction o by a lightweight MLP ?? as follows:

$$
L _ { s } ( \mathbf { 0 } ) \approx \rho _ { s } \sum _ { j = 0 } ^ { n ^ { 2 } } c _ { j } c _ { j } ^ { t } ( \mathbf { 0 } ) , \quad w i t h \quad c _ { j } ^ { t } ( \mathbf { 0 } ) = G ( f _ { t } , \mathbf { 0 } )\tag{25}
$$

This design significantly alleviates the storage overhead while maintaining flexibility in modeling view-dependent appearance.

## B BRDF MODEL

We adopt the microfacet specular shading model according to:

$$
f ( \mathbf { i } , \mathbf { o } ) = { \frac { D F G } { 4 ( \mathbf { n } \cdot \mathbf { o } ) ( \mathbf { n } \cdot \mathbf { i } ) } }\tag{26}
$$

where $D , F , G$ correspond to normal distribution function, fresnel term, and geometry term. Their specific expressions are as follows:

$$
D ( \mathbf { n } , \mathbf { h } , a ) = { \frac { a ^ { 2 } } { \pi ( ( \mathbf { n } \cdot \mathbf { h } ) ^ { 2 } ( a ^ { 2 } - 1 ) + 1 ) ^ { 2 } } }\tag{27}
$$

$$
F = F _ { 0 } + \left( 1 - F _ { 0 } \right) \left( 1 - \left( \mathbf { h } \cdot \mathbf { o } \right) \right) ^ { 5 }\tag{28}
$$

$$
{ \displaystyle G ( { \bf n } , { \bf o } , { \bf i } , k ) = G _ { s u b } ( { \bf n } , { \bf o } , k ) \cdot G _ { s u b } ( { \bf n } , { \bf i } , k ) }\tag{29}
$$

$$
G _ { s u b } ( \mathbf { n } , \mathbf { v } , k ) = { \frac { \mathbf { n } \cdot \mathbf { v } } { ( \mathbf { n } \cdot \mathbf { v } ) ( 1 - k ) + k } }\tag{30}
$$

where n is normal, h is half-way vector. Roughness ?? determines ?? and $k ,$ where $a = r ^ { 2 }$ and $\begin{array} { r } { k = \frac { r ^ { 4 } } { 2 } . F _ { 0 } } \end{array}$ in ?? is the basic reflection ratio, calculated by metallic ?? and albedo c as follows:

$$
F _ { 0 } = ( 1 - m ) * 0 . 0 4 + m * { \bf c }\tag{31}
$$

## C IMPLEMENTATION DETAILS

We conducted comprehensive experiments using an NVIDIA RTX 4090 GPU and used the Adam optimizer for all parameter updates. For object level data, Our hybrid rendering branch rendering speed is 96.4 and PBR branch is 130.9 in FPS, exhibiting the real-time rendering capability of our proposed inverse rendering method.

The model is first trained for 30,000 iterations using only the hybrid rendering branch. The view-independent components of radiance transfer are initialized at the beginning of training. After 3,000 iterations, the view-dependent components are activated. During initialization, the reflection intensity is set to 0.01 for all Gaussians, and the radiance transfer order is set to 3. The MLP ?? consists of three layers with 64 units each. It takes the reflection direction as input and concatenates transfer features at the second layer. The first two layers use the ReLU activation function. Additionally, a ReLU operation is applied after the dot product with the spherical harmonic lighting. After training for 30,000 iterations, visibility information is baked into voxel grids with a resolution of 1283. Subsequently, both the hybrid rendering and physicallybased rendering branches are jointly supervised for another 10,000 iterations. During this stage, the geometry is fine-tuned, and the appearance is decomposed into material and lighting components. Both the reflection map and environment map are configured with a resolution of 6 Ã 128 Ã 128, which balances computational efficiency and rendering quality.

In addition to the optimization mentioned in the main text, we also include the following commonly used loss terms:

Bilateral Smoothness. We believe that normal n, reflection intensity $R _ { i : }$ , reflection roughness $R _ { r } ,$ , metallic $m ,$ and roughness ?? will not change drastically in color-smooth regions. We define a smooth constraint as:

$$
\mathcal { L } _ { s , f } = | | \nabla f | | e x p ( - | | \nabla C _ { g t } | | )\tag{32}
$$

where $f$ represents the screen-space buffer of above attributes. For each term, the corresponding $\lambda _ { f } = 0 . 0 1$

$$
\mathcal { L } _ { s } = \sum \lambda _ { f } \mathcal { L } _ { s , f }\tag{33}
$$

Object Mask Constraint. If there is a mask indicating the object, we can constrain the optimization by a binary cross-entropy loss:

$$
\mathcal { L } _ { o } = - M \log O - \left( 1 - M \right) \log ( 1 - O )\tag{34}
$$

where ?? is the mask of the object and $\begin{array} { r } { O = \sum _ { i } ^ { N } T _ { i } \alpha _ { i } } \end{array}$ , and the corresponding $\lambda _ { o } = 0 . { \overset { . } { } . }$ 1

## D MORE COMPARISONS

We provide additional results for relighting and novel view synthesis to enable a comprehensive comparison. Notably, the ball from the Shiny Blender dataset does not include ground truth (GT) relighting data. However, we showcase our results of the ball to highlight the performance advantages of our method.

We present quantitative evaluations of albedo decomposition using PSNR, SSIM, and LPIPS on both the Shiny Blender dataset and the TensoIR dataset, as shown in Table 6.

Table 6: Albedo decomposition quality comparison.
<table><tr><td colspan="2"></td><td>TensoIR</td><td>GS-IR</td><td>R3DG</td><td>Ours</td></tr><tr><td rowspan="2">PSNRâ</td><td>TensoIR</td><td>29.19</td><td>32.04</td><td>28.27</td><td>31.97</td></tr><tr><td>Shiny Blender</td><td>22.17</td><td>20.97</td><td>20.69</td><td>24.47</td></tr><tr><td rowspan="2">SSIMâ</td><td>TensoIR</td><td>0.952</td><td>0.920</td><td>0.918</td><td>0.939</td></tr><tr><td>Shiny Blender</td><td>0.877</td><td>0.859</td><td>0.871</td><td>0.913</td></tr><tr><td rowspan="2">LPIPSâ</td><td>TensoIR</td><td>0.080</td><td>0.092</td><td>0.070</td><td>0.052</td></tr><tr><td>Shiny Blender</td><td>0.184</td><td>0.160</td><td>0.141</td><td>0.085</td></tr></table>

<!-- image-->

Figure 11: Visual comparison of using radiance transfer and spherical harmonics  
<!-- image-->  
Figure 12: Qualitative comparisons on real scenes.

Figure 11 visualizes the radiance transfer component of our model. This component effectively enhances surface normals. Our radiance transfer maintains low-frequency characteristics better than spherical harmonics, preventing the appearance of floating artifacts and resulting in smoother surfaces with more accurate normals.

Figure 12 shows the results on the Ref-Real dataset, where our method achieves high-quality performance even on real-world data without requiring masks. Additionally, we performed relighting tests on the kitchen and garden scenes from the Mip-NeRF 360 dataset, as shown in Figure 13.

Figure 15 compares the novel view synthesis results of our method (RTR-GS) with those of other approaches. Enlarged views of local regions are included to emphasize details. Our method demonstrates superior reflection clarity and captures finer details compared to others.

Figure 16 shows a comparison of relighting results between our approach and other inverse rendering methods. For both diffuse and specular objects, our method produces more accurate and realistic relighting outcomes. Specifically, reflective surfaces exhibit precise and detailed reconstructions of reflections. Additionally, as shown in Table 7, we provide quantitative relighting results for selected objects from the datasets used.

GT  
Rendering  
Relight1  
Relight2  
<!-- image-->

Figure 13: Relighting results on real scene dataset.  
<!-- image-->  
Figure 14: Material editing results.

Table 7: Relighting quality of some objects in terms of PSNRâ and SSIMâ on the TensoIR dataset (top 4 rows), Shiny Blender dataset (middle 5 rows), and Stanford ORB dataset (bottom 5 rows).
<table><tr><td></td><td>TensoIR</td><td>R3DG</td><td>GS-IR</td><td>Ours</td></tr><tr><td>hotdog</td><td>27.87/.932</td><td>24.40/.922</td><td>26.86/.921</td><td>28.91/.943</td></tr><tr><td>ficus</td><td>24.30/.946</td><td>28.74/.941</td><td>23.08/.872</td><td>31.05/.953</td></tr><tr><td>lego</td><td>27.57/.924</td><td>28.23/.924</td><td>21.27/.854</td><td>25.68/.914</td></tr><tr><td>armadillo</td><td>34.46/.975</td><td>32.70/.951</td><td>32.73/.941</td><td>34.76/.967</td></tr><tr><td>car</td><td>26.15/.913</td><td>22.37/.881</td><td>23.58/.872</td><td>28.74/.947</td></tr><tr><td>coffee</td><td>18.37/.845</td><td>16.76/.865</td><td>19.95/.877</td><td>19.98/.909</td></tr><tr><td>helmet</td><td>17.28/.791</td><td>19.31/.848</td><td>17.99/.797</td><td>24.44/.918</td></tr><tr><td>teapot</td><td>33.73/.979</td><td>27.44/.976</td><td>29.59/.969</td><td>35.68/.989</td></tr><tr><td>toaster</td><td>15.95/.682</td><td>17.54/.774</td><td>14.80/.715</td><td>21.97/.879</td></tr><tr><td>baking_001</td><td>26.53/.961</td><td>26.70/.969</td><td>25.95/.965</td><td>25.71/.969</td></tr><tr><td>car_002</td><td>26.65/.964</td><td>28.95/.963</td><td>29.54/.965</td><td>29.69/.975</td></tr><tr><td>chips_002</td><td>28.65/.947</td><td>32.32/.969</td><td>33.46/.973</td><td>33.71/.974</td></tr><tr><td>grogu_001</td><td>25.73/.959</td><td>27.37/.968</td><td>28.55/.970</td><td>27.39/.972</td></tr></table>

Figure 17 presents the results generated by our method under five different lighting conditions. The outputs are consistent and accurate for both diffuse and specular objects, demonstrating the robustness of our approach. Furthermore, as shown in Figure 14, our method also produces reliable results after material editing.

GT

Ours

R3DG

GS-IR

3DGS-DR

GShader

TensoIR

<!-- image-->

Figure 15: Qualitative comparisons on synthetic scenes.

GT

Ours

R3DG

GShader

<!-- image-->

GS-IR

<!-- image-->

<!-- image-->

TensoIR

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

Figure 16: Qualitative comparisons of relighting with different environment lighting.

Hotdog

Lego

Armadillo

Ficus

Ball

Toaster

Car

Toy Car

<!-- image-->

Figure 17: Relighting results of our method on synthetic dataset. Our method can also provide high-quality relighting results for diffuse objects and specular objects.