# SSR-GS: Separating Specular Reflection in Gaussian Splatting for Glossy Surface Reconstruction

Ningjing Fan1 and Yiqun Wang1â

Chongqing University, Choingqing, China

Abstract. In recent years, 3D Gaussian splatting (3DGS) has achieved remarkable progress in novel view synthesis. However, accurately reconstructing glossy surfaces under complex illumination remains challenging, particularly in scenes with strong specular reflections and multi-surface interreflections. To address this issue, we propose SSR-GS, a specular reflection modeling framework for glossy surface reconstruction. Specifically, we introduce a prefiltered Mip-Cubemap to model direct specular reflections efficiently, and propose an IndiASG module to capture indirect specular reflections. Furthermore, we design Visual Geometry Priors (VGP) that couple a reflection-aware visual prior via a reflection score (RS) to downweight the photometric loss contribution of reflectiondominated regions, with geometry priors derived from VGGT, including progressively decayed depth supervision and transformed normal constraints. Extensive experiments on both synthetic and real-world datasets demonstrate that SSR-GS achieves state-of-the-art performance in glossy surface reconstruction.

Keywords: Gaussian Splatting Â· Specular Reflection Â· Surface Reconstruction

## 1 Introduction

Accurate surface reconstruction from multi-view images remains a long-standing challenge in computer vision and graphics, with applications spanning animation, robotics, AR/VR, and autonomous driving. Neural Radiance Fields (NeRF) [24] have recently achieved impressive fidelity [8, 22, 28], but their dense neural representation incurs high computational costs and long training times. To address these limitations, 3D Gaussian Splatting (3DGS) [18] models scenes using explicit 3D Gaussian primitives, enabling real-time rendering with high-quality view synthesis, thus offering a practical alternative for diverse applications.

Although 3DGS enables real-time, high-quality novel view synthesis, it is primarily rendering-oriented and suffers from limited surface reconstruction accuracy due to the unstructured Gaussians and reliance on image reconstruction loss, which can lead to geometric deviation. Recent works have tried to improve geometric fidelity, such as SuGaR [11], 2DGS [12], GOF [40], and PGSR [2];

<!-- image-->  
GT CD

<!-- image-->  
Ours 0.067

<!-- image-->  
Ref-Gaussian 0.260

<!-- image-->  
Ref-GS 0.072

<!-- image-->  
PGSR 0.124

<!-- image-->  
GOF 0.109  
Fig. 1: Surface reconstruction results on toaster. CD denotes the Chamfer distance.

and to enhance specular modeling, e.g., Spec-Gaussian [36], 3DGS-DR [19], Ref-GS [42], and Ref-Gaussian [37]. However, existing methods still struggle to faithfully reconstruct glossy surfaces with strong specular reflections (Fig. 1). In such cases, reflected radiance is often imperfectly separated from the diffuse component, which can cause light leakage and ultimately lead to geometric artifacts such as surface collapse in highly reflective regions.

In this work, we propose SSR-GS, a framework for separating specular reflection in Gaussian splatting for high-fidelity glossy surface reconstruction. We decouple diffuse and specular components: the diffuse term is integrated via volumetric compositing along the ray, while the specular term is factorized into a material component (via volumetric blending) and incident illumination estimated from physically based surface rendering. Specular reflection is further decomposed into direct and indirect components: direct reflection is computed using a Mip-Cubemap with view-consistent environment sampling, and indirect reflection is modeled by the proposed IndiASG to capture complex multi-bounce effects while remaining separated from the diffuse term. To enhance geometric fidelity and cross-view consistency, we incorporate Visual Geometry Priors (VGP), coupling a visual prior (VP) with geometry priors (GP). The VP, implemented via the reflection score (RS), suppresses reflection-dominated regions, while the GP applies VGGT-inferred depth and transformed normal constraints to regularize geometry, enabling more stable optimization and higher-quality reconstruction under complex reflections.

Our main contributions are:

Mip-Cubemap environment representation: We propose a Mip-Cubemap environment representation for modeling direct specular reflections, enabling multi-scale environment sampling and more accurate roughness-aware reflection rendering.

â IndiASG for indirect specular reflection modeling: We identify that inadequate modeling of indirect specular reflections can destabilize geometry estimation. Based on this insight, we propose IndiASG, which explicitly models indirect specular reflections to improve geometric stability and enable Gaussians to better capture multi-view consistent geometry.

Visual Geometry Priors (VGP): We propose visual geometry priors that couple a visual prior reflection score (RS) with geometry priors inferred by VGGT, which work synergistically to constrain and stabilize geometry.

## 2 Related Works

## 2.1 Surface Reconstruction via Neural Rendering

NeRF [24] pioneered volumetric scene modeling in neural rendering, inspiring subsequent geometry-aware extensions such as signed distance fields (e.g., NeuS [31] and its variants [6, 8, 32, 33]), occupancy-based formulations [25], and reflective surface handling [5, 21, 22, 27].

With the advent of 3D Gaussian Splatting (3DGS) [18], explicit 3D Gaussian primitives have been explored for surface reconstruction. However, due to the discrete and unstructured nature of Gaussians, accurate surface extraction remains challenging. Methods such as SuGaR [11] introduce regularization terms to constrain Gaussians to surfaces, followed by Poisson reconstruction [17]. NeuSG [3], GSDF [39], and 3DGSR [23] optimize SDFs jointly with Gaussian models, but neural optimization is computationally expensive. 2DGS [12] employs 2D Gaussian primitives and TSDF fusion [13] for surface extraction. GOF [40] extracts geometry from Gaussian opacity fields without Poisson or TSDF fusion. PGSR [2] leverages unbiased depth rendering and integrates single- and multi-view geometry through regularization for accurate reconstruction. GausSurf [30] further incorporates multi-view constraints and normal priors [38] to improve quality while reducing computation. MILo [10] introduces a differentiable Gaussian Splatting framework that extracts meshes directly during training, bridging volumetric and surface representations for accurate and lightweight surface reconstruction. Despite these advances in surface extraction and geometric regularization, existing methods primarily focus on general scene reconstruction and do not explicitly address the challenges posed by glossy surfaces.

## 2.2 Modeling Glossy Reflections

GaussianShader [14] enhances 3DGS rendering of reflective surfaces through a simplified shading function with reliable normal estimation. 3iGS [26] improves the view-dependent specular quality by factorizing a continuous illumination field and optimizing per-Gaussian BRDF features. Relightable 3D Gaussians [7] enables physically based relighting by jointly optimizing per-point normals, BRDFs, and incident lighting through differentiable rendering. Spec-Gaussian [36] replaces SH-based colors with an anisotropic spherical Gaussian (ASG) appearance field, enabling improved modeling of high-frequency and anisotropic view-dependent specular effects. EnvGS [34] models complex viewdependent reflections by introducing environment Gaussian primitives jointly optimized with scene Gaussians, rendered efficiently via GPU ray tracing. Ref-GS [42] integrates deferred rendering and directional encoding, reducing viewdependent ambiguities and introducing a spherical Mip-Grid to capture surface roughness. Ref-Gaussian [37] enables real-time reconstruction of reflective objects with inter-reflection via physically based deferred rendering and Gaussiangrounded inter-reflection modeling. IRGS [9] models inter-reflections in inverse rendering via differentiable 2D Gaussian ray tracing and Monte Carlo optimization for indirect lighting estimation. GlossyGS [20] and MaterialRefGS [41] address geometryâmaterial ambiguity in reflective scenes through microfacet priors, multi-view consistent material inference, and environment modeling. GOGS [35] applies physics-based Gaussian surfels with geometric priors and refined specular modeling for high-fidelity glossy object reconstruction and relighting. Overall, these works progressively improve the modeling of view-dependent and reflective effects, ranging from enhanced shading functions to physically grounded inverse rendering and environment-aware illumination representations. However, accurately reconstructing geometry under strong specularities and complex lighting remains challenging, as reflected radiance can interfere with surface estimation and cause geometric artifacts. Our method mitigates this by improving the separation of specular reflections and stabilizing geometry optimization, preventing reflected structures from being baked into the reconstructed surface.

## 3 Preliminary

## 3.1 3D Gaussian Splatting (3DGS)

3D Gaussian splatting [18] models a scene using a collection of 3D Gaussian primitives. Each Gaussian is characterized by a center $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , a covariance $\pmb { \Sigma } \in$ R3Ã3, an opacity scalar $\alpha ,$ and view-dependent color coefficients c (represented via spherical harmonics). The spatial contribution of a Gaussian is defined as:

$$
\mathcal { G } ( \pmb { x } ) = \exp \left[ - \frac { 1 } { 2 } ( \pmb { x } - \pmb { \mu } ) ^ { T } \pmb { \Sigma } ^ { - 1 } ( \pmb { x } - \pmb { \mu } ) \right] .\tag{1}
$$

Rendering is performed via splatting and point-based Î±-blending. For a pixel, the accumulated color C is computed by traversing K depth-sorted Gaussians overlapping the ray:

$$
C = \sum _ { k = 1 } ^ { K } c _ { k } \alpha _ { k } \mathcal { G } _ { k } ^ { \prime } \prod _ { j = 1 } ^ { k - 1 } \big ( 1 - \alpha _ { j } \mathcal { G } _ { j } ^ { \prime } \big ) ,\tag{2}
$$

where $\mathcal { G } ^ { \prime }$ represents the projected 2D Gaussian on the image plane.

## 3.2 Physically Based Rendering

Physically based rendering (PBR) models light transport according to physical principles. The outgoing radiance is described by the rendering equation [15]:

$$
L _ { o } ( \mathbf { x } , \omega _ { o } ) = L _ { e } ( \mathbf { x } , \omega _ { o } ) + \int _ { \Omega } f _ { r } ( \mathbf { x } , \omega _ { i } , \omega _ { o } ) L _ { i } ( \mathbf { x } , \omega _ { i } ) ( \mathbf { n } \cdot \omega _ { i } ) d \omega _ { i } .\tag{3}
$$

We adopt a CookâTorrance microfacet BRDF [4] with a Disney-style energyconserving formulation [1], decomposing the BRDF into diffuse and specular components:

$$
f _ { \mathrm { d i f f } } ( \mathbf { x } , \omega _ { i } , \omega _ { o } ) = ( 1 - F ( \omega _ { i } , \mathbf { h } ) ) ( 1 - m ) \frac { c _ { \mathrm { a l b e d o } } } { \pi } ,\tag{4}
$$

$$
f _ { \mathrm { s p e c } } ( \mathbf { x } , \omega _ { i } , \omega _ { o } ) = \frac { D ( r , \mathbf { n } , \mathbf { h } ) F ( \omega _ { i } , \mathbf { h } ) G ( \omega _ { i } , \omega _ { o } , \mathbf { h } ) } { 4 ( \mathbf { n } \cdot \omega _ { i } ) ( \mathbf { n } \cdot \omega _ { o } ) } .\tag{5}
$$

where $F ( \omega _ { i } , \mathbf { h } )$ is the Fresnel term, for which the Schlick approximation is employed:

$$
F ( \omega _ { i } , \mathbf { h } ) \approx F _ { 0 } + ( 1 - F _ { 0 } ) ( 1 - \omega _ { i } \cdot \mathbf { h } ) ^ { 5 } ,\tag{6}
$$

where $F _ { 0 }$ denotes the Fresnel reflectance at normal incidence; m is the metalness; calbedo is the diffuse albedo color; $\begin{array} { r } { \mathbf { h } = \frac { \omega _ { i } + \omega _ { o } } { \Vert \omega _ { i } + \omega _ { o } \Vert } } \end{array}$ is the half-vector; r is the roughness; $D ( r , \mathbf { n } , \mathbf { h } )$ is the normal distribution function; $G ( \omega _ { i } , \omega _ { o } , \mathbf { h } )$ is the geometry attenuation term.

Specular reflection can be factorized into a material-dependent term $M _ { \mathrm { s p e c } }$ and a lighting-dependent term $I _ { \mathrm { s p e c } }$ [16]:

$$
L _ { o } ^ { \mathrm { s p e c } } ( \mathbf { x } , \omega _ { o } ) \approx \underbrace { \int _ { \Omega } \frac { F G } { 4 ( \mathbf { n } \cdot \omega _ { o } ) } d \omega _ { i } } _ { M _ { \mathrm { s p e c } } } \cdot \underbrace { \int _ { \Omega } L _ { i } ( \omega _ { i } ) D ( \omega _ { i } ) d \omega _ { i } } _ { I _ { \mathrm { s p e c } } } .\tag{7}
$$

## 4 Method

Our framework builds upon a physically based rendering formulation with explicit specular modeling. We first present the overall framework in Sec. 4.1. Direct and indirect specular reflections are modeled using a Mip-Cubemap (Sec. 4.2) and IndiASG (Sec. 4.3), respectively. To enhance geometric stability in reflective regions, we incorporate Visual Geometry Priors (VGP) (Sec. 4.4). The overall pipeline is shown in Fig. 2.

## 4.1 Overview

Each Gaussian is endowed with physically meaningful material parameters: the Fresnel reflectance at normal incidence $f _ { 0 } \in \mathbb { R } ^ { 3 }$ and a diffuse component $c _ { \mathrm { d i f f } } \in \mathbb { R } ^ { 3 }$ Through volumetric rendering (Eq. 2), we obtain pixel-wise $F _ { 0 }$ and $C _ { \mathrm { d i f f } }$ , where $C _ { \mathrm { d i f f } }$ corresponds to the diffuse BRDF term $( 1 - m ) { \frac { c _ { \mathrm { a l b e d o } } } { \pi } }$ in Eq. 4. The Fresnel term $F$ is computed via Eq. 6, and the final radiance $L _ { r g b }$ is given by:

$$
\begin{array} { r l } & { L _ { r g b } = L _ { d i f f } + L _ { s p e c } , } \\ & { L _ { d i f f } = \left( 1 - F \right) C _ { \mathrm { d i f f } } , \quad L _ { s p e c } = M _ { \mathrm { s p e c } } \left( \left( 1 - w _ { \mathrm { v i s } } \right) L _ { \mathrm { s p e c } } ^ { \mathrm { d i r e c t } } + w _ { \mathrm { v i s } } L _ { \mathrm { s p e c } } ^ { \mathrm { i n d i } } \right) , } \end{array}\tag{8}
$$

where $L _ { \mathrm { s p e c } } ^ { \mathrm { d i r e c t } }$ and $L _ { \mathrm { s p e c } } ^ { \mathrm { i n d i } }$ denote the direct and indirect specular radiance terms described in Sec. 4.2 and Sec. 4.3, respectively. $w _ { \mathrm { v i s } }$ denotes the visibility weight obtained via ray tracing on the mesh reconstructed, which blends direct and indirect specular contributions. $M _ { \mathrm { s p e c } }$ is a specular material-dependent term computed via Eq. 7.

<!-- image-->  
Fig. 2: An overview of our SSR-GS pipeline. We rasterize 3D Gaussians to obtain surface buffers including normals, depth, opacity, diffuse component $C _ { \mathrm { d i f f } }$ , roughness, $F _ { 0 } ,$ , and alpha, and supervise them with geometry priors (GP). After rasterization, we extract a mesh via TSDF fusion and perform mesh-based ray tracing to estimate visibility $w _ { \mathrm { v i s } }$ . Direct specular reflection is queried from a Mip-Cubemap environment map, while indirect specular reflection is modeled by IndiASG. Finally, we apply physically based deferred rendering $( \mathrm { E q . ~ } 8 )$ to produce the rendered image. We additionally compute the visual prior (VP) (reflection score) and use it to down-weight reflectiondominated pixels in the photometric loss for stable geometry initialization.

## 4.2 Mip-Cubemap for Direct Specular Reflection

We approximate direct specular reflection using a roughness-aware environment map query, avoiding the costly hemispherical integration in the rendering equation. Specifically, we introduce a physically motivated normal distribution function (NDF) prefilter approximation, which reduces the original integral to a single Mip-Cubemap lookup. We query the direct specular reflection from the environment map using a roughness-aware mipmap hierarchy $\{ \ell _ { 0 } , \ldots , \ell _ { L _ { \mathrm { m a x } } - 1 } \}$ where $L _ { \mathrm { m a x } }$ denotes the total number of mip levels in the environment map. Higher mip levels correspond to progressively blurrier representations, modeling broader specular lobes induced by surface roughness. The mip level â is determined by the surface roughness r as

$$
\ell = r ^ { 2 } \cdot ( L _ { \mathrm { m a x } } - 1 ) .\tag{9}
$$

Direct specular reflection is then obtained via trilinear sampling of the environment map over direction and mip level:

$$
L _ { \mathrm { s p e c } } ^ { \mathrm { d i r e c t } } = E ^ { \mathrm { t r i l i n e a r } } ( \omega _ { r } , \ell )\tag{10}
$$

where $E$ denotes the environment map and Etrilinear $( \omega _ { r } , \ell )$ queries it at reflection direction $\omega _ { r }$ and mip level â (bilinear filtering within each level and linear interpolation across adjacent levels). Mipmap prefiltering implicitly accounts for roughness-dependent lobe shaping, avoiding explicit angular integration.

Our Mip-Cubemap differs from the cubemap in 3DGS-DR [19] and the ${ \mathrm { S p h } } { \mathrm { - M i p } }$ in Ref-GS [42] in both representation and filtering. 3DGS-DR uses a standard cubemap without roughness-aware mip selection, whereas we build a mipmap hierarchy and choose mip levels based on roughness $\left( \operatorname { E q . 9 } \right)$ for prefiltered specular sampling. Compared with the spherical parameterization in Ref-GS, our cubemap avoids projection distortion and naturally supports mip-based filtering. Ablations in Sec. 6.3 validate this design.

## 4.3 IndiASG for Indirect Specular Reflection

To address the challenge of inaccurate geometry caused by multi-surface indirect illumination, we propose IndiASG, a compact, learning-based local light field representation for indirect specular reflection. IndiASG models the reflection using a fixed set of anisotropic spherical Gaussian lobes, with a neural predictor $F _ { \Theta }$ estimating the per-lobe radiometric parameters from a surface point and a reflection direction. This design enables physically consistent indirect specular modeling while maintaining accurate surface reconstruction.

At each surface point, indirect specular reflection is modeled as the sum of $N _ { \mathrm { l o b e } } = 3 3$ anisotropic spherical Gaussian lobes over the upper hemisphere, i.e., the indirect specular illumination is represented as a superposition of these lobes. Formally, each lobe $j$ is defined by $( \mathbf { a } _ { j } , \lambda _ { j } , \mu _ { j } , \omega _ { j } , \omega _ { j } ^ { \lambda } , \omega _ { j } ^ { \mu } )$ , where $\mathbf { a } _ { j } \in \mathbb { R } ^ { 3 }$ denotes RGB amplitude, $\lambda _ { j }$ and $\mu _ { j }$ control Gaussian sharpness along $\omega _ { j } ^ { \lambda }$ and $\omega _ { j } ^ { \mu }$ , respectively, and $( \omega _ { j } , \omega _ { j } ^ { \lambda } , \omega _ { j } ^ { \mu } )$ are precomputed unit vectors forming an orthonormal frame in tangent space. The set of $\omega _ { j }$ follows a fixed hierarchical layout: one lobe at the zenith $\omega _ { 0 } = ( 0 , 0 , 1 )$ , and four concentric rings at polar angles $\begin{array} { r } { \theta _ { \ell } = \frac { \ell \pi } { 8 } , \quad \ell = 1 , \dots , 4 } \end{array}$ , each containing eight evenly spaced azimuthal directions $\begin{array} { r } { \phi _ { k } = \frac { 2 \pi k } { 8 } , \quad k = 0 , \ldots , 7 } \end{array}$ (Fig. 2), with $( \omega _ { j } ^ { \lambda } , \omega _ { j } ^ { \mu } )$ obtained via orthonormal completion of $\omega _ { j }$ in tangent space.

Given the predicted radiometric parameters $( \mathbf { a } _ { j } , \lambda _ { j } , \mu _ { j } )$ and the precomputed geometric frame $( \omega _ { j } , \omega _ { j } ^ { \lambda } , \omega _ { j } ^ { \mu } )$ , the indirect specular reflection in a reflection direction $\omega _ { r }$ is:

$$
L _ { \mathrm { s p e c } } ^ { \mathrm { i n d i } } ( \omega _ { r } ) = \sum _ { j = 1 } ^ { N _ { \mathrm { l o b e } } } \mathbf { a } _ { j } \ \operatorname* { m a x } ( 0 , \omega _ { r } \cdot \omega _ { j } ) \exp \bigl ( - \lambda _ { j } ( \omega _ { r } \cdot \omega _ { j } ^ { \lambda } ) ^ { 2 } - \mu _ { j } ( \omega _ { r } \cdot \omega _ { j } ^ { \mu } ) ^ { 2 } \bigr ) ,\tag{11}
$$

where max $( 0 , \omega _ { r } \cdot \omega _ { j } )$ enforces a hemisphere constraint. The exponential term models anisotropic angular falloff in the local tangent frame defined by $\omega _ { j } ^ { \lambda }$ and $\omega _ { j } ^ { \mu }$ , with $\lambda _ { j }$ and $\mu _ { j }$ controlling the sharpness along the two principal axes.

We predict the per-lobe parameters via a learned mapping

$$
F _ { \theta } : ( { \bf p } , \omega _ { r } , r , C _ { r e s } ) \ \mapsto \ ( { \bf a } _ { j } , \lambda _ { j } , \mu _ { j } ) ,\tag{12}
$$

where p and $\omega _ { r }$ are encoded with multi-frequency positional encoding (PE) and integrated directional encoding (IDE) [27], respectively; r denotes the surface

roughness; and $C _ { r e s }$ denotes the residual specular signal. The indirect specular radiance $L _ { \mathrm { s p e c } } ^ { \mathrm { i n d i } }$ is then evaluated via Eq. (11).

## 4.4 Visual Geometry Priors (VGP)

Visual Prior (VP): Reflection Score (RS) We take inspiration from Ref-NeuS [8] by employing a reflection score (RS) to identify regions with high multiview variance. By attenuating the photometric optimization gradients in specular reflection regions, we reduce the adverse influence of view-dependent appearance variations on geometry updates, thereby ensuring stable surface reconstruction.

Multi-view Visibility and Occlusion Given a reference image $I _ { \mathrm { r e f } }$ and its rendered depth map $D _ { \mathrm { r e f } }$ , we back-project each pixel u to a 3D surface point p in world coordinates. Let $\mathbf { r } _ { d } ( \mathbf { u } )$ denote the ray direction in camera coordinates. The inverse projection $\pi ^ { - 1 }$ is formulated as:

$$
\mathbf { p } = \pi ^ { - 1 } ( \mathbf { u } , D _ { \mathrm { r e f } } ( \mathbf { u } ) ) = \mathbf { R } ^ { \top } \left( D _ { \mathrm { r e f } } ( \mathbf { u } ) \cdot \mathbf { r } _ { d } ( \mathbf { u } ) - \mathbf { t } \right) ,\tag{13}
$$

where R and t denote the rotation and translation of the world-to-camera extrinsic matrix, respectively.

For a source view $m \in \mathcal { M }$ with camera center $\mathbf { o } _ { m } ,$ projection $\pi _ { m } .$ and depth map $D _ { m }$ , we define a visibility indicator $\nVdash _ { m } ( \mathbf { p } )$ . The point p is visible in view m if:

1. Field of View: The projected coordinate $\mathbf { u } _ { m } = \pi _ { m } ( \mathbf { p } )$ lies within the image boundaries.

2. Depth-based Occlusion: The point p should not be occluded in view m. We perform a depth agreement check by comparing the point-to-camera distance with the depth map at its projected pixel:

$$
\big | \big | \mathbf { p } - \mathbf { o } _ { m } \big | \big | _ { 2 } - D _ { m } \big ( \mathbf { u } _ { m } \big ) \big | < \tau _ { \mathrm { o c c } } ,\tag{14}
$$

where $\tau _ { \mathrm { o c c } }$ is a depth tolerance threshold (set to 0.15 in our experiments).

3. View Sufficiency: The point p must be visible in at least K source views to calculate a reliable statistic (we use $K = 5 )$ ).

Reflection Score Formulation We define the reflection score $\mathrm { R S } ( \mathbf { u } )$ as the mean absolute photometric deviation between the reference view and valid source views. Let $\mathcal { V } ( \mathbf { u } ) = \{ m \in \mathcal { M } \mid \mathcal { k } _ { m } ( \mathbf { p } ) = 1 \}$ be the set of valid views:

$$
\mathrm { R S } ( \mathbf { u } ) = \frac { 1 } { | \mathcal { V } ( \mathbf { u } ) | } \sum _ { m \in \mathcal { V } ( \mathbf { u } ) } \left\| I _ { m } ( \mathbf { u } _ { m } ) - I _ { \mathrm { r e f } } ( \mathbf { u } ) \right\| _ { 1 } .\tag{15}
$$

A higher reflection score indicates stronger view-dependent appearance changes, which are often caused by specular reflections and other non-Lambertian effects.

We use $I _ { \mathrm { r e f } } ( \mathbf { u } )$ as the anchor (instead of the mean over visible views) since RS later modulates the reference-view photometric loss (Eq. 16). This aligns RS with the reference view and suppresses the influence of view-dependent specularities on geometry reconstruction.

Reflection-aware Photometric Loss To reduce the adverse impact of specular reflections during early optimization, we incorporate RS into the training objective in Stage 1. The RS-weighted photometric loss is defined as:

$$
\mathcal { L } _ { \mathrm { r g b } } = \frac { 1 } { | \varOmega | } \sum _ { { \bf u } \in \varOmega } \frac { | | \hat { C } ( { \bf u } ) - C _ { \mathrm { g t } } ( { \bf u } ) | | _ { 1 } } { \mathrm { R S } ( { \bf u } ) } ,\tag{16}
$$

where $\hat { C }$ and $C _ { \mathrm { g t } }$ denote the rendered and ground-truth colors. Pixels with large reflection scores (i.e., high multi-view variance) are down-weighted, reducing the influence of view-dependent specular reflections on geometry updates.

Geometry Priors (GP) From VGGT [29], we obtain a prior depth map $D _ { \mathrm { V G G T } }$ and an associated confidence map $C _ { \mathrm { V G G T } }$ . We define a VGGT prior regularization that combines a depth consistency loss and a normal consistency loss, with per-pixel weights derived from CVGGT.

The depth term aligns the predicted depth map $D$ to the VGGT prior DVGGT:

$$
\mathcal { L } _ { \mathrm { V G G T - D } } = \operatorname* { m i n } _ { \omega , b } \left. \omega D + b - D _ { \mathrm { V G G T } } \right. _ { 2 } ^ { 2 } .\tag{17}
$$

where the scalars $\omega$ and b are estimated by least squares because the VGGT depth prior is not an absolute metric depth and differs by an unknown global scale and shift.

NVGGT are estimated from $D _ { \mathrm { V G G T } }$ by back-projecting pixels into 3D and computing the normalized cross product of vertical and horizontal offsets:

$$
N _ { \mathrm { V G G T } } = \frac { \left( P _ { u } - P _ { d } \right) \times \left( P _ { r } - P _ { l } \right) } { \left\| \left( P _ { u } - P _ { d } \right) \times \left( P _ { r } - P _ { l } \right) \right\| _ { 2 } } ,\tag{18}
$$

where $P _ { u } , P _ { d } , P _ { l } , P _ { r }$ are the back-projected 3D point maps corresponding to neighboring pixels (up, down, left, and right), respectively.

The normal term combines an $\ell _ { 1 }$ loss and an angular consistency loss between the predicted normal map N and the VGGT-derived normal map NVGGT:

$$
\mathcal { L } _ { \mathrm { V G G T - N } } = \lVert N - N _ { \mathrm { V G G T } } \rVert _ { 1 } + \lambda \left( 1 - \frac { N \cdot N _ { \mathrm { V G G T } } } { \lVert N \rVert _ { 2 } \lVert N _ { \mathrm { V G G T } } \rVert _ { 2 } } \right) ,\tag{19}
$$

where $\lambda$ is the weighting factor of the angular consistency loss, and is set to 0.5 in our experiments.

Per-pixel weights are obtained via a logarithmic transform followed by minâ max normalization:

$$
W _ { \mathrm { V G G T } } = \mathrm { N o r m } ( \log ( 1 + C _ { \mathrm { V G G T } } ) ) .\tag{20}
$$

The final VGGT prior loss is defined as a confidence-weighted combination of the depth and normal terms:

$$
\mathcal { L } _ { \mathrm { V G G T } } = W _ { \mathrm { V G G T } } \cdot ( \mathcal { L } _ { \mathrm { V G G T - D } } + \mathcal { L } _ { \mathrm { V G G T - N } } ) .\tag{21}
$$

## 5 Training

<table><tr><td></td><td>ball</td><td>car</td><td>coffee</td><td>helmet</td><td>teapot</td><td>toaster</td><td>mean</td></tr><tr><td>Ref-NeRF [27]</td><td>1.55</td><td>14.93</td><td>12.24</td><td>29.48</td><td>9.23</td><td>42.87</td><td>18.38</td></tr><tr><td>ENVIDR [21]</td><td>0.74</td><td>7.10</td><td>9.23</td><td>1.66</td><td>2.47</td><td>6.45</td><td>4.61</td></tr><tr><td>UniSDF [28]</td><td>0.45</td><td>6.88</td><td>8.00</td><td>1.72</td><td>2.80</td><td>6.45</td><td>8.71</td></tr><tr><td>PGSR [2] MILo [10]</td><td>66.93</td><td>4.62</td><td>2.91</td><td>6.01</td><td>1.01</td><td>15.31</td><td>16.13</td></tr><tr><td></td><td>61.30</td><td>4.47</td><td>2.36</td><td>4.87</td><td>0.62</td><td>16.49</td><td>15.02</td></tr><tr><td>GaussianShader [14]</td><td>7.03</td><td>14.05</td><td>14.93</td><td>9.33</td><td>7.17</td><td>13.08</td><td>10.93</td></tr><tr><td>3DGS-DR [19]</td><td>0.85</td><td>2.32</td><td>2.21</td><td>1.67</td><td>0.53</td><td>6.99</td><td>2.43</td></tr><tr><td>Ref-Gaussian [37]</td><td>0.71</td><td>1.91</td><td>2.34</td><td>1.85</td><td>0.48</td><td>5.70</td><td>2.17</td></tr><tr><td>Ref-GS [42]</td><td>1.05</td><td>2.02</td><td>3.61</td><td>1.99</td><td>0.69</td><td>3.92</td><td>2.21</td></tr><tr><td>GS-ROR2 [44]</td><td>0.47</td><td>2.06</td><td>5.47</td><td>1.82</td><td>0.52</td><td>5.52</td><td>2.64</td></tr><tr><td>RTR-GS [43]</td><td>22.44</td><td>1.86</td><td>2.87</td><td>1.44</td><td>0.61</td><td>4.49</td><td>5.62</td></tr><tr><td>MaterialRefGS [41]</td><td>0.63</td><td>1.86</td><td>1.79</td><td>1.46</td><td>0.44</td><td>7.02</td><td>2.20</td></tr><tr><td>Ours</td><td>0.77</td><td>1.88</td><td>1.97</td><td>1.40</td><td>0.43</td><td>2.65</td><td>1.52</td></tr></table>

Table 1: Quantitative surface reconstruction comparison on the ShinySynthetic dataset (normal MAE). CD is omitted since invisible inner surfaces in the GT meshes make it unreliable, as discussed in Ref-NeuS [8] for this dataset. Best results are highlighted in red and second-best in yellow.

As shown in Tab. 4, we employ a two-stage optimization strategy to progressively optimize geometry and illumination. In Stage 1, we perform geometry initialization with Visual Geometry Priors (VGP), and use RS to downweight the photometric loss in the reflection-dominated region. Indirect illumination is disabled, and the specular reflection term is simplified to direct specular reflection: $L _ { \mathrm { s p e c } } = M _ { \mathrm { s p e c } } { \cdot } L _ { \mathrm { s p e c } } ^ { \mathrm { d i r e c t } }$ In Stage 2, we enable indirect illumination using the IndiASG model and adopt the full rendering described in Sec. 4.1, in which the VGP reweighting is disabled to allow full photometric supervision.

<table><tr><td></td><td>Mip-Cubemap IndiASG</td><td>VGP</td></tr><tr><td>Stage 1</td><td>â</td><td>â</td></tr><tr><td>Stage 2</td><td>â</td><td>â</td></tr></table>

Table 3: Two-stage strategy.

## 6 Experiment

## 6.1 Implementation Details

Our training process consists of two stages: Stage 1 runs for 10,000 iterations with densification at iteration 8,000, and Stage 2 runs for 20,000 iterations. The resolution of the cubemap is $6 \times 1 2 8 \times 1 2 8$ with four mipmap levels, downsampled by 4. $F _ { \Theta }$ consists of three hidden layers with 128 units each. PE uses a frequency of 6, while IDE uses a frequency of 4. All experiments are conducted on a single NVIDIA RTX 4090 GPU with 24 GB of memory.

## 6.2 Comparison

Datasets and Evaluation Metrics We evaluate our method on two synthetic datasets, ShinySynthetic [27] and GlossySynthetic [22], and a real-world dataset, Ref-Real [27]. All scenes feature glossy surfaces with strong specular reflections and multi-surface indirect reflections, posing significant challenges for accurate surface reconstruction. Geometry accuracy is evaluated using the mean angular error (MAE) of surface normals and the Chamfer distance (CD) of the reconstructed mesh.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=9>angel bell  cat horse luyupotiontbell teapotmean</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=9>CDâ</td></tr><tr><td rowspan=1 colspan=1>PGSR [2]</td><td rowspan=1 colspan=5>0.77 3.08 3.39 0.90 1.16</td><td rowspan=1 colspan=1>3.99</td><td rowspan=1 colspan=1>4.59</td><td rowspan=1 colspan=1>4.57</td><td rowspan=1 colspan=1>2.81</td></tr><tr><td rowspan=1 colspan=1>MILo [10]</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>16.52</td><td rowspan=1 colspan=1>2.84</td><td rowspan=1 colspan=1>1.02</td><td rowspan=1 colspan=1>44.64</td><td rowspan=1 colspan=1>3.39</td><td rowspan=1 colspan=1>5.08</td><td rowspan=1 colspan=1>4.40</td><td rowspan=1 colspan=1>9.83</td></tr><tr><td rowspan=1 colspan=1>GaussianShader [14]</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>1.10</td><td rowspan=1 colspan=1>2.56</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>1.07</td><td rowspan=1 colspan=1>4.74</td><td rowspan=1 colspan=1>5.74</td><td rowspan=1 colspan=1>3.40</td><td rowspan=1 colspan=1>2.53</td></tr><tr><td rowspan=1 colspan=1>Ref-Gaussian [37]</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>1.68</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>1.01</td><td rowspan=1 colspan=1>0.95</td></tr><tr><td rowspan=1 colspan=1>Ref-GS [42]</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>1.73</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>1.05</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=1 colspan=1>GS-ROR2 [44]</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>1.66</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.75</td></tr><tr><td rowspan=1 colspan=1>RTR-GS [43]</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>2.75</td><td rowspan=1 colspan=1>2.99</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>1.21</td><td rowspan=1 colspan=1>4.93</td><td rowspan=1 colspan=1>2.81</td><td rowspan=1 colspan=1>2.84</td><td rowspan=1 colspan=1>2.39</td></tr><tr><td rowspan=1 colspan=1>MaterialRefGS [41]</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>1.97</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>1.02</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.60</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MAEâ</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>PGSR [2]</td><td rowspan=1 colspan=1>4.90</td><td rowspan=1 colspan=1>6.39</td><td rowspan=1 colspan=1>8.43</td><td rowspan=1 colspan=1>5.97</td><td rowspan=1 colspan=1>5.91</td><td rowspan=1 colspan=1>11.12</td><td rowspan=1 colspan=1>12.62</td><td rowspan=1 colspan=1>6.28</td><td rowspan=1 colspan=1>7.70</td></tr><tr><td rowspan=1 colspan=1>MILo [10]</td><td rowspan=1 colspan=1>4.61</td><td rowspan=1 colspan=1>11.52</td><td rowspan=1 colspan=1>5.55</td><td rowspan=1 colspan=1>6.52</td><td rowspan=1 colspan=1>5.97</td><td rowspan=1 colspan=1>10.45</td><td rowspan=1 colspan=1>13.16</td><td rowspan=1 colspan=1>7.09</td><td rowspan=1 colspan=1>8.11</td></tr><tr><td rowspan=1 colspan=1>GaussianShader [14]</td><td rowspan=1 colspan=1>2.90</td><td rowspan=1 colspan=1>1.60</td><td rowspan=1 colspan=1>4.33</td><td rowspan=1 colspan=1>3.27</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1>9.52</td><td rowspan=1 colspan=1>5.60</td><td rowspan=1 colspan=1>3.01</td><td rowspan=1 colspan=1>4.35</td></tr><tr><td rowspan=1 colspan=1>3DGS-DR [19]</td><td rowspan=1 colspan=1>4.46</td><td rowspan=1 colspan=1>4.53</td><td rowspan=1 colspan=1>4.64</td><td rowspan=1 colspan=1>6.59</td><td rowspan=1 colspan=1>5.65</td><td rowspan=1 colspan=1>4.44</td><td rowspan=1 colspan=1>5.39</td><td rowspan=1 colspan=1>3.38</td><td rowspan=1 colspan=1>4.89</td></tr><tr><td rowspan=1 colspan=1>Ref-Gaussian [37]</td><td rowspan=1 colspan=1>1.79</td><td rowspan=1 colspan=1>1.16</td><td rowspan=1 colspan=1>3.15</td><td rowspan=1 colspan=1>4.03</td><td rowspan=1 colspan=1>3.15</td><td rowspan=1 colspan=1>3.04</td><td rowspan=1 colspan=1>2.02</td><td rowspan=1 colspan=1>1.20</td><td rowspan=1 colspan=1>2.44</td></tr><tr><td rowspan=1 colspan=1>Ref-GS [42]</td><td rowspan=1 colspan=1>1.99</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>2.93</td><td rowspan=1 colspan=1>3.18</td><td rowspan=1 colspan=1>2.82</td><td rowspan=1 colspan=1>3.64</td><td rowspan=1 colspan=1>1.87</td><td rowspan=1 colspan=1>1.18</td><td rowspan=1 colspan=1>2.32</td></tr><tr><td rowspan=1 colspan=1>GS-ROR2 [44]</td><td rowspan=1 colspan=1>2.09</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>3.37</td><td rowspan=1 colspan=1>2.85</td><td rowspan=1 colspan=1>3.15</td><td rowspan=1 colspan=1>4.04</td><td rowspan=1 colspan=1>2.64</td><td rowspan=1 colspan=1>1.04</td><td rowspan=1 colspan=1>2.51</td></tr><tr><td rowspan=1 colspan=1>RTR-GS [43]</td><td rowspan=1 colspan=1>3.01</td><td rowspan=1 colspan=1>3.05</td><td rowspan=1 colspan=1>2.98</td><td rowspan=1 colspan=1>6.53</td><td rowspan=1 colspan=1>3.67</td><td rowspan=1 colspan=1>7.95</td><td rowspan=1 colspan=1>4.21</td><td rowspan=1 colspan=1>2.09</td><td rowspan=1 colspan=1>4.19</td></tr><tr><td rowspan=1 colspan=1>MaterialRefGS [41]</td><td rowspan=1 colspan=1>1.81</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>2.42</td><td rowspan=1 colspan=1>3.16</td><td rowspan=1 colspan=1>3.07</td><td rowspan=1 colspan=1>2.68</td><td rowspan=1 colspan=1>1.77</td><td rowspan=1 colspan=1>1.18</td><td rowspan=1 colspan=1>2.12</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>2.15</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>1.50</td><td rowspan=1 colspan=1>3.59</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>2.84</td><td rowspan=1 colspan=1>1.80</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>2.05</td></tr></table>

Table 2: Quantitative surface reconstruction comparison on the GlossySynthetic dataset (CDÃ102 and normal MAE). Best results are highlighted in red and secondbest in yellow.

Comparisons We quantitatively evaluate performance on the ShinySynthetic dataset in Tab. 1 and the GlossySynthetic dataset in Tab. 2. Since the Ref-Real dataset does not provide ground-truth meshes, quantitative geometric evaluation is not available. The results demonstrate that our surface reconstruction achieves state-of-the-art performance. Qualitative evaluation on the ShinySynthetic and GlossySynthetic datasets is shown in Fig. 3. In the car, our method avoids surface bumps in strongly textured regions, while correctly capturing the concave structure of the tires. In the coffee, the region between the spoon and the cup exhibits complex indirect illumination and shadowing; our model accurately reconstructs these challenging areas, yielding high-fidelity geometry. In the cat, our method cleanly separates the cat from its base without undesired connections and faithfully reconstructs fine structures such as the whiskers. In the bell and teapot scenes, the surfaces are highly glossy and exhibit strong intra-object indirect specular reflections. Our method reconstructs these regions with highfidelity surfaces, avoiding erroneous geometric artifacts.

## 6.3 Ablation Study

We perform ablation experiments on ShinySynthetic [27] and GlossySynthetic [22] to evaluate the contribution of the individual components in our framework, thereby validating their effectiveness. Quantitative results are reported in Tab. 4, and qualitative comparisons are shown in Fig. 5.

<!-- image-->  
Image  
Ours  
MaterialRefGS  
GS-ROR2GS-ROR2  
Ref-Gaussian  
Ref-GS

Fig. 3: Qualitative results of surface reconstruction on ShinySynthetic (car, coffee) and GlossySynthetic (cat, bell, teapot) datasets. Since the released GS-ROR2 code cannot extract meshes on the ShinySynthetic, we instead visualize its normal results for comparison.

(1) Mip-Cubemap: We ablate our Mip-Cubemap environment representation with two variants: w/o Mip, which queries a single-resolution cubemap (akin to the Cubemap in 3DGS-DR [19]), and w/o Cube, which replaces the cubemap with a spherical mipmap representation (akin to Sph-Mip in Ref-GS [42]). As shown in Tab. 4 and Fig. 5, combining the cubemap with mip-level querying provides the best quality.

(2) IndiASG: We introduce IndiASG to explicitly model indirect specular reflection (Sec. 4.3). To evaluate its effectiveness, we conduct an ablation study by removing the indirect illumination modeling (w/o IASG), relying solely on the Mip-Cubemap representation for specular rendering. As shown in Tab. 4, the full model shows a slight advantage over w/o IASG, though the improvement is marginal in the averaged metrics. This advantage can be understated because the scenes contain only small regions with indirect specular reflections, so the quantitative results are less sensitive to IndiASGâs gains. Moreover, normal MAE may miss geometric errors; e.g., in the cat scene (Fig. 5), w/o IASG produces a misaligned base while the normals remain largely correct, yielding a small normal MAE despite noticeable geometry errors.

<!-- image-->

<!-- image-->  
Ours

<!-- image-->

<!-- image-->

Fig. 4: Qualitative results of surface reconstruction on Ref-Real (gardenspheres, toycar) dataset.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>Mip-Cubemap</td><td rowspan=1 colspan=1>IndiASG</td><td rowspan=1 colspan=5>VGP</td><td rowspan=2 colspan=1>Full[Model</td></tr><tr><td rowspan=1 colspan=1>w/o Mip</td><td rowspan=1 colspan=1>w/o Cube</td><td rowspan=1 colspan=1>w/o IASG</td><td rowspan=1 colspan=1>w/o VGP</td><td rowspan=1 colspan=1>w/o VP</td><td rowspan=1 colspan=1>w/o GP</td><td rowspan=1 colspan=1>w/o GP-D</td><td rowspan=1 colspan=1>w/o GP-N</td></tr><tr><td rowspan=1 colspan=1>MAEâ</td><td rowspan=1 colspan=1>1.56</td><td rowspan=1 colspan=1>1.62</td><td rowspan=1 colspan=1>1.52</td><td rowspan=1 colspan=1>2.25</td><td rowspan=1 colspan=1>1.55</td><td rowspan=1 colspan=1>2.15</td><td rowspan=1 colspan=1>1.53</td><td rowspan=1 colspan=1>2.86</td><td rowspan=1 colspan=1>1.52</td></tr><tr><td rowspan=1 colspan=1>MAEâ</td><td rowspan=1 colspan=1>2.07</td><td rowspan=1 colspan=1>2.17</td><td rowspan=1 colspan=1>2.06</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>2.07</td><td rowspan=1 colspan=1>2.74</td><td rowspan=1 colspan=1>2.05</td><td rowspan=1 colspan=1>2.72</td><td rowspan=1 colspan=1>2.05</td></tr><tr><td rowspan=1 colspan=1>CDâ</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.60</td></tr></table>

Table 4: Ablation study on different components. The first row corresponds to the ShinySynthetic dataset, while the remaining rows to the GlossySynthetic dataset.

(3) VGP: To comprehensively evaluate the role of Visual Geometry Priors (VGP), we consider five variants: ${ \bf w } / { \bf o }$ VGP removes both VP and GP; w/o VP removes VP while retaining GP; w/o GP removes GP while retaining VP; w/o GP-D removes the depth term $\mathcal { L } _ { \mathrm { V G G T - D } }$ and keeps only the normal term $\mathcal { L } _ { \mathrm { V G G T - N } }$ within GP; and w/o GP-N removes the normal term $\mathcal { L } _ { \mathrm { V G G T - N } }$ and keeps only the depth term $\mathcal { L } _ { \mathrm { V G G T - D } }$ within GP.

As shown in Fig. 5, removing both priors (w/o VGP) results in the most severe degradation, with unstable geometry and noticeable surface artifacts in reflective regions. When only the visual prior is removed $\left( \mathbf { w } / \mathbf { o } { \mathbf { \Delta V P } } \right)$ , strong specular gradients interfere with early-stage optimization, leading to distorted structures in areas with multi-surface reflections (cat). Similarly, removing only the geometry prior (w/o GP) causes surfaces under complex illumination to become less consistent and less smooth, even though RS still mitigates part of the specular interference. When using only a single geometric cue, performance drops compared to the full model. In w/o GP-D (normal only), the reconstructed geometry exhibits locally correct normals but suffers from depth layering artifacts, leading to stratified or offset surfaces along the viewing direction. In w/o GP-N (depth only), although the global depth structure is relatively preserved, local surface orientation becomes less accurate, and fine geometric details are weakened. In contrast, our full model jointly leverages both depth and normal supervision, achieving geometrically consistent surfaces with accurate global structure and sharp local detail. These results demonstrate that our proposed VGP combines VP and GP to yield complementary benefits and is essential for high-quality surface reconstruction under complex reflective conditions.

<!-- image-->  
Fig. 5: Ablation study on different components.

## 7 Conclusion

We propose SSR-GS, a framework for separating specular reflection in Gaussian Splatting for glossy surface reconstruction. Our method mitigates geometric artifacts in surface reconstruction under strong reflections and complex lighting. Specifically, we explicitly decouple diffuse and specular components and further decompose specular reflection into direct and indirect terms. Direct reflection is modeled via the proposed Mip-Cubemap representation for roughness-aware, view-consistent environment map sampling, while indirect reflection is modeled using the proposed IndiASG, enabling accurate representation of complex multibounce illumination. In addition, we incorporate a hybrid Visual Geometry Priors (VGP), including a reflection-aware suppression mechanism based on the Reflection Score (RS) and complementary depthânormal constraints from VGGT, to improve geometric fidelity. Extensive experiments demonstrate that SSR-GS achieves state-of-the-art performance in geometry reconstruction.

## References

1. Burley, B., Studios, W.D.A.: Physically-based shading at disney. In: Acm siggraph. vol. 2012, pp. 1â7. vol. 2012 (2012)

2. Chen, D., Li, H., Ye, W., Wang, Y., Xie, W., Zhai, S., Wang, N., Liu, H., Bao, H., Zhang, G.: Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics pp. 1â12 (2024). https://doi.org/10.1109/TVCG.2024.3494046

3. Chen, H., Li, C., Lee, G.H.: Neusg: Neural implicit surface reconstruction with 3d gaussian splatting guidance. arXiv preprint arXiv:2312.00846 (2023)

4. Cook, R.L., Torrance, K.E.: A reflectance model for computer graphics. ACM Siggraph Computer Graphics 15(3), 307â316 (1981)

5. Fan, Y., Skorokhodov, I., Voynov, O., Ignatyev, S., Burnaev, E., Wonka, P., Wang, Y.: Factored-neus: Reconstructing surfaces, illumination, and materials of possibly glossy objects. ArXiv abs/2305.17929 (2023), https://api.semanticscholar. org/CorpusID:258960329

6. Fu, Q., Xu, Q., Ong, Y.S., Tao, W.: Geo-neus: Geometry-consistent neural implicit surfaces learning for multi-view reconstruction. In: Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., Oh, A. (eds.) Advances in Neural Information Processing Systems. vol. 35, pp. 3403â3416. Curran Associates, Inc. (2022), https://proceedings.neurips.cc/paper_files/paper/2022/file/ 16415eed5a0a121bfce79924db05d3fe-Paper-Conference.pdf

7. Gao, J., Gu, C., Lin, Y., Zhu, H., Cao, X., Zhang, L., Yao, Y.: Relightable 3d gaussian: Real-time point cloud relighting with brdf decomposition and ray tracing. arXiv:2311.16043 (2023)

8. Ge, W., Hu, T., Zhao, H., Liu, S., Chen, Y.C.: Ref-neus: Ambiguity-reduced neural implicit surface learning for multi-view reconstruction with reflection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4251â4260 (October 2023)

9. Gu, C., Wei, X., Zeng, Z., Yao, Y., Zhang, L.: Irgs: Inter-reflective gaussian splatting with 2d gaussian ray tracing. In: CVPR (2025)

10. GuÃ©don, A., Gomez, D., Maruani, N., Gong, B., Drettakis, G., Ovsjanikov, M.: Milo: Mesh-in-the-loop gaussian splatting for detailed and efficient surface reconstruction. ACM Transactions on Graphics (TOG) 44(6), 1â15 (2025)

11. GuÃ©don, A., Lepetit, V.: Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5354â5363 (June 2024)

12. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. In: SIGGRAPH 2024 Conference Papers. Association for Computing Machinery (2024). https://doi.org/10.1145/3641519.3657428

13. Izadi, S., Kim, D., Hilliges, O., Molyneaux, D., Newcombe, R., Kohli, P., Shotton, J., Hodges, S., Freeman, D., Davison, A., Fitzgibbon, A.: Kinectfusion: realtime 3d reconstruction and interaction using a moving depth camera. In: Proceedings of the 24th Annual ACM Symposium on User Interface Software and Technology. p. 559â568. UIST â11, Association for Computing Machinery, New York, NY, USA (2011). https://doi.org/10.1145/2047196.2047270, https: //doi.org/10.1145/2047196.2047270

14. Jiang, Y., Tu, J., Liu, Y., Gao, X., Long, X., Wang, W., Ma, Y.: Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In: Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5322â5332 (June 2024)

15. Kajiya, J.T.: The rendering equation. In: Proceedings of the 13th annual conference on Computer graphics and interactive techniques. pp. 143â150 (1986)

16. Karis, B., Games, E.: Real shading in unreal engine 4. Proc. Physically Based Shading Theory Practice 4(3), 1 (2013)

17. Kazhdan, M., Hoppe, H.: Screened poisson surface reconstruction 32(3) (Jul 2013). https://doi.org/10.1145/2487228.2487237, https://doi.org/10.1145/ 2487228.2487237

18. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July 2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

19. Keyang, Y., Qiming, H., Kun, Z.: 3d gaussian splatting with deferred reflection (2024)

20. Lai, S., Huang, L., Guo, J., Cheng, K., Pan, B., Long, X., Lyu, J., Lv, C., Guo, Y.: Glossygs: Inverse rendering of glossy objects with 3d gaussian splatting. IEEE Transactions on Visualization and Computer Graphics pp. 1â14 (2025). https: //doi.org/10.1109/TVCG.2025.3547063

21. Liang, R., Chen, H., Li, C., Chen, F., Panneer, S., Vijaykumar, N.: Envidr: Implicit differentiable renderer with neural environment lighting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 79â89 (October 2023)

22. Liu, Y., Wang, P., Lin, C., Long, X., Wang, J., Liu, L., Komura, T., Wang, W.: Nero: Neural geometry and brdf reconstruction of reflective objects from multiview images. ACM Trans. Graph. 42(4) (Jul 2023). https://doi.org/10.1145/ 3592134, https://doi.org/10.1145/3592134

23. Lyu, X., Sun, Y.T., Huang, Y.H., Wu, X., Yang, Z., Chen, Y., Pang, J., Qi, X.: 3dgsr: Implicit surface reconstruction with 3d gaussian splatting. ACM Trans. Graph. 43(6) (Nov 2024). https://doi.org/10.1145/3687952, https://doi. org/10.1145/3687952

24. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM 65(1), 99â106 (Dec 2021). https://doi.org/10.1145/3503250, https:// doi.org/10.1145/3503250

25. Oechsle, M., Peng, S., Geiger, A.: Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 5589â5599 (October 2021)

26. Tang, Z.J., Cham, T.J.: 3igs: Factorised tensorial illumination for 3d gaussian splatting. In: European Conference on Computer Vision. pp. 143â159. Springer (2024)

27. Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J.T., Srinivasan, P.P.: Ref-nerf: Structured view-dependent appearance for neural radiance fields. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5481â5490 (2022). https://doi.org/10.1109/CVPR52688.2022.00541

28. Wang, F., Rakotosaona, M.J., Niemeyer, M., Szeliski, R., Pollefeys, M., Tombari, F.: Unisdf: Unifying neural representations for high-fidelity 3d reconstruction of complex scenes with reflections. In: NeurIPS (2024)

29. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2025)

30. Wang, J., Liu, Y., Wang, P., Lin, C., Hou, J., Li, X., Komura, T., Wang, W.: Gaussurf: Geometry-guided 3d gaussian splatting for surface reconstruction. arXiv preprint arXiv:2411.19454 (2024)

31. Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W.: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689 (2021)

32. Wang, Y., Skorokhodov, I., Wonka, P.: Hf-neus: Improved surface reconstruction using high-frequency details. In: Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., Oh, A. (eds.) Advances in Neural Information Processing Systems. vol. 35, pp. 1966â1978. Curran Associates, Inc. (2022)

33. Wang, Y., Skorokhodov, I., Wonka, P.: Pet-neus: Positional encoding tri-planes for neural surfaces. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12598â12607 (June 2023)

34. Xie, T., Chen, X., Xu, Z., Xie, Y., Jin, Y., Shen, Y., Peng, S., Bao, H., Zhou, X.: Envgs: Modeling view-dependent appearance with environment gaussian. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5742â 5751 (2025)

35. Yang, X., Wei, M.: Gogs: High-fidelity geometry and relighting for glossy objects via gaussian surfels. arXiv preprint arXiv:2508.14563 (2025)

36. Yang, Z., Gao, X., Sun, Y.T., Huang, Y.H., Lyu, X., Zhou, W., Jiao, S., Qi, X., Jin, X.: Spec-gaussian: Anisotropic view-dependent appearance for 3d gaussian splatting. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) Advances in Neural Information Processing Systems. vol. 37, pp. 61192â61216. Curran Associates, Inc. (2024), https://proceedings.neurips.cc/ paper _ files / paper / 2024 / file / 708e0d691a22212e1e373dc8779cbe53 - Paper - Conference.pdf

37. Yao, Y., Zeng, Z., Gu, C., Zhu, X., Zhang, L.: Reflective gaussian splatting. arXiv preprint (2024)

38. Ye, C., Qiu, L., Gu, X., Zuo, Q., Wu, Y., Dong, Z., Bo, L., Xiu, Y., Han, X.: Stablenormal: Reducing diffusion variance for stable and sharp normal. ACM Trans. Graph. 43(6) (Nov 2024). https://doi.org/10.1145/3687971, https: //doi.org/10.1145/3687971

39. Yu, M., Lu, T., Xu, L., Jiang, L., Xiangli, Y., Dai, B.: Advances in neural information processing systems. vol. 37, pp. 129507â129530. Curran Associates, Inc. (2024), https://proceedings.neurips.cc/paper_files/paper/2024/file/ ea13534ee239bb3977795b8cc855bacc-Paper-Conference.pdf

40. Yu, Z., Sattler, T., Geiger, A.: Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (2024)

41. Zhang, W., Tang, J., Zhang, W., Fang, Y., Liu, Y.S., Han, Z.: MaterialRefGS: Reflective gaussian splatting with multi-view consistent material inference. In: Advances in Neural Information Processing Systems (2025)

42. Zhang, Y., Chen, A., Wan, Y., Song, Z., Yu, J., Luo, Y., Yang, W.: Ref-gs: Directional factorization for 2d gaussian splatting. ArXiv abs/2412.00905 (2024), https://api.semanticscholar.org/CorpusID:274436898

43. Zhou, Y., Zhang, F., Wang, Z., Zhang, L.: Rtr-gs: 3d gaussian splatting for inverse rendering with radiance transfer and reflection. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 6888â6897 (2025)

44. Zhu, Z.L., Wang, B., Yang, J.: Gs-ror: 3d gaussian splatting for reflective object relighting via sdf priors. ArXiv abs/2406.18544 (2024), https://api. semanticscholar.org/CorpusID:270764572