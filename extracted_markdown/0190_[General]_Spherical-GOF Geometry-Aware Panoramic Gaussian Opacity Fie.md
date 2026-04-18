# Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction

Zhe Yang1, Guoqiang Zhao1, Sheng Wu1, Kai Luo1, and Kailun Yang1,2,â 

<!-- image-->

<!-- image-->  
Fig. 1: Our ray casting-based method enables accurate panoramic Gaussian rendering from an input panorama and SfM poses. Compared to projection-based approaches (ODGS [1] and OmniGS [2]) and prior ray-based methods (SPaGS [3]), our method produces cleaner and more geometrically consistent depth and normals, avoiding texture-like ripple artifacts.

Abstractâ Omnidirectional images are increasingly used in robotics and vision due to their wide field of view. However, extending 3D Gaussian Splatting (3DGS) to panoramic camera models remains challenging, as existing formulations are designed for perspective projections and naive adaptations often introduce distortion and geometric inconsistencies.

We present Spherical-GOF, an omnidirectional Gaussian rendering framework built upon Gaussian Opacity Fields (GOF). Unlike projection-based rasterization, Spherical-GOF performs GOF ray sampling directly on the unit sphere in spherical ray space, enabling consistent ray-Gaussian interactions for panoramic rendering. To make the spherical ray casting efficient and robust, we derive a conservative spherical bounding rule for fast ray-Gaussian culling and introduce a spherical filtering scheme that adapts Gaussian footprints to distortionvarying panoramic pixel sampling.

Extensive experiments on standard panoramic benchmarks (OmniBlender and OmniPhotos) demonstrate competitive photometric quality and substantially improved geometric consistency. Compared with the strongest baseline, Spherical-GOF reduces depth reprojection error by 57% and improves cycle inlier ratio by 21%. Qualitative results show cleaner depth and more coherent normal maps, with strong robustness to global panorama rotations. We further validate generalization on OmniRob, a real-world robotic omnidirectional dataset introduced in this work, featuring UAV and quadruped platforms. The source code and the OmniRob dataset will be released at https://github.com/1170632760/Spherical-GOF.

## I. INTRODUCTION

Advances in robotics and embodied intelligence, together with the rise of AR/VR and digital-twin applications, have made 3D scene reconstruction increasingly important for perception, simulation, and environment understanding [4], [5]. Accurate geometric representations are essential for building realistic simulation environments and digital twins, enabling perception-driven analysis and supporting downstream embodied AI and human-robot interaction research [6], [7]. Conventional reconstruction pipelines [8], [9] relying on pinhole cameras typically require the acquisition of extensive image datasets, followed by the recovery of featurebased point cloud maps via Structure-from-Motion (SfM). Panoramic cameras offer an expansive Field of View (FoV), enabling efficient 360Â° scene coverage with far fewer images under practical sensing constraints, and have therefore attracted increasing attention in robotics and embodied vision [10], [11].

Simultaneously, methods based on Neural Radiance Fields (NeRF) [12] and 3DGS [13] have garnered significant attention due to their ability to not only recover scene geometry but also render high-fidelity photorealistic textures. Consequently, extending these paradigms to support panoramic cameras has become a focal point of research. Given the inherent ray-sampling nature of NeRF, adapting it to panoramic models is relatively straightforward, and several studies have demonstrated impressive performance in this domain [14], [15], [16]. However, such methods inevitably inherit NeRFâs intrinsic limitations, specifically low rendering efficiency and protracted training times. Conversely, while 3DGS offers a significant leap in rendering speed, adapting it to panoramic imaging presents substantial challenges. Standard

3DGS [13]explicitly represents scenes using 3D Gaussians and relies on a projection mechanism tailored for pinhole camera models. Specifically, the projected covariance is computed by applying the Jacobian of the camera model to linearly approximate the inherently non-linear projection, under the assumption that a projected 3D Gaussian remains an ellipsoidal Gaussian on the image plane. Such geometric assumptions substantially impede the direct extension of 3D Gaussian Splatting to panoramic imagery; moreover, even spherical-aware splatting variants that achieve strong visual quality often prioritize optical appearance, leaving geometryoriented accuracy less explicitly emphasized [1], [2], [3].

To address these challenges, we propose Spherical-GOF, a ray-space framework for omnidirectional Gaussian rendering. Building on GOF [17], Spherical-GOF performs ray sampling directly on the unit sphere, enabling projectionconsistent ray-Gaussian interactions for panoramic imaging without relying on planar projection approximations. To ensure robust and efficient rendering, we further derive a conservative spherical bounding strategy for Gaussian primitives, which supports reliable rayâGaussian culling in omnidirectional settings. In addition, we introduce a spherical filtering scheme that adapts Gaussian footprints to panoramic pixel sampling, effectively mitigating aliasing artifacts induced by spherical distortion and improving rendering stability. Quantitatively, Spherical-GOF reduces depth reprojection error by 57% and improves cycle consistency by 21% over the strongest baseline, while maintaining competitive rendering performance. These improvements translate into more accurate geometry and more reliable mesh extraction, making the method better suited for downstream tasks that require consistent surface reconstruction.

In summary, this paper makes the following contributions:

â¢ We propose Spherical-GOF, a spherical ray-space GOF sampling framework for ERP panoramas, which improves geometric reconstruction accuracy of omnidirectional Gaussian rendering by avoiding local linearization errors introduced by planar projection.

â¢ We introduce a panoramic filter and sphere-metricconsistent geometric regularization that stabilize training and reduce the influence of high-frequency appearance textures on geometry, leading to cleaner depth and more coherent normal estimates.

â¢ We conduct extensive experiments on public panoramic benchmarks to evaluate both photometric and geometric quality, and further validate camera-model adaptation on our robot-captured omnidirectional data, showing that our formulation can be applied to diverse omnidirectional camera setups with only minor modifications.

## II. RELATED WORK

3D Gaussian splatting. Recent advances in 3D reconstruction have been largely driven by NeRF [12] and 3DGS [13], both of which enable high-quality novel view synthesis. In particular, 3DGS has gained popularity for its real-time rendering speed. It represents a scene as a set of anisotropic 3D Gaussians with learnable attributes such as scale, rotation, opacity, and color, and renders them efficiently via rasterization. However, this efficiency relies on a local affine approximation of the projection, which can become unreliable for wide-FoV or highly distorted camera models.

To improve rendering correctness and geometric reliability, several works modify the Gaussian representation or evaluation strategy. For example, 2D Gaussian Surfels (2DGS) [18] represent primitives as local surface patches, enabling more accurate depth and normal estimation. GOF [17] instead formulates Gaussian rendering in a volumetric occupancystyle manner, producing well-defined depth and normal outputs. HTGS [19] further improves 3DGS by introducing perspective-correct, ray-based splat evaluation that avoids matrix inversion and stabilizes optimization. These developments highlight the importance of projection-consistent formulations, especially when extending Gaussian-based reconstruction to omnidirectional panoramic imagery.

Panoramic 3D reconstruction. Thanks to the wide FoV of panoramic cameras [10], panorama-based 3D reconstruction has received increasing attention. In NeRF, extending volumetric rendering to panoramic imagery is relatively straightforward since radiance is evaluated along rays; representative examples include EgoNeRF [16] and other NeRF-based methods for omnidirectional images [20], [21]. In contrast, adapting 3DGS to panoramic images is more challenging because its rasterization pipeline is tightly coupled with the camera projection model and typically assumes that a 3D Gaussian remains an elliptical 2D Gaussian after projection.

To support panoramic rendering, some methods derive the Jacobian of panoramic projection and propagate covariance accordingly, such as OmniGS [2], ErpGS [22], and 360- GS [23]. However, they still rely on a local affine approximation, which can break down in highly distorted regions such as the polar areas of equirectangular panoramas and lead to projection inconsistencies. To avoid a single global projection, other approaches render on intermediate surfaces. For instance, ODGS [1] projects Gaussians onto locally defined tangent planes and maps them back to the panorama, while face-based methods render on cubemap-like piecewise perspective surfaces, optionally introducing additional transition planes to reduce cross-face discontinuities, before mapping back to the panorama [24]. While such strategies alleviate severe distortion, they may introduce additional overhead and can suffer from approximation/stitching issues across pieces.

More recently, SPaGS [3] extends the ray casting formulation of HTGS [19] to spherical panoramas via omnidirectional ray-splat intersection and bounding-box-based rasterization, enabling projection-consistent panoramic rendering without local affine projection approximations.

## III. METHODOLOGY

A. Preliminary: Classical Gaussian Splatting and Gaussian Opacity Fields

3DGS is an emerging approach to 3D reconstruction. It takes as input a set of posed images and a sparse point cloud, which is typically obtained from SfM, it initializes a collection of anisotropic 3D Gaussians and optimizes their parameters by differentiable rendering under a photometric reconstruction loss. Each Gaussian i is parameterized by its mean position $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ , color $\mathbf { c } _ { i } .$ opacity $o _ { i }$ , and covariance $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ . The covariance of each Gaussian is commonly represented using a rotation matrix $\mathbf { R } _ { i } \in S O ( 3 )$ and per-axis scales $\mathbf { s } _ { i } \in \mathbb { R } ^ { 3 }$ . Let $\mathbf { S } _ { i } = \mathrm { d i a g } ( \mathbf { s } _ { i } )$ denote the scaling matrix. The covariance is then given by

$$
\pmb { \Sigma } _ { i } = \mathbf { R } _ { i } \mathbf { S } _ { i } \mathbf { S } _ { i } ^ { \top } \mathbf { R } _ { i } ^ { \top } .\tag{1}
$$

Each Gaussian defines a 3D density

$$
G _ { i } ( \mathbf { x } ) = \mathrm { e x p } \left( - \frac { 1 } { 2 } ( \mathbf { x } - \pmb { \mu } _ { i } ) ^ { \top } \pmb { \Sigma } _ { i } ^ { - 1 } ( \mathbf { x } - \pmb { \mu } _ { i } ) \right) .\tag{2}
$$

Since perspective projection is inherently non-linear, classical 3DGS adopts the EWA-splatting formulation [25], which approximates the projected footprint by a 2D Gaussian via local first-order linearization. Let $\mathbf { x } _ { i } ^ { c }$ denote the mean transformed to the camera coordinate system, and let $\pi ( \cdot )$ be the perspective projection. The Jacobian of the projection evaluated at $\mathbf { x } _ { i } ^ { c }$ is denoted as $\mathbf { J } _ { i } \in \mathbb { R } ^ { 2 \times 3 }$ . The screen-space covariance is computed as

$$
\begin{array} { r } { \boldsymbol { \Sigma } _ { i } ^ { ' } = \mathbf { J } _ { i } \left( \mathbf { R } _ { w } ^ { c } \boldsymbol { \Sigma } _ { i } \left( \mathbf { R } _ { w } ^ { c } \right) ^ { \top } \right) \mathbf { J } _ { i } ^ { \top } . } \end{array}\tag{3}
$$

where $\mathbf { R } _ { w } ^ { c }$ denotes the rotation from the world coordinate system to the camera coordinate system.

The resulting 2D Gaussian footprint at pixel coordinate $\mathbf { u } \in \mathbb { R } ^ { 2 }$ is

$$
G _ { i } ^ { ' } ( { \mathbf { u } } ) = \exp \left( - \frac { 1 } { 2 } ( { \mathbf { u } } - { \boldsymbol { \mu } } _ { i } ^ { ' } ) ^ { \top } ( \Sigma _ { i } ^ { ' } ) ^ { - 1 } ( { \mathbf { u } } - { \boldsymbol { \mu } } _ { i } ^ { ' } ) \right) ,\tag{4}
$$

where $\pmb { \mu } _ { i } ^ { ' } = \pi ( \mathbf { x } _ { i } ^ { c } )$ denotes the projected mean. Finally, the pixel color is computed as:

$$
\mathbf { C } ( \mathbf { u } ) = \sum _ { j = 1 } ^ { N } \mathbf { c } _ { j } \alpha _ { j } ( \mathbf { u } ) T _ { j } ( \mathbf { u } ) .\tag{5}
$$

Here, $\alpha _ { j } ( { \mathbf { u } } ) = o _ { j } G _ { j } ^ { ' } ( { \mathbf { u } } )$ , and $\begin{array} { r } { T _ { j } ( \mathbf { u } ) = \prod _ { k = 1 } ^ { j - 1 } \left( 1 - \alpha _ { k } ( \mathbf { u } ) \right) } \end{array}$ denotes the accumulated transmittance.

## B. Panoramic Rendering via Gaussian Opacity Field

Gaussian Opacity Fields [17] offer a ray-based rendering formulation that avoids screen-space projection approximations. GOF evaluates each Gaussianâs opacity accumulation directly along camera rays, improving geometric consistency and yielding more accurate geometry estimation.

Specifically, given the camera center $\mathbf { o } \in \mathbb { R } ^ { 3 }$ and a ray direction $\textbf { r } \in \ \mathbb { R } ^ { 3 }$ , any point along the ray can be written as ${ \bf x } = { \bf o } + t { \bf r }$ , where t denotes the ray depth. For a 3D Gaussian i with mean $\mathbf { x } _ { i } .$ , rotation $\mathbf { R } _ { i }$ , and scaling matrix $\mathbf { S } _ { i } = \mathrm { d i a g } ( \mathbf { s } _ { i } )$ , we transform the ray into the local coordinate frame of the Gaussian,

$$
\mathbf { o } _ { i } = \mathbf { S } _ { i } ^ { - 1 } \mathbf { R } _ { i } ( \mathbf { o } - \mathbf { x } _ { i } ) ,\tag{6}
$$

$$
\mathbf { r } _ { i } = \mathbf { S } _ { i } ^ { - 1 } \mathbf { R } _ { i } \mathbf { r } ,\tag{7}
$$

$$
\mathbf { x } _ { i } ( t ) = \mathbf { o } _ { i } + t \mathbf { r } _ { i } .\tag{8}
$$

Substituting the point in the Gaussian local frame into the Gaussian formulation yields the 1D response along the ray:

$$
\begin{array} { r l r } {  { \mathcal { G } _ { i } ^ { \mathrm { 1 D } } ( t ) = \exp ( - \frac { 1 } { 2 } \mathbf { x } _ { i } ( t ) ^ { \top } \mathbf { x } _ { i } ( t ) ) } } \\ & { } & { = \exp ( - \frac { 1 } { 2 } ( \mathbf { r } _ { i } ^ { \top } \mathbf { r } _ { i } t ^ { 2 } + 2 \mathbf { o } _ { i } ^ { \top } \mathbf { r } _ { i } t + \mathbf { o } _ { i } ^ { \top } \mathbf { o } _ { i } ) ) . } \end{array}\tag{9}
$$

According to Eq. (9), the exponent is a quadratic function of t, whose maximum is attained at $\scriptstyle t ^ { * } = - { \frac { B } { A } }$ , where $A = \mathbf { r } _ { i } ^ { \top } \mathbf { r } _ { i }$ and $B = \mathbf { o } _ { i } ^ { \top } \mathbf { r } _ { i }$

Since GOF evaluates Gaussian contributions along rays, the rendering no longer depends on a specific projection model. This property enables a seamless extension to panoramic imaging without introducing projectionapproximation errors. We define the camera coordinate system such that the z-axis points forward and the x-axis points to the right, consistent with the pinhole convention. For an equirectangular panorama, the longitude and latitude of a 3D point x are defined as $\varphi = \mathrm { a t a n 2 } ( x _ { x } , x _ { z } ) \in [ - \pi , \pi ]$ and $\phi = \arcsin ( - x _ { y } / \Vert \mathbf { x } \Vert _ { 2 } ) \in [ - \pi / 2 , \pi / 2 ]$ . Using these definitions, the panoramic projection is given by

$$
\operatorname { P r o j } ( \mathbf { x } ) = \left( { \frac { W } { 2 \pi } } \varphi + { \frac { W } { 2 } } , - { \frac { H } { \pi } } \phi + { \frac { H } { 2 } } \right) ^ { \top } .\tag{10}
$$

In addition, during preprocessing, it is necessary to determine the tile range influenced by each Gaussian for efficient rendering. In the original 3DGS pipeline, this range is estimated using the projection Jacobian, which is not suitable for panoramic imaging. Since directly computing the exact longitudinal and latitudinal extent of an anisotropic Gaussian on the panorama is challenging, we approximate each Gaussian as a sphere whose diameter is determined by its longest principal axis. We then compute conservative upper and lower bounds of the longitude and latitude covered by this sphere. Although this estimated range is looser than the exact projection, it guarantees that no valid ray-Gaussian contributions are clipped.

## C. Optimization Strategies for Panoramic Rendering

Due to latitude-dependent distortion in panoramic projections, Gaussians with identical 3D size can occupy markedly different areas on the image plane depending on their latitude. As a result, Gaussians at higher latitudes tend to accumulate much larger gradients than those at lower latitudes. To balance the splitting thresholds across different latitudes, we modify the Gaussian gradients to be latitudedependent. We define

$$
w _ { \mathrm { l a t } } = \mathrm { c l a m p } ( \cos ( \phi ) , \epsilon ) ,\tag{11}
$$

This weight is multiplied by the densification score to suppress excessive splitting near the poles.

For ERP panoramas, the angular resolution varies with latitude, so identically sized Gaussians may have very different pixel footprints. To avoid sub-pixel footprints that cause aliasing and instability, we assign each Gaussian an isotropic filter radius $f _ { i }$ according to the panoramaâs angular resolution. For each visible camera, we compute the camera-space distance $r = \lVert \mathbf { x } _ { \mathrm { c a m } } \rVert _ { 2 }$ and the latitude $\phi = \arcsin ( - x _ { y } / r )$ Given an image resolution $W \times H$ , the vertical and horizontal angular resolutions are

$$
\Delta \theta _ { \mathrm { l a t } } = \frac { \pi } { H } , \qquad \Delta \theta _ { \mathrm { l o n } } = \frac { 2 \pi } { W } \cos \phi .\tag{12}
$$

and we set $\Delta \theta = \operatorname* { m a x } ( \Delta \theta _ { \mathrm { { l a t } } } , \Delta \theta _ { \mathrm { { l o n } } } )$ . This yields a pixelsupport candidate $f _ { \mathrm { c a n d } } ~ = ~ r \Delta \theta$ . For a single Gaussian primitive, we take the maximum over all visible cameras and apply a constant factor Îº:

$$
f = \kappa \operatorname* { m a x } _ { \mathrm { c a m s } } f _ { \mathrm { c a n d } } .\tag{13}
$$

Given the per-axis scale $\mathbf { s } \in \mathbb { R } ^ { 3 }$ , we inflate the Gaussian scale with an isotropic radius $f \colon$

$$
\tilde { \mathbf { s } } = \sqrt { \mathbf { s } \odot \mathbf { s } + f ^ { 2 } \mathbf { 1 } } ,\tag{14}
$$

where â denotes the Hadamard product and $\textbf { 1 } \in \mathbb { R } ^ { 3 }$ is an all-ones vector. This inflation enforces a stable lower bound on the Gaussian extent and avoids sub-pixel footprints.

Since the inflation changes the Gaussian volume, we compensate the opacity to preserve density consistency:

$$
o \gets o \cdot \sqrt { \frac { \prod ( \mathbf { s } \odot \mathbf { s } ) } { \prod ( \widetilde { \mathbf { s } } \odot \widetilde { \mathbf { s } } ) } } ,\tag{15}
$$

where Q(Â·) denotes the product of vector components. We apply this procedure to all Gaussians, which maintains sufficient image support in distant or high-latitude regions and suppresses numerical instability and aliasing.

## D. Panorama-Aware Geometric Losses

Ray-space rendering removes screen-space projection errors, but optimizing geometry from photometric supervision alone remains ill-posed. Without additional constraints, appearance variations can be partially explained by geometry, leading to high-frequency artifacts in depth and normals. We therefore augment the photometric objective with panoramaaware geometric regularizers:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { d n } \mathcal { L } _ { \mathrm { d n } } + \lambda _ { j 1 } \mathcal { L } _ { \mathrm { j u m p } 1 } + \lambda _ { j 2 } \mathcal { L } _ { \mathrm { j u m p } 2 } . } \end{array}\tag{16}
$$

Here, $\mathcal { L } _ { \mathrm { r g b } }$ denotes the standard reconstruction loss used in Gaussian rendering, and the additional terms regularize normals and depth to suppress high-frequency geometric artifacts.

Opacity-based valid mask. Geometric regularization is only meaningful on pixels with sufficient rendered opacity. We therefore define the valid pixel set

$$
\Omega = \{ { \bf u } | \alpha ( { \bf u } ) > \tau \} ,\tag{17}
$$

where $\alpha ( { \mathbf { u } } )$ denotes the accumulated opacity at pixel u and Ï is a fixed threshold.

Depth-normal consistency. We encourage consistency between the rendered normal map ${ \bf N } ( { \bf u } )$ and a depth-induced normal map $\mathbf { N } ^ { d } ( \mathbf { u } )$ computed from the rendered depth. Let $\mathbf { p } ( \mathbf { u } )$ denote the 3D point obtained by back-projecting pixel u with its rendered depth. We estimate

$$
\mathbf { N } ^ { d } ( \mathbf { u } ) = \mathrm { n o r m } ( \Delta _ { x } \mathbf { p } ( \mathbf { u } ) \times \Delta _ { y } \mathbf { p } ( \mathbf { u } ) ) ,\tag{18}
$$

Here, $\Delta _ { x }$ and $\Delta _ { y }$ denote first-order finite differences of $\mathbf { p } ( \mathbf { u } )$ along the image x- and y-directions using neighboring pixels, and we minimize the angular discrepancy between ${ \bf N } ( { \bf u } )$ and $\mathbf { N } ^ { d } ( \mathbf { u } )$

$$
\mathcal { L } _ { d n } = \sum _ { \mathbf { u } \in \Omega } \omega _ { \mathrm { l a t } } ( \phi ) \left( 1 - \left| \mathbf { N ( \mathbf { u } ) } ^ { \top } \mathbf { N } ^ { d } ( \mathbf { u } ) \right| \right) ,\tag{19}
$$

Here, â¦ denotes the opacity-based valid set defined in (17), and we apply the latitude weight $\omega _ { \mathrm { l a t } } ( \phi )$ in (11) to compensate for $\mathrm { E R P s }$ latitude-dependent distortion and balance the contributions across latitudes.

Depth jump regularization. We suppress depth oscillations by applying hinge penalties to log-depth differences. Let Ïµ be a small constant and define $z ( \mathbf { u } ) = \log ( \operatorname* { m a x } ( D ( \mathbf { u } ) , \epsilon ) )$ and $s ( \phi ) = \mathrm { m a x } ( \cos \phi , \epsilon )$ . We compute first-order differences

$$
\Delta _ { x } z ( \mathbf { u } ) = \frac { z ( \mathbf { u } + \Delta _ { x } ) - z ( \mathbf { u } ) } { s ( \phi ) } ,\tag{20}
$$

$$
\Delta _ { y } z ( \mathbf { u } ) = z ( \mathbf { u } + \Delta _ { y } ) - z ( \mathbf { u } ) ,\tag{21}
$$

and the corresponding hinge responses

$$
E _ { x } ( \mathbf { u } ) = \mathrm { m a x } \big ( | \Delta _ { x } z ( \mathbf { u } ) | - \tau _ { 1 } , 0 \big ) ,\tag{22}
$$

$$
E _ { y } ( \mathbf { u } ) = \operatorname* { m a x } \bigr ( | \Delta _ { y } z ( \mathbf { u } ) | - \tau _ { 1 } , 0 \bigr ) .\tag{23}
$$

With edge-aware weights $w _ { x } ( \mathbf { u } ) = \exp ( - \beta \| \partial _ { x } I ( \mathbf { u } ) \| )$ and $\begin{array} { r l r } { w _ { y } ( \mathbf { u } ) } & { { } = } & { \exp ( - \beta \| \partial _ { y } I ( \mathbf { u } ) \| ) } \end{array}$ computed from the input panorama I, we define

$$
\mathcal { L } _ { \mathrm { j u m p 1 } } = \sum _ { \mathbf { u } \in \Omega } \omega _ { \mathrm { l a t } } ( \phi ) \left( w _ { x } ( \mathbf { u } ) E _ { x } ( \mathbf { u } ) + w _ { y } ( \mathbf { u } ) E _ { y } ( \mathbf { u } ) \right) .\tag{24}
$$

We further penalize second-order log-depth differences to obtain ${ \mathcal { L } } _ { \mathrm { j u m p } 2 }$ , using the same horizontal ERP correction to reduce ripple-like artifacts.

## IV. EXPERIMENTS

## A. Experimental Setup

Datasets.

We conduct experiments on three datasets: the synthetic OmniBlender [16], the real-world OmniPhotos [26], and OmniRob. OmniBlender is rendered in Blender, offering controlled omnidirectional imagery without operator-induced motion artifacts. OmniPhotos is captured using a consumer 360Â° camera in real environments.

To evaluate transfer across camera models in robotic settings, we collect OmniRob on two platforms equipped with different omnidirectional cameras. OmniRob-UAV contains aerial sequences captured by an Antigravity UAV that provides full equirectangular panoramas. OmniRob-Quadruped contains ground-level sequences captured by a Unitree Go2 legged robot equipped with an annular panoramic camera. Overall, OmniRob includes four scenes, with two scenes per platform. This setup enables evaluation under heterogeneous viewpoints and camera parameterizations.

The annular camera used in OmniRob-Quadruped has a limited vertical field of view of $[ - 3 9 ^ { \circ } , 6 ^ { \circ } ]$ . To facilitate controlled comparisons under restricted vertical coverage, we crop the UAV panoramas to $[ - 4 0 ^ { \circ } , 2 0 ^ { \circ } ]$ to form pseudoannular observations. Following ODGS [1], we subsample

<table><tr><td></td><td colspan="5">OmniBlender-Indoor [16]</td><td colspan="5">OmniBlender-Outdoor</td><td colspan="5">OmniPhotos [26]</td></tr><tr><td>Method</td><td>DREâ</td><td>CIRâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>DREâ</td><td>CIRâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>DREâ</td><td>CIRâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>EgoNeRF [16]</td><td>0.1254</td><td>49.47</td><td>31.427</td><td>0.8724</td><td>0.1916</td><td>0.4416</td><td>56.41</td><td>29.195</td><td>0.8746</td><td>0.1264</td><td>0.3443</td><td>51.93</td><td>26.283</td><td>0.8038</td><td>0.1951</td></tr><tr><td>ODGS [1]</td><td>0.2022</td><td>37.25</td><td>31.853</td><td>0.8873</td><td>0.1215</td><td>0.3116</td><td>46.34</td><td>28.884</td><td>0.8876</td><td>0.0885</td><td>0.3702</td><td>62.33</td><td>25.582</td><td>0.8247</td><td>0.1802</td></tr><tr><td>OmniGS [2]</td><td>0.0591</td><td>61.00</td><td>36.046</td><td>0.9233</td><td>0.0720</td><td>0.0965</td><td>58.59</td><td>32.683</td><td>0.9276</td><td>0.0479</td><td>0.1891</td><td>46.48</td><td>29.241</td><td>0.9005</td><td>0.0905</td></tr><tr><td>SPaGS [3]</td><td>0.0453</td><td>73.86</td><td>35.353</td><td>0.9248</td><td>0.1093</td><td>0.0944</td><td>67.75</td><td>32.412</td><td>0.9270</td><td>0.0588</td><td>0.1295</td><td>74.41</td><td>28.493</td><td>0.8936</td><td>0.1171</td></tr><tr><td>Ours</td><td>0.0169</td><td>90.56</td><td>34.684</td><td>0.9198</td><td>0.0709</td><td>0.0416</td><td>85.28</td><td>31.403</td><td>0.9195</td><td>0.0682</td><td>0.0620</td><td>86.02</td><td>27.797</td><td>0.8872</td><td>0.0978</td></tr></table>

TABLE I: Quantitative comparison on public omnidirectional benchmarks. We report rendering quality (PSNR/SSIM/LPIPS), Depth reprojection error (DRE), and Cycle inlier ratio (CIR).  
EgoNeRF [16]  
ODGS [1]

OmniGS [2]  
SPaGS [3]  
Ours  
<!-- image-->  
Fig. 2: Qualitative comparison on OmniBlender [16]. We show RGB renderings and geometry-related outputs for representative scenes. Compared to projection-based panoramic 3DGS baselines, our method yields smoother and more structurally consistent depth, with fewer texture-aligned ripples on planar regions. Normal maps are computed from the rendered depth for visualization.

100 images per scene, use 25 views for training and 25 for testing, and reserve the remaining views unless otherwise specified. We initialize the sparse point cloud and camera poses using OpenMVG [9].

Implementation details. Baselines are reproduced using their official settings. Our method is trained for 8k iterations on a single NVIDIA RTX 4090 GPU, with densification stopped at 4k iterations. We set $\lambda _ { j 1 } = 0 . 4 5 , \lambda _ { j 2 } = 0 . 3 2 .$ and $\lambda _ { d n } = 0 . 0 3$ for geometry regularization. We ramp up geometry regularization during training and enable the depthnormal term only in the later stage.

Evaluation metrics. We evaluate image-space rendering quality using PSNR, SSIM [27], and LPIPS [28], where LPIPS is computed with the AlexNet backbone [29].

To assess multi-view geometric stability without groundtruth depth, we report a depth reprojection consistency error (DRE). For a pixel u in view i with rendered depth $D _ { i } ( \mathbf { u } )$ we back-project it to 3D, transform the point to view j, and reproject it to obtain a correspondence uâ² and the predicted depth $D _ { \mathrm { p r o j } } ( \mathbf { u } ^ { \prime } )$ . We then sample the rendered depth of view j at uâ² by bilinear interpolation, denoted $D _ { j } ( \mathbf { u } ^ { \prime } )$ , and define

$$
e ( \mathbf { u } ) = \frac { | D _ { \mathrm { p r o j } } ( \mathbf { u } ^ { \prime } ) - D _ { j } ( \mathbf { u } ^ { \prime } ) | } { D _ { j } ( \mathbf { u } ^ { \prime } ) + \epsilon } .\tag{25}
$$

We report the mean of $e ( \mathbf { u } )$ over all valid overlapping pixels across the same set of view pairs. A pixel is valid if $\mathbf { u } ^ { \prime }$ lies inside the image domain of view j and both $D _ { i } ( \mathbf { u } )$ and $D _ { j } ( \mathbf { u } ^ { \prime } )$ are defined.

We additionally report the Cycle Inlier Ratio (CIR), which measures the fraction of pixels that remain consistent under a forward-backward reprojection cycle. Starting from u in view $i ,$ we reproject to view $j$ to obtain $\mathbf { u } ^ { \prime } ,$ , and then reproject $\mathbf { u } ^ { \prime }$ back to view i to obtain uË. We count u as an inlier if the round-trip pixel error $\| \hat { \mathbf { u } } - \mathbf { u } \| _ { 2 }$ is below a threshold $\tau _ { \mathrm { c y c } }$ (we use $\tau _ { \mathrm { c y c } } = 2$ pixels). CIR is computed as the inlier ratio over the same valid overlapping pixels and view pairs. Lower DRE and higher CIR indicate more stable, view-consistent geometry.Intuitively, DRE measures the severity of crossview depth drift, while CIR reflects the coverage of viewconsistent geometry.

<table><tr><td></td><td colspan="5"> $\theta = 0 ^ { \circ }$ </td><td colspan="5"> $\theta = 6 0 ^ { \circ }$ </td><td colspan="5"> $\theta = 9 0 ^ { \circ }$ </td></tr><tr><td>Method</td><td>DREâ</td><td>CIRâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>DREâ</td><td>CIRâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>DREâ</td><td>CIRâ</td><td>PSNRâ</td><td>SSIMâLPIPSâ</td><td></td></tr><tr><td>ODGS [1]</td><td>| 0.2717 43.03</td><td></td><td>29.25</td><td>0.8862</td><td>0.1007</td><td></td><td>0.2968</td><td>42.43 25.63</td><td>0.8622</td><td>0.1161</td><td></td><td>0.2977</td><td>38.81 24.59</td><td>0.8440</td><td>0.1295</td></tr><tr><td>OmniGS [2]</td><td>0.0848 58.42</td><td></td><td>33.91</td><td>0.9260</td><td>0.0969</td><td>0.0887</td><td>58.05</td><td>24.28</td><td>0.8386</td><td>0.1461</td><td>0.0857</td><td>60.04</td><td>23.01</td><td>0.8193</td><td>0.1689</td></tr><tr><td>SPaGS [3]</td><td>0.0954</td><td>67.76</td><td>33.48</td><td>0.9262</td><td>0.0771</td><td>0.0945</td><td>67.92</td><td>34.89</td><td>0.9444</td><td>0.0589</td><td>0.0955</td><td>67.75</td><td>34.50</td><td>0.9407</td><td>0.0644</td></tr><tr><td>Ourss</td><td>0.0326 87.20</td><td></td><td>32.61</td><td>0.9203</td><td>0.0697</td><td>0.0323</td><td>89.21</td><td>31.12</td><td>0.9188</td><td>0.0525</td><td>0.0322</td><td>90.36</td><td>30.27</td><td>0.9065</td><td>0.0585</td></tr></table>

TABLE II: Rotation robustness on OmniBlender [16]. Models are trained on canonical poses and evaluated under additional random rotations within Â±Î¸.  
ODGS [1]

OmniGS [2]  
SPaGS [3]  
Ours  
<!-- image-->  
Ground truth  
Fig. 3: Rotation robustness under global panorama rotations. We train with canonical poses and evaluate the same scene under additional global rotations of 0â¦, 60â¦, and 90â¦. Each result includes a zoom-in inset from a fixed region for easier comparison. Projection-based methods (ODGS [1] and OmniGS [2]) show increasing rotation-dependent degradation, while SPaGS [3] and our method remain more stable under large rotations.

## B. Experiment Results

Quantitative comparison. Table I summarizes photometric quality and multi-view geometry consistency on OmniBlender and OmniPhotos. Overall, OmniGS and SPaGS achieve strong photometric performance, and our method remains competitive on image-space metrics. More importantly, our method consistently improves geometric consistency across all benchmarks, achieving the lowest DRE and the highest CIR in every setting. On OmniBlender-Indoor, our method reduces DRE by 62.7% and increases CIR by 22.6% relative to SPaGS, indicating substantially more viewconsistent depth on scenes with large planar structures. Similar improvements are observed on OmniBlender-Outdoor and OmniPhotos, demonstrating stable cross-view geometry under both synthetic and real-world panoramas. This gain comes with a trade-off in training efficiency: SPaGS is the most training-efficient baseline and typically converges in tens of minutes, whereas our current implementation converges in about 1 hour.

Qualitative comparison. Fig. 2 presents qualitative comparisons on the OmniBlender, where large planar surfaces make geometric artifacts particularly apparent. While all methods achieve visually plausible RGB renderings, we observe clear differences in the reconstructed geometry. Projection-based panoramic 3DGS baselines often introduce high-frequency distortions in depth that align with image texture patterns, resulting in ripple-like structures on otherwise planar regions. In contrast, our method produces significantly smoother depth maps with cleaner discontinuities, indicating stronger geometric coherence.

We further visualize surface orientation using normal maps. Although our method can directly render normals, we derive normals from the rendered depth for all methods to ensure a fair comparison. Due to the extremely large field of view in ERP panoramas, depth-to-normal conversion can be sensitive to sampling density; nevertheless, normals from our reconstructions remain noticeably more stable on planar regions and are less correlated with appearance textures compared to prior methods. This qualitative behavior is consistent with our quantitative results in Table I.

<table><tr><td>Variant</td><td>DREâ</td><td>CIRâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Full (Ours)</td><td>0.0439</td><td>84.87</td><td>32.61</td><td>0.9195</td><td>0.0687</td></tr><tr><td>w/o  $\mathcal { L } _ { d n }$ </td><td>0.0397</td><td>85.25</td><td>32.70</td><td>0.9203</td><td>0.0669</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { j u m p } }$ </td><td>0.0692</td><td>76.87</td><td>32.75</td><td>0.9212</td><td>0.0668</td></tr><tr><td>w/o  $\mathcal { L } _ { d n }$  and  ${ \mathcal { L } } _ { \mathrm { j u m p } }$ </td><td>0.0577</td><td>77.66</td><td>32.84</td><td>0.9222</td><td>0.0648</td></tr></table>

TABLE III: Ablation study on OmniBlender [16].

<!-- image-->  
(a) Full

<!-- image-->

<!-- image-->  
(b) w/o $\mathcal { L } _ { d n }$

(c) w/o ${ \mathcal { L } } _ { \mathrm { j u m p } }$  
<!-- image-->  
(d) w/o $\mathcal { L } _ { d n }$ and ${ \mathcal { L } } _ { \mathrm { j u m p } }$  
Fig. 4: Qualitative ablation on OmniBlender [16].

Rotation robustness. Panoramic projection is highly nonlinear, and projection-based 3DGS variants often approximate it using screen-space local linearization. This approximation becomes less accurate in strongly distorted regions and can induce orientation-dependent rendering. We evaluate rotation robustness on OmniBlender by training with canonical poses and testing with additional random global rotations within Â±Î¸. Quantitative results are reported in Table II, and qualitative comparisons are shown in Fig. 3.

Projection-based methods, including ODGS and OmniGS, degrade progressively as the rotation magnitude increases. OmniGS is particularly sensitive: at $\theta { = } 9 0 ^ { \circ }$ , its PSNR drops by about 32% relative to the canonical setting, and LPIPS increases by about 74%. ODGS exhibits the same trend with a smaller magnitude, with roughly 16% PSNR degradation at Î¸=90â¦. As shown in Fig. 3, zoom-in regions become increasingly blurred, and fine textures near high latitudes can collapse into thin, needle-like streaks after rotation. These artifacts are consistent with distortion-sensitive screen-space linearization, which changes the effective sampling pattern with the panorama orientation.

In contrast, SPaGS and our method remain largely stable across large rotations. At Î¸=90â¦, our method reduces PSNR by only about 7% while maintaining consistently low LPIPS, and SPaGS stays essentially invariant. DRE and CIR vary less with rotation because they measure cross-view consistency without requiring ground-truth depth and apply outlier filtering, making them less sensitive to purely photometric degradations. Overall, the results indicate that evaluating Gaussian contributions in ray space yields more rotationstable panoramic rendering.

Ablation. Table III reports an ablation of $\mathcal { L } _ { d n }$ and ${ \mathcal { L } } _ { \mathrm { j u m p } }$

<table><tr><td colspan="4">Setting / Method DREâ CIRâ PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td colspan="6">OmniRob-UAV (full panorama)</td></tr><tr><td>SPaGS [3]</td><td>0.0715</td><td>73.44</td><td>35.28</td><td>0.9515</td><td>0.1908</td></tr><tr><td>Ours</td><td>0.0184</td><td>92.34</td><td>33.05</td><td>0.9297</td><td>0.0858</td></tr><tr><td colspan="6">OmniRob-Quadruped (annular camera)</td></tr><tr><td>SPaGS [3]</td><td>0.2614</td><td>19.49</td><td>19.90</td><td>0.7492</td><td>0.3272</td></tr><tr><td>Ours</td><td>0.1568</td><td>47.63</td><td>19.60</td><td>0.7517</td><td>0.2077</td></tr><tr><td colspan="6">Cropped-UAV (pseudo-annular)</td></tr><tr><td>SPaGS [3]</td><td>0.0770</td><td>83.20</td><td>35.24</td><td>0.9550</td><td>0.1098</td></tr><tr><td>Ours</td><td>0.0270</td><td>90.82</td><td>30.72</td><td>0.8879</td><td>0.1210</td></tr></table>

TABLE IV: Analysis on OmniRob. We compare SPaGS and ours on OmniRob-UAV, OmniRob-Quadruped, and the Cropped-UAV annular setting.

OmniRob-UAV  
<!-- image-->  
Fig. 5: Qualitative results on OmniRob annular panoramas. Rows show GT, SPaGS (RGB/normal), and Ours (RGB/normal).

Removing ${ \mathcal { L } } _ { \mathrm { j u m p } }$ leads to a clear degradation in geometric consistency, with higher DRE and lower CIR. In contrast, adding $\mathcal { L } _ { d n }$ can slightly reduce these proxy metrics, since DRE/CIR measures cross-view consistency rather than absolute correctness. We therefore also refer to Fig. 4, where $\mathcal { L } _ { d n }$ and ${ \mathcal { L } } _ { \mathrm { j u m p } }$ visibly smooth planar regions and suppress ripple-like artifacts.

Adaptation to annular panoramic cameras. Unlike stitched panoramas, annular observations can be obtained from a single omnidirectional capture and therefore avoid panorama stitching seams and associated photometric inconsistencies [30]. Moreover, annular images correspond to a different parameterization of omnidirectional measurements and thus provide a practically motivated stress test for robustness under re-parameterization. We evaluate our formulation on OmniRob under full-panorama, annular, and pseudoannular observations, as summarized in Table IV and Fig. 5. Both SPaGS and our method can be applied to OmniRob-UAV, OmniRob-Quadruped, and Cropped-UAV with only minor modifications, while exhibiting complementary behavior. SPaGS emphasizes pixel fidelity and achieves higher

<!-- image-->  
Fig. 6: Mesh extraction results on OmniBlender [16]. Our method reconstructs clean surfaces with fewer holes and reduced texture-induced artifacts.

PSNR and SSIM. In contrast, our method emphasizes perceptual quality and geometric consistency, achieving lower LPIPS and DRE with higher CIR. The Quadruped setting is more challenging due to capture conditions and artifacts introduced by annular unwrapping, so we additionally include Cropped-UAV as a controlled pseudo-annular setting to reduce nuisance factors.

## C. Implications for Embodied Intelligence

Our method produces geometrically consistent depth from omnidirectional observations, enabling mesh extraction for downstream embodied tasks. Although panoramic images inherently provide limited spatial resolution due to their large field of view, the reconstructed meshes preserve the overall scene structure and major geometric layouts. Fig. 6 visualizes representative meshes reconstructed. By reducing texture-induced ripple artifacts in depth, our method leads to smoother and more stable surface reconstructions compared to competing approaches. Such explicit 3D representations can directly support navigation, obstacle avoidance, and motion planning in robotic systems, highlighting the practical relevance of our approach for embodied applications.

## V. CONCLUSION

In this work, we have presented Spherical-GOF, a spherical ray-space GOF sampling and omnidirectional Gaussian rendering framework for ERP panoramas. By using an ERPaware footprinting strategy and sphere-metric-consistent geometric regularization, our method achieves competitive photometric quality while significantly improving multi-view geometric consistency, producing cleaner depth and more coherent normal maps with reduced sensitivity to highfrequency appearance textures, and remaining robust under large global panorama rotations. We further verified its generality on real-world robot-captured omnidirectional data from a panoramic UAV and a quadruped equipped with a ring-band camera.

Future work will explore higher-quality and more efficient geometric reconstruction under omnidirectional imaging, including improved geometry priors and faster spherical sampling/rendering strategies.

## REFERENCES

[1] S. Lee, J. Chung, J. Huh, and K. M. Lee, âODGS: 3D scene reconstruction from omnidirectional images with 3D gaussian splattings,â in Proc. NeurIPS, 2024, pp. 57 050â57 075.

[2] L. Li, H. Huang, S.-K. Yeung, and H. Cheng, âOmniGS: Fast radiance field reconstruction using omnidirectional gaussian splatting,â in Proc. WACV, 2025, pp. 2260â2268.

[3] J. Li, F. Hahlbohm, T. Scholz, M. Eisemann, J. Tauscher, and M. A. Magnor, âSPaGS: Fast and accurate 3D gaussian splatting for spherical panoramas,â in Computer Graphics Forum, 2025, p. e70171.

[4] D. Selvaratnam and D. Bazazian, â3D reconstruction in robotics: A comprehensive review,â Computers & Graphics, vol. 130, p. 104256, 2025.

[5] Y. Bao et al., â3D gaussian splatting: Survey, technologies, challenges, and opportunities,â IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 7, pp. 6832â6852, 2025.

[6] A. Melnik et al., âDigital twin generation from visual data: A survey,â arXiv preprint arXiv:2504.13159, 2025.

[7] R. Jin et al., âGS-Planner: A gaussian-splatting-based planning framework for active high-fidelity reconstruction,â in Proc. IROS, 2024, pp. 11 202â11 209.

[8] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in Proc. CVPR, 2016, pp. 4104â4113.

[9] P. Moulon, P. Monasse, R. Perrot, and R. Marlet, âOpenMVG: Open multiple view geometry,â in Proc. RRPR, 2016, pp. 60â74.

[10] H. Ai, Z. Cao, and L. Wang, âA survey of representation learning, optimization strategies, and applications for omnidirectional vision,â International Journal of Computer Vision, vol. 133, no. 8, pp. 4973â 5012, 2025.

[11] H. Huang and S.-K. Yeung, â360VO: Visual odometry using a single 360 camera,â in Proc. ICRA, 2022, pp. 5594â5600.

[12] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[13] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, â3D gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics (TOG), vol. 42, no. 4, pp. 1â14, 2023.

[14] T. MÃ¼ller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics primitives with a multiresolution hash encoding,â ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[15] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, âTensoRF: Tensorial radiance fields,â in Proc. ECCV, 2022, pp. 333â350.

[16] C. Choi, S. M. Kim, and Y. M. Kim, âBalanced spherical grid for egocentric view synthesis,â in Proc. CVPR, 2023, pp. 12 663â12 673.

[17] Z. Yu, T. Sattler, and A. Geiger, âGaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes,â ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp. 1â13, 2024.

[18] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2D gaussian splatting for geometrically accurate radiance fields,â in Proc. SIGGRAPH, 2024, p. 32.

[19] F. Hahlbohm et al., âEfficient perspective-correct 3D gaussian splatting using hybrid transparency,â in Computer Graphics Forum, 2025, p. e70014.

[20] H. Huang, Y. Chen, T. Zhang, and S.-K. Yeung, âReal-time omnidirectional roaming in large scale indoor scenes,â in Proc. SIGGRAPH Asia Technical Communications, 2022, pp. 1â5.

[21] S. Kulkarni, P. Yin, and S. Scherer, â360FusionNeRF: Panoramic neural radiance fields with joint guidance,â in Proc. IROS, 2023, pp. 7202â7209.

[22] S. Ito, N. Takama, K. Ito, H.-T. Chen, and T. Aoki, âErpGS: Equirectangular image rendering enhanced with 3D gaussian regularization,â in Proc. ICIP, 2025, pp. 2850â2855.

[23] J. Bai, L. Huang, J. Guo, W. Gong, Y. Li, and Y. Guo, â360-GS: Layout-guided panoramic gaussian splatting for indoor roaming,â in Proc. 3DV, 2025, pp. 1042â1053.

[24] Z. Shen, C. Lin, S. Huang, L. Nie, K. Liao, and Y. Zhao, âYou need a transition plane: Bridging continuous panoramic 3D reconstruction with perspective gaussian splatting,â arXiv preprint arXiv:2504.09062, 2025.

[25] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, âEWA splatting,â IEEE Transactions on Visualization & Computer Graphics, vol. 8, no. 03, pp. 223â238, 2002.

[26] T. Bertel, M. Yuan, R. Lindroos, and C. Richardt, âOmniPhotos: casual 360Â° VR photography,â ACM Transactions on Graphics (TOG), vol. 39, no. 6, pp. 1â12, 2020.

[27] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, 2004.

[28] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proc. CVPR, 2018, pp. 586â595.

[29] A. Krizhevsky, I. Sutskever, and G. E. Hinton, âImageNet classification with deep convolutional neural networks,â in Proc. NeurIPS, 2012, pp. 1097â1105.

[30] C. Shin, W. O. Cho, and S. J. Kim, âSeam360GS: Seamless 360Â° gaussian splatting from real-world omnidirectional images,â in Proc. ICCV, 2025, pp. 28 970â28 979.