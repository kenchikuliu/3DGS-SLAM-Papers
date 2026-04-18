# 6D Channel Knowledge Map Construction via Bidirectional Wireless Gaussian Splatting

Juncong Zhou, Chao Hu, Guanlin Wu, Zixiang Ren, Han Hu, Juyong Zhang, Rui Zhang, Fellow, IEEE, and Jie Xu, Fellow, IEEE

AbstractâThis paper investigates the construction of channel knowledge map (CKM) from sparse channel measurements. Different from conventional two-/three-dimensional (2D/3D) CKM approaches assuming fixed base station configurations, we present a six-dimensional (6D) CKM framework named bidirectional wireless Gaussian splatting (BiWGS), which is capable of modeling wireless channels across dynamic transmitter (Tx) and receiver (Rx) positions in 3D space. BiWGS uses Gaussian ellipsoids to represent virtual scatterer clusters and environmental obstacles in the wireless environment. By properly learning the bidirectional scattering patterns and complex attenuation profiles based on channel measurements, these ellipsoids inherently capture the electromagnetic transmission characteristics of wireless environments, thereby accurately modeling signal transmission under varying transceiver configurations. Experiment results show that BiWGS significantly outperforms classic multi-layer perception (MLP) for the construction of 6D channel power gain map with varying Tx-Rx positions, and achieves spatial spectrum prediction accuracy comparable to the state-of-theart wireless radiation field Gaussian splatting (WRF-GS) for 3D CKM construction. This validates the capability of the proposed BiWGS in accomplishing dimensional expansion of 6D CKM construction, without compromising fidelity.

Index TermsâChannel knowledge map (CKM), 6D CKM construction, bidirectional wireless Gaussian splatting.

## I. INTRODUCTION

Acquisition of accurate channel state information (CSI) is becoming increasingly important for resource allocation and beamforming optimization in wireless communication systems. This holds particular significance for future sixthgeneration (6G) networks with ultra-dense base station (BS) deployment, extremely large-scale antenna array (ELAA), and ultra-high bandwidth [1], [2]. Conventionally, real-time CSI acquisition is achieved through pilot-based channel estimation and limited feedback, which, however, may cause prohibitive signaling overhead [3]. Recently, channel knowledge map (CKM) has emerged as a promising solution to tackle this challenge [4]. CKM provides a priori channel knowledge (e.g., channel power gain [5], beam index [6], channel angle [7], and CSI), enabling environment-aware communication while reducing and even eliminating the requirements of real-time channel measurements. In particular, CKM can be represented in the forms of a spatial database, images, or neural network models, which constitutes a mapping relationship from wireless environments and spatial positions of transceivers to channel knowledge. CKM is envisioned to enable a wide range of environment-aware applications, including predictive communication, resource allocation, beam tracking, unmanned aerial vehicle (UAV) placement, as well as sensing and localization [8].

CKMs can be categorized as base station (BS)-to-any (B2X) and any-to-any (X2X) CKMs, depending on the input dimension of transceiver positions [4]. In particular, the B2X CKM utilizes the two-dimensional (2D) or three-dimensional (3D) position of mobile user as input, providing channel knowledge at that specific position relative to a fixed-position BS, thereby supporting BS-centric communications. By contrast, the X2X CKM exploits both the transmitter (Tx) and receiver (Rx) positions as input, providing the channel knowledge at varying Tx and Rx positions, which makes it suitable for X2X and device-to-device (D2D) communications. Depending on whether 2D or 3D space is considered, the X2X CKMs can be constructed in 4D (with 2D Tx/Rx positions) or 6D (with 3D Tx/Rx positions) formats, respectively. In particular, the 6D X2X CKM provides the most comprehensive wireless environment/channel information, which is crucial for solving complex tasks such as environment-aware BS deployment [9] and 3D trajectory planning for low-altitude UAVs [10].

The construction of CKM is essential for its practical implementation. In the literature, various methods for CKM construction have been proposed. The representative ones include ray tracing [11], [12], interpolation [13]â[15], deep learning [16]â[18], and wireless radiation field (WRF) [19]â [21]. First, the ray tracing approach enables high-accuracy channel reconstruction by leveraging physical environment information. It models electromagnetic (EM) waves as particles and simulates their interactions with the environment through reflection, scattering, and diffraction [11], [12]. Ray tracing achieves high accuracy when precise environmental geometry and material properties are known, but its practical application is hindered by the extreme computational complexity and the difficulty in acquiring complete environment knowledge a priori. Next, the interpolation approaches aim to reconstruct channel knowledge at unmeasured positions by using limited channel measurements at reference positions via techniques like K-nearest neighbors (KNN) [13], matrix completion [14], and Kriging interpolation [15], in which their spatial correlations and relative distances are utilized. However, the interpolation-based methods rely on the assumption of fixed Tx or Rx positions, and are highly dependent on the stationarity of the environment without accounting for the environmental geometric structure. Therefore, these methods are less effective in dynamic or complex scenarios, and are only suitable for constructing 2D/3D B2X CKMs but not applicable for 4D/6D X2X CKMs. Furthermore, deep learning approaches have also been used as an efficient approach for CKM construction due to their ability to learn complex nonlinear mappings from limited measurement data. A typical example is multi-layer perception (MLP) [16], which directly learns the channel features from the Tx-Rx position pair inputs to predict the channel power gain. However, the MLP method lacks the capacity to incorporate geometric information of the environment, resulting in significant performance degradation in complex environments. Beyond MLPs, deep learning-based image processing techniques have also been employed (see, e.g., [17], [18]). The authors in [17] proposed a convolutional neural network (CNN)-based method called RadioUnet, which uses the Tx positions and city maps as inputs to estimate channel gains at arbitrary Rx positions in the city, thereby constructing a channel gain map. Furthermore, the authors in [18] introduced a super-resolution (SR)-based CKM construction method, in which the SR residual network (SRResNet) is employed to recover a high-quality CKM image from sparse, low-resolution observation data. However, the image processing-based approaches are proposed for 2D B2X and 4D X2X CKMs, not applicable for 6D X2X CKMs of our interest.

Recently, the WRF approaches have emerged as a new approach for CKM construction. The emergence of WRF approaches is inspired by the recent advances of radiance field rendering techniques (especially the Neural Radiance Fields (NeRF) [22] and 3D Gaussian Splatting (3DGS) [23]) for 3D scene reconstruction in the computer graphics field. While NeRF implicitly represents the radiance field as an MLP that is trained from a few images to synthesize novelview images, 3DGS explicitly represents the radiance field as Gaussian ellipsoids colored via spherical harmonics (SH). In practice, NeRF is relatively time-consuming in both training and inference [24], while 3DGS can achieve high-resolution rendering at faster speeds through tile-based rendering. With the great success of NeRF and 3DGS in computer graphics and motivated by the fact that both light and wireless signals are EM waves, the NeRF and 3DGS techniques have recently been employed as effective WRF approaches for CKM construction. For instance, a NeRF-based WRF method named neural radiofrequency radiance fields (NeRF2) [19] was proposed for spatial spectrum prediction for a single-input multiple-output (SIMO) wireless communication system with a signal-antenna mobile user as Tx and a multi-antenna fixed-position BS as Rx. NeRF2 employs the 3D Tx position, ray direction with respect to Rx, and voxel position (position of sampled points along the transmission ray) as the inputs of MLP, which fits the amplitude attenuation and phase shifts of wireless signals in specific receive directions, thereby constructing a 3D B2X CKM. To reduce the computational complexity of NeRF2, a follow-up work [20] developed Neural Wireless Radiance Fields (NeWRF), which estimates the angle of arrival (AOA) of received signals via spatial signal classification algorithms such as multiple signal classification (MUSIC), thus minimizing the number of generated rays for tracing and accordingly reducing the computation complexity for training and inference. On the other hand, building upon 3DGS, wireless radiation field Gaussian splatting (WRF-GS) was proposed in [21] to use 3D Gaussian ellipsoids to model the WRF, which capture the electromagnetic transmission characteristics from received signals, thus improving the accuracy of 3D CKM construction at enhanced computational efficiency. Despite the progress in using NeRF and 3DGS for CKM construction, these prior works focused on the scenario when the Rx position is static (e.g., when Rx is a BS), thus making them suitable for 3D B2X CKMs, but not for 6D X2X CKMs.

Different from prior WRF works focusing on 3D B2X CKM construction, in this paper we consider the construction of 6D X2X CKM for a SIMO system with varying 3D Tx-Rx positions. In particular, we propose a new framework named Bidirectional Wireless Gaussian Splatting (BiWGS), which uses Gaussian ellipsoids to represent virtual scatterer clusters as well as environmental obstacles. BiWGS is inspired by the Bidirectional Gaussian splatting (BiGS) algorithm [25] in the computer graphics field, which is a 3DGS-based design for 3D scene reconstruction under varying illumination. Different from conventional 3DGS, BiGS uses bidirectional spherical harmonics (BSH) function to model the optical bidirectional scattering pattern of Gaussian ellipsoids based on the AOA and angle of departure (AOD), capturing the dynamic illumination. Motivated by this, the proposed BiWGS incorporates the idea of BiGS into CKM construction, in which the parameters of Gaussian ellipsoids and BSHs are properly learned to capture the bidirectional scattering patterns and complex attenuation profiles under different Tx-Rx position pairs, thereby enabling the efficient construction of 6D CKM.

The main results of this paper are listed as follows.

â¢ First, we adopt a scatterer-cluster-based channel model to facilitate the BiWGS design. This model represents the wireless channel between any Tx-Rx pair as a multipath channel comprising multiple scattering paths and a potential line-of-sight (LOS) path, each subject to complex attenuations caused by environmental obstacles along the paths. Building on this channel modeling, we propose the BiWGS framework, which leverages Gaussian ellipsoids to serve as both virtual scatterer clusters and environmental obstacles, effectively representing the wireless environment. Notably, each Gaussian ellipsoid is characterized by a bidirectional scattering profile, enabling the construction of 6D X2X CKMs.

â¢ Then, we adopt the Gaussian ellipsoid representation to model the attenuation of SIMO channel paths. Initially, the BSH function is used for each Gaussian ellipsoid to represent the bidirectional complex scattering coefficient of its associated virtual scatterer clusters. Subsequently, for each channel path, wireless splatting is employed to compute the complex attenuation induced by the obstruction of each Gaussian ellipsoid, and these individual attenuations are aggregated via wireless rendering to reconstruct the attenuation over the path. By combining the channel paths from all possible AOAs at Rx, we obtain the SIMO channel vector between any Tx-Rx position pair.

â¢ In the training process, we use the weighted sum of the spatial spectrum prediction loss and the channel power gain loss as the loss function of BiWGS, in which the spatial spectrum represents the angular power distribution of the wireless channel. In addition, an adaptive density control strategy is applied during the backward propagation for training, dynamically adjusting the number of Gaussian ellipsoids and their sizes based on the gradient magnitudes.

â¢ Finally, we provide experiment results to validate the effectiveness of our method in CKM construction. The proposed BiWGS demonstrates significant performance gains over the benchmark MLP scheme in constructing 6D channel power gain maps across varying Tx-Rx pairs. Furthermore, in 3D B2X CKM construction, BiWGS achieves spatial spectrum prediction accuracy on a par with the state-of-the-art (SOTA) benchmark WRF-GS. These results demonstrate that BiWGS effectively achieves dimensional expansion from 3D B2X CKM to 6D X2X CKM, while maintaining high fidelity.

The remainder of this paper is organized as follows. Section II reviews the basics of BiGS for 3D scene reconstruction. Section III presents the BiWGS framework for 6D X2X CKM construction. Section IV presents experiment results. Finally, Section V concludes the paper.

Notations: Vectors and matrices are denoted by boldface lowercase and upper-case letters, respectively. For any vector or matrix, $( \cdot ) ^ { T }$ and $( \cdot ) ^ { H }$ refer to its transpose and conjugate transpose, respectively. $\pmb { A } \otimes \pmb { B }$ represents the Kronecker product of matrices A and $B . \ \Vert \cdot \Vert _ { 2 }$ represents the Euclidean norm. $\mathbb { R } ^ { m \times n }$ and $\mathbb { C } ^ { m \times n }$ represent the spaces of real and complex matrices with dimension $m \times n ,$ respectively. The imaginary unit is denoted as $j = { \sqrt { - 1 } } . \ | \ \cdot \ |$ denotes the amplitude of a complex number.

## II. REVIEW OF BIGS FOR 3D SCENE RECONSTRUCTION WITH DYNAMIC ILLUMINATION

Before we proceed to present our proposed BiWGS method, this section provides a brief review of BiGS for 3D scene reconstruction with varying illumination in the computer graphics field. 3DGS has demonstrated considerable success in reconstructing 3D scenes through collections of 3D Gaussian ellipsoids under static illumination. BiGS is an extension of the conventional 3DGS method. It reconstructs 3D scenes from images at limited input views and synthesizes images from novel views under varying illumination by learning the lightdependent color pattern of Gaussian ellipsoids.

## A. Bidirectional Gaussian Ellipsoid Representation

First, we introduce the basic principle of BiGS scene representation. The modeling of Gaussian ellipsoids in BiGS follows the 3DGS methodology: It represents scenes via a set of anisotropic Gaussian ellipsoids, which are determined by 3D Gaussian distributions. These ellipsoids characterize spatially varying radiance fields in the environment, thus capturing both the geometric structure and visual appearance of a scene without the need for normal estimation [23]. Moreover, BiGS incorporates the optical scattering function for each Gaussian ellipsoid to characterize its light-dependent color patterns under dynamic illumination, enabling scene representation under dynamic lighting conditions, a capability notably absent in the original 3DGS methodology. During the rendering process, these ellipsoids are projected onto the image plane via splatting for an arbitrary view. The final synthesized image is subsequently generated by employing the optical rendering equation, which aggregates the radiance contributions of all splatted 2D Gaussian ellipsoids per pixel through Î±-blending.

Next, we explain the Gaussian ellipsoid representation for 3D scene in detail. For every Gaussian ellipsoid, its corresponding 3D Gaussian distribution is expressed as

$$
G ( \pmb { x } ) = \exp \bigg \{ - \frac { 1 } { 2 } ( \pmb { x } - \pmb { \mu } ) ^ { T } \pmb { \Sigma } ^ { - 1 } ( \pmb { x } - \pmb { \mu } ) \bigg \} ,\tag{1}
$$

where $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ denotes the mean vector of ellipsoid, $\pmb { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ denotes the covariance matrix of ellipsoid governing its spatial extent and orientation, $\textbf { \textit { x } } \in \ \mathbb { R } ^ { 3 }$ denotes the 3D position, and the normalization constant of the Gaussian distribution is omitted. Moreover, the covariance matrix Î£ in (1) is determined by scaling matrix $\pmb { S } \in \mathbb { R } ^ { 3 \times 3 }$ and rotation matrix $\pmb { R } \in \mathbb { R } ^ { 3 \times 3 }$ , i.e.,

$$
\pmb { \Sigma } = \pmb { R } \pmb { S } \pmb { S } ^ { T } \pmb { R } ^ { T } .\tag{2}
$$

BiGS models the light-dependent color pattern of each Gaussian ellipsoid through three key components. The first component is incident radiance $\begin{array} { r l } { l ( \theta ^ { \prime } , \phi ^ { \prime } ) } & { { } = } \end{array}$ $[ l _ { \mathrm { R } } ( \theta ^ { \prime } , \phi ^ { \prime } ) , l _ { \mathrm { G } } ( \theta ^ { \prime } , \phi ^ { \prime } ) , l _ { \mathrm { B } } ( \theta ^ { \prime } , \phi ^ { \prime } ) ] ^ { T }$ , which quantifies the amount of radiance ellipsoid received from the light source at the incident direction $( \theta ^ { \prime } , \phi ^ { \prime } )$ , with $\theta ^ { \prime }$ and $\phi ^ { \prime }$ denoting the incident elevation and azimuth angles, respectively. The second component is optical scattering function $\begin{array} { r l r } { { \bf f } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) } & { = } & { [ f _ { \mathrm { R } } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) , f _ { \mathrm { G } } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) } \end{array}$ , $f _ { \mathrm { B } } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) ] ^ { T }$ which characterizes the bidirectional scattering properties of ellipsoid. Here, Î¸ and $\phi$ denote the scattering elevation and azimuth angles, respectively. The third component is RGB color $\begin{array} { l } { \displaystyle \boldsymbol { c } ( \theta , \phi ) = [ c _ { \mathrm { R } } ( \theta , \phi ) } \end{array}$ $c _ { \mathrm { G } } ( \theta , \phi ) , c _ { \mathrm { B } } ( \theta , \phi ) ] ^ { T }$ , which encodes scattered radiance at viewing direction $( \theta , \phi )$ . Here, the subscripts R, G, and B denote the red, blue, and green components of RGB color, respectively. The relationship among these components is given by

$$
c _ { i } ( \theta , \phi ) = \int _ { S ^ { 2 } } l _ { i } ( \theta ^ { \prime } , \phi ^ { \prime } ) f _ { i } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) \mathrm { d } \Omega ^ { \prime } , i \in \{ \mathrm { R } , \mathrm { G } , \mathrm { B } \} ,\tag{3}
$$

where $S ^ { 2 }$ is unit sphere, and $\mathrm { d } \Omega ^ { \prime } ~ = ~ \sin ( \theta ^ { \prime } ) \mathrm { d } \theta ^ { \prime } \mathrm { d } \phi ^ { \prime }$ . (3) quantifies the angular dependence of lightâobject interaction.

This dependence, determined by the relative configuration of the light source and the object together with the objectâs material parameters, is captured by the optical scattering function ${ \pmb f } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } )$ , which encodes the bidirectional scattering response of Gaussian ellipsoids within the environment. Next, we specify the concrete form of ${ \pmb f } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } )$ , which is further decomposed into an angle-independent term ${ \boldsymbol \rho } \in \mathbb { R } ^ { 3 }$ and an angle-dependent term $s ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) \in \mathbb { R } ^ { 3 }$ , i.e.,

$$
\pmb { f } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) = \pmb { \rho } + \pmb { s } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) ,\tag{4}
$$

where $\rho$ captures the angle-independent material properties of an object (e.g., albedo).

## B. Optical Rendering

Furthermore, we elaborate on the optical rendering process, which comprises two sequential stages: splatting of 3D Gaussian ellipsoids and Î±-blending on the image plane.

1) Splatting: After the RGB colors of all Gaussian ellipsoids are computed via (3), the rendering of images initiates with splatting 3D Gaussians ellipsoids $G ( \pmb { x } )$ onto the image plane, yielding the corresponding 2D Gaussian ellipsoids $G ^ { \prime } ( \pmb { x } _ { \mathrm { 2 D } } )$ . The splatting stage consists of two core transformations, i.e., view transformation and perspective projection. The view transformation converts the scene from absolute world coordinates to camera coordinates through an affine mapping. Subsequently, the perspective projection maps the 3D Gaussian ellipsoids onto the image plane, obtaining 2D Gaussian distributions that represent their projected forms. The mean vector and covariance matrix of one Gaussian ellipsoid after splatting are respectively expressed as

$$
\begin{array} { l } { { \pmb { \mu } ^ { \prime } = \varphi ( \pmb { W } \pmb { \mu } + \pmb { d } ) , } } \\ { { \pmb { \Sigma } ^ { \prime } = \pmb { J } \pmb { W } \pmb { \Sigma } \pmb { W } ^ { T } \pmb { J } ^ { T } , } } \end{array}\tag{5}
$$

where W and d denote the rotation and translation transformation, respectively, $W \mu + d$ denotes the whole view transformation, function $\varphi ( \cdot ) : \mathbb { R } ^ { 3 } \to \mathbb { R } ^ { 3 }$ denotes the non-linear perspective projection, and J denotes the Jacobian matrix of $\varphi ( \cdot )$ denoting the affine approximation of the perspective projection.

Moreover, the 2D mean vector $\mu _ { \mathrm { 2 D } } \in \mathbb { R } ^ { 2 }$ is obtained by truncating the third row of projected mean vector $\mu ^ { \prime }$ and the 2D covariance matrix $\bar { \mathbf { Z } } _ { \mathrm { 2 D } } \bar { \in } \bar { \mathbb { R } } ^ { 2 \times 2 }$ is obtained by truncating the third row and column of the projected covariance matrix $\Sigma ^ { \prime }$ . Here, the perspective projection with 3D ellipsoids is to accurately characterize the transformation of objects following the optical rule that nearer objects are larger and farther objects are smaller, while the truncation is to keep consistent with the 2D image plane since the depth information has been embodied in the transformed parameters. Finally, we yield the 2D Gaussian distribution after splatting as

$$
G ^ { \prime } ( \pmb { x } _ { \mathrm { 2 D } } ) = \exp \bigg \{ - \frac { 1 } { 2 } ( \pmb { x } _ { \mathrm { 2 D } } - \pmb { \mu } _ { \mathrm { 2 D } } ) ^ { T } \pmb { \Sigma } _ { \mathrm { 2 D } } ^ { - 1 } ( \pmb { x } _ { \mathrm { 2 D } } - \pmb { \mu } _ { \mathrm { 2 D } } ) \bigg \} ,\tag{6}
$$

where $\scriptstyle 2 \mathrm { D }$ denotes the 2D position on the image plane.

2) Î±-Blending: Following the splatting stage, all 2D Gaussian ellipsoids are sorted according to their depth before projection (distance to the image plane). Subsequently, the optical rendering is employed to realize Î±-blending, thereby computing the RGB color for each pixel on the image plane. For pixel $^ { O , }$ the rendering equation is expressed as

$$
c _ { o } ^ { \mathrm { p i x e l } } = \sum _ { i = 1 } ^ { N } c _ { i } ( \theta , \phi ) \alpha _ { i } \prod _ { k = 1 } ^ { i - 1 } ( 1 - \alpha _ { k } ) ,\tag{7}
$$

where $\alpha _ { i }$ denotes the opacity of the i-th sorted ellipsoid related to the pixel, $c _ { i } ( \theta , \phi )$ denotes ellipsoidâs RGB color obtained in $( 3 ) .$ , and $\boldsymbol { c } _ { o } ^ { \mathrm { p i x e l } }$ denotes the rendered RGB color of pixel $o .$

The opacity in (7) varies spatially between the center and the periphery of the projected 2D ellipsoid. This spatial variation is modeled by

$$
\alpha _ { i } = \alpha _ { i } ^ { \mathrm { m a x } } G ^ { \prime } ( { \pmb x } _ { o } ^ { \mathrm { p i x e l } } ) ,\tag{8}
$$

where $\alpha _ { i } ^ { \mathrm { { m a x } } }$ denotes the maximum opacity of the i-th ellipsoid, $\pmb { x } _ { o } ^ { \mathrm { p i x e l } }$ represents the 2D position of pixel o to be rendered on the image plane, and $G ^ { \prime } ( \cdot )$ is defined in (6). (8) governs the spatially varying opacity distribution within Gaussian ellipsoids, exhibiting a monotonically decreasing opacity profile with increasing radial distance from the ellipsoid center. The rendering process in (7) is completed once the colors of all pixels have been fully computed.

C. Representation and Property of Bidirectional Scattering Pattern

It remains to determine the bidirectional scattering component $s ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } )$ in (4). In particular, $s ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } )$ is represented and fit by the following BSH function:

$$
s ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) = \sum _ { i = 1 } ^ { ( D + 1 ) ^ { 2 } } \sum _ { k = 1 } ^ { ( D + 1 ) ^ { 2 } } { \bf { a } } _ { i k } { \bf { y } } _ { k } ( \theta ^ { \prime } , \phi ^ { \prime } ) { \bf { y } } _ { i } ( \theta , \phi ) ,\tag{9}
$$

where $\mathbf { a } _ { i k } \in \mathbb { R } ^ { 3 }$ denotes the learnable BSH coefficients, D denotes the SH degree, ${ \bf \it y } _ { i } ( \theta , \phi )$ denotes the i-th element of the SH basis vector consisting of all possible SH basis determined by the SH degree, i.e.,

$$
\begin{array} { r l } & { \pmb { y } ( \theta , \phi ) = \bigl [ y _ { 0 , 0 } ( \theta , \phi ) , y _ { 1 , - 1 } ( \theta , \phi ) , y _ { 1 , 0 } ( \theta , \phi ) , y _ { 1 , 1 } ( \theta , \phi ) } \\ & { , \ldots , y _ { D , - D } ( \theta , \phi ) , \ldots , y _ { D , 0 } ( \theta , \phi ) , \ldots , y _ { D , D } ( \theta , \phi ) \bigr ] , } \end{array}\tag{10}
$$

where $y _ { n , i } ( \theta , \phi )$ denotes the SH basis determined by the associated Legendre polynomials [26], i.e.,

$$
y _ { n , i } ( \theta , \phi ) = \sqrt { \frac { 2 n + 1 } { 2 \pi } \frac { ( n - i ) ! } { ( n + i ) ! } } H _ { n } ^ { i } ( \cos \theta ) \cos ( i \phi ) , i = 0 , 1 , \ldots , n ,\tag{11a}
$$

$$
y _ { n , i } ( \theta , \phi ) = \sqrt { \frac { 2 n + 1 } { 2 \pi } \frac { ( n - i ) ! } { ( n + i ) ! } } H _ { n } ^ { i } ( \cos \theta ) \sin ( i \phi ) , i = - n , \ldots , - 1 ,\tag{11b}
$$

$$
H _ { n } ^ { i } ( t ) = \frac { ( - 1 ) ^ { i } } { 2 ^ { n } n ! } ( 1 - t ^ { 2 } ) ^ { i / 2 } \frac { d ^ { n + i } } { d t ^ { n + i } } ( t ^ { 2 } - 1 ) ^ { n } , \quad i = 0 , 1 , \dots , n ,\tag{11c}
$$

$$
H _ { n } ^ { - i } ( t ) = ( - 1 ) ^ { i } \frac { ( n - i ) ! } { ( n + i ) ! } H _ { n } ^ { i } ( t ) , \quad i = 1 , \ldots , n ,\tag{11d}
$$

for any $n \in \{ 1 , \ldots , D \}$ , where $H _ { n } ^ { i } ( t )$ and $H _ { n } ^ { - i } ( t )$ denote the associated Legendre polynomials.

Furthermore, the optical scattering function needs to satisfy the reciprocity property in order to be physically meaningful. Specifically, when the incident direction and scattering direction switch, the value of the optical scattering function should be the same. Towards this end, we enforce the bidirectional scattering component $s ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } )$ to satisfy such reciprocity, which is expressed as

$$
\begin{array} { r } { s ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) = s ( \pi - \theta ^ { \prime } , \pi + \phi ^ { \prime } , \pi - \theta , \pi + \phi ) , \forall \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } . } \end{array}\tag{12}
$$

## D. Training Process

During the training process, an adaptive density control strategy governs the number of ellipsoids and their sizes through cloning, splitting, and pruning operations. Cloning inserts new ellipsoids in under-reconstructed regions, while splitting subdivides oversized ellipsoids in over-reconstructed regions. Moreover, ellipsoids exhibiting maximum opacity below a predefined threshold undergo pruning due to their negligible contribution to the scene representation. The three adaptive density control operations are executed periodically at a fixed interval to maintain the fidelity of scene representation. Notably, the BiGS framework uses the training result of the 3DGS [23] under static illumination as the initialization of the BiGS model, and the adaptive density control strategy is only employed during this initialization phase.

## III. BIWGS FOR 6D X2X CKM CONSTRUCTION

Motivated by BiGS that represents the radiance field via bidirectional Gaussian ellipsoids to capture dynamic illumination conditions, we present BiWGS, a novel method to construct X2X 6D CKM for arbitrary Tx-Rx position pairs.

For ease of exposition, we consider a narrowband SIMO1 wireless system consisting of one Tx and one Rx, in which the Tx is equipped with a single transmit antenna and the Rx is deployed with a uniform planar array (UPA) of $N = N _ { v } \times$ $N _ { h }$ antennas. It is assumed that the Rxâs UPA is horizontally aligned with the ground plane. It is also assumed that the Rxâs reception domain is a hemisphere shown in Fig. 1, similarly as in [19].

We consider the frequency-flat fading channel model in the narrowband scenario, in which the wireless channel from Tx to Rx corresponds to the combination of multiple channel paths including reflection, scattering, and diffraction. Under this setup, we are interested in characterizing the X2X 6D CKM, which provides a function mapping from the 3D Tx-Rx position pair (a 6D parameter) to the channel knowledge. In particular, we consider the CSI of the N Ã 1 channel vector as the channel knowledge of interest. As such, the mapping via CKM is described as

$$
\begin{array} { r } { \pmb { h } = q _ { E } ( \pmb { p } _ { \mathrm { t } } , \pmb { p } _ { \mathrm { r } } ) , } \end{array}\tag{13}
$$

where $p _ { \mathrm { t } } , p _ { \mathrm { r } } \ \in \ \mathbb { R } ^ { 3 }$ denote the 3D Tx and Rx positions, respectively, the subscript E denotes the wireless environment comprising both the geometric structure of the physical scene and the electromagnetic properties such as permittivity and conductivity, $\pmb { h } \in \mathbb { C } ^ { N \times 1 }$ denotes the channel vector, $q _ { E } ( \cdot )$ denotes the CKM or the mapping relationship that maps from the Tx-Rx position pair to the channel vector. For simplicity, we assume that $q _ { E } ( \cdot )$ is time-invariant, which is valid for static environment or the dominant static component of time-varying environment.

<!-- image-->

Fig. 1. Rxâs hemispherical reception domain.  
<!-- image-->  
Fig. 2. Illustration of one ellipsoid representing a scatterer cluster. Each ellipsoid is seen as one scatterer cluster consisting of multiple scattering paths at different received angles.

In the 6D CKM construction problem, we aim to find the CKM or the mapping function $q _ { E } ( \cdot )$ based on historical channel measurements. Supposing that there are K channel measurements in the training set, including the Tx-Rx position pair $\tilde { p } _ { \mathrm { t } , i } , \tilde { p } _ { \mathrm { r } , i }$ and its related channel vector $\tilde { h } _ { i }$ . The set of channel measurements or the training set is given by

$$
\mathcal { T } = \Big \{ ( \tilde { p } _ { \mathrm { t } , i } , \tilde { p } _ { \mathrm { r } , i } , \tilde { h } _ { i } ) , i = 1 , \dots , K \Big \} .\tag{14}
$$

The CKM construction problem is thus formulated as learning the mapping function $q _ { E } ( \cdot )$ from the training set T .

Remark 4.1: In general, the mapping function $q _ { E } ( \cdot )$ of CKM can be represented in different forms depending on the adopted reconstruction methods. For instance, for interpolation-based methods, the CKM is constructed as a spatial database; for image processing-based methods, the CKM is constructed as an image; while for NeRF-based methods, the CKM is constructed as a neural network. By contrast, this paper represents the CKM via a set of bidirectional Gaussian ellipsoids.

Remark 4.2: Notice that conventional WRF-based designs like NeRF2 [19] and WRF-GS [21] are only applicable for 3D B2X CKM with fixed ${ p } _ { \mathrm { r } }$ , but not applicable for 6D X2X CKM in (13) of our interest. We will propose BiWGS to construct 6D X2X CKM with varying pt and ${ p } _ { \mathrm { r } }$ as inputs.

A. Scatterer Clusters based Channel Modeling and Gaussian Ellipsoid Representation

To facilitate our BiWGS design, this subsection presents a scatterer-cluster-based channel model and represents the wireless channel through bidirectional Gaussian ellipsoids. Motivated by the widely-adopted channel models considering scatterer clusters [27], we consider a scatterer-cluster-based channel model, which consists of a number of S scatterer clustersâeach contributing multiple scattering pathsâand one direct path2:

$$
\begin{array} { r l } & { \displaystyle h = \sum _ { i = 1 } ^ { S } \sum _ { k = 1 } ^ { \nu _ { i } } h _ { i , k } ^ { s } + q _ { d } h ^ { L } } \\ & { \displaystyle = \sum _ { \underbrace { i = 1 \ k = 1 } } ^ { S } b ( \theta _ { i , k } , \phi _ { i , k } ) \frac { \lambda \Gamma _ { i , k } ( \theta _ { i , k } , \phi _ { i , k } , \theta _ { i , k } ^ { \prime } , \phi _ { i , k } ^ { \prime } ) } { \big ( 4 \pi \big ) ^ { 3 / 2 } d _ { \mathrm { \tiny { d } } , i , k } d _ { \mathrm { \tiny { r } } , i , k } } \Theta _ { i , k } e ^ { - j \frac { 2 \pi } { \lambda } \big ( d _ { \mathrm { \tiny { d } } , i , k } + d _ { \mathrm { \tiny { r } } , i , k } \big ) } } \\ & { \displaystyle + \underbrace { q _ { d } b ( \theta _ { L } , \phi _ { L } ) \frac { \lambda } { 4 \pi d _ { L } } \Theta _ { L } e ^ { - j \frac { 2 \pi } { \lambda } d _ { L } } } _ { \mathrm { D i r e t a t h } } . } \end{array}\tag{15}
$$

In (15), Î¨i denotes the number of scattering paths of each scatterer cluster i, $\pmb { h } _ { i , k } ^ { s }$ denotes the channel vector of the $( i , k ) .$ th scattering path, $\ddot { h } ^ { L }$ denotes the channel vector of direct channel path, Î» denotes the wavelength, $\theta _ { i , k } , \phi _ { i , k } , \theta _ { i , k } ^ { \prime }$ , and $\phi _ { i , k } ^ { \prime }$ denote the scattering elevation angle, scattering azimuth angle, incident elevation angle, and incident azimuth angle of the (i, k)-th scattering path, respectively, $d _ { \mathrm { t } , i , k }$ and $d _ { \mathrm { r } , i , k }$ denote the distance to Tx and Rx in the (i, k)-th scattering path, respectively, $\Theta _ { i , k } \in \mathbb { C }$ denotes the complex attenuation coefficient representing extra amplitude attenuation and phase shifting induced by the obstruction along the (i, k)-th scattering path. Furthermore, $\Gamma ( \theta _ { i , k } , \phi _ { i , k } , \theta _ { i , k } ^ { \prime } , \phi _ { i , k } ^ { \prime } ) \in \mathbb { C }$ denotes the bidirectional complex scattering coefficient, denoting the change of amplitude and phase caused by scattering. Notably, $\Gamma ( \theta _ { i , k } , \phi _ { i , k } , \theta _ { i , k } ^ { \prime } , \phi _ { i , k } ^ { \prime } )$ is fundamentally related to bi-static radar cross section, which characterizes the bidirectional angular dependence of scattering on both incident and scattering angles [28]. For the direct path, $\Theta _ { L } ~ \in ~ \mathbb { C }$ is a complex attenuation coefficient caused by obstruction along the direct path, $\theta _ { L }$ and $\phi _ { L }$ denote the elevation and azimuth angles of the direct path, respectively, $d _ { L }$ denotes the distance between Tx and Rx. Notably, the LOS path is a special case of the direct path without any obstruction. $q _ { d }$ is a 0-1 indicator denoting if a direct path exists (Tx lies within the Rxâs hemispherical reception domain, as illustrated in Fig. 1), which is given by

$$
q _ { d } = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { T x ~ i s ~ n o t ~ w i t h i n ~ t h e ~ R x ' s ~ r e c e p t i o n ~ d o m a i n } } , } \\ { 1 , } & { { \mathrm { T x ~ i s ~ w i t h i n ~ t h e ~ R x ' s ~ r e c e p t i o n ~ d o m a i n } } . } \end{array} \right. }\tag{16}
$$

In addition, $b ( \cdot )$ denotes the steering vector. For the UPA, we have

$$
\begin{array} { l } { { b _ { v } ( \theta _ { i , k } ) = \displaystyle \frac { 1 } { N _ { v } } [ 1 , e ^ { j 2 \pi \frac { d v } { \lambda } \cos \theta _ { i , k } } , \dots , e ^ { j 2 \pi \frac { d v } { \lambda } ( N _ { v } - 1 ) \cos \theta _ { i , k } } ] ^ { T } , } } \\ { { b _ { h } ( \theta _ { i , k } , \phi _ { i , k } ) } } \\ { { = \displaystyle \frac { 1 } { N _ { h } } [ 1 , e ^ { j 2 \pi \frac { d _ { h } } { \lambda } \sin \theta _ { i , k } \sin \phi _ { i , k } } , \dots , e ^ { j 2 \pi \frac { d _ { h } } { \lambda } ( N _ { h } - 1 ) \sin \theta _ { i , k } \sin \phi _ { i , k } } ] ^ { T } , } } \\ { { b ( \theta _ { i , k } , \phi _ { i , k } ) = b _ { v } ( \theta _ { i , k } ) \otimes b _ { h } ( \theta _ { i , k } , \phi _ { i , k } ) , } } \end{array}\tag{17}
$$

where $d _ { h }$ and $d _ { v }$ represent the horizontal and vertical spacing between two adjacent antennas, respectively.

Note that in (15), S is a parameter depending on the environment. The reflection and diffraction can be equivalently represented by a cluster of scatterers. Furthermore, in (15), we only consider the case with one single hop of scattering by ignoring the multi-hop scattering. This is consistent with channel models in [29]â[31] to facilitate our proposed BiWGS design3.

In the BiWGS design, we employ Gaussian ellipsoids as virtual scatterer clusters corresponding to one or more scattering paths [32], as shown in Fig. 2. To represent the wireless channel via Gaussian ellipsoids, we first reformulate the channel model in (15), by expressing it as the combination of scattering signal paths from every possible AoA of Rx, i.e.,

$$
\begin{array} { l } { { \displaystyle h = \sum _ { i = 1 } ^ { S } \sum _ { k = 1 } ^ { \Psi _ { i } } h _ { i , k } ^ { s } + q _ { d } h ^ { L } } } \\ { ~ = \displaystyle \int _ { - \pi / 2 } ^ { \pi / 2 } \int _ { 0 } ^ { \pi } h ^ { s } ( \theta , \phi ) \mathrm { d } \theta \mathrm { d } \phi + q _ { d } h ^ { L } , }  \\ { ~ \approx \displaystyle \sum _ { \phi \in \mathbb { Z } } \sum _ { \theta \in \mathcal { V } } h ^ { s } ( \theta , \phi ) + q _ { d } h ^ { L } , }  \end{array}\tag{18}
$$

where $h ^ { s } ( \theta , \phi )$ denotes the scattering channel vector related to the AOA (Î¸, Ï), and $\mathcal { Z }$ and V denote the set of sampled azimuth and elevation angles, respectively, by discretization of AOAs at the Rx with a specified angular resolution (determined by the Rxâs reception domain in Fig. 1). Note that for each sampled ray along the angle $( \theta , \phi )$ , it may penetrate multiple Gaussian ellipsoids, each of which results in a distinct scattering path associated with that ellipsoid. As such, multiple rays may traverse the same ellipsoid to generate multiple resolvable scattering paths. Let Er(Î¸, Ï) denote the set of ellipsoids that are penetrated by the sampled ray related to AOA (Î¸, Ï). The channel model in (15) is reformulated as

$$
\begin{array} { l } { { \displaystyle h \approx \sum _ { \phi \in \mathcal { Z } } \sum _ { \theta \in \mathcal { V } } \sum _ { m \in \mathcal { E } _ { \mathrm { r } } ( \theta , \phi ) } \bigg [ b ( \theta , \phi ) \frac { \lambda \Gamma _ { m } \left( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } \right) } { ( 4 \pi ) ^ { 3 / 2 } d _ { \mathrm { t } , m } d _ { \mathrm { r } , m } } } } \\ { { \displaystyle \quad \cdot \Theta _ { m } ( \theta , \phi , p _ { \mathrm { t } } , p _ { \mathrm { r } } , p _ { m } ) e ^ { - j \frac { 2 \pi } { \lambda } ( d _ { \mathrm { t } , m } + d _ { \mathrm { r } , m } ) } \bigg ] } } \\ { { \displaystyle \quad + q _ { d } b ( \theta _ { L } , \phi _ { L } ) \frac { \lambda } { 4 \pi d _ { L } } \Theta _ { L } ( p _ { \mathrm { t } } , p _ { \mathrm { r } } ) e ^ { - j \frac { 2 \pi } { \lambda } d _ { L } } , } } \end{array}\tag{19}
$$

<!-- image-->  
Fig. 3. Illustration of Tx-side and Rx-side wireless splatting. T1, T2, and T3 represent three Tx-ellipsoid paths associated with three ellipsoids, respectively. ${ \bf R } _ { 1 }$ and $\mathrm { R _ { 2 } }$ represent two different sampled rays of Rx, respectively.

where ${ \pmb p } _ { m }$ denotes the position of ellipsoid m, $\Theta _ { m } ( \theta , \phi , p _ { \mathrm { t } } , p _ { \mathrm { r } } , { p _ { m } } )$ denotes the complex attenuation, which is determined by the AOA, position of ellipsoid m, Tx position, and Rx position. Under our Gaussian ellipsoid representation, the attenuation $\Theta _ { m } \mathopen { } \mathclose \bgroup \left( \theta , \phi , p _ { \mathrm { t } } , p _ { \mathrm { r } } , \pmb { p } _ { m } \aftergroup \egroup \right)$ is caused by the obstruction related to ellipsoid m at AOA (Î¸, Ï) of Rx.

In particular, we define the 3D Gaussian ellipsoids similarly as in (1) and (2), which capture both scattering behavior and obstruction attenuation during the signal transmission. In the following, we use Gaussian ellipsoids to represent the complex attenuation and bidirectional complex scattering coefficient. Specifically, $\Theta _ { m } \mathopen { } \mathclose \bgroup \left( \theta , \phi , p _ { \mathrm { t } } , p _ { \mathrm { r } } , \pmb { p } _ { m } \aftergroup \egroup \right)$ will be determined by wireless splatting and wireless rendering, and $\Gamma _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } )$ will be represented by BSH function, as will be elaborated in detail next.

B. Representation of $\Theta _ { m } ( \theta , \phi , \pmb { p } _ { \mathrm { t } } , \pmb { p } _ { \mathrm { r } } , \pmb { p } _ { m } )$ via Wireless Splatting and Rendering

This subsection delineates the representation of complex attenuation $\Theta _ { m } \mathopen { } \mathclose \bgroup \left( \theta , \phi , p _ { \mathrm { t } } , p _ { \mathrm { r } } , \pmb { p } _ { m } \aftergroup \egroup \right)$ via wireless splatting and wireless rendering process within the BiWGS framework.

1) Wireless Splatting: The wireless splatting stage entails two geometric transformations: a view transformation and a parallel projection. Different from conventional splatting in (5) operated on an image plane defined by a physical cameraâs pose, in this paper, we use virtual projection planes to realize wireless splatting. These virtual projection planes are established by specifying a normal vector and an anchor point in 3D space, which uniquely defines the planeâs orientation and position. Notably, the wireless splatting is different on the Tx and Rx side due to different choices of normal vectors.

<!-- image-->  
Fig. 4. Comparison of projection methods: (a) Parallel projection for CKM construction of our interest. (b) Perspective projection in 3DGS and BiGS.

In the Tx-side wireless splatting, each ellipsoid will receive the incident signal only from the Tx due to the consideration of single-hop scattering only. Therefore, we consider only one virtual projection plane. For ellipsoid m, the anchor point is the position of the ellipsoid itself, and the normal vector is chosen as the incident unit direction vector, which is expressed as

$$
{ \pmb { n } } _ { \mathrm { t } } = \frac { { \pmb { p } } _ { m } - { \pmb { p } } _ { \mathrm { t } } } { \| { \pmb { p } } _ { m } - { \pmb { p } } _ { \mathrm { t } } \| _ { 2 } } .\tag{20}
$$

In the Rx-side wireless splatting, multiple virtual projection planes are established for different sampled AOAs, each corresponding to a distinct transmission path. The anchor point for all planes is set at the position of Rx. For certain AOA $( \theta _ { o } , \phi _ { o } )$ , the normal vector of virtual projection plane is expressed as

$$
\pmb { n } _ { \mathrm { r } } = [ \mathrm { s i n } ( \theta _ { o } ) \mathrm { c o s } ( \phi _ { o } ) , \mathrm { s i n } ( \theta _ { o } ) \mathrm { s i n } ( \phi _ { o } ) , \mathrm { c o s } ( \theta _ { o } ) ] ^ { T } .\tag{21}
$$

Fig. 3 shows both the Tx-side and Rx-side wireless splatting. However, the implementation of the parallel projection differs significantly from its perspective projection counterpart in (5). In the computer graphics domain, perspective projection is used to emulate human vision or camera image acquisition by simulating depth perception. By contrast, our methodology employs parallel projection, which maintains distanceinvariant projection scaling. This distinction is caused by the fact that there is no camera in a wireless communication context, thereby eliminating the requirement for a perspective projection. A comparison of the two above projection methods is shown in Fig. 4.

After the virtual projection plane is established, the wireless splatting is expressed as

$$
\begin{array} { r l } & { \mu _ { \mathrm { t } } ^ { \prime } = W _ { \mathrm { t } } \mu + d _ { \mathrm { t } } = [ \mu _ { \mathrm { t } , x } ^ { \prime } , \mu _ { \mathrm { t } , y } ^ { \prime } , \mu _ { \mathrm { t } , z } ^ { \prime } ] ^ { T } , } \\ & { \Sigma _ { \mathrm { t } } ^ { \prime } = W _ { \mathrm { t } } \Sigma W _ { \mathrm { t } } ^ { T } , } \\ & { \mu _ { \mathrm { r } } ^ { \prime } = W _ { \mathrm { r } } \mu + d _ { \mathrm { r } } = [ \mu _ { \mathrm { r } , x } ^ { \prime } , \mu _ { \mathrm { r } , y } ^ { \prime } , \mu _ { \mathrm { r } , z } ^ { \prime } ] ^ { T } , } \\ & { \Sigma _ { \mathrm { r } } ^ { \prime } = W _ { \mathrm { r } } \Sigma W _ { \mathrm { r } } ^ { T } . } \end{array}\tag{22}
$$

Notably, the distances $d _ { \mathrm { t } , m }$ and $d _ { \mathrm { r } , m }$ in (19) are also determined during the wireless splatting process. Based on (22), the distances are expressed as

$$
d _ { \mathrm { t } , m } = \mu _ { \mathrm { t } z } ^ { \prime } , \quad d _ { \mathrm { r } , m } = \mu _ { \mathrm { r } z } ^ { \prime } .\tag{23}
$$

Subsequently, we truncate the third row of mean vectors $\mu _ { \mathrm { t } } ^ { \prime } ,$ $\mu _ { \mathrm { r } } ^ { \prime }$ and third row/column of covariance matrices Î£â²t, $\Sigma _ { \mathrm { r } } ^ { \prime } .$

These operations yield the 2D Gaussian parameters $\pmb { \mu } _ { \mathrm { 2 D } }$ and $\Sigma _ { \mathrm { 2 D } }$ for projected 2D Gaussian ellipsoids. The 2D Gaussian distributions after wireless splatting for Tx and Rx sides are denoted as $G _ { \mathrm { t } } ^ { \prime } ( \cdot )$ and $G _ { \mathrm { r } } ^ { \prime } ( \cdot )$ , respectively. The concrete form is the same as that in (6).

2) Wireless Rendering: After the wireless splatting stage, the complex attenuation caused by obstruction among ellipsoids is calculated by the wireless rendering equation. For an ellipsoid $m ,$ the complex obstruction attenuation of its relevant scattering path is decomposed into Tx-side attenuation $\Theta _ { \mathrm { t } , m } \mathopen { } \mathclose \bgroup \left( p _ { \mathrm { t } } , p _ { m } \aftergroup \egroup \right)$ , and Rx-side attenuation $\Theta _ { \mathrm { r } , m } ( \theta , \phi , p _ { \mathrm { r } } , p _ { m } )$ which is given by

$$
\Theta _ { m } ( \theta , \phi , \pmb { p } _ { \mathrm { t } } , \pmb { p } _ { \mathrm { r } } , \pmb { p } _ { m } ) = \Theta _ { \mathrm { t } , m } ( \pmb { p } _ { \mathrm { t } } , \pmb { p } _ { m } ) \Theta _ { \mathrm { r } , m } ( \theta , \phi , \pmb { p } _ { \mathrm { r } } , \pmb { p } _ { m } ) .\tag{24}
$$

For the Tx-side wireless rendering, the wireless rendering equation is given by

$$
\Theta _ { \mathrm { t } , m } ( p _ { \mathrm { t } } , p _ { m } ) = \prod _ { k \in \mathcal { F } _ { \mathrm { t } } ( m ) } ( 1 - \alpha _ { m , k } ) e ^ { - j \frac { 2 \pi } { \lambda } \gamma _ { m , k } } ,\tag{25}
$$

where $\mathcal { F } _ { \mathrm { t } } ( m )$ denotes the set of ellipsoids along the Txellipsoid path, which is determined in the wireless splatting stage, $\alpha _ { m , k }$ denotes the opacity of ellipsoid $k , \ \gamma _ { m , k } \in [ 0 , \lambda ]$ denotes the length of equivalent path that signal travels within ellipsoid k, physically related the refractive index of material, and $( 1 - \alpha _ { m , k } ^ { \dot { } } ) e ^ { - j \frac { 2 \bar { \pi } } { \lambda } \gamma _ { m , k } }$ represents the complex attenuation caused by the obstruction of ellipsoid k.

In (25), $\alpha _ { m , k }$ and $\gamma _ { m , k }$ are obtained in the same way similarly as in (8) based on the splatted 2D Gaussian ellipsoids, which is given by

$$
\begin{array} { r } { \alpha _ { m , k } = \alpha _ { k } ^ { \operatorname* { m a x } } G _ { \mathrm { t } , m , k } ^ { \prime } ( \pmb { x } _ { \mathrm { 2 D } } ^ { m } ) , } \\ { \gamma _ { m , k } = \gamma _ { k } ^ { \operatorname* { m a x } } G _ { \mathrm { t } , m , k } ^ { \prime } ( \pmb { x } _ { \mathrm { 2 D } } ^ { m } ) , } \end{array}\tag{26}
$$

where $\alpha _ { k } ^ { \mathrm { m a x } }$ and $\gamma _ { k } ^ { \operatorname* { m a x } }$ denote the maximum opacity and length of equivalent path of ellipsoid k, respectively, $G _ { \mathrm { t } , m , k } ^ { \prime } ( \cdot )$ denotes the 2D Gaussian distribution of ellipsoid k via the $\mathrm { T x } -$ side wireless splatting of ellipsoid m, and $\pmb { x } _ { \mathrm { 2 D } } ^ { m }$ denotes the 2D position of ellipsoid m in the virtual projection plane.

Furthermore, for the Rx-side rendering, the wireless rendering equation is given by

$$
\Theta _ { \mathrm { r } , m } ( \theta , \phi , \mathbf { p } _ { \mathrm { r } } , \mathbf { p } _ { m } ) = \alpha _ { m , m } \prod _ { l \in \mathcal { F } _ { \mathrm { r } } ( m ) } ( 1 - \alpha _ { m , l } ) e ^ { - j \frac { 2 \pi } { \lambda } \gamma _ { m , l } } ,\tag{27}
$$

where $\mathcal { F } _ { \mathrm { r } } ( m )$ denotes the set of other ellipsoids appearing along the ellipsoid-Rx path obstructing the transmission, which is also determined in the wireless splatting stage. $\alpha _ { m , m } , \alpha _ { m , l }$ and $\gamma _ { m , l }$ are defined similarly as in (26):

$$
\begin{array} { r } { \alpha _ { m , m } = \alpha _ { m } ^ { \mathrm { m a x } } G _ { \mathrm { r } , m , m } ^ { \prime } ( \pmb { x } _ { \mathrm { 2 D } } ^ { \mathrm { r } } ) , } \\ { \alpha _ { m , l } = \alpha _ { l } ^ { \mathrm { m a x } } G _ { \mathrm { r } , m , l } ^ { \prime } ( \pmb { x } _ { \mathrm { 2 D } } ^ { \mathrm { r } } ) , } \\ { \gamma _ { m , l } = \gamma _ { l } ^ { \mathrm { m a x } } G _ { \mathrm { r } , m , l } ^ { \prime } ( \pmb { x } _ { \mathrm { 2 D } } ^ { \mathrm { r } } ) , } \end{array}\tag{28}
$$

where $G _ { \mathrm { r } , m , l } ^ { \prime } ( \cdot )$ denotes the 2D Gaussian distribution of ellipsoid l via the Rx-side wireless splatting of ellipsoid m, and $\pmb { x } _ { \mathrm { 2 D } } ^ { r }$ denotes the 2D position of Rx in the virtual projection plane.

Note that the discrepancy between (26) and (28) stems from the different anchor point chosen in the Tx-side and Rx-side wireless splatting. Also note that the wireless rendering in (25) and (27) is different from the optical rendering counterpart in (7). Different from (7) focusing on real RGB values, the formulations in (25) and (27) incorporate two critical physical mechanisms: amplitude attenuation governed by the Gaussian ellipsoidâs opacity Î±, which quantifies signal amplitude reduction along obstructed transmission paths, and phase shifting introduced through the length of equivalent path $\gamma ,$ capturing phase distortion due to dielectric interactions in obstacles. This extends traditional rendering theory to scenarios of wireless signal transmission.

In addition, we obtain the complex obstruction attenuation $\Theta _ { L } ( p _ { \mathrm { t } } , p _ { \mathrm { r } } )$ , similarly as the Tx-side wireless rendering procedure, by employing sequential direct wireless splatting and rendering operations. However, there are notable differences. First, for wireless splatting of direct path, the anchor point is the Rx position, and the normal vector for virtual projection plane is given as

$$
{ \pmb { n } } _ { d } = \frac { { \pmb { p } } _ { \mathrm { { r } } } - { \pmb { p } } _ { \mathrm { { t } } } } { \| { \pmb { p } } _ { \mathrm { { r } } } - { \pmb { p } } _ { \mathrm { { t } } } \| _ { 2 } } .\tag{29}
$$

Moreover, the distance of the direct path $d _ { L }$ is equal to the distance between Tx and Rx. As such, the complex attenuation $\Theta _ { L }$ of direct path is expressed as

$$
\Theta _ { L } ( p _ { \mathrm { t } } , p _ { \mathrm { r } } ) = \prod _ { k \in \mathcal { E } _ { \mathrm { r } } ( \theta _ { L } , \phi _ { L } ) } ( 1 - \alpha _ { k } ) e ^ { - j \frac { 2 \pi } { \lambda } \gamma _ { k } } ,\tag{30}
$$

where $\mathcal { E } _ { \mathrm { r } } ( \theta _ { L } , \phi _ { L } )$ denotes the set of ellipsoids along the direct path.

Remark 4.3: Comparing the wireless rendering equations (25), (27), and (30) with the optical rendering equation (7) highlights a decisive difference: phase. The wireless formulation retains phase, whereas the optical counterpart discards it because visible light operates at much higher frequencies (hence much shorter wavelengths), causing phase to oscillate beyond the temporal resolution of standard detectors and be hard to trace [19]. Nevertheless, phase is important in wireless communication systems due to its critical role in constructive/destructive signal combination of multipath components, which directly governs signal integrity at Rx.

Remark 4.4: Another distinction between optical rendering and wireless transmission models lies in the absence of distance-dependent attenuation terms in the optical rendering equation. This distinction arises from fundamental differences in receiver perception mechanisms. Distance-dependent attenuation occurs in both wireless communications and free-space optical (FSO) systems [33], causing a reduction of absolute intensity. Crucially, these systems employ electronic receivers capable of measuring the absolute intensity of EM wave. Nevertheless, within the computer graphics field (including 3DGS and BiGS), rendered images are ultimately perceived by the human eyes. Human eyes employ complicated physiological filtering and processing on incident light, and perceive relative intensity of light in a logarithmic way [34]. Therefore, distance-dependent attenuation is negligible for human eyes except at sufficiently large distances, which is rarely encountered in optical 3D reconstruction contexts.

C. Representation of Bidirectional Complex Scattering Coefficient $\Gamma _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } )$ via BSH

This subsection delineates the representation of bidirectional complex scattering coefficient $\Gamma _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } )$ via BSH within the BiWGS framework. Similar to BiGS, we employ the BSH function to fit the bidirectional complex scattering coefficient. Mathematically, BSH can directly approximate both real and imaginary components of $\Gamma _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } )$ . However, experimentally, we find that directly fitting the bidirectional complex scattering coefficient via BSH may easily lead to training instability, manifesting as gradient explosion. To stabilize the convergence, we decompose $\Gamma _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } )$ into two coefficients to optimize separately during the training process, which is given by

$$
\begin{array} { r } { \Gamma _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } ) = \underbrace { Z _ { m } } _ { \mathrm { A n g l e - i n d e p e n d e n t } } \underbrace { V _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } ) } _ { \mathrm { A n g l e - d e p e n d e n t } } , } \end{array}\tag{31}
$$

where the coefficient $Z _ { m } \in \mathbb { R }$ denotes an angle-independent coefficient that will be directly optimized in the backward propagation, and $V _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } )$ denotes an angledependent coefficient satisfying $| V _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } ) | \ \leq \ 1$ that will be fit by the BSH function. With the decomposition in (31), the training of the proposed BiWGS method will become more stable.

Next, the coefficient $V _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } )$ is fit via the BSH function. To facilitate the understanding, the coefficient is rewritten as

$$
\begin{array} { r l } & { V _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } ) } \\ & { = \mathcal { R } ( V _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } ) ) + j \mathcal { Z } ( V _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } ) ) , } \end{array}\tag{32}
$$

where $\mathcal { R } ( \cdot )$ and $\boldsymbol { \mathcal { T } } ( \cdot )$ denote the real and imaginary parts of a complex number, respectively. Next, we use 2 groups of BSH coefficients to fit the real and imaginary parts of $V _ { m } ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } )$ , respectively, which are expressed as

(33)

$$
\begin{array} { r l } & { \mathcal { R } \big ( V _ { m } \big ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } \big ) \big ) } \\ & { \quad = \displaystyle \sum _ { i = 1 } ^ { ( D + 1 ) ^ { 2 } } \sum _ { k = 1 } ^ { ( D + 1 ) ^ { 2 } } a _ { i , k , m } ^ { \mathrm { R e } } y _ { k } \big ( \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } \big ) y _ { i } \big ( \theta _ { m } , \phi _ { m } \big ) , } \\ & { \mathcal { T } \big ( V _ { m } \big ( \theta _ { m } , \phi _ { m } , \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } \big ) \big ) } \\ & { \quad = \displaystyle \sum _ { i = 1 } ^ { ( D + 1 ) ^ { 2 } ( D + 1 ) ^ { 2 } } a _ { i , k , m } ^ { \mathrm { I m } } y _ { k } \big ( \theta _ { m } ^ { \prime } , \phi _ { m } ^ { \prime } \big ) y _ { i } \big ( \theta _ { m } , \phi _ { m } \big ) , } \end{array}\tag{34}
$$

where $a _ { i , k , m } ^ { \mathrm { R e } }$ and $a _ { i , k , m } ^ { \mathrm { I m } }$ are BSH coefficients of the real part and imaginary part, respectively.

Note that $V _ { m } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } )$ should also fit the reciprocity, similarly as that in (12), in which the bidirectional complex scattering coefficient remains identical when the incident and scattering directions are interchanged. In other words, we have

$$
V _ { m } ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } ) = V _ { m } ( \pi - \theta ^ { \prime } , \pi + \phi ^ { \prime } , \pi - \theta , \pi + \phi ) , \forall \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } .\tag{35}
$$

To ensure the reciprocity condition in (35), we impose the symmetric structures on the BSH coefficients at different indexes $i , k$ in (33) and (34). Towards this end, when $D = 3 .$

we define two index sets partitioning the SH basis according to their degrees for clarity4 expressed as

$$
\begin{array} { l } { { \mathcal { D } _ { e } = \{ 1 , 5 , 6 , 7 , 8 , 9 \} , } } \\ { { \mathcal { D } _ { o } = \{ 2 , 3 , 4 , 1 0 , 1 1 , 1 2 , 1 3 , 1 4 , 1 5 , 1 6 \} . } } \end{array}\tag{36}
$$

According to the concrete forms of the SH basis (11a), (11b), the following properties are obtained

$$
\left\{ \begin{array} { l l } { { \pmb y } _ { i } ( \theta _ { m } , \phi _ { m } ) = { \pmb y } _ { i } ( \pi - \theta _ { m } , \pi + \phi _ { m } ) , \quad i \in \mathcal { D } _ { e } , } \\ { { \pmb y } _ { i } ( \theta _ { m } , \phi _ { m } ) = - { \pmb y } _ { i } ( \pi - \theta _ { m } , \pi + \phi _ { m } ) , \quad i \in \mathcal { D } _ { o } . } \end{array} \right.\tag{37}
$$

Substituting (33), (34), and (37) into (35), we obtain

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { a _ { i , k , m } ^ { l } = a _ { k , i , m } ^ { l } , \quad i , k \in \mathcal { D } _ { e } \vee i , k \in \mathcal { D } _ { o } , } \\ { a _ { i , k , m } ^ { l } = - a _ { k , i , m } ^ { l } , \quad \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{38}
$$

for any $l \in \{ \mathrm { R e } , \mathrm { I m } \}$ . By preserving the symmetric structures of BSH coefficients specified in (38), reciprocity is guaranteed.

It is also interesting to compare the BSH representation in BiGS for optical rendering versus our proposed BiWGS for CKM construction. The BiGS method employs the 3 groups of BSH coefficients for each ellipsoid to model the optical bidirectional scattering function, thereby determining the ellipsoidâs RGB color attributes for specific viewing directions under certain illumination. In contrast, the proposed BiWGS method adapts 2 groups of BSH coefficients to characterize bidirectional complex scattering coefficients for each ellipsoid, thus fitting the complex scattering pattern for specific Tx-Rx position pairs.

## D. Overall Process

Substituting (25), (27), (30), and (31) into (19), we obtain the wireless channel represented by BiWGS method as

$$
\begin{array} { r l } & { \mu _ { \mathrm { \Delta } } \approx \displaystyle \sum _ { j \in \mathrm { } } \displaystyle \sum _ { i = \mathrm { : } i \neq j } \displaystyle \sum _ { \alpha \in \mathrm { : } i \neq j } \displaystyle \sum _ { \alpha \in \mathrm { : } i \neq j } [ \mathrm { b } ( i , \alpha ) \mathrm { : }  } \\ & {  \quad - \displaystyle \sum _ { \alpha \in \mathrm { : } i \neq j } ^ { \mathrm { - , \beta } \neq \alpha } c _ { \alpha \in \mathrm { : } i \neq j } \mathrm { ~ T ~ } \displaystyle \prod _ { \alpha ^ { \prime } \in \mathrm { : } i \neq j } \ [ - i \alpha _ { \mathrm { : } i \neq j } ] e ^ { - i \sum _ { \alpha ^ { \prime } \in \mathrm { : } i \neq j } ^ { \mathrm { \beta } } \alpha _ { \mathrm { ~ \mathrm { ~ \mathrm { ~ \scriptsize ~ { ~ f ~ } ~ } ~ \alpha ^ { \prime } ~ } } } }  } \\ & {  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad }  \\ &  \mathrm { \mathrm { ~ \scriptsize ~ \quad ~ \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } } \\ &  \displaystyle \mathrm { ~ \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \displaystyle \mathrm { ~ \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \displaystyle \mathrm  ~ \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}\tag{39}
$$

Our BiWGS method requires training data consisting of Tx-Rx position pairs, corresponding wireless channel, and associated spatial spectrum. Notably, we use the spatial spectrum to facilitate the training. It characterizes the angular power distribution of the wireless channel.

<!-- image-->  
Fig. 5. Illustration of BiWGS.

TABLE I  
MAJOR DIFFERENCES: BIGS VERSUS BIWGS
<table><tr><td>Characteristic</td><td>BiGS</td><td>BiWGS</td></tr><tr><td>Task</td><td>3D reconstruction under dynamic illumination</td><td>6D CKM construction under varying Tx-Rx position pair</td></tr><tr><td>Target of rendering</td><td>Real RGB color  $\boldsymbol { c } _ { o } ^ { \mathrm { p i x e l } }$  of each pixel of image</td><td>Complex channel vector h</td></tr><tr><td>Projection method</td><td>Perspective projection</td><td>Parallel projection</td></tr><tr><td>Distance-dependent attenuation</td><td>Not explicitly modeled</td><td>Explicitly modeled</td></tr><tr><td>Number of projection planes</td><td>Single image plane at camera</td><td>Multiple virtual projection planes at Rx</td></tr><tr><td>Phase information</td><td>Not modeled</td><td>Essential component</td></tr><tr><td>BSH fitting objective</td><td>Optical scattering function  $f ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } )$ </td><td>Bidirectional complex scattering coefficient  $\Gamma ( \theta , \phi , \theta ^ { \prime } , \phi ^ { \prime } )$ </td></tr><tr><td>Groups of BSH coefficient</td><td>Three for R, G, B attributes</td><td>Two for real and imaginary parts</td></tr></table>

Given the angular resolution z and v for azimuth and elevation angles, the spatial spectrum is defined as

$$
\mathbf { I } = \left[ \begin{array} { c c c } { P ( \theta _ { 1 } , \phi _ { 1 } ) } & { \cdot \cdot \cdot } & { P ( \theta _ { 1 } , \phi _ { z } ) } \\ { \vdots } & { \cdot } & { \vdots } \\ { P ( \theta _ { v } , \phi _ { 1 } ) } & { \cdot \cdot \cdot } & { P ( \theta _ { v } , \phi _ { z } ) } \end{array} \right] ,\tag{40}
$$

where $P ( \theta , \phi )$ denotes the power of the received signal at a certain AOA, which is generated via the Conventional Beamforming (CBF) (also called Bartlett beamforming) method [36] for its simplicity, which is expressed as

$$
P ( \theta , \phi ) = \| \pmb { b } ^ { H } ( \theta , \phi ) \pmb { h } \| _ { 2 } ^ { 2 } .\tag{41}
$$

In the spatial spectrum, high-intensity regions (red in Fig. 6a) indicate stronger power at the corresponding AoA, suggesting a possible LOS path. However, the amplitude of LOS path surpasses that of scattering path by several orders of magnitude. Therefore, training on linear-scale spectrum introduces a bias toward the dominant LOS path, thereby suppressing contributions of weaker paths. This bias significantly degrades the inference performance of the proposed BiWGS model in NLOS scenarios. To mitigate this limitation, we apply a logarithmic transformation, converting the spectrum into the dB scale, which enhances the modelâs ability to capture bidirectional scattering patterns. Fig. 6 illustrates the difference between linear-scale and dB-scale spatial spectrum. It is notable that the spatial spectra in Fig. 6 are the polar coordinates representation of spatial spectrum defined in (40) to facilitate visualization. In this representation, the radial coordinate is discretized into v concentric rings, while the angular coordinate is sampled at z points along each ring, corresponding to elevation and azimuth resolutions, respectively.

<!-- image-->  
(a) linear

<!-- image-->  
(b) dB-scale  
Fig. 6. Comparison of linear and dB-scale spatial spectrum (a): linear spectrum. (b): dB-scale spectrum.

Furthermore, the loss function is designed as the mixture of spectrum loss $\mathcal { L } _ { s }$ and channel power gain loss $\mathcal { L } _ { g } .$

TABLE II COMPARISON OF 3D CKM CONSTRUCTION PERFORMANCE
<table><tr><td rowspan="2">Method</td><td colspan="2">Conference room</td><td colspan="2">Bedroom</td><td colspan="2">Office</td><td colspan="2">Average</td></tr><tr><td>SSIMâ</td><td>LPIPS</td><td>SSIMâ</td><td>LPIPS</td><td>SSIMâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>LPIPS</td></tr><tr><td>NeRF2 [19]</td><td>0.5977</td><td>0.5718</td><td>0.7138</td><td>0.5477</td><td>0.6911</td><td>0.5186</td><td>0.6675</td><td>0.5460</td></tr><tr><td>WRF-GS [21]</td><td>0.6919</td><td>0.5616</td><td>0.7210</td><td>0.5217</td><td>0.6444</td><td>0.6313</td><td>0.6858</td><td>0.5715</td></tr><tr><td>Proposed BiWGS</td><td>0.6610</td><td>0.4396</td><td>0.7002</td><td>0.4168</td><td>0.6748</td><td>0.5131</td><td>0.6787</td><td>0.4565</td></tr></table>

<!-- image-->  
Fig. 7. Three 3D physical environments of datasets; conference room (left), bedroom (center), and office (right).

â¢ Spectrum loss: Spectrum loss is defined by the $\mathcal { L } _ { 2 }$ loss in term of mean square error (MSE) between the dB-scale ground-truth spectrum $I _ { \mathrm { g t } } ^ { \mathrm { d B } }$ and predicted spectrum $I _ { \mathrm { p r e d } } ^ { \mathrm { d B } } ,$ which is expressed as

$$
\mathcal { L } _ { s } = \mathcal { L } _ { 2 } ~ ( I _ { \mathrm { g t } } ^ { \mathrm { d B } } , I _ { \mathrm { p r e d } } ^ { \mathrm { d B } } ) ,\tag{42}
$$

where I is determined by (40).

â¢ Channel power gain loss: The dB-scale channel power gain is expressed as

$$
g _ { i } ^ { \mathrm { d B } } = 1 0 \log _ { 1 0 } ( \| h _ { i } \| _ { 2 } ^ { 2 } ) , i \in \{ \mathrm { g t , p r e d } \} .\tag{43}
$$

Accordingly, channel power gain loss is defined by the $\mathcal { L } _ { 1 }$ loss, which is the mean absolute error (MAE) between the ground-truth channel power gain $g _ { \mathrm { g t } } ^ { \mathrm { d B } }$ and the predicted channel power gain $g _ { \mathrm { p r e d } } ^ { \mathrm { d B } }$ as follows

$$
\begin{array} { r } { \mathcal { L } _ { g } = \mathcal { L } _ { 1 } ~ ( g _ { \mathrm { g t } } ^ { \mathrm { d B } } , g _ { \mathrm { p r e d } } ^ { \mathrm { d B } } ) . } \end{array}\tag{44}
$$

Finally, the loss function is defined as the weighted sum of the above two losses, which is expressed as

$$
\mathcal { L } = \eta _ { 1 } \mathcal { L } _ { s } + \eta _ { 2 } \mathcal { L } _ { g } ,\tag{45}
$$

where $\eta _ { 1 }$ and $\eta _ { 2 }$ are the weighting coefficients to control the importance of different components of the loss function.

During the training process, the parameters of every Gaussian ellipsoid, including mean vector Âµ, rotation matirx R, scaling matrix S, maximum opacity $\alpha ^ { \mathrm { m a x } }$ , BSH coefficents $a _ { i , k } ^ { \mathrm { R e } }$ and $a _ { i , k } ^ { \mathrm { I m } }$ for $i , k \in \{ 1 , \ldots , D + 1 \}$ , maximum length of equivalent path $\gamma ^ { \mathrm { m a x } }$ , and angle-independent coefficient $Z ,$ are optimized via stochastic gradient descent using the adaptive moment estimation (Adam) optimizer. In addition, we apply the adaptive density control strategy in the training stage similarly as 3DGS illustrated in Section II-C to control the numbers and size of Gaussian ellipsoids within the environment. Fig. 5 illustrates the overall pipeline of the proposed BiWGS method. Furthermore, several major differences between the BiGS and BiWGS are summarized in Table I.

## IV. EXPERIMENT RESULTS

In this section, we evaluate the performance of our proposed BiWGS algorithm for 6D CKM construction. Specifically, we use three synthesis 3D scenes in [20] as our 3D physical environment, shown in Fig. 7. Moreover, we utilize the NVIDIA Sionna ray-tracing simulator [12] to generate our datasets within the 3D physical environment. In our simulations, the frequency of signals is set as 6 GHz, with all transmission paths limited to a maximum scattering/reflection times of 3. The Rx is equipped with a half-wavelength UPA antenna configuration with $N _ { v } = N _ { h } = 4$ . The azimuth and elevation resolutions of the spatial spectrum are set to z = 180 and v = 180, respectively, with resolution of 1â¦. Lastly, taichi [37] is used to implement parallelized computations in the CUDA kernels.

We consider both 3D and 6D CKM construction for performance comparison5. In the 3D CKM construction experiments, all datasets comprise channel measurements acquired at a fixed Rx position with uniformly sampled Tx positions. Each dataset undergoes 90%â10% training-test partitioning. Moreover, the performance is evaluated by the quality of predicted spatial spectrums. Specifically, we use two evaluation metrics including structural similarity index measure (SSIM) and learned perceptual image patch similarity (LPIPS). Compared with SSIM, LPIPS incorporates spatial ambiguities to describe high-dimensional feature similarities between spectra [38]. In contrast, for the 6D CKM construction experiment, all datasets consist of two distinct sets of channel measurements: a training set containing measurements from 9 distinct Tx positions, and a test set containing measurements from a different Tx position (unseen in the training set). Rx positions at both the training set and the test set are uniformly sampled. Moreover, an 80%â20% training-to-test data ratio is maintained across all datasets. Performance for 6D CKM construction is evaluated based on the accuracy of the channel power gain prediction, quantified by the MAE and normalized mean absolute error (NMAE). Furthermore, all Tx and Rx maintain a minimum distance of one wavelength from 3D environmental objects to avoid reactive near-field region interactions [39]. Any Tx or Rx violating this criterion will be discarded from the dataset.

<!-- image-->  
Fig. 8. Comparative visualization of spatial spectrum predictions for conference room.

<!-- image-->  
Fig. 9. Comparative visualization of spatial spectrum predictions for bedroom.

Table. II compares the median SSIM and LPIPS metrics of our method for 3D CKM construction, versus the benchmark schemes NeRF2 [19] and WRF-GS [21] at three different scenarios. On average, the median SSIM of BiWGS, WRF-GS, and $\mathrm { N e R F ^ { 2 } }$ are 0.6787, 0.6858, and 0.6675, respectively. While the median LPIPS of BiWGS, WRG-GS, and NeRF2 are 0.4565, 0.5715, and 0.5460, respectively. Consequently, the proposed BiWGS achieves a comparable performances as the SOTA method WRF-GS at the SSIM metric (with only a 0.007 gap). Moreover, the proposed BiWGS reaches the SOTA performance at the LPIPS metric. This is due to the fact that our explicit, bidirectional modelling can capture the features of the wireless transmission environment better than the implicit, unidirectional model WRF-GS or NeRF2. Furthermore, Figs. 8, 9, and 10 provide a visual comparison of the predicted spatial spectra among three methods and the ground truth across three scenarios. In each scenario, three example spots, corresponding to distinct Tx positions, are presented for illustration. The results show that both WRF-GS and NeRF2 exhibit blurred spectral patterns with evident spatial ambiguities, whereas the proposed BiWGS method produces spectra with substantially fewer ambiguities. This improvement highlights the superior capability of BiWGS in preserving spectral fidelity, which is consistent with its advantage under the LPIPS metric.

<!-- image-->  
Fig. 10. Comparative visualization of spatial spectrum predictions for office.

TABLE III  
CHANNEL POWER GAIN PREDICTION PERFORMANCE FOR 6D CKM CONSTRUCTION ACROSS ENVIRONMENTS
<table><tr><td rowspan=1 colspan=1>Scenarios</td><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>MLPs [16]</td><td rowspan=1 colspan=1>BiWGS</td></tr><tr><td rowspan=1 colspan=1>Conf. room</td><td rowspan=1 colspan=1>MAENMAE</td><td rowspan=1 colspan=1>4.40 dB0.096</td><td rowspan=1 colspan=1>3.68 dB0.080</td></tr><tr><td rowspan=1 colspan=1>Bedroom</td><td rowspan=1 colspan=1>MAENMAE</td><td rowspan=1 colspan=1>7.81 dB0.158</td><td rowspan=1 colspan=1>4.93 dB0.101</td></tr><tr><td rowspan=1 colspan=1>Office</td><td rowspan=1 colspan=1>MAENMAE</td><td rowspan=1 colspan=1>14.60 dB0.237</td><td rowspan=1 colspan=1>6.70 dB0.109</td></tr></table>

Table. III compares the 6D CKM construction performance between the proposed BiWGS method and the classical MLPbased approach [16] for channel power gain prediction. Note that NeRF2 and WRF-GS are not applicable for 6D CKM construction here. The results demonstrate that the proposed BiWGS approach exhibits a significant performance advantage in 6D CKM construction. Furthermore, the results indicate that BiWGS exhibits strong transferability, learning electromagnetic transmission characteristics from known Tx configurations and achieving high-accuracy channel power gain predictions at novel, unobserved Tx positions.

## V. CONCLUSION

This paper proposes BiWGS, a novel 6D CKM construction method inspired by the optical BiGS architecture. Our proposed method learns the bidirectional scattering patterns of Gaussian ellipsoids to accurately fit the electromagnetic transmission characteristics of the wireless environment, thereby enabling the construction of 6D CKM. Comprehensive experiment evaluations demonstrate that BiWGS achieves spatial spectrum prediction accuracy comparable to SOTA 3D CKM construction techniques while also supporting 6D CKM construction. This represents a dimensionality expansion without compromising prediction fidelity. Some interesting directions for future extensions include computational complexity reduction, cross-frequency wideband 6D CKM construction, and BiWGSâs applications.

## REFERENCES

[1] Z. Ren, L. Qiu, J. Xu, and D. W. K. Ng, âSensing-assisted sparse channel recovery for massive antenna systems,â IEEE Trans. Veh. Technol., vol. 73, pp. 17824â17829, Nov. 2024.

[2] F. Fang, H. Zhang, J. Cheng, S. Roy, and V. C. Leung, âJoint user scheduling and power allocation optimization for energy-efficient noma systems with imperfect CSI,â IEEE Journal on Selected Areas in Communications, vol. 35, pp. 2874â2885, Dec. 2017.

[3] M. Giordani, M. Polese, A. Roy, D. Castor, and M. Zorzi, âA tutorial on beam management for 3GPP NR at mmWave frequencies,â IEEE Commun. Surv. Tutorials, vol. 21, pp. 173â196, Sep. 2018.

[4] Y. Zeng and X. Xu, âToward environment-aware 6G communications via channel knowledge map,â IEEE Wireless Commun., vol. 28, pp. 84â91, Mar. 2021.

[5] H. Sun, L. Zhu, and R. Zhang, âChannel gain map estimation for wireless networks based on scatterer model,â IEEE Trans. Wireless Commun., Apr. 2025.

[6] D. Wu, Y. Zeng, S. Jin, and R. Zhang, âEnvironment-aware and training-free beam alignment for mmwave massive MIMO via channel knowledge map,â in 2021 IEEE Int. Conf. Commun. Workshops (ICC Workshops), pp. 1â7, IEEE, Jun. 2021.

[7] D. Wu, Y. Zeng, S. Jin, and R. Zhang, âEnvironment-aware hybrid beamforming by leveraging channel knowledge map,â IEEE Trans. Wireless Commun., vol. 23, pp. 4990â5005, Oct. 2023.

[8] Y. Zeng, J. Chen, J. Xu, D. Wu, X. Xu, S. Jin, X. Gao, D. Gesbert, S. Cui, and R. Zhang, âA tutorial on environment-aware communications via channel knowledge map for 6G,â IEEE Commun. Surveys Tuts., vol. 26, pp. 1478â1519, Feb. 2024.

[9] X. Xia, K. Xu, W. Xie, Y. Xu, N. Sha, and Y. Wang, âMultiple aerial base station deployment and user association based on binary radio map,â IEEE Internet Things J., vol. 10, no. 19, pp. 17206â17219, 2023.

[10] S. Zhang and R. Zhang, âRadio map-based 3D path planning for cellularconnected uav,â IEEE Trans. Wireless Commun., vol. 20, pp. 1975â1989, Nov. 2020.

[11] N. Suga, R. Sasaki, M. Osawa, and T. Furukawa, âRay tracing acceleration using total variation norm minimization for radio map simulation,â IEEE Wireless Commun. Lett., vol. 10, pp. 522â526, Nov. 2020.

[12] J. Hoydis, F. A. Aoudia, S. Cammerer, M. Nimier-David, N. Binder, G. Marcus, and A. Keller, âSionna RT: Differentiable ray tracing for radio propagation modeling,â in 2023 IEEE Global Commun. Conf. Workshops (GLOBECOM Workshops), pp. 317â321, IEEE, Dec. 2023.

[13] Y. Zhang and S. Wang, âK-nearest neighbors gaussian process regression for urban radio map reconstruction,â IEEE Commun. Lett., vol. 26, pp. 3049â3053, Sep. 2022.

[14] H. Sun and J. Chen, âPropagation map reconstruction via interpolation assisted matrix completion,â IEEE Trans. Signal Process., vol. 70, pp. 6154â6169, Dec. 2022.

[15] Y. Hu and R. Zhang, âA spatiotemporal approach for secure crowdsourced radio environment map construction,â IEEE/ACM Trans. Netw., vol. 28, pp. 1790â1803, May. 2020.

[16] K. Saito, Y. Jin, C. Kang, J. i. Takada, and J. S. Leu, âTwo-step path loss prediction by artificial neural network for wireless service area planning,â IEICE Commun. Exp., vol. 8, pp. 611â616, Sep. 2019.

[17] R. Levie, CÂ¸ . Yapar, G. Kutyniok, and G. Caire, âRadioUNet: Fast radio map estimation with convolutional neural networks,â IEEE Trans. Wireless Commun., vol. 20, pp. 4001â4015, Feb. 2021.

[18] S. Wang, X. Xu, and Y. Zeng, âDeep learning-based CKM construction with image super-resolution,â arXiv preprint arXiv:2411.08887, 2024.

[19] X. Zhao, Z. An, Q. Pan, and L. Yang, âNeRF2: Neural radio-frequency radiance fields,â in Proc. 29th Annu. Int. Conf. Mobile Comput. Netw. (Mobicom), pp. 1â15, Oct. 2023.

[20] H. Lu, C. Vattheuer, B. Mirzasoleiman, and O. Abari, âNeWRF: a deep learning framework for wireless radiation field reconstruction and channel prediction,â in Proc. 41st Int. Conf. Mach. Learn. (ICML), pp. 33147â33159, Jul. 2024.

[21] C. Wen, J. Tong, Y. Hu, Z. Lin, and J. Zhang, âWRF-GS: Wireless radiation field reconstruction with 3D Gaussian splatting,â in 2025 IEEE Conf. Comput. Commun. (INFOCOM), pp. 1â10, IEEE, May. 2025.

[22] B. Mildenhall, P. Srinivasan, M. Tancik, J. Barron, R. Ramamoorthi, and R. Ng, âNeRF: Representing scenes as neural radiance fields for view synthesis,â in Proc. Eur. Conf. Comput. Vis. (ECCV), Nov. 2020.

[23] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Gaussian Â¨ splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, pp. 139â1, Jul. 2023.

[24] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM Trans. Graph., vol. 41, pp. 1â15, Jul. 2022.

[25] L. Zhenyuan, X. Li, Y. Guo, B. Bickel, and R. Zhang, âBiGS: Bidirectional primitives for relightable 3D Gaussian splatting,â in Int. Conf. 3D Vis. (3DV), Mar. 2025.

[26] C. Efthimiou and C. Frye, Spherical Harmonics in p Dimensions. World Scientific, 2014.

[27] 3rd Generation Partnership Project (3GPP), âStudy on channel model for frequencies from 0.5 to 100 GHz,â Technical Report TR 38.901, 3GPP, Jul. 2025. [Online]. Available: https://www.3gpp.org/.

[28] E. Huang, C. DeLude, J. Romberg, S. Mukhopadhyay, and M. Swaminathan, âAnisotropic scatterer models for representing RCS of complex objects,â in 2021 IEEE Radar Conf. (RadarConf), pp. 1â6, IEEE, May 2021.

[29] Y. Huang, J. Yang, W. Tang, C.-K. Wen, S. Xia, and S. Jin, âJoint localization and environment sensing by harnessing NLOS components in RIS-aided mmwave communication systems,â IEEE Trans. Wireless Commun., vol. 22, pp. 8797â8813, Apr. 2023.

[30] Y. Zhang, Y. Wan, and A. Liu, âAn efficient massive MIMO channel reconstruction method based on virtual scatterer map,â IEEE Wireless Commun. Lett., Jun. 2025.

[31] A. Chowdary, A. Bazzi, and M. Chafii, âOn hybrid radar fusion for integrated sensing and communication,â IEEE Trans. Wireless Commun., vol. 23, pp. 8984â9000, Aug. 2024.

[32] Y. Zhang, J. Zhang, X. Hu, et al., âA unified RCS modeling of typical targets for 3GPP ISAC channel standardization and experimental analysis,â arXiv preprint arXiv:2505.20673, 2025.

[33] H. Kaushal, V. Jain, and S. Kar, Free Space Optical Communication, vol. 60. Springer, 2017.

[34] H. Gross, F. Blechinger, and B. Achtner, âHuman eye,â Handbook of Optical Systems: Volume 4: Survey of Optical Instruments, vol. 4, pp. 1â 87, 2008.

[35] R. Ramamoorthi and P. Hanrahan, âA signal-processing framework for reflection,â ACM Trans. Graph., vol. 23, pp. 1004â1042, Oct. 2004.

[36] M. S. Bartlett, âPeriodogram analysis and continuous spectra,â Biometrika, vol. 37, pp. 1â16, Jun. 1950.

[37] Y. Hu, T.-M. Li, L. Anderson, J. Ragan-Kelley, and F. Durand, âTaichi: a language for high-performance computation on spatially sparse data structures,â ACM Trans. Graph., vol. 38, pp. 1â16, Nov. 2019.

[38] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 586â595, Jun. 2018.

[39] Y. Liu, Z. Wang, J. Xu, C. Ouyang, X. Mu, and R. Schober, âNearfield communications: A tutorial review,â IEEE Open J. Commun. Soc., vol. 4, pp. 1999â2049, Aug. 2023.