# PS-GS: Gaussian Splatting for Multi-View Photometric Stereo

Yixiao Chen1, Bin Liang2, Hanzhi Guo1, Yongqing Cheng1, Jiayi Zhao1, Dongdong Weng1,3\*

1Beijing Engineering Research Center of Mixed Reality and Advanced Display, School of Optics and Photonics, Beijing

Institute of Technology, China

2China Software Testing Center, Beijing, China

3Zhengzhou Research Institute, Beijing Institute of Technology, China

crgj@bit.edu.cn

## Abstract

Integrating inverse rendering with multi-view photometric stereo (MVPS) yields more accurate 3D reconstructions than the inverse rendering approaches that rely on fixed environment illumination. However, efficient inverse rendering with MVPS remains challenging. To fill this gap, we introduce the Gaussian Splatting for Multi-view Photometric Stereo (PS-GS), which efficiently and jointly estimates the geometry, materials, and lighting of the object that is illuminated by diverse directional lights (multi-light). Our method first reconstructs a standard 2D Gaussian splatting model as the initial geometry. Based on the initialization model, it then proceeds with the deferred inverse rendering by the full rendering equation containing a lighting-computing multi-layer perceptron. During the whole optimization, we regularize the rendered normal maps by the uncalibrated photometric stereo estimated normals. We also propose the 2D Gaussian ray-tracing for single directional light to refine the incident lighting. The regularizations and the use of multi-view and multi-light images mitigate the ill-posed problem of inverse rendering. After optimization, the reconstructed object can be used for novelview synthesis, relighting, and material and shape editing. Experiments on both synthetic and real datasets demonstrate that our method outperforms prior works in terms of reconstruction accuracy and computational efficiency.

## Introduction

Inverse rendering (IR) focuses on reconstructing geometry, materials, and lighting from captured posed images. Based on IR, many downstream tasks such as relighting and material editing can be implemented, which play a pivotal role in various fields, such as games, augmented reality, cultural heritage, and film production. However, IR is still a longterm problem in computer graphics and vision as it is an illposed problem, particularly when input images are captured in an unknown environment illumination or with sparse views. The success of neural radiance field (NeRF) (Mildenhall et al. 2020), which leverages a directional-aware multilayer perceptron (MLP) to represent 3D scenes through volume rendering, has inspired many NeRF-based approaches (Yao et al. 2022; Zhang et al. 2023; Wu et al. 2025) to proceed with the IR and address the ill-posed issue. Although these methods demonstrate compelling reconstruction of geometry and reflectance, they still face challenges in terms of explicit editing and excessive computational costs, which limit their further application.

More recently, 3D Gaussian Splatting(3DGS) (Kerbl et al. 2023) has attracted much attention due to its ability of realtime and high-fidelity rendering in novel view synthesis (NVS). 3DGS models a 3D scene by a set of explicit 3D Gaussian primitives and renders pixels by alpha-blending through a specially designed rasterizer, which can be exploited to achieve efficient inverse rendering. However, as its difficulty in accurately simulating ray-based effects, some 3DGS-based IR methods have adopted simplified versions of the rendering equation (Jiang et al. 2024; Liang et al. 2024) or proposed a 3D Gaussian ray-tracing technique (Gao et al. 2024). Another challenge of these methods is the struggle with accurately computing normal only by a depth-related regularization without exploiting the relationship between primitive normal and geometry attributes. To this end, 2D Gaussian Splatting (2DGS) (Huang et al. 2024) and 2DGS-based works (Gu et al. 2025; Yao et al. 2025) leverage the normal of the elliptical disk as the normal for each primitive, which acquires more compelling normal results than 3DGS-based methods. However, as the lack of additional information in optimization, the rendered normal maps are still over-smooth and lack detail. Moreover, when proceeding with sparse-view reconstruction, the ill-posed problem gets worse, which leads to a decline in the overall quality of reconstruction.

On the other hand, photometric stereo (PS) is a technique that utilizes captured single-view images illuminated by diverse directional lights (multi-light) to obtain per-pixel surface normal of the scene. Although their ability to recover fine surface details, PS approaches (Chen et al. 2019; Ikehata 2023) with single-view input are not capable of reconstructing a full 3D shape. To simultaneously obtain surface details and full 3D shape, the multi-view photometric stereo (MVPS) (Park et al. 2017; Logothetis, Mecca, and Cipolla 2019) method that combines multi-view stereo (MVS) method with PS is proposed . By incorporating the neural implicit field and the point-based splatting, PS-NeRF (Yang et al. 2022) and DPIR (Chung, Choi, and Baek 2024) provide the solution to MVPS-based inverse rendering, respectively. Despite their advancements, they either suffer from excessive computational costs like congeners or use a simplified rendering equation that is unable to acquire more diverse materials.

To address the aforementioned issues altogether, we propose Gaussian Splatting for Multi-view Photometric Stereo (PS-GS), an inverse rendering approach that extends 2D Gaussian Splatting to physically-based deferred rendering with MVPS. We adopt the strategy of PS-NeRF (Yang et al. 2022) for optimization, which leverages guidance normals estimated via uncalibrated photometric stereo to regularize the rendered normal maps and uses multi-view and multilight images for training. Benefiting from the efficient 2D Gaussian ray-tracing technique proposed by IRGS (Gu et al. 2025), we modify this technique to estimate the visibility for the object illuminated under single directional light (SDL), and leverage its visibility result to regularize the lighting.

Our method starts with pretraining a standard 2D Gaussian splatting model. Based on this model, we then use the full physically-based rendering equation without simplification to jointly optimize materials, geometry, and lighting. In the rendering equation, a multi-layer perceptron (MLP) is leveraged to model lighting that is not modeled by the parameters of Gaussian primitives. The rendered normal maps and incident lighting are regularized by the UPS-estimated normals and SDL 2D Gaussian ray-tracing computed visibility, respectively. We exploit pre-acquisition of diverse parameter maps as regularization to mitigate the ill-posed problem of inverse rendering. To efficiently render the multiview and multi-light images, we proceed with the shading in image space, which can both achieve rapid rendering and improve reconstruction quality.

To the best of our knowledge, this is the first method to integrate Gaussian splatting and MVPS, which can accurately and efficiently proceed with geometry, materials, and lighting estimation and enable novel view synthesis, relighting, and shape and material editing. Experiments on both real and synthetic dataset confirm that our method has improved training time and reduced the usage of GPU memory compared to previous MVPS-based inverse rendering approaches while achieving comparable or better results. In summary, contributions of this paper are as follows:

â¢ We innovate an efficient physically-based deferred inverse rendering approach for multi-view photometric stereo, which jointly optimizes geometry, materials, and lighting based on 2DGS.

â¢ We propose to refine the rendered normal maps of Gaussian model by the UPS estimated normals, which significantly enhances the quality of surface estimation.

â¢ We modify 2D Gaussian ray-tracing to suit SDL and leverage its visibility results to regularize the MLPcomputed lighting, which improves the lighting reconstruction.

â¢ Our method achieves compelling results on many tasks, even for sparse-view inputs, including NVS, relighting, and material and shape editing.

## Related work

Novel view synthesis (NVS) is a long-standing problem in computer vision and graphics, which focuses on generating images under unseen viewpoints by a collection of captured images of the scene. Neural radiance fields (NeRF) (Mildenhall et al. 2020) leverage neural networks to represent the scene as a continuous 5D function, which has catalyzed a trend in high-fidelity NVS of complicated 3D scenes. Subsequent developments have improved the NeRF in accelerating training speed (Muller et al. 2022; Hu et al. Â¨ 2022), enhancing the rendering quality (Barron et al. 2021; Hu et al. 2023), enabling sparse views reconstruction (Wang et al. 2023; Shi et al. 2024), etc. More recently, 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) renders explicit 3D Gaussian opacity fields by a well-designed tile-based rasterization pipeline, achieving the state-of-the-art performance on high-fidelity real-time NVS. Many following works have been inspired by 3DGS to achieve various tasks, such as dynamic scene reconstruction (Yang et al. 2024; Li et al. 2024a), animatable avatar (Li et al. 2024b; Hu et al. 2025), and SLAM (Yan et al. 2024; Matsuki et al. 2024). Based on 3DGS, 2D Gaussian Splatting (2DGS) (Huang et al. 2024) has further expanded geometry reconstruction, which also improves the ability to simulate ray-based effects. In this paper, we adopt 2D Gaussian primitives to represent scenes and proceed with SDL ray-tracing on them, which can lead to more accurate surface and lighting reconstruction.

Inverse rendering (IR) aims to reconstruct materials and lighting from observed images, which is modeled as the interaction of the illumination and surface geometry. However, it suffers from the ill-posed problem due to the inherent ambiguity between captured images and underlying properties. Inspired by the success of NeRF in 3D representation, NeRF-based IR methods model the interaction of light with the neural sense representation that contains various material properties (Zhang et al. 2021; Wang et al. 2024). However, all these approaches almost suffer from long training times. Recent methods have combined the 3DGS with IR by attaching lighting and material parameters to each Gaussian primitive. Owing to its specially designed rendering pipeline, 3DGS-based approaches significantly outperform the NeRF-based works on reconstruction quality and training speed. GS-IR (Liang et al. 2024) realizes 3DGS-based IR by simplifying the rendering equation via split-sum approximation and using baked volumes to tackle the occlusion in modeling indirect illumination. R3DG (Gao et al. 2024) renders each Gaussian individually with full rendering equation, using 3D Gaussian ray-tracing to compute visibility. GShader (Jiang et al. 2024) applies an explicit approximation to the rendering equation by simplified shading functions and predicts normal residual for precise normal estimation. However, methods based on 3DGS compute normals solely from the depth information of Gaussian primitives, without leveraging other geometric cues, thus facing challenges in achieving high-precision surface estimation. The 2DGS-based approaches, whose normal of each Gaussian corresponds directly to the surfel normal, demonstrate strong performance in both accurate normal estimation and shape reconstruction by integrating depth information. Ref-Gaussian (Yao et al. 2025), built upon 2DGS, models reflective objects by leveraging pre-integration to simplify the rendering equation and computing the visibility of reflection terms via ray-tracing based on the extracted mesh. IRGS (Gu et al. 2025) proposes a 2D Gaussian ray-tracing technique to model inter-reflections without simplifying the rendering equation for inverse rendering. However, as constant environment illumination of these works, the inherent ambiguity remains, leading to over-smooth surface reconstruction.

Multi-view photometric stereo (MVPS) leverages multiview captured images under varying illumination to realize more accurate 3D surface reconstruction. Traditional approaches (Park et al. 2017; Logothetis, Mecca, and Cipolla 2019) leverage simplified surface reflectance and separately reconstruct a coarse shape by MVS and per-view surface normal by PS, and the obtained normal is then used to refine the coarse shape. More recently, MVPS has been combined with implicit neural representation or point-based model to acquire high-fidelity reconstruction results. Kaya et al. first proposed the NeRF-based MVPS approach that integrates depth maps with normal maps from a pretrained PS network to reconstruct complete shape. PS-NeRF (Yang et al. 2022) jointly estimates the geometry, material, and illumination by NeRF-based shadow-aware inverse rendering, and proceeds with regularization on surface normal by the normals estimated from UPS method (Chen et al. 2019). RNb-Neus (Brument et al. 2024) leverages both explicit 2.5D shape representation and implicit neural shape representation to accurately reconstruct the shape. DPIR (Chung, Choi, and Baek 2024) exploits hybrid point-volumetric geometry representation and specially designed point-based visibility detection method to jointly optimize the point locations, radii, surface normals, and reflectance with differentiable point-based rendering. However, these methods either suffer from excessive training time, are merely applicable to shape reconstruction, or are incapable of recovering materials of the surface.

## Preliminary

## 2D Gaussian Splatting

In standard 2DGS (Huang et al. 2024), each Gaussian is represented as 2D oriented planar disk, which is defined by a position $^ { p , }$ two scaling factors $s _ { u }$ and $s _ { v } .$ , two principal tangential vector $\mathbf { \Delta } _ { t _ { u } }$ and $t _ { v }$ , opacity o, and color coefficients c. The influence of each 2D Gaussian is defined as:

$$
\mathcal { G } ( \mathbf { u } ( \boldsymbol { x } ) ) = \exp \left( - \frac { u ^ { 2 } + v ^ { 2 } } { 2 } \right) ,\tag{1}
$$

where $( u , v )$ are ray-splat intersection coordinates in the local tangent space, x represents the homogeneous ray passing through pixel $( x , y )$ and intersecting the 2D Gaussian primitive at ${ \boldsymbol { z } } . \mathbf { u } ( { \pmb x } ) = ( u , v )$ is the intersection point function, which can be described by:

$$
\pmb { x } = ( x z , y z , z , 1 ) ^ { T } = \pmb { W } \pmb { H } ( u , v , 1 , 1 ) ^ { T } ,\tag{2}
$$

where W is the transformation matrix from world space to camera space and

$$
\pmb { H } = \left[ \begin{array} { c c c c } { s _ { u } \pmb { t _ { u } } } & { s _ { v } \pmb { t _ { v } } } & { \mathbf { 0 } } & { p } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} \right] \in \mathbb { R } ^ { 4 \times 4 } .\tag{3}
$$

Unlike 3DGS, which struggles to directly derive normals from geometry parameters, the normal for each 2D Gaussian is defined as the cross product of two tangential vectors of the Gaussian disk:

$$
n = t _ { u } \times t _ { v } .\tag{4}
$$

During rendering, 2D Gaussians are projected in camera space. The final rendered pixel color C is calculated by:

$$
\pmb { C } _ { i } = \sum _ { n \in N } \pmb { c } _ { n } \alpha _ { n } T _ { n } , \quad T _ { n } = \prod _ { m = 1 } ^ { n - 1 } ( 1 - \alpha _ { m } ) ,\tag{5}
$$

where $T _ { n }$ is the accumulated transparency and $\alpha _ { n } = o _ { n }$ $\mathcal { G } _ { n } ( \mathbf { u } ( \pmb { x } ) )$ ). See more details of 2DGS in (Huang et al. 2024).

## Rendering equation

In PS-GS, we replace spherical harmonics (SH) lighting with physically-based rendering (PBR) and adopt deferred shading (Hargreaves and Harris 2004). The illumination at a point r is given by the rendering equation:

$$
L ( \omega _ { o } , \pmb { r } ) = \int _ { \Omega } f ( \omega _ { o } , \omega _ { i } , \pmb { r } ) L _ { i } ( \omega _ { i } , \pmb { r } ) ( \omega _ { i } \cdot \pmb { n } ) d \omega _ { i } ,\tag{6}
$$

where $\omega _ { o }$ is the outgoing radiance direction, $\omega _ { i }$ is the incident radiance direction, n is the surface normal of point $r , f$ is the bidirectional reflectance distribution function (BRDF), and $L _ { i } ( \omega _ { i } , \pmb { r } )$ is the incident radiance. The simplified Disney BRDF model (Burley and Studios 2012) is used in this work, which contains only diffuse albedo Î±, metallic m, and roughness r as parameters. Each parameter is the extra property of the Gaussian primitive. The BRDF in Equation 6 can be devided into diffuse term $f _ { d }$ and a specular term $f _ { s }$ as follow:

$$
f _ { d } = \frac { 1 - m } { \pi } \alpha , \quad f _ { s } = \frac { D F G } { 4 ( \omega _ { i } \cdot n ) ( \omega _ { o } \cdot \mathbf { n } ) } ,\tag{7}
$$

where D, F , and $G$ represent the GGX normal distribution function, the Fresnel term, and the geometry term, respectively. See more details of the rednering equation that is uesd in this work in supplementary materials.

## Gaussian Splatting for Multi-view Photometric Stereo

Our goal is to simultaneously estimate geometry, materials, and lighting by multi-light images of a 3D object captured from N views. We define the multi-light images of each view m as $I ^ { m } = \{ I _ { 1 } ^ { m } , I _ { 2 } ^ { m } , \dots , I _ { l } ^ { m } \}$ }, where l denotes the index of light. Notably, we only focus on object-level IR in this work, and scene-level IR is beyond the scope.

Figure 1 illustrates the overall pipeline of our method. The proposed PS-GS consists of two stages to achieve efficient Gaussian-based inverse rendering for MVPS. In the first stage, we pretrain a standard 2DGS model as initialization. In the second stage, based on a pretrained model, we jointly optimize the surface normals, materials, and lighting, using the full rendering equation with a lighting prediction network. We also leverage multi-light images to estimate the guidance normals of the object for each view by UPS method, and the normal maps of the Gaussian model in both stages are regularized by the guidance normals. Moreover, we use the visibility computed by the modified 2D Gaussian ray-tracing that is suitable for SDL to regularize the MLPcomputed incident lighting.

<!-- image-->  
Figure 1: Overview of PS-GS. Starting with a set of multi-view and multi-light images, we pretrain a standard 2D Gaussian model for geometry initialization. Based on the pretrained model, we jointly optimize the geometry, materials, and lighting. To encourage accurate surface reconstruction, we compute guidance normals by the uncalibrated photometric stereo (UPS) method to regularize the rendered normal maps from the Gaussian model. We also modify the 2D Gaussian ray-tracing technique to suit single directional light (SDL), and leverage its visibility results to refine the incident lighting.

## Stage I: 2D Gaussian Pretraining

Before estimating geometry, materials, and lighting by inverse rendering with MVPS, we first train a standard 2DGS model. This stage can significantly decrease the training time by rapidly providing a reliable initialized geometry for the following inverse rendering process. During the rendering process, we also render depth D and normal maps $\mathcal { N }$ which is described by:

$$
\{ { \mathcal { D } } , { \mathcal { N } } \} = \sum _ { i = 1 } ^ { N } w _ { i } \{ d _ { i } , n _ { i } \} , { \mathrm { ~ w h e r e ~ } } w _ { i } = { \frac { T _ { i } a _ { i } } { \sum _ { i = 1 } ^ { N } T _ { i } a _ { i } } } .\tag{8}
$$

In Stage I, We optimize our model from a set of posed multi-view light-averaged images and their masks. The total loss function of this stage is defined as:

$$
{ \mathcal { L } } _ { \mathrm { S t a g e I } } = { \mathcal { L } } _ { c } + \lambda _ { n , c } { \mathcal { L } } _ { n , c } + \lambda _ { n , r } L _ { n , r } + \lambda _ { o } { \mathcal { L } } _ { o } ,\tag{9}
$$

where $\mathcal { L } _ { c }$ and ${ \mathcal { L } } _ { n , c }$ are the RGB reconstruction loss and normal consistency loss from seminal 2DGS (Huang et al. 2024), respectively. Î» denotes the corresponding loss weight.

To enhance the shape reconstruction and surface details, we leverage the normals estimated by the UPS method using multi-light images to regularize the rendered normal map:

$$
\mathcal { L } _ { n , r } = \sum _ { i = 1 } ^ { P } \left( \mathcal { N } _ { r } - T _ { c 2 w } ( \mathcal { N } _ { e } ) \right) ,\tag{10}
$$

where $\textstyle { \mathcal { N } } _ { r }$ is the rendered normal map from Gaussian model, $\mathcal { N } _ { e }$ is the UPS estimated normal, and $T _ { c 2 w }$ is the transformation from camera coordinate system to the world coordinate system. To decrease the creation of Gaussians outside the mask in image space, we also use a binary cross mask entropy loss:

$$
\mathcal { L } _ { o } = - M \log O - ( 1 - M ) \log ( 1 - O ) .\tag{11}
$$

## Stage II: Inverse Rendering with MVPS

With the initial geometry parameters $( \mathrm { i . e . , } p , t _ { u } , t _ { v } , s _ { u } , s _ { v } ,$ o, n) from Stage I, based on multi-view and multi-light images, we perform joint optimization using the full rendering equation with MLP-computed incident lighting. Different from previous works, such as GS-IR (Liang et al. 2024), we utilize deferred shading to capture sharper highlights and decrease the GPU memory usage. In the following subsections, we will describe each component of our method in detail.

Shape and Material Modeling Based on the pretrained 2D Gaussian model as initialized shape, the full rendering equation with BRDF model is used. We set the BRDF materials (i.e., Î±, r, and m) are the learnable parameters of each Gaussian, which is similar to the position p, and obtain the per-Gaussian normal by Equation 4. Then, the PBR with deferred shading is performed based on pixel-level feature maps given by a 2D Gaussian alpha-blending process:

$$
\mathbf { Q } = \sum _ { n \in N } \mathbf { q } _ { n } \alpha _ { n } T _ { n } , { \mathrm { w h e r e } } \mathbf { Q } = { \binom { A } { M } } , \mathbf { q } _ { i } = { \left[ \begin{array} { l } { \alpha _ { i } } \\ { m _ { i } } \\ { r _ { i } } \\ { n _ { i } } \end{array} \right] } \ .\tag{12}
$$

Gaussian splatting based deferred shading treats alphablending as a smoothing filter, enabling stabilized material optimization and yielding more reliable reconstruction. To further refine the surface details, we apply the regularization between the UPS estimated normals and rendered normals like Stage I.

Light Modeling Explicit parameterizing all single directional incident lights as learnable parameters of each Gaussian to be rendered is expensive. Rather than modeling the incident lighting explicitly, we utilize a global neural network to estimate all the single directional lighting for each Gaussian. This implicit incidence lighting network is described by:

$$
\hat { L } _ { i n } = f _ { \theta } \big ( \pmb { p } , t , \pmb { s } , \pmb { n } , \omega _ { i n } , \omega _ { o u t } \big ) ,\tag{13}
$$

where $f _ { \theta }$ and $\theta$ are the neural network and its parameters, respectively. We also apply a Fourier encoding to position p as in NeRF (Mildenhall et al. 2020), which is omitted in Equation 13. By exploiting the neural network to predict the incident lighting, we let the network implicitly learn the local and global lighting in the scene, which enables a GPU memory-efficient lighting modeling.

In order to constrain the neural network to predict physically reasonable results in radiance intensity and shadow, we modified the 2D Gaussian ray-tracing technique from IRGS (Gu et al. 2025) to be suitable for SDL setup. Specifically, we replace Monte Carlo sampling by SDL sampling in 2D Gaussian ray-tracing and perform ray-tracing on the reconstructed Gaussian model. To make training efficient, ray-tracing is only applied on the pretrained model obtained in Stage I, as the objectâs geometry exhibits slight changes in Stage II. Finally, we perform ${ \dot { \mathcal { L } } } _ { i n c } ,$ which is a $\mathcal { L } _ { 1 }$ loss, on the ray-tracing computed visibility and the network outputted lighting.

Rendering In Stage II, each image is illuminated by a specific directional light, and the rendering equation at this condition can be rewritten as:

$$
L ( \omega _ { o } , \pmb { r } ) = \hat { L } _ { i n } f ( \omega _ { o } , \omega _ { i } , \pmb { r } ) ( \pmb { \omega } _ { i } \cdot \pmb { n } ) ,\tag{14}
$$

where surface positions r can be derived from the rendered depth and normal maps for each pixel coordinate.

Total Training Loss The total training loss function utilized for this stage is:

$$
{ \mathcal { L } } _ { \mathrm { S t a g e I I } } = { \mathcal { L } } _ { c } + \lambda _ { n , c } { \mathcal { L } } _ { n , c } + \lambda _ { n , r } { \mathcal { L } } _ { n , r } + \lambda _ { o } { \mathcal { L } } _ { o } + \lambda _ { i n c } { \mathcal { L } } _ { i n c } ,\tag{15}
$$

which combines the loss function of Stage I and the incident lighting regularization term. Î» denotes the corresponding loss weight.

## Experimental

PS-GS enables efficient and accurate reconstruction of geometry, materials, and lighting. In the following, we present the comparison and ablation experiment results of PS-GS, which is evaluated on synthetic and real-world datasets. Please refer to supplementary materials for implementation details.

## Datasets and metrics

As MVPS is uesd in our method, we conduct experiments on multi-view multi-light datasets for the inverse rendering task, including a synthetic dataset: PS-NeRF Synthesis dataset (Yang et al. 2022) (including the object named Armadillo and Bunny), and a real-world dataset: DiLiGenT-MV dataset (Li et al. 2020) (including the object named Bear, Buddha, Reading, Cow, and Pot2). We adopt commonly used quantitative metrics for different results. Specifically, we use PSNR, SSIM (Wang et al. 2004), and LPIPS (Zhang et al. 2018) to evaluate the reconstructed NVS and relighted images. The mean angular error (MAE) in degrees is used for surface normal evaluation under test views. We compare PS-GS to two state-of-the-art MVPS-based inverse rendering methods: PS-NeRF and DPIR (Chung, Choi, and Baek 2024), as well as two state-of-the-art 2DGS-based inverse rendering methods: R3DG (Gao et al. 2024) and IRGS (Gu et al. 2025).

For the comparison experiments, we have two configurations. As R3DG and IRGS are assumed to have constant environment illumination, we leverage the lighting-averaged images similar to Stage I to train on 15 views and test on 5 novel views. Since PS-NeRF, DPIR, and our PS-GS render multi-view and multi-light images, similar to DPIR, we use 15 views and 16 lightings for training and use 5 novel views and 96 lightings for testing. For the comparison with R3DG and IRGS, we only compute the MAE of rendered normal maps, as the rendered images of these works vary from the MVPS-based works. This comparison strategy is adopted from the DPIR and PS-NeRF. For the comparison with DPIR and PS-NeRF, we simultaneously evaluate NVS, relighting, and normal MAE without any averaging.

<table><tr><td></td><td>Bear</td><td>Buddha</td><td>Read- ing</td><td>Cow</td><td>Pot2</td><td>Arma- dillo</td><td>Bunny</td></tr><tr><td colspan="8">PSNRâ</td></tr><tr><td>PS-NeRF | DPIR</td><td>|34.72 40.22</td><td>31.69 33.98</td><td>32.85 33.21</td><td>37.17</td><td>39.74 39.30 36.76</td><td>31.13 30.11</td><td>32.44 31.33</td></tr><tr><td>Ours</td><td>39.95</td><td>37.54</td><td>34.63</td><td>41.76</td><td>41.62</td><td>33.32</td><td>32.87</td></tr><tr><td></td><td></td><td></td><td></td><td>SSIMâ</td><td></td><td></td><td></td></tr><tr><td colspan="8"></td></tr><tr><td>PS-NeRF DPIR</td><td>|0.982</td><td>0.962</td><td>0.972</td><td>0.989</td><td>0.986</td><td>0.980</td><td>0.983</td></tr><tr><td>Ours</td><td>0.981</td><td>0.966</td><td>0.980</td><td>0.990</td><td>0.983</td><td>0.978</td><td>0.980</td></tr><tr><td></td><td>0.984</td><td>0.977</td><td>0.983  $\mathrm { L P I P S \downarrow ( \times 1 0 ^ { - 2 } ) }$ </td><td>0.992</td><td>0.989</td><td>0.987</td><td>0.987</td></tr><tr><td colspan="8"></td></tr><tr><td>PS-NeRF</td><td>3.91</td><td>3.62</td><td>2.86</td><td>2.22</td><td>2.96</td><td>2.08</td><td>1.50</td></tr><tr><td>DPIR</td><td>2.32</td><td>2.71</td><td>1.85</td><td>0.642</td><td>1.28</td><td>3.93</td><td>2.40</td></tr><tr><td>Ours</td><td>2.08</td><td>1.72</td><td>1.40</td><td>0.931</td><td>0.927</td><td>1.83</td><td>1.21</td></tr></table>

Table 1: Quantitative comparison of novel-view synthesis and relighting on the objects of DiLiGenT-MV dataset and PS-NeRF synthesis dataset. In this table, the best results are in bold.

## Comparation Results

Results for NVS and Relighting In Table 1, we present the experiment results of NVS and relighting task across DiliGent-MV and PS-NeRF datasets. PS-GS achieves stateof-the-art performance in most objects than previous methods, demonstrating its ability to accurately reconstruct. PS-GS also completes training in a relatively short time of 0.7 hours and low GPU memory usage of 6.5 GB, reflecting its efficiency in MVPS-based inverse rendering. Figure 2 illustrates a qualitative comparison against competitors, visualizing rendering images at NVS and relighting task. The results of PS-NeRF have some strange shadows and introduce over-smooth rendered images and relatively low overall intensity. DPIR, which employs point-based splatting and shadow, finds that the edge of its shadow exhibits distinct discrete point-like shadow. In contrast, our Gaussian Splatting based method enables smooth shadow edges, which makes rendering more realistic. We also provide the inverse rendering result in Figure 3, such as the estimated diffuse map, roughness map, metallic map, and incident lighting. PS-GS achieves more compelling reconstruction results.

Relighting

<!-- image-->  
Relighting  
GT image  
NVS  
Ours

<!-- image-->  
Relighting

<!-- image-->  
PS-NeRF  
NVS

<!-- image-->  
DPIR

Figure 2: Qualitative comparison of novel view synthesis (NVS) and relighting. The red box displays the details.

GT image

<!-- image-->  
Rendered  
Albedo  
Roughness  
Metallic

<!-- image-->  
Normal  
Incidence

Figure 3: Inverse rendering results on Bear, Reading, and Bunny.

Results for Normal Accuracy Table 2 shows the quantitative comparison results on estimated normal map between our method with both MVPS-based and Gaussian-based inverse rendering baselines. PS-GS demonstrates superior surface reconstruction compared to other methods, as demonstrated by the high quality of the normal maps. This improvement is largely attributed to the specifically designed geometry modeling and regularization strategies described above. As shown in Figure 4, the normal maps estimated by R3DG and IRGS are too smooth to present the details of the object surface. In contrast, PS-GS overcomes this challenge, achieving accuracy and more details for normal maps. Our approach utilizes multiple illumination images of each view, as opposed to fixed environment illumination, providing more information for reconstruction. Additionally, our method enhances the reasonableness and realism of normal maps efficiently, which is benefited by the advantage of the 2DGS model frame.

## Ablation Study

Regularization for Rendered Normal Maps Benefiting from the surface normals estimated by the UPS method, we regularize the rendered normal maps of the Gaussian model, which can mitigate ambiguity in surface estimation. Figure 5 and Table 3 present the result of our full model and the model without normal regularization. The absence of normal regularization leads to a smooth reconstruction result that lacks details. The full model significantly improves the rendered normal accuracy and recovered surface details.

<!-- image-->

<!-- image-->  
R3DG  
IR-GS

<!-- image-->

<!-- image-->  
PS-NeRF  
DPIR

<!-- image-->

<!-- image-->  
Ours  
GT

Figure 4: Qualitative comparison on normal accuracy.  
<!-- image-->  
GT

<!-- image-->  
Ours

<!-- image-->  
w/o Ln

Figure 5: Ablation study on normal regularization. The red box displays the details.

Regularization for Incident Lighting We modify the 2D Gaussian ray-tracing technique to be suitable for SDL. We then leverage its visibility result to regularize the MLPcomputed incident lighting. As shown in Figure 6 and Table 3, without using the regularization, the incident lighting is baked into the estimated albedo maps. The regularization helps to estimate relatively accurate intensity for both albedo maps and incident lighting.

<!-- image-->  
Figure 6: Ablation study on incident lighting regularization. We also provide the diffuse map rendered by DPIR and PS-NeRF for reference.

The Standard 2DGS Model Pretraining The results, as shown in Table 3, emphasize the superiority of the standard 2DGS model pretraining over only performing inverse rendering with an equal number of training iterations. Simultaneously recovering the geometry, materials, and lighting only from input images is difficult. The usage of standard 2DGS pretraining process can provide a reliable geometry initialization before the inverse rendering, which offers superiority in both the quality of reconstruction and the speed of shape convergence.

<table><tr><td></td><td>Bear</td><td>Buddha</td><td>Reading</td><td>Cow</td><td>Pot2</td><td>Armadillo</td><td>Bunny</td><td>GPU memoryâ</td><td>Timeâ</td></tr><tr><td>R3DG</td><td>10.48</td><td>22.08</td><td>19.25</td><td>9.13</td><td>13.00</td><td>15.15</td><td>15.12</td><td>8 G</td><td>1 h</td></tr><tr><td>IRGS</td><td>11.48</td><td>23.38</td><td>17.80</td><td>11.54</td><td>16.16</td><td>15.51</td><td>15.60</td><td>7 G</td><td>0.7 h</td></tr><tr><td>PS-NeRF</td><td>5.03</td><td>12.35</td><td>9.37</td><td>5.96</td><td>7.59</td><td>5.03</td><td>5.53</td><td>9.6 G</td><td>&gt;22 h</td></tr><tr><td>DPIR</td><td>4.86</td><td>11.26</td><td>8.95</td><td>4.36</td><td>6.50</td><td>4.41</td><td>5.53</td><td>9 G</td><td>2 h</td></tr><tr><td>Ours</td><td>3.68</td><td>8.92</td><td>8.21</td><td>4.87</td><td>6.08</td><td>3.53</td><td>3.47</td><td>6.5 G</td><td>0.7 h</td></tr></table>

Table 2: Qualitative comparison of normal accuracy (quantified by MAE(â)) on the objects of DiLiGenT-MV dataset and PS-NeRF synthesis dataset. In this table, the best results are in bold.

<table><tr><td rowspan="2"></td><td rowspan="2">PSNRâ SSIMâ</td><td rowspan="2"> $\begin{array} { c } { { \mathrm { L P I P S \downarrow } } } \\ { { ( \times 1 0 ^ { - 2 } ) } } \end{array}$ </td><td rowspan="2">MAEâ</td><td rowspan="2">Timeâ</td></tr><tr><td></td></tr><tr><td>w/o  ${ \mathcal { L } } _ { n }$ </td><td>37.50</td><td>0.984</td><td>1.50</td><td>7.02 0.7 h</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { i n c }$ </td><td>37.92</td><td>0.985</td><td>1.37 5.64</td><td>0.7 h</td></tr><tr><td>w/o Stage I</td><td>37.05</td><td>0.984</td><td>1.51 5.69</td><td>2.0 h</td></tr><tr><td>10 views</td><td>37.17</td><td>0.984</td><td>1.55 5.72</td><td>0.6 h</td></tr><tr><td>5 views</td><td>34.93</td><td>0.970</td><td>2.49 8.34</td><td>0.5 h</td></tr><tr><td>Ours</td><td>37.67</td><td>0.986</td><td>1.44 5.57</td><td>0.7 h</td></tr></table>

Table 3: Ablation studies on various components of PS-GS. In this table, the results are averaged among all the objects, and the best results are in bold.

Experiments on Training Views Based on the MVPS technique, our method exploits the multi-view and multilight images in the inverse rendering process, which is able to mitigate the inherent ambiguity of constant illuminated images. Table 3 and Figure 7 show the reconstruction results with varying numbers of training views. Despite the decline of the metric values, our method obtained satisfying reconstruction results even when only a 5-view input is provided, which demonstrates the ability to reconstruct in sparse views.

GT  
<!-- image-->

<!-- image-->  
Ours

<!-- image-->  
10-views

<!-- image-->  
5-views

Figure 7: Ablation study on training views.

## Applications

Owing to the explicit Gaussian primitive representation, materials and geometry editing can be achieved conveniently through PS-GS by editing the parameters of Gaussian primitives. Specifically, PS-GS allows material editing for both diffuse and specular term through replacing them with alternatives or adjusting the intensity of the reconstructed materials. Furthermore, the intuitive shape removal is supported by our method via simply removing the Gaussian primitives from the reconstructed geometry model. The editing results for materials and geometry editing of PS-GS are shown in Figure 8.

<!-- image-->

<!-- image-->  
Shape Removal  
Figure 8: Applications on PS-GS, including shape removal and material editing. For material editing, the first column is the result of replacing the diffuse map with the alternative, and the second column is the result of reducing the intensity of the specular map.

## Conclusion

In this paper, we introduce Gaussian splatting for Multi-view Photometric Stereo (PS-GS), an inverse rendering method that integrates 2D Gaussian Splatting and physically-based deferred shading for MVPS. Our method first reconstructs an object with a standard 2D Gaussian model as initialization. It then jointly optimizes geometry, materials, and lighting of the object by multi-view and multi-light images based on the full rendering equation with a lighting prediction network. To mitigate the inherent ambiguity of inverse rendering, we also perform the regularization for the rendered normal maps by the normals obtained from multi-light images via the UPS method. We further propose a regularization on the estimated lighting based on the visibility computed by a modified 2D Gaussian ray-tracing method that is suitable for SDL. Experiments on both synthetic and real datasets demonstrate the superiority of PS-GS over existing methods in terms of quantitative metrics, visual quality, and efficiency. Notably, our method enables high-quality reconstructions even under 5 input views.

## References

Barron, J. T.; Mildenhall, B.; Tancik, M.; Hedman, P.; Martin-Brualla, R.; and Srinivasan, P. P. 2021. Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 5835â5844. IEEE Computer Society.

Brument, B.; Bruneau, R.; Queau, Y.; M Â´ elou, J.; Lauze, Â´ F. B.; Durou, J.-D.; and Calvet, L. 2024. RNb-NeuS: Reflectance and Normal-Based Multi-View 3D Reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5230â5239. IEEE Computer Society.

Burley, B.; and Studios, W. D. A. 2012. Physically-based shading at Disney. In ACM SIGGRAPH.

Chen, G.; Han, K.; Shi, B.; Matsushita, Y.; and Wong, K.- Y. K. 2019. Self-Calibrating Deep Photometric Stereo Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE Computer Society.

Chung, H.-G.; Choi, S.; and Baek, S.-H. 2024. Differentiable Point-based Inverse Rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4399â4409. IEEE Computer Society.

Gao, J.; Gu, C.; Lin, Y.; Li, Z.; Zhu, H.; Cao, X.; Zhang, L.; and Yao, Y. 2024. Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing. In European conference on computer vision, 73â 89. Springer.

Gu, C.; Wei, X.; Zeng, Z.; Yao, Y.; and Zhang, L. 2025. IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 10943â10952. IEEE Computer Society.

Hargreaves, S.; and Harris, M. 2004. Deferred shading. In Game Developers Conference.

Hu, H.; Fan, Z.; Wu, T.; Xi, Y.; Lee, S.; Pavlakos, G.; and Wang, Z. 2025. Expressive Gaussian human avatars from monocular RGB video. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS â24. Red Hook, NY, USA: Curran Associates Inc. ISBN 9798331314385.

Hu, T.; Liu, S.; Chen, Y.; Shen, T.; and Jia, J. 2022. Efficient-NeRF Efficient Neural Radiance Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12902â12911. IEEE Computer Society.

Hu, W.; Wang, Y.; Ma, L.; Yang, B.; Gao, L.; Liu, X.; and Ma, Y. 2023. Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19717â19726. IEEE Computer Society.

Huang, B.; Yu, Z.; Chen, A.; Geiger, A.; and Gao, S. 2024. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In ACM SIGGRAPH 2024 Conference Papers, 1â11.

Ikehata, S. 2023. Scalable, Detailed and Mask-Free Universal Photometric Stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13198â13207. IEEE Computer Society.

Jiang, Y.; Tu, J.; Liu, Y.; Gao, X.; Long, X.; Wang, W.; and Ma, Y. 2024. GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5322â5332. IEEE Computer Society.

Kaya, B.; Kumar, S.; Sarno, F.; Ferrari, V.; and Van Gool, L. 2022. Neural Radiance Fields Approach to Deep Multi-View Photometric Stereo. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 1965â1977. IEEE Computer Society.

Kerbl, B.; Kopanas, G.; Leimkuehler, T.; and Drettakis, G. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph., 42(4).

Li, M.; Zhou, Z.; Wu, Z.; Shi, B.; Diao, C.; and Tan, P. 2020. Multi-View Photometric Stereo: A Robust Solution and Benchmark Dataset for Spatially Varying Isotropic Materials. IEEE Transactions on Image Processing, 29: 4159â 4173.

Li, Z.; Chen, Z.; Li, Z.; and Xu, Y. 2024a. Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8508â8520. IEEE Computer Society.

Li, Z.; Zheng, Z.; Wang, L.; and Liu, Y. 2024b. Animatable Gaussians: Learning Pose-Dependent Gaussian Maps for High-Fidelity Human Avatar Modeling. In Processings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19711â19722. IEEE Computer Society.

Liang, Z.; Zhang, Q.; Feng, Y.; Shan, Y.; and Jia, K. 2024. GS-IR: 3D Gaussian Splatting for Inverse Rendering. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21644â21653. IEEE Computer Society.

Logothetis, F.; Mecca, R.; and Cipolla, R. 2019. A Differential Volumetric Approach to Multi-View Photometric Stereo. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 1052â1061. IEEE Computer Society.

Matsuki, H.; Murai, R.; Kelly, P. H. J.; and Davison, A. J. 2024. Gaussian Splatting SLAM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18039â18048. IEEE Computer Society.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In European Conference on Computer Vision, 405â421. Springer.

Muller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In-Â¨ stant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4).

Park, J.; Sinha, S. N.; Matsushita, Y.; Tai, Y.-W.; and Kweon, I. S. 2017. Robust Multiview Photometric Stereo Using Planar Mesh Parameterization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(8): 1591â1604.

Shi, R.; Wei, X.; Wang, C.; and Su, H. 2024. ZeroRF: Fast Sparse View 360Â° Reconstruction with Zero Pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21114â21124. IEEE Computer Society.

Wang, G.; Chen, Z.; Loy, C. C.; and Liu, Z. 2023. SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 9065â9076. IEEE Computer Society.

Wang, H.; Hu, W.; Zhu, L.; and Lau, R. W. 2024. Inverse Rendering of Glossy Objects via the Neural Plenoptic Function and Radiance Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19999â20008. IEEE Computer Society.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600â612.

Wu, S.; Basu, S.; Broedermann, T.; Van Gool, L.; and Sakaridis, C. 2025. PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE Computer Society.

Yan, C.; Qu, D.; Xu, D.; Zhao, B.; Wang, Z.; Wang, D.; and Li, X. 2024. GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19595â19604. IEEE Computer Society.

Yang, W.; Chen, G.; Chen, C.; Chen, Z.; and Wong, K.-Y. K. 2022. PS-NeRF: Neural Inverse Rendering for Multi-view Photometric Stereo. In European conference on computer vision, 266â284. Springer.

Yang, Z.; Gao, X.; Zhou, W.; Jiao, S.; Zhang, Y.; and Jin, X. 2024. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20331â20341. IEEE Computer Society.

Yao, Y.; Zeng, Z.; Gu, C.; Zhu, X.; and Zhang, L. 2025. Reflective Gaussian Splatting. In Processings of International Conference on Representation Learning, volume 2025, 68695â68711.

Yao, Y.; Zhang, J.; Liu, J.; Qu, Y.; Fang, T.; McKinnon, D.; Tsin, Y.; and Quan, L. 2022. Neilf: Neural incident light field for physically-based material estimation. In European conference on computer vision, 700â716. Springer.

Zhang, J.; Yao, Y.; Li, S.; Liu, J.; Fang, T.; McKinnon, D.; Tsin, Y.; and Quan, L. 2023. NeILF++: Inter-Reflectable Light Fields for Geometry and Material Estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3601â3610. IEEE Computer Society.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE Computer Society.

Zhang, X.; Srinivasan, P. P.; Deng, B.; Debevec, P.; Freeman, W. T.; and Barron, J. T. 2021. NeRFactor: neural factorization of shape and reflectance under an unknown illumination. ACM Trans. Graph., 40(6).