# ARGS: Advanced Regularization on Aligning Gaussians over the Surface

Jeong Uk Lee KAIST jeong19@kaist.ac.kr

Sung Hee Choi KAIST sunghee@kaist.edu

## Abstract

Reconstructing high-quality 3D meshes and visuals from 3D Gaussian Splatting(3DGS) still remains a central challenge in computer graphics. Although existing models such as SuGaR offer effective solutions for rendering, there is is still room to improve improve both visual fidelity and scene consistency. This work builds upon SuGaR by introducing two complementary regularization strategies that address common limitations in both the shape of individual Gaussians and the coherence of the overall surface. The first strategy introduces an effective rank regularization, motivated by recent studies on Gaussian primitive structures. This regularization discourages extreme anisotropyâspecifically, âneedle-likeâ shapesâby favoring more balanced, âdisk-likeâ forms that are better suited for stable surface reconstruction. The second strategy integrates a neural Signed Distance Function (SDF) into the optimization process. The SDF is regularized with an Eikonal loss to maintain proper distance properties and provides a continuous global surface prior, guiding Gaussians toward better alignment with the underlying geometry. These two regularizations aim to improve both the fidelity of individual Gaussian primitives and their collective surface behavior. The final model can make more accurate and coherent visuals from 3DGS data.

## 1. Introduction

The ability to reconstruct detailed 3D models from collections of 2D images has long been a central pursuit in computer vision and graphics. Recently, 3DGS [5] has emerged as a groundbreaking technique, achieving realtime rendering speeds and state-of-the-art quality for novel view synthesis. By representing scenes with millions of explicit 3D Gaussian primitives and employing a differentiable tile-based rasterizer, 3DGS bypasses the costly neural network queries inherent in methods like Neural Radiance Fields(NeRFs) [6], thus offering significant advantages in training and rendering efficiency. This advancement has accelerated considerable interest in various applications, from virtual and augmented reality to interactive content creation.

Despite its success in rendering, converting the optimized, and often unstructured, set of 3D Gaussians into high-quality, explicit surface meshesâa preferred representation for many downstream graphics applications like editing, simulation, and animationâremains a significant challenge. The optimization process in 3DGS can result in Gaussians becoming unorganized or adopting highly anisotropic shapes [4] where one variance dominates the others. Such configurations can lead to visual artifacts and suboptimal geometric details in novel views or areas with sparse input data.

To address these issues, SuGaR(Surface-Aligned Gaussian Splatting) [2] was introduced as a method for efficient 3D mesh reconstruction and high-quality rendering from 3DGS representations. SuGaR employs a regularization term to encourage Gaussians to align with scene surfaces, extracts a mesh using Poisson reconstruction, and then binds new Gaussians to this mesh for refined rendering and editability. While SuGaR significantly improves mesh extraction from 3DGS, the geometric conditioning of individual Gaussians and the global coherence of the represented surface offer avenues for further enhancement.

This paper proposes two synergistic enhancements to the SuGaR framework, aiming to improve both the fidelity of individual Gaussian primitives and the overall consistency of the reconstructed surface. Building on the effective rank regularization introduced by [4], our method is able to represent local patches well with respect to the object surfaces. This term directly analyzes the covariance structure of each Gaussian, penalizing the formation of overly elongated, needle-like shapes and encouraging more voluminous, disk-like primitives. Such well-conditioned Gaussians are more robust for representing local surface and can reduce geometric artifacts.

Our second regularization incorporates a co-optimized neural Signed Distance Function(SDF) to serve as a global surface prior. Inspired by advancements in neural implicit surface learning, such as NeuS [11] or VolSDF [17], we jointly train a small multi-layer perceptron (MLP) to represent an SDF of the scene alongside the Gaussian optimization. This neural SDF is regularized by an Eikonal term to ensure it maintains valid distance field properties. A novel loss component then encourages the 3D Gaussians to concentrate near the zero-level set of this dynamically learned SDF, promoting a more globally coherent and continuous surface representation across the entire scene.

By combining these principled regularization strategies, our enhanced SuGaR methodology aims to achieve significantly improved geometric accuracy, reduce surface noise and artifacts, and produce more detailed and consistent mesh rendering from 3D Gaussian Splatting. This work seeks to further bridge the gap between the efficient rendering capabilities of 3DGS and the demand for high-quality, structured 3D assets in modern graphics pipelines.

## 2. Related Work

Our work is built upon recent advancements in 3DGS and methods for mesh conversion and geometric regularization, as well as techniques for neural implicit surface representation. We situate our contributions within these areas.

## 2.1. Mesh Conversion from Neural Volumetric Representations

Obtaining high-quality, explicit surface meshes from neural volumetric representations like 3D Gaussian Splatting or Neural Radiance Fields is crucial for many downstream applications but presents distinct challenges.

For 3DGS, which achieves real-time rendering with explicit primitives, the primary challenge lies in interpreting a surface from a collection of potentially unorganized Gaussians [5]. SuGaR, the foundation of our work, addresses this by regularizing Gaussians for surface alignment and then extracting a mesh via Poisson reconstruction from a derived density field [2]. Other 3DGS-focused methods for geometry improvement or meshing include 2DGS [3], which uses 2D Gaussian disks for view-consistent geometry, GOF (Gaussian Opacity Fields) [19], which employs ray-Gaussian intersection for density estimation and regularization, and NeuSG [1], which combines 3DGS with neural implicit models for surface refinement. GS2Mesh [12] uses stereo image pairs to estimate depth maps and get Truncated Signed Distance Function (TSDF) volume. Then it uses Marching Cube algorithm to extract mesh.

For NeRFs, which learn a continuous volumetric density and radiance, extracting accurate geometry is difficult because NeRFs are optimized for novel view synthesis, often resulting in fuzzy or noisy geometry when a simple density threshold is used. To overcome this, NeRFMeshing [9] proposes distilling a pre-trained NeRF into a Truncated Signed Distance Field represented by a Signed Surface Approximation Network(SSAN). The SSAN is supervised using depth percentile estimates rendered from the NeRF to approximate points inside, on, and outside the surface, respectively.

A mesh is then extracted from this TSDF using Marching Cubes. NeRF2Mesh [10] distill the geometry encoded in a trained NeRF into a SDF. Eikonal Loss is applied to the SDF to enforce it to behave like a true distance function. Then Marching cube is used for mesh extraction.

## 2.2. Regularization of 3D Gaussian Primitives

The rendering quality of meshes derived from 3DGS heavily relies on the shape and disposition of the Gaussian primitives. Unconstrained, these primitives can adopt undesirable anisotropic forms, such as âneedle-likeâ shapes, leading to artifacts [4].

SuGaRâs original regularization encourages surface alignment by matching distance values derived from Gaussians to depth-based cues. This is an effective approach but may not fully prevent all degeneracies in individual Gaussian shapes. The Effective Rank Regularization, which forms one of our key enhancements, directly addresses this by analyzing the covariance of each Gaussian. Along with Effective Rank regularization, FreGS [20] uses Total Variation(TV) regularization which encourages smoothness in Gaussian positions and features to promote spatial coherence.

## 2.3. Neural Implicit Surfaces and SDF-based Reconstruction

Neural implicit representations, especially SDF, have proven powerful for high-fidelity 3D surface reconstruction [11, 17]. An SDF defines a surface as its zero-level set, $f ( x ) = 0$ , and ideally satisfies the Eikonal constraint, $\| \nabla f ( \pmb { x } ) \| _ { 2 } = 1$ , for true distance properties.

Several methods learn neural SDFs from multi-view images. IDR (Multiview Neural Surface Reconstruction) [16] uses differentiable surface rendering but often requires masks and can face optimization challenges with complex scenes.

To improve robustness, recent works combine SDF with volume rendering principles. [11] introduces a volume rendering scheme based on an SDF-derived density, enabling mask-free training and better handling of complex geometry. [17] proposes modeling the volume density directly as a function of the SDF, specifically using Laplaceâs Cumulative Distribution Function(CDF) applied to the SDF values. This formulation is argued to provide a beneficial inductive bias, allow for accurate ray sampling via opacity error bounds, and facilitate shape/appearance disentanglement. Both NeuS and VolSDF emphasize the Eikonal loss for regularizing the learned SDF. Other related works include UNISURF [8], which learns an occupancy field via volume rendering.

As mentioned, [9] also leverages a learned implicit representation by distilling it from a pre-trained NeRF using depth cues derived from the NeRFâs volume rendering. This implicit representation then allows for standard mesh extraction.

<!-- image-->  
Figure 1. The general pipeline of our model. The images are processed in 3DGS and SuGaR to create mesh. During the process, we incorporate additional regularization to better match the gaussians to the surface.

Our work is inspired by the success of these methods in achieving high-quality surfaces rendering via learned SDFs. We propose to integrate a co-optimized neural SDF, similar in spirit to those in NeuS or VolSDF but learned in conjunction with explicit Gaussian primitives, to serve as a global surface prior within the SuGaR framework.

## 2.4. Positioning the Proposed Work

Our method enhances the SuGaR framework through two primary, synergistic regularization strategies. The effective rank regularization targets the local geometric quality of individual Gaussians, promoting well-conditioned, disk-like shapes. Simultaneously, the co-optimized neural SDF, inspired by principles from implicit surface learning methods such as NeuS and VolSDF. This dual sided learning can be found in many other works. For example, methods like GSurf [15] learns a SDF that is directly supervised by the 3D Gaussian primitives. This approach aims to tightly couple the two representations by training them concurrently. GSDF [18] employs a novel dual-branch structure where 3DGS branch offers efficient rendering, while the SDF branch manages a neural Signed Distance Field, focused on accurate surface reconstruction. This dual strategy aims to improve both the intrinsic quality of the Gaussian primitives and their collective adherence to a coherent surface.

## 3. Preliminary

In this section, we introduce the concept of effective rank regularization from [4], a key component for our methodâs ability to create well-conditioned Gaussians that represent local surface patches. The effective rank erank $\left( \mathcal { G } _ { K } \right)$ is computed from its 3D scaling factors $\pmb { s } _ { k } = ( s _ { k 1 } , s _ { k 2 } , s _ { k 3 } )$

We first need to obtain normalized squared scales.

$$
\mathbf { q } = ( q _ { 1 } , q _ { 2 } , q _ { 3 } ) = \left( { \frac { s _ { 1 } ^ { 2 } } { S } } , { \frac { s _ { 2 } ^ { 2 } } { S } } , { \frac { s _ { 3 } ^ { 2 } } { S } } \right) , \quad S = \sum _ { i = 1 } ^ { 3 } s _ { i } ^ { 2 }\tag{1}
$$

Then we need to take Shannon entropy of these terms.

$$
\pmb { H } ( \mathcal { G } _ { k } ) = \pmb { H } ( q _ { 1 } , q _ { 2 } , q _ { 3 } ) = - \sum _ { i = 1 } ^ { 3 } q _ { i } \log q _ { i }\tag{2}
$$

We can obtain the effective rank of Gaussians by then taking its exponent.

$$
\mathrm { e r a n k } ( \mathcal G _ { K } ) = \exp \{ H ( \mathcal G _ { k } ) \}\tag{3}
$$

A penalty term is applied, which sharply increases as erank $\left( \mathcal { G } _ { K } \right)$ approaches to 1. This term is formulated as max $( - \log ( \operatorname { e r a n k } ( \mathcal { G } _ { k } ) - 1 + \epsilon ) , 0 )$ where Ïµ is a small constant for numerical stability. The smallest scaling factor $s _ { k 3 }$ of the Gaussian is added to this penalty. The final per-Gaussian loss can be written as:

$$
\mathcal { L } _ { e f f } = \sum _ { k } \operatorname* { m a x } \left( - \log ( \operatorname { e r a n k } ( \mathcal { G } _ { k } ) - 1 + \epsilon ) , 0 \right) + s _ { k 3 }\tag{4}
$$

## 4. Method

## 4.1. Overall Pipeline

Our approach enhances the SuGaR framework[2] for 3D mesh conversion and rendering from 3DGS. We introduce two primary regularization strategies during SuGaRâs coarse optimization phase: an effective rank regularization to refine the local geometry of individual Gaussian primitives and a co-optimized neural Signed Distance Function to establish a global surface prior. These components work synergistically to improve the overall look and coherence of the mesh.

<!-- image-->  
Figure 2. In the original Gaussian Splatting, many Gaussians degenerate into âneedle-likeâ shape. Also we prefer Gaussians to align to the surface. For these reasons, we optimize Gaussians to have effective rank of 2.

The core of our method integrates these novel regularizations into the alignment stage of gaussians of SuGaR. The pipeline generally proceeds as follows:

1. Optimization: The process begins with a 3DGS model that has performed the initial optimization, typically from a sparse point cloud derived from Structure-from-Motion.

2. Coarse SuGaR Optimization with Novel Regularizers: This stage is crucial for further optimization of the Gaussian primitives. Alongside SuGaRâs original surface alignment objectives, our two key regularization components are applied. First the effective rank regularization is introduced to improve the shape of individual Gaussians. Second, a neural SDF network is jointly trained, and associated losses (Eikonal and SDF-Gaussian consistency) are applied to enforce a global, continuous surface structure. The loss function that is then applied can be written as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { s u g a r } + \lambda _ { S D F - G a u s s } \mathcal { L } _ { e f f } } \\ { + \lambda _ { 2 } \mathcal { L } _ { E i k o n a l } + \lambda _ { 3 } \mathcal { L } _ { S D F - G a u s s } } \end{array}\tag{5}
$$

3. Mesh Exteraction: After the coarse optimization, a 3D mesh is extracted from the regularized Gaussian representation using SuGaRâs established procedure, which involves sampling points on a density level set and applying Poisson reconstruction.

4. Further Refinement: The extracted mesh can subsequently be used for SuGaRâs refinement stage, where new Gaussians are bound to the mesh and further optimized. Our primary contributions focus on enhancing the coarse optimization to yield a superior mesh

rendering quality.

These regularizations are activated after an initial number of training iterations to allow the Gaussians to settle roughly in their place. The overall pipeline of the model is displayed in Fig. 1.

## 4.2. Effective Rank Regularization for Gaussian Primitives

Following the method proposed by Hyung et al. [4], we incorporate an effective rank regularization loss to prevent 3D Gaussians from degenerating into undesirable âneedle-likeâ shapes (effective rank â 1). This loss encourages each Gaussian primitive to adopt a more geometrically stable, âdisk-likeâ shape (effective rank â 2), which is better suited for representing surface patches. The visualization of effective rank is shown in Fig. 2. The effective rank is computed from the Gaussianâs 3D scaling factors. The per-Gaussian loss, adapted from [4], is then formulated as Eq. (7). This loss is averaged over all Gaussians and added to the main training objective, weighted by a hyperparameter $\lambda _ { e r a n k }$

## 4.3. Implicit SDF Constraint for Global Surface Coherence

To provide a strong global prior for surface geometry and encourage a more coherent arrangement of Gaussian primitives, we jointly optimize a neural network representing an implicit SDF of the scene, $f _ { n e u r a l } ( x )$ , alongside the Gaussians. The SDF is represented by a Multi-Layer Perceptron (MLP), termed Implicit SDFNet in our model. As shown in Fig. 3, this network takes 3D point coordinates as input and outputs a scalar SDF value. However, standard Multi-Layer Perceptrons tend to learn low-frequency variations in data much more easily than high-frequency variations. This is known as âspectral bias.â To capture high-frequency details, input coordinates are first transformed using positional encoding (e.g., 6 frequency bands using sinusoidal functions).

<!-- image-->  
Figure 3. The Implicit SDFNet is comprised of 8 hidden layers with 256 neurons per each layer. Input vector with positional encoding in fed into the network to output a single SDF value.

sin $\iota ( 2 ^ { i } \cdot x )$ and $\cos ( 2 ^ { i } \cdot x )$ for different i.

The MLP consists of 8 hidden layers with 256 neurons each using a Softplus activation function. Skip connections are incorporated between the original encoded input to an intermediate layerâs input to improve gradient flow and detail representation. The network weights are initialized as 0 with small negative value on output layer bias, similar to practices in [11] and [16], to promote stable SDF learning.

Eikonal Loss: To ensure that $f _ { n e u r a l } ( x )$ behaves as a valid distance field, an Eikonal loss is applied. This loss penalizes deviations of the $\mathrm { { s D F } } { \mathrm { { s } } }$ gradient norm from being 1:

$$
\mathcal { L } _ { E i k o n a l } = \mathbb { E } _ { x } \left( \| \nabla f _ { n e u r a l } ( { \pmb x } ) \| _ { 2 } - 1 \right) ^ { 2 }\tag{6}
$$

Points $x$ for this loss are sampled around Gaussians in the scene, and gradients are computed via automatic differentiation. This loss is weighted by $\lambda _ { e i k o n a l }$

SDF-Gaussian Consistency Loss: We introduce To ensure a consistency loss so that the explicit Gaussians can align to the learned implicit surface. This loss encourages the regions represented by the Gaussian primitives to lie on or near the zero-level set of $f _ { n e u r a l } ( x )$ . This is achieved by sampling points $x _ { i }$ based on the distribution of the current Gaussian primitives and penalizing the squared SDF values at these locations:

$$
\mathcal { L } _ { \mathrm { S D F - G a u s s } } = \mathbb { E } _ { x _ { i } \sim \mathrm { G a u s s i a n s } } \left( f _ { n e u r a l } ( x _ { i } ) \right) ^ { 2 }\tag{7}
$$

Since the distance between sampled point and SDF surface gets penalized, the Gaussians and SDF will align as we iterate the process. This loss is weighted by $\lambda _ { S D F - G a u s s } .$

By integrating these regularization techniques, our methodology aims to produce Gaussians that are individually well-shaped and collectively define a more accurate and globally consistent surface. At the same time, learned SDF converges to objectâs true shape, making the Gaussians align more closely with the surface and thereby improving the rendering quality of the subsequently converted mesh.

## 5. Experiment

We used the Mip-NeRF 360 dataset for our experiments. First, we ran 7,000 iterations of vanilla Gaussian Splatting to allow the Gaussians to form a coarse shape. Then, for 8,000 iterations, we performed the Gaussian alignment process using our defined loss functions. The experiments were conducted on an RTX 4090 GPU, and mesh conversion took approximately 1.5 to 2 hours.

## 5.1. Result

In Tab. 1, we show quantitative result comparing basic SuGaR model and our newly proposed model. The result show that our method shows better rendering quality in many different metrics.

We also show qualitative result with Fig. 4 of actual mesh renderings on Mip-NeRF 360 dataset. The rendered images show that extracted mesh with our method covers more area(less white areas) and show better rendering. Also note that original SuGaR model cannot properly render the electric wire at upper right side of the image whereas our model does it better. Also our model doesnât have artifacts that original SuGaR model had as shown in the middle and bottom row.

<table><tr><td rowspan="2">Method</td><td colspan="3">Stump</td><td colspan="3">Bonsai</td></tr><tr><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>2DGS</td><td>0.7885</td><td>26.5749</td><td>0.1948</td><td>0.9508</td><td>31.8650</td><td>0.0730</td></tr><tr><td>SuGaR</td><td>0.7657</td><td>26.8959</td><td>0.2056</td><td>0.9477</td><td>31.9473</td><td>0.0955</td></tr><tr><td>Ours</td><td>0.7927</td><td>27.2257</td><td>0.1922</td><td>0.9509</td><td>32.3401</td><td>0.0823</td></tr><tr><td>Method</td><td></td><td>Counter</td><td></td><td></td><td>Garden</td><td></td></tr><tr><td rowspan="2"></td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>0.9102</td><td>28.9066</td><td>0.0967</td><td>0.8906</td><td>28.0669</td><td>0.0855</td></tr><tr><td>2DGS SuGaR</td><td>0.9059</td><td>28.4876</td><td>0.1175</td><td>0.9124</td><td>29.5415</td><td>0.0823</td></tr><tr><td>Ours</td><td>0.9120</td><td>29.0136</td><td>0.1089</td><td>0.9252</td><td>30.3401</td><td>0.0783</td></tr><tr><td>Method</td><td></td><td>Kitchen</td><td></td><td></td><td>Room</td><td></td></tr><tr><td rowspan="2"></td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td></td></tr><tr><td>0.9468</td><td>31.1812</td><td>0.0477</td><td>0.9404</td><td></td><td>LPIPSâ</td></tr><tr><td>2DGS SuGaR</td><td>0.9462</td><td>31.5774</td><td>0.0587</td><td>0.9078</td><td>30.3269</td><td>0.1004</td></tr><tr><td>Ours</td><td>0.9524</td><td>31.9214</td><td>0.0496</td><td>0.9812</td><td>29.7830 30.5670</td><td>0.1294 0.1121</td></tr></table>

Table 1. Our method with advanced regularization show better performance than original SuGaR model.  
<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Ground Truth

<!-- image-->  
SuGaR

<!-- image-->  
Ours  
Figure 4. Qualitative mesh render comparison between SuGaR and our model.

## 5.2. Ablation Study

In Tab. 2, we demonstrate the effectiveness of our method. We have conducted an ablation study without using Effective Rank regularization and Implicit SDF regularization. We observed that including either effective rank or implicit net regularization slightly improves the overall performance. Including both regularizations further improves the rendering quality. The regularization steps add up to 20 minutes to the process, but we prioritized better rendering quality over the increased time.

<table><tr><td rowspan="2">Method</td><td colspan="3">Stump</td><td colspan="3">Bonsai</td></tr><tr><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>w/o Any regularization</td><td>0.7657</td><td>26.8959</td><td>0.2056</td><td>0.9477</td><td>31.9473</td><td>0.0955</td></tr><tr><td>w/o Effective Rank</td><td>0.7792</td><td>27.0218</td><td>0.2001</td><td>0.9501</td><td>32.1174</td><td>0.0933</td></tr><tr><td>w/o Implicit Net</td><td>0.7649</td><td>26.9913</td><td>0.1948</td><td>0.9496</td><td>31.9870</td><td>0.0938</td></tr><tr><td>Ours</td><td>0.7927</td><td>27.2257</td><td>0.1922</td><td>0.9509</td><td>32.3401</td><td>0.0823</td></tr><tr><td rowspan="3">Method</td><td></td><td>Counter</td><td></td><td></td><td>Garden</td><td></td></tr><tr><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>0.9059</td><td>28.4876</td><td>0.1175</td><td>0.9124</td><td>29.5415</td><td>0.0823</td></tr><tr><td>w/o Any regularization w/o Effective Rank</td><td>0.9089</td><td>29.0072</td><td>0.1116</td><td>0.9188</td><td>29.8991</td><td>0.0801</td></tr><tr><td>w/o Implicit Net</td><td>0.9068</td><td>28.8721</td><td>0.1136</td><td>0.9209</td><td>30.0290</td><td>0.0798</td></tr><tr><td>Ours</td><td>0.9120</td><td>29.0136</td><td>0.1089</td><td>0.9252</td><td>30.3401</td><td>0.0783</td></tr></table>

Table 2. We get higher rendering quality by incorporating our regularization terms.

## 6. Future Work

While our proposed enhancements demonstrate promising mesh rendering quality from 3D Gaussian Splatting via the SuGaR framework, several limitations and avenues for future research exist:

Limitations from vanilla GS: Since our approach heavily relies on the quality of the Gaussians fed into SuGaRâs mesh extraction pipeline, the limitations of vanilla Gaussian Splatting persist in our model. For instance, non-Lambertian surfaces like glossy or reflective materials may still show significant challenges. The effectiveness of the current regularization under such extreme conditions requires further evaluation. Recent works have explored combining Gaussian Splatting with ray tracing to address these issues [7, 13]. Additionally, works like EnvGS [14]focuses on learning spherical harmonics from multiple viewpoints.

Computational Overhead: The introduction of a neural SDF network and the computation of additional loss terms inevitably add to the training time and computational cost compared to the original SuGaR pipeline or vanilla 3DGS. Optimizing the efficiency of these new components, perhaps by using more compact SDF network architectures or adaptive application of the losses, could be explored.

Mesh Extraction Method: Our approach enhances the alignment of the Gaussians fed into SuGaRâs mesh extraction pipeline, which relies on point sampling from a density level set and Poisson reconstruction. If the co-optimized neural SDF achieves very high fidelity, directly extracting the mesh from this SDF could be an alternative. This would represent a more significant departure from the SuGaR pipeline but might offer different trade-offs in terms of

detail and smoothness.

Interaction with Refinement Stage: Our current enhancements primarily focus on the alignment stage of the SuGaR pipeline. A more detailed investigation into how an improved coarse mesh impacts the subsequent optional refinement stage (where new Gaussians are bound to the mesh) could yield further insights and potential optimizations.

Gaussian Primitives: We addressed the anisotropy caused by âneedle-likeâ Gaussians through effective rank regularization, but a low-pass filtering tendency still persists. This can cause fine details to be blurry unless an extremely large number of Gaussians are used. DyGASR [21] attempts to mitigate this issue by replacing Gaussians with a generalized exponential function for splatting. By using a higher exponent, the representation can capture sharper transitions, reducing the number of primitives needed to preserve crisp details.

In the future, these regularization techniques could be applied to other types of explicit 3D representations where maintaining temporal coherence of both primitive shapes and global surfaces is critical. In this work, we introduced improvements to the SuGaR framework to enable more reliable 3D mesh rendering from 3DGS representations. While 3DGS excels at real-time rendering, turning its Gaussian primitives into clean, accurate surface meshes remains a difficult task. To address this, we incorporated two complementary regularization strategies during SuGaRâs initial Gaussian-aligning phase.

The first is an effective rank regularization, designed to improve the shape quality of individual Gaussians. It discourages thin, elongated âneedle-likeâ primitives and instead promotes more compact, disk-shaped onesâbetter suited to capturing local surface geometry. The second is a joint optimization approach that incorporates a neural SDF. This SDF is trained alongside the Gaussians, constrained by an Eikonal loss and a consistency loss that links the SDF to the Gaussians. As a result, it acts as a smooth global prior, helping to guide the Gaussians toward a more coherent and unified surface.

Together, these two forms of regularizationâone targeting local shape and the other encouraging global alignmentâlead to Gaussians that are not only individually wellformed but also collectively produce a more accurate and consistent representation of the scene. This significantly improves the quality of the rendering, making it more visually coherent. These changes make downstream applications like editing, animation, and physical simulation much easier. Our goal is to bridge the gap between the fast rendering advantages of 3DGS and the practical demands of high-quality geometric assets in graphics workflows.

## References

[1] Hanlin Chen, Chen Li, Yunsong Wang, and Gim Hee Lee. Neusg: Neural implicit surface reconstruction with 3d gaussian splatting guidance, 2025. 2

[2] Antoine Guedon and Vincent Lepetit. Sugar: Surface- Â´ aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In CVPR, pages 5354â 5363, 2024. 1, 2, 3

[3] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, pages 1â11, 2024. 2

[4] Junha Hyung, Susung Hong, Sungwon Hwang, Jaeseong Lee, Jaegul Choo, and Jin-Hwa Kim. Effective rank analysis and regularization for enhanced 3d gaussian splatting. In NeurIPS, pages 110412â110435, 2024. 1, 2, 3, 4

[5] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4):1â14, 2023. 1, 2

[6] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1

[7] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray tracing: Fast tracing of particle scenes. ACM Trans. Graph., 43 (6):1â19, 2024. 7

[8] Michael Oechsle, Songyou Peng, and Andreas Geiger. Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 5569â5579, 2021. 2

[9] Marie-Julie Rakotosaona, Fabian Manhardt, Diego Martin Arroyo, Michael Niemeyer, Abhijit Kundu, and Federico

Tombari. Nerfmeshing: Distilling neural radiance fields into geometrically-accurate 3d meshes. In 2024 International Conference on 3D Vision (3DV), pages 1156â1165, 2024. 2

[10] Jiaxiang Tang, Hang Zhou, Xiaokang Chen, Tianshu Hu, Errui Ding, Jingdong Wang, and Gang Zeng. Delicate textured mesh recovery from nerf via adaptive surface refinement. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 17693â17703, 2023. 2

[11] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: learning neural implicit surfaces by volume rendering for multi-view reconstruction. In NeurIPS, pages 27171â27183, 2021. 1, 2, 5

[12] Yaniv Wolf, Amit Bracha, and Ron Kimmel. Gs2mesh: Surface reconstruction from gaussian splatting via novel stereo views. In Computer Vision â ECCV 2024: 18th European Conference, Milan, Italy, September 29âOctober 4, 2024, Proceedings, Part LXXXIX, pages 207â-224, 2024. 2

[13] Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, and Zan Gojcic. 3dgut: Enabling distorted cameras and secondary rays in gaussian splatting. Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 7

[14] Tao Xie, Xi Chen, Zhen Xu, Yiman Xie, Yudong Jin, Yujun Shen, Sida Peng, Hujun Bao, and Xiaowei Zhou. Envgs: Modeling view-dependent appearance with environment gaussian. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 5742â5751, 2025. 7

[15] Baixin Xu, Jiangbei Hu, Jiaze Li, and Ying He. Gsurf: 3d reconstruction via signed distance fields with direct gaussian supervision, 2024. 3

[16] Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Ronen Basri, and Yaron Lipman. Multiview neural surface reconstruction by disentangling geometry and appearance. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pages 2492â2502, 2020. 2, 5

[17] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. In Proceedings of the 35th International Conference on Neural Information Processing Systems, pages 4805â4815, 2021. 1, 2

[18] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. Gsdf: 3dgs meets sdf for improved neural rendering and reconstruction. In Advances in Neural Information Processing Systems, pages 129507â129530, 2024. 3

[19] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. Communications of the ACM, 43(6):1â 13, 2024. 2

[20] Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and Eric Xing. Fregs: 3d gaussian splatting with progressive frequency regularization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21424â21433, 2024. 2

[21] Shengchao Zhao and Yundong Li. Dygasr: Dynamic generalized gaussian splatting with surface alignment for accelerated 3d mesh reconstruction. In Pattern Recognition and Computer Vision: 7th Chinese Conference, PRCV 2024,

Urumqi, China, October 18â20, 2024, Proceedings, Part VI, pages 299â-312, 2024. 7