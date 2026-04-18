# Decoupling Motion and Geometry in 4D Gaussian Splatting

Yi Zhang \* 1 Yulei Kang \* 1 Jian-Fang Hu 1

<!-- image-->  
Figure 1. Graphic illustration of the proposed VeGaS in comparison with 4DGS. (a) Both methods share the same random initialization. (b) During training, VeGaS fits trajectories (as illustrated by the blue curves) using a decoupled motion and geometry formulation, whereas 4DGS adopts a coupled modeling scheme. (c) At inference, VeGaS employs time-varying velocity and geometry to capture complex trajectories and deformations, while 4DGS assumes constant velocity and time-invariant geometry. (d) Qualitative results show that VeGaS yields higher rendering fidelity, whereas 4DGS exhibits noticeable artifacts.

## Abstract

High-fidelity reconstruction of dynamic scenes is an important yet challenging problem. While recent 4D Gaussian Splatting (4DGS) has demonstrated the ability to model temporal dynamics, it couples Gaussian motion and geometric attributes within a single covariance formulation, which limits its expressiveness for complex motions and often leads to visual artifacts. To address this, we propose VeGaS, a novel velocity-based 4D Gaussian Splatting framework that decouples Gaussian motion and geometry. Specifically, we introduce a Galilean shearing matrix that explicitly incorporates time-varying velocity to flexibly model complex non-linear motions, while strictly isolating the effects of Gaussian motion from the geometryrelated conditional Gaussian covariance. Further-

more, a Geometric Deformation Network is introduced to refine Gaussian shapes and orientations using spatio-temporal context and velocity cues, enhancing temporal geometric modeling. Extensive experiments on public datasets demonstrate that VeGaS achieves state-of-the-art performance.

## 1. Introduction

Photorealistic modeling of real-world dynamic scenes aims to synthesize images at arbitrary viewpoints and time instants. It remains a fundamental challenge in computer vision and machine learning, with broad applications in VR/AR, immersive gaming, and cinematic production. The difficulty stems from the diversity of real-world dynamics, ranging from rigid-body motion, where geometry should remain invariant under transformations, to non-rigid deformations, where both motion and geometry evolve over time under distinct physical constraints.

Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) and subsequent variants (Chen et al., 2022; Sun et al., 2022;

Hu et al., 2022; Muller et al.Â¨ , 2022; Fridovich-Keil et al., 2022; Takikawa et al., 2021; Xu et al., 2022) pioneered implicit functional representations for reconstructing complex scenes. More recently, 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) has emerged as a compelling alternative that achieves real-time rendering by explicitly modeling the scene as a set of discrete 3D Gaussian primitives. Despite their success, both paradigms are largely tailored to static scenes and lack native mechanisms for representing temporal dynamics.

To address this, 4DGaussians (Wu et al., 2024) introduce a deformation field network to jointly learn the position offsets and geometric deformations of Gaussian points at each timestamp. 4DGS and its variants (Yang et al., 2023; Gao et al., 2025) further improved the modeling quality by proposing a 4D Gaussian Splatting representation that integrates temporal dynamics into the 3D Gaussian primitives. This representation computes both the spatial position and geometry attributes (i.e., shape and orientation) of a Gaussian at a given timestamp through the conditional distribution of the 4D Gaussian covariance, yielding a constantvelocity linear motion and a time-independent geometric model. However, the time-invariant properties of its velocity and geometric representation limit its ability to capture complex non-linear motion and expressiveness during inference. Furthermore, due to the Gaussian covariance modeling approach, the optimization of Gaussian motion and geometric parameters becomes coupled. This affects geometric modeling during complex motion fitting, making the model prone to artifacts, as illustrated in Fig. 1.

In this paper, we present VeGaS (Velocity-based Decoupling of Motion and Geometry in 4D Gaussian Splatting), a novel framework that decouples motion and geometry in 4D Gaussian Splatting to enhance dynamic scene rendering. Drawing inspiration from Galilean transformations, we propose a Galilean shearing matrix incorporating timevarying velocity for flexible modeling of complex non-linear motions, which naturally integrates into the original Gaussian covariance through congruence transformation. The transformed covariance inherently decouples the effects of Gaussian motion and geometric modeling, preventing interference with geometric modeling when fitting complex trajectories. Additionally, we present a lightweight network that independently refines the shape and orientation of the Gaussians over time. By decoupling the optimization of motion and geometry, our framework overcomes the limitations of previous approaches, providing a more expressive and reliable solution for dynamic scene reconstruction. Our contributions are summarized as follows:

â¢ We propose a novel decoupled motion and geometry framework VeGaS which effectively addresses the artifact issues arising from covariance coupling in 4D

Gaussian Splatting modeling.

â¢ We introduce a novel Gaussian motion modeling approach by incorporating time-variant velocity into the 4DGS representation and propose a deformation network to model time-varying Gaussian geometry, enhancing 4DGS expressiveness.

â¢ We conduct extensive experiments on public datasets, demonstrating that our method consistently achieves state-of-the-art results in both visual quality and quantitative performance.

## 2. Related Works

Static Scene Novel View Synthesis. Novel view synthesis is a crucial and challenging task in machine learning and computer vision. Neural Radiance Fields (NeRF) (Mildenhall et al., 2021), a pioneering work in this field, introduces an implicit scene representation that models color and density using multilayer perceptrons (MLP), delivering highquality rendering results. Subsequently, numerous research has emerged aimed at improving the training and rendering efficiency of vanilla NeRF, employing techniques such as compact data representations (Chen et al., 2022; Sun et al., 2022; Hu et al., 2022; Muller et al.Â¨ , 2022; Fridovich-Keil et al., 2022; Takikawa et al., 2021; Xu et al., 2022), or compressing neural networks (Gordon et al., 2023; Reiser et al., 2021). Other works (Barron et al., 2021; 2022; 2023; Verbin et al., 2024) focus on improving vanilla NeRF by mitigating aliasing artifacts or enhancing surface reflections, thereby boosting overall rendering performance. Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) has made significant advances in scene reconstruction due to its fast rendering. This explicit modeling approach offers greater flexibility and controllability, motivating further research (Zhu et al., 2025; Ye et al., 2024; Guo et al., 2024) to explore the application of 3DGS in areas such as 3D semantic segmentation and scene editing.

Dynamic Scene Novel View Synthesis. Generating novel views of a dynamic scene from a series of captured 2D images presents a greater challenge compared to static scenes. Many works have extended NeRF-based methods to dynamic scenes. Some methods (Pumarola et al., 2021; Park et al., 2021a;b) incorporate time as a conditioning variable and learn deformation fields that warp points from a canonical space to their corresponding positions at each time step. NeRFPlayer (Song et al., 2023) further decomposes the 4D spatiotemporal space into static, deforming, and new regions, and introduces a hybrid feature streaming scheme for efficient neural field modeling. To improve efficiency, HexPlane (Cao & Johnson, 2023) explicitly represents dynamic scenes using six learned feature planes, significantly reducing training time. Despite these advances, achieving real-time rendering with NeRF for scenes involving complex dynamics and view-dependent effects remains a significant challenge. Consequently, several recent works (Luiten et al., 2024; Huang et al., 2024; Lin et al., 2024; Lu et al., 2024; Yang et al., 2024) have explored extending 3DGS to dynamic scenes. Wu et al. (Wu et al., 2024) use a deformation field network to capture the Gaussian deformations in position, rotation, and scaling, enabling accurate Gaussian transformations over time. 4DGS (Yang et al., 2023) incorporates temporal dynamics by extending the Gaussians to a 4D representation that combines spatial and temporal dimensions. Although these Gaussian-based methods effectively capture the temporal dynamics in scene modeling, they do not comprehensively account for the variable motion dynamics of Gaussian points.

<!-- image-->  
Figure 2. Overview of our proposed velocity-based decoupling of motion and geometry for 4D Gaussian Splatting. (a) Original velocity v0 is transformed using the time-varying velocity v(t) calculated from the shearing matrix, enabling original 4D Gaussians to move along non-linear trajectories in continuous time. (b) The deformation network predicts geometric transformations of Gaussians at any time based on velocity information, time query t, and 4D Gaussian center $\mu _ { 4 D }$ . (c) Combining the velocity and geometry transformations, the rendered images are obtained through differentiable rasterization of transformed Gaussians at each frame.

## 3. Method

## 3.1. Preliminary

3D Gaussian Splatting. 3D Gaussian Splatting (Kerbl et al., 2023) explicitly renders static scenes using a set of anisotropic 3D Gaussian distributions. Each Gaussian primitive is defined by its center position $\mu \in \mathbb { R } ^ { 3 }$ and covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times \mathbf { \bar { 3 } } }$ . To ensure the semi-positive definiteness of the covariance matrix and simplify the optimization process, the covariance matrix Î£ is decomposed into a rotation matrix R and a scaling matrix $\mathbf { S } = \mathrm { d i a g } ( s _ { x } , s _ { y } , s _ { z } )$

$$
\begin{array} { r } { \pmb { \Sigma } = \mathbf { R } \mathbf { S } \mathbf { S } ^ { T } \mathbf { R } ^ { T } . } \end{array}\tag{1}
$$

In the 3D Gaussian representation, a set of spherical harmonic (SH) coefficients are also employed to represent view-dependent color, along with an opacity $\alpha \in [ 0 , 1 ]$

4D Gaussian Splatting. To render dynamic scenes, 4D Gaussian Splatting (4DGS) (Yang et al., 2023) reformulate the center position, covariance matrix, and Gaussian color in 3D Gaussian as follows: (1) Extending Gaussian center position with temporal position as $\mu = ( \mu _ { x } , \mu _ { y } , \mu _ { z } , \mu _ { t } ) \in \mathbb { R } ^ { 4 }$ ; (2) Re-formulating Covariance matrix as $\mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } = \mathbf { R } \mathbf { S } \mathbf { S } ^ { T } \mathbf { R } ^ { T } \in$ $\mathbb { R } ^ { 4 \times 4 }$ , where $\mathrm { ~ \bf ~ S ~ } = \mathrm { ~ \it ~ d i a g } ( s _ { x } , s _ { y } , s _ { z } , s _ { t } )$ is a scaling matrix and $\mathbf { R } \in \mathbb { R } ^ { 4 \times 4 }$ is a rotation matrix; (3) Representing the Gaussian color by spherical harmonic coefficients $\mathbf { h } \ \in \mathbb { R } ^ { 3 ( k _ { v } + 1 ) ^ { 2 } ( k _ { t } + 1 ) }$ , where $k _ { v }$ is the view degrees of freedom, and $k _ { t }$ indicates the time degree of freedom. At each time step, the 4D Gaussian induces a conditional 3D Gaussian distribution, which can be expressed as:

$$
\begin{array} { r l } & { \mu _ { x y z | t } = \mu _ { 1 : 3 } + ( t - \mu _ { t } ) { \Sigma _ { 1 : 3 , 4 } } { \Sigma _ { 4 , 4 } } ^ { - 1 } , } \\ & { { \Sigma _ { x y z | t } } = { \Sigma _ { 1 : 3 , 1 : 3 } } - { \Sigma _ { 1 : 3 , 4 } } { \Sigma _ { 4 , 4 } } ^ { - 1 } { \Sigma _ { 4 , 1 : 3 } } , } \end{array}\tag{2}
$$

where the 3D center $\mu _ { x y z | t }$ increases linearly over time with a constant velocity of $\pmb { \Sigma } _ { 1 : 3 , 4 } \pmb { \Sigma } _ { 4 , 4 } ^ { - 1 }$ . The induced 3D covariance $\Sigma _ { x y z | t }$ is time-invariant, indicating that the 4D Gaussianâs spatial shape and orientation do not depend on the temporal variable.

Although 4DGS supports dynamic scene reconstruction, its constant-velocity motion assumption and time-invariant geometric attributes (i.e., geometry is independent of time) together limit its capacity to represent complex motions and time-varying geometric deformations. Moreover, Î£ conflates geometric structure and motion dynamics into a single parameterization, coupling their updates during optimization rather than allowing them to be tuned independently. As a result, the model struggles to disentangle the temporal evolution of geometry and motion, which can degrade reconstruction quality and introduce artifacts.

## 3.2. Motion-Geometric Decoupled Representation

To overcome the limitations of 4DGS, a natural and intuitive solution is to introduce a time-varying velocity that enables non-linear trajectories, thereby enhancing the flexibility of 4D Gaussians in representing complex dynamics. Crucially, this motion enhancement must be strictly decoupled from geometric modeling, such that the velocity influences only the spatial position while preserving the intrinsic 3D shape and orientation of each Gaussian. We achieve this by introducing a motionâgeometry decoupled representation, whose formulation is grounded in a Galilean shearing analysis.

Theoretical Analysis. To model the motion of Gaussian points over time, we draw inspiration from the Galilean transformation in classical mechanics, using a spatiotemporal shearing operation to drag points along their trajectories.

Definition 3.1 (Galilean Shearing). In a 4D spatio-temporal continuum $( x , y , z , t )$ , any linear transformation that imparts a constant velocity $\mathbf { v } \in \mathbb { R } ^ { 3 }$ to a point while preserving the absolute temporal coordinate $\mathbf { \Psi } ( t ^ { \prime } = t )$ is equivalent to a Galilean transformation. This is represented by a shearing matrix V acting on the 4D coordinates:

$$
\begin{array}{c} { \binom { \mathbf { x } ^ { \prime } } { t ^ { \prime } } } = \mathbf { V } \left( { \begin{array} { l } { \mathbf { x } } \\ { t } \end{array} } \right) = { \binom { \mathbf { I } _ { 3 } } { \mathbf { 0 } } } \quad \mathbf { v }  \\ { \mathbf { \Phi } } \end{array} \tag{3}
$$

where $\mathbf { I } _ { 3 }$ denotes the $3 \times 3$ identity matrix.

Remark 3.2. Physically, matrix V modulates trajectories by mapping the static temporal axis to a slanted trajectory in spacetime, where the slope corresponds to v. Since $\operatorname* { d e t } ( \mathbf { V } ) = 1$ , this transformation preserves the 4D Gaussian volume and conserves the total probability mass of the Gaussian density function, ensuring the feasibility of applying the shear transformation to the Gaussian representation.

While the above definition provides a method to introduce velocity through shearing, it is essential to demonstrate that such a 4D transformation does not distort the rendered 3D geometry at any given time. The intrinsic 3D geometry of a 4D Gaussian at any time instance t is determined by its conditional distribution $P ( \mathbf { x } | t )$ . The covariance of this distribution, which characterizes the 3D ellipsoidâs shape and orientation, is given by the Schur complement of the temporal block within the joint 4D covariance matrix Î£. This allows us to prove the stability of the 3D geometry using the principle of Schur complement stability.

Theorem 3.3 (Schur Complement Invariance). Let $\Sigma \in$ $\mathbb { R } ^ { 4 \times 4 }$ be a symmetric positive semi-definite covariance matrix of a 4D Gaussian, and let $\begin{array} { r } { \Sigma ^ { \prime } = \mathbf { V } \Sigma \mathbf { V } ^ { \top } } \end{array}$ be the congruence transformation induced by the shearing matrix V. Denoting by Schu $\cdot _ { 4 , 4 } ( \cdot )$ , the Schur complement with respect to the temporal dimension, the following invariance holds:

$$
S c h u r _ { 4 , 4 } ( \Sigma ^ { \prime } ) = S c h u r _ { 4 , 4 } ( \Sigma ) .\tag{4}
$$

Theorem 3.3 (proof details in Appendix D) ensures that while the Gaussian leans in 4D spacetime to represent a velocity vector v, its 3D cross-section at any given time t retains the same shape, scale, and orientation as its rest state. Although the Galilean transformation is originally defined for constant velocity, its shearing formulation can be naturally extended to accommodate time-varying velocities, enabling the modeling of complex, non-linear trajectories.

Shearing-based Motion Modeling. Building on the above analysis, we extend the velocity shearing matrix V to incorporate a time-varying instantaneous velocity $\mathbf { v } ( t ) \ =$ $( v _ { x } , v _ { y } , v _ { z } ) ^ { \top }$ as follows:

$$
\begin{array} { r } { \mathbf { V } = \left( \begin{array} { c c } { \mathbf { I } _ { 3 } } & { \mathbf { v } ( t ) } \\ { \mathbf { 0 } } & { 1 } \end{array} \right) . } \end{array}\tag{5}
$$

Applying the shearing matrix V to the original covariance Î£, we construct the velocity-aware 4D covariance matrix $\Sigma ^ { \prime }$ via a congruence transformation, which preserves symmetry and positive semi-definiteness (proof in Appendix F):

$$
\begin{array} { r } { \pmb { \Sigma ^ { \prime } } = \pmb { \mathrm { V } } \pmb { \mathrm { R } } \pmb { \mathrm { S } } \pmb { \mathrm { S } } ^ { T } \pmb { \mathrm { R } } ^ { T } \pmb { \mathrm { V } } ^ { T } = \pmb { \mathrm { V } } \pmb { \Sigma } \pmb { \mathrm { V } } ^ { T } . } \end{array}\tag{6}
$$

By performing block-wise partitioning on the transformed covariance matrix $\Sigma ^ { \prime }$ and leveraging the properties of the multivariate Gaussian distribution, we derive the induced conditional 3D Gaussian distribution at time t (detailed calculation in Appendix B). The corresponding conditional mean $\mu _ { x y z | t } ^ { \prime }$ and covariance matrix $\Sigma _ { x y z \mid t } ^ { \prime }$ are given by:

$$
\begin{array} { r l } & { \mu _ { x y z | t } ^ { \prime } = \mu _ { 1 : 3 } + \left( \pmb { \Sigma } _ { 1 : 3 , 4 } \pmb { \Sigma } _ { 4 , 4 } ^ { - 1 } + \mathbf { v } ( t ) \right) ( t - \mu _ { t } ) , } \\ & { \pmb { \Sigma } _ { x y z | t } ^ { \prime } = \pmb { \Sigma } _ { 1 : 3 , 1 : 3 } - \pmb { \Sigma } _ { 1 : 3 , 4 } \pmb { \Sigma } _ { 4 , 4 } ^ { - 1 } \pmb { \Sigma } _ { 4 , 1 : 3 } , } \end{array}\tag{7}
$$

where the time-varying velocity $\mathbf { v } ( t )$ explicitly models the non-linear motion evolution of Gaussian point trajectories over time. By comparing Eq. (2) with Eq. (7), we observe that the conditional covariance $\Sigma _ { x y z \mid t } ^ { \prime }$ is identical to that in the original 4DGS formulation and remains independent of $\mathbf { v } ( t )$ . This indicates that introducing the shearing matrix V affects only the trajectory of the Gaussian center, while preserving its intrinsic 3D shape and orientation, which is consistent with the theoretical analysis Theorem 3.3.

Non-linear Trajectory Integration. Let $\mathbf { v } _ { 0 } = \pmb { \Sigma } _ { 1 : 3 , 4 } \pmb { \Sigma } _ { 4 , 4 } ^ { - 1 }$ denote the time-invariant velocity component. The cumulative displacement $\Delta \mu _ { 3 D }$ relative to the spatial mean $\mu _ { 1 : 2 }$ can be calculated by integrating the total instantaneous velocity over time:

$$
\Delta \mu _ { 3 D } = \int _ { \mu _ { t } } ^ { t } ( \mathbf { v } ( \tau ) + \mathbf { v } _ { 0 } ) d \tau = \int _ { \mu _ { t } } ^ { t } \mathbf { v } ( \tau ) d \tau + \mathbf { v } _ { 0 } ( t - \mu _ { t } ) .\tag{8}
$$

To handle non-linear trajectories beyond simple constant velocity, we model the time-variant velocity $\mathbf { v } ( \tau )$ as a continuous function parameterized by a set of $N _ { v }$ equidistantly sampled velocity anchors across the temporal domain $T .$ Each anchor is associated with a learnable velocity vector, serving as a nodal point for motion optimization. For any query time t, the instantaneous velocity $\mathbf { v } ( t )$ is obtained via linear interpolation between the two temporally adjacent velocity anchors.

We employ an efficient segmented numerical integration scheme based on the temporal distance between t and $\mu _ { t } \colon$ (1) Intra-anchor Interpolation: When t and $\mu _ { t }$ reside within the same anchor interval, the displacement is calculated as the area of a single trapezoid formed by v(t) and $\mathbf { v } ( \mu _ { t } )$ (2) Cross-anchor Accumulation: For intervals spanning multiple anchors, the integral is decomposed into a left boundary trapezoid (from $\mu _ { t }$ to the next anchor), a right boundary trapezoid (from the last anchor to t), and a series of interior segments representing full inter-anchor intervals.

To avoid redundant computations for the interior segments, we utilize a prefix sum of the anchor velocities. This allows the cumulative displacement of any number of full intervals to be computed in $\mathcal { O } ( 1 )$ time. Let $t _ { k }$ denote the timestamp of the k-th anchor and m represent the number of complete intervals. The cumulative displacement is computed as:

$$
\int _ { t _ { k } } ^ { t _ { k + m } } \mathbf { v } ( \tau ) d \tau = \sum _ { i = k } ^ { k + m - 1 } \frac { \mathbf { v } ( t _ { i } ) + \mathbf { v } ( t _ { i + 1 } ) } { 2 } \cdot \Delta t ,\tag{9}
$$

where $\begin{array} { r } { \Delta t { } = \frac { T } { N _ { v } - 1 } } \end{array}$ is the constant temporal stride.

## 3.3. Geometric Deformation Network

While the shearing matrix V enhances the flexibility of Gaussian motion, complex dynamic scenes often involve high-frequency geometric deformations (e.g., non-rigid muscle movement or clothing wrinkles) that are more challenging to model than motion variations. Unlike motion, which is modeled using a velocity-parameterized covariance, we introduce a lightweight deformation network to capture the time-varying geometric attributes of the Gaussians for more accurate deformation representation.

The network $\mathcal { F } _ { \theta }$ takes the spatio-temporal context and a condition t as input, predicting residuals for scaling, rotation, and position. To provide the network with explicit motion cues, we also incorporate the velocity. Specifically, the deformation network is formulated as:

$$
\Delta \mathbf { s } , \Delta \mathbf { q } , \Delta \mathbf { q } _ { r } = \mathcal { F } _ { \boldsymbol { \theta } } \big ( \gamma ( \mu _ { 3 \mathbf { D } } ) , \gamma ( \mu _ { t } ) , \gamma ( t _ { q } ) , \gamma ( \mathcal { V } ) \big ) ,\tag{10}
$$

where $\mu _ { \mathbf { 3 D } } \in \mathbb { R } ^ { 3 }$ denotes the canonical 3D Gaussian center, $\mu _ { t } \in \mathbb { R }$ is the Gaussian temporal mean, $t _ { q } \in \mathbb { R }$ is the target query time, and $\gamma \in \mathbb { R } ^ { N _ { v } \times \hat { 3 } }$ represents the motion velocity feature, formed by concatenating the velocity vectors from the $N _ { v }$ anchors. To capture high-frequency variations, each input is mapped to a higher-dimensional space via positional encoding Î³(Â·):

$$
\begin{array} { r } { \gamma ( { \bf x } ) = \left[ { \bf x } , { \sin } ( 2 ^ { i } { \bf x } ) , { \cos } ( 2 ^ { i } { \bf x } ) \right] _ { i = 0 } ^ { L _ { \bf x } - 1 } , } \end{array}\tag{11}
$$

where L $L _ { \mathbf { x } }$ denotes the feature-specific frequency bands. The encoded features are concatenated and fed into the MLPbased blocks of $\mathcal { F } _ { \theta }$ , which comprises successive linear layers followed by Layer Normalization and ReLU activations. The network outputs residuals $\Delta \mathbf { s } \in \mathbb { R } ^ { 4 }$ for scaling, and $\Delta \mathbf { q } , \Delta \mathbf { q } _ { r } \in \mathbb { R } ^ { 4 }$ as quaternion-based rotation. The deformed attributes are then updated as:

$$
{ \bf S } ^ { \prime } = { \bf S } + d i a g ( \Delta { \bf s } ) , { \bf q } ^ { \prime } = { \bf q } \otimes \Delta { \bf q } , { \bf q } _ { r } ^ { \prime } = { \bf q } _ { r } \otimes \Delta { \bf q } _ { r } ,\tag{12}
$$

where $\otimes$ denotes the quaternion multiplication. We then follow (Yang et al., 2023) and employ a dual-quaternion calculation strategy to construct rotations in 4D space. More specifically, the final 4D rotation $\mathbf { R ^ { \prime } }$ is then constructed from the refined quaternions $\mathbf { q } ^ { \prime }$ and ${ \bf q } _ { r } ^ { \prime }$ as ${ \bf R } ^ { \prime } = { \bf q } ^ { \prime } { \bf q } _ { r } ^ { \prime }$

## 3.4. Loss Function

Our system is optimized by minimizing the following loss:

$$
L = \lambda L _ { L 1 } + ( 1 - \lambda ) L _ { \mathrm { D - S S I M } } ,\tag{13}
$$

where $L _ { L 1 } = \| \mathbf { I } - \mathbf { I } _ { \mathrm { g t } } \| _ { L 1 }$ is employed to quantify the pixelwise difference between the ground truth image $\mathbf { I } _ { \mathrm { g t } }$ and the corresponding rendering I. $L _ { \mathrm { D - S S I M } } = 1 - \mathrm { S S I M } ( \mathbf { I } , \mathbf { I } _ { \mathrm { g t } } )$ evaluates the perceptual quality of the rendered image relative to the ground truth.

## 4. Experiments

## 4.1. Experimental Setup

Datasets. We evaluate our method on two representative benchmarks: (1) D-NeRF Dataset (Pumarola et al., 2021): A monocular dataset consists of videos from eight synthetic scenes. To evaluate the modelâs performance, we follow the previous approaches (Fridovich-Keil et al., 2023; Yang et al., 2023) and use the standard testing protocol, where the views used for model training and testing are different. (2) Neural 3D Video Dataset (Neu3DV) (Li et al., 2022b): A realworld dataset includes six multi-view video scenes captured by 18 to 21 cameras. Consistent with existing approaches (Cao & Johnson, 2023; Fridovich-Keil et al., 2023; Yang et al., 2023), one viewpoint is reserved for testing, while the remaining viewpoints are used for training.

Evaluation Metrics. We evaluate the rendering quality using three widely adopted image quality metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index

Table 1. Comparison with state-of-the-art methods on the multi-view real-world Neural 3D Video dataset.
<table><tr><td>Methods</td><td>Venue</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>Neural Volumes (Lombardi et al., 2019)</td><td>TOG</td><td>22.80</td><td>0.94</td><td>0.30</td></tr><tr><td>LLFF (Mildenhall et al., 2019)</td><td>TOG</td><td>23.24</td><td>0.92</td><td>0.24</td></tr><tr><td>DyNeRF (Li et al., 2022b)</td><td>CVPR</td><td>29.58</td><td>0.98</td><td>0.10</td></tr><tr><td>K-Planes-explicit (Fridovich-Keil et al., 2023)</td><td>CVPR</td><td>30.88</td><td>0.98</td><td></td></tr><tr><td>K-Planes-hybrid (Fridovich-Keil et al., 2023)</td><td>CVPR</td><td>31.63</td><td>0.98</td><td>-</td></tr><tr><td>Mix Voxels-L (Wang et al., 2023)</td><td>ICCV</td><td>30.80</td><td>0.98</td><td>0.13</td></tr><tr><td>StreamRF (Li et al., 2022a)</td><td>NeurIPS</td><td>29.58</td><td>-</td><td>-</td></tr><tr><td>NeRFPlayer (Song et al., 2023)</td><td>TVCG</td><td>30.69</td><td>0.97</td><td>0.11</td></tr><tr><td>HyperReel (Attal et al., 2023)</td><td>CVPR</td><td>31.10</td><td>0.96</td><td>0.10</td></tr><tr><td>4DGaussians (Wu et al., 2024)</td><td>CVPR</td><td>31.02</td><td>0.97</td><td>0.15</td></tr><tr><td>4DGS (Yang et al., 2023)</td><td>ICLR</td><td>32.01</td><td>0.97</td><td>0.10</td></tr><tr><td>4DGV (Dai et al., 2025)</td><td>TOG</td><td>32.55</td><td></td><td>-</td></tr><tr><td>Ours</td><td></td><td>32.68</td><td>0.98</td><td>0.09</td></tr></table>

Table 2. Quantitative comparison of methods on the monocular synthetic D-NeRF dataset.
<table><tr><td>Methods</td><td>Venue</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>T-NeRF (Pumarola et al., 2021) D-NeRF (Pumarola et al., 2021) TiNeuVox (Fang et al., 2022)</td><td>CVPR CVPR SIGGRAPH Asia</td><td colspan="3">29.51 0.95 29.67</td></tr><tr><td>K-Planes-explicit (Fridovich-Keil et al., 2023)</td><td rowspan="3">CVPR CVPR</td><td>32.67 31.05 31.61</td><td>0.95 0.97 0.97</td><td>0.07 0.04</td></tr><tr><td>K-Planes-hybrid (Fridovich-Keil et al., 2023)</td><td>0.97</td><td></td><td>- </td></tr><tr><td>V4D (Gan et al., 2023)</td><td>33.72 0.98</td><td></td><td>0.02</td></tr><tr><td>4DGaussians (Wu et al., 2024)</td><td rowspan="2">TVCG CVPR ICLR</td><td colspan="3">33.30</td></tr><tr><td>4DGS (Yang et al., 2023)</td><td>34.09</td><td>0.98 0.98</td><td>0.03 0.02</td></tr><tr><td>7DGS (Gao et al., 2025)</td><td>ICCV</td><td>34.34</td><td>0.97</td><td>0.03</td></tr><tr><td>Ours</td><td></td><td>34.67</td><td>0.99</td><td>0.02</td></tr></table>

Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018).

Implementation Details. Following (Yang et al., 2023), we use the Adam optimizer and adopt the same hyperparameter setting, including loss weight, learning rate, threshold and initialized number of Gaussians, to train our VeGaS model with a total of 30k iterations. The densification is terminated at the midpoint of the optimization schedule. A Gaussian filter with a threshold $p ( t ) < 0 . 0 5$ is employed to screen Gaussians for novel view rendering. For the time-variant velocity, we set the learning rate to $2 e ^ { - 3 }$ and the number of anchors to 6. The learning rate of the deformation network decays from $8 e ^ { - 4 }$ to 1.6eâ6, with weight regularization of $1 e ^ { - 6 }$ for training stability.

Compared Approaches. We compare our method with the current state-of-the-art approaches, including Neural Volumes (Lombardi et al., 2019), LLFF (Mildenhall et al., 2019), DyNeRF (Li et al., 2022b), K-Planes (Fridovich-Keil et al., 2023), MixVoxels-L (Wang et al., 2023), StreamRF (Li et al., 2022a), NeRFPlayer (Song et al., 2023), Hyper-Reel (Attal et al., 2023), 4DGaussians (Wu et al., 2024),

4DGS (Yang et al., 2023), 4DGV (Dai et al., 2025), D-NeRF (Pumarola et al., 2021), TiNeuVox (Fang et al., 2022), V4D (Gan et al., 2023), and 7DGS (Gao et al., 2025). Following the evaluation protocol in (Yang et al., 2023), results on the Neu3DV and D-NeRF datasets are presented. For 4DGS, we reproduce the experimental results using their official released code. Specifically, for the Neu3DV dataset, Neural Volumes, LLFF, DyNeRF, and StreamRF only provide performance results for the flames salmon scene. For NeRFPlayer and HyperReel, only SSIM is reported instead of MS-SSIM as with the other methods. Additionally, results for 4DGaussians are provided for the Spinach, Beef, and Steak scenes. On the D-NeRF dataset, for 4DGaussians, rendering is performed at 800 Ã 800 resolution, with downsampling by a factor of 2 for the other methods.

## 4.2. Results on Dynamic Scenes

Results on multi-view real scenes. Table 1 presents a quantitative comparison between our framework and state-of-theart methods on the Neural 3D Video (Neu3DV) dataset, a multi-view real-world benchmark. As shown, our method consistently outperforms prior approaches across all evaluation metrics. In particular, compared with 4DGS, our approach improves PSNR from 32.01 to 32.68, achieving an absolute gain of 0.67 dB. Meanwhile, LPIPS is reduced from 0.10 to 0.09, corresponding to a relative improvement of over 10%. The improved LPIPS scores indicate better preservation of fine-grained details and perceptual sharpness in the reconstructed scenes.

<!-- image-->  
Ground Truth

<!-- image-->

<!-- image-->  
4DGS

<!-- image-->  
4DGaussians  
Figure 3. Qualitative result on the Neural 3D Video dataset. Our method exhibits noticeably higher visual quality compared to others.

Table 3. Ablation study on the components of our design.
<table><tr><td>Methods</td><td>|PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>baseline (Yang et al., 2023)</td><td>32.01</td><td>0.97</td><td>0.10</td></tr><tr><td>+ velocity</td><td>32.07</td><td>0.97</td><td>0.10</td></tr><tr><td>+ geometric modeling</td><td>32.17</td><td>0.97</td><td>0.10</td></tr><tr><td>VeGaS (Ours Full)</td><td>32.68</td><td>0.98</td><td>0.09</td></tr></table>

Fig.3 provides qualitative comparisons of novel view synthesis on the Neu3DV dataset. The visual results show that 4DGS suffers from noticeable artifacts, including distorted backgrounds outside the windows in the flame salmon scene, as well as degraded cross-section textures of the steak in the sear steak scene. These artifacts stem from both the coupling of Gaussian motion modeling and geometric attributes through the covariance matrix and the time-invariant nature of Gaussian properties, which limit accurate fitting of complex dynamics. 4DGaussians exhibits local blurring and a loss of fine-grained details, such as the background regions outside the windows in flame salmon, the flames in flame salmon and flame steak, as well as the steak surface and finger regions in sear steak. Compared to these methods, VeGaS consistently produces results with higher visual fidelity. Fine-grained details, including irregular flame patterns, clear outdoor scenery through windows, and welldefined finger structures are better preserved, indicating more faithful geometric reconstruction and sharper perceptual quality.

Results on monocular synthetic scenes. We evaluate our method on monocular dynamic scenes, which are widely regarded as challenging due to the inherent ambiguity and limited information available from single-view observations. Tab. 2 presents a comparison between our framework and state-of-the-art methods on the D-NeRF dataset, a monocular synthetic benchmark. Our method achieves superior performance compared to all competing approaches across the evaluated metrics. These results demonstrate that our method can reliably exploit temporal coherence to compensate for the lack of multi-view constraints, achieving robust dynamic scene reconstruction under monocular settings.

Fig.4 illustrates qualitative comparisons on the D-NeRF dataset. VeGaS accurately reconstructs fine-grained structural details, including the vertical ridges of the armor in the hook scene and the detailed arm structures in the mutant scene, demonstrating noticeably superior synthesis quality compared to competing approaches.

## 4.3. Ablation Studies

Extensive ablation studies are conducted on the Neu3DV dataset to validate the effectiveness of our method.

Effectiveness of Velocity. To assess the impact of velocity

hook

<!-- image-->  
Ground Truth

<!-- image-->  
VeGaS (Ours)

<!-- image-->  
4DGS

<!-- image-->  
4DGaussians

Figure 4. Qualitative results on the D-NeRF dataset, where our method captures finer details than other methods.  
<!-- image-->  
Ground Truth

<!-- image-->  
VeGaS (Ours Full)

<!-- image-->  
VeGaS (Velocity Only)

<!-- image-->  
VeGaS (Geometric Modeling Only)

Figure 5. Visualization of ablation studies on the Neu3DV dataset, where we individually remove velocity and geometric modeling to evaluate their impact on the modelâs performance. Continuous video frames are extracted to observe the effects of removing these components on the modelâs performance.

modeling, we incorporate only the proposed time-varying velocity. As shown in Fig. 5 for the sear steak scene, introducing velocity modeling significantly improves the reconstruction of rigid object motion. It more accurately captures the motion trajectory of the meat clamp, resulting in clearer and more consistent object contours compared to the version utilizing only geometry modeling.

Effectiveness of Geometric Modeling. We further analyze the contribution of geometric modeling by only introducing the proposed Geometric Deformation Network, while disabling the velocity component. As illustrated in Fig. 5 for the flame steak scene, this geometric enhancement substantially improves the reconstruction quality of highly deformable objects such as flames, leading to more faithful shape variations.

Full Method. By jointly incorporating time-varying velocity and the Geometric Deformation Network, the complete VeGaS framework achieves a significant improvement over the original 4DGS baseline, as reported in Tab. 3. These results indicate that effective dynamic scene reconstruction benefits from both flexible motion modeling and accurate geometric deformation modeling.

## 5. Conclusion

We introduce VeGaS, a velocity-aware framework that decouples motion and geometry in 4D Gaussian Splatting. VeGaS overcomes a core limitation of prior methods, which conflate motion and geometric attributes within a unified covariance formulation. Drawing inspiration from Galilean shearing, we parameterize motion via a time-varying velocity, enabling flexible modeling of non-linear dynamics while maintaining a geometry-preserving conditional covariance. In addition, a geometric deformation network is employed to enhance temporal geometric expressiveness. Extensive experiments on real-world and synthetic benchmarks validate the effectiveness of the proposed design.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, which focuses on practical novel view synthesis. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

Attal, B., Huang, J.-B., Richardt, C., Zollhoefer, M., Kopf, J., OâToole, M., and Kim, C. Hyperreel: High-fidelity 6- dof video with ray-conditioned sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16610â16620, 2023.

Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., and Srinivasan, P. P. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5855â5864, 2021.

Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., and Hedman, P. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5470â5479, 2022.

Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., and Hedman, P. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19697â19705, 2023.

Cao, A. and Johnson, J. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 130â141, 2023.

Chen, A., Xu, Z., Geiger, A., Yu, J., and Su, H. Tensorf: Tensorial radiance fields. In European conference on computer vision, pp. 333â350. Springer, 2022.

Dai, P., Zhang, P., Dong, Z., Xu, K., Peng, Y., Ding, D., Shen, Y., Yang, Y., Liu, X., Lau, R. W., et al. 4d gaussian videos with motion layering. ACM Transactions on Graphics (TOG), 44(4):1â14, 2025.

Fang, J., Yi, T., Wang, X., Xie, L., Zhang, X., Liu, W., NieÃner, M., and Tian, Q. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia 2022 Conference Papers, pp. 1â9, 2022.

Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., and Kanazawa, A. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5501â5510, 2022.

Fridovich-Keil, S., Meanti, G., Warburg, F. R., Recht, B., and Kanazawa, A. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12479â12488, 2023.

Gan, W., Xu, H., Huang, Y., Chen, S., and Yokoya, N. V4d: Voxel for 4d novel view synthesis. IEEE Transactions on Visualization and Computer Graphics, 30(2):1579â1591, 2023.

Gao, Z., Planche, B., Zheng, M., Choudhuri, A., Chen, T., and Wu, Z. 7dgs: Unified spatial-temporal-angular gaussian splatting. arXiv preprint arXiv:2503.07946, 2025.

Gordon, C., Chng, S.-F., MacDonald, L., and Lucey, S. On quantizing implicit neural representations. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 341â350, 2023.

Guo, J., Ma, X., Fan, Y., Liu, H., and Li, Q. Semantic gaussians: Open-vocabulary scene understanding with 3d gaussian splatting. arXiv preprint arXiv:2403.15624, 2024.

Hu, T., Liu, S., Chen, Y., Shen, T., and Jia, J. Efficientnerf efficient neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12902â12911, 2022.

Huang, Y.-H., Sun, Y.-T., Yang, Z., Lyu, X., Cao, Y.-P., and Qi, X. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4220â4230, 2024.

Kerbl, B., Kopanas, G., Leimkuhler, T., and Drettakis, G. 3d Â¨ gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

Li, L., Shen, Z., Wang, Z., Shen, L., and Tan, P. Streaming radiance fields for 3d video synthesis. Advances in Neural Information Processing Systems, 35:13485â13498, 2022a.

Li, T., Slavcheva, M., Zollhoefer, M., Green, S., Lassner, C., Kim, C., Schmidt, T., Lovegrove, S., Goesele, M., Newcombe, R., et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5521â5531, 2022b.

Lin, Y., Dai, Z., Zhu, S., and Yao, Y. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21136â21145, 2024.

Lombardi, S., Simon, T., Saragih, J., Schwartz, G., Lehrmann, A., and Sheikh, Y. Neural volumes: Learning dynamic renderable volumes from images. arXiv preprint arXiv:1906.07751, 2019.

Lu, Z., Guo, X., Hui, L., Chen, T., Yang, M., Tang, X., Zhu, F., and Dai, Y. 3d geometry-aware deformable gaussian splatting for dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8900â8910, 2024.

Luiten, J., Kopanas, G., Leibe, B., and Ramanan, D. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV), pp. 800â809. IEEE, 2024.

Mildenhall, B., Srinivasan, P. P., Ortiz-Cayon, R., Kalantari, N. K., Ramamoorthi, R., Ng, R., and Kar, A. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1â14, 2019.

Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

Muller, T., Evans, A., Schied, C., and Keller, A. InstantÂ¨ neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1â15, 2022.

Park, K., Sinha, U., Barron, J. T., Bouaziz, S., Goldman, D. B., Seitz, S. M., and Martin-Brualla, R. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5865â5874, 2021a.

Park, K., Sinha, U., Hedman, P., Barron, J. T., Bouaziz, S., Goldman, D. B., Martin-Brualla, R., and Seitz, S. M. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021b.

Pumarola, A., Corona, E., Pons-Moll, G., and Moreno-Noguer, F. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10318â 10327, 2021.

Reiser, C., Peng, S., Liao, Y., and Geiger, A. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 14335â14345, 2021.

Song, L., Chen, A., Li, Z., Chen, Z., Chen, L., Yuan, J., Xu, Y., and Geiger, A. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance

fields. IEEE Transactions on Visualization and Computer Graphics, 29(5):2732â2742, 2023.

Sun, C., Sun, M., and Chen, H.-T. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5459â5469, 2022.

Takikawa, T., Litalien, J., Yin, K., Kreis, K., Loop, C., Nowrouzezahrai, D., Jacobson, A., McGuire, M., and Fidler, S. Neural geometric level of detail: Real-time rendering with implicit 3d shapes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11358â11367, 2021.

Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J. T., and Srinivasan, P. P. Ref-nerf: Structured viewdependent appearance for neural radiance fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Wang, F., Tan, S., Li, X., Tian, Z., Song, Y., and Liu, H. Mixed neural voxels for fast multi-view video synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19706â19716, 2023.

Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., and Wang, X. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310â20320, 2024.

Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., and Neumann, U. Point-nerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5438â5448, 2022.

Yang, Z., Yang, H., Pan, Z., and Zhang, L. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642, 2023.

Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., and Jin, X. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20331â20341, 2024.

Ye, M., Danelljan, M., Yu, F., and Ke, L. Gaussian grouping: Segment and edit anything in 3d scenes. In European conference on computer vision, pp. 162â179. Springer, 2024.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference

on computer vision and pattern recognition, pp. 586â595, 2018.

Zhu, R., Qiu, S., Liu, Z., Hui, K.-H., Wu, Q., Heng, P.-A., and Fu, C.-W. Rethinking end-to-end 2d to 3d scene segmentation in gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 3656â3665, 2025.

## A. Additional Experiment Results

<!-- image-->

<!-- image-->  
VeGaS (Ours)

<!-- image-->  
4DGS

<!-- image-->  
4DGaussians

Figure 6. Additional qualitative result on the Neural 3D Video dataset. Our method exhibits noticeably higher visual quality compared to others.

## B. Detailed derivation of the conditional distribution of a 4D Gaussian

## B.1. Transformation of the 4D Covariance Matrix

Given the Galilean shearing matrix V , the modified covariance matrix is given by:

$$
\Sigma ^ { \prime } = V R S S ^ { T } R ^ { T } V ^ { T } = V \Sigma V ^ { T } ,\tag{14}
$$

where R denotes the rotation matrix and S denotes the scaling matrix. The shearing velocity matrix V explicitly represents the positional transformation of a Gaussian point in the 4D spatio-temporal continuum.

By expanding the block matrix multiplication, we can derive the structure of the covariance matrix Î£â²:

$$
\Sigma ^ { \prime } = { \binom { \mathbf { I } _ { 3 } } { \mathbf { 0 } } } \quad v ( t ) \biggr ) \left( \begin{array} { l l } { \Sigma _ { 1 : 3 , 1 : 3 } } & { \Sigma _ { 1 : 3 , 4 } } \\ { \Sigma _ { 4 , 1 : 3 } } & { \Sigma _ { 4 , 4 } } \end{array} \right) \left( \begin{array} { l l } { \mathbf { I } _ { 3 } } & { \mathbf { 0 } } \\ { v ( t ) ^ { \top } } & { 1 \rule { 0 ex } { 5 ex } } \end{array} \right) .\tag{15}
$$

Carrying out the multiplication for the block matrix $\Sigma ~ = ~ \left( \begin{array} { c c } { { \Sigma _ { 1 : 3 , 1 : 3 } } } & { { \Sigma _ { 1 : 3 , 4 } } } \\ { { \Sigma _ { 4 , 1 : 3 } } } & { { \Sigma _ { 4 , 4 } } } \end{array} \right)$ , the individual blocks of the updated covariance matrix Î£â² are obtained as follows:

$$
\begin{array} { r l } & { \Sigma _ { 1 : 3 , 1 : 3 } ^ { \prime } = \Sigma _ { 1 : 3 , 1 : 3 } + v ( t ) \Sigma _ { 4 , 1 : 3 } + \Sigma _ { 1 : 3 , 4 } v ( t ) ^ { \top } } \\ & { \qquad + v ( t ) \Sigma _ { 4 , 4 } v ( t ) ^ { \top } , } \\ & { \Sigma _ { 1 : 3 , 4 } ^ { \prime } = \Sigma _ { 1 : 3 , 4 } + v ( t ) \Sigma _ { 4 , 4 } , } \\ & { \Sigma _ { 4 , 1 : 3 } ^ { \prime } = \Sigma _ { 4 , 1 : 3 } + \Sigma _ { 4 , 4 } v ( t ) ^ { \top } , } \\ & { \quad \Sigma _ { 4 , 4 } ^ { \prime } = \Sigma _ { 4 , 4 } . } \end{array}\tag{16}
$$

## B.2. General Multivariate Conditional Gaussian Distribution

Before applying the conditional logic to the 4D Gaussian, we first derive the general form of the conditional distribution. Consider a multivariate Gaussian random variable $\mathbf { x } \sim { \mathcal { N } } ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ partitioned into two subsets $\mathbf { x } _ { a }$ and $\mathbf { x } _ { b } .$ :

$$
\mathbf { x } = { \binom { \mathbf { x } _ { a } } { \mathbf { x } _ { b } } } , \quad { \boldsymbol { \mu } } = { \binom { { \boldsymbol { \mu } } _ { a } } { \boldsymbol { \mu } _ { b } } } , \quad { \boldsymbol { \Sigma } } = { \binom { { \boldsymbol { \Sigma } } _ { a a } } { \boldsymbol { \Sigma } _ { b a } } } . \quad { \boldsymbol { \Sigma } } _ { b b } \quad .\tag{17}
$$

To find the conditional distribution $p ( \mathbf { x } _ { a } | \mathbf { x } _ { b } )$ , we construct a linear transformation to decorrelate $\mathbf { x } _ { a }$ from $\mathbf { x } _ { b }$ . Let ${ \textbf { z } } =$ ${ \bf x } _ { a } - { \bf A } { \bf x } _ { b }$ . We seek a matrix A such that z and $\mathbf { x } _ { b }$ are uncorrelated (and thus independent for Gaussians):

$$
\mathrm { C o v } ( \mathbf { z } , \mathbf { x } _ { b } ) = \mathrm { C o v } ( \mathbf { x } _ { a } , \mathbf { x } _ { b } ) - \mathbf { A } \mathrm { C o v } ( \mathbf { x } _ { b } , \mathbf { x } _ { b } ) = \boldsymbol { \Sigma } _ { a b } - \mathbf { A } \boldsymbol { \Sigma } _ { b b } = \mathbf { 0 } .\tag{18}
$$

Solving for A yields $\begin{array} { r } { { \bf A } = \pmb { \Sigma } _ { a b } \pmb { \Sigma } _ { b b } ^ { - 1 } } \end{array}$ . Since z is independent of $\mathbf { x } _ { b } ,$ , the conditional expectation is derived as:

$$
\begin{array} { r l } & { \mathbb { E } [ { \mathbf { x } } _ { a } \vert { \mathbf { x } } _ { b } ] = \mathbb { E } [ { \mathbf { A } } { \mathbf { x } } _ { b } + { \mathbf { z } } \vert { \mathbf { x } } _ { b } ] = { \mathbf { A } } { \mathbf { x } } _ { b } + \mathbb { E } [ { \mathbf { z } } ] } \\ & { \quad \quad \quad = { \boldsymbol { \Sigma } } _ { a b } { \boldsymbol { \Sigma } } _ { b b } ^ { - 1 } { \mathbf { x } } _ { b } + ( { \pmb { \mu } } _ { a } - { \boldsymbol { \Sigma } } _ { a b } { \boldsymbol { \Sigma } } _ { b b } ^ { - 1 } { \pmb { \mu } } _ { b } ) } \\ & { \quad \quad \quad = { \pmb { \mu } } _ { a } + { \boldsymbol { \Sigma } } _ { a b } { \boldsymbol { \Sigma } } _ { b b } ^ { - 1 } ( { \mathbf { x } } _ { b } - { \boldsymbol { \mu } } _ { b } ) . } \end{array}\tag{19}
$$

The conditional covariance is simply the variance of the residual z:

$$
\mathrm { C o v } ( \mathbf { x } _ { a } | \mathbf { x } _ { b } ) = \mathrm { C o v } ( \mathbf { z } ) = \pmb { \Sigma } _ { a a } - \pmb { \Sigma } _ { a b } \pmb { \Sigma } _ { b b } ^ { - 1 } \pmb { \Sigma } _ { b a } .\tag{20}
$$

## B.3. Application to Our 4D Gaussian Representation

Applying the general derivation above to our specific 4D case, we map the spatial coordinates to partition a (indices $1 : 3 )$ and the temporal coordinate to partition b (index 4). We substitute the transformed covariance blocks $\Sigma ^ { \prime }$ into the conditional formulas.

The modified conditional mean $\mu _ { x y z | t } ^ { \prime }$ is given by:

$$
\begin{array} { r l } & { \mu _ { x y z | t } ^ { \prime } = \mu _ { 1 : 3 } + \Sigma _ { 1 : 3 , 4 } ^ { \prime } ( \Sigma _ { 4 , 4 } ^ { \prime } ) ^ { - 1 } ( t - \mu _ { t } ) } \\ & { \qquad = \mu _ { 1 : 3 } + ( \Sigma _ { 1 : 3 , 4 } + v ( t ) \Sigma _ { 4 , 4 } ) \Sigma _ { 4 , 4 } ^ { - 1 } ( t - \mu _ { t } ) } \\ & { \qquad = \mu _ { 1 : 3 } + \left( \Sigma _ { 1 : 3 , 4 } \Sigma _ { 4 , 4 } ^ { - 1 } + v ( t ) \right) ( t - \mu _ { t } ) . } \end{array}\tag{21}
$$

Similarly, the modified conditional covariance $\Sigma _ { x y z \mid t } ^ { \prime }$ is calculated as:

$$
\Sigma _ { x y z \mid t } ^ { \prime } = \Sigma _ { 1 : 3 , 1 : 3 } ^ { \prime } - \Sigma _ { 1 : 3 , 4 } ^ { \prime } ( \Sigma _ { 4 , 4 } ^ { \prime } ) ^ { - 1 } \Sigma _ { 4 , 1 : 3 } ^ { \prime } .\tag{22}
$$

To simplify this, we first expand the subtraction term $K = \Sigma _ { 1 : 3 , 4 } ^ { \prime } ( \Sigma _ { 4 , 4 } ^ { \prime } ) ^ { - 1 } \Sigma _ { 4 , 1 : 3 } ^ { \prime } \mathrm { . }$

$$
\begin{array} { r l } & { K = ( \Sigma _ { 1 : 3 , 4 } + v ( t ) \Sigma _ { 4 , 4 } ) \Sigma _ { 4 , 4 } ^ { - 1 } \left( \Sigma _ { 4 , 1 : 3 } + \Sigma _ { 4 , 4 } v ( t ) ^ { \top } \right) } \\ & { \quad = \left( \Sigma _ { 1 : 3 , 4 } \Sigma _ { 4 , 4 } ^ { - 1 } + v ( t ) \right) \left( \Sigma _ { 4 , 1 : 3 } + \Sigma _ { 4 , 4 } v ( t ) ^ { \top } \right) } \\ & { \quad = \Sigma _ { 1 : 3 , 4 } \Sigma _ { 4 , 4 } ^ { - 1 } \Sigma _ { 4 , 1 : 3 } + \Sigma _ { 1 : 3 , 4 } v ( t ) ^ { \top } + v ( t ) \Sigma _ { 4 , 1 : 3 } + v ( t ) \Sigma _ { 4 , 4 } v ( t ) ^ { \top } . } \end{array}\tag{23}
$$

Substituting K and the expression for $\Sigma _ { 1 : 3 , 1 : 3 } ^ { \prime }$ from Eq. (16) back into the conditional covariance formula, we observe that all terms involving v(t) cancel out:

$$
\begin{array} { l } { { \Sigma _ { x y z | t } ^ { \prime } = \left( \Sigma _ { 1 : 3 , 1 : 3 } + v ( t ) \Sigma _ { 4 , 1 : 3 } + \Sigma _ { 1 : 3 , 4 } v ( t ) ^ { \top } + v ( t ) \Sigma _ { 4 , 4 } v ( t ) ^ { \top } \right) - K } } \\ { { \qquad = \Sigma _ { 1 : 3 , 1 : 3 } - \Sigma _ { 1 : 3 , 4 } \Sigma _ { 4 , 4 } ^ { - 1 } \Sigma _ { 4 , 1 : 3 } . } } \end{array}\tag{24}
$$

This proves that the shape of the conditional covariance remains invariant under the Galilean shear transformation, identical to the original spatial covariance.

## C. Schur Complement

Definition C.1 (Schur Complement). Schur Complement]Consider a partitioned symmetric matrix $\mathbf { M } \in \mathbb { R } ^ { ( n + m ) \times ( n + m ) }$ decomposed into block matrices as follows:

$$
\mathbf { M } = \left( \begin{array} { c c } { \mathbf { A } } & { \mathbf { B } } \\ { \mathbf { B } ^ { \top } } & { \mathbf { D } } \end{array} \right) ,\tag{25}
$$

where $\mathbf { A } \in \mathbb { R } ^ { n \times n } , \mathbf { B } \in \mathbb { R } ^ { n \times m }$ , and $\mathbf { D } \in \mathbb { R } ^ { m \times m }$ is invertible. The Schur complement of the block D in M, denoted as $\mathbf { M } / \mathbf { D }$ , is defined as:

$$
\mathbf { M } / \mathbf { D } \triangleq \mathbf { A } - \mathbf { B } \mathbf { D } ^ { - 1 } \mathbf { B } ^ { \intercal } .\tag{26}
$$

Remark C.2 (Probabilistic Interpretation). In the context of multivariate Gaussian distributions, if M represents the joint covariance matrix of two random vectors $\mathbf { x } \in \mathbb { R } ^ { n }$ and $\mathbf { y } \in \mathbb { R } ^ { m }$ , the Schur complement $\mathbf { M } / \mathbf { D }$ corresponds precisely to the conditional covariance of x given $\mathbf { y } \ ( \mathrm { i . e . , C o v ( x | \mathbf { y } ) } )$ .

## D. Proof of Schur Complement Invariance

Theorem D.1 (Schur Complement Invariance). Let $\pmb { \Sigma } \in \mathbb { R } ^ { 4 \times 4 }$ be a symmetric positive semi-definite covariance matrix of a 4D Gaussian, and let $\begin{array} { r } { \pmb { \Sigma ^ { \prime } } = \mathbf { V } \pmb { \Sigma } \mathbf { V } ^ { \top } } \end{array}$ be the congruence transformation induced by the shearing matrix V. Denoting by $S c h u r _ { 4 , 4 } ( \cdot )$ the Schur complement with respect to the temporal dimension, the following invariance holds:

$$
S c h u r _ { 4 , 4 } ( \Sigma ^ { \prime } ) = S c h u r _ { 4 , 4 } ( \Sigma ) .\tag{27}
$$

Proof. We partition the covariance matrix Î£ and the shearing matrix V into block forms corresponding to the spatial (1 : 3) and temporal (4) dimensions:

$$
\Sigma = \left( \begin{array} { l l } { \Sigma _ { x x } } & { \Sigma _ { x t } } \\ { \Sigma _ { t x } } & { \Sigma _ { t t } } \end{array} \right) , \quad \mathbf { V } = \left( \begin{array} { l l } { \mathbf { I } _ { 3 } } & { \mathbf { v } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} \right) ,\tag{28}
$$

where $\begin{array} { r } { \sum _ { x x } \in \mathbb { R } ^ { 3 \times 3 } , \sum _ { x t } \in \mathbb { R } ^ { 3 \times 1 } , \sum _ { t t } \in \mathbb { R } ^ { 1 \times 1 } } \end{array}$ , and $\mathbf { v } \in \mathbb { R } ^ { 3 \times 1 }$ is the velocity vector. Note that $\Sigma _ { t x } = \Sigma _ { x t } ^ { \top }$

First, we compute the block structure of the transformed covariance $\begin{array} { r } { \pmb { \Sigma ^ { \prime } } = \mathbf { V } \pmb { \Sigma } \mathbf { V } ^ { \top } } \end{array}$ . Expanding the matrix multiplication yields:

$$
\begin{array} { r l } & { \Sigma ^ { \prime } = \left( \begin{array} { c c } { { \bf { I } } _ { 3 } } & { { \bf { v } } } \\ { { \bf { 0 } } } & { 1 } \end{array} \right) \left( \begin{array} { c c } { { \bf { \Sigma } } _ { x x } } & { { \bf { \Sigma } } _ { x t } } \\ { { \bf { \Sigma } } _ { t x } } & { { \bf { \Sigma } } _ { t t } } \end{array} \right) \left( \begin{array} { c c } { { \bf { I } } _ { 3 } } & { { \bf { 0 } } } \\ { { \bf { v } } ^ { \top } } & { 1 } \end{array} \right) } \\ & { \quad \quad = \left( \begin{array} { c c } { { \bf { \Sigma } } _ { x x } + { \bf { v } } { \bf { \Sigma } } _ { t x } } & { { \bf { \Sigma } } _ { x t } + { \bf { v } } { \Sigma } _ { t t } } \\ { { \bf { \Sigma } } _ { t x } } & { { \bf { \Sigma } } _ { t t } } \end{array} \right) \left( \begin{array} { c c } { { \bf { I } } _ { 3 } } & { { \bf { 0 } } } \\ { { \bf { v } } ^ { \top } } & { 1 } \end{array} \right) } \\ & { \quad \quad = \left( \begin{array} { c c } { { \bf { \Sigma } } _ { x x } + { \bf { v } } { \Sigma } _ { t x } + \left( { \bf { \Sigma } } _ { x t } + { \bf { v } } { \Sigma } _ { t t } \right) { \bf { v } } ^ { \top } } & { { \Sigma } _ { x t } + { \bf { v } } { \Sigma } _ { t t } } \\ { { \bf { \Sigma } } _ { t x } + { \bf { \Sigma } } _ { t t } { \bf { v } } ^ { \top } } & { { \Sigma } _ { t t } } \end{array} \right) . } \end{array}\tag{29}
$$

From the expansion above, we identify the individual blocks of $\Sigma ^ { \prime }$ :

$$
\begin{array} { r } { \sum _ { x x } ^ { \prime } = \pmb { \Sigma } _ { x x } + \mathbf { v } \pmb { \Sigma } _ { t x } + \pmb { \Sigma } _ { x t } \mathbf { v } ^ { \top } + \mathbf { v } \pmb { \Sigma } _ { t t } \mathbf { v } ^ { \top } , } \end{array}\tag{30a}
$$

$$
\begin{array} { r } { \Sigma _ { x t } ^ { \prime } = \Sigma _ { x t } + \mathbf { v } \Sigma _ { t t } , } \end{array}\tag{30b}
$$

$$
\Sigma _ { t t } ^ { \prime } = \Sigma _ { t t } .\tag{30c}
$$

The Schur complement of the transformed matrix, denoted as $S ^ { \prime }$ , is defined by:

$$
\begin{array} { r } { S ^ { \prime } = \mathrm { S c h u r } _ { 4 , 4 } ( \Sigma ^ { \prime } ) = \Sigma _ { x x } ^ { \prime } - \Sigma _ { x t } ^ { \prime } ( \Sigma _ { t t } ^ { \prime } ) ^ { - 1 } ( \Sigma _ { x t } ^ { \prime } ) ^ { \top } . } \end{array}\tag{31}
$$

Substituting the expressions from Eqs. (30a)â(30c) into the definition, we focus on expanding the subtraction term $K =$ $\Sigma _ { x t } ^ { \prime } ( \Sigma _ { t t } ^ { \prime } ) ^ { - \bar { 1 } } ( \Sigma _ { x t } ^ { \prime } ) ^ { \top }$ :

$$
\begin{array} { r l } & { K = ( \Sigma _ { x t } + \mathbf { v } \Sigma _ { t t } ) \Sigma _ { t t } ^ { - 1 } \bigl ( \Sigma _ { t x } + \Sigma _ { t t } \mathbf { v } ^ { \top } \bigr ) } \\ & { ~ = ( \Sigma _ { x t } \Sigma _ { t t } ^ { - 1 } + \mathbf { v } \bigr ) \bigl ( \Sigma _ { t x } + \Sigma _ { t t } \mathbf { v } ^ { \top } \bigr ) } \\ & { ~ = \Sigma _ { x t } \Sigma _ { t t } ^ { - 1 } \Sigma _ { t x } + \Sigma _ { x t } \mathbf { v } ^ { \top } + \mathbf { v } \Sigma _ { t x } + \mathbf { v } \Sigma _ { t t } \mathbf { v } ^ { \top } . } \end{array}\tag{32}
$$

Now, we subtract K from $\Sigma _ { x x } ^ { \prime }$

$$
\begin{array} { r l } & { S ^ { \prime } = \Sigma _ { x x } ^ { \prime } - K } \\ & { \quad = \left( \Sigma _ { x x } + \mathbf { v } \Sigma _ { t x } + \Sigma _ { x t } \mathbf { v } ^ { \top } + \mathbf { v } \Sigma _ { t t } \mathbf { v } ^ { \top } \right) } \\ & { \quad \quad - \left( \Sigma _ { x t } \Sigma _ { t t } ^ { - 1 } \Sigma _ { t x } + \Sigma _ { x t } \mathbf { v } ^ { \top } + \mathbf { v } \Sigma _ { t x } + \mathbf { v } \Sigma _ { t t } \mathbf { v } ^ { \top } \right) . } \end{array}\tag{33}
$$

Observing the terms, we see that $\mathbf { v } \pmb { \Sigma } _ { t x } , \pmb { \Sigma } _ { x t } \mathbf { v } ^ { \top }$ , and $\mathbf { v } \Sigma _ { t t } \mathbf { v } ^ { \top }$ appear in both parts and cancel out perfectly. The remaining terms are:

$$
S ^ { \prime } = \Sigma _ { x x } - \Sigma _ { x t } \Sigma _ { t t } ^ { - 1 } \Sigma _ { t x } = \mathrm { S c h u r } _ { 4 , 4 } ( \Sigma ) .\tag{34}
$$

Thus, the Schur complement is invariant under the Galilean shear transformation.

## E. Generalization of Constant-Velocity Galilean Shearing to Time-Varying Velocity

While the standard Galilean transformation is defined for inertial frames with constant velocity, real-world dynamics often involve non-linear trajectories with time-varying velocities. Here, we provide the mathematical justification for extending the shearing mechanism to model such motions using local linearization.

First-Order Taylor Approximation. Consider a 4D Gaussian centered at temporal coordinate $\mu _ { t } ,$ , tracking a particle moving along a non-linear trajectory $\gamma ( t ) \in \mathbb { R } ^ { 3 }$ . We aim to approximate this trajectory within the local temporal neighborhood of the Gaussian, defined effectively by its temporal variance $\Sigma _ { t t }$

Expanding the trajectory $\gamma ( t )$ around the center time $\mu _ { t }$ using a Taylor series, we obtain:

$$
\gamma ( t ) = \gamma ( \mu _ { t } ) + { \frac { d \gamma } { d t } } { \bigg | } _ { t = \mu _ { t } } ( t - \mu _ { t } ) + O \left( ( t - \mu _ { t } ) ^ { 2 } \right) .\tag{35}
$$

Let $\begin{array} { r } { \mathbf { v } = \frac { d \gamma } { d t } | _ { t = \mu _ { t } } } \end{array}$ denote the instantaneous velocity at the center of the Gaussian. Neglecting higher-order terms ${ \mathcal O } ( ( t - \mu _ { t } ) ^ { 2 } )$ (which is a valid assumption for Gaussians with small temporal extent), the trajectory is approximated as a linear function:

$$
\hat { \gamma } ( t ) \approx \gamma ( \mu _ { t } ) + { \bf v } \cdot ( t - \mu _ { t } ) .\tag{36}
$$

Equivalence to Galilean Shearing. Recall the operation of the Galilean shearing matrix V defined in the main text on a spatial point x relative to the temporal center:

$$
{ \binom { \mathbf { x } ^ { \prime } } { t ^ { \prime } } } = { \binom { \mathbf { I } _ { 3 } } { \mathbf { 0 } } } \ \mathbf { v }  \\  \left( t ^ { \prime } \right) = { \binom { \mathbf { I } _ { 3 } } { \mathbf { 0 } } } \ \mathbf { \binom { \mathbf { x } } { \mathit { 1 } } } \left( t - \mu _ { t } \right) = { \binom { \mathbf { x } + \mathbf { v } ( t - \mu _ { t } ) } { t - \mu _ { t } } } .\tag{37}
$$

By setting the initial position $\mathbf { x } = \gamma ( \mu _ { t } )$ , the spatial component becomes $\mathbf { x } ^ { \prime } = \gamma ( \mu _ { t } ) + \mathbf { v } ( t - \mu _ { t } )$ , which is identical to the first-order approximation in Eq. (36).

Thus, applying the shearing matrix V parameterized by the instantaneous velocity $\mathbf { v } ( t )$ is mathematically equivalent to locally linearizing the non-linear trajectory along the tangent direction at $t = \mu _ { t }$

## F. Preservation of Symmetric Positive Semi-Definiteness

A valid covariance matrix must be symmetric and positive semi-definite (SPSD) to represent a meaningful probability distribution. We prove that the congruence transformation induced by the Galilean shearing matrix preserves these essential properties.

Lemma F.1 (Invariance of SPSD under Congruence Transformation.). Let $\pmb { \Sigma } \in \mathbb { R } ^ { 4 \times 4 }$ be a symmetric positive semi-definite matrix $( \pmb { \Sigma } \succeq 0 )$ , and let $\mathbf { V } \in \mathbb { R } ^ { 4 \times 4 }$ be the Galilean shearing matrix. The transformed covariance matrix $\begin{array} { r } { \pmb { \Sigma ^ { \prime } } = \mathbf { V } \pmb { \Sigma } \mathbf { V } ^ { \top } } \end{array}$ remains symmetric and positive semi-definite.

Proof. We verify these properties separately:

Symmetry. By definition, Î£ is symmetric, so $\pmb { \Sigma } = \pmb { \Sigma } ^ { \top }$ . Taking the transpose of the transformed matrix $\Sigma ^ { \prime }$ :

$$
\begin{array} { r l } & { \left( \boldsymbol { \Sigma } ^ { \prime } \right) ^ { \top } = ( \mathbf { V } \boldsymbol { \Sigma } \mathbf { V } ^ { \top } ) ^ { \top } } \\ & { ~ = ( \mathbf { V } ^ { \top } ) ^ { \top } \boldsymbol { \Sigma } ^ { \top } \mathbf { V } ^ { \top } } \\ & { ~ = \mathbf { V } \boldsymbol { \Sigma } \mathbf { V } ^ { \top } } \\ & { ~ = \boldsymbol { \Sigma } ^ { \prime } . } \end{array}\tag{38}
$$

Thus, $\Sigma ^ { \prime }$ is symmetric.

Positive Semi-Definiteness. By definition, $\pmb { \Sigma } \succeq 0$ implies that for any non-zero vector $\mathbf { x } \in \mathbb { R } ^ { 4 }$ , the quadratic form satisfies $\mathbf { x } ^ { \top } \pmb { \Sigma } \mathbf { x } \geq 0$ . Consider the quadratic form of the transformed matrix $\Sigma ^ { \prime }$ with respect to an arbitrary vector $\mathbf { y } \in \mathbb { R } ^ { 4 }$ :

$$
Q ( \mathbf { y } ) = \mathbf { y } ^ { \top } \pmb { \Sigma } ^ { \prime } \mathbf { y } = \mathbf { y } ^ { \top } ( \mathbf { V } \pmb { \Sigma } \mathbf { V } ^ { \top } ) \mathbf { y } .\tag{39}
$$

Using the associative property of matrix multiplication, we can regroup the terms:

$$
Q ( \mathbf { y } ) = ( \mathbf { y } ^ { \top } \mathbf { V } ) \pmb { \Sigma } ( \mathbf { V } ^ { \top } \mathbf { y } ) .\tag{40}
$$

Let $\mathbf { z } = \mathbf { V } ^ { \top } \mathbf { y }$ . The equation becomes:

$$
Q ( \mathbf { y } ) = \mathbf { z } ^ { \top } \Sigma \mathbf { z } .\tag{41}
$$

Since Î£ is positive semi-definite, $\mathbf { z } ^ { \top } \pmb { \Sigma } \mathbf { z } \geq 0$ holds for any vector z. Consequently, $\mathbf { y } ^ { \top } \pmb { \Sigma } ^ { \prime } \mathbf { y } \geq 0$ for all $\mathbf { y }$ .

Therefore, Î£â² is positive semi-definite.

Corollary F.2 (Preservation of Positive Definiteness). Furthermore, if Î£ is strictly positive definite $( \pmb { \Sigma } \succ 0 )$ and V is non-singular, then $\Sigma ^ { \prime }$ is also strictly positive definite. For the Galilean shearing matrix, the determinant is:

$$
\operatorname* { d e t } ( \mathbf { V } ) = 1 \neq 0 .\tag{42}
$$

Since V is invertible (full rank), the transformation strictly preserves the positive definiteness of the Gaussian covariance.