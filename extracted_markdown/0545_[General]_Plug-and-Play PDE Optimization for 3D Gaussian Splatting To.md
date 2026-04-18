# Plug-and-Play PDE Optimization for 3D Gaussian Splatting: Toward High-Quality Rendering and Reconstruction

ANONYMOUS AUTHOR(S) SUBMISSION ID: 2152

3DGS  
3DGS+PDEO  
<!-- image-->  
Fig. 1. We present PDEO, a novel, plug-and-play optimization framework designed to enable stable optimization of 3D Gaussians and enhance existing 3DGS-based approaches for tasks such as novel view synthesis and surface reconstruction. Our method achieves high-quality results in both rendering and reconstruction. More results are provided in the accompanying video.

3D Gaussian Splatting (3DGS) has revolutionized radiance field reconstruction by achieving high-quality novel view synthesis with fast rendering speed, introducing 3D Gaussian primitives to represent the scene. However, 3DGS encounters blurring and floaters when applied to complex scenes, caused by the reconstruction of redundant and ambiguous geometric structures. We attribute this issue to the unstable optimization of the Gaussians. To address this limitation, we present a plug-and-play PDE-based optimization method that overcomes the optimization constraints of 3DGSbased approaches in various tasks, such as novel view synthesis and surface reconstruction. Firstly, we theoretically derive that the 3DGS optimization procedure can be modeled as a PDE, and introduce a viscous term to ensure stable optimization. Secondly, we use the Material Point Method (MPM) to obtain a stable numerical solution of the PDE, which enhances both global and local constraints. Additionally, an effective Gaussian densification strategy and particle constraints are introduced to ensure fine-grained details. Extensive qualitative and quantitative experiments confirm that our method achieves state-of-the-art rendering and reconstruction quality.

CCS Concepts: â¢ Computing methodologies â Rendering; Point-based models; Machine learning approaches; â¢ Mathematics of computing â Differential equations.

Additional Key Words and Phrases: novel view synthesis, radiance fields, 3D gaussians, PDE

## ACM Reference Format:

Anonymous Author(s). 2025. Plug-and-Play PDE Optimization for 3D Gaussian Splatting: Toward High-Quality Rendering and Reconstruction. ACM Trans. Graph. 1, 1 (September 2025), 15 pages. https://doi.org/10.1145/nnnnnnn. nnnnnnn

## 1 INTRODUCTION

The reconstruction of 3D scenes from multi-view images is a classic problem in computer vision and computer graphics. Recent advances in Neural Radiance Fields (NeRF) [Mildenhall et al. 2021] have revolutionized this task by introducing implicit neural representations, achieving state-of-the-art results. A notable follow-up is 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023], which has gained increasing attention due to its high-quality, real-time rendering performance, attributed to its explicit point-based representation and efficient splatting process.

When applied to complex scenes, 3DGS encounters blurring and floaters, as validated in Figure 1, resulting in degraded rendering and reconstruction quality. As shown in Figure 2(a), 3DGS tends to employ large Gaussians to fill voids in the scene, which struggle to accurately represent high-frequency details, resulting in overreconstruction and visible blurring. Although small Gaussians are more effective at capturing high-frequency scene details, they tend to introduce numerous floaters, as demonstrated in Fig. 2(b). Regions with limited scene coverage tend to produce floaters in novel views, as the Gaussians are optimized to align with the training views.

Existing works [Ye et al. 2024; Zhang et al. 2024b] propose dividing large Gaussians into a greater number of smaller Gaussians using effective densification criteria. These methods employ an adaptive approach by fitting the scene with an excessive number of Gaussians, which is neither storage-efficient nor effective for rendering.

Through intensive study, we have identified the reason why small Gaussians are prone to unstable optimization. According to the gradient computation, the magnitude of the positional gradient is significantly higher than that of the other attribute gradients when the Gaussian scale is small. Consequently, 3DGS tends to move these small Gaussians to fit the scene, thereby hindering the optimization of other Gaussian attributes during the optimization process. This abrupt positional change results in redundant and ambiguous geometric structures. To ensure stable gradient optimization, existing gradient optimization methods typically emphasize gradient clipping [Pascanu et al. 2013; Zhang et al. 2019], normalization [Ioffe 2015; Santurkar et al. 2018], and weight decay [Yong et al. 2020; Zhang et al. 2018]. However, these methods are heuristic in nature and inevitably lead to information loss.

In this paper, we aim to enable 3DGS to bypass its original optimization weaknesses and achieve more efficient and stable optimization. Building on the above observation, we propose the following insights: (1) The 3DGS optimization procedure can be modeled as the discretization of a partial differential equation (PDE). In this formulation, the attributes of the 3DGS are treated as functions of time. (2) Inspired by fluid simulation [MÃ¼ller et al. 2003], we introduce a viscous term into the PDE to suppress abrupt motion changes and achieve stable optimization. The viscous term, which constrains particles through the local average velocity, effectively prevents abrupt changes in the motion of particles.

We propose a novel, plug-and-play optimization framework based on PDEs, termed PDEO, that enhances existing 3DGS-based approaches for tasks such as novel view synthesis and surface reconstruction. The goal is to adapt large Gaussians into smaller ones to better capture high-frequency details, and to enable stable optimization of small Gaussians for improved rendering and reconstruction quality. Firstly, we theoretically derive that the 3DGS optimization procedure can be modeled as a PDE, and introduce a viscous term to ensure stable optimization. Secondly, we employ the Material Point Method (MPM) [Jiang et al. 2016] to solve the PDE, thereby enforcing both global and local constraints for optimization. Finally, we propose explicit particle constraints to enforce small-scale, high-confidence Gaussians in accordance with the particle hypothesis and an effective Gaussian densification strategy to to ensure fine-grained details. Extensive experiments demonstrate that our PDEO improves upon state-of-the-art methods as a plug-and-play optimizer, consistently enhancing performance in both novel view synthesis and surface reconstruction.

In summary, the main contributions are provided as follows:

â¢ We propose a novel, plug-and-play optimization framework based on PDEs, which enhances existing 3DGS-based approaches in novel view synthesis and surface reconstruction.

â¢ We formulate the 3DGS optimization procedure as a PDE and introduce a viscous term to ensure stable optimization of Gaussians.

<!-- image-->

(a)  
<!-- image-->  
Training View Novel View (b)

<!-- image-->  
(c)  
Fig. 2. Optimization of 3D Gaussians. (a) The redundantly of large 3D Gaussians. (b) The ambiguity of small 3D Gaussians. (c) Visualization results.

## 2 RELATED WORK

## 2.1 Novel View Synthesis

The recent success of Neural Radiance Fields (NeRF) [Mildenhall et al. 2021] introduces an implicit scene representation that achieves high rendering quality in novel view synthesis. Subsequent methods [Barron et al. 2022; Philip and Deschaintre 2023; Warburg et al. 2023; Wirth et al. 2023] are proposed to improve the original NeRF. For instance, Mip-NeRF [Barron et al. 2022] proposes a new feature representation of the integrated positional encoding to improve the rendering quality. Later, MipNeRF360 [Barron et al. 2022] extends this method to unbounded scenes by using a non-linear scene parameterization. Another line of work [Chen et al. 2023; Liu et al. 2023; Somraj and Soundararajan 2023] focuses on improving the efficiency of NeRF, which proposes to accelerate training and rendering by introducing volumetric features [Fridovich-Keil et al. 2022; Yu et al. 2021] or sparse hash-based grids [MÃ¼ller et al. 2022].

3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] improves training and rendering speed by introducing anisotropic 3D Gaussians and efficient splatting, which supports forward rasterization and avoids the shortcomings of expensive sampling and queries. Some subsequent works on 3DGS further enhance performance for novelview synthesis. For example, AbsGS [Ye et al. 2024] and Fregs [Zhang et al. 2024b] enhance the densification strategy of 3DGS to achieve more accurate density adjustments. GES [Hamdi et al. 2024], DisC-GS [Qu et al. 2024] and 3D-HGS [Li et al. 2024] refine the basis function representation of 3D Gaussians to provide a more precise and detailed representation. Recently, 3DGS2 [Lan et al. 2025] proposes a second-order convergent training algorithm for 3DGS, which achieves a tenfold increase in training speed.

## 2.2 Neural Surface Reconstruction

Due to the absence of surface constraints, NeRF cannot extract high-quality surfaces. NeuS [Wang et al. 2021] introduces a Signed Distance Field (SDF) to represent the geometric surfaces of the scene and improves the rendering formulations to achieve more accurate results. Neuralangelo [Li et al. 2023] introduces hash encoding into the SDF to enable detailed large-scale scene reconstruction. Binary Opacity Grids [Reiser et al. 2024] employ a discrete opacity grid to represent the scene, allowing for a more accurate representation. However, these methods continue to demand substantial training time owing to the high computational cost of volume rendering.

Recently, various studies [Chung et al. 2024; Geiger et al. 2024; Jiang et al. 2016; Zhang et al. 2024a] have extended 3DGS to surface reconstruction. For instance, SuGaR [GuÃ©don and Lepetit 2024] introduces a regularization term that encourages the 3D Gaussians to align with the surface, facilitating more effective mesh extraction. GOF [Yu et al. 2024b] proposes a ray-tracing-based volume rendering approach to enable direct extraction of geometry from unbounded scenes. 2DGS [Geiger et al. 2024] and RaDeGS [Zhang et al. 2024a] approximate surfaces with Gaussians by imposing shape constraints and incorporating depth information. Different from these methods that introduce explicit geometric constraints, our method uses a PDE-based optimization strategy, achieving more stable optimization and effectively eliminating redundant and ambiguous geometric structures.

## 2.3 Gradient Optimization

During the optimization process, it is not uncommon for one gradient to be significantly larger than the others, a phenomenon known as gradient explosion, which is a widely known issue in optimization. Gradient clipping [Pascanu et al. 2013] is a widely used technique that constrains the gradient by applying an upper limit to the gradient magnitude. Subsequent works [Qian et al. 2021; Zhang et al. 2019] have built upon this approach by introducing more adaptive truncation methods. Batch normalization (BN) [Ioffe 2015; Santurkar et al. 2018] constrains the gradient by normalizing the attributes through a transformation, thereby facilitating a more stable optimization process. Weight decay [Yong et al. 2020; Zhang et al. 2018] modifies the loss function by adding a stabilizing term to constrain the gradient. Although these methods impose reasonable constraints on the gradient, they inevitably result in information loss in the original gradient. In contrast, our approach modifies the governing PDE to stabilize the gradient while preserving the integrity of the original gradient information.

## 2.4 Material Point Method

Simulating natural phenomena for virtual worlds is a crucial application that remains extremely challenging. The Material Point Method (MPM) [Sulsky et al. 1995] has been demonstrated to be an effective hybrid particle/grid method for simulating various solid/fluid materials in the solution of a partial differential equation (PDE), emerging as a generalization of the Particle-in-Cell (PIC) and Fluid Implicit Particle (FLIP) methods [Jiang et al. 2016]. MPM methods combine Lagrangian material particles [Bargteil et al. 2006; Ummenhofer et al. 2019] with Eulerian Cartesian grids [Takagi et al. 2012; Tompson et al. 2017], which discretizes the initial PDE problem using material particles. For example, Stomakhin et al. [Stomakhin et al. 2013] employ the MPM to simulate snow, producing convincing results. Yue et al. [Yue et al. 2015] demonstrate that MPM is also suitable for simulating complex fluids, such as foams. In this work, we propose PDE-GS, which models 3DGS optimization as a PDE, thereby introducing the MPM to solve the 3DGS optimization and achieve more stable and efficient optimization.

## 3 PRELIMINARY AND MOTIVATION

## 3.1 Preliminary

3.1.1 3D Gaussian Splatting. 3DGS [Kerbl et al. 2023] employs a set of learnable 3D Gaussians that encapsulate surrounding information to represent the scene explicitly. Each 3D Gaussians ???? is parameterized by learnable attributes of center position $\mu _ { i } ,$ opacity $\hat { o } _ { i } ,$ color $\hat { \mathbf { c } } _ { i }$ and a covariance matrix ??,

$$
g _ { i } ( \pmb { \mu } ) = \exp ( - \frac { 1 } { 2 } ( \pmb { \mu } - \pmb { \mu } _ { i } ) ^ { T } \Sigma ^ { - 1 } ( \pmb { \mu } - \pmb { \mu } _ { i } ) )\tag{1}
$$

where the covariance matrix ?? is denoted by the rotation matrix ?? and the scaling matrix ?? as $\Sigma = R S S ^ { T } R ^ { T }$

To render an image, the 3D Gaussians are projected onto the image plane and converted into 2D Gaussians through the splatting operation [Zwicker et al. 2001]. Subsequently, the color ?? of a pixel is computed by combining ?? ordered Gaussians using ??-blending,

$$
C = \sum _ { i \in N } \hat { \mathbf { c } } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )\tag{2}
$$

where $\alpha _ { j }$ is computed by the 2D Gaussian multiplied with the opacity ??Ë?? . $\hat { o } _ { i }$

3.1.2 Material Point Method. MPM [Jiang et al. 2016] is a discrete method for solving PDE, widely used in solid and fluid simulation [MÃ¼ller et al. 2003]. MPM combines the two perspectives of the system: the Lagrangian description and the Eulerian description. In the Lagrangian description, the system is regarded as a discrete phase comprising numerous independent particles, each endowed with its own attributes. In contrast, the Eulerian description treats the system as a continuum phase, which enables a global description of the particle motion.

Specifically, the motion equation of particles evolves over time ?? as: $\begin{array} { r } { f ( \pmb { \upsilon } , \pmb { x } ) = \frac { \partial \pmb { v } } { \partial t } } \end{array}$ , where ?? is velocity and ?? is position. Then, MPM is used to discrete the function as: $f ( v , x ) = v ^ { t + 1 } - v ^ { t }$

## 3.2 Gradient Analysis

3DGS employs gradient descent for scene optimization, a process that is essential for achieving high-quality scene representation. Each Gaussian ???? is associated with a set of trainable attributes $\Gamma _ { i } ^ { t } ~ = ~ \{ \mu _ { i } , \mathbf { c } _ { i } , o _ { i } , s _ { i } , \mathbf { q } _ { i } \}$ , where $\pmb { \mu } _ { i }$ denotes the center position, c?? represents the spherical harmonic coefficients, ???? is the opacity attribute, $\mathbf { \boldsymbol { s } } _ { i }$ refers to the scale attributes, and ${ \bf q } _ { i }$ is the quaternion representing the rotation attributes. Here, $\hat { o } _ { i } = \operatorname { S i g } ( o _ { i } )$ denotes the opacity, where Sig(Â·) represents the sigmoid function. $\hat { \mathbf { s } } _ { i } = \exp ( \mathbf { s } _ { i } )$ denotes the scaling vector. During optimization, the update of each attribute is given by,

$$
\triangle \gamma _ { i } = \sigma \frac { \partial L } { \partial \gamma _ { i } } , \gamma _ { i } \in \Gamma _ { i }\tag{3}
$$

where ?? denotes the learning rate and ?? represents the loss function. For simplicity, we consider a single pixel ?? with the L2 loss, $L =$ $| | C - C _ { g t } | | ^ { 2 }$ , where ?? and $C _ { g t }$ denote the rendered color and the ground truth color at pixel ??, respectively. The gradient of the loss can be computed using the chain rule,

$$
\frac { \partial L } { \partial \gamma _ { i } } = 2 \big ( C - C _ { g t } \big ) \frac { \partial C } { \partial \gamma _ { i } } , \gamma _ { i } \in \Gamma _ { i }\tag{4}
$$

By integrating along the viewing ray ?? associated with pixel ?? and considering ?? ordered Gaussians, the equation can be expanded as,

$$
\frac { \partial C } { \partial \gamma _ { i } } = \sum _ { k \in N } \frac { \partial ( T _ { k } C _ { k } \operatorname { S i g } ( o _ { k } ) \int _ { \mathbf { x } \in l } g _ { k } ( \mathbf { x } ) d \mathbf { x } ) } { \partial \gamma _ { i } }\tag{5}
$$

where $\begin{array} { r } { T _ { k } = \prod _ { j = 1 } ^ { k - 1 } ( 1 - \alpha _ { j } ) } \end{array}$ denotes the transmittance.

As demonstrated in Appendix A.1, the magnitude of the positional gradient is significantly greater than that of the other parameter gradients when the scale of the Gaussian is small.

$$
\frac { \partial L } { \partial \pmb { \mu } _ { i } } \gg \frac { \partial L } { \partial \mathbf { c } _ { i } } \sim \frac { \partial L } { \partial o _ { i } } \sim \frac { \partial L } { \partial \mathbf { s } _ { i } } \sim \frac { \partial L } { \partial ( \mathbf { q } _ { i } \cdot \boldsymbol { r } _ { q , i } ) }\tag{6}
$$

where â¼ denotes asymptotic equivalence, and $r _ { q , i }$ denotes the update direction of $q _ { i } ,$ which is governed by the definition of the quaternion.

## 4 METHOD

In this paper, we propose a new plug-and-play optimization framework, called PDEO, which leverages PDEs to enhance the rendering and reconstruction quality of 3DGS-based methods. An overview of our framework is shown in Fig. 3. In Section 4.1, we first establish the PDE formulation for the 3DGS optimization procedure and introduce a viscous term to enhance the stability of the optimization. Secondly, we employ the MPM to solve the PDE by Particle-to-Grid (P2G) and Grid-to-Particle (G2P) strategies in Section 4.2. Finally, we propose explicit particle constraints to enforce small-scale, highconfidence Gaussians in accordance with the particle hypothesis in Section 4.3.

## 4.1 PDE based 3DGS Optimization

In this section, we establish the PDE formulation for the 3DGS optimization procedure, which allows us control 3DGS optimization by explicitly modifying the PDE.

4.1.1 Formulation. In a PDE, time represents the sequence of the attribute update process, allowing the system state to transition to the next state by changing attributes, similar to the iteration steps in 3DGS optimization. Thus, the attributes of Gaussians in the update process are the functions of time ??. For the original 3DGS, the optimization procedure can be expressed as: $\begin{array} { r } { \pmb { \mu } _ { i } ^ { t + 1 } = \pmb { \mu } _ { i } ^ { t } + \sigma \frac { \partial { \cal L } ^ { t } } { \partial \pmb { \mu } _ { i } ^ { t } } } \end{array}$ where ?? is the learning rate and $\mu _ { i } ^ { t }$ is the position of Gaussians ???? at time ??. We define the discrete velocity $\boldsymbol { \upsilon } _ { i } ^ { t }$ of Gaussians ?? at time ?? as $v _ { i } ^ { t } = \pmb { \mu } _ { i } ^ { t + 1 } - \pmb { \mu } _ { i } ^ { t }$ . Thus the velocity equation in continuous form is:

$$
{ \boldsymbol { v } } _ { i } ^ { t } = \sigma \frac { \partial L ^ { t } } { \partial \pmb { \mu } _ { i } ^ { t } }\tag{7}
$$

Then, we calculate the partial derivatives of the equation with time ?? :

$$
\frac { d v _ { i } ^ { t } } { d t } = \sigma \nabla \frac { d L ^ { t } } { d t } = \sigma \frac { d } { d t } ( \frac { \partial L ^ { t } } { \partial \mu _ { i } ^ { t } } ) = \sigma \sum _ { \gamma _ { i } ^ { t } \in \Gamma _ { i } ^ { t } } \sigma \frac { \partial L ^ { t } } { \partial \gamma _ { i } ^ { t } } \cdot \frac { \partial } { \partial \gamma _ { i } ^ { t } } ( \frac { \partial L ^ { t } } { \partial \mu _ { i } ^ { t } } )\tag{8}
$$

where â½ is the differential operator on position $\mu _ { i } ^ { t } , \gamma _ { i } ^ { t }$ represents the attribute of ???? in the attribute set $\Gamma _ { i } ^ { t } = \{ \mu _ { i } ^ { t } , \mathbf { c } _ { i } ^ { t } , o _ { i } ^ { t } , \mathbf { \bar { s } } _ { i } ^ { t } , \mathbf { \bar { q } } _ { i } ^ { t } \}$ . According the definition of the time derivative and the NewtonâLeibniz formula, the final motion equation is defined as:

$$
\frac { d \boldsymbol { v } _ { i } ^ { t } } { d t } = \frac { \partial \boldsymbol { v } _ { i } ^ { t } } { \partial t } + \boldsymbol { v } _ { i } ^ { t } \cdot \nabla \boldsymbol { v } _ { i } ^ { t } = \frac { \sigma ^ { 2 } } { 2 } \sum _ { \gamma _ { i } ^ { t } \in \Gamma _ { i } ^ { t } } \nabla ( \frac { \partial L ^ { t } } { \partial \gamma _ { i } ^ { t } } ) ^ { 2 }\tag{9}
$$

4.1.2 Viscous Term. Unlike 3DGS optimization, particle position updating is stable and controllable during fluid simulation, which is attributed to the viscous term [MÃ¼ller et al. 2003] in the motion equations.

$$
{ \frac { \partial { \boldsymbol { v } } } { \partial t } } + { \boldsymbol { v } } \cdot \nabla { \boldsymbol { v } } + { \frac { 1 } { \rho } } \nabla { \boldsymbol { p } } = { \boldsymbol { F } } + { \boldsymbol { v } } \nabla \cdot \nabla { \boldsymbol { v } }\tag{10}
$$

where $\begin{array} { r } { \nabla \pmb { v } = \pmb { 0 } , } \end{array}$ ?? is time, $\rho$ is density, $\mathcal { P }$ is pressure, ?? is viscosity, ?? is gravity acceleration, and ?? is the velocity of the fluid field, which is equal to the derivative of the particle position $\mu , { \mathrm { i . e . ~ } } \upsilon = { \frac { \partial \mu } { \partial t } }$ The viscous term ??â½ Â· â½?? essentially imparts an acceleration to the particles in the system, directing them towards the average velocity of their surroundings, which can be equivalently interpreted as mixing the velocity of the particles with the average velocity of the surrounding particles.

Inspired by fluid simulation [MÃ¼ller et al. 2003], we introduce a viscous term into the 3DGS optimization procedure. Therefore, we rewrite Eq.9 as:

$$
\frac { d \boldsymbol { v } _ { i } ^ { t } } { d t } = \frac { \partial \boldsymbol { v } _ { i } ^ { t } } { \partial t } + \boldsymbol { v } _ { i } ^ { t } \cdot \nabla \boldsymbol { v } _ { i } ^ { t } = \frac { \sigma ^ { 2 } } { 2 } \sum _ { \gamma _ { i } ^ { t } \in \Gamma _ { i } ^ { t } } \nabla ( \frac { \partial L ^ { t } } { \partial \gamma _ { i } ^ { t } } ) ^ { 2 } + ( 1 - \lambda _ { g } ) \nabla \cdot \nabla \boldsymbol { v } _ { i } ^ { t }\tag{11}
$$

where $\lambda _ { g }$ is the weighting coefficient. Following the fundamental tenet of PDE, when ?? is equal to zero, the energy of ?? diminishes in a gradual manner with respect to ?? and ultimately approaches zero. Thus, introducing the viscosity does not change the solution of the equation as ?? tends to infinity, which is the theoretical result of the 3DGS optimization.

To this end, the discrete solution can be computed as:

$$
\pmb { \mu } _ { i } ^ { t + 1 } = \pmb { \mu } _ { i } ^ { t } + \sigma \frac { \partial { L } ^ { t } } { \partial \pmb { \mu } _ { i } ^ { t } } + \frac { 1 - \lambda _ { g } } { | \boldsymbol { N } _ { i } | } \sum _ { j \in { N } _ { i } } ( \pmb { \upsilon } _ { j } ^ { t } - \pmb { \upsilon } _ { i } ^ { t } )\tag{12}
$$

where $N _ { i }$ is the neighbour set of Gaussian ???? .

<!-- image-->  
Fig. 3. Overview of the proposed PDEO. 3D Gaussians are initialized by COLMAP [Schonberger and Frahm 2016]. We formulate the 3DGS optimization procedure as a Partial Differential Equation (PDE) and introduce a viscosity term to achieve stable optimization. Specifically, we employ the Material Point Method (MPM) to solve the PDE by Particle-to-Grid (P2G) and Grid-to-Particle (G2P). The velocity field is constructed to store the excess velocity of Gaussians and gradually release it to ensure the stability of Gaussian motion. In addition, we propose explicit particle constraints to enforce small-scale, high-confidence Gaussians in accordance with the particle hypothesis

## 4.2 MPM based Solution

In this section, we present numerical simulations of the 3DGS optimization procedure according to the discretization form of Eq.12. We can approximate the equation as:

$$
\pmb { \mu } _ { i } ^ { t + 1 } = \pmb { \mu } _ { i } ^ { t } + \sigma \frac { \partial L ^ { t } } { \partial \pmb { \mu } _ { i } ^ { t } } + \frac { 1 - \lambda _ { g } } { | N _ { i } | } \sum _ { j \in N _ { i } } ( \frac { \partial L ^ { t } } { \partial \pmb { \mu } _ { j } ^ { t } } - \frac { \partial L ^ { t } } { \partial \pmb { \mu } _ { i } ^ { t } } )\tag{13}
$$

Since calculating the motion of each 3D Gaussian based on its neighbour is computationally expensive after introducing the viscous term, we treat 3D Gaussians as particles and employ the MPM to solve this problem. Specifically, we incorporate the Particle-to-Grid (P2G) and Grid-to-Particle (G2P) strategies into the 3DGS optimization procedure, suppressing particle motion while providing additional motion guidance to solve the motion equation. We construct a velocity field by dividing the scene space into voxel grids. Particles can update motion by storing excess velocity in the voxel grids and gaining additional velocity from the voxel grids. Therefore, particles are effectively regulated using local information for the velocity field, thereby introducing the viscous term into the optimization procedure.

4.2.1 Particle-to-Grid. The P2G process constructs a grid which stores the excess velocity of particles in the voxel grids. As mentioned above, the position of the Gaussian $g _ { i }$ is updated by $\triangle \pmb { \mu } _ { i } ^ { t } =$ ${ \partial L ^ { t } } / { \partial \mu _ { i } ^ { t } }$ , which is computed from the gradient of the loss. Smallerscale Gaussians are more prone to positional mutations, which leads to instability in the optimization procedure. Thus, a reasonable reduction in velocity would be an optimization benefit. Specifically, we employ the P2G process to attenuate the particle velocity $\triangle \mu _ { i } ^ { t }$ ?? while also preserving the motion characteristics of the particles. We store the excess velocity of the particle $g _ { i }$ into the voxel grid $V _ { n }$ at step ?? ,

$$
\boldsymbol { v } _ { n } ^ { t + 1 } = \lambda _ { g } \boldsymbol { v } _ { n } ^ { t } + \left( 1 - \lambda _ { g } \right) \triangle \boldsymbol { v } _ { n } ^ { t } = \lambda _ { g } \boldsymbol { v } _ { n } ^ { t } + \frac { 1 - \lambda _ { g } } { | \boldsymbol { R } _ { n } ^ { t } | } \sum _ { g _ { i } \in \boldsymbol { R } _ { n } ^ { t } } \triangle \mu _ { i } ^ { t }\tag{14}
$$

where $R _ { n } ^ { t }$ belongs to $R ^ { t } = \{ R _ { 0 } ^ { t } , . . . . , R _ { N } ^ { t } \}$ is the set of particles contained within the voxel grid $V _ { n } , \upsilon _ { n } ^ { t }$ is the voxel velocity saved in $V _ { n } ,$ and $\lambda _ { g }$ is weighting coefficient. We show that the selection of $\lambda _ { g }$ has no impact on the total gradient in the Appendix A.2.

4.2.2 Grid-to-Particle. The grid not only suppresses particle velocity but also provides additional motion guidance for the particles. Since the velocity field represents the average motion tendency of particles in the voxel grid, the voxel velocity is then used to guide the motion of the particles:

$$
\bigtriangleup \hat { \pmb { \mu } } _ { i } ^ { t } = \lambda _ { \hat { p } } \triangle \pmb { \mu } _ { i } ^ { t } + \big ( 1 - \lambda _ { \hat { p } } \big ) \pmb { \upsilon } _ { n } ^ { t } , \pmb { \mu } _ { i } ^ { t + 1 } = \pmb { \mu } _ { i } ^ { t } + \bigtriangleup \hat { \pmb { \mu } } _ { i } ^ { t }\tag{15}
$$

where the particle velocity is suppressed by the coefficient $\lambda _ { p }$ and the $\triangle \hat { \mu } _ { i } ^ { t }$ is the updated velocity. The updated velocity represents the most likely direction of position optimization for the particles. The velocities of the different particles interact with each other, thereby cancelling out abrupt changes in position attributes across different directions while receiving additional velocity guidance from the voxel velocity. Consequently, the variation of the position gradient is successfully guided by the viscosity term.

## 4.3 Particle Constraints

4.3.1 Scale Loss. In PDE, particles are scale-free attributes. Conversely, Gaussian functions with large scales can occupy a large space, which is contrary to the assumptions of PDE systems. Therefore, we introduce scale constraints for 3D Gaussians:

$$
L _ { s } = \frac { 1 } { \left| G _ { k } \right| } \sum _ { g _ { i } \in G _ { k } } m a x ( s ^ { * } - \beta , 0 )\tag{16}
$$

where $s ^ { * }$ means the largest scale of $g _ { i } , G _ { k }$ is the set of 3D Gaussians which is visible in viewpoint $k ,$ and $\beta$ is the margin for the scale. This loss punishes the large scales of Gaussians. The small-scale Gaussians ensure the ability to capture high-frequency details.

Table 1. Quantitative results on Mip-NeRF 360 [Barron et al. 2022], Tanks&Temples [Knapitsch et al. 2017] and Scanet++ [Yeshwanth et al. 2023] for Novel view synthesis. The best results are highlighted in bold. PDEO consistently improves the performance.
<table><tr><td>Dataset</td><td colspan="4">Mip-NeRF360[Barron et al. 2022]</td><td colspan="4">Tanks&amp;Temples[Knapitsch et al. 2017]</td><td colspan="4">Scanet++[Yeshwanth et al. 2023]</td></tr><tr><td>Method</td><td>PSNRâ SSIMâ</td><td>Lâ emâ FPSâ</td><td></td><td></td><td>PSNRâ SSIMâ</td><td></td><td></td><td>LPSâ Memâ FPSâ</td><td></td><td>PSNRâ SSIMâ</td><td>LIPSâ Memâ FPSâ</td><td></td><td></td></tr><tr><td>3DGS</td><td>27.77 0.827</td><td>0.244</td><td>295</td><td>163.1</td><td>21.63</td><td>0.768</td><td>0.322</td><td>299</td><td>44.3</td><td>27.83</td><td>0.911</td><td>0.185 192</td><td>74.2</td></tr><tr><td>GES</td><td>27.71 0.844</td><td>0.224</td><td>369</td><td>106.3</td><td>21.59</td><td>0.768</td><td>0.330</td><td>162</td><td>64.1</td><td>27.86</td><td>0.912</td><td>0.190 94.1</td><td>97.9</td></tr><tr><td>AbaGS</td><td>27.81 0.850</td><td>0.207</td><td>804</td><td>125.1</td><td>21.37</td><td>0.755</td><td>0.326</td><td>340 40.0</td><td></td><td>27.67 0.907</td><td>0.185</td><td>121</td><td>101.8</td></tr><tr><td>MipGS</td><td>27.98 0.858</td><td>0.213</td><td>303</td><td>108.5</td><td>20.98</td><td>0.757</td><td>0.326</td><td>357 52.1</td><td></td><td>27.80 0.913</td><td>0.177</td><td>224</td><td>135.2</td></tr><tr><td>2DGS</td><td>27.42 0.841</td><td>0.228</td><td>476</td><td>42.3</td><td>21.02</td><td>0.756</td><td>0.357</td><td>188</td><td>21.8</td><td>27.91 0.911</td><td>0.196</td><td>107</td><td>36.8</td></tr><tr><td>RaDeGS</td><td>28.03 0.866</td><td>0.198</td><td>536</td><td>118.6</td><td>20.80</td><td>0.750</td><td>0.345</td><td>239 57.5</td><td></td><td>27.97 0.911</td><td>0.180</td><td>165</td><td>103.4</td></tr><tr><td>MCMC</td><td>27.91 0.845</td><td>0.186</td><td>714</td><td>40.4</td><td>21.03</td><td>0.744</td><td>0.318</td><td>691 55.7</td><td></td><td>28.01 0.918</td><td>0.182</td><td>470.3</td><td>52.5</td></tr><tr><td>SpecGS</td><td>27.96 0.866</td><td>0.173</td><td>1147</td><td>7.9</td><td>21.02</td><td>0.751</td><td>0.322</td><td>498</td><td>19.7</td><td>27.89 0.912</td><td>0.195</td><td>159</td><td>56.1</td></tr><tr><td>3DGS+PDEO</td><td>27.78 0.831</td><td>0.242</td><td>186</td><td>225.5</td><td>21.89</td><td>0.768</td><td>0.320</td><td>125</td><td>146.9</td><td>27.87</td><td>0.911</td><td>0.190 66.7</td><td>260.0</td></tr><tr><td>GES+PDEO</td><td>27.99 0.834</td><td>0.232</td><td>133</td><td>166.1</td><td>22.08</td><td>0.768</td><td>0.325</td><td>97.0</td><td>176.0</td><td>27.92</td><td>0.911 0.192</td><td>53.5</td><td>283.0</td></tr><tr><td>MipGS+PDEO</td><td>28.08 0.870</td><td>0.211</td><td>137</td><td>108.5</td><td>22.12</td><td>0.761</td><td>0.320</td><td>79.0</td><td>148.5</td><td>27.91 0.913</td><td>0.169</td><td>48.5</td><td>254.5</td></tr><tr><td>2DGS+PDEO</td><td>27.42 0.832</td><td>0.273</td><td>63.8</td><td>94.5</td><td>21.03</td><td>0.749</td><td>0.363</td><td>100</td><td>64.6</td><td>27.93 0.911</td><td>0.195</td><td>102</td><td>81.7</td></tr><tr><td>RaDeGS+PDEO</td><td>28.16 0.852</td><td>0.213</td><td>187</td><td>171.1</td><td>22.61</td><td>0.768</td><td>0.332</td><td>95.1</td><td>118.4</td><td>28.06 0.911</td><td>0.189</td><td>65.0</td><td>227.9</td></tr><tr><td>MCMC+PDEO</td><td>28.12 0.833</td><td>0.213</td><td>198</td><td>73.3</td><td>22.77</td><td>0.780</td><td>0.295</td><td>210</td><td>73.9</td><td>28.23</td><td>0.919 0.182</td><td>212</td><td>86.9</td></tr><tr><td>SpecGS+PDEO</td><td>28.81 0.875</td><td>0.173</td><td>99.6</td><td>65.4</td><td>22.16</td><td>0.780</td><td>0.316</td><td>345</td><td>28.2</td><td>28.10</td><td>0.919 0.185</td><td>115</td><td>66.8</td></tr><tr><td>3DGS(rander)</td><td>26.61 0.764</td><td>0.318</td><td>258</td><td>78.8</td><td>20.84</td><td>0.734</td><td>0.380</td><td>261</td><td>66.1</td><td>27.55</td><td>0.908 0.202</td><td>164.9</td><td>92.4</td></tr><tr><td>+PDEO(rander)</td><td>27.75 0.825</td><td>0.233</td><td>89.4</td><td>122.3</td><td>21.71</td><td>0.745</td><td>0.361</td><td>101</td><td>71.5</td><td>27.64</td><td>0.908 0.199</td><td>59.5</td><td>138.0</td></tr><tr><td>MCMC(rander)</td><td>27.62 0.832</td><td>0.203</td><td>473</td><td>55.6</td><td>21.00</td><td>0.735</td><td>0.333</td><td>469</td><td>35.8</td><td>27.92</td><td>0.918 0.187</td><td>354</td><td>60.7</td></tr><tr><td>+PDEO(rander)</td><td>27.85 0.861</td><td>0.187</td><td>189</td><td>85.9</td><td>22.64</td><td>0.771</td><td>0.321</td><td>187</td><td>54.6</td><td>27.98</td><td>0.918 0.182</td><td>98.7</td><td>63.7</td></tr></table>

Table 2. Quantitative results on the DTU Dataset [Jensen et al. 2014] for surface reconstruction. We report the Chamfer Distance error of different methods. The best results are highlighted in bold. PDEO consistently improves the performance.
<table><tr><td>Method</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td></tr><tr><td>NeRF</td><td>1.90</td><td>1.60</td><td>1.85</td><td>0.58</td><td>0.81</td><td>2.28</td><td>1.27</td><td>1.47</td><td>1.67</td><td>2.05</td><td>1.07</td><td>0.88</td><td>1.06</td><td>1.15</td><td>0.96</td><td>1.37</td></tr><tr><td>NeuS</td><td>1.00</td><td>1.37</td><td>0.93</td><td>0.43</td><td>1.10</td><td>0.65</td><td>0.57</td><td>1.48</td><td>1.09</td><td>0.83</td><td>0.52</td><td>1.20</td><td>0.35</td><td>0.49</td><td>0.54</td><td>0.84</td></tr><tr><td>3DGS</td><td>1.62</td><td>1.25</td><td>1.41</td><td>1.13</td><td>2.57</td><td>2.10</td><td>1.39</td><td>1.97</td><td>1.82</td><td>1.34</td><td>1.41</td><td>1.90</td><td>1.10</td><td>1.14</td><td>1.29</td><td>1.56</td></tr><tr><td>SuGaR</td><td>1.47</td><td>1.33</td><td>1.13</td><td>0.61</td><td>2.25</td><td>1.71</td><td>1.15</td><td>1.63</td><td>1.62</td><td>1.07</td><td>0.79</td><td>2.45</td><td>0.98</td><td>0.88</td><td>0.79</td><td>1.32</td></tr><tr><td>GOF</td><td>0.50</td><td>0.82</td><td>0.37</td><td>0.37</td><td>1.12</td><td>0.74</td><td>0.73</td><td>1.18</td><td>1.29</td><td>0.68</td><td>0.77</td><td>0.90</td><td>0.42</td><td>0.66</td><td>0.49</td><td>0.74</td></tr><tr><td>2DGS</td><td>0.60</td><td>0.92</td><td>0.79</td><td>0.37</td><td>1.24</td><td>1.13</td><td>0.87</td><td>1.40</td><td>1.27</td><td>0.86</td><td>0.73</td><td>1.33</td><td>0.44</td><td>0.98</td><td>0.60</td><td>0.90</td></tr><tr><td>RaDeGS</td><td>0.46</td><td>0.78</td><td>0.36</td><td>0.39</td><td>0.81</td><td>0.77</td><td>0.76</td><td>1.19</td><td>1.24</td><td>0.63</td><td>0.70</td><td>0.87</td><td>0.36</td><td>0.69</td><td>0.48</td><td>0.70</td></tr><tr><td>3DGS+PDEO</td><td>1.48</td><td>1.01</td><td>1.11</td><td>0.59</td><td>2.35</td><td>1.75</td><td>1.07</td><td>1.69</td><td>1.77</td><td>0.97</td><td>1.03</td><td>1.97</td><td>1.13</td><td>1.10</td><td>1.20</td><td>1.34</td></tr><tr><td>2DGS+PDEO</td><td>0.59</td><td>0.90</td><td>0.70</td><td>0.39</td><td>0.89</td><td>0.86</td><td>0.82</td><td>1.31</td><td>1.29</td><td>0.74</td><td>0.73</td><td>1.43</td><td>0.44</td><td>0.72</td><td>0.48</td><td>0.82</td></tr><tr><td>RaDeGS+PDEO</td><td>0.45</td><td>0.77</td><td>0.36</td><td>0.37</td><td>0.73</td><td>0.75</td><td>0.75</td><td>1.18</td><td>1.16</td><td>0.59</td><td>0.67</td><td>0.84</td><td>0.38</td><td>0.68</td><td>0.47</td><td>0.68</td></tr></table>

4.3.2 Confidence Loss. Since Gaussians are described as particles in the PDE, it is necessary to avoid semi-transparent Gaussians. Therefore, we propose a confidence loss to ensure the high confidence of Gaussians, satisfying the particle hypothesis, which corresponds to the opacity of the Gaussian,

$$
L _ { t } = \frac { 1 } { G _ { k } } | G _ { k } | | o _ { i } - \left\lfloor 1 . 9 9 o _ { i } \right\rfloor \ : | _ { 2 } ^ { 2 }\tag{17}
$$

where âÂ·â denotes the floor operator and $o _ { i }$ denotes the opacity.

4.3.3 Gaussian Densification. Gaussian densification is used in 3DGS to clone and split new Gaussians to cover empty space, thus precisely representing underlying scenes. The original 3DGS averages the positional gradient of the view-space position to determine whether to perform densification. In our approach, the velocity field is also used to guide the process of cloning and splitting. Specifically, we calculate the cosine similarity measure between particle velocity â³???? and voxel velocity $_ { v _ { n } }$ to decide whether to perform the densify operation. Densify for Gaussian $g _ { i }$ is performed when it satisfies $\cos ( \triangle \pmb { \mu } _ { i } , \pmb { \upsilon } _ { n } ) > \theta _ { p } ,$ where ?????? (Â·) refers to cosine similarity, and $\theta _ { p }$ denotes the cosine threshold.

## 5 EXPERIMENTS

## 5.1 Experimental Setup

Datasets. In our experiments, we evaluated the proposed PDE-GS across a diverse range of real-world scenes to test its effectiveness in rendering and reconstruction. For novel view synthesis, we use 17 scenes from various datasets: 6 scenes from Mip-nerf360 dataset [Barron et al. 2022], 7 scenes from Tanks & Temples dataset [Knapitsch et al. 2017], and 4 scenes from ScanNet++ dataset [Yeshwanth et al. 2023]. For surface reconstruction, we conduct the experiments on 15 scenes from the DTU [Jensen et al. 2014] and 7 scenes from Tanks & Temples dataset [Knapitsch et al. 2017]. These scenes contain both bounded indoor and unbounded outdoor environments, enabling a comprehensive evaluation.

Table 3. Quantitative results on Tanks&temples [Knapitsch et al. 2017] for surface reconstruction. We report the F1-score of different methods. The best results are highlighted in bold. RaDeGS+PDEO achieves the best F1-score among all compared methods.
<table><tr><td>Method</td><td>2DGS</td><td>RaDeGS</td><td>SuGaR</td><td>RaDeGS+PDEO</td></tr><tr><td>Barn</td><td>0.387</td><td>0.470</td><td>0.171</td><td>0.588</td></tr><tr><td>Caterpillar</td><td>0.210</td><td>0.255</td><td>0.129</td><td>0.343</td></tr><tr><td>Courthouse</td><td>0.126</td><td>0.100</td><td>0.084</td><td>0.128</td></tr><tr><td>Ignatius</td><td>0.517</td><td>0.668</td><td>0.351</td><td>0.780</td></tr><tr><td>Meetingroom</td><td>0.250</td><td>0.240</td><td>0.180</td><td>0.610</td></tr><tr><td>Truck</td><td>0.379</td><td>0.462</td><td>0.225</td><td>0.591</td></tr><tr><td>Church</td><td>0.054</td><td>0.018</td><td>0.035</td><td>0.078</td></tr><tr><td>Mean</td><td>0.275</td><td>0.316</td><td>0.168</td><td>0.445</td></tr></table>

Implementation. To achieve high-quality rendering and reconstruction performance, our PDEO can be easily integrated into existing 3DGS-based methods, such as MipGS [Yu et al. 2024a] or 2DGS [Geiger et al. 2024], for the tasks of novel view synthesis and surface reconstruction. To ensure consistent evaluation, we use the default parameters of the original methods. We set $\lambda _ { g } = 0 . 8 ,$ $\lambda _ { \mathcal { P } } = 0 . 8 , \psi = 0 . 2 , \theta _ { \mathcal { P } } = 1 2 0 ^ { \circ } , \beta = 0 . 6 , \omega _ { t } = 0 . 0 4 , \omega _ { s } = 0 . 0 4$ and ?? increasing from 1 to 2.5 with iteration gradually. All our experiments are conducted on a single V100 GPU.

Metrics. To evaluate the rendering quality, we report PSNR, SSIM, and LPIPS to measure the performance of each dataset. To evaluate the reconstruction quality, we report the Chamfer Disrance (CD) on DTU dataset [Jensen et al. 2014] and the F1-score on Tanks & Temples dataset [Knapitsch et al. 2017].

## 5.2 Comparison

5.2.1 Novel View Synthesis. We integrate the proposed PDEO into state-of-the-art 3DGS-based methods for novel view synthesis, and compare it with 3DGS [Kerbl et al. 2023], GES [Hamdi et al. 2024], AbsGS [Ye et al. 2024], MipGS [Yu et al. 2024a], 2DGS [Geiger et al. 2024], RaDeGS [Zhang et al. 2024a], 3DGS MCMC and SpecGS [Yang et al. 2024].

We report the quantitative results in Table 1. PDEO consistently improves the performance of the original methods in terms of PSNR, SSIM, and LPIPS. We can see that SpecGS+PDEO achieves the best performance. The quantitative results are shown in Fig. 4, demonstrating that PDEO significantly reduces artifacts and floaters while improving rendering quality. For a clear comparison, we also provide visualizations of Gaussian ellipsoids in Fig. 5. Overall, the proposed PDEO significantly enhances 3DGS-based methods while also improving memory efficiency.

5.2.2 Surface Reconstruction. PDEO is integrated with state-of-theart 3DGS-based methods for surface reconstruction and compared with 2DGS [Geiger et al. 2024], RaDeGS [Zhang et al. 2024a], and

Table 4. Ablation study on Mip-NeRF360 Dataset [Barron et al. 2022] for novel view synthesis. We study the influence of each component in our method on the rendering quality and memory usage in Mip-NeRF360 Dataset [Barron et al. 2022].
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Memâ</td></tr><tr><td>Baseline</td><td>27.71</td><td>0.844</td><td>0.224</td><td>369</td></tr><tr><td>w/o P2G and G2P</td><td>27.51</td><td>0.830</td><td>0.227</td><td>240</td></tr><tr><td>w/o Our Densification</td><td>27.96</td><td>0.834</td><td>0.230</td><td>136</td></tr><tr><td>w/o Scale Loss</td><td>27.75</td><td>0.831</td><td>0.236</td><td>132</td></tr><tr><td>w/o Confidence Loss</td><td>27.87</td><td>0.845</td><td>0.219</td><td>177</td></tr><tr><td>Full</td><td>27.99</td><td>0.834</td><td>0.232</td><td>133</td></tr><tr><td> $\operatorname { F u l l } ( \lambda _ { g } = 0 . 5 )$ </td><td>27.84</td><td>0.835</td><td>0.235</td><td>151</td></tr><tr><td> $\operatorname { F u l l } ( \lambda _ { g } = 0 . 9 )$ </td><td>27.66</td><td>0.834</td><td>0.233</td><td>160</td></tr><tr><td> $\mathrm { F u l l } ( \lambda _ { p } = 0 . 5 )$ </td><td>27.69</td><td>0.835</td><td>0.237</td><td>143</td></tr><tr><td> $\mathrm { F u l l } ( \bar { \lambda _ { p } } { = } 0 . 9 )$ </td><td>27.56</td><td>0.833</td><td>0.229</td><td>207</td></tr></table>

SuGaR [GuÃ©don and Lepetit 2024]. As shown in Table 2 and Table 3, PDEO consistently enhances 3DGS-based methods on the DTU dataset in terms of CD error, and on the Tanks & Temples dataset in terms of F1-score. RaDeGS+PDEO achieves qualitatively better reconstructions with more accurate and smoother geometry, as shown in Fig. 6. This demonstrates that PDEO can remove floaters and preserve geometric details to improving reconstruction quality.

## 5.3 Ablation Studies

In this section, we conduct ablation experiments to study the effectiveness of each component of PDEO. We conduct experiments on Mip-NeRF360 dataset [Barron et al. 2022] for novel view synthesis. The quantitative results of the ablations are reported in Table 4 and GES [Hamdi et al. 2024] is used as the baseline.

Effects of P2G and G2P. In Table 4, we examine the impact of P2G and G2P. The absence of this strategy leads to a significant decline in rendering quality, which leads to a decrease in rendering quality. Our approach introduces the viscosity term into the optimization procedure using P2G and G2P strategies, which can ensure stable optimization of Gaussians while reducing memory usage. Additionally, the qualitative rendering results are illustrated in Fig. 7, which demonstrate that P2G and G2P help mitigate artifacts and floaters.

Effects of Gaussian Densification. As shown in Table 4, removing the Gaussian densification strategy results in a degradation of rendering quality, demonstrates that the strategy can achieve more accurate Gaussian densification to fit the details of scenes.

Effects of Scale Loss and Confidence Loss. We analyze the effects of scale loss and confidence loss. Table 4 shows that with a similar amount of memory usage, there is a significant degradation in rendering quality when removing scale loss or confidence loss. Fig. 7 evidences that scale loss helps limit the scale attribute of Gaussians, which facilitates a better reconstruction of scene details.

## 6 CONCLUSION

The reconstruction of detailed features in a scene requires optimizing numerous small-scale 3D Gaussians. However, to these 3D Gaussians, the sensitivity magnitude of the positional gradient is significantly higher than that of the other parameter gradients. The unequal optimization treatment to different Gaussian attributes according to the computation of gradient magnitude leads to the unstable optimization of 3DGS. Therefore, we propose PDEO, which builds the correspondence between the 3DGS optimization and the PDE simulation, to control and guide the 3DGS optimization. Our experimental results demonstrate its effectiveness in enhancing render and reconstruction quality.

Limitation. Our method exhibits a couple of limitations. Firstly, Our method does not involve particle rotation. Future research could incorporate the influence of the spatial voxel grids on particle rotation in the MPM simulation. Secondly, although we introduce voxel grids to guide the optimization of particle direction, our approach still struggles to reliably relocate 3D Gaussians from other regions into areas with substantial gaps in point cloud initialization (e.g., regions of the scene that lack an initial point cloud). Addressing these limitation represents a promising direction for future research.

## REFERENCES

Adam W Bargteil, Tolga G Goktekin, James F Oâbrien, and John A Strain. 2006. A Semi-Lagrangian Contouring Method for Fluid Simulation. ACM Transactions on Graphics (TOG) 25, 1 (2006), 19â38.

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. 2022. Mip-Nerf 360: Unbounded Anti-Aliased Neural Radiance Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5470â5479.

Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and Andrea Tagliasacchi. 2023. Mobilenerf: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16569â16578.

Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee. 2024. Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 811â820.

Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. 2022. Plenoxels: Radiance Fields without Neural Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5501â5510.

Andreas Geiger, Shenghua Gao, Anpei Chen, Zehao Yu, and Binbin Huang. 2024. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. ACM SIGGRAPH 2024 Conference Papers (2024).

Antoine GuÃ©don and Vincent Lepetit. 2024. Sugar: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5354â5363.

Abdullah Hamdi, Luke Melas-Kyriazi, Jinjie Mai, Guocheng Qian, Ruoshi Liu, Carl Vondrick, Bernard Ghanem, and Andrea Vedaldi. 2024. GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19812â19822.

Sergey Ioffe. 2015. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv preprint arXiv:1502.03167 (2015).

Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik AanÃ¦s. 2014. Large Scale Multi-View Stereopsis Evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 406â413.

Chenfanfu Jiang, Craig Schroeder, Joseph Teran, Alexey Stomakhin, and Andrew Selle. 2016. The Material Point Method for Simulating Continuum Materials. 1â52.

Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics (ToG) 42, 4 (2023), 139:1â139:14.

Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Transactions on Graphics (ToG) 36, 4 (2017), 1â13.

Lei Lan, Tianjia Shao, Zixuan Lu, Yu Zhang, Chenfanfu Jiang, and Yin Yang. 2025. 3DGSË2: Near Second-order Converging 3D Gaussian Splatting. arXiv preprint arXiv:2501.13975 (2025).

Haolin Li, Jinyang Liu, Mario Sznaier, and Octavia J. Camps. 2024. 3D-HGS: 3D Half-Gaussian Splatting. arXiv preprint arXiv:2406.02720 (2024).

Zhaoshuo Li, Thomas MÃ¼ller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. 2023. Neuralangelo: High-Fidelity Neural Surface Reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8456â8465.

Xinhang Liu, Yu-Wing Tai, and Chi-Keung Tang. 2023. Clean-NeRF: Reformulating NeRF to account for View-Dependent Observations. arXiv preprint arXiv:2303.14707

(2023).

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing Scenes as Neural Radiance Fields for View Synthesis. Commun. ACM 65, 1 (2021), 99â106.

Matthias MÃ¼ller, David Charypar, and Markus Gross. 2003. Particle-Based Fluid Simulation for Interactive Applications. In Proceedings of the 2003 ACM SIG-GRAPH/Eurographics Symposium on Computer Animation. Citeseer, 154â159.

Thomas MÃ¼ller, Alex Evans, Christoph Schied, and Alexander J. Keller. 2022. Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. ACM transactions on graphics (TOG) 41, 4 (2022), 1â15.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. 2013. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning. 1310â 1318.

Julien Philip and Valentin Deschaintre. 2023. Floaters No More: Radiance Field Gradient Scaling for Improved Near-Camera Training. arXiv preprint arXiv:2305.02756 (2023).

Jiang Qian, Yuren Wu, Bojin Zhuang, Shaojun Wang, and Jing Xiao. 2021. Understanding Gradient Clipping in Incremental Gradient Methods. In International Conference on Artificial Intelligence and Statistics. PMLR, 1504â1512.

Haoxuan Qu, Zhuoling Li, Hossein Rahmani, Yujun Cai, and Jun Liu. 2024. DisC-GS: Discontinuity-Aware Gaussian Splatting. arXiv preprint arXiv:2405.15196 (2024).

Christian Reiser, Stephan Garbin, Pratul Srinivasan, Dor Verbin, Richard Szeliski, Ben Mildenhall, Jonathan Barron, Peter Hedman, and Andreas J. Geiger. 2024. Binary Opacity Grids: Capturing Fine Geometric Detail for Mesh-Based View Synthesis. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1â14.

Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. 2018. How Does Batch Normalization Help Optimization? Advances in Neural Information Processing Systems 31 (2018).

Johannes L Schonberger and Jan-Michael Frahm. 2016. Structure-from-Motion Revisited. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4104â4113.

Nagabhushan Somraj and Rajiv Soundararajan. 2023. Vip-Nerf: Visibility Prior for Sparse Input Neural Radiance Fields. In ACM SIGGRAPH 2023 Conference Proceedings. 1â11.

Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph Teran, and Andrew J. Selle. 2013. A Material Point Method for Snow Simulation. ACM Transactions on Graphics (TOG) 32, 4 (2013), 1â10.

Deborah Sulsky, Shi-Jian Zhou, and Howard L J. Schreyer. 1995. Application of A Particle-in-Cell Method to Solid Mechanics. Computer Physics Communications 87, 1-2 (1995), 236â252.

Shu Takagi, Kazuyasu Sugiyama, Satoshi Ii, and Yoichiro Matsumoto. 2012. A Review of Full Eulerian Methods for Fluid Structure Interaction Problems. (2012).

Jonathan Tompson, Kristofer Schlachter, Pablo Sprechmann, and Ken Perlin. 2017. Accelerating Eulerian Fluid Simulation with Convolutional Networks. In International Conference on Machine Learning. PMLR, 3424â3433.

Benjamin Ummenhofer, Lukas Prantl, Nils Thuerey, and Vladlen Koltun. 2019. Lagrangian Fluid Simulation with Continuous Convolutions. In International Conference on Learning Representations.

Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021. Neus: Learning Neural Implicit Surfaces by Volume Rendering for Multi-View Reconstruction. arXiv preprint arXiv:2106.10689 (2021).

Frederik Warburg, Ethan Weber, Matthew Tancik, Aleksander Holynski, and Angjoo Kanazawa. 2023. Nerfbusters: Removing Ghostly Artifacts from Casually Captured Nerfs. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 18120â18130.

Tristan Wirth, Arne Rak, Volker Knauthe, and Dieter W Fellner. 2023. A Post Processing Technique to Automatically Remove Floater Artifacts in Neural Radiance Fields. In Computer Graphics Forum, Vol. 42. Wiley Online Library, e14977.

Ziyi Yang, Xinyu Gao, Yang-Tian Sun, Yihua Huang, Xiaoyang Lyu, Wen Zhou, Shaohui Jiao, Xiaojuan Qi, and Xiaogang Jin. 2024. Spec-gaussian: Anisotropic viewdependent appearance for 3d gaussian splatting. Advances in Neural Information Processing Systems 37 (2024), 61192â61216.

Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou. 2024. AbsGS: Recovering Fine Details for 3D Gaussian Splatting. arXiv preprint arXiv:2404.10484 (2024).

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias NieÃner, and Angela Dai. 2023. Scannet++: A High-Fidelity Dataset of 3D Indoor Scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12â22.

Hongwei Yong, Jianqiang Huang, Xiansheng Hua, and Lei Zhang. 2020. Gradient centralization: A new optimization technique for deep neural networks. In Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part I 16. Springer, 635â652.

Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. 2021. Plenoctrees for Real-Time Rendering of Neural Radiance Fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5752â5761.

Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024a. Mip-Splatting: Alias-Free 3D Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19447â19456.

Zehao Yu, Torsten Sattler, and Andreas J. Geiger. 2024b. Gaussian Opacity Fields: Efficient and Compact Surface Reconstruction in Unbounded Scenes. arXiv preprint arXiv:2404.10772 (2024).

Yonghao Yue, Breannan Smith, Christopher Batty, Changxi Zheng, and Eitan J. Grinspun. 2015. Continuum Foam: A Material Point Method for Shear-Dependent Flows. ACM Transactions on Graphics (TOG) 34, 5 (2015), 1â20.

Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. 2024a. RaDe-GS: Rasterizing Depth in Gaussian Splatting. arXiv preprint arXiv:2406.01467 (2024).

Guodong Zhang, Chaoqi Wang, Bowen Xu, and Roger Grosse. 2018. Three mechanisms of weight decay regularization. arXiv preprint arXiv:1810.12281 (2018).

Jingzhao Zhang, Tianxing He, Suvrit Sra, and Ali Jadbabaie. 2019. Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity. arXiv preprint arXiv:1905.11881 (2019).

Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and Eric Xing. 2024b. Fregs: 3D Gaussian Splatting with Progressive Frequency Regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21424â21433.

Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. 2001. EWA Volume Splatting. In Proceedings Visualization, 2001. 29â538.

GT  
3DGS  
3DGS+PDEO  
SpecGS  
SpecGS+PDEO  
<!-- image-->  
Fig. 4. Qualitative comparisons of different methods on scenes from Mip-NeRF360 [Barron et al. 2022] and Tanks&Temples [Knapitsch et al. 2017] and Scanet++[Yeshwanth et al. 2023] datasets for novel view synthesis. PEDO significantly reduces artifacts and floaters while improving rendering quality.  
GT  
3DGS

3DGS+PDEO  
GES  
GES+PDEO  
<!-- image-->  
Fig. 5. Visualization of Gaussian ellipsoids. PEDO eliminates floater Gaussians and recovers fine geometric details.

GT

<!-- image-->

2DGS

<!-- image-->

2DGS+PDEO

<!-- image-->

RaDeGS

<!-- image-->

RaDe-GSPDEO

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

Fig. 6. Qualitative comparisons of different methods on scenes from Tanks&Temples [Knapitsch et al. 2017] datasets for surface reconstruction. PEDO improves the quality of the reconstruction.  
<!-- image-->  
Full

<!-- image-->  
w/o P2G and G2P

<!-- image-->  
Full

<!-- image-->  
w/o Scale Loss

<!-- image-->  
Full

<!-- image-->  
w/o Confidence Loss

Fig. 7. Ablation of P2G and G2P, the Scale Loss and Confidence Loss

## A APPENDIX

## A.1 Gaussian Gradient Sensitivity Analysis

Attributes in 3DGS. For some attributes with restricted value ranges, 3DGS applies an activation function to map an unbounded attributes to a bounded value range. Below is a comparison of some attributes and their corresponding rendering properties used in 3DGS:

â¢ possion: $\pmb { \mu _ { i } } \in R ^ { 3 }$

â¢ color: $\hat { \mathbf { c } } _ { i , \phi } = f ( \phi , \mathbf { c } _ { i } ) ,$

â¢ opacity: $\hat { o } _ { i } = \operatorname { S i g } ( o _ { i } )$

â¢ scale: $\hat { \mathbf { s } } _ { i } = e ^ { ( \mathbf { s } _ { i } ) }$

â¢ rotation: $\mathbf q _ { i } \in R ^ { 4 }$

where $\pmb { \mu } _ { i }$ denotes the center position, $\mathbf { c } _ { i }$ represents the spherical harmonic coefficients, ???? is the opacity attribute, $\mathbf { \boldsymbol { s } } _ { i }$ refers to the scale attributes, and ${ \bf q } _ { i }$ is the quaternion representing the rotation attributes. Here, $\hat { o } _ { i } \ = \ \operatorname { S i g } ( o _ { i } )$ denotes the opacity, where Sig(Â·) represents the sigmoid function, and $\hat { \mathbf { s } } _ { i } = e ^ { ( \mathbf { s } _ { i } ) } \in R ^ { 3 }$ denotes the scaling vector.

<!-- image-->  
Fig. 8. Projection process of 3D Gaussians. Left. The view line of camera is orthogonal to the axis plane of 3D Gaussian.Right. The situation is the same for rotation Gaussians and rotation cameras, so we choose to rotate cameras. After a rotation R, the view line is not orthogonal to the plane.

3DGS to 2D splatting. For a simple non-rotated 3D Gaussian basis function:

$$
g _ { i } ( \pmb { \mu } ) = e ^ { - \frac { 1 } { 2 } ( \pmb { \mu } - \pmb { \mu } _ { i } ) ^ { T } \Sigma ^ { - 1 } ( \pmb { \mu } - \pmb { \mu } _ { i } ) } ,
$$

here $\pmb { \mu } = ( \mu _ { 1 } , \mu _ { 2 } , \mu _ { 3 } )$ is the sampling possition and $\pmb { \mu } _ { i } = ( \mu _ { i , 1 } , \mu _ { i , 2 } , \mu _ { i , 3 } )$ is the possition of the Gaussian ???? . If we integrate along one of the coordinate axes (1, 0, 0) throgh the point $\left( \mu _ { i , 1 } , \mu _ { 2 } , \mu _ { 3 } \right)$ and the corresponding pixel is $\pmb { u } = ( u _ { 1 } , u _ { 2 } )$ , To simplify, we let $x _ { i } = \mu - \mu _ { i } , ( x _ { i } =$ $( x _ { i , 1 } , x _ { i , 2 } , x _ { i , 3 } ) \}$ ). We obtain the following integral result:

$$
s p l a t _ { i } ( \boldsymbol { u } ) = s p l a t _ { i } ( u _ { 1 } , u _ { 2 } ) = \int g _ { i } ( \mu ) d \mu _ { 1 } = \int e ^ { - ( \frac { x _ { i , 1 } ^ { 2 } } { 2 \cdot \hat { s } _ { i , 1 } ^ { 2 } } + \frac { x _ { i , 2 } ^ { 2 } } { 2 \cdot \hat { s } _ { i , 2 } ^ { 2 } } + \frac { x _ { i , 3 } ^ { 2 } } { 2 \cdot \hat { s } _ { i , 3 } ^ { 2 } } ) } d \mu _ { 1 }
$$

$$
\mathbf { \Gamma } = e ^ { - ( \frac { x _ { i , 2 } ^ { 2 } } { 2 \cdot \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } } + \frac { x _ { i , 3 } ^ { 2 } } { 2 \cdot \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } } ) } \cdot \sqrt { 2 \pi } \cdot \hat { \mathbf { s } } _ { i , 1 } ,
$$

where ${ \boldsymbol { \Sigma } } ^ { - 1 }$ is a 3 Â· 3 metrix as:

$$
\left[ \begin{array} { c c c } { \frac { 1 } { 2 \hat { s } _ { i , 1 } ^ { 2 } } } & { 0 } & { 0 } \\ { 0 } & { \frac { 1 } { 2 \hat { s } _ { i , 2 } ^ { 2 } } } & { 0 } \\ { 0 } & { 0 } & { \frac { 1 } { 2 \hat { s } _ { i , 3 } ^ { 2 } } } \end{array} \right]
$$

This is the 2D splatting function at the pixel projected from point $( \mu _ { i , 1 } , \mu _ { 2 } , \mu _ { 3 } )$ in the absence of rotation.

And when we introduce a rotation matrix ??, the integral of the rotated function along a line passing through point $( \mu _ { i , 1 } , \mu _ { 2 } , \mu _ { 3 } )$ and parallel to the viewing direction is equivalent to the integral of the non-rotated function along a line passing through point $( \mu _ { i , 1 } , \mu _ { 2 } , \mu _ { 3 } )$ that has been rotated by ??.

Then We assume that the direction vector of the integration axis after rotation i ${ \mathrm { : } } r = ( r _ { 1 } , r _ { 2 } , r _ { 3 } )$ , where $r _ { 1 } ^ { 2 } + r _ { 2 } ^ { 2 } + r _ { 3 } ^ { 2 } = 1$ . So the integral result of the rotated function

$$
g _ { i } ( \pmb { \mu } ) = e ^ { - \pmb { x } _ { i } ^ { T } R ^ { T } \Sigma R \pmb { x } _ { i } } ,
$$

along (1, 0, 0), we integrate it

$$
s p l a t _ { i } ( \boldsymbol { u } ) = \int g _ { i } ( \boldsymbol { \mu } ) d \mu _ { 1 } = \int e ^ { - ( \frac { ( r _ { 1 } \cdot t ) ^ { 2 } } { 2 \cdot \hat { s } _ { 1 } ^ { 2 } } + \frac { ( r _ { 2 } \cdot t + x _ { i , 2 } ) ^ { 2 } } { 2 \cdot \hat { s } _ { 2 } ^ { 2 } } + \frac { ( r _ { 3 } \cdot t + x _ { i , 3 } ) ^ { 2 } } { 2 \cdot \hat { s } _ { 3 } ^ { 2 } } ) } d t ,
$$

we simplify it to

$$
e ^ { - ( \frac { x _ { i , 2 } ^ { 2 } } { 2 \cdot \hat { s } _ { 2 } ^ { 2 } } + \frac { x _ { i , 3 } ^ { 2 } } { 2 \cdot \hat { s } _ { 3 } ^ { 2 } } ) } \cdot \int e ^ { - ( \frac { r _ { 1 } ^ { 2 } \cdot t ^ { 2 } } { 2 \cdot \hat { s } _ { 1 } ^ { 2 } } + \frac { r _ { 2 } ^ { 2 } \cdot t ^ { 2 } } { 2 \cdot \hat { s } _ { 2 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } \cdot t ^ { 2 } } { 2 \cdot \hat { s } _ { 3 } ^ { 2 } } + \frac { r _ { 2 } \cdot t \cdot x _ { i , 2 } } { \hat { s } _ { 2 } ^ { 2 } } + \frac { r _ { 3 } \cdot t \cdot x _ { i , 3 } } { \hat { s } _ { 3 } ^ { 2 } } ) } d t ,
$$

so we introduce two coefficients $A = \frac { r _ { 1 } ^ { 2 } } { 2 { \cdot } \hat { \mathbf { s } } _ { 1 } ^ { 2 } } + \frac { r _ { 2 } ^ { 2 } } { 2 { \cdot } \hat { \mathbf { s } } _ { 2 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { 2 { \cdot } \hat { \mathbf { s } } _ { 3 } ^ { 2 } }$ , and $B ( x _ { i , 2 } , x _ { i , 3 } ) =$ $\begin{array} { r } { \frac { r _ { 2 } \cdot x _ { i , 2 } } { \hat { \mathbf { s } } _ { 2 } ^ { 2 } } + \frac { r _ { 3 } \cdot x _ { i , 3 } } { \hat { \mathbf { s } } _ { 3 } ^ { 2 } } ; } \end{array}$

$$
e ^ { - ( \frac { x _ { i , 2 } ^ { 2 } } { 2 \cdot \hat { s } _ { 2 } ^ { 2 } } + \frac { x _ { i , 3 } ^ { 2 } } { 2 \cdot \hat { s } _ { 3 } ^ { 2 } } ) + \frac { B ( x _ { i , 2 } , x _ { i , 3 } ) ^ { 2 } } { 4 \cdot A } } \cdot \int e ^ { - A ( t + \frac { B ( x _ { i , 2 } , x _ { i , 3 } ) } { 2 A } ) ^ { 2 } } d t ,
$$

so we can get

$$
s p l a t ( x _ { i } ) = s p l a t _ { i } ( u ) = e ^ { - ( \frac { x _ { i , 2 } ^ { 2 } } { 2 \cdot \hat { s } _ { 2 } ^ { 2 } } + \frac { x _ { i , 3 } ^ { 2 } } { 2 \cdot \hat { s } _ { 3 } ^ { 2 } } ) + \frac { B ( x _ { i , 2 } , x _ { i , 3 } ) ^ { 2 } } { 4 \cdot A } } \cdot \sqrt { \frac { \pi } { A } } ,
$$

as the splatting result of $g _ { i }$ at the pixel ??.

Renderring Gradient. For the energy term of rendering supervision, we can write it as:

$$
L = \sum _ { \boldsymbol { u } } ( r e n d e r ( \boldsymbol { u } ) - g t ( \boldsymbol { u } ) ) ^ { 2 } ,
$$

here $g t ( \cdot )$ is the ground truth of the view and the ???????????? (??) is the render function of Gaussian splatting which can be writen as:

$$
r e n d e r ( { \pmb u } ) = \sum _ { i } T _ { i } \hat { \bf c } _ { i } ( \mathrm { S i g } ( o _ { i } ) { \cdot } s p l a t ( { \pmb x } _ { i } ) ) .
$$

where $\hat { \mathbf { c } } _ { i }$ is color and $T _ { i } = \Pi _ { k = 1 } ^ { i - 1 } ( 1 - \alpha _ { k } )$ is transmittance of $g _ { i }$ , here $\alpha _ { k } = \mathrm { S i g } ( o _ { k } ) { \cdot } s p l a t ( x _ { k } )$ is opacity. We find

$$
\frac { \partial L } { \partial \gamma _ { i } } = \sum _ { \pmb { u } } 2 ( r e n d e r ( \pmb { u } ) - g t ( \pmb { u } ) ) \cdot \sum _ { k } \frac { \partial ( T _ { k } \hat { \mathbf { c } } _ { k } ( \mathrm { S i g } ( o _ { k } ) \ast s p l a t ( \pmb { x } _ { k } ) ) ) } { \partial \gamma _ { i } }
$$

here ?? is also the index of gaussians, and $\gamma _ { i } \in \{ \pmb { \mu } _ { i } ^ { t } , c _ { i } ^ { t } , o _ { i } ^ { t } , s _ { i } ^ { t } , q _ { i } ^ { t } \}$ is the attributes of ???? . So we can only discuss

$$
\frac { \partial ( T _ { k } \cdot \hat { \mathbf { c } } _ { k } \cdot \alpha _ { k } ) } { \partial \gamma _ { i } } = \frac { \partial ( T _ { k } \hat { \mathbf { c } } _ { k } ( \operatorname { S i g } ( o _ { k } ) * s p l a t ( { \pmb x } _ { k } ) ) ) } { \partial \gamma _ { i } } ,
$$

if we want compare the gradients of diffierent attributes.

So when $k = i ,$ we have:

$$
\begin{array} { r l } & { \frac { \partial \left( T _ { i } \cdot \hat { \mathbf { c } } _ { i } \cdot \alpha _ { i } \right) } { \partial \hat { \mathbf { c } } _ { i } } = T _ { i } \alpha _ { i } \frac { \partial \hat { \mathbf { c } } _ { i } } { \partial c _ { i } } , } \\ & { \frac { \partial \left( T _ { i } \cdot \hat { \mathbf { c } } _ { i } \cdot \alpha _ { i } \right) } { \partial o _ { i } } = T _ { i } \hat { \mathbf { c } } _ { i } ( 1 - \mathrm { S i g } ( o _ { i } ) ) \mathrm { S i g } ( o _ { i } ) s p l a t ( { \mathbf { x } } _ { i } ) , } \\ & { \quad \frac { \partial \left( T _ { i } \cdot \hat { \mathbf { c } } _ { i } \cdot \alpha _ { i } \right) } { \partial \mu _ { i , j } } = T _ { i } \hat { \mathbf { c } } _ { i } \mathrm { S i g } ( o _ { i } ) ( s p l a t ( { \mathbf { x } } _ { i } ) ) _ { \mu _ { i , j } } , } \\ & { \quad \frac { \partial \left( T _ { i } \cdot \hat { \mathbf { c } } _ { i } \cdot \alpha _ { i } \right) } { \partial s _ { i , j } } = T _ { i } \hat { \mathbf { c } } _ { i } \mathrm { S i g } ( o _ { i } ) ( s p l a t ( { \mathbf { x } } _ { i } ) ) _ { s _ { i , j } } , } \end{array}
$$

here $( \cdot ) _ { \gamma }$ denotes the partial derivative. And when ?? is different from ??:

$$
\begin{array} { r l } & { \quad \frac { \partial \left( T _ { k } \cdot \hat { \mathbf { c } } _ { k } \cdot \alpha _ { k } \right) } { \partial c _ { i } } = 0 , } \\ & { \quad \frac { \partial \left( T _ { k } \cdot \hat { \mathbf { c } } _ { k } \cdot \alpha _ { k } \right) } { \partial o _ { i } } = - \frac { T _ { k } \hat { \mathbf { c } } _ { k } \alpha _ { k } \left( 1 - \mathrm { S i g } \left( o _ { i } \right) \right) \mathrm { S i g } \left( o _ { i } \right) s p l a t \left( x _ { i } \right) } { 1 - \alpha _ { i } } , } \\ & { \quad \frac { \partial \left( T _ { k } \cdot \hat { \mathbf { c } } _ { k } \cdot \alpha _ { k } \right) } { \partial \mu _ { i , j } } = - \frac { T _ { k } \hat { \mathbf { c } } _ { k } \alpha _ { k } \mathrm { S i g } \left( o _ { i } \right) \left( s p l a t \left( x _ { i } \right) \right) \mu _ { i , j } } { 1 - \alpha _ { i } } , } \\ & { \quad \frac { \partial \left( T _ { k } \cdot \hat { \mathbf { c } } _ { k } \cdot \alpha _ { k } \right) } { \partial s _ { i , j } } = - \frac { T _ { k } \hat { \mathbf { c } } _ { k } \alpha _ { k } \mathrm { S i g } \left( o _ { i } \right) \left( s p l a t \left( x _ { i } \right) \right) _ { s , j } } { 1 - \alpha _ { i } } , } \end{array}
$$

where

$$
s p l a t ( \pmb { x _ { i } } ) = e ^ { - ( \frac { ( \mu _ { i , 2 } - \mu _ { 2 } ) ^ { 2 } } { 2 \cdot \hat { \pmb { s } } _ { i , 2 } ^ { 2 } } + \frac { ( \mu _ { i , 3 } - \mu _ { 3 } ) ^ { 2 } } { 2 \cdot \hat { \pmb { s } } _ { i , 3 } ^ { 2 } } ) + \frac { B ( \mu _ { i , 2 } - \mu _ { 2 } , \mu _ { i , 3 } - \mu _ { 3 } ) ^ { 2 } } { 4 \cdot A } } \cdot \sqrt { \frac { \pi } { A } } .
$$

So if we ignore $( s p l a t ( \pmb { x } _ { i } ) ) _ { \mu _ { i , j } }$ and $( s p l a t ( \pmb { x } _ { i } ) ) _ { s _ { i , j } }$ , we can find the remaining parts of the items in the same group are of the same magnitude. For $\alpha _ { i } { \sim } \mathrm { S i g } ( o _ { i } )$ and $\begin{array} { r } { \hat { \mathbf { c } } _ { i } \sim \frac { \partial \hat { \mathbf { c } } _ { i } } { \hat { \mathbf { c } } _ { i } } \sim \alpha _ { i } \sim \left( 1 - \mathrm { S i g } ( o _ { i } ) \right) \sim s p l a t e ( { \boldsymbol x } _ { i } ) \sim 1 . } \end{array}$ So we can only judge $( s p l a t ( \pmb { x } _ { i } ) ) _ { \mu _ { i , j } }$ and $( s p l a t ( \pmb { x } _ { i } ) ) _ { s _ { i , j } }$ to compare the gradients.

Let $j = 2 ,$ , then we can get

$$
( s p l a t _ { i } ) _ { \mu _ { i , 2 } } = \frac { 1 } { \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } } s p l a t _ { i } \cdot ( \frac { \frac { r _ { 2 } r _ { 3 } ( \mu _ { i , 3 } - \mu _ { 3 } ) } { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } } - ( \frac { r _ { 1 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } } ) ( \mu _ { i , 2 } - \mu _ { 2 } ) } { \frac { r _ { 1 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } } + \frac { r _ { 2 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } } } ) ,
$$

here ???????????? = ???????????? (????). Obviously, we have $s p l a t _ { i } { \sim } 1 , ~ ( \mu _ { i , 3 } ~ -$ $\mu _ { 3 } ) { \sim } \hat { \mathbf { s } } _ { i , 3 }$ and $a x ^ { 2 } + b y ^ { 2 } \geq 2 \sqrt$ ????????, so we have

$$
\frac { \frac { r _ { 2 } r _ { 3 } ( \mu _ { i , 3 } - \mu _ { 3 } ) } { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } } } { \frac { r _ { 1 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } } + \frac { r _ { 2 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } } } \sim \frac { r _ { 2 } r _ { 3 } } { \frac { \hat { \mathbf { s } } _ { i , 3 } r _ { 1 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } } + \frac { \hat { \mathbf { s } } _ { i , 3 } r _ { 2 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 3 } } } \le \frac { r _ { 2 } r _ { 3 } } { \frac { 2 r _ { 2 } r _ { 3 } } { \hat { \mathbf { s } } _ { i , 2 } } } \sim \hat { \mathbf { s } } _ { i , 2 } ,
$$

and we have $( \mu _ { i , 2 } - \mu _ { 2 } ) { \sim } \hat { \mathbf { s } } _ { i , 2 }$ , so

$$
\frac { ( \frac { r _ { 1 } ^ { 2 } } { \hat { \bf s } _ { i , 1 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { \bf s } _ { i , 3 } ^ { 2 } } ) ( \mu _ { i , 2 } - \mu _ { 2 } ) } { \frac { r _ { 1 } ^ { 2 } } { \hat { \bf s } _ { i , 1 } ^ { 2 } } + \frac { r _ { 2 } ^ { 2 } } { \hat { \bf s } _ { i , 2 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { \bf s } _ { i , 3 } ^ { 2 } } } \leq ( \mu _ { i , 2 } - \mu _ { 2 } ) { \sim } \hat { \bf s } _ { i , 2 } .
$$

According to the definition of equivalence we can get $( s p l a t _ { i } ) _ { \mu _ { i , 2 } } \lesssim$ $\frac { 1 } { \hat { \mathsf { s } } _ { i , 2 } }$ , and when $r _ { 2 } ~ = ~ 0 , ~ ( s p l a t _ { i } ) _ { \mu _ { i , 2 } } { \sim } \frac { 1 } { \hat { \ s } _ { i . 2 } }$ . So $\begin{array} { r } { ( s p l a t _ { i } ) _ { \mu i , 2 } { \sim } \frac { 1 } { \hat { \mathbf { s } } _ { i , 2 } } } \end{array}$ . And similarly at $j = 3 ,$ we have $\begin{array} { r } { ( s p l a t _ { i } ) _ { \mu i , 3 } { \sim } \frac { 1 } { \hat { \mathbf { s } } _ { i , 3 } } } \end{array}$

Similarly, we also handle $( s p l a t _ { i } ) _ { s _ { i , j } }$ , as before, we only need to take $j = 2 ,$ , since the other value of ?? is the same as $j = 2 .$ . Noting that $\hat { \mathbf { s } } _ { i } = e ^ { s _ { i } }$ and $( \hat { \bf s } _ { i } ) _ { s _ { i } } = \hat { \bf s } _ { i }$ .

$$
\begin{array} { c } { { ( s p l a t _ { i } ) _ { s _ { i , 2 } } = - \frac { ( \frac { r _ { 1 } ^ { 2 } } { \hat { s } _ { i , 1 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { s } _ { i , 3 } ^ { 2 } } ) r _ { 2 } ^ { 2 } ( \mu _ { i , 2 } - \mu _ { 2 } ) ^ { 2 } } { 2 A ^ { 2 } \hat { s } _ { i , 2 } ^ { 4 } } + \frac { ( \frac { r _ { 1 } ^ { 2 } } { \hat { s } _ { i , 1 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { s } _ { i , 3 } ^ { 2 } } ) ( \mu _ { i , 2 } - \mu _ { 2 } ) ^ { 2 } } } } \\ { { - \frac { 2 r _ { 2 } r _ { 3 } ( \mu _ { i , 2 } - \mu _ { 2 } ) ( \mu _ { i , 3 } - \mu _ { 3 } ) ( \hat { \frac { s _ { i , 3 } ^ { 2 } { r _ { 1 } ^ { 2 } } } { \hat { s } _ { i , 1 } ^ { 2 } } + r _ { 3 } ^ { 2 } ) } } { \hat { s } _ { i , 2 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 4 } A ^ { 2 } } . } } \end{array}
$$

We analyze each item step by step. For $( \mu _ { i , 2 } - \mu _ { 2 } ) { \sim } \hat { \mathbf { s } } _ { i , 2 } .$

$$
\frac { ( \frac { r _ { 1 } ^ { 2 } } { \hat { \bf s } _ { i , 1 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { \bf s } _ { i , 3 } ^ { 2 } } ) r _ { 2 } ^ { 2 } ( \mu _ { i , 2 } - \mu _ { 2 } ) ^ { 2 } } { 2 A ^ { 2 } \hat { \bf s } _ { i , 2 } ^ { 4 } } \lesssim \frac { r _ { 2 } ^ { 2 } } { 2 A \hat { \bf s } _ { i , 2 } ^ { 2 } } \sim 1 ,
$$

Similarly, we obtain:

$$
\frac { ( \frac { r _ { 1 } ^ { 2 } } { \hat { s } _ { i , 1 } ^ { 2 } } + \frac { r _ { 3 } ^ { 2 } } { \hat { s } _ { i , 3 } ^ { 2 } } ) ( \mu _ { i , 2 } - \mu _ { 2 } ) ^ { 2 } } { A \hat { s } _ { i , 2 } ^ { 2 } } { \sim } 1
$$

The third item is slightly more complex, so we will handle it in two parts. Firstly, We will address the first part:

$$
E _ { 1 } : = \frac { 2 r _ { 1 } ^ { 2 } r _ { 2 } r _ { 3 } ( \mu _ { i , 2 } - \mu _ { 2 } ) ( \mu _ { i , 3 } - \mu _ { 3 } ) } { \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } A ^ { 2 } } .
$$

Note $\begin{array} { r } { E _ { 1 } = \frac { 2 \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } r _ { 1 } ^ { 2 } \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } r _ { 2 } r _ { 3 } \left( \mu _ { i , 2 } - \mu _ { 2 } \right) ( \mu _ { i , 3 } - \mu _ { 3 } ) } { ( \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } r _ { 1 } ^ { 2 } + \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } + \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } r _ { 3 } ^ { 2 } ) ^ { 2 } } } \end{array}$ , so a natural thinking is dividing it into two parts:

$$
\begin{array} { r } { E _ { 1 } = \frac { 2 \hat { s } _ { i , 2 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 1 } ^ { 2 } } { \hat { s } _ { i , 2 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 1 } ^ { 2 } + \hat { s } _ { i , 1 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } + \hat { s } _ { i , 1 } ^ { 2 } \hat { s } _ { i , 2 } ^ { 2 } r _ { 3 } ^ { 2 } } } \\ { \cdot \frac { \hat { s } _ { i , 1 } ^ { 2 } r _ { 2 } r _ { 3 } ( \mu _ { i , 2 } - \mu _ { 2 } ) ( \mu _ { i , 3 } - \mu _ { 3 } ) } { \hat { s } _ { i , 2 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 1 } ^ { 2 } + \hat { s } _ { i , 1 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } + \hat { s } _ { i , 1 } ^ { 2 } \hat { s } _ { i , 2 } ^ { 2 } r _ { 3 } ^ { 2 } } . } \end{array}
$$

For (????,2 â ??2)â¼sË??,2 and (????,3 â ??3)â¼sË??,3, we have:

$$
\begin{array} { r l } & { \frac { 2 \hat { s } _ { i , 2 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 1 } ^ { 2 } } { \hat { s } _ { i , 2 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 1 } ^ { 2 } + \hat { s } _ { i , 1 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } + \hat { s } _ { i , 1 } ^ { 2 } \hat { s } _ { i , 2 } ^ { 2 } r _ { 3 } ^ { 2 } } \lesssim 1 , } \\ & { \frac { \hat { s } _ { i , 1 } ^ { 2 } r _ { 2 } r _ { 3 } \left( \mu _ { i , 2 } - \mu _ { 2 } \right) \left( \mu _ { i , 3 } - \mu _ { 3 } \right) } { \hat { s } _ { i , 2 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 1 } ^ { 2 } + \hat { s } _ { i , 1 } ^ { 2 } \hat { s } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } + \hat { s } _ { i , 1 } ^ { 2 } \hat { s } _ { i , 2 } ^ { 2 } r _ { 3 } ^ { 2 } } \lesssim 1 . } \end{array}
$$

Similarly, we address the second part:

$$
E _ { 2 } : = \frac { 2 r _ { 2 } r _ { 3 } ^ { 3 } ( \mu _ { i , 2 } - \mu _ { 2 } ) ( \mu _ { i , 3 } - \mu _ { 3 } ) } { \hat { \mathbf { s } } _ { i , 2 } ^ { 2 } \hat { \mathbf { s } } _ { i , 3 } ^ { 4 } A ^ { 2 } } .
$$

Note $\begin{array} { r } { E _ { 2 } = \frac { 2 r _ { 2 } r _ { 3 } ( \mu _ { i , 3 } - \mu _ { 3 } ) r _ { 3 } ^ { 2 } ( \mu _ { i , 2 } - \mu _ { 2 } ) } { ( \frac { \hat { s } _ { i , 3 } ^ { 2 } \hat { s } _ { i , 2 } r _ { 1 } ^ { 2 } } { \hat { s } _ { i , 1 } ^ { 2 } } + \frac { \hat { s } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } } { \hat { s } _ { i , 2 } } + \hat { s } _ { i , 2 } r _ { 3 } ^ { 2 } ) ^ { 2 } } } \end{array}$ , so we divide it into two parts:

$$
E _ { 2 } = \frac { 2 r _ { 2 } r _ { 3 } \left( \mu _ { i , 3 } - \mu _ { 3 } \right) } { \frac { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } \hat { \mathbf { s } } _ { i , 2 } r _ { 1 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } } + \frac { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 2 } } + \hat { \mathbf { s } } _ { i , 2 } r _ { 3 } ^ { 2 } } \cdot \frac { r _ { 3 } ^ { 2 } \left( \mu _ { i , 2 } - \mu _ { 2 } \right) } { \frac { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } \hat { \mathbf { s } } _ { i , 2 } r _ { 1 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 1 } ^ { 2 } } + \frac { \hat { \mathbf { s } } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } } { \hat { \mathbf { s } } _ { i , 2 } } + \hat { \mathbf { s } } _ { i , 2 } r _ { 3 } ^ { 2 } } .
$$

Then we have:

$$
\frac { 2 r _ { 2 } r _ { 3 } ( \mu _ { i , 3 } - x _ { 3 0 } ) } { \frac { \hat { s } _ { i , 3 } ^ { 2 } \hat { s } _ { i , 2 } r _ { 1 } ^ { 2 } } { \hat { s } _ { i , 1 } ^ { 2 } } + \frac { \hat { s } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } } { \hat { s } _ { i , 2 } } + \hat { s } _ { i , 2 } r _ { 3 } ^ { 2 } } \lesssim 1 ,
$$

ACM Trans. Graph., Vol. 1, No. 1, Article . Publication date: September 2025.

$$
\frac { r _ { 3 } ^ { 2 } ( \mu _ { i , 2 } - x _ { 2 0 } ) } { \frac { \hat { s } _ { i , 3 } ^ { 2 } \hat { s } _ { i , 2 } r _ { 1 } ^ { 2 } } { \hat { s } _ { i , 1 } ^ { 2 } } + \frac { \hat { s } _ { i , 3 } ^ { 2 } r _ { 2 } ^ { 2 } } { \hat { s } _ { i , 2 } } + \hat { s } _ { i , 2 } r _ { 3 } ^ { 2 } } \lesssim 1 .
$$

So we have $( s p l a t _ { i } ) _ { s _ { i , 2 } } \lesssim$ 1, and when $r _ { 2 } = r _ { 3 } = 0 , ( s p l a t _ { i } ) _ { s _ { i , 2 } } \sim 1$ According to the definition of equivalence, $( s p l a t _ { i } ) _ { s _ { i , 2 } } \sim 1 .$ . And the same as other value of ??.

So we can find in certain ?? and $k ,$ we have the realation that:

$$
\hat { \bf s } _ { i , j } \frac { \partial ( T _ { k } \hat { \bf c } _ { k } \alpha _ { k } ) } { \partial \mu _ { i , j } } \sim \frac { \partial ( T _ { k } \hat { \bf c } _ { k } \alpha _ { k } ) } { \partial c _ { i } } \sim \frac { \partial ( T _ { k } \hat { \bf c } _ { k } \alpha _ { k } ) } { \partial o _ { i } } \sim \frac { \partial ( T _ { k } \hat { \bf c } _ { k } \alpha _ { k } ) } { \partial s _ { i , j } } .
$$

Specially, the rotation attribute $q _ { i } ^ { t }$ which is a quaternion array is updating as:

$$
\begin{array} { c } { q _ { i } ^ { t + 1 } = q _ { i } ^ { t } + \triangle q _ { i } ^ { t } , } \\ { \| \triangle q _ { i } ^ { t } \| = \| \displaystyle \frac { \frac { \partial L } { \partial q _ { i } } + q _ { i } ^ { t } } { \| \frac { \partial L } { \partial q _ { i } } + q _ { i } ^ { t } \| } - q _ { i } ^ { t } \| \leq 2 \operatorname* { m a x } ( \| q _ { i } \| ) } \end{array}
$$

By the definition of a quaternionic array we have $\| q _ { i } \| \leq 1$ , then we obtain $\triangle q _ { i } ^ { t } \sim 1$ . So we can get

$$
\hat { \bf s } _ { i } \frac { \partial L } { \partial \pmb { \mu } _ { i } } \sim \frac { \partial L } { \partial c _ { i } } \sim \frac { \partial L } { \partial o _ { i } } \sim \frac { \partial L } { \partial s _ { i } } \sim \Delta q _ { i } ^ { t } ,
$$

so if we define the direction vector of $\triangle q _ { i } ^ { t }$ as $r _ { q , i } ^ { t } .$ by the definition of partial derivatives, the updating of rotation attribute have

$$
\triangle q _ { i } ^ { t } = \frac { \partial L } { \partial ( q _ { i } ^ { t } { \cdot } r _ { q , i } ^ { t } ) } .
$$

When the scales of Gaussians are small, we can get

$$
\frac { \partial L } { \partial \pmb { \mu _ { i } } } \gg \frac { \partial L } { \partial c _ { i } } \sim \frac { \partial L } { \partial o _ { i } } \sim \frac { \partial L } { \partial s _ { i } } \sim \frac { \partial L } { \partial ( q _ { i } ^ { t } \cdot r _ { q , i } ^ { t } ) } .
$$

That means the Gaussians will more willing to change their places to reduce the energy, which will more likely cause the large-scale random drift and leading the local minimum. To achieve optimal results, we aim for all variables to change in a relatively consistent manner. To this end, it is natural to consider decelerating the changes in the positional attributes of the 3D Gaussians. Specifically, we formulate the 3DGS optimization procedure as the discretization of a Partial Differential Equation (PDE) and employ the viscosity coefficient, allowing spatial positions to absorb and gradually release the positional gradients of the 3D Gaussians.

And due to the Gaussian function property, we have

$$
\sum _ { s p l a t _ { k } \geq \epsilon } s p l a t _ { k } \sim 1 ,
$$

where ?? is the 0.99 confidence bound for the Gaussian function. So for the same 3D Gaussian $g _ { k }$ at different scales $\hat { \mathbf { s } } _ { k }$ , we have

$$
\sum _ { s p l a t _ { k } \geq \epsilon } \frac { \partial ( T _ { k } \cdot \hat { \mathbf { c } } _ { k } \cdot \alpha _ { k } ) } { \partial \mu _ { k } } = \sum _ { s p l a t _ { k } \geq \epsilon } O ( \frac { 1 } { \hat { \mathbf { s } } _ { k } } ) T _ { k } \hat { \mathbf { c } } _ { k } \operatorname { S i g } ( o _ { k } ) s p l a t _ { k } = O ( \frac { 1 } { \hat { \mathbf { s } } _ { k } } ) ,
$$

and the position gradient $\frac { \partial L } { \partial \mu _ { i , j } } \mathrm { o f } g _ { k }$ is proportional to $\begin{array} { r } { \sum \frac { \partial ( T _ { k } \cdot \hat { \mathbf { c } } _ { k } \cdot \boldsymbol { \alpha } _ { k } ) } { \partial \mu _ { k } } } \end{array}$ so we have the relationship of position gradients between diffierent 3D Gaussians:

$$
\hat { \bf s } _ { i , j } \frac { \partial L } { \partial \mu _ { i , j } } \sim \hat { \bf s } _ { k , j } \frac { \partial L } { \partial \mu _ { k , j } } .
$$

Observation. 3DGS represents a complex scene as a set of 3D Gaussians. However, various 3DGS methods [Geiger et al. 2024;

Yu et al. 2024a] suffer from the common limitation of blurring and floaters due to the reconstruction of redundant and ambiguous geometric structures, leading to degraded rendering and reconstruction quality. We attribute the blurring and floaters to the occlusion of redundant large Gaussians and the ambiguity of small Gaussians, as shown in Fig. 2. The large 3D Gaussians fail to capture highfrequency details and tend to obstruct other Gaussians, resulting in redundancy and manifesting as blurring in the novel view. For small 3D Gaussians, due to the unstable gradient, floaters tend to appear in regions of the scene that are poorly observed, as the Gaussians tend to shift their positions toward observed views during the 3DGS optimization process, thereby resulting in ambiguous geometric structures.

## A.2 The The velocity Voxel in Space

Building. We aim to construct a loss function that considers the positional gradient field for a particle located at spatial position ??:

$$
v ( \pmb { \mu } ) = \sigma \frac { \partial L } { \partial \pmb { \mu } } .
$$

However, since the attributes of the particles are unknown, this term cannot be directly calculated. Moreover, as the motion equations in 3DGS are based on the gradients of the scene rendering results, particles with different color attributes will exhibit different movement tendencies, typically lacking a linear relationship. Therefore, simply averaging attributes of the particles near a spatial location and then using this averaged set of attributes to compute the positional gradient is meaningless.

We aim for the velocity field at $\pmb { \mu }$ to indicate the most likely displacement of a Gaussian sphere at this location. Therefore, we choose to construct and update the velocity field by the local average velocity of $\mu ,$ as shown in $\operatorname { E q } . 3 1$

This approach is mathematically meaningful: if we consider the positional gradients of 3D Gaussians as points in a three-dimensional space, the positional gradients of 3D Gaussians near ?? form a point cloud in this velocity feild. We want ?? (??) to be positioned at the center of the largest cluster within this point cloud, which the arithmetic mean can achieve. Additionally, the arithmetic mean can counterbalance the impact of large-scale Brownian motion on the spatial velocity field caused by abrupt changes in positional gradients.Then we obtain the spatial velocity field $v ( \mu )$

Total Impact of Gradient Field. We renew the field by $\triangle v _ { n } ^ { t } =$ $\textstyle { \frac { 1 } { | R _ { n } ^ { t } | } } \sum _ { g _ { i } \in R _ { n } ^ { t } }$ â³?????? in Eq.31. So we have the total impact of $\triangle v _ { n } ^ { t }$ in the field by adding it in every steps:

$$
I ( \triangle v _ { n } ^ { t } ) = \sum _ { l \geq t } \triangle v _ { n } ^ { t } ( l ) ,
$$

where $\triangle v _ { n } ^ { t } ( l )$ means portion of $\upsilon _ { n } ^ { t } ( l )$ occupied by $\triangle v _ { n } ^ { t }$ . So we have

$$
I ( \triangle v _ { n } ^ { t } ) = ( 1 - \lambda _ { g } \triangle v _ { n } ^ { t } \sum _ { l \geq t } \lambda _ { g } ^ { ( l - t + 1 ) } \to \triangle v _ { n } ^ { t } .
$$

Therefore, regardless of the coefficient $\lambda _ { g } ,$ each updated vector will have a weight of 1 in the overall influence on the field throughout spacetime. At the same moment, the total weight of this vector on the gradient is always 1. Thus, no matter the chosen weighting

coefficient, the value of this velocity field can naturally represent Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009 the magnitude of the gradient.