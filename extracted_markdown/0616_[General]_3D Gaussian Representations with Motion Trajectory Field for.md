# 3D Gaussian Representations with Motion Trajectory Field for Dynamic Scene Reconstruction

Xuesong Li1,2, Lars Petersson1, Vivien Rolland1

1CSIRO 2The Australian National University

xuesong.li@csiro.au

## Abstract

This paper addresses the challenge of novel-view synthesis and motion reconstruction of dynamic scenes from monocular video, which is critical for many robotic applications. Although Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have demonstrated remarkable success in rendering static scenes, extending them to reconstruct dynamic scenes remains challenging. In this work, we introduce a novel approach that combines 3DGS with a motion trajectory field, enabling precise handling of complex object motions and achieving physically plausible motion trajectories. By decoupling dynamic objects from static background, our method compactly optimizes the motion trajectory field. The approach incorporates time-invariant motion coefficients and shared motion trajectory bases to capture intricate motion patterns while minimizing optimization complexity. Extensive experiments demonstrate that our approach achieves state-of-the-art results in both novel-view synthesis and motion trajectory recovery from monocular video, advancing the capabilities of dynamic scene reconstruction1.

## 1. Introduction

The novel-view synthesis and motion reconstruction of dynamic scenes from a monocular video is crucial for various applications such as robot perception and virtual reality [20, 21, 23, 28, 28, 38, 42]. With the great success of NeRF in rendering novel-view for static scenes by using implicit representation, many methods [4, 5, 7, 30, 31, 45] have extended NeRF to model dynamic scenes and have shown a promising visual quality. However, it is challenging for NeRF-based approaches to achieve real-time rendering due to a large number of sampling points and network forward computation for each sampled point alone each light ray.

Recently, 3DGS [13] provides an explicit representation with anisotropic 3D Gaussians for the static scene, and its representation and tile-based rasterization allows fast training and real-time rendering. Different approaches [25, 28, 48] extend this representation for modelling dynamic scenes. The straightforward approach is to optimize perframe 3DGS [28], which can recover the physical trajectories of Gaussian points. Still, it requires multi-view images and large storage memory and cannot be generalized to monocular video inputs. Other approaches [43, 48] decouple dynamic scenes into a set of 3DGS in canonical space and an implicit neural field (MLPs) for modelling dynamic deformation. However, they usually require computationally expensive forward passes of the neural network, lowering the rendering speed of the original 3DGS, besides, the neural deformation field models deformation independently for each timestamp without considering temporal consistency and motion prior, these implicit representations fail to capture the intricate details of objects in the scene and cannot generate physically plausible motion trajectories.

The reconstruction of dynamic scenes for both novelview synthesis and underlying motion from a monocular video is a challenging and under-constrained optimization problem. To tackle it, we assume that the underlying motion of the dynamic foreground shares several non-rigid motion trajectory bases [1, 23, 40], and then propose to model a dynamic scene with a motion trajectory field in 3GS representation. The motion trajectory field, modelling dynamic scenes with a compact and regular motion representation, can handle complex object motions precisely and produces high-quality renderings with physically plausible motion awareness [23, 40]. We model the dynamic scene learnable motion trajectory basis and lightweight multi-layer perceptron (MLP) for motion coefficient, which allows fast rendering. Sharing global motion basis across Gaussians encourages neighbouring points to have similar motion. The timeinvariant motion coefficients and the small number of motion trajectory bases greatly reduce the optimization complexity, which helps to capture the underlying intricate motion pattern. To better optimize the motion trajectory field and reduce the number of Gaussian primitives, we decouple the dynamic scenes into dynamic objects and static backgrounds, so that the motion trajectory field can be modelled compactly without requiring too many 3DGS points, as it has been observed that model entire scenes with a deformation field has the issue of over-consuming GPU memory cased by over-densification. Our representation can provide an easy way to do regularization on motion trajectory and reduce the complexity of this optimization problem. The experimental results show that our work can achieve high-fidelity novel-view synthesis and meanwhile, recover the underlying motion trajectories from a monocular video with real-time rendering. In summary, our main contributions include threefold: 1). we propose to combine motion trajectory field and 3DGS for dynamic scene reconstruction; 2). we design the specialized regularization terms for static/dynamic segmentation and motion trajectory recovery; 3). we have conducted extensive quantitative and qualitative experiments to evaluate the performance of our method.

## 2. Related work

## 2.1. Dynamic view synthesis

View synthesis for dynamic scenes has become a significant research focus in 3D vision, particularly with the development of NeRF [29]. NeRF-based methods have demonstrated exceptional results in novel view synthesis by implicitly modelling 3D scenes. Extensions [9, 19, 22, 27, 35, 38] of NeRF to dynamic scenes have utilized timeconditioned latent codes and explicit deformation fields to capture temporal variations. These approaches aim to model scene motion and appearance changes, though the high computational costs and slow rendering times have limited their practical use. Methods such as hash encoding [30, 41], explicit voxel grid [6, 8] and feature grid planes [4, 7, 37] have accelerated training and improved dynamic scene handling. However, real-time rendering remains challenging for NeRF-based methods due to the computationally expensive volume rendering process. Recently, 3DGS [13] has emerged as an efficient alternative for dynamic scene reconstruction and view synthesis. Unlike NeRF, 3DGS leverages explicit 3D Gaussian representations combined with differentiable splatting, allowing for real-time rendering without the need for expensive volume rendering. Various methods [11, 16, 28, 47, 48] have extended 3DGS to dynamic scenes. [28] pioneered the dynamic 3DGS by iteratively optimizing Gaussian parameters frame-by-frame. [11, 43, 48] employs a deformation field to model Gaussian transformations across time. These approaches effectively reduce memory consumption and improve efficiency, but challenges like motion ambiguities and maintaining temporal consistency persist.

## 2.2. Deformation recovery from dynamic scenes

Recovering deforming 3D shapes from a monocular video is important for scene understanding [3]. The 2D point correspondences across temporal frames are usually required for modelling deformation [2, 18, 32, 39, 51]. [2] proposes a trajectory space for the motion of 3D points and assumes that trajectories of points are a composite of a small number of discrete cosine transform (DCT) trajectory bases, and its variants include convolutional trajectory structure [18] and trajectory subspace [18]. With the success of NeRF, [40] has integrated trajectory space into NeRF by representing the motion of 3D sampled points with DCT trajectory bases for modelling dynamic scenes. To model long videos, [23] introduces learnable motion trajectory basis functions to extend its modelling capabilities. However, these implicit representations still require a long training time and suffer from slow inference. To model the deformation with explicit 3D Gaussian representation, [28] optimize a set of Gaussian primitives at the first frame and obtain their trajectories across all frames through per-frame optimization, but this approach requires multi-view images to provide a well-constrained optimization process. Gaussian-flow [25] models the motion trajectory of Gaussian points in both the time and frequency domains and the time domain. [10] introduces the optical flow to enhance the motion modelling of dynamic scenes. Shape-of-motion [42] utilizes depth maps and 2D tracks for constraining the motion of Gaussian primitives. However, these methods usually require strong data-driven priors, which could be noisy and degenerate the performance. In this work, we propose to use learnable motion trajectory bases to model the motion of 3DGS, which can not only achieve real-time rendering, but also recover of underlying motion trajectory of 3D Gaussian primitives.

## 3. Methodology

## 3.1. Preliminary: 3D points in trajectory space

All deformable 3D points can be represented into low-rank trajectory space [1]. To represent the structure at a specific time t, we organize the 3D coordinates of the points into a matrix $\pmb { x } ^ { t } \in \mathbb { R } ^ { 3 \times P } , \ \pmb { x } ^ { t } \ = \ \left\lceil x _ { 1 } ^ { t } \quad x _ { 2 } ^ { t } \quad \cdots \quad x _ { P } ^ { t } \right\rceil$ , where $x _ { t i } ~ \in ~ \mathbb { R } ^ { 3 }$ denotes the 3D coordinates of point i at time t. The complete time-varying structure is represented by concatenating these deformable structures into $X = \left\lceil \pmb { x } ^ { 1 } \pmb { x } ^ { 2 } \dots \pmb { x } ^ { N } \right\rceil ^ { T } \in \mathbb { R } ^ { ( N \times 3 ) \times P }$ . The row space of this matrix represents the shape space [44], while the column space is referred to as the trajectory space. For the shape and motion recovery, the structures are usually effectively represented using $k ( k \ll P )$ basis shapes [39], since the trajectory space holds a dual relationship with the shape space, the trajectory space can be equivalently represented with other k basis trajectory vectors. To express the time-varying structure using trajectory bases, we view the deformation as a set of trajectories $T _ { i } ~ \in ~ \mathbb { R } ^ { N \times 3 }$ , i.e. $T _ { i } = \left[ x _ { i } ^ { 1 } x _ { i } ^ { 2 } \cdots x _ { i } ^ { N } \right] ^ { T }$

<!-- image-->  
Figure 1. The pipeline of our framework. 3DGS at time t are transformed from global 3DGS using learnable motion bases $( \theta , \lambda , \eta )$ and trajectory coefficients $( \sigma , \beta , \gamma )$ . Projection, density control, and Gaussian rasterization follow original paper [13].

Each trajectory Ti can be described as a linear combination of basis trajectories: $\begin{array} { r } { T _ { i } = \sum _ { j = 1 } ^ { k } a _ { i j } \Theta _ { j } } \end{array}$ , where $\Theta _ { j } \in$ $\mathbb { R } ^ { N \times 3 }$ is a basis trajectory, and $a _ { i j }$ are the coefficients corresponding to that basis vector. The time-varying structure matrix X can then be factorized into an inverse projection matrix $\Theta \in \mathbb { R } ^ { ( N \times 3 ) \times k }$ and a coefficient matrix $\overset { \cdot } { A } \in \overset { \cdot } { \mathbb { R } } ^ { k \times P }$ i.e. $X \ = \ \Theta A .$ , where $A = \Big \lceil \pmb { a } _ { 1 } \quad \pmb { a } _ { 2 } \quad \cdot \cdot \cdot \quad \pmb { a } _ { P } \Big \rceil , \pmb { a } _ { i } =$ $\left[ a _ { i 1 } \quad a _ { i 2 } \quad \cdots \quad a _ { i k } \right] ^ { T } , \quad \Theta = \left[ \Theta _ { 1 } \quad \Theta _ { 2 } \quad \cdots \quad \Theta _ { k } \right]$

The principal benefit of the trajectory space representation is that a basis can be pre-defined and can compactly approximate most real trajectories. Various bases such as the Hadamard Transform, Discrete Fourier Transform, and Discrete Wavelet Transform [24, 40] can all represent trajectories in an object-independent manner. This prior information will relieve the burden of optimizing the motion of 3D points in a dynamic scene. This paper uses the DCT basis to initialize all basis trajectories, allowing optimization with a good initialization.

## 3.2. 3D Gaussians in trajectory space

Given a sequence of images from a dynamic scene with frames $( I _ { 1 } , I _ { 2 } , . . . , I _ { N } )$ , and known camera parameters, our method aims to synthesize the novel-view image at any time point to recover the geometric dynamic, such as point trajectory.

3D Guassians [13] was originally designed to represent static scenes. To extend its capabilities for dynamic scenes, we model the dynamics using a motion trajectory field. We assume a set of compact and moving 3D Gaussian points represents the dynamic scene, where each point follows a trajectory across all time frames. Each 3D Gaussian point has a time-varying covariance matrix but a time-invariant color coefficient (i.e., spherical harmonics). For the timevarying structure, we define a global reference point $x _ { i } ^ { \star } \in$ $\mathbb { R } ^ { 3 }$ for each trajectory $T _ { i }$ , which represents the global position of the point throughout the entire sequence. The global reference point $\boldsymbol { x } _ { i } ^ { \star }$ can be the initial position or a point in the middle of the trajectory $T _ { i } ,$ differing from points in canonical space [26, 35, 48]. There are several reasons for defining the global reference point $x _ { i } ^ { \star } \colon 1 )$ . it helps mitigate global offsets by representing structures relative to global points, leading to more accurate and stable reconstructions; 2). global points can be flexibly selected or learned for different types of motion and deformation; 3). global points are useful for adaptive density control in 3DGS during optimization, helping prune transparent Gaussian points while densifying points in high-frequency areas. By decoupling global points and focusing on relative structures, we achieve a more robust and accurate process for deformation reconstruction.

We model the dynamic scene using relative time-varying structures, as shown in Fig. 1. The relative position of the ith 3D Gaussian at time t is given by $\Delta x _ { i } ^ { t } = x _ { i } ^ { t } - x _ { i } ^ { \star }$ , where $\boldsymbol { x } _ { i } ^ { t }$ is the center of the ith 3D Gaussian. Therefore, the relative trajectory of the ith 3D Gaussian is $\begin{array} { r l } { \Delta T _ { i } } & { { } = } \end{array}$ $[ \Delta x _ { i } ^ { 0 } , \Delta x _ { i } ^ { 1 } , . . . , \Delta x _ { i } ^ { \bar { N } } ] ^ { T }$ , and the relative motion trajectories for all 3D Gaussian points can be represented as $\Delta T =$ $[ \Delta T _ { 0 } , \Delta T _ { 1 } , \dots , \Delta T _ { P } ] \ \stackrel { \cdot } { \in } \ \mathbb { R } ^ { ( N \times 3 ) \times P }$ . The relative trajectory of the ith point can be expressed as a linear combination of basis trajectories: $\begin{array} { r } { \Delta T _ { i } = \sum _ { j = 1 } ^ { k } \sigma _ { j } ^ { x } ( x _ { i } ^ { \star } ) \theta _ { j } } \end{array}$ , where $\theta _ { j } ~ \in ~ \mathbb { R } ^ { N }$ are the trajectory basis vectors over all time frames, $\theta _ { j } ~ = ~ [ \theta _ { j } ^ { 0 } , \theta _ { j } ^ { 1 } , \ldots , \bar { \theta } _ { j } ^ { N } ] ^ { T }$ , and $\sigma _ { i } ^ { x } ( x _ { i } ^ { \star } ) ~ \in ~ \mathbb { R } ^ { 3 }$ are the trajectory coefficients modeled with MLPs and shared across all time frames. All global points are detached when inputted into the motion trajectory field, without background propagation. For the ith 3D Gaussian point at time t, its relative and global positions can be represented as:

$$
\Delta x _ { i } ^ { t } = \sum _ { j = 1 } ^ { k } \sigma _ { j } ( x _ { i } ^ { \star } ) \theta _ { j } ^ { t }\tag{1}
$$

$$
x _ { i } ^ { t } = x _ { i } ^ { \star } + \sum _ { j = 1 } ^ { k } \sigma _ { j } ( x _ { i } ^ { \star } ) \theta _ { j } ^ { t }\tag{2}
$$

We represent scene motion using a motion trajectory field, described through learnable basis functions. For each 3D Gaussian point, we encode its trajectory coefficients with an MLP $\mathcal { F } _ { t c : }$ , as follows:

$$
\{ \sigma _ { j } ( x _ { i } ^ { \star } ) \} _ { j = 1 } ^ { k } = \mathcal { F } _ { t c } ( \mathcal { G } ( x _ { i } ^ { \star } ) )\tag{3}
$$

where $\sigma _ { j } \in \mathbb { R } ^ { 3 }$ are basis coefficients (separate for x, $y ,$ and z) and $\mathcal { G }$ represents positional encoding. We choose $k ~ = ~ 4 0$ basis functions. The encoding function ${ \mathcal { G } } _ { : }$ , with linearly increasing frequency, is expressed as:

$$
\mathcal { G } ( x _ { i } ^ { \star } ) = \left( \sin ( 2 ^ { k } \pi x _ { i } ^ { \star } ) , \cos ( 2 ^ { k } \pi x _ { i } ^ { \star } ) \right) _ { k = 0 } ^ { L - 1 }\tag{4}
$$

where $L = 1 2$ for encoding global reference point $x _ { i } ^ { \star }$ This choice is based on the assumption that scene motion tends to occur at low frequencies [50].

The global learnable motion basis, $\{ \theta _ { j } ^ { t } \} _ { j = 1 } ^ { k }$ , where $\theta _ { j } ^ { t } \in$ R, is introduced to replace the trajectories basis in the original trajectory space [1]. These bases span every time step t of the input video and are optimized jointly with the MLP. Similarly with [23, 40], we initialize the basis $\{ \theta _ { j } \} _ { j = 1 } ^ { k }$ using the DCT basis, but fine-tune it during optimization along with other components. This fine-tuning is necessary because a fixed DCT basis often fails to capture the wide range of real-world motions [23]. The global reference points $x _ { i } ^ { \star }$ are initialized using points from COLMAP [36], a standard procedure in 3DGS [13].

Using Eq. (1) and Eq. (2), we can generate all Gaussian central points across all time frames, $\begin{array} { r l } { X } & { { } = } \end{array}$ $[ { \pmb x } ^ { 1 } , { \pmb x } ^ { 2 } , . . . , { \pmb x } ^ { N } ] ^ { T } \ \in \ \mathbb { R } ^ { ( N \times 3 ) \times P }$ , each row corresponds to all points at a single time frame, while each column represents the trajectory of a point. This motion trajectory field models the dynamic 3D Gaussian points. However, we observed that if each trajectory shares the same covariance across all time frames, using only a time-varying Gaussian centre is insufficient to effectively model dynamic scenes. Since 3DGS includes anisotropic covariance, each Gaussian may exhibit temporally different rotations to capture $\mathrm { d y } .$ namic geometry and appearance changes. To better model scene deformations, we extend the motion trajectory field ( Eq. (1) and Eq. (2)), to also account for time-varying covariance $\Sigma _ { i } = r _ { i } { s _ { i } } { s _ { i } ^ { T } } { r _ { i } ^ { T } }$ [13], and we use l motion-scale bases and m rotation bases, as the following equations:

$$
s _ { i } ^ { t } = s _ { i } ^ { \star } + \sum _ { j = 1 } ^ { l } \beta _ { j } ( x _ { i } ^ { \star } ) \lambda _ { j } ^ { t }\tag{5}
$$

$$
r _ { i } ^ { t } = r _ { i } ^ { \star } + \sum _ { j = 1 } ^ { m } \gamma _ { j } ( x _ { i } ^ { \star } ) \eta _ { j } ^ { t }\tag{6}
$$

To simplify the model, we keep opacity and radiance time-invariant, sharing them across all times and global reference points. The geometrical structure and covariance are time-varying, but each trajectory shares the same spherical harmonics coefficients, as the covariance and central points already account for changes in viewing perspectives over time. For the Gaussian i at time t, we can obtain its Gaussian primitives $\{ x _ { i } ^ { t } , s _ { i } ^ { t } , r _ { i } ^ { t } , \alpha _ { i } , c _ { i } \}$ . By applying rendering equations from [13], we can render the image at any time frame.

## 3.3. Static and dynamic separation

We assign an additional indicator parameter to each 3D Gaussian primitive to represent its states (static or dynamic). This distinction allows us to select dynamic Gaussian points for supervising the motion field. When modeling entire dynamic scenes using an implicit deformation field [48], we observed that the static part can mistakenly acquire trajectories (see Fig. 5). This not only harms novelview synthesis for the static part but also introduces noisy supervision during training neural motion field. To address this, we assign a continuous parameter $p \in [ 0 , 1 ]$ to indicate the probability of a Gaussian primitive belonging to either the static background or dynamic foreground. Ideally, dynamic Gaussians in the foreground should have $p = 1$ , while others in the background should have $p = 0$ . Thus, the parameters of each 3D Gaussian primitive are extended to $\{ x _ { i } ^ { t } , s _ { i } ^ { t } , r _ { i } ^ { t } , \alpha _ { i } , c _ { i } , p _ { i } \}$ We render a static/dynamic segmentation mask (MË ) using volume rendering by replacing the $c _ { i }$ with $p _ { i }$ in rendering equations [13]. A pseudosegmentation mask (M) for dynamic objects is generated using segment and tracking anything [12, 15, 46]. The segmentation loss can be obtained as follows:

$$
\mathcal { L } _ { m } = | | \hat { \mathcal { M } } - \mathcal { M } | | ^ { 2 }\tag{7}
$$

When rendering the segmentation map, Gaussian attributes of location, covariance, and opacity are detached without gradient, so that this loss only optimizes its state probability $p _ { i }$ . We set a high threshold (0.8) for selecting the static Gaussian to mitigate the impact of segmentation errors on rendering and motion modelling, as it is observed that the motion field can model static background but not the other way around. The parameter $p _ { i }$ indicates whether a Gaussian is dynamic. When the model is well-optimized, $p _ { i }$ should be either 0 or 1, based on which we design the point-wise loss $\mathcal { L } _ { 3 m r }$ to force each Gaussian to have only one state. We add this loss at the late stage of the training process.

$$
\begin{array} { r } { \mathcal { L } _ { 3 m r } = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } - \left( p _ { i } \log ( p _ { i } ) + ( 1 - p _ { i } ) \log ( 1 - p _ { i } ) \right) } \end{array}\tag{8}
$$

## 3.4. Motion regularization

In the motion trajectory field, each point only provides location-dependent coefficients and all points share the same global motion basis which provides a weak prior on a smooth trajectory. To inject physical-based priors into the motion trajectory field, we apply the as rigid as possible loss [17, 28], $\mathcal { L } ^ { \mathrm { a r a p } }$ , to achieve the temporal smoothness. We assume that each Gaussian and their neighboring Gaussian points should follow rigid transformation of the coordinate system cross two timesteps. The $\mathcal { L } ^ { \mathrm { a r a p } }$ is defined as:

$$
\mathcal { L } _ { a r a p } = \frac { 1 } { k | S | } \sum _ { i \in \mathcal { S } } \sum _ { j \in \mathrm { k n n } _ { i } } w _ { i , j } \| ( x _ { j } ^ { t - 1 } - x _ { i } ^ { t - 1 } ) -\tag{9}
$$

We randomly select S points to enforce the $\mathcal { L } ^ { \mathrm { a r a p } }$ , and for each selected point, a set of neighboring Gaussians is selected for it using the k-nearest neighbors (knn), and the loss is down-weighted by an isotropic Gaussian weighting factor:

$$
w _ { i , j } = \exp \left( - \rho _ { w } \| x _ { j } ( t ) - x _ { i } ( t ) \| _ { 2 } ^ { 2 } \right)
$$

We proposed a spatial smoothness loss to enforce smoothness over neighbouring spatial locations. We apply a perturbing Ïµ on the input global points $x ^ { \star }$ and encourage the location-dependent attributes at location $x _ { i } ^ { \star } + \epsilon$ to be consistent with $x _ { i } ^ { \star }$ . The spatial smoothness term is:

$$
\begin{array} { r l } { \mathcal { L } _ { s p } = } & { { } \| \sigma _ { j } ( x _ { i } ^ { \star } ) - \sigma _ { j } ( x _ { i } ^ { \star } + \epsilon ) \| _ { 2 } + } \\ { w _ { \beta } \| \beta _ { j } ( x _ { i } ^ { \star } ) - \beta _ { j } ( x _ { i } ^ { \star } + \epsilon ) \| _ { 2 } + } \\ { w _ { \gamma } \| \gamma _ { j } ( x _ { i } ^ { \star } ) - \gamma _ { j } ( x _ { i } ^ { \star } + \epsilon ) \| _ { 2 } } \end{array}\tag{10}
$$

where the perturbing valueâs magnitude is adaptively set according to the scale of the scene, i.e., Ïµ = 0.001 â scale, and $w _ { \beta }$ and $w _ { \gamma }$ are corresponding weighting factor for rotation and scale.

All these motion regularisers are applied only on dynamic Gaussian points using the foreground/background mask. We observe that dynamic Gaussian point tends to have a larger gradient flow than static points, since dynamic region has a large photometric loss and is also constrained with motion regularisers. If we apply the similar density control mechanism to both points, this can cause over-dense dynamic point; therefore, we set a large gradient threshold for densifying the dynamic points.

When the model is well optimized, we can render the image with the learned Gaussian primitives $\left\{ \mathbf { x } _ { i } ^ { * } , \mathbf { s } _ { i } ^ { * } , \mathbf { r } _ { i } ^ { * } , \alpha _ { i } , c _ { i } , p _ { i } \right\}$ with trajectory motion field only applied on dynamic Gaussian points. With the rendered images, we can calculate the Photometric loss $\mathcal { L } _ { p h o }$ , which is the same as [13]. Therefore, our final loss to optimize the model is as follows:

$$
\mathcal { L } = \mathcal { L } _ { p h o } + \mathcal { L } _ { m } + \mathcal { L } _ { 3 m r } + \lambda _ { a } \mathcal { L } _ { a r a p } + \lambda _ { s } \mathcal { L } _ { s p }\tag{11}
$$

## 4. Experiments

## 4.1. Experimental setup

## 4.1.1. Dataset:

We conduct various experimental evalution on two dataset: D-NeRF [35] and HyperNeRF [33], and both are monocular video dataset. D-NeRF is a synthetic dataset, including 8 sets of dynamic scenes, and each scene features complex motion such as articulated objects and human actions. Every image is 800Ã800 with 100 to 200 images per scene. Most images only contain the dynamic object without static background, therefore we do not apply the mask loss, i.e. $\mathcal { L } _ { 3 m r }$ and ${ \mathcal { L } } _ { m }$ , for this dataset. HyperNeRF is a real-world dataset that includes a monocular video on both real rigid and non-rigid deformable objects. We create a mask for each object using track anything with one click [46]. All images are down-sampled to 540Ã960 for a fair comparison with other baselines.

## 4.1.2. Implementation details:

The entire training iterations are 50k, and we firstly only train the global Gaussian points up to 5k iterations without incorporating trajectory fields for stable training, then we start to optimize the trajectory motion field jointly until the end and stop the densification until 25k iteration. The Adam optimizer [14] is used to optimize our model with different learning rates for each module, and the step learning rate (StepLR) scheduler in PyTorch [34] is used to adjust the learning rate. The learning rate for optimizing the 3DGS is the same as the original paper [13]. For optimizing the trajectory motion field, we use a learning rate of 1e â 3 for the coefficient MLP model with a decay factor of 0.5 for every 15k iterations in StepLR and use the learning rate 5e â 4 for trajectory basis with the same decay factors. The gradient thresholds for densification are 4e â 4 and 8e â 4 for static and dynamic points respectively. $\lambda _ { a }$ and $\lambda _ { s }$ are set to 0.3 and 0.6 respectively. both of wÎ² and $w _ { \gamma }$ are 0.5. The 3DGS are initialized randomly for D-NeRF [35] and The initialization for HyperNeRF [33] is from Structure-frommotion points derived from COLMAP [36].

## 4.2. Comparison results

## 4.2.1. Results on synthetic dataset

We compared our methods with all relevant baselines with the monocular synthetic dataset from D-NeRF [35]. We choose 3DGS [13], D-NeRF [35], TiNeuVox [6], Tensor4D [37], K-Planes [7], and D-3DGS [48] as the baselines for this dataset, because we can find the reported performance in their paper for a fair comparison. In this synthesis dataset, static backgrounds are automatically removed and we donât apply the mask loss when optimizing the model. The proposed method (âOursâ in the table) outperforms other baseline methods in most of the scenes, demonstrating superior performance in terms of PSNR, SSIM, and LPIPS, particularly excelling in the Bouncing Balls, T-Rex, Stand Up, and Jumping Jacks scenes. The strong results suggest that it is well-suited for high-quality, perceptually accurate image reconstruction tasks, especially in dynamic synthetic scenes. D-3DGS is a close competitor, showcasing its robustness in certain scenarios, while K-Planes and TiNeu-Vox also offer competitive alternatives. We can also tell that D-3DGS is another strong performer, showcasing its robustness in certain scenarios.

Table 1. Quantitative comparison of our method and baselines evaluated on the synthetic dataset [35]. We report three metrics: peak signalto-noise ratio (PSNR), structural similarity index (SSIM), and learned perceptual image patch similarity (LPIPS) with VGG model [49], for eight scenes with full image resolution 800Ã800. The color coding in the table highlights the best , second -best, and third -best performance. The results of baselines are gathered from the paper on the corresponding methods.
<table><tr><td rowspan="2">Method</td><td colspan="3">Hell Warrior</td><td colspan="3">Mutant</td><td colspan="3">Hook</td><td colspan="3">Bouncing Balls</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3D-GS</td><td>29.89</td><td>0.9155</td><td>0.1056</td><td>24.53</td><td>0.9336</td><td>0.0580</td><td>21.71</td><td>0.8876</td><td>0.1034</td><td>23.20</td><td>0.9591</td><td>0.0600</td></tr><tr><td>D-NeRF</td><td>24.06</td><td>0.9440</td><td>0.0707</td><td>30.31</td><td>0.9672</td><td>0.0392</td><td>29.02</td><td>0.9595</td><td>0.0546</td><td>38.17</td><td>0.9891</td><td>0.0323</td></tr><tr><td>TiNeuVox</td><td>27.10</td><td>0.9638</td><td>0.0768</td><td>31.87</td><td>0.9607</td><td>0.0474</td><td>30.61</td><td>0.9599</td><td>0.0592</td><td>40.23</td><td>0.9926</td><td>0.0416</td></tr><tr><td>Tensor4D</td><td>31.26</td><td>0.9254</td><td>0.0735</td><td>29.11</td><td>0.9451</td><td>0.0640</td><td>28.63</td><td>0.9433</td><td>0.0636</td><td>24.47</td><td>0.9622</td><td>0.0437</td></tr><tr><td>K-Planes</td><td>24.58</td><td>0.9520</td><td>0.0824</td><td>32.50</td><td>0.9713</td><td>0.0362</td><td>28.12</td><td>0.9489</td><td>0.0662</td><td>40.05</td><td>0.9934</td><td>0.0322</td></tr><tr><td>D-3DGS</td><td>41.54</td><td>0.9873</td><td>0.0234</td><td>42.63</td><td>0.9951</td><td>0.0052</td><td>37.42</td><td>0.9867</td><td>0.0144</td><td>41.01</td><td>0.9953</td><td>0.0093</td></tr><tr><td>Ours</td><td>41.67</td><td>0.9877</td><td>0.02361</td><td>43.27</td><td>0.9962</td><td>0.0043</td><td>38.68</td><td>0.989</td><td>0.0115</td><td>40.15</td><td>0.9929</td><td>0.0297</td></tr><tr><td rowspan="2">Method</td><td colspan="3">Lego</td><td colspan="3">T-Rex</td><td colspan="3">Stand Up</td><td colspan="3">Jumping Jacks</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3D-GS</td><td>22.10</td><td>0.9384</td><td>0.0607</td><td>21.93</td><td>0.9539</td><td>0.0487</td><td>21.91</td><td>0.9301</td><td>0.0785</td><td>20.64</td><td>0.9297</td><td>0.0828</td></tr><tr><td>D-NeRF</td><td>25.56</td><td>0.9363</td><td>0.0821</td><td>30.61</td><td>0.9671</td><td>0.0535</td><td>33.13</td><td>0.9781</td><td>0.0355</td><td>32.70</td><td>0.9779</td><td>0.0388</td></tr><tr><td>TiNeuVox</td><td>26.64</td><td>0.9258</td><td>0.0877</td><td>31.25</td><td>0.9666</td><td>0.0478</td><td>34.61</td><td>0.9797</td><td>0.0326</td><td>33.49</td><td>0.9771</td><td>0.0408</td></tr><tr><td>Tensor4D</td><td>23.24</td><td>0.9183</td><td>0.721</td><td>23.86</td><td>0.9351</td><td>0.0544</td><td>30.56</td><td>0.9581</td><td>0.0363</td><td>24.20</td><td>0.9253</td><td>0.0667</td></tr><tr><td>K-Planes</td><td>28.91</td><td>0.9695</td><td>0.0331</td><td>30.43</td><td>0.9737</td><td>0.0310</td><td>33.10</td><td>0.9793</td><td>0.0310</td><td>31.11</td><td>0.9708</td><td>0.0468</td></tr><tr><td>D-3DGS</td><td>33.07</td><td>0.9794</td><td>0.0183</td><td>38.10</td><td>0.9933</td><td>0.0098</td><td>44.62</td><td>0.9951</td><td>0.0063</td><td>37.72</td><td>0.9897</td><td>0.0126</td></tr><tr><td>Ours</td><td>30.48</td><td>0.9703</td><td>0.0284</td><td>38.71</td><td>0.9939</td><td>0.0089</td><td>46.28</td><td>0.9976</td><td>0.0046</td><td>40.01</td><td>0.9930</td><td>0.0086</td></tr></table>

The qualitative comparison results are shown in Fig. 2, from which we can tell that our proposed method closely matches the ground truth, preserving finer details and better geometric consistency. While D-3DGS also perform well, capturing many key features, it still falls short in some areas of fine detail preservation. D-NeRF, Tensor4D, and K-Planes struggle with blurring and less accurate geometry, especially noticeable in the finer areas. For instance, our proposed method can model the relative position between the knee- and spiked-wristband on the arm much more accurately than D-3DGS, Tensor4D, and K-planes for the scene of the HellWarrior, meanwhile, our rendering performance is better than TiNeuVox and D-NeRF given the similar geometrical modelling, for the Stand Up, we can tell that our method can preserve the details of hair while other methods failed to model this. The visualization of motion trajectory is visualized in Fig. 3, in which the motion trajectory can accurately illustrate how objects move. The proposed method stands out by achieving high fidelity and being effective in modelling moving 3D geometry.

## 4.2.2. Results on real-world dataset

We evaluate our method on the real-world dataset HyperNeRF [33] and do the comparison with the previous the stateof-the-art: NeRF [29], Nerfies, HyperNeRF [33], TiNeu-Vox [6], and Gaussian-flow [25]. The experimental results, shown in Tab. 2, demonstrate that the proposed method achieves the best performance for novel view synthesis, outperforming other approaches across multiple scenes. It records the highest mean scores in PSNR and SSIM, particularly excelling in the Broom, 3D Printer, and Chicken scenes. While Gaussian-flow [25] shows strong performance, particularly in the Peel Banana scene with the highest SSIM, it ranks second overall. TiNeuVox consistently performs well, securing third place in most cases. Comparatively, methods like NeRF, Nerfies, and HyperNeRF show lower performance. These results indicate the effectiveness of the proposed method for high-quality image reconstruction in real-world scenarios. The qualitative comparison for novel view synthesis on two scenes is represented in Fig. 4.

K-Planes

TiNeuVox

D-3DGS

<!-- image-->  
Figure 2. Qualitative results of baselines and our method against the ground truth (GT) on a synthetic dataset. The visualization highlights the ability of each method to reconstruct geometrical poses and fine details across different scenes. We visualize the four scenes: HellWarrior, Jumping Jacks, Stand Up, and T-Rex from top to bottom.

<!-- image-->  
Figure 3. Visualization of motion trajectory on D-NeRF.

Our proposed method stands out by producing the most accurate and sharp reconstructions, with sharp details and minimal artifacts, that closely match the ground truth. Other methods, particularly NeRF and Nerfies, exhibit significant blurring and distortions, especially in regions with motion. HyperNeRF, TiNeuVox, and Gaussian-Flow show improvements but still lag behind the visual quality achieved by the proposed method. The motion trajectory is visualized in the rightmost image of Fig. 6.

## 4.3. Ablation study

## 4.3.1. static and dynamic separation

Foreground masks are used to explicitly separate static and dynamic Gaussian primitives. Without separating them, most of the static background will start to move. We can find this from the leftmost image in Fig. 6, from which we can tell that most of the static background points are with trajectory, which can not only degrade the synthesis quality of static background but also cause over-densification of Gaussian points and slow rendering speed. This ablation study evaluates the impact of mask segmentation and its regularization on image quality, speed, and resource efficiency, shown in Tab. 3. We find that introducing mask segmentation $( { \mathcal { L } } _ { m } )$ leads to a noticeable improvement, with PSNR increasing to 26.1, SSIM to 0.853, and the speed improving to 22 FPS. Further refining the model by combining mask segmentation with an additional refinement technique $( { \mathcal { L } } _ { m } + { \mathcal { L } } _ { 3 m r } )$ results in the best performance and the highest speed of 26 FPS, using only 480k primitives. The results demonstrate the effectiveness of our mask segmentation module.

<!-- image-->

Figure 4. Qualitative comparison. Each row shows results from six methods: NeRF, Nerfies, HyperNeRF, TiNeuVox, Gaussian-Flow, and our method, along with the ground truth (GT).  
<!-- image-->  
Figure 5. Visualization of static and dynamic Gaussian primitives. The images, from left to right, show the ground truth, rendered image, foreground mask, rendered static scene, and rendered dynamic object. For the foreground mask, black represents the dynamic foreground.

Table 2. Per-scene quantitative comparisons on HyperNeRF [33] dataset. Results are collected from the corresponding papers. Our method can achieve the best quality for novel view synthesis in this real-world dataset.
<table><tr><td rowspan="2">Method</td><td colspan="2">Broom</td><td colspan="2">3D Printer</td><td colspan="2">Chicken</td><td colspan="2">Peel Banana</td><td colspan="2">Mean</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td></tr><tr><td>NeRF</td><td>19.9</td><td>0.653</td><td>20.7</td><td>0.780</td><td>19.9</td><td>0.777</td><td>20.0</td><td>0.769</td><td>20.1</td><td>0.745</td></tr><tr><td>Nerfies</td><td>19.2</td><td>0.567</td><td>20.6</td><td>0.830</td><td>26.7</td><td>0.943</td><td>22.4</td><td>0.872</td><td>22.2</td><td>0.803</td></tr><tr><td>HyperNeRF</td><td>19.3</td><td>0.591</td><td>20.0</td><td>0.821</td><td>26.9</td><td>0.948</td><td>23.3</td><td>0.896</td><td>22.4</td><td>0.814</td></tr><tr><td>TiNeuVox</td><td>21.5</td><td>0.686</td><td>22.8</td><td>0.841</td><td>28.3</td><td>0.947</td><td>24.4</td><td>0.873</td><td>24.3</td><td>0.837</td></tr><tr><td>Gaussian-flow</td><td>22.8</td><td>0.709</td><td>25.0</td><td>0.877</td><td>30.4</td><td>0.945</td><td>27.0</td><td>0.917</td><td>26.3</td><td>0.862</td></tr><tr><td>Ours</td><td>23.4</td><td>0.745</td><td>26.7</td><td>0.896</td><td>31.2</td><td>0.951</td><td>26.2</td><td>0.893</td><td>26.9</td><td>0.871</td></tr></table>

<table><tr><td>Seg</td><td>PSNRâ</td><td>SSIMâ</td><td>FPSâ</td><td>Num(k) â</td></tr><tr><td>N/A</td><td>25.4</td><td>0.847</td><td>13</td><td>740</td></tr><tr><td> ${ \mathcal { L } } _ { m }$ </td><td>26.1</td><td>0.853</td><td>22</td><td>510</td></tr><tr><td> $\mathcal { L } _ { m } + \mathcal { L } _ { 3 m r }$ </td><td>26.9</td><td>0.871</td><td>26</td><td>480</td></tr></table>

Table 3. Ablation study of mask segmentation. Num, the number of Gaussian primitives, is multiplied by 103.

<!-- image-->  
Figure 6. Visualization of motion trajectory in the HyperNeRF dataset. The leftmost image shows the rendered result without static and dynamic separation, the middle two images include static and dynamic separation (one zoomed in), and the rightmost image is rendered from the model with motion regularization.

## 4.3.2. Motion Regularization

Tab. 4 and Fig. 6 present the results of an ablation study on motion regularization, comparing different combinations of regularization techniques and their effects on image quality. The baseline âN/Aâ without any regularization achieves the highest performance, but its trajectory is not smooth. When the as-rigid-as-possible regularization $( \mathcal { L } _ { a r a p } )$ is applied, both metrics decrease slightly. Similarly, using spatial smoothness regularization $( \mathcal { L } _ { s p } )$ shows a minor reduction compared to the baseline. Combining both regularization techniques leads to a decreased performance but with the smoothest trajectory. This suggests that although motion regularization slightly reduces the overall image quality metrics, it contributes positively to motion consistency (as shown in the rightmost image in Fig. 6, which is critical for certain robotics applications. Noteworthy, our method can still achieve state-of-the-art rendering performance with motion regularization.

<table><tr><td></td><td>N/A</td><td> $\mathcal { L } _ { a r a p }$ </td><td> $\mathcal { L } _ { s p }$ </td><td> $\mathcal { L } _ { a r a p } + \mathcal { L } _ { s p }$ </td></tr><tr><td>PSNRâ SSIMâ</td><td>27.4 0.879</td><td>27.0 0.873</td><td>27.2 0.875</td><td>26.9 0.871</td></tr></table>

Table 4. Ablation studies of motion regularization.

## 5. Conclusion

In this paper, we proposed a novel method for reconstructing dynamic scenes and recovering motion trajectories from monocular video input by combining 3DGS with a motion trajectory field. Our approach effectively handles complex non-rigid motions while achieving real-time rendering performance. The decoupling of dynamic and static components allows for efficient representation and reduced GPU memory consumption. Through extensive quantitative and qualitative evaluations, we demonstrated that our method can produce high-quality novel-view synthesis and physically plausible motion trajectories. The proposed framework provides a significant step forward in the real-time rendering of dynamic scenes and can be a valuable tool for various applications such as virtual reality and robotic manipulation. Future work may explore further optimizations in motion trajectory representation and extend the method to more diverse dynamic scene scenarios.

## References

[1] Ijaz Akhter, Yaser Sheikh, Sohaib Khan, and Takeo Kanade. Nonrigid structure from motion in trajectory space. Advances in neural information processing systems, 21, 2008. 1, 2, 4

[2] Ijaz Akhter, Yaser Sheikh, Sohaib Khan, and Takeo Kanade. Trajectory space: A dual representation for nonrigid structure from motion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(7):1442â1456, 2010. 2

[3] Christoph Bregler, Aaron Hertzmann, and Henning Biermann. Recovering non-rigid 3d shape from image streams. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition. CVPR 2000 (Cat. No. PR00662), pages 690â696. IEEE, 2000. 2

[4] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130â141, 2023. 1, 2

[5] Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B Tenenbaum, and Jiajun Wu. Neural radiance flow for 4d view synthesis and video processing. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 14304â14314. IEEE Computer Society, 2021. 1

[6] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias NieÃner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia 2022 Conference Papers, pages 1â9, 2022. 2, 6

[7] Sara Fridovich-Keil, Giacomo Meanti, Frederik RahbÃ¦k Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12479â12488, 2023. 1, 2, 6

[8] Wanshui Gan, Hongbin Xu, Yi Huang, Shifeng Chen, and Naoto Yokoya. V4d: Voxel for 4d novel view synthesis. IEEE Transactions on Visualization and Computer Graphics, 2023. 2

[9] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5712â5721, 2021. 2

[10] Zhiyang Guo, Wengang Zhou, Li Li, Min Wang, and Houqiang Li. Motion-aware 3d gaussian splatting for efficient dynamic scene reconstruction. arXiv preprint arXiv:2403.11447, 2024. 2

[11] Kai Katsumata, Duc Minh Vo, and Hideki Nakayama. An efficient 3d gaussian representation for monocular/multi-view dynamic scenes. arXiv preprint arXiv:2311.12897, 2023. 2

[12] Lei Ke, Mingqiao Ye, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu, et al. Segment anything in high quality. Advances in Neural Information Processing Systems, 36, 2024. 4

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4):1â14, 2023. 1, 2, 3, 4, 5, 6

[14] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), San Diega, CA, USA, 2015. 5

[15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015â4026, 2023. 4

[16] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting. arXiv preprint arXiv:2312.00112, 2023. 2

[17] Suryansh Kumar, Yuchao Dai, and Hongdong Li. Monocular dense 3d reconstruction of a complex dynamic scene from two perspective frames. In Proceedings of the IEEE international conference on computer vision, pages 4649â4657, 2017. 5

[18] Suryansh Kumar, Yuchao Dai, and Hongdong Li. Spatiotemporal union of subspaces for multi-body non-rigid structure-from-motion. Pattern Recognition, 71:428â443, 2017. 2

[19] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5521â5531, 2022. 2

[20] Xuesong Li and Jose E Guivant. Efficient and accurate object detection with simultaneous classification and tracking under limited computing power. IEEE Transactions on Intelligent Transportation Systems, 24(6):5740â5751, 2023. 1

[21] Xuesong Li, Jose Guivant, and Subhan Khan. Real-time 3d object proposal generation and classification using limited processing resources. Robotics and Autonomous Systems, 130:103557, 2020. 1

[22] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6498â 6508, 2021. 2

[23] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. Dynibar: Neural dynamic image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 2, 4

[24] Zhengqi Li, Richard Tucker, Noah Snavely, and Aleksander Holynski. Generative image dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24142â24153, 2024. 3

[25] Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21136â 21145, 2024. 1, 2, 6

[26] Qingming Liu, Yuan Liu, Jiepeng Wang, Xianqiang Lv, Peng Wang, Wenping Wang, and Junhui Hou. Modgs: Dy-

namic gaussian splatting from causually-captured monocular videos. arXiv preprint arXiv:2406.00434, 2024. 3

[27] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13â23, 2023. 2

[28] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713, 2023. 1, 2, 5

[29] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2, 6

[30] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022. 1, 2

[31] Atsuhiro Noguchi, Xiao Sun, Stephen Lin, and Tatsuya Harada. Neural articulated radiance field. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5762â5772, 2021. 1

[32] Hyun Soo Park, Takaaki Shiratori, Iain Matthews, and Yaser Sheikh. 3d reconstruction of a moving point from a series of 2d projections. In Computer VisionâECCV 2010: 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings, Part III 11, pages 158â171. Springer, 2010. 2

[33] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M. Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. ACM Trans. Graph., 40(6), 2021. 5, 6, 8

[34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 5

[35] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318â10327, 2021. 2, 3, 5, 6

[36] Johannes Lutz Schonberger and Jan-Michael Frahm. Â¨ Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 4, 5

[37] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu, Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16632â 16642, 2023. 2, 6

[38] Fengrui Tian, Shaoyi Du, and Yueqi Duan. Mononerf: Learning a generalizable dynamic radiance field from monocular videos. In Proceedings of the IEEE/CVF Interna-

tional Conference on Computer Vision, pages 17903â17913, 2023. 1, 2

[39] Lorenzo Torresani, Aaron Hertzmann, and Chris Bregler. Nonrigid structure-from-motion: Estimating shape and motion with hierarchical priors. IEEE transactions on pattern analysis and machine intelligence, 30(5):878â892, 2008. 2

[40] Chaoyang Wang, Ben Eckart, Simon Lucey, and Orazio Gallo. Neural trajectory fields for dynamic novel view synthesis. arXiv preprint arXiv:2105.05994, 2021. 1, 2, 3, 4

[41] Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, and Huaping Liu. Masked space-time hash encoding for efficient dynamic scene reconstruction. Advances in Neural Information Processing Systems, 36, 2024. 2

[42] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764, 2024. 1, 2

[43] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20310â20320, 2024. 1, 2

[44] Jing Xiao, Jin-xiang Chai, and Takeo Kanade. A closedform solution to non-rigid shape and motion recovery. In Computer Vision-ECCV 2004: 8th European Conference on Computer Vision, Prague, Czech Republic, May 11-14, 2004. Proceedings, Part IV 8, pages 573â587. Springer, 2004. 2

[45] Wenhui Xiao, Remi Chierchia, Rodrigo Santa Cruz, Xuesong Li, David Ahmedt-Aristizabal, Olivier Salvado, Clinton Fookes, and Leo Lebrat. Neural radiance fields for the real world: A survey. arXiv preprint arXiv:2501.13104, 2025. 1

[46] Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, and Feng Zheng. Track anything: Segment anything meets videos. arXiv preprint arXiv:2304.11968, 2023. 4, 5

[47] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642, 2023. 2

[48] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20331â20341, 2024. 1, 2, 3, 4, 6

[49] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6

[50] Zhoutong Zhang, Forrester Cole, Richard Tucker, William T Freeman, and Tali Dekel. Consistent depth of moving objects in video. ACM Transactions on Graphics (TOG), 40(4):1â 12, 2021. 4

[51] Yingying Zhu and Simon Lucey. Convolutional sparse coding for trajectory reconstruction. IEEE transactions on pattern analysis and machine intelligence, 37(3):529â540, 2013. 2