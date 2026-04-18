# E-3DGS: Event-Based Novel View Rendering of Large-Scale Scenes Using 3D Gaussian Splatting

Sohaib Zahid1,2 Viktor Rudnev1,2 Eddy Ilg1 Vladislav Golyanik2 1Saarland University 2MPI for Informatics, SIC

## Abstract

Novel view synthesis techniques predominantly utilize RGB cameras, inheriting their limitations such as the need for sufficient lighting, susceptibility to motion blur, and restricted dynamic range. In contrast, event cameras are significantly more resilient to these limitations but have been less explored in this domain, particularly in large-scale settings. Current methodologies primarily focus on frontfacing or object-oriented (360-degree view) scenarios. For the first time, we introduce 3D Gaussians for event-based novel view synthesis. Our method reconstructs large and unbounded scenes with high visual quality. We contribute the first real and synthetic event datasets tailored for this setting. Our method demonstrates superior novel view synthesis and consistently outperforms the baseline EventNeRF by a margin of 11â25% in PSNR (dB) while being orders of magnitude faster in reconstruction and rendering.

## 1. Introduction

Novel view synthesis offers a fundamental approach to visualizing complex scenes by generating new perspectives from existing imagery. This has many potential applications, including virtual reality, movie production and architectural visualization [27]. An emerging alternative to the common RGB sensors are event cameras, which are bioinspired visual sensors recording events, i.e. asynchronous per-pixel signals of changes in brightness or color intensity.

Event streams have very high temporal resolution and are inherently sparse, as they only happen when changes in the scene are observed. Due to their working principle, event cameras bring several advantages, especially in challenging cases: they excel at handling high-speed motions and have a substantially higher dynamic range of the supported signal measurements than conventional RGB cameras. Moreover, they have lower power consumption and require varied storage volumes for captured data that are often smaller than those required for synchronous RGB cameras [5, 19].

The ability to handle high-speed motions is crucial in static scenes as well, particularly with handheld moving cameras, as it helps avoid the common problem of motion blur. It is, therefore, not surprising that event-based novel view synthesis has gained attention, although color values are not directly observed. Notably, because of the substantial difference between the formats, RGB- and event-based approaches require fundamentally different design choices.

The first solutions to event-based novel view synthesis introduced in the literature demonstrate promising results [12, 25] and outperform non-event-based alternatives for novel view synthesis in many challenging scenarios. Among them, EventNeRF [25] enables novel-view synthesis in the RGB space by assuming events associated with three color channels as inputs. Due to its NeRF-based architecture [17], it can handle single objects with complete observations from roughly equal distances to the camera. It furthermore has limitations in training and rendering speed: the MLP used to represent the scene requires long training time and can only handle very limited scene extents or otherwise rendering quality will deteriorate. Hence, the quality of synthesized novel views will degrade for larger scenes.

We present Event-3DGS (E-3DGS), i.e., a new method for novel-view synthesis from event streams using 3D Gaussians [9] demonstrating fast reconstruction and rendering as well as handling of unbounded scenes. The technical contributions of this paper are as follows:

â¢ With E-3DGS, we introduce the first approach for novel view synthesis from a color event camera that combines 3D Gaussians with event-based supervision.

â¢ We present frustum-based initialization, adaptive event windows, isotropic 3D Gaussian regularization and 3D camera pose refinement, and demonstrate that highquality results can be obtained.

â¢ Finally, we introduce new synthetic and real event datasets for large scenes to the community to study novel view synthesis in this new problem setting.

Our experiments demonstrate systematically superior results compared to EventNeRF [25] and other baselines. The source code and dataset of E-3DGS are released1.

## 2. Related Work

## 2.1. Novel View Synthesis from RGB Inputs

Novel view synthesis of rigid scenes is predominantly handled assuming RGB inputs. A widely used approach to this problem is to learn coordinate-based neural scene representations allowing rendering novel views at test time. Earlier works such as Neural Radiance Fields (NeRF) and its direct follow-ups [17, 27] used implicit neural representations in combination with volume rendering. They are based on expensive-to-optimize Multi-Layer Perceptrons (MLPs) and are slow at training and evaluation while requiring a relatively low amount of storage space once they are trained. Their stochastic ray sampling requires many samples to obtain an accurate scene approximation, and shooting rays through empty space constitutes unnecessary overhead. Most of these approaches focus on single objects or bounded scenes. Recent techniques accelerate neural MLPbased representations or ray sampling [20, 24] or avoid MLPs [2, 4, 26] by using voxel grids. Some techniques [3] support unbounded scenes by employing radial basis functions, thereby overcoming the limitations of voxel-gridbased methods. Several ray tracing-based methods support large-scale scenes and uncontrolled camera trajectories thanks to progressive NeRF optimization [16, 29]. Instant-NGPs [20] are neural feature volumes with a hash grid that can be learned and evaluated quickly at test time. They can also handle multi-scale training scenarios efficiently.

A promising recent development is the shift from ray tracing to rasterization, marked by the introduction of 3D Gaussian Splatting (3DGS) [9]. This approach presents an alternative paradigm for 3D reconstruction and novel view synthesis using differentiable rasterization with 3D Gaussians as geometric primitives. Since GPU technology and algorithmic research have evolved over several decades to provide high performance for rasterization applications, 3DGS trains substantially quicker and provides much higher rendering throughput than NeRF. Moreover, since it explicitly represents the geometry, it can scale easily as the scene size increases with no special handling required for unbounded scenes. Our approach adopts the 3D Gaussian representation and presents its application to the supervision from event streams. It inherits thereby the advantages of event streams and 3DGS for view synthesis.

## 2.2. Novel View Synthesis from Event Streams

Event-aided sparse odometry and simultaneous localization and mapping approaches are distantly related to our setting, as they do not allow photo-realistic and dense rendering of novel views [7, 10, 13, 22].

As previously discussed, event cameras represent an alternative to RGB sensors for dense novel view synthesis, and some initial work was done on learning 3D scene representations from event streams only. EventNeRF [25] is a seminal framework for training MLP-based implicit 3D representations (see Sec. 2.1) using frames of accumulated color events. While it demonstrates impressive results, it is restricted to camera trajectories with uniform motion and the assumption that the background is a constant color (triggering no events). E-NeRF [12] is another work that resembles the training methodology of EventNeRF for singlechannel (intensity) event cameras and allows training a colored 3D representation from a combination of blurry RGB images and grayscale events. Robust E-NeRF by Low and Lee [14] is a model aiming to reduce the issues caused by uncontrolled camera motion. They introduce the refractory period to the event generation model, i.e. the time during which a pixel is inactive after an event firing. Supervision happens on the level of individual events, and they reformulate the event loss to handle intra-pixel variances of the contrast threshold optimized during training. All these methods adopt ray tracing and can be primarily applied on 360Â° object-centric datasets or front-facing trajectories.

Our approach differs from previous event-based methods in that it demonstrates that rasterization can be efficiently combined with event-based supervision instead of ray tracing. The main design choices of our method are tailored to 3D Gaussians. As a result, our method inherits the primary advantages of 3DGS [9], such as fast training and inference. Similar to EventNeRF [25], our method supports color. However, in contrast, it is not limited to single objects and can handle large-scale scenes.

## 3. Preliminaries

## 3.1. 3D Gaussians

3D Gaussian Splatting [9] is a high-quality and efficient scene representation. The Gaussians are defined by a 3D covariance matrix $\Sigma _ { i }$ centered around a point $\mu _ { i } { \mathrm { : } }$

$$
G _ { i } ( { \pmb x } ) = \exp \left( - \frac { 1 } { 2 } ( { \pmb x } - { \pmb \mu } _ { i } ) ^ { T } { \pmb \Sigma } _ { i } ^ { - 1 } ( { \pmb x } - { \pmb \mu } _ { i } ) \right) ,\tag{1}
$$

and their overlay models the geometry at scene location x. Each Gaussian is additionally associated with an opacity $o _ { i }$ and spherical harmonics that model view-dependent color. For rendering purposes, the means $\pmb { \mu } _ { i }$ and covariance matrices $\Sigma _ { i }$ are transformed into image coordinates. The projected matrix $\Sigma _ { i } ^ { \prime }$ can be obtained by applying the viewing transformation $W$ and the Jacobian J of the affine approximation of the projective transformation:

$$
\pmb { \Sigma } _ { i } ^ { \prime } = J \pmb { W } \pmb { \Sigma } _ { i } \pmb { W } ^ { T } \pmb { J } ^ { T } .\tag{2}
$$

The third row and column of $\Sigma _ { i } ^ { \prime }$ are dropped to obtain a 2D matrix. Using Equation 1, one can then evaluate the different Gaussians i that overlap with an image pixel x and obtain alpha values as $\alpha _ { i , x } = o _ { i } G _ { i } ^ { \prime } ( { \pmb x } )$ . The Gaussians are then sorted according to their depth, and alpha blending for every pixel is performed by combining the view-dependent colors $c _ { i }$ using the following equation:

<!-- image-->  
Figure 1. Overview of our E-3DGS Method. We use 3D Gaussians [9] as the scene representation and assume that initial noisy camera poses are available. We randomly initialize the scene with our frustum-based initialization (Sec. 4.2) and then optimize the Gaussians and the camera poses jointly (Sec. 4.5). To obtain a high-quality reconstruction of both, low-frequency structure and high-frequency detail, we propose a strategy using a large event window from $t _ { s _ { 1 } }$ to t and a small one from $t _ { s _ { 2 } }$ to t (Sec. 4.3). We then define the loss $\mathcal { L } _ { \mathrm { r e c o n } }$ (Sec. 4.6) between renderings from our model at the current time t (indicated green) and previous times $t _ { s _ { 1 } }$ (indicated orange) and $t _ { s _ { 2 } }$ (indicated red), and the accumulated incoming events $E ( t _ { s _ { 1 } } , t )$ and $E ( t _ { s _ { 2 } } , t )$ . We regularize the 3D Gaussians with the loss $\mathcal { L } _ { \mathrm { i s o } }$ (Sec. 4.4).

$$
C _ { x } = \sum _ { i = 1 } ^ { N } T _ { i , x } \alpha _ { i , x } c _ { i } ,\tag{3}
$$

where $\begin{array} { r } { T _ { i , x } = \prod _ { k = 1 } ^ { i - 1 } ( 1 - \alpha _ { k , x } ) } \end{array}$ represents the transmittance.

## 3.2. Event Formation Model

Event cameras generate a continuous stream of events denoted as $e = ( { \pmb x } , p , \tau )$ , where x are the pixel coordinates at which an event is triggered at time Ï , and $p \in \{ - 1 , + 1 \}$ signifies the polarity of the event, indicating an increase or decrease in the logarithmic intensity by the predefined contrast threshold $\Delta .$ . Thus, the relationship between the triggered event and the logarithmic image intensity reads:

$$
L _ { \mathbf { x } } ( \tau ) - L _ { \mathbf { x } } ( \tau ^ { \mathrm { p r e v } } ) = p \Delta ,\tag{4}
$$

where $\tau ^ { \mathrm { p r e v } }$ is the time when the previous event for the pixel was triggered. This concept can then be generalized to

apply for an accumulation of events within a time interval $( \tau _ { 1 } , \tau _ { 2 } ) | \tau _ { 1 } < \tau _ { 2 }$ for a pixel location x as follows:

$$
L _ { \pmb { x } } ( \tau _ { 2 } ) - L _ { \pmb { x } } ( \tau _ { 1 } ) = \sum _ { \tau _ { 1 } < \tau _ { t } \leq \tau _ { 2 } } p _ { t } \Delta \stackrel { \mathrm { d e f } } { = } E _ { \pmb { x } } ( t _ { 1 } , t _ { 2 } ) ,\tag{5}
$$

where $t _ { 1 } , t _ { 2 }$ index the sequence of events closest to $\tau _ { 1 } , \tau _ { 2 }$

## 4. The E-3DGS Method

Our aim is to learn a 3D representation of a static scene using only a color event stream, where each pixel observes changes in brightness corresponding to one of the red, green, or blue channels according to a Bayer pattern, with known camera intrinsics $K _ { t } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ , and noisy initial poses $\begin{array} { r } { P _ { t } ~ \in ~ \mathbb { R } ^ { 3 \times 4 } } \end{array}$ , at reasonably high-frequency time steps indexed by t. Following 3DGS [9], we represent our scene by anisotropic 3D Gaussians. Our methodology comprises a technique to initialize Gaussians in the absence of a Structure from Motion (SfM) point cloud, adaptive event frame supervision of 3DGS, and a pose refinement module. An overview of our method is provided in Fig. 1.

Our E-3DGS method is not restricted to scenes of a certain size and can handle unbounded environments. It does not rely on any assumptions regarding the background color, type of camera motion, or speed. Thus, it ensures robust performance across a wide range of scenarios.

## 4.1. Event Stream Supervision

There are two main categories of approaches to learning 3D scene representations from event streams. Some apply the loss to single events [14] based on Eq. (4). Others use the sum of events $E _ { x } ( t _ { 1 } , t _ { 2 } )$ from Eq. (5). We choose the second approach, as rasterization in 3DGS is well suited to efficiently render entire images rather than individual pixels.

To optimize our Gaussian scene representation using event data, we can make a logical equivalence between the observed event stream and the scene renderings. To do so, we replace the true logarithmic intensities $L _ { x }$ in Eq. (5) with the rendered logarithmic intensities $\hat { L } _ { x }$ from our scene, and the times Ï with the camera poses $P _ { t }$ that were used to render the scene at the respective time steps. Following the approach used in [25], the log difference is then point-wise multiplied with a Bayer filter $F$ to obtain the respective color channel. We can finally calculate the error between the logarithmic change from our model and the actual change observed from the event stream, and define the following per-pixel loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { \pmb { x } } \left( t _ { 1 } , t _ { 2 } \right) = } \\ & { \left\| F \odot ( \hat { L } _ { \pmb { x } } ( P _ { t _ { 2 } } ) - \hat { L } _ { \pmb { x } } ( P _ { t _ { 1 } } ) ) - F \odot E _ { \pmb { x } } \left( t _ { 1 } , t _ { 2 } \right) \right\| _ { 1 } , } \end{array}\tag{6}
$$

where $\ " { â°}$ denotes pixelwise multiplication.

## 4.2. Frustum-Based Initialization

In the original 3DGS [9], the Gaussians are initialized using a point cloud obtained from applying SfM on the input images. The authors also experimented with initializing the Gaussians at random locations within a cube. While this worked for them with a slight performance drop, it requires an assumption about the extent of the scene.

Applying SfM directly to event streams is more challenging than RGB inputs [10] and exploring this aspect is not the primary focus of this paper. In the absence of an SfM point cloud, we use the randomly initialized Gaussians and extend this approach to unbounded scenes. To this end, we initialize a specified number of Gaussians (on the order of 104) in the frustum of each camera. This gives two benefits: 1) All the initialized Gaussians are within the observable area, and 2) We only need one loose assumption about the scene, which is the maximum depth $z _ { \mathrm { f a r } }$

## 4.3. Adaptive Event Window

Rudnev et al. [25] demonstrated in EventNeRF that using a fixed event window duration results in suboptimal reconstruction. They find that larger windows are essential for capturing low-frequency color and structure, and smaller ones are essential for optimization of finer high-frequency details. While they randomly sampled the event window duration, a drawback is that it does not consider the camera speed and event rate, thus the sampled windows may contain too many or too few events. As our dataset features variable camera speeds, we improve upon this by sampling the number of events rather than the window duration. To achieve this, for each time step we randomly sample a target number of events from within the range $[ N _ { \mathrm { m i n } } , N _ { \mathrm { m a x } } ]$ Given a time step t, we search for a previous time step $t _ { s }$ such that the number of events in the event frame $E ( t _ { s } , t )$ is approximately equal to the desired number.

When determining $N _ { \mathrm { m a x } }$ , we find that for values where details and low-frequency structure are optimal, 3DGS tends to get unstable and sometimes prunes away Gaussians in homogeneous areas. While this can be mitigated by choosing a much larger $N _ { \mathrm { m a x } }$ , this again deteriorates the details. Therefore, we propose a strategy to incorporate both, small and large windows. For each t, we choose two earlier time steps $t _ { s _ { 1 } }$ and $t _ { s _ { 2 } }$ . The ranges for sampling the event counts for both are empirically chosen to be $\begin{array} { r } { [ \frac { N _ { \mathrm { m a x } } } { 1 0 } , N _ { \mathrm { m a x } } ] } \end{array}$ and $\textstyle \left[ { \frac { N _ { m a x } } { 3 0 0 } } , { \frac { N _ { \mathrm { m a x } } } { 3 0 } } \right]$ . We then render frames from our model at times $t , t _ { s _ { 1 } }$ and $t _ { s _ { 2 } }$ , and use two concurrent losses for the event windows $E _ { x } \left( t _ { s _ { 1 } } , t \right)$ and $E _ { x } \left( t _ { s _ { 2 } } , t \right)$ .

## 4.4. As-Isotropic-As-Possible Regularization

In 3DGS, Gaussians are unconstrained in the direction perpendicular to the image plane. This lack of constraint can result in elongated and overfitted Gaussians. And while they may appear correct from the training views, they introduce significant artifacts when rendered from novel views by manifesting as floaters and distortions of object surfaces. We also observe that the lack of multi-view consistency and tendency to overfit destabilize the pose refinement.

To mitigate these issues, we draw inspiration from Gaussian Splatting SLAM [15] and SplaTAM [8], and apply isotropic regularization:

$$
\mathcal { L } _ { \mathrm { i s o } } = \frac { 1 } { | \mathcal { G } | } \sum _ { g \in \mathcal { G } } \left. \boldsymbol { S } _ { g } - \boldsymbol { \bar { S } } _ { g } \right. _ { 1 } ,\tag{7}
$$

where $\mathcal { G }$ is the set of Gaussians visible in the image. Eq. (7) imposes a soft constraint on the Gaussians to be as isotropic as possible. We find that it helps to improve pose refinement, minimizes floaters and enhances generalizability.

## 4.5. Pose Refinement

To obtain the most accurate results, we allow the poses to be refined during optimization by modeling the refined pose as $P _ { t } ^ { \prime } = P _ { t } ^ { e } P _ { t }$ , where $P _ { t } ^ { e }$ is an error correction transform. Instead of directly optimizing $P _ { t } ^ { e }$ as a $3 \times 3$ matrix, following Hempel et al. [6] we represent it as $[ r _ { 1 } \ r _ { 2 } \ T ]$ , where $r _ { 1 }$ and $r _ { 2 }$ represent two rotation vectors of the rotation matrix $R = \left[ r _ { 1 } \ r _ { 2 } \ r _ { 3 } \right]$ , while $T$ is the translation. We can then obtain the $P _ { t } ^ { e }$ matrix from the representation using Gram-Schmidt orthogonalization (see details in Supplement II), hence ensuring that during optimization, our error correction transform always represents a valid transformation matrix. $P _ { t } ^ { e }$ is initialized to be the identity transform. Since the loss function from $\operatorname { E q . }$ (6) depends on the camera pose as well, it allows us to use the same loss to backpropagate and obtain gradients for pose refinement.

As our goal is to refine the estimated noisy poses rather than perform SLAM, this training signal is sufficient for our needs. Moreover, we observe that poses tend to diverge with 3DGS due to the periodic opacity reset. To combat this, we impose a soft constraint with an additional pose regularization, that encourages the matrices $P _ { t } ^ { e }$ to stay close to the identity matrix I :

$$
\mathcal { L } _ { \mathrm { p o s e } } = \| P _ { t _ { s _ { 1 } } } ^ { e } - I \| _ { 2 } + \| P _ { t _ { s _ { 2 } } } ^ { e } - I \| _ { 2 } + \| P _ { t } ^ { e } - I \| _ { 2 } ,\tag{8}
$$

with all terms weighted equally.

## 4.6. Optimization

Eq. (6) defines the reconstruction loss per pixel for a single event frame. However, naively averaging these per-pixel losses over whole images leads to problems. For small event windows, most pixels have no events, which are not very informative but will then make up the majority of the loss. To address this, we compute separate averages of the losses for pixels with events $\mathcal { X } _ { \mathrm { e v s } }$ and pixels without events ${ \mathcal { X } } _ { \mathrm { n o e v s } } .$ These averages are then scaled by the hyperparameter $\alpha =$ 0.3 to obtain the complete weighted reconstruction loss:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { r e c o n } } \left( t _ { s } , t \right) = \displaystyle \frac { \alpha } { \left| \mathcal { X } _ { \mathrm { n o e v s } } \right| } \cdot \left( \sum _ { x \in \mathcal { X } _ { \mathrm { n o e v s } } } \mathcal { L } _ { x } \left( t _ { s } , t \right) \right) + } & { } \\ { + } & { \displaystyle \frac { 1 - \alpha } { \left| \mathcal { X } _ { \mathrm { e v s } } \right| } \cdot \left( \sum _ { x \in \mathcal { X } _ { \mathrm { e v s } } } \mathcal { L } _ { x } \left( t _ { s } , t \right) \right) . } \end{array}\tag{9}
$$

To obtain the final loss, we take a weighted sum of the reconstruction losses for the two event windows from Sec. 4.3 along with the isotropic and pose regularization:

$$
\begin{array} { r } { \mathcal { L } = ~ \lambda _ { 1 } \mathcal { L } _ { \mathrm { r e c o n } } \left( t _ { s _ { 1 } } , t \right) ~ + ~ \lambda _ { 2 } \mathcal { L } _ { \mathrm { r e c o n } } \left( t _ { s _ { 2 } } , t \right) } \\ { ~ + ~ \lambda _ { \mathrm { i s o } } \mathcal { L } _ { \mathrm { i s o } } ~ + ~ \lambda _ { \mathrm { p o s e } } \mathcal { L } _ { \mathrm { p o s e } } } \end{array} ,\tag{10}
$$

where $\lambda _ { 1 } , \lambda _ { 2 }$ and $\lambda _ { \mathrm { i s o } }$ are hyper-parameters. In our experiments, we use $\lambda _ { 1 } = \lambda _ { 2 } = 0 . 6 5 .$ , and $\lambda _ { \mathrm { i s o } }$ is set to 10 initially and reduced to 1 after $1 0 ^ { 4 }$ iterations.

## 5. Experimental Evaluation

## 5.1. Implementation details

We provide the full implementation details in the supplemental material. Running our method on a scene takes one to two hours (depending on the scene size) with a single NVIDIA GeForce RTX 3090.

<!-- image-->  
(a)

<!-- image-->  
(b)  
Figure 2. Two different views of the scene with inanimate objects assembled in the multi-view studio of MPI for Informatics.

## 5.2. Datasets

We next describe the new event datasets we provide to analyze large-scale scenes, along with the existing datasets that we use in the experiments.

E-3DGS-Real. Our real dataset was captured within a studio environment. The scene consists of a diverse set of objects, as shown in Fig. 2. We used a DAVIS346C color event camera to capture our scene with a resolution of 346 Ã 260. The contrast threshold settings were kept at their default values, which are symmetric. We capture multiple clips of the scene, each roughly 60â120s long with varying motion characteristics and levels of scene coverage. The captured data consists of the event stream and RGB images at 2.5 frames per second. The studio is equipped with 115 traditional cameras distributed uniformly across the walls and capturing 4K footage at 50 FPS. Similar to the approach of Millerdurai et al. and annotation of the EE3D-R (Real) dataset [18], we use these cameras to estimate and track the camera pose by detecting a checkerboard mounted to the event camera rig, providing tracking data at a frequency of up to 50 Hz. Note that in some timestamps the checkerboard is not detected due to occlusions and thus the 50 Hz is only the best case. The data from the external cameras is relevant for camera pose estimation, but cannot be used as ground truth because of the significantly different perspectives from the training views.

E-3DGS-Synthetic. For creating the synthetic dataset, we choose three scenes of UnrealEgo [1]. We rendered 60s clips of each scene at 1000 FPS. The scenes contain largescale environments and exhibit various types of surfaces, including reflections. We noticed that a few of the small highly reflective objects (e.g., metallic rods) cause unnatural aliasing in the renders, so we changed them to use diffuse materials. The event generation model from Sec. 4 was used to simulate event data from these high-fidelity frames. While we had access to pose data 1000 Hz, we downsampled it to 50 Hz to simulate a real-world setting in which the poses are estimated from externally captured RGB images. E-3DGS-Synthetic-Hard. This dataset is designed specifically to highlight and rigorously evaluate the key contributions of our method during the ablation study. To assess the significance of our pose refinement moduleâwhich cannot be quantitatively evaluated on the E-3DGS-Real datasetâ we introduce artificial noise into the E-3DGS-Synthetic dataset, which is carefully matched to the one observed in real data (see Supplement III for details). This allows us to assess the performance of our pose refinement module effectively. In addition to introducing noise, we also address the issue of camera speed variation. While the camera speed in the E-3DGS-Synthetic dataset generally stays within a narrow range, this does not fully test the capabilities of our adaptive event windows. To create a more challenging scenario, we varied the camera speed sinusoidally, with a ratio between its maximum and minimum speed of 100. This modification enables a more comprehensive evaluation of our adaptive event windows.

TUM-VIE. This dataset consists of recordings from a Prophesee Gen4 sensor [11]. RGB views from an externally calibrated camera are also provided. The camera extrinsics are tracked at 120 Hz. Two of the recordings have been used in Robust E-NeRF [14]; we train our method on these recordings, namely mocap-1d-trans and mocap-desk2 to compare with Robust E-NeRF. However, as also argued in Low and Lee [14], these recordings are not well suited for novel view synthesis since the captures are predominantly front-facing, with some small displacements either in circles or from side to side.

EventNeRF Datasets. EventNeRF [25] provides 360â¦ object-centric event data, which we use to show that our method also outperforms previous methods on objectcentric data. To be consistent with the original work, we evaluate our method on poses that are a part of the training trajectory instead of novel views, for our evaluation metrics to be comparable to theirs. We train our method on the synthetic sequences to perform the quantitative comparison. In these experiments, the background color is set to 159/255, following the original paper [25].

## 5.3. Evaluation Metrics

For E-3DGS-Real dataset, the RGB frames are of too low quality to be used for evaluation purposes, and, therefore, we only perform qualitative comparisons. With TUM-VIE, as suggested in Robust E-NeRF [14], it is not trivial to do the tone mapping correctly. Therefore, we do quantitative evaluation only with the synthetic datasets. For the evaluation on synthetic data, keeping in line with the previous literature, we adopt the following evaluation metrics:

â¢ Peak Signal-to-Noise Ratio (PSNR);

â¢ Learned Perceptual Image Patch Similarity (LPIPS) [31];

â¢ Structural Similarity Index Measure (SSIM).

## 5.3.1 Color Correction

As our method only learns logarithmic differences rather than absolute color intensities, there is an ambiguity in the reconstructed color balance and illumination of the scene. Hence, color needs to be adjusted, as otherwise, the evaluation metrics will be less meaningful. We correct predicted images using the following equation:

$$
L _ { c } ^ { \prime } = L ^ { \prime } + \left( \mathbb { E } [ L ] - \mathbb { E } [ L ^ { \prime } ] \right) ,\tag{11}
$$

where $L _ { c } ^ { \prime }$ is the color corrected logarithmic image and $^ { 6 6 } \mathbb { E } [ \cdot ] ^ { , \ast }$ is the expectation operator. Eq. (11) is applied separately to each color channel, which effectively aligns the per-channel logarithmic means of the predicted images with the ground-truth ones. Since in the synthetic setting, we already know the exact contrast threshold, there is no need for correcting the scale of the image as done in some previous works [14, 25]. Since we lack reference images for the real dataset, neither evaluation nor color correction is applicable to it. However, some minor color and contrast adjustments are manually made for better visualization.

## 5.4. Comparisons to Related Methods

RGB-Based Methods. We train Deblur-GS [28] on blurry RGB images from our E-3DGS-Real dataset to establish a reference using RGB inputs. We also convert the event stream to images using E2VID [23] and apply 3DGS (referred to as âE2VID + 3DGSâ). This method is evaluated on all E-3DGS datasets. To train both methods, we interpolate the camera poses at discrete time steps provided by the external tracking system, which is necessary because the pose timestamps do not align with the frame timestamps. We use Spherical Linear Interpolation (SLERP) for the rotations and Linear Interpolation (LERP) for the translations to obtain the camera poses for the images.

Event-Based Methods. For comparison with event-based methods, we train EventNeRF [25] on all E-3DGS datasets. To adapt it for our datasets, we normalize the camera poses within a unit sphere and following NeRF++ [30] added a background network to model areas outside the sphere, as the scene extent is unknown. Furthermore, the maximum event window length is increased by the factor of 10 to aid convergence (up to one second). We do not train our method on the synthetic dataset provided by Robust E-NeRF [14], as it is designed for extremely long refractory periods that are not observed in other datasets. However, we compare their method to ours on two sequences from TUM-VIE in Fig. 3, namely mocap-1d-trans and mocap-desk2.

## 5.4.1 Observations

The results of all evaluations are reported in Tables 1â2 and Figs. 3â6. As visible, our method consistently outperforms

Deblur-GS

EventNeRF

<!-- image-->

Figure 3. Comparison of E-3DGS against the baselines and ablation study on the E-3DGS-Real dataset. Deblur-GS, E2VID + 3DGS and EventNeRF suffer from various issues including blurring, floaters, and noise. In contrast, our method delivers clear details, such as the intricate structure of the sculptureâs face.
<table><tr><td rowspan="2">Method</td><td colspan="3">Company</td><td colspan="3">ScienceLab</td><td colspan="3">Subway</td><td colspan="3">Average</td></tr><tr><td>âPSNR</td><td>âLPIPS</td><td>âSSIM</td><td>âPSNR</td><td>âLPIPS</td><td>âSSIM</td><td>âPSNR</td><td>âLPIPS</td><td>âSSIM</td><td>âPSNR</td><td>âLPIPS</td><td>âSSIM</td></tr><tr><td>EventNeRF [25]</td><td>19.59</td><td>0.41</td><td>0.65</td><td>17.22</td><td>0.46</td><td>0.60</td><td>18.71</td><td>0.34</td><td>0.67</td><td>16.80</td><td>0.50</td><td>0.61</td></tr><tr><td>E2VID [23] + 3DGS [9]</td><td>9.79</td><td>0.37</td><td>0.48</td><td>11.86</td><td>0.38</td><td>0.54</td><td>9.79</td><td>0.40</td><td>0.43</td><td>10.48</td><td>0.38</td><td>0.49</td></tr><tr><td>E-3DGS (ours)</td><td>20.78</td><td>0.29</td><td>0.72</td><td>18.41</td><td>0.28</td><td>0.73</td><td>19.92</td><td>0.20</td><td>0.74</td><td>19.70</td><td>0.26</td><td>0.73</td></tr></table>

Table 1. Comparison of several methods on the E-3DGS-Synthetic dataset: We outperform the baselines by a large margin in all cases. Furthermore, E2VID + 3DGS shows lower PSNR but achieves better LPIPS than EventNeRF due to E2VIDâs frame reconstruction, which has poor color consistency but an adequate level of edge details (see Fig. 6). Green and yellow are the best and the second-best, respectively.

<table><tr><td rowspan=2 colspan=3>EventNeRF [25]          E-3DGS (ours)SceneâPSNRâLPIPSâSSIMâPSNRâLPIPSâSSIM</td></tr><tr><td rowspan=1 colspan=1>âPSNRâLPIPSâSSIM</td><td rowspan=1 colspan=1>âPSNRâLPIPSâSSIM</td></tr><tr><td rowspan=1 colspan=1>Chair</td><td rowspan=1 colspan=1>30.62  0.05  0.94</td><td rowspan=1 colspan=1>30.42   0.03  0.95</td></tr><tr><td rowspan=1 colspan=1>Drums</td><td rowspan=1 colspan=1>27.43  0.07  0.91</td><td rowspan=1 colspan=1>31.07  0.03  0.95</td></tr><tr><td rowspan=1 colspan=1>Ficus</td><td rowspan=1 colspan=1>31.94   0.05  0.94</td><td rowspan=1 colspan=1>34.08  0.02  0.96</td></tr><tr><td rowspan=3 colspan=1>HotdogLegoMaterials</td><td rowspan=1 colspan=1>30.26   0.04  0.94</td><td rowspan=1 colspan=1>30.79  0.03  0.96</td></tr><tr><td rowspan=2 colspan=1>25.84   0.13  0.8924.10   0.07  0.94</td><td rowspan=1 colspan=1>30.74  0.04  0.94</td></tr><tr><td rowspan=1 colspan=1>33.73  0.02  0.97</td></tr><tr><td rowspan=1 colspan=1>Mic</td><td rowspan=1 colspan=1>31.78   0.03  0.96</td><td rowspan=1 colspan=1>35.87  0.02  0.98</td></tr><tr><td rowspan=1 colspan=1>Average</td><td rowspan=1 colspan=1>28.85   0.06  0.93</td><td rowspan=1 colspan=1>32.39  0.03  0.96</td></tr></table>

Table 2. Comparisons on the synthetic EventNeRF dataset. Our method demonstrates significant improvements over EventNeRF across all evaluation metrics.

the baselines both on synthetic and real data. In the Event-NeRF object-centric datasets, our method shows clear superiority across almost all evaluation metrics. The only exception is a marginally lower PSNR score on the âChairâ scene, as detailed in Table 2. The general performance advantage is further backed by the qualitative results in Fig. 4, where our method produces more accurate reconstructions.

Similarly, on the E-3DGS-Synthetic dataset, E-3DGS significantly surpasses both EventNeRF and E2VID+3DGS by a wide margin; see Table 1. The qualitative results on the E-3DGS-Real dataset, highlighted in Fig. 3, further demonstrate our methodâs superior performance: Deblur-GS struggles with excessive blur; EventNeRF suffers from noise due to ray sampling and memory constraints, and E2VID+3DGS exhibits noisy Gaussians and floaters.

While Robust E-NeRF achieves higher local contrast, it struggles with global brightness consistency due to singleevent training; see Fig. 5. Our E-3DGS maintains consistent brightness across the scene, with only a slight reduction in local contrast. Note that we can observe some holes and floaters near the outer peripheries in Figs. 3 and 5. These effects are due to out-of-bound areas at the edges of the observations that occur as a result of the undistortion of the event stream.

## 5.5. Ablation Studies

To evaluate the effects of individual contributions, we do extensive qualitative and quantitative ablation studies. We primarily train different variants of our method on the E-3DGS-Real and E-3DGS-Synthetic-Hard datasets, focusing on the effects of four key components: ${ \mathcal { L } } _ { \mathrm { i s o } } , { \mathcal { L } } _ { \mathrm { p o s e } } ,$ , Pose Refinement (PR), and the Adaptive Event Window (AW).

For the ablation experiments without adaptive window, we use a maximum time interval $T _ { \mathrm { m a x } }$ instead of maximum events $N _ { \mathrm { m a x } }$ to sample the event windows. The value of $T _ { \mathrm { m a x } }$ is computed from $N _ { \mathrm { m a x } } .$ , such that the average event window size remains approximately similar.

<table><tr><td colspan="3">Components</td><td colspan="3">Company</td><td colspan="3">ScienceLab</td><td colspan="3">Subway</td><td colspan="3">Average</td></tr><tr><td>Liso</td><td>Lpose PR</td><td>AW</td><td>âPSNR</td><td>âLPIPS</td><td>â SSIM</td><td>âPSNR</td><td>âLPIPS</td><td>â SSIM</td><td>âPSNR</td><td>âLPIPS</td><td>âSSIM</td><td>âPSNR</td><td>âLPIPS</td><td>âSSIM</td></tr><tr><td>â</td><td></td><td>â</td><td>20.742</td><td>0.404</td><td>0.661</td><td>18.823</td><td>0.414</td><td>0.677</td><td>18.923</td><td>0.436</td><td>0.619</td><td>19.496</td><td>0.418</td><td>0.652</td></tr><tr><td></td><td>â</td><td>J</td><td>200.519</td><td>0.434</td><td>0.631</td><td>18.099</td><td>0.454</td><td>0.631</td><td>19.401</td><td>0.475</td><td>0.601</td><td>19.340</td><td>0.454</td><td>0.621</td></tr><tr><td></td><td></td><td></td><td>20.229</td><td>0.539</td><td>0.606</td><td>17.646</td><td>0.587</td><td>0.601</td><td>18.746</td><td>0.6200</td><td>0.569</td><td>18.874</td><td>0.582</td><td>0.592</td></tr><tr><td>â</td><td></td><td></td><td>20.667</td><td>0.427</td><td>0.642</td><td>18.354</td><td>0.440</td><td>00.657</td><td>18.742</td><td>0.440</td><td>0.606</td><td>19.2544</td><td>0.436</td><td>0.635</td></tr><tr><td>â</td><td></td><td></td><td>20.845</td><td>0.441</td><td>0.623</td><td>17.792</td><td>0.472</td><td>0.616</td><td>19.475</td><td>0.469</td><td>0.600</td><td>19.371</td><td>0.460</td><td>0.613</td></tr><tr><td></td><td></td><td></td><td>19.834</td><td>0.537</td><td>0.583</td><td>117.317</td><td>0.577</td><td>0.571</td><td>18.111</td><td>0.605</td><td>0.532</td><td>18.421</td><td>0.573</td><td>0.562</td></tr></table>

Table 3. Ablation study on the E-3DGS-Synthetic-Hard dataset. The overall tendency is that the performance declines when one of the components is removed, confirming their contribution to the overall performance. Notably, E-3DGS without AW consistently ranks second, while omitting $L _ { \mathrm { i s o } }$ often results in third place or close. (PR: Pose Refinement, AW: Adaptive Event Window). Green, yellow, and orange indicate the best, second-best, and third-best results, respectively.

<!-- image-->  
Figure 4. Comparison of E-3DGS vs. EventNeRF on the synthetic EventNeRF dataset. EventNeRF struggles with noise in the Drums sequence, blurriness in Ficus, and background artifacts in Lego and Materials sequences, while E-3DGS handles these issues well.

<!-- image-->  
Figure 5. Comparison of E-3DGS vs. Robust E-NeRF on the TUM-VIE dataset. While Robust E-NeRF achieves higher local contrast, it suffers from globally inconsistent brightness. E-3DGS produces consistent brightness across the scene, albeit with some detail loss (e.g., in the table texture of the mocap-desk2 sequence).

<!-- image-->  
Figure 6. Comparison of E-3DGS vs. baselines on the E-3DGS-Synthetic dataset. E2VID + 3DGS struggles with poor color reconstruction but captures edges and structure reasonably well. EventNeRF suffers from noise and a lack of sharpness. In contrast, our method delivers clear details and accurate colors, with only minor issues in certain areas (such as the coat on a chair in the ScienceLab sequence. Best viewed with zoom.

The results are reported in Table 3 and Figs. 3 and 6. Removing $L _ { i s o }$ results in a noticeable performance drop, but removing $L _ { i s o }$ and $L _ { p o s e }$ jointly leads to a much more significant decline. This is likely because $L _ { p o s e }$ prevents pose divergence in unstable conditions, while removal of $\boldsymbol { L } _ { i s o }$ causes instability due to overfitting. Similar effects could occur when combining $\boldsymbol { L } _ { i s o }$ with pose refinement.

## 6. Conclusion

We show that E-3DGS effectively combines the strengths of 3D Gaussian splatting and event-based supervision for 3D reconstruction and novel view synthesis of large-scale scenes. It significantly outperforms the baselines quantitatively and qualitatively, while being orders of magnitude faster. One aspect beyond the scope of this paper is lifting the requirement for camera pose initialization through an external process. We believe this work paves the way for robust and scalable large-scale scene reconstruction utilizing the advantages of event cameras to capture details in challenging conditions, such as low light and fast motion.

## References

[1] Hiroyasu Akada, Jian Wang, Soshi Shimada, Masaki Takahashi, Christian Theobalt, and Vladislav Golyanik. Unrealego: A new dataset for robust egocentric 3d human motion capture. In ECCV, pages 1â17. Springer, 2022. 5

[2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. ICCV, 2023. 2

[3] Zhang Chen, Zhong Li, Liangchen Song, Lele Chen, Jingyi Yu, Junsong Yuan, and Yi Xu. Neurbf: A neural fields representation with adaptive radial basis functions. In ICCV, pages 4182â4194, 2023. 2

[4] Fridovich-Keil and Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2022. 2

[5] Guillermo Gallego et al. Event-based vision: A survey. IEEE TPAMI, 44(01):154â180, 2022. 1

[6] Thorsten Hempel, Ahmed A. Abdelrahman, and Ayoub Al-Hamadi. Toward robust and unconstrained full range of rotation head pose estimation. IEEE TIP, 33:2377â2387, 2024. 4, 10

[7] Javier Hidalgo-Carrio, Guillermo Gallego, and Davide Â´ Scaramuzza. Event-aided direct sparse odometry. In CVPR, 2022. 2

[8] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. In CVPR, 2024. 4, 11

[9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4), 2023. 1, 2, 3, 4, 7, 11, 12

[10] Hanme Kim, Stefan Leutenegger, and Andrew J. Davison. Real-time 3d reconstruction and 6-dof tracking with an event camera. In ECCV, 2016. 2, 4

[11] Simon Klenk, Jason Chui, Nikolaus Demmel, and Daniel Cremers. Tum-vie: The tum stereo visual-inertial event dataset. In IEEE/RSJ IROS, pages 8601â8608. IEEE, 2021. 6

[12] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers. E-nerf: Neural radiance fields from a moving event camera. IEEE RA-L, 8(3):1587â1594, 2023. 1, 2

[13] Simone Klenk, Marvin Motzet, Lukas Koestler, and Daniel Cremers. Deep event visual odometry. In 3DV, 2024. 2

[14] Weng Fei Low and Gim Hee Lee. Robust e-nerf: Nerf from sparse & noisy events under non-uniform motion. In ICCV, pages 18335â18346, 2023. 2, 4, 6

[15] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian Splatting SLAM. In CVPR, 2024. 4, 11

[16] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H. Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In CVPR, 2023. 2

[17] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:

Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[18] Christen Millerdurai, Hiroyasu Akada, Jian Wang, Diogo Luvizon, Christian Theobalt, and Vladislav Golyanik. Eventego3d: 3d human motion capture from egocentric event streams. In CVPR, 2024. 5

[19] Christen Millerdurai, Diogo Luvizon, Viktor Rudnev, AndreÂ´ Jonas, Jiayi Wang, Christian Theobalt, and Vladislav Golyanik. 3d pose estimation of two interacting hands from a monocular event camera. In 3DV, 2024. 1

[20] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 41(4):102:1â102:15, 2022. 2

[21] Grigorios A Pavliotis. Stochastic processes and applications. Texts in applied mathematics, 60, 2014. 11

[22] Henri Rebecq, Timo Horstschaefer, Guillermo Gallego, and Davide Scaramuzza. Evo: A geometric approach to eventbased 6-dof parallel tracking and mapping in real time. IEEE RA-L, 2:593â600, 2017. 2

[23] Henri Rebecq, Rene Ranftl, Vladlen Koltun, and Davide Â´ Scaramuzza. High speed and high dynamic range video with an event camera. IEEE TPAMI, 2019. 6, 7, 12

[24] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In ICCV, 2021. 2

[25] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. Eventnerf: Neural radiance fields from a single colour event camera. In CVPR, 2023. 1, 2, 4, 6, 7, 11, 12

[26] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. CVPR, pages 5449â5459, 2022. 2

[27] Ayush Tewari et al. Advances in Neural Rendering. Eurographics, 2022. 1, 2

[28] Chen Wenbo and Liu Ligang. Deblur-gs: 3d gaussian splatting from camera motion blurred images. I3D, 7(1), 2024. 6, 12

[29] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, and Dahua Lin. Bungeenerf: Progressive neural radiance field for extreme multi-scale scene rendering. In ECCV, pages 106â122, 2022. 2

[30] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv:2010.07492, 2020. 6

[31] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6

# E-3DGS: Event-Based Novel View Rendering of Large-Scale Scenes Using 3D Gaussian Splatting

Supplementary Material

This supplement provides additional details and insights into the methods and experiments discussed in the main paper. In Sec. I, we elaborate on our frustum-based initialization, explaining the sampling strategy and how it ensures effective Gaussian placement in the scene. Sec. II provides further details on our pose refinement, specifically the use of Gram-Schmidt orthogonalization to maintain valid transformations during optimization. In Sec. III, we analyze the camera pose noise in the E-3DGS-Real dataset and describe the process we use to simulate realistic pose perturbations for the E-3DGS-Synthetic-Hard dataset. Sec. IV outlines the implementation details, including adjustments to the original 3DGS training schedule to improve convergence. Sec. V covers our evaluation, highlighting the measures we take to ensure reliable results, particularly for the ablation studies. Finally, we present a comprehensive comparison in Sec. VI showcasing additional visual results and ablation studies on the E-3DGS-Real, E-3DGS-Synthetic, and E-3DGS-Synthetic-Hard datasets. These experiments expand on the results from the main paper and further demonstrate the effectiveness of our method across different scenarios.

## I. Frustum-Based Initialization

As described in Sec. 4.2 of the main paper, our approach involves initializing a fixed number of Gaussians, denoted as $N _ { g } .$ If we have $N _ { t }$ camera poses, we distribute the Gaussians across these poses, resulting in $N _ { g } / N _ { t }$ Gaussians being initialized for each pose. The initialization process begins by sampling points within the camera frustum in normalized device coordinates (NDC). However, instead of uniformly sampling all three coordinates $( x , y , z )$ in NDC, we adopt a different strategy for depth (z-axis).

We observe that when depth was sampled directly in NDC, most Gaussians would cluster very close to the near plane $( z _ { \mathrm { n e a r } } )$ , leading to poor scene coverage. To address this, we sample the depth uniformly in camera coordinates between $z _ { \mathrm { n e a r } }$ and $z _ { \mathrm { f a r } }$ . This ensures a more even distribution of Gaussians across the entire depth range.

Once the depth is sampled in camera coordinates, it is converted into NDC. Next, the x and $y$ coordinates are sampled uniformly in NDC. With x, y, and z values now in NDC, we un-project them back into the world coordinates. This conversion gives us the final positions for the Gaussians in the 3D scene. Next, the entire process is repeated for each camera frustum associated with the given poses $P _ { t }$ , ensuring a comprehensive initialization across all views. Therefore, the distribution of Gaussians is effectively tied to

the observable scene regions.

## II. Pose Refinement and Gram-Schmidt Orthogonalization

In Sec. 4.5 of the main paper, we introduce our approach to pose refinement, where the refined pose $P _ { t } ^ { \prime }$ is modeled as $P _ { t } ^ { \prime } = P _ { t } ^ { e } P _ { t }$ , with $P _ { t } ^ { e }$ being an error correction transform. Rather than directly optimizing $P _ { t } ^ { e }$ as a 3Ã3 matrix, we represent it using two rotation vectors $r _ { 1 }$ and $r _ { 2 }$ and a translation vector $T ,$ , following the method of Hempel et al. [6]. This representation allows us to ensure that $P _ { t } ^ { e }$ remains a valid transformation matrix during optimization.

To maintain the orthogonality of the rotation matrix, we apply Gram-Schmidt orthogonalization to r1 and $r _ { 2 }$ to compute the final rotation matrix $R = [ r _ { 1 } ^ { \prime } , r _ { 2 } ^ { \prime } , r _ { 3 } ^ { \prime } ]$ . The process is as follows:

$$
\begin{array} { l } { { \displaystyle r _ { 1 } ^ { \prime } = \frac { r _ { 1 } } { \| r _ { 1 } \| } \mathrm { , } } } \\ { { \displaystyle r _ { 2 } ^ { \prime } = \frac { r _ { 2 } - \left( r _ { 1 } ^ { \prime } \cdot r _ { 2 } \right) r _ { 1 } ^ { \prime } } { \| r _ { 2 } - \left( r _ { 1 } ^ { \prime } \cdot r _ { 2 } \right) r _ { 1 } ^ { \prime } \| } \mathrm { , } } } \\ { { \displaystyle r _ { 3 } ^ { \prime } = r _ { 1 } ^ { \prime } \times r _ { 2 } ^ { \prime } \mathrm { , ~ a n d } } } \end{array}\tag{12}
$$

$$
P _ { t } ^ { e } = \left[ \begin{array} { c c c c } { { \vert } } & { { \vert } } & { { \vert } } & { { \vert } } \\ { { r _ { 1 } ^ { \prime } } } & { { r _ { 2 } ^ { \prime } } } & { { r _ { 3 } ^ { \prime } } } & { { T } } \\ { { \vert } } & { { \vert } } & { { \vert } } & { { \vert } } \\ { { 0 } } & { { 0 } } & { { 0 } } & { { 1 } } \end{array} \right] .
$$

Here, $r _ { 1 } ^ { \prime }$ is the normalized version of $r _ { 1 }$ , and $r _ { 2 } ^ { \prime }$ is obtained by subtracting the projection of $r _ { 2 }$ onto $r _ { 1 } ^ { \prime }$ and normalizing the result. The third vector $r _ { 3 } ^ { \prime }$ is calculated as the cross product of $r _ { 1 } ^ { \prime }$ and $r _ { 2 } ^ { \prime } ,$ , ensuring that the resulting rotation matrix is orthogonal. The final error correction matrix $P _ { t } ^ { e }$ is then constructed using these orthogonal vectors and the translation vector $T .$

This approach guarantees that the pose refinement remains valid throughout the optimization process, contributing to the stability and accuracy of our method.

## III. Pose Perturbation in E-3DGS-Synthetic-Hard

As described in Sec. 5.2 of the main paper, we provide the E-3DGS-Synthetic-Hard dataset that differs from E-3DGS-Synthetic in two aspects: 1) The camera speed is highly varied and 2) the camera extrinsics exhibit noise similar in characteristics to the noise observed in the real data. To quantify the camera pose noise in the E-3DGS-Real dataset, we compare the refined training camera trajectories with the initial trajectories. Our analysis reveals that these errors are time-correlated. Based on this observation and by examining the scale of these errors, we introduce synthetic perturbations in the E-3DGS-Synthetic dataset using a random walk with decay, specifically the OrnsteinâUhlenbeck process [21], which ensures the perturbations have zero mean while remaining time-correlated.

<!-- image-->

(a) Rotation errors for both E-3DGS-Real and E-3DGS-Synthetic-Hard show a similar error distribution.  
<!-- image-->  
(b) Larger translation errors are applied to E-3DGS-Synthetic-Hard, compared to those in E-3DGS-Real, to account for the larger scene size and ensure a sufficiently challenging difficulty level for meaningful ablation studies.  
Figure I. Comparison of estimated pose errors in the E-3DGS-Real dataset versus the synthetically introduced errors in the E-3DGS-Synthetic-Hard dataset. The synthetic perturbations are generated using an OrnsteinâUhlenbeck process to match the time-correlated nature and variance of the real data.

We calibrate the variance of the synthetic perturbations to match the rotation errors observed in the real data. For translation, we apply a higher level of perturbation, given that the synthetic scenes are significantly larger in scale than the real data. This adjustment ensures that translation errors are proportionally scaled, creating a comparable difficulty level for the ablation studies. The noise patterns are illustrated in Fig. I.

## IV. Implementation Details

Our codebase is based on 3DGS [9]. We train the method for $6 \cdot 1 0 ^ { 4 }$ instead of $\mathrm { 3 \cdot 1 0 ^ { 4 } }$ iterations, allowing the pose refinement to converge. The original paper performs both, densification and opacity resets of the Gaussians until

$1 . 5 \cdot 1 0 ^ { 4 }$ iterations. In our case, we perform opacity resets until $3 \cdot 1 0 ^ { 4 }$ and densification until $5 \cdot 1 0 ^ { 4 }$ iterations. From our analysisâwhile opacity resets are important to remove floatersâthey also hamper the reconstruction quality. Therefore, once the scene is reasonably converged, we stop resetting opacity and only densify the scene to get better reconstruction.

Furthermore, 3DGS uses the fixed threshold value 2 Â· $1 0 ^ { - 4 }$ to decide whether a Gaussian should be split up during the densification. We start the optimization with the same value, however, we linearly decrease it to $4 \cdot 1 0 ^ { - 5 }$ over $4 \cdot 1 0 ^ { 4 }$ iterations. First, this allows our method to refine the poses with larger Gaussians, providing more support, and second, reduce the threshold in later stages to obtain a more detailed reconstruction. We initialize $N _ { g } = 5 \cdot 1 0 ^ { 4 }$ Gaussians in all our trainings.

In the experiments with pose refinement, we restrict the number of spherical harmonics to one, as it allows for better pose refinement [8, 15]. For the experiments with perfect poses, we follow the original 3DGS approach and use three spherical harmonics. In all experiments, except those conducted with the EventNeRF dataset [25], we consistently use $N _ { \mathrm { m a x } } { = } 1 0 ^ { 6 }$ events for the window size. As sequences of the latter are very short and do not contain enough events for such large windows, we use $N _ { \mathrm { m a x } } { = } 1 0 ^ { 5 }$ for them. Training the full method takes one to two hours with a single NVIDIA GeForce RTX 3090, depending on the scene size.

## V. Further Evaluation Details (Ablations)

To ensure the reliability of the results, all ablation studies are conducted four times, with evaluation metrics averaged to provide more accurate insights and minimize the effects of coincidence. For the E-3DGS-Synthetic-Hard dataset, where the camera poses are perturbed, direct evaluation is not feasible due to slight misalignments between the learned 3D scene and the ground truth. To correct this, we first freeze the Gaussians and then refine the test poses with a small learning rate to ensure proper convergence. This alignment process allows the test views to match the ground truth accurately, enabling precise evaluation.

## VI. Additional Comparisons and Ablations

In this section, we expand on the main paper experiments by showing additional results on E-3DGS-Real, E-3DGS-Synthetic, and E-3DGS-Synthetic-Hard datasets. Fig. II demonstrates the performance of E-3DGS in comparison to Deblur-GS [28], E2VID [23]+3DGS [9] and Event-NeRF [25] on the E-3DGS-Real dataset. These baselines exhibit severe artifacts such as blur, floaters and noise. In the same figure, we also demonstrate the impact of the key components of our method. Removing $L _ { \mathrm { i s o } }$ leads to increased amounts of floaters and other artifacts. As the captured camera poses contain noise, pose refinement (PR) is crucial to achieve accurate results. Hence, without it, the model cannot produce accurate predictions, resulting in severe artifacts and blurriness. However, the model without the adaptive windows (AW) shows similar performance to the full model. That is likely due to the overall uniformity of the camera speeds in the used dataset, which diminishes the potential impact of adaptive event windows.

In Fig. III, we compare E-3DGS against EventNeRF [25] and E2VID [23]+3DGS [9] on E-3DGS-Synthetic dataset. Both baselines perform poorly: While E2VID+3DGS captures the edges and the general structure, it struggles with color representation, and EventNeRF reconstruction is much noisier and blurrier compared to our method. In contrast, our E-3DGS outperforms them, showing clear and sharp novel views with accurate color representation. Some issues are still observable but are mostly in less supervised areas, e.g., on the roof in ScienceLab or Subway scenes.

Lastly, Fig. IV visualizes results of the ablation study on the E-3DGS-Synthetic-Hard dataset. In comparison to E-3DGS-Synthetic, this dataset has artificially added camera extrinsics noise, which we describe in Sec III, and drastically increased camera speed variation (Sec. 5.2). While these changes make obtaining high reconstruction quality more difficult, our full method still works well, outperforming all ablated models. As on the E-3DGS-Real, removing $L _ { \mathrm { i s o } }$ results in severe artifacts (e.g., in the first view of Company or in the second view of Subway). E-3DGS-Synthetic-Hard dataset has camera pose noise, and, hence, using pose refinement (PR) is important, as removing it results in blurriness and artifacts. Removing the adaptive event windows (AW) leads to deterioration; e.g., the method without AW exhibits artifacts on the sofa in the first view of the Company sequence that are absent in the results of the full method. It is also noteworthy that while all ablated models struggle with the second view of the Subway sequence, the full method, nevertheless, achieves a better result: The structure is clearer and more recognizable with fewer artifacts.

<!-- image-->  
Figure II. Comparison of E-3DGS against the baselines and ablation study on E-3DGS-Real. As observed in the main paper, Deblur-GS, E2VID + 3DGS, and EventNeRF exhibit issues such as blurring, floaters and noise. Notably, the ablation study highlights the impact of removing key components. Removing $L _ { \mathrm { i s o } }$ leads to an increase in floaters and artifacts. In contrast, the experiment without adaptive event windows (AW) shows little difference in performance. This is likely due to the relatively consistent camera speeds in this dataset that reduce the potential benefits of AW.

<!-- image-->  
Figure III. Comparison of E-3DGS vs. baselines on the E-3DGS-Synthetic dataset. As observed in the main paper, E2VID + 3DGS struggles with poor color reconstruction but captures edges and structure reasonably well. EventNeRF suffers from noise and a lack of sharpness. In contrast, our method delivers clear details and accurate colors, with issues mainly confined to less observed areas, such as the roof. Best viewed with zoom.

<!-- image-->  
Figure IV. Ablation study of E-3DGS on the E-3DGS-Synthetic-Hard dataset. The increased difficulty of this dataset leads to overall performance deterioration compared to E-3DGS-Synthetic, but our full method still performs well. The version without the adaptive event window (AW) is closest to the full method but shows more artifacts. For example, in the first column of the Company sequence, the sofa shows some artifacts in the AW-removed version that are absent in the full method. Similar minor artifacts are visible elsewhere. The second column of the Subway sequence is interesting, as all versions struggle with reconstructing it. Even so, the full method demonstrates a better structure and fewer artifacts than the others.