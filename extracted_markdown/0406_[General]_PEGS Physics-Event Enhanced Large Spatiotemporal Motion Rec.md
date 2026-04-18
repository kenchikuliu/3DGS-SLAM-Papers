# PEGS: Physics-Event Enhanced Large Spatiotemporal Motion Reconstruction via 3D Gaussian Splatting

Yijun Xuâ 1, Jingrui Zhangâ 2, Hongyi Liu1, Yuhan Chen3, Yuanyang Wang2, Qingyao Guo2, Dingwen Wang2, Lei Yu\*4 and Chu He\*1

1School of Electronic Information, Wuhan University 2School of Computer Science, Wuhan University 3College of Mechanical and Vehicle Engineering, Chongqing University 4School of Artificial Intelligence, Wuhan University

## Abstract

Reconstruction of rigid motion over large spatiotemporal scales remains a challenging task due to limitations in modeling paradigms, severe motion blur, and insufficient physical consistency. In this work, we propose PEGS, a framework that integrates Physical priors with Event stream enhancement within a 3D Gaussian Splatting pipeline to perform deblurred target-focused modeling and motion recovery. We introduce a cohesive triple-level supervision scheme that enforces physical plausibility via an acceleration constraint, leverages event streams for high-temporal resolution guidance, and employs a Kalman regularizer to fuse multi-source observations. Furthermore, we design a motion-aware simulated annealing strategy that adaptively schedules the training process based on real-time kinematic states. We also contribute the first RGB-Event paired dataset targeting natural, fast rigid motion across diverse scenarios. Experiments show PEGSâs superior performance in reconstructing motion over large spatiotemporal scales compared to mainstream dynamic methods.

## 1 Introduction

Dynamic reconstruction acts as the driving force of modern film, interactive gaming and virtual reality. Integrating generative approaches [29, 4, 22] enables breakthroughs in controllable dynamic synthesis, pushing visual expression beyond imaginative boundaries. Concurrently, subtle deformations such as elastic collision [48] and delicate biological tremors [15] can be meticulously captured.

<!-- image-->  
Figure 1: PEGS takes a monocular blurry video and event streams as input to perform 4D motion recovery. The framework first focuses on the target for deblurred 3D Gaussian reconstruction, then estimates the SE-3 transformations of the motion sequence. By integrating physical priors with event enhancement, PEGS effectively reconstructs large motions, producing outputs applicable to downstream tasks.

Yet, a dark cloud still looms: Reconstruction of complex rigid motion over long spatiotemporal spans remains an underexplored field. The difficulties lie in: (1) modeling paradigms; (2) motion blur; (3) physical consistency.

Most existing paradigms [25, 20, 44, 36, 21] are tailored for non-rigid motions that exhibit limited spatial range and temporal duration. The discrete sampling deformation framework struggles to represent intense nonlinear motion, often resulting in severe artifacts and trajectory fragmentation [36]. In parallel, the commonly used benchmarks [20, 25, 24] share these shortcomings: most are characterized by minimal clear deformation and are constructed under idealized assumptions. As a result, the focus has shifted towards optimizing photometric similarity, leaving the challenge of accurate motion recovery comparatively underexplored.

Motion blur stems from the inherent imaging limitations of RGB cameras. Event camera, as a bio-inspired sensor, offers a promising solution to this issue. By asynchronously recording brightness changes and generating microsecondlevel event pulses, it exhibits low latency and high dynamic range. Some studies [6, 35, 40, 8, 41, 19] integrate event streams into neural rendering under blur or low-light conditions. The applicability of these approaches is primarily limited to static scenes or subtle deformation blur, failing to handle challenging cases such as fast moving object.

Another challenge arises from the absence of physical consistency. Relying solely on photometric supervision struggles to constrain the vast 3D solution space, especially for motions across large spatiotemporal spans. NeRF-based implicit methods [32, 20, 24] encode motion cues within black-box network, making it difficult to model physical interactions. In contrast, 3DGS [16] provides structural support for scene understanding through its explicit representation. Consequently, some works [15, 9, 22, 39, 27] incorporate off-the-shelf models to enhance physical realism. However, the model stacking essentially layers new approximations upon existing ones, thereby introducing compounded uncertainty. More critically, it bypasses the first principles in physics, ultimately reducing the reconstruction task to an uninterpretable compensatory computation.

To address these issues, we propose PEGSâa Gaussian Splatting framework that integrates physical priors with event enhancement, aiming to achieve large spatiotemporal motion reconstruction. Unlike deformation field works, we leverage the temporal continuity of video sequences and the explicit Gaussian representation to directly estimate per-frame SE-3 affine transformations. We design a cohesive triple-level supervision scheme: (1) To overcome the physical distortion in vision-only approaches, we propose an acceleration consistency constraint derived from Newtonian dynamics for precise trajectory estimation. (2) To tackle blurring caused by fast motion, we introduce event stream supervision to enhance the recovery of dynamic details. (3) Furthermore, we devise a Kalman regularizer to unify multi-source observations and minimize estimation errors.

Overall, our main contributions can be outlined as:

â¢ We introduce PEGS, a framework for large spatiotemporal motion reconstruction. Particularly, we contribute the first RGB-Event paired dataset targeting natural motion across large spans with diverse scenarios.

â¢ We developed a triple-level supervision scheme with acceleration constraints, event enhancement, and Kalman regularization to tackle the critical issues of physical absence, motion blur, and multi-source observation.

â¢ We developed a motion-aware simulated annealing (MSA) strategy that adaptively schedules the training process based on real-time velocity and displacement, thereby enhancing convergence efficiency and stability.

## 2 Related Work

## 2.1 Dynamic Neural Rendering

Current dynamic neural rendering can be categorized into: (1) Point deformation [25]; (2) Time-aware volume rendering [20, 32]; (3) Gaussian deformation[44, 36, 49, 12]. A common issue lies in the difficulty of deformation fields in modeling large inter-frame displacements. Furthermore, the networks prioritize fitting appearance similarity while neglecting geometric accuracy. A typical manifestation is that when large areas of approximately uniform color exist, Gaussian splatting tend to undergo disordered drift within these color-homogeneous regions [21, 9].

## 2.2 Physics-Enhanced Dynamic 3DGS

Integrating physical assistance to enhance 3DGS has gained widespread adoption. Works such as [9, 15, 50] leverage off-the-shelf monocular depth estimation [28] for scene initialization or sparse reconstruction. [48] devises a Spring-Mass model to simulate the falling and collision behaviors of elastic objects. Tailored for 3D dataset generation, [22] introduces the PyBullet [5] to emulate objectâs placement. [4, 39] improve pose estimation by enforcing constraints on scale consistency and local rigidity. Current approaches largely rely on off-the-shelf models to alleviate the ill-posed problems inherent in vision-only systems. However, pretrained engines exhibit a sharp drop in confidence when confronted with unusual scenarios, and their black-box nature renders errors untraceable. Differently, we propose a novel strategy that is directly grounded in the first principles of physics [27], rather than depending on external models.

## 2.3 Event-based Neural Reconstruction

Event-based neural rendering has emerged as a promising direction for 3D scene perception. Early methods [26, 2] combined events with RGB to recover clear 3D scenes from blurred inputs, and [38] extended this to dynamic settings. Recently, focus has shifted to integrating events into 3DGS [16], leading to methods like [6, 35, 40] for static scenes, and [8, 41, 19] for dynamic ones. Nevertheless, these approaches are still constrained to small camera motions or minor object deformations. Another bottleneck lies in dataset collection. Most studies are verified on synthetic RGB-E pairs [8, 19], with images rendered in Blender [1] and events simulated via V2E [13]. Though some efforts have been made to collect real-world data [37, 41], they are typically captured using controlled turntables, resulting in limited scene complexity and motion patterns. In contrast, we target a more challenging settingâdeblurring under large span, fast motion, and introduce a data acquisition scheme detailed in 5.1.

## 3 Preliminaries

## 3.1 3D Gaussian Splatting

As a point-based representation, 3DGS [16] structures a scene into a set of Gaussian ellipsoids, each characterized by the center location x and a 3D covariance matrix Î£:

$$
\pmb { \mathcal { G } } ( \pmb { x } ) = e x p ( - \frac { 1 } { 2 } \pmb { x } ^ { T } \Sigma ^ { - 1 } \pmb { x } ) ,\tag{1}
$$

where the $\pmb { \Sigma } = \pmb { R } \pmb { S } \pmb { S } ^ { T } \pmb { R } ^ { T }$ is parameterized by scaling matrix S and rotation matrix R.

During the splatting, Gaussians are projected onto a plane. This relies the 2D covariance matrix $\Sigma ^ { \prime } = J W \Sigma W ^ { T } J ^ { T }$ in the camera coordinate, where W is the view transformation matrix, and J is the Jacobian matrix of the affine projection. For rendering, the color C of each pixel is determined by the combined effect of the corresponding N depth-ordered Gaussiansâ color $c _ { i }$ and opacity $\alpha _ { i } \mathbf { : }$

$$
C = \sum _ { i \in N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) .\tag{2}
$$

## 3.2 Event Signals

When an event sensor detects that the brightness change at pixel $( x , y )$ exceeds the threshold $\epsilon ,$ it asynchronously reports an event polarity $p \mathrm { : }$

$$
p = { \left\{ \begin{array} { l l } { + 1 } & { { \mathrm { i f } } \ l o g ( x , y , \tau ) - l o g ( x , y , \tau ^ { \prime } ) > \epsilon } \\ { - 1 } & { { \mathrm { i f } } \ l o g ( x , y , \tau ^ { \prime } ) - l o g ( x , y , \tau ) > \epsilon , } \end{array} \right. }\tag{3}
$$

where $\tau _ { 0 }$ and Ï denotes the first and the last timestamp of the observed event respectively. Introducing a unit integral impulse function $\delta ( \tau )$ , the discrete polarities can be expressed as a continuous event signal: $\begin{array} { r } { \pmb { e } ( \tau ) = p \delta ( \tau ) } \end{array}$ . Based on this, the proportional intensity change over the time interval can be computed as the integral of the event stream:

$$
E ( \tau ) = \int _ { \tau _ { 0 } } ^ { \tau } e ( \tau ) d \tau ,\tag{4}
$$

then the image IÏ captured at time $\tau$ can be computed as:

$$
I ( \tau ) = I ( \tau _ { 0 } ) \cdot e x p ( \epsilon E ( \tau ) ) .\tag{5}
$$

According to the assumption of the Event-based Double Integral (EDI) model [23]: A blurry image is the average of N latent sharp images over the exposure time. Therefore, given the threshold Ïµ, the blurry image $\scriptstyle { I _ { \tau } } .$ , and the corresponding event stream $E ( \tau )$ , latent sharp images at any instant during the exposure time can be recovered.

## 4 Method

The PEGS framework consists of two tightly coupled stages: target modeling and motion recovery (see Fig.2). The first stage leverages event-based deblurring and an improved point density control strategy to reconstruct a target-focused 3D Gaussian representation from blurry RGB inputs. The subsequent motion recovery stage estimates the complete SE-3 motion trajectory through a triple-level supervision strategy comprising: an acceleration consistency constraint derived from Newtonian dynamics, high-temporal resolution event stream supervision, and Kalman regularization that fuses multi-source observations. Furthermore, we introduce a MSA strategy that adaptively schedules the training process to enhance convergence efficiency and stability.

## 4.1 Target Modeling

The task of this stage is to reconstruct target-focused Gaussians from blurred input. The pipeline comprises event-based deblurring, target centering, GS reconstruction incorporating an improved point density control strategy, and finally, inversely registering the Gaussians from the centered coordinate back to the original complete scene.

<!-- image-->  
Figure 2: PEGS reconstructs a target-focused 3D Gaussian scene from blurry images and then estimates its full SE(3) motion trajectory. This is achieved by combining event-based deblurring with a triple-level motion supervision strategy that enforces acceleration consistency, event stream alignment, and Kalman regularization. A motion-aware simulated annealing scheduler further boosts training convergence.

Image Deblurring. The preliminary workflow of 3DGS [16] involves initial point cloud and camera pose estimation, which typically requires COLMAP package [31]. Unfortunately, this process often fails for blurry image sequences [26, 6]. For image deblurring, based on EDI model [23] and MAENet [11], we divide the exposure duration corresponding to each blurry image into N timestamps, thereby obtaining N â 1 corresponding event bins $\{ B _ { i } \} _ { i = 1 } ^ { N - 1 }$ , a nd subsequently recover N deblurred images $\{ I _ { i } \} _ { i = 1 } ^ { N }$

Centralization. Based on the deblurred image, we first employ the SAM [17] to separate dynamic objects from static backgrounds. Subsequently, by establishing an object-centric normalized coordinate [4], we achieve the decomposition of complex projectile motion: the translational component is eliminated, and the objectâs autorotation is equivalent to pseudo multi-view observations, which are then fed into [31] for initializing N pesudo camera poses $\{ P _ { i } \} _ { i = 1 } ^ { N }$ The process mitigates the negative impact of complex motion on the accuracy of pseudo-view estimation, and facilitates subsequent centroid computation for physical constraints.

Reconstruction. Following the 3DGS [16], we use a set of Gaussian kernels to explicitly reconstruct the object. Throughout the optimization, we design the loss function with reference to [6, 46, 16]. Specifically, we average N rendered deblurred images $\{ \hat { I } _ { i } \} _ { i = 1 } ^ { N }$ for each corresponding blurry input, thus obtaining the rendered blurry image $\hat { I } _ { b l u r }$ :

$$
\hat { I } _ { b l u r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { I } _ { i } ,\tag{6}
$$

then the real blurry input $\pmb { I _ { b l u r } }$ is used as the supervisory signal to compute the loss:

$$
\mathcal { L } _ { b l u r } = ( 1 - \lambda ) \Vert I _ { b l u r } - \hat { I } _ { b l u r } \Vert _ { 1 } + \lambda \mathcal { L } _ { D - S S I M } .\tag{7}
$$

Notably, the original 3DGS [16] framework exhibits limitations in geometric accuracy when modeling focal objects, where ill-defined boundaries adversely affect subsequent centroid localization. To address this, we implement an improved point density control strategy [43] to hard prune the jagged edges and spiky artifacts, while eliminating isolated points that float distant from the target.

Up to this point, we have reconstructed $\scriptstyle { \mathcal { G } } _ { 0 }$ at a centralized scale with refined appearance and geometric representation from blurry inputs.

Registration. Before performing motion trajectory recovery, we register $\mathscr { G } _ { 0 }$ under the centered scale to the original complete dynamic scene. We use a set of learnable affine transformations $\boldsymbol { T _ { r e g } } = [ r , t , s ]$ to achieve this, i.e., quaternion rotation $\pmb r \in \mathbb R ^ { 4 }$ , translation vector $\pm \in \mathbb { R } ^ { 3 }$ , and constant scale s. Thus the target Gaussian $\pmb { \mathcal { G } } _ { 1 }$ can be represented as $\mathbf { \Delta } \mathbf { T } _ { r e g } \odot \mathbf { \mathcal { G } } _ { 0 }$ , and the optimization is to minimize $\mathcal { L } _ { R e g }$ between the rendered image of $\pmb { \mathcal { G } } _ { 1 }$ and the first deblurred frame $\scriptstyle { I _ { 0 } }$ of original dynamic scene:

$$
\begin{array} { r } { \pmb { T } _ { r e g } = \pmb { \mathrm { a r g m i n } } \mathcal { L } _ { r e g } ( R e n d e r ( \pmb { \mathcal { G } } _ { 1 } ) , I _ { 0 } ) . } \end{array}\tag{8}
$$

In addition to $\mathcal { L } _ { G S } ~ [ 1 6 ]$ , we further apply VGG loss ${ \mathcal { L } } _ { V G G }$ and grid-based total variation loss LT V [48, 36] to achieve a better awareness for accurate registration. The total registration loss can be formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { r e g } = \lambda _ { 1 } \mathcal { L } _ { G S } + \lambda _ { 2 } \mathcal { L } _ { V G G } + \lambda _ { 3 } \mathcal { L } _ { T V } . } \end{array}\tag{9}
$$

During the entire modeling phase, we reconstruct a clear Gaussians $\pmb { \mathcal { G } } _ { 1 }$ of the target from blurry inputs. We will proceed to recover the full motion sequence based on this.

## 4.2 Motion Recovery

This stage recovers the full motion sequence through frame-wise estimation of the objectâs SE(3) pose transformations. As mentioned before, in a video sequence, each blurry input at time t corresponds to N deblurred images with a constant exposure interval ât, which yields N â 1 pose changes $\hat { P } _ { t } = [ \hat { R } _ { t } , \hat { \pmb { T } } _ { t } ]$ to be learned.

In addition to the fundamental image-level blur loss (consistent with Eq.7, but applying the complete non-centralized blurry input as ground-truth), we introduce a cohesive triple-level supervision scheme: (1) Physics-level Acceleration Loss to enforce temporal smoothness and physical consistency; (2) Event-level Loss to supervise high temporal resolution motion details; and (3) Fusion-level Kalman Loss that estimates the optimal pose from multi-source observation as a regularization term. Details are as follows:

Acceleration Consistency Loss. An object moving in a constant force field maintains constant acceleration. Based on this, we impose smoothness constraints on the acceleration to ensure the estimated displacements adhere to physical consistency. Firstly, we calculate the objectâs center of mass. Since we have implemented point density control and isotropic processing, the centroid can be approximated as:

$$
\sigma = \frac { \sum _ { i = 1 } ^ { M } R _ { i } ^ { 3 } \mu _ { i } } { \sum _ { i = 1 } ^ { M } R _ { i } ^ { 3 } } ,\tag{10}
$$

where $R _ { i }$ and $\pmb { \mu _ { i } }$ denote the radius and position of total M kernels. In turn, the objectâs acceleration at timestamp t can be calculated from the displacement of centroid as:

$$
\pmb { a } _ { t } = \frac { \left[ \pmb { \sigma } _ { ( t + \Delta t ) } - \pmb { \sigma } _ { t } \right] - \left[ \pmb { \sigma } _ { t } - \pmb { \sigma } _ { ( t - \Delta t ) } \right] } { ( \Delta t ) ^ { 2 } } ,\tag{11}
$$

where $\Delta t$ denotes the time interval between two adjacent deblurred frames. Based on this, we minimize the acceleration variation between consecutive timestamps:

$$
\mathcal { L } _ { A c c } = \sum _ { t = 0 } ^ { T } \lVert \pmb { a } _ { t } - \pmb { a } _ { t + \Delta t } \rVert _ { 2 } ^ { 2 } .\tag{12}
$$

Event Stream Loss. Leveraging the microsecond-level resolution of event cameras, we further impose constraints on the continuous motion recovery. As mentioned earlier, each blurred frame corresponds to $N - 1$ event bins $\{ B _ { i } \} _ { i = 1 } ^ { N - 1 }$ and N deblurred frames $\{ I _ { i } \} _ { i = 1 } ^ { N }$ . According to [23], we integrate these to obtain $N - 1$ event maps $\{ E _ { i } \} _ { i = 1 } ^ { N - 1 }$ , which serve as the ground truth supervision signals. Correspondingly, we synthesize $N - 1$ simulated event sequences $\{ \hat { E } _ { i } \} _ { i = 1 } ^ { N - 1 }$ by computing the intensity changes between consecutive N rendered images [6]. The estimation of motion details is thereby constrained by comparing the differences between the simulated and real event maps:

$$
\mathcal { L } _ { E v e n t } = \sum _ { i = 1 } ^ { N - 1 } \lVert \pmb { E } _ { i } - \hat { \pmb { E } _ { i } } \rVert _ { 1 } .\tag{13}
$$

Kalman Fusion Loss. Though event signals provide sufficient motion cues, the data is inherently sparse with unavoidable noise. On the other hand, dynamic models offer strong constraints on object motion, but real-world disturbances such as air resistance and non-ideal force fields can cause deviations in physical predictions. To this end, we employ a Kalman Fusion (KF) [34] to combine the physical model with event stream observations, thereby obtaining an optimal estimation reference. Detailed steps are as follows:

Step1 - Initialization: We begin by mathematically modeling the KF process:

â¢ State vector. We define the objectâs instantaneous displacement and velocity as the state vector $\pmb { X } _ { t } = [ \pmb { T } _ { t } , \pmb { v } _ { t } ]$

â¢ System model. Based on acceleration constraints (described by Eq.12), the displacement at current moment can be predicted from the previous optimal estimate.

â¢ Observation. We feed the clear images deblurred by events into RAFT [10] to compute the optical flow, from which the observed instantaneous displacement $\tilde { \pmb { T } } _ { t }$ is mapped.

<!-- image-->  
Figure 3: MSA strategy. Bottom: An object accelerates in a constant force field, with the velocity v and displacement s vary at each timestamp. Top: The red curve traces the initial learning rate, and blue points signify its value after exponential decay.

Step2 - Prediction: Based on the system model and the optimal estimate $X _ { t \mid t - \Delta t }$ from the previous moment, we predict the current state vector $X _ { t } \colon$

$$
\begin{array} { r l } & { X _ { t \mid t - \Delta t } = F X _ { t - \Delta t } + G a _ { t } + w _ { t } , } \\ & { F = \left[ \begin{array} { l l } { I } & { \Delta t } \\ { 0 } & { I } \end{array} \right] , G = \left[ \begin{array} { l } { \frac { 1 } { 2 } ( \Delta t ) ^ { 2 } } \\ { ~ \Delta t } \end{array} \right] , Q = \left[ \begin{array} { l l } { \sigma _ { T } ^ { 2 } } & { ~ 0 } \\ { 0 } & { \sigma _ { v } ^ { 2 } } \end{array} \right] . } \end{array}\tag{14}
$$

The state transition is governed by F . G denotes the control input matrix, and the control term $\mathbf { } \mathbf { a } _ { t }$ is computed from Eq.11. Process noise ${ \pmb w } _ { t }$ (comprising $\sigma _ { T } ^ { 2 }$ and $\sigma _ { v } ^ { 2 } )$ , which characterizes model uncertainty, has its covariance matrix defined as $Q = \mathbb { E } [ { \pmb w } _ { t } { \pmb w } _ { t } ^ { T } ]$ The predicted covariance matrix is then obtained through the following relation:

$$
\boldsymbol { C } _ { t | t - \Delta t } = \boldsymbol { F } \boldsymbol { C } _ { t - \Delta t } \boldsymbol { F } ^ { T } + \boldsymbol { Q } .\tag{15}
$$

Step3 - Correction: In the update step, we introduce observation to correct the predicted state: $\pmb { z } _ { t } = \pmb { H } \tilde { \pmb { T } } _ { t } + \pmb { u } _ { t }$ , where $H = [ I , 0 ]$ is the observation matrix, $\pmb { R } = \mathbb { E } [ \pmb { u } _ { t } \pmb { u } _ { t } ^ { T } ]$ is the covariance of event observation noise.

Step4 - Updation: Based on prediction and correction, we can dynamically update the Kalman gain $\pmb { K } _ { t }$ and the posterior covariance $C _ { t } \mathrm { : }$

$$
\pmb { K } _ { t } = \pmb { C } _ { t | t - \Delta t } \pmb { H } ^ { T } ( \pmb { H } \pmb { C } _ { t | t - \Delta t } \pmb { H } ^ { T } + \pmb { R } ) ^ { - 1 } ,\tag{16}
$$

$$
\begin{array} { r } { C _ { t } = ( I - K _ { t } H ) C _ { t | t - \Delta t } . } \end{array}\tag{17}
$$

Finally, we obtain the optimal estimated value at the current timestamp:

$$
X _ { t } = X _ { t | t - \Delta t } + K _ { t } ( z _ { t } - H { X _ { t | t - \Delta t } } ) .\tag{18}
$$

Therefore, the translation component $\mathbf { \delta } _ { \mathbf { \mathcal { T } } _ { t } }$ from the optimal estimate $X _ { t }$ will be updated to the reference signal $P _ { t }$ to supervise the learning of the target $\hat { P } _ { t }$

$$
\mathcal { L } _ { K F } = \sum _ { t = 0 } ^ { T } \lVert \pmb { P } _ { t } - \hat { \pmb { P } } _ { t } \rVert _ { 2 } ^ { 2 } .\tag{19}
$$

<table><tr><td>Datasets</td><td>Source</td><td>Phy.</td><td>Mono.</td><td>Mod.</td><td>Type</td></tr><tr><td>D-NeRF [25]</td><td>S</td><td>Ã</td><td>â</td><td>RGB</td><td>Def.</td></tr><tr><td>Neural3D [20]</td><td>S</td><td>Ã</td><td>Ã</td><td>RGB</td><td>Def.</td></tr><tr><td>PEGASUS [22]</td><td>S</td><td>PyBullet</td><td>â</td><td>RGB</td><td>Rigid</td></tr><tr><td>SpringGS b [48]</td><td> $\mathrm { 6 S + 3 R }$ </td><td>Hooke&#x27;s Law</td><td>Ã</td><td>RGB</td><td>Def.</td></tr><tr><td>DE-NeRF b [38]</td><td> $\mathrm { 3 S + 3 R }$ </td><td>Ã</td><td>Ã</td><td>RGB+E</td><td>Def.</td></tr><tr><td>E-4DGS [8]</td><td> $\mathrm { 8 S + 3 R }$ </td><td>Ã</td><td>Ã</td><td>RGB+E</td><td>Def.</td></tr><tr><td>Event-boost [42]</td><td> $\mathrm { 8 S + 4 R }$ </td><td>Ã</td><td>Ã</td><td>RGB+E</td><td>Def.</td></tr><tr><td>PEGS</td><td> $6 \mathrm { S } + 6 \mathrm { R }$ </td><td>Newton&#x27;s Law</td><td>â</td><td>RGB+E</td><td>Rigid</td></tr></table>

Table 1: Datasets comparison (Phy.-Whether physical prior is introduced; Mono.- Whether monocular; Mod.-Modality; S-Synthetic and R-Real; Def.-Deformable).

Essentially, the KF process performs Bayesian estimation of event observations with a Newtonian mechanics model, thereby providing an adaptively calibrated regularization with multi-source observation fusion for training.

In summary, the total loss for motion recovery stage is:

$$
\mathcal { L } _ { T o t a l } = \lambda _ { 1 } \mathcal { L } _ { B l u r } + \lambda _ { 2 } \mathcal { L } _ { A c c } + \lambda _ { 3 } \mathcal { L } _ { E v e n t } + \lambda _ { 4 } \mathcal { L } _ { K F } .\tag{20}
$$

## 4.3 Motion-aware Simulated Annealing

Fixed optimization paradigms struggle to accurately model variable-speed motion. Specifically, a fixed initial learning rate $l _ { i n i t }$ may fail during high-speed motion phases, while exhibiting significant oscillations at low speeds.

To address this, we design a novel MSA strategy, as shown in Fig.3. Based on the optimal estimation of instantaneous displacement $\pmb { T } _ { t }$ from KF process, we adaptively schedule the learning rate of the next timestamp:

$$
l ( n ) _ { t + \Delta t } = ( \frac { l _ { m a x } - l _ { m i n } } { | \pmb { T _ { m a x } } | } \cdot | \pmb { T _ { t } } | + l _ { m i n } ) \cdot \gamma ^ { e ( n ) } ,\tag{21}
$$

where $| \pmb { T _ { m a x } } |$ represents the maximum observed displacement, $l _ { m i n }$ and $l _ { m a x }$ are the empirically determined values. This implies that large displacements require a large learning rate for fast convergence, whereas small displacements need a small learning rate to avoid oscillations.

As the step index n increases during training, the Gaussians progressively converge toward target spatial positions. Thus we implement a gradually shrinking step size via exponentially decaying $e ( n )$ to enable refinement, which controlled by the decay factor $\gamma \in ( 0 , 1 )$

<!-- image-->  
Figure 4: Schematic of RGB-Event spatiotemporal synchronization acquisition device (left) and imaging process (right).

## 5 Experiments

## 5.1 Experimental Settings

Datasets. By extending the primary part of the dataset established by [43], we constructed the first dataset dedicated to natural rigid motion across large spatiotemporal scales. It covers diverse fundamental dynamics: horizontal / oblique projectile motion, free fall and friction. The dataset consists of 6 sets of synthetic and 9 sets of real-world data. Table.1 summarizes the comparisons with various benchmarks.

â¢ Synthetic scenes. We collected 6 models from the public 3D community [1], projecting them with random initial velocities and autorotation. A uniform force field was deployed to simulate real conditions (e.g., gravity and resistance), which is rarely considered in previous benchmarks. A fixed camera was used to capture at 120 FPS and 1024\*1024 resolution. Clear images were averaged after superimposition to synthesize blurred images, and event streams were simulated by V2E [13].

â¢ Real-world scenes. The field currently suffers from a severe scarcity of 4D RGB-E datasets, with very few being open-sourced [8, 42]. The only publicly available dataset [38] remains limited to minor deformations. Furthermore, unlike common ideal conditions [42, 37], where objects are placed on a controlled turntable rotating uniformly, we built an RGB-Event spatiotemporally synchronized imaging device to flexibly capture scenes, as shown in Fig.4. The setup uses an STM32 microcontroller to trigger PWM signals controlling a FLIR camera for imaging (720\*540). Light is spatially synchronized to a DVXplorer camera (640\*480) via a beam splitter to record event streams.

<!-- image-->  
Figure 5: Qualitative comparison on the synthetic dataset. For clearer results, please refer to the video in the supplementary material.

In the novel view reconstruction task, the training and validation sets are divided according to [16]. For motion recovery, we focus on recovering the motion across the entire continuous time sequence (i.e., all frames), thus no split is applied. All competitors adopt the aforementioned division scheme to ensure fair metric computation.

Metrics. For video reconstruction, we adopt PSNR, SSIM, and LPIPS [33, 47]. For motion recovery, we calculate the bounding boxes of the target in both real and rendered images, and then assess the spatial accuracy of the trajectory using IoU, absolute trajectory error (ATE), and RMSE [45, 30, 9]. All evaluations are performed after complete background removal.

Baselines. Given the severe scarcity of open-source, comparable 4D eventbased algorithms, we selected RGB-only methods 4DGS [36], DynamicGS [21] and MotionGS [49]. For a fair comparison, we fed the deblurred images from our trained MAE network into them so as to assess the baselinesâ dynamic modeling capability with clear input (denoted as âE + 4DGSâ, etc.). Baselines E2GS [6], ComoGS [18], and DeblurGS [3] are specifically designed for blurry scene reconstruction. CFGS [9] shares a similar 6DoF pose estimation with our approach. However, its depth-based initialization fails entirely for focused targets. Therefore, we initialized it with our pre-trained Gaussians for an equitable comparison. Furthermore, considering the characteristics of rigid motion, we also included 2DGS [14], and InstantSplat [7] as static-scene baselines.

Implementation Details. We conducted experiments using PyTorch on a NVIDIA RTX 4090 GPU. During the target modeling stage, we cropped the target region from the raw data and centered it on a 512Ã512 pixel canvas. Besides, we optimized a set of [r, t, s] within 10,000 iterations with a learning

<!-- image-->  
Ground-Truth  
PEGS(ours)

Figure 6: Qualitative comparison on the real dataset.

rate of 5e-5 to align the target-centered Gaussians with the first frame of the dynamic scene. This process is efficient and typically completes in minutes. For the motion recovery stage, the initial learning rate was set to 1.0e-3 for the frame with minimal velocity, with a base iteration count of 1,000. The loss weights for $\mathcal { L } _ { B l u r } , \mathcal { L } _ { A c c } , \mathcal { L } _ { E v e n t }$ , and $\mathcal { L } _ { K F }$ were set to 0.7, 0.1, 0.15, and 0.05, respectively.

## 5.2 Comparison

Comparison of video reconstruction. Table.2 summarize the quantitative results of different methods for video reconstruction on real and synthetic data respectively. Our approach exhibits consistent performance. The introduced density control improves both visual and geometric fidelity, serving as a robust basis for recovering motion trajectories. [9] is limited by its depth inference pipeline. For modeling specific objects in constrained settings, monocular depth estimation faces an inherently underdetermined problem. This prevents the maintenance of 3D consistency across long sequences. By introducing physical or rigid constraints, [21, 49] achieve better results than the visual-only [36].

<table><tr><td rowspan="2"></td><td colspan="3">Grenade</td><td colspan="3">Missile</td><td colspan="3"></td><td colspan="3"></td><td colspan="3"></td><td colspan="3"></td><td colspan="3"></td><td colspan="3">Mean</td></tr><tr><td>[PSNRâ SSIMâ LPIPSâ|</td><td></td><td></td><td></td><td>|PSNRâ SSIMâ</td><td>LPIPSâ|</td><td></td><td>|PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>|PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td></td><td>[|PSNRâ SSIMâ LPIPSâ|</td><td></td><td>|PSNRâ</td><td></td><td>SSIMâ LPIPSâ</td><td></td><td>|PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>E + 2DGS [14]</td><td>24.82</td><td>0.895</td><td>0.153</td><td>17.42</td><td>0.714</td><td>0.325</td><td></td><td>15.97</td><td>0.847</td><td>0.195</td><td>17.99</td><td>0.792</td><td>0.218</td><td>15.19</td><td>0.758</td><td>0.275</td><td>21.56</td><td></td><td></td><td>0.185</td><td>18.83</td><td>0.806</td><td>0.225</td></tr><tr><td>E + InstantSplat [7]</td><td>25.44</td><td>0.900</td><td>0.139</td><td>17.58</td><td>0.724</td><td>0.300</td><td></td><td>17.03 0.852</td><td>0.175</td><td></td><td>18.10 0.797</td><td>0.221</td><td></td><td>16.10</td><td>0.761</td><td>0.255</td><td>24.21</td><td>0.830 0.877</td><td>0.165</td><td>19.74</td><td></td><td></td><td>0.209</td></tr><tr><td>CFGS [9]</td><td>21.82</td><td>0.873</td><td>0.231</td><td></td><td>13.70 0.675</td><td>0.259</td><td></td><td>14.47</td><td>0.845</td><td>0.284</td><td>18.00</td><td>0.788</td><td>0.307</td><td>11.93</td><td>0.739</td><td>0.317</td><td>21.37</td><td></td><td>0.196</td><td>16.88</td><td></td><td>0.818 0.790</td><td>0.266</td></tr><tr><td>E + 4DGS [36]</td><td>20.52</td><td>0.907</td><td>0.112</td><td>22.60</td><td>0.829</td><td>0.082</td><td></td><td>20.25 0.859</td><td>0.154</td><td></td><td>23.92 0.947</td><td>0.063</td><td></td><td>17.91</td><td>0.934</td><td>0.182</td><td>Ã</td><td>0.822 Ã</td><td>Ã</td><td>21.04</td><td></td><td>0.895</td><td>0.119</td></tr><tr><td>E + MotionGS [49]</td><td>22.64</td><td>0.876</td><td>0.166</td><td>19.14</td><td>0.857</td><td>0.156</td><td></td><td>15.39 0.823</td><td>0.217</td><td></td><td>0.854 22.70</td><td>0.119</td><td></td><td>16.99</td><td>0.902</td><td>0.115</td><td>17.37</td><td>0.855</td><td>0.151</td><td>19.04</td><td></td><td>0.861</td><td>0.154</td></tr><tr><td>E + DynamicGS [21]</td><td>36.32</td><td>0.971</td><td>0.064</td><td>26.37</td><td>0.906</td><td>0.032</td><td></td><td>28.44 0.939</td><td>0.026</td><td></td><td>31.81</td><td>0.942 0.050</td><td></td><td>25.36</td><td>0.897</td><td>0.018</td><td>Ã</td><td>Ã</td><td>Ã</td><td>29.66</td><td></td><td>0.931</td><td>0.038</td></tr><tr><td>Ours</td><td>33.11</td><td>0.964</td><td>0.015</td><td></td><td>28.32 0.913</td><td>0.029</td><td></td><td>26.07</td><td>0.937 0.016</td><td></td><td>33.08</td><td>0.956 0.019</td><td></td><td>24.76</td><td>0.917</td><td>0.024</td><td>32.87</td><td>0.945</td><td>0.016</td><td>29.70</td><td></td><td>0.939</td><td>0.020</td></tr></table>

Table 2: Quantitative comparison of video reconstruction by different methods.

<table><tr><td rowspan="2"></td><td colspan="3">Grenade</td><td colspan="3">Missile</td><td colspan="3">Cola</td><td colspan="3">Doll</td><td colspan="3">Chocolate</td><td colspan="3">Rugby</td><td colspan="3">Mean</td></tr><tr><td>IoUâ</td><td></td><td>ATEâ RMSEâ</td><td>IoUâ</td><td>ATEâ</td><td>RMSEâ</td><td>IoUâ</td><td>ATEâ</td><td>RMSEâ</td><td>IoUâ</td><td>ATEâ</td><td>RMSEâ</td><td>IoUâ</td><td>ATEâ RMSEâ</td><td></td><td>IoUâ</td><td>ATEâ RMSEâ</td><td></td><td>IoUâ</td><td>ATEâRMSEâ</td></tr><tr><td>CFGS [9]</td><td>0.249 0.550</td><td></td><td>0.602</td><td>0.203</td><td>0.129</td><td>0.218</td><td>0.154</td><td>0.533</td><td>0.563</td><td>0.310 0.358</td><td>0.459</td><td></td><td>0.318</td><td>0.574</td><td>0.610</td><td>0.094 0.562</td><td></td><td>0.604</td><td>0.221</td><td>0.451</td><td>0.509</td></tr><tr><td>E + 4DGS [36]</td><td>0.359 0.661</td><td></td><td>0.703</td><td>0.608</td><td>0.194</td><td>0.248</td><td>0.354</td><td>0.630</td><td>0.661</td><td>0.401</td><td>0.563</td><td>0.621</td><td>0.287</td><td>0.668</td><td>0.687</td><td>Ã</td><td>Ã</td><td>Ã</td><td></td><td>0.402 0.543</td><td>0.584</td></tr><tr><td>E + MotionGS [49]</td><td></td><td>0.288 0.474</td><td>0.505</td><td>0.541</td><td>0.655</td><td>0.704</td><td>0.423</td><td>0.511</td><td>0.551</td><td>0.501</td><td>0.505</td><td>0.559</td><td>0.549</td><td>0.501</td><td>0.522</td><td>0.249</td><td>0.643</td><td>0.668</td><td>0.425</td><td>0.548</td><td>0.585</td></tr><tr><td>E + DynamicGS [21]</td><td></td><td>0.548 0.227</td><td>0.269</td><td>0.910</td><td>0.118</td><td>0.156</td><td>0.898</td><td>0.182</td><td>0.226</td><td>0.663</td><td>0.311</td><td>0.363</td><td>0.995 0.070</td><td></td><td>0.090</td><td>Ã</td><td>Ã</td><td>Ã</td><td>0.803</td><td>0.182</td><td>0.221</td></tr><tr><td>DeblurGS [3]</td><td></td><td>0.693 0.217</td><td>0.253</td><td>0.824</td><td>0.098</td><td>0.173</td><td>Ã</td><td>Ã</td><td>Ã</td><td>0.852</td><td>0.132</td><td>0.249</td><td>0.947</td><td>0.186</td><td>0.233</td><td></td><td>0.893 0.057</td><td>0.169</td><td>0.842 0.138</td><td></td><td>0.215</td></tr><tr><td>ComoGS [18]</td><td></td><td>0.593 0.268</td><td>0.317</td><td>0.732</td><td>0.179</td><td>0.177</td><td>0.586 0.624</td><td></td><td>0.653</td><td>0.649</td><td>0.160</td><td>0.273</td><td>0.941</td><td>0.191</td><td>0.249</td><td></td><td>0.880 0.070</td><td>0.185</td><td>0.730 0.265</td><td></td><td>0.309</td></tr><tr><td>Ours</td><td></td><td>0.972 0.174</td><td>0.261</td><td>0.983 0.098</td><td></td><td>0.142</td><td>0.984 0.076</td><td></td><td>0.089</td><td>0.971 0.097</td><td></td><td>0.163</td><td>0.953</td><td>0.184</td><td>0.232</td><td></td><td>0.946 0.029 0.157</td><td></td><td></td><td></td><td>|0.968 0.109 0.174</td></tr></table>

Table 3: Quantitative comparison of motion recovery by different methods on synthetic datasets.

Comparison of motion recovery. Figure 7 presents uniformly sampled frames from complete sequences, with quantitative results provided in Table 4. Among the evaluated methods, PMGS delivers strong performance across all three tracking metrics, indicating accurate estimation of both translational and rotational motion. In contrast, even when initialized with our well-trained 3D model, CFGS diverges significantly as its Gaussians are driven toward infinite spatial positions under purely photometric supervision in an attempt to fit image similarity. 4DGS and MotionGS exhibit noticeable trajectory fragmentation and visual artifacts, reflecting common failure modes of dynamic representations when confronted with large-scale motions. DynamicGS shows encouraging results but generalizes poorly to certain object instances. Although ComoGS and DeblurGS achieve reasonable motion trajectory estimation in most scenarios, they still fall short in complete motion deblurring. Collectively, these results underscore the importance of incorporating physical constraints and event-based cues for robust reconstruction of large blur motions.

## 5.3 Ablation Study

Effectiveness of point density control. As shown in Table 5, the performance drop of Model 1 suggest that poor reconstruction quality adversely affects tracking accuracy. Inaccurate geometric and appearance representations (floating Gaussians and artifacts) may lead to target localization deviations, resulting in accumulated errors over time.

<table><tr><td rowspan="2"></td><td colspan="3">Shark ATEâ RMSEâ|</td><td colspan="3">Bear</td><td colspan="3"></td><td colspan="3"></td><td colspan="3"></td><td colspan="3"></td><td colspan="3">Doraemon</td><td colspan="3">Mean</td></tr><tr><td>IoUâ</td><td></td><td></td><td>IoUâ</td><td>ATEâ RMSEâ|</td><td></td><td></td><td>IoUâ</td><td>ATEâ</td><td>RMSEâ|</td><td>IoUâ</td><td>ATEâ</td><td>RMSEâ|</td><td>IoUâ</td><td></td><td>ATEâ</td><td>RMSEâ</td><td>IoUâ</td><td>ATEâ</td><td>RMSEâ</td><td>IoUâ</td><td>ATEâ</td><td>RMSEâ</td></tr><tr><td>E + 4DGS [36]</td><td>|0.141 0.505</td><td></td><td>0.556</td><td>Ã</td><td>Ã</td><td></td><td>Ã</td><td>|0.144</td><td>0.553</td><td>0.592</td><td>0.076</td><td>0.335</td><td>0.374</td><td>0.190</td><td></td><td>0.214</td><td>0.305</td><td>0.202</td><td>0.312</td><td>0.448</td><td>0.151</td><td>0.383</td><td>0.566</td></tr><tr><td>E + DynamicGS [21]</td><td>0.679 0.094</td><td></td><td>0.155</td><td>Ã</td><td>Ã</td><td></td><td>Ã</td><td>0.262</td><td>0.439</td><td>0.470</td><td>0.214</td><td>0.183</td><td>0.270</td><td>0.378</td><td>0.152</td><td>0.235</td><td>0.559</td><td>0.146</td><td></td><td>0.261</td><td>0.418 0.203</td><td></td><td>0.232</td></tr><tr><td>DeblurGS [3]</td><td>0.834 0.033</td><td></td><td>0.142</td><td></td><td>0.876 0.024</td><td></td><td>0.138</td><td>0.909</td><td>0.021</td><td>0.136</td><td>0.195</td><td>0.236</td><td>0.356</td><td>0.373</td><td>0.176</td><td>0.247</td><td></td><td>0.497</td><td>0.208</td><td>0.317</td><td>0.614</td><td>0.116</td><td>0.223</td></tr><tr><td>Comos [18]</td><td>Ã</td><td>Ã</td><td>Ã</td><td>0.087</td><td>0.219</td><td></td><td>0.256</td><td>0.053</td><td>0.211</td><td>0.242</td><td>0.050</td><td>0.452</td><td>0.521</td><td>0.292</td><td>0.201</td><td>0.317</td><td>0.410</td><td></td><td>0.205</td><td>0.325</td><td>0.178</td><td>0.257</td><td>0.332</td></tr><tr><td>Ours</td><td></td><td>|0.862 0.027</td><td>0.097</td><td></td><td>|0.953 0.050</td><td></td><td>0.052</td><td>[0.931 0.066</td><td></td><td>0.094</td><td></td><td>0.664 0.057</td><td>0.060</td><td></td><td>0.822 0.024</td><td>0.028</td><td>0.628 0.075</td><td></td><td></td><td>0.167</td><td>0.810 0.050</td><td></td><td>0.083</td></tr></table>

Table 4: Quantitative comparison of motion recovery by different methods on real datasets.

<!-- image-->  
Figure 7: While PEGS exhibits minor differences in color rendering, it accurately recovers the motion trajectory under fast friction.

Effectiveness of physical constraints. In Model 2, the $\mathcal { L } _ { A c c }$ and MSA strategy are excluded, while retaining only photometric supervision and a fixed training paradigm. Trajectory accuracy declines significantly, with AME increasing 0.06 and RMSE increasing by 0.82. Additionally, the removal of the MSA strategy substantially increases the optimization cost and introduces potential risks of training oscillation.

Effectiveness of event enhancement. The removal of $\mathcal { L } _ { E v e n t }$ led to degradation in both reconstruction and motion recovery (1.56 dB drop in PSNR and a 0.05 decline in RMSE), which demonstrates the high temporal resolution supervision provided by event streams compensates for motion cues lost by RGB cameras when capturing large motions.

Effectiveness of Kalman fusion. Though removing the $\mathcal { L } _ { K F }$ regularization slightly reduces computational costs, it leads to a decrease in stability, quantified as a 0.49 dB drop in PSNR and a 0.016 increase in ATE. In more challenging scenarios, such as when external disturbances are present or the captured event stream data contains pronounced noise (e.g., in low-light conditions), the role of this strategy in ensuring system robustness would become more critical.

## 6 Conclusion

In this paper, we present PEGS for the reconstruction of large spatiotemporal motion. By combining a triple-level supervision scheme featuring acceleration constraints, event stream enhancement, and Kalman fusion with MSA strategy, we effectively tackle physical inaccuracy and motion blur while ensuring robust convergence. The explicit and physically consistent motion representation offered by PEGS establishes a foundation for dynamic scene understanding and simulation, showing promising potential for exploring real-world 3D physical interactions. Our future investigations will be directed towards addressing demanding scenarios, exemplified by systems with variable force fields, targets at extreme scales, and low-quality image acquisition.

<table><tr><td rowspan="2"></td><td colspan="3">Video reconstruction</td><td colspan="4">Motion recovery</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>IoUâ</td><td>ATEâ</td><td>RMSEâ</td><td>Timesâ</td></tr><tr><td>DeblurGS</td><td>25.01</td><td>0.895</td><td>0.073</td><td>0.728</td><td>0.127</td><td>0.219</td><td>23 hours</td></tr><tr><td> $\mathrm { w / o }$  density control</td><td>27.47</td><td>0.918</td><td>0.050</td><td>0.863</td><td>0.133</td><td>0.184</td><td>57 mins</td></tr><tr><td>w/o LAcc &amp; MSA</td><td>26.35</td><td>0.912</td><td>0.054</td><td>0.847</td><td>0.140</td><td>0.211</td><td>72 mins</td></tr><tr><td>w/o LEvent</td><td>28.14</td><td>0.924</td><td>0.048</td><td>0.926</td><td>0.103</td><td>0.179</td><td>51 mins</td></tr><tr><td> $\mathrm { w } / \mathrm { o } ~ { \mathcal { L } } _ { K F }$ </td><td>29.21</td><td>0.939</td><td>0.041</td><td>0.832</td><td>0.096</td><td>0.152</td><td>46 mins</td></tr><tr><td>PEGS (Full)</td><td>29.70</td><td>0.950</td><td>0.032</td><td>0.889</td><td>0.080</td><td>0.129</td><td>56 mins</td></tr></table>

Table 5: The full model achieved the best performance, with a substantially lower training time than DeblurGS.

## References

[1] Blender Foundation. Blender â a 3d modelling and rendering package. Blender Foundation, Amsterdam, 2018.

[2] Marco Cannici and Davide Scaramuzza. Mitigating motion blur in neural radiance fields with events and frames. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9286â9296, 2024.

[3] Wenbo Chen and Ligang Liu. Deblur-gs: 3d gaussian splatting from camera motion blurred images. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 7:1 â 15, 2024.

[4] Wen-Hsuan Chu, Lei Ke, and Katerina Fragkiadaki. Dreamscene4d: Dynamic multi-object scene generation from monocular videos. arXiv preprint arXiv:2405.02280, 2024.

[5] Erwin Coumans and Yunfei Bai. Pybullet quickstart guide. ed: PyBullet Quickstart Guide. https://docs. google. com/document/u/1/d, 2021.

[6] Hiroyuki Deguchi, Mana Masuda, Takuya Nakabayashi, and Hideo Saito. E2gs: Event enhanced gaussian splatting. ArXiv, abs/2406.14978, 2024.

[7] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds. arXiv preprint arXiv:2403.20309, 2(3):4, 2024.

[8] Chaoran Feng, Zhenyu Tang, Wangbo Yu, Yatian Pang, Yian Zhao, Jianbin Zhao, Li Yuan, and Yonghong Tian. E-4dgs: High-fidelity dynamic reconstruction from the multi-view event cameras. ArXiv, abs/2508.09912, 2025.

[9] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20796â 20805, 2024.

[10] Mathias Gehrig, Mario MillhÂ¨ausler, Daniel Gehrig, and Davide Scaramuzza. E-raft: Dense optical flow from event cameras. In 2021 International Conference on 3D Vision (3DV), pages 197â206. IEEE, 2021.

[11] Zhiyang Guo, Wen gang Zhou, Li Li, Min Wang, and Houqiang Li. Motionaware 3d gaussian splatting for efficient dynamic scene reconstruction. IEEE Transactions on Circuits and Systems for Video Technology, 35:3119â3133, 2024.

[12] Bing He, Yunuo Chen, Guo Lu, Li Song, and Wenjun Zhang. S4d: Streaming 4d real-world reconstruction with gaussians and 3d control points. ArXiv, abs/2408.13036, 2024.

[13] Yuhuang Hu, Shih-Chii Liu, and Tobi Delbruck. v2e: From video frames to realistic dvs events. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1312â1321, 2021.

[14] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1â11, 2024.

[15] Yiming Huang, Beilei Cui, Long Bai, Ziqi Guo, Mengya Xu, Mobarakol Islam, and Hongliang Ren. Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 197â207. Springer, 2024.

[16] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÂ¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

[17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015â4026, 2023.

[18] Jungho Lee, Donghyeong Kim, Dogyoon Lee, Suhwan Cho, Minhyeok Lee, Wonjoon Lee, Taeoh Kim, Dongyoon Wee, and Sangyoun Lee. Comogaussian: Continuous motion-aware gaussian splatting from motion-blurred images. arXiv preprint arXiv:2503.05332, 2025.

[19] Jia Li, Jiaxu Wang, Junhao He, Mingyuan Sun, Renjing Xu, Qiang Zhang, Jiahang Cao, Ziyi Zhang, Yi Gu, and Jingkai SUN. Degs: Deformable event-based 3d gaussian splatting from rgb and event stream. 2025.

[20] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5521â5531, 2022.

[21] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV), pages 800â809. IEEE, 2024.

[22] Lukas Meyer, Floris Erich, Yusuke Yoshiyasu, Marc Stamminger, Noriaki Ando, and Yukiyasu Domae. Pegasus: Physically enhanced gaussian splatting simulation system for 6dof object pose dataset generation. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 10710â10715. IEEE, 2024.

[23] Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley, Miaomiao Liu, and Yuchao Dai. Bringing a blurry frame alive at high frame-rate with an event camera. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6820â6829, 2019.

[24] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021.

[25] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10318â10327, 2021.

[26] Yunshan Qi, Lin Zhu, Yu Zhang, and Jia Li. E2nerf: Event enhanced neural radiance fields from blurry images. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 13208â13218, 2023.

[27] Jinsheng Quan, Chunshi Wang, and Yawei Luo. Particlegs: Particle-based dynamics modeling of 3d gaussians for prior-free motion extrapolation. ArXiv, abs/2505.20270, 2025.

[28] RenÂ´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179â12188, 2021.

[29] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023.

[30] Denys Rozumnyi, JiËrÂ´Ä± Matas, Marc Pollefeys, Vittorio Ferrari, and Martin R Oswald. Tracking by 3d model estimation of unknown objects in videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14086â14096, 2023.

[31] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104â4113, 2016.

[32] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 29(5):2732â2742, 2023.

[33] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004.

[34] Greg Welch, Gary Bishop, et al. An introduction to the kalman filter. 1995.

[35] Yuchen Weng, Zhengwen Shen, Ruofan Chen, Qi Wang, and Jun Wang. Eadeblur-gs: Event assisted 3d deblur reconstruction with gaussian splatting. ArXiv, abs/2407.13520, 2024.

[36] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310â20320, 2024.

[37] Jingqian Wu, Peiqi Duan, Zongqiang Wang, Changwei Wang, Boxin Shi, and Edmund Y Lam. Dark-evgs: Event camera as an eye for radiance field in the dark. arXiv preprint arXiv:2507.11931, 2025.

[38] Tong Wu, Jiali Sun, Yu-Kun Lai, and Lin Gao. De-nerf: Decoupled neural radiance fields for view-consistent appearance editing and high-frequency environmental relighting. ACM SIGGRAPH 2023 Conference Proceedings, 2023.

[39] Butian Xiong, Xiaoyu Ye, Tze Ho Elden Tse, Kai Han, Shuguang Cui, and Zhen Li. Sa-gs: Semantic-aware gaussian splatting for large scene reconstruction with geometry constrain. ArXiv, abs/2405.16923, 2024.

[40] Tianyi Xiong, Jiayi Wu, Botao He, Cornelia Fermuller, Yiannis Aloimonos, Heng Huang, and Christopher A Metzler. Event3dgs: Event-based 3d gaussian splatting for high-speed robot egomotion. arXiv preprint arXiv:2406.02972, 2024.

[41] Wenhao Xu, Wenming Weng, Yueyi Zhang, Ruikang Xu, and Zhiwei Xiong. Event-boosted deformable 3d gaussians for dynamic scene reconstruction. 2024.

[42] Wenhao Xu, Wenming Weng, Yueyi Zhang, Ruikang Xu, and Zhiwei Xiong. Event-boosted deformable 3d gaussians for dynamic scene reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 28334â28343, 2025.

[43] Yijun Xu, Jingrui Zhang, Yuhan Chen, Dingwen Wang, Lei Yu, and Chu He. Pmgs: Reconstruction of projectile motion across large spatiotemporal spans via 3d gaussian splatting. ArXiv, abs/2508.02660, 2025.

[44] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20331â20341, 2024.

[45] Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, and Thomas Huang. Unitbox: An advanced object detection network. In Proceedings of the 24th ACM international conference on Multimedia, pages 516â520, 2016.

[46] Wangbo Yu, Chaoran Feng, Jiye Tang, Jiashu Yang, Zhenyu Tang, Xu Jia, Yuchao Yang, Li Yuan, and Yonghong Tian. Evagaussians: Event stream assisted gaussian splatting from blurry images. arXiv preprint arXiv:2405.20224, 2024.

[47] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018.

[48] Licheng Zhong, Hong-Xing Yu, Jiajun Wu, and Yunzhu Li. Reconstruction and simulation of elastic objects with spring-mass 3d gaussians. In European Conference on Computer Vision, pages 407â423. Springer, 2024.

[49] Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng, Jiahao Lu, Wenfei Yang, Tianzhu Zhang, and Yongdong Zhang. Motiongs: Exploring explicit motion guidance for deformable 3d gaussian splatting. ArXiv, abs/2410.07707, 2024.

[50] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European conference on computer vision, pages 145â163. Springer, 2024.