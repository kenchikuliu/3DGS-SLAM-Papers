# EBAD-Gaussian: Event-driven Bundle Adjusted Deblur Gaussian Splatting

Yufei Deng   
Sichuan University   
Chengdu, China   
yufeideng@stu.scu.edu.cn   
Chenwei Tang   
Sichuan University   
Chengdu, China   
tangchenwei@scu.edu.cn

Deng Xiong Stevens institute of Technology Hoboken, NJ, China dxiong@stevens.edu

Yuanjian Wang   
Sichuan University   
Chengdu, China   
wangyuanjian@stu.scu.edu.cn

Jizhe Zhou Sichuan University Chengdu, China jzzhou@scu.edu.cn

Jiancheng Lv   
Sichuan University   
Chengdu, China   
lvjiancheng@scu.edu.cn

Rong Xiao Sichuan University Chengdu, China rxiao@scu.edu.cn

Jiahao Fan Sichuan University Chengdu, China fanjh@scu.edu.cn

Huajin Tang Zhejiang University Hangzhou, China htang@zju.edu.cn

## Abstract

While 3D Gaussian Splatting (3D-GS) achieves photorealistic novel view synthesis, its performance degrades with motion blur. In scenarios with rapid motion or low-light conditions, existing RGBbased deblurring methods struggle to model camera pose and radiance changes during exposure, reducing reconstruction accuracy. Event cameras, capturing continuous brightness changes during exposure, can effectively assist in modeling motion blur and improving reconstruction quality. Therefore, we propose Event-driven Bundle Adjusted Deblur Gaussian Splatting (EBAD-Gaussian), which reconstructs sharp 3D Gaussians from event streams and severely blurred images. This method jointly learns the parameters of these Gaussians while recovering camera motion trajectories during exposure time. Specifically, we first construct a blur loss function by synthesizing multiple latent sharp images during the exposure time, minimizing the difference between real and synthesized blurred images. Then we use event stream to supervise the light intensity changes between latent sharp images at any time within the exposure period, supplementing the light intensity dynamic changes lost in RGB images. Furthermore, we optimize the latent sharp images at intermediate exposure times based on the event-based double integral (EDI) prior, applying consistency constraints to enhance the details and texture information of the reconstructed images. Extensive experiments on synthetic and real-world datasets show that EBAD-Gaussian can achieve high-quality 3D scene reconstruction under the condition of blurred images and event stream inputs.

## CCS Concepts

â¢ Computing methodologies â Reconstruction.

## Keywords

3D Reconstruction, Event Camera, Motion Blur Removal, Multimodal Fusion

## 1 Introduction

Recent advances in 3D reconstruction and novel view synthesis have driven the development of key applications, including augmented and virtual reality (AR/VR) [1, 2], robot navigation [3, 4], scene understanding [5, 6], and 3D content generation [7, 8]. Among these, 3D Gaussian Splatting (3D-GS) [9, 10] can generate highfidelity novel views with high rendering speed, provided there are high-quality 2D images and accurate camera poses. However, in practical applications, obtaining high-quality images and precise poses can be challenging. Image clarity is influenced by both the exposure time and the cameraâs motion speed. During exposure, the sensor continuously records light. If the camera or object moves rapidly, the same object will be imaged at different locations on the sensor, causing its trajectory to be captured as motion blur. The longer the exposure time and the faster the camera moves, the more pronounced the objectâs motion blur becomes, resulting in a blurrier image. Blurry images lead to feature point matching errors, increasing the deviation of camera poses and point clouds estimated by Structure from Motion (SfM) [11], while the loss of details affects Gaussian parameter optimization, ultimately reducing the quality of novel view synthesis. Therefore, under conditions of fast camera motion and low light, how to recover sharp 3D-GS scenes from motion-blurred images becomes a key issue.

In practical applications, when the scene remains static, motion blur arises from camera pose changes during exposure. The key challenge in reconstructing sharp 3D-GS is accurately recovering the motion trajectory to simulate blur formation. Recently, BAD-Gaussians [12] modeled the physical image formation process of motion-blurred images and jointly optimized the camera motion trajectory and Gaussian parameters during exposure, enabling the recovery of high-quality 3D scenes from blurred images. However, this method is only applicable to scenes where blur is caused by global camera motion and struggles under conditions of low lighting and fast motion. This is due to the low signal-to-noise ratio of images, which leads to increased errors in camera trajectory estimation under low-light conditions; meanwhile, fast motion introduces stronger spatially inconsistent blur, making the global trajectory modeling assumption invalid. Therefore, under these extreme conditions, it remains challenging for this method to recover fine details in the reconstructed scene.

Event cameras provide a more effective solution for the deblurring task in low-light and fast-motion scenarios [13â16]. As a bio-inspired visual sensor, they asynchronously detect brightness changes in the scene, capturing subtle dynamic information with high temporal resolution, thus compensating for the key information lost during the exposure time in RGB cameras [17]. Inspired by this, E2NeRF [13] introduces event information, using consistency loss and structure-aware constraints to guide the neural radiance field (NeRF) [18] in capturing spatiotemporal structural changes, achieving scene reconstruction without sharp image supervision. Ev-DeblurNeRF [14] further utilizes event streams to model the motion blur process, optimizing the NeRF by combining event information and RGB data, thereby improving the reconstruction quality. Compared to methods relying solely on RGB cameras, these methods can recover more accurate details. However, achieving real-time rendering and synthesizing high-fidelity novel views with complex details presents a significant challenge for these approaches.

To address the above challenges, this paper proposes the EBAD-Gaussian method, which effectively suppresses motion blur and achieves high-quality real-time reconstruction by fusing event streams and RGB bimodal data, and optimizing Gaussian parameters and camera poses within the exposure time during training. Specifically, we first model the cumulative effect of relative camera motion over the exposure time and static scene on imaging. By synthesizing multiple latent sharp images, we construct a blur loss function to minimize the difference between real blurred images and synthesized blurred images, providing a foundational guarantee for the clarity and structural fidelity of the reconstructed images. Next, we simulate the predicted event data by modeling the brightness change during the exposure time and compare it with the real event stream, serving as the event rendering loss. Additionally, we compare the event-based double integral (EDI) [19] deblurred image with the latent sharp image at intermediate exposure moments to construct the EDI loss. By jointly constraining the 3D-GS training process with the above loss functions, EBAD-Gaussian not only effectively removes motion blur but also significantly improves reconstruction accuracy and real-time rendering capability. Experiments on both synthetic and real data demonstrate that the proposed method achieves high-fidelity real-time reconstruction under complex motion and low-light conditions, providing an efficient solution for the 3D reconstruction of static scenes. In summary, we make the following contributions:

â¢ To address the limitations of traditional 3D-GS in low-light and high-speed scenarios, we leverage the high dynamic range characteristics of event streams to compensate for high-frequency details in underexposed regions.

â¢ We propose a novel EBAD-Gaussian framework that incorporates a motion blur loss and jointly optimizes camera poses and Gaussian parameters using both event and RGB data, thereby significantly enhancing the reconstruction quality.

â¢ Experiments on both synthetic and real data validate that our method achieves precise motion trajectory estimation and high-fidelity real-time 3D reconstruction for scenes with severe blur images and corresponding event data.

## 2 Related Work

## 2.1 Recovering 3D Structure from Blurred Images

In high-quality 3D scene reconstruction, sharp and high-fidelity images are crucial as supervision signals. However, motion-blurred images, commonly encountered in the real world, can severely impact the accurate reconstruction of 3D scenes. Recent advances in neural rendering have addressed the motion blur problem through distinct methodological approaches. In the seminal work of Deblur-NeRF [20] and DP-NeRF [21], the image degradation process is modeled via learned blur kernels that approximate the integration of radiance over the exposure period. Building upon this framework, BAD-NeRF [22] introduces a physically-grounded blur formation model, where the implicit neural representation is jointly optimized with the camera trajectory through a differentiable bundle adjustment formulation. This co-optimization paradigm simultaneously refines both the scene geometry and the temporally-varying camera poses during the exposure interval. However, these NeRF-based methods suffer from high computational costs and long training times, making it difficult for them to meet the demands of real-time applications. Building upon recent advancements in 3D-GS [23, 24], BAD-Gaussians [12]proposed a novel framework that integrates 3D-GS scene representation with physically-based blur modeling and bundle adjustment for deblurring reconstruction. This approach demonstrates superior computational efficiency compared to neural radiance field methods, achieving both faster rendering speeds and accelerated convergence. Nevertheless, significant challenges remain in processing heavily degraded images captured under extreme conditions of low illumination and rapid motion.

## 2.2 Event-Based 3D Scene Reconstruction

The event camera is a novel bio-inspired visual sensor. Unlike standard cameras that capture images at a fixed frame rate, event cameras generate an event stream by detecting changes in pixel brightness. They are characterized by low latency, high dynamic range, low power consumption, and high temporal resolution. Inspired by the excellent performance of event cameras, several studies have attempted to reconstruct 3D scenes from the event stream captured by event cameras, especially under conditions of low light and rapid camera motion. Recently, various event-based 3D reconstruction methods[25â28] have been proposed, including EventNeRF [29], Ev-NeRF [30], SaENeRF [31], Robust e-NeRF [32] and its variants [33]. These methods focus on reconstructing 3D scenes from the event streams generated by fast-moving event cameras. However, reconstructions relying solely on event data often suffer from limited texture quality.

E-NeRF[34] was the first to utilize both event streams and intensity images for 3D reconstruction, enabling sharper radiance field generation. Nevertheless,this method does not model the formation of blurred images, limiting its reconstruction fidelity. To address this, Ev-DeblurNeRF [14] incorporated a blur formation model and leveraged event data to supervise brightness variations during camera exposure, leading to improved reconstruction results. However, it does not optimize camera poses during training, which restricts its performance. EBAD-NeRF[35] further optimizes the reconstruction quality by jointly optimizing the camera pose and implicit representation of the scene throughout the exposure time. Yet, the high computational cost of ray sampling in neural radiance field frameworks remains a significant bottleneck, limiting the real-time applicability of such methods in practical scenarios.

## 3 Methods

EBAD-Gaussian leverages dual-modal data, namely images with motion blur and event data, to jointly optimize Gaussian parameters and camera poses during the exposure time, enabling the reconstruction of sharp and high-quality 3D Gaussian splatting. Specifically, we first use the EDI [36] algorithm to process the blurred images and the corresponding event stream to generate the deblurred latent sharp image. Next, the EDI deblurred image is input into COLMAP [37] to generate an SfM sparse point cloud and an initial camera pose recovered from the motion. Then, we initialize the 3D-GS [38] scene representation and project it onto the camera imaging plane, rasterizing to generate the latent sharp image. Finally, the average of the latent sharp images is utilized to simulate motion blur, and the loss is constructed by comparing it to the real blurred image; the EDI deblurred image constraint ensures consistency between the latent sharp image captured at intermediate moments during the exposure time; and the constraint imposed by the integral of the event stream, when multiplied by a threshold, regulates the differences in the logarithmic domain of brightness between the latent sharp images at different moments during the exposure time. Through these three constraints, we can effectively optimize the Gaussian parameters and camera pose, achieving high-fidelity modeling of motion blur and generating high-quality dynamic scene reconstruction results (see Figure 1).

## 3.1 Motion Blur Modeling with 3D-GS

3D Gaussian Splatting models static scenes using an explicit set of Gaussian primitives. Each Gaussian primitive is characterized by its center position $\mu \in \mathbb { R } ^ { 3 }$ , covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ , opacity $\alpha \in [ 0 , 1 ]$ , and third-order spherical harmonic coefficients, which are used to represent its spatial structure, color attributes, and directional dependence. During the rendering process, 3D-GS projects each Gaussian primitive from the world coordinate system onto the cameraâs image plane, and the projection covariance on the image plane can be computed using the following formula:

$$
\Sigma ^ { \prime } = J W \Sigma W ^ { \top } J ^ { \top }\tag{1}
$$

Where ?? is the transformation matrix representing the change from the world coordinate system to the camera coordinate system, and ?? is the Jacobian approximation of the perspective projection. Subsequently, the system sorts all the Gaussians based on depth and performs image composition using volume rendering[39, 40]. The final color at the pixel position $\boldsymbol { u } = ( x , y )$ is given :

$$
C ( u ) = \sum _ { i = 1 } ^ { n } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) \quad\tag{2}
$$

Where $c _ { i }$ is the color of the ??-th Gaussian and $\alpha _ { i }$ is its opacity. Since this process is fully differentiable, it supports optimization based on image supervision.

The imaging process of modern digital cameras is essentially the spatiotemporal integration of photons over a finite exposure time. When there is relative motion between the camera and the scene being captured, the radiance value recorded at each pixel ?? on the sensor can be modeled as the temporal integration of the latent sharp image over the exposure time window $[ f - \tau / 2 , f + \tau / 2 ] ;$

$$
B ( u ) = { \frac { 1 } { \tau } } \int _ { f - { \frac { \tau } { 2 } } } ^ { f + { \frac { \tau } { 2 } } } I ( u , t ) d t\tag{3}
$$

Where $B ( u ) \in \mathbb { R } ^ { W \times H \times 3 }$ represents the observed blurry image, with tensor dimensions corresponding to the image width ?? , height ??, and the RGB channels of radiance. $I ( u , t ) \in \mathbb { R } _ { + } ^ { W \times H \times 3 }$ represents the radiance received at the pixel ?? on the sensor plane at time ?? under ideal non-blurry conditions. ?? is the total exposure time, and $f$ is the middle exposure time, satisfying:

$$
f = \frac { t _ { \mathrm { s t a r t } } + t _ { \mathrm { e n d } } } { 2 }\tag{4}
$$

where $t _ { \mathrm { s t a r t } }$ and $t _ { \mathrm { e n d } }$ are the start and end times of the camera exposure, respectively. $\textstyle { \frac { 1 } { \tau } }$ is the normalization factor used to ensure that the dimensionality of the integral result remains consistent. To model motion blur with 3D-GS, we define the image rendered by 3D-GS at the camera pose at each time $t _ { i }$ as $C _ { t _ { i } } ( u )$ . Therefore, the sharp image $I ( u , t _ { i } )$ can be equivalently viewed as the sharp rendering result at different time points, corresponding to $C _ { t _ { i } } ( u )$ Based on this, we can model the blurry image as the time average of multiple sharp images:

$$
B ( u ) \approx \frac { 1 } { n } \sum _ { i = 1 } ^ { n } C _ { t _ { i } } ( u )\tag{5}
$$

Were $\{ t _ { i } \} _ { i = 1 } ^ { n }$ are the time points uniformly sampled within the exposure interval, corresponding to the camera poses $\{ P ( t _ { i } ) \} _ { i = 1 } ^ { n } ,$ which are used to dynamically render the sharp image frames $C _ { t _ { i } } ( u )$ This averaging form serves as a numerical approximation of the temporal integral, allowing us to model motion blur in the real camera imaging process through multi-time image composition within the 3D-GS framework.

Thus, in 3D-GS, ?? (??) can be viewed as the time-varying sharp image ?? (??, ??), which reflects the sceneâs observation at a certain moment under different camera poses. The blurry image ??(??) is then formed by averaging these sharp images over time, enabling differentiable modeling from Gaussian parameters to blurry images.

## 3.2 Event-based Motion Modeling

An Event Camera is a visual sensor that captures brightness changes with high temporal resolution (typically on the microsecond scale) in an asynchronous manner. Unlike traditional frame cameras, an event camera does not record complete images, but instead triggers events at each pixel when the logarithmic brightness change exceeds a set threshold Î.

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { P } \Theta = \log I ( u , t ) - \log I ( u , t _ { \mathrm { p r e v } } ) , \quad \mathrm { w h e r e } \quad \Theta > 0 } \end{array} } \end{array}\tag{6}
$$

where $I ( u , t )$ represents the brightness of the pixel at position $u =$ (??, ??) at time ?? , and $p \in \{ + 1 , - 1 \}$ is the polarity, indicating whether the brightness has increased or decreased. Thus, each event can be represented as a quadruple $\boldsymbol { e } = ( x , y , t , p )$

<!-- image-->  
Figure 1: Overview of the proposed EBAD-Gaussian framework. The method jointly optimizes the 3D Gaussian parameters and camera poses during exposure, using motion blur constraints, event supervision, and EDI priors to achieve high-fidelity scene reconstruction.

This single event records the pixel change at a certain moment. However, in practical applications, we are more concerned with the flow or temporal sequence of events. An event stream consists of multiple events ordered by time, describing the trend of intensity changes at the pixel location over a period of time. Each event is associated with the previous event through its timestamp ?? and pixel position $( x , y )$ . Given a pixel location ?? and a time window $[ t _ { \mathrm { p r e } } , t _ { \mathrm { c u r } } ]$ , the cumulative event flow $E ( t _ { \mathrm { p r e } } , t _ { \mathrm { c u r } } , u )$ within this window is defined as:

$$
E ( t _ { \mathrm { p r e } } , t _ { \mathrm { c u r } } , u ) = \sum _ { s \in [ t _ { \mathrm { p r e } } , t _ { \mathrm { c u r } } ] } \rho ( s , u )\tag{7}
$$

where ?? (??, ??) represents the polarity of the event triggered at pixel location ?? at time ??.

According to the event generation model, the cumulative polarity of events is linearly related to the logarithmic brightness difference between images:

$$
E ( t _ { \mathrm { p r e } } , t _ { \mathrm { c u r } } , u ) = \frac { \log { I ( t _ { \mathrm { c u r } } , u ) } - \log { I ( t _ { \mathrm { p r e } } , u ) } } { \Theta }\tag{8}
$$

where $I ( t _ { \mathrm { p r e } } , u )$ and $I ( t _ { \mathrm { c u r } } , u )$ represent the brightness of the latent sharp image at pixel ?? at times $t _ { \mathrm { p r e } }$ and $t _ { \mathrm { c u r } } ,$ respectively. This equation establishes an explicit relationship between the event stream and the image brightness.

To associate event streams with traditional image representations, the EDI model proposes a method that combines the event sequence with a blurred image to estimate the latent sharp image. Specifically, let the blurred image ??(??) be captured during the camera exposure interval $[ t _ { s } , t _ { e } ]$ , where $\boldsymbol { u } = ( x , y )$ denotes a pixel location. During the exposure, the camera continuously receives event data that records changes in pixel brightness. To describe this process, we introduce a logarithmic brightness function $\boldsymbol { L } ( \boldsymbol { u } , t )$ , defined as:

$$
L ( u , t ) = L ( u , t _ { r } ) + \int _ { t _ { r } } ^ { t } \Delta L ( u , s ) \ d s\tag{9}
$$

where $t _ { r }$ is a reference time, and $\Delta L ( u , s )$ denotes the brightness change per unit time, which can be approximately reconstructed from the event stream. By substituting this brightness change model into the blurred image expression Eq. 3, we obtain:

$$
B ( u ) = \frac { 1 } { \tau } \int _ { t _ { s } } ^ { t _ { e } } \exp ( L ( u , t ) ) d t , \quad \mathrm { w h e r e } ~ \tau = t _ { e } - t _ { s }\tag{10}
$$

Further substituting the expression for $\boldsymbol { L } ( \boldsymbol { u } , t )$ , we derive a nonlinear integral equation with respect to the latent sharp image at the reference time $L ( u , t _ { r } )$ . Let $I ( u , t _ { r } ) = \exp ( L ( u , t _ { r } ) )$ represent the latent sharp image, then the entire modeling process can be seen as decomposing the blurred image $B ( u )$ using the event stream Î?? to recover $I ( u , t _ { r } )$ . By optimizing this model, the EDI method can not only restore the sharp image but also recover the pixel-wise brightness variation trajectory during the exposure, i.e., the motion trace captured by the camera.

In this paper, we construct constraints using event information, designing supervised losses aimed at optimizing the reconstruction quality, starting from both the event generation mechanism and the blurry image formation process.

## 3.3 Multi-modal Constraints for Motion Deblurring

3.3.1 Image Deblurring Constraint. Based on the previous modeling of motion blur, the blurry image can be approximated as the average result of multiple latent sharp images at different time

points. Therefore, based on the motion blur model during the exposure time, we construct the motion blur loss by approximately reconstructing the blurry image:

$$
L _ { \mathrm { b l u r } } = \bigl ( 1 - \lambda _ { \mathrm { S S I M } } \bigr ) L _ { \mathrm { 1 } } + \lambda _ { \mathrm { S S I M } } L _ { \mathrm { S S I M } }\tag{11}
$$

where $L _ { 1 }$ loss measures the pixel-level intensity difference of the blurry image, ensuring that the reconstructed image is consistent with the real image in terms of brightness distribution; $L _ { S S I M }$ loss measures the structural similarity loss of the blurry image, preserving high-frequency information such as edges and textures. $\lambda _ { \mathrm { S S I M } }$ is a weighting factor that balances the contributions of the two losses, and in this paper, it is set to 0.2 to balance pixel-level accuracy and structural fidelity.

$$
L _ { 1 } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } | B _ { g t } ^ { k } ( u ) - \hat { B _ { k } } ( u ) |\tag{12}
$$

where $\hat { B _ { k } } ( u )$ is the ??-th blurry image synthesized by 3D-GS rendering, and $B _ { g t } ^ { k } ( u )$ is the real blurry image captured, ?? is the total number of blurred images.

3.3.2 Event Temporal Consistency Constraint. In order to fully leverage the event dataâs ability to perceive brightness changes at high temporal resolution, this paper designs a supervised loss function based on event information. Considering the rapid changes in the scene during the exposure time, the event loss between images at a single time point often suffers from insufficient information. Following the approach in [29], we introduce a multi-time window random sampling mechanism to establish brightness change constraints between images at different time points.

Specifically, the exposure interval $[ t _ { f } - \tau / 2 , t _ { f } + \tau / 2 ]$ is uniformly discretized into ?? sub-time windows $\{ ( t _ { i } , t _ { i + 1 } ] \} _ { i = 0 } ^ { n - 1 }$ , and the event polarities within each sub-window are accumulated to obtain the observed event polarity $E ( t _ { S } , t _ { i + 1 } , u )$ . Simultaneously, the logbrightness values of the images at each time point are rendered by 3D-GS, and the predicted brightness difference is constructed as:

$$
\hat { E } ( t _ { s } , t _ { i + 1 } , u ) = \frac { 1 } { \Theta } \left( \log I ( t _ { i + 1 } , u ) - \log I ( t _ { s } , u ) \right)\tag{13}
$$

where $t _ { s }$ is the randomly sampled start time for each sub-window, ensuring that multiple latent image states during the exposure period are covered.

Based on this, the event supervision loss function is defined as:

$$
L _ { e v } = \frac { 1 } { n - 1 } \sum _ { i = 0 } ^ { n - 2 } \left. E ( t _ { s } , t _ { i + 1 } , u ) - \hat { E } ( t _ { s } , t _ { i + 1 } , u ) \right. _ { 1 } , \mathrm { w h e r e ~ } t _ { s } \in [ t _ { 0 } , t _ { i } ]\tag{14}
$$

where ?? is the latent number of latent sharp images within the exposure time. The loss utilizes information from the event stream to provide brightness change details at different moments during the exposure time of a blurred image, thereby offering strong support for eliminating motion blur.

3.3.3 EDI Prior Constraint. In this paper, we propose an event loss optimization strategy based on the intermediate moment clarity prior. By leveraging the high temporal resolution and high dynamic range characteristics of the event stream, we jointly optimize the multi-window event loss and motion blur loss. Specifically, the EDI prior is used as the clarity prior at the intermediate moment. We design the following EDI prior loss function:

$$
L _ { \mathrm { E D I } } = \bigl ( 1 - \lambda _ { \mathrm { S S I M } } \bigr ) L _ { 1 } + \lambda _ { \mathrm { S S I M } } L _ { \mathrm { S S I M } }\tag{15}
$$

where $L _ { 1 }$ loss measures the pixel-level intensity difference between the de-blurry image and latent sharp image:

$$
L _ { 1 } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } | B _ { E D I } ^ { k } ( u ) - \hat { B } _ { s h a r p } ^ { k } ( u ) |\tag{16}
$$

where $\hat { B } _ { s h a r p } ^ { k } ( u )$ is the latent sharp image at intermediate moments during the exposure time , and $B _ { E D I } ^ { k } ( u )$ is the de-blurry image computed by EDI.

3.3.4 Final Loss Function. In this paper, we propose a comprehensive loss function by integrating RGB images and event stream information to model motion blur. The goal is to improve the accuracy and reconstruction quality of image deblurring through multiple constraints. The final comprehensive loss function can be expressed as:

$$
L _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { b l u r } } L _ { \mathrm { b l u r } } + \lambda _ { \mathrm { e v } } L _ { \mathrm { e v } } + \lambda _ { \mathrm { E D I } } L _ { \mathrm { E D I } }\tag{17}
$$

where $L _ { \mathrm { b l u r } }$ represents the motion blur loss, $L _ { \mathrm { e v } }$ represents the event-based loss, and ??EDI represents the EDI prior loss.

## 4 Experiments

## 4.1 Dataset and Experimental Setup

4.1.1 Dataset. We have opted to utilize the dataset comprising dual modalities of blurred images and event data, as proposed by Qi et al. [15], for our deblurring experiments. This dataset includes both synthetic and real-world data. For synthetic data, it encompassed five challenging deblurring scenes, with the simulation of motion blur incorporating scenarios such as camera shake and variations in camera motion speed during exposure, all designed to emulate realistic motion blur effects. For real-world data, it included two scenes: Bar and Classroom, captured using the DAVIS-346 color event camera [41]. Both scenes were recorded under low-light conditions using a handheld camera. Unlike EBAD-NeRF [15], which evaluates using only four ground-truth images from specific viewpoints, our experiments incorporated all 28 newly synthesized images from novel viewpoints for the Bar scene and all 16 ground-truth images for the Classroom scene. Additionally, we used the deblurred images generated by the EDI algorithm, along with novel view synthesized ground truth images, as inputs to COLMAP to re-estimate camera poses. This process successfully provided initial camera poses and sparse SfM point clouds, which were used to initialize the 3D-GS.

4.1.2 Experimental Setup. For motion blur modeling, we adopt the same number of discrete latent sharp images ?? as EBAD-NeRF [15] to ensure consistency and fair comparison. Specifically, we set ?? = 5 for synthetic datasets and $n = 7$ for real-world sequences. Regarding the loss function weight coefficients, we set $\lambda _ { \mathrm { b l u r } } = 1$ , $\lambda _ { \mathrm { E D I } } = 1 , \lambda _ { \mathrm { e v } } = 0 . 1$ , and ??SSIM is set to 0.2. To ensure sufficient model convergence and stable performance, we train the network for 20,000 iterations.

<!-- image-->  
(a) Factory

<!-- image-->  
(b) GT

<!-- image-->  
(c) ??blur

<!-- image-->  
(d) ??ev + ??EDI

<!-- image-->  
(e) ??blur + ??ev

<!-- image-->  
(f) ??blur + ??EDI

<!-- image-->  
(g) ??total

Figure 2: Qualitative novel views rendering results of different loss functions on a synthetic dataset.  
<!-- image-->  
(a) Bar

<!-- image-->  
(b) GT

<!-- image-->  
(c) ??blur

<!-- image-->  
(d) $L _ { \mathbf { e v } } + L _ { \mathbf { E D I } }$

<!-- image-->  
(e) ??blur + ??ev

<!-- image-->  
(f) $L _ { \mathbf { b l u r } } + L _ { \mathbf { E D I } }$

<!-- image-->  
(g) ??total  
Figure 3: Qualitative novel views rendering results of different loss functions on real dataset.

Table 1: Ablation study of different loss combinations on synthetic and real datasets.
<table><tr><td rowspan="2">Loss Combination</td><td colspan="3">Synthetic Dataset</td><td colspan="3">Real Dataset</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td> $L _ { \mathrm { b l u r } }$ </td><td>26.89</td><td>0.803</td><td>0.258</td><td>30.24</td><td>0.858</td><td>0.148</td></tr><tr><td> $L _ { \mathrm { b l u r } } + L _ { \mathrm { e v } }$ </td><td>29.19</td><td>0.874</td><td>0.159</td><td>31.08</td><td>0.902</td><td>0.162</td></tr><tr><td> $L _ { \mathrm { b l u r } } + L _ { \mathrm { E D I } }$ </td><td>28.72</td><td>0.841</td><td>0.144</td><td>30.47</td><td>0.878</td><td>0.132</td></tr><tr><td> $L _ { \mathrm { e v } } + L _ { \mathrm { E D I } }$ </td><td>25.00</td><td>0.814</td><td>0.231</td><td>30.04</td><td>0.897</td><td>0.176</td></tr><tr><td> $L _ { \mathrm { t o t a l } }$ </td><td>29.99</td><td>0.875</td><td>0.119</td><td>31.17</td><td>0.903</td><td>0.161</td></tr></table>

4.1.3 Quantitative Evaluation Metrics. We employ Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM)[42], and AlexNet-based Learned Perceptual Patch Similarity (LPIPS) [43, 44] to evaluate the similarity between rendered images and ground truth images.

## 4.2 Ablation Study

To investigate the collaborative mechanism of the multi-modal loss functions, we conduct a systematic ablation study using a controlled variable approach. Table 1 shows the quantitative impact of different loss combinations on both synthetic and real-world sequences, with results averaged over all evaluated scenes. Figures 2 and 3 illustrate the corresponding visual results, clearly showcasing the performance of each method under novel view rendering scenarios.

4.2.1 Experimental Analysis on Synthetic Dataset. When only using the motion blur loss $L _ { \mathrm { b l u r } } .$ , the deblurring performance of the model is relatively poor, with the reconstructed images exhibiting noticeable detail blurring. This is mainly due to the lack of constraints on the intensity changes during exposure. After introducing the event loss $L _ { \mathrm { e v } }$ , both PSNR and SSIM improve, indicating that the event stream effectively supplements the dynamic information missing from the standard camera and alleviates the loss of highfrequency details. However, the LPIPS remains at 0.159, which is still higher than that of the combination using ??EDI, suggesting limited improvement in detail restoration.Replacing $L _ { \mathrm { e v } }$ with the event double integral image loss ??EDI reduces the LPIPS to 0.144 and results in sharper edge details. However, the PSNR drops to 28.72, indicating insufficient global luminance consistency.

4.2.2 Experimental Analysis on Real Dataset. After combining the three losses, the metrics show the best overall performance. The complementary nature of the three losses is equally effective in real-world scenarios. $L _ { \mathrm { b l u r } }$ constrains the global motion blur, $L _ { \mathrm { e v } }$ constrains the intensity variation during the exposure time, and ??EDI enhances edge details. As a result, the fused model achieves the optimal reconstruction quality in real scenes. Although the LPIPS score slightly increases compared to the combination of $L _ { \mathrm { b l u r } } + .$ ??EDI, this small fluctuation is acceptable, and the overall performance still outperforms the other combinations.

<!-- image-->  
Figure 4: Qualitative novel view synthesis results of different methods.

## 4.3 Comparison with Existing Methods

4.3.1 Experimental Comparison on Synthetic Dataset. This experiment comprehensively compares the quantitative results of different methods on the new viewpoint synthesis task for synthetic sequences (as shown in Table 2), and also provides the results of new viewpoint rendering (Figure 4) as qualitative evaluation references. First, the 3D-GS method directly utilizes blurred images for reconstruction, which results in noticeable blurring in the new viewpoint rendering results. Quantitative results show that this method performs relatively poorly in terms of image quality, structural similarity, and perceptual quality, requiring improvement. Secondly, the 3D-GS+MPRNet method achieves some improvements by replacing the blurred images with those deblurred by the MPRNet[45] module. The results show that SSIM increases to 0.665, indicating that the structural similarity of the image has improved. However, PSNR and LPIPS slightly decrease, suggesting that MPRNet still has limited effectiveness in handling severe motion blur scenarios. In contrast, BAD-GS[23] significantly improves the indicators and shows strong performance in motion blur modeling. Furthermore, the BAD-GS cubic method, which uses bilinear B-spline interpolation for trajectory recovery on top of BAD-GS, results in worse performance. In our figures and tables, we denote this variant as BAD-GS\* for clarity. Nevertheless, its performance is still better than 3D-GS and 3D-GS+MPRNet, indicating that bilinear B-spline interpolation has a limited impact on the synthetic dataset and does not significantly degrade performance. The 3D-GS+EDI method, using the event-driven deblurring algorithm EDI, significantly improves image quality, demonstrating the effectiveness of event information. The EBAD-NeRF method, by combining motion blur modeling with event flow supervision of light intensity changes during exposure and optimizing the camera pose over the exposure time in a neural radiance field, shows near-optimal performance. However, the rendering speed of EBAD-NeRF is limited by the NeRF query MLP, preventing real-time rendering. Finally, our proposed method performs the best among all the tested methods. Although the SSIM of 0.871 is slightly lower than EBAD-NeRF, it performs outstandingly in PSNR and LPIPS, the two key indicators. More importantly, our method supports real-time rendering, which gives it a significant advantage in practical applications.

Table 2: Comparison of Different Methods on Synthetic Dataset
<table><tr><td>Scene</td><td>Metric</td><td>3D-GS</td><td>3D-GS+MPR</td><td>BAD-GS</td><td>BAD-GS*</td><td>3D-GS+EDI</td><td>EBAD-NeRF</td><td>Our Method</td></tr><tr><td rowspan="3">Cozyroom</td><td>PSNRâ</td><td>27.84</td><td>26.61</td><td>31.12</td><td>29.53</td><td>30.19</td><td>30.60</td><td>32.68</td></tr><tr><td>SSIMâ</td><td>0.855</td><td>0.840</td><td>0.930</td><td>0.905</td><td>0.898</td><td>0.948</td><td>0.935</td></tr><tr><td>LPIPSâ</td><td>0.222</td><td>0.239</td><td>0.084</td><td>0.130</td><td>0.104</td><td>0.114</td><td>0.067</td></tr><tr><td rowspan="3">Factory</td><td>PSNRâ</td><td>20.30</td><td>21.38</td><td>21.23</td><td>22.33</td><td>25.46</td><td>28.25</td><td>28.79</td></tr><tr><td>SSIMâ</td><td>0.539</td><td>0.590</td><td>0.628</td><td>0.680</td><td>0.788</td><td>0.915</td><td>0.865</td></tr><tr><td>LPIPSâ</td><td>0.486</td><td>0.434</td><td>0.342</td><td>0.339</td><td>0.178</td><td>0.158</td><td>0.129</td></tr><tr><td rowspan="3">Pool</td><td>PSNRâ</td><td>27.75</td><td>27.65</td><td>29.19</td><td>29.15</td><td>30.15</td><td>31.62</td><td>32.92</td></tr><tr><td>SSIMâ</td><td>0.733</td><td>0.742</td><td>0.787</td><td>0.797</td><td>0.807</td><td>0.924</td><td>0.881</td></tr><tr><td>LPIPSâ</td><td>0.363</td><td>0.350</td><td>0.228</td><td>0.257</td><td>0.175</td><td>0.156</td><td>0.106</td></tr><tr><td rowspan="3">Tanabata</td><td>PSNRâ</td><td>19.46</td><td>19.57</td><td>20.49</td><td>20.76</td><td>24.09</td><td>25.45</td><td>26.93</td></tr><tr><td>SSIMâ</td><td>0.570</td><td>0.592</td><td>0.669</td><td>0.670</td><td>0.670</td><td>0.890</td><td>0.838</td></tr><tr><td>LPIPSâ</td><td>0.489</td><td>0.435</td><td>0.351</td><td>0.382</td><td>0.240</td><td>0.206</td><td>0.151</td></tr><tr><td rowspan="3">Wine</td><td>PSNRâ</td><td>20.32</td><td>20.35</td><td>23.36</td><td>21.33</td><td>25.35</td><td>26.72</td><td>28.67</td></tr><tr><td>SSIMâ</td><td>0.564</td><td>0.564</td><td>0.732</td><td>0.650</td><td>0.755</td><td>0.889</td><td>0.852</td></tr><tr><td>LPIPSâ</td><td>0.496</td><td>0.479</td><td>0.290</td><td>0.363</td><td>0.246</td><td>0.196</td><td>0.140</td></tr><tr><td rowspan="3">Average</td><td>PSNRâ</td><td>23.13</td><td>23.11</td><td>25.08</td><td>24.62</td><td>27.05</td><td>28.53</td><td>30.00</td></tr><tr><td>SSIMâ</td><td>0.652</td><td>0.665</td><td>0.749</td><td>0.740</td><td>0.784</td><td>0.913</td><td>0.874</td></tr><tr><td>LPIPSâ</td><td>0.411</td><td>0.387</td><td>0.259</td><td>0.294</td><td>0.189</td><td>0.166</td><td>0.119</td></tr></table>

Table 3: Comparison of Different Methods on Real Dataset
<table><tr><td>Scene</td><td>Metric</td><td>3D-GS</td><td>3D-GS+MPR</td><td>BAD-GS</td><td>BAD-GS*</td><td>3D-GS+EDI</td><td>EBAD-NeRF</td><td>Our Method</td></tr><tr><td>Bar</td><td>PSNRâ</td><td>25.07</td><td>24.23</td><td>28.08</td><td>28.03</td><td>28.16</td><td>28.03</td><td>29.87</td></tr><tr><td></td><td>SSIMâ</td><td>0.744</td><td>0.731</td><td>0.781</td><td>0.780</td><td>0.774</td><td>0.807</td><td>0.880</td></tr><tr><td></td><td>LPIPSâ</td><td>0.442</td><td>0.475</td><td>0.226</td><td>0.245</td><td>0.259</td><td>0.209</td><td>0.221</td></tr><tr><td>Classroom</td><td>PSNRâ</td><td>23.37</td><td>23.07</td><td>30.64</td><td>31.51</td><td>27.86</td><td>30.72</td><td>32.47</td></tr><tr><td></td><td>SSIMâ</td><td>0.751</td><td>0.751</td><td>0.857</td><td>0.879</td><td>0.852</td><td>0.908</td><td>0.926</td></tr><tr><td></td><td>LPIPSâ</td><td>0.359</td><td>0.344</td><td>0.137</td><td>0.110</td><td>0.148</td><td>0.121</td><td>0.102</td></tr><tr><td>Average</td><td>PSNRâ</td><td>24.22</td><td>23.65</td><td>29.36</td><td>29.77</td><td>28.01</td><td>29.37</td><td>31.17</td></tr><tr><td></td><td>SSIMâ</td><td>0.748</td><td>0.741</td><td>0.819</td><td>0.830</td><td>0.813</td><td>0.857</td><td>0.903</td></tr><tr><td></td><td>LPIPSâ</td><td>0.400</td><td>0.409</td><td>0.182</td><td>0.178</td><td>0.203</td><td>0.165</td><td>0.161</td></tr></table>

4.3.2 Experimental Comparison on Real Dataset. In this experiment, we conducted a comparative analysis of different methods for the new view synthesis task on real sequences, and provided quantitative evaluation results (as shown in Table 3) and qualitative evaluation results (as shown in Figure 4). Firstly, the 3D-GS method performs relatively poorly on real sequences due to the blur present in the input images. Quantitative results show that the image quality, structural similarity, and perceptual quality in real scenes are at moderate levels, with considerable room for improvement. Secondly, the 3D-GS+MPRNet method shows a decline in quantitative results. This may be because the MPRNet module lacks generalization ability in handling the complex blur situations in real scenes, leading to no significant improvement in overall performance. In contrast, the BAD-GS method shows strong performance on real sequences, indicating that the method can effectively model the blurry physical processes in real scenes, thus achieving significant improvements in image quality, structural similarity, and perceptual quality. Furthermore, the BAD-GS\* method further enhances performance. PSNR reaches 29.77, SSIM increases to 0.830, and LPIPS decreases to 0.178, indicating that the bilinear B-spline interpolation method can effectively optimize trajectory recovery in real sequences, further improving overall performance. The 3D-GS+EDI method achieves a good performance on real sequences by using the EDI deblurring algorithm. Although the performance is slightly lower than BAD-GS and BAD-GS\*, it still outperforms 3D-GS and 3D-GS+MPRNet. The results indicate that event-driven deblurring algorithms have good deblurring effects in real scenes. However, because this method does not simultaneously optimize camera poses and scene representation, its performance still has some limitations. The EBAD-NeRF method also demonstrates strong performance on real datasets. While it excels in terms of image quality and structural similarity, its performance on the LPIPS metric still leaves room for improvement. The discrepancy in the metrics reported for the EBAD-NeRF method compared to the original paper arises from the use of real datasets with more viewpoints in our experiments. Quantitative results show that our proposed method performs the best on real sequences, effectively improving image reconstruction in real scenes.

## 5 Conclusion

In summary, the proposed EBAD-Gaussian method fully leverages the complementary advantages of event streams and images, effectively addressing the reconstruction quality limitations of 3D-GS when dealing with severe motion blur. Based on a physically grounded motion blur model, we introduce three types of key supervisory signals: motion blur constraints to ensure baseline reconstruction quality, event stream supervision to recover highfrequency details, and the EDI prior to enhance physical consistency. On this basis, our framework jointly optimizes the Gaussian parameters and the camera poses during exposure, thereby achieving high-fidelity 3D scene reconstruction. Experimental results on both synthetic and real datasets validate the effectiveness of EBAD-Gaussian in camera pose estimation and scene recovery.

## References

[1] Zhenyu Tang, Junwu Zhang, Xinhua Cheng, Wangbo Yu, Chaoran Feng, Yatian Pang, Bin Lin, and Li Yuan. Cycle3d: High-quality and consistent image-to-3d generation via generation-reconstruction cycle. arXiv preprint arXiv:2407.19548, 2024.

[2] Steve Chi-Yin Yuen, Gallayanee Yaoyuneyong, and Erik Johnson. Augmented reality: An overview and five directions for ar in education. Journal of Educational Technology Development and Exchange (JETDE), 4(1):11, 2011.

[3] Faiza Gul, Wan Rahiman, and Syed Sahal Nazli Alhady. A comprehensive study for robot navigation techniques. Cogent Engineering, 6(1):1632046, 2019.

[4] Yufei Deng, Rong Xiao, Jiaxin Li, and Jiancheng Lv. Semantic-pixel associative information improving loop closure detection and experience map building for efficient visual representation. In International Conference on Neural Information Processing, pages 393â404. Springer, 2023.

[5] Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, and Qing Li. Semantic gaussians: Open-vocabulary scene understanding with 3d gaussian splatting, 2024.

[6] Hongyu Zhou, Jiahao Shao, Lu Xu, Dongfeng Bai, Weichao Qiu, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi Liao. Hugs: Holistic urban 3d scene understanding via gaussian splatting, 2024.

[7] Peng Dai, Feitong Tan, Xin Yu, Yifan Peng, Yinda Zhang, and Xiaojuan Qi. Go-nerf: Generating objects in neural radiance fields for virtual reality content creation, 2024.

[8] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 300â309, 2023.

[9] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering, 2023.

[10] Guikun Chen and Wenguan Wang. A Survey on 3D Gaussian Splatting, April 2024. arXiv:2401.03890 [cs].

[11] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104â4113, 2016.

[12] Lingzhe Zhao, Peng Wang, and Peidong Liu. Bad-gaussians: Bundle adjusted deblur gaussian splatting, 2024.

[13] Yunshan Qi, Lin Zhu, Yu Zhang, and Jia Li. E2nerf: Event enhanced neural radiance fields from blurry images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13254â13264, 2023.

[14] Marco Cannici and Davide Scaramuzza. Mitigating motion blur in neural radiance fields with events and frames, 2024.

[15] Yunshan Qi, Lin Zhu, Yifan Zhao, Nan Bao, and Jia Li. Deblurring neural radiance fields with event-driven bundle adjustment. In Proceedings of the 32nd ACM International Conference on Multimedia, MM â24, page 9262â9270. ACM, October 2024.

[16] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers. E-nerf: Neural radiance fields from a moving event camera, 2023.

[17] Guillermo Gallego, Tobi Delbruck, Garrick Orchard, Chiara Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger, Andrew J. Davison, Jorg Conradt, Kostas Daniilidis, and Davide Scaramuzza. Event-based vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(1):154â180, January 2022.

[18] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis, 2020.

[19] Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley, Miaomiao Liu, and Yuchao Dai. Bringing a blurry frame alive at high frame-rate with an event camera, 2018.

[20] Li Ma, Xiaoyu Li, Jing Liao, Qi Zhang, Xuan Wang, Jue Wang, and Pedro V Sander. Deblur-nerf: Neural radiance fields from blurry images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12861â12870, 2022.

[21] Dogyoon Lee, Minhyeok Lee, Chajin Shin, and Sangyoun Lee. Dp-nerf: Deblurred neural radiance field with physical scene priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12386â 12396, June 2023.

[22] Peng Wang, Lingzhe Zhao, Ruijie Ma, and Peidong Liu. BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4170â4179, June 2023.

[23] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for Gaussian splatting. arXiv preprint arXiv:2409.06765, 2024.

[24] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splatting as markov chain monte carlo. In Advances in Neural Information Processing Systems (NeurIPS), 2024. Spotlight Presentation.

[25] Tianyi Xiong, Jiayi Wu, Botao He, Cornelia Fermuller, Yiannis Aloimonos, Heng Huang, and Christopher A Metzler. Event3dgs: Event-based 3d gaussian splatting for fast egomotion. arXiv preprint arXiv:2406.02972, 2024.

[26] Bharatesh Chakravarthi, Aayush Atul Verma, Kostas Daniilidis, Cornelia Fermuller, and Yezhou Yang. Recent event camera innovations: A survey. arXiv preprint arXiv:2408.13627, 2024.

[27] Haiqian Han, Jianing Li, Henglu Wei, and Xiangyang Ji. Event-3dgs: Event-based 3d reconstruction using 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:128139â128159, 2024.

[28] Viktor Rudnev, Gereon Fox, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. Dynamic eventnerf: Reconstructing general dynamic scenes from multi-view event cameras. arXiv preprint arXiv:2412.06770, 2024.

[29] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. Eventnerf: Neural radiance fields from a single colour event camera, 2022.

[30] Inwoo Hwang, Junho Kim, and Young Min Kim. Ev-nerf: Event based neural radiance field. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 837â847, January 2023.

[31] Yuanjian Wang, Yufei Deng, Rong Xiao, Jiahao Fan, Chenwei Tang, Deng Xiong, and Jiancheng Lv. Saenerf: Suppressing artifacts in event-based neural radiance fields. arXiv preprint arXiv:2504.16389, 2025.

[32] Weng Fei Low and Gim Hee Lee. Robust e-nerf: Nerf from sparse and noisy events under non-uniform motion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[33] Weng Fei Low and Gim Hee Lee. Deblur e-nerf: Nerf from motion-blurred events under high-speed or low-light conditions. In European Conference on Computer Vision (ECCV), 2024.

[34] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers. E-nerf: Neural radiance fields from a moving event camera. IEEE Robotics and Automation Letters, 8(3):1587â1594, 2023.

[35] Yunshan Qi, Lin Zhu, Yifan Zhao, Nan Bao, and Jia Li. Deblurring neural radiance fields with event-driven bundle adjustment. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 9262â9270, 2024.

[36] Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley, Miaomiao Liu, and Yuchao Dai. Bringing a blurry frame alive at high frame-rate with an event camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6820â6829, 2019.

[37] Johannes Lutz SchÃ¶nberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[38] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

[39] Nelson Max. Optical models for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics, 1(2):99â108, 1995.

[40] James T Kajiya and Brian P Von Herzen. Ray tracing volume densities. ACM SIGGRAPH computer graphics, 18(3):165â174, 1984.

[41] Gemma Taverni, Diederik Paul Moeys, Chenghan Li, Celso Cavaco, Vasyl Motsnyi, David San Segundo Bello, and Tobi Delbruck. Front and back illuminated dynamic and active pixel vision sensors comparison. IEEE Transactions on Circuits and Systems II: Express Briefs, 65(5):677â681, 2018.

[42] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image

Processing, 13(4):600â612, 2004.

[43] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[44] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. Commun. ACM, 60(6):84â90, May 2017.

[45] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Multi-stage progressive image restoration. In CVPR, 2021.