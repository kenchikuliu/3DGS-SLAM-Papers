# E-4DGS: High-Fidelity Dynamic Reconstruction from the Multi-view Event Cameras

Chaoran Fengâ   
School of Electronic and Computer Engineering, Peking University   
chaoran.feng@stu.pku.edu.cn Yian Zhao   
School of Electronic and   
Computer Engineering, Peking University   
zhaoyian@stu.pku.edu.cn Zhenyu Tangâ   
School of Electronic and   
Computer Engineering, Peking University   
zhenyutang@stu.pku.edu.cn   
Jianbin Zhao   
School of Future   
Technology, Dalian   
University of Technology   
1518272584@mail.dlut.edu.cn

Wangbo Yuâ School of Electronic and Computer Engineering, Peking University wbyu@stu.pku.edu.cn

Li Yuanâ  School of Electronic and Computer Engineering, Peking University yuanli-ece@pku.edu.cn

Yatian Pang   
National University of   
Singapore   
yatian_pang@u.nus.edu

Yonghong Tianâ  School of Electronic and Computer Engineering, Peking University yhtian@pku.edu.cn

<!-- image-->  
Multi-view High-speed Event Camera Trajectory  
High-Fidelity Dynamic Reconstruction

Figure 1: Our E-4DGS reconstructs temporally consistent and photorealistic dynamic scenes using event streams and sparse RGB frames captured from multi-view moving cameras, effectively handling complex motion and lighting variations.

## Abstract

Novel view synthesis and 4D reconstruction techniques predominantly rely on RGB cameras, thereby inheriting inherent limitations such as the dependence on adequate lighting, susceptibility to motion blur, and a limited dynamic range. Event cameras, offering advantages of low power, high temporal resolution and high dynamic range, have brought a new perspective to addressing the scene reconstruction challenges in high-speed motion and lowlight scenes. To this end, we propose E-4DGS, the first event-driven dynamic Gaussian Splatting approach, for novel view synthesis from multi-view event streams with fast-moving cameras. Specifically, we introduce an event-based initialization scheme to ensure stable training and propose event-adaptive slicing splatting for time-aware reconstruction. Additionally, we employ intensity importance pruning to eliminate floating artifacts and enhance 3D consistency, while incorporating an adaptive contrast threshold for more precise optimization. We design a synthetic multi-view camera setup with six moving event cameras surrounding the object in a 360-degree configuration and provide a benchmark multi-view event stream dataset that captures challenging motion scenarios. Our approach outperforms both event-only and event-RGB fusion baselines and paves the way for the exploration of multi-view eventbased reconstruction as a novel approach for rapid scene capture. The code and dataset are available on the project page.

## CCS Concepts

â¢ Computing methodologies â Reconstruction.

## Keywords

Event-driven 4D Reconstruction, 3D Gaussian Splatting, Novel View Synthesis, High-speed Robot Egomotion.

## ACM Reference Format:

Chaoran Feng, Zhenyu Tang, Wangbo Yuâ, Yatian Pang, Yian Zhao, Jianbin Zhao, Li Yuan, and Yonghong Tian. 2025. E-4DGS: High-Fidelity Dynamic Reconstruction from the Multi-view Event Cameras. In Proceedings of the 33rd ACM International Conference on Multimedia (MM â25), October 27â31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 16 pages. https://doi.org/ 10.1145/3746027.3754777

## 1 Introduction

Novel view synthesis (NVS) and dynamic scene reconstruction are critical for immersive applications such as virtual and augmented reality (VR/AR) [30, 84], scene understanding [5, 77], 3D content creation [7, 31, 45, 63, 91], and autonomous driving tasks [17, 78, 86]. While Neural Radiance Fields (NeRF) [42] has recently achieved remarkable success in photorealistic rendering of static scenes, their extension to dynamic scenarios remains challengingâprimarily due to substantial training time. In contrast, 3D Gaussian Splatting (3DGS) [27] provides notable advantages in real-time rendering and significantly faster training. Yet, existing dynamic extensions of 3DGS struggle to handle scenes with fast motion effectively, primarily due to the inherent limitations of RGB cameras, which, owing to their high latency and limited dynamic range, are prone to motion blur when capturing fast-moving scenes.

Compared to RGB cameras that capture images at fixed intervals, event cameras operate asynchronously by recording brightness changes as event spikes with microsecond-level latency, offering extremely low latency and high dynamic range [13, 58, 66] Owing to such advantageous, event cameras have recently been adopted for novel view synthesis and scene reconstruction tasks [75]. For example, event-driven NeRF methods [12, 26, 29, 53] leverage event accumulation frames and depend on known or estimated camera trajectories to reconstruct NeRF representation. In parallel, eventdriven 3DGS approaches [19, 25, 33, 70, 83] utilize the sharp structural information provided by event streams to reconstruct 3DGS representation, enabling efficient rendering and training. However, these methods are primarily designed for static scene reconstruction and are not well-suited for modeling dynamic environments. In the more challenging task of dynamic scene reconstruction, relying solely on a single event camera inherently limits the ability to capture complete scene dynamicsâespecially in scenarios involving fast motion, large deformations, or severe occlusions. Moreover, the coupling between object and camera motion can often lead to mutual cancellation of contrast changes, resulting in neutralized events [9, 14, 19] that obscure fine-grained geometric details.

Based on the above observation, we aim to investigate the following research question: How can we efficiently reconstruct a highfidelity dynamic scenes using multi-view fast-moving event cameras? With the captured multi-view event streams, a straightforward approach is to adopt a two-stage pipeline: First, reconstructing intensity frames from the event streams using E2VID [9, 52] and obtain Gaussian initialization points from COLMAP [56] ; Then, applying an off-the-shelf reconstruction method for futher reconstruction [8, 79, 81, 82]. However, this naÃ¯ve solution compromises the temporal precision and sparsity of event data by converting it into intensity frames, introducing accumulation error and extensive costs, resulting in degraded reconstruction consistency.

To this end, we propose E-4DGS, an end-to-end event-based framework for high-fidelity dynamic 3D reconstruction from multiview event streams. To address the initialization challenge under sparse event observations, we introduce an event-specific strategy to generate stable Gaussian primitives without relying on RGBbased Structure from Motion (SfM). We further design an eventadaptive slicing mechanism that segments and accumulates event streams for accurate supervision, and propose a multi-view 3D consistency regularization to enhance structural alignment. Additionally, E-4DGS supports optional refinement using a few motionblurred RGB frames. To our knowledge, this is the first event-only framework enabling view-consistent 3D Gaussian reconstruction in dynamic scenes. For evaluation, we introduce a multi-view synthetic event dataset that serves as a benchmark for dynamic scene reconstruction. The dataset encompasses a diverse set of dynamic scenes with simultaneous camera and object motion, ranging from "mild" to "strong". We compare our method against two-stage baselines that utilize E2VID for intensity reconstruction followed by frame-based methods, trained either with event streams alone or with a combination of RGB videos and event sequences. Our approach significantly outperforms all baselines, achieving state-ofthe-art results while enabling continuous and temporally coherent reconstruction of dynamic scenes. These results demonstrate that operating directly on raw event data, especially under challenging conditions with camera motion, yields higher-fidelity dynamic scene reconstruction compared to methods relying on reconstructed RGB frames. To summarize, the main contributions are as follows:

â¢ We present E-4DGS, the event-driven approach for reconstructing adynamic 3D Gaussian Splatting representation from multi-view event streams.

â¢ We introduce an event-based initialization scheme for stable training, propose event-adaptive slicing splatting and adaptive event threshold for supervision, and design intensity importance pruning to enhance 3D consistency.

â¢ We construct a multi-view synthetic dataset with moving cameras for 4D reconstruction from event streams. Our method achieves state-of-the-art performance, and we will release our work to support future research.

## 2 Related Work

## 2.1 Dynamic Reconstruction from RGB Frames

Modeling dynamic scenes from moving RGB cameras alone is still a challenging open task in computer vision. A widely used approach to this problem is to learn coordinate-based neural scene representations allowing rendering novel views and representing dynamic scenes. Previous works such as neural radiance field (NeRF) and its variants D-NeRF [49] and more [3, 34, 46, 47, 80] used implicit neural representations in combination with volume rendering. They are based on Multi-Layer Perceptrons (MLPs), which are relatively compact and require minimal storage space once trained. However, they are expensive to optimize and lead to slow training and evaluation which limits its expansion on the realtime rendering and real-world applications. The recently emerging 3DGS [27] and its variants [4, 18, 62, 87] have reshaped the landscape of dynamic radiance fields due to its efficiency and flexibility. The pioneering work Deformable3DGS (D3DGS) [81] enhances dynamic Gaussian representations with a tiny deformable field for tracking the motion of Gaussian points. Similarly, other methods [21, 38, 44, 57, 65, 68, 69, 82] models Gaussian motion using point-tracking functions for stable point moving. Our approach adopts D3DGS as the dynamic representation due to its simple and efficient structure, and then presents its application to the supervision from event streams. It inherits thereby the advantages of event streams and 3DGS for dynamic view synthesis.

## 2.2 Dynamic Reconstruction from Event Data

Event cameras have been widely used to reconstruct dynamic scenes from non-blurry RGB Frames of fast motion. Previous works, including model-based methods [43, 53] and learning-based methods [22, 52, 64], process event and RGB frames with 2D priors but lack 3D consistency. Other event-based methods address tasks such as detection, tracking, and image/3D reconstruction, including lip reading [55, 60], object tracking [73, 94], and pose estimation [16, 33, 95]. However, these methods still do not incorporate 3D priors to reconstruct scene appearance and are not applicable to represent 3D scenes, which is our goal of proposed E-4DGS.

For static scene reconstruction, recent event-based methods [2, 25, 32, 37, 50, 61, 67, 71, 74, 83, 85, 85, 88, 92] have achieved highfidelity 3D reconstruction and NVS tasks using supervision from event pixels or event accumulation. These methods primarily rely on consistent event sequences from a single mono-event camera. However, extending static scene representations to dynamic scenes with event streams is a challenging task, as the movement of objects and the simultaneous motion of the event camera can introduce ambiguity in the events. Different from only a single mono-camera setting, our proposed E-4DGS reconstructs the dynamic scene with the multi-view camera setting, providing more multi-view consistency details.

Recently, a growing trend is the use of dynamic neural radiance fields (DNeRF) or Dynamic 3DGS (4DGS) for dynamic scene representation and novel view synthesis. DE-NeRF [39] and EBGS [76] reconstruct dynamic scenes using monocular event streams and RGB frames from a moving camera, modeling deformations in a canonical space. The former is based on DNeRF, while the latter relies on 4DGS. EvDNeRF [1], which is based on canonical volumes, and DynEventNeRF [54], which uses temporally-conditioned MLPbased NeRF, both utilize multi-view event streams to reconstruct dynamic scenes. However, the former does not model appearance, and the latter is trained slowly due to volume rendering. In contrast, our proposed E-4DGS achieves higher-quality reconstruction by accurately capturing complex geometries and lighting effects than NeRF-based models, while also offering fast training and inference speeds for real-time, real-world applications.

## 3 Preliminaries

## 3.1 Deformable 3D Gaussian Splatting

Deformable3DGS [81] offers an explicit method for representing a 4D dynamic scene G with the canonical space and th deformable space based on 3D Gaussian Splatting [27]. In the canonical space, these 3D Gaussian points have the following parameters: mean point $\mu ,$ covariance matrix Î£, opacity ??, and color c and a 3D Gaussian point $G ( x ) \in \mathbb { G }$ is defined as follows:

$$
G ( x ) = e ^ { - { \frac { 1 } { 2 } } ( x - \mu ) { } ^ { T } \Sigma ^ { - 1 } ( x - \mu ) }\tag{1}
$$

where, Î£ is divided into two learnable components: the quaternion ?? represents rotation, and the 3D-vector ?? represents scaling. then, the color of each pixel can be calculated using the following formula:

$$
C ( x ) = \sum _ { i \in N ( x ) } c _ { i } \alpha _ { i } ( x ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } ( x ) \right) ,\tag{2}
$$

where $\begin{array} { r } { \alpha _ { i } ( x ) = \sigma _ { i } \exp \left( - \frac { 1 } { 2 } ( x - \mu _ { i } ^ { 2 D } ) ^ { T } \Sigma ^ { - 1 } ( x - \mu _ { i } ^ { 2 D } ) \right) } \end{array}$ , and ?? is the number of Gaussian points that intersect with the pixel ??.

In the deformable space, Deformable3DGS employ a compact MLP layer to represent motion of Gaussian points. Given timestamp ?? and center position ?? of 3D Gaussians as inputs, the deformation MLP produces offsets, which subsequently transform the canonical 3D Gaussians to the deformed space:

$$
( \Delta x , \Delta r , \Delta s ) = \mathcal { F } _ { \theta } ( \gamma ( \mathsf { s g } ( x ) ) ) , \gamma ( t ) )\tag{3}
$$

where $s g ( \cdot )$ indicates a stop-gradient operation, ?? denotes the positional encoding as defined in [81]. Therefore, a dynamic Gaussian point can be represented as $G ( x + \Delta x , r + \Delta r , s + \Delta s )$ at timestamp ?? .

## 3.2 Event Generation Model

A single event is represented as $\boldsymbol { e } _ { k } = ( x _ { k } , y _ { k } , p _ { k } , t _ { k } )$ in the event streams $\varepsilon ,$ denoting a brightness change registered by an event sensor at timestamp $t _ { k } .$ , pixel location $\mathbf { u } _ { \mathbf { k } } = ( x _ { k } , y _ { k } )$ in the event camera frame with polarity $p _ { k } \in \{ - 1 , + 1 \}$ . The change between adjacent timestamps can be calculated from intensity images ?? .

$$
L ( { \bf u } _ { k } , t _ { k } ) - L ( { \bf u } _ { k } , t _ { k - 1 } ) = \sum _ { t _ { k - 1 } < t \leq t _ { k } } p _ { t } C ^ { p _ { t } } \overset { \mathrm { d e f } } { = } \Delta E _ { { \bf u } _ { k } } ( t _ { k - 1 } , t _ { k } ) ,\tag{4}
$$

$$
\begin{array} { r } { \mathrm { w h e r e } \quad L = \log ( I ) . } \end{array}\tag{5}
$$

Here, the thresholds $C ^ { p } \in \{ C ^ { - 1 } , C ^ { + 1 } \}$ define boundaries for classifying the event as positive or negative, with the polarity of an event indicating a positive or negative change in logarithmic illumination.

Therefore, given a supervisory event stream $\varepsilon ,$ we can supervise our proposed E-4DGS by comparing the predicted brightness change $\Delta \hat { E } ( t _ { k - 1 } , t _ { k } )$ and the ground truth $\Delta E ( t _ { k - 1 } , t _ { k } )$ by Equation (4) over all image pixels. In general, we substitute intensity frames $\hat { I } _ { t }$ with the rendered results $\hat { C } _ { t }$ and can utilize photo-realistic loss [81] between the predicted intensity frames and the groundtruth event of event-based single integral (ESI) [53]:

$$
\mathcal { L } _ { g s } = \sum _ { \mathbf { u } _ { k } \in \hat { I } } ( \lambda \mathcal { L } _ { 1 } ( \Delta \hat { E } _ { \mathbf { u } _ { k } } , \Delta E _ { \mathbf { u } _ { k } } ) + ( 1 - \lambda ) \mathcal { L } _ { D - S S I M } ( \Delta \hat { E } _ { \mathbf { u } _ { k } } , \Delta E _ { \mathbf { u } _ { k } } ) )\tag{6}
$$

<!-- image-->  
Figure 2: The overview of E-4DGS. Our framework establishes temporal-coherent 4D representations through a cascaded processing of event streams: The event-driven initialization constructs spatio-temporal gaussians via center-focus density fields, followed by differentiable feature distillation where adaptive slicing operators disentangle high-frequency part for GS optimization. Cross-view consistency is then imposed through deformable Gaussian reprojection coupled with saliency pruning, while multi-modal alignment ultimately achieves photometric fidelity via kernel-attentive RGB-event synchronization.

## 4 Method

We propose E-4DGS, a method for high-fidelity dynamic scene reconstruction using sparse event camera streams. Given multiview event data capturing a dynamic scene, E-4DGS reconstructs a 4D model that allows novel view generation at arbitrary times. To address the challenges posed by the sparse nature of event data and the dynamic characteristics of the scene, we introduce an event-based initialization strategy (Section 4.1), an event-aware slicing splatting technique to preserve geometric details (Section 4.2), and multi-view 3D consistency regularization for improved scene fidelity (Section 4.3). Additionally, we utilize adaptive event supervision and color recovery to enhance the reconstruction quality (Section 4.4). The overview of our method is illustrated in Figure 2.

## 4.1 Event-based Initialization

The Gaussian primitives are initialized using a point cloud derived from Structure-from-Motion (SfM) [40] with RGB frames in the vanilla 3DGS. However, their performance is hindered by inaccurate dynamic Gaussian initialization due to view inconsistencies caused by object motion. Furthermore, applying SfM to extract Gaussian points from event sequences is more challenging than using RGB frames with COLMAP [56], due to the sparse nature of event streams. Some methods [19, 25, 70, 74] randomly initialize Gaussians within a fixed cube without considering unbounded scenes. Other methods perform better than random initialization but are more complex. Elite-3DGS [93] employs a two-stage approach with E2VID [52] to convert events into images, followed by SfM for point cloud initialization, while E-3DGS [83] uses exposure enhancement [59] method before obtaining the SfM points.

Thus, we adopt an event-specific strategy for Gaussian point initialization, balancing performance and efficiency. Specifically, 1) For object scenes, we initialize the point cloud with 100,000 points in a fixed cube, consistent with original 3DGS settings; 2)

<!-- image-->

<!-- image-->  
(a) Visualization of a sharp edge passing a pixel in an event sensor.  
(b) Photocurrent or intensity change at the pixel (xk ,yk ).  
Figure 3: Demonstration of how time window discretizations can influence the count of events between timestep pairs. The time window $( t _ { i } , t _ { i + 1 } )$ produces two negative events, whereas $\left( t _ { i } , t _ { i + 2 } \right)$ results in event neutralization.

For medium or large scenes, we employ a dense-to-sparse radiative sphere initialization, mimicking realistic distribution where point density is highest at the center and decreases toward the boundaries. We set sphereâs radius to $r = 1 0 . 0$ with 200,000 initial points.

We also experimented with initializing the Gaussian primitives using random pointcloud and E2VID+COLMAP, and further details are provided in the supplementary materials. While our approach yielded a slight performance drop than the E2VID+COLMAPâs performance, the latter requires more computational complexity.

## 4.2 Event-adaptive Slicing Splatting

In event-based scene reconstruction pipelines, the slicing strategy for the event stream significantly influences reconstruction quality. As the duration of the event time window $( t _ { i } , t _ { i + 1 } )$ increases, the predicted events become a discretized, aliased representation of the continuous brightness variations in the scene.

For instance, Figure 3 illustrates that measurements recorded by the event sensor between timestamps $( t _ { i } , t _ { i + 1 } )$ produce three negative events at the selected pixel, whereas measurements over the interval $\left( t _ { i } , t _ { i + 2 } \right)$ yield no events. This effect is particularly notable in our pipeline, as the process of accumulating polarity inherently neutralizes events. Moreover, existing works [53, 74] have demonstrated that using consistently short windows impedes the propagation of high-level illumination, while consistently long windows often result in a loss of local detail. While they randomly sampled the length of event timestamp window, a drawback is that it does not take into account the camera speed or event rate, causing the sampled windows to contain either too many or too few events. Additionally, Hu et al. [24] and Han et al. [20] revealed that regions with uniform and smooth intensities typically do not trigger any events, leading to spatial sparsity in the event streams used as supervisory signals.

Based on the aforementioned observations, we propose an eventadaptive slicing strategy to address this issue. Specifically, during the training of our E-4DGS, we deliberately vary the time window of batched events and incorporate event noise during the event accumulation process. Notably, these modifications lead to an improved generation of finely-sliced events at test time. The detailed process of event-adaptive slicing are as follow:

1) Event Accumulation Range Setting: For each timestamp, we randomly sample and slice a target number of events streams within the event count range $[ N _ { m i n } , N _ { m a x } ]$

2) Event Accumulation Jitter: During our sampling process, we add Gaussian noise to pixels that do not record any events within the whole event timestamp window. This augmentation enhances gradient optimization in smooth regions and increases overall robustness of the pipeline against noisy events. It serves as event sampling in [39], and the whole process is defined as:

$$
\Delta E _ { \mathbf { u } } ( t _ { s t a r t } , t _ { e n d } ) = \left\{ \begin{array} { l l } { \displaystyle \int _ { t _ { s } } ^ { t _ { e } } { p _ { \tau } C ^ { p _ { \tau } } } , d \tau } & { \mathrm { i f } \kappa _ { \mathrm { t r i g } } \neq 0 , } \\ { \Delta \cdot N \big ( 0 , \sigma _ { \mathrm { n o i s e } } ^ { 2 } \big ) } & { \mathrm { i f } \kappa _ { \mathrm { t r i g } } = 0 . } \end{array} \right.\tag{7}
$$

where, $\Delta E ( \cdot )$ denotes the event frame accumulated from all event polarities triggered at pixel coordinate u within the current event time window. $\mathbb { k } _ { \mathrm { t r i g } }$ denotes the spiking of the events. ??start, ??end, and $\Delta t = t _ { \mathrm { e n d } } - t _ { \mathrm { s t a r t } }$ represent the start timestamp, end timestamp, and the time interval of the event time window, respectively.

This strategy not only guarantees a diverse range of event window lengths, but also curtails the loss of fine details that can occur due to neutralization. Moreover, it helps preserve critical geomerty details, thereby enhancing the overall fidelity of the reconstruction.

## 4.3 Intensity Importance Pruning

In the vanilla Gaussian Splatting pipeline, the opacity of all Gaussian points is gradually reduced, and points with low transparency are pruned during the Gaussian pruning stage. However, this method is unsuitable for our event-based approach, as it results in excessive coupling between the canonical and deformation fields and simultaneous camera and object motion, further exacerbating the issue. Therefore, we eliminate the reset opacity operation same as in [10]. and drawing inspiration from LightGaussians [11], which emphasizes a compact representation of static scenes by pruning redundant Gaussians based on spatial attributes such as transparency and volume, we adopt a specialized strategy, Intensity Importance

<!-- image-->  
Figure 4: The process of the intensity importance pruning.

Pruning (IIP), to remove floaters across both the canonical and deformable spaces. With this strategy, the importance of each Gaussian point is computed for each training viewpoint at every timestamp. Gaussian primitives with an importance score below a fixed threshold are then pruned, effectively mitigating the floater issue and enhance the 3D consistency from multi-view event streams.

Specifically, for a Gaussian point $g _ { i } \in \mathbb { G } ,$ , the Gaussian importance ???? over the images I of all training views and timestamps T , is defined as follows:

$$
w _ { i } = \underset { \mathbf { x } \in I , t \in \mathcal { T } } { \operatorname { M a x } } \left( \alpha _ { i } \left( \mathbf { x } \mid t \right) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \left( \mathbf { x } \mid t \right) \right) \right) .\tag{8}
$$

Here, $I \in \mathcal { I }$ denotes the intensity image. We prune Gaussian points whose importance scores satisfy $w _ { i } < 0 . 0 1 5$ , following the approach [11]. As shown in Figure 4, our method effectively removes floating artifacts absent from the training views. In addition, we perform Gaussian cloning and splitting following the 3DGS protocol, ensuring that child Gaussian points inherit the dynamic characteristics of their parent Gaussian points.

## 4.4 Event Supervision and Optimization

Adaptive Event Supervision. According to previous work [35], the ground truth of the event contrast inherently contains some errors. Additionally, in the real scenes captured by an event sensor, the event constrast threshold $C ^ { p }$ varies due to the environment disturbance, which can make Equation. 5 impractical to use in a realworld setup. Thus, if we directly apply the photometric loss with Equation. 6 to compare the rendered intensity frames with those derived from event data, the inherent discrepancies will be strictly penalized during optimization, which may in fact degrade the overall reconstruction quality. To bridge the gap between synthetic and real event data, we introduce learnable threshold parameters $\hat { C }$ and compute the rendered intensity frame as follows:

$$
\Delta \hat { E } _ { { \bf u } _ { k } } ( t _ { k - 1 } , t _ { k } ) = \hat { L } ( { \bf u } _ { k } , t _ { k } ) - \hat { L } ( { \bf u } _ { k } , t _ { k - 1 } ) \stackrel { \mathrm { d e f } } { \simeq } \sum _ { t _ { k - 1 } < t \leq t _ { k } } \hat { p } _ { t } \hat { C } ,\tag{9}
$$

Here, we can simpfy this process as follows:

$$
N _ { g t } ( \cdot ) _ { t _ { k - 1 } } ^ { t _ { k } } = \frac { 1 } { C } \left( ( V ( \cdot , t _ { 2 } ) - V ( \cdot , t _ { 1 } ) ) \right) ,\tag{10}
$$

$$
N _ { \hat { p } r e d } ( \cdot ) _ { t _ { k - 1 } } ^ { t _ { k } } = \frac { 1 } { \hat { C } } \left( ( \hat { L } ( \cdot , t _ { k } ) - \hat { L } ( \cdot , t _ { k - 1 } ) ) \right) ,\tag{11}
$$

$$
\mathcal { L } _ { R e c o n } = \frac { 1 } { H \times W } \sum _ { \mathbf { u } \in \hat { L } } \sqrt { ( N _ { g t } ( \mathbf { u } ) - N _ { \hat { \rho } r e d } ( \mathbf { u } ) ) ^ { 2 } + \epsilon ^ { 2 } }\tag{12}
$$

Here, $V ( { \bf u } , t )$ denotes the photovoltage in event pixel u at timestamp $t ,$ and ?? is a small constant added for numerical stability. $N _ { g t } ( \mathbf { u } )$ and $N _ { p r e d } ( \mathbf { u } )$ represent the g.t. and predicted event count maps over $( t _ { k - 1 } , t _ { k } ] \mathrm { y }$ . The overall event supervision loss is given by:

$$
\mathcal { L } _ { \mathrm { E v e n t } } = \lambda _ { \mathrm { R e c o n } } \mathcal { L } _ { \mathrm { R e c o n } } + \lambda _ { \mathrm { T V } } \mathcal { L } _ { \mathrm { T V } } ,\tag{13}
$$

where ${ \mathcal { L } } _ { \mathrm { T V } }$ is a total variation regularization term encouraging spatial smoothness, and $\lambda _ { \mathrm { R e c o n } } , \lambda _ { \mathrm { T V } }$ are weighting factors balancing the contributions of each component.

Combined Gain and Offset Correction. Since event cameras only capture logarithmic intensity differences rather than absolute log-intensity values, the predicted log-intensity ??Ë from our 4DGS method is determined only up to an additive offset for each color channel. Moreover, there is a scale ambiguity in the reconstructed color balance and illumination of the scene, when only the event contrast threshold is known. Thus, itâs necessary to correct and align the color value for every color channel like previous works [37, 53, 89], using the correction formula as follow:

$$
\hat { L } ( \mathbf { u } _ { k } , t _ { k } ) \stackrel { \mathrm { d e f } } { = } g _ { c } \cdot \hat { L } ( \mathbf { u } _ { k } , t _ { k } ) + \Delta c ,\tag{14}
$$

where, $g _ { c }$ and Î?? are the color correction parameters, and derived via ordinary least squares [37] with the ground-truth log-intensity $L ( \mathbf { u } _ { k } , t _ { k } )$ as defined in Section 3.2. Notably, the images captured by a separate standard camera are affected by saturation in realworld scenes due to its limited dynamic range, and they are not raw recordings but have undergone lossy in-camera image processing. Moreover, the contrast threshold of real event cameras varies spatially across the image plane and temporally over time [24], making accurate color correction challenging and potentially leading to misalignment in the synthesized views of real scenes.

## 5 Experiments

## 5.1 Experimental Setting

5.1.1 Implementation Details. (1) Training Assumption: To reconstruct dynamic scenes using Gaussian Splatting [27] from highspeed, multi-view event cameras, we assume that our method leverages accurate camera intrinsics and high-quality, frequencyconsistent extrinsics to enable precise interpolation at arbitrary timestamps. Specifically, we apply linear interpolation for camera positions and spherical linear interpolation (SLERP) for camera rotations For the synthetic event dataset, we adopt the original contrast thresholds $C ^ { + 1 }$ and $C ^ { - 1 }$ from the v2e simulation settings [24]. In the real-world autonomous driving dataset, we initialize the contrast thresholds using expected values of the event camera settings. This prior assumption provides a stable starting point, leading to more consistent training and improved 3D reconstruction performance.

(2) Training Details: We implemented E-4DGS based on the official code of Deformable3DGS [81], Gaussianflow [36], E-NeRF [29] and Event3DGS [19, 74] with Pytorch and conduct all experiments on a single NVIDIA RTX 4090 GPU. During training, we render at a resolution of 346 Ã 260 for the synthetic dataset and retain the original resolution $6 4 0 \times 4 8 0$ for real-scene data. Events are accumulated into frames using our adaptive slicing strategy (Section 4.2), where the number of events per temporal window is randomly sampled from a predefined range to introduce temporal diversity and enhance robustness. Specifically, we set $\left[ N _ { \operatorname* { m i n } } , N _ { \operatorname* { m a x } } \right] =$ $[ 5 \times 1 0 ^ { 3 } , 1 0 ^ { 4 } ]$ for object-level scenes and $[ 1 0 ^ { 5 } , 1 0 ^ { 6 } ]$ for indoor or large-scale scenes. Additionally, Gaussian noise $( \sigma _ { \mathrm { n o i s e } } = 0 . 0 2 )$ is injected into event-void pixels during accumulation to improve optimization in textureless regions. Each scene is trained for 50,000 iterations using the Adam optimizer. The overall loss consists of an event-based supervision loss, a total variation regularization term and a RGB reconstruction loss (opt. ), weighted by $\lambda _ { \mathrm { R e c o n } } = 1 . 0 ,$ , and $\lambda _ { \mathrm { T V } } = 0 . 0 0 5 , \lambda _ { \mathrm { R G B } } = 1 . 0$ , respectively. The stabilization constant ?? in $\mathcal { L } _ { \mathrm { R e c o n } }$ is set to 0.001. The learnable event contrast threshold ??Ë is initialized to 0.15 for synthetic scenes and 0.2 for real-scene scenes, and is jointly optimized during training. To prevent interference with dynamic scene modeling, opacity reset is disabled like in [10] and color correction is applied only at inference time in all scenes.

5.1.2 Evaluation Metrics. For synthetic and real-scene datasets, we employ the Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) , and VGG-based Learned Perceptual Image Patch Similarity (LPIPS) to evaluate the similarity between rendered novel views and ground-truth novel views.

5.1.3 Baselines. At the time of writing, the event-based dynamic reconstruction methods Dynamic EventNeRF [54] and EBGS [76] have not been publicly released. Although EvDNeRF [1] is opensourced, it focuses solely on modeling geometric edges rather than performing holistic scene reconstruction. Consequently, we compare our proposed method against RGB-based baselines that do not utilize event data and are trained either on blurry RGB recordings or on RGB videos reconstructed from events using E2VID [52]. We choose Deformable3DGS [81] and Deblur4DGS [72] as the RGBbased baseline with blurry RGB inputs or event-integral inputs.

## 5.2 Experimental Evaluation

5.2.1 Synthetic dataset. To generate synthetic data, we render 8 dynamic scenes in Blender [6] at 3000 FPS from six moving viewpoints uniformly distributed around the object at the same height. The rendered sequences are then processed by the event simulator v2e [24] to produce corresponding event streams.

(a) Novel View Synthesis: As demonstrated in Table 1, our proposed E-4DGS outperforms the baselines E2VID + D3DGS across all synthetic scenes in all metrics. This result is intuitive, as E2VID benefits from being trained on a large dataset but does not account for 3D consistency, whereas our method explicitly incorporates it. Moreover, EvDNeRF only models the edge of a single object and does not capture the appearance of the dynamic scene, leading to inferior performance compared to the two-stage method and our proposed E-4DGS. The qualitative comparison of novel view synthesis in Figure 5 shows that our method produces reconstructed scenes with fewer floaters and more photorealistic rendering results.

Event/RGB Frame  
E2VID+D3DGS  
Oursw/. Event  
D3DGS  
Deblur4DGS  
Oursw/.Event& RGB  
<!-- image-->  
Figure 5: Qualitative results of novel view synthesis. Compared with 4D reconstruction-based methods [72, 81], our approach produces more realistic rendering results with fine-grained details in the synthetic and real scenes.

Table 1: Quantitative comparison of different methods for novel view synthesis from event streams. The best and second-best results are highlighted in bold and underlined, respectively. The average value is computed across 8 synthetic scenes.
<table><tr><td rowspan="2">Method</td><td colspan="3">Lego</td><td colspan="3">Rubik&#x27;s Cube</td><td colspan="3">Capsule</td><td colspan="3">Restroom</td><td colspan="3">Average</td></tr><tr><td>âPSNR</td><td>âSSIM</td><td>âLPIPS</td><td>âPSNR</td><td>âSSIM</td><td>âLPIPS</td><td>âPSNR</td><td>âSSIM</td><td>âLPIPS</td><td>âPSNR</td><td>âSSIM</td><td>âLPIPS</td><td>âPSNR</td><td>âSSIM</td><td>âLPIPS</td></tr><tr><td>D3DGSw/o blur</td><td>26.47</td><td>0.910</td><td>0.098</td><td>20.30</td><td>0.868</td><td>0.207</td><td>31.23</td><td>0.956</td><td>0.077</td><td>28.05</td><td>0.935</td><td>0.074</td><td>23.81</td><td>0.861</td><td>0.173</td></tr><tr><td>D3DGSw/ blur</td><td>23.62</td><td>0.821</td><td>0.250</td><td>18.12</td><td>0.804</td><td>0.351</td><td>27.51</td><td>0.905</td><td>0.181</td><td>26.46</td><td>0.908</td><td>0.160</td><td>21.73</td><td>0.797</td><td>0.296</td></tr><tr><td>E2VID + D3DGS</td><td>20.57</td><td>0.765</td><td>0.347</td><td>16.16</td><td>0.752</td><td>0.404</td><td>26.06</td><td>0.851</td><td>0.268</td><td>24.87</td><td>0.856</td><td>0.247</td><td>19.88</td><td>0.728</td><td>0.397</td></tr><tr><td>Deblur4DGS</td><td>23.17</td><td>0.813</td><td>0.265</td><td>17.68</td><td>0.786</td><td>0.375</td><td>28.06</td><td>0.908</td><td>0.176</td><td>26.35</td><td>0.900</td><td>0.162</td><td>21.66</td><td>0.797</td><td>0.291</td></tr><tr><td>E-4DGSevent-only</td><td>26.85</td><td>0.912</td><td>0.084</td><td>20.97</td><td>0.882</td><td>0.185</td><td>31.85</td><td>0.959</td><td>0.071</td><td>28.83</td><td>0.942</td><td>0.069</td><td>25.38</td><td>0.896</td><td>0.134</td></tr><tr><td>E-4DGSevent&amp; RGB</td><td>27.23</td><td>0.925</td><td>0.078</td><td>21.23</td><td>0.895</td><td>0.172</td><td>32.41</td><td>0.963</td><td>0.068</td><td>29.02</td><td>0.949</td><td>0.067</td><td>25.62</td><td>0.903</td><td>0.129</td></tr></table>

(b) Motion Blur Decoupling: Using event sequences for deblurring blurry RGB frames is a common task. In our experiments, we simulate blurry images using Blender [6] by integrating images over the exposure time using LERP and SLERP, which yields realistic, motion-dependent blur. Table 1 show that our method perform better than all 4D reconstruction baselines. The results of our proposed method are better than the two-stage method which is combining E2VID with frame-based D3DGS. Furthermore, our method outperforms the frame-based 4D deblurring baseline [72]1, demonstrating that inherent blur-resistant characteristics of events offer greater advantages than relying solely on blur formation.

Table 2: Ablation study of each component.
<table><tr><td colspan="4">Method Components</td><td colspan="4">Synthetic Datasets</td></tr><tr><td> $\mathcal { L } _ { r e c o n }$ </td><td> $\mathcal { L } _ { t v }$ </td><td>ESS</td><td>AES</td><td>IP</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td><td>25.38</td><td>0.896</td><td>0.134</td></tr><tr><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td><td>23.68</td><td>0.858</td><td>0.178</td></tr><tr><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td><td>23.89</td><td>0.865</td><td>0.171</td></tr><tr><td>â</td><td>â</td><td>â</td><td>-</td><td></td><td>24.71</td><td>0.876</td><td>0.153</td></tr><tr><td>â</td><td>â</td><td>â</td><td></td><td></td><td>23.97</td><td>0.863</td><td>0.172</td></tr><tr><td>â</td><td>â</td><td>-</td><td>-</td><td>â</td><td>25.13</td><td>0.881</td><td>0.142</td></tr></table>

(c) Dynamic Reconstruction with Event and Frame Fusion: We combine event sequences and blurry frames by an event-RGB weighted combination, caculated as follows:

$$
\mathcal { L } _ { F u s i o n } = \mathcal { L } _ { E v e n t } + \lambda _ { R G B } \mathcal { L } _ { R G B }\tag{15}
$$

Here, L?????? is the original photo-realistic rendering loss of D3DGS, including the L1 and D-SSIM loss terms [81]. Due to the discrete nature of events, while event sequences capture sharp edges, they are noisy in low-light or uniform areas, causing fog-like artifacts in dynamic scenes. The color in shaded areas may also be slightly off and requires correction [37, 53, 54], as it is inferred from derivative-like data rather than directly measured. Incorporating RGB frames addresses these issues by preserving low-frequency texture details while retaining the sharp high-frequency features from event sequences. As shown in Figure 5, our method reconstructs a sharp dynamic scene with accurate colors, achieving the best performance as reported in Table 1. Therefore, color event data from cameras like DVS346C is unnecessary, as the predicted color values of event rays can be directly mapped to grayscale.

w/o. AES  
w/. AES  
Ground-Truth  
<!-- image-->  
Figure 6: The Performance of the adaptive event supervision on the real-scene of the DSEC dataset.

5.2.2 Real-scene dataset. The real-world experiments are conducted on the interlaken_00_c, interlaken_00_d, and zurich_city_00_a sequences from the autonomous driving dataset DSEC [15] captured from a modern, high-resolution event sensorâProphesee Gen3.1. However, the real-world experiments primarily serve as a qualitative benchmark, as the existing datasets [23, 28, 48], are not specifically designed for the task of NVS and lacks multi-view event streams with settings comparable to our synthetic dataset. This limitation is partly due to the fact that the target novel-view images are captured using a single standard RGB camera, which suffers from saturation effects because of its relatively limited dynamic range. Moreover, these images are not raw sensor outputs but have undergone in-camera image processing, often lossy in nature. In addition, the spectral response curve of the event camera is not publicly available, making color correction potentially inaccurate when aligning synthesized views with real images. Consequently, the dataset does not support accurate quantitative NVS evaluation.

Table 3: Ablation study on the robustness of deblurring. The best and second results are bold and underlined, respectively.
<table><tr><td>Blur Degree Metrics</td><td>Mild blur PSNRâ/SSIMâ/LPIPSâ PSNRâ/SSIMâ/LPIPSâ PSNRâ/SSIMâ/LPIPSâ</td><td>Medium blur</td><td>Strong blur</td></tr><tr><td>D3DGS</td><td> $1 9 . 0 5 \mathrm { ~ / ~ } 0 . 6 2 \mathrm { ~ / ~ } 0 . 4 1$ </td><td> $1 8 . 8 9 \mathrm { ~ / ~ } 0 . 6 1 \mathrm { ~ / ~ } 0 . 4 1$ </td><td> $1 6 . 9 8 / 0 . 5 2 / 0 . 5 7$ </td></tr><tr><td>E2VID+D3DGS</td><td> $1 7 . 7 9 \mathrm { ~ / ~ } 0 . 5 5 \mathrm { ~ / ~ } 0 . 4 9$ </td><td> $1 7 . 6 9 / 0 . 5 4 / 0 . 4 9$ </td><td> $1 7 . 9 3 / 0 . 5 5 / 0 . 4 9$ </td></tr><tr><td>Deblur4DGS</td><td> $1 9 . 2 3 / 0 . 6 4 / 0 . 3 8$ </td><td> $1 8 . 9 2 \mathrm { ~ / ~ } 0 . 6 1 \mathrm { ~ / ~ } 0 . 4 2$ </td><td> $1 6 . 6 6 / 0 . 5 0 / 0 . 5 9$ </td></tr><tr><td> $\mathrm { E } { - } 4 \mathrm { D G S } _ { e v e n t - o n l y }$ </td><td> $\mathbf { 2 4 . 8 1 } / \mathbf { 0 . 8 7 } / \mathbf { 0 . 1 7 }$ </td><td> ${ \bf 2 4 . 3 2 / 0 . 8 6 / 0 . 1 9 }$ </td><td> $\mathbf { 2 1 . 5 9 / 0 . 7 6 / 0 . 2 8 }$ </td></tr><tr><td> $\mathrm { E } { - } 4 \mathrm { D } \mathrm { G } \mathrm { S } _ { e v e n t \& } \quad \mathrm { G } B$ </td><td> $\mathbf { 2 4 . 9 5 / 0 . 8 8 / 0 . 1 7 }$ </td><td> $\mathbf { 2 4 . 7 8 \ : / 0 . 8 7 / 0 . 1 7 }$ </td><td> $\mathbf { 2 2 . 0 6 } \mathrm { ~ / ~ } \mathbf { 0 . 8 0 } \mathrm { ~ / ~ } \mathbf { 0 . 2 6 }$ </td></tr></table>

For real-scene evaluation of dynamic reconstruction with event and frame fusion, the E2VID + D3DGS baseline recovers more visual details overal. However, the proposed E-4DGS produces fewer artifacts, particularly around foreground objects. While Deblur4DGS improves on D3DGS, both struggle to recover fine details, such as distant lettering (Figure 5). E-4DGS outperforms E2VID + D3DGS in high-frequency details and geometry. None achieve fully photorealistic quality due to single-view supervision, though such quality is often unnecessary for many robotics applications, emphasizing the need for more multiview event data.

## 5.3 Ablation Evaluation

To assess the contribution of each component, we train various model variants on both synthetic and real sequences, focusing on evaluating effects of event-adaptive slicing splatting (ESS), adaptive event supervision (AES), and intensity importance pruning (IIP).

Effect of Different Components. For evaluation without ESS, we use a fixed event sampling number for accumulation instead of the specific strategy, and a fixed event threshold value for evaluation without adaptive event supervision. As shown in Table 2, incorporating ESS and IIP leads to a clear performance improvement. ESS addresses the non-uniform spatial-temporal distribution of event data, while IIP enhances multi-view consistency, together boosting reconstruction performance. Although the adaptive event supervision component slightly reduces performance on the synthetic dataset, it significantly improves texture fidelity and temporal consistency in real-world scenarios, as demonstrated in Figure 6.

Effect of Motion Blur at Different Levels To assess the robustness of deblurring, we simulate blurry images with varying degrees of motion blurâmild, medium, and strongâby integrating RGB frames over the exposure time in Blender [6], creating realistic, motion-dependent blur patterns. With the synthetic scene Garage, we find that E-4DGS consistently outperforms all baselines across all blur levels, achieving the highest performance in Table 3. Ours also reconstructs sharper scene details and more accurate object boundaries, particularly under strong blur, demonstrating its superior ability to preserve spatial structure and temporal consistency.

## 6 Conclusion

In this paper, we propose E-4DGS, a novel paradigm for real-time dynamic view synthesis based on dynamic 3DGS using multi-view event sequences. We design a synthetic multi-view camera setup with six moving event cameras surrounding an object in a 360- degree configuration and provide a benchmark multi-view event stream dataset that captures challenging motion scenarios. Our approach outperforms both event-only and event-RGB fusion baselines, paving the way for the exploration of multi-view event-based reconstruction as a novel approach for rapid scene capture. Future work will focus on addressing the challenges of handling largerscale dynamic scenes and improving computational efficiency for real-world applications such as autonomous driving and immersive virtual environments.

## Acknowledgement

This work was supported in part by Natural Science Foundation of China (No.62332002 and No.62202014), and Shenzhen KQTD (No.20240729102051063).

## References

[1] Anish Bhattacharya, Ratnesh Madaan, Fernando Cladera, Sai Vemprala, Rogerio Bonatti, Kostas Daniilidis, Ashish Kapoor, Vijay Kumar, Nikolai Matni, and Jayesh K Gupta. 2024. Evdnerf: Reconstructing event data with dynamic neural radiance fields. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 5846â5855.

[2] Marco Cannici and Davide Scaramuzza. 2024. Mitigating Motion Blur in Neural Radiance Fields with Events and Frames. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[3] Ang Cao and Justin Johnson. 2023. HexPlane: A Fast Representation for Dynamic Scenes. In Computer Vision and Pattern Recognition (CVPR). https://caoang327. github.io/HexPlane/

[4] Kang Chen, Jiyuan Zhang, Zecheng Hao, Yajing Zheng, Tiejun Huang, and Zhaofei Yu. 2024. USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting. arXiv preprint arXiv:2411.10504 (2024).

[5] Runnan Chen, Youquan Liu, Lingdong Kong, Xinge Zhu, Yuexin Ma, Yikang Li, Yuenan Hou, Yu Qiao, and Wenping Wang. 2023. Clip2scene: Towards labelefficient 3d scene understanding by clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7020â7030.

[6] Blender Online Community. 2018. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam. http://www. blender.org

[7] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. 2024. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems 36 (2024).

[8] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 2024. 4d-rotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes. In ACM SIGGRAPH 2024 Conference Papers. 1â11.

[9] Burak Ercan, Onur Eker, Canberk Saglam, Aykut Erdem, and Erkut Erdem. 2024. Hypere2vid: Improving event-based video reconstruction via hypernetworks. IEEE Transactions on Image Processing (2024).

[10] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, Zhangyang Wang, and Yue Wang. 2024. InstantSplat: Unbounded Sparse-view Pose-free Gaussian Splatting in 40 Seconds. arXiv:2403.20309 [cs.CV]

[11] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. 2023. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. arXiv preprint arXiv:2311.17245 (2023).

[12] Chaoran Feng, Wangbo Yu, Xinhua Cheng, Zhenyu Tang, Junwu Zhang, Li Yuan, and Yonghong Tian. 2025. AE-NeRF: Augmenting Event-Based Neural Radiance Fields for Non-ideal Conditions and Larger Scene. arXiv preprint arXiv:2501.02807 (2025).

[13] Guillermo Gallego, Tobi Delbruck, Garrick Orchard, Chiara Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger, Andrew J Davison, Jorg Conradt, Kostas Daniilidis, et al. 2020. Event-based vision: A survey. IEEE Trans. Pattern Analysis and Machine Intelligence (PAMI) (2020).

[14] Guillermo Gallego, Henri Rebecq, and Davide Scaramuzza. 2018. A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition. 3867â3876.

[15] Mathias Gehrig, Willem Aarents, Daniel Gehrig, and Davide Scaramuzza. 2021. DSEC: A Stereo Event Camera Dataset for Driving Scenarios. IEEE Robotics and Automation Letters (2021). doi:10.1109/LRA.2021.3068942

[16] Gaurvi Goyal, Franco Di Pietro, Nicolo Carissimi, Arren Glover, and Chiara Bartolozzi. 2023. Moveenet: online high-frequency human pose estimation with an event camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4024â4033.

[17] Shuang Guo and Guillermo Gallego. 2024. CMax-SLAM: Event-based rotationalmotion bundle adjustment and SLAM system using contrast maximization. IEEE Transactions on Robotics (2024).

[18] Yijia Guo, Liwen Hu, Yuanxi Bai, Jiawei Yao, Lei Ma, and Tiejun Huang. 2024. Spikegs: Reconstruct 3d scene via fast-moving bio-inspired sensors. arXiv preprint arXiv:2407.03771 (2024).

[19] Haiqian Han, Jianing Li, Henglu Wei, and Xiangyang Ji. 2024. Event-3DGS: Event-based 3D Reconstruction Using 3D Gaussian Splatting. Advances in Neural Information Processing Systems 37 (2024), 128139â128159.

[20] Haiqian Han, Jiacheng Lyu, Jianing Li, Henglu Wei, Cheng Li, Yajing Wei, Shu Chen, and Xiangyang Ji. 2024. Physical-Based Event Camera Simulator. In European Conference on Computer Vision. Springer, 19â35.

[21] Bing He, Yunuo Chen, Guo Lu, Qi Wang, Qunshan Gu, Rong Xie, Li Song, and Wenjun Zhang. 2024. S4D: Streaming 4D Real-World Reconstruction with Gaussians and 3D Control Points. arXiv:2408.13036 [cs.CV]

[22] Weihua He, Kaichao You, Zhendong Qiao, Xu Jia, Ziyang Zhang, Wenhui Wang, Huchuan Lu, Yaoyuan Wang, and Jianxing Liao. 2022. Timereplayer: Unlocking the potential of event cameras for video interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17804â17813.

[23] Javier Hidalgo-CarriÃ³, Guillermo Gallego, and Davide Scaramuzza. 2022. Eventaided direct sparse odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5781â5790.

[24] Yuhuang Hu, Shih-Chii Liu, and Tobi Delbruck. 2021. v2e: From video frames to realistic DVS events. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1312â1321.

[25] Jian Huang, Chengrui Dong, and Peidong Liu. 2024. IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera. arXiv preprint arXiv:2410.08107 (2024).

[26] Inwoo Hwang, Junho Kim, and Young Min Kim. 2023. Ev-nerf: Event based neural radiance field. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 837â847.

[27] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics (TOG) (2023). https://repo-sam.inria.fr/fungraph/3dgaussian-splatting/

[28] Simon Klenk, Jason Chui, Nikolaus Demmel, and Daniel Cremers. 2021. TUM-VIE: The TUM stereo visual-inertial event dataset. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 8601â8608.

[29] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers. 2023. E-nerf: Neural radiance fields from a moving event camera. IEEE Robotics and Automation Letters 8, 3 (2023), 1587â1594.

[30] Seungjun Lee and Gim Hee Lee. 2025. DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting. arXiv:2503.24210 [cs.CV] https://arxiv.org/abs/2503.24210

[31] Hao Li, Curise Jia, Peng Jin, Zesen Cheng, Kehan Li, Jialu Sui, Chang Liu, and Li Yuan. 2023. Freestyleret: Retrieving images from style-diversified queries. arXiv preprint arXiv:2312.02428 (2023).

[32] Hao Li, Da Long, Li Yuan, Yu Wang, Yonghong Tian, Xinchang Wang, and Fanyang Mo. 2025. Decoupled peak property learning for efficient and interpretable electronic circular dichroism spectrum prediction. Nature Computational Science (2025), 1â11.

[33] Yuchen Li\*, Chaoran Feng\*, Zhenyu Tang, Kaiyuan Deng, Wangbo Yu, Yonghong Tian, and Li Yuan. 2025. GS2E: Gaussian Splatting is an Effective Data Generator for Event Stream Generation. arXiv preprint arXiv:2505.15287 (2025).

[34] Jinwei Lin. 2024. Dynamic NeRF: A Review. arXiv preprint arXiv:2405.08609 (2024).

[35] Songnan Lin, Ye Ma, Zhenhua Guo, and Bihan Wen. 2022. Dvs-voltmeter: Stochastic process-based event simulator for dynamic vision sensors. In European Conference on Computer Vision. Springer, 578â593.

[36] Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. 2024. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21136â21145.

[37] Weng Fei Low and Gim Hee Lee. 2023. Robust e-nerf: Nerf from sparse & noisy events under non-uniform motion. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

[38] Jiahao Lu, Jiacheng Deng, Ruijie Zhu, Yanzhe Liang, Wenfei Yang, Xu Zhou, and Tianzhu Zhang. 2025. Dn-4dgs: Denoised deformable network with temporalspatial aggregation for dynamic scene rendering. Advances in Neural Information Processing Systems 37 (2025), 84114â84138.

[39] Qi Ma, Danda Pani Paudel, Ajad Chhatkuli, and Luc Van Gool. 2023. Deformable neural radiance fields using rgb and event cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3590â3600.

[40] Branislav Micusik and Tomavs Pajdla. [n. d.]. Structure from motion with wide circular field of view cameras. ([n. d.]).

[41] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. 2019. Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines. arXiv:1905.00889 [cs.CV] https://arxiv.org/abs/1905.00889

[42] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In European Conference on Computer Vision (ECCV). https://www.matthewtancik.com/nerf

[43] Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley, Miaomiao Liu, and Yuchao Dai. 2019. Bringing a blurry frame alive at high frame-rate with an event camera. In Computer Vision and Pattern Recognition (CVPR).

[44] Yatian Pang, Peng Jin, Shuo Yang, Bin Lin, Bin Zhu, Zhenyu Tang, Liuhan Chen, Francis EH Tay, Ser-Nam Lim, Harry Yang, et al. 2024. Next patch prediction for autoregressive visual generation. arXiv preprint arXiv:2412.15321 (2024).

[45] Yatian Pang, Bin Zhu, Bin Lin, Mingzhe Zheng, Francis EH Tay, Ser-Nam Lim, Harry Yang, and Li Yuan. 2024. DreamDance: Animating Human Images by

Enriching 3D Geometry Cues from 2D Poses. arXiv preprint arXiv:2412.00397 (2024).

[46] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. 2021. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5865â5874.

[47] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. 2021. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228 (2021).

[48] Shihan Peng, Hanyu Zhou, Hao Dong, Zhiwei Shi, Haoyue Liu, Yuxing Duan, Yi Chang, and Luxin Yan. 2024. CoSEC: A coaxial stereo event camera dataset for autonomous driving. arXiv preprint arXiv:2408.08500 (2024).

[49] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2021. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10318â10327.

[50] Yunshan Qi, Lin Zhu, Yu Zhang, and Jia Li. 2023. E2NeRF: Event Enhanced Neural Radiance Fields from Blurry Images. In International Conference on Computer Vision (ICCV).

[51] Maxime Raafat and contributors. 2024. BlenderNeRF: Easy NeRF synthetic dataset creation within Blender. https://github.com/maximeraafat/BlenderNeRF. Accessed: 2025-04-07.

[52] Henri Rebecq, RenÃ© Ranftl, Vladlen Koltun, and Davide Scaramuzza. 2019. High Speed and High Dynamic Range Video with an Event Camera. IEEE Trans. Pattern Anal. Mach. Intell. (T-PAMI) (2019).

[53] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. 2023. EventNeRF: Neural Radiance Fields from a Single Colour Event Camera. In Computer Vision and Pattern Recognition (CVPR).

[54] Viktor Rudnev, Gereon Fox, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. 2024. Dynamic EventNeRF: Reconstructing General Dynamic Scenes from Multi-view Event Cameras. arXiv preprint arXiv:2412.06770 (2024).

[55] Arman Savran, Raffaele Tavarone, Bertrand Higy, Leonardo Badino, and Chiara Bartolozzi. 2018. Energy and computation efficient audio-visual voice activity detection driven by event-cameras. In 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018). IEEE, 333â340.

[56] Johannes L Schonberger and Jan-Michael Frahm. 2016. Structure-from-motion Revisited. In Computer Vision and Pattern Recognition (CVPR).

[57] Jiwei Shan, Zeyu Cai, Cheng-Tai Hsieh, Shing Shin Cheng, and Hesheng Wang. 2025. Deformable Gaussian Splatting for Efficient and High-Fidelity Reconstruction of Surgical Scenes. arXiv preprint arXiv:2501.01101 (2025).

[58] Zihang Shao, Xuanye Fang, Yaxin Li, Chaoran Feng, Jiangrong Shen, and Qi Xu. 2023. EICIL: joint excitatory inhibitory cycle iteration learning for deep spiking neural networks. Advances in Neural Information Processing Systems 36 (2023), 32117â32128.

[59] Noah Snavely, Steven M Seitz, and Richard Szeliski. 2006. Photo tourism: exploring photo collections in 3D. In ACM siggraph 2006 papers. 835â846.

[60] Ganchao Tan, Yang Wang, Han Han, Yang Cao, Feng Wu, and Zheng-Jun Zha. 2022. Multi-grained spatio-temporal features perceived network for event-based lip-reading. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20094â20103.

[61] Wei Zhi Tang, Daniel Rebain, Kostantinos G Derpanis, and Kwang Moo Yi. 2024. LSE-NeRF: Learning Sensor Modeling Errors for Deblured Neural Radiance Fields with RGB-Event Stereo. arXiv preprint arXiv:2409.06104 (2024).

[62] Zhenyu Tang, Chaoran Feng, Xinhua Cheng, Wangbo Yu, Junwu Zhang, Yuan Liu, Xiaoxiao Long, Wenping Wang, and Li Yuan. 2025. NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting for Compact 3D Representations. arXiv preprint arXiv:2503.23162 (2025).

[63] Zhenyu Tang, Junwu Zhang, Xinhua Cheng, Wangbo Yu, Chaoran Feng, Yatian Pang, Bin Lin, and Li Yuan. 2024. Cycle3D: High-quality and Consistent Image-to-3D Generation via Generation-Reconstruction Cycle. arXiv preprint arXiv:2407.19548 (2024).

[64] Stepan Tulyakov, Daniel Gehrig, Stamatios Georgoulis, Julius Erbach, Mathias Gehrig, Yuanyou Li, and Davide Scaramuzza. 2021. Time lens: Event-based video frame interpolation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 16155â16164.

[65] Diwen Wan, Yuxiang Wang, Ruijie Lu, and Gang Zeng. 2024. Template-free Articulated Gaussian Splatting for Real-time Reposable Dynamic View Synthesis. arXiv preprint arXiv:2412.05570 (2024).

[66] Haoyang Wang, Ruishan Guo, Pengtao Ma, Ciyu Ruan, Xinyu Luo, Wenhua Ding, Tianyang Zhong, Jingao Xu, Yunhao Liu, and Xinlei Chen. 2025. Towards Mobile Sensing with Event Cameras on High-mobility Resource-constrained Devices: A Survey. arXiv preprint arXiv:2503.22943 (2025).

[67] Jiaxu Wang, Junhao He, Ziyi Zhang, Mingyuan Sun, Jingkai Sun, and Renjing Xu. 2024. EvGGS: A Collaborative Learning Framework for Event-based Generalizable Gaussian Splatting. arXiv preprint arXiv:2405.14959 (2024).

[68] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. 2024. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764 (2024).

[69] Jiahao Wu, Rui Peng, Zhiyan Wang, Lu Xiao, Luyang Tang, Jinbo Yan, Kaiqiang Xiong, and Ronggang Wang. 2025. Swift4D: Adaptive divide-and-conquer Gaussian Splatting for compact and efficient reconstruction of dynamic scene. arXiv preprint arXiv:2503.12307 (2025).

[70] Jingqian Wu, Shuo Zhu, Chutian Wang, and Edmund Y Lam. 2024. Ev-GS: Eventbased gaussian splatting for efficient and accurate radiance field rendering. In 2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 1â6.

[71] Jingqian Wu, Shuo Zhu, Chutian Wang, Boxin Shi, and Edmund Y Lam. 2024. SweepEvGS: Event-Based 3D Gaussian Splatting for Macro and Micro Radiance Field Rendering from a Single Sweep. arXiv preprint arXiv:2412.11579 (2024).

[72] Renlong Wu, Zhilu Zhang, Mingyang Chen, Xiaopeng Fan, Zifei Yan, and Wangmeng Zuo. 2024. Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video. arXiv preprint arXiv:2412.06424 (2024).

[73] Ziyi Wu, Mathias Gehrig, Qing Lyu, Xudong Liu, and Igor Gilitschenski. 2024. Leod: Label-efficient object detection for event cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16933â16943.

[74] Tianyi Xiong, Jiayi Wu, Botao He, Cornelia Fermuller, Yiannis Aloimonos, Heng Huang, and Christopher A Metzler. 2024. Event3DGS: Event-based 3D Gaussian Splatting for Fast Egomotion. arXiv preprint arXiv:2406.02972 (2024).

[75] Chuanzhi Xu, Haoxian Zhou, Haodong Chen, Vera Chung, and Qiang Qu. 2025. A Survey on Event-driven 3D Reconstruction: Development under Different Categories. arXiv preprint arXiv:2503.19753 (2025).

[76] Wenhao Xu, Wenming Weng, Yueyi Zhang, Ruikang Xu, and Zhiwei Xiong. 2024. Event-boosted Deformable 3D Gaussians for Fast Dynamic Scene Reconstruction. arXiv preprint arXiv:2411.16180 (2024).

[77] Le Xue, Mingfei Gao, Chen Xing, Roberto MartÃ­n-MartÃ­n, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. 2023. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1179â1189.

[78] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. 2024. Gs-slam: Dense visual slam with 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

[79] Jinbo Yan, Rui Peng, Luyang Tang, and Ronggang Wang. 2024. 4D Gaussian Splatting with Scale-aware Residual Field and Adaptive Optimization for Realtime rendering of temporally complex dynamic scenes. In Proceedings of the 32nd ACM International Conference on Multimedia. 7871â7880.

[80] Zhiwen Yan, Chen Li, and Gim Hee Lee. 2023. Nerf-ds: Neural radiance fields for dynamic specular objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8285â8295.

[81] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. 2024. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20331â20341.

[82] Zeyu Yang, Zijie Pan, Xiatian Zhu, Li Zhang, Yu-Gang Jiang, and Philip HS Torr. 2024. 4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives. arXiv preprint arXiv:2412.20720 (2024).

[83] Xiaoting Yin, Hao Shi, Yuhan Bao, Zhenshan Bing, Yiyi Liao, Kailun Yang, and Kaiwei Wang. 2024. E-3DGS: Gaussian Splatting with Exposure and Motion Events. arXiv preprint arXiv:2410.16995 (2024).

[84] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. 2025. TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models. arXiv preprint arXiv:2503.05638 (2025).

[85] Wangbo Yu, Chaoran Feng, Jiye Tang, Xu Jia, Li Yuan, and Yonghong Tian. 2024. EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images. arXiv preprint arXiv:2405.20224 (2024).

[86] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. 2024. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048 (2024).

[87] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19447â19456.

[88] Shenghai Yuan, Jinfa Huang, Yujun Shi, Yongqi Xu, Ruijie Zhu, Bin Lin, Xinhua Cheng, Li Yuan, and Jiebo Luo. 2024. MagicTime: Time-lapse Video Generation Models as Metamorphic Simulators. arXiv preprint arXiv:2404.05014 (2024).

[89] Sohaib Zahid, Viktor Rudnev, Eddy Ilg, and Vladislav Golyanik. 2025. E-3DGS: Event-based Novel View Rendering of Large-scale Scenes Using 3D Gaussian Splatting. 3DV (2025).

[90] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. 2024. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825 (2024).

[91] Junwu Zhang, Zhenyu Tang, Yatian Pang, Xinhua Cheng, Peng Jin, Yida Wei, Wangbo Yu, Munan Ning, and Li Yuan. 2023. Repaint123: Fast and high-quality one image to 3d generation with progressive controllable 2d repainting. arXiv preprint arXiv:2312.13271 (2023).

[92] Zixin Zhang, Kanghao Chen, and Lin Wang. 2024. Elite-EvGS: Learning Eventbased 3D Gaussian Splatting by Distilling Event-to-Video Priors. arXiv preprint arXiv:2409.13392 (2024).

[93] Zixin Zhang, Kanghao Chen, and Lin Wang. 2024. Elite-evgs: Learning eventbased 3d gaussian splatting by distilling event-to-video priors. arXiv preprint arXiv:2409.13392 (2024).

[94] Chunhui Zhao, Yakun Li, and Yang Lyu. 2023. Event-based real-time moving object detection based on imu ego-motion compensation. In 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 690â696.

[95] Shihao Zou, Chuan Guo, Xinxin Zuo, Sen Wang, Pengyu Wang, Xiaoqin Hu, Shoushun Chen, Minglun Gong, and Li Cheng. 2021. Eventhpe: Event-based 3d human pose and shape estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10996â11005.

Detailed descriptions of dataset construction and training configurations are provided in Section A of the appendix. Section B presents the implementation of our proposed initialization strategy and compares it with existing methods. Further experimental results and ablation studies are reported in Section C.

## A Dataset Preparations

## A.1 Synthetic datasets

We manually create eight synthetic scenes with six viewpoints arranged in a 360-degree configuration around the object or scene. Each scene is designed as a center-focus setup, with an object placed at the center. For these scenes, we render six dynamic scenarios at a resolution of 346 Ã 260 in Blender [6] at 3000 FPS with the BlenderNeRF addon [51]. Six moving viewpoints are uniformly distributed around the object in a spherical spiral motion at a constant height. Event streams are generated using the v2e framework [24]. Additionally, leveraging the camera trajectory data, we simulate blurry images by integrating RGB frames over the exposure time, with varying degrees of motion blurâmild, medium, and strong.

For training and evaluation, we use six viewpoints for training and set the llffhold value to 8 for testing. For event-only dynamic reconstruction, RGB frames are converted to grayscale for evaluation, with event streams used exclusively as input. In the event-RGB fusion dynamic reconstruction, full-resolution color images are used in conjunction with event slices as input modalities.

Data Composition The proposed synthetic dataset consists of five dynamic objects, three dynamic indoor scenes, as follows:

â¢ Dynamic objects. We design five object models in Blender, including Lego, Rubikâs Cube, MC Toy, Hinge, and Cubes. The dynamic Lego model is derived from the static Lego in the NeRF dataset [42], to which we add animation.

â¢ Dynamic indoor scenes. We design three indoor models with dynamic objects in Blender, including Capsule, Restroom and Garage.

All models are licensed under CC-BY 4.0 and will be open-source.

Data Limitations. The synthetic data in this work is generated using the v2e framework [24], which simulates events based on images. However, this approach is inherently limited in handling extreme lighting conditions, such as overexposure or very low light. In these scenarios, the images themselves lack crucial information due to the nature of the lighting, which restricts the ability to accurately simulate event data for such conditions.The left is the pointcloud of Gaussian initilization and the right is the novel view of the Restroom scene.

## A.2 Real-scene datasets

We adopt the DSEC dataset [15], a large-scale real-world dataset designed for driving scenarios, to evaluate our method under realistic and dynamic conditions. The dataset was captured using a synchronized sensor rig mounted on a vehicle, consisting of a Prophesee Gen3.1 event camera, a global shutter RGB camera, and a Velodyne LiDAR. The event camera records asynchronous brightness changes at a spatial resolution of 640Ã480 and provides high temporal resolution (down to microseconds), enabling the capture of fast motion and high dynamic range scenes. The RGB camera outputs global shutter images at 1024Ã768 resolution with fixed frame intervals. Calibration files are provided to align coordinate systems of the sensors.

Each sequence in DSEC contains temporally synchronized event streams, RGB frames, LiDAR point clouds, camera intrinsics/extrinsics, and time-stamped poses obtained via visual-inertial odometry. For our experiments, we select three representative sequences: interlaken_00_c, interlaken_00_d, and zurich_city_00_a, which cover diverse urban and suburban environments.

Since the dataset is not originally designed for Novel View Synthesis (NVS), we perform several processing steps to construct suitable input-output pairs:

â¢ Image-Event Alignment: For each RGB frame, we extract a corresponding event stream by accumulating events within a fixed temporal window around the image timestamp. Events outside the desired range are discarded to reduce background noise.

â¢ View Subsampling: We uniformly sample camera viewpoints along the driving trajectory. Following the standard LLFF [41] protocol, we use every 8 consecutive views for training and hold out the next view for evaluation.

â¢ Modality Handling: For event-only models, RGB frames are converted to grayscale as evaluation and only event streams are used as input. For event-RGB fusion settings, the full-resolution color images are used jointly with the event slices as input modalities.

â¢ Frame Curation: Frames suffering from severe motion blur or under-/over-exposure are excluded to ensure a clean evaluation set. We also ignore frames with poor localization confidence based on pose metadata.

Although the dataset offers event streams, RGB images, and LiDAR data, its forward-facing setup with narrow baseline viewpoints makes it inherently unsuitable for tasks requiring diverse multi-view observations, such as high-fidelity 3D reconstruction and novel view synthesis. As a result, we use these sequences only for qualitative visualization.

## B More details of Pointcloud Initialization

In this section, we explore the impact of different point cloud initialization methods on the rendering performance of E-4DGS in three proposed indoor scenes. Compared to the random initialization, commonly used in methods such as [25, 70], using the sparse point clouds from Structure-from-Motion (SfM) [40] significantly improves rendering accuracy when only motion events are utilized, with the PSNR metric increasing from 24.21 dB to 24.87 dB.

<!-- image-->

Figure 7: Virtual camera setup in Blender for synthetic dataset generation. The six simulated DAVIS 346C event cameras are positioned to match the layout of our real-world multi-view recording environment.  
<!-- image-->  
Figure 8: Qualitative comparison of different initialization methods and our method achieves a trade-off between efficiency and performace.

To further demonstrate the trade-off between efficiency and performance achieved by our proposed method, we compare the effect of point cloud initialization using event-to-video approaches in Table 4. Using E2VID[52] to convert event data into images and generating point clouds through SfM yields further accuracy improvements. However, this process introduces additional computational costs and time due to reliance on learning-based methods.

Table 4: Ablation Study on Different Initialization Method.
<table><tr><td>Method</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>Time/h</td></tr><tr><td>Random Init.</td><td>21.56</td><td>0.785</td><td>0.233</td><td>0.9</td></tr><tr><td>E2VID+SfM</td><td>24.87</td><td>0.866</td><td>0.170</td><td>2.5</td></tr><tr><td>Ours</td><td> $2 4 . 2 1 _ { ( - 0 . 6 6 ) }$  2</td><td> $0 . 8 5 4 _ { ( - 0 . 0 1 2 ) }$ </td><td>2  $ { ^ { 0 . 1 7 6 } } _ { ( }  { ^ { \mathrm { ~ ~ \tiny ~ { ~ \chi ~ } ~ } } } )$ </td><td> $1 . 1 _ { ( - 1 . 4 ) }$ </td></tr></table>

As shown in Figure 8, we visualize the impact of different initialization methods on event-based 4DGS rendering. When random initialization is used, the 4DGS reconstruction based on motion events suffers from noticeable artifacts and a lack of detail. The E2VID + COLMAP-based SfM method improves scene reconstruction, but at the cost of significantly lower runtime efficiency. In contrast, our method employs a radial initialization after considering a center-focus environment, yielding comparable rendering results to the two-stage initialization approach, despite slightly lower quantitative metrics. This validates the key role of our approach in improving the efficiency of event-driven explicit dynamic reconstruction.

## C Additional experiments

## C.1 Performance of adaptive event threshold

To bridge the gap between dense image rendering and sparse event streams, our E-4DGS framework incorporates a learnable event contrast threshold ??Ë. This parameter governs the sensitivity of event triggering, and is jointly optimized with other model parameters. Rather than relying on a fixed threshold, we allow ??Ë to dynamically evolve to better accommodate diverse temporal changes in intensity. As shown in Fig. 9, the synthetic dataset demonstrates a relatively stable threshold behavior, aligning with its lower noise and controlled motion. In contrast, real-scene data produces more frequent and stronger burst patterns, requiring a more adaptive threshold to handle high-frequency voltage changes effectively. This adaptiveness ensures accurate contrast modeling for event supervision, contributing to the photometric alignment between rendered and observed event data.

Real Scene  
<!-- image-->

Synthetic Scene  
<!-- image-->  
Figure 9: Visualization of the adaptive event threshold ??Ë during training on both synthetic and real-scene datasets. For the synthetic dataset (bottom), ??Ë is initialized to 0.15 and remains relatively stable with occasional spikes. For the real-scene dataset (top), ??Ë is initialized to 0.2 and exhibits more pronounced temporal fluctuations due to sensor noise and real-world intensity transitions. These burst-like perturbations reflect dynamic changes in photovoltage, which are used to compute the contrast between adjacent frames. A properly adjusted ??Ë is critical for robustly converting such contrast into events during training.

## C.2 Qualitative comprisons of the motion deblurring

In the main paper, we have already presented qualitative results under varying levels of motion blur. In this section, we further provide additional visual comparisons on the synthetic dataset to evaluate the robustness of different methods across mild, medium, and severe blur conditions. As shown in Figure 10, increasing blur levels degrade the reconstruction quality of baseline methods to varying degrees. Compared to D3DGS, which struggles to recover sharp structures under heavy blur, and E2VID+D3DGS, which introduces artifacts from video reconstruction, our method E-4DGS consistently produces sharper and more temporally coherent results. Although Deblur4DGS mitigates some blur-related degradation, it lacks the geometric consistency offered by our event-guided framework. Overall, E-4DGS achieves high-fidelity reconstructions across all blur settings, demonstrating its robustness and effectiveness under challenging motion scenarios.

## C.3 Per-Scene Breakdown

Table 5 presents the quantitative results of all methods for each of the eight synthetic scene sequences, simulated with default settings that are optimal for all methods. The per-scene results are generally consistent with the aggregate metrics, as discussed in Section 5.1.2. Our method outperforms the baselines in most scenes and shows comparable performance in others.

## D Broader Impact and Limitations

Broader Impact. The proposed E-4DGS framework opens up new possibilities for high-fidelity 4D reconstruction in domains where traditional cameras fall short due to motion blur or limited dynamic range. By leveraging the high temporal resolution of event cameras, our method enables temporally coherent scene modeling under rapid motion, which is beneficial for a variety of real-world applications including autonomous robotics, high-speed inspection, sports analytics, and scientific visualization in challenging illumination conditions. Furthermore, the ability to reconstruct dynamic scenes using purely event-based supervision contributes to the development of low-latency, power-efficient visual systems, which are particularly relevant for resource-constrained or edge computing scenarios.

Table 5: Per-synthetic scene breakdown under the default setting.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Method</td><td colspan="8">Synthetic Scene</td><td rowspan="2">Average</td></tr><tr><td>Lego</td><td>Rubik&#x27;s Cube</td><td>MC-Toy</td><td>Hinge</td><td>Cubes</td><td>Capsule</td><td>Restroom</td><td>Garage</td></tr><tr><td rowspan="6">PSNR â</td><td> $\mathrm { D 3 D G S } _ { w / o \ b l u r }$ </td><td>26.47</td><td>20.30</td><td>31.23</td><td>28.05</td><td>21.75</td><td>21.64</td><td>20.67</td><td>20.36</td><td>23.81</td></tr><tr><td> $\mathrm { D 3 D G S } _ { w / \ b l u r }$ </td><td>23.62</td><td>18.12</td><td>27.51</td><td>26.46</td><td>19.67</td><td>20.08</td><td>19.32</td><td>19.05</td><td>21.73</td></tr><tr><td> $_ { \mathrm { E 2 V I D + D 3 D G S } }$ </td><td>20.57</td><td>16.16</td><td>26.06</td><td>24.87</td><td>17.98</td><td>18.49</td><td>17.17</td><td>17.79</td><td>19.88</td></tr><tr><td> $\mathrm { D e b l u r 4 D G S }$ </td><td>23.17</td><td>17.68</td><td>28.06</td><td>26.35</td><td>19.27</td><td>20.10</td><td>19.39</td><td>19.23</td><td>21.66</td></tr><tr><td> $\mathrm { E } { - } 4 \mathrm { D G S } _ { e v e n t - o n l y }$ </td><td>26.85</td><td>20.97</td><td>31.85</td><td>28.83</td><td>22.36</td><td>24.23</td><td>23.17</td><td>24.81</td><td>25.38</td></tr><tr><td> $\mathrm { E } { - } 4 \mathrm { D } \mathrm { G } \mathrm { S } _ { e v e n t \& R G B }$ </td><td>27.23</td><td>21.23</td><td>32.41</td><td>29.02</td><td>22.42</td><td>24.39</td><td>23.30</td><td>24.95</td><td>25.62</td></tr><tr><td rowspan="6">LPIPS â</td><td> $\mathrm { D 3 D G S } _ { w / o \ b l u r }$ </td><td>0.099</td><td>0.207</td><td>0.077</td><td>0.074</td><td>0.129</td><td>0.271</td><td>0.278</td><td>0.251</td><td>0.173</td></tr><tr><td> $\mathrm { D 3 D G S } _ { w / \ b l u r }$ </td><td>0.250</td><td>0.351</td><td>0.181</td><td>0.160</td><td>0.175</td><td>0.436</td><td>0.406</td><td>0.409</td><td>0.296</td></tr><tr><td> $_ { \mathrm { E 2 V I D + D 3 D G S } }$ </td><td>0.346</td><td>0.404</td><td>0.267</td><td>0.247</td><td>0.298</td><td>0.595</td><td>0.527</td><td>0.493</td><td>0.397</td></tr><tr><td> $\mathrm { D e b l u r 4 D G S }$ </td><td>0.265</td><td>0.375</td><td>0.176</td><td>0.162</td><td>0.181</td><td>0.402</td><td>0.386</td><td>0.385</td><td>0.291</td></tr><tr><td> $\mathrm { E } { - } 4 \mathrm { D G S } _ { e v e n t - o n l y }$ </td><td>0.084</td><td>0.185</td><td>0.071</td><td>0.069</td><td>0.120</td><td>0.183</td><td>0.189</td><td>0.172</td><td>0.134</td></tr><tr><td> $\mathrm { E } { - } 4 \mathrm { D } \mathrm { G } \mathrm { S } _ { e v e n t \pm } \kappa _ { R G B }$ </td><td>0.078</td><td>0.172</td><td>0.068</td><td>0.067</td><td>0.119</td><td>0.178</td><td>0.184</td><td>0.165</td><td>0.129</td></tr><tr><td rowspan="6">SSIM â</td><td> $\mathrm { D 3 D G S } _ { w / o \ b l u r }$ </td><td>0.910</td><td>0.868</td><td>0.956</td><td>0.936</td><td>0.924</td><td>0.770</td><td>0.765</td><td>0.757</td><td>0.861</td></tr><tr><td> $\mathrm { D 3 D G S } _ { w / \ b l u r }$ </td><td>0.821</td><td>0.804</td><td>0.905</td><td>0.908</td><td>0.905</td><td>0.730</td><td>0.686</td><td>0.620</td><td>0.797</td></tr><tr><td> $_ { \mathrm { E 2 V I D + D 3 D G S } }$ </td><td>0.765</td><td>0.752</td><td>0.851</td><td>0.856</td><td>0.852</td><td>0.655</td><td>0.547</td><td>0.549</td><td>0.728</td></tr><tr><td> $\mathrm { D e b l u r 4 D G S }$ </td><td>0.813</td><td>0.786</td><td>0.908</td><td>0.900</td><td>0.898</td><td>0.736</td><td>0.695</td><td>0.643</td><td>0.797</td></tr><tr><td> $\mathrm { E } { - } 4 \mathrm { D G S } _ { e v e n t - o n l y }$ </td><td>0.912</td><td>0.882</td><td>0.959</td><td>0.942</td><td>0.931</td><td>0.842</td><td>0.829</td><td>0.874</td><td>0.896</td></tr><tr><td> $\mathrm { E } { - } 4 \mathrm { D } \mathrm { G } \mathrm { S } _ { e v e n t \pm } \kappa _ { R G B }$ </td><td>0.925</td><td>0.895</td><td>0.963</td><td>0.949</td><td>0.933</td><td>0.848</td><td>0.835</td><td>0.879</td><td>0.903</td></tr></table>

Limitations. While E-4DGS demonstrates promising results in dynamic 3D scene reconstruction, certain scenarios, such as those involving extreme motion or significant occlusions, may present challenges for the method. The performance is highly dependent on the availability of synchronized multi-view event data and precise camera calibration. These aspects are areas for further exploration to enhance robustness and generalizability in more complex environments.

Project Release. We implemented E-4DGS based on the official code of Deformable3DGS [81], Gaussianflow [36], E-NeRF [29] and Event3DGS [19] with Pytorch Upon the publication of the paper, we will release the project materials.

<!-- image-->  
Figure 10: Qualitative comparison under varying motion blur levels on synthetic scenes. As blur severity increases, baseline methods (D3DGS and Deblur4DGS) suffer from degraded reconstructions with noticeable artifacts or loss of geometric consistency. In contrast, our method (E-4DGS) produces high-fidelity renderings with sharper details and improved temporal coherence across all blur levels, demonstrating its robustness and effectiveness under fast motion.