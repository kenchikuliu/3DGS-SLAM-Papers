# Dark-EvGS: Event Camera as an Eye for Radiance Field in the Dark

Jingqian Wu1 Peiqi Duan2,3 Zongqiang Wang4 Changwei Wang5 Boxin Shi2,3 Edmund Y. Lam1

1 The University of Hong Kong

2 State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University

3 National Engineering Research Center of Visual Technology, School of Computer Science, Peking University

4 Chinese Academy of Science 5 Qilu University of Technology

jingqianwu@connect.hku.hk {duanqi0001, shiboxin}@pku.edu.cn wangzongqiang2022@mail.ia.ac.cn changweiwang@sdas.org elam@eee.hku.hk

## Abstract

In low-light environments, conventional cameras often struggle to capture clear multi-view images of objects due to dynamic range limitations and motion blur caused by long exposure. Event cameras, with their high-dynamic range and high-speed properties, have the potential to mitigate these issues. Additionally, 3D Gaussian Splatting (GS) enables radiance field reconstruction, facilitating bright frame synthesis from multiple viewpoints in low-light conditions. However, naively using an event-assisted 3D GS approach still faced challenges because, in low light, events are noisy, frames lack quality, and the color tone may be inconsistent. To address these issues, we propose Dark-EvGS, the first event-assisted 3D GS framework that enables the reconstruction of bright frames from arbitrary viewpoints along the camera trajectory. Triplet-level supervision is proposed to gain holistic knowledge, granular details, and sharp scene rendering. The color tone matching block is proposed to guarantee the color consistency of the rendered frames. Furthermore, we introduce the first real-captured dataset for the event-guided bright frame synthesis task via 3D GS-based radiance field reconstruction. Experiments demonstrate that our method achieves better results than existing methods, conquering radiance field reconstruction under challenging low-light conditions. The code and sample data are included in the supplementary material.

## 1. Introduction

Low-light radiance field reconstruction plays a crucial role in various real-world applications, including nighttime photography [16], surveillance [15], and autonomous driving [8]. However, traditional frame-based cameras struggle to capture high-quality images in dark environments due to their limited dynamic range and reliance on long exposure times. These constraints result in significant noise and motion blur, making it difficult to capture clear, bright frames of objects from multiple viewpoints in low-light conditions.

<!-- image-->  
Figure 1. In the dark, only noisy events and dark blurred frames can be captured (a), which is challenging for existing approaches for radiance field reconstruction. Our approach takes the raw events and sparse dark frames as input (c), and it reconstructs a radiance field, enabling arbitrary viewpoint synthesis (b). Our approach is capable of synthesizing dense, bright, and sharp views, enhancing visibility and detail (d).

Event cameras are a novel type of sensor that captures brightness changes asynchronously per pixel, marking a shift from conventional frame-based imaging [29, 37, 38]. With high dynamic range (HDR) and high temporal resolution capabilities, they can simultaneously capture bright and dark regions with no motion blur, producing rich unseen signals omitted by frame-based cameras, providing a potential solution for reconstructing in low-light conditions. Event-based video reconstruction methods were proposed [6, 14, 20] to output consecutive frames from events. Nevertheless, either they are limited to grayscale images [6, 20], or primarily focus on the short exposure problem [14] rather than low-light conditions (less than 40 lux). Event data can also be applied to radiance field reconstruction, but existing approaches [21, 25, 27] are only designed for normal light conditions (around 300 lux).

Reconstructing a radiance field in the dark is a hard task because of three main challenges: 1) Signals captured lack quality as events are noisier and frames contain lower intensity (Fig. 1 (a) and (c)); 2) color tone consistency is required for the rendered frames; and 3) there are currently no eventbased dark radiance field reconstruction work and dataset available so no prior solution and data can be referenced for solving this problem. Due to these challenges, not only did the existing event-based radiance field approach [21, 25, 27] fail to become an effective solution, but also, event-based video reconstruction work [6, 14, 20] cannot be used directly in this task.

In this paper, we propose Dark-EvGS by integrating the event camera and 3D Gaussian Splatting (GS) [12], a radiance field reconstruction technique, enabling bright frame reconstruction from arbitrary viewpoints along the camera trajectory. The high-dynamic sensing capability of event cameras enables Dark-EvGS to have precise perception of camera motion, which can provide valuable guidance for radiance field reconstruction performance (Fig. 1 (b)) and render bright frames from multiple viewpoints of objects in low-light conditions (Fig. 1 (d)).

Dark-EvGS conquers the mentioned challenges in three directions: 1) We propose a triplet-level supervision mechanism specifically tailored for handling noisy event and frame signals in low-light radiance field reconstruction; 2) the Color Tone Matching Block (CTMB) is proposed to guarantee color consistency of the rendered frames; 3) we collect a dataset for the event-guided bright frame synthesis task via 3D GS-based radiance field reconstruction, which includes paired low-light frames, bright ground truth frames, event streams, and corresponding camera poses. Extensive experiments demonstrate that our method outperforms previous state-of-the-art techniques across all samples. The outlined key technical contributions are:

â¢ We introduce Dark-EvGS, the first event-guided bright frame synthesis method via 3D GS-based radiance field reconstruction from arbitrary viewpoints in dark environments.

â¢ We present a triplet-level supervision strategy to recover missed holistic structures, restore fine-grained details, and enhance sharpness for 3D GS training and for rendering color-consistent bright frames in low-light environments.

â¢ We collect the first real-world event-based low-light radiance field reconstruction dataset with paired low-light, bright-light frames, event streams, and corresponding camera poses. We will release the dataset and code to facilitate future research.

## 2. Related Work

General and Dark Radiance Field. Radiance field rendering and novel-view synthesis are fundamental tasks in graphics and computer vision with extensive applications in fields like robotics and virtual reality. Neural Radiance Fields (NeRF) [17] and 3D GS [12] have made substantial progress in these tasks. While NeRF produces high-fidelity renderings with intricate detail, the high number of samples required to accumulate the color information for each pixel results in low rendering efficiency and extended training times [12]. 3D GS, on the other hand, employs a set of optimized Gaussians to achieve state-of-the-art quality in reconstruction and rendering speed [25]. Starting from sparse point clouds generated by Structure-from-Motion (SfM), 3D GS uses differentiable rendering to control the density and refine parameters adaptively.

Most NeRF and 3D GS-based methods use frame data from traditional cameras [18]. Few works have been attempted to reconstruct radiance fields in dark and low-light environments due to the limited dynamic range of a framebased camera. Zhang et al. [35] attempt to solve robot exploration in the dark using 3D Gaussians Relighting, however, a constant illumination is placed in front of the robot. Mildenhall et al. [18] use NeRF to reconstruct the radiance field in the dark by treating it as a denoising problem from HDR frames. However, it requires an HDR sensor in real applications and does not explore the effect of motion blur in low-light conditions. Therefore, no existing work has attempted to leverage the dynamic range and temporal resolution property of an event camera to solve such an issue. Event-Based Image Reconstruction. Many works have been devoted in the field of events to frame reconstruction [1, 6, 20]. However, these methods do not work well for reconstructing frames from low-light events. EvLow-Light [14] attempts to reconstruct bright-light video by incorporating event data. However, the focus is primarily on the short-to-long exposure problem rather than the darkto-bright reconstruction. In their case, the darkness of the frames is caused by short exposure times, not actual lowlight conditions, meaning the scene remains well-lit from the event cameraâs perspective. In contrast, our task focuses on using an event camera to reconstruct bright radiance fields in real low-light environments. Others [19, 34] have tried to use an event to guide video reconstruction in the dark. However, estimating absolute intensity values in a video solely from brightness changes recorded in events is a highly ill-posed problem [14]. Thus, directly applying any of these models to our task is not feasible.

Event-Based Radiance Field Reconstruction. Event cameras, with their unique characteristics, such as motion-blur resistance, high dynamic range, low latency, and energy efficiency, are increasingly utilized in computer vision and computational imaging applications [5, 10, 33, 36]. Apa rotation matrix R and a scaling matrix S:

<!-- image-->  
Figure 2. Overview of the Dark-EvGS pipeline for radiance field reconstruction in the dark. We obtain dark frames, events, and camera parameters using an event camera under low lights. The frame encoder and event encoder will extract features, which will be then forwarded to multimodal coherence modeling. The proposed Color Tone Consistency Module takes the features and decodes them into pseudo bright frames with consistent color tone (described in Sec. 3.3). Via the 3D GS model (described in Sec. 3.1), we render bright and shape frames when given the corresponding camera position and pose and formulate (blue arrow) the rendered results to predicted event signals (described in Sec. 3.2). The same process applies when formulating captured events to ground truth supervision signals. To supervise accurate training, the proposed triplet-level loss (red lines) provides a holistic view while keeping the rendered results sharp and accurate in detail (described in Sec. 3.4). As a result, our method enables high-quality radiance field reconstruction in the dark.

proaches integrating NeRF with event data [13, 21] apply volumetric rendering using event supervision, which takes advantage of NeRFâs inherent multi-view consistency to extract coherent scene structures. However, the computational demands for training and optimizing event-based NeRF pipelines remain substantial [25]. Recent works have aimed to integrate 3D GS with event data [3, 25â27, 31, 32]. Regardless of the base approach, existing event-involved approaches were designed for general scenes. For those who take event streams as input only, the noisier and more random nature of events under dark and low-light environments makes these approaches unrobust and ineffective for reconstruction. For others who also utilize blurred frames as additional input, it further suffers from the low intensity in low-light frames.

## 3. Methodology

## 3.1. Preliminary on 3D Gaussian Splatting

The 3D Gaussian Splatting (3D GS) approach [12] models detailed 3D scenes as point clouds, where each point is represented as a Gaussian, defining the structure of the scene. Each Gaussian is characterized by a 3D covariance matrix Î£ and a central location x:

$$
G ( x ) = \exp \left( - \frac 1 2 { \bf x } ^ { T } \Sigma ^ { - 1 } { \bf x } \right) ,\tag{1}
$$

where the central location x serves as the Gaussianâs mean. To enable differentiable optimization, Î£ is decomposed into

$$
\Sigma = R S S ^ { T } R ^ { T } .\tag{2}
$$

Rendering from various viewpoints utilizes the splatting technique from [30], which projects Gaussians onto camera planes. This technique, based on the method in [39], involves the viewing transformation matrix W and the Jacobian J of the affine projective transformation. In-camera coordinates, the covariance matrix $\Sigma ^ { \prime }$ is then represented as:

$$
\Sigma ^ { \prime } = J W \Sigma W ^ { T } J ^ { T } .\tag{3}
$$

To summarize, each Gaussian point includes several attributes: position $\mathbf { x } \in \mathbb { R } ^ { 3 }$ , color encoded via spherical harmonics coefficients $\mathbf { \chi } ) \in \mathbb { R } ^ { k }$ (with k indicating the degrees of freedom), opacity $\alpha \in \mathbb { R }$ , rotation quaternion $\mathbf { q } \in \mathbb { R } ^ { 4 }$ , and scaling factor $\mathbf { s } \in \mathbb { R } ^ { 3 }$ . For each pixel, the combined color and opacity values of all Gaussians are computed using the Gaussian representation from Equation 1. The blending of colors C for N ordered points projected onto a pixel is:

$$
C = \sum _ { i \in N } \mathbf { c } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{4}
$$

where $\mathbf { c } _ { i }$ and $\alpha _ { i }$ represent the color and density of each point, respectively. These values are influenced by each Gaussianâs covariance matrix Î£ and modified through perpoint opacity and spherical harmonics color coefficients.

<!-- image-->  
(a) Raw Events

<!-- image-->  
(b) Filtered Events  
Figure 3. Frame-based cameras are unable to capture enough signals for residence reconstruction in the dark. Event cameras, on the other hand, can capture ample signals, including ignored details by frame-based cameras (a). However, these raw events captured are rather noisy and random (a), which is one of the biggest challenges for existing methods [14, 20, 21, 25]. To solve this problem, we proposed a utilization (Sec. 3.2) for noisy events for accurate and clean supervisory signals (b).

## 3.2. Noisy Events in Dark as Supervisory Signals

The event sensor output can be formulated as:

$$
E _ { g t } = \Gamma \left\{ \log \left( \frac { I _ { t } + c } { I _ { t - w } + c } \right) , \epsilon \right\} ,\tag{5}
$$

where $E _ { g t }$ is the captured events, $I _ { t }$ and $I _ { t - w }$ are intensity images at the two timestamps, $\Gamma \{ \theta , \epsilon \}$ represents the conversion function from log-intensity to events, c is an offset value to prevent log(0), Ïµ is the event triggering threshold. $\Gamma \{ \theta , \epsilon \} = 1$ when $\theta \geq \epsilon ,$ indicating positive event triggered, and $\Gamma \{ \theta , \epsilon \} = - 1$ when $\theta \leq - \epsilon$ , indicating negative event triggered [4]. Else, no events are triggered.

Events triggered in low-light environments are much noisier [11]. To address noise, which is often more prevalent in microsecond-level event data [28], we preprocess the event stream using the y-noise filter from [7]. As shown in Fig. 3 (a), raw events captured in the dark contain extreme noise and randomness. After applying the noise filter, the cleaned event stream becomes more helpful as a supervision signal, as shown in Fig. 3 (b). Noise filtering is essential for accurate view rendering, as shown by ablation studies in the supplementary material.

Our objective, depicted in Fig. 2, is to reconstruct a radiance field using differentiable 3D Gaussian functions $G ,$ guided by event and corresponding blurred dark frames. We accumulate the ground truth event signal between timestamps $t _ { 1 }$ and $t _ { 2 }$ to form the supervisory signal using Eq. 5 by aggregating the polarities of all events occurring between times t and $t - w .$ , indexed by their pixel location.

At each timestamp, $t _ { k }$ , a rendering result is produced with the camera at pose $p _ { k }$ . Thus, we calculate two rendered frames from the 3D GS model: $I _ { 1 } ~ = ~ G ( p _ { 1 } )$ and $I _ { 2 } = G ( p _ { 2 } )$ , where $I _ { 1 }$ and $I _ { 2 }$ are RGB renderings at times $t _ { 1 }$ and $t _ { 2 }$ , respectively. Here, G denotes the 3D GS model, and $p _ { 1 }$ and $p _ { 2 }$ are the camera poses at timestamps $t _ { 1 }$ and $t _ { 2 }$

The logarithmic image representation is defined as $L ( I _ { t } ) =$ log $\left( ( I _ { t } ) ^ { g } + \kappa \right)$ , where $\kappa = 1 \times 1 0 ^ { - 5 }$ (to prevent NaN values) and $g$ is a gamma correction factor set to 2.2 across all experiments, following [2, 21]. From this, we obtain the predicted cumulative difference $E _ { p r e d } = L ( I _ { 2 } ) - L ( I _ { 1 } )$ .

## 3.3. Color Tone Consistency Module

Pure event utilization and supervision are insufficient for radiance field reconstruction on real data [9, 25], let alone reconstruction under low-light environments. Therefore, prior knowledge is required in the 3D GS training pipeline. Following EvLowLight [14], we deploy feature encoders for both frame and event modality along with the Multimodal Coherence Modeling block (MCM). We then propose the Color Tone Consistency Module (CTCM), which consists of a proposed Color Tone Matching Block (CTMB) along with the decoders from MCM. CTMB takes the decoded frames during the decoding process and corrects the color tone. Finally, the CTCM outputs the predicted pseudo frames using the corrected frames and the exposure parameter predicted from the MCM. Specifically, the CTMB, inspired by recent advancements in transposed self-attention [24], takes the last feature map $F \in \mathsf { \bar { R } } ^ { H \times W \times C }$ , and derives query $Q ,$ , key $K ,$ , and value V representations using a 1Ã1 convolution followed by a depth-wise convolution. Subsequently, Q and V are reshaped, and an attention map $M \in \mathbf { \bar { \Gamma } } \mathbf { \bar { \mathbb { R } } } ^ { C \times \bar { C } }$ is computed through matrix multiplication. The output of the CTMB $F _ { \mathrm { o u t } }$ represents the color-corrected feature, and the bright light frame with corrected color tone can be derived from $F _ { \mathrm { o u t } }$ and the exposure parameter from MCM. This allows CTCM to generate global color correction via channel-wise self-attention while enhancing local color adjustments through convolutional operations.

## 3.4. Triplet Supervision for Radiance Field in Dark

There are three major challenges in supervision under low light conditions: First, events are too noisy to provide accurate supervision alone. Second, pseudo-bright frames generated by CTCM cannot provide robust details due to the data gap between pre-trained knowledge and actual training scenarios. Third, in low light environments, captured dark frames are easier to motion blur as a longer exposure time is required. Thus, a sharpening module is needed because the pseudo-bright frames may also contain motion blur when given blurred dark frames.

To tackle these challenges, we propose triplet-level supervision to train and reconstruct radiance field representation in the dark accurately. Specifically, frame-level holistic supervision, event-level granular supervision, and mixedmodality sharpening supervision. At the first level, Dark-EvGS leverages pseudo-bright frames, though it may be inaccurate in detail, to obtain a holistic view of knowledge through frame-level supervision. At the second level, filtered events serve as a supervisory signal to refine granular details that are ignored in previously used frames. Finally, at the third level, to mitigate motion blur effects in dark scenes, a combined event and frame supervision approach is applied using a mixed-modality sharpening strategy.

Frame-based Holistic Supervision. Pure events supervision brings additional noise under low lights and does not lead to good reconstruction results (Fig. 6 (f)). Though the pseudo bright light frame, generated by CTCM, may contain blur and minor defects in details, as ground truth, it provides a holistic overview of how the object in the current frame looks and supervises the rendered frame so it follows that high-level holistic framework.

At a time window with range $t _ { 1 }$ and $t _ { 2 } ,$ given two lowlight dark frames $I _ { l o w } ^ { t _ { 1 } }$ and $I _ { l o w } ^ { t _ { 2 } }$ captured at these two timestamps, and the ground truth accumulated events $E _ { g t }$ between the time window, the Color Tone Consistency Module, CTCM, generates two estimated bright frame $B ^ { t _ { 1 } }$ , and $B ^ { t _ { 2 } }$ , where $( \bar { B } ^ { t _ { 1 } } , B ^ { t _ { 2 } } ) = \boldsymbol { \mathrm { C } } \boldsymbol { \mathrm { T C } } \boldsymbol { \mathrm { M } } ( I _ { l o w } ^ { t _ { 1 } } , \bar { I } _ { l o w } ^ { t _ { 2 } } , E _ { g t } )$ . We then use the rendered frame from the GS model to compute L2 loss with an estimated bright frame B. Therefore, the frame-based holistic loss is defined as:

$$
\mathcal { L } _ { h o l } ( I , B ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( B _ { i } - I _ { i } ) ^ { 2 } .\tag{6}
$$

Event-level Granular Supervision. Granular details are ignored by the frame camera when captured in the dark due to limited dynamic range. Furthermore, the estimated pseudo-bright frame from CTCM can only give a holistic view, ignoring the same region of sharp and accurate details. Event streams captured by the event camera play a critical role in supervising details and minors because they are high in temporal resolution, so details between frames can be captured, and they are wide in dynamic range, so unrevealed information can be seen. We use a y-noise filter [7] Y to form ground truth event $E _ { g t } = Y ( E _ { r a w } )$ from raw captured events $E _ { r a w }$ . To learn $E _ { p r e d }$ with the formulated clean event signals $E _ { g t }$ , we adopt the classic event supervision loss from [21], where N is the number of viewpoints:

$$
\mathcal { L } _ { e v e n t } ( E _ { p r e d } , E _ { g t } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( E _ { p r e d } ^ { i } - E _ { g t } ^ { i } ) ^ { 2 } .\tag{7}
$$

This helps to refine details that the frame-based holistic supervision failed to reconstruct.

Mixed-Modality Sharpening Supervision. Because the generated pseudo-bright frames from CTCM may contain motion blur, applying a strict L2 loss leads to the rendered results being blurry as well. To correct motion blur in the rendering process, we propose a mixed-modality sharpening loss. The key idea is: though individual generated bright frames from CTCM contain motion blur, consecutive frames retain temporal consistency - motion blur and sharp areas of nearby frames below to similar area and thus can be compensated by subtraction (Fig. 4 green boxes). Mathematically speaking, the subtraction (Fig. 4 absolute difference) between two generated bright frames is sharp and stands for the pixel shift in spatial. Thus, it can be a good guide for the spatial shift to rendered GS results. Therefore, we map the sharp subtraction results using the logarithmic function L, defined in section 3.2, to formulate the computed event signal and then supervise the predicted events from the 3D GS network. This effectively eliminates the blurring effect and produces sharper results during rendering. The mixed-modality sharpening loss ${ \mathcal { L } } _ { m i x }$ is defined as:

<!-- image-->  
Figure 4. Illustration of our proposed mix-modality sharpening loss. The blurred regions (green boxes) can be offset, thus producing sharper rendering results.

$$
\begin{array} { c } { { \displaystyle \mathcal { L } _ { m i x } \big ( I ^ { t _ { 1 } } , I ^ { t _ { 2 } } , B ^ { t _ { 1 } } , B ^ { t _ { 2 } } \big ) = } } \\ { { \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( L ( I ^ { t _ { 1 } } - I ^ { t _ { 2 } } ) - L ( B ^ { t _ { 1 } } - B ^ { t _ { 2 } } ) ) ^ { 2 } . } } \end{array}\tag{8}
$$

Final Loss. The final loss function is expressed as:

$$
\mathcal { L } = \mathcal { L } _ { h o l } + \lambda _ { 1 } \mathcal { L } _ { e v e n t } + \lambda _ { 2 } \mathcal { L } _ { m i x } ,\tag{9}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ is set to 0.25 across all experiments.

Noise filtering using the y-noise filter effectively removes interfering signals, especially in dark and low-light, while the triplet supervision keeps the training stable across varying conditions. This stability ensures consistent rendering performance, regardless of data source, and is supported by ablation studies in the supplementary materials.

## 4. Experiments

This section provides details of our data collection, experiment setup, evaluation results, and analysis. Due to space limitations, ablation tests and rest are included in the supplementary materials.

<!-- image-->  
Figure 5. Demonstration of hardware setup and dataset collection. An object is placed on a turntable with an event camera pointing to it (a), and signals are captured both in bright (b) and dark (c). In low-light environments, frames captured lack information and blur, and events are noisy.

## 4.1. Data Collection

In this subsection, we introduce the real-world data collection process in detail. Figure 5 (a) demonstrates the hardware setup. To collect this real-world dataset, we place a static object on a constant, unknown-speed turntable. A Davis 346 Color event camera is also held steady, pointing to the object. We use a TES 1339R light meter to measure the darkness of the environment in lux. Six scenes are collected, and for each scene, we set two types of lighting conditions to form paired data: bright (around 300 lux) and dark (less than 40 lux). We cover all items using curtains, ensuring no natural light is involved. A light with adjustable brightness is placed directly above the object. With bright light, use the highest brightness and capture the corresponding frames, events, and camera info when turning. Then, with the same 360-degree rotation trajectory, leaving limited lights on the object, and capturing the same set of data under this low-light condition. For six scenes in total, three are classified as moderate (between 40 and 20 lux) level data (âBaseballâ, âHouseâ, âLionâ), and three are classified as challenging (less than 20 lux). For each scene, paired dark-bright frames, event streams, and camera poses were prepared. We use the bright light frame as the evaluation ground truth in all our experiments. Dark events and/ or dark frames are the only signals that can be used during the training and supervising process. Figure 5 (b) and (c) demonstrate the example data collected under both lighting conditions. As clearly shown by the contrast, frames captured under normal lights are sharp and bright; events are cleaner. However, frames captured under dark lights have low intensity, lose critical information, and canât be processed by either machine or human. Concurrently, events captured were much noisier and random to handle.

## 4.2. Implementation Details

Our implementation builds upon the 3D GS framework [12], leveraging its primary structure and functionalities. As

3D GS requires a point cloud input, we randomly initialize 103 points to create the starting point cloud, as the structurefrom-motion initialization [22] in the original setup is not directly applicable to event data. We use the pre-trained weights from EvLowLight on its existing modules with randomly initialized CTMB weights as a start. They are then tuned with five hundred unrelated real-world dark-bright data pairs and corresponding events for 5 epochs to provide robust prior knowledge on pseudo frames. All experiments were conducted on an NVIDIA RTX 3090 GPU. We used specific hyperparameters to ensure optimal performance. Training was conducted for a total of 30,000 iterations for all scenes. For position optimization, the learning rate was scheduled to decay from $1 . 6 \times 1 0 ^ { - 4 } \mathrm { t o } 1 . 6 \times 1 0 ^ { - 6 }$ The learning rates for optimizing features, opacity, scaling, and rotation were set to $2 . 5 \times 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 2 } , 5 \times 1 0 ^ { - 3 }$ and $1 \times 1 0 ^ { - 3 }$ , respectively.

## 4.3. Comparisons against Related Methods

We compare the radiance field reconstruction results and efficiency of various methods [12, 14, 20, 21, 25] on our proposed dataset, with the following aspects: (a) Reconstruction Quality: We used PSNR, SSIM, and LPIPS to measure the similarity in brightness, structure, and perception between the reconstructed images and bright-light images. In this context, higher PSNR and SSIM values indicate better image quality, while lower LPIPS values signify better perceptual similarity to the ground truth. (b) Efficiency: We evaluated training time per item, real-time rendering FPS, and GPU memory usage to assess the training efficiency, real-time rendering performance, and resource consumption of each method.

Quantitative Evaluation. As shown in Table 1, our method outperforms other methods across various realworld scenes. In comparison to other baseline approaches, such as EventNeRF [21], Ev-GS [25], and EvLowLight [14], our approach consistently delivers higher PSNR and SSIM values while maintaining lower LPIPS values, showcasing notable improvements in brightness, structure, and perceptual similarity.

In terms of efficiency, our method achieves a balance between efficiency and performance, requiring only 4 minutes for training in most scenes, comparable to efficient methods like Ev-GS, and far faster than EventNeRF [21], which requires up to 26 hours. Our approach also delivers competitive real-time rendering speeds, maintaining approximately 41 FPS on moderate scenes and 35 FPS on challenging scenes, effectively balancing high-quality output with rendering speed. Additionally, our GPU memory usage remains lightweight and easily reproducible for most researchers and devices.

Quantitative Evaluation. As illustrated in the comparison between Fig. 6 (a) and Fig. 6 (i), frames captured under low-light conditions lose certain details and cannot accurately reflect the true scene compared to frames captured under bright light conditions. As shown in Fig. 6 (b), event data is less affected by lighting and can capture details, such as shadowed areas, that are invisible to standard cameras. Leveraging the strengths of event data, our specially designed method produces renderings with greater detail and clarity, as shown in Fig. 6 (h). Compared to methods such as 3D GS [12], E2VID [20] with 3D GS, Event-NeRF [21], EvGS [25], and EvLowLight [14] with 3D GS, our reconstructed images are closer in detail and structure to the brightly-lit reference images, representing the actual scene under low-light conditions, as shown in Fig. 6 (c-g).

Table 1. Quantitative Comparison Against Other Methods on Real Dataset. The evaluation metrics include PSNR, SSIM, and LPIPS across various scenes. â indicates that higher values are better, while â indicates that lower values are better. The metrics in bold are ranked first, and the underlined metrics are ranked second.
<table><tr><td colspan="10">Moderate Scene</td></tr><tr><td>Method</td><td>PSNRâ</td><td>Baseball SSIM</td><td>LPIPS â</td><td>PSNRâ</td><td>House SSIM</td><td>LPIPSâ</td><td>PSNRâ</td><td>Lion SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3D GS [12]</td><td>9.91</td><td>0.15</td><td>0.52</td><td>8.79</td><td>0.21</td><td>0.55</td><td>9.32</td><td>0.23</td><td>0.58</td></tr><tr><td>EEVIDD [20]</td><td>9.78</td><td>0.37</td><td>0.65</td><td>9.32</td><td>0.49</td><td>0.9</td><td>9.54</td><td>0.46</td><td>0..72</td></tr><tr><td>EventNeRF [21]</td><td>9.05</td><td>0.59</td><td>0.58</td><td>10.06</td><td>00.62</td><td>0.67</td><td>10.43</td><td>0.60</td><td>0.68</td></tr><tr><td>EvGs [25]</td><td>15.82</td><td>0..72</td><td>0.51</td><td>113.27</td><td>0.67</td><td>00.57</td><td>114.95</td><td>0.68</td><td>0.60</td></tr><tr><td>E2GS [3]</td><td>14.90</td><td>0.70</td><td>0..55</td><td>14.31</td><td>0.68</td><td>0.55</td><td>14.19</td><td>0.70</td><td>0.62</td></tr><tr><td>SweepEvGS [26]</td><td>20.25</td><td>0.79</td><td>0.40</td><td>19.29</td><td>0.72</td><td>0.41</td><td>20.37</td><td>0.75</td><td>0.45</td></tr><tr><td>EvLowLight [14  Ours</td><td>20.34</td><td>0.83</td><td>0.38</td><td>119.31</td><td>0.75</td><td>0.45</td><td>118.86</td><td>00.74</td><td>0.49</td></tr><tr><td></td><td>26.58</td><td>0.87</td><td>0.31</td><td>23.26</td><td>0.81</td><td>0.36</td><td>31.22</td><td>0.85</td><td>0.30</td></tr><tr><td colspan="10">Challenging Scene</td></tr><tr><td>Method</td><td>PSNRâ</td><td>Panda SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>Badminton SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>Cat</td><td>LPIPSâ</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>SSIMâ</td><td></td></tr><tr><td>3D GS [12] E2VID [20]</td><td>8.73</td><td>0.08</td><td>0.69</td><td>14.69</td><td>0.58</td><td>0.35</td><td>10.61</td><td>0.09</td><td>0.46</td></tr><tr><td></td><td>9.38</td><td>0.38</td><td>0.70</td><td>8.26</td><td>0.15</td><td>00.65</td><td>8.92</td><td>0.35</td><td>0.64</td></tr><tr><td>EventNeRF [21]</td><td>9.23</td><td>00.56</td><td>0.69</td><td>55.91</td><td>0..14</td><td>0.48</td><td>5.95</td><td>00.33</td><td>0.46</td></tr><tr><td>EvGS  [25]</td><td>12.37</td><td>0.63</td><td>0.60</td><td>10.89</td><td>0.17</td><td>0.35</td><td>10.58</td><td>0.21</td><td>00.37</td></tr><tr><td>E2GS S [3]</td><td>14.20</td><td>0.69</td><td>0.50</td><td>13.72</td><td>0.68</td><td>0.60</td><td>11.73</td><td>0.27</td><td>0.39</td></tr><tr><td>SweepEvGs [26</td><td>17.22</td><td>0.73</td><td>0.47</td><td>18.84</td><td>0.33</td><td>0.28</td><td>18.56</td><td>0.72</td><td>0.45</td></tr><tr><td>EvLowLight [14]</td><td>116.28</td><td>0.76</td><td>0.46</td><td>118.87</td><td>0.31</td><td>0.20</td><td>19.67</td><td>0.8</td><td>0.38</td></tr><tr><td> Ours</td><td>20.84</td><td>0.80</td><td>0.36</td><td>223.70</td><td>0.42</td><td>0.19</td><td>21.64</td><td>0.83</td><td>0.33</td></tr></table>

Table 2. Comparison of training time (Train), rendering performance (FPS), and GPU memory usage (Mem) across different methods on Moderate Scene and Challenging Scene. The metrics include training time (minutes and hours), frames per second (FPS) for real-time rendering, and GPU memory usage (GB) during training. â indicates that higher values are better, while â indicates that lower values are better. The metrics in bold are ranked first, and the underlined metrics are ranked second.
<table><tr><td rowspan="2">Method</td><td colspan="3">Moderate Scene</td><td colspan="3">Challenging Scene</td></tr><tr><td>Trainâ</td><td>FPSâ</td><td>Memâ</td><td>Trainâ</td><td>FPS</td><td>Memâ</td></tr><tr><td>3D GS [12]</td><td>3min</td><td>40</td><td>1GB</td><td>3min</td><td>32</td><td>1.1GB</td></tr><tr><td>E2VID [20]</td><td>5min</td><td>35</td><td>1.8GB</td><td>5min</td><td>30</td><td>1.8GB</td></tr><tr><td>EventNeRF [21]</td><td>19h</td><td>0.5</td><td>12GB</td><td>26h</td><td>0.3</td><td>12GB</td></tr><tr><td>EvGS [25]</td><td>4.5min</td><td>33</td><td>1.5GB</td><td>4.5min</td><td>31</td><td>2.1GB</td></tr><tr><td>E2GS [3]</td><td>30min</td><td>30</td><td>5GB</td><td>32min</td><td>28</td><td>5GB</td></tr><tr><td>SweepEvGS [26]</td><td>8min</td><td>38</td><td>2GB</td><td>8min</td><td>40</td><td>2GB</td></tr><tr><td>EvLowLight [14]</td><td>11h</td><td>3.5</td><td>2GB</td><td>10h</td><td>3.5</td><td>3GB</td></tr><tr><td>Ours</td><td>4min</td><td>41</td><td>1.2GB</td><td>4min</td><td>35</td><td>1.7GB</td></tr></table>

Analysis on Why Existing Methods Fail. We classify existing frame-based and event-based methods into several categories and analyze the challenges.

3D GS with Frame Camera under Low Light: As shown in Fig. 6 (a), frame cameras failed to capture critical information under low light due to limited dynamic range. This leads to bad radiance field reconstruction and rendering with missing details as well in Fig. 6 (c). However, event cameras are able to capture unseen information in dark environments.

Event-Based Radiance Field Reconstruction: As illustrated in Fig. 1 (b), Fig. 3 (b), and Fig. 5 (c), event streams tend to be highly noisy and random in low-light situations, which undermines accurate rendering. This is because, like all vision sensors, event cameras must comply with the laws of physics, specifically photon counting. Under low-light conditions, the scarcity of photons leads to increased noise and slower counting rates [11]. This makes existing pure event-based radiance field reconstruction methods [25] fail, as they cannot handle the additional noise introduced from low-light and dark environments (Fig. 6 (f)). For the same reason, those hybrid radiance field reconstruction methods, such as EventNeRF [21] that take events and frames, would not be effective as well (Fig. 6 (e)).

Event-Based Frame Reconstruction with 3D GS: This is the classical two-stage solution that first generates frames from dark events using event-based frame reconstruction methods such as E2VID [20] and then trains a 3D GS pipeline using the generated frames. This idea is straightforward but naive because those tools are trained and designed for normal lighting conditions, not for dark, and therefore, could not handle noisy events from the dark. As shown in Fig. 6 (d), the reconstructed results from E2VID [20] lose details of the turntable for the house scene and lose details of the eyes for the lion scene. Therefore, this necessitates the development of more sophisticated and specialized architectures, along with tailored supervision mechanisms for event-based radiance field reconstruction in the dark.

<!-- image-->  
Figure 6. A visual comparison of different approaches on different scenes captured: (a) dark frames captured by a frame-based camera under low lights; (b) filtered events captured under low lights by an event camera; (c) results from 3D GS [12] using dark frames; (d) rendering results from 3D GS using reconstructed frames from E2VID [20]; (e) results from EventNeRF [21]; (f) results from Ev-GS [25]; (g) results from Ev-LowLight; (h) results from ours; (i) ground truth frames captured under bright lights. Our approach renders brighter and sharper results purely using signals captured under dark environments compared to existing approaches across all scenes.

Event-Based Video Enhancement with 3D GS: This type of work [14] is designed to enhance input videos to bright videos, such as EvLowLight [14]. However, it was designed to tackle the short exposure problem rather than the mentioned challenges. Therefore, when taking dark frames from low light environments as inputs, EvLowLight produces poor results due to the task gap. Thus, using such a tool directly to reconstruct a bright frame and train a 3D GS framework would be an ineffective solution. Second, there is currently a large gap in training data distribution between this type of work and real-world radiance field reconstruction in low-light environments. EvLowLight [14] uses synthetic paired short-long exposure frames and simulated events using a simulator [11]. However, these synthetic and simulated data do not reflect the real-world situation: synthetic dark and bright pairs do not reflect the realworld lighting contrast, and events simulated out of the synthetic dark frames are also too clean to represent the realworld noise [23]. Training on such synthetic data makes EvLowLight perform poorly on real-world data and applications. Thus, creating the first real-world event dataset that contrasts both low-light and well-lit scenes is essential and would provide a valuable contribution to this field.

## 5. Conclusion

In this work, we introduce Dark-EvGS, the first eventguided 3D Gaussian Splatting framework for bright frame synthesis from arbitrary views in low-light environments. Leveraging the high dynamic range and speed of event cameras, Dark-EvGS overcomes noise, motion blur, and poor illumination challenges faced by conventional methods. We propose a triplet-level supervision strategy to enhance scene understanding, detail refinement, and sharpness restoration, along with a Color Tone Matching Block to ensure color consistency. Additionally, we introduce the first real-world dataset for event-guided bright frame synthesis in low-light settings. Extensive experiments show that Dark-EvGS outperforms prior approaches, establishing it as a robust solution for low-light radiance field reconstruction.

Limitation. While Dark-EvGS demonstrates significant advancements in radiance field reconstruction under lowlight conditions, it is not without limitations. A key challenge is that the current framework only supports static radiance field reconstruction because Dark-EvGS was built on the 3D GS pipeline, which does not account for temporal variations effectively. For future work, we will explore dynamic radiance field reconstruction in dark.

## References

[1] Pablo Rodrigo Gantier Cadena, Yeqiang Qian, Chunxiang Wang, and Ming Yang. SPADE-E2VID: Spatially-adaptive denormalization for event-based video reconstruction. IEEE Transactions on Image Processing, 30:2488â2500, 2021. 2

[2] International Electrotechnical Commission et al. Multimedia systems and equipment-color measurement and management-part 2-1. Color management-Default RGB color space-sRGB, 1999. 4

[3] Hiroyuki Deguchi, Mana Masuda, Takuya Nakabayashi, and Hideo Saito. E2GS: Event enhanced gaussian splatting. In 2024 IEEE International Conference on Image Processing, pages 1676â1682. IEEE, 2024. 3, 7

[4] Peiqi Duan, Zihao W Wang, Xinyu Zhou, Yi Ma, and Boxin Shi. EventZoom: Learning to denoise and super resolve neuromorphic events. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12824â12833, 2021. 4

[5] Peiqi Duan, Yi Ma, Xinyu Zhou, Xinyu Shi, Zihao W Wang, Tiejun Huang, and Boxin Shi. NeuroZoom: Denoising and super resolving neuromorphic events and spikes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 2

[6] Burak Ercan, Onur Eker, Canberk Saglam, Aykut Erdem, and Erkut Erdem. HyperE2VID: Improving event-based video reconstruction via hypernetworks. IEEE Transactions on Image Processing, 2024. 1, 2

[7] Yang Feng, Hengyi Lv, Hailong Liu, Yisa Zhang, Yuyao Xiao, and Chengshan Han. Event density based denoising method for dynamic vision sensor. Applied Sciences, 10(6): 2024, 2020. 4, 5

[8] Daniel Gehrig and Davide Scaramuzza. Low-latency automotive vision with event cameras. Nature, 629(8014):1034â 1040, 2024. 1

[9] Haiqian Han, Jianing Li, Henglu Wei, and Xiangyang Ji. Event-3DGS: Event-based 3d reconstruction using 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:128139â128159, 2025. 4

[10] Jin Han, Yixin Yang, Peiqi Duan, Chu Zhou, Lei Ma, Chao Xu, Tiejun Huang, Imari Sato, and Boxin Shi. Hybrid high dynamic range imaging fusing neuromorphic and conventional images. IEEE Transactions on pattern analysis and machine intelligence, 45(7):8553â8565, 2023. 2

[11] Yuhuang Hu, Shih-Chii Liu, and Tobi Delbruck. v2e: From video frames to realistic dvs events. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1312â1321, 2021. 4, 7, 8

[12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3D Gaussian Splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4):139â1, 2023. 2, 3, 6, 7, 8

[13] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers. E-NeRF: Neural radiance fields from a moving event camera. IEEE Robotics and Automation Letters, 8(3):1587â1594, 2023. 3

[14] Jinxiu Liang, Yixin Yang, Boyu Li, Peiqi Duan, Yong Xu, and Boxin Shi. Coherent event guided low-light video enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10615â10625, 2023. 1, 2, 4, 6, 7, 8

[15] Haoyue Liu, Shihan Peng, Lin Zhu, Yi Chang, Hanyu Zhou, and Luxin Yan. Seeing motion at nighttime with an event camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25648â 25658, 2024. 1

[16] Xiaoning Liu, Zongwei Wu, Ao Li, Florin-Alexandru Vasluianu, Yulun Zhang, Shuhang Gu, Le Zhang, Ce Zhu, Radu Timofte, Zhi Jin, et al. NTIRE 2024 challenge on low light image enhancement: Methods and results. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6571â6594, 2024. 1

[17] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 2

[18] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P Srinivasan, and Jonathan T Barron. NeRF in the dark: High dynamic range view synthesis from noisy raw images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16190â16199, 2022. 2

[19] Federico Paredes-Valles and Guido CHE De Croon. Back to Â´ event basics: Self-supervised learning of image reconstruction for event cameras via photometric constancy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3446â3455, 2021. 2

[20] Henri Rebecq, Rene Ranftl, Vladlen Koltun, and Davide Â´ Scaramuzza. Events-To-Video: Bringing modern computer vision to event cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3857â3866, 2019. 1, 2, 4, 6, 7, 8

[21] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. EventNeRF: Neural radiance fields from a single colour event camera. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4992â5002, 2023. 2, 3, 4, 5, 6, 7, 8

[22] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In IEEE Conference on Computer Vision and Pattern Recognition, pages 4104â4113, 2016. 6

[23] Timo Stoffregen, Cedric Scheerlinck, Davide Scaramuzza, Tom Drummond, Nick Barnes, Lindsay Kleeman, and Robert Mahony. Reducing the sim-to-real gap for event cameras. In Proceedings of the European Conference on Computer Vision, pages 534â549. Springer, 2020. 8

[24] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Åukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4

[25] Jingqian Wu, Shuo Zhu, Chutian Wang, and Edmund Y. Lam. Ev-GS: Event-based gaussian splatting for efficient and accurate radiance field rendering. In 2024 IEEE International Workshop on Machine Learning for Signal Processing, pages 1â6, 2024. 2, 3, 4, 6, 7, 8

[26] Jingqian Wu, Shuo Zhu, Chutian Wang, Boxin Shi, and Edmund Y Lam. SweepEvGS: Event-based 3D gaussian splatting for macro and micro radiance field rendering from a single sweep. IEEE Transactions on Circuits and Systems for Video technology, 2025. 7

[27] Tianyi Xiong, Jiayi Wu, Botao He, Cornelia Fermuller, Yiannis Aloimonos, Heng Huang, and Christopher Metzler. Event3DGS: Event-based 3d gaussian splatting for highspeed robot egomotion. In 8th Annual Conference on Robot Learning, 2024. 2, 3

[28] Runzhao Yang, Tingxiong Xiao, Yuxiao Cheng, Qianni Cao, Jinyuan Qu, Jinli Suo, and Qionghai Dai. SCI: A spectrum concentrated implicit neural compression for biomedical data. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4774â4782, 2023. 4

[29] Yixin Yang, Jinxiu Liang, Bohan Yu, Yan Chen, Jimmy S Ren, and Boxin Shi. Latency correction for event-guided deblurring and frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24977â24986, 2024. 1

[30] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Oztireli, Â¨ and Olga Sorkine-Hornung. Differentiable surface splatting for point-based geometry processing. ACM Transactions on Graphics, 38(6):1â14, 2019. 3

[31] Wangbo Yu, Chaoran Feng, Jiye Tang, Xu Jia, Li Yuan, and Yonghong Tian. EvaGaussians: Event stream assisted gaussian splatting from blurry images. arXiv preprint arXiv:2405.20224, 2024. 3

[32] Sohaib Zahid, Viktor Rudnev, Eddy Ilg, and Vladislav Golyanik. E-3dgs: Event-based novel view rendering of large-scale scenes using 3d gaussian splatting. arXiv preprint arXiv:2502.10827, 2025. 3

[33] Pei Zhang, Haosen Liu, Zhou Ge, Chutian Wang, and Edmund Y. Lam. Neuromorphic imaging with joint image deblurring and event denoising. IEEE Transactions on Image Processing, 33:2318â2333, 2024. 2

[34] Song Zhang, Yu Zhang, Zhe Jiang, Dongqing Zou, Jimmy Ren, and Bin Zhou. Learning to see in the dark with events.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 666â682. Springer, 2020. 2

[35] Tianyi Zhang, Kaining Huang, Weiming Zhi, and Matthew Johnson-Roberson. DarkGS: Learning neural illumination and 3d gaussians relighting for robotic exploration in the dark. arXiv preprint arXiv:2403.10814, 2024. 2

[36] Shuo Zhu, Zhou Ge, Chutian Wang, Jing Han, and Edmund Y Lam. Efficient non-line-of-sight tracking with computational neuromorphic imaging. Optics Letters, 49(13): 3584â3587, 2024. 2

[37] Shuo Zhu, Chutian Wang, Haosen Liu, Pei Zhang, and Edmund Y Lam. Computational neuromorphic imaging: Principles and applications. In Computational Optical Imaging and Artificial Intelligence in Biomedical Sciences, pages 4â 10, 2024. 1

[38] Yi-Fan Zuo, Jiaqi Yang, Jiaben Chen, Xia Wang, Yifu Wang, and Laurent Kneip. DEVO: Depth-event camera visual odometry in challenging conditions. In 2022 International Conference on Robotics and Automation, pages 2179â2185, 2022. 1

[39] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Surface splatting. In Conference on Computer Graphics and Interactive Techniques, pages 371â378, 2001. 3