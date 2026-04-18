# EGS-SLAM: RGB-D Gaussian Splatting SLAM with Events

Siyu Chenâ, Shenghai Yuanâ, Thien-Minh Nguyen,

Zhuyu Huang, Chenyang Shi, Jin Jing, Lihua Xie Fellow, IEEE

AbstractâGaussian Splatting SLAM (GS-SLAM) offers a notable improvement over traditional SLAM methods, in enabling photorealistic 3D reconstruction that conventional approaches often struggle to achieve. However, existing GS-SLAM systems perform poorly under persistent and severe motion blur commonly encountered in real-world scenarios, leading to significantly degraded tracking accuracy and compromised 3D reconstruction quality. To address this limitation, we propose EGS-SLAM, a novel GS-SLAM framework that fuses event data with RGB-D inputs to simultaneously reduce motion blur in images and compensate for the sparse, discrete nature of event streams, enabling robust tracking and high-fidelity 3DGS reconstruction. Specifically, our system explicitly models the cameraâs continuous trajectory during exposure, supporting event and blur-aware tracking and mapping on a unified 3DGS scene. Furthermore, we introduce a learnable camera response function to align the dynamic ranges of events and images, along with a noevent loss to suppress ringing artifacts during reconstruction. We validate our approach on a new dataset comprising synthetic and real-world sequences with significant motion blur. Extensive experimental results demonstrate that EGS-SLAM consistently outperforms existing GS-SLAM systems in both trajectory accuracy and photorealistic 3DGS reconstruction. The source code will be available at https://github.com/Chensiyu00/EGS-SLAM.

Index TermsâGaussian Splatting, SLAM, Event Camera.

## I. INTRODUCTION

Simultaneous Localization and Mapping (SLAM) is fundamental to robotic autonomy, enabling agents to estimate their pose and build environmental maps in unknown settings [1]â [5]. While most visual SLAM systems [6]â[9] achieve high localization accuracy, they struggle to reconstruct detailed geometry and photorealistic appearance. Recently, 3D Gaussian Splatting (3DGS) has been introduced into SLAM [10]â[13], offering explicit scene representations with real-time rendering and richer reconstructions. However, these methods typically assume high-quality, blur-free images as input.

Nevertheless, under fast or continuous motion, conventional cameras often produce motion-blurred frames, leading to the loss of critical visual details. This degradation adversely affects both camera tracking and scene reconstruction, limiting the performance of 3DGS-based SLAM systems. Although existing SLAM approaches [14], [15] attempt to address this issue through image-only deblurring techniques, they remain ineffective under sustained or strong blur, as they fail to recover sufficient detail for reliable operation. Consequently, current high-accuracy, photorealistic SLAM systems struggle to operate reliably in environments affected by motion blur.

To address these limitations, incorporating additional sensing modalities is a promising direction. Event cameras offer a compelling alternative, as their microsecond-level asynchronous brightness sensing inherently avoids motion blur. In existing works [16], [17], event data has been employed in SLAM systems for robust camera tracking and in 3D Gaussian reconstruction [18]â[20] to recover sharp and detailed scenes under severe motion blur. To date, no existing work has integrated event information into the GS-SLAM framework to simultaneously enable online tracking under blurred conditions and the construction of high-fidelity 3D Gaussian maps.

Integrating event data into a GS-SLAM framework faces two main challenges. First, event streams are sparse and respond to per-pixel brightness changes, while image frames are captured at discrete intervals based on exposure. This inconsistency makes it challenging to fuse events and images for joint tracking and mapping within the 3DGS. Second, events and images differ significantly in their dynamic range: event data has inherently high dynamic range (HDR), whereas standard images have low dynamic range (LDR) and are constrained by exposure settings. This mismatch complicates fusion and the construction of a unified 3D Gaussian Splatting representation that effectively leverages both modalities.

To overcome these challenges, we propose an Event RGB-D GS-SLAM framework that uses event, image and depth within the cameraâs exposure time for accurate tracking and mapping. We model the continuous camera trajectory during each exposure interval and use it to render blur-aware images and corresponding event maps from 3DGS. These rendered signals are compared against the accumulated event maps, obtained by integrating the raw event stream within the same interval, and the image for temporally consistent tracking and mapping. To bridge the dynamic range gap between HDR events and LDR images, we introduce a learnable Camera Response Function (CRF) that transforms both modalities into a shared intensity space. This unified design enables our system to robustly operate under motion blur, leveraging the complementary advantages of both sensing modalities.

Our contributions can be summarized as:

â¢ To our knowledge, we present the first E-RGB-D GS-SLAM framework that incorporates event data alongside RGB and depth information. By jointly modeling these modalities within the cameraâs exposure time, our method enables robust tracking and mapping under severe blur.

â¢ We design an event-aided tracker and mapper for GS-SLAM that operate on blurry images, events, and depth to achieve accurate tracking and high-quality mapping. In the tracker and mapper, we incorporate a learnable CRF to align HDR events with LDR images and introduce a no-event loss to suppress ringing artifacts.

<!-- image-->  
Fig. 1: The pipeline of Gaussian Splatting SLAM with Event. Our system integrates eventâimage-depth tracking and mapping within a unified 3DGS map. For each frame, the pose is estimated by jointly rendering events, image and depth. Once converged, the mapping module updates the keyframe window and the 3DGS map if selected.

â¢ We construct a dataset containing both synthetic and realworld sequences with challenging motion blur. Extensive experiments show that our method outperforms existing GS-SLAM and classical baselines, both in camera localization and in reconstructing high-fidelity 3DGS.

## II. RELATED WORKS

This section presents deblur GS/NeRF, event-based GS/NeRF, and GS-SLAM which are relevant to this work.

Deblur GS/NeRF A line of research tackles 3D GS/NeRF reconstruction from motion-blurred images. Deblur-NeRF [21] introduces a deformable sparse kernel to recover sharp NeRFs from blurry inputs. BAD-NeRF [22] models dynamic blur trajectories to reconstruct clean scenes, and BAD-GS [23] extends this idea to 3DGS, yielding faster training and rendering while preserving detail. Seiskari et al. [24] incorporate velocity from visual-inertial odometry to mitigate both motion-blur and rolling-shutter artifacts, whereas BARD-GS [25] jointly models camera and object motion for dynamic scenes. Despite this progress, purely image-based methods still falter under extreme or persistent blur and all of these methods are offline.

Event-based GS/NeRF. Event cameras provide asynchronous, blur-free measurements that are well suited for 3D reconstruction. E2NeRF [26] couples blur- and eventrendering losses to obtain sharp NeRFs from heavily blurred images. Ev-DeblurNeRF [27] learns an event-to-pixel response to denoise events and boost reconstruction quality, while E-NeRF [28] achieves event-only NeRF from a single sensor. Event3DGS [19] combines events with blur modeling for crisp 3DGS results, and IncEventGS [29] attains high-quality, posefree 3DGS using events alone. However, all of these methods (except IncEventGS) require offline initialization of camera poses via structure-from-motion (SfM) before optimizing the neural or Gaussian scene representation and therefore cannot support online reconstruction and localization.

GS-SLAM Integrating 3DGS into SLAM has produced more photorealistic and efficient maps. GS-SLAM [11],

SplaTAM [12], and MonoGS [10] unify tracking and mapping within a single 3DGS representation, while PhotoSLAM [13] reconstructs 3DGS from ORB-SLAM poses. These systems perform well on sharp inputs but severely degrade under motion blur. MBA-SLAM [15] embeds the camera-imaging process to handle blurred frames, and I2-SLAM [14] incorporates the camera-response function for improved mapping. Nonetheless, all of these are image-only methods, which remain vulnerable to continuous and intense motion blur due to their reliance on blurred frame observations.

To our knowledge, only one NeRF-based SLAM [30] considers combining the event and frames to achieve online tracking and mapping. Crucially, no prior work has yet integrated events and RGB images into GS-SLAM that delivers online accurate tracking and high-quality 3DGS mapping.

## III. METHOD

Our SLAM system is built around three tightly integrated components: (i) a unified 3DGS map, (ii) an event-image tracking process, and (iii) an incremental event-image mapping process, as illustrated in Fig. 1. The 3DGS map serves as the only scene representation for tracking and mapping. For each incoming frame, the tracking module jointly leverages the event stream, depth and image for tracking, estimating a continuous camera trajectory by rendering both signals from the 3DGS map over the frameâs exposure interval. Once the pose of the incoming frame has converged, the mapping module evaluates its overlap with the latest keyframes to decide whether it should be selected as a new keyframe. If selected, it updates and maintains the current keyframe window and then updates the 3DGS map accordingly.

## A. 3D Gaussian Splatting Map Representation

Following [31], a 3D scene is represented as a set of 3D Gaussians G, each characterized by a mean $\mu _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ , a covariance matrix $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 } = R _ { i } \bar { s _ { i } } s _ { i } ^ { T } R _ { i } ^ { T }$ represented by a rotation matrix $R \in \mathrm { S O ( 3 ) }$ given by a unit quaternion $q _ { i }$ and a scale $s _ { i } \in \mathbb { R } ^ { 3 }$ , an opacity $o _ { i } \in \mathbb { R }$ , and a color $\boldsymbol { c } _ { i } \in \mathbb { R } ^ { 3 }$ . For efficiency, the spherical harmonic representation is omitted, as done in [10]. To render an image from these 3D Gaussians, we first project them onto the image plane using the camera pose:

$$
\begin{array} { r } { \hat { \pmb { \mu } } _ { i } = \pi ( T _ { c w } , \pmb { \mu } _ { i } ) ; \hat { \Sigma } _ { i } = J R _ { c w } \Sigma _ { i } R _ { c w } ^ { T } J ^ { T } , } \end{array}\tag{1}
$$

where $\hat { \mu } _ { i }$ is the projected mean, $\pi ( \cdot )$ represents the projection function, and $T _ { c w }$ is the world-to-camera transformation. $\hat { \Sigma } _ { i }$ is the projected covariance. $J$ is the Jacobian of the affine transformation, and $R _ { c w }$ is the rotation matrix of $T _ { c w }$ . The rendered color $\pmb { \mathcal { T } } ( \pmb { u } )$ and depth $\pmb { \mathcal { D } } ( \pmb { u } )$ of the pixel at the u position are then computed by the Î±-blending of all reprojected Gaussians overlapping on the pixel, sorted by the depth:

$$
\pmb { \mathcal { Z } } ( \pmb { u } ) = \sum _ { i \in \mathcal { N } } \pmb { c _ { i } \alpha _ { i } } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ; \pmb { \mathcal { D } } ( \pmb { u } ) = \sum _ { i \in \mathcal { N } } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $d _ { i }$ denotes the projected depth of the center of the i-th 3D Gaussian and $\grave { \alpha _ { i } } ~ = ~ o _ { i } e ^ { - \frac { 1 } { 2 } ( \pmb { u } - \hat { \mu } _ { i } ) ^ { T } \hat { \Sigma } _ { i } ^ { - 1 } ( \pmb { u } - \hat { \mu } _ { i } ) }$ . For simplicity, we denote $T _ { c w }$ as $\mathbf { T }$ in the rest of the paper.

## B. Camera Motion Modeling during Exposure

To model the cameraâs continuous motion trajectory within a single exposure duration, we assume the motion is linear and express the pose at any instant $\eta \in [ 0 , \tau ]$ by interpolating between the start pose $\mathbf { T } _ { 0 }$ and end pose $\mathbf { T } _ { \boldsymbol { \tau } }$

$$
\pmb { T } ( \eta ) = \left[ \begin{array} { c c } { \mathrm { { S l e r p } } ( R _ { 0 } , R _ { \tau } , \frac { \eta } { \tau } ) } & { ( 1 - \frac { \eta } { \tau } ) \mathbf { t } _ { 0 } + \frac { \eta } { \tau } \mathbf { t } _ { \tau } } \\ { \mathbf { 0 } } & { 1 } \end{array} \right] ,\tag{3}
$$

where $R _ { 0 } , R _ { \tau } \in \mathrm { S O } ( 3 )$ and $\mathbf { t } _ { 0 } , \mathbf { t } _ { \tau } \in \mathbb { R } ^ { 3 }$ are the rotation and translation components of $\mathbf { T } _ { 0 }$ and $\mathbf { T } _ { \boldsymbol { \tau } }$ , representing the camera poses at the start $( \eta = 0 )$ and end $( \eta = \tau )$ of exposure. Slerp(Â·) denotes spherical linear interpolation in $\mathrm { S O ( 3 ) }$

## C. Blur-Aware Tracking with Event

The tracking module in our SLAM system estimates the optimized camera poses $( T _ { 0 } ^ { * }$ and $T _ { \tau } ^ { * } )$ within a pre-built 3D Gaussian map by jointly minimizing photometric, depth, and event residuals between rendered outputs and sensor observations when a new frame arrives. The rendering processes for both the image and the event are illustrated in Fig. 2.

Photometric Loss We adopt the physical imaging process formulation addressing the limitations of conventional static exposure assumptions. The static model ignores how intensity changes over time during capture, which directly causes motion blur. Following [14], [19], [23], digital camera imaging fundamentally involves two sequential stages: light capturing during sensor exposure with continuous photon accumulation over time, followed by photoelectric conversion that transforms the collected light into measurable electrical signals. This physical process is mathematically modeled as temporal integration of simulated latent sharp frames over the exposure duration and can be approximated by the discrete model as:

$$
\tilde { I } ( \boldsymbol { u } ) = \int _ { 0 } ^ { \tau } \mathcal { I } ( \mathbf { T } ( \eta ) , \boldsymbol { u } ) d \eta \approx \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathcal { I } ( \mathbf { T } ( \eta _ { k } ) , \boldsymbol { u } ) ,\tag{4}
$$

where $\tilde { I } ( u )$ is integrated HDR image over the exposure time. Ï is the exposure time, and $\mathcal { T } ( \mathbf { T } ( \eta _ { k } ) , \pmb { u } )$ specifies the instantaneous latent sharp image under pose $\mathbf { T } ( \eta _ { k } )$ derived by Eq. (1) and Eq. (2). The timestamps $\eta _ { k }$ are evenly divided based on the number of events, ensuring that each trajectory segment covers approximately the same distance. Since event data inherently possesses HDR characteristics while images collected by the normal camera are represented in LDR, the scene representation is HDR. Inspired by [30], we then introduce a Camera Response Function (CRF) to map the rendered HDR imagery into LDR space, thereby better preserving the joint characteristics of event data and image in tracking and mapping. Inspired by [14], [32], we obtain the synthesized LDR image $\hat { \pmb { I } } ( { \pmb u } )$ by applying a trainable CRF to the rendered HDR image $\tilde { I } ( u )$ :

<!-- image-->  
Fig. 2: Illustration of blur-aware image and event map rendering. Rendered images along the trajectory are aggregated via a CRF for the final image. Event maps are generated by computing logarithmic brightness differences between consecutive rendered frames.

$$
\begin{array} { r l } & { \hat { \cal I } ( u ) = \mathrm { C R F } _ { \mathrm { l e a k y } } ( \tilde { \cal I } ( u ) ) } \\ & { \quad \quad = \left\{ \begin{array} { l l } { \alpha \tilde { \cal I } ( u ) , } & { \mathrm { i f ~ } \tilde { \cal I } ( u ) < 0 } \\ { \mathrm { I n t e r p } ( \tilde { \cal I } ( u ) , { \cal Q } ) } & { \mathrm { i f ~ } 0 \le \tilde { \cal I } ( u ) \le 1 , } \\ { - \frac { \alpha } { \sqrt { \tilde { \cal I } ( u ) } } + \alpha + 1 , } & { \mathrm { i f ~ } \tilde { \cal I } ( u ) > 1 } \end{array} \right. } \end{array}\tag{5}
$$

where Interp(Â·) represents the linear interpolation function, $Q$ means the trainable control nodes, and Î± is set as 0.01 in our system. We uniformly fix N output levels in the LDR space and associate each with a trainable HDR intensity, forming the control nodes $Q$ and shared all frames. Given an HDR input $\tilde { I } ( u )$ , its corresponding LDR output is obtained by linearly interpolating between adjacent control nodes, to approximate a differentiable CRF. The photometric loss $L _ { I }$ is then designed:

$$
L _ { I } = \left. \hat { I } - I _ { o b s } \right. _ { 1 } ,\tag{6}
$$

where $I _ { o b s }$ denote the LDR image acquired by the image camera, and $\left\| \cdot \right\| _ { 1 }$ 1 represents the $L _ { 1 }$ -norm operator.

Event Loss The event stream $\mathbb { E } = \{ e _ { m } \}$ asynchronously captures spatiotemporal brightness changes, where each event $e _ { m } = \{ \pmb { u } _ { m } , t _ { m } , p _ { m } \}$ includes a pixel location $\mathbf { \boldsymbol { u } } _ { m }$ , timestamp $t _ { m } ,$ and polarity $p _ { m } \in \{ + 1 , - 1 \}$ . An event is triggered when the logarithmic brightness change at pixel $\mathbf { \Delta } \pmb { u } _ { m }$ exceeds a predefined threshold $\theta \ > \ 0$ , i.e., $\mid \log ( L ( x , t _ { i } + \Delta t ) ) \ -$ $\log ( L ( x , t _ { i } ) ) \vert > \theta$ . Due to their discrete and sparse nature, raw events are not directly suitable for training 3DGS. To address this, we aggregate events by position and polarity over short time intervals to construct a dense event map:

<table><tr><td>Method</td><td>Metric</td><td>room0</td><td>room1</td><td>room2</td><td>office0</td><td>office1</td><td>office3</td><td>office4</td><td>Avg.</td><td>Rendering FPS</td></tr><tr><td rowspan="3">PhotoSLAM [13]</td><td>PSNR[dB]â</td><td>18.61</td><td>19.66</td><td></td><td>23.85</td><td></td><td>17.74</td><td>14.98</td><td></td><td rowspan="3">1075.8</td></tr><tr><td>SSIMâ</td><td>0.569</td><td>0.658</td><td></td><td>0.710</td><td></td><td>0.666</td><td>0.616</td><td></td></tr><tr><td>LPIPSâ</td><td>0.390</td><td>0.385</td><td></td><td>0.347</td><td></td><td>0.321</td><td>0.456</td><td></td></tr><tr><td rowspan="3">MonoGS [10]</td><td>PSNR[dB]â</td><td>20.47</td><td>21.97</td><td>24.05</td><td>26.75</td><td>26.76</td><td>21.41</td><td>21.79</td><td>23.32</td><td></td></tr><tr><td>SSIMâ</td><td>0.632</td><td>0.697</td><td>0.768</td><td>0.789</td><td>0.830</td><td>0.749</td><td>0.769</td><td>0.748</td><td>1113.9</td></tr><tr><td>LPIPSâ</td><td>0.454</td><td>0.451</td><td>0.335</td><td>0.365</td><td>0.335</td><td>0.267</td><td>0.391</td><td>0.371</td><td></td></tr><tr><td rowspan="3">MonoGS [10] (Refined)</td><td>PSNR[dB]â</td><td>21.20</td><td>22.66</td><td>24.43</td><td>26.97</td><td>27.12</td><td>21.65</td><td>23.31</td><td>23.91</td><td></td></tr><tr><td>SSIMâ</td><td>0.651</td><td>0.709</td><td>0.777</td><td>0.798</td><td>0.836</td><td>0.762</td><td>0.798</td><td>0.762</td><td>1052.7</td></tr><tr><td>LPIPSâ</td><td>0.383</td><td>0.388</td><td>0.307</td><td>0.320</td><td>0.299</td><td>0.249</td><td>0.328</td><td>0.325</td><td></td></tr><tr><td rowspan="3">Ours</td><td>PSNR[dB]â</td><td>24.06</td><td>26.30</td><td>27.61</td><td>31.72</td><td>33.38</td><td>26.50</td><td>23.79</td><td>27.62</td><td></td></tr><tr><td>SSIM â</td><td>0.744</td><td>0.783</td><td>0.838</td><td>0.885</td><td>0.927</td><td>0.846</td><td>0.806</td><td>0.833</td><td>1134.2</td></tr><tr><td>LPIPSâ</td><td>0.229</td><td>0.256</td><td>0.172</td><td>0.142</td><td>0.123</td><td>0.113</td><td>0.242</td><td>0.182</td><td></td></tr></table>

TABLE I: Rendering performance comparison of RGB-D SLAM methods on EventReplica. It should be noted that, although Photoslam may experience tracking loss in some sequences, it has the ability to reinitialize itself. The results presented here incorporate the mapping outcomes subsequent to the reinitialization process. â means the reinitialization attempt did not succeed. Our method outperforms the existing methods.

$$
E _ { k } ( \boldsymbol { u } ) = \sum _ { e _ { m } \in \mathcal { E } _ { k } } p _ { m } ,\tag{7}
$$

where $\mathscr { E } _ { k } = \left\{ e _ { m } ~ | ~ \pmb { u } _ { m } = \pmb { u } , \eta _ { k - 1 } < t _ { m } < \eta _ { k } \right\}$ is the subset of events within the n-th time window. Since event maps cannot be directly rendered from the Gaussian scene representation, we simulate them using the event generation model [19], [33] by computing the difference between the logarithmic brightness values of two consecutive rendered frames:

$$
\hat { E } _ { k } ( \pmb { u } ) = \mathrm { l o g } ( \hat { B } ( \pmb { T } ( \eta _ { k } ) , \pmb { u } ) ) - \mathrm { l o g } ( \hat { B } ( \pmb { T } ( \eta _ { k - 1 } ) , \pmb { u } ) ) ,\tag{8}
$$

where $\hat { B }$ denotes the grayscale brightness obtained from the rendered RGB image ËI via the BT.601 luma transform [34].

The event loss is then defined as the L1-distance between the accumulated event map $\scriptstyle { E _ { k } }$ and rendered event maps $\hat { E } \colon$

$$
L _ { H E } = \frac { 1 } { K } \sum _ { n = 0 } ^ { K - 1 } \sum _ { { E } _ { k } ( \pmb { u } ) \neq 0 } \Big | \Big | \theta \cdot { E } _ { k } ( \pmb { u } ) - \hat { { E } } _ { k } ( \pmb { u } ) \Big | \Big | _ { 1 } .\tag{9}
$$

To further leverage event-based supervision, inspired by [28], we define a no-event loss to penalize predicted undesired photometric changes at locations with no events:

$$
L _ { N E } = \frac { 1 } { K } \sum _ { n = 0 } ^ { K - 1 } \sum _ { E _ { k } ( \pmb { u } ) = 0 } \left\| \hat { \pmb { E } } _ { k } ( \pmb { u } ) \right\| _ { 1 } .\tag{10}
$$

Unlike prior work [28], we assume that in the absence of events, no photometric change has occurred. This assumption helps enforce temporal consistency and accelerates convergence in our GS-SLAM. The final event loss is designed as:

$$
L _ { E } = L _ { H E } + \lambda _ { N E } L _ { N E } ,\tag{11}
$$

where $\lambda _ { N E }$ is the weighting factor for the no-event loss.

Depth Loss The observed depth cannot be directly aligned with motion-blurred frames. Inspired by [14], we define the depth loss as the minimum discrepancy between the rendered depth D and the sensor depth $D _ { \mathrm { o b s } }$ during the exposure:

$$
L _ { D } = \operatorname* { m i n } _ { k } \left\| D _ { \mathrm { o b s } } - \mathcal { D } ( \mathbf { \cal { T } } ( \eta _ { k } ) ) \right\| _ { 1 } .\tag{12}
$$

TABLE II: Comparison of tracking results ATE (cm) on EventReplica. L means the method loses tracking in the sequence. \* indicates odometry-only methods that do not support consistent or photorealistic 3D scene reconstruction.
<table><tr><td>Scenes</td><td>RampVO* [35]</td><td>ORB-SLAM2* [6]</td><td>Photo-SLAM [13]</td><td>MonoGS [10]</td><td>Ours -</td></tr><tr><td>room0</td><td>4.27</td><td>5.57</td><td>6.64</td><td>12.76</td><td>I 6.70</td></tr><tr><td>room1</td><td>13.11</td><td>L</td><td>L</td><td>8.45</td><td>- 3.26</td></tr><tr><td>room2</td><td>3.43</td><td>L</td><td>L</td><td>3.64</td><td>I 3.16</td></tr><tr><td>office0</td><td>4.41</td><td>L</td><td>L</td><td>7.44</td><td>I 3.47</td></tr><tr><td>office1</td><td>3.09</td><td>L</td><td>L</td><td>7.78</td><td>I 3.53</td></tr><tr><td>office3</td><td>4.33</td><td>6.48</td><td>6.78</td><td>7.91</td><td>4.71</td></tr><tr><td>office4</td><td>5.87</td><td>L</td><td>L</td><td>16.55</td><td>| 12.41</td></tr><tr><td>Avg.</td><td>5.50</td><td>â¢</td><td>-</td><td>9.22</td><td>I 5.32</td></tr></table>

This formulation encourages alignment between the depth map and the latent sharp image within the exposure window.

Pose Optimization To estimate the continuous trajectory during the current frameâs exposure, we optimize the control nodes while keeping the 3D Gaussian map G fixed. The optimized control nodes $T _ { 0 } ^ { * }$ and $T _ { \tau } ^ { * }$ are obtained by solving the following objective through iterative optimization:

$$
T _ { 0 } ^ { * } , T _ { \tau } ^ { * } = \arg \operatorname* { m i n } _ { T _ { 0 } , T _ { \tau } } \left( \lambda _ { E } L _ { E } + \lambda _ { I D } \left( \lambda _ { I } L _ { I } + \lambda _ { D } L _ { D } \right) \right) ,\tag{13}
$$

where $\lambda _ { E } , \lambda _ { I }$ , and $\lambda _ { D }$ are weights that control the individual contributions of each term, and $\lambda _ { I D }$ balances the combined image and depth terms relative to the event term.

## D. Mapping Process

The mapping process comprises two core components: keyframe management and 3DGS map updating.

Keyframe Management After tracking converges, we follow [10] for keyframe selection and keyframe window maintenance. A keyframe is inserted when the overlapped visible 3D Gaussians (IoU) with the latest keyframe falls below a threshold, promoting viewpoint diversity or camera movement exceeds a threshold scaled by the current frameâs average depth. To maintain a bounded keyframe set $\mathcal { W } _ { k }$ for mapping for computational efficiency, we remove historical keyframes whose overlap with the new keyframe is below another lower threshold to preserve local relevance. If no keyframes are removed, we adopt the strategy from [36] to remove the most redundant keyframe to keep the window size fixed.

<!-- image-->  
Fig. 3: Comparison of rendering quality on the EventReplica dataset. Our method can achieve sharper reconstructions.

3DGS Map Updating After the creation of a novel keyframe, new Gaussians are instantiated and incorporated into the established 3DGS representation. Following [10], the mean values $\pmb { \mu }$ of the new Gaussians are initialized via depth backprojection leveraging estimated camera poses. The remaining parameters (including rotation $q ,$ scale $s ,$ opacity $^ { O , }$ and color c) are initialized according to the strategy proposed in [31]. We then need to optimize the parameters. Subsequent optimization of 3DGSs incorporates an isotropic regularization term [10] to mitigate artifacts caused by skinny Gaussians:

$$
L _ { i s o } = \frac { 1 } { | \mathcal { G } | } \sum _ { i = 1 } ^ { | \mathcal { G } | } \left\| s _ { i } - \mathbf { 1 } \cdot \overline { { s } } _ { i } \right\| ,\tag{14}
$$

where $\overline { { s } } _ { i }$ denotes the mean of the scale $s _ { i }$ . Two randomly selected historical keyframes, $\mathbf { w } _ { r } ^ { 1 }$ and $\mathbf { w } _ { r } ^ { 2 }$ , are added to the current window $\mathcal { W } _ { c }$ to form the extended mapping set $\mathcal { W } ^ { \prime } =$ ${ \mathcal W } _ { c } \cup \{ { \bf w } _ { r } ^ { 1 } , { \bf w } _ { r } ^ { 2 } \}$ . The final mapping loss can be designed as:

$$
L _ { m a p } = \lambda _ { i s o } L _ { i s o } + \sum _ { w \in \mathcal { W } ^ { \prime } } ( \lambda _ { E } L _ { E } ^ { w } + \lambda _ { I D } ( \lambda _ { I } L _ { I } ^ { w } + \lambda _ { D } L _ { D } ^ { w } ) ) _ { , }\tag{15}
$$

where $L _ { I } ^ { w } , L _ { E } ^ { w } , L _ { D } ^ { w }$ are the photometric loss, the event loss, and the depth loss of the frame w and $\lambda _ { i s o }$ is taken as 10. It is worth noting that the no-event loss plays a crucial role during mapping, as it significantly reduces the ringing problemâan artifact commonly observed in deblurring methods. We jointly optimize the parameters of the Gaussians and finetune the camera poses of the latest k keyframes in a sliding window to improve consistency. After the optimization of mapping, we prune the Gaussians for mapping stability.

## IV. EXPERIMENTS

In this section, we first introduce the dataset we collected and used in our experimental setup. We then present qualitative visual comparisons with existing methods to demonstrate the superior performance of our approach in both tracking and mapping. Next, we provide additional visualizations of the rendered results, illustrating that our method outperforms existing approaches in reconstructing sharper and higherquality 3DGS scenes. Finally, we conduct an ablation study on the contributions of event information, the Camera Response Function, the no-event loss, and the integration of GS-SLAM with single-frame deblurring, demonstrating the importance of each component in the overall effectiveness of our system.

## A. Dataset

EventReplica We created a synthetic event dataset, EventReplica, by extending the Replica dataset from [37]. The original images were resized to 459Ã260 pixels and cropped to 448Ã256 pixels. Following [25], [26], [30], we used FILM [38] to generate intermediate frames, which were then converted to event streams using VID2E [39]. The physical motion of the frames is synthesized by summing all interpolated frames within the exposure time, with the final frame serving as the sharp ground truth and its depth map being directly adopted as the depth ground truth for the corresponding blurred frame.

DEVD Our data acquisition system consists of a DAVIS346 color event camera for capturing both events and images, a RealSense D435i depth camera for acquiring depth information, and the FZMotion Motion Capture System for providing ground-truth poses. Since both the D435i depth module and the motion capture system emit 850nm infrared lightâwhich introduces significant noise to the event cameraâwe installed an infrared-cut filter (transmitting only the 400-700nm visible spectrum) in front of the event sensor. Our dataset comprises four scenes: Mahjong, Mountain, Table, and Testbed.

## B. Implementation details

We conducted our experiments on an NVIDIA RTX 4080 GPU. The event threshold Î¸ was set to 0.25 for synthetic data and 0.3 for real data and $\lambda _ { N E } = 0 . 4$ . We combined RGB and depth losses with weights $\lambda _ { I } = 0 . 9$ and $\lambda _ { D } = 0 . 1$ . For the

<!-- image-->  
Fig. 4: Comparison of rendering quality on the DEVD dataset. Our method can achieve sharper reconstructions.

TABLE III: Comparison of tracking results ATE (cm) on DEVD. â means the occurrence of some frame drops in this sequence. \* indicates odometry-only methods that do not support consistent or photorealistic 3D scene reconstruction.
<table><tr><td>Scenes</td><td>RampVO* [35]</td><td>ORB-SLAM2* [6]</td><td>Photo-SLAM [13]</td><td>MonoGS [10]</td><td>|</td><td>Ours</td></tr><tr><td>Mahjong1</td><td>4.90</td><td>3.92x</td><td>4.34X</td><td>3.17</td><td>|</td><td>1.45</td></tr><tr><td>Mahjong2</td><td>3.77</td><td>3.72X</td><td>5.29</td><td>2.55</td><td>I |</td><td>1.19</td></tr><tr><td>Mountain1</td><td>1.96</td><td>3.47</td><td>4.50</td><td>4.30</td><td>|</td><td>1.18</td></tr><tr><td>Mountain2</td><td>1.18</td><td>4.85</td><td>4.98</td><td>3.52</td><td>|</td><td>1.70</td></tr><tr><td>Table1</td><td>4.00</td><td>10.27</td><td>25.29</td><td>6.73</td><td>|</td><td>2.97</td></tr><tr><td>Table2</td><td>4.04</td><td>4.41</td><td>4.21</td><td>10.77</td><td>| |</td><td>6.56</td></tr><tr><td>Testbed1</td><td>2.29</td><td>5.26</td><td>3.69</td><td>5.15</td><td>|</td><td>2.66</td></tr><tr><td>Testbed2</td><td>3.10</td><td>4.45</td><td>4.22</td><td>12.41</td><td>|</td><td>7.58</td></tr><tr><td>Avg.</td><td>3.16</td><td>5.04</td><td>7.06</td><td>6.07</td><td>â</td><td>3.16</td></tr></table>

synthetic dataset, we set $\lambda _ { E } = 0 . 0 5$ and $\lambda _ { D I } = 0 . 9 5 ;$ for the real dataset, $\lambda _ { E } = 0 . 1 5$ and $\lambda _ { D I } = 0 . 8 5 .$ A sliding window of size 10 was used for mapping, and the latest 5 frames were optimized in the backend. For fair comparison, we also set MonoGS to use the same window size. Other than that and the event-related parts, all settings are the same as MonoGS. Boldface indicates the best performing method and underline indicates the second best in all experimental tables.

## C. Quantitative Evaluation

In this section, we benchmark our method on EventReplica and DEVD, comparing it against existing GS-SLAM approaches, including Photo-SLAM [13], and MonoGS [10]. We further include a comparison with ORB-SLAM2 [6], a classical RGB-D SLAM system, and RampVO [35], the SOTA learning-based frame-event VO, to provide a comprehensive performance assessment. We use the single-thread mode of MonoGS [10] and evaluate Absolute Trajectory Error (ATE) over the entire trajectory, along with PSNR, SSIM, and LPIPS [10], [11] for reconstruction quality comparisons.

Evaluation on synthesis dataset: EventReplica Scene reconstruction quality comparisons are shown in Tab. I, where the rendered images are evaluated against sharp ground-truth references. It is worth noting that although PhotoSLAM suffers from frequent and severe tracking loss, it can reinitialize and continue reconstruction; the results we report include such reinitialized reconstructions. MonoGS and PhotoSLAM still exhibit substantial performance degradation when reconstructing from motion-blurred images and are generally unable to recover clean and photorealistic scene appearances. Even when refined through offline reconstruction, MonoGS consistently fails to restore sharp and detailed content. In contrast, our method can significantly outperform the original MonoGS. Specifically, the PSNR improves from 23.32dB to 27.62dB (+4.20dB), SSIM increases from 0.748 to 0.833 (+0.085), and LPIPS drops from 0.371 to 0.183 (â0.189). These improvements demonstrate the superior capability of our approach in producing sharp and clear 3DGS.

Trajectory comparisons are reported in Tab. II. Our method outperforms existing approaches in 6 out of 7 sequences, and We reduced the average error from 9.22cm to 5.32cm, achieving an improvement of approximately 42.23%. ORB-SLAM2 and PhotoSLAM experience significant tracking degradation due to motion blur, which hampers keypoint detection and causes frequent mismatches. MonoGS also shows a notable drop in tracking performance, as it relies on the assumption of blur-free inputs and struggles under blurred conditions. RampVO, with its strong learning-based trackers, achieves comparable tracking performance to our method; however, unlike RampVO, which only produces a sparse point cloud lacking photometric information, our method enables photorealistic and high-quality clear scene reconstruction.

Evaluation on real dataset: DEVD Since ground-truth clean images are unavailable for the real-world dataset, direct quantitative comparisons are not feasible. Therefore, we present qualitative visual comparisons in the subsequent section to demonstrate the effectiveness of our method in recovering sharp and high-quality scene appearances.

In Tab. III, our method outperforms the existing baselines on the tracking performance, achieving the best ATE in 7 out of 8 sequences as well as the lowest average error overall. Compared to MonoGS, our approach yields a substantial improvement in tracking accuracy, with an average ATE reduction of approximately 47% from 6.07cm to 3.16cm. Although ORB-SLAM2 and Photo-SLAM incorporate relocalization mechanisms to recover from tracking failures, they still suffer from overall inferior performance compared to ours. RampVO achieves comparable tracking performance, but can only reconstruct a sparse point cloud, while our method enables photorealistic and sharp scene reconstruction.

<table><tr><td rowspan="2">Event Tracking</td><td rowspan="2">Event Mapping</td><td colspan="4">Room0</td><td colspan="4">Room1</td><td colspan="4">Office1</td></tr><tr><td>ATE[cm]â</td><td>PSNR[dB]â</td><td>SSIMâ</td><td>LPIPSâ</td><td>ATE[cm]â</td><td>PSNR[dB]â</td><td>SSIMâ</td><td>LPIPSâ</td><td>ATE[cm]â</td><td>PSNR[dB]â</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Ã</td><td>Ã</td><td>11.61</td><td>19.99</td><td>0.618</td><td>0.321 I</td><td>6.93</td><td>22.28</td><td>0.704</td><td>0.348</td><td>7.07</td><td>28.13</td><td>0.837</td><td>0.198</td></tr><tr><td>Ã</td><td>â</td><td>11.47</td><td>19.88</td><td>0.617</td><td>0.310</td><td>6.84</td><td>21.81</td><td>0.688</td><td>0.346</td><td>8.02</td><td>27.35</td><td>0.827</td><td>0.199</td></tr><tr><td>â</td><td>Ã</td><td>6.45</td><td>23.14</td><td>0.714</td><td>0.255</td><td>4.75</td><td>25.85</td><td>0.768</td><td>0.281</td><td>3.60</td><td>32.39</td><td>0.914</td><td>0.160</td></tr><tr><td>â</td><td>â</td><td>6.70</td><td>24.06</td><td>0.744</td><td>0.229</td><td>3.26</td><td>26.30</td><td>0.83</td><td>0.256</td><td>3.53</td><td>33.38</td><td>0.927</td><td>0.123</td></tr></table>

TABLE IV: The ablation study ATE(cm) analyzing the impact of event information on EventReplica.

TABLE V: Ablation study on ATE (cm) evaluating the impact of event information on the DEVD.
<table><tr><td>Event Tracking</td><td>Event Mapping</td><td>Mahjong1</td><td>Mahjong2</td><td>Mountain1</td><td>Table1</td></tr><tr><td>Ã</td><td>Ã</td><td>12.74</td><td>18.59</td><td>2.66</td><td>5.29</td></tr><tr><td>Ã</td><td>â</td><td>14.77</td><td>6.54</td><td>2.56</td><td>5.18</td></tr><tr><td>â</td><td>Ã</td><td>2.08</td><td>1.63</td><td>1.75</td><td>4.01</td></tr><tr><td>â</td><td>â</td><td>1.45</td><td>1.19</td><td>1.18</td><td>2.97</td></tr></table>

TABLE VI: The ablation study on ATE (cm) analyzing the effect of the CRF.
<table><tr><td>Settings</td><td>Mahjong1</td><td>Mahjong2</td><td>Mountain1</td><td>Tablel</td></tr><tr><td>w/o CRF</td><td>1.49</td><td>1.21</td><td>1.24</td><td>3.28</td></tr><tr><td>w CRF (Ours)</td><td>1.45</td><td>1.19</td><td>1.18</td><td>2.97</td></tr></table>

Runtime analysis The runtime analysis is carried out on the office1 sequence of the EventReplica dataset, where our method runs at 0.55 FPS and MonoGS reaches 1.75 FPS. The difference mainly comes from our design choice to perform more rendering times in order to combine events and images for improved performance. Importantly, this design enables our method to achieve robust tracking under challenging motion blur and to reconstruct sharp, high-fidelity 3D scenesâcapabilities that MonoGS lacks despite its faster runtime. Overall, our approach prioritizes reconstruction quality and robustness, which are crucial for real-world deployment.

## D. Qualitative Evaluation

In Fig. 3 and Fig. 4, we compare the mapping results of our method with PhotoSLAM and MonoGS. As shown, our rendered outputs can recover clean 3D images from blurry inputs, significantly outperforming the previous methods.

## E. Ablation Study

Event Information We evaluate the contribution of event information to tracking and mapping in Tab. IV and Tab. V. We can see that when both event-based tracking and mapping are disabled, the system relies on multiple image renderings for direct tracking. This setup can be seen as a deblur-SLAM approach based solely on image and depth inputs [14], [15]. In such cases, the system struggles to recover sharp scene appearances from continuous motion blur and to achieve reliable 3D tracking. Although it may show marginal improvements over MonoGS, the overall performance remains limited.

<!-- image-->  
Fig. 5: The ablation study on the No-Event Loss. The noevent loss is highly effective in removing ringing artifacts that resemble ripple patterns.

TABLE VII: An ablation study on single-image deblurring.
<table><tr><td rowspan="2">Method</td><td colspan="3">Room0</td><td colspan="3">Room1</td></tr><tr><td>ATE[cm]â</td><td>PSNR[dB]â</td><td>LPIPSâ</td><td>ATE[cm]â</td><td>PSNR[dB]â</td><td>LPIPSâ</td></tr><tr><td>MonoGS</td><td>12.76</td><td>20.47</td><td>0.454</td><td>8.45</td><td>21.97</td><td>0.451</td></tr><tr><td>MonoGS+EDI</td><td>16.96</td><td>18.18</td><td>0.429</td><td>11.6</td><td>19.67</td><td>0.446</td></tr><tr><td>Ours</td><td>6.70</td><td>24.06</td><td>0.229</td><td>3.26</td><td>26.30</td><td>0.256</td></tr></table>

Enabling event-based mapping without incorporating event tracking does not yield performance gains. This is primarily due to the absence of a robust tracking mechanism, which leads to inaccurate pose estimates. These inaccuracies degrade mapping quality, which in turn further hinders tracking performanceâcreating a negative feedback loop that significantly impairs the systemâs overall effectiveness. In contrast, enabling event-based tracking without mapping results in a notable performance boost. This improvement stems from the precise pose estimation enabled by the event stream. With accurate poses, high-quality image reconstructions can be achieved using only the RGB frames and depth data. Our full model that employs both event-based tracking and mapping achieves the best overall performance, especially on real-world datasets.

No Event Loss We observe that in real-world scenarios where depth quality is limited, ringing artifactsâcharacterized by ripple-like distortionsâeasily appear in the rendered images. In Fig. 5, we compare the effectiveness of the no-event loss and find that it substantially suppress these artifacts.

Camera Response Function We evaluate the impact of the Camera Response Function (CRF) on the Majhong1, Mahjong2, Mountain1, and Table1 sequences. As shown in Tab. VI, the performance degrades when the CRF is not applied. This degradation is primarily due to the inherent difference in dynamic ranges between the event data and frame images in real-world datasets. Without CRF calibration, forcing the two modalities into a shared range leads to substantial inconsistencies. By introducing the CRF, we achieve better alignment between modalities, reducing these inconsistencies and enhancing overall system performance.

Single-Frame Deblur For single-frame deblurring, we incorporate a widely used method, EDI [40], into MonoGS [10]. As shown in Tab. VII, the results reveal that the performance degrades after applying EDI, even compared to the baseline without deblurring. This degradation arises from artifacts introduced by single-frame deblurring methods, which act as noise and adversely affect the tracking and mapping results.

## V. CONCLUSION

In this paper, we presented the first E-RGB-D Gaussian Splatting SLAM framework that integrates event data with image and depth inputs to enable robust tracking and highfidelity mapping under motion blur. By explicitly modeling the cameraâs continuous trajectory during exposure and introducing a learnable CRF, our method effectively bridges the temporal and dynamic range discrepancies between asynchronous event streams and conventional image frames. Additionally, we proposed a no-event loss to suppress ringing artifacts, further improving the reconstruction quality. Extensive evaluations on both synthetic and real-world datasets demonstrate that our approach consistently outperforms existing GS-SLAM baselines in terms of localization accuracy and 3D scene fidelity. As the current system relies on depth input for both tracking and mapping, future work will focus on extending the framework to monocular setups.

## REFERENCES

[1] J. Liu and G. Hu, âRelative localization estimation for multiple robots via the rotating ultra-wideband tag,â IEEE Robot. Autom. Lett., vol. 8, no. 7, pp. 4187â4194, 2023.

[2] S. Yuan, B. Lou, T.-M. Nguyen et al., âLarge-scale uwb anchor calibration and one-shot localization using gaussian process,â arXiv preprint arXiv:2412.16880, 2024.

[3] J. Li, X. Xu, J. Liu et al., âUa-mpc: Uncertainty-aware model predictive control for motorized lidar odometry,â IEEE Robot. Autom. Lett., 2025.

[4] C. Wang, D. Gao, K. Xu, J. Geng et al., âPyPose: A library for robot learning with physics-based optimization,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023.

[5] Z. Yang, K. Xu, S. Yuan, and L. Xie, âA fast and light-weight noniterative visual odometry with rgb-d cameras,â Unmanned Syst., vol. 13, no. 03, pp. 957â969, 2025.

[6] R. Mur-Artal and J. D. Tardos, âOrb-slam2: An open-source slam system Â´ for monocular, stereo, and rgb-d cameras,â IEEE Trans. Robot., vol. 33, no. 5, pp. 1255â1262, 2017.

[7] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez et al., âOrb-slam3: An accurate open-source library for visual, visualâinertial, and multimap slam,â IEEE Trans. Robot., vol. 37, no. 6, pp. 1874â1890, 2021.

[8] Z. Teed and J. Deng, âDroid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras,â Adv. Neural Inf. Process. Syst., vol. 34, pp. 16 558â16 569, 2021.

[9] S. Chen, K. Liu, C. Wang et al., âSalient sparse visual odometry with pose-only supervision,â IEEE Robot. Autom. Lett., 2024.

[10] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 18 039â18 048.

[11] C. Yan, D. Qu, D. Xu et al., âGs-slam: Dense visual slam with 3d gaussian splatting,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024.

[12] N. Keetha, J. Karhade, K. M. Jatavallabhula et al., âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 21 357â21 366.

[13] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Real-time simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 21 584â21 593.

[14] G. Bae, C. Choi, H. Heo, S. M. Kim, and Y. M. Kim, âI2-slam: Inverting imaging process for robust photorealistic dense slam,â in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 72â89.

[15] P. Wang, L. Zhao, Y. Zhang, S. Zhao, and P. Liu, âMba-slam: Motion blur aware dense visual slam with radiance fields representation,â arXiv preprint arXiv:2411.08279, 2024.

[16] W. Chamorro, J. Sola, and J. Andrade-Cetto, âEvent-based line slam in real-time,â IEEE Robot. Autom. Lett., vol. 7, no. 3, pp. 8146â8153, 2022.

[17] A. R. Vidal, H. Rebecq et al., âUltimate slam? combining events, images, and imu for robust visual slam in hdr and high-speed scenarios,â IEEE Robot. Autom. Lett., vol. 3, no. 2, pp. 994â1001, 2018.

[18] W. Yu, C. Feng, J. Tang et al., âEvagaussians: Event stream assisted gaussian splatting from blurry images,â arXiv preprint arXiv:2405.20224, 2024.

[19] T. Xiong, J. Wu, B. He et al., âEvent3dgs: Event-based 3d gaussian splatting for high-speed robot egomotion,â arXiv preprint arXiv:2406.02972, 2024.

[20] H. Han, J. Li, H. Wei, and X. Ji, âEvent-3dgs: Event-based 3d reconstruction using 3d gaussian splatting,â Adv. Neural Inf. Process. Syst., vol. 37, pp. 128 139â128 159, 2024.

[21] L. Ma, X. Li, J. Liao et al., âDeblur-nerf: Neural radiance fields from blurry images,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 12 861â12 870.

[22] P. Wang, L. Zhao, R. Ma, and P. Liu, âBad-nerf: Bundle adjusted deblur neural radiance fields,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 4170â4179.

[23] L. Zhao, P. Wang, and P. Liu, âBad-gaussians: Bundle adjusted deblur gaussian splatting,â in Proc. Eur. Conf. Comput. Vis., 2024.

[24] O. Seiskari, J. Ylilammi, V. Kaatrasalo et al., âGaussian splatting on the move: Blur and rolling shutter compensation for natural camera motion,â in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 160â177.

[25] Y. Lu, Y. Zhou, D. Liu et al., âBard-gs: Blur-aware reconstruction of dynamic scenes via gaussian splatting,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 16 532â16 542.

[26] Y. Qi, L. Zhu, Y. Zhang, and J. Li, âE2nerf: Event enhanced neural radiance fields from blurry images,â in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 13 254â13 264.

[27] M. Cannici and D. Scaramuzza, âMitigating motion blur in neural radiance fields with events and frames,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 9286â9296.

[28] S. Klenk, L. Koestler, D. Scaramuzza, and D. Cremers, âE-nerf: Neural radiance fields from a moving event camera,â IEEE Robot. Autom. Lett., vol. 8, no. 3, pp. 1587â1594, 2023.

[29] J. Huang, C. Dong, and P. Liu, âInceventgs: Pose-free gaussian splatting from a single event camera,â arXiv preprint arXiv:2410.08107, 2024.

[30] D. Qu, C. Yan, D. Wang, J. Yin et al., âImplicit event-rgbd neural slam,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 19 584â19 594.

[31] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, â3D Gaussian Splatting for Real-Time Radiance Field Rendering,â ACM Trans. Graph., vol. 42, no. 4, pp. 1â14, 2023.

[32] K. Jun-Seong, K. Yu-Ji, M. Ye-Bin, and T.-H. Oh, âHdr-plenoxels: Selfcalibrating high dynamic range radiance fields,â in Proc. Eur. Conf. Comput. Vis. Springer, 2022, pp. 384â401.

[33] H. Rebecq, D. Gehrig, and D. Scaramuzza, âEsim: an open event camera simulator,â in Conf. Robot Learn. (CoRL). PMLR, 2018, pp. 969â982.

[34] R. BT et al., âStudio encoding parameters of digital television for standard 4: 3 and wide-screen 16: 9 aspect ratios,â Int. Telecommun. Union (ITU), CCIR Rep., 2011.

[35] R. Pellerito, M. Cannici et al., âDeep visual odometry with events and frames,â in IEEE/RSJ Int. Conf. Intell. Robots Syst., June 2024.

[36] J. Engel, V. Koltun, and D. Cremers, âDirect sparse odometry,â IEEE Trans. Pattern Anal. Mach. Intell., vol. 40, no. 3, pp. 611â625, 2017.

[37] Z. Zhu, S. Peng, V. Larsson et al., âNice-slam: Neural implicit scalable encoding for slam,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 12 786â12 796.

[38] F. Reda, J. Kontkanen, E. Tabellion, D. Sun, C. Pantofaru, and B. Curless, âFilm: Frame interpolation for large motion,â in Proc. Eur. Conf. Comput. Vis. Springer, 2022, pp. 250â266.

[39] D. Gehrig, M. Gehrig, J. Hidalgo-Carrio, and D. Scaramuzza, âVideo Â´ to events: Recycling video datasets for event cameras,â in IEEE Conf. Comput. Vis. Pattern Recog., June 2020.

[40] L. Pan, C. Scheerlinck, X. Yu, R. Hartley, M. Liu, and Y. Dai, âBringing a blurry frame alive at high frame-rate with an event camera,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 6820â6829.