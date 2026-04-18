# EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering

Toshiya Yura1,2

toshiya.yura@sony.com

Ashkan Mirzaei2

ashkan@cs.toronto.edu

Igor Gilitschenski2

gilitschenski@cs.toronto.edu

1Sony Semiconductor Solutions Corporation, 2University of Toronto

## Abstract

We introduce a method for using event camera data in novel view synthesis via Gaussian Splatting. Event cameras offer exceptional temporal resolution and a high dynamic range. Leveraging these capabilities allows us to effectively address the novel view synthesis challenge in the presence of fast camera motion. For initialization of the optimization process, our approach uses prior knowledge encoded in an event-to-video model. We also use spline interpolation for obtaining high quality poses along the event camera trajectory. This enhances the reconstruction quality from fast-moving cameras while overcoming the computational limitations traditionally associated with event-based Neural Radiance Field (NeRF) methods. Our experimental evaluation demonstrates that our results achieve higher visual fidelity and better performance than existing eventbased NeRF approaches while being an order of magnitude faster to render.

## 1. Introduction

Event cameras are a vision modality that draws strong inspiration from biological vision and neuromorphic design. In these sensors, each pixel operates independently by triggering events upon a log-intensity change above a pre-defined threshold [16, 37]. This results in an asynchronous data stream as opposed to capturing absolute intensity values at fixed intervals in traditional cameras. Event cameras offer low latency, high temporal resolution, and a high dynamic range while also ensuring minimal energy consumption [8, 38]. Consequently, they are particularly suitable for dynamic and challenging lighting conditions, as well as environments with rapid camera motions where classical cameras may exhibit strong motion blur [38, 63, 64, 78]. These advantages have motivated the adoption of event cameras in a wide range of applications covering most of the traditional computer vision tasks [13, 15, 27, 50, 51].

Recently, novel view synthesis has seen significant advances. This progress has been particularly driven by the emergence and development of Neural Radiance Fields (NeRF) [1â3, 33, 46, 74]. Recent works have led to NeRF improvements in terms of rendering speed, and adoption of NeRFs in 3D editing and other downstream applications [9, 43, 47â49, 84]. 3D Gaussian Splatting (3DGS) [26, 69] achieves real-time rendering by replacing the underlying representation with 3D Gaussians and using fast differentiable rasterization. The substantial speed improvement of 3DGS is achieved without compromising the visual quality.

The success of NeRFs in view-synthesis has led to their adoption in applications with event-based cameras [25, 31, 42, 59]. Conventional NeRF approaches struggle with challenges, such as 3D reconstruction in environments with rapidly changing lighting conditions or high-speed objects due to RGB camera limitations. The integration of NeRFs with event-based cameras leverages their unique advantages and incorporates them into the NeRF framework to address the challenges that conventional NeRF approaches face.

However, one crucial area of research focuses on expediting the rendering processes to fully unlock the potential of event-based NeRFs. Leveraging the advantages of 3D Gaussian Splatting with events requires adapting the rasterization process to allow supervision with events or accumulated events rather than RGB images.

In summary, this paper contributes a method for 3D Gaussian Splatting from event data by leveraging the following technical ideas:

1) An integration of event accumulation with the optimization process of 3DGS.

2) An event-to-video guided Structure from Motion (SfM) approach for initializing the 3DGS optimization process.

3) Use of cubic spline trajectory interpolation to assign camera poses to events at high rates.

<!-- image-->  
Figure 1. EventSplat derives 3D representations of scenes in the form of 3D Gaussians from event data, enabling fast real-time renderings of scenes captured with event cameras via rasterization of 3D Gaussians. The use of event-based input is particularly advantageous when traditional RGB cameras fail due to various reasons including poor lighting conditions or motion blur from fast-moving cameras.

Our evaluations show that these design choices achieve state-of-the-art event-based novel view synthesis performance while benefiting from fast rendering speeds. Thus, our method represents an advancement in addressing the speed constraints associated with event-based NeRFs. It opens up new possibilities for real-time novel view synthesis for environments where traditional imaging techniques struggle.

## 2. Related Work

## 2.1. Event Cameras for Vision Applications

Motivated by the low energy consumption, high dynamic range, and high asynchronous temporal resolution of event cameras, event-based vision has recently garnered significant attention [16]. These advantageous properties have led to promising advancements in downstream tasks such as object tracking, detection, and recognition [14, 18â 21, 28, 34, 52, 61, 71], as well as optical flow estimation [4, 22, 83]. However, being a novel data format, event data necessitates specialized architectures and algorithms to be effectively processed and utilized in various applications. In this work, we present a method to exploit event data for real-time novel view synthesis, a scenario that has not been addressed previously.

## 2.2. Volumetric Rendering for View Synthesis

Neural Radiance Fields (NeRFs)[46] have been the stateof-the-art in view synthesis, enabled by volumetric rendering and neural scene representations. Significant research has been conducted to improve NeRFs in terms of training and rendering speed [6, 9, 23, 32, 49, 58, 60, 76] and visual quality [1, 2, 12, 39, 65]. However, due to the expensive volumetric sampling process, NeRF-based methods have generally been unable to render high-quality views in real time. Recently, 3D Gaussian Splatting (3DGS) [7, 10, 26, 53, 85] proposed a rasterization-based approach for differentiably rendering scenes represented as sets of 3D Gaussian ellipsoids. This method achieves real-time renderings while preserving the quality of state-of-the-art NeRF-based methods. Since the introduction of 3DGS, real-time capabilities of 3DGS have resulted in rapid adoption of Gaussians in several downstream tasks including dynamic scene reconstruction [73], 3D asset generation [75], SLAM [45], and 4D object generation [40]. For view synthesis from event data, we leverage 3DGS and adapt it to be trainable using event sequences.

## 2.3. Novel View Synthesis via Event Data

Motivated by the unique benefits of event-based cameras, such as high dynamic range, low latency, and high temporal resolution, there has been significant research interest in adopting event data in applications like 3D reconstruction and novel view synthesis in cases where traditional RGB cameras underperform [16, 54, 81, 82]. Recent methods explore using NeRFs with event-vision supervision [25, 31, 42, 59] to reconstruct scenes and render them from novel views. While capable of generating high-quality renderings, NeRF-based methods are generally slow to render due to expensive sampling requirements [6, 17, 57, 62]. On the other hand, 3DGS requires RGB data for supervision, and adapting 3DGS for scenarios where the supervision signal comes from an event-based camera has been challenging [11, 66, 68, 70, 72, 77, 80]. In contrast to previous event-based view synthesis methods, we enable the use of event data to supervise 3DGS, allowing for real-time renderings and state-of-the-art quality using only event data, supported by event-based initial point clouds and effective camera trajectory interpolation.

## 3. Background

## 3.1. Event Camera Data

While traditional cameras output an intensity image I, event cameras model changes in the log intensity L = log(I)

of the current camera view. Similar to traditional cameras, they have a pixel array. However, in contrast to traditional cameras, each pixel acts as an independent sensor and triggers an event whenever the difference between the current log-intensity $l _ { x , y } ( t )$ and the log-intensity at the time of the most recent event $l _ { x , y } ( t _ { \mathrm { r e c e n t } } )$ surpasses a given threshold $\Delta > 0 ;$ , i.e. when

$$
l _ { x , y } ( t ) - l _ { x , y } ( t _ { \mathrm { r e c e n t } } ) > \Delta .\tag{1}
$$

This threshold is also known as contrast sensitivity [16]. It can be different for positive and negative polarity changes.

Formally, each event is a tuple $e : = ( t , x , y , p )$ where $t \in$ R is a timestamp at which it was triggered, $( x , y ) \in \mathbb { Z } ^ { 2 }$ are its pixel coordinates, and $p \in \{ - 1 , 1 \}$ is the eventâs polarity indicating whether the log-intensity increased or decreased. Similar to traditional color cameras, a Bayer RGB pattern is used to obtain color information in color event cameras. In such sensors, each individual event provides information only for one color channel.

## 3.2. 3D Gaussian Splatting

To enable novel view synthesis, 3DGS learns a scene representation that consists of a set of ellipsoids with viewdependent color information encoded as spherical harmonics [7, 10, 26, 53, 85]. The essence of the method revolves around transforming optimized 3D Gaussians into 2D imagery based on predetermined camera positions Subsequently the approach calculates pixel values accordingly by arranging and rasterization [7, 10, 26, 53, 85]. The transformation process involves mapping 3D Gaussian distributions, shaped like ellipsoids, onto a two-dimensional plane as ellipses for the purpose of rendering. This is achieved by applying a specific viewing transformation, denoted by W , along with a 3D covariance matrix Î£. The outcome is the derivation of a 2D covariance matrix Î£â² as follows

$$
\Sigma ^ { \prime } = J W \Sigma W ^ { \top } J ^ { \top } ,\tag{2}
$$

where the Jacobian J is used to linearly approximate projective transformations.

The pixel rendering procedure begins by pinpointing a pixelâs location, represented as $p ,$ within the image plane and then assess its closeness to the intersecting Gaussians. This step involves gauging the Gaussianâs depth via applying a viewing transformation, denoted as W [7]. This process forms an sorted sequence of Gaussians, labeled $N _ { s }$ Subsequently, a technique known as alpha compositing is employed to determine the pixelâs color

$$
I = \sum _ { i \in N _ { s } } \left( c _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } ) \right) .\tag{3}
$$

Here, $c _ { i }$ and $\alpha _ { i } ^ { \prime }$ indicates the learned color and the opacity, respectively.

## 4. Method

The goal of our work is to obtain a Gaussian Splatting-based 3D representation of a static scene using a data stream from a moving event camera. Formally, we are given a sequence of N events $e _ { k } = ( t _ { k } , x _ { k } , y _ { k } , p _ { k } )$ with corresponding camera poses $P _ { k } \in \mathrm { S E } ( 3 )$ denoted as $\mathcal { E } : = \{ ( e _ { k } , \overline { { P _ { k } } } ) \} _ { k = 1 } ^ { \bar { N } }$ . In practice, such dense pose information may not be available. However, it can be computed from sparse poses via interpolation. The interpolation to estimate event camera poses is explained in Sec. 5.1. From the posed event sequence, we aim to obtain a 3DGS-based scene representation G that can be used for novel view synthesis of traditional rgb and grayscale images.

The key idea of our optimization approach is to model changes in logarithmic image intensity. That is, for a trajectory sequence over a given time interval [a, b] and corresponding image intensity $I ( t ) \in \mathbb { R } ^ { H \times W \times 3 }$ , our work models

$$
E ( a , b ) : = \int _ { a } ^ { b } \log { ( I ^ { \prime } ( t ) ) } \mathrm { d } t .\tag{4}
$$

For loss computation, we use both, event camera data and 3DGS to approximate this quantity. In the case of event cameras, this approximation can be obtained directly by creating an image of log intensity changes. Approximating Eq. (4) with 3DGS, however, requires additional modifications to the training process.

Method Overview In its original formulation, 3D Gaussian Splatting uses a sequence of posed images as its training data. To make Gaussian Splatting applicable to event data, we introduce several modifications to the original method: First, we approximate Eq. (4) by accumulating events into an image representation $\dot { D } \in \mathbb { R } ^ { H \times W }$ using a random sub-trajectory of the given event stream \mathcal {E} (Sec. 4.1). This representation contains the sum of quantized log-intensity changes over the sub-trajectory. Second, we compute the corresponding log-intensity changes by rasterizing two views from the Gaussian scene representation (Sec. 4.2). For color cameras, this additionally requires a remosaicing step to account for the fact that the accumulated image representation of events only contains single color information for each pixel. To improve the 3DGS optimization process, we leverage an event-to-video guided structure from motion (Sec. 4.3). We also leverage cubic spline interpolation to more accurately estimate camera sub-trajectories as the rate at which the camera is tracked may be too coarse (Sec. 4.4). The full scene representation learning process is illustrated in Fig. 2.

## 4.1. Event Accumulation

To obtain an image of accumulated events, we first randomly sample start and endpoints $k _ { \mathrm { s t a r t } } , k _ { \mathrm { e n d } } \ \in \ \left\{ 1 \ldots N \right\}$ from our trajectory sequence. The lengths of those sequences are chosen randomly with a maximum length ranging from 1% to 10% of the total number of events. This allows for considering scene information at different scales in order to have both global and local geometry information trained. Then, we obtain the entries of the accumulated log-intensity change image D at pixel location $x ,$ y as

<!-- image-->  
Figure 2. Overview of our 3D Gaussian Splatting training with moving event camera data. Event data streams from $t _ { k _ { s t a r t } }$ to $t _ { k _ { e n d } }$ are 4/ ââ 2025/03/20 SSS ç¬¬3ç ç©¶é¨é accumulated into D, and distill the point priors from an event-to-video model. The log-difference image $\hat { D }$ is obtained from 3D Gaussian Splatting and Rasterization at two camera view points. It is computed as in Eq. (6) and remosaicing is performed in case of a color event camera. The respective poses are estimated by cubic spline interpolation.

$$
D _ { x , y } : = \sum _ { \stackrel { k \in \{ k _ { \mathrm { s t a r t } } , \ldots , k _ { \mathrm { e n d } } \} } { x _ { k } = x , y _ { k } = y } } p _ { k } \Delta .\tag{5}
$$

The resulting image serves as an approximation to Eq. (4), i.e. $D \approx E ( t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } )$ This method can easily be adapted to the case where the event-threshold $\Delta$ is asymmetric, i.e., when it is different for positive and negative polarity events.

## 4.2. Training View Generation and Remosaicing

To generate the log-difference image $\hat { D } \in \mathbb { R } ^ { H \times W }$ , we compute

$$
\hat { D } : = \hat { L } _ { 2 } - \hat { L } _ { 1 }\tag{6}
$$

which requires the remosaiced log images $L _ { 1 } , L _ { 2 } \in \mathbb { R } ^ { H \times W }$ at the poses $P _ { k _ { \mathrm { s t a r t } } }$ and $P _ { k _ { \mathrm { e n d } } }$ respectively. These poses are the same that were used for generating our target $D .$ . To obtain $L _ { 1 }$ and $L _ { 2 }$ we first use the rasterizer to generate corresponding views $I _ { 1 } , I _ { 2 } \in \mathbb { R } ^ { H \times W \times 3 }$ from the current Gaussian scene representation. Next, we perform a standard remosaicing operation because events are triggered per pixel asynchronously, not allowing us to use conventional demosaicing methods [5, 29, 35, 36, 41, 44]

$$
{ \mathrm { R e m o s a i c i n g : } } \mathbb { R } ^ { H \times W \times 3 }  \mathbb { R } ^ { H \times W }\tag{7}
$$

which reintroduces the Bayer RGB pattern. It consists of two steps: First, a set of color-channel specific matrices of size $2 \times 2$ are Hadamard multiplied with each $2 \times 2$ pixel block in the image. The channel specific matrices $\mathbf { \bar { \mathcal { R } } } , G , B \in \mathbb { R } ^ { 2 \times 2 }$ are given by

$$
R = \left[ \begin{array} { l l } { 1 } & { 0 } \\ { 0 } & { 0 } \end{array} \right] , \quad G = \left[ \begin{array} { l l } { 0 } & { 1 } \\ { 1 } & { 0 } \end{array} \right] , \quad B = \left[ \begin{array} { l l } { 0 } & { 0 } \\ { 0 } & { 1 } \end{array} \right] .\tag{8}
$$

The thus modified channels have non-zero entries at nonoverlapping pixel locations. From this, the single-channel remosaiced image is obtained by addition across the channels. Finally, entry-wise logarithm computation yields the desired images $\hat { L } _ { 1 } , \hat { L } _ { 2 }$ and thus DË.

## 4.3. Event-to-Video Guided Initialization

Event cameras capture only the pixels that indicate changes in brightness. This results in sparse information that is often insufficient for effective supervision. Thus, reconstructing 3D information, including background details, solely from event data is exceedingly challenging. To simplify the optimization process we propose leveraging the prior knowledge encoded in an event-to-video model. We use this pretrained model to generate images from event streams that can subsequently be processed by Structure from Motion (SfM) techniques. Although the generated images often contain significant noise and do not accurately reproduce the underlying RGB image values, they retain substantial texture and background information. This information is sufficient to provide initial positions for the Gaussians, improve the 3DGS optimization process and lead to faster convergence compared to basic initialization with random initial points.

## 4.4. Event Camera Trajectory Interpolation

The camera poses, $P _ { k _ { \mathrm { s t a r t } } }$ and $P _ { k _ { \mathrm { c n d } } }$ , used to obtain the log images $\hat { L } _ { 1 }$ and $\hat { L } _ { 2 }$ in Eq. (6), are in most cases not given a priori. This is because event camera poses are typically obtained from a tracking or motion estimation system that operates at a fixed rate, which may not be aligned with the rate of event accumulation. Thus, to obtain an estimate of the complete trajectory of the event camera, we adopt an efficient trajectory interpolation method for camera poses. Our work utilizes Cubic Spline Interpolation and Spherical Cubic Spline Interpolation. These generate a smooth path by interpolating between discrete command points using third-degree polynomials. This interpolation technique closely approximates real-world camera motion by maintaining non-linear continuity in both velocity and acceleration of camera movement.

## 4.5. Loss Function

3D Gaussian Splatting is updated using a loss that combines a $\mathcal { L } _ { 1 }$ reconstruction term and a perceptual term \mathcal {L}_{\text {SSIM}} that is based on the Structural Similarity Index Measure (SSIM) [67]

$$
\begin{array} { r } { \mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { \mathrm { S S I M } } . } \end{array}\tag{9}
$$

To fully utilize real-world event camera data, undistortion operation is performed as the accumulated image D is computed. We account for this by only incurring the loss where events were actually accumulated. This allows supervising the 3D representation more efficient.

## 5. Experiments

## 5.1. Datasets

Our evaluation of 3D Gaussian Splatting from a moving event camera encompasses both synthetic and real-world scenes. Our method assumes that we can access the intrinsic camera matrix, its lens distortion parameters, and the cameraâs constant-rate poses, sampled at a high frequency to ensure precise interpolation for the event camera at any chosen time point. Estimation of camera poses at any given timestep is crucial for precise scene reconstruction from event data, as events can asynchronously occur at arbitrary timesteps between the constant-rate sampled camera poses.

## 5.1.1. Synthetic Scenes

The synthetic dataset used in our experiments consists of event data streams obtained from ESIM [55], initially introduced in [42], encompassing seven synthetic scenes with a white background. These simulated event data streams feature a diverse range of photorealistic objects with complex structures. The event camera in these synthetic scenes moves around the objects, providing diverse geometric and lighting conditions. This setup allows for a comprehensive evaluation of the robustness of our method against the baselines in a controlled environment.

## 5.1.2. Real-World Scenes

The experiments on real scenes were conducted using subsets of the EDS [24], and TUM-VIE [30] datasets. From EDS, we utilize the sequences 03_rocket_earth_dark, 07_ziggy_and_fuzz_hdr, 08_peanuts_running, 11_all_characters and 13_airplane. From TUM-VIE, we utilize the sequences mocap-1d-trans and mocap-desk2. These sequences primarily capture various objects in 360-degree scenes and on a desk from a forward-facing perspective. These real sequences are relatively well-suited for novel view synthesis and were recorded using the high-resolution event sensor, the Prophesee Gen 3 in the EDS and the Prophesee Gen 4 in the TUM-VIE dataset, respectively.

## 5.2. Baselines

We benchmarked our method against Robust-e-NeRF and E2VID [56] + 3D Gaussian Splatting (E2VID+3DGS). Robust-e-NeRF incorporates InstantNGP, which is significantly faster compared to traditional NeRF methods, but still falls short compared to 3DGS. E2VID generates RGB images from an event data stream, followed by feeding the images into original 3D Gaussian Splatting.

## 5.3. Metrics

We utilized commonly used metrics to evaluate novel view synthesis, including Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS [79]) with VGG16, to comprehensively assess the quality of rendered images. Itâs worth noting that SSIM and LPIPS are perceptual metrics introduced to align more closely with human preference, with LPIPS specifically tailored to mimic human perception. These metrics address issues associated with PSNR, such as favoring blurry images. Additionally, we measured and compared the rendering speeds of our method, Robust-e-NeRF, and E2VID+3DGS.

Since event cameras capture variations in log-radiance rather than absolute log-radiance values, the predicted intensity I(t) from the 3D Gaussian Splatting has an unknown offset. To rectify this limitation, a linear color transformation is designed to adjust our predictions in the logarithmic domain [42, 59]. This transformation is both necessary and adequately effective for aligning our predictions with the reference data. It ensures that the reconstructed intensity values are properly calibrated and aligned with the observed event data.

<!-- image-->  
Figure 3. Generated Synthetic images comparing our work, event-based NeRF, and E2VID+3DGS qualitatively, with rendering times SSS ç¬¬3ç ç©¶é¨é shown at the bottom of each image. Our work and Robust-e-NeRF outperform others in most scenes. In some cases, e.g., the drums scene, our work performs better and achieves significantly faster rendering times than conventional Robust-e-NeRF composed of InstantNGP.

Table 1. Generated synthetic scenes in comparison between our work, previous event-based NeRF and E2VID+3DGS quantitatively. We supervised our model with randomly initialized points.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Method</td><td colspan="7">Synthetic Scene</td><td rowspan="2">Mean</td></tr><tr><td>chair</td><td>drums</td><td>ficus</td><td>hotdog</td><td>lego</td><td>materials</td><td>mic</td></tr><tr><td rowspan="3">PSNRâ</td><td>E2VID + 3DGS</td><td>21.39</td><td>19.86</td><td>19.90</td><td>15.55</td><td>18.17</td><td>20.08</td><td>20.10</td><td>19.29</td></tr><tr><td>Robust e-NeRF</td><td>30.24</td><td>23.15</td><td>30.71</td><td>28.07</td><td>27.34</td><td>24.98</td><td>32.87</td><td>28.19</td></tr><tr><td>Ours</td><td>28.69</td><td>25.81</td><td>29.90</td><td>22.91</td><td>29.22</td><td>27.16</td><td>33.27</td><td>28.14</td></tr><tr><td rowspan="3">SSIMâ</td><td>E2VID + 3DGS</td><td>0.934</td><td>0.915</td><td>0.922</td><td>0.897</td><td>0.895</td><td>0.901</td><td>0.957</td><td>0.917</td></tr><tr><td>Robust e-NeRF</td><td>0.958</td><td>0.897</td><td>0.971</td><td>0.953</td><td>0.934</td><td>0.923</td><td>0.981</td><td>0.945</td></tr><tr><td>Ours</td><td>0.953</td><td>0.947</td><td>0.966</td><td>0.940</td><td>0.945</td><td>0.936</td><td>0.986</td><td>0.953</td></tr><tr><td rowspan="3">LPIPS â</td><td>E2VID + 3DGS</td><td>0.076</td><td>0.094</td><td>0.108</td><td>0.208</td><td>0.145</td><td>0.125</td><td>0.069</td><td>0.118</td></tr><tr><td>Robust e-NeRF</td><td>0.040</td><td>0.091</td><td>0.022</td><td>0.095</td><td>0.074</td><td>0.052</td><td>0.029</td><td>0.057</td></tr><tr><td>Ours</td><td>0.047</td><td>0.052</td><td>0.028</td><td>0.098</td><td>0.055</td><td>0.060</td><td>0.015</td><td>0.051</td></tr></table>

## 5.4. Results and Evaluation

After completing the training of our model on each scene using event sequence data, the model acquires a set of 3D Gaussian ellipsoids representing the scene. By rasterizing these Gaussians, we are able to render various arbitrary views of each scene in real-time. The renderings of novel views, obtained from held-out views or test views provided by the datasets, are then used for evaluations.Bac

<!-- image-->  
SSS ç¬¬3ç ç©¶é¨éFigure 4. Qualitative comparisons of the images generated by our method, event-based NeRF and E2VID+3DGS show that our technique appears to recover details in 5 real scenes. First three rows(07_ziggy_and_fuzz_hdr, 08_peanuts_running and 11_all_characters) are EDS dataset and last two rows are TUM-VIE dataset(mocap-1d-trans and mocap-desk2).

Table 2. Real scenes in comparison between our work, previous event-based NeRF and E2VID+3DGS quantitatively.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Method</td><td colspan="5">Real Scene</td><td rowspan="2">Mean</td></tr><tr><td>03</td><td>07</td><td>08</td><td>11</td><td>13</td></tr><tr><td rowspan="3">PSNR â</td><td>E2VID + 3DGS</td><td>15.67</td><td>15.05</td><td>14.03</td><td>13.83</td><td>18.96</td><td>15.51</td></tr><tr><td>Robust e-NeRF</td><td>19.19</td><td>14.78</td><td>14.75</td><td>14.43</td><td>18.10</td><td>16.25</td></tr><tr><td>Ours</td><td>20.78</td><td>19.14</td><td>17.53</td><td>17.79</td><td>19.05</td><td>18.86</td></tr><tr><td rowspan="3">SSIMâ</td><td>E2VID + 3DGS</td><td>0.716</td><td>0.689</td><td>0.642</td><td>0.691</td><td>0.723</td><td>0.692</td></tr><tr><td>Robust e-NeRF</td><td>0.846</td><td>0.815</td><td>0.735</td><td>0.569</td><td>0.729</td><td>0.739</td></tr><tr><td>Ours</td><td>0.835</td><td>0.816</td><td>0.745</td><td>0.789</td><td>0.774</td><td>0.792</td></tr><tr><td rowspan="3">LPIPS â</td><td>E2VID + 3DGS</td><td>0.266</td><td>0.378</td><td>0.402</td><td>0.415</td><td>0.415</td><td>0.375</td></tr><tr><td>Robust e-NeRF</td><td>0.324</td><td>0.476</td><td>0.567</td><td>0.700</td><td>0.650</td><td>0.543</td></tr><tr><td>Ours</td><td>0.239</td><td>0.351</td><td>0.424</td><td>0.391</td><td>0.407</td><td>0.363</td></tr></table>

## 5.4.1. Synthetic Scenes

We evaluate our methodology on a synthetic dataset using the metrics described in Sec. 5.3. Quantitative results are shown in Tab. 1, and visual samples and rendering times in Fig. 3. Rendering times are provided in milliseconds (ms). Our method achieves higher average SSIM and compared to previous works. A high SSIM value indicates structural similarity between generated and reference images, underscoring our modelâs accuracy in capturing essential scene geometry. Additionally, a low LPIPS score reflects perceptual quality, showing the produced images are both realistic and visually engaging. The elevated scores demonstrate that our novel view synthesis model faithfully reproduces the visual and perceptual attributes of the original scene. Moreover, the rendering speed of both our method and E2VID+3DGS significantly surpass that of Robust-e-NeRF. Therefore, our work maintains superior quality and speed compared to Robust-e-NeRF, highlighting its effectiveness in both fidelity and performance.

As shown in Fig. 3, the quality of rendered images remains high in scenes featuring drums and microphones, achieving novel view synthesis without sacrificing quality in other scenes as well.

## 5.4.2. Real Scenes

The EDS dataset is the only one available for quantifying the performance of novel view synthesis. Therefore, we primarily conducted quantitative and qualitative assessments, as shown Tab. 2 and Fig. 4. Our method successfully recovers fine details in all sequences and contrasts in the mocap-1d-trans image, particularly. This is particularly noticeable in the images containing the checkerboard and the cover of a book. Additionally, it effectively smooths uniform areas such as the checkerboard and the desk surface in the mocap-1d-trans and mocap-desk2. This smoothness likely results from the accumulation of event streams, which may have enhanced the signal-to-noise ratio due to the aggregation process. The visible artifacts near the borders of all synthesized views are attributed to the relatively narrow field of view of the event cameras.

## 5.5. Ablations

To demonstrate the effectiveness of event-to-video guided initialization and camera trajectory estimation with cubic spline interpolation, we conducted ablation experiments by separately adding 1) event-to-video guided initialization and 2) both the initialization and cubic spline interpolation. As shown in Tab. 3, event-to-video guided initialization outperforms random initialization. Furthermore, Fig. 5 shows that our initialization approach results in better image quality (e.g. backgrounds) compared to random initialization.

Table 3. Ablation results for the event-to-video guided initialization and cubic spline interpolation.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>random init</td><td>18.02</td><td>0.767</td><td>0.397</td></tr><tr><td>guided init</td><td>18.75</td><td>0.788</td><td>0.359</td></tr><tr><td>guided init + cubic</td><td>18.86</td><td>0.792</td><td>0.363</td></tr></table>

<!-- image-->  
Figure 5. Qualitative comparisons of the images generated by random initialization and event-to-video guided initialization are shown in 07_ziggy_and_fuzz_hdr scene in EDS dataset.

Adding cubic spline interpolation further enhances performance.

## 5.6. Limitations

SSS ç¬¬3ç ç©¶é¨é 03/20 Our method, though effective, has some limitations. It is designed exclusively for novel view synthesis in static scenes and cannot handle scenarios with dynamic moving objects, which we consider to be an exciting avenue for future research. Additionally, our approach trains 3D Gaussian Splatting on event-accumulated images between two viewpoints, representing relative intensity images. Consequently, similar to other event-based view synthesis methods, the model cannot directly estimate absolute intensity images and requires a linear transformation with evaluation data as a reference. This linear transformation is required only during inference and does not impact the training.

## 6. Conclusion and Future Work

In this work, we introduce a novel method for 3D Gaussian Splatting from moving event cameras. Our approach achieves the higher visual fidelity and better performance of novel view synthesis in static scenes. We distill geometric priors from the event-to-video model for initialization and leverage effective camera trajectory interpolation. Thus, EventSplat yields improved metrics compared to existing event-based NeRF approaches. Moreover, the incorporation of 3D Gaussian Splatting enables our model to render scenes at significantly higher frame rates. This enhances the practicality of real-time novel view synthesis with event cameras. Our work is thus particularly useful in scenarios where conventional imaging techniques are inadequate.

## Acknowledgments

The work is in part supported by Sony. We would also like to thank Ziyi Wu, Yash Kant, and Umangi Jain, for valuable discussions and support.

## References

[1] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-NeRF: A multiscale representation for antialiasing neural radiance fields. ICCV, 2021. 1, 2

[2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022. 2

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields. In ICCV, 2023. 1

[4] Ryad Benosman, Charles Clercq, Xavier Lagorce, Sio-Hoi Ieng, and Chiara Bartolozzi. Event-based visual flow. TNNLS, 2013. 2

[5] Hong Cao and Alex C Kot. Accurate detection of demosaicing regularity for digital image forensics. IEEE Trans. Inf. Forensics Secur, 2009. 4

[6] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In ECCV. Springer, 2022. 2

[7] Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890, 2024. 2, 3

[8] Jiaben Chen, Yichen Zhu, Dongze Lian, Jiaqi Yang, Yifu Wang, Renrui Zhang, Xinhang Liu, Shenhan Qian, Laurent Kneip, and Shenghua Gao. Revisiting Event-Based Video Frame Interpolation. In IROS, 2023. 1

[9] Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and Andrea Tagliasacchi. MobileNeRF: Exploiting the polygon rasterization pipeline for efficient neural field rendering on mobile architectures. In arXiv, 2022. 1, 2

[10] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3D using Gaussian Splatting. arXiv preprint arXiv:2309.16585, 2023. 2, 3

[11] Hiroyuki Deguchi, Mana Masuda, Takuya Nakabayashi, and Hideo Saito. E2gs: Event enhanced gaussian splatting. In 2024 IEEE International Conference on Image Processing (ICIP), 2024. 2

[12] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised NeRF: Fewer views and faster training for free. In CVPR, 2022. 2

[13] Davide Falanga, Kevin Kleber, and Davide Scaramuzza. Dynamic obstacle avoidance for quadrotors with event cameras. Sci. Robot, 2020. 1

[14] Guillermo Gallego, Jon EA Lund, Elias Mueggler, Henri Rebecq, Tobi Delbruck, and Davide Scaramuzza. Eventbased, 6-dof camera tracking from photometric depth maps. TPAMI, 2017. 2

[15] Guillermo Gallego, Henri Rebecq, and Davide Scaramuzza. A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation. In CVPR, 2018. 1

[16] Guillermo Gallego, Tobi DelbrÃ¼ck, Garrick Orchard, Chiara Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger, Andrew J. Davison, JÃ¶rg Conradt, Kostas Daniilidis, and Davide Scaramuzza. Event-based vision: A survey. PAMI, 2022. 1, 2, 3

[17] Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. FastNeRF: High-Fidelity Neural Rendering at 200FPS. In ICCV, 2021. 2

[18] Daniel Gehrig, Henri Rebecq, Guillermo Gallego, and Davide Scaramuzza. Asynchronous, photometric feature tracking using events and frames. In ECCV, 2018. 2

[19] Daniel Gehrig, Antonio Loquercio, Konstantinos G Derpanis, and Davide Scaramuzza. End-to-end learning of representations for asynchronous event-based data. In ICCV, 2019.

[20] Daniel Gehrig, Henri Rebecq, Guillermo Gallego, and Davide Scaramuzza. Eklt: Asynchronous photometric feature tracking using events and frames. IJCV, 2020.

[21] Mathias Gehrig and Davide Scaramuzza. Recurrent vision transformers for object detection with event cameras. In CVPR, 2023. 2

[22] Mathias Gehrig, Mario MillhÃ¤usler, Daniel Gehrig, and Davide Scaramuzza. E-raft: Dense optical flow from event cameras. In 3DV, 2021. 2

[23] Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall, Jonathan T. Barron, and Paul Debevec. Baking neural radiance fields for real-time view synthesis. In ICCV, 2021. 2

[24] Javier Hidalgo-CarriÃ³, Guillermo Gallego, and Davide Scaramuzza. Event-aided direct sparse odometry. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 5

[25] Inwoo Hwang, Junho Kim, and Young Min Kim. Ev-NeRF: Event based neural radiance field. In WACV, 2023. 1, 2

[26] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ToG, 2023. 1, 2, 3, 12

[27] Hanme Kim, Stefan Leutenegger, and Andrew J Davison. Real-time 3d reconstruction and 6-dof tracking with an event camera. In CVPR, 2016. 1

[28] Junho Kim, Jaehyeok Bae, Gangin Park, Dongsu Zhang, and Young Min Kim. N-imagenet: Towards robust, fine-grained object recognition with event cameras. In ICCV, 2021. 2

[29] Ron Kimmel. Demosaicing: Image reconstruction from color CCD samples. IEEE Trans. Image Process, 1999. 4

[30] Simon Klenk, Jason Chui, Nikolaus Demmel, and Daniel Cremers. TUM-VIE: The TUM Stereo Visual-Inertial Event Dataset. In IROS, 2021. 5

[31] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers. E-nerf: Neural radiance fields from a moving event camera. RA-L, 2023. 1, 2

[32] Andreas Kurz, Thomas Neff, Zhaoyang Lv, Michael ZollhÃ¶fer, and Markus Steinberger. AdaNeRF: Adaptive sampling for real-time rendering of neural radiance fields. In ECCV, 2022. 2

[33] Christoph Lassner and Michael Zollhofer. Pulsar: Efficient Sphere-Based Neural Rendering. In CVPR, 2021. 1

[34] Jianing Li, Jia Li, Lin Zhu, Xijie Xiang, Tiejun Huang, and Yonghong Tian. Asynchronous spatio-temporal memory network for continuous event-based object detection. TIP, 2022. 2

[35] Xin Li. Demosaicing by successive approximation. IEEE Trans. Image Process, 2005. 4

[36] Xin Li, Bahadir Gunturk, and Lei Zhang. Image demosaicing: A systematic survey. In VCIP. SPIE, 2008. 4

[37] Patrick Lichtsteiner, Christoph Posch, and Tobi Delbruck. A 128Ã 128 120 db 15 Âµs latency asynchronous temporal contrast vision sensor. JSSC, 2008. 1

[38] Songnan Lin, Jiawei Zhang, Jinshan Pan, Zhe Jiang, Dongqing Zou, Yongtian Wang, Jing Chen, and Jimmy Ren. Learning Event-Driven Video Deblurring and Interpolation. In ECCV, 2020. 1

[39] David B Lindell, Dave Van Veen, Jeong Joon Park, and Gordon Wetzstein. BACON: Band-limited coordinate networks for multiscale scene representation. In CVPR, 2022. 2

[40] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. arXiv preprint arXiv:2312.13763, 2023. 2

[41] Philippe Longere, Xuemei Zhang, Peter B Delahunt, and David H Brainard. Perceptual assessment of demosaicing algorithm performance. Proceedings of the IEEE, 2002. 4

[42] Weng Fei Low and Gim Hee Lee. Robust e-NeRF: NeRF from Sparse & Noisy Events under Non-Uniform Motion, 2023. 1, 2, 5, 6, 12

[43] Li Ma, Xiaoyu Li, Jing Liao, Qi Zhang, Xuan Wang, Jue Wang, and Pedro V. Sander. Deblur-NeRF: Neural Radiance Fields From Blurry Images. In CVPR, 2022. 1

[44] Henrique S Malvar, Li-wei He, and Ross Cutler. Highquality linear interpolation for demosaicing of bayerpatterned color images. In ICASSP, 2004. 4

[45] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. arXiv preprint arXiv:2312.06741, 2023. 2

[46] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: representing scenes as neural radiance fields for view synthesis. ECCV, 2021. 1, 2

[47] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan, and Jonathan T. Barron. NeRF in the Dark: High Dynamic Range View Synthesis From Noisy Raw Images. In CVPR, 2022. 1

[48] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G. Derpanis, Jonathan Kelly, Marcus A. Brubaker, Igor Gilitschenski, and Alex Levinshtein. SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting With Neural Radiance Fields. In CVPR, 2023.

[49] Thomas MÃ¼ller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. TOG, 2022. 1, 2

[50] Marc Osswald, Sio-Hoi Ieng, Ryad Benosman, and Giacomo Indiveri. A spiking neural network model of 3d perception for event-based neuromorphic stereo vision systems. Sci. Rep., 2017. 1

[51] Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley, Miaomiao Liu, and Yuchao Dai. Bringing a blurry frame alive at high frame-rate with an event camera. In CVPR, 2019. 1

[52] Etienne Perot, Pierre De Tournemire, Davide Nitti, Jonathan Masci, and Amos Sironi. Learning to detect objects with a 1 megapixel event camera. NeurIPS, 2020. 2

[53] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D Diffusion, 2022. 2, 3

[54] Henri Rebecq, Timo HorstschÃ¤fer, Guillermo Gallego, and Davide Scaramuzza. Evo: A geometric approach to eventbased 6-dof parallel tracking and mapping in real time. RA-L, 2016. 2

[55] Henri Rebecq, Daniel Gehrig, and Davide Scaramuzza. ESIM: an open event camera simulator. In CoRL. PMLR, 2018. 5

[56] Henri Rebecq, RenÃ© Ranftl, Vladlen Koltun, and Davide Scaramuzza. High speed and high dynamic range video with an event camera. PAMI, 2019. 5

[57] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs. In ICCV, 2021. 2

[58] Christian Reiser, Richard Szeliski, Dor Verbin, Pratul P. Srinivasan, Ben Mildenhall, Andreas Geiger, Jonathan T. Barron, and Peter Hedman. MERF: Memory-efficient radiance fields for real-time view synthesis in unbounded scenes. In arXiv, 2023. 2

[59] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. EventNeRF: Neural radiance fields from a single colour event camera. In CVPR, 2023. 1, 2, 6, 12

[60] Sara Fridovich-Keil and Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2022. 2

[61] Amos Sironi, Manuele Brambilla, Nicolas Bourdis, Xavier Lagorce, and Ryad Benosman. HATS: Histograms of averaged time surfaces for robust event-based object classification. In CVPR, 2018. 2

[62] Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles Loop, Derek Nowrouzezahrai, Alec Jacobson, Morgan McGuire, and Sanja Fidler. Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes. In CVPR, 2021. 2

[63] Stepan Tulyakov, Daniel Gehrig, Stamatios Georgoulis, Julius Erbach, Mathias Gehrig, Yuanyou Li, and Davide Scaramuzza. Time lens: Event-based video frame interpolation. In CVPR, 2021. 1

[64] Stepan Tulyakov, Alfredo Bochicchio, Daniel Gehrig, Stamatios Georgoulis, Yuanyou Li, and Davide Scaramuzza. Time Lens++: Event-Based Frame Interpolation With Parametric Non-Linear Flow and Multi-Scale Fusion. In CVPR, 2022. 1

[65] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T. Barron, and Pratul P. Srinivasan. Ref-NeRF: Structured view-dependent appearance for neural radiance fields. CVPR, 2022. 2

[66] Jiaxu Wang, Junhao He, Ziyi Zhang, Mingyuan Sun, Jingkai Sun, and Renjing Xu. Evggs: A collaborative learning framework for event-based generalizable gaussian splatting. arXiv preprint arXiv:2405.14959, 2024. 2

[67] Z. Wang, E.P. Simoncelli, and A.C. Bovik. Multiscale Structural Similarity for Image Quality Assessment. In ACSSC, 2003. 5

[68] Yuchen Weng, Zhengwen Shen, Ruofan Chen, Qi Wang, and Jun Wang. Eadeblur-gs: Event assisted 3d deblur reconstruction with gaussian splatting. arXiv preprint arXiv:2407.13520, 2024. 2

[69] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. arXiv preprint arXiv:2310.08528, 2023. 1

[70] Jingqian Wu, Shuo Zhu, Chutian Wang, and Edmund Y Lam. Ev-gs: Event-based gaussian splatting for efficient and accurate radiance field rendering. arXiv preprint arXiv:2407.11343, 2024. 2, 12, 14

[71] Ziyi Wu, Mathias Gehrig, Qing Lyu, Xudong Liu, and Igor Gilitschenski. Leod: Label-efficient object detection for event cameras. arXiv preprint arXiv:2311.17286, 2023. 2

[72] Tianyi Xiong, Jiayi Wu, Botao He, Cornelia Fermuller, Yiannis Aloimonos, Heng Huang, and Christopher Metzler. Event3dgs: Event-based 3d gaussian splatting for high-speed robot egomotion. In 8th Annual Conference on Robot Learning, 2024. 2

[73] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Realtime Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting. In ICLR, 2024. 2

[74] Lior Yariv, Peter Hedman, Christian Reiser, Dor Verbin, Pratul P Srinivasan, Richard Szeliski, Jonathan T Barron, and Ben Mildenhall. BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis. In SIGGRAPH, 2023. 1

[75] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In CVPR, 2024. 2

[76] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. PlenOctrees for real-time rendering of neural radiance fields. In ICCV, 2021. 2

[77] Wangbo Yu, Chaoran Feng, Jiye Tang, Xu Jia, Li Yuan, and Yonghong Tian. Evagaussians: Event stream assisted gaussian splatting from blurry images. arXiv preprint arXiv:2405.20224, 2024. 2

[78] Zhiyang Yu, Yu Zhang, Deyuan Liu, Dongqing Zou, Xijun Chen, Yebin Liu, and Jimmy S. Ren. Training weakly supervised video frame interpolation with events. In ICCV, 2021. 1

[79] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR, 2018. 5

[80] Zixin Zhang, Kanghao Chen, and Lin Wang. Elite-evgs: Learning event-based 3d gaussian splatting by distilling event-to-video priors. arXiv preprint arXiv:2409.13392, 2024. 2

[81] Yi Zhou, Guillermo Gallego, Henri Rebecq, Laurent Kneip, Hongdong Li, and Davide Scaramuzza. Semi-Dense 3D Reconstruction with a Stereo Event Camera. In ECCV, 2018. 2

[82] Yi Zhou, Guillermo Gallego, and Shaojie Shen. Event-based stereo visual odometry. T-RO, 2021. 2

[83] Alex Zihao Zhu and Liangzhe Yuan. Ev-flownet: Selfsupervised optical flow estimation for event-based cameras. In RSS, 2018. 2

[84] Baozhou Zhu, Peter Hofstee, Johan Peltenburg, Jinho Lee, and Zaid Alars. AutoReCon: Neural Architecture Searchbased Reconstruction for Data-free Compression, 2021. 1

[85] M. Zwicker, H. Pfister, J. van Baar, and M. Gross. EWA volume splatting. In VIS, 2001. 2, 3

## A. Implementation Details

## A.1. Algorithm

Our optimization and densification algorithm is shown in Algorithm 1. All modifications compared to the original Gaussian Splatting process [26] are highlighted in green.

Algorithm 1 Optimization and Densification   
w, h: width and height of the training images   
M â Event-to-VideoGuidedPoints() \triangleright Positions   
S, C, A â InitAttributes() \triangleright Covariances, Colors,   
Opacities   
i â 0 \triangleright Iteration Count   
while not converged do   
$k _ { s t a r t } , k _ { e n d } , D \gets$ GenerateTrainingView()   
$I _ { 1 } \gets \mathrm { R a s t e r i z e } ( M , S , C , A , k _ { s t a r t } )$   
$\boldsymbol { I _ { 2 } } \gets \mathrm { R a s t e r i z e } ( M , S , C , A , k _ { e n d } )$   
$\hat { L } _ { 1 } \gets \mathrm { L o g } ( \mathrm { R e m o s a i c i n g } ( I _ { 1 } ) )$   
$\hat { L } _ { 2 } \gets \mathrm { L o g } ( \mathrm { R e m o s a i c i n g } ( I _ { 2 } ) )$   
$\mathcal { L }  \mathrm { L o s s } ( \hat { L } _ { 2 } - \hat { L } _ { 1 } , D )$ \triangleright Loss   
M , S, C, A â Adam(âL) \triangleright Backprop & Step   
if IsRefinementIteration(i) then   
for all Gaussians $( \mu , \Sigma , c , \alpha )$ in $( M , S , C , A )$ do   
if Î± < Ïµ or IsTooLarge(Âµ, Î£) then \triangleright Pruning   
RemoveGaussian()   
end if   
if $\nabla _ { p } L > \tau _ { p }$ then \triangleright Densification   
$\mathbf { f } \parallel S \parallel > \tau _ { S }$ then \triangleright Over-reconstruction   
SplitGaussian(Âµ, Î£, c, Î±)   
else \triangleright Under-reconstruction   
CloneGaussian $( \mu , \Sigma , c , \alpha )$   
end if   
end if   
end for   
end if   
$i \gets i + 1$   
end while

## A.2. Hyper-parameters and Optimizations

Our approach adopts original 3D Gaussian Splatting as the backbone as it allows for high quality view synthesis with high-speed rendering. The Gaussian Model is initialized with spherical harmonics degree and several parameters, including xyz coordinates, features, scaling, rotation, and opacity. The model sets up essential functions for covariance, opacity, and rotation activations. The model includes functions to densify and prune Gaussians based on gradient thresholds and opacity. This ensures efficient use of computational resources by adding new Gaussians where needed and removing those that are not contributing effectively. Training utilizes the similar optimization strategies and hyper-parameter settings originally proposed for

3D Gaussian Splatting including position, feature, opacity, scaling, and rotation. The learning rate is scheduled to adjust dynamically during training. The only opacity learning rate was changed from 0.05 to 0.01 to make the training more stable. The instability seems to result from the 3D Gaussian Splatting model being supervised from multiview points with different accumulation lengths.

## A.2.1. Contrast threshold

Both Robust-e-NeRF and our method were co-optimized and trained with the symmetric contrast thresholds initialized at $C _ { + 1 } / C _ { - 1 } = 1 . 0$ (more precisely set at $C _ { - 1 } = 0 . 2 5 )$ in synthetic datasets and the EDS dataset, and asymmetric contrast threshold initialized at $C _ { + 1 } / C _ { - 1 } = 1 . 4 5 8$ (set at $C _ { - 1 } = 0 . 2 5 ) [ 4 2 ]$ in the TUM-VIE dataset.

## A.3. Experiment Setup

Our research and development efforts are deeply rooted in the principles of 3D Gaussian Splatting [26] methodology. In pursuit of advancing these technologies, we trained our models for more than 30k iterations (set at 40k iterations). This training was conducted on a NVIDIA GeForce RTX4090 GPU. Training time of synthetic scenes take 1-2 hours and that of real scenes take 1-3 hours at 40k iterations.

## B. Additional Experimental Results

## B.1. Qualitative Results in Synthetic Scenes

Fig. 6 shows the quantitative results of all methods for all seven synthetic scenes. The qualitative results are similar to the quantitative evaluation numbers as shown in Tab. 1. In the drums, lego, materials, and mic scenes, fine details seem to be well reconstructed. The chair and ficus reconstruction results appear to be similar details. In the hotdog case, it seems that the images produced by our method are not as well reconstructed compared to Robust-e-NeRF.

## B.2. Qualitative Results on Different Synthetic Datasets

We evaluated our method on the same scenes used in EV-GS [70] from EventNeRF dataset [59] to compare grayscale results. We computed the mean values across 4 scenes(chair, ficus, hotdog and mic). As shown in Tab. 4, our approach outperforms EV-GS in terms of PSNR and SSIM. Furthermore, Fig. 7 shows qualitative results for 4 synthetic scenes in grayscale, demonstrating that the generated images are reconstructed effectively.

## B.3. Qualitative Results in Real-World Scenes

Fig. 8 and Fig. 9 present additional quantitative results for the scenes 03_rocket_earth_dark, 07_ziggy_and_fuzz_hdr, 08_peanuts_running, 11_all_characters and 13_airplane, as

E2VID+3DGS

Robust-e-NeRF

Ours

Ground truth

<!-- image-->  
Figure 6. Generated images are shown, qualitatively comparing our work, event-based NeRF, and E2VID+3DGS in all synthetic scenes.

Table 4. Quantitative evaluation of mean values across the 4 synthetic scenes from [70].
<table><tr><td rowspan="2">Metric</td><td colspan="2">PSNR â</td><td colspan="2">SSIM â</td></tr><tr><td>Ours</td><td>EV-GS</td><td>Ours</td><td>EV-GS</td></tr><tr><td>Mean</td><td>29.48</td><td>26.6</td><td>0.959</td><td>0.925</td></tr></table>

<!-- image-->  
11/ ââ 2025/03/20 SSS ç¬¬3ç ç©¶é¨éFigure 7. Generated images qualitatively comparing our method with ground truth across 4 synthetic scenes.

well as for the mocap-1d-trans, mocap-desk2 scenes, respectively. Our method demonstrates the ability to reconstruct fine detail in both real-world data. However, there is still room for improvement in the quality of reconstruction for some real-world scenes, particulary concerning floating point clouds and the back wall.

<!-- image-->  
Figure 8. For each scene in the EDS dataset, we show generated images from two viewpoints alongside the ground truth image, comparing our work with event-based NeRF and E2VID+3DGS.

E2VID+3DGS  
Robust-e-NeRF  
Ours  
Ground truth  
<!-- image-->  
SSS ç¬¬3ç ç©¶é¨éFigure 9. For each scene in TUM-VIE dataset, we show generated images from two viewpoints alongside the ground truth image, comparing our work with event-based NeRF and E2VID+3DGS.