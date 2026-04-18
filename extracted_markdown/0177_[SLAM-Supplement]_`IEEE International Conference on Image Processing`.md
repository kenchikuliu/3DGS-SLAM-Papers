# E2GS: EVENT ENHANCED GAUSSIAN SPLATTING

Hiroyuki Deguchi\*, Mana Masuda\*, Takuya Nakabayashi, Hideo Saito

Keio University

## ABSTRACT

Event cameras, known for their high dynamic range, absence of motion blur, and low energy usage, have recently found a wide range of applications thanks to these attributes. In the past few years, the field of event-based 3D reconstruction saw remarkable progress, with the Neural Radiance Field (NeRF) based approach demonstrating photorealistic view synthesis results. However, the volume rendering paradigm of NeRF necessitates extensive training and rendering times. In this paper, we introduce Event Enhanced Gaussian Splatting (E2GS), a novel method that incorporates event data into Gaussian Splatting, which has recently made significant advances in the field of novel view synthesis. Our E2GS effectively utilizes both blurry images and event data, significantly improving image deblurring and producing high-quality novel view synthesis. Our comprehensive experiments on both synthetic and real-world datasets demonstrate our E2GS can generate visually appealing renderings while offering faster training and rendering speed (140 FPS). Our code is available at https://github.com/deguchihiroyuki/E2GS.

Index Termsâ novel view synthesis, deblurring, eventbased vision

## 1. INTRODUCTION

In the task of 3D scene reconstruction and novel view synthesis, we witnessed tremendous progress over the past few years. Especially, after the NeRF (Neural Radiance Field) [1] marked a significant milestone, leading to the active development of various neural rendering techniques for 3D scene reconstruction [2, 3]. Among these, 3D Gaussian Splatting [4] emerged as a simple yet computationally efficient method. It has gained recognition for its rapid training and rendering capabilities. However, these methods generally operate under ideal conditions and often struggle with motion blur, which can severely affect the quality of rendering.

Event cameras, inspired by biological vision systems, asynchronously capture changes in pixel brightness instead of recording absolute intensity at fixed frame rates as traditional frame-based RGB cameras do. This unique approach offers several benefits over conventional cameras, including no-motion blur, high dynamic range, low power consumption, and lower latency. These advantages have spurred the development of various methods to address a range of computer vision challenges, such as optical-flow estimation [5], and video interpolation [6]. To utilize these advantages, event cameras found their direction for development in 3D scene reconstruction tasks to handle high-speed camera movements or low lighting conditions which is hard for conditional RGB cameras [7, 8, 9]. While these methods showed photorealistic image rendering results compared to conditional RGB cameras in such conditions, they still require high computational complexity to train the whole network due to the ray-sampling strategy of the NeRF-based approach.

In this paper, we propose Event Enhanced Gaussian Splatting (E2GS), the first approach that incorporates event data into Gaussian Splatting. By effectively incorporating the blurred RGB image and event data, our E2GS showed a visually appealing image deblurring and novel view synthesis result as shown in Fig.1. Our extensive experiments also showed our E2GS achieved better or competitive results while offering 60 times faster training and 3500 times faster rendering speed compared to E2NeRF.

## 2. RELATED WORKS

## 2.1. 3D Scene Reconstruction

3D scene reconstruction is one of the fundamental functionality of computer vision. Recent advancements in 3D scene reconstruction have gained more attention after the emergence of NeRF [1]. While several methods emerged to strengthen the NeRF-based approach [2, 3], there is one research direction to accelerate network training and image rendering speed [10]. Following this research interests, Kerbl et al.proposed 3D Gaussian Splatting [4], which eliminates the need for raysampling and instead uses Gaussians to present 3D space, which allows faster training and rendering.

From Blurry Images We often observe blurriness in some parts or whole scenes when we casually take pictures. Various factors such as object motion, camera shake, and lens defocusing cause this blurriness. One conditional approach to deblurring images is to estimate the blur kernel or Point Spread Function (PSF) and deconvolve the image. Some works have been proposed to deblur images with the training of the 3D

<!-- image-->

<!-- image-->  
Event Stream

<!-- image-->  
Novel View from 3D Gaussian Splatting

<!-- image-->  
Novel View from E2GS

Fig. 1: When we take as input blurry images of a scene from multiple views, the rendering results of original 3D Gaussian Splatting [4] are also severely blurred. In contrast, our E2GS achieves sharper scene rendering by utilizing event data.

scene reconstruction framework. Deblur-NeRF[11] is a pioneering work that employs an additional MLP to estimate per-pixel blur kernel. Lee et al. [12] proposed to use additional MLP to manipulate the covariance of each Gaussian to model blurriness.

## 2.2. Event-based 3D Scene Reconstruction

Event cameras, also known as dynamic vision sensors (DVS) [13], asynchronously capture pixel brightness changes, drawing inspiration from biological vision systems. This unique recording framework effectively addresses the issue of information loss between frames, a common problem in framebased RGB cameras. Event cameras offer several benefits, including no motion blur, high dynamic range, low power consumption, and reduced latency. Due to these advantages, they have shown remarkable results in various tasks like optical flow estimation [14], depth estimation [15], and feature detection and tracking [16]. Recently, Ev-NeRF [17] and Event-NeRF [8] have managed to train NeRF models solely using the event data. However, these methods experience noticeable artifacts and chromatic aberration, and they also exhibit limited generalization ability in pose estimation for neural representation learning. Meanwhile, E2NeRF [7] has successfully trained a sharper NeRF by utilizing both blurry RGB images and corresponding event data. Despite this advancement, it still suffers from prolonged training and rendering times due to the ray-sampling-based NeRF rendering strategy.

## 3. METHOD

The overview of our method is illustrated in Fig. 2. The input of our method is a set of blurry images and event stream of a static scene. In our E2GS framework, we first perform preprocessing using the correspondence between event data and blurred images. Then, we use two types of loss functions to train the Gaussian Splatting considering the blur.

## 3.1. Preliminary

3D Gaussian Splatting. To represent a volumetric scene and render it, we adopt methods from 3D Gaussian Splatting, which proposes differentiable rasterization. The Gaussians are defined by a full 3D covariance matrix Î£ defined in world space [18]:

$$
G ( { \bf x } ) = e ^ { - \frac { 1 } { 2 } { \bf x } ^ { T } { \bf \Sigma } } ^ { - 1 } { \bf x }  .\tag{1}
$$

To render the novel views, the covariance matrix in the camera coordinates of the novel view can be obtained as:

$$
\begin{array} { r } { \pmb { \Sigma ^ { \prime } } = \mathbf { J } \mathbf { W } \pmb { \Sigma } \mathbf { W } ^ { T } \mathbf { J } ^ { T } . } \end{array}\tag{2}
$$

where J is the Jacobian of the affine approximation of the projective transformation and W is the viewing transform matrix. To directly optimize the Î£, it is expressed as:

$$
\begin{array} { r } { \pmb { \Sigma } = \mathbf { R } \mathbf { S } \mathbf { S } ^ { T } \mathbf { R } ^ { T } , } \end{array}\tag{3}
$$

where S is the scaling matrix and R is the rotation matrix.

<!-- image-->  
Fig. 2: The overview of the Event Enhanced Gaussian Splatting.

Event Data. Event cameras asynchronously report an event $e ( x , y , \tau , p )$ when they detect the brightness changes of pixel $( x , y )$ exceeds the threshold C at time Ï . Instead of reporting the actual intensity value $L ( x , y , \tau )$ , they report intensity change direction p which is defined as follows;

$$
p ( x , y , \tau ) = \left\{ \begin{array} { l l } { + 1 } & { \mathrm { i f } l ( x , y , \tau ) - l ( x , y , \tau ^ { \prime } ) > C } \\ { - 1 } & { \mathrm { i f } l ( x , y , \tau ) - l ( x , y , \tau ^ { \prime } ) < - C } \end{array} \right. ,\tag{4}
$$

where $l ( x , y , \tau ) ~ = ~ \log ( L ( x , y , \tau ) )$ and $\tau ^ { \prime }$ represents the timestamp of the last observed event at pixel $( x , y )$

## 3.2. Preprocessing

To utilize a framework for high temporal resolution event data, it is necessary to prepare the initial point cloud for Gaussian splatting and N equally spaced camera poses during the exposure time of each viewpoint. The specific steps are detailed below.

Image Deblurring. Given a set of blurred images and event stream corresponding to the exposure time of each image, we prepare N timestamps $\{ t _ { i } \} _ { i = 1 } ^ { N }$ which divides the event stream equally into $N - 1$ event bins $\{ B _ { i } \} _ { i = 1 } ^ { N - 1 }$ for a more accurate estimate of the intensity change during the exposure time:

$$
B _ { i } = \{ e _ { j } ( x _ { j } , y _ { j } , \tau _ { j } , p _ { j } ) \} _ { j = 1 } ^ { N _ { e } ^ { i } } ( t _ { i } < \tau _ { i } \leq t _ { i + 1 } ) ,\tag{5}
$$

where $N _ { e } ^ { i }$ indicates the number of events in i-th event bin. To estimate N camera poses at each time $t _ { i \cdot }$ , we use Event-based Double Integral (EDI) [19]. The EDI model assumes that the blurred image is the average of multiple sharp images during the exposure time. Furthermore, based on the relationship between the event data and the change in brightness described in Eq. 4, it is assumed that a sharp image at a certain time can be represented by adding events. From this assumption, the image $I _ { i }$ at the moment $t _ { i }$ can be expressed as follows:

$$
I _ { i + 1 } = I _ { i } \sum _ { j = 1 } ^ { N _ { e } ^ { i } } \mathrm { e x p } ( C p _ { j } ) .\tag{6}
$$

The blurry image $I _ { b l u r }$ can be expressed as the average of the images at each timestamp since we set each timestamp to equally divide the exposure time:

$$
I _ { b l u r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { I _ { i } } .\tag{7}
$$

Camera Pose and Initial Point Cloud Estimation. To estimate the initial 3D Gaussian coordinate and camera pose, we feed all deblurred image sets $\{ I _ { i } \} _ { i = 1 } ^ { N }$ of each blurry image to COLMAP Structure-from-Motion package [20]. Without the image deblurring, the COLMAP often fails as reported in [7]

## 3.3. Loss function

To learn the scene from blurred images, we use two types of losses: Image Rendering Loss and Event Rendering Loss. Image Rendering Loss. To adapt 3D Gaussian Splatting taking blurry images as input, we introduce image rendering loss. With N camera poses $\{ P _ { i } \} _ { i = 1 } ^ { N }$ of each view, we render N rendered images $\left\{ \hat { I } _ { i } \right\} _ { i = 1 } ^ { N }$ . Since we set each timestamp to equally divide the exposure time, we can estimate the blurry image by taking the average of the N rendered images:

$$
\hat { I } _ { b l u r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { I } _ { i } ,\tag{8}
$$

The loss function is L1 loss combined with a D-SSIM loss which compares $\hat { I } _ { b l u r }$ and supervision $I _ { b l u r }$ . The final image rendering loss $\mathcal { L } _ { b l u r }$ is written as follows using a weight loss parameter wD-SSIM:

$$
\mathcal { L } _ { b l u r } = ( 1 - w _ { \mathrm { D - S S I M } } ) \mathcal { L } _ { 1 } + w _ { \mathrm { D - S S I M } } \mathcal { L } _ { \mathrm { D - S S I M } } .\tag{9}
$$

Event Rendering Loss. While image rendering loss simply averages N images to simulate a blurred frame, it does not take into account the temporal process of blurring. To utilize event information to supervise the continuous blurring process with high temporal resolution, we employed event loss. Given estimated images from N poses, we first randomly select the two frames $\left\{ I _ { n } , I _ { m } \right\} \left( n < m \right)$ from $\left\{ \hat { I } _ { i } \right\} _ { i = 0 } ^ { N }$ and convert them into grayscale intensity images $L _ { n }$ and $L _ { m }$ . We take the difference of the two intensity values in the log domain and divide it by the threshold $C$ for each pixel $( x , y )$ to estimate the number of events between two frames:

$$
\hat { B } _ { n m } ^ { \prime } = \left\{ \begin{array} { l l } { \lfloor \frac { \log ( L _ { m } ) - \log ( L _ { n } ) } { C } \rfloor } & { \mathrm { i f } L _ { n } ( x , y ) < L _ { m } ( x , y ) } \\ { \lceil \frac { \log ( L _ { m } ) - \log ( L _ { n } ) } { C } \rceil } & { \mathrm { i f } L _ { n } ( x , y ) \geq L _ { m } ( x , y ) } \end{array} \right.\tag{10}
$$

We use the mean squared error to evaluate the error between estimated event bin image $\hat { B } _ { n m } ^ { \prime }$ and ground truth event bin image $B _ { n m } ^ { \prime } .$ , storing an actual number of events for each pixel. Note that there are cancelations of the positive and negative events in the GT event bin image since our model assumes the monotonic intensity change between the timesteps:

$$
\mathcal { L } _ { e v e n t } = \| \hat { B } _ { n m } ^ { \prime } - B _ { n m } ^ { \prime } \| _ { 2 } ^ { 2 } .\tag{11}
$$

Finally, we combine two loss function $\mathcal { L } _ { b l u r }$ and $\mathcal { L } _ { e \nu e n t }$ by using a weight parameter $w _ { e v e n t }$ to obtain the following loss

$$
\mathcal { L } = \mathcal { L } _ { b l u r } + w _ { e v e n t } \mathcal { L } _ { e v e n t } ,\tag{12}
$$

## 4. EXPERIMENTS

## 4.1. Experimental Setup

We evaluated our E2GS on two different tasks: Image deblurring and novel view synthesis. For the image deblurring task, we evaluate the rendering results from the perspective of the blurry image set. for the novel view synthesis task, we evaluate the rendering results from the perspective not used in the blurry image set.

Implementation Details. Our code is based on 3D Gaussian Splatting [4]. We train each scene with 30k iterations on a single NVIDIA RTX A5000 GPU. For all data, we set $w _ { \mathrm { D - S S I M } } = 0 . 2 , w _ { e v e n t } = 0 . 0 0 5$ , and N = 5. We set the different thresholds for positive and negative events to estimate the event bin image $C _ { p o s } = 0 . 2 , C _ { n e g } = 0 . 3$ . The rest of the parameters follow the 3D Gaussian Splatting default values.

Comparison Methods. To evaluate the effectiveness of utilizing event data to solve the image deblurring task and novel view synthesis task, we compared our model with normal Gaussian Splatting (GS) [4], which takes blurry images as input. Note that we obtained the initial point cloud and camera poses by using deblurred images of EDI same as our methods since COLMAP often fails when we use blurry images. The other comparison method is E2NeRF [7], which is a state-ofthe-art method that solves the image deblurring and the novel view synthesis tasks by utilizing a NeRF-based approach. We also report âGS w/ $\mathcal { L } _ { b l u r } \mathbf { \dot { \Pi } } $ â result which only uses blur Loss ${ \mathcal { L } } _ { \mathrm { b l u r } }$ to evaluate the effectiveness of the event loss $\mathcal { L } _ { \mathrm { e v e n t } }$

Table 1: Quantitative evaluation of our method on the image deblurring. The results in the table are the averages of the six synthetic scenes from NeRF [1].
<table><tr><td>Image Deblur</td><td>GS</td><td>E2NeRF</td><td>GS w/  $\mathcal { L } _ { b l u r }$ </td><td>E2GS (Ours)</td></tr><tr><td>PSNRâ</td><td>22.92</td><td>29.77</td><td>30.20</td><td>30.84</td></tr><tr><td>SSIMâ</td><td>0.886</td><td>0.960</td><td>0.951</td><td>0.957</td></tr><tr><td>LPIPSâ</td><td>0.105</td><td>0.073</td><td>0.064</td><td>0.059</td></tr></table>

Table 2: Quantitative evaluation of our method on the novel view synthesis. The results in the table are the averages of the six synthetic scenes from NeRF [1].
<table><tr><td>View Synthesis</td><td>GS</td><td>E2NeRF</td><td>GS w/  $\overline { { \mathcal { L } _ { b l u r } } }$ </td><td>E2GS (Ours)</td></tr><tr><td>PSNRâ</td><td>22.15</td><td>29.56</td><td>28.33</td><td>28.89</td></tr><tr><td>SSIMâ</td><td>0.878</td><td>0.962</td><td>0.943</td><td>0.949</td></tr><tr><td>LPIPSâ</td><td>0.113</td><td>0.073</td><td>0.071</td><td>0.069</td></tr></table>

Evaluation Metrics. To quantitatively evaluate the quality of the rendered image we employed three extensively recognized metrics to evaluate image quality for the synthetic dataset: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and the Learned Perceptual Image Patch Similarity (LPIPS) [21]. Since the real-world data does not contains ground truth sharp images, we use Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) [22], which evaluates the naturalness of the image without any references based on the distribution of the brightness.

## 4.2. Datasets

To evaluate the effectiveness of our method, we use E2NeRF [7] dataset.

Synthetic data: Synthetic set contains six synthetic scenes (chair, ficus, hotdog, lego, materials, and mic), and it uses the Camera Shakify plugin in Blender to simulate camera shake. The event data are simulated by V2E[23]. Each scene has 100 views of blurry images estimated by 17 different camera poses from the Camera Shakify plugin, its corresponding event data, and camera poses.

Real-world data: Real-world set contains five challenging scenes (letter, lego, camera, plant, and toys) captured by DAVIS346 color event camera [24]. Each scene has 30 views of blurry images and the corresponding event data.

## 4.3. Quantitative Evaluation

Synthetic data: Tab. 1 shows the result of the image deblurring and Tab. 2 shows the result of the novel view synthesis. Tab. 1 shows the result on the image deblurring task, E2GS achieves better or comparable results with E2NeRF. Tab. 2 shows the result on the novel view synthesis task, E2GS achieves better or comparable results with E2NeRF. On both tasks, E2GS outperforms both GS and GS w/ $\mathcal { L } _ { b l u r }$ in all three metrics, which shows the effectiveness of utilizing events and event loss to render novel views from blurry image frames.

Table 3: Quantitative evaluation of the image deblurring task. Showing the BRISQUE results of five scenes from E2NeRF [7] and the average of the five scenes.
<table><tr><td>Image Deblur</td><td>letter</td><td>lego</td><td>camera</td><td>toys</td><td>plant</td><td>Avg.</td></tr><tr><td>GS</td><td>40.68</td><td>39.52</td><td>21.76</td><td>43.66</td><td>38.26</td><td>36.78</td></tr><tr><td>E2NeRF</td><td>44.33</td><td>34.09</td><td>28.89</td><td>43.41</td><td>32.23</td><td>36.59</td></tr><tr><td>E2GS (Ours)</td><td>37.62</td><td>35.2</td><td>19.93</td><td>38.87</td><td>30.87</td><td>32.50</td></tr></table>

Table 4: Quantitative evaluation of the novel view synthesis task. Showing the BRISQUE results of five scenes from E2NeRF [7] and the average of the five scenes.
<table><tr><td>View Synthesis</td><td>letter</td><td>lego</td><td>camera</td><td>toys</td><td>plant</td><td>Avg.</td></tr><tr><td>GS</td><td>40.83</td><td>39.02</td><td>22.01</td><td>44.28</td><td>39.25</td><td>37.08</td></tr><tr><td>E2NeRF</td><td>44.19</td><td>34.23</td><td>28.77</td><td>43.42</td><td>32.03</td><td>36.53</td></tr><tr><td>E2GS (Ours)</td><td>37.10</td><td>35.64</td><td>19.90</td><td>38.74</td><td>32.49</td><td>32.77</td></tr></table>

Table 5: Training and rendering time evaluation.
<table><tr><td></td><td>E2NeRF</td><td>E2GS (Ours)</td></tr><tr><td>Training time</td><td>2 days</td><td>50 min</td></tr><tr><td>Rendering (FPS)</td><td>0.04</td><td>140</td></tr></table>

Real-world data: Tab. 3 and Tab. 4 shows the quantitative result of real-world data for the image deblurring task and the novel view synthesis task respectively. E2GS outperformed other comparable methods for both tasks.

## 4.4. Qualitative Evaluation

Synthetic data: We report the rendering result of synthetic data of our E2GS and two baseline methods in Fig. 5. GS produces reasonable rendering results from their blurry RGB inputs. E2NeRF is achieved to reconstruct the sharp image by utilizing the event data, but they fail to reconstruct the details of the scenes, e.g. small parts and reflection of the surface. Real-world data: Fig. 3 and Fig. 4 show the rendering result of the real-world dataset on the image deblurring task and the novel view synthesis task respectively. Our E2GS achieves to render sharp images for both tasks.

## 4.5. Training Time and Rendering Speed

Tab. 5 shows training time and rendering FPS of E2NeRF and our E2GS. For this evaluation, we use the synthetic dataset with 800 Ã 800 resolution as input. Thanks to the rasterizingbased image rendering framework, our E2GS drastically reduces both training time and rendering time compared to E2NeRF. More specifically, our E2GS reduced the training time to 1/60, and the rendering speed to 1/3500 compared to E2NeRF.

<!-- image-->  
Fig. 3: Qualiative comparison of the image deblurring task on the real-world dataset.

<!-- image-->  
Fig. 4: Qualiative comparison of the novel view synthesis task on the real-world dataset.

## 5. CONCLUSION

In this paper, we propose Event Enhanced Gaussian Splatting (E2GS), the novel framework that effectively utilizes event data into Gaussian Splatting to reconstruct sharp scenes from blurry RGB frames. Comprehensive experiments using the synthetic dataset and the real-world dataset demonstrate that our E2GS achieves visually appealing rendering quality and significantly faster training and rendering speed (140 FPS) compared to previous state-of-the-art methods. Future research directions include addressing dynamic scenes with fast-moving subjects, e.g. sports scenes, which are challenging to handle only by using RGB frame cameras.

Acknowledgment This work was partly supported by JST SPRING, Grant Number JPMJSP2123, and JSPS KAKENHI Grant Number JP23H03422.

GS

<!-- image-->  
Fig. 5: Qualiative comparison on the synthetic dataset. Refer to the red box to see the detailed reconstruction quality. Zoom in for the best view.

## 6. REFERENCES

[1] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Nge, âNerf: Representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, no. 1, pp. 99â 106, 2021.

[2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan, âMip-nerf: A multiscale representation for anti-aliasing neural radiance fields,â in ICCV, 2021, pp. 5855â5864.

[3] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer, âD-nerf: Neural radiance fields for dynamic scenes,â in CVPR, 2021, pp. 10318â 10327.

[4] Kerbl Bernhard, Kopanas Georgios, LeimkuhlerÂ¨ Thomas, and Drettakis George, â3d gaussian splatting for real-time radiance field rendering,â TOG, vol. 42, no. 4, July 2023.

[5] Mathias Gehrig, Mario Millhausler, Daniel Gehrig, and Â¨ Davide Scaramuzza, âE-raft: Dense optical flow from event cameras,â in 3DV, 2021, pp. 197â206.

[6] Songnan Lin, Jiawei Zhang, Jinshan Pan, Zhe Jiang, Dongqing Zou, Yongtian Wang, Jing Chen, and Jimmy Ren, âLearning event-driven video deblurring and interpolation,â in ECCV, 2020, pp. 695â710.

[7] Yunshan Qi, Lin Zhu, Yu Zhang, and Jia Li, âE2nerf: Event enhanced neural radiance fields from blurry images,â in ICCV, 2023, p. 837â847.

[8] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik, âEventnerf: Neural radiance fields from a single colour event camera,â in CVPR, 2023.

[9] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers, âE-nerf: Neural radiance fields from a moving event camera,â RAL, 2023.

[10] Thomas Muller, Alex Evans, Christoph Schied, and Â¨ Alexander Keller, âInstant neural graphics primitives with a multiresolution hash encoding,â ToG, vol. 41, no. 4, pp. 1â15, 2022.

[11] Li Ma, Xiaoyu Li, Jing Liao, Qi Zhang, Xuan Wang, Jue Wang, and Pedro V Sander, âDeblur-nerf: Neural radiance fields from blurry images,â in CVPR, 2022, p. 12861â12870.

[12] Byeonghyeon Lee, Howoong Lee, Xiangyu Sun, Usman Ali, and Eunbyung Park, âDeblurring 3d gaussian splatting,â arXiv preprint arXiv:2401.00834, 2024.

[13] Patrick Lichtsteiner, Christoph Posch, and Tobi Delbruck, âA 128Ã 128 120 db 15 Âµs latency asynchronous temporal contrast vision sensor,â IEEE J. Solid-State Circuits, vol. 43, no. 2, pp. 566â576, 2008.

[14] Himanshu Akolkar, Sio-Hoi Ieng, , and Ryad Benosman, âReal-time high speed motion prediction using fast aperture-robust event-driven visual flow,â TPAMI, vol. 44, no. 1, pp. 361â372, 2020.

[15] Tsuyoshi Takatani, Yuzuha Ito, Ayaka Ebisu, Yinqiang Zheng, and Takahito Aoto, âEvent-based bispectral photometry using temporally modulated illumination,â in CVPR, 2021, p. 15638â15647.

[16] Jiqing Zhang, Bo Dong, Haiwei Zhang, Jianchuan Ding, Felix Heide, Baocai Yin, and Xin Yang, âSpiking transformers for event-based single object tracking,â in CVPR, 2022, p. 8801â8810.

[17] Inwoo Hwang, Junho Kim, and Young Min Kim, âEvnerf: Event based neural radiance field,â in CVPR, 2023, p. 837â847.

[18] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross, âEwa volume splatting,â in Proceedings Visualization. IEEE, 2001, pp. 29â538.

[19] Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley, Miaomiao Liu, and Yuchao Dai, âBringing a blurry frame alive at high frame-rate with an event camera,â in CVPR, 2019, pp. 6820â6829.

[20] Johannes L Schonberger and Jan-Michael Frahm, âStructure-from-motion revisited,â in CVPR, 2016, pp. 4104â4113.

[21] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in CVPR, 2018, pp. 586â595.

[22] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik, âNo-reference image quality assessment in the spatial domain,â TIP, vol. 21, no. 12, pp. 4695â4708, 2012.

[23] Yuhuang Hu, Shih-Chii Liu, and Tobi Delbruck, âv2e: From video frames to realistic dvs events,â in CVPR, 2021, pp. 1312â1321.

[24] Christian Brandli, Raphael Berner, Minhao Yang, Shih-Chii Liu, and Tobi Delbruck, âA 240Ã 180 130 db 3 Âµs latency global shutter spatiotemporal vision sensor,â IEEE J. Solid-State Circuits, vol. 49, no. 10, pp. 2333â2341, 2014.