<!-- page 1 -->
RaindropGS: A Benchmark for 3D Gaussian Splatting under Raindrop
Conditions
Zhiqiang Teng1
Tingting Chen2
Beibei Lin2
Zifeng Yuan2
Xuanyi Li1
Xuanyu Zhang1
Shunli Zhang1
1Beijing Jiaotong University, 2National University of Singapore
24115171@bjtu.edu.cn, tinting.c@u.nus.edu beibei.lin@u.nus.edu, zyuan@u.nus.edu
24110477@bjtu.edu.cn, 24115169@bjtu.edu.cn, slzhang@bjtu.edu.cn
Abstract
3D Gaussian Splatting (3DGS) under raindrop condi-
tions suffers from severe occlusions and optical distor-
tions caused by raindrop contamination on the camera
lens, substantially degrading reconstruction quality. Exist-
ing benchmarks mainly rely on synthetic raindrop datasets
with known camera poses (constrained images), assuming
ideal conditions. However, in real-world scenarios, rain-
drops severely hinder accurate pose estimation and point
cloud initialization, while the large domain gap between
synthetic and real data further limits generalization.
To
tackle these issues, we introduce RaindropGS, a compre-
hensive benchmark designed to evaluate the full 3DGS
pipeline—from unconstrained, raindrop-corrupted images
to clear 3DGS reconstructions. We first collect real-world
paired 3D scenes, each containing three aligned image
sets: raindrop-focused, background-focused, and rain-free
ground truth. These real-world data enable reliable evalu-
ation of reconstruction quality under different focus condi-
tions. Building on this foundation, our benchmark estimates
camera poses and initializes point clouds from raindrop-
corrupted images, followed by raindrop removal for 3D
Gaussian optimization. This setup enables end-to-end eval-
uation of how errors accumulated across the 3DGS recon-
struction pipeline affect the final reconstruction quality un-
der real-world raindrop interference. Through comprehen-
sive experiments and analyses, we reveal critical insights
into the performance limitations of existing 3DGS methods
on unconstrained raindrop images and quantify the impact
of individual pipeline components. These insights establish
clear directions for developing more robust 3DGS methods
under raindrop conditions.
1. Introduction
3D Gaussian Splatting (3DGS) in raindrop-contaminated
scenes presents significant challenges [4, 16, 25], as ad-
herent raindrops on camera lenses cause severe occlusions
and optical distortions [3, 17, 18, 27, 41].
These ar-
tifacts disrupt image correspondence, degrade the qual-
ity of camera pose estimation and point cloud initializa-
tion [13, 21, 36, 44], both of which are essential for suc-
cessful 3DGS reconstruction. Moreover, the presence of
raindrops varies across views, blurring images by chang-
ing the camera focal plane [15, 16, 39, 39, 40], introducing
multi-view inconsistencies that further hinder reconstruc-
tion fidelity [19, 26].
Several recent methods [17, 18, 27] have explored 3D
Gaussian Splatting under raindrop scenarios and demon-
strated promising results on synthetic datasets. However,
such evaluation settings are overly idealized and fail to cap-
ture the complexity and diversity of real-world conditions.
To be specific, these methods typically assume the rain-
drop inputs are constrained images, where a clear details of
both raindrops shape and background scenes, a good cam-
era pose and point cloud initialization. However, acquiring
such information from real-world raindrop-affected images
is challenging [9, 23, 43]. Inaccuracies in pose estimation
and point cloud initialization can significantly degrade the
quality of subsequent 3DGS reconstruction [5, 9, 34]. Fur-
thermore, the substantial domain gap between synthetic and
real raindrops raises concerns about generalization. Meth-
ods validated on synthetic datasets often fail to perform well
when applied to real-world scenes.
To address these issues, we introduce RaindropGS,
a comprehensive benchmark for evaluating the complete
raindrop 3DGS pipeline—from unconstrained, raindrop-
corrupted input images to clear 3DGS reconstructions.
As illustrated in Figure 1, our pipeline consists of three
stages: data preparation, data processing, and raindrop-
aware 3DGS evaluation. For data preparation, we compare
1
arXiv:2510.17719v2  [cs.CV]  27 Dec 2025

<!-- page 2 -->
Real-world Dataset Collection
COLMAP
VGGT
Point Cloud + 
Camera Pose
Camera Pose
Pose Estimation & Point Cloud Initialization
Uformer
Single Image Raindrop Removal
3DGS Methods
3D Gaussians
with
SFM/VGGT
Points
3D Gaussians 
with Random 
Initial Points 
Raindrop-focused Images  
Background-focused Images  
Data Preparation
Data Processing
Raindrop-aware 3DGS Evaluation
Success
Fail
Fail
Restormer
IDT
3DGS-MCMC
WeatherGS
3DGS
Gaussian in the Wild
Figure 1. 3DGS Raindrop Reconstruction Benchmark Pipeline. We develop the first benchmark for comprehensively evaluating 3DGS
performance under raindrop conditions. The benchmark begins with real-world dataset collection, proceeds through data processing, and
ends with a raindrop- aware 3DGS evaluation. In particular, we assess how raindrop-induced image contamination reduces the number
of points available for cloud initialization and degrades camera pose estimation, and how these factors impact the performance of 3DGS
methods.
the effects of different types of images (raindrop-focused
and background-focused) on the subsequent reconstruction
process. During data processing, we evaluate the perfor-
mance of camera pose estimation and point cloud initializa-
tion, as well as single-image raindrop removal algorithms.
In raindrop-aware 3DGS evaluation, we consider methods
that may affect the performance of real-world raindrop re-
construction, such as raindrops and point cloud Gaussian
optimization.
As illustrated in Figure 2 (a-c), these synthetic datasets
are valuable but exhibit limitations, such as the same rain-
drop shape and position across different views. To over-
come these limitations, we collect a real-world 3D recon-
struction dataset captured under raindrop conditions. For
each scene, three aligned image sets are acquired: raindrop-
focused, background-focused, and rain-free ground truth.
This design enables evaluation of the full pipeline in real-
world scenarios as well as under different focus conditions.
As shown in Figure 2 (d), our RaindropGS dataset reflects
real-world conditions, featuring multiple focus settings and
a diverse range of raindrop characteristics.
Using the collected dataset, we process the images
(both raindrop-focused and background-focused) to ob-
tain the corresponding rain-free images, estimated camera
pose, and initialized point cloud. To analyze the impact
of raindrops on the real-world dataset collection, we use
COLMAP [31] and VGGT [35] to estimate the camera
pose and initialize the point cloud, enabling us to investi-
gate how sequence-based and feed-forward approaches in-
fluence the performance of 3DGS methods.
We include
three widely used deraining methods, Uformer [37], At-
GAN [28], Restormer [42], and IDT [38] in the rain-
drop removal stage, comparing the impact of different rain-
drop removal methods on subsequent 3DGS reconstruction
performance.
For the raindrop-aware 3DGS evaluation,
we integrate multiple 3DGS variants, including the origi-
nal 3DGS [11], WeatherGS [27], GS-W [43], and 3DGS-
MCMC [12], to evaluate the impact of different reconstruc-
tion strategies on raindrop-corrupted inputs. These methods
are evaluated under varying pre-processing pipelines and
focus conditions to assess their robustness and adaptability.
Through rigorous quantitative and qualitative analy-
ses, we evaluate the performance of state-of-the-art 3DGS
methods under raindrop conditions, as well as their pre-
processing stages. The results revealing their strengths, lim-
itations, and sensitivity to different pre-processing and fo-
cus settings. These findings not only benchmark the cur-
rent progress but also highlight key challenges and future
directions for improving 3DGS performance in real-world
adverse environments. Our main contributions are summa-
rized as follows:
• We introduce the first 3DGS benchmark for raindrop-
2

<!-- page 3 -->
(a) DerainNeRF
(b) WeatherGS
(c) DerainGS
Raindrop-focused
Ground-truth
Background-focused
Raindrops diversity
R 𝑅𝑒𝑎𝑙  𝑤𝑜𝑟𝑙𝑑
R 𝑅𝑎𝑖𝑛𝑑𝑟𝑜𝑝𝑠 𝑑𝑖𝑣𝑒𝑟𝑠𝑖𝑡𝑦
R 𝑅𝑎𝑖𝑛/𝐵𝐺− focused
𝒅 𝑹𝒂𝒊𝒏𝒅𝒓𝒐𝒑𝑮𝑺
Figure 2. Example of existing raindrop 3D datasets (DerainNeRF [17], WeatherGS [27], DerainGS [18]) and our RaindropGS Dataset. As
indicated by the red boxes, existing datasets exhibit the same raindrop distribution across different viewpoints; in contrast, the green boxes
illustrate the diversity of raindrop distributions in our dataset. For each viewpoint, we include both raindrop-focused and background-
focused images and provide corresponding clear images for 3DGS performance evaluation.
contaminated scenes, covering the complete pipeline
from unconstrained, raindrop-corrupted images to the fi-
nal 3D Gaussian reconstructions.
• We collect real-world paired 3D scenes, each containing
three aligned image sets: raindrop-focused, background-
focused, and rain-free ground truth, enabling comprehen-
sive evaluation of reconstruction quality across different
focus conditions.
• We comprehensively validate existing 3DGS methods on
our benchmark, revealing their strengths and limitations,
and providing insights into future research directions.
2. Related Work
3DGS Reconstruction under Raindrop Conditions
In
recent years, 3DGS has emerged as a powerful technique
for scene reconstruction. Unlike NeRF [20], it represents
scenes using a sparse set of 3D Gaussians, enabling real-
time rendering [1]. However, standard 3DGS benchmarks
assume clear input views, and performance often degrades
when images contain transient occlusions such as raindrops
on the lens [14, 18, 27].
To address this issue, several methods [17, 18, 27] have
been developed to improve 3D reconstruction in raindrop
scenes. WeatherGS [27] first generates raindrop masks to
identify occluded regions and then reconstructs clear scenes
by excluding these areas during 3D Gaussian Splatting.
Meanwhile, DerainGS [18] incorporates a dedicated image
enhancement module to remove raindrop artifacts and em-
ploys supervised Gaussian-ellipsoid fitting, achieving 3D
deraining in the final output. These methods are trained on
synthetic raindrops and deliver strong results under the as-
sumption of accurate camera pose estimation and reliable
point cloud initialization. However, they overlook the ini-
tial disruptions that real raindrops introduce to both pose
estimation and point cloud initialization, resulting in poor
generalization to real-world raindrop scenarios.
Raindrop Removal Methods
To mitigate lens occlusion
artifacts, single-image derain methods have been exten-
sively studied. Early works such as Raindrop Removal Net-
work [29] leverage visual attention to segment and inpaint
raindrop regions, while UMAN [32] extends this idea with
multiscale feature fusion. AtGAN [28] removes raindrops
using an attention-guided generative adversarial network
that identifies raindrop regions and reconstructs the miss-
ing background. More recently, transformer-based restora-
tion models (for example, Restormer [42], Uformer [37]
, DiT [24] and IDT [38]) demonstrate superior restoration
3

<!-- page 4 -->
under heavy rainfall by modeling long range dependencies.
However, these methods process each image independently
and do not enforce cross-view consistency, leading to re-
construction artifacts when applied as a preprocessing step
for 3D reconstruction.
3D Raindrop Reconstruction Benchmark and Dataset
Current 3DGS raindrop reconstruction methods focus pri-
marily on the Gaussian fitting stage and ignore the influ-
ence of earlier steps on the training process, such as camera
pose estimation and point cloud initialization. In addition,
they rely on synthetic training datasets created by Blender
on clear images [17, 18], which creates a significant domain
gap and prevents accurate evaluation in real-world condi-
tions. A few real-world datasets have tried to simulate rain
on camera lenses for stereo or small scale multi view se-
tups. DerainNeRF [17] captures stereo pairs by spraying
water onto a glass plate in front of a calibrated rig and pro-
vides binary raindrop masks. WeatherGS [27] extracts key
frames from publicly available rainy videos but does not
supply a ground truth reference. Overall, existing datasets
remain mostly synthetic and do not reflect real-world rain-
drop interference, and current algorithms overlook the early
stages of the pipeline, making their performance evaluation
under real conditions unreliable.
To address this challenge, we revisit the complete 3DGS
raindrop reconstruction pipeline and develop a benchmark
covering every stage: data preparation, data processing, and
raindrop-aware 3DGS evaluation. To evaluate current algo-
rithms and guide future research, we compile a real-world
dataset of eleven scenes.
3. RaindropGS Benchmark and Dataset
Figure 1 illustrates the overall pipeline of our bench-
mark RaindropGS, showing the process from unconstrained
raindrop-corrupted images to clean 3D Gaussian represen-
tations. It consists of data preparation, data processing, and
raindrop-aware 3DGS evaluation.
Unlike existing methods that rely on synthetic datasets
for
quantitative
evaluation,
we
collect
real-world
raindrop/ground-truth
image
pairs,
enabling
realistic
assessment of each stage in the 3DGS reconstruction
pipeline.
Our data processing employs structure-from-
motion (SfM) or feed-forward methods to estimate camera
poses and initialize the point cloud. The raindrop-aware
3DGS evaluation measures the robustness of existing 3DGS
methods under real-world challenges, including inaccurate
point cloud initialization and imperfect raindrop removal.
3.1. Data Preparation
We first describe our data collection process, including the
underlying optical refraction model and acquisition setup.
Table 1. Comparative raindrop 3D Reconstruction datasets. Com-
pared to existing collections, our dataset spans a greater vari-
ety of scenes and distinguishes between raindrop-focused and
background-focused captures.
Dataset
Scene count
Images
(Real)
GT
(Real)
Camera focus
Real Synthetic
Raindrop Background
DerainNeRF
3
2
20–25
×
×
✓
DerainGS
7
6
22–35
×
×
✓
RaindropGS
11
×
24–53
✓
✓
✓
We also present dataset statistics and comparisons with ex-
isting datasets.
Data Collection
To begin with, we consider a pinhole
camera model focused on the background plane [8].
In
the absence of optical distortion (e.g., caused by raindrops),
all scene elements located on the focal plane would appear
sharp and well-defined [7]. However, raindrops adhering
to a thin cover glass placed directly in front of the lens act
as miniature convex lenses, introducing optical distortion
and causing defocus [10]. When background rays intersect
a raindrop, they are refracted at the curved surface of the
drop decided by Snell Law [2]. In contrast, rays that do
not encounter any raindrop travel without deviation through
the imaging system to the sensor. Consequently, refracted
and non-refracted rays map to spatially distinct locations on
the image plane, illustrating how the presence of raindrops
directly affects the imaging distortion. Furthermore, since
raindrops don’t fully transmit light, the regions under rain-
drops exhibit localized intensity attenuation. This attenua-
tion produces visible artifacts.
By contrast, another alternative configuration in which
the camera is set to focus on the raindrop plane rather than
the background plane. In this configuration, the image plane
captures sharp, in-focus representations of the raindrop sur-
faces. Under these circumstances, more distant background
features, seen through each raindrop, appear as miniaturized
projections and are blurred in areas outside the raindrops.
To create the dataset, we use a pan-tilt sphere platform
to keep the camera stationary. We then follow a standard-
ized protocol grounded in optical refraction principles to en-
sure consistent camera alignment while allowing raindrops
to vary in location, shape and size. The setup consists of
two professional tripods with ball heads, a calibrated pres-
sure sprayer, and a glass plate with over 98 percent light
transmittance.
Data Statistics
We summarize our dataset in Table 1. The
dataset includes 11 real-world scenes, each containing 24
to 53 images captured under unconstrained raindrop condi-
tions. For every viewpoint, three aligned images are pro-
4

<!-- page 5 -->
vided: a raindrop-focused image, a background-focused
image, and a clean ground-truth image. The raindrops in
each viewpoint vary randomly in shape, number, and size,
closely replicating real-world conditions. In contrast, exist-
ing synthetic datasets for 3DGS lack representation of cam-
era focus effects on raindrop images and do not include di-
verse raindrop appearances across multiple viewpoints.
Focus Shift
During image capture (Figure 2(d)), rain-
drops adhering to the front glass shift the camera’s focal
plane. When many raindrops lie within the depth of field,
the camera focuses on them and the background becomes
blurred. Conversely, if only a few raindrops fall within the
focal region, the camera focuses on the background and the
raindrops appear out of focus. Most synthetic datasets ig-
nore focus variation and render both background and rain-
drops as sharply in focus, which may reduce 3DGS recon-
struction accuracy on real images. The RaindropGS dataset
explicitly addresses this issue by capturing each scene under
both raindrop-focused and background-focused conditions
to support more realistic 3DGS raindrop evaluation.
3.2. Data Processing
Our data processing pipeline consists of two main com-
ponents:
pose estimation and point cloud initialization,
and single-image raindrop removal pre-processing. Unlike
existing raindrop Gaussian splatting methods that assume
known camera poses and accurate point clouds, our bench-
mark directly estimates both the camera poses and an ini-
tial point cloud from the raindrop-affected images. This ap-
proach enables us to evaluate the robustness of the subse-
quent 3DGS reconstruction against potential errors in pose
estimation and inaccuracies in the initial point cloud. To
obtain a clean 3DGS reconstruction in the raindrop-aware
3DGS evaluation stage, we apply raindrop removal tech-
niques to the multi-view raindrop images.
Pose Estimation and Point Cloud Initialization
To es-
timate the camera pose and initialize the point cloud from
multi-view raindrop images, we employ COLMAP [31] and
Visual Geometry Grounded Transformer (VGGT) [35].
COLMAP is a robust tool capable of performing both
Structure-from-Motion (SfM) [30] and Multi-View Stereo
(MVS) [6]. We leverage SfM to estimate intrinsic and ex-
trinsic camera parameters and MVS to generate the initial
point cloud. However, raindrop interference often impedes
reliable feature matching across viewpoints. This results
in significant errors in estimated camera parameters and a
drastic reduction in initialized point cloud density. To over-
come the limitations of SfM, we employ VGGT as a com-
parative baseline. VGGT, a feed-forward unified method for
pose estimation and point cloud generation, is more robust
to raindrop interference due to its use of DINO [22].
In raindrop-focused scenes, the background is often
too blurred for reliable scene initialization, causing both
COLMAP [31] and VGGT [35] to fail. In certain scenes,
COLMAP may suffer a substantial reduction in the num-
ber of matchable camera poses due to degraded Correspon-
dence Search [31] performance, which ultimately leads to
reconstruction failure. To address cases where raindrop in-
terference and blur reduce the initial point cloud produced
by COLMAP and VGGT, we employ a random point-cloud
initialization strategy. Specifically, 100,000 points are ran-
domly initialized, matching the order of magnitude of point
counts obtained from ground-truth scenes.
Raindrop Removal
Since traditional 3DGS methods do
not incorporate raindrop removal capabilities, we em-
ploy four widely used single-image restoration mod-
els.
Uformer [37] applies non-overlapping window-
based self-attention and a multi-scale restoration modulator,
demonstrating superior capability in restoring details from
raindrop-affected and blurry images. Restormer [42] lever-
ages multi-Dconv head transposed attention and a gated-
Dconv feed-forward network to restore high-quality images,
while IDT [38] employs a dual Transformer with window-
and spatial-based designs for rain streak and raindrop re-
moval. AtGAN [28] removes raindrops using an attention-
guided generative adversarial network that identifies rain-
drop regions and reconstructs the missing background.
All raindrop removal methods are trained on the Rain-
drop Clarity dataset [10] to acquire raindrop removal capa-
bilities. Raindrop Clarity is a dataset containing both day-
time and nighttime image pairs, though we only use the day-
time data for training. Furthermore, Raindrop Clarity in-
cludes both background-focused and raindrop-focused im-
age pairs, making it well-suited for our task.
3.3. Raindrop-aware 3DGS Evaluation
With the estimated camera poses and initialized point cloud,
we proceed to evaluate four representative 3DGS meth-
ods: 3DGS [11], WeatherGS [27], GS-W [43], and 3DGS-
MCMC [12]. Among these, 3DGS [11] serves as a stan-
dard baseline for 3D Gaussian splatting. WeatherGS [27]
incorporates single-image raindrop removal, so we omit
the explicit raindrop removal step in its data processing
pipeline. GS-W [43] is specifically designed for challeng-
ing conditions and unconstrained image collections, mak-
ing it more robust to inconsistent multi-view inputs. 3DGS-
MCMC [12], on the other hand, does not rely on accurate
point cloud initialization. Each of the aforementioned meth-
ods has its own advantages, making their evaluation in our
benchmark both meaningful and insightful.
5

<!-- page 6 -->
Table 2. The quantitative evaluation of baseline approaches on the RaindropGS dataset. GS-W achieves the best performance with VGGT.
These 3DGS variants excel on background-focused dataset but show significantly lower performance on raindrop-focused dataset. We
highlight the best and second-best results for each metric.
3DGS
3DGS-MCMC
GS-W
WeatherGS
Focus
Metrics
Uformer
Restormer
IDT
Uformer
Restormer
IDT
Uformer
Restormer
IDT
x
RD-
focused
PSNR↑
13.894
13.876
13.958
15.109
15.005
14.994
16.099
15.400
15.873
13.070
SSIM ↑
0.346
0.345
0.350
0.383
0.380
0.383
0.512
0.484
0.511
0.307
LPIPS ↓
0.657
0.653
0.658
0.654
0.649
0.659
0.808
0.828
0.798
0.658
BG-
focused
PSNR ↑
17.906
17.741
18.094
18.219
18.148
18.239
19.123
19.074
17.818
17.124
SSIM ↑
0.478
0.469
0.480
0.482
0.477
0.483
0.555
0.550
0.507
0.428
LPIPS ↓
0.459
0.455
0.438
0.486
0.482
0.478
0.483
0.479
0.526
0.436
GT
VGGT-3DGS
VGGT-MCMC
VGGT-WeatherGS
Training Views
VGGT-GS-W
Figure 3. Qualitative Comparison among 3DGS methods: On the raindrop-focused dataset, the original 3DGS loses structural integrity,
while GS-W and 3DGS-MCMC retains scene completeness. WeatherGS faithfully reconstructs the scene after raindrop removal but fails
to correct background blur. On the background-focused dataset, all methods show moderate performance with artifacts.
4. Experiments
4.1. Implementation Details
We standardize all scene images to a resolution of 1024 *
576 for uniform comparison. During VGGT-based cam-
era pose estimation and point cloud initialization, we follow
the preprocessing protocol of VGGT [35], resizing each in-
put image to 518 * 518 before processing. Likewise, for
Restormer [42] and IDT [38], we downscale images to 128
* 128; for Uformer [37] and AtGAN [28], we downscale
images to 256 * 256, and then tile the results to restore the
original resolution.
For all 3DGS methods, we follow the official optimiza-
tion settings: 30,000 iterations for 3DGS, WeatherGS, and
3DGS-MCMC, and 70,000 iterations for GS-W.
For 3DGS-MCMC, we set the initial point cloud size to
100,000, based on the number of points that VGGT can ini-
tialize in our dataset. All models are implemented in Py-
Torch and trained on 8 NVIDIA RTX 3090 GPUs.
4.2. Quantitative Comparison
Table 2 compares the impact of background-focused (BG-
focused) and raindrop-focused (RD-focused) captures on
3DGS performance using VGGT [35].
For the original
3DGS method, raindrop-focused images exhibit a 4 dB
drop compared to background-focused images, due to back-
ground blur and light refraction. VGGT processes all scenes
but generates a point cloud with 0 points for raindrop-
focused images, for which we use a randomly initialized
point cloud.
Table 5 and Table 4 compares the performance of
VGGT [35] and COLMAP [31].
For camera pose esti-
mation, we use VGGT and COLMAP to estimate poses
on the ground-truth images and compare these estimates
with the poses obtained from the corresponding images.
VGGT yields more accurate camera pose estimates; both
methods accurately recover poses for background-focused
images but exhibit substantial performance degradation on
raindrop-focused images. In terms of point cloud initializa-
tion, VGGT outperforms COLMAP for background point
clouds, yet it fails to initialize the raindrop-focused dataset.
6

<!-- page 7 -->
Table 3. Comparison of the performance of different raindrop removal and restoration methods. BG-focused = background-focused, RD-
focused = raindrop-focused. All model weights are taken from Raindrop Clarity and evaluated under the officially recommended settings.
We highlight the best and second-best results for both background-focused and raindrop-focused datasets.
AtGAN
IDT
Restormer
Uformer
Metric
BG-focused
RD-focused
BG-focused
RD-focused
BG-focused
RD-focused
BG-focused
RD-focused
PSNR ↑
22.366
19.758
24.203
19.847
28.442
24.055
28.997
24.465
SSIM ↑
0.683
0.458
0.765
0.481
0.861
0.708
0.880
0.736
LPIPS ↓
0.233
0.365
0.157
0.344
0.108
0.235
0.106
0.218
VGGT-Uformer-MCMC
VGGT-Restormer-MCMC
GT
Background-VGGT-MCMC
Background-COLMAP-MCMC
GT
Training Views
Training Views
Figure 4. Qualitative Comparison across data pre-processing: Uformer and Restormer exhibit comparable raindrop removal capability.
While COLMAP, when successful in estimating camera poses, reconstructs more fine-grained details than VGGT.
Table 3 reports the performance of different derain-
ing and restoration methods on raindrop-affected images.
Uformer [37] and Restormer [42] show comparable results,
with Uformer [37] slightly outperforming Restormer [42]
in reconstruction metrics, while IDT [38] lags substantially
behind the other two methods.
For 3DGS methods, GS-W [43] with VGGT [35] and
Uformer [37] preprocessing achieves the best performance
(PSNR = 19.123), due to its adaptive optimization strategy
for handling occlusions and environmental variations in out-
door scenes. The second best is 3DGS-MCMC [12] (PSNR
= 18.239) on background-focused scenes with IDT [38] and
VGGT, which shows robustness to initialization.
4.3. Qualitative Comparison
Figures 3 and 4 show the qualitative results of 3DGS [11],
WeatherGS [27], GS-W [43], and 3DGS-MCMC [12],
along with their Uformer [37] and Restormer [42] out-
puts. Scenes with background-focused images exhibit gen-
erally good performance, while raindrop-focused images
pose significant reconstruction challenges due to the loss
of background details and the presence of undetected rain-
7

<!-- page 8 -->
Table 4. Quantitative analysis of the impact of camera pose estimation and point cloud initialization using COLMAP [31] and VGGT [35]
on 3DGS raindrop reconstruction performance. Three reconstruction methods, 3DGS [11], GS-W [43], and WeatherGS [27], are used for
testing. COLMAP failed entirely on the raindrop-focused dataset; only background-focused results are reported. “×” indicates reconstruc-
tion failures caused by unsuccessful camera pose estimation. We highlight the best and second-best results for each scene.
VGGT (3DGS)
COLMAP (3DGS)
VGGT (GS-W)
COLMAP (GS-W)
VGGT (WeatherGS)
COLMAP (WeatherGS)
Scene
PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓
corner
17.458
0.453
0.453
×
×
×
19.253
0.561
0.387
×
×
×
17.282
0.392
0.413
×
×
×
beartoy
17.249
0.625
0.435
21.452
0.785
0.343
19.474
0.711
0.415
18.623
0.688
0.451
15.611
0.584
0.442
19.993
0.723
0.332
bicycle
19.670
0.456
0.353
20.822
0.647
0.253
19.722
0.496
0.361
17.864
0.443
0.559
19.278
0.404
0.328
20.227
0.512
0.280
dustbin
16.780
0.374
0.568
19.414
0.516
0.431
19.205
0.484
0.691
×
×
×
17.069
0.347
0.467
×
×
×
flower
14.272
0.188
0.578
16.649
0.336
0.464
14.814
0.283
0.667
×
×
×
9.977
0.098
0.723
10.710
0.118
0.726
parkbear
17.479
0.435
0.464
×
×
×
18.953
0.507
0.490
×
×
×
17.271
0.376
0.441
×
×
×
popmart
17.404
0.631
0.506
16.376
0.665
0.473
17.632
0.714
0.500
15.776
0.680
0.608
15.995
0.582
0.496
15.202
0.598
0.459
rustydesk 21.255
0.484
0.429
21.741
0.615
0.377
21.498
0.543
0.499
17.813
0.500
0.704
20.721
0.419
0.365
21.726
0.539
0.322
stone
20.538
0.552
0.402
20.128
0.659
0.349
21.688
0.603
0.439
21.620
0.627
0.489
21.043
0.564
0.307
19.990
0.586
0.334
door
16.957
0.582
0.406
×
×
×
18.986
0.650
0.431
×
×
×
16.997
0.519
0.380
×
×
×
Table 5. Comparison of camera pose estimation (AUC@30 [33])
and the number of points in the point cloud between VGGT and
COLMAP on background- and raindrop-focused datasets. BG-
focused = background-focused, RD-focused = raindrop-focused.
VGGT
COLMAP
BG-focused
RD-focused
BG-focused
RD-focused
AUC@30
0.91
0.34
0.79
0.17
Num. of Points
69401.11
x
5476.89
302.50
drops. Uformer and Restormer outperform WeatherGS in
restoring raindrop-degraded images. Among 3DGS vari-
ants, 3DGS suffers from detail loss, GS-W introduces ar-
tifacts, and 3DGS-MCMC offers improved quality over
3DGS. Weather-GS exhibits considerable blurriness due to
limitations in handling raindrop-degraded images but shows
strong multi-view consistency.
In scenes that both COLMAP and VGGT successfully
reconstruct, COLMAP recovers finer geometric and pho-
tometric detail.
However, under raindrop interference,
COLMAP fails to reconstruct any raindrop-focused scenes
from its estimated camera poses and point clouds. By con-
trast, VGGT demonstrates superior robustness.
4.4. Discussion
Our benchmark reveals how raindrop-induced degradations
affect the entire 3DGS reconstruction pipeline. We summa-
rize three key insights as follows.
Impact of Different Focus Conditions As shown in
Table 5, raindrop-affected images significantly degrade
the accuracy of camera pose estimation and point cloud
initialization, especially under raindrop-focused condi-
tions. Although VGGT provides improved robustness on
background-focused images, it still fails easily when strong
blur and focal-plane shifts occur in raindrop-focused views.
These observations suggest that future methods must ex-
plicitly handle the geometric ambiguities introduced by fo-
cus shifts and adherent raindrops. Moreover, downstream
3D reconstruction should incorporate strategies that can tol-
erate inaccurate poses and sparse or noisy point clouds,
as these errors propagate throughout the entire 3DGS op-
timization process.
Impact of Raindrop Removal Methods Table 2 shows that
applying robust raindrop removal methods can improve 3D
reconstruction quality. However, a substantial performance
gap remains, indicating that raindrop removal alone is insuf-
ficient for reliable reconstruction. In practice, downstream
3DGS methods must remain effective even when the re-
moval results are imperfect, making it necessary to lever-
age multi-view information to compensate for incomplete
or inaccurate restoration. Moreover, the removal quality
varies across views, introducing multi-view inconsistencies
that propagate into 3DGS optimization and often manifest
as floaters or structural artifacts. Addressing these incon-
sistencies and improving robustness to imperfect raindrop
removal will be important directions for future research.
Impact of 3DGS Methods Table 2 indicates the perfor-
mance of various 3DGS approaches under raindrop-affected
scenes. Overall, GS-W [43] achieves the best reconstruc-
tion quality, largely due to its robustness to inconsistent or
transient objects. In our setting, raindrops naturally behave
as view-inconsistent elements, and GS-W’s design allows it
to better tolerate such variability. This observation suggests
that future research should explore reconstruction strategies
that explicitly account for view inconsistency, transient arti-
facts, or incomplete observations, enabling 3DGS pipelines
to remain stable even under severe multi-view degradations.
5. Conclusion
In summary, RaindropGS offers a novel benchmark for
evaluating 3DGS methods under real-world raindrop con-
ditions. By addressing the limitations of previous synthetic
8

<!-- page 9 -->
datasets, we provide a more effective assessment of 3DGS
performance in practical,
unconstrained environments.
Through the evaluation of multiple 3DGS variants, we
identify the accumulated errors in camera pose estimation,
point cloud initialization, raindrop removal, and 3DGS
methods. Our findings highlight the strengths and weak-
nesses of existing approaches, offering insights into their
performance under raindrop-corrupted conditions. These
results underscore the need for more robust techniques
to handle diverse raindrop characteristics and multi-view
inconsistencies.
RaindropGS not only contributes to the
advancement of 3D reconstruction under challenging condi-
tions but also lays the foundation for future research aimed
at improving 3DGS performance in real-world applications.
References
[1] Yanqi Bao, Tianyu Ding, Jing Huo, Yaoli Liu, Yuxin Li,
Wenbin Li, Yang Gao, and Jiebo Luo. 3d gaussian splatting:
Survey, technologies, challenges, and opportunities. IEEE
Transactions on Circuits and Systems for Video Technology,
2025. 3
[2] Max Born and Emil Wolf. Principles of optics: electromag-
netic theory of propagation, interference and diffraction of
light. Elsevier, 2013. 4
[3] Wenhui Chang, Hongming Chen, Xin He, Xiang Chen,
and Liangduo Shen.
Uav-rain1k: A benchmark for rain-
drop removal from uav aerial imagery. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 15–22, 2024. 1
[4] Qiyu Dai, Xingyu Ni, Qianfan Shen, Wenzheng Chen, Bao-
quan Chen, and Mengyu Chu. Rainygs: Efficient rain synthe-
sis with physically-based gaussian splatting. In Proceedings
of the Computer Vision and Pattern Recognition Conference,
pages 16153–16162, 2025. 1
[5] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A
Efros, and Xiaolong Wang. Colmap-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 20796–20805,
2024. 1
[6] Yasutaka Furukawa, Carlos Hern´andez, et al.
Multi-view
stereo: A tutorial. Foundations and trends® in Computer
Graphics and Vision, 9(1-2):1–148, 2015. 5
[7] Zhixiang Hao, Shaodi You, Yu Li, Kunming Li, and Feng Lu.
Learning from synthetic photorealistic raindrop for single
image raindrop removal. In Proceedings of the IEEE/CVF
International Conference on Computer Vision Workshops,
pages 0–0, 2019. 4
[8] Richard Hartley and Andrew Zisserman. Multiple view ge-
ometry in computer vision.
Cambridge university press,
2003. 4
[9] Zhisheng Huang, Peng Wang, Jingdong Zhang, Yuan Liu,
Xin Li, and Wenping Wang.
3r-gs: Best practice in op-
timizing camera poses along with 3dgs.
arXiv preprint
arXiv:2504.04294, 2025. 1
[10] Yeying Jin, Xin Li, Jiadong Wang, Yan Zhang, and Malu
Zhang. Raindrop clarity: A dual-focused dataset for day and
night raindrop removal. In European Conference on Com-
puter Vision, pages 1–17. Springer, 2024. 4, 5
[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 5, 7, 8
[12] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar,
Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splat-
ting as markov chain monte carlo. Advances in Neural In-
formation Processing Systems, 37:80965–80986, 2024. 2, 5,
7
[13] Dmytro Kotovenko, Olga Grebenkova, and Bj¨orn Ommer.
Edgs: Eliminating densification for efficient convergence of
3dgs. arXiv preprint arXiv:2504.13204, 2025. 1
[14] Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc
Pollefeys, and Torsten Sattler. Wildgaussians: 3d gaussian
splatting in the wild. arXiv preprint arXiv:2407.08447, 2024.
3
[15] Xin Li, Yeying Jin, Xin Jin, Zongwei Wu, Bingchen Li, Yufei
Wang, Wenhan Yang, Yu Li, Zhibo Chen, Bihan Wen, et al.
Ntire 2025 challenge on day and night raindrop removal for
dual-focused images: Methods and results. In Proceedings
of the Computer Vision and Pattern Recognition Conference,
pages 1172–1183, 2025. 1
[16] Yizhou Li, Yusuke Monno, and Masatoshi Okutomi. Dual-
pixel raindrop removal. IEEE Transactions on Pattern Anal-
ysis and Machine Intelligence, 2024. 1
[17] Yunhao Li, Jing Wu, Lingzhe Zhao, and Peidong Liu. De-
rainnerf: 3d scene estimation with adhesive waterdrop re-
moval. In 2024 IEEE International Conference on Robotics
and Automation (ICRA), pages 2787–2793. IEEE, 2024. 1,
3, 4
[18] Shuhong Liu, Xiang Chen, Hongming Chen, Quanfeng Xu,
and Mingrui Li. Deraings: Gaussian splatting for enhanced
scene reconstruction in rainy environments.
In Proceed-
ings of the AAAI Conference on Artificial Intelligence, pages
5558–5566, 2025. 1, 3, 4
[19] Xianqiang Lyu, Hui Liu, and Junhui Hou. Rainyscape: Un-
supervised rainy scene reconstruction using decoupled neu-
ral rendering. In Proceedings of the 32nd ACM International
Conference on Multimedia, pages 10920–10929, 2024. 1
[20] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
3
[21] Zhongyan Niu, Zhen Tan, Jinpu Zhang, Xueliang Yang, and
Dewen Hu. Hgsloc: 3dgs-based heuristic camera pose refine-
ment. In 2025 IEEE International Conference on Robotics
and Automation (ICRA), pages 1–7. IEEE, 2025. 1
[22] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervision.
arXiv preprint arXiv:2304.07193, 2023. 5
[23] Weihong Pan, Xiaoyu Zhang, Hongjia Zhai, Xiaojun Xi-
ang, Hanqing Jiang, and Guofeng Zhang. Liberated-gs: 3d
9

<!-- page 10 -->
gaussian splatting independent from sfm point clouds.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 26675–26685, 2025. 1
[24] William Peebles and Saining Xie. Scalable diffusion models
with transformers. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 4195–4205,
2023. 3
[25] Ivana Petrovska and Boris Jutzi. Impact of rain on 3d recon-
struction with multi-view stereo, neural radiance fields and
gaussian splatting. ISPRS Annals of the Photogrammetry,
Remote Sensing and Spatial Information Sciences, 10:169–
176, 2025. 1
[26] Ivana Petrovska and Boris Jutzi.
Seeing beyond vegeta-
tion: A comparative occlusion analysis between multi-view
stereo, neural radiance fields and gaussian splatting for 3d re-
construction. ISPRS Open Journal of Photogrammetry and
Remote Sensing, page 100089, 2025. 1
[27] Chenghao Qian, Yuhu Guo, Wenjing Li, and Gustav
Markkula. Weathergs: 3d scene reconstruction in adverse
weather conditions via gaussian splatting.
arXiv preprint
arXiv:2412.18862, 2024. 1, 2, 3, 4, 5, 7, 8
[28] Rui Qian, Robby T Tan, Wenhan Yang, Jiajun Su, and Jiay-
ing Liu. Attentive generative adversarial network for rain-
drop removal from a single image. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 2482–2491, 2018. 2, 3, 5, 6
[29] Rui Qian, Robby T Tan, Wenhan Yang, Jiajun Su, and Jiay-
ing Liu. Attentive generative adversarial network for rain-
drop removal from a single image. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 2482–2491, 2018. 3
[30] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 5
[31] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 2, 5, 6, 8
[32] Ming-Wen Shao, Le Li, De-Yu Meng, and Wang-Meng Zuo.
Uncertainty guided multi-scale attention network for rain-
drop removal from a single image. IEEE Transactions on
Image Processing, 30:4828–4839, 2021. 3
[33] Jianyuan Wang, Christian Rupprecht, and David Novotny.
Posediffusion: Solving pose estimation via diffusion-aided
bundle adjustment. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, pages 9773–9783,
2023. 8
[34] Jiaxu Wang, Ziyi Zhang, Junhao He, and Renjing Xu. Pfgs:
High fidelity point cloud rendering via feature splatting. In
European Conference on Computer Vision, pages 193–209.
Springer, 2024. 1
[35] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 2, 5, 6, 7, 8
[36] Weijia Wang. Real-time fast 3d reconstruction of heritage
buildings based on 3d gaussian splashing.
In 2024 IEEE
2nd International Conference on Sensors, Electronics and
Computer Engineering (ICSECE), pages 1014–1018. IEEE,
2024. 1
[37] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang
Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general
u-shaped transformer for image restoration. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 17683–17693, 2022. 2, 3, 5, 6, 7
[38] Jie Xiao, Xueyang Fu, Aiping Liu, Feng Wu, and Zheng-
Jun Zha. Image de-raining transformer. IEEE transactions
on pattern analysis and machine intelligence, 45(11):12978–
12995, 2022. 2, 3, 5, 6, 7
[39] Shaodi You, Robby T Tan, Rei Kawakami, and Katsushi
Ikeuchi. Adherent raindrop detection and removal in video.
In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 1035–1042, 2013. 1
[40] Shaodi You, Robby T Tan, Rei Kawakami, Yasuhiro
Mukaigawa, and Katsushi Ikeuchi. Adherent raindrop mod-
eling, detection and removal in video. IEEE transactions on
pattern analysis and machine intelligence, 38(9):1721–1733,
2015. 1
[41] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19447–19456,
2024. 1
[42] Syed Waqas Zamir, Aditya Arora, Salman Khan, Mu-
nawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang.
Restormer: Efficient transformer for high-resolution image
restoration. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 5728–5739,
2022. 2, 3, 5, 6, 7
[43] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li,
Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d
gaussian splatting for unconstrained image collections. In
European Conference on Computer Vision, pages 341–359.
Springer, 2024. 1, 2, 5, 7, 8
[44] Chengxuan Zhu, Renjie Wan, Yunkai Tang, and Boxin Shi.
Occlusion-free scene recovery via neural radiance fields. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 20722–20731, 2023. 1
10
