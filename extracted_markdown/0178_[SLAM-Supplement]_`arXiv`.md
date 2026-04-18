<!-- page 1 -->
EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images
Wangbo Yu1,2*, Chaoran Feng1*, Jiye Tang3, Jiashu Yang4, Zhenyu Tang1, Xu Jia4, Yuchao Yang1,
Li Yuan1,2† and Yonghong Tian1,2†
1Peking University, 2Peng Cheng Laboratory
3University of Science and Technology of China
4Dalian University of Technology
Abstract
3D Gaussian Splatting (3D-GS) has demonstrated excep-
tional capabilities in synthesizing novel views of 3D scenes.
However, its training is heavily reliant on high-quality im-
ages and precise camera poses. Meeting these criteria can
be challenging in non-ideal real-world conditions, where
motion-blurred images frequently occur due to high-speed
camera movements or low-light environments. To address
these challenges, we introduce Event Stream Assisted Gaus-
sian Splatting (EvaGaussians), a novel approach that har-
nesses event streams captured by event cameras to facili-
tate the learning of high-quality 3D-GS from blurred im-
ages. Capitalizing on the high temporal resolution and dy-
namic range offered by event streams, we seamlessly inte-
grate them into the initialization and optimization of 3D-
GS, thereby enhancing the acquisition of high-fidelity novel
views with intricate texture details. To remedy the absence
of evaluation benchmarks incorporating both event streams
and RGB frames, we present two novel datasets compris-
ing RGB frames, event streams, and corresponding camera
parameters, featuring a wide variety of scenes and vari-
ous camera motions. We then conduct a thorough evalu-
ation of our method, comparing it with leading techniques
on the provided benchmark. The comparison results reveal
that our approach not only excels in generating high-fidelity
novel views, but also offers faster training and inference
speeds. Video results are available at the project page.
1. Introduction
Novel view synthesis from 2D image collections has pre-
sented a persistent challenge within the field of computer
vision and computer graphics. This task stands as a fun-
damental component in various vision applications, such
as virtual reality [16, 36, 45, 48], robotics navigation [32,
*These authors contributed equally to this work.
†Corresponding author.
PSNR: 21.93 dB
ATE: 1.083
FPS:  92.30
BAD-Gaussians
[ECCV2024]
EvDeblurNeRF
[CVPR2024]
PSNR: 23.23 dB
ATE: 2.531 
FPS: 3.72
Ground Truth
City Block Scene
Novel View Synthesis
Data Inputs
EvaGaussians
[Ours]
PSNR: 24.41 dB
ATE: 0.879
FPS:  93.70
w. event + w. pose opt.
w. event + w/o. pose opt.
w/o. event + w. pose opt.
Event Inputs
Blurry Inputs
Figure 1. EvaGaussians integrates blurry images and event streams
to reconstruct sharp 3D-GS for novel view synthesis.
46, 54], scene understanding [14, 19, 20], and many others,
thereby prompting significant research efforts over the last
decades. Amid pioneering works,
3D Gaussian Splatting (3D-GS) [13] achieves notable
success in generating high-fidelity novel views. It learns
3D Gaussians with the lightweight learnable parameters,
and leverages a tile-based rasterization technique to render
arXiv:2405.20224v3  [cs.CV]  6 Dec 2024

<!-- page 2 -->
novel views, thereby surpassing NeRFs [26] in both train-
ing and rendering efficiency. However, the optimization of
3D-GS heavily relies on accurate camera poses and point
cloud initialization produced by COLMAP [34], which ne-
cessitates high-quality images without blurring and with ad-
equate lighting. Fulfilling such conditions can be challeng-
ing in real-world situations.
For example, in UAVs and
robotics, rapid camera movement is common when captur-
ing images or recording videos, which often result in signifi-
cant motion blur. The mismatched features between blurred
images can lead to inaccurate pose calibrations and point
cloud initialization, thereby hindering the training process
of 3D-GS.
Recent studies have demonstrated the significant poten-
tial of event-based cameras in alleviating motion blur in im-
ages captured by conventional frame-based cameras [11, 12,
18, 29, 36, 37, 49]. Serving as an innovative bio-inspired
visual sensor, event cameras asynchronously report the log-
arithmic intensity changes of each pixel captured, and can
record higher temporal resolution and dynamic range data
in contrast to conventional cameras.
Motivated by this,
prior works [2, 30], have attempted to leverage the event
streams captured by event cameras to supervise the train-
ing of NeRFs. However, achieving real-time rendering and
synthesizing high-fidelity novel views with intricate details
poses substantial challenges for these methods.
To address these challenges, we introduce Event Stream
Assisted Gaussian Splatting (EvaGaussians), which lever-
ages the event streams captured by event cameras to en-
hance the learning of high-quality 3D-GS from motion-
blurred images. Harnessing the exceptional temporal res-
olution and dynamic range offered by event streams, we
use them to assist in the initialization of 3D-GS, and in-
corporate them to jointly optimize 3D-GS and camera tra-
jectories of blurry images through a blur reconstruction loss
and an event reconstruction loss. Due to the geometric am-
biguity caused by blurry images, we further propose two
event-assisted depth regularization terms to stabilize the ge-
ometry of 3D-GS. Through optimizing the 3D-GS in a pro-
gressive manner, our method can recover a high-quality 3D-
GS that facilitates the real-time generation of high-fidelity
novel views. To summarize, our contributions can be delin-
eated as follows:
• We propose Event Stream Assisted Gaussian Splatting
(EvaGaussians), a framework tailored for reconstruct-
ing a high-quality 3D-GS from motion-blurred images
with the assistance of event camera. Once trained, our
method is capable of recovering intricate details of the in-
put blurry images and allows high-fidelity real-time novel
view synthesis.
• We contribute two novel datasets, including a synthetic
dataset containing diverse scenes with various scales, and
a real-world dataset captured by the color DAVIS346
event camera [1], both feature various camera motions.
We believe they will set a benchmark for future re-
searches.
• We conduct a comprehensive evaluation of the proposed
method and compare it with several strong baselines. The
results reveal that our approach not only excels in gen-
erating high-fidelity novel views but also provides faster
training and inference speeds.
2. Related Works
2.1. Reconstructing 3D Scene from Blurry Images
Reconstructing a high-quality 3D Scene typically requires
high-fidelity, sharp images as supervision.
However,
motion-blurred images often occur in real world scenarios,
thus hindering accurate reconstruction of 3D scenes. Sev-
eral studies have been proposed to address this issue. For
example, Deblur-NeRF [24] and DP-NeRF [17] attempted
to learn a blur formation kernel to model the image blur-
ring process. BAD-NeRF [41] further physically modeled
the blurry images formation process, and adopted a bundle-
adjustment strategy to jointly optimize NeRF parameters
and the camera poses during the exposure time.
These
NeRF-based methods lacked real-time rendering capabili-
ties and suffered from extended training times. With the
rapid advancement of 3D-GS, a concurrent work, BAD-
Gaussians [52], proposed to utilize 3D-GS as representa-
tion and follow the blur modeling and bundle-adjustment
strategy adopted in [41] to achieve deblurring reconstruc-
tion. Although it achieved real-time rendering and faster
convergence compared with prior works, it still struggled
to handle severely blurred images in which COLMAP [34]
will fail to produce the initial point clouds. Furthermore,
it employed linear interpolation between the start and end
camera poses to model camera trajectory during exposure
time, necessitating careful selection of poses for more sta-
ble optimization.
2.2. Reconstructing 3D Scene from Event Streams
Motivated by the exceptional properties offered by event
cameras, several studies attempted to reconstruct 3D scenes
from event streams captured by event cameras, particularly
in low-light conditions with fast camera motion. For exam-
ple, EventNeRF [33], Ev-NeRF [10] and other concurrent
works [43, 44, 47, 48, 51] explored the reconstruction of
a 3D representation from a rapidly moving event camera.
Robust e-NeRF [22] and its variants [23] further extended
this task to the more challenging scenario of non-uniform
camera motion, taking into account the refractory period of
event cameras. These methods were typically designed to
be supervised solely by information captured from a sin-
gle event camera. Recently, E-NeRF [15], E2NeRF [30],
and EvDeblurNeRF [3] proposed to jointly utilize event

<!-- page 3 -->
streams captured by event cameras and motion-blurred im-
ages captured by standard frame-based cameras to recon-
struct a NeRF representation. Compared to methods that
rely solely on event cameras, these methods can recover ac-
curate color details. Additionally, in contrast to RGB-only
methods, they are better at handling motion blur. However,
these NeRF-based methods suffer from long training and
inference times, and face instability during training, which
limit their further application.
3. Method
3.1. Preliminary
Event camera is a type of bio-inspired sensor that can asyn-
chronously record intensity changes [7]. In contrast to con-
ventional cameras that are restricted to sequentially produce
frames at a fixed frame rate, event cameras asynchronously
trigger events in each pixel when their intensity change ex-
ceeds a constant threshold, featuring properties such as low
latency and high dynamic range. Formally, let Ixy(t) de-
note the instantaneous intensity at pixel coordinate (x, y)
at time t, and Lxy(t) denotes its logarithm.
An event
p = ±1 will be triggered whenever the change of Lxy(t)
surpasses the threshold c, where the polarity represents the
direction (increase or decrease) of changes. Let δt0(t) be
the impulse function at time t0 with a unit integral, the
event can therefore be expressed as a continuous-time sig-
nal exy(t) = p δt0(t), where t0 signifies the time at which
the event occurs. Then, the proportional intensity change
during a time interval [s, t] can be computed as the integral
of events that occurred between times s and t, expressed
as Exy(t) =
R t
s exy(h)dh. Given that each pixel can be
treated separately in the event camera, the subscripts can be
omitted:
E(t) =
Z t
s
e(h)dh.
(1)
We can then represent the logarithmic intensity change as:
L(t)−L(s) = c E(t), rewrite as L(t) = L(s)+c E(t), and
subsequently obtain the actual intensity change:
I(t) = I(s) · exp(c E(t)).
(2)
Therefore, when an image I(s) is captured at time s, and
the event stream is recorded during the time interval [s, t],
the image I(t) can be obtained by warping I(s) using Eq. 2.
3.2. Event-assisted Initialization
The optimization of 3D-GS requires camera calibration and
point cloud initialization using COLMAP [34]. However,
this process can fail when dealing with images that have
significant motion blur. Motion-blurred images are resulted
from camera movements during the exposure time, which
can be mathematically represented as:
B = 1
τ
Z s+τ/2
s−τ/2
I(t)dt,
(3)
where B denotes a captured blurry image, which is equiva-
lent to averaging the instantaneous latent images I(t) during
the exposure time [s −τ/2, s + τ/2].
To obtain initial camera poses and point clouds for 3D-
GS optimization, we first preprocess the motion-blurred
images using the Event-based Double Integral (EDI) [29]
model, which can be derived through substituting Eq. 2 into
Eq. 3:
B = I(s) · 1
τ
Z s+τ/2
s−τ/2
exp(c E(t))dt.
(4)
Given the predefined threshold c, a blurry image B, and the
recorded event stream E(t), the EDI model (Eq. 4) allows
the derivation of I(s), following which the latent image I(t)
at any moment within the exposure time can be estimated
through Eq. 2. As shown in Figure. 2(A), given a total of
K blurry images {Bj}K
j=1, for each of them, we uniformly
sample n time stamps during their exposure time to obtain
a series of EDI-estimated latent images rich in texture fea-
tures, denoted as {Ii}n
i=1, then obtain their poses {Pi}n
i=1
and the initial point cloud of the scene using COLMAP [34].
After initialization, a straightforward approach to opti-
mizing the 3D-GS is to use the EDI-estimated latent images
and poses as supervision. However, although these images
provide more texture features than the original blurry im-
age, they still do not fully recover the ideal latent image and
exhibit relatively low visual quality, which also introduces
inaccuracies into the camera poses, thereby leading to un-
satisfactory optimization results. To more robustly recover
a sharp 3D-GS from motion-blurred images, we propose to
harness the advantages of event streams and seamlessly in-
tegrate them into the optimization process of 3D-GS.
3.3. Event-assisted Bundle Adjustment
As introduced in Eq. 3, during the exposure time, a motion-
blurred image can be decomposed into a series of latent
images along a specific camera trajectory, which can be
roughly approximated by the EDI-produced camera poses
{Pi}n
i=1 according to Eq. 4. Motivated by this, we jointly
optimize these camera poses and the 3D-GS attributes in a
bundle adjustment manner [41] to simultaneously recover
the blur-formation camera trajectories and a sharp 3D-GS.
As shown in Figure. 2(B), we add each of the EDI-produced
camera poses a learnable offset {di}n
i=1 as correction pa-
rameters, resulting a learnable camera trajectory {ePi}n
i=1,
where ePi = Pi + di. In each training iteration, we si-
multaneously render n images {eIi}n
i=1 from the 3D-GS

<!-- page 4 -->
. . .
. . .
. . .
(A) Event-assisted Initialization
Sobel 
Operator
X
Y
Loss
Computation
(B) Event-assisted Bundle Adjustment
(C.2) Intensity-aware Depth
Regularization Loss
Intensity 
Image
Rasterize
Rendered Intensity 
Image
(C.1) Intensity Reconstruction Loss
Gradient Flow
Forward Flow (exposure time)
Projection 
(3)
T
SE
Î
n
...
Event Stream
Camera Poses
n
. . .
+
+
+
+
Differentiable
Rasterization
Adaptive
Density Control
Blurry View
Sampled  View
Intensity 
Reconstruction Loss
(C.1) 
Event Reconstruction Loss
ESIM
Simulated Event Maps
...
Real Event Maps
...
Blur Reconstruction Loss
Real Blurry Image
Simulated Blurry Image
Intensity-aware Depth
Regularization Loss
(C.2)
...
Initialization
Sfm Points
COLMAP
Intensity Image
Rendered Depth 
Blurry Image
Formation
Camera Poses
. . .
…
…
Forward Flow (sampled time)
n
Rasterize
Figure 2. Overview of EvaGaussians. We use event streams to assist in the initialization of 3D-GS and incorporate them to jointly
optimize both 3D-GS and the camera trajectories of blurry images during the exposure time, utilizing a blur reconstruction loss and an
event reconstruction loss. Additionally, we propose two event-assisted depth regularization terms to stabilize the geometry of 3D-GS.
along the camera trajectory of the blurry view, and sim-
ulate the formation of motion-blurred images using a dis-
crete approximation of Eq 3, expressed as eB = 1
n
Pn
i=1 eIi.
Consequently, for a total of K real-captured blurry images
{Bj}K
j=1, we can obtain their simulated versions {eBj}K
j=1
through each corresponding learnable camera trajectory.
Blur Reconstruction Loss. With the simulated blurry im-
ages, we use the real captured blurry images {Bj}K
j=1 to
serve as image level supervision.
Specifically, for each
blurry image Bj and its simulated version eBj, we employ a
blur reconstruction loss to minimize their photometric error,
expressed as
Lblur = (1−λ1)·∥Bj−eBj∥1+λ1·D-SSIM(Bj, eBj). (5)
The formulation of blur reconstruction loss is the same as in
the original 3D-GS [13], it differs in utilizing blurry images
as supervision and jointly optimizing the 3D-GS attributes
and the camera trajectories, thus facilitating an initial de-
blurring reconstruction of 3D-GS.
Event Reconstruction Loss.
Leveraging the abundant
high-frequency information offered by the event streams,
we further adopt an event reconstruction loss to aid in 3D-
GS optimization. Specifically, we uniformly divide the ex-
posure time into m = n −1 intervals, each with a dura-
tion of
τ
m. Subsequently, we integrate the recorded event
stream along these time intervals using Eq. 1, resulting in
m event maps {Ei}m
i=1 to serve as event level supervision.
During training, for the j-th blurry view, we convert the
rendered image sequence {eIi}n
i=1 on the camera trajectory
into event maps {eEi}m
i=1, using a differentiable event sim-
ulator [9, 31], and constrain the discrepancies between the
simulated event maps and the ground truth event maps, ex-
pressed as:
Levent = 1
m
m
X
i=1
∥Ei −eEi∥1.
(6)
The event reconstruction loss further aids in recovering a
sharp 3D-GS with improved texture details.
3.4. Event-assisted Geometry Regularization
The blurry color images are captured only during the expo-
sure time and are much sparser than the event stream. Rely-
ing on such low-quality image-level supervision may cause
the 3D-GS to overfit on the training images, resulting in
significant floaters and inferior geometry, which affects the
quality of novel view synthesis. Leveraging Eq. 1 and Eq. 4,
given the continuously recorded event streams E(t), we can

<!-- page 5 -->
derive continuous grayscale intensity images G(t) that are
rich in geometric information and can function beyond the
exposure time. Motivated by this, we further propose two
event-assisted geometry regularization terms to aid in 3D-
GS training.
Intensity Reconstruction Loss.
As shown in Fig-
ure. 2(C.1), during training, we randomly sample contin-
uous time t between the interval of two adjacent blurry im-
age, and derive the grayscale intensity image G(t) using
Eq. 2. We then minimize the difference between it and the
rendered intensity image from 3D-GS, expressed as:
Lint = (1−λ2)·∥G(t)−eG(t)∥1+λ2·D-SSIM(G(t), eG(t)),
(7)
where eG(t) is converted from the colored render result.
Intensity-aware Depth Regularization Loss. As shown
in Figure. 2(C.2), to further improve the geometry of 3D-
GS, inspired by [4, 8], we adopt an intensity-aware depth
regularization loss during training, defined as:
Ldepth = 1
N
X
x,y
( |∂x eDxy(t)|e−β|∂xGxy(t)|
+ |∂y eDxy(t)|e−β|∂yGxy(t)| ),
(8)
where eD(t) is the rendered depth map, (x, y) denotes the
pixel location, N is the total number of pixels, and β is set
to 2 in our experiments. The horizontal and vertical gradi-
ents are calculated by applying convolution operations with
5 × 5 Sobel kernels [39]. This regularization is founded on
the observation that depth transitions in an image often cor-
respond to changes in intensity. Therefore, it ensures that
the spatial variation of depth closely matches that of the in-
tensity image, thereby reducing geometric artifacts at object
boundaries.
The total loss function is the combination of the above
losses, defined as:
Ltotal =λblurLblur + λeventLevent
+ λintLint + λdepthLdepth.
(9)
4. Experiments
4.1. Implementation Details
Progressive Training.
We implemented EvaGaussians
based on the official code of 3D-GS [13]. The training pro-
cess spans 50,000 iterations, with an event reconstruction
loss introduced after a 3,000-iteration warmup and we omit
the densification process to streamline and simplify the sub-
sequent optimization. Additionally, we adopt a coarse-to-
fine training strategy, starting with rendering at a low res-
olution (0.3× downsampling in the early 30% iterations)
and progressively increasing the size of the rendered views
to full resolution. All experiments were conducted using a
single NVIDIA RTX 4090 GPU.
Hyperparameter Setting. During the training process, we
set λ1 = 0.2, λblur = 1.0, λdepth = 1.0e−2, λevent =
5.0e−3 and λint = 1.0e−3 for the loss function, and used
n = 9 for the number of poses to be optimized during the
exposure time. In implementing the loss Levent, we con-
figured the positive threshold as cpos = 0.25 and the nega-
tive threshold as cneg = 0.25 for synthetic scenes, and set
cpos = 0.197 and cneg = 0.241 for real scenes.
4.2. Datasets
To facilitate a comprehensive evaluation, we introduce two
novel datasets, with an overview provided below. Detailed
information is presented in the supplementary.
EvaGaussians-Blender Dataset. We construct a synthetic
dataset covering a variety of scene scales, coupling with di-
verse camera trajectories and event data. For large-scale
scenes, we employ Blender to craft five distinct scenes, in-
cluding city blocks and natural landscapes. For medium-
scale scenes, we craft three scenes using Blender, and re-
design the camera trajectories of four scenes from DeblurN-
eRF [25].
For object-level scenes, we create six scenes
based on the NeRF-synthetic [26] dataset.
We simulate
motion blur by manually placing multi-view cameras, ran-
domly adjusting camera poses, and performing linear inter-
polation between the original and perturbed positions for
each view. The images are rendered from these interpo-
lated poses and blended in RGB space to produce the final
blurry images. The corresponding event streams are sim-
ulated using ESIM [31] and V2E [9]. The resulting large-
scale and medium-scale scenes comprise 35 views of blurry
images along with their corresponding event data, whereas
the object-level scenes feature 100 views of blurry images.
EvaGaussians-DAVIS Dataset.
We manually recorded
five real-world scenes using the Color DAVIS346 event
camera [38], which has a resolution of 346×260 pixels and
an exposure time of 100 milliseconds for the RGB frames.
The dataset includes three object-level scenes and two in-
door scenes. After processing, the final dataset consists of
30 images per scene, along with the recorded event streams,
each showcasing various blur and lighting conditions.
4.3. Experiment Settings
Baselines.
We compare our method with three types of
baselines: 1) NeRF [26] and 3D-GS [13] directly trained
on the blurry images, referring to as B-NeRF and B-3DGS.
2) Deblur rendering methods, including BAD-NeRF [41],
BAD-GS [52], E2NeRF [30], and EDNeRF [3]. Among
these, the first two methods simulate motion blur and opti-
mize camera trajectories without event stream, whereas the
latter two are event-assisted methods without camera tra-
jectory optimization. 3) Image deblur methods, including
UFP [6] (single-image deblurring), EDI [29] (event-based
deblurring), and EFNet [35] (learnable event-based deblur-

<!-- page 6 -->
Table 1. Quantitative comparisons of novel view synthesis across large-scale, medium-scale, object-level, and real-world scenes. The table
reports the average performance for each scale, demonstrating that our method consistently surpasses previous state-of-the-art approaches
across all metrics. Best-performing results are highlighted in bold and second-best results in underline.
Scene Type
Metric
B-NeRF
B-3DGS
UFP-GS
EDI-GS
EFN-GS
E2NeRF
BAD-NeRF
BAD-GS
EDNeRF
Ours
Large-scale
PSNR↑
21.33
21.48
21.36
22.31
22.69
22.96
23.85
23.86
24.63
26.02
SSIM↑
.6781
.6876
.6600
.6855
.6826
.7066
.7323
.7325
.7525
.8064
LPIPS↓
.4249
.3971
.3736
.3823
.3631
.3751
.3480
.3473
.3279
.2680
Medium-scale
PSNR↑
24.08
24.80
26.38
26.44
26.13
27.78
28.46
28.46
28.91
30.47
SSIM↑
.7173
.7512
.8022
.8012
.7981
.8656
.8791
.8789
.8854
.9164
LPIPS↓
.3617
.3187
.2639
.2581
.2726
.1985
.1823
.1816
.1692
.1519
Objects
PSNR↑
22.28
22.34
25.16
24.94
25.45
29.61
27.33
27.86
29.83
30.24
SSIM↑
.9041
.9049
.9275
.9248
.9289
.9638
.9476
.9501
.9655
.9698
LPIPS↓
.1479
.1471
.1174
.1208
.1103
.0735
.0928
.0911
.0722
.0702
Real-world
BRISQUE↓
92.25
73.80
62.94
62.75
62.93
61.52
61.50
60.89
58.63
53.96
NIQE↓
15.00
12.01
10.17
10.20
10.21
9.440
10.00
9.902
9.011
8.371
PIQE↓
65.92
52.74
45.03
44.83
44.84
46.76
43.95
43.51
44.63
41.53
RankIQA↓
9.428
7.542
6.439
6.411
6.411
5.573
6.285
6.223
5.320
4.895
MetaIQA↑
.1241
.1418
.1732
.1737
.1737
.1809
.1773
.1790
.1909
.1969
ring). We process input blurry images with them and train
the vanilla 3D-GS with pre-deblurred images. The resulting
baselines are referred to as UFP-GS, EDI-GS, and EFN-GS.
Evaluation Metrics.
For synthetic datasets, we employ
the Peak Signal-to-Noise Ratio (PSNR), Structural Similar-
ity Index Measure (SSIM) [42], and VGG-based Learned
Perceptual Image Patch Similarity (LPIPS) [50] to evaluate
the similarity between rendered novel views and ground-
truth novel views. For real-world datasets, since the sharp
ground-truth images are unavailable, we utilize several
No-Reference Image Quality Assessment (NR-IQA) met-
rics for evaluation, including BRISQUE [27], NIQE [28],
PIQE [40], RankIQA [21], and MetaIQA [53], which allow
for image evaluation when lacking ground truth images.
4.4. Synthetic Data Experiments
We evaluate our approach across a variety of scenes, includ-
ing large-scale scenes, medium-scale scenes, and object-
level scenes. Quantitative assessments of novel view syn-
thesis are shown in the first three rows of Table. 1. The
deblurring results of input views are detailed in the sup-
plementary. It can be found that our method achieves sub-
stantial improvements in most of the metrics, especially in
challenging large scenes. Specifically, both B-NeRF and B-
3DGS produce blurry novel views since they are directly
trained on blurred images.
The image deblurring-based
baselines, UFP-GS, EDI-GS and EFN-GS, also produced
inferior results, because the image deblurring process po-
tentially corrupts the 3D consistency of the training im-
ages. Notably, our approach outperforms BAD-GS [52] and
BAD-NeRF [41], due to their limited capability in modeling
complex textures. In addition, our method also surpasses
the event-assisted methods E2NeRF [30] and EDNeRF [3]
in producing high-quality novel views with intricate details,
with better training and rendering efficiency. An extended
analysis of all the baselines is provided in the supplemen-
tary.
The qualitative results are illustrated in Figure. 3, where
the first three rows of Figure. 3(a) shows novel view syn-
thesis results, and Figure. 3(b) shows both novel view and
deblurring view synthesis results. More visualization re-
sults are provided in the supplementary. It can be found
that although E2NeRF [30] performs well in object-level
scenes, it struggles in medium and large-scale scene mod-
eling, producing significant blurring results. Additionally,
BAD-GS [52] falls short in regions with significant color
and depth variations, and produces overly smooth back-
ground textures.
Although EDNeRF [3] exhibits overall
satisfactory performance, its complex network architecture
prolongs the training time (about 7 hours per scene) and
precludes real-time rendering. In comparison, our method
overcomes the baselines in producing high-fidelity novel
views, and significantly reducing training time as well as
demonstrating substantial advantages in real-time applica-
tion scenarios.
4.5. Real-world Data Experiments
We present the quantitative results on the captured real-
world data in the last row of Table. 1.
It can be found
that our method achieves superior performance compared
to other approaches.
Specifically, for NR-IQA metrics,
we achieve improvements in BRISQUE [27], NIQE [28],
PIQE [40], and RankIQA [21] by 15.38%, 19.50%, 11.49%,
and 22.83% respectively. We also achieve an increase in

<!-- page 7 -->
Ground Truth
E2NeRF
BAD-GS
EDNeRF
Ours
Desert
Dormitory
Forests&
Lake
pokémon
(b)  Novel view synthesis and deblurring view synthesis results in ficus, outpool, and cozyroom scenes.
Blurry View
Novel View
Blurry View Input
Ours
GT
EDI-GS
EDNeRF
EFN-GS
BAD-GS
E2NeRF
UFP-GS
B-3DGS
Blurry View
Novel View
Blurry View
Novel View
(a)  Overview of Rendering Novel Views in desert, dormitory, forestlake, and pokémon scenes.
 
Figure 3. Qualitative comparison on the synthetic and real dataset. We show the rendering novel views on the top section (a) and exhibit
both novel view synthesis results and input view deblurring results on the bottom section (b). It shows that our method achieves better
performance in recovering the training blurry views as well as rendering novel views. More results are presented in the supplementary.
19.38% in MetaIQA [53]. The qualitative comparisons are
shown in the last row of Figure. 3(a) and in the supplemen-
tary, which further demonstrate that our method is capa-
ble of reconstructing detailed textures, ultimately achieving
higher-quality novel view synthesis.

<!-- page 8 -->
Table 2. Quantitative ablation on proposed loss functions. Best-performing results are highlighted in bold and second results in underline.
Large-scale
Medium-scale
Real Scene
Lblur
Levent
Ldepth
Lint
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
BRISQUE↓
NIQE↓
PIQE↓
MetaIQA↑
RankIQA↓
✓
✗
✗
✗
24.98
.7865
.2986
29.10
.8954
.1673
57.32
8.799
43.26
.1977
6.125
✓
✓
✗
✗
25.71
.7949
.2745
29.94
.9098
.1551
56.16
8.582
42.51
.2009
5.066
✓
✓
✓
✗
25.54
.8003
.2711
30.05
.9139
.1542
56.17
8.499
42.23
.1996
4.923
✓
✓
✓
✓
26.02
.8064
.2680
30.47
.9164
.1519
53.95
8.371
41.52
.2129
4.895
Table 3. Ablation study about the impact of pose optimization.
ATE ↓
Medium-scale
Large-scale
Initial poses
0.0598 ± 0.0167
0.3825 ± 0.1245
Optimized poses
0.0461 ± 0.0129
0.0489 ± 0.0112
Table 4. Robustness against motion blur level.
Blur level
Mild Blur
Medium Blur
Strong Blur
PSNR↑
26.38
25.71
25.04
SSIM↑
.8163
.7949
.7886
LPIPS↓
.2694
.2745
.2802
4.6. Ablation Study
Camera Poses Optimization. We firstly conduct ablations
to investigate the effect of the number of camera poses opti-
mized in the exposure time. We select five large scenes from
our synthetic dataset for evaluation. In the experiments, we
vary the number of camera poses, denoted as n, from 5,
9, 13, and 17. The quantitative results of the novel view
rendering are displayed in Figure. 4. It indicates that the
results reach a bottleneck at 9 poses. Beyond this point,
the improvements are limited and may potentially lead to
local convergence issues. Based on these experiments, we
choose n = 9 camera poses to achieve a balance between
rendering performance and training efficiency. Here, we
also provide comparison with BAD-NeRF [41] and BAD-
GS [52]. These two methods typically use linear interpola-
tion to obtain camera trajectory, while our camera trajec-
tories are estimated from the decomposed latent images,
which provides more accurate initialization and helps our
method achieves better performance. Moreover, we con-
duct quantitative experiments using 9 camera poses to com-
pute ATE (Average Trajectory Error) of the initial poses
produced by COLMAP [34] and the optimized poses, the
results are shown in Table. 3, which validates the effective-
ness of pose optimization.
Effectiveness of The Loss Functions. We conduct novel
view synthesis experiments on the proposed datasets to val-
idate the effectiveness of the training losses. The quanti-
tative results, as shown in Table. 2, indicate that using only
the blur reconstruction loss leads to suboptimal outputs, per-
forming poorly and lacking high-frequency details on both
5
9
13
17
Pose Num
23.0
23.5
24.0
24.5
25.0
25.5
26.0
PSNR 
Ours
BAD-NERF
BAD-GS
PSNR
LPIPS
0.250
0.275
0.300
0.325
0.350
0.375
0.400
LPIPS 
Figure 4. Ablation on number of poses in the camera trajectory.
synthetic and real-world datasets. In contrast, incorporat-
ing Levent, Ldepth, and Lint enables our proposed method to
produce high-fidelity novel views with intricate details.
Robustness Against Motion Blur Levels. To validate the
robustness of our method in handling different levels of mo-
tion blur, we set up three different camera speeds in the city
blocks scene of the synthetic dataset to obtain images with
varying degrees of blur. Images with mild blur are cap-
tured at half the default camera motion speed, images with
medium blur are captured at the default motion speed, and
images with strong blur are captured at twice the default
motion speed. The quantitative results of novel view syn-
thesis are listed in Table. 4. It can be observed that the re-
sults show no significant fluctuations across different levels
of blur, demonstrating the robustness of our method to vary-
ing motion blur levels. Please refer to the supplementary for
more ablations of our method.
5. Conclusions
This paper introduces Event Stream Assisted Gaussian
Splatting (EvaGaussians), a novel framework that seam-
lessly integrates the event streams captured by an event
camera into the training of 3D-GS, effectively address-
ing the challenges of reconstructing high-quality 3D-GS

<!-- page 9 -->
from motion-blurred images.
We contribute two novel
datasets and conduct comprehensive experiments. The re-
sults demonstrate that our method outperforms previous
state-of-the-art deblurring rendering techniques in terms of
novel view synthesis quality, without sacrificing inference
efficiency. Despite its promising performance, our method
may still face challenges when reconstructing scenes with
extremely intricate textures from severely blurred images.
We will release our code and dataset for future research.

<!-- page 10 -->
References
[1] Christian Brandli, Lorenz Muller, and Tobi Delbruck. Real-
time, high-speed video decompression using a frame- and
event-based davis sensor. In 2014 IEEE International Sym-
posium on Circuits and Systems (ISCAS), pages 686–689,
2014. 2, 12
[2] Marco Cannici and Davide Scaramuzza. Mitigating motion
blur in neural radiance fields with events and frames.
In
CVPR, 2024. 2
[3] Marco Cannici and Davide Scaramuzza. Mitigating motion
blur in neural radiance fields with events and frames. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 2024. 2, 5, 6, 20, 21
[4] Mauro Comi, Alessio Tonioni, Max Yang, Jonathan Trem-
blay, Valts Blukis, Yijiong Lin, Nathan F Lepora, and Lau-
rence Aitchison. Snap-it, tap-it, splat-it: Tactile-informed
3d gaussian splatting for reconstructing challenging surfaces.
arXiv preprint arXiv:2403.20275, 2024. 5
[5] Blender Online Community. Blender - a 3D modelling and
rendering package. Blender Foundation, Stichting Blender
Foundation, Amsterdam, 2018. 12, 15
[6] Zhenxuan Fang, Fangfang Wu, Weisheng Dong, Xin Li, Jin-
jian Wu, and Guangming Shi. Self-supervised non-uniform
kernel estimation with flow-based motion prior for blind im-
age deblurring. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 18105–
18114, 2023. 5, 21
[7] Guillermo Gallego, Tobi Delbr¨uck, Garrick Orchard, Chiara
Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger,
Andrew J Davison, J¨org Conradt, Kostas Daniilidis, et al.
Event-based vision: A survey. IEEE TPAMI, 2020. 3
[8] Philipp Heise, Sebastian Klose, Brian Jensen, and Alois
Knoll. Pm-huber: Patchmatch with huber regularization for
stereo matching. In ICCV, 2013. 5
[9] Yuhuang Hu, Shih-Chii Liu, and Tobi Delbruck. v2e: From
video frames to realistic dvs events.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 1312–1321, 2021. 4, 5, 12
[10] Inwoo Hwang, Junho Kim, and Young Min Kim. Ev-nerf:
Event based neural radiance field.
In Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer
Vision, pages 837–847, 2023. 2
[11] Zhe Jiang, Yu Zhang, Dongqing Zou, Jimmy Ren, Jiancheng
Lv, and Yebin Liu. Learning event-based motion deblurring.
In CVPR, 2020. 2
[12] Peng Jin, Bo Zhu, Li Yuan, and Shuicheng Yan. Moh: Multi-
head attention as mixture-of-head attention. arXiv preprint
arXiv:2410.11842, 2024. 2
[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM TOG, 2023. 1, 4, 5, 21
[14] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik. Lerf: Language embedded
radiance fields. In CVPR, 2023. 1
[15] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and
Daniel Cremers. E-nerf: Neural radiance fields from a mov-
ing event camera. IEEE Robotics and Automation Letters,
2023. 2
[16] Joseph J LaViola Jr. Bringing vr and spatial 3d interaction to
the masses through video games. IEEE Computer Graphics
and Applications, 2008. 1
[17] Dogyoon Lee, Minhyeok Lee, Chajin Shin, and Sangyoun
Lee. Dp-nerf: Deblurred neural radiance field with physical
scene priors. In CVPR, 2023. 2
[18] Songnan Lin, Jiawei Zhang, Jinshan Pan, Zhe Jiang,
Dongqing Zou, Yongtian Wang, Jing Chen, and Jimmy Ren.
Learning event-driven video deblurring and interpolation. In
ECCV, 2020. 2
[19] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu,
Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt,
Eric Xing, and Shijian Lu.
Weakly supervised 3d open-
vocabulary segmentation. In NeurIPS, 2023. 1
[20] Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xu-
anlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao
Su. Openshape: Scaling up 3d shape representation towards
open-world understanding. In NeurIPS, 2024. 1
[21] Xialei Liu, Joost Van De Weijer, and Andrew D Bagdanov.
Rankiqa: Learning from rankings for no-reference image
quality assessment. In Proceedings of the IEEE international
conference on computer vision, pages 1040–1049, 2017. 6
[22] Weng Fei Low and Gim Hee Lee.
Robust e-nerf: Nerf
from sparse & noisy events under non-uniform motion. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, 2023. 2
[23] Weng Fei Low and Gim Hee Lee. Deblur e-nerf: Nerf from
motion-blurred events under high-speed or low-light condi-
tions. In European Conference on Computer Vision, pages
192–209. Springer, 2025. 2
[24] Li Ma, Xiaoyu Li, Jing Liao, Qi Zhang, Xuan Wang, Jue
Wang, and Pedro V Sander. Deblur-NeRF: Neural Radiance
Fields from Blurry Images. In CVPR, 2022. 2, 12
[25] Li Ma, Xiaoyu Li, Jing Liao, Qi Zhang, Xuan Wang, Jue
Wang, and Pedro V Sander. Deblur-nerf: Neural radiance
fields from blurry images. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 12861–12870, 2022. 5
[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing Scenes as Neural Radiance Fields for View
Synthesis. In ECCV, 2020. 2, 5, 12, 21
[27] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad
Bovik. No-reference image quality assessment in the spa-
tial domain. IEEE TIP, 21(12):4695–4708, 2012. 6
[28] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Mak-
ing a “completely blind” image quality analyzer. IEEE Sig-
nal processing letters, 20(3):209–212, 2012. 6
[29] Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley,
Miaomiao Liu, and Yuchao Dai. Bringing a blurry frame
alive at high frame-rate with an event camera.
In CVPR,
2019. 2, 3, 5, 21
[30] Yunshan Qi, Lin Zhu, Yu Zhang, and Jia Li. E2nerf: Event
enhanced neural radiance fields from blurry images.
In
ICCV, 2023. 2, 5, 6, 20, 21

<!-- page 11 -->
[31] Henri Rebecq, Daniel Gehrig, and Davide Scaramuzza.
Esim: an open event camera simulator. In Conference on
robot learning, pages 969–982. PMLR, 2018. 4, 5
[32] Antoni Rosinol, John J Leonard, and Luca Carlone. Nerf-
slam: Real-time dense monocular slam with neural radiance
fields. In IROS, 2023. 1
[33] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and
Vladislav Golyanik. Eventnerf: Neural radiance fields from a
single colour event camera. In Computer Vision and Pattern
Recognition (CVPR), 2023. 2
[34] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion Revisited. In CVPR, 2016. 2, 3, 8
[35] Lei Sun, Christos Sakaridis, Jingyun Liang, Qi Jiang, Kailun
Yang, Peng Sun, Yaozu Ye, Kaiwei Wang, and Luc Van
Gool. Event-based fusion for motion deblurring with cross-
modal attention. In European Conference on Computer Vi-
sion, pages 412–428. Springer, 2022. 5, 21
[36] Zhenyu Tang, Junwu Zhang, Xinhua Cheng, Wangbo Yu,
Chaoran Feng, Yatian Pang, Bin Lin, and Li Yuan.
Cy-
cle3d:
High-quality and consistent image-to-3d genera-
tion via generation-reconstruction cycle.
arXiv preprint
arXiv:2407.19548, 2024. 1, 2
[37] Zhenyu Tang, Junwu Zhang, Xinhua Cheng, Wangbo Yu,
Chaoran Feng, Yatian Pang, Bin Lin, and Li Yuan.
Cy-
cle3d:
High-quality and consistent image-to-3d genera-
tion via generation-reconstruction cycle.
arXiv preprint
arXiv:2407.19548, 2024. 2
[38] Gemma Taverni.
Applications of Silicon Retinas: From
Neuroscience to Computer Vision. PhD thesis, Universit¨at
Z¨urich, 2020. 5
[39] Manoj K Vairalkar and SU Nimbhorkar. Edge detection of
images using sobel operator. International Journal of Emerg-
ing Technology and Advanced Engineering, 2(1):291–293,
2012. 5
[40] N Venkatanath, D Praneeth, Maruthi Chandrasekhar Bh,
Sumohana S Channappayya, and Swarup S Medasani. Blind
image quality evaluation using perception based features.
In 2015 twenty first national conference on communications
(NCC), pages 1–6. IEEE, 2015. 6
[41] Peng Wang, Lingzhe Zhao, Ruijie Ma, and Peidong Liu.
BAD-NeRF: Bundle Adjusted Deblur Neural Radiance
Fields. In CVPR, 2023. 2, 3, 5, 6, 8, 15
[42] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 6
[43] Jingqian Wu, Shuo Zhu, Chutian Wang, and Edmund Y Lam.
Ev-gs: Event-based gaussian splatting for efficient and accu-
rate radiance field rendering. In 2024 IEEE 34th Interna-
tional Workshop on Machine Learning for Signal Processing
(MLSP), pages 1–6. IEEE, 2024. 2
[44] Tianyi Xiong, Jiayi Wu, Botao He, Cornelia Fermuller, Yian-
nis Aloimonos, Heng Huang, and Christopher A Metzler.
Event3dgs: Event-based 3d gaussian splatting for fast ego-
motion. arXiv preprint arXiv:2406.02972, 2024. 2
[45] Linning Xu, Vasu Agrawal, William Laney, Tony Garcia,
Aayush Bansal, Changil Kim, Samuel Rota Bul`o, Lorenzo
Porzi, Peter Kontschieder, Aljaˇz Boˇziˇc, et al. Vr-nerf: High-
fidelity virtualized walkable spaces.
In SIGGRAPH Asia,
2023. 1
[46] Lin Yen-Chen, Pete Florence, Jonathan T Barron, Alberto
Rodriguez, Phillip Isola, and Tsung-Yi Lin. inerf: Inverting
neural radiance fields for pose estimation. In IROS, 2021. 1
[47] Xiaoting Yin, Hao Shi, Yuhan Bao, Zhenshan Bing, Yiyi
Liao, Kailun Yang, and Kaiwei Wang. E-3dgs: Gaussian
splatting with exposure and motion events. arXiv preprint
arXiv:2410.16995, 2024. 2
[48] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li,
Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan,
and Yonghong Tian. Viewcrafter: Taming video diffusion
models for high-fidelity novel view synthesis. arXiv preprint
arXiv:2409.02048, 2024. 1, 2
[49] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu,
Shaofeng Zhang, Yujun Shi, Ruijie Zhu, Xinhua Cheng,
Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark
for metamorphic evaluation of text-to-time-lapse video gen-
eration. arXiv preprint arXiv:2406.18522, 2024. 2
[50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 6
[51] Zixin Zhang, Kanghao Chen, and Lin Wang.
Elite-evgs:
Learning event-based 3d gaussian splatting by distilling
event-to-video priors.
arXiv preprint arXiv:2409.13392,
2024. 2
[52] Lingzhe Zhao, Peng Wang, and Peidong Liu. Bad-gaussians:
Bundle adjusted deblur gaussian splatting. In ECCV, 2024.
2, 5, 6, 8, 20, 21
[53] Hancheng Zhu, Leida Li, Jinjian Wu, Weisheng Dong, and
Guangming Shi.
Metaiqa:
Deep meta-learning for no-
reference image quality assessment.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 14143–14152, 2020. 6, 7
[54] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In CVPR, 2022. 1

<!-- page 12 -->
EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images
Supplementary Material
A. Dataset Details
A.1. EvaGaussians-Blender Dataset
A.1.1. Dataset Overview
We use Blender [5] to craft nine indoor and outdoor 3D
scenes, and further incorporate four 3D scenes from De-
blurNeRF [24] and six 3D objects from the NeRF-Synthetic
dataset [26] as our base scenes. We then design various
camera trajectories to simulate motion-blurred images on
these base scenes, and generate the corresponding event
streams using V2E [9]. Visualization of our crafted scenes
are shown in Figure. 5 and Figure. 8, and an overview of the
design purposes of these crafted scenes is provided below:
Indoor Scenes
• Classroom: A typical classroom setting featuring desks,
chairs, a blackboard, and educational posters. This scene
is designed to simulate an academic environment, ideal
for educational and surveillance applications.
• Caf´e: A cozy caf´e with tables, chairs, a counter, and var-
ious decorations. This scene mimics a social setting, pro-
viding a dynamic backdrop for testing social interaction
algorithms and retail analytics.
• Dormitory: A student dormitory room equipped with
beds, study desks, personal belongings, and typical dorm
furniture. This scene represents a personal living space,
useful for smart home and security applications.
Outdoor Scenes
• Desert: A vast, arid landscape with sand dunes and sparse
vegetation. This scene is perfect for testing navigation
and object detection in harsh, unstructured environments.
• City Blocks: Urban scenes featuring streets, buildings, ve-
hicles, and pedestrians. This environment is essential for
autonomous driving, urban planning, and smart city ap-
plications.
• Lake: A serene natural setting with dense forests sur-
rounding a tranquil lake. This scene provides a complex
environment for testing outdoor navigation, environmen-
tal monitoring, and wildlife tracking.
• Forests: A rugged terrain with forested areas and scat-
tered boulders. This scene is useful for off-road naviga-
tion and geological survey applications.
• Venice:
A picturesque representation of Venice with
canals, bridges, and historic architecture. This scene of-
fers a unique setting for cultural heritage preservation,
tourism, and urban analytics.
• London: A bustling cityscape of London with iconic land-
marks, streets, and a dynamic urban environment. This
scene supports applications in tourism, traffic manage-
ment, and city modeling.
A.1.2. Camera Settings
To render the base scenes and simulate motion blur, we con-
figure the virtual camera in Blender with a resolution of
400 × 600, and set the scaling factor to 1.0. The virtual
camera utilized a perspective model with a shutter speed
of 1/180 seconds. Subsequently, we developed a dedicated
script to generate camera trajectory and motion blur. An ex-
ample is shown in Figure. 9. Along each predefined virtual
camera trajectory, we uniformly sampled 35 camera poses,
adding a certain level of jitter to create the training set. We
recorded the start and end time of the camera exposure time,
the positions, and 20 intermediate frames during the expo-
sure time (obtained through linear interpolation between the
start and end positions). We then uniformly sample 100
camera poses along the same trajectory to form the test set.
Using the event camera simulator from V2E [9], we simu-
late the event stream for each camera trajectory and synthe-
size the event bins from the event stream at the start and end
of the exposure time.
A.2. EvaGaussians-DAVIS Dataset
We use the color DAVIS346 event camera [1] to record our
real-world event and RGB sequences and utilize the default
camera settings provided in the DV software that comes
with the camera. We name the five captured scenes as desk
& chair, washroom, pok´emon, pillow, and bag.
A.2.1. Camera Calibration
We calibrated the event camera using the DV software pro-
vided by DAVIS. During the calibration process, we used
a 6 × 9 checkerboard pattern with a square size of 30 mm.
In the software configuration, we set the width to 9, height
to 6, and square size to 30 mm. We then ran the calibra-
tion module and moved the calibration pattern in front of
the camera. The software detected the pattern and collected
images, highlighting the detected area in green, as shown in
Figure. 10. We set the minimum detections parameter to 50
to ensure a sufficient number of samples and used the con-
secutive detections parameter to ensure consistent pattern
detection. Additionally, we enabled the image verification
option to check the collected images in real-time, discarding
inaccurately detected images and replacing them with new
ones. We evaluate the calibration accuracy using the repro-
jection error er as Eq. 10 and the epipolar error eepipolar as

<!-- page 13 -->
Rendered
Mode
WireFrame
Mode
Classroom
Café
Dormitory
Figure 5. Visualization of EvaGaussians-Blender Indoor Scenes. The sizes of the Caf´e and Classroom scenes are approximately 15 × 7 × 4
meters, while the Dormitory scene is approximately 5 × 5 × 4 meters (with an additional outdoor garden, making the overall scene size
20 × 20 × 6 meters).
Figure 6. Visualization of Reprojection Errors and Epipolar Er-
rors. The figure illustrates the 50 sets of reprojection errors and
epipolar errors generated during the calibration process. The re-
projection error er represents the average discrepancy between the
observed points and the projected points, calculated as shown in
Eq. 10. The epipolar error eepipolar represents the average distance
between points in one camera and the epipolar lines calculated
from the other camera for each pair of images, calculated as shown
in Eq. 11. As shown in the figure, the average reprojection error is
approximately 0.5, and the average epipolar error is approximately
0.05, indicating a high level of accuracy in the calibration process.
Eq. 11 in stereo calibration. The reprojection error is calcu-
lated as follows:
er = 1
n
n
X
i=1
∥xi −ˆxi∥
(10)
where xi represents the observed points and ˆxi represents
the projected points. The epipolar error is calculated as the
COLMAP
Optimized 
Mild 
Blur
Medium 
Blur
Strong
Blur
GT Poses
Produced Poses
Figure 7. Visualization of pose accuracy in different level of mo-
tion blur.
average epipolar error for each point in all collected images.
For each pair of images, the error is calculated as the sum

<!-- page 14 -->
Venice
Forests
London
Desert
City Blocks
Lake
Rendered
Mode
WireFrame
Mode
Rendered
Mode
WireFrame
Mode
Figure 8. Visualization of EvaGaussians-Blender Outdoor Scenes. These scenes include rich details ane diverse components like sky, lake,
river, desert, forest, cities, roads. All scenes cover an area of more than 1 square kilometer.
of the distances between the points in one camera and the
epipolar lines calculated from the other camera (m is the
number of acquired images, n is the number of points). The
formula is as follows:
eepipolar =
1
m × n
m
X
i=1
n
X
j=1
[d(P1i,j, l2,i,j) + d(P2i,j, l1,i,j)]
(11)
where P1i,j and P2i,j are the projection points of the jth
point in the ith image in two cameras, and l1,i,j and l2,i,j
are the epipolar lines corresponding to the jth point in the
ith image calculated from the other camera. The maximum
allowable error can be set under Max Reprojection Error.
The stereo calibration also calculates the error caused by
the epipolar constraint, which can be set under Max Epipo-
lar Error. Once the calibration is successful, the results are
saved and the undistorted output is displayed. The visual-
ization results of the two types of errors are shown in Fig-
ure. 6. This process ensures the accuracy of the calibration,
thereby improving the measurement accuracy and stability
in subsequent applications.
A.2.2. Camera Settings
We recorded the five real scenes using the calibration pa-
rameters obtained during the calibration process. By adjust-
ing the indoor lighting and shooting angles, we ensured the
richness of the recorded scene details. The adopted event
camera has a spatial resolution of 346 × 260, a temporal
resolution of 1 µs, a typical latency of less than 1 ms, a
maximum throughput of 12 MEps, and a dynamic range of
approximately 120 dB (with 50% of the pixels responding
to 80% contrast changes under 0.1-100k lux conditions).

<!-- page 15 -->
(a) Desert                                       (b) Classroom                                        (c) Café                           
(d) Forests
(e) Lake                                          (f) Dormitory                                   (g) City Blocks                   
(h) London
Figure 9. Visualization of Camera Trajectory. The trajectories depicted were manually configured within Blender [5] to ensure precise
control over the camera paths. For the purpose of visualization, these trajectories have been normalized.
Figure 10. Illustration of Camera Calibration. The left panel shows the checkerboard pattern captured from various positions and
angles, with detected corner points utilized for calibration. The right panel presents the calibrated checkerboard pattern, demonstrating
the corresponding points and lines between two cameras, which reflect the geometric relationship and accuracy achieved after calibration.
Different colored lines indicate the correspondences between points during the calibration process.
The contrast sensitivity is 14.3% (ON) and 22.5% (OFF)
(with 50% of the pixels responding). These parameters en-
sure that the event camera can stably and efficiently record
scene information under various lighting conditions and dy-
namic ranges.
B. Detailed Experiments
B.1. Synthetic Data Experiments
B.1.1. Deblurring View Synthesis Comparison
Refering to [41], we additionally provide deblurring view
synthesis (DVS) results on our proposed EvaGaussians-
Blender dataset, and show more qualitative results of novel
view synthesis (NVS). For object-level scenes, Table. 5 and

<!-- page 16 -->
Table 5. Quantitative comparisons of DVS on object-level scenes. The results indicate that our method outperforms previous state-of-the-
art approaches, consistently achieving better performance across all metrics.
Deblur View
B-NeRF
B-3DGS
UFP-GS
EDI-GS
EFN-GS
E2NeRF
BAD-NeRF
BAD-GS
EDNeRF
Ours
PSNR↑
22.87
23.01
27.99
27.92
28.12
29.70
28.33
28.61
29.95
30.02
SSIM↑
.9068
.9092
.9501
.9495
.9508
.9589
.9576
.9582
.9599
.9605
LPIPS↓
.1450
.1437
.0743
.0747
.0739
.0722
.0734
.0732
.0720
.0719
Table 6. Quantitative comparisons of DVS on the medium-scale scenes. The results show that our method surpasses previous state-of-the-
art approaches, achieving better performance consistently across all metrics.
Deblur View
B-NeRF
B-3DGS
UFP-GS
EDI-GS
EFN-GS
E2NeRF
BAD-NeRF
BAD-GS
EDNeRF
Ours
PSNR↑
24.27
25.05
26.60
26.65
26.30
27.95
28.62
28.70
29.12
30.26
SSIM↑
.7254
.7631
.8135
.8100
.8068
.8743
.8883
.8890
.8951
.9241
LPIPS↓
.3513
.3101
.2547
.2486
.2628
.1874
.1735
.1715
.1588
.1419
Table 7. Quantitative comparison of DVS on large-scale scenes. The results demonstrate that our method consistently achieves better
performance across all metrics.
Deblur View
B-NeRF
B-3DGS
UFP-GS
EDI-GS
EFN-GS
E2NeRF
BAD-NeRF
BAD-GS
EDNeRF
Ours
PSNR↑
21.42
21.58
21.43
22.36
22.75
23.02
23.92
23.98
24.79
25.85
SSIM↑
.6795
.6914
.6690
.6943
.6915
.7155
.7412
.7425
.7614
.8039
LPIPS↓
.4185
.3860
.3672
.3710
.3520
.3689
.3468
.3459
.3168
.2635
Figure. 12 present the quantitative and qualitative results
of ours and the comparison baselines across six synthetic
scene sequences. From the qualitative results, it is evident
that our method excels in reconstructing fine details and
maintaining high fidelity in both NVS and DVS. In terms of
quantitative results, our method outperforms baseline meth-
ods in most scenes. For medium-scale scenes and large-
scale scenes, the quantitative results are shown in Table. 6
and Table. 7, and the qualitative results of NVS and DVS
are shown in Figure. 13, which demonstrate that our model
achieves better performance in both tasks.
B.1.2. Per-scene Comparison for Novel View Synthesis
In this subsection, we present a detailed per-scene analy-
sis of the novel view synthesis performance in medium and
large scale scenes from the EvaGaussians-Blender dataset,
to evaluate the effectiveness of our method across different
challenging scenes.
Table. 8 shows the PSNR value of the NVS results,
which demonstrates that our proposed method consistently
outperforms other approaches across various scenes. The
detailed metrics for SSIM and LPIPS in Table. 9 and Ta-
ble. 10 further show that our model excels in maintaining
structural integrity and perceptual quality in synthesized
views. Specifically, in medium-scale scenes, our method
exhibits robust performance, particularly in complex envi-
ronments where maintaining detail and minimizing artifacts
are challenging. This can be demonstrated in scenes such
as cozyroom and factory, where our method achieves sig-
nificant improvements in both PSNR and SSIM. For large-
scale scenes, scenes like desert and city blocks highlight the
model’s capability to generalize across different scales and
provide high-quality novel view synthesis.
We also present more qualitative results of novel view
synthesis in Figure. 14 and Figure. 15. The results highlight
our model’s ability to reconstruct fine details and maintain
high color accuracy beyond the comparison baselines.
B.2. Real-world Data Experiments
In this section, we present a comprehensive per-scene anal-
ysis of NVS results on the EvaGaussians-DAVIS dataset.
The qualitative results are shown in Figure. 11, where the
first column shows the blurry image used for training, and
the following rows show the deblur results of different
methods. The results demonstrate that our method consis-
tently excels in reconstructing fine details compared to other
methods.
We further report the per-scene quantitative results to
validate our robustness to different scenes. As introudced
in the main text, the adopted metrics include BRISQUE,
NIQE, PIQE, MetaIQA, and RankIQA, which can effec-
tively assess the quality of synthesized views in a no-

<!-- page 17 -->
Table 8. The Novel View Synthesis Results of PSNR ↑in the EvaGaussians-Blender Dataset. The highest values in each category
are highlighted in bold to indicate the best results.
Models
Medium
Large
Classroom Dormitory Caf´e Pool Cozyroom Factory Tanabata Desert City Blocks London Forests Lake
Blurry-NeRF
25.42
26.72
21.23 26.66
15.42
26.76
26.38
19.04
19.34
20.00
23.60
24.67
Blurry-GS
25.59
26.83
21.34 26.17
20.76
26.48
26.46
19.67
19.48
20.18
23.75
24.34
UFP-GS
28.64
29.68
21.54 26.79
21.62
27.60
28.77
20.78
20.85
16.61
22.96
25.60
EDI-GS
28.49
29.90
22.62 26.36
21.17
29.06
27.46
20.66
20.68
20.27
23.97
25.99
EFNET-GS
28.59
29.48
22.04 26.36
21.42
27.57
27.43
20.62
21.70
20.36
25.57
25.18
E2NeRF
28.87
30.77
26.16 28.92
21.23
29.85
28.65
20.02
21.78
21.30
25.90
25.78
BAD-NeRF
30.28
31.23
27.18 28.71
21.68
30.28
29.85
21.09
21.91
22.91
26.35
26.99
BAD-Gaussians
30.28
31.23
27.18 28.72
21.68
30.28
29.85
21.10
21.93
22.93
26.35
26.98
EvDeblurNeRF
31.83
28.95
28.66 29.69
22.01
31.20
30.02
21.62
22.23
23.88
27.10
28.29
Ours
34.38
32.97
30.41 30.26
22.71
31.85
30.71
24.88
23.71
23.99
27.62
29.90
Table 9. The Novel View Synthesis of SSIM ↑in the EvaGaussians-Blender Dataset. The highest values in each category are
highlighted in bold to indicate the best results.
Models
Medium
Large
Classroom Dormitory Caf´e
Pool
Cozyroom Factory Tanabata Desert City Blocks London Forests Lake
Blurry-NeRF
0.7086
0.8281
0.5682 0.7442
0.5098
0.8567
0.8057
0.6023
0.6325
0.6732
0.7002 0.7823
Blurry-GS
0.7154
0.8312
0.5638 0.7265
0.7632
0.8519
0.8064
0.6386
0.6341
0.6819
0.7026 0.7807
UFP-GS
0.8701
0.9281
0.5706 0.7527
0.7729
0.8640
0.8569
0.6172
0.6144
0.5807
0.6911 0.7968
EDI-GS
0.8456
0.9291
0.5977 0.7288
0.7703
0.8963
0.8408
0.6567
0.5655
0.6882
0.7158 0.8012
EFNET-GS
0.8692
0.9259
0.5841 0.7288
0.7797
0.8643
0.8345
0.6164
0.6366
0.6069
0.7628 0.7901
E2NeRF
0.8723
0.9319
0.7869 0.8795
0.7724
0.9245
0.8915
0.6169
0.6822
0.6734
0.7467 0.8139
BAD-NeRF
0.8978
0.9337
0.8041 0.8794
0.7764
0.9353
0.9271
0.6314
0.6867
0.6932
0.7583 0.8919
BAD-Gaussians
0.8992
0.9351
0.8031 0.8781
0.7755
0.9345
0.9266
0.6346
0.6851
0.6944
0.7578 0.8908
EvDeblurNeRF
0.9023
0.8935
0.8513 0.8885
0.7854
0.9454
0.9311
0.6589
0.7023
0.7159
0.7726 0.9129
Ours
0.9402
0.9580
0.9033 0.9108
0.8052
0.9592
0.9382
0.8152
0.7405
0.7145
0.8152 0.9465
reference manner.
As shown in Table. 11, our model
achieves the best BRISQUE scores across all scenes, high-
lighting its ability to produce visually appealing and less
distorted images. For NIQE, as presented in Table. 13, our
approach significantly outperforms the baselines, achiev-
ing the lowest average NIQE score. This demonstrates our
method’s robustness in generating high-quality images with
minimal perceptual artifacts. In terms of PIQE, Table. 14
shows that our model again leads in performance, achiev-
ing the lowest PIQE scores, which underscores the effec-
tiveness of our model in preserving image details and re-
ducing noise. Furthermore, our method excels in MetaIQA
and RankIQA evaluations, as detailed in Tables. 15 and Ta-
ble. 16, respectively. The highest MetaIQA scores and low-
est RankIQA scores across most scenes affirm the overall
better visual quality and fidelity of our synthesized views
compared to baseline models. Overall, these results demon-
strate the robustness of our method, particularly in handling
complex scenes and maintaining high visual quality across
diverse scenarios.
B.3. Ablation Study
In this section, we present an additional ablation study on
the robustness of pose optimization in different blur level.
We redesign three different levels of motion blur sequences
in medium-scale scenes and compare the Average Trajec-
tory Error (ATE) between the initial poses produced by
COLMAP and the optimized poses. Figure. 7 illustrates the
visualization of COLMAP poses and the optimized poses
on the City Blocks scene.
As the motion blur becomes
more severe, the accuracy of the COLMAP poses is sig-
nificantly impacted, while the optimized poses maintain a
higher level of accuracy. In a horizontal comparison, the op-
timized poses better match the ground truth across various
levels of blur, demonstrating the effectiveness of pose opti-
mization. Table. 12 presents the quantitative results, further

<!-- page 18 -->
Table 10. The Novel View Synthesis of LPIPS ↓in the EvaGaussians-Blender Dataset. The highest values in each category are
highlighted in bold to indicate the best results.
Models
Medium
Large
Classroom Dormitory Caf´e
Pool
Cozyroom Factory Tanabata Desert City Blocks London Forests Lake
Blurry-NeRF
0.3987
0.2998
0.4528 0.3848
0.5545
0.2063
0.2348
0.4447
0.5273
0.3971
0.3425 0.4127
Blurry-GS
0.3824
0.2873
0.4554 0.4198
0.2432
0.2091
0.2335
0.4231
0.4116
0.3925
0.3379 0.4206
UFP-GS
0.2838
0.1361
0.4511 0.3646
0.2331
0.1928
0.1856
0.3816
0.3342
0.4069
0.3383 0.4069
EDI-GS
0.2914
0.1267
0.4038 0.4102
0.2388
0.1446
0.1911
0.4062
0.3857
0.3901
0.3295 0.4001
EFNET-GS
0.2846
0.1373
0.4529 0.4102
0.2345
0.1926
0.1958
0.3912
0.3222
0.3932
0.2962 0.4128
E2NeRF
0.2821
0.1165
0.2871 0.2048
0.2369
0.1073
0.1546
0.3938
0.3713
0.3928
0.3589 0.3588
BAD-NeRF
0.2365
0.1083
0.2715 0.2103
0.2278
0.0992
0.1224
0.3947
0.3618
0.3224
0.3467 0.3142
BAD-Gaussians
0.2384
0.1078
0.2695 0.2094
0.2262
0.0985
0.1215
0.3965
0.3604
0.3215
0.3452 0.3129
EvDeblurNeRF
0.2217
0.1455
0.2447 0.1926
0.1942
0.0768
0.1086
0.3823
0.3497
0.2972
0.3163 0.2941
Ours
0.1927
0.0928
0.2311 0.1859
0.1873
0.0715
0.1023
0.2053
0.2835
0.2983
0.2857 0.2674
Table 11. The Novel View Synthesis of BRISQUE in the EvaGaussians-DAVIS Dataset. The highest values in each category are
highlighted in bold to indicate the best results.
Models
BRISQUE ↓
Desk & Chair
Washroom
Pok´emon
Pillow
Bag
Average
B-NeRF
63.9428
102.7828
109.2711
97.4778
87.7699
92.2489
B-3DGS
51.1542
82.2262
87.4169
77.9823
70.2159
73.7991
UFP-GS
43.6932
69.7354
74.6684
66.6098
59.9711
62.9356
EDI-GS
43.4811
69.9923
74.3044
66.2849
59.6835
62.7492
EFN-GS
43.5235
69.8473
74.2875
66.3158
59.6523
62.7253
E2NeRF
36.2148
64.4332
77.0112
67.1324
62.8063
61.5196
BAD-NeRF
42.6285
68.5219
72.8474
64.9852
58.5133
61.4993
BAD-GS
42.2065
67.8434
72.1261
64.3418
57.9339
60.8903
EDNeRF
34.5687
61.5044
73.5012
64.0809
59.4969
58.6304
Ours
32.9225
58.5756
70.0211
61.0294
58.1876
56.1472
demonstrating the effectiveness and robustness of pose op-
timization in handling different levels of motion blur.
C. Broader Impacts
Our proposed EvaGaussians leverages event cameras to as-
sist novel view synthesis from low-quality, blurred images.
It has the potential to bring about both positive and negative
societal impacts.
On the positive side, our method can improve the effi-
ciency of surveillance systems by reconstructing clear 3D
images from low-quality footage, enabling better identifica-
tion of individuals and objects in challenging conditions.
This can bolster public safety and aid in criminal inves-
tigations.
Additionally, the ability to reconstruct scenes
from blurred inputs can enhance the performance of au-
tonomous vehicles, drones, and robots, enabling them to
navigate more accurately in poor visibility conditions, lead-
ing to safer and more efficient transportation and logistics.
In situations where traditional cameras may struggle to cap-
ture clear images under extreme conditions, our method can
provide valuable information for first responders and res-
cue teams, helping them make informed decisions and po-
tentially saving lives. Furthermore, our technique can be
applied to medical imaging, allowing for better visualiza-
tion of internal structures and more accurate diagnoses, ul-
timately leading to improved patient outcomes.
On the negative side, the enhanced surveillance capabili-
ties enabled by our method may raise privacy concerns. For
example, our method could be used for malicious purposes,
such as stalking or spying on individuals without their con-
sent. It is important to establish regulations and guidelines
to prevent such misuse.

<!-- page 19 -->
Table 12. Robustness of pose optimization.
ATE ↓
Mild Blur
Medium Blur
Strong Blur
COLMAP poses
0.0598 ± 0.0167
0.0783 ± 0.0245
0.0973 ± 0.0352
Optimized poses
0.0461 ± 0.0129
0.0586 ± 0.0149
0.0677 ± 0.0208
Table 13. The Novel View Synthesis of NIQE in the EvaGaussians-DAVIS Dataset.
Models
NIQE ↓
Desk & Chair
Washroom
Pok´emon
Pillow
Bag
Average
B-NeRF
11.4380
15.5386
16.9153
16.2316
14.8844
15.0016
B-3DGS
9.1504
12.4309
13.5323
12.9853
11.9075
12.0113
UFP-GS
7.7736
10.6170
11.3827
10.9266
10.1715
10.1743
EDI-GS
7.7778
10.5662
11.5024
11.0375
10.1214
10.2011
EFN-GS
7.8235
10.5412
11.4868
11.0743
10.1288
10.2109
E2NeRF
6.8907
9.0924
11.6383
10.1830
9.3954
9.4400
BAD-NeRF
7.6253
10.3590
11.2769
10.8211
9.9229
10.0011
BAD-GS
7.5498
10.2565
11.1652
10.7139
9.8247
9.9020
EDNeRF
6.5775
8.6791
11.1092
9.7201
8.9684
9.0109
Ours
6.1898
8.1117
10.2662
8.9455
8.3421
8.3711
Table 14. The Novel View Synthesis of PIQE in the EvaGaussians-DAVIS Dataset.
Models
PIQE ↓
Desk & Chair
Washroom
Pok´emon
Pillow
Bag
Average
B-NeRF
58.4566
68.8701
73.4701
54.0862
74.7257
65.9217
B-3DGS
46.7653
55.0961
58.7761
43.2689
59.7806
52.7374
UFP-GS
39.9453
47.0612
50.2026
36.9659
50.9631
45.0276
EDI-GS
39.7505
46.8317
49.9597
36.7786
50.8135
44.8268
EFN-GS
39.7748
46.7951
49.9346
36.7891
50.8275
44.8242
E2NeRF
40.9825
49.2988
50.4868
39.7896
53.2203
46.7556
BAD-NeRF
38.9711
45.9134
48.9801
36.0574
49.8171
43.9478
BAD-GS
38.5852
45.4588
48.4951
35.7004
49.3239
43.5127
EDNeRF
39.1197
47.0580
48.1919
37.9810
50.8012
44.6304
Ours
36.5717
44.3304
44.0641
35.3059
47.3749
41.5294
Table 15. The Novel View Synthesis of MetaIQA in the EvaGaussians-DAVIS Dataset.
Models
MetaIQA ↑
Desk & Chair
Washroom
Pok´emon
Pillow
Bag
Average
B-NeRF
0.1419
0.1272
0.1085
0.1122
0.1307
0.1241
B-3DGS
0.1621
0.1454
0.1240
0.1283
0.1494
0.1418
UFP-GS
0.1976
0.1774
0.1521
0.1564
0.1823
0.1732
EDI-GS
0.1986
0.1780
0.1518
0.1571
0.1830
0.1737
EFN-GS
0.1928
0.1835
0.1579
0.1643
0.1799
0.1757
E2NeRF
0.1959
0.2020
0.1619
0.1723
0.1722
0.1809
BAD-NeRF
0.2027
0.1817
0.1549
0.1603
0.1867
0.1773
BAD-GS
0.2047
0.1835
0.1565
0.1620
0.1886
0.1790
EDNeRF
0.2067
0.2132
0.1709
0.1819
0.1817
0.1909
Ours
0.2135
0.2188
0.1975
0.2188
0.2160
0.2129

<!-- page 20 -->
Table 16. The Novel View Synthesis of RankIQA in the EvaGaussians-DAVIS Dataset.
Models
RankIQA ↓
Desk & Chair
Washroom
Pok´emon
Pillow
Bag
Average
B-NeRF
7.4896
10.4454
10.7921
9.6578
8.7541
9.4278
B-3DGS
5.9917
8.3563
8.6337
7.7262
7.0032
7.5422
UFP-GS
5.1153
7.1215
7.3741
6.6005
5.9829
6.4389
EDI-GS
5.0929
7.1029
7.3386
6.5673
5.9528
6.4109
EFN-GS
5.1046
7.0893
7.3158
6.5789
5.9682
6.4114
E2NeRF
4.5141
6.4997
5.7764
5.7600
5.3166
5.5733
BAD-NeRF
4.9931
6.9636
7.1948
6.4385
5.8360
6.2852
BAD-GS
4.9436
6.8947
7.1235
6.3748
5.7783
6.2230
EDNeRF
4.3089
6.2043
5.5138
5.4981
5.0749
5.3200
Ours
3.9369
5.6564
5.0867
5.1538
4.6422
4.8952
Blurry GT              
E  NeRF
BAD-GS              
EDNeRF
Ours
2  
Figure 11. NVS results on the EvaGaussian-DAVIS dataset. The first column shows the blurry image used for training, and the following
rows show the deblurring results of different methods. The results demonstrate that our method consistently excels in reconstructing fine
details compared to other methods [3, 30, 52].

<!-- page 21 -->
Blurry View
Novel View
Blurry View
Novel View
Blurry View
Novel View
Blurry View
Novel View
Blurry View
Novel View
Blurry View
Novel View
Blurry View Input
Ours
GT
BAD-GS
EDNeRF
EDI-GS
UFP-GS
B-3DGS
E2NeRF
EFT-GS
Figure 12. Visualization of DVS and NVS of Object-level Scenes in the EvaGaussian-Blender Dataset. The DVS results are highlighted
in the red bouding box. The results demonstrate that our method consistently excels in reconstructing fine details and maintaining high
color accuracy compared to other methods [3, 6, 13, 26, 29, 30, 35, 52].

<!-- page 22 -->
Blurry View Input
Blurry View
Novel View
Ours
GT
BAD-GS
EDNeRF
EDI-GS
UFP-GS
B-3DGS
E2NeRF
EFT-GS
Blurry View
Novel View
Blurry View
Novel View
Blurry View
Novel View
Figure 13. Visualization of DVS and NVS of results. The DVS results are highlighted in the red bouding box. The results demonstrate
that our method consistently excels in reconstructing fine details and maintaining high color accuracy compared to other methods.

<!-- page 23 -->
UFP-GS
EFN-GS
EDI-GS
B-3DGS
B-NeRF
Figure 14. Visualization of Novel View Synthesis of All-redesigned Scenes with B-NeRF, B-3DGS, EDI-GS, EFN-GS and UFP-GS
in the EvaGaussian-Blender Dataset.

<!-- page 24 -->
Ground Truth
EvA GS(Ours)
EDNeRF
BAD-GS
E2NeRF
Figure 15. Visualization of Novel View Synthesis of All-redesigned Scenes with E2NeRF, BAD-GS, EDNeRF and EvAGS in the
EvaGaussian-Blender Dataset.
