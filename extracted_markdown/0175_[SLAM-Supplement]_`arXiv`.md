<!-- page 1 -->
IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera
Jian Huang1,2
Chengrui Dong1,2
Xuanhua Chen2,3
Peidong Liu2∗
1Zhejiang University
2Westlake University
3Northeastern University
{huangjian39, dongchengrui, chenxuanhua, liupeidong}@westlake.edu.cn
https://github.com/WU-CVGL/IncEventGS
Abstract
Implicit neural representation and explicit 3D Gaussian Splat-
ting (3D-GS) for novel view synthesis have achieved remark-
able progress with frame-based camera (e.g. RGB and RGB-
D cameras) recently. Compared to frame-based camera, a
novel type of bio-inspired visual sensor, i.e. event camera,
has demonstrated advantages in high temporal resolution,
high dynamic range, low power consumption, and low la-
tency, which make it favored for many robotic applications.
In this work, we present IncEventGS, an incremental 3D Gaus-
sian Splatting reconstruction algorithm with a single event
camera, without the assumption of known camera poses. To
recover the 3D scene representation incrementally, we ex-
ploit the tracking and mapping paradigm of conventional
SLAM pipelines for IncEventGS. Given the incoming event
stream, the tracker first estimates an initial camera motion
based on prior reconstructed 3D-GS scene representation.
The mapper then jointly refines both the 3D scene represen-
tation and camera motion based on the previously estimated
motion trajectory from the tracker. The experimental results
demonstrate that IncEventGS delivers superior performance
compared to prior NeRF-based methods and other related
baselines, even if we do not have the ground-truth camera
poses. Furthermore, our method can also deliver better per-
formance compared to state-of-the-art event visual odometry
methods in terms of camera motion estimation.
1. Introduction
Reconstructing accurate 3D scene representations from 2D
images has been a long-standing challenge in computer vi-
sion and robotics, driving substantial efforts over the past few
decades. Among those pioneering works, Neural Radiance
Fields (NeRF) [19] and 3D Gaussian Splatting (3D-GS) [11],
stand out for their utilization of differentiable rendering tech-
niques, and have garnered significant attention due to their
capability to recover high-quality 3D scene representation
*Corresponding author.
from 2D images. Commonly used sensors for 3D scene recon-
struction are usually frame-based cameras, such as the RGB
and RGB-D cameras. They usually capture full-brightness
intensity images within a short exposure time at a regular
frequency. Due to the characteristic of this data-capturing
process, they often suffer from motion blur or fail to capture
accurate and informative intensity information under fast mo-
tion and low-light conditions, which would further affect the
performance of downstream applications.
The event camera, a bio-inspired sensor, has gained signif-
icant attention in recent years for its potential to address the
limitations of frame-based cameras under challenging con-
ditions. Unlike conventional cameras, event cameras record
brightness changes asynchronously at each pixel, emitting
events when a predefined threshold is surpassed. This unique
operation offers several advantages over conventional cam-
eras, in terms of high temporal resolution, high dynamic range,
low latency, and power consumption. Although event cameras
have attractive characteristics for challenging environments,
they cannot be directly integrated into existing frame-based
3D reconstruction algorithms that rely on processing dense
2D brightness intensity images.
Several pioneering works have been proposed to exploit
event stream [3, 12, 23] to recover the motion trajectory and
scene representation. While existing methods deliver im-
pressive performance, they usually exploit 2.5D semi-dense
depth maps to represent the 3D scene, and bundle adjustment
(BA) is hardly performed, due to the asynchronous and sparse
characteristics of event data stream. Klenk et al. [15] re-
cently proposed to convert event stream into event voxel grids,
and then adapt a previous frame-based deep visual odome-
try pipeline [30] for accurate camera motion estimation. As
NeRF exhibited impressive scene representation capability
recently, several works [8, 14, 17, 25] explore to recover the
underlying dense 3D scene NeRF representation from event
stream, by assuming ground-truth poses are available.
In contrast to those works, we propose IncEventGS, an
incremental dense 3D scene reconstruction algorithm from a
single event camera, by exploiting Gaussian Splatting as the
underlying scene representation. Different from prior event-
1
arXiv:2410.08107v4  [cs.CV]  25 Mar 2025

<!-- page 2 -->
based NeRF reconstruction methods, IncEventGS does not
require any ground-truth camera poses, which is more chal-
lenging and provides more flexibility for real-world robotic
application scenarios. To overcome the challenges brought by
unknown poses, IncEventGS adopts the tracking and mapping
paradigm of conventional SLAM pipelines [20]. In partic-
ular, IncEventGS exploits prior explored and reconstructed
3D scenes for camera motion estimation of incoming event
stream during the tracking stage. Both the 3D-GS scene repre-
sentation and camera motions are then jointly optimized (i.e.
event-based bundle adjustment) during the mapping stage, for
more accurate scene representation and motion estimation.
The 3D scene is progressively expanded and densified. The ex-
perimental results on both synthetic and real datasets demon-
strate that IncEventGS can recover the underlying 3D scene
representation and camera motion trajectory accurately. In
particular, IncEventGS outperforms prior NeRF-based meth-
ods and other related baselines in terms of scene representa-
tion recovery, even without ground-truth poses. Furthermore,
our method also delivers better camera motion estimation
accuracy than the most recent state-of-the-art visual odometry
algorithm, in terms of the Absolute Trajectory Error (ATE)
metric. The recovered 3D scene representation can be further
used to render novel brightness images. Our main contribu-
tions can be summarized as follows:
• We present an incremental 3D Gaussian Splatting recon-
struction algorithm from a single event camera, without
requiring the ground-truth camera poses.
• We propose a novel initialization strategy tailored to the
event data stream, which is vital to the success of the algo-
rithm.
• The experimental results on both the synthetic and real
datasets demonstrate the superior performance of our
method over prior NeRF-based methods and related base-
lines in terms of novel view synthesis and better perfor-
mance over state-of-the-art event-based visual odometry
algorithm in terms of camera motion estimation.
2. Related Works
We review two main areas of prior works: event-based neural
radiance fields and 3D Gaussian Splatting, which are the most
related to our work.
Event-based Neural Radiance Fields.
Prior works
[8, 14, 25] propose to exploit event stream to recover the
neural radiance fields with known camera motion trajectory.
Low et al. [17] further improves the reconstruction algorithm
to handle sparse and noisy events under non-uniform mo-
tion. The recovered neural radiance fields can then be used
to render novel view brightness images. The ground-truth
poses are usually computed from corresponding brightness
images via COLMAP [26] or provided by the indoor motion-
capturing system. Recently, Qu et al. [22] proposed to in-
tegrate event measurements into an RGB-D implicit neural
SLAM framework and achieved robust performance in mo-
tion blur scenarios. Li et al. [16] also proposed exploiting
event measurements and a single blurry image to recover the
underlying neural 3D scene representation. In contrast to
those works, IncEventGS conducts incremental 3D scene re-
construction without requiring any prior ground-truth poses,
which is more challenging and provides more flexibility for
practical robotic application scenarios. The method further
exploits 3D Gaussian Splatting as the underlying scene repre-
sentation, which demonstrates better image rendering quality
and efficiency, compared to the NeRF-based representation.
3D Gaussian Splatting.
3D Gaussian Splatting [11] pro-
poses a novel explicit 3D representation to further improve
both the training and rendering efficiency compared to Neural
Radiance Fields. Due to its impressive efficient scene rep-
resentation capability, several pioneering works have been
proposed to exploit 3D-GS for incremental 3D reconstruction.
For example, Keetha et al. [10] propose an RGBD-based 3D-
GS SLAM, employing an online tracking and mapping system
tailored to the underlying Gaussian representation. Yan et
al. [36] implement a coarse-to-fine camera tracking approach
based on the sparse selection of Gaussians. Matsuki et al. [18]
propose to apply 3D Gaussian Splatting to do incremental 3D
reconstruction using a single moving monocular or RGB-D
camera. Huang et al. [7] exploit ORB-SLAM3 to compute
accurate camera poses and feed it into a 3D-GS algorithm
for dense mapping. Fu et al. [2] use monocular depth estima-
tion with 3D-GS. Yugay et al. [37] combine DROID-SLAM
[29] based camera tracking with active and inactive 3D-GS
sub-maps. Wang et al. [33] integrate bundle-adujustment and
3DGS [38] to estimated camera trajectory within the expo-
sure time. Hu et al. [6] propose a novel depth uncertainty
model to ensure the selection of valuable Gaussian primitives
during optimization. While those methods deliver impres-
sive performance in terms of 3D scene recovery and motion
estimation, they usually assume the usage of frame-based
images (i.e. either RGB or RGB-D date). On the contrary,
we propose to exploit pure event measurements for incremen-
tal 3D-GS reconstruction. Several concurrent studies have
recently explored the use of 3D Gaussians for event-based
reconstruction, including EvGGS [31], Event3DGS [35], and
E2GS [1]. EvGGS and Event3DGS rely solely on event data,
whereas E2GS incorporates both event data and blurry images.
However, the key difference between these methods and ours
is that they all rely on ground-truth poses.
3. Method
The overview of our IncEventGS is shown in Fig. 1. Given
only a single event camera, IncEventGS incrementally per-
forms tracking and dense mapping under the framework of 3D
Gaussian Splatting, to recover both the camera motion trajec-
tory and 3D scene representation simultaneously. Since event
data are asynchronous, they cannot be directly integrated with
the 3D-GS representation. We therefore process the event
data stream into chunks according to a fixed time window.
2

<!-- page 3 -->
𝐓𝐓k
𝐓𝐓k+∆𝒕𝒕
−
Continuous Camera Trajectory Model
𝐋𝐋𝐞𝐞𝐞𝐞𝐞𝐞𝐞𝐞𝐞𝐞 𝐋𝐋𝑺𝑺𝑺𝑺𝑺𝑺𝑺𝑺
−
̂𝐈𝐈𝒌𝒌
̂𝐈𝐈𝒌𝒌+∆𝒕𝒕
෠𝐄𝐄(𝐱𝐱)
𝐄𝐄(𝐱𝐱)
Operation Flow
Gradient Flow
Control Knot
Sampled Point
∑
Accumulation
−
Log Difference
−MSE/SSIM Loss
Pretrained
Marigold Model
Boot-
Strapping
Input Event Stream
Time
𝒕𝒕k+∆𝒕𝒕
𝒕𝒕k
3D Gaussians 𝐺𝐺𝜃𝜃
𝐓𝐓𝒏𝒏−𝟐𝟐
𝐓𝐓𝒏𝒏−𝟏𝟏
𝐓𝐓𝒏𝒏
X  Learnable
Optimized Traj.
Initialized Traj.
Initialization
∑
Figure 1. The pipeline of IncEventGS. IncEventGS processes incoming event stream by dividing it into chunks and representing the camera
trajectory as a continuous model. It randomly samples two close consecutive timestamps to integrate the corresponding event streams. Two
brightness images are rendered from 3D-GS at the corresponding poses, and we minimize the photometric loss between the synthesized and
measured brightness change. During initialization, a pre-trained depth estimation model estimates depth from the rendered images to bootstrap
the system.
We associate each chunk with a continuous time trajectory
parameterization in the se(3) space. Two close consecutive
timestamps (i.e., tk and tk+∆t, where ∆t is a small time inter-
val) can be randomly sampled, and the measured brightness
change E(x) can be computed from the corresponding event
stream. Based on the parameterized trajectory, the correspond-
ing camera poses (i.e., Tk, Tk+∆t) can be determined and
the brightness images (i.e., ˆIk, ˆIk+∆t) can be further rendered
from the 3D-GS. The synthesized brightness change ˆE(x)
can be computed for event loss computation.
During the tracking stage, we optimize only the camera
motion trajectory of the newly accumulated event chunk and
exploit the recovered trajectory to initialize the dense bundle
adjustment (BA) algorithm for the mapping stage. During
the mapping stage, we continuously densify 3D Gaussians
for newly explored areas and prune transparent 3D Gaussians.
For computational efficiency, we exploit a sliding window of
the latest chunks and perform BA only within this window for
both 3D-GS reconstruction and motion trajectory estimation.
We will detail each component as follows.
3.1. 3D Scene Representation
Following 3D-GS [11], the scene is represented by a set of 3D
Gaussian primitives, each of which contains mean position
µ ∈R3 in the world coordinate, 3D covariance Σ ∈R3×3,
opacity o ∈R, and color c ∈R3. To ensure that the co-
variance matrix remains positive semi-definite throughout the
gradient descent, the covariance Σ is parameterized using a
scale vector s ∈R3 and rotation matrix R ∈R3×3:
Σ = RSST RT ,
(1)
where scale matrix S = diag([s]) is derived from the scale
vector s ∈R3.
In order to enable rendering, 3D-GS projects 3D Gaussian
primitives to the 2D image plane from a given camera pose
Tc =

Rc ∈R3×3, tc ∈R3	
using following equation:
Σ′ = JRcΣRT
c JT ,
(2)
where Σ′ ∈R2×2 is the 2D covariance matrix, J ∈R2×3
is the Jacobian of the affine approximation of the projective
transformation. After projecting 3D Gaussians onto the image
plane, the color of each pixel is determined by sorting the
Gaussians according to their depth and then applying near-to-
far α-blending rendering via the following equation:
I =
N
X
i
ciαi
i−1
Y
j
(1 −αj) ,
(3)
where ci is the learnable color of each Gaussian, and αi is
the alpha value computed by evaluating the 2D covariance Σ′
multiplied with the learned Gaussian opacity o:
αi = oi · exp (−σi) ,
σi = 1
2∆T
i Σ′−1∆i,
(4)
where ∆i ∈R2 is the offset between the pixel center and the
3

<!-- page 4 -->
2D Gaussian center. Depth is rendered by:
D =
N
X
i
diαi
i−1
Y
j
(1 −αj) ,
(5)
where di denotes the z-depth of the center of the i-th 3D Gaus-
sian to the camera. We also render alpha map to determine
visibility:
V =
N
X
i
αi
i−1
Y
j
(1 −αj) ,
(6)
The derivations presented above demonstrate that the ren-
dered pixel color, denoted as I in Eq. (3), is a function that
is differentiable with respect to the learnable attributes of
3D-GS, and the camera poses Tc. This facilitates our bun-
dle adjustment formulation, accommodating a set of event
chunks and inaccurate camera motion trajectories within the
framework of 3D-GS.
3.2. Event Data Formation Model
An event camera records changes in brightness as a stream of
events asynchronously. To relate 3D-GS representation with
the event stream, we sample two close consecutive timestamps
(i.e., tk and tk+∆t, where ∆t is a small time interval), and the
measured brightness change between ∆t is:
E(x) = C{ei(x, ti, pi)}tk<ti<tk+∆t,
(7)
where e(x, ti, pi) is the ith event within the defined time in-
terval corresponding to pixel x and C is the fixed contrast
threshold. The corresponding camera poses Tk and Tk+∆t
can be interpolated from the camera motion trajectory param-
eterization, and two corresponding brightness images (i.e. ˆIk
and ˆIk+∆t) can be rendered from 3D-GS. The synthesized
brightness change ˆE is modelled as:
ˆE(x) = log(ˆIk+∆t(x)) −log(ˆIk(x)),
(8)
where ˆE(x) depends on the parameters of both the motion
trajectory parameters and 3D-GS, and is differentiable with
respect to them.
Both in tracking and mapping, inspired by the work of [25],
we segment the current event chunks into nseg equal segments
according to the number of events, obtaining nseg timestamps
that correspond to the end of each segment. We then randomly
select one timestamp from these nseg timestamps to serve as
tk+∆t, and we randomly sample an integer nwin between
the integer bounds nlow and nup. The index of tk is equal
to the index of tk+∆t subtract nwin. nseg, nlow and nup are
hyperparameters. This sampling strategy enables the model
to capture both local and global information.
3.3. Camera Motion Trajectory Modeling
Since each event chunk usually contains too many events, we
sample a portion of them according to the total number of
events during optimization. Following [32, 38], we formulate
the corresponding poses (i.e. Tk and Tk+∆t) at the beginning
and end of the sampled event portion within each chunk,
by employing a camera motion trajectory. The trajectory is
represented through linear interpolation between two camera
poses, one at the beginning of the chunk Tstart ∈SE(3) and
the other at the end Tend ∈SE(3). The camera pose at time
tk can thus be expressed as follows:
Tk = Tstart · exp( tk −tstart
tend −tstart
· log(T−1
start · Tend)), (9)
where tstart and tend represent the timestamps corresponding
to the boundary of the event chunk. It follows that Tk is
differentiable with respect to both Tstart and Tend. The ob-
jective of IncEventGS is thus to estimate both Tstart and Tend
for each event chunk, along with the learnable parameters of
3D-GS Gθ.
3.4. Incremental Tracking and Mapping
For both tracking and mapping, we aim to minimize the dif-
ference between the synthesized and measured brightness
changes. In particular, we compute the loss of the latest event
chunk only for the tracking stage and minimize the following
loss function:
T∗
start, T∗
end =
argmin
Tstart,Tend
E(x) −ˆE(x)

2 ,
(10)
where ˆE(x) and E(x) are the synthesized and measured
brightness changes respectively, corresponding to a randomly
sampled event portion within the latest event chunk.
Once the tracking is done, we insert the latest event chunk
to the mapper and exploit the estimated T∗
start and T∗
end as
the initial value of the chunk to perform dense bundle adjust-
ment. For computational consideration, we exploit a sliding
window BA of the latest nw chunks, and nw is a hyperparame-
ter. In particular, we optimize both the motion trajectories and
the 3D-GS jointly by minimizing the following loss functions:
L = (1 −λ)Levent + λLssim,
(11)
Levent =
Ei(x) −ˆEi(x)

2 ,
(12)
Lssim = SSIM(Ei(x), ˆEi(x)),
(13)
where λ is a hyperparameter, SSIM is the structural dissimi-
larity loss [34]. As the event data streams in, we alternatively
perform tracking and mapping.
3.5. System Initialization and Boot-strapping
Conventional frame-based 3D-GS methods usually require
a good initial point cloud and camera poses computed via
COLMAP [27] for initialization. However, they are usually
not easy to obtain if we are only given a single event camera.
We therefore initialize the 3D-GS by sampling point cloud
randomly within a bounding box. The first m event chunks
4

<!-- page 5 -->
(where m is a hyperparameter) are selected for initialization,
and all corresponding camera poses are randomly initialized
to be near the identity matrix. We then minimize the loss
computed by Eq. (11) with respect to the attributes of 3D-GS
and the parameters of camera motion trajectories jointly.
Through experiments, we found that the above initializa-
tion procedure consistently produces satisfactory brightness
images. However, the 3D structure remains of low quality
due to the short baselines of the moving event camera. We
further find that it could potentially affect the performance
of the whole pipeline without a good initial 3D structure as
more event data is received. Therefore, we utilize a monoc-
ular depth estimation network [9] to predict a dense depth
map from the rendered brightness image after the pipeline is
trained for certain iterations. This depth map is then used to
re-initialize the centers of the 3D Gaussians by unprojecting
the pixel depths, after which we repeat the minimization of
Eq. (11) for system bootstrapping. More details about system
re-initialization can be found in our supplementary material.
3D-GS Incrementally Growing.
As the camera moves,
new Gaussians is periodically introduced to cover newly ex-
plored regions. After tracking, we obtain an accurate camera
pose estimate for each new event chunk. The centers of new
Gaussians are determined by:
p = T · π−1(u, du)
(14)
where u ∈R2 is pixel coordinate in the image plane, du
is depth of the 3D point p projecting onto the image plane,
which is rendered by Eq. (5), π−1 denotes camera inverse
projection, T is the camera pose from tracking. To ensure
that new Gaussians are only added in previously unmapped
areas, a visibility mask is computed to guide the expansion of
the Gaussian splatting process, as following:
M(p) = V < λV ,
(15)
where V is the rendered alpha map and λV is a hyperparame-
ter.
4. Experiments
4.1. Experimental Setups.
Implementation Details.
All experiments are conducted
on a desktop PC equipped with a 5.73GHz AMD Ryzen 9
7900x CPU and an NVIDIA RTX 3090 GPU. The first m = 3
event chunks are used for initialization. During the mapping
stage, a sliding window size of nw = 20 is employed for the
bundle adjustment algorithm. The hyperparameters are set
as follows: λ = 0.05, λV = 0.8, and nseg = 100. For the
synthetic dataset, nlow = 400k and nup = 500k, while for
the real dataset, nlow = 60k and nup = 80k. Each event
chunk has a time interval of 50 ms. The learning rate of the
camera poses is set to 1e-4 and that for the attributes of 3D-
GS are set the same as the original 3D-GS work. The number
of optimization steps for initialization is 4500, and that for
tracking and mapping are set to 200 and 1500 respectively.
The contrast threshold C of the event camera is set to 0.1 for
synthetic datasets and 0.2 for real datasets empirically.
Baselines and Evaluation Metrics.
To the best of our
knowledge, there are no existing event-only NeRF or 3D-GS
methods that do not rely on ground-truth poses, making direct
comparisons challenging. Therefore, we conduct a thorough
comparison of our method with several event-based NeRF
approaches, including E-NeRF [14], EventNeRF [25], and
Robust e-NeRF [17], as well as our custom implemented two-
stage method (i.e. E2VID [24] + COLMAP [27] + 3DGS
[11]). E-NeRF, EventNeRF, and Robust e-NeRF leverage
implicit neural radiance fields for 3D scene representation,
requiring ground-truth camera poses for accurate NeRF re-
construction. For E2VID + COLMAP + 3DGS, event data
is first converted into brightness images using E2VID. The
camera poses are then estimated from these images using
COLMAP, and 3D-GS is trained with the generated images
and poses. Both the quantitative and qualitative comparisons
are performed on the synthetic dataset. Since there are no
paired ground truth images for the real dataset, we only per-
form qualitative comparisons on the real dataset. In terms
of motion trajectory evaluations, we use the publicly avail-
able state-of-the-art event-only visual odometry methods, i.e.
DEVO (mono) [15] and ESVO2(stereo) [21], for comparison.
The metrics used for novel view synthesis (NVS) include
the commonly employed PSNR, SSIM, and LPIPS. For mo-
tion trajectory evaluations, we utilize Absolute Trajectory
Error (ATE). To ensure fair comparisons, we employ the eval-
uation code provided by EventNeRF to compute the NVS
metrics, which applies a linear color transformation between
predictions and ground truth. Additionally, we use the public
EVO toolbox [5] to compute the trajectory metrics.
Benchmark Datasets.
To properly evaluate the perfor-
mance of NVS and motion trajectory estimation, we synthe-
sized event data using the 3D scene models from the Replica
dataset [28]. In particular, we exploit the room0, room2, of-
fice0, office2, and office3 scenes. We render high frame rate
RGB images at 1000 Hz with a resolution of 768x480 pix-
els. These images are then converted to grayscale, and the
event data is generated via the events simulator [4]. The con-
trast threshold is set to 0.1. To simulate real-world camera
motions, we exploit the same motion trajectories as that of
NICE-SLAM [39] for data generation.
We use the event dataset provided by TUM-VIE [13] for
real data evaluations, which is also used by E-NeRF and
Robust e-NeRF. TUM-VIE captures the event datasets by a
pair of Prophesee Gen4 HD event cameras with a resolution
of 1280x720 pixels. We only use the left-event camera data
for our experiment.
5

<!-- page 6 -->
room0
room2
office0
office2
office3
E-NeRF
EventNeRF
Robust e-NeRF
E2VID+
COLMAP+3DGS
Ours
GT
Figure 2. Qualitative evaluation of novel view image synthesis on the Replica dataset. The experimental results demonstrate that our method
renders higher-quality images with fewer artifacts compared to event-based NeRF and two-stage approaches.
room0
room2
office0
office2
office3
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
E-NeRF
13.99
0.58
0.51
15.56
0.47
0.58
18.91
0.51
0.57
13.05
0.65
0.44
14.01
0.62
0.48
EventNeRF
17.29
0.62
0.39
16.02
0.54
0.64
18.90
0.43
0.62
15.18
0.66
0.45
16.77
0.73
0.33
Robust e-NeRF
17.26
0.84
0.18
16.43
0.50
0.52
18.93
0.52
0.56
16.81
0.81
0.25
19.22
0.84
0.18
E2VID+
COLMAP+3DGS
14.45
0.44
0.52
15.74
0.51
0.55
18.91
0.31
0.68
14.03
0.57
0.48
13.25
0.47
0.53
Ours
24.31
0.85
0.17
23.75
0.79
0.23
25.64
0.54
0.30
21.74
0.82
0.23
21.18
0.88
0.13
Table 1. NVS performance comparison on Replica dataset. All results are calculated using the same code and ground-truth images. The results
demonstrate that our method outperforms NeRF-based and two-stage methods.
4.2. Ablation Study
We conduct ablation studies to confirm our design choices.
In particular, we study the effect of a monocular depth esti-
mation network for system bootstrapping and event slicing
hyperparameters nlow, nup. The experiments are conducted
with the Replica dataset, and the results are shown in Table 3
and Table 4, respectively.
We found that depth initialization significantly impacts
pose estimation, reducing the Average Trajectory Error (ATE)
from 1.534 cm to 0.064 cm. Additionally, this improvement
in pose estimation leads to a slight enhancement in Novel
View Synthesis (NVS) performance. These results verify
the importance of using depth initialization during the boot-
strapping stage.
We compare several combinations of hyperparameters nlow
and nup, which refer to the range of event slicing window
6

<!-- page 7 -->
room0
room2
office0
office2
office3
1d
3d
6dof
desk
desk2
DEVO
0.289
0.266
0.138
0.281
0.156
0.147
0.303
2.93
0.732
0.201
E2VID+COLMAP
17.93
59.96
105.19
18.414
17.28
4.268
16.90
9.88
21.57
10.13
ESVO2
-
-
-
-
-
0.337
1.066
0.587
1.147
2.506
Ours
0.046
0.067
0.045
0.046
0.054
0.115
0.298
0.251
0.231
0.129
Table 2. Pose accuracy (ATE, cm) on Replica and TUM-VIE datasets. The results demonstrate that our method delivers better performance in
terms of camera motion estimation. Since ESVO2 is a stereo event camera method, it cannot run on the Replica dataset.
Setting
PSNR↑
SSIM↑
LPIPS↓
ATE
full
21.74
0.82
0.23
0.046
w/o
17.80
0.76
0.26
1.534
Table 3. Ablation Study about Depth Initialization. The unit of
ATE is cm. The experimental results demonstrate the effectiveness
of the initialization strategy. It not only improves the quality of
rendered images, but also improves the accuracy of the camera
motion estimation significantly.
Setting
PSNR↑
SSIM↑
LPIPS↓
ATE
1k-10k
16.07
0.64
0.46
0.167
10k-50k
18.41
0.72
0.33
0.079
80k-200k
20.99
0.79
0.25
0.079
400k-500k
21.74
0.82
0.23
0.046
500k-600k
20.95
0.79
0.23
0.050
600k-700k
18.06
0.75
0.28
0.214
Table 4. Ablation Study on Event Slice Window Size (Hyperparame-
ters nlow and nup). The unit of ATE is cm.
size. Table 4 demonstrates that both too small and too large
window sizes negatively impact the performance of Novel
View Synthesis (NVS) and pose estimation. Consequently,
we select nlow = 400k and nup = 500k for our experiments
on the Replica dataset.
4.3. Quantitative Evaluations.
We conduct quantitative evaluations against event NeRF
methods(E-NeRF, EventNeRF, and Robust e-NeRF) and
our custom implemented two-stage method (i.e. E2VID +
COLMAP + 3DGS) in terms of the quality of NVS and pose
estimation performance.
The NVS performance is evaluated on Replica-dataset and
the results are presented in Table 1. It is important to note that
the metrics are lower than those typically observed in stan-
dard NeRF/3D-GS methods for RGB images, primarily due
to the lack of absolute brightness supervision. Even though
NeRF-based methods use ground truth poses for training, In-
cEventGS still significantly outperforms them, highlighting
the advantages of our approach utilizing a 3D Gaussian rep-
resentation. Additionally, our method greatly surpasses two-
stage method that also employs 3D Gaussian representation,
demonstrating superior pose estimation and the effectiveness
of our bundle adjustment technique.
We evaluate pose estimation performance using the ATE
metric on both synthetic and real datasets, comparing our
method with DEVO, ESVO2 and E2VID + COLMAP. The re-
sults, presented in Table 2, show that our method outperforms
both baselines, validating the effectiveness of our incremental
tracking and mapping technique.
4.4. Qualitative Evaluations.
We evaluate our method against event NeRF methods and
two-stage method qualitatively in terms of novel view im-
age synthesis, both on synthetic and real data. The results
are presented in both Fig. 2 and Fig. 3. It demonstrates
that our method can deliver better novel view images, while
event NeRF methods and two-stage method render images
with additional artifacts. Compared to NeRF-based methods,
our approach demonstrates the advantage of IncEventGS by
leveraging 3D Gaussian Splatting as the underlying scene rep-
resentation. In contrast to two-stage method, our dense bundle
adjustment optimizes both 3D Gaussian Splatting and camera
pose using event data, whereas two-stage approaches tend to
accumulate errors over time, as confirmed by the experimental
results. We also provide representative visualization of ATE
error mapped onto trajectories in Fig. 4, both on synthetic
and real dataset. It demonstrates that IncEventGS is able to
recover more accurate motion trajectories.
5. Conclusion
We present the first pose-free 3D Gaussian splatting recon-
struction model based on event camera, i.e. IncEventGS, and
is useful for real world applications. We adopt the tracking
and mapping paradigm in conventional SLAM pipeline to do
incremental motion estimation and 3D scene reconstruction
simultaneously. To handle the continuous and asynchronous
characteristics of event stream, we exploit a continuous tra-
jectory model to model the event data formation process.
The experimental results on both synthetic and real datasets
demonstrate the superior performance of IncEventGS over
prior state-of-the-art methods in terms of high-quality novel
image synthesis and camera pose estimation.
Acknowledgements.
This work was supported in part by
NSFC under Grant 62202389, in part by a grant from the
Westlake University-Muyuan Joint Research Institute, and in
part by the Westlake Education Foundation.
7

<!-- page 8 -->
desk
desk2
1d-trans
3d-trans
6dof
E2VID+
COLMAP+3DGS
E-NeRF
EventNeRF
Robust e-NeRF
Ours
reference
Figure 3. Qualitative evaluation for novel view image synthesis on real dataset. It demonstrates that our method is able to render better images
with fewer artifacts than event NeRF methods and two-stage methods. Note that there are no GT images aligned with the event camera, and we
choose the closest images from the RGB camera and crop them to the same size as the rendered images for visual comparisons.
DEVO
E2VID+COLMAP
ours
office0
6dof
Figure 4. Representative visualization of ATE error mapped onto trajectories for the synthetic (office0) and real (6dof) datasets, generated by
the EVO toolbox using the same ground truth poses, demonstrating the superior performance of our method in pose estimation.
8

<!-- page 9 -->
References
[1] Hiroyuki Deguchi, Mana Masuda, Takuya Nakabayashi, and
Hideo Saito. E2gs: Event enhanced gaussian splatting. In 2024
IEEE International Conference on Image Processing (ICIP),
pages 1676–1682. IEEE, 2024. 2, 1
[2] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros,
and Xiaolong Wang. Colmap-free 3d gaussian splatting. arXiv
preprint arXiv:2312.07504, 2023. 2
[3] Guillermo Gallego, Henri Rebecq, and Davide Scaramuzza. A
unifying contrast maximization framework for event cameras,
with applications to motion, depth and optical flow estimation.
In CVPR, 2018. 1
[4] Daniel Gehrig, Mathias Gehrig, Javier Hidalgo-Carrió, and
Davide Scaramuzza. Video to events: Recycling video datasets
for event cameras. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 3586–
3595, 2020. 5
[5] Michael Grupp.
evo:
Python package for the evalua-
tion of odometry and slam.
https://github.com/
MichaelGrupp/evo, 2017. 5
[6] Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li, Liangjing
Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. Cg-
slam: Efficient dense rgb-d slam in a consistent uncertainty-
aware 3d gaussian field. arXiv preprint arXiv:2403.16095,
2024. 2
[7] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung.
Photo-slam: Real-time simultaneous localization and photore-
alistic mapping for monocular, stereo, and rgb-d cameras. In
Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2024. 2
[8] Inwoo Hwang, Junho Kim, and Young Min Kim. Ev-nerf:
Event based neural radiance field.
In Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer
Vision, pages 837–847, 2023. 1, 2
[9] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger,
Rodrigo Caye Daudt, and Konrad Schindler. Repurposing
diffusion-based image generators for monocular depth estima-
tion. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024. 5, 1
[10] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for
dense rgb-d slam. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024. 2, 1
[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and
George Drettakis. 3d gaussian splatting for real-time radiance
field rendering. ACM Transactions on Graphics, 42(4), 2023.
1, 2, 3, 5
[12] Hanme Kim, Stefan Leutenegger, and Andrew J. Davison.
Real-time 3d reconstruction and 6-dof tracking with an event
camera. In ECCV, 2016. 1
[13] Simon Klenk, Jason Chui, Nikolaus Demmel, and Daniel Cre-
mers. Tum-vie: The tum stereo visual-inertial event dataset. In
2021 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), pages 8601–8608. IEEE, 2021. 5
[14] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel
Cremers. E-nerf: Neural radiance fields from a moving event
camera. IEEE Robotics and Automation Letters, 8(3):1587–
1594, 2023. 1, 2, 5
[15] Simon Klenk, Marvin Motzet, Lukas Koestler, and Daniel
Cremers. Deep event visual odometry. In 2024 International
Conference on 3D Vision (3DV), pages 739–749. IEEE, 2024.
1, 5
[16] Wenpu Li, Pian Wan, Peng Wang, Jinghang Li, Yi Zhou, and
Peidong Liu. Benerf: Neural radiance fields from a single
blurry image and event stream. In European Conference on
Computer Vision (ECCV), 2024. 2
[17] Weng Fei Low and Gim Hee Lee. Robust e-nerf: Nerf from
sparse and noisy events under non-uniform motion. In Proceed-
ings of the IEEE/CVF International Conference on Computer
Vision (ICCV), 2023. 1, 2, 5
[18] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J
Davison.
Gaussian splatting slam.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition, pages 18039–18048, 2024. 2, 1
[19] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view synthe-
sis. Communications of the ACM, 65(1):99–106, 2021. 1
[20] Raul Mur-Artal and Juan D Tardós. Orb-slam2: An open-
source slam system for monocular, stereo, and rgb-d cameras.
IEEE Transactions on Robotics, 33(5):1255–1262, 2017. 2
[21] Junkai Niu, Sheng Zhong, Xiuyuan Lu, Shaojie Shen,
Guillermo Gallego, and Yi Zhou. Esvo2: Direct visual-inertial
odometry with stereo event cameras. IEEE Transactions on
Robotics, 2025. 5
[22] Delin Qu, Chi Yan, Dong Wang, Jie Yin, Dan Xu, Bin Zhao,
and Xuelong Li. Implicit event-rgbd neural slam. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024. 2
[23] Henri Rebecq, Timo Horstschaefer, Guillermo Gallego, and
Davide Scaramuzza. Evo: A geometric approach to event-
based 6-dof parallel tracking and mapping in real time. IEEE
Robotics and Automation Letters, 2017. 1
[24] Henri Rebecq, René Ranftl, Vladlen Koltun, and Davide Scara-
muzza. High speed and high dynamic range video with an
event camera. IEEE transactions on pattern analysis and ma-
chine intelligence, 43(6):1964–1980, 2019. 5
[25] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and
Vladislav Golyanik. Eventnerf: Neural radiance fields from a
single colour event camera. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 4992–5002, 2023. 1, 2, 4, 5
[26] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion Revisited. In CVPR, pages 4104–4113, 2016.
2
[27] Johannes Lutz Schönberger and Jan-Michael Frahm. Structure-
from-motion revisited. In Conference on Computer Vision and
Pattern Recognition (CVPR), 2016. 4, 5
[28] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl
Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian
Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kim-
berly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham,
Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Ba-
tra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele,
Steven Lovegrove, and Richard Newcombe.
The Replica
9

<!-- page 10 -->
dataset: A digital replica of indoor spaces. arXiv preprint
arXiv:1906.05797, 2019. 5
[29] Zachary Teed and Jia Deng. DROID-SLAM: Deep Visual
SLAM for Monocular, Stereo, and RGB-D Cameras. Advances
in neural information processing systems, 2021. 2
[30] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch visual
odometry. Advances in Neural Information Processing Systems,
2023. 1
[31] Jiaxu Wang, Junhao He, Ziyi Zhang, Mingyuan Sun, Jingkai
Sun, and Renjing Xu. Evggs: A collaborative learning frame-
work for event-based generalizable gaussian splatting. arXiv
preprint arXiv:2405.14959, 2024. 2, 1
[32] Peng Wang, Lingzhe Zhao, Ruijie Ma, and Peidong Liu. BAD-
NeRF: Bundle Adjusted Deblur Neural Radiance Fields. In
Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 4170–4179, 2023. 4
[33] Peng Wang, Lingzhe Zhao, Yin Zhang, Shiyu Zhao, and
Peidong Liu.
Mba-slam: Motion blur aware dense visual
slam with radiance fields representation.
arXiv preprint
arXiv:2411.08279, 2024. 2
[34] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 4
[35] Tianyi Xiong, Jiayi Wu, Botao He, Cornelia Fermuller, Yian-
nis Aloimonos, Heng Huang, and Christopher A Metzler.
Event3dgs: Event-based 3d gaussian splatting for fast ego-
motion. arXiv preprint arXiv:2406.02972, 2024. 2, 1
[36] Chi Yan, Delin Qu, Dong Wang, Dan Xu, Zhigang Wang, Bin
Zhao, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, 2024. 2
[37] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Oswald.
Gaussian-slam: Photo-realistic dense slam with gaussian splat-
ting. arXiv preprint arXiv:2312.10070, 2023. 2
[38] Lingzhe Zhao, Peng Wang, and Peidong Liu. Bad-gaussians:
Bundle adjusted deblur gaussian splatting. In European Con-
ference on Computer Vision (ECCV), 2024. 2, 4
[39] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun
Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys.
Nice-slam: Neural implicit scalable encoding for slam. In
Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 12786–12796, 2022. 5
10

<!-- page 11 -->
IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera
Supplementary Material
Continuous Camera Trajectory Model
Pretrained
Marigold Model
3D Gaussians
𝐓𝐓𝟏𝟏
𝐓𝐓𝟐𝟐
Figure 5. The re-initialization process of IncEventGS.
6. More Details about Re-initialization
The re-initialization process is illustrated in Fig. 5. After the
first-time initialization, we can render a brightness image from
3D-GS at pose T1, where T1 represents the camera pose at
the end of the first event chunk. To improve the 3D structure of
3D-GS, we use a monocular depth estimation network [9] to
predict a dense depth map from the rendered brightness image.
This depth map is then used to re-initialize the centers of the
3D Gaussians by unprojecting the pixel depths at camera
pose T1, as illustrated in Fig. 5. After re-centering the 3D
Gaussians, we perform the initialization process again to
achieve both accurate 3D structure and exceptional brightness
image rendering performance.
7. Comparison with Gaussian-based Event
Methods
To further evaluate our method, we conducted additional
comparisons against state-of-the-art Gaussian-based event ap-
proaches. Since Event3DGS [35] had not been open-sourced,
we chose to compare against E2GS[1] and EvGGS[31]. In
particular, we removed the supervision of blurred image in
E2GS and exploited the pretrained weight of EvGGS for
comparisons. As shown in Table 6, our method still outper-
forms those two baselines event though they used ground truth
poses. Since EvGGS is a generalizable method based on a
feed-forward network, it has limited generalization capability
on unseen dataset.
8. Experiments in Fast-Motion Scenarios
Fast camera movement can induce motion blur, making
it challenging to reconstruct the scene and estimate camera
poses using RGB-based algorithms. We compare our event-
based method with two state-of-the-art pose-free Gaussian
SLAM implementations: MonoGS [18] (RGB modality) and
Method
Synthetic (768 × 480)
Real-world (1280 × 720)
Training
Storage
Training
Storage
ENeRF
12 hour
253M
12 hour
253M
EventNeRF
21 hour
14M
24 hour
14M
Robust e-NeRF
11 hour
745M
13 hour
745M
Ours
0.5 hour
65M
2 hour
55M
Table 5. Average model efficiency comparison.
SplaTAM [10] (RGBD modality). By leveraging the high
temporal resolution of event cameras, our method experiences
minimal performance degradation, even under fast motion.
Additionally, it is more effective at preserving high-frequency
information in the scene. As shown in Fig. 6, our approach
delivers superior novel view synthesis results, particularly
during rapid camera movement.
9. Experiments on Color Event Datasets
Our method can also be applied to color event datasets by
integrating the Bayer filter [25], as shown below:
Levent =
F ⊙Ei(x) −F ⊙ˆEi(x)

2
(16)
Lssim = SSIM(F ⊙Ei(x), F ⊙ˆEi(x))
(17)
Furthermore, our method can be extended to incorporate train-
ing with ground-truth poses.
We conducted experiments on the EventNeRF dataset
[25], which focuses on object reconstruction. Due to the
dataset’s limited features, pose estimation is challenging; nei-
ther COLMAP nor DEVO can estimate camera poses on this
dataset. As shown in Fig. 7, our method can still successfully
optimize both the 3D scene and camera poses even without
ground-truth poses, though it produces minor artifacts. When
trained with ground-truth poses, our method achieves im-
proved novel view synthesis, with fewer artifacts and sharper
textures.
10. Time Evaluations
As shown in Table 5, our method has a significant ad-
vantage in training time compared to NeRF-based methods.
Additionally, our method achieves an NVS rendering speed of
approximately 500 FPS, whereas NeRF-based methods reach
only about 0.5 FPS.
We mainly focus on demonstrating the effectiveness (i.e.
in terms of novel view synthesis and pose estimation) by
exploiting 3D-GS representation for event camera, and have
not tried to improve the efficiency of the proposed method. In
1

<!-- page 12 -->
room0
room2
office0
office2
office3
SplaTAM
MonoGS
Ours
Blur
GT
Figure 6. Qualitative evaluation of novel view image synthesis on the Replica dataset. The experimental results demonstrate that our method
renders higher-quality images when the camera is moving fast.
lego
materials
drums
chair
ficus
Ours(wo)
Ours(w)
EventNeRF
GT RGB
Figure 7. Qualitative evaluation of novel view image synthesis on color event dataset. Ours (wo) denotes our method trained without
ground-truth camera poses, while Ours (w) denotes the method trained with ground-truth camera poses.
2

<!-- page 13 -->
room0
room2
office0
office2
office3
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
E2GS*
21.75
0.77
0.25
23.11
0.82
0.20
20.09
0.75
0.18
18.62
0.78
0.20
20.13
0.84
0.16
EvGGS
15.16
0.37
0.62
15.85
0.34
0.61
18.51
0.37
0.59
10.95
0.27
0.69
13.13
0.29
0.66
Ours
24.31
0.85
0.17
23.75
0.79
0.23
25.64
0.54
0.30
21.74
0.82
0.23
21.18
0.88
0.13
Table 6. NVS performance comparison on Replica dataset. * denotes we removed the supervision of blurred images from the original
E2GS.The result demonstrates that our method outperforms those two baseline methods.
particular, for the ease of the development, we still adopt the
Adam optimizer with a small learning rate (i.e. 1e-4) from
PyTorch for both motion and 3D-GS estimation. It requires
around 0.3s and 1.7s per event chunk to converge for both
tracking and mapping respectively. We would further improve
the efficiency by using a second-order optimization method
(e.g. levenberg-marquardt algorithm), which has been proved
to converge much faster to the optimal solution compared to
an first-order optimizer (e.g. Adam).
3
