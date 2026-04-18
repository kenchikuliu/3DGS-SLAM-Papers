<!-- page 1 -->
1
GeMS: Efficient Gaussian Splatting for Extreme Motion Blur
Gopi Raju Matta, Trisha Reddypalli, Vemunuri Divya Madhuri, and Kaushik Mitra
Abstract—We introduce GeMS, a framework for 3D Gaussian
Splatting designed to handle severely motion-blurred images.
State-of-the-art deblurring method for extreme motion blur, such
as ExBluRF, as well as Gaussian Splatting-based approaches
like Deblur-GS, typically assume access to corresponding sharp
images for camera pose estimation and point cloud generation,
which is an unrealistic assumption. Additionally, methods relying
on COLMAP initialization, such as BAD-Gaussians, fail due to
the lack of reliable feature correspondences in cases of severe
motion blur. To address these challenges, we propose GeMS, a 3D
Gaussian Splatting (3DGS) framework that reconstructs scenes
directly from extremely motion-blurred images. GeMS integrates:
(1) VGGSfM, a deep learning-based Structure from Motion
(SfM) pipeline which estimates camera poses and generates point
clouds directly from severely motion-blurred images; (2) 3DGS-
MCMC (Markov Chain Monte Carlo) enables robust scene ini-
tialization by treating Gaussians as samples from an underlying
probability distribution, eliminating heuristic densification and
pruning strategies; and (3) Joint optimization of camera motion
trajectory and Gaussian parameters which ensures stable and
accurate reconstruction. While this pipeline produces reasonable
reconstructions, extreme motion blur can still introduce inaccu-
racies, especially when all input views are severely blurred. To
address this, we propose GeMS-E, which integrates a progressive
refinement step when event data is available. Specifically, we
perform (4) Event-based Double Integral (EDI) deblurring, which
first restores deblurred images from motion-blurred inputs using
events. These deblurred images are then fed into the GeMS
framework, leading to improved pose estimation, point cloud
generation, and hence overall reconstruction quality. Both GeMS
& GeMS-E achieve state-of-the-art performance on synthetic as
well as real-world datasets, demonstrating their effectiveness in
handling extreme motion blur. To the best of our knowledge,
we are the first to effectively address this motion deblurring
problem in extreme blur scenarios within a 3D Gaussian Splatting
framework directly from severely motion blurred images.
I. INTRODUCTION
Motion blur is a fundamental and unsolved challenge for
3D scene reconstruction and novel view synthesis, especially
in real-world scenarios involving high-speed camera motion
or low-light conditions. Such conditions are ubiquitous in
practical settings ranging from robotics and autonomous vehi-
cles to handheld photography, where capturing sharp images
is often impossible. Robust 3D reconstruction from motion-
blurred inputs is therefore essential for advancing computer
vision systems.
Despite recent advances such as Neural Radiance Fields
(NeRF) [1] and 3D Gaussian Splatting (3DGS) [2], these
methods fundamentally rely on sharp images and accu-
rate camera poses. Traditional Structure-from-Motion (SfM)
pipelines like COLMAP [3] are especially brittle under heavy
blur: they depend on reliable feature correspondences for
Computational Imaging Lab, Department of Electrical Engineering, IIT
Madras, Chennai, India.
keypoint detection and matching, but blur severely degrades
texture information, leading to unreliable matches, poor pose
estimation, and ultimately, failure to reconstruct the scene. As
a result, there is currently no practical solution for robust 3D
reconstruction from extremely motion-blurred images without
sharp-image supervision, a critical gap in the literature.
To address this open problem, we introduce GeMS, a funda-
mentally new, efficient 3D Gaussian Splatting framework that
reconstructs sharp 3D scenes directly from severely blurred
inputs. Our approach is not a simple stacking of existing
techniques, but a tightly integrated, self-correcting system built
on three core innovations:
Blur-Robust Initialization with VGGSfM: We replace
COLMAP’s brittle feature matching with VGGSfM [4], a
deep, differentiable SfM framework. VGGSfM’s learned 2D
point tracking and end-to-end optimization provide robust,
blur-tolerant initialization of camera poses and point clouds,
succeeding where all classical and many recent methods fail.
This enables downstream modules to operate even in the
presence of extreme motion blur.
Probabilistic Scene Modeling with 3DGS-MCMC: We in-
corporate a probabilistic formulation using 3DGS-MCMC [5],
which treats Gaussians as samples from a scene distribution
and uses Markov Chain Monte Carlo (MCMC) sampling
to adaptively densify and refine geometry. This avoids the
heuristic and brittle densification strategies of prior work,
maintaining high reconstruction quality even when the input
is sparse or noisy due to extreme motion blur.
Joint Trajectory-Geometry Optimization: Because mo-
tion blur arises from continuous camera motion during
exposure, we jointly optimize camera trajectories (using
B´ezier curves) and Gaussian parameters with physics-based
losses [6]. This joint optimization is essential for aligning the
reconstructed geometry with the actual blur formation process
and correcting errors that would otherwise propagate through
the pipeline.
The synergy of these components is critical: VGGSfM’s
robust initialization enables effective probabilistic refinement,
while joint optimization ensures global consistency and error
correction. Our systematic experiments across a spectrum of
blur levels demonstrate that each module is necessary, and
their integration yields state-of-the-art performance. GeMS
consistently outperforms both traditional and recent methods,
especially as blur severity increases.
For extreme blur cases where all input views are severely
blurred, we further extend our framework to GeMS-E, when
event data is available. By incorporating event-based deblur-
ring using the Event-based Double Integral (EDI) model [7],
GeMS-E leverages high-temporal-resolution event data to re-
cover sharp images. These EDI deblurred images are seam-
lessly fed to our GeMs framework. This extension outperforms
arXiv:2508.14682v1  [cs.CV]  20 Aug 2025

<!-- page 2 -->
2
Input
GeMS
GeMS-E
Ground Truth
Fig. 1: Overview: GeMS reconstructs sharp 3D scenes directly from extremely motion-blurred images without relying on COLMAP or
sharp image supervision. GeMS-E leverages event streams, when available, to further enhance reconstruction quality in the most challenging
cases.
all event-driven baselines in both synthetic and real-world
datasets.
Comprehensive experiments demonstrate that GeMS and
GeMS-E achieve state-of-the-art results, outperforming exist-
ing methods in both synthetic and real-world motion-blur sce-
narios. Our GeMs/GeMS-E results are illustrated in Figure 1
Our key contributions are as follows:
• We propose GeMS, an efficient 3D Gaussian Splatting
framework that enhances robustness in extreme motion-
blurred scenarios by leveraging VGGSfM for camera
pose estimation and point cloud generation without sharp
image supervision. The framework further incorporates
MCMC-based Gaussian initialization and optimization,
along with joint optimization of camera poses and Gaus-
sian parameters, to effectively restore sharp scene repre-
sentations.
• When event data is available, GeMS-E integrates event-
based deblurring using the EDI model, enabling sharp
image reconstruction and novel sharp view synthesis even
in extreme motion blur conditions, particularly in cases
where all input views are severely blurred. To further sup-
port research in this area, we introduce a complementary
synthetic event dataset EveGeMS specifically curated for
extreme blur scenarios.
• We conduct a systematic analysis across multiple blur
levels, demonstrating the robustness of both VGGSfM
and 3DGS-MCMC modules under severe motion blur.
• Through extensive evaluations on both synthetic and real-
world datasets, we demonstrate superior performance in
deblurring and novel view synthesis, achieving state-of-
the-art reconstruction quality while significantly improv-
ing computational and memory efficiency.
To the best of our knowledge, this is the first work to
effectively address the challenge of reconstructing sharp 3D
scene from extremely motion blurred images within a 3D
Gaussian Splatting framework, without relying on impractical
sharp image supervision for pose and point cloud initialization.
II. RELATED WORK
A. SfM for NeRF/3DGS Initialization
Accurate camera pose estimation and sparse 3D recon-
struction are essential for initializing NeRF and 3D Gaussian
Splatting (3DGS). COLMAP [3] is the most widely used
incremental SfM pipeline, but its reliance on SIFT-based
feature matching makes it vulnerable to motion blur and low-
texture failures. GLOMAP [8] improves scalability with a
global SfM approach, while HLOC [9] enhances wide-baseline
localization using SuperGlue [10]. However, both methods
remain dependent on pairwise feature correspondences, limit-
ing their effectiveness under extreme motion blur. To address
this, Pixel-Perfect SfM [11] refines both keypoint locations
and camera poses by optimizing a featuremetric error using
dense deep features. This improves the geometric accuracy
of SfM pipelines, making it more robust to detection noise
and appearance variations. Further advancing SfM, VGGSfM
[4] replaces traditional keypoint matching with deep 2D
point tracking, eliminating the need for pairwise matching.
It jointly optimizes camera poses and 3D points via a fully
differentiable bundle adjustment layer, achieving state-of-the-
art performance on CO3D [12], IMC Phototourism [13], and
ETH3D [14]. Given its superior robustness in extreme motion
blur scenarios, VGGSfM is the preferred SfM method for
initializing 3DGS in our deblurring pipeline.
B. Novel View Synthesis
NeRF [1] has garnered significant attention in 3D vision
due to its remarkable ability to generate photo-realistic novel
views. At its core, NeRF employs a neural implicit representa-
tion optimized via differentiable volume rendering. Numerous
works have sought to enhance its rendering quality [15]–[19],
while others have focused on accelerating both training and
rendering [20]–[26], leading to significant improvements in
efficiency.
Recently, 3D Gaussian Splatting (3DGS) [27] has emerged
as an efficient alternative to radiance field models, excelling
in both fine-grained scene reconstruction and real-time ren-
dering. By replacing NeRF’s computationally expensive ray
marching [28] with a deterministic rasterization technique,
3DGS preserves visual fidelity while enabling rapid rendering.
However, 3DGS requires accurate camera poses and point
cloud initialization for effective reconstruction, which can be
challenging in motion-blurred scenarios. To further enhance
stability and accuracy, recent works have explored MCMC-
based Gaussian Splatting [5], treating Gaussians as proba-

<!-- page 3 -->
3
    
Deep Camera 
Predictor
Initial Cameras,
Initial point cloud
Differentiable 
Bundle Adjustment
 Cameras, point cloud
Training Loss
Deep 
Triangulator
 
Camera Trajectory
Averaging
Loss
Initialization
Input
Projection
Differentiable 
Rasterization
MCMC Update
Render
     
    
VGGSfM
Virtual Images
Operation Flow
SGLD Flow
Deep Point Tracker
GeMS-E
Point cloud
GeMS
Event Bins
EDI
Event Frames
Blurry Images
EDI Frames
Tracks
3D Gaussians
Cameras
Fig. 2: Our Method: GeMS addresses extreme motion blur in 3D Gaussian Splatting framework. GeMS directly optimizes Gaussians on
blurred images without requiring COLMAP or sharp supervision, leveraging VGGSfM for robust SfM initialization, 3DGS-MCMC with
joint optimization for effective refinement of camera poses and Gaussian parameters. When event data is available, GeMS-E enhances this
by incorporating the EDI model to recover deblurred images, which are then fed into the GeMS pipeline for improved reconstructions.
bilistic samples for more robust initialization and optimization.
Inspired by this, our method integrates MCMC-based Gaus-
sian Splatting, improving reconstruction quality in challenging
motion-blurred scenarios.
C. Radiance Field Deblurring
NeRF-based approaches have explored various techniques
to reconstruct sharp scene representations from motion-blurred
images. Deblur-NeRF [29] and ExBluRF [30] model the blur
process during scene optimization but assume fixed camera
poses, which can be inaccurate under severe motion blur. To
address this, BAD-NeRF [31] jointly optimizes camera motion
and radiance fields, allowing pose refinement. However, its
reliance on implicit MLP-based representations leads to slow
optimization and rendering times. Explicit representations,
such as 3D Gaussian Splatting, have recently emerged as effi-
cient alternatives for real-time rendering. BAD-Gaussians [6]
extends 3DGS by jointly optimizing Gaussians and camera
trajectories, improving both efficiency and reconstruction qual-
ity. Similarly, Deblur-GS [32] refines Gaussian parameters to
achieve sharper reconstructions but still relies on COLMAP
for initialization, making it ineffective in extreme motion blur
scenarios.
Event-based methods have also been introduced to tackle
motion blur in 3D reconstruction. E2NeRF [33] and EBAD-
NeRF [34] integrate event streams into NeRF-based frame-
works, leveraging high temporal resolution for deblurring.
However, they inherit the inefficiencies of NeRF’s implicit
representations. More recently, E2GS [35] has attempted to
incorporate event data into Gaussian Splatting, but it fails
to effectively remove motion blur, introducing severe color
artifacts. To overcome these challenges, we propose a novel
Gaussian Splatting framework that directly processes motion-
blurred images while incorporating event-based deblurring in
extreme cases where event data is available. By leveraging
VGGSfM for robust pose estimation and MCMC-based Gaus-
sian Splatting for adaptive initialization, our approach jointly
optimizes camera poses and Gaussian parameters, overcoming
the limitations of previous methods and achieving high-quality,
real-time 3D reconstruction under severe motion blur.
III. BACKGROUND
A. 3D Gaussian Scene Representation
In the 3DGS framework [27], a scene is modeled as a
collection of 3D Gaussian distributions, each described by its
mean position µ ∈R3, a 3D covariance matrix Σ ∈R3×3,
opacity o ∈R, and color c ∈R3. The distribution of each
Gaussian is represented by the following formula:
G(x) = e−1
2 (x−µ)⊤Σ−1(x−µ)
(1)
The covariance matrix Σ is parameterized by a scale matrix
S ∈R3 and a rotation matrix R ∈R3×3. The rotation
matrix R is represented using a quaternion q ∈R4, while
the decomposition
Σ = RSST RT
(2)
ensures that Σ remains positive definite.
For rendering, 3D Gaussians are projected into 2D space
from the camera pose Tc = {Rc ∈R3×3, tc ∈R3} using the
following equations:
Σ′ = JRcΣRT
c JT
(3)
where Σ′ ∈R2×2 is the 2D covariance matrix and J ∈R2×3
is the Jacobian matrix for the projection.
To compute the rendered pixel colors, the 2D Gaussians are
rasterized based on their depth values:
C =
N
X
i=1
ciαi
i−1
Y
j=1
(1 −αj)
(4)

<!-- page 4 -->
4
where ci is the color of the i-th Gaussian, and αi is the
corresponding alpha value:
αi = oi · exp(−σi),
σi = 1
2∆T
i Σ′−1∆i
(5)
Here, ∆i ∈R2 represents the distance between the pixel
center and the center of the 2D Gaussian. The pixel color C
is differentiable with respect to both the Gaussian parameters
G and the camera pose Tc.
IV. METHOD
GeMS is an efficient 3D Gaussian Splatting framework for
deblurring and sharp novel view synthesis under extreme mo-
tion blur. It leverages VGGSfM [4] for camera pose estimation
and point cloud generation, followed by MCMC-based Gaus-
sian initialization and optimization [5]. A joint optimization
of camera trajectories and Gaussian parameters further re-
fines the reconstruction, ensuring robustness in highly blurred
scenarios. GeMS-E (Event-Assisted Deblurring): When event
data is available, we first recover sharp images using the EDI
model [7]. These deblurred images are then processed through
our GeMS framework to facilitate both motion deblurring and
sharp novel view synthesis, even in the presence of extreme
motion blur across all input images. Our overall framework is
illustrated in Figure 2. Background of 3D Gaussian Splatting
is presented in III-A.
A. GeMS (Direct Deblurring)
COLMAP [3] is the de facto choice for structure-from-
motion (SfM) initialization in NeRF and 3D Gaussian Splat-
ting (3DGS) pipelines. However, it requires sharp, high-quality
images for reliable feature matching, rendering it ineffective
under severe motion blur. Consequently, state-of-the-art de-
blurring methods that rely on COLMAP become ineffective in
such challenging scenarios. In contrast, our proposed GeMS
framework eliminates dependence on COLMAP by employ-
ing a robust and efficient Gaussian Splatting pipeline that
integrates VGGSfM with 3DGS-MCMC based optimization.
We then jointly optimize Gaussian parameters and camera
motion trajectories to refine pose inaccuracies. This systematic
approach enables accurate and sharp 3D scene reconstruction
directly from severely motion-blurred images.
1) Robust Initialization and Optimization with VGGSfM
and 3DGS-MCMC: Accurate initialization is critical for 3D
Gaussian Splatting (3DGS), especially in scenarios with severe
motion blur where traditional methods like COLMAP fail
due to unreliable feature correspondences. This limitation
affects several state-of-the-art deblurring pipelines such as
ExBluRF [30] and BAD-Gaussians [6], which depend on
COLMAP initialization. To overcome this, we integrate two
complementary techniques: VGGSfM, a fully differentiable
deep-learning-based Structure-from-Motion (SfM) pipeline for
robust camera pose estimation and point cloud generation;
and MCMC-based Gaussian Splatting with joint optimization,
which treats Gaussians as probabilistic samples and jointly
optimizes camera poses and Gaussian parameters for adaptive
and accurate scene reconstruction.
VGGSfM for SfM initialization:
Unlike traditional SfM
pipelines that rely on incremental image registration and brittle
feature matching, VGGSfM estimates all camera poses in an
end-to-end manner. It leverages recent advances in deep 2D
point tracking to extract reliable, pixel-accurate tracks without
explicit pairwise feature matching, ensuring robustness even
when image textures are severely degraded by blur. Rather
than gradual registration, VGGSfM employs a Transformer-
based model for global pose estimation, significantly improv-
ing stability under motion blur. Furthermore, it integrates a
differentiable bundle adjustment layer based on the Theseus
solver, enabling joint optimization of camera poses and 3D
points within a learning framework. This architecture allows
VGGSfM to produce valid initializations at extreme blur levels
where COLMAP fails. We presented systematic evaluations to
validate the effectiveness of VGGSfM at various blur levels
in Section V-D1.
3DGS-MCMC for joint optimization:
Building on the
3DGS-MCMC framework [5], we reinterpret Gaussian splat-
ting as a probabilistic sampling process. Gaussians are treated
as samples drawn from an underlying scene distribution
and updated using Stochastic Gradient Langevin Dynamics
(SGLD). This probabilistic approach allows Gaussians to
dynamically relocate to high-likelihood regions, eliminating
the need for heuristic cloning or pruning. Systematic eval-
uations with resepct to various blur levels are presented in
Section V-D1.
To address inaccuracies in camera poses introduced by
severe motion blur, we incorporate joint optimization of B´ezier
motion trajectories and Gaussian parameters. The joint opti-
mization is essential because motion blur fundamentally arises
from continuous camera motion during exposure. Initial poses
from VGGSfM provide only discrete viewpoints, failing to
capture the true motion path responsible for observed blur.
By jointly optimizing trajectories and geometry, we directly
model the physics of motion blur, where each blurry image
integrates sharp scene content along the camera’s continuous
trajectory. This co-adaptation enables iterative correction of
both trajectory errors and geometry misalignments, ensuring
the synthesized blur matches real input blur and significantly
reducing artifacts. By modeling and optimizing the camera
trajectory alongside scene geometry, we ensure geometric and
photometric consistency and improve reconstruction accuracy.
Synergistic integration of VGGSfM and 3DGS-MCMC
with joint optimization:
The integration of VGGSfM and
3DGS-MCMC with joint optimization creates a truly syner-
gistic, end-to-end differentiable pipeline that goes far beyond
simply stacking existing techniques. After initialization with
VGGSfM, the probabilistic scene modeling of 3DGS-MCMC
and the joint optimization of camera poses and scene geometry
are tightly coupled, allowing gradients and error signals to
flow seamlessly across modules during training. This design
enables each part of the system to actively compensate for
the limitations of the others: 3DGS-MCMC adaptively refines
and densifies the often noisy or sparse outputs from VGGSfM,
while joint optimization co-refines B´ezier camera trajectories
and Gaussian scene parameters to ensure physical and photo-

<!-- page 5 -->
5
Bench
Jars
Jars2
Postbox
Blurry RGB Image
Event Frame Sequence
Fig. 3: Our Complementary Event Dataset (EveGeMS): We present a synthetic event dataset designed for scenarios involving extreme
motion blur. While the RGB frames suffer from severe blur, each is accompanied by a sequence of event frames that preserve fine structural
and motion cues. This complementary information facilitates robust and accurate deblurring and novel view synthesis especially where all
input views are severely blurred.
metric consistency with the observed data.
2) Physical Motion Blur Image Formation Model: The
process of image formation in a digital camera involves the
accumulation of photons during the exposure period, which are
subsequently converted into electrical signals. Mathematically,
this can be expressed as an integration over a sequence of
virtual latent sharp images:
B(u) = ϕ
Z τ
0
Ct(u)dt
(6)
where B(u) ∈RH×W×3 represents the captured motion-
blurred image, with u ∈R2 denoting the pixel location in an
image of height H and width W. Here, ϕ is a normalization
factor, τ is the exposure time, and Ct(u)
∈
RH×W×3
corresponds to the latent sharp image at a given timestamp
t ∈[0, τ]. The motion-blurred image B(u) arises due to
camera movement during the exposure and is effectively the
average of all latent sharp images Ct(u) over time. In practice,
this integral is approximated using a finite number n of
discrete samples, leading to the following discrete formulation:
B(u) ≈1
n
n−1
X
i=0
Ci(u)
(7)
The extent of motion blur in an image is influenced by
the camera’s movement during exposure. A rapidly moving
camera within a given exposure time creates severe motion
blur. Note that B(u) remains differentiable with respect to
each virtual sharp image Ci(u), which is a key property for
optimization in motion deblurring tasks.
3) Camera Motion Trajectory Modeling in 3DGS-MCMC
for Pose optimization: To model camera motion during ex-
posure, we parameterize the pose in Special Euclidean group
in 3 dimensions (SE(3)), which is a mathematical structure
that describes all possible rigid body transformations in 3D
space. While BAD-Gaussians [6] employ linear and cubic
spline interpolation for trajectory estimation, these methods
prove inadequate for handling severe motion blur. Hence our
method adopts B´ezier curve interpolation inspired from [32],
which provides a smoother and more accurate representation.
Given a B´ezier curve of degree M, the camera motion is
represented using M + 1 control points Tj (j = 0, ..., M).
The interpolated camera pose at time t is given as:
Tt =
M
Y
j=0
exp
M
j

(1 −u)M−juj · log(Tj)

(8)
where u = t/τ
∈[0, 1] and τ is the exposure time.
This formulation ensures smooth motion trajectory estimation
while remaining differentiable, enabling joint optimization for
accurate deblurring.
4) Loss Functions: Given a set of K motion-blurred im-
ages, the goal is to jointly estimate the camera motion trajec-
tory for each image and the learnable parameters of 3DGS, θ
(i.e., mean position µ, 3D covariance Σ, opacity o, and color
c). For this joint estimation framework, we draw inspiration
from BAD-Gaussians [6]. This estimation is achieved by
minimizing the following loss function, which combines an
L1 loss with a D-SSIM (1-SSIM) term. The loss is computed
between Bk(u), the kth synthesized blurry image generated
via 3DGS and its corresponding real captured counterpart,
Bgt
k (u).
L = (1 −λ)L1 + λLD-SSIM
(9)
5) Joint Optimization:
To optimize both the learnable
Gaussian parameters θ and the camera poses T (represented
using a B´ezier curve of degree M with M + 1 control points
Tj, j = 0, ..., M), the required Jacobians are computed to
ensure proper gradient flow. As shown in [6], the gradient of
the loss with respect to θ is given by:
∂L
∂θ =
K−1
X
k=0
∂L
∂Bk
· 1
n
n−1
X
i=0
∂Bk
∂Ci
∂Ci
∂θ
(10)
while the gradient with respect to the camera pose is:

<!-- page 6 -->
6
∂L
∂T =
K−1
X
k=0
∂L
∂Bk
· 1
n
n−1
X
i=0
∂Bk
∂Ci
∂Ci
∂θ
∂θ
∂T
(11)
For clarity, the explicit dependence on u in Bk(u) and
Ci(u) is omitted. The camera poses are parameterized using
their corresponding Lie algebra representations in SE(3), each
expressed as a 6D vector.
B. GeMS-E (Event-Assisted Deblurring)
When event data is available, we extend our pipeline with
GeMS-E to tackle scenarios where all input views are severely
motion-blurred. In this setting, we first use the event camera
data to generate sharp deblurred images, which are then passed
through our GeMS pipeline. Specifically, these deblurred
images are used exclusively for initializing the SfM stage
(VGGSfM), enabling accurate camera pose estimation and
sparse geometry that would otherwise be unattainable from
the blurred inputs alone. Importantly, during the subsequent
reconstruction process, we do not use the deblurred images for
supervision; instead, the optimization is guided by the original
blurry images using a physics-based blur formation model. The
photometric loss is computed with respect to the observed
blurred inputs, ensuring that the final scene reconstruction
remains consistent with the actual captured data. This selective
integration allows GeMS-E to benefit from robust event-based
initialization while maintaining physically faithful blur-aware
optimization, resulting in superior performance under extreme
motion blur conditions.
1) Event Generation: Unlike frame-based cameras that
record pixel brightness at a fixed frame rate, event cam-
eras asynchronously generate an event e(x, y, τ, p) when the
change in brightness of pixel (x, y) in the logarithmic domain
exceeds a threshold Θ at time τ.
px,y,τ =
(
−1,
log(Ix,y,τ) −log(Ix,y,τ−∆τ) < −Θ
+1,
log(Ix,y,τ) −log(Ix,y,τ−∆τ) > Θ
(12)
where p denotes the direction of the brightness change, and
I(x,y,τ) represents the brightness value of pixel (x, y) at
time τ. Since events are generated asynchronously, they are
typically grouped into b event bins, divided equally over
time, to facilitate processing. Given a blurred image with
exposure time from tstart to tend and the associated event data
{ei}tstart<τi≤tend, we can generate {Bk}b
k=1 as follows:
Bk = {ei(xi, yi, τi, pi)}tk−1<τi≤tk
(13)
where tk = tstart + k
b texp is the time division point between
bins and texp = tend −tstart represents the exposure time.
V. EXPERIMENTS
A. Experimental Setup
Datasets:
For evaluation, we use the synthetic dataset pro-
vided by ExBluRF [30], which includes eight diverse outdoor
scenes captured with challenging camera motion. Each scene
contains 20 to 40 blurry views for training and 4 to 6 test
views for evaluating novel view synthesis. Each blurry image
in ExBluRF is paired with sequences of sharp images, which
we utilize to create our complementary synthetic event dataset,
EveGeMS, as shown in Figure 3. Specifically, the sharp frames
recorded during the camera motion are processed using ESIM
[36] to generate the corresponding event stream, similar to
the synthetic datasets in E2NeRF. Additionally, we use a real-
world dataset from E2NeRF [33], captured with the DAVIS346
color event camera. This dataset consists of five challenging
scenes (i.e. letter, lego, camera, plant, and toys) with complex
textures and varied motion, where RGB frames were captured
with a 100ms exposure time, resulting in motion blur and
complex camera trajectories. By incorporating event streams
into the ExBluRF dataset, we contribute an extreme blurry
and event pair dataset. We will release our complementary
synthetic event dataset publicly upon acceptance to support
future research in event-based deblurring and view synthesis
under severe motion blur conditions. The complete event frame
sequences of our EveGeMS dataset for all scenes are provided
in the supplementary material.
Baseline Methods and Evaluation Metrics:
Our base-
lines include: state-of-the-art deep learning-based single-image
motion deblurring methods (MPRNet [37], Restormer [38]);
event-based motion deblurring methods (EDI [7], E2NeRF
[33], EBADNeRF [34]); and motion deblurring method de-
signed for extreme motion blur (ExBluRF* [30]) and 3D
Gaussian Splatting-based deblurring (BAD-Gaussians* [6]).
Note that both ExBluRF* and BAD-Gaussians* rely on pose
and point cloud initialization from sharp images, which is
an unrealistic assumption. In real-world scenarios, obtaining
pose and point cloud initializations directly from extremely
blurred images is not possible for NeRF and 3DGS based
deblurring methods that rely on COLMAP for initialization.
Therefore, we exclude direct comparisons with ExBluRF* and
BAD-Gaussians* on real dataset. However, we consider event-
based methods as EDI deblurred images from events enable
COLMAP initialization even under severe motion blur. We
evaluate our results using four standard metrics: PSNR and
SSIM for measuring image reconstruction quality, LPIPS for
perceptual similarity to the ground truth, and Absolute Pose
Error (APE) for assessing the accuracy of estimated camera
poses. Higher PSNR and SSIM values indicate better image
quality, while lower LPIPS and APE values indicate better
perceptual similarity and pose accuracy, respectively.
Implementation Details: Our method is implemented in Py-
Torch [39] within the 3DGS-MCMC [5] framework using the
gsplat [40] pipeline. We optimize both Gaussians and camera
poses in SE(3) B´ezier space using the Adam optimizer. For
B´ezier, we use 9 control points. The learning rate for Gaussians
follows the original 3DGS [27], while for camera poses, it is
set to 1×10−3. We set the number of virtual camera poses (n
in Eq. 7) to 15, ensuring a balance between performance and
efficiency. We use 13 event bins for event-based deblurring
(EDI). All experiments are conducted on an NVIDIA RTX
4090 GPU with a data factor of 2, using 7k iterations for all
experiments and comparisons.

<!-- page 7 -->
7
TABLE I: Quantitative comparisons for sharp novel view synthesis (deblurring + view synthesis) on the Synthetic Dataset. The table
is organized into three groups for fair comparison: (1) Methods using only motion-blurred images (w/o Events), (2) Methods using event data
as additional input (w/ Events), and (3) Methods requiring sharp images for SfM initialization (w/ Sharp Supervision). To ensure fairness,
metric rankings are reported separately for Groups 1 and 2. Group 3 methods, such as ExBluRF* and BAD-Gaussians*, are included only
for reference and are not ranked, as they rely on sharp images and hence are not practical. Best and second-best results within each ranked
group are highlighted in green and orange , respectively.
w/o Events
w/ Events
w/ Sharp Supervision(*)
Scene
Metric
MPRNet
Restormer
GeMS (Ours)
EDI+3DGS
E2NeRF
EBAD-NeRF
GeMS-E (Ours)
ExBluRF*
BAD-Gaussians*
Bench
PSNR↑
25.35
26.39
29.86
28.95
25.41
28.15
33.55
31.93
32.54
SSIM↑
0.678
0.720
0.841
0.865
0.708
0.822
0.924
0.877
0.901
LPIPS↓
0.425
0.356
0.118
0.201
0.438
0.172
0.063
0.111
0.046
Camellia
PSNR↑
24.84
25.14
28.56
22.46
28.07
24.33
29.47
28.02
28.83
SSIM↑
0.669
0.690
0.821
0.762
0.721
0.743
0.873
0.715
0.815
LPIPS↓
0.395
0.351
0.129
0.271
0.329
0.192
0.108
0.313
0.099
Dragon
PSNR↑
29.96
28.37
32.43
33.27
30.89
33.99
37.01
33.45
36.98
SSIM↑
0.731
0.704
0.818
0.842
0.697
0.864
0.925
0.828
0.930
LPIPS↓
0.454
0.465
0.171
0.243
0.433
0.202
0.069
0.180
0.045
Jars
PSNR↑
25.36
25.57
31.42
28.13
29.85
28.89
32.35
30.85
31.52
SSIM↑
0.680
0.687
0.879
0.831
0.775
0.838
0.898
0.840
0.867
LPIPS↓
0.406
0.371
0.127
0.238
0.334
0.198
0.108
0.156
0.078
Jars2
PSNR↑
24.33
26.43
28.14
24.74
27.71
27.39
28.79
30.89
28.94
SSIM↑
0.745
0.814
0.873
0.812
0.770
0.863
0.906
0.860
0.851
LPIPS↓
0.358
0.275
0.173
0.262
0.383
0.171
0.133
0.113
0.114
Postbox
PSNR↑
25.89
26.52
27.74
24.99
30.66
26.82
31.33
31.40
26.40
SSIM↑
0.736
0.753
0.788
0.789
0.813
0.826
0.906
0.864
0.757
LPIPS↓
0.318
0.286
0.150
0.228
0.262
0.151
0.070
0.095
0.123
Stone Lantern
PSNR↑
24.97
26.68
28.29
26.48
30.47
26.29
29.43
28.24
28.29
SSIM↑
0.785
0.831
0.849
0.825
0.836
0.802
0.894
0.765
0.843
LPIPS↓
0.342
0.280
0.195
0.270
0.324
0.264
0.152
0.236
0.143
Sunflowers
PSNR↑
28.86
29.55
29.47
31.38
31.74
30.98
33.69
34.46
34.06
SSIM↑
0.837
0.847
0.854
0.914
0.850
0.903
0.938
0.920
0.942
LPIPS↓
0.242
0.206
0.163
0.144
0.310
0.117
0.077
0.093
0.065
Average
PSNR↑
26.19
26.83
29.49
27.55
29.35
28.36
31.95
31.15
30.95
SSIM↑
0.733
0.756
0.840
0.830
0.771
0.833
0.908
0.834
0.863
LPIPS↓
0.368
0.324
0.153
0.232
0.352
0.183
0.097
0.162
0.089
B. Quantitative Results
Reconstruction Quality: Table I organizes methods into three
categories for fair comparison: (1) those using only motion-
blurred images (w/o Events), (2) those leveraging event data as
additional input (w/ Events), and (3) methods requiring sharp
images for SfM initialization (w/ Sharp Supervision), which
are included only for reference due to their impracticality in
real-world severe blur. Within this framework, our method
achieves superior reconstruction quality across all relevant
metrics, PSNR, SSIM, and LPIPS.
GeMS, which operates solely on motion-blurred images
without event data, outperforms all competing methods in its
group and even surpasses event-based approaches, demonstrat-
ing robustness to extreme blur. When event data is available,
GeMS-E further improves performance, achieving an aver-
age 2.5 dB PSNR gain over the state-of-the-art event-based
method E2NeRF and most significantly, our GeMS-E delivers
a 1 dB PSNR improvement over sharp-supervised baselines
(ExBluRF*, BAD-Gaussians*) despite their privileged access
to sharp images. This performance gap is particularly notable
given that sharp-supervised methods rely on impractical initial-
ization. The integration of event-based deblurring, VGGSfM-
based SfM initialization, and 3DGS-MCMC joint optimization
enables GeMS-E to deliver high-fidelity reconstructions with
significantly lower computational cost. Overall, our approach
provides a practical and effective solution for reliable 3D
reconstruction in severely motion-blurred scenarios.
GeMS-E Outperforms Sharp-Supervised Baselines BAD-
Gaussians* & ExBluRF*:
GeMS-E achieves superior per-
formance primarily due to its highly accurate initialization and
advanced motion modeling. Event-based deblurring (EDI) pro-
duces sharp-enough images for VGGSfM to estimate camera
poses that are very close to ground truth (mean APE: 0.0862),
which is crucial for reliable 3D reconstruction under extreme
motion blur. Compared to BAD-Gaussians, GeMS-E provides
robust probabilistic refinement through 3DGS-MCMC, which
adaptively densifies geometry using noise-aware sampling
and is more resilient to pose errors than the heuristic-based
approach in BAD-Gaussians. Additionally, GeMS-E’s joint
optimization of B´ezier trajectories and Gaussian parameters
ensures physical consistency, whereas BAD-Gaussians’ linear
or spline motion approximation can introduce geometric inac-
curacies, as presented in Figure 8. In contrast to ExBluRF,
GeMS-E excels at modeling complex, continuous camera

<!-- page 8 -->
8
Camellia
Dragon
Jars
Postbox
Stone lantern
Input
Restormer
EDI
E2NeRF 
EBAD-NeRF
ExBluRF* 
GeMS-E
Ground Truth
Bench
Sunflowers
Jars2
GeMS
BAD-Gaussians*
Fig. 4: Results on the Synthetic Dataset: Our method effectively removes severe motion blur, reconstructing sharp results with high fidelity.
Compared to existing approaches, it better preserves fine details and structural consistency while reducing color artifacts, demonstrating
robustness under extreme blur conditions. Note that ExBluRF* and BAD-Gaussians* rely on pose and point cloud initializations from sharp
images.
motion: while ExBluRF’s voxel-based representation strug-
gle with extreme motion blur, GeMS-E leverages a bundle-
adjusted radiance field representation enabling higher-quality
reconstructions. This combination of accurate initialization
using VGGSfM from EDI motion-blurred images and 3DGS-
MCMC joint optimization with sophisticated motion modeling
enables GeMS-E to consistently outperform both ExBluRF
and BAD-Gaussians, even when those methods have access
to sharp image supervision.
Training Time and GPU Memory Consumption:
We

<!-- page 9 -->
9
10
Input
Restormer
MPRNet
EDI
E2NeRF
EBAD-NeRF
GeMS-E
`
GeMS
Lego
Letter
Plant
Camera
Toys
Fig. 5: Results on the Real Dataset: Our method reconstructs sharp and high-quality images from severely motion-blurred real-world
inputs. In contrast, existing methods struggle with artifacts, noise, loss of fine details, and text degradation. Our framework effectively
restores textures and structural consistency, as evident in the insets.
evaluate the efficiency of our approach against state-of-the-
art event-based methods, EBAD-NeRF and E2NeRF, in terms
of training time and GPU memory usage on real dataset. As
shown in Table V, our method significantly reduces training
time, completing optimization in approximately 7 minutes
per scene, whereas EBAD-NeRF and E2NeRF require over
6 hours and 14 hours, respectively. Moreover, Table VI high-
lights the GPU memory consumption across different scenes,
where our approach requires only ∼1.55 GiB on an average,
compared to the excessive demands of EBAD-NeRF (∼14.49
GiB) and E2NeRF (∼15.16 GiB). These results confirm the
scalability and hardware efficiency of our method, making it
practical for real-world applications.
C. Qualitative Results
We present qualitative comparisons for deblurring on both
synthetic and real datasets in Figure 4 and Figure 5, respec-
tively. Our method GeMS-E consistently outperforms existing
approaches, producing sharper reconstructions with fewer ar-
tifacts and improved texture details. In synthetic scenes such
as Stone Lantern and Jars, competing methods struggle to
recover fine structures, often leading to over-smoothed or
distorted reconstructions, as indicated by the red arrows in
Figure 4. In contrast, our approach faithfully restores object
details and edges. Similarly, in real-world datasets as shown
in Figure 5, our method achieves superior clarity, particularly
in challenging regions with high-frequency textures, such as
text on the CVPR 2023 poster or specular highlights on

<!-- page 10 -->
10
Input
w/o Events + w/o MCMC
GeMS(w/o Events)
w/o MCMC
GeMS-E
w/o VGGSfM
Ground Truth
Fig. 6: Ablations on different modules of our framework on the Synthetic Dataset: Our method (GeMS) achieves strong deblurring
with MCMC helping to reduce artifacts and enhance reconstruction quality. With events (GeMS-E), the results become even sharper and
more detailed, demonstrating the added benefit of event information.
Input
w/o Events + w/o MCMC
GeMS(w/o Events)
w/o MCMC
GeMS-E
Fig. 7: Ablation study on different modules of our framework on the Real Dataset: GeMS produces high-quality reconstructions, with
MCMC effectively suppressing artifacts. Incorporating event data in GeMS-E further refines the results, demonstrating the effectiveness of
our approach across different blur settings.
metallic surfaces. Unlike prior event-based methods, such as
E2NeRF and EBAD-NeRF, which introduce color distortions
or fail to fully remove motion blur, our approach effectively
preserves accurate color distributions while restoring sharp de-
tails. Moreover, our method demonstrates improved robustness
in handling fine-grained textures, such as plant leaves and
intricate object boundaries, where others exhibit blurring or
ghosting artifacts. This advantage is evident in diverse scenes,
reinforcing the effectiveness of our framework.
D. Ablations
To comprehensively evaluate our framework, we first assess
the robustness of each module across progressive motion blur
levels, quantifying their effectiveness under varying degrees of
blur. We then conduct targeted analyses in extreme blur sce-
narios, specifically validating the performance and individual
contributions of each module.
1) Robustness of Modules across various Blur Levels: VG-
GSfM Robustness:
To systematically evaluate VGGSfM’s
robustness to motion blur, we designed a controlled experi-
ment using multi-frame averaging to simulate increasing blur
severity. Starting with a burst sequence of 11 sharp images
per viewpoint, we created progressively blurred inputs by
averaging 1, 3, 5, 7, 9, and 11 consecutive frames (Figure 9).
Each blurred image was processed through both VGGSfM
and COLMAP. While COLMAP failed to register images
beyond moderate blur levels, VGGSfM consistently registered
all images across all blur levels, producing stable camera poses
and dense point clouds even under extreme blur (Table III).
To further evaluate VGGSfM’s effectiveness, we plotted the
translational (x, y, z) and rotational (yaw, pitch, roll) poses

<!-- page 11 -->
11
Fig. 8: Impact of trajectory representations and the number of virtual cameras: Comparison of Linear, Spline, and Bezier trajectory
representations across different virtual camera counts. Bezier interpolation consistently performs better than Linear and Spline representations,
demonstrating its advantage in generating high-quality sharp novel views.
TABLE II: Ablation study for novel sharp view synthesis (deblurring + novel view synthesis) on the synthetic dataset. We evaluate
the impact of different components, VGGSfM, MCMC and EDI on novel view synthesis performance. The inclusion of each module leads to
considerable improvements in PSNR, SSIM, and LPIPS, with their combination yielding the best overall results, demonstrating the synergistic
benefits of our approach.
Method
Metric
Bench
Camellia
Dragon
Jars
Jars2
Postbox
Stone L.
Sunflowers
Average
w/o MCMC + w/o EDI + w/ VGGSfM
PSNR↑
30.06
27.80
33.19
30.89
28.51
25.86
27.22
28.64
29.02
SSIM↑
0.832
0.772
0.831
0.842
0.838
0.688
0.821
0.827
0.806
LPIPS↓
0.097
0.118
0.081
0.097
0.102
0.177
0.169
0.226
0.133
w/ MCMC + w/o EDI + w/ HLOC
PSNR↑
30.32
27.07
31.01
28.62
27.85
22.94
24.73
31.76
28.04
SSIM↑
0.838
0.717
0.773
0.752
0.816
0.586
0.706
0.898
0.761
LPIPS↓
0.101
0.154
0.118
0.146
0.104
0.218
0.251
0.090
0.148
GeMS (w/ MCMC + w/o EDI + w/ VGGSfM)
PSNR↑
29.86
28.56
32.43
31.42
28.14
27.74
28.29
29.47
29.49
SSIM↑
0.841
0.821
0.818
0.879
0.873
0.788
0.849
0.854
0.840
LPIPS↓
0.118
0.129
0.171
0.127
0.173
0.150
0.195
0.163
0.153
w/o MCMC + w/ EDI + w/ VGGSfM
PSNR↑
32.63
28.70
36.41
31.66
28.60
28.52
27.97
33.97
31.06
SSIM↑
0.901
0.811
0.933
0.865
0.839
0.803
0.855
0.942
0.869
LPIPS↓
0.042
0.101
0.039
0.084
0.104
0.085
0.145
0.069
0.083
w/ MCMC + w/ EDI + w/ COLMAP
PSNR↑
32.33
30.05
35.03
31.44
28.93
30.67
28.29
33.51
31.28
SSIM↑
0.914
0.872
0.866
0.876
0.886
0.887
0.852
0.933
0.886
LPIPS↓
0.081
0.111
0.203
0.147
0.161
0.098
0.224
0.086
0.139
GeMS-E (w/ MCMC + w/ EDI + w/ VGGSfM)
PSNR↑
33.55
29.47
37.01
32.35
28.79
31.33
29.43
33.69
31.95
SSIM↑
0.924
0.873
0.925
0.898
0.906
0.906
0.894
0.938
0.908
LPIPS↓
0.063
0.108
0.069
0.108
0.133
0.070
0.152
0.077
0.097
of the blurry reconstructions with respect to the sharp ones,
as well as the translational pose errors relative to ground
truth for each blur level (Figure 10 & Table IV). This clearly
demonstrates that VGGSfM remains robust and reliable even
in extreme motion blur scenarios where traditional methods
fail.
3DGS-MCMC Robustness:
To systematically evaluate the
robustness of 3DGS-MCMC to blur-corrupted initializations,
we generated point clouds from images with varying levels of
motion blur using VGGSfM, resulting in increasingly noisy
and sparse initial geometry (e.g., blur-3, blur-5).
We conducted two systematic experiments to thoroughly
evaluate the robustness of 3DGS-MCMC under varying levels
of motion blur. In the first experiment, we focused on the
effect of blur-corrupted initializations by generating point
clouds from images with different degrees of motion blur
using VGGSfM, resulting in increasingly noisy and sparse
geometry. These blur-degraded point clouds were then used
as the starting input for both standard 3DGS and 3DGS-
MCMC, with the reconstruction loss computed against sharp
ground truth images. The results, as shown in Figure 11 clearly
demonstrates that standard 3DGS suffers significant perfor-
mance degradation as blur increases, whereas 3DGS-MCMC
maintains stable reconstruction quality across all blur levels.
This quantitatively validates that the probabilistic sampling
framework of MCMC is inherently robust to initialization
quality, and specifically effective for blur-corrupted point
cloud initializations.
In the second experiment, we assessed the deblurring perfor-
mance within our GeMS framework by comparing GeMS with
MCMC-based probabilistic refinement against a variant of
GeMS without MCMC. Both versions were initialized directly
on motion-blurred images and optimized end-to-end using
photometric loss against the original blurry images, across a
range of blur levels. As reported in Figure 12 GeMS with
MCMC consistently maintains better reconstruction quality
across increasing blur severity. Furthermore, our ablation re-
sults (Figure 6, Figure 7) reveal that the non-MCMC variant

<!-- page 12 -->
12
3
5
9
11
7
Sharp
Fig. 9: Synthetic Motion Blur Levels: Images generated by averaging 1 (sharp), 3, 5, 7, 9, and 11 consecutive frames from an 11-image
burst at each viewpoint, illustrating the increasing severity of motion blur used for evaluation.
TABLE
III:
SfM
comparison:
Performance
of
VGGSfM
vs
COLMAP across various blur levels.
Blur
VGGSfM
COLMAP
#Pts
#Imgs
#Pts
#Imgs
Sharp
23498
20
4987
20
3
22873
20
1257
20
5
21660
20
606
20
7
18518
20
246
15
9
21351
20
x
x
11
20283
20
x
x
TABLE IV: VGGSfM Pose translation error statistics (in meters)
across blur levels with respect to ground truth.
Blur
RMSE ↓
Mean ↓
Med. ↓
Std ↓
Max ↓
3
0.18
0.16
0.13
0.09
0.35
5
0.32
0.27
0.20
0.18
0.65
7
0.40
0.36
0.29
0.17
0.75
9
0.61
0.49
0.36
0.36
1.43
11
0.72
0.52
0.35
0.50
2.24
(a) Translation (X, Y,
Z)
(b)
Rotation
(Yaw,
Pitch, Roll)
(c) Pose Error metrics
Fig. 10: Robustness of VGGSfM Pose Estimation across Blur
Levels: Evaluation of VGGSfM’s pose estimation performance using
sharp versus motion-blurred inputs. (a) Estimated translation param-
eters (X,Y,Z) for 20 views compared to ground truth. (b) Estimated
rotation parameters (yaw, pitch, roll) for corresponding views versus
ground truth. (c) Absolute pose error statistics for each blur level
relative to ground truth.
exhibits noticeable artifacts and structural inconsistencies in
extreme motion blur scenarios, highlighting the practical value
of MCMC for reliable scene recovery under challenging
extreme motion blur conditions.
2) Component-wise Analysis for Extreme Motion Blur: To
quantify the contribution of each module (VGGSfM, MCMC,
EDI) in GeMS and GeMS-E for extreme motion deblurring,
we performed ablation studies on the synthetic dataset, with
results summarized in Table II.
Our analysis shows that COLMAP fails entirely under
severe motion blur, while HLOC, although able to run, lags
behind VGGSfM by an average of 1.45 dB in PSNR, highlight-
ing VGGSfM’s superior robustness in challenging conditions.
The inclusion of MCMC-based probabilistic refinement further
improves reconstruction quality, contributing an additional
0.47 dB in PSNR and noticeably reducing artifacts, as seen
in Figure 6. Event-based deblurring (EDI) provides the most
significant boost, with its integration yielding a 2.46 dB PSNR
improvement over GeMS and producing the sharpest and most
faithful reconstructions, particularly in extreme blur scenarios.
The full GeMS-E pipeline, combining VGGSfM, MCMC, and
Fig. 11: MCMC robustness to various blur-corrupted point
cloud initializations: PSNR comparison of 3DGS-MCMC with
3DGS across various blur point cloud initializations obtained from
VGGSfM.
Fig. 12: MCMC robustness to various blur levels in GeMS:
PSNR comparison of GeMS for deblurring with and without MCMC
across various blur levels, SfM initialization are obtained from
VGGSfM.
EDI, achieves the highest overall performance, as evidenced by
both quantitative results and qualitative examples in Figure 7.

<!-- page 13 -->
13
TABLE V: Training Time Comparison on the Real Dataset
(hh:mm:ss): Comparison of training times for different methods
on real datasets, highlighting the efficiency of our approach.
Camera↓
Lego↓
Letter↓
Plant↓
Toys↓
Avg↓
EBAD-NeRF
06:44:00
06:44:00
06:44:00
06:44:00
06:44:00
06:44:00
E2NeRF
14:58:00
14:58:00
14:58:00
14:58:00
14:58:00
14:58:00
Ours
00:07:14
00:07:23
00:07:18
00:06:58
00:07:11
00:07:13
TABLE VI: GPU Memory Usage on the Real Dataset (MiB):
Evaluation of GPU memory consumption across different methods,
demonstrating the significant reduction in GPU memory usage.
Camera↓
Lego↓
Letter↓
Plant↓
Toys↓
Avg↓
EBAD-NeRF
14836
14836
14836
14836
14836
14836
E2NeRF
15532
15532
15532
15532
15532
15532
Ours
1571
1897
1730
1341
1382
1584
3) Number of Virtual Cameras & Trajectory Representa-
tions: We conducted experiments to analyze the impact of
the number of interpolated virtual camera poses within the
duration of exposure, denoted as n, and the effectiveness
of different motion trajectory representations. We varied n
across {5, 10, 15, 20}, with the corresponding rendering results
summarized in Figure 8. Our findings indicate that increasing
n up to a certain threshold effectively mitigates severe motion
blur, beyond which performance begins to decline; hence,
we adopt n = 15. Additionally, we conducted ablations
using linear interpolation, cubic B-splines, and B´ezier curves
for trajectory representations, with results also summarized
in Figure 8. Our findings demonstrate that B´ezier curves
outperform other methods, as they better capture the complex,
non-uniform motion present in heavily blurred scenes. B´ezier
curves consistently achieve superior performance across all
metrics in severe motion blur scenarios, providing smoother
and more accurate trajectory representations, making them the
optimal choice for our framework.
VI. CONCLUSION
In this work, we introduced GeMS, an efficient 3D Gaussian
Splatting framework that reconstructs sharp 3D scenes directly
from severely motion-blurred images. By integrating VGGSfM
for blur-robust initialization, 3DGS-MCMC for probabilistic
scene modeling, and joint trajectory-geometry optimization,
our approach forms a tightly coupled, end-to-end differentiable
pipeline where each component compensates for the limita-
tions of the others. This synergy enables mutual refinement:
VGGSfM’s initialization is adaptively densified by 3DGS-
MCMC, while joint optimization continuously aligns scene
geometry and camera motion using a physics-based blur image
formation model. When event data is available, we extend this
framework with GeMS-E by using the Event-based Double
Integral (EDI) model to deblur the inputs; these deblurred
images are then passed through our GeMS framework for
further refinement, especially in scenarios where all input
images are severely motion blurred. Extensive experiments on
both synthetic and real-world datasets demonstrate that GeMS
and GeMS-E consistently outperform state-of-the-art methods
in accuracy, efficiency, and robustness, even under extreme
motion blur. Our work establishes a new paradigm for motion-
robust 3D reconstruction, moving beyond the limitations of
COLMAP and enabling reliable scene recovery in extreme
motion blur scenarios previously considered intractable. Future
work will explore deblurring with an extremely sparse set of
images, ultimately pushing the framework’s limits to handle
the most challenging scenario: reconstructing sharp scene from
a single motion-blurred image under extreme motion blur.
Additionally, we aim to extend GeMS / GeMS-E to dynamic
scenes.
REFERENCES
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “NeRF: Representing Scenes as Neural Radiance Fields for
View Synthesis,” Aug. 2020.
[2] B.
Kerbl,
G.
Kopanas,
T.
Leimk¨uhler,
and
G.
Drettakis,
“3D
Gaussian Splatting for Real-Time Radiance Field Rendering,” ACM
TOG,
vol.
42,
no.
4,
July
2023.
[Online].
Available:
https:
//repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
[3] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion Revisited,”
in CVPR, 2016. [Online]. Available: https://github.com/colmap/colmap
[4] J. Wang, N. Karaev, C. Rupprecht, and D. Novotny, “Vggsfm: Visual
geometry grounded deep structure from motion,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2024,
pp. 21 686–21 697.
[5] S. Kheradmand, D. Rebain, G. Sharma, W. Sun, Y.-C. Tseng, H. Isack,
A. Kar, A. Tagliasacchi, and K. M. Yi, “3d gaussian splatting as markov
chain monte carlo,” Advances in Neural Information Processing Systems,
vol. 37, pp. 80 965–80 986, 2025.
[6] L. Zhao, P. Wang, and P. Liu, “BAD-Gaussians: Bundle Adjusted Deblur
Gaussian Splatting,” in ECCV.
Springer, 2024.
[7] L. Pan, C. Scheerlinck, X. Yu, R. Hartley, M. Liu, and Y. Dai, “Bringing
a Blurry Frame Alive at High Frame-Rate With an Event Camera,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2019, pp. 6820–6829.
[8] L. Pan, D. Bar´ath, M. Pollefeys, and J. L. Sch¨onberger, “Global
structure-from-motion revisited,” in European Conference on Computer
Vision.
Springer, 2024, pp. 58–77.
[9] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, “From coarse
to fine: Robust hierarchical localization at large scale,” in CVPR, 2019.
[10] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “SuperGlue:
Learning feature matching with graph neural networks,” in CVPR, 2020.
[11] P. Lindenberger, P.-E. Sarlin, V. Larsson, and M. Pollefeys, “Pixel-
perfect structure-from-motion with featuremetric refinement,” in Pro-
ceedings of the IEEE/CVF international conference on computer vision,
2021, pp. 5987–5997.
[12] J. Reizenstein, R. Shapovalov, P. Henzler, L. Sbordone, P. Labatut,
and D. Novotny, “Common objects in 3d: Large-scale learning and
evaluation of real-life 3d category reconstruction,” in Proceedings of
the IEEE/CVF international conference on computer vision, 2021, pp.
10 901–10 911.
[13] Y. Jin, D. Mishkin, A. Mishchuk, J. Matas, P. Fua, K. M. Yi, and
E. Trulls, “Image matching across wide baselines: From paper to
practice,” International Journal of Computer Vision, vol. 129, no. 2,
pp. 517–547, 2021.
[14] T. Schops, J. L. Schonberger, S. Galliani, T. Sattler, K. Schindler,
M. Pollefeys, and A. Geiger, “A multi-view stereo benchmark with high-
resolution images and multi-camera videos,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2017, pp. 3260–
3269.
[15] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields,” in CVPR, 2021.
[16] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in CVPR,
2022.
[17] Y. Jiang, P. Hedman, B. Mildenhall, D. Xu, J. T. Barron, Z. Wang, and
T. Xue, “Alignerf: High-fidelity neural radiance fields via alignment-
aware training,” CVPR, 2023.
[18] L. Wu, J. Y. Lee, A. Bhattad, Y. Wang, and D. Forsyth, “Diver: Real-
time and accurate neural radiance fields with deterministic integration
for volume rendering,” 2022.

<!-- page 14 -->
14
[19] K. Zhang, G. Riegler, N. Snavely, and V. Koltun, “NeRF++: Analyzing
and improving neural radiance fields,” arXiv:2010.07492, 2020.
[20] D. B. Lindell, J. N. Martel, and G. Wetzstein, “Autoint: Automatic
integration for fast neural volume rendering,” in CVPR, 2021.
[21] C. Reiser, S. Peng, Y. Liao, and A. Geiger, “Kilonerf: Speeding up
neural radiance fields with thousands of tiny mlps,” in ICCV, 2021.
[22] A. Yu, R. Li, M. Tancik, H. Li, R. Ng, and A. Kanazawa, “Plenoctrees
for real-time rendering of neural radiance fields,” in ICCV, 2021.
[23] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and
A. Kanazawa, “Plenoxels: Radiance fields without neural networks,” in
CVPR, 2022.
[24] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” TOG, 2022.
[25] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance
fields,” in ECCV, 2022.
[26] C. Sun, M. Sun, and H.-T. Chen, “Direct voxel grid optimization: Super-
fast convergence for radiance fields reconstruction,” CVPR, 2022.
[27] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” TOG, 2023.
[28] N. Max, “Optical models for direct volume rendering,” IEEE Transac-
tions on Visualization and Computer Graphics, pp. 99–108, 1995.
[29] L. Ma, X. Li, J. Liao, Q. Zhang, X. Wang, J. Wang, and P. V. Sander,
“Deblur-NeRF: Neural Radiance Fields from Blurry Images,” in CVPR,
2022. [Online]. Available: https://limacv.github.io/deblurnerf/
[30] D. Lee, J. Oh, J. Rim, S. Cho, and K. M. Lee, “Exblurf: Efficient
radiance fields for extreme motion blurred images,” in ICCV, 2023.
[31] P. Wang, L. Zhao, R. Ma, and P. Liu, “BAD-NeRF: Bundle Adjusted
Deblur Neural Radiance Fields,” in CVPR, 2023. [Online]. Available:
https://wangpeng000.github.io/BAD-NeRF/
[32] W. Chen and L. Liu, “Deblur-gs: 3d gaussian splatting from camera
motion blurred images,” Proceedings of the ACM on Computer Graphics
and Interactive Techniques, vol. 7, no. 1, pp. 1–15, 2024.
[33] Y. Qi, L. Zhu, Y. Zhang, and J. Li, “E2NeRF: Event Enhanced Neural
Radiance Fields from Blurry Images,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 13 254–13 264.
[34] Y. Qi, L. Zhu, Y. Zhao, N. Bao, and J. Li, “Deblurring neural radiance
fields with event-driven bundle adjustment,” in Proceedings of the 32nd
ACM International Conference on Multimedia, 2024, pp. 9262–9270.
[35] H. Deguchi, M. Masuda, T. Nakabayashi, and H. Saito, “E2gs: Event
enhanced gaussian splatting,” in 2024 IEEE International Conference
on Image Processing (ICIP).
IEEE, 2024, pp. 1676–1682.
[36] H. Rebecq, D. Gehrig, and D. Scaramuzza, “ESIM: An Open Event
Camera Simulator,” in Proceedings of The 2nd Conference on Robot
Learning.
PMLR, Oct. 2018, pp. 969–982.
[37] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang,
and L. Shao, “Multi-stage Progressive Image Restoration,” in CVPR,
2021. [Online]. Available: https://github.com/swz30/MPRNet
[38] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang,
“Restormer: Efficient transformer for high-resolution image restoration,”
in CVPR, 2022.
[39] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,
T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An
Imperative Style, High-performance Deep Learning Library,” Advances
in neural information processing systems, vol. 32, 2019. [Online].
Available: https://pytorch.org/
[40] V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye,
J. Hu, M. Tancik et al., “gsplat: An open-source library for gaussian
splatting,” Journal of Machine Learning Research, vol. 26, no. 34, pp.
1–17, 2025.
