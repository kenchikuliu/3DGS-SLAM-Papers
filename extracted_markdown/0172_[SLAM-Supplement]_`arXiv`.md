<!-- page 1 -->
arXiv:2505.15287v1  [cs.CV]  21 May 2025
GS2E: Gaussian Splatting is an Effective Data
Generator for Event Stream Generation
Yuchen Li1*
Chaoran Feng1*
Zhenyu Tang1
Kaiyuan Deng2
Wangbo Yu1
Yonghong Tian1 †
Li Yuan1 †
1School of Electronic and Computer Engineering, Peking University
2Holcombe Department of Electrical and Computer Engineering, Clemson University
{yuchenli, chaoran.feng, zhenyutang, wbyu}@stu.pku.edu.cn, {yhtian, yuanli-ece}@pku.edu.cn
https://intothemild.github.io/GS2E.github.io/
Figure 1: We propose GS2E, a high-fidelity synthetic dataset designed for 3D event-based vision,
comprising over 1150 scenes. GS2E examples of RGB frames and event streams are shown above.
Abstract
We introduce (GS2E) (GAUSSIAN SPLATTING TO EVENT GENERATION), a
large-scale synthetic event dataset for high-fidelity event vision tasks, captured
from real-world sparse multi-view RGB images. Existing event datasets are often
synthesized from dense RGB videos, which typically lack viewpoint diversity
and geometric consistency, or depend on expensive, difficult-to-scale hardware
setups. GS2E overcomes these limitations by first reconstructing photorealistic
static scenes using 3D Gaussian Splatting, and subsequently employing a novel,
physically-informed event simulation pipeline. This pipeline generally integrates
adaptive trajectory interpolation with physically-consistent event contrast threshold
modeling. Such an approach yields temporally dense and geometrically consistent
event streams under diverse motion and lighting conditions, while ensuring strong
alignment with underlying scene structures. Experimental results on event-based
3D reconstruction demonstrate GS2E’s superior generalization capabilities and its
practical value as a benchmark for advancing event vision research.
*These authors contributed equally to this work.
†Corresponding author.
Preprint.

<!-- page 2 -->
1
Introduction
Event cameras, provide high temporal resolution, low latency, and high dynamic range, making them
uniquely suited for tasks involving fast motion and challenging lighting conditions [11, 78]. These
advantages have been demonstrated in various applications such as autonomous driving [16, 18],
drone navigation [58, 5], and 3D scene reconstruction [26, 77, 25, 59, 54, 76]. In particular, their
ability to capture asynchronous brightness changes enables accurate motion perception for 3D
reconstruction and novel view synthesis (NVS) tasks, surpassing the capabilities of conventional
RGB sensors under fast motion and dynamic illumination [79]. However, despite their potential, the
advancement of event-based vision algorithms is significantly limited by the scarcity of large-scale,
high-quality event datasets, especially those offering multi-view consistency and aligned RGB data.
This bottleneck has slowed the development of hybrid approaches that aim to combine event and RGB
signals for high-fidelity 3D scene understanding and reconstruction. While event streams provide
accurate geometric and motion cues through the high-frequency edge information, RGB frames,
often motion-blurred, contribute essential color features with low-frequent details but can suffer from
degraded textural information for rendering. While event streams offer precise geometric and motion
cues through high-frequency edge information, RGB frames, though often degraded by motion blur,
provide complementary low-frequency texture and essential color details for photorealistic rendering.
However, the lack of large-scale datasets that jointly exploit these complementary signals limits
progress in event-based 3D scene understanding and generation [80].
As illustrated in Figure 2, existing efforts to conduct event-based 3D reconstruction datasets fall into
three main categories: (1) Real-world capture: This involves dedicated hardware setups such as
synchronized event-RGB stereo rigs or multi-sensor arrays (e.g., DAVIS-based systems [69]). While
providing realistic data, these setups are expensive, prone to calibration errors, and difficult to scale
to diverse scenes and camera configurations, as seen in systems like Dynamic EventNeRF [60]. (2)
Video-driven synthesis: v2e [24] and Vid2E [13] generate event streams from dense, high-framerate
RGB videos. Although flexible and accessible, they suffer from limited viewpoint diversity and lack
geometric consistency, making them suboptimal for multi-view reconstruction tasks. (3) Simulation
via computer graphics engines: Recent approaches [56, 20, 28, 43] leverage 3D modeling tools like
Blender [6] or Unreal Engine [8] to simulate photo-realistic scenes and generate event data along
with RGB, depth, and pose annotations. These allow fine-grained control over camera trajectories
and lighting, enabling multi-view, physically-consistent dataset synthesis. However, such pipelines
may introduce a domain gap due to non-photorealistic rendering or oversimplified dynamics [64].
Video-driven and graphics-based approaches offer greater scalability by enabling controllable and
repeatable data generation among above methods. However, video-based methods rely on densely
sampled RGB frames from narrow-baseline views, often resulting in motion blur and limited geomet-
ric diversity. Simulation with physical engines offers greater control over scenes and trajectories, yet
frequently suffer from domain gaps due to non-photorealistic rendering and simplified dynamics [52].
Moreover, an important yet often underexplored factor influencing the realism of synthetic event
data is the contrast threshold (CT), which defines the minimum log-intensity change required to
trigger an event. While many existing simulators [39, 13] adopt fixed or heuristic CT values, recent
analyses [64, 27, 19] show that CT values vary considerably across sensors, scenes, and even within
the same sequence. This variability induces a significant distribution shift between synthetic and
real event data, thereby limiting the generalization capability of models trained on simulated streams.
We posit that accurate modeling of contrast thresholds as data-dependent and adaptive parameters
is crucial for generating realistic and transferable event representations. Incorporating physically
informed CT sampling, potentially complemented by plausible noise considerations, can significantly
enhance sim-to-real transferability in downstream tasks such as 3D reconstruction [87, 10, 83, 10, 97]
and optical flow estimation [101, 49, 9, 34, 12].
Based on the above observation, we propose a novel pipeline for synthesizing high-quality, geometry-
consistent multi-view event data from sparse RGB inputs. Leveraging 3D Gaussian Splatting
(3DGS) [32], we first reconstruct photorealistic 3D scenes from a sparse set of multi-view images
with known poses. We then generate continuous trajectories via adaptive interpolation and render
dense RGB sequences along these paths. These sequences are fed into our physically-informed event
simulator [39]. This simulator employs our data-driven contrast threshold modeling to ensure event
responses are consistent with real sensor behaviors, and inherently maintains geometric consistency
through the 3DGS-rendered views and trajectories. Our approach requires no dense input video and
2

<!-- page 3 -->
Dynamic EventNeRF, E2NeRF
(2) Video-driven Synthesis
(3) Simulation via computer graphics engines
Method
Pros
Cons
(1) Real-world Capture
Dynamic EventNeRF
[CVPR 2025 Workshop]
No domain gap, Reflects real-
world complexity 
Expensive setups, 
Calibration errors
3D Asset
ESIM
Engines
High-quality Event, Controllable camera trajectories 
Ground-truth Priors (Depth, Optic flow, Normal, Poses)
Real-Synthetic Domain Gap 
ICNS, Robust e-NeRF, Deblur e-NeRF
V2E, V2CE, Vid2e
Flexible, Easy to access, No hardware 
constraints
Limited viewpoint diversity, Geometric 
inconsistency, Motion-blur
V2E
[CVPR2021 Workshop]
Robust e-NeRF
[ICCV 2023]
Event Model
Super
Slomo
Figure 2: Overview and comparison of event-based 3D dataset construction methods. We
compare (1) real-world capture, (2) video-driven synthesis, and (3) simulation via computer graphics
engines in terms of commonly used methods, strengths, and drawbacks.
preserves the geometric fidelity of the original scene. The controllable virtual setup enables diverse
motion patterns and blur levels, supporting the training of robust event-based models.
To summarize, our main contributions are:
• We propose a novel simulation pipeline for generating multi-view event data from sparse
RGB images, leveraging 3DGS for high-fidelity reconstruction and novel view synthesis.
• We propose an adaptive trajectory interpolation strategy coupled with a physically-grounded
contrast threshold model, jointly enabling the synthesis of temporally coherent and sensor-
consistent event streams.
• We construct and release a benchmark dataset comprising photorealistic RGB frames,
motion-blurred sequences, accurate camera poses, and multi-view event streams, facilitating
research in structure-aware event vision.
2
Related Work
2.1
Optimization-based Event Simulators
Early event simulation methods, such as those proposed by Kaiser et al.[30] and Mueggler et al.[47],
generated events by thresholding frame differences or rendering high-framerate videos. However,
these approaches failed to capture the inherent asynchronous and low-latency characteristics of
event sensors. Subsequent works like ESIM [56] and Vid2E [14] improved realism by incorporating
per-pixel log-intensity integration and optical flow-based interpolation to approximate event trig-
gering more faithfully. V2E [24] further advanced realism by modeling sensor-level attributes such
as bandwidth limitations, background activity noise, and Poisson-distributed firing. More recent
simulators including V2CE [95], ICNS [29], and DVS-Voltmeter [39], introduced hardware-aware
components, accounting for effects such as latency, temperature-dependent noise, and local motion
dynamics. PECS [21] extended this direction by modeling the full optical path through multispectral
photon rendering. Despite their increased physical fidelity, most of these simulators operate purely
on 2D image or video inputs and do not exploit the underlying 3D structure of scenes. Furthermore,
the prevalent use of fixed contrast thresholds across all viewpoints and scenes fails to reflect the
variability observed in real sensors, thereby introducing a significant domain gap in simulated data.
2.2
Learning-based Event Simulators
Recent efforts have explored deep learning to synthesize event streams in a data-driven manner.
EventGAN [100] employed GANs to generate event frames from static images, while Pantho et
al. [50] learned to generate temporally consistent event tensors or voxel representations. Domain-
adaptive simulators such as Gu et al. [17] jointly synthesized camera trajectories and event data,
improving realism under target distributions. However, learning-based approaches generally suffer
from limited interpretability and require retraining when transferred to new scenarios, leading to
3

<!-- page 4 -->
weaker robustness compared to physics-inspired models. In contrast, our method follows a physically
grounded yet geometry-aware paradigm: we first reconstruct high-fidelity 3D scenes via 3DGS then
synthesize event streams by simulating photorealistic motion blur and modeling the contrast threshold
distribution observed in real-world data. This enables us to generate temporally coherent, multi-view
consistent event data with improved realism and transferability.
3
Method
3.1
Preliminary
3D Gaussian Splatting.
3D Gaussian Splatting [32] represents a scene as a set of anisotropic
Gaussians. Each Gaussian Gi is defined by
Gi(x) = exp

−1
2 (x −µi)⊤(Ri diag(si)2 R⊤
i )−1(x −µi)

,
(1)
where µi ∈R3 is the mean, qi 7→Ri ∈SO(3) is the rotation, and si ∈R3
+ is the scale. View-
dependent radiance coefficients ci and opacity αi ∈[0, 1] are optimized via differentiable rasterization
under an ℓ1 photometric loss. Each Gaussian is transformed by the camera pose T ∈SE(3) and
projected at render time, then the resulting 2D covariance is
Σ′
i = Ji T Σi T⊤J⊤
i ,
(2)
where Ji denotes the Jacobian of the projection and each pixel colors ˆC are composited as follows:
ˆC =
X
k∈N
ck αk
Y
j<k
(1 −αj).
(3)
Event Generation Model.
Event cameras produce an asynchronous stream of tuples (x, y, ti, pi)
by thresholding changes in log-irradiance [11, 15]. Denoting the last event time at pixel (x, y) by
tref, define as:
∆log L = log L(x,y)(ti) −log L(x,y)(tref).
(4)
An event of polarity pi ∈{+1, −1} is emitted whenever |∆log L| ≥c:
pi =
+1,
∆log L ≥c,
−1,
∆log L ≤−c.
(5)
After each event, tref is updated to ti. This simple thresholding mechanism yields a high-temporal-
resolution, sparse stream of brightness changes suitable for downstream vision tasks. Here, we
introduce the off-the-shell model DVS-Voltmeter [39] as the event generation model, which incorpo-
rates physical characteristics of DVS circuits. Unlike deterministic models, DVS-Voltmeter treats the
voltage evolution at each pixel as a stochastic process, specifically a Brownian motion with drift. In
this formulation, the photovoltage change ∆Vd over time is modeled as
∆Vd(t) = µ∆t + σW(∆t),
(6)
where µ is a drift term capturing systematic brightness changes, σ denotes the noise scale influenced
by photon reception and leakage currents, and W(·) represents a standard Brownian motion. Events
are then generated when the stochastic voltage process crosses either the ON or OFF thresholds.
This physics-inspired modeling enables it to produce events with realistic timestamp randomness and
noise characteristics, providing more faithful supervision for event-based vision tasks.
3.2
Pipeline Overview
Our pipeline generates multi-view, geometry-consistent event data from sparse RGB inputs. The
process begins with collecting sparse multi-view RGB images along with their corresponding camera
poses (§3.3). Using these inputs, we reconstruct high-fidelity scene geometry and appearance
via 3DGS [32] (§3.4), providing a solid foundation for the subsequent steps. To simulate diverse
observations, we generate smooth, controllable virtual camera trajectories by reparameterizing the
original pose sequence based on velocity constraints, followed by interpolation of dense viewpoints
along the trajectory (§3.5). Finally, the generated RGB sequences are fed into our optimized event
generation module to synthesize temporally coherent, multi-view-consistent event streams (§3.6).
This well-structured pipeline enables scalable and controllable event data generation from sparse
RGB inputs, ensuring both accuracy and efficiency. The overall pipeline is shown in Figure 3.
4

<!-- page 5 -->
3D Scene Representation
(Sec. 3.4)
Multi-view Images
(Sec. 3.3)
Multi-view Event Stream
(Sec. 3.5)
Virtual Camera Trajectory Generation
Base Trajectory Refinement and Densification
……
Novel-View Trajectory Augmentation
Rasterization
Cubic B-Spline
Dense Viewpoints
……
Calculate 𝜇, 𝜎
(Browning Motion)
𝐼!"#(𝑥, 𝑦)
Sample polarity(p)
of an event triggered
Inverse Gaussian
sample timestamp(t)
𝜇= #
$ (𝐼! + 𝐼!"#),  𝜎= (𝐼! −𝐼!"#) 
𝑃(𝑂𝑁) =
𝑒"$%&!"/(# −1
𝑒"$%&!"/(# −𝑒"$%&!$$/(#
𝑡)*/)++ ~ 𝐼𝐺(
𝑐)*/)++
𝜇
,
𝑐)*/)++
$
𝜎$
)
(𝑡, 𝑥, 𝑦, 𝑝)
𝐼!(𝑥, 𝑦)
Event Stream Synthesis
(Sec. 3.6)
FAST
SLOW
……
Origin
Novel
Figure 3: Overview of the proposed GS2E pipeline. Starting from sparse multi-view RGB images
and known camera poses, we reconstruct high-fidelity scene representations using 3D Gaussian
Splatting. Virtual camera trajectories are then synthesized via velocity-aware reparameterization and
interpolation. The rendered image sequences are passed to a volumetric event simulator to generate
temporally coherent and geometrically consistent event streams.
3.3
Data Collection
To support high-fidelity reconstruction and geometry-consistent event generation, we leverage two
complementary datasets. The first is MVImgNet [90], a large-scale multi-view image collection
comprising 6.5 million frames from 219,199 videos across 238 object categories. We sample 1,000
diverse scenes suitable for 3D reconstruction and motion-aware event synthesis from this dataset. To
supplement MVImgNet’s object-centric diversity with scene-level structural richness, we incorporate
DL3DV [40], a dense multi-view video dataset offering accurate camera poses and ground-truth depth
maps across 10,000 photorealistic indoor and ourdoor scenes. We also sample 50 diverse scenes from
its 140 benchmark scenes. DL3DV provides high-quality geometry and illumination cues that are
critical for evaluating spatial and temporal consistency in event simulation.
Totally, we select 1050 scenes from these datasets which enable us to construct a diverse benchmark
for sparse-to-event generation, supporting both object-level and scene-level evaluation under motion
blur and asynchronous observation conditions.
3.4
3D Scene Representation
We employ 3DGS as detailed in Sec. 3.1 to reconstruct high-fidelity 3D scenes from sparse input
views. These views are represented by their corresponding camera poses {Pi = (Ri, Ti)}N
i=1, where
Ri ∈SO(3) is the rotation matrix, and Ti ∈R3 is the translation vector. For the MVImgnet and
DL3DV datasets, we typically use N = 30 and 100 for the number of input views. Given the image
sequence {Ii}N
i=1, we train a 3DGS model for 30,000 iterations to reconstruct a high-fidelity 3D
radiance field. This radiance field captures both the scene’s geometry and appearance, serving as the
foundation for subsequent trajectory interpolation and event stream synthesis.
3.5
Virtual Camera Trajectory Generation
To simulate continuous camera motion essential for realistic event data synthesis, we transform the
initial discrete set of camera poses, often obtained from structure-from-motion with COLMAP [61],
into temporally dense and spatially smooth trajectories. This process involves two primary stages:
(1) initial trajectory refinement and adaptive densification, (2) followed by an optional augmentation
stage for enhanced motion diversity.
5

<!-- page 6 -->
3.5.1
Base Trajectory Refinement and Densification
The raw camera poses {Pi = (Ri, Ti)}N
i=1 can exhibit jitter or abrupt transitions, detrimental to
high-fidelity event simulation. We first address this through local pose smoothing and then generate a
dense base trajectory using velocity-controlled interpolation.
Pose Stabilization via Local Trajectory Smoothing.
To mitigate local jitter and discontinuities,
we apply a temporal smoothing filter to the original camera poses. For each pose Pi, we define a
local temporal window Wi = {Pj | |j −i| ≤w, j ∈N +} with a half-width w (e.g., w = 2). The
smoothed pose P ′
i = (R′
i, T′
i) is computed as:
T′
i =
1
|Wi|
X
j∈Wi
Tj,
(7)
R′
i = Slerp

{Rj}j∈Wi, 1
2

,
(8)
where Slerp(·) denotes spherical linear interpolation of rotations, evaluated at the temporal midpoint
of the window. This procedure enhances local continuity, yielding a smoothed sequence {P ′
i}N
i=1
suitable for subsequent densification.
Velocity-Controlled Dense Interpolation.
Building upon the smoothed poses {P ′
i}, we generate
a temporally uniform but spatially adaptive dense trajectory. Given a desired interpolation mul-
tiplier γ > 1, the target number of poses in the dense trajectory is M = ⌈γ · N⌉. These poses,
{ ˜Pj = ( ˜Rj, ˜Tj)}M−1
j=0 , are sampled at evenly spaced normalized time steps tj = j/(M −1). To
achieve adaptive spatial sampling, we first quantify the motion between adjacent smoothed poses.
The displacement δi between P ′
i and P ′
i+1 is defined as a weighted combination of rotational and
translational changes:
δi = α · θi + β · ∥T′
i+1 −T′
i∥2,
(9)
where θi = cos−1  Tr(R′
i+1(R′
i)⊤)−1
2

is the geodesic distance between orientations R′
i and R′
i+1,
and α, β are weighting coefficients. The cumulative path length up to pose P ′
i is si = Pi−1
k=0 δk, with
s0 = 0. The total path length is sN−1. We then introduce a user-defined velocity profile, which can
be a continuous function v(t) or a discrete list {vk}M−2
k=0 , controlling the desired speed along the
trajectory. This profile dictates the sampling density: higher velocities lead to sparser sampling in
terms of path length per time step. The target path length ˜sj corresponding to each time step tj is
computed by normalized cumulative velocity:
˜sj = sN−1 ·
Pj−1
k=0 vk · ∆t
PM−2
l=0
vl · ∆t
,
(10)
where ∆t = 1/(M −1). Finally, we fit a cubic B-spline curve to the control points {(si, P ′
i)}
(parameterized by cumulative path length si) and sample this spline at the reparameterized path
lengths {˜sj} to obtain the dense trajectory { ˜Pj}. This base trajectory serves as a foundation for
rendering image sequences.
3.5.2
Novel-View Trajectory Augmentation for Enhanced Motion Diversity
To further enrich the dataset with varied camera movements, we generate multiple novel-view mini-
trajectories. These are derived by sampling keyframes from the dense base trajectory { ˜Pj}M−1
j=0
(generated in §3.5.1) and interpolating new paths between them. Specifically, we uniformly sample
G groups of K keyframes from { ˜Pj} without replacement:
Kg =
n
˜P (g)
i1 , . . . , ˜P (g)
iK
o
⊂{ ˜Pj},
g = 1, . . . , G,
(11)
where K ≤Kmax is the number of keyframes per mini-trajectory and we set K = 5. For each group
Kg, we compute cumulative pose displacements along its K keyframes using the metric δk (Eq. 9),
resulting in a local path length s′
K−1. We then generate F uniformly spaced spatial targets along this
local path:
ˆsℓ= ℓ· s′
K−1
F −1 ,
ℓ= 0, . . . , F −1.
(12)
6

<!-- page 7 -->
Novel-view poses { ˆP (g)
ℓ
}F −1
ℓ=0 are interpolated by fitting either a cubic B-spline or a Bézier curve
(with a randomly selected degree d ∈{2, 3, 4, 5}) to the keyframes in Kg, parameterized by their
local cumulative path length, and then sampling at {ˆsℓ}. Each interpolated pose ˆP (g)
ℓ
inherits its
camera intrinsics from the first keyframe ˜P (g)
i1
in its group. Each resulting sequence Cg = { ˆP (g)
ℓ
}F −1
ℓ=0
constitutes a geometry-consistent, temporally uniform mini-trajectory. The collection of all G groups,
{Cg}G
g=1, provides a diverse set of camera motions. In our experiments, we typically set G = 3 and
F = 150. These augmented trajectories, along with the base trajectory, are used for rendering image
sequences for event synthesis.
3.6
Event Synthesis from Rendered Sequences
Given high-temporal-resolution image sequences rendered along diverse virtual camera trajectories
(§3.5), we synthesize event streams using the DVS-Voltmeter model [39], which stochastically
models pixel-level voltage accumulation. This simulator provides temporally continuous and proba-
bilistically grounded event generation, effectively mitigating aliasing artifacts introduced by 3DGS
rasterization [32], especially under fast camera motion.
As discussed in Eq. (5) (Sec. 3.1), a key parameter is the contrast threshold c, denoting the minimum
log-intensity change required to trigger an event. Following the calibration strategy of Stoffregen et
al. [64], we empirically sweep c within [0.25, 1.5], observing that:
• Low thresholds (c ≤0.4) yield dense, low-noise events, resembling IJRR [48], but may
introduce floater artifacts when used with 3DGS (see §B).
• High thresholds (c ≥0.8) produce sparse events with pronounced dynamic features, similar
to MVSEC [99].
The target datasets, MVImgNet [90] and DL3DV [40], generally exhibit moderate motion and textured
surfaces, statistically between IJRR and HQF [64]. Based on this observation and experimental
validation (§4), we adopt c ∈[0.2, 0.5], balancing fine detail preservation and temporal coherence
through the stochastic nature of the Voltmeter model.
4
Experiments
Implementation details. For the reconstruction stage, we carefully collect approximately 2k high-
quality multi-view images from 2 public datasets: MVImageNet [90] and DL3DV [40]: we choose
and render 1.8k scenes from MVImageNet and 100 scenes from DL3DV. We employ the official
interplementation verision of 3DGS [32] in the original setting. For the event generation stage, we
utilize the DVS-Voltmeter simulator [39] to synthesize events from the rendered RGB sequences. We
adopt the following sensor-specific parameters to closely mimic the behavior of real DVS sensors:
the ON and OFF contrast thresholds are both set to ΘON = ΘOFF = 1 as default. We conduct our
experiment on the NVIDIA RTX 3090 24GB, and more details and settings are in the appendix.
Evaluation Baselines and Metrics. We evaluate the proposed dataset across three key dimensions:
(1) 3D reconstruction quality (Sec. 4.1), (2) the domain gap between synthetic and real-world event
streams (Sec. 4.2), and (3) applicability to a range of downstream tasks (Sec. 4.3). For reconstruction-
related evaluations, we adopt standard full-reference image quality metrics: PSNR, SSIM [75], and
LPIPS [94], to assess both novel view synthesis and event-based deblurring performance. To further
evaluate perceptual quality in image restoration and video interpolation tasks, we also employ no-
reference metrics, including CLIPIQA [72], MUSIQ [31], and RANKIQA [42]. These metrics help
quantify the realism, fidelity, and temporal consistency of outputs under different usage scenarios.
4.1
Reliability of 3D Reconstruction
We trained the 3DGS model on the MVImgNet dataset following a rigorous selection of input views
and optimized the model for 30,000 iterations. We then evaluated its rendering fidelity against the
ground-truth test set. Quantitative results demonstrate high reconstruction quality, with an average
PSNR of 29.8, SSIM of 0.92, and a perceptual LPIPS score of 0.14. These results establish a
solid foundation for leveraging 3DGS as a core module for camera pose control, high frame-rate
interpolation, and photorealistic rendering, thereby enabling physically grounded event simulation.
7

<!-- page 8 -->
Table 1: Comparison of different methods under varying motion speeds. Metrics are averaged over
each category (↑: higher is better; ↓: lower is better).
Category
Method
Mild Speed
Medium Speed
Strong Speed
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Event-only
E-NeRF [33]
21.67
0.827
0.216
20.93
0.815
0.244
20.11
0.792
0.260
Event-3DGS [19]
11.19
0.623
0.649
10.34
0.387
0.695
10.68
0.374
0.712
Event-fusion
E-NeRF [33]
21.89
0.834
0.208
21.05
0.820
0.239
20.57
0.809
0.251
Event-3DGS [19]
24.31
0.884
0.118
21.88
0.832
0.224
19.36
0.793
0.295
4.2
Reliability of event sequences
Due to the lack of widely accepted quantitative metrics for evaluating event data quality, recent
proposals such as the Event Quality Score (EQS) [3] offer promising directions for future research.
However, as the EQS implementation is not publicly available, we instead performed qualitative
evaluations using real-world RGB-event datasets. Specifically, we employed the DSEC dataset
from the Robotics and Perception Group at the University of Zurich, which provides synchronized
recordings from RGB and DVS cameras in driving scenarios. To approximate static 3D scene
conditions, we selected scenes with rigid object motion and limited amplitude variation. Visual
comparisons demonstrate that our method produces event distributions more consistent with real data
than conventional video-driven synthesis approaches.
4.3
Application to Multiple Tasks
To evaluate the generalization and practicality of our proposed event dataset, we benchmark it across
three event-vision tasks: 3D reconstruction, image deblurring, and image/video reconstruction. For
all tasks, we compare several state-of-the-art methods, and report standard image quality metrics.
Event-based 3D Reconstruction. We first evaluate the utility of our dataset in static 3D scene
reconstruction. To assess robustness under motion-induced challenges, we simulate static scenes
with varying camera motion speeds (mild, medium, and strong), allowing controlled evaluation of
temporal consistency and appearance fidelity. We further compare grayscale-only and RGB-colored
supervision settings to investigate the effect of color information. As shown in Table 1, while all
methods exhibit performance drops under faster motion, models trained with color supervision
consistently achieve better perceptual quality. These results highlight the versatility of our dataset in
supporting both grayscale and color-aware pipelines, and its suitability for evaluating spatiotemporal
consistency in static scenes captured via asynchronous event observations.
Event-based Video Reconstruction. Finally, we assess the utility of GS2E as a benchmark for event-
driven video reconstruction tasks [57, 70], including frame interpolation and intensity reconstruction.
Owing to its fine-grained temporal resolution, accurate camera motion, and realistic lighting variations,
GS2E provides a challenging yet structured testbed for evaluating reconstruction quality under high-
speed motion. As shown in Table 2, existing models exhibit improved motion continuity and reduced
ghosting artifacts when evaluated on our dataset, highlighting its effectiveness.
4.4
Ablation study
Interplation Methods of Trajectory. To analyze how different interpolation strategies influence
the quality of synthesized event streams and their impact on downstream reconstruction tasks, we
compare linear, Bézier, and cubic B-spline methods for virtual camera trajectory generation, shown
V2E
GS2E (Ours)
GT Event
GT RGB
Figure 4: Qualitative comparison of synthesized event distributions using GS2E versus traditional
video-driven event synthesis methods, evaluated against real-world event data from the DSEC dataset.
8

<!-- page 9 -->
Event-3DGS
Event-fusion
Ground Truth
Blurred Image
Ground Truth
Deblurred Image
(a) Event-vision Task: 3D Reconstruction
(b) Event-vision Task: Image Deblurring
Figure 5: Application to Multiple Tasks. We benchmark it across event-vision tasks: 3D reconstruc-
tion and image deblurring.
Event-based Image Deblurring. We further
test whether event streams generated from our
dataset can support high-quality image restora-
tion under motion blur. Leveraging recent event-
guided deblurring frameworks [62, 65], we eval-
uate reconstruction performance using both syn-
thetic blurry frames paired with our event data.
The results indicate that our dataset effectively
captures motion-dependent blur patterns and
high-frequency temporal cues, which help de-
blurring models produce sharper and more tem-
porally consistent outputs, particularly in low-
light and fast-moving scenes.
Table 2: Comparison of image deblurring and
video reconstruction methods.
Method
PSNR↑
SSIM↑
LPIPS↓
Deblurring Task
D2Net [62]
29.61
0.932
0.113
EFNet [65]
31.26
0.940
0.098
Method
CLIPIQA↑
MUSIQ↑
RANKIQA↓
Video Reconstruction Task
E2VID [57]
0.139
46.52
4.879
TimeLen++ [70]
0.144
48.68
4.325
as in Figure 6. While linear interpolation is efficient, its velocity discontinuities at control points can
undermine temporal coherence in high-fidelity reconstruction. Cubic B-splines, by ensuring smooth
higher-order continuity, yield more realistic trajectories. We thus use cubic B-spline interpolation
with velocity control as the default, balancing smoothness and trajectory realism.
Linear Interpolation
Bézier Interpolation
Velocity-Controlled Dense
B-spline Interpolation (Ours)
Figure 6: Comparison of different interpolation methods shows that our method is smoother and has
speed control capabilities.
5
Conclusion
We presented GS2E, a large-scale synthetic dataset for high-fidelity event stream generation from
sparse real-world multi-view RGB inputs. Our framework combines 3DGS-based scene reconstruc-
tion with a physically grounded simulation pipeline that integrates adaptive trajectory interpolation
and contrast threshold modeling. This design enables temporally dense and geometrically consistent
event synthesis across diverse motion and lighting conditions, effectively bridging the gap between
scalable data generation and sensor-faithful event representation. Extensive experiments validate the
utility of GS2E in downstream tasks such as 3D reconstruction and video interpolation. Future work
will explore integrating exposure-aware camera models into the 3DGS rendering process to better
capture real-world lighting variations and extend the dataset’s applicability to dynamic scenes.
9

<!-- page 10 -->
A
Details of Velocity-Controlled Reparameterization
In our code, we provide two ways to precisely control speed. These are using continuously defined
functions and a discrete speed list.
A.1
Continuous speed function
A positive, analytic function
v : [0, 1] →R>0
(dimensionless),
sampled at normalised time t, directly prescribes the speed curve. In the released dataset we adopt
v(t) = 0.25 sin(t) + 1.1,
t ∈[0, 1],
(13)
A.2
Speed list
An arbitrary-length float array r = {rk}L−1
k=0 (rk > 0) is interpreted as multipliers of a base frame
rate fbase = 2400 fps: the k-th temporal segment

tk, tk+1

of length ∆T = T/L is rendered at
fk = rkfbase. To obtain a continuous speed curve we blend neighbouring segments with a cubic
B-spline1 in a 2τ-second window centred at each boundary,
v(t) = Bspline
 t; r, τ

,
τ ≃0.1 ∆T.
A.3
From speed curve to arc-length samples
Let M be the desired number of interpolated frames, we sample the chosen speed interface on a
uniform grid tj = j/(M −1):
uj = v(tj),
j = 0, . . . , M −2;
(14)
∆sj =
uj
PM−2
k=0 uk
S;
(15)
s0 = 0,
sj+1 = sj + ∆sj.
(16)
Equation (15) rescales the sampled speeds so that P
j ∆sj = S, ensuring the full geometric path is
covered.
A.4
Evaluating the spline
Each interpolated pose ˜Pj = ( ˜Rj, ˜Tj) is obtained by querying the spline at the renormalised
arc-length s⋆
j:
˜Pj = P
 sj

,
j = 0, . . . , M −1.
Because sj+1 −sj ∝v(tj), the linear and angular velocities of the discrete trajectory { ˜Pj} follow
the prescribed speed profile with frame-level accuracy.
A.5
Practical remarks
• Choice of interface. The analytic form (13) is convenient for dataset-level consistency; the speed
list form offers frame-accurate speed control for bespoke sequences.
• Continuity. Both interfaces yield a C2 speed curve, hence the final trajectory is at least C1,
avoiding jerk during rendering.
• Complexity. The whole pipeline is linear in N+M and is CPU-friendly (< 0.5 µs per interpolated
pose).
Summary. Either a compact analytic law (13) or an arbitrary-length speed list can be mapped, via
Eq. (15), to B-spline arc-length samples, providing reliable and precise control over camera velocity
for every rendered frame.
1Order 3 suffices to reach C2 continuity while keeping local support.
10

<!-- page 11 -->
B
Details of choosing the contrast threshold
In our experiments, we found that when we set the contrast threshold c ≤0.75, visible floater
artifacts appeared during the visualization of the event stream. These artifacts occur when the
viewpoint changes and certain Gaussians—originally situated in the background and expected to
be occluded—are mistakenly treated as part of the visible foreground. This misclassification leads
to variations in illumination that induce apparent voltage changes, which the simulator erroneously
interprets as valid event triggers. As a result, the synthesized event stream contains non-physical
textures, manifesting as spurious structures or noise in the visualization. As shown in the figure 7,
once we raise c to 1 or higher, the floater becomes almost invisible.
It is worth noting that when the contrast threshold is set too low, according to the research results
in [53], it will lead to a loss of dynamic range. Therefore, in this paper, we tend to set a larger c
to solve both problems simultaneously. To ensure that events are not overly sparse and sufficient
information integrity is retained, the GS2E dataset was simulated with the parameter setting c = 1.
c = 0.25
c = 0.5
c = 0.75
c = 1.25
c = 1
c = 1.5
Figure 7: Selecting the same viewpoint and time window(1000 us), visualize events simulated from
3DGS with different contrast threshold(c) values. The results show that when c ≤0.75, error events
generated by floater Gaussians can be seen on the integral event diagram, while this phenomenon is
greatly alleviated when c ≥1.
C
Implementation Details
For the reconstruction stage, we carefully collect approximately 2k high-quality multi-view images
from 2 public datasets: MVImageNet [90] and DL3DV [40]: we choose and render 1.8k scenes
from MVImageNet and 100 scenes from DL3DV. We employ the official interplementation verision
of 3DGS [32] in the original setting. For the event generation stage, we utilize the DVS-Voltmeter
simulator [39] to synthesize events from the rendered RGB sequences. We adopt the following
sensor-specific parameters to closely mimic the behavior of real DVS sensors: the ON and OFF
contrast thresholds are both set to ΘON = ΘOFF = 1 as default. The dvs camera parameters are
calibrated as k1 = 0.5, k2 = 1e−3, k3 = 0.1, k4 = 0.01, k5 = 0.1, k6 = 1e−5, following the
original DVS-Voltmeter setting. These control the brightness-dependent drift µ and variance σ2 of
the stochastic process, which determine the polarity distribution and the inverse-Gaussian timestamp
sampling for each event.
All events are simulated at 2400 FPS temporal resolution and stored with microsecond timestamps
for high-fidelity spatio-temporal alignment. The overall process are conducted on a workstation
equipped with 8×NVIDIA RTX 3090 GPUs. The selected MVImageNet clip images vary in size,
but most are approximately 1080p in resolution. When training 3DGS on MVImageNet, each
scene takes an average of 16 minutes. For the camera pose upsampling and trajectory control
stage, using an interpolation factor of γ = 5, the strategy ada_speed, and the velocity function
v(t) = 0.25 sin(t) + 1.1, the average runtime per scene is approximately 45 seconds.
During event simulation, we adopt the same camera parameter configuration as mentioned previously.
However, the simulation time varies significantly depending on the motion amplitude and speed of
the camera, as well as the scene complexity, making it difficult to estimate a consistent runtime.
For the DL3DV dataset, each scene contains 300–400 images. To ensure higher reconstruction and
rendering quality, as well as to generate longer event streams, we do not downsample the input image
11

<!-- page 12 -->
resolution, nor do we slice the image or event sequences. Using the same hardware configuration
as with MVImageNet, the average per-scene training time is approximately 27 minutes, and the
rendering time is around 41 minutes.
D
Existing Event-based 3D Reconstruction Datasets
To contextualize the contribution of GS2E, Table 3 provides a comprehensive comparison of existing
event-based 3D datasets and 3D reconstruction methods [45, 10, 54, 2, 92, 66, 98, 55, 60, 87, 100,
77, 81, 25, 19, 59, 76, 1, 83, 91, 43, 44, 97, 73, 26, 99, 41, 16, 37, 51, 4, 67, 68, 88, 84, 63, 82, 93,
23, 71, 36, 35, 18, 85, 89, 86, 12, 96, 46, 22, 7]. We categorize these into static scenes and dynamic
scenes, based on whether the underlying geometry remains constant or involves temporal variation.
Attributes.
Each dataset is evaluated along key axes:
• Data Type: Whether sharp and/or blurry RGB frames are provided. Blurry frames support
deblurring tasks, while sharp ones aid in geometry fidelity.
• Scene Num / Scale: Number of distinct scenes and their spatial scope (object-level vs.
medium/large indoor scenes).
• GT Poses: Availability of ground-truth camera extrinsics.
• Speed Profile: Whether camera motion follows uniform or non-uniform velocity.
• Multi-Trajectory: Whether each scene supports multiple trajectory simulations, enabling
consistent multi-view observations.
• Device: Capture source—real event sensors (e.g., DAVIS346C, DVXplore) or simulated
streams (e.g., ESIM, Vid2E, V2E).
• Data Source: Origin of the base scene data (e.g., NeRF renderings, Blender, Unreal Engine,
or real-world scenes).
Key Findings.
We observe that existing datasets are limited in several aspects:
• Most datasets focus on small-scale, object-centric scenes with limited spatial or temporal
diversity.
• Simulators typically use simplified trajectories and fixed contrast thresholds, which constrain
realism.
• Real event data remains scarce and often lacks consistent trajectory coverage or paired
ground truth.
• Multi-trajectory support is rare, impeding evaluation under view-consistency and generaliza-
tion settings.
Positioning of GS2E.
Our proposed GS2E benchmark is designed to address these limitations by:
• Leveraging 3D Gaussian Splatting to reconstruct photorealistic static scenes from sparse
real-world RGB inputs.
• Generating controllable, dense virtual trajectories with adaptive speed profiles and multiple
interpolated paths per scene.
• Synthesizing events via a physically-informed simulator that incorporates realistic contrast
threshold modeling.
• Supporting both object- and scene-level scales with consistent multi-view alignment and
temporal density.
By filling the gaps in scale, realism, and trajectory diversity, GS2E enables more robust evaluation of
event-based 3D reconstruction and rendering methods.
12

<!-- page 13 -->
E
Limitation and Broader impacts
Limitation. While GS2E provides high-fidelity, geometry-consistent event data under a wide range of
camera trajectories and motion patterns, it remains fundamentally limited by its reliance on rendered
RGB images from 3DGS. Specifically, the current pipeline inherits the photometric constraints of
3D Gaussian Splatting , which may not faithfully replicate extreme illumination conditions such
as overexposure or underexposure. As a result, scenes with very low light or high dynamic range
may not be accurately modeled in terms of event triggering behavior. Additionally, our framework
currently assumes static scenes; dynamic object motion is not yet modeled. In future work, we plan
to extend the simulator by incorporating physically-realistic camera models into the 3DGS rendering
pipeline, enabling explicit control over exposure, tone mapping, and sensor response curves to better
approximate real-world lighting variability.
Broader impacts. This work introduces a scalable, geometry-consistent synthetic dataset for event-
based vision research. On the positive side, it lowers the barrier for training high-performance models
in domains such as autonomous driving, robotics, and augmented reality, where event-based sensing
offers advantages under fast motion or challenging lighting. By providing a flexible, physically-
grounded simulation framework, the work supports reproducible and ethical AI development. On
the negative side, improved realism in synthetic event data may inadvertently enable misuse such as
generating adversarial inputs or synthetic surveillance data. These risks are mitigated by the dataset’s
academic licensing and transparency in its construction pipeline. Furthermore, the data generation
framework may raise privacy concerns if adapted for real-scene reproduction, which warrants further
community discussion and the adoption of usage safeguards.
F
License of the used assets
• 3D Gaussian Splatting [32]: A publicly available method with its dataset released under
the CC BY MIT license.
• MVImgNet [90]: A publicly available dataset released under the CC BY 4.0 license.
• DL3DV [40]: A publicly available dataset released under the CC BY 4.0 license.
• GS2E: A publicly available dataset released under the CC BY MIT license.
13

<!-- page 14 -->
Method
Column
Type
Color Frame
Scene Num
Scene Scale
GT poses
Speed
Multi-Trajectory
Device
Data Source
Blurry
Sharp
Static Scenes
Event-NeRF [59]
CVPR 2023
Synthetic
×
✓
7
object
✓
Uniform
×
Blender+ESIM
NeRF
Synthetic
✓
✓
7
object
✓
Uniform
×
Blender+ESIM
NeRF
E2NeRF [54]
ICCV 2023
Real
✓
×
5
medium&large
×
Uniform
×
DAVIS346C
Author Collection
Robust e-NeRF [43]
ICCV 2023
Synthetic
×
✓
7
object
✓
Non-Uniform
✓
Blender+ESIM
NeRF
Deblur e-NeRF [44]
ECCV 2024
Synthetic
×
✓
7
object
✓
Non-Uniform
✓
Blender+ESIM
NeRF
Synthetic
✓
✓
9
medium&large
✓
Uniform
✓
Blender+ESIM
NeRF, Deblur-NeRF +Author Collection
EvaGaussian [87]
Arxiv 2024
Real
✓
×
5
medium&large
×
Uniform
×
DAVIS346C
Author Collection
PAEv3D [74]
ICRA 2024
Real
×
✓
6
object
×
Uniform
×
DVXplore event camera
Author Collection
EvDebluRF
CVPR 2024
Synthetic
✓
✓
4
medium
✓
Uniform
×
Blender+ESIM
Deblur-NeRF
Real
✓
×
5
medium
×
Uniform
×
DAVIS346C
Author Collection
EvGGS [73]
ICML 2024
Synthetic
×
×
64
object
✓
Uniform
×
Blender+V2E
Author Collection
IncEventGS [25]
CVPR 2025
Synthetic
×
✓
6
large
✓
Uniform
×
Vid2E
Replica
E-3DGS [92]
3DV 2025
Synthetic
×
✓
3
medium
✓
Non-Uniform
×
-
UnrealEgo
AE-NeRF [10]
AAAI 2025
Synthetic
×
✓
3
medium
✓
Non-Uniform
×
-
UnrealEgo
EF-3DGS [38]
Arxiv 2024
Synthetic
×
✓
9
large
✓
Uniform
×
Vid2E
Tanks and Temples
LSE-NeRF [66]
Arxiv 2024
Real
✓
✓
10
medium&large
✓
Non-Uniform
-
Prophesee EVK-3 HD + Blackfly S (GigE)
Author Collection
GS2E
Submission
Synthetic
✓
✓
1150
medium&large
✓
Non-Uniform
✓
3DGS + DVS-Voltmetre
Author Collection
Dynamic Scenes
DE-NeRF [45]
ICCV 2023
Synthetic
×
✓
3
object
✓
Uniform
×
Blender+ESIM
Author Collection
Real
×
✓
6
medium&large
✓
Uniform
×
Samsung DVS Gen3, DAVIS 346C
Color Event Camera, HS-ERGB(Timeslen)
EvDNeRF [1]
CVPR 2024 Workshop
Synthetic
×
×
3
object
✓
Uniform
✓
Blender+ESIM
Kubric
Dynamic EventNeRF [60]
CVPR 2025 Workshop
Synthetic
×
✓
5
object
✓
Uniform
✓
Blender+ESIM
Author Collection
Real
×
✓
16
medium&large
✓
Uniform
✓
DAVIS346C
Author Collection
Table 3: Comparison of existing event-based 3D reconstruction datasets, categorized by scene type, motion profile, sensor modality, and simulation pipeline.
14

<!-- page 15 -->
References
[1] Anish Bhattacharya, Ratnesh Madaan, Fernando Cladera, Sai Vemprala, Rogerio Bonatti,
Kostas Daniilidis, Ashish Kapoor, Vijay Kumar, Nikolai Matni, and Jayesh K Gupta. Evd-
nerf: Reconstructing event data with dynamic neural radiance fields. In Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5846–5855, 2024.
[2] Marco Cannici and Davide Scaramuzza. Mitigating motion blur in neural radiance fields with
events and frames. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2024.
[3] Kaustav Chanda, Aayush Atul Verma, Arpitsinh Vaghela, Yezhou Yang, and Bharatesh
Chakravarthi. Event quality score (eqs): Assessing the realism of simulated event camera
streams via distances in latent space, 2025.
[4] Kang Chen, Jiyuan Zhang, Zecheng Hao, Yajing Zheng, Tiejun Huang, and Zhaofei Yu. Usp-
gaussian: Unifying spike-based image reconstruction, pose correction and gaussian splatting.
arXiv preprint arXiv:2411.10504, 2024.
[5] Peiyu Chen, Weipeng Guan, and Peng Lu. Esvio: Event-based stereo visual inertial odometry.
IEEE Robotics and Automation Letters, 8(6):3661–3668, 2023.
[6] Blender Online Community. Blender - a 3D modelling and rendering package. Blender
Foundation, Stichting Blender Foundation, Amsterdam, 2018.
[7] Yufei Deng, Yuanjian Wang, Rong Xiao, Chenwei Tang, Jizhe Zhou, Jiahao Fan, Deng Xiong,
Jiancheng Lv, and Huajin Tang. Ebad-gaussian: Event-driven bundle adjusted deblur gaussian
splatting. arXiv preprint arXiv:2504.10012, 2025.
[8] Unreal Engine. Unreal engine. Retrieved from Unreal Engine: https://www. unrealengine.
com/en-US/what-is-unreal-engine-4, 2018.
[9] Zhenxuan Fang, Fangfang Wu, Weisheng Dong, Xin Li, Jinjian Wu, and Guangming Shi.
Self-supervised non-uniform kernel estimation with flow-based motion prior for blind image
deblurring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18105–18114, 2023.
[10] Chaoran Feng, Wangbo Yu, Xinhua Cheng, Zhenyu Tang, Junwu Zhang, Li Yuan, and
Yonghong Tian.
Ae-nerf: Augmenting event-based neural radiance fields for non-ideal
conditions and larger scene. arXiv preprint arXiv:2501.02807, 2025.
[11] Guillermo Gallego, Tobi Delbrück, Garrick Orchard, Chiara Bartolozzi, Brian Taba, Andrea
Censi, Stefan Leutenegger, Andrew J Davison, Jörg Conradt, Kostas Daniilidis, et al. Event-
based vision: A survey. IEEE Trans. Pattern Analysis and Machine Intelligence (PAMI),
2020.
[12] Guillermo Gallego, Henri Rebecq, and Davide Scaramuzza. A unifying contrast maximization
framework for event cameras, with applications to motion, depth, and optical flow estimation.
In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
3867–3876, 2018.
[13] Daniel Gehrig, Mathias Gehrig, Javier Hidalgo-Carrió, and Davide Scaramuzza. Video
to events: Recycling video datasets for event cameras. In Computer Vision and Pattern
Recognition (CVPR), 2020.
[14] Daniel Gehrig, Mathias Gehrig, Javier Hidalgo-Carrió, and Davide Scaramuzza. Video
to events: Recycling video datasets for event cameras. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 3586–3595, 2020.
[15] Daniel Gehrig, Antonio Loquercio, Konstantinos G Derpanis, and Davide Scaramuzza. End-
to-end learning of representations for asynchronous event-based data. In Proceedings of the
IEEE/CVF International Conference on Computer Vision, pages 5633–5643, 2019.
15

<!-- page 16 -->
[16] Mathias Gehrig, Willem Aarents, Daniel Gehrig, and Davide Scaramuzza. DSEC: A stereo
event camera dataset for driving scenarios. IEEE Robotics and Automation Letters, 2021.
[17] Daxin Gu, Jia Li, Yu Zhang, and Yonghong Tian. How to learn a domain-adaptive event
simulator? In Proceedings of the 29th ACM International Conference on Multimedia, pages
1275–1283, 2021.
[18] Shuang Guo and Guillermo Gallego. Cmax-slam: Event-based rotational-motion bundle
adjustment and slam system using contrast maximization. IEEE Transactions on Robotics,
2024.
[19] Haiqian Han, Jianing Li, Henglu Wei, and Xiangyang Ji.
Event-3dgs: Event-based 3d
reconstruction using 3d gaussian splatting. Advances in Neural Information Processing
Systems, 37:128139–128159, 2024.
[20] Haiqian Han, Jiacheng Lyu, Jianing Li, Henglu Wei, Cheng Li, Yajing Wei, Shu Chen, and
Xiangyang Ji. Physical-based event camera simulator. In European Conference on Computer
Vision, pages 19–35. Springer, 2024.
[21] Haiqian Han, Jiacheng Lyu, Jianing Li, Henglu Wei, Cheng Li, Yajing Wei, Shu Chen, and
Xiangyang Ji. Physical-based event camera simulator. In Computer Vision – ECCV 2024: 18th
European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XLV,
page 19–35, Berlin, Heidelberg, 2024. Springer-Verlag.
[22] Weihua He, Kaichao You, Zhendong Qiao, Xu Jia, Ziyang Zhang, Wenhui Wang, Huchuan Lu,
Yaoyuan Wang, and Jianxing Liao. Timereplayer: Unlocking the potential of event cameras
for video interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 17804–17813, 2022.
[23] Javier Hidalgo-Carrió, Guillermo Gallego, and Davide Scaramuzza. Event-aided direct sparse
odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5781–5790, 2022.
[24] Yuhuang Hu, Shih-Chii Liu, and Tobi Delbruck. v2e: From video frames to realistic dvs events.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 1312–1321, 2021.
[25] Jian Huang, Chengrui Dong, and Peidong Liu. Inceventgs: Pose-free gaussian splatting from a
single event camera. arXiv preprint arXiv:2410.08107, 2024.
[26] Inwoo Hwang, Junho Kim, and Young Min Kim. Ev-nerf: Event based neural radiance field.
In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision,
pages 837–847, 2023.
[27] Xiao Jiang, Fei Zhou, and Jiongzhi Lin. Adv2e: Bridging the gap between analogue circuit
and discrete frames in the video-to-events simulator, 2024.
[28] Damien Joubert, Alexandre Marcireau, Nic Ralph, Andrew Jolley, André Van Schaik, and
Gregory Cohen. Event camera simulator improvements via characterized parameters. Frontiers
in Neuroscience, 15:702765, 2021.
[29] Damien Joubert, Alexandre Marcireau, Nic Ralph, Andrew Jolley, André van Schaik, and
Gregory Cohen. Event camera simulator improvements via characterized parameters. Frontiers
in Neuroscience, 15:702765, 2021.
[30] Jacques Kaiser, Juan Camilo Vasquez Tieck, Christian Hubschneider, Peter Wolf, Michael
Weber, Michael Hoff, Alexander Friedrich, Konrad Wojtasik, Arne Rönnau, Ralf Kohlhaas,
Rüdiger Dillmann, and Johann Marius Zöllner. Towards a framework for end-to-end control
of a simulated vehicle with spiking neural networks. 2016 IEEE International Conference on
Simulation, Modeling, and Programming for Autonomous Robots (SIMPAR), pages 127–134,
2016.
16

<!-- page 17 -->
[31] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale
image quality transformer. In Proceedings of the IEEE/CVF international conference on
computer vision, pages 5148–5157, 2021.
[32] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian
splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 2023.
[33] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers. E-nerf: Neural
radiance fields from a moving event camera. IEEE Robotics and Automation Letters, 8(3):1587–
1594, 2023.
[34] Seungjun Lee and Gim Hee Lee. Diet-gs: Diffusion prior and event stream-assisted motion
deblurring 3d gaussian splatting, 2025.
[35] Hao Li, Jinfa Huang, Peng Jin, Guoli Song, Qi Wu, and Jie Chen. Weakly-supervised 3d spatial
reasoning for text-based visual question answering. IEEE Transactions on Image Processing,
32:3367–3382, 2023.
[36] Hao Li, Curise Jia, Peng Jin, Zesen Cheng, Kehan Li, Jialu Sui, Chang Liu, and
Li Yuan. Freestyleret: Retrieving images from style-diversified queries. arXiv preprint
arXiv:2312.02428, 2023.
[37] Wenpu Li, Pian Wan, Peng Wang, Jinghang Li, Yi Zhou, and Peidong Liu. Benerf: neural
radiance fields from a single blurry image and event stream. In European Conference on
Computer Vision, pages 416–434. Springer, 2025.
[38] Bohao Liao, Wei Zhai, Zengyu Wan, Tianzhu Zhang, Yang Cao, and Zheng-Jun Zha. Ef-3dgs:
Event-aided free-trajectory 3d gaussian splatting. arXiv preprint arXiv:2410.15392, 2024.
[39] Songnan Lin, Ye Ma, Zhenhua Guo, and Bihan Wen. Dvs-voltmeter: Stochastic process-based
event simulator for dynamic vision sensors. In European Conference on Computer Vision
(ECCV), 2022.
[40] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo,
Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based
3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 22160–22169, 2024.
[41] Peidong Liu, Xingxing Zuo, Viktor Larsson, and Marc Pollefeys. MBA-VO: Motion Blur
Aware Visual Odometry. In International Conference on Computer Vision (ICCV), 2021.
[42] Xialei Liu, Joost Van De Weijer, and Andrew D Bagdanov. Rankiqa: Learning from rankings
for no-reference image quality assessment. In Proceedings of the IEEE international conference
on computer vision, pages 1040–1049, 2017.
[43] Weng Fei Low and Gim Hee Lee. Robust e-nerf: Nerf from sparse & noisy events under
non-uniform motion. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023.
[44] Weng Fei Low and Gim Hee Lee. Deblur e-nerf: Nerf from motion-blurred events under
high-speed or low-light conditions. In European Conference on Computer Vision, pages
192–209. Springer, 2025.
[45] Qi Ma, Danda Pani Paudel, Ajad Chhatkuli, and Luc Van Gool. Deformable neural radiance
fields using rgb and event cameras. In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 3590–3600, 2023.
[46] Yongrui Ma, Shi Guo, Yutian Chen, Tianfan Xue, and Jinwei Gu. Timelens-xl: Real-time event-
based video frame interpolation with large motion. In European Conference on Computer
Vision, pages 178–194. Springer, 2024.
[47] Elias Mueggler, Henri Rebecq, Guillermo Gallego, Tobi Delbruck, and Davide Scaramuzza.
The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry,
and slam. International Journal of Robotics Research, 36(2):142–149, 2017.
17

<!-- page 18 -->
[48] Elias Mueggler, Henri Rebecq, Guillermo Gallego, Tobi Delbruck, and Davide Scaramuzza.
The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry,
and slam. The International Journal of Robotics Research, 36(2):142–149, 2017.
[49] Liyuan Pan, Miaomiao Liu, and Richard Hartley. Single image optical flow estimation with an
event camera. In Computer Vision and Pattern Recognition (CVPR), 2020.
[50] Md Jubaer Hossain Pantho, Joel Mandebi Mbongue, Pankaj Bhowmik, and Christophe Bobda.
Event camera simulator design for modeling attention-based inference architectures. Journal
of Real-Time Image Processing, 19(2):363–374, 2022.
[51] Cheng Peng, Yutao Tang, Yifan Zhou, Nengyu Wang, Xijun Liu, Deming Li, and Rama
Chellappa. Bags: Blur agnostic gaussian splatting through multi-scale kernel modeling. In
European Conference on Computer Vision, pages 293–310. Springer, 2024.
[52] Mirco Planamente, Chiara Plizzari, Marco Cannici, Marco Ciccone, Francesco Strada, Andrea
Bottino, Matteo Matteucci, and Barbara Caputo. Da4event: towards bridging the sim-to-real
gap for event cameras using domain adaptation. IEEE Robotics and Automation Letters,
6(4):6616–6623, 2021.
[53] Mirco Planamente, Chiara Plizzari, Marco Cannici, Marco Ciccone, Francesco Strada, Andrea
Bottino, Matteo Matteucci, and Barbara Caputo. Da4event: towards bridging the sim-to-real
gap for event cameras using domain adaptation. IEEE Robotics and Automation Letters,
6(4):6616–6623, 2021.
[54] Yunshan Qi, Lin Zhu, Yu Zhang, and Jia Li. E2nerf: Event enhanced neural radiance fields
from blurry images. In International Conference on Computer Vision (ICCV), 2023.
[55] Yunshan Qi, Lin Zhu, Yifan Zhao, Nan Bao, and Jia Li. Deblurring neural radiance fields with
event-driven bundle adjustment. In Proceedings of the 32nd ACM International Conference on
Multimedia, pages 9262–9270, 2024.
[56] Henri Rebecq, Daniel Gehrig, and Davide Scaramuzza. Esim: an open event camera simulator.
In Conference on robot learning, pages 969–982. PMLR, 2018.
[57] Henri Rebecq, René Ranftl, Vladlen Koltun, and Davide Scaramuzza. High speed and high
dynamic range video with an event camera. IEEE Trans. Pattern Anal. Mach. Intell. (T-PAMI),
2019.
[58] Antoni Rosinol, John J Leonard, and Luca Carlone. Nerf-slam: Real-time dense monocular
slam with neural radiance fields. In International Conference on Intelligent Robots and Systems
(IROS), 2023.
[59] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. Eventnerf:
Neural radiance fields from a single colour event camera. In Computer Vision and Pattern
Recognition (CVPR), 2023.
[60] Viktor Rudnev, Gereon Fox, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik.
Dynamic eventnerf: Reconstructing general dynamic scenes from multi-view event cameras.
arXiv preprint arXiv:2412.06770, 2024.
[61] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion Revisited. In
Computer Vision and Pattern Recognition (CVPR), 2016.
[62] Wei Shang, Dongwei Ren, Dongqing Zou, Jimmy S Ren, Ping Luo, and Wangmeng Zuo.
Bringing events into video deblurring with non-consecutively blurry frames. In Proceedings
of the IEEE/CVF international conference on computer vision, pages 4531–4540, 2021.
[63] Zihang Shao, Xuanye Fang, Yaxin Li, Chaoran Feng, Jiangrong Shen, and Qi Xu. Eicil: joint
excitatory inhibitory cycle iteration learning for deep spiking neural networks. Advances in
Neural Information Processing Systems, 36:32117–32128, 2023.
18

<!-- page 19 -->
[64] Timo Stoffregen, Cedric Scheerlinck, Davide Scaramuzza, Tom Drummond, Nick Barnes,
Lindsay Kleeman, and Robert Mahony. Reducing the sim-to-real gap for event cameras. In
Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020,
Proceedings, Part XXVII 16, pages 534–549. Springer, 2020.
[65] Lei Sun, Christos Sakaridis, Jingyun Liang, Qi Jiang, Kailun Yang, Peng Sun, Yaozu Ye,
Kaiwei Wang, and Luc Van Gool. Event-based fusion for motion deblurring with cross-modal
attention. In European Conference on Computer Vision, pages 412–428. Springer, 2022.
[66] Wei Zhi Tang, Daniel Rebain, Kostantinos G Derpanis, and Kwang Moo Yi. Lse-nerf: Learning
sensor modeling errors for deblured neural radiance fields with rgb-event stereo. arXiv preprint
arXiv:2409.06104, 2024.
[67] Zhenyu Tang, Chaoran Feng, Xinhua Cheng, Wangbo Yu, Junwu Zhang, Yuan Liu, Xiaoxiao
Long, Wenping Wang, and Li Yuan. Neuralgs: Bridging neural fields and 3d gaussian splatting
for compact 3d representations. arXiv preprint arXiv:2503.23162, 2025.
[68] Zhenyu Tang, Junwu Zhang, Xinhua Cheng, Wangbo Yu, Chaoran Feng, Yatian Pang, Bin Lin,
and Li Yuan. Cycle3d: High-quality and consistent image-to-3d generation via generation-
reconstruction cycle. arXiv preprint arXiv:2407.19548, 2024.
[69] Gemma Taverni. Applications of Silicon Retinas: From Neuroscience to Computer Vision.
PhD thesis, Universität Zürich, 2020.
[70] Stepan Tulyakov, Alfredo Bochicchio, Daniel Gehrig, Stamatios Georgoulis, Yuanyou Li, and
Davide Scaramuzza. Time lens++: Event-based frame interpolation with parametric non-linear
flow and multi-scale fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 17755–17764, 2022.
[71] Stepan Tulyakov, Daniel Gehrig, Stamatios Georgoulis, Julius Erbach, Mathias Gehrig,
Yuanyou Li, and Davide Scaramuzza. Time lens: Event-based video frame interpolation.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
16155–16164, 2021.
[72] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look
and feel of images. In AAAI, 2023.
[73] Jiaxu Wang, Junhao He, Ziyi Zhang, Mingyuan Sun, Jingkai Sun, and Renjing Xu. Evggs:
A collaborative learning framework for event-based generalizable gaussian splatting. arXiv
preprint arXiv:2405.14959, 2024.
[74] Jiaxu Wang, Junhao He, Ziyi Zhang, and Renjing Xu. Physical priors augmented event-based
3d reconstruction. In 2024 IEEE International Conference on Robotics and Automation
(ICRA), pages 16810–16817. IEEE, 2024.
[75] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment:
from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–
612, 2004.
[76] Jingqian Wu, Shuo Zhu, Chutian Wang, Boxin Shi, and Edmund Y Lam. Sweepevgs: Event-
based 3d gaussian splatting for macro and micro radiance field rendering from a single sweep.
arXiv preprint arXiv:2412.11579, 2024.
[77] Tianyi Xiong, Jiayi Wu, Botao He, Cornelia Fermuller, Yiannis Aloimonos, Heng Huang, and
Christopher A Metzler. Event3dgs: Event-based 3d gaussian splatting for fast egomotion.
arXiv preprint arXiv:2406.02972, 2024.
[78] Chuanzhi Xu, Haoxian Zhou, Haodong Chen, Vera Chung, and Qiang Qu. A survey on
event-driven 3d reconstruction: Development under different categories. arXiv preprint
arXiv:2503.19753, 2025.
[79] Chuanzhi Xu, Haoxian Zhou, Haodong Chen, Vera Chung, and Qiang Qu. A survey on
event-driven 3d reconstruction: Development under different categories. arXiv preprint
arXiv:2503.19753, 2025.
19

<!-- page 20 -->
[80] Chuanzhi Xu, Haoxian Zhou, Langyi Chen, Haodong Chen, Ying Zhou, Vera Chung, and
Qiang Qu. A survey of 3d reconstruction with event cameras: From event-based geometry to
neural 3d rendering, 2025.
[81] Wenhao Xu, Wenming Weng, Yueyi Zhang, Ruikang Xu, and Zhiwei Xiong. Event-boosted de-
formable 3d gaussians for fast dynamic scene reconstruction. arXiv preprint arXiv:2411.16180,
2024.
[82] Jiashu Yang, Ningning Wang, Yian Zhao, Chaoran Feng, Junjia Du, Hao Pang, Zhirui Fang,
and Xuxin Cheng. Kongzi: A historical large language model with fact enhancement. arXiv
preprint arXiv:2504.09488, 2025.
[83] Xiaoting Yin, Hao Shi, Yuhan Bao, Zhenshan Bing, Yiyi Liao, Kailun Yang, and Kaiwei Wang.
E-3dgs: Gaussian splatting with exposure and motion events. arXiv preprint arXiv:2410.16995,
2024.
[84] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera
trajectory for monocular videos via diffusion models. arXiv preprint arXiv:2503.05638, 2025.
[85] Wangbo Yu, Jinhao Du, Ruixin Liu, Yixuan Li, and Yuesheng Zhu. Interactive image inpainting
using semantic guidance. In 2022 26th international conference on pattern recognition (ICPR),
pages 168–174. IEEE, 2022.
[86] Wangbo Yu, Yanbo Fan, Yong Zhang, Xuan Wang, Fei Yin, Yunpeng Bai, Yan-Pei Cao, Ying
Shan, Yang Wu, Zhongqian Sun, et al. Nofa: Nerf-based one-shot facial avatar reconstruction.
In ACM SIGGRAPH 2023 conference proceedings, pages 1–12, 2023.
[87] Wangbo Yu, Chaoran Feng, Jiye Tang, Xu Jia, Li Yuan, and Yonghong Tian. Evagaussians:
Event stream assisted gaussian splatting from blurry images. arXiv preprint arXiv:2405.20224,
2024.
[88] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao,
Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models
for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.
[89] Wangbo Yu, Li Yuan, Yan-Pei Cao, Xiangjun Gao, Xiaoyu Li, Wenbo Hu, Long Quan, Ying
Shan, and Yonghong Tian. Hifi-123: Towards high-fidelity one image to 3d content generation.
In European Conference on Computer Vision, pages 258–274. Springer, 2024.
[90] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng
Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A large-scale dataset
of multi-view images. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 9150–9161, 2023.
[91] Toshiya Yura, Ashkan Mirzaei, and Igor Gilitschenski. Eventsplat: 3d gaussian splatting from
moving event cameras for real-time rendering. arXiv preprint arXiv:2412.07293, 2024.
[92] Sohaib Zahid, Viktor Rudnev, Eddy Ilg, and Vladislav Golyanik. E-3dgs: Event-based novel
view rendering of large-scale scenes using 3d gaussian splatting. 3DV, 2025.
[93] Junwu Zhang, Zhenyu Tang, Yatian Pang, Xinhua Cheng, Peng Jin, Yida Wei, Wangbo Yu,
Munan Ning, and Li Yuan. Repaint123: Fast and high-quality one image to 3d generation with
progressive controllable 2d repainting. arXiv preprint arXiv:2312.13271, 2023.
[94] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unrea-
sonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE
conference on computer vision and pattern recognition, pages 586–595, 2018.
[95] Zhongyang Zhang, Shuyang Cui, Kaidong Chai, Haowen Yu, Subhasis Dasgupta, Upal
Mahbub, and Tauhidur Rahman. V2CE: Video to continuous events simulator. In IEEE
International Conference on Robotics and Automation (ICRA), 2024.
[96] Ziran Zhang, Xiaohui Li, Yihao Liu, Yujin Wang, Yueting Chen, Tianfan Xue, and Shi
Guo. Egvd: Event-guided video diffusion model for physically realistic large-motion frame
interpolation. arXiv preprint arXiv:2503.20268, 2025.
20

<!-- page 21 -->
[97] Zixin Zhang, Kanghao Chen, and Lin Wang. Elite-evgs: Learning event-based 3d gaussian
splatting by distilling event-to-video priors. arXiv preprint arXiv:2409.13392, 2024.
[98] Lingzhe Zhao, Peng Wang, and Peidong Liu. Bad-gaussians: Bundle adjusted deblur gaussian
splatting. arXiv preprint arXiv:2403.11831, 2024.
[99] Alex Zihao Zhu, Dinesh Thakur, Tolga Özaslan, Bernd Pfrommer, Vijay Kumar, and Kostas
Daniilidis. The multivehicle stereo event camera dataset: An event camera dataset for 3d
perception. IEEE Robotics and Automation Letters, 3(3):2032–2039, 2018.
[100] Alex Zihao Zhu, Ziyun Wang, Kaung Khant, and Kostas Daniilidis. Eventgan: Leveraging
large scale image datasets for event cameras. pages 1–11, 2021.
[101] Alex Zihao Zhu, Liangzhe Yuan, Kenneth Chaney, and Kostas Daniilidis. Unsupervised
event-based learning of optical flow, depth, and egomotion. In Computer Vision and Pattern
Recognition (CVPR), 2019.
21
