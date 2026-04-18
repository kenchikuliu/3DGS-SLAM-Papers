<!-- page 1 -->
ReconDrive: Fast Feed-Forward 4D Gaussian
Splatting for Autonomous Driving Scene
Reconstruction
Haibao Yu2,1,∗,‡ Kuntao Xiao1,∗, Jiahang Wang1, Ruiyang Hao3,1, Yuxin Huang4,1, Guoran Hu5,1, Haifang
Qin1, Bowen Jing1, Yuntian Bo1, Ping Luo2
1Tuojing Intelligence
2The University of Hong Kong, 3King’s College London
4The University of Sydney, 5Mohamed bin Zayed University of Artificial Intelligence
∗equal contribution, ‡corresponding author
High-fidelity visual reconstruction and novel-view synthesis are essential for realistic closed-loop
evaluation in autonomous driving. While 4D Gaussian Splatting (4DGS) offers a promising balance of
accuracy and efficiency, existing per-scene optimization methods require costly iterative refinement,
rendering them unscalable for extensive urban environments.
Conversely, current feed-forward
approaches often suffer from degraded photometric quality. To address these limitations, we propose
ReconDrive, a feed-forward framework that leverages and extends the 3D foundation model VGGT
for rapid, high-fidelity 4DGS generation. Our architecture introduces two core adaptations to tailor
the foundation model to dynamic driving scenes: (1) Hybrid Gaussian Prediction Heads, which
decouple the regression of spatial coordinates and appearance attributes to overcome the photometric
deficiencies inherent in generalized foundation features; and (2) a Static-Dynamic 4D Composition
strategy that explicitly captures temporal motion via velocity modeling to represent complex dynamic
environments. Benchmarked on nuScenes across reconstruction, synthesis, and 3D perception tasks,
ReconDrive outperforms all existing feed-forward baselines and surpasses per-scene optimization
methods in eight out of nine evaluation metrics. By delivering superior quality with substantially
fewer resources, our method provides a highly scalable and efficient solution for large-scale autonomous
driving scene reconstruction and simulation.
Code: https://github.com/TuojingAI/ReconDrive
1
Introduction
Closed-loop evaluation is increasingly vital for end-to-end autonomous driving, requiring the ego-vehicle to
receive sensor observations that faithfully correspond to its actions within a simulated environment. Visual
scene reconstruction and novel-view synthesis from real-world multi-view imagery provide the bedrock for
such simulations. Existing approaches generally fall into three categories: asset importation into game engines
like CARLA Dosovitskiy et al. (2017), implicit representation via NeRF Mildenhall et al. (2021) or video
diffusion, and explicit geometric modeling. Among these, 4D Gaussian Splatting (4DGS) Wu et al. (2024), an
extension of 3DGS Kerbl et al. (2023) with a temporal dimension, effectively balances geometric accuracy,
photometric fidelity, and real-time rendering, making it ideal for interactive driving simulations.
However, traditional 4DGS methods, such as Street Gaussian Yan et al. (2024), rely on per-scene optimization.
These approaches iteratively optimize Gaussian kernels to minimize rendering loss, often requiring LiDAR
priors for initialization. This non-data-driven paradigm fails to leverage shared structural knowledge across
scenes and incurs substantial computational overhead for every new environment, limiting scalability. In
contrast, feed-forward models like VGGT Wang et al. (2025) demonstrate the potential of 3D foundation models
for efficient reconstruction by directly producing explicit geometry. Yet, extending VGGT to 4D Gaussian
reconstruction for autonomous driving presents three primary challenges. First is photometric deficiency, as
VGGT features lack the fine-grained detail necessary to regress high-fidelity appearance attributes. Second is
temporal staticity, where static backbones cannot represent the dynamic motion of traffic participants. Third
is domain and calibration mismatch, where data gaps lead to 3D geometry prediction errors and a failure to
1
arXiv:2603.07552v1  [cs.CV]  8 Mar 2026

<!-- page 2 -->
Multi-view Input at Time 
Multi-view Input at Time 
Dynamic Object Masks
SAM 2
DINOv2
VGGT Backbone
GCPH
GPPH
Gaussian Params
Gaussian Centers
Calibration
Alternating
Attention
+ LoRA
Displacement
Spatiotemporal  
Alignment
Ego Tranformation
Dynamic Gaussian Moving
Concatenation
Velocity
Estimation
4D Gaussians
Figure 1
ReconDrive Inference Framework. ReconDrive is a powerful feed-forward 4D Gaussian splatting
generation framework tailored for urban scene reconstruction and novel-view synthesis. In this framework, we select
two context frames from each segment of an urban scene as input, and adopt a static-dynamic composition strategy to
represent 4D Gaussians. To further enhance geometric precision, we design a dedicated Gaussian prediction head,
consisting of a Gaussian Parameter Prediction Head (GPPH) and a Gaussian Center Prediction Head (GCPH), which
enables the generation of Gaussians with more accurate geometric attributes.
utilize the pre-calibrated sensor intrinsics and extrinsics inherent in driving datasets.
To address these limitations, we propose ReconDrive, a feed-forward framework that reconstructs 4D Gaussian
Splatting representations from vision-only inputs. ReconDrive utilizes a pre-trained VGGT-style backbone to
directly regress spatio-temporal Gaussian parameters, enabling efficient reconstruction at scale without per-
scene optimization. The inference pipeline is built upon three design pillars. First, we employ hybrid Gaussian
prediction heads; we concatenate raw images with dense pixel-wise features to provide the photometric cues
required for regressing appearance attributes like opacity and spherical harmonics, while incorporating camera
calibration directly into the centers head for precise spatial positioning. Second, we utilize a static-dynamic
4D composition, leveraging the SAM2 Ravi et al. (2024) foundation model to extract object masks and
assign temporal velocities to dynamic Gaussians through pixel-wise correspondence. Third, we implement
segment-wise temporal fusion to handle view sparsity over long driving sequences by partitioning the scene
into segments and fusing localized Gaussian clusters into a unified 4D representation. The overall inference
framework is illustrated in Fig. 1. During training, we utilize frozen VGGT Wang et al. (2025) weights
and apply LoRA Hu et al. (2022) for parameter-efficient fine-tuning on urban driving data. We introduce a
novel-frame projection loss to facilitate stable geometry reconstruction against unseen temporal views.
To ensure a fair comparison, we establish a comprehensive benchmark on the nuScenes Caesar et al. (2020)
dataset, reproducing five representative optimization and feed-forward methods under a unified protocol.
We evaluate performance across scene reconstruction, novel-view synthesis, and downstream tasks such as
3D object detection and tracking. Within this benchmark, ReconDrive significantly outperforms existing
feed-forward approaches across all metrics and surpasses per-scene optimization methods in eight out of nine
categories. Our results demonstrate that feed-forward approaches can achieve competitive and often superior
novel-view synthesis performance while requiring substantially fewer resources for Gaussian generation. This
highlights the promising potential of feed-forward paradigms for large-scale autonomous driving reconstruction.
Our contributions can be summarized as follows.
• We introduce ReconDrive, a feed-forward framework that can directly generate 4D Gaussian splatting
without per-scene optimization, to reconstruct urban scenes and enable novel-view synthesis across time.
• We introduce hybrid Gaussian prediction heads, a static-dynamic composition strategy, and a segment-wise
2

<!-- page 3 -->
temporal fusion representation. Combined with specialized training, these components enable 3D foundation
models in dynamic urban environments.
• We build autonomous driving scene reconstruction benchmark by reproducing typical baselines on the
nuScenes Caesar et al. (2020). Our ReconDrive achieves state-of-the-art performance among feed-forward
methods and competitive results against optimization-based baselines across multiple evaluation protocols.
2
Related Work
Feed-Forward 3D and 4D Reconstruction.
Feed-forward approaches have demonstrated significant
potential in directly processing multi-view images to infer 3D geometric attributes. Foundational architectures
such as DUSt3R Wang et al. (2024) and VGGT Wang et al. (2025) bypass traditional optimization by directly
predicting 3D point clouds and depth maps. To further enhance efficiency, FastVGGT Shen et al. (2025)
introduces region-based random sampling, which markedly reduces inference latency, particularly for large-scale
inputs. Concurrently, increasingly powerful models like Concurrently, models like Depth Anything V3 Lin
et al. (2025) and SAM3D Chen et al. (2025b) have advanced generalizable 3D scene understanding. Beyond
point clouds, MVSplat Chen et al. (2024) and AnySplat Jiang et al. (2025) extend this paradigm to Gaussian
Splatting, though primarily for static environments. Recent works have begun to address temporal dynamics
within this paradigm. StreamVGGT Zhuo et al. (2025) enables streaming reconstruction by maintaining a
memory cache of previously computed tokens and leveraging temporal causal attention to accelerate inference
over time. In the driving domain, DriveVGGT Jia et al. (2025) utilizes the VGGT backbone within a
scale-aware framework to output depth and point clouds specifically for autonomous driving data. Our work
shifts the focus from static geometric design to enhancing dynamic scene representation and novel-view fidelity.
By adapting the VGGT formulation specifically for the complexities of autonomous driving, we achieve
significant gains in visual quality and temporal consistency through improved feature utilization.
Gaussian Splatting for Autonomous Driving Scene Reconstruction.
Existing methods primarily rely
on per-scene optimization, iteratively refining Gaussian kernels to minimize rendering loss. AutoSplat Khan
et al. (2025) achieves high-fidelity reconstruction by imposing geometric constraints on varying regions, while
PVG Chen et al. (2023) introduces periodic vibration Gaussians for dynamic urban environments. To handle
the complexity of traffic, StreetGaussians Yan et al. (2024) decomposes scenes into static backgrounds and
moving vehicles, and DrivingGaussian Zhou et al. (2024) introduces a dynamic Gaussian graph for moving
objects. Similarly, S3Gaussian Huang et al. (2024) utilizes a spatial-temporal field network to model 4D
dynamics. Despite their high quality, these optimization-based methods require hours of computation per scene,
making them unscalable for large-scale autonomous driving datasets. In contrast, the feed-forward paradigm
generalizes to diverse scenes without requiring per-scene optimization. Drivingforward Tian et al. (2025)
reconstructs adjacent frames through interpolation, while STORM Yang et al. (2025) employs a transformer
to generate per-frame Gaussians and scene flow via learnable motion tokens. However, these methods do not
leverage 3D foundation models, which limits their representative capacity. More recently, WorldSplat Zhu et al.
(2025) introduced a 4D-aware latent diffusion model for pixel-aligned Gaussian production, and DGGT Chen
et al. (2025a) proposed diffusion-based refinement to mitigate motion artifacts. Nevertheless, the integration
of diffusion models significantly increases inference latency. Unlike these approaches, ReconDrive leverages
a 3D foundation model to enable scalable 4D Gaussian prediction with the well-designed architecture. By
incorporating explicit motion and temporal consistency, we achieve efficient urban scene reconstruction without
the need for iterative optimization or high-latency post-refinement.
3
Method
We introduce ReconDrive, a feed-forward framework to process the urban scenes and generate the 4D gaussian
splatting in one pass. We start by introducing the problem in Sec. 3.1, followed by our model design in Sec. 3.2
and 4D gaussian splatting in Sec. 3.3, and finally the training in Sec. 3.4.
3

<!-- page 4 -->
3.1
Problem Definition and Overall Inference
Problem Formulation.
We formalize visual scene reconstruction and novel-view synthesis as the recovery
of a continuous 4D representation from discrete observations. The input is an urban scene sequence D =
{(Io
t , Co, Et) | t ∈[0, T], o = 1, . . . , O}, where Io
t ∈RH×W ×3 is the RGB image captured at time t by the o-th
camera. Each camera is characterized by a static calibration matrix Co, which encapsulates both intrinsic
parameters and the fixed extrinsic transformation relative to the ego-vehicle coordinate system. The temporal
movement of the scene is captured by the time-varying ego-vehicle pose Et ∈SE(3), which maps the vehicle’s
local coordinate frame to a global world frame. The objective is to utilize a feed-forward model to map D
to a 4D Gaussian Splatting representation that decouples static geometry from dynamic traffic participants.
For any arbitrary timestamp t′ ∈[0, T] and a novel viewpoint Enovel
t′
, the model retrieves and projects the
corresponding temporal Gaussian kernels to render high-fidelity observations.
Challenges.
Developing a feed-forward 4DGS framework for autonomous driving entails three primary
hurdles: 1) Scalability vs. Efficiency: Balancing extensive spatial coverage of large-scale urban environments
with the requirements for real-time rendering. 2) Spatio-Temporal Modeling: Reconstructing dynamic scenes
where 4D Gaussians must explicitly decouple and model motion, moving beyond the static assumptions of
traditional multi-view foundation models. 3) Geometric-Photometric Alignment: Ensuring that regressed
Gaussian attributes maintain both accurate 3D geometry for spatial consistency and high photometric fidelity
for realistic sensor simulation. In this work, we address these requirements by extending the capabilities of
existing 3D foundation models. We propose a specialized architecture and training strategy that leverages the
geometric priors of the foundation model while adapting them to the high-dynamic and calibrated nature of
driving scenes.
Inference Pipeline Overview.
To maintain computational tractability over long-duration sequences, we
adopt a segment-wise representation. The scene is partitioned into temporal segments {[Ts, Ts+1]}, with 4D
Gaussians generated for each segment independently. This approach maximizes environmental coverage while
limiting the active Gaussian count during rendering to ensure real-time performance. Within each segment, a
4D Gaussian kernel Gi(t) is defined by a center-moving paradigm:
Gi(t) = (µi(t), Ri, si, αi, ci)
(1)
where Ri ∈SO(3) is the rotation matrix, si ∈R3 is the anisotropic scaling vector, αi ∈[0, 1] is the opacity,
and ci ∈Rk denotes the spherical harmonic coefficients for view-dependent color. We further employ a
static-dynamic composition strategy. For static background Gaussians, the center µi(t) remains constant. For
dynamic object Gaussians, we assume local linear motion within a segment:
µi(t) = µi,init + vi · (t −Ts)
(2)
where vi ∈R3 represents the regressed velocity vector. By using the frames {Ts, Ts+1} as context, our
ReconDrive framework regresses these spatio-temporal parameters to construct a unified 4D representation.
The detailed architecture and training strategy follow in the subsequent sections.
3.2
Backbone and Prediction Heads
We employ a feature backbone to extract dense image tokens, which are processed by a dual-path architecture
consisting of the Gaussian Center Prediction Head (GCPH) and the Gaussian Parameter Prediction Head
(GPPH). Together, these heads regress spatial coordinates and appearance attributes to form the complete
Gaussian parameters.
Feature Backbone.
The input consists of multi-view images from two context frames (2 × O images, where
O = 6 for the nuScenes dataset Caesar et al. (2020)). The backbone follows the VGGT Wang et al. (2025)
architecture, utilizing two cascaded stages to extract and fuse spatio-temporal features:
• Image Tokenization: Each image is first downsampled to a resolution of H′ × W ′ (with a scale factor
of 3). We then utilize a DINOv2 Oquab et al. (2023) encoder to transform the resized images into dense
4

<!-- page 5 -->
feature tokens. The encoder applies a patch size of p = 14, resulting in tokens of dimension d = 1024 and a
spatial grid resolution of (H′/p) × (W ′/p).
• Alternating-Attention Fusion: The tokens from all 2 × O images are concatenated and passed through
24 Transformer layers utilizing Alternating-Attention. This mechanism alternates between frame-wise
self-attention, which maintains local structural consistency within each timestamp, and global self-attention,
which captures cross-view and temporal geometric correlations.
Hybrid Gaussian Prediction Head.
While VGGT Wang et al. (2025) excels at recovering explicit 3D
geometry, such as depth and point maps, its latent features are optimized for structural consistency rather than
the fine-grained photometric details required for high-fidelity rendering. To bridge this gap, we move beyond
direct regression from backbone features and design the Hybrid Gaussian Prediction Head to leverage the
geometric priors of 3D foundation models while resolving their inherent limitations. Specifically, it overcomes
photometric deficiency by fusing raw image cues into the attribute regression path for high-fidelity rendering.
Simultaneously, it eliminates spatial misalignment by explicitly incorporating sensor calibration into the
GCPH, ensuring the reconstructed scene is precisely anchored within the ego-vehicle’s calibrated coordinate
system. In the Appendix, we further illustrate the spatial misalignment encountered when directly employing
VGGT for autonomous driving scene reconstruction and 3D geometry generation.
• Gaussian Center Prediction Head. We firstly adopt a Dense Prediction Head (DPT) Ranftl et al.
(2021) to upsample the fused transformer features back to the originally resized image resolution (H
′ × W
′)
as FCo
t . Each FCo
t is then input into a 3 × 3 convolutional layer to generate the pixel-wise depth map
DP o
t with H
′ × W
′ size. Next, we project each depth map into the 3D space with camera intrinsic and
extrinsic parameters, to obtain the gaussian 3D center corresponding to the pixel. Note that the two frames’
gaussian 3D centers are located in their ego coordinate separately. Utilizing the calibrated sensor data
improves more accurate gaussian location prediction.
• Gaussian Parameter Prediction Head. Similar to Gaussian Center Prediction Head, we also use DPT
to upsample the fused transformer features back to the originally resized image resolution (H
′ × W
′) as
F o
t . Then we incorporate a shortcut connection to concatenate the original image with the upsampled
features to FP o
t . This is extremely important in the gaussian parameters prediction, because it enables the
capture of high-frequency texture and color details that may be lost in transformer feature downsampling.
A convolutional layer is used to process FP o
t to generate the extra gaussian parameters.
Finally, we obtain a pixel-wise 3D gaussian set of the two context frames Ts and Ts+1: G(Ts) and G(Ts+1).
Furthermore, to mitigate the domain gap between general 3D data and large-scale urban driving scenes, we
adopt a parameter-efficient fine-tuning strategy. We utilize frozen pre-trained VGGT Wang et al. (2025)
weights as a robust structural prior and apply LoRA Hu et al. (2022) to adapt the model to the autonomous
driving domain. For our implementation, we configure the LoRA adapters with a rank r = 8 and a scaling
factor α = 32.
3.3
4D Gaussian Generation
This part further explains how we obtain 4D Gaussians with static-dynamic composition and generated GTs
and GTs+1.
Dynamic Object Mask and Motion Estimation.
The autonomous driving environment can be primarily
categorized into two components: static background elements (e.g., sky, roads, buildings, and vegetation)
and dynamic traffic participants (e.g., vehicles and pedestrians)—the latter being the main contributors to
dynamic scene changes. Here, we estimate the motion flow for the Gaussian kernels corresponding to these
dynamic traffic participants.
First, we leverage the foundation model SAM2 Ravi et al. (2024) to extract instance-level regions of traffic
participants (with a particular focus on vehicles) from the two context frames. SAM2 is chosen for its robust
mask extraction capability and efficient inference speed, which aligns with our real-time inference requirements.
Next, we calculate the velocities of dynamic objects in the ego-coordinate system of frame Ts. The nuScenes
dataset Caesar et al. (2020) provides 3D bounding box annotations for each object. For each dynamic object,
5

<!-- page 6 -->
we transform its 3D location at frame Ts+1 into the ego-coordinate system of frame Ts, then compute the
object’s displacement. Within the short time segment [Ts, Ts+1], we assume the object undergoes rigid-body
motion with a constant velocity. Thus, the object’s velocity is estimated as:
v = displacement / (Ts+1 −Ts)
(3)
Figure 2 Mapping Among Images, Masks,
Dense Features, and Gaussian Indices. (a)
Input image. (b) Dynamic object mask. (c) Dense
features.
(d) Gaussian kernels.
Each pixel in
the dense feature map generates a corresponding
Gaussian kernel, enabling consistent pixel-wise
mapping across all these components.
Note that we can also calculate the object displacement by
two-frame gaussian centers for dataset without 3d object
annotations. Through this process, we obtain the dynamic
object motion flow V with dimensions H
′ × W
′ × 3, where
all pixel regions within the object’s mask share the same
velocity. As illustrated in Fig. 2, a pixel-wise spatial map-
ping exists between the image space and the dense feature
space associated with the Gaussian kernels. Thus, for the
i-th Gaussian kernel at frame Ts, we derive its 3D center
at any time t using the following equation:
µi(t) = µi(Ts) + V (i) · (t −Ts),
t ∈[Ts, Ts+1].
(4)
Temporal Alignment and Fusion.
Here we explain how
we align and fuse the two context-frame Gaussians G(Ts)
and G(Ts+1) into final 4D Gaussians for segment [Ts, Ts+1]
by transforming the frame Ts’s Gaussians into frame Ts’s
ego coordinate and time Ts. Firstly, we spatially transform
the frame Ts+1’s Gaussians into frame Ts’s ego coordinate system with the ego transformation matrix from
frame Ti’s ego vehicle pose ETi to frame Ti+1’s ego vehicle pose ETi+1. We mainly consider transforming the
gaussian centers, rotation, and color coefficients parameters. Then, temporally align the G(Ts+1) to time
Ts by moving the dynamic object gaussian centers with velocity flow V . The two steps cannot be swapped
because the velocity field is computed in Ts’s ego frame. Finally, we get transformed Gaussians G
′(Ts+1),
and concatenate the two Gaussians G(Ts) and G
′(Ts+1). These concatenated Gaussians and the velocity flow
comprise the final 4D gaussian splatting.
3.4
Training
Training Loss.
To enhance visual fidelity, we adopt perceptual loss using a pre-trained VGG-19 net-
work Simonyan & Zisserman (2014), which computes the L1 distance between high-level features of rendered
images ˆI and real images I: Lpercep =
ϕ(ˆI) −ϕ(I)

1, where ϕ(·) denotes the feature extractor of a specific
convolutional layer, encouraging perceptually consistent features. L2 loss is used to constrain pixel-wise
intensity differences between rendered and real images: Ll2 =
ˆI −I

2. Combining them, the rendering loss
for supervising photometric rendering is formulated as:
Lrender = λpercep · Lpercep + λl2 · Ll2
(5)
To stabilize Gaussian center prediction training and improve geometric accuracy, we introduce projection
loss to enforce consistency when warping a single source frame t to the reference frame t′. It combines a
weighted L1 loss (measuring pixel-wise intensity differences between warped predicted images ˆIwarped and
warped ground-truth images Iwarped, GT) and a weighted SSIM loss (evaluating structural and perceptual
consistency via 1 −SSIM(ˆIwarped, Iwarped, GT)). Both components are computed under a valid warping mask,
with the final loss defined as:
Lproject = Lmasked (λl1 · Ll1 + λssim · Lssim)
(6)
where Lmasked is the mask-weighted loss computation, and λl1, λssim are the weight coefficients for L1 and
SSIM loss. More projection loss details are provided in the Appendix.
6

<!-- page 7 -->
Figure 3 Visual Comparisons of Scene Reconstruction and Novel-View Synthesis. Compared with per-scene
optimization methods (Street Gaussians, PVG, DeformableGS, OminiRe), our ReconDrive maintains high-quality
visual rendering in both scene reconstruction (Original View) and novel-view synthesis (1–3 meters of lateral movement
leftward and rightward). Compared with the feed-forward method DrivingForward, ReconDrive preserves accurate
geometric consistency—evident in the tree and the surroundings—and the performance improvement becomes more
pronounced as the moving distance increases. Additionally, it exhibits less image distortion and blurriness, particularly
in image boundary regions. Extended visual comparisons are provided in the Appendix.
The norm loss acts as a regularization term for Gaussian scale and opacity parameters:
Lnorm = λscale · E [∥scalemaps∥2] + λopacity · E

|opacitymaps|

(7)
where scalemaps denotes Gaussian scale parameter maps, and the L2-norm ∥· ∥2 constrains scale magnitude
to avoid extreme values; opacitymaps represents Gaussian opacity maps, and the L1-norm enforces sparsity by
encouraging opacities to converge to 0 or 1. λscale and λopacity are corresponding weight coefficients.
Finally, ReconDrive is trained end-to-end with the total loss:
L = Lrender + Lproject + Lnorm.
(8)
4
Experiments
This section presents a visual urban scene reconstruction benchmark comprising state-of-the-art per-scene
optimization and feed-forward methods on the nuScenes Caesar et al. (2020) dataset, and compares our
approach against these baselines under multiple evaluation protocols to demonstrate its effectiveness.
4.1
Benchmark
Dataset.
The nuScenes Caesar et al. (2020) is a classical autonomous driving dataset originally designed
for 3D perception tasks, comprising 1,000 urban scenes. Each scene has approximately 20 s duration and
contains multi-view images captured by six cameras. We use the original 700 training scenes to train the
feed-forward model and select 14 representative validation scenes from the original validation split, covering
diverse conditions in time (day/night), weather (sunny/rainy), driving behaviors (static/straight/turning),
and traffic density. The detailed list of selected validation scenes is provided in the Appendix. Following Li
7

<!-- page 8 -->
Method
Visual Scene Reconstruction
Novel-View Synthesis
Inference Speed ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
(Per Scene)
Per-Scene Optimization Methods
Street Gaussians Yan et al. (2024)
29.18
0.8824
0.1658
22.98
0.6959
0.2948
31min
PVG Chen et al. (2023)
29.58
0.8839
0.2200
23.48
0.6919
0.2897
23min
DeformableGS Yang et al. (2024)
28.93
0.8832
0.1610
23.73
0.6919
0.2342
46min
OminiRe Chen et al. (2025c)
29.42
0.8853
0.1577
23.01
0.6885
0.2762
35min
Feed-Forward Methods
Drivingforward Tian et al. (2025)
22.83
0.7650
0.2563
21.88
0.6866
0.2979
5s
ReconDrive (Ours)
32.66
0.9589
0.0618
23.99
0.7234
0.2591
15s
Table 1 Scene Reconstruction and Novel-View Synthesis Evaluation Results on the nuScenes Dataset.
et al. (2025), we adopt a frame rate of 12 Hz as key frames. For each evaluation scene, frames at 0.5-second
intervals (i.e., every 6th frame) serve as context frames for reconstruction, while the remaining frames are
reserved for novel-view synthesis evaluation.
Evaluation Protocols and Metrics.
We evaluate the performance of models in the following three
manners. We also report the gaussian generation speed for each scene.
• Visual Scene Reconstruction. We use the context frames (i.e., every 6th frame) as ground truth images,
and use Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Mapping (SSIM), and Learned
Perceptual Image Patch Similarity (LPIPS) as evaluation metrics.
• Novel-View Synthesis. We use the non-context frames (i.e., every 2nd-5th frames) as ground truth images,
and also use PSNR, SSIM, and LPIPS as evaluation metrics.
• Novel-View Synthesis for 3D Perception. To evaluate novel-view synthesis for 3D perception, we simulate
lateral camera shifts by displacing the ego-vehicle trajectory by ±1m, ±2m, and ±3m relative to the original
route. We then evaluate the 3D perception performance of UniAD Hu et al. (2023), which is pre-trained on
the original nuScenes images at 2 Hz, using these rendered multi-view outputs across all seven offsets (0m
and ±1m, ±2m, ±3m). For a consistent evaluation, we also provide the rendered images to UniAD at a 2
Hz frame rate. 3D perception results include 3D object detection and tracking (only vehicle type).
Baseline Models and Settings.
We evaluate two categories of baseline approaches, comprising per-scene
optimization methods and feed-forward methods. For the optimization-based category, we adapt the open-
source DriveStudio Chen et al. (2025c) codebase to support the 12 Hz nuScenes dataset Caesar et al. (2020),
as it was originally designed for the Waymo dataset Sun et al. (2020). Within this framework, we evaluate
four state-of-the-art optimization methods: Street Gaussians Yan et al. (2024), PVG Chen et al. (2023),
DeformableGS Yang et al. (2024), and OmniRe Chen et al. (2025c). Each model is trained on the designated
context frames for 30,000 iterations per validation scene. In the feed-forward category, we compare our work
against the state-of-the-art DrivingForward Tian et al. (2025) approach. To ensure a fair comparison under
similar settings, we provide multiple frames as input to this baseline. All methods are evaluated at a resolution
of 518 × 280 pixels, with detailed results provided in Table 1 and Table 2.
4.2
Main Experiment Results
Implementation Details.
We set the training-loss hyperparameters to λpercep = 0.05, λl2 = 1.0, λl1 = 0.85,
λssim = 0.15, and λscale = λopacity = 0.01. The model is optimized with AdamW using gradient accumulation
(8 steps). Training proceeds in two stages: (1) Single-frame pre-training for 10 epochs (batch size 4, learning
rate 2 × 10−5, weight decay 0.01), where each input frame Ti is also used as ground truth; (2) Dual-frame
fine-tuning for 2 epochs (batch size 2). Dynamic-object masks and motion are extracted for five vehicle
classes (car, truck, bus, trailer, construction vehicle). All experiments run on H800 GPUs, requiring
8
GPU-days. Evaluation results are reported in Tab. 1 and Tab. 2. Figure 3 presents visual comparisons of scene
reconstruction and novel-view synthesis using Scene-0103 from the nuScenes validation set. These qualitative
8

<!-- page 9 -->
results demonstrate ReconDrive’s ability to achieve high-fidelity rendering while maintaining precise spatial
alignment during view synthesis.
Visual Scene Reconstruction.
As shown in Tab. 1, per-scene optimization methods exhibit strong perfor-
mance due to their iterative parameter tailoring. Specifically, PVG Chen et al. (2023) and OminiRe Chen
et al. (2025c) achieve the highest PSNR (29.58) and the highest SSIM (0.8853) respectively, while De-
formableGS Yang et al. (2024) lead in LPIPS. In contrast, existing feed-forward baselines Tian et al. (2025)
lag significantly (e.g., PSNR 22.83 for Drivingforward), highlighting the traditional performance gap between
generalizable and scene-specific reconstruction. ReconDrive (Ours) effectively bridges this gap. It outperforms
all baselines in pixel, structural and perceptual metrics, achieving a superior PSNR of 32.66 and SSIM of
0.9589 and LPIPS of 0.0618.
Novel-View Synthesis Evaluation.
As shown in Tab. 1, per-scene optimization methods generally
outperform prior feed-forward baselines in novel-view synthesis, as iterative parameter tuning better captures
complex view-dependent variations. For instance, DeformableGS Yang et al. (2024) and Street Gaussians Yan
et al. (2024) achieve PSNRs of 23.73 and 22.98, respectively, while the feed-forward Drivingforward Tian
et al. (2025) reaches only 21.88. ReconDrive effectively bridges the performance gap between efficiency and
quality, establishing a new state-of-the-art for feed-forward 4DGS. While maintaining a comparable perceptual
quality (LPIPS) to the top-performing baseline, ReconDrive notably surpasses DeformableGS Yang et al.
(2024) by 0.26 dB in PSNR and 0.0315 in SSIM.. Furthermore, ReconDrive demonstrates substantial gains
over existing feed-forward approaches, exceeding Drivingforward by 2.11 dB in PSNR and 0.0368 in SSIM,
while significantly reducing perceptual error (LPIPS) by 0.0388. These results underscore that ReconDrive
not only eliminates the traditional trade-off between inference speed and synthesis quality but also provides
superior structural and perceptual consistency for novel-view generation in complex urban scenes.
Method
Detection
Tracking
mAP (%) ↑
AMOTA (%) ↑
Per-Scene Optimization Methods
Street Gaussians Yan et al. (2024)
14.6
11.9
PVG Chen et al. (2023)
18.5
14.4
DeformableGS Yang et al. (2024)
16.4
13.4
OminiRe Chen et al. (2025c)
16.1
12.9
Feed-Forward Methods
Drivingforward Tian et al. (2025)
23.4
13.3
ReconDrive (Ours)
26.7
18.9
Table 2 3D Perception Results on Novel-View Ren-
dered Scenes with 0 m, ±1 m, ±2 m, and ±3 m View
Offsets.
3D Perception Evaluation.
Analysis of the
3D perception results in Table 2 reveals that
per scene optimization methods, such as Street
Gaussians Yan et al. (2024), exhibit limited effi-
cacy with mAP ≤14.6% and AMOTA ≤11.9%.
While feed forward approaches generally excel in
detection—evidenced by DrivingForward Tian
et al. (2025) achieving an mAP of 23.4%—they
often fall short in tracking consistency, as seen in
the AMOTA of 13.3% for DrivingForward com-
pared to 14.4% for PVG Chen et al. (2023). In
contrast, ReconDrive achieves state of the art
performance across all metrics, reaching 26.7%
mAP and 18.9% AMOTA. These results underscore ReconDrive’s robust 3D perception capabilities, demon-
strating superior detection accuracy and tracking consistency even when subjected to novel view offsets.
Inference Efficiency.
To evaluate scalability, we calculate the average time cost for Gaussian generation per
scene (≈20 s duration). In ReconDrive, as adjacent temporal segments share context frames, we use caching
mechanism to eliminate redundant computation during inference. With this optimization, ReconDrive achieves
an inference speed of 15s per scene—orders of magnitude faster than per-scene optimization methods Yan et al.
(2024); Chen et al. (2023), which require about 30 minutes. While slightly slower than existing feed-forward
baselines Tian et al. (2025) (5s), ReconDrive offers a superior balance between high-fidelity reconstruction
and computational efficiency.
5
Conclusion
Developing efficient and scalable scene reconstruction methodologies is a fundamental requirement for building
closed-loop evaluation and simulation platforms within the autonomous driving industry. In this paper, we
9

<!-- page 10 -->
introduced ReconDrive, a feed-forward framework for efficient 4D Gaussian Splatting generation specifically
tailored for large-scale driving scenes. Unlike existing per-scene optimization methods that require extensive
computational resources, our model generates a complete scene representation in a single forward pass. This
efficiency is achieved through a static-dynamic composition strategy and segment-wise Gaussian aggregation,
which effectively adapt a 3D foundation model via our specialized Gaussian prediction architecture.
Experimental results on our new nuScenes-based benchmark demonstrate that ReconDrive outperforms
state-of-the-art per-scene optimization baselines across nearly all evaluation metrics. To the best of our
knowledge, this work represents the first instance where a feed-forward approach surpasses optimization-based
methods in reconstruction and novel-view synthesis performance through comprehensive experimentation.
These findings provide a clear direction and renewed confidence for the development of scalable, generative
simulation environments for large-scale autonomous driving.
Limitations and Future Work.
This study represents an initial effort toward achieving large-scale
autonomous driving scene reconstruction and novel-view synthesis via feed-forward 3D foundation models.
While our results demonstrate significant potential for this paradigm, several challenges remain to be addressed
in future research.
• Temporal Representation of Non-Rigid Motion: ReconDrive currently utilizes an explicit 4D Gaussian
Splatting representation based on linear motion estimation. While efficient, this formulation may struggle
to accurately represent complex non-rigid deformations or highly non-linear object trajectories. Future
work could involve integrating more expressive temporal kernels or learnable deformation fields to better
capture these dynamics.
• Temporal Consistency and Redundancy in Aggregation: Our current post-processing strategy for multi-
frame aggregation relies on pixel-wise outputs that may lead to Gaussian redundancy and suboptimal
handling of occluded regions. Transitioning toward a more integrated temporal fusion mechanism within
the latent space could reduce redundancy and improve the structural integrity of occluded surfaces.
• Precision in Dynamic Object Extraction: The current pipeline relies on SAM2 for object segmentation,
which can occasionally lead to inaccurate boundaries or missed detections. Furthermore, displacing dynamic
objects directly can leave disocclusion artifacts in the background. Investigating joint inpainting techniques
or end-to-end motion modeling could help mitigate these "background holes" and improve boundary fidelity.
• Computational Efficiency and Throughput: Although our feed-forward approach is orders of magnitude
faster than optimization-based methods, there remains substantial room for architectural optimization.
Researching lightweight foundation backbones and more efficient Gaussian sampling strategies will be
critical for achieving real-time performance in edge-computing environments.
• Scalability and Generalization: Expanding the training scope to encompass larger, more diverse datasets
is essential. Evaluating the model across a broader range of geographic locations and extreme weather
conditions will be a primary focus to ensure the robustness and generalizability of the feed-forward paradigm.
Broadening data diversity to cover different driving behaviors may further enhance robustness across varied
real-world scenarios Hao et al. (2025).
10

<!-- page 11 -->
References
Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan,
Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631, 2020.
Xiaoxue Chen, Ziyi Xiong, Yuantao Chen, Gen Li, Nan Wang, Hongcheng Luo, Long Chen, Haiyang Sun, Bing Wang,
Guang Chen, et al. Dggt: Feedforward 4d reconstruction of dynamic driving scenes using unposed images. arXiv
preprint arXiv:2512.03004, 2025a.
Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo,
Thibaut Hardin, Xiang Li, et al. Sam 3d: 3dfy anything in images. arXiv preprint arXiv:2511.16624, 2025b.
Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and
Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on
Computer Vision, pp. 370–386. Springer, 2024.
Yurui Chen, Chun Gu, Junzhe Jiang, Xiatian Zhu, and Li Zhang. Periodic vibration gaussian: Dynamic urban scene
reconstruction and real-time rendering. arXiv preprint arXiv:2311.18561, 2023.
Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio, Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan
Gojcic, Sanja Fidler, Marco Pavone, Li Song, and Yue Wang. Omnire: Omni urban scene reconstruction. In The
Thirteenth International Conference on Learning Representations, 2025c.
Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving
simulator. In Conference on robot learning, pp. 1–16. PMLR, 2017.
Ruiyang Hao, Bowen Jing, Haibao Yu, and Zaiqing Nie. Styledrive: Towards driving-style aware benchmarking of
end-to-end autonomous driving. arXiv preprint arXiv:2506.23982, 2025.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.
Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai
Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 17853–17862, 2023.
Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer,
and Shanghang Zhang. S3 Gaussian: Self-supervised street gaussians for autonomous driving. arXiv preprint
arXiv:2405.20323, 2024.
Xiaosong Jia, Yanhao Liu, Junqi You, Renqiu Xia, Yu Hong, and Junchi Yan. Drivevggt: Visual geometry transformer
for autonomous driving. arXiv preprint arXiv:2511.22264, 2025.
Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang,
Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on
Graphics (TOG), 44(6):1–16, 2025.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
Mustafa Khan, Hamidreza Fazlali, Dhruv Sharma, Tongtong Cao, Dongfeng Bai, Yuan Ren, and Bingbing Liu.
Autosplat: Constrained gaussian splatting for autonomous driving scene reconstruction. In 2025 IEEE International
Conference on Robotics and Automation (ICRA), pp. 8315–8321. IEEE, 2025.
Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang,
Tiancai Wang, et al. Uniscene: Unified occupancy-centric driving scene generation. In Proceedings of the Computer
Vision and Pattern Recognition Conference, pp. 11971–11981, 2025.
Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth
anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.
11

<!-- page 12 -->
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without
supervision. arXiv preprint arXiv:2304.07193, 2023.
René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the
IEEE/CVF international conference on computer vision, pp. 12179–12188, 2021.
Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint
arXiv:2408.00714, 2024.
You Shen, Zhipeng Zhang, Yansong Qu, Xiawu Zheng, Jiayi Ji, Shengchuan Zhang, and Liujuan Cao. Fastvggt:
Training-free acceleration of visual geometry transformer. arXiv preprint arXiv:2509.02560, 2025.
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv
preprint arXiv:1409.1556, 2014.
Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2446–2454, 2020.
Qijian Tian, Xin Tan, Yuan Xie, and Lizhuang Ma. Drivingforward: Feed-forward 3d gaussian splatting for driving
scene reconstruction from flexible surround-view input. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 39, pp. 7374–7382, 2025.
Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt:
Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 5294–5306, 2025.
Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made
easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20697–20709,
2024.
Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang
Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp. 20310–20320, 2024.
Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and
Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In European Conference on
Computer Vision, pp. 156–173. Springer, 2024.
Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi Li, Yurong You, Apoorva Sharma, Maximilian Igl, Peter
Karkus, Danfei Xu, et al. Storm: Spatio-temporal reconstruction model for large-scale outdoor scenes. International
Conference on Learning Representations, 2025.
Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for
high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 20331–20341, 2024.
Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian:
Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pp. 21634–21643, 2024.
Ziyue Zhu, Zhanqian Wu, Zhenxin Zhu, Lijun Zhou, Haiyang Sun, Bing Wan, Kun Ma, Guang Chen, Hangjun Ye, Jin
Xie, et al. Worldsplat: Gaussian-centric feed-forward 4d scene generation for autonomous driving. arXiv preprint
arXiv:2509.23402, 2025.
Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer.
arXiv preprint arXiv:2507.11539, 2025.
12

<!-- page 13 -->
Appendix
This appendix is organized as follows: Section A provides the full list of validation scenes.
Section B
contains additional implementation details. Section C presents an ablation study on the spatial misalignment
encountered when directly using VGGT Wang et al. (2025) for 3D geometry generation. Section D presents
an ablation study evaluating how multi-frame inputs enhance the capability for novel-view synthesis. Finally,
Section E provides visualizations of dynamic Gaussians, and Section F offers extensive qualitative comparisons
of urban scene reconstruction and novel-view synthesis against baseline methods.
A
Validation Scenes List
In our experiment, we selected 14 representative scenes from the original nuScenes dataset’s Caesar et al.
(2020) validation split as the validation subset. Covering diverse conditions in time (day/night), weather
(sunny/rainy), driving behaviors (static/straight/turning), and traffic density, these scenes effectively evaluate
the novel-view synthesis ability for reconstruction while reducing the evaluation cost vs. the full validation
split. The 14 selected scenes are: scene-0014, scene-0018, scene-0098, scene-0100, scene-0103, scene-0270,
scene-0271, scene-0278, scene-0553, scene-0558, scene-0802, scene-0906, scene-0968, scene-1065.
B
Implementation Details
Architecture.
We resize the original input images from the nuScenes dataset Caesar et al. (2020) from
a resolution of 1600 × 900 × 3 to 518 × 280 × 3. Following VGGT Wang et al. (2025), we adopt a 14 × 14
patch size and retain the camera token along with four register tokens in the feature backbone. Specifically,
we feed the tokens from the 4-th, 11-th, 17-th, and 23-rd blocks into the Gaussian Center Prediction Head
(GCPH) and Gaussian Parameter Prediction Head (GPPH), while excluding the camera token and four
register tokens from the head inputs. For GCPH, we clamp outlier depth values to restrict the depth maps to
a range of 1.5m (minimum) to 110m (maximum). For GPPH, we upsample the features to the resized image
resolution (consistent with GCPH), process the resized image using a convolutional and ReLU layer, and then
concatenate the resulting features with the upsampled features.
Training Loss (Project Loss).
The project loss enforces photometric consistency between a warped source
image and the target image, leveraging a two-stage pipeline to ensure geometric validity and appearance
alignment for novel-view synthesis in urban scenes. Process is also illustrated in Fig. 4.
Start with the Target Image It (captured at time t) and its corresponding predicted depth map DPt. First,
backproject 2D pixels in It to 3D points in the target camera frame using the inverse camera intrinsics K−1
t
and DPt. For a pixel (u, v) in It, the 3D point Pt is computed as:
Pt = K−1
t
·


u · Dt(u, v)
v · Dt(u, v)
Dt(u, v)


Next, apply ego-motion transformation to convert Pt from the target frame to the source frame using the
6-DoF pose T(t →s) (rigid transformation matrix), yielding Ps = T(t →s) · Pt. Finally, perform projection
+ normalization to project Ps back to 2D pixel coordinates in the source camera view using the source
camera intrinsics Ks, then normalize the coordinates to the range [−1, 1] (required for grid sampling) to get
pix_coords.
Use pix_coords from the geometric pipeline to warp the Source Image Is via the grid_sample operation
(bilinear interpolation) in deep learning frameworks, generating the Warped Image I′
s. Concurrently, generate
a Warped Mask Ms to filter invalid regions:
Ms =
(
1
if pix_coords ∈[−1, 1]2 and non-NaN
0
otherwise
13

<!-- page 14 -->
Figure 4 Project Loss Computing Pipeline. Note that Kt and Ks are identical since the camera is fixed.
This mask excludes pixels projected outside the source image bounds or occluded during transformation.
The Project Loss is defined as the masked L1 loss between the Warped Image I′
s and the Target Image It,
ensuring only valid pixels contribute to training:
1
P Ms + ϵ
X
u,v
|I′
s(u, v) · Ms(u, v) −It(u, v) · Ms(u, v)|
where ϵ = 10−8 is a small constant to avoid division by zero, and P Ms is the number of valid pixels in
the Warped Mask. Element-wise multiplication with Ms suppresses gradients from invalid/occluded regions,
enforcing geometric and photometric alignment. The final project loss combines a weighted L1 loss (measuring
pixel-wise intensity differences between warped predicted images and warped ground-truth images) and a
weighted SSIM loss (evaluating structural and perceptual consistency) as follows.
Lproject = Lmasked (λl1 · Ll1 + λssim · Lssim)
(9)
Training.
To form training batches, we segment each scene into consecutive 6-frame clips, each covering
a 0.5-second duration, with the 1st and 6th frames serving as context frames. During training, for Project
Loss computation, we designate the 1st frame as the target frame It and the subsequent frame (i.e., the 2nd
frame) as the source frame Is. Additionally, we render the 4D Gaussians into different time frames, including
novel-time frames (spanning from the 1st to the 5th frame), with corresponding sampling probabilities of [0.7,
0.3, 0.2, 0.1, 0.1, 0.05].
C
Ablation Study on Spatial Misalignment with Original VGGT
In this section, we analyze the 3D geometric accuracy when directly employing the original VGGT model to
process multi-view imagery. Specifically, we input a single frame consisting of six-view images into the original
VGGT to estimate camera poses, depth maps, and point maps. We then evaluate the spatial alignment by
calculating the distance distribution between the point maps generated by VGGT and the ground truth
LiDAR point clouds from the nuScenes dataset. The resulting distribution is illustrated in Fig. 5.
Our results reveal a significant scale misalignment between the original VGGT point map outputs and the
LiDAR ground truths. This suggests that without domain-specific adaptation, the original VGGT cannot
accurately reconstruct the scale of complex urban environments. However, as shown in the visualization in
Fig. 6, we observe that when a global scale factor is applied to the VGGT point maps, they align closely with
the LiDAR data. This indicates that while the foundation model successfully captures the structural geometry
of the scene, it fails to recover the absolute metric scale. These findings further underscore the necessity of
incorporating pre-calibrated sensor parameters into the Gaussian prediction heads to build metrically accurate
4D Gaussian representations.
14

<!-- page 15 -->
0
10
20
30
40
50
x2 + y2 + z2
0.0
0.2
0.4
0.6
0.8
1.0
Density
Radius Distribution Comparison 
LiDAR
VGGT
Figure 5 Distribution of Spatial Distances between the Point Maps Generated by the Original VGGT
and Ground Truth LiDAR Point Clouds, highlighting significant metric misalignment.
(a)
(b)
(c)
Figure 6
Qualitative Comparison of VGGT-Generated Point Maps Across Various Scale Factors,
demonstrating the structural consistency despite metric scale ambiguity.
D
Ablation Study for Temporal Inputs
Method
Novel-View Synthesis
PSNR ↑
SSIM ↑
LPIPS ↓
ReconDrive-S
23.54
0.6940
0.3177
ReconDrive
23.99 (+0.45)
0.7234 (+0.0294)
0.2591 (-0.058)
Table 3
Evaluation Results with Different Frame
Input.
To analyze how multi-frame inputs improve
novel-view synthesis performance, we compare
ReconDrive-S (single-frame input) against Re-
conDrive (two-frame input). As shown in Table 3,
ReconDrive achieves a PSNR of 23.99, an SSIM
of 0.7234, and an LPIPS of 0.2591, significantly
outperforming ReconDrive-S across all metrics.
This demonstrates that multi-frame fusion en-
hances synthesis quality because leveraging multiple frames expands the reconstructed visual field and provides
complementary viewpoint information. Consequently, this leads to substantially higher fidelity and perceptual
quality in the rendered outputs.
E
Dynamic Gaussian Rendering Visualization
In Fig. 7, we render the dynamic Gaussians as time progresses, masking out other static Gaussians. The
resulting positions of the moving vehicles align well with the ground truth images, illustrating the effectiveness
of our dynamic object mask and motion estimation for dynamic scene synthesis.
15

<!-- page 16 -->
Figure 7 Dynamic Gaussian Rendering Visualization. The rendered dynamic vehicles exhibit accurate motion
and align closely with their counterparts in the ground truth images. Note that we have enhanced the exposure of the
rendered images for visualization purposes to facilitate comparative observation.
F
Visualization Comparisons with Diverse Methods
To further demonstrate the effectiveness of our ReconDrive framework, we conduct extra visualization
comparisons with five state-of-the-art methods, including Street Gaussians Yan et al. (2024), PVG Chen
et al. (2023), DeformableGS Yang et al. (2024), OminiRe Chen et al. (2025c), and DrivingForward Tian et al.
(2025)—under both Multi-Frame (MF) and Single-Frame (SF) modes across the selected validation scenes.
The results (Figs. 9–10) demonstrate ReconDrive’s superiority in both scene reconstruction and novel-view
synthesis.
16

<!-- page 17 -->
Figure 8 Visual Comparisons of Scene Reconstruction and Novel-View Synthesis on Scene-0098 (from
14 Selected Validation Scenes). Consistent with prior results, our ReconDrive maintains high-quality rendering in
both scene reconstruction (Original View) and novel-view synthesis (1–3m lateral movement) compared to per-scene
optimization methods (Street Gaussians, PVG, DeformableGS, OminiRe).
Figure 9 Visual Comparisons of Scene Reconstruction and Novel-View Synthesis on Scene-0968 (from
14 Selected Validation Scenes). Consistent with prior results, our ReconDrive maintains high-quality rendering in
both scene reconstruction (Original View) and novel-view synthesis (1–3m lateral movement) compared to per-scene
optimization methods (Street Gaussians, PVG, DeformableGS, OminiRe). Against the feed-forward DrivingForward,
it preserves precise visual consistency—clear in tree near the car —with more notable performance gains as movement
increases, plus less distortion/blurriness, especially at image boundaries.
17

<!-- page 18 -->
Figure 10 Visual Comparisons of Scene Reconstruction and Novel-View Synthesis on Scene-1065 (from
14 Selected Validation Scenes). Consistent with prior results, our ReconDrive maintains high-quality rendering in
both scene reconstruction (Original View) and novel-view synthesis (1–3m lateral movement) compared to per-scene
optimization methods (Street Gaussians, PVG, DeformableGS, OminiRe). Against the feed-forward DrivingForward
(MF/SF modes), it preserves precise visual consistency. This visualization further demonstrates that ReconDrive
performs exceptionally well in critical scenarios like nighttime environments.
18
