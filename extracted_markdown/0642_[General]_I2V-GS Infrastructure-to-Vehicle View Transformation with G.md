<!-- page 1 -->
I2V-GS: Infrastructure-to-Vehicle View Transformation with Gaussian Splatting
for Autonomous Driving Data Generation
Jialei Chen1
Wuhao Xu2
Sipeng He3
Baoru Huang4
Dongchun Ren1
1Yootta
2Soochow University
3Southeast University
4University of Liverpool
Abstract
Vast and high-quality data are essential for end-to-end au-
tonomous driving systems. However, current driving data
is mainly collected by vehicles, which is expensive and in-
efficient. A potential solution lies in synthesizing data from
real-world images. Recent advancements in 3D reconstruc-
tion demonstrate photorealistic novel view synthesis, high-
lighting the potential of generating driving data from im-
ages captured on the road. This paper introduces a novel
method, I2V-GS, to transfer the Infrastructure view To the
Vehicle view with Gaussian Splatting. Reconstruction from
sparse infrastructure viewpoints and rendering under large
view transformations is a challenging problem. We adopt
the adaptive depth warp to generate dense training views.
To further expand the range of views, we employ a cas-
cade strategy to inpaint warped images, which also en-
sures inpainting content is consistent across views. To fur-
ther ensure the reliability of the diffusion model, we uti-
lize the cross-view information to perform a confidence-
guided optimization. Moreover, we introduce RoadSight,
a multi-modality, multi-view dataset from real scenarios
in infrastructure views. To our knowledge, I2V-GS is the
first framework to generate autonomous driving datasets
with infrastructure-vehicle view transformation.
Experi-
mental results demonstrate that I2V-GS significantly im-
proves synthesis quality under vehicle view, outperforming
StreetGaussian in NTA-Iou, NTL-Iou, and FID by 45.7%,
34.2%, and 14.9%, respectively.
1. Introduction
In recent years, there has been great progress in end-to-end
autonomous driving systems [7, 8], which convert sensor in-
puts to control signals directly. However, one of the major
challenges for end-to-end autonomous driving is the need
for vast training data to achieve reliable performance [2].
These large datasets are essential for training models capa-
ble of handling complex and dynamic environments.
Currently,
the primary methods for acquiring au-
tonomous driving data can be divided into dedicated data
collection fleets, production vehicle fleets, and synthetic
datasets [2]. Nevertheless, these methods face significant
challenges.
Data collection via dedicated fleets, such as
the Waymo [21] and nuScenes [1] datasets, provides real-
istic environmental data, but it is relatively expensive due to
high operational costs associated with the vehicles, sensors,
and safety drivers. On the other hand, production vehicle
fleets, like Tesla’s, generate vast amounts of real-world driv-
ing data but face problems related to data privacy and high
data transmission costs [27].
These limitations have led
to the increasing adoption of dataset synthesis techniques,
which offer a cost-effective and efficient alternative while
capable of producing diverse scenarios. Synthetic datasets
typically can be divided into utilizing game engines for ren-
dering data [13, 20], generating driving scenarios with gen-
erative models [3, 4, 6, 22, 25, 26, 31], and viewpoint trans-
formations or adding additional objects based on real-world
dataset [9, 29, 34, 37, 38]. However, transitioning from
synthetic data to real-world applications presents signifi-
cant challenges, as algorithms must effectively bridge the
domain gap to ensure their performance remains robust in
real-world environments [14]. Synthesis datasets from im-
ages captured in real-world [34, 37] are constrained by the
size of the original collected data, limiting their scalability.
Considering the efficiency and validity, collecting infor-
mation from infrastructure sensors and then transforming
into vehicle views for autonomous driving system train-
ing is more effective and reliable, as illustrated in Fig. 1.
Given roadside cameras capture images continuously, this
approach enables the synthesis of a virtually unlimited num-
ber of datasets, improving the efficiency of data collection.
Nevertheless, in infrastructure-to-vehicle (I2V) view
transformation tasks, sparse viewpoints and large view
transformation pose challenges in rendering novel vehicle
views. Previous autonomous scene reconstruction methods
[9, 29, 38] leverage the motion of vehicles to obtain multi-
view images. However, this approach is not applicable in
the sparse and fixed viewpoints setting. Sparse view recon-
struction methods [12, 28] introduce depth prior as a con-
arXiv:2507.23683v1  [cs.CV]  31 Jul 2025

<!-- page 2 -->
Figure 1. We propose transferring images captured by infrastructure to vehicle views for autonomous driving system training to reduce the
cost of data collection (left). Previous Gaussian Splatting methods face challenges in synthesizing vehicle views. In contrast, our approach
significantly improves the image quality (right).
straint. However, they fail to render high-quality images
under large view transformation.
To address these challenges, we propose I2V-GS. Our
approach first calibrates monocular depth with LiDAR and
employs the adaptive depth warp to provide a dense training
view. To further expand the range of views, we utilize the
diffusion model to inpaint holes in warped images, where a
cascade strategy is adopted to ensure the consistency of in-
painting content across views. To further improve the relia-
bility of the diffusion model, we utilize cross-view informa-
tion to assess the inpaint content to carry out a confidence-
guided optimization for pseudo views. Moreover, we intro-
duce RoadSight, a multi-modality, multi-view dataset from
real scenarios for I2V view transformation. To our knowl-
edge, I2V-GS is the first framework to generate autonomous
driving datasets with I2V view transformation. As shown
in Fig. 1, our approach enhances the novel view synthesis
quality in vehicle view, achieving a relative improvement in
NTA-Iou, NTL-Iou, and FID by 45.7%, 34.2%, and 14.9%,
respectively, comparing with StreetGaussian [29].
The main contributions of this work are as follows:
• We present I2V-GS, the first framework that generates
autonomous driving datasets with infrastructure-vehicle
view transformation.
• We propose the adaptive depth warp to generate dense
training views, enabling rendering high-quality images
under sparse view input and large view transformation
settings.
• We introduce the cascade diffusion strategy to guarantee
content consistency among pseudo views and leverage
cross-view information in confidence-guided optimiza-
tion for reliable inpaint content.
2. Related Work
2.1. Novel View Synthesis from Sparse View
Recent advances in 3D Gaussian Splatting (3DGS) have
sought to address the sparse-view reconstruction challenge
through two primary paradigms: depth-prior supervision
and diffusion-based refinement.
Depth-guided methods
leverage geometric priors to compensate for insufficient
multi-view constraints. Works like DNGaussian [12] em-
ploy monocular depth estimators [32] to constrain Gaussian
positions. While these approaches mitigate floaters and im-
prove surface coherence, their reliance on scale-ambiguous
monocular predictions can lead Gaussians to distribute to
suboptimal positions.
Diffusion-based methods leverage
generative models to predict missing details. Deceptive-
NeRF [15] pioneers this direction by iteratively refining
neural radiance fields using diffusion-model to refine ren-
dered novel views.
Subsequent 3DGS adaptations like
SparseGS [28] apply score distillation sampling (SDS) [18]
to align Gaussian renderings with diffusion priors. While
effective for detail synthesis, these methods suffer from
content inconsistency in unseen areas.
2.2. Driving Scene Synthesis
Reconstruction-based Method. Early methods [23, 24]
utilize neural radiance fields (NeRF) [16] to reconstruct the
driving scene. Though these methods achieve high-quality
rendering results, they suffer from long training and infer-
ence time. Recently, 3DGS [10] introduces an efficient pro-
cess, that represents scenes with a set of anisotropic Gaus-
sians and achieves high-quality rendering from sparse point
cloud inputs with adaptive density control. Several works
[9, 29, 29, 38] extend 3DGS to model driving scenes by
decomposing the static background and dynamic objects.
However, These methods can only render interpolate views,
where sensor data closely matches the training data distri-
bution, which is inadequate for training autonomous driv-

<!-- page 3 -->
ing models. More recently, some works [17, 30, 34, 37]
propose to adapt the diffusion model to reconstruct driv-
ing scenes and generate extrapolated views. However, these
methods rely on the motion of vehicles to obtain multi-view
images, limiting their applicability in I2V transformation
tasks where viewpoints are fixed.
Generative-based Method. Recently, generative mod-
els have shown significant potential in generating unseen
and future views based on the current frame. Many works
[3, 4, 6, 22, 25, 26, 31] extent video diffusion models into
autonomous driving to simulate different driving scenarios.
Though generating many scenarios, they fail to capture the
underlying 3D model, leading to inconsistent geometry and
texture in the generated videos.
3. Preliminary: 3D Gaussian Splatting
3DGS [10] represents scenes with a set of differentiable
3D Gaussians. Each 3D Gaussians consists of learnable
attributes: position µ, rotation r, scaling s, opacity o, and
spherical harmonic (SH) coefficients. Formally, the impact
of a 3D Gaussian on location x is defined by the Gaussian
distribution:
G(x) = exp

−1
2(x −µ)TΣ−1(x −µ)

,
(1)
where Σ is the covariance matrix, which can be decomposed
into Σ = RSSTRT, R ∈R4 is a rotation matrix expressed
with quaternions, and S ∈R3 is a diagonal scaling ma-
trix. The 3D Gaussian is projected onto the 2D image planes
for rendering, where the projected 2D Gaussian is sorted by
its depth value. The final rendering equation for the color
ˆC(X) of each pixel X is:
ˆC(X) =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αi),
(2)
αi = oiG2D
i
(p),
(3)
where ci is the color defined from the SH coefficients and
αi is the density calculated by multiplying the projected 3D
Gaussian with the opacity oi. The covariance matrix after
projection is calculated by Σ′ = JWΣW T JT , where J is
the Jacobian of the affine approximation of the projective
transformation and W represents the viewing transforma-
tion matrix.
The 3D Gaussians G is optimized by the combination of
RGB loss, depth loss, and SSIM loss:
Lori = λ·∥ˆI −I∥1+(1−λ)·SSIM(ˆI, I)+∥ˆD−D∥1, (4)
where ˆI and I refer to rendered and ground truth image, ˆD
and D represent rendered and ground truth depth, SSIM(·)
is the operation of the Structural Similarity Index Measure,
and λ is the loss weight.
4. I2V-GS
As shown in Fig. 3, in the I2V view transformation task,
sparse viewpoints and large view transformation cause the
rendering of novel vehicle views difficult. To address these
challenges, we propose a novel framework, I2V-GS. As is
shown in Fig. 2, we first warm up the Gaussian optimiza-
tion with sparse input views. Then, we present a cascade
pseudo view generation method to provide dense training
views. Specifically, we utilize LiDAR to calibrate monoc-
ular depth to provide a real depth and propose the adaptive
depth warp to generate proper pseudo views (Sec. 4.1 and
4.2). The cascade strategy is employed to inpaint holes in
warped images iteratively to guarantee content consistency
and enable a wide range of training views (Sec. 4.3). To
ensure the inpaint content aligns with the real world, the
cross-view information is applied to assess the inpaint con-
tent in optimization (Sec. 4.4).
4.1. Adaptive Depth Warp
To provide dense training views, we generate pseudo views
V′ around the input views V via forward warping ψ. Specif-
ically, we project rendered 3D points p = (x, y, z)T under
viewpoint Vi to a novel view through:
p′ = KR′R−1K−1p + K(T ′ −R′R−1T)
(5)
where p is from the depth map obtain in Sec. 4.2, p′, R′,
and T ′ are the target view’s 3D points, rotation matrix, and
translation vector, respectively. K is camera intrinsic, T
is the source view’s translation, and R is the source view’s
rotation. Then, p′ can be projected onto the image plane
with:
1
||wnorm||


u
v
1

= K
R
T
0
1





x
y
z
1




(6)
where ||wnorm|| is used to normalize the homogeneous co-
ordinate and (u, v) is the pixel coordinate. Then, (u, v) are
rasterized into Iwarp, Dwarp, and Mwarp via z-buffering
and bilinear sampling.
Directly applying depth warping with fixed displace-
ment and rotations often leads to two critical failure cases:
1) over-warping from excessive displacements that amplify
geometric errors, causing distorted artifacts, and 2) under-
warping from insufficient displacements that limit view-
point variation.
To balance this trade-off, we introduce
an adaptive depth warp strategy constrained by pixel-level
spatial consistency. For 3D points observed in the source
view V, let (∆u, ∆v)⊤denote the pixel displacement in the
pseudo-view V′. Replacing (x, y, z, 1)⊤in Eq. 6 with p and
p′, we express the displacement relationship as follows:

<!-- page 4 -->
Adaptive
Warp
Diffusion 
Model
Gaussian Optimization
Cascade Strategy
Pseudo View Generation
•  •  •  
•  •  •  
Rendered Image & Depth
Novel Pseudo View Warped Image
Refined Warped Image
Input View GT Images
Pseudo View Rendered Image
Confidence Weight
 Training Views at Iteration Ti+2
 Training Views at Iteration Ti+1
 Training Views at Iteration Ti
Confidence-Guided Optimize
Rasterizer
Input View
Pseudo View
•  •  •  
•  •  •  
Prompt: a realistic traffic 
intersection
Add Noise
Figure 2. The overall framework of I2V-GS. Initially, the input views warm up the Gaussian optimization. Then, given the rendered
image and depth of the current training view, the adaptive depth warp is employed to generate a novel pseudo view warped image, and
the diffusion model is leveraged to inpaint unseen areas. Subsequently, the refined warped image would be added to training views and
leverage the cross-view information to perform a confidence-guide optimization. Furthermore, we employ a cascade strategy to generate
pseudo views progressively to ensure content consistency.
∆u
∆v

=
1
z(z + ∆tz)
fx∆txz + cx∆tzz −∆tzx
fy∆tyz + cy∆tzz −∆tzy

(7)
where ∆t = (∆tx, ∆ty, ∆tz)⊤= T ′ −T represents the
relative translation between views, and fx, fy, cx , cy are
camera intrinsics. Given a pre-defined warp difference ε,
we can formulate the equation as:

∆u
∆v

∞
≤ε
(8)
Solving Eq. 8 involves parameters ∆tx, ∆ty, ∆tz, z, and
ε. For practical optimization, we first set ∆tz = 0 to de-
couple horizontal/vertical shifts, which will also decouple
x and y in Eq. 7. After that, z is set to the minimum scene
depth zmin as a conservative estimate. This yields simpli-
fied bounds as below, and the depth warp can be controlled
by giving a proper ε (see appendix for more details):





∆tx ≤εzmin
fx
∆ty ≤εzmin
fy
(9)
4.2. LiDAR-Anchored Monocular Depth Calibra-
tion
The performance of depth warp critically depends on
the accuracy of scene geometry estimation. Nevertheless,
3DGS [10] tends to generate imprecise geometries due to
insufficient multi-view constraints in sparse view settings.
To tackle this problem, we propose the monocular depth
with LiDAR prior, which aligns monocular predictions with
accurate LiDAR measurements, to provide precise geome-
try guidance.
Standard monocular depth estimation converts disparity
dmono to depth Dmono via:
Dmono = btrain · ftrain
dmono
(10)
where btrain and ftrain are domain-specific parameters tied
to the training data distribution, e.g. forward-facing vehicle
cameras in Waymo [21].
However, infrastructure images differ significantly from
the training data distribution, which may cause disparity
prediction bias and depth offsets when applying Eq. 10. To
address this misalignment, we propose a generalized formu-
lation:
Dalign =
c1
dmono + c2
+ c3
(11)
where c1 corresponds to the product b·f in Eq. 10, ensuring

<!-- page 5 -->
Algorithm 1 Cascade Strategy
Require: Initial Gaussian model G0, diffusion model D,
rasterization R, training iterations T, depth warp steps
Tw
for t = 0, ..., T −1 do
if t in Tw then
ˆI′, ˆD′ ←{R(Gt, V′
j−1)}F ′
j=1
Iwarp, Dwarp, Mwarp ←ψ(ˆI′, ˆD′ | V′
j−1, V′
j)
I′, D′, M ′ ←D(Iwarp, Dwarp, Mwarp)
V′
j(gt) ←I′, D′, M ′
end if
ˆIt, ˆDt ←{R(Gt, Vi)}F
i=0
Compute loss L ←Lori,
Backpropagate loss and update Gt+1
ˆI′
t ←{R(Gt, {R(Gt, V′
j)}F ′
j=1
Compute loss L′ ←Lcon(ˆI′
t, I′
gt),
Backpropagate loss and update Gt+1
end for
return GT
consistency with the classical framework. c2 and c3 are used
to rectify potential biases and offsets in the disparity and
depth values.
Then, we leverage the LiDAR depth Dlidar to conduct a
pair sample P =
n
D(i,j)
lidar, d(i,j)
mono

| D(i,j)
lidar ̸= 0
o
to op-
timize parameter c1, c2 and c3, where (i, j) denotes pixel
coordinates. The optimization is processed via nonlinear
least squares with Huber loss LH:
min
c1,c2,c3
X
P
LH

Dlidar −
c1
dmono + c2
−c3

(12)
4.3. Cascade Strategy
One key challenge with depth warp is that it usually in-
troduces occlusion holes in pseudo views. While diffusion
models can inpaint missing regions, their stochastic nature
causes inconsistent content across frames. In this paper, we
propose a cascade strategy. As illustrated in Alg. 1, we first
warm up the model with input sparse views. Then, we carry
out depth warp to provide dense training views. Specifi-
cally, we generate pseudo views in a cascade manner. In
each round generation, the novel pseudo view Vj is based
on the prior pseudo view Vj−1. Following the warp, a latent
diffusion model [19] is employed to inpaint the occluded re-
gions. In this process, both the warped images and their cor-
responding hole masks are first encoded into a latent space
h, where the inpainting operation is performed to yield a
refined latent representation ˆh. The final, refined pseudo
view is obtained by decoding ˆh to an RGB image. This
cascade mechanism enables the propagation of information
from prior views, ensuring the consistency of inpaint con-
tents across views. Moreover, the conventional depth warp
is inherently limited by the accumulation of occlusion holes
that arise from large viewpoint shifts. In contrast, incor-
porating inpainting methods effectively mitigates these oc-
clusions, thereby broadening the operational range of depth
warping and enabling a wider range of training views.
4.4. Confidence-Guided Optimization
While diffusion-based inpainting helps complete occluded
regions, the stochastic generation process may introduce
semantic inconsistencies between pseudo-views and actual
scenes. To mitigate this, we propose a confidence-guided
optimization scheme that leverages multi-view consensus
to weight supervision signals. We adopt the L2 difference
to detect pixel alignment and SSIM to evaluate perceptual
similarity at the patch-structure level. Given the inpainted
image I′ and rendered image ˆI′ under V′
j, the combined
confidence weight is:
W = λ1 ·(1−L2(ˆI′, I′))+(1−λ1)·SSIM(ˆI′, I′) (13)
where W ∈(0, 1) represents the confidence weight, with
values closer to 1 indicating higher confidence. Then, the
confidence-guided loss can be expressed as:
Lcon = E
h
W ·
ˆI′, I′
1
i
(14)
where E(·) is the expectation. Then, the confidence weight
can reduce the impact of the mismatching area while maxi-
mizing the error in other regions.
We utilize the original optimization [10] in Eq. 4 for in-
put views and confidence-guided optimization for pseudo
views. The total loss function is:
L = Lori + Lcon
(15)
5. Experiment
5.1. RoadSight Dataset
Existing autonomous driving datasets mainly rely on
vehicle-mounted sensors, while neglecting the potential of
infrastructure-based perception.
Although [36] and [35]
represent an infrastructure-vehicle cooperative dataset for
improving 3D object detection, they cannot provide cross-
ing views for scene reconstruction. Therefore, we intro-
duce RoadSight, a multi-modality, multi-view dataset from
real scenarios for I2V view transformation. This section de-
scribes the specifications of infrastructure sensors and how
we set up these sensors.
Sensor Specification. The data collection uses solid-
state LiDAR, blind-spot LiDAR, and high-resolution cam-
eras. The details of sensors are listed in Tab. 1. All sensors
are hardware-synchronized via GPS-PPS signals, ensuring

<!-- page 6 -->
Figure 3. The layout of sensors. All sensors are attached to the
traffic lights.
temporal alignment. Fig. 3 posts the specified layout. For
all scenarios, sensors are attached to the traffic lights. The
camera lens angles are roughly tilted down about 3 degrees
and manually adjusted to align with the center of the road.
Besides, an online extrinsic calibration module is developed
to update the sensor poses and maintain the accuracy of data
collection.
Collection and Statistics. RoadSight covers recordings
from 4 urban intersections in Suzhou Autonomous Driving
Demonstration Area, spanning diverse conditions:
• Traffic Density: Peak hours (40%), Off-peak (60%);
• Illumination: Daytime (70%), Night (30%).
After raw data collection, we manually select 50 represen-
tative scenes, each lasting 20 seconds. The videos are sam-
pled at 10Hz and synchronized with LiDAR scans.
Privacy Protection. RoadSight prioritizes ethical data
collection and privacy protection.
The whole dataset is
collected from public roads with government authoriza-
tion.
Besides, we employ professional labeling tools to
anonymize faces and license plates, ensuring the protection
of personal identities.
5.2. Experiment Setup
Dataset. We conduct the experiments on the RoadSight
dataset, where 4 representative scenes are selected. Further-
more, to validate the robustness, we assess our method on
10 selected sequences with surrounding videos and LiDAR
point clouds from Waymo Dataset [21]. For a fair compari-
son, we train the 3DGS-based methods using the first frame
and 4DGS-based methods using the first 10 frames.
Metric.
Following [37], we adopt Novel Trajectory
Agent IoU (NTA-IoU) and Novel Trajectory Lane IoU
(NTL-IoU), which detect vehicles and road lanes in novel
trajectory viewpoints and compare them with ground truth
LiDAR1
LiDAR2
Camera
Manufacturer
Innovusion
Bpearl
Hikivision
Model
FALCON
RS
2CD7U8XJM
Resolution
-
-
1920×1080
Frequency
10 Hz
10 Hz
25 Hz
HFOV
120◦
360◦
89◦
VFOV
25◦
90◦
46.5◦
Range
500 m
30 m
-
Accuracy
±5 cm
±3 cm
-
Table 1. Specification for Solid-State LiDAR (LiDAR1), Blind-
Spot LiDAR (LiDAR2), and Camera. The HFOV and VFOV rep-
resent horizontal and vertical fields of view, respectively.
after projection. Additionally, we utilize the FID [5] to as-
sess the difference in feature distribution between the syn-
thesized novel view and the original view.
Implementation Details.
Our model is trained for
60,000 iterations with the Adam optimizer [11].
We
adopt Depth-Anything [33] as monocular depth estima-
tion model. We initially warm up the optimization with
Gaussian model [10] for 3,000 iterations and then generate
pseudo views with a cascade strategy every 3,000 iterations
for three cycles.
5.3. Comparison with State-of-the-art
Results on RoadSight Dataset. We synthesize novel vehi-
cle views from the captured images. Tab. 2 demonstrates
our improvements on NTA-Iou, NTL-Iou, and FID, outper-
forming StreerGaussian [29] with a 45.7% increase in NTA-
Iou, 34.2% increase in NTL-Iou, and 14.9% in FID. These
enhancements are visually demonstrated in Fig. 4. Our ap-
proach renders high-quality foreground vehicles and back-
ground elements in vehicle views, resulting in more realistic
driving environments. In contrast, baseline methods suffer
from artifacts and blurry results due to the lack of capacity
of their model under large view transformation.
Results on Waymo Dataset [21]. To validate the ro-
bustness of our approach, we conduct experiments on the
Waymo dataset [21], where the novel view is synthesized
along novel trajectories shifting from the recorded trajec-
tories. The results are reported in Tab. 2. Compared to
StreetGaussian [29], our approach achieves 29.7% NTA-Iou
improvement, 4.2% NTL-Iou improvement, and 33.3% re-
duction FID. These achievements can be observed in Fig. 5,
where our approach demonstrates robust rendering quality
under trajectory shifting.
5.4. Ablation Study
To verify the effectiveness of the proposed method, we iso-
late each of these modules separately while keeping the
other modules unchanged, evaluating the metrics and illus-

<!-- page 7 -->
RoadSight
Waymo [21]
NTA-IoU↑
NTL-IoU↑
FID↓
NTA-IoU↑
NTL-IoU↑
FID↓
DNGaussian [12]
0.561
50.02
265.18
0.491
49.28
89.91
SparseGS [28]
0.554
67.74
224.62
0.392
49.27
93.34
StreetGaussian [29]
0.552
63.03
231.84
0.498
50.19
110.37
S3Gaussian [9]
0.538
59.13
237.41
0.384
48.75
130.43
I2V-GS (Ours)
0.804
84.62
197.35
0.646
52.31
73.54
Table 2. Comparison of NTA-Iou, NTL-Iou, and FID scores under novel view.
Figure 4. Quality comparison of vehicle view rendering with DNGaussian [12], SparseGS [28], StreetGaussian [29], and S3Gaussian [9]
on RoadSight dataset. Four input images are from the same intersection with the same timestamp.
trating the visualization results. As shown in Tab. 3, the
performance decreases when replacing any of the modules.
LiDAR Monocular Depth.
We adopt the original
monocular depth and employ Pearson correlation loss in
optimization to evaluate the effectiveness.
From Fig.
6
‘w/o LiDAR mono depth’, it can be observed that monocu-

<!-- page 8 -->
Figure 5. Quality comparison of novel trajectory rendering with DNGaussian [12], SparseGS [28], StreetGaussian [29], and S3Gaussian [9]
on Waymo [21] dataset.
Figure 6. Ablation study on proposed methods. Original monocular depth is adopted in ‘w/o LiDAR mono depth’ and fixed depth warp is
employed in ‘w/o adaptive warp’.
NTA-IoU↑
NTL-IoU↑
w/o LiDAR mono depth
0.513
43.86
w/o adaptive warp
0.682
74.41
w/o cascade strategy
0.786
83.49
w/o confidence-guide
0.783
81.57
Complete model
0.804
84.62
Table 3. Ablation studies on proposed methods.
lar depth with Pearson loss fails to constrain the position of
Gaussians, causing blurring and noise.
Adaptive Depth Warp. We employ fixed depth warp
in Fig. 6 ‘w/o adaptive warp’. Although the quality of the
bottom image is acceptable, there are artifacts in the top
image. This indicates that adaptive depth warp is robust to
scenario changes to generate diverse training views.
Cascade Strategy. We separately inpaint holes for each
warped image in Fig. 6 ‘w/o cascaded strategy’. The traf-
fic light in the top image is significantly different from the
input, where the red light is changed to a green light, while
that in the bottom image is distorted.
Confidence-Guided Optimisation.
The confidence-
guided optimisation is removed in Fig. 6 ‘w/o confidence-
guide’. It is evident that confidence-guided optimization
can avoid the inaccuracy of the diffusion model and improve
the render quality.
6. Conclusion
In this paper, we present I2V-GS, the first framework for
generating autonomous driving datasets with I2V view
transformation.
To address challenges caused by the
sparse view input and large view transformation, we first
adopt LiDAR to calibrate monocular depth to provide an

<!-- page 9 -->
accurate depth.
Then these depths are utilized to carry
the adaptive depth warp to generate dense training views.
The cascade strategy is introduced to inpaint holes in
warped images iteratively to ensure the consistency of
inpaint content across views, which also enables the depth
warp to generate a wider range of views. The cross-view
information is employed to guide the optimization to ensure
the reliability of pseudo views.
Extensive experiments
demonstrate that our approach significantly improves the
view synthesis quality from the vehicle view. These results
highlight the possibility of leveraging I2V-GS to generate
training data for end-to-end autonomous driving systems.
References
[1] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora,
Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Gi-
ancarlo Baldan, and Oscar Beijbom.
nuscenes: A multi-
modal dataset for autonomous driving. In 2020 IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 11621–11631, 2020. 1
[2] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, An-
dreas Geiger, and Hongyang Li.
End-to-end autonomous
driving: Challenges and frontiers. IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, 2024. 1
[3] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo
Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view
generation with diverse 3d geometry control. arXiv preprint
arXiv:2310.02601, 2023. 1, 3
[4] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta,
Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang
Li.
Vista:
A generalizable driving world model with
high fidelity and versatile controllability.
arXiv preprint
arXiv:2405.17398, 2024. 1, 3
[5] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
Bernhard Nessler, and Sepp Hochreiter. Gans trained by a
two time-scale update rule converge to a local nash equilib-
rium. Advances in neural information processing systems,
30, 2017. 6
[6] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez,
George Fedoseev, Alex Kendall, Jamie Shotton, and Gian-
luca Corrado.
Gaia-1: A generative world model for au-
tonomous driving. arXiv preprint arXiv:2309.17080, 2023.
1, 3
[7] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi
Yan, and Dacheng Tao. St-p3: End-to-end vision-based au-
tonomous driving via spatial-temporal feature learning. In
European Conference on Computer Vision, pages 533–549.
Springer, 2022. 1
[8] Yi Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima,
Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wen Wang,
Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao,
and Hongyang Li. Planning-oriented autonomous driving.
In 2023 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 17853–17862, 2022. 1
[9] Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An,
Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer,
and Shanghang Zhang.
S3gaussian:
Self-supervised
street gaussians for autonomous driving.
arXiv preprint
arXiv:2405.20323, 2024. 1, 2, 7, 8
[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2, 3, 4, 5, 6
[11] Diederik P. Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. CoRR, abs/1412.6980, 2014. 6
[12] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun
Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d
gaussian radiance fields with global-local depth normaliza-
tion. In 2024 IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 20775–20785, 2024. 1, 2, 7,
8
[13] Yiming Li, Shunli Ren, Pengxiang Wu, Siheng Chen, Chen
Feng, and Wenjun Zhang. Learning distilled collaboration
graph for multi-agent perception. Advances in Neural Infor-
mation Processing Systems, 34:29541–29552, 2021. 1
[14] Mingyu Liu, Ekim Yurtsever, Jonathan Fossaert, Xingcheng
Zhou, Walter Zimmer, Yuning Cui, Bare Luka Zagar, and
Alois C Knoll. A survey on autonomous driving datasets:
Statistics, annotation quality, and a future outlook.
IEEE
Transactions on Intelligent Vehicles, 2024. 1
[15] Xinhang Liu, Jiaben Chen, Shiu-Hong Kao, Yu-Wing Tai,
and Chi-Keung Tang.
Deceptive-nerf/3dgs:
Diffusion-
generated pseudo-observations for high-quality sparse-view
reconstruction. In European Conference on Computer Vi-
sion, pages 337–355. Springer, 2024. 2
[16] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[17] Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu,
Wenkang Qin, Guan Huang, Chen Liu, Yuyin Chen, Yida
Wang, Xueyang Zhang, et al. Recondreamer: Crafting world
models for driving scene reconstruction via online restora-
tion. arXiv preprint arXiv:2411.19548, 2024. 3
[18] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall.
Dreamfusion: Text-to-3d using 2d diffusion.
arXiv
preprint arXiv:2209.14988, 2022. 2
[19] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In 2022 IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
10684–10695, 2022. 5
[20] Germ´an Ros, Laura Sellart, Joanna Materzynska, David
V´azquez, and Antonio M. L´opez. The synthia dataset: A
large collection of synthetic images for semantic segmenta-
tion of urban scenes. 2016 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 3234–3243,
2016. 1
[21] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien
Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
Yuning Chai, Benjamin Caine, et al.
Scalability in per-
ception for autonomous driving: Waymo open dataset. In

<!-- page 10 -->
2024 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 2446–2454, 2020. 1, 4, 6, 7, 8
[22] Alexander Swerdlow, Runsheng Xu, and Bolei Zhou. Street-
view image generation from a bird’s-eye view layout. IEEE
Robotics and Automation Letters, 9(4):3578–3585, 2024. 1,
3
[23] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Prad-
han, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T. Bar-
ron, and Henrik Kretzschmar.
Block-nerf: Scalable large
scene neural view synthesis. In 2024 IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 8248–
8258, 2022. 2
[24] Haithem Turki, Jason Y. Zhang, Francesco Ferroni, and
Deva Ramanan. Suds: Scalable urban dynamic scenes. In
2023 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 12375–12385, 2023. 2
[25] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jia-
gang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-
drive world models for autonomous driving. In European
Conference on Computer Vision, pages 55–72. Springer,
2024. 1, 3
[26] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen,
and Zhaoxiang Zhang. Driving into the future: Multiview
visual forecasting and planning with world model for au-
tonomous driving. In 2024 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 14749–14759,
2024. 1, 3
[27] Chulin Xie, Zhong Cao, Yunhui Long, Diange Yang, Ding
Zhao, and Bo Li. Privacy of autonomous vehicles: Risks,
protection methods, and future directions. 2022. 1
[28] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay,
Pradyumna Chari, and Achuta Kadambi. Sparsegs: Real-
time 360° sparse view synthesis using gaussian splatting.
arXiv preprint arXiv:2312.00206, 2023. 1, 2, 7, 8
[29] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang,
Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou,
and Sida Peng. Street gaussians: Modeling dynamic urban
scenes with gaussian splatting. In European Conference on
Computer Vision, pages 156–173, 2024. 1, 2, 6, 7, 8
[30] Yunzhi Yan, Zhen Xu, Haotong Lin, Haian Jin, Haoyu Guo,
Yida Wang, Kun Zhan, Xianpeng Lang, Hujun Bao, Xi-
aowei Zhou, et al.
Streetcrafter:
Street view synthesis
with controllable video diffusion models.
arXiv preprint
arXiv:2412.13188, 2024. 3
[31] Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu
Li, Bo Dai, Kashyap Chitta, Peng Wu, Jia Zeng, Ping Luo,
Jun Zhang, Andreas Geiger, Yu Qiao, and Hongyang Li.
Generalized predictive model for autonomous driving.
In
2024 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 14662–14672, 2024. 1, 3
[32] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi
Feng, and Hengshuang Zhao. Depth anything: Unleashing
the power of large-scale unlabeled data. In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 10371–10381, 2024. 2
[33] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any-
thing v2. Advances in Neural Information Processing Sys-
tems, 37:21875–21911, 2024. 6
[34] Zeyu Yang, Zijie Pan, Yuankun Yang, Xiatian Zhu, and
Li Zhang.
Driving scene synthesis on free-form trajecto-
ries with generative prior. arXiv preprint arXiv:2412.01717,
2024. 1, 3
[35] Haibao Yu, Yizhen Luo, Mao Shu, Yiyi Huo, Zebang Yang,
Yifeng Shi, Zhenglong Guo, Hanyu Li, Xing Hu, Jirui
Yuan, and Zaiqing Nie.
Dair-v2x: A large-scale dataset
for vehicle-infrastructure cooperative 3d object detection. In
2022 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 21329–21338, 2022. 5
[36] Haibao Yu, Wen-Yen Yang, Hongzhi Ruan, Zhenwei Yang,
Yingjuan Tang, Xuming Gao, Xin Hao, Yifeng Shi, Yifeng
Pan, Ning Sun, Juan Song, Jirui Yuan, Ping Luo, and Za-
iqing Nie.
V2x-seq: A large-scale sequential dataset for
vehicle-infrastructure cooperative perception and forecast-
ing. In 2023 IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 5486–5495, 2023. 5
[37] Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu,
Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen,
Boyuan Wang, Youyi Zhang, et al. Drivedreamer4d: World
models are effective data machines for 4d driving scene rep-
resentation. arXiv preprint arXiv:2410.13571, 2024. 1, 3,
6
[38] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang,
Deqing Sun, and Ming-Hsuan Yang.
Drivinggaussian:
Composite gaussian splatting for surrounding dynamic au-
tonomous driving scenes. In 2023 IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21634–
21643, 2023. 1, 2
