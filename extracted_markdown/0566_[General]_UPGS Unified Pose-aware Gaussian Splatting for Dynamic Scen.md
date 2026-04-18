<!-- page 1 -->
UPGS: Unified Pose-aware Gaussian Splatting for
Dynamic Scene Deblurring
Zhijing Wu1 and Longguang Wang2
1 University of Cambridge, Cambridge, UK
zw436@cantab.ac.uk
2 The Shenzhen Campus of Sun Yat-sen University, Shenzhen, China
wanglg9@mail.sysu.edu.cn
Abstract. Reconstructing dynamic 3D scenes from monocular video
has broad applications in AR/VR, robotics, and autonomous naviga-
tion, but often fails due to severe motion blur caused by camera and
object motion. Existing methods commonly follow a two-step pipeline,
where camera poses are first estimated and then 3D Gaussians are opti-
mized. Since blurring artifacts usually undermine pose estimation, pose
errors could be accumulated to produce inferior reconstruction results.
To address this issue, we introduce a unified optimization framework by
incorporating camera poses as learnable parameters complementary to
3DGS attributes for end-to-end optimization. Specifically, we recast cam-
era and object motion as per-primitive SE(3) affine transformations on
3D Gaussians and formulate a unified optimization objective. For stable
optimization, we introduce a three-stage training schedule that optimizes
camera poses and Gaussians alternatively. Particularly, 3D Gaussians
are first trained with poses being fixed, and then poses are optimized
with 3D Gaussians being untouched. Finally, all learnable parameters
are optimized together. Extensive experiments on the Stereo Blur and
BARD-GS dataset with challenging real-world sequences demonstrate
that our method achieves significant gains in reconstruction quality and
pose estimation accuracy over prior dynamic deblurring methods.
Keywords: Gaussian splatting · Dynamic scene reconstruction · Motion
Deblurring
1
Introduction
Recent advances in explicit 3D Gaussian Splatting (3DGS) [8] and implicit Neu-
ral Radiance Fields (NeRF) [25] have pushed photorealistic reconstruction from
static 3D scenes to fully dynamic 4D ones. These techniques promote emerg-
ing applications such as mobile AR/VR, autonomous driving, and robotics, in
which data are almost acquired with a single, moving camera in unconstrained
environments [7, 23, 51]. In these scenarios, severe motion-induced blur usually
remains a dominant factor that limits reconstruction quality.
Motion blur arises as the sensor integrates light over a finite exposure while
the object or the camera moves, which consists of two components: camera-
induced blur from ego-motion and object-induced blur from fast, often non-rigid
arXiv:2509.00831v3  [cs.CV]  15 Mar 2026

<!-- page 2 -->
2
Z. Wu and L. Wang
motion [6,32]. These motions produce blurring artifacts that hinder high-quality
4D reconstruction. The major reasons are twofold: (i) Low pose accuracy.
Structure-from-motion pipelines such as COLMAP depend on sharp, repeatable
keypoints to triangulate a sparse cloud and estimate camera poses. However, mo-
tion blur loses high-frequency details, frustrates feature matching, and produces
inaccurate camera poses [12,31]. (ii) High ill-posedness. As each blurred pixel
integrates light from a trajectory rather than a fixed 3D location, which weak-
ens the geometric constraint and increases the ill-posedness to reconstruct 4D
scenes [10].
Over the last years, numerous efforts [13–15, 22, 28, 37, 39] have been made
to reconstruct sharp scenes from motion-blurred imagery, yet most methods still
focus on static 3D scenes. Recently, dynamic deblurring has emerged [2,10,20,
21, 35] and is mainly built on NeRF. Since the continuous representation nat-
urally favors smooth, low-frequency results, fast object motions are typically
more difficult to be optimized, leading to unsatisfactory solutions. In addition,
volumetric ray marching through a deep MLP requires long training time [25],
making deployment for time-sensitive AR/VR use cases challenging. Recent ad-
vances in Gaussian splatting has provided a new approach, yet their applications
in dynamic deblurring is still under-investigated.
Intuitively, 4D reconstruction from a blurry video is a joint optimization
problem in which camera pose estimation and sharp scene reconstruction are
deeply coupled [1,17]. On the one hand, sharp reconstruction improves the ac-
curacy of pose estimation by providing higher-quality keypoints. On the other
hand, accurate camera poses alleviate the ill-posedness to facilitate sharp images
to be reconstructed. However, most prior approaches ignore the interaction be-
tween pose estimation and scene reconstruction. These methods commonly em-
ploy COLMAP poses to reconstruct the scene without further pose refinement.
As a result, pose errors can be accumulated to degrade the final reconstruction
results.
In this paper, we present UPGS, a Unified Pose-aware Gaussian Splatting
framework for dynamic scene deblurring. Specifically, to avoid error accumula-
tion caused by erroneous COLMAP poses, we formulate camera and object mo-
tions as learnable per-primitive SE(3) affine transformations on 3D Gaussians
and incorporate them for joint optimization.
As affine transformations are highly coupled with 3D Gaussians, an alter-
native training scheme is employed. Particularly, we first leave out affine trans-
formations to optimize 3D Gaussians using initial COLMAP poses. In this way,
static regions can be roughly reconstructed while dynamic regions still suffer poor
quality. Then, we include time-conditioned SE(3) transformations for training
while fixing 3D Gaussians, refining the geometry of dynamic regions. Finally,
both affine transformations and 3D Gaussians are jointly optimized. Extensive
evaluations on the Stereo Blur benchmark and challenging real-world sequences
show that our UPGS achieves superior reconstruction fidelity and pose accuracy
compared to existing dynamic deblurring methods.

<!-- page 3 -->
Abbreviated paper title
3
– We propose UPGS, a Unified Pose-aware Gaussian Splatting framework
that jointly optimizes camera poses and 3D Gaussian scene parameters in
an end-to-end manner to reconstruct dynamic scenes from blurry monocular
video.
– We recast both camera and object motion as per-primitive SE(3) affine warps
on 3D Gaussians, enabling joint refinement in the same optimization pipeline.
– We introduce a three-stage training schedule—scene-only, pose-only, then
full joint optimization—to prevent error accumulation and accelerate con-
vergence.
– We demonstrate that UPGS yields sharp reconstructions and more accurate
camera-pose estimates on challenging real-world sequences from the Stereo
Blur [50] and BARD-GS datasets [18], indicating its effectiveness for dynamic
deblurring tasks.
2
Related Work
In this section, we first review previous 2D deblurring methods, and then present
advanced novel view synthesis methods. Finally, we focus on discussing recent
advances regarding deblurring dynamic reconstruction.
2.1
2D Deblurring
The goal of deblurring is to recover sharp imagery from blur caused by camera
shake or object motion. Classical methods cast this as an inverse-filtering prob-
lem with hand-crafted priors [4,9,24,30], while modern deep-learning approaches
train end-to-end multi-scale, recurrent, attention- and adversarial-based net-
works [11,26,36,46] on large datasets (e.g GoPro [26], DeblurGAN [11]), enabling
real-time single-image deblurring. Video deblurring further leverages temporal
cues by aligning or fusing features across frames via optical flow, deformable
convolutions, or recurrent/3D networks [27,29,34,40,49]. Despite these advances
in 2D image/video space, such models lack explicit 3D scene geometry model-
ing. Our work formulates deblurring as a 4D reconstruction problem within the
Gaussian Splatting framework.
2.2
Dynamic Novel View Synthesis
Gaussian Splatting (GS) [8] has emerged as a popular choice for modeling 3D
scenes due to its explicit structure, high rendering efficiency, and real-time per-
formance. Recent works have extended GS into 4D to handle dynamic scene
reconstruction from monocular input. Early attempts, such as Dynamic 3D
Gaussians [19], independently tracked each Gaussian across timestamps to cap-
ture scene motion, while Deformable-3D-Gaussians [45] introduced a deforma-
tion field via an MLP to learn time-dependent transformations of canonical
Gaussians. 4D Gaussian Splatting [38, 42, 44] advanced this idea by directly

<!-- page 4 -->
4
Z. Wu and L. Wang
lifting Gaussians into 4D space, jointly encoding spatial and temporal varia-
tions to enable high-quality real-time dynamic rendering. More recent develop-
ments, including Disentangled 4D Gaussian Splatting [3], significantly improve
efficiency by factorizing spatial and temporal components, while benchmarking
efforts [16] have analyzed the performance and brittleness of various monocular
4D GS pipelines. These milestones have collectively established GS as a com-
petitive framework for 4D reconstruction, capable of modeling complex motions
while maintaining real-time rendering capabilities. However, existing 4D GS ap-
proaches still rely on sharp, high-quality inputs and accurate camera poses to
achieve good reconstruction fidelity, which limits their robustness in real-world
scenarios involving motion blur. To address this, our method aims to reconstruct
4D scenes directly from blurry monocular inputs, jointly handling deblurring and
dynamic scene modeling.
2.3
Deblurring Dynamic Reconstruction
Recently, a variety of works have been proposed to recover high-quality 3D recon-
structions from blurred inputs. NeRF-based approaches [14,15,22,37] treat blur
as part of the forward model by learning spatially varying kernels or simulating
exposure via camera-trajectory subframes. 3DGS-based static deblurring folds
the same exposure model into an explicit pipeline, with Debluring 3DGS [13]
as a pioneering example. BAD-Gaussians [48] and DeblurGS [28] share a work-
flow that jointly estimates exposure-time camera motion and optimizes Gaussian
scene parameters by rendering virtual sharp subframes along a continuous SE(3)
trajectory, averaging them to synthesize the observed blur, and driving a pho-
tometric loss against the input images. These approaches deliver high-quality
results on static scenes but do not handle dynamics. Turning to dynamic 4D de-
blurring, [2,21,35] generates latent rays within the exposure and performs time-
aware volume rendering. However, training often requires significantly longer
runtimes, and the implicit MLP’s smoothness and capacity bias toward low-
frequency content causes fast non-rigid motions to be underfit. Our approach
fills this gap with a 4D Gaussian-splatting formulation that reconstructs di-
rectly from blurry monocular video while explicitly decoupling camera-induced
blur via subframe warps along with the object motion, improving efficiency and
preserving high-frequency structure in dynamic areas.
3
Method
In this section, we first present the preliminaries in Sec. 3.1. Then, we introduce
our unified pose-aware framework in Sec. 3.2 and detail our optimization strategy
is Sec. 3.3.
3.1
Preliminaries
Gaussian Splatting Primer 3D Gaussian Splatting (3DGS) encodes a scene
as a collection of anisotropic 3\text {D} Gaussians. Each primitive is parameterised by

<!-- page 5 -->
Abbreviated paper title
5
𝑡
𝑡1
𝑡2
𝑡3
t3
(1)
t3
(2)
t3
(3)
t3
(4)
𝐿𝑡𝑜𝑡𝑎𝑙
Motion Blur Formulation
෢
𝐵𝑡
𝐵𝑡
Exposure time
𝑃m→𝑛
𝐺m→𝑛
Per-primitive SE(3) affine transformation
=
𝑡
𝑡
𝑡
Blurry Input
SfM Initialization
Stage1
Stage2
Stage3
Scene-pose Stagewise Optimization
rasterize
Render
Trainable
Frozen
Fig. 1: An overview of UPGS. We adopt a three-stage training schedule for opti-
mization. Camera motion is represented as trainable SE(3) affine transformations (Sec
3.2) on Gaussian primitives, thereby camera poses can be optimized together with the
reconstructed scene. Using COLMAP poses for initialization, we first optimize Gaus-
sian primitives with poses fixed. Next, with Gaussian primitives being frozen, we refine
only the affine transformations. In the final stage, we jointly fine-tune scene and pose
so they co-adapt (Sec 3.3), yielding sharper renders, higher reconstruction fidelity, and
more accurate camera trajectories.
a centre \ p rotect \mathbf  {x}\in {R}^{3}, covariance \ p rotect \bm  {\Sigma }\in {R}^{3\times 3}, opacity \alpha , and view-dependent colour
using learnable spherical-harmonic (SH) coefficients. Under a given camera pose,
every Gaussian is projected to the image plane, yielding a 2\text {D} footprint whose
shape is controlled by the projected covariance \protect \bm  {\Sigma }'. For a pixel location \protect \mathbf  {x}, colour
is accumulated with front-to-back alpha blending:
  \ha t
 
{c}
(\ mathb
f {
x
}) 
\
; = \; \s
u
m _{i\in \mathcal {N}} c_{i}\,\alpha _{i}(\mathbf {x}) \prod _{j=1}^{\,i-1}\!\bigl (1-\alpha _{j}(\mathbf {x})\bigr ), \label {eq:alpha_blend} 
(1)
where c_i is the SH-predicted colour and \alpha _i(\mathbf {x}) is the opacity of the i-th Gaussian.
Scene parameters \if mm od e \lbrace \else \textbraceleft \fi \mathbf {x},\boldsymbol {\Sigma },\alpha ,\text {SH}\} are optimised by minimising an image-space
loss between the rasterized images and their reference counterparts across all
viewpoints.
Gaussian Deformation for Object Motion Following Deblur4DGS [43],
object deformation is modeled by \protect \mathrm  {SE}(3) transforms \ifm mode \lbrace \else \textbraceleft \fi A_t, E_t\} via Shape-of-Motion

<!-- page 6 -->
6
Z. Wu and L. Wang
Algorithm 1 Stagewise Pose–Scene Optimization
Require: scene θ, poses ϕ, epochs Emax, stop-gradient operator ⊥(·)
1: stage ←1
2: for e = 0 to Emax −1 do
3:
for t in minibatch do
4:
if stage = 1 then
\triangleright scene warm-up
5:
θ ←Uθ
 Ltotal(t; θ, ⊥(ϕ))

6:
if ∆psnr < ϵ1 ∧anchor > τa then
7:
stage ←2
8:
end if
9:
else if stage = 2 then
\triangleright pose refinement
10:
ϕ ←Uϕ
 Ltotal(t; ⊥(θ), ϕ)

11:
if ∆pose < ϵ2 ∧P[L] > τgain then
12:
stage ←3
13:
end if
14:
else
\triangleright joint fine-tune
15:
θ ←Uθ
 Ltotal(t; θ, ϕ)

;
ϕ ←Uϕ
 Ltotal(t; θ, ϕ)

16:
end if
17:
end for
18: end for
[38], i.e.
  \bm { G_ {\math r m {dym},t}} = A_t\,\bm {G_{\mathrm {dym},c}} + E_t. \label {eq:gaussian_deform} 
(2)
Here At ∈R3×3 is the rigid rotation matrix and Et ∈R3 is the translation vector
at time t. The canonical dynamic Gaussians \protect \bm  {G_{\mathrm {dym},c}} are obtained by uniformly
dividing the video into L segments and selecting the frame with the highest
Laplacian sharpness in each segment as the reference. Rather than learning a sep-
arate \protect \mathrm  {SE}(3) for every subframe, we estimate only the start- and end-of-exposure
parameters (with weights \pm w_t/2) and interpolate intermediate subframe defor-
mations by
w w_ {
t
, i }  =
 \ B
i
gl (1 - \frac {i-1}{N-1}\Bigr )\odot \frac {w_t}{2} \;+\; \frac {i-1}{N-1}\odot \Bigl (-\frac {w_t}{2}\Bigr ). \label {eq:interp_weights} w
(3)
Canonical Gaussians, \protect \mathrm  {SE}(3) deformation modules, and exposure weights w_t are
then optimized jointly using dynamic reconstruction loss.
Motion Blur Formation Motion blur arises because, for a frame stamped at
time t, the sensor integrates light over an exposure interval while the camera or
scene moves. The exposure duration at time t is denoted as \delta _t. Let I_t(u ,v) be
the instantaneous rendering at pixel (u, v) at time t. The recorded blurry pixel
is the temporal integral
  B_t (u , v
)  \
;
=\; \ ph i \int _{0}^{\delta _t} I_t(u,v)\,d\delta , \label {eq:blur_integral} 
(4)

<!-- page 7 -->
Abbreviated paper title
7
with normalisation factor \phi . In practice we discretise the exposure time \delta _t at
global time t into N latent sharp frames \ifmmod
e \lbrace \else \textbraceleft \fi I_i\}_{i=0}^{N-1}
and approximate
  B_t (u ,
v
)
 \;
\
app
rox \ ; \frac {1}{N}\sum _{i=0}^{N-1} I_n(u,v). \label {eq:blur_discrete} 
(5)
This discrete form enables differentiable simulation of motion blur: sharp images
rendered at intermediate timestamps are averaged to reproduce the observed
blurry input.
3.2
Unified Pose-Aware Framework
Our unified pose-aware framework is illustrated in Fig. 1. To make camera pose
trainable within the 3DGS framework, we formulate the camera motion as an
affine transformation of the scene under a given fixed pose. In this way, camera
motion can be modeled with object motion using unified representations, which
are then applied to Gaussian primitives. Given a blurry frame B_t at timestamp t,
we estimate the camera pose using CLOMAP and employ it to initialize the affine
transformation P_ t  \in \mathrm {SE}(3). To capture the camera trajectory during exposure
time, we subdivide the exposure time into N intervals such that the camera
poses for subframes can be denoted as:
  P_
t
^
{
(m)}
 
=  \bi
g
l
 [R _ t ^{ ( m ) } ,\,t_t^{(m)}\bigr ] \quad \text {for } m=1,\dots ,N, \label {eq:subframe_poses} 
(6)
Without loss of generality, one of these poses is selected as a reference pose:
  P_
t
^
{
(n)}
 
=  \bi
g
l
 [R_t^{(n)},\,t_t^{(n)}\bigr ]. \label {eq:reference_pose} 
(7)
Crucially, for any two poses P ^{(n)} and P ^{(m)}, there exists an affine transformation
T ^{n\to m} on the Gaussians \ p rotect  \bm  {G} = \{\bm {G}_{\mathrm {sta}}, \bm {G}_{\mathrm {dym}}\} to satisfy:
 
 
I\ b igl 
( \
b
m  {G},\, P ^{(m
)}\bigr ) = I\!\bigl (T^{n\to m}(\bm {G}),\,P^{(n)}\bigr ), \label {eq:equivalent_render} 
(8)
where I
 G, P (m)
denotes the image rasterization using 3DGS G under the pose
P (m). Specifically, we employ the affine transformation to map the reference view
into the m-th subframe view,
  \b
e
g
i
n {a
l
ig ned}
 
R
_ t^{
\
,
n
\to 
m
} &
 = \
b
i gl (
R
_t^{(m)}\bigr )^{\!\top }\,R_t^{(n)}, \\ t_t^{\,n\to m} & = \bigl (R_t^{(m)}\bigr )^{\!\top }\bigl (t_t^{(m)} - t_t^{(n)}\bigr ) \end {aligned} \label {eq:affine_rt_tt} 
(9)
and apply these transformations directly to each Gaussian primitive \ p
rote ct \bm   {G}=(\mu _i,\Sigma _i,c_i,\alpha _i) via
  
\ b e gin
 
{a l i gne
d
}
 \
m u  _i'
 
& 
=
 R_t
^
{\,n\to m}\,\mu _i + t_t^{\,n\to m}, \\ \Sigma _i' & = R_t^{\,n\to m}\,\Sigma _i\,\bigl (R_t^{\,n\to m}\bigr )^{\!\top } \end {aligned} \label {eq:affine_mu_sigma} 
(10)
leaving color c_i and opacity \alpha _i unchanged. As a result, by rendering the warped
3D Gaussians \ifmmode \lbrace \else \textbraceleft \fi \bm {G}'\} from the fixed reference pose P _t^
{(n)}
and averaging over all
intermediate subframes, we faithfully mimic the true motion blur— the integral
of the scene’s geometry under the camera’s continuous motion—directly in the
geometry domain.

<!-- page 8 -->
8
Z. Wu and L. Wang
Input
GT
Deblur4DGS
Ours
Shape-of-motion
4DGS
Fig. 2: Visual Comparison on Stereo Blur and BARD-GS Dataset. The
orange boxes highlights regions with intense dynamic motion and the blue boxes
indicate purely static areas.
3.3
Optimization Strategy
Since pose and geometry are tightly coupled, a fully joint optimization can be
highly ill-posed and produce trivial solutions. Minor errors in camera pose esti-
mation may skew the scene fit, which in turn misguides subsequent pose updates,
leading to error accumulation and unsatisfactory reconstruction results. To ad-
dress this, we partition the training phase into three successive stages: Stage 1
optimizes only the scene parameters, warm up the model with the basic scene
geometry. Stage 2 freezes the scene and updates only the camera poses, enabling
the system to digest the camera motion and understand the source of blur.
Stage 3 then performs full joint optimization of both scene and pose. The overall
training strategy is summarized in Algorithm 1.
The transition between stages is determined automatically by two adaptive
gates. Gate 1 monitors whether the scene representation has saturated. It checks:
– ∆psnr < ϵ1: whether the increase in validation PSNR has plateaued.
– anchor > τa: the extent to which the static Gaussians begin to shift as the
model attempts to explain residual errors by implicitly mimicking camera
motion.
  \beg i
n
 {aligned
}
 \text  {anchor} = \sum _{g \in G_{\text {static}}} \left ( \lVert \Delta \mu _g \rVert ^{2} + \lVert \Delta R_g \rVert ^{2} \right ) \end {aligned} 
(11)
Once these criteria are met, the scene has extracted the majority of learnable
appearance structure, and pose optimization is activated.

<!-- page 9 -->
Abbreviated paper title
9
Table 1: Quantitative results on the Stereo and BARD datasets. Red and yellow
denote the best and second-best per column, respectively.
Stereo Blur Dataset
BARD Dataset
PSNR ↑SSIM ↑LPIPS ↓
Training
time
↓
PSNR ↑SSIM ↑LPIPS ↓
Training
time
↓
DyBluRF
23.82
0.690
0.471
49
16.25
0.523
0.971
87
Shape-of-Motion
27.49
0.922
0.192
3.7
17.92
0.718
0.434
8
4DGS
26.33
0.733
0.345
0.8
17.28
0.609
0.877
1.8
Deblur4DGS
29.02
0.745
0.165
6.9
18.80
0.720
0.372
13
Ours
30.14
0.911
0.107
4.2
19.50
0.731
0.550
10
Gate 2 evaluates whether pose refinement is exhausted. It detects:
– ∆pose < ϵ2: whether the decrease in pose error has plateaued.
– P[L](θ, ϕ) > τgain: whether enabling scene updates would yield a meaningful
improvement in the overall objective. To assess this, we simulate a virtual
optimizer step on the scene parameters as if scene learning were enabled. We
then evaluate the corresponding decrease in loss by computing a lightweight
look-ahead gain proxy from the scene gradients. A marked improvement is
identified when incorporating scene parameters results in a clear reduction
in total loss.
  \begi n { aligned}  \mathcal {P}[\mathcal {L}](\theta ,\phi ) = L_{\mathrm {pose}}(\phi ) - L_{\mathrm {joint}}(\theta ,\phi ) \end {aligned} 
(12)
When both conditions hold, the optimizer proceeds to Stage 3 for joint refinement
of geometry and camera motion. The detailed values for the gating hyperparam-
eters are provided in the supplementary material.
To avoid over-smoothed edges between the static and dynamic regions of the
scene, we train background and foreground Gaussians in an end-to-end pipeline.
Particularly, we learn G_{\mathrm {sta}} and G_{\mathrm {dym}} jointly by minimizing
  \b
e
gin { align e d} 
&
L
_
{\m
a
thrm {dym} } \bi
g
l
 (\bm
 {G_{\mathrm {sta}}},\,\bm {G_{\mathrm {dym}}}; P_t^{(m)}\bigr ) \\ = &\bigl \lVert I\bigl (\bm {G_{\mathrm {sta}}},\,\bm {G_{\mathrm {dym}}}; P_t^{(m)}\bigr ) - I(t) \bigr \rVert ^{2}, \end {aligned} \label {eq:loss_dym} 
(13)
which regularizes all motion-induced blur over sub-frames m. To further enforce
a crisp background, we introduce
  L_{\m
a
thrm {s
t
a
tic
}
}\big l 
(
\ bm {G_{\ma
thrm {sta}}}; P_t\bigr ) = \bigl \lVert I\bigl (\bm {G_{\mathrm {sta}}}; P_t\bigr ) - I_{\mathrm {static}}(t)\bigr \rVert ^2, \label {eq:loss_static} 
(14)
The final objective is defined as:
  L_{\ m athr
m
 {tot al}}
 
=  L_{\ma
t
hrm 
{
dym}}\bigl (\bm {G_{\mathrm {sta}}},\,\bm {G_{\mathrm {dym}}}\bigr ) + L_{\mathrm {static}}\bigl (\bm {G_{\mathrm {sta}}}\bigr ), \label {eq:loss_total} 
(15)
which is optimized end-to-end so that background and foreground Gaussians
co-adapt, capturing dynamic blur while preserving static details.

<!-- page 10 -->
10
Z. Wu and L. Wang
3.4
Discussion
We position UPGS against existing joint scene-pose deblurring pipelines by high-
lighting how it departs from prior neural-based approaches. BARF [17] resolves
pose–scene entanglement by gradually unmasking high-frequency positional en-
codings inside NeRF. However, it applies only to sharp, static input. We perform
joint optimisation and reconstruction from dynamic, motion-blurred video. Un-
like Deblur4DGS [43] which first optimises a frozen static background and then
deblurs the dynamic foreground with extra regularisation, UPGS trains static
and dynamic Gaussians together from the outset, avoiding additional loss terms.
The recent BARD-GS [18] learns one global pose per frame in camera space and
interpolates intermediate viewpoints, while we represents both scene and pose
in a unified geometry space and directly captures the entire camera trajectory.
4
Experiment
In this section, we first introduce the implementation details, and then compare
the performance of our method against previous approaches. Finally, we conduct
ablation experiments to demonstrate the effectiveness of our method designs.
4.1
Implementation details
Dataset. We evaluate our method on the Stereo Blur dataset [35] and BARD-
GS dataset [18]. Stereo Blur dataset comprises six scenes exhibiting significant
motion blur from both camera and object motion. Each sequence was captured
with a ZED stereo camera: the left view provides the blurry input, while the
right view serves as the sharp ground truth. Details of the blur synthesis process
are described in [50]. For each scene, we extract 24 frames, and obtain camera
extrinsics using COLMAP [33]. BARD-GS dataset are specifically collected for
dynamic deblurring task. We further subselect frames that exhibit the most
representative motions within each scene.
Training Configurations. Our model is trained on the blurry videos, then
evaluated on the sharp counterparts. Either one of the rendered subframe (i.e
start, middle, or end) is selected as the deblurred scene and subsequently com-
pared with the sharp ground truth to perform the evaluation. Our method is
trained for 200 epoch. All experiments were performed on Nvidia RTX 4080
32GB GPU.
Metrics. We perform thorough quantitative comparisons by measuring the
synthesized novel views with established metrics: Peak Signal-to-Noise Ratio
(PSNR) [5], Structural Similarity Index Measure (SSIM) [41], and Learned Per-
ceptual Image Patch Similarity (LPIPS) [47], which together capture both re-
construction fidelity and perceptual quality.

<!-- page 11 -->
Abbreviated paper title
11
Table 2: Deblurring results across three regions on the Stereo Blur dataset
Methods
Dynamic region
Static region
Edge region
PSNR↑/SSIM↑/LPIPS↓PSNR↑/SSIM↑/LPIPS↓PSNR↑/SSIM↑/LPIPS↓
Deblur4DGS
28.283 / 0.817 / 0.125
29.672 / 0.716 / 0.182
24.844 / 0.869 / 0.135
Ours
29.315 / 0.955 / 0.056
30.531 / 0.878 / 0.158
26.511 / 0.988 / 0.021
w/o affine
Input
GT
Ours
w/o stagewise
Blurry View
Fig. 3: Visual Comparison of Ablation Studies on a) Affine warp representation
b) Stagewise optimization strategy
4.2
Comparison with State-of-the-Art Methods
Quantitative Results. The quantitative results on the Stereo Blur [35] and
BARD-GS dataset [18] are summarized in Table 1. We compare our pipeline
among DyBluRF [35], Shape-of-Motion [38], 4DGS [42], and Deblur4DGS [43].
Our method attains the best PSNR and lowest LPIPS among all baselines, and a
high SSIM that is close to the top score. Among the most comparable baselines,
Deblur4DGS [43] shares a Gaussian-splatting backbone and is designed specifi-
cally for motion-blurred novel-view synthesis. Against it, we improve PSNR by
1.12 dB, SSIM by 0.166, and reduce LPIPS by 0.058 (≈35% relative). These gains
indicate lower reconstruction error, higher structural fidelity, and substantially
better perceptual quality, corresponding to sharper textures, cleaner boundaries,
and fewer blur artifacts in the renders. In addition, Table 2 reports the metrics
on three regions—dynamic (FG), static (BG), and edges—to pinpoint where de-
blurring quality changes. The edge region is a boundary band around moving
objects, where pose and motion errors couple most and artifacts tend to con-
centrate. Compared with Deblur4DGS (the closest 4DGS deblurring baseline),
our method improves all three regions, with the largest gains on edges. As illus-
trated in the supplementary, our method achieves more significant improvements
on scenes with challenging blurs, such as street, skating, and women. This further
validates the superiority of our method.

<!-- page 12 -->
12
Z. Wu and L. Wang
w/o affine
Input
GT
Ours
Blurry View
Fig. 4: Ablation Study: Effect of affine transformations on geometric modeling.
Visual Results. We further compare the visual results produced by different
methods in Fig. 2. Qualitatively, our reconstructions are sharper across both
static backgrounds and moving subjects. In motion-dominant regions where cam-
era and object trajectories interact, our method produces crisper edges that align
closely with ground-truth boundaries, with far fewer jittered Gaussians along
contours. Artifacts such as ghosts and background bleed-through are largely
suppressed. As further shown in the supplementary material, our results main-
tain continuous, uniform blur trails as objects move, reducing frame-to-frame
flicker and the patchy over-blending or under-blending artifacts in the baseline
results.
4.3
Ablation Study
The ablation study is conducted to investigate the effectiveness of the affine
transformations and the optimiation strategy. The quantitative results are shown
in Table 3.
Affine Transformations Without affine transformations, subframe camera
poses are decoupled from 3D Gaussians such that a notable performance drop
is observed in Table 3. First, the quality of edge regions declines significantly.
As shown in Fig. 3, jitter artifacts are obvious in boundary regions across all
scenes. Second, the interior geometry deteriorates, with object bodies losing
high-frequency features and structural details. As shown in Fig. 4, incorporating
the affine transformations substantially improves geometric fidelity in regions
with rich structure. For instance, facial features and clothing wrinkles are ren-
dered much more clearly.
Optimization Strategy The stagewise training strategy reduces jitter and
semi-transparent Gaussians along object boundaries, as shown in Fig. 3. Its
benefit is most pronounced in scenes with noticeable camera motion. In Fig. 5
the camera undergoes large rotations and translations. By reconstructing the
scene first with poses fixed, the subsequent pose-only phase estimates motion
against an informative scene prior, which provides a strong geometric anchors

<!-- page 13 -->
Abbreviated paper title
13
Table 3: Ablation studies on affine warp and stagewise scene-pose optimization.
Per-scene results are provided in the supplementary material.
Stereo Dataset
BARD-GS Dataset
PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓
w/o Affine
29.306
0.893
0.123
18.772
0.734
0.528
w/o Stagewise 28.778
0.879
0.168
18.421
0.677
0.531
Ours
30.143
0.911
0.107
19.502
0.731
0.550
Table 4: Ablations on number of latent frame and reference-frame evaluation. Red
and Yellow indicate best and second best.
PSNR↑SSIM↑LPIPS↓Time
N=3
28.245
0.788
0.164
2.2h
N=5
28.998
0.847
0.142
3.4h
N=7
30.143
0.9118
0.1073
4.2h
N=9
30.098
0.9097
0.1123
5.1h
N=11
29.755
0.9028
0.1202
6.3h
N=13
29.945
0.9067
0.1152
6.9h
(a) Effect of frame count.
Reference
Frame
Stereo Blur
BARD
P↑
S↑
L↓
P↑
S↑
L↓
Middle
29.17
0.867
0.134
18.33
0.680
0.624
Sharpest (FG)
29.49
0.874
0.114
19.19
0.725
0.596
Sharpest
29.72
0.898
0.111
19.08
0.722
0.601
Ours
30.14 0.911 0.107 19.50 0.731 0.550
(b) Reference pose comparison.
for recovering the camera trajectory. This yields more accurate dynamic trajec-
tories and markedly sharper static regions. The static portions of this scene are
densely detailed, with intricate items that blur easily when the camera shakes,
which makes deblurring difficult. With our training strategy, clean textures and
crisp edges can be well reconstructed such that both foreground and background
regions are rendered with noticeably higher quality.
Latent Sharp Frame N We also conduct experiments on N in Stereo Blur
dataset, to investigate the number of latent sharp frames within an exposure,
aiming to balance reconstruction quality and training time. As stated in Sec 3.1,
N is the count of latent sharp frames whose temporal average reproduces the
observed blurry image. We sweep N = 7 to 13. Table 4a shows clear gains up to
N = 7 and only marginal improvement beyond, while computational cost grows
roughly linearly. We therefore set N = 7 for our final configuration.
Reference pose sensitivity We compare different reference pose selections
in Table 4b, including the middle subframe, the sharpest subframe in the fore-
ground region, and the sharpest subframe over the entire image. In our method,
we select as reference the subframe that contains the largest number of visible
foreground Gaussians. Empirically, this choice yields consistently strong perfor-
mance across datasets. While selecting the sharpest foreground subframe pro-
duces comparable results, the performance differences are generally small. In our
experiments, the sharpest subframe often lies close to the subframe with maximal

<!-- page 14 -->
14
Z. Wu and L. Wang
Input
GT
Ours
w/o stagewise
Blurry View
Time
Fig. 5: Ablation Studies: Impact of our optimization strategy under large camera
motion.
foreground visibility, suggesting that these choices lead to similar optimization
behavior in this setting.
5
Conclusion
We proposed UPGS, a unified, pose-aware Gaussian-splatting framework for
dynamic deblurring that optimizes camera poses alongside 3D Gaussian primi-
tives and simulates motion blur in the geometry domain with SE(3) transforms.
The pipeline delivers higher reconstruction fidelity and more accurate poses than
prior dynamic deblurring methods, with visible reductions in boundary jitter and
blur artifacts. Experiments on Stereo Blur and challenging real footage confirm
consistent gains over dynamic deblurring baselines. UPGS offers a practical path
to robust 4D reconstruction from blurry monocular input.
References
1. Bian, W., Wang, Z., Li, K., Bian, J.W., Prisacariu, V.A.: Nope-nerf: Optimising
neural radiance field with no pose prior. In: Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition. pp. 4160–4169 (2023)
2. Bui, M.Q.V., Park, J., Oh, J., Kim, M.: Moblurf: Motion deblurring neural radiance
fields for blurry monocular video. IEEE Transactions on Pattern Analysis and
Machine Intelligence (2025)

<!-- page 15 -->
Abbreviated paper title
15
3. Feng, H., Sun, H., Xie, W.: Disentangled 4d gaussian splatting: Towards faster and
more efficient dynamic scene rendering. arXiv preprint arXiv:2503.22159 (2025)
4. Fergus, R., Singh, B., Hertzmann, A., Roweis, S.T., Freeman, W.T.: Removing
camera shake from a single photograph. In: Acm Siggraph 2006 Papers, pp. 787–
794 (2006)
5. Huynh-Thu, Q., Ghanbari, M.: Scope of validity of psnr in image/video quality
assessment. Electronics letters 44(13), 800–801 (2008)
6. Ji, X., Jiang, H., Zheng, Y.: Motion blur decomposition with cross-shutter guid-
ance. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition. pp. 12534–12543 (2024)
7. Jiang, Y., Yu, C., Xie, T., Li, X., Feng, Y., Wang, H., Li, M., Lau, H., Gao, F.,
Yang, Y., et al.: Vr-gs: A physical dynamics-aware interactive gaussian splatting
system in virtual reality. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–1
(2024)
8. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)
9. Krizhevsky, A.: Advances in neural information processing systems. (No Title)
p. 1097 (2012)
10. Kumar, A., et al.: Dynamode-nerf: Motion-aware deblurring neural radiance field
for dynamic scenes. In: Proceedings of the Computer Vision and Pattern Recogni-
tion Conference. pp. 21728–21738 (2025)
11. Kupyn, O., Budzan, V., Mykhailych, M., Mishkin, D., Matas, J.: Deblurgan: Blind
motion deblurring using conditional adversarial networks. In: Proceedings of the
IEEE conference on computer vision and pattern recognition. pp. 8183–8192 (2018)
12. Lee, B., Lee, H., Ali, U., Park, E.: Sharp-nerf: Grid-based fast deblurring neural
radiance fields using sharpness prior. In: Proceedings of the IEEE/CVF Winter
Conference on Applications of Computer Vision. pp. 3709–3718 (2024)
13. Lee, B., Lee, H., Sun, X., Ali, U., Park, E.: Deblurring 3d gaussian splatting. In:
European Conference on Computer Vision. pp. 127–143. Springer (2024)
14. Lee, D., Lee, M., Shin, C., Lee, S.: Dp-nerf: Deblurred neural radiance field with
physical scene priors. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. pp. 12386–12396 (2023)
15. Lee, D., Oh, J., Rim, J., Cho, S., Lee, K.M.: Exblurf: Efficient radiance fields for
extreme motion blurred images. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision. pp. 17639–17648 (2023)
16. Liang, Y., Okunev, M., Uy, M.A., Li, R., Guibas, L., Tompkin, J., Harley, A.W.:
Monocular dynamic gaussian splatting: Fast, brittle, and scene complexity rules.
arXiv preprint arXiv:2412.04457 (2024)
17. Lin, C.H., Ma, W.C., Torralba, A., Lucey, S.: Barf: Bundle-adjusting neural radi-
ance fields. In: Proceedings of the IEEE/CVF international conference on computer
vision. pp. 5741–5751 (2021)
18. Lu, Y., Zhou, Y., Liu, D., Liang, T., Yin, Y.: Bard-gs: Blur-aware reconstruction
of dynamic scenes via gaussian splatting. In: Proceedings of the Computer Vision
and Pattern Recognition Conference. pp. 16532–16542 (2025)
19. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking
by persistent dynamic view synthesis. In: 2024 International Conference on 3D
Vision (3DV). pp. 800–809. IEEE (2024)
20. Luo, X., Sun, H., Peng, J., Cao, Z.: Dynamic neural radiance field from defo-
cused monocular video. In: European Conference on Computer Vision. pp. 142–
159. Springer (2024)

<!-- page 16 -->
16
Z. Wu and L. Wang
21. Luthra, A., Gantha, S.S., Song, X., Yu, H., Lin, Z., Peng, L.: Deblur-nsff: Neural
scene flow fields for blurry dynamic scenes. In: Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision. pp. 3658–3667 (2024)
22. Ma, L., Li, X., Liao, J., Zhang, Q., Wang, X., Wang, J., Sander, P.V.: Deblur-
nerf: Neural radiance fields from blurry images. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 12861–12870 (2022)
23. Matsuki, H., Murai, R., Kelly, P.H., Davison, A.J.: Gaussian splatting slam. In:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition. pp. 18039–18048 (2024)
24. Michaeli, T., Irani, M.: Blind deblurring using internal patch recurrence. In: Eu-
ropean conference on computer vision. pp. 783–798. Springer (2014)
25. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Commu-
nications of the ACM 65(1), 99–106 (2021)
26. Nah, S., Hyun Kim, T., Mu Lee, K.: Deep multi-scale convolutional neural network
for dynamic scene deblurring. In: Proceedings of the IEEE conference on computer
vision and pattern recognition. pp. 3883–3891 (2017)
27. Nah, S., Son, S., Lee, K.M.: Recurrent neural networks with intra-frame iterations
for video deblurring. In: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. pp. 8102–8111 (2019)
28. Oh, J., Chung, J., Lee, D., Lee, K.M.: Deblurgs: Gaussian splatting for camera
motion blur. arXiv preprint arXiv:2404.11358 (2024)
29. Pan, J., Bai, H., Tang, J.: Cascaded deep video deblurring using temporal sharpness
prior. In: Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. pp. 3043–3051 (2020)
30. Pan, J., Sun, D., Pfister, H., Yang, M.H.: Blind image deblurring using dark channel
prior. In: Proceedings of the IEEE conference on computer vision and pattern
recognition. pp. 1628–1636 (2016)
31. Peng, C., Chellappa, R.: Pdrf: progressively deblurring radiance field for fast scene
reconstruction from blurry images. In: Proceedings of the AAAI Conference on
Artificial Intelligence. vol. 37, pp. 2029–2037 (2023)
32. Rozumnyi, D., Oswald, M.R., Ferrari, V., Pollefeys, M.: Motion-from-blur: 3d
shape and motion estimation of motion-blurred objects in videos. In: Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
pp. 15990–15999 (2022)
33. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings
of the IEEE conference on computer vision and pattern recognition. pp. 4104–4113
(2016)
34. Su, S., Delbracio, M., Wang, J., Sapiro, G., Heidrich, W., Wang, O.: Deep video
deblurring for hand-held cameras. In: Proceedings of the IEEE conference on com-
puter vision and pattern recognition. pp. 1279–1288 (2017)
35. Sun, H., Li, X., Shen, L., Ye, X., Xian, K., Cao, Z.: Dyblurf: Dynamic neural
radiance fields from blurry monocular video. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. pp. 7517–7527 (2024)
36. Tao, X., Gao, H., Shen, X., Wang, J., Jia, J.: Scale-recurrent network for deep
image deblurring. In: Proceedings of the IEEE conference on computer vision and
pattern recognition. pp. 8174–8182 (2018)
37. Wang, P., Zhao, L., Ma, R., Liu, P.: Bad-nerf: Bundle adjusted deblur neural
radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition. pp. 4170–4179 (2023)

<!-- page 17 -->
Abbreviated paper title
17
38. Wang, Q., Ye, V., Gao, H., Austin, J., Li, Z., Kanazawa, A.: Shape of motion: 4d
reconstruction from a single video. arXiv preprint arXiv:2407.13764 (2024)
39. Wang, X., Yin, Z., Zhang, F., Feng, D., Wang, Z.: Mp-nerf: More refined deblurred
neural radiance field for 3d reconstruction of blurred images. Knowledge-Based
Systems 290, 111571 (2024)
40. Wang, X., Chan, K.C., Yu, K., Dong, C., Change Loy, C.: Edvr: Video restora-
tion with enhanced deformable convolutional networks. In: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition workshops.
pp. 0–0 (2019)
41. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment:
from error visibility to structural similarity. IEEE transactions on image processing
13(4), 600–612 (2004)
42. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang,
X.: 4d gaussian splatting for real-time dynamic scene rendering. In: Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition. pp. 20310–
20320 (2024)
43. Wu, R., Zhang, Z., Chen, M., Fan, X., Yan, Z., Zuo, W.: Deblur4dgs: 4d gaussian
splatting from blurry monocular video. arXiv preprint arXiv:2412.06424 (2024)
44. Yang, Z., Yang, H., Pan, Z., Zhang, L.: Real-time photorealistic dynamic
scene representation and rendering with 4d gaussian splatting. arXiv preprint
arXiv:2310.10642 (2023)
45. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaussians
for high-fidelity monocular dynamic scene reconstruction. In: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition. pp. 20331–
20341 (2024)
46. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H., Shao, L.:
Multi-stage progressive image restoration. In: Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition. pp. 14821–14831 (2021)
47. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable
effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE
conference on computer vision and pattern recognition. pp. 586–595 (2018)
48. Zhao, L., Wang, P., Liu, P.: Bad-gaussians: Bundle adjusted deblur gaussian splat-
ting. In: European Conference on Computer Vision. pp. 233–250. Springer (2024)
49. Zhou, S., Zhang, J., Pan, J., Xie, H., Zuo, W., Ren, J.: Spatio-temporal filter adap-
tive network for video deblurring. In: Proceedings of the IEEE/CVF international
conference on computer vision. pp. 2482–2491 (2019)
50. Zhou, S., Zhang, J., Zuo, W., Xie, H., Pan, J., Ren, J.S.: Davanet: Stereo deblurring
with view aggregation. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. pp. 10996–11005 (2019)
51. Zhou, X., Lin, Z., Shan, X., Wang, Y., Sun, D., Yang, M.H.: Drivinggaussian: Com-
posite gaussian splatting for surrounding dynamic autonomous driving scenes. In:
Proceedings of the IEEE/CVF conference on computer vision and pattern recog-
nition. pp. 21634–21643 (2024)

<!-- page 18 -->
18
Z. Wu and L. Wang
Supplementary Material
A
Unified Pose-aware Formulation
Given a Gaussian set G and two camera poses P (m) and P (n), where each
extrinsic is defined as:
  P^ {
(
\cdo t )}
 
=
 
\
begi n  {bma trix }  R^{(\cdot )} & t^{(\cdot )} \\ 0 & 1 \end {bmatrix}, \quad R^{(\cdot )} \in \mathbb {R}^{3\times 3},\; t^{(\cdot )} \in \mathbb {R}^{3}. 
Our objective is to mimic the change of perspective from P (m) to P (n) by con-
ducting an equivalent affine transformation on Gaussian under a fixed perspec-
tive (i.e., P (m)). Mathematically, our objective is to learn an affine transforma-
tion to satisfy the following equation:
 
 
I\ b igl 
( \
m
athca l  {G}
,P^{(n)}\bigr ) \;=\; I\bigl (T(\mathcal {G}),P^{(m)}\bigr ), 
where I(G, P (n)) denotes the image rendered from G under the camera pose P (n).
Given camera intrinsics K, camera extrinsics P, and gaussian model G, we can
then obtain:
  \b e
g
i n {ali g ned
}
 
T(\m a t hbf {G}) 
&
 
=  
\big l (P
^
{
(
m
)
}\bi gr )^ {-1} \,P^{ ( n)} \\[4p
t
]
 
& = \begin {bmatrix} R^{(m)\,\top } & -\,R^{(m)\,\top }t^{(m)} \\ 0 & 1 \end {bmatrix} \begin {bmatrix} R^{(n)} & t^{(n)} \\ 0 & 1 \end {bmatrix} \\[4pt] & = \begin {bmatrix} R^{(m)\,\top }R^{(n)} & R^{(m)\,\top }t^{(n)} - R^{(m)\,\top }t^{(m)} \\ 0 & 1 \end {bmatrix}. \end {aligned} 
Therefore, we have:
  \be g in { aligne
d}  R_ { T(\m athbf  {G}) } & = R^{(m)\,\top }R^{(n)}, \\ t_{T(\mathbf {G})} & = R^{(m)\,\top }t^{(n)} - R^{(m)\,\top }t^{(m)}. \end {aligned} 
This transformation can be applied directly to the Gaussians to mimic the per-
spective change from P (m) to P (n). In Fig. 6, we validate that rendering with our
proposed formulation yields results nearly identical to those rasterized directly
from the source pose.
In the presence of motion blur, pose inaccuracies are prone to be absorbed
into the scene geometry. Previous methods commonly use image-loss to implicitly
drive camera pose optimization. As a result, these methods suffer inferior per-
formance when blurry artifacts undermine the pose estimation accuracy. With
our pose-aware formulation, we can introduce camera pose as learnable affine
transformations that are compatible with 3DGS framework for end-to-end opti-
mization, thereby producing superior performance.
B
Gating Hyperparameter Specification
To ensure reproducible gating behaviour across all experiments, we report the
exact hyperparameter values used for the two adaptive gates as shown in Table 5.

<!-- page 19 -->
Abbreviated paper title
19
Stereo Blur
BARD-GS
ϵ1
2.4 × 10−2
4.0 × 10−2
τa
3.92 × 10−5
1.92 × 10−5
ϵ2
7.0 × 10−3
4.46 × 10−5
τgain
7.3 × 10−4
3.0 × 10−3
Table 5: Gating hyperparameters for the two adaptive gating criteria
Card
Toycar
Poster
Method
PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS
4DGS
14.04
0.54
0.99
14.98
0.56
0.91
20.10
0.70
0.80
Shape-of-motion 15.41
0.74
0.50
16.25
0.75
0.43
20.86
0.75
0.40
Deblur4DGS
18.15
0.75
0.32
16.63
0.77
0.38
21.72
0.74
0.33
Ours
18.05
0.77
0.53
18.72
0.80
0.55
22.80
0.76
0.52
Windmill
Kitchen
Shark-spin
Method
PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS
4DGS
18.21
0.69
0.81
17.70
0.54
0.91
18.66
0.63
0.83
Shape-of-motion 17.75
0.76
0.41
18.59
0.62
0.45
18.65
0.68
0.42
Deblur4DGS
17.99
0.74
0.39
18.61
0.64
0.42
19.71
0.70
0.39
Ours
18.08
0.78
0.57
19.25
0.61
0.58
20.12
0.68
0.55
Table 6: Quantitative comparison across all six scenes (Card, Toycar, Poster,
Windmill, Kitchen, Shark-spin) on the BARD-GS dataset. Metrics include
PSNR, SSIM, and LPIPS.
C
Additional Results and Ablations
We record the full quantitative results for all scenes across both datasets, com-
paring our method against existing baselines. The metrics are summarized in
Table 6. The ablation results on affine transformations and our training strategy
across all six scenes are illustrated in Table 7, which further demonstrate the
effectiveness of our method designs.
D
Demo videos
We also provide demo.mov for qualitative comparisons under real-world condi-
tions. Compared with the baseline, our renderings demonstrate noticeably im-
proved temporal consistency: object contours remain stable without jitter, tex-
tures are coherent over time with minimal flicker or shimmer, and motion-blur
streaks appear smooth and continuous. These improvements are particularly evi-
dent in challenging regions where fast object motion overlaps with strong camera
shake. The supplementary demo footage provides visual evidence of these effects,

<!-- page 20 -->
20
Z. Wu and L. Wang
Input
Input
Error Map
𝑃(1)
𝑃(2)
𝑃(1)
𝑃(1)
Render from source pose 
Render from affined pose
𝑃(1)
𝑃(2)
𝐼𝑠𝑜𝑢𝑟𝑐𝑒
𝐼𝑎𝑓𝑓𝑖𝑛𝑒
𝐼𝑠𝑜𝑢𝑟𝑐𝑒− 𝐼𝑎𝑓𝑓𝑖𝑛𝑒
Fig. 6: Validation of the unified pose-aware formulation. Rendering the trans-
formed Gaussians from a fixed reference pose produces images that are nearly identical
to those rendered directly from the source poses, and the error maps confirm only neg-
ligible differences.
showcasing sharper, more stable novel views that complement the quantitative
results presented in the paper.

<!-- page 21 -->
Abbreviated paper title
21
PSNR ↑
Street
Staking
Seesaw
Man
Women
Third
Avg
Affine
30.170
30.252
29.514
26.144
27.753
32.003
29.306
Stagewise
29.751
30.796
29.290
25.709
25.824
31.300
28.778
Ours
30.383
31.877
29.786
28.766
27.693
32.352
30.143
SSIM ↑
Street
Staking
Seesaw
Man
Women
Third
Avg
Affine
0.913
0.875
0.923
0.844
0.877
0.926
0.893
Stagewise
0.903
0.900
0.919
0.828
0.807
0.916
0.879
Ours
0.920
0.915
0.929
0.906
0.869
0.932
0.911
LPIPS ↓
Street
Staking
Seesaw
Man
Women
Third
Avg
Affine
0.103
0.131
0.103
0.189
0.101
0.109
0.123
Stagewise
0.133
0.158
0.107
0.215
0.244
0.151
0.168
Ours
0.096
0.122
0.096
0.092
0.137
0.101
0.107
Table 7: Quantitative comparison across all six scenes (Street, Skating, Seesaw,
Man, Women, Third) on the Stereo Blur dataset. Metrics include PSNR, SSIM,
and LPIPS.
