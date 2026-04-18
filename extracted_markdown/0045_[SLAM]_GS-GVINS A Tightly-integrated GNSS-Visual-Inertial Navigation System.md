<!-- page 1 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
1
GS-GVINS: A Tightly-integrated GNSS-Visual-Inertial
Navigation System Augmented by 3D Gaussian Splatting
Zelin Zhou
, Saurav Uprety
, Shichuang Nie
, and Hongzhou Yang
, Senior Member, IEEE
Abstract—Accurate navigation is critical for autonomous vehi-
cles in today’s diverse traffic environments. In recent years, the
integration of Global Satellite Navigation System (GNSS), Inertial
Navigation System (INS), and camera has demonstrated signif-
icant robustness and high accuracy for navigation in complex
driving environments. Most integrated systems rely on feature-
tracking based visual odometry, which suffers from the prob-
lem of feature sparsity, high dynamics, significant illumination
changes, outlier sensitivity, etc. Recently, the emergence of 3D
Gaussian Splatting (3DGS) has drawn significant attention in the
area of 3D map reconstruction and visual SLAM. While extensive
research has explored 3DGS for indoor trajectory tracking using
visual sensor alone or in combination with Light Detection
and Ranging (LiDAR) and Inertial Measurement Unit (IMU),
its integration with GNSS for large-scale outdoor navigation
remains underexplored. To address these concerns, we proposed
GS-GVINS: a tightly-integrated GNSS-Visual-Inertial Navigation
System augmented by 3DGS. This system leverages 3D Gaussian
as a continuous differentiable scene representation in large-
scale outdoor environments, enhancing navigation performance
through the constructed 3D Gaussian map. Notably, GS-GVINS
is the first GNSS-Visual-Inertial navigation application that
directly utilizes the analytical jacobians of SE3 camera pose
with respect to 3D Gaussians. To maintain the quality of 3DGS
rendering in extreme dynamic states, we introduce a motion-
aware 3D Gaussian pruning mechanism, updating the map based
on relative pose translation and the accumulated opacity along
the camera ray. For validation, we test our system under different
driving environments: open-sky, sub-urban, and urban. Both
self-collected and public datasets are used for evaluation. The
results demonstrate the effectiveness of GS-GVINS in enhancing
navigation accuracy across diverse driving environments.
Index Terms—Localization, multi-sensor fusion, 3D Gaussian
Splatting, SLAM
I. INTRODUCTION
N
AVIGATION is a critical component of modern au-
tonomous vehicles and robotics. Continuous, high-
accuracy navigation is essential for the optimal performance of
advanced driving automation applications. The Global Satellite
Navigation System (GNSS) can deliver drift-free centimeter-
level global positioning in open-sky environments through
real-time kinematics (RTK) positioning mode [1]. However,
GNSS performance can degrade significantly in scenarios
involving non-line-of-sight (NLOS) signals or multipath inter-
ference. While Inertial Navigation System (INS) and camera-
based visual odometry provide continuous, real-time pose
(Corresponding author: Hongzhou Yang)
The
authors
are
with
the
Department
of
Geomatics
Engineering,
Shulich
School
of
Engineering,
University
of
Calgary,
Alberta,
Canada.
(e-mail:
zelin.zhou1@ucalgary.ca;
saurav.uprety1@ucalgary.ca;
sunnie.nie@ucalgary.ca; honyang@ucalgary.ca)
This work has been submitted to the IEEE for possible publication.
Copyright may be transferred without notice, after which this version may
no longer be accessible.
estimation in the local frame, their navigation solutions suffer
from error accumulation. To leverage the complementary ad-
vantages of each sensor, the GNSS-Visual-Inertial Navigation
System (GVINS) [2] [3] [4] has been proposed to enhance lo-
calization availability, state estimation accuracy and robustness
for navigation in complex environments.
Monocular visual odometry in GVINS leverages visual
data and epipolar geometry to estimate camera pose using
only a low-cost camera. Most GVINS implementations utilize
point-feature tracking methods to introduce additional visual
constraints into the batch optimization process for state esti-
mation [2] [3] [4] [5] [6]. Feature detection plays a crucial
role in this process, where corner detectors such as Moravec
[7], Harris [8], Forstner [9], Shi-Tomasi [10], FAST [11]
and ORB [12] are commonly employed to extract features.
Subsequently, detected features are tracked across sequential
frames using Lucas-Kanade (LK) optical flow method [13].
Both the 3D positions of the tracked features and the camera
poses are jointly optimized in a windowed bundle adjustment
framework [14]. In this approach, the state estimation problem
is formulated to minimize the re-projection errors of the
tracked 3D landmarks. Despite its robustness in excluding
noisy visual measurement, its performance is highly sensitive
to the choice of feature detector and matching thresholds.
False matching are more likely in challenging lighting and
imaging conditions, such as under direct sunlight, in shadows,
or when dealing with image scale variance and blurry image.
Additionally, low-texture surfaces can lead to feature sparsity,
further increasing the estimation error in visual navigation.
Alternatively, Direct Methods leverage the gradient magni-
tude and direction of image pixel intensities to bypass these
limitations [15] [16] [17]. With the exploitation of possibly
all information in the image, Direct Visual Odometry (DVO)
has demonstrated superior performance in scenarios with low-
texture scenes [18], challenging lighting conditions [19], and
situations involving camera defocus and motion blur [20], by
minimizing photometric errors.
3D Gaussians have recently emerged as a state-of-the-
art (SOTA) scene representation for 3D reconstruction and
visual SLAM [21] [22]. The 3D Gaussian Splatting (3DGS)
technique employs explicit and differentiable Gaussian ellip-
soids as the sole scene representations, which can be rapidly
rasterized into images. By leveraging an efficient tile-based
CUDA rasterization algorithm [23] and the computational
power of modern graphical processing units (GPUs), 3DGS
can achieve rendering speeds of up to 200 frame-per-second
at 1080p resolution [24]. The combination of fast rendering
and high-quality novel-view synthesis enables 3DGS-based
SLAM to perform accurate real-time camera tracking. Ex-
arXiv:2502.10975v1  [cs.RO]  16 Feb 2025

<!-- page 2 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
2
tensive research of 3DGS has been conducted for indoor
mapping and trajectory estimation, using either only a camera
[22] [24] [25] or a camera integrated with LiDAR or IMU
[26] [27] [28]. However, the integration of 3DGS with GNSS
for navigation in large-scale, complex outdoor environments
remains a research gap.
To address the aforementioned concerns, we present GS-
GVINS, a tightly-integrated GNSS-Visual-Inertial Navigation
System augmented by 3DGS. Our system exploits the raw
measurement error sources of 3DGS and integrates seam-
lessly with GNSS RTK tight integration factors, INS factors,
feature-tracking based visual factors, motion constraints, and
outlier rejection algorithms within a non-linear factor graph
optimization (FGO) framework. Six degree-of-freedom (DOF)
transformations between the local and global frames are jointly
optimized using a sliding-window approach. To account for
the fluctuation in 3DGS rendering quality during the frontend
tracking phase, we introduce a weighting scheme for 3DGS
factors based on the backend mapping loss of keyframes.
Furthermore, to ensure a consistent and more stable Gaussian
map when the vehicle approaches extreme dynamic states, we
propose a Gaussian map pruning mechanism guided by the
relative pose translation and ray’s accumulated opacity. The
key innovations and contributions of this paper are highlighted
as follows:
• A monocular 3DGS-augmented GNSS-Visual-Inertial
Navigation System based on non-linear factor graph op-
timization is proposed. This is the first application which
applies analytical Jacobians of SE(3) camera pose with
respect to 3D Gaussians within a GVINS framework.
• A weighting scheme for 3DGS factors within the graph
optimization process is proposed, leveraging the L1 pho-
tometric loss from 3DGS keyframe mapping. This en-
sures that the importance of image rendering performance
for camera tracking can be effectively managed based on
the quality of current Gaussian map.
• A motion-aware 3D Gaussian pruning mechanism is
proposed to remove unstable Gaussians from the map
when the vehicle approaches extreme dynamic states.
This ensures the quality of the 3D Gaussian map is
maintained during near-stationary motion or rapid mo-
tion variations, providing more accurate observations for
camera tracking.
II. RELATED WORKS
A. GNSS-Visual-Inertial Tight Integration
To achieve a consistently accurate and reliable navigation
solution in complex real-world scenarios, multi-sensor fusion
is widely recognized as the most effective approach. The
tightly coupled integration of GNSS, visual, and inertial data
enables the joint optimization of estimated parameters by
fusing raw measurements and leveraging all available error
sources from each sensor.
In recent years, multiple GVINS frameworks have been
developed and published. [29] represents the first attempt to in-
tegrate raw GNSS measurements of pseudorange and Doppler
shift into an optimization-based visual-inertial SLAM sys-
tem. It demonstrates superior performance compared to exist-
ing visual-inertial SLAM and GNSS Single-Point-Positioning
(SPP), especially in GNSS-denied environments. GVINS [3]
introduces a coarse-to-fine initialization technique that accu-
rately establishes the real-time transformation between global
measurements and local states, significantly reducing the
GNSS state initialization time. Extensive evaluations validate
its accuracy and robustness compared to other SOTA visual-
inertial-odometry (VIO) systems. However, GVINS supports
only the SPP algorithm for GNSS factors, limiting its ability
to fully exploit the potential of GNSS positioning. To better
utilize the advantages of INS, IC-GVINS [4] augments visual
feature tracking and landmark triangulation by incorporating
precise INS information. This integration significantly im-
proves GVINS performance in high-dynamic conditions and
complex environments. However, IC-GVINS only performs
GNSS solution-level integration with raw inertial and visual
measurements in a batch optimization framework. To fully
harness the capabilities of GNSS, GICI-LIB [2] incorporates
nearly all GNSS measurement error sources in its sensor
fusion. Its RTK-RRR processing mode delivers superior pose
estimation performance compared to other SOTA GNSS-
Visual-Inertial navigation systems, particularly in challenging
environments.
Despite these innovations, the primary focus of advance-
ments has been on GNSS and INS. However, further improve-
ment of GNSS or INS performance is increasingly challenging
and presents a bottleneck. Meanwhile, visual sensors remain
low-cost alternatives, but visual-based navigation still heavily
relies on feature tracking and the minimization of reprojection
errors. These areas require further attention to enhance overall
navigation capabilities in GVINS. With the rapid evolution
of graphical computing resources, more sophisticated visual
representations and computer graphics techniques are being
developed and can now be efficiently applied to address a
variety of engineering challenges. These advancements open
new opportunities to push the boundaries of visual navigation.
B. 3DGS in SLAM
Traditional SLAM systems rely on points [30], surfels
[31] or voxel grids [32] as scene representations, enabling
direct and fast computations. However, these methods often
struggle to achieve high-fidelity mapping due to their fixed
spatial resolution and the lack of correlation among the 3D
primitives. Neural-based SLAM [33] [34] [35] offers improved
mapping quality but suffers from computationally expensive
training processes, making it unsuitable for many real-time
applications. In contrast, 3DGS has demonstrated superiority
in efficient, real-time, and high-resolution image rendering.
This is achieved by using expressive anisotropic 3D Gaussians
as scene representations and leveraging tile-based rasterization
techniques. As a result, 3DGS-based SLAM systems outper-
form traditional SLAM methods by enabling rapid photo-
realistic rendering, and achieve superior performance and
efficiency in both mapping and tracking. GS-SLAM [22] is the
first to incorporate real-time 3DGS into the SLAM pipeline

<!-- page 3 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
3
Fig. 1: Overview of the 3DGS module: The module performs 3D Gaussian optimization (orange arrows) and pose optimization
(purple arrows), each operating in its own thread and intercommunicating.
using RGB-D rendering. It integrates an adaptive expansion
strategy to add or remove noisy 3D Gaussians, which enhances
the mapping performance. Additionally, GS-SLAM introduces
an effective coarse-to-fine method for selecting reliable 3D
Gaussians, enhancing camera pose optimization. MonoGS [24]
represents the first application of 3DGS in real-time monoc-
ular SLAM. It replaces the offline Structure-from-Motion
(SfM) process used in the original 3DGS algorithm with
direct optimization against 3D Gaussians, taking advantage
of their wide convergence basin for robust camera tracking.
MonoGS also introduces a geometric verification and regu-
larization technique to address ambiguities in incremental 3D
dense reconstruction. SplaTAM [25] proposes a single RGB-
D camera SLAM framework that utilizes a silhouette mask to
capture the presence of scene density. Both online tracking
and mapping evaluations show that SplaTAM outperforms
existing methods. Despite these advancements, 3DGS-based
SLAM faces challenges, including error accumulation during
pose tracking, scale ambiguity in monocular 3DGS SLAM,
and degradation of pose tracking caused by inconsistent or
erroneous 3D Gaussian maps. Our navigation system takes
advantage of the tight integration of 3DGS with GNSS and
INS to address these challenges.
III. SYSTEM OVERVIEW
GS-GVINS adopts most of its system architecture, including
the initialization algorithm, marginalization, and FGO pipeline
(specifically for the tight integration of GNSS RTK, INS, and
feature-based visual factors) from [2]. Additionally, we intro-
duce a 3DGS module to support 3DGS factors. The system
comprises I/O hardware controllers, data decoders/encoders,
data streamers, and estimation processors. Each component
operates in its own thread, enabling real-time, pseudo-real-
time, and post-processing configurations. As the implementa-
tions of other factors are thoroughly documented in the system
manual of [2], this paper focuses on the 3DGS factor and its
implementation.
Figure 1 illustrates the architecture of the 3DGS module,
which can be modularized into data preprocessing, 3DGS
initialization, 3DGS forward, backward and keyframing. Each
incoming RGB image in the sequence is masked using binary
gradients (based on a preset threshold) to exclude weak edges.
If the 3D Gaussian map has not been initialized, the first
frame is used for initialization. With the initial estimated pose
TTT 0
CW obtained from GNSS/IMU and visual initialization, a
dense RGB-D point cloud is generated to establish the initial
colors and 3D positions (coarse depths are initialized and
will be optimized) of the Gaussian in the world frame. After
initializing each Gaussian’s scaling, rotation, and opacity,
frustum culling removes any Gaussians outside the camera’s
field of view. Overlapping Gaussians along each pixel’s line of
sight are then sorted by depth. The 3D Gaussian ellipsoids are
projected onto the 2D image plane using the estimated camera
pose and projection matrix, and the tile-based parallel CUDA
rendering synthesizes the final pixel colors. An L1 photometric
loss is computed by comparing the rendered image with the
ground-truth image. During back-propagation, the Jacobians
of the Gaussian attributes and the camera pose are evaluated.
For mapping, the positional gradients enable adaptive den-
sity control of the 3D Gaussians, allowing Gaussians to be

<!-- page 4 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
4
Fig. 2: Factor graph structure in GS-GVINS.
split or cloned to better represent fine geometric details. A
small keyframe window is maintained, adding new keyframes
for novel views and removing outdated ones. New Gaussians
are also inserted to capture additional scene details, while
redundant Gaussians are pruned. The 3D Gaussian map within
the current window is then jointly optimized.
With the 3D map constructed in this manner, pose tracking
is performed by re-rendering the current viewpoint, comparing
it against the incoming frame, and minimizing the photometric
loss through camera pose optimization.
IV. METHODOLOGY
The key innovations of this paper are demonstrated by
focusing on the elaboration of the 3DGS algorithm and its
integration into sensor fusion.
In this section, we first introduce the 3DGS-augmented
navigation. Next, we elaborate on the weighting scheme for
the 3DGS factor based on the 3DGS mapping loss. Finally, we
present the motion-aware Gaussian map pruning algorithm.
A. 3DGS-augmented Navigation
In GS-GVINS, a factor graph representation is employed to
describe the non-linear Least-Squares (LSQ) problem, where
the estimated parameters and measurements are represented
as nodes, and the residuals are represented as edges. Figure
2 illustrates the factor graph of the LSQ problem in GS-
GVINS, where the Gaussian states are represented by yellow
rectangles as they are optimized independently from the FGO.
The solution of the LSQ problem can be defined by (1).
ˆχ = arg min
χ
(
∥zp −Hpχ∥2 + ∥zr −hr(χr, χI)∥2
+ ∥zI −hI(χI)∥2 + ∥zc −hc(χc, χI)∥2
+ ∥zGS −hGS(χI)∥2
)
(1)
χr = [BtT
r ,WG pT ,WG vT , dfr, N s
rrb,i]T
(2)
χI = [W pT , qW
B
T ,W vT , ba, bg]T
(3)
χc = [Btc, qB
C ,W pl]T
(4)
The state vector is defined as χ = [χr, χI, χc], where
the subscripts r, I, c and GS correspond to GNSS receiver,
INS, camera, and 3DGS, respectively. Their full elements are
shown in (2), (3) and (4). BtT
r represents the GNSS antenna
extrinsic parameters, WGpT and WGvT denote the position
and velocity of the vehicle body in Earth-Centered Earth-
Fixed (ECEF) frame, dfr represents the frequencies for each
satellite system, N s
rrb,i refers to single-difference carrier phase
ambiguities. For the INS states, W pT , qW
B
T and W vT represent
the position, orientation and velocity of the vehicle body in
East-North-Up (ENU) world frame. Its origin is defined as the
first estimated GNSS position during system initialization. ba
and bg are the biases of accelerometer and gyroscope. For the
feature-based camera states, Btc and qB
C represent the camera’s
extrinsic parameters, and W pl refers to the 3D positions of
landmarks in ENU frame. z denotes the measurements and h
are the non-linear measurement models. zp and Hp represent
the pseudo-measurements and linearized Jacobian of the prior
information, which are computed during marginalization. Note
that zc, hc and χc only refer to the information related
to the image-feature-based reprojection error factor. In GS-
GVINS, 3D Gaussian parameters are optimized independently
from the FGO, considering only the INS states in the 3DGS
measurement model.
3DGS factor applies the principle of differentiable 3DGS
to provide additional information for pose estimation. In our
monocular case, the L1 photometric error can be modeled
based on the current Gaussian map and the estimated camera
pose, illustrated in (5).
Ephotometric =
I(G, TCW ) −¯I

1
(5)
where I(G, TCW ) represents the rendered image from the
3D Gaussian map G based on TCW . And ¯I is the ground truth
image from observation. In the loss computation, optimized
affine brightness parameters are utilized to account for varying
exposure conditions. An RGB boundary mask is applied to
focus the penalty on edge pixels while disregarding non-
edge regions. Additionally, the opacity parameter derived from
rendering is incorporated to penalize pixels with low-opacity
values, ensuring a more robust loss evaluation. We optimize
the estimated camera pose by minimizing the photometric
error in (5).
In the forward process of 3DGS, the RGB-D point clouds
extracted from the image are initialized as 3D anisotropic
Gaussians G. In our outdoor monocular setting, depth esti-
mates are initialized with a large integer value, added by noise
with high variance. Each Gaussian Gi is characterized by its
optical properties: color ci and opacity αi, and its geometric
properties: mean (3D position of center) µi
W and covariance
(ellipsoidal shape) Σi, defined in world space.
Gi(x) = e−1
2 (x)T Σ−1
i
(x)
(6)

<!-- page 5 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
5
The view-dependent radiance described by spherical harmon-
ics (SHs) are omitted in this work for performance efficiency
and simplicity.
After 3D Gaussian initialization, the rasterization iterates
over each Gi and project it from 3D world space G(µW , ΣW )
into 2D image space G′(µI, ΣI) through a projective trans-
formation, illustrated by (7).
µI = π(TCW · µW ),
ΣI = JW ΣW W T JT
(7)
where π is the projection matrix and TCW ∈SE(3) represents
the camera pose that transform world frame into camera frame.
J is the Jacobian of the affine approximate of the projective
transformation and W is the rotation part of TCW . Given the
3D position of the center of each Gi, we formulate a list N that
sorts all the overlapping Gaussians in 3D camera-view space
with respect to the ray of each pixel, based on their depths. By
traversing N from front to back, the final pixel color in the 2D
image space is synthesized through the volumetric rendering
along a ray, as shown in (8)
C =
N
X
n=1
cnα′
n
n−1
Y
j=1
(1 −α′
j)
(8)
α′
n = αn × e−1
2 (x′−µ′
n)T Σ′−1
n
(x′−µ′
n)
(9)
where cn is the optimized color of Gn ∈N, α′
n is the
opacity value in projected Gaussian space, explained by (9),
and α′
j represents the opacity of each previous Gaussian in
the list. The per-pixel contribution from the listed Gaussians
are decayed in the order of sequence, by the transmittance
based on all previous blended Gaussians. In (9), x′ and
µ′
n are the coordinates of image pixel and the projected
Gaussian ’s center. To facilitate the rasterization efficiency,
tile-based parallel rendering is applied in CUDA programming
architecture.
In the backward process of 3DGS, the calculated loss
value is used to compute the gradients through backward-
propagation.
For mapping, both optical parameters and geometric pa-
rameters of 3D Gaussians are optimized through iterative
rendering against the training view and first-order Stochastic
Gradient Descent (SGD) techniques. Since the Gaussians’
attributes do not have correlation with parameters of GNSS or
INS, the mapping is conducted independently from the FGO.
Keyframes are selected based on the co-visibility of Gaussian
map and the relative translation of vehicle. If the intersection
of constructed Gaussians between the current frame and the
last keyframe drops below a threshold, or if the relative
translation since the last keyframe surpass the median depth of
the current Gaussian map, the current frame is registered as a
new keyframe. A small window of keyframes is maintained to
capture the scene in the surrounding environment. A keyframe
will be removed from the window if its overlap with the latest
keyframe drops below a threshold, along with its associated
3D Gaussians, which will also be removed from the map.
Furthermore, we prune redundant Gaussians based on their
visibility. In the outdoor monocular case, accurately estimating
the 3D positions of many Gaussians remains challenging, and
those associated with dynamic objects should be excluded
from the map. These Gaussians usually violate multi-view
consistency and will quickly vanish during optimization. To
refine the 3D Gaussians map during optimization, we follow
the same procedure as the original 3DGS work for adaptive
density control. We further adopt the isotropic regularization
techniques from [24] to penalize the Gaussians which are
highly elongated along the viewing direction and cause arte-
facts during rendering, further improve the tracking accuracy.
During tracking, both camera pose and the affine bright-
ness parameters are optimized. To reduce the computational
overhead of automatic differentiation, the explicit analytical
Jacobian of SE(3) camera pose with respect to 3D Gaussians,
derived from [24], is integrated into our FGO pipeline. In (7),
both µI and ΣI are differentiable with respect to TCW . The
minimal Jacobians of 6-D world-camera transformation can be
derived using Lie algebra as:
∂µI
∂TCW
= ∂µI
∂µC
DµC
DTCW
(10)
∂ΣI
∂TCW
= ∂ΣI
∂J
∂J
∂µC
DµC
DTCW
+ ∂ΣI
∂W
DW
DTCW
(11)
The derivatives of camera pose lay on the manifold for min-
imal parametrization, ensuring the dimension of the Jacobians
match the exact degrees of freedom of SE(3). The partial
derivatives with respect to T ∈SE(3) can be described by
(12)
Df(T )
DT
△= lim
τ→0
Log(f(Exp(τ)◦) ◦f(T )−1)
τ
(12)
where τ ∈se(3) represents the 6-D twists on the tangent space
of manifold, ◦is a group composition, Log and Exp are the
logarithmic and exponential mappings between Lie Group and
Lie algebra. Therefore, the Jacobians of camera pose in (10)
and (11) are derived as:
DµC
DTCW
= [I
µ×
C],
DW
DTCW
=


0
−W×
:,1
0
−W×
:,2
0
−W×
:,3


(13)
where × represents the skew symmetric matrix form, W×
:,i
indexes the ith column of the rotation matrix.
B. 3DGS Mapping Loss based Weighting Scheme
The 3DGS factor computes analytical Jacobians of camera
pose using the L1 photometric loss by rendering the 3D
Gaussian map at viewpoint and compare it with the ground
truth image. The quality of rendering strongly depends on
the accuracy of the optimized 3D Gaussian map. A Gaussian
map optimized with a higher mapping loss introduces more
rendering errors in camera tracking, resulting in less reliable
pose estimation.
In the window-based mapping strategy, all keyframes within
the window contribute to the map optimization at a given
viewpoint. However, in complex large-scale and continuous
driving scenarios, it is infeasible to optimize a 3D Gaussian
from multiple viewing directions. The optimized 3D Gaussians

<!-- page 6 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
6
may appear distorted when rendered from different viewing
directions later. This is a typical issue that can degrade tracking
performance due to the resulting erroneous rendering.
To account for the uncertainties from the 3DGS mapping
optimization, we introduce a weighting scheme for 3DGS
factor based on its mapping loss. The full weighting scheme
can be described by (14).
σ2
3DGS,i =
X
∀k∈W
Ek
photometric + λisotropicEisotropic
(14)
Eisotropic =
|G|
X
j=1
∥sj −˜sj · 1∥1
(15)
where σ2
3DGS,i is the variance of 3DGS factor at the ith
frame, W is the current keyframe window, k represents the kth
keyframe, Ek
photometric is the photometric loss computed using
(5), λisotropic is the pre-defined isotropic weighting factor, sj
and ˜sj are scaling parameters and its mean for the jth Gaussian
in the map. This considers the contribution of error sources
from all the keyframes in the current window and the isotropic
regularization of over-stretched Gaussians.
C. Motion-Aware 3D Gaussian Pruning
When the vehicle approaches extreme dynamic states, such
as near-static conditions or high-dynamic motion, camera
tracking performance during non-keyframes can be impacted
by noisy rendering. Erroneous renderings are more likely to
occur due to positional errors in 3D Gaussians and the predom-
inantly forward-oriented motion of the vehicle. These errors
may cause 3D Gaussians to appear misplaced or excessively
close to the viewpoint, leading to rendering artifacts such as
blurriness, holes, or floating elements, as shown in Figure 8.
The optimization of the 3D Gaussian map relies on all
keyframes within the current window. However, the proba-
bilities of keyframe registration decrease significantly when
the vehicle undergoes rapid deceleration or comes to a halt.
As a result, the current Gaussian map is updated much less
frequently, as no modifications occur within the keyframe
window. Additionally, due to the inherent scale ambiguity in
a monocular setting, the optimized map may appear distorted
when viewed from different angles. This distortion becomes
more pronounced during high-dynamic motion, where larger
displacements between sequential viewpoints introduce rel-
atively more significant rendering artifacts, particularly in
forward-oriented motion. As a result, rendering from extreme
dynamic viewpoints can lead to high photometric loss during
tracking.
To address this issue, we propose a motion-aware 3D Gaus-
sian pruning mechanism. This approach removes redundant
3D Gaussians from the map that could introduce noise in
the rendering during extreme motion. The pruning process
is guided by the computed relative translation between the
consecutive frames, and the accumulated opacity α from the
rays that capture the most recently inserted 3D Gaussians
(from the latest keyframe). This mechanism is illustrated in
(16).
Gstatic =





G\Gredundant,
if ti < λmin or ti > λmax,
∀i ∈[n −9, n],
G,
otherwise.
(16)
Gredundant =
[
k∈R
{Gj |
X
j∈Rk
αj ≥0.5}
(17)
where ti is the relative translation of the ith frame with
respect to its previous frame, and λmin and λmax are the
predefined thresholds of relative translation for low-dynamic
and high-dynamic case, respectively; G denotes the set of 3D
Gaussians in the current map, while Gredundant represent the
Gaussians to be pruned. R is the set of rays that intersect the
newly inserted 3D Gaussians, with k denoting an individual
ray from this set. Rk represents the set of Gaussians along
the kth ray that are positioned in front of the newly inserted
Gaussians. Finally, Gj refers to each individual 3D Gaussian
along the kth ray.
If the displacement between consecutive viewpoints falls
below or exceeds predefined thresholds, we infer that the
vehicle is undergoing extreme dynamic motion. As described
in (16), 3D Gaussians in the current map are pruned by
removing those originating from older keyframes and pre-
venting the integration of new 3D Gaussians from recent
keyframes. This approach ensures that the latest keyframe,
which generally provides a clearer view and a more accurate
camera-to-3D Gaussian orientation, is prioritized in extreme
motion scenarios.
After sorting the 3D Gaussians along the camera ray, we
identify and remove those that cause the accumulated opacity
α to reach 0.5 before blending with new Gaussians. This
process reduces the density of closer 3D Gaussians in the
vehicle’s heading direction, preventing unstable Gaussians
from accumulating in regions with near-stationary motion or
rapid motion variations.
V. EXPERIMENTS AND RESULTS
To validate our proposed system, we conduct extensive
experiments using both the open-sourced UrbanNav [36]
dataset and self-collected data, covering a variety of GNSS
environments ranging from open-sky to deep urban scenarios.
All datasets are recorded in kilometers-scale large outdoor
environments with several vehicle turns, stops, and elevation
variations. The evaluation focuses on absolute pose error
(APE) and compares GS-GVINS results with GICI-LIB and
IC-GVINS, which are SOTA open-sourced GVINSs.
This section is structured as follows: First, we briefly
describe the experimental setup for our self-collected data and
evaluation. Next, we detail the dataset characteristics and the
configurations used in the evaluation. Finally, we present and
analyze the results.
A. Experimental Setup
For our self-collected data, we utilized a multi-sensor in-
tegrated platform equipped with hardware-based time syn-
chronization (milliseconds level accuracy) to collect GNSS

<!-- page 7 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
7
rover, inertial, and visual raw measurements. The platform
consists of a u-blox F9P GNSS receiver, an ICM-20948
MEMS IMU, and two FLIR Blackfly-S RGB cameras. An
STM32 Micro-Controller-Unit (MCU) manages system time
synchronization using the GPS PPS signal, as well as data
streaming and storage. A u-blox patch antenna is connected
to the u-blox F9P receiver. The GNSS, inertial and visual
data are collected in 1 Hz, 100 Hz and 10 Hz respectively.
For groundtruth generation, the system is equipped with a
NovAtel SPAN system comprising an OEM7 receiver and an
Epson EG320N IMU. The time synchronization algorithm and
the full details can be found in [37]. GNSS reference station
data were collected from a Trimble R9 receiver located on the
rooftop of the CCIT building at the University of Calgary. The
baseline distance between the reference station and the rover
consistently remained below 10 km. All data are stored on a
laptop using ROS 2 [38].
Fig. 3: Multi-sensor data collection system mounted on a
vehicle (left), with top view (upper-right) and bottom view
(lower-right) of the control box.
To evaluate our proposed system, we replay the ROS
bag data in a pseudo-real time configuration. The data are
published via ROS topics and streamed into the GS-GVINS
in the order of their message timestamps. The entire system
operates within the environment of Advanced Computing
Cluster (ARC) at the University of Calgary. During runtime,
the system utilizes 4 out of 7 units of an NVIDIA A100-MIG
GPU with 40 GB VRAM, 4 cores of an Intel(R) Xeon(R)
Silver 4316 CPU running at 2.30 GHz, and 70 GB of RAM.
B. Dataset Characteristics and Configuration
Figure 4 provides views of scenes from each validation
dataset. For consistency, we categorize the environments based
on urbanization levels into three types: open-sky, sub-urban,
and urban. Two self-collected datasets are used to evaluate
system performance in open-sky scenario; the first part of the
UrbanNav-Deep dataset and the UrbanNav-Medium dataset
are used to assess performance in sub-urban environments;
while the second part of the UrbanNav-Deep dataset and
the UrbanNav-Harsh dataset are used for evaluation in urban
settings.
Both Open-sky A and B datasets are collected near the
University of Calgary, where A is captured near the CCIT
building and B is primarily collected on a highway. The Sub-
urban A dataset covers an area near Hung Hom in Hongkong,
(a) Open-Sky A
(b) Open-Sky B
(c) Sub-urban A
(d) Sub-urban B
(e) Urban A
(f) Urban B
Fig. 4: Scenes from the validation datasets, featuring diverse
environments.
featuring side buildings, small tunnels, and loops. The Sub-
urban B focus on the area near Tsim Sha Tsui Substation in
the central area of Hongkong, encompassing medium-height
buildings and numerous dynamic objects. The Urban A and B
datasets are collected in deep urban environments where sev-
eral GNSS signal blockages occur, resulting in frequent NLOS
and multipath signals. Urban A is collected in the central area
of Hung Hom, characterized by narrow roads surrounded by
densely packed building and heavy traffic. Urban B is collected
in an ultra-dense urban canyon with narrow streets flanked by
high-rising buildings, bridges, pedestrians and dense vehicle
traffic.
Figure 5 illustrates the GNSS status in both sub-urban and
urban environments. The available satellite number, Geometric
Dilution of Precision (GDOP), and GNSS RTK fix status are
visualized. To exclude GNSS outlier measurements, thresholds
are applied: a minimum GNSS signal-to-noise ratio (SNR) of
30.0 dB and a minimum satellite elevation of 7.0 degrees.
When the GNSS solution is unavailable, the satellite number
is plotted as zero. A fix status of 3 indicates a float ambiguity
resolution for the GNSS RTK solution.
C. Evaluation of Localization
To validate the performance of GS-GVINS, we computed
the APE of the navigation solutions from GS-GVINS against
the ground truth and benchmark them against the tightly
coupled solutions from GICI-LIB and IC-GVINS. All datasets

<!-- page 8 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
8
Fig. 5: Visualization of available satellite number, GDOP, and GNSS RTK fix status for Sub-urban and Urban scenarios.
are evaluated with successful GNSS/IMU and visual ini-
tializations. Table 1 summarizes the APE results for each
environment type using the mentioned datasets. The values
in bold indicate the lowest APE in each category. For Urban-
B data tested with IC-GVINS, the solution is not applicable
as the estimator diverges.
TABLE I
3D APE OF GS-GVINS, GICI, and IC-GVINS
(METERS/DEGREES).
Data
Environment
Data ID
APE (Translation [m] / Rotation [◦])
GS-GVINS
GICI
IC-GVINS
Open-sky
A
0.073 / 3.748
0.071 / 3.293
0.085 / 3.844
B
0.084 / 3.154
0.081 / 3.028
0.266 / 3.623
Sub-urban
A
1.873 / 3.622
1.952 / 3.562
6.644 / 5.562
B
3.255 / 0.864
3.934 / 1.012
11.596 / 2.477
Urban
A
4.521 / 1.646
4.584 / 1.647
7.465 / 0.913
B
6.043 / 2.015
7.003 / 1.957
- / -
It is evident that GS-GVINS outperforms both GICI-LIB
and IC-GVINS in the majority of the datasets. In GNSS-
friendly Open-sky A and B environments, GS-GVINS and
GICI demonstrate competitive performance. In these scenar-
ios, GNSS RTK provides high-accuracy absolute positioning,
which predominantly influences the solution estimation. Since
the GNSS RTK factor implementations in GS-GVINS is
adopted from GICI-LIB, the improvements in absolute pose
estimation are limited. Additionally, as inertial and visual
navigation have minimal influences in these scenarios, the
following discussion will focus on the sub-urban and urban
environments.
In both the sub-urban and urban datasets, GS-GVINS
demonstrates significant improvements of 3D translation es-
timation over GICI and IC-GVINS. In Sub-urban A, GS-
GVINS reduces 3D translation RMSE by 4.05 % and 71.81
%, compared to GICI and IC-GVINS, respectively. In Sub-
urban B, GS-GVINS further reduces translation errors by
17.26 % (GICI) and 71.93 % (IC-GVINS), along with a 14.62
Fig. 6: Visualization of 3DGS under different conditions. The
left, middle and right column represents 3D Gaussians map,
rendered image and groundtruth image, respectively.
% reduction in 3D rotation RMSE. In the more challeng-
ing environment, Urban A, GS-GVINS improves translation
accuracy compared to GICI and IC-GVINS by 1.37 % and
39.44 %, respectively. Despite GS-GVINS achieves a slightly
better rotation estimation than GICI, the lowest 3D rotation
RMSE (0.913 degrees) is achieved by IC-GVINS. In Urban
B, GS-GVINS achieves a 13.71 % reduction in 3D translation
errors compared to GICI, further demonstrating its robustness
in translation estimation under challenging conditions.

<!-- page 9 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
9
(a) Featureless environment
(b) Static environment
Fig. 7: Opacity shading: A jet colormap represents the opti-
mized opacity of projected Gaussians.
(a) 3D Gaussians map
(b) Rendered image
(c) Time shading
(d) Ground truth
Fig. 8: 3DGS in high dynamic condition
Despite of the improvements of 3D translation estimation
by GS-GVINS, its improvements in 3D rotation estimation re-
main limited. This limitation in rotation accuracy is partly due
to the depth initialization strategy for the monocular setting. To
compromise the complex and dynamic driving environments,
we uniformly initialize the depth of each new Gaussian based
on the approximated median depth of the scene with added
noise, then refine it during rasterization. However, this coarse
depth initialization, combined with the lack of multi-view
optimization for 3D Gaussians, can introduce rotation scale
errors, which limit the accuracy of 3D rotation estimation in
GS-GVINS.
Figure 9 illustrates the 3D absolute translation error over
GPS time for each dataset in the ENU directions. The red
dashed vertical line represents the initialization time for GS-
GVINS and GICI, as GS-GVINS adopts its initialization
algorithm from GICI, resulting in identical completion times.
The 3D Gaussian map is initialized in the next image frame
after feature-based visual initialization. The blue vertical line
marks the initialization time for IC-GVINS. Due to the dif-
ferent initialization strategies of GICI and IC-GVINS, and the
dependency of IC-GVINS on the availability of GNSS RTK
solutions, their initialization times vary. In Figure 9 (d), the
absolute translation error of IC-GVINS is not included, as the
estimator diverges.
In Figure 9 (a), a notable reduction in the vertical (up)
translation error from GS-GVINS is observed indicated by
the red dashed box. This improvement can also be observed
near the right-side tail of the 3D trajectory in Figure 10.
The corresponding driving scenario is depicted in the first,
fourth and fifth rows of Figure 6, where the vehicle is going
through an underground tunnel. Extreme camera exposure
conditions are observed during this time period, where the
tracked visual features can be lost or mismatched. However,
3DGS maintains high-fidelity rendering despite these condi-
tions. Furthermore, the fourth row of Figure 6 highlights a
featureless scene, which makes visual feature detection and
tracking challenging. Despite this, 3DGS accurately renders
the scene while preserving fine details. Figure 7 (a) displays
the rendering from the opacity shader, where the opacity values
extracted from the Gaussian rendering are visualized using a
jet colormap (warmer colors indicate higher opacity values,
and colder colors indicate lower opacity values). The comb-
like patterns on the tunnel walls are precisely textured by
the optimized opacity values. Although GNSS performance
deteriorates during this period, as reflected in Figure 5, the
3DGS factor effectively bounds the vertical translation errors.
In Figure 9 (b), GS-GVINS significantly reduces translation
errors in the ENU directions compared to GICI between
95610s and 95660s (black box). This period coincides with
frequent fluctuations in GDOP values and a sharp decline in
satellite numbers, indicating poor GNSS performance. The
corresponding scenario is shown in the third row of Figure
6, where the vehicle is reaching to a stop. The motion-aware
pruning mechanism aids in achieving accurate rendering in
this case. Figure 7 (b) presents the opacity map, revealing
a well-defined outline and structure of objects in the scene,
which helps establish better zero-motion constraints from the
3DGS factor. Around 95680s, a sharp increase in GDOP value,
combined with fewer than five available GNSS satellites,
results in large error spikes in both east and north directions.
However, GS-GVINS significantly reduces the east and verti-
cal positioning error (red box) compared to both GICI and IC-
GVINS, bringing the vertical positioning error close to zero.
The second row of Figure 6 illustrates this environment, where
dense tree canopies and advertising boards cast shadows over
the left side of the street, making feature tracking difficult.
Figures 9 (c) and (d) display ENU error plots for more
urbanized environments (Urban A and B). In Urban A, pro-
longed low satellite availability and extreme GDOP values
(indicated by the red box) force the estimator to rely more
on inertial and visual navigation. The last row of Figure 6
characterizes the street scene, featuring narrow roads, high-
rise buildings, and parked cars, all of which are mapped by
3DGS. In this scenario, the 3DGS factor effectively constraints
the growth of translation errors, as GNSS service is largely
unavailable. As shown in Figure 9 (c), GS-GVINS improves
both east and north translation errors to GICI and IC-GVINS
as show in the red box. However, during the periods indicated
by the black boxes, 3DGS factor seems to hinder translation
error reduction. As the DOP values improve and the num-
ber of available satellites increases during this period, the
influence of the 3DGS factor likely diminishes, weakening
the impact of GNSS-based positioning during optimization.
GS-GVINS achieves a significant reduction in north-direction
error compared to GICI, as highlighted by the blue box,

<!-- page 10 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
10
(a) Sub-urban A
(b) Sub-urban B
(c) Urban A
(d) Urban B
Fig. 9: 3D Absolute Translation Error in ENU directions over GPS time.
improving vehicle turning performance as shown in the third
row of Figure 10 (near the end of the trajectory). Most of
the cases discussed earlier also occur in Urban B, where GS-
GVINS constrains the east and up error highlighted by the
red box in Figure 9 (d). During these periods, with varying
illumination caused by urban shadows and the presence of
numerous pedestrians, the 3DGS factor effectively controls
error drift in feature-based visual-inertial navigation.
VI. CONCLUSION
In this paper, we proposed the GS-GVINS, a tightly-
integrated
GNSS-Visual-Inertial
Navigation
System
aug-
mented by 3DGS, incorporating a mapping loss-based weight-
ing scheme for 3DGS factor and a motion-aware Gaussian
pruning mechanism. Comprehensive evaluation of pose es-
timation have been conducted using the data from com-
plex, large-scale outdoor environments. Experimental results
demonstrate significant improvements, particularly in transla-
tion accuracy, compared to SOTA sensor-fusion libraries GICI
and IC-GVINS.
Additionally, the results highlight the robustness of 3DGS
in handling challenging conditions such as extreme exposure,
featureless scenes, and shadowed areas—scenarios that typi-
cally degrade feature-tracking-based navigation. Furthermore,
the pruning mechanism enhances mapping quality during ex-
treme vehicle dynamics, improving pose estimation accuracy
in both low- and high-dynamic motion.
Since GS-GVINS operates in a pseudo-real-time configu-
ration, we believe it can achieve real-time performance with
sufficient computational resources. Moreover, GS-GVINS has
the potential for high-fidelity map reconstruction in typi-
cal driving scenarios, particularly when supplemented with
accurate depth information from external sensors. A well-
reconstructed 3D Gaussian map can further enhance GNSS
performance, especially by improving outlier detection, where
GNSS signal rays are traced against the 3D reconstructed
Gaussian map to identify NLOS signals or multipath.
ACKNOWLEDGMENTS
We acknowledge the support of the Natural Sciences and
Engineering Research Council of Canada (NSERC); we also
extend our appreciation to Cheng Chi for providing support
and discussion regarding the GICI library.
REFERENCES
[1] Lachapelle G, Falkenberg W, Casey M. ”Use of phase data for accu-
rate differential GPS kinematic positioning,” Bulletin Geodesique. 1987
Dec;61:367-77.
[2] Chi C, Zhang X, Liu J, Sun Y, Zhang Z, Zhan X. Gici-lib: A
gnss/ins/camera integrated navigation library. IEEE Robotics and Au-
tomation Letters. 2023 Oct 16.
[3] Cao S, Lu X, Shen S. GVINS: Tightly coupled GNSS–visual–inertial
fusion for smooth and consistent state estimation. IEEE Transactions on
Robotics. 2022 Jan 4;38(4):2004-21.
[4] Niu X, Tang H, Zhang T, Fan J, Liu J. IC-GVINS: A robust, real-time,
INS-centric GNSS-visual-inertial navigation system. IEEE robotics and
automation letters. 2022 Nov 23;8(1):216-23.

<!-- page 11 -->
IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT. PREPRINT VERSION FEBRUARY 2025
11
Fig. 10: 3D and 2D trajectory plots for Sub-urban A, Sub-
urban B, Urban A and Urban B (from top to bottom).
[5] Gakne PV, O’Keefe K. Tightly-coupled GNSS/vision using a sky-
pointing camera for vehicle navigation in urban areas. Sensors. 2018 Apr
17;18(4):1244.
[6] Song J, Li W, Duan C, Zhu X. R2-GVIO: A Robust, Real-Time GNSS-
Visual-Inertial State Estimator in Urban Challenging Environments. IEEE
Internet of Things Journal. 2024 Mar 20.
[7] Moravec HP. Obstacle avoidance and navigation in the real world by a
seeing robot rover. Stanford University; 1980.
[8] Harris CG, Pike JM. 3D positional integration from image sequences.
Image and Vision Computing. 1988 May 1;6(2):87-90.
[9] F¨orstner W. A feature based correspondence algorithm for image match-
ing. ISPRS ComIII, Rovaniemi. 1986:150-66.
[10] Shi J. Good features to track. In1994 Proceedings of IEEE conference
on computer vision and pattern recognition 1994 Jun 21 (pp. 593-600).
IEEE.
[11] Rosten E, Drummond T. Machine learning for high-speed corner de-
tection. InComputer Vision–ECCV 2006: 9th European Conference on
Computer Vision, Graz, Austria, May 7-13, 2006. Proceedings, Part I 9
2006 (pp. 430-443). Springer Berlin Heidelberg.
[12] Rublee E, Rabaud V, Konolige K, Bradski G. ORB: An efficient
alternative to SIFT or SURF. In2011 International conference on computer
vision 2011 Nov 6 (pp. 2564-2571). Ieee.
[13] Lucas BD, Kanade T. An iterative image registration technique with an
application to stereo vision. InIJCAI’81: 7th international joint conference
on Artificial intelligence 1981 Aug 24 (Vol. 2, pp. 674-679).
[14] Fraundorfer F, Scaramuzza D. Visual odometry: Part ii: Matching,
robustness, optimization, and applications. IEEE Robotics & Automation
Magazine. 2012 Feb 16;19(2):78-90.
[15] Engel J, Sch¨ops T, Cremers D. LSD-SLAM: Large-scale direct monoc-
ular SLAM. InEuropean conference on computer vision 2014 Sep 6 (pp.
834-849). Cham: Springer International Publishing.
[16] Engel J, Koltun V, Cremers D. Direct sparse odometry. IEEE transactions
on pattern analysis and machine intelligence. 2017 Apr 12;40(3):611-25.
[17] Zhao C, Tang Y, Sun Q, Vasilakos AV. Deep direct visual odome-
try. IEEE transactions on intelligent transportation systems. 2021 Apr
16;23(7):7733-42.
[18] Lovegrove S, Davison AJ, Ibanez-Guzm´an J. Accurate visual odometry
from a rear parking camera. In2011 IEEE Intelligent Vehicles Symposium
(IV) 2011 Jun 5 (pp. 788-793). IEEE.
[19] Alismail H, Kaess M, Browning B, Lucey S. Direct visual odometry
in low light using binary descriptors. IEEE Robotics and Automation
Letters. 2016 Dec 5;2(2):444-51.
[20] Newcombe RA, Lovegrove SJ, Davison AJ. DTAM: Dense tracking and
mapping in real-time. In2011 international conference on computer vision
2011 Nov 6 (pp. 2320-2327). IEEE.
[21] Charatan D, Li SL, Tagliasacchi A, Sitzmann V. pixelsplat: 3d gaussian
splats from image pairs for scalable generalizable 3d reconstruction.
InProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition 2024 (pp. 19457-19467).
[22] Yan C, Qu D, Xu D, Zhao B, Wang Z, Wang D, Li X. Gs-slam:
Dense visual slam with 3d gaussian splatting. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024
(pp. 19595-19604).
[23] Kerbl B, Kopanas G, Leimk¨uhler T, Drettakis G. 3d gaussian splatting
for real-time radiance field rendering. ACM Trans. Graph.. 2023 Aug
1;42(4):139-.
[24] Matsuki H, Murai R, Kelly PH, Davison AJ. Gaussian splatting slam.
InProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition 2024 (pp. 18039-18048).
[25] Keetha N, Karhade J, Jatavallabhula KM, Yang G, Scherer S, Ramanan
D, Luiten J. SplaTAM: Splat Track & Map 3D Gaussians for Dense RGB-
D SLAM. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition 2024 (pp. 21357-21366).
[26] Jiang C, Gao R, Shao K, Wang Y, Xiong R, Zhang Y. Li-gs: Gaussian
splatting with lidar incorporated for accurate large-scale reconstruction.
IEEE Robotics and Automation Letters. 2024 Dec 26.
[27] Lang X, Li L, Zhang H, Xiong F, Xu M, Liu Y, Zuo X, Lv J. Gaussian-
LIC: Photo-realistic LiDAR-Inertial-Camera SLAM with 3D Gaussian
Splatting. arXiv preprint arXiv:2404.06926. 2024 Apr 10.
[28] Xiao R, Liu W, Chen Y, Hu L. LiV-GS: LiDAR-Vision Integration for
3D Gaussian Splatting SLAM in Outdoor Environments. IEEE Robotics
and Automation Letters. 2024 Nov 25.
[29] Liu J, Gao W, Hu Z. Optimization-based visual-inertial SLAM tightly
coupled with raw GNSS measurements. In2021 IEEE International Con-
ference on Robotics and Automation (ICRA) 2021 May 30 (pp. 11612-
11618). IEEE.
[30] Sandstr¨om E, Li Y, Van Gool L, Oswald MR. Point-slam: Dense neural
point cloud-based slam. InProceedings of the IEEE/CVF International
Conference on Computer Vision 2023 (pp. 18433-18444).
[31] Behley J, Stachniss C. Efficient surfel-based SLAM using 3D laser range
data in urban environments. InRobotics: science and systems 2018 Jun
26 (Vol. 2018, p. 59).
[32] Muglikar M, Zhang Z, Scaramuzza D. Voxel map for visual SLAM.
In2020 IEEE International Conference on Robotics and Automation
(ICRA) 2020 May 31 (pp. 4181-4187). IEEE.
[33] Li R, Wang S, Gu D. DeepSLAM: A robust monocular SLAM sys-
tem with unsupervised deep learning. IEEE Transactions on Industrial
Electronics. 2020 Mar 25;68(4):3577-87.
[34] Sucar E, Liu S, Ortiz J, Davison AJ. imap: Implicit mapping and
positioning in real-time. InProceedings of the IEEE/CVF international
conference on computer vision 2021 (pp. 6229-6238).
[35] Zhu Z, Peng S, Larsson V, Xu W, Bao H, Cui Z, Oswald MR, Pollefeys
M. Nice-slam: Neural implicit scalable encoding for slam. InProceedings
of the IEEE/CVF conference on computer vision and pattern recognition
2022 (pp. 12786-12796).
[36] Hsu LT, Huang F, Ng HF, Zhang G, Zhong Y, Bai X, Wen W. Hong
Kong UrbanNav: An open-source multisensory dataset for benchmarking
urban navigation algorithms. NAVIGATION: Journal of the Institute of
Navigation. 2023 Dec 21;70(4).
[37] Uprety S, Lyu Z, Zhou Z, Zambra RJ, Vishwanath A, Lee R, Li B, Yang
H. How much do Na¨ıve and Hardware Time Synchronization Methods
Affect Multi-Sensor Integrated Positioning?. In Proceedings of the 37th
International Technical Meeting of the Satellite Division of The Institute
of Navigation (ION GNSS+ 2024) 2024 Sep 20 (pp. 1506-1518).
[38] Macenski S, Foote T, Gerkey B, Lalancette C, Woodall W. Robot
operating system 2: Design, architecture, and uses in the wild. Science
robotics. 2022 May 11;7(66):eabm6074.
