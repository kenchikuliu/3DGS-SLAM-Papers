<!-- page 1 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
1
HI-SLAM2: Geometry-Aware Gaussian SLAM for
Fast Monocular Scene Reconstruction
Wei Zhang1, Qing Cheng2,4, David Skuddis1, Niclas Zeller3, Daniel Cremers2,4 and Norbert Haala1
Better
Ours
(a) 3DGS Map
(b) Mesh
(c) Renders
Fig. 1: Our method builds a 3D Gaussian Splatting (3DGS) map (a) to reconstruct complex scenes using only monocular
input. We are able to extract accurate and detailed mesh reconstructions (b) with high-quality renderings (c). The right figure
illustrates the trade-off between geometric accuracy and visual appearance, as some methods prioritize one aspect over the
other. Compared to existing methods, our approach excels among RGB-only methods, denoted by ▲, and also surpasses recent
RGB-D methods, denoted by  , in both geometry and appearance reconstruction.
Abstract—We present HI-SLAM2, a geometry-aware Gaussian
SLAM system that achieves fast and accurate monocular scene
reconstruction using only RGB input. Existing Neural SLAM or
3DGS-based SLAM methods often trade off between rendering
quality and geometry accuracy, our research demonstrates that
both can be achieved simultaneously with RGB input alone.
The key idea of our approach is to enhance the ability for
geometry estimation by combining easy-to-obtain monocular
priors with learning-based dense SLAM, and then using 3D
Gaussian splatting as our core map representation to efficiently
model the scene. Upon loop closure, our method ensures on-
the-fly global consistency through efficient pose graph bundle
adjustment and instant map updates by explicitly deforming
the 3D Gaussian units based on anchored keyframe updates.
Furthermore, we introduce a grid-based scale alignment strategy
to maintain improved scale consistency in prior depths for
finer depth details. Through extensive experiments on Replica,
ScanNet, Waymo Open, ETH3D SLAM and ScanNet++ datasets,
we demonstrate significant improvements over existing Neural
SLAM methods and even surpass RGB-D-based methods in
both reconstruction and rendering quality. The project page and
source code are available at https://hi-slam2.github.io/.
Index Terms—Visual SLAM, Dense Reconstruction, Deep
Learning for Visual Perception.
I. INTRODUCTION
D
ENSE 3D scene reconstruction from imagery remains
one of the most fundamental challenges in computer
vision, robotics, and photogrammetry. Achieving real-time and
accurate 3D reconstruction from images alone can enable
numerous applications, from autonomous navigation to mobile
1Institute for Photogrammetry and Geoinformatics, University of Stuttgart,
Germany {firstname.lastname}@ifp.uni-stuttgart.de
2Technical University of Munich, Germany
3Karlsruhe University of Applied Sciences, Germany
4Munich Center for Machine Learning, Germany
surveying and immersive AR. While many existing solutions
rely on RGB-D [1], [2], [3], [4], [5] or LiDAR sensors [6],
[7], [8], [9], [10], these approaches have inherent limitations.
LiDAR systems require expensive hardware setups and an
additional camera for capturing color information, while RGB-
D sensors suffer from limited operational range and sensitive
to varying lighting conditions. Vision-based monocular scene
reconstruction thus offers a promising lightweight and cost-
effective alternative.
The fundamental challenge in monocular 3D reconstruction
stems from the lack of explicit scene geometry measure-
ments [11]. Traditional visual SLAM methods [12], [13], [14],
[15], [16], [17] developed over decades and typically pro-
vide only sparse or semi-dense map representations, proving
insufficient for detailed scene understanding and complete
reconstruction. While dense SLAM approaches [18], [19],
[20] attempt to address this limitation through per-pixel depth
estimation, they remain susceptible to significant depth noise
and struggle to achieve complete, accurate reconstructions.
Recent advances in deep learning have revolutionized
many key components of 3D reconstruction, including optical
flow [21], [22], depth estimation [23], [24], and normal esti-
mation [24], [25]. These improvements have been integrated
into SLAM systems through monocular depth networks [26],
multi-view stereo techniques [27], and end-to-end neural
approaches [28]. However, even with these advancements,
current systems often produce reconstructions with artifacts
due to noisy depth estimates, limited generalization capability,
or excessive computational requirements. The emergence of
Neural SLAM methods, particularly those based on neural
implicit fields [5], [29], [30] and 3D Gaussian Splatting
(3DGS) [31], [32], [33], has shown promising results. Yet
these approaches typically prioritize either rendering quality
arXiv:2411.17982v3  [cs.RO]  2 Feb 2026

<!-- page 2 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
2
Image
I
Pose
T
Tracking
NeRF/
3DGS θ
Joint
Optimization
ΔT
Δθ
Image
I
Pose
T
Tracking
NeRF/
3DGS θ
ΔT
Δθ
Features
f
Joint
Optimization
(a) Map-centric e.g. MonoGS
(b) Hybrid Design
Depth
d
Init.
Fig. 2: Comparison of SLAM paradigms: while map-centric
SLAM employs a unified map representation for both tracking
and joint optimization, the hybrid design approach utilizes
learning-based features and bundle adjustment for tracking,
producing depth as an intermediate scene representation. This
is then used to initialize the 3D map and supervise the joint
optimization of camera poses and scene geometry.
or geometry accuracy, creating an undesirable trade-off. Our
work addresses this limitation by simultaneously improving
both aspects without compromise either. As shown in Fig. 1,
our approach achieves superior performance across both ge-
ometry accuracy and rendering quality, surpassing not only
RGB-based methods but also RGB-D-based approaches.
In this paper, we aim to advance the state-of-the-art in dense
monocular SLAM for 3D scene reconstruction. We present HI-
SLAM2, a geometry-aware Gaussian Splatting SLAM system
that achieves accurate and fast monocular scene reconstruction
using RGB input alone. The key idea of our approach lies
in enhancing geometry estimation by combining monocular
geometry priors with learning-based dense SLAM, while lever-
aging 3DGS as our compact map representation for efficient
and accurate scene modeling. As depicted in Fig. 2, unlike
map-centric SLAM methods, we adopt a hybrid approach that
utilizes learning-based dense SLAM to generate depth as a
proxy, which serves both to initialize scene geometry and to
guide map optimization. This hybrid design decouples the map
training from tracking while seamlessly recoupling pose and
map later during joint optimization, ensuring both efficiency
and accuracy.
For depth estimation, we introduce a scale-grid based align-
ment strategy that effectively addresses scale distortions in
monocular depth priors, significantly improving depth estima-
tion accuracy. Our surface depth rendering employs unbiased
depth calculation at ray-Gaussian intersection points [34], en-
abling more precise surface fitting. To enhance surface recon-
struction, particularly in low-texture regions, we incorporate
monocular normal priors into 3DGS training, ensuring the con-
sistency of reconstructed surfaces. By deforming 3D Gaussian
units using keyframe pose updates, we enable efficient online
map updates, boosting both speed and flexibility in mapping.
Furthermore, unlike hash grid based methods [35], [36] that
require a predefined scene boundary, our approach allows the
map to grow incrementally as new areas are explored without
any prior knowledge of scene size.
We validate our approach through extensive experiments on
both synthetic and real-world datasets, including Replica [37],
ScanNet [38], Waymo Open [39], ETH3D SLAM [40], and
ScanNet++ [41]. Our method achieves substantial improve-
ments in both reconstruction and rendering quality compared
to existing Neural SLAM methods, surpassing even RGB-D-
based methods in accuracy. Our method is particularly well-
suited for real-time applications that demand rapid and reliable
scene reconstruction in scenarios where depth sensors are
impractical.
In summary, our work advances the state-of-the-art in dense
monocular SLAM through the following contributions:
• A geometry-aware Gaussian SLAM framework achieving
high-fidelity RGB-only reconstruction through efficient
online mapping and joint optimization of camera poses
and Gaussian map.
• An enhanced depth estimation approach leveraging ge-
ometry priors and improved scale alignment to compen-
sate for monocular prior distortions and enable accurate
surface reconstruction.
• A balanced system achieving superior performance in
both geometry and appearance reconstruction across syn-
thetic and real-world datasets.
II. RELATED WORKS
A. Depth Estimation
Depth estimation can be broadly categorized into multi-
view and monocular approaches. Classic multi-view methods
rely on geometry principles, utilizing techniques such as patch
matching [42] or cost aggregation [43]. Recent learning-based
approaches MVSNet [23] and DeepMVS [44] have greatly
improved the consistency of depth estimation across video
sequences. In parallel, monocular depth estimation has seen
remarkable progress, with methods like MiDaS [45] and
OmniData [24] demonstrating impressive generalization across
diverse datasets. However, these monocular approaches suffer
from scale ambiguity, producing depth maps with inconsistent
scales between frames. Our work addresses this limitation
through a novel scale-grid alignment strategy that estimates
spatially varying depth scales, enabling more accurate depth
estimation compared to previous method [36] that relied on a
single, rigid scale transformation.
B. Surface Reconstruction
Surface
reconstruction
typically
follows
a
two-stage
pipeline: camera pose estimation through Structure-from-
Motion (SfM) [46], [42], followed by multi-view stereo [47],
[48] for dense reconstruction. While widely adopted, these
methods are computationally intensive and often produce in-
complete reconstructions due to depth estimation uncertainties.
Neural implicit representations [49] and their variants [50],
[51], [52] have demonstrated high-quality reconstruction ca-
pabilities but remain computationally demanding. Recent ad-
vances in 3DGS [53] offer more efficient rendering compared
to NeRF-based approaches, and its variants [54], [34] show
promising geometry reconstruction capabilities. Our approach
leverages 3DGS for efficient scene representation while main-
taining high-quality reconstruction, effectively addressing the
speed-quality trade-off inherent in previous methods.

<!-- page 3 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
3
Online Tracking
Online Loop Closing
Offline Refinement
Continuous Mapping
Keyframe
Buffer
Keyframe
Selection
Extract Mono. 
Priors
LC Detection 
Sim(3)-based
PGBA
Post-Keyframe
Selection
Full BA
Joint Refinement 
3DGS & Poses
RGB Stream
Init Gaussians
{µi, i, i, ci}
Differentiable
Surface Rasterization
Densification
Keyframe 
Estimation
T, D
Local 
Updades
T, D, s
Global 
Updates
T, D
Map 
update
JDSA
 Local BA
Refined 
Keyframe 
Attributes
I, D, T
Relative pose edge
Loop closures
Fig. 3: System Overview: Our framework consists of four key stages: online camera tracking, online loop closing, online
mapping, continuous mapping, and offline refinement. The camera tracking is performed using a recurrent-network-based
approach to estimate camera poses T and generate depth maps D from RGB input. Depth priors are incorporated into the
tracking process through our proposed Joint Depth and Scale Alignment (JDSA) strategy improving depth estimation accuracy.
For 3D scene representation, we use 3DGS to model scene geometry, enabling efficient online map updates. These updates are
integrated with Sim(3)-based pose graph Bundle Adjustment (BA) for online loop closing, allowing for scale drift correction
via scale updates ∆s, and achieving both fast updates and high-quality rendering. In the offline refinement stage, camera
poses and scene geometry undergo full BA, followed by joint optimization of Gaussian primitives and camera poses to further
enhance global consistency.
C. Dense Visual SLAM
Dense SLAM methods traditionally relied on volumetric
representations such as Truncated Signed Distance Functions
(TSDF) [55] or 3D voxel grids [56], [57] for scene geom-
etry modeling. The emergence of neural implicit representa-
tions [49] enabled high-quality scene reconstructions within
dense visual SLAM [5], [29], but at significant computa-
tional cost and often requiring RGB-D input. Recent monoc-
ular approaches like NICER-SLAM [58] and HI-SLAM [36]
have demonstrated promising results using only RGB input.
The 3DGS-based methods Splat-SLAM [33] and GLORIE-
SLAM [59] showcase the potential of 3DGS for real-time
dense reconstruction. However, these methods still face chal-
lenges in balancing computational efficiency with reconstruc-
tion quality. Our work addresses these limitations through
key innovations in depth estimation, scene consistency, and
computational efficiency, achieving both high-quality geome-
try and appearance reconstruction in real-time.
III. METHODS
Our system is designed to enable fast and accurate cam-
era tracking and scene reconstruction from monocular RGB
input. As illustrated in Fig. 3, the system comprises four
key components: an online tracker, an online loop closing
module, a continuous mapper, and an offline refinement stage.
The online camera tracker (Sec. III-B) leverages a learning-
based dense SLAM frontend to estimate camera poses and
depth maps. Global consistency and real-time performance are
achieved through the online loop closure module (Sec. III-C),
which combines loop closure detection with efficient Pose
Graph Bundle Adjustment (PGBA). For scene representation,
we employ 3DGS (Sec. III-D), enabling efficient online map
construction, updates, and high-quality rendering. The offline
refinement stage (Sec. III-E) enhances reconstruction quality
through full BA and joint optimization of Gaussian map and
camera poses ensures optimal global consistency. The final
mesh is generated by fusing rendered depth maps through
TSDF fusion.
A. Comparison to HI-SLAM
The current HI-SLAM2 system represents a significant
advancement over our previous work HI-SLAM [36], with
improvements across multiple aspects that enhance tracking
accuracy and reconstruction quality substantially. The key
improvements can be summarized as follows:
• Depth Prior Integration: We propose a novel spatially-
adaptive scale-grid alignment strategy that effectively
addresses nonlinear scale distortions in monocular depth
priors. In contrast to HI-SLAM’s single scale align-
ment, our 2D grid-based method with bilinear interpo-
lation accommodates spatially varying distortions. This
is achieved without introducing dependencies between
depth pixels, as each depth value is treated as an in-
dependent variable. This design preserves the efficiency
of solving the optimization problem using the Schur
complement. The improved scale alignment enhances the
accuracy of depth estimation, thereby facilitating Gaus-

<!-- page 4 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
4
sian initialization and optimizing the map reconstruction
with depth supervision.
• Map Representation: We replace HI-SLAM’s neural
implicit field representation with 3DGS, transitioning
from an implicit to explicit representation. This change
provides several benefits: (1) significantly faster ren-
dering, (2) efficient online map updates through direct
Gaussian primitive updates rather than neural network
weight optimization, (3) incremental map growth without
predefined scene boundaries, and (4) enhanced geometry
preservation as evidenced by the 1.54cm improvement in
reconstruction accuracy on the Replica dataset (Table IV).
• Hierarchical
Optimization: Our system employs a
multi-stage optimization pipeline that includes online
tracking with local BA, online loop closure with Sim(3)-
based PGBA, and global full BA. Last but not least,
the joint refinement of both Gaussian map parameters
and camera poses, which couples the tracking frontend
with the mapping backend more tightly. In contrast, HI-
SLAM only performs pose optimization in the tracking
frontend, which can lead to inconsistencies between
estimated poses and map geometry. This hierarchical
approach reduces the Absolute Trajectory Error (ATE) by
29.3% compared to online tracking alone (Table X) on
the Replica dataset, yielding a globally more consistent
reconstruction.
B. Online Tracking
Our online tracking module builds upon a learning-based
dense visual SLAM method [20] to estimate camera poses and
depth maps of keyframes. By leveraging dense per-pixel infor-
mation through a recurrent optical flow network, our system
can robustly track the camera in challenging scenarios, such
as low-textured environments and fast movements. To match
per-pixel correspondences among all overlapping frames, we
construct a keyframe graph (V, E) which represents the co-
visibility relationships between every pair of keyframes. The
graph nodes V correspond to keyframes, each containing a
pose T ∈SE(3) and an estimated depth map d. Graph edges
E connect keyframes with sufficient overlap, determined by
their optical flow correspondences. To synchronize the esti-
mated states with other modules aiding continuous mapping
and online loop closing, a keyframe buffer is maintained to
store the information of all keyframes and their respective
states.
The tracker begins with keyframe selection where each
incoming frame is assessed to determine if it should be
selected as a keyframe. This decision is based on the average
flow distance relative to the last keyframe calculated through
a single pass of the optical flow network [21] and a prede-
fined threshold dflow. For selected keyframe, we extract the
monocular priors, including depth and normal priors, through a
pretrained neural network [24]. While the depth priors are used
directly by the tracker module to facilitate depth estimation,
the normal priors are used by the scene representation mapper
for 3D Gaussian map optimization as extra geometry cues.
Following [20], we initialize the system state after collecting
Ninit = 12 keyframes. The initialization performs bundle
*
=
Mono prior
Scales
Aligned prior
GT
Fig. 4: Example of scale alignment of monocular depth.
adjustment (BA) on a keyframe graph, where edges connect
keyframes within an index distance of 3, ensuring sufficient
overlap for reliable convergence. Since a monocular system
does not have an absolute scale, we normalize the scale by
setting the mean of all keyframe depths to one. This scale is
then hold as the system scale by fixing the poses of the first
two keyframes in subsequent BA optimizations. Afterwards,
each time a new keyframe is added, we perform local BA to
estimate the camera poses and depth maps of the keyframes in
the current keyframe graph. Edges between the new keyframe
and neighboring keyframes with sufficient overlap are added to
the graph. With the optical flow prediction f, the reprojection
error is minimized by using the flow-predicted target, denoted
as ˇpij = pi + f, and the current reprojection induced by
camera poses and depths as source. The local BA optimization
problem can be formulated as:
arg min
T,d
X
(i,j)∈E
∥ˇpij −Π(TijΠ−1(pi, di))∥2
Σij
(1)
where Tij = Tj · T−1
i
denotes the rigid body transformation
from keyframe i to keyframe j, and di refers to the depth
map of keyframe i in inverse depth parametrization, Π and
Π−1 represent the camera projection and back-projection
functions, respectively. Σij is a weight matrix with diagonals
representing the prediction confidences from the optical flow
network. The confidence effectively ensures the robustness
of the optimization by reducing the influence of the outliers
caused by occlusions or low-texture regions. Depth estimates
in under-confident regions, where the depth cannot be accu-
rately estimated, are further refined using monocular depth
priors in the subsequent step.
Incorporate Monocular Depth Prior: To overcome the
challenge of depth estimation in difficult areas such as low-
textured or occluded regions, we incorporate the easy-to-
obtain monocular depth priors [24] into the online tracking
process. In the RGB-D mode of [20], depth observations
are directly used to compute the mean squared error during
BA optimization. However, we can not directly follow the
same manner because predicted monocular depth priors exhibit
inconsistent scales. To address this, [36] proposes estimating a
depth scale and an offset for each depth prior as optimization
parameters. Although this approach helps align an overall prior
scale, we found that it is not sufficient to fully correct the scale
distortions inherent in monocular depth priors.
To further improve this, we propose estimating a 2D depth
scale grid with coefficients si of dimension (m, n) for each
depth prior ˇdi. The depth scale at every pixel can be obtained
by bilinear interpolation Bi(pi, si) on the grid based on its
four surrounding grid coefficients. This spatially-varying scale
formulation makes it more flexible to align the prior depth with

<!-- page 5 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
5
the estimated depth by BA and helps to reduce the influence
of noise in the depth prior. Using the sampled depth scales,
the scale-aligned depth prior can be obtained as ˇdi·Bi(pi, si).
Then we formulate the depth prior factor rd as follows:
rd = ∥ˇdi · Bi(pi, si) −di∥2
(2)
The grid resolution is set to 2 × 2. Higher resolutions may
introduce instability, particularly in low-texture regions where
optical flow predictions have low confidence, resulting in
insufficient weighting in the reprojection error term for each
grid tile. The scale coefficients are initially set to ones. After
the system converges, the scale coefficients for new depth
priors are initialized using those from the previous depth
priors.
As reported in [36], directly incorporating the depth prior
factor into the BA optimization, i.e. jointly optimizing the
camera poses, depths, and scale coefficients, can make the
system prone to scale drift and hinder convergence. To address
this, similar to the approach in [36], we introduce a joint
depth and scale alignment (JDSA) module to estimate the prior
scales separately with the following objective:
arg min
s,d
X
(i,j)∈E
∥ˇpij −Π(TijΠ−1(pi, di))∥2
Σij+
X
i∈V
∥ˇdi · Bi(pi, si) −di∥2
(3)
By interleaving the JDSA optimization with the local BA
optimization, we ensure that the system scale remains stable
and the depth prior is well-aligned, providing depth estimation
with a better initial guess. We use the damped Gauss-Newton
algorithm to solve the optimization problem. For the sake
of the optimization efficiently, we separate scale and depth
variables as follows:
 B
E
ET
C
 ∆s
∆d

=
v
w

(4)
where B, E, C are the blocks of the Hessian matrix and v,
and w are the gradient vector of the linearized system. Since
the dimension of matrix B is much smaller than C, we can
solve the system efficiently by first solving for ∆s and then
∆d using the Schur complement.
∆s = (B −EC−1ET )−1(v −EC−1w)
∆d = C−1(w −ET ∆s)
(5)
Matrix C is diagonal since the scale alignment in Eq. 2 is
applied to the depth prior rather than the depth variables.
This preserves the independence between the depth variables
allowing us to invert C efficiently as C−1 = diag( 1
c1 , .., 1
cn ).
Fig. 4 shows an example of the scale alignment for monocular
depth priors. Note that the estimated spatially-varying scales
result in well-aligned depth prior with respect to the ground
truth depth.
C. Online Loop Closing
While our online tracker can robustly estimate camera
poses, measurement noise inevitably accumulates over time
and travel distance, which leads to pose drift. Additionally,
monocular systems are prone to scale drift due to inherent
scale unobservability. To correct both pose and scale drifts
and enhance the global consistency of the 3D map, our
online loop closing module searches for potential loop closures
and performs global optimization on the entire history of
keyframes using a Sim(3)-based PGBA first proposed in [36].
Loop Closure Detection: Loop closure detection is per-
formed in parallel to the online tracking. For each selected
new keyframe, we calculate the optical flow distances dof
between the new keyframe and all previous keyframes. We
define three criteria to select loop closure candidates. First,
dof must fall below a predefined threshold τflow, ensuring
sufficient co-visibility for reliable convergence in recurrent
flow updates. Second, orientation differences based on cur-
rent pose estimations should remain within a threshold τori.
Finally, the frame index difference must exceed a minimum
threshold τtemp beyond the current local BA window. When all
criteria are met, we add edges connecting the keyframe pairs
in forward and revert re-projection directions in our keyframe
graph.
Sim(3)-Based Pose Graph Bundle Adjustment: When
loop closure candidates are identified, inspired by the effi-
ciency of PGBA in [60], [36], we choose pose graph BA over
full BA to balance computational efficiency with accuracy.
To address scale drift, we adopt Sim(3) representations for
keyframe poses, enabling per-keyframe scale correction as
proposed in [61]. Before each optimization run, we convert
the latest pose estimates from SE(3) to Sim(3) group and
initialize scales with ones. The pixel warping step follows
Eq. 1, with the SE(3) transformation replaced by a Sim(3)
transformation.
Constructing the pose graph involves connecting poses
through relative pose edges. Following [60], [36], we de-
rive relative poses from dense correspondences of inactive
reprojection edges which are retained when their associated
keyframes leave the sliding window of local BA. These dense
correspondences offer a reliable basis for computing relative
poses because they have been refined for multiple iterations
when they are active in the sliding window. The reprojection
error term from Eq. 1 is used, but here the optimization focuses
solely on relative poses ˇTij under the assumption that depth
estimates are accurate. To incorporate uncertainty, we estimate
variances Σrel
ij
for the relative poses based on the adjustment
theory [62] as:
Σrel
ij = (J∆Tij −r)T Σij(J∆Tij −r)(JT ΣijJ)−1
(6)
where J, r, and ∆Tij are the Jacobian, reprojection residuals,
and relative pose update from the previous iteration, respec-
tively. These variances serve as weights in PGBA. The final
objective of PGBA is to minimize the sum of relative pose
factors and reprojection factors:
arg min
T,d
X
(i,j)∈E∗
∥ˇpij −Π(TijΠ−1(pi, di))∥2
Σij+
X
(i,j)∈E+
∥log(ˇTij · Ti · T−1
j )∥2
Σrel
ij
(7)

<!-- page 6 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
6
where E∗represents detected loop closures, and E+ denotes
the set of relative pose factors. To ensure the convergence of
the optimization and account for potential outliers in relative
pose factors, we apply a damped version of Gauss-Newton
algorithm to find the optimal solution as follows:
H = H + ϵ · I + λ · H
(8)
where H denotes the Hessian matrix. The damping factor
ϵ = 10−4 and regularization factor λ = 10−1 serve two
critical functions: preventing convergence to local minima and
improving numerical conditioning, while maintaining rapid
convergence. Following optimization, we convert the opti-
mized poses back to the SE(3) for subsequent tracking. The
depth maps are scaled according to the corresponding Sim(3)
pose transformations. Additionally, as detailed in Sec. III-D,
we update the 3D Gaussian primitives based on the pose
updates of their anchor keyframes.
D. 3D Scene Representation
We adopt 3DGS [53] as our scene representation model-
ing scene appearance and geometry. Unlike implicit neural
representations such as NeRF, 3DGS provides an explicit
representation that enables efficient online map updates and
high-quality rendering. The scene is represented by a set of 3D
anisotropic Gaussians G = {gi}M
i=1, where each 3D Gaussian
unit is defined as:
gi(x) = e−(x−µi)⊤Σ−1
i
(x−µi),
(9)
where µi ∈R3 denotes the Gaussian mean and Σi ∈R3×3
represents the covariance matrix in world coordinates. The
covariance matrix Σi is decomposed into orientation Ri and
scale Si = diag{si} ∈R3×3, such that Σi = RiSiST
i RT
i .
Each Gaussian also carries attributes for opacity oi ∈[0, 1]
and color ci ∈R3. Unlike the original 3DGS [53], we simplify
the color representation by using direct RGB values instead
of spherical harmonics, reducing optimization complexity. To
handle view-dependent color variations, we employ exposure
compensation during the offline refinement stage (Sec. III-E).
The rendering process projects these 3D Gaussians onto the
image plane using perspective transformation:
µ′
i = π(Ti · µi),
Σ′
i = JWΣiWT JT
(10)
where J represents the Jacobian of the perspective transforma-
tion and W denotes the rotation matrix of keyframe pose Ti.
After depth-based sorting of the projected 2D Gaussians, pixel
colors and depths are computed through α-blending along each
ray from near to far:
ˆC =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj),
ˆD =
X
i∈N
diαi
i−1
Y
j=1
(1 −αj)
(11)
where N represents the set of Gaussians intersecting the ray,
ci is the color of the i-th Gaussian, and αi represents the
pixel translucency calculated by evaluating the opacity of i-th
Gaussian at the intersection point.
Unbiased Depth: Previous works [31], [32] directly use
the depth at the Gaussian mean, which introduces estimation
biases when rays intersect the Gaussian at points distant from
its mean. Following [34], we compute an unbiased depth by
determining the actual ray-Gaussian intersection point along
the ray direction. This depth is calculated by solving the
planar equation at the intersection of the ray and Gaussian
surface. Since all rays from the same viewpoint that intersect
a given Gaussian are co-planar, the intersection equation needs
to be solved only once per Gaussian. This approach maintains
the computational efficiency of splat-based rasterization while
significantly improving depth accuracy. We demonstrate the
benefits of this unbiased depth computation through ablation
studies in Sec. IV-G.
Map Update:The map update process adjusts the 3D Gaus-
sian units based on the updates of keyframe pose to ensure
global consistency of the 3D map. This update happens both
online during the Sim(3)-based PGBA and offline during the
global full BA. To enable rapid and flexible updates to the
3D scene representation, we deform the mean, orientation,
and scale of each Gaussian unit. Specifically, means and
orientations are transformed according to the relative SE(3)
pose change between the previous and updated keyframes,
while scales are adjusted using the scale factors derived from
the Sim(3) pose representation.
The update equations for each Gaussian unit are:
µ′
j = (T′−1
i
· Ti · µj)/si,
R′
j = R′−1
i
· Ri · Rj,
s′
j = si · sj
(12)
where µ′
j, R′
j, and s′
j represent the updated mean, orientation,
and scale of the j-th Gaussian, respectively. This transforma-
tion ensures that the geometric relationships between Gaus-
sians are preserved while accommodating the refined keyframe
poses, maintaining the accuracy and completeness of the 3D
reconstruction.
Exposure
Compensation: Real-world captures exhibit
varying exposures across different views due to illumina-
tion changes and view-dependent reflectance. These variations
introduce color inconsistencies that can significantly impact
reconstruction quality. Following [32], [63], we address this
challenge by optimizing per-keyframe exposure parameters
through a 3×4 affine transformation matrix. For a rendered
image ˆI, the exposure correction is formulated as:
ˆI′ = A · ˆI + b
(13)
where A denotes the 3×3 color transformation matrix and
b represents the 3×1 bias vector. During the offline refine-
ment stage, these exposure parameters are jointly optimized
alongside camera poses and scene geometry, as detailed in
Sec. III-E.
Map Management: To ensure that newly observed regions
are well represented, we initialize a set of 3D Gaussian
primitives when each new keyframe is created to populate

<!-- page 7 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
7
the Gaussian map. The initialization process begins by un-
projecting the estimated depth map of the keyframe into 3D
space. Specifically, each pixel’s depth value is back-projected
to generate a 3D point, which serves as the mean µi of
a new Gaussian primitive. The orientation is initialized as
the unit orientation. The initial scale Si is set by finding
the average distance of nearest 3 neighbors to adapt to the
local point density, while opacity oi is initialized to 0.5 to
allow the optimization itself to update. Color attributes ci
are assigned based on the corresponding pixel’s RGB value
from the keyframe. To maintain map compactness and prevent
redundancy, we apply random downsampling with a factor
ψ. This downsampling ensures computational efficiency while
preserving enough spatial coverage. To control map growth,
we implement a pruning strategy that removes Gaussians with
low opacity to eliminate redundant or insignificant primitives.
We reset the opacity values every 500 iterations and perform
interleaved densification and pruning every 150 iterations to
balance map size and quality. A detailed analysis of map size
evolution is presented in Sec. IV-H.
Optimization Losses: The 3DGS representation is opti-
mized using a combination of photometric, geometric, and
regularization losses. The photometric loss Lc measures the L1
difference between the exposure-compensated rendered image
ˆI′ and the observed image I. The depth loss Ld computes the
L1 difference between the rendered depth ˆD and the estimated
depth ¯D from the interleaved BA and JDSA optimization:
Lc =
X
k∈K
|ˆI′
k −Ik|, Ld =
X
k∈K
| ˆDk −¯Dk|
(14)
where K denotes the keyframes in the local window during
online mapping or all keyframes during offline refinement. To
enhance geometric supervision, we incorporate normal priors
into the optimization. The estimated normals are derived from
rendered depth maps using cross products of depth gradients
along image plane axes. The normal loss Ln is defined as a
cosine embedding loss:
Ln =
X
k∈K
|1 −ˆNT
k · ¯Nk|
(15)
To prevent artifacts due to excessively slender Gaussians, we
apply a regularization term to the scale of the 3D Gaussians:
Ls =
X
i∈G
|si −¯si|
(16)
where ¯si denotes the mean scale of the i-th Gaussian, penal-
izing ellipsoid stretching. The final loss combines these terms
with appropriate weights as follows:
L = λcLc + λdLd + λnLn + λsLs
(17)
where λc, λd, λn, and λs are the respective weights. We
optimize Gaussian parameters using the Adam optimizer [64],
performing 10 iterations per new keyframe.
E. Offline Refinement
Following the online processing, we implement three se-
quential offline refinement stages to enhance global consis-
tency and map quality: post-keyframe insertion, full BA, and
joint pose and map refinement.
(a)
(b)
Fig. 5: View coverage analysis in two scenarios: (a) Optimal
case where consecutive keyframes maintain sufficient overlap,
ensuring proper multi-view coverage. (b) Suboptimal case
where newly observed regions in keyframe Kt lack adequate
observations. Our system addresses this by inserting additional
post-keyframes (shown in blue) to enhance view coverage.
Post-Keyframe Insertion: The first refinement stage iden-
tifies regions with insufficient view coverage, particularly
areas near view frustum boundaries. These regions typically
arise when forward camera motion is followed by backward
rotational movement, as illustrated in Fig. 5. During online
processing, keyframe selection relies on average optical flow
between neighboring frames, as view coverage cannot be fully
evaluated without future trajectory information. To identify
under-observed regions in the offline stage, we project each
keyframe’s pixels onto its adjacent keyframes and quantify
the percentage of pixels that fall outside the fields of view
of neighboring keyframes. When this percentage exceeds a
predetermined threshold, we flag the region as having insuffi-
cient observations. Additional keyframes are then inserted at
these locations, and new Gaussian primitives are populated in
the same manner as the keyframes inserted during the online
process. This ensures more complete scene reconstruction and
preserves critical details at scene boundaries.
Full Bundle Adjustment: While our online loop-closing
module achieves global consistency through efficient Sim(3)-
based PGBA, full BA further enhances system accuracy.
PGBA offers superior computational efficiency compared to
full BA, but introduces minor approximation errors when
abstracting dense correspondences into relative pose edges.
Specifically, PGBA computes reprojection factors only for
loop closure edges, while full BA performs comprehensive
optimization by re-computing reprojection factors in Eq. 1
for all overlapping keyframe pairs, including both neighboring
and loop closure frames. As demonstrated in Sec. IV-G, this
improves the global consistency of camera poses and scene
geometry at a finer granularity.
Joint Pose and Map Refinement: The final refinement
stage jointly optimizes the Gaussian map and camera poses
based on the results of full BA. While the online mapping stage
limits optimization iterations per keyframe to maintain real-
time performance, the offline refinement enables comprehen-
sive optimization across all keyframes. To facilitate joint pose
refinement, we compute pose Jacobians during rasterization-
based rendering. Additionally, we also optimize per-keyframe
exposure compensation parameters to ensure a better global

<!-- page 8 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
8
color consistency. Unlike the full BA stage which employs
the Gauss-Newton algorithm, this joint refinement step utilizes
the Adam optimizer [64] with first-order gradient descent,
leveraging our existing mapping pipeline.
IV. EXPERIMENTS
To evaluate the performance of the proposed system,
we conducted extensive experiments on several challenging
datasets, including the synthetic Replica dataset [37] and
the real-world datasets ScanNet [38], Waymo [39], ETH3D
SLAM [40], and ScanNet++ [41]. We begin by providing
implementation details and evaluation metrics, followed by
quantitative and qualitative comparisons on camera tracking
accuracy, geometry, and appearance reconstruction quality
against state-of-the-art baselines. Subsequently, we present
ablation studies to analyze the impact of different design
choices. Finally, we present the runtime performance and map
size analysis.
A. Implementation Details
Our system is implemented using PyTorch [65] and CUDA
for GPU acceleration, with evaluations performed on an
Nvidia RTX 4090 GPU and Intel Core i9-12900K CPU.
For optical flow and geometry prior prediction, we utilize
pretrained models from [20] and [24] respectively. For map
refinement optimization, we use 2000 iterations for the Replica
dataset and 26000 iterations for ScanNet and ScanNet++
datasets, ensuring fair comparison with existing methods. The
loss weights of map optimization remain consistent across all
experiments: color loss (λc) at 0.95, depth loss (λd) at 0.25,
and scale loss (λs) at 10. The normal loss weight (λn) is set
to 0.1 for the Replica dataset and increased to 0.5 for ScanNet
and ScanNet++ datasets to enhance geometric supervision on
real-world data. The downsampling factor ψ is set to 32 across
all experiments, providing an optimal balance between map
size and quality.
B. Datasets
Replica Dataset [37] provides synthetic indoor scenes with
high-quality reconstructions, featuring complex geometry and
textures. We evaluate using eight RGB-D sequences from [5].
The sequences have perfect camera poses and reconstructions
make it ideal for benchmarking dense visual SLAM methods.
ScanNet Dataset [38] offers real-world RGB-D captures for
3D scene reconstruction. Following [36], we use eight se-
quences for tracking evaluation and six additional sequences
for geometry reconstruction assessment, using the RGB-D
reconstructions as ground truth. ETH3D SLAM [40] provides
a diverse set of real-world RGB-D sequences with motion-
capture ground truth poses, featuring challenging conditions
such as extreme lighting variations and complete darkness.
Waymo Open dataset [39] provides real-world outdoor data
with ground truth vehicle poses. The front-view camera images
are used as input for our system. Following the evaluation
protocol of [66], we evaluate the tracking accuracy and ren-
dering quality using 9 sequences. ScanNet++ [41] presents
GT
GLORIE-SLAM
Ours
Oﬃce2
HI-SLAM
Room0
Room2
Fig. 6: Qualitative comparison on geometry reconstruction on
Replica dataset.
a large-scale indoor dataset with laser-scanned ground truth,
enabling evaluation of dense SLAM reconstruction quality.
While the dataset includes both DSLR and iPhone captures, we
specifically evaluate on the iPhone sequences, which present
additional challenges due to their lower image quality.
C. Evaluation Metrics
We evaluate our system’s performance across three key
aspects. Camera tracking accuracy is quantified using Absolute
Trajectory Error (ATE), measuring the precision of estimated
camera poses. For geometry reconstruction quality, we adopt
three metrics from [5]: average accuracy [cm], average com-
pleteness [cm], and completeness ratio [%] (representing the
percentage of reconstruction within 5cm of ground truth).
For appearance quality assessment, we evaluate keyframe
renderings using standard photometric metrics: PSNR (Peak
Signal-to-Noise Ratio), SSIM (Structural Similarity Index),
and LPIPS (Learned Perceptual Image Patch Similarity). In all
result tables, we highlight performance rankings using: first ,
second , and third .
D. Camera Tracking Accuracy
We evaluate the camera tracking accuracy of our system
against state-of-the-art dense visual SLAM methods on in-
door Replica, ScanNet, ETH3D SLAM datasets and outdoor
Waymo datasets, including comparisons with RGB-D based
approaches. The ATE results in Tables I and II demonstrate
the superior tracking accuracy of our system. RGB-D methods
SplatTAM [31] and MonoGS [32], despite having access to
depth measurements for map-based tracking, achieve lower
accuracy than hybrid approaches. Our system, along with
Splat-SLAM [33], represents the class of hybrid methods that

<!-- page 9 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
9
TABLE I: Comparison of camera tracking accuracy for RGB
and RGB-D methods on Replica dataset with results in [cm].
Method
ro-0 ro-1 ro-2 of-0 of-1 of-2 of-3 of-4 Avg.
RGB-D input
NICE-SLAM[29]
0.97 1.31 1.07 0.88 1.00 1.06 1.10 1.13 1.07
ESLAM[67]
0.63 0.71 0.70 0.52 0.57 0.55 0.58 0.72 0.63
Point-SLAM[68]
0.61 0.41 0.37 0.38 0.48 0.54 0.69 0.72 0.52
SplatTAM[31]
0.31 0.40 0.29 0.47 0.27 0.29 0.32 0.55 0.36
MonoGS[32]
0.44 0.32 0.31 0.44 0.52 0.23 0.17 2.25 0.58
RGB input
DROID-SLAM[20] 0.34 0.13 0.27 0.25 0.42 0.32 0.52 0.40 0.33
NICER-SLAM[58]
1.36 1.60 1.14 2.12 3.23 2.12 1.42 2.01 1.88
GLORIE-SLAM[59] 0.31 0.37 0.20 0.29 0.28 0.45 0.45 0.44 0.35
Splat-SLAM[31]
0.29 0.33 0.25 0.29 0.35 0.34 0.42 0.43 0.34
MGS-SLAM[69]
0.36 0.35 0.32 0.35 0.28 0.26 0.32 0.34 0.32
Ours
0.23 0.22 0.19 0.23 0.27 0.25 0.37 0.33 0.26
TABLE II: Camera tracking accuracy for RGB and RGB-D
methods on ScanNet dataset with results in [cm].
Method
0000 0054 0059 0106 0169 0181 0207 0233 Avg.
RGB-D input
NICE-SLAM[29]
12.00 20.90 14.00 7.90 10.90 13.40 6.20 9.00 11.8
ESLAM[67]
7.30 36.30 8.50 7.50 6.50
9.00 5.70 4.30 10.6
Co-SLAM[35]
7.10 12.80 11.10 9.40 5.90 11.80 7.10 6.10 8.90
Point-SLAM[68]
10.20 28.00 7.80 8.70 22.20 14.80 9.50 6.10 14.3
LoopSplat[70]
4.20
7.50
7.50 8.30 7.50 10.60 7.90 5.20 7.70
RGB input
GO-SLAM[71]
5.90 13.30 8.30 8.10 8.40
8.30 6.90 5.30 8.10
GLORIE-SLAM[59] 5.50
9.40
9.10 7.00 8.20
8.30 7.50 5.10 7.50
Splat-SLAM[31]
5.57
9.50
9.11 7.09 8.26
8.39 7.53 5.17 7.58
HI-SLAM[36]
6.43
9.97
7.22 6.56 8.53
7.65 8.43 5.23 7.47
Ours
5.82
8.64
7.30 6.80 8.25
7.41 7.40 4.93 7.07
TABLE III: Camera tracking and rendering results on Waymo
open dataset averaged over 9 sequences.
Metrics
NICER-
SLAM[58]
GLORIE-
SLAM[59]
Photo-
SLAM[72] MonoGS[32]
OpenGS-
SLAM[66]
Ours
ATE [m] ↓
19.59
0.536
19.95
8.529
0.839
0.457
PSNR ↑
12.22
18.83
17.73
21.80
23.99
28.99
SSIM ↑
0.622
0.702
0.741
0.780
0.800
0.872
LPIPS ↓
0.726
0.572
0.674
0.577
0.434
0.219
Fig. 7: Analysis of successful sequences relative to ATE error
thresholds on the ETH3D SLAM dataset.
effectively combine dense SLAM with deep learning founda-
tions. The global BA of DROID-SLAM [20] was enabled in
all experiments. While it employs global BA, our additional
global pose and map joint refinement further improves tracking
accuracy beyond the baseline method. We further compare
our system on the ETH3D SLAM dataset with state-of-the-
art sparse and dense methods, including ORB-SLAM3 [17],
DPVO [73], DPV-SLAM [74], and MASt3R-SLAM [75].
Figure 7 illustrates the cumulative success curves based on
ATE thresholds and ATE statistics. As none of the methods
can successfully track all sequences, we use the Area Under
the Curve (AUC) metric to assess both accuracy and robustness
with an upper ATE threshold of 0.5m. Our system achieves
the highest AUC among all methods and lowest mean and
median ATE. Out of total 61 sequences, 6 sequences failed due
to complete darkness, while 4 sequences encountered tracking
failures caused by lighting changes and view occlusions. These
limitations could potentially be addressed in the future by inte-
grating place recognition for relocalization to reduce the stan-
dard deviation of ATE. Our evaluation on the Waymo Open
dataset (Table III) further validates our approach, where we
achieve the lowest ATE among all competing methods. This
highlights the ability of our system to generalize effectively
to challenging large-scale outdoor environments with complex
scene geometries that typically pose substantial difficulties for
monocular systems.
E. Geometry Reconstruction Quality
TABLE IV: Reconstruction evaluation on Replica dataset for
implicit and explicit rgb methods. Ours surpass other methods
especially large margin in accuracy.
.
Method
Metric
ro-0
ro-1
ro-2
of-0
of-1
of-2
of-3
of-4 Avg.
NeRF-based
NICER-
SLAM[58]
Acc.[cm] ↓
2.53 3.93 3.40 5.49 3.45 4.02 3.34 3.03 3.65
Comp.[cm] ↓
3.04 4.10 3.42 6.09 4.42 4.29 4.03 3.87 4.16
Comp.Rat[%]↑88.75 76.61 86.10 65.19 77.84 74.51 82.01 83.98 79.37
GO-
SLAM[71]
Acc.[cm] ↓
4.60 3.31 3.97 3.05 2.74 4.61 4.32 3.91 3.81
Comp.[cm] ↓
5.56 3.48 6.90 3.31 3.46 5.16 5.40 5.01 4.79
Comp.Rat[%]↑73.35 82.86 74.23 82.56 86.19 75.76 72.63 76.61 78.00
HI-
SLAM[36]
Acc.[cm] ↓
3.21 3.74 3.16 3.87 2.60 4.62 4.25 3.53 3.62
Comp.[cm] ↓
3.25 3.08 4.09 5.29 8.83 4.42 4.06 3.72 4.59
Comp.Rat[%]↑86.99 87.19 80.82 72.55 72.44 80.90 81.04 82.88 80.60
3DGS-based
GLORIE
SLAM[59]
Acc.[cm] ↓
2.84 3.07 3.05 2.98 2.06 3.32 3.34 2.92 2.96
Comp.[cm] ↓
4.65 3.55 3.64 2.39 3.43 4.54 4.57 4.78 3.95
Comp.Rat[%]↑81.96 85.78 84.50 88.82 85.07 82.09 80.41 81.04 83.72
Splat-
SLAM[33]
Acc.[cm] ↓
1.99 1.91 2.06 3.96 2.03 3.45 2.15 1.89 2.43
Comp.[cm] ↓
3.78 3.38 3.34 2.75 3.33 4.36 3.96 4.25 3.64
Comp.Rat[%]↑85.47 86.88 86.12 87.32 85.17 81.37 82.25 82.95 84.69
Ours
Acc.[cm] ↓
1.35 1.40 1.87 1.40 1.18 1.94 1.70 1.70 1.57
Comp.[cm] ↓
3.33 3.27 3.66 2.07 3.23 4.29 3.84 4.26 3.49
Comp.Rat[%]↑87.45 85.91 86.13 89.41 85.63 81.73 82.52 83.23 85.25
In Table IV, we evaluate our geometry reconstruction results
against recent NeRF-based and 3DGS-based methods on the
Replica dataset, demonstrating superior performance in both
accuracy and completeness metrics. As illustrated in Fig. 6, our
method produces smoother reconstructions while preserving
fine geometric details compared to GLORIE-SLAM [59] and
HI-SLAM [36]. This is particularly evident in complex scene
elements such as chair legs and shelf-mounted vases, where
our results more closely match the ground truth. Qualitative
comparisons on the ScanNet dataset (Fig. 8) further highlight
our advantages over Splat-SLAM [33], showing more accu-
rate geometry without floating artifacts and achieving better
completeness. Additional qualitative results on the ScanNet++
dataset (Fig. 10) demonstrate our system’s capability to fully
reconstruct challenging scenes, including low-texture surfaces

<!-- page 10 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
10
Ours
Splat-
SLAM
Scene 0000
Scene 0106
Scene 0207
Fig. 8: Qualitative comparison on geometry and appearance reconstruction on ScanNet dataset.
like floors and walls. Notably, our reconstructions even capture
glass windows that are missing in the laser scanner ground
truth.
F. Appearance Reconstruction Quality
TABLE V: Rendering quality evaluations on Replica dataset
for RGB and RGB-D methods.
Method
Metric
ro-0
ro-1
ro-2
of-0
of-1
of-2
of-3
of-4
Avg.
RGB-D input
Point-
SLAM[68]
PSNR ↑
32.40 34.08 35.50 38.26 39.16 33.99 33.48 33.49 35.17
SSIM ↑
0.97
0.98
0.98
0.98
0.99
0.96
0.96
0.98
0.98
LPIPS ↓
0.11
0.12
0.11
0.10
0.12
0.16
0.13
0.14
0.12
Splat
TAM[31]
PSNR ↑
32.86 33.89 35.25 38.26 39.17 31.97 29.70 31.81 34.11
SSIM ↑
0.98
0.97
0.98
0.98
0.98
0.97
0.95
0.95
0.97
LPIPS ↓
0.07
0.10
0.08
0.09
0.09
0.10
0.12
0.15
0.10
Mono
GS[32]
PSNR ↑
34.83 36.43 37.49 39.95 42.09 36.24 36.70 36.07 37.50
SSIM ↑
0.95
0.96
0.97
0.97
0.98
0.96
0.96
0.96
0.96
LPIPS ↓
0.07
0.08
0.08
0.07
0.06
0.08
0.07
0.10
0.07
RGB input
GLORIE
-SLAM[59]
PSNR ↑
28.49 30.09 29.98 35.88 37.15 28.45 28.54 29.73 31.04
SSIM ↑
0.96
0.97
0.96
0.98
0.99
0.97
0.97
0.97
0.97
LPIPS ↓
0.13
0.13
0.14
0.09
0.08
0.15
0.11
0.15
0.12
Splat-
SLAM[33]
PSNR ↑
32.25 34.31 35.95 40.81 40.64 35.19 35.03 37.40 36.45
SSIM ↑
0.91
0.93
0.95
0.98
0.97
0.96
0.95
0.98
0.95
LPIPS ↓
0.10
0.09
0.06
0.05
0.05
0.07
0.06
0.04
0.06
Ours
PSNR ↑
35.48 36.93 38.53 42.28 43.16 37.31 36.99 38.95 38.71
SSIM ↑
0.96
0.97
0.97
0.98
0.98
0.97
0.97
0.97
0.97
LPIPS ↓
0.04
0.04
0.03
0.02
0.03
0.04
0.04
0.04
0.03
Tables III,
V and
VI present our rendering quality
evaluation results. On the Replica dataset, our system sig-
nificantly outperforms competing methods, achieving supe-
rior PSNR and LPIPS metrics. For the ScanNet dataset, we
demonstrate better performance than RGB-D methods while
matching the strong baseline of Splat-SLAM [33]. As detailed
in our ablation study (Sec. IV-G), we could achieve even
higher rendering quality by relaxing geometric constraints by
normal loss, but this would compromise geometry accuracy.
Instead, our system balances the trade-off between geometry
and appearance quality. On the Waymo Open dataset, Fig-
ure 9 shows qualitative comparisons of our rendered RGB
images with those from other methods, and Table III presents
the corresponding quantitative results. Our system achieves
significantly higher rendering quality. This enhanced visual
fidelity can be attributed to two key factors: firstly our superior
TABLE VI: Rendering quality evaluations on ScanNet dataset
for rgb and rgbd methods.
Method
Metric
0000
0059
0106
0169
0181
0207
Avg.
RGB-D input
Point
-SLAM[68]
PSNR ↑
21.30 19.48 16.80 18.53 22.27 20.56 19.82
SSIM ↑
0.81
0.77
0.68
0.69
0.82
0.75
0.75
LPIPS ↓
0.48
0.50
0.54
0.54
0.47
0.54
0.51
Splat
TAM[31]
PSNR ↑
18.70 20.91 19.84 22.16 22.01 18.90 20.42
SSIM ↑
0.71
0.79
0.81
0.78
0.82
0.75
0.78
LPIPS ↓
0.48
0.32
0.32
0.34
0.42
0.41
0.38
Gaussian
-SLAM[76]
PSNR ↑
28.54 26.21 26.26 28.60 27.79 28.63 27.67
SSIM ↑
0.93
0.93
0.93
0.92
0.92
0.91
0.92
LPIPS ↓
0.27
0.21
0.22
0.23
0.28
0.29
0.25
RGB input
GLORIE
-SLAM[59]
PSNR ↑
23.42 20.66 20.41 25.23 21.28 23.68 22.45
SSIM ↑
0.87
0.83
0.84
0.91
0.76
0.85
0.84
LPIPS ↓
0.26
0.31
0.31
0.21
0.44
0.29
0.30
Splat-
SLAM[33]
PSNR ↑
28.68 27.69 27.70 31.14 31.15 30.49 29.48
SSIM ↑
0.83
0.87
0.86
0.87
0.84
0.84
0.85
LPIPS ↓
0.19
0.15
0.18
0.15
0.23
0.19
0.18
Ours
PSNR ↑
28.62 27.22 28.13 31.28 30.37 30.03 29.27
SSIM ↑
0.85
0.87
0.90
0.90
0.90
0.86
0.88
LPIPS ↓
0.28
0.23
0.21
0.18
0.25
0.30
0.24
tracking accuracy; and secondly the effective integration of
depth and normal supervision in our mapping pipeline.
G. Ablation Study
TABLE VII: Depth accuracy of our final rendered depth
compared to the prior depth aligned using different strategies,
as well as BA depth with and without JDSA assistance,
evaluated on the Replica dataset.
Abs Diff
[m] ↓
Abs Rel
[%] ↓
Sq Rel
[%] ↓
RMSE
[m] ↓
δ <1.05
[%] ↑
δ <1.25
[%] ↑
Depth Type
Prior(one scale)
0.147
6.70
4.62
0.18
66.69
94.85
Prior(scale grid)
0.074
3.41
0.52
0.10
77.45
99.66
BA estimate
0.059
2.86
0.37
0.09
83.52
99.74
BA with JDSA
0.046
1.99
0.51
0.11
91.84
99.32
Ours Rendered
0.015
0.67
0.10
0.04
98.65
99.81
Monocular Prior Integration We first evaluate the depth
estimation accuracy improved by our different modules on
the Replica dataset, as depth accuracy is crucial for scene
reconstruction quality. Table VII quantifies the effectiveness of
different approaches. Comparing Prior (one scale) with Prior

<!-- page 11 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
11
GT
Ours
OpenGS-SLAM
13476
106762
100613
MonoGS
Fig. 9: Rendering quality comparison on the Waymo Open
Dataset in unbounded outdoor scenes. Our method captures
finer details of the driving environment, while other methods
produce noticeably blurrier results.
TABLE VIII: Ablation study on the choice of depth prior
models, evaluating rendering quality, reconstruction accuracy,
and inference time per frame on the Replica dataset.
PSNR
[dB] ↑
Acc
[cm] ↓
Comp
[cm] ↓
Comp Rat
[%] ↑
Runtime
[ms] ↓
Ours + Metric3D
38.59
1.60
3.58
85.11
61
Ours + DA V2
38.68
1.57
3.48
85.31
32
Ours + ZoeDepth
38.60
1.64
3.60
85.00
52
Ours + Omnidata
38.71
1.57
3.49
85.25
6
(scale grid) demonstrates that our grid-based scale alignment
significantly outperforms the single-scale alignment from HI-
SLAM [36], better addressing inherent scale distortions in
monocular depth priors. Further analysis compares two depth
estimation approaches: BA estimate (using BA alone) and BA
with JDSA (using interleaved BA and JDSA optimization).
The results confirm that incorporating JDSA with alternat-
ing optimization outperforms BA alone, validating our depth
prior integration strategy. The final rendered depth from our
Gaussian map achieves the highest accuracy, validating the
effectiveness of our complete pipeline.
We further investigate the incorporation of alternative depth
and normal prior predictors. For depth priors, including Met-
ric3D [77], ZoeDepth [78], and Depth Anything (DA) V2 [79],
we present results in Table VIII. Notably, ZoeDepth and DA
V2 were not trained on any Replica images, yet still achieve
comparable performance, indicating that our method does not
rely on indirect biases from the training data. For normal
priors, we evaluate EESNU [25] and DSINE [80], as shown
in Table IX. The results demonstrate that our method remains
compatible with these alternatives. Nevertheless, our chosen
prior, OmniData, offers a more efficient trade-off between
performance and computational cost. This choice also aligns
TABLE IX: Ablation study on the choice of normal prior
models.
PSNR
[dB] ↑
Acc
[cm] ↓
Comp
[cm] ↓
Comp Rat
[%] ↑
Runtime
[ms] ↓
Ours + EESNU
38.60
1.70
3.64
84.61
5
Ours + DSINE
38.52
1.62
3.55
85.02
37
Ours + OmniData
38.71
1.57
3.49
85.25
6
TABLE X: Ablation study on the progressive improvement
in trajectory accuracy on the Replica dataset, averaged over
8 sequences. From left to right, each stage refines the pose
estimation based on the previous stage.
Online
Tracking
Online
PGBA
Offline
Full BA
Joint Pose Map
Refinement
ATE [cm] ↓
0.42
0.33
0.32
0.26
with the baseline methods, which likewise employ OmniData
as the geometry prior.
Trajectory Accuracy Table X demonstrates the progressive
improvement in pose estimation accuracy on the Replica
dataset through our system pipeline. Starting with initial
estimates from the online tracking module, accuracy is first
enhanced through online PGBA based loop closing. A sub-
sequent full BA further refines these results, with the final
joint pose and 3DGS map refinement achieving the highest
trajectory accuracy. This systematic improvement across stages
validates the effectiveness of our hierarchical optimization
approach.
Component Analysis To evaluate key design components,
we conduct ablation studies by removing individual com-
ponents. Table XI confirms each module’s contribution to
system performance. The grid-based scale alignment proves
crucial, as its removal significantly degrades reconstruction
accuracy. Similarly, the unbiased depth rendering enhances
both rendering quality and geometric accuracy, with qualitative
comparison shown in Fig. 11. While the normal loss slightly
affects appearance metrics, it substantially improves geometry
quality. The final joint pose and map refinement further
enhances accuracy through improved global consistency.
H. Performance Analysis
Our system achieves real-time performance with the online
tracking, loop closing and mapping operating at 22 frames per
second (FPS) on the Replica dataset, and takes only 12 seconds
for offline refinement. On the ScanNet dataset, despite more
rapid camera movements, it maintains robust performance
online above 10 FPS, with offline refinement taking a few
minutes due to a higher number of optimization iterations.
Fig. 12 illustrates the map size evolution, GPU memory
consumption, and processing speed for sequence Scene0000.
While the Gaussian count initially grows during new area
exploration, our efficient map pruning strategy stabilizes the
map size by eliminating redundant Gaussians in revisited
regions. The memory analysis reveals that both allocated and
reserved GPU memory usage remain stable throughout the
sequence, with periodic fluctuations corresponding to pruning
cycles. Reserved memory gradually increases as additional

<!-- page 12 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
12
(a) f34d532901
(b) 39f36da05b
(c) 8b5caf3398
(d) b20a261fdf
GT
Ours
Fig. 10: Reconstructed meshes of four selected sequences on ScanNet++ Dataset.
TABLE XI: Ablation study on the impact of the different
proposed modules on the reconstruction performance on the
Replica dataset, averaged over 8 sequences.
PSNR
[dB] ↑
SSIM
↑
LPIPS
↓
Acc.
[cm] ↓
Comp.
[cm] ↓
Comp. Rat
[%] ↑
w/o grid-based scale align
37.18
0.97
0.05
1.68
3.58
84.04
w/o unbiased depth render 38.23
0.97
0.04
1.73
3.92
81.16
w/o Lnormal
39.09
0.98
0.03
2.46
4.09
82.40
w/o joint pose map refine
37.25
0.96
0.04
1.61
3.55
84.29
Ours
38.71
0.97
0.03
1.57
3.49
85.25
Based on center-depth
Based on intersection-depth
Fig. 11: Reconstruction quality comparison on the Room0
scene of the Replica dataset, using rendered depth based on
the depth at the Gaussian center versus our approach, which
uses the depth at the ray-Gaussian intersection.
buffers are allocated to store keyframe states. The processing
frame rate decreases during rapid motion due to more frequent
keyframe insertion, and increases during stable motion, re-
maining consistent across different phases. This demonstrates
our system’s scalability and efficiency in managing large-
scale environments while maintaining predictable resource
consumption.
I. Evaluation on Self-Collected Robot Data
We evaluate our system on a self-collected dataset captured
by our robotic platform. Figure 13 shows the platform and
the results in a large factory hall. The robot is equipped
with stereo cameras. However, only the monocular input from
Fig. 12: Evolution of map size, GPU memory usage, and
system speed over frame index for the Scene0000 sequence
from ScanNet, evaluated on an RTX 4090 GPU. The number
of Gaussians grows as new areas are explored and stabilizes
with the pruning strategy. System speed denotes the processing
frame rate for online tracking, loop closing, and mapping.
Allocated and reserved GPU memory usage are both reported
to provide resource analysis.
the left camera is used for this experiment. The recorded
sequence has intotal 4073 frames with a duration of 6 minutes
and 52 seconds. We compare our rendering results with the
baseline method DROID-SLAM + 3DGS, where the estimated
camera poses and point cloud from DROID-SLAM serve as
input and used to initialize the Gaussians. For our system,
it completes online stage including tracking, loop closuing,
and mapping in 5 minutes and 20 seconds, followed by 3
minutes and 32 seconds for the offline refinement stage. In
contrast, the baseline requires 4 minutes and 24 seconds for
DROID-SLAM and an additional 15 minutes and 2 seconds
for 3DGS mapping. Overall, our method takes only about
half the runtime of the baseline while achieving comparable
rendering quality and significantly better geometry, without
the floaters observed in the baseline results. Furthermore, we
evaluate the trajectory accuracy using the centimeter-accurate
photogrammetric reference described in [81], as shown in

<!-- page 13 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
13
Fig. 13: Evaluation on self-collected data with the robot navigating through a large factory hall. Our robot (a) is equipped
with stereo cameras. Our system operates solely on the monocular input from the left camera and reconstructs the scene along
with the estimated camera trajectory (b). The rendered color and depth maps (c) achieve comparable visual quality while
exhibiting significantly better geometry and free of floaters. In contrast, the baseline DROID-SLAM + 3DGS (d) suffers from
geometric artifacts and floating points. We refer readers to our supplementary video, which showcases novel view renderings
on out-of-sequence viewpoints.
MASt3R-SLAM 
lost track
X [m]
Y [m]
ATE
Fig. 14: Qualitative comparison of the estimated robot trajec-
tory in the factory hall.
Fig. 14. While MASt3R-SLAM [75] struggles to track camera
poses due to poor generalizability to the new environment, our
system maintains stable tracking and successfully detects loop
closures with the highest accuracy.
J. Extention to Semantic Reconstruction
(a)
(b)
Fig. 15: Semantic reconstruction results: (a) outdoor scene
from 100613 of Waymo Open dataset, (b) indoor scene from
Room1 of the Replica dataset.
As an extension, we demonstrate the capability of our
system for semantic scene reconstruction by incorporating
2D semantic information into the 3DGS representation. Each
Gaussian primitive is augmented with semantic color channels
in addition to its existing geometric and appearance attributes.
These semantic channels can be efficiently rasterized to the
image plane alongside color and depth, enabling simultaneous
semantic colorization of the reconstructed scene. For semantic
optimization, we maintain the same pipeline structure as our
depth and pose optimization framework, with an additional L1
semantic RGB loss term that measures the absolute difference
between rendered and ground truth semantic color maps.
Following the evaluation protocol of [82], we assess our
semantic reconstruction performance on the Replica dataset
using mean Intersection over Union (mIoU) as the primary
metric. We evaluate on four standard sequences to enable
direct comparison with existing baseline methods.
TABLE XII: Semantic reconstruction results evaluated by
mIoU metric on 4 sequences of the Replica dataset.
Method
ro-0
ro-1
ro-2
of-0
Avg.mIoU[%]↑
RGB-D
NIDS-SLAM[82]
82.45
84.08
76.99
85.94
82.37
DNS-SLAM[83]
88.32
84.90
81.20
84.66
84.77
SNI-SLAM[84]
88.42
87.43
86.16
87.63
87.41
SGS-SLAM[85]
92.95
92.91
92.10
92.90
92.72
Hier-SLAM[86]
95.25
95.81
95.73
95.52
95.58
RGB
HI-SLAM[36]
74.93
79.55
80.90
71.53
76.72
Ours
90.27
92.80
91.11
92.45
91.65
Tab. XII presents quantitative results comparing our ap-
proach against recent RGB-D and RGB-only semantic SLAM
methods, including concurrent work Hier-SLAM [86]. Our
system achieves competitive performance compared to state-
of-the-art RGB-D methods, while significantly outperforming
the RGB-only baseline HI-SLAM [36] by a margin of 14.93%.
This substantial improvement can be attributed to two key fac-
tors: (1) our more accurate geometry reconstruction provides
better surface boundaries for semantic label assignment, and
(2) the explicit 3DGS representation allows for sharper seman-
tic boundaries compared to implicit NeRF-based approaches
that often struggle with object delineation in complex scenes
or regions with small objects. Fig. 15 demonstrates qualitative
results: indoor scene from the Replica dataset with ground
truth semantic labels, and outdoor driving scene from the

<!-- page 14 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
14
Waymo Open dataset using Mask2Former [87] predictions as
semantic inputs.
V. CONCLUSION
This article presents HI-SLAM2, a novel monocular SLAM
system that achieves fast and accurate dense 3D scene re-
construction through four complementary modules. The online
tracking module enhances depth and pose estimation by inte-
grating depth priors with grid-based scale alignment, while
parallel PGBA in the online loop closing module corrects
pose and scale drift. Our mapping approach leverages 3D
Gaussian splatting for compact scene representation, contin-
uously refined during SLAM tracking. We enhance geometric
consistency through monocular normal priors and unbiased
ray-Gaussian intersection depth for splat-based rasterization.
During offline refinement, we achieve high-fidelity recon-
struction by incorporating exposure compensation and per-
forming joint optimization of camera poses, 3DGS map, and
exposure parameters. Extensive evaluations on challenging
datasets demonstrate that HI-SLAM2 outperforms state-of-the-
art methods in accuracy and completeness while maintaining
superior runtime performance. Our system achieves high-
quality geometry and appearance reconstruction without the
typical trade-offs observed in other methods.
Limitations: Our system has three main limitations: First, the
current proximity-based loop closure detection shows limited
robustness in the ETH3D dataset when encountering view
occlusions and textureless regions, suggesting the need for
learned feature-based place recognition. Second, in city-scale
scenes, mapping quality can degrade due to limited opti-
mization budget, indicating the need for submap optimization
strategies. Third, the system assumes static environments.
Incorporating dynamic object detection and tracking, along
with motion segmentation, would enable robust operation in
dynamic environments.
ACKNOWLEDGMENT
This
work
is
partially
supported
by
the
Deutsche
Forschungsgemeinschaft (DFG, German Research Foundation)
under Germany’s Excellence Strategy - EXC 2120/1 - project
number 390831618. Qing Cheng is supported by the DAAD
program Konrad Zuse Schools of Excellence in Artificial
Intelligence, and the Federal Ministry of Research, Technology
and Space.
REFERENCES
[1] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim,
A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon,
“Kinectfusion: Real-time dense surface mapping and tracking,” in 2011
10th IEEE international symposium on mixed and augmented reality.
Ieee, 2011, pp. 127–136.
[2] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J.
Davison, “Elasticfusion: Dense slam without a pose graph.” in Robotics:
science and systems, vol. 11.
Rome, Italy, 2015, p. 3.
[3] A. Dai, M. Nießner, M. Zollh¨ofer, S. Izadi, and C. Theobalt, “Bundle-
fusion: Real-time globally consistent 3d reconstruction using on-the-fly
surface reintegration,” ACM Transactions on Graphics (ToG), vol. 36,
no. 4, pp. 1–18, 2017.
[4] D. Azinovi´c, R. Martin-Brualla, D. B. Goldman, M. Nießner, and
J. Thies, “Neural rgb-d surface reconstruction,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022, pp. 6290–6301.
[5] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “imap: Implicit mapping and
positioning in real-time,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2021, pp. 6229–6238.
[6] F. Lu and E. Milios, “Globally consistent range scan alignment for
environment mapping,” Autonomous robots, vol. 4, pp. 333–349, 1997.
[7] J. Zhang, S. Singh et al., “Loam: Lidar odometry and mapping in real-
time.” in Robotics: Science and systems, vol. 2, no. 9.
Berkeley, CA,
2014, pp. 1–9.
[8] W. Hess, D. Kohler, H. Rapp, and D. Andor, “Real-time loop closure in
2d lidar slam,” in 2016 IEEE international conference on robotics and
automation (ICRA).
IEEE, 2016, pp. 1271–1278.
[9] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, “Fast-lio2: Fast direct lidar-
inertial odometry,” IEEE Transactions on Robotics, vol. 38, no. 4, pp.
2053–2073, 2022.
[10] Y. Pan, X. Zhong, L. Wiesmann, T. Posewsky, J. Behley, and C. Stach-
niss, “Pin-slam: Lidar slam using a point-based implicit neural repre-
sentation for achieving global map consistency,” IEEE Transactions on
Robotics, vol. 40, pp. 4045–4064, 2024.
[11] C. Cadena, L. Carlone, H. Carrillo, Y. Latif, D. Scaramuzza, J. Neira,
I. Reid, and J. J. Leonard, “Past, present, and future of simultaneous
localization and mapping: Toward the robust-perception age,” IEEE
Transactions on robotics, vol. 32, no. 6, pp. 1309–1332, 2016.
[12] A. J. Davison, I. D. Reid, N. D. Molton, and O. Stasse, “Monoslam:
Real-time single camera slam,” IEEE transactions on pattern analysis
and machine intelligence, vol. 29, no. 6, pp. 1052–1067, 2007.
[13] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “Orb-slam: a versatile
and accurate monocular slam system,” IEEE transactions on robotics,
vol. 31, no. 5, pp. 1147–1163, 2015.
[14] J. Engel, T. Sch¨ops, and D. Cremers, “LSD-SLAM: Large-scale di-
rect monocular SLAM,” in European conference on computer vision.
Springer, 2014, pp. 834–849.
[15] C. Forster, Z. Zhang, M. Gassner, M. Werlberger, and D. Scaramuzza,
“Svo: Semidirect visual odometry for monocular and multicamera sys-
tems,” IEEE Transactions on Robotics, vol. 33, no. 2, pp. 249–265,
2016.
[16] T. Qin, P. Li, and S. Shen, “Vins-mono: A robust and versatile monocular
visual-inertial state estimator,” IEEE Transactions on Robotics, vol. 34,
no. 4, pp. 1004–1020, 2018.
[17] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and J. D.
Tard´os, “ORB-SLAM3: An accurate open-source library for visual,
visual-inertial, and multimap slam,” IEEE Transactions on Robotics,
vol. 37, no. 6, pp. 1874–1890, 2021.
[18] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “Dtam: Dense
tracking and mapping in real-time,” in 2011 international conference on
computer vision.
IEEE, 2011, pp. 2320–2327.
[19] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE
transactions on pattern analysis and machine intelligence, vol. 40, no. 3,
pp. 611–625, 2017.
[20] Z. Teed and J. Deng, “DROID-SLAM: Deep visual SLAM for monoc-
ular, stereo, and RGB-D cameras,” Advances in Neural Information
Processing Systems, vol. 34, pp. 16 558–16 569, 2021.
[21] ——, “RAFT: Recurrent all-pairs field transforms for optical flow,” in
European conference on computer vision. Springer, 2020, pp. 402–419.
[22] H. Xu, J. Zhang, J. Cai, H. Rezatofighi, and D. Tao, “Gmflow: Learning
optical flow via global matching,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2022, pp. 8121–
8130.
[23] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “Mvsnet: Depth inference
for unstructured multi-view stereo,” in Proceedings of the European
conference on computer vision (ECCV), 2018, pp. 767–783.
[24] A. Eftekhar, A. Sax, J. Malik, and A. Zamir, “Omnidata: A scalable
pipeline for making multi-task mid-level vision datasets from 3d scans,”
in Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 10 786–10 796.
[25] G. Bae, I. Budvytis, and R. Cipolla, “Estimating and exploiting the
aleatoric uncertainty in surface normal estimation,” in Proceedings of
the IEEE/CVF International Conference on Computer Vision, 2021, pp.
13 137–13 146.
[26] K. Tateno, F. Tombari, I. Laina, and N. Navab, “Cnn-slam: Real-time
dense monocular slam with learned depth prediction,” in Proceedings of
the IEEE conference on computer vision and pattern recognition, 2017,
pp. 6243–6252.

<!-- page 15 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
15
[27] Z. Teed and J. Deng, “Deepv2d: Video to depth with differentiable
structure from motion,” in International Conference on Learning Rep-
resentations, 2020, pp. 1–13.
[28] C. Godard, O. Mac Aodha, M. Firman, and G. J. Brostow, “Digging
into self-supervised monocular depth estimation,” in Proceedings of the
IEEE/CVF international conference on computer vision, 2019, pp. 3828–
3838.
[29] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and
M. Pollefeys, “Nice-slam: Neural implicit scalable encoding for slam,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 12 786–12 796.
[30] A. Rosinol, J. J. Leonard, and L. Carlone, “Nerf-slam: Real-time
dense monocular slam with neural radiance fields,” in 2023 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS).
IEEE, 2023, pp. 3437–3444.
[31] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer,
D. Ramanan, and J. Luiten, “Splatam: Splat track & map 3d gaussians
for dense rgb-d slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 21 357–21 366.
[32] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian splatting
slam,” in Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2024, pp. 18 039–18 048.
[33] E. Sandstr¨om, G. Zhang, K. Tateno, M. Oechsle, M. Niemeyer, Y. Zhang,
M. Patel, L. Van Gool, M. Oswald, and F. Tombari, “Splat-slam:
Globally optimized rgb-only slam with 3d gaussians,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) Workshops, June 2025, pp. 1680–1691.
[34] B. Zhang, C. Fang, R. Shrestha, Y. Liang, X. Long, and P. Tan,
“Rade-gs: Rasterizing depth in gaussian splatting,” arXiv preprint
arXiv:2406.01467, 2024.
[35] H. Wang, J. Wang, and L. Agapito, “Co-slam: Joint coordinate and
sparse parametric encodings for neural real-time slam,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 13 293–13 302.
[36] W. Zhang, T. Sun, S. Wang, Q. Cheng, and N. Haala, “HI-SLAM:
Monocular real-time dense mapping with hybrid implicit fields,” IEEE
Robotics and Automation Letters, 2024.
[37] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel,
R. Mur-Artal, C. Ren, S. Verma et al., “The replica dataset: A digital
replica of indoor spaces,” arXiv preprint arXiv:1906.05797, 2019.
[38] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and
M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor
scenes,” in Proceedings of the IEEE conference on computer vision and
pattern recognition, 2017, pp. 5828–5839.
[39] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui,
J. Guo, Y. Zhou, Y. Chai, B. Caine et al., “Scalability in perception
for autonomous driving: Waymo open dataset,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2020,
pp. 2446–2454.
[40] T. Sch¨ops, T. Sattler, and M. Pollefeys, “BAD SLAM: Bundle adjusted
direct RGB-D SLAM,” in Conference on Computer Vision and Pattern
Recognition (CVPR), 2019.
[41] C. Yeshwanth, Y.-C. Liu, M. Nießner, and A. Dai, “Scannet++: A high-
fidelity dataset of 3d indoor scenes,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 12–22.
[42] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Proceedings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 4104–4113.
[43] H. Hirschmuller, “Stereo processing by semiglobal matching and mu-
tual information,” IEEE Transactions on pattern analysis and machine
intelligence, vol. 30, no. 2, pp. 328–341, 2007.
[44] P.-H. Huang, K. Matzen, J. Kopf, N. Ahuja, and J.-B. Huang, “Deepmvs:
Learning multi-view stereopsis,” in Proceedings of the IEEE conference
on computer vision and pattern recognition, 2018, pp. 2821–2830.
[45] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards
robust monocular depth estimation: Mixing datasets for zero-shot cross-
dataset transfer,” IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 44, no. 3, 2022.
[46] C. Wu, “Visualsfm: A visual structure from motion system,” http://www.
cs. washington. edu/homes/ccwu/vsfm, 2011.
[47] H. Hirschmuller, “Accurate and efficient stereo processing by semi-
global matching and mutual information,” in 2005 IEEE computer soci-
ety conference on computer vision and pattern recognition (CVPR’05),
vol. 2.
IEEE, 2005, pp. 807–814.
[48] C. Barnes, E. Shechtman, A. Finkelstein, and D. B. Goldman, “Patch-
match: A randomized correspondence algorithm for structural image
editing,” ACM Trans. Graph., vol. 28, no. 3, p. 24, 2009.
[49] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[50] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Transactions on
Graphics (ToG), vol. 41, no. 4, pp. 1–15, 2022.
[51] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance
fields,” in European conference on computer vision.
Springer, 2022,
pp. 333–350.
[52] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and
A. Kanazawa, “Plenoxels: Radiance fields without neural networks,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022, pp. 5501–5510.
[53] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[54] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting
for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024
Conference Papers, 2024, pp. 1–11.
[55] B. Curless and M. Levoy, “A volumetric method for building complex
models from range images,” in Proceedings of the 23rd annual confer-
ence on Computer graphics and interactive techniques, 1996, pp. 303–
312.
[56] L. Koestler, N. Yang, N. Zeller, and D. Cremers, “Tandem: Tracking and
dense mapping in real-time using deep multi-view stereo,” in Conference
on Robot Learning.
PMLR, 2022, pp. 34–45.
[57] X. Zuo, N. Yang, N. Merrill, B. Xu, and S. Leutenegger, “Incremental
dense reconstruction from monocular video with guided sparse feature
volume fusion,” IEEE Robotics and Automation Letters, vol. 8, no. 6,
pp. 3876–3883, 2023.
[58] Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and
M. Pollefeys, “Nicer-slam: Neural implicit scene encoding for rgb slam,”
in 2024 International Conference on 3D Vision (3DV).
IEEE, 2024.
[59] G. Zhang, E. Sandstr¨om, Y. Zhang, M. Patel, L. Van Gool, and M. R.
Oswald, “Glorie-slam: Globally optimized rgb-only implicit encoding
point cloud slam,” arXiv preprint arXiv:2403.19549, 2024.
[60] W. Zhang, S. Wang, X. Dong, R. Guo, and N. Haala, “Bamf-slam:
Bundle adjusted multi-fisheye visual-inertial slam using recurrent field
transforms,” in 2023 IEEE International Conference on Robotics and
Automation (ICRA), 2023, pp. 6232–6238.
[61] H. Strasdat, J. Montiel, and A. J. Davison, “Scale drift-aware large scale
monocular slam.” in Robotics: Science and Systems, vol. 2, no. 3, 2010,
p. 5.
[62] W. Niemeier, Ausgleichungsrechnung: statistische auswertemethoden.
de Gruyter, 2008.
[63] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and
G. Drettakis, “A hierarchical 3d gaussian representation for real-time
rendering of very large datasets,” ACM Transactions on Graphics (TOG),
vol. 43, no. 4, pp. 1–15, 2024.
[64] D.
P.
Kingma
and
J.
Ba,
“Adam:
A
method
for
stochastic
optimization,” CoRR, vol. abs/1412.6980, 2014. [Online]. Available:
https://api.semanticscholar.org/CorpusID:6628106
[65] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,
T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An
imperative style, high-performance deep learning library,” Advances in
neural information processing systems, vol. 32, pp. 8024–8035, 2019.
[66] S. Yu, C. Cheng, Y. Zhou, X. Yang, and H. Wang, “Rgb-only gaussian
splatting slam for unbounded outdoor scenes,” in 2025 IEEE Inter-
national Conference on Robotics and Automation (ICRA), 2025, pp.
11 068–11 074.
[67] M. M. Johari, C. Carta, and F. Fleuret, “Eslam: Efficient dense slam
system based on hybrid representation of signed distance fields,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 17 408–17 419.
[68] E. Sandstr¨om, Y. Li, L. Van Gool, and M. R. Oswald, “Point-slam:
Dense neural point cloud-based slam,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 18 433–18 444.
[69] P. Zhu, Y. Zhuang, B. Chen, L. Li, C. Wu, and Z. Liu, “Mgs-slam:
Monocular sparse tracking and gaussian mapping with depth smooth
regularization,” IEEE Robotics and Automation Letters, vol. 9, no. 11,
pp. 9486–9493, 2024.
[70] L. Zhu, Y. Li, E. Sandstr¨om, S. Huang, K. Schindler, and I. Armeni,
“Loopsplat: Loop closure by registering 3d gaussian splats,” in 2025
International Conference on 3D Vision (3DV).
IEEE, 2025, pp. 156–
167.

<!-- page 16 -->
PUBLISHED IN IEEE TRANSACTIONS ON ROBOTICS. DOI: 10.1109/TRO.2025.3626627
16
[71] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, “Go-slam: Global
optimization for consistent 3d instant reconstruction,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision (ICCV),
October 2023.
[72] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-slam: Real-
time simultaneous localization and photorealistic mapping for monocular
stereo and rgb-d cameras,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 21 584–21 593.
[73] Z. Teed, L. Lipson, and J. Deng, “Deep patch visual odometry,” Ad-
vances in Neural Information Processing Systems, vol. 36, pp. 39 033–
39 051, 2023.
[74] L. Lipson, Z. Teed, and J. Deng, “Deep patch visual slam,” in European
Conference on Computer Vision.
Springer, 2024, pp. 424–440.
[75] R. Murai, E. Dexheimer, and A. J. Davison, “Mast3r-slam: Real-
time dense slam with 3d reconstruction priors,” in Proceedings of the
Computer Vision and Pattern Recognition Conference, 2025, pp. 16 695–
16 705.
[76] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-slam:
Photo-realistic dense slam with gaussian splatting,” arXiv preprint
arXiv:2312.10070, 2023.
[77] W. Yin, C. Zhang, H. Chen, Z. Cai, G. Yu, K. Wang, X. Chen, and
C. Shen, “Metric3d: Towards zero-shot metric 3d prediction from a sin-
gle image,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision, 2023, pp. 9043–9053.
[78] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. M¨uller, “Zoedepth:
Zero-shot transfer by combining relative and metric depth,” arXiv
preprint arXiv:2302.12288, 2023.
[79] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao,
“Depth anything v2,” Advances in Neural Information Processing Sys-
tems, vol. 37, pp. 21 875–21 911, 2024.
[80] G. Bae and A. J. Davison, “Rethinking inductive biases for surface
normal estimation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 9535–9545.
[81] V. Ress, J. Meyer, W. Zhang, D. Skuddis, U. Soergel, and N. Haala,
“3d gaussian splatting aided localization for large and complex indoor-
environments,” The International Archives of the Photogrammetry, Re-
mote Sensing and Spatial Information Sciences, vol. XLVIII-G-2025,
pp. 1283–1290, 2025.
[82] Y. Haghighi, S. Kumar, J.-P. Thiran, and L. Van Gool, “Neural implicit
dense semantic slam,” arXiv preprint arXiv:2304.14560, 2023.
[83] K. Li, M. Niemeyer, N. Navab, and F. Tombari, “Dns-slam: Dense neural
semantic-informed slam,” in 2024 IEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 7839–7846.
[84] S. Zhu, G. Wang, H. Blum, J. Liu, L. Song, M. Pollefeys, and
H. Wang, “Sni-slam: Semantic neural implicit slam,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 21 167–21 177.
[85] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, “Sgs-
slam: Semantic gaussian splatting for neural dense slam,” in European
Conference on Computer Vision.
Springer, 2024, pp. 163–179.
[86] B. Li, Z. Cai, Y.-F. Li, I. Reid, and H. Rezatofighi, “Hier-slam: Scaling-
up semantics in slam with a hierarchically categorical gaussian splat-
ting,” in IEEE International Conference on Robotics and Automation
(ICRA), 2025.
[87] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar,
“Masked-attention mask transformer for universal image segmentation,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022, pp. 1290–1299.
