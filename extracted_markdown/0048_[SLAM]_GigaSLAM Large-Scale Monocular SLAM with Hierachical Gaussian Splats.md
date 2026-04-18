<!-- page 1 -->
arXiv:2503.08071v2  [cs.RO]  10 Jun 2025
GigaSLAM: Large-Scale Monocular SLAM with Hierarchical Gaussian Splats
Kai Deng1, Yigong Zhang1, Jian Yang1, 2, Jin Xie1, 2
1Nankai University, Tianjin, China
2Nanjing University Suzhou Campus, Suzhou, China
dengkai@mail.nankai.edu.cn, zyg025@nankai.edu.cn, csjyang@nankai.edu.cn, csjxie@nju.edu.cn
Loop Closure
 Render Image # 2474
PSNR: 23.91 db; SSIM: 0.9464; LPIPS: 0.3064 
Render Image # 8501
PSNR: 24.95 db; SSIM: 0.9638 ; LPIPS:0.2571
RTE: 0.12%
Render Image # 11638
PSNR: 24.69 db; SSIM: 0.9558 ; LPIPS:0.2839
Figure 1. GigaSLAM achieves robust pose estimation and mapping accuracy across unbounded, multi-kilometer-scale outdoor sequences
while preserving high-fidelity scene rendering quality, highlighting the effectiveness of our approach for long-range, real-world scenarios.
Abstract
Tracking and mapping in large-scale, unbounded out-
door environments using only monocular RGB input
presents substantial challenges for existing SLAM systems.
Traditional Neural Radiance Fields (NeRF) and 3D Gaus-
sian Splatting (3DGS) SLAM methods are typically limited
to small, bounded indoor settings. To overcome these chal-
lenges, we introduce GigaSLAM1, the first RGB NeRF /
3DGS-based SLAM framework for kilometer-scale outdoor
environments, as demonstrated on the KITTI, KITTI 360, 4
Seasons and A2D2 datasets. Our approach employs a hier-
archical sparse voxel map representation, where Gaussians
are decoded by neural networks at multiple levels of detail.
This design enables efficient, scalable mapping and high-
fidelity viewpoint rendering across expansive, unbounded
scenes. For front-end tracking, GigaSLAM utilizes a metric
depth model combined with epipolar geometry and PnP al-
gorithms to accurately estimate poses, while incorporating
a Bag-of-Words-based loop closure mechanism to maintain
robust alignment over long trajectories. Consequently, Gi-
gaSLAM delivers high-precision tracking and visually faith-
ful rendering on urban outdoor benchmarks, establishing
a robust SLAM solution for large-scale, long-term scenar-
ios, and significantly extending the applicability of Gaus-
1https://github.com/DengKaiCQ/GigaSLAM
sian Splatting SLAM systems to unbounded outdoor envi-
ronments.
1. Introduction
Simultaneous localization and mapping (SLAM) from a
single monocular video is a longstanding challenge in com-
puter vision [3]. Conventional SLAM approaches [4, 8, 26,
27] seek accurate camera pose tracking and high-quality
map geometry. Recent advances in neural radiance fields
(NeRF) [24] and 3D gaussian splatting (3DGS) [17] have
inspired the SLAM community [50] to enhance map ca-
pabilities—enabling rich appearance encoding and realis-
tic rendering from free viewpoints.
Novel SLAM tech-
niques using NeRFs and 3DGS have shown great promise
for online mapping and rendering [16, 23, 61, 63, 64].
These capabilities open new possibilities for applications
in AR/VR, autonomous driving, and drone navigation by
allowing users and agents to render from different perspec-
tives. Unlike traditional 3D GS or NeRF methods that rely
on a pre-reconstructed static scene from SfM[36], SLAM-
based approaches must dynamically build and update the
3D scene online. This online nature introduces unique chal-
lenges, especially when handling loop closures and global
scene corrections.
1

<!-- page 2 -->
Despite these advancements, one key challenge remains
for such systems: current NeRF and 3DGS-based SLAM
frameworks are often limited in scene scale, with most re-
lying on RGB-D depth priors for accurate mapping and
tracking in larger scenes [16, 23, 63]. This is due to two
main reasons: the limitations of scene representations and
challenges in global alignment. Implicit methods, such as
NeRFs [61, 63, 64], have limited representational capac-
ity and are often confined to bounded regions. Their re-
liance on manually pre-defined scene bounding boxes be-
comes impractical in expansive outdoor environments with
dynamic scales and undefined boundaries.
Furthermore,
the prevalent dense volumetric grid representations exhibit
cubic space complexity O(n3), which incurs prohibitive
memory and computational costs when scaling to outdoor
scenes spanning thousands of cubic meters. Explicit meth-
ods, like Gaussian splatting [16, 23], are not memory-
efficient, as their size scales with scene growth, impacting
computational and memory efficiency. Meanwhile, concur-
rent work like OpenGS-SLAM [58] integrates 3R modules
[53] with 3D GS, achieving capability on short hundred-
meter-scale outdoor sequences (Waymo[43]).
However,
its Transformer-based 3R design incurs O(n2) memory
costs due to self-attention, limiting scalability to kilometer-
scale trajectories. Current 3R-based forward 3D reconstruc-
tion frameworks remain limited in scalability, as no prior
method has demonstrated robust performance on sequences
exceeding kilometer-level spatial extents. The second chal-
lenge lies in global alignment. In large-scale scenes, loop
closure is crucial, as it effectively reduces global drift.
However, most NeRF and 3DGS SLAM methods rely on
incremental gradient-based registration, which is not well-
suited for global alignment under such scenarios.
As a
result, existing NeRF and 3DGS SLAM methods are re-
stricted to small-scale, bounded indoor environments or
short-term outdoor scenes and often depend on RGB-D data
for robust scene alignment.
To address these challenges, we present GigaSLAM, a
Gaussian Splatting-based SLAM framework designed to
scale to large, outdoor, long-term, unbounded environ-
ments. The core of GigaSLAM’s technical contribution is
a novel hierarchical sparse voxel representation designed
for large-scale rendering-capable SLAM, with each level
encoding Gaussians at varying levels of detail. This map
representation has two key advantages for scaling: 1) it is
boundless, dynamically expanding as the camera moves; 2)
it enables content-aware, efficient rendering at large scale
through Level-of-Details (LoD) representation, reducing
the need to load hundreds of millions of Gaussians for a
single frame. We further enhance map geometry and cam-
era pose tracking accuracy using a data-driven monocular
metric depth module [29]. Finally, we integrate Bag-of-
Words loop closure detection [9] and design a comprehen-
sive post-closure Splats map update to maintain accuracy in
rendering. GigaSLAM has been shown to scale up to tens
of kilometers of travel distance in urban driving scenarios
[11, 12, 18, 55]. It achieves robust and accurate tracking
with the ability to render from any viewpoint, all on a sin-
gle GPU. To our knowledge, no existing framework offers
this capability.
We validate our methods on large-scale outdoor scenes
from the KITTI, KITTI 360, 4 Seasons and A2D2 dataset
[11, 12, 18, 55].
Our results indicate that this approach
is significantly more accurate and robust, outperforming
other monocular SLAM methods in average tracking per-
formance on long-term outdoor datasets comparing to cur-
rent 3D GS SLAM methods [23, 35] which are tailored for
indoor scenes with monocular RGB images.
Therefore,
existing NeRF and 3D Gaussian Splatting-based SLAM
frameworks struggle to handle these large-scale, unbounded
scenarios effectively. Our contributions are as follows: 1)
GigaSLAM, a novel Gaussian Splats-based SLAM frame-
work for large-scale, unbounded environments; 2) a hierar-
chical map representation for dynamic growth and level-of-
detail rendering in large-scale SLAM; 3) an efficient loop
closure procedure applicable to Gaussian splats map repre-
sentations.
2. Related Work
Traditional SLAM
Modern SLAM systems are typically
formulated as joint optimization problems [3], where the
goal is to estimate a robot’s pose (position and orienta-
tion) from video input. A representative system is ORB-
SLAM [27], along with its extended versions ORB-SLAM2
[26] and ORB-SLAM3 [4], which use feature points and
keyframes for monocular SLAM. In more recent studies, re-
searchers have incorporated deep learning models to solve
specific sub-modules within SLAM, which are then inte-
grated into traditional optimization-based frameworks. For
example, the authors of ∇SLAM [15] propose using au-
tomatic differentiation to model SLAM as a differentiable
computation graph. Similarly, CodeSLAM [2] introduces
an autoencoder-based representation for dense monocular
SLAM by learning compact geometric descriptors from im-
ages.
Several works [19, 45] have also embedded Bun-
dle Adjustment into end-to-end differentiable networks.
DROID-SLAM [46] exemplifies this trend by integrating
Dense Bundle Adjustment directly into an optical flow esti-
mation pipeline.
NeRF-based SLAM
The introduction of Neural Radi-
ance Fields (NeRF) [24] has inspired a range of SLAM
systems that leverage neural rendering for mapping and lo-
calization. iMAP [42] is one of the first works to explore
NeRF-based scene reconstruction for SLAM. Building on
this, NICE-SLAM [63] adopts a grid-based hierarchical
2

<!-- page 3 -->
representation to improve efficiency in indoor mapping.
NICER-SLAM [64] further introduces geometric and op-
tical flow constraints, along with a warping loss, to improve
consistency while supporting monocular input. The authors
of Point-SLAM [34] employ point-based neural fields for
explicit scene representation, though their method requires
RGB-D input and does not integrate with traditional SLAM
architectures. GO-SLAM [61] extends NICE-SLAM by in-
corporating loop closure detection to enhance global recon-
struction consistency. However, since NeRF-based systems
generally require pre-defined scene boundaries, they strug-
gle to scale to unbounded outdoor environments, limiting
their applicability in such scenarios.
Gaussian Splatting-based SLAM
The emergence of 3D
Gaussian Splatting (3DGS) [17] has motivated researchers
to explore its potential for SLAM. The earliest SLAM adap-
tations of 3DGS include SplaTAM [16] and MonoGS [23].
In SplaTAM, the authors propose an online SLAM frame-
work that employs Gaussian primitives and differentiable
contour-guided optimization for both tracking and map-
ping. MonoGS takes advantage of the explicit and com-
pact nature of Gaussians, introducing geometric validation
and regularization to address ambiguities in dense recon-
struction; the method supports both RGB and RGB-D in-
puts, though RGB-only input leads to reduced tracking and
mapping quality. GS-SLAM [56] introduces an adaptive
extension strategy for efficient map updates and reconstruc-
tion of novel regions. RGBD GS-ICP SLAM [14] incor-
porates Generalized ICP with 3DGS to improve localiza-
tion precision.
Splat-SLAM [35], based on the tracking
framework of DROID-SLAM [46], achieves state-of-the-
art accuracy in indoor scenes using only RGB input, but
its performance in outdoor environments remains unveri-
fied. VPGS-SLAM [7] is an independent work shortly after
our study. It maintains LiDAR sensors as input, thereby in-
curring higher hardware costs typical of such LiDAR-based
systems. Overall, current 3DGS-based SLAM systems tend
to rely on RGB-D or LiDAR data and are mostly evaluated
on indoor scenes, with limited validation on large-scale or
long-term outdoor sequences.
3. Method
GigaSLAM maps large-scale outdoor environments from
monocular RGB input using a hierarchical sparse voxel
structure. An overview is shown in Figure 2.
3.1. Preliminaries
Gaussian Splatting [17]
A Gaussian primitive G
=
(c, s, α, µ, Σ) is defined by its color c ∈R3, scale s ∈R3,
opacity α ∈R, mean vector µ ∈R3, and diagonal co-
variance matrix Σ, which represents the ellipsoidal shape
and position of the Gaussian in 3D. Each Gaussian prim-
itive G is derived from a sparse voxel Vt,i via a neural
network decoder, represented as a multi-layer perceptron
Fθ(·) with learnable parameters θ. At each time t, the map
Mt = (Vt, Gt) consists of sparse voxels Vt,i and Gaussian
splats Gt, with each Vt,i denoting the i-th voxel at time t.
For rendering, the primitives are sorted by their distance
to the camera, and alpha compositing is applied. Each 3D
Gaussian N(µ, Σ) projects onto the image plane as a 2D
Gaussian N(µI, ΣI):
µI = −π(ξ · µ),
ΣI = JRΣRT JT ,
(1)
where π(·) is the projection function, ξ ∈SE(3) is the cam-
era pose, J is the Jacobian, and R is the camera rotation
matrix. This setup ensures end-to-end differentiability of
the 3D Gaussian splatting.
The pixel color Cp at a pixel position p = (u, v) and the
depth DGS
p
at p are computed as:
Cp =
|Gt|
X
i=1
ciαi
i−1
Y
j=1
(1 −αj), DGS
p
=
|Gt|
X
i=1
ziαi
i−1
Y
j=1
(1 −αj), (2)
where zi is the camera-to-mean distance for the i-th Gaus-
sian.
Voxelized Gaussian Representation
Our work leverages
Scaffold-GS [22] for a voxelized 3D representation. This
representation offers fast rendering, quick convergence, and
improved memory efficiency by encoding 3DGS data into
feature vectors.
Additionally, it allows multiple points
mapped to the same voxel to be merged into a single voxel
representation, reducing memory usage by minimizing re-
dundancy. The scene is divided into sparse voxels. The
center position of each voxel is referred to as the anchor
position V ∈RN×3, each containing a context feature
vector ˆfv ∈R32, a scaling factor lv ∈R3, and k off-
sets Ov ∈Rk×3. Each voxel is decoded into k Gaussians
via shared MLPs. Using MLPs Fα, Fcolor, Fquan, Fscaling, the
distance δ = ∥xv −xc∥2 from the voxel to the camera, and
the viewing direction ⃗dvc = (xv−xc)/δ, the corresponding
factors of Gaussian primitives are generated by,
{αi}k
i=1 = Fα( ˆfv, δ, ⃗dvc),
(3)
ci, Σi and si can be got in similar way and µi is calculated:
{µi}k
i=1 = xv + {Ov,i}k
i=1 · lv.
(4)
Once decoded by MLPs, the Gaussians are rendered
through the Splat operation as described in the previous
part.
3

<!-- page 4 -->
Metric Depth
Estimation
Gaussians
Decoder
Tracking & Pose Refinement
Update
Key Point Match
E / PnP Tracking
Keyframe Opt.
Spatial Hash
Frame
Feature Update
RGB Stream
New Anchors
Unique Anchors
Level-of-Detail 
Representation
Layer k
Layer k+1
Layer k-1
...
...
Loop Detection
Recon. Loss
Trajectory Output
Rendering Output
ئ
ح
MLPs
Figure 2. Overview of our algorithm. GigaSLAM processes monocular RGB input to map large-scale outdoor environments using a
hierarchical sparse voxel structure. By this structure we solve the challenging problem at long distances outdoor scenarios.
3.2. Mapping
In large-scale SLAM, the key challenge is choosing a scal-
able, expressive, and flexible map representation for out-
door environments. Unlike indoor scenes, the open nature
of outdoor environments makes implicit representations like
NeRF impractical, as they struggle with infinite extents and
dynamic depth ranges. For 3D GS, the large number of dis-
tant Gaussian primitives can reduce rendering efficiency, as
they contribute little to the output but significantly increase
computational load.
Hierarchical Representation
Using a voxel-based repre-
sentation we establish a hierarchical structure by adjusting
voxel size. As shown in Figure 3 (left), increasing voxel
detail requires more computational resources but provides
limited improvement for distant elements like buildings or
the sky. Therefore, a 3D GS representation with finer voxels
for close scenes and coarser ones for distant scenes is ben-
eficial. LoD also resolves potential “collision” issues (right
side of Figure 3), where Gaussians from previous views
overlap with those in subsequent frames in a long sequence,
which will have a negative impact on camera pose track-
ing. By applying coarser details to distant Gaussians, LoD
ensures that nearby reconstructions remain clear, enhancing
efficiency and accuracy in large-scale outdoor mapping.
We voxelize the point cloud with varying voxel sizes
based on camera distance, creating a hierarchical structure
for rendering efficiency. The scene is divided into multiple
levels, each with different resolutions, from fine to coarse.
Given m levels of voxel sizes {ϵ1, · · · , ϵm} and LoD thresh-
olds {r1, · · · , rm−1}, each voxel is assigned a specific level
of detail L ∈N, and sparse voxels within the field of view
are selected based on distance. The voxelization process
proceeds as follows:
V =
P
ϵl

· ϵl | ϵl ∈{ϵ1, · · · , ϵm}

,
(5)
where P ∈R3 is the point cloud position, and ϵl corre-
sponds to the voxel size at level l, determined by the camera
distance. The voxelization process is similar to Octree-GS
[31] but we do not maintain an octree.
When rendering, appropriate sparse voxels are then se-
lected based on their proximity to the camera. To determine
the voxel selection mask, we calculate the Euclidean dis-
tance dv = ∥xv −xc∥2 between each voxel xv and the
camera xc. For each level l, voxels are selected if: 1) They
fall within the distance range [rl−1, rl); 2) Their level label
L(xv) matches the current level l; 3) They are visible from
the camera. The mask is computed as:
mask′
i =









1,
dv < r1 & Li == 1,
1,
rl−1 ≤dv < rl & Li == l,
1,
dv ≥rm−1 & Li == m,
0,
otherwise
maski = mask′
i & visible(xv).
(6)
This mask ensures that only voxels within the specified
level distance range and visible to the camera are selected,
which will be feed into MLPs to generate Gaussians for
Splat operation mentioned in Section 3.1.
Map Update
We optimize the map representation based
on the method in [17], using a combination of L1 color dis-
tance and SSIM to constrain the current view.
Lrender =
M
X
m=1
λ1∥IGS
m −Igt
m∥1 + λ2 SSIM(IGS
m , Igt
m), (7)
where m represents the pixel coordinates of the current im-
age, IGS is the rendered RGB image, Igt is the ground truth
RGB image, SSIM(, ) is the D-SSIM term, and λ1 and λ2
are hyperparameters.
4

<!-- page 5 -->
Near View
Far View
Level-of-Detail  Representation
Layer k
Layer k+1
Layer k-1
...
...
Voxel Size = 0.5 
Num. of Anchors: 1x
Voxel Size = 0.05
Num. of Anchors: 1.2x
Voxel Size = 0.01
Num. of Anchors: 1.89x
Voxel Size = 0.005
Num. of Anchors: 2.55x
ئ
ح
With Level-of-Detail
No Level-of-Detail
Figure 3. Efficiency and effectiveness of LoD Map Representation. (Left) Voxel refinement improves reconstruction for nearby objects,
with diminishing gains for distant scenes. A hierarchical 3DGS approach balances coarser and finer voxel representations. (Right) “Colli-
sion” issue where Gaussians from distant views overlap with those in subsequent frames in a long sequence.
To improve the geometric accuracy of 3D Gaussian
depth rendering, we apply the smoothing method from [6]
to reduce overfitting in sparse monocular input settings:
Lsmooth =
X
dj∈Adj(di)
1ne(di, dj)|di −dj|2,
(8)
where Lsmooth denotes the smoothness loss, Adj(di) repre-
sents the set of neighboring points to di, and 1ne(di, dj) is
an indicator function for whether di and dj form a geomet-
ric edge. |di −dj|2 is the Euclidean distance between di
and dj, with edges extracted using a Canny operator [5].
We bring isotropic regularisation in MonoGS [23] to pe-
nalize primitives with a high aspect ratio:
Liso =
|G|
X
i=1
∥si −¯si · 1∥1,
(9)
where |G| is the number of Gaussian primitives being opti-
mized, and ¯s is the mean scaling.
The total loss function for optimization is defined as:
Ltotal = Lrender + λiLiso + λsLsmooth,
(10)
where λi andλs are hyperparameters.
Map Expansion
A key challenge in large-scale dynamic
expansion is the overlap of newly created voxels with ex-
isting ones. Due to the complex geometry of long outdoor
sequences, current 3D GS SLAM methods struggle to effi-
ciently detect such duplicates. A fast method is needed to
check for voxel duplication, especially when expanding into
unexplored areas and registering new maps. In our system,
new voxels are generated by estimating the camera pose ξ,
transforming the metric depth into a point cloud, and vox-
elizing it hierarchically based on distance. Due to signifi-
cant visual overlap between consecutive viewpoints, many
newly created voxels already exist in the map, causing re-
dundancy.
Our scene representation addresses this issue through a
spatial hashing mechanism. Specifically, we deduplicate the
anchor points of voxels by applying a spatial hash function
[48]:
h(x) =
 d
M
i=1
xiπi
!
mod T,
(11)
where x = (x1, x2, x3) is the 3D coordinate of the an-
chor point, πi are prime numbers (following the setting of
[25], π1 = 1, π2 = 2654435761, π3 = 805459861), and
T = 263. This spatial hashing method allows us to perform
deduplication in constant time, ensuring that only unique
voxels are retained in the map. By reducing redundant voxel
entries, this approach is essential for efficient large-scale
voxelized Gaussian Splatting SLAM, minimizing memory
usage and computational overhead while preserving an ac-
curate map in overlapping regions.
3.3. Camera Tracking
Online Pose Tracking
We develop our tracking module
based on DF-VO [59], using optical flow differences for
2D-2D and 2D-3D tracking. Depth extraction is performed
using networks like Monodepth [13], DPT [30], and Depth
Anything [57], which work well for short sequences but suf-
fer from metric ambiguity, degrading SLAM performance.
UniDepth [29] mitigates this ambiguity by standardizing the
camera space transformation, allowing us to recover metric
depth before tracking. To maintain depth consistency, we
use RANSAC to correct scale errors between frames as de-
scribed in [59].
Feature points are extracted using DISK [52], and the
matched point pairs are used to estimate motion with Light-
Glue [20].
These methods first establish 2D-2D corre-
spondences, which are used to estimate camera motion via
epipolar geometry [59]. Specifically, given a pair of images
(Ii, Ij), we could obtain a set of 2D-2D correspondences
(pi, pj).
Using the epipolar constraint, the fundamental
matrix F or the essential matrix E can be solved, where
5

<!-- page 6 -->
F = K−T EK−1 is related to the camera intrinsics K. De-
composing F or E then allows us to recover the camera mo-
tion parameters [R, t] of ξ. If epipolar geometry fails due to
motion degeneracy or scale ambiguity we employ the Geo-
metric Robust Information Criterion (GRIC) [59] to select
the appropriate motion model. In cases where the essential
matrix is unreliable, we switch to the Perspective-n-Point
(PnP) method, which estimates the camera pose by min-
imizing reprojection error using 2D-3D correspondences.
Further details are provided in the Appendix.
3.4. Loop Closure
In our system, we integrate a proximity-based loop clo-
sure detection with a traditional SLAM back-end to enhance
long-term localization accuracy. We detect loop closures
using image retrieval techniques based on DBoW2 [9], fol-
lowed by Sim(3) optimization [21, 39] with a smoothness
term and loop closure constraints:
arg min
S1,··· ,SN
N
X
i
∥logSim(3)(∆S−1
i,i+1 · S−1
i
· Si+1)∥2
2
+
L
X
(j,k)
∥logSim(3)(∆Sloop
jk · S−1
j
· Sk)∥2
2,
(12)
where Si is the absolute similarity of keyframe i, the
first term is the smoothness term between consecutive
keyframes, the second term represents the error for the loop
closure between keyframes j and k, ∆S is the relative sim-
ilarity between keyframes. Details could be found in the
supplementary material.
After pose update, Our approach applies a rigid trans-
formation to all voxels across different levels to maintain
spatial consistency. We update all voxels in the global coor-
dinate frame to align with the optimized camera poses. Let
pi ∈R3 be the i-th anchor point, originally associated with
the j-th camera pose. Given the original pose T(j)
old ∈SE(3)
and the optimized pose T(j)
new ∈SE(3), the updated position
of the anchor point is given by:
pnew
i
= T(j)
new ·

T(j)
old
−1
·

pi
1

.
(13)
To reduce memory consumption for large-scale voxels,
we process the update in batches. The entire anchor point
set of voxels P = {p1, . . . , pN} is divided into M disjoint
subsets B1, . . . , BM, each of size B ≪N. Within each
batch Bm, the update rule remains consistent, but applied
locally:
∀p ∈Bm,
pnew = T(π(p))
new
·

T(π(p))
old
−1
·

p
1

, (14)
where π(p) maps each point p to its associated camera
pose index. This ensures that anchor points remain correctly
aligned across the updated camera coordinate frames. Sub-
sequently, a re-voxelization process (Eq. 5) is required to
adaptively refine the map structure. To efficiently manage
voxels introduced by loop closure corrections, we leverage
the spatial hashing mechanism (Eq. 11), which enables fast
lookup of updated voxels.
4. Experiments
4.1. Experimental Setup
We designed our experimental setup to evaluate Gi-
gaSLAM’s scalability and versatility across diverse envi-
ronments by using large outdoor datasets, with comprehen-
sive metrics assessing both tracking accuracy and map qual-
ity.
4.1.1. Dataset and Metrics
We evaluate our system on datasets: KITTI [11], KITTI
360 [18] 4 Seasons[55], A2D2[12]. KITTI is the primary
dataset due to its kilometer-scale, long sequences, offering
a challenging outdoor environment for SLAM. Unlike other
methods, which are limited to smaller datasets or indoor
scenes, our approach effectively handles large-scale outdoor
scenarios.
For tracking accuracy, we report the Absolute Trajectory
Error (ATE) [41] of the keyframes on three dataset: KITTI
[11], KITTI 360 [18] 4 Seasons[55]. Due to the absence
of GT pose matrices in the A2D2 dataset, ATE computa-
tion becomes infeasible. We thus visualize our algorithm’s
tracking performance through projecting of its trajectory on
Google Map in Figure 8. Mapping quality is evaluated us-
ing photometric rendering metrics such as Peak Signal-to-
Noise Ratio (PSNR) [54], Structural Similarity Index Mea-
sure (SSIM) [54], and Learned Perceptual Image Patch Sim-
ilarity (LPIPS) [60], which collectively capture both pixel-
level and perceptual differences in rendered images.
4.1.2. Implementation Details
Our experiments were conducted on machine with Ubuntu
22.04, and equipped with 12 Intel Xeon Gold 6128 3.40
GHz CPUs, 67GB of RAM, and an NVIDIA RTX 4090
GPU with 24GB of VRAM for the majority of tests. For
certain ultra-long sequences, such as those in KITTI and
KITTI-360 that exceed 4,000 frames, we utilized a high-
memory system with 128GB of RAM, 20 Intel Xeon Plat-
inum 8467C CPUs, and an NVIDIA L20 GPU with 48GB
of VRAM to accommodate large-scale outdoor scenes. Our
SLAM pipeline builds on the code structure of MonoGS in
PyTorch, leveraging CUDA to accelerate splatting opera-
tions. To ensure runtime efficiency, we use a multi-process
setup for tracking, mapping and loop closure.
6

<!-- page 7 -->
MonoGS
GT Traj
Splat SLAM
（Huge Gaussians Removed）
Ours LoD 0
GT Pointcloud
End
Start
Splat SLAM
Inconsist Scale (Loop Closure Failed) 
Details
Splat SLAM
Details
Ours LoD 0
Figure 4. This figure evaluates rendering and geometric quality via global view on KITTI 06 Sequence. Splat SLAM exhibits severe scale
inconsistency , whereas our method maintains precise scale coherence even in long outdoor sequences.
Methods
LC
Render
Avg.
00
01
02
03
04
05
06
07
08
09
10
seq. frames
-
-
2109
4542
1101
4661
801
271
2761
1101
1101
4071
1591
1201
seq. length (m)
-
-
2012.243
3724.19
2453.20
5067.23
560.89
393.65
2205.58
1232.88
649.70
3222.80
1705.05
919.52
contains loop
-
-
-
✓
✗
✓
✗
✗
✓
✓
✓
✓
✓
✗
ORB-SLAM2 (w/o LC) [26]
✗
✗
69.727
40.65
502.20
47.82
0.94
1.30
29.95
40.82
16.04
43.09
38.77
5.42
ORB-SLAM2 (w/ LC) [26]
✓
✗
54.816
6.03
508.34
14.76
1.02
1.57
4.04
11.16
2.19
38.85
8.39
6.63
LDSO [10]
✓
✗
22.425
9.32
11.68
31.98
2.85
1.22
5.10
13.55
2.96
129.02
21.64
17.36
DF-VO [59]
✗
✗
16.440
14.45
117.40
19.69
1.00
1.39
3.61
3.20
0.98
7.63
8.36
3.13
DROID-VO [46]
✗
✗
54.188
98.43
84.20
108.80
2.58
0.93
59.27
64.40
24.20
64.55
71.80
16.91
DPVO [47]
✗
✗
53.609
113.21
12.69
123.40
2.09
0.68
58.96
54.78
19.26
115.90
75.10
13.63
DROID-SLAM [46]
-
✗
100.278
92.10
344.60
107.61
2.38
1.00
118.50
62.47
21.78
161.60
72.32
118.70
DPV-SLAM [21]
✓
✗
53.034
112.80
11.50
123.53
2.50
0.81
57.80
54.86
18.77
110.49
76.66
13.65
DPV-SLAM++ [21]
✓
✗
25.749
8.30
11.86
39.64
2.50
0.78
5.74
11.60
1.52
110.90
76.70
13.70
MonoGS [23]
✗
✓
/
failed
543.47
failed
failed
20.75
failed
137.22
failed
failed
failed
failed
Splat-SLAM [35]
-
✓
/
83.07×
failed
failed
3.40
1.72
33.01×
130.75
14.35
52.07×
27.42×
63.55
Ours (w/o LC)
✗
✓
16.437
7.09
129.74
12.34
2.49
2.25
5.92
2.61
2.59
9.48
4.03
2.27
Ours (w/ LC)
✓
✓
15.576
6.83
127.39
11.30
2.18
1.88
4.36
2.11
2.12
7.04
3.94
2.18
Table 1. Camera Tracking Results (ATE RMSE [m] ↓) on the KITTI Dataset. LC denotes loop closure. DROID-SLAM and Splat-SLAM
use implicit loop detection via the pose factor graph, which works indoors but fails in large outdoor environments (see suppl.). [num]×
indicates that Splat-SLAM crashes in Mapping Mode, where values are obtained with Tracking-only Mode. Our method is the only
approach capable of achieving high-fidelity rendering from the current viewpoint while maintaining relatively strong tracking performance
on long-sequence outdoor datasets.
Given the detail richness of outdoor scenes, we use a
rendering resolution of 480 pixels in width (scaled propor-
tionally in height) to optimize computational efficiency and
memory usage. To evaluate reconstruction quality, these
rendering images are upsampled to the original resolution
using bicubic interpolation for efficiency. Employing a deep
learning-based super-resolution algorithm, however, may
yield higher reconstruction quality than the values reported
in this section.
4.2. KITTI Dataset
Table 1 presents the tracking performance of our method on
the KITTI [11] dataset (data from DPV-SLAM [21]). Over-
all, our method shows strong tracking performance, partic-
ularly on long and complex sequences. Figure 7 compares
camera trajectories on sequence 00, where our approach
maintains stable and accurate pose estimates. In contrast,
DROID-SLAM [46] exhibits significant scale drift on long
sequences, indicating limited robustness.
MonoGS [23]
performs worse—it crashes after a few hundred frames and
fails to continue tracking. As shown in Figure 6 and Table 3,
Splat-SLAM (PSNR 17.90 db)
Ours (PSNR 24.17 db)
Traj. of Seq. 06
Frame #330
Frame #725
Figure 5.
Comparison of rendering efficiency between Splat-
SLAM and our method on KITTI-06. Splat-SLAM suffers effi-
ciency drops at U-turns due to excessive visible Gaussians, affect-
ing distant detail, while our method remains stable.
the inaccurate poses from MonoGS degrade both mapping
and tracking quality, making it unsuitable for KITTI’s long,
outdoor sequences. Our method, however, delivers consis-
tently accurate poses over extended trajectories, showing
robustness unmatched by these baselines.
Figure 4 qualitatively compares reconstructions from
our method (Ours LoD 0), MonoGS, and Splat SLAM on
KITTI 06. From left to right, it shows global reconstruc-
tion, ground truth pointcloud and trajectory, and zoom-ins
7

<!-- page 8 -->
MonoGS
GT
Ours
Seq. 08
Splat-SLAM
MonoGS
GT
Ours
Seq. 07
Splat-SLAM
MonoGS
GT
Ours
Seq. 09
Splat-SLAM
MonoGS
GT
Ours
Seq. 10
Splat-SLAM
(a) Sequence 07 & 08
MonoGS
Ours
Seq. 08
Splat-SLAM
MonoGS
GT
Ours
Seq. 09
Splat-SLAM
MonoGS
GT
Ours
Seq. 10
Splat-SLAM
(b) Sequence 09 & 10
Figure 6. Rendering results on the KITTI dataset are shown for MonoGS (with RGB input) Splat-SLAM and our proposed method. Our
method maintains stable rendering across extended outdoor sequences, while MonoGS and Splat-SLAM struggle on this scene.
Ours (w/ LC)
Splat-SLAM
 (Tracking-only Mode)
DROID SLAM
MonoGS 
(Crashed during running)
KITTI Dataset
4 Seasons Dataset
Neighborhood (Ours)
Business Campus (Ours)
Figure 7. (Left) Trajectory estimation of different SLAM methods
on sequence 00 of the KITTI dataset with RGB input. Our method
demonstrates stable tracking over long outdoor scenes, unlike the
scale drift in DROID-SLAM and the tracking failure of MonoGS.
(Right) Our method on 4 Seasons dataset.
on local geometry. MonoGS outputs sparse, noisy maps
with major detail loss. Splat SLAM captures more struc-
ture but suffers from scale inconsistency due to failed loop
closure. Our method reconstructs more consistent geometry
throughout the sequence and better preserves scene details,
even in distant areas. Zoom-in views highlight our strength
in capturing thin structures and avoiding over-splatting.
4.3. KITTI 360, 4 Seasons & A2D2 Dataset
To further evaluate the robustness of our method in exten-
sive outdoor scenarios, we conducted an experiment on the
KITTI 360 [18], 4 Seasons[55] & A2D2[12] dataset. Un-
like the KITTI [11] dataset, where sequence lengths peak
at 4,661 frames with an average of approximately 2,109
frames, KITTI 360 sequences are significantly longer, av-
eraging 8,497 frames and reaching a maximum length of
14,607 frames. So do 4 Seasons and A2D2 datasets that
have an average of 20,000 frame input. These extended tra-
jectories introduce unique challenges in monocular RGB-
based SLAM, particularly due to the accumulation of errors
over such long sequences, which can severely impact track-
ing accuracy and mapping fidelity.
Notably, almost no existing monocular RGB-based
SLAM or VO system has been fully evaluated on these
datasets to date. Our method, however, demonstrated the
capability to process these ultra-long sequences effectively,
providing stable and continuous camera pose estimations
across the full length of each sequence. Initial results (Fig-
ure 8 and Table 2) indicate acceptable performance on ultra-
long sequences, highlighting our system’s resilience in mit-
igating error accumulation over extended trajectories. On
the 4 Seasons dataset, the substantial increase in the number
of input frames leads to a dramatic rise in memory require-
ments, rendering DROID-SLAM inapplicable due to GPU
memory exhaustion. As shown, DROID-SLAM runs out of
memory on most sequences. Even in the two sequences
where it is able to run, our method significantly outper-
forms it. Although our approach demonstrates strong per-
formance on these sequences, potential loop closures were
8

<!-- page 9 -->
KITTI 360 Dataset
4 Seasons Dataset
Methods
Avg.
0000
0002
0003
0004
0005
0006
0007
0009
0010
Avg.
Business Campus
Office Loop
Old Town
Neighborhood
City Loop
seq. frames
8497
11518
14607
1031
11587
6743
9699
3396
14056
3836
20200
17280
15177
28999
11121
28424
seq. length (m)
6971.088
8403.15
11501.28
1378.73
9975.28
4690.74
7979.92
4887.51
10579.68
3343.49
4906.55
3132.38
3710.66
5258.60
2078.37
10352.74
contains loop
-
✓
✓
✗
✓
✓
✓
✗
✓
✗
-
✓
✓
✱
✓
✱
DROID-SLAM [46]
193.307
110.07
233.87
10.79
169.11
139.02
113.81
577.39
165.34
220.36
/
OOM
175.63
OOM
158.19
OOM
Ours (w/o LC)
61.402
25.53
56.59
29.13
116.15
20.51
28.52
203.02
16.83
56.35
99.098
8.81
36.85
69.71
7.48
372.64
Ours (w/ LC)
47.107
17.72
34.98
27.55
40.54
20.46
19.12
197.33
13.22
53.03
92.950
7.33
21.12
67.89
6.28
362.13
Table 2. Camera Tracking Results (ATE RMSE [m] ↓) on the KITTI 360 & 4 Seasons Dataset. OOM stands for Out-of-Memory. ✱marks
a closed-loop sequence with just one hardly detectable start-end loop point; DBoW failed to identify this latent loop closure.
KITTI 360 Dataset
Seq. 0005 (Ours)
Seq. 0005 (DROID SLAM) Seq. 0006 (DROID SLAM)
Seq. 0002 (Ours)
Seq. 0004 (Ours)
Seq. 0006 (Ours)
A2D2 Dataset
Ours on Munich (27451 Frames Input)
Ours on Ingolstadt (22232 Frames Input)
w/ LC
w/o LC
w/o LC
w/ LC
Figure 8. (Left) Camera trajectory on the KITTI 360 dataset. Unlike DROID SLAM, which fails due to implicit loop closure, our method
explicitly handles large-scale trajectories. (Right) Our method on A2D2 dataset.
Methods
Metrics
Avg.
00
01
02
03
04
05
06
07
08
09
10
MonoGS
PSNR ↑
11.09
10.09×
16.40
8.78×
11.83×
17.43
10.83×
12.66
8.69×
6.90×
9.71×
8.66×
(RGB)
SSIM ↑
0.38
0.43×
0.58
0.31×
0.36×
0.55
0.37×
0.43
0.34×
0.27×
0.27×
0.32×
[23]
LPIPS ↓
0.79
0.82×
0.66
0.85×
0.82×
0.55
0.84×
0.76
0.86×
0.89×
0.81×
0.84×
Splat-SLAM
PSNR ↑
/
20.27×
failed
failed
21.10
19.42
20.33×
17.90
20.72
20.48×
20.86×
12.48
(RGB)
SSIM ↑
/
0.77×
failed
failed
0.64
0.68
0.67×
0.61
0.70
0.65×
0.68×
0.31
[35]
LPIPS ↓
/
0.41×
failed
failed
0.59
0.52
0.55×
0.66
0.51
0.59×
0.58×
0.76
Ours
PSNR ↑
24.22
24.14
24.91
22.71
24.40
25.22
24.92
24.17
24.88
23.42
23.03
24.09
(RGB)
SSIM ↑
0.95
0.96
0.96
0.95
0.95
0.96
0.96
0.96
0.97
0.94
0.95
0.95
LPIPS ↓
0.31
0.28
0.33
0.33
0.33
0.30
0.28
0.30
0.25
0.29
0.34
0.35
Table 3. Rendering performance on KITTI dataset. [num]× in-
dicates that MonoGS or Splat-SLAM crashes before completing
all frames, and the values are averaged over the processed frames
before failure. failed indicates that the tracking module of Splat-
SLAM returned NaN values, causing the algorithm to fail for the
entire sequence.
not detected due to dataset limitations (i.e., DBoW failed to
find matches). Nevertheless, our system still maintains rea-
sonable tracking performance under such challenging con-
ditions. This underscores the scalability and robustness of
our approach for unbounded, long-sequence outdoor SLAM
tasks.
4.4. Ablation and further Studies
We performed ablation studies on KITTI seq. 06 (Table
4), incorporating a depth prior into MonoGS (RGB) and
adding the LoD GS module to its backend. Despite the
depth prior, MonoGS shows performance degradation (Sec-
tion 3.2). Integrating LoD GS partially mitigates this, and
with additional modules, our method outperforms others.
LoD GS improves rendering by adaptively selecting voxel
sizes based on distance, balancing detail and efficiency (Ta-
ble 5).
Method
ATE [m]
MonoGS
137.22
MonoGS + UniDepth
100.03
Ours w/ LoD GS only
47.33
Ours w/o LC
2.61
Ours w/ LC
2.11
Table 4. Ablation experi-
ments of components
We compared our method with
Splat-SLAM [35] on the KITTI
dataset and observed that while
Splat-SLAM performs well in in-
door environments, its rendering
time increases significantly with
the number of frames in large out-
door sequences. In contrast, our
method maintains more stable performance, as its render-
ing time does not scale as drastically. To ensure a fair com-
parison, both methods were tested with the same rendering
resolution, 3DGS CUDA rasterization code, and identical
optimization settings. As shown in Figure 5, in the KITTI-
06 sequence, Splat-SLAM’s active 3D Gaussian ratio spikes
to nearly 80% after two U-turns, leading to unstable com-
putation. Our method, leveraging a hierarchical voxelized
9

<!-- page 10 -->
3D Gaussian representation, effectively controls the num-
ber of Gaussians involved in optimization, ensuring more
stable and efficient performance in large-scale outdoor en-
vironments.
Level(s) Num.
Voxel Size
Distance Partition
Avg. Frustum Vox. Num.
PSNR
GPU MEM
1 level
[0.1]
[]
411,612
24.29 db
22.46 GiB
2 levels
[0.1, 0.25]
[20]
223,627
24.64 db
15.82 GiB
3 levels
[0.1, 0.25, 1]
[20, 40]
116,639
24.05 db
11.77 GiB
4 levels
[0.1, 0.25, 1, 5]
[20, 40, 80]
34,721
24.21 db
9.46 GiB
5 levels
[0.1, 0.25, 1, 5, 25]
[20, 40, 80, 160]
21,342
24.17 db
8.62 GiB
Table 5. Ablation study of LoD on KITTI 06 Seq.
5. Conclusion
We present GigaSLAM, the first SLAM system for long-
term, kilometer-scale outdoor sequences using monocular
RGB input.
By employing a hierarchical sparse voxel
structure and a metric depth module, GigaSLAM enables
efficient large-scale mapping and robust pose estimation.
Evaluated on the KITTI, KITTI-360, 4 Seasons and A2D2
datasets, our system demonstrates strong performance
and good scalability for outdoor SLAM tasks.
Look-
ing ahead, future work will focus on improving loop
closure detection, particularly under high-speed motion.
Enhancing system stability in such scenarios will be
key to advancing reliable SLAM for ultra-large-scale,
real-world applications. Moreover, extending GigaSLAM
to operate under more dynamic conditions and diverse
outdoor environments could further improve its practicality.
References
[1] Jia-Wang Bian, Yu-Huan Wu, Ji Zhao, Yun Liu, Le Zhang,
Ming-Ming Cheng, and Ian Reid. An evaluation of feature
matchers for fundamental matrix estimation. arXiv preprint
arXiv:1908.09474, 2019. 3
[2] Michael Bloesch, Jan Czarnowski, Ronald Clark, Stefan
Leutenegger, and Andrew J Davison. Codeslam—learning
a compact, optimisable representation for dense visual slam.
In Proceedings of the IEEE conference on computer vision
and pattern recognition, pages 2560–2568, 2018. 2
[3] Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Latif,
Davide Scaramuzza, Jos´e Neira, Ian Reid, and John J
Leonard. Past, present, and future of simultaneous localiza-
tion and mapping: Toward the robust-perception age. IEEE
Transactions on robotics, 32(6):1309–1332, 2016. 1, 2
[4] Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez,
Jose M. M. Montiel, and Juan D. Tardos. ORB-SLAM3: An
accurate open-source library for visual, visual–inertial, and
multimap SLAM. IEEE Transactions on Robotics, 37(6):
1874–1890, 2021. 1, 2
[5] John Canny. A computational approach to edge detection.
IEEE Transactions on pattern analysis and machine intelli-
gence, (6):679–698, 1986. 5
[6] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee.
Depth-regularized optimization for 3d gaussian splatting in
few-shot images. arXiv preprint arXiv:2311.13398, 2023. 5
[7] Tianchen Deng, Wenhua Wu, Junjie He, Yue Pan, Xirui
Jiang, Shenghai Yuan, Danwei Wang, Hesheng Wang,
and Weidong Chen.
Vpgs-slam: Voxel-based progressive
3d gaussian slam in large-scale scenes.
arXiv preprint
arXiv:2505.18992, 2025. 3
[8] Jakob Engel, Thomas Sch¨ops, and Daniel Cremers.
Lsd-
slam: Large-scale direct monocular slam.
In Computer
Vision–ECCV 2014:
13th European Conference, Zurich,
Switzerland, September 6-12, 2014, Proceedings, Part II 13,
pages 834–849. Springer, 2014. 1
[9] Dorian G´alvez-L´opez and Juan D Tardos. Bags of binary
words for fast place recognition in image sequences. IEEE
Transactions on robotics, 28(5):1188–1197, 2012. 2, 6, 4
[10] Xiang Gao, Rui Wang, Nikolaus Demmel, and Daniel Cre-
mers.
Ldso: Direct sparse odometry with loop closure.
In 2018 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pages 2198–2204. IEEE, 2018.
7
[11] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we
ready for autonomous driving? the kitti vision benchmark
suite. In 2012 IEEE conference on computer vision and pat-
tern recognition, pages 3354–3361. IEEE, 2012. 2, 6, 7, 8,
1, 5
[12] Jakob Geyer,
Yohannes Kassahun,
Mentar Mahmudi,
Xavier Ricou, Rupesh Durgesh, Andrew S Chung, Lorenz
Hauswald, Viet Hoang Pham, Maximilian M¨uhlegg, Sebas-
tian Dorn, et al. A2d2: Audi autonomous driving dataset.
arXiv preprint arXiv:2004.06320, 2020. 2, 6, 8
[13] Cl´ement Godard, Oisin Mac Aodha, and Gabriel J Bros-
tow.
Unsupervised monocular depth estimation with left-
right consistency. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 270–279,
2017. 5
[14] Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. Rgbd gs-icp
slam. arXiv preprint arXiv:2403.12550, 2024. 3
[15] Krishna Murthy Jatavallabhula, Soroush Saryazdi, Ganesh
Iyer, and Liam Paull.
gradslam: Automagically differen-
tiable slam. arXiv preprint arXiv:1910.10672, 2019. 2
[16] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat track & map 3d gaussians
for dense rgb-d slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
21357–21366, 2024. 1, 2, 3, 5
[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4):1–14, 2023. 1, 3, 4, 5
[18] Yiyi Liao, Jun Xie, and Andreas Geiger. Kitti-360: A novel
dataset and benchmarks for urban scene understanding in 2d
and 3d. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 45(3):3292–3310, 2022. 2, 6, 8, 1, 5
[19] Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Lars-
son, and Marc Pollefeys.
Pixel-perfect structure-from-
motion with featuremetric refinement. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 5987–5997, 2021. 2
10

<!-- page 11 -->
[20] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Polle-
feys. Lightglue: Local feature matching at light speed. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 17627–17638, 2023. 5, 2
[21] Lahav Lipson, Zachary Teed, and Jia Deng.
Deep patch
visual slam. In European Conference on Computer Vision,
pages 424–440. Springer, 2025. 6, 7, 4, 5
[22] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 3
[23] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 18039–18048, 2024. 1, 2, 3, 5, 7,
9
[24] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
2, 5
[25] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG), 41(4):1–15, 2022. 5
[26] Raul Mur-Artal and Juan D. Tardos.
ORB-SLAM2: An
open-source SLAM system for monocular, stereo, and RGB-
d cameras.
IEEE Transactions on Robotics, 33(5):1255–
1262, 2017. 1, 2, 7
[27] Raul Mur-Artal, J. M. M. Montiel, and Juan D. Tardos. ORB-
SLAM: A versatile and accurate monocular SLAM system.
IEEE Transactions on Robotics, 31(5):1147–1163, 2015. 1,
2, 5
[28] David Nist´er. An efficient solution to the five-point relative
pose problem. IEEE transactions on pattern analysis and
machine intelligence, 26(6):756–770, 2004. 3
[29] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia
Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth:
Universal monocular metric depth estimation. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 10106–10116, 2024. 2, 5, 4
[30] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 12179–12188, 2021. 5
[31] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. arXiv
preprint arXiv:2403.17898, 2024. 4
[32] Antoni Rosinol, John J Leonard, and Luca Carlone. Nerf-
slam: Real-time dense monocular slam with neural radiance
fields. In 2023 IEEE/RSJ International Conference on Intel-
ligent Robots and Systems (IROS), pages 3437–3444. IEEE,
2023. 1
[33] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary
Bradski. Orb: An efficient alternative to sift or surf. In 2011
International conference on computer vision, pages 2564–
2571. Ieee, 2011. 4
[34] Erik Sandstr¨om, Yue Li, Luc Van Gool, and Martin R Os-
wald. Point-slam: Dense neural point cloud-based slam. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 18433–18444, 2023. 3, 5
[35] Erik Sandstr¨om, Keisuke Tateno, Michael Oechsle, Michael
Niemeyer, Luc Van Gool, Martin R Oswald, and Federico
Tombari. Splat-slam: Globally optimized rgb-only slam with
3d gaussians. arXiv preprint arXiv:2405.16544, 2024. 2, 3,
7, 9, 1, 5
[36] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 1
[37] Hava T Siegelmann and Eduardo D Sontag. Turing com-
putability with neural nets. Applied Mathematics Letters, 4
(6):77–80, 1991. 1
[38] Hava T Siegelmann and Eduardo D Sontag. On the com-
putational power of neural nets. In Proceedings of the fifth
annual workshop on Computational learning theory, pages
440–449, 1992. 1
[39] Hauke Strasdat, J Montiel, and Andrew J Davison. Scale
drift-aware large scale monocular slam. Robotics: science
and Systems VI, 2(3):7, 2010. 6, 4
[40] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl
Ren, Shobhit Verma, et al. The replica dataset: A digital
replica of indoor spaces. arXiv preprint arXiv:1906.05797,
2019. 1
[41] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the evalua-
tion of rgb-d slam systems. In 2012 IEEE/RSJ international
conference on intelligent robots and systems, pages 573–580.
IEEE, 2012. 6
[42] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davi-
son. imap: Implicit mapping and positioning in real-time. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 6229–6238, 2021. 2
[43] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien
Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
Yuning Chai, Benjamin Caine, et al. Scalability in perception
for autonomous driving: Waymo open dataset. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 2446–2454, 2020. 2
[44] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Prad-
han, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron,
and Henrik Kretzschmar. Block-nerf: Scalable large scene
neural view synthesis. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
8248–8258, 2022. 1
[45] Chengzhou Tang and Ping Tan. Ba-net: Dense bundle ad-
justment network. arXiv preprint arXiv:1806.04807, 2018.
2
[46] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam
for monocular, stereo, and rgb-d cameras. Advances in neu-
ral information processing systems, 34:16558–16569, 2021.
2, 3, 7, 9, 1, 5
11

<!-- page 12 -->
[47] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch vi-
sual odometry. Advances in Neural Information Processing
Systems, 36, 2024. 7, 5
[48] Matthias Teschner, Bruno Heidelberger, Matthias M¨uller,
Danat Pomerantes, and Markus H Gross. Optimized spa-
tial hashing for collision detection of deformable objects. In
Vmv, pages 47–54, 2003. 5
[49] Philip HS Torr, Andrew W Fitzgibbon, and Andrew Zisser-
man. The problem of degeneracy in structure and motion
recovery from uncalibrated image sequences. International
Journal of Computer Vision, 32:27–44, 1999. 4
[50] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandstr¨om,
Stefano Mattoccia, Martin R Oswald, and Matteo Poggi.
How nerfs and 3d gaussian splatting are reshaping slam: a
survey. arXiv preprint arXiv:2402.13255, 4, 2024. 1
[51] Haithem Turki,
Deva Ramanan,
and Mahadev Satya-
narayanan.
Mega-nerf:
Scalable construction of large-
scale nerfs for virtual fly-throughs.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 12922–12931, 2022. 1
[52] Michał Tyszkiewicz, Pascal Fua, and Eduard Trulls. Disk:
Learning local features with policy gradient. Advances in
Neural Information Processing Systems, 33:14254–14265,
2020. 5, 2
[53] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20697–
20709, 2024. 2
[54] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 6
[55] Patrick Wenzel, Rui Wang, Nan Yang, Qing Cheng, Qadeer
Khan, Lukas von Stumberg, Niclas Zeller, and Daniel Cre-
mers. 4seasons: A cross-season dataset for multi-weather
slam in autonomous driving. In Pattern Recognition: 42nd
DAGM German Conference, DAGM GCPR 2020, T¨ubingen,
Germany, September 28–October 1, 2020, Proceedings 42,
pages 404–417. Springer, 2021. 2, 6, 8
[56] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
19595–19604, 2024. 3
[57] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi
Feng, and Hengshuang Zhao. Depth anything: Unleashing
the power of large-scale unlabeled data. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10371–10381, 2024. 5
[58] Sicheng Yu, Chong Cheng, Yifan Zhou, Xiaojun Yang, and
Hao Wang. Rgb-only gaussian splatting slam for unbounded
outdoor scenes. arXiv preprint arXiv:2502.15633, 2025. 2
[59] Huangying Zhan, Chamara Saroj Weerasekera, Jia-Wang
Bian, Ravi Garg, and Ian Reid. Df-vo: What should be learnt
for visual odometry?
arXiv preprint arXiv:2103.00933,
2021. 5, 6, 7, 3, 4
[60] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 6
[61] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo
Poggi. Go-slam: Global optimization for consistent 3d in-
stant reconstruction. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, pages 3727–3737,
2023. 1, 2, 3, 5
[62] Zhengyou Zhang. Determining the epipolar geometry and
its uncertainty: A review. International journal of computer
vision, 27:161–195, 1998. 3
[63] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 12786–12796, 2022.
1, 2, 5
[64] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui,
Martin R Oswald, Andreas Geiger, and Marc Pollefeys.
Nicer-slam: Neural implicit scene encoding for rgb slam.
arXiv preprint arXiv:2302.03594, 2023. 1, 2, 3, 5
12

<!-- page 13 -->
GigaSLAM: Large-Scale Monocular SLAM with Hierarchical Gaussian Splats
Supplementary Material
A. What Challenges Are We Facing on Outdoor
Long-Sequence Datasets?
Most monocular SLAM methods are well-validated on small-
scale datasets but remain underexplored in large-scale outdoor
long-sequence scenarios. DROID-SLAM [46], though effec-
tive indoors, struggles with scale errors and computational
overhead in outdoor datasets due to its reliance on dense bun-
dle adjustment within a factor graph. MonoGS [23], based on
3D Gaussian Splatting (3DGS) [17], offers a promising alter-
native with its explicit representation, enabling high-fidelity
mapping in unbounded environments.
However, both ap-
proaches face significant challenges when applied to outdoor
long-sequence datasets, as discussed in the following sections.
A.1. DROID-SLAM based Methods
DROID-SLAM [46], introduced in 2021, represents a signif-
icant advancement in SLAM systems by making the entire
SLAM pipeline fully differentiable. This innovation enabled
the seamless integration of SLAM with deep learning tech-
niques, outperforming traditional non-learning-based SLAM
algorithms on smaller-scale datasets. Before DROID-SLAM,
deep learning-based SLAM methods struggled to match the
performance and reliability of classical approaches.
At its core, DROID-SLAM reformulates the SLAM prob-
lem as a joint optimization task, minimizing errors in pose
estimation and map construction. It employs a recurrent itera-
tive structure, leveraging the Turing-complete nature of recur-
rent neural networks (RNNs) [37, 38] to perform iterative op-
timization. This structure allows DROID-SLAM to iteratively
refine dense correspondences and pose estimates, utilizing
dense optical flow for robust matching between keyframes.
Furthermore, DROID-SLAM integrates a dense bundle ad-
justment mechanism based on factor graph optimization, en-
abling accurate pose refinement without requiring explicit
loop closure modules.
The system’s differentiable design and robust tracking ca-
pabilities have made it a foundational component in subse-
quent SLAM algorithms, such as GO-SLAM [61], NeRF-
SLAM [32] and Splat-SLAM [35]. These methods extend
DROID-SLAM’s principles, combining them with advanced
mapping techniques like Neural Radiance Fields (NeRF)
[24] or 3D Gaussian Splatting [17].
Such integrations
have achieved state-of-the-art performance on indoor datasets,
highlighting DROID-SLAM’s adaptability and theoretical ro-
bustness.
However, DROID-SLAM encounters significant chal-
lenges in outdoor, long-sequence datasets like KITTI [11] and
KITTI 360 [18], primarily due to its reliance on optical flow to
construct the factor graph. The core of DROID-SLAM lies in
performing Dense Bundle Adjustment (DBA) on this factor
graph to optimize pose and map estimates. While effective
in smaller-scale datasets with dense co-visibility, the factor
graph’s sparsity in outdoor settings limits the effectiveness of
this approach. Specifically, the co-visibility matrix in outdoor
sequences has fewer edges due to the reliance on optical flow,
which inherently reduces connections between frames. As a
result, the optimization window for the DBA module in large-
scale outdoor datasets is significantly smaller than in confined
indoor scenarios.
This small optimization window amplifies the accumu-
lation of scale drift, as errors that could otherwise be cor-
rected through a larger window of jointly optimized frames
remain unaddressed.
Figure A.1 illustrates this limitation
by comparing the co-visibility graphs of DROID-SLAM on
KITTI, KITTI 360 and Replica [40]. The co-visibility graph
for KITTI reveals sparse connections, primarily linking each
frame to its nearest 4 to 5 neighbors. In contrast, the graph
for the indoor Replica dataset is significantly denser, reflect-
ing DROID-SLAM’s inherent suitability for structured, small-
scale environments.
In addition to the sparsity challenge, maintaining even
a small factor graph and executing Dense Bundle Adjust-
ment over kilometer-scale sequences comes at a steep com-
putational cost. DROID-SLAM relies on implicit loop clo-
sure through inter-frame co-visibility relationships, which be-
comes less effective in outdoor scenarios where loops are in-
frequent and harder to detect. The computational resources
required to process and optimize the factor graph in such sce-
narios are prohibitively high, further limiting its applicability
to unbounded environments.
A.2. MonoGS
NeRF-based SLAM systems [61, 63, 64] have shown great
potential for high-quality scene reconstruction, particularly in
indoor environments. However, they face significant limita-
tions when applied to unbounded outdoor sequences. The pri-
mary challenge with NeRF lies in the slow training and ren-
dering speeds, which make it difficult to process large-scale,
long-sequence datasets in real-time. While recent techniques
like Block-NeRF [44] and Mega-NeRF [51] have made strides
in scalability, they are still not used for long-duration SLAM
tasks, especially in outdoor environments.
In contrast, MonoGS [23] addresses one of these chal-
lenges by using 3D Gaussian Splatting (3DGS) [17] instead
of NeRF. The key advantage of 3DGS is its ability to repre-
sent scenes with smooth, continuously differentiable Gaussian
blobs, which can be rendered efficiently at high frame rates.
1

<!-- page 14 -->
(a) Indoor Scene
(b) Replica Room 0
(c) Replica Room 2
(d) Replica Office 2
(e) Replica Office 3
(f) Outdoor Scene
(g) KITTI Seq. 00
(h) KITTI Seq. 05
(i) KITTI 360 Seq. 0000
(j) KITTI 360 Seq. 0004
Figure A.1. The co-visibility matrix of DROID-SLAM for the KITTI, KITTI 360 and Replica dataset.
This allows for real-time mapping and tracking. By adopting
3DGS, MonoGS overcomes the slow training and rendering
issues associated with NeRF-based methods, enabling high-
fidelity, real-time SLAM performance with just monocular
RGB input. MonoGS achieves this through innovations like
an analytic Jacobian for pose optimization, isotropic shape
regularization for geometric consistency, and a resource al-
location strategy that maintains map accuracy without com-
promising efficiency.
Thus, MonoGS not only solves some of the key pain points
of slow training and rendering inherent in NeRF but also
provides a more scalable and efficient solution for SLAM
tasks. Due to its seamless integration of 3DGS into the SLAM
framework, our method adopts MonoGS as its foundational
codebase.
MonoGS is particularly notable for its reliance
on 3DGS as both the scene representation and the founda-
tion of its tracking module. However, this heavy dependence
on 3DGS rendering introduces significant challenges, partic-
ularly in outdoor long-sequence scenarios. The tracking pro-
cess in MonoGS estimates camera poses by minimizing ren-
dering losses, which inherently requires 3DGS to produce
high-fidelity novel views. While this approach works well in
indoor environments—where the relatively small scale, sim-
ple scene structures, and limited camera motion allow 3DGS
to converge quickly—it becomes problematic in outdoor set-
tings with larger scales and more complex geometries.
In outdoor scenarios, the convergence of 3DGS rendering
is significantly slower due to the increased complexity and
scale of the scenes. As a result, MonoGS often attempts to op-
timize pose tracking using a partially converged or inaccurate
3DGS map. This leads to pose estimation errors, which ac-
cumulate over time, especially during large-scale camera mo-
tions. Furthermore, in outdoor settings, the inaccurate depth
of distant Gaussians introduces additional errors, making sub-
sequent pose tracking increasingly unreliable. These issues
create a feedback loop, where pose inaccuracies degrade the
3DGS map, which in turn further worsens pose estimation.
Over long sequences, this cycle can eventually destabilize the
entire system.
Figure A.2 provides a clear illustration of these limitations.
At the beginning of the sequence, although the mapping qual-
ity is suboptimal, the rendered scene remains recognizable,
indicating that 3DGS is functioning adequately for the initial
frames. However, as the sequence progresses, the cumula-
tive pose errors and mapping inaccuracies cause the rendered
scene to become increasingly distorted. By the second major
turn, MonoGS’s tracking module is overwhelmed by these er-
rors, leading to a complete breakdown of the system. The ren-
dered scene at this stage is entirely unrecognizable, demon-
strating the cascading failure of both the tracking and map-
ping components.
So what challenges are we facing on outdoor long-
sequence datasets? Outdoor long-sequence datasets expose
fundamental limitations in existing SLAM systems due to
their expansive scale, complex scene geometries, and diverse
camera motions.
Systems like DROID-SLAM encounter
scale drift and computational inefficiencies from sparse co-
visibility and small optimization windows, while MonoGS
struggles with slow 3D Gaussian Splatting convergence, lead-
ing to compounding pose and mapping inaccuracies
B. Details about Pose Tracking
To estimate camera motion between images, we begin by ex-
tracting image features using the DISK network [52], which
provides robust descriptors capable of capturing rich feature
representations. These features are then matched using Light-
Glue [20], a state-of-the-art deep learning-based matcher. By
leveraging adaptive filtering and dynamic weighting strate-
2

<!-- page 15 -->
Frame # 40
Frame # 55
Frame # 65
Frame # 80
Frame # 90
Frame # 105
Frame # 110
Frame # 125
Frame # 130
Frame # 135
Frame # 150
Frame # 160
Frame # 165
Frame # 175
Frame # 190
Frame # 195
Frame # 200 (Degradation Start)
Frame # 220
Frame # 240
Frame # 205
Frame # 250
Frame # 260
Frame # 270
Frame # 280
Frame # 300
Frame # 315
Frame # 325
Frame # 330 (Tracking Lost)
Frame # 360 (Tracking Lost)
Frame # 400 (Tracking Lost)
MonoGS
GT
MonoGS
GT
MonoGS
GT
MonoGS
GT
MonoGS
GT
MonoGS
GT
Figure A.2. MonoGS on KITTI Seqence 00.
gies, LightGlue establishes reliable correspondences, even in
challenging scenarios with low texture or significant view-
point changes.
We adopt the methodology from DF-VO [59] for estimat-
ing the poses with fundamental matrix F or essential matrix
E [1, 28, 62]. With the matched feature points pi and pj,
these matrices could be computed using the classical epipolar
constraint:
F = K−T EK−1,
E = [t] × R.
(B.1)
Here, pi and pj represent the homogeneous coordinates
of corresponding points in the two images, expressed as p =
[u, v, 1]T , where (u, v) are the pixel coordinates. The epipolar
constraint is enforced as:
pT
j K−T EK−1pi = 0.
(B.2)
The camera motion [R, t] is recovered by decomposing F
or E.
While effective in many scenarios, this approach can en-
counter challenges under certain conditions, such as motion
degeneracy (e.g., pure rotation) or scale ambiguity inherent in
the essential matrix.
To refine camera pose estimation, the Perspective-n-Point
(PnP) algorithm is employed [59], which minimizes the re-
projection error using 3D-2D correspondences:
e =
X
∥K(RXi + t) −pj∥2.
(B.3)
The required 3D information for 3D-2D correspondences
3

<!-- page 16 -->
is derived from dense depth maps extracted using the
UniDepth model [29]. By providing depth estimates from
monocular RGB images, UniDepth ensures a reliable repre-
sentation of the scene’s structure, mitigating depth ambiguity
in monocular setups.
To enhance robustness, we adopt the geometric robust in-
formation criterion (GRIC) [49, 59] as a model selection strat-
egy. GRIC evaluates the suitability of essential matrix decom-
position, identifying cases of motion or structure degeneracy.
The GRIC function is defined as:
GRIC =
X
ρ(e2
i ) + log(4)dn + log(4n)k,
(B.4)
with
ρ(e2
i ) = min

e2
i
2(r −d)σ2 , 1

.
(B.5)
Here, d is the structure dimension, n is the number of
matched features, k is the number of motion model param-
eters, and σ is the standard deviation of measurement error.
When GRICF exceeds GRICH, we switch to PnP, utilizing
UniDepth-derived depth information for improved pose esti-
mation.
By combining robust feature matching with LightGlue,
dense depth estimation from UniDepth, and GRIC-based
model selection, our system addresses the limitations of tra-
ditional epipolar geometry pipelines, achieving improved re-
silience and accuracy in monocular setups.
C. Loop Correction
In our system, we integrate proximity-based loop closure de-
tection with a traditional SLAM back-end, using image re-
trieval techniques to identify and correct loop closures, par-
ticularly for enhancing long-term localization accuracy. For
this, we utilize DBoW2 [9] for image retrieval by detecting
candidate image pairs that suggest a loop closure, extracting
ORB [33] features from each frame. These feature extraction,
indexing, and search operations occur concurrently in a sepa-
rate thread, minimizing runtime overhead. Additionally, Non-
Maximum Suppression (NMS) to prevent overly frequent de-
tections referred to [21]. We perform a Sim(3) optimization
for global pose estimates by optimizing a smoothness term
and loop closure constraints using the Levenberg-Marquardt
algorithm. The loop correction method closely follows the
work of [21, 39]. This method is a classic Sim(3)-based op-
timization approach that has been applied to various SLAM
methods over the past fifteen years. Given the SE(3) poses
of all keyframes, suppose a loop closure is detected between
frame j and frame k. Define the similarity transformation as
Si = (ti, Ri, si) ∈Sim(3), and compute the residual between
the two frames as:
rjk = logSim(3)(∆Sloop
jk · S−1
j
· Sk).
(C.1)
Without an explicit pose factor graph, a virtual factor graph
can be constructed by considering only the connections be-
tween adjacent frames.
The residual between consecutive
frames is defined as:
ri = logSim(3)(∆S−1
i,i+1 · S−1
i
· Si+1).
(C.2)
The objective function for optimization is then formulated as:
arg min
S1,··· ,SN
N
X
i
∥ri∥2
2 +
L
X
(j,k)
∥rjk∥2
2.
(C.3)
Expanding the residual terms gives:
arg min
S1,··· ,SN
N
X
i
∥logSim(3)(∆S−1
i,i+1 · S−1
i
· Si+1)∥2
2
+
L
X
(j,k)
∥logSim(3)(∆Sloop
jk · S−1
j
· Sk)∥2
2.
(C.4)
This
objective
is
optimized
using
the
Levenberg-
Marquardt (LM) algorithm. After optimization, the updated
similarity transformations Si are used to update the global
poses as Gi ←(ti, Ri).
To simplify the optimization, the objective function re-
duces to:
min ∥r(S)∥2 →min ∥logSim(3)(∆Sloop
jk ·S−1
j
·Sk)∥2. (C.5)
At each optimization step, the relative transformation ∆Sloop
jk
is first computed and treated as a constant C:
r = ∥logSim(3)(C · S−1
j
· Sk)∥2.
(C.6)
The Jacobian matrix is then computed as:
Jj = ∂r
∂Sj
,
Jk = ∂r
∂Sk
,
J = [Jj, Jk].
(C.7)
The update increment ∆S = [∆Sj, ∆Sk] is estimated using
a first-order Taylor approximation:
r ≈r + J∆S.
(C.8)
Thus, the optimization problem reduces to:
∆S = arg min ∥r + J∆S∥2
= arg min(r + J∆S)⊤(r + J∆S).
(C.9)
Expanding the expression:
∆S = arg min(∥r∥2+∆S⊤J⊤J∆S+2∆S⊤J⊤r). (C.10)
Taking the derivative with respect to ∆S and setting it to zero
leads to:
J⊤J∆S = −J⊤r.
(C.11)
4

<!-- page 17 -->
To prevent divergence due to large steps, a damping term
λ diag(J⊤J) is added, along with a small regularization term
ϵI to ensure numerical stability:
(J⊤J + λ diag(J⊤J) + ϵI)∆S = −J⊤r.
(C.12)
This results in solving a linear system of the form:
A∆x = b.
(C.13)
In the next iteration, the updated Sim(3) transformations
are used to recompute ∆Sloop
jk , and the process continues iter-
atively until convergence.
D. Visualization of Camera Trajectory on KITTI
and KITTI 360 Dataset.
The main paper presents trajectory comparisons for the KITTI
[11] and KITTI 360 [18] datasets, respectively.
This ap-
pendix provides a detailed discussion of these visualizations,
further illustrating the performance of our method in main-
taining accurate and stable camera pose estimates across long
sequences.
For KITTI, our method demonstrates consistent trajectory
alignment with ground truth across challenging sections of the
sequence as demonstrated in Figure C.1. For KITTI 360, Fig-
ure C.2 provides a more comprehensive evaluation of ultra-
long trajectories, spanning up to 14,607 frames.
Notably,
DROID-SLAM [46] achieves competitive performance in se-
quence 0003, where its trajectory slightly outperforms ours.
However, across the majority of sequences, DROID-SLAM
exhibits substantial scale drift, consistent with the challenges
described in Section A. These large-scale deviations under-
mine its ability to provide reliable pose estimates over ex-
tended sequences. In contrast, our method maintains stable
and accurate camera poses throughout, highlighting its re-
silience to error accumulation and scalability to unbounded
scenarios.
We also present the rendered images in Figure
C.3 and C.4 of our approach alongside those of MonoGS [23]
on the KITTI dataset. Consistent with the discussion in Sec-
tion A, MonoGS performs poorly on long-sequence outdoor
datasets such as KITTI, whereas our method demonstrates ro-
bust performance.
Overall, these visualizations underscore the adaptability
and robustness of our system across both traditional and ultra-
long outdoor SLAM tasks, with detailed trajectory plots re-
vealing its superior performance in maintaining trajectory fi-
delity.
Our paper closely follows a series of recent works, such as
[16, 21, 23, 34, 35, 46, 47, 61, 63, 64] et al., which use ATE as
a metric for evaluating tracking accuracy. However, we also
note that some more earlier works [59] use other metrics, such
as translation and rotation drift over segment lengths of 100
to 800 meters with loop closure disabled. Since recent works
have not adopted this metric and ATE is a better measure of
long-term tracking performance in long outdoor sequences,
we report ATE in the main paper and provide T/R Drift data
in the Table D.1 for reference.
Methods
Metric
00
01
02
03
04
05
06
07
08
09
10
ORB SLAM w/o LC
terr
11.43
107.57
10.34
0.97
1.3
9.04
14.56
9.77
11.46
9.3
2.57
[27]
rerr
0.58
0.89
0.26
0.19
0.27
0.26
0.26
0.36
0.28
0.26
0.32
DF-VO
terr
2.33
39.46
3.24
2.21
1.43
1.09
1.15
0.63
2.18
2.4
1.82
[59]
rerr
0.63
0.5
0.49
0.38
0.3
0.25
0.39
0.29
0.32
0.24
0.38
MonoGS
terr
/
99.4
/
/
7.34
/
101.73
/
/
/
/
[23]
rerr
/
27.02
/
/
3.57
/
10.82
/
/
/
/
Splat-SLAM
terr
17.97
/
/
2.41
33.78
5.4
33.28
10.05
12.67
7.13
30.82
[35]
rerr
1.62
/
/
0.33
0.56
0.48
0.49
0.42
0.72
0.24
3.71
Ours w/o LC
terr
1.38
41.08
1.49
2.21
2.23
1.49
1.6
1.06
2.29
1.05
1.19
rerr
0.43
0.92
0.39
0.58
0.19
0.39
0.39
0.47
0.61
0.28
2.27
Table D.1. Translation and rotation drift of different methods
E. Limitation of our work
Currently, research on kilometer-scale outdoor monocular
RGB SLAM using NeRF [24] or 3DGS [17] is still in its
nascent stages. Our method is specifically tailored for au-
tonomous driving scenarios and places less emphasis on other
scene types. Our approach is not the best solution for indoor
environments.
Moreover, limitations such as motion blur,
camera shake, glare, overexposure, and low-texture scenes
can reduce tracking accuracy, though these challenges are
more pronounced in non-driving scenarios. Additionally, the
memory requirements of NeRF or 3DGS present challenges
for city-scale scenes.
Future work could explore solutions to these issues, ex-
tending applicability beyond driving-focused datasets and fur-
ther improving robustness in various types of environments.
5

<!-- page 18 -->
Seq. 06
Seq. 07
Seq. 08
Seq. 09
Seq. 00
Seq. 02
Seq. 05
Seq. 03
Figure C.1. Camera trajectory visualization for the KITTI dataset.
Seq. 0000
Seq. 0002
Seq. 0004
Seq. 0005
Seq. 0006
Seq. 0009
Seq. 0003
Seq. 0007
Seq. 0010
Ours
DROID-SLAM
Ours
DROID-SLAM
Figure C.2. Camera trajectory visualization for the KITTI 360 dataset.
6

<!-- page 19 -->
Seq. 04
MonoGS
GT
Ours
Splat-SLAM
MonoGS
GT
Ours
Splat-SLAM
Seq. 05
MonoGS
GT
Ours
Seq. 06
Splat-SLAM
MonoGS
GT
Ours
Seq. 00
Splat-SLAM
MonoGS
GT
Ours
Seq. 03
Splat-SLAM
Figure C.3. Rendering result for KITTI dataset of Seq. 00 to 06.
7

<!-- page 20 -->
MonoGS
GT
Ours
Seq. 09
Splat-SLAM
MonoGS
GT
Ours
Seq. 08
Splat-SLAM
MonoGS
GT
Ours
Seq. 07
Splat-SLAM
MonoGS
GT
Ours
Seq. 10
Splat-SLAM
Figure C.4. Rendering result for KITTI dataset of Seq. 07 to 10.
8
