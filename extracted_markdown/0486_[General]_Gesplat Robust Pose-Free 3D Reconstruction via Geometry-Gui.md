<!-- page 1 -->
Gesplat: Robust Pose-Free 3D Reconstruction via
Geometry-Guided Gaussian Splatting
Jiahui Lu1, Haihong Xiao1, Xueyan Zhao1, Wenxiong Kang1
Abstract
Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have advanced
3D reconstruction and novel view synthesis, but remain heavily dependent on accurate
camera poses and dense viewpoint coverage. These requirements limit their applicabil-
ity in sparse-view settings, where pose estimation becomes unreliable and supervision
is insufficient. To overcome these challenges, we introduce Gesplat, a 3DGS-based
framework that enables robust novel view synthesis and geometrically consistent re-
construction from unposed sparse images. Unlike prior works that rely on COLMAP
for sparse point cloud initialization, we leverage the VGGT foundation model to obtain
more reliable initial poses and dense point clouds. Our approach integrates several key
innovations: 1) a hybrid Gaussian representation with dual position-shape optimization
enhanced by inter-view matching consistency; 2) a graph-guided attribute refinement
module to enhance scene details; and 3) flow-based depth regularization that improves
depth estimation accuracy for more effective supervision. Comprehensive quantitative
and qualitative experiments demonstrate that our approach achieves more robust per-
formance on both forward-facing and large-scale complex datasets compared to other
pose-free methods.
Keywords: Unposed sparse-view synthesis, 3D reconstruction
Email addresses: aujiahui@mail.scut.edu.cn (Jiahui Lu), auhhxiao@mail.scut.edu.cn
(Haihong Xiao), auxyzhao@scut.edu.cn (Xueyan Zhao), auwxkang@scut.edu.cn (Wenxiong Kang)
arXiv:2510.10097v2  [cs.CV]  27 Oct 2025

<!-- page 2 -->
GT
Ours
Instantsplat
NoPe-NeRF
CF-3DGS
PSNR:16.429
PSNR:14.588
PSNR:11.462
PSNR:10.439
GT
Ours
Instantsplat
NoPe-NeRF
CF-3DGS
PSNR:16.429
PSNR:14.588
PSNR:11.462
PSNR:10.439
Fig 1: Novel View Synthesis Comparisons.We introduce Gesplat, an efficient framework for novel view
synthesis and 3D reconstruction from sparse-view unposed inputs.Compared with other pose-free methods,
our method achieves higher PSNR and reconstructs more consistent geometry with finer details.
1. Introduction
Novel view synthesis and 3D reconstruction are long-standing fundamental goals
in computer vision, playing a critical role in applications like autonomous navigation,
VR/AR, and robotics. While existing methods can reconstruct scenes from multi-
ple posed images, acquiring dense, well-covered image sets in real-world scenarios
is often impractical and costly. The scarcity of views leads to insufficient supervi-
sion during training, causing artifacts and flawed reconstructions. Therefore, achieving
high-quality 3D reconstruction from only a few sparsely overlapping images remains a
critical challenge.
Neural Radiance Fields (NeRF)[1],an implicit neural representation that has shown
remarkable success in reconstructing scenes and synthesizing photorealistic novel views.
To improve performance under sparse views, NeRF-based methods have attempted in
the aspect of pretraining [2, 3], regularization terms [4, 5], and external priors [6, 7],
achieving significantly improved results. Nonetheless, these approaches still suffer
from high training and rendering costs.
More recently, 3D Gaussian Splatting (3DGS) [8] emerges as an efficient explicit
scene representation that employs Gaussian primitives for fast reconstruction and real-
time rendering. Subsequent studies [9, 10, 11] further reduce the number of required
training views while preserving competitive quality. Nonetheless, these approaches
generally assume known and accurate camera poses, which are rarely satisfied in prac-
2

<!-- page 3 -->
tical sparse-view settings. In fact, with limited viewpoint overlap, SfM pipelines such
as COLMAP [12] often fail to produce reliable camera parameters. Reconstructing
3D scenes from images without pose information thus remains a challenging yet es-
sential task. Existing pose-free techniques [13, 14, 15] eliminate the need for SfM
by jointly optimizing scene representation and camera poses end-to-end. However,
they typically assume dense video-level coverage and incur high computational costs.
InstantSplat [16] employs DUSt3R [17] for coarse geometric initialization, but often
trades geometric accuracy for consistency under sparse inputs. To overcome these lim-
itations, we introduce VGGT [18], a feed-forward model that generates high-fidelity
dense point clouds and reliable camera poses. By integrating convolutional inductive
biases with self-attention mechanisms, VGGT preserves fine scene details while ensur-
ing global consistency. Its powerful feature matching capability maintains robustness
under sparse views, and its strong generalization supports diverse scenes, opening new
avenues for 3D reconstruction and pose estimation.
In sparse-view settings, limited input data results in inconsistent scene geometry,
and the high interdependence among Gaussian attributes forces the optimization to
make a trade-off between optimizing shape and position. Some optimization-based
methods [19, 20] excel in real-time scene reconstruction, but when applied to sparse-
view scenarios, they usually introduce invalid geometric priors, leading to incorrect
topology and scale ambiguity. Incorporating additional geometric priors is essential
to constrain scene structure and improve the accuracy of 3D point positions. Depth
information is a commonly used prior due to its connection between 2D images and
3D structure. Some researches [21, 22] incorporate monocular depth priors [23] as
regularization, yet such handcrafted constraints often suffer from scale inaccuracy and
multi-view inconsistency, which can adversely affect final rendering quality.
In this paper, we propose Gesplat, an effective framework that incorporates appro-
priate geometric priors to constrain the scene structure while introducing optimization
and regularization techniques to refine the scene details. Inspired by [24], we adopt a
hybrid Gaussian representation combining ordinary and ray-based Gaussians leverag-
ing matching priors. By binding Gaussians to matching rays, we enhance multi-view
consistency and preserve geometric structure. Furthermore, we introduce dual opti-
3

<!-- page 4 -->
mization of position and shape to stabilize training and ensure accurate surface conver-
gence. After obtaining a relatively accurate scene representation, we employ a Graph
Neural Network (GNN) [25] to further optimize Gaussian attributes. The Gaussians,
treated as vertices, are connected based on spatial neighbor relationship, and learnable
offsets are applied to update their attributes. To achieve more accurate depth estima-
tion, we utilize an explicit depth computation method within an epipolar geometry
framework through optical flow, following [26]. The estimated reliable depth maps are
used sequentially for regularization, further improving the rendering quality. During
training, we preserve joint training from [16] to optimize the Gaussian parameters and
camera poses at the same time. Novel view synthesis comparisons with other pose-free
methods are shown in Fig.1, which exhibit the superior performance of our method.
In summary, our method mainly includes the following contributions:
• we introduce a hybrid Gaussian representation with dual optimization of position
and shape based on matching priors,
• we design a graph-guided optimization module to refine Gaussian attributes for
detailed scene recovery,
• we apply flow-based depth regularization to improve the quality of rendered im-
ages during training,
• extensive experiments on LLFF and Tanks and Temples datasets show that our
method significantly improves scene reconstruction and novel view synthesis
from sparse-view pose-free images.
2. Related Work
2.1. 3D Reconstruction
As a fundamental challenge in computer vision, 3D reconstruction aims to recover
the 3D geometric structure of a scene from a set of input 2D images. Traditional
optimization-based methods, such as Structure from Motion (SfM) [12] and Multi-
View Stereo (MVS) [27, 28], rely on rigorous geometric principles to estimate ac-
curate camera parameters and dense point clouds. COLMAP, which integrates SfM
4

<!-- page 5 -->
and MVS, is one of the most popular and powerful open source tools.
However,
its high computational cost, sensitivity to errors in SfM-estimated camera poses and
sparse point clouds, and dependence on dense viewpoint coverage limit its applicabil-
ity in textureless regions and areas with low overlap.The advent of deep learning has
spurred the development of learning-based reconstruction methods to overcome some
limitations of traditional pipelines. As a pioneering work, MVSNet [29] constructs
a 3D cost volume and uses 3D convolutional neural networks to regress depth maps.
Follow-up works such as R-MVSNet [30] alleviate memory consumption through re-
current networks, and CVP-MVSNet [31] introduces a coarse-to-fine pyramid structure
to improve efficiency and accuracy. Nonetheless, these approaches typically assume
that camera poses are known and accurate, which is usually unreliable in real-world
scenarios. More recently, DUSt3R [17] notably proposes an end-to-end model for
feed-forward 3D reconstruction. Unlike traditional methods that require camera cali-
bration or viewpoint poses, DUSt3R leverages a Transformer architecture to directly
estimate 3D point clouds from unconstrained image pairs, followed by global align-
ment to produce a complete scene representation. Its sibling, MASt3R [32], augments
DUSt3R with a second network head to generate dense local features and introduces a
novel matching loss for constraint. To avoid the computational cost of iterative post-
optimization required by DUSt3R, VGGT [18] introduces a feed-forward network that
directly generates more continuous point clouds and accurate camera poses, capable
of processing more than two images simultaneously. VGGT achieves state-of-the-art
reconstruction performance compared to DUSt3R and MASt3R. Inspired by this, we
adopt VGGT for dense scene initialization.
2.2. Novel View Synthesis
Novel view synthesis (NVS) aims to render high-quality images from unseen view-
points. Neural Radiance Fields (NeRF) [1], which implicitly represent scenes using
multi-layer perceptions (MLPs) and synthesize images via volume rendering, has be-
come one of the most influential NVS methods. Despite achieving high-quality recon-
structions, NeRF suffers from high computational costs, leading to slow training and
inference. Subsequent research has improved NeRF in various aspects, e.g., quality
5

<!-- page 6 -->
[33, 34], efficiency [35, 36], and pose-free [13, 14, 37]. More recently, 3D Gaussian
Splatting (3DGS) [8] explicitly represents scenes using anisotropic 3D Gaussians [38],
enabling rapid reconstruction and real-time rendering while maintaining high visual
quality. 3DGS has been extensively explored in many aspects, such as [11, 20, 39, 40],
demonstrating superior performance compared with NeRF-based approaches. How-
ever, conventional NVS methods heavily rely on dense input views and accurate cam-
era poses derived from SfM, which is often impractical to attain, leading to significant
performance degradation under sparse view settings.
2.3. Sparse-View Novel View Synthesis
Sparse-view novel view synthesis addresses a practical but more challenging task
of synthesizing novel views from a limited set of input images, where overfitting oc-
curs due to insufficient supervision. To mitigate these issues, various strategies have
been proposed. Methods like [41, 42] apply regularization techniques to reduce float-
ing artifacts and enhance geometric consistency. [6, 43] incorporate depth priors from
SfM or monocular depth estimation to supervise scene geometry, thereby reducing
geometric errors and improving rendering image quality. Generalizable NeRF mod-
els like [3, 44] enable feed-forward inference, allowing fast adaptation to new scenes
and improving practicality. Despite these advancements, NeRF-based methods remain
computationally intensive due to volume rendering.In contrast to NeRF-based meth-
ods, several studies combined with 3DGS attempt to eliminate reconstruction incon-
sistency. To better capture spatial structure, [21, 22] introduce depth information into
the explicit 3DGS representation. However, inaccuracies in monocular depth estima-
tion often degrade reconstruction quality in sparse-view scenarios. [10, 45, 46] focus
on robust scene initialization and removal of incorrectly positioned Gaussians. Some
methods [46, 47, 48] introduce various regularization strategies to mitigate geometric
degradation and suppress rendering artifacts. Alternative scene representations, such
as [24, 49], are designed to produce accurate and detailed 3D scenes with enhanced ge-
ometry. To better preserve geometric structure completeness and details, we introduce
a hybrid Gaussian representation to learn scene structure, with flow-based depth regu-
larization and graph-guided optimization incorporated to refine reconstruction results
6

<!-- page 7 -->
in weakly textured regions.
Flow
Flow
Depth 
Estimation
Depth 
Estimation
Sparse Inputs
Vggt
Vggt
Sparse Inputs
Vggt
Matching Priors
Matching Priors
Ray1
Ray2
G1
G2
View1
View2
G3
G4
Hybrid Representation 
and Optimization
G1,G2
G3,G4
Ray1,Ray2
G1,G2
G3,G4
Ray1,Ray2
G1,G2
G3,G4
Ray1,Ray2
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
G1,G2
G3,G4
Ray1,Ray2
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
Ray1
Ray2
G1
G2
View1
View2
G3
G4
Hybrid Representation 
and Optimization
G1,G2
G3,G4
Ray1,Ray2
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
Ray1
Ray2
G1
G2
View1
View2
G3
G4
Hybrid Representation 
and Optimization
G1,G2
G3,G4
Ray1,Ray2
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
Rendered Depth 1
Rendered Depth 1
Flow-estimated Depth 1
Flow-estimated Depth 1
Rendered Depth 2
Rendered Depth 2
L1
L1
Lrg
Lrg
Rendered Depth 1
Flow-estimated Depth 1
Rendered Depth 2
L1
Lrg
L1+LSSIM
L1+LSSIM
Rendered Image
Rendered Image
Ground Truth
Ground Truth
L1+LSSIM
Rendered Image
Ground Truth
Projection
Ray
Projection
Ray
Joint 
Optimization
Joint 
Optimization
2
3
3
Flow 
Predictor
Flow 
Predictor
Operation Flow
Operation Flow
Gradient Flow
Gradient Flow
Operation Flow
Gradient Flow
1
Graph-guided 
Optimization
μ
s
r
c
zd
μ
+
o
=
zd
μ
+
o
=
‘μ ‘μ
’s ’s
‘r ‘r
’c ’c
d
z
μ
'
'
+
o
=
d
z
μ
'
'
+
o
=
z
Δz
Δ
s
Δ s
Δ
r
Δr
Δ
c
Δc
Δ
3D Gaussian
3D Gaussian
Graph Net
Graph Net
3D Gaussian
Graph Net
α
α
Δ α
Δ
‘
α
‘
α
Flow
Flow
Depth 
Estimation
Depth 
Estimation
Sparse Inputs
Vggt
Vggt
Sparse Inputs
Vggt
Matching Priors
Matching Priors
Ray1
Ray2
G1
G2
View1
View2
G3
G4
Hybrid Representation 
and Optimization
G1,G2
G3,G4
Ray1,Ray2
G1,G2
G3,G4
Ray1,Ray2
G1,G2
G3,G4
Ray1,Ray2
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
G1,G2
G3,G4
Ray1,Ray2
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
Ray1
Ray2
G1
G2
View1
View2
G3
G4
Hybrid Representation 
and Optimization
G1,G2
G3,G4
Ray1,Ray2
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
Ray1
Ray2
G1
G2
View1
View2
G3
G4
Hybrid Representation 
and Optimization
G1,G2
G3,G4
Ray1,Ray2
Matched Pixel Coordinate
Projected Pixel Coordinate
Binding Gaussian Pair
Non-Structure Gaussian
Matching Ray Pair
Projection
Projection Error
Rendered Depth 1
Rendered Depth 1
Flow-estimated Depth 1
Flow-estimated Depth 1
Rendered Depth 2
Rendered Depth 2
L1
L1
Lrg
Lrg
Rendered Depth 1
Flow-estimated Depth 1
Rendered Depth 2
L1
Lrg
L1+LSSIM
L1+LSSIM
Rendered Image
Rendered Image
Ground Truth
Ground Truth
L1+LSSIM
Rendered Image
Ground Truth
Projection
Ray
Projection
Ray
Joint 
Optimization
Joint 
Optimization
2
3
3
Flow 
Predictor
Flow 
Predictor
Operation Flow
Operation Flow
Gradient Flow
Gradient Flow
Operation Flow
Gradient Flow
1
Graph-guided 
Optimization
μ
s
r
c
zd
μ
+
o
=
zd
μ
+
o
=
‘μ ‘μ
’s ’s
‘r ‘r
’c ’c
d
z
μ
'
'
+
o
=
d
z
μ
'
'
+
o
=
z
Δz
Δ
s
Δ s
Δ
r
Δr
Δ
c
Δc
Δ
3D Gaussian
3D Gaussian
Graph Net
Graph Net
3D Gaussian
Graph Net
α
α
Δ α
Δ
‘
α
‘
α
Fig 2: Overall framework of Gesplat. Given a few input images, we first generate dense point clouds and
camera poses from VGGT, extract matching priors and predict optical flow. Subsequently, we randomly ini-
tialize hybrid gaussian representation, using ray-based Gaussians to optimize the gaussian position. Graph-
guided optimization and joint optimization are then applied to refine the Gaussians and camera poses. Finally,
we employ flow-estimated depth and matching depth as rendering geometry regularization.
3. Method
Given a limited set of RGB images with camera poses and point clouds generated
by VGGT, we introduce Gesplat for sparse-view 3D scene reconstruction. Our model
consists of four key components: 1) hybrid Gaussian representation, 2) graph-guided
optimization, 3) flow-based depth regularization, and 4) joint optimization. Next, we
first review the framework of 3DGS, and then provide detailed descriptions of the mod-
ules in our method. An overview of Gesplat is illustrated in Fig. 2.
3.1. Preliminary
3.1.1. Gaussian Splatting
3DGS[8] explicitly represents a scene through anisotropic 3D Gaussians, achieving
high-quality scene reconstruction and real-time rendering. Each Gaussian is parame-
terized by: a position vector µ ∈R3, a covariance matrix Σ ∈R3×3, spherical harmonic
(SH) coefficients, and an opacity α ∈[0, 1). The influence for a point x in 3D space is
7

<!-- page 8 -->
defined as:
G(x) = e−1
2 (x−µ)T Σ−1(x−µ).
(1)
To ensure effective computation, the covariance matrix Σ is decomposed as Σ = RS S TRT,
where R ∈R3×3 is a rotation matrix and S ∈R3×1 is a scaling matrix. To enable inde-
pendent optimization of R and S , 3DGS represents them with a rotation factor r ∈R4
and a scaling factor s ∈R3, ensuring r is normalized to a valid unit quaternion. With
a view transformation matrix W and the Jacobian of the affine approximation of the
projective transformation J, 3D Gaussians are projected to the 2D image plane for
rendering:
Σ′ = JWΣWT JT.
(2)
Point-based rendering blends N ordered Gaussians overlapping a pixel and computes
the color C of the pixel via alpha blending:
C =
X
i∈N
ciα′
i
i−1
Y
j=1
(1 −α′
j),
(3)
where ci is the color of the i-th Gaussian, and α′
i is the multiplication of opacity αi and
Σ′. Depth D is computed similarly:
D =
X
i∈N
diα′
i
i−1
Y
j=1
(1 −α′
j),
(4)
where di is the depth of the i-th Gaussian. After rendering, the photometric loss be-
tween the rendered image and the ground truth image is computed to optimized the
model. All learnable parameters are optimized via stochastic gradient descent using a
combination of L1 and D-SSIM [50] loss:
Lphoto = (1 −λ)L1 + λLD−S S IM.
(5)
3.1.2. Dense Initialization
Although 3DGS is efficient in real-time rendering, its initialization relies on a
sparse point cloud and accurate camera parameters from Structure-from-Motion (SFM)
[12]. This usually leads to inadequate information in sparse-view scenarios, which
causes overfitting on training views and overly smooth textures. Thanks to advances
8

<!-- page 9 -->
in deep learning, recent learning-based 3D reconstruction methods bypass the need for
explicit keypoint extraction and matching of traditional methods in an end-to-end way.
Among these, DUSt3R [17] estimates 3D point clouds directly from 2D image pairs
without precise camera calibration, and employs a global alignment strategy for multi-
view 3D reconstruction. However, its iterative post-optimization steps result in high
computation cost. To address this, we utilize VGGT [18], a feedforward neural net-
work that produces high-fidelity dense point clouds and accurately estimates camera
poses even from sparse and low-overlap datasets. This ensures geometric continuity in
the reconstructed scene and provides a more robust initialization for 3DGS.
3.2. Hybrid Gaussian Representation and Optimization
In the sparse-view scenario, 3DGS model struggles to learn the complete scene
structure with inconsistent information from training views, leading to poor reconstruc-
tion quality in novel views. To better learn the scene geometry, we extract matching
priors from a pre-trained matching model [51] and introduce a hybrid Gaussian rep-
resentation. Ray-based Gaussians are bound to matching ray pairs across views and
optimized along the ray direction, while ordinary non-structured Gaussians represent
background regions visible in individual views.
3.2.1. Matching Priors
Matching priors refer to ray correspondence and ray position. 1) Ray Correspon-
dence: A pair of matching rays {ri, rj} from view i and view j corresponds to the same
3D point. Given image Ii and Ij, matching ray pair {ri, rj}, pixel points {pi, pj}, intrin-
sic parameters {Ki, Kj}, and extrinsic parameters {[Ri, ti], [Rj, tj]}, we assume that each
ray intersects the surface at point Xi and Xj, satisfying Xi = Xj. Thus we can get the
projection from view i to view j as:
pj = pi→j(Xi) = π(KjRT
j (Xi −t j)),
(6)
where π([x, y, z]T) = [x/z, y/z]T. The projection from view j to view i is defined simi-
larly. 2) Ray Position: We emphasize that ray position refers to the positions visible in
two or more views. In the sparse-view scenario, non-overlapping regions often cause
overfitting. Therefore, regions visible across multiple views are very crucial.
9

<!-- page 10 -->
3.2.2. Ray-based Gaussian
Given an image pair Ii and I j with N matching ray pairs {rk
i , rk
j}N
k=1, there exist N
corresponding pairs of ray-based Gaussians {Gk
i ,Gk
j}N
k=1. Each primitive has similar
attributes with ordinary Gaussian but differs in position representation. The new repre-
sentation of position µ′ is as follows:
µ′ = o + zd,
(7)
where o is the camera center, d is the ray direction, and z is a randomly initialized
learnable distance factor.
3.2.3. Position Optimization
Leveraging the binding strategy between ray-based Gaussians and their matching
rays, we optimize Gaussian positions using the projection relation. For a matching
ray pair {ri, rj} with corresponding Gaussians {Gi,G j} at positions µ′
i = oi + zidi and
µ′
j = oj + z jdj, the projection error is defined as:

Li→j
gp =∥pj −pi→j(µ′
i) ∥
L j→i
gp =∥pi −p j→i(µ′
j) ∥.
(8)
The total Gaussian position loss Lgp is the average of all projection errors.
3.2.4. Rendering Geometry Optimization
In addition to position, other inaccurate attributes can also affect rendering quality.
To address this, we optimize the rendering geometry. Through Eq. (4), we obtain the
rendering depth maps Di and D j. Then we use the depth values {Di(pi), Dj(pj)} in
matching pixels to yield the corresponding 3D points:
Pi = Ri(Di(pi)K−1
i epi)) + ti,
(9)
where epi is the 2D homogeneous coordinate of pixel pi (P j is similar). With the pro-
jection relation, the rendering depth projection error is given by:

Li→j
rg
=∥pj −pi→j(Pi) ∥
L j→i
rg
=∥pi −p j→i(P j) ∥.
(10)
The total rendering geometry loss Lrg is the average of all these errors.
10

<!-- page 11 -->
3.3. Graph-guided Optimization
Considering the view inconsistency under a sparse-view setting, we introduce a
Graph Neural Network (GNN) to better learn the spatial geometry structure, improv-
ing the scene reconstruction quality. We utilize a graph G = (V, E) to capture complex
geometric features for Gaussians, where vertices V = {vi}M
i=1 = {zi, si, ri, ci, αi}M
i=1 rep-
resent the Gaussian attributes, and edges E describe the spatial adjacency based on K
nearest neighbors (KNN). The K nearest neighbors of point µi are calculated as:
KNN(µi) = {µ j | j ∈argmin j d(µi, µj), j = 1, · · · , K},
(11)
where the Euclidean distance d(µi, µj) =∥µi −µ j ∥2. And the edges are defined as:
E = {(i, j) | µj ∈KNN(µi) ∧d(µi, µj) < r},
(12)
where r is a radius threshold. Upon generating the graph network, we formulate its
output as follows:
δ = G(V, E) = {∆zi, ∆si, ∆ri, ∆ci, ∆αi}M
i=1.
(13)
Then we introduce the output as the refinement of Gaussian attributes:
z′
i = zi + λz · ∆zi,
s′
i = si + λs · ∆si,
r′
i = si + λr · ∆ri,
c′
i = ci + λc · ∆ci,
α′
i = αi + λα · ∆αi,
(14)
and the refined Gaussian position is given by:
µ′′
i = oi + z′
idi.
(15)
We set the regularization parameters λz, λs, λr, λc, and λα to 0.1, 0.1, 0.05, 0.01, and 1,
respectively.
11

<!-- page 12 -->
3.4. Flow-based Depth Regularization
In addition to attaining high reconstruction quality using matching priors and ren-
dering depth, we also introduce the depth estimated through optical flow to calibrate the
geometric structure. Since monocular depth estimation often fails to deliver accurate
depth information, we incorporate epipolar priors as a constraint to explicitly compute
the depth maps.
3.4.1. Epipolar Line
For a point pi in view i, the corresponding epipolar line lpi in view j is defined as:
ax + by + c = 0,
(a, b, c)⊤= Fi→j(xi, yi, 1)⊤,
(16)
where Fi→j is the fundamental matrix from view i to view j, and (xi, yi) is the coordinate
of the point pi.
3.4.2. Flow-estimated Depth
Given a pre-trained optical flow predictor f(•) [52], we compute the optical flow
between views i and j as:
Mi→j
flow = f(Ii, Ij),
(17)
where Ii and Ij is a matching image pair. Given a point pi in image Ii, the predicted
corresponding point bpj in image I j is estimated as:
bp j = pi + Mi→j
flow(pi).
(18)
Due to estimation errors, bp j may not accurately lies on the epipolar line lpi. Therefore,
we select the perpendicular foot pj as the approximate estimation of bpj :
pj =
 b2bxj −abby j −ac
a2 + b2
, a2byj −abbx j −bc
a2 + b2
!
,
(19)
where (bxj,by j) is the coordinate of point bpj, and {a, b, c} are the parameters of the epipo-
lar line lpi. Then the depth value of the point pi is calculated as:
Di→j(pi, ¯pj) =
|H × −(RiR−1
j T j −Ti)|
|(K−1
i (xi, yi, 1)T) × H| ,
where H =(RjR−1
i )−1K−1
j (¯xj, ¯yj, 1)T.
(20)
12

<!-- page 13 -->
Here, × denotes the cross product, and {Ki, Kj} and {[Ri, ti], [Rj, tj]} are the camera
intrinsics and extrinsics for view i and view j, respectively.
3.4.3. Cross-view Depth Blending
Given image set {Ii | i = 1, 2, · · · , N}, for a point pi in Ii, we compute the pre-
dicted matching points {pj | j , i} and the corresponding depths in other N −1 views.
Although p j lies on the epipolar line, the distance error that p j deviates from the true
correspondence bp j still persists. To handle the variance of depth estimation across
views, we introduce a cross-view depth blending strategy. Depth is represented as the
distance information from a pixel point to the camera. Given Oi as the camera center
of view i, we define a reference distance disre f as the distance from the 3D point P j to
Oi. And the projection distance dispro in view j is the distance from the projected point
pj to the epipole oij. The distance between p j and the true matching point pj is de-
noted as ∆dispro, corresponding to a 3D distance ∆disre f . A smaller ∆disre f indicates
a more accurate depth estimation. Through the fundamental theorem of calculus, the
transformation between ∆disref and ∆dispro is computed as:
∆disref =
Z dispro+∆dispro
dispro
dis′
re f (dispro) ddispro
≈dis′
re f (dispro)∆dispro,
(21)
where dis′
ref (dispro) is the gradient of disre f (dispro), indicating the sensitivity of depth
to the changes of dispro. A smaller dis′
re f (dispro) implies higher confidence in the esti-
mated depth. Using geometric relations, the formulation of dis′
re f (dispro) is expressed
as:
dis′
ref (dispro) = t sin β sin2(α + θ)
m sin θ sin2(α + β)
,
(22)
where t = OiOj, m = Ojoij, α = ∠OiOjPj, θ = ∠Ojoi jp j, β = ∠OjOiP j.Since we seek
the view j with the smallest ∆disre f from other N −1 views, assuming that ∆dispro
is constant across views for the same optical flow predictor, we only need to compare
dis′
ref (dispro). Finally, the blended depth Di(pi) for point pi is given by:
Di(pi) = Di→k(pi),
k = arg min
j dis′
re f (disi→j,pi
pro
).
(23)
13

<!-- page 14 -->
3.4.4. Depth Regularization
Given the rendered depth map D and the flow-estimated depth map bD, the depth
loss is computed using loss function L1:
Ldepth = L1(bD, D).
(24)
3.5. Joint Optimization
To jointly optimize noisy point clouds and inaccurate camera poses, we introduce
joint optimization, which updates both Gaussians and camera parameters via image
loss. Given a Gaussian set G and camera parameters T, we minimize the difference
between the ground truth image and the rendered image using gradient descent:
G∗, T ∗= arg min
G,T
X
v∈N
HW
X
i=1
∥e
Ci
v(G, T) −Ci
v ∥.
(25)
During testing, since predicted camera poses for novel views may still be noisy, we
use the trained Gaussian model to refine test camera poses via Eq.25 for 500 iterations
per-image for more accurate evaluation.
3.6. Training Objectives
3.6.1. Loss Function
The total training loss is formulated as :
L = Lphoto + λgpLgp + λrgLrg + λdepthLdepth,
(26)
where Lphoto is the photometric loss, Lgp is the Gaussian position loss, Lrg is the ren-
dering geometry loss, and Ldepth is the depth loss.
3.6.2. Training Details
We set λgp = 1.0, λdepth = 0, λrg = 0 for the first 2000 iterations, and then adjust
to λdepth = 0.1, λrg = 0.3 to avoid sub-optimization during early stage of training .
Our model is built upon InstantSplat [16] and trained for 5000 iterations. The learning
rate for the learnable depth z starts at 0.1 and decays to 1.6 × 10−6. To mitigate initial
errors and computational cost from the graph network, we only use it for Gaussian
optimization in the final 200 iterations.
14

<!-- page 15 -->
4. Experiments
In this section, we conduct experiments to demonstrate the effectiveness of our
approach. We first describe the datasets, evaluation metrics, and baselines used for
comparison. Next, we compare our method with other pose-free methods on LLFF [53]
and Tanks and Temples [54] datasets, followed by an analysis of the results. Finally,
we perform a series of ablation studies to validate the efficacy of key modules in our
method.
4.1. Experimental Settings
4.1.1. Datasets
Under sparse-view settings, we evaluate Gesplat using LLFF and Tanks and Tem-
ples datasets. LLFF provides a collection of real-world scenes with forward-facing
camera trajectories, while Tanks and Temples is a large-scale dataset containing both
indoor and outdoor scenes with complex environments. Both datasets consist of eight
scenes each. Following InstantSplat [16], we randomly sample 24 images from each
scene, with 12 images held out as test views (including the first and last images). For
the remaining 12 images, we select 3 to 9 images for training under sparse-view con-
ditions.
4.1.2. Baselines & Metrics
We compare Gesplat with other pose-free approaches, including Nope-NeRF [14],
CF-3DGS [15], and InstantSplat [16]. Among them, Nope-NeRF and CF-3DGS lever-
age monocular depth estimation and ground-truth camera intrinsics, while InstantSplat
relies on DUSt3R [17] to generate dense point clouds. All of the baselines are trained
using official code with default settings. We evaluate the performance on novel view
synthesis task using common metrics: Peak Signal-to-Noise Ratio (PSNR), Structural
Similarity Index (SSIM) [50], and Learned Perceptual Image Patch Similarity (LPIPS)
[55].
4.1.3. Implementation Details
Our implementation is built on the PyTorch framework, and all experiments are
conducted on one NVIDIA RTX 3090 GPU. For Nope-NeRF and CF-3DGS, we selete
15

<!-- page 16 -->
input images at the same resolution as original datasets. Using DUSt3R, InstantSplat
predicts multi-view stereo depth maps at a resolution of 512. Our method trains with a
resolution of 518 × 392 using VGGT [18].
Table 1: Quantitative comparison on LLFF (3,6,9 views). The best, second best, and third best entries are
marked in red, orange, and yellow, respectively.
Method
(3 views)
(6 views)
(9 views)
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
NoPe-NeRF[14]
12.029
0.326
0.696
12.084
0.332
0.692
11.883
0.330
0.691
CF-3DGS[15]
7.891
0.193
0.637
14.380
0.398
0.488
14.849
0.411
0.496
InstantSplat[16]
16.726
0.465
0.355
19.315
0.582
0.280
20.616
0.622
0.255
Ours
18.869
0.570
0.355
21.994
0.692
0.231
23.299
0.738
0.193
CF-3DGS
NoPe-NeRF
InstantSplat
Ours
GT
CF-3DGS
NoPe-NeRF
InstantSplat
Ours
GT
Fig 3:
Qualitative comparison between Gesplat and various pose-free methods on LLFF datasets(6
views).The reconstruction of our method is more accurate and exhibits finer details compared with other
competitors.
16

<!-- page 17 -->
4.2. Experimental Results and Analysis
4.2.1. Results on LLFF
Quantitative comparisons are summarized in Tab. 1. Gesplat achieves the best
performance across PSNR, SSIM, and LPIPS compared to other pose-free methods.
Nope-NeRF utilizes Multilayer Perceptions (MLPs) to prioritize learning the global
scene structure, which effectively suppresses floating artifacts but may produce images
suffering from over-blur and require a long training time. CF-3DGS builds upon 3DGS
with local and global optimization, enabling more accurate geometric reconstruction
and real-time rendering, yet its complex optimization process may lead to loss of fine
details. As a feed-forward model, InstantSplat uses DUSt3R to estimate dense point
clouds and camera poses, mitigating the information scarcity of 3DGS in sparse views
and improving geometric consistency. However, it may occasionally generate inexis-
tent details, reducing the reconstruction quality. By leveraging VGGT for dense 3D
geometry recovery, our method uses a hybrid Gaussian representation to learn scene
geometry and refines Gaussian attributes and camera parameters via a graph network
and joint optimization. Furthermore, we incorporate depth cues from optical flow es-
timation to improve rendering quality. Qualitative results under 6 training views are
shown in Fig. 3, where Gesplat not only reconstructs complete geometry but also re-
covers more accurate and high-frequency details.
4.2.2. Results on Tanks and Temples
To evaluate the performance of Gesplat on large-scale complex scenes, we con-
duct further comparisons on Tanks and Temples dataset. With a training setting of 6
views, Gesplat achieves the best results across all metrics as shown in Tab. 2, demon-
strating our robustness on challenging large-scale environments. We also visualize the
qualitative comparisons in Fig. 4 to demonstrate the effectiveness of Gesplat.
4.3. Ablation Study
We conduct a series of ablation studies to validate the effectiveness of each mod-
ule in Gesplat and the rationality of parameter selection. Under a 3 views setting, we
17

<!-- page 18 -->
Table 2: Quantitative comparison on Tank and Temples. All models are trained with six input views.The
best result is highlighted in bold.
scenes
Ours
NoPe-NeRF[14]
CF-3DGS[15]
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Ballroom
20.641
0.662
0.231
9.873
0.216
0.715
11.897
0.260
0.496
Barn
22.409
0.739
0.249
12.885
0.507
0.726
14.235
0.491
0.493
Church
20.379
0.652
0.293
14.301
0.355
0.680
13.629
0.350
0.576
Family
22.200
0.735
0.218
11.961
0.429
0.775
14.617
0.514
0.497
Francis
23.487
0.695
0.324
15.125
0.420
0.719
16.961
0.549
0.476
Horse
22.618
0.768
0.208
10.629
0.397
0.705
16.797
0.617
0.383
Ignatius
22.563
0.669
0.264
12.892
0.283
0.735
13.132
0.324
0.553
Museum
20.636
0.588
0.323
11.872
0.285
0.764
15.798
0.495
0.439
mean
21.867
0.689
0.264
12.442
0.362
0.727
14.633
0.450
0.489
Barn
Family
Francis
Horse
Ignatius
3_views
NoPe-NeRF
CF-3DGS
Ours
Barn
Family
Francis
Horse
Ignatius
3_views
NoPe-NeRF
CF-3DGS
Ours
6_views
NoPe-NeRF
CF-3DGS
Ours
6_views
NoPe-NeRF
CF-3DGS
Ours
9_views
NoPe-NeRF
CF-3DGS
Ours
9_views
NoPe-NeRF
CF-3DGS
Ours
Barn
Family
Francis
Horse
Ignatius
3_views
NoPe-NeRF
CF-3DGS
Ours
Barn
Family
Francis
Horse
Ignatius
3_views
NoPe-NeRF
CF-3DGS
Ours
6_views
NoPe-NeRF
CF-3DGS
Ours
6_views
NoPe-NeRF
CF-3DGS
Ours
9_views
NoPe-NeRF
CF-3DGS
Ours
9_views
NoPe-NeRF
CF-3DGS
Ours
Fig 4: Qualitative comparison on TNT(3,6,9 views).The reconstruction of our method is more accurate and
exhibits finer details.
18

<!-- page 19 -->
select the fortress scene from the LLFF dataset to establish our design choices. Quan-
titative and qualitative results of the ablation study are presented in Tab. 3 and Fig. 5,
respectively.
Table 3: Ablations on the key modules. We evaluate them on the fortress scene of LLFF.
Method
PSNR↑
SSIM↑
LPIPS↓
Ours
21.14
0.60
0.30
w/o Depth Reg.
20.43
0.60
0.30
w/o Graph Opt.
20.58
0.56
0.31
w/o Hybrid Rep.
18.67
0.48
0.38
4.3.1. Effect of hybrid representation (Hybrid Rep.)
As shown in Tab. 3, the removal of the hybrid representation leads to a drop in
PSNR from 21.14 dB to 18.67 dB. As illustrated in Fig. 5, the visual quality of the
rendered image also degrades significantly, with noticeable holes and blurred texture
of the table. These results demonstrate that, compared to the original Gaussian rep-
resentation, introducing hybrid Gaussian representation based matching priors better
preserves geometric information, improves structural completeness, and reduces the
risk of overfitting in sparse view scenes.
4.3.2. Effect of graph-guided optimization (Graph Opt.)
Although graph-guided optimization is only applied in the last 200 iterations, its
removal causes a quantitative decrease of 0.56 dB in PSNR, as shown in Tab. 3. From
the visual comparison in Fig. 5, we observe that the edges of the floor become blurred
and the details of the fortress are over-smoothed without graph-guided optimization.
This indicates that graph-guided optimization effectively refines Gaussian attributes
and enhances scene details.
4.3.3. Effect of flow-based depth regularization (Depth Reg.)
Without flow-based depth regularization, it results in a slight decrease in PSNR, as
reported in Tab. 3. Compared to our full model, the absence of this module also blurs
the edges of the fortress and fails to accurately reconstruct the gaps in the flooring, as
shown in Fig. 5. These results underscore the importance of this strategy.
19

<!-- page 20 -->
Ours
GT
GT
w/o Depth Reg.
w/o Graph Opt.
w/o Hybrid Rep.
N.A.
N.A.
Ours
GT
GT
w/o Depth Reg.
w/o Graph Opt.
w/o Hybrid Rep.
N.A.
N.A.
Fig 5: Ablation study on the key modules of our model.
4.3.4. Effect of model parameter settings
We further investigate the effectiveness of parameter selection using the trex scene
from the LLFF dataset. The parameters under study include the weight of the depth
loss λdepth and the regularization parameters for Gaussian attribute updates in the graph
network: λz, λs, λr, λc, and λα. The results are summarized in Tab. 4.The value
of λdepth influences the geometric continuity of the scene. We find that λdepth = 0.1
yields the best performance. Increasing λdepth leads to degradation across all metrics,
indicating that excessive depth regularization over-smooths texture details and reduces
reconstruction accuracy. Since we expect the graph network to refine scene details
through Gaussian attribute updating, we place greater emphasis on SSIM and LPIPS
during evaluation. Based on the results in Tab. 4, we set λz = 0.1, λs = 0.1, λr = 0.05,
λc = 0.01, and λα = 1.0.
5. Conclusion
In this paper, we introduce Gesplat to address the challenge of sparse-view novel
view synthesis from unposed images. Our framework first employs VGGT to gen-
erate initial dense point clouds and camera pose estimations. To enhance structural
20

<!-- page 21 -->
Table 4: Ablations on model parameter settings. We evaluate them on the trex scene of LLFF.
Parameter setting
PSNR↑
SSIM↑
LPIPS↓
λdepth
0.1
19.532
0.649
0.318
0.2
19.102
0.623
0.360
0.3
17.173
0.570
0.392
λz
0.05
19.558
0.647
0.320
0.1
19.532
0.649
0.318
0.2
19.494
0.649
0.319
λs
0.01
19.505
0.647
0.321
0.1
19.532
0.649
0.318
1.0
19.576
0.647
0.318
λr
0.03
19.586
0.648
0.322
0.05
19.532
0.649
0.318
0.07
19.362
0.646
0.322
λc
0.01
19.532
0.649
0.318
0.1
19.451
0.643
0.320
1.0
18.679
0.641
0.416
λα
1
19.532
0.649
0.318
2
19.596
0.647
0.320
3
19.498
0.647
0.322
consistency, we introduce a hybrid Gaussian representation optimized with matching
priors, enforcing geometric coherence through multi-view structural and rendering con-
straints. We further incorporate flow-based depth regularization to improve rendering
accuracy and a graph-guided optimization module to refine Gaussian attributes for de-
tailed scene recovery. Finally, we jointly optimize camera parameters and Gaussian
representations. Extensive comparisons with other pose-free methods show that our
approach achieves state-of-the-art performance on both forward-facing and large-scale
scenes. A limitation remains when handling largely non-overlapping views, where
matching priors become unreliable. Future work will explore more robust geometric
constraints for extreme sparsity scenarios.
References
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng,
Nerf: Representing scenes as neural radiance fields for view synthesis, Commu-
nications of the ACM 65 (1) (2021) 99–106.
21

<!-- page 22 -->
[2] M. M. Johari, Y. Lepoittevin, F. Fleuret, Geonerf: Generalizing nerf with geome-
try priors, in: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 18365–18375.
[3] A. Yu, V. Ye, M. Tancik, A. Kanazawa, pixelnerf: Neural radiance fields from one
or few images, in: Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, 2021, pp. 4578–4587.
[4] J. Yang, M. Pavone, Y. Wang, Freenerf: Improving few-shot neural rendering
with free frequency regularization, in: Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2023, pp. 8254–8263.
[5] S. Seo, D. Han, Y. Chang, N. Kwak, Mixnerf: Modeling a ray with mixture
density for novel view synthesis from sparse inputs, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp.
20659–20668.
[6] G. Wang, Z. Chen, C. C. Loy, Z. Liu, Sparsenerf: Distilling depth ranking for
few-shot novel view synthesis, in: Proceedings of the IEEE/CVF international
conference on computer vision, 2023, pp. 9065–9076.
[7] Y. Xie, H. Xiao, W. Kang, Tri 2 plane: Advancing neural implicit surface recon-
struction for indoor scenes, IEEE Transactions on Multimedia (2025).
[8] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, 3d gaussian splatting for real-
time radiance field rendering., ACM Trans. Graph. 42 (4) (2023) 139–1.
[9] D. Charatan, S. L. Li, A. Tagliasacchi, V. Sitzmann, pixelsplat: 3d gaussian splats
from image pairs for scalable generalizable 3d reconstruction, in: Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp.
19457–19467.
[10] H. Xiong, S. Muttukuru, R. Upadhyay, P. Chari, A. Kadambi, Sparsegs:
Real-time 360◦sparse view synthesis using gaussian splatting, arXiv preprint
arXiv:2312.00206 (2023).
22

<!-- page 23 -->
[11] K. Cheng, X. Long, K. Yang, Y. Yao, W. Yin, Y. Ma, W. Wang, X. Chen, Gaus-
sianpro: 3d gaussian splatting with progressive propagation, in: Forty-first Inter-
national Conference on Machine Learning, 2024.
[12] J. L. Schonberger, J.-M. Frahm, Structure-from-motion revisited, in: Proceedings
of the IEEE conference on computer vision and pattern recognition, 2016, pp.
4104–4113.
[13] Z. Wang, S. Wu, W. Xie, M. Chen, V. A. Prisacariu, NeRF−−: Neural radi-
ance fields without known camera parameters, arXiv preprint arXiv:2102.07064
(2021).
[14] W. Bian, Z. Wang, K. Li, J.-W. Bian, V. A. Prisacariu, Nope-nerf: Optimising
neural radiance field with no pose prior, in: Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, 2023, pp. 4160–4169.
[15] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, X. Wang, Colmap-free 3d
gaussian splatting, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 20796–20805.
[16] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic,
M. Pavone, G. Pavlakos, et al., Instantsplat: Sparse-view gaussian splatting in
seconds, arXiv preprint arXiv:2403.20309 (2024).
[17] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, J. Revaud, Dust3r: Geometric 3d
vision made easy, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 20697–20709.
[18] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny, Vggt: Vi-
sual geometry grounded transformer, in: Proceedings of the Computer Vision and
Pattern Recognition Conference, 2025, pp. 5294–5306.
[19] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao,
G. Zhang, Pgsr: Planar-based gaussian splatting for efficient and high-fidelity sur-
face reconstruction, IEEE Transactions on Visualization and Computer Graphics
(2024).
23

<!-- page 24 -->
[20] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, B. Dai, Scaffold-gs: Structured
3d gaussians for view-adaptive rendering, in: Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, 2024, pp. 20654–20664.
[21] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, L. Gu, Dngaussian: Opti-
mizing sparse-view 3d gaussian radiance fields with global-local depth normal-
ization, in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2024, pp. 20775–20785.
[22] Z. Zhu, Z. Fan, Y. Jiang, Z. Wang, Fsgs: Real-time few-shot view synthesis using
gaussian splatting, in: European conference on computer vision, Springer, 2024,
pp. 145–163.
[23] R. Ranftl, A. Bochkovskiy, V. Koltun, Vision transformers for dense prediction,
in: Proceedings of the IEEE/CVF international conference on computer vision,
2021, pp. 12179–12188.
[24] R. Peng, W. Xu, L. Tang, J. Jiao, R. Wang, et al., Structure consistent gaus-
sian splatting with matching prior for few-shot novel view synthesis, Advances in
Neural Information Processing Systems 37 (2024) 97328–97352.
[25] H. Xiao, H. Xu, Y. Li, W. Kang, Multi-dimensional graph interactional network
for progressive point cloud completion, IEEE Transactions on Instrumentation
and Measurement 72 (2022) 1–12.
[26] Y. Zheng, Z. Jiang, S. He, Y. Sun, J. Dong, H. Zhang, Y. Du, Nexusgs: Sparse
view synthesis with epipolar depth priors in 3d gaussian splatting, in: Proceedings
of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26800–
26809.
[27] J. L. Schönberger, E. Zheng, J.-M. Frahm, M. Pollefeys, Pixelwise view selection
for unstructured multi-view stereo, in: European conference on computer vision,
Springer, 2016, pp. 501–518.
24

<!-- page 25 -->
[28] K. Chen, Z. Yuan, H. Xiao, T. Mao, Z. Wang, Learning multi-view stereo with
geometry-aware prior, IEEE Transactions on Circuits and Systems for Video
Technology (2025).
[29] Y. Yao, Z. Luo, S. Li, T. Fang, L. Quan, Mvsnet: Depth inference for unstruc-
tured multi-view stereo, in: Proceedings of the European conference on computer
vision (ECCV), 2018, pp. 767–783.
[30] Y. Yao, Z. Luo, S. Li, T. Shen, T. Fang, L. Quan, Recurrent mvsnet for high-
resolution multi-view stereo depth inference, in: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2019, pp. 5525–5534.
[31] J. Yang, W. Mao, J. M. Alvarez, M. Liu, Cost volume pyramid based depth in-
ference for multi-view stereo, in: Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2020, pp. 4877–4886.
[32] V. Leroy, Y. Cabon, J. Revaud, Grounding image matching in 3d with mast3r, in:
European Conference on Computer Vision, Springer, 2024, pp. 71–91.
[33] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman, Mip-nerf 360:
Unbounded anti-aliased neural radiance fields, in: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2022, pp. 5470–5479.
[34] D. Verbin, P. Hedman, B. Mildenhall, T. Zickler, J. T. Barron, P. P. Srini-
vasan, Ref-nerf: Structured view-dependent appearance for neural radiance fields,
in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), IEEE, 2022, pp. 5481–5490.
[35] S. J. Garbin, M. Kowalski, M. Johnson, J. Shotton, J. Valentin, Fastnerf: High-
fidelity neural rendering at 200fps, in: Proceedings of the IEEE/CVF international
conference on computer vision, 2021, pp. 14346–14355.
[36] C. Sun, M. Sun, H.-T. Chen, Direct voxel grid optimization: Super-fast conver-
gence for radiance fields reconstruction, in: Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, 2022, pp. 5459–5469.
25

<!-- page 26 -->
[37] Z. Cheng, C. Esteves, V. Jampani, A. Kar, S. Maji, A. Makadia, Lu-nerf: Scene
and pose estimation by synchronizing local unposed nerfs, in: Proceedings of
the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18312–
18321.
[38] M. Zwicker, H. Pfister, J. Van Baar, M. Gross, Ewa volume splatting, in: Pro-
ceedings Visualization, 2001. VIS’01., IEEE, 2001, pp. 29–538.
[39] Y. Chen, Q. Wu, W. Lin, M. Harandi, J. Cai, Hac: Hash-grid assisted context
for 3d gaussian splatting compression, in: European Conference on Computer
Vision, Springer, 2024, pp. 422–438.
[40] B. Lee, H. Lee, X. Sun, U. Ali, E. Park, Deblurring 3d gaussian splatting, in:
European Conference on Computer Vision, Springer, 2024, pp. 127–143.
[41] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. Sajjadi, A. Geiger, N. Radwan,
Regnerf: Regularizing neural radiance fields for view synthesis from sparse in-
puts, in: Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 5480–5490.
[42] D. Lee, D. Kim, J. Lee, M. Lee, S. Lee, S. Lee, Sparse-derf: Deblurred neu-
ral radiance fields from sparse view, IEEE Transactions on Pattern Analysis and
Machine Intelligence (2025).
[43] K. Deng, A. Liu, J.-Y. Zhu, D. Ramanan, Depth-supervised nerf: Fewer views and
faster training for free, in: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2022, pp. 12882–12891.
[44] Q. Wang, Z. Wang, K. Genova, P. P. Srinivasan, H. Zhou, J. T. Barron, R. Martin-
Brualla, N. Snavely, T. Funkhouser, Ibrnet: Learning multi-view image-based
rendering, in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2021, pp. 4690–4699.
[45] Z. Bao, G. Liao, K. Zhou, K. Liu, Q. Li, G. Qiu, Loopsparsegs: Loop based
sparse-view friendly gaussian splatting, IEEE Transactions on Image Processing
(2025).
26

<!-- page 27 -->
[46] J. Zhang, J. Li, X. Yu, L. Huang, L. Gu, J. Zheng, X. Bai, Cor-gs: sparse-view 3d
gaussian splatting via co-regularization, in: European Conference on Computer
Vision, Springer, 2024, pp. 335–352.
[47] Z. Liu, J. Su, G. Cai, Y. Chen, B. Zeng, Z. Wang, Georgs: Geometric regulariza-
tion for real-time novel view synthesis from sparse inputs, IEEE Transactions on
Circuits and Systems for Video Technology (2024).
[48] H. Park, G. Ryu, W. Kim, Dropgaussian: Structural regularization for sparse-view
gaussian splatting, in: Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, 2025, pp. 21600–21609.
[49] Y. Wan, M. Shao, Y. Cheng, W. Zuo, S2gaussian: Sparse-view super-resolution
3d gaussian splatting, in: Proceedings of the Computer Vision and Pattern Recog-
nition Conference, 2025, pp. 711–721.
[50] Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, Image quality assessment:
from error visibility to structural similarity, IEEE transactions on image process-
ing 13 (4) (2004) 600–612.
[51] X. Shen, Z. Cai, W. Yin, M. Müller, Z. Li, K. Wang, X. Chen, C. Wang,
Gim: Learning generalizable image matcher from internet videos, arXiv preprint
arXiv:2402.11095 (2024).
[52] X. Shi, Z. Huang, D. Li, M. Zhang, K. C. Cheung, S. See, H. Qin, J. Dai, H. Li,
Flowformer++: Masked cost volume autoencoding for pretraining optical flow
estimation, in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2023, pp. 1599–1610.
[53] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ramamoorthi,
R. Ng, A. Kar, Local light field fusion: Practical view synthesis with prescriptive
sampling guidelines, ACM Transactions on Graphics (ToG) 38 (4) (2019) 1–14.
[54] A. Knapitsch, J. Park, Q.-Y. Zhou, V. Koltun, Tanks and temples: Benchmarking
large-scale scene reconstruction, ACM Transactions on Graphics (ToG) 36 (4)
(2017) 1–13.
27

<!-- page 28 -->
[55] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, The unreasonable ef-
fectiveness of deep features as a perceptual metric, in: Proceedings of the IEEE
conference on computer vision and pattern recognition, 2018, pp. 586–595.
28
