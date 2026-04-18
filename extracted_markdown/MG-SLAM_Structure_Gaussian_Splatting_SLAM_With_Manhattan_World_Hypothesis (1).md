<!-- page 1 -->
17034
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING, VOL. 22, 2025
MG-SLAM: Structure Gaussian Splatting SLAM
With Manhattan World Hypothesis
Shuhong Liu , Tianchen Deng , Graduate Student Member, IEEE, Heng Zhou , Liuzhuozheng Li,
Hongyu Wang , Member, IEEE, Danwei Wang , Life Fellow, IEEE, and Mingrui Li , Student Member, IEEE
Abstract—Gaussian Splatting SLAMs have made signiﬁcant
advancements in improving the eﬃciency and ﬁdelity of real-
time reconstructions. However, these systems often encounter
incomplete reconstructions in complex indoor environments,
characterized by substantial holes due to unobserved geometry
caused by obstacles or limited view angles. To address this
challenge, we present Manhattan Gaussian SLAM, an RGB-
D system that leverages the Manhattan World hypothesis to
enhance geometric accuracy and completeness. By seamlessly
integrating fused line segments derived from structured scenes,
our method ensures robust tracking in textureless indoor areas.
Moreover, The extracted lines and planar surface assumption
allow strategic interpolation of new Gaussians in regions of
missing geometry, enabling eﬃcient scene completion. Exten-
sive experiments conducted on both synthetic and real-world
scenes demonstrate that these advancements enable our method
to achieve state-of-the-art performance, marking a substantial
improvement in the capabilities of Gaussian SLAM systems.
Note to Practitioners—This paper was motivated by the limi-
tations of Gaussian Splatting SLAM systems in complex indoor
environments, where textureless surfaces and obstructed views
often lead to substantial tracking errors and incomplete maps.
While existing systems excel in high-ﬁdelity reconstruction, they
struggle with frame-to-frame or point-feature tracking in large-
scale environments, particularly with signiﬁcant camera rotations
and obscured structures. In this paper, we enhance the neural
dense SLAM by integrating the Manhattan World hypothesis,
applying its parallel line and planar surface constraints for
more robust tracking and mapping. We incorporate line segment
features into both tracking and mapping to improve structural
accuracy. Moreover, we propose a post-optimization method
Received 29 December 2024; revised 22 March 2025; accepted 5 May
2025. Date of publication 9 June 2025; date of current version 1 July 2025.
This article was recommended for publication by Associate Editor Z. Liu
and Editor J. Yi upon evaluation of the reviewers’ comments. This work was
supported in part by the National Research Foundation, Singapore, under NRF
Medium Sized Centre scheme (CARTIN), ASTAR under National Robotics
Programme under Grant M22NBK0109; and in part by the National Research
Foundation, Singapore. (Shuhong Liu and Tianchen Deng contributed equally
to this work.) (Corresponding author: Mingrui Li.)
Shuhong Liu and Liuzhuozheng Li are with the Department of Information
Science and Technology and the Department of Complexity Science and
Engineering, The University of Tokyo, Tokyo 113-8654, Japan.
Tianchen Deng is with the Institute of Medical Robotics and the Department
of Automation, Shanghai Jiao Tong University, Shanghai 200240, China, and
also with the Key Laboratory of System Control and Information Processing,
Ministry of Education, Shanghai 200240, China.
Heng Zhou is with the Department of Mechanical Engineering, Columbia
University, New York, NY 10027 USA.
Hongyu Wang and Mingrui Li are with the Department of Computer
Science, Dalian University of Technology, Dalian 116024, China (e-mail:
mmclmr@mail.dlut.edu.cn).
Danwei Wang is with the School of Electrical and Electronic Engineering,
Nanyang Technological University, Singapore 639798.
Digital Object Identiﬁer 10.1109/TASE.2025.3575772
that interpolates new Gaussian primitives to eﬀectively ﬁll gaps
on planar surfaces. Extensive experiments on multiple datasets
demonstrate the superiority of our approach in large-scale indoor
environments, resulting in more accurate tracking and mapping.
Index Terms—SLAM, 3DGS, Manhattan World.
I. INTRODUCTION
S
IMULTANEOUS Localization and Mapping (SLAM) is
a fundamental problem in computer vision that aims to
map an environment while simultaneously tracking the cam-
era pose. Learning-based dense SLAM methods, particularly
neural radiance ﬁeld (NeRF) approaches [1], [2], [3], [4],
[5], [6], [7], [8], have demonstrated remarkable improvements
in capturing dense photometric information and providing
accurate global reconstruction over traditional systems based
on sparse point clouds [9], [10], [11], [12], [13], [14], [15].
However, NeRF methods still face drawbacks such as over-
smoothing, bounded scene representation, and computational
ineﬃciencies [16], [17], [18], [19]. Recently, Gaussian-based
SLAM [20], [21], [22], [23], [24], [25] has emerged as
a promising approach utilizing volumetric Gaussian primi-
tives [26]. Leveraging these explicit representations, Gaussian
SLAMs deliver high-ﬁdelity rendering and ﬁne-grained scene
reconstruction, overcoming the limitations of NeRF-based
methods.
Despite their strengths, Gaussian SLAM faces notable
challenges in indoor scenes, which are often characterized
by textureless surfaces and complex spatial layouts. These
environments hinder robust tracking due to a lack of suf-
ﬁcient texture details critical for camera pose optimization.
Moreover, the complex geometry of indoor scenes often leads
to substantial unobserved areas due to occlusions or limited
view coverage. These unseen regions pose a critical yet largely
unexplored challenge for Gaussian SLAM, as the Gaussian
representation can hardly interpolate the unobserved geometry
without multi-view optimization. Consequently, substantial
holes and gaps are left in the unseen areas of the map, an
issue that has been largely overlooked in previous Gaussian
SLAM studies.
To overcome these challenges, we leverage the renowned
Manhattan World (MW) hypothesis [28] as a foundational
strategy for reﬁning and completing scene geometries. This
assumption posits that the built environment predominantly
adheres to a grid-like structure, with surfaces and lines align-
ing with three orthogonal directions. These lines and planar
1558-3783 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artiﬁcial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 2 -->
LIU et al.: MG-SLAM: STRUCTURE GAUSSIAN SPLATTING SLAM WITH MANHATTAN WORLD HYPOTHESIS
17035
surfaces impose meaningful constraints on the tracking and
mapping processes in the Gaussian SLAM system.
Speciﬁcally, we encompass enhancements in tracking, map-
ping, and planar surface completion. In tracking, we utilize
fused line features derived from the structured scenes as robust
feature foundations in textureless areas, backprojecting and
reprojecting these line segments for pose optimization and
full bundle adjustment. In mapping, we apply a photometric
loss for the reprojected line features to reﬁne the map. This
approach ensures that the reconstructed scene adheres closely
to the true structure of the environment, thereby improving
both its geometry accuracy and rendering quality.
Furthermore, the MW hypothesis facilitates the identiﬁca-
tion and interpolation of structured surfaces, such as ﬂoors and
ceilings. These planar surfaces are critical to deﬁning the over-
all geometry of indoor scenes but are often partially obscured
or missing from the captured views [29], [30]. By segmenting
these incomplete surfaces—reﬁned by the extracted lines as
boundaries—we can predict their continuation beyond the
directly observed portions by generating new Gaussians. This
strategy enables us to optimize the representation of large
surfaces within the scene, enhancing the completeness of the
rendered map. Finally, we compress the Gaussian representa-
tion into mesh surfaces by incorporating regularization terms
through Poisson reconstruction [31]. This approach enables
the extraction of high-quality mesh, previously unavailable in
the Gaussian SLAM systems, making it readily available for
downstream tasks.
Overall, our work presents the following key contributions:
• We propose MG-SLAM, a novel RGB-D Gaussian SLAM
system that capitalizes on the MW hypothesis. This
assumption introduces lines and planar surfaces for robust
tracking, map reﬁnement, and surface completion for
neural-dense SLAM systems.
• We incorporate line segments along with an additional
fusion and ﬁltering strategy into the neural-dense SLAM
system, eﬀectively improving its tracking capabilities in
textureless indoor environments and enhancing the quality
of the dense Gaussian map.
• We establish hypothesis surfaces using extracted line
segments that represent planar boundaries. These surfaces
guide our eﬃcient interpolation of new Gaussians to
ﬁll gaps and holes in the reconstructed map, seamlessly
addressing areas where current Gaussian SLAM systems
face limitations due to unobserved geometry.
• Extensive experiments conducted on both large-scale syn-
thetic and real-world datasets demonstrate that our system
oﬀers state-of-the-art (SOTA) tracking and comprehensive
map reconstruction, achieving 50% lower ATE and 5dB
enhancement in PSNR on large-scale Apartment dataset,
meanwhile operating at a high frame rate.
II. RELATED WORK
A. Neural Dense SLAM
Neural Implicit SLAM systems [1], [3], [4], [6], [32],
[33], [34], [35], [36] leveraging NeRF [37] are adept at
handling complex reconstruction using implicit volumetric
representation. Despite these advancements, NeRF methods
often struggle with the over-smoothing issue, where ﬁne-
grained object-level geometry and features are diﬃcult to
capture during reconstruction [23]. Moreover, these methods
suﬀer from catastrophic loss as the scenes are implicitly
represented by MLPs.
In contrast, the high-ﬁdelity and fast rasterization capa-
bilities of 3D Gaussian Splatting [26] enable higher quality
and eﬃciency on scene reconstruction [20], [21], [23], [24],
[38], [39], [40], [41], [42], [43], [44]. MonoGS [21] utilizes
a map-centric SLAM approach that employs 3D Gaussian
representation for dynamic and high-ﬁdelity scene capture.
SplaTAM [20] adopts an explicit volumetric approach using
isotropic Gaussians, enabling precise camera tracking and map
densiﬁcation. Photo-SLAM [24] and RTG-SLAM [39] inte-
grate the traditional feature-based tracking system [45], [46]
with Gaussian mapping, providing robust tracking and excep-
tional real-time processing capabilities. However, existing
Gaussian SLAM systems lack eﬀective camera pose optimiza-
tion in textureless environments, limiting tracking accuracy
in indoor scenes. Moreover, they struggle to eﬀectively
reconstruct unobserved areas, often resulting in incomplete
reconstructions with gaps and holes. This limitation becomes
more problematic in settings where the camera’s movement is
restricted, leading to signiﬁcant unmodeled areas in structured
indoor scenes. Additionally, current Gaussian-based SLAM
approaches face challenges in direct mesh generation due to
the discrete nature of 3D Gaussian primitives, which compli-
cates surface extraction. To address this, recent Gaussian-based
systems [22], [38], [42] apply TSDF fusion [47] on ren-
dered images to produce meshes. However, this approach is
limited to observed viewpoints, leaving unobserved regions
incomplete. More critically, it fundamentally depends on the
oﬄine volumetric TSDF projection using rendered images,
which is independent of the reconstructed Gaussian maps.
To overcome these challenges, our method incorporates line
segments and planar surface assumption to seamlessly ﬁll the
gaps of structured surfaces and directly extract high-quality
mesh from volumetric Gaussian representations.
B. SLAM With Structure Optimization
Line features are known to signiﬁcantly enhance camera
pose optimization by capturing high-level geometric elements
and structural properties [48], [49]. Traditional SLAM systems
[13], [50], [51], [52], [53], [54], [55], [56] combine point
features, line segments, or planar surfaces to reﬁne cam-
era pose estimation and improve map reconstruction. These
approaches are particularly eﬀective in dynamic or textureless
environments, where point-based methods often face substan-
tial challenges.
Building on the MW hypothesis, recent advancements in
SLAM systems leveraging planar constraints have further
mitigated tracking and mapping drift. For instance, [57]
demonstrates accurate camera pose estimation and sparse 3D
map generation in monocular SLAM. [58] achieves drift-free
rotational motion estimation by leveraging structural regulari-
ties captured by line features. ManhattanSLAM [59] extends
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 3 -->
17036
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING, VOL. 22, 2025
Fig. 1.
Visualization of MG-SLAM on scene0000 00 and scene0207 00
of the ScanNet dataset [27]. Our method leverages robust line segments to
achieve superior camera pose estimation and scene reconstruction results.
Moreover, by applying structural surface constraints, we enhance and complete
the planar surfaces of the scene through the insertion of new Gaussian
primitives to ﬁll gaps.
this concept by optimizing camera trajectories and gener-
ating sparse maps with points, lines, and planes, alongside
dense surfel-based reconstructions. Similarly, [60] incorporates
Manhattan frame re-identiﬁcation to build robust rotational
constraints, which are tightly integrated into a bundle adjust-
ment framework. Planar-SLAM [61] focuses on planar mesh
reconstruction by utilizing line and sparse point features, while
[62] introduces drift-free rotation estimation through the use
of Gaussian spheres. Moreover, [63] reﬁnes spatial constraints
through advanced bundle adjustment leveraging structure con-
straints. [64] integrates low-cost LiDAR in structured scenes
for better view coverage.
Despite these advancements, most existing systems lever-
aging structure or planar constraints rely on sparse map
representations, such as point clouds or simple planar surfels,
which are insuﬃcient for reconstructing ﬁne-grained maps
with detailed textures. Additionally, the sparse map leads
to discontinuities in the reconstructed map, making surface
extraction and subsequent optimization diﬃcult. To bridge this
gap, we incorporate line features into the neural dense system,
enhancing its tracking capabilities in indoor environments and
enabling structured surface optimization.
III. METHOD
Figure 2 illustrates the pipeline of our proposed method.
Under the constraints of the Manhattan World, MG-SLAM
introduces line segments and structured surfaces to enhance
camera pose estimation and map reconstruction. Section III-B
details the tracking mechanism that utilizes both point and line
features. We utilize a speciﬁc strategy for fusing line segments
to ensure reliable identiﬁcation of line features. Section III-C
discusses the Gaussian representation, including a specialized
loss term dedicated to the reconstruction of line segments.
Section III-D describes the completion and reﬁnement of the
scene, grounded in the assumption of structured surfaces.
Section III-E describes the mesh generation utilizing regular-
ization losses.
A. Notation
We deﬁne the 2D image domain as P
∈R2, which
encompasses appearance color information C ∈N3, semantic
color S ∈N3, and depth data D ∈R+. Transitioning to the
3D world domain, denoted as X ∈R3, we introduce a camera
projection function π : X →P, mapping a 3D point Xi to its
2D counterpart Pi, and conversely, a backprojection function
θ : P →X, for the reverse mapping. (R, t) ∈S E (3) deﬁnes
the camera pose. I is the identity matrix.
B. Tracking
We utilize the backprojection of point features and line
segments extracted from 2D images into 3D space in par-
allel for optimizing the camera pose, based on PLVS [13].
Moreover, we propose strategies for line segment fusion and
suppression, designed to merge shorter segments into longer
ones and eliminate unstable lines. This method is grounded in
the understanding that longer line features tend to oﬀer greater
reliability and robustness throughout the tracking process.
1) Point Reprojection Error: Given a point observation pi
within a speciﬁc keyframe k, its reprojection error can be
computed as:
Lp
2D =
Pi −π(RkX w
i + Tk)

ΣPi ∈R
(1)
where X w
i is the 3D point back-projected to the world domain
and transformed to the camera frame X c
i . The term ΣPi =
σ2PiI2 represents the covariance matrix, encapsulating the
variance σ2Pi of point detection noise at diﬀerent scales within
a Gaussian image pyramid [13].
2) Line Reprojection and Backprojection Error: For line
segments, represented in the 3D world as a pair of endpoints
(Xp, Xq) ∈R3 and in the image plane as (Pp, Pq) ∈R2,
we identify the line segments on each level of the image
pyramid using the EDlines method [65] for eﬃcient extraction.
Matching line segments leverage the Line Band Descriptor
(LBD) method [66], and matches are scored according to the
Hamming distances between descriptors.
Upon observing a line segment l
=
(Pp, Pq)k, its
back-projected 3D endpoints can be computed as L
=
(θ(Pp), θ(Pq))k. Then the reprojection error of the line segment
can be interpreted as:
Ll
2D =

dk · π
 RkX w
p + t

dk · π
 RkX w
q + t



Σ2D
∈R2
(2)
dk =
¯Pp × ¯Pq
 ¯Pp × ¯Pq
|k
(3)
where
¯Pp = [PT
p , 1]T, ¯Pq = [PT
q , 1]T. The covariance matrix
Σ2D = σ2
diI2 is assumed to be diagonal for simplicity. Further,
we calculate the perpendicular distance in 3D between a map
point X w
i ∈R3 and a 3D line segment L as follows:
el
3D(L, X c
i )k =
(X c
i −θ(Pp)) × (X c
i −θ(Pq))

k
θ(Pp) −θ(Pq)

k
(4)
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 4 -->
LIU et al.: MG-SLAM: STRUCTURE GAUSSIAN SPLATTING SLAM WITH MANHATTAN WORLD HYPOTHESIS
17037
Fig. 2.
The two-phase pipeline illustration of our proposed MG-SLAM. The upper section visualizes the parallel processes of the tracking and mapping
systems. The lower section presents the post-optimization of scene interpolation and mesh extraction.
which
considers
the
cross-product
of
the
diﬀerences
between the point’s position and the back-projected positions
of the line’s endpoints, normalized by the distance between
these endpoints. Additionally, the distance between the world
domain point X w
i
and its corresponding back-projected point
in the image plane Pi ∈R2 is determined by:
el
P(Pi, X c
i ) =
X c
i −θ(Pi)

k ,
X c
i = RkX w
i + Tk
(5)
Consequently, the back projection error is formulated as:
Ll
3D =
e3D(θ(Pi), X w
p ) + β · eP(Pi, X w
p )
e3D(θ(Pi), X w
q ) + β · eP(Pi, X w
q )

k
∈R2
(6)
In this context, β ∈[0, 1] acts as a weighting parameter,
ensuring the endpoints of the 3D line segment remain stable
throughout the optimization process, thereby preventing drift.
3) Line Segment Fusion: The stability of tracking can be
notably enhanced by the presence of elongated line segments.
However, the extraction process, particularly through EDLines,
often yields fragmented segments as a consequence of various
disturbances like image noise. Our method integrates the
following key steps: (1) Fusing line segments that are direc-
tionally aligned within a one-degree angle diﬀerence, ensuring
they follow the same path. (2) Ensuring these segments are
close—within a 10-pixel distance of each other’s nearest
endpoints—yet do not overlap, preserving their distinctness
while allowing for precise merging. (3) Verifying the vertical
distance between the corresponding endpoints of one segment
to the entirety of another is less than certain pixel threshold,
maintaining geometric consistency. Only segments fulﬁlling
all three criteria are merged, producing longer and more
reliable lines. Furthermore, we ﬁlter out segments that fall
below a predeﬁned length threshold, relative to the image size,
to improve the system’s tracking robustness.
4) Full Bundle Adjustment Error: The overall objective
function for the full bundle adjustment is given by:
LBA =
X
K
X
U
ρ
Lp
2D

ΣP

+
X
V
ρ
Ll
2D

Σ2D

+
X
V
ρ
Ll
3D

Σ3D

(7)
Here, K represents the set of chosen keyframes, while U
and V denote the sets of points and lines extracted within the
current frame. The Huber cost function ρ(∥e∥)Σ−1) = eTΣ−1e is
applied to mitigate the inﬂuence of outliers. The optimization
process utilizes the Levenberg-Marquardt method [67], [68],
which solves the augmented normal equations by iteratively
updating the parameters Θ as:
(AT
ϵ Σ−1
ϵ Aϵ + ΛI)∆Θ = −AT
ϵ Σ−1
ϵ ϵ
(8)
C. Mapping
The map representation is based on 3D Gaussians primitives
G = αN(µw, Σw), where α ∈[0, 1] is the opacity, µw and
Σw are mean and covariance in world coordinate. Each Gi is
associated with the color of appearance and semantic feature
fi = {ci ∈R3, si ∈R3}. Semantic segmentation is utilized to
identify the surface such as ﬂoors for structure optimization
and surface extraction, explained in Section III-D.
1) Scene Representation: We use the standard point ren-
dering formula [69], [70] to splat G to render 2D image:
µP = π(RµG + T), ΣP = JRΣGRT JT
(9)
where µG and ΣG are mean and covariance of the Gaussian
primitives. J is the Jacobian of the linear approximation of
π(·). For each pixel pix ∈P, the inﬂuence of N Gaussians on
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 5 -->
17038
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING, VOL. 22, 2025
this pixel can be combined by sorting the Gaussians in depth
order and performing front-to-back alpha-blending:
Ppix =
X
i∈N
fiαi
i−1
Y
j=1
(1 −α j), Dpix =
X
i∈N
ziαi
i−1
Y
j=1
(1 −α j)
(10)
where Ppix and Dpix represent the pixel-wise appearance (Cpix
for color and Spix for semantic features) and depth respectively.
z is the distance to the mean of the Gaussian G along the
camera ray.
2) Mapping Loss: The scene representation is optimized
using keyframes obtained from the tracking system by mini-
mizing the photometric residual:
Lpix =
Cpix −CGT
pix
 + λS
Spix −SGT
pix

+ λD
Dpix −DGT
pix

(11)
In this context, Gaussian primitives are optimized to adjust
their optical and geometrical parameters to closely replicate
the observed scene with ﬁne-grained details. Furthermore, to
impose constraints on the scene utilizing the identiﬁed line
features that precisely delineate the lines and potential edges
in the current view, an additional line residual is introduced
for pixels associated with line features to enhance the map’s
accuracy. Consequently, the aggregate loss for mapping is
determined as follows:
LMapping =
X
pix∈P
Lpix + λℓ
X
pix∈V
Lpix
(12)
λS, λD and λℓin Equation (11) and Equation (12) are
weighting parameters.
D. Structure Optimization
In our optimization process, we focus particularly on reﬁn-
ing the representation of expansive surfaces that formulate the
indoor scene, such as the ﬂoor and ceiling. By applying the
MW assumption, we introduce the planar hypothesis surfaces
that are informed by the structural regularity of the extracted
3D line features. These constraints are employed to identify
surfaces that are not adequately represented and to interpolate
new Gaussians for densifying or ﬁlling in gaps with textures
that are reasonably predicted.
1) Gaussian Density: In our Gaussian map, we deﬁne the
density of the Gaussian ν : R3 →R+ as the sum of the
Gaussian values weighted by their alpha-blending coeﬃcients
at any given grid points p as:
ν(p) =
X
Gi
αGi exp(−1
2(p −µT
Gi)Σ−1
Gi (p −µGi) ,
(13)
where the µGi, ΣGi, αGi are the means, covariances, and alpha-
blending coeﬃcients of the Gaussians G. To simplify the
calculation, ν(p) can be approximated following [31] as:
ν∗(p) = αG∗exp(−1
2(p −µT
G∗)Σ−1
G∗(p −µG∗) ,
(14)
where G∗is the nearest Gaussian that contributes most to p.
This Gaussian density function ν∗(·) facilitates the subsequent
identiﬁcation of under-represented regions on the planar sur-
faces and the mesh generation process.
2) Map Calibration: Since the SLAM system relies on the
initial camera pose as a reference frame, the reconstructed
scenes usually do not satisfy the orthogonality assumption of
the Manhattan World. To overcome this issue, we calibrate the
reconstructed scene by applying a calibrating matrix K to the
coordinates and covariance of the Gaussians. This matrix K
is derived from clustering the directions of 3D line segments
that are presumed to align with the scene axes.
Algorithm 1 The Pseudo-Code for Calibration and Identiﬁca-
tion of Surface Boundary
Require: Line segments L ∈R3 and the reconstructed Gaus-
sian map G
1: Lﬁltered ←select |ℓ| > Tℓ▷Filter out small line segments
2: for each line segment ℓin Lﬁltered do
3:
{direction} ←(ℓ.end −ℓ.start).normalized()
▷Add direction vector
4: end for
5: ⃗x,⃗y,⃗z ←k means({direction}, k = 3) ▷Find axes direc-
tions
6: K ←I ∈R4 ▷Deﬁne calibration matrix
7: K[0 : 3, 0] ←⃗x
8: K[0 : 3, 1] ←⃗y
9: K[0 : 3, 2] ←⃗z
10: for each Gi in G do
11:
µGi ←K · µGi ▷Calibrate coordinates of Gaussian
12:
γGi ←K · γGi ▷Calibrate rotation of Gaussian
13: end for
14: for each line segment ℓin L do
15:
ℓ←K · ℓ▷Calibrate line segment
16: end for
17: B ←select maximum and minimum x and y for endpoints
in L
▷Deﬁne surface boundary
18: Output: boundary of surface B ∈R2
After aligning the structured surface boundaries to orthog-
onal directions, they extend across the xy →R2 plane. We
use the calibrated line segments to outline the rectangular
boundary of the hypothesized planes, capitalizing on the
dense line features commonly found at the corners of scenes.
Algorithm 1 presents the detailed implementation.
3) Surface Interpolation: Algorithm 2 shows our surface
interpolation strategy. Speciﬁcally, we utilize the semantic
information, incorporated in Equation (11), for identifying
the planar surfaces such as ﬂoors and ceilings. The chosen
target Gaussians are then projected from the 3D space onto
the 2D hypothesis xy plane. Subsequently, we apply a density
threshold to the sampling density function, deﬁned in Equation
(14), to detect potential holes or gaps on the surface. New
Gaussian primitives are generated at the identiﬁed gaps that
fall below a density threshold. Finally, we train a PointNet++
model [72], ﬁtting it to the presented surface to learn the
presented color patterns and interpolate the texture color of the
new Gaussians based on their spatial correlation. We empiri-
cally found that the PointNet++ network, which incorporates
spatial correlation, is suﬃciently eﬀective in predicting the
color patterns of new primitives on the gaps of structured
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 6 -->
LIU et al.: MG-SLAM: STRUCTURE GAUSSIAN SPLATTING SLAM WITH MANHATTAN WORLD HYPOTHESIS
17039
Algorithm 2 The Pseudo-Code for Scene Completion
Require: The calibrated Gaussian scene G, the mask M
extracted from semantic segmentation, and the boundary
of surface B, threshold of density Tdensity.
1: Mtg ←semantic masks that represent target surfaces of
the scene, e.g. ﬂoor
2: for each structure surface Gi in G(Mtg) do
3:
Gi ←DBSCAN.fit(Gi)
▷Apply DBSCAN [71] to eliminate outliers
4:
¯z ←avg(z) for z ∈µGi
5:
Π ←grid(B) ▷Create the 2D grid of hypothesis surface
6:
Di ←compute the Gaussian density for each point of
the grid in Π.
7:
if Di ≤Tdensity then
8:
µpred ←generates new Gaussians centers
9:
end if
10:
B∗
Gi ←ramdomly sample Gaussians from G to form
training batch.
11:
for each batch Bi in B∗
Gi do
12:
F(µGi|Gi ∈Bi) ←train PointNet++ [72] model on
Bi
13:
end for
14:
Cpred ←F(µpred) ▷Predict the color of new Gaussians
15:
Gpred ←formulate new Gaussians using {µpred} and
{Cpred}
16: end for
17: G ←Gpred ▷Update the overall scene
18: Output: structure optimized G
surfaces. Given that these planar surfaces typically exhibit less
texture than speciﬁc objects, employing more sophisticated
models does not provide substantial beneﬁts but increases
computational expenses.
Through this method, we aim to seamlessly combine the
detailed representation provided by Gaussians with the need
for explicit scene geometry that accounts for unseen views,
thereby achieving a more comprehensive and accurate recon-
struction of planar surfaces in indoor environments.
E. Mesh Generation
We employed the surface extraction method proposed by
[31] with a novel normal regularization term during the scene
reﬁnement. To compress the Gaussians to align closely with
the surface, [31] forces the Gaussians to be ﬂat by setting one
of their scaling factors nearly to zero. Consequently, ν∗(p) can
be further simpliﬁed to ˆν∗(p) = αG∗exp
 −2s−2
G∗⟨p −µG∗, nG∗⟩2
where sG represents the smallest scaling factor and nG its
corresponding normal. Here, αG∗= 1 is set to avoid semi-
transparent Gaussians.
Subsequently, a regularization term is used to align the SDF
function derived from the density function with its estimation
from the scene as:
Lsdf =
1
|Xp|
X
x∈Xp
|sd f(x) −
ˆ
sd f(x)|
(15)
Here, Xp
is the sampled 3D points and
sd f(x)
=
±sG∗p
−2 log(ˆν∗(x)) represents the ideal SDF. The estimated
TABLE I
MG-SLAM HYPERPARAMETERS
SDF
ˆ
sd f(x) is determined by subtracting the depth of x from
the corresponding rendered depth map. This regularization
term encourages closer alignment of the estimated surface with
the observed depth.
Moreover, to derive smooth surfaces, particularly in ﬂat
and textureless ﬂoor areas, we adopt the MW hypothesis,
which assumes that planar ﬂoors are orthogonal to the vertical
axis. We enforce this constraint by introducing a normal
regularization term for the estimated SDF:
Lnormal =
1
|Xﬂoor|
X
x2D∈Pﬂoor
ˇˇˇˇ1 −ˆn · ∇f(x)
∥∇f(x)∥
ˇˇˇˇ
(16)
In this context, the projected points x2D, which pass through
the camera ray and fall within the ﬂoor area, are encouraged
to align their normal to the ideal vector ˆn = ⟨0, 0, 1⟩. The
set Pﬂoor consists of image pixels identiﬁed as the ﬂoor using
semantic segmentation.
IV. EXPERIMENT
A. Experiment Settings
1) Implementation Details: Our tracking system is imple-
mented in C++, while the mapping system uses Python3
and CUDA C. The experiments were conducted using a
single RTX A100-80GB GPU and 24-core AMD EPYC 7402
processor. We use Adam optimizer for Gaussian representation
optimization and network training. The hyperparameters for
3D Gaussian splitting [26] are the same as the original paper.
Table I presents the values for hyperparameters in our system.
For Gaussian insertion, in each keyframe, we sampled µG in
Equation (9) from the rendered depth map Dpix following
the distribution of N(Dpix, 0.2σD). For uninitialized areas,
we initialize Gaussians by sampling from N( ¯Dpix, 0.5σD)
where ¯Dpix is the mean value. For pruning, we remove the
Gaussians with opacity less than 0.6. Our system incorporates
a segmentation loss introduced in [23], and we leverage this
semantic information to eﬀectively extract structured surfaces,
such as ﬂoors and ceilings, from the reconstructed scenes.
We subsequently train a PointNet++ [72] network for each
structure surface and interpolate the missing geometry by
predicting the color of inserted Gaussians.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 7 -->
17040
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING, VOL. 22, 2025
TABLE II
QUANTITATIVE COMPARISON OF OUR METHOD AND THE BASELINES IN TRAINING VIEW RENDERING ON THE REPLICA-V1 DATASET [73]
2) Datasets: We evaluate our method using two datasets:
Replica [73], a synthetic dataset, and ScanNet [27], a chal-
lenging real-world dataset. The large-scale Replica Apartment
dataset used in our experiments was released by Tandem [74].
For the Replica dataset [73], the ground-truth camera pose
and semantic maps are obtained through Habitat simulation
[75]. In the case of the ScanNet dataset [27], the ground-truth
camera poses are derived using BundleFusion [76]. Moreover,
we further validate our method on a private data trajectory
collected using our physical platform.
3) Metrics: To assess the quality of the reconstruction, we
employ metrics such as PSNR, SSIM, and LPIPS. For evaluat-
ing camera pose, we use the average absolute trajectory error
(ATE RMSE). The real-time processing capability, essential
for SLAM systems, is measured in frames per second (FPS).
Best results are shaded as ﬁrst, second, and third.
4) Baselines:
We evaluate our tracking and mapping
results against state-of-the-art methods, including NeRF-based
approaches such as NICE-SLAM [1], Co-SLAM [3], ESLAM
[4], and Point-SLAM [6], as well as recent Gaussian-based
methods including SplaTAM [20], MonoGS [21], Photo-
SLAM [24], and RTG-SLAM [39]. The results for MonoGS
[21] were obtained using its RGB-D mode.
B. Evaluation on Replica-V1 Dataset
In line with conventional evaluations from previous stud-
ies, we performed quantitative comparisons of training view
rendering on the Replica-V1 dataset [73], which comprises
8 single-room scenes. As presented in Table II, our method
demonstrates superior rendering quality. This improvement
stems from the use of a denser primitive distribution, as
adopted in MonoGS [21], leading to higher PSNR values,
TABLE III
SYSTEM COMPARISON IN TERMS OF TRACKING, MAPPING, RENDER-
ING FPS, AND MEMORY USAGE BETWEEN OUR METHOD AND THE
NEURAL DENSE BASELINES ON THE REPLICA DATASET [73].
THE VALUES REPRESENT THE AVERAGE OUTCOMES ACROSS
8 SCENES. NOTE THAT RTG-SLAM [39] AND PHOTO-
SLAM [24] INCORPORATE ORB-SLAM2 [45]
AND
ORB-SLAM3 [77] FOR TRACKING
and the integration of the segment reconstruction loss, which
contributes to a competitive structural similarity score.
Additionally, Table III provides comparisons of tracking
accuracy and system eﬃciency. By leveraging line features,
our approach achieves state-of-the-art tracking performance
compared to baseline methods, further contributing to optimal
map reconstruction quality. Regarding system eﬃciency, we
assessed the frame rate of each component and memory usage.
Unlike systems employing frame-to-frame tracking [3], [4],
[6], [20], [21], which exhibit signiﬁcantly low tracking speeds,
methods such as Photo-SLAM [24], RTG-SLAM [39], and
ours achieve notable advantages in tracking performance with
exceptional frame rates. This is attributed to the incorporation
of traditional feature-based optimization, which avoids itera-
tive rendering and photometric loss computations for camera
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 8 -->
LIU et al.: MG-SLAM: STRUCTURE GAUSSIAN SPLATTING SLAM WITH MANHATTAN WORLD HYPOTHESIS
17041
TABLE IV
QUANTITATIVE COMPARISON
OF OUR METHOD
AND
THE BASELINE
APPROACHES
IN TRAINING VIEW RENDERING
ON
THE REPLICA
APARTMENT DATASET [73]. THE UNDERLINE INDICATES THAT
RELOCALIZATION WAS TRIGGERED DUE TO ACCUMULATED
TRACKING ERRORS. THE DASH INDICATES
SYSTEM FAILURES
pose optimization—a common bottleneck in neural dense
systems—thereby remarkably enhancing tracking speed. For
mapping, compared to Photo-SLAM [24] and RTG-SLAM
[39], which utilize relatively sparse primitive representations,
our method opts for a denser map to improve reconstruction
quality. While this results in slower mapping speeds and
moderate memory usage, it oﬀers a balanced trade-oﬀfor
higher reconstruction ﬁdelity.
C. Evaluation on Replica-Apartment Dataset
To evaluate the robustness of the system in large-scale
indoor environments, we evaluate MG-SLAM on the Replica
Apartment dataset [73]. This dataset contains extensive
multi-room scenes, complex object geometries, and looping
trajectories across rooms. Table IV presents the rendering
quality of our method compared to both NeRF-based and
Gaussian-based approaches over ﬁve selected scenes. MG-
SLAM shows notable improvements, particularly achieving a
7dB improvement in the apartment 0 scene over Gaussian
SLAM systems. This optimal performance is largely attributed
to the inclusion of the fused line segments, which lays
a solid foundation for loop closure and pose optimization.
Compared to the NeRF-based Point-SLAM [6], which uti-
lizes additional neural point clouds for robust tracking and
ﬁne-grained mapping, our method delivers superior averaged
reconstruction outcomes while maintaining optimal real-time
processing capabilities, achieving a system frame rate 50 times
faster.
Table V presents the quantitative evaluations of tracking
ATE [cm] on the apartment scenes. Our method achieves state-
of-the-art tracking accuracy compared to both frame-to-frame
TABLE V
QUANTITATIVE COMPARISON OF OUR METHOD WITH TRADITIONAL,
NERF-BASED, AND GAUSSIAN-BASED RGB-D SLAM SYSTEMS IN
TERMS OF ATE [CM] ON THE REPLICA APARTMENT DATASET [73].
THE UNDERLINED VALUES INDICATE THAT RELOCALIZATION
WAS NECESSARY DUE TO LOSS OF TRACKING
TABLE VI
QUANTITATIVE COMPARISON OF OUR METHOD AND THE BASELINES IN
TRAINING VIEW RENDERING ON THE SCANNET DATASET [27]
and feature-based tracking systems. Speciﬁcally, ORB-based
systems [24], [39], [45], [77] struggle in textureless planar
environments, such as the stairs or corridors in apartmen 0
and apartment 1, leading to tracking failures, frequent relo-
calizations, and errors in the reconstructed scenes. Leveraging
the segment fusion strategy, which provides robust line fea-
tures, our method also outperforms PLVS [13] under scenarios
involving extensive camera movement and rotation which are
commonly encountered in the apartment dataset.
Moreover, we conducted qualitative comparisons on novel-
view synthesis against Gaussian-based approaches [20], [21],
[24] as shown in Figure 3. These comparisons highlight the
superior performance of our method over baseline approaches
in large-scale complex indoor environments.
D. Evaluation on ScanNet Dataset
We provide quantitative assessments of reconstruction qual-
ity using the ScanNet dataset [27] in Table VI. Our approach
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 9 -->
17042
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING, VOL. 22, 2025
Fig. 3. The novel view sythesis of the scene apartment 0 from the Replica Apartment dataset [73]. The top left shows the line segments extracted in 3D
space. The bottom left illustrates the overall reconstructed scene.
TABLE VII
QUANTITATIVE COMPARISON OF OUR METHOD AND THE NEURAL DENSE
BASELINES IN TERMS OF ATE [CM] ON THE SCANNET DATASET [27]
delivers state-of-the-art results, outperforming other Gaussian-
based methods by a notable 3dB in PSNR in real-world
environments. The tracking evaluation results are shown
in Table VII. Our method remarkably reduces the ATE
RMSE (cm) error, achieving 20% improvements over baseline
approaches.
In Figure 4, we visualize the novel-view synthesis results
for MG-SLAM, comparing it with both NeRF-based and
Gaussian-based SLAM systems. Our method exhibits robust
and high-ﬁdelity reconstruction capabilities. In comparison to
Point-SLAM [6], our method oﬀers complete scene recon-
structions and ﬁner texture details, beneﬁting from the use of
Gaussians to handle complex geometries. Among Gaussian-
based methods, SplaTAM [20] struggles in large scenes
with complex camera loops due to its lack of bundle
adjustment in the tracking system. Similarly, MonoGS [21]
generally delivers reliable quality but struggles with object-
level reconstruction drift. Photo-SLAM [24] achieves better
reconstruction results compared to other baseline approaches
but suﬀers from ﬂoaters and artifacts that compromise its
geometric accuracy. In contrast, MG-SLAM excels with robust
tracking and eﬀective bundle adjustment incorporating line
segments, enabling superior detailed reconstructions even sur-
passing the ground-truth mesh derived from oﬄine methods.
E. Evaluation on Physical Platform
To assess our system’s performance in real-world envi-
ronments, we utilzie a private data trajectory gathered with
our physical platform. The robot was deployed in the NTU
building to navigate and collect data from a challenging long
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 10 -->
LIU et al.: MG-SLAM: STRUCTURE GAUSSIAN SPLATTING SLAM WITH MANHATTAN WORLD HYPOTHESIS
17043
Fig. 4. Qualitative comparison of our method and the baselines for novel-view synthesis on the ScanNet dataset [27]. The outcomes show that our method
provides more robust and ﬁne-grained reconstructions in real-world complex scenes compared to current NeRF-based and Gaussian-based approaches.
Fig. 5. Qualitative comparison of our method and the baselines on the trajectory collected using our physical platform. The left side displays the line and point
features extracted by our tracking system. The right side presents the reconstruction comparisons with Gaussian-based methods, with MG-SLAM showing
more reliable reconstruction results in long-horizon and textureless indoor environments.
corridor, notable for its lack of texture and extensive length
exceeding 200 meters. Ground-truth poses were computed
by registering the LiDAR point cloud with the scanned
point cloud generated by the Leica ScanStation. Figure 5
displays the comparisons of training view synthesis between
our method and Gaussian-based systems, where our method
demonstrates signiﬁcant advantages; however, it still lacks
some details compared to the ground-truth images due to
the long-horizon trajectory. Table VIII provides quantitative
results, showing that our method achieves more accurate
tracking and mapping performance. The baseline approaches
struggle with tracking in textureless environments, further
aﬃrming the eﬀectiveness of our method in indoor scenes.
F. Evaluation of Scene Completion
The Gaussian SLAM faces limitations in interpolating the
geometry of unseen views. This issue is especially evident
in indoor scenes that feature complex layouts, where basic
surfaces like ﬂoors are often obscured and poorly represented.
Figure 6 qualitatively compared our method with recent
Gaussian-based approaches in the Replica scenes [73] that
presented uncovered geometry in camera trajectories. Utilizing
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 11 -->
17044
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING, VOL. 22, 2025
Fig. 6. Qualitative comparison on the novel-view synthesis of our method and the Gaussian SLAM baselines for hole ﬁllings on the Replica dataset [73].
Our method demonstrates superior capability in interpolating and ﬁlling gaps on structured planar surfaces, such as ﬂoors and ceilings.
TABLE VIII
QUANTITATIVE COMPARISON OF OUR METHOD AND GAUSSIAN-BASED
APPROACHES IN TERMS OF TRACKING AND MAPPING PERFORMANCE
ON TRAJECTORY COLLECTED USING OUR PHYSICAL PLATFORM.
THE DASH INDICATES SYSTEM FAILURES
the surface interpolation strategy based on the MW hypothesis
[28], our method can accurately detect and proactively gener-
ate new Gaussians eﬃciently to ﬁll gaps with certain textures,
whereas Gaussian baseline methods exhibit substantial defects
on structured surfaces.
V. ABLATION STUDY
In this section, we provide a comprehensive ablation anal-
ysis of the hyperparameters and each component’s eﬀect.
A. Ablation of Hyperparameters
Our tracking system, which advances the foundation of
PLVS system [13], utilizes feature points and fused line
segments for optimizing camera poses and performing bundle
adjustments. The upper row of Figure 7 illustrates the eﬀects
Fig. 7.
The ablation study examining the impact of the number of points,
line segments, keyframe intervals, and downsample ratios in MG-SLAM on
scene0000 00 from the ScanNet dataset [27].
of varying the number of points and line segments on the
ATE loss [cm] and the frame rate of the tracking system. We
identiﬁed an optimal region where having too few points and
lines results in a lack of suﬃcient anchor features, whereas too
many points and lines can lead to notable feature mismatches
that reduce tracking accuracy. The tracking FPS reduces as a
tradeoﬀwith the increase in the number of features. Compared
to tracking solely by point features, including line segment
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 12 -->
LIU et al.: MG-SLAM: STRUCTURE GAUSSIAN SPLATTING SLAM WITH MANHATTAN WORLD HYPOTHESIS
17045
Fig. 8. The ablation study of the line segment extraction outcomes for our method on the scene frl apartment 4 from the Replica Apartment dataset [73]. By
incorporating backprojection, ﬁltering, and segment fusion, our system generates rich and robust line segment features for subsequent tracking and mapping
procedures.
Fig. 9. The visualization of the ablation study in scene0000 00 from the ScanNet dataset [27]. From left to right, the results are displayed starting completely
without line features, adding line segments, integrating segment fusion, and ﬁnally incorporating line photometric loss in mapping. Key diﬀerences are
highlighted with colored boxes.
Fig. 10. The ablation study compares surface identiﬁcation approaches. The
identiﬁed hypothesis surfaces using diﬀerent methods are shaded in black.
The left ﬁgure displays the scene without calibration, where the primary
axes of the scene do not align with the world coordinates. The middle ﬁgure
demonstrates the application of PCA to the centers of Gaussians µG associated
with structured surfaces, where the boundaries of µG deﬁne the surface. The
right ﬁgure shows the calibration results, achieved by clustering line segments
and deﬁning surfaces based on the boundaries of these line features.
features noticeably decreases the frame rate; however, it still
maintains a high rate over 30 FPS and does not become
the speed bottleneck of the overall systems. The bottom
row of Figure 7 illustrates the eﬀects of default keyframe
intervals and downsample ratio on the Gaussian map. During
the mapping process, we uniformly downsample the point
cloud generated from RGB-D input by a speciﬁc ratio to
accelerate the system and conserve memory. We observed a
marginal PSNR disparity when maintaining a relatively small
downsample ratio.
B. Ablation of Line Segment Extraction
The
extracted
line
segments
play
a
crucial
role
in
serving as robust feature foundations in the subsequent
optimization processes. Figure 8 illustrates the ablation results
for our feature extraction approach for these 3D line segments,
demonstrating the eﬃcacy of segment backprojection, line
fusion, and ﬁltering strategies in providing clear and accurate
line features.
C. Ablation of Tracking and Mapping Loss
MG-SLAM employs line segments for robust camera pose
optimization and ﬁne-grained map reconstruction. Speciﬁcally,
fused line segments are utilized in the bundle adjustment
of the tracking procedure, and a photometric loss related
to line features is integrated into the map optimization. We
use scene0000 00 from the ScanNet dataset [27] to conduct
the ablation study that evaluates the impact of each loss.
This scene was chosen because it contains rich line features,
which allow for a clear assessment of each component’s
eﬀectiveness. Table IX presents the ATE [cm] and PSNR [dB]
in relation to the use of each loss. We observed that inte-
grating line features remarkably enhances tracking accuracy,
and the fusion of line segments, which facilitates robust edge
extraction, further improves the performance. For mapping,
the photometric loss provides additional geometric constraints,
thereby oﬀering optimal reconstruction quality.
In Figure 9, we present a typical case in scene0000 00
to demonstrate the eﬀectiveness of each loss component.
This case features a bike that appears multiple times along
the camera trajectory. Compared to relying solely on point
features, adding fused line segments signiﬁcantly improves
the quality of map reconstruction. This enhancement mitigates
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 13 -->
17046
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING, VOL. 22, 2025
Fig. 11. The ablation of normal regularization for surface extraction on the Replica dataset [73]. We compare the normal map of the extracted mesh, both
without and with normal regularization, to the ground-truth mesh. The regularization reﬁnes the planar surfaces of the scene, i.e. ﬂoors, eﬀectively.
TABLE IX
QUANTITATIVE
COMPARISON
OF
THE
ABLATION
STUDY
ON
LINE
LOSSES, MEASURING ATE [CM] AND PSNR [DB], CONDUCTED ON
SCENE0000 00 OF THE SCANNET DATASET [27]. THE CHECK-
MARK SYMBOL INDICATES THE EMPLOYMENT OF THE METHOD
scene drift by providing more accurate camera pose estimation.
Moreover, incorporating the photometric loss of line features
in mapping results in more detailed object-level geometry and
more precise reconstruction of line-like textures.
D. Ablation of Surface Extraction
We employ hypothesis surfaces regularized by line seg-
ments to pinpoint potential gaps on structured surfaces,
as described in Section III-D. These identiﬁed hypothesis
surfaces are designed to precisely mirror the positions of
the planar surfaces with minimal redundancy. To delineate
these surfaces, we initially adjust the reconstructed map to
adhere to the orthogonality assumption of MW. Figure 10
displays the hypothesized rectangular planes produced with-
out calibration, with PCA calibration based on the centers
of Gaussian primitives, and with calibration using features
from extracted segments. It is evident that our segment-based
method eﬃciently captures the ﬂoor surface with reduced
redundancy.
E. Ablation of Mesh Generation
Following map reconstruction and surface interpolation,
we extract meshes from the Gaussian map, as outlined in
Section III-E. Figure 11 illustrates the results of mesh extrac-
tion with and without the proposed normal regularization loss.
We observe that incorperating this regularization term results
in smoother mesh extraction on the structured surface such as
ﬂoors compared to the original method introduced by [31].
VI. CONCLUSION
In this study, we present MG-SLAM, a Gaussian-based
SLAM method based on the MW hypothesis. MG-SLAM
employs fused line segments and point features for robust pose
estimation and map reﬁnement. Furthermore, by leveraging the
line segments and planar surface assumption, we eﬃciently
generate new Gaussian primitives in gaps on structural sur-
faces caused by obstructions or unseen geometry. Extensive
experiments demonstrate that our method delivers state-of-the-
art tracking and mapping performance, while also maintaining
real-time processing speed.
VII. LIMITATIONS
MW assumes surface planes generally align with the three
orthogonal directions deﬁned by the Cartesian coordinate sys-
tem. This presents challenges in reﬁning planes or layouts that
are slanted or not strictly orthogonal. To better accommodate
diverse outdoor urban environments, future research could
incorporate multiple horizontal dominant directions, as seen in
Atlanta World [78], include additional sloping directions like
those in Hong Kong World [79], or adopt uniform inclination
angles for the dominant directions proposed in San Francisco
World [80]. Additionally, since the optimization of our dense
map primarily relies on line features and large planar surfaces,
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 14 -->
LIU et al.: MG-SLAM: STRUCTURE GAUSSIAN SPLATTING SLAM WITH MANHATTAN WORLD HYPOTHESIS
17047
it struggles with reﬁning piece-wise or ﬁne-grained structural
planes, such as walls and windows. One potential approach is
to leverage the coplanarity of surfaces for structural regulariza-
tion [61], [81], although this proves particularly challenging in
indoor environments with complex layouts where identifying
coplanar surfaces is diﬃcult.
ACKNOWLEDGMENT
Any opinions, ﬁndings, and conclusions or recommenda-
tions expressed in this article are those of the authors and do
not reﬂect the views of National Research Foundation.
REFERENCES
[1]
Z. Zhu et al., “NICE-SLAM: Neural implicit scalable encoding for
SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2022, pp. 12786–12796.
[2]
X. Kong, S. Liu, M. Taher, and A. J. Davison, “VMAP: Vectorised object
mapping for neural ﬁeld SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jun. 2023, pp. 952–961.
[3]
H. Wang, J. Wang, and L. Agapito, “Co-SLAM: Joint coordinate and
sparse parametric encodings for neural real-time SLAM,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023,
pp. 13293–13302.
[4]
M. M. Johari, C. Carta, and F. Fleuret, “ESLAM: Eﬃcient dense SLAM
system based on hybrid representation of signed distance ﬁelds,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023,
pp. 17408–17419.
[5]
Z. Zhu et al., “NICER-SLAM: Neural implicit scene encoding for RGB
SLAM,” in Proc. Int. Conf. 3D Vis. (3DV), Mar. 2024, pp. 42–52.
[6]
E. Sandstr¨om, Y. Li, L. Van Gool, and M. R. Oswald, “Point-SLAM:
Dense neural point cloud-based SLAM,” in Proc. IEEE/CVF Int. Conf.
Comput. Vis. (ICCV), Oct. 2023, pp. 18387–18398.
[7]
T. Deng et al., “PLGSLAM: Progressive neural scene represenation with
local to global bundle adjustment,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit., vol. 34, pp. 19657–19666, Jun. 2024.
[8]
T. Deng et al., “NeSLAM: Neural implicit mapping and self-supervised
feature tracking with depth completion and denoising,” IEEE Trans.
Autom. Sci. Eng., vol. 22, no. 3, pp. 12309–12321, Mar. 2025.
[9]
R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “DTAM: Dense
tracking and mapping in real-time,” in Proc. Int. Conf. Comput. Vis.,
Nov. 2011, pp. 2320–2327.
[10] R. F. Salas-Moreno, R. A. Newcombe, H. Strasdat, P. H. J. Kelly, and
A. J. Davison, “SLAM++: Simultaneous localisation and mapping at the
level of objects,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.,
Jun. 2013, pp. 1352–1359.
[11] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM: A
versatile and accurate monocular SLAM system,” IEEE Trans. Robot.,
vol. 31, no. 5, pp. 1147–1163, Oct. 2015.
[12] H. Gao, X. Zhang, J. Wen, J. Yuan, and Y. Fang, “Autonomous indoor
exploration via polygon map construction and graph-based SLAM using
directional endpoint features,” IEEE Trans. Autom. Sci. Eng., vol. 16,
no. 4, pp. 1531–1542, Oct. 2019.
[13] L. Freda, “PLVS: A SLAM system with points, lines, volumetric
mapping, and 3D incremental segmentation,” 2023, arXiv:2309.10896.
[14] Z. Wang et al., “LF-VISLAM: A SLAM framework for large ﬁeld-
of-view cameras with negative imaging plane on mobile agents,”
IEEE
Trans.
Autom.
Sci.
Eng.,
vol. 21,
no. 4,
pp. 6321–6335,
Oct. 2024.
[15] X. Lin, J. Ruan, Y. Yang, L. He, Y. Guan, and H. Zhang, “Robust
data association against detection deﬁciency for semantic SLAM,” IEEE
Trans. Autom. Sci. Eng., vol. 21, no. 1, pp. 868–880, Jan. 2024.
[16] L. Xu et al., “Grid-guided neural radiance ﬁelds for large urban scenes,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
2023, pp. 8296–8306.
[17] W. Yan, Y. Chen, W. Zhou, and R. Cong, “MVoxTi-DNeRF: Explicit
multi-scale voxel interpolation and temporal encoding network for
eﬃcient dynamic neural radiance ﬁeld,” IEEE Trans. Autom. Sci. Eng.,
vol. 22, pp. 5096–5107, 2025.
[18] Z. Li, C. Wu, L. Zhang, and J. Zhu, “DGNR: Density-guided neural
point rendering of large driving scenes,” IEEE Trans. Autom. Sci. Eng.,
vol. 22, pp. 4394–4407, 2025.
[19] Y. Zhang, G. Chen, and S. Cui, “Eﬃcient large-scale scene representa-
tion with a hybrid of high-resolution grid and plane features,” Pattern
Recognit., vol. 158, Feb. 2025, Art. no. 111001.
[20] N. Keetha et al., “SplaTAM: Splat, track & map 3D Gaussians for
dense RGB-D SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. (CVPR), Jun. 2024, pp. 21357–21366.
[21] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, “Gaussian splat-
ting SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2024, pp. 18039–18048.
[22] V.
Yugay,
Y.
Li,
T.
Gevers,
and
M.
R.
Oswald,
“Gaussian-
SLAM: Photo-realistic dense SLAM with Gaussian splatting,” 2023,
arXiv:2312.10070.
[23] M. Li et al., “SGS-SLAM: Semantic Gaussian splatting for neural dense
SLAM,” in Proc. Eur. Conf. Comput. Vision. Switzerland: Springer, Oct.
2024, pp. 163–179.
[24] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-SLAM: Real-time
simultaneous localization and photorealistic mapping for monocular,
stereo, and RGB-D cameras,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jun. 2024, pp. 21584–21593.
[25] T. Deng et al., “Compact 3D Gaussian splatting for dense visual SLAM,”
2024, arXiv:2403.11247.
[26] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, “3D Gaussian
splatting for real-time radiance ﬁeld rendering,” ACM Trans. Graph.,
vol. 42, no. 4, pp. 1–14, Aug. 2023.
[27] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and
M. Nießner, “ScanNet: Richly-annotated 3D reconstructions of indoor
scenes,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR),
Jul. 2017, pp. 5828–5839.
[28] J. M. Coughlan and A. L. Yuille, “The Manhattan world assumption:
Regularities in scene statistics which enable Bayesian inference,” in
Proc. Adv. Neural Inf. Process. Syst., vol. 13, Jan. 2000, pp. 845–851.
[29] H. Guo et al., “Neural 3D scene reconstruction with the Manhattan-world
assumption,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2022, pp. 5501–5510.
[30] X. Zhou et al., “Neural 3D scene reconstruction with indoor pla-
nar priors,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 9,
pp. 6355–6366, Sep. 2024.
[31] A. Gu´edon and V. Lepetit, “SuGaR: Surface-aligned Gaussian splatting
for eﬃcient 3D mesh reconstruction and high-quality mesh rendering,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
2024, pp. 5354–5363.
[32] M. Li, J. He, Y. Wang, and H. Wang, “End-to-end RGB-D SLAM with
multi-MLPs dense neural implicit representations,” IEEE Robot. Autom.
Lett., vol. 8, no. 11, pp. 7138–7145, Nov. 2023.
[33] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, “GO-SLAM: Global opti-
mization for consistent 3D instant reconstruction,” in Proc. IEEE/CVF
Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 3727–3737.
[34] W. Zhang, T. Sun, S. Wang, Q. Cheng, and N. Haala, “HI-SLAM:
Monocular real-time dense mapping with hybrid implicit ﬁelds,” IEEE
Robot. Autom. Lett., vol. 9, no. 2, pp. 1548–1555, Feb. 2024.
[35] L. Liso, E. Sandstr¨om, V. Yugay, L. Van Gool, and M. R. Oswald,
“Loopy-SLAM: Dense neural SLAM with loop closures,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024,
pp. 20363–20373.
[36] H. Zhou et al., “MoD-SLAM: Monocular dense mapping for unbounded
3D scene reconstruction,” IEEE Robot. Autom. Lett., vol. 10, no. 1,
pp. 484–491, Jan. 2025.
[37] J. Mccormac, R. Clark, M. Bloesch, A. Davison, and S. Leutenegger,
“Fusion++: Volumetric object-level SLAM,” in Proc. Int. Conf. 3D Vis.
(3DV), Sep. 2018, pp. 32–41.
[38] C. Yan et al., “GS-SLAM: Dense visual SLAM with 3D Gaussian
splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2024, pp. 19595–19604.
[39] Z. Peng et al., “RTG-SLAM: Real-time 3D reconstruction at scale using
Gaussian splatting,” in Proc. Special Interest Group Comput. Graph.
Interact. Techn. Conf. Conf. Papers, Jul. 2024, pp. 1–11.
[40] F. Tosi et al., “How NeRFs and 3D Gaussian splatting are reshaping
SLAM: A survey,” 2024, arXiv:2402.13255.
[41] J. Huang, M. Li, L. Sun, A. X. Tian, T. Deng, and H. Wang, “NGM-
SLAM: Gaussian splatting SLAM with radiance ﬁeld submap,” 2024,
arXiv:2405.05702.
[42] L. Zhu, Y. Li, E. Sandstr¨om, S. Huang, K. Schindler, and I. Armeni,
“LoopSplat: Loop closure by registering 3D Gaussian splats,” in Proc.
Int. Conf. 3D Vis. (3DV), Aug. 2024, pp. 1–10.
[43] S. Ha, J. Yeon, and H. Yu, “RGBD GS-ICP SLAM,” in Proc. Eur. Conf.
Comput. Vis. Cham, Switzerland: Springer, Oct. 2024, pp. 180–197.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 15 -->
17048
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING, VOL. 22, 2025
[44] J. Hu et al., “CG-SLAM: Eﬃcient dense RGB-D SLAM in a consistent
uncertainty-aware 3D Gaussian ﬁeld,” in Proc. Eur. Conf. Comput. Vis.
Cham, Switzerland: Springer, Oct. 2024, pp. 93–112.
[45] R. Mur-Artal and J. D. Tard´os, “ORB-SLAM2: An open-source SLAM
system for monocular, stereo, and RGB-D cameras,” IEEE Trans.
Robot., vol. 33, no. 5, pp. 1255–1262, Oct. 2017.
[46] C.-M. Chung et al., “Orbeez-SLAM: A real-time monocular visual
SLAM with ORB features and NeRF-realized mapping,” in Proc. IEEE
Int. Conf. Robot. Autom. (ICRA), May 2023, pp. 9400–9406.
[47] B. Curless and M. Levoy, “A volumetric method for building complex
models from range images,” in Proc. Annu. Conf. Comput. Graph.
Interact. Tech., 1996, pp. 303–312.
[48] L. Zhou, G. Huang, Y. Mao, S. Wang, and M. Kaess, “EDPLVO:
Eﬃcient direct point-line visual odometry,” in Proc. Int. Conf. Robot.
Autom. (ICRA), May 2022, pp. 7559–7565.
[49] Z. Xu et al., “PLPL-VIO: A novel probabilistic line measurement model
for point-line-based visual-inertial odometry,” in Proc. IEEE/RSJ Int.
Conf. Intell. Robots Syst. (IROS), Oct. 2023, pp. 5211–5218.
[50] Q. Sun, J. Yuan, X. Zhang, and F. Duan, “Plane-edge-SLAM: Seamless
fusion of planes and edges for SLAM in indoor environments,” IEEE
Trans. Autom. Sci. Eng., vol. 18, no. 4, pp. 2061–2075, Oct. 2021.
[51] L. Xu, H. Yin, T. Shi, D. Jiang, and B. Huang, “EPLF-VINS: Real-time
monocular visual-inertial SLAM with eﬃcient point-line ﬂow features,”
IEEE Robot. Autom. Lett., vol. 8, no. 2, pp. 752–759, Feb. 2023.
[52] F. Shu, J. Wang, A. Pagani, and D. Stricker, “Structure PLP-SLAM:
Eﬃcient sparse mapping and localization using point, line and plane
for monocular, RGB-D and stereo cameras,” in Proc. IEEE Int. Conf.
Robot. Autom. (ICRA), May 2023, pp. 2105–2112.
[53] W. Zhao, H. Sun, X. Zhang, and Y. Xiong, “Visual SLAM combining
lines and structural regularities: Towards robust localization,” IEEE
Trans. Intell. Vehicles, vol. 9, no. 6, pp. 5047–5064, Jun. 2024.
[54] Q. Chen et al., “VPL-SLAM: A vertical line supported point line
monocular SLAM system,” IEEE Trans. Intell. Transp. Syst., vol. 25,
no. 8, pp. 9749–9761, Aug. 2024.
[55] H. Jiang, R. Qian, L. Du, J. Pu, and J. Feng, “UL-SLAM: A
universal monocular line-based SLAM via unifying structural and non-
structural constraints,” IEEE Trans. Autom. Sci. Eng., vol. 22, no. 3,
pp. 2682–2699, Jun. 2025.
[56] Y. Wang, Y. Tian, J. Chen, C. Chen, K. Xu, and X. Ding, “MSSD-
SLAM: Multifeature semantic RGB-D inertial SLAM with structural
regularity for dynamic environments,” IEEE Trans. Instrum. Meas.,
vol. 74, pp. 1–17, 2025.
[57] H. Li, J. Yao, J.-C. Bazin, X. Lu, Y. Xing, and K. Liu, “A monocular
SLAM system leveraging structural regularity in Manhattan world,” in
Proc. IEEE Int. Conf. Robot. Autom. (ICRA), May 2018, pp. 2518–2525.
[58] J. Liu and Z. Meng, “Visual SLAM with drift-free rotation estima-
tion in Manhattan world,” IEEE Robot. Autom. Lett., vol. 5, no. 4,
pp. 6512–6519, Oct. 2020.
[59] R. Yunus, Y. Li, and F. Tombari, “ManhattanSLAM: Robust planar
tracking and mapping leveraging mixture of Manhattan frames,” in Proc.
IEEE Int. Conf. Robot. Autom. (ICRA), May 2021, pp. 6687–6693.
[60] X. Peng, Z. Liu, Q. Wang, Y.-T. Kim, and H.-S. Lee, “Accurate visual-
inertial SLAM by Manhattan frame re-identiﬁcation,” in Proc. IEEE/RSJ
Int. Conf. Intell. Robots Syst. (IROS), Sep. 2021, pp. 5418–5424.
[61] Y. Li, R. Yunus, N. Brasch, N. Navab, and F. Tombari, “RGB-D SLAM
with structural regularities,” in Proc. IEEE Int. Conf. Robot. Autom.
(ICRA), May 2021, pp. 11581–11587.
[62] X. Li, W. Wang, J. Chen, and X. Zhang, “DR-SLAM: Drift rejection
SLAM with Manhattan regularity for indoor environments,” Adv. Robot.,
vol. 36, no. 20, pp. 1049–1059, Oct. 2022.
[63] X. Zhang, S. Li, Q. Liu, P. Liu, and Q. Wang, “A visual SLAM approach
to deeply explore the spatial constraints in indoor environment based on
the Manhattan hypothesis,” J. Sensors, vol. 2023, no. 1, Jan. 2023, Art.
no. 4152171.
[64] E. Jeong, J. Lee, S. Kang, and P. Kim, “Linear four-point LiDAR SLAM
for Manhattan world environments,” IEEE Robot. Autom. Lett., vol. 8,
no. 11, pp. 7392–7399, Nov. 2023.
[65] C. Akinlar and C. Topal, “EDLines: A real-time line segment detector
with a false detection control,” Pattern Recognit. Lett., vol. 32, no. 13,
pp. 1633–1642, Oct. 2011.
[66] L. Zhang and R. Koch, “An eﬃcient and robust line segment matching
approach based on LBD descriptor and pairwise geometric consistency,”
J. Vis. Commun. Image Represent., vol. 24, no. 7, pp. 794–805, Oct.
2013.
[67] K. Levenberg, “A method for the solution of certain non-linear problems
in least squares,” Quart. J. Appl. Math., vol. 2, no. 2, pp. 164–168, Jul.
1944.
[68] D. W. Marquardt, “An algorithm for least-squares estimation of nonlin-
ear parameters,” J. Soc. Ind. Appl. Math., vol. 11, no. 2, pp. 431–441,
Jun. 1963.
[69] W. Yifan, F. Serena, S. Wu, C. ¨Oztireli, and O. Sorkine-Hornung,
“Diﬀerentiable surface splatting for point-based geometry processing,”
ACM Trans. Graph., vol. 38, no. 6, pp. 1–14, Dec. 2019.
[70] G. Kopanas, J. Philip, T. Leimk¨uhler, and G. Drettakis, “Point-based
neural rendering with per-view optimization,” in Computer Graphics
Forum. Hoboken, NJ, USA: Wiley, 2021, pp. 29–43.
[71] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A density-based algo-
rithm for discovering clusters in large spatial databases with noise,” in
Proc. 2nd Int. Conf. Knowl. Discovery Data Mining, 1996, pp. 226–231.
[72] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “PointNet++: Deep hierarchical
feature learning on point sets in a metric space,” in Proc. Adv. Neural
Inf. Process. Syst., Jan. 2017, pp. 5099–5108.
[73] J. Straub et al., “The replica dataset: A digital replica of indoor spaces,”
2019, arXiv:1906.05797.
[74] L. Koestler, N. Yang, N. Zeller, and D. Cremers, “TANDEM: Tracking
and dense mapping in real-time using deep multi-view stereo,” in Proc.
Conf. Robot Learn., 2022, pp. 34–45.
[75] M. Savva et al., “Habitat: A platform for embodied AI research,” in
Proc. IEEE/CVF Int. Conf. Comput. Vis., Oct. 2019, pp. 9339–9347.
[76] A. Dai, M. Nießner, M. Zollh¨ofer, S. Izadi, and C. Theobalt,
“BundleFusion: Real-time globally consistent 3D reconstruction using
on-the-ﬂy surface reintegration,” ACM Trans. Graph., vol. 36, no. 4,
p. 1, Aug. 2017.
[77] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. M. Montiel, and
J. D. Tard´os, “ORB-SLAM3: An accurate open-source library for visual,
visual–inertial, and multimap SLAM,” IEEE Trans. Robot., vol. 37,
no. 6, pp. 1874–1890, Dec. 2021.
[78] G. Schindler and F. Dellaert, “Atlanta world: An expectation maximiza-
tion framework for simultaneous low-level edge grouping and camera
calibration in complex man-made environments,” in Proc. IEEE Comput.
Soc. Conf. Comput. Vis. Pattern Recognit., CVPR., vol. 1, Jun. 2004,
pp. 203–209.
[79] H. Li et al., “Hong Kong world: Leveraging structural regularity for
line-based SLAM,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45,
no. 11, pp. 13035–13053, Nov. 2023.
[80] J. Ham, M. Kim, S. Kang, K. Joo, H. Li, and P. Kim, “San Francisco
world: Leveraging structural regularities of slope for 3-DoF visual
compass,” IEEE Robot. Autom. Lett., vol. 10, no. 1, pp. 382–389, Jan.
2025.
[81] S. Hong, J. He, X. Zheng, and C. Zheng, “LIV-GaussMap: LiDAR-
inertial-visual fusion for real-time 3D radiance ﬁeld map rendering,”
IEEE Robot. Autom. Lett., vol. 9, no. 11, pp. 9765–9772, Nov. 2024.
Shuhong Liu received the bachelor’s degree from
the Department of Electrical and Computer Engi-
neering, University of Waterloo, ON, Canada, and
the master’s degree in creative informatics from the
Information Science and Technology, The Univer-
sity of Tokyo, Tokyo, Japan, where he is currently
pursuing the Ph.D. degree with the Department of
Mechano-Informatics. His research interests include
3D computer vision, visual SLAM, and computation
photography.
Tianchen Deng (Graduate Student Member, IEEE)
is currently pursuing the Ph.D. degree in control
science and engineering with Shanghai Jiao Tong
University, Shanghai, China. His main research
interests include 3D Reconstruction, long-termvisual
simultaneous localization and mapping (SLAM), and
vision-based localization.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 16 -->
LIU et al.: MG-SLAM: STRUCTURE GAUSSIAN SPLATTING SLAM WITH MANHATTAN WORLD HYPOTHESIS
17049
Heng Zhou is currently pursuing the master’s
degree in mechanical engineering with Columbia
University. He is also a member of the Creative
Machines Laboratory, led by Prof. Hod Lipson. His
research interests include SLAM and 3D reconstruc-
tion. His current work focuses primarily on the
integration of 3D Gaussian techniques with SLAM
applications.
Liuzhuozheng Li is currently pursuing the MS.C.
degree in complexity science and engineering with
The University of Tokyo, Tokyo, Japan. His main
research interests include the deep generative model
and computer vision.
Hongyu Wang (Member, IEEE) received the B.S.
degree in electronic engineering from Jilin Univer-
sity of Technology, Changchun, China, in 1990,
the M.S. degree in electronic engineering from the
Graduate School, Chinese Academy of Sciences,
Beijing, China, in 1993, and the Ph.D. degree in
precision instrument and optoelectronics engineering
from Tianjin University, Tianjin, China, in 1997.
He is currently a Professor with Dalian University
of Technology, Dalian, China. His research inter-
ests include image processing, computer vision, 3-D
reconstruction, and simultaneous localization and mapping (SLAM).
Danwei Wang (Life Fellow, IEEE) received the B.E.
degree from South China University of Technology,
China, in 1982, and the M.S.E. and Ph.D. degrees
from the University of Michigan, Ann Arbor, MI,
USA, in 1984 and 1989, respectively. He is a fellow
of the Academy of Engineering Singapore. He was
a recipient of the Alexander von Humboldt Fellow-
ship, Germany. He served as the general chairperson,
the technical chairperson, and various positions for
several international conferences. He was an invited
guest editor of various international journals. He is
a Distinguished Lecturer of the IEEE Robotics and Automation Society.
Mingrui
Li
(Student Member, IEEE) is cur-
rently pursuing the Ph.D. degree with the School
of Information and Communication Engineering,
Dalian University of Technology, Dalian, China.
His research interests include 3D reconstruction,
simultaneous localization and mapping (SLAM), and
computer vision.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:54 UTC from IEEE Xplore.  Restrictions apply.
