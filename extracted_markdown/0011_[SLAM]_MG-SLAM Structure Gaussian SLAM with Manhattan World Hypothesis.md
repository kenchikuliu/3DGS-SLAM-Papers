<!-- page 1 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
1
MG-SLAM: Structure Gaussian Splatting SLAM with Manhattan
World Hypothesis
Shuhong Liu∗, Tianchen Deng∗, Heng Zhou, Liuzhuozheng Li, Hongyu Wang Member, IEEE,
Danwei Wang Fellow, IEEE, Mingrui Li†
Abstract—Gaussian Splatting SLAMs have made significant
advancements in improving the efficiency and fidelity of real-
time reconstructions. However, these systems often encounter
incomplete reconstructions in complex indoor environments,
characterized by substantial holes due to unobserved geometry
caused by obstacles or limited view angles. To address this
challenge, we present Manhattan Gaussian SLAM, an RGB-D
system that leverages the Manhattan World hypothesis to
enhance geometric accuracy and completeness. By seamlessly
integrating fused line segments derived from structured scenes,
our method ensures robust tracking in textureless indoor areas.
Moreover, The extracted lines and planar surface assumption
allow strategic interpolation of new Gaussians in regions of
missing geometry, enabling efficient scene completion. Extensive
experiments conducted on both synthetic and real-world scenes
demonstrate that these advancements enable our method to
achieve state-of-the-art performance, marking a substantial
improvement in the capabilities of Gaussian SLAM systems.
Note to Practitioners—This paper was motivated by the limi-
tations of Gaussian Splatting SLAM systems in complex indoor
environments, where textureless surfaces and obstructed views
often lead to substantial tracking errors and incomplete maps.
While existing systems excel in high-fidelity reconstruction, they
struggle with frame-to-frame or point-feature tracking in large-
scale environments, particularly with significant camera rotations
and obscured structures. In this paper, we enhance the neural
dense SLAM by integrating the Manhattan World hypothesis,
applying its parallel line and planar surface constraints for
more robust tracking and mapping. We incorporate line segment
features into both tracking and mapping to improve structural
accuracy. Moreover, we propose a post-optimization method
that interpolates new Gaussian primitives to effectively fill gaps
on planar surfaces. Extensive experiments on multiple datasets
demonstrate the superiority of our approach in large-scale indoor
environments, resulting in more accurate tracking and mapping.
Index Terms—SLAM, 3DGS, Manhattan World.
∗These authors contributed equally to this work.
† Corresponding author: mmclmr@mail.dlut.edu.cn
Shuhong Liu and Liuzhuozheng Li are with Department of Information
Science and Technology and Department of Complexity Science and Engi-
neering, The University of Tokyo, Tokyo 113-8654, Japan. Tianchen Deng is
with Institute of Medical Robotics and Department of Automation, Shanghai
Jiao Tong University, and Key Laboratory of System Control and Information
Processing, Ministry of Education, Shanghai 200240, China. Heng Zhou
is with Department of Mechanical Engineering, Columbia University, New
York 10027, United States. Danwei Wang is with School of Electrical
and Electronic Engineering, Nanyang Technological University, Singapore.
Hongyu Wang and Mingrui Li are with Department of Computer Science,
Dalian University of Technology, Dalian 116024, China. This research is
supported by the National Research Foundation, Singapore, under the NRF
Medium Sized Centre scheme (CARTIN), ASTAR under National Robotics
Programme with Grant No. M22NBK0109, and by the National Research
Foundation, Singapore. Any opinions, findings and conclusions or recommen-
dations expressed in this material are those of the authors and do not reflect
the views of National Research Foundation.
Fig. 1.
Visualization of MG-SLAM on scene0000 00 and scene0207 00
of the ScanNet dataset [29]. Our method leverages robust line segments to
achieve superior camera pose estimation and scene reconstruction results.
Moreover, by applying structural surface constraints, we enhance and complete
the planar surfaces of the scene through the insertion of new Gaussian
primitives to fill gaps.
I. INTRODUCTION
Simultaneous Localization and Mapping (SLAM) is a
fundamental problem in computer vision that aims to map
an environment while simultaneously tracking the camera
pose. Learning-based dense SLAM methods, particularly neu-
ral radiance field (NeRF) approaches [1]–[9], have demon-
strated remarkable improvements in capturing dense photomet-
ric information and providing accurate global reconstruction
over traditional systems based on sparse point clouds [10]–
[16]. However, NeRF methods still face drawbacks such
as over-smoothing, bounded scene representation, and com-
putational inefficiencies [17]–[20]. Recently, Gaussian-based
SLAM [21]–[27] has emerged as a promising approach uti-
lizing volumetric Gaussian primitives [28]. Leveraging these
explicit representations, Gaussian SLAMs deliver high-fidelity
rendering and fine-grained scene reconstruction, overcoming
the limitations of NeRF-based methods.
Despite their strengths, Gaussian SLAM faces notable chal-
lenges in indoor scenes, which are often characterized by tex-
tureless surfaces and complex spatial layouts. These environ-
ments hinder robust tracking due to a lack of sufficient texture
details critical for camera pose optimization. Moreover, the
complex geometry of indoor scenes often leads to substantial
unobserved areas due to occlusions or limited view coverage.
arXiv:2405.20031v4  [cs.RO]  11 Jan 2026

<!-- page 2 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
2
These unseen regions pose a critical yet largely unexplored
challenge for Gaussian SLAM, as the Gaussian representation
can hardly interpolate the unobserved geometry without multi-
view optimization. Consequently, substantial holes and gaps
are left in the unseen areas of the map, an issue that has been
largely overlooked in previous Gaussian SLAM studies.
To overcome these challenges, we leverage the renowned
Manhattan World (MW) hypothesis [30] as a foundational
strategy for refining and completing scene geometries. This
assumption posits that the built environment predominantly
adheres to a grid-like structure, with surfaces and lines align-
ing with three orthogonal directions. These lines and planar
surfaces impose meaningful constraints on the tracking and
mapping processes in the Gaussian SLAM system.
Specifically, we encompass enhancements in tracking, map-
ping, and planar surface completion. In tracking, we utilize
fused line features derived from the structured scenes as robust
feature foundations in textureless areas, backprojecting and
reprojecting these line segments for pose optimization and
full bundle adjustment. In mapping, we apply a photometric
loss for the reprojected line features to refine the map. This
approach ensures that the reconstructed scene adheres closely
to the true structure of the environment, thereby improving
both its geometry accuracy and rendering quality.
Furthermore, the MW hypothesis facilitates the identifica-
tion and interpolation of structured surfaces, such as floors and
ceilings. These planar surfaces are critical to defining the over-
all geometry of indoor scenes but are often partially obscured
or missing from the captured views [31], [32]. By segmenting
these incomplete surfaces—refined by the extracted lines as
boundaries—we can predict their continuation beyond the
directly observed portions by generating new Gaussians. This
strategy enables us to optimize the representation of large
surfaces within the scene, enhancing the completeness of the
rendered map. Finally, we compress the Gaussian representa-
tion into mesh surfaces by incorporating regularization terms
through Poisson reconstruction [33]. This approach enables
the extraction of high-quality mesh, previously unavailable in
the Gaussian SLAM systems, making it readily available for
downstream tasks.
Overall, our work presents the following key contributions:
• We propose MG-SLAM, a novel RGB-D Gaussian
SLAM system that capitalizes on the MW hypothesis.
This assumption introduces lines and planar surfaces for
robust tracking, map refinement, and surface completion
for neural-dense SLAM systems.
• We incorporate line segments along with an additional
fusion and filtering strategy into the neural-dense SLAM
system, effectively improving its tracking capabilities
in textureless indoor environments and enhancing the
quality of the dense Gaussian map.
• We establish hypothesis surfaces using extracted line
segments that represent planar boundaries. These surfaces
guide our efficient interpolation of new Gaussians to
fill gaps and holes in the reconstructed map, seamlessly
addressing areas where current Gaussian SLAM systems
face limitations due to unobserved geometry.
• Extensive experiments conducted on both large-scale
synthetic and real-world datasets demonstrate that our
system offers state-of-the-art (SOTA) tracking and com-
prehensive map reconstruction, achieving 50% lower ATE
and 5dB enhancement in PSNR on large-scale Apartment
dataset, meanwhile operating at a high frame rate.
II. RELATED WORK
A. Neural Dense SLAM
Neural Implicit SLAM systems [1], [3], [4], [6], [34]–
[38] leveraging NeRF [39] are adept at handling complex re-
construction using implicit volumetric representation. Despite
these advancements, NeRF methods often struggle with the
over-smoothing issue, where fine-grained object-level geome-
try and features are difficult to capture during reconstruction
[24]. Moreover, these methods suffer from catastrophic loss as
the scenes are implicitly represented by MLPs.
In contrast, the high-fidelity and fast rasterization capabili-
ties of 3D Gaussian Splatting [28] enable higher quality and
efficiency on scene reconstruction [21], [22], [24], [25], [40]–
[48]. MonoGS [22] utilizes a map-centric SLAM approach
that employs 3D Gaussian representation for dynamic and
high-fidelity scene capture. SplaTAM [21] adopts an explicit
volumetric approach using isotropic Gaussians, enabling pre-
cise camera tracking and map densification. Photo-SLAM [25]
and RTG-SLAM [41] integrate the traditional feature-based
tracking system [49], [50] with Gaussian mapping, providing
robust tracking and exceptional real-time processing capabili-
ties. However, existing Gaussian SLAM systems lack effective
camera pose optimization in textureless environments, limiting
tracking accuracy in indoor scenes. Moreover, they struggle
to effectively reconstruct unobserved areas, often resulting in
incomplete reconstructions with gaps and holes. This limita-
tion becomes more problematic in settings where the camera’s
movement is restricted, leading to significant unmodeled areas
in structured indoor scenes. Additionally, current Gaussian-
based SLAM approaches face challenges in direct mesh gen-
eration due to the discrete nature of 3D Gaussian primitives,
which complicates surface extraction. To address this, recent
Gaussian-based systems [23], [40], [44] apply TSDF fusion
[51] on rendered images to produce meshes. However, this ap-
proach is limited to observed viewpoints, leaving unobserved
regions incomplete. More critically, it fundamentally depends
on the offline volumetric TSDF projection using rendered
images, which is independent of the reconstructed Gaussian
maps. To overcome these challenges, our method incorporates
line segments and planar surface assumption to seamlessly fill
the gaps of structured surfaces and directly extract high-quality
mesh from volumetric Gaussian representations.
B. SLAM with Structure Optimization
Line features are known to significantly enhance camera
pose optimization by capturing high-level geometric elements
and structural properties [52], [53]. Traditional SLAM systems
[14], [54]–[60] combine point features, line segments, or
planar surfaces to refine camera pose estimation and improve

<!-- page 3 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
3
Fig. 2.
The two-phase pipeline illustration of our proposed MG-SLAM. The upper section visualizes the parallel processes of the tracking and mapping
systems. The lower section presents the post-optimization of scene interpolation and mesh extraction.
map reconstruction. These approaches are particularly effec-
tive in dynamic or textureless environments, where point-based
methods often face substantial challenges.
Building on the MW hypothesis, recent advancements in
SLAM systems leveraging planar constraints have further
mitigated tracking and mapping drift. For instance, [61]
demonstrates accurate camera pose estimation and sparse 3D
map generation in monocular SLAM. [62] achieves drift-free
rotational motion estimation by leveraging structural regulari-
ties captured by line features. ManhattanSLAM [63] extends
this concept by optimizing camera trajectories and gener-
ating sparse maps with points, lines, and planes, alongside
dense surfel-based reconstructions. Similarly, [64] incorporates
Manhattan frame re-identification to build robust rotational
constraints, which are tightly integrated into a bundle adjust-
ment framework. Planar-SLAM [65] focuses on planar mesh
reconstruction by utilizing line and sparse point features, while
[66] introduces drift-free rotation estimation through the use
of Gaussian spheres. Moreover, [67] refines spatial constraints
through advanced bundle adjustment leveraging structure con-
straints. [68] integrates low-cost LiDAR in structured scenes
for better view coverage.
Despite these advancements, most existing systems lever-
aging structure or planar constraints rely on sparse map
representations, such as point clouds or simple planar surfels,
which are insufficient for reconstructing fine-grained maps
with detailed textures. Additionally, the sparse map leads
to discontinuities in the reconstructed map, making surface
extraction and subsequent optimization difficult. To bridge this
gap, we incorporate line features into the neural dense system,
enhancing its tracking capabilities in indoor environments and
enabling structured surface optimization.
III. METHOD
Figure 2 illustrates the pipeline of our proposed method.
Under the constraints of the Manhattan World, MG-SLAM
introduces line segments and structured surfaces to enhance
camera pose estimation and map reconstruction. Section III-B
details the tracking mechanism that utilizes both point and line
features. We utilize a specific strategy for fusing line segments
to ensure reliable identification of line features. Section III-C
discusses the Gaussian representation, including a specialized
loss term dedicated to the reconstruction of line segments.
Section III-D describes the completion and refinement of the
scene, grounded in the assumption of structured surfaces.
Section III-E describes the mesh generation utilizing regular-
ization losses.
A. Notation
We define the 2D image domain as P
∈R2, which
encompasses appearance color information C ∈N3, semantic
color S ∈N3, and depth data D ∈R+. Transitioning to the
3D world domain, denoted as X ∈R3, we introduce a camera
projection function π : X →P, mapping a 3D point Xi to its
2D counterpart Pi, and conversely, a backprojection function
θ : P →X, for the reverse mapping. (R, t) ∈SE (3) defines
the camera pose. I is the identity matrix.
B. Tracking
We utilize the backprojection of point features and line
segments extracted from 2D images into 3D space in par-
allel for optimizing the camera pose, based on PLVS [14].
Moreover, we propose strategies for line segment fusion and
suppression, designed to merge shorter segments into longer
ones and eliminate unstable lines. This method is grounded in
the understanding that longer line features tend to offer greater
reliability and robustness throughout the tracking process.

<!-- page 4 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
4
1) Point Reprojection Error: Given a point observation pi
within a specific keyframe k, its reprojection error can be
computed as:
Lp
2D = ∥Pi −π(RkX w
i + Tk)∥ΣPi ∈R
(1)
where X w
i
is the 3D point back-projected to the world do-
main and transformed to the camera frame X c
i . The term
ΣPi = σ2PiI2 represents the covariance matrix, encapsulating
the variance σ2Pi of point detection noise at different scales
within a Gaussian image pyramid [14].
2) Line Reprojection and Backprojection Error: For line
segments, represented in the 3D world as a pair of endpoints
(Xp, Xq) ∈R3 and in the image plane as (Pp, Pq) ∈R2,
we identify the line segments on each level of the image
pyramid using the EDlines method [69] for efficient extraction.
Matching line segments leverage the Line Band Descriptor
(LBD) method [70], and matches are scored according to the
Hamming distances between descriptors.
Upon observing a line segment l
=
(Pp, Pq)k, its
back-projected 3D endpoints can be computed as L
=
(θ(Pp), θ(Pq))k. Then the reprojection error of the line seg-
ment can be interpreted as:
Ll
2D =
dk · π(RkX w
p + t)

dk · π(RkX w
q + t)


Σ2D
∈R2 ,
(2)
dk =
¯
Pp × ¯
Pq
 ¯
Pp × ¯
Pq
|k
(3)
where ¯
Pp = [PT
p , 1]T , ¯
Pq = [PT
q , 1]T . The covariance matrix
Σ2D = σ2
diI2 is assumed to be diagonal for simplicity. Further,
we calculate the perpendicular distance in 3D between a map
point X w
i ∈R3 and a 3D line segment L as follows:
el
3D(L, X c
i )k = ∥(X c
i −θ(Pp)) × (X c
i −θ(Pq))∥k
∥θ(Pp) −θ(Pq)∥k
(4)
which considers the cross-product of the differences between
the point’s position and the back-projected positions of the
line’s endpoints, normalized by the distance between these
endpoints. Additionally, the distance between the world do-
main point X w
i
and its corresponding back-projected point in
the image plane Pi ∈R2 is determined by:
el
P(Pi, X c
i ) = ∥X c
i −θ(Pi)∥k ,
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
tionally aligned within a one-degree angle difference, ensuring
they follow the same path. (2) Ensuring these segments are
close—within a 10-pixel distance of each other’s nearest
endpoints—yet do not overlap, preserving their distinctness
while allowing for precise merging. (3) Verifying the vertical
distance between the corresponding endpoints of one segment
to the entirety of another is less than certain pixel threshold,
maintaining geometric consistency. Only segments fulfilling
all three criteria are merged, producing longer and more
reliable lines. Furthermore, we filter out segments that fall
below a predefined length threshold, relative to the image size,
to improve the system’s tracking robustness.
4) Full Bundle Adjustment Error: The overall objective
function for the full bundle adjustment is given by:
LBA =
X
K
X
U
ρ

∥Lp
2D∥ΣP
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
Here, K represents the set of chosen keyframes, while U and V
denote the sets of points and lines extracted within the current
frame. The Huber cost function ρ(∥e∥)Σ−1) = eT Σ−1e is
applied to mitigate the influence of outliers. The optimization
process utilizes the Levenberg-Marquardt method [71], [72],
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
identify the surface such as floors for structure optimization
and surface extraction, explained in Section III-D.
1) Scene Representation: We use the standard point ren-
dering formula [73], [74] to splat G to render 2D image:
µP = π(RµG + T), ΣP = JRΣGRT JT
(9)
where µG and ΣG are mean and covariance of the Gaussian
primitives. J is the Jacobian of the linear approximation of
π(·). For each pixel pix ∈P, the influence of N Gaussians
on this pixel can be combined by sorting the Gaussians in
depth order and performing front-to-back alpha-blending:
Ppix =
X
i∈N
fiαi
i−1
Y
j=1
(1 −αj), Dpix =
X
i∈N
ziαi
i−1
Y
j=1
(1 −αj)
(10)
where Ppix and Dpix represent the pixel-wise appearance
(Cpix for color and Spix for semantic features) and depth
respectively. z is the distance to the mean of the Gaussian
G along the camera ray.
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

<!-- page 5 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
5
In this context, Gaussian primitives are optimized to adjust
their optical and geometrical parameters to closely replicate
the observed scene with fine-grained details. Furthermore, to
impose constraints on the scene utilizing the identified line
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
In our optimization process, we focus particularly on refin-
ing the representation of expansive surfaces that formulate the
indoor scene, such as the floor and ceiling. By applying the
MW assumption, we introduce the planar hypothesis surfaces
that are informed by the structural regularity of the extracted
3D line features. These constraints are employed to identify
surfaces that are not adequately represented and to interpolate
new Gaussians for densifying or filling in gaps with textures
that are reasonably predicted.
1) Gaussian Density: In our Gaussian map, we define the
density of the Gaussian ν : R3 →R+ as the sum of the
Gaussian values weighted by their alpha-blending coefficients
at any given grid points p as:
ν(p) =
X
Gi
αGiexp(−1
2(p −µT
Gi)Σ−1
Gi (p −µGi) ,
(13)
where the µGi, ΣGi, αGi are the means, covariances, and alpha-
blending coefficients of the Gaussians G. To simplify the
calculation, ν(p) can be approximated following [33] as:
ν∗(p) = αG∗exp(−1
2(p −µT
G∗)Σ−1
G∗(p −µG∗) ,
(14)
where G∗is the nearest Gaussian that contributes most to p.
This Gaussian density function ν∗(·) facilitates the subsequent
identification of under-represented regions on the planar sur-
faces and the mesh generation process.
2) Map Calibration: Since the SLAM system relies on the
initial camera pose as a reference frame, the reconstructed
scenes usually do not satisfy the orthogonality assumption of
the Manhattan World. To overcome this issue, we calibrate the
reconstructed scene by applying a calibrating matrix K to the
coordinates and covariance of the Gaussians. This matrix K
is derived from clustering the directions of 3D line segments
that are presumed to align with the scene axes.
After aligning the structured surface boundaries to orthogo-
nal directions, they extend across the xy →R2 plane. We
use the calibrated line segments to outline the rectangular
boundary of the hypothesized planes, capitalizing on the
dense line features commonly found at the corners of scenes.
Algorithm 1 presents the detailed implementation.
Algorithm 1 The pseudo-code for calibration and identifica-
tion of surface boundary
Require: Line segments L ∈R3 and the reconstructed Gaussian map G
1: Lfiltered ←select |ℓ| > Tℓ
▷Filter out small line segments
2: for each line segment ℓin Lfiltered do
3:
{direction} ←(ℓ.end −ℓ.start).normalized()
▷Add direction vector
4: end for
5: ⃗x, ⃗y, ⃗z ←k means({direction}, k = 3)
▷Find axes directions
6: K ←I ∈R4
▷Define calibration matrix
7: K[0 : 3, 0] ←⃗x
8: K[0 : 3, 1] ←⃗y
9: K[0 : 3, 2] ←⃗z
10: for each Gi in G do
11:
µGi ←K · µGi
▷Calibrate coordinates of Gaussian
12:
γGi ←K · γGi
▷Calibrate rotation of Gaussian
13: end for
14: for each line segment ℓin L do
15:
ℓ←K · ℓ
▷Calibrate line segment
16: end for
17: B ←select maximum and minimum x and y for endpoints in L
▷Define surface boundary
18: Output: boundary of surface B ∈R2
Algorithm 2 The pseudo-code for scene completion
Require: The calibrated Gaussian scene G, the mask M extracted from
semantic segmentation, and the boundary of surface B, threshold of
density Tdensity.
1: Mtg ←semantic masks that represent target surfaces of the scene, e.g.
floor
2: for each structure surface Gi in G(Mtg) do
3:
Gi ←DBSCAN.fit(Gi)
▷Apply DBSCAN [75] to eliminate outliers
4:
¯z ←avg(z) for z ∈µGi
5:
Π ←grid(B)
▷Create the 2D grid of hypothesis surface
6:
Di ←compute the Gaussian density for each point of the grid in Π.
7:
if Di ≤Tdensity then
8:
µpred ←generates new Gaussians centers
9:
end if
10:
B∗
Gi ←ramdomly sample Gaussians from G to form training batch.
11:
for each batch Bi in B∗
Gi do
12:
F(µGi|Gi ∈Bi) ←train PointNet++ [76] model on Bi
13:
end for
14:
Cpred ←F(µpred)
▷Predict the color of new Gaussians
15:
Gpred ←formulate new Gaussians using {µpred} and {Cpred}
16: end for
17: G ←Gpred
▷Update the overall scene
18: Output: structure optimized G
3) Surface Interpolation: Algorithm 2 shows our surface
interpolation strategy. Specifically, we utilize the semantic
information, incorporated in Equation (11), for identifying the
planar surfaces such as floors and ceilings. The chosen target
Gaussians are then projected from the 3D space onto the 2D
hypothesis xy plane. Subsequently, we apply a density thresh-
old to the sampling density function, defined in Equation (14),
to detect potential holes or gaps on the surface. New Gaussian
primitives are generated at the identified gaps that fall below a
density threshold. Finally, we train a PointNet++ model [76],
fitting it to the presented surface to learn the presented color
patterns and interpolate the texture color of the new Gaussians
based on their spatial correlation. We empirically found that
the PointNet++ network, which incorporates spatial correla-
tion, is sufficiently effective in predicting the color patterns
of new primitives on the gaps of structured surfaces. Given
that these planar surfaces typically exhibit less texture than
specific objects, employing more sophisticated models does

<!-- page 6 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
6
not provide substantial benefits but increases computational
expenses.
Through this method, we aim to seamlessly combine the
detailed representation provided by Gaussians with the need
for explicit scene geometry that accounts for unseen views,
thereby achieving a more comprehensive and accurate recon-
struction of planar surfaces in indoor environments.
E. Mesh Generation
We employed the surface extraction method proposed
by [33] with a novel normal regularization term during
the scene refinement. To compress the Gaussians to align
closely with the surface, [33] forces the Gaussians to be
flat by setting one of their scaling factors nearly to zero.
Consequently, ν∗(p) can be further simplified to ˆν∗(p) =
αG∗exp
 −2s−2
G∗⟨p −µG∗, nG∗⟩2
where sG represents the
smallest scaling factor and nG its corresponding normal. Here,
αG∗= 1 is set to avoid semi-transparent Gaussians.
Subsequently, a regularization term is used to align the SDF
function derived from the density function with its estimation
from the scene as:
Lsdf =
1
|Xp|
X
x∈Xp
|sdf(x) −ˆ
sdf(x)|
(15)
Here,
Xp
is
the
sampled
3D
points
and
sdf(x)
=
±sG∗
p
−2 log(ˆν∗(x)) represents the ideal SDF. The estimated
SDF ˆ
sdf(x) is determined by subtracting the depth of x from
the corresponding rendered depth map. This regularization
term encourages closer alignment of the estimated surface with
the observed depth.
Moreover, to derive smooth surfaces, particularly in flat
and textureless floor areas, we adopt the MW hypothesis,
which assumes that planar floors are orthogonal to the vertical
axis. We enforce this constraint by introducing a normal
regularization term for the estimated SDF:
Lnormal =
1
|Xfloor|
X
x2D∈Pfloor
1 −ˆn ·
∇f(x)
∥∇f(x)∥

(16)
In this context, the projected points x2D, which pass through
the camera ray and fall within the floor area, are encouraged
to align their normal to the ideal vector ˆn = ⟨0, 0, 1⟩. The
set Pfloor consists of image pixels identified as the floor using
semantic segmentation.
IV. EXPERIMENT
A. Experiment Settings
Implementation Details Our tracking system is implemented
in C++, while the mapping system uses Python3 and CUDA
C. The experiments were conducted using a single RTX A100-
80GB GPU and 24-core AMD EPYC 7402 processor. We
use Adam optimizer for Gaussian representation optimization
and network training. The hyperparameters for 3D Gaussian
splitting [28] are the same as the original paper. Table I
presents the values for hyperparameters in our system. For
Gaussian insertion, in each keyframe, we sampled µG in
Equation (9) from the rendered depth map Dpix following
TABLE I
MG-SLAM HYPERPARAMETERS
Symbol
Explanation
Value
Np
number of point features in tracking
4e3
Nℓ
number of line segments in tracking
200
rd
downsample ratio for Gaussian initialization
8
nmap
mapping iteration
150
SHdegree
degree of the spherical harmonics
0
lfeat
learning rate for SH features
2.5e-3
lopacity
learning rate for opacity
0.05
lscale
learning rate for scaling
1e-3
lrotat
learning rate for rotation
1e-3
λD
weighting term for depth loss in mapping
0.10
λS
weighting term for semantic loss in mapping
0.10
λℓ
re-weighting term for line feature loss
0.25
Tℓ
line segment filtering threshold
0.08
Tdensity
density threshold for hole detection
0.90
bG
batch size for PointNet++ [76] training
1e4
ne
number of epoch to train PointNet++ [76]
25
the distribution of N(Dpix, 0.2σD). For uninitialized areas,
we initialize Gaussians by sampling from N( ¯Dpix, 0.5σD)
where ¯Dpix is the mean value. For pruning, we remove the
Gaussians with opacity less than 0.6. Our system incorporates
a segmentation loss introduced in [24], and we leverage
this semantic information to effectively extract structured
surfaces, such as floors and ceilings, from the reconstructed
scenes. We subsequently train a PointNet++ [76] network for
each structure surface and interpolate the missing geometry
by predicting the color of inserted Gaussians.
Datasets We evaluate our method using two datasets: Replica
[77], a synthetic dataset, and ScanNet [29], a challenging
real-world dataset. The large-scale Replica Apartment dataset
used in our experiments was released by Tandem [78]. For
the Replica dataset [77], the ground-truth camera pose and
semantic maps are obtained through Habitat simulation [79].
In the case of the ScanNet dataset [29], the ground-truth
camera poses are derived using BundleFusion [80]. Moreover,
we further validate our method on a long-trajectory dataset
collected using our physical platform [81].
Metrics To assess the quality of the reconstruction, we employ
metrics such as PSNR, SSIM, and LPIPS. For evaluating
camera pose, we use the average absolute trajectory error
(ATE RMSE). The real-time processing capability, essential
for SLAM systems, is measured in frames per second (FPS).
Best results are shaded as first , second , and third .
Baselines We evaluate our tracking and mapping results
against state-of-the-art methods, including NeRF-based ap-
proaches such as NICE-SLAM [1], Co-SLAM [3], ESLAM
[4], and Point-SLAM [6], as well as recent Gaussian-based
methods including SplaTAM [21], MonoGS [22], Photo-
SLAM [25], and RTG-SLAM [41]. The results for MonoGS
[22] were obtained using its RGB-D mode.
B. Evaluation on Replica-V1 Dataset
In line with conventional evaluations from previous stud-
ies, we performed quantitative comparisons of training view

<!-- page 7 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
7
TABLE II
QUANTITATIVE COMPARISON OF OUR METHOD AND THE BASELINES IN TRAINING VIEW RENDERING ON THE REPLICA-V1 DATASET [77].
Methods
Metrics
Avg.
Room0
Room1
Room2
Office0
Office1
Office2
Office3
Office4
PSNR↑
30.54
28.88
28.51
29.37
35.44
34.63
26.56
28.79
32.16
NeRF-SLAM
Co-SLAM [3]
SSIM↑
0.850
0.892
0.843
0.851
0.854
0.826
0.814
0.866
0.856
LPIPS↓
0.188
0.213
0.205
0.215
0.177
0.181
0.172
0.163
0.176
PSNR↑
29.08
25.32
27.77
29.08
33.71
30.20
28.09
28.77
29.71
ESLAM [4]
SSIM↑
0.929
0.875
0.902
0.932
0.960
0.923
0.943
0.948
0.945
LPIPS↓
0.336
0.313
0.298
0.248
0.184
0.228
0.241
0.196
0.204
PSNR↑
35.17
32.40
34.08
35.50
38.26
39.16
33.99
33.48
33.49
Point-SLAM [6]
SSIM↑
0.975
0.974
0.975
0.980
0.983
0.986
0.960
0.960
0.979
LPIPS↓
0.124
0.113
0.116
0.111
0.100
0.118
0.156
0.132
0.142
PSNR↑
33.98
32.48
33.72
34.96
38.34
39.04
31.90
29.70
31.68
SplaTAM [21]
SSIM↑
0.969
0.975
0.970
0.982
0.982
0.982
0.965
0.950
0.946
LPIPS↓
0.099
0.072
0.096
0.074
0.083
0.093
0.100
0.118
0.155
PSNR↑
35.68
33.78
34.32
36.56
39.14
39.83
34.47
33.25
34.08
Gaussian-SLAM
MonoGS [22]
SSIM↑
0.962
0.954
0.957
0.963
0.972
0.976
0.962
0.960
0.950
LPIPS↓
0.087
0.071
0.086
0.075
0.074
0.087
0.098
0.098
0.105
PSNR↑
34.95
30.71
33.51
35.02
38.47
39.08
33.03
33.78
36.02
Photo-SLAM [25]
SSIM↑
0.942
0.899
0.934
0.951
0.964
0.961
0.938
0.938
0.952
LPIPS↓
0.059
0.075
0.057
0.043
0.050
0.047
0.077
0.066
0.054
PSNR↑
35.43
31.56
34.21
35.57
39.11
40.27
33.54
32.76
36.48
RTG-SLAM [41]
SSIM↑
0.982
0.967
0.979
0.981
0.990
0.992
0.981
0.981
0.984
LPIPS↓
0.109
0.131
0.105
0.115
0.068
0.075
0.134
0.128
0.117
PSNR↑
36.90
34.67
35.52
37.10
40.04
41.38
35.91
34.85
35.75
Ours
SSIM↑
0.981
0.976
0.978
0.980
0.987
0.988
0.980
0.977
0.978
LPIPS↓
0.086
0.070
0.084
0.070
0.076
0.083
0.101
0.095
0.112
TABLE III
SYSTEM COMPARISON IN TERMS OF TRACKING, MAPPING, RENDERING
FPS, AND MEMORY USAGE BETWEEN OUR METHOD AND THE NEURAL
DENSE BASELINES ON THE REPLICA DATASET [77]. THE VALUES
REPRESENT THE AVERAGE OUTCOMES ACROSS 8 SCENES. NOTE THAT
RTG-SLAM [41] AND PHOTO-SLAM [25] INCORPORATE ORB-SLAM2
[49] AND ORB-SLAM3 [82] FOR TRACKING.
Methods
Track.
ATE
[cm]↓
Track.
FPS
[f/s]↑
Map.
FPS
[f/s]↑
Render.
FPS
[f/s]↑
SLAM
FPS
[f/s]↑
Param.
Size
[mb]↓
NeRF
Co-SLAM [3]
1.12
10.2
10.0
35
9.26
8.2
ESLAM [4]
0.63
9.92
2.23
2.5
1.82
73.7
Point-SLAM [6]
0.54
0.42
0.06
1.3
0.05
69.6
SplaTAM [21]
0.55
0.35
0.22
122
0.14
277.1
3DGS
MonoGS [22]
0.58
2.57
3.85
642
1.80
42.2
Photo-SLAM [25]
0.59
38.72
30.04
765
16.46
59
RTG-SLAM [41]
0.49
45.36
18.67
692
12.65
71
Ours
0.45
35.32
3.96
324
3.76
78.5
rendering on the Replica-V1 dataset [77], which comprises
8 single-room scenes. As presented in Table II, our method
demonstrates superior rendering quality. This improvement
stems from the use of a denser primitive distribution, as
adopted in MonoGS [22], leading to higher PSNR values,
and the integration of the segment reconstruction loss, which
contributes to a competitive structural similarity score.
Additionally, Table III provides comparisons of tracking
accuracy and system efficiency. By leveraging line features,
our approach achieves state-of-the-art tracking performance
compared to baseline methods, further contributing to optimal
map reconstruction quality. Regarding system efficiency, we
assessed the frame rate of each component and memory usage.
Unlike systems employing frame-to-frame tracking [3], [4],
[6], [21], [22], which exhibit significantly low tracking speeds,
methods such as Photo-SLAM [25], RTG-SLAM [41], and
TABLE IV
QUANTITATIVE COMPARISON OF OUR METHOD AND THE BASELINE
APPROACHES IN TRAINING VIEW RENDERING ON THE REPLICA
APARTMENT DATASET [77]. THE UNDERLINE INDICATES THAT
RELOCALIZATION WAS TRIGGERED DUE TO ACCUMULATED TRACKING
ERRORS. THE DASH INDICATES SYSTEM FAILURES.
Methods
Metrics Avg.
ap.0
ap.1
ap.2
frl.0
frl.4
PSNR↑24.82 29.10 22.86 23.29 23.52 25.33
NeRF-SLAM
Co-SLAM [3]
SSIM↑0.816 0.905 0.766 0.771 0.822 0.814
LPIPS↓0.410 0.321 0.440 0.462 0.367 0.461
PSNR↑24.05 25.34 21.75 22.64 24.63 25.90
ESLAM [4]
SSIM↑0.818 0.866 0.752 0.794 0.837 0.842
LPIPS↓0.350 0.375 0.392 0.351 0.327 0.305
PSNR↑34.28 34.95 32.27 33.31 36.01 34.87
Point-SLAM [6]
SSIM↑0.955 0.972 0.929 0.944 0.960 0.970
LPIPS↓0.180 0.153 0.205 0.211 0.156 0.176
PSNR↑25.16 13.12 24.57 25.52 30.72 31.86
SplaTAM [21]
SSIM↑0.790 0.415 0.821 0.858 0.924 0.930
LPIPS↓0.323 0.656 0.302 0.258 0.201 0.198
PSNR↑26.87 21.89 26.87 27.92 30.70 26.97
Gaussian-SLAM
MonoGS [22]
SSIM↑0.868 0.864 0.856 0.873 0.886 0.863
LPIPS↓0.285 0.397 0.285 0.273 0.225 0.247
PSNR↑28.78 29.07 22.73 24.59 34.16 33.36
Photo-SLAM [25]
SSIM↑0.888 0.922 0.796 0.848 0.940 0.932
LPIPS↓0.224 0.227 0.293 0.354 0.115 0.129
PSNR↑
-
-
29.08 29.14 33.88
-
RTG-SLAM [41]
SSIM↑
-
-
0.900 0.909 0.933
-
LPIPS↓
-
-
0.232 0.232 0.181
-
PSNR↑34.35 34.54 32.37 32.92 36.24 35.66
Ours
SSIM↑0.956 0.970 0.925 0.946 0.966 0.972
LPIPS↓0.223 0.269 0.255 0.244 0.168 0.180
ours achieve notable advantages in tracking performance with
exceptional frame rates. This is attributed to the incorporation
of traditional feature-based optimization, which avoids itera-
tive rendering and photometric loss computations for camera
pose optimization—a common bottleneck in neural dense
systems—thereby remarkably enhancing tracking speed. For

<!-- page 8 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
8
Fig. 3. The novel view sythesis of the scene apartment 0 from the Replica Apartment dataset [77]. The top left shows the line segments extracted in 3D
space. The bottom left illustrates the overall reconstructed scene.
mapping, compared to Photo-SLAM [25] and RTG-SLAM
[41], which utilize relatively sparse primitive representations,
our method opts for a denser map to improve reconstruction
quality. While this results in slower mapping speeds and
moderate memory usage, it offers a balanced trade-off for
higher reconstruction fidelity.
C. Evaluation on Replica-Apartment Dataset
To evaluate the robustness of the system in large-scale
indoor environments, we evaluate MG-SLAM on the Replica
Apartment dataset [77]. This dataset contains extensive multi-
room scenes, complex object geometries, and looping trajec-
tories across rooms. Table IV presents the rendering quality
of our method compared to both NeRF-based and Gaussian-
based approaches over five selected scenes. MG-SLAM shows
notable improvements, particularly achieving a 7dB improve-
ment in the apartment 0 scene over Gaussian SLAM systems.
This optimal performance is largely attributed to the inclusion
of the fused line segments, which lays a solid foundation
for loop closure and pose optimization. Compared to the
TABLE V
QUANTITATIVE COMPARISON OF OUR METHOD WITH TRADITIONAL,
NERF-BASED, AND GAUSSIAN-BASED RGB-D SLAM SYSTEMS IN
TERMS OF ATE [CM] ON THE REPLICA APARTMENT DATASET [77]. THE
UNDERLINED VALUES INDICATE THAT RELOCALIZATION WAS NECESSARY
DUE TO LOSS OF TRACKING.
Methods
Avg.
ap.0
ap.1
ap.2
frl.0
frl.4
Trad.
ORB-SLAM2 [49]
5.06
6.85
9.83
6.71
1.29
1.00
ORB-SLAM3 [82]
4.41
5.39
7.98
6.31
1.09
1.27
PLVS [14]
3.72
7.06
4.67
4.04
1.45
1.39
NeRF
Co-SLAM [3]
4.14
5.30
4.78
6.31
2.01
2.30
ESLAM [4]
5.54
9.22
7.09
7.01
3.33
1.07
Point-SLAM [6]
5.88
6.39
17.28
3.54
1.02
1.19
SplaTAM [21]
6.97
25.14
15.67
6.17
2.74
3.31
3DGS
MonoGS [82]
6.55
20.69
9.51
7.57
2.98
6.14
Photo-SLAM [25]
4.41
5.39
7.98
6.31
1.09
1.27
RTG-SLAM [41]
5.06
6.99
9.09
6.90
1.33
0.98
Ours
2.77
5.03
3.87
3.15
0.92
0.88
NeRF-based Point-SLAM [6], which utilizes additional neural
point clouds for robust tracking and fine-grained mapping, our
method delivers superior averaged reconstruction outcomes
while maintaining optimal real-time processing capabilities,

<!-- page 9 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
9
Fig. 4. Qualitative comparison of our method and the baselines for novel-view synthesis on the ScanNet dataset [29]. The outcomes show that our method
provides more robust and fine-grained reconstructions in real-world complex scenes compared to current NeRF-based and Gaussian-based approaches.
achieving a system frame rate 50 times faster.
Table V presents the quantitative evaluations of tracking
ATE [cm] on the apartment scenes. Our method achieves state-
of-the-art tracking accuracy compared to both frame-to-frame
and feature-based tracking systems. Specifically, ORB-based
systems [25], [41], [49], [82] struggle in textureless planar
environments, such as the stairs or corridors in apartmen 0
and apartment 1, leading to tracking failures, frequent relocal-
izations, and errors in the reconstructed scenes. Leveraging the
segment fusion strategy, which provides robust line features,
our method also outperforms PLVS [14] under scenarios
involving extensive camera movement and rotation which are
commonly encountered in the apartment dataset.
Moreover, we conducted qualitative comparisons on novel-
view synthesis against Gaussian-based approaches [21], [22],
[25] as shown in Figure 3. These comparisons highlight the
superior performance of our method over baseline approaches
in large-scale complex indoor environments.
D. Evaluation on ScanNet Dataset
We provide quantitative assessments of reconstruction qual-
ity using the ScanNet dataset [29] in Table VI. Our approach
delivers state-of-the-art results, outperforming other Gaussian-
based methods by a notable 3dB in PSNR in real-world
environments. The tracking evaluation results are shown in Ta-
ble VII. Our method remarkably reduces the ATE RMSE (cm)
error, achieving 20% improvements over baseline approaches.
In Figure 4, we visualize the novel-view synthesis results
for MG-SLAM, comparing it with both NeRF-based and
Gaussian-based SLAM systems. Our method exhibits robust
and high-fidelity reconstruction capabilities. In comparison
to Point-SLAM [6], our method offers complete scene re-
constructions and finer texture details, benefiting from the
TABLE VI
QUANTITATIVE COMPARISON OF OUR METHOD AND THE BASELINES IN
TRAINING VIEW RENDERING ON THE SCANNET DATASET [29].
Methods
Metrics Avg.
0000 0059 0106 0169 0207
PSNR↑19.26 20.47 18.56 18.47 18.71 20.12
NeRF-SLAM
Co-SLAM [3]
SSIM↑0.692 0.701 0.680 0.672 0.684 0.724
LPIPS↓0.573 0.559 0.581 0.585 0.578 0.564
PSNR↑16.27 16.67 15.34 16.52 15.51 17.32
ESLAM [4]
SSIM↑0.650 0.686 0.632 0.627 0.655 0.650
LPIPS↓0.492 0.452 0.452 0.528 0.492 0.538
PSNR↑18.61 20.12 18.58 16.47 18.23 19.66
Point-SLAM [6]
SSIM↑0.737 0.806 0.765 0.676 0.686 0.750
LPIPS↓0.523 0.485 0.499 0.544 0.542 0.544
PSNR↑19.02 17.81 19.60 19.23 20.55 17.95
SplaTAM [21]
SSIM↑0.726 0.602 0.796 0.741 0.785 0.705
LPIPS↓0.337 0.467 0.290 0.322 0.260 0.346
PSNR↑20.08 19.76 19.25 20.18 20.57 20.62
Gaussian-SLAM
MonoGS [22]
SSIM↑0.782 0.772 0.767 0.785 0.790 0.798
LPIPS↓0.300 0.387 0.289 0.272 0.256 0.295
PSNR↑20.76 21.74 20.07 20.70 20.34 20.94
Photo-SLAM [25]
SSIM↑0.790 0.771 0.805 0.792 0.792 0.790
LPIPS↓0.293 0.375 0.280 0.269 0.248 0.292
PSNR↑16.79 18.62 15.56 14.97 18.07 18.52
RTG-SLAM [41]
SSIM↑0.743 0.756 0.682 0.726 0.772 0.773
LPIPS↓0.480 0.468 0.531 0.480 0.451 0.459
PSNR↑23.71 25.69 23.62 22.95 22.86 23.42
Ours
SSIM↑0.838 0.846 0.838 0.829 0.836 0.840
LPIPS↓0.262 0.282 0.255 0.260 0.236 0.277
TABLE VII
QUANTITATIVE COMPARISON OF OUR METHOD AND THE NEURAL DENSE
BASELINES IN TERMS OF ATE [CM] ON THE SCANNET DATASET [29].
Methods
Avg.
0000
0059
0106
0169
0207
NeRF
Co-SLAM [3]
8.46
11.14
9.36
5.90
7.14
8.75
ESLAM [4]
7.68
8.47
8.70
7.58
7.45
6.20
Point-SLAM [6]
11.68
10.24
7.81
8.65
22.16
9.54
Gaussian
SplaTAM [21]
12.04
12.83
10.10
17.72
12.08
7.46
MonoGS [22]
13.86
15.94
13.03
19.44
10.44
10.46
Photo-SLAM [25]
8.40
7.62
7.76
9.36
10.01
7.23
RTG-SLAM [41]
8.70
8.04
6.82
9.22
10.15
9.25
Ours
6.77
5.95
6.41
8.07
7.29
6.14

<!-- page 10 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
10
Fig. 5. Qualitative comparison of our method and the baselines on the trajectory collected using our physical platform. The left side displays the line and point
features extracted by our tracking system. The right side presents the reconstruction comparisons with Gaussian-based methods, with MG-SLAM showing
more reliable reconstruction results in long-horizon and textureless indoor environments.
TABLE VIII
QUANTITATIVE COMPARISON OF OUR METHOD AND GAUSSIAN-BASED
APPROACHES IN TERMS OF TRACKING AND MAPPING PERFORMANCE ON
TRAJECTORY COLLECTED USING OUR PHYSICAL PLATFORM. THE DASH
INDICATES SYSTEM FAILURES.
Methods
ATE [cm]↓PSNR [dB]↑
SSIM↑
LPIPS↓
SplaTAM [21]
-
-
-
-
MonoGS [22]
27.04
18.28
0.751
0.677
Photo-SLAM [25]
18.29
22.92
0.798
0.620
Ours
7.72
25.47
0.831
0.575
use of Gaussians to handle complex geometries. Among
Gaussian-based methods, SplaTAM [21] struggles in large
scenes with complex camera loops due to its lack of bundle
adjustment in the tracking system. Similarly, MonoGS [22]
generally delivers reliable quality but struggles with object-
level reconstruction drift. Photo-SLAM [25] achieves better
reconstruction results compared to other baseline approaches
but suffers from floaters and artifacts that compromise its
geometric accuracy. In contrast, MG-SLAM excels with robust
tracking and effective bundle adjustment incorporating line
segments, enabling superior detailed reconstructions even sur-
passing the ground-truth mesh derived from offline methods.
E. Evaluation on Physical Platform
To assess our system’s performance in real-world envi-
ronments, we utilzie a private data trajectory gathered with
our physical platform. The robot was deployed in the NTU
building to navigate and collect data from a challenging long
corridor, notable for its lack of texture and extensive length
exceeding 200 meters. Ground-truth poses were computed
by registering the LiDAR point cloud with the scanned
point cloud generated by the Leica ScanStation. Figure 5
displays the comparisons of training view synthesis between
our method and Gaussian-based systems, where our method
demonstrates significant advantages; however, it still lacks
some details compared to the ground-truth images due to the
long-horizon trajectory. Table VIII provides quantitative re-
sults, showing that our method achieves more accurate tracking
and mapping performance. The baseline approaches struggle
with tracking in textureless environments, further affirming the
effectiveness of our method in indoor scenes.
F. Evaluation of Scene Completion
The Gaussian SLAM faces limitations in interpolating the
geometry of unseen views. This issue is especially evident
in indoor scenes that feature complex layouts, where basic
surfaces like floors are often obscured and poorly represented.
Figure 6 qualitatively compared our method with recent
Gaussian-based approaches in the Replica scenes [77] that
presented uncovered geometry in camera trajectories. Utilizing
the surface interpolation strategy based on the MW hypothesis
[30], our method can accurately detect and proactively gener-
ate new Gaussians efficiently to fill gaps with certain textures,
whereas Gaussian baseline methods exhibit substantial defects
on structured surfaces.
V. ABLATION STUDY
In this section, we provide a comprehensive ablation anal-
ysis of the hyperparameters and each component’s effect.
A. Ablation of Hyperparameters
Our tracking system, which advances the foundation of
PLVS system [14], utilizes feature points and fused line
segments for optimizing camera poses and performing bundle
adjustments. The upper row of Figure 7 illustrates the effects

<!-- page 11 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
11
Fig. 6. Qualitative comparison on the novel-view synthesis of our method and the Gaussian SLAM baselines for hole fillings on the Replica dataset [77].
Our method demonstrates superior capability in interpolating and filling gaps on structured planar surfaces, such as floors and ceilings.
Fig. 7.
The ablation study examining the impact of the number of points,
line segments, keyframe intervals, and downsample ratios in MG-SLAM on
scene0000 00 from the ScanNet dataset [29].
of varying the number of points and line segments on the
ATE loss [cm] and the frame rate of the tracking system. We
identified an optimal region where having too few points and
lines results in a lack of sufficient anchor features, whereas too
many points and lines can lead to notable feature mismatches
that reduce tracking accuracy. The tracking FPS reduces as a
tradeoff with the increase in the number of features. Compared
to tracking solely by point features, including line segment
features noticeably decreases the frame rate; however, it still
maintains a high rate over 30 FPS and does not become
the speed bottleneck of the overall systems. The bottom
row of Figure 7 illustrates the effects of default keyframe
intervals and downsample ratio on the Gaussian map. During
the mapping process, we uniformly downsample the point
cloud generated from RGB-D input by a specific ratio to
accelerate the system and conserve memory. We observed a
marginal PSNR disparity when maintaining a relatively small
downsample ratio.
B. Ablation of Line Segment Extraction
The extracted line segments play a crucial role in serving
as robust feature foundations in the subsequent optimiza-
tion processes. Figure 8 illustrates the ablation results for
our feature extraction approach for these 3D line segments,
demonstrating the efficacy of segment backprojection, line
fusion, and filtering strategies in providing clear and accurate
line features.
C. Ablation of Tracking and Mapping Loss
MG-SLAM employs line segments for robust camera pose
optimization and fine-grained map reconstruction. Specifically,

<!-- page 12 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
12
Fig. 8. The ablation study of the line segment extraction outcomes for our method on the scene frl apartment 4 from the Replica Apartment dataset [77]. By
incorporating backprojection, filtering, and segment fusion, our system generates rich and robust line segment features for subsequent tracking and mapping
procedures.
Fig. 9. The visualization of the ablation study in scene0000 00 from the ScanNet dataset [29]. From left to right, the results are displayed starting completely
without line features, adding line segments, integrating segment fusion, and finally incorporating line photometric loss in mapping. Key differences are
highlighted with colored boxes.
TABLE IX
QUANTITATIVE COMPARISON OF THE ABLATION STUDY ON LINE LOSSES,
MEASURING ATE [CM] AND PSNR [DB], CONDUCTED ON scene0000 00
OF THE SCANNET DATASET [29]. THE CHECKMARK SYMBOL INDICATES
THE EMPLOYMENT OF THE METHOD.
Methods
Metric
line
segment
segment
fusion
line
photo.
loss
ATE
[cm]
PSNR
[dB]
7.41
20.65
6.58
22.89
5.95
24.72
5.95
25.69
fused line segments are utilized in the bundle adjustment
of the tracking procedure, and a photometric loss related
to line features is integrated into the map optimization. We
use scene0000 00 from the ScanNet dataset [29] to conduct
the ablation study that evaluates the impact of each loss.
This scene was chosen because it contains rich line features,
which allow for a clear assessment of each component’s
effectiveness. Table IX presents the ATE [cm] and PSNR [dB]
in relation to the use of each loss. We observed that inte-
grating line features remarkably enhances tracking accuracy,
and the fusion of line segments, which facilitates robust edge
extraction, further improves the performance. For mapping,
the photometric loss provides additional geometric constraints,
thereby offering optimal reconstruction quality.
In Figure 9, we present a typical case in scene0000 00
Fig. 10. The ablation study compares surface identification approaches. The
identified hypothesis surfaces using different methods are shaded in black.
The left figure displays the scene without calibration, where the primary
axes of the scene do not align with the world coordinates. The middle figure
demonstrates the application of PCA to the centers of Gaussians µG associated
with structured surfaces, where the boundaries of µG define the surface. The
right figure shows the calibration results, achieved by clustering line segments
and defining surfaces based on the boundaries of these line features.
to demonstrate the effectiveness of each loss component.
This case features a bike that appears multiple times along
the camera trajectory. Compared to relying solely on point
features, adding fused line segments significantly improves
the quality of map reconstruction. This enhancement mitigates
scene drift by providing more accurate camera pose estimation.
Moreover, incorporating the photometric loss of line features
in mapping results in more detailed object-level geometry and
more precise reconstruction of line-like textures.

<!-- page 13 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
13
Fig. 11. The ablation of normal regularization for surface extraction on the Replica dataset [77]. We compare the normal map of the extracted mesh, both
without and with normal regularization, to the ground-truth mesh. The regularization refines the planar surfaces of the scene, i.e. floors, effectively.
D. Ablation of Surface Extraction
We employ hypothesis surfaces regularized by line segments
to pinpoint potential gaps on structured surfaces, as described
in Section III-D. These identified hypothesis surfaces are de-
signed to precisely mirror the positions of the planar surfaces
with minimal redundancy. To delineate these surfaces, we
initially adjust the reconstructed map to adhere to the orthogo-
nality assumption of MW. Figure 10 displays the hypothesized
rectangular planes produced without calibration, with PCA
calibration based on the centers of Gaussian primitives, and
with calibration using features from extracted segments. It is
evident that our segment-based method efficiently captures the
floor surface with reduced redundancy.
E. Ablation of Mesh Generation
Following map reconstruction and surface interpolation, we
extract meshes from the Gaussian map, as outlined in Sec-
tion III-E. Figure 11 illustrates the results of mesh extraction
with and without the proposed normal regularization loss. We
observe that incorperating this regularization term results in
smoother mesh extraction on the structured surface such as
floors compared to the original method introduced by [33].
VI. CONCLUSION
In this study, we present MG-SLAM, a Gaussian-based
SLAM method based on the MW hypothesis. MG-SLAM
employs fused line segments and point features for robust pose
estimation and map refinement. Furthermore, by leveraging the
line segments and planar surface assumption, we efficiently
generate new Gaussian primitives in gaps on structural sur-
faces caused by obstructions or unseen geometry. Extensive
experiments demonstrate that our method delivers state-of-the-
art tracking and mapping performance, while also maintaining
real-time processing speed.
VII. LIMITATIONS
MW assumes surface planes generally align with the three
orthogonal directions defined by the Cartesian coordinate sys-
tem. This presents challenges in refining planes or layouts that
are slanted or not strictly orthogonal. To better accommodate
diverse outdoor urban environments, future research could
incorporate multiple horizontal dominant directions, as seen in
Atlanta World [83], include additional sloping directions like
those in Hong Kong World [84], or adopt uniform inclination
angles for the dominant directions proposed in San Francisco
World [85]. Additionally, since the optimization of our dense
map primarily relies on line features and large planar surfaces,
it struggles with refining piece-wise or fine-grained structural
planes, such as walls and windows. One potential approach is
to leverage the coplanarity of surfaces for structural regulariza-
tion [65], [86], although this proves particularly challenging in
indoor environments with complex layouts where identifying
coplanar surfaces is difficult.
REFERENCES
[1] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and
M. Pollefeys, “Nice-slam: Neural implicit scalable encoding for slam,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 12 786–12 796.
[2] X. Kong, S. Liu, M. Taher, and A. J. Davison, “vmap: Vectorised
object mapping for neural field slam,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2023, pp. 952–
961.
[3] H. Wang, J. Wang, and L. Agapito, “Co-slam: Joint coordinate and
sparse parametric encodings for neural real-time slam,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 13 293–13 302.

<!-- page 14 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
14
[4] M. M. Johari, C. Carta, and F. Fleuret, “Eslam: Efficient dense slam
system based on hybrid representation of signed distance fields,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 17 408–17 419.
[5] Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and
M. Pollefeys, “Nicer-slam: Neural implicit scene encoding for rgb slam,”
in International Conference on 3D Vision (3DV), March 2024.
[6] E. Sandstr¨om, Y. Li, L. Van Gool, and M. R. Oswald, “Point-slam:
Dense neural point cloud-based slam,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 18 433–18 444.
[7] T. Deng, G. Shen, T. Qin, J. Wang, W. Zhao, J. Wang, D. Wang, and
W. Chen, “Plgslam: Progressive neural scene represenation with local to
global bundle adjustment,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 19 657–19 666.
[8] T. Deng, Y. Wang, H. Xie, H. Wang, R. Guo, J. Wang, D. Wang, and
W. Chen, “Neslam: Neural implicit mapping and self-supervised feature
tracking with depth completion and denoising,” IEEE Transactions on
Automation Science and Engineering, vol. 22, pp. 12 309–12 321, 2025.
[9] S. Liu, L. Gu, Z. Cui, X. Chu, and T. Harada, “I2-nerf: Learning
neural radiance fields under physically-grounded media interactions,” in
Advances in Neural Information Processing Systems (NeurIPS), 2025.
[10] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “Dtam: Dense
tracking and mapping in real-time,” in 2011 International Conference
on Computer Vision, 2011, pp. 2320–2327.
[11] R. F. Salas-Moreno, R. A. Newcombe, H. Strasdat, P. H. Kelly, and A. J.
Davison, “Slam++: Simultaneous localisation and mapping at the level
of objects,” in Proceedings of the IEEE conference on computer vision
and pattern recognition, 2013, pp. 1352–1359.
[12] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “Orb-slam: a versatile
and accurate monocular slam system,” IEEE transactions on robotics,
vol. 31, no. 5, pp. 1147–1163, 2015.
[13] H. Gao, X. Zhang, J. Wen, J. Yuan, and Y. Fang, “Autonomous indoor
exploration via polygon map construction and graph-based slam using
directional endpoint features,” IEEE Transactions on Automation Science
and Engineering, vol. 16, no. 4, pp. 1531–1542, 2019.
[14] L. Freda, “Plvs: A slam system with points, lines, volumetric mapping,
and 3d incremental segmentation,” arXiv preprint arXiv:2309.10896,
2023.
[15] Z. Wang, K. Yang, H. Shi, P. Li, F. Gao, J. Bai, and K. Wang, “Lf-vislam:
A slam framework for large field-of-view cameras with negative imaging
plane on mobile agents,” IEEE Transactions on Automation Science and
Engineering, vol. 21, no. 4, pp. 6321–6335, 2024.
[16] X. Lin, J. Ruan, Y. Yang, L. He, Y. Guan, and H. Zhang, “Robust
data association against detection deficiency for semantic slam,” IEEE
Transactions on Automation Science and Engineering, vol. 21, no. 1,
pp. 868–880, 2024.
[17] L. Xu, Y. Xiangli, S. Peng, X. Pan, N. Zhao, C. Theobalt, B. Dai,
and D. Lin, “Grid-guided neural radiance fields for large urban scenes,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 8296–8306.
[18] W. Yan, Y. Chen, W. Zhou, and R. Cong, “Mvoxti-dnerf: Explicit multi-
scale voxel interpolation and temporal encoding network for efficient dy-
namic neural radiance field,” IEEE Transactions on Automation Science
and Engineering, pp. 1–12, 2024.
[19] Z. Li, C. Wu, L. Zhang, and J. Zhu, “Dgnr: Density-guided neural point
rendering of large driving scenes,” IEEE Transactions on Automation
Science and Engineering, pp. 1–14, 2024.
[20] Y. Zhang, G. Chen, and S. Cui, “Efficient large-scale scene representa-
tion with a hybrid of high-resolution grid and plane features,” Pattern
Recognition, vol. 158, p. 111001, 2025.
[21] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer,
D. Ramanan, and J. Luiten, “Splatam: Splat, track & map 3d gaussians
for dense rgb-d slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024.
[22] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, “Gaussian splat-
ting slam,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024.
[23] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-slam:
Photo-realistic dense slam with gaussian splatting,” arXiv preprint
arXiv:2312.10070, 2023.
[24] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, “Sgs-
slam: Semantic gaussian splatting for neural dense slam,” in European
Conference on Computer Vision.
Springer, 2025, pp. 163–179.
[25] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-slam: Real-
time simultaneous localization and photorealistic mapping for monocular
stereo and rgb-d cameras,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 21 584–21 593.
[26] T. Deng, Y. Chen, L. Zhang, J. Yang, S. Yuan, D. Wang, and W. Chen,
“Compact 3d gaussian splatting for dense visual slam,” arXiv preprint
arXiv:2403.11247, 2024.
[27] M. Li, S. Liu, T. Deng, and H. Wang, “Densesplat: Densifying gaus-
sian splatting slam with neural radiance prior,” IEEE Transactions on
Visualization and Computer Graphics, pp. 1–14, 2025.
[28] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, pp. 1–14, 2023.
[29] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and
M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor
scenes,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2017, pp. 5828–5839.
[30] J. Coughlan and A. L. Yuille, “The manhattan world assumption: Reg-
ularities in scene statistics which enable bayesian inference,” Advances
in Neural Information Processing Systems, vol. 13, 2000.
[31] H. Guo, S. Peng, H. Lin, Q. Wang, G. Zhang, H. Bao, and X. Zhou,
“Neural 3d scene reconstruction with the manhattan-world assumption,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 5511–5520.
[32] X. Zhou, H. Guo, S. Peng, Y. Xiao, H. Lin, Q. Wang, G. Zhang,
and H. Bao, “Neural 3d scene reconstruction with indoor planar pri-
ors,” IEEE Transactions on Pattern Analysis and Machine Intelligence,
vol. 46, no. 9, pp. 6355–6366, 2024.
[33] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting
for efficient 3d mesh reconstruction and high-quality mesh rendering,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024.
[34] M. Li, J. He, Y. Wang, and H. Wang, “End-to-end rgb-d slam with
multi-mlps dense neural implicit representations,” IEEE Robotics and
Automation Letters, vol. 8, no. 11, pp. 7138–7145, 2023.
[35] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, “Go-slam: Global
optimization for consistent 3d instant reconstruction,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 3727–3737.
[36] W. Zhang, T. Sun, S. Wang, Q. Cheng, and N. Haala, “Hi-slam:
Monocular real-time dense mapping with hybrid implicit fields,” IEEE
Robotics and Automation Letters, 2023.
[37] L. Liso, E. Sandstr¨om, V. Yugay, L. Van Gool, and M. R. Oswald,
“Loopy-slam: Dense neural slam with loop closures,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 20 363–20 373.
[38] H. Zhou, Z. Guo, Y. Ren, S. Liu, L. Zhang, K. Zhang, and M. Li,
“Mod-slam: Monocular dense mapping for unbounded 3d scene recon-
struction,” IEEE Robotics and Automation Letters, vol. 10, no. 1, pp.
484–491, 2025.
[39] J. McCormac, R. Clark, M. Bloesch, A. Davison, and S. Leutenegger,
“Fusion++: Volumetric object-level slam,” in 2018 International Con-
ference on 3D Vision (3DV).
IEEE, 2018, pp. 32–41.
[40] C. Yan, D. Qu, D. Wang, D. Xu, Z. Wang, B. Zhao, and X. Li, “Gs-
slam: Dense visual slam with 3d gaussian splatting,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 19 595–19 604.
[41] Z. Peng, T. Shao, Y. Liu, J. Zhou, Y. Yang, J. Wang, and K. Zhou, “Rtg-
slam: Real-time 3d reconstruction at scale using gaussian splatting,” in
ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.
[42] F. Tosi, Y. Zhang, Z. Gong, E. Sandstr¨om, S. Mattoccia, M. R. Oswald,
and M. Poggi, “How nerfs and 3d gaussian splatting are reshaping slam:
a survey,” arXiv preprint arXiv:2402.13255, 2024.
[43] M. Li, J. Huang, L. Sun, A. X. Tian, T. Deng, and H. Wang, “Ngm-slam:
Gaussian splatting slam with radiance field submap,” arXiv preprint
arXiv:2405.05702, 2024.
[44] L. Zhu, Y. Li, E. Sandstr¨om, S. Huang, K. Schindler, and I. Armeni,
“Loopsplat: Loop closure by registering 3d gaussian splats,” in 2024
International Conference on 3D Vision (3DV).
IEEE, 2024.
[45] S. Liu, X. Chen, H. Chen, Q. Xu, and M. Li, “Deraings: Gaussian
splatting for enhanced scene reconstruction in rainy environments,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39,
no. 5, 2025, pp. 5558–5566.
[46] S. Liu, C. Bao, Z. Cui, Y. Liu, X. Chu, L. Gu, M. V. Conde, R. Umagami,
T. Hashimoto, Z. Hu et al., “Realx3d: A physically-degraded 3d
benchmark for multi-view visual restoration and reconstruction,” arXiv
preprint arXiv:2512.23437, 2025.
[47] S. Ha, J. Yeon, and H. Yu, “Rgbd gs-icp slam,” in European Conference
on Computer Vision.
Springer, 2025, pp. 180–197.

<!-- page 15 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
15
[48] J. Hu, X. Chen, B. Feng, G. Li, L. Yang, H. Bao, G. Zhang, and
Z. Cui, “Cg-slam: Efficient dense rgb-d slam in a consistent uncertainty-
aware 3d gaussian field,” in European Conference on Computer Vision.
Springer, 2025, pp. 93–112.
[49] R. Mur-Artal and J. D. Tard´os, “Orb-slam2: An open-source slam
system for monocular, stereo, and rgb-d cameras,” IEEE transactions
on robotics, vol. 33, no. 5, pp. 1255–1262, 2017.
[50] C.-M. Chung, Y.-C. Tseng, Y.-C. Hsu, X.-Q. Shi, Y.-H. Hua, J.-F. Yeh,
W.-C. Chen, Y.-T. Chen, and W. H. Hsu, “Orbeez-slam: A real-time
monocular visual slam with orb features and nerf-realized mapping,”
in 2023 IEEE International Conference on Robotics and Automation
(ICRA).
IEEE, 2023, pp. 9400–9406.
[51] B. Curless and M. Levoy, “A volumetric method for building complex
models from range images,” in Proceedings of the 23rd annual confer-
ence on Computer graphics and interactive techniques, 1996, pp. 303–
312.
[52] L. Zhou, G. Huang, Y. Mao, S. Wang, and M. Kaess, “Edplvo: Efficient
direct point-line visual odometry,” in 2022 International Conference on
Robotics and Automation (ICRA).
IEEE, 2022, pp. 7559–7565.
[53] Z. Xu, H. Wei, F. Tang, Y. Zhang, Y. Wu, G. Ma, S. Wu, and X. Jin,
“Plpl-vio: a novel probabilistic line measurement model for point-
line-based visual-inertial odometry,” in 2023 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS).
IEEE, 2023,
pp. 5211–5218.
[54] Q. Sun, J. Yuan, X. Zhang, and F. Duan, “Plane-edge-slam: Seamless
fusion of planes and edges for slam in indoor environments,” IEEE
Transactions on Automation Science and Engineering, vol. 18, no. 4,
pp. 2061–2075, 2021.
[55] L. Xu, H. Yin, T. Shi, D. Jiang, and B. Huang, “Eplf-vins: Real-time
monocular visual-inertial slam with efficient point-line flow features,”
IEEE Robotics and Automation Letters, vol. 8, no. 2, pp. 752–759, 2022.
[56] F. Shu, J. Wang, A. Pagani, and D. Stricker, “Structure plp-slam:
Efficient sparse mapping and localization using point, line and plane
for monocular, rgb-d and stereo cameras,” in 2023 IEEE International
Conference on Robotics and Automation (ICRA).
IEEE, 2023, pp.
2105–2112.
[57] W. Zhao, H. Sun, X. Zhang, and Y. Xiong, “Visual slam combining
lines and structural regularities: Towards robust localization,” IEEE
Transactions on Intelligent Vehicles, 2023.
[58] Q. Chen, Y. Cao, J. Hou, G. Li, S. Qiu, B. Chen, X. Xue, H. Lu, and
J. Pu, “Vpl-slam: A vertical line supported point line monocular slam
system,” IEEE Transactions on Intelligent Transportation Systems, 2024.
[59] H. Jiang, R. Qian, L. Du, J. Pu, and J. Feng, “Ul-slam: A universal
monocular line-based slam via unifying structural and non-structural
constraints,” IEEE Transactions on Automation Science and Engineer-
ing, pp. 1–18, 2024.
[60] Y. Wang, Y. Tian, J. Chen, C. Chen, K. Xu, and X. Ding, “Mssd-
slam: Multifeature semantic rgb-d inertial slam with structural regularity
for dynamic environments,” IEEE Transactions on Instrumentation and
Measurement, vol. 74, pp. 1–17, 2025.
[61] H. Li, J. Yao, J.-C. Bazin, X. Lu, Y. Xing, and K. Liu, “A monoc-
ular slam system leveraging structural regularity in manhattan world,”
in 2018 IEEE International Conference on Robotics and Automation
(ICRA).
IEEE, 2018, pp. 2518–2525.
[62] J. Liu and Z. Meng, “Visual slam with drift-free rotation estimation in
manhattan world,” IEEE Robotics and Automation Letters, vol. 5, no. 4,
pp. 6512–6519, 2020.
[63] R. Yunus, Y. Li, and F. Tombari, “Manhattanslam: Robust planar
tracking and mapping leveraging mixture of manhattan frames,” in 2021
IEEE International Conference on Robotics and Automation (ICRA).
IEEE, 2021, pp. 6687–6693.
[64] X. Peng, Z. Liu, Q. Wang, Y.-T. Kim, and H.-S. Lee, “Accurate visual-
inertial slam by manhattan frame re-identification,” in 2021 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS).
IEEE, 2021, pp. 5418–5424.
[65] Y. Li, R. Yunus, N. Brasch, N. Navab, and F. Tombari, “Rgb-d slam
with structural regularities,” in 2021 IEEE international conference on
Robotics and automation (ICRA).
IEEE, 2021, pp. 11 581–11 587.
[66] X. Li, W. Wang, J. Chen, and X. Zhang, “Dr-slam: drift rejection slam
with manhattan regularity for indoor environments,” Advanced Robotics,
vol. 36, no. 20, pp. 1049–1059, 2022.
[67] X. Zhang, S. Li, Q. Liu, P. Liu, and Q. Wang, “A visual slam approach
to deeply explore the spatial constraints in indoor environment based
on the manhattan hypothesis,” Journal of Sensors, vol. 2023, no. 1, p.
4152171, 2023.
[68] E. Jeong, J. Lee, S. Kang, and P. Kim, “Linear four-point lidar slam for
manhattan world environments,” IEEE Robotics and Automation Letters,
2023.
[69] C. Akinlar and C. Topal, “Edlines: A real-time line segment detector
with a false detection control,” Pattern Recognition Letters, vol. 32,
no. 13, pp. 1633–1642, 2011.
[70] L. Zhang and R. Koch, “An efficient and robust line segment matching
approach based on lbd descriptor and pairwise geometric consistency,”
Journal of visual communication and image representation, vol. 24,
no. 7, pp. 794–805, 2013.
[71] K. Levenberg, “A method for the solution of certain non-linear problems
in least squares,” Quarterly of applied mathematics, vol. 2, no. 2, pp.
164–168, 1944.
[72] D. W. Marquardt, “An algorithm for least-squares estimation of non-
linear parameters,” Journal of the society for Industrial and Applied
Mathematics, vol. 11, no. 2, pp. 431–441, 1963.
[73] W. Yifan, F. Serena, S. Wu, C. ¨Oztireli, and O. Sorkine-Hornung,
“Differentiable surface splatting for point-based geometry processing,”
ACM Transactions on Graphics (TOG), vol. 38, no. 6, pp. 1–14, 2019.
[74] G. Kopanas, J. Philip, T. Leimk¨uhler, and G. Drettakis, “Point-based
neural rendering with per-view optimization,” in Computer Graphics
Forum, vol. 40.
Wiley Online Library, 2021, pp. 29–43.
[75] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A density-based al-
gorithm for discovering clusters in large spatial databases with noise,”
in Proceedings of the Second International Conference on Knowledge
Discovery and Data Mining, 1996, p. 226–231.
[76] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical
feature learning on point sets in a metric space,” Advances in neural
information processing systems, vol. 30, 2017.
[77] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel,
R. Mur-Artal, C. Ren, S. Verma et al., “The replica dataset: A digital
replica of indoor spaces,” arXiv preprint arXiv:1906.05797, 2019.
[78] L. Koestler, N. Yang, N. Zeller, and D. Cremers, “Tandem: Tracking and
dense mapping in real-time using deep multi-view stereo,” in Conference
on Robot Learning.
PMLR, 2022, pp. 34–45.
[79] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain,
J. Straub, J. Liu, V. Koltun, J. Malik et al., “Habitat: A platform for
embodied ai research,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2019, pp. 9339–9347.
[80] A. Dai, M. Nießner, M. Zollh¨ofer, S. Izadi, and C. Theobalt, “Bundle-
fusion: Real-time globally consistent 3d reconstruction using on-the-fly
surface reintegration,” ACM Transactions on Graphics (ToG), vol. 36,
no. 4, p. 1, 2017.
[81] T. Deng, G. Shen, X. Chen, S. Yuan, H. Shen, G. Peng, Z. Wu, J. Wang,
L. Xie, D. Wang et al., “Mcn-slam: Multi-agent collaborative neural slam
with hybrid implicit neural scene representation,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2025.
[82] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and J. D.
Tard´os, “Orb-slam3: An accurate open-source library for visual, visual–
inertial, and multimap slam,” IEEE Transactions on Robotics, vol. 37,
no. 6, pp. 1874–1890, 2021.
[83] G. Schindler and F. Dellaert, “Atlanta world: An expectation maximiza-
tion framework for simultaneous low-level edge grouping and camera
calibration in complex man-made environments,” in Proceedings of
the 2004 IEEE Computer Society Conference on Computer Vision and
Pattern Recognition, 2004. CVPR 2004., vol. 1.
IEEE, 2004, pp. I–I.
[84] H. Li, J. Zhao, J.-C. Bazin, P. Kim, K. Joo, Z. Zhao, and Y.-H.
Liu, “Hong kong world: Leveraging structural regularity for line-based
slam,” IEEE Transactions on Pattern Analysis and Machine Intelligence,
vol. 45, no. 11, pp. 13 035–13 053, 2023.
[85] J. Ham, M. Kim, S. Kang, K. Joo, H. Li, and P. Kim, “San francisco
world: Leveraging structural regularities of slope for 3-dof visual com-
pass,” IEEE Robotics and Automation Letters, 2024.
[86] S. Hong, J. He, X. Zheng, C. Zheng, and S. Shen, “Liv-gaussmap: Lidar-
inertial-visual fusion for real-time 3d radiance field map rendering,”
IEEE Robotics and Automation Letters, 2024.
VIII. BIOGRAPHY SECTION

<!-- page 16 -->
IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING
16
Shuhong Liu is currently pursuing the Ph.D. degree
in Department of Mechano-informatics, Information
Science and Technology with the University of
Tokyo, Tokyo, Japan. Before that, he received his
bachelor’s degree in Department of Electrical and
Computer Engineering at University of Waterloo,
Ontario, Canada, and the master’s degree in Creative
Informatics, Information Science and Technology,
the University of Tokyo, Tokyo, Japan. His research
interests include 3D computer vision, visual SLAM,
and computation photography.
Tianchen Deng is currently pursuing the Ph.D. de-
gree in control science and engineering with Shang-
hai Jiao Tong University, Shanghai, China. His main
research interests include 3D Reconstruction, long-
term visual simultaneous localization and mapping
(SLAM), and vision-based localization.
Heng Zhou is currently a master student in Me-
chanical Engineering at Columbia University. He
is also a member of the Creative Machines Lab,
led by Professor Hod Lipson. His research interests
include SLAM and 3D reconstruction. His current
work focuses primarily on the integration of 3D
Gaussian techniques with SLAM applications.
Liuzhuozheng Li is currently pursuing the MS.C.
degree in Complexity Science and Engineering at
The University of Tokyo, Tokyo, Japan. His main
research interests include the deep generative model
and computer vision.
Hongyu Wang (Member, IEEE) received his B.S.
degree in electronic engineering from the Jilin Uni-
versity of Technology, Changchun, China, in 1990,
the M.S. degree in electronic engineering from the
Graduate School, Chinese Academy of Sciences,
Beijing, China, in 1993, and the Ph.D. degree in
precision instrument and optoelectronics engineering
from Tianjin University, Tianjin, China, in 1997. He
is currently a Professor with the Dalian University
of Technology, Dalian, China. His research interests
include image processing, computer vision, 3-D re-
construction, and simultaneous localization and mapping (SLAM).
Danwei Wang (Life Fellow, IEEE) received the
B.E. degree from the South China University of
Technology, China, in 1982, and the M.S.E. and
Ph.D. degrees from the University of Michigan, Ann
Arbor, MI, USA, in 1984 and 1989, respectively.
He is a fellow of the Academy of Engineering
Singapore. He was a recipient of the Alexander von
Humboldt Fellowship, Germany. He served as the
general chairperson, the technical chairperson, and
various positions for several international confer-
ences. He was an invited guest editor of various
international journals. He is a Distinguished Lecturer of the IEEE Robotics
and Automation Society.
Mingrui Li is currently pursuing the Ph.D. degree
with the School of Information and Communica-
tion Engineering, Dalian University of Technology,
Dalian, China. His research interests include 3D re-
construction, simultaneous localization and mapping
(SLAM), and computer vision.
