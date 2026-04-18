<!-- page 1 -->
1
GSM-GS: Geometry-Constrained Single and Multi-view Gaussian
Splatting for Surface Reconstruction
Xiao Ren, Yu Liu, Ning An, Jian Cheng, Xin Qiao, and He Kong
Abstract—Recently, 3D Gaussian Splatting has emerged as a
prominent research direction owing to its ultrarapid training
speed and high-fidelity rendering capabilities. However, the
unstructured and irregular nature of Gaussian point clouds poses
challenges to reconstruction accuracy. This limitation frequently
causes high-frequency detail loss in complex surface microstruc-
tures when relying solely on routine strategies. To address
this limitation, we propose GSM-GS: a synergistic optimization
framework integrating single-view adaptive sub-region weighting
constraints and multi-view spatial structure refinement. For
single-view optimization, we leverage image gradient features to
partition scenes into texture-rich and texture-less sub-regions.
The reconstruction quality is enhanced through adaptive filtering
mechanisms guided by depth discrepancy features. This preserves
high-weight regions while implementing a dual-branch constraint
strategy tailored to regional texture variations, thereby improving
geometric detail characterization. For multi-view optimization,
we introduce a geometry-guided cross-view point cloud as-
sociation method combined with a dynamic weight sampling
strategy. This constructs 3D structural normal constraints across
adjacent point cloud frames, effectively reinforcing multi-view
consistency and reconstruction fidelity. Extensive experiments
on public datasets demonstrate that our method achieves both
competitive rendering quality and geometric reconstruction. See
our interactive project page.
Index Terms—Gaussian Splatting, Surface Reconstruction, Ge-
ometry Texture, Multi-view Consistency.
I. INTRODUCTION
New View Synthesis (NVS) and geometric surface recon-
struction have broad applications spanning robotic environ-
mental perception [1]–[4] and navigation [5]–[8], AR/VR sys-
tems [9], 3D content generation/editing [10], and autonomous
driving [11], [12]. Conventional multi-view stereo (MVS)
methods typically estimate depth information through cross-
view feature matching, followed by depth fusion to reconstruct
object surfaces. While these methods achieve satisfactory
performance in many scenarios, they demand significant com-
putational resources and struggle to accurately capture surface
texture details in complex scenes while ensuring multi-view
consistency.
Among existing NVS and 3D reconstruction-related meth-
ods, Neural Radiance Field (NeRF) [13] implicitly represents
the 3D scene and achieves new view rendering through an
Xiao Ren, Yu Liu, and He Kong (corresponding author) are with
the School of Automation and Intelligent Manufacturing, Southern Uni-
versity
Science
and
Technology,
Shenzhen
518055,
China;
Emails:
[12431359,12250044]@mail.sustech.edu.cn; kongh@sustech.edu.cn. Ning An
and Jian Cheng are with the Research Institute of Mine Artificial Intel-
ligence, China Coal Research Institute, and the State Key Laboratory of
Intelligent Coal Mining and Strata Control, Beijing 100031, China; Emails:
ning.an.010@foxmail.com; jiancheng@tsinghua.org.cn. Xin Qiao is with the
Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University,
710049, Shaanxi, China; Email: wudiqx@xjtu.edu.cn.
analysis-by-synthesis approach. As discussed in the existing
literature [14], due to constraints imposed by implicit repre-
sentations, the original NeRF method typically produces 3D
reconstructions with limited accuracy, and its ray-sampling
and volume-rendering paradigm requires training times of tens
of hours. Although recent works, including Instant-NGP [15],
reduce the training time to a few minutes, the reconstruction
performance still needs considerable improvement.
Recently, the 3D Gaussian Splatting (3DGS) [16] technique
has been widely used in the field of 3D reconstruction and new
view synthesis due to its high-quality rendering and ultrashort
training time. The technique utilizes a set of anisotropic Gaus-
sian ellipsoids to represent the 3D scene, where each Gaussian
ellipsoid contains physical attributes such as position, color,
opacity, and covariance parameters. Despite the advantages
of the 3DGS method, it still faces challenges in high-fidelity
reconstruction [17] and rendering [18].
For example, in surface texture-rich regions, the optimiza-
tion process based on photometric consistency constraints
is susceptible to interference from high-frequency texture
components, leading to ambiguous decoupling between normal
vector orientation and Gaussian ellipsoid anisotropy covari-
ance parameters. Moreover, in texture-less or uniformly shaded
regions, the lack of sufficient texture gradient information
makes normal vector estimation prone to local minima, result-
ing in over-smoothed geometric details and distorted surface
curvature. Across multiple views, the absence of cross-view
geometry-appearance constraints for discrete Gaussian prim-
itives leads to reconstruction artifacts and geometric distor-
tion [19] in novel view synthesis.
In this paper, we propose a novel surface reconstruction and
novel view rendering framework that improves reconstruction
and rendering quality through single-view and multi-view
geometry-guided constraints. On the one hand, for single-view
optimization, we first achieve adaptive partitioning of texture-
rich and texture-less regions based on image gradient fea-
tures and construct confidence factors using depth discrepancy
metrics to dynamically filter high-confidence regions. Differ-
ential normal vector constraints are then adaptively applied
according to regional characteristics to enhance fine surface
detail capture, which effectively suppresses high-frequency
geometric feature degradation during iterative optimization.
On the other hand, to improve multi-view consistency, we
utilize adjacent point cloud frames across views and simul-
taneously compute weight information derived from depth
discrepancy transformations in neighboring viewpoints. These
weights are fused to generate global confidence metrics, which
sample stable geometric priors from paired point cloud frames
to establish spatial normal consistency constraints. By inte-
arXiv:2602.12796v1  [cs.CV]  13 Feb 2026

<!-- page 2 -->
2
Fig. 1.
Spatial distribution comparison of Gaussian ellipsoids. This figure compares the spatial ellipsoid distributions reconstructed by the 3DGS, PGSR,
and Ours algorithms for the rendered scene. The 3DGS results exhibit suboptimal performance, as their Gaussian ellipsoids fail to conform closely to object
surfaces. While both PGSR and ours employ thin Gaussian ellipsoids for surface approximation, our method introduces novel constraints derived from single-
view and multi-view paradigms. This optimization yields more regular ellipsoid distributions and significantly enhanced surface conformity.
grating confidence-guided hierarchical geometric constraints
across single and multiple views, we build continuous geomet-
ric correlation models between unstructured point clouds. This
approach overcomes inter-view structural misalignment caused
by discrete Gaussian distributions while enhancing surface
detail reconstruction accuracy. The spatial distribution of the
Gaussian ellipsoid before and after optimization is shown in
Fig. 1. Our contributions can be summarised as follows:
• We propose a single-view adaptive sub-region constraint
with dual-branch optimization, which guides scene de-
coupling and weight filtering through image gradients
for accurate reconstruction of high-frequency geometric
features such as surface microstructures.
• We introduce a weight-guided dynamic sampling strat-
egy with cross-view geometric normal correlation con-
straints, constructing a 3D continuous geometric correla-
tion model via multi-view global weight adaptive filtering
of point cloud data, effectively addressing multi-view
consistency loss in complex scenes.
• A comprehensive algorithmic evaluation is conducted on
benchmark datasets (DTU [20], Mip-NeRF360 [21], and
Tanks and Temples [22]), including systematic compar-
ative evaluation with advanced 3D reconstruction frame-
works and detailed analysis of surface reconstruction and
novel view synthesis performance.
The remainder of this paper is organized as follows. Sec-
tion II reviews related work on 3D reconstruction and Gaussian
Splatting. Section III introduces the preliminaries of Planar-
based 3D Gaussian Splatting. Section IV details our proposed
GSM-GS framework. Section V presents the experimental
implementation, comparative results, and ablation studies.
Finally, Section VI concludes the paper.
II. RELATED WORK
A. Multi-view Stereo
In 3D surface reconstruction, multi-view stereo is a classical
problem. MVS generates intermediate geometric representa-
tions such as point clouds, voxel meshes [23], and depth maps
through multi-view geometric matching. Traditional methods
like COLMAP [24] rely on photometric consistency con-
straints in strongly textured regions and demonstrate excellent
performance in structured scenes, but their performance de-
grades significantly in weakly textured or occluded regions. To
address texture-deficient scenarios, learning-based MVS meth-
ods have been developed. CasMVSNet [25], [26] employs cas-
caded cost volume filtering to enhance depth accuracy while
reducing GPU memory consumption. TransMVSNet [27] in-
tegrates attention mechanisms from Transformer architectures
to strengthen feature relevance. However, these methods could
produce artifacts at occlusion boundaries and lack multi-view
consistency due to their per-view depth prediction paradigm.
B. Neural Radiance Fields
The mapping of spatial coordinates to color and volume
density in Neural Radiance Fields has driven significant ad-
vances in Novel View Synthesis. For instance, Mip-NeRF [28]
employs conical frustums to characterize scenes at multiple
scales, while Mip-NeRF 360 [21] extends this framework to
handle unbounded scenes. BungeeNeRF [29] adopts a pro-
gressive residual learning strategy to capture large-scale scene
details hierarchically. Pyramid NeRF [30] proposes a coarse-
to-fine reconstruction approach by progressively augmenting
high-frequency details using image pyramids. BARF [31]
introduces frequency-scheduled positional encoding to enable
training without precise camera poses. Another critical direc-
tion in NeRF research is high-fidelity 3D scene reconstruction.
Methods like NeuS [32] and BakedSDF [33] leverage signed
distance functions to represent surfaces and ensure watertight

<!-- page 3 -->
3
geometry reconstruction. Neuralangelo [34] combines multi-
resolution hash grids with SDFs for large-scale scene mod-
eling, while Nerf2Mesh [35] implements reprojection-error-
driven adaptive optimization to co-optimize mesh vertex dis-
tributions and volumetric density parameters. Although NeRF-
based frameworks achieve impressive surface reconstruction
and rendering quality, balancing training efficiency with re-
construction fidelity remains a challenge.
C. 3D Gaussian Splatting
3D Gaussian Splatting has become a prominent method
for scene rendering and surface reconstruction. Its explicit
representation using anisotropic 3D Gaussian primitives en-
ables rapid reconstruction and real-time rendering [36]. Cur-
rent optimized variants of 3DGS address rendering, dynamic
scenes, and large-scale reconstruction [37], [38]. For instance,
DNGaussian [39] employs depth regularisation to achieve
high-quality novel view synthesis with reduced sampling.
DN-Splatter [40] enhances Gaussian primitive representations
through base model optimization. To improve geometric fi-
delity, RaDe-GS [41] introduces depth/normal map rendering
during rasterization. GOF [42] leverages Gaussian opacity
fields for surface reconstruction, and SuGaR [43] applies
regularisation constraints to align Gaussians with surfaces,
extracting Poisson-reconstructed meshes via density fields.
2DGS [44] constrains Gaussian primitives by setting their z-
axis covariance to zero for planar projection. SolidGS [45]
integrates Gaussian surfel splatting to optimize primitive repre-
sentations. PGSR [46] incorporates unbiased depth estimation
and multi-view consistency constraints for accurate surface
reconstruction. MPGS [47] constrains Gaussian ellipsoids onto
multiple planes to reduce redundancy based on planar priors,
enhancing operational speed while preserving rendering qual-
ity. PUP 3D-GS [48] introduces an uncertainty-based pruning
method to quantify Gaussian importance, reducing compu-
tational overhead while improving reconstruction accuracy.
Furthermore, SpecTRe-GS [49] integrates a ray-tracing mech-
anism within the 3D space to achieve high-fidelity rendering
of specular surfaces.
Although existing methods have demonstrated notable im-
provements in reconstruction quality, their performance re-
mains to be enhanced for texture-less scenarios. To address
this, we introduce specialized constraints for texture-less re-
gions and refine the 3D point cloud reconstruction, thereby im-
proving reconstruction fidelity. Different from mesh-integrated
approaches like GaMeS [50] that trade precision for editing
flexibility via fixed topological priors, our method prioritizes
high-fidelity reconstruction via depth and normal consistency.
III. PRELIMINARY: PLANAR-BASED 3D GAUSSIAN
SPLATTING
The 3D Gaussian Splatting framework employs a collection
of anisotropic ellipsoids Gi to represent physical scenes, where
each ellipsoid follows a Gaussian distribution parameterized
by its centroid position µi
∈R3, opacity αi, spherical
harmonic (SH) coefficients, and covariance matrix Σi ∈R3×3.
These parameters collectively determine the ellipsoid’s color
ci ∈R3 and geometric shape. Formally, the Gaussian ellipsoid
distribution in the world coordinate system is defined as:
Gi(x|µi, Σi) = exp

−1
2(x −µi)⊤Σ−1
i (x −µi)

,
(1)
where the covariance matrix Σi admits decomposition into
a rotation matrix Ri ∈SO(3) and scaling matrix Si ∈
R3×3diag, satisfying Σi = RiSiS⊤
i R⊤
i . Through the trans-
formation matrix W
∈SE(3), the Gaussian ellipsoid is
transformed from world to camera coordinates, yielding the
projected mean vector µ′
i = KW[µ⊤
i
1]⊤and covariance
matrix Σ′
i = JWΣiW ⊤J⊤. Here, J ∈R2×3 denotes the
Jacobian matrix of perspective projection with radial distor-
tion approximation, while K ∈R3×3 represents the camera
intrinsic matrix. Combining the color ci and opacity αi of the
Gaussian ellipsoid, the RGB image in the current viewpoint
is rendered according to the α blend, and the specific process
can be expressed as follows:
C =
X
i∈P
Tiαici, Ti =
i−1
Y
j=1
(1 −αj),
(2)
where P denotes the Gaussian ordered by depth and Ti denotes
the cumulative transmittance. Similarly, the normal map N
and the distance map D of the scene can be rendered via
α-blending:
N =
X
i∈|N|
Tiαini, D =
X
i∈|N|
Tiαidi.
(3)
Here, ni denotes the normal vector of the Gaussian ellipsoid
surface, and di denotes the distance from the ellipsoid center
to the camera center. Based on the normal and depth maps,
the unbiased depth map ˆD ∈RH×W can be computed:
ˆD(p) =
D
N(p)K−1˜p,
(4)
where ˜p denotes the homogeneous coordinate of the 2D image-
plane position p = [u, v]⊤. In 3DGS, in addition to the
photometric loss of the base image Lrgb, in order to suppress
the effect of floating points on the reconstruction quality, the
homography matrix Hrn is used to maintain the geometric
multiview consistency Lmvpro and the photometry multi-view
consistency Lmvncc:
(
Lmvpro =
1
|V|
P
pr∈V ∥pr −HnrHrnpr) ∥
Lmvncc =
1
|V|
P
pr∈V (1 −NCC (Ir (pr) , In (Hrnpr))) ,
(5)
where pr denotes the pixel position in the reference frame,
pn is obtained by projecting pr to adjacent frames via the
single response matrix Hrn, and Ir(pr) and In(Hrnpr) denote
the pixel blocks of a particular size centred on pr and pn;
V = pr| ∥(pr −HnrHrnpr)∥≤θ is the set of pixels for which
the reprojection error has not exceeded a threshold; the nor-
malized cross-correlation (NCC) value NCC(Ir(pr), In(pn))
measures the local patch similarity. The final multi-view geo-
metric constraint is formulated as Lmvg = Lmvpro + Lmvncc.

<!-- page 4 -->
4
Fig. 2.
GSM-GS Overview. The algorithm framework takes sparse point cloud data and image input, initializes each point cloud into a thin Gaussian
ellipsoid, and processes a single-view adaptive partitioning constraint, dual-branch optimization strategy, weight-guided dynamic sampling strategy, and cross-
view geometric correlation normal constraint. The system is innovatively optimized from the perspectives of single-view and multi-view, and finally outputs
high-quality reconstruction and rendering results.
IV. METHODOLOGY
As shown in Fig. 2, the GSM-GS method enhances 3D
Gaussian reconstruction accuracy through single and multi-
view co-optimization. In this section, we detail the GSM-
GS framework’s methodology. In subsection IV-A, we present
the sub-region adaptive weighting strategy with single-view
normal constraints; in subsection IV-B, we introduce cross-
view geometric spatial normal constraints; in subsection IV-C,
we describe the loss formulation integrating reconstruction
quality, rendering fidelity, and geometric accuracy optimiza-
tion. Furthermore, to improve readability, definitions of math-
ematical symbols are described in detail in Appendix A.
A. Sub-region Adaptive Weighting with Single-view Normal
Constraints
In order to accurately characterize the surface features
of the object in a single view, a sub-region optimization
strategy is designed in this summary; the overall process is
as in Algorithm 1. First, image gradient analysis dynamically
partitions the scene. High-confidence regions are then selected
within partitioned areas using weight metrics. To adapt to
regional characteristics and refine error propagation in the
optimization process, we implement differentiated error com-
putation methods across regions, augmented by confidence-
aware weighting to modulate error impact.
1) Reliance on Regional Screening: This framework em-
ploys two depth computation methods from [46] to obtain
the rendered depth map D and unbiased depth map
ˆD.
When the reconstruction and rendering quality approaches
photorealistic accuracy, the depth discrepancy between D and
ˆD diminishes. The depth representations, therefore, satisfy
the asymptotic consistency condition: lim |D −ˆD| →0 as
geometric reconstruction converges to the true surface. This
discrepancy is consequently transformed into a weight metric
W ∈[0, 1] through:
W(i, j) = 1 −∆D(i, j)
∥∆D∥∞,Ω
,
∀(i, j) ∈Ω,
(6)
where (i, j) denotes pixel coordinates, W(i, j) ∈W repre-
sents the weight value at (i, j), ∆D(i, j) = |D(i, j)−ˆD(i, j)|
defines the depth discrepancy at (i, j), and Ωis the image
domain. Using weight map W, an adaptive thresholding
mechanism extracts high-weight trust regions H ⊂Ω, pri-

<!-- page 5 -->
5
Fig. 3. Filtering trust regions based on converting the difference between rendering depth and weight information. (a) is the RGB map of the real scene;
(b) the difference between the rendering depth and the unbiased depth is used to calculate the difference, and then the difference is converted into weight
information to reflect the reconstruction effect of each point in the scene; (c) the scene is partitioned using the weight information, and the low weight part is
filtered out(the bright part of the map), and the low weight part is filtered out(the darker region of the map), and the data of the trust region is prioritized to
added into the system for computation in the subsequent regularization. The data from the trusted regions is prioritized in the subsequent regularisation and
added to the system for computation.
oritized due to their higher reliability, as shown in Fig. 3.
These weights simultaneously quantify feature importance to
optimize constraint contributions in system optimization. The
trust region selection criterion is:
H = {(i, j) ∈Ω| W(i, j) ≥θ},
(7)
where θ is a set threshold, and the region above this threshold
is the trust region.
2) Texture Feature Decoupling: The image gradient is
utilized to dynamically divide the texture-rich and texture-
deficient regions of the scene. For the input image I, the
horizontal gradient Gx and vertical gradient Gy are computed
using the Sobel kernel [51]:
Gx(i, j) = ∂I(i, j)
∂x
∗Kx,
Gy(i, j) = ∂I(i, j)
∂y
∗Ky,
(8)
where Kx and Ky represent the Sobel operator in the hori-
zontal and vertical directions, respectively. The final gradient
magnitude can be calculated from the Euclidean parameter of
Gx(i, j) and Gy(i, j):
G(i, j) =
q
Gx(i, j)2 + Gy(i, j)2.
(9)
Since the Sobel operator incorporates smoothing and com-
putes gradients based on intensity differences, it effectively
suppresses noise and is robust to global illumination variations.
To achieve robust dynamic segmentation, we set the threshold
value τ to the 75th percentile of the gradient magnitude
distribution. Consequently, pixels with gradients exceeding τ
are classified as texture-rich regions R, while those with lower
gradients are assigned to texture-less regions B, as shown in
Fig. 4. The detailed dynamic process is described as:
p(i, j) ∈
(
R
if G(i, j) ≥τ
B
otherwise
,
(10)
where p(i, j) represents the pixel.
3) Single View Branch Normal Constraints: In the
[46]
method, the original single-view constraints are constructed
by weighting the rendered normal map by the power-of-
five image gradient and the difference in unbiased depth.
The method has some limitations: in texture-less regions, the
image gradient tends to 0 due to missing texture features,
and the contribution of the normal constraint term decays.
Meanwhile, the traditional normal constraint term is sensitive
to noise and prone to geometric ambiguity, leading to holes
or floating artifacts on the object’s surface. To address the
above problems, we propose a sub-area adaptive constraint
method. Different normal constraints are designed in texture-
rich and texture-less regions, and the weight W is introduced
in the texture-rich region to adaptively regulate the importance,
and the robustness of the system is enhanced by the regional
differentiation process.
In the reliable texture-rich region, based on the principle of
orthogonality, the orthogonal constraint term Lcross between
the depth gradient and the normal direction is introduced,
which has the mathematical form:
Lcross =
1
|IR∩H|
X
i,j∈IR∩H

∂(∇D)
∂x
· Ny −∂(∇D)
∂y
· Nx
 .
(11)
Fig. 4.
Calculate the gradient of the original image and classify the scene
into texture-rich and texture-less regions according to the threshold. (a)
Original RGB image. (b) Calculated gradient of the image, with the blue
bias representing a flatter gradient region and the white bias a higher gradient
region. (c) According to the threshold screening, the red part is a texture-rich
region and the blue part is a texture-less region.

<!-- page 6 -->
6
Denote the system constraint under single-view as
Lsvn =
1
|IR∩H|
X
(i,j)∈Ω
IR∩H(i, j)W(i, j) (∆N) + λ1Lcross,
(12)
where λ1 weights the internal orthogonality constraint Lcross;
IR∩H(i, j) ⊂Ωdesignates the pixel set within texture-
abundant trust regions, in which |IR∩H(i, j)| represents the
regional pixel cardinality; ∆N = ∥Nd −N∥1 denotes the
difference between the normal map Nd and the rendered
normal obtained through unbiased depth, while ∇D denotes
the gradient field computed from the discrepancy between
unbiased depth and rendered depth values. The geometric
surface undergoes optimization through trust-region normal
constraints, with concurrent construction of gradient fields in-
corporating depth-normal orthogonality constraints to mitigate
texture misinterpretation. System constraint terms are refined
via adaptive contribution weighting, ensuring high-frequency
normal orientations maintain geometric fidelity while sup-
pressing gradient conflict-induced over-regularisation artifacts.
In textured-less regions, a total variation (TV) regulariza-
tion [52] term weighted by color similarity is used:
∆N
i,j = exp (−|Ii,j −Ii−1,j|) (Ni,j −Ni−1,j)2
+ exp (−|Ii,j −Ii,j−1|) (Ni,j −Ni,j−1)2 ,
TVnormal = 1
|B|
X
(i,j)∈Ω
α(i, j)∆N
i,j.
(13)
Let |B| represent the pixel cardinality of set B, where
α(i, j) ∈B indexes pixel positions. Ii,j and Ni,j denote the
RGB intensity and surface normal vector at coordinate (i, j),
respectively. A color similarity-weighted exponential decay
operator adaptively modulates the smoothing intensity. Within
texture-impoverished regions, color consistency-driven geo-
metric smoothness constraints simultaneously mitigate texture
scarcity-induced reconstruction errors and suppress stochastic
noise propagation. The chromatic similarity weighting pre-
serves physically plausible discontinuities while preventing
edge over-smoothing, thereby enhancing both geometric com-
pleteness and reconstruction stability in texture-less domains
through discontinuity-aware regularization.
This framework implements a regionally-specialized con-
straint paradigm, enforcing strict constraints via dual-criteria
Fig. 5. Analysis of reconstruction results based on normal maps. (a) Ground-
truth RGB image; (b) Normal map reconstructed by the baseline PGSR
algorithm; (c) Normal map reconstructed using the proposed single-view
constrained optimization strategy. Comparative results demonstrate that our
method yields more accurate geometric reconstruction than the baseline,
particularly in preserving fine-grained details of real-world scenes.
Algorithm 1: Texture-aware Normal Loss
Input:
D, ˆD, Md, N, Igt : Geometric information
λ1, λ2, τ, θ : Weight and threshold
Output:
Lsvgeo : Texture-aware normal loss
for t = 0 to k do
W ←DeltaDepthWeight(D, ˆD)
G ←ComputeImageGradient(Igt)
for each (i, j) ∈IR∩H do
H ←{(i, j) | W(i, j) ≥θ}
R ←{(i, j) | G(i, j) ≥τ},
B ←Ω\ R
∇D(i, j) ←DeltaDepthGrad(D(i, j), ˆD(i, j))
Ltex ←SVRTexNorLoss(Nd(i, j), N(i, j))
Lcross ←DeltaDepthNorCross(∇D, N)
Lsvn+ = Ltex + λ1Lcross
for each (i, j) ∈B do
TVnormal(i, j)+ = SVWTextNorLoss(Igt, N)
Lsvn ←
1
|R∩H|Lsvn,
TVnormal ←
1
|B|TVnormal
Lsvgeo ←Lsvn + λ2TVnormal
return Lsvgeo
evaluation of trust regions and texture abundance for primary
texture-rich domains while employing streamlined trust region
assessment for spatially limited texture-impoverished areas.
The resultant single-view geometric constraint formulation is
expressed as:
Lsvgeo = Lsvn + λ2TVnormal,
(14)
where λ2 regulates the intensity of the total variation term
TVnormal. To validate the effectiveness of the dual-branch
constraint strategy based on regional texture feature differ-
ences, a comparative analysis of normal maps rendered by
the optimized method and the baseline PGSR algorithm is
presented in Fig. 5. The results demonstrate that the proposed
method significantly enhances geometric detail reconstruction
in both texture-rich regions (dinosaur head) and texture-less
regions (ground and introductory sign), thereby improving
geometric fidelity in real-world scene representation.
B. Geometry-guided Multi-view Consistency Constraints
Under multi-view constraints, the 3D object surface con-
straints are constructed based on the principle of re-projection
error minimization, and the algorithm framework is shown
in Fig. 6. The specific workflow proceeds as follows: Dur-
ing the system’s rendering and reconstruction phases, the
nearest neighboring view to the current perspective is first
identified. Depth information from both views is then utilized
to project corresponding pixels into 3D space, generating
complementary point clouds. Leveraging the relative pose
transformation between the current and neighboring views, the
neighboring point cloud is transformed into the current view’s
coordinate system. Under ideal reconstruction and rendering
conditions, these aligned point clouds would exhibit perfect
spatial congruence in geometric space, accompanied by iden-
tical surface normal orientations. Global confidence weights
are derived by integrating view-specific weighting metrics

<!-- page 7 -->
7
Fig. 6.
Guaranteeing multi-view consistency to minimize spatial artifacts
through cross-view spatial point cloud geometric constraints. Where rc and
rn are camera rays, Pc, Pn are depth-mapped point cloud points, PQs
c
, PQs
n
are point clouds sampled based on the global weights Wavg,ni
c, ni
n are the
surface normal vectors of the point cloud in two frames. The current view
{Vc, Tc} and the neighbouring view {Vn, Tn} are obtained first, and the
two frames of the point cloud are generated from the depth map, and then
the block of the point cloud is sampled based on the depth difference weights
and the global weights to complete the constraints on the surface normal of
the three-dimensional space.
from both perspectives. High-confidence point cloud samples
are subsequently selected to establish cross-view geometric
consistency constraints, thereby enhancing the framework’s
reconstruction fidelity and rendering precision.
1) Weight-guided Geometric Sampling Strategy: In this
method, 3D point cloud data Pc, Pn are based on the current
view Vc and its spatially nearest neighboring view Vn. The
distance depth maps D1, D2 and unbiased depth maps ˆD1, ˆD2
are output by the rendering operation. In order to reduce
the computational complexity, a geometric sampling strategy
based on global weight guidance is proposed: firstly, the global
weight Wavg is obtained by fusing the cross-view weights
Wc, Wn, which are calculated as follows:
Wavg(i, j) = βWc(i, j) + (1 −β)Wn(i, j).
(15)
Let Wc(i, j) ∈RH×W and Wn(i, j) ∈RH×W denote the
view-specific confidence weights computed for the current
view and its nearest neighbouring view, respectively, with
β = 0.5 balancing their contributions. The global confidence
weights Wavg(i, j), derived as a convex combination of Wc
and Wn, quantify the joint reliability of cross-view geometric
reconstruction. To harmonize computational efficiency with
reconstruction fidelity, a confidence-driven adaptive sampling
strategy is proposed to selectively retain stable geometric prim-
itives from the 3D point cloud. This strategy initiates by con-
structing a binary geometric validity mask Md ∈{0, 1}H×W
through projection consistency verification, defined as:
Md(i, j) =
(
1,
(ui, vi) ∈[0, W) × [0, H) ∧zi ≥ϵd
0,
otherwise
,
(16)
where (ui, vi) is the normalised projected coordinates of the
3D point cloud in the image plane, H and W are the image
size, and ϵd is the depth threshold, set to 0.1 m following the
default setting of the original framework [46], to filter near-
field noise from the camera. The set of candidate regions is
constructed by logically operating the global weights Wavg
with the mask Md:
Q = {(i, j) ∈Ω| Md(i, j) = 1 ∧Wavg(i, j) ≥γ},
(17)
where γ is the adaptive dynamic threshold, set to 30% of Wavg
to dynamically filter low-confidence regions and ensure the
quality of the candidate set Q. Subsequently, the top sample
rate S of high-weighted points are sampled in descending
order of confidence:
Qs = {(is, js) ∈Q|rank(Wavg(is, js)) ≤S}.
(18)
The operator rank(·) in Eq. 18 sorts the weights in descend-
ing order. The global-weight-driven geometric sampling strat-
egy employs prioritized probabilistic sampling within high-
confidence regions, thereby enforcing geometric constraints
exclusively on stable geometric primitives. Crucially, this
strategy functions as an implicit occlusion-aware mechanism
by automatically excluding regions where significant depth
discrepancies yield low weights W, thereby preventing erro-
neous geometric associations from degrading the optimization.
2) Cross-view Sampled Point Cloud Geometry Constraints:
Following the weight-driven geometric sampling strategy,
high-confidence point sets Pc and their nearest-neighbor coun-
terparts Pn are extracted from the current-view point cloud
PQs
c
and the neighboring-view point cloud PQs
n . To establish
geometrically consistent constraints using the sampled 3D
point clouds, both sets must reside within a unified coordinate
system. Consequently, PQs
n
undergoes a rigid transformation
to align with the current-view coordinate frame. Let Tc, Tn ∈
SE(3) denote the poses of the current and neighboring views,
respectively, within the world coordinate system:
Tc =
Rc
tc
0T
1

,
Tn =
Rn
tn
0T
1

,
(19)
where Rc, Rn ∈SO(3) is the rotation matrix and tc, tn ∈R3
is the translation vector. According to the relative positional
transformation TcTn
−1, the nearest-neighbor frame point pn
is transformed to the current frame coordinate system, and the
transformed point cloud is:
PQs
n→c =
(
Π3

TcT−1
n ˜pn
  ∀˜pn = [p⊤
n 1]⊤, pn ∈PQs
n
)
,
(20)
where Π3(·) projects homogeneous coordinates to three-
dimensional coordinates, and ˜p ∈R4 is the homogeneous
coordinate representation of the point pn. The PQs
c
and PQs
n→c
are constrained to the surface normals of the two frames of
the sampled point cloud under the same coordinate system

<!-- page 8 -->
8
based on the assumption of the local plane. Domain facets of
3 × 3 are extracted at corresponding points in the sampled
point clouds PQs
c
and PQs
n
and are denoted as {x1, . . . , x9}
and {y1, . . . , y9}, respectively. Principal Component Analysis
(PCA) [53] is utilised to solve for the normal vector of each
point in PQs
n
and PQs
n→c, denoted as nQs
c
and nQs
n . The normal
vector of each point cloud is calculated as:
ni
c = arg min
v∈R3
∥v∥=1
v⊤
M
X
k=1
(xk −µc)(xk −µc)⊤v,
ni
n = arg min
v∈R3
∥v∥=1
v⊤
M
X
k=1
(yk −µn)(yk −µn)⊤v,
(21)
where ni
c ∈nQs
c , ni
n ∈nQs
n
denote the two-frame point
cloud surface normal vectors; µc =
1
M
PM
k=1 xk and µn =
1
M
PM
k=1 yk denote the mean value of localized surface sheets
of the current view and the transformed view, respectively
(M = 9); v is the direction of the unit normal vector con-
straints of the PCA solution (||v|| = 1). To quantify the local
geometric stability, the normalized curvature is defined based
on the PCA eigenvalues. The higher the curvature, the less flat
the plane is, and the more unstable the geometric estimate is,
so an exponential decay weight function is constructed:
wi
κ = exp
 
−10 ·
η3
c
P3
k=1 ηkc
!
,
η1
c ≥η2
c ≥η3
c,
(22)
where ηk
c represents the eigenvalue of the local surface sheet in
the current view, and wi
κ ∈wκ denotes the local surface sheet
curvature. Using the curvature to weight the surface normal
constraints, the weighted cosine similarity loss term is:
Lmvgeo =
1
|K|
X
i∈|K|
wi
κ ·

1 −
ni
c
T ni
n


,
(23)
where K denotes the set of point cloud surface facets, |K|
denotes the number of surfaces, and wi
κ ∈wκ denotes the
curvature. The process of constructing multi-view consistency
constraints based on geometric guidance is shown in Algo-
rithm 2.
C. Loss Functions
To summarise, incorporating the RGB reconstruction loss
Lrgb, employed in 3D Gaussian Splatting, yields the final total
Algorithm 2: Multi-view Geometry Constraints
Input:
Vc, Vn : Camera perspective
Tc, Tn : Camera pose
G : Gaussian Representation
Output:
Lmvgeo : Multi-view Geometry loss
{Dc, ˆDc, Nc} ←Render(Vc, G)
{Dn, ˆDn, Nn} ←Render(Vn, G)
{Pc, Pn} ←GetPointsFromDepth(Vc, Vn, ˆDc, ˆDn)
{Wc, Wn} ←DeltaDepthWeight(Dc, ˆDc, Dn, ˆDn)
Wavg ←βWc + (1 −β)Wn
Q ←{(i, j) ∈Ω| Md(i, j) ←1 ∧Wavg(i, j) ≥γ}
Qs ←{(is, js) ∈Q | rank(Wavg(is, js)) ≤S}
PQs
n→c ←
(
Π3

TcT−1
n [p⊤
n 1]⊤
, pn ∈PQs
n
)
for each pc ∈PQs
c
and pn ∈PQs
n
do
// Compute eigenvalues and
eigenvectors of local patch
(λ1
c, λ2
c, λ3
c; v1
c, v2
c, v3
c) ←PatchPCA(pc, P = 3)
(λ1
n, λ2
n, λ3
n; v1
n, v2
n, v3
n) ←PatchPCA(pn, P = 3)
// Select eigenvector corresponding
to smallest eigenvalue
ni
c ←v1
c
(λ1
c ≤λ2
c ≤λ3
c)
ni
n ←v1
n
(λ1
n ≤λ2
n ≤λ3
n)
κi ←
λ3
c
P3
k=1 λkc
,
wi
κ ←exp(−10κi)
Lmvgeo+ = wi
κ ·

1 −
ni
c
⊤ni
n


return Lmvgeo
loss term for the training process:
L = Lrgb + Lsvgeo + λ3Lmvgeo,
(24)
where λ3 > 0 denotes a weighting factor. For the weight
settings of the system constraints, for the photometric con-
straints, we keep the same weights as in 3DGS, with the in-
novative single-view geometric constraints weights λ1 = 0.05
and λ2 = 0.01, and the multi-view cross-view point cloud
geometric constraints weights are set to λ3 = 0.001.
V. EXPERIEMENTS
A. Experimental Settings
Datasets and Evaluation Metrics: To validate the effec-
tiveness of the proposed method, experiments were conducted
TABLE I
QUANTITATIVE ANALYSIS OF THE 3D SCENE RECONSTRUCTION ACCURACY OF THE ALGORITHM USING CHAMFER DISTANCE (MM)↓ON THE DTU
DATASET. ”RED”, ”ORANGE”, AND ”YELLOW” INDICATE THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS, RESPECTIVELY.
Methods
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
Mean
Mean Time
NeRF [13]
1.90
1.60
1.85
0.58
2.28
1.27
1.47
1.67
2.05
1.07
0.88
2.53
1.06
1.15
0.96
1.49
> 0.8h
VolSDF [14]
1.14
1.26
0.81
0.49
1.25
0.70
0.72
1.29
1.18
0.70
0.66
1.08
0.42
0.61
0.55
0.86
> 0.8h
NeuS [32]
1.00
1.37
0.93
0.43
1.10
0.65
0.57
1.48
1.09
0.83
0.52
1.20
0.35
0.49
0.54
0.84
> 8.5h
2DGS [44]
0.52
0.82
0.35
0.42
0.93
0.97
0.82
1.23
1.24
0.64
0.68
1.27
0.42
0.67
0.48
0.76
0.15h
GOF [42]
0.53
0.83
0.40
0.38
1.35
0.82
0.79
1.26
1.30
0.69
0.71
1.37
0.52
0.64
0.51
0.81
0.58h
RaDe-GS [41]
0.43
0.75
0.34
0.37
0.84
0.72
0.67
1.20
1.24
0.64
0.62
0.85
0.35
0.66
0.47
0.68
0.13h
PGSR [46]
0.37
0.55
0.42
0.35
0.78
0.58
0.49
1.08
0.64
0.59
0.48
0.53
0.30
0.37
0.35
0.53
0.28h
Ours
0.34
0.55
0.37
0.34
0.77
0.55
0.49
1.04
0.64
0.58
0.47
0.48
0.30
0.36
0.33
0.51
0.45h

<!-- page 9 -->
9
Fig. 7. Qualitative analysis of surface reconstruction on DTU. We compare with existing methods to better capture object surface details and more clearly
represent object surface features.
on three widely used datasets: DTU [20], Mip-NeRF360 [21],
and Tanks and Temples [22]. The DTU dataset comprises
15 scenes that cover both reflective and shadowed areas;
this dataset is commonly used to verify 3D reconstruction
performance. The Mip-NeRF360 dataset provides 360-degree
omnidirectional data across 9 indoor and outdoor scenes. Its
long viewpoint trajectories, multi-scale geometric structures,
and complex lighting variations rigorously test algorithms
for continuous viewpoint synthesis and detail rendering. The
Tanks and Temples dataset features large-scale real-world
scenes with dynamic illumination and intricate occlusions,
enabling a comprehensive evaluation of algorithmic robust-
ness in unconstrained environments. To quantify algorithm
performance, Chamfer distance (CD) was used to measure 3D
reconstruction quality; Peak Signal-to-Noise Ratio (PSNR),
Structural Similarity Index (SSIM), and Learned Perceptual
Image Patch Similarity (LPIPS) metrics were used to evaluate
novel view synthesis performance.
Baselines: To evaluate the superiority of this method,
a
comprehensive
comparative
analysis
of
3DGS
[16],
2DGS [44], GOF [42], RaDe-GS [41], PGSR [46], and
the proposed approach is performed on the Mip-NeRF360,
Tanks and Temple, and DTU datasets in terms of both novel
view synthesis and 3D reconstruction accuracy. Experimental
comparisons are performed in various scenarios, integrating
qualitative and quantitative analyses to provide comprehensive
empirical support for the superiority of the method presented
in this document.
Implementation Details: The framework is architected
using PyTorch, C++, and CUDA technology stacks, with all
experimental validations conducted on an NVIDIA GeForce
RTX 4090 GPU featuring 24GB of VRAM. The system ini-
tialization phase employs COLMAP-generated input images,
point clouds, and camera poses as foundational data. We
implement a two-phase optimization strategy across 30,000
training iterations: initial 7,000 iterations prioritize parameter
optimization through photometric constraints, while subse-
quent 23,000 iterations enhance model fidelity by incorporat-
ing single/multiview geometric constraints. Our hybrid densi-
fication module synergistically integrates the AbsGS [54] and
GOF [42] methodologies, with activation limited to the first
15,000 iterations. The remaining hyperparameters maintain
consistency with the original 3D Gaussian Splatting imple-
mentation.
In the design of the geometric constraint mechanism, we
set multiple thresholds to better regulate the system effect: the
single-view scene is configured with a confidence threshold
θ = 0.8 to screen valid regions; a texture feature difference

<!-- page 10 -->
10
TABLE II
QUANTITATIVE COMPARISON OF THE TANKS AND TEMPLE DATASET,
COMPARING THE F1-SCORE, THE EXPERIMENTAL RESULTS SHOW THAT
OUR METHOD ACHIEVES OPTIMAL RESULTS IN MOST SCENARIOS.
2DGS [44]
GOF [42]
RaDe-GS [41]
PGSR [46]
Ours
Caterpillar
0.22
0.40
0.36
0.41
0.43
Courthouse
0.14
0.28
0.27
0.20
0.21
Ignatius
0.51
0.65
0.72
0.77
0.79
Meetingroom
0.05
0.06
0.04
0.13
0.13
Truck
0.16
0.16
0.14
0.21
0.22
Mean
0.22
0.31
0.31
0.34
0.36
threshold τ, set to the 75th percentile of the image gradient
histogram, for feature delineation; the multiview environment
adopts a depth threshold ϵd = 0.1 to eliminate near-field
noise and further optimizes the geometric constraints using
S = 16 with a random point cloud block sampling strategy.
The sensitivity analysis results of the above parameters are
presented in Fig. 13 and 14 of Appendix B-A, as well as
Fig. 15 and Table VII in Appendix B-B.
B. Geometry Reconstruction
The geometric reconstruction performance of the algorithm
is analyzed on the DTU and Tanks and Temple datasets,
respectively. Qualitative and quantitative results are shown in
Table I, Table II and Fig. 7.
In terms of training efficiency (rightmost column of Ta-
ble I), our method requires an average of 0.45 h. This is
slightly higher than RaDe-GS (0.13 h), 2DGS (0.15 h), and
PGSR (0.28 h). However, it substantially improves geometric
accuracy at a modest increase in computation cost. The other
methods have CD values of 0.68, 0.76, and 0.53 mm, while
our method achieves 0.51 mm. It is also more efficient than
the high-accuracy GOF (0.58 h). Ablation studies in the
Appendix C-B show that the single-view constraint (Lsvgeo)
adds negligible computation (0.28 h →0.30 h). The increase
in total time mainly comes from the cross-view constraint
(Lmvgeo), which involves explicit SE(3) rigid transformations
and per-point PCA spectral analysis. These results confirm that
our method achieves high-fidelity geometric reconstruction
while maintaining computational efficiency.
To evaluate the generalization capability of our method in
complex scenes, we conducted group-wise comparisons of F1-
Score on the Training subset of the Tanks and Temple dataset.
The results in Table
II show that our method achieves an
average F1-Score of 0.36, corresponding to improvements of
16.4%–63.6% over 2DGS (0.22), GOF (0.31), and RaDe-GS
(0.31), and a 5.9% relative improvement over the second-best
Fig. 8. The figure shows the quantitative analysis of 3DGS, RaDe-GS, PGSR, and the present algorithm in terms of rendering effect, where GT represents
the real scene. The rendering effects of four different scenes in the Mip-NeRF360 dataset, including indoor and outdoor, are compared in detail, and the
corresponding PSNR and LPIPS metrics of each algorithm are listed.

<!-- page 11 -->
11
TABLE III
QUANTITATIVE RESULTS OF RENDERING QUALITY FOR NEW VIEW SYNTHESIS ON THE MIP-NERF360 DATASET. “RED”, “ORANGE”, AND “YELLOW”
REPRESENT THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS.
Dataset
Outdoor scenes
Indoor scenes
Average on all scenes
Method | Metric
SSIM ↑
PSNR ↑
LPIPS ↓
SSIM ↑
PSNR ↑
LPIPS ↓
SSIM ↑
PSNR ↑
LPIPS ↓
3DGS [16]
0.742
25.03
0.232
0.931
31.20
0.164
0.837
28.12
0.198
2DGS [44]
0.703
24.17
0.287
0.910
30.06
0.214
0.807
27.12
0.250
GOF [42]
0.746
24.81
0.208
0.917
30.40
0.189
0.832
27.61
0.199
PGSR [46]
0.752
24.74
0.203
0.929
30.16
0.159
0.841
27.45
0.181
Ours
0.749
24.78
0.199
0.932
30.66
0.151
0.841
27.72
0.175
Fig. 9. Quantitative analysis on the LLFF dataset. Based on the comparison of
PSNR metric changes during the experiments, it can be seen that our method
has better algorithmic performance.
method PGSR (0.34), demonstrating its overall advantage in
complex outdoor scene reconstruction. On the DTU dataset,
our method attains a reconstruction accuracy with an average
chamfer distance of 0.51 mm, representing a 3.8% improve-
ment over PGSR (0.53 mm). It achieves the best results in
13 out of 15 evaluation scenes and reduces the error by 9.4%
in Scan 110, indicating high robustness and consistency in
geometric reconstruction quality.
C. Novel View Synthesis
To validate the rendering quality, three (3DGS, PGSR,
GSM-GS) better-performing algorithms were selected for
quantitative and qualitative analyses on the Mip-NeRF360
dataset, as shown in Table III and Fig. 8. The proposed method
achieves competitive results across all three key metrics,
SSIM, PSNR, and LPIPS, with particularly outstanding LPIPS
performance in both indoor and outdoor scenarios. For the
counter indoor scene, our approach accurately restores line
segment details on the floor, closely approximating the real
scene. In the treehill outdoor scene, it effectively eliminates
floor unevenness and distortion artifacts prevalent in other
algorithms. This advantage stems from our branching geomet-
ric constraint strategy, which weights plausible regions and
applies differentiated regularisation to texture-rich and texture-
deficient areas, thereby capturing geometric details through-
out the scene. For room and flowers scenes, the introduced
cross-view 3D point cloud geometric constraints significantly
enhance spatial consistency, particularly in preserving geo-
metric coherence of complex 3D structures, demonstrating
the method’s competitive rendering capabilities. More detailed
experimental settings and implementation details are provided
in Table VIII of Appendix C-A.
To demonstrate the superiority of the algorithm proposed in
this paper over the benchmark method (PGSR), we conducted
extensive experiments on the LLFF dataset [55]. The PSNR
metrics at 7000, 10000, 15000, 20000, 25000, and 30000
iterations are documented, as shown in Fig. 9. Analysis of
the results indicates that our method uniquely incorporates 2D
spatial fine-detail capture and 3D spatial point cloud geometry
constraints. This integration results in a more stable PSNR
enhancement trend and a higher final value throughout the
iterative process. After 15000 iterations, system densification
ceases, and the point cloud count stabilizes. Our method im-
poses more effective geometric constraints on the 3D Gaussian
point clouds, leading to a closer fit of the Gaussian ellipsoid
distribution to the real scene surface, as shown in Fig. 10.
The progressively widening PSNR gap relative to PGSR
demonstrates improved accuracy in 3D scene reconstruction
and validates the enhanced capability of our approach to
characterize the scene.
D. Ablation Studies
To verify the effectiveness of the texture-guided branching-
based constraints and surface normal constraints between
cross-view point cloud frames proposed in this paper for
single-view, we conducted ablation experiments on the DTU
and Mip-NeRF360 datasets. The results of the qualitative
and quantitative analyses are shown in Table IV, Table V,
Fig. 11, and Fig. 12. Among them, L denotes the benchmark
constraints of the baseline method [46], which includes image
photometric consistency, single-view geometric constraints,
and multi-view geometric constraints; Lsvgeo and Lmvgeo rep-
resent the single-view and multi-view geometric constraints,
respectively, innovatively designed in this paper’s optimization
modules. Therefore, (L + Lsvgeo) and (L + Lmvgeo) denote

<!-- page 12 -->
12
Fig. 10. Comparison of spatial Gaussian ellipsoid distributions. (a) represents the real RGB map; (b) represents the spatial Gaussian ellipsoid distribution of
the PGSR method; (c) represents the spatial Gaussian ellipsoid distribution of our method. It can be found that the Gaussian ellipsoid of our method fits the
real surface of the object better.
the single-view and multi-view geometric constraints schemes,
respectively, in the baseline method optimized with the inno-
vative constraints proposed in this paper.
TABLE IV
ABLATION STUDY OF OUR MODEL ON RECONSTRUCTION ACCURACY
ACROSS THE DTU DATASETS.
L
Lsvgeo
Lmvgeo
CD↓
✓
×
×
0.53
✓
✓
×
0.50
✓
×
✓
0.48
✓
✓
✓
0.48
Fig. 11. Ablation study on DTU datasets. Observing the details of the recon-
structed model, it is found that all of our proposed optimization constraints are
able to improve compared to the previous method, with a finer characterization
of the surface details.
Comparison of the experimental data shows that, compared
to the benchmark method L, introducing either the single-view
optimization constraints(L + Lsvgeo) or the multi-view opti-
mization constraints(L + Lmvgeo) significantly improves both
3D reconstruction accuracy and new view synthesis quality.
Furthermore, the synergistic integration of the two constraints
TABLE V
ABLATION STUDY OF OUR MODEL ON NOVEL VIEW SYNTHESIS ACROSS
THE MIP-NERF360 DATASETS.
L
Lsvgeo
Lmvgeo
SSIM↑
PSNR↑
LPIPS ↓
✓
×
×
0.928
30.22
0.179
✓
✓
×
0.930
30.77
0.172
✓
×
✓
0.929
30.35
0.179
✓
✓
✓
0.930
30.80
0.172
Fig. 12. Ablation study on Mip-NeRF360 datasets. Experiments with novel
view synthesis on the Mip-NeRF360 dataset demonstrate that all of our
proposed innovations improve rendering somewhat.
(L + Lsvgeo + Lmvgeo) drives the system to achieve bet-
ter performance. Specifically, the single-view texture-guided
branching constraint significantly enhances robustness and
generalization ability to dynamic scene changes by modeling
multi-scale scene detail features.
Concurrently, the cross-view point cloud inter-frame surface
normal constraint effectively eliminates geometric artifacts in
3D space by reinforcing multi-view geometric consistency,
thereby optimizing reconstruction rendering. Taken together,
each constraint mechanism proposed in this paper indepen-

<!-- page 13 -->
13
dently improves the baseline model’s performance, with all
modules contributing significantly to enhancing the final re-
construction and rendering results. More detailed ablation
results on scene reconstruction are provided in Appendix C-B.
VI. CONCLUSION
This paper proposes GSM-GS, which achieves region parti-
tioning and adaptive constraints through single-view gradients
and adaptive weighting. Furthermore, it introduces a geometry-
guided cross-view point cloud association mechanism to ef-
fectively enforce multi-view consistency within the 3D space.
The core innovation of this work lies in the proposal of region-
specific customized constraint strategies and the construction
of a multi-view consistency enhancement mechanism based
on 3D spatial characteristics, which effectively suppresses
geometric artifacts. Extensive experiments on public datasets
featuring challenging scenarios, such as texture-less regions,
occlusions, and specular highlights, demonstrate the superi-
ority and robustness of the proposed method in both surface
reconstruction and novel view synthesis tasks.
Nevertheless, the method still faces challenges when dealing
with transparent materials, highly reflective surfaces, and in-
tricate thin-walled structures due to the inherent ambiguity be-
tween geometry and appearance induced by complex lighting
effects. Future work will focus on exploring more advanced
scene representations and optimization strategies to further
enhance the model’s robustness and generalization capabilities
in complex scenes.
REFERENCES
[1] S. Cheng, C. Sun, S. Zhang, and D. Zhang, “Sg-slam: A real-time
rgb-d visual slam toward dynamic scenes with semantic and geometric
information,” IEEE Transactions on Instrumentation and Measurement,
vol. 72, pp. 1–12, 2023.
[2] D. Su, H. Kong, S. Sukkarieh, and S. Huang, “Necessary and sufficient
conditions for observability of slam-based tdoa sensor array calibration
and source localization,” IEEE Transactions on Robotics, vol. 37, no. 5,
pp. 1451–1468, 2021.
[3] J. Wakulicz, H. Kong, and S. Sukkarieh, “Active information acquisition
under arbitrary unknown disturbances,” in 2021 IEEE International
Conference on Robotics and Automation (ICRA).
IEEE, 2021, pp.
8429–8435.
[4] H.
Zhao,
W.
Guan,
and
P.
Lu,
“Lvi-gs:
Tightly
coupled
li-
dar–visual–inertial slam using 3-d gaussian splatting,” IEEE Transac-
tions on Instrumentation and Measurement, vol. 74, pp. 1–10, 2025.
[5] G. Li, S. Fan, Y. Zhang, Y. Wang, Q. Wang, F. Yu, W. Jin, and Y. Wang,
“A novel tightly coupled attitude and heading measurement method
based on full-sky polarization mode for bionic navigation system,” IEEE
Transactions on Instrumentation and Measurement, vol. 73, pp. 1–11,
2024.
[6] B. Xu, J. Hu, and Y. Guo, “An acoustic ranging measurement aided
sins/dvl integrated navigation algorithm based on multivehicle cooper-
ative correction,” IEEE Transactions on Instrumentation and Measure-
ment, vol. 71, pp. 1–15, 2022.
[7] D. Wang, X. Xu, Y. Yao, T. Zhang, and Y. Zhu, “A novel sins/dvl
tightly integrated navigation method for complex environment,” IEEE
Transactions on Instrumentation and Measurement, vol. 69, no. 7, pp.
5183–5196, 2020.
[8] J. Wang, Y. He, D. Su, K. Itoyama, K. Nakadai, J. Wu, S. Huang, Y. Li,
and H. Kong, “Slam-based joint calibration of multiple asynchronous
microphone arrays and sound source localization,” IEEE Transactions
on Robotics, vol. 40, pp. 4024–4044, 2024.
[9] N. Deng, Z. He, J. Ye, B. Duinkharjav, P. Chakravarthula, X. Yang, and
Q. Sun, “Fov-nerf: Foveated neural radiance fields for virtual reality,”
IEEE Transactions on Visualization and Computer Graphics, vol. 28,
no. 11, pp. 3854–3864, 2022.
[10] T. Yi, J. Fang, J. Wang, G. Wu, L. Xie, X. Zhang, W. Liu, Q. Tian,
and X. Wang, “Gaussiandreamer: Fast generation from text to 3d
gaussians by bridging 2d and 3d diffusion models,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 6796–6807.
[11] L. Wang, K. Cheng, S. Lei, S. Wang, W. Yin, C. Lei, X. Long, and
C.-T. Lu, “Dc-gaussian: Improving 3d gaussian splatting for reflective
dash cam videos,” arXiv preprint arXiv:2405.17705, 2024.
[12] K. Yang, J. Wang, M. Xu, K. Wang, and Z. Chen, “Drgs-slam: Depth-
regularized 3-d gaussian slam with hierarchical strategy and virtual
viewpoints supervision,” IEEE Transactions on Instrumentation and
Measurement, vol. 74, pp. 1–9, 2025.
[13] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[14] L. Yariv, J. Gu, Y. Kasten, and Y. Lipman, “Volume rendering of neural
implicit surfaces,” Advances in Neural Information Processing Systems,
vol. 34, pp. 4805–4815, 2021.
[15] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Transactions on
Graphics, vol. 41, no. 4, pp. 1–15, 2022.
[16] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[17] Z. Huang and M. Gong, “Textured-gs: Gaussian splatting with spatially
defined color and opacity,” arXiv preprint arXiv:2407.09733, 2024.
[18] T. Shen, S. Liu, J. Feng, Z. Ma, and N. An, “Topology-aware 3d
gaussian splatting: Leveraging persistent homology for optimized struc-
tural integrity,” in Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 39, no. 7, 2025, pp. 6823–6832.
[19] M. Yu, T. Lu, L. Xu, L. Jiang, Y. Xiangli, and B. Dai, “Gsdf: 3dgs
meets sdf for improved neural rendering and reconstruction,” Advances
in Neural Information Processing Systems, vol. 37, pp. 129 507–129 530,
2024.
[20] R. Jensen, A. Dahl, G. Vogiatzis, E. Tola, and H. Aanæs, “Large scale
multi-view stereopsis evaluation,” in Proceedings of the IEEE conference
on computer vision and pattern recognition, 2014, pp. 406–413.
[21] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 5470–5479.
[22] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics (ToG), vol. 36, no. 4, pp. 1–13, 2017.
[23] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and
A. Kanazawa, “Plenoxels: Radiance fields without neural networks,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022, pp. 5501–5510.
[24] J. L. Schonberger and J.-M. Frahm, “Structure-from-Motion Revisited,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2016, pp. 4104–4113.
[25] X. Gu, Z. Fan, S. Zhu, Z. Dai, F. Tan, and P. Tan, “Cascade cost
volume for high-resolution multi-view stereo and stereo matching,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2020, pp. 2495–2504.
[26] X. Qiao, C. Ge, P. Deng, H. Wei, M. Poggi, and S. Mattoccia, “Depth
restoration in under-display time-of-flight imaging,” IEEE Transactions
on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 5668–
5683, 2023.
[27] Y. Ding, W. Yuan, Q. Zhu, H. Zhang, X. Liu, Y. Wang, and X. Liu,
“Transmvsnet: Global context-aware multi-view stereo network with
transformers,” in Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2022, pp. 8585–8594.
[28] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields,” in Proceedings of the IEEE/CVF inter-
national conference on computer vision, 2021, pp. 5855–5864.
[29] Y. Xiangli, L. Xu, X. Pan, N. Zhao, A. Rao, C. Theobalt, B. Dai, and
D. Lin, “Bungeenerf: Progressive neural radiance field for extreme multi-
scale scene rendering,” in European conference on computer vision,
2022, pp. 106–122.
[30] J. Zhu, H. Zhu, Q. Zhang, F. Zhu, Z. Ma, and X. Cao, “Pyramid nerf:
Frequency guided fast radiance field optimization,” International Journal
of Computer Vision, vol. 131, no. 10, pp. 2649–2664, 2023.

<!-- page 14 -->
14
[31] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, “Barf: Bundle-adjusting
neural radiance fields,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2021, pp. 5741–5751.
[32] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “Neus:
Learning neural implicit surfaces by volume rendering for multi-view
reconstruction,” arXiv preprint arXiv:2106.10689, 2021.
[33] L. Yariv, P. Hedman, C. Reiser, D. Verbin, P. P. Srinivasan, R. Szeliski,
J. T. Barron, and B. Mildenhall, “Bakedsdf: Meshing neural sdfs
for real-time view synthesis,” in ACM SIGGRAPH 2023 Conference
Proceedings, 2023, pp. 1–9.
[34] Z. Li, T. M¨uller, A. Evans, R. H. Taylor, M. Unberath, M.-Y. Liu, and
C.-H. Lin, “Neuralangelo: High-fidelity neural surface reconstruction,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 8456–8465.
[35] C. Zhang, Y. Zhou, and L. Zhang, “Voxel-mesh hybrid representation for
real-time view synthesis by meshing density field,” IEEE Transactions
on Visualization and Computer Graphics, 2024.
[36] Y. Li, C. Lyu, Y. Di, G. Zhai, G. H. Lee, and F. Tombari, “Geogaussian:
Geometry-aware gaussian splatting for scene rendering,” in European
Conference on Computer Vision, 2024, pp. 441–457.
[37] Y. Gao, Y. Dai, H. Li, W. Ye, J. Chen, D. Chen, D. Zhang, T. He,
G. Zhang, and J. Han, “Cosurfgs: Collaborative 3d surface gaussian
splatting with distributed learning for large scene reconstruction,” arXiv
preprint arXiv:2412.17612, 2024.
[38] J. Lin, Z. Li, X. Tang, J. Liu, S. Liu, J. Liu, Y. Lu, X. Wu, S. Xu, Y. Yan
et al., “Vastgaussian: Vast 3d gaussians for large scene reconstruction,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 5166–5175.
[39] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, “Dngaus-
sian: Optimizing sparse-view 3d gaussian radiance fields with global-
local depth normalization,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2024, pp. 20 775–20 785.
[40] M. Turkulainen, X. Ren, I. Melekhov, O. Seiskari, E. Rahtu, and
J. Kannala, “Dn-splatter: Depth and normal priors for gaussian splatting
and meshing,” in 2025 IEEE/CVF Winter Conference on Applications
of Computer Vision (WACV).
IEEE, 2025, pp. 2421–2431.
[41] B. Zhang, C. Fang, R. Shrestha, Y. Liang, X. Long, and P. Tan,
“Rade-gs: Rasterizing depth in gaussian splatting,” arXiv preprint
arXiv:2406.01467, 2024.
[42] Z. Yu, T. Sattler, and A. Geiger, “Gaussian opacity fields: Efficient adap-
tive surface reconstruction in unbounded scenes,” ACM Transactions on
Graphics (TOG), vol. 43, no. 6, pp. 1–13, 2024.
[43] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting
for efficient 3d mesh reconstruction and high-quality mesh rendering,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 5354–5363.
[44] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting
for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024
conference papers, 2024, pp. 1–11.
[45] Z. Shen, Y. Liu, Z. Chen, Z. Li, J. Wang, Y. Liang, Z. Yu, J. Zhang,
Y. Xu, S. Schaefer et al., “Solidgs: Consolidating gaussian sur-
fel splatting for sparse-view surface reconstruction,” arXiv preprint
arXiv:2412.15400, 2024.
[46] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu,
H. Bao, and G. Zhang, “Pgsr: Planar-based gaussian splatting for
efficient and high-fidelity surface reconstruction,” IEEE Transactions on
Visualization and Computer Graphics, 2024.
[47] D. Li, S.-S. Huang, and H. Huang, “Mpgs: Multi-plane gaussian splat-
ting for compact scenes rendering,” IEEE Transactions on Visualization
and Computer Graphics, vol. 31, no. 5, pp. 3256–3266, 2025.
[48] A. Hanson, A. Tu, V. Singla, M. Jayawardhana, M. Zwicker, and
T. Goldstein, “Pup 3d-gs: Principled uncertainty pruning for 3d gaussian
splatting,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2025, pp. 5949–5958.
[49] J. Tang, F. Fei, Z. Li, X. Tang, S. Liu, Y. Chen, B. Huang, Z. Chen,
X. Wu, and B. Shi, “Spectre-gs: Modeling highly specular surfaces
with reflected nearby objects by tracing rays in 3d gaussian splatting,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2025, pp. 16 133–16 142.
[50] L. G. et al., “Mesh-based gaussian splatting for real-time large-scale
deformation,” arXiv preprint arXiv:2402.04796, 2024.
[51] N. Kanopoulos, N. Vasanthavada, and R. Baker, “Design of an image
edge detection filter using the sobel operator,” IEEE Journal of Solid-
State Circuits, vol. 23, no. 2, pp. 358–367, 1988.
[52] Z. Liang, Q. Zhang, Y. Feng, Y. Shan, and K. Jia, “Gs-ir: 3D gaussian
splatting for inverse rendering,” in Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, 2024, pp. 21 644–
21 653.
[53] A. Ma´ckiewicz and W. Ratajczak, “Principal components analysis (pca),”
Computers & Geosciences, vol. 19, no. 3, pp. 303–342, 1993.
[54] Z. Ye, W. Li, S. Liu, P. Qiao, and Y. Dou, “Absgs: Recovering fine details
in 3d gaussian splatting,” in Proceedings of the 32nd ACM International
Conference on Multimedia, 2024, pp. 1053–1061.
[55] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ra-
mamoorthi, R. Ng, and A. Kar, “Local light field fusion: Practical view
synthesis with prescriptive sampling guidelines,” ACM Transactions on
Graphics (ToG), vol. 38, no. 4, Jul. 2019.

<!-- page 15 -->
15
APPENDIX
This section is structured as follows: Appendix A provides a
systematic definition of the key notations employed through-
out this paper; Appendix B presents an in-depth sensitivity
analysis of the core hyperparameters; and Appendix C offers
supplementary qualitative and quantitative evaluations across
multiple datasets to further validate the proposed method.
APPENDIX A
SYMBOL DEFINITION
In this section, we provide a detailed explanation of the key
symbols involved in the methodology, as shown in Table VI.
TABLE VI
SUMMARY OF MATHEMATICAL NOTATIONS AND SYMBOLS
Symbol
Definition
Ω
Image domain
H
High-weight trust regions
θ
Trust regions threshold
R, B
Texture-rich and texture-less regions
τ
Gradient Threshold
M
Geometric validity mask
ϵd
Depth Threshold
γ
Adaptive dynamic threshold
Q, Qs
Candidate set and high-confidence regions
S
Sampling Rate
rank(·)
Descending sorting operator
Π3(·)
Projection operator (Homogeneous to Euclidean)
APPENDIX B
SINGLE-VIEW AND MULTI-VIEW PARAMETER SETTINGS
This section investigates the sensitivity of key parameters
under both single-view and multi-view conditions. Specif-
ically, we evaluate the trust threshold θ for distinguishing
reliable from unreliable regions, the gradient-based threshold
τ for segregating texture-rich and texture-less areas, the depth
threshold ϵd, the adaptive dynamic threshold γ, and sampling
rate S.
A. Single-view Parameter Settings
To evaluate the efficacy of the exploration threshold θ in
identifying reliable regions, the selected high-weight regions
are superimposed onto the depth difference map using a
semi-transparent blue mask(as shown in Fig. 13). In these
maps, blue/cyan represents low depth error (high confidence),
while yellow/red indicates high error (low confidence). An
optimal threshold should ensure that the high-weight mask
predominantly covers blue/cyan areas; coverage of yellow/red
regions implies misjudgment due to an overly loose threshold.
Observations reveal that for θ < 0.5, the mask erroneously
extends into significant yellow/red high-error regions, indicat-
ing insufficient separation. Conversely, within the 0.7 −0.9
range, the boundaries become distinct and align well with low-
error areas. Consequently, we selected θ = 0.8 as the default
setting, where the high-weight regions exhibit the best spatial
consistency with low-error areas, achieving an optimal balance
between accuracy and robustness.
To evaluate the sensitivity of the segmentation performance
to the threshold τ (distinguishing texture-rich from texture-
less regions), we systematically analyze gradient magnitude
distributions and corresponding quantile thresholds across
multiple scenes Fig. 14. The leftmost column displays gradient
histograms, while subsequent columns illustrate segmentation
results derived from thresholds at quantiles ranging from
p = 10% to p = 90%. Experimental results indicate that low
quantiles (10–30%) yield excessively low thresholds, causing
widespread misclassification of weak gradient regions as tex-
tured areas. Conversely, high quantiles (90%) result in overly
aggressive thresholds, leading to the significant loss of struc-
tural edges. Only the intermediate range (50–80%) demon-
strates stable segmentation consistency across diverse scenes,
where the proportion of high-texture regions, denoted as hf,
decreases smoothly and monotonically with p. Consequently,
the optimal range for τ is constrained to p ∈[50%, 80%]. In
this study, we select the threshold corresponding to p = 75%,
as it resides within this stable interval and ensures robust
texture partitioning.
B. Multi-view Parameter Settings
Following the PGSR framework [46], we set the depth
threshold ϵd in mask Md to 0.1m. This enforces a physical
Fig. 13. Sensitivity analysis of the confidence threshold θ. The optimal value (θ = 0.8) aligns high-weight regions (blue mask) with low-error areas in the
depth difference map, providing reliable region selection and balancing accuracy with robustness.

<!-- page 16 -->
16
Fig. 14.
Sensitivity analysis of texture segmentation quantiles. Across diverse scenes, p ∈[50%, 80%] provides the most stable partitioning, while other
ranges cause noise over-segmentation or edge loss. Thus, p = 75% is selected as the optimal setting for robust texture-less region separation.
lower bound zi ≥ϵd, effectively filtering near-field high-
frequency noise and low-SNR artifacts to enhance reconstruc-
tion robustness.
We introduce an adaptive threshold γ, set to 30% of the
global average weight Wavg, to enforce a minimum quality
baseline for the candidate set Q. By acting as a preliminary
filter, γ excludes low-confidence points to ensure Wavg(i, j) ≥
γ, thereby guaranteeing a valid geometric lower bound. This
configuration maintains a sufficient candidate pool (|Q| ≫S)
while securing the robustness of the sampling strategy across
diverse scenarios.
Fig. 15. Sensitivity analysis of sampling parameter S regarding PSNR and
total computation time. The yellow solid lines represent our PSNR (dB)
and the blue dashed lines represent total training time (min). The horizontal
red and green dotted lines denote the baseline performance of PGSR. Peak
accuracy is achieved at S = 16, after which the accuracy gains diminish
while computational costs continue to rise linearly.
A sensitivity analysis was conducted on the key parameter S
of the weight-guided geometric sampling strategy to balance
reconstruction accuracy against computational efficiency (as
shown in Fig. 15 and Table VII). The reconstruction accuracy
exhibited a trend of rapid initial improvement followed by
saturation and a marginal decline as S increased; specifically,
the PSNR peaked at 27.95 dB when S = 16. Compared
TABLE VII
QUANTITATIVE IMPACT OF SAMPLING PARAMETER S ON
RECONSTRUCTION PERFORMANCE AND TIME CONSUMPTION.
Scenes
Metric
S
0
4
8
16
32
64
flower
PSNR ↑
27.55
27.63
27.64
27.72
27.68
27.74
All Time ↓
20.45
30.85
31.86
35.63
40.10
51.15
trex
PSNR ↑
27.70
27.77
27.82
27.95
27.92
27.75
All Time ↓
33.00
34.86
36.10
39.44
45.35
56.96
to the low sampling rate of S = 4 (27.77 dB), this 0.18
dB improvement validates the efficacy of moderate sampling
in capturing geometric details. However, when S > 16, the
accuracy gains diminished or even slightly regressed (27.75 dB
at S = 64). This is primarily because an excessive sampling
count introduces points with low weights and insufficient
geometric confidence, allowing geometric noise to interfere
with the overall optimization process. Meanwhile, the total
computation time increased linearly and substantially, from
39.44 min at S = 16 to 56.96 min at S = 64. Consequently,
S = 16 achieves optimal accuracy while maintaining accept-
able computational overhead, representing the ideal trade-off
between precision and efficiency; it is thus selected as the
default sampling parameter.
APPENDIX C
ADDITIONAL RESULTS
A. Results on Mip-NeRF360
Table VIII presents a scene-wise quantitative comparison on
the Mip-NeRF360 dataset, demonstrating our method’s robust-
ness against state-of-the-art baselines. Notably, we outperform

<!-- page 17 -->
17
TABLE VIII
QUANTITATIVE COMPARISON ON THE MIP-NERF 360 DATASET. WE REPORT SSIM, PSNR, AND LPIPS METRICS ACROSS DIFFERENT SCENES. ”RED”,
”ORANGE”, AND ”YELLOW” INDICATE THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS, RESPECTIVELY.
Method
Metric
bicycle
bonsai
counter
flowers
garden
kitchen
room
stump
treehill
3DGS [16]
SSIM ↑
0.779
0.947
0.915
0.622
0.874
0.933
0.928
0.783
0.653
PSNR ↑
25.65
32.37
29.19
21.84
27.82
31.51
31.74
26.96
22.86
LPIPS ↓
0.203
0.175
0.178
0.329
0.103
0.113
0.191
0.208
0.318
2DGS [44]
SSIM ↑
0.731
0.929
0.890
0.572
0.838
0.914
0.905
0.756
0.618
PSNR ↑
24.70
31.18
28.07
21.07
26.57
30.15
30.83
26.14
22.39
LPIPS ↓
0.272
0.229
0.233
0.377
0.149
0.148
0.245
0.260
0.377
GOF [42]
SSIM ↑
0.787
0.937
0.902
0.638
0.868
0.916
0.913
0.794
0.643
PSNR ↑
25.48
31.57
28.69
21.66
27.42
30.68
30.65
26.98
22.49
LPIPS ↓
0.180
0.198
0.203
0.280
0.107
0.137
0.218
0.196
0.278
PGSR [46]
SSIM ↑
0.793
0.945
0.914
0.636
0.872
0.932
0.926
0.797
0.661
PSNR ↑
25.64
31.59
28.29
21.44
27.43
30.76
29.99
26.89
22.28
LPIPS ↓
0.186
0.169
0.172
0.264
0.103
0.113
0.180
0.193
0.271
Ours
SSIM ↑
0.793
0.947
0.917
0.633
0.875
0.934
0.930
0.798
0.648
PSNR ↑
25.61
31.97
28.71
21.46
27.59
31.03
30.94
27.00
22.23
LPIPS ↓
0.178
0.160
0.164
0.260
0.098
0.109
0.172
0.186
0.275
TABLE IX
SUPPLEMENTARY ABLATION STUDIES WERE CONDUCTED USING THE DTU DATASET. BEST RESULTS ARE HIGHLIGHTED IN RED.
Method
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
Mean
Mean Time
L
0.37
0.55
0.42
0.35
0.78
0.58
0.49
1.08
0.64
0.59
0.48
0.53
0.30
0.37
0.35
0.53
0.28h
+Lmvgeo
0.37
0.55
0.39
0.34
0.78
0.58
0.49
1.06
0.64
0.58
0.47
0.50
0.30
0.37
0.34
0.52
0.45h
+Lsvgeo
0.36
0.56
0.36
0.34
0.77
0.58
0.49
1.05
0.63
0.58
0.47
0.48
0.30
0.36
0.33
0.51
0.30h
L + Lsvgeo + Lmvgeo
0.36
0.56
0.37
0.34
0.77
0.57
0.49
1.03
0.64
0.58
0.47
0.48
0.30
0.36
0.33
0.51
0.45h
the strong baseline PGSR in LPIPS across 8 of 9 scenes and
achieve superior PSNR in 7 of 9 scenes. These results validate
that by imposing strict geometric constraints, our approach
effectively mitigates visual artifacts while maintaining com-
petitive rendering fidelity in complex environments.
B. Results on DTU Dataset
Comprehensive ablation studies on the DTU dataset (Ta-
ble IX) corroborate the efficacy of our method. Results indicate
that incorporating Lsvgeo or Lmvgeo individually enhances
reconstruction metrics. Their joint application yields perfor-
mance comparable to individual constraints due to perfor-
mance saturation, yet it consistently outperforms the baseline.
Although these constraints incur a moderate increase in train-
ing time, the computational overhead remains manageable.
Thus, the proposed method offers a favorable balance between
reconstruction fidelity and efficiency.
