<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
1
Accurate and Complete Surface Reconstruction
from 3D Gaussians via Direct SDF Learning
Wenzhi Guo, Guangchi Fang, Mingqing Wei, Jie Qin, Bing Wang∗
Abstract—3D Gaussian Splatting (3DGS) has recently emerged
as a powerful paradigm for photorealistic view synthesis, rep-
resenting scenes with spatially distributed Gaussian primitives.
While highly effective for rendering, achieving accurate and
complete surface reconstruction remains challenging due to the
unstructured nature of the representation and the absence of
explicit geometric supervision. In this work, we propose DiGS,
a unified framework that embeds Signed Distance Field (SDF)
learning directly into the 3DGS pipeline, thereby enforcing strong
and interpretable surface priors. By associating each Gaussian
with a learnable SDF value, DiGS explicitly aligns primitives with
underlying geometry and improves cross-view consistency. To fur-
ther ensure dense and coherent coverage, we design a geometry-
guided grid growth strategy that adaptively distributes Gaussians
along geometry-consistent regions under a multi-scale hierarchy.
Extensive experiments on standard benchmarks, including DTU,
Mip-NeRF 360, and Tanks & Temples, demonstrate that DiGS
consistently improves reconstruction accuracy and completeness
while retaining high rendering fidelity. Our code is available at
https://github.com/DARYL-GWZ/DIGS.
Index Terms—3D Gaussian Splatting, Surface reconstruction,
Rendering.
I. INTRODUCTION
Recent advances in 3D Gaussian Splatting (3DGS) [1]
have transformed neural scene representation and rendering.
By modeling a scene as a collection of anisotropic Gaussian
primitives and rendering them through a carefully engineered
rasterization pipeline, 3DGS achieves real-time, photo-realistic
view synthesis. Its efficiency and fidelity have quickly spurred
adoption in diverse applications, ranging from augmented and
virtual reality [2]–[5] to autonomous driving [6]–[8].
Despite these advances, accurate geometry reconstruction
remains a fundamental challenge for Gaussian-based repre-
sentations. Unlike volumetric fields or mesh-based methods,
Gaussian primitives are inherently unstructured and provide
only weak geometric cues under multi-view supervision. As a
result, they often exhibit drift and misalignment with underly-
ing surfaces, leading to limited reconstruction fidelity [9]–[11].
Moreover, existing growth strategies for Gaussian models [1],
[12] typically produce sparse and uneven distributions, failing
W. Guo is with the Department of Aeronautical and Aviation Engineering,
The Hong Kong Polytechnic University, Hong Kong SAR, China and the De-
partment of Computer Science and Technology, Nanjing University, Nanjing,
China (e-mail: wenzhi.guo@connect.polyu.hk).
Mingqing Wei and Jie Qin are with School of Artificial Intelligence,
Nanjing University of Aeronautics and Astronautics, Nanjing, China, and the
(e-mail: mqwei@nuaa.edu.cn, jie.qin@nuaa.edu.cn).
Guangchi Fang and Bing Wang are with the Department of Aeronautical and
Aviation Engineering, The Hong Kong Polytechnic University, Hong Kong
SAR, China (e-mail: guangchi.fang@gmail.com, bingwang@polyu.edu.hk)
Fig. 1.
Conceptual Overview of DiGS. DiGS advances Gaussian splat-
ting from an appearance-driven representation toward a geometry-preserving
paradigm, by embedding signed distance supervision into Gaussian primitives
under a geometry-guided growth strategy.
to guarantee comprehensive scene coverage in complex or low-
texture environments. These shortcomings suggest that current
pipelines are largely appearance-driven, with geometry treated
as an auxiliary objective rather than a core principle.
To address this, recent advances have sought to augment
Gaussian splatting with stronger geometric priors. Geometric
regularization methods [13], [14] introduce constraints such as
normals or silhouettes, while surface extraction pipelines [12],
[15] recover meshes only in a post-processing stage. SDF-
based formulations provide a more direct coupling to geom-
etry: 3DGSR [16] supervises Gaussians with signed distance
values but suffers from artifacts under sparse guidance, and
GSDF [17] adopts a dual-branch optimization that separates
rendering and reconstruction, thereby increasing model com-
plexity and weakening consistency between appearance and
geometry. Similarly, PGSR [14] leverages composite geomet-
ric losses but requires post-training refinement for complete-
ness, while Octree-GS [12] employs hierarchical multi-scale
expansion guided primarily by appearance, often resulting in
uneven coverage. Collectively, these approaches demonstrate
the potential of SDF-augmented Gaussians, yet they leave
unresolved the fundamental question of how geometry and
appearance can be unified within a single, efficient framework.
Collectively, these approaches demonstrate the potential of
SDF-augmented Gaussians, yet they leave unresolved the
fundamental question: how can geometry and appearance be
unified within a single Gaussian framework, such that signed
distance priors and growth strategies are jointly leveraged to
achieve both accurate and complete surface reconstruction?
In this paper, we present DiGS, a unified framework that
advances Gaussian splatting from an appearance-driven rep-
resentation to a geometry-preserving paradigm. Unlike prior
approaches that either impose weak geometric regularization
arXiv:2509.07493v2  [cs.CV]  21 Sep 2025

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
2
Fig. 2.
Detailed Framework of DiGS. Starting from an SfM-initialized LoD grid, DiGS associates each cell with SDF-augmented Gaussians. Gaussian
splatting and signed distance values are jointly predicted, with SDFs converted to opacity via a differentiable kernel. During training, a geometry-guided grid
growth strategy expands Gaussians along geometry-consistent regions.
[14], [15] or decouple geometry and appearance into sepa-
rate optimization branches [16], [17], DiGS integrates signed
distance supervision directly into the Gaussian primitives,
treating geometry as a first-class objective within the ren-
dering pipeline. Each Gaussian is constrained by a learned
signed distance value at its center, while a geometry-guided
grid growth strategy systematically expands Gaussians along
surface-consistent regions under a multi-scale level-of-detail
(LoD) hierarchy. Crucially, SDF learning and geometry-guided
grids growth are mutually dependent and jointly optimized:
SDF supervision prevents Gaussian drift and grids growth
guarantees surface completeness. Their tight coupling forms
a necessary and self-reinforcing system, establishing DiGS
as a new unified paradigm for geometry-aware Gaussian
splatting that simultaneously achieves high-fidelity rendering
and accurate surface reconstruction.
The main contributions are summarized as follows:
• We propose DiGS, a unified SDF-Gaussian framework,
which directly embeds signed distance supervision into
Gaussian primitives, achieving precise and consistent
surface reconstruction.
• We develop a geometry-guided Gaussian grid growth
mechanism that facilitates dense and spatially coher-
ent coverage, particularly effective for complex or low-
texture geometries.
• Extensive experiments on DTU, Mip-NeRF 360, and
Tanks & Temples demonstrate consistent improvements
in reconstruction accuracy and completeness compared to
state-of-the-art Gaussian-based and hybrid methods.
II. RELATED WORK
A. Neural Surface Reconstruction
Neural surface reconstruction has evolved as a result of
the interplay between implicit representations and geometric
constraints. The pioneering work of NeRF [18] introduced
volumetric rendering for view synthesis, but it faced significant
limitations in surface modeling due to its density-based formu-
lation, which lacked explicit geometric constraints. To address
this, NeuS [19] and VolSDF [20] tightly coupled Signed
Distance Fields (SDFs) with probabilistic volume rendering,
enabling surface extraction through the zero-level set of the
SDF, offering a more precise model for surface representation.
These methods laid the foundation for surface-aware neural
rendering but still struggled with handling complex geometries
or large-scale scenes. A comprehensive survey of such neural
implicit surface reconstruction paradigms is provided in [21],
which systematizes recent advances and open challenges.
Subsequent works advanced the state of geometric fidelity
through hybrid implicit-explicit representations. For exam-
ple, UNISURF [22] unified implicit surfaces with volume
rendering, offering improved detail preservation. Similarly,
MonoSDF [23] and Geo-Neus [24] introduced monocular
depth and normal priors for regularization, helping stabilize
the optimization process by providing additional geometric
priors. Extensions such as Vox-Surf [25] explored voxel-based
implicit surface representation, while H-SDF [26] proposed a
hybrid sign-distance formulation to support arbitrary topolo-
gies. Indoor scene reconstruction has also benefited from
hybrid priors, as shown in [27], where fine-grained geometry
is recovered by leveraging normal enhancements. In parallel,

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
3
semantic-driven surface modeling from point clouds, such as
RangeUDF [28], demonstrated that semantic cues can further
improve surface reconstruction robustness in challenging real-
world environments.
Beyond hybrid priors, point-guided approaches such as PG-
NeuS [29] and its progressive-growing tri-plane extension
PGT-NeuS [30] improved robustness and efficiency by lever-
aging sparse point correspondences and hierarchical repre-
sentations. Prior-driven surface learning was further explored
in PSDF [31], which injects geometric priors into implicit
learning for better multi-view reconstruction. In parallel, scal-
able formulations like [32] addressed the challenge of high-
quality reconstruction in large-scale settings, improving both
accuracy and training efficiency. Complementary to these,
decomposition-based formulations such as DM-NeRF [33] en-
abled geometry decomposition and manipulation directly from
2D images, highlighting the potential of scene factorization for
enhancing surface-aware neural rendering.
Accelerated frameworks such as Instant-NGP [34] and Neu-
ralangelo [35] utilized multi-resolution hash grids and coarse-
to-fine optimization strategies to enhance detail recovery and
computational efficiency. Meanwhile, task-specific extensions
emerged, for example H2O-NeRF [36] tailored radiance field
reconstruction to the special case of two-hand-held objects,
demonstrating the adaptability of neural implicit surfaces to
constrained domains.
3DGSR [16] was one of the first methods to integrate SDFs
directly with 3D Gaussian Splatting (3DGS) to enforce surface
constraints while maintaining rendering speed. This approach
introduced loose coupling strategies to align Gaussian primi-
tives with SDF-derived surfaces, achieving a more detailed re-
construction than previous density-based methods. Despite the
notable progress, the scalability and computational efficiency
of these methods remained a challenge, particularly for large-
scale and high-resolution scenes. For instance, BakedSDF [37]
and NeRFMeshing [38] require hours of training and struggle
with efficiently handling large-scale scene reconstruction.
Recent advancements, such as H3DNet [39], further empha-
size the importance of geometric priors to stabilize the opti-
mization of neural implicit surfaces, ensuring both geometric
accuracy and robustness. Scaffold-GS [40] introduced hier-
archical Gaussian structures for better alignment with scene
geometry, but these approaches still face significant trade-offs
between surface accuracy and computational efficiency.
B. Gaussian Splatting Based Surface Reconstruction
The introduction of 3D Gaussian Splatting (3DGS) [1],
[41], [42] marked a breakthrough in real-time rendering for
neural surface reconstruction, by representing scenes as sets of
anisotropic Gaussians that can be rasterized efficiently. While
3DGS excels in rendering photorealistic images, it inherently
struggles with geometric accuracy due to the unstructured
nature of Gaussian primitives. Early attempts to reconcile
rendering efficiency with surface quality include methods like
SuGaR [15], which employed Poisson reconstruction, and
2DGS [13], which collapsed 3D Gaussians into planar surfels.
These methods improved the overall quality but failed to
preserve fine details or handle complex geometries.
NeuSG [43] and GOF [12] explored hybrid implicit-explicit
paradigms by using SDFs or opacity fields to guide the
placement of Gaussians. However, they face critical limi-
tations, particularly in the alignment between the Gaussian
primitives and the underlying surface. For example, Scaffold-
GS [40] improved the structural alignment of the Gaussians
but sacrificed high-frequency details, while PGSR [40] used
planar decomposition techniques that limited the adaptability
of the approach to more complex topologies.
3DGSR [16] proposed loose coupling strategies to align
Gaussians with SDF-derived surfaces, enhancing the detail
of the reconstruction while maintaining rendering efficiency.
However, 3DGSR still faced challenges in achieving full
geometric coverage and handling sparse data effectively.
Beyond these hybrid paradigms, several recent works ex-
panded the scope and adaptability of 3DGS to various
scenarios. MPGS [44] introduced a multi-plane Gaussian
representation to improve compact scene rendering, while
LoopSparseGS [45] proposed loop-based strategies for sparse-
view friendly reconstruction. Fov-GS [46] leveraged foveated
rendering to enable efficient handling of dynamic scenes,
and RGAvatar [47] extended Gaussian splatting to relightable
4D human avatar modeling. Similarly, PlGS [48] addressed
robust panoptic lifting, and Look-at-the-Sky [49] specialized
Gaussian splatting for outdoor environments. iVR-GS [50]
explored editable and explorable visualization via inverse vol-
ume rendering, and ARAP-GRF [51] integrated deformation
models for as-rigid-as-possible Gaussian field editing.
GSDF [17] later unified 3D Gaussian Splatting with SDFs
through mutual geometry supervision, offering an improved
surface representation. However, GSDF still struggled with
computational efficiency and could not completely resolve
issues related to large-scale scene reconstruction. GARF
[52] introduced geometry-aware Gaussian refinement, but this
method retained computational bottlenecks due to iterative
density refinement processes.
These existing methods highlight the need for a framework
that can tightly couple geometry and appearance while pre-
serving the computational efficiency of 3D Gaussian Splatting.
Our work in DiGS directly addresses these challenges by
integrating SDF learning with 3D Gaussian Splatting in a
unified manner. By combining normal-aligned growth and
depth-guided grid expansion, DiGS ensures accurate surface
reconstruction, robust spatial coverage, and high rendering effi-
ciency. This innovative approach not only enhances geometric
fidelity but also provides a scalable solution for textureless
scenes, addressing the limitations of previous methods.
III. PRELIMINARY
a) 3D
Gaussian
Splatting:
3D Gaussian Splatting
(3DGS) [1] represents a scene as a collection of spatially
distributed anisotropic 3D Gaussians {Gi}N
i=1, where each
primitive is parameterized by its center pi ∈R3, a covariance
matrix Σi ∈R3×3, color ci ∈R3, and opacity αi ∈[0, 1].
The covariance matrix is typically factorized into a scaling and
rotation component to define elliptical spatial support.

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
4
To render an image, each Gaussian is projected into screen
space, and its contribution is accumulated via front-to-back
α-compositing:
C(u) =
X
i∈N
Tiαici,
Ti =
i−1
Y
j=1
(1 −αj),
(1)
where N denotes the sorted set of Gaussians affecting pixel
u, and Ti represents the transmittance.
While 3DGS delivers photorealistic rendering with real-
time performance, it lacks explicit geometric structure,
and its opacity-driven formulation provides limited surface-
awareness. Our work builds on this foundation by embedding
signed distance priors directly into the Gaussian represen-
tation, enabling explicit surface alignment while preserving
rendering efficiency.
b) Surface Regularization: To enhance geometric fi-
delity in multi-view reconstruction, recent methods introduce
regularization objectives that enforce consistency across spatial
and photometric domains. For instance, PGSR [14] proposes
a composite geometric loss Lgeo consisting of: (i) single-view
geometric priors Lsvgeo, (ii) multi-view geometric consistency
Lmvgeo, and (iii) photometric alignment loss Lmvrgb. These
constraints improve robustness in untextured regions and guide
reconstructions toward structurally plausible surfaces.
Instead of optimizing loss terms post hoc, our framework
integrates geometric priors directly into the representation
by learning signed distance values at each Gaussian center,
inherently encoding surface proximity and enabling structure-
aware supervision during rendering and growth.
IV. METHOD
We propose DiGS, a unified framework that tightly couples
implicit surface modeling with explicit radiance representation
under the Gaussian splatting paradigm. Rather than treating ge-
ometry and appearance as disjoint components, DiGS embeds
a Signed Distance Field (SDF) directly into each Gaussian
primitive, and integrates a multi-scale Level-of-Detail (LoD)
structure with a geometry-guided growth strategy. In this way,
surface priors are inherently encoded within the rendering
process, leading to precise surface alignment and efficient,
adaptive scene representation. Importantly, all components of
our approach are co-designed and jointly optimized end-to-
end. This is in stark contrast to prior methods like GSDF
[17], which separate geometry and radiance into dual branches,
or Octree-GS [53] and PGSR [14], which perform multi-
resolution refinement as a post-process. By unifying these
aspects in a single pipeline, our method ensures that geometry
and appearance remain consistent and mutually reinforcing
throughout optimization. We next detail each part of the
method, organized as follows: an SDF-Gaussian representation
(IV-A) defines our base scene parameterization, a geometry-
guided grid growth mechanism (IV-B) progressively refines
this representation, and a set of optimization objectives (IV-
C) enforces geometric fidelity and photometric accuracy.
Fig. 3.
Network Architecture. Given Gaussian position and view direction
as inputs, an MLP predicts radiance features and signed distance values. The
SDF is mapped to opacity through a learnable transfer function, ensuring
surface awareness, while the radiance branch produces color for rendering.
A. Unified SDF-Gaussian Representation
Our scene representation augments each Gaussian splat
with an explicit SDF value and is structured in a multi-
scale fashion. This SDF-Gaussian representation provides a
common foundation for modeling both surface geometry and
appearance. Unlike approaches that simply combine separate
modules, we design this representation as a unified structure.
In particular, the SDF at each Gaussian influences its rendering
opacity, and the LoD hierarchy is built to accommodate the
SDF-guided growth of Gaussians.
1) Octree-Based LoD Initialization: To efficiently cover
scenes at multiple scales, we begin by partitioning space with
a hierarchical octree grid. Each octree level ℓ= 0, 1, . . . , Lmax
corresponds to a voxel size:
sℓ= s0 · 2ℓ.
(2)
where s0 is the base voxel size. Coarser levels (small ℓ) thus
span larger regions with fewer primitives, while finer levels
(large ℓ) focus on local details. We choose the number of
levels Lmax based on scene complexity and desired detail.
This ensures we capture fine details without an excessive
number of primitives. The LoD structure enables adaptive
refinement: broad areas are first represented coarsely, and only
regions with complex geometry receive finer subdivisions.
We initialize the octree using a sparse Structure-from-Motion
(SfM) point cloud, which provides an initial approximation of
scene geometry. Starting from the root (level 0) covering the
entire scene, we subdivide octree cells down to a certain depth
where sufficient SfM points fall inside. This yields an initial
set of occupied voxels across levels, giving low-resolution
coverage of the scene. Notably, the LoD grid in DiGS is not
a fixed, static structure: it will be dynamically updated during
training by our growth procedure .
2) Gaussian Position Encoding: Within each occupied oc-
tree cell, we allocate a small set of Gaussian primitives to
represent local geometry and appearance. Specifically, each
voxel at level ℓis associated with k Gaussians (with k is 10
in our implementation). We parameterize the positions of these
Gaussians relative to the voxel center xv as:

p0, . . . , pk−1
	
= xv + {C0 · L0, . . . , Ck−1 · Lk−1} .
(3)

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
5
where each Ci ∈R3 is a learnable offset and Li ∈R3
is a learnable scale factor. Intuitively, Ci and Li allow each
Gaussian to shift from the cell center and spread around
the voxel. This neural position encoding grants the model
flexibility to fit complex local surface geometry: by adjusting
the offsets, multiple Gaussians in a cell can align along
different surface structures.
Each Gaussian is also associated with additional properties,
such as a color and an anisotropic covariance matrix that
defines its spatial extent. This design is crucial for capturing
high-frequency geometry in regions of interest while keeping
less interesting regions coarse. It also naturally complements
the LoD structure by allowing newly subdivided cells to inherit
and refine the features from their parent cell.
At any 3D Gaussian primitive center p, we can query
the neural grid features via interpolation. We employ an
inverse distance weighting (IDW) scheme to aggregate features
from nearby voxel centers. Specifically, we find the eight
surrounding grid vertices that form the corners of the cell
containing p at the appropriate LoD level, and compute an
interpolated feature F(p) as:
F(p) =
8
X
i=1
wi
P
j wj
f ′,
where wi =
1
∥xi −p∥+ ϵ.
(4)
where fi is the feature stored at vertex xi, and ϵ is a small
constant for numerical stability. This weighted sum smoothly
blends the features of the nearby Gaussians or grid nodes,
ensuring that F(p) changes continuously as p moves through
space. We then decode F ′(p) from F(p) via MLP and directly
split the one-dimensional value of the decoded neural Gaussian
Primitive feature F ′(p) as the SDF value f(p), the whole
process is shown on Figure 3.
3) SDF Supervision and Opacity Modulation: To infuse
geometric awareness into each Gaussian, we assign a Signed
Distance value to its center. Let f(p) denote the signed
distance from point p to the nearest surface in the scene,
with the convention that f(p) = 0 on the surface, f(p) > 0
outside (in free space), and f(p) < 0 inside solid regions. In
our representation, each Gaussian center pi carries a learnable
parameter fi ≈f(pi) that estimates the true SDF at that
location. The collection of all Gaussians’ SDFs thus implicitly
defines the scene’s surface as the zero-level set:
S = {p ∈R3 | f(p) = 0}.
(5)
which we aim to reconstruct accurately. During training, we
provide SDF supervision to guide fi toward this true signed
distance field (details in Sec. IV-C1), effectively encoding
strong geometric priors into the model. The result is that
geometry is no longer an afterthought: it is a core part of
the representation, directly influencing how each primitive is
treated in rendering.
To make use of these SDF values in our differentiable
rendering process, we introduce an SDF-guided opacity mod-
ulation for the Gaussians. In standard Gaussian splatting
(3DGS), each Gaussian has an opacity that determines how
much it contributes along a ray, but this opacity is typically
learned as an independent parameter or derived from image
colors, offering only indirect geometric meaning. In DiGS, we
compute each Gaussian’s opacity αi as a function of its SDF
value fi. Specifically, we define a smooth kernel that peaks
when the Gaussian lies on the surface (fi = 0) and decays as
the Gaussian moves away from the surface:
αi = exp

−f(pi)2
δ2

.
(6)
where δ is a learnable bandwidth parameter controlling the
sharpness of the decay. This SDF-to-opacity mapping is a
differentiable function that effectively locks each Gaussian to
the surface: Gaussians with fi ≈0 (near the surface) will
have high opacity (contribute strongly to rendering), whereas
Gaussians with large |fi| (far from the surface either outside
or inside) become nearly transparent. By plugging this αi
into the volume rendering, we ensure that only surface-aligned
primitives are visibly rendered, and any Gaussian that drifts
off the surface will automatically diminish in influence. The
parameter δ can be learned or set to a small value so that
the opacity drops off quickly with distance, yielding a tight
approximation of an actual surface.
Crucially, our framework directly couples geometry and
appearance via SDF-modulated opacity, unifying radiance and
surface representations in a single branch. Each Gaussian
encodes both geometry and color, ensuring consistency with-
out the dual pipelines and enforcement mechanisms required
by GSDF [17]. This tight integration simplifies optimization,
avoids geometry–appearance misalignments, and yields more
coherent, cross-view consistent reconstructions.
B. Geometry-Guided Grid Growth
Even with multi-scale priors, an initial Gaussian grid may
leave surfaces under-sampled, especially in weakly textured
or SfM-sparse regions. To overcome this, DiGS employs
a geometry-guided grid growth strategy that inserts new
Gaussians during training based on depth and normal cues,
ensuring accurate surface coverage. Unlike offline refinements,
growth is integrated into end-to-end optimization, leveraging
the LoD octree for placement and SDF/normal supervision for
precision. The process follows three stages: (1) depth-normal
estimation, (2) coarse-to-fine insertion, and (3) Progressive
refinement with growth and pruning. Pseudocode is provided
in Algorithm 1 for reference.
1) Depth-Normal Estimation: We trigger the growth pro-
cedure at a certain training iteration (in our implementation,
at t = 5000 iterations). At that point, we leverage multi-
view geometry from the input images to estimate dense depth
and surface normal maps for each training view. For this
purpose, we can employ a robust multi-view stereo technique
or leverage the current state of the model itself. In our
implementation, we found it effective to use a plane-guided
depth estimation similar to the one in PGSR [14]: by enforcing
local planarity and photometric consistency across views, we
compute a dense depth map Di(u, v) and normal map Ni(u, v)
for each image Vi (where (u, v) are pixel coordinates). This
process is akin to a one-pass stereo reconstruction from the
images, and it benefits from the fact that by iteration 5000 our

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
6
Algorithm 1 Grid Growth Strategy
Require: Training views {Vi}, Depth/normal model, LoD
octree, θthresh, sdown, Current iteration t
Ensure: Updated LoD structure
1: if t = 5000 then
▷Trigger growth at iteration 5000
2:
for each view Vi ∈{Vi} do
3:
Step 1: Generate depth/normal maps
4:
Di, Ni ←DepthNormalEstimation(Vi)
5:
Step 2: Filter observations
6:
for each pixel p in Di do
7:
v ←ViewingDirection(p)
8:
n ←Ni(p)
9:
if arccos(|v · n|) > θthresh then
10:
Discard p
▷Remove grazing angles
11:
end if
12:
end for
13:
Step 3: Back-project point cloud
14:
Praw ←Backproject(Di, Vi.camera)
15:
Pfiltered ←UniformDownsample(Praw, sdown)
16:
Step 4: Insert into LoD
17:
for each point p ∈Pfiltered do
18:
for ℓ= 0 to Lmax do
▷Coarse-to-fine
insertion
19:
Grid ←FindGrid(p, LoD[ℓ].resolution)
20:
if Grid not occupied then
21:
Initialize
grid
with
center
at
voxel.center
22:
ni ←Normal(p, Ni)
23:
xi, yi ←OrthogonalTangents(np)
24:
R ←[xi, yi, ni]
▷Local frame
25:
Σ ←R · diag(σ2
t , σ2
t , σ2
n) · R⊤
26:
for j = 1 to k do
27:
InitializeGaussianGrid
28:
(position = Grid.center,
29:
covariance = Σ)
30:
grid.AddGaussian(Gj)
31:
end for
32:
LoD[ℓ].AddGrid(grid)
33:
break
▷Insert at coarsest possible
level
34:
end if
35:
end for
36:
end for
37:
end for
38: end if
model’s poses and coarse shape are already reasonably aligned
with the training images.
Once we have the depth Di and normal Ni for a view v,
we filter these observations to ensure we only add high-quality
points. We iterate over each pixel p = (u, v) and consider
the depth value d = Di(u, v) with its corresponding normal
n = Ni(u, v). We discard points that are likely erroneous or
not robust:
if arccos(|v · n|) > θthresh,
discard point.
(7)
if the normal n is not consistent with the viewing direction,
or if the depth confidence is low. After filtering, we obtain
a set of reliable surface points for each view. These points
densely sample the scene surfaces as seen from that view.
2) Aggressive Coarse-to-Fine Insertion: With the collected
3D surface points from all views, we proceed to aggressively
insert new Gaussians into our representation to cover these
surfaces. The goal is to fill in any gaps in the current Gaussian
set by using the depth-informed points as candidate locations
for new primitives. A naive approach might add a Gaussian
at the exact location of every point, but this could lead to an
unmanageable number of primitives and a lot of redundancy.
Instead, we insert points in a structured manner aligned with
our LoD octree. For each candidate point p (with normal
n), we traverse the LoD hierarchy from coarse to fine and
determine where to place Gaussians such that all levels are
appropriately populated:
LoD-Aware Placement: Starting at the coarsest level ℓ= 0,
we check if p falls inside an existing voxel at that level that
already contains a Gaussian. If not (meaning the coarse grid
had a hole at that location), we create a new Gaussian grid
cell at level 0 covering p and initialize it with k Gaussians
(distributed by the same offset scheme as before). We orient
the covariance of these new Gaussians according to the normal
n, i.e., we align each new Gaussian’s principal axis with n
so that they lie roughly tangent to the surface. We also set
the SDF values of these new Gaussians to f ≈0 initially to
integrate them smoothly into the model.
Σ = R


σ2
t
0
0
0
σ2
t
0
0
0
σ2
n

R⊤,
where R = [ai, bi, ni].
(8)
Here, σt and σn denote the Gaussian radii in the tangent
and normal direction. We set σt = 1.0 and σn = 0.1
to limit the expansion of the Gaussian along the normal.
This design allows each Gaussian to capture local surface
curvature and orientation, laying a geometry-aware foundation
for subsequent optimization stages.
We then proceed to level ℓ= 1: we find the child cell of the
previously considered cell that contains p, and similarly check
if that child exists. If not, we insert a new cell at level 1 with
Gaussians (again oriented by n). We repeat this process for
each level up to ℓ= Lmax. In essence, each point p can spawn
at most one new cell per level, ensuring that all scales have a
representation of that surface point if it was missing. If at some
level the cell already existed (because perhaps another nearby
point already triggered its creation or it was present from
initialization), we skip adding at that level to avoid duplicates.
Avoiding Redundancy: This coarse-to-fine insertion en-
sures hierarchical consistency: a newly added fine Gaussian
will also have support from coarser levels. By design, we
do not add a new Gaussian into an occupied voxel to avoid
redundant representations. This strategy contrasts with a naive
global insertion of points which could clump many Gaussians
in an area already represented. By checking occupancy at each
level, we keep the Gaussian distribution even and controlled.
Moreover, inserting from coarse to fine allows the model

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
7
Fig. 4. The Qualitative Results on the Mip-NeRF 360 dataset [54]. Our method demonstrates superior reconstruction quality, particularly in complex geometric
regions and fine structural details as highlighted by the red box.
to maintain approximate coverage at all scales, which is
important for both rendering.
After processing all candidate points in this manner, we
update the LoD octree data structure to include all newly
created cells and Gaussians. The result is a significantly
densified Gaussian representation that now explicitly covers
surfaces that were previously missed or under-sampled. We
call this an “aggressive” growth because it can substantially
increase the number of Gaussians in one go. However, because
we insert guided by actual geometry and enforce the octree
structure, the growth is targeted and does not lead to gratuitous
increase in regions that were already adequately modeled.
3) Progressive Growth and Pruning: After the aggressive
insertion step, we resume the end-to-end training of the model
with the expanded set of Gaussians. To handle the sudden
increase in parameters, we employ a progressive learning rate
schedule: we temporarily boost the learning rate for the newly
added Gaussians’ parameters while slightly reducing it for the
pre-existing ones, ensuring the new Gaussians quickly adjust
to integrate with the scene without disturbing the already
learned structure. Over time, these learning rates are unified
again as the new primitives become part of the model. This
strategy helps maintain training stability during growth.
As training continues, the new Gaussians will have their
SDF values and colors refined by the loss functions (Sec. IV-
C), and any initial errors in their placement will be corrected.
Because we inserted Gaussians at all LoD levels for each point,
the model can learn a consistent multi-scale representation of
newly added regions: coarse new Gaussians ensure that even
from distant viewpoints the region contributes, while fine ones
add detail up close. The LoD coupling also means that our
earlier interpolation scheme and SDF field remain continuous
despite the insertions, we added features at new grid vertices,
and our IDW interpolation naturally incorporates those when
querying points in those regions.
To keep the model efficient, we also perform pruning of
redundant or unhelpful Gaussians as training proceeds. In
particular, if a coarse Gaussian’s region has been completely
taken over by finer Gaussians, we can safely deactivate or
remove that coarse Gaussian. This pruning prevents double-
counting of contributions and focuses computation on the
needed primitives. Similarly, if any Gaussian satisfies both low
opacity (α < τα) or the signed distance (|SDF| > τsdf), we can
remove it from the model to streamline the representation. This
keeps the scene representation compact and clean, containing
mostly surface-relevant Gaussians.
C. Optimization and Loss Functions
With the representation and growth strategy defined, we now
describe the optimization objectives that guide the training of
DiGS. Our loss design follows the principle of treating geom-
etry as a first-class objective, while also ensuring high-quality
radiance reconstruction. We employ a set of complementary
losses: (1) geometric losses that supervise the SDF values and
enforce SDF properties, (2) a flattening loss that regularizes
Gaussian shape along surfaces, (3) an appearance loss for RGB
reconstruction, and (4) a combined objective that balances all
terms. Each loss term plays a specific role, and importantly,
they act in concert on our unified pipeline. This means, for
example, that improving geometric alignment via SDF losses
also immediately benefits the rendering, and vice versa. We
detail each category of loss below.
1) Geometric Losses: To supervise the SDF values attached
to Gaussians, we design losses that encourage those values to
reflect true surface distances and to satisfy the mathematical
properties of an SDF. First, we introduce an SDF center loss
LSDF that penalizes the absolute SDF value at any Gaussian
which is expected to lie on the surface. Concretely, for each
Gaussian i that corresponds to a measured surface point, we
add a term |f(pi) −0| to the loss. We determine which

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
8
TABLE I
QUANTITATIVE RESULTS OF CHAMFER DISTANCE (MM) ON DTU DATASET. OUR METHOD OUTPERFORMS EXISTING 3D GS AND NERF-BASED
APPROACHES, ACHIEVING STATE-OF-THE-ART ACCURATE RECONSTRUCTION QUALITY ON THE DTU DATASET. ”RED”, ”ORANGE” AND ”YELLOW”
DENOTE THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS.
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
Time
VolSDF
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
>12h
NeuS
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
>12h
Neuralangelo
0.37
0.72
0.35
0.35
0.87
0.54
0.53
1.29
0.97
0.73
0.47
0.74
0.32
0.41
0.43
0.61
>128h
SuGaR
1.47
1.33
1.13
0.61
2.25
1.71
1.15
1.63
1.62
1.07
0.79
2.45
0.98
0.88
0.79
1.33
1h
GSDF
0.59
0.94
0.46
0.38
1.30
0.77
0.73
1.59
1.29
0.76
0.59
1.22
0.38
0.52
0.51
0.80
>2h
Scaffold-GS
7.23
6.23
6.48
7.44
8.17
4.27
5.78
5.45
6.57
6.36
5.05
5.95
6.32
5.62
2.90
4.63
0.8h
Octree-GS
4.3
3,45
4,56
3.45
5.67
2.345
3.45
4.45
3.32
3.35
2.45
4.56
4.45
3.45
4.34
4.23
0.7h
2DGS
0.46
0.91
0.39
0.39
1.01
0.83
0.81
1.36
1.27
0.76
0.70
1.40
0.40
0.76
0.52
0.80
0.32h
GOF
0.50
0.82
0.37
0.37
1.12
0.74
0.73
1.18
1.29
0.68
0.77
0.90
0.42
0.66
0.49
0.74
2h
PGSR
0.34
0.58
0.29
0.29
0.78
0.58
0.54
1.01
0.73
0.51
0.49
0.69
0.31
0.37
0.38
0.53
1h
Ours
0.45
0.44
0.25
0.33
0.69
0.49
0.46
0.92
0.57
0.47
0.43
0.46
0.28
0.35
0.32
0.46
1.2h
TABLE II
QUANTITATIVE RESULTS OF RECONSTRUCTION ON TNT DATASET [55].
Scene
NeuS
Geo-Neus
Neuralangelo
2D GS
GOF
PGSR
Ours
Barn
0.29
0.33
0.70
0.36
0.44
0.66
0.68
Caterpillar
0.29
0.26
0.36
0.23
0.41
0.41
0.42
Courthouse
0.17
0.12
0.28
0.13
0.28
0.21
0.25
Ignatius
0.83
0.72
0.89
0.44
0.68
0.80
0.83
Meetingroom
0.24
0.20
0.32
0.16
0.28
0.29
0.28
Truck
0.45
0.45
0.46
0.26
0.58
0.60
0.61
Mean
0.38
0.35
0.50
0.30
0.46
0.50
0.51
TABLE III
PERFORMANCE COMPARISON OF NOVEL VIEW SYNTHESIS ON MIP-NERF
360 DATASET [54]. OUR METHOD HAS A STRONG PERFORMANCE.
Indoor Scenes
Outdoor Scenes
All Scenes
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
NeRF
26.84
0.790
0.370
21.46
0.458
0.515
24.15
0.624
0.443
Deep Blending
26.40
0.844
0.261
21.54
0.524
0.364
23.97
0.684
0.313
Instant-NGP
29.15
0.880
0.216
22.90
0.566
0.371
26.03
0.723
0.294
Mip-NeRF360
31.72
0.917
0.180
24.47
0.691
0.283
28.10
0.804
0.232
NeuS
25.10
0.789
0.319
21.93
0.629
0.600
23.74
0.720
0.439
3DGS
30.99
0.926
0.199
24.24
0.705
0.283
27.24
0.803
0.246
2DGS
30.39
0.923
0.183
24.33
0.709
0.284
27.03
0.804
0.239
SuGaR
29.44
0.911
0.216
22.76
0.631
0.349
26.10
0.771
0.283
GOF
30.80
0.928
0.167
24.76
0.742
0.225
27.78
0.835
0.196
PGSR
30.41
0.930
0.161
24.45
0.730
0.224
27.43
0.830
0.193
Ours
30.60
0.927
0.171
24.73
0.739
0.254
27.67
0.833
0.212
Gaussians should lie on the surface. Then:
LSDF-Center =
X
i
|f(pi)|2.
(9)
driving those f(pi) to zero. Next, we impose the Eikonal
loss [56] to ensure the learned SDF behaves like a true signed
Distance Field in the continuous space. The Eikonal term
enforces that the gradient magnitude of f(p) is 1 everywhere
(which is a defining property of distance fields). We then add:
LEikonal = Epix∼X (∥∇f(p)∥2 −1)2 .
(10)
which encourages |∇f| = 1. This regularizes the SDF
field to prevent pathological solutions (e.g., constant or zero
everywhere) and promotes smoothness in geometry. It also
couples with the SDF center loss: having f = 0 at surface
points and unit gradient means that nearby points will get
correct sign distances. Overall, these geometric losses provide
strong supervision of the SDF compared to prior methods: for
example, PGSR [14] used normal consistency losses indirectly,
whereas we directly learn the SDF which inherently encodes
surface normals (via ∇f) and distances. By integrating this
into our pipeline, we ensure multi-view geometric consistency
is achieved through the SDF representation itself, rather than
requiring separate multi-view consistency losses. This simpli-
fies the training and yields a more interpretable geometric
model as a byproduct. The overall SDF consistency can be
geometrically regularized to:
LSDF = λ4Lsdf-center + λ5LEikonal.
(11)
λ4 and λ5 are the parameters of the corresponding loss.
2) Flattening Loss for Gaussian Covariance: Each Gaus-
sian is an anisotropic ellipsoid parameterized by covariance
Σi = RiSiS⊤
i R⊤
i , where Ri ∈R3×3 is a rotation matrix
and Si
= diag(si1, si2, si3) scales along principal axes.
Following [13], the smallest scale direction defines the local
surface normal ni. Minimizing this smallest scale encourages
flattening along ni, promoting compact, planar Gaussians:
Ls = λ6
X
i
min(s4, s5, s6).
3) RGB Reconstruction Loss: For appearance, we use an
RGB reconstruction loss to train the color output of our model.
We render each training image via our differentiable splatting
(with SDF-based opacities) and compare to the ground truth
image. We denote by Ci(u, v) the color predicted by our model
for pixel (u, v) in view Vi, and CGT
i
(u, v) the corresponding
ground truth pixel color. We then define:
Lrgb = (1 −λ7)∥˜I −Ii∥1 + λ7 · SSIM(˜I, Ii).
(12)
where ˜I denotes the rendered image, ˜I denotes the ground
truth image and λ1 controls the trade-off.
4) Final Objective: The total loss is a weighted sum of the
above terms:
L = Lrgb + Ls + Lgeo + LSDF.
(13)
V. EXPERIMENT
A. Experimental Setup
Datasets and Baselines. We evaluate our method on three
diverse datasets: DTU [57], Mip-NeRF 360 [54], TnT [55].
These datasets cover a range of 3D scene complexities, provid-
ing a comprehensive benchmark for reconstruction accuracy

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
9
Fig. 5. The Qualitative Results on DTU dataset. Our method has more complete and accurate reconstruction results in scenes scan40, scan55, and scan69,
and better reconstruction details in scan37, scan65, and scan110.
and visual fidelity. For comparison, we select a set of state-
of-the-art reconstruction methods: NeRF [18], NeuS [19],
Neuralangelo [35], Deep Blending [58], INGP [34], 3DGS
[1], Scaffold-gs [40], Octree-gs [53], SuGar [15], and PGSR
[14], each representing a notable approach in 3D scene surface
reconstruction.
Evaluation Metrics. We assess the performance of our
method using several standard reconstruction and rendering
metrics: Chamfer Distance, F1-score, along with image quality
metrics including PSNR, SSIM, and LPIPS. These metrics
allow for a holistic evaluation of both geometric fidelity
and photorealistic rendering quality. Mesh reconstruction is
achieved using a TSDF-based extraction method [14], which
ensures consistency in the surface.

<!-- page 10 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
10
Fig. 6. The Qualitative Comparison of Reconstruction Results on the Truck, Ignatius, and Barn scenes from the TnT Dataset [55]. Our method demonstrates
superior fidelity in preserving geometric details and structural integrity. Notably, our approach achieves sharper edges, and more consistent surface continuity.
Implementation Details. The optimization process incorpo-
rates multiple loss components, including L1 loss, SSIM loss,
image photometric loss, Gaussian flattening loss, single-view
loss, and multi-view loss. The corresponding weight factors,
λ4 to λ7, are set to 0.01, 0.01, 100, and 0.05, respectively.
These weights are carefully tuned to balance the contributions
of each loss term, ensuring optimal convergence and faithful
reconstruction throughout the training process.
Our training follows the schedule outlined in 3D Gaussian
Splatting, with a total of 30,000 epochs. The densification
process begins at epoch 1500 and concludes at epoch 15,000,
facilitating gradual scene refinement. To enhance geomet-
ric accuracy in sparse regions, the depth-aggressive growth
strategy is activated at epoch 5,000. We employ a progres-
sive learning rate schedule for neural network components,
gradually increasing their learning capacity over time, while
maintaining a fixed learning rate for the grid positions to
preserve spatial stability throughout training.
All experiments are conducted on a single NVIDIA RTX
3090 GPU. For large-scale datasets that exceed the GPU
memory capacity, data loading is offloaded to the CPU. This
strategy mitigates GPU memory pressure, ensuring stable
training throughput and the ability to handle larger datasets
without compromising performance.
B. Reconstruction
Accuracy. We evaluate reconstruction accuracy on both
DTU [57] and Tanks & Temples (TnT). Table I reports Cham-
fer Distance on DTU, where our method achieves the lowest
error among all compared approaches, clearly outperforming
Gaussian-based 3DGS [1] and neural implicit baselines such
as NeRF [18] and NeuS [19]. Table II further presents F1
scores on TnT, which jointly capture precision and recall under
a strict distance threshold. Our method consistently achieves
the highest average F1, indicating both accurate alignment to
ground truth geometry and reliable coverage of fine details.
These results demonstrate that explicitly embedding signed
distance supervision into Gaussian primitives significantly
improves geometric fidelity across benchmarks.
Qualitative results corroborate the quantitative improve-
ments. As shown in Figures 4, 5,and 6, our reconstructions
exhibit sharper boundaries, cleaner topology, and better preser-
vation of thin structures compared to competing methods. In
contrast, baselines such as GOF [12] often display surface
drift, misalignments, or incomplete recovery, particularly in
regions with fine-scale details or weak textures. Our method
also generalizes robustly to real-world Mip-NeRF 360 scenes,
where it produces geometrically faithful surfaces despite com-
plex illumination and scale variations.
Completeness. Beyond accuracy, we assess the complete-
ness of reconstructed surfaces, which is particularly challeng-
ing in occluded or low-texture regions. Prior Gaussian-based
pipelines often rely on appearance-driven growth strategies, re-
sulting in uneven Gaussian distributions and noticeable holes.
In contrast, our geometry-guided aggressive grid growth adap-
tively expands Gaussians along surface-consistent directions,
ensuring dense coverage and robust scene reconstruction.
Figures 4 and 6 highlight this advantage: our method yields
reconstructions with more uniform Gaussian placement and
significantly fewer missing regions, even under severe occlu-
sions or texture sparsity. Competing methods frequently fail
to recover such areas, leaving gaps or fragmented structures,
while our approach produces coherent and watertight models.
These results confirm that the proposed growth mechanism
plays a critical role in achieving visually complete reconstruc-
tions, complementing the quantitative accuracy improvements.

<!-- page 11 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
11
Fig. 7. Qualitative ablation on DTU [57] and Mip-NeRF360 [54]. Rows from top to bottom show the model without geometry-guided grid growth, without
Eikonal loss, without SDF-center loss, and the full model. Left block shows reconstructed geometry with surface normals on DTU objects. Right block shows
novel view synthesis results on Mip-NeRF360 with zoom-in insets. Removing any component degrades edge sharpness, normal consistency, and coverage,
while the full model recovers thin structures and fills missing regions.
Fig. 8. The Process of Aggressive Grid Growth strategy. At 5000 iterations,
the Grid Growth strategy is executed.
C. Novel View Synthesis
For the evaluation of novel view synthesis quality, we adopt
the rigorous evaluation protocol established by 3DGS and
conduct a comprehensive assessment on the Mip-NeRF360
[54] benchmark dataset. This allows us to thoroughly evaluate
the proposed method’s performance in generating novel views
from sparse input perspectives. To ensure a balanced and fair
comparison, we benchmark our method against two categories
of state-of-the-art (SOTA) approaches: (1) methods that have
demonstrated exceptional performance in novel view synthesis
tasks and (2) surface reconstruction techniques that share
similar technical principles with our framework.
As detailed in Table III, our approach not only excels
in high-precision surface geometry reconstruction but also
achieves remarkable performance in terms of visual quality
metrics for novel view synthesis. This includes improvements
in image fidelity, detail retention, and depth consistency when
viewed from unseen viewpoints. The results clearly highlight
that our method balances geometric accuracy with photo-
realistic rendering quality, offering a more robust solution
for novel view synthesis compared to existing techniques.
These advancements are underpinned by the synergy between
our SDF-constrained 3D Gaussian representation and adaptive
grid growth strategy, which together enhance both geometric
consistency and visual realism.
D. Ablation Study
We evaluate three principal design choices of our method:
SDF-constrained 3D Gaussians enforced by an SDF-center
loss, Eikonal regularization for gradient and normal consis-
tency, and geometry-guided aggressive grid growth driven by
depth and normal cues. Quantitative results on DTU and Mip-
NeRF360 are reported in Table IV. Qualitative comparisons
are shown in Fig. 7, and the growth schedule is illustrated
in Fig. 8. Across both datasets, removing any single compo-
nent consistently degrades geometric accuracy and perceptual

<!-- page 12 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
12
TABLE IV
QUANTITATIVE ABLATION ON DTU [57] AND MIP-NERF360 [54]. THE FULL MODEL ACHIEVES THE BEST SCORES ON BOTH DATASETS, CONFIRMING
THE COMPLEMENTARY EFFECTS OF SDF ANCHORING, EIKONAL REGULARIZATION, AND GEOMETRY-GUIDED AGGRESSIVE GRID GROWTH.
DTU
Mip-NeRF360
Method
CD↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
W/o SDF-Center constraint
0.49
34.65
0.74
0.370
27.43
0.828
0.217
W/o Eikonal Regularization
0.51
34.18
0.68
0.361
27.33
0.827
0.220
W/o Aggressive Grid Growth
0.50
34.13
0.70
0.317
27.45
0.826
0.227
Full Model
0.46
35.63
0.81
0.219
27.67
0.833
0.212
quality, which indicates complementary effects among the
proposed modules.
a) SDF-constrained 3D Gaussians representation: Bind-
ing each Gaussian to the SDF zero level set through the
center prior provides geometric anchoring that suppresses
surface drift and floating artifacts. Without this loss, CD
on DTU increases from 0.46 to 0.49, PSNR decreases from
35.63 to 34.65, and SSIM drops from 0.81 to 0.74. On Mip-
NeRF360, PSNR declines from 27.67 to 27.43, SSIM from
0.833 to 0.828, and LPIPS rises from 0.212 to 0.217. Normal
visualizations in Fig. 7 reveal softened edges and small-scale
misalignments in high-curvature regions. These observations
show that SDF anchoring turns Gaussians into surface-attached
primitives rather than view-dependent volumetric proxies.
b) Eikonal regularization: The Eikonal term enforces
unit SDF gradients and yields well-behaved normals and a
thin, stable level set. Removing this regularization leads to
over-smoothed geometry. On DTU, CD rises to 0.51, SSIM
falls to 0.68, and PSNR decreases by 1.45 dB from 35.63 to
34.18. On Mip-NeRF360, PSNR reduces to 27.33, SSIM to
0.827, and LPIPS increases to 0.220. Visual evidence in Fig. 7
shows loss of crisp boundaries on the bench slats and the
tabletop rim together with locally inconsistent normals. These
results highlight the role of the Eikonal term in preserving fine
detail and enforcing a coherent surface.
c) Geometry-guided aggressive grid growth: We intro-
duce a one-shot aggressive expansion of the Gaussian grid
scheduled at 5k iterations and guided by depth residuals
and normal disagreement. In contrast to uniform or heuristic
upsampling, this policy allocates new primitives only where
geometry is under-explained, which rapidly covers unmodeled
regions while avoiding redundant capacity in flat areas. Dis-
abling this module reduces DTU SSIM from 0.81 to 0.70
and increases LPIPS from 0.219 to 0.317. On Mip-NeRF360,
PSNR drops from 27.67 to 27.45, SSIM from 0.833 to 0.826,
and LPIPS rises to 0.227. Fig. 7 shows incomplete coverage
and gaps along thin structures such as bench slats and in low-
texture ground regions when the growth policy is removed.
The data indicate that deciding where and when to grow
matters as much as the total amount of growth.
VI. CONCLUSION
We presented DiGS, a unified framework that embeds
signed distance supervision directly into Gaussian primitives
and integrates a geometry-guided grid growth strategy under a
multi-scale hierarchy, thereby transforming Gaussian splatting
into a geometry-preserving paradigm. By tightly coupling
geometry and appearance within a single optimization process,
DiGS achieves accurate and complete surface reconstruction
while retaining high-fidelity rendering. Extensive experiments
validate that this joint design yields sharp structural details,
robust surface consistency, and efficient scalability across di-
verse datasets. We envision future extensions of DiGS toward
explicit mesh integration, dynamic scene modeling, and data
compression, further broadening its applicability in real-time,
geometry-aware 3D reconstruction.
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[2] H. Zhai, X. Zhang, B. Zhao, H. Li, Y. He, Z. Cui, H. Bao, and G. Zhang,
“Splatloc: 3d gaussian splatting-based visual localization for augmented
reality,” IEEE Transactions on Visualization and Computer Graphics,
2025.
[3] H. Song, “Toward realistic 3d avatar generation with dynamic 3d
gaussian splatting for ar/vr communication,” in 2024 IEEE Conference
on Virtual Reality and 3D User Interfaces Abstracts and Workshops
(VRW).
IEEE, 2024, pp. 869–870.
[4] Y. Jiang, C. Yu, T. Xie, X. Li, Y. Feng, H. Wang, M. Li, H. Lau, F. Gao,
Y. Yang et al., “Vr-gs: A physical dynamics-aware interactive gaussian
splatting system in virtual reality,” in ACM SIGGRAPH 2024 Conference
Papers, 2024, pp. 1–1.
[5] H. Lian, K. Liu, R. Cao, Z. Fei, X. Wen, and L. Chen, “Integration
of 3d gaussian splatting and neural radiance fields in virtual reality fire
fighting,” Remote Sensing, vol. 16, no. 13, p. 2448, 2024.
[6] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, “Driv-
inggaussian: Composite gaussian splatting for surrounding dynamic au-
tonomous driving scenes,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2024, pp. 21 634–21 643.
[7] M. Khan, H. Fazlali, D. Sharma, T. Cao, D. Bai, Y. Ren, and B. Liu,
“Autosplat: Constrained gaussian splatting for autonomous driving scene
reconstruction,” arXiv preprint arXiv:2407.02598, 2024.
[8] G. Hess, C. Lindstr¨om, M. Fatemi, C. Petersson, and L. Svensson,
“Splatad: Real-time lidar and camera rendering with 3d gaussian splat-
ting for autonomous driving,” arXiv preprint arXiv:2411.16816, 2024.
[9] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, and Y. He, “3d gaussian
splatting as new era: A survey,” IEEE Transactions on Visualization and
Computer Graphics, 2024.
[10] K. Cheng, X. Long, K. Yang, Y. Yao, W. Yin, Y. Ma, W. Wang, and
X. Chen, “Gaussianpro: 3d gaussian splatting with progressive propa-
gation,” in Forty-first International Conference on Machine Learning,
2024.
[11] T. Wu, Y.-J. Yuan, L.-X. Zhang, J. Yang, Y.-P. Cao, L.-Q. Yan, and
L. Gao, “Recent advances in 3d gaussian splatting,” Computational
Visual Media, vol. 10, no. 4, pp. 613–642, 2024.
[12] Z. Yu, T. Sattler, and A. Geiger, “Gaussian opacity fields: Efficient adap-
tive surface reconstruction in unbounded scenes,” ACM Transactions on
Graphics (TOG), vol. 43, no. 6, pp. 1–13, 2024.
[13] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting
for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024
conference papers, 2024, pp. 1–11.

<!-- page 13 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
13
[14] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu,
H. Bao, and G. Zhang, “Pgsr: Planar-based gaussian splatting for
efficient and high-fidelity surface reconstruction,” IEEE Transactions on
Visualization and Computer Graphics, 2024.
[15] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting
for efficient 3d mesh reconstruction and high-quality mesh rendering,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 5354–5363.
[16] X. Lyu, Y.-T. Sun, Y.-H. Huang, X. Wu, Z. Yang, Y. Chen, J. Pang,
and X. Qi, “3dgsr: Implicit surface reconstruction with 3d gaussian
splatting,” ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp.
1–12, 2024.
[17] M. Yu, T. Lu, L. Xu, L. Jiang, Y. Xiangli, and B. Dai, “Gsdf: 3dgs
meets sdf for improved neural rendering and reconstruction,” Advances
in Neural Information Processing Systems, vol. 37, pp. 129 507–129 530,
2024.
[18] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[19] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “Neus:
Learning neural implicit surfaces by volume rendering for multi-view
reconstruction,” arXiv preprint arXiv:2106.10689, 2021.
[20] L. Yariv, J. Gu, Y. Kasten, and Y. Lipman, “Volume rendering of neural
implicit surfaces,” Advances in Neural Information Processing Systems,
vol. 34, pp. 4805–4815, 2021.
[21] X. Zhang, R. Yu, and S. Ren, “Neural implicit representations for
multi-view surface reconstruction: A survey,” IEEE Transactions on
Visualization and Computer Graphics, 2025.
[22] M. Oechsle, S. Peng, and A. Geiger, “Unisurf: Unifying neural implicit
surfaces and radiance fields for multi-view reconstruction,” in Proceed-
ings of the IEEE/CVF international conference on computer vision,
2021, pp. 5589–5599.
[23] Z. Yu, S. Peng, M. Niemeyer, T. Sattler, and A. Geiger, “Monosdf:
Exploring monocular geometric cues for neural implicit surface recon-
struction,” Advances in neural information processing systems, vol. 35,
pp. 25 018–25 032, 2022.
[24] Q. Fu, Q. Xu, Y. S. Ong, and W. Tao, “Geo-neus: Geometry-consistent
neural implicit surfaces learning for multi-view reconstruction,” Ad-
vances in Neural Information Processing Systems, vol. 35, pp. 3403–
3416, 2022.
[25] H. Li, X. Yang, H. Zhai, Y. Liu, H. Bao, and G. Zhang, “Vox-
surf: Voxel-based implicit surface representation,” IEEE Transactions on
Visualization and Computer Graphics, vol. 30, no. 3, pp. 1743–1755,
2022.
[26] L. Wang, Y.-T. Liu, J. Yang, W. Chen, X. Meng, B. Yang, J. Li, and
L. Gao, “Hsdf: Hybrid sign and distance field for neural representation of
surfaces with arbitrary topologies,” IEEE Transactions on Visualization
and Computer Graphics, vol. 31, no. 9, pp. 5215–5228, 2024.
[27] S. Ye, Y. Hu, M. Lin, Y.-H. Wen, W. Zhao, Y.-J. Liu, and W. Wang,
“Indoor scene reconstruction with fine-grained details using hybrid
representation and normal prior enhancement,” IEEE Transactions on
Visualization and Computer Graphics, 2024.
[28] B. Wang, Z. Yu, B. Yang, J. Qin, T. Breckon, L. Shao, N. Trigoni, and
A. Markham, “Rangeudf: Semantic surface reconstruction from 3d point
clouds,” arXiv preprint arXiv:2204.09138, 2022.
[29] C. Zhang, W. Su, Q. Xu, X. Liao, and W. Tao, “Pg-neus: Robust and
efficient point guidance for multi-view neural surface reconstruction,”
IEEE Transactions on Visualization and Computer Graphics, 2024.
[30] X.-K. Xiang, Y.-J. Yuan, W.-B. Hu, Y.-T. Liu, Y.-W. Ma, and L. Gao,
“Pgt-neus: Progressive-growing tri-plane representation for neural sur-
face reconstruction,” IEEE Transactions on Visualization and Computer
Graphics, 2025.
[31] W. Su, C. Zhang, Q. Xu, and W. Tao, “Psdf: Prior-driven neural implicit
surface learning for multi-view reconstruction,” IEEE Transactions on
Visualization and Computer Graphics, 2024.
[32] L. Yang, B. Deng, and J. Zhang, “Scalable and high-quality neural
implicit representation for 3d reconstruction,” IEEE Transactions on
Visualization and Computer Graphics, 2025.
[33] B. Wang, L. Chen, and B. Yang, “Dm-nerf: 3d scene geometry
decomposition and manipulation from 2d images,” arXiv preprint
arXiv:2208.07227, 2022.
[34] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM transactions on
graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.
[35] Z. Li, T. M¨uller, A. Evans, R. H. Taylor, M. Unberath, M.-Y. Liu, and
C.-H. Lin, “Neuralangelo: High-fidelity neural surface reconstruction,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 8456–8465.
[36] X. Liu, Q. Zhang, X. Huang, Y. Feng, G. Zhou, and Q. Wang, “H
{2}
o-nerf: Radiance fields reconstruction for two-hand-held objects,” IEEE
Transactions on Visualization and Computer Graphics, 2025.
[37] L. Yariv, P. Hedman, C. Reiser, D. Verbin, P. P. Srinivasan, R. Szeliski,
J. T. Barron, and B. Mildenhall, “Bakedsdf: Meshing neural sdfs
for real-time view synthesis,” in ACM SIGGRAPH 2023 Conference
Proceedings, 2023, pp. 1–9.
[38] M.-J. Rakotosaona, F. Manhardt, D. M. Arroyo, M. Niemeyer, A. Kundu,
and F. Tombari, “Nerfmeshing: Distilling neural radiance fields into
geometrically-accurate 3d meshes,” in 2024 international conference on
3D vision (3DV).
IEEE, 2024, pp. 1156–1165.
[39] E. Ramon, G. Triginer, J. Escur, A. Pumarola, J. Garcia, X. Gir´o-
i Nieto, and F. Moreno-Noguer, “H3d-net: Few-shot high-fidelity 3d
head reconstruction,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2021, pp. 5620–5629.
[40] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.
[41] G. Fang and B. Wang, “Mini-splatting: Representing scenes with a
constrained number of gaussians,” in European Conference on Computer
Vision.
Springer, 2024, pp. 165–181.
[42] ——, “Mini-splatting2: Building 360 scenes within minutes via aggres-
sive gaussian densification,” arXiv preprint arXiv:2411.12788, 2024.
[43] H. Chen, C. Li, and G. H. Lee, “Neusg: Neural implicit surface
reconstruction with 3d gaussian splatting guidance,” arXiv preprint
arXiv:2312.00846, 2023.
[44] D. Li, S.-S. Huang, and H. Huang, “Mpgs: Multi-plane gaussian splat-
ting for compact scenes rendering,” IEEE Transactions on Visualization
and Computer Graphics, 2025.
[45] Z. Bao, G. Liao, K. Zhou, K. Liu, Q. Li, and G. Qiu, “Loopsparsegs:
Loop based sparse-view friendly gaussian splatting,” IEEE Transactions
on Image Processing, 2025.
[46] R. Fan, J. Wu, X. Shi, L. Zhao, Q. Ma, and L. Wang, “Fov-gs:
Foveated 3d gaussian splatting for dynamic scenes,” IEEE Transactions
on Visualization and Computer Graphics, 2025.
[47] Z. Fan, S.-S. Huang, Y. Zhang, D. Shang, J. Zhang, Y. Guo, and
H. Huang, “Rgavatar: Relightable 4d gaussian avatar from monocular
videos,” IEEE Transactions on Visualization and Computer Graphics,
2025.
[48] Y. Wang, X. Wei, M. Lu, and G. Kang, “Plgs: Robust panoptic lifting
with 3d gaussian splatting,” IEEE Transactions on Image Processing,
2025.
[49] Y. Wang, J. Wang, R. Gao, Y. Qu, W. Duan, S. Yang, and Y. Qi, “Look
at the sky: Sky-aware efficient 3d gaussian splatting in the wild,” IEEE
Transactions on Visualization and Computer Graphics, 2025.
[50] K. Tang, S. Yao, and C. Wang, “ivr-gs: Inverse volume rendering
for explorable visualization via editable 3d gaussian splatting,” IEEE
Transactions on Visualization and Computer Graphics, 2025.
[51] X. Tong, T. Shao, Y. Weng, Y. Yang, and K. Zhou, “As-rigid-as-
possible deformation of gaussian radiance fields,” IEEE Transactions
on Visualization and Computer Graphics, 2025.
[52] Y. Shi, D. Rong, B. Ni, C. Chen, and W. Zhang, “Garf: Geometry-aware
generalized neural radiance field,” arXiv preprint arXiv:2212.02280,
2022.
[53] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, “Octree-gs: To-
wards consistent real-time rendering with lod-structured 3d gaussians,”
arXiv preprint arXiv:2403.17898, 2024.
[54] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 5470–5479.
[55] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics, vol. 36, no. 4, 2017.
[56] A. Gropp, L. Yariv, N. Haim, M. Atzmon, and Y. Lipman, “Im-
plicit geometric regularization for learning shapes,” arXiv preprint
arXiv:2002.10099, 2020.
[57] H. Aanæs, R. R. Jensen, G. Vogiatzis, E. Tola, and A. B. Dahl,
“Large-scale data for multiple-view stereopsis,” International Journal
of Computer Vision, vol. 120, pp. 153–168, 2016.
[58] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Bros-
tow, “Deep blending for free-viewpoint image-based rendering,” ACM
Transactions on Graphics (ToG), vol. 37, no. 6, pp. 1–15, 2018.
