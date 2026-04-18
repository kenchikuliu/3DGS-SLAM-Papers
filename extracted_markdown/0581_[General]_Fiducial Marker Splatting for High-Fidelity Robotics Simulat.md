<!-- page 1 -->
Fiducial Marker Splatting for High-Fidelity Robotics Simulations
Diram Tabaa
Gianni Di Caro
Carnegie Mellon University
{dtabaa, gdicaro}@andrew.cmu.edu
Figure 1. Proof-of-concept greenhouse environment constructed from 2D Gaussian splats trained on cucumber greenhouse images. Our
fiducial markers, generated without prior splatting-based training, are placed within the scene to support high-fidelity robotics simulation.
Abstract
High-fidelity 3D simulation is critical for training mo-
bile robots, but its traditional reliance on mesh-based rep-
resentations often struggle in complex environments, such
as densely packed greenhouses featuring occlusions and
repetitive structures. Recent neural rendering methods, like
Gaussian Splatting (GS), achieve remarkable visual realism
but lack flexibility to incorporate fiducial markers, which
are essential for robotic localization and control. We pro-
pose a hybrid framework that combines the photorealism
of GS with structured marker representations.
Our core
contribution is a novel algorithm for efficiently generating
GS-based fiducial markers (e.g., AprilTags) within cluttered
scenes. Experiments show that our approach outperforms
traditional image-fitting techniques in both efficiency and
pose-estimation accuracy.
We further demonstrate the framework’s potential in a
greenhouse simulation. This agricultural setting serves as
a challenging testbed, as its combination of dense foliage,
similar-looking elements, and occlusions pushes the limits
of perception, thereby highlighting the framework’s value
for real-world applications.
1. Introduction
Autonomous mobile robots have been increasingly adopted
in real-world scenarios over the past few years. This adop-
tion has been seen in diverse fields such as logistics [10, 44],
manufacturing [14, 39], and agriculture [17, 46]. A cen-
tral theme across all these applications are traditional mo-
bile robotics challenges, including mapping and localiza-
tion [8, 31], path planning [21, 26, 36], and navigation
[13, 37]. Due to safety and cost constraints, these prob-
lems are typically first addressed in a simulated environ-
ment prior to real-world testing, which gives simulation a
central role in robotics research.
Accordingly,
robotic simulation environments have
evolved substantially.
In particular, 3D simulators like
Gazebo [20] have improved the ability to test robotic solu-
tions by emulating real-time visual input from the environ-
ment, whether in the form of RGB images, LiDAR data or
depth maps. This modeling commonly relies on using 3D
meshes to represent objects (e.g. chairs, walls) which are
1
arXiv:2508.17012v1  [cs.CV]  23 Aug 2025

<!-- page 2 -->
manually created using CAD software [42, 43] or through
3D reconstruction methods [3, 45].
In agriculture, such
simulators have been used to model robotic tasks in green-
houses [16, 40] and open fields [38].
Although mesh-based methods have expanded the scope
of robotics simulation, they still lack sufficient realism. This
is due to the reduced face count needed for real-time emula-
tion and the nonrealistic imagery produced by raster-based
methods. Although increased computational power from
consumer graphics hardware has led to solutions such as
real-time ray tracing, a core issue with mesh-based meth-
ods remains the effort required to produce these meshes.
This is especially true for objects with complex geometries,
such as plants and trees, whether through manual design or
procedural algorithms. Recent generative AI models for 3D
mesh generation, such as MeshGPT [35] and MeshDiffu-
sion [25], are still incapable of accurately modeling com-
plex non-convex structures.
Recent breakthroughs in radiance field rendering have
enabled models that can faithfully capture real-world scenes
using only a collection of reference images. Specifically,
3D Gaussian Splatting [18] allows for the creation of real-
istic radiance fields that can be rendered in real time and
support inference from novel camera poses not included in
the training set. These advances have led to widespread
adoption in robotics simulation [5, 23, 47] because of their
ability to model unconstrained, true-to-life scenes. How-
ever, the novelty of this representation means that little work
has been done to integrate classical scene elements into ra-
diance fields. This limitation impacts certain applications,
particularly those involving fiducial markers, which are crit-
ical for localization in complex environments.
In this paper, we introduce a novel approach to using
Gaussian Splatting in robotics simulation. We demonstrate
the ability to integrate elements of classic robotics simula-
tion into Gaussian Splatting representations by presenting a
new algorithm to generate fiducial markers using Gaussian
primitives. We show that this method outperforms classic
Gaussian Splatting fitting approaches in both efficiency and
the visual quality of the generated markers. To validate our
framework, we present a proof-of-concept simulation in a
greenhouse environment explicitly addressing localization
tasks. This agricultural setting was chosen specifically as it
represents a challenging domain with dense clutter and vi-
sual ambiguity, highlighting the potential of our framework
to enable realistic and flexible simulation for a broad range
of robotics research. Our contributions are threefold:
• We present a universal fiducial marker representation
based on Gaussian primitives and a novel algorithm to
generate these markers.
• We show that our algorithm outperforms standard Gaus-
sian Splatting approaches in both efficiency and recogniz-
ability.
• We demonstrate the application of this framework to agri-
cultural robotics simulations through a proof-of-concept.
2. Related Work
Our discussion of related work is divided into two sections
to provide context for the core motivations behind our con-
tributions. In 2.1, we review current work in radiance field
rendering and highlight the need for a framework that can
generate visual elements, such as fiducial markers, without
prior training. Subsequently, 2.2 surveys existing agricul-
tural simulation environments, showing how our work is
motivated by the need to combine the photorealistic ren-
dering of radiance field methods with the comprehensive
capabilities of robotics simulation.
2.1. Radiance field rendering
Radiance fields model scenes as continuous functions that
describe how light rays emanate from objects in a scene.
Classical methods [11, 22] represented images as slices of
the radiance field, but these approaches required a large
number of images for reconstruction.
NeRF [27] intro-
duced a breakthrough by parameterizing radiance fields
with neural networks, thereby reducing storage require-
ments by leveraging the inference capability of trained mod-
els.
Subsequent work [1, 2, 28] focused on improving
NeRF in terms of resolution and rendering speed, but these
methods remained constrained by the computational cost of
volumetric sampling. More recently, 3D Gaussian Splat-
ting (3DGS) [18] addressed this limitation by representing
radiance fields with Gaussian primitives that can be effi-
ciently rasterized, enabling real-time radiance field render-
ing. Building on this idea, methods such as 2D Gaussian
Splatting [15] and SuGaR [12] extended 3DGS to achieve
more accurate surface modeling.
Despite these advances, radiance field methods still re-
quire per-scene training, which is not only time-consuming
but also highly sensitive to input sparsity and camera pose
accuracy. PixelSplat [4] mitigates this issue by pretraining a
network to infer Gaussian primitives directly from two im-
ages in a single feed-forward pass. MVSplat [6] extends this
approach to multiple sparse input images. However, these
methods still depend on large-scale pretraining and, more
importantly, assume a general setting where the target scene
function is unknown. To the best of our knowledge, no prior
work has addressed this challenge for simple geometric el-
ements in a reconstruction-independent manner, which is
particularly relevant in the context of fiducial markers.
2.2. Simulation in Agricultural Robotics
The simulation of botanical entities, a cornerstone of us-
ing simulation in modern agricultural robotics, has a rich
history in computer graphics. The field was pioneered by
procedural, rule-based methods, most notably the L-system
2

<!-- page 3 -->
introduced by Lindenmayer [24]. Originally a model for
cellular development, the L-system’s recursive grammar
proved highly effective at simulating the branching growth
patterns of plants. While foundational, the deterministic
nature of early L-systems struggled to capture the inherent
randomness of natural forms.
This limitation spurred the development of alternative
procedural techniques, including the use of fractals to gen-
erate self-similar structures [30] and particle systems to
model dynamic elements like leaves and blossoms [33]. A
different school of thought moved beyond pure procedu-
ralism, with some researchers focusing on creating mod-
els grounded in detailed botanical measurements [7], while
others prioritized plausible visual realism over strict biolog-
ical accuracy, as demonstrated by Weber et al. [41], who
used parameterized conic structures to create realistic trees.
These foundational techniques established the diverse ap-
proaches to plant modeling that underpin the high-fidelity
simulators used today.
In the robotics context, simulators such as Gazebo
mostly rely on mesh-based environment modelling, either
curated by manual creation of CAD models or by utilizing
procedural algorithms to automatically create them. In the
specific agricultural context, GroIMP [19] and OpenAlea
[32] presented open-source frameworks for procedural gen-
eration of 3D plant meshes. Although those frameworks
provided flexibility to generate heterogenous collections of
plant meshes, they are still limited by the scope of manu-
ally adjusted generation parameters. In addition, while such
models have been used in robotics simulation for tasks like
navigation, they are rarely useful when it comes to creat-
ing realistic simulation environments that enable sim-to-real
transfer, especially in vision-based tasks, such as fruit iden-
tification and counting.
3. Fiducial Marker Splatting
In this section, we introduce our method for representing
fiducial markers with Gaussian primitives without relying
on prior splatting-based training. The section is organized
into three parts. First, we briefly describe the specific repre-
sentation employed (2DGS). Second, we show how rectan-
gles can be approximated with Gaussian primitives. Finally,
we demonstrate how fiducial markers can be decomposed
into a minimal set of primitives for compact representation.
3.1. Preliminary: 2D Gaussian Splatting
2D Gaussian Splatting (2DGS) [15] is a method for mod-
eling and reconstructing geometrically accurate radiance
fields from multi-view images. The core idea of 2DGS is to
represent the 3D scene as a collection of 2D oriented planar
Gaussian splats (i.e. ellipses). Unlike 3D Gaussians, these
2D primitives provide a more view-consistent geometry and
are intrinsically better for representing surfaces. Each 2D
Gaussian primitive is defined by a set of parameters: a cen-
ter point pk, two principal tangent vectors tu and tv which
define the orientation of the ellipse in the 3D space, and
two scaling factors, su and sv, which control the variance
or shape of the elliptical splat. Additionally, each primitive
has an associated color ck and opacity αk.
The final color of a pixel is rendered by alpha blending
the 2D Gaussian primitives that project onto it. The prim-
itives are sorted from front to back along the viewing ray.
The color C for a pixel is computed by the following alpha
blending formula:
C =
X
k∈K
ckα′
k
k−1
Y
j=1
(1 −α′
j)
where K is the set of sorted Gaussian indices, ck is the
color of the k-th Gaussian, and α′
k is the opacity of the k-
th Gaussian modulated by its Gaussian function evaluated
at the pixel location. This formulation allows for differen-
tiable rendering, which is key for optimizing the parameters
of the Gaussian primitives to reconstruct the scene.
3.2. Gaussian Approximation of Fiducial Markers
Piecewise-constant planar marker.
Rectilinear fiducial
markers (e.g., AprilTags [29]) admit a decomposition into
axis-aligned rectangles in a local planar coordinate frame.
This yields a closed-form, piecewise-constant function from
coordinates to grayscale intensity.
Let {Ri}n
i=1 be pairwise-disjoint, axis-aligned rectan-
gles whose union defines the marker domain Ω⊂R2:
Ri = [ax
i , bx
i ] × [ay
i , by
i ],
(1)
Ri ∩Rj = ∅
∀i ̸= j,
(2)
n
[
i=1
Ri = Ω.
(3)
Define M : Ω→[0, 1] (grayscale) by
M(x) =
n
X
i=1
ci χRi(x),
(4)
where ci ∈[0, 1] and
χRi(x) =
(
1,
x ∈Ri,
0,
otherwise.
(5)
Smooth approximation for 2D Gaussian splatting.
To
obtain a differentiable representation suitable for 2D Gaus-
sian splatting (2DGS), we approximate each χRi with a fi-
nite mixture of anisotropic 2D Gaussians that concentrate
mass near rectangle edges while retaining a solid interior.
3

<!-- page 4 -->
Figure 2. Overview of the fiducial marker splatting pipeline. A binary marker is first partitioned into connected components via DFS,
which are then tessellated into rectilinear polygons. Each polygon is processed with the minimal rectilinear partition algorithm (Sec. 3.3)
to obtain rectangles. These rectangles are then parameterized and converted into rectangular Gaussian approximators (Sec. 3.2), yielding
the final set of 2D Gaussian primitives.
Consider a rectangle with center c ∈R2, half-sizes
sx, sy > 0, refinement levels L ∈N, and density modifier
ρ ≥1.
Level 0 (interior seed). Place a single anisotropic Gaus-
sian at the center:
µ0 = c,
(6)
Σ0 = S0S⊤
0 ,
S0 = diag
sx
γ , sy
γ

,
(7)
with γ = 3.0 serving as the 2DGS render cutoff hyperpa-
rameter.
Levels l = 1, . . . , L −1 Define the level-dependent off-
set
dl =
 sx(1 −2−l), sy(1 −2−l)

,
(8)
and the first-quadrant “corner” point pl = c + dl. Place a
corner Gaussian at pl with
µl = pl,
(9)
Σl = diag
 σ2
xl, σ2
yl

,
(10)
where
σxl = sx
γ 2l ,
σyl = sy
γ 2l .
(11)
Arms toward the corner. Populate two orthogonal arms
meeting at pl:
• Horizontal arm: place 2 l−1 Gaussians with means µi =
 cx + ox
i , cy + dly

where ox
i ∈[0, dlx) are uniformly
spaced, and covariances Σi = diag

σ2
∥,i, σ2
⊥,l

with
σ∥,i = (sx −ox
i )/γ and σ⊥,l = σyl.
• Vertical arm: symmetrically, place 2 l−1 Gaussians with
means µj =
 cx + dlx, cy + oy
j

where oy
j ∈[0, dly), and
covariances Σj = diag

σ2
⊥,l, σ2
∥,j

with σ∥,j = (sy −
oy
j )/γ. and σ⊥,l = σxl.
Mirror the first-quadrant set across the axes through c to
populate all four sides at level l. The final mixture aggre-
gates components from levels l = 0, . . . , L−1. To ensure a
visually solid interior, upweight early-level components by
a factor ρ (equivalently, replicate them ρ times).
4

<!-- page 5 -->
Category
String Length
QR Version
Dimensions (px)
Small
20
2
25 × 25
Medium
100
5
37 × 37
Large
250
10
57 × 57
Huge
1000
22
105 × 105
Table 1. QR Code categories based on string length and corre-
sponding QR versions.
3.3. Rectangular Partitioning of Fiducial Markers
As discussed in the previous subsection, reducing the num-
ber of 2D Gaussian primitives at the level of individual rect-
angles is essential to prevent performance degradation. Be-
yond this, additional optimization can be achieved at the
scale of the entire marker. In particular, the primitive count
can be reduced by minimizing the number of rectangles
generated during partitioning, since fewer, larger rectangles
eliminate redundant inner edges.
To accomplish this, the fiducial marker is first parti-
tioned into connected components via Depth First Search
(DFS) based on pixel color values. Each connected compo-
nent is then converted into a rectilinear polygon, potentially
with holes. For each rectilinear polygon, the minimal rec-
tilinear partition algorithm of Ferrari et al. [9] is applied.
This algorithm identifies the largest independent set of non-
intersecting, axis-parallel concave vertex chords, partitions
the polygon accordingly, and then further partitions any re-
maining concave vertices, which are guaranteed not to con-
nect to one another after the initial step. A detailed descrip-
tion of the algorithm is provided in the Appendix. Finally,
the resulting rectangles are parameterized by their center
coordinates and sx, sy scales, yielding the rectangular ap-
proximators introduced in Section 3.2. The overall pipeline
is illustrated in Figure 2.
4. Experiments
4.1. Experimental Setup
Datasets.
Since no prior work exists on this problem, an
evaluation dataset was constructed using AprilTags [29] and
QR Codes. For AprilTags, five tags were selected from the
36h11 standard. For QR Codes, five tags were generated
for each size category, where the size was determined by
the length of randomly generated fixed-length strings, as
defined in Table 1. To obtain the camera poses required
for 2DGS training, four distinct viewpoint collections were
designed (Figure 3) in order to evaluate the sensitivity of the
baseline to viewpoint quality. Combining the generated tags
with these viewpoint sets resulted in 20 trainable scenes for
AprilTags and 80 for QR Codes, for a total of 100 distinct
scenes stored in COLMAP [34] format. Finally, Blender
Figure 3. The four viewpoint collections designed for 2DGS train-
ing. These sets were constructed to evaluate the sensitivity of the
baseline to viewpoint quality and sparsity
was used to model the scenes by attaching each tag as a
texture to a 2×2 plane and rendering images from the pre-
defined viewpoints.
Evaluation Metrics.
For evaluation, we assess the read-
ability of the rasterized 2D Gaussian Fitted tags by mea-
suring the maximum viewing angle at which the QR code
remains decodable from a fixed distance and azimuth. We
also report standard Novel view synthesis metrics, that is
PSNR/SSIM/LPIPS, one 20 randomly selected test view-
points. In addition, we measure the time required to gen-
erate the primitives and the total number of primitives pro-
duced. The goal of these metrics to show how our method
can yield comparable, if not better results to trained meth-
ods with a major reduction on inference time and memory
footprint through a reduction in gaussian primitives count.
Implementation Details.
We implement the fiducial
marker pipeline using Shapely and NumPy. For rasteriza-
tion, we employ the off-the-shelf 2D Gaussian splatting ras-
terizer by Huang et al. [15], and unless stated otherwise, the
same hyperparameters are used for training all comparison
baselines. Training is performed for 7,000 iterations on an
RTX A6000 GPU. All rasterizations are likewise executed
on this GPU to ensure fairness.
4.2. Results
Runtime and Primitive Count
We report the average
number of Gaussian primitives generated by each method
across marker categories, along with the corresponding
5

<!-- page 6 -->
Figure 4. Qualitative results of rasterized fiducial markers and QR codes under varying viewpoints. Our rectangular Gaussian
approximation produces stable and sharp renderings, while baseline methods exhibit artifacts such as blurring, distortions, or missing
regions (highlighted in red).
Figure 5.
Mean Gaussian primitive counts across categories,
showing our method achieves substantially lower counts than all
2DGS baselines.
Method
Mean Training time (s)
2DGS
3 views
423
5 views
392
9 views
386
17 views
371
Ours (no training)
1.04
Table 2.
Wall-clock comparison aggregated across categories.
“Training” is technically a misnomer for Ours, which is an algo-
rithmic construction without learning; we report its construction
time to emphasize that results are effectively instantaneous rela-
tive to 2DGS.
runtime for training / primitive generation. As shown in
Fig. 5, our method consistently achieves the lowest prim-
itive counts across all categories, with the gap becoming
more pronounced as marker size increases. For instance, on
HUGE markers, our approach reduces the primitive budget
by more than 40% compared to the strongest 2DGS base-
line. This reduction translates directly into faster inference,
as fewer primitives must be rasterized at test time.
Im-
portantly, our method maintains readability while provid-
ing lower runtime overhead, highlighting its efficiency in
both representation compactness and rendering cost. Con-
struction cost is likewise minimal: as summarized in Ta-
ble 2, 2DGS requires 423/392/386/371 s of training for
3/5/9/17 views, respectively, whereas our method per-
forms no learning and completes primitive construction in
1.04 s. This is a ∼380× reduction in setup time and, to-
gether with the smaller primitive budget, yields faster test-
time rendering without sacrificing readability.
Image Quality and Readability
In terms of readability,
we demonstrate that our method outperforms the baseline
even with more dense views (Table 3), and at a fraction
of the number of Gaussian primitives involved. Note that,
as QR complexity increases, the readability of our method
continues to improve, which also suggests that algorithms
that perform localization using AR tags would have no trou-
ble here since they operate on the same grid-structured,
high-contrast cues as QR codes.
It might be surprising to report mixed results in classical
image quality metrics (Table 4); however, this is expected
because 2DGS is trained directly on photographs and, in
our setup, is exposed to views that are very similar to the
test frames. As a result, 2DGS is optimized to reproduce
the captured pixels and thus attains higher PSNR/SSIM
when many views are available. Our approach, in contrast,
never uses the images during optimization and reconstructs
6

<!-- page 7 -->
QR Category
Ours
2DGS (3 Views)
2DGS (5 Views)
2DGS (9 Views)
2DGS (17 Views)
θdet (◦) ↑
θdecode (◦) ↑
θdet (◦)
θdecode (◦)
θdet (◦)
θdecode (◦)
θdet (◦)
θdecode (◦)
θdet (◦)
θdecode (◦)
small
80.0
80.0
73.8
60.0
79.4
68.2
81.6
75.4
81.0
79.0
medium
81.4
81.2
76.8
59.0
80.4
68.0
81.6
76.6
81.8
75.2
large
82.0
81.8
77.8
58.0
80.0
67.2
82.0
77.8
82.2
76.6
huge
84.0
82.2
77.8
56.8
79.4
65.4
82.2
72.0
82.6
75.4
Table 3. Maximum detection and decoding angles (in degrees). We report the maximum angle θ at which markers remain detectable
(θdet) and decodable (θdecode). The proposed method consistently yields higher angles across all QR categories, with stronger gains in
decoding robustness. Best results are highlighted in bold.
Category
Ours
2DGS (3 Views)
2DGS (5 Views)
2DGS (9 Views)
2DGS (17 Views)
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
AR Tag
27.97
0.83
0.05
18.41
0.90
0.12
30.38
0.98
0.04
35.03
0.99
0.02
38.70
0.99
0.02
QR Small
25.41
0.83
0.05
19.86
0.90
0.12
26.90
0.96
0.04
33.82
0.99
0.02
37.21
0.99
0.02
QR Medium
23.58
0.84
0.06
19.64
0.89
0.12
26.97
0.96
0.04
33.52
0.99
0.02
36.10
0.99
0.02
QR Large
21.25
0.85
0.10
20.74
0.90
0.11
27.83
0.97
0.03
33.02
0.99
0.02
34.88
0.99
0.02
QR Huge
18.80
0.88
0.08
20.62
0.91
0.09
27.33
0.97
0.03
31.04
0.99
0.02
32.52
0.99
0.02
Table 4. Novel View Synthesis We report PSNR/SSIM/LPIPS for our method and for 2DGS trained with 3/5/9/17 views. Because 2DGS
is trained on photographs and sees views highly similar to the test frames, its per-pixel metrics improve markedly with more views. Our
method never uses the images and reconstructs the symbolic code layout, so photometric scores are lower, yet it is optimized for machine
readability (cf. decoding results) and operates at a fraction of the compute/primitive budget. Best numbers are in bold.
the signal from symbolic structure, so per-pixel colors and
shading may differ from the input photos even when the
underlying bit layout is correct. Pixel-wise metrics penal-
ize these benign deviations, while the decoding and detec-
tion angles reflect what matters for this task: machine read-
ability under foreshortening, where our method is consis-
tently stronger. Importantly, these gains are achieved with-
out any training and at a fraction of the compute and mem-
ory cost, indicating that we deliver better task performance
than training-based reconstruction despite a lower photo-
metric similarity.
4.3. Greenhouse Simulation
In this section, we demonstrate how our fiducial marker
splatting framework can be transferred to radiance field
scenes for robotics simulation.
As a proof of concept,
we construct a greenhouse environment using 2D Gaussian
splats trained from real images of a cucumber greenhouse.
Individual splats are generated from these images and then
stitched together to form a coherent miniature greenhouse
scene, shown in Figure 6. In Gazebo simulations, using the
fiducial markers, a mobile robot could effectively localize
in the environment. This prototype highlights the potential
of our approach for building lightweight, splat-based envi-
ronments that can serve as testbeds for robotic perception
and control.
5. Conclusion
We introduced a method for representing fiducial markers
with Gaussian primitives that avoids reliance on splatting-
based training. Our approach combines rectangle approx-
imation with minimal rectilinear partitioning to produce
Figure 6.
Top: Greenhouse Simulation Environment with 2D
Gaussian Splatting. Bottom: Greenhouse Simulation with Fidu-
cial Marker Splatting
7

<!-- page 8 -->
compact representations that preserve readability while sig-
nificantly reducing primitive counts.
Experiments show
that this reduction lowers runtime and inference cost, with
the benefits becoming more pronounced as marker size in-
creases.
As
future
work,
we
plan
to
extend
this
idea
beyond
rectangles,
enabling
Gaussian
primitives
to
represent
arbitrary
shapes
such
as
vector
drawings.
This would broaden the scope of our framework to
structured visual representations beyond fiducial mark-
ers.
References
[1] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. ICCV, 2021. 2
[2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. CVPR, 2022. 2
[3] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Hal-
ber, Matthias Niessner, Manolis Savva, Shuran Song, Andy
Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-
d data in indoor environments. International Conference on
3D Vision (3DV), 2017. 2
[4] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent
Sitzmann. pixelsplat: 3d gaussian splats from image pairs for
scalable generalizable 3d reconstruction. In CVPR, 2024. 2
[5] Timothy Chen, Ola Shorinwa, Joseph Bruno, Aiden Swann,
Javier Yu, Weijia Zeng, Keiko Nagami, Philip Dames, and
Mac Schwager.
Splat-Nav: Safe Real-Time Robot Navi-
gation in Gaussian Splatting Maps. IEEE Transactions on
Robotics, 41:2765–2784, 2025. 2
[6] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. MVSplat: Efficient 3D Gaussian Splatting from Sparse
Multi-view Images. In Computer Vision – ECCV 2024, pages
370–386, Cham, 2025. Springer Nature Switzerland. 2
[7] Phillippe de Reffye, Claude Edelin, Jean Franc¸on, Marc
Jaeger, and Claude Puech. Plant models faithful to botanical
structure and development. In Proceedings of the 15th an-
nual conference on Computer graphics and interactive tech-
niques, pages 151–158, New York, NY, USA, 1988. Associ-
ation for Computing Machinery. 3
[8] Xiang Feng, Wen Jie Liang, Hai Zhou Chen, Xiao Yu Liu,
and Fang Yan. Autonomous Localization and Navigation for
Agricultural Robots in Greenhouse. Wireless Personal Com-
munications, 131(3):2039–2053, 2023. 1
[9] L Ferrari, P.V Sankar, and J Sklansky. Minimal rectangular
partitions of digitized blobs. Computer Vision, Graphics, and
Image Processing, 28(1):58–71, 1984. 5, 1
[10] Giuseppe Fragapane, Hans-Henrik Hvolby, Fabio Sgarbossa,
and Jan Ola Strandhagen. Autonomous Mobile Robots in
Hospital Logistics. In Advances in Production Management
Systems. The Path to Digital Transformation and Innovation
of Production Management Systems, pages 672–679, Cham,
2020. Springer International Publishing. 1
[11] Steven J. Gortler, Radek Grzeszczuk, Richard Szeliski, and
Michael F. Cohen. The lumigraph. In Proceedings of the
23rd Annual Conference on Computer Graphics and Inter-
active Techniques, pages 43–54, New York, NY, USA, 1996.
Association for Computing Machinery. 2
[12] Antoine Gu´edon and Vincent Lepetit.
SuGaR: Surface-
Aligned Gaussian Splatting for Efficient 3D Mesh Re-
construction and High-Quality Mesh Rendering.
2024
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 5354–5363, 2024. 2
[13] Suman Harapanahalli, Niall O Mahony, Gustavo Velasco
Hernandez, Sean Campbell, Daniel Riordan, and Joseph
Walsh.
Autonomous Navigation of mobile robots in fac-
tory environment. Procedia Manufacturing, 38:1524–1531,
2019. 1
[14] Radim Hercik, Radek Byrtus, Rene Jaros, and Jiri Koziorek.
Implementation of Autonomous Mobile Robot in SmartFac-
tory.
Applied Sciences, 12(17):8912, 2022.
Number: 17
Publisher: Multidisciplinary Digital Publishing Institute. 1
[15] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In SIGGRAPH 2024 Conference Papers.
Association for Computing Machinery, 2024. 2, 3, 5
[16] Antun Ivanovic, Marsela Polic, Jelena Tabak, and Matko Or-
sag.
Render-in-the-loop Aerial Robotics Simulator: Case
Study on Yield Estimation in Indoor Agriculture. In 2022
International Conference on Unmanned Aircraft Systems
(ICUAS), pages 787–793, 2022. 2
[17] Nilay Jadav, Harsh Chhajed, Upesh Patel, Devesh Jani, and
Aum Barai. AI-Enhanced Quad-Wheeled Robot for Targeted
Plant Disease Surveillance in Greenhouses. In 2023 Global
Conference on Information Technologies and Communica-
tions (GCITC), pages 1–7, 2023. 1
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2
[19] O. Kniemeyer, G. Buck-Sorlin, and W. Kurth. GroIMP as a
platform for functional-structural modelling of plants. Fron-
tis, pages 43–52, 2007. 3
[20] N. Koenig and A. Howard. Design and use paradigms for
Gazebo, an open-source multi-robot simulator.
In 2004
IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), pages 2149–2154 vol.3, 2004. 1
[21] N. Vimal Kumar and C. Selva Kumar. Development of col-
lision free path planning algorithm for warehouse mobile
robot. Procedia Computer Science, 133:456–463, 2018. 1
[22] Marc Levoy and Pat Hanrahan. Light field rendering. In
Proceedings of the 23rd Annual Conference on Computer
Graphics and Interactive Techniques, pages 31–42, New
York, NY, USA, 1996. Association for Computing Machin-
ery. 2
[23] Xinhai Li, Jialin Li, Ziheng Zhang, Rui Zhang, Fan Jia,
Tiancai Wang, Haoqiang Fan, Kuo-Kun Tseng, and Ruip-
ing Wang. RoboGSim: A Real2Sim2Real Robotic Gaussian
Splatting Simulator, 2024. 2
8

<!-- page 9 -->
[24] Aristid Lindenmayer. Mathematical models for cellular in-
teractions in development I. Filaments with one-sided inputs.
Journal of Theoretical Biology, 18(3):280–299, 1968. 3
[25] Zhen
Liu,
Yao
Feng,
Michael
J.
Black,
Derek
Nowrouzezahrai, Liam Paull, and Weiyang Liu.
Meshd-
iffusion: Score-based generative 3d mesh modeling.
In
International Conference on Learning Representations,
2023. 2
[26] Mohd Saiful Azimi Mahmud, Mohamad Shukri Zainal
Abidin, Zaharuddin Mohamed, Muhammad Khairie Id-
ham Abd Rahman, and Michihisa Iida. Multi-objective path
planner for an agricultural mobile robot in a virtual green-
house environment. Computers and Electronics in Agricul-
ture, 157:488–499, 2019. 1
[27] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 2
[28] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. ACM Trans. Graph., 41(4):102:1–
102:15, 2022. 2
[29] Edwin Olson. AprilTag: A robust and flexible visual fiducial
system. In 2011 IEEE International Conference on Robotics
and Automation, pages 3400–3407, 2011. 3, 5
[30] Peter E. Oppenheimer. Real time design and animation of
fractal plants and trees. SIGGRAPH Comput. Graph., 20(4):
55–64, 1986. 3
[31] Yaoqiang Pan, Hao Cao, Kewei Hu, Hanwen Kang, and Xing
Wang. A Novel Perception and Semantic Mapping Method
for Robot Autonomy in Orchards, 2023. 1
[32] Christophe Pradal,
Samuel Dufour-Kowalski,
Fr´ed´eric
Boudon, Christian Fournier, and Christophe Godin.
Ope-
nAlea: a visual programming and component-based soft-
ware platform for plant modelling. Functional plant biology:
FPB, 35(10):751–760, 2008. 3
[33] William T. Reeves and Ricki Blau. Approximate and proba-
bilistic algorithms for shading and rendering structured par-
ticle systems. SIGGRAPH Comput. Graph., 19(3):313–322,
1985. 3
[34] Johannes
Lutz
Sch¨onberger
and
Jan-Michael
Frahm.
Structure-from-motion revisited.
In Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2016. 5
[35] Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Ta-
tiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela
Dai, and Matthias Nießner.
MeshGPT: Generating Tri-
angle Meshes with Decoder-Only Transformers.
In 2024
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 19615–19625. IEEE Computer
Society, 2024. 2
[36] K. S. Suresh, R. Venkatesan, and S. Venugopal. Mobile robot
path planning using multi-objective genetic algorithm in in-
dustrial automation.
Soft Computing, 26(15):7387–7400,
2022. 1
[37] Kosmas Tsiakas, Alexios Papadimitriou, Eleftheria Maria
Pechlivani, Dimitrios Giakoumis, Nikolaos Frangakis, Anto-
nios Gasteratos, and Dimitrios Tzovaras. An Autonomous
Navigation Framework for Holonomic Mobile Robots in
Confined Agricultural Environments. Robotics, 12(6):146,
2023. 1
[38] Naoum Tsolakis, Dimitrios Bechtsis, and Dionysis Bochtis.
AgROS: A Robot Operating System Based Emulation Tool
for Agricultural Robotics. Agronomy, 9(7):403, 2019. 2
[39] Hendrik Unger, Tobias Markert, and Egon M¨uller. Evalu-
ation of use cases of autonomous mobile robots in factory
environments. Procedia Manufacturing, 17:254–261, 2018.
1
[40] Brent Van De Walker, Brendan Byrne, Joshua Near, Blake
Purdie, Matthew Whatman, David Weales, Cole Tarry, and
Medhat Moussa. Developing a Realistic Simulation Envi-
ronment for Robotics Harvesting Operations in a Vegetable
Greenhouse. Agronomy, 11(9):1848, 2021. 2
[41] Jason Weber and Joseph Penn. Creation and rendering of
realistic trees.
In Proceedings of the 22nd annual con-
ference on Computer graphics and interactive techniques,
pages 119–128, New York, NY, USA, 1995. Association for
Computing Machinery. 3
[42] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Lin-
guang Zhang, Xiaoou Tang, and Jianxiong Xiao.
3D
ShapeNets: A deep representation for volumetric shapes.
In 2015 IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pages 1912–1920. IEEE Computer So-
ciety, 2015. 2
[43] Yu Xiang, Roozbeh Mottaghi, and Silvio Savarese. Beyond
PASCAL: A benchmark for 3D object detection in the wild.
In IEEE Winter Conference on Applications of Computer Vi-
sion, pages 75–82, 2014. 2
[44] Oh Wei Xuan, Hazlina Selamat, and Mohd Taufiq Muslim.
Autonomous Mobile Robot for Transporting Goods in Ware-
house and Production.
In Advances in Intelligent Manu-
facturing and Robotics, pages 555–565, Singapore, 2024.
Springer Nature. 1
[45] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d
indoor scenes. In Proceedings of the International Confer-
ence on Computer Vision (ICCV), 2023. 2
[46] Dar´ıo Fernando Y´epez-Ponce, Jos´e Vicente Salcedo, Pa´ul D.
Rosero-Montalvo, and Javier Sanchis.
Mobile robotics in
smart farming: current trends and applications. Frontiers in
Artificial Intelligence, 6, 2023. Publisher: Frontiers. 1
[47] Yuhang Zheng, Xiangyu Chen, Yupeng Zheng, Songen Gu,
Runyi Yang, Bu Jin, Pengfei Li, Chengliang Zhong, Zeng-
mao Wang, Lina Liu, et al.
Gaussiangrasper:
3d lan-
guage gaussian splatting for open-vocabulary robotic grasp-
ing. arXiv preprint arXiv:2403.09637, 2024. 2
9

<!-- page 10 -->
Fiducial Marker Splatting for High-Fidelity Robotics Simulations
Supplementary Material
6. Minimum Rectilinear Partition Algorithm
As noted in the main text, we now explain the rectangula-
tion algorithm we used; the method follows [9] and is not
our contribution. The pseudo-code is given in Algorithm 1
Given a rectilinear polygon P = (V, E) with possible
interior rectilinear holes H = {(Vi, Ei)}k
i=1, we collect all
concave vertices Vc from the outer boundary and the hole
boundaries. From Vc we enumerate all valid axis-aligned
chords—horizontal Ch and vertical Cv—that lie completely
inside P. We then build a bipartite graph G = (Cv, Ch, Ec)
whose edges connect perpendicular chords that geometri-
cally intersect.
Computing a maximum bipartite match-
ing M and the corresponding minimum vertex cover S (by
K˝onig’s theorem) yields a maximum set of pairwise non-
intersecting chords I = (Cv ∪Ch) \ S.
Drawing all chords in I partitions P into smaller rectilin-
ear subpolygons Pr with induced holes Hr (Algorithm 1).
Each subpolygon is then rectangulated by a greedy rou-
tine: starting from concave vertices not already incident to
drawn chords, we insert maximal interior axis-aligned seg-
ments (extending until hitting an existing edge, a chord, or
a hole boundary) and iterate until only rectangles remain.
The union over subpolygons gives the final partition R.
7. Additional Visualizations
We provide additional visualizations for two aspects: (i) ex-
tended qualitative comparisons with 2DGS (see Sec. 4); and
(ii) a demonstration of incorporating and detecting AprilT-
ags within a Gaussian Splatting greenhouse simulation. All
results use the same AprilTag / QR code assets and train-
ing data as in the main paper— we only include additional
views/images.
Algorithm 1 Minimum Rectilinear Partition
Input: Rectilinear polygon P = (V, E), Interior recti-
linear polygons H = {(Vi, Ei)}k
i=1.
Output: A minimum set of non-overlapping rectangles
R that partitions P.
Vc ←CONCAVEVERTICES(V, E)
for i ←1, k do
Vc ←Vc ∪CONCAVEVERTICES(Vi, Ei)
end for
Ch ←AXISPARALLELCHORDS(Vc, axis = (1, 0))
Cv ←AXISPARALLELCHORDS(Vc, axis = (0, 1))
Ec ←{(ci, cj) ∈Ch × Cv : INTERSECT(ci, cj)}
G ←(Cv, Ch, Ec)
▷Bipartite Graph
M ←MAXBIPARTITEMATCHING(G)
S ←MINVERTEXCOVER(G, M)
▷K˝onig’s Theorem
I ←(Cv ∪Ch) \ S
Pr, Hr ←PARTITION(P, H, I)
R ←∅
for j ←1, |Pr| do
Vj, Ej ←Pr[j]
Hj ←Hr[j]
Rj ←GREEDYPARTITION((Vj, Ej), Hj)
R ←R ∪Rj
end for
return R
1

<!-- page 11 -->
Figure 7. Additional qualitative comparisons on planar fiducials. Columns: Ours, 2DGS trained with 3/5/9/17 views, and Ground
Truth. Under narrow training baselines (3–5 views), 2DGS degrades sharply—showing thickness/halo artifacts and texture distortions,
especially at oblique viewpoints—improving only gradually with more views. Our method remains planar and crisp across viewpoints.
Figure 8. Normal predictions on a planar AprilTag. Each panel shows RGB (left half) and the estimated normal map (right half). Left:
ours maintains a uniform planar normal. Right: 2DGS entangles texture with geometry, producing spurious relief where the tag pattern
appears in the normals.
2

<!-- page 12 -->
(a) AprilTag detections overlaid on the Gaussian Splatting greenhouse render.
(b) Depth map from the same viewpoint.
(c) Per-pixel surface normals (RGB) from the same viewpoint.
Figure 9. Additional greenhouse visualizations: (a) AprilTag detection overlay, (b) depth map, and (c) surface normals.
3
