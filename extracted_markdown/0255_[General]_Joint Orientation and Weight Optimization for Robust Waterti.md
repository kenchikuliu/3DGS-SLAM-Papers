<!-- page 1 -->
JOINT ORIENTATION AND WEIGHT OPTIMIZATION FOR ROBUST
WATERTIGHT SURFACE RECONSTRUCTION VIA
DIRICHLET-REGULARIZED WINDING FIELDS
Jiaze Li
Nanyang Technological University
Singapore
Daisheng Jin
Nanyang Technological University
Singapore
Fei Hou
Institute of Software, Chinese Academy of Sciences
University of Chinese Academy of Sciences
China
Junhui Hou
City University of Hong Kong
Hong Kong
Zheng Liu
China University of Geosciences (Wuhan)
China
Shiqing Xin
Shandong University
China
Wenping Wang
Texas A&M University
United States of America
Ying He∗
Nanyang Technological University
Singapore
ABSTRACT
We propose Dirichlet Winding Reconstruction (DIWR), a robust method for reconstructing watertight
surfaces from unoriented point clouds with non-uniform sampling, noise, and outliers. Our method
uses the generalized winding number (GWN) field as the target implicit representation and jointly
optimizes point orientations, per-point area weights, and confidence coefficients in a single pipeline.
The optimization minimizes the Dirichlet energy of the induced winding field together with additional
GWN-based constraints, allowing DIWR to compensate for non-uniform sampling, reduce the impact
of noise, and downweight outliers during reconstruction, with no reliance on separate preprocessing.
We evaluate DIWR on point clouds from 3D Gaussian Splatting, a computer-vision pipeline, and
corrupted graphics benchmarks. Experiments show that DIWR produces plausible watertight surfaces
on these challenging inputs and outperforms both traditional multi-stage pipelines and recent joint
orientation-reconstruction methods.
Keywords 3D reconstruction · unoriented point clouds · normal orientation · generalized winding numbers · Dirichlet
energy
1
Introduction
Reconstructing high-quality 3D surfaces from point clouds is an important problem in computer graphics and geometry
processing, with applications in digital heritage preservation, robotic perception, and virtual content creation. Modern
acquisition devices can capture large point clouds with little effort, but the raw sensor output is typically discrete and
disorganized. Real-world scans often exhibit non-uniform sampling, strong noise, outliers, and missing regions, which
makes surface reconstruction challenging.
∗Corresponding author.
arXiv:2602.13801v1  [cs.CV]  14 Feb 2026

<!-- page 2 -->
Multi-stage pipelines
Input
{𝐩𝐢}
Outlier
removal
Denoising
Normal
orientation
Reconstruction
(e.g., sPSR({𝐩𝐢, 𝐧𝐢}))
Watertight
surface
Optimization
𝑓({𝐧𝐢}; 𝐩𝐢)
Optimization
𝑓( 𝐧𝐢, 𝑎" , 𝑐" ; 𝐩𝐢)
Input
{𝐩𝐢}
Watertight
surface
Joint orientation-reconstruction pipelines (e.g., iPSR, WNNC, DWG, FaCE, etc.)
DiWR (Ours)
Input
{𝐩𝐢}
Watertight
surface
Reconstruction
(e.g., sPSR({𝐩𝐢, 𝐧𝐢}))
Reconstruction
(e.g., sPSR({𝐩𝐢, 𝐧𝐢, 𝑐", 𝑎"}))
Input
Filtered pts
WNNC
WNNC⋆
Ours
Figure 1: Conceptual illustration and comparisons. Left: classical multi-stage pipelines typically perform outlier
removal and denoising, then estimate and globally orient normals, and finally apply a reconstruction solver (e.g.,
sPSR [Kazhdan and Hoppe 2013]) to obtain a watertight surface. Recent unified approaches (e.g., WNNC [Lin
et al.
2024], DWG [Liu et al.
2025a], and FaCE [Scrivener et al.
2025]) couple normal orientation with an
implicit representation, enabling reconstruction directly from raw, unoriented point clouds. Our method further extends
this unified formulation by additionally optimizing per-point area weights and confidence coefficients, improving
robustness to non-uniform sampling, noise, and outliers. Right: qualitative results on the Bunny model with increasing
corruption (left to right). We show the corrupted inputs, the filtered points after preprocessing, WNNC applied to the
raw inputs (WNNC) and to the filtered points (WNNC⋆), and our results. As the corruption level increases, WNNC
becomes sensitive to outliers and sampling irregularity. In contrast, DiWR operates directly on the raw inputs without
preprocessing and maintains coherent watertight reconstructions across a wider range of corruption levels.
Many classical reconstruction methods assume oriented point clouds. Implicit function methods such as Poisson
Surface Reconstruction (PSR) [Kazhdan et al. 2006] and Screened PSR (sPSR) [Kazhdan and Hoppe 2013] are widely
used because they produce smooth and watertight surfaces and can tolerate moderate noise when reliable normals
are available. However, many acquisition pipelines output unoriented point clouds, and enforcing global orientation
consistency is difficult, especially for shapes with complex topology, thin structures, or sparse observations.
A standard way to handle unoriented inputs is to decompose the problem into separate stages. A classic example
is the work of Huang et al. [2009], which introduces a robust point-cloud consolidation pipeline as a preprocessing
step for surface reconstruction. Starting from raw scans with noise, outliers, non-uniform sampling, and even closely
spaced surface sheets, they first produce a denoised, outlier-free, and more evenly distributed particle set via weighted
locally optimal projection, which improves the reliability of local PCA normal estimation. They then propose an
iterative normal-estimation framework that combines priority-driven orientation propagation with orientation-aware
PCA to obtain globally consistent normals, enabling conventional reconstruction solvers to generate cleaner surfaces.
Such multi-stage pipelines are easy to assemble from existing components, but they can be fragile because errors
introduced early are difficult to correct later, especially in challenging scenarios. These limitations have motivated
unified formulations that couple normal orientation with implicit reconstruction, enabling reconstruction directly from
raw, unoriented point clouds.
Despite this progress, robust reconstruction from unoriented point clouds with severe imperfections remains difficult.
State-of-the-art unified methods can perform well on reasonably clean inputs, but their performance often degrades
when the input exhibits extreme non-uniformity, strong noise, or a large fraction of outliers. A key limitation is
that most existing approaches mainly treat orientation as the primary unknown and implicitly assume that (i) the
discrete sampling adequately represents the underlying surface measure and (ii) all points are equally reliable. Under
non-uniform sampling and heavy corruption, these assumptions bias the implicit field and its gradients, leading to
unstable orientation updates and degraded reconstructions.
To address this limitation, we extend the unified formulation by jointly optimizing point orientations together with per-
point area weights and confidence coefficients. Specifically, we adopt the generalized winding number (GWN) [Jacobson
2

<!-- page 3 -->
et al. 2013] as the target implicit representation. Its point-based formulation depends explicitly on orientations and
per-point weights, and it provides a clear inside-outside indicator that can be evaluated efficiently on point clouds. We
regularize the induced GWN field by minimizing its Dirichlet energy, together with additional GWN-based constraints
that encourage the desired winding-number values on reliable surface samples and control surface area. Since these
variables are coupled nonlinearly, we optimize them in an alternating manner, holding the other variables fixed when
updating one group. Our method, Dirichlet Winding Reconstruction (DIWR), iteratively (i) updates orientations to
improve global consistency, (ii) refines per-point area weights to compensate for non-uniform sampling, and (iii)
estimates confidence coefficients to down-weight outliers and low-quality samples. Unlike existing methods, this joint
optimization allows the reconstruction process itself to adapt to uneven sampling and corrupted inputs, reducing reliance
on separate preprocessing. As a result, DIWR produces stable, high-quality surfaces on challenging point clouds with
significant noise and outliers (see Figure 1).
2
Related Work
Point cloud reconstruction has been studied for more than four decades. Earlier approaches often rely on computational
geometry. Representative methods include α-Shape [Edelsbrunner and Mücke 1994], Ball Pivoting [Bernardini et al.
1999], Power Crust [Amenta et al. 2001], and Tight Cocone [Dey and Goswami 2003], among others. These methods
are computationally efficient and can often provide theoretical guarantees under sampling conditions. However, they
lack robustness when handling real-world inputs that often exhibit various types of defects.
Most modern surface reconstruction pipelines rely on implicit representations, following the seminal work of Hoppe
et al. [1992], because of their robustness and practical performance. Many subsequent methods assume that reliable,
consistently oriented normals are given or can be estimated, including moving least squares and Poisson-based
approaches [Ohtake et al. 2003; Kazhdan et al. 2006; Kazhdan and Hoppe 2013]. In practice, however, obtaining
accurate and globally consistent normals from raw scans remains difficult, which limits the applicability of these
methods to real-world unoriented point sets.
To remove the dependency on pre-oriented normals, several methods solve for normal orientation and an implicit surface
together. Iterative PSR (iPSR) [Hou et al. 2022] alternates between reconstructing an implicit function and updating
normals using its gradients. Ma et al. [2024] incorporate isovalue constraints into the Poisson formulation to solve for
globally consistent normal orientations and the implicit function simultaneously via a single sparse linear least-squares
system. Scrivener et al. [2025] estimate normals by modeling a Faraday-cage electrostatic effect via a Poisson system
and using gradients of the resulting field to orient normals, improving robustness in the presence of interior artifacts.
Parametric Gauss Reconstruction (PGR) [Lin et al. 2022] combines normal consistency with gradient-field constraints
through parametric Gauss mapping. AGR [Ma et al. 2025] extends PGR by adding a convection term to the Laplace
operator to utilize anisotropic directional information in the point cloud. This approach leads to more effective linear
equations and enhances orientation and reconstruction quality, especially for thin structures and small holes.
Recently, techniques based on GWN [Jacobson et al. 2013] have gained popularity, as GWN enables global inside-
outside reasoning and is efficient to evaluate on point clouds [Barill et al.
2018]. GCNO [Xu et al.
2023] and
BIM [Liu et al. 2024a] use winding-number-based constraints or energies to recover globally consistent orientation
from random initialization. WNNC [Lin et al. 2024] further enforces agreement between normals and the gradients
of the induced winding field. DWG [Liu et al. 2025a] introduces a diffusion-based framework that supports highly
parallel computation and scales well on GPUs.
In addition to GWN-based formulations, several optimization-driven techniques have been proposed for unoriented
point clouds. Gotsman and Hormann [2024] formulate normal orientation via O(n) sparse linear systems derived
from Stokes’ theorem. Huang et al. [2024] propose stochastic normal orientation by optimizing a probabilistic
objective that combines global inside-outside cues with local consistency. Beyond orientation, variational methods
also provide implicit reconstruction formulations. Huang et al. [2019] define the implicit function as the solution to
a constrained quadratic optimization problem, enabling exact interpolation of the input points. Xia and Ju [2025]
accelerate this variational framework by exploiting the locality of natural neighborhoods, substantially improving
runtime and scalability.
While the above methods significantly improve reconstruction from unoriented inputs, most of them mainly optimize
orientations (and the implicit field) while assuming point positions are reliable. In contrast, our method also optimizes
per-point area weights and confidence coefficients, which enhances robustness to uneven sampling, severe noise, and a
high outlier ratio.
There is also a large body of work on deep learning methods for 3D reconstruction. These approaches typically learn
signed or unsigned distance functions directly from unoriented points [Ma et al. 2021; Chen et al. 2023; Wang et al.
3

<!-- page 4 -->
2023a b 2024; Ren et al. 2023; Wang et al. 2023b; Li et al. 2024; Hu et al. 2025; Xu et al. 2025]. While this avoids
an explicit normal-orientation step, such methods are often limited to small- to middle-scale inputs due to their high
memory consumption. We provide additional discussion and quantitative comparisons with learning-based methods in
the Appendix.
3
Preliminaries
The generalized winding number [Jacobson et al. 2013] extends the classical winding number from closed curves to
closed surfaces, distinguishing the inside and outside of a solid. Originating from potential theory, the winding number
can be interpreted as a scalar potential field whose level sets represent the geometric enclosure of space. Let Ω⊂R3
be a solid with smooth boundary ∂Ω. For a point x ∈R3, denote by n(y) the outward unit normal at y ∈∂Ω. The
continuous winding number field is defined as
w(x) = 1
4π
Z
∂Ω
⟨n(y), y −x⟩
∥y −x∥3
dS(y),
(1)
where dS is the area element. The generalized winding number measures how many times the surface wraps around
the query point x. For a closed orientable surface, w(x) takes values close to 1 in the interior of Ω, close to 0 in the
exterior, and approximately 0.5 on the boundary ∂Ω.
Given an oriented point set {(pi, ni)}n
i=1 sampled from ∂Ω, Equation (1) can be discretized as [Barill et al. 2018]:
w(q) ≈
n
X
i=1
ai
⟨ni, pi −q⟩
4π∥pi −q∥3 ,
(2)
where ai is the local surface element area associated with pi.
Ideally, the area weights {ai} should be computed from a geodesic Voronoi diagram on the underlying surface [Wang
et al.
2015], since such cells provide a principled discretization of the surface integral. However, without point
connectivity and orientation, constructing accurate geodesic Voronoi cells from a raw point cloud is technically
challenging. In practice, one often uses a local planar approximation based on 2D Voronoi diagrams [Barill et al. 2018;
Xu et al. 2023; Liu et al. 2024a]: for each point pi, estimate a local tangent plane via PCA, project a neighborhood
of pi onto this plane, and compute the 2D Voronoi cell on the plane. The area of this 2D cell is then used as an
approximation of the corresponding geodesic Voronoi area weight ai. This discrete form enables efficient evaluation of
GWN directly on point clouds, and has been used in geometry processing tasks such as inside–outside queries, Boolean
operations, shape analysis, and global normal orientation.
The GWN field w is harmonic in R3 \ ∂Ωand satisfies the usual jump conditions across ∂Ω. In particular, ∇w
aligns with the globally consistent outward normals on the surface, indicating that GWN encodes both inside–outside
information and orientation continuity. This makes GWN a reliable implicit representation and a useful foundation
for formulating globally consistent normal orientation. Takayama et al. [2014] explore orienting polygon soups by
minimizing the Dirichlet energy of the induced winding field using a 0-1 integer programming formulation. In practice,
this approach is computationally prohibitive due to the large number of binary variables. In contrast, Liu et al. [2024a]
relax the problem by treating orientations as continuous variables and solve it via nonlinear optimization on unoriented
point clouds.
GWNs have found extensive applications in digital geometry processing, including robust inside-outside segmenta-
tion [Jacobson et al. 2013], Boolean operations [Zhou et al. 2016], containment queries for parametric geometry [Liu
et al. 2025b], and point cloud orientation [Xu et al. 2023; Lin et al. 2024], among others.
4
Method
Let P = {pi}n
i=1 be an unoriented point set sampled from a watertight manifold surface, possibly corrupted by noise
and outliers. We treat the point orientations N = {ni}n
i=1 and area weights a = {ai}n
i=1 as unknown variables. To
reduce the negative effects of noise and outliers, we assign each point pi a confidence coefficient ci ∈[0, 1], where
ci = 1 indicates an inlier and ci = 0 an outlier. Denote by c = {ci}n
i=1. We collect these unknowns into θ := (N, a, c),
and express the generalized winding number field at a query point q as
wθ(q) =
n
X
i=1
aici
(pi −q) · ni
4π∥pi −q∥3 .
(3)
4

<!-- page 5 -->
Input
t = 0
t = 1
t = 2
t = 3
t = 4
S
Error vs. iter.
Figure 2: Overview of DIWR on Kitten (n = 60000, bu = 0.54, bo = 0.16, bσ = 6.79 × 10−5).Starting from random
orientations, the algorithm alternates updates of point orientations together with per-point area weights ai and confidence
coefficients ci. Each intermediate panel (from t = 1 to t = 4) visualizes the reconstructed surface, point orientations,
cut views of the induced GWN field, the effective weights aici, and the distribution of confidence coefficients ci (shown
on a log scale to accommodate the large dynamic range in counts). After convergence, we retain high-confidence points
and pass their orientations and effective weights to sPSR to obtain a watertight surface S. The rightmost plot reports the
mean orientation error versus iteration.
4.1
Objective Functions
Dirichlet Energy
In the continuous setting, the winding field induced by a globally consistent orientation is harmonic
in R3 \ ∂Ωand therefore minimizes the Dirichlet energy among fields with the same boundary conditions. In our
discrete setting, however, the target surface ∂Ωis unknown. We thus exclude a thin band around the surface by
removing a neighborhood of the input points that are deemed reliable. Concretely, let I = { i | ci ≥τin } be the set
of high-confidence points (with τin as a threshold), and define the excluded band Uδ(θ) = S
i∈I B(pi, δ), where δ
controls the band thickness. We then define the Dirichlet energy in the remaining region as
Ediri(θ) =
Z
B\Uδ(θ)
∥∇wθ(x)∥2 dV (x),
(4)
where dV is the volume element, and B is a bounding box that surrounds the input points with a sufficient margin.
Low-confidence points (small ci) are considered outliers and are not used to define the excluded band.
Discrete Evaluation
To evaluate Ediri in practice, we approximate the volume integral by uniform sampling in
B. Since the integrand is not defined on the (unknown) surface, we ignore samples that fall inside the excluded
band and down-weight samples close to it. Specifically, for each high-confidence point pi with ci > τin, we remove
energy samples inside the ball B(pi, rs), where rs controls the band thickness (we set τin = 0.9 and rs = 0.03 in
all experiments). For samples that lie outside but near this ball, we compute a partial-volume weight based on the
intersection volume between the ball and the voxel centered at the sample [Jones and Williams 2017]. The resulting
weight is proportional to the remaining (non-intersected) voxel volume and lies in [0.5, 1]. With a uniform grid of
sampling points q and voxel volume Vc, the discrete approximation takes the form
bEdiri(θ) =
X
q∈Q
δqVc ∥∇wθ(q)∥2 ,
(5)
where Q is the set of sampled grid points and δq is the partial-volume weight described above.
Surface Points
For a closed orientable surface, the winding number field takes values close to 1 inside and close to 0
outside; ideally, w = 1
2 on the boundary ∂Ω. We therefore introduce a surface term that encourages high-confidence
5

<!-- page 6 -->
(a)
(b)
(c)
Figure 3: Area-weight optimization compensates for non-uniform sampling on the corrupted Armadillo model (n =
172, 500, bσ = 1.6 × 10−4, bo = 0.13, bu = 0.30). (a) Input point cloud exhibiting a dense-to-sparse sampling pattern
and outliers; we initialize all area weights ai uniformly. (b) Optimized effective weights aici (visualized with a heat
colormap, warm = larger) become spatially adaptive: inlier weights increase in sparsely sampled regions and decrease in
densely sampled regions, while outliers receive negligible effective weights through ci. (c) The rebalanced weights yield
a more balanced discrete winding-number integration, stabilizing the induced field (shown in the inset) and improving
downstream watertight reconstruction.
points to lie near the 1
2-level set of the induced field:
Esurf(θ) = 1
|I|
X
i∈I

wθ(pi) −1
2
2
.
(6)
When sampling is sparse or highly non-uniform, this term provides extra anchoring constraints at reliable samples,
helping prevent drift of the 1
2-level set.
Stable Surface Area
To avoid degenerate solutions (e.g., driving all weights to zero) and to stabilize the optimization,
we constrain the total effective surface area to remain roughly constant within each optimization stage. Specifically, we
penalize deviations of the weighted area sum from its value at the start of the current stage:
Earea(a, c) =

n
X
i=1
aici −
n
X
i=1
ab
icb
i
 ,
(7)
where ab
i and cb
i represent the values of ai and ci at the beginning of the current optimization stage. This term encourages
the total effective surface area P
i aici to remain stable during the current stage.
Polarized Confidence Coefficients
To encourage the confidence weights ci of outliers to decrease toward 0 and
thus reduce their influence on optimization and reconstruction, we introduce a binary term Econf that encourages ci to
concentrate near 0 and 1:
Econf(c) =
n
X
i=1
|ci(1 −ci)|.
(8)
Without this term, many points may have intermediate confidence values, allowing outliers to continue affecting the
winding field and potentially biasing ∇wθ and the related orientation updates. Encouraging near-binary confidences
helps create a clearer separation between inliers and outliers and improves the stability of the overall optimization.
4.2
Optimization
Initialization
For each input point pi, we initialize the confidence at ci = 1 and set the orientation ni to a random
unit vector. We compute the initial area weights {ai} using the 2D Voronoi approximation from prior work [Barill et al.
2018]. For heavily corrupted inputs (e.g., with many outliers), this geometric estimation can become unreliable; in such
cases, we optionally initialize all area weights uniformly (e.g., ai ←1) and let the subsequent optimization refine them.
6

<!-- page 7 -->
(a)
(b)
(c)
(d)
(e)
(f)
Figure 4: Resetting and optimizing the confidence coefficients in one c-optimization stage on Thai Statue (n =
240, 000, bσ = 0.00026, bo = 0.16, bu = 0.45). (a)-(c): an early iteration where orientations are still inaccurate; (d)-(f): a
later iteration close to convergence. (a, d) Coarse initialization via bi-means clustering of the current GWN values:
points in the outlier cluster are shown in purple, while the remaining points are treated as inliers in yellow (b,e)
Density-based stratification softens this binary initialization and assigns multi-level confidences in [0, 1]. (c,f) After
optimization, the confidence distribution becomes strongly bimodal, concentrating near 0 and 1. For visualization, we
show low-confidence points in gray and color the remaining points by their effective weights aici using a heat colormap.
1.0
0.0
(a)
(b)
Figure 5: Density stratification for resetting confidence coefficients. We illustrate the refinement step using a simplified
example with three density levels. Each dot is an input point; color encodes confidence (blue: ci = 0, red: ci = 1,
with warmer colors indicating higher confidence). (a) After the coarse bi-means split on winding values, each point
receives a binary confidence (0 for the outlier cluster and 1 for the inlier cluster). We then group points into density
levels according to their local density ρi (each box denotes one level). (b) Within each density level, we replace the
binary assignments by the level-wise mean confidence, assigning all points in the level the same averaged value. This
produces a smoother initialization with confidence values in [0, 1], which is used as the starting point for the subsequent
c-optimization.
Finally, we precompute a local density estimate ρi by counting the number of neighbors within a ball B(pi, rρ), where
we set the default radius rρ = 0.06 in our implementation. This density is later used in the density-stratified reset of
confidence coefficients.
Strategies
The per-point area weights ai, confidence coefficients ci, and orientations ni all influence the induced
GWN field. Because these variables are coupled nonlinearly, optimizing them together can create a complex and
potentially unstable problem. We therefore use a staged alternating strategy: in each stage, we update one group of
variables while keeping the others fixed, which enhances numerical stability and makes the optimization easier to
control.
Since the three variable groups influence each other, the overall procedure is iterative. Each outer iteration consists of
three stages: (i) DWG-based orientation update for {ni}, (ii) optimization of area weights {ai}, and (iii) optimization
of confidence coefficients {ci}. Within an outer iteration, we first alternate between stages (i) and (ii) until the area
weights stabilize. Specifically, letting ab
i denote the area weights at the beginning of the current a-optimization stage,
we monitor the average relative change δa = 1
n
Pn
i=1
(ai −ab
i)/ab
i
. We stop alternating between the orientation
update and the area-weight optimization when δa ≤ϵa. We then proceed to stage (iii) to update {ci}. After finishing
the confidence optimization, we advance to the next outer iteration. The details of each stage are described next. See
Algorithm 1 for high-level pseudo-code.
7

<!-- page 8 -->
ALGORITHM 1: Dirichlet Winding Reconstruction (DiWR)
Input: Unoriented points {pi}n
i=1, area-weight threshold ϵa, normal-update threshold ϵn, maximum outer iterations tmax, and
balancing weights λi (i = 1, . . . , 5)
Output: Reconstructed watertight surface S
for each input point pi do
ci ←1
Initialize ai via local 2D Voronoi approximation
Randomly initialize ni as a unit vector
end
t ←1
repeat
repeat
// inner loop until area weights stabilize
Update {ni} using DWG with effective weights {aici}
Optimize {ai} by minimizing Eq. (9)
until δa ≤ϵa;
Optimize {ci} by minimizing Eq. (10)
Compute normal change ∆n on high-confidence points
t ←t + 1
until ∆n ≤ϵn or t > tmax;
Retain high-confidence points I = {i | ci ≥τin}
Run sPSR on {(pi, ni)}i∈I (optionally with screening weights proportional to {aici}i∈I) to obtain S
return S
Optimizing Area Weights {ai}
With orientations {ni} and confidence coefficients c fixed, we optimize the area
weights a by minimizing a weighted sum of the Dirichlet energy, the surface-point term, and the stable-area term:
min
a
bEdiri(N, a, c) + λ1 Esurf(N, a, c) + λ2 Earea(a, c),
(9)
where λ1, λ2 > 0 are balancing weights that control the trade-off among the terms.
Optimizing Confidence Coefficients {ci}
Since our optimization is staged, the current winding field provides a
useful signal for separating inliers from outliers. We therefore reset the confidence coefficients at the beginning of
each c-optimization stage using the current field. The reset consists of a coarse split followed by a refinement step
(see Figure 4). (i) Coarse inlier/outlier split from winding values. We apply bi-means clustering [Shen et al. 2009;
Wang and Feng 2015] to the current winding values {wθ(pi)}, obtaining two clusters. Let w := 1
n
Pn
i=1 wθ(pi) be
the global mean winding value. We treat as outliers the cluster whose mean winding value has the larger absolute
deviation from w. Points in this cluster are assigned ci ←0, while points in the other cluster are assigned ci ←1. (ii)
Refinement via density stratification. The binary assignment above can be overly coarse, especially near regions
where the winding values are ambiguous. To obtain a smoother initialization, we stratify the points into 128 density
levels according to their precomputed densities ρi. Within each level, we replace the bi-means clustering induced binary
confidence by average confidence of the points in the same level, yielding multi-level values in [0, 1] (see Figure 5).
To avoid disrupting points that are already consistent with current surfaces, we keep ci unchanged for points whose
winding values lie close to the global mean (i.e., ∥wθ(pi) −w| ≤0.1 in our implementation). This refinement produces
an initialization that reflects both global winding-field consistency and local sampling density.
Starting from the above initialization, we optimize c while keeping orientations N and area weights a fixed. The
objective is
min
c∈[0,1]n bEdiri(N, a, c) + λ3Esurf(N, a, c) + λ4Earea(a, c) + λ5Econf(c),
(10)
where Econf encourages near-binary confidences and the weights λ3, λ4, λ5 > 0 balance the terms.
Updating Point Orientation {ni}
To update the orientations {ni}, we leverage DWG [Liu et al. 2025a], a parallel
and GPU-friendly algorithm that iteratively constructs a winding field from randomly initialized normals and drives
them toward global consistency. In our implementation, we use DWG as a black-box normal update operator: given the
current orientations {ni} and the effective per-point weights {aici}, DWG returns an updated set of orientations.
4.3
Reconstruction
After optimization, we obtain per-point orientations {ni}, area weights {ai}, and confidence coefficients {ci}. We
then reconstruct a watertight surface using Screened Poisson Surface Reconstruction (sPSR) [Kazhdan and Hoppe
8

<!-- page 9 -->
2013]. Due to the polarization term (Eq. (8)), the optimized confidence coefficients become approximately binary. We
therefore treat low-confidence points as outliers and retain only points with ci ≥τin. The retained points, along with
their orientations ni, define the input normal field for sPSR.
A key difference from the standard sPSR pipeline is that we optionally provide sPSR with per-point weights derived from
our optimization. In sPSR, the screened term controls how strongly the reconstructed implicit function is encouraged to
match prescribed values at the sample locations, and the default setting treats samples uniformly [Kazhdan and Hoppe
2013]. In our pipeline, when the input exhibits strong non-uniform sampling and/or residual outliers, we weight the
screened term using the effective area weights ciai. This rebalances the contribution of samples across regions with
different densities and further reduces the influence of low-confidence points, preventing densely sampled areas from
dominating the reconstruction.
On easier inputs with relatively low sampling non-uniformity bu and/or low noise bσ and outlier rate bo, we find that
directly feeding the oriented points into sPSR without additional weighting already produces high-quality watertight
reconstructions. This is because sPSR is robust to mild sampling variation and moderate defections through its global
formulation and adaptive discretization. We therefore mainly enable the weighted screened term for challenging cases
where non-uniformity and outliers are more severe.
5
Experimental Results
DIWR repeatedly invokes DWG [Liu et al. 2025a], a GPU-based algorithm, for point orientation. To reduce the
overhead caused by frequent data transfers between the CPU and GPU, we implement the core components of DIWR
in CUDA. Optimization is performed using RMSProp, which is chosen for its simplicity and ease of use in parallel
processing. All experiments are conducted on a single NVIDIA GeForce RTX 4090 GPU. For an input point cloud
with 100K points, one run of area-weight optimization, confidence optimization, and DWG-based orientation update
typically takes 30, 20, and 2 seconds, respectively. CPU-based baselines are evaluated on a high-end workstation
equipped with an Intel Xeon(R) Gold 6430 CPU and 128 GB of RAM.
5.1
Test Models
We evaluate on 25 models from three categories: (i) point clouds produced by a computer-vision pipeline, specifically
the recent vision foundation model VGGT [Wang et al. 2025]; (ii) point clouds extracted from 3D Gaussian Splatting
(3DGS) reconstructions of multi-view images [Kerbl et al. 2023]; and (iii) commonly used graphics benchmarks.
For the 3DGS category, we use multi-view images from the OmniObject3D dataset [Wu et al. 2023], which provides
ground-truth meshes for evaluation. These images are background-free, allowing 3DGS [Kerbl et al. 2023] to produce
comparatively clean Gaussian primitives. In this setting, most spurious primitives appear as interior outliers, i.e.,
Gaussians located inside the true surface boundary. Many of these primitives have non-zero opacity and contribute
to rendering, but they do not correspond to valid surface geometry. The reconstructed Gaussian centers also exhibit
non-uniform sampling, especially in areas with little texture and on planar or low-curvature surfaces. For each Gaussian
primitive, we use its center as an input point; no other Gaussian attributes (e.g., opacity, anisotropy, or color) are used.
The computer vision category is built from more challenging multi-view images [Yao et al. 2020] with cluttered
backgrounds and less controlled capture conditions. Under a sparse-view setting (five to ten views), the point clouds
generated by VGGT [Wang et al. 2025] are typically dense but often contain substantially more noise and outliers.
Moreover, even when the underlying geometry is a single-layer surface, the reconstructed points frequently form shells
around it, that is, several closely spaced layers enclosing the true surface. When background structures lie close to the
object of interest, additional spurious surface sheets may also appear around the object. These imperfections make
accurate surface reconstruction significantly more difficult than the 3DGS category.
The graphics-benchmark category includes three widely used models: Armadillo, Dragon and Kitten. For each model,
we generate four variants by applying spatially varying non-uniform sampling, perturbing point positions (by up to
2% times the bounding-box side length), and injecting outliers (up to 20%). This results in 4 variants per model and
a total of 12 test cases in this category. Variants of the same base model are indexed by a subscript indicating the
corruption level, with larger subscripts corresponding to higher levels of distortion and difficulty. In addition, we
perform a controlled stress test using the Bunny model, creating 125 test cases that span a broad range of noise levels,
outlier rates, and sampling non-uniformity. Complete results for this stress test are provided in the Appendix.
To quantify the difficulty of the test models, we report three measures that capture local noise bσ, sampling non-uniformity
bu, and outlier contamination bo, where lower values indicate higher-quality input point clouds. The definitions of these
measures are provided in the Appendix. Figure 6 visualizes the distribution of our test models in this measure space.
9

<!-- page 10 -->
Figure 6: Distribution of test models in the quality-measure space (bσ, bo, bu). The x-axis shows the noise level bσ, the
y-axis shows the outlier rate bo, and the color map encodes the sampling non-uniformity bu. Each marker corresponds to
one input point cloud. Insets illustrate typical defects: uneven sampling, noise, outliers (both interior and exterior),
near-surface “thickness”, and dense but detached sheet-like fragments around the object.
Model
MSP
WNNC
WNNC⋆
FaCE
FaCE⋆
DWG
DWG⋆
NSH
NSH⋆
LoSF-UDF LoSF-UDF⋆
Ours
Name
n
bu
bσ
bo
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
Category
3DGS
Boat
30,384
0.29 0.0015 0.17 43.96 0.70
5.52
0.86
9.52
0.85
5.76
0.93 17.03 0.76 19.19 0.69 15.11 0.77
6.46
0.86
9.11
0.88 14.80 0.80 15.87
0.89
4.63 0.96
Doll
59,237
0.32 0.0024 0.16 65.40 0.64 24.50 0.75
8.64
0.93
4.47
0.97
8.64
0.94 13.97 0.86
8.95
0.95 21.98 0.79 14.14 0.88 40.00 0.81 50.44
0.88
4.78 0.98
Orna
133,111 0.41 0.0021 0.17 28.02 0.83 27.31 0.72 11.60 0.91 11.81 0.95 15.19 0.92 24.51 0.79 19.02 0.93 17.47 0.81 21.12 0.85 24.89 0.80 33.93
0.88
9.38 0.95
Sofa
199,601 0.36 0.0015 0.13 24.03 0.81 25.16 0.74 11.07 0.89 13.28 0.94 17.99 0.86 18.12 0.82 24.10 0.82 14.53 0.85 18.56 0.81 16.92 0.85 29.59
0.92
8.91 0.95
Mean (9 models)
100,268 0.33 0.0012 0.11 32.44 0.76 21.37 0.75 10.30 0.89
7.46
0.93 11.56 0.88 16.25 0.82 15.52 0.84 15.68 0.81 16.48 0.84 23.72 0.78 30.83
0.87
6.74 0.93
Graphics
Armadillo4
180,000 0.31 0.0027 0.16
9.04
0.87 23.45 0.77 10.57 0.83 64.24 0.63 17.11 0.81 58.13 0.63 11.28 0.78 15.90 0.63
7.37
0.86 14.66 0.58 26.50
0.74
5.79 0.90
Dragon4
120,000 0.29 0.0027 0.16 10.23 0.89 12.65 0.78
9.43
0.85 32.29 0.73 10.51 0.87 22.37 0.74 16.65 0.75 19.04 0.66
8.52
0.84 10.36 0.64 33.41
0.52
5.13 0.91
Kitten4
60,000
0.34 0.0034 0.17 30.40 0.90 18.20 0.80 18.81 0.86 28.99 0.73 16.98 0.92 31.80 0.68 28.35 0.88 18.66 0.74 17.05 0.84 83.37 0.53 64.55
0.75
6.10 0.97
Mean (12 models) 118,750 0.38 0.0015 0.16 14.03 0.92 25.08 0.73
8.90
0.90 34.85 0.73 10.07 0.93 36.26 0.67 13.34 0.89 44.19 0.56
5.96
0.67 14.07 0.53 16.30
0.56
4.11 0.96
Table 1: Statistics on test models from the 3DGS category and graphics benchmarks, where ground truth meshes are
available. Here n is the number of input points, and bu, bσ, and bo are quality measures of sampling non-uniformity,
noise level, and outlier rate, respectively. Lower values indicate higher-quality input point clouds. A method name
marked with ⋆indicates that outlier removal (PointCleanNet) and denoising (PCDNF) are applied as preprocessing.
Chamfer distances (CD, ↓) are multiplied by 103, and normal consistency (NC, ↑) is reported as cosine similarity, where
1 indicates perfect agreement. The best results are highlighted in red. Due to space constraints, the table shows results
on seven representative models, and reports category-wise average metrics. More results are provided in the Appendix.
The models cover a wide range of difficulties, with many in the middle-to-right regions, indicating substantial noise,
non-uniform sampling, and outlier rates that create challenging reconstruction scenarios.
5.2
Baselines
We compare DIWR with (i) a traditional multi-stage pipeline (MSP) that performs denoising/filtering, normal orien-
tation, and reconstruction from oriented points, (ii) recent methods that jointly solve normal orientation and surface
reconstruction from unoriented inputs, and (iii) deep learning methods, including both supervised and self-supervised
approaches.
For the multi-stage pipeline, we use a state-of-the-art method at each stage. Specifically, for outlier removal we use
PointCleanNet [Rakotosaona et al. 2020], a learning-based approach that predicts an outlier label per point and removes
the detected outliers. For denoising, we use PCDNF [Liu et al. 2024b], an end-to-end network that denoises point
clouds while jointly filtering and estimating normals to better preserve geometric features. For normal orientation,
we use Dipole [Metzer et al. 2021], which first predicts locally coherent normal directions within patches and then
propagates orientations globally via dipole propagation to obtain globally consistent normals. Finally, given the filtered
and oriented points, we reconstruct a watertight surface using sPSR [Kazhdan and Hoppe 2013].
We also include recent joint orientation-reconstruction baselines, including WNNC [Lin et al. 2024], DWG [Liu et al.
2025a], and FaCE [Scrivener et al. 2025], to evaluate the benefit of jointly optimizing orientation, area weights, and
confidence within a unified framework. Since these methods are not primarily designed for severe noise and high outlier
rates, we additionally report results with PointCleanNet and PCDNF as preprocessing to reduce their sensitivity to
corrupted inputs. For completeness, we also report results without preprocessing.
Deep learning methods are often limited to small- to middle-scale point clouds due to their high GPU memory demands
during training and inference. We compare against two representative methods. NSH [Wang et al.
2023a] is a
self-supervised approach that fits a neural signed distance field directly from unoriented point clouds; beyond the
standard first-order Eikonal regularization, it enforces a singular-Hessian constraint to stabilize training and suppress
spurious geometry. LoSF-UDF [Hu et al. 2025] is a supervised framework that learns an unsigned distance field (UDF)
10

<!-- page 11 -->
Input/Filtered pts
WNNC
FaCE
DWG
NSH
DiWR/Optimized
pts
Figure 7: Representative qualitative results on point clouds extracted from (top) 3DGS primitives, (middle) a computer-
vision pipeline (VGGT), and (bottom) a degraded graphics benchmark. The leftmost column shows the raw input
and the preprocessed point cloud after outlier removal and denoising. For each baseline, we report results on both the
corrupted input (top row) and the preprocessed input (bottom row). FaCE is unable to produce reliable results on the
VGGT point cloud due to the large number of input points. For DIWR, we also visualize the effective weights aici
as heatmaps; in sparsely sampled regions, inliers are assigned higher optimized area weights, compensating for low
sampling density. All visualizations are rendered at high resolution to support close-up inspection; see the Appendix for
more results.
from local shape functions. It trains a lightweight attention-based network on synthetic, smooth local patches with
known ground-truth distances, enabling the predictor to infer UDF values from a fixed-radius neighborhood around
each query point. This local, patch-based formulation reduces reliance on global topology and encourages the network
to focus on locally consistent geometric cues, which makes LoSF-UDF comparatively resilient to moderate noise and
sparse outliers, provided that each neighborhood still contains sufficient inlier support.
5.3
Comparison
Table 1 reports quantitative results for the 3DGS and graphics benchmark categories, where ground-truth meshes are
available. This allows us to evaluate reconstruction quality using standard accuracy metrics such as Chamfer Distance
(CD) and Normal Consistency (NC). For models from the computer vision pipeline category, where ground truth is
11

<!-- page 12 -->
(a)
(b)
(c)
(d)
Figure 8: Suppressing dense spurious sheets in a VGGT point cloud (n = 962, 492, bσ = 7.2 × 10−4, bo = 0.17, bu =
0.25). (a) The input contains dense near-surface layers and detached sheet-like fragments that are locally coherent
and can be mistaken as true surface evidence by baseline methods. Because these sheets do not form closed boundary
surfaces, they are inconsistent with a smooth winding field and can be suppressed by DIWR. (b) Initializing all points
with uniform confidence coefficients and weights, DWG produces a winding field with sharp local variations around
these structures, resulting in a high discrete Dirichlet energy bEdiri = 3.35. (c) After DIWR optimization, points on
isolated sheets are assigned low confidence and small effective weights aici, and the winding field becomes smoother
with a reduced energy bEdiri = 0.22. (d) As a result, the final reconstructed surface contains substantially fewer artifacts
caused by these spurious sheets.
unavailable, we instead provide qualitative results in Figure 7, with additional visualizations included in the Appendix.
Overall, DIWR achieves consistently strong performance across all three model categories, whereas no baseline attains
comparable results across all categories. We show representative results in Figures 1 and 7, and we provide a detailed
analysis below. More results are included in the Appendix.
For the 3DGS category, our results show that interior outliers and non-uniform sampling pose major challenges to GWN-
based methods such as WNNC and DWG. Interior outliers corrupt the discrete winding-number integration, which
ideally aggregates contributions only from boundary samples, thereby biasing the induced field and its gradients. In
addition, both WNNC and DWG are sensitive to non-uniform sampling and tend to degrade in sparsely sampled regions.
Although DIWR is also based on GWN, it explicitly optimizes per-point area weights and confidence coefficients. The
confidence coefficients suppress interior outliers by down-weighting their contributions to the winding field, while
the area weights compensate for non-uniform sampling by rebalancing the discrete integration. As a result, DIWR
improves robustness to both interior outliers and sampling irregularity. FaCE addresses the issue of interior outliers
through a different mechanism: it simulates a Faraday-cage effect to create a conductive enclosure that shields the
interior from external fields. This makes FaCE inherently robust to interior contamination, enabling it to reconstruct
complete geometry. Quantitatively, averaged over nine 3DGS models, FaCE achieves performance comparable to
DIWR in terms of CD and NC. Qualitatively, however, DIWR often produces cleaner surfaces with fewer artifacts (see
Figure 7 (top) and the Appendix).
Point clouds produced by VGGT often contain dense near-surface layers and detached sheet-like fragments. Although
these structures are locally coherent and can appear plausible, they often fail to represent a closed boundary and
therefore do not support a globally consistent inside-outside indicator. When incorporated into the discrete winding-
number construction, such fragments introduce sharp spatial variations in the winding field and increase its (discrete)
Dirichlet energy. By explicitly regularizing the winding field via Dirichlet energy and jointly optimizing the confidence
coefficients, DIWR reduces the influence of these fragments by driving their confidences toward 0, which shrinks their
effective weightsaici. This allows the reconstruction to focus on the subset of samples that collectively supports a
smooth, globally consistent winding field, resulting in substantially fewer sheet-induced artifacts in the final watertight
surface (Figure 8). In our experiments, baseline methods are particularly sensitive to these near-surface layers and
detached sheets; even with outlier removal and denoising as preprocessing, residual fragments often remain and continue
to bias the reconstruction.
For the graphics benchmarks, the injected non-uniform sampling, measurement noise, and high outlier rates make
surface reconstruction particularly challenging. When applied directly to these corrupted inputs, all baseline methods
fail to recover coherent surfaces. Pre-filtering the points substantially improves their outputs; however, as discussed
above, any errors introduced during preprocessing (e.g., imperfect outlier removal or over-aggressive denoising) are
effectively irreversible and cannot be corrected in subsequent orientation and/or reconstruction stages. Consequently,
the baselines still often produce reconstructions with missing regions or severe geometric distortions. In contrast, our
12

<!-- page 13 -->
joint optimization removes the need for a separate preprocessing step and yields visually plausible surfaces under these
challenging conditions.
6
Conclusion
We presented DIWR, a robust method for reconstructing 3D watertight surfaces from unoriented point clouds with
imperfections such as uneven sampling, noise, and outliers. In contrast to existing pipelines that treat preprocessing,
normal orientation and surface reconstruction as separate steps, DIWR solves these subproblems jointly: it optimizes
point orientations together with per-point confidence coefficients and adaptive area weights, coupled through the
induced GWN field in a unified formulation. Extensive experiments demonstrate that this joint optimization improves
robustness on challenging inputs and produces visually plausible watertight surfaces across a wide range of conditions.
References
Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe. Poisson surface reconstruction. In Proceedings of the Fourth
Eurographics Symposium on Geometry Processing, SGP ’06, page 61–70, 2006.
Michael Kazhdan and Hugues Hoppe. Screened poisson surface reconstruction. ACM Trans. Graph., 32(3):1–13, 2013.
Hui Huang, Dan Li, Hao Zhang, Uri Ascher, and Daniel Cohen-Or. Consolidation of unorganized point clouds for
surface reconstruction. ACM Trans. Graph., 28(5):1–7, December 2009.
Siyou Lin, Zuoqiang Shi, and Yebin Liu. Fast and globally consistent normal orientation based on the winding number
normal consistency. ACM Trans. Graph., 43(6), 2024.
Weizhou Liu, Jiaze Li, Xuhui Chen, Fei Hou, Shiqing Xin, Xingce Wang, Zhongke Wu, Chen Qian, and Ying He.
Diffusing winding gradients (dwg): A parallel and scalable method for 3d reconstruction from unoriented point
clouds. ACM Trans. Graph., 44(2), 2025a.
Daniel Scrivener, Daniel Cui, Ellis Coldren, S. Mazdak Abulnaga, Mikhail Bessmeltsev, and Edward Chien. Faraday
cage estimation of normals for point clouds and ribbon sketches. ACM Trans. Graph., 44(4), July 2025.
Alec Jacobson, Ladislav Kavan, and Olga Sorkine-Hornung. Robust inside-outside segmentation using generalized
winding numbers. ACM Trans. Graph., 32(4), 2013.
Herbert Edelsbrunner and Ernst P. Mücke. Three-dimensional alpha shapes. ACM Trans. Graph., 13(1):43–72, January
1994.
F. Bernardini, J. Mittleman, H. Rushmeier, C. Silva, and G. Taubin. The ball-pivoting algorithm for surface reconstruc-
tion. IEEE Transactions on Visualization and Computer Graphics, 5(4):349–359, 1999.
Nina Amenta, Sunghee Choi, and Ravi Krishna Kolluri. The power crust. In Proceedings of the sixth ACM symposium
on Solid modeling and applications, pages 249–266, 2001.
Tamal K. Dey and Samrat Goswami. Tight cocone: A water-tight surface reconstructor. In Proceedings of the Eighth
ACM Symposium on Solid Modeling and Applications, pages 127–134, 2003.
Hugues Hoppe, Tony DeRose, Tom Duchamp, John McDonald, and Werner Stuetzle. Surface reconstruction from
unorganized points. In Proceedings of the 19th annual conference on computer graphics and interactive techniques,
pages 71–78, 1992.
Yutaka Ohtake, Alexander Belyaev, Marc Alexa, Greg Turk, and Hans-Peter Seidel. Multi-level partition of unity
implicits. ACM Trans. Graph., 22(3):463–470, 2003.
Fei Hou, Chiyu Wang, Wencheng Wang, Hong Qin, Chen Qian, and Ying He. Iterative poisson surface reconstruction
(ipsr) for unoriented points. ACM Trans. Graph., 41(4), 2022.
Yueji Ma, Yanzun Meng, Dong Xiao, Zuoqiang Shi, and Bin Wang. Flipping-based iterative surface reconstruction for
unoriented points. Computer Aided Geometric Design, 111:102315, 2024.
Siyou Lin, Dong Xiao, Zuoqiang Shi, and Bin Wang. Surface reconstruction from point clouds without normals by
parametrizing the gauss formula. ACM Trans. Graph., 42(2), 2022.
Yueji Ma, Dong Xiao, Zuoqiang Shi, and Bin Wang. Convection augmented gauss reconstruction for unoriented point
clouds. ACM Trans. Graph., 44(5), 2025.
Gavin Barill, Neil G Dickson, Ryan Schmidt, David IW Levin, and Alec Jacobson. Fast winding numbers for soups
and clouds. ACM Trans. Graph., 37(4):1–12, 2018.
13

<!-- page 14 -->
Rui Xu, Zhiyang Dou, Ningna Wang, Shiqing Xin, Shuangmin Chen, Mingyan Jiang, Xiaohu Guo, Wenping Wang, and
Changhe Tu. Globally consistent normal orientation for point clouds by regularizing the winding-number field. ACM
Trans. Graph., 42(4), 2023.
Weizhou Liu, Xingce Wang, Haichuan Zhao, Xingfei Xue, Zhongke Wu, Xuequan Lu, and Ying He. Consistent point
orientation for manifold surfaces via boundary integration. In ACM SIGGRAPH 2024 Conference Papers, pages
1–11, 2024a.
Craig Gotsman and Kai Hormann. A linear method to consistently orient normals of a 3d point cloud. In ACM
SIGGRAPH 2024 Conference Papers, SIGGRAPH ’24, 2024.
Guojin Huang, Qing Fang, Zheng Zhang, Ligang Liu, and Xiao-Ming Fu. Stochastic normal orientation for point
clouds. ACM Trans. Graph., 43(6), 2024.
Zhiyang Huang, Nathan Carr, and Tao Ju. Variational implicit point set surfaces. ACM Trans. Graph., 38(4):1–13, 2019.
Jianjun Xia and Tao Ju. Variational surface reconstruction using natural neighbors. ACM Trans. Graph., 44(4), July
2025.
Baorui Ma, Zhizhong Han, Yu-Shen Liu, and Matthias Zwicker. Neural-pull: Learning signed distance function from
point clouds by learning to pull space onto surface. In Proceedings of the 38th International Conference on Machine
Learning, ICML 2021, volume 139, pages 7246–7257, 2021.
Chao Chen, Yu-Shen Liu, and Zhizhong Han. Gridpull: Towards scalability in learning implicit representations from 3d
point clouds. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 18276–18288, 2023.
Zixiong Wang, Yunxiao Zhang, Rui Xu, Fan Zhang, Peng-Shuai Wang, Shuangmin Chen, Shiqing Xin, Wenping Wang,
and Changhe Tu. Neural-singular-hessian: Implicit neural representation of unoriented point clouds by enforcing
singular hessian. ACM Trans. Graph., 42(6), December 2023a.
Ruian Wang, Zixiong Wang, Yunxiao Zhang, Shuangmin Chen, Shiqing Xin, Changhe Tu, and Wenping Wang. Aligning
gradient and hessian for neural signed distance function. In NeurIPS, 2023b.
Zixiong Wang, Pengfei Wang, Peng-Shuai Wang, Qiujie Dong, Junjie Gao, Shuangmin Chen, Shiqing Xin, Changhe Tu,
and Wenping Wang. Neural-imls: Self-supervised implicit moving least-squares network for surface reconstruction.
IEEE Transactions on Visualization and Computer Graphics, 30(8):5018–5033, 2024.
Siyu Ren, Junhui Hou, Xiaodong Chen, Ying He, and Wenping Wang. Geoudf: Surface reconstruction from 3d point
clouds via geometry-guided distance representation. In ICCV, pages 14214–14224, 2023.
Shengtao Li, Ge Gao, Yudong Liu, Ming Gu, and Yu-Shen Liu. Implicit filtering for learning neural signed distance
functions from 3d point clouds. In European Conference on Computer Vision, pages 234–251, 2024.
Jiangbei Hu, Yanggeng Li, Fei Hou, Junhui Hou, Zhebin Zhang, Shengfa Wang, Na Lei, and Ying He. A Lightweight
UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions . In CVPR, pages 1297–1307,
June 2025.
Cheng Xu, Fei Hou, Wencheng Wang, Hong Qin, Zhebin Zhang, and Ying He. Details enhancement in unsigned
distance field learning for high-fidelity 3d surface reconstruction. AAAI, 39(8):8806–8814, Apr. 2025.
Xiaoning Wang, Xiang Ying, Yong-Jin Liu, Shi-Qing Xin, Wenping Wang, Xianfeng Gu, Wolfgang Mueller-Wittig,
and Ying He. Intrinsic computation of centroidal voronoi tessellation (cvt) on meshes. Computer-Aided Design, 58:
51–61, 2015.
Kenshi Takayama, Alec Jacobson, Ladislav Kavan, and Olga Sorkine-Hornung. Consistently orienting facets in polygon
meshes by minimizing the dirichlet energy of generalized winding numbers. CoRR, abs/1406.5431, 2014.
Qingnan Zhou, Eitan Grinspun, Denis Zorin, and Alec Jacobson. Mesh arrangements for solid geometry. ACM Trans.
Graph., 35(4), July 2016.
Shibo Liu, Ligang Liu, and Xiao-Ming Fu. Closed-form generalized winding numbers of rational parametric curves for
robust containment queries. ACM Trans. Graph., 44(4), July 2025b.
Bruce D Jones and John R Williams. Fast computation of accurate sphere-cube intersection volume. Engineering
Computations, 34(4):1204–1216, 2017.
Jie Shen, David Yoon, David Shehu, and Shang-Yeu Chang. Spectral moving removal of non-isolated surface outlier
clusters. Computer-Aided Design, 41(4):256–267, 2009. ISSN 0010-4485. doi: https://doi.org/10.1016/j.cad.2008.
09.003. Point-based Computational Techniques.
Yutao Wang and Hsi-Yung Feng. Outlier detection for scanned point clouds using majority voting. Computer-Aided
Design, 62:31–43, 2015. ISSN 0010-4485. doi: https://doi.org/10.1016/j.cad.2014.11.004.
14

<!-- page 15 -->
Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual
geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 5294–5306, June 2025.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4), July 2023.
Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Liang Pan Jiawei Ren, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian,
Dahua Lin, and Ziwei Liu. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction
and generation. In CVPR, 2023.
Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A
large-scale dataset for generalized multi-view stereo networks. In CVPR, pages 1790–1799, 2020.
Marie-Julie Rakotosaona, Vittorio La Barbera, Paul Guerrero, Niloy J. Mitra, and Maks Ovsjanikov. Pointcleannet:
Learning to denoise and remove outliers from dense point clouds. Computer Graphics Forum, 39(1):185–203, 2020.
Zheng Liu, Yaowu Zhao, Sijing Zhan, Yuanyuan Liu, Renjie Chen, and Ying He. Pcdnf: Revisiting learning-based
point cloud denoising via joint normal filtering. IEEE Transactions on Visualization and Computer Graphics, 30(8):
5419–5436, August 2024b.
Gal Metzer, Rana Hanocka, Denis Zorin, Raja Giryes, Daniele Panozzo, and Daniel Cohen-Or. Orienting point clouds
with dipole propagation. ACM Trans. Graph., 40(4), July 2021.
Radu Bogdan Rusu, Zoltan Csaba Marton, Nico Blodow, Mihai Dolha, and Michael Beetz. Towards 3d point cloud
based object maps for household environments. Robotics and Autonomous Systems, 56(11):927–941, 2008.
Appendix
In the Appendix, we introduce quantitative model-quality measures (Section A), recommend typical parameter settings
based on input corruption levels (Section B), and present extensive stress tests to evaluate the robustness of DIWR
(Section C). We also discuss several design choices in DIWR (Section E), report runtime and memory consumption
(Section D), provide full per-model statistics, and include additional qualitative comparisons (Section F). Finally, we
discuss the limitations of our method (Section G).
A
Model-Quality Measures
To quantitatively characterize the imperfections of our test inputs and to stratify their difficulty, we define three
model-quality measures that capture (i) noise/roughness, (ii) sampling non-uniformity, and (iii) outlier contamination.
All measures are computed from simple statistics of the input point set. We uniformly scale each model into a unit cube.
For each point pi, let Ni,k denote its k-nearest neighbors, and let Πi be the least-squares best-fit plane to Ni,k. We use
k ∈[10, 40] in practice.
We first estimate a global sampling scale from the average kNN distance:
si := 1
k
X
q∈Ni,k
∥pi −q∥,
bs := mediani si.
Here, si measures the local inter-point spacing around pi, and bs provides a robust estimate of the typical spacing of the
input.
We then measure local noise (or roughness) using the plane-fit residual:
σi :=
v
u
u
t1
k
X
q∈Ni,k
dist
 q, Πi
2,
bσ := mediani σi.
The median bσ summarizes the overall noise/roughness level.
To quantify global sampling irregularity, we measure the relative dispersion of the local scales {si}. Standard dispersion
metrics can be dominated by extreme density peaks (e.g., overlapping points) or by distant outliers, so we use a trimmed
coefficient of variation. Let s(1) ≤s(2) ≤· · · ≤s(n) be the sorted values of {si}, and let Sτ be the subset obtained by
discarding the bottom and top τ% values (we use τ = 10). We then define
bu := σ(Sτ)
µ(Sτ),
15

<!-- page 16 -->
where µ(·) and σ(·) are the sample mean and standard deviation. Smaller bu indicates more uniform sampling, while
larger values reflect stronger density variation.
Finally, we estimate outlier contamination from the global distribution of {si}, following the statistical outlier removal
idea [Rusu et al. 2008]. We compute the global mean µs and standard deviation σs of {si}, and classify points with
unusually sparse neighborhoods (large si) as outliers:
bo := #{ i | si > µs + 2σs }
n
,
where n is the number of input points. This measure serves as a simple proxy for the fraction of points that are isolated
or lie in low-density regions relative to the bulk of the data.
B
Parameter Settings
Our algorithm has several user-specified parameters: the area-weight convergence threshold ϵa, the normal-update
convergence threshold ϵn, the maximum number of iterations tmax, and the weights λi (i = 1, . . . , 5) used in the
area-weight optimization (Eq. (9)) and confidence optimization (Eq. (10)).
We use the following model-independent defaults in all experiments: ϵa = 0.15, ϵn = 0.02, and tmax = 10. The
remaining parameters are set based on the three model-quality measures bσ, bo, and bu, of the input point cloud, which
provide a global summary of noise/roughness, sampling non-uniformity, and outlier contamination for each model.
We fix the weight of the discrete Dirichlet-energy term to 1.0. The remaining weights λi in Eq. (9) and Eq. (10) are
scheduled during optimization to improve stability. Specifically, we start with relatively small weights λi(i = 1, · · · , 5),
because in early iterations the orientations, area weights, and confidence coefficients are still inaccurate and the winding
field benefits from stronger smoothness regularization. As the solution stabilizes, we gradually increase these weights
λi to strength the influence of constraint terms. For easy to moderate models with bσ ≤0.002, bu ≤0.3 and bo ≤0.08,
we use the initial weights λ1 = 5.0, λ2 = 1.0, λ3 = 1.0, λ4 = 0.5, and λ5 = 5 × 10−3. For more severely corrupted
inputs, we recommend smaller initial weights (e.g., halving the above values) so that the Dirichlet-energy regularization
remains dominant for longer.
C
Stress Tests
Setup
To thoroughly evaluate the robustness of DIWR under increasing corruption, we conduct controlled stress tests
on the Bunny model. Starting from a clean point cloud, we progressively increase the difficulty by varying sampling
non-uniformity bu, noise level bσ, and outlier rate bo. We sample each factor at five levels, yielding 53 = 125 test cases
in total. These resulting models cover the ranges bσ ∈[0.00018, 0.0064], bo ∈[0.042, 0.22], and bu ∈[0.10, 0.87] in a
roughly uniform manner. Figure 9 shows the distribution of these test cases in the quality-measure space (bσ, bo, bu).
Easy Cases
When sampling non-uniformity is mild (bu ≤0.3), the location disturbance is small (bσ ≤0.002), and the
outlier rate is low (bo ≤0.08), DIWR preserves fine-scale details and produces clean watertight reconstructions. In this
regime, most points receive high confidence, and the optimized solution stays close to the input geometry.
Moderate Cases
As sampling becomes more uneven (0.3 < bu ≤0.7) and both noise and outlier levels increase
(0.002 < bσ ≤0.005 and 0.08 < bo ≤0.17), the estimated confidence coefficients become increasingly bimodal: reliable
samples concentrate near high confidence, whereas outliers (and inconsistent measurements) are pushed toward low
confidence. Consequently, points that are incompatible with the recovered surface are down-weighed, while consistent
inliers continue to guide orientation and reconstruction. At this corruption level, baseline methods often exhibit typical
failure modes such as locally flipped regions, spurious surface components, or missing regions and over-smoothing
caused by aggressive preprocessing, whereas DIWR still produces coherent watertight surfaces.
Difficult Cases
When sampling becomes highly uneven (bu ≥0.7) or corruption is severe (bσ ≥0.005 or bo ≥0.17),
the input may no longer contain sufficient reliable geometric evidence to support accurate reconstruction. In such cases,
DIWR may produce incomplete or overly simplified surfaces, or fail to recover the correct structure.
Summary
For each test case, we not only measure the quantitative accuracy metrics via Chamfer distance and
normal consistency, we also visually inspect the reconstructed surfaces. We call a construction successful, if the overall
shape is reconstructed completely, i.e., no major geometric component is missing and there is no significant distortion.
Overall, DIWR successfully reconstructs 88% of the stress-test cases, spanning a wide range of noise, outliers, and
16

<!-- page 17 -->
non-uniform sampling, while baselines break down at substantially lower corruption levels. These results highlight the
benefit of jointly optimizing orientation and confidence: as input quality deteriorates, DIWR increasingly down-weights
unreliable samples and maintains stable behavior until the available inlier evidence becomes insufficient for meaningful
surface recovery. For comparison, the baseline methods’ successful rates range between 20% and 40% (see Fig. 10 for
visualization results).
bo = 0.089
bo = 0.16
bo = 0.14
bo = 0.17
bo = 0.21
bσ = 0.0025
bσ = 0.0043
bσ = 0.0052
bσ = 0.0053
bσ = 0.0053
bu = 0.28
bu = 0.53
bu = 0.35
bu = 0.46
bu = 0.50
Figure 9: Stress tests on the Bunny model with varying noise level bσ, outlier rate bo, and sampling non-uniformity bu. We
visualize the distribution of the 125 test cases in the 3D measure space (bσ, bo, bu): cases near the origin correspond to
low noise, low outlier rates, and nearly uniform sampling, while cases toward the far corner are the most challenging.
We also highlight five representative test cases (in red), approximately sampled along the main diagonal of the space, to
illustrate increasing levels of corruption from easy to hard.
D
Runtime and Memory
DIWR alternates three core components per iteration: area-weight optimization, confidence-coefficient optimization,
and a DWG-based orientation update. Both optimization subproblems are solved with RMSProp and run on the GPU.
The orientation update is performed with DWG [Liu et al. 2025a], which is highly parallel and also runs on the GPU.
After convergence, we invoke sPSR on the CPU to extract a watertight surface from the oriented points using the
effective weights {ciai}.
Because the iterative stages are fully GPU-parallel, the overall runtime is dominated by these three GPU components,
followed by a single CPU call to sPSR. For an input point cloud with n = 100K points, one run of area-weight
Input points
Filtered points
WNNC
WNNC⋆
FaCE
FaCE⋆
Ours
Visualization of aici
Figure 10: Stress-test results on five representative input cases with increasing difficulty. See also the accompanying
video.
17

<!-- page 18 -->
optimization, confidence optimization, and DWG-based orientation update takes approximately 30, 20, and 2 seconds,
respectively.
On inputs with moderate imperfections, DIWR typically converges in less than 5 iterations. For more challenging cases,
additional iterations may be required. Empirically, the orientations (and the corresponding reconstructed surfaces)
change only marginally beyond 10 iterations; we therefore set tmax = 10 as a hard stopping criterion.
The GPU memory footprint of DIWR is dominated by per-point arrays (e.g., ni, ai, ci, and their gradients), and thus
scales linearly with the number of points. RMSProp introduces additional per-variable state (e.g., running averages
of squared gradients) for a and c, which also scales as O(n). DWG has linear space complexity in practice, storing
per-point quantities and its internal acceleration structures on the GPU. To evaluate the Dirichlet energy, we additionally
allocate grid samples Q within the bounding box B together with their weights δq. This adds an O(|Q|) term. In
our implementation, |Q| is fixed by a constant-resolution grid (typically 643), so the overal memory complexity is
O(n + |Q|) = O(n).
E
Discussions
Why are Per-point Area Weights and Confidence Coefficients Separate Variables?
In our formulation, the
generalized winding number uses the product aici as the contribution of point pi. Although one could merge them
into a single per-point weight, we keep ai (area weight) and ci (confidence) as separate variables because they encode
different phenomena and should obey different priors. The area weight ai approximates the local surface element
represented by pi and is mainly used to compensate for non-uniform sampling; thus, ai is expected to vary smoothly
and remain consistent with local spacing and a global area (or mass) budget. In contrast, the confidence coefficient
ci ∈[0, 1] models sample reliability and is used to suppress outliers; in practice, ci often becomes sparse and close
to binary. Collapsing ai and ci into a single variable would mix these roles, making it difficult to impose appropriate
regularization and increasing ambiguity in the optimization. Separating them improves robustness, interpretability, and
numerical stability.
Why Not Optimize Point Locations?
Although optimizing point locations could further correct geometric pertur-
bations, DIWR does not treat {pi} as optimization variables. First, the derivative of the winding field with respect
to point positions is substantially more complex than the derivatives with respect to orientations and weights, and
evaluating it efficiently and stably at scale would add significant algorithmic and computational overhead. Second,
DIWR builds on DWG as the reconstruction backbone, which already tolerates moderate positional noise when the
per-point area weights ai are reasonably estimated. Empirically, jointly optimizing orientations, area weights, and
confidence coefficients captures most of the robustness gains in our target scenarios, while keeping the optimization
simple, stable, and scalable.
F
Additional Results
We provide qualitative and quantitative results for all the test models in the 3DGS and graphics-benchmark categories
(see Table 2 and Figures 11 and 13). For VGGT point clouds, ground-truth geometry is not available; we therefore
report qualitative results only (Figure 12).
We observe that preprocessing can be overly aggressive under severe corruption. After outlier removal and denoising,
the filtered point clouds often contain missing regions or become overly sparse in parts of the shape. Since subsequent
orientation and reconstruction stages cannot recover geometry that has been removed, many baselines exhibit a drop in
overall performance in terms of NC and CD (due to missing geometry or large distortions), even though the remaining
reconstructed regions may appear cleaner locally.
We also observe that LoSF-UDF [Hu et al. 2025] is particularly sensitive to density variation, largely because its
inference operates on local patches extracted with a fixed-radius neighborhood. In sparsely sampled regions, a fixed-
radius patch may contain only a small number of points (or even be nearly empty), providing insufficient geometric
evidence for the pre-trained network to extract meaningful features, from which reliable unsigned distances can be
inferred. Consequently, LoSF-UDF tends to produce unstable or incomplete reconstructions in low-density areas, even
when it performs well in densely sampled regions.
18

<!-- page 19 -->
Images
Raw pts
n = 30, 384 n = 100, 199 n = 59, 237 n = 89, 462 n = 82, 640 n = 133, 111 n = 108, 774 n = 99, 008
bσ = 0.0015
bσ = 0.0013
bσ = 0.0024
bσ = 0.0011
bσ = 0.0012
bσ = 0.0021
bσ = 0.0025
bσ = 0.0011
bo = 0.17
bo = 0.13
bo = 0.16
bo = 0.15
bo = 0.16
bo = 0.17
bo = 0.16
bo = 0.13
bu = 0.29
bu = 0.30
bu = 0.32
bu = 0.34
bu = 0.53
bu = 0.41
bu = 0.38
bu = 0.31
Filtered pts
MSP
WNNC
WNNC*
FaCE
FaCE*
NSH
NSH*
Ours
Figure 11: Results on the 3DGS category.
19

<!-- page 20 -->
Raw pts
Filtered pts
n = 636, 311
n = 298, 836
n = 341, 870
bσ = 0.00058
bσ = 0.00080
bσ = 0.00041
bo = 0.14
bo = 0.13
bo = 0.10
bu = 0.21
bu = 0.21
bu = 0.33
Images
Ours
WNNC
WNNC*
FaCE
FaCE*
NSH
NSH*
Figure 12: Results on the VGGT category.
Raw pts
Filtered pts
WNNC
WNNC⋆
DWG
DWG⋆
FaCe
FaCe⋆
LoSF
LoSF⋆
Ours
Figure 13: More results (Top: Kitte4; Bottom: Armadillo4) from common graphics benchmarks degraded with non-
uniform sampling, perturbed point positions and injected outliers.
20

<!-- page 21 -->
Model
MSP
WNNC
WNNC⋆
FaCE
FaCE⋆
DWG
DWG⋆
NSH
NSH⋆
LoSF-UDF LoSF-UDF⋆
Ours
Name
n
bu
bσ
bo
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
CD
NC
Category
3DGS
Boat
30,384
0.29
0.0015
0.17 43.96 0.70
5.52
0.86
9.52
0.85
5.76
0.93 17.03 0.76 19.19 0.69 15.11 0.77
6.46
0.86
9.11
0.88 14.80 0.80 15.87
0.89
4.63 0.96
Car
100,199 0.30
0.0013
0.13 29.24 0.80 17.34 0.77 11.56 0.88
5.79
0.93 11.45 0.89
5.34
0.91 21.53 0.84 12.40 0.80 11.65 0.86 11.60 0.77 15.10
0.86
4.76 0.92
Doll
59,237
0.32
0.0024
0.16 65.40 0.64 24.50 0.75
8.64
0.93
4.47
0.97
8.64
0.94 13.97 0.86
8.95
0.95 21.98 0.79 14.14 0.88 40.00 0.81 50.44
0.88
4.78 0.98
Light031
89,462
0.34
0.0011
0.15 20.46 0.82
5.76
0.85
5.58
0.90
4.72
0.94
5.16
0.91
6.34
0.89
9.79
0.78
7.41
0.83
5.35
0.92
6.44
0.83
9.74
0.89
4.97 0.92
Light032
82,640
0.53
0.0012
0.16 24.68 0.82 32.91 0.72
7.55
0.94
5.07
0.96
7.56
0.95 29.10 0.82 13.41 0.94 31.77 0.78 30.58 0.81 54.08 0.82 64.66
0.86
5.03 0.96
Orna
133,111 0.41
0.0021
0.17 28.02 0.83 27.31 0.72 11.60 0.91 11.81 0.95 15.19 0.92 24.51 0.79 19.02 0.93 17.47 0.81 21.12 0.85 24.89 0.80 33.93
0.88
9.38 0.95
Sofa
199,601 0.36
0.0015
0.13 24.03 0.81 25.16 0.74 11.07 0.89 13.28 0.94 17.99 0.86 18.12 0.82 24.10 0.82 14.53 0.85 18.56 0.81 16.92 0.85 29.59
0.92
8.91 0.95
Suit
108,774 0.38
0.0025
0.16 28.58 0.73 39.44 0.63 19.21 0.69
9.88
0.85 10.20 0.87 18.39 0.79 12.21 0.89 18.76 0.75 26.56 0.75 18.17 0.67 35.25
0.83
8.26 0.89
Train
99,008
0.31
0.0011
0.13 27.53 0.71 14.44 0.75 12.16 0.85
6.34
0.93 10.78 0.85 11.27 0.80 15.58 0.67 10.34 0.81 11.22 0.82 24.64 0.73 22.86
0.83
9.94 0.88
Mean (9 models)
100,268 0.33
0.0012
0.11 32.44 0.76 21.37 0.75 10.30 0.89
7.46
0.93 11.56 0.88 16.25 0.82 15.52 0.84 15.68 0.81 16.48 0.84 23.72 0.78 30.83
0.87
6.74 0.93
Graphics
Armadillo1
180,000 0.31
0.0025
0.16
7.37
0.90 23.09 0.77
6.66
0.89 60.35 0.65
5.05
0.91 38.29 0.66
9.55
0.78 13.86 0.64
5.96
0.88 22.57 0.59 22.60
0.73
4.97 0.92
Armadillo2
180,000 0.53
0.00017
0.16
3.28
0.97 60.69 0.60
2.31
0.98 43.97 0.69
2.47
0.98 47.68 0.64
3.37
0.93 36.06 0.76
2.27
0.98
4.58
0.95
6.53
0.96
2.42 0.98
Armadillo3
172,500 0.30
0.00016
0.13
3.29
0.97 56.00 0.60
2.32
0.97 45.65 0.71
2.46
0.98 50.36 0.60
3.37
0.94 29.85 0.81
2.27
0.97
4.65
0.95
6.54
0.96
2.44 0.98
Armadillo4
180,000 0.31
0.0027
0.16
9.04
0.87 23.45 0.77 10.57 0.83 64.24 0.63 17.11 0.81 58.13 0.63 11.28 0.78 15.90 0.63
7.37
0.86 14.66 0.58 26.50
0.74
5.79 0.90
Dragon1
120,000 0.31
0.0024
0.16 16.80 0.86 14.24 0.77 11.60 0.85 31.22 0.74 13.11 0.88 38.97 0.67 15.90 0.88 23.88 0.68 10.47 0.86 14.29 0.67 42.43
0.53
4.67 0.93
Dragon2
115,000 0.36
0.00016
0.13 17.50 0.88 20.47 0.67
6.20
0.91 24.68 0.78
7.39
0.96 26.98 0.66
7.04
0.95 28.47 0.72
5.15
0.97 21.47 0.76 30.45
0.55
2.86 0.98
Dragon3
120,000 0.48
0.00018
0.16 25.16 0.94 23.46 0.67
5.02
0.92 26.49 0.77
6.82
0.96 25.33 0.70
6.91
0.94 33.31 0.69
4.61
0.97 20.67 0.75 34.28
0.52
3.45 0.97
Dragon4
120,000 0.29
0.0027
0.16 10.23 0.89 12.65 0.78
9.43
0.85 32.29 0.73 10.51 0.87 22.37 0.74 16.65 0.75 19.04 0.66
8.52
0.84 10.36 0.64 33.41
0.52
5.13 0.91
Kitten1
60,000
0.35
0.0031
0.17 30.45 0.91 15.75 0.81 15.98 0.89 25.43 0.75 15.03 0.92 32.78 0.66 28.64 0.87 16.66 0.76 26.40 0.81 29.20 0.68 59.49
0.76
5.26 0.98
Kitten2
60,000
0.54 0.000068 0.17
7.10
0.98 17.93 0.78
8.94
0.94 18.33 0.82 11.69 0.95 31.93 0.72 13.43 0.96 25.16 0.76 16.13 0.87 18.21 0.94 41.78
0.84
3.18 0.99
Kitten3
57,500
0.44 0.000062 0.14
7.67
0.98 15.01 0.80
8.90
0.94 16.54 0.84 12.22 0.96 30.53 0.69 15.65 0.92 23.08 0.80 16.29 0.87 23.41 0.92 41.40
0.85
2.97 0.99
Kitten4
60,000
0.34
0.0034
0.17 30.40 0.90 18.20 0.80 18.81 0.86 28.99 0.73 16.98 0.92 31.80 0.68 28.35 0.88 18.66 0.74 17.05 0.84 83.37 0.53 64.55
0.75
6.10 0.97
Mean (12 models) 118,750 0.38
0.0015
0.16 14.03 0.92 25.08 0.73
8.90
0.90 34.85 0.73 10.07 0.93 36.26 0.67 13.34 0.89 44.19 0.56
5.96
0.67 14.07 0.53 16.30
0.56
4.11 0.96
Table 2: Statistics on all test models from the 3DGS category and graphics benchmarks, where ground truth meshes are
available. The best results are highlighted in red.
G
Limitations
DIWR is built on the generalized winding number field and is therefore primarily suited for reconstructing watertight,
manifold surfaces, similar to other GWN-based methods (e.g., GCNO [Xu et al. 2023], BIM [Liu et al. 2024a],
WNNC [Lin et al. 2024], and DWG [Liu et al. 2025a]). Extending the formulation to open surfaces with boundaries or
to non-manifold structures remains challenging, especially when the input is heavily affected by non-uniform sampling,
noise, and outliers.
Although DIWR improves robustness compared to existing approaches, it can still fail in extreme cases where the point
cloud provides too little reliable geometric evidence to guide the winding-field optimization (e.g., very sparse sampling
combined with strong noise and a high outlier ratio, like the most challenging cases in our stress tests). Such inputs are
inherently ambiguous, and Dirichlet regularization alone is insufficient to recover the correct geometry.
Finally, DIWR regularizes the winding field by minimizing Dirichlet energy, which favors smooth solutions and
therefore does not explicitly preserve sharp features or fine-scale details. As a result, CAD-like edges and corners, as
well as high-frequency geometric details, may be rounded or attenuated. Since our primary goal is robustness under
substantial input corruption, explicitly enforcing feature and detail preservation is challenging when the available
geometric evidence near such structures is sparse or unreliable.
21
