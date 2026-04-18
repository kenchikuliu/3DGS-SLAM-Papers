<!-- page 1 -->
I-PHYSGAUSSIAN: IMPLICIT PHYSICAL SIMULATION FOR 3D
GAUSSIAN SPLATTING
A PREPRINT
Yicheng Cao1, Zhuo Huang1, Yu Yao1, Yiming Ying2, Daoyi Dong3, Tongliang Liu1
1Sydney AI Centre, The University of Sydney; 2School of Mathematics and Statistics, University of Sydney;
3Australian Artificial Intelligence Institute, University of Technology Sydney
ABSTRACT
Physical simulation predicts future states of objects based on material properties and external loads,
enabling blueprints for both Industry and Engineering to conduct risk management. Current 3D
reconstruction-based simulators typically rely on explicit, step-wise updates, which are sensitive
to step time and suffer from rapid accuracy degradation under complicated scenarios, such as
high-stiffness materials or quasi-static movement. To address this, we introduce i-PhysGaussian,
a framework that couples 3D Gaussian Splatting (3DGS) with an implicit Material Point Method
(MPM) integrator. Unlike explicit methods, our solution obtains an end-of-step state by minimizing
a momentum-balance residual through implicit Newton-type optimization with a GMRES solver.
This formulation significantly reduces time-step sensitivity and ensures physical consistency. Our
results demonstrate that i-PhysGaussian maintains stability at up to 20× larger time steps than explicit
baselines, preserving structural coherence and smooth motion even in complex dynamic transitions.
1
Introduction
Recent advances in dynamic scene generation have enabled realistic spatiotemporal content for applications such as
robotic manipulation [1, 2, 3, 4] and video synthesis [5, 6, 7, 8]. Within this broader landscape, physics-based dynamic
simulation seeks to generate temporally coherent motion that strictly respects physical laws by coupling neural scene
representations with classical mechanics solvers. This capability is critical for forecasting object behavior under unseen
loading conditions [9, 10, 11, 12], enabling long-horizon analysis and design with high fidelity. For instance, in a
robotic manipulation task, a physics-grounded model allows a robot to simulate whether a planned grasp will induce
excessive deformation or slippage under varying forces before physical execution.
Physical simulation helps understand a real-world system by predicting the next state from the current configuration,
leading to broad applications in computer graphics and computational mechanics. The widely used approach is the
Material Point Method (MPM) [13], which iteratively computes subsequent steps via a hybrid particle-grid formulation
based on material properties and external priors. By combining Lagrangian particles with a background Eulerian grid,
MPM effectively handles large deformations and topological changes while retaining the computational convenience
and efficiency of regular grids.
Despite its success, most MPM-based reconstruction-driven simulators [14, 15, 16, 17, 18, 19] advance dynamics using
explicit time stepping. These methods approximate within-step evolution through start-of-step evaluations, applying a
single forward update to the system state. While adequate for sufficiently small intervals, numerical error and instability
grow rapidly as the time step increases, particularly when handling stiff materials, strong nonlinearities, or contact-rich
dynamics. Consequently, explicit works are strictly constrained by Courant–Friedrichs–Lewy (CFL) conditions [20].
These stability requirements often necessitate prohibitively small time steps, which not only increase computational
overhead but also accumulate discretization errors, thus limiting the practicality of long-horizon physical prediction.
These limitations motivate within-step modeling. We propose i-PhysGaussian, which replaces explicit stepping with an
Implicit MPM integrator. Concretely, we define a within-step momentum-balance residual that measures the mismatch
between the end-of-step momentum change and the impulse induced by internal stresses and external loads over the step.
We defer the full discretization and its relation to the underlying MPM transfer scheme to Section 2. We then iteratively
arXiv:2602.17117v1  [cs.LG]  19 Feb 2026

<!-- page 2 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
solve for the end-of-step state so that this residual becomes small, using a Newton-type optimization (with GMRES for
the linearized subproblems). This implicit formulation substantially relaxes time-step sensitivity and improves physical
consistency over a much wider range of step sizes. While implicit stepping incurs additional per-step optimization, it
can reduce the total number of integration steps by enabling larger stable time steps.
Our system comprises two modules: (1) a scene-representation (SR) module built on 3D Gaussian Splatting (3DGS) [21],
responsible for parsing the initial scene and rendering simulation results; and (2) an implicit physics simulation (i-
PS) module built on implicit MPM, responsible for predicting end-of-step states. Given a static 3DGS scene and a
configuration file specifying material parameters, simulation domain, and applied loads, the SR module preprocesses
the scene and converts it into a set of simulation particles (including surface Gaussians and optionally filled interior
particles following PhysGaussian). The i-PS module advances these particles by implicit stepping and returns the
updated state for rendering. Repeating this loop yields a dynamic 4D Gaussian sequence, a time-varying set of 3D
Gaussians (with positions and optionally other attributes) over discrete time steps.
In summary, our contributions are as follows:
• To the best of our knowledge, we are the first to integrate Newton–GMRES fully implicit MPM time stepping
into a 3DGS-based reconstruction-driven dynamic pipeline for 4D Gaussian sequences.
• We introduce a within-step momentum-balance formulation solved by Newton–GMRES, which substantially
improves robustness to large time steps and enhances physical consistency compared to explicit baselines.
• Open-source implementations of implicit MPM are scarce and predominantly C++; we will release a clean
Python implementation to facilitate reproducibility and future research.
1.1
Physics-Based Dynamic Simulation
With the rise of explicit scene representations (e.g., 3DGS), physics-based dynamic simulation has emerged as a
promising route for generating temporally coherent dynamics from reconstructed scenes [22]. Recent work increasingly
integrates differentiable rendering with physical priors to reduce non-physical drift under prescribed material properties
and external loads.
Recent methods couple 3DGS with physics solvers for physically plausible 4D Gaussian dynamics. Gaussian Splash-
ing [23] combines 3DGS with Position-Based Dynamics (PBD) [24, 25], while PhysGaussian [14] integrates 3DGS
with MPM [26, 27]. MPM is more suitable for large deformations and contact-rich dynamics and admits effective dif-
ferentiable implementations. Building on this paradigm, PIDG [28] treats Gaussians as Lagrangian material points and
enforces momentum-residual constraints for improved physical consistency. AS-DiffMPM [29] extends differentiable
MPM to system identification with arbitrarily shaped colliders, enabling richer object–environment interactions, while
GaussianFluent [30] further supports mixed materials and brittle fracture via continuum-damage MPM and synthesized
volumetric interiors.
A key bottleneck is time integration: explicit stepping can become unstable or inaccurate as the time step grows,
especially for stiff or highly nonlinear/contact-rich dynamics. Orthogonally, diffusion/video supervision [16, 31, 19]
and short-sequence supervision [18, 32, 33] reduce manual configuration by inferring physical parameters, while
feed-forward dynamics models (e.g., PhysGM [34]) amortize reconstruction and prediction for faster 4D synthesis.
Interactive systems (e.g., PhysTalk [35], VR-GS [36], GS-Verse [37]) emphasize real-time control. Even with these
advances, robustness under deliberately enlarged time steps remains comparatively underexamined.
2
Methodology
We propose i-PhysGaussian, built upon the PhysGaussian framework. As illustrated in Fig. 1, the system takes two
inputs: (i) a static 3D Gaussian scene and (ii) a configuration file specifying preprocessing, simulation, and rendering
options. The pipeline proceeds as follows. First, the input scene is preprocessed (e.g., aligned to the XYZ axes,
removing near-transparent Gaussian kernels, and applying particle filling in target regions). Next, the prepared scene,
together with the simulation parameters (materials and boundary conditions), is fed to an implicit MPM solver to predict
the next state. Each end-of-step state is then re-rendered as a Gaussian scene at the corresponding time instant. Iterating
this loop and stacking the per-step scenes along the temporal axis yields a 4D Gaussian representation (4DGS).
The distinction lies in the MPM time integration. Explicit MPM uses quantities evaluated at the beginning of the
step (forward Euler) and then transfers back to particles to obtain the end-of-step variables directly. As a result, it is
extremely sensitive to step time and prone to collapse in practice. In contrast, we adopt a standard implicit formulation
that couples start- and end-of-step variables via a within-step momentum residual, and compute the end-of-step state
2

<!-- page 3 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Figure 1: Overview of i-PhysGaussian. The pipeline starts with (i) a static 3D Gaussian scene and (ii) a configuration
file specifying preprocessing, simulation, and rendering options. The scene is first preprocessed (axis alignment, pruning
near-transparent Gaussians, and optional particle filling), then enhanced by an implicit MPM solver under the specified
material and boundary conditions. The end-of-step state is re-rendered as a Gaussian scene at each time iteratively,
which produces a temporally stacked 4D Gaussian representation (4DGS).
with Newton–GMRES until the residual meets a prescribed tolerance, thus demonstrating enhanced stability. Details
follow in Sec. 2.2.
2.1
Preliminaries
We review the standard pipeline and preliminaries for physical simulation under 3D Gaussian scenes.
3D Gaussian Splatting.
3DGS [21] represents a static scene as a set of anisotropic 3D Gaussian kernels. Let the
complete Gaussian scene be {Gp}P
p=1 including P material particles, with each kernelized as Gp = (µp, αp, Σp, βp),
where µp ∈R3 is the kernel center, αp ∈(0, 1] is the opacity, Σp ∈R3×3 is the (positive-definite) covariance matrix,
and βp is the spherical-harmonic coefficients for color.
To render a novel view for a given camera, each 3D Gaussian is projected onto the image plane as a 2D elliptical
footprint. The color at each pixel can be expressed as
C(u)=
K(u)
X
k=1
Tk(u) aπ(k)(u)cπ(k)(u)+TK(u)+1(u) Cbg,
(1)
where u denotes a pixel, π is the back-to-front order by Gaussian depth, cπ(k)(u) is the RGB Gaussian color,
aπ(k)(u) ∈[0, 1] is the per-pixel opacity from the projected 2D footprint, and Tk(u) = Q
j<k
 1 −aπ(j)(u)

is the
accumulated transmittance. Additionally, the last term is affected by default background Gaussian color Cbg. In our
work, we input static 3DGS and interpret each kernel as a simulation particle, which provides seamless compatibility:
simulated states can be rendered via Eq. (1) without additional transformation or conversion. Next, we employ 3DGS
under Physical scenarios via internal particle filling.
Internal Particle Filling.
Since 3DGS places Gaussians primarily on visible surfaces, the reconstructed geometry is
hollow. Therefore, it restricts dynamics to a thin shell, causing distortion and collapse under physics settings.
To recover volumetric matter, Xie et al. [14] proposed an internal particle filling procedure, where the voxelized target
region is analyzed by casting a ray along 3 axes, ±x, ±y, ±z. Further, based on the even–odd parity rule, we can
decide whether each voxel center aligns inside or outside the shape. For interior points, a filling Gaussian particle
is instantiated by employing the appearance parameters αp, βp from the nearest surface Gaussian. Afterwards, we
proceed to the mechanics of continuum movement.
3

<!-- page 4 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Continuum Mechanics.
To study the deformation and displacement of objects, we denote an initial object space Ω0
and a deformed one Ωt at time t. A material point moves according to a motion function x = ϕ(X, t). The deformation
gradient is F(X, t) = ∇Xϕ(X, t), which encodes local stretch, rotation, and shear. Further, to ensure physical
consistency, there are two constraints, namely mass conservation and momentum conservation. Mass conservation
ensures the particles stay consistent during movement without losing or creating new ones, formally,
Z
Bt ρ(x, t) dx =
Z
B0 ρ0(X) dX ,
(2)
where B0 ⊂Ω0 and Bt = ϕ(B0, t) ⊂Ωt denote the same set of points at the initial and the time-t, respectively. Then,
momentum conservation keeps the inertia consistent, which is balanced by the sum of internal and external forces.
ρ(x, t) ˙v(x, t) = ∇·σ(x, t) + f ext(x, t),
(3)
where ρ ˙v represents inertia, e.g., momentum rate per unit, ∇·σ is the internal force density generated by stresses; and
f ext is the external body-force density, e.g., gravity.
Such a continuum mechanics builds the foundation of motion and deformation. Further, we demonstrate how MPM can
explicitly solve the next-step movement prediction.
Explicit Material Point Method.
MPM solves dynamical update by alternating between particles and a background
Eulerian grid, which enables efficient and stable computation of movement. It starts from given particle states
{mp, xp, vp, Fp} at time t, and explicitly predict time t′ = t + ∆t via three stages: (i) particle to grid (P2G) which
leverages a shape function wIp to calculate the influence of particle mass on each grid node I, meanwhile creating the
grid mass mI and velocity vI; (ii) Grid update enables the next-step movement based on physical laws, including
velocity vI(t) update based on acceleration aI(t):
vI(t′)=vI(t) + ∆t aI(t), aI(t) = f int
I (t)+f ext
I
(t)
mI(t)
;
(4)
Further, (iii) grid to particle (G2P) transfers the updated grid velocities back to particles to obtain their positions.
vp(t′)=
X
I
wIp vI(t′), xp(t′) = xp(t) + ∆t vp(t′).
(5)
After new particle dynamics are obtained, it simultaneously updates deformation gradients, e.g., Fp, to ensure consis-
tency for the continuum mechanics.
Such an explicit MPM is simple but can become unstable or inaccurate for large ∆t, especially under stiff materials or
contact-rich dynamics. Thus, we are motivated to develop implicit time stepping to enhance MPM.
2.2
i-PhysGaussian: Implicit Material Point Method
Building on the above three-stage pipeline, our method starts from a static 3D Gaussian scene based on the material
configuration. Further, we align the scene to the eulerian frame with x–y–z axes and perform internal particle filling in
designated regions, which is initialized as Ω0, t = 0, with each Gaussian kernel denoted p. For particle-to-grid transfers,
our shape function wIp leverages the most common separable cubic B-spline function:
wIp =N
xI −xp
h

N
yI −yp
h

N
zI −zp
h

,
(6)
where (xp, yp, zp) is the position of particle p, (xI, yI, zI) is the coordinate of grid node I, h is the grid spacing, and
N(·) denotes the 1D cubic B-spline basis. Further, before G2P, the key contribution lies in our implicit grid update.
Within-step Momentum Residual.
Instead of accepting the outcome of an explicit update, we implicitly seek an
end-of-step state that is consistent with the impulse accumulated within a time step. To achieve this, we treat the volume
Ωt as a whole during trial movement to predict its potential outcome. Thus, based on the strong-form momentum
balance in Eq. (3), we have the weak form for grid node I:
Z
Ωt
ρNI ˙vdΩ= −
Z
Ωt
∇NI : σdΩ+
Z
Ωt
NIbdΩ+
Z
∂Ωt
NItdS,
(7)
where NI is the grid shape function, b is body-force density, and t is prescribed traction. Following implicit
MPM formulations [38, 39], we discretize the dynamics on the background grid and use a Newmark family update
4

<!-- page 5 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
parameterized by (β, γ). Critically, we aim to predict a trial state for the next step. Therefore, we introduce grid
displacement increment ∆uI over a step [tn, tn+1] to understand how the current movement is affected by the mechanics.
Particularly, considering an non-Dirichlet-constrained free node I ∈F, we induce the end-of-step kinematics based on
the Newmark relations:
an+1
I
(∆uI)= ∆uI −∆t vn
I −∆t2   1
2 −β

an
I
β ∆t2
,
(8)
vn+1
I
(∆uI)=vn
I +∆t

(1−γ)an
I +γ an+1
I
(∆uI)

.
(9)
To implicitly analyze the end-of-step trial, we quantify the mismatch between current inertia and the sum of internal
and external forces via a within-step momentum residual:
RI(∆u) = f ext
I
(∆u) + f int
I (∆u) −mI an+1
I
(∆uI), I ∈F,
(10)
where mI is the lumped grid mass for a grid node, and the internal force f int
I (∆u) is accumulated from particles using
the trial deformation and stress:
f int
I (∆u)=−
X
p
V 0
p Pp(∆u) ∇XNI(xp), Pp =τ p F−T
p
,
(11)
where V 0
p is the reference particle volume, Fp is the trial deformation gradient, τ p is the Kirchhoff stress from the
constitutive model, and ∇XNI denotes the shape-function gradient prescribed by configuration. Then, we aim to seek
the minimal-residual trial that balances the inertial and forces. During this process, we interpolate grid neighbors
surrounding the particle to obtain Fp(∆u), further updating the trial Ftrial
p
= (I + ∆t ∇vn+1) Fn
p. As a result, the
optimal trial is guaranteed for dynamic equilibrium with superior numerical stability, as shown in our experiments in
Sec. 3. Under the no-free lunch law, our trial process trades efficiency for physical consistency and temporal robustness,
which is justified in Secs. 3.2 and 3.3.
Dirichlet boundary conditions. For Dirichlet-constrained nodes I ∈D with prescribed movement, we enforce velocity
constraints by overwriting their state. Specifically, let vtar
I
be the prescribed velocity. We express the constraint in the
same Newmark form by defining a history term vhist
I
and a scalar S = γ/(β∆t), both determined by vn
I and an
I :
S ∆uI = vtar
I
−vhist
I
,
I ∈D,
(12)
which directly sets ∆uI for their end-of-step state. For evaluation, we only study convergence on free nodes F, and
exclude Dirichlet-constrained nodes to avoid mixing constraint-form and force-form residuals. Next, we carefully
explain our Newton-type solver to the implicit MPM.
2.3
Optimization
Next, we demonstrate our Newton-GMRES optimization.
Newton outer loop.
We solve the nonlinear system R(∆u) = 0 using an inexact Newton method with globalization.
We define the objective
ϕ(∆u) = 1
2∥R(∆u)∥2
F,
(13)
where ∥· ∥F is the ℓ2 norm accumulated over active free nodes, where ∆u(0) is initialized by a kinematic predictor,
∆u(0)
I
= ∆t vn
I + 1
2∆t2 an
I ,
I ∈F,
(14)
and then overwrite constrained nodes as in Eq. (12). At Newton iterate k, we linearize the residual around ∆u(k):
R(∆u(k) + δu) ≈R(∆u(k)) + J(∆u(k)) δu,
(15)
and compute an update direction by approximately solving
J(∆u(k)) δu(k) = −R(∆u(k)).
(16)
We employ an Eisenstat–Walker forcing term to adapt the inner linear-solve tolerance, improving efficiency by solving
early Newton steps more coarsely and tightening accuracy as the residual decreases.
To globalize the method, we perform a backtracking line search on ϕ and update
∆u(k+1) = ∆u(k) + α(k) δu(k).
(17)
We use an Armijo decrease test and a weak curvature check based on the directional derivative
ϕ′(0) =
D
R(∆u(k)), J(∆u(k)) δu(k)E
F .
(18)
If ϕ′(0) ≥0 (non-descent), we fall back to the steepest-descent direction δu = −R on free nodes. We terminate when
∥R(∆u)∥F falls below a prescribed tolerance or when the update stagnates.
5

<!-- page 6 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Table 1: BMF-gated stability frontier under time-step enlargement. For each scene and method, we report the largest
stable multiplier kmax and Fail.% (percentage of failed multipliers among all tested k values). We use kmax = 0 to
indicate that none of the tested multipliers passes the BMF gate.
FICUS
PILLOW2SOFA
BREAD
METHOD
kmax ↑FAIL.%↓kmax ↑
FAIL.%↓kmax ↑FAIL.%↓
i-PhysGaussian
20
0.0
20
0.0
20
0.0
PHYSGAUSSIAN
1
90.9
20
0.0
8
54.5
DREAMPHYSICS
2
90.9
14
45.5
1
90.9
PHYSICS3D
0
100.0
2
81.8
0
100.0
GMRES inner loop (right-preconditioned, matrix-free).
The Newton system (16) is large and sparse; assembling J
explicitly is expensive, especially with diverse constitutive models and plasticity. We therefore use right-preconditioned
GMRES and only require Jacobian–vector products (JVPs).
Matrix-free Jacobian action. Given the current iterate ∆u and a search direction p, we approximate
J(∆u) p ≈R(∆u + εp) −R(∆u −εp)
2ε
,
(19)
using a centered finite difference. The perturbation magnitude ε is chosen adaptively so that ∥εp∥∞is on the order
of 10−4, which stabilizes the numerical differentiation. We also project p to the free subspace (Dirichlet entries are
zeroed), ensuring hard constraints on JVPs.
Right preconditioner. We employ a diagonal right preconditioner W that combines an inertial term and a stiffness-like
diagonal accumulated during residual evaluation:
WI =
mI
β∆t2 + Kdiag
I
,
I ∈F,
(20)
where Kdiag
I
includes a material contribution (scaled by λ + 2µ) and an optional geometric term. We solve the
right-preconditioned system
J(∆u) W −1y = b,
b = −R(∆u),
(21)
by GMRES in the Krylov subspace generated by repeated application of the operator J(∆u) W −1. After obtaining y,
the Newton increment is recovered as
δu = W −1y.
(22)
GMRES details. GMRES builds an orthonormal basis via Arnoldi with (re)orthogonalization, where we use a two-pass
Modified Gram–Schmidt for numerical stability, apply Givens rotations to maintain an upper-triangular least-squares
system, and stop when the relative residual drops below a tolerance η prescribed by the outer Newton loop. The
resulting δu is returned to the Newton step, followed by line search and constraint overwrite as described above.
3
Experiments
In the experiments, we aim to address some core questions: (i) how far ∆t can be enlarged before runs become BMF-
invalid; (ii) how drift evolves with ∆t together with the BMF gate1; and (iii) whether improved physical simulation
comes at the expense of rendering.
3.1
Experiment Setup
Datasets.
We evaluate on two complementary sources that cover both realistic appearance complexity and controlled
physical interactions. PhysDreamer Dataset provides dense reconstructions from real-world captures with complex
geometry and high-frequency textures. We use Alocasia to stress thin structures and leaf-level details, and Hat to evaluate
fabric-like appearance where high-frequency patterns are visually salient. PhysGaussian Dataset offers synthetic and
visually cleaner scenes with controlled dynamics, which are well-suited for isolating simulator behavior under time-step
enlargement. We use Ficus (deformation under external loading), Bread (large deformation under interaction), and
Pillow2Sofa (contact-rich motion). We use PhysGaussian Dataset for physics metrics and PhysDreamer Dataset for
rendering metrics.
1values beyond kmax may be clamp-dominated
6

<!-- page 7 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Figure 2: BMF-gated failure rate by scene (and overall). Hatching is used only for visibility; the underlying value
remains 0.0%.
Baseline.
We compare against three reconstruction-driven pipelines. PhysGaussian couples 3DGS with an explicit
MPM simulator, isolating the effect of explicit vs. implicit stepping under the same rendering backbone. DreamPhysics
augments explicit physics with video supervision, testing how guidance interacts with reconstruction-driven dynamics.
Physics3D provides another video-guided baseline included in our unified trace-based evaluation. In contrast, i-
PhysGaussian integrates 3DGS with an implicit Newton–GMRES MPM solver, enabling stable integration at larger
time steps without video supervision.
In all evaluations, we only transform the position and shape of each 3DGS kernel. Particularly, positions are calculated
by the within-step residual solver, while shape is updated via the material constitutive law. The opacity and appearance
parameters are kept fixed.
3.2
Physical Simulation Fidelity
Because solvers expose different internal diagnostics, we adopt a unified trace-based protocol. We export per-particle
mass, volume, and trajectories from all methods and compute all metrics from these traces. We evaluate (i) stability
under time-step enlargement, (ii) time-step robustness relative to 1 × ∆t reported together with the BMF gate, and (iii)
reference-free physical plausibility at 1 × ∆t. We provide visual comparison videos in the supplementary material.
3.2.1
Stability
Evaluation Criteria.
Particles evolve within a bounded simulation domain [0, grid_lim]3. Any coordinate that
leaves the domain is hard-clamped to the boundary with a small margin ϵ = 10−6 (i.e., each component is clipped to
[ϵ, grid_lim −ϵ]). We refer to particles that trigger this clamping in a step as collapsed particles.
To quantify clamp-dominated degeneracy, we use the boundary-hit mass fraction (BMF), defined as the fraction of
total particle mass belonging to collapsed particles (see C.1 for the exact formula).
BMF-gated failure. Let T be the number of rendered frames. An iteration is considered failed if clamp dominance
persists over time, measured by the exceedance ratio r =
1
T
PT
t=1 I[BMFt > 0.5], and r > 0.5 leads to failure.
For each scene, we sweep time-step multipliers k ∈{1, 2, 4, . . . , 20} and apply the above gate to each run. Based on
the pass/fail outcomes, we report two aggregate indicators per method: (1) the largest multiplier that passes the gate,
denoted kmax; and (2) Fail.%, the percentage of failed multipliers among all tested k values. We also report the overall
failure rate aggregated over all settings.
Results and Analysis.
Tab. 1 reports the BMF-gated stability frontier (kmax) and Fail.% across time-step multipliers.
i-PhysGaussian is uniformly stable, reaching kmax = 20 with 0.0% failures on all scenes. PhysGaussian is stable on
Pillow2Sofa (kmax = 20) but breaks down on Ficus (kmax = 1) and Bread (kmax = 8), indicating reduced robustness
7

<!-- page 8 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
(a) Ficus
(b) Ficus
(c) Pillow2Sofa
(d) Pillow2Sofa
(e) Bread
(f) Bread
Figure 3: Time-step robustness with the BMF gate. Left column: COMD↓; right column: mwRMSD↓. Both are
plotted versus multiplier k relative to 1 × ∆t. Dashed lines mark kmax; values beyond kmax may be clamp-dominated.
of explicit stepping under harder dynamics. DreamPhysics and Physics3D fail much earlier on the challenging scenes,
with high failure rates and small kmax.
Fig. 2 visualizes the same trend: i-PhysGaussian has 0.0% failures throughout, while the baselines concentrate failures
on Ficus/Bread PhysGaussian or across all scenes (DreamPhysics/Physics3D). For a finer-grained view across all
multipliers, we defer the per-k stability heatmaps and additional analyses to Appendix C.
3.2.2
Time-step Robustness
Evaluation Criteria.
We quantify how enlarging the time step changes the resulting trajectories relative to the 1 × ∆t
reference, and interpret the drift curves jointly with the BMF gate. In particular, results beyond kmax may be dominated
by hard clamping and should be treated as diagnostic signals rather than faithful trajectory comparisons. We report two
complementary drift measures: COMD, which compares the mass-weighted center-of-mass trajectories to the 1 × ∆t
reference and reflects global motion bias; and mwRMSD, a mass-weighted particle-level deviation from the same
reference that captures local deformation and fine-grained motion changes. To avoid underestimating divergence due to
hard boundary clamping, mwRMSD applies a conservative penalty to particles that are clamped in either trajectory. See
Appendix D.1 and Appendix D.2 for full definitions.
Results and Analysis.
Stability does not necessarily imply trajectory correctness. Since particle-level ground truth is
unavailable, we treat the trajectory 1 × ∆t as a reference and measure how trajectories drift as the time step is enlarged.
To avoid subtle drift caused by boundary projection, mwRMSD applies a conservative penalty to particles clamped in
either trajectory, as shown in Appendix D.2.
8

<!-- page 9 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
(a) Ficus
(b) Ficus
(c) Pillow2Sofa
(d) Pillow2Sofa
(e) Bread
(f) Bread
Figure 4: Reference-free impulse/torque irregularity at 1 × ∆t (per-frame). Left column: impulse irregularity↓;
right column: torque irregularity↓. Lower values indicate smoother evolution of net impulse/torque (i.e. fewer bursty
non-physical jitters).
Fig. 3 reports drift versus the multiplier k (left: COMD; right: mwRMSD), with dashed lines indicating each method’s
stability frontier kmax. Across all three scenes, i-PhysGaussian consistently exhibits the lowest drift and the slowest
growth with k, indicating improved robustness at both global (center-of-mass) and particle levels. PhysGaussian remains
competitive on Pillow2Sofa but deviates noticeably on Bread as k increases, while DreamPhysics and Physics3D show
larger drift and stronger variability on the harder scenes. For a compact cross-k summary, we additionally report
normalized AUC scores in Appendix D.3.
3.2.3
Physical Plausibility
Evaluation Criteria.
Stability and trajectory-level consistency do not necessarily imply physical plausibility. When
particle-level ground truth is unavailable, our robustness metrics treat the 1×∆t trajectory as a proxy reference; however,
this reference itself may deviate from basic conservation principles due to numerical errors [40, 41], hard clamping,
or implementation details. To reduce reliance on a potentially imperfect reference, we conduct a reference-free
physical-consistency diagnosis at 1 × ∆t for each scene and method.
We report three trace-derived diagnostics from two complementary aspects—mass and momentum: (1) mass drift
(MassDrift), which checks whether the total mass changes over time; (2) impulse irregularity (∆2P), which captures
high-frequency irregular variations of the total linear momentum; and (3) torque irregularity (∆2L), which captures
9

<!-- page 10 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
(a) PSNR (↑)
(b) SSIM (↑)
(c) LPIPS (↓)
Figure 5: Static rendering fidelity at 1 × ∆t (first frame). We compare the first rendered frame against the paired
ground-truth image and report PSNR/SSIM/LPIPS (higher is better for PSNR/SSIM; lower is better for LPIPS).
high-frequency irregular variations of the total angular momentum. All quantities are computed directly from exported
traces and require no external-force decomposition. Full details are shown in Appendices E.1 and E.2.
Results and Analyses.
We confirm that mass conservation for all methods and scenes at 1×∆t, as shown in
Appendix E.3. Here, we focus on momentum-based diagnostics. Fig. 4 reports reference-free impulse and torque
irregularity, capturing bursty high-frequency jitters in the evolution of net linear/angular momentum. Across scenes,
i-PhysGaussian exhibits smoother curves with fewer spikes, indicating a continuous evolution of net impulse/torque.
Importantly, near-zero irregularity does not necessarily imply better physical behavior. In Ficus and Bread, some
baselines produce almost flat curves close to zero; the trajectory overlays in Fig. 8 suggest this is more consistent
with kinematic degeneration toward an almost static trajectory, where motion amplitude is strongly suppressed,
and thus momentum variations collapse. In Pillow2Sofa, all methods show aligned peaks that coincide with major
contact/collision moments, supporting that these diagnostics capture physically interpretable events. Additional
discussion and details are provided in Appendix E.4.
3.3
Dynamic Rendering Quality
We further examine the rendering-side behavior of physics-driven pipelines to verify that introducing an implicit MPM
solver does not compromise visual quality. Since rendering differences are most evident on real-captured, texture-rich
scenes, we adopt a complementary protocol: (i) first-frame fidelity against paired ground truth, and (ii) reference-free
exposure stability over the rendered sequence.
3.3.1
Static Rendering Fidelity
Evaluation Criteria.
We evaluate the first rendered frame (t=1) with the paired ground-truth image of each model
and report PSNR/SSIM/LPIPS to capture pixel-wise structure and perceptual fidelity.
Results and Analysis.
Fig. 5 summarizes first-frame fidelity on two real-captured scenes. Overall, i-PhysGaussian
matches PhysGaussian across PSNR/SSIM/LPIPS, indicating no systematic rendering degradation from using an
implicit solver. DreamPhysics is competitive on Alocasia but drops more on Hat, suggesting scene-dependent
sensitivity. Physics3D performs noticeably worse in PSNR/LPIPS, implying larger pixel-level and perceptual deviations,
while SSIM differences are generally smaller across methods.
3.3.2
Dynamic Exposure Stability
Evaluation Criteria.
Since real-captured scenes typically lack frame-aligned ground-truth videos, we adopt a
reference-free exposure diagnostic. We measure the per-frame saturation ratio (Sat.%), i.e., the fraction of near-
saturated (blown-out) pixels, and summarize the sequence by its mean/std/range (max–min).
Results and Analysis.
Tab. 2 shows that i-PhysGaussian maintains low saturation on both real-captured scenes
and closely matches PhysGaussian (mean Sat.%: 0.15 on Alocasia and 0.05 on Hat), indicating that introducing an
implicit MPM solver does not cause systematic over-exposure degradation. In contrast, Physics3D exhibits extremely
high saturation means (Alocasia: 98.14, Hat: 92.93) with near-zero variability, consistent with saturation-dominated
degenerate outputs. Overall, our pipeline preserves exposure stability on visually complex real scenes and remains
comparable to strong explicit-MPM baselines.
10

<!-- page 11 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Table 2: Over-exposure diagnostic at 1 × ∆t. We report saturation ratio statistics (mean/std/range), where SatRatiot
is the percentage of pixels whose luminance satisfies Yt(p) ≥0.98. Lower values indicate fewer over-exposed regions.
METHOD
SAT.% MEAN↓SAT.% STD↓SAT.% RANGE↓
ALOCASIA
I-PHYSGAUSSIAN
0.15
0.0105
0.0805
PHYSGAUSSIAN
0.15
0.0005
0.0046
DREAMPHYSICS
0.17
0.0003
0.0017
PHYSICS3D
98.14
0.0000
0.0000
HAT
I-PHYSGAUSSIAN
0.05
0.0005
0.0053
PHYSGAUSSIAN
0.05
0.0017
0.0218
DREAMPHYSICS
0.43
0.0000
0.0000
PHYSICS3D
92.93
0.0000
0.0000
4
Conclusion
In summary, i-PhysGaussian tightly couples 3D Gaussian Splatting with a fully implicit MPM integrator by enforcing
within-step momentum balance and solving the end-of-step state via Newton–GMRES, substantially mitigating the
step-size sensitivity and failure modes of explicit methods under stiff, highly nonlinear, and contact-rich dynamics;
experiments demonstrate consistent improvements in large-step stability, trajectory-level robustness to time-step
enlargement, and reference-free physical-consistency diagnostics, while preserving competitive rendering fidelity
and exposure stability on real, texture-rich scenes. These robustness gains, however, come at the cost of higher
per-step computation due to the nested implicit solver, particularly in challenging regimes, motivating future work
on streamlining the solver through stronger preconditioning and warm starts, adaptive inexact solves with early-exit
criteria, and reuse of intermediate quantities, to bridge the efficiency gap while retaining large-step stability for scaling
to larger scenes and longer-horizon simulations.
References
[1] Sangmin Lee, Sungyong Park, and Heewon Kim. Dynscene: Scalable generation of dynamic robotic manipulation
scenes for embodied ai. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages
12166–12175, 2025.
[2] Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson,
David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated robot learning via
generative simulation. arXiv preprint arXiv:2311.01455, 2023.
[3] Pushkal Katara, Zhou Xian, and Katerina Fragkiadaki. Gen2sim: Scaling up robot learning in simulation with
generative models. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6672–6679.
IEEE, 2024.
[4] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning
compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024.
[5] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller,
Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise
camera control. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6121–6132,
2025.
[6] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, László Jeni, Sergey
Tulyakov, and Hsin-Ying Lee. 4real: Towards photorealistic 4d scene generation via video diffusion models.
Advances in Neural Information Processing Systems, 37:45256–45280, 2024.
[7] Wen-Hsuan Chu, Lei Ke, and Katerina Fragkiadaki. Dreamscene4d: Dynamic multi-object scene generation from
monocular videos. Advances in Neural Information Processing Systems, 37:96181–96206, 2024.
[8] Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li,
Rynson Lau, Wangmeng Zuo, et al. Voyager: Long-range and world-consistent video diffusion for explorable 3d
scene generation. ACM Transactions on Graphics (TOG), 44(6):1–15, 2025.
11

<!-- page 12 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
[9] Dinesh Jayaraman and Kristen Grauman. Learning to look around: Intelligently exploring unseen environments
for unknown tasks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
1238–1247, 2018.
[10] Zhuo Huang, Xiaobo Xia, Li Shen, Bo Han, Mingming Gong, Chen Gong, and Tongliang Liu. Harnessing
out-of-distribution examples via augmenting content and style. In The Eleventh International Conference on
Learning Representations, 2023.
[11] Zhuo Huang, Muyang Li, Li Shen, Jun Yu, Chen Gong, Bo Han, and Tongliang Liu. Winning prize comes from
losing tickets: Improve invariant learning by exploring variant parameters for out-of-distribution generalization.
International Journal of Computer Vision, pages 1–19, 2024.
[12] Mingyu Li, Tao Zhou, Zhuo Huang, Jian Yang, Jie Yang, and Chen Gong. Dynamic weighted adversarial
learning for semi-supervised classification under intersectional class mismatch. ACM Transactions on Multimedia
Computing, Communications and Applications, 20(4):1–24, 2024.
[13] Chenfanfu Jiang, Craig Schroeder, Andrew Selle, Joseph Teran, and Alexey Stomakhin. The affine particle-in-cell
method. ACM Transactions on Graphics (TOG), 34(4):1–10, 2015.
[14] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian:
Physics-integrated 3d gaussians for generative dynamics. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 4389–4398, 2024.
[15] Yuchen Lin, Chenguo Lin, Jianjin Xu, and Yadong Mu. Omniphysgs: 3d constitutive gaussians for general
physics-based dynamics generation. arXiv preprint arXiv:2501.18982, 2025.
[16] Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui Li, Wangmeng Zuo, and Rynson WH Lau. Dream-
physics: Learning physics-based 3d dynamics with video diffusion priors. In Proceedings of the AAAI Conference
on Artificial Intelligence, volume 39, pages 3733–3741, 2025.
[17] Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and
Chuang Gan. Pac-nerf: Physics augmented continuum neural radiance fields for geometry-agnostic system
identification. arXiv preprint arXiv:2303.05512, 2023.
[18] Junhao Cai, Yuji Yang, Weihao Yuan, Yisheng He, Zilong Dong, Liefeng Bo, Hui Cheng, and Qifeng Chen. Gic:
Gaussian-informed continuum for physical property identification and simulation. Advances in Neural Information
Processing Systems, 37:75035–75063, 2024.
[19] Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan. Physics3d: Learning
physical properties of 3d gaussians via video diffusion. arXiv preprint arXiv:2406.04338, 2024.
[20] Richard Courant, Kurt Friedrichs, and Hans Lewy. Über die partiellen differenzengleichungen der mathematischen
physik. Mathematische annalen, 100(1):32–74, 1928.
[21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
[22] Zhuo Huang, Gang Niu, Bo Han, Masashi Sugiyama, and Tongliang Liu. Towards out-of-modal generaliza-
tion without instance-level modal correspondence. In The Thirteenth International Conference on Learning
Representations, 2025.
[23] Yutao Feng, Xiang Feng, Yintong Shang, Ying Jiang, Chang Yu, Zeshun Zong, Tianjia Shao, Hongzhi Wu, Kun
Zhou, Chenfanfu Jiang, et al. Gaussian splashing: Unified particles for versatile motion synthesis and rendering.
In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 518–529, 2025.
[24] Matthias Müller, Bruno Heidelberger, Marcus Hennix, and John Ratcliff. Position based dynamics. Journal of
Visual Communication and Image Representation, 18(2):109–118, 2007.
[25] Miles Macklin, Matthias Müller, and Nuttapong Chentanez. Xpbd: position-based simulation of compliant
constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games, pages 49–54,
2016.
[26] Deborah Sulsky, Zhen Chen, and Howard L Schreyer. A particle method for history-dependent materials. Computer
methods in applied mechanics and engineering, 118(1-2):179–196, 1994.
[27] Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana, and Chenfanfu Jiang. A moving
least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM
Transactions on Graphics (TOG), 37(4):1–14, 2018.
[28] Haoqin Hong, Ding Fan, Fubin Dou, Zhi-Li Zhou, Haoran Sun, Congcong Zhu, and Jingrun Chen. Physics-
informed deformable gaussian splatting: Towards unified constitutive laws for time-evolving material field. arXiv
preprint arXiv:2511.06299, 2025.
12

<!-- page 13 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
[29] Federico Vasile, Ri-Zhao Qiu, Lorenzo Natale, and Xiaolong Wang. Gaussian-augmented physics simulation and
system identification with complex colliders. arXiv preprint arXiv:2511.06846, 2025.
[30] Bei Huang, Yixin Chen, Ruijie Lu, Gang Zeng, Hongbin Zha, Yuru Pei, and Siyuan Huang. Gaussianfluent:
Gaussian simulation for dynamic scenes with mixed materials. arXiv preprint arXiv:2601.09265, 2026.
[31] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and
William T Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation. In European
Conference on Computer Vision, pages 388–406. Springer, 2024.
[32] Jinxi Li, Ziyang Song, Siyuan Zhou, and Bo Yang. Freegave: 3d physics learning from dynamic videos by gaussian
velocity. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12433–12443, 2025.
[33] Jinxi Li, Ziyang Song, and Bo Yang. Trace: Learning 3d gaussian physical dynamics from multi-view videos. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8820–8829, 2025.
[34] Chunji Lv, Zequn Chen, Donglin Di, Weinan Zhang, Hao Li, Wei Chen, Yinjie Lei, and Changsheng Li. Physgm:
Large physical gaussian model for feed-forward 4d synthesis. arXiv preprint arXiv:2508.13911, 2025.
[35] Luca Collorone, Mert Kiray, Indro Spinelli, Fabio Galasso, and Benjamin Busam. Phystalk: Language-driven
real-time physics in 3d gaussian scenes. arXiv preprint arXiv:2512.24986, 2025.
[36] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin
Yang, et al. Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In ACM
SIGGRAPH 2024 Conference Papers, pages 1–1, 2024.
[37] Anastasiya Pechko, Piotr Borycki, Joanna Waczy´nska, Daniel Barczyk, Agata Szyma´nska, Sławomir Tadeja, and
Przemysław Spurek. Gs-verse: Mesh-based gaussian splatting for physics-aware interaction in virtual reality.
arXiv preprint arXiv:2510.11878, 2025.
[38] James Edward Guilkey and Jeffrey A Weiss. Implicit time integration for the material point method: Quantitative
and algorithmic comparisons with the finite element method. International Journal for Numerical Methods in
Engineering, 57(9):1323–1338, 2003.
[39] Deborah Sulsky and A Kaul. Implicit dynamics in the material-point method. Computer Methods in Applied
Mechanics and Engineering, 193(12-14):1137–1170, 2004.
[40] Haoyu Wang, Zhuo Huang, Zhiwei Lin, and Tongliang Liu. Noisegpt: Label noise detection and rectification
through probability curvature. In The Thirty-eighth Annual Conference on Neural Information Processing Systems,
2024.
[41] Zhuo Huang, Chao Xue, Bo Han, Jian Yang, and Chen Gong. Universal semi-supervised learning. In Advances in
Neural Information Processing Systems, volume 34, pages 26714–26725, 2021.
[42] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106,
2021.
[43] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering
of neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages
5752–5761, 2021.
[44] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields
with thousands of tiny mlps. In Proceedings of the IEEE/CVF international conference on computer vision, pages
14335–14345, 2021.
[45] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity
neural rendering at 200fps. In Proceedings of the IEEE/CVF international conference on computer vision, pages
14346–14355, 2021.
[46] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a
multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022.
[47] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 5501–5510, 2022.
[48] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for
radiance fields reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pages 5459–5469, 2022.
[49] Scott G Bardenhagen, Edward M Kober, et al. The generalized interpolation material point method. Computer
Modeling in Engineering and Sciences, 5(6):477–496, 2004.
13

<!-- page 14 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
[50] Chenfanfu Jiang, Craig Schroeder, Andrew Selle, Joseph Teran, and Alexey Stomakhin. The affine particle-in-cell
method. ACM Transactions on Graphics (TOG), 34(4):1–10, 2015.
[51] Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana, and Chenfanfu Jiang. A moving
least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM
Transactions on Graphics (TOG), 37(4):1–14, 2018.
[52] Gilles Daviet and Florence Bertails-Descoubes. A semi-implicit material point method for the continuum
simulation of granular materials. ACM Transactions on Graphics (TOG), 35(4):1–13, 2016.
[53] Shyamini Kularathna and Kenichi Soga. Implicit formulation of material point method for analysis of incompress-
ible materials. Computer Methods in Applied Mechanics and Engineering, 313:673–686, 2017.
[54] Zhuo Huang, Miaoxi Zhu, Xiaobo Xia, Li Shen, Jun Yu, Chen Gong, Bo Han, Bo Du, and Tongliang Liu. Robust
generalization against photon-limited corruptions via worst-case sharpness minimization. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16175–16185, 2023.
[55] Zhuo Huang, Li Shen, Jun Yu, Bo Han, and Tongliang Liu. Flatmatch: Bridging labeled data and unlabeled
data with cross-sharpness for semi-supervised learning. In Advances in Neural Information Processing Systems,
volume 36, pages 18474–18494, 2023.
[56] Yuanming Hu, Jiancheng Liu, Andrew Spielberg, Joshua B Tenenbaum, William T Freeman, Jiajun Wu, Daniela
Rus, and Wojciech Matusik. Chainqueen: A real-time differentiable physical simulator for soft robotics. In 2019
International conference on robotics and automation (ICRA), pages 6265–6271. IEEE, 2019.
[57] Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, and Frédo Durand.
Difftaichi: Differentiable programming for physical simulation. arXiv preprint arXiv:1910.00935, 2019.
14

<!-- page 15 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Appendix for “i-PhysGaussian:
Implicit Physical Simulation for 3D Gaussian Splatting”
In the appendix, we first provide extended background on scene reconstruction and Material Point Methods (Section A).
We then summarize the hardware setup and the per-scene time-step schedules used for all methods (Section B), including
the impulse rescaling rule for the one-substep excitation in FICUS. Next, we detail the definitions of our stability
and time-step robustness metrics and report additional results (Sections C–D.3). We further provide reference-free
physical-consistency diagnostics (mass drift and impulse/torque irregularity) with additional qualitative overlays and
per-scene discussions (Sections E.1–E.4). Finally, we include additional rendering-quality diagnostics and ablations on
key components of the Newton–GMRES solver (Sections F–9).
A
Related Work
A.1
Scene Reconstruction
Neural scene representations have emerged as the dominant paradigm for high-fidelity scene reconstruction and novel
view synthesis. NeRF [42] models a continuous 5D radiance field and optimizes a multilayer perceptron (MLP)
with differentiable volume rendering, achieving impressive view consistency and fine geometric detail. However,
NeRF-style representations are typically implicit—encoding the scene within network weights—which often results
in computationally intensive training and rendering. While extensive follow-up research has improved efficiency
via factorization, acceleration structures, or alternative parameterizations [43, 44, 45, 46, 47, 48], the learned scene
content generally remains implicit. This characteristic limits direct inspection, editing, and seamless integration into
downstream modules.
To achieve an explicit and directly renderable representation, recent studies have explored point-based and primitive-
based scene models. 3D Gaussian Splatting (3DGS) [21] represents a scene as a set of anisotropic Gaussians with
low-order spherical harmonics and visibility-aware splatting, enabling real-time rendering alongside high-quality
reconstructions. Crucially, 3DGS provides an explicit, editable representation that can be saved and manipulated
without decoding a neural network. This makes it particularly well-suited as a backbone for physics-based dynamic
simulation, where the reconstructed scene must interface directly with a mechanics solver. This explicitness also
simplifies exporting geometry and state data to physics solvers and facilitates the re-rendering of simulated states, which
is essential for reconstruction-driven dynamics.
A.2
Material Point Methods
The Material Point Method (MPM), introduced by Sulsky et al. [26], is a hybrid particle–grid framework for continuum
mechanics: Lagrangian material points carry the state, while an Eulerian background grid facilitates the updates. This
design avoids mesh entanglement during large deformations and preserves the computational efficiency of regular grids.
Transfers between particles and the grid can introduce interpolation errors. To mitigate these effects, Bardenhagen
and Kober proposed the Generalized Interpolation Material Point (GIMP) method [49], which assigns particles a
finite domain and smooths transfer weights across cell boundaries. Subsequent developments, such as APIC [50] and
MLS-MPM [51], have further improved linear/angular momentum conservation and the stability of P2G/G2P transfers.
For time integration, while explicit schemes remain widely used due to their simplicity, semi-implicit [52, 53] and
fully implicit variants [38, 39, 54, 55] have long been established in traditional simulation. These variants improve
robustness for stiff materials and enable stable roll-outs with larger time steps. Despite these advantages, implicit MPM
remains underexplored in reconstruction-driven dynamic pipelines (e.g., those built on NeRF or 3DGS), where most
methods still rely on explicit stepping for advancing dynamics.
More recently, differentiable MPM [56, 57] has leveraged automatic differentiation to provide end-to-end gradients
for control, parameter identification, and learning-based simulation, further broadening the applicability of MPM in
data-driven settings.
15

<!-- page 16 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Table 3:
Ficus scene:
time-step schedule and particle-impulse boundary condition.
We sweep k
∈
{1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20} by setting substep_dt=k × 10−4 while keeping frame_dt=4.0e−2 and
frame_num=125 fixed. The particle impulse uses force=[fx,0,0], num_dt=1, and start_time=0.0.
TIME
BOUNDARY_CONDITIONS: PARTICLE_IMPULSE
TIME-SIZEsubstep_dtframe_dtframe_num
fx
num_dt
start_time
1 × ∆t
1.0e−4
4.0e−2
125
−1.8e−1
1
0.0
2 × ∆t
2.0e−4
4.0e−2
125
−9.0e−2
1
0.0
4 × ∆t
4.0e−4
4.0e−2
125
−4.5e−2
1
0.0
6 × ∆t
6.0e−4
4.0e−2
125
−3.0e−2
1
0.0
8 × ∆t
8.0e−4
4.0e−2
125
−2.25e−2
1
0.0
10 × ∆t
1.0e−3
4.0e−2
125
−1.8e−2
1
0.0
12 × ∆t
1.2e−3
4.0e−2
125
−1.5e−2
1
0.0
14 × ∆t
1.4e−3
4.0e−2
125
−1.28571e−2
1
0.0
16 × ∆t
1.6e−3
4.0e−2
125
−1.125e−2
1
0.0
18 × ∆t
1.8e−3
4.0e−2
125
−1.0e−2
1
0.0
20 × ∆t
2.0e−3
4.0e−2
125
−9.0e−3
1
0.0
B
Reproducibility and Hardware.
We follow a unified evaluation protocol across methods and keep camera settings and per-scene configurations consistent
whenever applicable. All experiments are conducted on a single NVIDIA RTX 4090 GPU (24GB).
Tables 3–5 summarize the time-step schedules for the FICUS, PILLOW2SOFA, and BREAD scenes (and the im-
pulse boundary condition for FICUS). We sweep the time-step multiplier k ∈{1, 2, 4, . . . , 20} by scaling the sub-
step size as substep_dt(k) = k · substep_dt@1 × ∆t while keeping frame_dt and frame_num fixed. Since
frame_dt/substep_dt(k) is not always an integer, we follow the implementation in our code and set
step_per_frame = round

frame_dt
substep_dt(k)

,
and advance each frame using exactly step_per_frame substeps of size substep_dt(k). Notably, the FICUS scene
includes a particle_impulse boundary condition that acts for exactly one substep. To ensure a consistent total
impulse across different k, we scale the force magnitude proportionally as 1/k.
In addition, Physics3D and DreamPhysics adopt a different numerical convention for Young’s modulus E in their
configuration files: Ecfg = 1.0 corresponds to a physical modulus of Ephys = 107. To align material settings across
methods, we apply the conversion
Ecfg = Ephys
107
(equivalently, Ephys = 107Ecfg),
and report all E values in physical units elsewhere unless stated otherwise. We therefore convert and re-scale E
according to this mapping to align the material settings across methods. For more implementation details and scene-
specific parameters, please refer to the configuration files included in our release.
Table 6 summarizes the material presets used in our experiments and their underlying constitutive formulations, spanning
hyperelasticity (Neo-Hookean), elastoplasticity (Hencky elasticity with J2/von Mises yield), frictional granular behavior
(Drucker–Prager-type plasticity), and rate-dependent viscoplasticity.
16

<!-- page 17 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Table 4: Pillow2Sofa scene: time-step schedule. We sweep k ∈{1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20} by setting
substep_dt=k × 10−4 while keeping frame_dt=2.0e−2 and frame_num=300 fixed.
TIME
TIME-SIZEsubstep_dtframe_dtframe_num
1 × ∆t
1.0e−4
2.0e−2
300
2 × ∆t
2.0e−4
2.0e−2
300
4 × ∆t
4.0e−4
2.0e−2
300
6 × ∆t
6.0e−4
2.0e−2
300
8 × ∆t
8.0e−4
2.0e−2
300
10 × ∆t
1.0e−3
2.0e−2
300
12 × ∆t
1.2e−3
2.0e−2
300
14 × ∆t
1.4e−3
2.0e−2
300
16 × ∆t
1.6e−3
2.0e−2
300
18 × ∆t
1.8e−3
2.0e−2
300
20 × ∆t
2.0e−3
2.0e−2
300
Table 5:
Bread scene:
time-step schedule.
We sweep k
∈
{1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20} by setting
substep_dt=k × 10−4 while keeping frame_dt=1.0e−2 and frame_num=300 fixed.
TIME
TIME-SIZEsubstep_dtframe_dtframe_num
1 × ∆t
1.0e−4
1.0e−2
300
2 × ∆t
2.0e−4
1.0e−2
300
4 × ∆t
4.0e−4
1.0e−2
300
6 × ∆t
6.0e−4
1.0e−2
300
8 × ∆t
8.0e−4
1.0e−2
300
10 × ∆t
1.0e−3
1.0e−2
300
12 × ∆t
1.2e−3
1.0e−2
300
14 × ∆t
1.4e−3
1.0e−2
300
16 × ∆t
1.6e−3
1.0e−2
300
18 × ∆t
1.8e−3
1.0e−2
300
20 × ∆t
2.0e−3
1.0e−2
300
C
Stability Metrics and Additional Results
C.1
Boundary-Hit Mass Fraction (BMF) definition
At frame t, BMF is defined as the fraction of total particle mass belonging to collapsed particles,
BMFt =
P
i∈Ct mi
P
i mi
,
where Ct denotes the set of particles that were clamped during the step that produces frame t.2
C.2
Additional Heatmap Details
Fig. 6 further offers a fine-grained characterization of stability via a per-setting heatmap. For each (scene, method, k),
the cell color encodes the exceedance ratio
r = 1
T
T
X
t=1
I[BMFt > 0.5] ,
i.e. the fraction of frames in which BMF exceeds the threshold (lower is more stable), and we overlay “×” to mark
settings that are deemed failed under the unified BMF-gate rule. Consistent with the bar-chart summary, i-PhysGaussian
maintains low r and never triggers failure across all scenes and multipliers. PhysGaussian enters the high-r regime
earlier on Bread as k increases, whereas DreamPhysics and Physics3D show earlier and broader regions of high r on
Pillow2Sofa and Bread, which explains their higher failure rates and smaller kmax.
2Concretely, a particle is included in Ct if any coordinate is clipped to ϵ or grid_lim −ϵ by the clamp rule.
17

<!-- page 18 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Table 6: Material scope and constitutive models.
MATERIAL
TYPE
MODEL
jelly
ELASTIC
NEO-HOOKEAN
metal
PLASTIC
J2 (VON MISES)
foam
DAMAGE
J2 + SOFTENING
snow/sand
FRICTIONAL
DRUCKER–PRAGER
plasticine
VISCOPLASTIC
RATE-DEPENDENT J2
Figure 6: Stability heatmaps across time-step multipliers. Color indicates exceedance ratio r = 1
T
PT
t=1 I[BMFt >
0.5] (lower is better); × marks runs that fail the gate (r > 0.5).
D
Time-step Robustness Metrics and Additional Results
D.1
Center-of-mass drift (COMD) definition
Let x(k)
t,i ∈R3 be the position of particle i at frame t under the k × ∆t setting, with mass mi. Let x(1)
t,i denote the
1 × ∆t reference. We first compute the mass-weighted center of mass at each frame:
c(k)
t
=
P
i mix(k)
t,i
P
i mi
.
(23)
We then define COMD as the time-averaged normalized displacement to the reference:
COMD(k) = 1
T
T
X
t=1
1
grid_lim
c(k)
t
−c(1)
t

2 .
(24)
18

<!-- page 19 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Table 7: Normalized AUC of drift curves (lower is better), computed over k ∈{1, 2, 4, . . . , 20} on BMF-valid runs.
METHOD
COMD AUC↓
MWRMSD AUC↓
FICUS
i-PhysGaussian
0.0184
0.0279
PHYSGAUSSIAN
0.5975
0.8509
DREAMPHYSICS
0.0556
0.8539
PHYSICS3D
0.0000
0.8539
PILLOW2SOFA
i-PhysGaussian
0.0023
0.0382
PHYSGAUSSIAN
0.0139
0.0406
DREAMPHYSICS
0.4484
0.4756
PHYSICS3D
0.8169
0.8867
BREAD
i-PhysGaussian
0.0138
0.2126
PHYSGAUSSIAN
0.3394
0.5202
DREAMPHYSICS
0.8359
0.9737
PHYSICS3D
0.0000
0.9737
D.2
Mass-weighted RMS Drift (mwRMSD) with clamping penalty
To ensure hard clamping does not artificially suppress divergence (by projecting out-of-domain motion back to the
boundary), we impose a conservative penalty on particles that are clamped in either trajectory. Let C(k)
t
and C(1)
t
denote
the sets of particles that are clamped during the simulation step producing frame t under k× and 1×, respectively, and
define
Ct = C(k)
t
∪C(1)
t
.
(25)
We compute the per-particle deviation:
d(k)
t,i =
x(k)
t,i −x(1)
t,i

2 ,
(26)
and define a penalized deviation with an upper bound Dmax = grid_lim:
ˆd(k)
t,i =
(Dmax,
i ∈Ct,
min

d(k)
t,i , Dmax

,
otherwise.
(27)
Finally, mwRMSD is defined as the time-averaged normalized mass-weighted RMS deviation:
mwRMSD(k) = 1
T
T
X
t=1
1
grid_lim
v
u
u
t
P
i mi

ˆd(k)
t,i
2
P
i mi
.
(28)
D.3
Additional AUC details
To summarize robustness across multipliers with a single number, we report the AUC, which is computed over all tested
multipliers k ∈1, 2, 4, . . . , 20, and should be interpreted together with the BMF gate since clamp-dominated regimes
can distort drift magnitudes. Lower AUC indicates smaller overall drift under time-step enlargement.
Overall, i-PhysGaussian achieves the lowest (or near-lowest) AUC across scenes and metrics, consistent with the
drift curves in Fig. 3. PhysGaussian is close to i-PhysGaussian on Pillow2Sofa but degrades on Bread, whereas
DreamPhysics and Physics3D incur substantially larger AUC on the harder scenes. We interpret COMD and mwRMSD
jointly: COMD captures global motion bias while mwRMSD reflects particle-level deviation and deformation details.
19

<!-- page 20 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
E
Physical Simulation Fidelity Metrics and Additional Results
E.1
Mass drift definition
Let xt,i ∈R3 denote the position of particle i at frame t with mass mi under the 1 × ∆t setting, for t = 1, . . . , T. We
define the total mass
Mt =
X
i
mi,
(29)
and report the per-frame mass drift relative to the first frame:
MassDriftt = |Mt −M1|
M1 + ε
,
t = 1, . . . , T,
(30)
where ε is a small constant for numerical stability.
E.2
Momentum and irregularity
We estimate particle velocities from trajectories via finite differences:
vt,i =







xt+1,i−xt,i
∆t
,
t = 1,
xt+1,i−xt−1,i
2∆t
,
1 < t < T,
xt,i−xt−1,i
∆t
,
t = T,
(31)
and compute the total linear momentum
Pt =
X
i
mi vt,i.
(32)
We also compute the instantaneous center of mass
ct =
P
i mixt,i
P
i mi
,
(33)
and define the total angular momentum about ct:
Lt =
X
i
(xt,i −ct) × (mi vt,i) .
(34)
We measure the second-order temporal difference of the total momenta:
∆2Pt = Pt −2Pt−1 + Pt−2,
t = 3, . . . , T,
(35)
∆2Lt = Lt −2Lt−1 + Lt−2,
t = 3, . . . , T.
(36)
We normalize them using fixed scales to avoid division instability in near-static phases:
ImpulseIrrt =
∆2Pt

2
Pscale + ε,
TorqueIrrt =
∆2Lt

2
Lscale + ε.
(37)
Here we use the domain-scale, time-step-aware normalization
Pscale = Mtotal
grid_lim
∆t
,
Lscale = Mtotal
grid_lim2
∆t
,
(38)
where Mtotal = P
i mi is the total mass.
Lower values indicate smoother (less jittery) temporal evolution of net impulse/torque, while sharp spikes suggest
bursty non-physical impulses or numerical artifacts (e.g., due to hard clamping projection or unstable contact handling).
E.3
Mass conservation sanity check
E.4
Additional per-scene discussion
Ficus.
Impulse/torque irregularities of i-PhysGaussian and PhysGaussian show periodic, wave-like patterns consistent
with the oscillatory swaying motion driven by external excitation. Some baselines yield nearly flat curves close to
zero; together with the trajectory overlays in Appendix Fig. 8 suggest kinematic degeneration toward an almost static
trajectory.
20

<!-- page 21 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
(a) Ficus
(b) Pillow2Sofa
(c) Bread
Figure 7: Reference-free mass conservation at 1 × ∆t (per-frame). We plot MassDriftt = |Mt−M1|
M1+ε
over frames for
each scene (lower is better).
Pillow2Sofa.
All methods show well-aligned peaks in impulse/torque irregularity, corresponding to key con-
tact/collision moments between the pillow and the sofa. The trajectory overlays indicate broadly plausible collision
dynamics across methods in this scene.
Bread.
Overall, i-PhysGaussian exhibits smoother curves with fewer bursty spikes. PhysGaussian shows more abrupt
spikes, consistent with rapid structural/contact-state changes. Some baselines again exhibit near-zero irregularities,
consistent with degeneration-to-static behavior as indicated by the trajectory overlays.
21

<!-- page 22 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Ficus
Pillow2Sofa
Bread
i-PhysGaussian
PhysGaussian
DreamPhysics
Physics3D
Figure 8: Trajectory overlays at a key simulation moment. Near-flat irregularity curves can correspond to kinematic
degeneration (suppressed motion), as visualized here.
F
Dynamic Rendering Quality Metrics and Additional Results
Per-frame Saturation Ratio Definition
For a rendered RGB frame at time t, It ∈[0, 1]H×W ×3, we compute
luminance
Yt(p) = 0.2126 Rt(p) + 0.7152 Gt(p) + 0.0722 Bt(p),
(39)
22

<!-- page 23 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
where p indexes pixels. Given a saturation threshold τ = 0.98, the per-frame saturation ratio is
SatRatiot =
1
HW
X
p
I[ Yt(p) ≥τ ] .
(40)
We summarize the series {SatRatiot}T
t=1 by its mean, standard deviation, and range (max–min):

µ(SatRatio), σ(SatRatio), rng(SatRatio)

.
(41)
Implementation Notes.
We compute SatRatio directly on the rendered frames produced by each pipeline at 1 × ∆t.
All frames are assumed to be normalized to [0, 1] per channel before luminance computation. We use a fixed threshold
τ = 0.98 to capture near-saturated (blown-out) pixels in a consistent manner across methods and scenes.
Additional Discussion.
The mean Sat.% reflects the overall extent of over-exposed regions, while std/range describe
the temporal variability of the near-saturated fraction. When the mean level is already low, a relatively larger std/range
typically corresponds to local threshold crossings around τ (e.g., thin structures, specular highlights, or high-contrast
boundaries during motion) rather than large-area exposure failures. Conversely, a “high-mean, near-zero-variance”
pattern indicates persistent saturation and is characteristic of saturation-dominated degenerate outputs.
23

<!-- page 24 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Table 8: Line-search ablation on the i-PhysGaussian implicit MPM solver. Success is the frame-level success rate (a
frame is successful only if all substeps converge). Speedup is computed as P tbase
Newton/ P tnoLS
Newton. RelEnd is the mean
Newton residual ratio Rend/R0 averaged over converged substeps.
SCENE ∆t
SUCCESS↑
BASE/NOLS
∆PP
SPEEDUP↑
RELEND↓
BASE
NOLS
PILLOW2SOFA
1X 63.3 / 62.6
−0.7
1.02×
6.28E-4
6.28E-4
10X 71.2 / 72.4
+1.2
1.00×
3.09E-4
2.94E-4
20X 66.4 / 62.6
−3.8
0.99×
6.20E-4
5.53E-4
BREAD
1X
0.3 / 0.0
−0.3
1.74×
1.57E-5
1.59E-5
10X 99.3 / 0.0
−99.3
0.99×
4.19E-5
5.72E-5
20X 99.3 / 0.0
−99.3
1.50×
5.24E-5
–
G
Ablations
In this subsection, we conduct an ablation study on the implicit MPM solver of i-PhysGaussian, focusing on how
key numerical strategies within the Newton–GMRES framework affect the trade-off between stability and efficiency.
Specifically, we ablate (1) the backtracking line search used in the Newton updates, and (2) the Eisenstat–Walker (EW)
forcing term that adaptively controls the accuracy of the inner GMRES solves (i.e., tolerance scheduling).
Evaluation protocol and implementation note.
We report three solver-trace-derived diagnostics: stability (Success),
efficiency (Speedup), and convergence quality (RelEnd), as listed in the corresponding tables. A frame is deemed
successful only if all substeps within that frame converge. In our implementation, the number of substeps per frame is
an integer
step_per_frame = round
 frame_dt
substep_dt

,
(42)
so the effective simulated time interval per frame is step_per_frame · substep_dt (which may differ slightly from
frame_dt when the ratio is not an integer). We apply the same convention consistently across all ablated variants, and
all frame-level metrics are computed at the exported frame indices.
G.1
Ablation 1: W/O Line Search
The backtracking line search serves as a step-size controller for the Newton update. Given the search direction returned
by the inner GMRES solve, we first try a full step (α = 1) and check whether it decreases the Newton residual (or the
corresponding objective). If the residual fails to decrease, we backtrack and shrink α ∈(0, 1) until a sufficient decrease
condition is met. Disabling line search removes these extra evaluations and can speed up the loop in some cases, but at
the cost of reduced robustness, especially on strongly non-linear or stiff scenes where full Newton steps are more likely
to overshoot.
Tab. 8 reports stability (Success), efficiency (Speedup), and convergence quality (RelEnd) under the line-search
ablation. The results support the intended role of line search: when Newton–GMRES directions are already well-
behaved, line search is rarely needed; when the problem becomes strongly non-linear, it can be critical for robustness.
On pillow2sofa, removing line search has only a minor impact across time steps: Success changes by at most a few
percentage points (−0.7pp at 1×, +1.2pp at 10×, and −3.8pp at 20×), while Speedup stays near 1.0× (0.99–1.02×).
Consistently, RelEnd remains in the same range, indicating that the Newton updates are typically “safe” on this scene
and the line search mainly acts as a safeguard rather than a frequently-triggered stabilizer.
In contrast, the bread scene amplifies the importance of line search. While the base solver achieves near-perfect
success at 10× and 20× (both 99.3%), noLS collapses to 0.0% success under the same settings, indicating that full
Newton steps frequently overshoot and lead to divergence without step-size control. The “–” RelEnd entry at 20× under
noLS further corroborates this: there are essentially no converged substeps from which Rend/R0 can be aggregated.
24

<!-- page 25 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
Table 9: Fixed forcing-term ablation for the i-PhysGaussian implicit MPM solver (base: Eisenstat–Walker adaptive
forcing term; fixed: constant GMRES tolerance shared across scenes). Success is the frame-level success rate (a frame
is successful only if all substeps converge). Speedup is computed as P tbase
Newton/ P tfixed
Newton (larger is faster for fixed).
GMRES iters reports mean/max GMRES iterations aggregated over converged Newton solves. RelEnd is the mean
Newton residual ratio Rend/R0 averaged over converged substeps.
SCENE ∆t
SUCCESS↑
BASE/FIXED
SPEEDUP↑
GMRES ITERS
(MEAN/MAX)
BASE/FIXED
RELEND↓
BASE/FIXED
PILLOW2SOFA
1X 63.3 / 63.3 1.22×
7.0 / 18
4.0 / 22
1.53E-6
1.55E-6
10X 70.3 / 70.3 0.99×
4.1 / 18
4.1 / 18
2.66E-5
1.92E-5
20X 71.0 / 71.0 1.01×
6.1 / 19
5.3 / 32
8.49E-5
6.58E-5
BREAD
1X
0.3 / 0.3
0.99×
11.0 / 21
14.3 / 46
6.63E-6
6.65E-6
10X 99.3 / 99.3 1.00×
14.5 / 39
20.0 / 54
4.21E-5
4.22E-5
20X 99.3 / 99.3 0.89×
28.3 / 57
44.3 / 84
5.26E-5
5.24E-5
(We note that Speedup becomes less informative when Success collapses, since failing runs may terminate early and
distort aggregated runtimes.)
G.2
Ablation 2: Fixed forcing term (w/o Eisenstat–Walker)
Fixed-forcing-term Newton–GMRES uses a manually chosen tolerance to decide when the inner GMRES loop
terminates. Intuitively, when the Newton residual is large, the GMRES solve does not need to be very accurate—an
approximate solution can still provide a useful descent direction. As the iteration approaches convergence, however,
GMRES typically must solve more accurately to prevent outer progress from stalling. This stage-dependent accuracy
demand motivates the Eisenstat–Walker (EW) strategy, which adapts the GMRES stopping criterion based on observed
outer residual reduction, automatically balancing efficiency and robustness.
In contrast, enforcing a fixed forcing term simplifies the solver but generally requires scene- and step-dependent tuning:
a tolerance that is “good” for one setting may become too loose (under-solving) or too strict (over-solving) elsewhere.
Tab. 9 compares base (EW adaptive forcing) against fixed forcing in terms of stability (Success), efficiency (Speedup),
and inner-solve workload/convergence quality (GMRES iters and RelEnd). Overall, the two variants do not exhibit a
clear “converge vs. diverge” split on the reported scenes, since the frame-level success rates remain identical across
all settings. However, the fixed strategy shows noticeably higher sensitivity in computational behavior: depending
on the scene and time step, it may either reduce typical GMRES work or introduce heavier tails (larger worst-case
iterations), which is precisely the regime EW aims to handle by adapting inner accuracy to outer progress.
On pillow2sofa, Success is unchanged between base and fixed at all time steps (63.3% at 1×, 70.3% at 10×, and
71.0% at 20×). In terms of efficiency, fixed forcing achieves a clear gain at 1 × ∆t with Speedup = 1.22×. This
aligns with reduced average GMRES effort: the mean GMRES iterations drop from 7.0 (base) to 4.0 (fixed), while
RelEnd remains essentially the same (1.53e-6 vs. 1.55e-6). The benefit is not uniform across time steps: at 10× and
20×, Speedup becomes marginal (0.99× and 1.01×), and the iteration profile becomes more mixed. Notably at 20×,
while the mean GMRES iters slightly decreases (6.1 →5.3), the max increases substantially (19 →32), indicating that
a single fixed tolerance can reduce typical workload but may also yield heavier-tail inner solves in more difficult phases.
Across all pillow2sofa settings, RelEnd stays within the same order of magnitude, suggesting that these shifts mainly
reflect tolerance sensitivity rather than a systematic change in outer convergence quality.
On the more challenging bread scene, the sensitivity of fixed forcing becomes more pronounced in compute cost.
Again, Success remains identical between base and fixed (0.3% at 1×, 99.3% at 10× and 20×), and RelEnd is nearly
unchanged, indicating comparable convergence quality on converged substeps. However, the GMRES workload
increases consistently under fixed forcing, especially at larger time steps: at 1×, the mean/max GMRES iters increase
from 11.0/21 to 14.3/46; at 10×, from 14.5/39 to 20.0/54; and at 20×, from 28.3/57 to 44.3/84. Correspondingly,
efficiency degrades at 20× with Speedup = 0.89× (i.e., a slowdown). Since Success and RelEnd remain essentially
25

<!-- page 26 -->
i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting
A PREPRINT
unchanged, this pattern is most consistent with over-solving: fixed forcing spends substantially more GMRES iterations
to meet a constant inner residual target, yet this extra inner accuracy does not translate into a meaningfully better outer
convergence outcome. Taken together, these results motivate our choice of Eisenstat–Walker: it reduces per-scene
manual tuning and helps avoid both under-solving and over-solving when rolling out difficult scenes under large time
steps.
26
