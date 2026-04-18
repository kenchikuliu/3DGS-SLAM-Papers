<!-- page 1 -->
3DGS2-TR: Scalable Second-Order Trust-Region Method
for 3D Gaussian Splatting
Roger Hsiao 1 Yuchen Fang 1 Xiangru Huang 1 Ruilong Li 2 Hesam Rabeti 2 Zan Gojcic 2 Javad Lavaei 1
James Demmel 1 Sophia Shao 1
Abstract
We propose 3DGS2-TR, a second-order optimizer
for accelerating the scene training problem in
3D Gaussian Splatting (3DGS). Unlike existing
second-order approaches that rely on explicit or
dense curvature representations, such as 3DGS-
LM (H¨ollein et al., 2025) or 3DGS2(Lan et al.,
2025), our method approximates curvature us-
ing only the diagonal of the Hessian matrix, esti-
mated efficiently via Hutchinson’s method. Our
approach is fully matrix-free and has the same
complexity as ADAM (Kingma, 2014), O(n) in
both computation and memory costs. To ensure
stable optimization in the presence of strong non-
linearity in the 3DGS rasterization process, we
introduce a parameter-wise trust-region technique
based on the squared Hellinger distance, regu-
larizing updates to Gaussian parameters. Under
identical parameter initialization and without den-
sification, 3DGS2-TR is able to achieve better
reconstruction quality on standard datasets, us-
ing 50% fewer training iterations compared to
ADAM, while incurring less than 1GB of peak
GPU memory overhead (17% more than ADAM
and 85% less than 3DGS-LM), enabling scala-
bility to very large scenes and potentially to dis-
tributed training settings.
1. Introduction
Recent advances in radiance field representations have revo-
lutionized 3D content creation, enabling high-fidelity, pho-
torealistic scene reconstruction from sparse input views.
Neural Radiance Fields (NeRF) (Mildenhall et al., 2021)
pioneered coordinate-based neural scene representations,
achieving remarkable rendering quality but suffering from
slow training and inference times. To address these limita-
1University of California, Berkeley 2NVIDIA. Correspondence
to: Roger Hsiao <roger hsiao@berkeley.edu>.
Preprint. February 3, 2026.
Figure 1. Overview of our proposed method.
tions, 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023)
proposed an explicit and efficient scene representation that
models geometry and appearance using anisotropic 3D
Gaussians optimized through differentiable rasterization.
This formulation preserves view-dependent effects while
enabling real-time rendering, rapidly establishing 3DGS as
the new standard for real-time novel view synthesis—the
task of rendering photorealistic images from previously un-
seen camera viewpoints given a limited set of input images.
This capability is central to applications in virtual reality,
augmented reality, robotics, and 3D content creation.
Building on the original 3DGS framework’s demonstra-
tion that anisotropic Gaussians coupled with GPU-friendly
1
arXiv:2602.00395v1  [cs.CV]  30 Jan 2026

<!-- page 2 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
rasterization can achieve both high-quality synthesis and
real-time performance (Kerbl et al., 2023), a number of sub-
sequent works have proposed strategies to further improve
rasterization efficiency. Recent efforts focus on accelerating
forward rasterization and inference through optimized GPU
kernels, memory access patterns, and tile-based rendering
strategies, enabling efficient large-scale deployment and
real-time visualization (Durvasula et al., 2023; Feng et al.,
2025; Mallick et al., 2024; Ye et al., 2025; Papantonakis
et al., 2024; Zhao et al., 2024).
Despite these advances in rasterization and scene represen-
tation, scene training in 3DGS remains a bottleneck in the
pipeline. Although significantly faster than NeRF, 3DGS
still requires training from scratch for each new scene, tak-
ing 20-40 minutes on commercial GPUs for commonly used
datasets (Kerbl et al., 2023). This limits the applicability of
3DGS in large-scale or real-time exploration scenarios.
Several lines of research address this training bottleneck
through orthogonal approaches. One direction targets the
backward pass, which accumulates gradients into Gaussian
splats with poor GPU utilization due to resource contention
when many pixels share the same Gaussian primitives. Re-
cent work has identified this as a dominant bottleneck and
proposed alternative accumulation schemes or kernel re-
designs to mitigate atomic contention (Durvasula et al.,
2023; Ye et al., 2025). Another approach reduces train-
ing time through more compact scene representations. Re-
lated efforts improve the densification process or introduce
pruning strategies to reduce the total number of Gaussian
primitives while preserving reconstruction quality, thereby
indirectly accelerating training speed (Kheradmand et al.,
2024; Fang & Wang, 2024; Rota Bul`o et al., 2024; Ali et al.,
2024; Zhang et al., 2024b; Hanson et al., 2025b).
Our work aligns most closely with a third direction: improv-
ing the optimizer itself. The standard optimizer for 3DGS
is ADAM (Kingma, 2014), a first-order gradient-descent
method widely used in deep learning. While first-order
optimizers are easy to implement and scale well, they suf-
fer from slow convergence in highly non-convex and ill-
conditioned parameter spaces (Sutton, 1986; Dauphin et al.,
2014; Bottou et al., 2018). The optimization landscape of
3DGS is particularly challenging due to the strong coupling
between geometry parameters (position, rotation, scale) and
appearance parameters (opacity, color), leading to inefficient
convergence and excessive training iterations. Additionally,
first-order methods require meticulous tuning of learning
rates for each parameter (Sutton, 1986; Schaul et al., 2013;
Dauphin et al., 2014; Bottou et al., 2018).
3DGS occupies a unique position in machine learning: the
model quality scales with the number of parameters, yet
each parameter remains highly interpretable as a compo-
nent of an unnormalized 3D Gaussian distribution. This has
motivated prior work to explore second-order optimization
algorithms—such as Newton’s method, Gauss-Newton, or
Levenberg-Marquardt—to achieve superlinear convergence
(H¨ollein et al., 2025; Lan et al., 2025; Pehlivan et al., 2025).
However, the 3DGS rasterization function poses fundamen-
tal challenges for second-order methods. First, it is highly
nonlinear due to the sequential rendering of Gaussian splats
at each pixel, where the transmittance seen by each splat
depends on the opacities of all preceding splats in depth or-
der. Second, the loss function is only piecewise continuous
since depth-based sorting of Gaussian splats causes render
order to change discontinuously with position parameters.
These discontinuities become more frequent in regions with
dense clusters of splats, which commonly occur towards
the end of training. Moreover, existing implementations of
second-order methods incur high memory overhead from
storing curvature information and suffer from expensive
per-iteration costs due to matrix operations, offsetting any
convergence speedup and making them unsuitable for pro-
duction use with large scenes.
In practice, second-order optimizers for 3DGS have yet
to outperform first-order methods in either reconstruction
quality or convergence speed, leading to limited adoption.
However, recent work in deep learning, such as Sophia
(Liu et al., 2023), introduces a lightweight, second-order
approach that utilizes a diagonal Hessian estimate to ac-
count for loss surface curvature. By adapting step sizes to
sharp or flat regions with minimal computational overhead,
Sophia achieves faster convergence and superior stability in
large-scale non-convex tasks such as large language model
training. Nevertheless, the performance of Sophia and its
variants remains largely unexplored in computer vision ap-
plications.
Based on these observations, we propose the following three
principles to accelerate scene training in 3DGS.
1. Cheap per-iteration computation. Since 3DGS rasteri-
zation is highly nonlinear with frequent discontinuities,
we favor many small, inexpensive steps over a few
large, expensive ones that would likely violate local
linear or quadratic approximations. Given that the
backward pass is the primary training bottleneck, our
strategy introduces modest computational overhead in
pre- or post-processing the update step to accelerate
convergence and reduce the total number of backward
passes required.
2. Parameter-linear memory scaling. Persistent storage
across iterations should scale with the number of Gaus-
sian parameters, not with the number of pixels, which
often exceed the parameter count by orders of mag-
nitude for large scenes with high-resolution training
images.
2

<!-- page 3 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
3. Trust region constraints. Update step sizes should be
bounded by a trust region that limits the impact of
nonlinear interactions between Gaussian splats. Specif-
ically, we bound the change in transmittance experi-
enced by each splat before and after the update, ensur-
ing steps remain within the region of validity for local
approximation.
Guided by these principles, we present 3DGS2-TR, a
second-order optimizer for accelerated 3DGS scene training.
We summarize our main contributions as follows:
1. We apply a second-order update rule that uses Hes-
sian diagonals to 3DGS training, which has the same
O(n) complexity as ADAM in both computation and
memory.
2. We propose an effective trust region for Gaussian pa-
rameter updates that bounds the squared Hellinger
distance of each Gaussian splat before and after op-
timization steps, providing a principled constraint on
geometric changes and requiring only a single tunable
hyperparameter.
2. Related Work
2.1. Novel View Synthesis
Novel view synthesis aims to generate photorealistic im-
ages from arbitrary viewpoints given a set of input images.
Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) pi-
oneered implicit scene representations using multilayer per-
ceptrons to encode volumetric density and view-dependent
appearance, achieving photorealistic results through vol-
umetric ray marching but suffering from slow training
and inference. In contrast, 3D Gaussian Splatting (3DGS)
(Kerbl et al., 2023) models scenes explicitly as collections
of anisotropic 3D Gaussians, enabling real-time rendering
through efficient differentiable rasterization while maintain-
ing high visual fidelity. This has established 3DGS as the
state-of-the-art for real-time novel view synthesis, inspiring
extensive research on improving rendering quality (Lu et al.,
2024; Yu et al., 2024), scaling to larger environments (Kerbl
et al., 2024; Song et al., 2024), and enhancing geometric
accuracy.
2.2. Accelerating 3DGS Training and Rendering
While 3DGS achieves real-time rendering, training remains
a bottleneck for large-scale scenes. Recent work addresses
this through complementary strategies.
Compact representations. Several methods aim to reduce
the number of Gaussian primitives while preserving quality.
EAGLES (Girish et al., 2024) uses quantized embeddings
and coarse-to-fine training. LightGaussian (Fan et al., 2024)
prunes low-contribution Gaussians and compresses spher-
ical harmonics. C3DGS (Lee et al., 2024) learns binary
masks to remove redundant primitives, while Speedy-Splat
(Hanson et al., 2025a) introduces tight bounding boxes and
dual pruning strategies.
Rasterization optimizations. AdR-Gaussian (Wang et al.,
2024) culls low-opacity Gaussian-tile pairs and balances
workloads. FlashGS (Feng et al., 2025) and gsplat (Ye et al.,
2025) optimize CUDA kernels and memory access patterns.
Additional improvements include modified densification
heuristics (Fang & Wang, 2024; Kheradmand et al., 2024)
that reduce Gaussian proliferation.
2.3. Second-Order Optimizers
Recent research in large-scale machine learning explores re-
placing ADAM (Kingma, 2014) with second-order methods
to accelerate convergence. Classical second-order meth-
ods—Newton’s method, Gauss-Newton, and Levenberg-
Marquardt—achieve superlinear convergence via Hessian
curvature information but face O(n2) memory and O(n3)
computational costs. Lightweight alternatives such as Ada-
Hessian (Yao et al., 2021) and Sophia (Liu et al., 2023),
instead use diagonal Hessian approximations for efficient
large-scale optimization, avoiding the storage and computa-
tion overhead while gaining improvement in performance.
Applications of second-order optimizers have been explored
in 3DGS as well. For example, 3DGS-LM (H¨ollein et al.,
2025) integrates the Levenberg-Marquardt method, while
3DGS2 (Lan et al., 2025) partitions parameters and solves
small Newton systems in sequence. Recently, (Pehlivan
et al., 2025) adopted a matrix-free design, applying the pre-
conditioned conjugate gradient method to solve the system,
and introduced a pixel sampling strategy based on residuals.
However, some fundamental challenges have not been ade-
quately addressed: 3DGS rasterization is highly nonlinear
and only piecewise continuous due to depth-based sorting,
violating smoothness assumptions of classical second-order
methods. Moreover, the high parameter and pixel counts
make it intractable to materialize full matrices, necessitating
careful camera perspective grouping and image subsampling
(H¨ollein et al., 2025; Lan et al., 2025). Consequently, exist-
ing approaches only work well on small scenes or when ini-
tialized near a sufficiently good local minimum. In contrast,
our work adapts Sophia’s lightweight framework to 3DGS
with modifications tailored to Gaussian-based representa-
tions while maintaining efficiency and broad applicability.
3

<!-- page 4 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
3. Background and Notations
3.1. Review of Gaussian Splatting
3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) repre-
sents a scene as a collection of 3D Gaussians splats, each
parametrized by its position µ, scale S, rotation R, opacity
α, and color C. We can write each Gaussian as an unnormal-
ized Gaussian distribution
G(z) = αC exp(−1
2(z −µ)T Σ−1(z −µ))
(1)
= αC(2π)
3
2 det(Σ)
1
2 N(z; µ, Σ)
(2)
The covariance is split into separate rotation and scaling
parameters as Σ = RT ST SR. C is the view-dependent
color modeled using spherical harmonic coefficients of order
3. To render an image of size H×W from a given viewpoint,
all Gaussians are first projected into 2D Gaussian splats via
a tile-based differentiable rasterizer. The projected splats are
then α-blended along each camera ray to obtain the color
Cω at pixel ω:
Cω =
X
i∈K
Ci¯αiTi, with Ti =
i−1
Y
j=1
(1 −¯αj),
where K denotes the set of Gaussian kernels of size |K|,
Ci is the color of the i-th splat along the ray, ¯αi is the
2D Gaussian’s evaluated opacity, and Ti represents trans-
mittance. The complete Gaussian parameter vector x con-
catenates the parameters of all |K| kernels by groups, (e.g.
x = (xposition, xscaling, xrotation, xopacity, xcolor). To fit
the Gaussian parameter x, 3DGS minimizes the discrepancy
between the rendered and ground-truth images. At each
pixel ω, the loss function is written as
Lω = (1 −λ)Lω,L1 + λLω,D−SSIM.
(3)
The L1 and SSIM losses can each be further split into three
components, one for each RGB channel. The total loss is
the mean of the loss over all pixels and all channels.
3.2. Optimization Problem Formulation
Let x ∈Rn be the parameter vector by concatenating the
parameters of all Gaussian kernels. Let M be the number
of training images. Each training image emits 6 × H × W
loss components with a total of m loss components over
all images. We use fi(x) to denote the vector loss function
for image i, where each entry is the square root of a loss
component, so that the 3DGS scene training problem can be
formulated as a nonlinear least-squares problem
min
x
f(x) :=
1
2m


f1(x)T
· · ·
fM(x)T T 
2
2 .
(4)
The full Jacobian matrix is denoted as J(x)
=

J1(x)T
· · ·
JM(x)T T and the pseudo-Hessian matrix
is H(x) = J(x)T J(x) = PM
i=1 Ji(x)T Ji(x).
Algorithm 1 3DGS2-TR
Require: Initial parameters x1 ∈Rn, Hessian diagonals
update interval l ∈N, EMA decay rates θ1, θ2 ∈(0, 1),
trust-region parameter ϵ > 0, Hutchinson sample size
ν ∈N.
1: Set bg0 = 0, bD1−l = 0.
2: for t = 1 to T do
3:
Sample a mini-batch S1 of images and compute the
stochastic gradient gt.
4:
Update EMA bgk = θ1bgk−1 + (1 −θ1)gk.
5:
if t mod l = 1 then
6:
Sample a mini-batch S2 of images and compute
Dt = Hutch(xt, S2, ν).
7:
Update EMA bDt = θ2 bDt−l + (1 −θ2)Dt.
8:
else
9:
bDt = bDt−1.
10:
end if
11:
Compute the update step ∆xt = −bD−1
t
ˆgt.
12:
Compute the parameter-wise trust-region radius
η = SHD(xt, ϵ).
13:
Apply parameter-wise clipping
b∆xt = ∆xt.clip(−η, +η).
14:
Update the parameters xt+1 = xt + b∆xt.
15: end for
4. Methodology
3DGS2-TR is a scalable second-order optimization frame-
work for solving the 3DGS training objective (4). Our
method adopts a stochastic Gauss–Newton formulation
and incorporates curvature information while remaining
memory- and compute-efficient. An overview of the pro-
posed pipeline is illustrated in Figure 1, whereas the full
algorithm is summarized in Algorithm 1.
4.1. Algorithm Design
We initialize the 3D Gaussian primitives using a sparse point
cloud obtained from structure-from-motion (SfM). Given
the parameters xt at iteration t, we first subsample a mini-
batch of images S1 to estimate the stochastic gradient
gt = 1
m
M
|S1|
X
i∈S1
Ji(xt)T fi(xt),
where fi(xt) denotes the residual vector and Ji(xt) is its
Jacobian of the i-th image with respect to the Gaussian pa-
rameters xt. To reduce gradient noise and improve stability,
we maintain an exponential moving average (EMA) of the
4

<!-- page 5 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
gradients with decay rate θ1 ∈(0, 1),
bgt = θ1bgt−1 + (1 −θ1)gt.
To incorporate second-order information while maintaining
O(n) memory and computation costs, we follow the spirit
of Sophia (Liu et al., 2023) and estimate the diagonal of
the Gauss–Newton matrix JT J using Hutchinson’s method.
Specifically, every l iterations, we subsample another mini-
batch of images S2 and compute
Dt = Hutch(xt, S2, ν)
= 1
m
M
|S2|
1
ν
ν
X
i=1
z(i) ⊙

X
j∈S2
Jj(xt)T Jj(xt)z(i)

,
where {z(i)}m
i=1 are independent Rademacher random vec-
tors and ⊙denotes element-wise multiplication. The scaling
factors M|S1|−1, M|S2|−1 are required to maintain an unbi-
ased estimate of the gradient and the Hessian diagonal after
sampling. Such image subsampling and diagonal approxi-
mation significantly reduce computational and memory over-
head, since the Hessian-vector product Jj(xt)T Jj(xt)z(i)
corresponds to exactly one forward and backward pass on a
training image. To further stabilize curvature estimates, we
maintain an EMA of Dt with decay rate θ2 and reuse the
previous estimate in intermediate iterations, i.e.,
bDt =
(
θ2 bDt−1 + (1 −θ2)Dt if t
mod l = 1,
bDt−1
otherwise.
Given bDt and bgt, we compute the update step
∆xt = −bD−1
t
bgt.
To improve robustness under strong nonlinearity and dis-
continuities in the 3DGS rasterization process, we apply
a parameter-wise trust-region constraint. Specifically, we
compute a trust-region radius for each parameter
η = SHD(xt, ϵ),
based on the squared Hellinger distance, which bounds the
change of each Gaussian primitive induced by the update.
Details of this construction are provided in Section 4.2.
The final update is obtained by parameter-wise clipping,
b∆xt = ∆xt.clip(−η, +η),
followed by the parameter update
xt+1 = xt + b∆xt.
(5)
The minibatch sizes are selected based on the tradeoff be-
tween algorithmic performance and computational cost. In
our experiments, we set |S1| = |S2| = ν = 1 and l = 10,
which corresponds to an approximately 10% overhead com-
pared to ADAM.
Figure 2. Example scene with a single elongated Gaussian splat.
The x-axis and y-axis lie on the page; the z-axis comes out of the
page. The green arrows mark the directions in which the Gaussian
has more freedom; while the red arrows denote otherwise.
4.2. Trust-Region Based on Squared Hellinger Distance
As in Sophia (Liu et al., 2023), individual update steps
can occasionally become excessively large due to the high
variance of Hutchinson’s estimator. This issue is further
exacerbated in 3DGS by the highly nonlinear and piecewise-
continuous nature of the rasterization process, particularly
in later optimization stages where dense clusters of Gaussian
primitives frequently emerge.
In ADAM (Kingma, 2014), this instability is typically ad-
dressed through careful learning-rate tuning and scheduled
decay of the position updates. On the other hand, 3DGS-
LM (H¨ollein et al., 2025) mitigates nonlinearity by introduc-
ing an adaptive regularization term that effectively interpo-
lates between Gauss–Newton and gradient descent updates.
However, both approaches operate at a global or group level
and do not exploit the strong interpretability of individual
Gaussian parameters.
We instead propose a parameter-wise trust-region strategy
that leverages the explicit geometric and photometric mean-
ing of each parameter in 3DGS. Since interactions between
Gaussian splats primarily occur through their projected opac-
ity ¯α on the image plane, a natural trust-region design should
directly limit changes to this quantity across viewpoints.
To build intuition, consider a scene containing a single
anisotropic Gaussian (see Figure 2). Intuitively, the Gaus-
sian should translate or expand more conservatively along
its short axis to avoid abrupt changes in rendered pixels.
Additionally, rotations about the long axis should be less
restrictive than those about the short axis. Finally, more
transparent Gaussians should be allowed to evolve more
rapidly than highly opaque ones. All of these constraints
can be formalized by constructing parameter-wise trust re-
gions that bound the discrepancy between the Gaussian splat
before and after an update.
Squared Hellinger distance.
We quantify the magnitude
of an update to a Gaussian G using the squared Hellinger
distance
H2(G, G′) =
Z p
G(z) −
p
G′(z)
2
dz.
(6)
5

<!-- page 6 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
Figure 3. A single Gaussian fitting example where the Gaussian splat is initialized with a small perturbation. Images from left to right
show the progression of optimization to fit the perturbed splat to the ground truth splat (denoted with the orange circle). Both ADAM and
3DGS2-TR eventually recover the Gaussian parameters close to the ground truth, but 3DGS2-TR deforms the Gaussian much less due to
the trust-region bound, which increases the stability of training.
For unnormalized Gaussian primitives G = Z · N(z; µ, Σ)
and G′ = Z′ · N(z; µ′, Σ′) with probability mass Z, Z′,
H2(G, G′) emits a closed-form solution
H2(G, G′) = 1
2(Z + Z′)
−(Z · Z′)
1
2 · Σ1 · exp(∆µT Σ−1
2 ∆µ),
where ∆µ = µ −µ′, Σ1 = det(Σ)
1
4 det(Σ′)
1
4
det

Σ+Σ′
2
 1
2
, and Σ2 =
Σ+Σ′
2
.
In the 3DGS setting, the probability mass satisfies Z =
αC det(Σ)1/2 and Z′ = α′C′ det(Σ′)1/2. While alternative
divergence measures (e.g., KL divergence) could also be
employed, we find that the squared Hellinger distance yields
trust-region bounds that are both simple to compute and
intuitively interpretable, being the integral of the coordinate-
wise difference between two Gaussian splats.
Scale normalization.
Because the apparent mass of a ren-
dered Gaussian can be arbitrarily scaled by moving along
the viewing direction, we normalize the squared Hellinger
distance by the determinant of the scale matrix. Specifi-
cally, we rescale H2(G, G′) by det(Σ)−1/2 = det(S)−1,
recalling that Σ = RT ST SR with orthogonal rotation ma-
trix R. This normalization can be interpreted as comparing
Gaussians at a fixed effective distance to the camera. For all
parameters except color, we treat the opacity α as the total
mass of the Gaussian, as only the opacity component affects
future Gaussians to be rendered.
Parameter-wise trust-region radius.
We now derive a
parameter-wise trust-region radius η = SHD(xt, ϵ) such
that, when a single parameter is perturbed while all others
remain fixed, the normalized squared Hellinger distance
between the Gaussian before the update G(xt) and after the
update G(xt + η) is bounded by ϵ:
η = SHD(xt, ϵ) := arg max
|η|

|η| : H2(G, G′)
det(S)
< ϵ

.
Below, we summarize the resulting trust-region bounds for
each parameter type; detailed derivations are deferred to
Appendix B.
Bound on the mean µ = (µx, µy, µz).
For perturbations
∆µ = (∆µx, ∆µy, ∆µz), with c ∈{x, y, z}, we require
|∆µc| <
r
−8 Σcc ln

1 −ϵ
α

.
Bound on the scale matrix S.
Let
S =


Sx
Sy
Sz

,
S′ =


S′
x
S′
y
S′
z

,
and define ∆S = S′ −S with blockwise differences
∆Sx, ∆Sy, ∆Sz, then with c ∈{x, y, z}, we require
|∆Sc| <
r
2S2c ϵ
α
.
Bound on the opacity α.
Let α′ = α + ∆α. To control
the mass discrepancy, we require
|∆α| <
√
4α ϵ.
6

<!-- page 7 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
Bound on the color C.
Let C = Cr + Cg + Cb, C′ =
C′
r + C′
g + C′
b and ∆Cc = C′
c −Cc for c ∈{r, g, b}. We
impose
|∆Cc| <
r
4Cc ϵ
α
.
Bound on the rotation R.
The rotation matrix R is pa-
rameterized by a quaternion (qx, qy, qz, qw) with squared
norm q2 = q2
x+q2
y+q2
z +q2
w. Let βc > 0 denote a geometry-
dependent constant that upper bounds the sensitivity of the
induced Gaussian displacement with respect to perturba-
tions in the c-th quaternion component (see Appendix B).
To ensure the rotation update remains within the trust region,
for c ∈{x, y, z, w} it suffices to require
|∆qc| <
r
−8
βc
ln

1 −ϵ
α

.
We illustrate the effects of the proposed Hellinger-distance-
based trust region using a single Gaussian fitting example
in Figure 3.
5. Evaluations
We now present the evaluation results of our method. The
forward-mode automatic differentiation of the 3DGS ras-
terization kernel is implemented in CUDA, based on an
existing third-party implementation. The parameter-wise
trust-region radius is implemented as a custom CUDA ker-
nel. The training pipeline and the Sophia optimizer is im-
plemented in PyTorch following the original 3DGS imple-
mentation.
Our experiments are run on A100-SXM4 GPUs with 6921
CUDA cores and 80 GB VRAM. We evaluate our method
on the same datasets as 3DGS, namely, all scenes from
MiP-NeRF360, two scenes from Tanks & Temples, and two
scenes from Deep Blending (Barron et al., 2022; Knapitsch
et al., 2017; Hedman et al., 2018). For each dataset, we
initialize the 3D Gaussian splats with the standard SfM
point cloud. As our diagonal estimator currently does not
handle inserting new diagonal entries, all of our experiments
are performed without densification.
In all experiments, we set the Hessian diagonal update inter-
val to l = 10 to balance computational cost and curvature
accuracy, following the default configuration of Sophia (Liu
et al., 2023), and choose the exponential moving average
(EMA) parameters θ1 = 0.9 and θ2 = 0.999. The trust-
region radius ϵ follows a exponential decay schedule from
10−6 to 10−8 over the course of training.
For our baselines, we compare our method against ADAM,
the SOTA first-order method, and 3DGS-LM, the only
second-order optimizer for 3DGS with open-source imple-
mentation.
We also include an ablation study which applies our
Hellinger-distance-based trust region to the ADAM update
step (ADAM-TR).
Since 3DGS-LM has a high computation overhead, we only
run it for 150 iterations in total, and report the results at 35,
75, and 150 iterations. We run all other methods for 30k
iterations and report results at 7k, 15k, and 30k iterations.
We use the same train/test split and report the same metrics
(SSIM, PSNR, LPIPS) on the test images as proposed by
3DGS.
The main quantitative results are presented in Table 1. (Re-
sults for individual scenes can be found in Appendix A.)
Qualitative comparisons for some scenes are shown in Fig-
ure 4. 3DGS2-TR significantly outperforms other methods,
reaching comparable or better reconstruction quality using
50% fewer iterations than first-order methods (ADAM and
ADAM-TR). In the case of Tanks & Temples, 3DGS2-TR
is able to exceed the best ADAM PSNR by 0.56dB at 7k
iterations and 1.19dB at 30k iterations. Moreover, 3DGS2-
TR requires less than 1GB of additional GPU memory for
training in all scenes, which is 17% more than ADAM and
85% less than 3DGS-LM on average.
In contrast, 3DGS-LM does not produce comparable results
with the same initialization, given significantly more time
and resources. We also note that by applying our trust-
region clipping to the ADAM updates (ADAM-TR), we
are already able to achieve better quality than the vanilla
ADAM optimizer.
6. Discussions and Limitations
Our method achieves much faster convergence per step com-
pared to ADAM. However, due to a naive implementation of
the Sophia optimizer in PyTorch, our second-order method
has a higher run time per iteration, which is not a fundamen-
tal limitation of the algorithm itself. We plan to optimize the
PyTorch implementation to reach comparable performance
to ADAM for tensor manipulation in future work.
Compared to other proposed second-order optimizers, our
method is fairly non-intrusive to the vanilla training pipeline,
which makes it amenable to orthogonal improvements
to 3DGS training, such as Markov Chain Monte Carlo
(MCMC) densification (Kheradmand et al., 2024), co-
regularization (Zhang et al., 2024a), drop-out Gaussians
(Park et al., 2025), etc. However, one current limitation with
our work is that we do not support adding or moving around
Gaussian splats. Naively inserting rows and columns to the
Hessian matrix results in a biased estimation of the diagonal
entries, leading to degraded performance, which we aim to
resolve in future work.
7

<!-- page 8 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
Table 1. The main quantitative result. Best results are boldfaced and second best are underlined.
Method
SSIM ↓
PSNR ↑
LPIPS ↓
Time
(s)
Peak GPU
Mem (GB)
@7k
@15k
@30k
@7k
@15k
@30k
@7k
@15k
@30k
mipnerf
ADAM
0.667
0.679
0.684
24.45
24.85
25.08
0.406
0.400
0.420
212
5.60
3DGS-LM 1
0.493
0.520
0.549
20.02
21.02
21.88
0.598
0.577
0.548
2602
47.75
ADAM-TR (Ours)
0.674
0.688
0.693
24.54
25.02
25.21
0.394
0.388
0.411
235
5.64
3DGS2-TR (Ours) 2
0.682
0.692
0.696
24.80
25.19
25.39
0.390
0.385
0.402
652
6.51
deepblending
ADAM
0.845
0.854
0.857
26.65
27.18
27.36
0.372
0.365
0.387
180
5.42
3DGS-LM 1
0.810
0.833
0.845
23.72
24.89
25.62
0.437
0.405
0.385
2235
33.33
ADAM-TR (Ours)
0.855
0.866
0.869
26.56
27.34
27.58
0.357
0.351
0.375
195
5.51
3DGS2-TR (Ours) 2
0.859
0.867
0.869
26.91
27.54
27.74
0.354
0.349
0.367
600
6.26
tandt
ADAM
0.723
0.755
0.766
20.69
21.41
21.66
0.309
0.298
0.341
201
3.17
3DGS-LM 1
0.661
0.717
0.749
20.00
21.09
21.81
0.429
0.366
0.331
2105
25.53
ADAM-TR (Ours)
0.774
0.791
0.798
21.62
22.16
22.42
0.273
0.264
0.295
227
3.27
3DGS2-TR (Ours) 2
0.787
0.800
0.805
22.12
22.63
22.85
0.262
0.255
0.279
484
4.12
1 3DGS-LM results are reported @35, @75, and @150 iterations.
2 System implementation is not fully optimized. See Section 6 for details.
Figure 4. Qualitative comparison of different methods of the truck, playroom, and room scenes. The red boxes highlight regions where
3DGS2-TR significantly outperforms other methods.
7. Conclusion
We present 3DGS2-TR, a second-order optimizer for train-
ing 3D Gaussian Splatting scenes. We approximate the
curvature information of the loss function using a stochastic
diagonal estimator, which eliminates the storage and com-
putation overhead of classical second-order methods, thus
maintains the same O(n) complexity as ADAM in both
computational cost and memory. Furthermore, we intro-
duce a parameter-wise trust-region technique based on the
squared Hellinger distance to bound the update step size.
Instead of tuning learning rates for each parameter group,
our method requires only one hyperparameter. We show
that 3DGS2-TR achieves better reconstruction quality after
50% of training iterations, while incurring only 10% of com-
putation overhead and less than 1GB of memory overhead
compared to ADAM. Our method is a drop-in replacement
for the ADAM optimizer to accelerate 3DGS training in all
settings.
8

<!-- page 9 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.
References
Ali, M. S., Qamar, M., Bae, S.-H., and Tartaglione, E. Trim-
ming the fat: Efficient compression of 3d gaussian splats
through pruning. arXiv preprint arXiv:2406.18214, 2024.
Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P.,
and Hedman, P. Mip-nerf 360: Unbounded anti-aliased
neural radiance fields. CVPR, 2022.
Bottou, L., Curtis, F. E., and Nocedal, J. Optimization
methods for large-scale machine learning. SIAM review,
60(2):223–311, 2018.
Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Gan-
guli, S., and Bengio, Y. Identifying and attacking the
saddle point problem in high-dimensional non-convex
optimization. Advances in neural information processing
systems, 27, 2014.
Durvasula, S., Zhao, A., Chen, F., Liang, R., Sanjaya, P. K.,
and Vijaykumar, N. Distwar: Fast differentiable render-
ing on raster-based rendering pipelines. arXiv preprint
arXiv:2401.05345, 2023.
Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z., et al.
Lightgaussian: Unbounded 3d gaussian compression with
15x reduction and 200+ fps. Advances in neural informa-
tion processing systems, 37:140138–140158, 2024.
Fang, G. and Wang, B. Mini-splatting: Representing scenes
with a constrained number of gaussians. In European
Conference on Computer Vision, pp. 165–181. Springer,
2024.
Feng, G., Chen, S., Fu, R., Liao, Z., Wang, Y., Liu, T.,
Hu, B., Xu, L., Pei, Z., Li, H., et al. Flashgs: Efficient
3d gaussian splatting for large-scale and high-resolution
rendering. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pp. 26652–26662, 2025.
Girish, S., Gupta, K., and Shrivastava, A. Eagles: Efficient
accelerated 3d gaussians with lightweight encodings. In
European Conference on Computer Vision, pp. 54–71.
Springer, 2024.
Hanson, A., Tu, A., Lin, G., Singla, V., Zwicker, M., and
Goldstein, T. Speedy-splat: Fast 3d gaussian splatting
with sparse pixels and sparse primitives. In Proceedings
of the Computer Vision and Pattern Recognition Confer-
ence, pp. 21537–21546, 2025a.
Hanson, A., Tu, A., Singla, V., Jayawardhana, M., Zwicker,
M., and Goldstein, T. Pup 3d-gs: Principled uncertainty
pruning for 3d gaussian splatting. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pp.
5949–5958, 2025b.
Hedman, P., Philip, J., Price, T., Frahm, J.-M., Drettakis,
G., and Brostow, G. Deep blending for free-viewpoint
image-based rendering. 37(6):257:1–257:15, 2018.
H¨ollein, L., Boˇziˇc, A., Zollh¨ofer, M., and Nießner, M. 3dgs-
lm: Faster gaussian-splatting optimization with levenberg-
marquardt. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pp. 26740–26750,
2025.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis, G. 3d
gaussian splatting for real-time radiance field rendering.
ACM Trans. Graph., 42(4):139–1, 2023.
Kerbl, B., Meuleman, A., Kopanas, G., Wimmer, M., Lan-
vin, A., and Drettakis, G. A hierarchical 3d gaussian rep-
resentation for real-time rendering of very large datasets.
ACM Transactions on Graphics (TOG), 43(4):1–15, 2024.
Kheradmand, S., Rebain, D., Sharma, G., Sun, W., Tseng,
Y.-C., Isack, H., Kar, A., Tagliasacchi, A., and Yi, K. M.
3d gaussian splatting as markov chain monte carlo. Ad-
vances in Neural Information Processing Systems, 37:
80965–80986, 2024.
Kingma, D. P. Adam: A method for stochastic optimization.
arXiv preprint arXiv:1412.6980, 2014.
Knapitsch, A., Park, J., Zhou, Q.-Y., and Koltun, V. Tanks
and temples: Benchmarking large-scale scene reconstruc-
tion. ACM Transactions on Graphics, 36(4), 2017.
Lan, L., Shao, T., Lu, Z., Zhang, Y., Jiang, C., and Yang,
Y. 3dgs2: Near second-order converging 3d gaussian
splatting. In Proceedings of the Special Interest Group
on Computer Graphics and Interactive Techniques Con-
ference Conference Papers, pp. 1–10, 2025.
Lee, J. C., Rho, D., Sun, X., Ko, J. H., and Park, E. Compact
3d gaussian representation for radiance field. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 21719–21728, 2024.
Liu, H., Li, Z., Hall, D., Liang, P., and Ma, T. Sophia: A
scalable stochastic second-order optimizer for language
model pre-training. arXiv preprint arXiv:2305.14342,
2023.
Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., and
Dai, B. Scaffold-gs: Structured 3d gaussians for view-
adaptive rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pp. 20654–20664, 2024.
9

<!-- page 10 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
Mallick, S. S., Goel, R., Kerbl, B., Steinberger, M., Carrasco,
F. V., and De La Torre, F. Taming 3dgs: High-quality
radiance fields with limited resources. In SIGGRAPH
Asia 2024 Conference Papers, pp. 1–11, 2024.
Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T.,
Ramamoorthi, R., and Ng, R. Nerf: Representing scenes
as neural radiance fields for view synthesis. Communica-
tions of the ACM, 65(1):99–106, 2021.
Papantonakis, P., Kopanas, G., Kerbl, B., Lanvin, A., and
Drettakis, G. Reducing the memory footprint of 3d gaus-
sian splatting. Proceedings of the ACM on Computer
Graphics and Interactive Techniques, 7(1):1–17, 2024.
Park, H., Ryu, G., and Kim, W. Dropgaussian: Structural
regularization for sparse-view gaussian splatting. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference, pp. 21600–21609, 2025.
Pehlivan, H., Camiletto, A. B., Foo, L. G., Habermann,
M., and Theobalt, C. Second-order optimization of gaus-
sian splats with importance sampling. arXiv preprint
arXiv:2504.12905, 2025.
Rota Bul`o, S., Porzi, L., and Kontschieder, P. Revising den-
sification in gaussian splatting. In European Conference
on Computer Vision, pp. 347–362. Springer, 2024.
Schaul, T., Zhang, S., and LeCun, Y. No more pesky learn-
ing rates. In Proceedings of the 30th International Con-
ference on Machine Learning, pp. 343–351, 2013.
Song, K., Zeng, X., Ren, C., and Zhang, J. City-on-web:
Real-time neural rendering of large-scale scenes on the
web. In European Conference on Computer Vision, pp.
385–402. Springer, 2024.
Sutton, R. S. Two problems with backpropagation and
other steepest-descent learning procedures for networks.
In Proceedings of the annual meeting of the cognitive
science society, volume 8, 1986.
Wang, X., Yi, R., and Ma, L. Adr-gaussian: Accelerating
gaussian splatting with adaptive radius. In SIGGRAPH
Asia 2024 Conference Papers, pp. 1–10, 2024.
Yao, Z., Gholami, A., Shen, S., Mustafa, M., Keutzer, K.,
and Mahoney, M. Adahessian: An adaptive second order
optimizer for machine learning. In proceedings of the
AAAI conference on artificial intelligence, volume 35, pp.
10665–10673, 2021.
Ye, V., Li, R., Kerr, J., Turkulainen, M., Yi, B., Pan, Z.,
Seiskari, O., Ye, J., Hu, J., Tancik, M., et al. gsplat: An
open-source library for gaussian splatting. Journal of
Machine Learning Research, 26(34):1–17, 2025.
Yu, Z., Chen, A., Huang, B., Sattler, T., and Geiger, A. Mip-
splatting: Alias-free 3d gaussian splatting. In Proceedings
of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 19447–19456, 2024.
Zhang, J., Li, J., Yu, X., Huang, L., Gu, L., Zheng, J., and
Bai, X. Cor-gs: sparse-view 3d gaussian splatting via
co-regularization. In European Conference on Computer
Vision, pp. 335–352. Springer, 2024a.
Zhang, Z., Song, T., Lee, Y., Yang, L., Peng, C., Chellappa,
R., and Fan, D. Lp-3dgs: Learning to prune 3d gaussian
splatting. Advances in Neural Information Processing
Systems, 37:122434–122457, 2024b.
Zhao, H., Weng, H., Lu, D., Li, A., Li, J., Panda, A., and
Xie, S. On scaling up 3d gaussian splatting training. In
European Conference on Computer Vision, pp. 14–36.
Springer, 2024.
10

<!-- page 11 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
A. Quantitative Results Per Scene
Table 2. Results on Deep Blending. Best results are boldfaced and second best are underlined.
Method
Scene
Deep Blending
SSIM ↓
PSNR ↑
LPIPS ↓
Time
(s)
Peak GPU
Mem (GB)
@7k
@15k
@30k
@7k
@15k
@30k
@7k
@15k
@30k
ADAM
drjohnson
0.841
0.855
0.861
26.60
27.39
27.64
0.368
0.360
0.390
182
5.82
3DGS-LM 1
drjohnson
0.811
0.837
0.849
24.78
25.87
26.37
0.432
0.393
0.373
3344
47.65
ADAM-TR (Ours)
drjohnson
0.852
0.865
0.869
26.50
27.30
27.56
0.355
0.347
0.378
200
5.90
3DGS2-TR (Ours) 2
drjohnson
0.855
0.865
0.869
26.54
27.25
27.48
0.353
0.346
0.371
606
6.68
ADAM
playroom
0.849
0.852
0.854
26.70
26.97
27.07
0.376
0.371
0.385
177
5.03
3DGS-LM 1
playroom
0.809
0.829
0.840
22.66
23.91
24.87
0.443
0.417
0.398
1125
19.01
ADAM-TR (Ours)
playroom
0.859
0.867
0.869
26.63
27.38
27.61
0.359
0.354
0.373
189
5.12
3DGS2-TR (Ours) 2
playroom
0.863
0.868
0.870
27.28
27.84
27.99
0.354
0.351
0.363
594
5.84
1 3DGS-LM results are reported @35, @75, and @150 iterations.
2 System implementation is not fully optimized. See Section 6 for details.
Table 3. Results on Tanks & Temples. Best results are boldfaced and second best are underlined.
Method
Scene
Tanks & Temples
SSIM ↓
PSNR ↑
LPIPS ↓
Time
(s)
Peak GPU
Mem (GB)
@7k
@15k
@30k
@7k
@15k
@30k
@7k
@15k
@30k
ADAM
train
0.697
0.745
0.758
19.83
20.91
21.21
0.318
0.302
0.362
209
3.45
3DGS-LM 1
train
0.644
0.688
0.720
19.13
19.92
20.57
0.425
0.377
0.347
2162
28.36
ADAM-TR (Ours)
train
0.740
0.763
0.771
20.21
20.87
21.22
0.297
0.287
0.321
229
3.54
3DGS2-TR (Ours) 2
train
0.757
0.775
0.781
20.68
21.32
21.62
0.283
0.276
0.302
503
4.43
ADAM
truck
0.750
0.766
0.773
21.56
21.90
22.12
0.301
0.293
0.320
192
2.88
3DGS-LM 1
truck
0.679
0.747
0.779
20.86
22.25
23.05
0.432
0.355
0.315
2047
22.69
ADAM-TR (Ours)
truck
0.807
0.819
0.824
23.02
23.44
23.62
0.250
0.242
0.268
224
3.00
3DGS2-TR (Ours) 2
truck
0.816
0.826
0.829
23.55
23.93
24.07
0.240
0.234
0.255
465
3.82
1 3DGS-LM results are reported @35, @75, and @150 iterations.
2 System implementation is not fully optimized. See Section 6 for details.
11

<!-- page 12 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
Table 4. Results on MipNeRF-360. Best results are boldfaced and second best are underlined.
Method
Scene
MipNeRF-360
SSIM ↓
PSNR ↑
LPIPS ↓
Time
(s)
Peak GPU
Mem (GB)
@7k
@15k
@30k
@7k
@15k
@30k
@7k
@15k
@30k
ADAM
treehill
0.503
0.516
0.520
21.43
21.65
21.69
0.549
0.544
0.558
194
3.47
3DGS-LM 1
treehill
0.397
0.413
0.430
19.31
19.99
20.52
0.645
0.634
0.617
2648
42.58
ADAM-TR (Ours)
treehill
0.510
0.524
0.529
21.47
21.71
21.75
0.538
0.533
0.552
210
3.46
3DGS2-TR (Ours) 2
treehill
0.520
0.530
0.533
21.73
21.84
21.84
0.533
0.529
0.543
600
4.26
ADAM
counter
0.852
0.866
0.872
26.49
27.15
27.53
0.279
0.270
0.297
241
7.12
3DGS-LM 1
counter
0.678
0.709
0.741
20.68
21.95
23.07
0.505
0.480
0.447
3531
61.56
ADAM-TR (Ours)
counter
0.859
0.871
0.876
26.58
27.25
27.59
0.272
0.265
0.290
270
7.24
3DGS2-TR (Ours) 2
counter
0.865
0.874
0.877
26.91
27.51
27.77
0.268
0.263
0.282
761
8.18
ADAM
stump
0.508
0.517
0.520
22.77
22.83
22.86
0.542
0.538
0.550
182
2.60
3DGS-LM 1
stump
0.384
0.397
0.416
20.44
20.85
21.30
0.670
0.662
0.643
2121
33.61
ADAM-TR (Ours)
stump
0.512
0.528
0.532
22.78
22.93
22.99
0.529
0.524
0.546
187
2.64
3DGS2-TR (Ours) 2
stump
0.525
0.534
0.537
22.98
23.08
23.13
0.522
0.518
0.532
589
3.39
ADAM
bonsai
0.893
0.904
0.908
28.37
29.24
29.69
0.285
0.279
0.299
237
8.49
3DGS-LM 1
bonsai
0.679
0.721
0.760
21.04
22.68
24.06
0.505
0.482
0.450
1921
43.60
ADAM-TR (Ours)
bonsai
0.901
0.911
0.914
28.75
29.67
30.03
0.273
0.268
0.288
278
8.63
3DGS2-TR (Ours) 2
bonsai
0.905
0.913
0.916
29.11
29.92
30.31
0.270
0.265
0.283
750
9.59
ADAM
bicycle
0.480
0.498
0.504
21.58
21.84
21.95
0.523
0.516
0.539
185
3.70
3DGS-LM 1
bicycle
0.346
0.360
0.376
18.96
19.47
19.89
0.676
0.666
0.652
2265
42.61
ADAM-TR (Ours)
bicycle
0.494
0.516
0.523
21.70
22.04
22.11
0.505
0.496
0.530
203
3.76
3DGS2-TR (Ours) 2
bicycle
0.511
0.526
0.532
21.92
22.17
22.24
0.494
0.488
0.510
565
4.53
ADAM
kitchen
0.880
0.891
0.898
27.95
28.64
29.11
0.193
0.185
0.209
271
8.41
3DGS-LM 1
kitchen
0.592
0.640
0.698
20.94
22.38
23.66
0.490
0.450
0.401
3257
66.58
ADAM-TR (Ours)
kitchen
0.888
0.897
0.902
28.03
28.81
29.20
0.186
0.179
0.200
307
8.35
3DGS2-TR (Ours) 2
kitchen
0.892
0.899
0.903
28.51
29.13
29.47
0.183
0.177
0.194
761
9.51
ADAM
flowers
0.353
0.362
0.365
18.77
18.88
18.89
0.605
0.602
0.614
187
3.41
3DGS-LM 1
flowers
0.265
0.277
0.292
17.02
17.51
17.89
0.725
0.711
0.690
2654
48.45
ADAM-TR (Ours)
flowers
0.363
0.374
0.377
18.81
18.94
18.97
0.594
0.590
0.605
202
3.39
3DGS2-TR (Ours) 2
flowers
0.370
0.377
0.379
18.93
19.02
19.04
0.592
0.589
0.599
555
4.18
ADAM
room
0.870
0.879
0.883
28.80
29.33
29.67
0.309
0.302
0.323
212
8.82
3DGS-LM 1
room
0.718
0.766
0.797
21.96
23.79
25.32
0.488
0.467
0.438
2696
49.79
ADAM-TR (Ours)
room
0.877
0.886
0.890
28.77
29.49
29.81
0.296
0.290
0.312
238
8.91
3DGS2-TR (Ours) 2
room
0.881
0.888
0.891
29.00
29.70
30.18
0.292
0.286
0.304
691
9.72
ADAM
garden
0.660
0.678
0.687
23.87
24.11
24.29
0.370
0.363
0.387
202
4.36
3DGS-LM 1
garden
0.373
0.394
0.428
19.88
20.53
21.23
0.674
0.644
0.595
2327
40.97
ADAM-TR (Ours)
garden
0.664
0.685
0.692
23.98
24.32
24.45
0.356
0.349
0.379
217
4.42
3DGS2-TR (Ours) 2
garden
0.672
0.688
0.694
24.12
24.37
24.50
0.353
0.346
0.369
591
5.26
1 3DGS-LM results are reported @35, @75, and @150 iterations.
2 System implementation is not fully optimized. See Section 6 for details.
12

<!-- page 13 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
B. Parameter-wise trust-region bounds based on squared Hellinger distance
We derive parameter-wise trust-region bounds such that when one parameter is updated while all other parameters fixed, the
normalized distance between the Gaussian primitive before and after the update guarantees
H2(G, G′)
det(S)
< ϵ.
Bound on the mean µ.
Let µ = (µx, µy, µz), µ′ = (µ′
x, µ′
y, µ′
z), ∆µ = µ′−µ. We first let ∆µx ̸= 0, ∆µy = 0, ∆µz = 0,
then
H2(G, G′) = C

1 −exp

−∆µT Σ−1∆µ
8

= C

1 −exp

−∆µ2
x
8Σxx

.
Setting H2(G, G′)/ det(S) < ϵ gives α

1 −exp

−∆µ2
x
8Σxx

< ϵ. After rearranging the terms, we have
|∆µx| <
r
−8 Σxx ln

1 −ϵ
α

.
Using similar derivation, we have
|∆µy| <
r
−8 Σyy ln

1 −ϵ
α

,
|∆µz| <
r
−8 Σzz ln

1 −ϵ
α

.
Bound on the scale matrix S.
Let
S =


Sx
Sy
Sz

,
S′ =


S′
x
S′
y
S′
z

,
and define ∆S = S′ −S with differences ∆Sx, ∆Sy, ∆Sz.
Note that C = αC det(Σ)1/2 = αC det(S), C′ = αC det(S′), we first let ∆Sx ̸= 0 while ∆Sy = 0, ∆Sz = 0, and denote
ρx = αSySz. Then
H2(G, G′) = 1
2(ρxSx + ρxS′
x) −ρx(SxS′
x)1/2 (Sx)
1
2 (S′
x)
1
2 SySz
( S2x+S′2
x
2
)
1
2 SySz
= ρx
Sx + S′
x
2
−ρx
SxS′
x

S2x+S′2
x
2
 1
2
Since S′
x = Sx + ∆Sx, we have
d
d∆Sx H2 = ρx

1
2 +
−2S2
x
(2S2x+∆Sx)2

, and
d2
d∆S2x H2 = ρx(4S2
x · (2Sx + ∆Sx)−3). We thus
can approximate H2(G, G′) ≈
ρx
2Sx ∆S2
x.
Setting H2(G, G′)/ det S < ϵ gives
|∆Sx| <
s
2Sx det(S)ϵ
ρx
=
r
2S2xϵ
α
.
We can similarly obtain
|∆Sy| <
r
2S2y ϵ
α
, and |∆Sz| <
r
2S2z ϵ
α
.
Bound on the opacity α.
Let ρα = det(Σ)1/2 = det(S), then
H2(G, G′) = ρα
α + α′
2
−(αα′)1/2

.
13

<!-- page 14 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
Let α′ = α + ∆α, then
d
d∆αH2 = ρα
1
2 −1
2α(α2 + α∆α)−1/2

and
d2
d∆α2 H2 = ρα
1
4α2(α2 + α∆α)−3/2

.
Thus, we can approximate
H2(G, G′) ≈ρα
4α∆α2.
Setting H2(P, Q)/ det(S) < ϵ gives
|∆α| <
s
4α det(S)ϵ
ρα
=
√
4αϵ.
Bound on the color C.
Let C = Cr + Cg + Cb and similarly define C′ and ∆Cc = C′
c −Cc for c ∈{r, g, b}. First, let
∆Cr ̸= 0 and ∆Cg = ∆Cb = 0. Let ρr = α det(Σ)1/2 = α det(S), then
H2(G, G′) = ρr
1
2(Cr + C′
r) −(CrC′
r)1/2

.
We can approximate
H2(G, G′) ≈ρr
4Cr
∆C2
r.
Setting H2(P, Q)/ det(S) < ϵ gives
|∆Cr| <
s
4Cr det(S)ϵ
ρr
=
r
4Crϵ
α .
Using similar derivation, we also have
|∆Cg| <
r
4Cg ϵ
α
, |∆Cb| <
r
4Cb ϵ
α
.
Bound on the rotation R.
The rotation matrix R is parameterized by a unnormalized quaternion ˜q = (qx, qy, qz, qw) with
squared norm r2 = ∥q∥2 = q2
x + q2
y + q2
z + q2
w, yielding
R =


1 −
2(q2
y+q2
z)
r2
2(qxqy−qwqz)
r2
2(qxqz+qwqy)
r2
2(qxqy+qwqz)
r2
1 −2(q2
z+q2
x)
r2
2(qyqz−qwqx)
r2
2(qxqz−qwqy)
r2
2(qyqz+qwqx)
r2
1 −
2(q2
x+q2
y)
r2

.
We can similarly construct R′, which is parameterized by another unnormalized quaternion q′ = (q′
x, q′
y, q′
z, q′
w) with
squared norm r′2 = ∥q′∥2 = q′2
x + q′2
y + q′2
z + q′2
w.
Since Σ = RT ST SR, Σ′ = R′T ST SR′ and R, R′ are orthogonal, we have det(Σ) = det(Σ′) = det(S)2. Let ∆R =
R−1R′ also be an orthogonal rotation matrix. Then
H2(G, G′) = C
 
1 −
det(S)
det( Σ+Σ′
2
)
1
2
!
= C
 
1 −
det(S)
det( S2+∆RT S2∆R
2
)
1
2
!
.
Setting H2(G, G′)/ det(S) < ϵ and rearranging the terms gives
1 −
det(S)
det( S2+∆RT S2∆R
2
)
1
2 < ϵ
α.
14

<!-- page 15 -->
Scalable Second-Order Optimizer with Hellinger Distance Trust-Region for Large-Scale 3D Gaussian Splatting
To satisfy the above inequality, it is sufficient that det

S2+∆RT S2∆R
2

<

det(S)
1−ϵ/α
2
, which is sufficient if
det

S2 
I+S−2∆RT S2∆R
2

<

det(S)
1−ϵ/α
2
. Since for two matrices A and B, det(AB) = det(A) det(B), we only
need det

I+S−2∆RT S2∆R
2

<

1
1−ϵ/α
2
, equivalently,
det

I +

−I + I + S−2∆RT S2∆R
2

<

1
1 −ϵ/α
2
.
By the relation det(I + X) ≤exp(tr(X)), it is sufficient to have
tr
I + S−2∆RT S2∆R
2
−I

< −2 ln

1 −ϵ
α

.
By the linearity of trace, we only need to show
tr(S−2∆RT S2∆R) −3 < −4 ln

1 −ϵ
α

.
(7)
Next, we investigate the approximation of tr(S−2∆RT S2∆R). For an unnormalized quaternion ˜q = (qx, qy, qz, qw) with
r2 = ∥˜q∥2, the unnormalized rotation matrix is
˜R(˜q) =


r2 −2(q2
y + q2
z)
(2qxqy −2qwqz)
(2qxqz + 2qwqy)
(2qxqy + 2qwqz)
r2 −2(q2
z + q2
x)
(2qyqz −2qwqx)
(2qxqz −2qwqy)
(2qyqz + 2qwqx)
r2 −2(q2
x + q2
y)


which can be normalized as R = ˜R/r2.
Let the update to the unnormalized quaternion be ∆q = (∆qx, ∆qy, ∆qz, ∆qw). We consider an update in the direction of
∆q with a step size a, denoted as q′ = ˜q +a∆q with r′2 = (qx +a∆qx)2 +(qy +a∆qy)2 +(qz +a∆qz)2 +(qw +a∆qw)2
Let R′ = ˜R(q′)/r′2 = R∆R = R(I + E), where R, ∆R, R′ are orthogonal, and E = RT R′ −I = RT ˜R′/r′2 −I. Then
with the relation tr(S−2∆RT S2∆R) = tr(S−1∆RT SS∆RS−1) = ∥S∆RS−1∥2
F = ∥S(I + E)S−1∥2
F , we have
∂
∂x∥S(I + E)S−1∥2
F = 2tr((S(I + E)S−1)T (S∂xES−1)),
∂2
∂x2 ∥S(I + E)S−1∥2
F = 2∥S∂xES−1∥2
F + 2tr((S(I + E)S−1)T (S∂2
xES−1)),
where the partial derivative to x is short for the partial derivative to qx. We have similar formulas for the partial derivative to
qy, qz, qw. For the variable qc ∈{qx, qy, qz, qw}, it can be computed that ∂cE = RT (r−2∂c ˜R −2qcr−4 ˜R), and
∂2
cE = RT (−2qcr−4∂c ˜R + r−2∂2
c ˜R + 6q2
cr−6 ˜Rc −2qcq−4∂c ˜R −2q−4 ˜R + 2q2
cr−6 ˜R)
= RT (−2qcr−4∂c ˜R + r−2∂2
c ˜R + 8q2
cr−6 ˜R −2qcr−4∂c ˜R −2r−4 ˜R).
Denote T(∆q) = ∥S(I+E)S−1∥2
F , then when ∆qx ̸= 0, ∆qy = 0, ∆qz = 0, ∆qw = 0, we have T(∆qx) = 3, T ′(∆qx) =
0, βx = T ′′(∆qx) = 2∥S∂xES−1∥2
F +2tr(∂2
xE). We thus can approximate tr(S−2∆RT S2∆R) by its second-order Taylor
expansion and obtain the approximation of (7) as 3 + 1
2βx(∆qx)2 −3 < −4 ln(1 −ϵ/α). Rearranging the terms, we obtain
|∆qx| <
p
−8/βx ln(1 −ϵ/α).
Using similar derivation, we also obtain the bounds
|∆qy| <
q
−8/βy ln(1 −ϵ/α), |∆qz| <
p
−8/βz ln(1 −ϵ/α), and |∆qw| <
p
−8/βw ln(1 −ϵ/α).
15
