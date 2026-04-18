<!-- page 1 -->
Moment-Based 3D Gaussian Splatting: Resolving Volumetric Occlusion with
Order-Independent Transmittance
Jan U. M¨uller
Robin Tim Landsgesell
Leif Van Holland
Patrick Stotko
Reinhard Klein
University of Bonn
muellerj@cs.uni-bonn.de, landsgesell@uni-bonn.de, {holland,stotko,rk}@cs.uni-bonn.de
Abstract
The recent success of 3D Gaussian Splatting (3DGS) has
reshaped novel view synthesis by enabling fast optimiza-
tion and real-time rendering of high-quality radiance fields.
However, it relies on simplified, order-dependent alpha
blending and coarse approximations of the density inte-
gral within the rasterizer, thereby limiting its ability to
render complex, overlapping semi-transparent objects. In
this paper, we extend rasterization-based rendering of 3D
Gaussian representations with a novel method for high-
fidelity transmittance computation, entirely avoiding the
need for ray tracing or per-pixel sample sorting. Building
on prior work in moment-based order-independent trans-
parency, our key idea is to characterize the density distribu-
tion along each camera ray with a compact and continuous
representation based on statistical moments. To this end, we
analytically derive and compute a set of per-pixel moments
from all contributing 3D Gaussians. From these moments, a
continuous transmittance function is reconstructed for each
ray, which is then independently sampled within each Gaus-
sian. As a result, our method bridges the gap between ras-
terization and physical accuracy by modeling light attenu-
ation in complex translucent media, significantly improving
overall reconstruction and rendering quality.
1. Introduction
Novel view synthesis has witnessed tremendous progress
in recent years, driven initially by volumetric approaches
such as Neural Radiance Fields (NeRF) [24] and its many
extensions[3–5]. These implicit radiance-field formulations
substantially improved visual fidelity through physically
motivated volumetric integration. More recently, the emer-
gence of 3D Gaussian Splatting (3DGS) [18] has shifted
the paradigm toward explicit representations. By model-
ing scenes as collections of 3D Gaussians, 3DGS enables
remarkably fast training and real-time rendering while con-
tinuing to achieve state-of-the-art image quality. However,
this efficiency comes at the cost of physical accuracy. Core
approximations limit the faithfulness and robustness of the
representation, which includes replacing volumetric inte-
gration with splatting, assuming non-overlapping Gaussians
with a correct front-to-back ordering, and modeling opacity
independently of the spatial extent of Gaussians.
Recent works have begun addressing these shortcom-
ings from two opposing directions. Some methods aban-
don splatting entirely and instead adopt ray-tracing-based
formulations that avoid the above approximations and pro-
vide physically accurate volumetric rendering of Gaussian
primitives [10, 25].
Others retain splatting to preserve
its performance advantages, but target specific limitations.
StopThePop [33] mitigates popping artifacts caused by in-
correct Gaussian ordering under view changes, whereas
Vol3DGS [36] reintroduces proper volumetric integration
of density, yet still assumes non-overlapping Gaussians and
requires a correct rendering order. Despite this progress,
achieving the physical accuracy of ray-traced approaches
while maintaining the high efficiency of rasterization-based
splatting remains an open problem.
In this work, we present MB3DGS, a splatting-based
method that performs accurate, order-independent rasteriza-
tion of 3D Gaussians via moments. In contrast to prior ap-
proaches, we treat opacity in a fully volumetric manner by
modeling the combined density of potentially overlapping
Gaussians to compute the exact emitted radiance. Assum-
ing only piecewise-constant density, analogous to volumet-
ric methods such as NeRF, we derive an efficient numerical
quadrature rule for radiance computation. To achieve order-
independent rendering, we reconstruct the moments of the
transmittance function.
Since a naive formulation intro-
duces numerical instability, we employ a power transform
and derive a closed-form recurrence relation between mo-
ments. Combined with a confidence-interval-based formu-
lation that produces correct screen-space bounds for more
efficient rasterization, MB3DGS yields more stable and
consistent results, particularly in visually complex regions
where accurate volumetric modeling is essential.
In summary, our key contributions are:
1
arXiv:2512.11800v1  [cs.CV]  12 Dec 2025

<!-- page 2 -->
• A splatting-based, physically accurate rendering formu-
lation that computes emitted radiance from the combined
density of potentially overlapping 3D Gaussians.
• An efficient numerical quadrature rule derived under the
assumption of piecewise-constant density, enabling volu-
metric radiance computation.
• A power-transform-based moment representation with
a closed-form recurrence, resolving numerical insta-
bility in transmittance-moment computation, enabling
order-independent rasterization without requiring Gaus-
sian sorting.
• A confidence-interval-based screen-space bounding strat-
egy that enables robust and faster rasterization and im-
proves consistency in visually complex regions.
We release our code, data, and additional results at:
https://vc-bonn.github.io/mb3dgs/
2. Related Work
Splatting-based Approaches.
3D Gaussian Splatting
(3DGS) [18] enables real-time, high-quality radiance field
rendering and has sparked extensive follow-up work ad-
dressing its artifacts and physical limitations. Several meth-
ods reduce view-dependent popping via improved or sort-
free rasterization [16, 19, 33], while others mitigate aliasing
and scale distortions using Mip-filtering or analytic integra-
tion [21, 42]. More physically grounded variants introduce
volumetrically consistent integration [13, 36], explore al-
ternative primitive parameterizations [17], or extend splat-
ting to complex cameras and secondary effects [40]. Some
works argue that optimization, rather than volumetric cor-
rectness, is the primary driver of fidelity [9].
Our method differs by explicitly modeling the combined
density of overlapping Gaussians to compute physically ac-
curate transmittance within a rasterization pipeline.
Ray Tracing-based Approaches.
To overcome the inher-
ent limitations of splatting, another line of research adopts
ray tracing for Gaussian primitives. Moenne et al. [25] and
Mai et al. [22] introduce efficient frameworks for volumet-
ric or ellipsoidal ray tracing, eliminating popping and en-
abling physically based effects Blanc et al. [7] and Condor
et al. [10] propose Gaussian primitives tailored for volumet-
ric ray-traced rendering, while other works address trans-
parency via stochastic sampling [35] or introduce Gaus-
sian opacity fields for volumetric geometry extraction [43].
While these approaches offer high physical accuracy, they
rely on the computationally heavier ray-tracing pipeline.
In contrast, our method achieves volumetric fidelity
through moment-based transmittance reconstruction while
retaining the efficiency of rasterization.
Order-Independent Transparency.
Order-independent
transparency (OIT) techniques seek correct blending of
semi-transparent geometry without sorting. Classic exact
methods such as the A-buffer [8] and depth peeling [11] are
accurate but costly, motivating real-time approximations in-
cluding weighted averaging [6], physically inspired blend-
ing [23], and moment-based transparency [26], as well as
recent learning-based methods [38].
Our work is conceptually related to moment-based OIT
but adapts these ideas to volumetric Gaussian primitives,
enabling continuous transmittance reconstruction along
each ray within a splatting framework.
3. Moment-based 3D Gaussian Splatting
This section presents our method for approximating the vol-
ume rendering of 3D Gaussians (see Sec. 3.1) while raster-
izing each one individually. Our approach first computes
per-ray density moments to recover a continuous, order-
independent transmittance function. This function enables
the independent evaluation of the volume rendering integral
for each Gaussian via numerical quadrature (see Sec. 3.2).
We then derive a geometric proxy for rasterization that ac-
curately models perspective distortion in anisotropic Gaus-
sians and detail the use of adjoint rendering for gradient
computation (see Sec. 3.3). Finally, we describe the opti-
mization and adaptive densification process for this volu-
metric representation (see Sec. 3.4). An overview of the
process in described in Fig. 1.
3.1. Volume Rendering
In volume rendering, a participating medium is defined by a
volume in space containing particles that interact with light
through absorption, emission, and scattering. The spatial
distribution of these particles is described by a density func-
tion, representing extinction, which modulates the intensity
of these light interactions.
Following most work in this
area [4, 5, 18, 24], we only consider absorption and emis-
sion. Thus, the final appearance of the object is determined
by integrating these interactions along camera rays passing
through the medium. To solve this integration problem, we
assume density is piecewise-constant within small intervals
and recovery of the transmittance is possible from the sta-
tistical moments of the density, and we explicitly make no
simplifying assumptions regarding self-occlusion or occlu-
sion between Gaussians.
Each Gaussian particle defines both a localized density
distribution and its appearance. The density σ at a position
x is the weighted sum of all Gaussian contributions:
σ(x) =
X
i
wi G(x | µi, Σi)
(1)
where
G(x) = e−1
2 (x−µi)T Σ−1
i
(x−µi).
In
contrast
to 3DGS [18], which uses an opacity-centric model
2

<!-- page 3 -->
Moment pass
Quadrature pass
Rescale
Camera / Hyperparams
Gaussians 
Camera / Hyperparams
Vulkan
Torch
Shader pass
Accumulative
frame buffer
Figure 1. Overview of our order-independent differentiable rasterization pipeline. The forward pass features two separate accumulation
passes: a Moment pass computes and sums per-Gaussian density moments to derive a continuous transmittance function. A Quadrature
pass then independently evaluates the volume rendering integral for each Gaussian, computing and summing its radiance and penalty
contributions. Also depicted is the adjoint gradient flow, which propagates derivatives from the final loss back through all stages. Notably,
this includes backpropagation through both the radiance contributions and the transmittance to optimize the Gaussian parameters.
(wi ∈[0, 1]),
our
physically-motivated
approach
only
requires non-negative weights (wi ≥0),
enforced via
a softplus function.
The appearance is modeled by an
emission term (Le)i(d), which depends on direction d and
is represented using spherical harmonic (SH) coefficients
fi ∈R48 up to degree l = 3. Each Gaussian primitive is
thus defined by a weight wi ∈R≥0, a mean µi ∈R3, a
covariance Σi, and the SH coefficients fi. To ensure that
the covariance matrix remains positive semi-definite during
optimization,
it is parameterized via Σ = RSST RT
with a rotation R ∈SO(3) and a diagonal scaling matrix
S ∈R3×3. For all remaining details regarding the param-
eterization, we refer the reader to the original publication
[18].
The
observed
radiance
L
along
a
camera
ray
r(t) = o + t · d from origin o in direction d within
an emission-absorption medium is given by the volume
rendering equation:
L =
Z tf
tn
T(t) σ(xt) Le(xt, d) dt + T(tf) Lbg(xtf , d)
(2)
where the transmittance T(t) = e−
R t
tn σ(xs) ds describes
the mediums’ permeability, Le(x, d) is the emitted radi-
ance, xt is shorthand for r(t), tn and tf are the near and far
integration bounds, and Lbg is the incident radiance from
the background. To ensure that the total light emitted per
unit distance at x from all particles is the sum of their indi-
vidual contributions, we define the emitted radiance as
Le(x, d) =
1
σ(x)
X
i
σi(x) (Le)i(d).
(3)
To solve the integral for each Gaussian, its 3D density
contribution along the ray is first expressed as a 1D Gaus-
sian in the ray’s coordinate system:
σi(t) = wi G(xt | µi, Σi) = ωi e
−(t−µi)2
2Σ2
i
(4)
To avoid explicit inversion of Σ, we follow [25].
Let
og = S−1 RT (o −µ) and dg = S−1 RT d, then the pa-
rameters of the 1D Gaussian are given by:
Σ2
i =
1
dTg dg
,
µi = −oT
g dg
dTg dg
,
ωi = wi eK
(5)
with
K = −1
2oT
g og + 1
2
(oT
g dg)2
dTg dg
(6)
Please refer to Appendix A for the detailed derivation.
3.2. Order-Independent Transmittance
To derive the quadrature, we first note that when plugging in
Eq. (3) into Eq. (2), the density term σ cancels out. Swap-
ping the order of integration and summation isolates the in-
tegral in terms of the i-th particle and transmittance. This
ray integral is then split into a sum of integrals over con-
tinuous, non-overlapping intervals [tj, tj+1]. We assume a
piecewise constant density, such that σ(t) = σ(tj) for any
t ∈[tj, tj+1], which implies σi(t) is also piecewise con-
stant. This leads to the quadrature for the i-th particle’s
contribution to the Volume Rendering Equation (VRE):
Li ≈
N
X
j=1
(T(tj) −T(tj+1))σi(tj)
σ(tj) (Le)i(d).
(7)
with σ(tj) = −(tj+1 −tj)−1 log(T(tj+1)/T(tj)). Please
refer to Appendix B for the detailed derivation.
To solve the quadrature for each particle individually,
we adapt the work on order-independent transparency by
3

<!-- page 4 -->
(a) 3D scene configuration.
(b) 2D screen-space proxies.
Figure 2. Comparison of 2D splat proxy accuracy. The affine
approximation used by 3DGS provides a poor bound, resulting in
a proxy that provides insufficient coverage and is offset from the
true perspectively-correct level set. Our method computes a tighter
geometric proxy that conservatively bounds the true projection,
ensuring all contributions are correctly rasterized.
M¨unstermann et al. [26] to our problem statement. Their
proposed approach builds on the concepts of statistical mo-
ments and the moment problem.
Let τ : R →R be a monotonic increasing,
right-
continuous function, which defines a unique Lebesgue-
Stieltjes measure µτ.
The k-th raw moment mk of this
measure is defined as mk =
R ∞
−∞xk dµτ(x).
When τ
is absolutely continuous, this measure µτ has a density
σ(x) = τ ′(x) with respect to the Lebesgue measure, allow-
ing the moment to be computed as mk =
R ∞
−∞xkσ(x) dx.
In general, a finite number of moments does not character-
ize a unique measure. However, a lower and upper bounds
on the set of measures characterized by the finite moments
can be computed [37].
M¨unstermann et al. [26] model occlusion along a view
ray via an absorbance function A(z) = −ln T(z), with
T(z) being the transmittance at depth z.
For discrete
transparent surfaces at depths zl with opacities αl, the
absorbance function A(z) = P
l=0,zl<z −ln(1 −αl) is a
monotonic, right-continuous step function and an instance
of τ.
The associated Lebesgue-Stieltjes measure µA is
a sum of weighted Dirac delta functions, where each
surface corresponds to a point mass: µA = P
l=0 wl δzl
with weights wl = −ln(1 −αl).
Rather than storing
the full measure, its first 2n + 1 moments with n = 4,
mk =
R
zk dµA(z), are computed and stored per pixel. To
reconstruct the bounds on the absorbance at a query depth
η, a unique canonical representation of the moments is con-
structed. This representation is itself a discrete measure,
τη = Pn
i=0 w′
i δxi, with n + 1 points of support that has
the same moments as µA and is constrained such that one
of its support points is the query depth itself, i.e., x0 = η
[39].
The problem is thus reduced to finding the unknown
locations {xi}n
i=1 and weights {w′
i}n
i=0.
The locations
are found as the roots of the degree-n kernel polyno-
mial K(x) = xT H−1 η, where H is the Hankel matrix
of the moments (Hij = mi+j). Once the locations {xi}
are known, the weights are found by solving the lin-
ear Vandermonde system given by the moment equations
mk = Pn
i=0 w′
i xk
i . Finally, the bounds are computed from
this discrete measure:
L(η) =
X
xi<η
w′
i
and
U(η) =
X
xi≤η
w′
i.
(8)
The transmittance then is T(τ) = (1 −β)L + βU with
β = 0.25.
Recent work has proven that these moment-
bounds are differentiable [39].
In our volumetric setting, the optical depth along the
ray, τ(t) = −log(T(t)) =
R t
tn σt(s) ds, serves as the con-
tinuous and differentiable analog to the discrete absorbance
function A(z) used by M¨unstermann et al.. As τ(t) is abso-
lutely continuous, the Radon-Nikodym theorem guarantees
its associated Lebesgue-Stieltjes measure, µτ, has a density
with respect to the Lebesgue measure. This density is pre-
cisely the sum of 1D Gaussians, σ(t). The moments of this
measure are therefore computed by integrating against this
continuous density:
mk =
Z tf
tn
tk σ(t) dt.
(9)
However, direct computation of mk is numerically
unstable on intervals [tn, tf] where tf ≫1, as individual
particle moments grow rapidly, since (mk)i ≥ωiµk
i Σi . To
stabilize this, we warp the domain [tn, tf] to [0, 1] using the
transformation
ˆg(t) = (f(t) −f(tn))/(f(tf) −f(tn)).
M¨unstermann et al. [26] proposed this transformation
with f(t) = log(t) to address a similar problem.
The
choice of the non-linear function f(t) is critical for min-
imizing linearization error [4, 26, 27]. As parameterized
power transformations are particularly effective when
combined with local linear approximations [5], we follow
this approach and set f(t) to be the Power-Transform
fλ(2t) with λ = −1.5. This provides a robust mapping,
behaving linearly for near distances while transitioning to
an inverse-like function for far distances. For a detailed
analysis, see [2].
We therefore compute the moments by integration
powers of the warped distance against the density:
ˆmk = P
i
R tf
tn ˆg(t)kσi(t)dt.
To solve the inner integral
( ˆmk)i, we linearize ˆg(t) using a Taylor expansion at each
particle’s mean µi.
Assuming an unbounded medium
(tf →∞), this yields a recurrence for k ≥2 with closed-
4

<!-- page 5 -->
form base cases:
( ˆm0)i = ωi Σi
rπ
2 (1 −erf(bn))
(10)
( ˆm1)i = ˆg(µi) ( ˆm0)i −ˆg′(µi) ωi Σ2
i e
−(tn−µi)2
2Σ2
i
(11)
( ˆmk)i = ˆg(µi) ( ˆmk−1)i + β (k −1) ( ˆmk−2)i −Bi(k)
(12)
where
β = ˆg′(µi)2 Σ2
i
is
scaled
variance
and
near
boundary
term
Bi(k) = −ˆg′(µi) ωi Σ2
i uk−1
n
e−b2
n
with
bn = (tn −µi)/(
√
2Σi)
and
linearized
distance
un = ˆg(µi) + ˆg′(µi) (tn −µi).
An alternative to the polynomial basis, explored in order-
independent occluder literature [29–32], are trigonometric
moments ˜mk with a Fourier basis. We adapt this concept to
compute trigonometric density moments,
˜mk =
X
j
Z tf
tn

e(2π−θ)iˆg(t)k
σj(t) dt
(13)
where i is the imaginary unit. Linearizing ˆg(t) at µj via a
Taylor expansion, and again assuming tf →∞, provides a
closed-form approximation for the inner integral ( ˜mk)j:
( ˜mk)j ≈ωj
rπ
2 Σj eiαˆg(µj)−
Σ2
j β2
2
(1 −erf(vn))
(14)
with
vn = (tn −µj)/(
√
2Σj) −i(Σjβ)/
√
2,
phase
α = k(2π −θ), and β = αˆg′(µj). This requires evaluating
erf(vn) with a complex argument; we approximate this
using a first-order Taylor expansion in the imaginary
direction. Notably, ˆm0 and ˜m0 are exactly equal to the total
optical depth τ(tf) and involve no approximation. Please
refer to Appendix C for detailed derivations.
Given an order-independent estimate for the optical
depth, Eq. (7) can be estimated for each particle individ-
ually.
To correct for visual opacity fluctuations arising
from moment-based transmittance estimation, we renormal-
ize the final radiance. Following M¨unstermann et al., we
scale the accumulated radiance P Li by the ratio of the true
scene opacity, 1 −e−m0 (derived from the zeroth moment
m0), to the estimated opacity O. This O is accumulated
in the alpha-channel alongside the individual radiance con-
tributions Li by accumulating (Le)i = (r, g, b, 1) The final,
stabilized radiance L is:
L = 1 −e−m0
max(ϵ, O)
n
X
i=1
Li + e−m0 Lbg
(15)
where the denominator is clamped by ϵ > 0 for stability.
3.3. Rasterisation and Adjoint Rendering
The rendering process maps efficiently to the GPU raster-
ization pipeline because both the per-particle moment and
(a) Ground Truth
(b) Vol3DGS [36]
(c) StopThePop [33]
(d) Ours
Figure 3. Qualitative comparison on a synthetic scene designed to
evaluate complex color blending.
radiance quadrature contributions can be evaluated individ-
ually and in an arbitrary order. Our forward rendering ap-
proach, illustrated in Fig. 1, proceeds in four main stages.
First, a culling pass visibility tests each 3D Gaussian against
the camera frustum using its bounding sphere, defined by
r = max diag(S). Next, two separate accumulation passes
rasterize all visible Gaussians: the moment generation pass
evaluates and sums per-Gaussian moments, and the radi-
ance quadrature pass accumulates the per-Gaussian radi-
ance quadrature, with both passes utilizing additive frame-
buffers. Finally, a per-pixel normalization pass rescales the
observed radiance to ensure correct scene opacity.
Confidence Interval-based Rasterization
Rasterization
of Gaussians requires a screen-space proxy, like a quad,
whose shape is derived from projecting the 3D covariance
matrix. In EWA-based splatting [44] and 3DGS [18], this
projection uses a locally-affine approximation of the per-
spective transform to map the 3D covariance to its 2D
screen-space counterpart. This shared method cleanly de-
couples geometry from appearance: the covariance matrix
exclusively determines the splat’s screen-space footprint,
independent of learned scalar parameters (like peak density
or opacity) that modulate its final intensity. This screen-
space proxy, however, is unsuitable for our volumetric ren-
dering approach. The EWA proxy fails to cover all screen
areas where the particle has meaningful radiance contribu-
tions (see Fig. 2). This under-coverage becomes especially
pronounced as particles approach the camera or if their co-
variance is ill-conditioned. We therefore derive a new geo-
metric proxy for a perspective camera that covers all radi-
ance contributions within a confidence threshold.
The required geometric proxy area is dictated by the par-
ticle’s isolated opacity. Along a ray, this opacity integral
simplifies for a single Gaussian to a closed-form solution:
Z ∞
tn
Ti(tn →t) σi(t) dt = 1 −e−¯τi
(16)
with optical depth ¯τi evaluated using Eq. (10).
The 1D
5

<!-- page 6 -->
Gaussian parameters (µi, Σi, ωi) describe the particle’s
density distribution along the ray and are functions of
that ray’s origin and direction.
By extension, the op-
tical depth ¯τi and the particle’s opacity are also func-
tions of the ray direction.
Under a perspective camera
model, a pixel phom = (u, v, 1)T maps to a ray direction
dp = normalize(K−1phom) via the intrinsic matrix K.
Opacity, being a function of ray direction, can therefore be
interpreted as a function of pixel position.
The
new
geometric
proxy
encloses
the
implicit
screen-space curve defined by the confidence interval
c = 1 −e−¯τi. Assuming the Gaussian is at a reasonable
distance from the camera (i.e., µi > tn −4
√
2Σi), the full
level set equation remains complex due to the interdepen-
dence of ωi and Σi on the ray direction d.
To obtain
a tractable solution, we replace Σi with an upper bound,
which produces a valid, larger geometric proxy. This sim-
plification isolates the level set of ωi, allowing it to be ex-
pressed in a quadratic form pT
hom W phom = 0. The 3 × 3
symmetric matrix W is:
W = (mmT −κM)
(17)
with m = (K−1)T Σ−1 µ , M = (K−1)T Σ−1 K−1 and
κ = 2(log(C) −log(w)) + µT Σ−1µ.
(18)
with C =

−log(1 −c)
√
uT Σ−1u

/
 √
2π ∥u∥2

with
u = normalize(µ −o). This quadratic form yields an el-
lipse. We partition W into a 2 × 2 block W2×2, a vector
w2×1, and a scalar w33 to derive the standard statistical rep-
resentation for a point p = (u, v)T on the ellipse:
(p −µ2d)T Σ−1
2d (p −µ2d) = 1
(19)
where the 2D mean µ2d is:
µ2d = −W −1
2×2 w2×1
(20)
and the 2D quadrature matrix Σ2d is:
Σ2d = (µT
2d W2×2 µ2d −w33) W −1
2×2
(21)
This geometric proxy is then rasterized as an aligned rect-
angle following [44] however without additional scaling of
the semi-major and semi-minor axis of the ellipse since
the eigenvalues of Σ2d already capture all scaling effects.
Please refer to Appendix D for the detailed derivation.
Adjoint Rendering
Since hardware-accelerated rasteri-
zation is not fully differentiable, we require a custom ad-
joint rendering method. Our forward pass consists of three
stages: a Moment pass to compute a per-pixel moment tex-
ture m from Gaussian parameters Θ, a Quadrature pass to
compute radiance L and penalty P, and a rescaling pass
(a) Ground Truth
(b) Power Moments N = 3
(c) Trig. Moments N = 3
(d) Trig. Moments N = 5
Figure 4. Visual comparison of Power Moments and Trigono-
metric Moments (using N = 3 and N = 5 intervals) against the
ground truth on synthetic data.
that uses the first moment m0 to produce the final radiance
ˆL. A naive backward pass inverting these stages is inef-
ficient. It requires two separate reduction operations and
numerous intermediate adjoint framebuffers, including one
for the opacity rescaling derivative ∂L/∂ˆL · ∂ˆL/∂m0.
We introduce an optimized backward pass that resolves
these inefficiencies.
We first observe that the derivative
from the Rescaling pass, ∂ˆL/∂m0, can be re-evaluated and
folded into the other backward stages, eliminating the need
for three additive framebuffers. We then consolidate all gra-
dient computations into a single, efficient reduction. This
optimized pass, begins with an Adjoint Moment stage that
computes a per-pixel adjoint moment texture δm. A subse-
quent Gradient stage re-rasterizes all Gaussians, using δm
and the upstream gradients ∂L/∂L and ∂L/∂P to perform
a single reduction over all covered pixels (u, v), yielding
the final gradient ∇Θi:
∇Θi =
X
u,v
∂L
∂Li
∂Li
∂Θi
+ ∂L
∂Pi
∂Pi
∂Θi
+ ∂L
∂mi
∂mi
Θi
(22)
Please refer to Appendix D for a more detailed discussion.
3.4. Training and Densification
We optimize our model using an objective function that ex-
tends the 3DGS losses with a novel consistency regularizer,
Lconsistency, to mitigate overfitting. The moment-based trans-
mittance reconstruction can be inaccurate for complex den-
sities, and our regularizer enforces consistency between the
predicted transmittance and the analytical density of indi-
vidual Gaussian particles. The full objective is
L = (1 −α) L + α LD-SSIM + λ Lconsistency.
(23)
with α
=
0.2 and λ
=
0.1.
The regularization
term
is
derived
from
the
physical
constraint
that
the
optical
depth
over
any
ray
interval
[tj, tj+1],
given
by
τ(tn →tj+1) −τ(tn →tj),
must
be
6

<!-- page 7 -->
(a) Ground Truth
(b) Vol3DGS [36]
(c) EVER [22]
(d) Ours
Figure 5. Qualitative comparison of our method against two SOTA methods on scenes from Tanks and Temples [20] and Mipnerf-360 [4].
greater
than
or
equal
to
the
analytical
optical
depth τij of any single particle i within that inter-
val.
We penalize violations of this condition using
Pi = P
j max(0, τij −(τθ(tn →tj+1) −τθ(tn →tj)))2.
This penalty, summed over all visible particles, implic-
itly enforces the physical monotonicity of the learned
transmittance Tθ.
We further adapt the 3DGS Adaptive Density Con-
trol (ADC) to our density-based medium, where opac-
ity is view-dependent.
Standard pruning fails as it re-
lies on view-independent opacity.
We introduce a ro-
bust, view-independent metric for pruning based on a
particle’s opacity when viewed along its shortest axis:
oi = 1 −e−
√
2πwi min(sx,sy,sz).
This criterion effectively
prunes particles while preventing the creation of thin, view-
dependent particles that cause overfitting. We also invert
this equation to initialize particle peak densities winit from
the input point cloud.
Finally, we modify the cloning and splitting operations
to preserve density integrity. When cloning a particle, we
correct the introduced density bias by halving its peak den-
sity, wi ←1
2wi. We replace the stochastic splitting mech-
anism with a deterministic operation that splits a particle
along its longest eigenvector. The new means are offset
by µnew = µ ± δ dsplit and the corresponding scale is re-
duced by Σnew = γ Σsplit. We derived the optimal param-
eters (γ ≈0.639, δ ≈0.613 · Σsplit) by numerically mini-
mizing the change in the particle’s opacity contribution, sig-
nificantly improving optimization stability. Please refer to
Appendix E for a detailed derivations and discussions.
4. Evaluation
Implementation.
We use PyTorch [28] and the 3DGS
codebase [18] for training. Our rasterizer is implemented
via the Vulkan API, using Slang and its Slang-D extension
[1] for automatic differentiation.
Results.
We evaluate our method on three established
novel view synthesis benchmarks: The nine scenes from
Mip-NeRF 360 [4], train and truck scenes from Tanks and
Temples [20] as well as the drjohnson and playroom scenes
from DeepBlending [14]. Following previous literature, we
report PSNR, SSIM, and LPIPS computed on held-out tar-
get views. For a fair comparison, all methods are trained on
the same input views and evaluated on the same test splits,
using each method’s provided codebase and hyperparame-
ters (unless otherwise noted). We compare against the stan-
dard Gaussian splatting baseline (3DGS) [18], StopThePop
[33], EVER [22], Vol3DGS [36], and Don’t Splat Your
Gaussians [10].
The results are summarized in Tab. 1. The values sug-
gest competitive performance compared to volumetric ex-
tensions, whereas standard 3DGS and StopThePop often re-
main ahead in the evaluated quality metrics. Nevertheless,
Fig. 5 shows that our approach can resolve complex light
interactions more robustly than previous volumetric-aware
approaches. Semi-transparent effects like the reflection of
7

<!-- page 8 -->
Table 1. Quantitative comparison of Gaussian Splatting (GS) methods on three datasets. Higher is better for PSNR/SSIM (↑) and lower
is better for LPIPS (↓). We used the publically available code to reproduce the results, where possible. Results with dagger (†) are taken
from the respective publication instead.
Method
MipNeRF-360
Tanks & Temples
DeepBlending
PSNR ↑
SSIM ↑
LPIPS ↓
# Points
PSNR ↑
SSIM ↑
LPIPS ↓
# Points
PSNR ↑
SSIM ↑
LPIPS ↓
# Points
3DGS [18]
27.43
0.813
0.218
3.36×106
23.72
0.846
0.178
1.78×106
29.46
0.900
0.247
2.98×106
StopThePop [33]
27.31
0.814
0.213
3.29×106
23.16
0.843
0.173
1.81×106
29.92
0.905
0.234
2.81×106
Vol3DGS [36]
27.44
0.820
0.201
3.00×106
23.67
0.851
0.174
1.06×106
29.61
0.905
0.242
3.60×106
EVER [22]
25.60
0.772
0.299
3.89×106
22.59
0.842
0.199
6.38×106
28.12
0.891
0.353
2.54×106
Don’t Splat† [10]
27.32
0.793
–
–
22.09
0.797
–
–
28.06
0.878
–
–
Ours
25.96
0.760
0.245
2.12×106
22.18
0.825
0.194
1.37×106
29.14
0.900
0.248
2.69×106
the building on the windshield (row 1) and specular high-
lights on the metal bowl (row 3) are reconstructed with less
noise and higher sharpness. Volumetric-like regions such as
distant trees (row 2) exhibit similar improvements.
To illustrate how our approach handles the challenging
scenario of intersecting Gaussians, we conducted a small
experiment on a synthetic scene consisting of six Gaus-
sians that intersect and produce non-trivial volumetric color
blending effects. For a fair comparison of different render-
ers, we converted the scene to the Gaussian representation
of the respective methods and optimized diffuse color, opac-
ity and scale for 1000 iterations to allow the methods to
best fit the data. Fig. 3 shows a comparison to a ground-
truth volumetric renderer (a). Non-volumetric techniques
like StopThePop [33] are unable to model the intersection
of the Gaussians faithfully, as the naive rasterization in-
evitably has to blend the splats in order along the z-axis
(b). Volumetric-aware extensions like Vol3DGS [36] reduce
sharp boundary artifacts, but still suffer from unrealistic col-
ors (c). In contrast, our method significantly improves the
resulting color blending (d).
Ablation.
A synthetic comparison (see Fig. 4) analyzes
the choice of moment functions and the number of quadra-
ture intervals (N).
Power moments paired with our
quadrature tend to overestimate splat visibility in certain
views. Trigonometric moments prove more robust: N = 3
achieves correct splat ordering but inaccurate sizing, while
N = 5 closely matches the ground truth.
Real-world ablations (see Tab. 2) disable individual com-
ponents. Removing regularization slightly degrades image
metrics and increases runtime, attributed to a minor increase
in outliers. Using the EWA geometric proxy, rather than our
method, also slightly decreases aggregate metrics and in-
creases runtime without significantly altering particle count
due to overestimating small, translucent splats. Finally, re-
verting to the default ADC degrades metrics despite a sub-
stantial increase in total particle count and runtime, an effect
we attribute to a high outlier count.
Table 2. Component analysis on the Tanks and Temples Truck [20]
scene. We report results for ablations of each component.
PSNR ↑
Time [h] ↓
# Points
w/o Reg.
24.14
3.45
1.08×106
EWA Geom.
24.10
6.20
1.59×106
Default ADC
23.50
12.88
3.82×106
Full Model
24.25
2.83
1.06×106
Limitations
Although our approach improves volumet-
ric consistency over pure splatting, it still inherits limita-
tions from its underlying densification and camera assump-
tions. On challenging scenes with fine, highly parallaxed
structures such as flowers in Mip-NeRF 360, we observe
under-reconstruction and residual blur, which we attribute
to conservatively tuned adaptive density control (ADC).
As in other volumetric variants of 3DGS, our quality re-
mains tightly coupled to these heuristic splitting and prun-
ing thresholds, suggesting that more advanced densification
strategies could further improve performance. Moreover,
our physically motivated density field makes the method
more susceptible to calibration errors. Similar to Vol3DGS,
which reports opaque artifacts on miscalibrated scenes like
treehill in Mip-NeRF 360, our model tends to explain pose
and distortion inconsistencies by localized overfitting, lead-
ing to below-average metrics compared to opacity-centric
splatting baselines. We therefore see improving ADC for
our volumetric formulation and increasing robustness to im-
perfect camera calibration as complementary directions for
future work.
5. Conclusion
We presented MB3DGS, a moment-based formulation for
physically accurate, order-independent rendering of 3D
Gaussian representations. By modeling the combined den-
sity of overlapping Gaussians and reconstructing a continu-
ous transmittance function from analytically computed mo-
ments, our approach overcomes the inherent limitations of
alpha-blended splatting. The proposed power-transformed
8

<!-- page 9 -->
moment recurrence and confidence-interval–based raster-
ization enable stable, efficient rendering while preserv-
ing the performance benefits of modern GPU rasterization
pipelines. Our results demonstrate significantly improved
reconstruction fidelity, particularly in complex translucent
and highly detailed regions where traditional splatting fails.
Overall, MB3DGS bridges the gap between rasterization
and physically grounded volumetric rendering, providing a
practical path toward accurate and real-time Gaussian-based
scene representations.
6. Acknowledgments
This work has been funded by the Federal Ministry of
Research, Technology and Space of Germany and the state
of North Rhine-Westphalia as part of the Lamarr Institute
for Machine Learning and Artificial Intelligence, by the Eu-
ropean Regional Development Fund and the state of North
Rhine-Westphalia under grant number EFRE-20801085
(Gen-AIvatar), by the state of North Rhine-Westphalia
as part of the Excellency Start-up Center.NRW (U-
BO-GROW) under grant number 03ESCNW18B, and
additionally by the Ministry of Culture and Science North
Rhine-Westphalia under grant number PB22-063A (InVir-
tuo 4.0: Experimental Research in Virtual Environments).
References
[1] Sai Praveen Bangaru, Lifan Wu, Tzu-Mao Li, Jacob
Munkberg, Gilbert Bernstein, Jonathan Ragan-Kelley, Fredo
Durand, Aaron Lefohn, and Yong He. Slang. d: Fast, mod-
ular and differentiable shader programming. ACM Transac-
tions on Graphics (TOG), 42(6), 2023. 7
[2] Jonathan T Barron.
A power transform.
arXiv preprint
arXiv:2502.10647, 2025. 4
[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In IEEE/CVF International Conference
on Computer Vision (ICCV), pages 5855–5864, 2021. 1
[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. In IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2022.
2, 4, 7
[5] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-
based neural radiance fields.
In IEEE/CVF International
Conference on Computer Vision (ICCV), 2023. 1, 2, 4
[6] Louis Bavoil and Kevin Myers. Order independent trans-
parency with dual depth peeling. NVIDIA OpenGL SDK, 1
(12), 2008. 2
[7] Hugo Blanc, Jean-Emmanuel Deschaud, and Alexis Paljic.
Raygauss: Volumetric gaussian-based ray casting for pho-
torealistic novel view synthesis. In IEEE/CVF Winter Con-
ference on Applications of Computer Vision (WACV). IEEE,
2025. 2
[8] Loren Carpenter. The a-buffer, an antialiased hidden surface
method. In Annual Conference on Computer Graphics and
Interactive Techniques (SIGGRAPH), 1984. 2
[9] Adam Celarek, George Kopanas, George Drettakis, Michael
Wimmer, and Bernhard Kerbl. Does 3d gaussian splatting
need accurate volumetric rendering? In Computer Graphics
Forum (CGF). Wiley Online Library, 2025. 2, 20
[10] Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic,
Simon Green, Piotr Didyk, and Adrian Jarabo. Don’t splat
your gaussians: Volumetric ray-traced primitives for mod-
eling and rendering scattering and emissive media.
ACM
Transactions on Graphics (TOG), 44(1), 2025. 1, 2, 7, 8
[11] Cass Everitt.
Interactive order-independent transparency.
White paper, NVIDIA, 2(6), 2001. 2
[12] Izrail Solomonovich Gradshteyn and Iosif Moiseevich
Ryzhik. Table of integrals, series, and products. Academic
press, 2014. 13
[13] Florian Hahlbohm, Fabian Friederichs, Tim Weyrich, Li-
nus Franke, Moritz Kappel, Susana Castillo, Marc Stam-
minger, Martin Eisemann, and Marcus Magnor.
Efficient
perspective-correct 3d gaussian splatting using hybrid trans-
parency. In Computer Graphics Forum (CGF). Wiley Online
Library, 2025. 2
[14] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
George Drettakis, and Gabriel Brostow. Deep blending for
free-viewpoint image-based rendering. ACM Transactions
on Graphics (TOG), 37(6), 2018. 7
[15] Roger A Horn and Charles R Johnson.
Matrix analysis.
Cambridge university press, 2012. 18
[16] Qiqi Hou, Randall Rauwendaal, Zifeng Li, Hoang Le, Farzad
Farhadzadeh, Fatih Porikli, Alexei Bourd, and Amir Said.
Sort-free gaussian splatting via weighted sum rendering.
In International Conference on Learning Representations
(ICLR), 2025. 2
[17] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In ACM SIGGRAPH Conference Papers,
2024. 2
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering.
ACM Transactions on Graphics
(TOG), 42(4), 2023. 1, 2, 3, 5, 7, 8, 20
[19] Shakiba Kheradmand,
Delio Vicini,
George Kopanas,
Dmitry Lagun, Kwang Moo Yi, Mark Matthews, and An-
drea Tagliasacchi. Stochasticsplats: Stochastic rasterization
for sorting-free 3d gaussian splatting. In IEEE/CVF Interna-
tional Conference on Computer Vision (ICCV), 2025. 2
[20] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and temples: Benchmarking large-scale scene
reconstruction. ACM Transactions on Graphics (TOG), 36
(4), 2017. 7, 8
[21] Zhihao Liang, Qi Zhang, Wenbo Hu, Lei Zhu, Ying Feng,
and Kui Jia.
Analytic-splatting: Anti-aliased 3d gaussian
splatting via analytic integration. In European Conference
on Computer Vision (ECCV). Springer, 2024. 2
[22] Alexander Mai, Peter Hedman, George Kopanas, Dor
Verbin, David Futschik, Qiangeng Xu, Falko Kuester,
9

<!-- page 10 -->
Jonathan T Barron, and Yinda Zhang.
Ever: Exact volu-
metric ellipsoid rendering for real-time view synthesis. In
IEEE/CVF International Conference on Computer Vision
(ICCV), 2025. 2, 7, 8
[23] Morgan McGuire and Louis Bavoil. Weighted blended order-
independent transparency.
Journal of Computer Graphics
Techniques (JCGT), 2(4), 2013. 2, 12
[24] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view
synthesis.
In European Conference on Computer Vision
(ECCV), 2020. 1, 2
[25] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Ric-
cardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja
Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray trac-
ing: Fast tracing of particle scenes. ACM Transactions on
Graphics (TOG), 43(6), 2024. 1, 2, 3, 11
[26] Cedrick M¨unstermann, Stefan Krumpen, Reinhard Klein,
and Christoph Peters.
Moment-based order-independent
transparency. ACM on Computer Graphics and Interactive
Techniques, 1(1), 2018. 2, 4, 12, 13, 16, 17
[27] Thomas Neff, Pascal Stadlbauer, Mathias Parger, Andreas
Kurz, Joerg H Mueller, Chakravarty R Alla Chaitanya, An-
ton Kaplanyan, and Markus Steinberger. Donerf: Towards
real-time rendering of compact neural radiance fields using
depth oracle networks. In Computer Graphics Forum (CGF).
Wiley Online Library, 2021. 4
[28] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An im-
perative style, high-performance deep learning library. Ad-
vances in Neural Information Processing Systems (NeurIPS),
32, 2019. 7
[29] Christoph Peters. Non-linearly quantized moment shadow
maps. In High Performance Graphics (HPG), 2017. 5
[30] Christoph Peters and Reinhard Klein. Moment shadow map-
ping.
In ACM SIGGRAPH Symposium on Interactive 3D
Graphics and Games (I3D), 2015.
[31] Christoph Peters, Cedrick Munstermann, Nico Wetzstein,
and Reinhard Klein. Beyond hard shadows: Moment shadow
maps for single scattering, soft shadows and translucent oc-
cluders. In ACM SIGGRAPH Symposium on Interactive 3D
Graphics and Games (I3D), 2016.
[32] Christoph Peters, Cedrick M¨unstermann, Nico Wetzstein,
and Reinhard Klein. Improved moment shadow maps for
translucent occluders, soft shadows and single scattering.
Journal of Computer Graphics Techniques (JCGT), 6(1),
2017. 5
[33] Lukas Radl, Michael Steiner, Mathias Parger, Alexan-
der Weinrauch, Bernhard Kerbl, and Markus Steinberger.
Stopthepop: Sorted gaussian splatting for view-consistent
real-time rendering. ACM Transactions on Graphics (TOG),
43(4), 2024. 1, 2, 5, 7, 8, 11
[34] Samuel Rota Bul`o, Lorenzo Porzi, and Peter Kontschieder.
Revising densification in gaussian splatting.
In European
Conference on Computer Vision (ECCV). Springer, 2024. 21
[35] Xin Sun, Iliyan Georgiev, Yun Fei, and Miloˇs Haˇsan.
Stochastic ray tracing of 3d transparent gaussians.
arXiv
preprint arXiv:2504.06598, 2025. 2
[36] Chinmay Talegaonkar, Yash Belhe, Ravi Ramamoorthi, and
Nicholas Antipa. Volumetrically consistent 3d gaussian ras-
terization. In IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2025. 1, 2, 5, 7, 8, 11, 21
[37] ´Arp´ad Tari, Mikl´os Telek, and Peter Buchholz.
A uni-
fied approach to the moments based distribution estimation–
unbounded support. In European Workshop on Performance
Engineering, pages 79–93. Springer, 2005. 4
[38] Grigoris Tsopouridis, Andreas A Vasilakis, and Ioannis Fu-
dos.
Deep and fast approximate order independent trans-
parency. In Computer Graphics Forum (CGF). Wiley Online
Library, 2024. 2
[39] Markus Worchel and Marc Alexa. Moment bounds are dif-
ferentiable: Efficiently approximating measures in inverse
rendering.
ACM Transactions on Graphics (TOG), 44(4),
2025. 4
[40] Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas
Moenne-Loccoz, and Zan Gojcic.
3dgut: Enabling dis-
torted cameras and secondary rays in gaussian splatting.
In IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2025. 2
[41] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong
Dou. Absgs: Recovering fine details in 3d gaussian splatting.
In ACM International Conference on Multimedia (MM),
2024. 21
[42] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), 2024. 2
[43] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Transactions on Graphics (TOG),
43(6), 2024. 2, 11
[44] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and
Markus Gross.
Ewa volume splatting.
In Visualization.
IEEE, 2001. 5, 6
10

<!-- page 11 -->
A. Ray Parameterized Density
Central to our approach is the re-parameterization of the
density as a sum 1D Gaussians along a ray, parameterized
by distance (Eq. 4-6). This section provides details about
the derivation of this result and contextualizes it with re-
spect to prior work.
Let a ray be defined as r(t) = o + t d, parameterized by
distance t from origin o in direction d. The density at any
position x is a Gaussian mixture:
σ(x) =
X
i
wi G(x | µi, Σi)
where wi, µi, and Σi are the weight, mean, and covariance
of the i-th component.
We first analyze the density of a single component along
the ray. Substituting r(t) gives:
σi(r(t)) = wi e−1
2 (o+td−µi)T Σ−1
i
(o+td−µi).
As the following derivation applies independently to each
component, we now drop the index i from wi, µi, Σi for
notational simplicity.
Let v = o −µ, substitute it in the exponent and expand
it into its quadratic form:
(v + t · d)T Σ−1(v + t · d)
= vT Σ−1v + t vT Σ−1d + t dT Σ−1v + t2 dT Σ−1v
Since Σ−1 is symmetric, vT Σ−1d = dT Σ−1v. So, the
cross terms combine to 2t dT Σ−1v. The exponent is thus
−1
2
 vT Σ−1v + 2t dT Σ−1v + t2 dT Σ−1d

This is quadratic in t: A t2 + B t + C, where
A = −1
2dT Σ−1d, B = −dT Σ−1v, C = −1
2vT Σ−1v.
So, the i-th particle density as a function of distance t is
σ(t) = w eA t2+B t+C.
We want to rewrite this into the
standard form of a 1D Gaussian
ω e−(t−µ)2
2Σ2
Recall the quadratic expansion of the 1D Gaussian expo-
nent:
−(t −µ)2
2Σ2
= −1
2Σ2 t2 + µ
Σ2 t −(µ)2
2Σ2
We derive the 1D parameters by matching the exponent
At2 + Bt + C to the target 1D Gaussian form. Compar-
ing the t2 coefficient yields the variance Σ2:
A = −1
2Σ2 ⇒Σ2 = −1
2A =
1
dT Σ−1d
Comparing the t coefficient yields the mean µ:
B = −µ
2Σ2 ⇒µ = B Σ2 = −dT Σ−1v
dT Σ−1d
which leaves C = −µ2/(2Σ2) as the coefficient that is in-
dependent of t.
The 1D Gaussian weight ω is determined by the peak
amplitude, which we find by completing the square on the
exponent:
At2 + Bt + C = A

t + B
2A
2
+

C −B2
4A

.
The term independent of t, K = C −B2/(4A), is the value
of the exponent at its peak where t = µ. Substituting the
known expressions for A, B, and C gives:
K = −1
2vT Σ−1v + 1
2
(dT Σ−1v)2
dT Σ−1d .
The density as a function of ray distance t thus simplifies
to w eK e−(t−µ)2/(2Σ2). Finally, the 1D Gaussian weight is
ω = w eK.
Recall the covariance definition Σ = RSST RT , where
R is a rotation matrix and S is a diagonal scale matrix. To
avoid explicit inversion of Σ, we follow [25] by defining
transformed vectors:
og = S−1RT (o −µ)
and
dg = S−1RT d
This
simplifies
the
above
quadratic
terms,
yield-
ing
vT Σ−1v = oT
g og,
dT Σ−1d = dT
g dg,
and
dT Σ−1v = dT
g og.
Substituting these into the expres-
sions for µ, Σ, and K directly yields Equations 4-6 from
the main paper.
This 1D re-parameterization builds on related concepts
in prior work.
StopThePop [33] and 3D Gaussian Ray-
tracing [25] use the mean of the 1D Gaussian (the point
of highest amplitude) to define a single distance from the
camera to the particle. Other opacity-based methods, such
as Vol3DGS [36] and Gaussian Opacity Fields [43], also
derive a similar 1D parameterization.
Gaussian Opacity
Fields [43] further modifies this 1D Gaussian to mimic a
threshold function, which is then used to derive a single
particle’s transmittance. Crucially, all these methods use
alpha-blending for rendering. This approach does not prop-
erly resolve intersections between Gaussian particles, a lim-
itation addressed by our volumetric formulation.
B. Quadrature
This section derives the quadrature for the volume render-
ing equation (Eq. 7, main paper), details the opacity rescal-
ing (Eq. 15), and describes the interval spacing for the Eq.
7 quadrature. While the main paper uses compact notation
11

<!-- page 12 -->
T(t) for transmittance over the interval [tn, t], this section
uses the more explicit notation T(a →b) to denote trans-
mittance over the interval [a, b].
We substitute the definitions for the medium’s density
σ(x) (Eq. 1) and emitted radiance Le(x, d) (Eq. 3) into the
integrand of the volume rendering equation (Eq. 2). This
substitution simplifies the σ(x)Le(x, d) term, as the total
density σ(x) cancels out, leaving a sum of radiance contri-
butions from each particle:
σ(xt) Le(xt, d) =
n
X
i=1
σi(xt) (Le)i(d)
Substituting this result back into Eq. 2, the linearity of inte-
gration permits exchanging the summation and the integral:
L =
n
X
i=1
Z tf
tn
T(tn →t) σi(xt) (Le)i(d) dt
|
{z
}
Li
Discretize the interval [tn, tf] into N small, contiguous seg-
ments where the j-th segment spans from tj to tj+1 with
length ∆j = tj+1 −tj. The integral for Li becomes a sum
over these segments:
Li =
N
X
j=1
Z tj+1
tj
T(tn →t) σi(xt) (Le)i(d) dt.
The transmittance T(tn →t) can be split into the product
T(tn →tj) T(tj →t). The j-th segment of the integral for
Li becomes
Li,j =
Z tj+1
tj
T(tn →tj) T(tj →t) σi(xt) (Le)i(d) dt
Within each small segment [tj, tj+1], we assume the den-
sity σ(xt) is piecewise constant and equal to its value at
the start of the interval, σ(xtj). Since each Gaussian has an
infinite support, this assumption about the total density also
implies that each particles local density is piecewise con-
stant. Therefore terms T(tn →tj), σi(xtj), and (Le)i(d)
are constant within this interval and can be pulled out:
Li,j ≈T(tn →tj) σi(xtj) (Le)i(d)
Z tj+1
tj
T(tj →t) dt
The remaining integral under this assumption evaluates as:
Z tj+1
tj
T(tj →t) dt =
Z tj+1
tj
e
−
R t
tj σ(xs) ds dt
=
Z tj+1
tj
e−σ(xtj )(t−tj) dt = 1 −e−σ(xtj )∆j
σ(xtj)
The term 1 −e−σ(xtj )δj can be expressed using the trans-
mittance over the segment T(tj →tj+1) since with con-
stant density it is T(tj →tj+1) = e−σ(xtj )δj. Substituting
this back, the contribution of segment j to particle i is:
Li,j ≈T(tn →tj) σi(xtj) (Le)i(d)
1 −T(tj →tj+1)
σ(xtj)

Using the multiplicative property of the transmittance again,
the visibility can be evaluated using transmittance values
from the ray’s near bound to the interval’s left and right
edge:
T(tn →tj) (1 −T(tj →tj+1))
= T(tn →tj) −T(tn →tj+1).
Substituting this back and summing the contributions Li,j
over all segments yields the quadrature presented in Eq. 7:
Li ≈
N
X
j=1
(T(tn →tj) −T(tn →tj+1)) σi(tj)
σ(tj) (Le)i(d).
A direct evaluation of σ(tj) is not possible in a rasteriza-
tion setting. However, under the assumption of piecewise-
constant density, we can solve T(tj →tj+1) = e−σ(xtj )∆j
for σ(tj) given the transmittance over the interval
T(tj →tj+1). The multiplicative property of the transmit-
tance also allows this expression to be written in terms of
transmittance starting from the ray’s near bound. We there-
fore evaluate the total density within j-th interval to be
σ(tj) ≈−1
∆j
log
T(tn →tj+1)
T(tn →tj)

.
Recall
that
we
estimate
the
optical
depth
τ(tn →t) =
R t
tn σ(xs) ds at distance t from the den-
sity moments m0, . . . , mk. The transmittance under this
estimation of the optical depth is T(tn →t) = e−τ(tn→t).
Substituting this relationship into the estimate of the total
transmittance gives the numerically more stable expression
σ(tj) = 1
∆j
(τ(tn →tj+1) −τ(tn →tj))
Opacity Rescaling
Previous work on order-independent
transparency [23, 26] observed that an under- or overesti-
mation of the transmittance causes the volumetric scene’s
overall opacity to fluctuate across the image. To stabilize
the visual appearance, they rescale the radiance using the
ratio of the true scene opacity 1 −T(tn →tf), to the esti-
mated scene opacity, O:
1 −T(tn →tf)
O
12

<!-- page 13 -->
The true transmittance T(tn →tf) is recovered from the
zeroth moment as e−m0. The estimated opacitay O is the
sum of particle opacities, O = P
i Oi.
Following prior work, we apply the opacity ratio to
rescale the observed radiance. The final radiance prediction
in Eq. 15 of the main paper is:
ˆL =
1 −e−m0
max(ϵ, Pn
i=1 Oi)
n
X
i=1
Li + e−m0 Lbg
where the denominator is clamped to ϵ > 0 to prevent divi-
sion by zero. Each Oi is computed concurrently with Li in
single render pass by accumulating a homogeneous ”color”
(i.e. appending 1 to the RGB components). Under the as-
sumption of a piecewise-constant density, this opacity for
the i-th particle is given by the quadrature
Oi =
N
X
j=1
(T(tn →tj) −T(tn →tj+1)) σi(xtj)
σ(tj) .
Sample Spacing
To numerically evaluate the integral Li
for each particle, we require an efficient sampling strategy.
We use inverse transform sampling to concentrate samples
{tj}N
j=1 in the particle’s high-density region along the ray.
First, a set of standard normal samples {xj} is pre-
computed once by applying the inverse normal CDF, Φ−1,
to uniformly spaced samples on the interval [0, 1]. In the
shader, these are then transform using
tj = µi + κ Σi xj
where κ ≥1 is a scaling parameter (κ = 3 is used in all
experiments). For κ = 1, this corresponds to importance
sampling the particle’s density σi. Increasing κ broadens
the sampling distribution, transitioning it towards a more
uniform distribution.
Finally, the samples {tj} are sampled to the integration
interval [tn, tf]. This is necessary to respect the integra-
tion bounds, as particle density outside this interval does not
contribute to Li. For partially visible Gaussians (i.e. those
truncated by the bounds), this clamping clusters samples at
the boundaries tn or tf, effectively truncating the sampling
distribution to the visible segment.
C. Analytical Density Moments
This section proves the lower-bound on the density power-
moment, which justifies integrating ˆg(t)k against the den-
sity instead of tk. We also derive the recurrence relation
used to evaluate the power moments (Eqs. 10-12 in the main
paper) and the closed-form expression for the trigonometric
moments (Eq. 14 in the main paper). Additionally, this sec-
tion provides further details on the Taylor approximation
used to evaluate erf for complex arguments and compares
our density moments to those described by M¨unstermann et
al. [26]. All derivations utilize the density parameterization
with respect to the ray distance presented in Eq. 4 of the
main paper.
Lower bounds on Power Moments
If tf ≥µi +
√
2Σi
and tn ≤µi (holds e.g. for tn = 0, tf = ∞and any µi ≥0
but also justifiable in practical settings due to clipping and
tf = 106), choose interval I = [µi, µi +
√
2Σi] ⊂[tn, tf].
If
√
2Σi > 0, then for t ∈I we have t > µi and by mono-
tonicity of tk ≤(µi)k. Therefore,
(mk)i ≥ωi
Z
I
tke
−(t−µi)2
2Σ2
i
dt ≥ωi µk
i
Z
I
e
−(t−µi)2
2Σ2
i
dt.
The remaining integral is just a (truncated) Gaussian mass
which we can express in terms of the error function. Change
variables u = (t −µi)/(2Σ2
i ). Then
(mk)i ≥ωi µk
i
√
2 Σi
Z 1
0
e−u2 du.
This lead to a lower-bound in the value of the k-th moment
(mk)i ≥c0 ωi µk
i Σi
with
c0 =
rπ
2 erf(1) ≈1.0562.
This lower-bound holds for most splats with a reasonable
distance from the camera. Therefore, the moments of most
splats to grow at least as fast as µk
i . The exponential in-
crease with k causes numerical instability when using the
moments as a medium descriptor.
Power Moments
Recall that the warping function ˆg maps
[tn, tf] to [0, 1], is differentiable and strictly monotone. To
avoid the exponential growth of the power moments with
increasing k, we define moments that integrate the warped
distance against the density:
ˆmk =
Z tf
tn
ˆg(t)k σ(t) dt.
By linearity, the k-th power moment can be written as the
sum of power moments for the individual particles:
ˆmk =
X
i
ωi
Z tf
tn
ˆg(t)k e
−(t−µi)2
2Σ2
i
dt.
The inner integral is solvable in elementary functions and
erf for many choices of ˆg but can be solved if ˆg is a polyno-
mial. [12] Our choice of warping function and most func-
tions that are suitable to map perceptually significant dis-
tances to well-resolved floating-point ranges do not permit
for a practical closed-form solution. However, each parti-
cles contribution to the moment is concentrated around its
13

<!-- page 14 -->
mean, we therefore use a first-order Taylor approximation
of the warped distance: ut = ˆg(µi) + ˆg′(µi) (t −µi) which
reduces the problem to a polynomial. Since each integral is
to be solved individually we center the linearization at µi.
The moments over the warped distance is therefore approx-
imated as
ˆmk ≈
X
i
ωi
Z tf
tn
uk
t e
−(t−µi)2
2Σ2
i
dt
|
{z
}
( ˆmk)i
We derive a recurrence relationship in order to evaluate
the inner integral.
Pulling out one factor of the linearized distance from the
exponent uk
t and substitute by is definition, gives
ωi
Z tf
tn
(ˆg(t) + ˆg′(µi) (t −µi)) uk−1
t
e
−(t−µi)2
2Σ2
i
dt.
Expand the integral into two parts:
( ˆmk)i = ˆg(µi) ωi
Z tf
tn
uk−1
t
e
−(t−µi)2
2Σ2
i
dt
+ ˆg′(µi) ωi
Z tf
tn
(t −µi) uk−1
t
e
−(t−µi)2
2Σ2
i
dt.
(24)
By definition, the first part of the expanded integral is
ˆg(µi)( ˆmk−1)i. The remaining integral, I, can be expressed
in term of the total derivative of the Gaussian since
∂
∂te
−(t−µi)2
2Σ2
i
= −t −µi
Σ2
i
e
−(t−µi)2
2Σ2
i
⇒(t −µi) e
−(t−µi)2
2Σ2
i
= −Σ2
i
∂
∂te
−(t−µi)2
2Σ2
i
.
Hence
I = −ˆg′(µi) ωi Σ2
i
Z tf
tn
uk−1
t
∂
∂t
 
e
−(t−µi)2
2Σ2
i
!
dt.
Use integration by parts with functions f(t) = uk−1
t
and
g(t) = e
−(t−µi)2
2Σ2
i
which implies f ′(t) = (k−1) g′(µi) uk−2
t
since u′
t = ˆg′(µi).
I = −ˆg′(µi) ωi Σ2
i
"
uk−1
t
e
−(t−µi)2
2Σ2
i
#tf
tn
+ ˆg′(µi)2 Σ2
i (k −1) ωi
Z tf
tn
uk−2
t
e
−(t−µi)2
2Σ2
i
dt
|
{z
}
(mk−2)i
.
Putting everything together, gives for k ≥2,
( ˆmk)i = ˆg(µi) ( ˆmk−1)i+ˆg′(µi)2 Σ2
i (k−1) ( ˆmk−2)i−Bi(k)
where Bi(k) is term to correct for density outside of the
integration boundary
Bi(k) = ˆg′(µi) ωi Σ2
i
"
uk−1
t
e
−(t−µi)2
2Σ2
i
#tf
tn
.
For the base case (m0)i, we have
( ˆm0)i = ωi
Z tf
tn
e
−(t−µi)2
2 Σ2
i
dt.
Applying change of variable with v = (t −µi)/(
√
2 Σi)
which implies t = µi +
√
2 Σi v, and dt =
√
2 Σi dv gives
( ˆm0)i = ωi
√
2 Σi
Z vf
vn
e−v2 dv
where the near bound is vn = (tn −µi)/(
√
2 Σi) and the
far bound vf = (tf −µi)/(
√
2 Σi). Writing the antideriva-
tive of the Gaussian function e−v2 in terms of the error func-
tion gives for any C
Z
e−v2 dv =
√π
2 erf(v) + C.
Thus, after simplfying constants
( ˆm0)i = ωiΣi
rπ
2 (erf(vf) −erf(vn)).
(25)
The base case ( ˆm1)i follows directly from Equation (24):
( ˆm1)i = ˆg(µi) ( ˆm0)i + ˆg′(µi) ωi
Z tf
tn
(t−µi) e
−(t−µi)2
2Σ2
i
dt
Substitute the total derivative in the remaining integral gives
Z tf
tn
(t −µi)e
−(t−µi)2
2Σ2
i
dt = −Σ2
i
Z tf
tn
∂
∂t e
−(t−µi)2
2Σ2
i
dt.
Thus by FTC,
( ˆm1)i = ˆg(µi) ( ˆm0)i −ˆg′(µi) ωi Σ2
i
"
e
−(t−µi)2
2Σ2
i
#tf
tn
.
Special case tf →∞: Since limt→∞erf(a + bt) = 1 for
fixed a, b > 0, we get the first recurrence basis presented in
Equation 10 of the main paper:
( ¯m0)i = ωiΣi
rπ
2

1 −erf
tn −µi
√
2 Σi

.
14

<!-- page 15 -->
For t ≥2µi. (t −µi)2 ≥(t/2)2 = t2/4. Hence
e
−(t−µi)2
2Σ2
i
≤e
−t2
8Σ2
i
(t ≥2µi).
Since e−ct2 →0 as tf →∞for any fixed c > 0, we get the
second recurrence basis present in Equation 11 of the main
paper:
( ¯m1)i = ˆg(µi) ( ¯m0)i + ˆg′(µi)ωiΣ2
i e
−(tn−µ)2
2Σ2
i
.
Since (a + b t)k e−c t2 →0 as t →∞for any fixed k and
a, b, c > 0 by repeated L’Hopital, it follows that
lim
t→∞(ˆg(µi) + ˆg′(µi)(t −µi))k−1e
−
(tf −µi)2
2Σ2
i
= 0.
The boundary term of the recurrence thus simplifies to the
boundary term referenced in Equation 12 in the main paper:
¯Bi(k) = −ˆg′(µi) ωi Σ2
i vk−1
n
e−(tn−µi)2
2Σ2
.
Trigonometric Moments
An alternative to the polyno-
mial basis for the moments is the Fourier basis for which
we derive the trigonometric moments for a Gaussian den-
sity function:
˜mk =
N
X
j=1
ωj
Z tf
tn
(e(2π−θ)iˆg(t))ke
−
(t−µj )2
2Σ2
j
dt
|
{z
}
=( ˜mk)j
Let α = k(2π −θ) be the angular frequency term. The in-
ner integral becomes:
( ˜mk)j = ωj
Z tf
tn
eiαˆg(t)e
−
(t−µj )2
2Σ2
j
dt
Substitute the Taylor expansion for ˆg(t) into the integral:
( ˜mk)j ≈ωj
Z tf
tn
eiα(ˆg(µj)+ˆg′(µj)(t−µj))e
−
(t−µj )2
2Σ2
j
dt
Split the exponential term: The term eiαˆg(µj) is a constant
with respect to t and can be moved outside the integral.
Let’s also define a new constant β = αˆg′(µj)
( ˜mk)j ≈ωj eiαˆg(µj)
Z tf
tn
eiβ(t−µj)e
−
(t−µj )2
2Σ2
j
dt
We now focus on solving the remaining integral, which
we’ll call J:
J =
Z tf
tn
eiβ(t−µj)e
−
(t−µj )2
2Σ2
j
dt
Perform a substitution: let u = t −µj which implies dif-
ferential du = dt.
The limits of integration change to
u ∈[tn −µj, tf −µj]:
J =
Z tf −µj
tn−µj
eiβue
−u2
2Σ2
j du =
Z tf −µj
tn−µj
e
−u2
2Σ2
j
+iβu
du.
The exponent is a quadratic in u.
−u2
2Σ2
j
+ iβu = −1
2Σ2
j

u2 −2Σ2
jiβu

We can solve this integral by completing the square for the
exponent with (iΣ2
jβ)2:
−u2
2Σ2
j
+ iβu = −1
2Σ2
j

(u −iΣ2
jβ)2 −(iΣ2
jβ)2
Simplifying terms using i2 = −1 gives the exponent
−u2
2Σ2
j
+ iβu = −(u −iΣ2
jβ)2
2Σ2
j
−Σ2
jβ2
2
.
Substituting this exponent back into the integral J:
J =
Z tf −µj
tn−µj
e
−
(u−iΣ2
j β)2
2Σ2
j
−
Σ2
j β2
2
du.
The term e−2Σ2
jβ2 is a constant and can be factored out:
J = e−
Σ2
j β2
2
Z tf −µj
tn−µj
e
−
(u−iΣ2
j β)2
2Σ2
j
du.
To match this form, we use another substitution:
Let
v = (u −iΣ2
jβ)/(
√
2Σj) such that v2 matches the ex-
ponent.
The differential is dv = (du)/(
√
2 Σj),
so
du =
√
2 Σj dv and the new limits of integration become:
vn = (tn −µj) −iΣ2
jβ
√
2Σj
and vf = (tf −µj) −iΣ2
jβ
√
2Σj
.
Substitute these into J:
J =
√
2 Σje−
Σ2
j β2
2
Z vf
vn
e−v2 dv
Now, apply the definition of the definite integral using the
error function:
Z vf
vn
e−v2 dv =
√π
2 erf(v)
vf
vn
Substitute this back into J and simplifying constants:
J =
rπ
2 Σje−
Σ2
j β2
2
(erf(vf) −erf(vn))
15

<!-- page 16 -->
Finally,
we
combine
all
the
parts.
Recall
that
(mk)j ≈ωjeiαˆg(µj) · J.
( ˜mk)j ≈ωjeiαˆg(µj)
rπ
2 Σje−
Σ2
j β2
2
(erf(vf) −erf(vn))

Combine the exponential terms:
( ˜mk)j ≈eiαˆg(µj)−
Σ2
j β2
2
ωjΣj
rπ
2 (erf(vf) −erf(vn))
Special case tf →∞: Recall the definition of vf which
has real part Re(vf) = (tf −µj)/(
√
2 Σj) and imaginary
part Im(vf) = −(Σj β)/
√
2. The error function erf(z) is
defined by the integral:
erf(z) =
2
√π
Z z
0
e−w2dw
The function e−w2 is analytic on the whole complex plane.
This means the integral is path-independent; the value de-
pends only on the endpoints. This allows us to split the
integral into two parts:
erf(xf + iy) =
2
√π
"Z xf
0
e−t2dt +
Z xf +iy
xf
e−w2dw
#
.
The first term is simply the definition of the real error func-
tion erf(xf). As xf →∞, this part converges to 1. Let’s
call the second term K. We parameterize the vertical path
by w = xf + is, where s goes from 0 to y. The differential
is dw = ids.
K =
2
√π
Z y
0
e−(xf +is)2(ids)
Expand the exponent and factor out the term e−x2
f , which
does not depend on s:
K = 2ie−x2
f
√π
Z y
0
es2e−2 i xf sds
Now, we take the limit as xf →∞. The term e−x2
f goes to
0. The integral
R y
0 es2e−2ixf sds remains bounded. Since y
is a finite constant, es2 is bounded on the interval [0, y]. The
term e−2ixf s is just an oscillation with magnitude 1. The in-
tegral of a bounded function over a finite interval is finite.
Because the limit is a term that vanishes times a bounded
term, the entire second term vanishes. Thus erf(vf) →1
as tf →∞. Therefore the particle’s k-th trigonometric mo-
ment presented in Equation 14 of the main paper is
( ˜mk)j ≈ωj
rπ
2 Σjeiαˆg(µj)−
Σ2
j β2
2
(1 −erf(vn)).
(a) Exact erf evaluation
(b) Linearized erf evaluation
Figure 6. Comparison of the trigonometric moment for a single
particle, plotted as a function of µi with fixed particle parameters
ωi, Σi. (a) The exact moment, computed using the complex erf
function. (b) Our first-order Taylor approximation. The error in-
troduced by the approximation is visibly localized close to zero.
Note that the remaining exponential term is a damped os-
cillator. Using euler’s formula, the oscillator term can be
evaluated as
e−
Σ2
j β2
2
(cos(α ˆg(µj)) + i sin(α ˆg(µj))).
For k ≥1, vn is a complex number, requiring the evaluation
of the error function erf with complex argument
(tn −µj) −iΣ2
jβ
√
2Σj
= t −µj
√
2Σj
| {z }
=a
−i Σjβ
√
2
|{z}
=b
Since this is non-trivial in a real-time setting, we approxi-
mate it with a first-order Taylor expansion around a in the
imaginary direction:
erf(a + ib) ≈erf(a) + ib
 2
√π e−a2
.
The first-order expansion is valid when the imaginary
component b = Σjβ/
√
2 is small.
Furthermore, the ap-
proximation error is inherently localized, as demonstrated
in Fig. 6. Since all higher-order terms of the Taylor series
include the factor e−a2, the approximation error vanishes
rapidly as |a| increases, ensuring high fidelity for t values
distant from the mean µj.
Comparison to MBOIT Moments
M¨unstermann et
al. [26] propose moments for rendering infinitesimally thin
transparent surfaces. Their power moments are defined as
mk =
X
j
−log(1 −αj) zk
and their trigonometric moments as
˜mk =
X
j
−log(1 −αj)e(2π−θ) i z+1
2
where z is the warped distance, αj is the surface opacity,
and i is the imaginary unit. These moments can be adapted
16

<!-- page 17 -->
(a) MBOIT power moments.
(b) Our power moments.
(c) MBOIT trigonometric moments.
(d) Our trigonometric moments.
Figure 7. Comparison of our Power and Trigonometric moments
with the MBOIT method as a function of µi. Our moments ex-
hibit a clear dampening effect near zero, which becomes more pro-
nounced for higher degrees, as shown by the k = 8 case (orange).
All moments are plotted for a single particle with fixed parameters-
(a) MBOIT power moments
(b) Our power moments
(c) MBOIT trigonometric moments
(d) Our trigonometric moments
(e) Ground truth
Figure 8. Visual comparison of MBOIT [26] Power and Trigono-
metric Moments to our Moments on synthetic data. All method
use N = 3 quadrature intervals.
to our domain by setting the particle opacity αj = 1−e−¯τj,
where ¯τj can be evaluated using Eq. (25) with modified
bounds. Fig. 7 shows a comparison to our moments. Our
moments exhibit a dampening effect that strengthens with
increasing k. This dampening is observable in both power
and trigonometric moments but is more pronounced for the
latter.
A comparison on synthetic data (Fig. 8) reveals negligi-
ble differences between MBOIT power moments and our
proposed power moments.
Both formulations underesti-
mate splat occlusion, resulting in excessive color blending
relative to the ground truth. While MBOIT trigonometric
moments achieve more accurate occlusion, they introduce
visual artifacts, particularly at the transition between the
green and red particles. Our trigonometric moments do not
exhibit this behavior.
D. Rasterisation and Adjoint Rendering
This section derives the geometric proxy for particle ras-
terization (Eqs. 17–21 in the main paper) and details the
proposed rasterization pipeline. Furthermore, we describe a
streamlined backward pass that employs adjoint rendering
for gradient computation.
Confidence-Interval Rasterisation
For the sake of clar-
ity, we will assume that the Gaussian parameters (µ, Σ) are
defined directly in the camera’s coordinate system.
The
opacity with which the camera observes the Gaussian is
given by the integral. For a single Gaussian component,
the integral can be written as:
Z ∞
tn
T(t)σ(t) dt = 1 −e−¯τ
Our geometric proxy is designed to encloses the confidence
interval
c = 1 −e−¯τ
for a given c ∈(0, 1) in the camera’s screen-space (we use
c = 0.01 in all experiments). Rearrange term to be an im-
plicit equation of the optical depth in an unbounded medium
−log(1 −c) = ωΣ
rπ
2

1 −erf
tn −µ
√
2Σ

.
To simplify the analysis, we assume the Gaussian is suf-
ficiently far from the camera’s near plane. That is the Gaus-
sian’s mean has a distance to the ray’s near bound greater
than
√
32Σ. Under this assumption we can neglect the ef-
fects of the erf term and the implicit equation becomes
−log(1 −c) = ωΣ
√
2π.
This assumption is justified since most visible particles al-
ready fulfill this assumption and for those which are closer
to the cameras near plane neglecting the erf term only leads
to an overestimation of its screen-space size. This might
harm performance but does not lead to a wrong appearance
of these splats.
Now, only ω and Σ are functions of the ray direction.
The weight-term ω varies exponentially with how aligned d
is to the v in the Σ−1-inner product, while Σ varies only by
a bounded square root factor determined by the eigenvalues
of the precision matrix Σ−1. To avoid the complexity of
analyzing the product of these two terms, the derivation re-
places Σ by a practical approximation evaluated along the
17

<!-- page 18 -->
view vector. While not a strict upper-bound, this approxi-
mation is empirically justified because ω decays exponen-
tially as the ray direction d diverges from v; thus, contribu-
tions where the approximation loosens are negligible. We
approximate the directional dependency as:
dT Σ−1d ≈
1
∥v∥2
2
vT Σ−1v
(26)
Consequently, we substitute Σ ≈∥v∥2/
√
vT Σ−1v, which
leaves us with analyzing the implicit curve:
−log(1 −c) = ω
∥v∥2
√
vT Σ−1v
√
2π.
Let C =

−log(1 −c)
√
vT Σ−1v

/
 √
2π ∥v∥2

cap-
ture all constants for a given confidence level. The implicit
curve ω = C can be expressed in terms of the ray direction
d by taking the logarithm:
log(C) = log(w) −1
2µT Σ−1µ + 1
2
 dT Σ−1µ
2
dT Σ−1d
Rearranging to isolate the terms dependent on d yields:
 dT Σ−1µ
2
dT Σ−1d
= 2(log(C) −log(w)) + µT Σ−1µ
The right-hand side is constant for a given level set. Let’s
call this constant κ. Letting A = Σ−1, the equation defin-
ing the surface of valid ray directions is:
(dT Aµ)2 −κ(dT Ad) = 0
A pixel with coordinates p = (u, v)T corresponds to a
homogeneous vector phom = (u, v, 1)T . The un-normalized
ray direction is dun = K−1phom where K is the intrinsic
matrix of a perspective camera. Since the level set equation
is scale-invariant with respect to d, we can substitute dun
directly:
(pT
hom(K−1)T Aµ)2 −κ(pT
hom(K−1)T AK−1phom) = 0
This is a quadratic form, pT
homWphom = 0, where the 3 × 3
symmetric matrix W defines the resulting conic section in
the image plane:
W = (mmT −κM)
with
vector
m = (K−1)T Aµ
and
matrix
M = (K−1)T AK−1.
To prove that the quadratic form pT
homW phom = 0 de-
scribes a real ellipse, we examine the eigenvalues of −W =
κM −mmT . Since M is congruent to the positive definite
precision matrix Σ−1, the base matrix κM has strictly pos-
itive eigenvalues 0 < µ1 ≤µ2 ≤µ3. Applying Cauchy’s
Interlacing Theorem for a rank-1 subtraction (Corollary
4.3.7 [15]) yields eigenvalues λi for −W that satisfy λ1 ≤
µ1 ≤λ2 ≤µ2 ≤λ3 ≤µ3, which immediately guarantees
that the two largest eigenvalues are positive (λ2, λ3 > 0).
The sign of the smallest eigenvalue λ1 is determined by the
secular equation at zero, f(0) = 1−mT (κM)−1m, which
reduces to 1 −κ−1µT Σ−1µ. Because the camera center is
assumed to be outside the Gaussian’s confidence interval
(µT Σ−1µ > κ), f(0) is negative, forcing the smallest root
λ1 < 0. The resulting signature (−, +, +) for −W implies
a signature of (+, −, −) for W , thereby characterizing a
real ellipse.
To transform the quadratic form into the standard repre-
sentation of an 2D ellipse
(p −µ2d)T Σ−1
2d (p −µ2d) = 1,
such that p = (u, v)T is on the ellipse, we first partition the
conic matrix W into a 2 × 2 block W2×2, a 2 × 1 vector
w2×1, and a scalar w33:
W =
W2×2
w2×1
wT
2×1
w33

.
Then, starting from the quadratic form pT
homWphom = 0, we
expand the left term and use the symmetry of scalar product:
pT W2×2p + 2wT
2×1p + w33 = 0
Define the screen-space center µ2d = −W −1
2×2w2×1 and
substitute wT
2×1 = −µT
2dW2×2:
pT W2×2p −2µT
2dW2×2p + w33 = 0
Completing the square with µT
2dW2×2µ2d:
(p −µ2d)T W2×2(p −µ2d) −µT
2dW2×2µ2d + w33 = 0
Adding µT
2dW2×2µ2d −w33 and dividing by the same
scalar gives
(p −µ2d)T
W2×2
µT
2dW2×2µ2d −w33(p −µ2d) = 1.
Finally, define the new ”covariance” matrix for the ellipse to
be Σ2d = (µT
2dW2×2µ2d −w33)W −1
2×2, where its inverse
is given by
Σ−1
2d =
W2×2
µT
2dW2×2µ2d −w33.
To find the tightest oriented bounding box for the ellipse
defined by (µ2d, Σ2d), we perform an eigen-decomposition
of Σ2d. The box is centered at µ2d and spanned by the
semi-axes vi = √λiei, where (λi, ei) are the eigenpairs of
Σ2d.
18

<!-- page 19 -->
Adjoint rendering
Since the rasterisation stage is imple-
mented using hardware-accelerated rasterisation, full auto-
differentiation through all operations is not available. In-
stead, adjoint rendering first computes adjoint moments
which allows to compute the gradients for particle parame-
ters in a single reduction pass.
The forward pass processes camera matrices V, K and
Gaussian parameters Θ, illustrated in Fig 1. of the main
paper.
A Moment pass first rasterizes all Gaussians; a
pixel shader computes the moment vector mi for each splat,
which are accumulated in an additive framebuffer to obtain
the per-pixel moment texture m. A subsequent Quadrature
pass, using V, K, Θ, and m, re-rasterizes the Gaussians. Its
pixel shader evaluates the radiance quadrature Li and con-
sistency penalty Pi, which are accumulated to produce the
observed radiance L. A Rescaling pass then uses a screen-
filling quad to pixel-wise rescale L using the first moment
m0 for correct opacity, yielding the final output radiance ˆL.
We omit the culling stage description for brevity, as non-
visible Gaussians simply receive a zero gradient. The re-
sulting image ˆL and per-pixel penalty values are passed to
PyTorch to compute the final loss L.
A naive backward pass inverts the forward operations,
see Fig 1. in the main paper, assuming all shader operations
are differentiated via automatic differentiation.
Starting
from the input gradients ∂L/∂ˆL and ∂L/∂P, the backward
Rescaling pass computes ∂L/∂L and ∂L/∂ˆL · ∂ˆL/∂m0.
The trivial derivative of the additive framebuffer accumula-
tion (∂V/∂Vi = 1 for a contributing splat i) is implicitly
handled by the backward rasterization, which gathers the
per-pixel textures ∂L/∂L and ∂L/∂P at pixels covered by
each splat. The backward Quadrature pass computes the
contribution to the adjoint moments, P
i
∂L
∂Li
∂Li
∂m + ∂L
∂Pi
∂Pi
∂m,
accumulating them in an additive framebuffer. It also com-
putes the derivative of the quadrature and penalty terms
w.r.t.
the splat parameters, which are reduced over all
pixels covered by the splat’s geometry: P
u,v
∂L
∂Li
∂Li
∂Θi +
∂L
∂Pi
∂Pi
∂Θi . Finally, the backward Moment pass takes the ad-
joint moment texture from the quadrature pass and ∂L/∂ˆL·
∂ˆL/∂m0 as input. It evaluates and reduces the derivative of
the density moment vectors w.r.t. the splat parameters over
all covered pixels: P
u,v
∂L
∂mi
∂mi
Θi .
The naive backward pass presents several issues: it re-
quires two reduction operations, creates intermediate ad-
joint textures for all forward pass framebuffers, and neces-
sitates an additional framebuffer for ∂L/∂ˆL·∂ˆL/∂m0. We
simplify this by observing that the derivative of the per-pixel
opacity re-scaling from the Rescaling pass can be pulled
into the other two stages, as it is evaluable from the forward
pass results L and m. This optimization eliminates three
additive framebuffer objects. Furthermore, given the ad-
joint moments, the per-splat gradients can be accumulated
in a single pass, avoiding two separate, expensive reduction
operations.
Our optimized backward pass therefore consists of two
stages, illustrated in Fig. 9 First, an Adjoint Moment pass
takes V, K, Θ, the forward results m, L, P, and the up-
stream gradients ∂L/∂L, ∂L/∂P as input. It rasterizes all
Gaussians, computing a per-splat, per-pixel adjoint moment
vector δmi, which is then accumulated into a per-pixel tex-
ture δm. Second, a Gradient pass inputs all arguments from
the previous pass, plus the adjoint moment texture δm. It
re-rasterizes all Gaussians, where the pixel shader computes
the per-splat gradient and performs the reduction over all
pixels covered by the splat’s proxy geometry, yielding the
final gradient ∇Θi:
∇Θi =
X
u,v
∂L
∂Li
∂Li
∂Θi
+ ∂L
∂Pi
∂Pi
∂Θi
+ ∂L
∂mi
∂mi
Θi
E. Optimisation
We derive the penalty term enforcing consistency between
density and moment-based transmittance. Additionally, we
detail the adaptation of adaptive density control to our vol-
umetric framework.
Regularisation
The moment-based reconstruction of the
transmittance function is limited in its complexity by the
number of moments provided to the reconstruction algo-
rithm. Consequently, the reconstructed transmittance over-
or might underestimates its true value along rays with
highly varying density. This allows the optimization pro-
cess to converge to minima which overfit on the training
views.
This behaviour is avoided by extending the 3DGS train-
ings objective with an additional regularisation objective to
ensure consistency between the predicted transmittance and
the actual density distribution along each ray.
L = (1 −α)L + αLD-SSIM + λLconsistency
Recall that the density is a sum of individual density val-
ues. Thus we can rewrite the transmittance on any interval
[tj, tj+1] as the product of transmittance values for the den-
sity of the i-th particle on the same interval:
T(tj →tj+1) =
Y
i
e
−
R tj+1
tj
σi(s)ds =
Y
i
Ti(tj →tj+1).
where Ti(tj →tj+1) is the transmittance when solely con-
sidering the density of the i-th particle. Since σi(xt) > 0
for any t, we have in particular the inequality
T(tj →tj+1) < Ti(tj →tj+1)
which hold for any interval [tj, tj+1] and any particle.
19

<!-- page 20 -->
Adjoint moments
pass
Gradient pass
Cam / Hyperparams
Shader pass
Accumulative
frame buffer
Figure 9. Architecture of the adjoint rendering pass. The pipeline executes in two passes: the computation of the adjoint moments, followed
by the calculation of the Gaussian parameter gradients. The backward step for opacity rescaling is folded into both passes for computational
efficiency. Inputs include the Gaussian parameters Θ, the forward pass outputs (moment texture m and radiance map L), and the upstream
gradients ∂L/∂ˆL and ∂L/∂P.
This inequality is the basis of our consistency regulariza-
tion that we apply to each interval used in the quadrature.
However, our order-independent transmittance predicts the
transmittance from tn to any t. We therefore reframe the
inequality to only involve the transmittance from tn to the
interval edges by first multiply both sides of the inequality
by T(tn →tj):
T(tn →tj) T(tj →tj+1) < T(tn →tj) Ti(tj →tj+1)
Using the multiplicative property of the transmittance, i.e.
T(tn →tj)T(tj →tj+1) is equal to T(tn →tj+1), substi-
tuting this into the inequality and dividing by T(tn →tj)
again gives the inequality
T(tn →tj+1)
T(tn →tj)
< Ti(tj →tj+1).
The fraction between the transmittance values is numer-
ically disadvantageous and can be avoided by rewriting the
inequality in terms of the optical depth. The right side of
the inequality can be solved analytically for each particle
and each interval:
Ti(tj →tj+1) = e−τij
where τij is the optical depth over the j-interval for the i-th
particle which can be expressed closed-form via the error
function. Further, applying the logarithm to both sides to
the inequality and multiplying both sides with −1 gives an
equivalent inequality for the optical depth
τ(tn →tj+1) −τ(tn →tj) ≥τij.
We reframe this inequality into a regularization term that
penalizes inconsistencies between the optical depth predic-
tions and the density of the i-th particle:
Pi =
N
X
j=1
max(0, τij −(τ(tn →tj+1) −τ(tn →tj)))2
We use squared penalty to more strongly punish strong out-
liers. The penalty is evaluated over the fixed quadrature in-
tervals [tj, tj+1]. The penalty loss then computes an aver-
age over all pixels and particles cover the pixel
Lconsistency =
1
Npixel
X
u,v
X
i
Pi
Adaptive Density Control
To progressively increase the
number of particles, we employ the adaptive density con-
trol (ADC) presented in the original 3DGS publication and
introduce only modifications to make it consistent with our
density-based medium. While more effective densification
schemes have been presented, we adapt the original ADC
formulation for better comparability to the 3DGS baseline
and more recent methods which also adapt this formulation.
If not state otherwise, our densification follows the steps
outlined in [18] and uses the same hyperparameters.
View-Independent Opacity ADC uses the splats opac-
ity parameter as the criterion whether or not to prune a par-
ticle. This criterion relies on the fact that 3DGS is ampli-
tude preserving for all viewing directions [9], that is, only
the splats shape changes but its ”transparency” is the same
from all angles.
Recall that our medium parameterisation assigns each
particle a maximal extinction instead of an opacity value.
Furthermore, our volumetric rendering approach deter-
mines a particles visibility via a numerical integration. A
key consequence of this approach is that an individual par-
ticle’s opacity becomes a view-dependent quantity.
Here, even a particle with low extinction but high stan-
dard deviation might have a high visible contribution when
viewed in direction of its longest axis. Contrary a particle
with high extinction can have a negligible visual contribu-
tion when it standard deviation is small enough. Therefore,
pruning based on particle’s maximal extinction or eigenval-
ues alone does not yield satisfactory results. Instead, we use
the particles highest opacity when viewed along its shortest
axis through its center for the opacity pruning. We specif-
ically use the shortest axis to reduce the potential for over-
20

<!-- page 21 -->
fitting where thin elongated particles are created during op-
timization which only have a visible contribution from very
specific viewing directions.
Recall that the particle’s individual opacity along a ray
while ignoring its surrounding medium is 1 −e−τ(tf ) with
τ(tf) = ωΣ
rπ
2

erf
tf −µ
√
2Σ

−erf
tn −µ
√
2Σ

The erf terms describe the visibility falloff the ray bound-
aries.
For the view-independent opacity we can upper-
bound this term by 2, which corresponds to the boundary
value for a particle that is in perfect view. The simplified
term becomes
1 −e−ωΣ
√
2π.
where ω and Σ are given by Eq. 5 and Eq. 6 in the main
paper.
The assumption that the particle is viewed along its
shortest axis implies that the viewing direction d is the
unit eigenvector corresponding to the smallest eigenvalue
λmin(Σ) of the covariance matrix Σ.
Consequently, the
quadratic form in the variance along this ray simplifies to:
dT Σ−1d =
1
λmin(Σ).
By construction of Σ we have λmin(Σ) = min(s2
x, s2
y, s2
z)
and therefore Σ =
p
λmin(Σ) = min (sx, sy, sz).
Given
that we view the particle through its center, the camera-to-
particle direction v is co-linear with d, such that v = c · d
for c ≥0. This co-linearity causes the exponent term in w
to vanish:
−1
2vT Σ−1v + 1
2
 dT Σ−1v
2
dT Σ−1d
= −1
2c2dT Σ−1d + c2 1
2
 dT Σ−1d
2
dT Σ−1d
= 0.
Therefore ω simplifies to w. Combining these results yields
the the view-independent opacity o:
o = 1 −e−
√
2π w min(sx,sy,sz).
We use the view-independent opacity to convert the ini-
tial point opacity into an initial density value when creating
the scene from the colmap scan. By solving the above equa-
tion for the peak density w, we get
winit =
−log(1 −oinit)
√
2πavg(sx, sy, sz)
we observed that using the average instead of minimal scal-
ing parameter slightly improved convergence. The original
ADC resets the opacity to a value slightly above the pruning
threshold at regular intervals, similar to [36], we found that
this does not improve our results and disabled the operation.
Cloning and Splitting ADC compares an average of the
screen-space gradient lengths to a threshold in addition to
a size criteria in order to decide whether to clone or split a
point. However, as a result of our modified rasterization ap-
proach, there are no screen-space gradients, which are orig-
inally used in ADC to determine weather to clone or split a
point. Instead we only have access to the 3D gradient of the
mean positions; where we observed stronger cancellation of
opposing gradient directions compared to screen-space gra-
dient which results in an overall lower magnitude. To com-
pensate for these effects, we employ the strategy proposed
in [41] and accumulate the absolute value of each gradient
component and use the norm of the accumulated directional
gradient magnitudes as the decision criterion. Furthermore,
we reduce the threshold from 2e −4 to 1e −4.
The cloning operation simply duplicates a selected par-
ticle without further changes to its parameters. This opera-
tion introduces a bias, since the cloned particle’s color con-
tribution becomes overly pronounced but can be corrected
for by change the splats opacity parameter. However, pre-
viously introduced corrections [34] are only valid for the
alpha-blending used in 3DGS and do not extend to our volu-
metric setting. Fortunately, the concept that the contribution
of a particle should not be increased by cloning, is easy to
adapt to the density since it is a linear quantity. We correct
for the bias by updating the peak density with:
w ←1
2w.
It is easy to verify that this correction avoids the over rep-
resentation of cloned points. Without the correction, the
cloning operations is appearance-wise equivalent to dou-
bling a cloned particles density. That is the density of a
cloned point would become
σ(t) = 2ωe−(t−µ)
2Σ2 .
Consequently, its opacity without taking the surrounding
medium into consideration would be
Z tf
tn
T(tn →t)σ(t)dt = 1 −e2τ(tf )
where τ(tf) is the particles optical depth. Recall
τ(tf) = ωΣ
rπ
2

erf
tf −µ
√
2Σ

−erf
tn −µ
√
2Σ

.
and ω = we−K. Thus scaling the maximal extinction by
half before cloning yields a density in which the cloned
points are not visually over-represent.
The splitting operation samples the mean for two new
points for each selected point and copies all other parame-
ters from the original except for the scaling parameter. The
21

<!-- page 22 -->
scaling parameters of the newly created points are down-
scaled by constant factor of 5/8. These newly created point
have a high likelihood to have a substantial overlap since
they are drawn from a Gaussian distribution. As pointed out
when addressing the clone-operation, creating overlapping
points without rescaling the density introduces a significant
bias that can destabilize the optimisation. We observed in
experiments that scaling the particles size parameters by a
constant factor was not sufficient to reduce the bias to get
a stable training behaviour. Furthermore, the randomness
within the splitting operation makes an analysis of the bias
difficult. We introduce a modified splitting operation which
allows us to minimize the bias introduced by this operation:
Each particle that satisfies the splitting-condition, is divided
into two new particles along the direction dsplit of its eigen-
vector with the highest eigenvalue. The means of the new
particles are shifted by a fixed offset δ > 0 with differing
sign along this direction:
µnew = µ ± δdsplit
To minimize the change in opacity of the newly created
points when viewed along dsplit, we downscale the largest
scale parameter that with γ ≤1:
Σnew = γΣsplit
All other parameters are simply copied from the split-up
particle to the new ones. Splitting and modifying parame-
ters only in one direction allows us to analyse the problem
for 1D Gaussians. The density of the particle before split-
ting is referred to as
σold(t) = e
−(t−µ)2
2Σ2
split
while the density after splitting the particle in two is
σnew(t) = e
−(t−(µ−δ))2
2(γΣsplit)2 + e
−(t−(µ+δ))2
2(γΣsplit)2 .
The main source of bias is the increased visual contribu-
tion of the newly created points over the original point. We
therefore choose δ, γ such that we minimize the changes to
the opacity:
min
δ,γ
Z ∞
−∞
(Told(tn →t)σold(t) −Tnew(tn →t)σnew(t))2dt
By substituting t′ =
t−µ
Σsplit , we can simplify this problem.
The density before splitting is a standard Gaussian
σold(t′) = e−1
2 t′2
and the density after the splitting operation is
σnew(t′) = e−

t′−
δ
Σsplit
2
2γ2
+ e−

t′+
δ
Σsplit
2
2γ2
Table 3. Per-scene metrics for our presented method.
Scene
PSNR ↑
SSIM ↑
LPIPS ↓
# Points
Time [h]
Bicycle
23.15
0.646
0.291
3 404 626
5.69
Bonsai
31.17
0.938
0.196
951 805
4.69
Counter
28.13
0.900
0.197
1 310 940
9.98
Garden
26.50
0.841
0.141
2 046 786
5.82
Flowers
18.99
0.488
0.374
2 766 098
11.06
Stump
24.51
0.682
0.286
1 994 725
4.86
Treehill
20.95
0.520
0.365
3 347 791
7.83
Room
30.88
0.923
0.202
1 569 865
4.90
Kitchen
29.35
0.906
0.150
1 645 792
9.25
Train
20.11
0.781
0.242
1 675 025
5.37
Truck
24.25
0.869
0.145
1 065 513
2.83
Playroom
29.60
0.908
0.242
1 562 347
14.85
DrJohnson
28.67
0.891
0.254
3 819 750
6.21
(Note, that γ is a dimensionless scaling parameter and
thus directly applies to the original problem domain.) Di-
rectly, solving this optimization problem would return the
trivial solution (δ = 0).
So instead, we constrain the
optimization problem such that the confidence intervals
[−cΣnew, cΣnew] = [−cγΣsplit, cγΣsplit] of each split par-
ticle lies within the confidence interval [−cΣsplit, cΣsplit] of
the original particle. In particular, we want to place the par-
ticles as far apart from each other as possible while staying
within the confidence intervals, which leads to the following
connection between γ and δ′:
γ = 1 −
δ
cΣsplit
⇒δ = cΣsplit (1 −γ)
We therefore substitute δ leading to
σnew(t′) = e−(t′−c (1−γ))2
2γ2
+ e−(t′+c (1−γ))2
2γ2
and solve the above minimization problem with variables
c, γ. Unlike the original problem, shrinking the particle by
γ always implies a corresponding shift away from the cen-
ter. A numerical optimization of this problem converges to
the solution:
γ = 0.6385502815246582 and δ = 0.6128153090966912
F. Results
An overview of all reported metrics on a per-scene basis is
reported in Tab. 3.
22
