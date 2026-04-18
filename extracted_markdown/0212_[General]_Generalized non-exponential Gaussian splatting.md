<!-- page 1 -->
Generalized non-exponential Gaussian splatting
SÉBASTIEN SPEIERER, Meta, Switzerland
ADRIAN JARABO, Meta, Spain
Reference
Superlinear (Ours)
Linear (Ours)
Sublinear (Ours)
Reconstructed
Exponential
Distance
0
1
Transmitance
Superlinear (Ours)
Linear (Ours)
Sublinear (Ours)
Exponential
PSNR: 32.80 dB
Avg. overdraw: 16.7
Speedup: 3.80×
PSNR: 32.95 dB
Avg. overdraw: 19.2
Speedup: 3.21×
PSNR: 33.55 dB
Avg. overdraw: 24.6
Speedup: 1.95×
PSNR: 33.75 dB
Avg. overdraw: 47.6
Speedup: 1.00×
0
20
40
60
80
100
120
Overdraw
Fig. 1. Non-exponential transmittance for 3D Gaussian Splatting (3DGS). We compare the baseline exponential transmittance model implicit in 3DGS
image formation model, against our new non-exponential transmittance models on the Dr Johnson scene, refined from a pretrained 3DGS asset. Top row:
Ground-truth reference (left) and reconstructed images for each transmittance model. Bottom row: A plot of the transmittance decay functions (left) and
per-pixel overdraw count visualizations. Our non-exponential models achieve comparable image quality to the baseline model, while reducing overdraws by
3–4×, directly translating to faster rendering.
In this work we generalize 3D Gaussian splatting (3DGS) to a wider family of
physically-based alpha-blending operators. 3DGS has become the standard
de-facto for radiance field rendering and reconstruction, given its flexibility
and efficiency. At its core, it is based on alpha-blending sorted semitranspar-
ent primitives, which in the limit converges to the classic radiative transfer
function with exponential transmittance. Inspired by recent research on
non-exponential radiative transfer, we generalize the image formation model
of 3DGS to non-exponential regimes. Based on this generalization, we use a
quadratic transmittance to define sub-linear, linear, and super-linear versions
of 3DGS, which exhibit faster-than-exponential decay. We demonstrate that
these new non-exponential variants achieve similar quality than the original
3DGS but significantly reduce the number of overdraws, which result on
speed-ups of up to 4× in complex real-world captures, on a ray-tracing-based
renderer.
CCS Concepts: • Computing methodologies →Rendering.
Additional Key Words and Phrases: gaussian splatting, radiance fields, non-
exponential transmittance
1
INTRODUCTION
Recent methods on radiance field rendering and reconstruction
[Mildenhall et al. 2020] have allowed photorealistic rendering of
real-world scenes and humans. In particular, 3D Gaussian splatting
Authors’ addresses: Sébastien Speierer, speierers@meta.com, Meta, Switzerland; Adrian
Jarabo, ajarabo@meta.com, Meta, Spain.
(3DGS) [Kerbl et al. 2023] has emerged as the most versatile repre-
sentation, and its flexibility and efficiency have made it particularly
attractive for applications including photorealistic avatars model-
ing [Qian et al. 2024; Saito et al. 2024], photorealistic rendering in
virtual reality [Jiang et al. 2024], scene understanding [Jang and Kim
2025; Zhou et al. 2024a], or generative three-dimensional graphics
[Tang et al. 2024].
3DGS uses trainable point-based volumetric semitransparent
emissive primitives, whose contribution is accumulated in each
pixel using multiplicative alpha-blending. While several alterna-
tives over the building blocks of the original technique have been
proposed, including alternative emissive functions [Diolatzis et al.
2024; Liu et al. 2025] or kernel primitives [Condor et al. 2025; Hamdi
et al. 2024; Huang et al. 2025; Liu et al. 2025], the image formation
model essentially stays the same.
In particular, the multiplicative-alpha-blending-based image for-
mation model is a discrete form of, and in the limit converges to,
the classic radiative transfer equation (RTE) [Chandrasekhar 2013;
von Lommel 1889], which predicts an exponential transmittance
under the assumption of volumes with uncorrelated particles. This
contrast with nature and real-life matter, where correlation has
been observed in most appearances, including clouds [Davis and
Marshak 2004], vegetation [Knyazikhin et al. 1998], porous [Bellet
arXiv:2603.02887v2  [cs.GR]  4 Mar 2026

<!-- page 2 -->
2
•
Sébastien Speierer and Adrian Jarabo
et al. 2009] and granular [Meng et al. 2015] materials, or periodic
crystal-like structures [Caglioti and Golse 2003]. This have fostered
the development of new radiative transfer frameworks [Bitterli
et al. 2018; Jarabo et al. 2018], that generalized the classic RTE to
non-exponential regimes.
In this work we generalize 3DGS to non-exponential transmit-
tance regimes, by first relating it with the classic RTE, and then
deriving its generalized non-exponential counterpart. Armed with
this physically-based framework, we then derive a set of new 3DGS
image formation models for the cases of linear- and quadratic-
like transmittance functions, including its forward and differential
modes using path-replay backpropagation [Vicini et al. 2021b]. We
show that these faster-than-exponential transmittance modes allow
to reconstruct photorealistic scenes with similar quality than the
exponential model, with the additional benefit of significantly less
overdraws, which translates into speed-ups of up-to around 4× in
real-world scenes in our ray-tracing-based renderer compared to
the baseline exponential model, as exampled in Fig. 1.
2
RELATED WORK
3D Gaussian splatting. A vast literature exists using and extend-
ing 3D Gaussian splatting: From the original proposal of Kerbl et al.
[2023], several works have been published improving on certain as-
pects of the method, including new kernels beyond Gaussians [Con-
dor et al. 2025; Hamdi et al. 2024; Huang et al. 2025; Liu et al. 2025],
new directional emission functions replacing spherical harmonics
[Diolatzis et al. 2024; Liu et al. 2025], spatially-varying textured
emission [Xu et al. 2024], or alternative primitives including linear
primitives [von Lützow and Nießner 2025], 3D convexes [Held et al.
2025c], triangles [Held et al. 2025b] and meshes [Held et al. 2025a].
Other works have focused on increasing the problem dimensionality
to the temporal domain [Wu et al. 2024; Yang et al. 2023], improving
the robustness of the optimization process [Kulhanek et al. 2024;
Rota Bulò et al. 2024; Sabour et al. 2025; Yin et al. 2025], or providing
scalable solutions for large scenes [Feng et al. 2025; Ren et al. 2025].
All these works are based on the same exponential image forma-
tion model; our work is orthogonal to these efforts, proposing a
generalized image formation model based on novel alpha-blending
compositing techniques of the splats.
Non-exponential light transport. Non-exponential or generalized
transport have been studied in fields including radiative transfer,
optics and neutron transport (see [d’Eon 2022] for an in-depth discus-
sion). In graphics, the first work exploring non-exponential regimes
was the work by Moon et al. [2007] in the context of granular media,
from a phenomenological approach. A more formal definition of
the problem was introduced by d’Eon [2014], though practical ren-
dering of non-exponential media was not possible until the works
of Jarabo et al. [2018] and Bitterli et al. [2018], which explored the
problem from two alternative perspectives: Jarabo et al. analyzed
the problem from a purely radiative perspective, while Bitterli et
al. developed a path-integral-based framework. Both works pro-
posed different non-exponential transmittance functions based on
the statistical properties of the media, similar to the one used by
Wrenninge et al. [2017]; this model was later extended by Guo et al.
[2019] introducing a more general transmittance model based on
fractional Gaussian fields. Vicini et al. [2021a] exploited the addi-
tional representation power of non-exponential media to improve
reconstruction of volumetric assets, which was also used by Condor
and Jarabo [2022] to accelerate rendering of complex luminaries.
Closer to us, Zhou et al. [2024b] used a linear transmittance model
for closed-form integration of volumetric 3D Gaussians for physics-
based rendering. In contrast, our work leverages non-exponential
light transport for defining new alpha-blending operators in the
context of infinitesimal splats.
Alpha blending and transparency. From the initial work of Smith
and Catmull [Smith 1995] on alpha compositing, alpha blending
has been a fundamental tool in graphics. Porter and Duff´s [1984]
seminal work established the mathematical foundation for com-
positing digital images. Blinn [2002a,b] provided new derivations
of the over operator and implementation considerations for vari-
ous pixel formats. Glassner [2015] revisited the interpretation of
alpha, providing a consistent and physically meaningful model of
pixel structure, which we use as foundation for our model. Further
works have generalized and formalized alpha compositing: Willis
[2006] proposed a projective alpha color model, which unifies vari-
ous compositing approaches using an algebraic perspective, while
Duff [2017] analyzed deep compositing using Lie algebras, provid-
ing a rigorous mathematical framework for advanced compositing
operations. Our work can be seen as a generalization of the alpha
blending operator defined by an arbitrary asymptotic transmittance
of semitransparent layers.
3
GAUSSIAN SPLATTING
3D Gaussian splatting [Kerbl et al. 2023] uses a set of ellipsoidal
volumetric semi-transparent primitives (Gaussians)1 to model the
radiance emitted in a scene. Each primitive is represented by its
directional emission 𝐸𝑖(𝝎), center c𝑖, covariance Σ𝑖, and opacity ℵ𝑖.
The contribution of each primitive to a pixel is defined by a kernel
𝐾(x|{x𝑖, Σ𝑖}) (usually a Gaussian, but others exist), which we use
to compute the pixel opacity 𝛼𝑖(y) = ℵ𝑖· 𝐾(y|{x𝑖, Σ𝑖}).
Then, for a point x in space (e.g., a sensor´s pixel), the incident
radiance 𝐿(x, 𝝎) from direction 𝝎is computed as the sum of emitted
radiance by a set of 𝑁volumetric primitives projected to x from
direction 𝝎as,
𝐿0(x, 𝝎) =
𝑁
∑︁
𝑖=1
𝐸𝑖(𝝎) · 𝛼𝑖(x𝑖) ·
Ö
𝑗<𝑖
 1 −𝛼𝑗(x𝑗)
+ 𝐿𝑏(x𝑏, 𝝎) ·
𝑁
Ö
𝑖=1
(1 −𝛼𝑖(x𝑖)) ,
(1)
with x𝑖= x −𝝎· 𝑡𝑖, 𝑡𝑖the distance of the primitive 𝑖to x, and
𝐿𝑏the contribution from the background. The first summand is
the contribution of the Gaussians, and the second is the contribu-
tion from the background due to transparency; this second term is
generally omitted, but we include it here for completeness. In the fol-
lowing, we remove it for simplicity. This formulation assumes that
Gaussians are projected as billboards oriented towards the camera
(splats), which is the common image formation model in Gaussian
1For consistency with the related literature, during the text we will use the term
Gaussian to name the primitives, though the kernel used might be different.

<!-- page 3 -->
Generalized non-exponential Gaussian splatting
•
3
splatting. Note that while other works have proposed working with
truly-volumetric primitives [Celarek et al. 2025; Condor et al. 2025;
Kheradmand et al. 2025], in this work we focus on the splat-based
image formation model. The splats are sorted front-to-bottom, so
that ∀𝑖< 𝑗then 𝑡𝑖< 𝑡𝑗. In the following, we remove the spatial and
directional dependency for clarity.
Physical interpretation. The image formation model in Equation (1)
is a multiplicative alpha-blending. In this context, we can interpret
𝛼𝑖as the area coverage of occluding matter normalized by the unit
volume [Porter and Duff 1984]. This has the physical unit of m−1,
and describes the probability of being occluded in the differential
thickness of the splat, analogous to what in linear transport is the
differential probability of extinction [d’Eon 2022].
It is trivial to see that Equation (1) is the discrete counterpart of the
radiative transfer equation (RTE) [Chandrasekhar 2013] for emissive
volumes, so that is equivalent to write (see proof in Section A)
𝐿0 =
∫∞
0
𝑄(x𝑡) · 𝜎(x𝑡) ·
𝑇(𝜏𝑡)
z}|{
𝑒−𝜏𝑡𝑑𝑡,
(2)
with x𝑡= x −𝝎· 𝑡with 𝑡the length along the ray, 𝜏𝑡=
∫𝑡
0 𝜎(x𝑠)𝑑𝑠
the optical depth, 𝑇(𝜏𝑡) the exponential transmittance predicted by
the Beer-Boulder-Lambert law, and 𝜎(x𝑠) and 𝑄(x𝑠) the density and
emission fields, respectively derived from the splats along the ray as
𝜎(x𝑡) =
𝑁
∑︁
𝑖=1
𝛿(𝑡−𝑡𝑖) 𝛼𝑖
and
𝑄(x𝑡) =
𝑁
∑︁
𝑖=1
𝛿(𝑡−𝑡𝑖) 𝐸𝑖,
(3)
with 𝛿the Dirac delta.
Going deeper on the physical foundations of this model, a key
assumption leading us to an exponential-like behavior in Eq. (2) is
that the probability of extinction at each splat is an independent
statistical process (see [Kostinski 2001, 2002]). For this to happen,
the underlying coverage at each differential point in a splat needs
to be fully uncorrelated from all the other splats contributing to
a pixel. Intuitively, we can think of a splat as a disk formed by
infinitesimally small random particles uniformly distributed in 2D,
occluding light with probability equal to its density 𝛼𝑖. Assuming
two uncorrelated splats occluding 50% of the light each, occlusion
at each splat will be an statistically-independent process, so the
total unoccluded light after passing through the two splats will be
(1 −0.5) × (1 −0.5) = 0.25 (Fig. 2, a). However, if the particles in the
second splat are fully negatively correlated to the particles from the
first one, all light that passes through the first one will hit particles
in the second one, and the total light passing through the two splats
will be 1 −0.5 −0.5 = 0, forming a waterproof surface occluding all
light (Fig. 2, b).
4
NON-EXPONENTIAL SPLATTING
In this section, we generalize the image formation model of 3DGS
to arbitrary transmittance regimes. For that, we exploit the rela-
tionship between continuous and discrete media, and leverage the
generalization of the RTE to non-exponential regimes to derive
our new non-exponential splatting method. Then, we propose and
evaluate a new set of non-exponential image formation models.
0%
100%
25%
50%
𝛼 = 0.5
100%
50%
(a) Exponential (uncorrelation)
(b) Linear (negative correlation)
Fig. 2. Physical interpretation of splats. Splats can be thought as disk
consisting of small random occluding particles. When light passes through
an slab, some light is lost when hitting these particles. For uncorrelated
splats (a), light occlusion in each splat is an independent stochastic process.
For negatively-correlated splats (b), on the other hand, all light that passes
through the first slab is occluded by the second one.
4.1
Background: The generalized Boltzmann equation
To account for regimes where light transport does not exhibit ex-
ponential transmittance, Larsen [2007] proposed the generalized
Boltzmann equation (GBE), which in its integral form for emissive
media reads
𝐿0 =
∫∞
0
𝑄(x𝑡) · 𝜎(x𝑡,𝜏𝑡) · 𝑇(𝜏𝑡) 𝑑𝑡.
(4)
For emissive media, the GBE is almost identical to the classic RTE
with a key difference: The differential extinction probability 𝜎(x𝑡,𝜏𝑡)
is not constant, but depends on the distance traveled by photons
before being extincted. This has a major implication in the rest of the
terms governing the light transport, since 𝜎(x𝑡,𝜏𝑡), the path-length
distribution2 𝑝(𝜏𝑡), and the transmittance 𝑇(𝜏𝑡) are directly related
by
𝜎(x𝑡,𝜏𝑡) = 𝑝(𝜏𝑡)
𝑇(𝜏𝑡) ,
with
𝑝(𝜏𝑡) = −𝑑
𝑑𝑡𝑇(𝜏𝑡) = −𝜎(x𝑡) · 𝑑
𝑑𝜏𝑇(𝜏𝑡).
(5)
Note that since we are modeling a non-scattering medium Larsen´s
GBE holds in our case; if we would be also considering scattering
we would need to account for a more complex form to maintain
reciprocity [Bitterli et al. 2018; d’Eon 2018; Jarabo et al. 2018].
4.2
Our method
Equipped with Eqs. (4) and (5) we can just simplify the GBE as the
product of the path-length distribution and emission, following:
𝐿0 =
∫∞
0
𝑄(x𝑡) · 𝑝(𝜏𝑡) 𝑑𝑡,
(6)
which by using the relationship between continuous and discrete
media described in Eq. (3), we can just write the contribution of
each Gaussian as a function of their emission and their probabil-
ity of extinction. Including back the background contribution, the
2The path-length distribution, also known as probability of extinction or chord-length
distribution, is the probability of being extincted after traveling certain optical depth.
Do not confuse it with the differential extinction probability 𝜎(x𝑡,𝑡), which is the
probability of extinction when traveling a differential step.

<!-- page 4 -->
4
•
Sébastien Speierer and Adrian Jarabo
Table 1. Summary of the mother transmittance functions 𝑇(𝜏𝑡) and the corresponding discrete probabilities ¯𝑝𝑖<𝑘for 𝑖< 𝑘used in our experiments. The linear
transmittance was proposed by Jarabo et al. [2018] to stochastically model negatively-correlated opaque surfaces. The quadratic is an empirical transmittance,
that generalizes the linear transmittance providing sublinear and superlinear decays. The blended transmittance reformulates the blended transmittance
proposed by Vicini et al. [2021a], which is a linear combination of linear and exponential transmittance. Finally, the power-law model was proposed by Davis
and Marshak [2004] to model fractal fluctuations in positively-correlated media. The operator [𝑥]+ = max(𝑥, 0) is the clamping operator.
Name
Parameters
𝑇(𝜏𝑡)
𝑝𝑖<𝑘
Exponential
N/A
exp(−𝜏𝑡)
𝛼𝑖· Î
𝑗<𝑖(1 −𝛼𝑗)
Linear
N/A
[1 −𝜏𝑡]+
𝛼𝑖
Quadratic
𝑐∈[−0.5, ∞)

1 −𝜏𝑡−𝑐
2 · 𝜏2
𝑡

+
𝛼𝑖· (1 + 𝑐· 𝜏𝑖)
Blended
𝛾∈[0, 1]
[lerp(1 −𝜏𝑡, exp(−𝜏𝑡),𝛾)]+
𝛼𝑖· (1 −𝛾+ 𝛾· Î
𝑗<𝑖(1 −𝛼𝑗))
[Vicini et al. 2021a]
𝛾∈[0, 1]
lerp([1 −𝜏𝑡]+ , exp(−𝜏𝑡),𝛾)
lerp(𝑝lin
𝑖<𝑘, 𝑝exp
𝑖<𝑘,𝛾)
Power-Law
𝑣∈(−1, ∞]

(1 + 𝜏𝑡· 𝑣)−1/𝑣
+
𝛼𝑖· (1 + 𝜏𝑖· 𝑣)−1+𝑣
𝑣
generalized Gaussian splatting image formation model becomes
𝐿0 =
Splats
z     }|     {
𝑁
∑︁
𝑖=1
𝐸𝑖· 𝑝𝑖+
Background
z     }|     {
𝐿𝑏· 𝑇𝑁+1 ,
(7)
where 𝑝𝑖is the discrete probability of extinction for Gaussian 𝑖
which depends on the mother transmittance function 𝑇(𝜏𝑡) chosen
for defining the model, and with 𝑇𝑖= 1 −Í
𝑗<𝑖𝑝𝑗the discrete
transmittance operator.
This equation is general, and works with any arbitrary base trans-
mittance function as soon it is physically correct. We detail later
how to choose such base transmittance functions, but a key aspect
that needs to be considered is that it always needs to be in greater or
equal to zero. With that constraint, we can just define the discrete
probability of extinction of each Gaussian as
𝑝𝑖=


−𝑑
𝑑𝑡𝑇(𝜏𝑖)
𝑖< 𝑘,
1 −Í
𝑗<𝑘𝑝𝑗
𝑖= 𝑘,
0
otherwise,
(8)
with 𝜏𝑖= Í
𝑗<𝑖𝛼𝑗the discrete optical depth, and 𝑘the primitive
that saturates transmittance so that Í𝑘
𝑖=0 𝑝𝑖≥1 and Í𝑘−1
𝑖=0 𝑝𝑖< 1.
Note that allowing for a primitive to saturate transmittance is a
key benefit from the generalized image formation model: Given the
multiplicative nature of the original Gaussian splatting, transmit-
tance never saturates to zero, and thus it is impossible to create
fully-opaque appearances except for splats with 𝛼𝑖= 1. In contrast,
in our model this fully-opaque behavior can be easily achieved by
choosing the appropriate base transmittance function.
4.3
Choosing a transmittance function
Our image formation model allows to use any arbitrary base trans-
mittance function𝑇(𝜏𝑡) to define the blending of the different splats.
However, for the transmittance to be physically meaningful there
are two conditions that must be taken into account: Given the phys-
ical definition of transmittance𝑇(𝜏𝑡) = 1−
∫𝑡
0 𝑝(𝜏𝑠)𝑑𝑠, and since the
path-length distribution 𝑝(𝜏𝑡) is a probability distribution, then a)
transmittance is always in the interval 𝑇(𝜏𝑡) ∈[0, 1] with 𝑇(0) = 1,
and b) it is always non-increasing. In addition to these two phys-
ical constraints, c) we impose that the slope of the transmittance
function at 𝜏𝑡= 0 is 𝑝(0) = 𝑑
𝑑𝑡𝑇(0) = 𝑑
𝑑𝑡𝜏𝑡= 𝜎(0), which effectively
makes the contribution of the first splat being always its opacity
𝑝1 = 𝛼1. Thus, the transmittance function only plays a role when
blending several splats.
With these three constraints, we propose six transmittance func-
tions, listed in Table 1, for which we analyze their performance in
toy-examples. Later in the radiance field experiments (Section 5) we
focus on the linear and quadratic mother transmittance functions.
4.4
Evaluation
Transmittance. We first evaluate the emerging transmittance when
blending a large set of low-opacity Gaussians. This allows us to vali-
date that the discrete transmittance ¯𝑇𝑖= 1−Í
𝑗<𝑖𝑝𝑗converges to the
mother continuous transmittance function𝑇(𝜏𝑡). Figure 3 shows the
results for transmittance functions with increasing effective decay
rate. As the effective decay rate increases, the appearance gets more
opaque, with transmittance quickly saturating to zero. This results
in significantly less overdraws for dense areas.
Blending. Figure 4 shows the results of blending three colored
Gaussians using different mother transmittance functions 𝑇(𝜏𝑡), in
two scenarios. In both cases, the results are similar, showing sharper
blending of the back Gaussians when using faster-than-exponential
transmittance functions, while at the same time obtaining more
opaque appearances, as shown in the inverse transmittance images.
5
RESULTS
We evaluate our non-exponential transmittance models in two exper-
imental settings: (i) reconstruction from scratch, where a 3D Gaussian
Splatting (3DGS) representation is optimized from COLMAP point
clouds and multi-view images, and (ii) refinement of existing as-
sets, where a pre-trained 3DGS scene produced by a conventional
rasterization-based renderer is fine-tuned under our ray-tracing
pipeline. In both cases, we compare three proposed transmittance
decay functions — superlinear, linear, and sublinear — against
the baseline exponential transmittance in the original 3DGS formu-
lation [Kerbl et al. 2023]. The two non-linear transmittance functions
are implemented using the quadratic transmittance with 𝑐= −0.5
(sublinear) and 𝑐= 0.5 (superlinear) respectively.

<!-- page 5 -->
Generalized non-exponential Gaussian splatting
•
5
Power-Law
(𝑣= 2)
Exponential
Quadratic
(𝑐= −0.5)
Vicini et al. [2021a]
(𝛾= 0.5)
Blend
(𝛾= 0.5)
Linear
Quadratic
(𝑐= 0.5)
Quadratic
(𝑐= 1)
Power-Law
(𝑣= −1)
1 −¯𝑇𝑁+1
Overdraws
Transmittance
𝜏
𝜏
𝜏
𝜏
𝜏
𝜏
𝜏
𝜏
𝜏
Fig. 3. Transmittance comparison. We render 100 elliptical Gaussians with random rotation with opacity ℵ= 0.02, distributed uniformly in depth, for
nine different transmittance functions, and compute the inverse transmittance 1 −¯𝑇𝑁+1 per pixel (top) and the number of overdraws until transmittance
saturates (middle, colormap goes from black to yellow). The bottom plots show the theoretical mother transmittance 𝑇(𝜏𝑡) of each model (green) compared
with the baseline exponential (dashed red), and the discrete transmittance ¯𝑇𝑖at the rendered splats (blue dots). Our predicted discrete transmittance matches
the theoretical one, which validates that our generalized model for splatting converges to the continuous generalized RTE. The faster is the decay, the less
overdraws are required to saturate, and the more opaque is the reconstructed appearance.
5.1
Implementation Details
Renderer. Our pipeline is implemented on top of Mitsuba [Jakob
et al. 2022]. Unlike the tile-based rasterizer of Kerbl et al. [2023], our
renderer evaluates Gaussian primitives via ray tracing, following
the approach of Moenne-Loccoz et al. [2024]. For each pixel, a ray
is cast through the scene’s bounding volume hierarchy and collects
all intersected Gaussians. Each intersected Gaussian 𝑖is rendered
as a billboard at a single depth 𝑡𝑖, located at the peak location of the
Gaussian along the ray. The intersected Gaussians are composited
front-to-back, with early termination when the accumulated trans-
mittance 𝑇𝑖saturates to zero. The maximum number of Gaussians
along the ray is set to 𝑁= 128 and all experiments were conducted
on a single NVIDIA RTX A6000 GPU.
For fairness, all results computed using the baseline exponential
transmittance are computed through our same pipeline and ren-
derer. Since ray-tracing is usually slower than rasterization-based
pipelines, the provided numbers with the baseline are slower than
previous works´. However, they serve as a baseline reference in
terms of both quality and speed-ups.
Loss function. Similar to Kerbl et al. [2023] we use a image-space
loss in sRGB space combining the ℓ1 and the structural similarity
(SSIM) losses, following
L = (1 −𝜆) L1 + 𝜆LSSIM .
(9)
Optimization. We use the bounded Adam optimizer [Condor
et al. 2025] with per-parameter learning rates modulated by an
exponential decay schedule from an initial rate of 1.0 down to 0.1
(see Table 2 for all hyperparameters used in the optimization). For
the backward phase, we use path-replay backpropagation [Vicini
et al. 2021b], with the recursive adjoints derived in Section C.
Coarse-to-fine scheduling. We use a coarse-to-fine strategy to ac-
celerate convergence at early steps. Training starts at 1
4 resolution
and is upsampled to 1
2 resolution at iteration 64, then to full resolu-
tion at iteration 200. The refinement pipeline operates at full reso-
lution throughout, since the Gaussians are already well-initialized
from the pretrained model.
Adaptive density control. We adopt the densification strategy of
Kerbl et al. [2023]. Every 100 iterations, Gaussians with accumulated
positional gradient magnitude exceeding 5 × 10−6 are candidates
for splitting. We split along the longest axis and disable cloning, as
we found split-only densification to be more stable for our trans-
mittance models. Pruning is applied every 200 iterations, removing
Gaussians with opacity below 0.008 or maximum scale below 10−4.
Densification is active only during the first 3,000 iterations, and the
maximum primitive count is capped at 200,000. In the refinement
pipeline, both densification and pruning are disabled; the Gaussian
count is inherited from the pretrained model (∼3.4M primitives for
realistic scenes).
5.2
Reconstruction
Initialization. Gaussians are initialized from a sparse points ob-
tained using COLMAP [Schönberger and Frahm 2016], with scales
set to each point’s nearest-neighbor distance (clamped to maximum
scale of 0.02), identity quaternion rotations, and initial opacity of
0.1. Colors are initialized from the COLMAP point colors. We addi-
tionally scatter 15,000 random Gaussians to help filling unobserved
regions.
Training budget. To highlight the practical impact of rendering
speed on optimization throughput, we impose a wall-clock time
budget rather than a fixed iteration count. For reconstruction, each

<!-- page 6 -->
6
•
Sébastien Speierer and Adrian Jarabo
Power-Law
(𝑣= 2)
Exponential
Quadratic
(𝑐= −0.5)
Vicini et al. [2021a]
(𝛾= 0.5)
Blend
(𝛾= 0.5)
Linear
Quadratic
(𝑐= 0.5)
Quadratic
(𝑐= 1)
Power-Law
(𝑣= −1)
RGB
Intensity
1 −¯𝑇𝑁+1
(a) 𝑁= 3 concentric Gaussians.
RGB
Intensity
1 −¯𝑇𝑁+1
(b) 𝑁= 3 anisotropic Gaussians, with different orientations.
Fig. 4. Blending comparison. We render 3 Gaussians with different colors and depths (red is closer, then green, and blue at the back) with opacity ℵ= 1, and
(a) without anisotropy and growing in size and (b) anisotropic and with different rotations in the image plane. All splats are parallel to the image plane, and do
not intersect. In both (a) and (b), the top row shows the result of blending, the middle row shows the scan-line at the center of the top image (with the black
dashed line the sum of the three Gaussians), and the bottom row is the inverse discrete transmittance 1 −¯𝑇𝑁+1. Faster-than-exponential transmittance (third
to ninth columns) show sharper blending, which for extreme cases (power-law with 𝑣= −1) results in unintuitive blending: This sharper blending behavior is
in part due to transparency saturating to one (see middle row), which creates 𝐶1 discontinuities on the blended primitives. Only the exponential transmittance
does not show this behavior, due to the statistical uncorrelation between each splat, at the cost of more overdraws.
run is limited to 5,000 s (∼83 min) with a maximum of 15,000 it-
erations. Within this budget, the superlinear model completes
roughly 13,000 iterations, while the exponential baseline manages
only ∼3,300—a ∼4× difference in optimizer steps, directly attribut-
able to the lower per-iteration rendering cost.
Results. We evaluate reconstruction quality on four scenes from
the NeRF synthetic dataset [Mildenhall et al. 2020] (Chair, Hotdog,
Lego, Materials). Quality and performance metrics are summarized
in the top half of Tables 3 and 4, respectively, and a visual comparison
is shown in Fig. 5.
As reported in Table 3, the superlinear model achieves the high-
est average PSNR (30.13 dB) and SSIM (0.9705), and the lowest MSE
(0.0010), outperforming the exponential baseline (28.99 dB, SSIM
0.9547, MSE 0.0014) by over 1 dB on average. The linear model
follows closely (29.88 dB, SSIM 0.9677), while even the sublinear
variant matches the exponential baseline in PSNR while produc-
ing substantially less overdraw. This quality advantage is a direct
consequence of the reduced rendering and optimization cost: as
shown in Table 4, the superlinear model renders at 50.3 FPS on
average compared to 9.1 FPS for the exponential model—a 5.5×
speedup—with an average overdraw of only 16.2 (72.1 for the base-
line). Under a fixed 5,000 s training budget, this speed-up allows the
optimizer to complete ∼4× more steps, leading to a substantially
better-converged reconstruction.
Figure 5 provides a qualitative comparison. Each scene is shown
as a row, with the ground-truth reference on the left and the four
transmittance models to the right. The rightmost column shows
PSNR convergence over both wall-clock time and iterations: the
non-exponential models converge faster in wall-clock time due to
their lower per-iteration cost, while their per-iteration convergence

<!-- page 7 -->
Generalized non-exponential Gaussian splatting
•
7
rate remains comparable to the exponential baseline. PSNR and FPS
values are annotated below each image.
5.3
Refinement
Initialization. We start from a pre-trained 3DGS asset, obtained
through the implementation of Kerbl et al. [2023] with the standard
image formation model, including the parameters of the Gaussians,
their opacity and SH color coefficients.
Training budget. For refinement, the budget is 1,800 s (30 min)
with a maximum of 1,000 iterations; the faster methods comfortably
finish all iterations, while the exponential baseline may not.
Results. For the refinement experiment, we use two indoor scenes
(Dr Johnson and Playroom [Barron et al. 2022]), and two outdoor
scenes (Train and Truck [Knapitsch et al. 2017]). We start from
pretrained models using the original splatting-based renderer [Kerbl
et al. 2023] (30,000 iterations, exponential transmittance). We fine-
tune these assets using our ray-tracing pipeline, continuing the
optimization for up to 1,000 iterations (or 30 min). Densification and
pruning are disabled; only the existing Gaussian parameters are fine-
tuned. Tables 3 and 4 (bottom half) show the reconstruction quality
and performance metrics for each scene, while Fig. 6 presents a
qualitative comparison.
The quality metrics in Table 3 confirm that our non-exponential
models maintain image fidelity after refinement. The sublinear
model achieves the best average PSNR (25.90 dB) and SSIM (0.8407),
closely followed by the exponential baseline (25.85 dB, SSIM 0.8399).
The linear and superlinear variants are less than 1 dB worse
on average, demonstrating that the Gaussians originally trained
under exponential transmittance can be successfully adapted to
non-exponential models with minimal quality loss.
The performance gains remain substantial even in this refinement
setting (Table 4). The superlinear model achieves 18.9 FPS on av-
erage versus 4.8 FPS for the exponential baseline—a 3.9× speedup—
while reducing average overdraw from 47.9 to 15.7. Figure 6 shows
that the refined images are visually similar across methods, while
the overdraw insets clearly reveal the efficiency advantage of the
non-exponential models.
6
DISCUSSION
We have seen experimentally that our faster-than-exponential trans-
mittance models consistently achieve 3–5× rendering speed-ups
with small-to-negligible impact on image quality, even improving
quality in the reconstruction case by enabling more optimization
iterations within the same time budget. The quadratic superlinear
model offers the best trade-off between speed and quality: It consis-
tently delivers the highest FPS and lowest overdraw while remaining
highly competitive in terms of image quality. The linear model
provides a balanced middle ground, while the sublinear variant
sacrifices some speed for the closest quality match to the exponen-
tial baseline—making it the safest drop-in replacement while still
resulting in a significant speed-up.
Limitations and Future Work. One limitation of our work is that
we have only experimented our transmittance models with ray-
tracing, which limits the comparison with state-of-the-art Gaussian
Table 2. Hyperparameters used for the reconstruction and refinement ex-
periments.
Parameter
Recon.
Refine.
Max iterations
15 000
1 000
Wall-clock budget
5 000 s
1 800 s
Max primitives
200 000
(inherited)
Centers LR
0.13
0.015
Scales LR
0.08
0.03
Quaternions LR
0.45
0.1
Opacity LR
1.0
1.0
SH coefficients LR
2.0
0.2
LR schedule
Exp. decay
Exp. decay
SSIM weight 𝜆
0.2
0.2
Coarse-to-fine
4× →2× →1×
Disabled
Densification interval
100 it.
Disabled
Pruning interval
200 it.
Disabled
SH degree
1
(inherited)
splatters. However, our new non-exponential models can be directly
ported to rasterization, which would still benefit from the significant
reduction in overdraws using faster-than-exponential transmittance
models, though sorting would still be required. Note however that
this performance gain is only possible if early termination when
transmittance saturates is available.
More conceptually, order-independent stochastic transparency
techniques [Kheradmand et al. 2025] are only available for exponen-
tial transmittance, where the binary Russian roulette applied on the
splats differential extinction probability is independent from previ-
ous splats. See Section B for a more thorough explanation. This is a
similar problem to delta tracking in heterogeneous non-exponential
media [Bitterli et al. 2018], and requires further investigation.
Finally, we have assumed a fixed pre-defined transmittance dur-
ing optimization. Including the transmittance parameters in the
optimization, even adaptively per Gaussian following the approach
of Vicini et al. [2021a], might increase the reconstruction quality, by
giving additional degrees of freedom in the image formation model.
Relationship with alpha blending. Our image formation model
defined by Eqs. (7) and (8) offers a physically-grounded form for
deriving new alpha blending operators, with any arbitrary decay
function. This directly relates with the physical interpretation pro-
vided by Glassner [2015] of alpha and compositing, and in particular
on how blending two semitransparent layers would depend on the
correlation between these two layers. For example, as described in
Fig. 2, the linear attenuation is a form of perfectly negative correla-
tion between scatterers. Further exploring physically-based trans-
mittance functions capable to model the whole range of inter-matter
correlation (from negative to positive) is still an open problem.
Conclusion. We have generalized 3D Gaussian splatting to sup-
port non-exponential transmittance regimes. From a theoretical
point of view, this has helped to link 3DGS with continuous volume
rendering. From the family of non-exponential 3DGS models we
have proposed, we have explored the usability of the quadratic
transmittance, analyzing the performance of superlinear, linear
and sublinear decays on radiance field reconstruction. Our results

<!-- page 8 -->
8
•
Sébastien Speierer and Adrian Jarabo
Chair
Reference
PSNR: 29.56 — FPS: 38.4
Superlinear (Ours)
PSNR: 28.88 — FPS: 24.0
Linear (Ours)
PSNR: 27.80 — FPS: 8.9
Sublinear (Ours)
PSNR: 27.27 — FPS: 8.3
Exponential
0
2000
4000
Time (s)
20
25
PSNR (dB)
0
5000
10000
Iteration
20
25
Lego
Reference
PSNR: 32.80 — FPS: 56.5
Superlinear (Ours)
PSNR: 32.50 — FPS: 38.9
Linear (Ours)
PSNR: 30.87 — FPS: 19.3
Sublinear (Ours)
PSNR: 29.98 — FPS: 8.4
Exponential
0
2000
4000
Time (s)
20
30
PSNR (dB)
0
5000
10000
15000
Iteration
20
30
Hotdog
Reference
PSNR: 34.53 — FPS: 36.4
Superlinear (Ours)
PSNR: 33.68 — FPS: 23.5
Linear (Ours)
PSNR: 32.32 — FPS: 9.2
Sublinear (Ours)
PSNR: 33.66 — FPS: 8.3
Exponential
0
2000
4000
Time (s)
20
30
PSNR (dB)
0
5000
10000
Iteration
20
30
Materials
Reference
PSNR: 28.80 — FPS: 69.9
Superlinear (Ours)
PSNR: 29.03 — FPS: 60.7
Linear (Ours)
PSNR: 28.29 — FPS: 24.4
Sublinear (Ours)
PSNR: 27.33 — FPS: 11.3
Exponential
0
2000
4000
Time (s)
20
25
PSNR (dB)
0
5000
10000
15000
Iteration
20
25
Fig. 5. Reconstruction results on NeRF synthetic scenes. Each row shows one scene (Chair, Hotdog, Lego and Materials), with the ground-truth reference
on the left and the reconstructions using each of the four image formation models to the right. The rightmost column plots PSNR convergence over wall-clock
time (top) and iteration count (bottom). Under a fixed time budget, the non-exponential models run significantly more iterations and converge to a higher
PSNR, while per-iteration convergence remains comparable across all models.
show that these new 3DGS models achieve comparable quality to
their classic exponential counterpart, while achieving 3 −4× less
overdraws, directly translated on significant speed-up during ren-
dering and training.
REFERENCES
Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.
2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In Proceedings
of CVPR.
Fabien Bellet, Elie Chalopin, Florian Fichot, Estelle Iacona, and Jean Taine. 2009. RDFI
determination of anisotropic and scattering dependent radiative conductivity tensors
in porous media: Application to rod bundles. International Journal of Heat and Mass
Transfer 52, 5-6 (2009), 1544–1551.
Benedikt Bitterli, Srinath Ravichandran, Thomas Müller, Magnus Wrenninge, Jan Novák,
Steve Marschner, and Wojciech Jarosz. 2018. A radiative transfer framework for
non-exponential media. ACM Trans. Graph. 37, 6 (2018), 1–17.
James F. Blinn. 2002a. Compositing. 1. Theory. IEEE Computer Graphics and Applications
22, 1 (2002), 83–87. doi:10.1109/38.974522
James F. Blinn. 2002b. Compositing, Part 2: Practice. IEEE Computer Graphics and
Applications 22, 2 (2002), 78–82. doi:10.1109/38.988747
Emanuele Caglioti and François Golse. 2003. On the distribution of free path lengths
for the periodic Lorentz gas III. Communications in Mathematical Physics 236, 2
(2003), 199–221.
Adam Celarek, George Kopanas, George Drettakis, Michael Wimmer, and Bernhard
Kerbl. 2025. Does 3D Gaussian Splatting Need Accurate Volumetric Rendering?
Computer Graphics Forum (2025), e70032.

<!-- page 9 -->
Generalized non-exponential Gaussian splatting
•
9
Drjohnson
Reference
PSNR: 31.64 — FPS: 10.3
Superlinear (Ours)
Avg. OD.: 16.7
PSNR: 31.70 — FPS: 8.7
Linear (Ours)
Avg. OD.: 19.2
PSNR: 32.59 — FPS: 5.3
Sublinear (Ours)
Avg. OD.: 24.6
PSNR: 32.37 — FPS: 2.7
Exponential
Avg. OD.: 47.6
0
25
50
75
100
125
Playroom
Reference
PSNR: 30.20 — FPS: 11.3
Superlinear (Ours)
Avg. OD.: 15.8
PSNR: 30.27 — FPS: 9.7
Linear (Ours)
Avg. OD.: 18.1
PSNR: 30.45 — FPS: 6.4
Sublinear (Ours)
Avg. OD.: 21.1
PSNR: 30.71 — FPS: 2.9
Exponential
Avg. OD.: 50.3
0
20
40
60
80
100
Truck
Reference
PSNR: 21.98 — FPS: 23.7
Superlinear (Ours)
Avg. OD.: 14.9
PSNR: 22.08 — FPS: 20.6
Linear (Ours)
Avg. OD.: 17.0
PSNR: 22.53 — FPS: 13.2
Sublinear (Ours)
Avg. OD.: 21.7
PSNR: 22.58 — FPS: 6.5
Exponential
Avg. OD.: 45.0
0
20
40
60
80
100
Train
Reference
PSNR: 16.77 — FPS: 30.5
Superlinear (Ours)
Avg. OD.: 15.3
PSNR: 17.19 — FPS: 26.7
Linear (Ours)
Avg. OD.: 17.7
PSNR: 18.01 — FPS: 15.9
Sublinear (Ours)
Avg. OD.: 24.1
PSNR: 17.75 — FPS: 7.3
Exponential
Avg. OD.: 48.6
0
20
40
60
80
100
Fig. 6. Refinement results on real scenes. Each row shows one scene (Dr Johnson, Playroom, Train and Truck), with the ground-truth reference on the
left and the reconstructions using each of the four image formation models to the right. Insets show the per-pixel overdraw count. The non-exponential
models maintain comparable visual quality comparable to the exponential baseline while substantially reducing overdraw, which results into 3–4× rendering
speedups.
Table 3. Image quality metrics across reconstruction and refinement experiments.
Superlinear (Ours)
Linear (Ours)
Sublinear (Ours)
Exponential
Scene
PSNR↑
MSE↓
SSIM↑
PSNR↑
MSE↓
SSIM↑
PSNR↑
MSE↓
SSIM↑
PSNR↑
MSE↓
SSIM↑
Reconstruction
Chair
29.16
0.0012
0.9721
28.85
0.0013
0.9654
28.13
0.0015
0.9530
27.72
0.0017
0.9471
Hotdog
32.77
0.0005
0.9777
32.01
0.0006
0.9735
30.52
0.0009
0.9669
32.39
0.0006
0.9731
Lego
30.26
0.0009
0.9748
30.19
0.0010
0.9731
29.44
0.0011
0.9622
28.61
0.0014
0.9536
Materials
28.33
0.0015
0.9573
28.47
0.0014
0.9589
28.04
0.0016
0.9532
27.26
0.0019
0.9448
Average
30.13
0.0010
0.9705
29.88
0.0011
0.9677
29.03
0.0013
0.9588
28.99
0.0014
0.9547
Refinement
Drjohnson
31.64
0.0007
0.9490
31.70
0.0007
0.9496
32.59
0.0006
0.9578
32.37
0.0006
0.9574
Playroom
30.20
0.0010
0.9096
30.27
0.0009
0.9109
30.45
0.0009
0.9152
30.71
0.0008
0.9195
Train
16.77
0.0210
0.6332
17.19
0.0191
0.6433
18.01
0.0158
0.6839
17.75
0.0168
0.6741
Truck
21.98
0.0063
0.7778
22.08
0.0062
0.7829
22.53
0.0056
0.8062
22.58
0.0055
0.8087
Average
25.15
0.0073
0.8174
25.31
0.0067
0.8217
25.90
0.0057
0.8407
25.85
0.0059
0.8399
Subrahmanyan Chandrasekhar. 2013. Radiative transfer. Courier Corporation.
Jorge Condor and Adrian Jarabo. 2022. A Learned Radiance-Field Representation for
Complex Luminaires. In Eurographics Symposium on Rendering.

<!-- page 10 -->
10
•
Sébastien Speierer and Adrian Jarabo
Table 4. Rendering performance across reconstruction and refinement experiments.
Superlinear (Ours)
Linear (Ours)
Sublinear (Ours)
Exponential
Scene
FPS↑
Speedup↑
Avg. OD↓
FPS↑
Speedup↑
Avg. OD↓
FPS↑
Speedup↑
Avg. OD↓
FPS↑
Speedup↑
Avg. OD↓
Reconstruction
Chair
38.4
4.63×
17.8
24.0
2.89×
20.8
8.9
1.07×
34.2
8.3
1.00×
79.0
Hotdog
36.4
4.39×
18.7
23.5
2.83×
21.0
9.2
1.11×
35.4
8.3
1.00×
73.2
Lego
56.5
6.73×
13.9
38.9
4.63×
18.2
19.3
2.30×
28.0
8.4
1.00×
70.3
Materials
69.9
6.19×
14.6
60.7
5.37×
16.8
24.4
2.16×
26.6
11.3
1.00×
65.9
Average
50.3
5.49×
16.2
36.8
3.93×
19.2
15.5
1.66×
31.0
9.1
1.00×
72.1
Refinement
Drjohnson
10.3
3.81×
16.7
8.7
3.22×
19.2
5.3
1.96×
24.6
2.7
1.00×
47.6
Playroom
11.3
3.90×
15.8
9.7
3.34×
18.1
6.4
2.21×
21.1
2.9
1.00×
50.3
Train
30.5
4.18×
15.3
26.7
3.66×
17.7
15.9
2.18×
24.1
7.3
1.00×
48.6
Truck
23.7
3.65×
14.9
20.6
3.17×
17.0
13.2
2.03×
21.7
6.5
1.00×
45.0
Average
18.9
3.89×
15.7
16.4
3.35×
18.0
10.2
2.09×
22.9
4.8
1.00×
47.9
Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic, Simon Green, Piotr Didyk,
and Adrian Jarabo. 2025.
Don’t Splat your Gaussians: Volumetric Ray-Traced
Primitives for Modeling and Rendering Scattering and Emissive Media. ACM Trans.
Graph. 44, 1 (2025), 1–17.
Anthony B. Davis and Alexander Marshak. 2004. Photon propagation in heteroge-
neous optical media with spatial correlations: enhanced mean-free-paths and wider-
than-exponential free-path distributions. Journal of Quantitative Spectroscopy and
Radiative Transfer 84, 1 (2004), 3–34.
Stavros Diolatzis, Tobias Zirr, Alexander Kuznetsov, Georgios Kopanas, and Anton
Kaplanyan. 2024. N-dimensional Gaussians for fitting of high dimensional functions.
In Proceedings of SIGGRAPH Conference Papers. 1–11.
Tom Duff. 2017. Deep Compositing Using Lie Algebras. ACM Trans. Graph. 36, 3 (2017),
1–13. doi:10.1145/3023386
Eugene d’Eon. 2014. Computer graphics and particle transport: our common heritage,
recent cross-field parallels and the future of our rendering equation. In Digipro 2014.
Eugene d’Eon. 2018. A reciprocal formulation of nonexponential radiative transfer. 1:
Sketch and motivation. Journal of Computational and Theoretical Transport 47, 1-3
(2018), 84–115.
Eugene d’Eon. 2022.
A Hitchhiker’s Guide to Multiple Scattering.
http://www.
eugenedeon.com/hitchhikers. Accessed: 2026-01-21.
Eric Enderton, Erik Sintorn, Peter Shirley, and David Luebke. 2010. Stochastic trans-
parency. In Proceedings of the I3D. 157–164.
Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, et al.
2025. FlashGS: Efficient 3D Gaussian Splatting for Large-Scale and High-Resolution
Rendering. In Proceedings of CVPR. 26652–26662.
Andrew S. Glassner. 2015. Interpreting Alpha. Journal of Computer Graphics Techniques
4, 2 (2015), 30–44. https://www.jcgt.org/published/0004/02/03/paper-lowres.pdf
Jie Guo, Yanjun Chen, Bingyang Hu, Ling-Qi Yan, Yanwen Guo, and Yuntao Liu. 2019.
Fractional Gaussian Fields for Modeling and Rendering of Spatially-Correlated
Media. ACM Trans. Graph. 38, 4 (2019).
Abdullah Hamdi, Luke Melas-Kyriazi, Jinjie Mai, Guocheng Qian, Ruoshi Liu, Carl
Vondrick, Bernard Ghanem, and Andrea Vedaldi. 2024. Ges: Generalized exponential
splatting for efficient radiance field rendering. In Proceedings of CVPR. 19812–19822.
Jan Held, Sanghyun Son, Renaud Vandeghen, Daniel Rebain, Matheus Gadelha, Yi Zhou,
Anthony Cioppa, Ming C. Lin, Marc Van Droogenbroeck, and Andrea Tagliasacchi.
2025a. MeshSplatting: Differentiable Rendering with Opaque Meshes. arXiv preprint
arXiv:2512.06818 (2025).
Jan Held, Renaud Vandeghen, Adrien Deliege, Abdullah Hamdi, Silvio Giancola, An-
thony Cioppa, Andrea Vedaldi, Bernard Ghanem, Andrea Tagliasacchi, and Marc Van
Droogenbroeck. 2025b. Triangle Splatting for Real-Time Radiance Field Rendering.
arXiv preprint arXiv:2505.19175 (2025).
Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio
Giancola, Andrea Vedaldi, Bernard Ghanem, and Marc Van Droogenbroeck. 2025c.
3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes. In
Proceedings of CVPR. 21360–21369.
Yi-Hua Huang, Ming-Xian Lin, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao,
and Xiaojuan Qi. 2025. Deformable Radial Kernel Splatting. In Proceedings of CVPR.
21513–21523.
Wenzel Jakob, Sébastien Speierer, Nicolas Roussel, Merlin Nimier-David, Delio Vicini,
Tizian Zeltner, Baptiste Nicolet, Miguel Crespo, Vincent Leroy, and Ziyi Zhang. 2022.
Mitsuba 3 renderer. https://mitsuba-renderer.org.
SungMin Jang and Wonjun Kim. 2025. Identity-aware language Gaussian splatting for
open-vocabulary 3D semantic segmentation. In Proceedings of ICCV. 20467–20476.
Adrian Jarabo, Carlos Aliaga, and Diego Gutierrez. 2018. A radiative transfer framework
for spatially-correlated materials. ACM Trans. Graph. 37, 4 (2018), 1–13.
Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li,
Henry Lau, Feng Gao, Yin Yang, and Chenfanfu Jiang. 2024. VR-GS: A Physical
Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality. In Pro-
ceedings of SIGGRAPH Conference Papers.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.
3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
4 (2023), 139–1.
Shakiba Kheradmand, Delio Vicini, George Kopanas, Dmitry Lagun, Kwang Moo Yi,
Mark Matthews, and Andrea Tagliasacchi. 2025. StochasticSplats: Stochastic Ras-
terization for Sorting-Free 3D Gaussian Splatting. arXiv preprint arXiv:2503.24366
(2025).
Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and
Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Trans. Graph. 36,
4 (2017).
Yuri Knyazikhin, Jörn Kranigk, Ranga B. Myneni, Oleg Panfyorov, and Gode Graven-
horst. 1998. Influence of small-scale structure on radiative transfer and photosyn-
thesis in vegetation canopies. Journal of Geophysical Research: Atmospheres 103, D6
(1998), 6133–6144.
Alexander B Kostinski. 2001. On the extinction of radiation by a homogeneous but
spatially correlated random medium. Journal of the Optical Society of America A 18,
8 (2001), 1929–1933.
Alexander B Kostinski. 2002. On the extinction of radiation by a homogeneous but
spatially correlated random medium: reply to comment. Journal of the Optical
Society of America A 19, 12 (2002), 2521–2525.
Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, and Torsten Sat-
tler. 2024. WildGaussians: 3D Gaussian Splatting in the Wild. arXiv preprint
arXiv:2407.08447 (2024).
Edward W. Larsen. 2007. A generalized Boltzmann equation for non-classical parti-
cle transport. In Proceedings of the International Conference on Mathematics and
Computations and Supercomputing in Nuclear Applications.
Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and Andrew Feng. 2025. Deformable
beta splatting. In Proceedings of SIGGRAPH Conference Papers. 1–11.
Johannes Meng, Marios Papas, Ralf Habel, Carsten Dachsbacher, Steve Marschner,
Markus Gross, and Wojciech Jarosz. 2015. Multi-scale modeling and rendering of
granular materials. ACM Trans. Graph. 34, 4 (2015), 1–13.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ra-
mamoorthi, and Ren Ng. 2020. Nerf: Representing scenes as neural radiance fields
for view synthesis. In Proceedings of ECCV.
Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez
Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 2024. 3D Gaussian
Ray Tracing: Fast Tracing of Particle Scenes. ACM Trans. Graph. (2024).
Jonathan T. Moon, Bruce Walter, and Stephen R. Marschner. 2007. Rendering discrete
random media using precomputed scattering solutions. In Proceedings of EGSR.
Thomas Porter and Tom Duff. 1984. Compositing digital images. In Proceedings of
SIGGRAPH. 253–259.

<!-- page 11 -->
Generalized non-exponential Gaussian splatting
•
11
Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain,
and Matthias Nießner. 2024. Gaussianavatars: Photorealistic head avatars with
rigged 3D Gaussians. In Proceedings of CVPR. 20299–20309.
Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai.
2025. Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D
Gaussians. IEEE Transactions on Pattern Analysis & Machine Intelligence 01 (2025),
1–15.
Samuel Rota Bulò, Lorenzo Porzi, and Peter Kontschieder. 2024. Revising densification
in Gaussian splatting. In Proceedings of ECCV. Springer, 347–362.
S. Sabour, L. Goli, G. Kopanas, M. Matthews, D. Lagun, L. Guibas, A. Jacobson, D. Fleet,
and A. Tagliasacchi. 2025. SpotlessSplats: Ignoring Distractors in 3D Gaussian
Splatting. ACM Trans. Graph. 44, 2 (2025), 1–11.
Shunsuke Saito, Gabriel Schwartz, Tomas Simon, Junxuan Li, and Giljoo Nam. 2024.
Relightable Gaussian codec avatars. In Proceedings of CVPR. 130–141.
Johannes Lutz Schönberger and Jan-Michael Frahm. 2016. Structure-from-Motion
Revisited. In Proceedings of CVPR.
Alvy Ray Smith. 1995. Alpha and the History of Digital Compositing. Technical Re-
port. Microsoft Corporation. https://www.cs.princeton.edu/courses/archive/spr05/
cos426/papers/smith95c.pdf
Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2024. DreamGaussian:
Generative Gaussian splatting for efficient 3D content creation. In Proceedings of
ICLR.
Delio Vicini, Wenzel Jakob, and Anton Kaplanyan. 2021a. A non-exponential transmit-
tance model for volumetric scene representations. ACM Trans. Graph. 40, 4 (2021),
1–16.
Delio Vicini, Sébastien Speierer, and Wenzel Jakob. 2021b. Path replay backpropagation:
Differentiating light paths using constant memory and linear time. ACM Trans.
Graph. 40, 4 (2021), 1–14.
Eugen von Lommel. 1889. Die Photometrie der diffusen Zuruckwerfung. (1889).
Nicolas von Lützow and Matthias Nießner. 2025. LinPrim: Linear Primitives for Differ-
entiable Volumetric Rendering. arXiv preprint arXiv:2501.16312 (2025).
Philip Willis. 2006. Projective Alpha Colour. Computer Graphics Forum 25, 1 (2006),
39–54. doi:10.1111/j.1467-8659.2006.00975.x
Magnus Wrenninge, Ryusuke Villemin, and Christophe Hery. 2017. Path traced sub-
surface scattering using anisotropic phase functions and non-exponential free flights.
Technical Report 17-07. Technical report.
G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang. 2024. 4D
Gaussian Splatting for Real-Time Dynamic Scene Rendering. In Proceedings of CVPR.
20310–20320.
Tian-Xing Xu, Wenbo Hu, Yu-Kun Lai, Ying Shan, and Song-Hai Zhang. 2024. Texture-
gs: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing. In
Proceedings of ECCV. Springer Nature Switzerland, Cham, 37–53.
Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. 2023. Real-time Photorealistic
Dynamic Scene Representation and Rendering with 4D Gaussian Splatting. arXiv
preprint arXiv:2310.10642 (2023).
Daheng Yin, Isaac Ding, Yili Jin, Jianxin Shi, and Jiangchuan Liu. 2025. TrackerSplat: Ex-
ploiting Point Tracking for Fast and Robust Dynamic 3D Gaussians Reconstruction.
In Proceedings of SIGGRAPH Asia 2025 Conference Papers. 1–11.
Hongyu Zhou, Jiahao Shao, Lu Xu, Dongfeng Bai, Weichao Qiu, Bingbing Liu, Yue Wang,
Andreas Geiger, and Yiyi Liao. 2024a. Hugs: Holistic urban 3D scene understanding
via Gaussian splatting. In Proceedings of CVPR. 21336–21345.
Yang Zhou, Songyin Wu, and Ling-Qi Yan. 2024b. Unified Gaussian Primitives for Scene
Representation and Rendering. (2024). arXiv:2406.09733
A
3D GAUSSIAN SPLATTING AS THE RTE
Here we derive the radiative transfer equation (RTE) for emissive
media (Eq. (2)) from the Gaussian splatting image formation model
(Eq. (1)). For that, we will start from the inner product term, that
accounts for the transparency from previous splats, and define it as
the transmittance from x to the point at splat x𝑖as
𝑇𝑖(x, x𝑖) =
Ö
𝑗<𝑖
 1 −𝛼𝑗(x𝑗) .
(A.1)
Now, using the continuous definition of the medium density in
Eq. (3), and by taking 𝑀small steps, we can rewrite 𝑇𝑖(x, x𝑖) as
𝑇(x, x𝑖) =
𝑀
Ö
𝑘=0
(1 −𝜎(x𝑘) · Δ𝑠),
(A.2)
with x𝑘= x + 𝝎· Δ𝑠· 𝑘, and Δ𝑠=
𝑡𝑖
𝑀with 𝑡𝑖= |x𝑖−x|. Now,
following the derivations from Kostinski [Kostinski 2001], we apply
logarithms to both sides and use a first order approximation on the
right to get
log (𝑇(x, x𝑖)) = log
 𝑀
Ö
𝑘=0
𝜎(x𝑘) · Δ𝑠
!
≈−
𝑀
∑︁
𝑘=0
𝜎(x𝑘) · Δ𝑠,
which by taking the limit Δ𝑠→0, applying exponentials to both
sides gives
𝑇(x, x𝑖) = exp

−
∫𝑡𝑖
0
𝜎(x𝑠) 𝑑𝑠

,
which is the heterogeneous transmittance predicted by the Beer-
Boulder-Lambert law.
Now, using the source and density definitions in Eq. (3), plugging
in𝑇(x, x𝑖), and taking 𝑀small steps until a maximum distance 𝑡max,
we can rewrite Eq. (1) as
𝐿0(x, 𝝎) =
𝑀
∑︁
𝑗=0
𝑄(x𝑘,𝜔) · 𝜎(x𝑘) · 𝑇(x, x𝑘) · Δ𝑠,
which by again taking the limit Δ𝑠→0 and setting 𝑡max = ∞results
into the Eq. (2):
𝐿0(x, 𝝎) =
∫∞
0
𝑄(x𝑘,𝜔) · 𝜎(x𝑘) · 𝑇(x, x𝑘) 𝑑𝑠.
(A.3)
B
TRANSMITTANCE AS A RUSSIAN-ROULETTE
PROCESS
Kheradmand et al. [2025] leveraged stochastic transparency [Ender-
ton et al. 2010] for avoiding sorting in 3DGS at the cost of intro-
ducing variance. For that, they transformed Eq. (1) into a recursive
estimator:
𝐿𝑖= 𝐸𝑖· 𝛼𝑖+ (1 −𝛼𝑖) · 𝐿𝑖+1,
(A.4)
which can be solved stochastically via Russian roulette, by choosing
whether we draw splat 𝑖with probability 𝛼𝑖, or if on the other hand
𝐿𝑖+1 is drawn. This allows to avoid sorting splats when rasterizing.
This approach is still valid within the framework of the general-
ized non-exponential light transport: By writing Eq. (A.2) using the
differential extinction probability (Eq. (5)) we get
𝑇(x, x𝑖) =
𝑀
Ö
𝑘<0
(1 −𝜎(x𝑘,𝜏𝑘) · Δ𝑠) =
𝑀
Ö
𝑘<0

1 −𝑝(𝜏𝑘)
𝑇(𝜏𝑘) · Δ𝑠

.
(A.5)
Proof. Starting with Eq. (A.5) and applying logarithms in both
sides, using a first-order approximation of the logarithm, we get
log(𝑇(x, x𝑖)) = log
 𝑀
Ö
𝑘<0

1 −𝑝(𝜏𝑘)
𝑇(𝜏𝑘) · Δ𝑠
!
=
𝑀
∑︁
𝑘<0

log

1 −𝑝(𝜏𝑘)
𝑇(𝜏𝑘) · Δ𝑠

≈
𝑀
∑︁
𝑘<0

−𝑝(𝜏𝑘)
𝑇(𝜏𝑘) · Δ𝑠

.

<!-- page 12 -->
12
•
Sébastien Speierer and Adrian Jarabo
Finally, taking the limit Δ𝑠→0, and using the relationship between
𝑝(𝜏) and 𝑇(𝜏) (Eq. (5)), and integrating we get
log(𝑇(x, x𝑖)) =
∫𝑡𝑖
0
−𝑝(𝜏𝑠)
𝑇(𝜏𝑠)𝑑𝑠
=
∫𝑡𝑖
0
−
−𝑑
𝑑𝑠𝑇(𝜏𝑠)
𝑇(𝜏𝑠)
𝑑𝑠
= log(𝑇(x, x𝑖)).
□
Now, to build the recursive form of our generalized 3DGS from
the transmittance defined in Eq. (A.5), we discretize the medium in
individual splats, which gives us
𝐿𝑖= 𝐸𝑖· 𝑝𝑖
𝑇𝑖
+

1 −𝑝𝑖
𝑇𝑖

· 𝐿𝑖+1.
(A.6)
Unfortunately, except for the case of exponential transmittance
where 𝑝𝑖
𝑇𝑖= 𝛼𝑖, the differential extinction probability defined by 𝑝𝑖
𝑇𝑖
is a function of the traversed optical depth (i.e., of all previous splats).
Thus, while stochastically we can early terminate the recursion,
sorting and evaluating the previous splats is still required.
C
ADJOINTS
In order to use path-replay backpropagation for efficient front-to-
back differentiable rendering, we cannot completely rely on auto-
matic differentiation for computing the adjoint phase recursively.
Here we derive the adjoints for both the linear and quadratic
image formation models. Note that the former is a special case of
the quadratic transmittance: We include it explicitly since given
its simpler form it allows for a more efficient implementation.
In order to compute the adjoints, we will first plug Eq. (8) in
Eq. (7), and rewrite 𝐿0 as
𝐿0 =
∑︁
𝑖<𝑘
𝐸𝑖· 𝑝𝑖+ 𝐸𝑘· (1 −
∑︁
𝑖<𝑘
𝑝𝑖) + 𝐿𝑏· (1 −
𝑁
∑︁
𝑖=1
𝑝𝑖)
=
∑︁
𝑖<𝑘
(𝐸𝑖−𝐸𝑘−𝐿𝑏) · 𝑝𝑖+ 𝐸𝑘+ 𝐿𝑏.
Note that for the background contribution 𝐿𝑏to be visible, trans-
mittance cannot saturate so in practice we can assume that when
𝑘= 𝑁, then 𝐸𝑘= 𝐿𝑏, and we can remove 𝐿𝑏from 𝐿0, leaving
𝐿0 =
∑︁
𝑖<𝑘
(𝐸𝑖−𝐸𝑘) · 𝑝𝑖+ 𝐸𝑘
=
∑︁
𝑖<𝑘
Δ𝐸𝑖· 𝑝𝑖+ 𝐸𝑘.
(A.7)
C.1
Linear transmittance
Plugging the 𝑝𝑖<𝑘term for linear transmittance into Eq. (A.7) we
get
𝐿0 =
∑︁
𝑖<𝑘
Δ𝐸𝑖· 𝑝𝑖+ 𝐸𝑘,
(A.8)
with derivative
𝛿𝐿0 =
∑︁
𝑖<𝑘
𝛿(Δ𝐸𝑖· 𝛼𝑖) + 𝛿𝐸𝑘
=
∑︁
𝑖<𝑘
(𝛿𝐸𝑖· 𝛼𝑖+ 𝐸𝑖· 𝛿𝛼𝑖−𝛿𝐸𝑘· 𝛼𝑖−𝐸𝑘· 𝛿𝛼𝑖) + 𝛿𝐸𝑘
=
∑︁
𝑖<𝑘

𝛿(𝐸𝑖· 𝛼𝑖) −𝐸𝑘· 𝛿𝛼𝑖

+ 𝛿𝐸𝑘· (1 −
∑︁
𝑖<𝑘
𝛼𝑖),
which can be written recursively as
𝛿𝐿𝑖=



𝛿(𝐸𝑖· 𝛼𝑖) −𝐸𝑘· 𝛿𝛼𝑖

+ 𝛿𝐿𝑖+1
𝑖< 𝑘
𝛿𝐸𝑘· 𝑇𝑘
𝑖= 𝑘
0
otherwise,
(A.9)
where𝑇𝑘can be computed from previous iterations, and 𝐸𝑘needs to
be computed in the forward pass and passed as input to the adjoint
pass.
C.2
Quadratic transmittance
Plugging the 𝑝𝑖<𝑘term for quadratic transmittance into Eq. (A.7)
we get
𝐿0 =
∑︁
𝑖<𝑘
Δ𝐸𝑖· 𝛼𝑖· (1 + 𝑐·
∑︁
𝑗<𝑖
𝛼𝑗) + 𝐸𝑘
(A.10)
=
∑︁
𝑖<𝑘
Δ𝐸𝑖· 𝛼𝑖+ 𝑐·
∑︁
𝑖<𝑘
 
Δ𝐸𝑖· 𝛼𝑖·
∑︁
𝑗<𝑖
𝛼𝑗
!
+ 𝐸𝑘.
The second sum can be expanded as
𝑁
∑︁
𝑖=0
Δ𝐸𝑖· 𝛼𝑖·
𝑖−1
∑︁
𝑗=0
𝛼𝑗= 0 · Δ𝐸0 · 𝛼0 + 𝛼0 · Δ𝐸1 · 𝛼1
+ (𝛼0 + 𝛼1) · Δ𝐸2 · 𝛼2 + ...
= 𝛼0 · (Δ𝐸1 · 𝛼1 + Δ𝐸2 · 𝛼2 + ...)
+ 𝛼1 · (Δ𝐸2 · 𝛼2 + Δ𝐸3 · 𝛼3 + ...) + ...
=
𝑁
∑︁
𝑖=0
 
𝛼𝑖·
𝑁
∑︁
𝑗=𝑖+1
Δ𝐸𝑗· 𝛼𝑗
!
.

<!-- page 13 -->
Generalized non-exponential Gaussian splatting
•
13
Now taking the derivative using the produce rule, and using the
definition of the discrete optical depth 𝜏𝑖= Í
𝑗<𝑖𝛼𝑗we get
𝛿𝐿0 =
∑︁
𝑖<𝑘
𝛿(Δ𝐸𝑖· 𝛼𝑖) + 𝑐·
∑︁
𝑖<𝑘
 
𝛿(Δ𝐸𝑖· 𝛼𝑖) ·
𝑖−1
∑︁
𝑗=0
𝛼𝑗
!
+ 𝑐·
∑︁
𝑖<𝑘
 
Δ𝐸𝑖· 𝛼𝑖·
𝑖−1
∑︁
𝑗=0
𝛿𝛼𝑗
!
+ 𝛿𝐸𝑘
=
∑︁
𝑖<𝑘
𝛿(Δ𝐸𝑖· 𝛼𝑖) + 𝑐·
∑︁
𝑖<𝑘
 
𝛿(Δ𝐸𝑖· 𝛼𝑖) ·
𝑖−1
∑︁
𝑗=0
𝛼𝑗
!
+ 𝑐·
∑︁
𝑖<𝑘
 
𝛿𝛼𝑖·
𝑘−1
∑︁
𝑗=𝑖+1
Δ𝐸𝑗· 𝛼𝑗
!
+ 𝛿𝐸𝑘
=
∑︁
𝑖<𝑘
 𝛿(Δ𝐸𝑖· 𝛼𝑖) + 𝑐· 𝛿(Δ𝐸𝑖· 𝛼𝑖) · 𝜏𝑖
+ 𝑐· 𝛿𝛼𝑖·
𝑘−1
∑︁
𝑗=𝑖+1
Δ𝐸𝑗· 𝛼𝑗
 + 𝛿𝐸𝑘
=
∑︁
𝑖<𝑘

𝛿(Δ𝐸𝑖· 𝛼𝑖) · (1 + 𝑐· 𝜏𝑖) + 𝑐· 𝛿𝛼𝑖· Θ𝑖

+ 𝛿𝐸𝑘,
with Θ𝑖defined recursively as
Θ𝑖=
(Í𝑘−1
𝑗=1 Δ𝐸𝑗· 𝛼𝑗
𝑖= 0
Θ𝑖−1 −Δ𝐸𝑖· 𝛼𝑖
𝑖> 0.
Further expanding Δ𝐸𝑖in 𝛿𝐿0 we get
𝛿𝐿0 =
∑︁
𝑖<𝑘
 𝛿(𝐸𝑖· 𝛼𝑖) −𝛿𝐸𝑘· 𝛼𝑖−𝐸𝑘· 𝛿𝛼𝑖
 · (1 + 𝑐· 𝜏𝑖)
+ 𝑐· 𝛿𝛼𝑖· Θ𝑖

+ 𝛿𝐸𝑘
=
∑︁
𝑖<𝑘
 𝛿(𝐸𝑖· 𝛼𝑖) −𝐸𝑘· 𝛿𝛼𝑖
 · (1 + 𝑐· 𝜏𝑖) −𝛿𝐸𝑘· 𝛼𝑖· (1 + 𝑐· 𝜏𝑖)
+ 𝑐· 𝛿𝛼𝑖· Θ𝑖

+ 𝛿𝐸𝑘
=
∑︁
𝑖<𝑘
 𝛿(𝐸𝑖· 𝛼𝑖) −𝐸𝑘· 𝛿𝛼𝑖
 · (1 + 𝑐· 𝜏𝑖) + 𝑐· 𝛿𝛼𝑖· Θ𝑖

+ 𝛿𝐸𝑘·
 
1 −
∑︁
𝑖<𝑘
𝛼𝑖· (1 + 𝑐· 𝜏𝑖)
!
=
∑︁
𝑖<𝑘
 𝛿(𝐸𝑖· 𝛼𝑖) −𝐸𝑘· 𝛿𝛼𝑖
 · (1 + 𝑐· 𝜏𝑖) + 𝑐· 𝛿𝛼𝑖· Θ𝑖

+ 𝛿𝐸𝑘· 𝑝𝑘.
With this, we can obtain a recursive formula for the adjoint:
𝛿𝐿𝑖=


 𝛿(𝐸𝑖· 𝛼𝑖) −𝐸𝑘· 𝛿𝛼𝑖
 · (1 + 𝑐· 𝜏𝑖) + 𝑐· 𝛿𝛼𝑖· Θ𝑖+ 𝛿𝐿𝑖+1
𝑖< 𝑘
𝛿𝐸𝑘· 𝑇𝑘
𝑖= 𝑘
0
otherwise.
(A.11)
Both 𝜏𝑖and 𝑇𝑘can be computed recursively from previous iter-
ations in the backward pass, while Θ𝑖requires Θ0, which can be
precomputed in the forward pass, and passed as input to the adjoint
pass with 𝐸𝑘.
