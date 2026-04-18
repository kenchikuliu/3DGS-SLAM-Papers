<!-- page 1 -->
Geometry-Grounded Gaussian Splatting
BAOWEN ZHANG, Hong Kong University of Science and Technology, Hong Kong
CHENXING JIANG, Hong Kong University of Science and Technology, Hong Kong
HENG LI, Hong Kong University of Science and Technology, Hong Kong
SHAOJIE SHEN, Hong Kong University of Science and Technology, Hong Kong
PING TAN, Hong Kong University of Science and Technology, Hong Kong
https://baowenz.github.io/geometry_grounded_gaussian_splatting
Multi-view 
Input Images
...
Reconstructed 
Surface
Ours
PGSR
Gaussians /
Stochastic Solids
Side Views
Method Comparison
Fig. 1. We prove that Gaussian primitives are equivalent to stochastic solids, and leverage this equivalence to reconstruct high-fidelity, multi-view-consistent
shapes from multi-view images.
Gaussian Splatting (GS) has demonstrated impressive quality and
efficiency in novel view synthesis. However, shape extraction from
Gaussian primitives remains an open problem. Due to inadequate
geometry parameterization and approximation, existing shape recon-
struction methods suffer from poor multi-view consistency and are
sensitive to floaters. In this paper, we present a rigorous theoretical
derivation that establishes Gaussian primitives as a specific type of
stochastic solids. This theoretical framework provides a principled
foundation for Geometry-Grounded Gaussian Splatting by enabling
the direct treatment of Gaussian primitives as explicit geometric rep-
resentations. Using the volumetric nature of stochastic solids, our
method efficiently renders high-quality depth maps for fine-grained
geometry extraction. Experiments show that our method achieves the
best shape reconstruction results among all Gaussian Splatting-based
methods on public datasets.
CCS Concepts: • Computing methodologies →Point-based models;
Volumetric models; Rendering.
Additional Key Words and Phrases: Gaussian Splatting, Stochastic
Solids, Shape Reconstruction
1
Introduction
3D shape reconstruction from multi-view images is a long-
standing problem with broad impact in virtual reality [Snavely
et al. 2006], autonomous driving [Schmied et al. 2023], and
robotics [Cadena et al. 2017; Engel et al. 2014]. Recent progress
has been driven by implicit neural representations, most notably
NeRF [Mildenhall et al. 2020]. Many state-of-the-art methods
Authors’ Contact Information: Baowen Zhang, Hong Kong University of Science
and Technology, Hong Kong, Hong Kong, bzhangcm@connect.ust.hk; Chenxing
Jiang, Hong Kong University of Science and Technology, Hong Kong, Hong
Kong, cjiangan@connect.ust.hk; Heng Li, Hong Kong University of Science and
Technology, Hong Kong, Hong Kong, eehengli@ust.hk; Shaojie Shen, Hong Kong
University of Science and Technology, Hong Kong, Hong Kong, eeshaojie@ust.hk;
Ping Tan, Hong Kong University of Science and Technology, Hong Kong, Hong
Kong, pingtan@ust.hk.
further adopt geometry-grounded radiance fields: they start
from a canonical geometry field (e.g., SDF/occupancy) and
derive the rendering formulation accordingly. Methods such as
VolSDF [Yariv et al. 2021] and NeuS [Wang et al. 2021] follow
this principle by anchoring the rendering to an explicit surface
and yielding reliable geometry that is consistent across views.
Despite these advances, geometry-grounded radiance fields
typically rely on dense sampling, e.g., ray marching, along
camera rays, resulting in slow training and inference.
In contrast, Gaussian Splatting [Kerbl et al. 2023] represents
scenes as a collection of Gaussian primitives and leverages
efficient rasterization, enabling fast optimization and real-time
novel view synthesis. Several follow-up works [Chen et al. 2024;
Guédon et al. 2025b; Huang et al. 2024; Yu et al. 2024c; Zhang
et al. 2024, 2025a] have extended Gaussian Splatting to shape
reconstruction with promising results. Nevertheless, Gaussian
Splatting does not inherently define a surface, unlike geometry-
grounded NeRF methods that start from an SDF/occupancy
field. Existing Gaussian Splatting–based methods therefore
extract depth or surfaces from the Gaussian radiance field
using heuristic rules. A more principled geometric formulation
can improve cross-view consistency and enable higher-fidelity
reconstruction, as shown in the right of Figure 1. Unlike these
heuristic pipelines, we provide a principled geometric foun-
dation for Gaussian primitives, enabling higher-fidelity shape
reconstruction.
In this paper, we adopt the philosophy of geometry-grounded
radiance fields by equipping Gaussians with a canonical geom-
etry field. We achieve this by leveraging the theoretical founda-
tion provided by the recent work ‘Objects as Volumes’ [Miller
et al. 2024], which offers a stochastic interpretation of the
geometry-grounded radiance field. Under this theory, we
arXiv:2601.17835v2  [cs.CV]  27 Jan 2026

<!-- page 2 -->
2
•
Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan
(a) Rasterization & Sort
(b) Gaussians as 
2D splats
3D GS
!(#)
(e) Binary search for 
median depth
!(#!"#, … )
#
$%
$&!"#
× $&!"#
$($
× $&!"#
$(%
× $&!"#
$(&
0
#!"#
…
!
(f) Closed form 
solution of gradient
!
0.5
(d) Median depth from 
discrete transmittance
Ours
!(#)
(c) Gaussians as 
stochastic solids
#
0
#!"#
Fig. 2. Overview of our depth-rendering pipeline. (a) We rasterize
Gaussian primitives and sort them by depth. (b) Standard Gaussian Splat-
ting yields step-wise transmittance under splat compositing. (c) Under our
stochastic solid formulation, attenuation is modeled continuously within
each primitive, yielding a smooth transmittance curve. (d) Prior work esti-
mates the ray-wise median depth as the point where transmittance drops
to 0.5. (e) Forward pass: we locate the median depth 𝑡𝑚𝑒𝑑, i.e., 𝑇= 0.5, via
binary search. (f) Backward pass: we backpropagate through 𝑡𝑚𝑒𝑑using a
closed-form gradient with respect to all Gaussians contributing to the ray.
analyze the rendering equation of Gaussian Splatting and
demonstrate that rendering a Gaussian primitive is identical to
rendering a stochastic solid (Section 4.1). This unifies rendering
formulations of Gaussian Splatting and NeRF-based methods,
allowing us, for the first time, to derive a geometric field for
Gaussian primitives. Using our formulation, we develop an
efficient depth-rendering method that approximates the isosur-
face of the geometric field and extracts finer-grained geometry
from Gaussian primitives (Section 4.2), exhibiting inherent
multi-view consistency and robustness to floaters.
Figure 2 illustrates our depth-rendering pipeline to exem-
plify our advantages in detail. Prior Gaussian Splatting–based
methods define the median depth along a ray as the location
where transmittance drops to 0.5, as shown in Figure 2(d).
However, because of discrete changes in transmittance, this
method fails to capture the joint effect of overlapping Gaussians
and leading to jagged depth steps. In contrast, the stochastic
solids model volumetric attenuation continuously and yield
a smooth transmittance curve. Building on this, we endow
Gaussian primitives with the same continuous behavior, en-
abling more detailed depth maps. To compute the median
depth, we exploit the monotonicity of transmittance and apply
a binary search to locate the 0.5-transmittance crossing. We
further derive a closed-form expression for the gradient of the
median depth with respect to the parameters of all Gaussians
along the ray for efficient backpropagation.
The main contributions of this paper are summarized as,
• We analyze the rendering equation of Gaussian Splatting
and demonstrate that the Gaussian primitives can be
regarded as stochastic solids, which provides theoret-
ical guidance for shape reconstruction from Gaussian
Splatting (Section 4.1).
• Based on this stochastic theory, we propose an efficient
method for rendering and optimizing depth maps from
Gaussian primitives, enabling accurate geometry extrac-
tion (Section 4.2).
• Extensive experiments demonstrate that our method
achieves the best reconstruction accuracy among Gauss-
ian Splatting-based methods, while maintaining opti-
mization efficiency of the Gaussian Splatting (Section 5).
2
Related Work
2.1
Continuous Radiance Fields
NeRF [Mildenhall et al. 2020] models a scene as a continuous
radiance field, typically parameterized by an MLP, and has
shown strong performance in challenging effects such as re-
flections and scattering [Andrea et al. 2023; Levy et al. 2023;
Tang et al. 2024]. Building on this backbone, Mip-NeRF intro-
duces an anti-aliased multiscale formulation through conical-
frustum rendering [Barron et al. 2021], and Mip-NeRF 360
extends it to unbounded scenes with specialized parameter-
ization and regularization [Barron et al. 2022]. To improve
efficiency, several works replace MLP ray marching with ex-
plicit volumetric parameterizations, e.g., Plenoxel’s voxel-grid
optimization [Fridovich-Keil et al. 2022] and SVRaster’s real-
time rasterization of adaptive sparse voxels [Sun et al. 2025];
Instant-NGP further accelerates training with multiresolution
hash grids [Müller et al. 2022].
While NeRF was originally designed for view synthesis,
recovering accurate geometry from a generic density field is
non-trivial, motivating surface-aware formulations that cou-
ple volume rendering with implicit surfaces. VolSDF [Yariv
et al. 2021], NeuS [Wang et al. 2021], and UNISURF [Oechsle
et al. 2021] parameterize density through a signed distance
function (SDF) and design rendering weights to obtain more
faithful surfaces. Neuralangelo further combines multireso-
lution hash-grid encodings with neural surface rendering to
achieve high-fidelity reconstruction from RGB captures [Li
et al. 2023]. GeoSVR [Li et al. 2025] explores explicit sparse
voxels for geometrically accurate surface reconstruction, lever-
aging uncertainty-aware depth constraints and voxel surface
regularization to improve detail and completeness. On the the-
oretical side, Objects as Volumes [Miller et al. 2024] provides
a stochastic-geometry view of representing opaque solids as
volumes and clarifies when exponential transmittance-based
models are physically consistent, offering principled insights
into surface-oriented volume rendering. Although these meth-
ods can reconstruct high-quality geometry, they generally suffer
from extreme time consumption.
2.2
Primitive Based Representations
Gaussian Splatting [Kerbl et al. 2023] represents 3D scenes
using a set of 3D Gaussian primitives. Combining with raster-
ization techniques, it avoids the time-consuming ray march-
ing process in NeRF rendering. As a result, it achieves both
real-time rendering and accelerated training. Building on this
foundation, Mip-Splatting [Yu et al. 2024a] addresses aliasing

<!-- page 3 -->
Geometry-Grounded Gaussian Splatting
•
3
by incorporating low-pass filters, while LightGaussian [Fan
et al. 2024] optimizes memory usage with a compact represen-
tation. VastGaussian [Lin et al. 2024] further extends Gaussian
Splatting to larger-scale scenes. StochasticSplats [Kheradmand
et al. 2025] adopts a Monte Carlo estimator to enable sort-free
rendering, further improving rendering efficiency.
Although 3DGS achieves high-quality novel-view synthesis,
the geometry recovered from purely photometric optimization
is often unreliable. To improve surface reconstruction, prior
work either imposes stronger geometric priors or adds geo-
metric supervision. SuGaR [Guédon and Lepetit 2024] and
NeuGS [Chen et al. 2023] favor surface-aligned (flattened) Gaus-
sians to better capture object boundaries and facilitate mesh
extraction. Related approaches [Huang et al. 2024; Zhang et al.
2025b] replace 3D Gaussians with 2D primitives to encourage
surface-like representations, although such constraints may
reduce modeling flexibility and become unstable in complex
scenes. GFSGS [Jiang et al. 2025] further leverages stochastic
solids to construct 2D surfels for shape reconstruction. Beyond
primitive design, 3DGSR [Lyu et al. 2024] and GSDF [Yu et al.
2024b] jointly optimize Gaussians with an implicit neural SDF
field, improving reconstruction fidelity while retaining splat-
ting efficiency, and PGSR [Chen et al. 2024] adds multi-view
geometric regularization.
Despite promising empirical progress, geometry extraction
in Gaussian Splatting still relies on heuristic depth definitions.
These heuristics often yield noisy depth maps that have poor
consistency across viewpoints and thus a weaker supervisory
signal for optimization. This raises a fundamental question of
whether Gaussian representations support an intrinsic notion
of geometry akin to NeRF-based methods. We address it by
adopting a stochastic approach to compute depth maps in a
more principled manner for high-quality shape reconstruction.
3
Preliminary
3.1
Gaussian Splatting
We first briefly revisit Gaussian Splatting (GS). A 3D Gaussian
primitive is defined as follows:
𝐺(x) = 𝑜𝑒−(x−x𝑐)⊤Σ−1 (x−x𝑐),
(1)
where 𝑜is the opacity, Σ ∈R3×3 is the covariance, x ∈R3 repre-
sents a point in 3D space, and x𝑐∈R3 denotes the Gaussian’s
center. To enable fast rasterization, Gaussian Splatting (GS)
methods employ a local affine approximation to project 3D
Gaussian primitives to 2D Gaussians on the image plane with
the covariance matrix Σ′
2𝐷The opacity of the 2D Gaussian 𝛼(u)
is defined as the maximum value of the projected 2D Gaussian:
𝛼(u) = 𝑜𝑒−(u−u𝑐)⊤Σ′−1
2𝐷(u−u𝑐),
(2)
where u is the coordinate of the pixels in the image space, u𝑐is
the projected center of the Gaussian. In this way, 3D Gaussian
primitives are projected into 2D Gaussians. These 2D Gaussians
are then sorted and alpha-blended to compute the final color.
More details can be found in the supplementary material.
3.2
Objects as Volumes
In this subsection, we provide a brief overview of [Miller et al.
2024], which presents a method to render stochastic solids using
volume rendering. For a stochastic opaque solid characterized
by its occupancy O and vacancy v, i.e., 1 −O, the authors derive
the attenuation coefficient 𝜎of the object as follows:
𝜎(x,𝜔) = |𝜔· ∇𝑙𝑜𝑔(v(x))| = |𝜔· ∇v(x)|
v(x)
,
(3)
where 𝜔is the viewing direction and x is the 3D position. With
this attenuation coefficient, they derive the volume rendering
for a stochastic solid as,
C =
∫𝑡𝑓
𝑡𝑛
𝑝(𝑡)c(x(𝑡),𝜔) 𝑑𝑡,
𝑝(𝑡) = 𝑇(𝑡)𝜎(x(𝑡),𝜔),
𝑇(𝑡) = 𝑒𝑥𝑝

−
∫𝑡
𝑡𝑛
𝜎(x(𝑠),𝜔) 𝑑𝑠

,
(4)
where 𝑝is the free-flight distribution [Miller et al. 2024] that
represents the statistical distribution of the distances that the
light travels before collision and serves as the weight for color
integration, and 𝑇(𝑡) is the transmittance along the ray.
In our work, we regard a 3D Gaussian primitive as a stochas-
tic solid and design an appropriate attenuation coefficient 𝜎
for it. With this coefficient, the volume rendering of a Gaussian
primitive, as described in Equation 4, is equivalent to its raster-
ized rendering. This enables us to study Gaussian Splatting in
a more principled manner and develop a shape reconstruction
method for Gaussian primitives.
4
Method
In the following sections, we first introduce our method for a
single Gaussian primitive. We then design an efficient method
for rendering depth maps from multiple Gaussian primitives.
4.1
Gaussian Primitives as Stochastic Solids
We treat a Gaussian primitive as a stochastic solid [Miller et al.
2024] and derive its rendering function. As shown in Figure 3,
we prove that, with a proper attenuation coefficient 𝜎, the
volume rendering of this stochastic Gaussian solid is equivalent
to the rasterization rendering of the original Gaussian Splatting.
Specifically, the opacity 𝛼for a pixel in Equation 2 corresponds
to the maximum value of the Gaussian function along that
pixel’s view ray (as proved in the supplementary). Therefore,
the rendered color of a single Gaussian is given by:
C = c𝛼= c𝐺(𝑡∗),
(5)
where 𝑡∗is the maximum point along the ray 𝑙: o + 𝜔𝑡, and we
denote 𝐺(o + 𝜔𝑡∗) by 𝐺(𝑡∗) for simplification.
Equation 5 cannot uniquely determine the attenuation co-
efficient. So, we impose three additional constraints. Given a
Gaussian primitive 𝐺(x), we assume that
– When 𝐺(x1) ≥𝐺(x2), it follows that o(x1) ≥o(x2), indi-
cating that positions closer to the Gaussian center have
higher occupancy;

<!-- page 4 -->
4
•
Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan
Gaussian
!(#)
Projection by 
Gaussian Splatting
Volume rendering with 
stochastic solid theory
Identical
Identical
Integrate %&
 along the ray
%&
density & 
along the ray
stochastic solid 
theory
stochastic solid
v =
1 −!(#)
stochastic
solid
Fig. 3. Given a Gaussian primitive, we regard it as a stochastic solid and derive an appropriate attenuation coefficient with Equation 3. With our attenuation
coefficient, the volume rendering of this stochastic solid is equivalent to the rasterization rendering developed in the original Gaussian Splatting.
– The occupancy of the solid approaches 0 when x is far
from the Gaussian center;
– The occupancy o(x) is differentiable from x.
This leads us to derive a straightforward and unique expression
for the vacancy:
v(x) =
√︁
1 −𝐺(x).
(6)
To prove Equation 6, we first derive the volume rendering
of a Gaussian primitive following the method in [Miller et al.
2024] as,
C =
∫∞
−∞
𝑇(𝑡)𝜎(x(𝑡),𝜔)c𝑑𝑡= c(1 −v(𝑡∗)2),
(7)
where the attenuation coefficient 𝜎originates from the stochas-
tic solid as described in Equation 3, resulting in the integral
in terms of vacancy v(𝑡∗). Compared with the rasterization in
Equation 5 and the volumetric rendering in Equation 7, we can
obtain,
C = c𝐺(𝑡∗) = c(1 −v(𝑡∗)2),
(8)
In other words, we derive the following condition,
v(𝑡∗) =
√︁
1 −𝐺(𝑡∗).
(9)
Therefore, a stochastic Gaussian solid can produce the same
rendering results as the rasterization in Gaussian Splatting if
its vacancy adheres to Equation 6. The proof of uniqueness and
other details can be found in our supplementary materials. Now,
we can use Equation 3 and Equation 6 to obtain attenuation
coefficients 𝜎inside a Gaussian primitive, allowing us to obtain
accurate depth maps and smooth optimization.
This property lets us move beyond heuristic geometry read-
outs, leading to a principled shape reconstruction approach
built directly on Gaussian primitives. In the following sections,
we apply this theory to Gaussian Splatting and demonstrate
that it substantially improves shape reconstruction.
4.2
Depth from Stochastic Solids
In Gaussian Splatting, photometric supervision alone is insuf-
ficient to reconstruct high-quality shapes. To better recover
surface geometry, recent works [Chen et al. 2024; Guédon
and Lepetit 2024; Huang et al. 2024] render depth maps from
Gaussian primitives and add geometric regularizers, and then
backpropagate their gradients to the Gaussian parameters.
Nevertheless, the rendered depth maps are noisy and have
poor cross-view consistency, e.g., as shown in Figure 8 and 4,
providing weak geometric supervision. This motivates us to
improve depth rendering in Gaussian Splatting by utilizing
attenuation coefficients derived from stochastic solids.
The rendering pipeline is shown in Figure 2. We first derive
our depth computation method, then show that it improves
multi-view consistency and produces cleaner depth maps.
4.2.1
Depth definition. Following prior Gaussian Splatting meth-
ods, we use the median depth 𝑡𝑚𝑒𝑑for geometric regularization:
𝑡𝑚𝑒𝑑= 𝑇−1(0.5),
(10)
where 𝑇−1(∗) is the inverse function of the transmittance 𝑇(𝑡).
Following prior work [Blanc et al. 2025a,b; Condor et al. 2025],
we assume that the events of a view ray intersecting different
Gaussians are statistically independent. Under this assumption,
the overall transmittance at 𝑡along the ray is the product of
the transmittance calculated at each Gaussian primitive as,
𝑇(𝑡) =
Ö
𝑖
𝑇𝑖(𝑡) ,
(11)
where 𝑇𝑖(𝑡) is the transmittance of the 𝑖-th Gaussian as:
𝑇𝑖(𝑡) =
(
v𝑖(𝑡),
𝑡≤𝑡∗
𝑖
v𝑖(𝑡∗
𝑖)2/v𝑖(𝑡),
𝑡> 𝑡∗
𝑖.
(12)
Here, 𝑡∗
𝑖is the Gaussian’s maximum point along the camera ray.
Equation 12 is derived from the continuous attenuation profile
within each Gaussian as defined in Equation 3, capturing more
detailed geometry information. The derivation of Equation 12
can be found in the supplementary material.
Discussion. Previous methods estimate depth either from
per-view depth planes [Yu et al. 2024c; Zhang et al. 2024] that
are view-dependent by design, or via opacity-weighted ray av-
eraging [Chen et al. 2024] that is easily biased by view-specific
floaters. These depth extraction strategies often lead to poor
cross-view consistency. In contrast, we will show that interpret-
ing Gaussian Splatting as a stochastic solid yields a median
depth estimate with strong multi-view consistency. Recall that
the median depth is the point where the transmittance first
reaches a fixed threshold, i.e.,𝑇= 0.5. From Equations 12 and 11,
if the overall transmittance crossing 𝑇= 0.5 occurs before the

<!-- page 5 -->
Geometry-Grounded Gaussian Splatting
•
5
Viewport
Ours
PGSR
GOF
Fig. 4. Depth maps of Gaussian Splatting-based methods. We visualize the depth maps by converting them to 3D points. Our method produces a clean
and smooth depth map. PGSR uses expected depth, yielding much noise at edges. GOF uses median depth and suffers from unsmooth depth changes.
(b) Median depth
(c) Expected depth
(a) Depth capture
! = 0.5
Fig. 5. Illustration of depth rendering methods. (a) The green plane’s
opacity 𝛼decreases smoothly from 1 on the right to 0 on the left. Conse-
quently, (b) the median depth changes in a step-like manner, whereas (c)
the expected depth varies continuously.
!!"#
0.5
%/v
!
v ( = v$ ( v% ( v& (
% ! = %$ ! %% ! %& !
!
1D profile of 3D function v "  
along the camera ray
v !
% !
0
Fig. 6. Illustration of vacancy along the camera ray. The vacancy value
on the ray equals the transmittance on the front side of the Gaussians.
peak of the contributing Gaussians, then the transmittance co-
incides with the 3D vacancy field as illustrated in Figure 6. So,
the depth map is a view-independent 0.5-level isosurface. This
regime is common because optimization clusters high-opacity
Gaussians near the surface, making the transmittance drop
mainly on their near sides. While floaters can still perturb the
ray, the median depth is more robust to such outliers than the
alpha-averaged expected depth, leading to stronger multi-view
consistency.
Beyond improved multi-view consistency, our method pro-
duces cleaner depth maps as shown in Figure 4. Depth obtained
via alpha-weighted compositing tends to interpolate between
foreground and background at boundaries, leading to blurred
silhouettes as shown in Figure 5. Median depth, defined by
the 𝑇= 0.5 crossing, gives sharper boundary transitions. How-
ever, in prior Gaussian Splatting formulations, transmittance
is updated in discrete steps, so the 0.5 crossing often snaps
to a single Gaussian; neighboring pixels may therefore select
different Gaussians, producing jagged artifacts. Our stochastic-
solid formulation models attenuation continuously within
each Gaussian, yielding a smooth transmittance function and
reducing staircasing while preserving sharp boundaries.
4.2.2
Implementation. In general, Equation 10 does not admit
a closed-form solution. To address this, we exploit the mono-
tonicity of transmittance along each ray and use an iterative
binary search to find the median depth. During backpropaga-
tion, we do not require an iterative search. Instead, we derive
the closed-form solution for the gradient of depth 𝑡𝑚𝑒𝑑with
respect to the Gaussians’ parameters as,
𝜕𝑡𝑚𝑒𝑑
𝜕𝜃
= −𝜕𝑇(𝑡𝑚𝑒𝑑;𝜃)
𝜕𝜃
/ 𝜕𝑇(𝑡;𝜃)
𝜕𝑡

𝑡=𝑡𝑚𝑒𝑑,
(13)
where 𝜃denotes the Gaussian parameters along the ray.
Equation 13 shows that the gradient can be distributed to all
contributing Gaussians along the ray, unlike previous methods
where the gradient of the median depth was only applied to a
single Gaussian. This stems from our stochastic-solid formula-
tion, which yields a differentiable transmittance function. As a
result, the median depth 𝑡𝑚𝑒𝑑varies smoothly with the Gauss-
ian parameters, providing denser supervision for optimization.
The derivation of Equation 13 and implementation details are
provided in the supplementary material.
4.3
Optimization with Stochastic Solids
We optimize scenes using photometric loss [Kerbl et al. 2023],
normal consistency loss [Huang et al. 2024], and multi-view
regularization [Chen et al. 2024]; details are provided in the
supplementary material. These losses require rendering RGB
images, normal maps, and depth maps. Fully volumetric ren-
dering for all modalities is computationally expensive [Blanc
et al. 2025a,b; Condor et al. 2025]. We therefore retain the
standard Gaussian Splatting approximation for RGB and nor-
mals [Zhang et al. 2024], while computing depth using Eq. 10.
Experiments show that this setting can significantly improve
the shape reconstruction accuracy of Gaussian Splatting, while
maintaining the efficiency. Nevertheless, we believe that extend-
ing our volumetric formulation to RGB and normal rendering
can further improve accuracy, which we leave for future work.
5
Experiments
We evaluate our method on several public datasets and compare
it with existing state-of-the-art methods.

<!-- page 6 -->
6
•
Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan
Table 1. Quantitative comparison on the DTU dataset [Jensen et al. 2014]. We report Chamfer Distance and average optimization time for different methods.
Among explicit Gaussian Splatting–based approaches, our method achieves the best results and attains accuracy comparable to GeoSVR. All Gaussian Splatting
methods are evaluated using half-resolution images.
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
implicit
NeRF [Mildenhall et al. 2020]
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
> 12h
VolSDF [Yariv et al. 2021]
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
> 12h
NeuS [Wang et al. 2021]
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
> 12h
Neuralangelo [Li et al. 2023]
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
> 12h
explicit
3D GS [Kerbl et al. 2023]
2.14
1.53
2.08
1.68
3.49
2.21
1.43
2.07
2.22
1.75
1.79
2.55
1.53
1.52
1.50
1.96
7.8m
2D GS [Huang et al. 2024]
0.48
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
11.3m
GOF [Yu et al. 2024c]
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
52m
3DGSR [Lyu et al. 2024]
0.44
0.96
0.40
0.36
1.02
0.80
0.64
1.20
1.08
0.97
0.54
0.72
0.37
0.52
0.42
0.70
RaDe-GS [Zhang et al. 2024]
0.43
0.75
0.35
0.37
0.81
0.74
0.74
1.19
1.20
0.65
0.61
0.84
0.35
0.66
0.46
0.68
8.2m
GFSGS [Jiang et al. 2025]
0.40
0.59
0.39
0.38
0.72
0.59
0.65
1.08
0.93
0.59
0.50
0.67
0.34
0.47
0.40
0.58
16.8m
PGSR [Chen et al. 2024]
0.34
0.54
0.44
0.37
0.78
0.57
0.49
1.06
0.63
0.59
0.47
0.50
0.30
0.37
0.34
0.52
30.5m
GeoSVR [Li et al. 2025]
0.32
0.51
0.30
0.33
0.71
0.48
0.42
1.03
0.62
0.56
0.33
0.46
0.30
0.34
0.32
0.47
53.3m
Ours (20k)
0.38
0.50
0.27
0.31
0.80
0.43
0.42
1.04
0.64
0.52
0.31
0.56
0.30
0.31
0.33
0.47
15.0m
Ours (30k)
0.37
0.50
0.27
0.31
0.81
0.43
0.42
1.05
0.64
0.52
0.32
0.58
0.30
0.31
0.33
0.48
25.3m
Table
2. Quantitative
comparison
on
the
Tanks
&
Temples
Dataset [Knapitsch et al. 2017]. We report the F1-score and average
optimization time.
Barn
Cat.
Cour.
Igna.
Meet.
Truc.
Mean
Time
implicit
NeuS
0.29
0.29
0.17
0.83
0.24
0.45
0.38
>24h
Geo-NeuS
0.33
0.26
0.12
0.72
0.20
0.45
0.35
>24h
Neurlangelo
0.70
0.36
0.28
0.89
0.32
0.48
0.50
>24h
explicit
2D GS
0.36
0.23
0.13
0.44
0.16
0.26
0.30
15.5m
GOF
0.51
0.41
0.28
0.68
0.28
0.59
0.46
71.6m
RaDe-GS
0.49
0.36
0.27
0.72
0.27
0.61
0.45
12.1m
PGSR
0.66
0.44
0.20
0.81
0.33
0.66
0.52
42.9m
GeoSVR
0.68
0.49
0.34
0.83
0.37
0.66
0.56
66.4m
Ours
0.70
0.56
0.38
0.81
0.42
0.70
0.60
32.1m
Implementation Details. We use a local affine approximation
and adopt RaDe-GS [Zhang et al. 2024] to estimate each Gauss-
ian peak 𝑡∗
𝑖. For efficiency, we follow gsplat [Ye et al. 2025] and
use warp-level reductions for gradient accumulation. We apply
the 3D filtering from Mip-Splatting [Yu et al. 2024a] (without
its 2D filter), the densification strategy from GOF [Yu et al.
2024c], and the exposure compensation from PGSR [Chen et al.
2024]. Multi-view regularization is implemented with a custom
CUDA kernel. We will release our code.
Datasets. We evaluate reconstruction accuracy on DTU [Jensen
et al. 2014] and Tanks & Temples (TnT) [Knapitsch et al. 2017].
Following prior work, we use the standard 15-scene DTU
split and the common 6-scene TnT subset. We report Chamfer
Distance on DTU and F1-score on TnT.
Mesh Extraction. Following previous works [Yu et al. 2024c;
Zhang et al. 2024], we apply the TSDF fusion [Curless and
Levoy 1996] implemented by Open3D [Zhou et al. 2018] to
extract meshes for the DTU dataset and adopt Marching Tetra-
hedra [Guédon et al. 2025a; Yu et al. 2024c] for large-scale
scenes in the Tanks & Temples dataset. Inspired by GOF, we
define an indicator function over 3D space for Marching Tetra-
hedra. Specifically, a point is classified as inside the mesh if it
is occluded in any training view, i.e., if its transmittance falls
below 0.5; otherwise, it is classified as outside.
5.1
Reconstruction Comparison
We compare our method against existing state-of-the-art meth-
ods in the shape reconstruction task. Table 1 and Table 2 show
the accuracy on the DTU and TnT datasets. The multi-view
regularizer adopted in PGSR and GeoSVR substantially boosts
DTU accuracy; using this regularization, our method achieves
comparable performance to both. In TnT, our method signifi-
cantly outperforms existing Gaussian Splatting-based methods
because of our depth-rendering formulation, which enables
finer geometric details, enforces view-consistent geometry, and
is robust to floaters. Figure 7 provides qualitative comparisons
among shape reconstruction methods. Our method recon-
structs finer details with more accurate geometry. Additional
qualitative results are shown in Figure 9 and Figure 10.
We report runtimes in Table 1 and Table 2. For the same
number of iterations, our method is faster than GeoSVR (15 vs.
53 min.) and PGSR (25 vs. 30 min.), thanks to a more efficient
implementation of multi-view regularization. Our runtime
is higher than the fastest baselines due to the added cost of
the binary search and the multi-view term. We expect further
speedups by tightening the initial depth interval of the binary
search, which we leave for future work.
5.2
Multi-view Consistency
Geometric consistency across views is essential for accurate
shape reconstruction. To evaluate each depth-rendering method,
we compute per-pixel cycle reprojection error during training.
For a reference and neighboring view, we render depth maps
of the reference view, back-project pixels to 3D, project into the
neighbor to sample the corresponding depth, then back-project
and reproject to the reference. The cycle error is the Euclidean
distance between the original and reprojected pixel locations.
We compare our method with PGSR [Chen et al. 2024]
and RaDe-GS [Zhang et al. 2024]. PGSR defines the ray-surface
intersection using a plane orthogonal to the shortest axis of each
3D Gaussian, whereas RaDe-GS uses the ray-wise maximizer
of the Gaussian response. For a fair comparison, we evaluate
RaDe-GS augmented with the same multi-view regularization
used in PGSR, and enable geometric regularization at 7K

<!-- page 7 -->
Geometry-Grounded Gaussian Splatting
•
7
Ground Truth
Ours
GeoSVR
PGSR
Fig. 7. Qualitative comparison on Tanks & Temples [Knapitsch et al. 2017] dataset. We compare our method with GeoSVR and PGSR. Our method
reconstructs plausible meshes with finer geometric details
15K
8K
7K
Neighboring 
view
Reference
view
(c)
(b)
(a)
Color map
0
1
0.2
0.6
30K
Fig. 8. Cycle reprojection error per iteration. We visualize the cycle reprojection error between a reference view and its nearest neighboring view
throughout optimization. Our method (a) attains lower errors on foreground regions and achieves full coverage faster than the other methods. PGSR (b) relies
on planar-based depth accumulation, which leads to weaker multi-view consistency than our volumetric formulation and suffers from noticeable floaters. We
add the multi-view regularization to RaDe-GS (c) and train it using expected depth. Zoomed-in patches are shown on the right.
iterations for all methods. As shown in Figure 8, our method
yields a better initialization and converges faster, achieving
the lowest reprojection error at 30K iterations, mainly due to
our depth formulation based on stochastic theory. In contrast,
the other methods exhibit noticeably larger reprojection errors,
and floaters that arise during training further exacerbate the
inconsistency. More results can be found in Figure 13.
5.3
Ablation Study
In this section, we evaluate the contribution of each component
when integrated into our method. Table 3 reports quantitative
results on the TnT dataset. The geometric multi-view term
𝐿𝑔𝑐penalizes cycle reprojection error; however, it brings only
marginal gains in our setting, because our depth-rendering
formulation already provides strong multi-view consistency.
In contrast, the normal consistency loss and the exposure
compensation module consistently improve reconstruction
quality. Finally, compared with the other two depth-rendering
baselines equipped with similar regularizers, our method
achieves a more accurate shape reconstruction.
Table 3. Ablation study on Tanks & Temples [Knapitsch et al. 2017].
The normal consistency loss and the single-view geometric loss in PGSR
have a similar formulation. We denote them as 𝐿𝑛. 𝐿𝑔𝑐is the geometric
consistency loss. ‘exposure’ represents the exposure compensation module
from PGSR. We toggle each term on/off (✓/–) and report the resulting
reconstruction accuracy.
PGSR
RaDe-GS
Ours
𝐿𝑔𝑐
✓
✓
–
✓
✓
✓
𝐿𝑛
✓
✓
✓
–
✓
✓
exposure
✓
✓
✓
✓
–
✓
F1-score
0.52
0.52
0.60
0.57
0.59
0.60
6
Conclusion
We reveal the intrinsic geometry for Gaussian Splatting. We
regard Gaussian primitives as stochastic solids and design
an appropriate attenuation function to make their volume
rendering identical to their rasterization-based rendering. The
stochastic theory enables depth rendering in a principled
manner. Experiments show that our method outperforms state-
of-the-art methods.

<!-- page 8 -->
8
•
Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan
References
Ramazzina Andrea, Bĳelic Mario, Walz Stefanie, Sanvito Alessandro, Scheuble
Dominik, and Heide Felix. 2023. ScatterNeRF: Seeing Through Fog with
Physically-Based Inverse Neural Rendering. The IEEE International Conference
on Computer Vision (ICCV).
Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo
Martin-Brualla, and Pratul P. Srinivasan. 2021. Mip-NeRF: A Multiscale
Representation for Anti-Aliasing Neural Radiance Fields. ICCV (2021).
Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter
Hedman. 2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance
Fields. CVPR (2022).
Hugo Blanc, Jean-Emmanuel Deschaud, and Alexis Paljic. 2025a. Raygauss:
Volumetric gaussian-based ray casting for photorealistic novel view synthesis.
In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).
IEEE, 1808–1817.
Hugo Blanc, Jean-Emmanuel Deschaud, and Alexis Paljic. 2025b. RayGaussX:
Accelerating Gaussian-Based Ray Marching for Real-Time and High-Quality
Novel View Synthesis. In Proceedings of the IEEE/CVF International Conference
on Computer Vision. 27575–27584.
Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Latif, Davide Scaramuzza,
José Neira, Ian Reid, and John J Leonard. 2017. Past, present, and future of
simultaneous localization and mapping: Toward the robust-perception age.
IEEE Transactions on robotics 32, 6 (2017), 1309–1332.
Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weĳian Xie, Shangjin Zhai, Nan
Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. 2024. Pgsr: Planar-based
gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE
Transactions on Visualization and Computer Graphics (2024).
Hanlin Chen, Chen Li, and Gim Hee Lee. 2023. Neusg: Neural implicit sur-
face reconstruction with 3d gaussian splatting guidance.
arXiv preprint
arXiv:2312.00846 (2023).
Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic, Simon Green, Piotr
Didyk, and Adrian Jarabo. 2025. Don’t Splat your Gaussians: Volumetric
Ray-Traced Primitives for Modeling and Rendering Scattering and Emissive
Media. ACM Trans. Graph. (Jan 2025). https://doi.org/10.1145/3711853
Brian Curless and Marc Levoy. 1996. A volumetric method for building complex
models from range images. In Proceedings of the 23rd annual conference on
Computer graphics and interactive techniques. 303–312.
Jakob Engel, Thomas Schöps, and Daniel Cremers. 2014. LSD-SLAM: Large-scale
direct monocular SLAM. In European conference on computer vision. Springer,
834–849.
Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang,
et al. 2024. Lightgaussian: Unbounded 3d gaussian compression with 15x
reduction and 200+ fps. Advances in neural information processing systems 37
(2024), 140138–140158.
Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht,
and Angjoo Kanazawa. 2022. Plenoxels: Radiance fields without neural
networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 5501–5510.
Antoine Guédon, Diego Gomez, Nissim Maruani, Bingchen Gong, George
Drettakis, and Maks Ovsjanikov. 2025a. MILo: Mesh-In-the-Loop Gaussian
Splatting for Detailed and Efficient Surface Reconstruction. ACM Transactions
on Graphics (2025). https://anttwo.github.io/milo/
Antoine Guédon, Tomoki Ichikawa, Kohei Yamashita, and Ko Nishino. 2025b.
MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photore-
alism From Sparse Views. In CVPR. 6001–6011.
Antoine Guédon and Vincent Lepetit. 2024. Sugar: Surface-aligned gaussian
splatting for efficient 3d mesh reconstruction and high-quality mesh rendering.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 5354–5363.
Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024.
2D Gaussian Splatting for Geometrically Accurate Radiance Fields. (2024).
https://doi.org/10.1145/3641519.3657428
Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs.
2014. Large scale multi-view stereopsis evaluation. In Proceedings of the IEEE
conference on computer vision and pattern recognition. 406–413.
Kaiwen Jiang, Venkataram Sivaram, Cheng Peng, and Ravi Ramamoorthi. 2025.
Geometry Field Splatting with Gaussian Surfels. In Proceedings of the Computer
Vision and Pattern Recognition Conference. 5752–5762.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM
Transactions on Graphics 42, 4 (July 2023). https://repo-sam.inria.fr/fungrap
h/3d-gaussian-splatting/
Shakiba Kheradmand, Delio Vicini, George Kopanas, Dmitry Lagun, Kwang Moo
Yi, Mark Matthews, and Andrea Tagliasacchi. 2025. StochasticSplats: Stochastic
Rasterization for Sorting-Free 3D Gaussian Splatting. In Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV).
Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and
Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Transactions
on Graphics 36, 4 (2017).
Deborah Levy, Amit Peleg, Naama Pearl, Dan Rosenbaum, Derya Akkaynak,
Simon Korman, and Tali Treibitz. 2023. SeaThru-NeRF: Neural Radiance Fields
in Scattering Media. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. 56–65.
Jiahe Li, Jiawei Zhang, Youmin Zhang, Xiao Bai, Jin Zheng, Xiaohan Yu, and Lin
Gu. 2025. GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface
Reconstruction. Advances in Neural Information Processing Systems (2025).
Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H Taylor, Mathias Unberath,
Ming-Yu Liu, and Chen-Hsuan Lin. 2023. Neuralangelo: High-fidelity neural
surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. 8456–8465.
Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi
Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, and Wenming Yang. 2024. Vast-
Gaussian: Vast 3D Gaussians for Large Scene Reconstruction. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
5166–5175.
Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen,
Jiangmiao Pang, and Xiaojuan Qi. 2024. 3dgsr: Implicit surface reconstruction
with 3d gaussian splatting. ACM Transactions on Graphics (TOG) 43, 6 (2024),
1–12.
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi
Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural
Radiance Fields for View Synthesis. In ECCV.
Bailey Miller, Hanyu Chen, Alice Lai, and Ioannis Gkioulekas. 2024. Objects as
Volumes: A Stochastic Geometry View of Opaque Solids. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
87–97.
Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022.
Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.
ACM Trans. Graph. 41, 4, Article 102 (July 2022), 15 pages. https://doi.org/10
.1145/3528223.3530127
Michael Oechsle, Songyou Peng, and Andreas Geiger. 2021. Unisurf: Unifying
neural implicit surfaces and radiance fields for multi-view reconstruction. In
Proceedings of the IEEE/CVF international conference on computer vision. 5589–
5599.
Aron Schmied, Tobias Fischer, Martin Danelljan, Marc Pollefeys, and Fisher Yu.
2023. R3D3: Dense 3D Reconstruction of Dynamic Scenes from Multiple
Cameras. In Proceedings of the IEEE International Conference on Computer Vision.
Noah Snavely, Steven M Seitz, and Richard Szeliski. 2006.
Photo tourism:
exploring photo collections in 3D. In ACM siggraph 2006 papers. 835–846.
Cheng Sun, Jaesung Choe, Charles Loop, Wei-Chiu Ma, and Yu-Chiang Frank
Wang. 2025. Sparse Voxels Rasterization: Real-time High-fidelity Radiance
Field Rendering. In CVPR.
Yunkai Tang, Chengxuan Zhu, Renjie Wan, Chao Xu, and Boxin Shi. 2024. Neural
Underwater Scene Representation. CVPR.
Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and
Wenping Wang. 2021. NeuS: Learning Neural Implicit Surfaces by Volume
Rendering for Multi-view Reconstruction. NeurIPS (2021).
Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume rendering
of neural implicit surfaces. Advances in neural information processing systems 34
(2021), 4805–4815.
Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan,
Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa.
2025. gsplat: An open-source library for Gaussian splatting. Journal of Machine
Learning Research 26, 34 (2025), 1–17.
Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. 2024b.
Gsdf: 3dgs meets sdf for improved neural rendering and reconstruction.
Advances in Neural Information Processing Systems 37 (2024), 129507–129530.
Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024a.
Mip-Splatting: Alias-free 3D Gaussian Splatting. Conference on Computer Vision
and Pattern Recognition (CVPR) (2024).
Zehao Yu, Torsten Sattler, and Andreas Geiger. 2024c. Gaussian opacity fields: Ef-
ficient adaptive surface reconstruction in unbounded scenes. ACM Transactions
on Graphics (ToG) 43, 6 (2024), 1–13.
Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and
Ping Tan. 2024. RaDe-GS: Rasterizing Depth in Gaussian Splatting. arXiv
preprint arXiv:2406.01467 (2024).
Ziyu Zhang, Binbin Huang, Hanqing Jiang, Liyang Zhou, Xiaojun Xiang, and
Shuhan Shen. 2025a. Quadratic Gaussian splatting: High quality surface
reconstruction with second-order geometric primitives. In ICCV. 28260–28270.
Ziyu Zhang, Binbin Huang, Hanqing Jiang, Liyang Zhou, Xiaojun Xiang, and
Shuhan Shen. 2025b. Quadratic Gaussian Splatting: High Quality Surface

<!-- page 9 -->
Geometry-Grounded Gaussian Splatting
•
9
Reconstruction with Second-order Geometric Primitives. In Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV). 28260–28270.
Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. 2018. Open3D: A Modern Library
for 3D Data Processing. arXiv:1801.09847 (2018).

<!-- page 10 -->
10
•
Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan
Fig. 9. Qualitative results on the DTU [Jensen et al. 2014] dataset.
Fig. 10. Qualitative results on the Tanks & Temples [Knapitsch et al. 2017] dataset.
Viewport
Ours
PGSR
RaDe-GS
GeoSVR
2DGS
GOF
Fig. 11. Qualitative comparison of depth rendering among our method and prior methods. We visualize depth maps by back-projecting them into 3D
point clouds. RaDe-GS uses multi-view regularization and expected depth; 2DGS uses expected depth.

<!-- page 11 -->
Geometry-Grounded Gaussian Splatting
•
11
Ground truth
Ours
PGSR
GeoSVR
Fig. 12. Qualitative comparison of novel view synthesis among our method and prior methods on the Mip-NeRF360 [Barron et al. 2022] dataset.
15K
8K
7K
Neighboring 
view
Reference
view
(c)
(b)
(a)
30K
15K
8K
7K
Neighboring 
view
Reference
view
(c)
(b)
(a)
Color map
0
1
0.2
0.6
30K
Color map
0
1
0.2
0.6
Fig. 13. Cycle reprojection error per iteration. We visualize the cycle reprojection error between a reference view and its nearest neighboring view
throughout optimization. We show the projection error of (a) our method, (b) PGSR, and (c) RaDe-GS with multi-view regularization. Zoomed-in patches are
shown on the right.

<!-- page 12 -->
12
•
Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan
Supplementary: Geometry-Grounded Gaussian Splatting
A
Implementation of Depth Rendering
In this section, we detail the implementation of the forward
and backward passes for depth rendering.
A.1
Forward Pass
We begin with the initial median depth 𝑡𝑖𝑛𝑖𝑡obtained from
RaDe-GS [Zhang et al. 2024]. We then establish an initial depth
interval [𝑡𝑖𝑛𝑖𝑡−𝑟, 𝑡𝑖𝑛𝑖𝑡+ 𝑟] and search for the median depth
within this range, setting r to 0.4 during training. To perform
the binary search, we need to traverse the Gaussians along
the camera ray and record the transmittance at the mid-point,
comparing it to the target value of 0.5. However, traversing
Gaussians can be time-consuming, so we aim to reduce the
number of Gaussian traversals. Specifically, instead of splitting
the interval into two segments by a midpoint, we evenly divide
it into eight segments using seven segment points and record
the transmittance at each segment point. After each traversal,
we locate the segment whose endpoint transmittance values
fall on opposite sides of 0.5 and use it as the new search
interval. Under this setting, a single traversal is equivalent to
three binary-search iterations. We repeat this process 5 times,
gradually narrowing the interval until the final depth error is
within 0.8 × 8−5 = 2.441 × 10−5. In the first pass, we also record
the transmittance values at both ends of the interval, i.e., 𝑡𝑖𝑛𝑖𝑡−𝑟,
𝑡𝑖𝑛𝑖𝑡+ 𝑟. If both values are above 0.5 or below 0.5, we mask the
pixel for geometric regularization.
A.2
Backward Pass
The backpropagation of depth can be calculated as,
𝜕𝐿
𝜕𝑡𝑚𝑒𝑑
· 𝜕𝑡𝑚𝑒𝑑
𝜕𝜃
,
(14)
where 𝐿denotes the loss, 𝑡𝑚𝑒𝑑is the median depth computed
in the forward pass, and 𝜃represents the parameters of the
Gaussians along the camera ray. We further write 𝜃= {𝜃𝑖},
where 𝜃𝑖denotes the parameters of the 𝑖-th Gaussian. The
first term 𝜕𝐿/𝜕𝑡𝑚𝑒𝑑is the input of the backward function, and
we need to calculate the second term. The formulation of the
second term is shown in Equation 13 of the main paper and
will be derived in section I. It is composed of 𝜕𝑇(𝑡;𝜃)/𝜕𝑡|𝑡=𝑡𝑚𝑒𝑑
and 𝜕𝑇(𝑡𝑚𝑒𝑑;𝜃)/𝜕𝜃. We traverse the Gaussians along the ray
twice, computing the two terms in separate passes.
In the first pass, we calculate 𝜕𝑇(𝑡;𝜃)/𝜕𝑡|𝑡=𝑡𝑚𝑒𝑑. Equation 10
and Equation 11 of the main paper show that,
𝑇(𝑡𝑚𝑒𝑑;𝜃) =
Ö
𝑖
𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖) = 0.5.
(15)
We can obtain:
𝜕𝑇(𝑡;𝜃)
𝜕𝑡
|𝑡=𝑡𝑚𝑒𝑑=
∑︁
𝑖
∑︁
𝑗≠𝑖
𝑇𝑗(𝑡𝑚𝑒𝑑;𝜃𝑗) 𝜕𝑇𝑖(𝑡;𝜃𝑖)
𝜕𝑡
|𝑡=𝑡𝑚𝑒𝑑
=
∑︁
𝑖
𝑇(𝑡𝑚𝑒𝑑;𝜃)
𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖)
𝜕𝑇𝑖(𝑡;𝜃𝑖)
𝜕𝑡
|𝑡=𝑡𝑚𝑒𝑑
=
∑︁
𝑖
0.5
𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖)
𝜕𝑇𝑖(𝑡;𝜃𝑖)
𝜕𝑡
|𝑡=𝑡𝑚𝑒𝑑.
(16)
After computing Equation 16, we modify the standard Gauss-
ian Splatting color-accumulation backward pass to additionally
compute 𝜕𝑇(𝑡𝑚𝑒𝑑;𝜃)/𝜕𝜃𝑖for each Gaussian.
𝜕𝑇(𝑡;𝜃)
𝜕𝜃𝑖
=
∑︁
𝑗≠𝑖
𝑇𝑗(𝑡𝑚𝑒𝑑;𝜃𝑗) 𝜕𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖)
𝜕𝜃𝑖
= 𝑇(𝑡𝑚𝑒𝑑;𝜃)
𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖)
𝜕𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖)
𝜕𝜃𝑖
=
0.5
𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖)
𝜕𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖)
𝜕𝜃𝑖
.
(17)
The 𝜕𝑇𝑖(𝑡;𝜃𝑖)/𝜕𝑡|𝑡=𝑡𝑚𝑒𝑑in Equation 16 and 𝜕𝑇𝑖(𝑡𝑚𝑒𝑑;𝜃𝑖)/𝜕𝜃𝑖in
Equation 17 can be easily derived from the closed formed
formulation of 𝑇𝑖, i.e., Equation 12 of the main paper. Using
Equation 13 of the main paper, we plug Equation 16 and Equa-
tion 17 into Equation 14. The gradients are backpropagated to
each Gaussian as,
𝜕𝐿
𝜕𝑡𝑚𝑒𝑑
· 𝜕𝑡𝑚𝑒𝑑
𝜕𝜃𝑖
=
𝜕𝐿
𝜕𝑡𝑚𝑒𝑑
· 𝜕𝑇(𝑡𝑚𝑒𝑑;𝜃)
𝜕𝜃𝑖
/(−𝜕𝑇(𝑡;𝜃)
𝜕𝑡
|𝑡=𝑡𝑚𝑒𝑑).
(18)
B
Gaussian Splatting Preliminaries
Gaussian Splatting represents a scene as a set of 3D Gaussian
primitives. A single primitive is parameterized by a center
x𝑐∈R3, an opacity 𝑜∈[0, 1], and a symmetric positive definite
covariance Σ ∈R3×3,
𝐺(x) = 𝑜exp

−(x −x𝑐)⊤Σ−1(x −x𝑐)

.
(19)
From 3D to screen-space Gaussians. To render efficiently, Gauss-
ian Splatting rasterizes each 3D Gaussian as an elliptical 2D
Gaussian on the image plane. Let the camera extrinsics map
world coordinates to camera coordinates, and denote the world-
to-camera rotation as W. The covariance in the camera frame
is
Σ𝑐𝑎𝑚= WΣW⊤.
(20)
Let 𝜋(·) be the perspective projection and u𝑐= 𝜋(x𝑐,𝑐𝑎𝑚) be
the projected center on the image plane. Using a local affine
approximation of 𝜋around x𝑐,𝑐𝑎𝑚, the screen-space covariance
is obtained via multiplying with a Jacobian matrix:
Σ′
2𝐷= J Σ𝑐𝑎𝑚J⊤= JWΣW⊤J⊤,
(21)
where J ∈R2×3 is the Jacobian of the perspective projection
evaluated at x𝑐,𝑐𝑎𝑚.

<!-- page 13 -->
Geometry-Grounded Gaussian Splatting
•
13
Per-pixel alpha. Given a pixel location u ∈R2, the Gaussian
contributes an opacity (alpha) determined by its screen-space
ellipse:
𝛼(u) = 𝑜𝑒−(u−u𝑐)⊤Σ′−1
2𝐷(u−u𝑐),
(22)
where u𝑐is the projected Gaussian center. In practice, each
Gaussian is evaluated only on pixels within a finite screen-space
support to keep rasterization fast.
Alpha compositing. For each pixel, let 𝑁denote the set of
Gaussians whose projected support overlaps that pixel. These
Gaussians are depth-sorted and accumulated using standard
alpha blending. Denoting the per-Gaussian color as c𝑖and
𝛼𝑖= 𝛼𝑖(u), the rendered color is
C(u) =
∑︁
𝑖∈𝑁
c𝑖𝛼𝑖
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗).
(23)
C
Loss Functions
During training, we employ three loss terms: the photometric
loss from Gaussian Splatting [Kerbl et al. 2023], the normal
consistency loss from 2D GS [Huang et al. 2024], and the multi-
view regularization from PGSR [Chen et al. 2024]. We describe
each term below.
Photometric loss. We follow [Kerbl et al. 2023] and define the
photometric loss as a weighted combination of an 𝐿1 term and
a D-SSIM term between the rendered image and the ground-
truth image:
𝐿𝑐= (1 −𝜆) 𝐿1 + 𝜆𝐿𝑆𝑆𝐼𝑀,
(24)
where 𝜆is a hyperparameter.
Normal consistency. Photometric supervision alone is insuffi-
cient to constrain geometry, so we introduce additional geomet-
ric regularization. Specifically, we adopt the normal-consistency
loss from 2D GS [Huang et al. 2024], which encourages the
Gaussian normals to agree with the surface normal estimated
from the rendered depth map. Concretely, we compute a refer-
ence normal ˜n by applying finite differences to the depth map
and penalize its angular deviation from each Gaussian normal:
L𝑛=
∑︁
𝑖
𝜔𝑖
 1 −n⊤
𝑖˜n,
(25)
where n𝑖is the normal of the 𝑖-th Gaussian along the ray, and
𝜔𝑖= 𝛼𝑖
Î𝑖−1
𝑗=1(1 −𝛼𝑗) is its alpha-compositing weight.
Multi-view regularization. We adopt the multi-view regu-
larization of PGSR [Chen et al. 2024] to our method, which
combines a photometric term with an explicit geometric cycle-
consistency term. Concretely, for each reference pixel u𝑟, we
approximate the local surface as a plane using the rendered
depth and normal, and use the induced plane homography to
relate the reference view to a neighboring view:
H𝑟𝑛= K𝑛

R𝑟𝑛+ T𝑟𝑛n⊤
𝑟
p⊤𝑟n𝑟

K−1
𝑟,
(26)
where K𝑟and K𝑛are the intrinsics, (R𝑟𝑛, T𝑟𝑛) is the relative
pose from the reference to the neighboring camera, n𝑟is the
rendered normal at u𝑟, and p𝑟is the 3D point in the reference
camera frame obtained from the rendered depth along the ray
through u𝑟.
Using H𝑟𝑛, we warp the neighboring image into the reference
frame and enforce patch-level photometric consistency via
normalized cross-correlation (NCC):
𝐿𝑝𝑐=
∑︁
u𝑟
𝑤(u𝑟)

1 −NCC 𝐼𝑟(u𝑟), 𝐼𝑛(H𝑟𝑛u𝑟)
,
(27)
where 𝐼𝑟and 𝐼𝑛denote the reference and neighboring images.
To handle occlusions and unreliable correspondences, PGSR
defines a confidence weight from a forward–backward repro-
jection cycle. Specifically, letting H𝑛𝑟denote the homography
that maps from the neighboring view back to the reference
view, the cycle reprojection error is
𝜙(u𝑟) = ∥u𝑟−H𝑛𝑟H𝑟𝑛u𝑟∥2 ,
(28)
which is the same reprojection error introduced in the main
paper. The confidence is then
𝑤(u𝑟) =
(
exp   −𝜙(u𝑟),
𝜙(u𝑟) < 1,
0,
𝜙(u𝑟) ≥1,
(29)
thus discarding pixels with large cycle error.
In addition to the photometric term, PGSR directly penal-
izes the cycle reprojection error to encourage view-consistent
geometry:
𝐿𝑔𝑐=
∑︁
u𝑟
𝑤(u𝑟) 𝜙(u𝑟).
(30)
The overall multi-view regularization is
𝐿𝑚𝑣= 𝑤𝑝𝑐𝐿𝑝𝑐+ 𝑤𝑔𝑐𝐿𝑔𝑐,
(31)
where 𝑤𝑝𝑐and 𝑤𝑔𝑐control the relative strength of photometric
and geometric consistency.
Our final training loss L is,
L = L𝑐+ 𝑤𝑛L𝑛+ 𝐿𝑚𝑣.
(32)
We use 𝑤𝑛= 0.05, 𝑤𝑝𝑐= 0.6, 𝑤𝑔𝑐= 0.02, and set 𝜆= 0.2 in
Equation 24.
D
Comparison on Novel View Synthesis
We further compare novel view synthesis results across meth-
ods. Table 4 shows the quantitative results on the Mip-NeRF
360 dataset. Our method achieves competitive performance
compared with existing surface reconstruction baselines, while
RayGaussX produces the overall best rendering quality. To
isolate the effect of specular modeling, we augment our model
with the spherical Gaussian mixture used in RayGaussX, de-
noted as Ours (SG). Notably, we use these Gaussian lobes only
in this experiment; all other experiments use our default model
without Gaussian lobes. Figure 14 shows the qualitative results
of our method.
E
Limitation and Future Work
To maintain the efficiency of Gaussian Splatting, this paper
only considers the volumetric effects when rendering depth.
Future works can combine the stochastic theory with existing
volume rendering methods [Blanc et al. 2025b; Kheradmand

<!-- page 14 -->
14
•
Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan
Render
Ground truth
Render (SG)
Mesh
Mesh (SG)
Fig. 14. Qualitative results on the Mip-NeRF 360 dataset [Barron et al. 2022]. We visualize novel view synthesis and shape reconstruction results for our
method, and for our method augmented with the spherical Gaussian appearance model of RayGauss [Blanc et al. 2025a]. Incorporating spherical Gaussians
improves rendering quality in specular regions.
Table 4. Quantitative results on Mip-NeRF 360 dataset. The best scores
are highlighted with colors.
Outdoor Scene
Indoor Scene
PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓
NVS
NeRF
21.46
0.458
0.515
26.84
0.790
0.370
Deep Blending
21.54
0.524
0.364
26.40
0.844
0.261
Instant NGP
22.90
0.566
0.371
29.15
0.880
0.216
Mip-NeRF 360
24.47
0.691
0.283
31.72
0.917
0.180
3DGS
24.67
0.728
0.240
30.96
0.924
0.187
SVRaster
24.68
0.738
0.206
30.65
0.927
0.161
RayGaussX
25.24
0.761
0.167
32.43
0.943
0.146
Surface Recon.
BakedSDF
22.47
0.585
0.349
27.06
0.836
0.258
SuGaR
22.93
0.629
0.356
29.43
0.906
0.225
2DGS
24.34
0.717
0.246
30.40
0.916
0.195
GOF
24.82
0.750
0.202
30.79
0.924
0.184
VCR-GauS
24.31
0.707
0.280
30.53
0.921
0.184
PGSR
24.76
0.752
0.203
30.36
0.934
0.147
GeoSVR
24.83
0.738
0.218
30.46
0.921
0.172
Ours (SG)
24.97
0.754
0.200
32.18
0.938
0.150
Ours
25.09
0.760
0.196
31.02
0.934
0.154
et al. 2025] to fully utilize the volumetric nature of stochastic
when rendering color and normal maps. We believe this will
lead to further improvement in shape reconstruction.
To compute the median depth, our binary search is initialized
with a fixed depth interval. This interval must be sufficiently
wide, which increases the number of search steps and slows
training. For large-scale scenes, the true median depth may even
fall outside this preset range, hindering effective optimization.
We leave it to future work to develop adaptive bracketing
strategies that reliably locate the median and tighten the initial
interval, further accelerating depth rendering.
In Marching Tetrahedra, while the vertex placement is
Gaussian-aware, the subsequent 3D Delaunay triangulation
step remains general-purpose. In practice, reconstructing thin
or near-planar structures often requires dense vertices. De-
signing Gaussian-specific tetrahedralization and refinement
strategies is a meaningful direction for future work.
As we have bridged the gap between Gaussian and NeRF
reconstruction methods, future work could consider adopting
geometric regularization from NeRF-based methods, e.g., Neu-
ralangelo [Li et al. 2023], for Gaussian Splatting-based methods
to enhance shape quality.
F
Proof of the Equation 6 in the main paper
As shown in Figure 15, Gaussian Splatting [Kerbl et al. 2023]
applies a local affine approximation when projecting a Gaussian
primitive. As a result, the light rays from the camera center
are parallel to each other. The 3D Gaussian values on each ray
form a 1D Gaussian function, which is denoted as 𝐺𝑢𝑣(𝑡). It is
Observe 3D Gaussian
Gaussian value on 
the ray
𝑡𝑡∗
𝑡𝑡∗
𝑡𝑡∗
Fig. 15. Gaussian Splatting uses local affine approximation so that rays
are parallel to each other. Gaussian functions on the rays have the same
variance but different maximum values.
interesting to notice that these 1D Gaussians on different light
rays have the same variance but different maximum values.
Gaussian Splatting performs volume rendering on a single
primitive as:
𝐺′
2𝐷(𝑢, 𝑣) =
∫+∞
−∞
𝐺𝑢𝑣(𝑡) 𝑑𝑡,
(33)

<!-- page 15 -->
Geometry-Grounded Gaussian Splatting
•
15
which is proportional to the maximum value of 𝐺𝑢𝑣(𝑡) because
of the identical spread, shown in Figure 15.
Furthermore, as shown in Equation 22, Gaussian Splatting
normalizes the maximum value of the 2D Gaussian to opacity
𝑜, which aligns the maximum value between the 2D and the 3D
Gaussians, ensuring that the 2D Gaussian value matches the
maximum value of the corresponding 1D Gaussian along the
ray. This conclusion facilitates the derivation of the stochastic
Gaussian solid. While our result is derived from local affine
projection, our method can be easily extended to perspective
projection.
G
Proof of "Gaussians as Stochastic Solids"
Miller et al. [2024] propose a method to render stochastic
opaque solids by converting the vacancy to the attenuation
coefficient:
𝜎(x,𝜔) = |𝜔· ∇𝑙𝑜𝑔(v(x))| = |𝜔· ∇v(x)|
v(x)
,
(34)
where v represents the vacancy of the stochastic solid.
In this section, we will prove that given a Gaussian primitive
𝐺(x) rendered by Gaussian Splatting, we can find a solid that
generates the same rendering results by the stochastic theory.
The vacancy of the solid should satisfy:
v(x) =
√︁
1 −𝐺(x).
(35)
We will prove that under the following constraints, the
stochastic solid can be uniquely determined:
– When 𝐺(x1) ≥𝐺(x2), it follows that o(x1) ≥o(x2), indi-
cating that positions closer to the Gaussian center have
higher occupancy;
– The occupancy of the solid approaches 0 when x is far
from Gaussian center;
– The occupancy o(x) is differentiable with respect to x.
Proof: Assume a line 𝑙parameterized by 𝑡passes through
𝐺(x). We get the value of𝐺(x) on the line forming a 1D Gaussian
function 𝐺(𝑡), where 𝑡goes from −∞to +∞and reaches the
maximum of the 1D Gaussian at 𝑡∗.
Firstly, we derive the color from volume rendering. According
to the first assumption, the vacancy function along this line𝑙has
the opposite monotonicity compared to the Gaussian function.
We will get the attenuation coefficient from Equation 34:
𝜎(𝑡) = |𝜔· ∇𝑙𝑜𝑔(v(x))|
= | 𝜕𝑙𝑜𝑔(v(x))
𝜕𝑡
|
=
(
−𝜕𝑙𝑜𝑔(v(x))
𝜕𝑡
,
𝑡≤𝑡∗
𝜕𝑙𝑜𝑔(v(x))
𝜕𝑡
,
𝑡> 𝑡∗
(36)
Since a Gaussian kernel has a uniform color, we can simplify
the volume rendering:
C =
∫𝑡=+∞
𝑡=−∞
𝑇(𝑡)𝜎(x(𝑡),𝜔)c𝑑𝑡
= c
∫𝑡=+∞
𝑡=−∞
𝑇(𝑡)𝜎(x(𝑡),𝜔) 𝑑𝑡
= c
∫𝑡=+∞
𝑡=−∞
−𝑑𝑇(𝑡)
= c𝑇(𝑡)
𝑡=−∞
𝑡=+∞= c(1 −𝑇(+∞))
(37)
We then substitute the Equation 36 into Equation 37 to get the
form of color from volume rendering:
𝑇(∞) = 𝑇(−∞,𝑡∗) ×𝑇(𝑡∗, +∞)
= 𝑒−
∫𝑡∗
−∞𝜎(x(𝑠),𝜔) × 𝑒−
∫+∞
𝑡∗
𝜎(x(𝑠),𝜔)
= 𝑒−(−𝑙𝑜𝑔(v(𝑡))
𝑡∗
−∞) × 𝑒−(𝑙𝑜𝑔(v(𝑡))
+∞
𝑡∗)
= v(𝑡∗)
v(−∞) × v(𝑡∗)
v(+∞)
= v(𝑡∗)2
C =c(1 −𝑇(+∞)) = c(1 −v(𝑡∗)2),
(38)
where we use the second assumption that v(∞) = 1 −o(∞) = 1.
Secondly, with the color derived from Gaussian Splatting
and volume rendering, we can find the relationship between
v(𝑡∗) and 𝐺(𝑡∗):
c𝐺(𝑡∗) = c(1 −v(𝑡∗)2) ⇒v(𝑡∗) =
√︁
1 −𝐺(𝑡∗)
(39)
Finally, we will generalize Equation 39 from maximum points
to all 3D points. Different lines 𝑙have different maximum
points, and Equation 39 should hold for the maximum point
on any line. Given any x ∈R3, we can always find the direction
𝜔∈S2 satisfying 𝜔· ∇𝐺(x) = 𝜕𝐺(x)
𝜕𝜔
= 0, indicating that x is the
maximum point along ray 𝑙: x + 𝑡𝜔. Therefore, the equation
should hold for any position x, which reaches the unique
solution of vacancy in Equation 35.
H
Derivation of Equation 12 of the main paper
In this section, we will derive the closed form 𝑇𝑖(𝑡) in Equa-
tion 12 of the main paper, which is also the negative integral
of the free-flight distribution −
∫
𝑝(𝑡) 𝑑𝑡. For brief notation, we
use 𝑇to denote the transmittance of a single Gaussian.
We start from 𝑡𝑛= −∞. Similar to Equation 38, when 𝑡< 𝑡∗,
𝑇(−∞,𝑡) = 𝑒−
∫𝑡∗
−∞𝜎(x(𝑠),𝜔) = v(𝑡).
(40)
When 𝑡> 𝑡∗,
𝑇(−∞,𝑡) = 𝑇(−∞,𝑡∗) × 𝑒−
∫𝑡∗
−∞𝜎(x(𝑠),𝜔)
= v(𝑡∗) × v(𝑡∗)
v(𝑡)
= v(𝑡∗)2
v(𝑡) .
(41)
In most cases, the Gaussian primitive is far from the camera,
so we can simply use 𝑇(−∞,𝑡) as 𝑇(𝑡).

<!-- page 16 -->
16
•
Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan
I
Derivation of Equation 13 of the main paper
In this section, we will derive the gradient of the depth 𝑡𝑚𝑒𝑑
with respect to the parameters of all the Gaussians along the
ray. Since 𝑇(𝑡𝑚𝑒𝑑) is a constant value of 0.5, its differential is 0.
𝑇(𝑡𝑚𝑒𝑑;𝜃) ≡0.5,
(42)
𝑑𝑇(𝑡𝑚𝑒𝑑;𝜃) ≡0,
(43)
where 𝜃represents the parameters of Gaussians along the ray.
We then expand the 𝑑𝑇and plug 𝑡𝑚𝑒𝑑into it to derive the
gradient:
𝑑𝑇(𝑡;𝜃) = 𝜕𝑇
𝜕𝑡𝑑𝑡+ 𝜕𝑇
𝜕𝜃· 𝑑𝜃
(44)
0 = 𝜕𝑇
𝜕𝑡𝑑𝑡𝑚𝑒𝑑+ 𝜕𝑇
𝜕𝜃· 𝑑𝜃
(45)
𝑑𝑡𝑚𝑒𝑑= (−𝜕𝑇
𝜕𝜃/ 𝜕𝑇
𝜕𝑡) · 𝑑𝜃.
(46)
So that the gradient of depth is derived as,
𝜕𝑡𝑚𝑒𝑑
𝜕𝜃
= −𝜕𝑇(𝑡𝑚𝑒𝑑;𝜃)
𝜕𝜃
/ 𝜕𝑇(𝑡;𝜃)
𝜕𝑡

𝑡=𝑡𝑚𝑒𝑑,
(47)
