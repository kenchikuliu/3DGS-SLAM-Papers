<!-- page 1 -->
Paper preprint
GEOSPLAT:
A
DEEP
DIVE
INTO
GEOMETRY-
CONSTRAINED GAUSSIAN SPLATTING
Yangming Li1, Chaoyu Liu1˚, Lihao Liu1, Simon Masnou2, Carola-Bibiane Sch¨onlieb1
1Department of Applied Mathematics and Theoretical Physics, University of Cambridge
2Institut Camille Jordan, Universit´e Claude Bernard Lyon 1
{yl874,cl920}@cam.ac.uk
ABSTRACT
A few recent works explored incorporating geometric priors to regularize the op-
timization of Gaussian splatting, further improving its performance. However,
those early studies mainly focused on the use of low-order geometric priors (e.g.,
normal vector), and they might also be unreliably estimated by noise-sensitive
methods, like local principal component analysis. To address their limitations, we
first present GeoSplat, a general geometry-constrained optimization framework
that exploits both first-order and second-order geometric quantities to improve the
entire training pipeline of Gaussian splatting, including Gaussian initialization,
gradient update, and densification. As an example, we initialize the scales of 3D
Gaussian primitives in terms of principal curvatures, leading to a better coverage
of the object surface than random initialization. Secondly, based on certain ge-
ometric structures (e.g., local manifold), we introduce efficient and noise-robust
estimation methods that provide dynamic geometric priors for our framework.
We conduct extensive experiments on multiple datasets for novel view synthesis,
showing that our framework, GeoSplat, significantly improves the performance of
Gaussian splatting and outperforms previous baselines.
1
INTRODUCTION
The photorealistic rendering quality and high efficiency of Gaussian splatting (Kerbl et al., 2023)
have fueled a surge of recent studies, which further investigate it from various perspectives, such as
memory consumption (Niedermayr et al., 2024) and texture (Gao et al., 2024).
Limitations in recent geometric regularization.
A notable perspective from those studies is to
treat geometric priors as a form of regularization for improving Gaussian splatting, given that the
Gaussian primitives should approximate the surfaces of 3D objects. For example, Li et al. (2025)
places new primitives in the tangent planes of original primitives, with the aim to reduce outlier ar-
tifacts. While obtaining performance gains, previous methods mainly focus on low-order geometric
quantities (e.g., normal vector) and do not much consider higher-order ones (i.e., curvature). The
two types of geometric information characterize very different aspects of 2D object surfaces (Lee,
2018), so the absence of either one might limit their potential.
Another key limitation of prior works is that they sometimes estimate geometric information using
methods that lack robustness. For example, local principal component analysis (PCA) (Kambhatla
& Leen, 1997; Li et al., 2025) is highly sensitive to noise, and the trained StableNormal model (Ye
et al., 2024; Wang et al., 2024) often fails on rarely seen data. The geometric priors in previous
methods are also static, which would gradually become inaccurate in characterizing the dynamic
geometry of Gaussian primitives during optimization.
Our geometric optimization framework: GeoSplat.
To address the limitations of prior works,
we first propose a general geometry-constrained optimization framework, GeoSplat, exploiting both
low-order and higher-order geometric priors to regularize the training pipeline of Gaussian splatting.
˚Corresponding author.
1
arXiv:2509.05075v3  [cs.CV]  26 Sep 2025

<!-- page 2 -->
Paper preprint
While being largely ignored by previous methods, the higher-order information (e.g., curvature)
characterizes a critical property of 2D surface: how it bends in the 3D space. In light of this, we
adopt such information to regularize the shapes of Gaussian primitives. For instance, an area with
low curvature indicates that it approximates a flat plane, so we can initialize the scales of Gaussian
primitives in this area as a large number. Low-order geometric quantities, such as tangent and normal
vectors, have already played key roles in recent works, and we further extend their application scopes
in our framework. For example, to reduce floating artifacts, we truncate the gradient update of a
Gaussian primitive in terms of its normal direction.
Secondly, we present an estimation method that can provide noise-robust geometric information
for our framework. Specifically, we assume that each Gaussian primitive resides on an underlying
surface that is locally a manifold (Lee, 2018), and we derive the analytical form of the shape operator,
which contains the curvature information. We also investigate another approach, based on the notion
of varifold from geometric measure theory (Allard, 1972; Simon, 1983; Menne, 2017), which in
some cases outperforms our manifold-based method. Both methods are highly efficient, and thus
can provide our framework with dynamic geometric priors during training.
Extensive experiments have been conducted on multiple benchmark datasets in novel view synthesis.
The results demonstrate that our framework, GeoSplat, significantly improves the performance of
Gaussian splatting and outperforms previous baselines.
2
PRELIMINARIES
In this section, we mainly revisit the basics of Gaussian splatting and formulate our problem setup,
establishing necessary notations. We also review basic geometry in Appendix A to explain some
geometric terminologies (e.g., tangent space) used in the main text.
2.1
GAUSSIAN SPLATTING
The core of Gaussian splatting is to represent a 3D scene as a number of Gaussian primitives
tGi
objuiPI, with each primitive parameterized by its opacity αi
obj P R, RGB color ci
obj P R3, mean
vector µi
obj P R3, and covariance matrix Σi
obj P R3ˆ3. The matrix Σi
obj is semipositive definite
and can be decomposed as Ri
objSi
objpRi
objSi
objqJ, where Si
obj “ diagprsi
1, si
2, si
3sJq is a diagonal
scale matrix and Ri
obj “ rri
1; ri
2; ri
3s P SOp3q is a rotation matrix.
A pixel at some image coordinate p P R ˆ R is synthesized by splatting and blending Gaussian
primitives. Specifically, the splatting operation first casts the 3D Gaussian primitive Npµi
obj, Σi
objq
in the word coordinate to a 2D primitive Npµi
plane, Σi
planeq in the image plane:
µi
plane “ ΠpTposeµi
objq,
Σi
plane “ JWposeΣi
objpJWposeqJ,
(1)
where Tpose P SEp3q is the camera pose consisting of rotation matrix Wpose and translation vector
bpose, and Π is the 3D to 2D coordinate projection, with J its Jacobian matrix. Then, the blending
operation renders the color of pixel p as
cimgppq “
ÿ
i
ci
objβippq
ź
jăi
p1 ´ βjppqq,
(2)
where βkppq :“ αk
objNpp; µk
plane, Σk
planeq quantifies the radiance of a Gaussian primitive. Under
this scheme, the optimization objective is to match rendered images with real images.
2.2
PROBLEM FORMULATION
Our primary goal is to leverage diverse geometric priors to improve Gaussian splatting optimization,
especially higher-order ones (e.g., curvature). Meanwhile, we also aim to estimate the priors in a
noise-robust manner. These two aspects are largely neglected in recent explorations (Wang et al.,
2024; Li et al., 2025) on geometric regularization.
2

<!-- page 3 -->
Paper preprint
Manifold-based setup.
Manifold (Lee, 2018) is a popular geometric language for characterizing
spatial objects. We provide an overview on it in Appendix A.1, mentioning a strict assumption that
the objects need to be everywhere smooth. It is not hard to find real 3D objects that violate this
assumption. For example, an apple has a singular point at the intersection between its stem and
body. To generalize this geometric concept, we consider a weaker setting.
Definition 2.1 (Merged Local Manifolds). Gaussian primitives tGi
objuiPI reside in an underlying
3D scene formed by finitely many continuous manifold tMjujPJ . Notably, each manifold Mj is
smooth up to a zero-measure set of singular points.
Compared with vanilla manifold, this new setting can accommodate typical 3D scenes, where there
are multiple independent spatial objects that might not be fully smooth. The zero-measure condition
is also important, permitting almost every point q P R3 in the scene to locate at a local region that
can be treated as a smooth manifold. For novel view synthesis, the surface Mj is normally 2D (we
denote its dimension as DMj “ 2), embedded in the 3D ambient space E “ R3 (DE “ 3q.
Most geometric quantities are locally defined, and thus fit our setting. Specifically, every Gaussian
primitive Gi
obj, i P I will lie on a certain underlying object surface Mj, j P J at position q “ µi
obj,
with tangent space TqMj spanned by orthonormal basis tui
du1ďdďDMj and normal space NqMj
formed by another basis trui
duDMj `1ďdďDE. These vector bases represent first-order geometric
quantities, and we also aim to consider higher-order ones, e.g. principal curvatures and directions
tpτ i
d, rwi
dqu1ďdďDMj , with each pair indicating that the underlying surface is bent with a magnitude
of τ i
d along direction rwi
d. For notational convenience, we will omit the subscript j of local manifold
Mj, since it is unique for each primitive Gi
obj.
Another geometric tool: varifolds.
We also explore adopting another tool from geometric mea-
sure theory, the notion of varifold (Allard, 1972; Simon, 1983; Menne, 2017), which is free from
the smoothness assumption of manifolds. Generally speaking, a varifold is a nonnegative Radon
measure (Rudin, 1987) on the product space E ˆGDM,DE where the second space denotes the Grass-
mannian of DM-dimensional vector subspaces of E (Lee, 2018). Two classes of varifolds useful
in our context are rectifiable varifolds and point cloud varifolds. A rectifiable varifold has the form
W “ θHDM
|M bδTxM where HDM
|M is the DM-dimensional Hausdorff measure restricted to a countably
rectifiable set M (Federer, 1969; Simon, 1983), δTxM is the Dirac measure in GDM,DE supported on
the tangent plane at x P M, and θ : M Ñ R is a nonnegative multiplicity function with θ ą 0
HDM-a.e. on M. A point cloud d-varifold has the form W “ ř
1ďjďN mjδxj b δPj where txjuN
j“1
is a finite set of points associated with d-planes tPjuN
j“1 and nonnegative masses tmjuN
j“1.
The notion of varifold is so flexible that it can be associated with the discrete locations of Gaussian
primitives tGi
objuiPI without supposing there is any smooth interpolation. More importantly, geo-
metric quantities valid for manifolds can be weakly or approximately defined for varifolds, such as
the approximate tangent space or the (approximate) generalized second fundamental form and mean
curvature. Due to the limited space of main text, we provide more details in Appendix A.2, with a
concise review of varifold theory.
3
METHOD: GEOMETRY-CONSTRAINED GAUSSIAN SPLATTING
In this section, we first present GeoSplat, a framework of geometry-constrained Gaussian splatting,
which exploits higher-order geometric priors that are largely neglected by previous works. Then, we
present noise-robust methods to estimate dynamic geometric priors. In contrast, previous methods
obtained static priors through noise-sensitive or domain-specific techniques.
3.1
GEOMETRY-CONSTRAINED OPTIMIZATION
Training a Gaussian splatting model involves initialization, optimization, and densification. In the
following, we show how each of these components can be geometrically improved.
3

<!-- page 4 -->
Paper preprint
3.1.1
CURVATURE-GUIDED GAUSSIAN INITIALIZATION
Curvature-guided covariance warm-up.
In common practices, most attributes of a Gaussian
primitive Gi
obj (e.g., the position vector µi
obj) are properly initialized before training to ensure
stable optimization and high performance. However, this is not the case for covariance matrix
Σi
obj, with its rotation part Ri
obj arbitrarily initialized and scale part simply set to be isotropic:
Si
obj “ psi
nbr{2qI. Here si
nbr denotes the average distance to a few neighbors.
We assume that the primitive Gi
obj locates at an underlying surface M1, so it is intuitive to warm up
its initial shape Σi
obj in terms of geometric priors. To begin with, we set one rotation direction as
the normal vector to M, with the corresponding scale specified as a tiny value:
ri
3 “ rui
3,
si
3 “ ξmin ! 1.
(3)
In this manner, Gaussian primitive Gi
obj will get largely squashed along the normal direction rui
3, and
thus tightly fit the 2D surface: tri
1, ri
2u Ă TqM.
Secondly and more importantly, other rotation directions ri
1, ri
2 are the eigenvectors of covariance
matrix Σi
obj, so they correspond to the tangent directions that lead to the maximum and minimum
variances psi
1q2, psi
2q2 (Anderson et al., 1958). In this regard, we respectively configure them as
principal directions, with a curvature-based scale constraint:
ri
d “ rwi
3´d, d P t1, 2u,
si
1{si
2 “ |τ i
2|´1{|τ i
1|´1.
(4)
In this expression, the maximum standard deviation si
1 (or minimum one si
2) in statistics is linked
with the low curvature τ i
2 (or high one τ i
1) in geometry. The core idea is that the points sampled on
a flat curve (along the low curvature direction wi
2) tend to be less concentrated (in the high-variance
direction ri
1) than a twisted one, given the same lengths.
The two scale entries si
1, si
2 are underdetermined with respect to a single constraint, so we impose
an extra area-invariant condition to finally solve them.
Proposition 3.1 (Curvature-constrained Primitive Scales). If the curvature-based constraint (i.e.,
Eq. (4)) and a new constraint: si
1si
2 “ psi
nbrq2{4, both hold, then the scale variables can be solved
as si
1 “ si
nbr
2
a
|τ i
1{τ i
2|, si
2 “ si
nbr
2
a
|τ i
2{τ i
1|. Notably, this solution ensures that the one-sigma region
of projected Gaussian primitive in the tangent plane TqM are area-invariant.
The significance of area invariance is to ensure that the primitive Gi
obj still has a sufficient coverage
over the surface M after re-scaling. We put the proof for this conclusion in Appendix C. To sum-
marize, the covariance matrix Σi
obj of Gaussian primitive Gi
obj can be fully warmed up in terms of
normal and curvature priors, with its components as
Ri
obj “ r rwi
2; rwi
1; rui
3s,
Si
obj “ diag
`“si
nbr
2
b
|τ i
1{τ i
2|, si
nbr
2
b
|τ i
2{τ i
1|, ξmin
‰J˘
.
(5)
This solution assumes non-zero curvatures: τ i
d ‰ 0, d P t1, 2u, but we can fix the zero case by
clamping it as minpmaxpτ i
d, ξminq, ξmaxq, where ξmax is a predefined large value.
Curvature-guided primitive upsampling.
The initial set of Gaussian primitives tGi
objuiPI are
not large enough, so Ververas et al. (2025) populated the low-curvature areas that had few primitives
before training. The mean curvature was used to identify such areas, though it suffered from an
inaccurate approximation using low-order quantities.
Our noise-robust methods (Sec. 3.2) provide full curvature information tτ i
du1ďdďDM, so we are able
to directly compute the mean curvature as pτ i
1 ` τ i
2q{2, without any approximation. It is not proper
to identify the low-curvature regions via mean curvature. A counter-example is the helical surface,
which has a zero mean curvature but is curved (i.e., ś
d τ i
d ‰ 0). In this regard, we adopt a more
reasonable quantity: mean absolute curvature (MAC), as sτ i “ př
d |τ i
d|q{DM.
1With abuse of notation, we mostly denote the 2D surface as M, potentially assuming it is a local manifold.
However, it should be considered as a rectifiable set M in the context of rectifiable varifolds.
4

<!-- page 5 -->
Paper preprint
If MAC is quite small, i.e. sτ i ă ξmin, which means the curvature is basically negligible, then we
can identify that the corresponding Gaussian primitive Gi
obj resides in an almost flat region. In this
case, we upsample its position through midpoint interpolation:
µi1
obj “ pµi
obj ` µ‹
objq{2, ‹ P KNNpiq,
(6)
which also applies to other attributes (e.g., color). Here KNN denotes the nearest neighbors in index
set I, and i1 is a new index for the set. Due to the flatness, this linear interpolation will locate the
new primitive Gi1
obj near the underlying surface M.
3.1.2
SHAPE-CONSTRAINED OPTIMIZATION
Truncated gradient update.
The rendering quality of Gaussian splatting is largely affected by
floating artifacts (Ungermann et al., 2024; Turkulainen et al., 2025). From a geometric point of view,
we can define a Gaussian primitive Gi
obj as an outlier if it significantly deviates from the underlying
surface M. In this sense, we propose to reduce the outliers through truncating the gradient update
with respect to the normal direction rui
3 as
µi
obj Ð µi
obj ´ ω
`
p∇iLqJ ` min
`
ξmin{}p∇iLqK}, 1
˘
p∇iLqK˘
,
(7)
where ω is the learning rate, ∇iL denotes the gradient of loss function L regarding position µi
obj,
p¨qJ denotes the orthogonal projection onto the tangent plane TqM, and p¨qK represents the projec-
tion onto the normal direction rui
3. With this truncation, a Gaussian primitive Gi
obj that lies on the
underlying surface M will still stay close to it after an aggressive gradient update ∇iL.
Shape regularization.
Another type of artifact that might degrade the performance of Gaussian
splatting is the Gaussian primitive Gi
obj with a needle-like shape (Hyung et al., 2024): covariance
matrix Σi
obj degenerates to be rank one. From a geometric perspective, the main scales si
1, si
2 of the
primitive Gi
obj must be regularized to ensure a rank two covariance matrix Σi
obj, so that it will have
a sufficient coverage over the underlying manifold M.
Based on the above insight and our prior discussion (i.e., Eq. 4), we present a hinge-like regulariza-
tion that will impose a penalty if the minor scale si
2 is too small:
Lscale “ max
`
0, si
1{si
2 ´ |τ i
1{τ i
2| ´ ξmin
˘
`
`
si
3
˘2,
(8)
with the last term to ensure the rank of matrix Σi
obj is smaller than 3, hence the Gaussian primi-
tive Gi
obj is closely attached to the surface M. For the hinge term, we again regard the curvature
information tτ 1
d, τ 2
du as natural guidance for shaping the scale matrix Si
obj. In a similar spirit, we
introduce another regularization for the rotation matrix Ri
obj as
Lrot “
`
1 ´ xri
1, rwi
2y
˘2 `
`
1 ´ xri
2, rwi
1y
˘2 ` p1 ´ xri
3, rui
3y
˘2,
(9)
aligning its components with the principal directions trwi
du1ďdďDM.
3.1.3
CURVATURE-REGULARIZED PRIMITIVE DENSIFICATION
Densification operations (i.e., split and clone) are indispensable in Gaussian splatting for rendering
high-texture regions, though they might generate floating artifacts (Li et al., 2025). From a geometric
viewpoint, that problem is caused by an inappropriate placement of new Gaussian primitive Gi1
obj, i1 R
I, making it an outlier for the underlying surface M.
To address this problem, we first restrict the split operation to produce the new primitive Gi1
obj close
to the surface M. Specifically, we interpolate the principal directions trwi
du1ďdďDM with a random
vector ρ “ rρ1, ρ2, ρ3sJ sampled from standard normal Np0, Iq as
µi1
obj “ µi
obj ` pρ2{τ i
2qrwi
2 ` pρ1{τ i
1qrwi
1 ` ρ3ξminrui
3.
(10)
Here the principal terms ensure the new primitive Gi1
obj stays near the tangent plane TqM, while the
last term allows a slight oscillation in the normal direction rui
3. For using curvatures tτ i
du1ďdďDM
as weights, it aims to bias the sampling towards sparse areas.
5

<!-- page 6 -->
Paper preprint
Secondly, in a similar way as our previous gradient truncation (i.e., Eq. (7)), we regularize the clone
operation by clipping the gradient accumulation s∇iL as
µi1
obj “ µi
obj ` ps∇iLqJ ` min
`
ξmin{}ps∇iLqK}, 1
˘
ps∇iLqK.
(11)
The truncated shift in the normal direction p¨qK will lead to fewer new outlier primitives.
3.2
NOISE-ROBUST AND EFFICIENT GEOMETRIC ESTIMATIONS
Previous works obtained noisy geometric priors through potentially unreliable methods (e.g., local
PCA), and the priors were also kept static during optimization. As Gaussian primitives got denser,
those priors became even more unreliable. To address their limitations, we present noise-robust and
efficient estimation methods that can provide dynamic geometric information.
Manifold-based method.
Our estimation method assumes that every Gaussian primitive Gi
obj lies
on a local manifold M (i.e., Definition 2.1). We summarize below how it works, and due to the
space limitation, we provide derivation details in Appendix E.
Theorem 3.2 (Simplified from Proposition E.2 and Theorem E.7). For a Gaussian primitive Gi
obj
that lies on the embedded local manifold M ãÑ E at position q “ µi
obj, the eigenvectors of its
tangential kernel matrix Ki (as formulated by Eq. (42)) that correspond to non-zero eigenvalues
form an orthonormal basis tui
du1ďdďDM in the tangent space TqM.
Likewise, another shape operator matrix Si (as formulated by Eq. (51)) can be decomposed into
pairs of eigenvalue and realigned eigenvector, which correspond to the principal curvatures and
directions tpτ i
d, rwi
dqu1ďdďDM for the primitive Gi
obj.
For the normal information rui
3 in 3D reconstruction, we can easily compute its unnormalized version
through the cross product of two tangent vectors: ui
d1 ˆ ui
d2, d1 ‰ d2.
Varifold-based method.
We have explored a different geometric approach based on varifolds.
Generalizing the notion of second fundamental form valid for manifolds, Buet et al. (2022) intro-
duced for point cloud varifolds an approximate second fundamental form (WSFF) as a matrix whose
eigendecomposition provides full approximate curvature information. In Appendix D, we adapt that
matrix to our setting, with its expression Bi for a Gaussian primitive Gi
obj as
$
’
’
&
’
’
%
Bi “
´
ÿ
jPKNNpiq
mjχϵp}µi
obj ´ µj
obj}q
¯´1
ÿ
jPKNNpiq
mjΥ1
ϵp}µi
obj ´ µj
obj}q
3}µi
obj ´ µj
obj}
Bi,j
Bi,j “ 2p rVjViqJsym
`
rui
3pµi
obj ´ µj
objqJ˘ rVjVi ` pµi
obj ´ µj
objqJ rVjrui
3ppViqJ rVjVi ´ Iq
,
(12)
where mj denotes a weight, χϵ, Υϵ are kernel functions that depend on the approximation scale ϵ,
symp□q “ p□` □Jq{2 is a symmetrization operation for any matrix □, and rVj “ I ´ rui
3prui
3qT is
the orthogonal projection matrix onto the tangent plane TqM, with its basis as Vi “ rui
1; ui
2s.
Dynamic estimations during optimization.
Our estimation methods are both noise-robust and
run-time efficient, which can process million-level Gaussian primitives in only tens of seconds. To
update geometric information for the growing Gaussian primitives, we perform estimations at evenly
spaced iterations during training, incurring a minor time increase.
4
RELATED WORK
Common 3D scenes have a clear geometric structure (e.g., smoothness), so there is an emerging
field in the literature (Wang et al., 2024; Bonilla et al., 2024; Li et al., 2025) that aims to regularize
the optimization of Gaussian splatting in terms of certain geometric information. For example,
Wang et al. (2024); Turkulainen et al. (2025) aligned the Gaussian primitive towards the normal
direction, and Li et al. (2025) not only considered that, but also placed the densified primitives in the
tangent plane. While achieving noticeable performance improvements, those earlier studies mainly
considered low-order geometric quantities (e.g., normal vector), neglecting the valuable information
6

<!-- page 7 -->
Paper preprint
Method
Metric
R0
R1
R2
OFF0 OFF1 OFF2 OFF3 OFF4
Vox-Fusion (Yang et al., 2022)
PSNRÒ 22.39 22.36 23.92 27.79 29.83 20.33 23.47 25.21
SSIMÒ
0.683 0.751 0.798 0.857 0.876 0.794 0.803 0.847
LPIPSÓ 0.303 0.269 0.234 0.241 0.184 0.243 0.213 0.199
Point-SLAM (Sandstr¨om et al., 2023)
PSNRÒ 32.40 34.08 35.50 38.26 39.16 33.99 33.48 33.49
SSIMÒ
0.974 0.977 0.982 0.983 0.986 0.960 0.960 0.979
LPIPSÓ 0.113 0.116 0.111 0.100 0.118 0.156 0.132 0.142
Gaussian-Splatting SLAM (Matsuki et al., 2024)
PSNRÒ 34.83 36.43 37.49 39.95 42.09 36.24 36.70 36.07
SSIMÒ
0.954 0.959 0.965 0.971 0.977 0.964 0.963 0.957
LPIPSÓ 0.068 0.076 0.070 0.072 0.055 0.078 0.065 0.099
GeoGaussian (Li et al., 2025)
PSNRÒ 35.20 38.24 39.14 42.74 42.20 37.31 36.66 38.74
SSIMÒ
0.952 0.979 0.970 0.981 0.970 0.970 0.964 0.967
LPIPSÓ 0.029 0.021 0.024 0.016 0.040 0.029 0.029 0.031
Our Model: GeoSplat, w/ Manifold-based Priors
PSNRÒ 36.37 39.55 40.36 43.38 42.70 37.74 37.06 39.31
SSIMÒ 0.976 0.982 0.985 0.984 0.988 0.975 0.969 0.972
LPIPSÓ 0.024 0.018 0.021 0.015 0.037 0.027 0.027 0.028
Our Model: GeoSplat, w/ Varifold-based Priors
PSNRÒ 36.35 39.54 40.05 43.41 42.64 37.72 37.08 39.52
SSIMÒ
0.973 0.975 0.981 0.985 0.986 0.974 0.968 0.981
LPIPSÓ 0.025 0.019 0.022 0.013 0.971 0.028 0.028 0.027
Table 1: Performance comparison on a number of Replica datasets.
Method
Metric
Room-1 Room-2 Office-2 Office-3
3D Gaussian Splatting (3DGS) (Kerbl et al., 2023)
PSNRÒ
40.79
39.10
37.88
36.04
SSIMÒ
0.973
0.974
0.962
0.975
LPIPSÓ
0.025
0.017
0.024
0.017
LightGS (Fan et al., 2024)
PSNRÒ
41.26
39.23
37.99
36.06
SSIMÒ
0.974
0.974
0.962
0.975
LPIPSÓ
0.023
0.017
0.023
0.016
GeoGaussian (Li et al., 2025)
PSNRÒ
41.43
39.46
38.54
36.19
SSIMÒ
0.976
0.975
0.967
0.977
LPIPSÓ
0.019
0.018
0.017
0.015
Our Model: GeoSplat, w/ Manifold-based Priors
PSNRÒ
41.81
39.97
38.75
36.61
SSIMÒ
0.978
0.979
0.971
0.978
LPIPSÓ
0.016
0.013
0.016
0.013
Our Model: GeoSplat, w/ Varifold-based Priors
PSNRÒ
41.92
39.80
38.68
36.60
SSIMÒ
0.980
0.977
0.969
0.979
LPIPSÓ
0.015
0.015
0.017
0.012
Table 2: Performance comparison on multiple ICL datasets.
contained in higher-order ones (e.g., curvature). As a rare case, Ververas et al. (2025) adopted the
mean curvature to identify flat areas, though this was not a fully correct strategy (as explained in
Sec. 3.1.1) and their implementation was still based on low-order quantities. A major contribution
of this paper is the introduction of GeoSplat: a geometry-constrained optimization framework that
exploits the higher-order geometric information neglected by prior works.
Previous studies were limited by potentially unreliable estimation methods, which yielded static
and noise-sensitive geometric priors. For example, Wang et al. (2024) adopted the StableNormal
model (Ye et al., 2024) to predict normal vectors, which tends to fail on unseen data, while Li
et al. (2025) employed Agglomerative Hierarchical Clustering (AHC) (Feng et al., 2014), a classical
clustering method that is not robust to sparsity. Even worse, as Gaussian primitives become denser
during optimization, these static priors become increasingly unreliable. Based on inherent geometric
structures (e.g., local manifolds), we propose efficient estimation methods to provide dynamic and
reliable geometric information during optimization.
5
EXPERIMENTS
In this section, we show extensive experiment results that verify the effectiveness of our framework.
We provide the details of our experiment setup and more results in Appendix F.
5.1
MAIN RESULTS
We compare our models with a number of baselines (e.g., GeoGaussian) on two types of view syn-
thesis datasets, Replica (Straub et al., 2019) and ICL (Handa et al., 2014), across 12 different scenes.
The results are provided in Table 1 and Table 2, showing that our optimization framework, GeoSplat,
is able to significantly improve Gaussian splatting and outperform previous baselines. For the first
7

<!-- page 8 -->
Paper preprint
Figure 1: The performance of our models and baselines in low-resource settings.
(a) Ground Truth (Case 1).
(b) 3DGS (Case 1).
(c) GeoGaussian (Case 1). (d) Our GeoSplat (Case 1).
(e) Ground Truth (Case 2).
(f) 3DGS (Case 2).
(g) GeoGaussian (Case 2). (h) Our GeoSplat (Case 2).
Figure 2: Ground-truth and rendered images on the low-resource ICL Office-2 dataset. The cases 1,
2 are respectively generated by our manifold-based and varifold-based models.
point, we can see that our varifold-based model achieves consistent noticeable performance gains
relative to 3DGS (Kerbl et al., 2023), which are 2.77% on ICL Room-1 in terms of PSNR, and our
manifold-based model also outperforms it by 0.93% on ICL Office-2 in terms of SSIM. For the
second point, we can see that both our manifold-based and varifold-based models significantly out-
perform the key baseline, GeoGaussian, on every Replica dataset. For example, we achieve higher
PSNR scores by 3.42% on Replica R1 and 2.01% on Replica OFF4. The consistent performance
gains over a dozen of datasets indicate that our framework is indeed effective.
5.2
LOW-RESOURCE SETTING
Since our models are regularized by various types of geometric priors, it is intuitive that we might
obtain even higher performance gains in low-resource settings, where observed views are sparse in
the dataset. To verify this intuition, we compare our models with the baselines on 4 datasets from
both Replica and ICL, with some percentage (pct.) of views excluded. The results are provided in
Fig. 2, showing that our models have much slower performance decreases when views got sparser,
even compared with GeoGaussian (which was regularized by low-order normal information). For
example, our performance gain in terms of PSNR is 3.11% on the full Replica R2 dataset, and can
be further enlarged as 7.93% when only 1{6 views are observed.
8

<!-- page 9 -->
Paper preprint
(a) Baseline: 3DGS.
(b) Baseline: GeoGaussian.
(c) Our Model: Varifold-based GeoSplat.
(d) Our Model: Manifold-based GeoSplat.
Figure 3: Colored Gaussian primitives on the low-resource ICL Office-2 dataset. The boxes in white
and yellow respectively indicate sparse areas and outlier artifacts.
Method
Replica R0
ICL Office-3
PSNRÒ SSIMÒ LPIPSÓ PSNRÒ SSIMÒ LPIPSÓ
Our Model: Manifold-based GeoSplat
36.37
0.976
0.024
36.61
0.978
0.013
w/o Curvature-guided Covariance Warm-up (Eq. (5))
36.01
0.969
0.026
36.43
0.976
0.014
w/o Curvature-guided Primitive Upsampling (Eq. (6))
36.12
0.971
0.025
36.51
0.976
0.015
w/o MAC, w/ Mean Curvature (Ververas et al., 2025)
36.25
0.973
0.025
36.58
0.977
0.014
w/o Truncated Gradient Update (Eq. (7))
36.09
0.970
0.026
36.47
0.975
0.015
w/o Shape Regularization (Eq. (8), Eq. (9))
35.98
0.965
0.027
36.39
0.973
0.016
w/o Curvature-regularized Densification (Eq. (10), Eq. (11))
36.05
0.970
0.026
36.48
0.975
0.015
Our Model: Varifold-based GeoSplat
36.35
0.973
0.025
36.60
0.979
0.012
w/o Curvature-guided Covariance Warm-up (Eq. (5))
36.00
0.963
0.027
36.41
0.976
0.015
w/o Curvature-guided Primitive Upsampling (Eq. (6))
36.17
0.968
0.026
36.53
0.977
0.013
w/o MAC, w/ Mean Curvature (Ververas et al., 2025)
36.23
0.971
0.026
36.57
0.978
0.013
w/o Truncated Gradient Update (Eq. (7))
36.05
0.967
0.028
36.45
0.977
0.014
w/o Shape Regularization (Eq. (8), Eq. (9))
35.93
0.961
0.029
36.37
0.975
0.016
w/o Curvature-regularized Densification (Eq. (10), Eq. (11))
36.03
0.964
0.028
36.43
0.977
0.014
Table 3: Experiment results from the ablation studies on our models.
5.3
CASE STUDIES
To visualize the effect of geometric regularization, we first compare the images rendered from our
model with those from the baselines on sparse ICL Office-2 (80% views excluded). The results are
demonstrated in Fig. 2. We can see that even the images generated by GeoGaussian (Subfig. 2c and
Subfig. (2g)) contain perceivable floating artifacts, exhibiting large holes on the ceiling. In contrast,
the images rendered by our models are noticeably cleaner and smoother, which benefits from our
geometric regularization strategies. Figure 3 further shows the primitive clouds of our models and
the baselines. We can see that the latter contain substantially more outlier primitives and holes in
low-texture regions. These observations confirm the effectiveness of our framework.
5.4
ABLATION STUDIES
We conduct ablation experiments to quantitatively measure the impacts of our regularization strate-
gies, further confirming their effectiveness. As shown in Table 3, the performance of our model gets
degraded after taking any component out from our framework. For example, without shape regular-
ization, the PSNR scores of our manifold-based model noticeably decrease by 1.07% on Replica R0
and 0.60% on ICL Office-3. We can also see that using MAC as the flat-area identifier, instead of
the mean curvature, makes our primitive upsampling strategy work better.
9

<!-- page 10 -->
Paper preprint
6
CONCLUSION
In this work, we introduce a general framework of geometry-constrained Gaussian splatting, with an
emphasize to exploit higher-order geometric information (e.g., curvature) that is largely neglected by
previous methods. The experiment results show that our framework significantly improves Gaussian
splatting and outperforms previous baselines, with very few artifacts in low-resource settings. Build-
ing on inherent geometric structures (e.g., local manifolds), we present efficient estimation methods
that provide our framework with noise-robust and dynamic geometric priors, thereby overcoming
the reliance on static and noise-sensitive priors in existing approaches.
REFERENCES
William K Allard. On the first variation of a varifold. Annals of mathematics, 95(3):417–491, 1972.
Theodore
Wilbur
Anderson,
Theodore
Wilbur
Anderson,
Theodore
Wilbur
Anderson,
Theodore Wilbur Anderson, and Etats-Unis Math´ematicien. An introduction to multivariate sta-
tistical analysis, volume 2. Wiley New York, 1958.
William J Anderson. Continuous-time Markov chains: An applications-oriented approach. Springer
Science & Business Media, 2012.
Lorenzo Bertini and Nicoletta Cancrini. The stochastic heat equation: Feynman-kac formula and
intermittence. Journal of statistical Physics, 78:1377–1401, 1995.
Sierra Bonilla, Shuai Zhang, Dimitrios Psychogyios, Danail Stoyanov, Francisco Vasconcelos, and
Sophia Bano.
Gaussian pancakes: geometrically-regularized 3d gaussian splatting for realis-
tic endoscopic reconstruction. In International Conference on Medical Image Computing and
Computer-Assisted Intervention, pp. 274–283. Springer, 2024.
Blanche Buet, Gian Paolo Leonardi, and Simon Masnou. A varifold approach to surface approxi-
mation. Archive for Rational Mechanics and Analysis, 226(2):639–694, 2017.
Blanche Buet, Gian Paolo Leonardi, and Simon Masnou. Weak and approximate curvatures of a
measure: a varifold perspective. Nonlinear Analysis, 222:112983, 2022.
Qing-Ming Cheng and Hongcang Yang. Estimates on eigenvalues of laplacian. Mathematische
Annalen, 331(2):445–460, 2005.
Ming Chuang, Linjie Luo, Benedict J Brown, Szymon Rusinkiewicz, and Michael Kazhdan. Es-
timating the laplace-beltrami operator by restricting 3d functions. In Computer graphics forum,
volume 28, pp. 1475–1484. Wiley Online Library, 2009.
Ronald R Coifman and St´ephane Lafon. Diffusion maps. Applied and computational harmonic
analysis, 21(1):5–30, 2006.
Pierre Comon, Xavier Luciani, and Andr´e LF De Almeida. Tensor decompositions, alternating least
squares and other tales. Journal of Chemometrics: A Journal of the Chemometrics Society, 23
(7-8):393–405, 2009.
Edward Brian Davies. Heat kernels and spectral theory. Number 92. Cambridge university press,
1989.
Manfredo P Do Carmo. Differential geometry of curves and surfaces: revised and updated second
edition. Courier Dover Publications, 2016.
Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaus-
sian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural
information processing systems, 37:140138–140158, 2024.
Herbert Federer. Geometric measure theory, volume Band 153 of Die Grundlehren der mathema-
tischen Wissenschaften. Springer-Verlag New York, Inc., New York, 1969.
10

<!-- page 11 -->
Paper preprint
Chen Feng, Yuichi Taguchi, and Vineet R Kamat. Fast plane extraction in organized point clouds
using agglomerative hierarchical clustering. In 2014 IEEE International Conference on Robotics
and Automation (ICRA), pp. 6218–6225. IEEE, 2014.
Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. Re-
lightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing.
In European Conference on Computer Vision, pp. 73–89. Springer, 2024.
Brian C Hall. Lie groups, lie algebras, and representations. In Quantum Theory for Mathematicians,
pp. 333–366. Springer, 2013.
Paul R Halmos. Measure theory, volume 18. Springer, 2013.
Ankur Handa, Thomas Whelan, John McDonald, and Andrew J Davison. A benchmark for rgb-d
visual odometry, 3d reconstruction and slam. In 2014 IEEE international conference on Robotics
and automation (ICRA), pp. 1524–1531. IEEE, 2014.
Junha Hyung, Susung Hong, Sungwon Hwang, Jaeseong Lee, Jaegul Choo, and Jin-Hwa Kim.
Effective rank analysis and regularization for enhanced 3d gaussian splatting. Advances in Neural
Information Processing Systems, 37:110412–110435, 2024.
Nandakishore Kambhatla and Todd K Leen. Dimension reduction by local principal component
analysis. Neural computation, 9(7):1493–1516, 1997.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splat-
ting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
Peter Lancaster and Kes Salkauskas. Surfaces generated by moving least squares methods. Mathe-
matics of computation, 37(155):141–158, 1981.
John M Lee. Introduction to Riemannian manifolds, volume 2. Springer, 2018.
Yanyan Li, Chenyu Lyu, Yan Di, Guangyao Zhai, Gim Hee Lee, and Federico Tombari. Geogaus-
sian: Geometry-aware gaussian splatting for scene rendering. In European Conference on Com-
puter Vision, pp. 441–457. Springer, 2025.
Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
18039–18048, 2024.
Ulrich Menne. The concept of varifold. Notices Amer. Math. Soc., 64(10):1148–1152, 2017. ISSN
0002-9920,1088-9477.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications
of the ACM, 65(1):99–106, 2021.
Simon Niedermayr, Josef Stumpfegger, and R¨udiger Westermann. Compressed 3d gaussian splatting
for accelerated novel view synthesis. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 10349–10358, 2024.
Patrick Perry and Michael W Mahoney. Regularized laplacian estimation and fast eigenvector ap-
proximation. Advances in Neural Information Processing Systems, 24, 2011.
Steven Roman, S Axler, and FW Gehring. Advanced linear algebra, volume 3. Springer, 2005.
Walter Rudin. Real and complex analysis. McGraw-Hill, Inc., 1987.
Erik Sandstr¨om, Yue Li, Luc Van Gool, and Martin R Oswald. Point-slam: Dense neural point cloud-
based slam. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp.
18433–18444, 2023.
Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings
of the IEEE conference on computer vision and pattern recognition, pp. 4104–4113, 2016.
11

<!-- page 12 -->
Paper preprint
Leon Simon. Lectures on geometric measure theory, volume 3 of Proceedings of the Centre for
Mathematical Analysis, Australian National University. Australian National University, Centre
for Mathematical Analysis, Canberra, 1983. ISBN 0-86784-429-9.
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image
recognition. arXiv preprint arXiv:1409.1556, 2014.
Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Im-
plicit neural representations with periodic activation functions. Advances in neural information
processing systems, 33:7462–7473, 2020.
Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel,
Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. The replica dataset: A digital replica of indoor
spaces. arXiv preprint arXiv:1906.05797, 2019.
Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto Seiskari, Esa Rahtu, and Juho Kannala.
Dn-splatter: Depth and normal priors for gaussian splatting and meshing. In 2025 IEEE/CVF
Winter Conference on Applications of Computer Vision (WACV), pp. 2421–2431. IEEE, 2025.
Paul Ungermann, Armin Ettenhofer, Matthias Nießner, and Barbara Roessle. Robust 3d gaussian
splatting for novel view synthesis in presence of distractors. In DAGM German Conference on
Pattern Recognition, pp. 153–167. Springer, 2024.
Michiel van den Berg and Jean-Franc¸ois Le Gall. Mean curvature and the heat equation. Mathema-
tische Zeitschrift, 215(1):437–464, 1994.
Evangelos Ververas, Rolandos Alexandros Potamias, Jifei Song, Jiankang Deng, and Stefanos
Zafeiriou. Sags: Structure-aware 3d gaussian splatting. In European Conference on Computer
Vision, pp. 221–238. Springer, 2025.
Jiepeng Wang, Yuan Liu, Peng Wang, Cheng Lin, Junhui Hou, Xin Li, Taku Komura, and Wen-
ping Wang. Gaussurf: Geometry-guided 3d gaussian splatting for surface reconstruction. arXiv
preprint arXiv:2411.19454, 2024.
Xiangyu Xu and Enrique Dunn. Discrete laplace operator estimation for dynamic 3d reconstruction.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October
2019.
Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion:
Dense tracking and mapping with voxel-based neural implicit representation. In 2022 IEEE In-
ternational Symposium on Mixed and Augmented Reality (ISMAR), pp. 499–507. IEEE, 2022.
Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang
Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal.
ACM Transactions on Graphics (TOG), 43(6):1–18, 2024.
Kˆosaku Yosida. Functional analysis, volume 123. Springer Science & Business Media, 2012.
12

<!-- page 13 -->
Paper preprint
A
REVIEW OF BASIC GEOMETRY
In this section, we will briefly review two different geometric structures: manifolds (Lee, 2018) in
differential geometry and varifolds (Allard, 1972; Menne, 2017) in geometric measure theory.
A.1
MANIFOLD IN RIEMANNIAN GEOMETRY
A smooth manifold M in Riemannian geometry (Lee, 2018) is associated with Riemannian metric
g. An informal understanding is to view it as a point set that locally looks like a vector space,
with inner products defined variously for different local spaces. In the rigorous formulation, such
a local space is conceptualized as the tangent plane TqM at every point q P M, with metric gq :
TqM ˆ TqM Ñ R as the inner product to measure both vector length }v}2 “ gqpv, vq, v P TqM
and similarity gqpv1, v2q, v1, v2 P TqM. All tangent planes share the same dimension, which is
consistent with the defined dimension DM P N of the global manifold, and their disjoint union, the
tangent bundle \qPMTqM, is denoted as T M. A smooth section V of the bundle (i.e., vector field
on the manifold) is a map that pairs every point q with a tangent vector Vpqq P TqM. We denote
the space of all smooth sections as VpMq, and represent the set containing all smooth function
f : M Ñ R as C8pMq. For the normal space NqM at each point q on the manifold M, it is easier
to define it in a Euclidean ambient space E “ RDE that isometrically embeds manifold M and has
a larger dimension DE ą DM. In that sense, the space NqM consists of normal vector n P E that
is perpendicular to the tangent space TqM.
A nice property of the Riemannian manifold is that its metric g uniquely determines an affine con-
nection ∇V1V2 : VpMqˆVpMq Ñ VpMq, which can be applied to further induce the Riemannian
curvature tensor RpV1, V2qV3 as
RpV1, V2qV3 “ ∇V1∇V2V3 ´ ∇V2∇V1V3 ´ ∇rV1,V2sV3,
(13)
where operator r, s represents the Lie bracket (Hall, 2013). This tensor defines all other types of
curvature. For example, the sectional curvature:
SecpV1, V2q “
gpRpV1, V2qV1, V2q
gpV1, V1qgpV2, V2q ´ gpV1, V2q2 ,
(14)
which measures how the manifold is curved in a 2D subspace.
A.2
ANOTHER TOOL: VARIFOLDS
While manifold is a popular geometric language, it relies on a strong assumption of smoothness.
The notion of varifold (Allard, 1972; Simon, 1983; Menne, 2017) is another mathematical tool that
requires only weak regularity. For the sake of clarity, we first focus on the simpler subclass of recti-
fiable varifolds. Given a countably DM-rectifiable set (Federer, 1969) M that is contained in an open
set ΩĂ E and a nonnegative function θ : M Ñ R that is positive HDM-almost everywhere, the rec-
tifiable varifold W is defined as the nonnegative Radon measure θHDM
|M bδTxM on the product space
MˆGDM,DE. Here HDM
|M denotes the DM-dimensional Hausdorff measure supported on M (Federer,
1969), GDM,DE is the Grassmannian manifold (Lee, 2018) formed by all DM-dimensional linear sub-
spaces of the ambient space E, and δTxM is the Dirac measure in GDM,DE supported on the tangent
plane TxM at x P M.
The action of W on a continuous function φ compactly supported in Ωˆ GDM,DE is
ż
ΩˆGDM,DE
φpq, TqdWpq, Tq “
ż
M
φpq, TqMqθpqqdHDMpqq.
(15)
Denoting as P the canonical projection from the product space M ˆ GDM,DE onto M, we define the
mass }W} of W as the Radon measure such that, for any Borel set B Ă Ω,
}W}pBq “ WpP´1pBqq “
ż
MXB
θpqqdHDMpqq.
(16)
13

<!-- page 14 -->
Paper preprint
A key concept in the theory of varifolds is the notion of first variation δW of a varifold W: for any
smooth compactly supported vector field X
δWpXq “
ż
ΩˆGDM,DE
divTXpqqdWpq, Tq,
(17)
where divT denotes the tangential divergence with respect to T. If the linear functional δW is locally
bounded, it follows from Riesz theorem (Yosida, 2012) and Radon-Nikodym theorem (Halmos,
2013) that it can be associated with a unique Radon measure that admits the decomposition
δW “ ´H}W} ` δSW,
(18)
where δSW is a vector-valued measure singular with respect to the mass }W}, and H is called the
generalized mean curvature. The latter coincides with the classical notion of mean curvature when
W is a rectifiable varifold associated with a smooth set and a locally constant multiplicity. More
generally, a generalized notion of second fundamental form can be defined for varifolds, see Buet
et al. (2022). The first variation of a point varifold is not locally bounded in general, so the above
decomposition does not hold. However, it is possible to define for point cloud varifolds consistent
notions of approximate mean curvature Buet et al. (2017) and approximate weak second fundamental
form, see Buet et al. (2022) and Appendix D.
B
MORE PRELIMINARIES ON MANIFOLD: UNIQUE CONNECTION
The affine connection ∇M that is compatible with the metric gM turns out to be unique. The Koszul
identity (Lee, 2018) provides an explicit formula for this connection. We show below its derivation
process. Based on the metric compatibility, we have
$
’
&
’
%
BM
U1gMpU2, U3q “ gMp∇M
U1U2, U3q ` gMpU2, ∇M
U1U3q
BM
U2gMpU3, U1q “ gMp∇M
U2U3, U1q ` gMpU3, ∇M
U2U1q
BM
U3gMpU1, U2q “ gMp∇M
U3U1, U2q ` gMpU1, ∇M
U3U2qq
.
(19)
By adding the first two equations and subtracting the last one, we get
BM
U1gMpU2, U3q ` BM
U2gMpU3, U1q ´ BM
U3gMpU1, U2q “ gMp∇M
U1U2 ` ∇M
U2U1, U3q
` gMp∇M
U1U3 ´ ∇M
U3U1, U2q ` gMp∇M
U2U3 ´ ∇M
U3U2, U1q.
(20)
In terms of the torsion-free condition:
∇M
U U1 ´ ∇M
U1U “ rU, U1s,
(21)
for any two vector fields U, U1, we can simplify the merged equation as
BM
U1gMpU2, U3q ` BM
U2gMpU3, U1q ´ BM
U3gMpU1, U2q “
gMp2∇M
U1U2 ` rU2, U1s, U3q ` gMprU1, U3s, U2q ` gMprU2, U3s, U1q.
(22)
By reorganizing this equality, we finally get
gMp∇M
U1U2, U3q “ 1
2
´
BM
U1gMpU2, U3q ` BM
U2gMpU3, U1q ´ BM
U3gMpU1, U2q`
gMprU1, U2s, U3q ´ gMprU1, U3s, U2q ´ gMprU2, U3s, U1q
¯
,
(23)
which holds for any smooth vector fields U1, U2, U3.
C
PROOF: CURVATURE-BASED SCALE INITIALIZATION
Given the curvature-based and area constraints as
si
1
si
2
“
ˇˇˇτ i
1
τ i
2
ˇˇˇ,
si
1si
2 “ psi
nbrq2
4
,
(24)
14

<!-- page 15 -->
Paper preprint
let us first solve the scale variables si
1, si
2. For the unknown scale si
1, we have
si
1 “
d´
si
1si
2
¯´si
1
si
2
¯
“
d
psi
nbrq2
4
ˇˇˇτ i
1
τ i
2
ˇˇˇ “ si
nbr
2
dˇˇˇτ i
1
τ i
2
ˇˇˇ.
(25)
Similarly, for the other one, we can derive
si
2 “
d´
si
1si
2
¯´
1
Msi
1
si
2
¯
“
d
psi
nbrq2
4
ˇˇˇτ i
2
τ i
1
ˇˇˇ “ si
nbr
2
dˇˇˇτ i
2
τ i
1
ˇˇˇ.
(26)
Then, we aim to analyze the second claim. In statistics, the one-sigma region of Gaussian primitive
Gi
obj is defined as the ellipsoid spanned by the rotation directions ri
1, ri
2, ri
3 of covariance matrix
Σi
obj, with its volume V measured by the scales si
1, si
2, si
3 as
V “ si
1si
2si
3 “ si
1si
2ξmin.
(27)
As Gaussian primitive Gi
obj in our setting is oriented in terms of the normal direction: ri
3 “ rui
3, the
area A of its projection to the tangent plane TqM can be simply measured as
A “ V
si
3
“ si
1si
2
1
4psi
nbrq2 “
´si
nbr
2
¯´si
nbr
2
¯
,
(28)
which is exactly the area before scale warm-up.
D
ADAPTED EXPRESSION OF VARIFOLD-BASED CURVATURES
Buet et al. (2022) introduced a notion of approximate second fundamental form (WSFF) for point
cloud varifolds, with convergence and consistency properties. The aim of this section is to adapt that
notion to our setting.
Method recap.
To obtain full curvature information from a varifold W, Buet et al. (2022) first
introduced a collection of G-linear variations indexed by i, j, k and defined by their action on C1
c pΩq
functions as
δi,j,kWpφq “
ż
ΩˆGDM,DE
Tj,kx∇Tφpqq, eiydWpq, Tq,
(29)
where every element T of the Grassmannian is identified with its orthogonal projection matrix pTjkq,
∇T represents the tangential gradient with respect to T, and ei denotes the i-th Euclidean basis
vector. We say that W has locally bounded variations whenever all G-linear variations are Radon
measures, in which case they can be decomposed as
δi,j,kW “ ´bi,j,k}W} ` δS
i,j,kW.
(30)
where bi,j,k P L1p}W}q and δS
i,j,kW are Radon measures singular with respect to }W}. When W
is a point cloud varifold associated with (point, tangent plane, mass) triplets tpqi, Ti, miquiPI, Buet
et al. (2022) introduced a consistent notion of approximate weak second fundamental form (WSFF)
defined as
$
’
’
&
’
’
%
Bi,j,kpql1q “
DM
2DE
ř
l2 ml2Υ1p
}ql1´ql2}
ϵ
qx
Tl2pql1´ql2q
}ql1´ql2} , tl1,l2,i,j,ky
ř
l2 ml2χp
}ql1´ql2}
ϵ
q
tl1,l2,i,j,k “ pTl2 ´ Tl1qj,kei ` pTl2 ´ Tl1qi,kej ´ pTl2 ´ Tl1qi,jek
,
(31)
where Υ, χ are suitable kernel functions, and ϵ ą 0 is the approximation scale. The approximate
curvature quantities can be finally obtained through the eigendecomposition of the matrix
Bpql1q “ sTJ
l1txBi,j,:pql1q, nl1yu1ďi,jďDM sTl1,
(32)
where i, j, : means taking all entries along the last index to form a vector, sTl1 denotes an orthonormal
basis of the tangent space Tl1, and nl1 is a normal vector.
15

<!-- page 16 -->
Paper preprint
Adaptation to our setting.
Now, let us adapt the above approximate WSFF B to our setting, fa-
cilitating practical computation. As the first step, we specify the points as the positions of Gaussian
primitives, M “ tµi
objuiPI, with each element µi
obj paired with a tangent plane rVi that is deter-
mined by the normal vector rui
3 as I ´ rui
3prui
3qJ. Here DM “ 2 for novel view rendering in the 3D
world E “ R3. For better notational consistencies, we also relabel the middle term Bi,j,kpql1q as
Bi
d1,d2,d3 (given ql1 “ µi
obj) and the subscript l2 as j.
Then, we reshape the inner product x¨, ti,j,d1,d2,d3y of term Bi based on its linearity:
p rVj ´ rViqd2,d3x rVjpµi
obj ´ µj
objq, ed1y ` p rVj ´ rViqd1,d3x¨, ed2y ´ p rVj ´ rViqd1,d2x¨, ed3y
}µi
obj ´ µj
obj}
.
(33)
As the tangential projection of a normal vector is zero, rVirui
3 “ 0, we have
sti,j,d1,d2 “
A!
x¨, ti,j,d1,d2,d3y}µi
obj ´ µj
obj}
)
1ďd3ďDE
, rui
3
E
“xp rVj ´ rViqd2,:, rui
3yx¨, ed1y ` xp¨qd1,:, rui
3yx¨, ed2y ´ p rVj ´ rViqd1,d2x rVjpµi
obj ´ µj
objq, rui
3y
“ x rVj
d2,:, rui
3yx¨, ed1y ` x rVj
d1,:, rui
3yx¨, ed2y ´ p rVj ´ rViqd1,d2x rVjpµi
obj ´ µj
objq, rui
3y.
(34)
With the above new term, we can convert the entry xBi
d1,d2,:, rui
3y into
ÿ
j
mjΥ1p}µi
obj ´ µj
obj}{ϵqsti,j,d1,d2
3}µi
obj ´ µj
obj} ř
k mkχp}µi
obj ´ µk
obj}{ϵq
,
(35)
given that DM “ 2, DE “ 3. Therefore, we express the approximate WSFF as an interpolated
matrix
Bi “
ÿ
j
mjΥ1p}µi
obj ´ µj
obj}{ϵq
`
pViqJsti,j,:,:Vi˘
3}µi
obj ´ µj
obj} ř
k mkχp}µi
obj ´ µk
obj}{ϵq
,
(36)
where Vi represents the tangential basis matrix rui
1; ui
2s in our setting. The last task is to simplify
the matrix multiplication term in the numerator:
Bi,j “ pViqJsti,j,:,:Vi
“ pViqJ´
rVjpµi
obj ´ µj
objqp rVjrui
3qJ ` rVjrui
3p¨qJ ` p rVj ´ rViqx rVjpµi
obj ´ µj
objq, rui
3y
¯
Vi
“ 2p rVjViqJsym
`
rui
3pµi
obj ´ µj
objqJ˘ rVjVi ` pµi
obj ´ µj
objqJ rVjrui
3ppViqJ rVjVi ´ Iq,
(37)
where sympAq denotes for any matrix A the operation 1
2pA ` AJq.
To summarize, the approximate WSFF adapted for our setting is
Bi “
´
ÿ
jPKNNpiq
mjχϵp}µi
obj ´ µj
obj}q
¯´1
ÿ
jPKNNpiq
mjΥ1
ϵp}µi
obj ´ µj
obj}q
3}µi
obj ´ µj
obj}
Bi,j,
(38)
where we relabel the original kernel functions as χϵ, Υ1
ϵ that depend on the approximation scale ϵ.
E
MANIFOLD STRUCTURE ESTIMATION FOR GAUSSIAN SPLATTING
Based on local manifold assumption (i.e., Definition 2.1), we aim to estimate multiple key geometric
properties (e.g., tangent vector) of the spatial object that hides behind 3D Gaussians tGi
objuiPI. Such
geometric information can be used to improve Gaussian splatting, as shown in Sec. 3.
There are several types of classical frameworks for estimating manifold structures from discrete data.
For example, moving least squares (Lancaster & Salkauskas, 1981) and heat equations (van den Berg
& Le Gall, 1994; Coifman & Lafon, 2006). Following the direction of heat equations, we provide
a full derivation procedure that shows how to practically estimate the manifold-related geometric
quantities mentioned in Sec. 2.2, considering both the positions and shapes of Gaussian primitives
in our problem setting. While this part should have overlaps with some previous works in this
classical direction, our main goal here is to make our method self-contained, providing practical
calculation guidance for future research in computer vision and graphics.
16

<!-- page 17 -->
Paper preprint
E.1
TANGENT VECTOR ESTIMATION
A 3D Gaussian Gi
obj essentially characterizes an uncertain point q “ µi
obj on the object surface,
with a covariance ellipse as Σi
obj. We will first study the estimation of tangent space TqM, which
is the basis of computing other geometric quantities (e.g., normal vectors).
E.1.1
DEFINITION OF CURVE-DERIVED TANGENT VECTORS
From the viewpoint of intrinsic geometry (Lee, 2018), the tangent vector vi
‹ P TqM at an arbitrary
point q “ µi
obj on the manifold M can be defined by a curve γi
‹ that passes through the point. This
definition in a strict sense relies on the concept of charts: which invertibly map a local manifold to
the Euclidean space, as the manifold M might be curved (i.e., non-zero curvature R). In the scenario
of Gaussian splatting, the potential manifold M naturally resides in an ambient space E “ R3, so
that we can simplify the definition as below.
Definition E.1 (Curve-based Tangent Vector Formulation). For an arbitrary vector vi
‹ that is tangent
to the embedded manifold M Ă E at a point µi
obj P M, there always exits a continuous curve
γi
‹ : R Ñ M that satisfies two conditions:
γi
‹p0q “ µi
obj,
Bsγi
‹psq|s“0 “ vi
‹.
(39)
Here the differential operator Bs is inherited from the Euclidean space E.
The significance of this definition is that we can convert the estimation of tangent vector vi
‹ to the
approximation of curve γi
‹psq on manifold M. The curve in differential calculus is zero-order, and
thus more tractable than the first-order tangent vector for a discrete point cloud.
There are infinitely many curves that can derive a tangent vector vi
‹ P TqM at point q “ µi
obj.
In the case of flat space, a natural choice is the coordinate curve γi
d that centers at point µi
obj and
moves along some dimension d P t1, 2, DE “ 3u. Formally speaking, this type of curve satisfies
both Eq. (39) and a new condition as
ϕd1pγi
dpsqq “ ϕd1pγi
dp0qq “ ϕd1pµi
objq, @d1 ‰ d, s P R,
(40)
where operation ϕd1p¨q “ r¨sd1 is to takes out the d1-th coordinate value from the input vector. If the
manifold M is curved, the above condition needs to be generalized as
γi
d “ arg max
γi‹:}vi‹}“cd
´
Bsϕdpγi
‹psqq|s“0
¯
,
(41)
where cd P R` is an arbitrary positive constant. This equation means that the curve heads in the
direction of maximizing its value at the d-th dimension. For the tangent vector derived by such a
coordinate curve γi
d, we denote it as vi
d.
E.1.2
ANALYSIS OF THE TANGENT SPACE STRUCTURE
For any point q “ µi
obj P M, the set of tangent vectors tvi
du1ďdďDE derived by coordinate curve
γi
‹psq has a size of DE “ 3, which is larger than the tangent space dimension: DTqM “ DM ă 3.
While this inequality is necessary, it is not sufficient to guarantee that the curve-derived vectors
tvi
dud span the entire tangent space TqM. A counter-example is that any two vectors differ only by
a scale factor, which cannot form a 2D plane.
From the perspective of linear algebra (Roman et al., 2005), to determine the rank of vector set
tvi
du1ďdďDE and extract independent components, it is critical to study the relation between two
tangent vectors vi
d1, vi
d2, d1 ‰ d2: inner product gq. In this regard, we construct the kernel matrix
Ki of inner product gqpvi
d1, vi
d2q, and confirm that the tangent space Tq is indeed spanned by curve-
derived vectors tvi
dud. The details are as follows.
Proposition E.2 (Tangential Kernel Analysis). For an arbitrary point q “ µi
obj on the embedded
Riemannian manifold M Ă E, the collection of curve-derived vectors tvi
du1ďdďDE span the entire
17

<!-- page 18 -->
Paper preprint
tangent space TqM, and the inner products between pairs of these vectors: gqpvi
d1, vi
d2q, 1 ď
d1, d2 ď DE “ 3, form a tangential kernel matrix Ki as
Ki :“
$
&
%
gqpvi
1, vi
1q
gqpvi
1, vi
2q
gqpvi
1, vi
3q
gqpvi
2, vi
1q
gqpvi
2, vi
2q
gqpvi
2, vi
3q
gqpvi
3, vi
1q
gqpvi
3, vi
2q
gqpvi
3, vi
3q
,
.
- ,
(42)
with its eigenvectors tui
du1ďdďDM that are paired with a positive eigenvalue λi
d ą 0 constituting
the basis of tangent space TqM.
Proof. The proof is provided in Appendix E.4.
We can see that the entire tangent space TqM is accessible once we can construct the kernel matrix
Ki. More significantly, we show that the unknown entries of this matrix can be easily computed
based on the zero-order curve γi
d that defines the first-order tangent vector vi
d.
Proposition E.3 (Curve-derived Riemannian Metric). For an embedded manifold M ãÑ E and the
coordinate-derived tangent vectors tvi
du1ďdďDE (as formulated in Definition E.1 and Eq. (41)) at
any point q “ µi
obj P M, the metric gq that takes two tangent vectors vi
d1, vi
d2, 1 ď d1, d2 ď DE
as its arguments can be expressed as
gqpvi
d1, vi
d2q “ 1
2Lbnz∆rϕd1, ϕd2spqq,
(43)
where ∆denotes the Laplacian operator on the function space C8pMq, and Lbnz□characterizes
how much it violates the Leibniz rule: □rϕd1ϕd2s ´ ϕd1□rϕd2s ´ ϕd2□rϕd1s.
Proof. The proof is provided in Appendix E.5.
The significance of this conclusion is a special expression of metric gq that only depends on the
accessible zero-order curve coordinate (e.g., ϕd1pqq “ rγi
d1p0qsd1 “ rµi
objsd1), without involving
the intractable first-order tangent vectors (e.g., vi
d2). While this expression still introduces a new
unknown term: Laplacian operator ∆, it can be efficiently estimated from the raw data tGi
objuiPI
through many mature techniques (Perry & Mahoney, 2011; Cheng & Yang, 2005; Chuang et al.,
2009; Xu & Dunn, 2019) in the literature.
E.1.3
LAPLACIAN OPERATOR ESTIMATION
The Laplacian operator ∆in the metric reformulation (i.e., Eq. (43)) is a type of linear operator L
in the function space C8pMq, satisfying
L raf ` bhs “ aL rfs ` bL rhs, @a, b P R, @f, g P C8pMq.
(44)
The estimation of linear operator L on discrete data points is a well-studied field. In the same
spirit as the Feynman-Kac formula (Bertini & Cancrini, 1995), we adopt the following Monte Carlo
approximation to efficiently evaluating the linear operator L rfs, f P C8pMq.
Proposition E.4 (Markov-chain Linear Operator Estimation). Suppose that the Gaussian centers
tci
objuiPI are uniformly sampled from the Riemannian manifold M, and linear operator L is the
infinitesimal generator of a homogeneous Markov chain. For any smooth function f P C8pMq, the
evaluation of operator L has an unbiased approximation:
L rfspqq « C
t|I|
ÿ
iPI
´
fpci
objq ´ fpqq
¯
ktpq, ci
objq,
(45)
as t converges to 0. Here C ą 0 is a constant that only depends on the manifold M, and ktp¨q is the
transition kernel of the Markov chain.
Proof. The proof is provided in Appendix E.6.
18

<!-- page 19 -->
Paper preprint
This conclusion is to estimate the linear operator L specified by its kernel kt. For approximating
the Laplacian operator L “ ∆, we can set the kernel kt as Gaussian (Davies, 1989):
ktpq1, q2q9 exp
´
´ 1
t }q1 ´ q2}2
2
¯
,
(46)
which is normalized by the integral over q1 or q2. In practical implementation, the sum operation
ř of Eq. (45) will be time-consuming if it runs over a large index set I. As the Gaussian kernel kt
exponentially decays with respect to the point distance } ¨ }2
2, we can limit the set as a few nearest
neighbors of the point q to improve efficiency and reduce noise.
Besides, a point set tGi
objuiPI might not be evenly distributed, with some sparse regions. This
problem can be addressed by adopting a bandwidth-adaptive kernel:
$
&
%
ktpq1, q2q9 exp
´
´ pq1 ´ q2qJΛtpq1, q2q´1pq1 ´ q2q
¯
Λtpq1 “ ci
obj, q2 “ cj
objq “ tΣi
objΣj
obj
.
(47)
Intuitively, this kernel kt is more sensitive to the neighbor q2 of a point q1 “ ci
obj if the correspond-
ing covariance matrix Σi
obj is of a large scale.
E.2
DIMENSION AND NORMAL VECTOR ESTIMATIONS
Starting with the curve-derived tangent vectors tvi
du1ďdďDE (as defined in Sec. E.1.1), we can easily
induce the normal vector n P NqM at any point q “ µi
obj on the manifold M. For example, if the
manifold is 2-dimensional: DM “ 2, then the unit norm n is unique and we can derive it from any
two tangent vectors vi
d1, vi
d2, 1 ď d1, d2 ď DE that are not co-linear:
n “ pvi
d1 ˆ vi
d2q{}vi
d1 ˆ vi
d2}2,
(48)
where ˆ denotes the vector cross product in three dimensions (DE “ 3). More conveniently, both
manifold dimension DM and normal space NqM are just side products from the kernel matrix
decomposition in Theorem E.2. The details are below.
Corollary E.5 (Induced Orthogonal Space). For any point q “ µi
obj on the embedded Riemannian
manifold M ãÑ E, the number of positive eigenvalue λd ą 0 of kernel matrix Ki (as specified by
Eq. (42)) is equal to the manifold dimension DM. More notably, the eigenvectors tui
duDM`1ďdďDE
that correspond to zero eigenvalue λd “ 0 form the basis of normal space NqM.
Proof. The proof is provided in Appendix E.7.
While the tangential kernel matrix Ki seems to incur more computations than the cross product ˆ,
the matrix is of a very small shape (i.e., 3 ˆ 3) and its factorization can be analytical solved, instead
of resorting to inefficient iteration algorithms (e.g., Alternating Least Squares (Comon et al., 2009)).
Plus, to distinguish more from the notations of tangent basis tui
du1ďdďDM, we might denote the
normal basis as trui
duDM`1ďdďDE to emphasize its directions.
E.3
CURVATURE ESTIMATION
From the angle of extrinsic differential geometry, the curvature is a geometric quantity that charac-
terizes how a hypersurface M bends within its ambient space E. There are various types of curvature
definitions (e.g., sectional curvature Sec), and they are all determined by the shape operator (Lee,
2018; Do Carmo, 2016) at every point q “ µi
obj on the maniold M:
sipn, u1, u2q : NqM ˆ TqM ˆ TqM Ñ R,
(49)
which conditions on a normal vector n and maps a pair of tangent vectors u1, u2 to a scalar. In the
case of Euclidean space E, it is intuitive that this operator ∫i is always valued as 0, indicating that
the entire space is flat. For a curved manifold M, the operator for curved-derived tangent vectors
tvi
du1ďdďDE can be expressed as follows.
19

<!-- page 20 -->
Paper preprint
Proposition E.6 (Curve-derived Shape Operator). For an arbitrary point q “ µi
obj on the embed-
ded Riemannian manifold M ãÑ E and any pair of curved-derived tangent vectors vi
d1, vi
d2, 1 ď
d1, d2 ď DE, the shape operator si for a normal vector n P NqM takes the form as
sipn, vi
d1, vi
d2q “ 1
8
´
Arϕd1, η, ϕd2spqq ` Arϕd2, η, ϕd1spqq ´ Arη, ϕd1, ϕd2spqq
¯
,
(50)
where Ar¨, ¨, ¨s is a second-order differential operator, with its value for any three functions f1, f2, f3
as nested Leibniz rules Lbnz∆rf1, Lbnz∆rf2, f3ss, and η P C8pMq is the height function, with its
value at any point q1 P M as the Euclidean inner product xn, q1 ´ qyE.
Proof. The proof is provided in Appendix E.8.
In the same spirit as Proposition E.3, this conclusion permits us to compute the shape operator si
with only the zero-order curves tvi
dudPr1,DEs. The Laplacian operator is typically unknown as in the
case of tangent space estimation, and we can get it from Proposition E.4.
With the shape operator si, we can compute the principal curvatures of manifold M through matrix
decomposition. The formal conclusion is as follows.
Theorem E.7 (Shape Operator Factorization). For any point q “ µi
obj on the embedded Rieman-
nian manifold M ãÑ E and a fixed normal vector n P NqM, the matrix:
Si :“ tsipn, ui
d1, ui
d2qu1ďd1,d2ďDM,
(51)
formed by the shape operator si and orthonormal tangent vectors Ui “ tui
du1ďdďDM, is with
eigenvalue τ i
d, 1 ď d ď DM corresponding to the curvature value and realigned eigenvector rwi
d :“
Uiwi
d, 1 ď d ď DM associated with the principle direction.
Proof. The proof is provided in Appendix E.9.
The dimension DM in 3D rendering is typically small than 3, so the eigen-decomposition of matrix
Si can be solved even analytically. The orthonormal matrix Ui involved in computation can also be
obtained from previous Proposition E.2.
E.4
PROOF: ESTIMATION OF THE TANGENT SPACE
The geometric analysis in this part is divided into three parts. We will first derive some basic terms
(e.g., metric) in terms of embedded Riemannian geometry. Then, we will characterize the set of
coordinate-derived tangent vectors tvi
du1ďdďDE and their tangential kernel matrix Ki.
E.4.1
PREPARATIONS
Let us first respectively denote the connections of Riemannian manifold M and Euclidean space E as
∇M, ∇E. Given that the manifold M is embedded in the ambient space E, an important conclusion
in differential geometry (Lee, 2018) is that
∇M
U1U2 “ p∇E
U1U2qπ, @U1, U2 P V.
(52)
Another key fact is that the metric gM
q
of any tangent space TqM can be induced by the inner
product gE
q of the ambient space E as
gM
q pu1, u2q “ gE
ιpqqpBιqru1s, Bιqru2sq “ gE
qpu1, u2q “ uJ
1 u2,
(53)
where u1, u2 P TqM, ι : M ãÑ E denotes the inclusion map, and Bιqr¨s represents the pushforward
operation at point q. With the above two conclusions, we can relate the gradient operator gradM of
the embedded manifold M to that gradE of the Euclidean space E. For any function f defined on
the manifold M, its gradient gradMf is formulated as
BM
u fpqq “ gM
q pgradM
q f, uq “ pgradM
q fqJu, @u P TqM,
(54)
20

<!-- page 21 -->
Paper preprint
where the first term BM
u fpqq represents the derivative of function f following the direction u. The
same conclusion applies to the Euclidean space E as
BE
ufpqq “ gE
qpgradE
qf, uq “ pgradE
qfqJu, @u P E.
(55)
Since the manifold M is embedded in the Euclidean space E, the tangent space TqM (i.e., a plane
that is tangent to the manifold M at point q) is also a subset of ambient space E. For any tangent
vector u P TqM Ă E, we can prove that the two differential operations BM
u , BE
u are equivalent based
on Definition E.1. Suppose that some smooth curve γ : R Ñ M on the manifold M derives this
tangent vector u, which means
γp0q “ q,
lim
ϵÑ0
γpϵq ´ γp0q
ϵ
“ u,
(56)
then the derivative BM
u fpqq of a smooth function f can be formulated as
BM
u fpqq “ lim
ϵÑ0
fpγpϵqq ´ fpγp0qq
ϵ
.
(57)
Because the manifold-valued curve γ Ď M also resides in the ambient space E, then the expression
at the right hand side can also define the term BE
ufpqq. Therefore, we have
pgradM
q fqJu “ BM
u fpqq “ BE
ufpqq “ pgradE
qfqJu.
(58)
Because this equality holds for any tangent vector u P TqM, then we can infer that the leftmost term
gradM
q f is the projection of the rightmost term gradE
qf on the tangent plane TqM. More formally,
we specify this relation as
gradM
q f “ pgradE
qfqπ.
(59)
The projection operation π here is defined in terms of the square norm } ¨ }2
2.
Secondly, let us derive the concrete form of coordinate-derived tangent vectors tvi
du1ďdďDE. For
some coordinate dimension d, this type of tangent vector vi
d represents the direction that the function
ϕdp¨q “ r¨sd increases the most. For a flat space E, the non-unit version rvi
d of the coordinate-derived
vector can be computed with vanilla differential calculus:
rvi
d “ gradE
qr¨sd “ lim
ϵÑ0
q ` ϵ ¨ ed ´ q
ϵ
“ ed,
(60)
where ed represents a basis vector in the Euclidean space E that has an entry as 1 at the d-th di-
mension and is with 0 in all other entries. For a curved embedded manifold M ãÑ E, the gradient
projection formula (i.e., Eq. (59)) suggests that the term rvi
d can be generalized as
rvi
d “ gradM
q r¨sd “ pgradE
qr¨sdqπ “ eπ
d.
(61)
The last projection term eπ
d can be further expanded as
ed ´
ÿ
DMăd1ďDE
gM
q pnd1, edqnd1 “ ed ´
ÿ
DMăd1ďDE
pnJ
d1edqnd1,
(62)
where tnd1ud1PpDM,DEs is a orthonormal basis in the normal space NqM. Combining the above
two equations, we get a concrete expression for the vector rvi
d as
rvi
d “ ed ´
ÿ
DMăd1ďDE
pnJ
d1edqnd1,
(63)
with its normalized version rvi
d{}rvi
d}2 as the coordinate-derived tangent vector vi
d.
E.4.2
RANK DETERMINATION
Then, we prove that coordinate-derived tangent vectors tvi
du1ďdďDE span the entire tangent space
TqM at point q “ µi
obj P M. It suffices to verify this fact for the collection of unnormalized
vectors trvi
du1ďdďDE. Note that the projection operation π is linear, so it can be represented by a
matrix. In terms of Eq. (63), we can reformulate it as
rvi
d “ Ied ´
ÿ
DMăd1ďDE
pnd1nJ
d1qed “
´
I ´
ÿ
DMăd1ďDE
pnd1nJ
d1q
¯
ed “ Led,
(64)
21

<!-- page 22 -->
Paper preprint
where I denotes the identity matrix and L is the concrete form of projection π. This matrix L is of
rank DM. Specifically, for any normal vector n‹, ‹ P pDM, DEs, we have
Ln‹ “
´
I ´
ÿ
DMăd1ďDE
pnd1nJ
d1q
¯
n‹ “ n‹ ´
ÿ
DMăd1ďDE
pnJ
d1n‹qnd1 “ n‹ ´ n‹ “ 0.
(65)
For any tangent vector u P TqM, we also have
Lu “
´
I ´
ÿ
DMăd1ďDE
pnd1nJ
d1q
¯
u “ u ´
ÿ
DMăd1ďDE
pnJ
d1uqnd1 “ u.
(66)
Therefore, any normal vector n‹ is an eigenvector of matrix L with the eigenvalue as 0, and any
tangent vector is also an eigenvector with the eigenvalue as 1. Clearly, the projection matrix L
has a rank as DM. Unnormalized vectors trvi
du1ďdďDE are actually computed from the Euclidean
basis tedu1ďdďDE with respect to the projection matrix L. Given that the basis tedud are mutually
independent and the matrix L is of rank DM, the vectors trvi
dud must have a rank as DM, and thus
they span the tangent plane TqM.
The last claim might not be very straightforward. We provide a simple proof for it to end this
subsection. By contradiction, suppose that the set of vectors trvi
du1ďdďDE are not able to span the
entire tangent plane TqM, then there exists a non-zero tangent vector u P TqM that is orthogonal
to them: gM
q pu, rvi
dq “ 0, @d P r1, DEs. Equivalently, we have
0 “ uJrvi
d “ uJpLedq “ puJLqed.
(67)
Since this equality is true for any Euclidean basis ed, then we can infer that uJL “ 0J. Note that
the matrix L is symmetric, we have
0 “ LJu “ Lu,
(68)
which contradicts Eq. (66). Hence, the set trvi
dud must span the whole space TqM.
E.4.3
KERNEL MATRIX ANALYSIS
Lastly, let us study the part with kernel matrix Ki. Note that the constant cd in Eq. (41) can be set as
an arbitrary number, so we do not distinguish the unit vector vi
j, j P r1, DEs from its unnormalized
version rvi
j in this subsection. A key property of this matrix is positive semidefiniteness. For any
vector ui “ rui
1, ui
2, ¨ ¨ ¨ , ui
DMsJ P RDE, please note that
puiqJKiui “
ÿ
1ďj,kďDE
gM
q pvi
j, vi
kqui
jui
k “
ÿ
1ďjďDE
ui
j
´
ÿ
1ďkďDE
ui
kgM
q pvi
j, vi
kq
¯
“
ÿ
1ďjďDE
ui
jgM
q pvi
j,
ÿ
1ďkďDE
ui
kvi
kq “ gM
q p
ÿ
1ďjďDE
ui
jvi
j,
ÿ
1ďkďDE
ui
kvi
kq ě 0,
(69)
where the last inequality holds because the inner product is positive-definite. Therefore, the property
is indeed true for kernel matrix Ki. Furthermore, suppose this vector ui is a non-zero eigenvector
of the matrix Ki, with the corresponding eigenvalue as λi, then we have
Kiui “ λiui.
(70)
Taking the Euclidean inner product with the same vector ui on both sides, we get
puiqJKiui “ λipuiqJui ùñ λi “ puiqJKiui
puiqJui
ě 0.
(71)
Therefore, the eigenvalue λi of tangential kernel matrix Ki is always positive or zero.
For a pair of eigenvector ui
‹ P RDE and eigenvalue λi
‹ P R, we will show that this eigenvector
corresponds to a tangent vector if the eigenvalue is positive (i.e., λi
‹ ą 0), otherwise it is paired with
a normal vector. Let us first look into the second case. Based on Eq. (70), we have
0 “ r0 ¨ ui
‹sd “ rKiui
‹sd “
ÿ
1ďjďDE
gM
q pvi
d, vi
jqui
‹,j “ gM
q pvi
j,
ÿ
1ďjďDE
ui
‹,jvi
jq,
(72)
22

<!-- page 23 -->
Paper preprint
where 1 ď d ď DE and ui
‹,j “ rui
‹sd. As this equality holds for any subscript d and the vector set
tvi
du1ďdďDE span the tangent space TqM, we can infer
ÿ
1ďjďDE
ui
‹,jvi
j “ 0.
(73)
By incorporating Eq. (63), we can expand the right hand side as
0 “
ÿ
1ďjďDE
ui
‹,j
´
ej ´
ÿ
DMăj1ďDE
pnJ
j1ejqnj1
¯
.
(74)
By further reshaping the equation, we can get
ÿ
1ďjďDE
ÿ
DMăj1ďDE
ui
‹,jpnJ
j1ejqnj1 “
ÿ
1ďjďDE
ui
‹,jej “ ui
‹,
(75)
indicating that the eigenvector ui
‹ with a zero eigenvalue λi
‹ “ 0 is a normal vector in the space
NqM. On the other hand, note that any normal vector ui
‹ P NqM makes Eq. (74) hold, so it is
an eigenvector of matrix Ki with a zero eigenvalue. In light of this fact, we can infer that every
eigenvector paired with a positive eigenvalue λi
‹ ą 0 is perpendicular to the normal space, and thus
tangent to the manifold: ui
‹ P TqM, which proves the first case.
Finally, we show that the positive eigenvalue can only be 1. If λi
‹ ą 0, we have
λi
‹ “ λi
‹pui
‹qJui
‹ “ pui
‹qJKiui
‹ “
ÿ
1ďj,kďDE
gM
q pvi
j, vi
kqui
‹,jui
‹,k
“ gM
q p
ÿ
1ďjďDE
ui
‹,jvi
j,
ÿ
1ďkďDE
ui
‹,kvi
kq.
(76)
In this regard, there exists a unit tangent vector sui
‹ such that
a
λi‹sui
‹ “
ÿ
1ďdďDE
ui
‹,dvi
d.
(77)
In terms of Eq. (70), the right hand side can be expanded as
a
λi‹sui
‹ “
ÿ
1ďdďDE
ui
‹,d
´
ed ´
ÿ
DMăd1ďDE
pnJ
d1edqnd1
¯
“
ÿ
1ďdďDE
ui
‹,ded ´
ÿ
1ďdďDE
ÿ
DMăd1ďDE
ui
‹,dpnJ
d1edqnd1
“ ui
‹ ´
ÿ
DMăd1ďDE
pnJ
d1ui
‹qnd1.
(78)
Note that the last sum term should be zero, as previously we prove that ui
‹ P TqM. Therefore, we
get
a
λi‹sui
‹ “ ui
‹, indicating that λi
‹ “ 1. The key here is that the vectors sui
‹, ui
‹ in both sides are
normalized: }sui
‹}2 “ 1, }ui
‹}2 “ 1.
E.5
PROOF: CURVE-BASED FORMULATION OF THE RIEMANNIAN METRIC
To make our proof more readable, we will first derive the second-order form of the Riemannian
metric for the Euclidean space, and then dive into the more general manifold.
E.5.1
ANALYSIS ON THE EUCLIDEAN SPACE
Let us first study the metric form in the flat space, and then dive into the curved case. In the Euclidean
case E, the metric gEpu1, u2q is the same everywhere: inner product uJ
1 u2 “ ř
i u1,iu2,i, and the
Laplacian operator ∆E is a sum of the second-order differential operators:
gEpu1, u2q “ uJ
1 u2 “
ÿ
1ďiďDE
u1,iu2,i,
∆E “
ÿ
1ďiďDE
B2
ei,
(79)
23

<!-- page 24 -->
Paper preprint
where ei represents the i-th Euclidean basis. For any two smooth functions f, h : E Ñ R, we
iteratively apply the Leibniz rule as
∆Erfhs “
ÿ
1ďiďDE
B2
eipfhq “
ÿ
1ďiďDE
BeiphBeif ` fBeihq
“
ÿ
1ďiďDE
BeiphBeifq ` BeipfBeihq
“
ÿ
1ďiďDE
´
BeihBeif ` hB2
eif
¯
`
ÿ
1ďiďDE
´
BeifBeih ` fB2
eih
¯
“ 2
ÿ
1ďiďDE
BeifBeih ` h
ÿ
1ďiďDE
B2
eif ` f
ÿ
1ďiďDE
B2
eih
“ 2gEpgradEf, gradEhq ` h∆Erfs ` f∆Erhs,
(80)
By rearranging the terms, we arrive at an equation as
gEpgradEf, gradEhq “ 1
2
´
∆Erfhs ´ h∆Erfs ´ f∆Erhs
¯
.
(81)
In essence, the right hand side measures how much the linear operator ∆E violates the Leibniz rule.
Based on this equality, our previous conclusion (i.e., Eq. (60)) indicates that
gE
qpvi
d1, vi
d2q “ gE
qpgradE
qϕd1, gradE
qϕd2q
“ 1
2
´
∆Erϕd1ϕd2spqq ´ ϕd1pqq∆Erϕd2spqq ´ ϕd2pqq∆Erϕd1spqq
¯
,
(82)
where vi
d1, vi
d2 are two coordinate-derived tangent vectors at point q “ µi
obj.
Importantly, we can see that the final conclusion is similar to the well-known Leibniz rule. To
simplify the notation, we specify the Leibniz operator Lbnz□as
Lbnz□rf1, f2s “ □rf1f2s ´ f1□rf2s ´ f2□rf1s,
(83)
while □is a certain linear operator. Under this scheme, the conclusion now is as
gE
qpvi
d1, vi
d2q “ 1
2Lbnz∆Erϕd1, ϕd2spqq,
(84)
where d1, d2 are two arbitrary numbers from t1, 2, ¨ ¨ ¨ , DEu.
E.5.2
ANALYSIS ON THE CURVED MANIFOLD
Then, for a curved manifold M, the metric gM
q
varies point by point in notation and we need to
generalize the Laplacian operator. Specifically, a general definition of this operator is that it stands
as the divergence divM of the gradient field gradMf of a function f:
∆Mrfs “ divMpgradMfq.
(85)
Let us begin with deriving the concrete form of gradient term gradMf. Formally speaking, suppose
that the basis fields are as tXM
d
: q ÞÑ TqMu1ďdďDM and the coefficient fields for this gradient
term are as tbd : M Ñ Ru1ďdďDM, then we can get the expanded form as
gradMf “
ÿ
1ďdďDM
bdXd.
(86)
In terms of Eq. (54), we further have
BM
Xd1 f “ gMpgradMf, Xd1q “
ÿ
1ďd,d1ďDM
bdgMpXd, Xd1q
“
ÿ
1ďd,d1ďDM
bdsgd,d1 “ rbJ sGsd1,
(87)
24

<!-- page 25 -->
Paper preprint
where d1 is an integer from r1, DMs and sG is a 2-dimensional tensor of metric coefficients tsgd,d1 –
gMpXd, Xd1qu1ďd,d1ďDM. By raising the metric index, we finally get
ÿ
1ďd1ďDM
BM
Xd1f ¨ sgd1,d “
ÿ
1ďd1ďDM
rbJ sGsd1sgd1,d “ rbJ sG sG´1sd “ bd,
(88)
where sgd1,d is the entry of inverse matrix sG´1 at the d1-th row and d-th column. For an arbitrary
vector field U P VpMq, the general definition of divergence operator divM relying on its expanded
form ř
1ďdďDM udXd is as
divMU “
ÿ
1ďj,kďDM
sgj,kgMp∇M
XjU, Xkq
“
ÿ
1ďj,kďDM
sgj,kgM´
∇M
Xj
´
ÿ
1ďdďDM
udXd
¯
, Xk
¯
“
ÿ
1ďj,kďDM
sgj,kgM´
ÿ
1ďdďDM
´
BM
Xjud ¨ Xd ` ud∇M
XjXd
¯
, Xk
¯
“
ÿ
1ďj,k,dďDM
sgj,kpBM
XjudqgMpXd, Xkq `
ÿ
1ďj,k,dďDM
sgj,kudgM´
∇M
XjXd, Xk
¯
“
ÿ
1ďj,k,dďDM
sgj,kpBM
Xjudqsgd,k `
ÿ
1ďj,k,dďDM
sgj,kudgM´
∇M
XjXd, Xk
¯
“
ÿ
1ďj,dďDM
BM
Xjud ¨ δj,d `
ÿ
1ďj,k,dďDM
sgj,kudgM´
∇M
XjXd, Xk
¯
.
(89)
where the third equality is derived based on the Leibniz rule and δj,d is the Kronecker delta function.
The further analysis of term ∇M
XjXd relies on the Christoffel symbols ΓM
j,k,d, 1 ď j, k, d ď DM for
the connection ∇M, which specifies
∇M
XjXk “
ÿ
1ďdďDM
ΓM
j,k,dXd.
(90)
Combining the above two equations, we have
divMU “
ÿ
1ďjďDM
BM
Xjuj `
ÿ
1ďj,k,dďDM
sgj,kudgM´
ÿ
1ďd1ďDM
ΓM
j,d,d1Xd1, Xk
¯
.
(91)
Similar to the first term in the right hand side, we can simplify the second term as
ÿ
1ďj,k,d,d1ďDM
sgj,kudΓM
j,d,d1gMpXd1, Xkq “
ÿ
1ďj,k,d,d1ďDM
sgj,kudΓM
j,d,d1sgd1,k
“
ÿ
1ďj,d,d1ďDM
udΓM
j,d,d1r sG´1 sGsj,d1 “
ÿ
1ďj,d,d1ďDM
udΓM
j,d,d1δj,d1 “
ÿ
1ďj,dďDM
udΓM
j,d,j.
(92)
Therefore, we get the concrete form of divergence operator divM as
divMU “
ÿ
1ďdďDM
BM
Xdud `
ÿ
1ďj,dďDM
ΓM
j,d,jud.
(93)
By incorporating this formula, we can compute the Laplacian of a function as
∆Mrfs “ divM´
ÿ
1ďdďDM
bdXd
¯
“
ÿ
1ďdďDM
BM
Xdbd `
ÿ
1ďj,dďDM
ΓM
j,d,jbd
“
ÿ
1ďdďDM
BM
Xd
´
ÿ
1ďd1ďDM
BM
Xd1f ¨ sgd1,d¯
`
ÿ
1ďj,dďDM
ΓM
j,d,j
´
ÿ
1ďd1ďDM
BM
Xd1f ¨ sgd1,d¯
“
ÿ
1ďd,d1ďDM
sgd1,dBM
XdBM
Xd1 f `
ÿ
1ďd,d1ďDM
´
BM
Xdsgd1,d `
ÿ
1ďjďDM
ΓM
j,d,jsgd1,d¯
BM
Xd1f.
(94)
Like in the Euclidean space, we are interested in the following expression:
∆Mrfhs ´ f∆Mrhs ´ h∆Mrfs.
(95)
25

<!-- page 26 -->
Paper preprint
We can anticipate that the second sum of every Laplacian term crosses out, since the first-order
differential operator BM
Xd1 follows the Leibniz rule:
BM
Xd1pfhq “ fBM
Xd1h ` hBM
Xd1f.
(96)
For the first sum of each Laplacian term, we repeatedly apply the Leibniz rule to it as
ÿ
1ďd,d1ďDM
sgd1,d´
BM
XdBM
Xd1pfhq ´ fBM
XdBM
Xd1 h ´ hBM
XdBM
Xd1f
¯
“
ÿ
1ďd,d1ďDM
sgd1,d´
BM
Xd
´
fBM
Xd1 h ` hBM
Xd1f
¯
´ fBM
XdBM
Xd1h ´ hBM
XdBM
Xd1f
¯
“
ÿ
1ďd,d1ďDM
sgd1,d´
BM
Xdf ¨ BM
Xd1h ` BM
Xdh ¨ BM
Xd1f
¯
“ 2
ÿ
1ďd,d1ďDM
sgd1,dpBM
Xdf ¨ BM
Xd1hq,
(97)
where the second last equality is derived based on the symmetry of inverse Riemannian metric sgd1,d.
Combining the above 4 equations, we have
∆Mrfhs ´ f∆Mrhs ´ h∆Mrfs “ 2
ÿ
1ďd,d1ďDM
sgd1,dpBM
Xdf ¨ BM
Xd1 hq
“ 2
ÿ
1ďd,d1ďDM
sgd1,dpgMpgradMf, Xdq ¨ BM
Xd1hq
“ 2gMpgradMf,
ÿ
1ďd,d1ďDM
sgd1,dpBM
Xd1hqXdq “ 2gMpgradMf, gradMgq,
(98)
where the last equality is derived based on Eq. (88). By rearranging the terms, we get
gMpgradMf, gradMgq “ 1
2
´
∆Mrfhs ´ f∆Mrhs ´ h∆Mrfs
¯
.
(99)
Note that this equality is the same as Eq. (81), despite the superscript difference. Hence, the conclu-
sion (i.e., Eq. (82)) in the Euclidean space E can be generalized to the curved space M. The above
equation is field-based, and we can simplify it to a point-based expression as
gM
q pvi
d1, vi
d2q “
´
∆Mrϕd1ϕd2spqq ´ ϕd1pqq∆Mrϕd2spqq ´ ϕd2pqq∆Mrϕd1spqq
¯
,
(100)
where ϕd1, ϕd2 correspond to f, g and vi
d1, vi
d2 are their gradients at point q. For notational conve-
nience, we can further simplify the conclusion with the Leibniz symbol Lbnz□as
gM
q pvi
d1, vi
d2q “ 1
2Lbnz∆Mrϕd1, ϕd2spqq,
(101)
where d1, d2 are any two numbers from t1, 2, ¨ ¨ ¨ , DEu.
E.6
PROOF: MARKOVIAN CHARACTERIZATION OF THE LINEAR OPERATOR
Markov-chain generator.
A key conclusion in the theory of Markov chain (Bertini & Cancrini,
1995; Anderson, 2012) is that any homogeneous continuous-time Markov process tZtu0ďtďă8
forms a semi-group in terms of its impact on a smooth function f : S Ñ R, where S denotes
the state space. Formally speaking, if we define a class of operators tEtutě0 on the function f as
Etrfspzq “ ErfpZtq | Z0 “ zs,
(102)
then those operators form a semi-group:
E0 “ Id,
Es`t “ Es ˝ Et,
(103)
where s, t are arbitrary non-negative numbers and symbol ˝ denotes the operator composition. In
this situation, it is proved that such a class of operators have an analytical function:
Et “ expptL q “
ÿ
kě0
tk
k!
´
L ˝ L ˝ ¨ ¨ ¨ ˝ L
¯
k-time composition
,
(104)
26

<!-- page 27 -->
Paper preprint
where L is a linear operator named as the infinitesimal generator. The evaluation of this linear
operator L on function f can also be computed as
L rfspzq “ lim
tÑ0
1
t
´
Etrfspzq ´ fpzq
¯
“ lim
tÑ0
1
t
´
ErfpZtq | Z0 “ zs ´ fpzq
¯
.
(105)
Let us consider the expectation term on the right hand side: it can be expanded as
Er¨s “
ż
fpz1qPpZt “ z1 | Z0 “ zqµpdz1q,
(106)
where the inside conditional probability term Pp¨q is called the Markov kernel, which is typically
re-symbolized as ktpz, z1q, z, z1 P S, and µ is some volume measure defined on the abstract space S
(e.g., smooth manifold M). Obviously, the kernel is always non-negative and satisfies the normal-
ization constraint:
ş
ktpz, z1qdµpz1q “ 1.
Monte Carlo approximation.
In light of the former background review of continuous-time
Markov chain, we can see that the evaluation of linear operator L with any test function f P C8pMq
can be approximated in an unbiased manner as
L rfspzq “ lim
tÑ0
1
t
´ ż
pfpz1q ´ fpzqqktpz, z1qµpdz1q
¯
,
(107)
when t is close to 0. Finally, suppose that the state space S has a bounded measure µpSq “
ş
zPS µpdzq ă 8, and the observed data tz1
iuiPr1,Ns are uniformly sampled from the state space
S, then we can arrive at the below conclusion:
L rfspzq “ lim
tÑ0
1
t
´ ż
pfpz1q ´ fpzqqktpz, z1qµpSq
1
µpSqµpdz1q
¯
“ µpSq lim
tÑ0
1
t
´ ż
pfpz1q ´ fpzqqktpz, z1qUSpz1qµpdz1q
¯
“ µpSq lim
tÑ0
1
t Ez1„USpz1q
”
pfpz1q ´ fpzqqktpz, z1q
ı
« µpSq
tN
´
ÿ
1ďiďN
pfpz1
iq ´ fpzqqktpz, z1
iq
¯
,
(108)
where number t in the final part is close to 0, and term US denotes the density of a uniform distri-
bution over space S. The uniform assumption also makes sense since this Markov chain tZtutě0 is
just manually designed (i.e., not a stochastic process realized in the real world).
E.7
PROOF: ESTIMATIONS OF THE DIMENSION AND NORMAL VECTORS
For an arbitrary point q “ µi
obj residing in the embedded Riemannian manifold M ãÑ E, we have
proved two key conclusions in Appendix E.4.3:
Conclusion-1: eigenvectors as normal vectors. Any normal vector n P NqM is an eigenvector of
the kernel matrix Ki, with a zero eigenvalue λd “ 0;
Conclusion-2: eigenvectors as tangent vectors. Any tangent vector u P TqM is an eigenvector of
the same matrix, with a unit eigenvalue λd “ 1.
We can infer that the eigenvectors tui
d | pui
d, λdq, λd “ 1ud in the second case form an orthonormal
basis of the tangent space TqM. Therefore, the number of positive eigenvalues is equal to the
manifold dimension DM. In a similar sense, the eigenvectors tui
d | pui
d, λdq, λd “ 0ud in the first
scenario constitute a basis for the normal space NqM.
E.8
PROOF: CURVED-BASED FORMULATION OF THE SHAPE OPERATOR
The shape operator si is a geometric quantity defined at every point q “ µi
obj on an embedded
manifold M ãÑ E. The operator si has many forms, and we adopt its definition (Lee, 2018) as
sipn, u1, u2q “ ´gE
qp∇E
u1N |q, u2q,
(109)
27

<!-- page 28 -->
Paper preprint
where N is a normal section (which extends the normal vector n at point q) and p¨q |q means to
take the value of expression p¨q at point q. In extrinsic differential geometry, the shape operator si
is closely related to the concept of height function:
ηpq1q “ gE
qpn, q1 ´ qq “ nJpq1 ´ qq.
(110)
Now, Let us verify this point. Based on Eq. (59), the derivative of this point is as
gradM
q1 η “ pgradE
q1ηqπ “ nπq1,
(111)
where πq1 means the perpendicular projection to the tangent plane Tq1M. Given that n K TqM, it
is obvious that gradM
q η “ 0. Therefore, the point q is both a zero point (ηpqq “ nJ0 “ 0) and a
critical point to the height function η.
Before proceeding further, we need to first derive the Hessian tensor HessM of an arbitrary function
f P C8pMq. The formal definition of this tensor is as
HessMfpU1, U2q “ gMp∇M
U1gradMf, U2q, @U1, U2 P VpMq.
(112)
Given that the affine connection ∇M is induced by the Riemannian metric gM, the Hessian tensor
field can be expanded as follows:
HessMfpU1, U2q “ BM
U1gMpgradMf, U2q ´ gMpgradMf, ∇M
U1U2q
“ ∇M
U1∇M
U2f ´ gMpgradMf, ∇M
U1U2q “ BM
U1BM
U2f ´ BM
∇M
U1U2f.
(113)
Based on previous conclusions (i.e., Eq. (52) and Eq. (59)) on the Embedded manifold M, the
Riemannian Hessian HessM can be related to the Euclidean Hessian HessE as
HessMfpU1, U2q “ BM
U1BM
U2f ´ gEpgradMf, ∇M
U1U2q
“ BE
U1BE
U2f ´ xgradEf ´ pgradEfqK, ∇M
U1U2yE
“ BE
U1BE
U2f ´ xgradEf, ∇E
U1U2 ´ p∇E
U1U2qKyE
“ BE
U1BE
U2f ´ BE
∇E
U1U2f ` BE
p∇E
U1U2qKf “ HessEfpU1, U2q ` BE
p∇E
U1U2qKf,
(114)
where K indicates the orthogonal projection to the normal bundle NM “ \qPMNqM. For the
metric-compatible connection ∇M, we can also show that the Riemannian Hessian HessM is sym-
metric, based on the torsion-free condition:
HessMfpU1, U2q “ BM
U1BM
U2f ´BM
∇M
U1U2f “ BM
U2BM
U1f ´BM
∇M
U2U1f “ HessMfpU2, U1q. (115)
The central equality holds as we can apply the torsion-free condition to function f.
Importantly, we show that the Hessian tensor HessMf is associated with the shape operator si.
Based on Eq. (52) and Eq. (59), we have
HessMfpU1, U2q “ gEpp∇E
U1gradMfqπ, U2q
“ gEp∇E
U1gradMf, U2q ´ gEpp∇E
U1gradMfqK, U2q “ gEp∇E
U1gradMf, U2q
“ gEp∇E
U1nΠ, U2q “ gEp∇E
U1pN1 ´ gEpN1, NqNq, U2q,
(116)
where Π represents the element-wise projection to the tangent bundle T M, and N1 denotes a con-
stant vector field valued as n at every point. By simplifying the last term, we get
HessMfpU1, U2q “ gEp∇E
U1N1, U2q ´ gEp∇E
U1pgEpN1, NqNq, U2q
“ 0 ´ gEp∇E
U1pgEpN1, NqqN ` gEpN1, Nq∇E
U1N, U2q “ ´gEpN1, NqgEp∇E
U1N, U2q.
(117)
The above equality is tensorized, and its form at point q P M is as
HessM
q fpu1, u2q “ ´gE
qpn1, nqgE
qp∇E
u1N, u2q “ ´gE
qp∇E
u1N |q, u2q,
(118)
which exactly matches the previous definition of the shape operator si.
28

<!-- page 29 -->
Paper preprint
As the last stage, we aim to simplify the below expression:
sipn, vi
d1, vi
d2q “ HessM
q ηpvi
d1, vi
d2q “ gMp∇M
gradMϕd1 gradMη, gradMϕd2q |q .
(119)
Based on the explicit expression (i.e., Eq. (23)) of affine connection ∇M, we can get
sipn, vi
d1, vi
d2q “ 1
2
´
BM
gradMϕd1 gMpgradMη, gradMϕd2q
` BM
gradMηgMpgradMϕd2, gradMϕd1q ´ BM
gradMϕd2 gMpgradMϕd1, gradMηq
` gMprgradMϕd1, gradMηs, gradMϕd2q ´ gMprgradMη, gradMϕd2s, gradMϕd1q
` gMprgradMϕd2, gradMϕd1s, gradMη
¯
|q .
(120)
Inside the brackets of the right hand side, the last three terms are similar in form. For the forth term,
we can expand it as follows:
gMprgradMϕd1, gradMηs, gradMϕd2q “ BM
rgradMϕd1,gradMηsϕd2 “
“ BM
gradMϕd1 BM
gradMηϕd2 ´ BM
gradMηBM
gradMϕd1 ϕd2
“ BM
gradMϕd1 gMpgradMη, gradMϕd2q ´ BM
gradMηgMpgradMϕd1, gradMϕd2q.
(121)
Applying the same derivation to the last two terms, we have
sipn, vi
d1, vi
d2q “ 1
2
´
BM
gradMϕd1 gMp¨q ` BM
gradMηgMp¨q ´ BM
gradMϕd2 gMp¨q
` BM
gradMϕd1 gMpgradMη, gradMϕd2q ´ BM
gradMηgMpgradMϕd1, gradMϕd2q
´ BM
gradMηgMpgradMϕd2, gradMϕd1q ` BM
gradMϕd2 gMpgradMη, gradMϕd1q
` BM
gradMϕd2 gMpgradMϕd1, gradMηq ´ BM
gradMϕd1 gMpgradMϕd2, gradMηq
¯
|q .
(122)
By deleting the same terms, we can get
sipn, vi
d1, vi
d2q “ 1
2
´
BM
gradMϕd1 gMpgradMη, gradMϕd2q
` BM
gradMϕd2 gMpgradMη, gradMϕd1q ´ BM
gradMηgMpgradMϕd1, gradMϕd2
¯
|q .
(123)
For the first term on the right hand side, we can reshape it as
BM
gradMϕd1 gMpgradMη, gradMϕd2q “ gMpgradMϕd1, gradMgMpgradMη, gradMϕd2qq
“ 1
2gMpgradMϕd1, gradMLbnz∆Mrη, ϕd2sqq
“ 1
4Lbnz∆Mrϕd1, Lbnz∆Mrη, ϕd2sqq :“ 1
4Arϕd1, η, ϕd2s,
(124)
where the last equality defines a new operator A for notational convenience. Likewise, we can
reshape the last two terms of shape operator si and finally express it as
sipn, vi
d1, vi
d2q “ 1
8
´
Arϕd1, η, ϕd2spqq ` Arϕd2, η, ϕd1spqq ´ Arη, ϕd1, ϕd2spqq
¯
,
(125)
which proves the main claim of this proposition.
E.9
PROOF: CURVATURE ESTIMATION
The shape operator si at an arbitrary point q “ µi
obj P M is linear in its tangent vector inputs,
given a fixed normal vector n P NqM. In this regard, the operator is commonly represented as a
matrix Mi in extrinsic differential geometry (Lee, 2018). It is a known fact that the eigenvalues and
eigenvectors of matrix Mi respectively correspond to the principle curvature and directions.
29

<!-- page 30 -->
Paper preprint
For the matrix Si P RDMˆDM, it can be represented as
Si “ pUiqJMiUi,
(126)
Here Ui is an orthonormal matrix as tui
du1ďdďDM is an orthonormal basis in the tangent space
TqM. Importantly, this means pUiqJUi “ UipUiqJ “ I. Suppose that pτ, wq is a pair of eigen-
value and eigenvector for matrix Mi as Miw “ τw, then we can get
SippUiqJwq “ pUiqJMippUipUiqJqw “ pUiqJMiw “ τpUiqJw.
(127)
Therefore, the pair pτ, pUiqJwq is the eigenvalue and eigenvector of matrix Si. In light of this
derivation, we can infer that the eigen-decomposition of matrix Si corresponds to the principle
curvatures and directions.
F
EXPERIMENT SETTINGS AND MORE RESULTS
In this section, we provide the details of our experiment settings, and show additional results that
confirm the effectiveness of our framework: GeoSplat.
F.1
EXPERIMENT SETTINGS
Benchmark datasets.
We compare our method with the baselines on two groups of publicly avail-
able datasets: Replica (Straub et al., 2019) and ICL (Handa et al., 2014), which respectively consist
of 8 and 4 3D scenes. The Replica datasets are collected from commonly seen living room and
offices, and the ICL datasets provide similar environments. A key baseline (Li et al., 2025) also
adopted these datasets for model evaluation, so we follow their train and test splits.
Baselines.
The vanilla Gaussian splatting (i.e., 3DGS) (Kerbl et al., 2023) is a natural baseline in
our setting for verifying the performance gains of our framework. In addition to this, an important
baseline is GeoGaussian (Li et al., 2025), which adopted a number of regularization strategies (e.g.,
co-planar constraints) based on low-order geometric priors (e.g., normal information) and covered
most of other works (Wang et al., 2024; Turkulainen et al., 2025) on geometric regularization. It
is necessary to show that our framework, which is compatible with their low-order regularization
and further exploits higher-order geometric information, can achieve better performance. We also
compare with Ververas et al. (2025) to show that the mean absolute curvature (MAC) is indeed a
better low-curvature area identifier than the mean curvature.
There are other baselines included for diversity. For example, Gaussian-Splatting SLAM (Matsuki
et al., 2024) is an end-to-end method that uses Gaussians as the map for incremental localization,
reconstruction, and rendering tasks, while Vox-Fusion (Yang et al., 2022) represents a hybrid method
that integrates the implicit neural representation (Sitzmann et al., 2020; Mildenhall et al., 2021) into
traditional volumetric rendering. For LightGS (Fan et al., 2024), it further refines the vanilla 3DGS
by pruning insignificant Gaussian primitives and fine-tuning further. To make fair comparisons, we
copy available results from Li et al. (2025), following the same experiment setup.
Evaluation metrics.
For quantitatively measuring the quality of novel view rendering, we follow
the standard evaluation metrics used in many previous works (Straub et al., 2019; Kerbl et al., 2023;
Li et al., 2025), including Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure
(SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). PSNR evaluates on a color-wise
basis, while SSIM measures the similarity between two images in terms of structural information,
luminance, and distortion. For LPIPS, it compares two images using their features extracted by a
pre-trained neural network, like VGG-Net (Simonyan & Zisserman, 2014).
Hyper-parameter and method configurations.
For the tiny and large thresholds ξmin, ξmax, we
respectively specify them as 0.001 and ξmean ` 3ξstd, where ξmean denotes the average of all es-
timated curvatures in the data and ξstd represents their standard deviation. The number of nearest
neighbors for primitive upsampling is set as 10. The position vector µi
obj is warmed up through
structure from motion (SfM) (Schonberger & Frahm, 2016). For the tangential projection operation
J, we implement it as vJ “ xui
1, vyui
1 `xui
2, vyui
2 for any vector v P E, which is vK “ xrui
3, vyrui
3
30

<!-- page 31 -->
Paper preprint
(a) Ground Truth (Case 1).
(b) 3DGS (Case 1).
(c) GeoGaussian (Case 1). (d) Our GeoSplat (Case 1).
(e) Ground Truth (Case 2).
(f) 3DGS (Case 2).
(g) GeoGaussian (Case 2). (h) Our GeoSplat (Case 2).
Figure 4: Ground-truth and rendered images on the low-resource ICL Room-2 dataset. The cases 1,
2 are respectively generated by our varifold-based and manifold-based models.
(a) Ground Truth (Case 1).
(b) 3DGS (Case 1).
(c) GeoGaussian (Case 1). (d) Our GeoSplat (Case 1).
(e) Ground Truth (Case 2).
(f) 3DGS (Case 2).
(g) GeoGaussian (Case 2). (h) Our GeoSplat (Case 2).
Figure 5: Ground-truth and rendered images on the low-resource Replica R1 dataset. The cases 1, 2
are respectively generated by our varifold-based and manifold-based models.
for projection K to the normal direction. The final loss function for optimization is the original
one (e.g., D-SSIM) plus our regularization terms Lscale, Lrot. We run our models on 3 NVIDIA
Tesla V100 GPU devices, with the performance improvements over the baselines are statistically
significant with p ă 0.05 under t-test.
F.2
MORE CASE STUDIES
We present additional case studies that show our models can render images that contain much fewer
artifacts than baselines. The results are provided in Fig. 4, Fig 5, and Fig. 6.
F.3
ENRICHED GAUSSIAN PRIMITIVES
To show how our curvature-guided primitive upsampling strategy works, we run it on Replica OFF2
and show the enriched primitive cloud in Fig. 7.
31

<!-- page 32 -->
Paper preprint
(a) Ground Truth (Case 1).
(b) 3DGS (Case 1).
(c) GeoGaussian (Case 1). (d) Our GeoSplat (Case 1).
(e) Ground Truth (Case 2).
(f) 3DGS (Case 2).
(g) GeoGaussian (Case 2). (h) Our GeoSplat (Case 2).
Figure 6: Ground-truth and rendered images on the low-resource Replica R2 dataset. The cases 1, 2
are respectively generated by our varifold-based and manifold-based models.
Figure 7: The initial Gaussian primitives of Replica OFF2 are enriched through our curvature-guided
primitive upsampling strategy. The points in red are newly added.
32
