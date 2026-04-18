<!-- page 1 -->
Extreme Views: 3DGS Filter for Novel View
Synthesis from Out-of-Distribution Camera Poses
Damian Bowness(B)
and Charalambos Poullis
Concordia University, Montréal, Québec, Canada
damian.bowness@gmail.com, charalambos@poullis.org
Abstract. When viewing a 3D Gaussian Splatting (3DGS) model from
camera positions significantly outside the training data distribution, sub-
stantial visual noise commonly occurs. These artifacts result from the
lack of training data in these extrapolated regions, leading to uncertain
density, color, and geometry predictions from the model.
To address this issue, we propose a novel real-time render-aware filtering
method. Our approach leverages sensitivity scores derived from inter-
mediate gradients, explicitly targeting instabilities caused by anisotropic
orientations rather than isotropic variance. This filtering method directly
addresses the core issue of generative uncertainty, allowing 3D recon-
struction systems to maintain high visual fidelity even when users freely
navigate outside the original training viewpoints.
Experimental evaluation demonstrates that our method substantially
improves visual quality, realism, and consistency compared to existing
Neural Radiance Field (NeRF)-based approaches such as BayesRays.
Critically, our filter seamlessly integrates into existing 3DGS rendering
pipelines in real-time, unlike methods that require extensive post-hoc
retraining or fine-tuning.
Code and results at https://damian-bowness.github.io/EV3DGS
Keywords: 3D Reconstruction · Rendering Techniques · Image-Based
Modeling.
1
Introduction
High-fidelity 3D scene reconstruction and novel view synthesis represent funda-
mental challenges in computer vision and graphics, with applications spanning
virtual reality, autonomous navigation, and digital content creation. The qual-
ity of rendered novel views directly depends on the completeness and accuracy
of the underlying 3D scene representation, making robust reconstruction from
limited viewpoints a critical research priority.
Recent advances in neural 3D reconstruction, particularly Neural Radiance
Fields (NeRF) and 3D Gaussian Splatting (3DGS), demonstrate remarkable ca-
pabilities for photorealistic novel view synthesis. 3DGS offers particular advan-
arXiv:2510.20027v1  [cs.CV]  22 Oct 2025

<!-- page 2 -->
2
D. Bowness et al.
tages for real-time applications due to its explicit 3D primitive representation
and efficient rasterization-based rendering pipeline.
However, a fundamental limitation emerges when these reconstructed 3D
models are rendered from camera viewpoints significantly outside the original
training distribution. The 3D reconstruction process lacks sufficient multi-view
constraints to accurately estimate scene geometry and appearance in extrapo-
lated regions. This manifests as rendering artifacts including floating primitives,
inconsistent geometry, and view-dependent noise that severely compromise vi-
sual quality.
Fig. 1: Novel view synthesis from an
OOD camera pose(red) compared to
the training cameras poses (blue).
During
training,
3DGS
optimizes
Gaussian primitive parameters using pho-
tometric supervision from available cam-
era views. However, regions visible from
extreme out-of-distribution (OOD) view-
points (see Fig. 1) often lack adequate
multi-view coverage, leading to poorly-
constrained optimization. This results in
Gaussian primitives with uncertain 3D
properties that produce artifacts when
rendered from novel camera poses.
Existing solutions attempt to address
these issues through improved training
strategies, additional regularization, or
uncertainty quantification during recon-
struction. However, such approaches typ-
ically require expensive retraining pro-
cedures or fundamental modifications to
the 3D reconstruction pipeline, making
them impractical for deployment scenar-
ios where pre-trained models must accommodate viewing trajectories that differ
significantly from the original data collection conditions.
We propose a fundamentally different approach: Rather than modifying the
3D reconstruction process itself, we introduce a real-time filtering method that
operates during rendering to identify and suppress unstable 3D primitives. Our
key insight is that rendering artifacts from out-of-distribution views primarily
result from anisotropic Gaussian primitives with elongated shapes or poor ori-
entations that reflect insufficient multi-view constraints during training.
Our method computes gradient-based sensitivity measures that capture how
rendered pixel colors respond to perturbations in 3D space. By analyzing these
gradients in a rotation-aligned coordinate system, we can identify Gaussian prim-
itives whose 3D properties lead to view-dependent instabilities. This enables tar-
geted removal of problematic primitives during rendering to improve novel view
synthesis quality.
The resulting system seamlessly integrates into existing 3DGS rendering
pipelines, requiring no modifications to the underlying 3D reconstruction method

<!-- page 3 -->
EV3DGS
3
or additional training data. This enables robust rendering from extreme view-
points while preserving real-time performance. Our technical contributions in-
clude:
– A novel gradient-based sensitivity analysis specifically designed to identify
unstable 3D Gaussian primitives resulting from incomplete multi-view re-
construction
– A rotation-aligned filtering approach that targets anisotropic reconstruction
uncertainties without requiring 3D model retraining
– Comprehensive validation demonstrating significant improvements in novel
view synthesis quality from OOD camera poses
2
Related Work
2.1
Neural Radiance Fields
Neural rendering techniques, particularly Neural Radiance Fields (NeRF) [9],
have significantly advanced novel-view synthesis by implicitly representing 3D
scenes as continuous volumetric fields encoded via neural networks. NeRF opti-
mizes a multi-layer perceptron (MLP) by sampling along camera rays to predict
color and opacity values, enabling photorealistic image rendering from unseen
viewpoints. However, this implicit representation poses challenges such as high
computational cost and slow rendering times, limiting their real-time application
potential.
To address uncertainties in NeRF reconstructions, Goli et al. [3] proposed
BayesRays, a Bayesian framework using Laplace approximation to quantify sys-
tematic uncertainties. However, BayesRays requires a post-hoc training proce-
dure to build uncertainty fields, making it impractical for deployment scenarios
where models must be used immediately after initial reconstruction without ad-
ditional optimization phases.
2.2
3D Gaussian Splatting
Recent efforts shifted toward explicit volumetric representations such as Plenox-
els [1] and 3DGS [7], addressing computational inefficiencies and enabling real-
time rendering. Specifically, 3DGS uses a set of explicit Gaussian primitives to
represent radiance fields efficiently.
Building on explicit representations, several works have explored uncertainty
quantification and rendering improvements for 3DGS. Jiang et al. [6] gener-
alize uncertainty quantification using Fisher information-based approximation
across training images, targeting optimal view selection during training but not
addressing rendering-time instabilities or anisotropic uncertainties. Hanson et
al. [4] integrate uncertainty attributes into 3DGS for pruning redundant prim-
itives, though their approach primarily targets spatial redundancy rather than
artifacts from directional instabilities during novel view rendering.

<!-- page 4 -->
4
D. Bowness et al.
Methods exploiting ray-Gaussian intersections have emerged to improve ren-
dering accuracy and facilitate geometry extraction. Keselman and Hebert [8] de-
velop a differentiable ray-based renderer that integrates algebraic surfaces with
Gaussian mixture models (GMMs). Their work provides analytic solutions for
computing intersections between rays and Gaussian primitives. Extending these
concepts, Gao et al. [2] adapt ray-Gaussian intersection solutions for 3DGS to
improve lighting effects and rendering. Yu et al. [15] construct volumetric opacity
fields from ray-Gaussian intersections for high-quality mesh extraction.
Despite significant advances, existing uncertainty quantification methods pri-
marily target isotropic noise, redundant primitives, or implicit representations.
There is a notable gap in efficiently addressing anisotropic instabilities and di-
rectional sensitivity inherent in explicit representations such as 3DGS. Current
approaches either incur substantial computational overhead, require retraining,
or inadequately address directional instabilities that significantly degrade ren-
dering quality and temporal coherence.
3
Background
3DGS is a hybrid volumetric rendering technique that models radiance fields
using a discrete set of explicit Gaussian volumes. By representing radiance fields
as a set of 3D Gaussians, 3DGS defines the probability distribution of the ra-
diance of a point in space. Each 3D Gaussian primitive, G, is independently
parameterized by its density parameters and spatial parameters. The density
parameters are color, c, and opacity, α. The spatial parameters are the mean, µ,
and covariance, Σ.
G(x) = e−1
2 (x−µ)T Σ−1(x−µ)
(1)
To ensure the covariance matrix remains positive semi-definite throughout
optimization the 3D Gaussians are represented as ellipsoids. Therefore the Gaus-
sian mean is represented by the ellipsoid’s center while the covariance matrix is
decomposed into a scaling matrix, S, and rotation matrix, R,
Σ = RSST RT
(2)
For rendering, 3D Gaussians are depth-sorted and projected onto a 2D image
plane. The color, C, of pixel, x, is computed by alpha compositing:
C(x) =
N
X
i=1
ciαiGi(xi)
 i−1
Y
j=1
 1 −αjGj(xj)

(3)
The parameters of each 3D Gaussian are optimized from back propagating
the loss between the rendered image and ground-truth image.

<!-- page 5 -->
EV3DGS
5
4
Methodology
We introduce a render-time gradient sensitivity analysis for 3D Gaussian Splat-
ting. For every Gaussian that a camera ray intersects, we find the depth along
the ray where that Gaussian contributes most to the pixel. We then assign a
sensitivity score indicating how much the pixel color would change under tiny
spatial nudges of position, orientation, or scale. Low scores mark stable and
trustworthy contributions; high scores flag regions likely to produce artifacts. To
better detect directional instabilities, the score is evaluated in a rotation-aligned
coordinate system that emphasizes orientation effects without over-penalizing
fine detail.
For a given view, we use a two-pass approach to select which Gaussians
are used to render the view (see Fig. 2). In the first pass, each intersection is
marked accepted or rejected based on the sensitivity threshold, and for every
Gaussian we count both rejected and total intersections. In the second pass, we
compute the rejection fraction for each Gaussian; if it exceeds a user-set limit,
that Gaussian is removed from the rendering process for the current viewpoint.
This targeted, view-conditioned filtering preserves stable, detail-carrying Gaus-
sian primitives while suppressing unstable ones, reducing flicker and floaters and
improving visual quality and geometric consistency under extreme OOD camera
poses.
Fig. 2: Pipeline of our two-pass filter

<!-- page 6 -->
6
D. Bowness et al.
4.1
Ray-Marching
To analyze the radiance field with fine spatial resolution, we adopt a ray-marching
approach. Unlike screen-space projection methods that evaluate Gaussians in 2D
after rasterization, ray-marching enables direct computation of the ray-Gaussian
interaction in 3D space, leading to more precise control over rendering dynam-
ics. This method is particularly suited for computing pointwise sensitivity, as it
allows us to localize the analysis to specific ray-Gaussian intersections.
Given a ray defined by its origin o ∈R3 and direction r ∈R3, any point x
on the ray is parameterized as:
x = o + tr
(4)
where t represents the distance along the ray.
To evaluate the contribution of a kth 3D Gaussian to the ray, we first trans-
form the ray into the Gaussian’s canonical coordinate system. This transforma-
tion normalizes spatial variation using the Gaussian’s scale Sk and orientation
Rk, and positions the ray relative to the Gaussian’s mean µk:
og = S−1
k Rk(o −µk)
(5)
rg = S−1
k Rkr
(6)
xg = og + trg
(7)
In this normalized space, the Gaussian contribution simplifies to a 1D uni-
variate Gaussian along the ray:
G1D(t) = e−1
2 xT
g xg
(8)
= e−1
2 (rT
g rgt2+2oT
g rgt+oT
g og)
(9)
This quadratic form of the exponent provides analytical tractability and nu-
merical stability, facilitating efficient determination of the maximum Gaussian
contribution without needing to solve higher-order equations.
The peak contribution occurs at the depth tmin, where the exponent reaches
its minimum:
tmin = −oT
g rg
rTg rg
(10)
After determining the depths for all Gaussians intersected by the ray, the final
pixel color is computed via depth-ordered alpha compositing across all Gaussians
intersected by the ray:
C(o, r) =
K
X
k=1
ckαkG1D
k (tk,min)
k−1
Y
j=1
(1 −αjG1D
j
(tj,min))
(11)

<!-- page 7 -->
EV3DGS
7
This formulation allows precise control over how Gaussians influence each
pixel, by ensuring accurate modeling of cumulative optical effects along viewing
rays, laying the groundwork for analyzing how sensitive each pixel is to local
changes in the 3D radiance field.
4.2
Color Gradient Sensitivity
To measure the sensitivity of the rendered color to spatial perturbations, we de-
rive the gradient of the composite color C(o, r) with respect to the 3D position
x. This involves computing the gradient of the alpha-blended color contribu-
tion from each Gaussian, taking into account how both direct and accumulated
transmittance change under spatial variation.
Starting from the definition, we express the gradient of the composite color
as:
∇C(x) =
K
X
k=1
ckak∇
k−1
Y
j=1
(1 −aj) +
k−1
Y
j=1
(1 −aj)∇Gk(xk)
(12)
where ai = αiGi(xi). The gradient of a single Gaussian term is given by:
∇xG(x) = e−1
2 xT Σ−1x(−1
2)
 2Σ−1x

= −G(x)Σ−1x
(13)
and the gradient of the accumulated transmittance product becomes:
∇x
Y
j
(1 −aj) =

Y
j
(1 −aj)

X
j
ajΣ−1
j
xj
1 −aj
(14)
Substituting these expressions into the original gradient equation yields:
∇C =
K
X
k=1
ckak
k−1
Y
j=1
(1 −aj)


k−1
X
j=1
ajΣ−1
j
xj
1 −aj
−Σ−1
k xk


(15)
This Jacobian-like quantity describes how sensitive the final color is to the
underlying spatial configuration. However, due to the involvement of matrix
inversions and eigenvalue computations, evaluating this fully is computationally
expensive and impractical for real-time rendering.
To improve computational practicality, we decouple the gradient from the
color vector ck by replacing it with the scalar constant ck = 1. This removes
color-specific variation and focuses on transmittance dynamics. This form cap-
tures the structural sensitivity of the scene to spatial changes, identifying regions
of high rendering instability while avoiding unnecessary per-color calculations.

<!-- page 8 -->
8
D. Bowness et al.
4.3
Directional Sensitivity and Rotation-Based Gradient Filtering
While scalar gradient magnitudes effectively quantify overall spatial sensitivity,
they do not capture directional instabilities. That is, situations in which the
rendered output is disproportionately sensitive to perturbations along specific
directions or axes. To address this limitation, we extend our analysis by cal-
culating directional gradients within a rotation-aligned coordinate system. By
isolating the influence of Gaussian orientation from its scale, this method enables
a precise assessment of sensitivity relative to the Gaussian’s principal axes. Con-
sequently, we can independently evaluate directional instabilities without the
confounding effects introduced by anisotropic scaling.
The core intuition is that Gaussians with strong anisotropic properties (i.e.
those elongated along 1 or 2 of the 3 axes) exhibit direction-dependent instability.
For example, a long, thin Gaussian may be stable along its major axis but
highly sensitive to perturbations along its minor axes. Traditional geometric
analyses such as Principal Component Analysis (PCA) use eigenvalue ratios of
the covariance matrix to characterize such anisotropy, but these are not render-
aware and they do not directly measure the impact on rendered output.
Instead, by applying the gradient filter in a rotation-only space, we can ex-
pose directional rendering instabilities. This is achieved by transforming the
covariance matrix using only its rotation matrix Rk, excluding the scale Sk.
In this aligned space, we compute sensitivity gradients relative to each principal
axis of the Gaussian, revealing how rendering behavior changes with orientation.
Therefore, we drop the scale transformation in Equations (5)–(7) and derive our
sensitivity metric:
S =
K
X
k=1
ak
k−1
Y
j=1
(1 −aj)


k−1
X
j=1
ajxj
1 −aj
−xk


(16)
where x = Rk(o −µk) + tRkr, which represent the ray-Gaussian intersection
point in the rotation-aligned Gaussian space.
This approach offers several advantages:
– Isolates rotational sensitivity: By excluding scale, we ensure that the
gradient reflects only changes due to orientation, not magnitude.
– Highlights unstable orientations: High directional sensitivity indicates
that minor misalignments can significantly affect rendering, flagging Gaus-
sians prone to producing visual artifacts.
– Complementary to PCA filtering: While PCA identifies noise due to
isotropic variations, our method detects and corrects instabilities due to
anisotropic orientation.
4.4
Aggregate Sensitivity Analysis
We introduce a two-pass filtering pipeline to evaluate and reject unstable Gaus-
sians based on their aggregate sensitivity scores.

<!-- page 9 -->
EV3DGS
9
In the first pass, we compute gradient-based sensitivity at each ray-Gaussian
intersection, as defined in Equation 16. A Gaussian’s contribution to a pixel’s
color is conditionally accepted or rejected according to a user-defined sensitivity
threshold (τgrad). For each Gaussian, a ray intersection’s contribution is either
accepted or rejected. We track 2 counts: rejected and total, where total is the
sum of accepted and rejected counts. These counts serve as inputs for the second
pass.
In the second pass, we compute the aggregate sensitivity score for each Gaus-
sian as the ratio of its rejected count to its total usage count. The aggregate
sensitivity score for a Gaussian with a zero usage count is 1. Gaussians with a
rejection ratio exceeding a user-defined threshold (τratio) are excluded from the
final rendering.
Ultimately, this filtering mechanism enables robust scene reconstruction by
attenuating the influence of unstable Gaussians. By selectively removing Gaus-
sians with high aggregate sensitivity, we reduce noise and enhance the spatial
consistency of the rendered view. This targeted use of rotation-aligned gradient
analysis allows us to remove distractors and improve rendering quality.
5
Experiments and Results
Experimental setup. We evaluate a modified Nerfstudio Splatfacto pipeline
[14], replacing the default projection with ray marching and inserting our gradient-
sensitivity computation plus two-pass filter for extreme OOD views. We use fixed
thresholds: τgrad=0.0001 and τratio=0.5, except for the Doctor Johnson scene
from the Deep Blending dataset which uses τratio=0.01. Threshold values are
chosen heuristically to suppress extreme, instability-inducing sensitivities.
Splatfacto models use COLMAP initialization and 30k iterations. Following
BayesRays, we train a Nerfacto model for 30k iterations and extract uncertainty
fields over an additional 1k iterations. Experiments are run on an RTX 2080 Ti;
images are downscaled by 30–50% (≈1500×700 px) for runtime constraints.
Datasets & protocol. We test on Deep Blending and NeRF On-the-go [5,13].
For Deep Blending, we render extreme OOD views by extrapolating far beyond
the training poses. For NeRF On-the-go, we render along the original trajectories
and compare to ground truth to gauge suppression of transient artifacts. Per-
ceptual quality is measured with no-reference image quality (NR-IQA) metrics:
NIQE, BRISQUE, and PIQE.
– Natural Image Quality Evaluator (NIQE):
assesses image quality based on deviations from a learned natural scene
statistics model. [11]
– Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE):
uses statistical features from locally normalized luminance coefficients to
predict perceived distortion. [10]

<!-- page 10 -->
10
D. Bowness et al.
– Perception based Image Quality Evaluator (PIQE):
quantifies perceptual distortions by analyzing block-wise degradation in an
image, emphasizing spatially significant distortions. [12]
These metrics are designed to capture human-perceived image quality by pe-
nalizing unnatural textures, noise, and distortions in the absence of reference
images. Extreme OOD views can produce large white regions, which bias NR-
IQA; therefore all renderings are cropped prior to scoring. For all three metrics,
lower values are better.
Evaluation. For each scene we render an animation that moves from in-distribution
viewpoints to an OOD orbital path. We compute NIQE, BRISQUE, PIQE per
frame and report per-scene averages in Table 1. To our knowledge our method
is the first real-time filter for 3DGS. Therefore we compare to BayesRays [3], a
real-time NeRF filtering baseline using a pre-trained uncertainty field.
Our filter achieves the lowest (best) NIQE, BRISQUE, and PIQE across
all scenes (see table 1). These gains in perceptual quality across extreme OOD
camera positions indicate effective suppression of anisotropy-induced artifacts
without over-smoothing. Example frames (unfiltered vs. filtered) for each scene
are provided in Fig. 3.
Playroom
Creepy Attic
Dr. Johnson
NQ ↓
BR ↓
PQ ↓
NQ ↓
BR ↓
PQ ↓
NQ ↓
BR ↓
PQ ↓
BayesRays 0.1 11.74
44.45
83.84
10.30
45.26
77.42
10.02
46.84
70.38
BayesRays 0.2
9.18
47.11
74.47
10.17
55.12
69.64
9.71
57.88
59.58
BayesRays 0.5 10.67
64.24
58.31
14.02
53.57
46.87
11.34
55.91
59.76
Ours
3.41
41.38
52.99
5.94
42.30
40.24
4.49
37.37
51.60
Table 1: NIQE (NQ), BRISQUE (BR), and PIQE (PQ) scores for Playroom,
CreepyAttic, and DrJohnson.
Connection to information measures. On NeRF On-the-go scenes, our
rotation-only, per-intersection gradient behaves as a fast proxy for the Fisher
Information Matrix. Instead of the usual expectation over Jacobian products, a
decoupled single-sample gradient captures the dominant directional sensitivity
(epistemic uncertainty) at render time. Large gradient magnitudes flag sparsely
constrained or ambiguous regions; conditionally filtering the associated Gaus-
sians suppresses anisotropic instabilities and reduces visible artifacts (see Fig. 4).

<!-- page 11 -->
EV3DGS
11
Playroom
Creepy Attic
Unfiltered
Filtered
Unfiltered
Filtered
Dr. Johnson
Unfiltered
Filtered
Fig. 3: Deep Blending example frames.
Original
3DGS
Our Filter
arcdetriomphe
patio_high
spot
Fig. 4: NeRF-On-the-Go samples highlight our filter’s suppression of ambiguous
regions from in-distribution camera poses.

<!-- page 12 -->
12
D. Bowness et al.
6
Ablation
We analyze the contribution of individual filtering components through a series
of ablation studies.
Single-Pass. We first implement a single-pass approach in which Gaussians
are not fully removed; instead, only their contributions are selectively filtered
based on the intermediate gradient at the ray-Gaussian intersection point. As
shown in Fig. 5(a), the centers of many Gaussians remain visible in the render.
This occurs because the gradient magnitude is low near a Gaussian’s center.
From Equation 16, this behavior is expected: the xk term is near zero when
the ray intersects the Gaussian near its mean. For the first Gaussian, there are
no prior contributions along the ray, therefore the cumulative sensitivity re-
mains at its initialized value of zero. As a result, initial Gaussians in a depth
ordered list, which are often responsible for occluding distant geometry in ex-
treme viewpoints, are preserved when intersected near their centers because their
sensitivities are near zero.
Scale-Incorporated. Next, we examine the impact of incorporating scale into
the intermediate gradient calculation, as described in Section 4.3. This leads to
a loss of anisotropic shape information, effectively normalizing Gaussians into
unit spheres in their local coordinate space. Fig. 5(b) and (c) illustrate how
this transformation causes elongated, noisy Gaussians to persist, while small
Gaussians with scales less than 1 (which are critical for scene detail) are over-
filtered. This is due to the normalization process that increases the magnitude
of the gradient vector which makes them more likely to exceed the rejection
threshold.
(a)
(b)
(c)
Fig. 5:
(a) Single-pass render of Playroom scene. τgrad. = 10−5
(b) and (c) Loss of detail when including scale (b) vs. no scale (c)
Filter Parameters. The effect of the two parameters in isolation is shown in
Fig. 6. The subfigures (6.a and 6.b) hold τratio = 0.5 while varying τgrad., and

<!-- page 13 -->
EV3DGS
13
subfigures (c–d) hold τgrad. = 10−5 while varying τratio. Visual inspection sug-
gests that using only one parameter yields different perceptual quality, whereas
the best performance comes from a scene-specific combination of both tuned by
the user.
(a)
τratio
=
0.5,
τgrad. = 0.0001
(b)
τratio
=
0.5,
τgrad. = 0.0005
(c) τgrad.
=
10−5,
τratio = 0.25
(d) τgrad.
= 10−5,
τratio = 0.75
Fig. 6: Panels (a–b) hold τratio = 0.5 and sweep τgrad.; panels (c–d) hold τgrad. =
10−5 and sweep τratio.
7
Conclusion
We introduce a novel sensitivity measurement for 3DGS that identifies and filters
anisotropic instabilities during rendering, without requiring retraining or scene-
specific tuning. By analyzing intermediate gradient responses from the differen-
tiable rasterization pipeline, our method targets the core source of generative
uncertainty: Directional instability arising from anisotropic orientations. This
filtering mechanism enables robust rendering even when users navigate freely
beyond the original training views, a setting where standard 3DGS models often
produce severe visual artifacts. Experimental results across complex, photoreal-
istic datasets demonstrate consistent improvements in perceptual quality metrics
surpassing baseline 3DGS and NeRF-based methods such as BayesRays.
Acknowledgement
This research was undertaken, in part, based on support from the Natural Sci-
ences and Engineering Research Council of Canada Grants RGPIN-2021-03479
(NSERC DG) and ALLRP 571887 - 2021 (NSERC Alliance).
References
1. Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenox-
els: Radiance fields without neural networks. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 5501–5510 (2022)

<!-- page 14 -->
14
D. Bowness et al.
2. Gao, J., Gu, C., Lin, Y., Zhu, H., Cao, X., Zhang, L., Yao., Y.: Relightable 3d
gaussian: Real-time point cloud relighting with brdf decomposition and ray tracing
(2005)
3. Goli, L., Reading, C., Sell´an, S., Jacobson, A., Tagliasacchi, A.: Bayes’ rays: Un-
certainty quantification in neural radiance fields. In: Conference on Computer Vi-
sion and Pattern Recognition (CVPR) (2024)
4. Hanson, A., Tu, A., Singla, V., Jayawardhana, M., Zwicker, M., Goldstein, T.: Pup
3d-gs: Principled uncertainty pruning for 3d gaussian splatting. arXiv (2024)
5. Hedman, P., Philip, J., Price, T., Frahm, J.M., Drettakis, G., Brostow, G.: Deep
blending for free-viewpoint image-based rendering. ACM Transactions on Graphics
(Proc. SIGGRAPH Asia) 37(6), 257:1–257:15 (2018)
6. Jiang, W., Lei, B., , Daniilidis, K.: Fisherrf: Active view selection and uncertainty
quantification for radiance fields using fisher information (2023)
7. Kerbl, B., Kopanas, G., Leimk¨uhler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Transactions on Graphics (2023)
8. Keselman, L., Hebert, M.: Approximate differentiable rendering with algebraic
surfaces. In: European Conference on Computer Vision (ECCV) (2022)
9. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV
(2020)
10. Mittal, A., Moorthy, A.K., Bovik, A.C.: No-reference image quality assessment in
the spatial domain. IEEE Transactions on Image Processing 21(12), 4695–4708
(2012). https://doi.org/10.1109/TIP.2012.2214050
11. Mittal, A., Soundararajan, R., Bovik, A.C.: Making a “completely blind” image
quality analyzer. IEEE Signal Processing Letters 20(3), 209–212 (2013). https:
//doi.org/10.1109/LSP.2012.2227726
12. N, V., D, P., Bh, M.C., Channappayya, S.S., Medasani, S.S.: Blind image quality
evaluation using perception based features. In: 2015 Twenty First National Con-
ference on Communications (NCC). pp. 1–6 (2015). https://doi.org/10.1109/
NCC.2015.7084843
13. Ren, W., Zhu, Z., Sun, B., Chen, J., Pollefeys, M., Peng, S.: Nerf on-the-go: Ex-
ploiting uncertainty for distractor-free nerfs in the wild. In: IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR) (2024)
14. Tancik, M., Weber, E., Ng, E., Li, R., Yi, B., Kerr, J., Wang, T., Kristoffersen,
A., Austin, J., Salahi, K., Ahuja, A., McAllister, D., Kanazawa, A.: Nerfstudio: A
modular framework for neural radiance field development. In: ACM SIGGRAPH
2023 Conference Proceedings. SIGGRAPH ’23 (2023)
15. Yu, Z., Sattler, T., Geiger, A.: Gaussian opacity fields: Efficient high-quality com-
pact surface reconstruction in unbounded scenes. SIGGRAPH ASIA (2024)
