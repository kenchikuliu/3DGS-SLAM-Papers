<!-- page 1 -->
PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation
Samarth Chopra1, B,
Jing Liang2,
Gershom Seneviratne1,
Dinesh Manocha1
1University of Maryland, College Park
2Stanford University
sachopra@umd.edu,
jinglms@stanford.edu,
gershom@umd.edu,
dmanocha@umd.edu
Abstract
Understanding physical properties such as friction, stiff-
ness, hardness, and material composition is essential for
enabling robots to interact safely and effectively with their
surroundings. However, existing 3D reconstruction meth-
ods focus on geometry and appearance and cannot infer
these underlying physical properties. We present PhysGS, a
Bayesian-inferred extension of 3D Gaussian Splatting that
estimates dense, per-point physical properties from visual
cues and vision–language priors. We formulate property es-
timation as Bayesian inference over Gaussian splats, where
material and property beliefs are iteratively refined as new
observations arrive.
PhysGS also models aleatoric and
epistemic uncertainties, enabling uncertainty-aware object
and scene interpretation. Across object-scale (ABO-500),
indoor, and outdoor real-world datasets, PhysGS improves
accuracy of the mass estimation by up to 22.8%, reduces
Shore hardness error by up to 61.2%, and lowers kinetic
friction error by up to 18.1% compared to deterministic
baselines. Our results demonstrate that PhysGS unifies 3D
reconstruction, uncertainty modeling, and physical reason-
ing in a single, spatially continuous framework for dense
physical property estimation. Additional results are avail-
able at https://samchopra2003.github.io/physgs.
1. Introduction
Understanding the physical properties of real-world envi-
ronments is critical for enabling robots to interact safely and
effectively with their surroundings [5, 10, 36, 63]. This ca-
pability is essential across a wide range of domains such
as navigation [34, 56], manipulation [1, 31], and surgical
robotics [31, 39, 62], and is required in both complex in-
door [40, 60, 63] and outdoor [10, 34, 48] environments.
In particular, accurately estimating physical attributes such
as friction [10, 34], elasticity [56], hardness [5], and den-
sity [60] is essential for safe and robust robot interaction
with diverse and unstructured real-world scenarios.
In general, estimating physical properties from visual
Per-point Property Estimation
(e.g. friction, hardness, density, stiffness)
Uncertainty Field
Friction Field
Dense Property Prediction 
(e.g. mass)
Outdoor Environments
Indoor Objects
(x, y, z)
First 
observation
New 
observations
Updated 
physical properties
Figure 1. Overview of PhysGS. Top: Our method estimates per-
point and dense physical properties (e.g., friction, hardness, den-
sity, stiffness, and mass) by combining vision–language material
priors with Bayesian updates over 3D Gaussian splats. Bottom:
PhysGS can also be deployed in outdoor environments to infer
scene-level properties such as friction and predictive uncertainty;
we visualize the total uncertainty (aleatoric + epistemic).
sensors remains a challenging task due to two primary rea-
sons [10, 60].
First, visually similar but physically dis-
tinct regions (e.g., mud vs.
asphalt, or grass vs.
rock)
are often difficult to distinguish, leading to brittle planning
1
arXiv:2511.18570v1  [cs.CV]  23 Nov 2025

<!-- page 2 -->
and control in complex scenes [10, 11].
While conven-
tional 3D mapping approaches, such as occupancy grids [8],
signed distance fields [38], or implicit neural representa-
tions [26, 37], can provide rich geometric detail, they pri-
marily focus on recovering object shape, and typically fail
to encode physical properties or material categories. Sec-
ond, although recent work has made progress in estimat-
ing physical properties in indoor environments [40, 60, 63],
outdoor scenes remain underexplored. Existing methods of-
ten focus on one or two specific physical properties of out-
door objects or terrains, such as friction [5, 10, 34], pliabil-
ity [56], or stiffness [5], and are not easily generalizable to
a broader set of physical attributes.
Recent advances in 3D vision and semantic perception
have increasingly focused on linking visual appearance to
underlying physical attributes, enabling models to distin-
guish between rigid, deformable, slippery, and compliant
surfaces [5, 10, 11, 60, 63]. In parallel, vision-language
models (VLMs) have demonstrated the ability to capture la-
tent physical properties, such as friction and elasticity, with
multimodal inputs [63]. These models can qualitatively rea-
son about forces, materials, and object dynamics from lan-
guage and imagery [40, 49, 55, 59, 64], highlighting their
potential as semantic priors for physical inference.
However, a core challenge in estimating physical prop-
erties from visual sensors is managing the uncertainty in-
herent in both sensing and inference.
Visual and depth
observations are often degraded by sensor noise, lighting
variation, occlusion, and calibration drift [6, 52–54]. This
type of uncertainty, known as aleatoric uncertainty, cap-
tures measurement noise and perceptual ambiguity [3, 24].
Simultaneously, learning-based models trained on limited
or domain-specific datasets frequently struggle to general-
ize to novel textures, materials, and environmental condi-
tions [18, 24, 25]. This is referred to as epistemic uncer-
tainty, which reflects the model’s incomplete or imperfect
knowledge of the world, often due to insufficient or biased
training data [3, 24].
Main Contributions: We propose a 3D physical prop-
erty estimation framework that integrates Bayesian infer-
ence into the Gaussian Splatting optimization process, treat-
ing each Gaussian primitive as a probabilistic entity whose
properties are updated via posterior refinement. This en-
ables PhysGS to estimate both point-level physical proper-
ties (e.g., friction, hardness, stiffness, density) and object-
level quantities (e.g., total mass), while producing cali-
brated aleatoric and epistemic uncertainty. Our novel con-
tributions include:
1. Bayesian-Inferred Gaussian Splatting.
We embed
Bayesian updates within the Gaussian Splatting pipeline,
allowing each Gaussian’s physical property values to be
updated through confidence-weighted posterior refine-
ment from observations.
2. Unified multi-property estimation across scales.
A
single Bayesian formulation supports diverse physical
properties, including friction, hardness, stiffness, den-
sity, and mass, at both the point level and the object level,
enabling fine-grained property mapping and global ag-
gregation within the same framework.
3. Generality across environments and object types.
PhysGS is broadly applicable to a wide range of indoor
and outdoor scenes and operates on both rigid and de-
formable objects, including vegetation, soil, and every-
day household materials, enabling consistent physical
property estimation across heterogeneous real-world set-
tings.
Across all datasets, including ABO-500, and a real-
world friction–hardness dataset, PhysGS achieves strong
gains over prior methods such as NeRF2Physics, CLIP-
based recognition, and direct VLM regression.
We ob-
serve improvements of up to 61.2% in Shore hardness error,
18.1% in kinetic friction error, and 22.8% in mass-density
error.
2. Related Work
2.1. Visual Property Fields
A major line of work estimates physical properties by as-
sociating scene objects with open-vocabulary physical se-
mantics, querying where specific physical property appear
in observed spaces. LERF grounds CLIP embeddings [45]
within NeRF, distilling multi-scale language features into a
dense, queryable 3D field that produces 3D relevancy maps
for text prompts [27]. Closely related open-vocabulary 3D
mapping approaches propagate physical–semantic features
into 3D reconstructions for zero-shot recognition and re-
trieval [22, 43].
With the rise of 3D Gaussian Splatting (3DGS)[26],
several methods directly inject language or semantic fea-
tures into Gaussian primitives, yielding fast, explicit, and
queryable 3D fields. LangSplat [44] distills 2D CLIP fea-
tures into a 3D language field over Gaussians for open-
vocabulary search. Related efforts [16, 21] assign seman-
tic Gaussians for open-vocabulary 3D understanding, show-
ing that explicit Gaussian fields are well-suited for encoding
and rendering high-dimensional properties beyond color.
Based on visual–semantic cues, recent work has further
incorporated language-context features to enhance physical
property estimation [60, 63]. NeRF2Physics[63] constructs
a language-embedded 3D feature space and performs zero-
shot kernel regression to estimate per-point physical prop-
erties, while GaussianProperty[60] extends this idea to 3D
Gaussians.
2

<!-- page 3 -->
2.2. Uncertainty-Aware and Probabilistic Scene Un-
derstanding
In vision and robotics, uncertainty is typically decom-
posed into aleatoric (data or sensor noise) and epistemic
(model or knowledge) components. This formulation has
become standard for loss design, model calibration, and
risk-sensitive decision making in perception systems [3,
4, 24].
Prior work formalizes these notions for vision-
based tasks and demonstrates how to jointly learn uncer-
tainty with outputs such as depth or segmentation, or to
approximate Bayesian inference via dropout or ensembles,
thereby improving robustness and out-of-distribution be-
havior [14, 25, 32].
Estimating the physical properties of real-world environ-
ments from sensors inherently involves uncertainty. Several
approaches introduce probabilistic maps, tail-risk measures,
and confidence-aware policies to quantify and mitigate this
uncertainty [5, 9, 12, 13, 41]. STEP [12] models traversabil-
ity as a stochastic variable and plans using a CVaR-based
risk formulation, validated across diverse field environ-
ments and the DARPA SubT Challenge.
Evidential and
Bayesian formulations extend this concept by outputting
full distributions rather than point estimates, enabling on-
line belief updates from new observations [3, 4, 10, 11].
EVORA [3] learns evidential traction distributions, explic-
itly separating aleatoric and epistemic components to assess
motion risk, while Ewen et al. [10, 11] maintain joint beliefs
over semantics and continuous properties to predict physi-
cal property maps (e.g. friction).
Beyond 2D perception, uncertainty has been integrated
into NeRF and 3D Gaussian Splatting (3DGS) frameworks
to quantify ambiguity arising from sparse views, occlu-
sions, and under-constrained geometry [33, 47].
Recent
works estimate spatial uncertainty post hoc for pre-trained
NeRFs [15, 33], propose probabilistic NeRFs [19], or di-
rectly model uncertainty in 3DGS [28] via variational or ev-
idential objectives, including dynamic and 4D settings [29].
Complementary efforts have explored uncertainty-aware on
variational Gaussian splatting and SLAM pipelines that
propagate uncertainty in pose and structure, demonstrat-
ing that per-Gaussian uncertainty can substantially enhance
mapping robustness and downstream reasoning [20, 46, 47].
3. Proposed Approach
In this section, we outline our Bayesian framework for
dense physical property estimation.
We begin with
a Dirichlet–Categorical model for fusing confidence-
weighted material labels across views (Sec. 3.1, 3.2),
then extend it to continuous properties using a Normal–
Inverse–Gamma prior to obtain calibrated aleatoric and
epistemic uncertainty (Sec. 3.3).
We then describe how
these Bayesian updates integrate with 3D Gaussian Splat-
ting, segmentation, and VLM prompting to produce per-
point property fields and object-level estimates (Sec. 3.4).
3.1. Preliminaries
Dirichlet–Categorical formulation.
We model discrete
material labels produced by the VLM using a Categorical
distribution and place a Dirichlet prior over its parameters.
The Dirichlet distribution is the conjugate prior to the Cat-
egorical likelihood, enabling closed-form Bayesian updates
as new observations are incorporated across views.
The Categorical distribution parameterized by θ
∈
[0, 1]K represents the probability that an observation be-
longs to class i:
f(z = i | θ) = θi.
(1)
The Dirichlet distribution defines a continuous K-variate
prior over θ, parameterized by α ∈RK
>0, as
f(θ | α) = Γ(PK
k=1 αk)
QK
k=1 Γ(αk)
K
Y
k=1
θαk−1
k
,
(2)
where Γ(·) is the Gamma function.
Given a set of n observed material labels Z
=
{z1, . . . , zn} drawn from a Categorical distribution, the
posterior predictive probability that a new observation be-
longs to material class i is
f(z = i | Z, α) =
Z
θ
f(z = i | θ) f(θ | Z, α) dθ.
(3)
Using conjugacy of the Dirichlet prior and Categorical like-
lihood, the integral simplifies to the closed-form expression
f(z = i | Z, α) =
˜αi
PK
j=1 ˜αj
,
(4)
where the posterior parameters are recursively updated as
˜αi ←αi(0) +
X
m: cm=i
λ pm,
(5)
with λ controlling the evidence strength contributed by each
observation and pm denoting the confidence provided by the
VLM for the m-th prediction.
3.2. Bayesian Inference for Material Property Esti-
mation
We introduce our hierarchical Bayesian framework for
estimating
material-specific
physical
properties
from
confidence-weighted
observations.
Building
on
the
Dirichlet–Categorical model of [11], we extend it with
a continuous posterior to jointly infer material class and
properties such as friction, density, and hardness.
3

<!-- page 4 -->
Multi-View Image Dataset
SAM
VLM Prompting
3DGS Reconstruction
Physical Property Estimation
Per-point* density
*: Point refers to voxel of fixed volume
Voxel-wise multiplication
for dense property prediction
Full object mass
Bayesian Inference + Uncertainty Quantification
Materials
Properties
Confidences
Q. Provided a picture, give a brief 
caption of the part, material, density 
and confidence of prediction.
x k prompts/answers
k = # images in dataset 
A. Material 1: 'bed frame side, wood,
 700 kg/m^3, 85%'
Material 2: 'mattress top, foam, 
25.00 kg/m^3, 90%'
Final Property Estimates
Uncertainties per mask
Wood: 665.615 kg/m^3
Foam: 35.060 kg/m^3
Fabric: 150.000 kg/m^3
Plastic: 800.000 kg/m^3 
Aleatoric uncertainty
Epistemic uncertainty
35.1 kg/m^3
64.0 kg
X
Figure 2. PhysGS architecture. Given multi-view images, SAM provides part-level segmentations that are used for 3D Gaussian Splatting
(3DGS) reconstruction. For each segmented part, a VLM produces material labels, density estimates, and confidence scores across multiple
views. These observations are fused using Bayesian inference with uncertainty quantification to obtain final per-material property distribu-
tions. By propagating the estimated densities over the reconstructed 3D Gaussian field, PhysGS predicts per-point density and full-object
mass.
Continuous property estimation.
While the Dirichlet–
Categorical formulation governs the discrete class probabil-
ities, we also require an estimate of the continuous physical
property ψ associated with each material. For each mate-
rial class i, we maintain confidence-weighted accumulators
that enable incremental computation of the first and second
moments using a running mean and variance formulation
proposed by [57] and generalized by [42]:
Wi =
X
m
pm,
Si =
X
m
pm ψm,
Qi =
X
m
pm ψ2
m,
(6)
representing the total weight, first moment, and second mo-
ment, respectively. These accumulators allow efficient on-
line updates without requiring access to past observations,
which is particularly beneficial in streaming or on-the-fly
reconstruction settings.
The posterior mean and variance for material i are then
estimated as
µi = Si
Wi
,
σ2
i = max
 Qi
Wi
−µ2
i , ϵ

,
(7)
yielding a Gaussian posterior
p(ψi | Z) = N(µi, σ2
i ),
(8)
which represents the system’s belief over the continu-
ous physical property for material i given all confidence-
weighted evidence Z.
This formulation integrates natu-
rally with the Dirichlet update by providing confidence-
weighted, incremental, and uncertainty-aware refinement as
new observations become available.
Hierarchical posterior. The resulting model is hierarchical
in nature, jointly estimating the discrete material identity z
and continuous physical property ψ:
p(z, ψ | Z, α) = p(ψ | z, Z) p(z | Z, α),
(9)
where p(z | Z, α) is the Dirichlet–Categorical posterior and
p(ψ | z, Z) is the Gaussian posterior. Applying the Law of
Total Probability as in [11], the overall predictive distribu-
tion over physical properties is
f(ψ | Z, α) =
K
X
i=1
f(ψ | z = i) f(z = i | Z, α).
(10)
Substituting Eq. (8) into Eq. (10) gives a closed-form
multimodal Gaussian mixture for the predicted material
properties:
f(ψ | Z, α) =
K
X
i=1
˜αi
PK
j=1 ˜αj
N(µi, σ2
i ).
(11)
This mixture formulation expresses the full posterior as a
weighted sum of unimodal Gaussian components, where
4

<!-- page 5 -->
each mode corresponds to a material class and is weighted
by its recursively updated class likelihood from the Dirich-
let posterior.
3.3. Uncertainty-Aware Property Fields
Uncertainty modeling via the Normal–Inverse–Gamma
prior. For each material i, the joint prior over the mean µi
and variance σ2
i of the property ψ is given by
p(µi, σ2
i | τi, κi, αi, βi) = N

µi
 τi, σ2
i
κi

Inv-Gamma
 σ2
i
 αi, βi

,
(12)
where τi denotes the prior mean, κi controls the preci-
sion on µi (i.e., the strength of accumulated evidence), and
(αi, βi) are the shape and scale parameters governing the
uncertainty in σ2
i .
Predictive Uncertainty update.
Given a new observa-
tion ψm associated with material class i and its confidence
pm, the posterior parameters (˜τi, ˜κi, ˜αi, ˜βi) can be updated
in closed-form, allowing sequential fusion of confidence-
weighted evidence without storing past data. This conju-
gate formulation provides closed-form expressions for the
predictive mean and variance of the property ψi.
The total predictive uncertainty decomposes into two
components:
Var[ψi] = E[σ2
i ]
| {z }
aleatoric
+ Var[µi]
| {z }
epistemic
.
(13)
The first term represents aleatoric uncertainty, which arises
from inherent noise in the observations and variability
within each material class. The second term represents epis-
temic uncertainty, corresponding to uncertainty in the es-
timated mean that decreases as more high-confidence ev-
idence is incorporated. For implementation, we compute
these moments directly from the NIG parameters:
E[σ2
i ] =
˜βi
˜αi −1,
Var[µi] = E[σ2
i ]
˜κi
.
(14)
The resulting aleatoric, epistemic, and total predictive
uncertainties provide interpretable measures of confidence
in both the per-class property estimates and the overall re-
construction. Regions with high aleatoric uncertainty reflect
sensor or perceptual noise, while high epistemic uncertainty
indicates insufficient or conflicting evidence about the un-
derlying material properties.
3.4. Learning Semantics and Physical Properties
Semantic Segmentation.
Given a multi-view image
dataset, we employ SAM [30] to produce pixel-accurate
masks that decompose each object into hierarchical levels
(whole, part, and sub-part), facilitating fine-grained seman-
tic understanding. The model outputs multiple candidate
masks at different granularities, which we refine by discard-
ing redundant or low-confidence predictions using SAM’s
built-in IoU and stability measures. The resulting segmen-
tation maps capture precise object boundaries and seman-
tically coherent regions, forming the basis for our down-
stream physical property estimation.
VLM Prompting. For each segmented image, we construct
a vision–language prompt comprising a triplet of images
arranged side by side, following the design of [60]. The
left image presents the complete object, the middle image
overlays the segmentation mask, and the right image iso-
lates the masked region of interest. Given an input image
I, this process yields k visual prompts corresponding to the
k masks predicted by SAM. We additionally condition the
VLM with a structured textual query that instructs it to (i)
provide a concise caption of the segmented part, (ii) iden-
tify its predominant material, and (iii) infer relevant phys-
ical properties such as friction, density, etc. The model is
further asked to report a normalized confidence score within
[0, 1], representing its belief in the prediction.
3D Gaussian Splatting. Given the VLM responses and the
refined physical property estimates from our Bayesian in-
ference scheme, we construct a material legend assigning
each material a unique color. The corresponding scene im-
ages are recolored accordingly and used as semantic inputs
for 3DGS reconstruction. This yields a semantic splat that
supports dense property inference, such as mass estimation.
Physical Property Estimation. Using the reconstructed
3D Gaussian field and inferred material properties, we
perform per-point and dense physical property estimation.
Each voxel is associated with a predicted property value
(e.g., friction or density), enabling spatial queries for per-
point properties or integration over the volume to obtain ag-
gregate measures such as total mass.
4. Experiments and Results
4.1. Implementation Details
We employ the splatfacto-big variant of Nerfstu-
dio [51] for 3D Gaussian Splatting, using default parame-
ters except for a random scale of 2.0 and a random back-
ground color. Each scene is trained for 20,000 iterations on
an NVIDIA RTX A5000 GPU.
For image segmentation, we use SAM [30] to obtain
whole, part, and sub-part material decompositions. Mate-
rial property estimation is performed using GPT-5 as the
vision–language model (VLM), conditioned on structured
visual–text prompts derived from the segmented images.
5

<!-- page 6 -->
Ours 
(Mass Density)
Ours 
(Material Segmentation)
Input RGB Image
NeRF2Physics
(Material Segmentation)
NeRF2Physics
(Mass Density)
Figure 3. Qualitative comparison on the ABO-500 dataset. For each object, we show the input RGB image, material segmentation
and mass-density predictions from NeRF2Physics, and the corresponding results from our method. PhysGS produces cleaner material
segmentation with fewer artifacts compared to NeRF2Physics and more consistent part boundaries, and yields sharper, more plausible
mass-density fields across diverse object categories.
4.2. Mass Estimation
Dataset. We employ the ABO dataset [7] for evaluating
mass prediction, which includes a large set of consumer
products listed on Amazon along with multi-view imagery,
segmentation masks, physical measurements, and metadata.
Specifically, we make use of the representative multi-view
benchmark ABO-500 curated by [63], which selects a bal-
anced subset of 500 items from the entire ABO dataset. It
is divided into 300 training, 100 validation, and 100 testing
instances.
Metrics.
Following prior work on visual mass estima-
tion [50], we evaluate using four complementary metrics
that measure both absolute and relative error between the
predicted mass ˆm and the ground-truth mass m:
• Absolute Difference Error (ADE): |m −ˆm|,
• Absolute Log Difference Error (ALDE): | ln m −ln ˆm|,
• Absolute Percentage Error (APE):
 m−ˆm
m
,
• Minimum Ratio Error (MnRE): min
  m
ˆm, ˆm
m

.
Baselines. We compare our system against several visual
and multimodal baselines on the ABO-500 dataset:
• Image2mass [50]: a CNN that infers mass directly from
RGB images and 3D bounding box dimensions.
• 2D CNN: a lightweight regression model built upon a
frozen ResNet50 [17] backbone, fine-tuned with addi-
tional layers for scalar mass prediction.
• LLaVA [35]: a vision-language model designed for in-
struction following.
• NeRF2Physics [63]: a NeRF-based approach that jointly
estimates 3D geometry and per-point physical properties
such as density, friction, and stiffness. It predicts mass
by integrating predicted density across the reconstructed
volume.
Qualitative Results.
Figure 3 presents qualitative re-
sults on the ABO-500 dataset. For each object, we com-
pare the material segmentation and mass-density predic-
tions from NeRF2Physics with those produced by PhysGS.
Our method yields substantially cleaner material segmen-
tations with fewer spurious labels and more coherent part
boundaries, while also producing sharper and more sta-
ble mass-density fields. These improvements are consis-
tent across a wide range of object categories, demonstrat-
ing the advantage of combining vision-language priors with
Bayesian inference over 3D Gaussian splats.
Quantitative Results.
We evaluate the accuracy of our
method on mass estimation using the ABO-500 test set
(100 objects). As shown in Table 1, traditional 2D meth-
ods such as Image2mass [50] and 2D CNNs exhibit high
mass estimation error due to their inability to capture 3D
structure or material composition. VLM approaches (e.g.
6

<!-- page 7 -->
Table 1. Mass estimation on ABO-500 test set (100 objects). ADE
is measured in kilograms. Bold: best model.
Method
ADE (↓) ALDE (↓) APE (↓) MnRE (↑)
Image2mass [50]
12.496
1.792
0.976
0.341
2D CNN
15.431
1.609
14.459
0.362
LLaVA [35]
17.328
1.893
1.837
0.306
NeRF2Physics [63]
8.730
0.771
1.061
0.552
Ours
8.254
0.999
0.819
0.474
Table 2.
Ablation study for mass estimation on ABO-500 val
set (100 objects). ADE is measured in kilograms. BI refers to
Bayesian Inference. Bold: best model.
Method
ADE (↓) ALDE (↓) APE (↓) MnRE (↑)
NeRF2Physics [63]
9.786
0.61
0.931
0.609
Ours (w/o BI)
9.728
0.770
0.717
0.561
Ours (with BI)
9.187
0.827
0.715
0.539
LLaVA [35]) show similar limitations, producing noisy pre-
dictions that vary across views. NeRF2Physics improves
accuracy by exploiting neural radiance fields, and achieves
the best ALDE (0.771) and MnRE (0.552) among exist-
ing baselines. PhysGS achieves the best performance on
two key metrics: it reduces ADE from 8.730 to 8.254,
corresponding to a 5.5% improvement, and reduces APE
from 1.061 to 0.819, a substantial 22.8% improvement over
NeRF2Physics.
Ablation Study. Table 2 shows that incorporating Bayesian
inference yields clear gains over both NeRF2Physics and
our non-Bayesian variant.
Updating material and prop-
erty beliefs across additional views reduces ADE by 5.6%
and improves APE by 6.4% compared to the version with-
out Bayesian updates.
These improvements demonstrate
that aggregating multi-view evidence to refine the posterior
distribution leads to more accurate mass estimation, con-
firming the benefit of treating physical properties as latent
variables that are iteratively updated rather than fixed from
single-view predictions.
4.3. Friction and Hardness Estimation
Dataset. To evaluate our model’s ability to infer dense, spa-
tially varying physical properties within objects, we lever-
age the friction and hardness dataset containing paired im-
age and real-world measurement data, curated by [63]. This
dataset includes 15 household objects captured across 13
scenes, using multi-view RGB images paired with per-point
measurements of kinetic friction coefficient and Shore A/D
hardness.
Metrics. We report the same evaluation metrics for per-
point friction and hardness estimation used in evaluation for
mass estimation as above: ADE, ALDE, APE, MnRE.
Ours 
(Shore Hardness)
Ours 
(Friction Coefficient)
Input RGB Image
Figure 4. Qualitative results for the friction and hardness dataset.
Given a single RGB view, PhysGS predicts dense friction coef-
ficients and Shore hardness values for a variety of household ob-
jects. The resulting property fields are spatially smooth, physically
plausible, and consistent across diverse materials and geometries.
Baselines. As before, we compare our method to several
visual and multimodal baselines.
• GPT-4V [61]: A large vision–language model capable of
processing masked regions in its prompt.
• CLIP [45]: A vision–language baseline that uses global
CLIP embeddings from the canonical view of the scene,
rather than the fused multi-view patch features used in
our method. This baseline evaluates how well static vi-
sual–semantic representations can generalize to physical
property prediction.
• NeRF2Physics [63]: Same as in mass estimation base-
line.
Qualitative Results. Figure 4 presents qualitative results
for friction and Shore hardness estimation on real objects.
From a single RGB view, PhysGS produces dense per-point
friction and hardness fields across materials such as rubber,
leather, plastic, and metal. The predictions capture fine-
grained material differences and exhibit clean boundaries
between regions with distinct friction and hardness char-
acteristics, reflecting the system’s ability to localize subtle
variations in contact and deformation properties.
Quantitative Results. Table 3 reports results for per-point
Shore A/D hardness and kinetic friction estimation on our
real-world dataset. Across all hardness metrics, PhysGS
achieves substantial gains over existing approaches. Rel-
ative to the best baseline, our method reduces ADE by
61.2%, lowers ALDE by 34.4%, and decreases APE by
16.5%, while improving MnRE by 8.4%.
For kinetic
friction, PhysGS again outperforms the strongest baseline
7

<!-- page 8 -->
Table 3. Estimation of per-point Shore hardness (left) and kinetic friction coefficient (right) on the real-world dataset. Bold: best model.
Shore Hardness (31 points, 11 objects)
Kinetic Friction (6 points, 6 objects)
Method
ADE (↓) ALDE (↓) APE (↓) MnRE (↑)
Method
ADE (↓) ALDE (↓) APE (↓) MnRE (↑)
GPT-4V [61]
32.752
0.330
0.304
0.758
GPT-4V [61]
0.209
0.430
0.549
0.692
CLIP [45]
32.857
0.294
0.266
0.774
CLIP [45]
0.222
0.455
0.602
0.654
NeRF2Physics [63]
34.295
0.315
0.276
0.765
NeRF2Physics [63]
0.155
0.321
0.360
0.736
Ours
12.721
0.193
0.222
0.839
Ours
0.131
0.263
0.365
0.805
Ours 
(Friction Coefficient)
Ours 
(Young's Modulus GPa)
Ours 
(Total Uncertainty)
Ours 
(Material Segmentation)
Input RGB Image
Figure 5. Outdoor scene results on real environments. From a single RGB view, PhysGS predicts material segmentation, friction coef-
ficients, Young’s modulus, and total uncertainty (aleatoric + epistemic). The method captures broad material variations across natural
terrain and vegetation while producing pixel-wise physical property estimates with associated confidence. Higher total uncertainty in
Rows 2 and 3 corresponds to scenes with dense clutter and visually ambiguous regions, where SAM provides less precise part-level masks
(e.g., separating leaf litter from wood or mud from grass). Rows 1 and 4 exhibit lower uncertainty due to clearer material boundaries and
more uniform regions.
(NeRF2Physics) on the majority of metrics, reducing ADE
by 15.5% and ALDE by 18.1%, and increasing MnRE by
9.4%. These gains highlight the benefit of integrating multi-
view evidence through Bayesian inference, which refines
the posterior distribution of material properties beyond what
single-view or deterministic models can infer.
4.4. Applications: Outdoor Scene Understanding
PhysGS can also estimate physical properties of outdoor en-
vironments, such as friction and stiffness, which are impor-
tant for reasoning about natural, vegetation-rich terrain [5].
Our method also provides per-pixel uncertainty (aleatoric +
epistemic) for these estimates. We demonstrate this capabil-
ity on the RUGD [58] and RELLIS-3D [23] datasets (Fig-
ure 3), which contain challenging outdoor scenes where ac-
curate physical property prediction and uncertainty assess-
ment are essential for interpreting complex terrain.
5. Conclusion, Limitations and Future Work
We presented PhysGS, a Bayesian-inferred 3D Gaussian
Splatting framework for estimating dense physical proper-
ties from RGB images and vision-language priors. Across
indoor and outdoor real-world datasets, PhysGS achieves
consistent gains over existing approaches. On ABO-500,
our method improves mass estimation by 5.5% in ADE and
22.8% in APE. PhysGS also reduces Shore hardness error
by up to 61.2% and kinetic friction error by up to 18.1%
relative to the strongest baselines. Outdoor experiments on
RUGD and RELLIS-3D further show that the method gen-
eralizes to complex natural environments, capturing mate-
rial segmentation, friction, stiffness, and uncertainty.
8

<!-- page 9 -->
A primary limitation of PhysGS lies in its sensitivity to
segmentation quality. When part-level masks merge visu-
ally similar materials or fail to isolate fine-grained regions,
the downstream physical property estimates inherit this am-
biguity, reducing material separation and increasing pre-
dictive uncertainty. This effect is visible in Fig. 5, where
cluttered outdoor regions lead to less precise SAM masks
and correspondingly higher total uncertainty. Future work
may incorporate VLM-guided segmentation refinement or
confidence-based mask filtering to automatically reject low-
quality masks and preserve fine-grained material structure.
References
[1] Aude Billard and Danica Kragic. Trends and challenges in
robot manipulation. Science, 364(6446):eaat8414, 2019. 1
[2] Katherine L Bouman, Bei Xiao, Peter Battaglia, and
William T Freeman. Estimating the material properties of
fabric from video. In Proceedings of the IEEE international
conference on computer vision, pages 1984–1991, 2013. 2
[3] Xiaoyi Cai, Siddharth Ancha, Lakshay Sharma, Philip R Os-
teen, Bernadette Bucher, Stephen Phillips, Jiuguang Wang,
Michael Everett, Nicholas Roy, and Jonathan P How. Evora:
Deep evidential traversability learning for risk-aware off-
road autonomy. IEEE Transactions on Robotics, 2024. 2,
3
[4] Xiaoyi Cai, James Queeney, Tong Xu, Aniket Datar, Chenhui
Pan, Max Miller, Ashton Flather, Philip R Osteen, Nicholas
Roy, Xuesu Xiao, et al. Pietra: Physics-informed eviden-
tial learning for traversing out-of-distribution terrain. IEEE
Robotics and Automation Letters, 2025. 3
[5] Jiaqi Chen, Jonas Frey, Ruyi Zhou, Takahiro Miki, Georg
Martius, and Marco Hutter. Identifying terrain physical pa-
rameters from vision-towards physical-parameter-aware lo-
comotion and navigation. IEEE Robotics and Automation
Letters, 2024. 1, 2, 3, 8
[6] Samarth Chopra, Fernando Cladera, Varun Murali, and Vi-
jay Kumar.
Agrinerf:
Neural radiance fields for agri-
culture in challenging lighting conditions.
arXiv preprint
arXiv:2409.15487, 2024. 2
[7] Jasmine Collins, Shubham Goel, Kenan Deng, Achlesh-
war Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas
F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al.
Abo: Dataset and benchmarks for real-world 3d object un-
derstanding.
In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 21126–
21136, 2022. 6
[8] Alberto Elfes. Using occupancy grids for mobile robot per-
ception and navigation. Computer, 22(6):46–57, 2002. 2
[9] Gian Erni, Jonas Frey, Takahiro Miki, Matias Mattamala,
and Marco Hutter. Mem: Multi-modal elevation mapping
for robotics and learning. In 2023 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS), pages
11011–11018. IEEE, 2023. 3
[10] Parker Ewen, Adam Li, Yuxin Chen, Steven Hong, and Ram
Vasudevan. These maps are made for walking: Real-time
terrain property estimation for mobile robots. IEEE Robotics
and Automation Letters, 7(3):7083–7090, 2022. 1, 2, 3
[11] Parker Ewen, Hao Chen, Yuzhen Chen, Anran Li, Anup
Bagali, Gitesh Gunjal, and Ram Vasudevan.
You’ve
got to feel it to believe it:
Multi-modal bayesian infer-
ence for semantic and property prediction. arXiv preprint
arXiv:2402.05872, 2024. 2, 3, 4
[12] David D Fan, Kyohei Otsu, Yuki Kubo, Anushri Dixit, Joel
Burdick, and Ali-Akbar Agha-Mohammadi. Step: Stochas-
tic traversability evaluation and planning for risk-aware off-
road navigation. arXiv preprint arXiv:2103.02828, 2021. 3
[13] Jonas Frey, Manthan Patel, Deegan Atha, Julian Nubert,
David Fan, Ali Agha, Curtis Padgett, Patrick Spieler,
Marco Hutter, and Shehryar Khattak. Roadrunner-learning
traversability estimation for autonomous off-road driving.
IEEE Transactions on Field Robotics, 2024. 3
[14] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian
approximation:
Representing model uncertainty in deep
learning. In international conference on machine learning,
pages 1050–1059. PMLR, 2016. 3
[15] Lily Goli, Cody Reading, Silvia Sell´an, Alec Jacobson,
and Andrea Tagliasacchi. Bayes’ rays: Uncertainty quan-
tification for neural radiance fields.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20061–20070, 2024. 3
[16] Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, and Qing Li.
Semantic gaussians: Open-vocabulary scene understanding
with 3d gaussian splatting. arXiv preprint arXiv:2403.15624,
2024. 2
[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 6
[18] Dan Hendrycks and Thomas Dietterich. Benchmarking neu-
ral network robustness to common corruptions and perturba-
tions. arXiv preprint arXiv:1903.12261, 2019. 2
[19] Matthew D Hoffman, Tuan Anh Le, Pavel Sountsov, Christo-
pher Suter, Ben Lee, Vikash K Mansinghka, and Rif A
Saurous.
Probnerf:
Uncertainty-aware inference of 3d
shapes from 2d images.
In International Conference on
Artificial Intelligence and Statistics, pages 10425–10444.
PMLR, 2023. 3
[20] Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li,
Liangjing Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng
Cui.
Cg-slam: Efficient dense rgb-d slam in a consistent
uncertainty-aware 3d gaussian field. In European Confer-
ence on Computer Vision, pages 93–112. Springer, 2024. 3
[21] Xu Hu, Yuxi Wang, Lue Fan, Junsong Fan, Junran Peng,
Zhen Lei, Qing Li, and Zhaoxiang Zhang. Semantic any-
thing in 3d gaussians. CoRR, 2024. 2
[22] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala,
Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang
Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, et al.
Conceptfusion: Open-set multimodal 3d mapping.
arXiv
preprint arXiv:2302.07241, 2023. 2
[23] Peng Jiang, Philip Osteen, Maggie Wigness, and Srikanth
Saripalli. Rellis-3d dataset: Data, benchmarks and analy-
9

<!-- page 10 -->
sis. In 2021 IEEE international conference on robotics and
automation (ICRA), pages 1110–1116. IEEE, 2021. 8
[24] Alex Kendall and Yarin Gal. What uncertainties do we need
in bayesian deep learning for computer vision? Advances in
neural information processing systems, 30, 2017. 2, 3
[25] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task
learning using uncertainty to weigh losses for scene geome-
try and semantics. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 7482–7491,
2018. 2, 3
[26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2
[27] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik. Lerf: Language embedded
radiance fields. In Proceedings of the IEEE/CVF interna-
tional conference on computer vision, pages 19729–19739,
2023. 2
[28] Junyoung Kim, Minsik Jeon, Jihong Min, Kiho Kwak, and
Junwon Seo. E2-bki: Evidential ellipsoidal bayesian kernel
inference for uncertainty-aware gaussian semantic mapping.
arXiv preprint arXiv:2509.11964, 2025. 3
[29] Mijeong Kim, Jongwoo Lim, and Bohyung Han. 4d gaus-
sian splatting in the wild with uncertainty-aware regulariza-
tion. Advances in Neural Information Processing Systems,
37:129209–129226, 2024. 3
[30] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. In Proceedings of the IEEE/CVF international confer-
ence on computer vision, pages 4015–4026, 2023. 5
[31] Yo Kobayashi, Jun Okamoto, and Masakatsu G Fujie. Phys-
ical properties of the liver and the development of an intel-
ligent manipulator for needle insertion. In Proceedings of
the 2005 IEEE international conference on robotics and au-
tomation, pages 1632–1639. IEEE, 2005. 1
[32] Balaji Lakshminarayanan, Alexander Pritzel, and Charles
Blundell. Simple and scalable predictive uncertainty estima-
tion using deep ensembles. Advances in neural information
processing systems, 30, 2017. 3
[33] Sibaek Lee, Kyeongsu Kang, Seongbo Ha, and Hyeonwoo
Yu.
Bayesian nerf: Quantifying uncertainty with volume
density for neural implicit fields. IEEE Robotics and Au-
tomation Letters, 2025. 3
[34] Jing Liang, Kasun Weerakoon, Tianrui Guan, Nare Kara-
petyan, and Dinesh Manocha. Adaptiveon: Adaptive out-
door local navigation method for stable and reliable actions.
IEEE Robotics and Automation Letters, 8(2):648–655, 2022.
1, 2
[35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning. Advances in neural information
processing systems, 36:34892–34916, 2023. 6, 7
[36] Haozhe Lou, Mingtong Zhang, Haoran Geng, Hanyang
Zhou, Sicheng He, Zhiyuan Gao, Siheng Zhao, Jiageng Mao,
Pieter Abbeel, Jitendra Malik, et al. Dream: Differentiable
real-to-sim-to-real engine for learning robotic manipulation.
In 3rd RSS Workshop on Dexterous Manipulation: Learning
and Control with Diverse Data. 1
[37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[38] Helen Oleynikova, Alexander Millane, Zachary Taylor, En-
ric Galceran, Juan Nieto, and Roland Siegwart.
Signed
distance fields:
A natural representation for both map-
ping and planning. In RSS 2016 workshop: geometry and
beyond-representations, physics, and scene understanding
for robotics. University of Michigan, 2016. 2
[39] Sadao Omata, Yoshinobu Murayama, and Christos E Con-
stantinou. Real time robotic tactile sensor system for the de-
termination of the physical properties of biomaterials. Sen-
sors and Actuators A: Physical, 112(2-3):278–285, 2004. 1
[40] Changmin Park, Beomjoon Lee, Haechan Jung, Haejin Jung,
and Changjoo Nam.
Understanding physical properties
of unseen deformable objects by leveraging large language
models and robot actions. arXiv preprint arXiv:2506.03760,
2025. 1, 2
[41] Manthan Patel, Jonas Frey, Deegan Atha, Patrick Spieler,
Marco Hutter, and Shehryar Khattak.
Roadrunner m&m-
learning multi-range multi-resolution traversability maps for
autonomous off-road navigation.
IEEE Robotics and Au-
tomation Letters, 2024. 3
[42] Philippe Pierre Pebay. Formulas for robust, one-pass par-
allel computation of covariances and arbitrary-order statisti-
cal moments. Technical report, Sandia National Laboratories
(SNL), Albuquerque, NM, and Livermore, CA . . . , 2008. 4
[43] Songyou
Peng,
Kyle
Genova,
Chiyu
Jiang,
Andrea
Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al.
Openscene: 3d scene understanding with open vocabularies.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 815–824, 2023. 2
[44] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
Hanspeter Pfister. Langsplat: 3d language gaussian splatting.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 20051–20060, 2024.
2
[45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PmLR, 2021. 2, 7, 8
[46] Erik Sandstr¨om, Ganlin Zhang, Keisuke Tateno, Michael
Oechsle, Michael Niemeyer, Youmin Zhang, Manthan Pa-
tel, Luc Van Gool, Martin Oswald, and Federico Tombari.
Splat-slam: Globally optimized rgb-only slam with 3d gaus-
sians. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 1680–1691, 2025. 3
[47] Luca Savant, Diego Valsesia, and Enrico Magli.
Mod-
eling uncertainty for gaussian splatting.
arXiv preprint
arXiv:2403.18476, 2024. 3
[48] Gershom Seneviratne, Kasun Weerakoon, Mohamed Elnoor,
Vignesh Rajgopal, Harshavarthan Varatharajan, Mohamed
10

<!-- page 11 -->
Khalid M Jaffar, Jason Pusey, and Dinesh Manocha. Cross-
gait: Cross-attention-based multimodal representation fusion
for parametric gait adaptation in complex terrains.
arXiv
preprint arXiv:2409.17262, 2024. 1
[49] Yinghao Shuai, Ran Yu, Yuantao Chen, Zijian Jiang, Xi-
aowei Song, Nan Wang, Jv Zheng, Jianzhu Ma, Meng
Yang, Zhicheng Wang, et al.
Pugs:
Zero-shot physi-
cal understanding with gaussian splatting.
arXiv preprint
arXiv:2502.12231, 2025. 2
[50] Trevor Standley, Ozan Sener, Dawn Chen, and Silvio
Savarese.
image2mass: Estimating the mass of an object
from its image.
In Conference on Robot Learning, pages
324–333. PMLR, 2017. 6, 7
[51] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li,
Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake
Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A
modular framework for neural radiance field development.
In ACM SIGGRAPH 2023 conference proceedings, pages 1–
12, 2023. 5
[52] Sebastian Thrun. Probabilistic algorithms in robotics. Ai
Magazine, 21(4):93–93, 2000. 2
[53] Sebastian Thrun. Probabilistic robotics. Communications of
the ACM, 45(3):52–57, 2002.
[54] Jonathan Tremblay, Aayush Prakash, David Acuna, Mark
Brophy, Varun Jampani, Cem Anil, Thang To, Eric Camer-
acci, Shaad Boochoon, and Stan Birchfield. Training deep
networks with synthetic data: Bridging the reality gap by
domain randomization. In Proceedings of the IEEE confer-
ence on computer vision and pattern recognition workshops,
pages 969–977, 2018. 2
[55] Yi Wang, Jiafei Duan, Dieter Fox, and Siddhartha Srinivasa.
Newton: Are large language models capable of physical rea-
soning?
In Findings of the association for computational
linguistics: EMNLP 2023, pages 9743–9758, 2023. 2
[56] Kasun Weerakoon,
Adarsh Jagan Sathyamoorthy,
Jing
Liang, Tianrui Guan, Utsav Patel, and Dinesh Manocha.
Graspe: Graph based multimodal fusion for robot navigation
in outdoor environments.
IEEE Robotics and Automation
Letters, 8(12):8090–8097, 2023. 1, 2
[57] DHD West. Updating mean and variance estimates: An im-
proved method. Communications of the ACM, 22(9):532–
535, 1979. 4
[58] Maggie Wigness, Sungmin Eum, John G Rogers, David Han,
and Heesung Kwon. A rugd dataset for autonomous naviga-
tion and visual perception in unstructured outdoor environ-
ments. In 2019 IEEE/RSJ International Conference on Intel-
ligent Robots and Systems (IROS), pages 5000–5007. IEEE,
2019. 8
[59] William Xie, Maria Valentini, Jensen Lavering, and Nikolaus
Correll. Deligrasp: Inferring object properties with llms for
adaptive grasp policies. arXiv preprint arXiv:2403.07832,
2024. 2
[60] Xinli Xu, Wenhang Ge, Dicong Qiu, ZhiFei Chen, Dongyu
Yan, Zhuoyun Liu, Haoyu Zhao, Hanfeng Zhao, Shunsi
Zhang, Junwei Liang, et al. Gaussianproperty: Integrating
physical properties to 3d gaussians with lmms. In Proceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision, pages 7231–7240, 2025. 1, 2, 5, 4
[61] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang,
Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn
of lmms: Preliminary explorations with gpt-4v (ision). arXiv
preprint arXiv:2309.17421, 2023. 7, 8
[62] Aiko Yoshizawa, Jun Okamoto, Hiroshi Yamakawa, and
Masakatsu G Fujie.
Robot surgery based on the physical
properties of the brain-physical brain model for planning and
navigation of a surgical robot. In Proceedings of the 2005
IEEE International Conference on Robotics and Automation,
pages 904–911. IEEE, 2005. 1
[63] Albert J Zhai, Yuan Shen, Emily Y Chen, Gloria X Wang,
Xinlei Wang, Sheng Wang, Kaiyu Guan, and Shenlong
Wang.
Physical property understanding from language-
embedded feature fields. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 28296–28305, 2024. 1, 2, 6, 7, 8
[64] Yizhen Zheng, Huan Yee Koh, Jiaxin Ju, Anh TN Nguyen,
Lauren T May, Geoffrey I Webb, and Shirui Pan. Large lan-
guage models for scientific discovery in molecular property
prediction. Nature Machine Intelligence, pages 1–11, 2025.
2
11

<!-- page 12 -->
PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation
Supplementary Material
A. Full Bayesian and Uncertainty Formulation
A.1. Observation Model
For completeness, we describe the observation model used
in PhysGS. Each observation corresponds to a segmented
region of the scene and contains semantic (material) and
physical (property) information extracted from the vision–
language model (VLM). The role of these observations in
the Bayesian updates is described in Sec. 3.
Observation tuple. For the m-th segmented region, we de-
fine
Om =
 cm, pm, ψm

,
(15)
where cm is the predicted material class, pm is the confi-
dence produced by the VLM, and ψm is the predicted phys-
ical property (e.g., friction, hardness, stiffness, density).
Across multiple views, the full set of observations is
Z = {O1, . . . , OM}.
(16)
The tuple (cm, pm, ψm) constitutes a noisy measure-
ment of the latent variables (zm, µzm, σ2
zm).
In particu-
lar, the predicted class cm serves as a noisy proxy for the
true (unobserved) material label zm, while the VLM es-
timate ψm provides a noisy observation of the underly-
ing material-specific physical property whose distribution
is governed by (µzm, σ2
zm). These observed quantities sup-
ply the confidence-weighted evidence used in the Bayesian
updates that follow.
A.2. Dirichlet–Categorical Posterior
The
material
fusion
process
follows
the
Dirichlet–
Categorical formulation introduced in Sec. 3.1.
The
Categorical likelihood and Dirichlet prior correspond to
Eqs. (1)–(2).
The posterior Dirichlet parameters update as in Eq. (5):
˜αi = αi(0) +
X
m: cm=i
λ pm.
(17)
The resulting posterior predictive distribution over mate-
rial classes is given in Eq. (4).
A.3. Continuous Property Estimation
Continuous physical properties are fused using confidence-
weighted running moments as introduced in Sec. 3.2 of the
main paper. The accumulators Wi, Si, and Qi match Eq. (6)
in the main paper.
The posterior mean and variance follow Eq. (7) in the
main paper:
µi = Si
Wi
,
σ2
i = max
 Qi
Wi
−µ2
i , ϵ

.
(18)
This defines the Gaussian posterior p(ψi | Z) shown in
Eq. (8) of the main paper.
A.4. Mixture Formulation
Marginalizing over discrete material classes using the hier-
archical model in Sec. 3.2 leads directly to the mixture dis-
tribution shown in Eq. (11), combining material probabili-
ties with the class-conditional Gaussian property estimates.
A.5. Normal–Inverse–Gamma Posterior
We extend our continuous estimator with the Normal–
Inverse–Gamma (NIG) prior introduced in Sec. 3.3. The
joint prior over (µi, σ2
i ) matches Eq. (12).
Given a weighted observation (ψm, pm), the closed-form
posterior updates (Eqs. (13)–(14)) are:
˜κi = κi + pm,
(19)
˜τi = κiτi + pmψm
κi + pm
,
(20)
˜αi = αi + pm
2 ,
(21)
˜βi = βi + pmκi(ψm −τi)2
2(κi + pm)
.
(22)
A.6. Predictive Uncertainty
The decomposition of predictive uncertainty into aleatoric
and epistemic components follows Eq. (13). The predictive
moments correspond directly to Eq. (14):
E[σ2
i ] =
˜βi
˜αi −1,
Var[µi] = E[σ2
i ]
˜κi
.
(23)
Aleatoric uncertainty reflects inherent variability within
a material class, while epistemic uncertainty captures uncer-
tainty in the estimated mean due to limited or inconsistent
evidence.
A.7. MMSE Estimate
As shown in Eq. (7), the posterior mean µi is the minimum
mean-square-error (MMSE) estimator:
ˆψi = µi.
(24)
This corresponds to the property value that minimizes
expected squared error and is therefore used as the single
representative estimate for each material class.
1

<!-- page 13 -->
Table 4. Stiffness estimation on MIT Fabric Properties dataset (30
objects). ADE is measured in lbf-in2. Bold: best model.
Method ADE (↓) ALDE (↓) APE (↓) MnRE (↑)
GPT-4V
0.563
2.380
19.986
0.210
GPT-5
0.126
1.053
2.887
0.452
Ours
0.040
0.725
1.338
0.553
A.8. Full Probabilistic Model
The complete hierarchical model underlying PhysGS is
summarized in Sec. 3 of the main paper and depicted in
Fig. 2. For completeness, we restate the probabilistic struc-
ture:
θ ∼Dirichlet(α(0)),
(25)
zm ∼Categorical(θ),
(26)
(µi, σ2
i ) ∼NIG(τi, κi, αi, βi),
(27)
ψm ∼N(µzm, σ2
zm).
(28)
This formulation provides the full probabilistic backbone
through which PhysGS jointly infers materials, continuous
properties, and calibrated uncertainty.
First, a Dirichlet
prior is placed over the material probabilities θ, reflect-
ing initial uncertainty about the frequency of each mate-
rial class. Each segmented region then draws a material
label zm from this Categorical distribution. For every ma-
terial class i, the mean and variance of its physical prop-
erty are modeled using a Normal–Inverse–Gamma (NIG)
prior, which captures both uncertainty in the material’s typ-
ical property value and its intrinsic variability. Finally, the
observed physical property ψm for region m is sampled
from the Gaussian distribution associated with its material
label. Together, this hierarchy defines how materials and
their continuous properties jointly generate the observations
used in the Bayesian inference procedure.
B. Additional Results
B.1. Stiffness Estimation
Dataset. We employ the MIT Fabric Properties Dataset [2]
for evaluating mass prediction, 30 different types of real
fabric along with measurements of their material proper-
ties. Since these are all videos, we curate an image dataset
from this, where all the different fabrics are evaluated for
their bending stiffness. While these are video datasets, they
are captured from a single view, and thus we evaluate our
model on one image per fabric. We pick the first frame of
every video.
Metrics. We report the same evaluation metrics for bend-
ing stiffness estimation (lbf-in2) used in evaluation for mass
Input RGB Image
Corduroy
Nylon Rip Stop
Pleather
Outdoor Polyester
Ours
(Bending Stiffness) Input RGB Image
Ours
(Bending Stiffness)
Figure 6.
Bending stiffness estimation on real fabric samples
from MIT Fabric Properties Dataset. Given an input RGB im-
age, PhysGS produces dense stiffness fields that capture material
differences across corduroy, nylon ripstop, outdoor polyester, and
pleather.
estimation as above: ADE, ALDE, APE, MnRE.
Baselines. We compare our model against several visual
and multimodal baselines on the ABO-500 dataset:
• GPT-4V: We provide GPT-4V with the image, and ask it
to estimate the physical stiffness of the fabric.
• GPT-5: Same prompt as GPT-4V.
Quantitative Results.
Table 4 reports quantitative re-
sults comparing our method against GPT-4V and GPT-5
VLM baselines. Across all metrics, PhysGS achieves the
strongest performance, reducing ADE by 68.3% compared
to GPT-5 and by more than an order of magnitude compared
to GPT-4V. Our method also attains the highest MnRE
score, indicating substantially improved scale consistency
in stiffness estimation. These gains highlight the effective-
ness of our Bayesian fusion framework in capturing fine-
grained material compliance even in visually ambiguous
textile structures.
Qualitative Results. Figure 6 presents qualitative bending
stiffness estimation results on real fabric samples from the
MIT Fabric Properties dataset. The dataset contains diverse
materials with visually similar appearances but substan-
tially different mechanical behavior, making stiffness pre-
diction particularly challenging. Across a variety of textile
types, including corduroy, nylon ripstop, outdoor polyester,
and pleather, PhysGS produces dense stiffness fields that
clearly delineate material differences. Each predicted stiff-
ness map exhibits smooth spatial variation and preserves
mask-level boundaries, reflecting the underlying compli-
ance characteristics of each fabric.
B.2. Terrain Friction Estimation
Dataset. We evaluate terrain friction prediction using the
Terrain Class Friction dataset from [10]. The dataset con-
2

<!-- page 14 -->
tains paired RGB images and friction measurements for
seven common indoor and outdoor terrain classes, including
carpet, concrete, laminated flooring, rubber, pebbles, rocks,
and wood (see Table 5). Following the protocol in [10], we
assess prediction accuracy against the mean coefficients of
friction obtained from their unimodal Gaussian fits for each
terrain class.
Table 5. Mean (µ) and standard deviation (σ) of coefficients of
friction reported in [10] for the Terrain Class Friction Dataset.
Terrain Class
µ
σ
Concrete
0.543
0.065
Pebbles
0.428
0.059
Rocks
0.478
0.113
Wood
0.372
0.055
Rubber
0.616
0.048
Carpet
0.583
0.068
Laminated Flooring
0.311
0.045
Metrics. We report the same evaluation metrics for terrain
friction estimation (lbf-in2) used in evaluation for mass es-
timation as above: ADE, ALDE, APE, MnRE.
Baselines. We compare our model against several visual
and multimodal baselines on the ABO-500 dataset:
• GPT-4V: We provide GPT-4V with the image, and ask it
to estimate the friction of the terrain.
• GPT-5: Same prompt as GPT-4V.
Quantitative Results. Table 6 reports quantitative results
comparing our method against GPT-4V and GPT-5 VLM
baselines. As the dataset consists of single-object, mostly
homogeneous surfaces, the benefits of precise part-level
segmentation are limited in this setting. Nevertheless, our
hierarchical prompting scheme enables both global and lo-
cal reasoning by guiding the VLM to focus on the dominant
surface region while still incorporating contextual cues such
as reflectance, roughness, and material structure. Across all
four metrics, ADE, ALDE, APE, and MnRE, our method
performs on par with or better than GPT-4V and GPT-5.
Qualitative Results. Figure 7 presents qualitative friction
estimation results on samples from the Terrain Class Fric-
tion dataset. Given an input RGB image, PhysGS produces
smooth and spatially consistent friction fields that align with
the visual regions of each surface.
The predicted maps
clearly distinguish materials such as carpet, wood, and com-
posite flooring, capturing their characteristic friction pat-
terns while preserving coherent region boundaries.
B.3. Outdoor Scene Analysis
Figure 5 shows qualitative results of PhysGS applied to
outdoor environments with diverse terrain types, vegeta-
tion, and natural materials.
From a single RGB image,
Table 6. Friction estimation on Terrain Class Friction dataset (30
objects). ADE is measured in lbf-in2. Bold: best model.
Method ADE (↓) ALDE (↓) APE (↓) MnRE (↑)
GPT-4V
0.129
0.315
0.286
0.747
GPT-5
0.146
0.253
0.291
0.779
Ours
0.126
0.251
0.290
0.783
Ours 
(Friction Coefficient)
Input RGB Image
Figure 7. Friction estimation on samples from the Terrain Class
Friction dataset. PhysGS produces smooth, coherent friction maps
that differentiate surfaces such as carpet, wood, and composite
flooring directly from RGB input.
our model predicts material segmentation, friction coeffi-
cients, stiffness (Young’s modulus) fields, and total uncer-
tainty (aleatoric + epistemic). These results demonstrate the
ability of PhysGS to extend beyond controlled indoor set-
tings and operate on unstructured outdoor scenes.
Across all examples, the predicted material maps pro-
vide reasonable semantic decomposition of natural surfaces
such as gravel, grass, bark, mud, water, and leaf litter. The
corresponding friction and stiffness fields reflect meaning-
3

<!-- page 15 -->
Original Image
Mask Overlay
Part Image
Figure 8. Imprecise masks generated by SAM as can be seen in
the mask overlay and the part images. This results in less clear
material boundaries and higher downstream uncertainty.
ful physical differences between these materials: solid re-
gions such as rock, concrete, or bark consistently receive
higher stiffness values, whereas deformable surfaces such
as mud and grass yield lower estimated moduli. Friction
estimates likewise align with expected terrain properties,
capturing transitions between slippery, saturated mud and
higher-friction vegetation or gravel.
The total uncertainty maps reveal a strong correlation be-
tween uncertainty and the quality of SAM-generated seg-
mentations, consistent with the discussion in the limitations
section (see Sec. 5). Rows 2 and 3 in Figure 5 contain dense
clutter, irregular textures, or ambiguous boundaries (e.g.,
intertwined vegetation or mud–grass transitions), leading
SAM to produce noisier part-level masks. As illustrated ex-
plicitly in Figure 8, these mask inaccuracies propagate into
the part images and result in less reliable material evidence.
In such cases, PhysGS assigns noticeably higher total un-
certainty, driven by both epistemic uncertainty from incon-
sistent material cues and aleatoric uncertainty arising from
intra-region variability.
Conversely, rows 1 and 4 in Figure 5 contain large, spa-
tially coherent surfaces (e.g., gravel, sky, uniform grass),
where SAM produces cleaner segmentations. In these set-
tings, PhysGS yields lower uncertainty and more stable
physical predictions across the scene. Taken together, these
results, supported by both Figures 5 and 8, demonstrate that
the Bayesian uncertainty estimates are meaningfully sensi-
tive to segmentation quality and reliably signal when the
input evidence is less trustworthy.
C. Additional Experimental Details
C.1. Prompting Details
Figures 9 and 10 show the exact prompting configurations,
inspired by [60], used for the MIT Fabric Properties dataset
and the RUGD outdoor dataset. In both cases, the VLM
is provided with the original RGB image, a segmentation-
mask overlay, and an isolated part image. The text prompt
    You are given three related images:
    1. The left image shows the full scene in its original form (Original Image).
    2. The middle image shows the same scene with a segmentation overlay 
        highlighting the region of interest.
    3. The right image isolates the visible portion of that region. Black areas are masked 
        and must be ignored. Only the colored region in the right image is relevant for 
        analysis.
    Your task:
    - Focus **only** on the visible (non-black) region in the right image.
    - Provide a brief caption describing that visible region (e.g., texture/pattern/
       appearance).
    - Predict the **most likely fabric material** the visible region is composed of (from 
      the library below).
    - For that material, estimate:
    - Its **friction coefficient** (range 0–1).
    - Its **bending stiffness** in **lbf·in²** (numeric).
    - Your **confidence** (range 0–1, two decimal places) in this material prediction.
    ### Output Format (strictly follow this structure):
    (caption, [material_1, friction_1, stiffness_1, confidence_1])
    ### Example:
    "plaid textile swatch, [wool, 0.45, 0.04, 0.88]"
    ### Rules:
    - Confidence ∈ [0.00, 1.00], exactly two decimal places.
    - Friction ∈ [0.00, 1.00].
    - Stiffness must be numeric in **lbf·in²** (e.g., 0.04).
    - Do **not** include any extra commentary, explanations, or units outside the 
       format.
    - Only describe and evaluate the **colored region** in the rightmost image; ignore 
       all black areas.
    - Material names must be chosen from the provided common material library: 
      {material_library}.
Visual Prompt:
Original Image
Mask Overlay
Part Image
Text Prompt:
Figure 9. VLM Prompt used to obtain material, friction, and bend-
ing stiffness predictions for the MIT Fabric Properties dataset.
directs the model to ignore masked regions and focus only
on the visible segment, ensuring that predictions are part-
specific rather than influenced by the surrounding scene.
We also maintain separate indoor and outdoor material li-
braries so the VLM selects from the most appropriate set of
materials for each environment.
For each part, the VLM returns one or more candi-
date materials with associated physical properties and confi-
dence scores. Each of these candidate predictions is treated
as a confidence-weighted observation within our Bayesian
framework, allowing PhysGS to fuse evidence across views
and produce consistent material and property estimates.
Importantly, the distribution of confidence across multiple
materials provides a direct signal of semantic ambiguity.
When the VLM is uncertain, often due to noisy or impre-
cise SAM segmentations, the confidence spread increases,
which propagates into higher predictive uncertainty in our
property fields, consistent with the trends discussed in the
limitations section (Sec. 5).
4

<!-- page 16 -->
You are given three related images:
    1. The left image shows the full outdoor scene in its original form (Original Image).
    2. The middle image shows the same scene with a segmentation overlay 
        highlighting the region of interest.
    3. The right image isolates the visible portion of that region; **black areas are 
        masked and must be ignored.** Only the colored region in the right image is 
        relevant for analysis.
    Your task:
    - Focus **only** on the visible (non-black) region in the right image.
    - Provide a brief caption describing that visible region.
    - Predict the **three most likely materials** the visible region could be composed of, 
in descending order of likelihood.
    - For each material, estimate:
    - Its **friction coefficient** (0–1 range).
    - Its **stiffness** (Young’s Modulus in **GPa**).
    - Your **confidence** (0–1, two decimal places) in that material prediction.
    ### Output Format (strictly follow this structure):
    (caption, [material_1, friction_1, stiffness_1, confidence_1], [material_2, friction_2, 
    stiffness_2, confidence_2], [material_3, friction_3, stiffness_3, confidence_3])
    ### Example:
    "rough gravel surface, [gravel, 0.72, 30.0, 0.90], [asphalt, 0.68, 50.0, 0.74], 
     [soil, 0.55, 15.0, 0.61])"
    ### Rules:
    - Each confidence value must be between 0.00 and 1.00 (two decimals).
    - Each friction value must be between 0.00 and 1.00.
    - Stiffness (Young’s Modulus) must be given in **GPa**, numeric only (e.g., 50.0).
    - Do **not** include any extra commentary, explanations, or text outside the 
      specified format.
    - Focus only on the **visible region** in the rightmost image; ignore background and 
      masked (black) regions entirely.
    - Material names must be chosen from the provided common material library: 
      {material_library}.
Visual Prompt:
Text Prompt:
Original Image
Mask Overlay
Part Image
Figure 10. VLM prompt used to obtain material, friction, and
stiffness predictions for the RUGD dataset. By predicting mul-
tiple plausible materials with associated confidences, this prompt-
ing strategy enables PhysGS to estimate the total uncertainty for
each mask.
C.2. Baseline Details
To benchmark PhysGS against existing vision–language
models, we evaluate GPT-4V and GPT-5 on the MIT Fabric
Properties and Terrain Class Friction datasets using a sim-
plified prompting strategy tailored for fair comparison (see
Figure 11). For each image, the VLM receives only the raw
RGB frame and is instructed to (1) describe the dominant
visible region, (2) predict the most likely material based
solely on visual appearance, and (3) estimate a friction co-
efficient, stiffness value, and confidence score.
This baseline prompt does not include segmentation cues
or part-based isolation, and therefore tests each VLM’s abil-
ity to infer material and physical properties directly from
appearance alone. The resulting predictions serve as a ref-
erence for evaluating the gains provided by our part-aware
prompting, used in PhysGS.
    You are given an image of a scene or object.
    Your task:
    - Observe the image and identify the **most dominant material** visible.
    - Provide a short caption describing the visible region (e.g., color, texture, 
      or surface characteristics).
    - Predict the **most likely material** based on visual appearance alone.
    - Estimate:
    - Its **friction coefficient** (range 0–1).
    - Its **stiffness** (in appropriate units depending on the material type, e.g., 
      GPa for solids or lbf·in² for fabrics).
    - Your **confidence** (range 0–1, two decimal places) in this material prediction.
    ### Output Format (strictly follow this structure):
    (caption, [material_1, friction_1, stiffness_1, confidence_1])
    ### Example:
    "shiny metallic surface, [metal, 0.25, 200.0, 0.93]"
    ### Rules:
    - Confidence ∈ [0.00, 1.00], exactly two decimal places.
    - Friction ∈ [0.00, 1.00].
    - Stiffness must be numeric (e.g., 0.03 or 50.0).
    - Do **not** include any commentary, units, or text outside the specified format.
    - Base your answer entirely on the **visual cues** in the provided image 
      (color, gloss, pattern, texture, reflectance).
Visual Prompt:
Text Prompt:
Figure 11. Baseline VLM Prompt used to obtain material, friction,
and bending stiffness predictions for the MIT Fabric Properties
dataset.
5
