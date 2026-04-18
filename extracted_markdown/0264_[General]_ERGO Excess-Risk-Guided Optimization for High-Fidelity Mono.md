<!-- page 1 -->
ERGO: Excess-Risk-Guided Optimization for High-Fidelity Monocular 3D
Gaussian Splatting
Zehua Ma1*
Hanhui Li1*
Zhenyu Xie3
Xiaonan Luo4
Michael Kampffmeyer5
Feng Gao2†
Xiaodan Liang1,3†
1Shenzhen Campus of Sun Yat-sen University
2Peking University
3Mohamed bin Zayed University of Artificial Intelligence
4Guilin University of Electronic Technology
5UiT The Arctic University of Norway
Abstract
Generating 3D content from a single image remains a fun-
damentally challenging and ill-posed problem due to the
inherent absence of geometric and textural information in
occluded regions. While state-of-the-art generative mod-
els can synthesize auxiliary views to provide additional su-
pervision, these views inevitably contain geometric incon-
sistencies and textural misalignments that propagate and
amplify artifacts during 3D reconstruction. To effectively
harness these imperfect supervisory signals, we propose
an adaptive optimization framework guided by excess risk
decomposition, termed ERGO. Specifically, ERGO decom-
poses the optimization losses in 3D Gaussian splatting into
two components, i.e., excess risk that quantifies the sub-
optimality gap between current and optimal parameters,
and Bayes error that models the irreducible noise inherent
in synthesized views. This decomposition enables ERGO
to dynamically estimate the view-specific excess risk and
adaptively adjust loss weights during optimization. Fur-
thermore, we introduce geometry-aware and texture-aware
objectives that complement the excess-risk-derived weight-
ing mechanism, establishing a synergistic global-local op-
timization paradigm. Consequently, ERGO demonstrates
robustness against supervision noise while consistently en-
hancing both geometric fidelity and textural quality of the
reconstructed 3D content.
Extensive experiments on the
Google Scanned Objects dataset and the OmniObject3D
dataset demonstrate the superiority of ERGO over existing
state-of-the-art methods.
1. Introduction
Single-image 3D content generation (as illustrated in Fig-
ure 1) has attracted considerable attention and facilitated
*Equal contribution.
†Corresponding author.
a wide range of applications spanning virtual reality, aug-
mented reality, and game development [63]. Nevertheless,
ensuring cross-view texture consistency and fidelity of the
generated 3D content remains challenging due to the lim-
ited information provided by single-view images. Exist-
ing methods for this task can be broadly categorized into
two mainstream paradigms, namely, feed-forward large re-
construction models (LRMs) and optimization-based meth-
ods. Specifically, LRMs [25, 30, 36] leverage large-scale
3D datasets to train end-to-end models for direct 3D content
prediction. These models exhibit high inference efficiency,
yet they demand substantial computational resources for
training and often suffer from degraded performance on out-
of-distribution data. In contrast, optimization-based meth-
ods [49, 53, 55] typically consist of an iterative optimiza-
tion process driven by heuristic objectives, such as score
distillation sampling (SDS) [48] that exploit texture priors
from pretrained image generative models. However, these
methods are prone to generating results with cross-view in-
consistency and blurry textures, due to inconsistent genera-
tive priors induced by diverse camera poses.
One potential solution to mitigate the information
scarcity in unobserved views is to employ multi-view diffu-
sion (MVD) models [40, 42, 51, 52, 64] to generate comple-
mentary views. With these synthesized views and cutting-
edge inverse rendering techniques, e.g., 3D Gaussian splat-
ting (3DGS) [33, 70], conducting an efficient optimization
pipeline becomes feasible. This strategy further has the po-
tential to mitigate the blurry textures caused by SDS-based
optimization methods, as generated content is grounded by
the explicit geometry representation of 3DGS.
Nevertheless, a direct integration of MVD models and
optimization-based methods tends to introduce new types
of artifacts due to inherent inconsistencies across the gen-
erated multi-view images. As shown in Figure 2, these in-
consistencies can be geometric or textural: geometry incon-
sistencies are mainly caused by the imperfect 3D modeling
arXiv:2602.10278v1  [cs.CV]  10 Feb 2026

<!-- page 2 -->
Input Image
ERGO (Ours)
Optimization-based
Feed-forward
Figure 1. Given single-view images as inputs, the proposed ERGO method can generate 3D objects with better texture consistency and
fidelity, compared with state-of-the-art optimization-based methods (e.g., [41]) and feed-forward large models (e.g., [23]).
ability of existing MVD models, while texture inconsisten-
cies result from the decoupling of texture generation from
explicit geometric constraints. Consequently, such view-
inconsistent images introduce spurious supervision signals,
thereby hindering the stability and convergence of the opti-
mization process.
To address the consistency limitations inherent in cur-
rent multi-view optimization frameworks, we propose an
excess-risk-guided optimization (ERGO) framework in this
paper. Unlike conventional optimization-based approaches
that assign heuristic and uniform weights to loss objectives
across all views, ERGO adaptively estimates and adjusts the
weights of each objective during the iterative optimization
process. Specifically, ERGO stems from excess risk decom-
position [22, 46, 59] that decomposes empirical risk (opti-
mization error) into excess risk and Bayes error. Excess
risk measures the discrepancy between current model pa-
rameters and theoretical optimal parameters, reflecting the
potential of the model for further improvement via opti-
mization. In contrast, Bayes error is inherently attributed to
noise in the supervision signals, which corresponds to the
geometric and textural inconsistencies in our context. Ac-
cordingly, estimating excess risk allows us to dynamically
modulate the optimization process and mitigate the adverse
effects of Bayes error. We achieve this by adaptively as-
signing higher global weights to loss objectives and views
with greater excess risk, as these components are more in-
formative for guiding the model toward the optimal param-
eter space.
Furthermore, to complement the global adaptive weight-
ing from excess risk estimation, we introduce a geometry-
aware objective and a texture-aware objective.
The
geometry-aware objective focuses on local geometric con-
sistency across multi-view images.
It leverages visibil-
ity maps generated via 3DGS to adaptively adjust the loss
weight of each local region based on its geometric relia-
bility. Meanwhile, the texture-aware objective models re-
gional texture complexity to facilitate local texture fidelity
and detail preservation. This global-local adaptive design
enables ERGO to simultaneously achieve inter-view consis-
tency (via global modulation) and intra-view fine-grained
quality (via local adjustment), effectively alleviating the
geometric and textural artifacts arising from naive MVD-
optimization integration.
Overall, our contributions can be summarized as follows:
• We propose ERGO, the first single-image to 3D frame-
work that enables adaptive multi-view optimization via
excess risk decomposition;
• A geometry-aware objective and a texture-aware objec-
tive are introduced to perform local adaptive modulation;

<!-- page 3 -->
• Extensive experiments on two public benchmarks (i.e.,
GSO [19], OmniObject3D [69]) demonstrate that ERGO
outperforms state-of-the-art methods both qualitatively
and quantitatively.
2. Related Work
2.1. Optimization-based Methods
With the success of image diffusion models,
many
optimization-based methods have been proposed for novel-
view synthesis and 3D content creation. DreamFusion [48]
first introduces the score distillation sampling (SDS) strat-
egy, which enables the generation of plausible 3D content
by leveraging image diffusion models to mitigate the need
for 3D datasets. A variational improvement of SDS is in-
troduced in [66], which reduces texture oversaturation and
improves diversity. To improve content fidelity, some re-
searchers [43, 49] further explore textual inversion, which
was originally designed for image customization.
More recently, a series of studies [9, 38, 49, 55]
adopt multi-stage optimization strategies to achieve high-
resolution and detail-rich 3D models.
For example,
Magic3D [38] and Magic123 [49] optimize low-resolution
neural radiance fields (NeRFs) in their first stage. In the sec-
ond stage, they transfer NeRFs to a more efficient 3D rep-
resentation [50] to generate high-resolution meshes. Fanta-
sia3D [9] and HumanNorm [27] disentangle geometry and
texture, recovering finer geometries and achieving photo-
realistic rendering. To further enhance the controllability
of 3D generation, NeuralLift [71], 3DStyle-Diffusion [75],
and Control3d [11] incorporate additional information (e.g.,
depth and sketch) for optimization.
While the above methods demonstrate superior perfor-
mance in zero-shot 3D synthesis, they still suffer from the
Janus (multi-faced) problem and blurred results caused by
their insufficient perception of 3D geometry.
Moreover,
they typically require minutes or even hours to generate a
single 3D model. To address these limitations, our proposed
ERGO framework employs 3D Gaussians to achieve effi-
cient modeling of geometry and provide geometric priors,
enabling the generation of realistic textures.
2.2. Feed-Forward Models
The advance of network architectures and large-scale 3D
datasets facilitates data-driven feed-forward models [7,
10, 14, 62, 73, 78], alleviating the prohibitive computa-
tional cost of optimization-based methods. For example,
Point·E [45] employs a transformer-based model to regress
RGB point clouds from input images and LION [60] con-
structs a hierarchical VAE, encoding point clouds into a la-
tent space for reconstruction. Both of these works generate
explicit 3D assets. In contrast, Shap-E [30] generates pa-
rameters for implicit functions that can be rendered as both
textured meshes and neural radiance fields. These early re-
search efforts produce plausible 3D models with simple ge-
ometry and texture, yet they yield collapsed results under
complex conditions due to the lack of large-scale data. Re-
cently, by utilizing large-scale 3D datasets [16, 17, 69] for
training, the generality and quality of 3D generative models
have been improved significantly.
As in feed-forward methods, LRM [23] is a pioneer in
utilizing a regression model to predict NeRF representations
from a single image, with subsequent works expanding it
to image-to-gaussian and image-to-mesh generation. [85]
employs a hybrid triplane-based Gaussian representation,
effectively balancing rendering speed and quality.
Mov-
ing away from regression-based models, [68, 79, 82, 83]
incorporate latent diffusion transformers (DiTs) for itera-
tive denoising and learning 3D distributions in latent space.
Clay [79] further scales up its generative model, enhancing
its generalization and diversification capabilities. To tackle
the limitations of implicit latent representations in efficient
encoding, Direct3D [68] adopts explicit triplane latent rep-
resentations and employs a geometric mapping network to
predict 3D occupancy grids.
Given a single reference image, these feed-forward mod-
els tend to produce blurry textures and overly smooth ge-
ometries when applied to unseen viewpoints. An alternative
approach involves combining multi-view generation mod-
els to supplement information in unseen viewpoints before
executing the 3D reconstruction process.
Building upon
the LRM architecture, Instant3D [36] and InstantMesh [72]
address the Janus problem and generate high-fidelity 3D
meshes. However, maintaining multi-view consistency re-
mains challenging and often leads to low-quality geomet-
ric structures.
Craftsman [37] employs a 3D diffusion
model conditioned on multiple views to generate coarse
3D geometry and a normal-based geometry refiner to en-
hance surface details significantly. LGM [56] further im-
proves upon this by utilizing Gaussian splatting for higher
fidelity 3D results. Additionally, [6, 12, 54] parameterizes
3D assets using pixel-aligned Gaussians and fuses multi-
ple per-view Gaussians, enabling the efficient generation of
high-resolution 3D content. While these models achieve
high inference efficiency, they typically incur significant
training costs and exhibit limited generalization to out-of-
distribution scenarios. ERGO, by contrast, formulates 3D
reconstruction as an optimization problem and directly ex-
ploits the rich priors of pretrained image generative models,
effectively mitigating these limitations.
2.3. Multi-view Diffusion Model
To address view ambiguities, researchers have used multi-
view diffusion [4, 52, 67, 77, 81, 84] to supplement missing
information. Pretrained on a billion-scale dataset of image-
text pairs, image generation models [1, 8, 20] have gained

<!-- page 4 -->
Artifacts Caused by Cross-view Inconsistency
View-Consistent and High-Quality 3D
Inconsistent Multi-view Images
Input Single Image
Recon-
struction
Excess-Risk-
Guided 
Optimization
Geometry Inconsistency
Texture Inconsistency
Multi-view 
Diffusion Model
Multi-view 
Diffusion Model
SDS-based 
Optimization
(d)
(a)
(b)
(c)
Blurry Texture and Lack of Details in Unseen Areas
Figure 2. Comparison of various optimization paradigms for single-image-to-3D generation, including (a) optimization-based methods,
(b) direct reconstruction with multi-view synthesized images, and (c) the proposed ERGO framework with adaptive objective weights. (d)
Illustration of two types of inconsistency caused by the direct reconstruction with multi-view inconsistent images.
abundant 3D prior knowledge. By simply fine-tuning the
image diffusion model on 3D datasets, Zero1-to-3 [40] is
able to generate novel views of the same object when given
a single view and the corresponding camera transforma-
tions. However, the multi-view images produced by Zero1-
to-3 often exhibit inconsistencies due to the lack of informa-
tion interaction across the different views. A series of stud-
ies [18, 24, 28, 29, 58] have been dedicated to improving
the consistency of multi-view images, which is crucial for
subsequent 3D reconstruction. Zero123++ [51] combines
six images in a 3 × 2 layout into a single frame, generat-
ing multiple views in one forward pass. SyncDreamer [41]
utilizes a 3D feature volume as a unified feature representa-
tion for multi-view images, facilitating information fusion
across different views. Wonder3D [42] learns the joint dis-
tribution of normal maps and RGB images, enhancing the
alignment of 3D object geometry and texture.
Certain studies have turned their attention to video gen-
eration models [2, 3, 5], given that video models’ inher-
ent temporal and spatial coherence dovetail perfectly with
the consistency required for multi-view generation. Var-
ious 3D guidance and conditions have been incorporated
into video generation models to improve multi-view con-
sistency. For instance, VideoMV [86] adopts a 3D-aware
denoise sampling strategy that effectively inserts images
rendered from 3D into its denoising process. Hi3D [76]
leverages depths as conditions for a video-to-video refiner,
generating high-resolution views with detailed texture. RE-
CONX [39] builds a global point cloud and uses it as 3D
structure guidance. Additionally, subsequent research has
improved upon model flexibility, allowing for more free-
dom in inputs and outputs. Vivid-1-to-3 [35] combines dif-
fusion models for both image and video to generate video
frames conditioned on camera trajectories. V3D [47] incor-
porates a pixelNeRF encoder, which could be seamlessly
adapted to any number of input images. SV3D [61] further
adds explicit camera controls for novel view synthesis so
that azimuths can be irregularly spaced and elevations can
vary per view. However, the multi-view images generated
by these methods still suffer from cross-view inconsisten-
cies. Our ERGO framework addresses this issue through
excess-risk-guided optimization, enabling the generation of
3D reconstructions with strong geometric consistency and
realistic textures.
3. Preliminaries
3.1. 3D Gaussian Splatting
3D Gaussian Splatting (3DGS) [33] is based on an explicit
representation that uses a set of points with optimizable
attributes for rendering an object or scene, e.g., position,
color, and density. The rendering process is conducted by
projecting each Gaussian point onto an image plane, which
can be formulated as,
C =
X
j∈N
cjαj
j−1
Y
k=1
(1 −αk) .
(1)
Here cj and αj denote the color and the opacity of each
point, respectively, N is the number of points, and C rep-
resents the color of a pixel. The Gaussian attributes can be
predicted by a pretrained model or optimized from scratch
when sufficient multi-view images are available.

<!-- page 5 -->
Single-view Image
…
K+1 Images with 
Inconsistencies
Multi-view 
Diffusion Model
3D Gaussian 
Splatting
Coarse 
3D Gaussians
ℒൌ෍𝑎௚௞ℒ௚௞
௄
௞ୀ଴
൅𝑎௧
௞ℒ௧
௞
Empirical Risk:
𝜀𝚯ℒ௜ൌ1
2 Δ𝚯∗ୃ𝛁𝟐ℒ௜𝚯Δ𝚯∗
Excess Risk:
Decomposition
Update Weight
Geometry-aware Objective
Visibility Maps
Depth Maps
Masks
&
&
Texture-aware Objective
Complexity Maps
Edge Maps
Excess-Risk-Guided Optimization
Cross-view Enhanced
3D Gaussians
&
𝚯
ℒ௚௞
ℒ௧
௞
Figure 3. The proposed ERGO framework for single-image 3D content generation. Given coarse 3D Gaussians and synthesized images
with inconsistencies, ERGO not only estimates excess risk to mitigate inconsistencies and modulates iterative optimization globally, but
also leverages the geometry-aware objective and the texture-aware objective to achieve localized refinement.
3.2. Optimization-based 3D Generation
We adopt the optimization-based paradigm for image-to-
3D generation, owing to its flexibility in model selection
and potential for targeted enhancement. A commonly used
strategy is score distillation sampling (SDS) [40, 48], which
minimizes the difference between the noise of a reference
image Ir and that of a rendered image I:
LSDS(I, Ir) = Et,p,ϵ

w(t) (ϵϕ (I; t, Ir, p) −ϵ) ∂I
∂Θ

.
(2)
Here, t is a randomly sampled timestep and p is a randomly
sampled camera pose. ϵ denotes Gaussian noise sampled
from a standard normal distribution, w(t) is a timestep-
related weight, ϵϕ(·) denotes a pretrained noise predictor,
and Θ denotes parameters to be optimized.
4. Methodology
In this section, we present the details of our excess-risk-
guided optimization framework (ERGO) for monocular 3D
Gaussian splatting (3DGS). The core of ERGO is to for-
mulate 3DGS within a weighted multi-objective paradigm
(Sec. 4.1), where the weights assigned to individual ob-
jectives are conditioned on the excess risks estimated from
noisy auxiliary images. The major objectives of ERGO in-
clude a geometry-aware objective (Sec. 4.2) and a texture-
aware objective (Sec. 4.3), which are specifically devised
to emphasize image regions with geometric consistency and
rich details, respectively. Detailed strategies for excess risk
estimation and weight updates are presented in Sec. 4.4.
4.1. Overall Framework
Although it is intuitive to leverage images generated by
an MVD model to optimize Gaussian attributes, a naive
strategy that directly optimizes over the generated images
will inevitably lead to geometric and textural inconsisten-
cies. Such inconsistencies arise from the lack of explicit
3D modeling in most existing MVD models and distribu-
tion discrepancies between training and test data. There-
fore, we propose the ERGO framework to adaptively adjust
the weights of the optimization objectives according to the
quality of the generated images and their effect on the opti-
mization process, as shown in Figure 3.
Specifically, given a reference image I0 and a set of K
auxiliary images A = {I1, ..., IK} generated by an off-the-
shelf MVD model (e.g., [51]), the ERGO framework is de-
signed to optimize Gaussian attributes Θ to solve the fol-
lowing multi-objective optimization problem:
min
Θ
M
X
m=1
K
X
k=0
amkLmk(Θ, Ik),
(3)
where M denotes the number of objectives, amk ∈[0, 1)
represents the weight of Lmk, and Ik ∈{I0, A}. We as-
sume a = {amk} lies in a probability simplex, and hence
P amk = 1. If each Lmk satisfies certain constraints (e.g.,
Lipschitz continuity) [22], Eq. (3) can achieve a Pareto sta-
tionary solution, i.e., an optimal solution that has no fea-
sible direction to reduce any of the weighted loss terms
amkLmk(Θ, Ik) without increasing others. While this con-
dition is non-trivial to satisfy, as some objectives are non-
convex or their tight upper bounds are difficult to derive,
our empirical results demonstrate that the proposed method
outperforms the baseline model without adaptive weights.
To model and suppress per-view noise in A, we propose
to decompose the expected loss of Θ with respect to Lmk
into the excess risk and the Bayes error as
R(Θ|Lmk) = ε(Θ|Lmk) + ε(Θ∗|Lmk),
(4)
where R(Θ|Lmk) = E[Lmk(Θ)] denotes the expected loss
and ε(Θ|Lmk) represents the excess risk. Θ∗denotes the
ideal but unknown parameters. The Bayes error ε(Θ∗|Lmk)
is irreducible, due to the stochastic nature of MVD genera-
tion and the partial observability of ground-truth multi-view
data. Fortunately, we can approximate ε(Θ|Lmk) using the

<!-- page 6 -->
GroundTruth
Synthesized 
Image
Viewpoint 
Transformation 
left 0°
left 30°
left 60°
left 90°
left 120° 
left 150°
left 180°
GroundTruth
Synthesized 
Image
Figure 4. The performance of the MVD baseline degrades as the
viewpoint transformation magnitude increases.
gradients of the empirical risk of Θ and its Taylor expan-
sion, without explicitly estimating Θ∗or ε(Θ∗|Lmk), as
detailed in Sec. 4.4.
By rearranging Eq. (4) as ε(Θ|Lmk) = R(Θ|Lmk) −
ε(Θ∗|Lmk), it is evident that the excess risk quantifies the
gap between the performance of the current parameters and
that of the optimal parameters. Therefore, we can use the
gradient norm of ε(Θ|Lmk), denoted as ||∇ε(Θ|Lmk)||,
to adaptively adjust amk and promote convergence. Under
the probability simplex assumption of a, the ERGO frame-
work allows us to approximate a Pareto stationary among
the objectives across all views. Note that Eq. (3) serves as
a global weighting mechanism, and hence we introduce the
geometry-aware objective and the texture-aware objective
in the subsequent sections to achieve localized adjustments
of objective weights.
4.2. Geometry-Aware Objective
We introduce the geometry-aware objective to alleviate the
absence of geometric constraints in conventional optimiza-
tion objectives of MVD models (e.g., SDS). The geometry-
aware objective consists of two components: (i) explicit ge-
ometric loss terms, and (ii) an adaptive loss term modulated
by geometric properties.
Geometry Correction. We first propose a geometry cor-
rection step to construct a coarse 3D Gaussian model Gc,
which serves as pseudo-geometric ground truth. We fol-
low [40] for constructing Gc by minimizing the SDS loss
over I = {I0, A}. We then use Gc to calculate depth maps
D = {D0, ..., DK} and object masks M = {M0, ..., MK}
corresponding to each view in I. Each depth map Dk is nor-
malized to [0, 1], and the elements belonging to the object in
Mk are 1, otherwise 0. Furthermore, to mitigate geometric
inconsistency across I, rather than using A for subsequent
optimizations, we also use Gc to render the counterparts of
the images in A and employ DDIM inversion [44] to convert
them back into the noise latent space of the MVD model
𝐼ହ
𝐼ସ
𝐼ଷ
𝐼଺
𝐼ଵ
𝐼ଶ
𝐼଴
𝑉ଵ
3D Gaussians
Figure 5. Illustration of visibility map generation. We lift the pix-
els from one image into 3D space and project them onto the adja-
cent image to identify the corresponding pixels. We then calculate
the differences between these corresponding pixels to generate the
visibility map.
for denoising. This allows us to acquire auxiliary images
with refined textures and geometric consistency. For con-
ciseness, we retain the notation A for these refined auxiliary
images.
Our geometry-aware objective Lg is formulated as fol-
lows:
Lg = λvLv + λdLd + λmLm.
(5)
λv, λd, and λm are scalar coefficients to normalize the
loss terms and preserve the probability simplex property of
the adaptive weights a. Ld and Lm are two explicit ge-
ometric losses defined as Ld = PK
k=0 ∥D′
k −Dk∥2
2 and
Lm = PK
k=0 ∥M ′
k −Mk∥2
2, where D′
k and M ′
k are the
depth map and object mask predicted during the optimiza-
tion process, respectively. Lv is the adaptive loss term mod-
ulated by visibility maps, which is the core of our geometry-
aware objective. The key idea of λv is to assign higher
weights to visible image regions, as these regions are ge-
ometrically more reliable, as shown in Figure 4.
Due to the coarse geometry of Gc, we refrain from ren-
dering methods (e.g., z-buffering) for visibility map com-
putation. As illustrated in Figure 5, we leverage D and M
to establish pixel-wise correspondences between adjacent
views and compute visibility maps. Let pk denote the cam-
era parameters for the k-view. We estimate the pixel-wise
correspondences between Dk and Dk+1 by first lifting the
pixels in Dk to the 3D space with the inverse of pk, fol-
lowed by projecting them to the image plane of view k + 1
using pk+1. This yields a warped depth map Dk→k+1, rep-
resenting the depth values of points from view k as observed
in view k + 1. The visibility map Vk+1 for view k + 1 is
then defined as,
Vk+1 = max(0, Mk+1 ⊙(Dk→k+1 −Dk+1)),
(6)

<!-- page 7 -->
Current Image
Adjacent Image
Visibility Map
Adaptive 
Weight Map
𝐼0
𝐼1
𝐼2
𝐼3
𝐼4
𝐼5
𝐼6
𝐼0
𝐼1
𝐼2
𝐼3
𝐼4
𝐼5
𝐼6
Figure 6. Visualizations of visibility maps and adaptive weight maps conditioned on discrepancy. These weight maps have high values in
visible and consistent regions.
where ⊙denotes element-wise multiplication. The intuition
behind Eq. (6) is that if the depth value of a pixel in Dk after
the transformation is smaller than that of the corresponding
pixel in Dk+1, then the 3D surface point corresponding to
the pixel is nearer the camera and thus visible.
Leveraging the visibility maps, we formally define the
visibility-aware loss term Lv as,
Lv =
K
X
k=0
Vk ⊙Wi ⊙∥I′
k −Ik∥2
2 +(1−Vk)⊙∥I′
k −Ik∥2
2 ,
(7)
where the first and second terms on the right-hand side re-
spectively modulate the supervision weights assigned to the
visible and invisible pixel regions. Wi = 1 −(Ik−1→k −
Ik)s is an adaptive weight that is conditioned on the dis-
crepancy between Ik and Ik−1→k, where the latter is ob-
tained via the same warping process as Dk−1→k. s > 0
is a hyperparameter. Wi assigns higher weights to regions
where the warped content Ik−1→k closely matches Ik, indi-
cating consistent multi-view geometry for supervision. We
set W0 = V0 = 1. Examples of the visibility maps and
weight maps are visualized in Figure 6.
4.3. Texture-Aware Objective
In addition to the geometry-aware objective, we also intro-
duce a texture-aware objective to better exploit the texture
priors in the MVD base model. However, the conventional
SDS loss tends to over-smooth fine-grained textures [31],
which is exacerbated when learning on view-inconsistent
images. To address this issue and facilitate the learning of
texture details, our texture-aware objective adaptively as-
signs SDS loss weights to distinct image regions based on
their texture complexity.
Formally, our texture-aware objective Lt is defined as,
Lt = λt
K
X
k=0
(1 −d(c, ck))W c
k ⊙LSDS(I′, Ik).
(8)
Here, LSDS is the SDS loss defined in Eq. (2). λt is a hy-
perparameter that balances the contribution of Lt among all
objectives. d(c, ck) denotes the distance (e.g., Euclidean
distance) between a randomly sampled camera center c and
the camera center of the reference image Ik.
The dis-
tance is normalized across all K + 1 reference images, i.e.,
d(c, ck) :=
d(c,ck)
PK
j=0 d(c,cj), ensuring reference images closer
c exert greater influence. W c
k is a texture complexity weight
that adaptively adjusts the SDS loss weight assigned to Ik.
Various methods can be used to implement W c
k, such as
pattern entropy [32] and weighted fusion of multiple visual
cues [15]. In this paper, we adopt a lightweight implemen-
tation by modeling texture complexity using the pixel-wise
edge map of the predicted image I′
k. This is achieved ef-
ficiently via the classic Sobel operator [21], which can be
formulated as,
W c
k = 1 + Sobel(I′
k) ⊛F,
(9)
where F denotes a Gaussian filter employed to smooth out
pixel-wise edge responses and ⊛represents the convolution
operation. In this way, our texture-aware objective not only
prioritizes reference images that are spatially closer to the
sampled camera to enhance geometric consistency, but also
allocates higher SDS loss weights to texture-rich regions,
effectively preserving fine-grained details.
4.4. Optimization
By integrating the geometry-aware objective and the
texture-aware objective, the overall optimization objective

<!-- page 8 -->
of our ERGO framework is formally defined as follows,
L =
K
X
k=0
ak
gLk
g + ak
t Lk
t ,
s.t.
K
X
k=0
ak
g + ak
t = 1,
(10)
where the superscript k indexes the components of both
adaptive weights and loss objectives corresponding to the
k-th view. We optimize the Gaussian parameters Θ and
the adaptive weights a alternatively, so that we can analyze
each loss term in L when we update a with fixed Θ.
Formally, let Li ∈{L0
g, ..., LK
g , L0
t, ..., LK
t } denote an
arbitrary loss component in the overall objective L. To esti-
mate the excess risk of Li, we follow [22] to assume the
optimal Gaussian parameters Θ∗can be locally approxi-
mated as Θ∗= Θ + ∆Θ∗, where ∆Θ∗denotes a small
perturbation in the local neighborhood of Θ. Furthermore,
the second-order Taylor expansion of Li(Θ+∆Θ) is given
by,
Li(Θ + ∆Θ) ≈
Li(Θ) + ∇Li(Θ)⊤∆Θ
+ 1
2∆Θ⊤∇2Li(Θ)∆Θ,
(11)
where ∇Li(Θ) denotes the gradient of Li(Θ) and
∇2Li(Θ) is the corresponding Hessian matrix. With the
assumption that Θ∗is locally optimal, ∇Li(Θ∗) = 0 and
hence,
∇Li(Θ) + ∇2Li(Θ)∆Θ∗= 0.
(12)
Substituting Eq. (12) into Eq. (11), we have:
Li(Θ∗) = Li(Θ) −1
2∆Θ∗⊤∇2Li(Θ)∆Θ∗.
(13)
Based on the excess risk decomposition defined in Eq. (4),
we can now estimate ε(Θ|Li) as,
ε(Θ|Li) = 1
2∆Θ∗⊤∇2Li(Θ)∆Θ∗.
(14)
Here ∇2Li(Θ) can be computed directly at Θ, and ∆Θ∗
can be solved by inverting ∇2Li(Θ) according to Eq. (12),
i.e., ∆Θ∗= −(∇2Li(Θ))−1∇Li(Θ).
Accordingly, we assign higher adaptive weights to loss
terms with larger excess risks. This enables us to focus on
under-optimized objectives and mitigate Bayes error (i.e.,
noise in auxiliary views). Following [22], we reformulate
the objective of Eq. (10) as min
Θ max
ai∈a aiLi, and update the
adaptive weight ai iteratively as follows:
an+1
i
=
an
i exp(ηε(Θ|Li))
P|a|
j=0 an
j exp(ηε(Θ|Lj))
,
(15)
where n denotes the n-th optimization step and η is a hy-
perparameter that controls the sensitivity of weight updates
to excess risk estimates.
5. Experiments
5.1. Setup
Implementation details. All experiments are conducted on
a single NVIDIA A100 GPU with 40 GB GPU memory. We
adopt Zero123++ [51] as our base MVD model. For each
test case, we randomly initialize 5000 3D Gaussian particles
within a sphere of radius 0.5. Each particle is set with an ini-
tial opacity value of 0.1 and a grey color with [r, g, b]=[128,
128, 128]. The Gaussian particles are optimized for 1500
interactions using the Adam optimizer [34]. The learning
rate for the position attribute is initialized to 1 × 10−3 and
is gradually decayed to 2 × 10−5. For the remaining Gaus-
sian attributes, including scale, rotation, opacity, and color,
the learning rates are set to 5×10−3, 5×10−3, 5×10−2, and
1×10−2, respectively. 3D Gaussian densification is carried
out every 100 iterations, while the opacity reset operation is
performed every 500 iterations. Following [74], we remove
floaters every 400 iterations using the K-Nearest Neighbors
algorithm. For the rendering of auxiliary images, the az-
imuth angle is sampled within the range [−180◦, 180◦], and
the elevation angle is constrained to [−30◦, 30◦]. The ren-
dering resolution increases incrementally from 128×128 to
512 × 512 when computing the SDS loss, while remaining
constant at 320 × 320 for pixel-wise loss calculations with
a fixed radius of 2 and FOV of 49.1◦. The hyperparameters
for balancing objectives are set as λv = 1 × 104, λd = 10,
λm = 1 × 103, and λt = 1. The scaling factor s for the
warping process in the geometry objective is set to 4, and
the sensitivity control of weight updates η is set to 3.
Datasets. We utilize two publicly available 3D datasets,
i.e., the Google Scanned Objects (GSO) dataset [19] and
the OmniObject3D dataset [69], both of which contain a di-
verse range of object categories. On the GSO dataset, we
utilize the same 30 objects as used in SyncDreamer [41],
spanning everyday items to animals. Furthermore, we ran-
domly select 50 objects from OmniObject3D, ensuring a
wide variety of categories to verify the versatility and effec-
tiveness of the proposed method. For each object, a single
image with a resolution of 320×320 is rendered as the refer-
ence image, while 16 multi-view images with camera poses
uniformly distributed over an azimuth range of 0◦to 360◦
and with a fixed elevation angle of −30◦.
Baselines and evaluation metrics.
We consider both
optimization-based methods and feed-forward large recon-
struction models for comparison. Specifically, we select
three representative optimization-based methods, includ-
ing SyncDreamer [41], Wonder3D [42], and DreamGaus-
sian [57]. Meanwhile, five cutting-edge large reconstruc-
tion models are selected, including LRM [23], LGM [56],
VideoMV [86], InstantMesh [72], and SAR3D [13]. All
these baselines are evaluated with their provided code
and models.
To conduct a multi-faceted assessment of

<!-- page 9 -->
Table 1. Quantitative comparison with state-of-the-art methods on GSO [19] and OmniObject3D [69].
Method
Venue
GSO dataset [19]
OmniObject3D dataset [69]
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
SyncDreamer [41]
ICLR 2024
18.11
0.8081
0.1768
16.80
0.7992
0.17821
Wonder3D [42]
CVPR 2024
17.22
0.7781
0.2146
14.63
0.7715
0.2245
DreamGaussian [57]
ICLR 2024
20.05
0.8048
0.1764
18.66
0.8072
0.1768
LRM [23]
ICLR 2024
18.10
0.7802
0.1847
17.26
0.8021
0.1674
LGM [56]
ECCV 2024
14.78
0.7377
0.2795
13.31
0.7403
0.2738
VideoMV [86]
Alibaba 2024
21.06
0.8364
0.1846
18.75
0.8243
0.1898
InstantMesh [72]
Tencent 2024
18.27
0.8066
0.1677
16.82
0.8139
0.1645
SAR3D [13]
CVPR 2025
17.01
0.8006
0.2104
15.92
0.8090
0.1937
ERGO
Ours
21.37
0.8426
0.1609
20.24
0.8854
0.1422
Input Image
ERGO (Ours)
LRM
LGM
InstantMesh
SAR3D
SyncDreamer
DreamGaussian
VideoMV
Wonder3D
Figure 7. Qualitative comparison on the GSO dataset [19] and the OmniObject3D dataset [69].
generation quality, three widely used metrics, including
PSNR [26], SSIM [65], and LPIPS [80], are used for evalu-
ation from the aspects of texture quality, geometry quality,
and perceptual quality, respectively.

<!-- page 10 -->
Input Image
ERGO (Ours)
LRM
LGM
InstantMesh
SAR3D
SyncDreamer
DreamGaussian
VideoMV
Wonder3D
Figure 8. Qualitative comparisons on in-the-wild images.
Table 2. Ablation study on the proposed components. GAO, TAO, and AW represent geometry-aware objective, texture-aware objective,
and adaptive weights, respectively.
Method
GAO
TAO
AW
|azimuth| ≤90◦
|azimuth| > 90◦
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Baseline
✗
✗
✗
21.97
0.8541
0.1458
18.79
0.8214
0.1838
Baseline + GAO
✓
✗
✗
22.54
0.8613
0.1287
19.32
0.8261
0.1689
Baseline + TAO
✗
✓
✗
22.12
0.8582
0.1422
19.00
0.8258
0.1822
ERGO w/o AW
✓
✓
✗
22.68
0.8690
0.1282
19.42
0.8327
0.1616
ERGO
✓
✓
✓
22.94
0.8813
0.1220
19.62
0.8397
0.1590
5.2. Comparison with State-of-the-art Methods
Quantitative comparisons. The quantitative performance
of ERGO against the eight selected state-of-the-art methods
on GSO and OminiObject3D is reported in Table 1. ERGO
consistently achieves the best performance across all three
evaluation metrics on both datasets. Particularly, our im-
plementation of 3DGS is built upon DreamGaussian [57],
and the performance gains of ERGO are significant, e.g.,
the SSIM values of ERGO on GSO and OminiObject3D
are 0.8426 and 0.8854, while those of DreamGaussian are
0.8048 and 0.8072. Furthermore, ERGO obtains the highest
PSNR and the lowest LPIPS, indicating that its generated
3D content has higher texture quality and cross-view con-
sistency. Among the competing methods, VideoMV [86]

<!-- page 11 -->
Input Image
ERGO
Baseline
Baseline + GAO
Baseline + TAO
ERGO w/o AW
Figure 9. Visual examples of the ablation study on the three key components of ERGO, namely, the geometry-aware objective (GAO), the
texture-aware objective (TAO), and the adaptive weighting mechanism (AW).
achieves the closest performance to ERGO, which we at-
tribute to its exploitation of video priors that inherently en-
code temporal smoothness as a proxy for multi-view geo-
metric and textural consistency. Nevertheless, ERGO sur-
passes VideoMV by employing the geometry-aware objec-
tive and the texture-aware objective with adaptive weights,
thereby achieving more robust optimization under inconsis-
tent multi-view supervision.
Qualitative comparisons on GSO and OminiObject3D.
Figure 7 shows the qualitative results of ERGO and the
baselines. One can see that SyncDreamer [41] tends to gen-
erate smooth geometry but often lacks finer details. While
Wonder3D [42] captures more detailed textures, it some-
times produces unrealistic geometry due to inconsistencies
in sparse multi-view images. DreamGaussian [57] can gen-
erate 3D models that match the reference image, but it tends
to produce blurry textures in unseen areas, as it relies solely
on SDS optimization to supplement novel view informa-
tion. As for feed-forward large models, LRM [23] is prone
to generating grid artifacts. LGM [56], which integrates
the multi-view diffusion model ImageDream [64], is lim-
ited by the quality of the generated multi-views and often
produces artifacts in unseen areas. Similar artifacts also ex-
hibit in the results of other methods like VideoMV [86], In-
stanceMesh [72], and SAR3D [13]. In contrast, our ERGO
framework generates high-quality 3D Gaussian models that
not only maintain consistency across various camera poses
but also excel in producing realistic textures with richer de-
tails, demonstrating its superiority over the baseline meth-
ods.
Qualitative comparisons on in-the-wild images. To eval-
uate the generalization ability of ERGO, we test ERGO on
diverse in-the-wild images, and the results are shown in Fig-
ure 8. ERGO demonstrates robust performance on objects
with complex geometries and intricate textures, effectively
mitigating artifacts observed in existing methods, such as
over-smoothing textures (e.g., the grids of the house in the
second column), cross-view inconsistency (the red hats in
the first column), and distorted geometric structures (the
wheels in the last column). These results validate that the
adaptive weighting mechanism of ERGO successfully sup-
presses supervision noise while preserving high-frequency
details and geometric integrity in challenging real-world
scenarios.
5.3. Ablation Studies
Effects of the Proposed Components. We conduct an ab-
lation study to thoroughly evaluate the efficacy of the pro-
posed components in ERGO. In this experiment, 3D items
are randomly sampled from the GSO dataset [19]. Start-
ing with an azimuth of 0 degrees, we render 16 views uni-
formly across the range of [-180, 180] degrees. We con-
struct three variants of ERGO, including the baseline with
the geometry-aware objective, the baseline with the texture-
aware objective, and ERGO without adaptive weights, to
evaluate the three key components of ERGO. From the re-
sults in Table 2, we can see that these three components
improve the performance of the baseline consistently. The
performance gains of the baseline with the geometry-aware
objective and those with the texture-aware objective are
similar, indicating that both geometry and texture guidance
are important for the image-to-3D task. The results of the
full framework against those of the variant without adaptive
weightings validate that exploiting excess risk to modulate
the optimization process yields better performance. Figure
9 compares the results of ERGO and its variants. These vi-
sual examples provide an intuitive demonstration of the ef-
ficacy of the proposed components. For instance, in the first
row, we can see that the baseline generates blurry floaters
around its boundaries. These floaters are removed signifi-
cantly by the geometry-aware objective, while they remain
but are sharper with the texture-aware objective. Finally,
unifying these components into the ERGO framework ob-
tains the most balanced and compelling results.
Effects of View Variations.
One of our assumptions is
that reconstruction quality degrades as the angular separa-
tion between a rendering view and the input reference view
increases. To validate this, we partition evaluation views

<!-- page 12 -->
Input Image
Geometry Inconsistency
Refined Views
Refined Views
3D Gaussians with Artifacts
Input Image
Texture Inconsistency
3D Gaussians with Artifacts
High-Quality 3D Gaussians
High-Quality 3D Gaussians
Figure 10. Visual examples of artifacts produced by performing 3DGS on inconsistent multi-view images (labeled in red). These artifacts
are mitigated by ERGO (labeled in green).
into two groups based on absolute azimuth deviation, i.e.,
|azimuth| ≤90 and |azimuth| > 90. The experimental
results are reported in Table 2.
These results confirm a
strong correlation between angular proximity to the refer-
ence view and reconstruction fidelity. The performance of
both ERGO and the baseline under |azimuth| ≤90 is con-
sistently superior to that under |azimuth| ≤90. Neverthe-
less, ERGO obtains considerable performance gains in both
groups, validating that our adaptive weighting mechanism
effectively suppresses unreliable supervision signals from
geometrically distant views.
Effects of Multi-view Inconsistency. In addition to the
above visual examples in the ablation study of the proposed
components, we demonstrate the effects of using multi-view
images with geometry and texture inconsistency for 3D cre-
ation in Figure 10. We can see that employing 3DGS di-
rectly on inconsistent images results in blurry and erroneous
textures. These artifacts are mitigated through two mecha-
nisms in ERGO. First, the geometry correction step that in-
verts renderings from 3D Gaussians back to the base MVD
model refines multi-view textures, e.g, the bottom of the
lamp in the refined views in Figure 10. Second, the adaptive
weighted objectives of ERGO further enhance the geome-
try structures and textures of the created 3D objects, as the
effects of inconsistent image regions are suppressed.
6. Conclusion
In this paper, we introduce ERGO, an excess-risk-guided
optimization framework for single-image 3D reconstruc-
tion. ERGO constructs a global-local adaptive paradigm
to tackle the critical geometric and textural inconsistency
issues and enable seamless integration of multi-view dif-
fusion models with optimization-based methods. From the
global perspective, an excess-risk-derived objective weight-
ing mechanism is introduced, while from the local perspec-
tive, a geometry-aware objective and a texture-aware ob-
jective are designed to achieve regional modulation. Ex-
tensive experiments on two public datasets as well as in-
the-wild images validate that ERGO effectively mitigates
spurious supervision signals and texture blurring artifacts,
significantly enhancing the cross-view consistency and fine-
grained fidelity of the reconstructed 3D content.
References
[1] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Sak-
sham Consul, Cyril Diagne, Tim Dockhorn, Jack English,
Zion English, Patrick Esser, Sumith Kulal, et al.
Flux. 1
kontext: Flow matching for in-context image generation and
editing in latent space. arXiv e-prints, 2025.
[2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel
Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi,
Zion English, Vikram Voleti, Adam Letts, et al. Stable video
diffusion: Scaling latent video diffusion models to large
datasets. arXiv preprint arXiv:2311.15127, 2023.
[3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dock-
horn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis.
Align your latents: High-resolution video synthesis with la-
tent diffusion models. In Proceedings of the IEEE Confer-
ence on Computer Vision and Pattern Recognition, pages
22563–22575, 2023.
[4] Yuanhao Cai, He Zhang, Kai Zhang, Yixun Liang, Mengwei
Ren, Fujun Luan, Qing Liu, Soo Ye Kim, Jianming Zhang,
Zhifei Zhang, et al.
Baking gaussian splatting into diffu-
sion denoiser for fast and scalable single-stage image-to-3d
generation and reconstruction. In Proceedings of the IEEE
International Conference on Computer Vision, pages 25062–
25072, 2025.
[5] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stable-
video: Text-driven consistency-aware diffusion video edit-
ing. In Proceedings of the IEEE International Conference
on Computer Vision, pages 23040–23050, 2023.
[6] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. Pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In Pro-

<!-- page 13 -->
ceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 19457–19467, 2024.
[7] Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen
Tu, Lingjie Liu, and Hao Su. Single-stage diffusion nerf: A
unified approach to 3d generation and reconstruction. In Pro-
ceedings of the IEEE International Conference on Computer
Vision, pages 2416–2425, 2023.
[8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze
Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo,
Huchuan Lu, et al.
Pixart-\α: Fast training of diffusion
transformer for photorealistic text-to-image synthesis.
In-
ternational Conference on Learning Representations, 2023.
[9] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fan-
tasia3d: Disentangling geometry and appearance for high-
quality text-to-3d content creation.
In Proceedings of the
IEEE International Conference on Computer Vision, pages
22246–22256, 2023.
[10] Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang,
Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo,
Thibaut Hardin, Xiang Li, et al. Sam 3d: 3dfy anything in
images. arXiv preprint arXiv:2511.16624, 2025.
[11] Yang Chen, Yingwei Pan, Yehao Li, Ting Yao, and Tao Mei.
Control3d: Towards controllable text-to-3d generation. In
Proceedings of the ACM International Conference on Multi-
media, pages 1148–1156, 2023.
[12] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In Proceedings of the European Confer-
ence on Computer Vision, pages 370–386, 2024.
[13] Yongwei Chen, Yushi Lan, Shangchen Zhou, Tengfei Wang,
and Xingang Pan. Sar3d: Autoregressive 3d object genera-
tion and understanding via multi-scale 3d vqvae. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, 2025.
[14] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexan-
der G Schwing, and Liang-Yan Gui. Sdfusion: Multimodal
3d shape completion, reconstruction, and generation. In Pro-
ceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 4456–4465, 2023.
[15] Silvia Elena Corchs, Gianluigi Ciocca, Emanuela Bricolo,
and Francesca Gasparini. Predicting complexity perception
of real world images. PloS one, 11(6):e0157986, 2016.
[16] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo,
Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte,
Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Anirud-
dha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana
Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A
universe of 10m+ 3d objects. In Advances in Neural Infor-
mation Processing Systems, pages 35799–35813, 2023.
[17] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs,
Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana
Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse:
A universe of annotated 3d objects. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion, pages 13142–13153, 2023.
[18] Zijun Deng, Xiangteng He, Yuxin Peng, Xiongwei Zhu, and
Lele Cheng. Mv-diffusion: Motion-aware video diffusion
model. In Proceedings of the ACM International Conference
on Multimedia, pages 7255–7263, 2023.
[19] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kin-
man, Ryan Hickman, Krista Reymann, Thomas B McHugh,
and Vincent Vanhoucke. Google scanned objects: A high-
quality dataset of 3d scanned household items. In Interna-
tional Conference on Robotics and Automation, pages 2553–
2560, 2022.
[20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim
Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified
flow transformers for high-resolution image synthesis. 2024.
[21] Rafael C Gonzalez. Digital image processing. 2009.
[22] Yifei He, Shiji Zhou, Guojun Zhang, Hyokun Yun, Yi Xu,
Belinda Zeng, Trishul Chilimbi, and Han Zhao.
Robust
multi-task learning with excess risks. pages 18094–18114,
2024.
[23] Zexin He and Tengfei Wang. Openlrm: Open-source large
reconstruction models, 2023.
[24] Lukas H¨ollein, Aljaˇz Boˇziˇc, Norman M¨uller, David Novotny,
Hung-Yu Tseng, Christian Richardt, Michael Zollh¨ofer, and
Matthias Nießner. Viewdiff: 3d-consistent image generation
with text-to-image models. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
5043–5052, 2024.
[25] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou,
Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao
Tan. Lrm: Large reconstruction model for single image to
3d. In International Conference on Learning Representa-
tions, 2024.
[26] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs.
ssim. In International Conference on Pattern Recognition,
pages 2366–2369, 2010.
[27] Xin Huang, Ruizhi Shao, Qi Zhang, Hongwen Zhang, Ying
Feng, Yebin Liu, and Qing Wang. Humannorm: Learning
normal diffusion model for high-quality and realistic 3d hu-
man generation.
In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 4568–
4577, 2024.
[28] Zehuan Huang, Hao Wen, Junting Dong, Yaohui Wang,
Yangguang Li, Xinyuan Chen, Yan-Pei Cao, Ding Liang, Yu
Qiao, Bo Dai, et al. Epidiff: Enhancing multi-view synthe-
sis via localized epipolar-constrained diffusion. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 9784–9794, 2024.
[29] Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi,
Lizhuang Ma, Yan-Pei Cao, and Lu Sheng.
Mv-adapter:
Multi-view consistent image generation made easy. In Pro-
ceedings of the IEEE International Conference on Computer
Vision, pages 16377–16387, 2025.
[30] Heewoo Jun and Alex Nichol.
Shap-e:
Generat-
ing conditional 3d implicit functions.
arXiv preprint
arXiv:2305.02463, 2023.
[31] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani
Lischinski. Noise-free score distillation. In International
Conference on Learning Representations, 2024.

<!-- page 14 -->
[32] Wei Ke, Tianliang Zhang, Jie Chen, Fang Wan, Qixiang Ye,
and Zhenjun Han. Texture complexity based redundant re-
gions ranking for object proposal.
In Proceedings of the
IEEE conference on computer vision and pattern recognition
workshops, pages 10–18, 2016.
[33] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4):139–1, 2023.
[34] Diederik Kingma and Jimmy Ba.
Adam: A method for
stochastic optimization.
In International Conference on
Learning Representations, 2015.
[35] Jeong-gi Kwak, Erqun Dong, Yuhe Jin, Hanseok Ko, Shweta
Mahajan, and Kwang Moo Yi.
Vivid-1-to-3: Novel view
synthesis with video diffusion models. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion, pages 6775–6785, 2024.
[36] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun
Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg
Shakhnarovich, and Sai Bi.
Instant3d:
Fast text-to-3d
with sparse-view generation and large reconstruction model.
In International Conference on Learning Representations,
2024.
[37] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen,
Ping Tan, and Xiaoxiao Long.
Craftsman: High-fidelity
mesh generation with 3d native generation and interactive
geometry refiner. arXiv preprint arXiv:2405.14979, 2024.
[38] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa,
Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler,
Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution
text-to-3d content creation. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
300–309, 2023.
[39] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang,
Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Re-
conx: Reconstruct any scene from sparse views with video
diffusion model. arXiv preprint arXiv:2408.16767, 2024.
[40] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tok-
makov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3:
Zero-shot one image to 3d object.
In Proceedings of the
IEEE International Conference on Computer Vision, pages
9298–9309, 2023.
[41] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie
Liu, Taku Komura, and Wenping Wang. Syncdreamer: Gen-
erating multiview-consistent images from a single-view im-
age. In International Conference on Learning Representa-
tions, 2024.
[42] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu,
Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang,
Marc Habermann, Christian Theobalt, et al. Wonder3d: Sin-
gle image to 3d using cross-domain diffusion. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 9970–9980, 2024.
[43] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, and
Andrea Vedaldi. Realfusion: 360deg reconstruction of any
object from a single image. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
8446–8455, 2023.
[44] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and
Daniel Cohen-Or. Null-text inversion for editing real images
using guided diffusion models, 2023.
[45] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela
Mishkin, and Mark Chen. Point-e: A system for generat-
ing 3d point clouds from complex prompts. arXiv preprint
arXiv:2212.08751, 2022.
[46] Luca Oneto, Sandro Ridella, and Davide Anguita. Informed
machine learning: Excess risk and generalization. Neuro-
computing, page 130521, 2025.
[47] Hanchuan Peng, Zongcai Ruan, Fuhui Long, Julie H Simp-
son, and Eugene W Myers. V3d enables real-time 3d visual-
ization and quantitative analysis of large-scale biological im-
age data sets. Nature biotechnology, 28(4):348–353, 2010.
[48] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall. Dreamfusion: Text-to-3d using 2d diffusion. In Inter-
national Conference on Learning Representations, 2023.
[49] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren,
Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Sko-
rokhodov, Peter Wonka, Sergey Tulyakov, and Bernard
Ghanem. Magic123: One image to high-quality 3d object
generation using both 2d and 3d diffusion priors. In Interna-
tional Conference on Learning Representations, 2024.
[50] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and
Sanja Fidler. Deep marching tetrahedra: a hybrid represen-
tation for high-resolution 3d shape synthesis. In Advances in
Neural Information Processing Systems, pages 6087–6101,
2021.
[51] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu,
Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao
Su. Zero123++: a single image to consistent multi-view dif-
fusion base model. arXiv preprint arXiv:2310.15110, 2023.
[52] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li,
and Xiao Yang. Mvdream: Multi-view diffusion for 3d gen-
eration. International Conference on Learning Representa-
tions, 2024.
[53] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen
Liu, Zhenda Xie, and Yebin Liu. Dreamcraft3d: Hierarchical
3d generation with bootstrapped diffusion prior. In Interna-
tional Conference on Learning Representations, 2024.
[54] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea
Vedaldi.
Splatter image: Ultra-fast single-view 3d recon-
struction. In Proceedings of the IEEE Conference on Com-
puter Vision and Pattern Recognition, pages 10208–10217,
2024.
[55] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi,
Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d
creation from a single image with diffusion prior. In Pro-
ceedings of the IEEE International Conference on Computer
Vision, pages 22819–22829, 2023.
[56] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang,
Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian
model for high-resolution 3d content creation. In European
Conference on Computer Vision, pages 1–18, 2024.
[57] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang
Zeng. Dreamgaussian: Generative gaussian splatting for ef-
ficient 3d content creation. In International Conference on
Learning Representations, 2024.

<!-- page 15 -->
[58] Shitao Tang, Jiacheng Chen, Dilin Wang, Chengzhou Tang,
Fuyang Zhang, Yuchen Fan, Vikas Chandra, Yasutaka Fu-
rukawa, and Rakesh Ranjan. Mvdiffusion++: A dense high-
resolution multi-view diffusion model for single or sparse-
view 3d object reconstruction. In Proceedings of the Euro-
pean Conference on Computer Vision, pages 175–191, 2024.
[59] Jiaye Teng, Jianhao Ma, and Yang Yuan. Towards under-
standing generalization via decomposing excess risk dynam-
ics. In International Conference on Learning Representa-
tions, 2022.
[60] Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany,
Sanja Fidler, Karsten Kreis, et al. Lion: Latent point diffu-
sion models for 3d shape generation. In Advances in Neural
Information Processing Systems, pages 10021–10039, 2022.
[61] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts,
David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin
Rombach, and Varun Jampani. Sv3d: Novel multi-view syn-
thesis and 3d generation from a single image using latent
video diffusion. In Proceedings of the European Conference
on Computer Vision, pages 439–457, 2024.
[62] Chen Wang, Jiatao Gu, Xiaoxiao Long, Yuan Liu, and
Lingjie Liu. Geco: Fast generative image-to-3d within one
second. IEEE Transactions on Visualization and Computer
Graphics, 2025.
[63] Chen Wang, Hao-Yang Peng, Ying-Tian Liu, Jiatao Gu, and
Shi-Min Hu. Diffusion models for 3d generation: A survey.
Computational Visual Media, 11(1):1–28, 2025.
[64] Peng Wang and Yichun Shi. Imagedream: Image-prompt
multi-view diffusion for 3d generation, 2023.
[65] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: From error visibility to
structural similarity. IEEE Transactions on Image Process-
ing, 13(4):600–612, 2004.
[66] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan
Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and
diverse text-to-3d generation with variational score distilla-
tion. Advances in Neural Information Processing Systems,
36:8406–8441, 2023.
[67] Hao Wen, Zehuan Huang, Yaohui Wang, Xinyuan Chen, and
Lu Sheng.
Ouroboros3d: Image-to-3d generation via 3d-
aware recursive diffusion. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
21631–21641, 2025.
[68] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi
Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scal-
able image-to-3d generation via 3d latent diffusion trans-
former. In Advances in Neural Information Processing Sys-
tems, pages 121859–121881, 2024.
[69] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren,
Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian,
et al. Omniobject3d: Large-vocabulary 3d object dataset for
realistic perception, reconstruction and generation. In Pro-
ceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 803–814, 2023.
[70] Tong Wu, Yu-Jie Yuan, Ling-Xiao Zhang, Jie Yang, Yan-
Pei Cao, Ling-Qi Yan, and Lin Gao. Recent advances in 3d
gaussian splatting. Computational Visual Media, 10(4):613–
642, 2024.
[71] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang,
and Zhangyang Wang. Neurallift-360: Lifting an in-the-wild
2d photo to a 3d object with 360deg views.
In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 4479–4489, 2023.
[72] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang,
Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d
mesh generation from a single image with sparse-view large
reconstruction models.
arXiv preprint arXiv:2404.07191,
2024.
[73] Yuxuan Xue, Xianghui Xie, Riccardo Marin, and Gerard
Pons-Moll. Gen-3diffusion: Realistic image-to-3d genera-
tion via 2d & 3d diffusion synergy. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 2025.
[74] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi
Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Gaussianob-
ject: Just taking four images to get a high-quality 3d object
with gaussian splatting. arXiv preprint arXiv:2402.10259,
2024.
[75] Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng
Chen, and Tao Mei. 3dstyle-diffusion: Pursuing fine-grained
text-driven 3d stylization with 2d diffusion models. In Pro-
ceedings of the ACM International Conference on Multime-
dia, pages 6860–6868, 2023.
[76] Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng
Chen, Chong-Wah Ngo, and Tao Mei. Hi3d: Pursuing high-
resolution image-to-3d generation with video diffusion mod-
els. In Proceedings of the ACM International Conference on
Multimedia, pages 6870–6879, 2024.
[77] Xianghui Yang, Yan Zuo, Sameera Ramasinghe, Loris Baz-
zani, Gil Avraham, and Anton van den Hengel. Viewfusion:
Towards multi-view consistency via interpolated denoising.
In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 9870–9880, 2024.
[78] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter
Wonka. 3dshape2vecset: A 3d shape representation for neu-
ral fields and generative diffusion models. ACM Transactions
on Graphics, 42(4):1–16, 2023.
[79] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu,
Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu.
Clay: A controllable large-scale generative model for creat-
ing high-quality 3d assets. ACM Transactions on Graphics,
43(4):1–20, 2024.
[80] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In 2018 IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 586–595, 2018.
[81] Xuying Zhang, Yupeng Zhou, Kai Wang, Yikai Wang, Zhen
Li, Shaohui Jiao, Daquan Zhou, Qibin Hou, and Ming-Ming
Cheng. Ar-1-to-3: Single image to consistent 3d object via
next-view prediction. In Proceedings of the IEEE Interna-
tional Conference on Computer Vision, pages 26273–26283,
2025.
[82] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang,
Pei Cheng, BIN FU, Tao Chen, Gang Yu, and Shenghua Gao.
Michelangelo: Conditional 3d shape generation based on

<!-- page 16 -->
shape-image-text aligned latent representation. In Advances
in Neural Information Processing Systems, pages 73969–
73982, 2023.
[83] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao,
Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng
Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffu-
sion models for high resolution textured 3d assets generation.
arXiv preprint arXiv:2501.12202, 2025.
[84] Wenyang Zhou, Lu Yuan, and Taijiang Mu. Multi3d: 3d-
aware multimodal image synthesis. Computational Visual
Media, 10(6):1205–1217, 2024.
[85] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li,
Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane
meets gaussian splatting: Fast and generalizable single-view
3d reconstruction with transformers. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion, pages 10324–10335, 2024.
[86] Qi Zuo, Xiaodong Gu, Lingteng Qiu, Yuan Dong, Zhengyi
Zhao, Weihao Yuan, Rui Peng, Siyu Zhu, Zilong Dong,
Liefeng Bo, et al. Videomv: Consistent multi-view gener-
ation based on large video generative model. arXiv preprint
arXiv:2403.12010, 2024.
