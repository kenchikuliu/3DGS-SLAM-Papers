<!-- page 1 -->
Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction
Reconstruction
Hao Tian1,4, Chenyangguang Zhang2, Rui Liu3, Wen Shen1,4, Xiaolin Qin1,4*
1Chengdu Institute of Computer Applications, Chinese Academy of Sciences
2Tsinghua University
3Minzu University of China
4University of Chinese Academy of Sciences
Abstract
This paper focuses on a challenging setting of simultane-
ously modeling geometry and appearance of hand-object
interaction scenes without any object priors.
We follow
the trend of dynamic 3D Gaussian Splatting based meth-
ods, and address several significant challenges. To model
complex hand-object interaction with mutual occlusion and
edge blur, we present interaction-aware hand-object Gaus-
sians with newly introduced optimizable parameters aiming
to adopt piecewise linear hypothesis for clearer structural
representation. Moreover, considering the complementarity
and tightness of hand shape and object shape during inter-
action dynamics, we incorporate hand information into ob-
ject deformation field, constructing interaction-aware dy-
namic fields to model flexible motions. To further address
difficulties in the optimization process, we propose a pro-
gressive strategy that handles dynamic regions and static
background step by step. Correspondingly, explicit regu-
larizations are designed to stabilize the hand-object rep-
resentations for smooth motion transition, physical inter-
action reality, and coherent lighting.
Experiments show
that our approach surpasses existing dynamic 3D-GS-based
methods and achieves state-of-the-art performance in re-
constructing dynamic hand-object interaction.
1. Introduction
Accurate hand–object interaction (HOI) reconstruction is
vital for VR and robotics [8], requiring precise shape mod-
eling and interaction capture. Despite the apparent simplic-
ity of daily actions like grasping or drinking, they involve
complex contact dynamics and severe occlusions that re-
main challenging to model.
Previous works [32, 36] reconstruct HOI scenes by treat-
ing interactive objects as known, relying on specific object
poses or templates. However, acquiring such poses or tem-
plates is costly, limiting their industrial applicability. Exist-
*Corresponding authors.
Fig. 1.
Differences between traditional 3D Gaussian-based
hand-object reconstruction and our interaction-aware model-
ing. Conventional 3D Gaussian approaches model the entire HOI
scene with a single, unified implicit field and rely primarily on
2D supervision. This design often leads to geometric ambiguities
during close interactions—such as collapsed clearances, blurred
contact boundaries, and non-physical merging of hand and ob-
ject surfaces (top). In contrast, our method explicitly decouples
hand and object representations into separate fields, introduces
interaction-aware parameters (w, o) to modulate occlusion and
edge sharpness, and leverages interaction-aware losses to preserve
fine-grained spatial relationships, enabling accurate and disentan-
gled dynamic reconstruction (bottom).
ing methods try to reduce the reliance on precise pose es-
timation and templates by several routes. [30, 40] employs
SDF-based approaches, integrating SDF to reconstruct dy-
namic hand-object scenes using neural networks, without
the need for specific object priors. However, these meth-
ods focus primarily on geometry reconstruction without ap-
pearance. With the advent of NeRF [22], some researchers
[18, 48] explore HOI scene reconstruction with both ge-
ometry and appearance by training implicit fields. How-
ever, due to the inefficiency of backward mapping-based
ray-rendering algorithms [22], these methods require sig-
nificant time and computational resources. Recently, 3D
Gaussian Splatting (3D-GS) [13] has demonstrated supe-
rior fidelity and speed in static scene reconstruction. Some
works [11, 41, 45] attempt dynamic reconstruction using
3D-GS, but they struggle with complex HOI scenarios in-
volving heavy occlusion and irregular rotations, failing to
1
arXiv:2511.14540v1  [cs.CV]  18 Nov 2025

<!-- page 2 -->
capture accurate interaction dynamics. Although EgoGaus-
sian [47] targets HOI reconstruction, it requires object pose
estimation and only presents results for interactive objects
without effective hand representation. In this work, we ad-
dress the limitations of 3D-GS-based methods by proposing
a model to simultaneously reconstruct the entire HOI scene
without requiring any object priors.
To successfully handle such a practical setting presents
significant challenges. First, drastic motions, mutual oc-
clusion and blur during interaction cause misalignment
and excessive overlap among Gaussians. To address this,
we model the interaction as a piecewise linear process
and present a novel representation termed interaction-aware
hand-object Gaussians. It introduces two parameters over
the traditional 3D-GS representation: weight w and radius
o. The weight w balances motion smoothness and noise re-
duction, with smaller values indicating weak structural in-
formation or occlusion. The radius o controls edge sharp-
ness, where smaller values produce clearer contours. The
combination of w and o effectively models the complex dy-
namic interaction, reducing blurring at interaction bound-
aries and enhancing visual quality. Second, previous meth-
ods [11, 45] use a single field to model Gaussian trans-
formations, which is insufficient for capturing drastic and
highly localized motions in HOI scenes, often leading to
significant loss of fine-grained motion details. On the other
hand, simply using separate fields for hand and object de-
formation overlooks their mutual interaction and geomet-
ric coupling during contact. To address this, we incorpo-
rate key-frame hand positions into the object field, enabling
interaction-aware transformations that accurately capture
dynamic changes caused by hand grasping. Third, consid-
ering the flexible motions, irregular rotations, and frequent
occlusions in HOI scenes, it is difficult to directly utilize
traditional 3D-GS optimization [13, 41, 45] to achieve de-
cent rendering quality. To address this, we design explicit
interaction-aware regularizations to explicitly stabilize the
position and rotation of hand-object Gaussians. Further-
more, we propose a progressive optimization mechanism to
achieve physically realistic hand-object interaction, ensure
smooth edge transitions, and enhance coherence in complex
HOI scenes.
Our contributions are summarized as follows:
• We propose a novel interaction-aware hand-object Gaus-
sian representation to model HOI scenes without any ob-
ject priors, effectively addressing mutual occlusion and
edge blur during interactions.
• To accurately model interaction changes on the hand-held
object, we incorporate hand information to enhance the
object field to represent relevant deformation.
• We employ a progressive optimization strategy with ex-
plicit 3D losses to benefit the fitting of the interaction-
aware Gaussians during dynamic reconstruction.
• Experiments show that our approach surpasses state-of-
the-art baselines [5, 11, 26, 41, 45], achieving superior
performance in reconstructing dynamic HOI scenes.
2. Related Works
Hand Representation. Early approaches [24, 25] focused
on estimating 2D or 3D keypoints from images. The intro-
duction of statistical hand models like MANO [31] revo-
lutionized parametric hand representation by jointly encod-
ing pose, shape, and 3D vertices. Recent method [1] typ-
ically employs regression networks to predict MANO pa-
rameters directly from images and optimize shape param-
eters for alignment. However, these methods suffer from
error propagation: initial MANO inaccuracies accumulate
downstream, causing cascading reconstruction errors. To
mitigate this, we propose a hand field to decouple hand de-
formation from strict MANO parameter dependencies.
Hand-Object Reconstruction.
Reconstructing hand-
object interaction from video remains a significant chal-
lenge in computer vision and graphics. Previous works fall
into two categories. The first [3, 7, 23, 38] reconstructs
hand-object interactions from multiview sources by fitting
objects into 2D images using 3D object priors. However,
these methods heavily rely on accurate 3D priors, which are
costly to obtain. The second stream [4, 10, 32] pre-learns
object templates to reduce reliance on priors. For exam-
ple, the MANO model [31] represents the canonical hand
space, with linear blending skinning driving the hand tem-
plate. EgoGaussian [47] reconstructs egocentric interaction
scenes using 3D Gaussian [13] by separating dynamic ob-
jects from the static background. However, it is sensitive
to object pose, probably failing under inaccurate poses, and
excludes interacting hands from reconstruction. As a result,
it misses the complete hand–object interaction context. Re-
cent work BIGS [26] reconstructs hand-object interactions
from monocular video using 3D-GS and a diffusion prior,
but assumes a known object mesh and excludes the back-
ground. In contrast, our method is category-agnostic, re-
quires no object priors, and reconstructs the full HOI scene.
Dynamic Scene Reconstruction. With the advent of
NeRF [22], many works [6, 16, 27–29, 37, 44] use MLPs to
represent implicit spaces as deformation fields with tempo-
ral information. However, their extensive training time lim-
its practical applicability. 3D Gaussian Splatting (3D-GS)
[13] emerges as a promising alternative for scene recon-
struction. Methods like [11, 12, 14, 15, 17, 19, 21, 33, 34,
41–43, 45, 50] explore dynamic reconstruction using 3D-
GS. For instance, [45] uses MLPs to learn Gaussian position
offsets per timestamp, which increases training time. [11]
introduces sparse control points to deform Gaussians, but
in hand-object interaction (HOI) scenes, redundant points
lead to inaccuracies and image tearing, failing to capture
intricate interactions.
These methods [11, 41, 45] input
2

<!-- page 3 -->
all Gaussians into a single MLP, which struggles to accu-
rately model complex interactive motions.
To overcome
significant challenges posed by HOI scenes, our method
introduces a novel interaction-aware hand-object Gaussian
representation, with adaptive losses and a progressive opti-
mization strategy.
3. Method
Our goal is to reconstruct dynamic hand-object interac-
tion (HOI) scenes from RGB egocentric videos at arbi-
trary timestamps without relying on any object shape pri-
ors.
We utilize three implicit fields to model the dy-
namic HOI scenes: the hand field FH and object field FO
to approximate the shape of the dynamic HOI region, as
well as the background field FBG to create a clean back-
ground and facilitate subsequent joint optimization. Sep-
arate modeling allows capturing clear hand-object appear-
ance and stable background scene in drastically changing
dynamic scenarios. First, by treating hand and object mod-
eling differently, significant occlusions could be solved via
more detailed supervision.
Second, backgrounds require
low-frequency updates, while hand-object interactions de-
mand high-frequency modeling.
Meanwhile, collaborat-
ing with such dynamic implicit fields, we adaptively im-
prove the Gaussian Splatting representation for HOI sce-
narios, addressing occlusion and contour clarity issues dur-
ing the interaction.
In optimization, we utilize only the
hand’s MANO parameters predicted by an off-the-shelf
hand tracker [46], which incurs negligible computational
overhead compared to Gaussian optimization (typically <
3% of total runtime). These pose estimates provide coarse
3D hand guidance to accelerate convergence without requir-
ing any object priors. To ensure physically plausible inter-
actions, we further introduce an interaction loss. A pro-
gressive and collaborative optimization framework is then
devised to achieve high-quality HOI scene reconstruction
through this lightweight 3D supervision and our interaction-
aware representation.
3.1. Preliminaries: Deformable Gaussian Splatting
3D Gaussian Splatting (3D-GS) [13] represents 3D scene
features using five parameters:
position, transparency,
spherical harmonic coefficients, rotation, and scaling. 3D-
GS explicitly defines each 3D Gaussian ellipsoid in space
using a covariance matrix Σ and the position vector ρ, as
shown in the following equation:
G(x) =
1
(2π)
3
2 |Σ|
1
2 exp(−1
2(x −ρ)⊤Σ−1(x −ρ)), (1)
here the Σ matrix can be decomposed into a rotation R
and a scaling S by Σ = RSS⊤R⊤.
Recent works
[11, 41, 45] combine 3D-GS with deformation fields for dy-
namic scenes, using an MLP to warp points from canonical
to target space:
Fθ(x, t) = (δxt, δrt, δst),
xt = x + δxt.
(2)
Each initialized 3D Gaussian’s center x is input to deforma-
tion field Fθ, which outputs time-dependent offsets (δxt,
δrt, δst) to wrap canonical Gaussians to target space.
3.2. Interaction-Aware Hand-Object Gaussians
To effectively capture the complex spatiotemporal motion
in the Hand-Object Interaction (HOI) scene, we propose to
decompose the dynamic HOI scene into three sets of Gaus-
sians and model each part individually. Moreover, we im-
prove traditional 4D Gaussian representations to overcome
the issues representing complex HOI motions in dynamic
scenarios. Traditional 4D Gaussians [11, 41, 45] tend to ne-
glect mutual influences between different interacted Gaus-
sians. Moreover, it is hard to depict contour edges in inter-
action process, leading to texture drift and edge blur. In-
spired by [11], we introduce two additional learnable pa-
rameters weight w ∈R+ and radius o ∈R+, forming
a novel representation termed the Interaction-Aware Hand-
Object Gaussian GHO. GHO focuses on interaction-aware
modeling via: (1) Weight w smooths the motion and re-
duces noise, it models mutual occlusion during interactions,
where a small weight w indicates weak structural infor-
mation and occlusion; (2) Radius o captures edge details,
where a small radius o corresponds to sharper geometric
contours near edges; (3) Since w is larger near the current
Gaussian and smaller farther from the edge, the combina-
tion of weight w and radius o effectively handles edge blur-
ring between hand-object interactions and the background.
GHO is expressed as follows:
GHO = {xiyizi, Ri, T i, Si, αi, ci, wi, oi} .
(3)
Due to different characteristics of hand motions, object mo-
tions and background scenes, we separately introduce mod-
eling of the dynamics of each component below.
Hand Gaussians. Hand Gaussians GH has the same opti-
mizable parameters with GHO, and is modeled with a hand-
implicit field FH to capture the time-varying transformation
of hand motion. This field FH takes the timestamp t and the
canonical position (xiyizi) of the i-th hand Gaussian as
inputs. Since our setting requires modeling dynamic HOI
scenes at any time from any view pose, we use t as the in-
put and construct the following formula:
∆GH = FH {γ (xiyizi) , γ (t)} ,
(4)
γ(·) is positional encoding [2, 45]. Adding noisesmooth [45]
to γ(t) prevents oversmoothing and retains hand details
while fitting coarse geometry.
3

<!-- page 4 -->
Fig. 2. Overview of interaction-aware hand-object Gaussians. We propose a novel framework for reconstructing dynamic HOI scenes
from RGB videos without object shape priors. The framework consists of three components: (1) Specialized Implicit Fields: separate
hand, object, and background fields disentangle dynamic interactions, with hand/object fields capturing high-frequency deformations and
occlusions (leveraging hand information for object’s interaction-aware deformation) while the background field maintains low-frequency
stability; (2) Interaction-aware Gaussian: enhances representation with adaptive weights w and radius o to address contour ambiguity and
occlusions; (3) Progressive Optimization: combines explicit supervision with physical interaction constraints for efficient convergence.
Object Gaussians. Hand-object interactions often cause
deformations or occlusions (e.g., holding). To enhance the
ability of the object field FO to capture interaction-aware
deformations, we introduce hand position as an additional
input to the object field. This overcomes the limitation of
implicit methods [41, 45], which generate global offsets
without explicit hand-object interaction modeling. The ob-
ject field FO takes both hand and object positions as inputs,
formulated as follows:
∆GO = FO

γ
  xk
i yk
i zk
i

⊕
 xk
j yk
j zk
j

, γ (t)
	
.
(5)
Here,
⊕concatenates hand (xk
i , yk
i , zk
i ) and object
(xk
j , yk
j , zk
j ) positions with the canonical Gaussian position
at key-frame k—the moment just before hand-object inter-
action. The object-implicit field FO predicts time-varying
offsets ∆GO at timestamp t, and uses linear annealing of
noisesmooth [45] to stabilize training.
Background Gaussians.
We construct background
Gaussians to better capture the smoothness and static nature
of the background and avoid the unstable dynamic changes
of the background Gaussian distribution caused by the in-
teraction of foreground hand-object, which will affect the
rendering quality of the background [11, 41]. Background
Gaussians GBG are based on the Deform3DGS model [45].
Their positions change over time, as formulated in Eq. (2),
using the background-implicit field FBG with timestamps t.
3.3. Explicit Interaction-Aware Regularizations
2D regularization is to constrain pixel errors in image space.
However, this is insufficient due to significant occlusion
and drastic motion in hand-object interaction scenes. To
enable Gaussians to accurately and efficiently model com-
plex hand-object interactions, besides 2D supervision, we
introduce explicit 3D regularizations from interaction pri-
ors. Critically, these losses require no object shape or pose
priors. We only leverage a lightweight, off-the-shelf hand
tracker [46] to provide coarse 3D hand guidance. These in-
clude object, hand, and interaction losses to stabilize the ro-
tation and transformation of interaction-aware hand-object
Gaussians, ensuring physically plausible dynamics without
relying on any object-specific assumptions.
Hand Loss. Hand movement in HOI scenes is fast, mak-
ing dynamic Gaussian fitting much slower and more chal-
lenging. Since MANO vertices explicitly represent the po-
sition of each point, we design a hand loss to optimize the
translation of hand Gaussians. To track their translation, we
use a single Chamfer Distance (CHD) to supervise Gaus-
sian translation in 3D space, we compute its distance to the
nearest vertex on the MANO Vh. This loss measures the
distance from each hand Gaussian to its closest point on the
MANO vertices, encouraging the Gaussians to populate the
hand surface, formulated as follows:
LH
trans = 1
N
N
X
i=1
min
v∈Vh ∥(xiyizi) −(xvyvzv)∥2
2 ,
(6)
where (xiyizi) denotes the i-th Gaussian position, and
(xvyvzv) represents the filtered points within MANO ver-
tices’ range (addressing arm-hand discrepancies).
Object Loss. Since object motion—both translation and
rotation—tends to follow hand motion, we regularize the
object field using hand cues. While translation is implic-
itly aligned via spatial proximity, rotation often suffers from
non-physical flipping, especially in passive contacts. We
thus introduce a hand-guided rotation loss LO
rot that aligns
4

<!-- page 5 -->
object Gaussians with the hand’s dominant rotational trend.
In tightly coupled interactions (e.g., grasping), object
rotation typically follows the hand’s dominant rotational
trend. We compute a global prior Rtarget
hand (t) ∈SO(3) by av-
eraging relevant MANO joint rotations via SVD-based av-
eraging [9]. To apply regularization only during contact, we
modulate the loss with the interaction-aware weight wO
j of
each object Gaussian j, which is small under occlusion/non-
contact and large when engaged. The contact-aware weight
is:
ωj(t) = σ(wO
j ),
(7)
where σ is the sigmoid function. The final loss is:
LO
rot = Et

1
N
N
X
j=1
ωj(t)
log

(Rtarget
hand (t))−1Robj
t,j

2

,
(8)
Robj
t,j ∈SO(3) is the predicted rotation of the j-th object
Gaussian at timestamp t, and log(·) denotes the logarithmic
map from SO(3) to its Lie algebra so(3). This metric-aware
penalty suppresses implausible rotations while preserving
local deformation freedom.
Interaction Loss. Reconstruction of grasping interac-
tions often suffers from edge blurring and mutual occlusion
of Gaussians. To regularize the physical reality, we intro-
duce the self-supervised Chamfer distance between hand
and object Gaussians. Our approach models the hand and
object separately, explicitly defining their positions. This
allows us to introduce an interaction loss to ensure proper
grasping, formulated as follows:
Linteraction =
1
max(|CH|, ϵ)
X
i∈CH
min
j∈CO ∥pi −pj∥2
2
+
1
max(|CO|, ϵ)
X
j∈CO
min
i∈CH ∥pi −pj∥2
2,
(9)
where ϵ = 10−6 avoids division by zero when no contacts
are detected. While this loss promotes hand–object prox-
imity, it does not prevent interpenetration. We therefore
use a separate penetration loss that penalizes overlapping
or overly close Gaussians from the hand and object. This
loss ensures the physical reality of the interaction and en-
hances the visual quality by reducing the distance between
the hand Gaussians and object Gaussians while preventing
overlap.
3.4. Progressive Optimization
In the Hand-Object Interaction (HOI) scene, complex ro-
tations, translations, and occlusions are common. Directly
optimizing all Gaussians leads to slow convergence and po-
sitional misalignment.
To address these issues, we pro-
pose a progressive optimization strategy for learning indi-
vidual implicit fields, which operates in five phases as be-
low: initialization, warm-up, HOI refinement, background
optimization, and collaborative reconstruction.
Initialization.
The MANO vertices [31] provide a
coarse initialization for hand geometry, obtained from an
off-the-shelf hand tracker [46] and used solely to bootstrap
GH. In contrast, for the object, we do not assume any shape,
category, or 3D bounding box prior. Instead, GO is randomly
initialized by uniformly sampling 3D points within an ex-
panded axis-aligned bounding box (AABB) of the MANO
vertices. For the background, GBG is initialized from SfM-
based sparse point clouds.
Warm-up. During the warm-up phase, we use the pro-
posed 3D losses besides the fundamental 2D losses. For
the hand field, we employ LH
trans to guide the deformation
of hand Gaussians GH, ensuring alignment with the target
pose.
For the object field, to stabilize interaction-aware
transformations, we use LO
rot. During the warm-up phase,
we periodically apply gradient-based density adjustments
[13] to optimize the initial Gaussian distribution.
HOI Refinement. We adaptively refine Gaussians by
assigning each Gaussian i a learnable weight wi and ra-
dius oi, oi controls its local influence range. The final re-
finement weight for the k-th nearest neighbor of Gaussian
i is obtained by: (1) computing spatial proximity weights
wspatial
ik
for the K nearest neighbors via a RBF kernel on
distance dik and oi (Eq. 10), (2) normalizing these weights
to sum to one (Eq. 11), and (3) modulating them with a
global importance weight σ(wi) (Eq. 12). This allows joint
learning of global importance wi and local context.
wspatial
ik
= exp

−d2
ik
2o2
i

,
k ∈NK(i),
(10)
where NK(i) denotes the set of K nearest neighbor Gaus-
sians for the i-th Gaussian, dik is the Euclidean distance be-
tween the centers of Gaussians i and k, and oi is the learn-
able radius parameter associated with Gaussian i.
ˆwspatial
ik
=
wspatial
ik
P
j∈NK(i) wspatial
ij
.
(11)
The refined weight of Gaussian i’s k-th neighbor is:
ˆwk
i = σ(wi) · ˆwspatial
ik
,
for
k ∈NK(i),
(12)
where σ(·) is the sigmoid function ensuring wi ∈(0, 1).
Additionally, we query the hand implicit field FH and the
object implicit field FO to obtain their respective rotation
matrices
 ∆R6D ∈R6
→
 ∆R ∈R3×3
and translation
offset ∆(xt
kyt
kzt
k). Using a linear blend of local rigid trans-
formations inspired by LBS [35], we refine the pose of the
5

<!-- page 6 -->
Methods
Translation
Translation&Rotation
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
4DGS [41]
24.86-5.46
0.80-0.13
0.47+0.18
23.68-0.48
0.85-0.01
0.39+0.02
Deform3DGS [45]
26.33-3.99
0.87-0.06
0.29 0.00
23.57-0.59
0.89+0.03
0.28-0.09
SC-GS [11]
25.08-5.24
0.84-0.09
0.46+0.17
17.32-6.84
0.71-0.15
0.48+0.11
Ours
30.32
0.93
0.29
24.16
0.86
0.37
Ours*
33.03+2.71
0.95+0.02
0.27-0.02
24.02-0.14
0.85-0.01
0.39+0.02
Tab. 1. Quantitative comparison on HOI4D [20]. Best and second-best results are bolded and italicized, respectively. Differences (red,
small font) are relative to the “Ours” row. Ours* denotes full-frame evaluation.
Methods
Translation&Rotation
PSNR ↑
SSIM ↑
LPIPS ↓
BIGS [26]
3.85-21.34
0.24-0.65
0.70+0.55
HOLD [5]
18.03-7.16
0.84-0.05
0.26+0.11
4DGS [41]
19.44-5.75
0.82-0.07
0.25+0.10
Deform3DGS [45]
9.68-15.51
0.36-0.53
0.65+0.50
SC-GS [11]
20.37-4.82
0.80-0.09
0.26+0.11
Ours
25.19
0.89
0.15
Ours*
25.17-0.02
0.89 0.00
0.16+0.01
BIGS†
24.51-0.68
0.92+0.03
0.07+0.08
Ours†
28.16+2.97
0.95+0.06
0.07-0.08
Tab. 2. Quantitative comparison on HO3D [7]. Best and second-
best results are bolded and italicized. Differences (red font) are
relative to “Ours”. Ours* denotes full-frame evaluation. † denotes
evaluation on hand and object regions only.
hand-object Gaussians as follows:
T t
k = (xt
kyt
kzt
k) + ∆(xt
kyt
kzt
k),
ρt
i =
X
k∈NK(i)
ˆwk
i
 ∆Rt
k((xiyizi) −(xt
kyt
kzt
k)) + T t
k

.
(13)
Here, (xiyizi) denotes the position of the i-th Gaussian in
canonical space, and ρt
i represents the deformed position of
the i-th Gaussian at timestamp t.
Background Optimization. We pretrain GBG for a fixed
number of iterations, performing periodic density adjust-
ments [13] to ensure a clean background initialization.
Collaborative Reconstruction. In the final stage, FH,
FO, and FBG independently deform their Gaussians into
a shared target space, enabling full HOI scene reconstruc-
tion at any timestamp t. Both hand and object Gaussians
employ HOI refinement (Eq. (13)) to update their param-
eters. The optimization is supervised by interaction con-
straints Linteraction (Eq. (9)) and 2D regularization terms.
This stage ensures physically plausible occlusion relation-
ships, smooth edge transitions, and lighting coherence, im-
proving both the geometric fidelity of reconstructed shapes
and the temporal smoothness of their motion dynamics.
4. Experiments
To validate our approach, we conduct comprehensive
comparisons with state-of-the-art baselines [11, 41, 45]
for dynamic scene reconstruction on both HOI4D [20]
and HO3D [7] datasets.
Additionally, we compare with
HOLD [5] and BIGS [26], two specialized methods
for hand-object interaction reconstruction, on the HO3D
dataset. Following [47], we evaluate pure translation and
translation-rotation using alternate-frame testing to assess
extrapolation to novel interactions. Metrics include PSNR,
SSIM [39], and LPIPS [49]. We further perform full-frame
evaluation for completeness (Table 1 and 2, Ours*). All ex-
periments run on an NVIDIA RTX 3090, achieving optimal
performance in 21,000 iterations (1h20m training time).
Implementation Details. We employ K nearest neigh-
bors for refinement and deformation, with the key-frame k
set to the timestamp just before hand–object contact. Both
Gaussians and the deformation model are optimized using
Adam. HOI4D [20] provides RGB-D videos with frame-
level hand–object poses and masks; we evaluate on two
purely translational and two translation–rotation scenes.
HO3D [7] offers real-world 3D pose annotations for actions
like pickup and rotation. We use camera 4 from HO3D and
select four translation–rotation sequences for egocentric re-
construction.
4.1. Quantitative Comparisons
HOI4D Dataset. We compare against 4DGS [41], De-
form3DGS [45], and SC-GS [11] using official code and
original HOI4D resolution (Table 1).
4DGS is sensitive
to initialization and underperforms in HOI settings. De-
form3DGS and SC-GS, relying on a single deformation
field, fail under occlusion and fast motion. Our interaction-
aware, progressively optimized model overcomes these is-
sues, reducing occlusion artifacts and blur while preserving
hand-object geometry. We achieve a +9% PSNR gain in
translation scenes and improve rotation-heavy scene PSNR
from 23.57 dB (Deform3DGS) to 24.16 dB.
HO3D Datasets. We downsample all input frames to half
resolution for efficient processing of large-scale sequences.
Although HO3D [7] provides camera parameters and hand-
6

<!-- page 7 -->
Fig. 3. Qualitative comparison of our approach and the baseline methods. We present reconstructions from our model and SOTA
baselines (4DGS [41], Deform3DGS [45], SC-GS [11]) on HOI4D and HO3D datasets.
object pose estimates, inherent inaccuracies and pose er-
rors adversely affect the performance of all methods (Ta-
ble 2). 4DGS is highly sensitive to input noise; SC-GS’s
sparse control points fail to model background–foreground
interactions; and Deform3DGS suffers most due to HO3D’s
pose errors (Appendix D of [45]), causing non-convergence.
HOLD [5], designed for geometry rather than view synthe-
sis, lags behind 3DGS-based methods. BIGS [26] reports
poor metrics because it reconstructs only foreground hand-
object without the background (foreground-only results: †
in Table 2). Our approach outperforms all baselines in full-
scene reconstruction.
4.2. Qualitative Comparisons
As shown in Fig. 3, our approach surpasses 4DGS [41], De-
form3DGS [45], SC-GS [11] and HOLD [5] in both appear-
ance and shape. In the HOI4D scene, baselines fail to han-
dle Gaussian offsets under dynamic lighting and interaction,
while our interaction-aware representation preserves shad-
ows and shapes. Isolated background Gaussians improve
contrast and dark details. For the HOI4D scene, separate
hand-object modeling and 3D losses effectively constrain
7

<!-- page 8 -->
Fig. 4. Novel view synthesis of our approach and SC-GS [11]. Our
method shows cleaner renderings from novel viewpoints (within
the egocentric viewing cone), whereas SC-GS suffers from notice-
able artifacts.
interaction-aware deformations, with collaborative recon-
struction smoothing motion and occlusion. In the HO3D
scene featuring irregular flipping, rotation, and finger flex-
ibility—4DGS falters under noisy or inaccurate data and
complex motion. SC-GS loses fine details in interaction
zones due to sparse control points and handles occlusions
poorly.
Deformable3DGS, sensitive to pose errors (Ap-
pendix D of [45]), fails to converge on HO3D as errors am-
plify. HOLD reconstructs hand and object geometry but
produces low-quality full-scene renderings.
Our method
uses w and o to reduce occlusion and blur, achieving su-
perior rendering quality on HO3D and HOI4D.
4.3. Ablation Study
Methods
PSNR↑
SSIM↑
LPIPS↓
w/o Interaction-Aware Module 28.76-4.20 0.91-0.04 0.40+0.05
w/o HOI Refinement
32.23-0.73 0.94-0.01 0.39+0.04
w/o Object Loss
31.45-1.51 0.94-0.01 0.38+0.03
w/o Hand Loss
32.45-0.51 0.95 0.00 0.37+0.02
w/o Interaction Loss
31.79-1.17 0.94-0.01 0.40+0.05
w/ noise σ = 0.01
32.80-0.16 0.95 0.00 0.35 0.00
w/ noise σ = 0.05
32.72-0.24 0.95 0.00 0.35 0.00
Full Model
32.96
0.95
0.35
Tab. 3. Ablation studies on HOI4D. All metrics are reported rel-
ative to the Full Model (differences in red, small font). Noise ro-
bustness results correspond to Fig. 5.
Table 3 reports ablation studies on HOI4D-Scene 1, evalu-
ating the removal of HOI refinement, and the interaction-
aware losses and module.
We also evaluate robust-
ness to imperfect initialization by adding Gaussian noise
N(0, σ2) to the randomly sampled initial object positions
with σ = 0.01 and 0.05.
Table 3 and Fig. 5, our full
model achieves near noise-free rendering quality under dif-
Fig. 5. Our method maintains consistently high rendering quality
across different noise levels, showing strong robustness to initial-
ization errors.
Fig. 6. Disentangled rendering of hand, object, and background.
Our method reconstructs each component with high fidelity while
maintaining coherent hand-object interaction.
ferent noise levels.
Removing HOI refinement degrades
PSNR/SSIM/LPIPS by 2.2%/1.1%/11.4%; ablating object,
hand, or interaction losses causes drops of (4.6%, 1.1%,
8.6%), (1.5%, —, 5.7%), and (3.5%, 1.1%, 14.3%), re-
spectively.
To validate our interaction-aware design, we
conducted an ablation study by eliminating both the field
parameters and their associated training scheme from our
framework.
Tab. 3 clearly shows the performance drop
when the interaction-aware module is absent.
5. Conclusion
In this paper, we propose interaction-aware hand-object
Gaussians with novel optimizable parameters, adopting
piecewise linear hypothesis for a clearer structural represen-
tation. This approach effectively captures complex hand-
object interactions, including mutual occlusion and edge
blur. Leveraging the complementarity and tight coupling
of hand and object shapes, we integrate hand information
into the object deformation field, constructing interaction-
aware dynamic fields for flexible motion modeling. To im-
prove optimization, we propose a progressive strategy that
separately handles dynamic regions and static backgrounds.
Additionally, explicit interaction-aware regularizations en-
hance motion smoothness, physical plausibility, and light-
ing coherence. Experiments show that our approach outper-
forms the baseline methods, achieving state-of-the-art re-
sults in reconstructing dynamic hand-object interactions.
8

<!-- page 9 -->
Limitations. As designed for interaction modeling,
the workflow consists of progressive optimization stages,
which could be unified upon the emergence of new stronger
optimizer.
Our method struggles with extreme cases
(exceedingly rapid motion/complex trajectories), poten-
tially addressable by integrating more interaction priors.
References
[1] Seungryul Baek, Kwang In Kim, and Tae-Kyun Kim. Push-
ing the envelope for rgb-based dense 3d hand pose estimation
via neural rendering. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
1067–1076, 2019. 2
[2] Bharat Lal Bhatnagar, Cristian Sminchisescu, Christian
Theobalt, and Gerard Pons-Moll. Loopreg: Self-supervised
learning of implicit surface correspondences, pose and shape
for 3d human mesh registration. Advances in Neural Infor-
mation Processing Systems, 33:12909–12922, 2020. 3
[3] Xingyu Chen, Yufeng Liu, Chongyang Ma, Jianlong Chang,
Huayan Wang, Tian Chen, Xiaoyan Guo, Pengfei Wan, and
Wen Zheng. Camera-space hand mesh recovery via semantic
aggregation and adaptive 2d-1d registration. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 13274–13283, 2021. 2
[4] Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed
Kocabas, Manuel Kaufmann, Michael J Black, and Otmar
Hilliges.
Arctic: A dataset for dexterous bimanual hand-
object manipulation. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
12943–12954, 2023. 2
[5] Zicong Fan, Maria Parelli, Maria Eleni Kadoglou, Xu
Chen, Muhammed Kocabas, Michael J Black, and Otmar
Hilliges. Hold: Category-agnostic 3d reconstruction of in-
teracting hands and objects from video. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 494–504, 2024. 2, 6, 7
[6] Xiang Guo, Jiadai Sun, Yuchao Dai, Guanying Chen, Xiao-
qing Ye, Xiao Tan, Errui Ding, Yumeng Zhang, and Jingdong
Wang. Forward flow for novel view synthesis of dynamic
scenes. In Proceedings of the IEEE/CVF International Con-
ference on Computer Vision, pages 16022–16033, 2023. 2
[7] Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vin-
cent Lepetit. Honnotate: A method for 3d annotation of hand
and object poses.
In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
3196–3206, 2020. 2, 6
[8] Ankur Handa, Karl Van Wyk, Wei Yang, Jacky Liang,
Yu-Wei Chao, Qian Wan, Stan Birchfield, Nathan Ratliff,
and Dieter Fox.
Dexpilot: Vision-based teleoperation of
dexterous robotic hand-arm system.
In 2020 IEEE Inter-
national Conference on Robotics and Automation (ICRA),
pages 9164–9170. IEEE, 2020. 1
[9] Richard Hartley, Jochen Trumpf, Yuchao Dai, and Hongdong
Li. Rotation averaging. International Journal of Computer
Vision, 103(3):267–305, 2013. 5
[10] Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kale-
vatykh, Michael J Black, Ivan Laptev, and Cordelia Schmid.
Learning joint reconstruction of hands and manipulated ob-
jects. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 11807–11816,
2019. 2
[11] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu,
Yan-Pei Cao, and Xiaojuan Qi.
Sc-gs: Sparse-controlled
gaussian splatting for editable dynamic scenes. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 4220–4230, 2024. 1, 2, 3, 4, 6,
7, 8
[12] Kai Katsumata, Duc Minh Vo, and Hideki Nakayama. A
compact dynamic 3d gaussian representation for real-time
dynamic view synthesis. In European Conference on Com-
puter Vision, pages 394–412. Springer, 2024. 2
[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 1, 2, 3, 5, 6
[14] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis.
Dynmf: Neural motion factorization for real-time dynamic
view synthesis with 3d gaussian splatting. In European Con-
ference on Computer Vision, pages 252–269. Springer, 2024.
2
[15] Jiahui Lei, Yijia Weng, Adam W Harley, Leonidas Guibas,
and Kostas Daniilidis.
Mosca: Dynamic gaussian fusion
from casual videos via 4d motion scaffolds. In Proceedings
of the Computer Vision and Pattern Recognition Conference,
pages 6165–6177, 2025. 2
[16] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang.
Neural scene flow fields for space-time view synthesis of dy-
namic scenes. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 6498–
6508, 2021. 2
[17] Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao.
Gaussian-flow: 4d reconstruction with dynamic 3d gaus-
sian particle. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21136–
21145, 2024. 2
[18] Jia-Wei Liu, Yan-Pei Cao, Tianyuan Yang, Zhongcong Xu,
Jussi Keppo, Ying Shan, Xiaohu Qie, and Mike Zheng
Shou.
Hosnerf: Dynamic human-object-scene neural ra-
diance fields from a single video.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 18483–18494, 2023. 1
[19] Xingyu Liu, Pengfei Ren, Qi Qi, Haifeng Sun, Zirui Zhuang,
Jing Wang, Jianxin Liao, and Jingyu Wang. Generalizable
hand-object modeling from monocular RGB images via 3d
gaussians. In The Thirty-ninth Annual Conference on Neural
Information Processing Systems, 2025. 2
[20] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan,
Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi.
Hoi4d: A 4d egocentric dataset for category-level human-
object interaction. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
21013–21022, 2022. 6
9

<!-- page 10 -->
[21] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In 2024 International Con-
ference on 3D Vision (3DV), pages 800–809. IEEE, 2024. 2
[22] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
2
[23] Gyeongsik Moon, Shoou-I Yu, He Wen, Takaaki Shiratori,
and Kyoung Mu Lee. Interhand2. 6m: A dataset and base-
line for 3d interacting hand pose estimation from a single
rgb image.
In Computer Vision–ECCV 2020: 16th Euro-
pean Conference, Glasgow, UK, August 23–28, 2020, Pro-
ceedings, Part XX 16, pages 548–564. Springer, 2020. 2
[24] Franziska Mueller,
Florian Bernard,
Oleksandr Sotny-
chenko, Dushyant Mehta, Srinath Sridhar, Dan Casas, and
Christian Theobalt. Ganerated hands for real-time 3d hand
tracking from monocular rgb. In Proceedings of the IEEE
conference on computer vision and pattern recognition,
pages 49–59, 2018. 2
[25] Franziska Mueller, Micah Davis, Florian Bernard, Oleksandr
Sotnychenko, Mickeal Verschoor, Miguel A Otaduy, Dan
Casas, and Christian Theobalt. Real-time pose and shape
reconstruction of two interacting hands with a single depth
camera. ACM Transactions on Graphics (ToG), 38(4):1–13,
2019. 2
[26] Jeongwan On, Kyeonghwan Gwak, Gunyoung Kang, Junuk
Cha, Soohyun Hwang, Hyein Hwang, and Seungryul Baek.
Bigs: Bimanual category-agnostic interaction reconstruction
from monocular videos via 3d gaussian splatting. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference (CVPR), pages 17437–17447, 2025. 2, 6, 7
[27] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien
Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
Martin-Brualla. Nerfies: Deformable neural radiance fields.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 5865–5874, 2021. 2
[28] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M Seitz.
Hypernerf:
A higher-
dimensional representation for topologically varying neural
radiance fields. arXiv preprint arXiv:2106.13228, 2021.
[29] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer.
D-nerf: Neural radiance fields
for dynamic scenes. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
10318–10327, 2021. 2
[30] Haozhe Qi, Chen Zhao, Mathieu Salzmann, and Alexander
Mathis. Hoisdf: Constraining 3d hand-object pose estima-
tion with global signed distance fields. In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 10392–10402. IEEE, 2024. 1
[31] Javier Romero, Dimitrios Tzionas, and Michael J Black. Em-
bodied hands: Modeling and capturing hands and bodies to-
gether. arXiv preprint arXiv:2201.02610, 2022. 2, 5
[32] Yu Rong, Takaaki Shiratori, and Hanbyul Joo.
Frankmo-
cap: Fast monocular 3d hand and body motion capture by
regression and integration. arXiv preprint arXiv:2008.08324,
2020. 1, 2
[33] Rui Song, Chenwei Liang, Yan Xia, Walter Zimmer, Hu Cao,
Holger Caesar, Andreas Festag, and Alois Knoll.
Coda-
4dgs: Dynamic gaussian splatting with context and defor-
mation awareness for autonomous driving. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 28031–28041, 2025. 2
[34] Colton Stearns, Adam Harley, Mikaela Uy, Florian Dubost,
Federico Tombari, Gordon Wetzstein, and Leonidas Guibas.
Dynamic gaussian marbles for novel view synthesis of casual
monocular videos. In SIGGRAPH Asia 2024 Conference Pa-
pers, pages 1–11, 2024. 2
[35] Robert W Sumner, Johannes Schmid, and Mark Pauly. Em-
bedded deformation for shape manipulation. In ACM sig-
graph 2007 papers, pages 80–es, 2007. 5
[36] Omid Taheri, Nima Ghorbani, Michael J Black, and Dim-
itrios Tzionas. Grab: A dataset of whole-body human grasp-
ing of objects. In Computer Vision–ECCV 2020: 16th Eu-
ropean Conference, Glasgow, UK, August 23–28, 2020, Pro-
ceedings, Part IV 16, pages 581–600. Springer, 2020. 1
[37] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael
Zollh¨ofer, Christoph Lassner, and Christian Theobalt. Non-
rigid neural radiance fields: Reconstruction and novel view
synthesis of a dynamic scene from monocular video.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 12959–12970, 2021. 2
[38] Dimitrios Tzionas, Luca Ballan, Abhilash Srikantha, Pablo
Aponte, Marc Pollefeys, and Juergen Gall. Capturing hands
in action using discriminative salient points and physics sim-
ulation. International Journal of Computer Vision, 118:172–
193, 2016. 2
[39] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 6
[40] Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen
Tyree, Thomas M¨uller, Alex Evans, Dieter Fox, Jan Kautz,
and Stan Birchfield. Bundlesdf: Neural 6-dof tracking and
3d reconstruction of unknown objects.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 606–617, 2023. 1
[41] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 20310–20320, 2024.
1, 2, 3, 4, 6, 7
[42] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng,
Yin Yang, and Chenfanfu Jiang.
PhysGaussian: Physics-
Integrated 3D Gaussians for Generative Dynamics .
In
2024 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 4389–4398, Los Alamitos,
CA, USA, 2024. IEEE Computer Society.
[43] Jiawei Xu, Zexin Fan, Jian Yang, and Jin Xie. Grid4d: 4d
decomposed hash encoding for high-fidelity dynamic gaus-
sian splatting. Advances in Neural Information Processing
Systems, 37:123787–123811, 2024. 2
10

<!-- page 11 -->
[44] Zhiwen Yan, Chen Li, and Gim Hee Lee. Nerf-ds: Neural ra-
diance fields for dynamic specular objects. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 8285–8295, 2023. 2
[45] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction.
In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 20331–20341, 2024. 1, 2, 3,
4, 6, 7, 8
[46] Yufei Ye, Yao Feng, Omid Taheri, Haiwen Feng, Shubham
Tulsiani, and Michael J Black. Predicting 4d hand trajectory
from monocular videos. arXiv preprint arXiv:2501.08329,
2025. 3, 4, 5
[47] Daiwei Zhang, Gengyan Li, Jiajie Li, Micka¨el Bressieux, Ot-
mar Hilliges, Marc Pollefeys, Luc Van Gool, and Xi Wang.
Egogaussian: Dynamic scene understanding from egocentric
video with 3d gaussian splatting, 2024. 2, 6
[48] Juze Zhang, Haimin Luo, Hongdi Yang, Xinru Xu, Qianyang
Wu, Ye Shi, Jingyi Yu, Lan Xu, and Jingya Wang. Neural-
dome: A neural modeling pipeline on multi-view human-
object interactions. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
8834–8845, 2023. 1
[49] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 6
[50] Shuai Zhang, Guanjun Wu, Zhoufeng Xie, Xinggang Wang,
Bin Feng, and Wenyu Liu.
Dynamic 2d gaussians: Geo-
metrically accurate radiance fields for dynamic objects. In
Proceedings of the 33rd ACM International Conference on
Multimedia, pages 8144–8153, 2025. 2
11
