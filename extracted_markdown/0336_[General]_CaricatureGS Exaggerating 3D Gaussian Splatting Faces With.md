<!-- page 1 -->
CaricatureGS: Exaggerating 3D Gaussian Splatting
Faces with Gaussian Curvature
Eldad Matmon
Amit Bracha
Noam Rotstein
Ron Kimmel
Technion – Israel Institute of Technology, Haifa, Israel
Figure 1. Photorealistic 3D caricature avatars produced by our method.
Abstract
A photorealistic and controllable 3D caricaturization
framework for faces is introduced. We start with an intrinsic
Gaussian curvature-based surface exaggeration technique,
which, when coupled with texture, tends to produce over-
smoothed renders. To address this, we resort to 3D Gaus-
sian Splatting (3DGS), which has recently been shown to
produce realistic free-viewpoint avatars. Given a multiview
sequence, we extract a FLAME mesh, solve a curvature-
weighted Poisson equation, and obtain its exaggerated
form.
However, directly deforming the Gaussians yields
poor results, necessitating the synthesis of pseudo–ground-
truth caricature images by warping each frame to its ex-
aggerated 2D representation using local affine transforma-
tions.
We then devise a training scheme that alternates
real and synthesized supervision, enabling a single Gaus-
sian collection to represent both natural and exaggerated
avatars. This scheme improves fidelity, supports local ed-
its, and allows continuous control over the intensity of the
caricature. In order to achieve real-time deformations, an
efficient interpolation between the original and exaggerated
surfaces is introduced. We further analyze and show that
it has a bounded deviation from closed-form solutions. In
both quantitative and qualitative evaluations, our results
outperform prior work, delivering photorealistic, geometry-
controlled caricature avatars.
Project page: https://c4ricaturegs.github.io
1. Introduction
Face caricaturization refers to the action of exaggerating
distinctive facial features while preserving identity. Despite
its promise for lifelike, immersive avatars, producing such
exaggerations in controllable, photorealistic 3D remains
an open challenge.
Successful mesh-based approaches
are based on geometric deformations with curvature-based
methods, such as the scale-aware Poisson framework [31].
When such deformed surfaces are rendered through tradi-
tional mesh-centric pipelines, such as texture mapping, the
results often appear unnatural [31]. Recently, 3D Gaussian
Splatting (3DGS) [18] has emerged as a potential multiview
representation that provides state-of-the-art real-time pho-
torealism by optimizing Gaussian primitives directly from
a given set of images taken from various directions.
This raises the following question.
Can we combine curvature-based geometric fidelity with
3DGS to generate photorealistic caricatures?
To address this, we start with a multiview video of a sub-
ject and its extracted FLAME mesh [23]. From this, solving
the weighted Poisson equation gives us the deformed cari-
cature mesh. We rig Gaussians to the original undeformed
surface and train them following a framework previously
proposed for facial expressions [21]. Later, at inference, we
deform the original mesh and its rigged Gaussians accord-
ing to the caricature mesh, stretching, shearing, and rotating
them. However, modeling these deformations as merely an
additional expression, using Gaussians optimized only on
1
arXiv:2601.03319v1  [cs.GR]  6 Jan 2026

<!-- page 2 -->
the input sequence, leads to low fidelity (see Fig. 5), re-
vealing a domain gap in which caricatures lie outside the
distribution of natural expression dynamics.
To bridge this gap and in the absence of real caricature
training data, we synthesize pseudo–ground truth (GT∗)
by warping each input frame with Local Affine Transfor-
mations (LAT) induced by the correspondence from the
original mesh to its curvature-exaggerated counterpart, pro-
ducing photorealistic supervision (see Sec. 3.2).
During
training, we stochastically alternate between real views and
GT∗views so that a single Gaussian set jointly models both
natural and caricatured deformations, allowing the Gaus-
sians to benefit from real ground truth while adapting to
GT∗.
To mitigate occlusion-related artifacts and protect
fine structures (e.g. hair and mesh boundaries), we apply
a spatial mask that freezes the affected Gaussians during
GT∗steps (Fig. 7). These Gaussians are updated only from
real frames, allowing a consistent appearance to accumulate
in their attributes.
Although trained only on the two sets of views, the op-
timized model offers additional flexibility and control at
inference. First, it generalizes across a continuous range
of caricature intensities, with the exaggeration level con-
trolled by an efficient linear interpolation as an approxi-
mation of the solution to the weighted Poisson equation, a
property that we demonstrate both theoretically and empiri-
cally. Moreover, this representation is robust to both global
and local deformations, enabling controlled localized edits,
such as exaggerating the nose size, while leaving unrelated
regions unchanged.
The new 3DGS animatable representation is the first, to
our knowledge, to enable photorealistic caricature render-
ing while faithfully retaining identity under caricature de-
formations. We compare it to the current state-of-the-art dy-
namic facial reconstruction model [21], which consistently
achieves higher scores and qualitative results in terms of im-
age fidelity, structural consistency, and identity preservation
metrics.
Our contributions include,
• A novel 3DGS training scheme that uses GT∗generated
with local Affine transformations that represent real and
caricature avatars.
• Curvature-weighted deformation with rigged 3DGS for
identity-preserving photorealistic caricatures.
• Real-time avatars supporting variable exaggeration levels
and fine-grained local control of facial features.
2. Related Work
2.1. Representation for 3D Head Avatars
Neural implicit representations have become a dominant ap-
proach for high-fidelity 3D head avatars, enabling photore-
alistic view synthesis from sparse multiview observations.
IMAvatar [45] combines 3D morphable-model param-
eters for pose and expression control using neural blend-
shapes and skinning fields to produce animatable head
avatars. ImFace [43] disentangles identity and expression
using two deformation fields applied to a signed distance
function (SDF) template. ImFace++ [44] extends this ap-
proach with a two-stage refinement framework that im-
proves detail preservation.
NeRFs [24] map spatial coordinates and viewing direc-
tions to radiance and density and render images via volu-
metric integration. For head avatars, Wang et al. [37] en-
code sparse views into a 3D structure-aware grid of anima-
tion codes refined by an MLP. Gafni et al. [9] integrate a
low-dimensional morphable face model with a neural scene
representation to obtain photorealistic, controllable avatars
from monocular video. Gao et al. [11] employ multilevel
voxel fields with low-dimensional expression coefficients to
capture elements beyond mesh blendshapes (e.g. hair and
accessories).
INSTA [47] accelerates dynamic NeRF by
embedding it around a surface representation to obtain an-
imatable avatars from short monocular video and Avatar-
MAV [39] decouples appearance from motion via motion-
aware neural voxel grids.
3D Gaussian splatting
[18] represents 3D scenes as
anisotropic Gaussian primitives, and renders them via dif-
ferentiable splatting.
In the context of head avatars,
Rig3DGS [28] reconstructed scenes in a canonical Gaus-
sian space and learned 3DMM-guided deformations for ef-
ficient and photorealistic animation, while HeadGaS [7] ex-
tended the representation with blendable Gaussians whose
attributes adapt to expression coefficients. MeGA [35] in-
troduced a hybrid mesh–Gaussian design, combining splats
with mesh geometry for high-fidelity rendering and editable
head avatars. GaussianAvatars [27] bound deformable 3D
Gaussians to a parametric face mesh via a binding inher-
itance strategy, and SurFhead [21] replaced the 3D Gaus-
sians with 2D Gaussian surfels [16], applying Jacobian
Blend Skinning and polar decomposition, achieving state-
of-the-art results in dynamic head reconstruction.
2.2. Mesh Deformation and Exaggeration
Classical mesh-based approaches realize deformations us-
ing geometry processing, e.g., Poisson/Laplacian editing
and related curvature-driven deformations [19, 32, 33, 42].
For faces, mesh-based deformation and caricaturization
have been explored through both geometry-driven and data-
driven approaches, evolving from early parametric face
models to modern neural deformation networks. Early work
by Blanz and Vetter [2] introduced the 3D Morphable Face
Model (3DMM), representing shape and texture as linear
combinations of example faces, enabling identity and ex-
pression manipulation.
In the caricature domain, Bren-
nan [3] developed an interactive system for producing line-
2

<!-- page 3 -->
LAT GT* generation
CaricatureGS Training
(1) Surface Caricaturization
Input 
multiview  
video
Extracted
FLAME 
Mesh
Caricaturization
Warping
Alternating Photometric Loss 
Render
Affine 2D 
Triangle 
Transformations
(3)
(2)
Mask
3DGS Binding
Figure 2. CaricatureGS generation framework. (1) From a subject’s multi-view video, we extract a FLAME mesh and compute a
curvature-driven caricature based on it. Combined with subject-specific FLAME parameters, this yields the subject’s caricature mesh.
(2) Per-triangle 2D affine transforms map the neutral mesh projection to its caricatured counterpart, warping each frame to generate
pseudo–ground-truth image pairs. (3) Anisotropic 3D Gaussians primitives are bound to the original mesh and transformed to the cari-
cature mesh via the corresponding 3D triangle transforms. Rendered neutral and caricature views are alternated and compared to their
pseudo–ground-truth counterparts in joint optimization.
drawn caricatures by exaggerating the vector differences be-
tween the features of a subject and an average face. Eigen-
satz [8] used curvature maps to enhance, smooth, and trans-
fer characteristics while preserving global structure. Later,
Sela et al. [31] proposed a scale-aware Poisson-based cur-
vature framework for surface caricaturization, exaggerating
geometric features while maintaining spatial and temporal
coherence.
Data-driven methods have enabled for more expressive
and automated mesh exaggerations. Wu et al. [38] learned
deformation patterns from artist-created examples to gener-
ate 3D caricatures from a single 2D portrait while preserv-
ing identity. Han et al. [12] introduced DeepSketch2Face,
where a CNN infers and refines 3D face or caricature
meshes from 2D sketches, while their later work Carica-
tureShop [13] combined vertex-wise Laplacian scaling with
deep learning to produce photorealistic, personalized 2D
caricatures from reconstructed 3D faces. Jung et al. [17]
advanced this idea by using an MLP to map latent codes
to 3D displacements, supporting controlled and diverse ex-
aggerations. More recent approaches focus on style adap-
tation and broader correspondences. Yan et al. [40] pre-
sented an alignment-aware 3D face morphing framework
with controller-based mapping for cross-species correspon-
dence. Olivier et al. [25] explored GAN-based style transfer
from scans to caricatures. Yoon et al. [41] proposed LeGO,
a one-shot method that fine-tunes a surface deformation net-
work to replicate a target style. An additional line of work
that can be adapted to facial exaggeration is the genera-
tive line, exemplified by Diffusion- and GAN-based 3DGS
editors [6, 22, 36], which operate primarily on appearance
while leaving the underlying geometry unchanged.
3. Method
Here, we introduce a method for creating controllable pho-
torealistic caricaturizations of human faces with 3DGS. Our
pipeline, illustrated in Fig. 2, begins with a multiview video
of a subject, from which we extract a FLAME-fitted mesh.
In Sec. 3.1, we describe how we deform the geometry to
obtain a caricaturized mesh. To supervise 3DGS training,
we generate pseudo–ground-truth caricature images (GT∗)
using a 2D warping scheme (Sec. 3.2). The Gaussian prim-
itives are then rigged to both the neutral and caricatured
meshes and optimized by minimizing alternating photomet-
ric losses between their renders, the original frames, and the
corresponding GT∗images (Sec. 3.3). Finally, we demon-
strate that this single shared Gaussian set, although trained
3

<!-- page 4 -->
only on these two image domains, supports real-time ren-
dering across a continuous range of exaggeration levels
via surface interpolation and enables region-specific edits
(Sec. 3.4).
3.1. Surface Caricaturization
Starting from the temporally consistent FLAME mesh ob-
tained by fitting the landmarks [34], we apply a curvature-
driven deformation that exaggerates facial geometry. Since
the mesh maintains consistent vertex correspondences
across frames, these deformations preserve temporal coher-
ence. To implement this deformation, we formulate it as a
weighted Poisson equation on the surface.
Let S ∈R3 be a surface with metric G and Gaussian
curvature K(p) for p ∈S. For γ ∈[0, γf], we define the
weighted Poisson equation
∆GSγ
=
∇G·
 w(γ)∇GS

.
(1)
We adopt the curvature-driven deformation model intro-
duced by [30], whose weights are given by w(γ) = |K|γ.
This gives, for each γ, the following family of Poisson equa-
tions :
∆GSγ
=
∇G·
 |K|γ∇GS

.
(2)
In order to derive the deformed surface we solve the PDE
by the following least-squares:
min
˜x ∥L˜x −b∥2
A.
(3)
L is the discrete Laplace–Beltrami operator, defined as
L = A−1W, A is a diagonal area matrix, W is the classic
cotangent weight matrix and b = ∇G ·
 |K|γ∇G(x)

. The
weighted norm is defined as ∥F∥2
A = trace(F T AF). We
denote by Sγ the solution of the weighted Poisson equation
in equation 2.
To accommodate open surfaces, where the Gaussian cur-
vature may be ill defined on ∂S or to allow precise user-
controlled exaggerations as discussed in Sec. 3.4, we im-
pose boundary conditions on the selected vertices, namely:
min
˜x∈Rn ∥L˜x −b∥2
A
s.t.
B˜x = x∗,
(4)
where B ∈{0, 1}m×n selects the rows corresponding to the
set of vertices and x∗are the prescribed boundary positions.
The same constrained system is solved independently for
the y and z coordinates.
An example of the resulting mesh deformation is illus-
trated in part (1) of Fig. 2.
3.2. GT∗Generation via Local Affine Transforms
With these deformed surfaces, the avatar’s geometry is rep-
resented in caricatured form. For photorealistic rendering,
we employ mesh-rigged 3DGS, detailed in Sec. 3.3. Since
using 3DGS without caricature optimization yields poor re-
sults (Sec. 4.2), training requires ground-truth supervision
images. As real caricature images do not exist, we gen-
erate pseudo–ground truth (GT∗): photorealistic caricature
images that preserve identity while ensuring multiview con-
sistency.
One possible way to obtain such supervision is one-shot
stylization (e.g., Zhou et al. [46]), which narrows the natu-
ral–caricature gap using a single exemplar image. However,
it fails to disentangle style from pose and identity, often
transferring both instead of style alone (see supplementary).
We therefore propose an alternative: Local Affine Transfor-
mations (LAT), illustrated in part (2) of Fig. 2.
LAT exploits the shared connectivity of the neutral and
deformed meshes, implying a per-triangle correspondence.
Consider corresponding 3D triangles X = {X1, X2, X3} ∈
R3 and Y
= {Y1, Y2, Y3} ∈R3.
Let π : R3 →R2
denote the image-plane projection, with xi = π(Xi) and
yi
= π(Yi) ∈R2.
Assuming {x1, x2, x3} are non-
collinear, there exists a unique affine map,
Φ(x) = Ax + b,
A ∈R2×2, b ∈R2,
(5)
such that Φ(x) = y. We then used these per-triangle 2D
affine transformations to map color from the original image
to the 2D projection of the deformed mesh. In practice,
we apply an inverse warp from each target pixel back to the
original image and use bilinear interpolation to avoid empty
regions.
Caricature deformation can reveal regions previously
self-occluded in the neutral pose or occlude regions that
were visible, leaving some pixels in GT∗without valid cor-
respondences. To address this, we generate 2D triangle-
level mask for occluded regions. In addition, because hair
strays fall outside the mesh limits and cannot be warped
reliably, we add the hair boundary to the mask. The final
output is pseudo–ground truth (GT∗): high-quality carica-
ture images that preserve identity, ensure multiview consis-
tency, and provide effective supervision for 3DGS, together
with masks indicating per-pixel validity (see appendix for
further details).
3.3. CaricatureGS Training
We model the avatar’s appearance photorealistically using
the 3D Gaussian Splatting framework [18].
Each Gaus-
sian gi stores local attributes: position µi, scale si, rota-
tion ri, opacity σi, and a view-dependent color ci. At each
time frame k ∈[0, N], the FLAME mesh M ⊂R3 is
represented by triangles {Tj[k]}M
j=1, where M is the num-
ber of mesh faces. To ensure spatial–temporal coherence,
each Gaussian Gi is linked [27] to a specific triangle Tj by
a binding index bi, converting its local attributes to world
space.
4

<!-- page 5 -->
Building on this rigged Gaussian setup, SurFhead [21]
used 2D Gaussian surfels [16], which represent surfaces as
oriented planar Gaussian disks, and replaced Linear Blend
Skinning (LBS) with Jacobian Blend Skinning (JBS) for
Gaussians deformations, namely,
Σ1/2
i
=
Jbrisi,
µ′
i = Jbµi + T x
j
where Jb
=
exp

X
i∈adj
vi log(Ui)

·
X
i∈adj
viPi, (6)
where vi are learned weights and T x
j
is the triangle’s
barycentric center. Ui and Pi are the rotations and stretches
from decomposing the Jacobian gradient J via polar de-
composition. Polar decomposition separates rotation and
stretch, ensuring geometrically accurate Gaussian deforma-
tions (see [21] for further details).
We show that a setup originally designed for natural fa-
cial expressions can be adapted to caricature modeling by
applying the deformed caricature mesh for Gaussian de-
formation and using GT∗for 3DGS optimization. Never-
theless, training exclusively on GT∗introduces occlusion-
induced artifacts and limits the model to a single expres-
sion level.
To overcome these limitations, we propose
a joint optimization procedure that alternates supervision
randomly between real video frames and their caricatured
GT∗counterparts, while maintaining a single shared set of
Gaussians, whose rigging ensures consistent kinematics
across both supervision domains. The masks introduced
in Sec. 3.2 prevent supervision of Gaussians correspond-
ing to caricature GT∗pixels that cannot be reliably warped.
The joint optimization scheme allows the caricatured 3DGS
to learn beyond GT∗by simultaneously filling occlusion-
induced holes using supervision from the original frames.
As further demonstrated in Sec. 5.2, this strategy effectively
captures hair details for our caricature avatar, despite hair
pixels being excluded from direct GT∗supervision. More-
over, as explained in Sec. 3.4, it also enables the generation
of intermediate caricatures at any level, at inference, with-
out additional capture.
3.4. CaricatureGS Features
The joint optimization not only complements the caricature
Gaussians with information absent from GT∗but present in
the original frames, it also provides controllability advan-
tages during inference.
Controlling Caricature Level. After joint training at the
target exaggeration level γf, we empirically observe that
the single-rigged Gaussian set generalizes seamlessly, ren-
dering avatars from meshes deformed for any γ ∈[0, γf]
without additional optimization.
However, obtaining the
deformed mesh for each γ requires solving a curvature-
weighted Poisson problem, which poses a runtime bottle-
Figure 3. Parametric trend of the error with respect to γ. The error,
normalized by the bounding-box diagonal of the mesh, increases
from both ends of γ, reaching a negligible maximum at
γf
2 , where
γf = 0.25.
neck and makes interactive control of caricature levels im-
practical. This motivates the need for a representation that
can be efficiently derived from the original mesh S0 and the
precomputed caricatured mesh Sγf . We define this repre-
sentation as a vertex-wise blend:
Sblend(γ)
=
(1 −α) S0 + α Sγf ,
α ≡γ
γf
. (7)
We
define
the
residual
between
the
approximation
Sblend(γ) and the exact solution S(γ) as
δS(γ) = Sblend(γ) −S(γ).
(8)
In the supplementary material, we show that the L2 energy
of this residual can be bounded using Poincaré inequality
together with the Lax-Milgram theorem given by
∥δS(γ)∥L2
≲
˜C γ(γf −γ) ∥∇GS0∥L2,
˜C
=
CP (ln |K|)2 emax{0,γf ln |K|},
(9)
with CP a constant.
This bound is zero at the end points γ = 0, γf, which
means there is no error, as expected from (7) and maximized
near γ = γf
2 , where it remains small in practice. Empiri-
cally, we evaluate the maximal deformation error between
Sblend(γ) and Sγ on varying γ and different subjects, nor-
malized by the mesh bounding-box diagonal. As shown in
Fig. 3, the worst-case deviation is negligible, supporting the
fidelity of the interpolation and confirming that it lies near
the theoretical midpoint of the exaggeration, as predicted.
This implies that, with this approximation, no additional
Poisson equations need to be solved when inferring new γ
values, thereby enabling full interactive control of carica-
ture levels. In Fig. 5, we illustrate that this interpolation
scheme enables a single set of Gaussians to smoothly rep-
resent shape deformations across the full range of γ.
5

<!-- page 6 -->
Figure 4. Visualizations of localized, semantically controlled fa-
cial exaggerations.
Localized Caricature Control.
Our curvature-weighted
model uses the local curvature K to generate a globally
consistent caricature by solving the unconstrained Pois-
son equation.
To target specific regions, we solve the
constrained least-squares system in Eq. (4), whereby only
the chosen region of interest undergoes curvature defor-
mations, producing a smooth and localized exaggerations
that blend harmonically with the rest of the face. Coupled
with the training scheme in Sec. 3.3, the 3DGS, rigged to
the mesh, faithfully tracks these deformations, so the same
Gaussian set realizes semantically controlled exaggerations
while preserving identity and global shape (see Fig. 4).
4. Experiments
We evaluate our caricaturized avatars along two main axes:
(i) photorealistic rendering, (ii) identity preservation. All
experiments are conducted on the NeRSemble dataset [20]
and compared against the recent state-of-the-art 4D avatar
reconstruction method of SurFhead
[21].
Unless noted
otherwise, we apply an unconstrained exaggeration with
γf = 0.25.
4.1. Dataset
The NeRSemble dataset [20] provides a multi-view fa-
cial performance dataset captured by 16 spatially ar-
ranged, synchronized high-resolution cameras.
It com-
prises 10 scripted sequences, 4 emotion-driven (EMO) and
6 expression-driven (EXP), plus an additional free self-
reenactment sequence. For fair comparison, we adopt the
same train/validation/test partition as in [21] with 120, 000
training iterations. Further implementation details are pro-
vided in the supplementary.
Method
CLIP-I ↑CLIP-D ↑CLIP-C ↑DINO ↑
SD ↑
SurFhead
0.67
0.0006
0.944
0.757
0.460
Ours
0.73
0.014
0.945
0.888
0.539
Table 1. Quantitative comparison for a caricature avatar. Higher is
better for all reported metrics.
4.2. Baseline
To the best of our knowledge, there are no explicit meth-
ods that construct a dynamic 3D photorealistic model from
an input multi-view video. To this end, we compare with
SurFhead [21] using the authors’ official implementation.
SurFhead achieves state-of-the-art performance in head re-
construction and reenactment and, in principle, can handle
mesh deformations through JBS, making it the most suit-
able baseline for comparison. We train the SurFhead on the
original input sequence and, at inference, we exaggerate the
underlying mesh using γf, as elaborated in Sec. 2.2, thereby
driving the Gaussians to represent a caricaturized avatar.
4.3. Metrics
Quantitative evaluation of caricature models is inherently
challenging due to their under-constrained nature and the
lack of ground-truth images. We use the following metrics
for evaluation:
• CLIP-I (Image–Prompt Similarity) [15]: Cosine similar-
ity between the rendered image and text in CLIP space.
• CLIP-D (Directional Similarity) [10]:
Measures the
change between source and edited images against the
change between source and edited prompts.
• CLIP-C (Spatial Consistency): Following [14], we report
CLIP image alignment between adjacent novel views of
image embeddings along a novel trajectory.
• DINO (Identity/Structure Consistency): Following [46],
we extract DINO [5] features from the renders and the
corresponding original test frames and compute the co-
sine similarity of the embeddings.
• SD (Score Distillation): Inspired by DreamFusion [26],
we define the reference-free metric as,
SD = 1 −
1
BTN
B,T,N
X
b,t,n
ϵθ

x(b,t,n)
t
, t

−ϵb,t,n

2
2
∥ϵb,t,n∥2
2
. (10)
where ϵθ(xt, t) is the noise predicted by the diffusion
model [29] at time step t, ϵ is the true noise, and B, T, N
refer to the image count, time step, and seed number, re-
spectively. Higher SD indicates that the rendered image is
more consistent with the training distribution of the diffu-
sion model, which is intended to approximate the natural
image distribution.
Text prompts are provided in the appendix. Together, these
metrics evaluate: (i) how well the renders reflect the carica-
6

<!-- page 7 -->
Ground Truth
Ours
Caricature intensity
SurFhead
𝛾= 0.1
𝛾= 0.25
𝛾= 0.15
𝛾= 0.25
Figure 5.
Rendering results from our pipeline [21].
SURFHEAD: Caricature generation by first reconstructing an avatar with the
state-of-the-art SURFHEAD model [21], followed by mesh exaggeration. Ours: Renderings across different caricature intensities. Our
approximation-based control interpolates smoothly along the caricature intensity axis while preserving visual fidelity.
ture intent (CLIP-I, CLIP-D, SD), (ii) identity preservation
and the extent to which exaggerations remain localized to
caricaturization (DINO, CLIP-D), and (iii) consistency of
generated views across novel trajectories (CLIP-C).
4.4. Results
Fig. 5 presents side-by-side renderings at the target exag-
geration level γf for our method and the baseline.
Our
approach maintains subject identity while delivering natu-
ral, visually pleasing exaggerations that remain consistent
across views, and reduces the distortions visible in the base-
line. The figure further illustrates caricature-level controlla-
bility by varying γ from 0 to γf, demonstrating continuous
control and showing that the approximation in Sec. 3.4 suc-
cessfully supports intermediate exaggeration levels.
For quantitative evaluation, we conduct a comprehensive
comparison using the metrics in Sec. 4.3. As summarized
in Tab. 1, our method consistently surpasses the baseline
across all measures, demonstrating that the learned edits
faithfully capture the intended caricature while preserving
both identity and view-consistency.
4.5. Diffusion Based Editing
As an additional baseline, we adapt a diffusion-driven, text-
guided, mesh-free 3DGS editor [6] for caricaturization. Us-
ing the authors’ implementation, we run 5,000 optimization
steps per prompt on multiview images of a subject, guided
by ControlNet-Pix2Pix. Fig. 6a presents a global edit, while
Fig. 6b shows a local edit, manually masked for face and
nose, respectively. While the edits appear visually plausible
in individual views, it is evident that, unlike our method,
this baseline suffers from (i) geometry drift, (ii) unstable,
view-dependent specularities, and (iii) poor multi-view co-
herence.
5. Ablations
5.1. Alternated Training
In this subsection, we demonstrate that training with GT∗,
generated using LAT, is essential for controlling the cari-
caturization level. As discussed in Sec. 4.4, training only
on input images fails to generalize: rendering with a carica-
tured mesh yields heavily degraded outputs. In the supple-
7

<!-- page 8 -->
(a) Edit instruction: “Turn him into a realistic caricature.” The
result exhibits skin-tone shifts and specular degradation.
(b) Edit instruction: “Make his nose bigger.” The geometry
falls apart and color inconsistencies appear across views.
Figure 6.
GaussianEditor [36] caricaturization attempts.
(a)
Global edit. (b) Local semantic edit. Both reveal degraded ge-
ometry and appearance fidelity, particularly in novel views.
mentary, we show that training solely with GT∗also fails:
neutral renders appear unrealistic, with distorted Gaussian
structures. These complementary failures underscore the
necessity of alternating both forms of supervision for ef-
fective caricaturization control.
5.2. Mask
Due to the nature of GT∗generation, certain fine details,
most notably hair, are often misrepresented during the car-
icature stage. To address this, we identify hair regions of
the mesh and freeze the corresponding Gaussian parame-
ters with a suitable mask during GT∗supervision iterations,
thereby preventing updates in those regions when the car-
icature is rendered (see Sec. 3.2). Fig. 7 illustrates the ef-
fect: on the left, hair regions are masked and remain frozen,
whereas on the right they are unfrozen and allowed to train
freely, resulting in unnaturally plastic-looking hair.
6. Limitations
While our method provides a powerful framework for pho-
torealistic 3D caricaturization, several limitations remain.
Figure 7.
Ablation on hair masking.
Without masking,
GT∗introduces visible artifacts in hair regions. Masking and freez-
ing Gaussians associated with hair during GT∗supervision effec-
tively prevents these artifacts.
Although our approach improves upon the baseline, resid-
ual specularity artifacts persist, and small eyelid inaccura-
cies—amplified by over-stretching in LAT, become visually
noticeable. This effect also extends to hair: training Carica-
ture 3DGS hair with input-view supervision alone (without
GT∗) substantially alleviates the issue. However, in some
cases, we observe slight over-smoothing of the hair. Quali-
tative examples of these effects are provided in the supple-
mentary material. Finally, the deformed FLAME mesh does
not fully span the space of facial expressions. For instance,
eyelid closure in caricatured results is imperfect: eyes that
should be completely shut under certain expressions often
remain slightly open, leading to misrepresentations of eye-
lid geometry in the final caricature.
7. Discussion
This work demonstrates that curvature-driven geometric de-
formation and mesh-rigged 3D Gaussian Splatting (3DGS)
can be combined into a single, controllable avatar model
that remains photorealistic under large exaggerations. The
key is a training scheme that alternates supervision between
real views and generated pseudo–ground-truth caricature
views, produced using per-triangle Local Affine Transfor-
mations (LAT) with reliability masks. One Gaussian set
is capable of jointly learning both natural and caricatured
appearance while retaining identity and expression. Prior
work indicates that deliberate shape exaggeration can am-
plify discriminative geometric cues for recognition [30].
Looking ahead, we hypothesize that integrating our con-
trollable exaggeration as a plug-in augmentation within
face-recognition pipelines could improve robustness to pose
and expression variability. Finally, coupling our geometry-
grounded deformations with diffusion-based editors may
enable semantically guided edits that are both photorealistic
and extend beyond appearance-only changes to joint control
of shape and appearance.
8

<!-- page 9 -->
References
[1] Kendall E. Atkinson and Weimin Han. Theoretical Numer-
ical Analysis: A Functional Analysis Framework. Springer,
3rd edition, 2009.
[2] Volker Blanz and Thomas Vetter. A Morphable Model For
The Synthesis Of 3D Faces.
[3] Susan E. Brennan. Caricature Generator: The Dynamic Ex-
aggeration of Faces by Computer. Leonardo, 18(3):170–178,
1985. Publisher: The MIT Press.
[4] Richard L. Burden and J. Douglas Faires. Numerical Analy-
sis. Brooks/Cole, 10th edition, 2010.
[5] Mathieu Caron, Hugo Touvron, Ishan Misra, Hervé Jégou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-
ing properties in self-supervised vision transformers.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), 2021.
[6] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng
Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu,
and Guosheng Lin. GaussianEditor: Swift and Controllable
3D Editing with Gaussian Splatting.
In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 21476–21485, Seattle, WA, USA, 2024.
IEEE.
[7] Helisa Dhamo, Yinyu Nie, Arthur Moreau, Jifei Song,
Richard Shaw, Yiren Zhou, and Eduardo Pérez-Pellitero.
HeadGaS: Real-Time Animatable Head Avatars via 3D
Gaussian Splatting. arXiv e-prints, art. arXiv:2312.02902,
2023.
[8] Michael
Eigensatz,
Robert
W.
Sumner,
and
Mark
Pauly.
Curvature-Domain Shape Processing.
Com-
puter Graphics Forum, 27(2):241–250, 2008.
_eprint:
https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-
8659.2008.01121.x.
[9] Guy Gafni, Justus Thies, Michael Zollhöfer, and Matthias
Nießner.
Dynamic Neural Radiance Fields for Monocu-
lar 4D Facial Avatar Reconstruction.
arXiv e-prints, art.
arXiv:2012.03065, 2020.
[10] Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik,
and Daniel Cohen-Or.
Stylegan-nada:
Clip-guided do-
main adaptation of image generators.
arXiv preprint
arXiv:2108.00946, 2021.
[11] Xuan Gao, Chenglai Zhong, Jun Xiang, Yang Hong, Yudong
Guo, and Juyong Zhang. Reconstructing Personalized Se-
mantic Facial NeRF Models From Monocular Video. arXiv
e-prints, art. arXiv:2210.06108, 2022.
[12] Xiaoguang Han, Chang Gao, and Yizhou Yu.
DeepS-
ketch2Face: a deep learning based sketching system for 3D
face and caricature modeling. ACM Transactions on Graph-
ics, 36(4):1–12, 2017.
[13] Xiaoguang Han, Kangcheng Hou, Dong Du, Yuda Qiu,
Yizhou Yu, Kun Zhou, and Shuguang Cui. CaricatureShop:
Personalized and Photorealistic Caricature Sketching, 2018.
arXiv:1807.09064 [cs].
[14] Ayaan Haque, Matthew Tancik, Alexei A. Efros, Aleksander
Holynski, and Angjoo Kanazawa.
Instruct-NeRF2NeRF:
Editing 3D Scenes with Instructions.
arXiv e-prints, art.
arXiv:2303.12789, 2023.
[15] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le
Bras, and Yejin Choi. CLIPScore: A Reference-free Eval-
uation Metric for Image Captioning.
arXiv e-prints, art.
arXiv:2104.08718, 2021.
[16] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2D Gaussian Splatting for Geometrically Ac-
curate Radiance Fields. In Special Interest Group on Com-
puter Graphics and Interactive Techniques Conference Con-
ference Papers ’24, pages 1–11, 2024.
arXiv:2403.17888
[cs].
[17] Yucheol Jung, Wonjong Jang, Soongjin Kim, Jiaolong Yang,
Xin Tong, and Seungyong Lee. Deep Deformable 3D Carica-
tures with Learned Shape Control. In Special Interest Group
on Computer Graphics and Interactive Techniques Confer-
ence Proceedings, pages 1–9, 2022. arXiv:2207.14593 [cs].
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis. 3D Gaussian Splatting for Real-Time
Radiance Field Rendering. ACM Transactions on Graphics,
42(4):1–14, 2023.
[19] ByungMoon Kim and Jarek Rossignac. Geofilter: Geometric
selection of mesh filter parameters. Comput. Graph. Forum,
24:295–302, 2005.
[20] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim
Walter, and Matthias Nießner. NeRSemble: Multi-view Ra-
diance Field Reconstruction of Human Heads. ACM Trans-
actions on Graphics, 42(4):1–14, 2023. arXiv:2305.03027
[cs].
[21] Jaeseong Lee, Taewoong Kang, Marcel C. Bühler, Min-Jung
Kim, Sungwon Hwang, Junha Hyung, Hyojin Jang, and
Jaegul Choo. SurFhead: Affine Rig Blending for Geomet-
rically Accurate 2D Gaussian Surfel Head Avatars, 2024.
arXiv:2410.11682 version: 1.
[22] Guohao Li, Hongyu Yang, Yifang Men, Di Huang, Weixin
Li, Ruijie Yang, and Yunhong Wang.
Generating Ed-
itable Head Avatars with 3D Gaussian GANs,
2024.
arXiv:2412.19149 [cs].
[23] Tianye Li, Timo Bolkart, Michael. J. Black, Hao Li, and
Javier Romero. Learning a model of facial shape and ex-
pression from 4D scans. ACM Transactions on Graphics,
(Proc. SIGGRAPH Asia), 36(6):194:1–194:17, 2017.
[24] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. CoRR, abs/2003.08934, 2020.
[25] Nicolas Olivier, Glenn Kerbiriou, Ferran Argelaguet Sanz,
Quentin Avril, Fabien Danieau, Philippe Guillotel, Ludovic
Hoyet, and Franck Multon. Study on Automatic 3D Facial
Caricaturization: From Rules to Deep Learning. Frontiers in
Virtual Reality, 2:1–15, 2022.
[26] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Milden-
hall. DreamFusion: Text-to-3D using 2D Diffusion, 2022.
arXiv:2209.14988 [cs].
[27] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide
Davoli, Simon Giebenhain, and Matthias Nießner.
Gaus-
sianAvatars: Photorealistic Head Avatars with Rigged 3D
Gaussians, 2024. arXiv:2312.02069 [cs].
[28] Alfredo Rivero, ShahRukh Athar, Zhixin Shu, and Dim-
itris Samaras.
Rig3DGS: Creating Controllable Portraits
9

<!-- page 10 -->
from Casual Monocular Videos.
arXiv e-prints, art.
arXiv:2402.03723, 2024.
[29] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Björn Ommer.
High-resolution image
synthesis with latent diffusion models.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10684–10695, 2022.
[30] Matan Sela, Yonathan Aflalo, and Ron Kimmel. Compu-
tational caricaturization of surfaces. Computer Vision and
Image Understanding, 141:1–17, 2015.
[31] Matan Sela, Yonathan Aflalo, and Ron Kimmel. Compu-
tational caricaturization of surfaces. Computer Vision and
Image Understanding, 141:1–17, 2015.
[32] Olga Sorkine and Marc Alexa. As-rigid-as-possible surface
modeling. In Proceedings of EUROGRAPHICS/ACM SIG-
GRAPH Symposium on Geometry Processing, pages 109–
116, 2007.
[33] Olga Sorkine, Daniel Cohen-Or, Yaron Lipman, Marc Alexa,
Christian Rössl, and Hans-Peter Seidel. Laplacian surface
editing. In Proceedings of the EUROGRAPHICS/ACM SIG-
GRAPH Symposium on Geometry Processing, pages 179–
188. ACM Press, 2004.
[34] Justus Thies, Michael Zollhöfer, Marc Stamminger, Chris-
tian Theobalt, and Matthias Nießner. Face2Face: Real-time
Face Capture and Reenactment of RGB Videos. arXiv e-
prints, art. arXiv:2007.14808, 2020.
[35] Cong Wang, Di Kang, He-Yi Sun, Shen-Han Qian, Zi-Xuan
Wang, Linchao Bao, and Song-Hai Zhang.
MeGA: Hy-
brid Mesh-Gaussian Head Avatar for High-Fidelity Render-
ing and Head Editing. arXiv e-prints, art. arXiv:2404.19026,
2024.
[36] Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, and
Qi Tian. GaussianEditor: Editing 3D Gaussians Delicately
with Text Instructions, 2024. arXiv:2311.16037 [cs].
[37] Ziyan Wang, Timur Bagautdinov, Stephen Lombardi, Tomas
Simon, Jason Saragih, Jessica Hodgins, and Michael Zoll-
höfer. Learning Compositional Radiance Fields of Dynamic
Human Heads. arXiv e-prints, art. arXiv:2012.09955, 2020.
[38] Qianyi Wu, Juyong Zhang, Yu-Kun Lai, Jianmin Zheng,
and Jianfei Cai.
Alive Caricature from 2D to 3D, 2018.
arXiv:1803.06802 [cs].
[39] Yuelang Xu, Lizhen Wang, Xiaochen Zhao, Hongwen
Zhang, and Yebin Liu. AvatarMAV: Fast 3D Head Avatar
Reconstruction Using Motion-Aware Neural Voxels. arXiv
e-prints, art. arXiv:2211.13206, 2022.
[40] Xirui Yan, Zhenbo Yu, Bingbing Ni, and Hang Wang. Cross-
Species 3D Face Morphing via Alignment-Aware Controller.
Proceedings of the AAAI Conference on Artificial Intelli-
gence, 36(3):3018–3026, 2022.
[41] Soyeon Yoon, Kwan Yun, Kwanggyoon Seo, Sihun Cha,
Jung Eun Yoo, and Junyong Noh.
LeGO: Leveraging a
Surface Deformation Network for Animatable Stylized Face
Generation with One Example, 2024. arXiv:2403.15227 [cs]
version: 1.
[42] Yizhou Yu, Kun Zhou, Dong Xu, Xiaohan Shi, Hujun Bao,
Baining Guo, and Heung-Yeung Shum. Mesh editing with
poisson-based gradient field manipulation.
In ACM SIG-
GRAPH 2004 Papers, pages 644–651. 2004.
[43] Mingwu Zheng, Hongyu Yang, Di Huang, and Liming
Chen.
ImFace: A Nonlinear 3D Morphable Face Model
with Implicit Neural Representations.
arXiv e-prints, art.
arXiv:2203.14510, 2022.
[44] Mingwu Zheng, Haiyu Zhang, Hongyu Yang, Liming Chen,
and Di Huang. ImFace++: A Sophisticated Nonlinear 3D
Morphable Face Model with Implicit Neural Representa-
tions. arXiv e-prints, art. arXiv:2312.04028, 2023.
[45] Yufeng Zheng, Victoria Fernández Abrevaya, Marcel C.
Bühler, Xu Chen, Michael J. Black, and Otmar Hilliges. I
M Avatar: Implicit Morphable Head Avatars from Videos.
arXiv e-prints, art. arXiv:2112.07471, 2021.
[46] Yang Zhou, Zichong Chen, and Hui Huang.
Deformable
One-shot Face Stylization via DINO Semantic Guidance,
2024. arXiv:2403.00459 [cs].
[47] Wojciech Zielonka, Timo Bolkart, and Justus Thies.
In-
stant Volumetric Head Avatars.
arXiv e-prints,
art.
arXiv:2211.12499, 2022.
10

<!-- page 11 -->
CaricatureGS: Exaggerating 3D Gaussian Splatting
Faces with Gaussian Curvature
Supplementary Material
8. Implementation considerations
Unless stated otherwise, we optimize each subject’s 3D
Gaussian Splatting model for 120,000 iterations, adhering
to SurFhead’s training protocol and evaluation split [21].
All experiments are run on a single NVIDIA RTX 3090
(24 GB VRAM). The optimization time per subject is ≈4
hours (this is offline training time, not rendering runtime.)
We used the NeRSemble dataset [34] with 10 subjects,
4 emotions (EMO), and 6 expressions (EXP). Expression
EXP2 is held for testing and Camera 8 serves as the valida-
tion view during training.
Caricaturization is performed once at the beginning of
the training by solving the unconstrained Poisson equation,
deforming the FLAME base template with γ = 0.25 (≈
1 min).
Because FLAME uses a shared template across subjects,
the deformed surface is saved and reused for all subjects.
Unless stated otherwise, we report metrics over 256 frames
from the rendered test sequence, aggregated across all cam-
era viewpoints.
CLIP configuration.
For text–image alignment, we use
OpenAI CLIP with the ViT-B/32 backbone and the li-
brary’s default preprocessing.
Prompts are: Source: “A realistic neutral head with nat-
ural lighting.” Edit: “A photorealistic caricature of a head
with a highly exaggerated nose and large ears, under natural
lighting.”
Defaults inherited.
The optimizer, learning rate sched-
ule,
degree
of
spherical
harmonics,
and
Gaussian
growth/pruning follow the SurFhead [21] configuration un-
less otherwise specified.
9. Linear Model and Error Analysis
Notation.
Let S(u, v) be a parametric surface, where
(u, v) ∈R2, with a metric G and K denotes the Gaussian
curvature at each point of the surface S, and
w(γ)
=
|K|γ = eγL,
L ≡ln |K|.
(11)
For γ ∈[0, γf], denote by Sγ the solution of the weighted
Poisson problem with Dirichlet boundary condition x∗on
∂S.
To avoid degeneracies at K = 0, we use ϵ to stabilize the
magnitude. Note, for convenience we refer to as |K|ϵ =
√
K2 + ϵ2 with fixed ϵ > 0. For brevity we write |K| to
denote this stabilized quantities.
1) Poisson equation with secant weights. The original
family is defined by
∆GSγ
=
∇G·
 w(γ) ∇GS

.
(12)
Note, that S0 and Sγf refer to γ = 0 and γ = γf, respec-
tively. Define the vertex blend,
Sblend(γ)
=
(1 −α) S0 + α Sγf ,
α ≡
γ
γf
.(13)
By linearity of ∆G and Equation (13)
∆GSblend(γ)
=
(1 −α) ∆GS0 + α ∆GSγf
=
∇G·

wsec(γ) ∇GS

,
(14)
where secant weight is
wsec(γ) = 1 + γ
γf
 |K|γf −1

.
(15)
Thus Sblend(γ) solves the exact Poisson equation at level γ
with w(γ) replaced by wsec(γ), and Sinterp|∂S = x∗(see
(4) for x∗).
2) Remainder and properties The secant wsec is the lin-
ear interpolant of w in [0, γf]. By the classical interpola-
tion remainder for C2 functions on a closed interval (e.g.,
[4, Thm. 3.1], [1, §3.3]), for every γ ∈[0, γf] there exists
ξ(γ) ∈(0, γf) such that
wsec(γ) −w(γ)
=
w′′(ξ)
2
γ(γf −γ).
(16)
Since w′′(γ) = L2eγL, we get
wsec(γ) −w(γ)
=
L2
2 eξL γ(γf −γ).
(17)
The secant model is exact at both endpoints (where α = 0
and α = 1, yielding a analytic expression in [0, γf] preserv-
ing the convexity-induced non-negativity.
Since w′′ ≥0, γ 7→w(γ) is convex, hence
wsec −w is nonnegative on [0, γf] and vanishes at the
endpoints.
In particular, at γ = γf/2,
wsec( γf
2 ) −w( γf
2 )

≤
γ2
f
8 L2 max (1, eγf L ).(18)
1

<!-- page 12 -->
The maximum of this upper bound occurs at γf/2 because
γ(γf −γ) is maximized there.
3) Poincaré and Lax–Milgram for residual bound.
Throughout, we approximate the γ–dependent weight
w(γ) = |K|γ by its secant wsec(γ) to enable a cheap vertex
blend instead of solving a new Poisson problem for each
γ. To justify this alternative, we should quantify how the
weight error propagates to a geometric residual δS(γ) ≡
S(γ) −Sblend(γ). The goal here is to derive a norm bound
on δS that depends only on: (i) ellipticity and Poincaré con-
stants of the domain, (ii) the magnitude of ∇GS0, and (iii)
the scalar secant remainder from Appendix Eq. (18). This
yields a mesh and metric agnostic error budget for the blend.
Setting (frozen operator).
Let (S, G) be a compact Rie-
mannian surface with Lipschitz boundary ∂S. We impose
Dirichlet conditions u

∂S = 0.
We fix the differential operators on the surface S,
namely, the gradient and the divergence w.r.t metric G.
Let V ≡H1
0(S) and define
a(u, v)
=
Z
S
⟨∇Gu, ∇Gv⟩G dAG
∥u∥V
≡
∥∇Gu∥L2(S).
(19)
We also define the dual norm by
∥F∥V ′ ≡
sup
v∈V \{0}
|F(v)|
∥v∥V
.
(20)
Using Poincaré inequality, there exists CP > 0 such that,
for all u ∈H1
0(S),
∥u∥L2(S) ≤CP ∥∇Gu∥L2(S) = CP ∥u∥V .
(21)
Hence ∥u∥V is a true norm on H1
0(S) and is equivalent to
the standard H1-norm on H1
0(S).
By Cauchy–Schwarz,
|a(u, v)|
≤
∥u∥V ∥v∥V
(boundedness),
a(v, v)
=
∥v∥2
V
(coercivity with α = 1) (22)
where coercivity means that there exists α > 0 such that
a(v, v) ≥α ∥v∥2
V
∀v ∈V.
Lax–Milgram. If a is bounded and coercive on the Hilbert
space V and F ∈V ′ is bounded, then, there exists a unique
solution u ∈V , solving a(u, v) = F(v) for all v ∈V , with
estimate
∥u∥V ≤1
α ∥F∥V ′
(22)
= ∥F∥V ′.
(23)
For each γ, we solve the weighted Poisson PDE given by
∆GSγ
=
∇G
 w(γ) ∇GS

,
Sγ

∂S = x∗. (24)
Let Sblend(γ) = (1 −α)S0 + αSγf with α = γ/γf, and
define
ψ(γ)
≡
wsec(γ) −w(γ)
R∆(γ)
≡
∇G
 ψ ∇GS

.
(25)
Define F ∈V ′ (weak residual functional) by
F(v)
=
⟨R∆, v⟩
=
Z
S
 ∇G(ψ ∇GS)

v dAG
=
−
Z
S
ψ ⟨∇GS, ∇Gv⟩G dAG,
(26)
with v

∂S = 0.
Using the dual norm and by Cauchy–Schwarz and
∥ψ∥L∞-bound, we readily have
|F(v)|
≤
∥ψ∥L∞(S) ∥∇GS∥L2(S) ∥∇Gv∥L2(S)
=
∥ψ∥L∞∥∇GS∥L2(S) ∥v∥V ,
(27)
and using (20) we get
∥F∥V ′
≤
∥ψ∥L∞∥∇GS∥L2(S).
(28)
Let δS ≡Sblend −Sγ. Subtract the weak forms for Sblend
and Sγ to obtain
a(δS, v)
=
a(Sblend, v) −a(Sγ, v)
=
Z
S
wsec ⟨∇GS, ∇Gv⟩G dAG
−
Z
S
w(γ) ⟨∇GS, ∇Gv⟩G dAG
=
Z
S
ψ ⟨∇GS, ∇Gv⟩G dAG
=
−
Z
S
∇G
 ψ ∇GS

v dAG
(∗)
≡
−F(v).
(29)
Where in (*) we use integration by parts and Dirichlet
boundary conditions on ∂S.
Testing with v = δS and using coercivity and duality,
∥δS∥2
V
=
a(δS, δS)
=
−F(δS) ≤∥F∥V ′ ∥δS∥V
⇒∥δS∥V
≤
∥F∥V ′.
(30)
Combining with the bound on ∥F∥V ′ yields the energy es-
timate
∥δS∥V
≤
∥ψ∥L∞(S) ∥∇GS∥L2(S)
∥δS∥V
≤
∥wsec −w∥L∞∥∇GS∥L2(S).
(31)
2

<!-- page 13 -->
Optional L2 bound.
By Poincaré on H1
0(S),
∥δS∥L2(S)
≤
CP ∥δS∥V
≤
CP ∥wsec −w∥L∞∥∇GS∥L2(S).(32)
In summary, the secant error bound yields the energy
bound for the residual δS by
∥δS(γ)∥L2 ≲
CP (ln ∥K∥)2emax(0,γf ln ∥K∥)
×γ(γf −γ) ∥∇GS∥L2(S).
(33)
which depends on geometric constants of the domain (CP ).
The curvature in (33) is evaluated at its global maximum
∥K∥= K∞= max
s∈S |K(s)|
(34)
We note that S0 = S (for γ = 0 by definition since there
is no deformation done to S), hence (33) can be written
using either terms.
10. Caricature GT∗via one-shot stylization
As discussed in Sec. 3, one-shot stylization methods (e.g.,
Deformable StyleGAN [46]) address the natural-caricature
domain gap by aligning DINO features and adapting a pre-
trained GAN to a single caricature exemplar. Given a target
style image (Fig. 8a), they synthesize stylized outputs for
arbitrary inputs. In practice, we observe pronounced iden-
tity–expression entanglement, which degrades both identity
fidelity and expression accuracy (Fig. 8). Moreover, the
outputs are not consistent across viewpoints or expressions:
under view changes or when transferring expressions from
the source, the method exhibits structural drift and a col-
lapse toward the reference style (Figs. 8b and 8c), limiting
its suitability for our 3DGS reconstruction setting.
Protocol.
We ran [46] using the official implementation,
employing Style1, Style2, and Style3 as target style
exemplars and EMO3, EMO4 for expression prompts.
11. Masking and GT∗
As noted in Sec. 3.2, GT∗supervision is constructed by pro-
jecting the FLAME mesh, fitted to each original frame, onto
the image. Consequently, the quality of GT∗inherits any
mesh–image misregistration. In practice, small fitting errors
that are negligible at γ=0 are amplified as the caricature
strength increases, with the most visible drift around deli-
cate geometry such as the eyelids and eyeballs; see Fig. 9.
In addition, the deformation can reveal triangles that were
occluded in the original projection (e.g., along the eyelid
crease), creating pixels with no reliable photometric sup-
port.
(a) Deformable StyleGAN [46]: stylization conditioned on a target style
exemplar.
(b) View variation induces identity drift and structural artifacts (e.g. neck
geometry).
(c) Expressions are not preserved, outputs bias toward the style exemplar
(e.g. persistent smile, forward gaze).
Figure 8. Limitations of one-shot stylization for caricature. Iden-
tity–expression entanglement and lack of view/expression consis-
tency hinder 3DGS supervision.
To prevent these failure modes, we build a visibility-
aware GT∗mask.
We (i) suppress supervision on trian-
gles that become newly visible at nonzero γ relative to the
original projection, and (ii) mask anatomically fragile re-
gions prone to amplified alignment error (eyelids, ear tips).
3

<!-- page 14 -->
Caricature intensity
Figure 9. FLAME–image misregistration under increasing car-
icature strength γ.
Projection drift concentrates on thin, high-
curvature structures (eyelids/iris rim) and grows with γ, introduc-
ing erroneous supervision if used unfiltered.
This filtering removes inconsistent labels before they reach
Gaussians anchored to those areas, yielding cleaner gra-
dients and more stable appearance/geometry during train-
ing.
The resulting GT∗thus preserves the benefits of
deformation-aware supervision while avoiding artifacts in-
troduced by projection drift and occlusions.
12. Ablation: Alternating Supervision
Setup.
As motivated in Sec. 5.1, we seek a single 3DGS
model that renders both the original avatar (γ=0) and its
caricatured counterpart (γ=γf). We compare three training
schedules using identical budgets: (i) Original-only: super-
vision from original frames only. (ii) GT∗-only: supervi-
sion from caricatured (GT∗) frames only. (iii) Alternating
(ours): alternating mini-batches from both sources. We set
the target exaggeration to γf=0.25 and evaluate along the
interpolation path γ ∈{0, 0.10, 0.15, 0.20, 0.25}.
Findings.
Original-only (i) fits the undeformed scene
well but fails to generalize to caricatured geometry Fig. 10,
yielding visible distortions under nonzero γ. Conversely,
GT∗-only (ii) represents the caricatured avatar but degrades
markedly at γ=0. In addition, GT∗-only exhibits systematic
artifacts around hair and other structures that extend beyond
the tracked mesh support (e.g. holes or under-coverage),
because those pixels are never directly supervised in the
warped domain, see Fig. 11.
Our alternate schedule (iii) maintains high fidelity at
both endpoints and produces smooth interpolation across γ
(see Fig. 12), avoiding the hair/occlusion failures seen in
(ii). Practically, alternating acts as a simple multi-domain
regularizer, as it preserves appearance outside the mesh sup-
port (from original frames) while learning the exaggerated
geometry and view-dependent effects required by GT∗.
Conclusions.
Alternating supervision is necessary to ob-
tain a single 3DGS that is faithful at γ=0 and γ=γf and
stable along the interpolation path, while training on either
domain alone leads to domain-specific overfitting and char-
acteristic failure modes.
4

<!-- page 15 -->
Figure 10. Training on original frames only
5

<!-- page 16 -->
Figure 11. Training on GT∗frames only.
6

<!-- page 17 -->
Figure 12. Training on both original and GT∗frames interleaved
7
