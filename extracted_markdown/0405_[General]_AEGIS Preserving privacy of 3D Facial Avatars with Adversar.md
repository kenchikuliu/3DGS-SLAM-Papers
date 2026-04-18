<!-- page 1 -->
AEGIS: Preserving privacy of 3D Facial Avatars with Adversarial Perturbations
Dawid Wolkiewicz
Wroclaw University of Science and Technology
Wroclaw Poland
dawid.wolkiewicz@pwr.edu.pl
Piotr Syga
Wroclaw University of Science and Technology
Wroclaw Poland
piotr.syga@pwr.edu.pl
Anastasiya Pechko
Jagiellonian University
Krakow Poland
anastasiya.pechko@doctoral.uj.edu.pl
Przemysław Spurek
Jagiellonian University
Krakow Poland
przemyslaw.spurek@uj.edu.pl
Figure 1. Photos enable 2D face verification but cannot pass 3D verification with liveness detection. Since 3D avatars can be used in such
systems, their identity must be protected. Unlike 2D masking, which must be reapplied for each pose, AEGIS achieves consistent identity
protection across different viewpoints and poses through geometry-aware 3D masking.
Abstract
The growing adoption of photorealistic 3D facial avatars,
particularly those utilizing efficient 3D Gaussian Splatting
representations, introduces new risks of online identity theft,
especially in systems that rely on biometric authentication.
While effective adversarial masking methods have been de-
veloped for 2D images, a significant gap remains in achiev-
ing robust, viewpoint-consistent identity protection for dy-
namic 3D avatars. To address this, we present AEGIS, the
first privacy-preserving identity masking framework for 3D
Gaussian Avatars that maintains subject’s perceived char-
acteristics. Our method aims to conceal identity-related fa-
cial features while preserving the avatar’s perceptual re-
alism and functional integrity. AEGIS applies adversarial
perturbations to the Gaussian color coefficients, guided by
a pre-trained face verification network, ensuring consistent
protection across multiple viewpoints without retraining or
modifying the avatar’s geometry. AEGIS achieves complete
de-identification, reducing face retrieval and verification
accuracy to 0%, while maintaining high perceptual qual-
ity (SSIM = 0.9555, PSNR = 35.52 dB). It also preserves
key facial attributes such as age, race, gender, and emo-
tion, demonstrating strong privacy protection with minimal
visual distortion.
1
arXiv:2511.17747v1  [cs.CV]  21 Nov 2025

<!-- page 2 -->
Operation flow
Gradient flow
AdaFace/ArcFace Model
Set of canonical pose parameters 
Canonical Camera parameters
Fixed reference embedding
Figure 2. An overview of the AEGIS identity masking pipeline. The process adversarially optimizes the color parameters Ct of a 3D
Gaussian avatar G to evade face recognition by the model F(·). A fixed reference embedding er is first obtained by rendering the original
avatar G0 under canonical camera vr and pose pr parameters, and passing the result through F(·). During each PGD optimization step,
a set of camera parameters {vk}K
k=1 is sampled to capture diverse viewpoints. The updated avatar Gt is rendered from these viewpoints
using the rendering function R(·) and alignment module A(·), producing a batch of images. Identity embeddings {ek} are then extracted
using F(·), and their average cosine similarity ¯s with the reference embedding er is computed. This similarity defines the logits (¯s for
”match” and −¯s for ”no match”), from which a cross-entropy loss is computed while targeting the ”no match” class. The resulting loss is
backpropagated through the network using PGD to update the color parameters Ct, yielding an adversarial 3D representation.
1. Introduction
In the era of the metaverse [23, 24] and rapid advances in
photorealistic visualization, 3D facial avatars are becoming
increasingly prevalent in online environments, particularly
those built upon the highly efficient 3D Gaussian Splatting
(3DGS) [10] representation. The creation and widespread
sharing of these visually faithful, controllable digital per-
sonas introduce a critical security concern: identity theft
through the extraction of biometric information. This threat
is particularly severe for systems that rely on face verifi-
cation and recognition for authentication. While capturing
facial data from online videos or calls already presents a pri-
vacy risk, the persistent, high-fidelity nature of a control-
lable 3D avatar constitutes a systemic vulnerability that can
be exploited for large-scale spoofing and impersonation.
The challenge of biometric privacy has been stud-
ied extensively in the domain of 2D imagery.
Exist-
ing adversarial masking techniques like Fawkes [20] and
LowKey [3], introduce subtle, imperceptible perturbations
that shift a user’s features in the embedding space, thereby
protecting images from biometric extraction. Other meth-
ods employ generative anonymization or adversarial edit-
ing to conceal identity-related attributes. However, these
2D pixel-space techniques are unsuitable for 3D avatars be-
cause they lack viewpoint consistency, as shown in Fig. 1.
Any change in camera angle or avatar animation alters the
2D projection, invalidating the previous perturbations and
resulting in unstable protection.
Moreover, 2D methods
do not modify the underlying 3D representation (the Gaus-
sian primitives) and therefore cannot guarantee persistent or
geometry-consistent privacy.
To address this gap, we introduce AEGIS (Adversarial
Evasion Gaussian Identity Shield, shown in Fig. 2), the
first viewpoint-consistent identity obfuscation method for
3D Gaussian Avatars. Our goal is to conceal identity-related
facial information while preserving perceptual realism, re-
semblance to the original subject and and overall usabil-
ity. AEGIS is a post-training approach that achieves pri-
vacy protection by perturbing only the first coefficients of
the spherical harmonics representation, which encode the
base color of the Gaussian primitives. This ensures that
the avatar’s geometry and higher-order view dependence re-
main unaltered. The obfuscation signal is obtained via con-
strained Projected Gradient Descent (PGD) [16] optimiza-
tion against a face verification objective, using ArcFace [4]
and AdaFace [11] embeddings. Because this optimization
occurs directly in the avatar parameter space and propagates
through a differentiable rendering pipeline, the resulting de-
identification is viewpoint-consistent, animation-stable, and
perceptually coherent.
The source code is available at https://github.
com/Woleek/AEGIS.
2

<!-- page 3 -->
We evaluate AEGIS by measuring retrieval and verifi-
cation risks using rendered avatars against large-scale 2D
face datasets, simulating realistic attack conditions. Our
method provides robust privacy guarantees, reducing match
rate and rank-k retrieval by up to 100 times, while preserv-
ing structural similarity (SSIM) and maintaining attribute-
level consistency across age, gender, race, and emotion.
This demonstrates that targeted color-space perturbations of
3D Gaussian primitives offer effective privacy from auto-
mated recognition systems, without compromising the vi-
sual fidelity and utility essential for immersive applications.
Below, we summarize the key contributions of this work:
• Viewpoint-consistent 3D identity protection. We pro-
pose AEGIS, the first privacy-preserving identity mask-
ing framework for 3D Gaussian Avatars, ensuring con-
sistent protection across viewpoints and animations with
maintained visual resemblance.
• Adversarial masking in spherical harmonics space.
AEGIS perturbs only the DC coefficients of spherical har-
monics in Gaussian primitives, preserving geometry and
view-dependent appearance.
• Privacy with high perceptual fidelity. Our optimization
guided by ArcFace and AdaFace reduces verification and
retrieval accuracy by up to 100× while maintaining high
SSIM and soft-trait consistency.
2. Related work
Basic face anonymization methods, such as blurring or
black rectangles, are sufficient for static or low-stakes sce-
narios but offer limited utility in dynamic contexts like
social media or video calls and are vulnerable to rever-
sal [21, 29]. Adversarial 2D protection methods, includ-
ing Fawkes [20] and LowKey [3], reduce recognition by
shifting features or applying evasion filters, yet they de-
pend on knowledge of the identification system and cannot
ensure viewpoint-consistent anonymization for 3D avatars.
Generative anonymization approaches, such as diffusion-
based methods [13], G2Face [27], latent code optimiza-
tion [2], and adversarial makeup editing [22], preserve at-
tributes while changing identity, but operate in 2D and can-
not maintain a persistent, controllable avatar or guarantee
consistency under head motion. Controllable photorealis-
tic head avatars, like GaussianAvatars [17], enable high-
fidelity reenactment across poses and expressions, but prior
work has not addressed privacy against biometric verifica-
tion. Our approach fills this gap by introducing in-renderer
adversarial masking on the DC coefficients of spherical har-
monics, preserving geometry and view-dependent appear-
ance while enforcing viewpoint-consistent identity suppres-
sion without assumptions on the adversary or introducing
recovery risks. This provides a practical solution for pro-
tecting identity in animated, photorealistic 3D avatars while
maintaining visual realism and control.
Figure 3. Visualization of example identity masks obtained using
AEGIS (for AdaFace) and reference 2D masking methods.
3. Method
The proposed avatar identity masking obfuscates identity-
related features of a 3D Gaussian-based avatar [17] and pre-
serves perceptual realism. It formulates masking as an ad-
versarial optimization applied to the DC coefficients (the
zeroth-order spherical harmonics), that encode the base,
view-independent color of each 3D Gaussian primitive [10].
Our pipeline integrates three key components: (1) a dif-
ferentiable renderer for Gaussian Avatars, (2) a pre-trained
face verification network (ArcFace [4] or AdaFace [11])
with built-in face alignment preprocessing, and (3) an ad-
versarial optimization framework. An overview of the pro-
posed method is presented in Fig. 2, which illustrates the
loop between rendering, face verification, and adversarial
optimization over Gaussian color coefficients.
The core hypothesis is that by perturbing the Gaussian
color coefficients in a structured manner, one can generate
a mask that renders the avatar unrecognizable to a face ver-
ification system, while maintaining visual fidelity. This de-
sign employs a privacy-preserving formulation, aiming to
3

<!-- page 4 -->
protect the digital avatar’s identity from automated recog-
nition systems. As the reference identity to be obfuscated,
we use a frontal render of the original avatar, representing
the subject prior to masking. This setting assumes access
to the verification network architecture but not to any real
enrollment data.
3.1. Underlying Models
In our setting, the 3D avatar is represented using the Gaus-
sian Avatars framework [17], which combines 3D Gaus-
sian Splatting for scene representation with the FLAME
parametric model for facial articulation. Our masking tech-
nique is applied to the fully trained avatar and does not af-
fect the initial creation process.
Gaussian Splatting [10] is a differentiable rendering
method for 3D scene reconstruction. The scene consists
of (n) anisotropic 3D Gaussian primitives, each defined
by a mean position (mi), covariance matrix (Σi ∈R3×3),
opacity (σi ∈R), and diffuse color coefficients Ci ∈Rk,
where k denotes the number of spherical harmonics (SH)
coefficients representing view-dependent color. The com-
plete representation can be formally expressed as the fol-
lowing set of triples:
G = {(N(mi, Σi), σi, Ci)}n
i=1 ,
The covariance matrix is defined as Σi = RiSiS⊤
i R⊤
i ,
where Ri, Si ∈R3×3, ensuring positive definiteness. Dur-
ing rendering, each 3D Gaussian is projected onto the 2D
image plane by transforming its covariance into screen
space: Σ′
i = JWiΣiW ⊤
i J⊤, where Σi ∈R3×3, Wi ∈
R3×3 is the world-to-camera transform, and J ∈R2×3 is
the Jacobian of the perspective projection, resulting in the
screen-space covariance Σ′
i ∈R2×2. The final pixel color
Cpx is obtained through differentiable alpha blending of 3D
Gaussians, sorted by depth. We assume that these are in-
dexed by i ∈{1, . . . , N}:
Cpx =
N
X
i=1
ciαi
i−1
Y
j=1
(1 −αj) ,
where ci denotes the color of the i-th Gaussian, and αi is
computed by evaluating a 2D Gaussian with covariance Σi,
scaled by a learned per-Gaussian opacity αi [28].
The optimization of 3D Gaussians is guided by minimiz-
ing a photometric reconstruction loss to achieve an optimal
spatial distribution of Gaussians. Further training and opti-
mization details are provided in [10].
FLAME (Faces Learned with an Articulated Model and
Expressions) [14] is a 3D morphable head model with N =
5,023 vertices and K = 4 joints. It maps low-dimensional
shape (β ∈R|β|), pose (p ∈R|p|), and expression (ψ ∈
R|ψ|) parameters to a full 3D mesh using Linear Blend
Skinning (LBS) with corrective blendshapes. Formally, it is
defined as a function M(β, p, ψ) : R|β|×|p|×|ψ| →R3N:
M(β, p, ψ) = W
 TP (β, p, ψ), J(β), p, W

,
where TP denotes the posed template obtained by adding
shape-, pose-, and expression-dependent offsets to the base
mesh T ∈R3N. W(·) is the standard skinning function
that rotates the vertices of TP around joints J ∈R3K and
applies linear smoothing using blend weights W ∈RK×N.
In our work, the avatar is created from multi-view video
data.
The FLAME model provides semantic correspon-
dence, enabling the identification of Gaussian primitives
corresponding to specific facial regions (e.g., eyes, lips).
The DC coefficients of the selected Gaussians are assem-
bled into a tensor C ∈RN×3 which serves as the optimiza-
tion variable during the adversarial masking process.
Figure 4. Effect of pose variation and privacy budget ϵ on identity
masking persistence (against AdaFace verification system). Top:
unmasked avatar with high similarity during verification across
poses. Middle: masked avatar with ϵ = 0.1 shows reduced sim-
ilarity but residual identity cues at some angles. Bottom: with
ϵ = 0.2, avatar is completely de-identified across all poses. These
results demonstrate the trade-off between visual fidelity and pri-
vacy strength, as higher ϵ values yield stronger privacy at the cost
of greater perceptual deviation.
3.2. Facial Verification
To quantitatively measure identity, the avatar is rendered
from a specific viewpoint with a consistent pose, and its
identity embedding is extracted. Let R(G, v, p) be the dif-
ferentiable rendering process, which synthesizes a 2D im-
age I of the avatar. This rendering is a function of: (1)
the 3D Gaussian model G, (2) a set of camera parame-
ters v defining the viewpoint and projection, and (3) a set
of pose parameters p controlling the underlying FLAME
model.
4

<!-- page 5 -->
Our optimization variable is the DC coefficients tensor C,
which defines the view-independent base color of the Gaus-
sian primitives. We can therefore denote the parametrized
model as GC.
To process the rendered image I, we apply a face
alignment function A(·) that produces face crops of size
112×112 pixels, after pose normalization. The alignment is
based on five facial landmarks predicted by the RetinaFace
model [5], as in [4, 11]. This operation compensates for
variations in scale and in-plane rotation, yielding a canon-
ical input for the verification network. Subsequently, the
aligned image A(I) is then passed to a pre-trained face ver-
ification network F(·) (e.g., ArcFace or AdaFace), which
produces a high-dimensional feature vector e ∈R512. The
identity embedding is obtained as e = F(A(I)). These
networks were selected for their robustness and established
ability to map facial images into an embedding space where
cosine similarity reflects identity proximity.
For the adversarial optimization, we employ Expectation
Over Transformation (EOT) [1], to ensure that the generated
mask remains robust under different viewpoints. We de-
fine a transformation distribution T over viewpoints v, cor-
responding to camera parameters, for which we uniformly
sample an angle α for x-rotation (pitch) and an angle β for
y-rotation (yaw), both within the range [−0.5, 0.5] radians.
At each optimization step, we draw K = 5 viewpoints
v1, . . . , vK ∼T. A neutral reference pose pr is used. The
identity embedding for a given viewpoint v is computed as
a differentiable function of the color tensor C:
ea(C, v) = F
 A(R(GC, v, pr))

,
(1)
and the feature vector ea is L2-normalized after extraction
to produce the unit-length identity embedding.
The reference identity embedding er is computed once
from the original, unmasked avatar (with the original col-
ors C0) and rendered with identical pose and default frontal
viewpoint. er vector serves as the constant target to which
the masked avatar’s embedding is compared. The cosine
similarity between the embeddings quantifies the identity
similarity:
s
 ea(C, v

, er) =
ea(C, v) · er
∥ea(C, v)∥∥er∥,
(2)
where · is the scalar product.
3.3. Adversarial Masking
To conceal the avatar’s identity, we perform adversarial op-
timization on the DC coefficients tensor C. The objective
is to find an optimal perturbed color tensor C⋆that mini-
mizes the expected identity similarity over the transforma-
tion distribution T, thereby pushing the avatar’s embedding
ea away from the reference embedding er. This constrained
optimization problem is formulated as:
C⋆= arg min
C
Ev∼T

s(ea(C, v), er)

s.t.
∥C −C0∥∞≤ϵ .
(3)
The constraint employs the ℓ∞(supremum) norm, which
measures the maximum absolute element-wise difference
between two tensors.
It restricts the optimization to
a bounded neighborhood (hypercube) around the original
color tensor, preventing the optimizer from introducing
drastic local color shifts or structural artifacts. For a tensor
X with components xi, this norm is defined as ∥X∥∞=
maxi |xi| . Accordingly, ∥C −C0∥∞computes the maxi-
mum absolute change across all color channels and Gaus-
sian components. The hyperparameter ϵ defines the pertur-
bation budget, i.e., the maximum allowed per-channel devi-
ation, bounded by the dynamic range of the original tensor:
0 ≤ϵ ≤max(C0) −min(C0), ensuring all perturbed col-
ors remain valid. Its value is selected empirically to balance
privacy preservation and visual realism.
We solve this optimization problem using the PGD [16].
To guide the optimization, we define the objective as
cross-entropy loss L based on the cosine similarity s =
s(ea(C, v), er) between embeddings. For a single view-
point v, the loss is:
L(C, v) = log(1 + e−2sλ) ,
(4)
with logits [−sλ, sλ] and a scaling constant λ = 10. The
expected loss under the transformation distribution is:
LEOT (C) = Ev∼T

L(C, v)

.
(5)
Maximizing this loss is equivalent to minimizing the iden-
tity similarity. The gradient ∇CLEOT thus provides the di-
rection in color space that most effectively increases dissim-
ilarity from the reference identity.
At each iteration t, PGD performs (1) a gradient as-
cent step to maximize the loss, followed by (2) a projection
step enforcing the ℓ∞constraint. The gradient is estimated
stochastically over K sampled viewpoints:
∇CLEOT (C(t)) ≈1
K
K
X
i=1
∇CL(C(t), vi),
(6)
and the update rule is:
C(t+1) = ϕC0,ϵ

C(t) + α · sign
 ∇CLEOT (C(t))

,
(7)
where α is the step size controlling the update magnitude,
set relative to the perturbation budget as α = (0.01/0.3)ϵ.
The sign function specifies the optimal step direction within
the ℓ∞-norm ball. The projection operator ϕC0,ϵ(·) enforces
the constraint by clipping each element:
ϕC0,ϵ(·) = clip(·, C0 −ϵ, C0 + ϵ) ,
(8)
5

<!-- page 6 -->
where clip(x, a, b) denotes an element-wise operation
that restricts each entry xi to the range [ai, bi], i.e.,
clip(xi, ai, bi) = min(max(xi, ai), bi). This ensures that
each updated tensor C(t+1) remains within the ϵ-ball cen-
tered at C0, satisfying the constraint at every iteration.
After Tmax iterations, the final optimized tensor C⋆=
C(Tmax) replaces the original DC coefficients in the avatar
model, yielding a de-identified avatar representation that
persistently conceals identity.
4. Evaluation
To systematically assess the proposed avatar identity mask-
ing framework, we evaluate the masked avatars from three
complementary perspectives: (1) identity retrieval, (2) face
verification, and (3) visual utility preservation.
Our evaluation methodology takes inspiration from [26],
where the authors proposed testing anonymization meth-
ods across the dual axes of privacy protection and utility
preservation. We extend this approach to improve the re-
liability and comparability of results between 2D and 3D
settings. Unlike [26], where 2D methods were tested on
large-scale benchmarks and 3D methods on a small, self-
collected dataset, we unify the evaluation by combining ren-
dered avatar datasets with established 2D face databases.
In the privacy-preserving evaluations, existing large-
scale 2D face datasets are employed to emulate a real-
world verification system with multiple enrolled identities.
Specifically, we use CelebA [15], a large-scale dataset of
over 200k celebrity faces serving as a diverse retrieval
gallery, and LFW (Labeled Faces in the Wild) [9], a bench-
mark dataset of unconstrained face photographs for thresh-
old calibration and verification testing.
Additionally, we construct a reference dataset of 10 in-
dividuals whose photos, obtained from the NeRSemble
dataset [12], were used to train Gaussian Avatars.
This
dataset provides frames from 11 different motion-capture
recordings, each capturing displays of distinct emotions and
expressions.
To ensure compatibility with facial verification models,
we retain only near-frontal images. Yaw, pitch, and roll
thresholds for frontal views are determined by rendering the
corresponding avatars with default camera parameters and
estimating pose from RetinaFace facial landmarks. These
thresholds are then slightly relaxed according to the ob-
served deviation in real images. For each subject, we sam-
ple 22 frames (2 per expression category) to approximate
the mean number of images per identity in CelebA (19.91).
We refer to this curated dataset as NeRSembleGT. In all ex-
periments, the effectiveness of identity masking is assessed
using 2D renderings of avatars, as all metrics and face veri-
fication models require 2D image inputs.
4.1. Identity Retrieval
The rank-k evaluation measures how easily anonymized
avatars can be linked back to their original identity among
a gallery of images. This experiment reflects a realistic re-
trieval scenario in which a privacy-preserving system must
ensure that anonymized faces are sufficiently displaced in
the embedding space, preventing re-identification.
For each anonymized render (query), we extract a face
embedding eq using the same face recognition model em-
ployed during masking, or the alternative one to evaluate
transferability across recognition systems. A gallery of real
embeddings {eg} is constructed from the CelebA test split
and NeRSembleGT datasets.
Cosine distance is used to
rank all gallery embeddings by similarity to query:
di = 1 −
eq · egi
∥eq∥∥egi∥.
(9)
The retrieval rank of the correct identity defines the identi-
fication score.
We report the Accuracy@k metric for k ∈[1, 50]:
Accuracy@k = 1
N
N
X
i=1
I[rank(i) ≤k] ,
(10)
where N denotes the total number of anonymized queries
and I is the indicator function. Lower rank-k accuracy in-
dicates stronger de-identification, as the masked avatars are
less likely to be linked to their original identity. A larger
value of k (e.g., k = 50) corresponds to a more lenient
retrieval criterion and thus represents a more challenging
evaluation setting for the masking method.
4.2. Face Verification
While rank-k retrieval evaluates resistance to identification,
face verification assesses whether the anonymized avatar
can still be accepted as the same individual under standard
biometric protocols.
The decision threshold τ for cosine similarity is cali-
brated using the LFW dataset. Positive and negative em-
bedding pairs are sampled, and the Equal Error Rate (EER)
point is computed by fitting the False Accept Rate (FAR)
and False Reject Rate (FRR) curves:
FAR(τ) = # False Accepts
# Negatives
,
FRR(τ) = # False Rejects
# Positives
.
(11)
The threshold τEER corresponds to the intersection point
where FAR = FRR, representing an optimal balance be-
tween false matches and false rejections. This value is then
used for all subsequent verification tests.
Next, we evaluate the masked avatars against their orig-
inal references from NeRSembleGT using this fixed veri-
fication threshold. For each pair (ea, er), the cosine sim-
ilarity s(ea, er) is computed.
An avatar is classified as
6

<!-- page 7 -->
Method
Rank
1 (%) ↓
Rank
50 (%) ↓
Match rate
(%) ↓
SSIM ↑
PSNR
(dB) ↑
Age
(diff.) ↓
Race
(%) ↑
Gender
(%) ↑
Emotion
(%) ↑
Real images
100
100
100
–
–
–
–
–
–
Avatar renders
100
100
100
1
∞
0
100
100
100
AEGIS ϵ = 0.05
100
100
100
0.9870
41.24
0.70
70
100
100
AEGIS ϵ = 0.1
40
70
70
0.9590
35.67
2.68
60
90
100
AEGIS ϵ = 0.2
0
0
0
0.8881
30.58
5.41
40
100
90
AEGIS ϵ = 0.3
0
0
0
0.8247
28.06
6.66
30
90
90
AEGIS (EFN) ϵ = 0.2
80
100
90
0.9587
35.48
2.67
60
90
80
AEGIS (FLN) ϵ = 0.3
70
100
100
0.9341
33.01
2.49
60
90
90
IdentityDP ϵ = 100
0
50
40
0.8224
27.76
3.00
40
90
70
IdentityDP ϵ = 1
0
20
10
0.8138
27.37
2.30
20
90
60
PixelDP (weak, ϵ = 20)
100
100
100
0.2807
23.13
2.70
90
100
90
PixelDP (strong, ϵ = 5)
100
100
100
0.0459
12.72
6.50
70
90
50
MetricSVD (weak, ϵ = 20)
90
100
100
0.8196
26.57
3.70
50
90
50
MetricSVD (strong, ϵ = 1)
0
10
10
0.6257
16.35
11.80
40
80
20
Table 1. Masking results on the ArcFace verification system. Arrows (↑, ↓) indicate desirable metric directions, and bold highlights the
best-performing configuration. For the AEGIS method, letters denote perturbation regions (E: eyes, F: forehead, N: nose, L: lips); if
unspecified, the method was applied to all Gaussians.
Method
Rank
1 (%) ↓
Rank
50 (%) ↓
Match rate
(%) ↓
SSIM ↑
PSNR
(dB) ↑
Age
(diff.) ↓
Race
(%) ↑
Gender
(%) ↑
Emotion
(%) ↑
Real images
100
100
100
–
–
–
–
–
–
Avatar renders
100
100
100
1
∞
0
100
100
100
AEGIS ϵ = 0.05
100
100
100
0.9859
41.38
1.33
100
80
100
AEGIS ϵ = 0.1
0
0
0
0.9555
35.52
2.62
60
100
80
AEGIS ϵ = 0.2
0
0
0
0.8812
30.08
3.71
50
100
80
AEGIS ϵ = 0.3
0
0
0
0.8139
27.29
4.64
40
100
80
AEGIS (EFN) ϵ = 0.2
0
10
10
0.9563
35.25
2.04
70
100
90
AEGIS (FLN) ϵ = 0.3
0
0
0
0.9287
32.61
1.97
50
90
80
IdentityDP ϵ = 100
10
40
40
0.8224
27.76
3.00
40
90
70
IdentityDP ϵ = 1
0
10
0
0.8138
27.37
2.30
20
90
60
PixelDP (weak, ϵ = 20)
100
100
100
0.2807
23.13
2.70
90
100
90
PixelDP (strong, ϵ = 5)
70
90
90
0.0459
12.72
6.50
70
90
50
MetricSVD (weak, ϵ = 20)
80
100
100
0.8196
26.57
3.70
50
90
50
MetricSVD (strong, ϵ = 1)
10
30
30
0.6257
16.35
11.80
40
80
20
Table 2. Masking results on the AdaFace verification system. Arrows (↑, ↓) indicate desirable metric directions, and bold highlights the
best-performing configuration. For the AEGIS method, letters denote perturbation regions (E: eyes, F: forehead, N: nose, L: lips); if
unspecified, the method was applied to all Gaussians.
a match achieves a similarity score s ≥τEER with any
of real reference images belonging to the same identity, and
as a no-match otherwise. The final verification accuracy
quantifies the proportion of anonymized avatars still rec-
ognized as their originals. Lower verification accuracy in-
dicates stronger privacy protection and reduced biometric
traceability.
4.3. Utility Preservation
Identity masking should protect privacy while preserving
visual quality and perceptual usability. We therefore assess
the utility preservation of the anonymized avatars relative
to their unaltered counterparts using both low-level fidelity
and high-level semantic consistency metrics.
For each pair of original and anonymized renders, two
traditional image quality metrics are calculated: (1) Struc-
7

<!-- page 8 -->
Masked on
ArcFace
(ϵ = 0.05)
ArcFace
(ϵ = 0.1)
ArcFace
(ϵ = 0.2)
ArcFace
(ϵ = 0.3)
AdaFace
(ϵ = 0.05)
AdaFace
(ϵ = 0.1)
AdaFace
(ϵ = 0.2)
AdaFace
(ϵ = 0.3)
Evaluated on
AdaFace
ArcFace
Rank 1 (%) ↓
100
0
0
0
100
60
0
0
Rank 50 (%) ↓
100
10
0
0
100
80
0
0
Match rate (%) ↓
100
0
0
0
100
80
0
0
Table 3. Cross-system evaluation, where the masking is optimized against one face verification system and tested against another.
tural Similarity Index (SSIM) that measures perceptual
structural consistency, and (2) Peak Signal-to-Noise Ratio
(PSNR), quantifying pixel-level fidelity.
Before metric computation, both images are aligned and
cropped to bounding box obtained from RetinaFace detec-
tor. Reported SSIM and PSNR values are averaged across
all paired samples.
To evaluate soft-trait preservation, we use the DeepFace
framework [19] to predict facial attributes from original and
anonymized render pairs. We compute four metrics: the
emotion match rate (agreement in predicted dominant emo-
tion), gender match rate (agreement in predicted gender),
race match rate (agreement in predicted racial category),
and age difference (the absolute difference between pre-
dicted ages).
High SSIM/PSNR and semantic consistency indicate
that the masking procedure preserves visual realism and us-
ability, while low identification and verification scores con-
firm effective identity protection.
5. Experiments
We evaluate AEGIS under varying privacy budgets, recog-
nition systems, and masking configurations. Experiments
are conducted on ten avatars generated using the Gaussian
Avatars method, each for a unique identity. Fig. 3 shows ex-
ample avatar renderings, presenting qualitative results that
compare AEGIS with 2D masking baselines.
We apply AEGIS masking with privacy budgets ϵ ∈
0.05, 0.1, 0.2, 0.3.
For each ϵ, we perform 300 steps of
adversarial optimization against two state-of-the-art face
recognition models, ArcFace [4] and AdaFace [11], treated
as independent adversaries, following the procedure de-
scribed in Sec. 3. The method is benchmarked against Iden-
tityDP [25], PixelDP [6], and MetricSVD [7]. Evaluation
follows the protocol from Sec. 4, assessing identity retrieval
(Rank-1, Rank-50), face verification (Match rate), and vi-
sual utility metrics (SSIM, PSNR, attribute consistency).
Table 1 summarizes the results when ArcFace serves as
the adversarial recognizer. At ϵ = 0.1, Rank-1 retrieval
decreases to 40% while maintaining high structural sim-
ilarity (SSIM 0.9590).
Increasing the privacy budget to
ϵ = 0.2 reduces Rank-1, Rank-50 and Match rate to 0%.
Compared with the baselines, IdentityDP and MetricSVD
exhibit greater degradation in image quality for compara-
ble privacy levels, while PixelDP fails to suppress identity
signal (100% Rank-1) and severely reduces image fidelity
(SSIM 0.0459).
Table 2 reports the corresponding results when targeting
AdaFace. AEGIS achieves full de-identification (0% Rank-
1, 0% Rank-50, 0% Match rate) at ϵ = 0.1, while preserv-
ing high visual quality (SSIM 0.9555, PSNR 35.52 dB) and
semantic consistency. None of the baseline methods reach
a comparable trade-off between privacy and perceptual fi-
delity under equivalent privacy constraints.
To assess transferability, we conduct cross-system eval-
uations (Table 3). Masks optimized for the ArcFace adver-
sary exhibit strong generalization to AdaFace. For instance,
at ϵ = 0.1, the ArcFace-trained masks reduce AdaFace’s
Rank-1 accuracy to 0% without additional optimization. In
contrast, masks trained against AdaFace transfer less effec-
tively to ArcFace, requiring larger perturbations to achieve
comparable de-identification performance. This asymmetry
may stem from differences in the models’ margin formu-
lations. ArcFace applies a uniform additive angular mar-
gin [4], while AdaFace adjusts this margin adaptively ac-
cording to sample difficulty, based on feature norms [11].
Ablation Study
We perform ablation studies to examine
two key aspects of AEGIS: (1) the impact of applying masks
to targeted facial regions, and (2) the stability of the identity
masking under pose variations.
We analyze the contribution of different facial regions
by perturbing selected subsets of the avatar’s Gaussians,
identified by FLAME binding (e.g., eyes, forehead, nose).
Representative combinations are reported in Tables 1 and 2.
Results indicate that region-specific masking is feasible but
less efficient than full-avatar masking. To achieve compa-
rable de-identification, targeted masking requires substan-
tially larger privacy budgets (ϵ = 0.3 or higher) than full-
avatar masking (ϵ = 0.1).
These findings suggest that
identity-relevant information is distributed across the full
avatar representation, and masking the entire model pro-
vides a more efficient privacy-utility balance.
We further evaluate the consistency of the learned masks
across pose changes (Fig. 4). For the unmasked avatar, co-
sine similarity between embeddings remains high across
8

<!-- page 9 -->
poses (0.55–0.82), indicating stable identity.
Applying
AEGIS with a low budget (ϵ = 0.1) substantially reduces
similarity (e.g., from 0.82 to −0.17 in the frontal view).
However, at certain angles, similarity (e.g., 0.19) may re-
main above typical verification thresholds, suggesting par-
tial identity leakage. A higher privacy budget (ϵ = 0.2) pro-
duces a more consistent de-identification effect, with nega-
tive values of cosine similarity across all poses and view-
points (−0.58 to −0.13). These results highlight the trade-
off: lower ϵ values preserve appearance with minimal visi-
ble change but may leave residual identity cues at extreme
poses, while higher ϵ values ensure stronger privacy at the
cost of increased perceptual deviation. In practice, such de-
viations may resemble natural degradations (e.g., compres-
sion artifacts in video communication) and remain visually
acceptable for typical downstream applications.
For additional experiments, visualized examples and ex-
tended ablation study see Appendix A and Appendix B.
6. Conclusion
We introduced AEGIS, an identity masking method for 3D
Gaussian Avatars that perturbs only the base color coef-
ficients through PGD, while preserving geometry, view-
dependent appearance, and the perceived identity of the
avatar. This approach allows users to present realistic 3D
avatars in applications such as video calls, social VR, or
other interactive environments, while safeguarding against
automatic verification and identity theft.
By optimizing
the avatar representation directly, AEGIS ensures view-
point consistency and animation stability by design. Ex-
perimental results indicate that the proposed method effec-
tively reduces automatic recognition accuracy while pre-
serving perceptual fidelity and high-level semantic features.
These findings confirm that geometry-consistent adversarial
color perturbations offer a practical and visually coherent
approach to protect the privacy of photorealistic 3D facial
avatars without diminishing their realism or functional util-
ity.
Limitations and Future Work
AEGIS perturbs only the
DC coefficients of spherical harmonics, preserving geom-
etry but restricting the masking signal and leaving minor
residual identity cues at extreme poses. Cross-system trans-
fer remains asymmetric, motivating future validation across
diverse avatar systems and 3D verification settings to im-
prove transferability and robustness.
References
[1] Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin
Kwok. Synthesizing robust adversarial examples. In Inter-
national conference on machine learning, pages 284–293.
PMLR, 2018. 5
[2] Simone Barattin, Christos Tzelepis, Ioannis Patras, and Nicu
Sebe. Attribute-preserving face dataset anonymization via
latent code optimization. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 8001–8010, 2023. 3
[3] Valeriia Cherepanova, Micah Goldblum, Harrison Foley,
Shiyuan Duan, John P Dickerson, Gavin Taylor, and Tom
Goldstein. Lowkey: Leveraging adversarial attacks to pro-
tect social media users from facial recognition. Proceedings
of the International Conference on Learning Representations
(ICLR). 2, 3
[4] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos
Zafeiriou. Arcface: Additive angular margin loss for deep
face recognition.
In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
4690–4699, 2019. 2, 3, 5, 8
[5] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kot-
sia, and Stefanos Zafeiriou. Retinaface: Single-shot multi-
level face localisation in the wild.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5203–5212, 2020. 5
[6] Liyue Fan. Image pixelization with differential privacy. In
IFIP Annual Conference on Data and Applications Security
and Privacy, pages 148–162. Springer, 2018. 8
[7] Liyue Fan. Practical image obfuscation with provable pri-
vacy. In 2019 IEEE international conference on multimedia
and expo (ICME), pages 784–789. IEEE, 2019. 8
[8] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy.
Explaining and harnessing adversarial examples. In 3rd In-
ternational Conference on Learning Representations, ICLR,
Conference Track Proceedings, 2015. 11
[9] Gary B Huang, Marwan Mattar, Tamara Berg, and Eric
Learned-Miller.
Labeled faces in the wild: A database
forstudying face recognition in unconstrained environments.
In Workshop on faces in’Real-Life’Images: detection, align-
ment, and recognition, 2008. 6
[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2, 3, 4
[11] Minchul Kim, Anil K Jain, and Xiaoming Liu.
Adaface:
Quality adaptive margin for face recognition. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 18750–18759, 2022. 2, 3, 5, 8
[12] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim
Walter, and Matthias Nießner. Nersemble: Multi-view ra-
diance field reconstruction of human heads.
ACM Trans.
Graph., 42(4), 2023. 6
[13] Han-Wei Kung, Tuomas Varanka, Sanjay Saha, Terence Sim,
and Nicu Sebe. Face anonymization made simple. In 2025
IEEE/CVF Winter Conference on Applications of Computer
Vision (WACV), pages 1040–1050, 2025. 3
[14] Tianye Li, Timo Bolkart, Michael. J. Black, Hao Li, and
Javier Romero. Learning a model of facial shape and ex-
pression from 4D scans. ACM Transactions on Graphics,
(Proc. SIGGRAPH Asia), 36(6):194:1–194:17, 2017. 4
[15] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.
Deep learning face attributes in the wild. In Proceedings of
9

<!-- page 10 -->
International Conference on Computer Vision (ICCV), 2015.
6
[16] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
Dimitris Tsipras, and Adrian Vladu. Towards deep learning
models resistant to adversarial attacks, 2019. 2, 5
[17] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide
Davoli, Simon Giebenhain, and Matthias Nießner.
Gaus-
sianavatars: Photorealistic head avatars with rigged 3d gaus-
sians. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 20299–20309,
2024. 3, 4
[18] J´erˆome Rony, Luiz G Hafemann, Luiz S Oliveira, Ismail Ben
Ayed, Robert Sabourin, and Eric Granger. Decoupling di-
rection and norm for efficient gradient-based l2 adversarial
attacks and defenses. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
4322–4330, 2019. 11
[19] Sefik Serengil and Alper Ozpinar. A benchmark of facial
recognition pipelines and co-usability performances of mod-
ules.
Journal of Information Technologies, 17(2):95–107,
2024. 8
[20] Shawn Shan, Emily Wenger, Jiayun Zhang, Huiying Li,
Haitao Zheng, and Ben Y. Zhao. Fawkes: Protecting pri-
vacy against unauthorized deep learning models.
In 29th
USENIX Security Symposium (USENIX Security 20), pages
1589–1604. USENIX Association, 2020. 2, 3
[21] Maitreya Suin, Kuldeep Purohit, and A. N. Rajagopalan.
Spatially-attentive patch-hierarchical network for adaptive
motion deblurring. In 2020 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 3603–
3612, 2020. 3
[22] Yuhao Sun, Lingyun Yu, Hongtao Xie, Jiaming Li, and
Yongdong Zhang.
Diffam:
Diffusion-based adversarial
makeup transfer for facial privacy protection. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 24584–24594, 2024. 3
[23] Xinyi Tu and Bruna De Castro e Silva. Are we ready for the
metaverse? implications, legal landscape, and recommenda-
tions for responsible development. Digital Society, 4(1):9,
2025. 2
[24] Hang Wang, Huansheng Ning, Yujia Lin, Wenxi Wang,
Sahraoui Dhelim, Fadi Farha, Jianguo Ding, and Mahmoud
Daneshmand. A survey on the metaverse: The state-of-the-
art, technologies, applications, and challenges. IEEE Inter-
net of Things Journal, 10(16):14671–14688, 2023. 2
[25] Yunqian Wen, Bo Liu, Li Song, Jingyi Cao, and Rong Xie.
Differential private identification protection for face images.
In Face De-identification: Safeguarding Identities in the
Digital Era, pages 75–108. Springer, 2024. 8
[26] Ethan Wilson, Vincent Bindschaedler, Sophie J¨org, Sean
Sheikholeslam, Kevin Butler, and Eakta Jain.
Towards
privacy-preserving photorealistic self-avatars in mixed real-
ity. arXiv preprint arXiv:2507.22153, 2025. 6
[27] Haoxin Yang, Xuemiao Xu, Cheng Xu, Huaidong Zhang,
Jing Qin, Yi Wang, Pheng-Ann Heng, and Shengfeng He.
G²face: High-fidelity reversible face anonymization via gen-
erative and geometric priors. IEEE Transactions on Informa-
tion Forensics and Security, 19:8773–8785, 2024. 3
[28] Wang Yifan, Felice Serena, Shihao Wu, Cengiz ¨Oztireli,
and Olga Sorkine-Hornung. Differentiable surface splatting
for point-based geometry processing. ACM Transactions on
Graphics, 38(6), 2019. 4
[29] Haoyu Zhai, Shuo Wang, Pirouz Naghavi, Qingying Hao,
and Gang Wang. Restoring gaussian blurred face images for
deanonymization attacks, 2025. 3
10

<!-- page 11 -->
Appendix A. Extended Ablation Study
A.1. Attacks on Different Gaussian Avatar Components
Here we investigate the effect of adversarial perturbations applied to components of the 3D Gaussian Splatting avatar rep-
resentation beyond the DC (base color) coefficients. Specifically, we evaluate attacks on: (1) AC coefficients (higher-order
spherical harmonics), (2) Gaussian positions, (3) scale, (4) rotation, and (5) opacity.
For each component, we perform ℓ∞-bounded PGD with 300 iterations and a privacy budget of ϵ = 0.1, keeping evaluation
protocols identical to the DC-based attack used in AEGIS. Evaluation results (for AdaFace) are summarized in Table 4, with
qualitative comparisons shown in Figure 5.
The results reveal two characteristic failure modes among non-DC attacks: insufficient identity suppression and severe
fidelity degradation. Perturbing opacity, rotation, or scale yields no privacy benefit (each achieves 100% Rank-1, Rank-
50, and Match Rate) indicating that the AdaFace verifier consistently recovers the true identity. Conversely, perturbing
AC coefficients or positions disrupts identity but introduces substantial artifacts. The AC attack attains complete identity
obfuscation (0% match rate) but drastically harms reconstruction quality (SSIM 0.6515; PSNR 22.61) and semantic attribute
preservation (age difference 8.84; emotion match accuracy 40%). Position perturbation offers partial identity suppression
(20% Rank-50 and Match Rate) yet still reduces fidelity (SSIM 0.8935), falling short of AEGIS.
Restricting perturbations to DC coefficients allows AEGIS to preserve the underlying Gaussian geometry and thus main-
tain high perceptual and semantic fidelity. Among all evaluated strategies, AEGIS uniquely satisfies both objectives, achiev-
ing complete de-identification, while preserving high perceptual consistency.
Method
Rank
1 (%) ↓
Rank
50 (%) ↓
Match rate
(%) ↓
SSIM ↑
PSNR
(dB) ↑
Age
(diff.) ↓
Race
(%) ↑
Gender
(%) ↑
Emotion
(%) ↑
AEGIS
0
0
0
0.9555
35.52
2.62
60
100
80
AC
0
0
0
0.6515
22.61
8.84
50
90
40
opacity
100
100
100
0.9977
51.69
0.49
90
100
90
position
0
20
20
0.8935
30.29
2.88
80
90
90
rotation
100
100
100
0.9795
37.39
0.92
90
100
100
scale
100
100
100
0.9927
43.78
1.01
90
100
90
Table 4. Quantitative evaluation of adversarial attacks applied to different components of the 3D Gaussian avatar representation, evaluated
against the AdaFace verification system and compared to the best-performing AEGIS configuration.
Figure 5. Qualitative comparison of adversarial perturbations applied to different components of the 3D Gaussian avatar representation.
Each row shows the resulting masked avatar under attacks targeting AC coefficients, opacity, position, rotation, and scale, contrasted with
the best-performing AEGIS configuration.
By limiting perturbations to only the DC components, AEGIS ensures the the avatar’s geometry remains unaltered, there-
fore preserving perceptual realism and utility of the avatar. AEGIS is the only method to successfully achieve the required
dual objective. It alone provides perfect identity protection (0% match rates) while maintaining the highest visual fidelity
(0.9555 SSIM, 35.52 PSNR) and the best overall semantic attribute preservation among all effective methods.
A.2. Different Adversarial Optimization Strategies
We extend our ablation by evaluating a broader set of adversarial optimization strategies, including both ℓ∞and Euclidean
(ℓ2) perturbation regimes. Beyond the iterative PGD attack used in AEGIS, we consider two additional families of methods.
First, we apply the Fast Gradient Sign Method (FGSM) [8], a single-step linearization that perturbs parameters in the direction
of the gradient sign. Second, we adopt the Decoupled Direction and Norm (DDN) attack [18], which iteratively optimizes
11

<!-- page 12 -->
perturbations by separating the gradient direction from the update magnitude, thereby seeking the minimal ℓ2 norm necessary
to induce a successful attack.
In the high-dimensional DC coefficient space, directly converting our ℓ∞budget (ϵ∞= 0.1) into its ℓ2 equivalent yields
ϵ2 = ϵ∞
√
d, where d denotes the dimensionality of the DC tensor. Empirically, this scaling produces large perturbation
magnitudes that introduce severe and visually unacceptable distortions. To obtain a more realistic perturbation scale, we
rely on DDN to estimate the minimal effective ℓ2 norm required for adversarial success. DDN converges to an average
perturbation magnitude of approximately ∥· ∥2 ≈5. Motivated by this finding, and by the observation that smaller fixed
budgets fail to suppress identity, we set the ℓ2 radius for fixed-budget attacks to ϵ2 = 10. Quantitative comparisons for all
ℓ∞and ℓ2 adversarial strategies are presented in Table 5, with qualitative visualizations provided in Figure 6.
Method
Rank
1 (%) ↓
Rank
50 (%) ↓
Match rate
(%) ↓
SSIM ↑
PSNR
(dB) ↑
Age
(diff.) ↓
Race
(%) ↑
Gender
(%) ↑
Emotion
(%) ↑
AEGIS
0
0
0
0.9555
35.52
2.62
60
100
80
DDN
80
90
90
0.9861
41.67
2.21
90
90
100
ℓ2-PGD (ϵ = 5)
10
60
60
0.9785
39.32
2.64
80
90
90
ℓ2-PGD (ϵ = 10)
0
0
0
0.9490
34.40
4.23
50
90
100
ℓ∞-FGSM (ϵ = 0.1)
100
100
100
0.9516
35.84
3.55
70
90
80
ℓ2-FGSM (ϵ = 5)
100
100
100
0.9778
39.02
2.36
60
90
80
ℓ2-FGSM (ϵ = 10)
90
100
100
0.9390
33.23
4.97
50
90
70
Table 5. Quantitative comparison of different adversarial optimization strategies under ℓ∞and ℓ2 constraints, evaluated against the AdaFace
verification system and compared to the best-performing AEGIS configuration.
Figure 6. Qualitative comparison of identity masking results produced by different adversarial optimization strategies (DDN, PGD, FGSM
under ℓ∞and ℓ2 norms), shown alongside the best-performing AEGIS configuration.
Across all evaluated optimization methods, AEGIS is the only approach that simultaneously achieves complete identity
suppression and high perceptual fidelity. Among ℓ2-based attacks, both DDN and low-budget ℓ2-PGD (ϵ = 5) fail to reliably
break verification, producing match rates of 90% and 60%, respectively. Increasing the perturbation radius to ϵ = 10 enables
ℓ2-PGD to reduce the match rate to 0%, but this comes at the expense of fidelity degradation (SSIM drops to 0.9490 and the
age difference increases to 4.23). Single-step FGSM variants perform worst overall, with both ℓ∞and ℓ2 FGSM achieving
match rates close to 100%, indicating that they are unable to generate sufficiently targeted yet visually subtle perturbations.
Overall, these results show that naive or low-budget ℓ2 attacks are ineffective for identity obfuscation, while larger ℓ2 budgets
compromise fidelity.
Appendix B. Additional Examples
B.1. Masking Persistence Under Rotation
We further evaluate the robustness of our identity masking to changes in viewpoint. Figure 7 illustrates renderings of the
unprotected avatar across a grid of camera poses, covering pitch and yaw values linearly sampled in the range [−0.8, 0.8]
radians. As shown, the subject’s identity remains clearly recognizable across nearly all viewpoints.
Figure 9 presents the same avatar after applying AEGIS with ϵ = 0.1, rendered from the identical set of camera positions.
The perturbation remains stable under rotation, yielding consistent suppression of identifiable facial characteristics across the
complete viewing grid.
These observations are corroborated by the similarity-score distributions in Figures 8 and 10. The unprotected avatar
exhibits a distribution concentrated above the verification threshold, indicating reliable identity recognition under pose vari-
12

<!-- page 13 -->
ation. In contrast, the protected avatar produces a distribution shifted decisively below the threshold, demonstrating that
AEGIS maintains effective identity masking over a wide range of viewing angles.
B.2. Additional Samples Visualization
We include in Figure 11 an expanded set of avatar renderings for all methods evaluated in Table 2. These qualitative examples
illustrate the characteristic visual behavior of each approach across a diverse set of subjects, complementing the quantitative
results.
13

<!-- page 14 -->
Figure 7. Verification results obtained by the AdaFace system across various rotation angles (poses) for the original, unmasked avatar.
Figure 8. Histogram showing the distribution of cosine similarities between the original avatar’s rotated renders and the NeRSebmleGT
reference photos. The renders correspond to the poses presented in Figure 7.
14

<!-- page 15 -->
Figure 9. Verification results obtained by the AdaFace system for the avatar protected using the AEGIS method with ϵ = 0.1, shown across
various rotation angles.
Figure 10. Histogram showing the distribution of cosine similarities between the AEGIS masked avatar’s rotated renders and the NeRSebm-
leGT reference photos. The renders correspond to the poses presented in Figure 9.
15

<!-- page 16 -->
Figure 11. Extended qualitative visualization of avatar masking results for all configurations evaluated in Table 2. (Top)
16

<!-- page 17 -->
(Bottom) Continuation of Fig. 11.
17
