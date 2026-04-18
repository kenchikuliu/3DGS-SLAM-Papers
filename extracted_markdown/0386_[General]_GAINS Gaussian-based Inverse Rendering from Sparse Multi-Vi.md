<!-- page 1 -->
GAINS: Gaussian-based Inverse Rendering from Sparse Multi-View Captures
Patrick Noras1,2
Jun Myeong Choi3
Didier Stricker1,2
Pieter Peers4
Roni Sengupta3
1University Kaiserslautern-Landau
2German Research Center for Artificial Intelligence
3University of North Carolina at Chapel Hill
4College of William & Mary
Ref-GS [ICLR’25]
GI-GS [ICLR’25]
GAINS (Ours)
Intrinsics
Inputs
Relighting
…
Figure 1. We introduce GAINS, GAussian-based INverse rendering from Sparse multi-view captures, which synergizes learning-based
priors related to monocular depth/normal, segmentation, intrinsic image decomposition (IID), and diffusion, to better disambiguate re-
flectance from lighting, leading to better intrinsics, novel view synthesis and relighting compared to existing state-of-the-art approaches
such as Ref-GS [36] and GI-GS [3]. Prior methods often overfit diffuse (e.g., missing yellow reflection from the ground in the first re-
lighting example for GI-GS) and/or specular reflections (e.g., for both Ref-GS and GI-GS the reflected details remain unchanged under
different lighting conditions). In contrast, GAINS improves estimation of material properties leading to better relighting in novel views.
Abstract
Recent advances in Gaussian Splatting-based inverse
rendering extend Gaussian primitives with shading parame-
ters and physically grounded light transport, enabling high-
quality material recovery from dense multi-view captures.
However, these methods degrade sharply under sparse-view
settings, where limited observations lead to severe ambi-
guity between geometry, reflectance, and lighting. We in-
troduce GAINS (Gaussian-based Inverse rendering from
Sparse multi-view captures), a two-stage inverse rendering
framework that leverages learning-based priors to stabi-
lize geometry and material estimation. GAINS first refines
geometry using monocular depth/normal and diffusion pri-
ors, then employs segmentation, intrinsic image decompo-
sition (IID), and diffusion priors to regularize material re-
covery. Extensive experiments on synthetic and real-world
datasets show that GAINS significantly improves material
parameter accuracy, relighting quality, and novel-view syn-
thesis compared to state-of-the-art Gaussian-based inverse
rendering methods, especially under sparse-view settings.
Project page: https://patrickbail.github.io/gains/
1
arXiv:2512.09925v1  [cs.CV]  10 Dec 2025

<!-- page 2 -->
1. Introduction
Inverse rendering (IR) aims to recover the intrinsic 3D
properties of a scene (i.e., geometry, material properties,
and lighting) from multi-view images. IR serves as a key
step toward physically grounded 3D scene understanding in
many downstream applications such as novel-view synthe-
sis, relighting, material and shape editing, etc. Over the past
decades, inverse rendering has seen remarkable progress,
evolving from early methods [24, 25, 38] that use simple in-
trinsic representations such as surface normals, albedo, and
spherical harmonics lighting, to modern approaches that
employ rich and realistic intrinsic representations such as
3D Gaussian primitives, physically-based BRDFs, and re-
alistic illumination [5, 6, 14, 16, 18, 28, 31, 41]. Similarly,
the scope of inverse rendering has broadened from simple,
controlled objects like faces to complex real-world scenes
containing intricate geometry and diverse materials.
Despite these advances, scaling inverse rendering from
simple objects to complex scenes with intricate details
typically requires dense multi-view captures that provide
strong geometric and photometric constraints that help dis-
ambiguate material properties, lighting, and shape, all of
which are entangled in the image formation process. How-
ever, when viewpoints are sparse, these constraints weaken,
leading to overfitting and degraded performance. In such
settings, existing methods often fail to recover accurate in-
trinsic properties, producing inconsistent material and light-
ing estimates. Moreover, traditional smoothness priors are
inadequate for resolving the albedo-lighting ambiguity, re-
sulting in poor novel-view synthesis (NVS) and relighting.
To address these challenges,
we propose GAINS
(GAussian-based INverse rendering from Sparse multi-
view captures), a novel inverse rendering framework de-
signed to operate robustly under sparse-view settings. We
focus on a Gaussian splatting based framework since state-
of-the-art approaches [3, 8, 36] found Gaussian splatting
to be more effective for inverse rendering both in terms
of quality and efficiency.
GAINS follows the standard
two-stage inverse rendering pipeline, where we first es-
timate geometry (Stage I), followed by an estimation of
material properties and lighting (Stage II). Geometry es-
timation forms the foundation of physically-based render-
ing, and thus, high-quality geometry is crucial for accu-
rate material and lighting recovery. In Stage I, we lever-
age learning-based priors from monocular depth and nor-
mal prediction networks as well as latent diffusion mod-
els, drawing inspiration from recent progress in sparse-view
Gaussian-based reconstruction methods [34, 35], as detailed
in Sec. 4. However, the primary contribution of GAINS
lies in Stage II, where we introduce three complementary
learning-based priors: (1) A segmentation guidance that
enforces multi-view consistency and reduces noise in ma-
terial maps. However, segmentation guidance lacks intrin-
sic knowledge of surface reflectance, limiting its generaliza-
tion to novel views and lighting conditions; (2) An Intrin-
sic Image Decomposition (IID) prior, implemented using
a state-of-the-art intrinsic image decomposition network,
that provides a strong initialization for albedo, at the cost
of poor cross-view consistency and weak generalization;
and (3) a latent diffusion prior that offers strong gener-
alization capabilities. However, the latent diffusion prior
struggles with material consistency and multi-view consis-
tency. Only by combining all three complementary priors
(Sec. 5), GAINS recovers robust material properties and
lighting while achieving stable novel-view synthesis and re-
lighting, even under extreme sparse input conditions.
We conduct extensive experiments on two synthetic
benchmarks (TensorIR [10], which provides ground-truth
albedo and relighting supervision, and Syn4Relight [44],
which additionally includes ground-truth roughness), as
well as one real-world dataset [29]. GAINS consistently
outperforms state-of-the-art Gaussian-based inverse render-
ing methods across geometry and material estimation tasks.
The improvements are especially pronounced under sparse-
view conditions, though our method also achieves compet-
itive results with dense inputs. We further perform detailed
ablation studies to analyze the contributions of each prior
and demonstrate how their combination yields the most ro-
bust and physically consistent inverse rendering results.
2. Related Work
Inverse Rendering (IR) decomposes images into geome-
try, material properties, and illumination. Classical methods
estimate surface shape and spatially varying material prop-
erties from controlled captures using analytic BRDF mod-
els and global illumination optimization [4, 23, 33]. With
the advent of learning-based scene representations such as
NeRF [19], neural IR frameworks [10, 27, 32, 39, 43] in-
tegrate simplified BRDF and lighting models to recover in-
trinsic scene parameters from multi-view inputs. Notably,
NeRFactor [43] estimates shape, materials, and lighting un-
der unknown illumination using smoothness and BRDF pri-
ors, while GaNI [32] and related work jointly optimize ge-
ometry and material parameters with near-field lighting and
neural radiance caching.
Inverse Rendering with Gaussian Splatting. Recently,
3D Gaussian Splatting (3DGS) [13] emerged as an effi-
cient explicit radiance representation, which subsequently
has been extended to IR [3, 9, 15, 26, 36].
Gaussian-
Shader [9] augments Gaussian primitives with BRDF pa-
rameters, while Ref-GS [36] and GI-GS [3] integrate
physically-based rendering (PBR) with deferred shading
and ray tracing.While Gaussian splatting based IR frame-
works provides higher efficiency and fidelity than neural
implicit representations, they assume dense multi-view cap-
tures and/or object-centric settings, making them less prac-
2

<!-- page 3 -->
Gaussians
ℒDepth
ℒNormal
ℒSDS
Novel Views
ℒColor
+
+
ℒICC
ℒIID
ℒ𝑀𝐼−𝑆𝐷𝑆
Segmentation
+
+
Ground Truth
 Views
Ground 
Truth
 Views
Novel lighting maps
Stage I
Stage II
Learned 
Environment map
Diffusion Model
Sparse Input views
Albedo
Metallic
Roughness
Figure 2. GAINS follows a two-stage inverse rendering pipeline: Stage I reconstructs geometry, and Stage II estimates material parameters
and lighting. In Stage I, we enhance geometry using learning-based priors from monocular depth, normal, and diffusion predictors. In
Stage II, we introduce three complementary priors: segmentation, intrinsic image decomposition (IID), and diffusion, to improve material
estimation, novel-view synthesis, and relighting. Each prior provides distinct benefits: segmentation boosts cross-view consistency of
specular parameters but degrades albedo; IID improves albedo accuracy but remains view-inconsistent; diffusion enhances generalization
to novel view and relighting but lacks material consistency. GAINS integrates these priors to achieve stable, high-quality estimated material
properties, leading to better relighting under novel views.
tical for real-world sparse-view scenarios.
Inverse Rendering with Sparse Views. Earlier approaches
for sparse-view or single-view IR rely on learned pri-
ors from synthetic data [14, 24, 25] without explicit geo-
metric and light-transport modeling leading to poor gen-
eraliation to complex real world scenes.
Recent works
like RelitLRM [42] leverage Large Reconstruction Models
(LRMs) and generative priors for object relighting, but they
can not estimate intrinsic components (e.g., material prop-
erties), nor do they generalize to complex real-world scenes.
Recent approaches for shape reconstruction from sparse
views using NeRF- or 3DGS-based representations increas-
ingly leverage learning-based priors, such as monocular
depth, normal, or diffusion-based guidance [8, 30, 34, 35,
45], demonstrating that combining learned priors with ex-
plicit geometric modeling leads to significant performance
gains. However, a sparse-input inverse rendering frame-
work that jointly integrates such learning-based priors with
explicit shape representations and physically based render-
ing remains largely unexplored, which our work addresses.
3. Overview
Our goal is to reconstruct robust and accurate geometry and
material properties from a sparse set of captured RGB im-
ages {Ii}N
i=1 (with N as low as 4-16 views) under known
camera intrinsics and extrinsics Ci = {Ki, Ri, ti}. Our
pipeline builds on 2D Gaussian Splatting (2DGS) [7], and
adapts it for robust Inverse Rendering (IR) from sparse input
views. We opt for 2DGS over 3D Gaussian Splatting [13]
due to its more accurate geometry reconstruction which is
essential for accurate material property estimation.
Scene representation. As in 2DGS, we parameterize the
geometry as: G = {µj, Sj, Rj, σj}M
j=1, where µj describes
the 3D position of each primitive, along with opacity σj,
scaling vector Sj and rotation matrix Rj; the surface normal
nj is also determined by Rj.
Material and lighting representation.
Following pre-
vious Gaussian splatting-based IR methods [3, 26, 36],
we use a physically-based rendering (PBR) deferred shad-
ing approach.
We employ the simplified Disney BRDF
model [1] and store for each Gaussian primitive the corre-
sponding BRDF parameters: M = {aj, rj, mj}M
j=1, where
aj ∈[0, 1]3 is the albedo, rj ∈[0, 1] the roughness and
mj ∈[0, 1] the metalicity. For efficiency, we employ the
split-sum approximation [11] to model the diffuse and spec-
ular surface reflectance:
Ldiffuse
=
(1 −m) · a
Z
Ω
L(ωi)ωi · n
π
dωi
(1)
Lspecular
≈
Z
Ω
DFG
4(ωo · n)ωidωi
|
{z
}
precomp. BRDF LUT
·
Z
Ω
D L(ωi)dωi
|
{z
}
prefiltered Env. Map
, (2)
where L(ωi) is the direct incident lighting stored as a
128 × 128 enviroment cubemap, n the surface normal. D
the GGX microfacet distribution, F Fresnel term, and G
3

<!-- page 4 -->
geometric shadowing term.
Optimization. The goal of IR is to optimize shape G, ma-
terial reflectance M, and lighting L of the scene to mini-
mize a re-rendering loss with respect to the input images.
Similarly to prior work, we solve the optimization in two
stages: (1) optimize for shape G, followed by (2) a joint
estimation of material parameters M and lighting L. To
improve robustness of the two-stage pipeline under sparse
input views, we augment each stage with appropriate addi-
tional loss terms. We improve geometry estimation (Sec.4)
by leveraging additional learning-based priors from a latent
diffusion model and a monocular depth and normal esti-
mation model following established best practices from re-
cent sparse-view Gaussian-based geometry reconstruction
methods [8, 34, 35]. We stabilize material and lighting es-
timation using three complementary learning-based priors:
segmentation (Sec. 5.1), IID (Intrinsic Image Decomposi-
tion; Sec.5.2), and diffusion priors (Sec.5.3).
4. GAINS Stage I: Shape Estimation
We start with the standard 2DGS shape estimation loss,
originally formulated for dense input views:
L = Lcolor + λDDLDD + λNCLNC,
(3)
where, Lcolor = (1−λ)L1+λLD-SSIM is a re-rendering loss;
LDD is an optional depth distortion loss, reducing depth
ambiguity by bringing intersected Gaussians closer along
each ray, and LNC = 1 −Ni
T ˜Ni is the normal consistency
loss with ˜
Ni being surface normals obtained from depth.
When applied to sparse input views, the above loss tends
to overfit to the captured viewpoints, resulting in significant
novel viewpoint artifacts. To better guide the IR optimiza-
tion towards a robust and generalizable solution, we intro-
duce additional learning based priors: (1) depth and normal
guidance, and (2) diffusion guidance.
Depth and Normal Guidance Inspired by prior work [8,
34, 35, 45] that leverages additional depth supervision to
alleviate geometric overfitting, we also include a depth loss
by enforcing similarity to a monocularly estimated depth
ˆDi [12]: LDC = 1 −PCC(Di, ˆ
Di), where PCC(·) is
the Pearson Correlation Coefficient. Additionally, we also
add a local depth ranking loss LDR [8] to prevent geo-
metric collapse from hard constraints and alleviate long-
range ambiguity. Moreover, given the importance of ac-
curate surface normals in material estimation, we impose
normal smoothness via an additional total variation loss
LNS = TV (Ni, ˆ
Ni), with the reference normals ˆ
Ni ob-
tained from a monocular depth estimator [12].
Diffusion Guidance Inspired by recent successes in lever-
aging diffusion models for zero-shot 3D reconstruction [17,
37], we guide the shape reconstruction through Score Dis-
tillation Sampling (SDS) towards realistic reconstructions
at unseen viewpoints Cj. Concretely, We sample 100 novel
viewpoints for which we render and perform a forward dif-
fusion, yielding noisy latents: Zj = αtRgeo(Cj, G) +
σtϵ, where timestep t ∼U(0.02, 0.98) and noise ϵ ∼
N(0, I0). The SDS loss is then calculated as: LSDS =
Et,ϵ
h
w(t)||(ϵϕ(Zj; t) −ϵ)||2
2
i
. However, the SDS loss is
not effective during early stages of the optimization when
the shape reconstruction is far from the target shape. We
therefore only include the SDS loss after iteration 10000
(out of 16000).
Summary The total loss function for shape estimation in
Stage I is:
Lgeo = Lcolor + λDCLDC + λDRLDR + λNCLNC
+ λNSLNS + λSDSLSDS + λBCELBCE,
(4)
with λDC = 0.005, λDR = 10, λNC = 1, λNS = 0.005,
λSDS = 0.0001. LBCE denotes an alpha mask render-
ing loss using a Binary-Cross Entropy (BCE) between the
rendered alpha from the learned opacity and ground-truth
alpha masks for our sparse inputs. If no ground-truth alpha
mask is provided, we assume the scene to reconstruct the
background as well, otherwise we set λBCE = 0.1. The
complete Stage I pipeline is summarized in Fig.2 (left).
5. GAINS Stage II: Material Properties
In the second stage, we jointly optimize material proper-
ties M = {(aj, rj, mj)} and environment lighting L using
the re-rendering loss Lcolor over 7000 additional iterations.
However, multiple combinations of different M and L pro-
duce similar appearances, leading the an ill-posed and am-
biguous material estimation. This problem is further exac-
erbated when only a sparse set of input views is available.
Although rendering under the recovered lighting and mate-
rial properties appears visually plausible, the material prop-
erties are overfitted, noisy, inconsistent, and not suitable for
relighting (Fig.1). To alleviate the impact of overfitting un-
der sparse inputs, we complement the loss with additional
learning based priors to better guide the optimization to-
wards more plausible material properties, and thus resulting
in more robust relighting. We synergize three complemen-
tary priors: segmentation (Sec.5.1), Intrinsic Image Decom-
position (IID) (Sec.5.2), and diffusion guidance (Sec.5.3).
5.1. Segmentation Guidance
While diffuse albedo varies significantly due to texture vari-
ations, we observe that specular material properties are typ-
ically consistent within semantically similar regions of an
object or scene. This suggests that segmentation cues can
help guide the estimation of specular material properties.
As a single semantic object can contain multiple subparts
with distinct materials, we employ subpart segmentation to
define a consistency loss. Instead of constraining the ma-
terial parameters to be identical for each subpart, we min-
4

<!-- page 5 -->
Synthetic4Relight [44] dataset (8 views)
Methods
NVS
Albedo
Relight
Roughness
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
MSE ↓
Ref-GS [36]
24.051
0.832
0.112
17.378
0.832
0.173
24.296
0.858
0.132
0.037
GI-GS [3]
27.047
0.902
0.115
20.479
0.882
0.125
23.923
0.828
0.112
0.056
Ours
30.23
0.943
0.083
22.97
0.914
0.107
25.582
0.916
0.097
0.026
Table 1. Quantitative evaluation on the synthetic Synthetic4Relight [44] dataset. Our method estimates better albedo and roughness
then baselines leading to better relighting performance. Additionally our method outperforms existing methods in novel-view synthesis.
( Red = best, Orange = 2nd best, and Yellow = 3rd best)
TensorIR [10] dataset (8 views)
Ref-Real [29] dataset (8 views)
Methods
NVS
Albedo
Relight
NVS
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Ref-GS [36]
24.867
0.839
0.119
19.864
0.709
0.219
25.203
0.719
0.129
20.15
0.47
0.353
GI-GS [3]
28.011
0.896
0.111
28.969
0.901
0.121
25.404
0.853
0.135
19.91
0.45
0.371
Ours
29.146
0.915
0.099
27.913
0.912
0.118
26.923
0.892
0.115
21.37
0.56
0.34
Table 2. Quantitative evaluation on TensorIR [10] dataset and real Ref-Real [29] dataset. On TensorIR dataset, our method estimates
better albedo than existing approaches along with superior relighting results. On Ref-real and TensorIR dataset, our method also performs
better novel-view synthesis compared to existing approaches. ( Red = best, Orange = 2nd best, and Yellow = 3rd best)
Method Variant
NVS
Albedo
Relight
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Ours (full)
25.106
0.789
0.236
23.991
0.788
0.276
22.926
0.739
0.274
Ours w/o MI-SDS
25.098
0.788
0.236
24.022
0.788
0.276
22.889
0.737
0.274
Ours w/o Seg
24.862
0.786
0.237
24.214
0.79
0.276
22.894
0.739
0.271
Ours w/o IID
25.148
0.791
0.233
23.706
0.786
0.274
22.878
0.737
0.272
Ours w/o Seg, IID, MI-SDS
24.936
0.788
0.234
23.813
0.785
0.276
22.76
0.738
0.27
Table 3. Evaluation of the contribution of each learning-based prior used in Stage II of our framework (IID, segmentation (Seg),
and diffusion (MI-SDS)) on the tensorIR [10] dataset using 8 viewpoints with best-crop masking. Removing the IID prior reduces
albedo reconstruction fidelity and leads to inconsistent material separation, resulting in degraded relighting performance. Excluding the
segmentation prior produces incorrect object and material boundaries, yielding noisy or unstable specular components and harming novel-
view synthesis. Finally, removing the diffusion prior leads to an overall drop in performance across all tasks, demonstrating its critical
role in stabilizing optimization and improving generalization under sparse-view settings. Overall, the full model consistently achieves the
strongest results across NVS, albedo, and relighting, and ablating any component leads to clear and measurable performance degradation.
( Red = best, Orange = 2nd best, Yellow = 3rd best)
imize their variance to support mixed materials and to ac-
count for imperfect segmentation boundaries.
To include segmentation guidance, we extend the Gaus-
sian primitives with a one-hot vector Ej = {0, 1}K that
encodes each Gaussian’s class membership.
We employ
training-free segmentation mask lifting [2] by rendering
SH colored images from V sampled novel view points
orbiting the scenery.
For each image { ˆIv}V
v=1 we gen-
erate (potentially view-inconsistent) segmentation masks
with SAM [22] along with mask-related features {fv,s =
[gv,s, hv,s]}V
v=1 where fv,s is the feature vector for a mask
region s of view v and gv,s, hv,s are CLIP [21] and DI-
NOv2 [20] features respectively.
For each view, we lift
the associated mask and features to the Gaussian that con-
tributes most (α-blending wise) and merge already collected
Gaussian objects from previous lifting operations with geo-
metric and feature similarity scores.
Next, we leverage the possibly imprecise segmentation
masks to define an intra-class consistency (ICC) on the
specular roughness and metallicity parameters:
LICC =
1
|Si|
X
si∈Si
γ(|si|) ·

Lr,m + La + Le

(5)
Lr,m = V ar(Rsi) + V ar(Msi)
(6)
La =

(1 −Rsi) + Msi

· 1
3
3
X
c=1
V ar(A(c)
si )
(7)
Le =

(1 −Msi) + Rsi

· Esi
(8)
where Si is the set of all masks rendered from a captured
viewpoint i, Asi, Rsi, and Msi are the rendered masked
albedo, roughness, and metallicity, along with Esi repre-
senting the masked re-render loss.
γ(|si|) = exp(25 ·
|Ii||si|
|Si|2 ) is a scaling function depending on the size of the
5

<!-- page 6 -->
Ours (full)
Ours w/o IID
Ours w/o Seg
Relight
Albedo
Ours w/o IID, Seg, SDS
Metallic
Roughness
Figure 3. Ablation studies on the gardensphere scene from the Ref-Real [29] dataset. In absence of learning-based priors (Ours w/o
IID, Seg, SDS in 1st col) reflectance maps are poorly reconstructed, especially metallicity and roughness. Without the IID prior (3rd col)
results in weaker specular effects (compared to the 2nd col). Without segmentation guidance (4th col) results in noise material maps across
objects.
mask si. The first variance reduction term Lr,m reduces ir-
regularities for specular roughness and metallic across all
masks.
While in general we attribute texture variations
to the diffuse texture, an ambiguity exists in the case of
mirror-like materials that reflects texture from the environ-
ment lighting. To address this ambiguity, we bias the op-
timization to attribute texture details to specular reflections
in the case of mirror-like materials by adding a loss term La
that encourages a variance reduction in diffuse albedo in re-
gions where we observe low roughness and high metallicity.
Lastly, regions with high specularity typically suffer from
larger geometry errors, and thus higher material estimation
errors. We therefore, include a loss term that prefers higher
specularity in regions with large reconstruction errors.
5.2. Intrinsic Image Decomposition Prior
While segmentation improves consistency in roughness and
metallicity over multiple viewpoints, it is not suited for
diffuse albedo which often contains high-frequency tex-
ture variations that are difficult to estimate from sparse
viewpoints. In the absence of appropriate regularization,
the model will overfit diffuse albedo and fail to disentan-
gle lighting effects from intrinsic albedo, which ultimately
leads to a subpar relighting performance.
Recent advances in monocular Intrinsic Image Decom-
position (IID) [12, 40] enable high quality diffuse albedo
decompositions, albeit potentially view-inconsistent.
We
therefore propose to leverage the IID to regularize the dif-
fuse albedo estimation while accounting for potential view-
inconsistencies:
LIID = β(τ) · L2(Ai, ˆAi),
(9)
where ˆAi is the IID diffuse albedo for the captured image
Ii obtained using RGB-X [40]. To mitigate the influence of
view-inconsistency in the IID, we weight the loss by β(τ)
(a lindear decrease) which depends on the current iteration
step τ, thereby relaxing the influence of the IID diffuse
6

<!-- page 7 -->
Ref-GS [ICLR’25]
GAINS (Ours)
Intrinsics
Inputs
Relighting
…
GI-GS [ICLR’25]
GIR [T-PAMI’25]
Figure 4. Qualitative comparison of intrinsic estimation and relighting on the sedan scene from the Ref-Real dataset [29] recon-
structed from 8 views. Column 1 shows novel-view intrinsic renderings in a 2×2 layout: (top) rendered albedo and surface normals,
(bottom) specular roughness and metallicity. Columns 2–4 show relighting results under three different environment maps from novel
viewpoints. GAINS recovers significantly more accurate intrinsics, enabling more realistic relighting.
albedo as the model converges. We opt against using addi-
tional IID supervision for specular roughness and metalicity
as we found these parameters to be less robust and exhibit
more inconsistencies across views
5.3. Multi-illuminated Score Distillation Sampling
While both segmentation guidance and IID guidance im-
prove the accuracy of the specular and diffuse material
properties respectively, they do not necessarily provide
good generalization to novel viewpoints or lighting. To ad-
dress these issues, we take inspiration from our usage of
SDS in the first stage to aid in reducing unrealistic artifacts
on novel viewpoints, and introduce a modified SDS loss to
improve view and lighting-consistency for material proper-
ties. To avoid overfitting to the learned environment map L,
we render relit images Rmat(Cj, G, M, El) from a novel
view Cj under randomly selected lighting from a predeter-
mined set {El}|E|
l=1, and define the SDS loss as:
LMI−SDS = Et,ϵ
h
w(t)||(ϵϕ(Zl
j; t) −ϵ)||2
2
i
,
(10)
where Zl
j = αtRmat(Cj, G, M, El) + σtϵ is the noisy dif-
fusion latent of the rendered scene under novel view and
lighting. Similarly as in the geometry reconstruction stage,
we start the diffusion guidance once the optimization solu-
tion has stabilized (i.e., after 3000 steps).
7

<!-- page 8 -->
NVS
Albedo
Roughness
Synthetic4Relight (8 views)
NVS
Albedo
Roughness
Ours
Ref-GS
GI-GS
GT
NVS
Albedo
Roughness
Figure 5. Qualitative comparison of albedo and roughness estimation and novel-view synthesis (NVS) on Synthetic4Relight [44]
dataset trained with 8 views. While all methods produce reasonable NVS, our method’s estimates significantly better albedo and rough-
ness than GI-GS and Ref-GS that overfit to limited training views and fail to disentangle reflectance from lighting.
5.4. Final Loss
Each loss contributes to a specific enhancement of the ma-
terial estimation process: parameter accuracy, multi-view
consistency, or novel views and lighting generalization. We
combine all losses as:
Lmat = Lcolor + λICCLICC + λIIDLIID
+ λMI−SDSLMI−SDS + λT V LT V ,
where λICC = 0.1, λIID = 1, λMI−SDS = 0.0001 and
λT V = 1. LT V represents a total variation loss on our
reconstructed lighting, acting as a smoothing term. Fig. 2
(right) summarizes the full Stage II pipeline.
6. Evaluation
Evaluation Framework.
We compare GAINS against
state-of-the-art IR frameworks that use Gaussian Splatting
for both objects and scenes: Ref-GS [36] and GI-GS [3]
over two synthetic datasets (TensorIR [10] for NVS, albedo
and relighting; and Synthetic4Relight [44] for NVS, albedo,
relighting and roughness), and a real-world dataset (Ref-
Real [29]), on which we quantitatively evaluate NVS per-
formance only. We also evaluated GIR [26] on real scenes
from Ref-Real, but we found that it fails to reconstruct
meaningful geometry and reflectance (Fig.4) while requir-
ing long per-scene training time (over 7 hours).
There-
fore, we opted not to include GIR in the evaluation over the
synthetic datasets. We evaluate NVS, albedo, and relight-
ing using three metrics: PSNR, SSIM and LPIPS, while
for roughness we employ MSE. We employ scale invari-
ant losses to evaluate albedo of all approaches by uniformly
scaling each albedo map to minimize MSE w.r.t. ground-
truth before applying an error metric. Unless specified oth-
erwise, all evaluations are performed with 8 sparse views
uniformly sampled from the dense views.
Performance Evaluation. Tab.1, and 2 quantitatively sum-
marize the comparison for the three datasets. In general,
GAINS yields more accurate albedo maps and relighting
across both TensorIR and Synthetic4Relight compared to
all prior methods. Additionally, we demonstrate that our
roughness estimations also surpass current methods. Fur-
thermore, GAINS achieves overall better results for NVS
than all competing methods.
However, error metrics do
not always capture important visual differences.
There-
fore, we also provide qualitative comparisons in Fig. 1, 4,
and 5. From the qualitative comparison, especially on the
real-world Ref-Real dataset (Fig.1 and 4), we note less bak-
ing of specular reflections, and generally more accurate re-
lighting results, e.g., the reflections of the ground (missing
for Ref-GS) and the sky (baking of captured reflection for
both Ref-GS and GI-GS) on the ball in Fig.1. Furthermore,
while GI-GS almost slightly competes in NVS against our
method, we observe that it produces less accurate shape and
8

<!-- page 9 -->
Syn4R
NVS - PSNR↑
Albedo - PSNR↑
Relighting - PSNR↑
Roughness - MSE↓
Number of views
Number of views
Number of views
Number of views
Figure 6. Comparison of our method with Ref-GS [36] and GI-GS [3] on the Synthetic4Relight [44] dataset for an increasing
numbers of input views. The figure is organized into: PSNR for novel-view synthesis (top left), albedo (top right), and relighting (bottom
left), and MSE for roughness (bottom right). Each bar chart shows results for 4, 8, 16, and 32 input views, with blue indicating Ref-GS,
orange indicating GI-GS, and green indicating GAINS (ours). GAINS consistently surpasses both baselines in sparse-view settings (4-8
views) and remains competitive as the number of views increases.
GAINS (Ours)
Ref-Gs [ICLR’25]
GI-GS [ICLR’25]
8 views
4 views
16 views
32 views
Figure 7. Relighting and material estimation comparison for varying number of views on the gardenspheres scene from the Ref-Real
[29] dataset. We visualize for each method (rows: Ref-GS [36], GI-GS [3], and GAINS (ours)) relit results for 4, 8, 16, and 32 input views.
For each result, we show the relit result under the bridge environment map, as well as (from left to right) the estimated albedo, metallic,
and roughness. GAINS consistently produces robust relighting and material estimations even from sparse views. In constrast competing
methods struggle to recover accurate materials, yielding degraded relighting quality.
9

<!-- page 10 -->
normals (e.g., the noisy normals on the car in the sedan
scene in Fig.4), indicating that the GI-GS NVS performance
is mainly due to overfitting. As shown in Fig.5, our method
demonstrates a clear advantage in material estimation under
sparse-view settings. While Ref-GS suffers from floaters
and GI-GS typically recovers reasonable geometry but fails
to produce accurate or stable albedo and roughness maps,
our approach, supported by learning-based priors, delivers
consistently superior NVS quality as well as more reliable
albedo and roughness estimations across all scenes.
View-dependent Performance. In addition to our com-
prehensive quantitative and qualitative evaluation on both
synthetic and real datasets, we further analyze the behav-
ior of our method as the number of available input views
increases. Fig. 6 summarizes the quantitative comparison
for PSNR on NVS, albedo, relighting, and roughness on the
Synthetic4Relight [44] dataset. We observe that in sparse-
view scenarios (4-8 views), our method consistently outper-
forms all baselines across all evaluation scenarios. As the
number of views increases (16-32), Ref-GS in particular
achieves better performance in NVS and relighting; how-
ever, our method continues to produce competitive results
within a small margin. Notably, while all methods benefit
from additional views, GAINS starts from a more stable and
reliable baseline in the extremely sparse-view regime, indi-
cating stronger robustness under challenging conditions. A
closer inspection of the roughness estimation reveals that
our method maintains stable and consistent roughness pre-
dictions regardless of view count due to segmentation guid-
ance. In contrast, techniques such as Ref-GS, which im-
pose no spatial constraints, eventually surpass our perfor-
mance at higher view counts by capturing finer-grained
roughness estimations that segmentation-based guidance is
unable to capture.
This illustrates a minor trade-off be-
tween robustness in sparse-view settings and the potential
for high-frequency detail recovery under dense supervision.
Complementing our quantitative experiments, we also pro-
vide qualitative relighting and material estimation results
(albedo, metallicity, and roughness) in Fig.7 on the garden-
spheres scene from the Ref-Real [29] dataset. Across all
view counts, GAINS demonstrates strong and stable mate-
rial reconstruction, producing accurate specular reflections
and consequently superior relighting performance. As re-
flected quantitatively in the roughness bar chart, this trend
is also clearly visible in the qualitative results. Specifically,
our roughness and metallic estimates remain stable and ro-
bust, exhibiting only minor improvements as the number of
input views increases. This further demonstrates GAINS’
advantage in low-view settings, where competing methods
often suffer from noisy, irregular, or unstable material re-
constructions. Improvements observed under dense inputs
mainly stem from enhanced geometry estimation and im-
proved lighting recovery. In contrast competing methods
exhibit pronounced shape degradation in low-view settings
(4-8 views).
Ablation. We conduct an ablation study on the TensoIR
[10] dataset using 8 input views with a best-crop masking
strategy to ensure that metrics reflect only the reconstructed
foreground regions. We focus our ablation on the different
learning-based priors that constitute the material estimation
stage (II): segmentation guidance, IID guidance, and the
diffusion prior. In addition, we also include a comparison to
baseline that includes only the additional losses for shape,
but not for material estimation. From Tab.3, we observe that
the IID component improves the accuracy of the albedo es-
timation and relighting quality, but compared to segmenta-
tion guidance or the diffusion prior, NVS quality degrades.
Segmentation guidance improves NVS performance. The
diffusion prior by itself improves all components, albeit to
a lesser degree than IID for albedo and relighting. How-
ever, when combining all three components, we see the
overall best performance. Fig. 3 provides further qualita-
tive evidence of these effects. Here we demonstrate how
in the absence of learning-based priors, our method fails to
recover reliable reflectance properties, with metallicity and
roughness being especially inaccurate. When the IID prior
is omitted, specular responses become significantly dimin-
ished, while removing segmentation guidance prevents the
model from learning consistent material assignments across
object regions.
7. Conclusion
We introduced GAINS, a Gaussian-based inverse rendering
framework tailored for sparse-view settings. By combining
segmentation, IID, and diffusion priors within a two-stage
optimization pipeline, GAINS effectively stabilizes geome-
try and material estimation. Extensive experiments on syn-
thetic and real-world datasets show that GAINS achieves
state-of-the-art relighting accuracy and competitive novel-
view synthesis compared to prior Gaussian-based IR meth-
ods. Qualitative results further demonstrate improved ma-
terial–lighting separation and reduced reflection baking.
GAINS highlights the power of synergizing complementary
learning priors for physically consistent inverse rendering
under sparse observations.
Acknowledgments:
This work is partially supported
by a National Institute of Health (NIH) NIBIB project
#R21EB035832 and #R21EB037440. Patrick Noras is sup-
ported by the DAAD-PROMOS scholarship during their re-
search stay as a visiting scholar at the University of North
Carolina at Chapel Hill.
10

<!-- page 11 -->
References
[1] Brent
Burley
and
Walt
Disney
Animation
Studios.
Physically-based shading at disney.
In Acm siggraph,
pages 1–7. vol. 2012, 2012. 3
[2] Rohan Chacko, Nicolai H¨ani, Eldar Khaliullin, Lin Sun,
and Douglas Lee.
Lifting by gaussians: A simple, fast
and flexible method for 3d instance segmentation. In 2025
IEEE/CVF Winter Conference on Applications of Computer
Vision (WACV), pages 3497–3507. IEEE, 2025. 5
[3] Hongze Chen, Zehong Lin, and Jun Zhang. Gi-gs: Global
illumination decomposition on gaussian splatting for inverse
rendering. In ICLR, 2025. 1, 2, 3, 5, 8, 9, 7, 10, 11
[4] Paul Debevec, Chris Tchou, Andrew Gardner, Tim Hawkins,
Charis Poullis, Jessi Stumpfel, Andrew Jones, Nathaniel
Yun, Per Einarsson, Therese Lundgren, et al.
Estimating
surface reflectance properties of a complex scene under cap-
tured natural illumination. 2004. 2
[5] Chun Gu, Xiaofei Wei, Zixuan Zeng, Yuxuan Yao, and Li
Zhang. IRGS: Inter-Reflective Gaussian Splatting with 2D
Gaussian Ray Tracing, 2025. 2
[6] Jon Hasselgren, Nikolai Hofmann, and Jacob Munkberg.
Shape, Light, and Material Decomposition from Images us-
ing Monte Carlo Rendering and Denoising. 2
[7] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In SIGGRAPH 2024 Conference Papers.
Association for Computing Machinery, 2024. 3
[8] Han Huang, Yulun Wu, Chao Deng, Ge Gao, Ming Gu,
and Yu-Shen Liu. Fatesgs: Fast and accurate sparse-view
surface reconstruction using gaussian splatting with depth-
feature consistency. In Proceedings of the AAAI Conference
on Artificial Intelligence, 2025. 2, 3, 4
[9] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaox-
iao Long, Wenping Wang, and Yuexin Ma. Gaussianshader:
3d gaussian splatting with shading functions for reflective
surfaces. arXiv preprint arXiv:2311.17977, 2023. 2
[10] Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang, Song-
fang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu, and Hao
Su. Tensoir: Tensorial inverse rendering. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2023. 2, 5, 8, 10, 1, 3, 7, 9, 11
[11] Brian Karis and Epic Games. Real shading in unreal engine
4. Proc. Physically Based Shading Theory Practice, 4(3):1,
2013. 3
[12] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Met-
zger, Rodrigo Caye Daudt, and Konrad Schindler. Repurpos-
ing diffusion-based image generators for monocular depth
estimation. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2024. 4,
6
[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2, 3
[14] Zhengqin Li,
Mohammad Shafiei,
Ravi Ramamoorthi,
Kalyan Sunkavalli, and Manmohan Chandraker.
Inverse
Rendering for Complex Indoor Scenes: Shape, Spatially-
Varying Lighting and SVBRDF From a Single Image. In
2020 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 2472–2481, Seattle, WA,
USA, 2020. IEEE. 2, 3
[15] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and Kui Jia.
Gs-ir: 3d gaussian splatting for inverse rendering. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21644–21653, 2024. 2
[16] Daniel Lichy, Jiaye Wu, Soumyadip Sengupta, and David W
Jacobs. Shape and material capture at home. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 6123–6133, 2021. 2
[17] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tok-
makov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3:
Zero-shot one image to 3d object, 2023. 4
[18] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng
Wang, Lingjie Liu, Taku Komura, and Wenping Wang.
NeRO: Neural Geometry and BRDF Reconstruction of Re-
flective Objects from Multiview Images.
ACM Trans.
Graph., 42(4):114:1–114:22, 2023. 2
[19] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis, 2020. 2
[20] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V.
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Rus-
sell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-
Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nico-
las Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou,
Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bo-
janowski. Dinov2: Learning robust visual features without
supervision, 2023. 5
[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PmLR, 2021. 5
[22] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junt-
ing Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-
Yuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feicht-
enhofer. Sam 2: Segment anything in images and videos.
arXiv preprint arXiv:2408.00714, 2024. 5
[23] Yoichi Sato, Mark D Wheeler, and Katsushi Ikeuchi. Object
shape and reflectance modeling from observation. In Pro-
ceedings of the 24th annual conference on Computer graph-
ics and interactive techniques, pages 379–387, 1997. 2
[24] Soumyadip Sengupta, Angjoo Kanazawa, Carlos D. Castillo,
and David W. Jacobs. SfSNet: Learning Shape, Reflectance
and Illuminance of Faces ‘in the Wild’. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion, pages 6296–6305, 2018. 2, 3
[25] Soumyadip Sengupta, Jinwei Gu, Kihwan Kim, Guilin Liu,
David W Jacobs, and Jan Kautz. Neural inverse rendering
11

<!-- page 12 -->
of an indoor scene from a single image.
In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 8598–8607, 2019. 2, 3
[26] Yahao Shi, Yanmin Wu, Chenming Wu, Xing Liu, Chen
Zhao, Haocheng Feng, Jian Zhang, Bin Zhou, Errui Ding,
and Jingdong Wang. Gir: 3d gaussian inverse rendering for
relightable scene factorization. IEEE Transactions on Trans-
actions on Pattern Analysis and Machine Intelligence, 2025.
2, 3, 8, 1, 5
[27] Pratul P Srinivasan,
Boyang Deng,
Xiuming Zhang,
Matthew Tancik, Ben Mildenhall, and Jonathan T Barron.
Nerv: Neural reflectance and visibility fields for relighting
and view synthesis. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
7495–7504, 2021. 2
[28] Cheng Sun, Guangyan Cai, Zhengqin Li, Kai Yan, Cheng
Zhang, Carl Marshall, Jia-Bin Huang, Shuang Zhao, and
Zhao Dong. Neural-PBIR Reconstruction of Shape, Mate-
rial, and Illumination. In 2023 IEEE/CVF International Con-
ference on Computer Vision (ICCV), pages 18000–18010,
Paris, France, 2023. IEEE. 2
[29] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler,
Jonathan T. Barron, and Pratul P. Srinivasan.
Ref-NeRF:
Structured view-dependent appearance for neural radiance
fields. CVPR, 2022. 2, 5, 6, 7, 8, 9, 10, 1
[30] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Zi-
wei Liu. Sparsenerf: Distilling depth ranking for few-shot
novel view synthesis. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 9065–9076,
2023. 3
[31] Haoyuan Wang, Wenbo Hu, Lei Zhu, and Rynson W.H.
Lau. Inverse Rendering of Glossy Objects via the Neural
Plenoptic Function and Radiance Fields. In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 19999–20008, Seattle, WA, USA, 2024.
IEEE. 2
[32] Jiaye Wu, Saeed Hadadan, Geng Lin, Matthias Zwicker,
David Jacobs, and Roni Sengupta. Gani: Global and near
field illumination aware neural inverse rendering.
arXiv
preprint arXiv:2403.15651, 2024. 2
[33] Rui Xia, Yue Dong, Pieter Peers, and Xin Tong. Recover-
ing shape and spatially-varying surface reflectance under un-
known illumination. ACM Transactions on Graphics (ToG),
35(6):1–12, 2016. 2
[34] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay,
Pradyumna Chari, and Achuta Kadambi. Sparsegs: Real-
time 360° sparse view synthesis using gaussian splatting.
Arxiv, 2023. 2, 3, 4
[35] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi
Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Gaussianob-
ject: High-quality 3d object reconstruction from four views
with gaussian splatting. ACM Transactions on Graphics, 43
(6), 2024. 2, 3, 4
[36] Yuxuan Yao, Zixuan Zeng, Chun Gu, Xiatian Zhu, and Li
Zhang. Reflective gaussian splatting. arXiv preprint, 2024.
1, 2, 3, 5, 8, 9, 7, 10, 11
[37] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi
Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang
Wang. Gaussiandreamer: Fast generation from text to 3d
gaussians by bridging 2d and 3d diffusion models. In CVPR,
2024. 4
[38] Ye Yu and William A. P. Smith. Inverserendernet: Learn-
ing single image inverse rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2019. 2
[39] Chong Zeng, Guojun Chen, Yue Dong, Pieter Peers, Hongzhi
Wu, and Xin Tong. Relighting neural radiance fields with
shadow and highlight hints. In ACM SIGGRAPH 2023 Con-
ference Proceedings, pages 1–11, 2023. 2
[40] Zheng Zeng, Valentin Deschaintre, Iliyan Georgiev, Yannick
Hold-Geoffroy, Yiwei Hu, Fujun Luan, Ling-Qi Yan, and
Miloˇs Haˇsan. Rgb↔x: Image decomposition and synthesis
using material- and lighting-aware diffusion models, 2024. 6
[41] Jingyang Zhang, Yao Yao, Shiwei Li, Jingbo Liu, Tian Fang,
David McKinnon, Yanghai Tsin, and Long Quan. NeILF++:
Inter-Reflectable Light Fields for Geometry and Material Es-
timation. In 2023 IEEE/CVF International Conference on
Computer Vision (ICCV), pages 3578–3587, Paris, France,
2023. IEEE. 2
[42] Tianyuan Zhang, Zhengfei Kuang, Haian Jin, Zexiang Xu,
Sai Bi, Hao Tan, He Zhang, Yiwei Hu, Milos Hasan,
William T Freeman, et al. Relitlrm: Generative relightable
radiance for large reconstruction models.
arXiv preprint
arXiv:2410.06231, 2024. 3
[43] Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul De-
bevec, William T Freeman, and Jonathan T Barron. Ner-
factor: Neural factorization of shape and reflectance under
an unknown illumination. ACM Transactions on Graphics
(ToG), 40(6):1–18, 2021. 2
[44] Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei
Jia, and Xiaowei Zhou. Modeling indirect illumination for
inverse rendering. In CVPR, 2022. 2, 5, 8, 9, 10, 1
[45] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang.
Fsgs:
Real-time few-shot view synthesis using gaussian
splatting, 2023. 3, 4
12

<!-- page 13 -->
GAINS: Gaussian-based Inverse Rendering from Sparse Multi-View Captures
Supplementary Material
A. Overview of Appendices
We categorize our appendices in the following way:
• Sec. B provides additional details of our experimental
setup, including the computational resources used and
factors influencing the performance of our method.
• Sec. C presents additional visual results on the Ten-
soIR [10] and Ref-Real [29] datasets, along with interme-
diate segmentation maps produced during training. We
include reconstructions to highlight the stability and ro-
bustness of our method.
B. Experiment, and Performance Details
Unless stated otherwise, all experiments were conducted on
a single NVIDIA RTX A4500 GPU with 20 GB of VRAM.
Our method operates in linear color space; datasets pro-
vided in sRGB are internally linearized, and final predic-
tions are converted back to sRGB before computing the cor-
responding loss terms. This ensures that all color-dependent
computations remain physically consistent while metrics re-
main comparable to prior work.
Training efficiency depends on two factors: the timing of
the diffusion-prior activation and the number of views used
during the intermediate segmentation-lifting stage. In our
default configuration, the average training time per scene
is approximately 1.5 hours. Notably, segmentation quality
has an impact on the performance of the Gaussian represen-
tation. While using more views for the iterative segmen-
tation merging improves consistency and reduces object-
level ambiguities, it also increases computational overhead:
1min for 25 views and 5mins for 100 views. This trade-
off becomes particularly important in novel-view synthe-
sis (NVS) and relighting tasks, where inaccurate or overly
coarse segmentation (e.g., multiple materials merging into a
single region or a single region being split up into multiple
segmentation classes) can degrade the final reconstruction.
This performance decrease is however not severe and thus
for all conducted experiments a total of 100 novel render
viewpoints are leveraged to create a robust segmentation.
C. Additional Results
We present additional qualitative results to further validate
the effectiveness of our method. Fig.8 and 9 provide qual-
itative comparisons on the TensoIR dataset [10], specifi-
cally on the hotdog and lego scenes. Our method demon-
strates stronger robustness and improved fidelity relative to
existing Gaussian-based inverse-rendering methods. Fig.10
shows reconstructed segmentation maps for several repre-
sentative scenes, highlighting the structural coherence of
our lifted segmentation across viewpoints. Following the
demonstration of our segmentation results, we visualize the
impact of the number of views used for the segmentation
lifting and merging procedure in Fig. 11, highlighting the
loss of material accuracy that is coupled to the quality of the
recovered segmentation. Fig. 12 includes the toycar scene
from the Ref-Real [29] dataset and compares our method
to current baselines, including Ref-GS [36], GI-GS [3], and
GIR [26]. Our method produces more faithful material de-
composition and relighting behavior while reducing visual
artifacts present in competing approaches.
Furthermore,
Fig. 13 extends the qualitative comparison on the toycar
scene by illustrating the impact on material and relighting
performance with 4 and 16 views. We observe how our
method degrades less in quality when decreasing the num-
ber of input views, while current Gaussian-based inverse-
rendering techniques demonstrate notable quality reduction.
Fig. 14, 15, 16 and 17 extend the prior discussion on re-
lighting performance by providing an extensive qualitative
comparison across all scenes of the TensorIR [10] dataset
under five different novel lighting conditions. Across all
scenes and environment maps, our method consistently pro-
duces the most faithful and visually coherent relighting re-
sults. Ref-GS is often affected by floaters. Furthermore,
Ref-GS also produces visually overly flat and muted re-
lighting outputs.
GI-GS, in contrast, struggles with in-
correctly reconstructed high-frequency regions, leading to
highly inconsistent lighting effects, an issue likely rooted
in its weaker normal prediction and less accurate geome-
try reconstruction. Fig. 18 presents additional bar charts
to illustrate the view-dependent performance on the Ten-
sorIR [10] dataset.
The observed trends closely mirror
those seen in our earlier Synthetic4Relight [44] experi-
ments.
Specifically, in sparse-view settings (4-8 views),
our method surpasses all baselines across nearly all met-
rics for NVS, albedo, and relighting. As the number of
input views increases (16-32), Ref-GS begins to slightly
outperform our method in selected metrics, although the
margin remains small and our approach continues to com-
pete strongly. Overall, while all methods benefit from ad-
ditional views, ours demonstrates a clear advantage in ex-
tremely sparse-view scenarios by starting from a substan-
tially stronger and more stable baseline.
1

<!-- page 14 -->
Ours
Ref-GS
GI-GS
GT
NVS
Normal
Albedo
Relight 
(Fireplace)
tensorIR: Hotdog (8 views)
Relight 
(Bridge)
PSNR: 7.94 
SSIM: 0.71
LPIPS: 0.29
PSNR: 8.39 
SSIM: 0.71
LPIPS: 0.255
PSNR: 23.13 
SSIM: 0.87
LPIPS: 0.15
PSNR: 19.99 
SSIM: 0.79
LPIPS: 0.185
PSNR: 23.93 
SSIM: 0.9
LPIPS: 0.111
PSNR: 22.06 
SSIM: 0.87
LPIPS: 0.115
PSNR: 2.079 
SSIM: 0.19
LPIPS: 0.53
PSNR: 7.4 
SSIM: 0.71
LPIPS: 0.234
PSNR: 22.08
SSIM: 0.87
LPIPS: 0.162
PSNR: 15.56 
SSIM: 0.91
LPIPS: 0.111
PSNR: 25.18 
SSIM: 0.92
LPIPS: 0.066
PSNR: 27.19 
SSIM: 0.93
LPIPS: 0.066
MAE: 26.12
MAE: 25.04
MAE: 17.17
Figure 8. Comparison of our method with Ref-GS [36] and GI-GS [3] on the hotdog scene from the TensoIR [10] dataset, recon-
structed from 8 input views. From top to bottom, the rows show the ground-truth reference, Ref-GS, GI-GS and GAINS (our method).
The columns present, from left to right: NVS renderings, predicted surface normals, albedo reconstruction, and relighting results. To com-
pensate for the albedo-lighting intensity ambiguity, we multiply each method’s relighting results by a global scale factor that minimizes
the MSE error with respect to the ground truth. For the hotdog scene, relighting is performed under the fireplace and bridge environment
map. Our method yields cleaner geometry with significantly fewer floaters, producing sharper and more coherent surface normals. This
improved geometric consistency, coupled with our segmentation-driven material recovery, results in more stable and visually persuasive
relighting when compared to Ref-GS and GI-GS. Overall, our approach achieves competitive or improved results across all evaluation axes
(NVS, normals, albedo, and relighting), demonstrating the effectiveness of our reconstruction pipeline in low-view settings.
2

<!-- page 15 -->
Ours
Ref-GS
GI-GS
GT
NVS
Normal
Albedo
tensorIR: Lego (8 views)
Relight 
(Courtyard)
Relight 
(City)
PSNR: 4.26 
SSIM: 0.43
LPIPS: 0.35
PSNR: 4.39 
SSIM: 0.43
LPIPS: 0.363
PSNR: 17.01 
SSIM: 0.77
LPIPS: 0.179
PSNR: 19.99 
SSIM: 0.79
LPIPS: 0.185
PSNR: 21.77 
SSIM: 0.84
LPIPS: 0.133
PSNR: 20.6 
SSIM: 0.83
LPIPS: 0.131
PSNR: 3.945 
SSIM: 0.35
LPIPS: 0.474
PSNR: 4.17 
SSIM: 0.46
LPIPS: 0.336
PSNR: 20.84
SSIM: 0.84
LPIPS: 0.168
PSNR: 17.73 
SSIM: 0.84
LPIPS: 0.14
PSNR: 25.13 
SSIM: 0.88
LPIPS: 0.155
PSNR: 23.77 
SSIM: 0.88
LPIPS: 0.108
MAE: 64.2
MAE: 53.41
MAE: 61.18
Figure 9. Comparison of our method with Ref-GS [36] and GI-GS [3] on the lego scene from the TensoIR [10] dataset, reconstructed
from 8 input views. From top to bottom, the rows show the ground-truth reference, Ref-GS, GI-GS and GAINS (our method). The columns
present, from left to right: NVS renderings, predicted surface normals, albedo reconstruction, and relighting results. To compensate for
the albedo-lighting intensity ambiguity, we multiply each method’s relighting results by a global scale factor that minimizes the MSE error
with respect to the ground truth. For the lego scene, relighting is shown under the courtyard and city illumination. Similar to our previous
comparison on the hotdog scene, our method produces robust geometry with little floaters. In combination with our segmentation-driven
material recovery, results are more stable and visually superior in context of relighting when compared to Ref-GS and GI-GS.
3

<!-- page 16 -->
GT
Segmentation
Figure 10. Reconstructed segmentation maps generated by our iterative lifting procedure. The first row shows the reference ground
truth images, while the second row displays our rendered segmentation maps. For each scene, we render 100 novel viewpoints in an orbital
trajectory around the object to perform iterative segmentation lifting and Gaussian-object merging. Despite the challenging setting of using
only 8 input training views, our method produces coherent and largely accurate segmentations, with only minor missegmentation in a few
regions.
Segmentation
Metallic
Roughness
100 views
50 views
25 views
Sampled Views
Ref-Real: sedan (8 views)
Figure 11. Impact of the number of novel views used for the segmentation lifting and merging mechanism. Each row presents the
resulting segmentation render in column 1, the metallic map in column 2, and the roughness map in column 3. From top to bottom, we
show results using 100, 50, and 25 novel views utilized for lifting and merging of gaussian objects. As the number of views decreases, some
regions become noticeably over-segmented, leading to increased noise and irregularities in the estimated material properties, defeating the
purpose of our proposed segmentation guidance. We also observe that the resulting segmentation consistency relies on the reconstructed
geometry from stage 1, demonstrating that a sufficient shape estimation is crucial for material recovery.
4

<!-- page 17 -->
Ref-GS [ICLR’25]
GAINS (Ours)
Intrinsics
Inputs
Relighting
…
GI-GS [ICLR’25]
GIR [T-PAMI’25]
Figure 12. Qualitative comparison of intrinsic estimation and relighting on the sedan scene from the Ref-Real dataset [29], recon-
structed from 8 views. The rows show results from top to bottom for: GIR [26], Ref-GS [36], GI-GS [3], and GAINS (Ours). Column 1
presents novel-view intrinsic renderings in a 2 × 2 layout: (top) albedo and surface normals, (bottom) specular roughness and metallic-
ity. Columns 2–4 show relighting results under three different novel environment maps. Across all lighting conditions, GAINS produces
more stable and realistic relighting, consistent shading, and fewer artifacts. Our method also recovers more accurate and smoother surface
normals, which directly improves shading behavior and enables physically meaningful relighting across viewpoints. The results further
highlight the importance of high-quality geometry reconstruction. GAINS exhibits far less noise in the recovered material maps, yielding
cleaner intrinsics and significantly more faithful relit appearances across all three environment maps.
5

<!-- page 18 -->
4 views
GAINS (Ours)
Ref-Gs [ICLR’25]
GI-GS [ICLR’25]
16 views
GAINS (Ours)
Ref-Gs [ICLR’25]
GI-GS [ICLR’25]
Figure 13. Qualitative comparison on the toycar scene trained with different numbers of input views (top: 4 views, bottom: 16
views). Results for the 8-view setting are provided in Fig.12. Each row corresponds to a method: GAINS (ours), Ref-GS, and GI-GS.
The first column shows intrinsic predictions arranged in a 2 × 2 grid (top-left: albedo, top-right: normals, bottom-left: metallic, bottom-
right: roughness), while the remaining columns show relighting under novel environment maps. Our method maintains accurate intrinsic
reconstruction and stable relighting quality across all training-views, whereas baseline methods degrade noticeably, particularly under
sparse inputs.
6

<!-- page 19 -->
Ours
Ref-Gs 
GI-GS 
GT
Forest
Fireplace
Bridge
Courtyard
City
PSNR: 8.23 
SSIM: 0.72 
LPIPS: 0.277
PSNR: 7.94 
SSIM: 0.71
LPIPS: 0.29
PSNR: 8.39 
SSIM: 0.71
LPIPS: 0.255
PSNR: 7.76 
SSIM: 0.67
LPIPS: 0.27
PSNR: 8.32 
SSIM: 0.71
LPIPS: 0.255
PSNR: 21.07 
SSIM: 0.81 
LPIPS: 0.16
PSNR: 23.13 
SSIM: 0.87
LPIPS: 0.15
PSNR: 19.99 
SSIM: 0.79
LPIPS: 0.185
PSNR: 20.36 
SSIM: 0.83
LPIPS: 0.166
PSNR: 19.57 
SSIM: 0.81
LPIPS: 0.185
PSNR: 21.94 
SSIM: 0.88 
LPIPS: 0.12
PSNR: 23.93 
SSIM: 0.9
LPIPS: 0.111
PSNR: 22.06 
SSIM: 0.87
LPIPS: 0.115
PSNR: 21.26 
SSIM: 0.87
LPIPS: 0.116
PSNR: 21.09 
SSIM: 0.88
LPIPS: 0.131
Figure 14. Relighting comparison across different lighting conditions on the hotdog scene from the TensoIR [10] dataset. Each
column shows different relighting results. Rows correspond to Ref-GS [36], GI-GS [3], and our method.
7

<!-- page 20 -->
Ours
Ref-Gs 
GI-GS 
GT
Forest
Fireplace
Bridge
Courtyard
City
PSNR: 8.46 
SSIM: 0.67
LPIPS: 0.307
PSNR: 8.64 
SSIM: 0.66
LPIPS: 0.332
PSNR: 8.93 
SSIM: 0.65
LPIPS: 0.313
PSNR: 8.83 
SSIM: 0.63
LPIPS: 0.326
PSNR: 9.28 
SSIM: 0.66
LPIPS: 0.309
PSNR: 17.85 
SSIM: 0.77 
LPIPS: 0.161
PSNR: 18.31 
SSIM: 0.81
LPIPS: 0.176
PSNR: 16.88 
SSIM: 0.75
LPIPS: 0.181
PSNR: 17.9 
SSIM: 0.77
LPIPS: 0.173
PSNR: 16.32 
SSIM: 0.75
LPIPS: 0.181
PSNR: 20.73 
SSIM: 0.84
LPIPS: 0.135
PSNR: 21.56 
SSIM: 0.84
LPIPS: 0.152
PSNR: 20.25 
SSIM: 0.82
LPIPS: 0.149
PSNR: 21.33 
SSIM: 0.81
LPIPS: 0.14
PSNR: 20.47 
SSIM: 0.84
LPIPS: 0.133
Figure 15. Relighting comparison across different lighting conditions on the lego scene from the TensoIR [10] dataset. Each column
shows different relighting results. Rows correspond to Ref-GS [36], GI-GS [3], and our method.
8

<!-- page 21 -->
Ours
Ref-Gs 
GI-GS 
GT
Forest
Fireplace
Bridge
Courtyard
City
PSNR: 2.19 
SSIM: 0.28
LPIPS: 0.467
PSNR: 1.52 
SSIM: 0.17
LPIPS: 0.466
PSNR: 2.01 
SSIM: 0.25
LPIPS: 0.467
PSNR: 2.12 
SSIM: 0.26
LPIPS: 0.453
PSNR: 2.51 
SSIM: 0.31
LPIPS: 0.453
PSNR: 15.84 
SSIM: 0.83 
LPIPS: 0.097
PSNR: 15.83 
SSIM: 0.84
LPIPS: 0.098
PSNR: 15.86 
SSIM: 0.83
LPIPS: 0.097
PSNR: 16.29 
SSIM: 0.83
LPIPS: 0.095
PSNR: 15.69 
SSIM: 0.82
LPIPS: 0.098
PSNR: 18.21 
SSIM: 0.86 
LPIPS: 0.09
PSNR: 17.85 
SSIM: 0.87
LPIPS: 0.09
PSNR: 18.50 
SSIM: 0.87
LPIPS: 0.087
PSNR: 19.03 
SSIM: 0.86
LPIPS: 0.088
PSNR: 18.92 
SSIM: 0.86
LPIPS: 0.082
Figure 16. Relighting comparison across different lighting conditions on the ficus scene from the TensoIR [10] dataset. Each column
shows different relighting results. Rows correspond to Ref-GS [36], GI-GS [3], and our method.
9

<!-- page 22 -->
Ours
Ref-Gs 
GI-GS 
GT
Forest
Fireplace
Bridge
Courtyard
City
PSNR: 27.20 
SSIM: 0.95 
LPIPS: 0.074
PSNR: 25.27 
SSIM: 0.93
LPIPS: 0.076
PSNR: 26.03 
SSIM: 0.94
LPIPS: 0.076
PSNR: 26.51 
SSIM: 0.92
LPIPS: 0.08
PSNR: 26.62 
SSIM: 0.94
LPIPS: 0.074
PSNR: 19.89 
SSIM: 0.89 
LPIPS: 0.103
PSNR: 19.82 
SSIM: 0.90
LPIPS: 0.099
PSNR: 19.72 
SSIM: 0.88
LPIPS: 0.106
PSNR: 19.91 
SSIM: 0.87
LPIPS: 0.107
PSNR: 19.17 
SSIM: 0.86
LPIPS: 0.11
PSNR: 24.09 
SSIM: 0.92 
LPIPS: 0.078
PSNR: 24.44 
SSIM: 0.92
LPIPS: 0.081
PSNR: 24.43 
SSIM: 0.92
LPIPS: 0.079
PSNR: 25.02 
SSIM: 0.91
LPIPS: 0.084
PSNR: 23.98 
SSIM: 0.92
LPIPS: 0.078
Figure 17. Relighting comparison across different lighting conditions on the armadillo scene from the TensoIR [10] dataset. Each
column shows different relighting results. Rows correspond to Ref-GS [36], GI-GS [3], and our method.
10

<!-- page 23 -->
Number of views
Number of views
Number of views
Number of views
Number of views
Number of views
Number of views
Number of views
Number of views
NVS - PSNR↑
NVS - SSIM↑
NVS - LPIPS↓ 
Albedo - PSNR↑
Albedo - SSIM↑
Albedo - LPIPS↓ 
Relighting - PSNR↑
Relighting - SSIM↑
Relighting - LPIPS↓ 
Figure 18. Comparison of our method with Ref-GS [36] and GI-GS [3] on the TensoIR [10] dataset across increasing numbers of
input views. Columns (left to right) report PSNR, SSIM, and LPIPS. Rows correspond to novel-view synthesis (NVS), albedo estimation,
and relighting. Each bar chart shows results for 4, 8, 16, and 32 input views, with blue indicating Ref-GS, orange indicating GI-GS, and
green indicating GAINS (ours). GAINS consistently surpasses both baselines across all metrics in sparse-view settings (4–8 views) and
remains competitive as the number of views increases.
11
