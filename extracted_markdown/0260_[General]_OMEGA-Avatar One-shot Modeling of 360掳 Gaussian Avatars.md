<!-- page 1 -->
OMEGA-Avatar: One-shot Modeling of 360◦Gaussian Avatars
ZEHAO XIA, Chongqing University, China
YIQUN WANG, Chongqing University, China
ZHENGDA LU, University of Chinese Academy of Sciences, China
KAI LIU, Chongqing University, China
JUN XIAO, University of Chinese Academy of Sciences, China
PETER WONKA, KAUST, Saudi Arabia
Fig. 1. Given a single portrait image across diverse styles (left), our method generates a high-fidelity Full-Head Gaussian avatar via a feed-forward network
(middle). The reconstructed avatars include the back of the head and are capable of high-quality animations (right).
Creating high-fidelity, animatable 3D avatars from a single image remains
a formidable challenge. We identified three desirable attributes of avatar
generation: 1) the method should be feed-forward, 2) model a 360◦full-head,
and 3) should be animation-ready. However, current work addresses only
two of the three points simultaneously. To address these limitations, we
propose OMEGA-Avatar, the first feed-forward framework that simultane-
ously generates a generalizable, 360◦-complete, and animatable 3D Gaussian
head from a single image. Starting from a feed-forward and animatable
framework, we address the 360◦full-head avatar generation problem with
two novel components. First, to overcome poor hair modeling in full-head
avatar generation, we introduce a semantic-aware mesh deformation module
that integrates multi-view normals to optimize a FLAME head with hair
while preserving its topology structure. Second, to enable effective feed-
forward decoding of full-head features, we propose a multi-view feature
splatting module that constructs a shared canonical UV representation from
features across multiple views through differentiable bilinear splatting, hier-
archical UV mapping, and visibility-aware fusion. This approach preserves
both global structural coherence and local high-frequency details across all
viewpoints, ensuring 360◦consistency without per-instance optimization.
Extensive experiments demonstrate that OMEGA-Avatar achieves state-of-
the-art performance, significantly outperforming existing baselines in 360◦
full-head completeness while preserving identity across different viewpoints.
Project Website: https://omega-avatar.github.io/OMEGA-Avatar/
CCS Concepts: • Computing methodologies →Reconstruction; Neural
networks; Animation.
Additional Key Words and Phrases: Single-view Feed-forward Generation,
Generalizable Full-head Avatars, Animation-ready, Gaussian Splatting
1
Introduction
Generating high-fidelity, animatable 3D full-head avatars from a
single image is a pivotal challenge in computer graphics.
To make this severely under-constrained problem tractable in
practice, we identified three essential properties that the avatar
generator must possess: First, it should be feed-forward. Due to
the under-constrained nature of single-image input, recovering a
complete and animatable full-head avatar inevitably relies on strong
learned priors from large-scale data. This makes a feed-forward
formulation essential for generalizing across identities, in contrast
to optimization-based approaches that require costly per-subject
tuning. Second, it must support 360◦full-head modeling. Unlike
standard facial reconstruction, a complete avatar requires inferring
, Vol. 1, No. 1, Article . Publication date: February 2026.
arXiv:2602.11693v1  [cs.GR]  12 Feb 2026

<!-- page 2 -->
2
•
Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, and Peter Wonka
occluded geometry and appearance, such as hair and the back of the
head, to ensure geometric completeness. This requirement typically
demands multi-view supervision to hallucinate reasonable content
in unseen regions. Third, the avatar should be animation-ready. Be-
yond static geometry, the model must support explicit control over
expressions and poses. This capability relies on modeling additional
motion-related information through a parametric model or video-
based animation supervision, so that the reconstructed avatar can be
consistently driven by explicit motion parameters while preserving
identity.
However, satisfying all three attributes remains elusive.
Some generative approaches[1, 22, 40] leverage 3D-aware GANs
or diffusion models to achieve 360◦head synthesis. However, they
lack explicit parametric head representations and remain constrained
by inversion-based or sampling-heavy workflows, hindering effi-
cient inference and animatable head generation. While FaceLift [28]
attempts to generate animation frame-by-frame via diffusion, the
stochastic nature of diffusion leads to temporal inconsistency and
jittering. Methods like FATE [50] and SOAP [26] achieve animatable
full heads by leveraging multi-view priors. However, both FATE and
SOAP rely on instance-specific pipelines with costly optimization
or remeshing. As a result, there is room to improve upon these
methods in terms of efficiency and generalizability. Recent one-shot
methods [6, 16] train feed-forward networks using large-scale 2D
video datasets [43]. While they excel at frontal reenactment, they
are unable to effectively decode full-head features in a feed-forward
manner, leading to structural distortions when rendering side and
back views and failing to generate 360◦full-head avatars.
To bridge these gaps, we propose Ω-Avatar, the first feed-forward
framework that generates a 360◦-complete and animatable 3D avatar
from a single image.
Within the feed-forward and animation-ready formulation, we
propose two novel components for tackling full-head avatar gen-
eration as illustrated in Fig. 2. First, to address the difficulty of
modeling complete head geometry under the constraint of limited
3D datasets, we leverage a pre-trained multi-view diffusion model
to augment large-scale video data, producing consistent multi-view
normal maps and RGB images without requiring multi-view capture.
The synthesized normal maps are used as geometric priors to op-
timize a FLAME mesh with hair, enforcing cross-view consistency
while preserving the underlying topology. Second, a key challenge
lies in how to effectively fuse discrete and scale-inconsistent multi-
view features into a unified representation suitable for feed-forward
decoding. To this end, we propose a multi-view feature splatting
mechanism that maps features from generated multiple views into
a shared canonical UV space. This UV-based representation serves
as a differentiable bridge, enabling robust aggregation of global
structure and local high-frequency details across viewpoints. The
resulting UV features are then decoded into 3D Gaussians and bound
to the FLAME mesh via UV mapping. By jointly leveraging these
two aspects, Ω-Avatar enables rapid generation of high-fidelity and
animatable avatars without per-instance optimization.
Our main contributions are as follows:
• We propose the first feed-forward framework that enables the gen-
eration of generalizable, high-fidelity, full-head, and animatable
avatars from a single image.
Table 1. Comparison of properties with state-of-the-art methods. Our
method is the first to simultaneously achieve feed-forward, 360◦full-head,
and animatable avatar reconstruction from a single image.
Method
Representation
Feed-Forward
360◦Full-Head
Animatable
Rodin [40]
Tri-plane
✓
✓
✗
PanoHead [1]
Tri-plane
✗∗
✓
✗
SphereHead [22]
Tri-plane
✗∗
✓
✗
FaceLift [15]
3DGS
✓
✓
✗
FATE [50]
3DGS
✗
✓
✓
GAGAvatar [6]
3DGS
✓
✗
✓
LAM [16]
3DGS
✓
✗
✓
SOAP [26]
Mesh
✗
✓
✓
Ours
3DGS
✓
✓
✓
∗Denotes methods requiring per-instance GAN inversion prior to reconstruction.
• We introduce a multi-view feature splatting module, a fully dif-
ferentiable multi-view feature fusion mechanism that constructs
a shared canonical UV feature map from discrete 2D multi-view
features for 3D Gaussian decoding.
• Extensive experiments demonstrate that our method achieves
state-of-the-art performance on multi-view full-head generation,
outperforming the main baseline (GAGAvatar) in both full-head
completeness and multi-view visual fidelity on both datasets, with
PSNR ↑2.95%, SSIM ↑0.30%, LPIPS ↓7.39%, and DS ↓8.03%.
2
Related work
Single-Image Head Avatar Reconstruction. Early 3DMM-based
approaches [18, 44] provided strong statistical priors but often
lacked fine-grained details. With the advent of Neural Radiance
Fields [32] (NeRF), the field has shifted toward implicit representa-
tions, which offer higher fidelity in modeling complex attributes like
hair [2, 19, 34, 53, 54]. Most NeRF-based methods rely on identity-
specific multi-view or video data, limiting generalization to unseen
subjects and raising privacy concerns. To avoid large-scale video
data, generative approaches [5, 13, 33] and 3D-aware GANs like
EG3D [4] achieve high visual quality via efficient tri-plane repre-
sentations. Reconstructing specific identities requires costly latent
inversion [37, 42], which often degrades accuracy and fails to fully
preserve identity. Addressing the limitations of data requirements
and computational cost, recent works has focused on one-shot 3D
head reconstruction [7, 10, 11, 20, 24, 25, 27, 29, 30, 39, 45–48, 52].
3D Gaussian Splatting (3DGS) [17] offers photorealistic rendering at
real-time speeds, recent one-shot single-image adaptations [6, 16],
represent significant progress. However, these methods often de-
teriorate in performance when rendered from diverse viewpoints,
meaning they are not pure 3D solutions. Our work addresses these
shortcomings by proposing a robust framework for one-shot 3D
single-image animation that maintains high rendering efficiency
and geometric consistency.
Generative Full-Head Reconstruction. Recent full-head model-
ing methods increasingly rely on generative models for novel view
synthesis. GAN-based approaches [1, 22] enable high-quality 360°
head synthesis via specialized volumetric representations. Diffusion-
based approaches [40, 49] generate triplane head representations
but remain static and unsuitable for animation.
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 3 -->
OMEGA-Avatar: One-shot Modeling of 360◦Gaussian Avatars
•
3
Fig. 2. Pipeline Overview. Given the source and target images, we leverage diffusion models to synthesize multi-view RGB images and corresponding normal
maps. These normal maps are used to semantic-aware mesh deformation, while pixel-wise features are extracted from multi-view RGB images. Multi-view
features are subsequently aggregated into a canonical UV feature map through the multi-view feature splatting module. The UV features and vertex features
extracted from the deformed mesh are decoded and anchored to the mesh via UV mapping. For animation, the expression and pose derived from the target
image are injected into the deformed mesh. Finally, the rendered output is enhanced by a neural refiner to generate the final full-head avatar.
SOAP [26] introduces a style-omniscient framework but relies on
costly multi-stage remeshing and direct FLAME vertex optimization,
which alters topology and degrades expression driving. To address
these limitations, we present a novel framework capable of effi-
cient, one-shot full-head reconstruction that supports high-quality
animation.
3
Methodology
3.1
Overview
We present a framework for generating high-fidelity, animatable
full-head Gaussian avatars from a single image, as illustrated in
Fig. 2. We leverage diffusion models to generate RGB images and
normal maps, which guide the semantic-aware deformation of a
FLAME mesh to obtain a personalized mesh. Our approach employs
a dual-branch architecture to generate Gaussians that construct the
3D representation. The UV Gaussian branch encodes appearance
details; we employ DINOv3 [38] with a Feature Pyramid Network
(FPN) to extract pixel-aligned features that fuse deep semantics with
high-frequency geometric details. These multi-view features are
aggregated into a unified UV feature map via splatted UV feature
map module. The vertex Gaussian branch anchors Gaussians directly
to the FLAME vertices to ensure structural coherence.
Finally, the Gaussians from two branches are driven by the under-
lying FLAME, which is deformed by target expression parameters
and refined by a neural renderer to yield final outputs.
In the following subsections. Sec. 3.2 describes the process of
semantic-aware mesh deformation. Sec. 3.3 elaborates on the canon-
ical full-head avatar construction, introducing the specifics of both
the UV and vertex Gaussian branches. Finally, the training strategy
and losses are presented in Sec. 3.4.
3.2
Semantic-aware Mesh Deformation.
We leverage the pretrained diffusion model from SOAP [26], which
builds upon the Unique3D [41] framework, to synthesize six RGB
images I𝑟𝑔𝑏and normal maps I𝑛𝑚𝑙from input.
Given the generated multi-view consistent normal maps I𝑛𝑚𝑙
and facial landmarks detected using [3], we aim to reconstruct a
personalized mesh while maintaining a clean topology suitable for
animation. Unlike previous FLAME tracking methods [36] that rely
solely on landmarks or single-view photometric constraints, our
approach is supervised by multi-view consistent normal maps. This
rich geometric prior allows us to recover intricate 3D details that
are typically lost in parametric fitting. To recover the personalized
geometry, we propose a single-stage optimization framework. In
contrast to prior art [26] that require multi-stage iterative remeshing
to handle deformations, our method efficiently deforms the mesh in
a single pass. We initialize the mesh using the FLAME template and
introduce a deformation field. Let M(𝛽,𝜃,𝜓) denote the standard
FLAME driven by shape 𝛽, pose 𝜃, and expression 𝜓. We optimize a
per-vertex offset ΔV ∈R𝑁×3 to capture high-frequency geometric
details, where 𝑁denotes the number of vertices. Deformed vertices
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 4 -->
4
•
Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, and Peter Wonka
Fig. 3. Semantic-aware Mesh Deformation. Direct optimization with
normal guidance disrupts the parametric structure of FLAME, causing severe
surface irregularities and topological artifacts (left). Note the holes in the
cranial region and the degeneration of facial features such as the eyes and
ears (highlighted in red boxes). Our approach (right) mitigates these issues
by incorporating semantic-aware topology preservation and a semantic-
aware Laplacian. We preserve the clean topology of FLAME and ensure
360◦geometric consistency, enabling the generation of fine details without
compromising facial structural integrity.
V′ are formulated as:
V′ = M(𝛽,𝜃,𝜓) + ΔV.
(1)
Semantic-aware Topology Preservation. A core challenge in mesh
deformation is balancing geometric fidelity with animatability. Un-
constrained optimization of ΔV across the entire mesh often disrupts
the inherent semantic topology of the FLAME, leading to degraded
expression reenactment quality. To address this, we introduce a
semantic-aware optimization strategy. Specifically, we utilize a se-
mantic mask to restrict the scope of the normal-guided deformation.
We apply the normal consistency constraints strictly to the non-
facial regions while preserving the parametric structure of the facial
region. This design disentangles the optimization of static geometry
from dynamic facial components, ensuring high-fidelity animation.
To ensure the deformation is both accurate and topologically
coherent, we optimize ΔV by minimizing the following composite
objective function:
L = 𝜆𝑛𝑚𝑙L𝑛𝑚𝑙+ 𝜆𝑙𝑚𝑘L𝑙𝑚𝑘+ 𝜆𝑙𝑎𝑝L𝑙𝑎𝑝,
(2)
The primary supervision comes from the normal consistency loss
L𝑛𝑚𝑙, which minimizes the 𝐿2 distance between the rendered normal
of the deformed mesh and the diffusion-predicted maps I𝑛𝑚𝑙. To
ensure semantic alignment, we employ a landmark loss L𝑙𝑚𝑘that
penalizes the projection error of 3D facial landmarks and enforces
symmetry in the canonical space.
Semantic-aware Laplacian. Although restricting normal constraints
to non-facial regions helps preserve facial topology, the cranial re-
gion remains susceptible to surface irregularities and topological
artifacts due to the lack of supervision from top-view, as illustrated
in Fig. 3. Standard Laplacian regularization is insufficient here: apply-
ing a uniform weight strong enough to smooth the cranial artifacts
would inadvertently over-smooth the facial features. To resolve this
conflict, we introduce a semantic-aware Laplacian smoothing term
L𝑙𝑎𝑝with a spatially-varying weighting scheme:
Fig. 4. Multi-view Feature Splatting. Taking a frontal view as an example,
we first obtain per-pixel UV coordinates and normals via rasterization, and
map features to UV space using differentiable bilinear splatting. We employ
hierarchical UV mapping, which builds a multi-resolution pyramid to fill
missing regions in a coarse-to-fine manner. Simultaneously, we calculate
a fusion weight map by combining view-dependent confidence and UV
sampling density. Finally, the visibility-aware fusion module aggregates the
weighted features from all views to generate the final UV feature map.
L𝑙𝑎𝑝=
𝑁
∑︁
𝑖=1
𝑤𝑖∥𝛿𝑖∥2,
𝛿𝑖= 𝑣′
𝑖−
1
|𝑁𝑛𝑒𝑖(𝑖)|
∑︁
𝑗∈𝑁𝑛𝑒𝑖(𝑖)
𝑣′
𝑗,
(3)
where 𝛿𝑖is the Laplacian coordinate approximation for vertex 𝑖.
Consistent with our semantic-aware strategy, the weight 𝑤𝑖is not
constant; we assign significantly higher weights to the hair and
boundary regions to suppress degeneration, while assigning mini-
mal weights to the facial region. This ensures the cranium remains
smooth without compromising the high-frequency details of the
face preserved by the parametric prior.
3.3
Canonical Gaussian Full-head Avatar Generation
We model the canonical full-head avatar with a hybrid Gaussian rep-
resentation that combines vertex Gaussians for expression control
and UV Gaussians for view-consistent appearance modeling.
The vertex Gaussian Branch anchors 3D Gaussians to FLAME
vertices to enable explicit expression driving. We extract a global
identity token 𝑓𝑖𝑑from the frontal view, as it provides the most
comprehensive identity information. Simultaneously, to distinguish
different vertices and encode spatial details, we assign a unique
learnable parameter to each vertex 𝑣(𝑖). The global token is con-
catenated with these local parameters and fed into a vertex decoder
to predict the Gaussian attributes Gvert = {𝑟(𝑖),𝑠(𝑖), 𝛼(𝑖),𝑐(𝑖)}. This
design ensures geometric stability during animation.
While the vertex branch ensures structural stability, the limited
number and discrete nature of FLAME vertices restrict vertex Gaus-
sians from representing high-frequency details. To capture appear-
ance while maintaining surface topological continuity, we introduce
the UV Gaussian Branch. Given 𝑘generated multi-view images,
we employ an image encoder to extract semantic feature maps F𝑘.
However, effectively aggregating feature maps from different views
is non-trivial due to view-dependent variations and inconsistent
feature scales across viewpoints. To achieve this, we propose a
Multi-view Feature Splatting module that lifts multi-view 2D
features onto a canonical UV map, enabling joint decoding of all
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 5 -->
OMEGA-Avatar: One-shot Modeling of 360◦Gaussian Avatars
•
5
Fig. 5. Novel view synthesis from single image on the Ava-256 dataset. Compared to state-of-the-art methods, our approach better preserves identity
consistency and high-quality rendering results, even under unseen and extreme side-view facial angles. Note that PanoHead and SphereHead require inputs
aligned to the FFHQ canonical space, which leads to differences in apparent scale. We use the red boxes to highlight the visual artifacts.
views in a single pass. Our approach consists of three integral com-
ponents: differentiable bilinear splatting, hierarchical UV mapping,
and visibility-aware fusion, as shown in Fig. 4. This enables stable
end-to-end gradient propagation from multi-view feature maps to
the UV feature map.
Differentiable Bilinear Splatting. We first establish dense cor-
respondences between screen pixels and UV coordinates. For a
given view 𝑘with camera pose P𝑘, we perform rasterization to re-
trieve a continuous UV coordinate u𝑝∈R2 for each pixel 𝑝in the
feature map. Direct nearest-neighbor assignment is prone to quan-
tization artifacts. Instead, we adopt differentiable bilinear splatting
and formulate it as a differentiable aggregation operator over the UV
domain. Given a pixel feature f𝑝and its continuous UV coordinate
u𝑝, bilinear splatting softly distributes the feature to neighboring
UV grid locations according to a bilinear kernel induced by u𝑝.
Aggregating contributions from all pixels yields a per-view UV
feature map U𝑘and a corresponding density map D𝑘:
U𝑘=
∑︁
𝑝
B(u𝑝) f𝑝,
D𝑘=
∑︁
𝑝
B(u𝑝),
(4)
where B(u𝑝) denotes the bilinear splatting operator induced by u𝑝.
D𝑘records the effective sampling density in the UV space.
Hierarchical UV Mapping. A critical challenge in splatting is the
resolution mismatch between screen space and texture space, which
often results in sparse coverage and "holes" in the UV map. To ad-
dress this issue, we introduce a hierarchical UV map representation.
Instead of splatting features only at the target UV resolution, we
splat them onto a pyramid of UV maps with progressively lower
resolutions. At each pyramid level𝑙, bilinear splatting produces a fea-
ture map U𝑘,𝑙, and an associated density map D𝑘,𝑙. To synthesize the
final high-resolution map, we employ a coarse-to-fine hole-filling
mechanism. Coarser UV maps, which possess broader receptive
fields and denser coverage, are upsampled to the target resolution.
We fuse these upsampled features with the finer-scale features us-
ing a soft, differentiable mask derived from the density map D𝑘,𝑙.
Specifically, regions with low splatting density at the fine level are
smoothly filled by the upsampled global context from coarser levels,
effectively repairing artifacts without introducing discontinuities.
Visibility-aware Fusion. After obtaining the completed UV feature
maps from one view, we fuse the multi-view UV feature maps into
a single canonical UV feature map. The fusion weight for each view
is defined by combining three complementary factors.
View Weight: Since frontal views typically contain richer infor-
mation, we assign each view a predefined global weight 𝛾𝑘.
View-dependent Confidence: We define a view-dependent geo-
metric confidence to measure the reliability of observations under
a given view 𝑘. For each pixel 𝑝, the included angle between the
surface normal n𝑝and the viewing direction v𝑘determines the
reliability of the feature. We aggregate these per-pixel confidence
scores into the UV domain using the same splatting operator:
C𝑘=
∑︁
𝑝
B(u𝑝) · max 0, n𝑝· v𝑘
 .
(5)
This naturally down-weights grazing-angle observations while pre-
serving strong signals from front-facing surfaces.
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 6 -->
6
•
Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, and Peter Wonka
Table 2. Quantitative Comparison of Novel View Synthesis on the Avatar-256 and NeRSemble Datasets. Colors denote the best and second-best
Ava-256 Novel Views
NeRSemble Novel Views
Method
PSNR ↑
SSIM ↑
LPIPS ↓
CSIM ↑
DS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
CSIM ↑
DS ↓
PanoHead
17.995
0.5932
0.2592
0.4556
0.1663
17.151
0.6098
0.2706
0.5742
0.1082
SphereHead
18.932
0.6238
0.2436
0.4714
0.1509
17.446
0.6183
0.2628
0.5781
0.1053
GAGAvatar
22.802
0.7714
0.1719
0.4682
0.1762
22.556
0.8027
0.1505
0.6867
0.1033
SOAP
21.335
0.7287
0.1948
0.5783
0.1890
19.539
0.7471
0.1975
0.6855
0.1128
LAM
21.802
0.7314
0.1997
0.4842
0.1818
20.263
0.7465
0.1853
0.6321
0.1118
Ours
23.244
0.7734
0.1592
0.5403
0.1651
23.221
0.8051
0.1435
0.6714
0.0950
UV Sampling Density: The accumulated density D𝑘,𝑙reflects how
many features contribute to a given UV coordinate at pyramid level
𝑙, serving as a measure of feature reliability. The final canonical UV
feature map U is obtained by a globally normalized weighted average
over all views𝑘and pyramid levels𝑙. We define the composite fusion
weight map W𝑘,𝑙as the element-wise product of view weight 𝛾𝑘,
the geometric confidence C𝑘, and the sampling density D𝑘,𝑙:
W𝑘,𝑙= 𝛾𝑘· C𝑘⊙D𝑘,𝑙.
(6)
Using these weights, the fused feature map is computed as:
U =
Í
𝑘
Í
𝑙W𝑘,𝑙⊙U𝑘,𝑙
Í
𝑘
Í
𝑙W𝑘,𝑙+ 𝜖.
(7)
This formulation prioritizes densely observed and front-facing re-
gions, while allowing coarser-scale features and alternative views
to smoothly fill missing or weakly observed areas.
3.4
Training Strategy and Losses
We train the framework using a self-reenactment scheme, minimiz-
ing the discrepancy between the rendered output and the driving
frame. The total objective function is a composite of reconstruction,
perceptual, multi-view constraints and local regularization (details
in Supplementary Material).
To balance 3D consistency and expression flexibility, we propose a
progressive side-view decay strategy. Strong multi-view constraints
are essential for full-head reconstruction but hinder expression
learning, while weak constraints cause frontal overfitting. We there-
fore start training with a high 𝜆𝑚𝑣to stabilize canonical geometry,
and gradually decay it to relax the constraints, enabling the avatar to
learn high-frequency facial expressions while preserving underlying
structure.
4
Experiments
4.1
Experimental Setting
Baselines and Evaluation Metric. We compare with five recent
works: PanoHead [1], SphereHead [22], GAGAvatar [6], LAM [16],
and SOAP [26]. PanoHead and SphereHead are state-of-the-art meth-
ods for 3D full-head reconstruction using 3D GANs. GAGAvatar and
LAM are the one-shot animatable avatar reconstruction methods
based on 3D Gaussian Splatting. SOAP is a diffusion-based method
that reconstructs the full head by remeshing a FLAME mesh. We
employ five quantitative metrics. For reconstruction quality and
perceptual fidelity, we report PSNR, SSIM, LPIPS [51], and Dream-
Sim [12]. To evaluate identity preservation, we report CSIM, which
computes cosine similarity of face recognition features extracted
using ArcFace [9].
4.2
Main Results
To evaluate our method’s 3D consistency and novel view synthesis
capabilities, we leverage two publicly available multi-view datasets:
NeRSemble [21] and Avatar-256 [31]. For both datasets, reconstruc-
tion is performed from a single input image, while multi-view ob-
servations are used only for evaluation; full details of the dataset
splits, view selection, and camera settings are provided in the sup-
plementary material. Furthermore, to demonstrate generalization in
unconstrained scenarios, we collect a set of in-the-wild face images
for qualitative assessment.
Qualitative results. Fig. 5 presents a qualitative comparison be-
tween our method and baselines under novel viewpoints. Unlike
competing approaches, our method synthesizes photorealistic 3D
renderings while maintaining consistency in fine-grained details
and identity, even under extreme viewpoint variations. In contrast,
PanoHead and SphereHead struggle to preserve identity across
views, suffering from severe identity-view ambiguity. Constrained
by single-view training data and the lack of 3D priors, GAGAvatar
and LAM frequently exhibit structural distortions at large viewing
angles. Regarding the mesh-based SOAP, its remeshing process is
prone to surface fragmentation, leading to visible artifacts. Conse-
quently, our approach demonstrates superior robustness, simultane-
ously ensuring identity preservation, and high-fidelity details.
Quantitative results. Tab. 2 reports quantitative results on the
NeRSemble and Avatar-256 datasets, respectively. In terms of vi-
sual fidelity, our method demonstrates superior rendering quality
as evidenced by the PSNR, SSIM, LPIPS, and DreamSim (DS) met-
rics, while maintaining strong identity consistency with high CSIM
scores. Regarding reconstruction efficiency, PanoHead and Sphere-
Head necessitate a time-consuming Pivotal Tuning Inversion pro-
cess to optimize latent codes; SOAP relies on a multi-stage remesh-
ing process for each subject; GAGAvatar, LAM, and our method
all employ feed-forward networks for 3D Gaussian prediction, the
former two baselines lack multi-view priors, resulting in suboptimal
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 7 -->
OMEGA-Avatar: One-shot Modeling of 360◦Gaussian Avatars
•
7
Fig. 6. Qualitative Ablation on the NeRSemble dataset. We compare results for models trained: (1) without multi-view diffusion, (2) without using
semantic-aware Laplacian term, (3) without differentiable bilinear splatting, (4) without hierarchical UV mapping, (5) without visibility-aware fusion.
performance on multi-view datasets compared to our approach. Ulti-
mately, our method realizes genuine one-shot, animatable full-head
reconstruction, achieving state-of-the-art performance on both the
NeRSemble and Avatar-256 benchmarks.
Results on In-the-wild Images. To further evaluate the generaliza-
tion capability, we collected a set of in-the-wild facial images. Fig. 7
presents qualitative comparisons against baseline methods in these
unconstrained scenarios. PanoHead and SphereHead often fail to
accurately reconstruct complex hairstyles. GAGAvatar and LAM suf-
fer from severe structural distortions in back and side views. While
SOAP recovers accurate global geometry, it frequently manifests
visible cracks in the back of the head due to mesh fragmentation.
These results collectively demonstrate the robust generalization and
superior full-head reconstruction capabilities of our method.
Table 3. Ablation of different components on the NeRSemble dataset.
Ablation Novel Views
Method
PSNR ↑SSIM ↑LPIPS ↓CSIM ↑
DS ↓
Single view
22.687
0.774
0.151
0.655
0.124
w/o S.a Laplacian
22.988
0.782
0.150
0.667
0.105
w/o D.B Splatting
23.187
0.789
0.141
0.643
0.098
w/o Hierarchical UV
22.997
0.785
0.149
0.655
0.103
w/o V.a fusion
23.121
0.793
0.151
0.597
0.110
Full (Ours)
23.221
0.805
0.144
0.671
0.095
4.3
Ablation Studies
We conduct ablation studies on single-view reconstruction using
the NeRSemble dataset. The quantitative and qualitative results are
presented in Tab. 3 and Fig. 6, respectively.
w/o MV Diffusion. We remove the multi-view diffusion step. Sig-
nificant artifacts appear at the rendering boundaries of the frontal
view. Furthermore, the synthesis quality for side and rear views
degrades drastically, failing to reconstruct a complete, animatable
3D avatar. This confirms the necessity of diffusion-based priors for
hallucinating unobserved geometry and texture.
w/o Semantic-aware Laplacian. We remove the semantic-aware
Laplacian constraint. As shown in Fig. 6, this leads to geometric
inconsistencies. The rendering fidelity decreases, with noticeable
surface irregularities emerging along the head and facial contours,
resulting in bumpy geometry, indicating that the Laplacian term is
crucial for maintaining surface smoothness and mesh integrity.
w/o Differentiable Bilinear Splatting. We replace our differen-
tiable bilinear splatting with a hard nearest-neighbor assignment
and fix the density weights to 1. Although the LPIPS metric shows
a slight improvement—likely due to the absence of interpolation
blur—all other quantitative metrics decline. Qualitatively, the lack
of differentiable gradients and soft distribution leads to quantization
artifacts and a loss of fine-grained details in the rendered results.
w/o Hierarchical UV Mapping. We disable the hierarchical pyra-
mid and splat features onto a fixed-resolution UV map. As observed
in Fig. 6, this results in significant blurring in areas like the nose and
ears. The mismatch between the feature map resolution and the UV
grid causes "holes" in the UV feature space, thereby degrading the
decoding capability and expressiveness of the Gaussian primitives.
w/o Visibility-aware Fusion. We remove the view-dependent
weights, relying solely on UV sampling density for aggregation. As
reported in Table 3, the CSIM score drops significantly, indicating a
loss of identity preservation. Visually, prominent artifacts appear
around the nasal bridge, where lower-quality features from oblique
side views override the high-fidelity information from the frontal
view. This demonstrates that visibility-aware weighting is essential
for correctly prioritizing high-confidence observations.
5
Conclusion
We presented Ω-Avatar, the first feed-forward framework that re-
constructs generalizable, full-head, and animatable 3D avatars from
a single image. By introducing a semantic-aware mesh deformation
module that integrates multi-view normals to optimize a haired
FLAME head, together with a multi-view feature splatting module
that aggregates full-head features into a shared canonical UV repre-
sentation, our method enables effective feed-forward decoding of
complete head geometry, offering a robust solution for efficient 3D
avatar creation.
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 8 -->
8
•
Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, and Peter Wonka
References
[1] Sizhe An, Hongyi Xu, Yichun Shi, Guoxian Song, Umit Y. Ogras, and Linjie
Luo. 2023. PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360deg. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). 20950–20959.
[2] Yunpeng Bai, Yanbo Fan, Xuan Wang, Yong Zhang, Jingxiang Sun, Chun Yuan,
and Ying Shan. 2023. High-fidelity Facial Avatar Reconstruction from Monocular
Video with Generative Priors. In IEEE/CVF Conference on Computer Vision and
Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE,
4541–4551. doi:10.1109/CVPR52729.2023.00441
[3] Adrian Bulat and Georgios Tzimiropoulos. 2017. How Far are We from Solving the
2D & 3D Face Alignment Problem? (and a Dataset of 230, 000 3D Facial Landmarks).
In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy,
October 22-29, 2017. IEEE Computer Society, 1021–1030. doi:10.1109/ICCV.2017.116
[4] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan,
Shalini De Mello, Orazio Gallo, Leonidas J. Guibas, Jonathan Tremblay, Sameh
Khamis, Tero Karras, and Gordon Wetzstein. 2022. Efficient Geometry-aware 3D
Generative Adversarial Networks. In IEEE/CVF Conference on Computer Vision
and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE,
16102–16112. doi:10.1109/CVPR52688.2022.01565
[5] Eric R. Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein.
2021. Pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware
Image Synthesis. In IEEE Conference on Computer Vision and Pattern Recognition,
CVPR 2021, virtual, June 19-25, 2021. Computer Vision Foundation / IEEE, 5799–
5809. doi:10.1109/CVPR46437.2021.00574
[6] Xuangeng Chu and Tatsuya Harada. 2024. Generalizable and Animatable Gaussian
Head Avatar. In Advances in Neural Information Processing Systems 38: Annual
Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
BC, Canada, December 10 - 15, 2024.
[7] Xuangeng Chu, Yu Li, Ailing Zeng, Tianyu Yang, Lijian Lin, Yunfei Liu, and
Tatsuya Harada. 2024. GPAvatar: Generalizable and Precise Head Avatar from
Image(s). In The Twelfth International Conference on Learning Representations, ICLR
2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.
[8] Radek Danecek, Michael J. Black, and Timo Bolkart. 2022. EMOCA: Emotion
Driven Monocular Face Capture and Animation. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24,
2022. IEEE, 20279–20290. doi:10.1109/CVPR52688.2022.01967
[9] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos
Zafeiriou. 2022. ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
IEEE Trans. Pattern Anal. Mach. Intell. 44, 10 (2022), 5962–5979. doi:10.1109/TPAMI.
2021.3087709
[10] Yu Deng, Duomin Wang, Xiaohang Ren, Xingyu Chen, and Baoyuan Wang. 2024.
Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data.
In IEEE/CVF Conference on Computer Vision and Pattern Recognition.
[11] Yu Deng, Duomin Wang, and Baoyuan Wang. 2024. Portrait4D-V2: Pseudo Multi-
view Data Creates Better 4D Head Synthesizer. In Computer Vision - ECCV 2024 -
18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings,
Part XVII (Lecture Notes in Computer Science, Vol. 15075). Springer, 316–333. doi:10.
1007/978-3-031-72643-9_19
[12] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang,
Tali Dekel, and Phillip Isola. 2023. DreamSim: Learning New Dimensions of
Human Visual Similarity using Synthetic Data. In Advances in Neural Information
Processing Systems 36: Annual Conference on Neural Information Processing Systems
2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.
[13] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. 2022. StyleNeRF: A
Style-based 3D Aware Generator for High-resolution Image Synthesis. In The
Tenth International Conference on Learning Representations, ICLR 2022, Virtual
Event, April 25-29, 2022. OpenReview.net.
[14] Jianzhu Guo, Xiangyu Zhu, and Zhen Lei. 2018. 3DDFA. https://github.com/
cleardusk/3DDFA.
[15] Yue Han, Junwei Zhu, Keke He, Xu Chen, Yanhao Ge, Wei Li, Xiangtai Li, Jiangn-
ing Zhang, Chengjie Wang, and Yong Liu. 2024. Face-Adapter for Pre-trained
Diffusion Models with Fine-Grained ID and Attribute Control. In Computer Vision
- ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024,
Proceedings, Part L (Lecture Notes in Computer Science, Vol. 15108). Springer, 20–36.
doi:10.1007/978-3-031-72973-7_2
[16] Yisheng He, Xiaodong Gu, Xiaodan Ye, Chao Xu, Zhengyi Zhao, Yuan Dong,
Weihao Yuan, Zilong Dong, and Liefeng Bo. 2025. LAM: Large Avatar Model for
One-shot Animatable Gaussian Head. CoRR abs/2502.17796 (2025). doi:10.48550/
ARXIV.2502.17796
[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans.
Graph. 42, 4 (2023), 139:1–139:14. doi:10.1145/3592433
[18] Taras Khakhulin, Vanessa Sklyarova, Victor Lempitsky, and Egor Zakharov. 2022.
Realistic One-Shot Mesh-Based Head Avatars. In Computer Vision - ECCV 2022 -
17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part II
(Lecture Notes in Computer Science, Vol. 13662). Springer, 345–362. doi:10.1007/978-
3-031-20086-1_20
[19] Taekyung Ki, Dongchan Min, and Gyeongsu Chae. 2024. Learning to Generate
Conditional Tri-plane for 3D-aware Expression Controllable Portrait Animation.
arXiv preprint arXiv:2404.00636 (2024).
[20] Tobias Kirschstein, Simon Giebenhain, Jiapeng Tang, Markos Georgopoulos, and
Matthias Nießner. 2024. GGHead: Fast and Generalizable 3D Gaussian Heads. In
SIGGRAPH Asia 2024 Conference Papers, SA 2024, Tokyo, Japan, December 3-6, 2024,
Takeo Igarashi, Ariel Shamir, and Hao (Richard) Zhang (Eds.). ACM, 126:1–126:11.
doi:10.1145/3680528.3687686
[21] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim Walter, and Matthias
Nießner. 2023. NeRSemble: Multi-view Radiance Field Reconstruction of Human
Heads. ACM Trans. Graph. 42, 4 (2023), 161:1–161:14. doi:10.1145/3592455
[22] Heyuan Li, Ce Chen, Tianhao Shi, Yuda Qiu, Sizhe An, Guanying Chen, and
Xiaoguang Han. 2024. SphereHead: Stable 3D Full-Head Synthesis with Spherical
Tri-Plane Representation. In Computer Vision - ECCV 2024 - 18th European Confer-
ence, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXV, Vol. 15133.
324–341. doi:10.1007/978-3-031-73226-3_19
[23] Tianye Li, Timo Bolkart, Michael. J. Black, Hao Li, and Javier Romero. 2017.
Learning a model of facial shape and expression from 4D scans. ACM Transactions
on Graphics, (Proc. SIGGRAPH Asia) 36, 6 (2017), 194:1–194:17.
[24] Weichuang Li, Longhao Zhang, Dong Wang, Bin Zhao, Zhigang Wang, Mulin
Chen, Bang Zhang, Zhongjian Wang, Liefeng Bo, and Xuelong Li. 2023. One-
Shot High-Fidelity Talking-Head Synthesis with Deformable Neural Radiance
Field. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR
2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 17969–17978. doi:10.1109/
CVPR52729.2023.01723
[25] Xueting Li, Shalini De Mello, Sifei Liu, Koki Nagano, Umar Iqbal, and Jan Kautz.
2023. Generalizable One-shot 3D Neural Head Avatar. In Advances in Neural Infor-
mation Processing Systems 36: Annual Conference on Neural Information Processing
Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, Alice
Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey
Levine (Eds.).
[26] Tingting Liao, Yujian Zheng, Adilbek Karmanov, Liwen Hu, Leyang Jin, Yuliang
Xiu, and Hao Li. 2025. SOAP: Style-Omniscient Animatable Portraits. CoRR
abs/2505.05022 (2025). doi:10.48550/ARXIV.2505.05022
[27] Xin Lu, Chuanqing Zhuang, Chenxi Jin, Zhengda Lu, Yiqun Wang, Wu Liu, and
Jun Xiao. 2025. LSF-Animation: Label-Free Speech-Driven Facial Animation via
Implicit Feature Representation. CoRR abs/2510.21864 (2025). doi:10.48550/ARXIV.
2510.21864
[28] Weijie Lyu, Yi Zhou, Ming-Hsuan Yang, and Zhixin Shu. 2025. FaceLift: Learning
Generalizable Single Image 3D Face Reconstruction from Synthetic Heads. In
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
12691–12701.
[29] Haoyu Ma, Tong Zhang, Shanlin Sun, Xiangyi Yan, Kun Han, and Xiaohui Xie. 2024.
CVTHead: One-shot Controllable Head Avatar with Vertex-feature Transformer.
In IEEE/CVF Winter Conference on Applications of Computer Vision, WACV 2024,
Waikoloa, HI, USA, January 3-8, 2024. IEEE, 6119–6129. doi:10.1109/WACV57701.
2024.00602
[30] Zhiyuan Ma, Xiangyu Zhu, Guojun Qi, Zhen Lei, and Lei Zhang. 2023. OTAvatar:
One-Shot Talking Face Avatar with Controllable Tri-Plane Rendering. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC,
Canada, June 17-24, 2023. IEEE, 16901–16910. doi:10.1109/CVPR52729.2023.01621
[31] Julieta Martinez, Emily Kim, Javier Romero, Timur M. Bagautdinov, Shunsuke
Saito, Shoou-I Yu, Stuart Anderson, Michael Zollhöfer, Te-Li Wang, Shaojie Bai,
Chenghui Li, Shih-En Wei, Rohan Joshi, Wyatt Borsos, Tomas Simon, Jason M.
Saragih, Paul Theodosis, Alexander Greene, Anjani Josyula, Silvio Maeta, Andrew
Jewett, Simion Venshtain, Christopher Heilman, Yueh-Tung Chen, Sidi Fu, Mo-
hamed Elshaer, Tingfang Du, Longhua Wu, Shen-Chi Chen, Kai Kang, Michael
Wu, Youssef Emad, Steven Longay, Ashley Brewer, Hitesh Shah, James Booth,
Taylor Koska, Kayla Haidle, Matthew Andromalos, Joanna Hsu, Thomas Dauer,
Peter Selednik, Timothy Godisart, Scott Ardisson, Matthew Cipperly, Ben Hum-
berston, Lon Farr, Bob Hansen, Peihong Guo, Dave Braun, Steven Krenn, He Wen,
Lucas Evans, Natalia Fadeeva, Matthew Stewart, Gabriel Schwartz, Divam Gupta,
Gyeongsik Moon, Kaiwen Guo, Yuan Dong, Yichen Xu, Takaaki Shiratori, Fabian
Prada, Bernardo Pires, Bo Peng, Julia Buffalini, Autumn Trimble, Kevyn McPhail,
Melissa Schoeller, and Yaser Sheikh. 2024. Codec Avatar Studio: Paired Human
Captures for Complete, Driveable, and Generalizable Avatars. In Advances in Neu-
ral Information Processing Systems 38: Annual Conference on Neural Information
Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15,
2024.
[32] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi
Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance
Fields for View Synthesis. In Computer Vision - ECCV 2020 - 16th European Confer-
ence, Glasgow, UK, August 23-28, 2020, Proceedings, Part I (Lecture Notes in Computer
Science, Vol. 12346). Springer, 405–421. doi:10.1007/978-3-030-58452-8_24
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 9 -->
OMEGA-Avatar: One-shot Modeling of 360◦Gaussian Avatars
•
9
[33] Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, and Ira
Kemelmacher-Shlizerman. 2022. StyleSDF: High-Resolution 3D-Consistent Image
and Geometry Generation. In IEEE/CVF Conference on Computer Vision and Pattern
Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE, 13493–13503.
doi:10.1109/CVPR52688.2022.01314
[34] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B. Gold-
man, Steven M. Seitz, and Ricardo Martin-Brualla. 2021. Nerfies: Deformable
Neural Radiance Fields. In 2021 IEEE/CVF International Conference on Computer
Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021. IEEE, 5845–5854.
doi:10.1109/ICCV48922.2021.00581
[35] Omkar Parkhi, Andrea Vedaldi, and Andrew Zisserman. 2015. Deep face recog-
nition. In BMVC 2015-Proceedings of the British Machine Vision Conference 2015.
British Machine Vision Association.
[36] Shenhan Qian. 2024. VHAP: Versatile Head Alignment with Adaptive Appearance
Priors. doi:10.5281/zenodo.14988309
[37] Daniel Roich, Ron Mokady, Amit H. Bermano, and Daniel Cohen-Or. 2023. Pivotal
Tuning for Latent-based Editing of Real Images. ACM Trans. Graph. 42, 1 (2023),
6:1–6:13. doi:10.1145/3544777
[38] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime
Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seung Eun Yi, Michaël Ra-
mamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang,
Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea
Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou,
Patrick Labatut, and Piotr Bojanowski. 2025. DINOv3. CoRR abs/2508.10104 (2025).
doi:10.48550/ARXIV.2508.10104
[39] Phong Tran, Egor Zakharov, Long-Nhat Ho, Anh Tuan Tran, Liwen Hu, and Hao
Li. 2024. VOODOO 3D: Volumetric Portrait Disentanglement for One-Shot 3D
Head Reenactment. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition.
[40] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Bal-
trusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, and Baining
Guo. 2023. RODIN: A Generative Model for Sculpting 3D Digital Avatars Us-
ing Diffusion. In IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 4563–4573.
doi:10.1109/CVPR52729.2023.00443
[41] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi
Duan, and Kaisheng Ma. 2024. Unique3D: High-Quality and Efficient 3D Mesh
Generation from a Single Image. In Advances in Neural Information Processing
Systems 38: Annual Conference on Neural Information Processing Systems 2024,
NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.
[42] Jiaxin Xie, Hao Ouyang, Jingtan Piao, Chenyang Lei, and Qifeng Chen. 2023.
High-fidelity 3D GAN Inversion by Pseudo-multi-view Optimization. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC,
Canada, June 17-24, 2023. IEEE, 321–331. doi:10.1109/CVPR52729.2023.00039
[43] Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong, and Ying Shan. 2022.
VFHQ: A High-Quality Dataset and Benchmark for Video Face Super-Resolution.
In The IEEE Conference on Computer Vision and Pattern Recognition Workshops
(CVPRW).
[44] Sicheng Xu, Jiaolong Yang, Dong Chen, Fang Wen, Yu Deng, Yunde Jia, and Xin
Tong. 2020. Deep 3D Portrait From a Single Image. In 2020 IEEE/CVF Conference on
Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19,
2020. Computer Vision Foundation / IEEE, 7707–7717. doi:10.1109/CVPR42600.
2020.00773
[45] Songlin Yang, Wei Wang, Yushi Lan, Xiangyu Fan, Bo Peng, Lei Yang, and Jing
Dong. 2024. Learning Dense Correspondence for NeRF-Based Face Reenactment.
In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-
Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024,
Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI
2014, February 20-27, 2024, Vancouver, Canada. AAAI Press, 6522–6530. doi:10.
1609/AAAI.V38I7.28473
[46] Xin Yao, Junyi He, Chang Li, Yang Xie, Haotian Luo, Wei Zheng, Hongxing
Qin, and Yiqun Wang. 2025. JFG-HMR: 3D joint feature-guided human mesh
recovery with global-local feature fusion. Comput. Graph. 132 (2025), 104339.
doi:10.1016/J.CAG.2025.104339
[47] Zhenhui Ye, Tianyun Zhong, Yi Ren, Jiaqi Yang, Weichuang Li, Jiawei Huang,
Ziyue Jiang, Jinzheng He, Rongjie Huang, Jinglin Liu, Chen Zhang, Xiang Yin,
Zejun Ma, and Zhou Zhao. 2024. Real3D-Portrait: One-shot Realistic 3D Talking
Portrait Synthesis. In The Twelfth International Conference on Learning Represen-
tations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.
[48] Wangbo Yu, Yanbo Fan, Yong Zhang, Xuan Wang, Fei Yin, Yunpeng Bai, Yan-
Pei Cao, Ying Shan, Yang Wu, Zhongqian Sun, and Baoyuan Wu. 2023. NOFA:
NeRF-based One-shot Facial Avatar Reconstruction. In ACM SIGGRAPH 2023
Conference Proceedings, SIGGRAPH 2023, Los Angeles, CA, USA, August 6-10, 2023,
Erik Brunvand, Alla Sheffer, and Michael Wimmer (Eds.). ACM, 85:1–85:12. doi:10.
1145/3588432.3591555
[49] Bowen Zhang, Yiji Cheng, Chunyu Wang, Ting Zhang, Jiaolong Yang, Yansong
Tang, Feng Zhao, Dong Chen, and Baining Guo. 2024. RodinHD: High-Fidelity
3D Avatar Generation with Diffusion Models. In Computer Vision - ECCV 2024 -
18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings,
Part XIV (Lecture Notes in Computer Science, Vol. 15072). Springer, 465–483. doi:10.
1007/978-3-031-72630-9_27
[50] Jiawei Zhang, Zijian Wu, Zhiyang Liang, Yicheng Gong, Dongfang Hu, Yao Yao,
Xun Cao, and Hao Zhu. 2025. FATE: Full-head Gaussian Avatar with Textural
Editing from Monocular Video. In IEEE/CVF Conference on Computer Vision and
Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025. Computer
Vision Foundation / IEEE, 5535–5545. doi:10.1109/CVPR52734.2025.00520
[51] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.
2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018,
Salt Lake City, UT, USA, June 18-22, 2018. Computer Vision Foundation / IEEE
Computer Society, 586–595. doi:10.1109/CVPR.2018.00068
[52] Shikun Zhang, Cunjian Chen, Yiqun Wang, Qiuhong Ke, and Yong Li. 2025. EA-
vatar: Expression-Aware Head Avatar Reconstruction with Generative Geometry
Priors. CoRR abs/2508.13537 (2025). doi:10.48550/ARXIV.2508.13537
[53] Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J. Black, and Otmar
Hilliges. 2023. PointAvatar: Deformable Point-Based Head Avatars from Videos. In
IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Van-
couver, BC, Canada, June 17-24, 2023. IEEE, 21057–21067. doi:10.1109/CVPR52729.
2023.02017
[54] Wojciech Zielonka, Timo Bolkart, and Justus Thies. 2023. Instant Volumetric
Head Avatars. In IEEE/CVF Conference on Computer Vision and Pattern Recognition,
CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 4574–4584. doi:10.1109/
CVPR52729.2023.00444
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 10 -->
10
•
Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, and Peter Wonka
Fig. 7. Additional Results on In-the-wild Images. Our method demonstrates great generalization ability and robustness towards in-the-wild images, and
provides realistic unseen view rendering results. We use the red boxes to highlight the visual artifacts.
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 11 -->
OMEGA-Avatar: One-shot Modeling of 360◦Gaussian Avatars
•
11
Fig. 8. More Visual results on NeRSemble dataset. Taking a single view as input, we perform 360◦novel view synthesis to compare our method with
state-of-the-art approaches. The results show that our method achieves superior multi-view consistency and accurately reconstructs unseen regions. We use
the red boxes to highlight the visual artifacts.
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 12 -->
12
•
Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, and Peter Wonka
Supplementary Material
A
...
In this supplementary material, we present additional results and
implementation details of our method. In Section A, we elaborate
on the training framework. Section B details the expression reenact-
ment pipeline. We then describe the specific experimental settings
in Section C. Section D presents additional results on novel view
synthesis and 360◦full-head reconstruction.
B
Training Details
To achieve high-fidelity reenactment with robust geometry, our
optimization is guided by a composite objective function compris-
ing reconstruction, perceptual, and multi-view consistency terms.
First, to ensure pixel-level photometric consistency, we employ an
𝐿1 reconstruction loss L𝑟𝑒𝑐. This is applied to both the raw splatted
image 𝐼𝑟𝑎𝑤and the final neural-refined image 𝐼𝑟𝑒𝑓. To further en-
hance high-frequency details in critical facial areas, we incorporate
a bounding-box loss L𝑏𝑜𝑥on the cropped facial region:
L𝑟𝑒𝑐= |𝐼𝑟𝑎𝑤−𝐼𝑑| + |𝐼𝑟𝑒𝑓−𝐼𝑑| + 𝜆𝑏𝑜𝑥L𝑏𝑜𝑥,
(8)
To preserve high-level facial identity and structural semantics, we
incorporate a perceptual loss L𝑝𝑒𝑟𝑐derived from a pre-trained face
recognition network [35]. Then, to alleviate depth ambiguity and en-
force 3D geometric consistency, we introduce a pseudo-multi-view
supervision L𝑚𝑣. We leverage the diffusion-generated multi-view
data as pseudo-ground truth. Crucially, this supervision is restricted
to the raw render result 𝐼𝑟𝑎𝑤, ensuring geometric constraints while
avoiding overfitting of the neural refiner to potential artifacts. Fi-
nally, we impose regularization terms L𝑙𝑜𝑐𝑎𝑙on the local position
offsets and scaling attributes of the UV Gaussians to prevent the
UV Gaussians from drifting away from the underlying surface. The
total objective function is formulated as:
L = L𝑟𝑒𝑐+ 𝜆𝑝𝑒𝑟𝑐L𝑝𝑒𝑟𝑐+ 𝜆𝑚𝑣L𝑚𝑣+ 𝜆𝑙𝑜𝑐𝑎𝑙L𝑙𝑜𝑐𝑎𝑙,
(9)
where the 𝜆terms are hyperparameters that balance the contribu-
tions.
Training Datasets. We utilize the VFHQ dataset [43] for training,
which comprises high-quality video clips from various interview
scenarios. To ensure data diversity and avoid consecutive redun-
dancy, we sample 25 to 75 frames per clip depending on its length,
resulting in a total of 571951 frames from 15,204 video clips. All
images are resized to 512 × 512. Following [6], we remove the back-
ground and track camera poses and FLAME [23] parameters for each
frame. Crucially, to bridge the gap between monocular observations
and 3D supervision, we employ our pre-trained diffusion model to
expand these monocular frames into pseudo-multi-view data, which
provide the necessary geometric and appearance constraints for
training.
Implementation Details. Our framework is implemented on the
PyTorch platform. We employ FLAME [23] as our driving 3DMM.
For feature extraction, we utilize DINOv3 [38] as the backbone net-
work, which remains frozen during the entire training process. To
enable multi-view supervision, we preprocess the VFHQ dataset by
generating multi-view images for each frame using a pre-trained dif-
fusion model. These generated pseudo-multi-views are then utilized
as ground truth for supervision during training. We use the Adam
[Kingma and Ba, 2014] optimizer with a learning rate of 1.0 × 10−4.
The model is trained for 250,000 iterations with a batch size of 6 on
two NVIDIA H100 GPUs.
C
Expression Reenactment
The goal of Expression Reenactment is to transfer the pose and ex-
pression from a driving image to the source image while preserving
the source identity and maintaining high fidelity. After decoding, the
Gaussian primitives from the vertex and UV branches are combined
to form a complete canonical 3D Gaussian avatar. To animate this
avatar, we first employ a face tracker [8] to extract the expression
and pose parameters from the driving image. These parameters are
then injected into the canonical avatar via the FLAME, explicitly
driving the Gaussians to the target state. Finally, the deformed Gaus-
sians are rendered using the camera parameters of the driving view
via splatting.
D
Experimental Details
We evaluate our method leverage two publicly available multi-view
datasets: NeRSemble [21] and Avatar-256 [31].
NeRSemble dataset The NeRSemble dataset consists of 425 identi-
ties, each captured by 16 fixed cameras. For quantitative evaluation,
we randomly sampled 10 identities and selected 9 sequences for
each identity. From each sequence, we further randomly sampled
50 frames. For every frame, we utilize camera 222200037 as the
single input view, as it represents the dataset-defined frontal per-
spective, while the remaining camera views serve as the test views.
The specific identities used for evaluation are listed in Tab. 4.
Ava-256 dataset The Ava-256 dataset consists of 256 identities,
each captured by 80 cameras, with over 5,000 frames per camera.
For qualitative evaluation, we randomly sampled 10 identities and
selected 20 random frames from the 5,000 available frames for each
identity. For each selected frame, we use camera 401168 as the single
input view. This camera is selected for its frontal perspective and
central position in the world coordinate system, while the remain-
ing views are reserved for testing. The specific identities used for
evaluation are listed in Tab. 4.
For PanoHead [1] and SphereHead [22], we apply 3DDFA-V2 [14]
to align the input image to the FFHQ canonical space. The aligned
image is then inverted using Pivotal Tuning Inversion [37] for single-
view head reconstruction.
For the remaining methods, we strictly follow their official im-
plementations to obtain the target-view camera parameters, and
render the output by rotating the camera to the target view.
E
MORE RECONSTRUCTION RESULTS
We provide additional visual results of our method. Fig. 9 presents
additional qualitative results of our method on the NeRSemble and
Avatar-256 datasets for novel view synthesis, while Fig. 10 presents
more results of our model on in-the-wild data, demonstrating 360◦
full-head reconstruction.
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 13 -->
OMEGA-Avatar: One-shot Modeling of 360◦Gaussian Avatars
•
13
Table 4. Identities used from the NeRSemble and Ava-256 datasets.
Index
NeRSemble ID
Ava-256 ID
1
017
20220614–1135–DNM410
2
070
20220808–0809–DPE040
3
124
20230308–1352–BDF920
4
175
20230328–0800–BLY735
5
214
20230405–1635–AAN112
6
218
20230726–1657–AYE877
7
304
20230728–0757–CRV122
8
306
20230810–1355–AJR151
9
371
20230901–1429–CPP930
10
490
20230908–1645–DHA971
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 14 -->
14
•
Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, and Peter Wonka
Fig. 9. Additional qualitative results on the NeRSemble and Avatar-256 datasets. We use the red boxes to highlight the visual artifacts.
, Vol. 1, No. 1, Article . Publication date: February 2026.

<!-- page 15 -->
OMEGA-Avatar: One-shot Modeling of 360◦Gaussian Avatars
•
15
Fig. 10. Full-head reconstruction results on in-the-wild images
, Vol. 1, No. 1, Article . Publication date: February 2026.
