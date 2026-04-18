<!-- page 1 -->
VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single
Image
TENG-FANG HSIAO, BO-KAI RUAN, YU-LUN LIU, HONG-HAN SHUAI
Source View
Edited View
Edited View
(b) Edited Mesh
(a) Input Mesh
(c) Edited Mesh
Editing
Render
Edit
Image
Edit
Image
VecSet
Edit
VecSet
Edit
Fig. 1. VecSet-Edit: Localized geometry and texture editing from a single image. Our method allows for localized 3D mesh editing guided by a
single-view 2D image. (Left) Given an input mesh (a), users can edit a rendered view to guide the 3D editing. As shown in (b) and (c), our method accurately
transfers these 2D semantic changes to the 3D mesh. This produces explicit geometric deformations (e.g., cat head, watermelon wheels) while preserving the
geometry of unedited regions (e.g., the car body remains unchanged). (Right) This localized capability enables precise multi-object editing in complex scenes.
3D editing has emerged as a critical research area to provide users with flexi-
ble control over 3D assets. While current editing approaches predominantly
focus on 3D Gaussian Splatting or multi-view images, the direct editing of
3D meshes remains underexplored. Prior attempts, such as VoxHammer, rely
on voxel-based representations that suffer from limited resolution and ne-
cessitate labor-intensive 3D mask. To address these limitations, we propose
VecSet-Edit, the first pipeline that leverages the high-fidelity VecSet Large
Reconstruction Model (LRM) as a backbone for mesh editing. Our approach
is grounded on a analysis of the spatial properties in VecSet tokens, revealing
that token subsets govern distinct geometric regions. Based on this insight,
we introduce Mask-guided Token Seeding and Attention-aligned Token
Gating strategies to precisely localize target regions using only 2D image
conditions. Also, considering the difference between VecSet diffusion process
versus voxel we design a Drift-aware Token Pruning to reject geometric
outliers during the denoising process. Finally, our Detail-preserving Tex-
ture Baking module ensures that we not only preserve the geometric details
of original mesh but also the textural information. More details can be found
in our project page: https://github.com/BlueDyee/VecSet-Edit/tree/main
CCS Concepts: • Computing methodologies →Mesh models; Shape
modeling; Image manipulation; Artificial intelligence.
Additional Key Words and Phrases: mesh editing, single image guidance,
large reconstruction model, token selection
1
Introduction
Recent progress in automated 3D asset generation [Chen et al. 2025;
Li et al. 2024, 2025d; Xiang et al. 2025; Xu et al. 2024; Zhao et al. 2025]
Author’s Contact Information: Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han
Shuai.
is rapidly increasing the supply of 3D content. Yet in real produc-
tion workflows, generated assets rarely satisfy downstream require-
ments. Creators often need localized, controllable refinements, such
as editing a specific part, adjusting geometry in a region, or modify-
ing appearance while preserving the asset identity. Re-generation
is typically not a reliable substitute, because it can inadvertently
change geometry, topology, or texture, making 3D editing essential
for turning generated outputs into usable assets.
Despite this need, most existing editing methods operate on inter-
mediate representations such as 3D Gaussian Splatting (3DGS) [Chen
et al. 2024a; Pandey et al. 2025; Wang et al. 2024a; Yan et al. 2024;
Zheng et al. 2025] or multi-view images [Bar-On et al. 2025; Edel-
stein et al. 2025; Huang et al. 2025b; Li et al. 2025b; Zhuang et al.
2024], rather than directly editing meshes with explicit topology
required by animation and physics-based simulation. While Vox-
Hammer [Li et al. 2025a] takes an important step by leveraging a
voxel-based Large Reconstruction Model (LRM) [Xiang et al. 2025]
for editing, it still faces two practical limitations: 1) it requires addi-
tional 3D mask annotation that is labor intensive and hard to scale,
and 2) its voxel granularity fundamentally constrains resolution and
fidelity compared with modern VecSet-based LRMs. These gaps call
for a mesh native editing framework that demands only lightweight
supervision, preserves asset identity, and fully exploits high fidelity
VecSet reconstruction backbones.
To address these limitations, we propose VecSet-Edit, a training-
free framework for localized 3D mesh editing built on a high-fidelity
VecSet reconstruction model [Hunyuan3D et al. 2025; Li et al. 2024,
arXiv:2602.04349v1  [cs.CV]  4 Feb 2026

<!-- page 2 -->
2
•
Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai
2025c,d; Zhang et al. 2024b; Zhao et al. 2025]. Given a reference mesh
and its rendered view, a target edited image, and a binary 2D mask,
VecSet-Edit produces an edited mesh that follows the requested
change while preserving identity and fine details outside the edit
region (as shown in Fig. 1). The core challenge is localization: VecSet
encodes geometry as an unordered token set, so naive token-space
editing often affects unintended regions. Our key observation is that
VecSet tokens are not spatially arbitrary; despite the unordered form,
they exhibit stable locality and consistently correspond to coherent
surface regions. Leveraging this property, we localize edits by oper-
ating on a region-specific token subset. Since 2D-to-token selection
can be noisy, we introduce Mask-guided Token Seeding to obtain
a coarse token set and Attention-aligned Token Gating to re-
tain only tokens most correlated with the target region, tightening
localization and preventing spillover.
With localization in place, we face a second challenge unique
to VecSet diffusion. During denoising, tokens are spatially mobile
rather than fixed on a grid, and early drift can cause irreversible
interference between edited and preserved regions. VecSet-Edit ad-
dresses this with Drift-aware Token Pruning, which removes
drifted tokens that violate geometric consistency with the refer-
ence, keeping boundaries clean. Finally, Detail-preserving Tex-
ture Baking transfers texture to the edited mesh while retaining
high-frequency details from the original asset. The key contributions
can be summarized as follows.
• We present VecSet-Edit, the first training-free framework that
enables localized 3D mesh editing directly in the latent VecSet
space of a high-fidelity reconstruction model, avoiding the reso-
lution bottleneck of voxel-based LRMs.
• To localize edits on an unordered set of VecSet tokens, Mask-
guided Token Seeding first derives a coarse editable token set
from a 2D mask, and Attention-aligned Token Gating then
retains tokens with the strongest spatial correlation to the target
region. This ensures editing does not leak to unintended areas.
• We further introduce Drift-aware Token Pruning to remove
drifted tokens that would otherwise break geometric consistency
during the denoising process, and propose Detail-preserving
Texture Baking to update textures only where geometry changes,
preserving appearance details in untouched regions.
2
Related Works
VecSet-based Large Reconstruction Models. Recent progress in Large
Reconstruction Models (LRMs) has enabled high-quality 3D asset
synthesis from text prompts or images. Existing approaches can
be grouped by their intermediate 3D representations, including
voxel-based methods [Wang et al. 2024b; Xiang et al. 2025; Yang
et al. 2025], triplane-based methods [Gao et al. 2025; Jun and Nichol
2023; Oztas et al. 2025], and VecSet-based methods [Hunyuan3D
et al. 2025; Li et al. 2024, 2025c,d; Zhao et al. 2025]. While voxel
and triplane representations provide convenient spatial parameter-
izations, they typically rely on grid-aligned feature fields whose
decoded surfaces do not necessarily expose mesh-level granularity
for region-targeted edits. In contrast, VecSet-based LRMs represent
surface geometry with a latent vector set, and have recently shown
strong reconstruction quality with practical inference efficiency.
However, their lack of explicit spatial indexing makes it unclear
how to associate user-specified edit regions with a subset of latent
tokens, which is crucial for localized mesh editing with identity
preservation. This gap motivates editing frameworks that uncover
reliable token to surface correspondence and support region-aware
manipulation on top of VecSet generative backbones.
3D Editing across Representations. Neural 3D editing has moved
beyond classical mesh deformation toward methods that modify
learned scene representations, including NeRF-based models [Dong
and Wang 2023; Haque et al. 2023; Zhuang et al. 2023] and 3D
Gaussian Splatting [Chen et al. 2024a; Qu et al. 2025; Wang et al.
2024a; Wu et al. 2024; Zhang et al. 2025]. For instance, Instruct-
NeRF2NeRF [Haque et al. 2023] applies diffusion-based guidance
on rendered views and iteratively updates the underlying NeRF to
realize the requested edits. Despite strong visual quality, these ap-
proaches primarily operate in radiance-field or splat space, making
it nontrivial to obtain a simulation-ready mesh. VoxHammer [Li
et al. 2025a] moves closer to mesh editing via a voxel-based LRM,
but faces a fidelity–cost tension at high resolution and dependence
on explicit 3D region specification. Conversely, by exploiting the
compact yet expressive VecSet structure, our method delivers high-
fidelity mesh editing results using only 2D mask condition, bridging
the gap between automated generation and precise control.
3
Preliminary
We build upon TripoSG [Li et al. 2025d], a VecSet-based LRM repre-
senting geometry as a compact set of latent vectors. It consists of a
Variational Autoencoder (VAE) for mesh-token mapping and a Dif-
fusion Transformer (DiT) for conditional generation. We leverage
the VAE to bridge mesh and latent spaces, enabling localized editing
by manipulating a selected subset of tokens during DiT denoising.
See Appendix A for backbone details.
VecSet VAE (geometry codec). The VAE maps between a surface
mesh and its latent representation. The encoder takes a surface
point cloud P′ ∈R𝑁𝑝′ ×6 and learnable queries P ∈R𝑁𝑝×6 (con-
taining position and normal). Queries extract geometry from P′
via cross-attention to yield latent tokens V = Encode(S) ∈R𝑁𝑝×𝐶.
Conversely, the decoder predicts a signed distance field (SDF) from
V to extract the reconstructed mesh S′ = Decode(V) via Marching
Cubes. We use these interfaces to seamlessly transition between
mesh and VecSet spaces.
VecSet DiT (conditional token denoising). The DiT performs con-
ditional generation over latent tokens via rectified flow [Liu et al.
2022]. Given an image 𝐼∈R𝐻×𝑊×3, we extract dense features
ℎ𝐼∈R𝐻×𝑊×𝐷using a frozen encoder (e.g., DINOv2 [Oquab et al.
2024]). Inference starts from noise V𝑇∼N (0, I) and iteratively
evolves toward V0 using an Euler solver:
V𝑡−Δ𝑡= V𝑡−𝑢𝜃(V𝑡,ℎ𝐼,𝑡) · Δ𝑡,
(1)
where 𝑢𝜃is the predicted velocity field, 𝑡is diffusion time, and
Δ𝑡= 1/𝑁𝑠is the step size.
The DiT is composed of 𝐿DiT stacked blocks that alternate be-
tween self-attention over VecSet tokens and cross-attention to image
features. For the 𝑙-th block, the update is given by:
bV(𝑙)
𝑡
= SelfAttn(V(𝑙)
𝑡,𝑡),
(2)

<!-- page 3 -->
VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image
•
3
(b) VecSet Encoding
(c) Token Selection
(d) Edit with Token Pruning
...
...
Attention-aligned
Token Gating
Add Noise
Mask-guided
Token Seeding
Attention-aligned
Token Gating
      Repaint until t = 0
Decode
(a) View Editing
Encode
Decode
DiT Block 
DiT Block 
DiT Block 
Summation of Attentive Layers
Repaint until t < 
Decode
Threshold 
Threshold 
Fig. 2. Overview of the VecSet-Edit framework. Given a mesh S and a user-edited target view 𝐼𝐸(a), the pipeline proceeds in three stages: (b) VecSet
Encoding: S is encoded into a set of latent tokens V, which serves as the workspace for editing. (c) Token Selection: To localize the editable region without
3D supervision, we analyze the internal attention maps of the LRM. Mask-guided Token Seeding aggregates informative cross-attention layers to identify initial
seed tokens V𝐼that align the 2D mask. Attention-aligned Token Gating then leverages self-attention correlations to expand this selection to the full geometric
structure, yielding the final editable subset V𝐸. (d) Edit with Token Pruning: We perform diffusion-based editing on V𝐸while constraining the preserved
tokens V𝑃. To prevent geometric artifacts, Drift-aware Token Pruning gets involved during denoising to detect and discard “conflict” tokens that drift into the
preserved regions but lack support from the editing condition. This ensures the final output faithfully respects both the target edit and the original structure.
V(𝑙+1)
𝑡
= CrossAttn(bV(𝑙)
𝑡,ℎ𝐼),
(3)
where feed-forward layers and residual connections are omitted for
clarity. These attention operations form the foundation of our token-
level analysis and manipulation, which we exploit for localized mesh
editing in the following sections.
4
Method
We consider the task of 3D mesh editing guided by 2D image condi-
tions. The overall pipeline is illustrated in Fig. 2. The input consists
of a reference mesh S and its rendered view 𝐼S ∈R𝐻×𝑊×3, a tar-
get edited image 𝐼𝐸∈R𝐻×𝑊×3 and a binary mask 𝑀𝐼∈R𝐻×𝑊
indicating the region to be edited in the image space. Our goal is
to (1) produce an edited mesh Sout whose geometry reflects the
semantic changes specified by 𝐼𝐸within the masked region, and
(2) faithfully preserve the original geometry and texture of S else-
where. To achieve this goal, we first leverage the VecSet Geometry
Property (Section 4.1), which reveals that localized mesh regions can
be controlled through appropriate subsets of VecSet tokens. This
observation allows us to reformulate editing as a token selection
problem. Building on this formulation, we introduce Token Selec-
tion (Section 4.2) to map the 2D mask to geometry-relevant tokens.
Based on the selected tokens, we propose VecSet-Edit (Section 4.3),
a unified pipeline that performs diffusion-based VecSet editing.
4.1
VecSet Geometry Property
Editing a mesh through its VecSet representation differs fundamen-
tally from de novo generation: we must preserve existing geometry
in protected regions while modifying others. However, VecSet to-
kens lack an explicit spatial partition, and attention creates non-local
dependencies, so a token can affect multiple regions rather than a
single isolated part. Therefore, controllable local editing hinges on
a concrete token–region association: Can we identify the tokens that
most strongly influence a user-specified region, enabling targeted edits
while keeping the rest unchanged?
To answer this, we first formalize region-faithful token subsets
using a 3D bounding volume and a reconstruction tolerance.
Definition 4.1 (VecSet Geometry Property). Given a 3D watertight
bounding volume B and a mesh S, we define the target region as
their intersection, SB ≔S∩B. We say that a subset of VecSet tokens
VB ⊂V satisfies the geometry property for SB under tolerance𝜖if
the Chamfer Distance (CD) between SB and the geometry decoded
from VB is below 𝜖:
CD(SB, Decode(VB )) < 𝜖.
(4)
Empirical verification. We conduct a sanity check with TripoSG [Li
et al. 2025d] on Edit3D-Bench [Li et al. 2025a] (300 pairs of mesh
and bounding box). To identify the subset VB corresponding to the
edit region, we leverage the spatial alignment inherent in the Vec-
Set encoding (Section A.1). Recall that each token is initialized via

<!-- page 4 -->
4
•
Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai
Mesh 
Surface
Points 
Index Set 
Bounding Box 
Encode
Decode
Decode
Chanfer
 Distance
Learnable Query
 Points 
Fig. 3. Illustration of the VecSet Geometry Property. We validate that
the unordered VecSet tokens exhibit spatial locality. Given a mesh S
and a bounding box B, we first identify the index set I corresponding
to query points P that fall within B. We then extract the token subset
VB = Gather(V, I). Finally, we quantify the reconstruction fidelity by mea-
suring the Chamfer Distance between the geometry decoded purely from
the subset (Decode(VB )), the reference geometry cropped from the full
reconstruction (Decode(V) ∩B) with cropped source mesh S ∩B.
Table 1. Reconstruction fidelity of the VecSet subset versus SB. We
report the percentage of samples with a Chamfer Distance smaller than the
threshold 𝜖.
Subset Type
𝜖= 0.30
𝜖= 0.10
𝜖= 0.05
𝜖= 0.01
Decode(V) ∩B
100%
100%
100%
98.6%
Decode(VB )
82.3%
73.5%
69.4%
44.2%
positional encodings derived from a query point p𝑖∈R6. Based on
the premise that these tokens retain strong spatial priors from the
encoding process, we select the subset falling within the box B. For-
mally, we define the index set I = {𝑖| pxyz
𝑖
∈B}, where pxyz
𝑖
denotes
the spatial coordinates of p𝑖. The local VecSet representation is then
constructed by restricting the latent to this subset: VB = {v𝑖}𝑖∈I.
The illustration of this verification pipeline can be found in Fig. 3.
Results. As shown in Table 1, 82.3% of samples reconstructed
from the subset VB achieve a Chamfer Distance below 𝜖= 0.30,
while 44.2% meet the stricter threshold of 𝜖= 0.01. Although tight
tolerances remain challenging, these results indicate that even this
naive selection often yields a region-faithful subset, supporting the
practical relevance of the VecSet Geometry Property. We expect the
reconstruction fidelity to further improve with more robust token
selection or refinement strategies.
Connection with Editing. The property motivates decoupled mesh
editing by partitioning the latent tokens into an editable subset and
a preservation subset: V ≔V𝐸⊕V𝑃, where V𝐸is intended to cap-
ture the user-specified edit region and V𝑃preserves the remaining
structure. Under this view, controllable editing reduces to a token
selection problem: identifying V𝐸for an instruction-defined target
region while keeping V𝑃unchanged.
4.2
Token Selection
Guided by the VecSet Geometry Property, we introduce a two Token
Selection mechanism to address distinct editing requirements. To
ensure semantic alignment with the 2D condition, we employ Mask-
guided Token Seeding; to maintain 3D spatial coherence, we utilize
Attention-aligned Token Gating.
4.2.1
Mask-guided Token Seeding. Inspired by prior text-to-image
editing works [Cao et al. 2023; Couairon et al. 2022; Hertz et al. 2022;
Hsiao et al. 2025a,b; Kulikov et al. 2025; Meng et al. 2021; Zhou
et al. 2025], which demonstrate that cross-attention maps provide
effective zero-shot cues for semantic localization, we adapt this idea
to the 3D VecSet DiT to identify geometry-relevant tokens from a 2D
edit mask. Intuitively, if a VecSet token consistently attends to pixels
inside the mask, it is likely responsible for generating geometry that
explains the masked region.
Let A(𝑙,𝑡)
cross ∈R𝑁𝑝×𝐻𝑊denote the cross-attention map in the Vec-
Set DiT at block 𝑙and diffusion timestep 𝑡(refer to Eq. (3)), and let
¯𝑀𝐼∈{0, 1}𝐻𝑊be the flattened 2D mask. We first measure the align-
ment between the𝑖-th token and the masked region by accumulating
its attention mass over masked pixels:
𝑎(𝑙,𝑡)
𝑀
(𝑖) =
𝐻𝑊
∑︁
𝑗=1
A(𝑙,𝑡)
cross[𝑖, 𝑗] · ¯𝑀𝐼[𝑗].
(5)
Since attention quality is inconsistent across layers [Zhou et al.
2025], naively aggregating all blocks introduces noise. Drawing
on the intuition that high KL divergence correlates with strong
condition alignment (as shown in Fig. 4), we evaluate layer informa-
tiveness via the KL-divergence of their cross-attention distributions.
Consequently, we restrict aggregation to a subset of layers Lattn,
filtering out uninformative patterns (details in Appendix B), and
average over timesteps:
˜𝑎𝑀(𝑖) = E𝑙∈Lattn,𝑡
h
𝑎(𝑙,𝑡)
𝑀
(𝑖)
i
.
(6)
We select tokens whose alignment score exceeds a threshold 𝜏𝐼1:
V𝐼= Seeding(V, 𝐼𝑆, 𝑀𝐼) = {v𝑖∈V | ˜𝑎𝑀(𝑖) > 𝜏𝐼} .
(7)
V𝐼represents the subset of VecSet tokens whose decoded geometry
explains the masked region in the conditioned view. As such, V𝐼
serves as a semantic proxy for the 3D edit region inferred purely
from 2D image supervision.
4.2.2
Attention-aligned Token Gating. To further improve spatial
coherence, we leverage the self-attention maps in the VecSet DiT,
which capture token–token interactions that often reflect geometric
proximity and topological adjacency. Starting from a reference token
set 𝑉𝐼, we use these attention-derived affinities to expand the set
by retrieving additional tokens that are most strongly coupled to
the same local region, yielding a more spatially consistent token
subset for editing. Specifically, let A(𝑙,𝑡)
self ∈R𝑁𝑝×𝑁𝑝denote the self-
attention map at block 𝑙and timestep 𝑡(refer to Eq. (2)). Given a
reference token subset Vref ⊂V, we define its membership indicator
mVref ∈{0, 1}𝑁𝑝, where mVref [𝑗] = 1 iff v𝑗∈Vref. We then score
each token by how much it attends to the reference set:
𝑎(𝑙,𝑡)
self (𝑖) =
𝑁𝑝
∑︁
𝑗=1
A(𝑙,𝑡)
self [𝑖, 𝑗] · mVref [𝑗].
(8)
1The choice of 𝜏𝐼can be found in Appendix B.

<!-- page 5 -->
VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image
•
5
Attention in Text-to-Image Model
Attention in VecSet Model
"a cat sleeping
under the sun"
Sun
0.10
0.20
0.20
0.20
0.18
Cat
0.20
0.20
0.20
0.10
0.18
Others 0.70
0.60
0.60
0.70
0.65
Sun
0.70
0.10
0.20
0.10
0.25
Cat
0.10
0.10
0.10
0.60
0.25
Others 0.20
0.80
0.70
0.30
0.50
...
Sorting 
DiT Block 
DiT Block 
DiT Block 
Visualizing 
Visualizing 
Fig. 4. Illustration of KL divergence in T2I and VecSet Diffusion pro-
cess. In the T2I models, the layers with higher divergence are more corre-
lated the prompt with object location. A similar pattern can be found in
the VecSet Model, where the tokens with higher KL divergence are more
correlated with the image.
Unlike cross-attention, self-attention exhibits more stable query-key
relationships across layers. As such, we average uniformly over all
layers and timesteps:
˜𝑎self(𝑖) = E𝑙,𝑡
h
𝑎(𝑙,𝑡)
self (𝑖)
i
.
(9)
Similarly, we select tokens whose score exceeds a threshold 𝜏𝐴:
V𝐴= TokenGating(V, Vref) = {v𝑖∈V | ˜𝑎self(𝑖) > 𝜏𝐴} .
(10)
The resulting subset V𝐴effectively captures tokens geometrically
adjacent to the reference set. This strategy leverages the geometry
property of VecSets, where strong self-attention correlations serve
as a reliable proxy for geometric connectivity.
4.3
VecSet-Edit Framework
We now present VecSet-Edit, a framework for image-guided lo-
calized mesh editing in VecSet space. Given a reference mesh S
encoded as VecSet tokens V, a source image 𝐼𝑆, a target edited image
𝐼𝐸, and a binary edit mask 𝑀𝐼, the objective is to modify only the
geometry associated with the masked region while strictly preserv-
ing the remaining structure. VecSet-Edit consists of three tightly
coupled stages: (i) identifying editable tokens via token selection, (ii)
performing constrained diffusion-based editing, and (iii) removing
diffusion-induced artifacts through token pruning.
Editable Token Decomposition. Following the VecSet Geometry
Property (Section 4.1), we first decompose the VecSet representation
into editable and preserved subsets. To obtain V𝐸, we apply Mask-
guided Token Seeding using the source image and mask:
V𝐼= TokenSeeding(V, 𝐼𝑆, 𝑀𝐼),
(11)
which identifies tokens whose decoded geometry explains the masked
image region. We then enforce spatial coherence by expanding this
set through Attention-aligned Token Gating:
V𝐸= TokenGating(V, V𝐼),
(12)
Adding Noise
DeNoise
Fig. 5. Illustration of the VecSet RePaint process (same input condi-
tion as Fig. 2). We visualize a toy example where tokens serve as particles
and their movement regions are denoted by circles. Blue and Red dots rep-
resent the preserved tokens V𝑃and edited tokens V𝐸. As illustrated, at
𝑡= 0.5𝑇, the overlap between V𝑃and V𝐸becomes irreversible due to the
contraction of the movement region.
with the preserved tokens defined as V𝑃= V \ V𝐸.
RePaint in VecSet Space. With V𝑃fixed, we perform diffusion-
based editing by adapting the RePaint strategy [Lugmayr et al. 2022]
to VecSet tokens. Starting from a noisy initialization at timestep
𝑇repaint, the editable tokens are iteratively denoised under the guid-
ance of the target image 𝐼𝐸, while the preserved tokens are con-
strained to follow their original diffusion trajectory:
V(𝑡−Δ𝑡)
𝐸
= V(𝑡)
𝐸
−Gather

𝑢𝜃(V(𝑡)
RP ,ℎ𝐼,𝑡), I𝐸

· Δ𝑡,
(13)
V(𝑡−Δ𝑡)
𝑃
= (1 −(𝑡−Δ𝑡)) 𝜖𝑃+ (𝑡−Δ𝑡) V𝑃,
(14)
V(𝑡−Δ𝑡)
RP
= V(𝑡−Δ𝑡)
𝐸
⊕V(𝑡−Δ𝑡)
𝑃
.
(15)
Here, I𝐸denotes the set of indices corresponding to the editable
tokens V𝐸, and Gather(·, I𝐸) extracts the output predictions from
the backbone model based on these indices. This process allows
the edited region to migrate toward the target geometry while
maintaining consistency with the preserved structure.
Drift-aware Token Pruning. Due to the spatial mobility of VecSet
tokens, early-stage denoising may cause editable tokens to drift
into regions that should remain unchanged, leading to geometric
overlap that cannot be corrected in diffusion steps. A toy example
of this scenario is shown in Fig. 5. Thus, we introduce Drift-aware
Token Pruning., which intervenes at a timestep 𝑇pruning ≤𝑇repaint to
explicitly resolve conflicts between edited and preserved geometry.
At the pruning timestep, we identify two complementary token
subsets. (i) Vcond, contains tokens that are supported by the target
image condition and thus should be retained for editing. (ii) Vconflict,
contains tokens that are structurally associated with the preserved
region and therefore risk introducing geometric redundancy. There-
fore, tokens in Vconflict should be removed to prevent geometric
interference with the preserved region; however, those that are also
supported by the image condition (i.e., , belonging to Vcond) are
essential for realizing the target edit and must be retained:
V
(𝑇pruning)
𝐸
←V
(𝑇pruning)
𝐸
\ (Vconflict \ Vcond) .
(16)

<!-- page 6 -->
6
•
Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai
Table 2. Quantitative comparison on Edit3D-Bench. We evaluate preservation quality on unedited regions using Chamfer Distance (CD), PSNR, SSIM,
and LPIPS. Additionally, we assess condition alignment using DINO-I and CLIP-T. The best and second best results are highlighted in bold and underlined,
respectively. We also provide the reconstruction performance of our backbone, TripoSG, which serves as the upper bound for preservation quality.
Method
Time
Unedited Region Preservation
Condition Alignment
CD ↓
PSNR (M) ↑
SSIM (M) ↑
LPIPS (M) ↓
FID ↓
DINO-I ↑
CLIP-T ↑
MVEdit [Chen et al. 2024b]
∼160𝑠
0.188
21.90
0.91
0.13
47.05
0.81
27.03
Instant3DiT [Barda et al. 2025]
∼20𝑠
0.124
16.76
0.81
0.28
72.13
0.71
25.71
Trellis [Xiang et al. 2025]
∼600𝑠
0.014
29.22
0.97
0.04
33.09
0.91
27.87
VoxHammer [Li et al. 2025a]
∼600𝑠
0.018
27.05
0.95
0.05
31.13
0.90
28.08
Ours
∼200𝑠
0.011
29.63
0.97
0.04
32.63
0.92
27.75
TripoSG (VAE Encode+Decode) [Li et al. 2025d]
∼100𝑠
0.006
31.88
0.98
0.02
16.21
-
-
VecSet
Edit
MV-Adapter
Texture
Baking
Detail-
Preserving
Texture
Baking
Fig. 6. Illustration of our proposed Detail-Preserving Texture Baking.
Relying solely on the standard MV-Adapter leads to visual discrepancies
in the preserved regions (highlighted in red box). In contrast, our Detail-
Preserving Texture Baking effectively mitigates these errors, maintaining
the fidelity of the original unedited areas (highlighted in green box).
The two sets are defined as:
Vcond = TokenSeeding(V
(𝑇pruning)
RP
, 𝐼𝐸, 𝑀𝐼),
(17)
Vconflict = TokenGating(V
(𝑇pruning)
RP
, V𝑃).
(18)
This pruning strategy removes geometrically conflicting tokens
while safeguarding tokens that are necessary to satisfy the edit,
thereby maintaining both structural integrity and semantic fidelity.
Final Pipeline. The VecSet-Edit process concludes by running the
RePaint loop (from 𝑇repaint to 0) with pruning at 𝑇pruning (detailed
algorithm can be found in Algorithm 1 of Appendix D). To finalize
the asset, we apply Detail-Preserving Texture Baking, which synthe-
sizes textures that are consistent with both the editing instructions
and the original mesh details (as shown in Fig. 6 and Appendix C).
5
Experiments
Implementation Details. We utilize TripoSG [Li et al. 2025d] as
our backbone architecture. We set the RePaint starting timestep
𝑇repaint = 0.7, the pruning timestep 𝑇pruning = 0.6. The setting of
token selection and corresponding sensitivity analysis of these pa-
rameters can be found in Appendix D. We evaluate VecSet-Edit on
the following benchmark:
Edit3D-Bench [Li et al. 2025a]. An established dataset for mesh
editing containing 300 samples. Each sample provides a source mesh,
a 3D bounding box, and a target editing image. Notably, distinct from
baselines that rely on the 3D bounding box for localization, our method
uses the box only for evaluation.
5.1
Experimental Results
The experimental results are presented in Table 2 and Fig. 7.
Preservation and Efficiency. A core challenge in 3D editing is pre-
venting unintended alterations. By leveraging the VecSet-based back-
bone, our method achieves the lowest Chamfer Distance (CD)—21%
lower than the previous SOTA—along with superior image-based
metrics (PSNR, SSIM, LPIPS). This indicates that VecSet-Edit ef-
fectively maintains the structural integrity and visual fidelity of
unedited regions. In contrast, multi-view methods (e.g., MVEdit,
Instant3DiT) often suffer from global distortions due to mesh mis-
alignment, while voxel-based approaches (e.g., VoxHammer) exhibit
higher CD errors attributable to limited grid resolution. Beyond
preservation quality, VecSet-Edit achieves a 2× speedup com-
pared to Trellis and VoxHammer, further validating the efficiency
advantages of the VecSet backbone.
Condition Alignment. We utilize DINO-I and CLIP-T to evaluate
visual and semantic consistency. As our pipeline is designed for
image-guided editing, we observe a significant advantage in DINO-I
scores, indicating that our geometry faithfully captures the struc-
tural details of the input image. While our CLIP-T scores are slightly
lower than other baselines, they remain competitive, reflecting our
design priority: optimizing for strict visual (image) alignment rather
than broad text correspondence.
Qualitative Comparison. As evidenced in Fig. 7, VecSet-Edit ex-
hibits superior editing capabilities visually. Structurally, the topo-
logical flexibility of our backbone allows us to excel at retaining
intricate geometries, such as hands (rows 1, 2) and fans (rows 3,
4). Notably, we achieve robust preservation of unedited regions,
surpassing the voxel-based SOTA, VoxHammer, without relying on
explicit 3D bounding boxes. This visual success underscores the
precision of our Token Selection strategy and the effectiveness of
Token Pruning in eliminating geometric artifacts. Texturally, our

<!-- page 7 -->
VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image
•
7
Source Mesh
MVEdit
Instant3DiT
Trellis
Voxhammer
Ours
Input Condition
Fig. 7. Qualitative comparison on Edit3D-Bench (The meshes are rendered from two views 0◦and 90◦). The results show that VecSet-Edit achieves
superior preservation of source mesh details (such as hands and fans) while faithfully adhering to the target image condition.
Table 3. Quantitative results of ablation study on Vecset Edit.
CD
PSNR
LPIPS
DINO-I
CLIP-T
RePaint
0.024
24.35
0.07
0.88
27.25
+ Token Seeding
0.006
31.71
0.02
0.82
26.37
+ Token Gating
0.011
29.71
0.04
0.90
27.51
+ Token Pruning
0.011
29.63
0.04
0.92
27.75
- Detail-Preseving
0.011
25.17
0.06
0.89
27.68
method ensures high fidelity. By incorporating Detail-preserving
Texture Baking, we successfully maintain original appearance
details, preventing the texture blurring observed in other baselines.
Additional visualizations can be found in Fig. 12.
5.2
Ablation Study
We quantitatively dissect the contribution of each component in
Table 3. Initially, the integration of Token Seeding and Token Gat-
ing yields a marked improvement in preservation metrics, demon-
strating that 2D image conditions alone suffice for precise region
localization. Subsequently, the application of Token Pruning sig-
nificantly mitigates geometric errors (reflected in lower CD scores)
by actively filtering out redundant tokens that drift into the pre-
served regions. Finally, our Detail-Preserving Texture Baking
module serves as the final refinement step, without this step we can
observe the decline in unedited region preservation.
6
Conclusion
We present VecSet-Edit, the first training-free framework for lo-
calized mesh editing within LRM latent spaces. By uncovering the
VecSet Geometry Property, we exploit the spatial locality of un-
ordered tokens for precise manipulation. We propose Mask-guided
Token Seeding and Attention-aligned Token Gating to localize
edits using only 2D supervision, coupled with Drift-aware Token
Pruning to ensure geometric consistency during diffusion. Finally,
our Detail-preserving Texture Baking maintains high-fidelity de-
tails in unedited regions. Evaluations on Edit3D-Bench demonstrate
that VecSet-Edit significantly outperforms existing approaches in
preservation and alignment, bridging the gap between generative
3D models and controllable production workflows.
References
Roi Bar-On, Dana Cohen-Bar, and Daniel Cohen-Or. 2025. EditP23: 3D Editing via
Propagation of Image Prompts to Multi-View. arXiv preprint arXiv:2506.20652 (2025).
Amir Barda, Matheus Gadelha, Vladimir G Kim, Noam Aigerman, Amit H Bermano,
and Thibault Groueix. 2025. Instant3dit: Multiview inpainting for fast editing of 3d
objects. In Proceedings of the Computer Vision and Pattern Recognition Conference.
16273–16282.
Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang
Zheng. 2023. Masactrl: Tuning-free mutual self-attention control for consistent
image synthesis and editing. In Proceedings of the IEEE/CVF international conference
on computer vision. 22560–22570.
Hansheng Chen, Bokui Shen, Yulin Liu, Ruoxi Shi, Linqi Zhou, Connor Z Lin, Jiayuan
Gu, Hao Su, Gordon Wetzstein, and Leonidas Guibas. 2024b. 3d-adapter: Geometry-
consistent multi-view diffusion for high-quality 3d generation. arXiv preprint
arXiv:2410.18974 (2024).
Rui Chen, Jianfeng Zhang, Yixun Liang, Guan Luo, Weiyu Li, Jiarui Liu, Xiu Li, Xiaoxiao
Long, Jiashi Feng, and Ping Tan. 2025. Dora: Sampling and benchmarking for 3d
shape variational auto-encoders. In Proceedings of the Computer Vision and Pattern

<!-- page 8 -->
8
•
Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai
Recognition Conference. 16251–16261.
Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhon-
gang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. 2024a. Gaussianeditor: Swift
and controllable 3d editing with gaussian splatting. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 21476–21485.
Wei Cheng, Juncheng Mu, Xianfang Zeng, Xin Chen, Anqi Pang, Chi Zhang, Zhibin
Wang, Bin Fu, Gang Yu, Ziwei Liu, et al. 2025. Mvpaint: Synchronized multi-view
diffusion for painting anything 3d. In Proceedings of the Computer Vision and Pattern
Recognition Conference. 585–594.
Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. 2022. DiffEdit:
Diffusion-based semantic image editing with mask guidance. In International Con-
ference on Learning Representations.
Jiahua Dong and Yu-Xiong Wang. 2023. Vica-nerf: View-consistency-aware 3d editing
of neural radiance fields. Advances in Neural Information Processing Systems 36
(2023), 61466–61477.
Yiftach Edelstein, Or Patashnik, Dana Cohen-Bar, and Lihi Zelnik-Manor. 2025. Sharp-It:
A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation. In
Proceedings of the Computer Vision and Pattern Recognition Conference. 21458–21468.
Will Gao, Dilin Wang, Yuchen Fan, Aljaz Bozic, Tuur Stuyck, Zhengqin Li, Zhao Dong,
Rakesh Ranjan, and Nikolaos Sarafianos. 2025. 3d mesh editing using masked
lrms. In Proceedings of the IEEE/CVF International Conference on Computer Vision.
7154–7165.
Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo
Kanazawa. 2023. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceed-
ings of the IEEE/CVF international conference on computer vision. 19740–19750.
Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel
Cohen-or. 2022. Prompt-to-Prompt Image Editing with Cross-Attention Control. In
International Conference on Learning Representations.
Teng-Fang Hsiao, Bo-Kai Ruan, and Hong-Han Shuai. 2025a. Training-and-Prompt-
Free General Painterly Harmonization via Zero-Shot Disentenglement on Style and
Content References. In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 39. 3545–3553.
Teng-Fang Hsiao, Bo-Kai Ruan, Yi-Lun Wu, Tzu-Ling Lin, and Hong-Han Shuai. 2025b.
Tf-ti2i: Training-free text-and-image-to-image generation via multi-modal implicit-
context learning in text-to-image models. arXiv preprint arXiv:2503.15283 (2025).
Junchao Huang, Xinting Hu, Shaoshuai Shi, Zhuotao Tian, and Li Jiang. 2025b. Edit360:
2d image edits to 3d assets from any angle. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision. 16618–16628.
Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao,
and Lu Sheng. 2025a. Mv-adapter: Multi-view consistent image generation made
easy. In Proceedings of the IEEE/CVF International Conference on Computer Vision.
16377–16387.
Team Hunyuan3D, Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, Huiwen
Shi, Jingwei Huang, Junlin Yu, Kunhong Li, Penghao Wang, et al. 2025. Hunyuan3d-
omni: A unified framework for controllable generation of 3d assets. arXiv preprint
arXiv:2509.21245 (2025).
Heewoo Jun and Alex Nichol. 2023. Shap-E: Generating Conditional 3D Implicit
Functions. arXiv:2305.02463
Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli.
2025. Flowedit: Inversion-free text-based editing using pre-trained flow models.
In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19721–
19730.
Lin Li, Zehuan Huang, Haoran Feng, Gengxiong Zhuang, Rui Chen, Chunchao Guo,
and Lu Sheng. 2025a. Voxhammer: Training-free precise and coherent 3d editing in
native 3d space. arXiv preprint arXiv:2508.19247 (2025).
Peng Li, Suizhi Ma, Jialiang Chen, Yuan Liu, Congyi Zhang, Wei Xue, Wenhan Luo, Alla
Sheffer, Wenping Wang, and Yike Guo. 2025b. Cmd: Controllable multiview diffusion
for 3d editing and progressive generation. In Proceedings of the Special Interest Group
on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–10.
Weiyu Li, Jiarui Liu, Hongyu Yan, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and
Xiaoxiao Long. 2024. Craftsman3d: High-fidelity mesh generation with 3d native
generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979 (2024).
Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao
Wu, Jiarui Liu, Zihao Wang, et al. 2025c. Step1x-3d: Towards high-fidelity and
controllable generation of textured 3d assets. arXiv preprint arXiv:2505.07747 (2025).
Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao
Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. 2025d. Triposg: High-
fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint
arXiv:2502.06608 (2025).
Xingchao Liu, Chengyue Gong, et al. 2022. Flow Straight and Fast: Learning to Generate
and Transfer Data with Rectified Flow. In International Conference on Learning
Representations.
William E Lorensen and Harvey E Cline. 1998. Marching cubes: A high resolution 3D
surface construction algorithm. In Seminal graphics: pioneering efforts that shaped
the field. 347–353.
Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc
Van Gool. 2022. Repaint: Inpainting using denoising diffusion probabilistic models.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
11461–11471.
Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and
Stefano Ermon. 2021. SDEdit: Guided Image Synthesis and Editing with Stochastic
Differential Equations. In International Conference on Learning Representations.
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil
Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby,
Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-
Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve
Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2024.
DINOv2: Learning Robust Visual Features without Supervision. Transactions on
Machine Learning Research (2024).
Ipek Oztas, Duygu Ceylan, and Aysegul Dundar. 2025. 3D Stylization via Large Recon-
struction Model. In Proceedings of the Special Interest Group on Computer Graphics
and Interactive Techniques Conference Conference Papers.
Karran Pandey, Anita Hu, Clement Fuji Tsang, Or Perel, Karan Singh, and Maria
Shugrina. 2025. Painting with 3D Gaussian Splat Brushes. In Proceedings of the
Special Interest Group on Computer Graphics and Interactive Techniques Conference
Conference Papers. 1–10.
Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas
Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent diffusion
models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023).
Yansong Qu, Dian Chen, Xinyang Li, Xiaofan Li, Shengchuan Zhang, Liujuan Cao, and
Rongrong Ji. 2025. Drag your gaussian: Effective drag-based editing with score
distillation for 3d gaussian splatting. In Proceedings of the Special Interest Group on
Computer Graphics and Interactive Techniques Conference Conference Papers.
Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. 2023. Tex-
ture: Text-guided texturing of 3d shapes. In ACM SIGGRAPH conference proceedings.
1–11.
Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, and Qi Tian. 2024a. Gaus-
sianeditor: Editing 3d gaussians delicately with text instructions. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition. 20902–20911.
Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu,
Chongxuan Li, Hang Su, and Jun Zhu. 2024b. CRM: Single Image to 3D Textured
Mesh with Convolutional Reconstruction Model. In European Conference on Com-
puter Vision. Springer.
Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian Reid, Philip Torr, and Vic-
tor Adrian Prisacariu. 2024. Gaussctrl: Multi-view consistent text-driven 3d gaussian
splatting editing. In European Conference on Computer Vision. Springer.
Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong
Chen, Xin Tong, and Jiaolong Yang. 2025. Structured 3d latents for scalable and
versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition
Conference. 21469–21480.
Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. 2024.
Instantmesh: Efficient 3d mesh generation from a single image with sparse-view
large reconstruction models. arXiv preprint arXiv:2404.07191 (2024).
Ziyang Yan, Lei Li, Yihua Shao, Siyu Chen, Zongkai Wu, Jenq-Neng Hwang, Hao Zhao,
and Fabio Remondino. 2024. 3dsceneeditor: Controllable 3d scene editing with
gaussian splatting. arXiv preprint arXiv:2412.01583 (2024).
Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao,
Xinhai Liu, Xinzhou Wang, Qingxiang Lin, Jiaao Yu, Lifu Wang, Jing Xu, Zebin He,
Zhuo Chen, Sicong Liu, Junta Wu, Yihang Lian, Shaoxiong Yang, Yuhong Liu, Yong
Yang, Di Wang, Jie Jiang, and Chunchao Guo. 2025. Hunyuan3D 1.0: A Unified
Framework for Text-to-3D and Image-to-3D Generation. arXiv:2411.02293
Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text com-
patible image prompt adapter for text-to-image diffusion models. arXiv preprint
arXiv:2308.06721 (2023).
Xianfang Zeng, Xin Chen, Zhongqi Qi, Wen Liu, Zibo Zhao, Zhibin Wang, Bin Fu,
Yong Liu, and Gang Yu. 2024. Paint3d: Paint anything 3d with lighting-less texture
diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. 4252–4262.
Dingxi Zhang, Yu-Jie Yuan, Zhuoxun Chen, Fang-Lue Zhang, Zhenliang He, Shiguang
Shan, and Lin Gao. 2025. Stylizedgs: Controllable stylization for 3d gaussian splatting.
IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).
Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei
Yang, Lan Xu, and Jingyi Yu. 2024b. Clay: A controllable large-scale generative
model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG) 43,
4 (2024), 1–20.
Yuqing Zhang, Yuan Liu, Zhiyu Xie, Lei Yang, Zhongyuan Liu, Mengzhou Yang, Runze
Zhang, Qilong Kou, Cheng Lin, Wenping Wang, et al. 2024a. Dreammat: High-
quality pbr material generation with geometry-and light-aware diffusion models.
ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–18.
Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei
Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. 2025. Hunyuan3d 2.0:

<!-- page 9 -->
VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image
•
9
Scaling diffusion models for high resolution textured 3d assets generation. arXiv
preprint arXiv:2501.12202 (2025).
Yang Zheng, Hao Tan, Kai Zhang, Peng Wang, Leonidas Guibas, Gordon Wetzstein,
and Wang Yifan. 2025. SplatPainter: Interactive Authoring of 3D Gaussians from
2D Edits via Test-Time Training. arXiv preprint arXiv:2512.05354 (2025).
Zijun Zhou, Yingying Deng, Xiangyu He, Weiming Dong, and Fan Tang. 2025. Multi-
turn Consistent Image Editing. arXiv preprint arXiv:2505.04320 (2025).
Jingyu Zhuang, Di Kang, Yan-Pei Cao, Guanbin Li, Liang Lin, and Ying Shan. 2024.
Tip-editor: An accurate 3d editor following both text-prompts and image-prompts.
ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–12.
Jingyu Zhuang, Chen Wang, Liang Lin, Lingjie Liu, and Guanbin Li. 2023. Dreameditor:
Text-driven 3d scene editing with neural fields. In SIGGRAPH Asia 2023 Conference
Papers. 1–10.

<!-- page 10 -->
10
•
Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai
Source Mesh
RePaint
+Token Seeding
+Token Gating
+Token Pruning
Input Condition
Fig. 8. Visual ablation of VecSet-Edit. We demonstrate the necessity of modules: Token Seeding localizes the edit to prevent the global distortion seen in
naive RePaint; Token Gating expands the selection to ensure coverage of the target region; and Token Pruning removes outlier tokens and geometric artifacts.
VecSet VAE
Marching Cubes
Grid Points
SDF
xT
VecSet DiT
Fig. 9. Illustration of VecSet VAE and VecSet DiT.
Generate
Mesh
VecSet
Edit
Defected 
Repaired
Fig. 10. Illustration of using VecSet-Edit to calibrate mesh. While the
generated mesh via TripoSG yields defective geometry due to single-view
limitations (row 1), VecSet-Edit utilizes additional views to refine these
defects, ensuring global consistency while preserving the original mesh
details.
Source
Mesh
w/o
Detail-
Preserving
w/
Detail-
Preserving
Fig. 11. Visual ablation of Detail-Preserving Texture Baking. Our pro-
posed baking strategy faithfully maintains the color fidelity of the edited
mesh while preserving high-frequency details, even in views not visible in
the editing condition 𝐼𝐸. Key differences are highlighted in red boxes.

<!-- page 11 -->
VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image
•
11
Source Mesh
MVEdit
Instant3DiT
Trellis
Voxhammer
Ours
Input Condition
Fig. 12. More qualitative comparison on Edit3D-Bench. Our proposed VecSet-Edit shows superior performance across different input scenarios.

<!-- page 12 -->
12
•
Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai
A
Details of VecSet-based LRM
The illustration of overall VecSet Encoding and Diffusion process
can be found in Fig. 9, other details can be found below.
A.1
VecSet VAE
To encode a mesh S, we first sample a point cloud P ∈R𝑁𝑝×6 from
the mesh surface, containing both position and normal information.
A learnable subset of tokens, initialized as P′ ∈R𝑁𝑝′ ×6, serves as
the initial latent queries (typically 𝑁𝑝= 50, 000 and 𝑁𝑝′ = 2048).
These surface points are integrated into the latent queries via cross-
attention to produce the VecSet tokens V, which encode the geo-
metric information of the input mesh:
V′ = CrossAttn(PosEmb(P′), PosEmb(P)),
(19)
V = SelfAttn(𝑖) (V′),
𝑖∈{1, . . . , 𝐿enc},
(20)
where CrossAttn(𝑄, 𝐾) denotes a cross-attention layer where queries
attend to keys, and SelfAttn(𝑖) represents the 𝑖-th self-attention
layer.
Once the VecSet representation V is obtained, we can decode it
into a Signed Distance Field (SDF). For an arbitrary query position
𝑥∈R3, the predicted signed distance 𝑑is computed as:
eV = SelfAttn(𝑖) (Linear(V)) ,
𝑖∈{1, . . . , 𝐿dec},
(21)
𝑑(𝑥) = CrossAttn

PosEmb(𝑥), eV

.
(22)
Finally, the explicit mesh S′ is extracted by applying Marching
Cubes [Lorensen and Cline 1998] on the predicted SDF at a spec-
ified resolution. In the following discussion, we abbreviate the
VecSet encoding process as Encode and the decoding pipeline (in-
cluding SDF prediction and Marching Cubes) as Decode, i.e., SR =
Decode(Encode(S)).
B
Details of Token Selection
B.1
Attentive-Layer Selection
Inspired by prior text-to-image editing works [Cao et al. 2023; Coua-
iron et al. 2022; Hertz et al. 2022; Hsiao et al. 2025a,b; Kulikov et al.
2025; Meng et al. 2021; Zhou et al. 2025], which demonstrate that
cross-attention maps provide effective zero-shot cues for semantic
localization, we adapt this principle to the 3D VecSet DiT framework
to identify geometry-relevant tokens from a 2D edit mask.
Recall from Eq. (3) that VecSet tokens V attend to image features
ℎ𝐼through cross-attention. We denote the resulting cross-attention
map as
A(𝑙,𝑡)
cross ∈R𝑁𝑝×(𝐻𝑊),
(23)
where 𝑁𝑝is the number of VecSet tokens, and (𝑙,𝑡) index the DiT
block and diffusion timestep, respectively. Each entry A(𝑙,𝑡)
cross[𝑖, 𝑗]
measures the attention weight from the 𝑖-th 3D token to the 𝑗-th
image pixel.
To quantify the relevance of a VecSet token to the user-specified
edit region, we aggregate its attention over the masked pixels:
𝑎(𝑙,𝑡)
𝑀
(𝑖) =
𝐻𝑊
∑︁
𝑗=1
A(𝑙,𝑡)
cross[𝑖, 𝑗] · ¯𝑀𝐼[𝑗],
(24)
where 𝑎(𝑙,𝑡)
𝑀
(𝑖) is a scalar that reflects how strongly the 𝑖-th token
aligns with the masked region of 𝐼𝑆and ¯𝑀𝐼∈{0, 1}𝐻𝑊is flatten
from 𝑀𝐼.2
A naive strategy is to average𝑎(𝑙,𝑡)
𝑀
(𝑖) across all blocks and timesteps
to obtain a global relevance score. However, we observe that seman-
tic alignment varies significantly across DiT blocks, consistent with
findings in prior diffusion analyses [Zhou et al. 2025]. Some lay-
ers exhibit sharp, condition-aware attention, while others produce
diffuse or uninformative patterns.
To identify informative layers, we analyze the sharpness of their
cross-attention distributions. For each DiT block 𝑙, we compute the
Kullback–Leibler (KL) divergence between token-wise attention and
its marginal distribution:
D (𝑙) =
∑︁
𝑡
𝑁𝑝
∑︁
𝑖=1
KL

A(𝑙,𝑡)
cross[𝑖, :]
 ¯a(𝑙,𝑡)
M

,
(25)
where ¯a(𝑙)
cross ∈R𝐻𝑊is the marginal attention defined by
¯a(𝑙,𝑡)
cross[𝑗] =
𝑁𝑝
∑︁
𝑖=1
A(𝑙,𝑡)
cross[𝑖, 𝑗].
(26)
Higher D (𝑙) indicates a sharper and more semantically aligned
attention pattern (see Fig. 4). We therefore select the top-𝐾most
informative layers:
𝐿attn = TopK𝑙
 D (𝑙).
(27)
B.2
Adaptive Threshold
Using only these layers, we compute the refined relevance score by
averaging across selected layers and timesteps:
˜𝑎𝑀(𝑖) = E𝑙∈𝐿attn,𝑡
h
𝑎(𝑙,𝑡)
𝑀
(𝑖)
i
.
(28)
Empirically, ˜𝑎𝑀exhibits a long-tailed distribution, where only a
small subset of tokens strongly aligns with the edit region. We
therefore define an adaptive threshold based on the top-𝛼% scores:
𝜏𝐼( ˜𝑎𝑀, 𝛼𝐼) = 𝛼𝐼
100 · mean Top10%( ˜𝑎𝑀) ,
(29)
similarily the computation of𝜏𝐴is also following Eq. (29) and control
by 𝛼𝐴.
C
Detail-Preserving Texture Baking
Texture baking is a fundamental step in automated mesh generation
pipelines [Cheng et al. 2025; Huang et al. 2025a; Hunyuan3D et al.
2025; Li et al. 2025d; Richardson et al. 2023; Zeng et al. 2024; Zhang
et al. 2024a; Zhao et al. 2025]. Given our focus on editing meshes
via single-image instructions, we prioritize methods designed for
single-view conditioned texture synthesis [Cheng et al. 2025; Huang
et al. 2025a; Li et al. 2025d; Zhao et al. 2025]. Specifically, we adopt
MV-Adapter [Huang et al. 2025a] as our texture backbone.
2Attention values are averaged across heads in practice.

<!-- page 13 -->
VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image
•
13
Geometry Difference
MV-Adapter + RePaint
Noise Reference
Fig. 13. Overview of the Detail-Preserving Texture Baking pipeline.
We compute geometric difference masks between the original and edited
meshes to guide the MV-Adapter, ensuring that texture generation is re-
stricted solely to the edited regions while preserving original details.
C.1
MV-Adapter
The MV-Adapter [Huang et al. 2025a] module operates in two stages.
The first stage, Image-Geometry-to-Multiview, begins by rendering
6-view surface normals from the source mesh S. These normals,
combined with the condition image 𝐼𝐸, guide a fine-tuned text-to-
image model (based on Stable Diffusion [Podell et al. 2023] and
IP-Adapter [Ye et al. 2023]) to synthesize consistent multi-view RGB
images. This generation process is governed by:
{𝐼mv
𝑗
}6
𝑗=1 = MV-Adapter({N𝑆
𝑗}6
𝑗=1, 𝐼𝐸),
(30)
where N𝑆
𝑗denotes the rendered surface normal for view 𝑗. The six
views consist of four horizontal azimuths {0◦, 90◦, 180◦, 270◦} and
two vertical elevations at {90◦, 270◦}.
The second stage, Texture Projection, projects these generated
views onto the mesh surface. Utilizing a differentiable renderer,
we perform gradient-based inverse rendering to optimize the UV
texture map, ensuring the rendered appearance aligns with the
generated multi-view images:
Stextured = TextureProjection(S, {𝐼mv
𝑗
}6
𝑗=1).
(31)
However, applying this global baking strategy naively can be
suboptimal. As illustrated in Fig. 6, regenerating the texture for the
entire mesh often degrades the high-frequency details of preserved
regions that should ideally remain unchanged.
C.2
Geometry-aware Texture RePaint
Leveraging the property that VecSet-Edit strictly preserves the geom-
etry of the reference mesh S outside the edited region, we optimize
the texturing process by exclusively updating areas with geometric
changes.
As shown in Fig. 13, we first quantify the geometric discrepancy
between the source mesh S and the edited output Sout in the ren-
dered view space. This yields a set of difference masks {Mmv
𝑗}6
𝑗=1,
defined as:
Mmv
𝑗
= I

|N𝑆out
𝑗
−N𝑆
𝑗| > 𝜏texture

,
(32)
where N𝑗represents the rendered normal map of view 𝑗, and 𝜏texture
is the sensitivity threshold for detecting geometric shifts.
Algorithm 1: VecSet-Edit
Input
:Reference VecSet V, source image 𝐼𝑆, edit image 𝐼𝐸,
mask 𝑀𝐼, repaint start 𝑇repaint, pruning time 𝑇pruning
Output:Edited VecSet Vout
1 V𝐼←TokenSeeding(V, 𝐼𝑆, 𝑀𝐼);
2 V𝐸←TokenGating(V, V𝐼);
3 Let I𝐸be indices such that V𝐸= Gather(V, I𝐸);
4 Sample 𝜖𝑃∼N (0, I);
5 V
(𝑇repaint)
RP
←(1 −𝑇repaint) 𝜖𝑃+𝑇repaint V;
6 V
(𝑇repaint)
𝐸
←Gather(V
(𝑇repaint)
RP
, I𝐸);
7 V
(𝑇repaint)
𝑃
←V
(𝑇repaint)
𝑅𝑃
\ 𝑉
(𝑇repaint)
𝐸
;
8 for 𝑡= 𝑇repaint, 𝑇repaint −Δ𝑡, . . . , Δ𝑡do
9
𝑣pred ←𝑢𝜃(V(𝑡)
RP ,ℎ𝐼,𝑡) ; // Full context prediction
10
V(𝑡−Δ𝑡)
𝐸
←V(𝑡)
𝐸
−Gather(𝑣pred, I𝐸) · Δ𝑡;
11
V(𝑡−Δ𝑡)
𝑃
←(1 −(𝑡−Δ𝑡)) 𝜖𝑃+ (𝑡−Δ𝑡) V𝑃;
12
if 𝑡= 𝑇pruning then
13
Vcond ←TokenSeeding(V(𝑡−Δ𝑡)
RP
, 𝐼𝐸, 𝑀𝐼);
14
Vconflict ←TokenGating(V(𝑡−Δ𝑡)
RP
, V𝑃);
15
V(𝑡−Δ𝑡)
𝐸
←V(𝑡−Δ𝑡)
𝐸
\ (Vconflict \ Vcond);
// Pruning
16
V(𝑡−Δ𝑡)
RP
←V(𝑡−Δ𝑡)
𝐸
⊕V(𝑡−Δ𝑡)
𝑃
;
17 return Vout ←V(0)
RP ;
These masks serve as spatial guidance for the MV-Adapter, re-
stricting the generative process to the modified regions. This mech-
anism mirrors the logic of our VecSet RePaint strategy (Eq. (15)),
but operates in the 2D pixel domain rather than the latent token
space. Consequently, we effectively preserve the high-frequency
texture details of the original mesh S while seamlessly propagating
the semantic information from the condition image 𝐼𝐸to the new
geometry Sout.
Formally, the multi-view in-painting process is defined as:
{𝐼RePaint
𝑗
}6
𝑗=1 = MV-RePaint({N𝑆out
𝑗
}, {Mmv
𝑗}, 𝐼𝐸),
(33)
followed by the final texture projection:
Stextured
out
= TextureProjection(Sout, {𝐼RePaint
𝑗
}6
𝑗=1).
(34)
D
VecSet-Edit Settings
All experiments were conducted on a single NVIDIA H100 GPU.
The VRAM requirement ranges from 22 to 30 GB depending on the
input complexity. We summarize the default hyperparameter con-
figurations of VecSet-Edit below. For the hyperparameter sensitivity
analysis, we varied one specific parameter while keeping the others
fixed at their default values, the evaluation results can be found in
Fig. 14. 3
D.1
LRM Backbone
Classifier-free Guidance Scale (𝑟= 10). This parameter controls
the alignment of the generated mesh with the condition image.
3Sensitivity experiments were conducted on a randomly selected 50% subset of the
Edit3D-Bench to reduce computational overhead

<!-- page 14 -->
14
•
Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai
Empirically, we observed that the mesh geometry remains relatively
robust to variations in this parameter.
Number of Inference Steps (𝑛step = 50). We adopt the default
configuration from TripoSG [Li et al. 2025d]. While increasing this
value typically improves generation quality, it also linearly increases
inference latency.
D.2
Editing Process
RePaint Timestep (𝑇repaint = 0.7). This parameter governs the
editing strength. A lower𝑇repaint constrains the output too strictly to
the source mesh, limiting editability. Conversely, an excessively high
𝑇repaint removes the geometric prior, leading to incoherence between
the edited and unedited regions. Crucially, this setting is intention-
dependent: for structural changes (e.g., removing an object), a higher
𝑇repaint is required; for tasks requiring strict coherence (e.g., local
detail refinement), a lower value is preferred.
Pruning Timestep (𝑇pruning = 0.6). This determines when to filter
out outlier tokens. Pruning too early (at high noise levels) compro-
mises geometric quality, as valid detail tokens may be misclassified
as outliers. Pruning too late leaves insufficient denoising steps for
the VecSet to realign the remaining tokens, resulting in disconnected
geometry.
Seeding Sensitive (𝛼I = 0.7). The seeding sensitive controls the
precision of the initial token selection. We keep this value high to en-
sure high recall (i.e., accurately locating all potential image-related
tokens). Extensive ablation was not performed on this parameter, as
the final selection quality is primarily dominated by the subsequent
gating step.
Gating Sensitive (𝛼A = 0.5). This parameters regulates the expan-
sion of the selected token set via attention mechanisms. Higher
values yield tighter constraints (smaller editing areas), while lower
values allow for broader structural changes. Similar to𝑇repaint, this is
intention-dependent. For example, replacing an entire head requires
a higher 𝛼A, whereas swapping only facial details requires a lower
𝛼A.
D.3
Texturing Process
Texture Difference Threshold (𝜏texture = 0.005). This controls
the sensitivity of the preservation mask. A larger value preserves
more of the original texture but may lead to misalignment with the
condition 𝐼𝐸. A lower value allows the condition image to influence
a larger portion of the surface, ensuring better visual alignment at
the cost of original texture preservation.
MV-RePaint Timestep (𝑇MV-repaint = 1.0). This controls the noise
level for texture generation. We use the maximum value to ensure
full denoising, guaranteeing global coherence in the generated tex-
ture.
E
Limitations
Hyperparameter Sensitivity. Although each proposed module
serves a distinct role in the editing pipeline, disentangling their
effects remains challenging. Unlike end-to-end methods controlled
solely by text instructions, our approach involves multiple thresh-
olds (e.g., for token selection and pruning) that require careful tuning
to balance editing strength with preservation.
Texture-Geometry Dependency. While our Detail-preserving Tex-
ture Baking effectively retains the original textural information, it
relies on the accurate localization of edits. In cases where the editing
process inadvertently alters the geometry of the preserved region
(i.e., mask leakage), the texture projection may become misaligned
or produce artifacts.

<!-- page 15 -->
VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image
•
15
Fig. 14. Sensitive test of VecSet-Edit hyperparameters.
