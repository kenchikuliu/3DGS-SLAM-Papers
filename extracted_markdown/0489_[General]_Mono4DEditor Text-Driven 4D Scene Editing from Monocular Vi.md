<!-- page 1 -->
Preprint
MONO4DEDITOR: TEXT-DRIVEN 4D SCENE EDITING
FROM MONOCULAR VIDEO VIA POINT-LEVEL LOCAL-
IZATION OF LANGUAGE-EMBEDDED GAUSSIANS
Jin-Chuan Shi1,2∗Chenye Su2∗
Jiajun Wang2
Ariel Shamir3
Miao Wang2†
1 Zhejiang University, China
2 State Key Laboratory of Virtual Reality Technology and Systems, Beihang University, China
3 Reichman University, Israel
Input Monocular Videos
View 1
View 2
View 1
View 2
To a swan covered in cosmic galaxy patterns.
View 1
View 2
View 1
View 2
To a brown Ragdoll cat.
View 1
View 2
View 1
View 2
Timestep 2
Edited 4D Scenes
View 1
View 2
Turn the bear into a panda, and turn leaves into autumn leaves. 
View 1
View 2
Timestep 1
Timestep 2
Timestep 1
Timestep 1
Timestep 2
Timestep 3
Timestep 4
Figure 1: Our approach Mono4DEditor allows users to edit 4D scenes from casual monocular video
with text instruction. Mono4DEditor achieves precise, high-quality editing of the instructed content
while maintaining irrelevant regions unchanged.
ABSTRACT
Editing 4D scenes reconstructed from monocular videos based on text prompts is
a valuable yet challenging task with broad applications in content creation and vir-
tual environments. The key difficulty lies in achieving semantically precise edits
in localized regions of complex, dynamic scenes, while preserving the integrity
of unedited content. To address this, we introduce Mono4DEditor, a novel frame-
work for flexible and accurate text-driven 4D scene editing. Our method aug-
ments 3D Gaussians with quantized CLIP features to form a language-embedded
dynamic representation, enabling efficient semantic querying of arbitrary spatial
regions. We further propose a two-stage point-level localization strategy that first
selects candidate Gaussians via CLIP similarity and then refines their spatial ex-
tent to improve accuracy. Finally, targeted edits are performed on localized regions
using a diffusion-based video editing model, with flow and scribble guidance en-
suring spatial fidelity and temporal coherence. Extensive experiments demonstrate
that Mono4DEditor enables high-quality, text-driven edits across diverse scenes
and object types, while preserving the appearance and geometry of unedited areas
and surpassing prior approaches in both flexibility and visual fidelity.
∗Equal Contribution. Work done during M.S. study at Beihang University.
†Corresponding author.
1
arXiv:2510.09438v1  [cs.CV]  10 Oct 2025

<!-- page 2 -->
Preprint
1
INTRODUCTION
Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) and 3D Gaussian Splatting (3DGS) (Kerbl
et al., 2023) have emerged as powerful representations for modeling photorealistic 3D scenes, which
have enabled a wide range of downstream applications, including novel view synthesis, relighting,
semantic embedding, and scene editing. In particular, 3DGS achieves high-fidelity real-time render-
ing and has quickly become a practical tool for 3D content creation. To reflect the dynamic nature
of the real world, recent works have extended these static representations to dynamic scenes (Fang
et al., 2022; Yang et al., 2024; Stearns et al., 2024), supporting tasks such as dynamic reconstruction,
motion tracking, and animation. In this work, we specifically focus on 4D scenes reconstructed from
monocular videos, which are more practical in real-world scenarios but inherently more challenging
than multi-view settings due to limited observations and ambiguities in geometry and motion.
Despite recent progress, editing 4D scenes remains a challenging and underexplored problem. A
key limitation is the lack of fine-grained control over arbitrary objects or regions specified by the
natural language. Recent works (Mou et al., 2024; He et al., 2024) enable text-driven editing in 4D
scenes, but rely on 2D diffusion models to guide edits, which often introduce unintended modifica-
tions in irrelevant regions due to the absence of precise localization mechanisms. In contrast, latest
methods for static scenes (Xu et al., 2024; Zhang et al., 2024; Zhuang et al., 2024; Liu et al., 2024a)
demonstrate that text-driven editing of specific objects or parts is feasible when accurate spatial lo-
calization is available. However, these techniques are not designed for monocular dynamic settings
and fail to account for object motion or temporal consistency. Ideally, a 4D editing system should
allow users to describe desired changes with text prompts, selectively modify only relevant regions
in space and time, and preserve the appearance and motion of all other content.
To address these challenges, we propose Mono4DEditor, a text-driven editing framework for 4D
scenes reconstructed from monocular videos. Our key idea is to represent dynamic scenes with
language-embedded Gaussians, where each 3D Gaussian is enriched with compact CLIP-based
semantic features. This representation enables semantic localization directly in 3D, forming the
basis for point-level localization and text-driven editing. In addition, diffusion-based video editing
models provide richer control conditions and can generate high-quality, temporally consistent edits
from text prompts, making them particularly beneficial for dynamic scene editing and well-suited
to our monocular setting. Specifically, our approach consists of three main stages: (1) we construct
a dynamic 3D Gaussian field augmented with quantized CLIP features, enabling each Gaussian to
carry natural language semantics; (2) we introduce a point-level localization module that identifies
and refines Gaussians relevant to a user-provided text query by combining 2D semantic supervision
with 3D decoding; (3) we update only the localized Gaussians using guidance from a diffusion-
based video editing model, leveraging optical flow and scribble to maintain spatial precision and
temporal coherence. This pipeline allows for high-fidelity, region-specific edits while preserving the
motion and appearance of unedited regions.
We evaluate Mono4DEditor on a diverse set of dynamic scenes that encompass both foreground and
background elements with varying appearances and motion patterns. As shown in the experiments,
our method accurately localizes target regions and generates high-quality, temporally coherent 4D
edits while preserving the background, highlighting the flexibility and effectiveness of our frame-
work for text-driven 4D scene editing.
This work specifically focuses on editing 4D scenes reconstructed from monocular videos, a prac-
tical yet challenging setting that differs from multi-view scenarios by providing only limited obser-
vations. Our contributions are threefold:
• First, we propose a unified framework for text-driven 4D scene editing that integrates language-
embedded Gaussian representations with diffusion-based video editing models, enabling precise
and temporally consistent edits.
• Second, we develop a novel point-level localization strategy as a key component of this frame-
work, which accurately identifies and refines editable regions to support flexible semantic control.
• Third, we conduct comprehensive experiments and ablation studies, showing that combining
language-embedded Gaussians with video editing models enables region-specific, temporally co-
herent edits while preserving unedited content.
2

<!-- page 3 -->
Preprint
2
RELATED WORK
2.1
RADIANCE FIELDS FOR DYNAMIC SCENES
Given the success of NeRF, several works have extended it by introducing spatiotemporal
fields (Fridovich-Keil et al., 2023; Shao et al., 2023) or deformation fields (Park et al., 2021;
Pumarola et al., 2021; Fang et al., 2022) to model scene dynamics. Recent works based on 3DGS
have achieved more efficient reconstruction and rendering of dynamic scenes by using deformation
fields (Yang et al., 2024; Huang et al., 2024) or high-dimensional Gaussian fields (Duan et al., 2024).
Meanwhile, many works (Wang et al., 2025; Lei et al., 2024; Stearns et al., 2024) have integrated
priors such as camera estimation, depth estimation, and 2D tracking for initialization, and have
learned a dynamic Gaussian field, realizing dynamic scene reconstruction from monocular video.
2.2
LANGUAGE-EMBEDDED RADIANCE FIELDS
Recent work has explored embedding language features into neural scene representations to en-
able semantic querying and editing. LERF (Kobayashi et al., 2022; Kerr et al., 2023) injects CLIP
features into NeRF for 3D region retrieval and interaction. Later works extend this idea to 3D Gaus-
sians (Shi et al., 2024; Qin et al., 2024; Jiang, 2023), achieving faster and more accurate querying
by attaching language features to Gaussians. However, these methods decode language queries in
2D space and cannot directly identify relevant Gaussians in 3D. OpenGaussian (Wu et al., 2024)
introduces point-level localization but is limited to static scenes. 4D-LangSplat (Li et al., 2025) and
Feature4X (Zhou et al., 2025) embeds semantics in dynamic Gaussians but lacks point-level preci-
sion and is not suitable for 4D scenes editing. In contrast, our method achieves accurate point-level
localization of language-embedded Gaussians, enabling precise 4D editing.
2.3
TEXT-DRIVEN RADIANCE FIELDS EDITING
Text-driven editing has been explored in static 3D scenes using image diffusion models (Haque
et al., 2023; Chen et al., 2024), where edits are guided by text but lack spatial precision, resulting
in unexpected changes outside the target area. In addition, recent methods for static scenes (Zhuang
et al., 2023; Xu et al., 2024; Zhang et al., 2024; Zhuang et al., 2024; Liu et al., 2024a) demonstrate
that fine-grained, text-driven editing is achievable with accurate semantic grounding. To handle
dynamic content, recent works (He et al., 2024; Mou et al., 2024; Shao et al., 2024; Kwon et al.,
2025) extend editing to 4D by combining frame-wise diffusion with temporal constraints, which
can’t accurately localize editing regions in 3D. Instruct-4DGS (Kwon et al., 2025) further needs
multi-view videos as input. Building on this insight, our method introduces point-level localization
of language-embedded Gaussians and leverages the video diffusion model, enabling selective, text-
guided edits with high spatial precision and temporal coherence in 4D scenes. The related works on
text-driven video editing are discussed in Appendix A.
3
METHOD
The input to our method is a monocular video of a 4D scene, along with a natural language prompt
describing the desired edit. Our goal is to modify any user-specified region of the scene based on a
text prompt, while leaving the rest of the scene untouched. The target region can belong to any part
of the scene, including static or dynamic elements.
3.1
PRELIMINARY: 3D GAUSSIAN SPLATTING
3D Gaussian Splatting (Kerbl et al., 2023) synthesizes photorealistic scenes by aggregating numer-
ous colored 3D Gaussians, which are projected onto image planes via differentiable rasterization.
Specifically, a 3D scene is represented by a set of Gaussians G, where each Gaussian is parameter-
ized by its center µ ∈R3, rotation R ∈SO(3), scale s ∈R3, opacity o ∈R, and color c ∈R3.
Given a camera C with known intrinsics and extrinsics, the Gaussians are then projected onto the
image plane and composited through a differentiable rasterizer R, yielding the final image:
I = R(µ, R, s, o, c; C).
(1)
3

<!-- page 4 -->
Preprint
1. Reconstruction
2. Localization
Video
Edit
Model
3. Editing
Input Video
Index Maps
Quantized
CLIP Features
…
Codebook
Language-Embedded 
Dynamic Gaussians
Localized Dynamic 
Gaussians
Localized Dynamic 
Gaussians
n epochs
Query Text:“bear”
Localize
Eq. (14)
m epochs
Precision-oriented 
Refinement
Eq. (16)
Localize
Eq. (14)
Recall-oriented 
Refinement
Eq. (15)
Edit Text: 
“a panda …”
Scribble
Edited 4D Scenes
Video Guidance
Control
Flow
RGB Loss
Depths
Tracks
Cameras
Figure 2: Overview of our method. Given a monocular video , we construct a Language-Embedded
Dynamic Gaussian field by enriching 3D Gaussians with quantized CLIP features (Section 3.2). We
then perform point-level localization to identify Gaussians relevant to the query, using 2D relevance
maps and 3D semantic decoding (Section 3.3). Finally, we apply text-driven editing with a diffusion-
based video model, modifying only the localized Gaussians to produce temporally consistent and
spatially precise edits (Section 3.4). The colored visualization in the Language-Embedded Dynamic
Gaussian field shows PCA results after semantic feature splatting.
3.2
LANGUAGE-EMBEDDED DYNAMIC GAUSSIANS
Inspired by recent advances in language-embedded fields (Kerr et al., 2023; Shi et al., 2024; Qin
et al., 2024), we propose a 4D Gaussian field enriched with semantic features from the input monoc-
ular video. By embedding language semantics into dynamic Gaussians, we enable region-specific
editing through natural text guides. This approach allows for precise modifications to targeted areas,
while preserving the unchanged content by freezing non-target Gaussians.
Previous work (Zhou et al., 2025) embeds language features into dynamic Gaussians from monoc-
ular video, enabling tasks such as segmentation. However, this approach often sacrifices rendering
efficiency and semantic fidelity due to the complex feature distillation and interpolation process. In
contrast, we adopt a quantization-based feature compression strategy from LEGaussians (Shi et al.,
2024), which efficiently encodes CLIP features while maintaining high-fidelity reconstruction and
enabling faster training, making it more suitable for real-time text-driven dynamic scene editing.
Data preprocessing. To reconstruct the dynamic field with semantic features from the input video,
we apply standard preprocessing steps: (1) extracting the camera intrinsics and extrinsics for each
frame, (2) extracting dynamic masks, monocular depth, and long-range 2D point tracks for fore-
ground pixels in each frame, and (3) obtaining pixel-level CLIP features for each frame. The first
two pre-processing steps are crucial for reconstructing the dynamic Gaussian field from monocular
video. We refer to previous works (Wang et al., 2025; Stearns et al., 2024; Lei et al., 2024) to extract
camera poses and obtain dynamic masks, depth, and 2D tracks using off-the-shelf methods.
Step three serves as the foundation for embedding semantics into the dynamic Gaussian field. First,
we use SAM2 (Ravi et al., 2024) to generate multi-class tracking masks for the video. For each
frame, we crop the image using these masks and then pass the cropped regions through the CLIP
image encoder (Radford et al., 2021) to obtain feature embeddings. These CLIP features are as-
signed to the corresponding pixels in the mask, embedding semantic information at the pixel level.
The video consists of t frames, each of size h×w, and the CLIP features for each frame are extracted
at the pixel level, resulting in a feature tensor Fclip ∈Rt×h×w×c, where c represents the number of
CLIP feature channels. Following the approach in LEGaussians (Shi et al., 2024), we quantize the
extracted CLIP features using a learnable codebook B ∈RN×c, where N is the number of code-
book entries. After quantization, we obtain a learned codebook and the corresponding index map
Mindex ∈Rt×h×w×1, which stores the closest codeword index for each pixel in each frame. These
index maps and the video-level semantic codebook B are used to embed semantics into the dynamic
Gaussian field. More details of the quantization of CLIP features will be presented in Appendix B.1.
Language-Embedded Dynamic Gaussians. Following the Shape-of-Motion (Wang et al., 2025),
we represent motion as a rigid transformation in SE(3) applied to canonical 3D Gaussians to model
4

<!-- page 5 -->
Preprint
dynamic parts of scenes, and the others remain static. At time t, the transformed position and ori-
entation are µt = Rtµ + tt and Rt = RtR, where Tt = (Rt, tt) ∈SE(3) denotes the rigid
transformation from the canonical frame to time t. We set the first frame to the canonical frame. To
regularize motion and reduce overfitting, a low-dimensional parameterization is adopted by intro-
ducing B global motion bases {T(b)
t }B
b=1 shared across all dynamic Gaussians. The transformation
at time t is then expressed as a weighted sum: Tt = PB
b=1 w(b)T(b)
t , with PB
b=1 w(b) = 1, where
w(b) are per-Gaussian, learnable coefficients.
We assign each 3D Gaussian a learnable semantic feature vector f ∈Rdf to encode compact
language semantics. Directly using discrete indices Mindex is not compatible with differentiable
rendering, so we instead render these continuous features and supervise them using the quantized
semantic map. At time t, each dynamic Gaussian has transformed parameters µt and Rt. These are
used to render a 2D semantic feature map I(t)
f
through differentiable rasterization R, analogously to
the photometric rendering in Eq. 1:
I(t)
f
= R(µt, Rt, s, o, f; Ct),
(2)
where f replaces color as the per-Gaussian channel to be composited, and Ct denotes the camera
at frame t. The rendered feature map I(t)
f
∈RH×W ×df is then decoded into a semantic index
distribution via a lightweight MLP D:
ˆ
M (t) = softmax(D(I(t)
f )) ∈RH×W ×N,
(3)
where N is the number of codebook entries in B, and ˆ
M (t) represents the predicted distribution over
discrete semantic indices.
Language Embedding Loss. To supervise the learning of semantic features, we use a cross-entropy
loss between the predicted semantic index distribution ˆ
M (t) and the ground-truth index map M (t)
index
obtained during feature quantization:
Llang = CE( ˆ
M (t), M (t)
index).
(4)
This objective encourages each dynamic Gaussian to encode a compact, differentiable semantic
descriptor that aligns with language embeddings, enabling region-specific manipulation via natural
language commands.
Reconstruction Loss. The reconstruction loss Lrec consists of four components: RGB loss, depth
loss, mask loss, and tracking loss. These losses supervise the appearance, geometry, and motion of
the scene and follow the implementation in Shape-of-Motion (Wang et al., 2025).
Optimization.
We optimize the Language-Embedded Dynamic Gaussians by minimizing a
weighted combination of the reconstruction loss and the language embedding loss:
L = λrecLrec + λlangLlang,
(5)
where λrec and λlang are weights that balance the two objectives.
3.3
POINT-LEVEL LOCALIZATION OF GAUSSIANS
While Language-Embedded Dynamic Gaussians encode semantic features on each 3D Gaussian,
these features only acquire meaning after being splatted onto the 2D image plane. In isolation,
the per-Gaussian embeddings lack explicit correspondence to the semantics of real-world objects.
Consequently, directly localizing and editing Gaussians based on text remains inaccurate and coarse.
Existing methods such as OpenGaussian (Wu et al., 2024) optimize the entire scene representation
to align with semantics, which is inefficient and lacks granularity for object-specific editing. In con-
trast, we introduce a point-level localization framework that accurately localize Gaussians relevant
to the user query, by combining 2D semantic supervision and 3D feature decoding. We further re-
fine the localization through a two-stage optimization that improves both recall and precision. This
approach enables efficient and accurate text-driven editing by isolating only the relevant Gaussians
without affecting unrelated regions.
5

<!-- page 6 -->
Preprint
Given a user-provided query text q, our goal is to localize all Gaussians that correspond to the
described object at a fine-grained point-level resolution. We achieve this by leveraging the semantic
features learned in Section 3.2.
Language-guided Localization in 2D and 3D. At each frame t, we render a 2D semantic feature
map I(t)
f
∈RH×W ×df from the trained Language-Embedded Dynamic Gaussians and decode it into
CLIP space using the trained decoder and codebook B (as in Eq. 2 and Eq. 3). Then, we compute the
relevance map between the image-aligned CLIP feature and the query text feature Fq using cosine
similarity, following the approach of Kerr et al. (Kerr et al., 2023):
R(t)(p) = cos

ˆ
M (t)(p) · B, Fq

,
(6)
where ˆ
M (t) is the predicted index distribution over codebook entries, and p is a 2D pixel location.
We threshold this relevance map with a hyperparameter τ to obtain a binary 2D mask M (t)
2D indicating
regions related to the text:
M (t)
2D (p) = 1
h
R(t)(p) > τ
i
,
(7)
where 1[·] is the indicator function.
We also perform text-based localization directly in 3D. For each Gaussian gi with semantic feature
fi, we first obtain the codebook index distribution ˆmi as in Eq. 3 using the decoder, and decode it
with the codebook to get its CLIP-space embedding ˜fi = B · ˆmi ∈Rc. Then, we compute its cosine
similarity with the query:
ri = cos

˜fi, Fq

.
(8)
We define a 3D Gaussian mask L3D = {gi | ri > τ}, indicating Gaussians localized by the query
text. We denote the entire 3D localization pipeline as a function:
L3D = Localize(q, τ),
(9)
which returns a set of Gaussians matching the query.
Recall-oriented Refinement. The initial Localize(q, τ) function may miss relevant Gaussians
(false negatives). To recover these, we render the complement set ¯L3D = {gi | ri ≤τ} and project
their features to 2D. We then restrict the optimization to pixels inside the 2D relevance mask M (t)
2D :
Lrecall = CE(D(I(t)
f ), M (t)
index)
for p ∈M (t)
2D .
(10)
This loss encourages previously missed Gaussians to move closer to the query’s semantic space. We
repeat this process for n epochs to improve recall.
Precision-oriented Refinement. While the recall stage recovers most relevant Gaussians, it may
also introduce unrelated ones (false positives). To refine precision, we freeze the correctly localized
Gaussians inside the 2D mask and optimize only those outside:
Lprecision = CE(D(I(t)
f ), M (t)
index)
for p /∈M (t)
2D .
(11)
This stage suppresses irrelevant Gaussians by aligning their features back to their original semantics.
The optimization runs for m epochs.
Final Localization.
After alternating between recall and precision stages, we apply the
Localize(q, τ) function again to obtain the final point-level Gaussian mask corresponding to
the user query. This localization process is more accurate and semantically coherent, enabling fine-
grained region editing with natural language.
3.4
TEXT-DRIVEN EDITING WITH VIDEO MODEL
Recent video editing models (Jiang et al., 2025; Bian et al., 2025) offer a variety of control modali-
ties, including masks, optical flow, scribble, and grayscale frames, to guide generative edits. Among
them, mask-based control allows restricting edits to spatial regions, but lacks the ability to influence
motion and appearance details within the masked area. Conversely, optical flow and scribble offer
6

<!-- page 7 -->
Preprint
richer motion and structure cues, leading to more coherent and realistic edits, but these methods
affect the entire video and cannot restrict edits to only a specific region.
Our approach aims to combine the spatial precision of mask-based editing with the rich motion and
appearance details offered by flow- and scribble-based controls. By utilizing 3D Gaussians localized
in Section 3.3, we can selectively optimize the desired parts of the scene while preserving the overall
structure and motion. This strategy allows us to benefit from high-fidelity editing signals without
compromising spatial specificity, enabling localized, realistic edits in dynamic scenes.
Video Guidance for Gaussians Editing.
We adopt the diffusion-based video editing model
VACE (Jiang et al., 2025), which supports text-driven editing guided by auxiliary signals. To strike a
balance between control strength and generative flexibility, we primarily use optical flow and scrib-
ble, which guide edits while preserving the underlying motion and scene structure. The control
signals are extracted from the original input video. Optical flow is computed using RAFT (Teed &
Deng, 2020) and scribble are extracted using (Chan et al., 2022). These control conditions, along
with the original video and text prompt q, are used to generate an edited video Vedit, which serves as
a reference for editing the localized Gaussians.
Localized Gaussian Editing. Given the localized set L3D from Eq. 9, we freeze all Gaussians out-
side L3D and only update those within the set. This ensures that editing is confined to semantically
relevant content, while preserving the rest of the scene.
Optimization Procedure. The process described here is part of training, where we optimize only the
Gaussians within L3D. We use the video Vedit, as a reference for updating the Gaussians. Specifically,
we render a video Vrender from the dynamic 3D Gaussians and minimize the pixel-wise loss between
the rendered video and the edited video:
Ledit = ∥Vrender −Vedit∥2
2.
(12)
During this process, the gradients are only propagated to the Gaussians in L3D, ensuring that the
editing is focused on the selected regions. This optimization typically proceeds for k epochs. After
the optimization, the dynamic Gaussians can be used for rendering, producing the final edited result.
4
EXPERIMENTS
4.1
EXPERIMENTAL SETUP
Dataset. To comprehensively assess the effectiveness of our method, we organize experiments on
scenes from DAVIS (Caelles et al., 2019), DyCheck iPhone (Gao et al., 2022) datasets, Dynerf
datasets (Li et al., 2022) and some videos collected from wild. These scenes, captured by a monoc-
ular camera, encompass a rich diversity of objects, set against varying backgrounds. We utilize the
DyCheck iPhone datasets for comparison with baselines and employ the other datasets to demon-
strate the feasibility of our method on casual monocular videos.
Baselines. We compare our method against two recent approaches for text-driven dynamic scene
editing. (1) Instruct 4D-to-4D (IN4D) (Mou et al., 2024) enhances InstructPix2Pix (Brooks et al.,
2023) with anchor-aware attention and optical flow-guided propagation to achieve temporally con-
sistent video edits, treating 4D scenes as pseudo-3D volumes.
(2) CTRL-D (He et al., 2024)
performs the editing by fine-tuning InstructPix2Pix on a reference image and then optimizing de-
formable 3D Gaussians in two stages, enabling controllable and consistent 4D scenes editing.
Metrics. Following prior works (Haque et al., 2023; Zhuang et al., 2023), we assess editing quality
using CLIP Text-Image directional similarity (Gal et al., 2022), which quantifies the consistency
between the intended textual edit and the resulting visual change. For fairness, all methods generate
output videos under a shared camera trajectory. The directional similarity is computed per frame and
averaged across time to obtain the final score. To complement the automatic metric with perceptual
insights, we conduct a user study. Participants view edited dynamic scenes from all methods under a
rotating viewpoint and select the most satisfying result. We collect 58 responses and report the voting
ratio per method. Quantitative comparisons use 3 scenes covering 9 distinct editing operations.
Implementation details are included in Appendix B.2.
7

<!-- page 8 -->
Preprint
Ours
Input
Ctrl-D
Turn the cat into a fox.
Instruct 4D-to-4D
Ours
Input
Ctrl-D
Instruct 4D-to-4D
Turn the cat into a tiger.
Ours
Input
Ctrl-D
Instruct 4D-to-4D
Ours
Input
Ctrl-D
Instruct 4D-to-4D
Turn the dog into brown.
Turn the dog into a robotic dog.
Figure 3: Comparison of editing results on the iPhone dataset. Our method achieves better temporal
coherence, finer details (e.g., whiskers, eyes, specular highlights), and more accurate motion, while
avoiding artifacts in unrelated regions. Baselines tend to over-edit or introduce distortions due to
limitations in 2D diffusion-based approaches.
4.2
QUALITATIVE RESULTS
Editing 4D Scenes. We present qualitative results of text-driven editing in 4D scenes from both
DAVIS and iPhone videos, as shown in Figure 1. The figure illustrates the input monocular video
and the output edited scene rendered from multiple novel views in two different time steps. Thanks to
our point-level localization mechanism, Mono4DEditor is able to precisely identify and edit only the
target regions described by the text prompt, while preserving all unrelated content. Furthermore, by
incorporating a video editing model guided by optical flow and scribble, our method produces real-
istic and temporally coherent modifications. The resulting edits not only match the spatial semantics
but also maintain consistent motion across time and viewpoints, demonstrating the robustness of our
text-driven 4D scene editing pipeline. More results are in Appendix C.2 and Appendix C.3.
Comparison with Baseline Methods. We compare Mono4DEditor with two baselines, IN4D and
CTRL-D, on the iPhone dataset (Figure 3). We evaluate two scenes: one involving a cat and another
featuring a dog, each with two distinct editing prompts. For the cat scene, each method’s result is
shown in three time steps; for the dog scene, in two time steps. Mono4DEditor consistently out-
performs the baselines in both temporal consistency and spatial accuracy. For example, our method
preserves sharp, temporally aligned textures when editing a cat into a tiger, while baselines show
flickering and misalignments despite handcrafted temporal modules. Additionally, Mono4DEditor
produces finer details, such as the whiskers on the edited fox or the glossy highlights on the robotic
dog’s leg, while maintaining realistic appearance across time. Our motion modeling is also more
accurate, as shown in the cat scene where baselines bias the head orientation due to 2D editing
limitations. Crucially, Mono4DEditor does not introduce artifacts in unrelated regions of the scene.
In contrast, baselines often modify background elements (e.g. the ground) when editing an object,
such as editing a dog into a robotic version, leading to noticeable texture inconsistencies. Moreover,
baseline approaches can inadvertently affect the overall color scheme of the scene when modifying
the color of specific objects, such as in the ”Turn the dog brown” task.
8

<!-- page 9 -->
Preprint
Input Frame
Query Mask: Black Swan
Full
w/o Ref
Full
w/o Ref
w/o R-Ref
w/o R-Ref
w/o P-Ref
w/o P-Ref
Turn the swan 
into white.
Figure 4:
Qualitative ablation study on the effect of localization refinement, including (1) input
frame, (2) query mask based on the input prompt, (3) localized Gaussian rendering and edited result
of the Full model, (4) result of w/o Ref, (5) result of w/o R-Ref and (6) result of w/o P-Ref.
4.3
QUANTITATIVE RESULTS
Table 1:
Quantitative comparison of edit-
ing methods based on CLIP similarity and
user preference. The CLIP similarity is com-
puted as the average text-image similarities
between the prompt and all output video
frames.
Method
CLIP Sim. ↑
User Pref. ↑
IN4D
25.24
28.62%
CTRL-D
26.04
28.39%
Ours
26.23
42.99%
Table 1 presents the quantitative results in the iPhone
dataset between our method and two baselines. We
measure the CLIP similarity between the editing re-
sults and the intended text prompts.
The results
indicate that our method outperforms existing ap-
proaches in terms of alignment between the results
and the edited prompts.
Additionally, a crowd-
sourced subjective evaluation is conducted, with fur-
ther details provided in Appendix C.1.
The user
preference scores demonstrate our method achieves
superior visual perceptual quality.
4.4
ABLATION STUDY
We perform an ablation study to assess the contribution of the two localization refinement steps
in our point-level localization module: R-Ref (Recall-oriented Refinement) and P-Ref (Precision-
oriented Refinement). We compare four variants: the full pipeline (Full), and three simplified ver-
sions: (w/o Ref), where both refinement are removed; (w/o R-Ref); and (w/o P-Ref). In the w/o Ref
setting, we apply the localization function defined in Eq. 9 to directly select Gaussians.
Figure 4 shows qualitative comparisons of a DAVIS dataset dynamic scene, visualizing localized
regions and edit results for each method. Compared to variants, the full method confines edits more
accurately to the intended region and better preserves the appearance of unrelated areas. Omitting
the refinement steps (w/o Ref) leads to over-selection of irrelevant Gaussians, causing unintended
changes and degraded rendering quality. Removing only R-Ref (w/o R-Ref) leads to incomplete
Gaussian localization, for example, residual dark spots remain on the body of the swan. Conversely,
removing P-Ref (w/o P-Ref) broadens selection coverage, but introduces noisy, unrelated Gaussians.
This causes large portions of the background to be incorrectly included in the edit region, leading to
visible blank artifacts in the final result. These observations highlight the complementary roles of
R-Ref and P-Ref in ensuring accurate and clean localization.
Table 2: Ablation studies on the DAVIS
dataset.
Variant
PSNR ↑
mIoU (%) ↑
w/o R-Ref
39.43
0.71
w/o P-Ref
29.70
0.13
w/o Ref
36.86
0.43
Full (Ours)
39.55
0.71
We further quantify this behavior in Table 2, using two
metrics: (1) PSNR between the localized Gaussian ren-
dering and the reconstruction within the query mask, and
(2) mIoU between the localized Gaussians and the text
queried mask.
Our method achieves superior perfor-
mance on all metrics, demonstrating the importance of re-
finement in achieving accurate localization of Gaussians
based on query text, which benefits localized edits and
preserves background fidelity.
The ablation studies of video editing models are presented in Appendix C.4.
9

<!-- page 10 -->
Preprint
5
CONCLUSION AND LIMITATION
Mono4DEditor demonstrates that enriching dynamic 3D Gaussians with language-aligned fea-
tures enables flexible and precise text-driven editing of 4D scenes reconstructed from monocu-
lar videos. By integrating a language-embedded Gaussian representation, point-level localization,
and diffusion-based video editing, our framework achieves high-quality, temporally coherent, and
region-specific edits while preserving unedited content. This shows that semantics can be embed-
ded into dynamic scene representations, opening new opportunities for controllable and interactive
4D content creation. Despite these advances, our method is still constrained by the limitations of
monocular reconstruction: the underlying Gaussian representation we adopt only supports monocu-
lar dynamic scene reconstruction, which restricts our framework to monocular videos. In addition,
reconstruction may suffer from depth and pose errors in highly dynamic scenarios, and diffusion-
based models can introduce temporal drift or motion artifacts. Future work will focus on extending
our approach to multi-view 4D representations, improving monocular reconstruction under challeng-
ing motion and enhancing generative models with stronger motion fidelity and regional control.
REFERENCES
Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu.
Videopainter: Any-length video inpainting and editing with plug-and-play context control. arXiv
preprint arXiv:2503.05639, 2025.
Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image
editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pp. 18392–18402, 2023.
Sergi Caelles, Jordi Pont-Tuset, Federico Perazzi, Alberto Montes, Kevis-Kokitsi Maninis, and
Luc Van Gool.
The 2019 davis challenge on vos: Unsupervised multi-object segmentation.
arXiv:1905.00737, 2019.
Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image
diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp.
23206–23217, 2023.
Caroline Chan, Fr´edo Durand, and Phillip Isola. Learning to generate line drawings that convey
geometry and semantics. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 7915–7925, 2022.
Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei
Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with
gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pp. 21476–21485, 2024.
Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 4d-
rotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes. In ACM
SIGGRAPH 2024 Conference Papers, pp. 1–11, 2024.
Zhiwen Fan, Kairun Wen, Wenyan Cong, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris
Ivanovic, Marco Pavone, Georgios Pavlakos, Zhangyang Wang, and Yue Wang. Instantsplat:
Sparse-view gaussian splatting in seconds, 2024.
Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias
Nießner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH
Asia 2022 Conference Papers, pp. 1–9, 2022.
Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo
Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12479–12488, 2023.
Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-
Or. Stylegan-nada: Clip-guided domain adaptation of image generators. ACM Transactions on
Graphics (TOG), 41(4):1–13, 2022.
10

<!-- page 11 -->
Preprint
Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dy-
namic view synthesis: A reality check. In NeurIPS, 2022.
Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features
for consistent video editing. arXiv preprint arXiv:2307.10373, 2023.
Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa.
Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, pp. 19740–19750, 2023.
Kai He, Chin-Hsuan Wu, and Igor Gilitschenski. Ctrl-d: Controllable dynamic 3d scene editing
with personalized 2d diffusion. arXiv preprint arXiv:2412.01792, 2024.
Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-
gs: Sparse-controlled gaussian splatting for editable dynamic scenes.
In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pp. 4220–4230, 2024.
Sicheng Jiang. Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature
Fields. University of California, Los Angeles, 2023.
Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one
video creation and editing. arXiv preprint arXiv:2503.07598, 2025.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splat-
ting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Lan-
guage embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pp. 19729–19739, 2023.
Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang
Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models
are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pp. 15954–15964, 2023.
Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via
feature field distillation. Advances in Neural Information Processing Systems, 35:23311–23330,
2022.
Joohyun Kwon, Hanbyel Cho, and Junmo Kim. Efficient dynamic scene editing via 4d gaussian-
based static-dynamic separation. In Proceedings of the Computer Vision and Pattern Recognition
Conference (CVPR), pp. 26855–26865, June 2025.
Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic
gaussian fusion from casual videos via 4d motion scaffolds. arXiv preprint arXiv:2405.17421,
2024.
Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim,
Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video
synthesis from multi-view video. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pp. 5521–5531, 2022.
Wanhua Li, Renping Zhou, Jiawei Zhou, Yingwei Song, Johannes Herter, Minghan Qin, Gao Huang,
and Hanspeter Pfister. 4d langsplat: 4d language gaussian splatting via multimodal large language
models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, 2025.
Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo
Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast, and robust struc-
ture and motion from casual dynamic videos. arXiv preprint arXiv:2412.04463, 2024.
Feng-Lin Liu, Hongbo Fu, Yu-Kun Lai, and Lin Gao. Sketchdream: Sketch-based text-to-3d gener-
ation and editing. ACM Transactions on Graphics (TOG), 43(4):1–13, 2024a.
11

<!-- page 12 -->
Preprint
Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with
cross-attention control. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 8599–8608, 2024b.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications
of the ACM, 65(1):99–106, 2021.
Linzhan Mou, Jun-Kun Chen, and Yu-Xiong Wang. Instruct 4d-to-4d: Editing 4d scenes as pseudo-
3d scenes using 2d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 20176–20185, 2024.
Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman,
Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higher-dimensional representation for
topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021.
Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and
Fisher Yu.
Unidepth: Universal monocular metric depth estimation.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10106–10116, 2024.
Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural
radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 10318–10327, 2021.
Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng
Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the
IEEE/CVF International Conference on Computer Vision, pp. 15932–15942, 2023.
Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d lan-
guage gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 20051–20060, 2024.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning, pp.
8748–8763. PmLR, 2021.
Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham
Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images
and videos. arXiv preprint arXiv:2408.00714, 2024.
Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu, Hongwen Zhang, and Yebin Liu. Ten-
sor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
16632–16642, 2023.
Ruizhi Shao, Jingxiang Sun, Cheng Peng, Zerong Zheng, Boyao Zhou, Hongwen Zhang, and Yebin
Liu. Control4d: Efficient 4d portrait editing with text. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pp. 4556–4567, 2024.
Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaus-
sians for open-vocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp. 5333–5343, 2024.
Chaehun Shin, Heeseung Kim, Che Hyun Lee, Sang-gil Lee, and Sungroh Yoon. Edit-a-video:
Single video editing with object-aware consistency. In Asian Conference on Machine Learning,
pp. 1215–1230. PMLR, 2024.
Colton Stearns, Adam Harley, Mikaela Uy, Florian Dubost, Federico Tombari, Gordon Wetzstein,
and Leonidas Guibas. Dynamic gaussian marbles for novel view synthesis of casual monocular
videos. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.
12

<!-- page 13 -->
Preprint
Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer
Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings,
Part II 16, pp. 402–419. Springer, 2020.
Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras.
Advances in neural information processing systems, 34:16558–16569, 2021.
Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in
neural information processing systems, 30, 2017.
Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo
Kanazawa. Shape of motion: 4d reconstruction from a single video. In International Confer-
ence on Computer Vision (ICCV), 2025.
Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu,
Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion
models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pp. 7623–7633, 2023.
Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao,
Haocheng Feng, Errui Ding, Jingdong Wang, et al.
Opengaussian: Towards point-level 3d
gaussian-based open vocabulary understanding. arXiv preprint arXiv:2406.02058, 2024.
Teng Xu, Jiamin Chen, Peng Chen, Youjia Zhang, Junqing Yu, and Wei Yang. Tiger: Text-instructed
3d gaussian retrieval and coherent editing. arXiv preprint arXiv:2405.14455, 2024.
Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided
video-to-video translation. In SIGGRAPH Asia 2023 Conference Papers, pp. 1–11, 2023.
Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pp. 20331–20341, 2024.
Qihang Zhang, Yinghao Xu, Chaoyang Wang, Hsin-Ying Lee, Gordon Wetzstein, Bolei Zhou, and
Ceyuan Yang. 3ditscene: Editing any scene via language-guided disentangled gaussian splatting.
arXiv preprint arXiv:2405.18424, 2024.
Shijie Zhou, Hui Ren, Yijia Weng, Shuwang Zhang, Zhen Wang, Dejia Xu, Zhiwen Fan, Suya You,
Zhangyang Wang, Leonidas Guibas, and Kadambi. Feature4x: Bridging any monocular video to
4d agentic ai with versatile gaussian feature fields. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2025.
Jingyu Zhuang, Chen Wang, Liang Lin, Lingjie Liu, and Guanbin Li. Dreameditor: Text-driven 3d
scene editing with neural fields. In SIGGRAPH Asia 2023 Conference Papers, pp. 1–10, 2023.
Jingyu Zhuang, Di Kang, Yan-Pei Cao, Guanbin Li, Liang Lin, and Ying Shan. Tip-editor: An ac-
curate 3d editor following both text-prompts and image-prompts. ACM Transactions on Graphics
(TOG), 43(4):1–12, 2024.
13

<!-- page 14 -->
Preprint
A
MORE RELATED WORKS
A.1
TEXT-DRIVEN VIDEO EDITING
Video editing models are the foundation of our Mono4DEditor. Several generative video editing
methods utilize training-free adaptations of pre-trained Text-to-Image models, which typically con-
vert spatial self-attention mechanisms into temporal-aware cross-frame attention to enforce consis-
tency (Ceylan et al., 2023; Khachatryan et al., 2023; Qi et al., 2023; Geyer et al., 2023; Yang et al.,
2023). Another strategy is per-video fine-tuning, where a image generation model is optimized on
a specific input video to learn its unique appearance and motion dynamics (Liu et al., 2024b; Wu
et al., 2023; Shin et al., 2024). More advanced methods like VACE (Jiang et al., 2025), by handling
various multimodal inputs through a unified framework, achieve versatile compositional editing.
In our framework, we leverage the compositional and multimodal control capabilities of VACE to
guide localized Gaussian updates, enabling spatially precise and temporally coherent text-driven 4D
editing.
B
TECHNICAL DETAILS
B.1
QUANTIZATION OF CLIP FEATURES
Following the approach in LEGaussians (Shi et al., 2024), we quantize the extracted CLIP features
Fclip ∈Rt×h×w×c using a learnable codebook B ∈RN×c, where N is the number of codebook
entries. For each frame i and pixel location p, we compute the cosine similarity between the CLIP
feature vector F (i)
clip(p) ∈Rc and all entries in the codebook B. The feature is then assigned the index
of the most similar codebook entry:
M (i)
index(p) = arg
max
j∈{1,...,N} cos

F (i)
clip(p), B(j)
,
(13)
Here, B(j) ∈Rc denotes the j-th codebook vector, and M (i)
index(p) ∈{1, . . . , N} is the index of the
closest codeword for pixel p in frame i based on cosine similarity.
To optimize the codebook B, we use a cosine similarity loss function, which encourages the code-
book entries to move closer to the CLIP feature vectors:
Lquant =
t
X
i=1
h×w
X
p=1

1 −cos

F (i)
clip(p), BM (i)
index(p)
.
(14)
This loss, inspired by the work of (Van Den Oord et al., 2017), is backpropagated to update the
codebook B, allowing it to better represent the distribution of the extracted CLIP features. After
training, we obtain a learned codebook and the corresponding index map Mindex ∈Rt×h×w×1,
which stores the closest codeword index for each pixel in each frame.
These index maps and the video-level semantic codebook B are used to embed semantics into the
dynamic Gaussian field.
B.2
IMPLEMENTATION DETAILS
We build upon Shape-of-Motion (Wang et al., 2025), originally built for monocular video, and ex-
tend it into a language-embedded dynamic 3D representation. Given a monocular video, we extract
camera poses via MegaSaM (Li et al., 2024) or Droid-slam (Teed & Deng, 2021) and obtain dy-
namic masks, depth, and 2D tracks using off-the-shelf methods (Ravi et al., 2024; Li et al., 2024;
Piccinelli et al., 2024; Teed & Deng, 2020). Following (Fan et al., 2024), we learn a camera pose
correction term to refine the predicted poses, leading to improved dynamic scene reconstruction. Se-
mantic features are obtained from SAM2 (Ravi et al., 2024) and dense CLIP (Radford et al., 2021),
then quantized with a codebook of size N=128 (Shi et al., 2024). In the phase of dynamic Gaussian
reconstruction, we follow the parameter configurations specified in Shape-of-Motion (Wang et al.,
2025). We utilize the Adam Optimizer for the optimization process. Concretely, we carry out 1000
optimization iterations for the initial fitting procedure and 500 epochs for the joint optimization pro-
cedure. For the initialization of Gaussians, 40,000 dynamic Gaussians are initialized for the dynamic
14

<!-- page 15 -->
Preprint
segment of the scene, while 100,000 static Gaussians are initialized for the static segment. Further-
more, we implement the identical adaptive Gaussian control for both dynamic and static Gaussians,
in accordance with the method described in 3DGS (Kerbl et al., 2023). Each 3D Gaussian is aug-
mented with a learnable semantic vector (df=8). Training optimizes a weighted sum of Lrec, Llang
(both weighted by 1), and geometry terms. Language-based selection uses similarity in the CLIP
space, combining 2D and 3D signals with a threshold τ=0.95. A two-stage refinement runs for
n=50 epochs to improve recall, and additional denoising is performed for m=10 epochs. We train
for 500 epochs for language-embedded reconstruction and another k=500 epochs for editing on a
single NVIDIA A6000.
C
MORE RESULTS
This section provides additional experimental results of our method. Appendix C.1 presents a quan-
titative comparison of results between our method and the baselines; Appendix C.2 demonstrates
more editing outcomes of our approach across different datasets; Appendix C.3 provides results of
semantic querying and Gaussian point localization using text prompt; and Appendix C.4 reports the
results of ablation experiments conducted on the video editing model.
C.1
DETAILED QUANTITATIVE RESULTS
To comprehensively evaluate the performance of our proposed method, we conducted a crowd-
sourced subjective evaluation against the baseline approaches IN4D (Mou et al., 2024) and CTRL-
D (He et al., 2024). The study was designed to gather qualitative feedback across three distinct
dimensions of video editing quality: Naturalness, which measures the realism of the edited re-
sults considering both spatial accuracy and temporal consistency; Prompt Fidelity, which assesses
how well the edits match the given text instruction; and Background Preservation, which evaluates
whether irrelevant regions remain unchanged.
For the evaluation, we curated nine different video editing scenarios. In each scenario, participants
were presented with the results generated by the three methods (IN4D, CTRL-D, and Ours) in a
randomized order. We then asked the participants to rank these three results from best (rank 1) to
worst (rank 3) for each of the three evaluation dimensions. A total of 58 participants completed
the survey for the three dimensions. Table 3 presents a detailed breakdown of the results, showing
the percentage of first-place votes each method received for each criterion. This quantitative data
supplements the main findings in our paper, offering a more granular view of user preferences.
Results show that Mono4DEditor consistently outperforms the baselines across all three evaluation
dimensions.
Table 3: Results of the crowd-sourced subjective evaluation. The values indicate the percentage of
first-place votes each method received across three evaluation dimensions. A total of 58 participants
provided valid rankings. The best result for each dimension is highlighted in bold.
Evaluation Dimension
IN4D
CTRL-D
Ours
Naturalness
31.72%
28.62%
39.66%
Prompt Fidelity
27.24%
21.03%
51.72%
Background Preservation 26.90%
35.52%
37.59%
C.2
MORE EDITING RESULTS
To better demonstrate the effectiveness of our editing capabilities, we have conducted additional
editing experiments on the Davis, DyNeRF datasets, and some videos collected from wild (Figure 5,
Figure 6, Figure 7). In Figure 5, we present the input videos from the DAVIS and DyNeRF datasets,
along with the rendered images generated from the text-driven editing results in different timestamps
and viewpoints. In Figure 6, we show the input image, the control signals of the video editing
model, and the output results of the video editing model. Additionally, we render the relevance maps
derived from text queries and the edited images from edited 4D scenes at different viewing angles
and time points. In Figure 7, to demonstrate the multistage editing capability, we perform text-driven
15

<!-- page 16 -->
Preprint
View 1
View 2
View 1
View 2
Timestep 1
Timestep 2
Turn the bear into a bear in Iron Man armor.
View 1
View 2
View 1
View 2
Timestep 1
Timestep 2
Drape the camel in a vibrant Indian red fabric with intricate gold patterns.
Input Monocular Videos
View 1
View 2
View 1
View 2
Timestep 1
Timestep 2
Turn the car into a Land Rover.
View 1
View 2
View 1
View 2
Timestep 1
Timestep 2
Turn the hat into a yellow Hard Hat.
Edited 4D Scenes
Figure 5: Editing results of Mono4DEditor on DAVIS and DyNerf datasets. Each example shows
monocular video input and the text-driven edited output at two different time steps, rendered from
two novel views. Our method achieves accurate localization, realistic appearance changes, and
preserves spatial and temporal consistency across views.
editing on different parts of the scene at each stage. We present the input video, along with rendered
relevance maps corresponding to text queries and edited results from different viewpoints at various
timestamps for each stage.
These results indicate that our method effectively localizes the regions to be edited according to text
instructions and produces high-quality edits with strong spatiotemporal consistency. Moreover, our
approach also prevents the video model from inadvertently modifying the background, ensuring that
modifications are restricted to the intended areas.
C.3
SEMANTIC QUERYING AND LOCALIZATION VISUALIZATION
To further analyze the semantic precision of our approach, we visualize the 2D query maps and
point-level Gaussian localizations in Figure 8.
We render 2D relevance maps by querying the
language-embedded Gaussian features with text prompts in both the original and novel views at
multiple time steps, and visualize the localized Gaussians.
These results demonstrate that our Language-Embedded Dynamic Gaussians capture rich semantic
information and enable fine-grained localization of both static and dynamic scene elements. For
instance, the method successfully isolates moving subjects such as an animal while excluding back-
ground content, ensuring that only intended regions are subject to editing.
C.4
ABLATION STUDY OF VIDEO EDITING MODEL
Figure 9 presents a comparison of editing results from different video editing models on the Dy-
Check iPhone dataset. Specifically, the result edited by Tune-A-Video (Wu et al., 2023) fails to
preserve the original motion features; the result edited by Text2Video-Zero (Khachatryan et al.,
2023) shows unnatural visual effects; in contrast, the result edited by VACE not only well retains
the original features but also achieves high-quality editing effects.
16

<!-- page 17 -->
Preprint
Input video
Control Signal
Edited Video
Edited Scene
Query Masks
View 1
View 1
View 2
View 2
View 1
View 1
View 2
View 2
Timestep 1
Timestep 2
View 1
View 2
View 1
View 2
View 1
View 1
View 2
View 2
Timestep 1
Timestep 2
Turn the seal look like Van Gogh's "The Starry Night”.
Turn the tiger to a leopard.
Figure 6: Editing results of Mono4DEditor on in-the-wild videos. Each example shows monocular
video input, control signal, edited video and query masks and the text-driven edited output at two
different time steps, rendered from two novel views. Our method achieves accurate localization,
realistic appearance changes and preserves the background in its original state.
Across different video editing models, our method consistently ensures that the regions unrelated
to the edit remain unaffected, thereby preserving the original scene context. Among these models,
VACE achieves the best overall performance: it not only provides high-quality editing results but
also maintains strong temporal consistency in the edited regions, ensuring smooth and coherent
transitions across frames.
Notably, VACE alone fails to keep semantically irrelevant regions unchanged—only when inte-
grated with our method can this limitation be resolved. For other less-performing models that may
introduce flaws (e.g., temporal inconsistency, shape distortion) in edited regions, our method still
effectively prevents semantically irrelevant areas from unintended alterations. This highlights both
VACE’s strengths in high-fidelity, temporally coherent edits and the indispensability of our method.
Our method not only complements VACE to protect non-edited regions but also generalizes across
video editing models. This further validates the advantage of our framework for stable, natural 4D
scene edits.
17

<!-- page 18 -->
Preprint
Input Video
Edited 4D Scene
Query Masks
View 1
View 1
View 2
View 2
View 1
View 1
View 2
View 2
Timestep 1
Timestep 2
View 1
View 2
View 1
View 2
View 1
View 1
View 2
View 2
Timestep 1
Timestep 2
Step 1: Turn the bear into a panda. 
Step 2: Turn leaves into autumn leaves. 
Figure 7:
Multi-stage editing results of Mono4DEditor on DAVIS datasets. Mono4DEditor edits
the scene in two stage: turn the bear into a panda in Step 1 and turn leaves into autumn leaves in
Step 2. Each step shows query masks and the text-driven edited output at two different time steps,
rendered from two novel views. Our method enables stage-by-stage editing of scenes while ensuring
high-quality editing at each stage.
Timestep 1
Timestep 2
Timestep 1
Timestep 2
View 1
View 2
View 1
View 2
Query Text: Bear
View 1
View 2
View 1
View 2
Query Text: Leaf
Timestep 1
Timestep 2
View 1
View 2
View 1
View 2
Query Text: Swan
Timestep 1
Timestep 2
View 1
View 2
View 1
View 2
Query Text: Camel
Rendering of Localized Gaussians
Rendering of Localized Gaussians
Rendering of Localized Gaussians
Rendering of Localized Gaussians
Timestep 1
Timestep 2
View 1
View 2
View 1
View 2
Query Text: Wall
Rendering of Localized Gaussians
Figure 8:
Visualization of 2D semantic queries and point-level localization. For each scene, we
show the 2D relevance map generated from the language-embedded Gaussian features from the
original and novel views at multiple time steps, as well as the rendering results of localized Gaus-
sians. Our method accurately isolates both static and dynamic regions relevant to the input query.
18

<!-- page 19 -->
Preprint
Timestep 1
Timestep 2
Timestep 1
Timestep 2
Timestep 1
Timestep 2
View 1
View 1
View 2
View 2
View 1
View 2
View 1
View 2
View 1
View 2
View 1
View 2
Timestep 1
Timestep 1
Timestep 2
Timestep 2
Timestep 2
Timestep 1
Turn the cat into a fox.
VACE
Text2Video-Zero
Turn-A-Video
Edited Videos
Edited 4D Scenes
Figure 9:
Qualitative ablation study on the effect of different video editing methods, includ-
ing Edited Videos results and Edited 4D Scenes results using (1) VACE (Jiang et al., 2025), (2)
Text2Video-Zero (Khachatryan et al., 2023) and (3) Tune-A-Video (Wu et al., 2023). VACE shows
natural and consistent effects across different timesteps and views, maintaining temporal coherence
in the edited region. Text2Video-Zero causes unnatural facial features and fur textures, while Tune-
A-Video fails to preserve the original shape and motion features. Importantly, our method ensures
that irrelevant areas of the 4D scenes remain unaffected, while VACE further excels by providing
superior temporal and spatial consistency in the edited areas.
19
