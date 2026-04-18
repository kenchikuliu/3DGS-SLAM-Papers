<!-- page 1 -->
PercHead: Perceptual Head Model
for Single-Image 3D Head Reconstruction & Editing
Antonio Oroz
Matthias Nießner
Tobias Kirschstein
Technical University of Munich
{antonio.oroz,niessner,tobias.kirschstein}@tum.de
Figure 1. PercHead. Our method reconstructs high-fidelity 3D heads from single input images, maintaining consistency across arbitrary
viewpoints. Beyond reconstruction, our fine-tuned editing model enables realistic 3D head generation from a segmentation map as geo-
metric input, with style controlled via a reference image or text prompt.
Abstract
We present PercHead, a method for single-image 3D
head reconstruction and semantic 3D editing — two
tasks that are inherently challenging due to severe view
occlusions, weak perceptual supervision, and the ambi-
guity of editing in 3D space. We develop a unified base
model for reconstructing view-consistent 3D heads from
a single input image. The model employs a dual-branch
encoder followed by a ViT-based decoder that lifts 2D
features into 3D space through iterative cross-attention.
Rendering is performed using Gaussian Splatting. At the
heart of our approach is a novel perceptual supervision
strategy based on DINOv2 [41] and SAM2.1 [48], which
provides rich, generalized signals for both geometric and
appearance fidelity.
Our model achieves state-of-the-art
performance in novel-view synthesis and, furthermore,
exhibits exceptional robustness to extreme viewing angles
compared to established baselines. Furthermore, this base
model can be seamlessly extended for semantic 3D editing
by swapping the encoder and fine-tuning the network. In
this variant, we disentangle geometry and style through
two distinct input modalities:
a segmentation map to
control geometry and either a text prompt or a reference
image to specify appearance. We highlight the intuitive and
powerful 3D editing capabilities of our model through a
lightweight, interactive GUI, where users can effortlessly
sculpt geometry by drawing segmentation maps and stylize
appearance via natural language or image prompts.
Project Website: https://antoniooroz.github.io/PercHead/
1. Introduction
Accurate reconstruction of 3D head geometry from a sin-
gle image, while maintaining robust consistency across ar-
bitrary input and target viewpoints, is a crucial yet unsolved
challenge. Success in this task would enable highly realistic
arXiv:2511.02777v1  [cs.CV]  4 Nov 2025

<!-- page 2 -->
and immersive applications, including virtual conferencing,
gaming, and personalized retail experiences, as well as open
the door to advanced capabilities such as camera control
over portraits, 3D virtual try-on, and editing that disentan-
gles geometry from style.
Faithful 3D head reconstruction from a single image is
an inherently under-constrained problem, as many plau-
sible 3D shapes can explain the same 2D observation.
While learning strong priors from large datasets is critical
to resolve this ambiguity, high-quality multi-view datasets
such as NeRSemble [29] and Ava-256 [40] remain lim-
ited in size, and large-scale datasets like FFHQ [26] and
VFHQ [59] provide only single-view 2D supervision or lim-
ited angle-variety. As a consequence, existing methods tend
to perform well for near-frontal views or viewpoints close
to the input, but deteriorate in identity fidelity or geomet-
ric realism under large viewpoint deviations. Furthermore,
humans are exceptionally sensitive to even small errors in
facial structure and appearance, demanding extremely high
visual fidelity from reconstructed 3D heads. Conventional
supervision strategies based on reconstruction or perceptual
losses often fall short of achieving the required realism, es-
pecially for high-frequency or occluded regions. Finally,
downstream editing tasks that aim to disentangle style (e.g.,
hair, skin tone) from geometry (e.g., facial structure) are
inherently ambiguous, as multiple plausible outputs can ex-
plain the same input.
Existing methods address the challenges of 3D head re-
construction through various forms of prior constraints and
supervision. Early approaches such as ROME [28] relied
on strong geometric constraints via mesh-based 3D Mor-
phable Models (3DMMs), limiting realism and expressive-
ness. GAN-based methods [1, 7] leveraged large-scale 2D
datasets and adversarial losses to synthesize realistic out-
puts, but often struggle with identity preservation and ro-
bustness under extreme viewpoints. GAGAvatar [8] intro-
duced a dual-lifting strategy trained on multi-frame video
data, which improves reconstruction quality near the input
view but fails to generalize well to more diverse angles.
The Large Avatar Model (LAM) [24] employs multi-frame
video supervision within a transformer-based architecture,
achieving strong frontal reconstructions but still exhibiting
degradation for views far from the input. LGM [54] em-
ploys a multi-view diffusion prior, but still falls short of the
realism required for high-quality human head synthesis.
To overcome these limitations, we propose a novel ar-
chitecture combining real multi-view data from Nersem-
ble [29], artificial multi-view head data from Cafca [5],
and diverse single-view images from FFHQ [26] to achieve
the best trade-off between diversity and strong 3D consis-
tency. Our approach employs a dual-branch encoder inte-
grating intermediate DINOv2 features [24, 41] with a task-
specific Vision Transformer (ViT) [12]. Our view-agnostic
ViT-Decoder exclusively cross-attends to 2D features, lift-
ing their information into 3D regions. Critically, to train
our 3D deocder pipeline, our method replaces conventional
perceptual and reconstruction losses, such as LPIPS, SSIM,
or L1, with the supervision of powerful foundational mod-
els: DINOv2 and SAM2.1 [48]. Surprisingly, this departure
from traditional objectives leads to more robust and gen-
eralized training signals, ultimately enhancing both visual
fidelity and 3D consistency.
Our method significantly surpasses state-of-the-art ap-
proaches, such as PanoHead, GAGAvatar, LGM and LAM,
particularly in reconstruction tasks involving extreme view-
points.
Furthermore, our generalized 3D reconstruction
model serves as a versatile backbone that can be extended to
tasks such as 3D editing. By replacing the encoder and fine-
tuning the model with segmentation maps [64] and CLIP
embeddings [47] as inputs, we enable advanced editing
capabilities that disentangle geometric control from style
modulation.
In summary, our contributions include:
• A novel architecture achieving superior single-image 3D
head reconstruction, especially at challenging viewing
angles.
• A perceptual loss formulation based on generalized vision
models DINOv2 and SAM2.1, significantly improving vi-
sual quality.
• Advanced 3D face editing capabilities are enabled by
fine-tuning our strong, generalized base model on FARL
segmentation maps and CLIP embeddings, demonstrat-
ing both its versatility in handling diverse inputs and its
transferability to novel downstream tasks.
2. Related Work
2.1. Single-Image 3D Face Reconstruction
Early single-image 3D reconstruction methods used mesh-
based models [3, 13, 15, 16, 28, 32, 33, 38, 55], offering
efficient geometry but limited detail and flexibility - such as
hair. GAN-based refinements [15, 16] improve realism but
still inherit the mesh constraints.
Early NeRF-based GANs [6, 19, 42] achieved high-
quality 3D head synthesis but were slow. EG3D [7] in-
troduced tri-plane representations to improve efficiency,
enabling
faster
rendering
with
great
visual
quality.
PanoHead [1] extended this to 360° head generation.
Nonetheless, these GAN-based methods require computa-
tionally expensive latent inversion [49] for 3D head recon-
struction.
Encoder-based variants [4] improve inference
speed but typically reduce reconstruction fidelity.
Most
GAN-based methods are also trained on 2D data and lack
strong multi-view supervision, limiting 3D consistency.
Portrait4D [10] and Portrait4Dv2 [11] address this by in-
troducing multi-view supervision, resulting in more consis-

<!-- page 3 -->
tent geometry. Nonetheless, NeRFs remain computation-
ally heavy and can still suffer from view-dependent vari-
ability.
With the emergence of Gaussian Splatting [27] as a
leading representation for fast, photorealistic 3D rendering,
several recent works have adopted it in the 3D head do-
main [18, 30, 37, 46]. However, these methods are either
unconditional generative models, subject-specific optimiza-
tions, or require multi-view or video input. GAGAvatar [8]
and LAM [24] apply Gaussian Splatting to single-image 3D
head reconstructions. While they are highly efficient and
produce impressive results for near-input view angles, both
deteriorate in performance for more diverse viewpoints.
Building on the popularity of diffusion models [21, 22,
25, 45, 50], numerous diffusion-based head methods [17,
44, 56, 62] have emerged; however, they are typically
image-based, require per-subject optimization, or fall short
of photorealistic 3D head synthesis. LGM [54] introduces
a multi-view diffusion prior to improve 3D consistency, but
still falls short in producing the photorealism and geometric
precision required for high-quality human head synthesis.
Many of the aforementioned methods either use ad-
versarial losses, which are costly to compute and noto-
riously difficult to train, or employ conventional percep-
tual losses such as LPIPS [63], which often underperforms
significantly compared to adversarial supervision. A few
works explore DINOv2-based losses [39, 51, 61], or more
generally rely on foundation model features for percep-
tual similarity [14], but typically neglect intermediate rep-
resentations or focus on 2D tasks.
Our approach lever-
ages multi-layer DINOv2 [41] features in combination with
SAM2.1 [48] image encodings to improve 3D head consis-
tency and sharpness.
2.2. 3D Face Editing
Several methods perform 3D editing in the latent space of
NeRF-based GANs [31, 34, 53, 58], but they do not support
natural language-based edits and instead rely on style ref-
erence images or manually specified attributes. There have
also been text-guided 3D editing models [20, 35, 43] in this
domain, but, like the others, they inherit the core limitations
of NeRF-based pipelines — like view-dependent artifacts
and limited 3D consistency.
ClipFace [2], on the other hand, enables text-guided edit-
ing of 3D Morphable Models (3DMMs), but in doing so
inherits the limited geometric expressiveness and visual fi-
delity of mesh-based representations, resulting in lower re-
alism compared to modern neural rendering approaches.
LAM [24] enables text- and image-based 3D avatar gen-
eration using diffusion models, but the editing process oc-
curs entirely in 2D and is later lifted into 3D space. This
design incurs substantial computational cost and limits di-
rect control over 3D structure.
In contrast, our method uses FARL segmentation maps
and CLIP features as 2D encodings that disentangle geom-
etry and style, and directly attends to them in 3D space to
synthesize the full 3D head efficiently and with explicit con-
trol.
3. Method
Our framework consists of two closely related models: (1)
a base single-image-to-3D lifting model, which reconstructs
a canonical 3D head representation from a single input im-
age, and (2) a fine-tuned 3D editing model, which adapts the
base model for a conditional editing task that disentangles
geometry and style.
As illustrated in Figure 2, apart from the modified en-
coding process in the editing model, both variants follow
the exact same pipeline: dense 2D features are extracted
from the input image, lifted into a canonical 3D representa-
tion via a transformer decoder, and converted into a set of
renderable 3D Gaussians. The editing model alters only the
encoding stage to incorporate conditional inputs, while the
remainder of the pipeline remains unchanged.
In the following, we first describe the base lifting model,
then introduce the encoding modifications for the editing
model in Section 3.5.
3.1. Dual-Branch Lift Encodings
We pre-process all input images using GAGAvatar’s [8]
tracking module, which also removes the background. We
then extract dense 2D features with a dual-branch ar-
chitecture:
a frozen DINOv2 [41] backbone capturing
rich semantic and low-level features at intermediate layers
(9, 19, 29, 39), inspired by LAM [24], and a lightweight,
task-adaptive ViT [12] encoder inspired by Masked Autoen-
coders (MAE) [23]. This design enables our model to lever-
age DINOv2 for extracting highly generalized, semantically
rich representations, while the ViT encoder can adapt to the
specific task without requiring fine-tuning of the DINOv2
backbone.
The encoder thus complements DINOv2 by
adding potentially missing fine-grained or context-specific
information that DINOv2 alone may not capture.
Both branches produce foreground-only patches P, i.e.,
patches containing pixels not discarded by GAGAvatar’s
tracking.
This reduces the number of processed patches
by up to 30%, significantly lowering computational cost.
We match the receptive fields and patch sizes of the two
branches, ensuring spatial alignment between their outputs.
This alignment allows us to concatenate ([·, ·]) the resulting
features patchwise and project them into the decoder em-
bedding space via an MLP:
F2D = MLP

F i
Enc, F i
Dino
|P |
i=1

(1)

<!-- page 4 -->
Figure 2. Overview of Our Method. Our framework supports 3D Reconstruction from a single image and 3D Editing from a segmentation
map and style input. Both tasks share a 3D ViT decoder that lifts 2D features via iterative cross-attention, differing only in the encoder.
The reconstruction model uses a dual-branch encoder with DINOv2 and a task-specific ViT; the editing model uses a segmentation ViT
and injects a global CLIP style token. Outputs are rendered via Gaussian Splatting and refined with a 2D CNN, with supervision from
DINOv2 and SAM2.1.
3.2. 3D ViT Decoder
Our modified, MAE-based [23], ViT decoder begins from
a base 3D representation derived from a fixed, upsampled
(65k vertices) FLAME template [36]. These vertices are
grouped into 3D patches, each containing 16 vertices. This
structured initialization provides a stable canonical shape
that the network can refine.
At each decoding layer i, the current 3D features cross-
attend to the most relevant 2D patch features F2D, inte-
grating image-conditioned information, and are then trans-
formed via an MLP:
F i
3D = F i−1
3D
+ MLPi  F i−1
3D
+ ATTNi  F i−1
3D , F2D

(2)
Skip connections between iterations allow information
to flow through all stages, making the decoder interpretable
as an iterative refinement mechanism. This process is vi-
sualized in Figure 2 - exact details on the visualization are
provided in the appendix.
Notably, to reduce computational cost, we omit any self-
attention among 3D patches. Each patch — representing a
group of 16 initial FLAME vertices and, later, 16 Gaussians
— acts independently. Global coherence emerges entirely
from the shared 2D feature context.
Unlike pixel-space
transformers where positions are fixed, our 3D patches can
move in space, adapting their positions to best represent lo-
cal geometry. This flexibility mitigates patch-boundary arti-
facts common in 2D pipelines without self-attention, while
preserving accurate structure and appearance.
3.3. Gaussian Splatting
Similar to the Large Gaussian Reconstruction Model
(GRM) [60], we apply PixelShuffle [52] to upsample each
of the 4,096 latent 3D patches into 16 Gaussians, resulting
in ∼65k in total. For each Gaussian, we predict position
offsets, scale, rotation, opacity, and color via independent
linear layers.
The predicted Gaussians are rendered with a differen-
tiable Gaussian Rasterizer [27], followed by a shallow CNN
to enhance sharpness and reduce artifacts.
3.4. Perceptual Loss Module
Our perceptual supervision module combines features from
two complementary sources: late-stage features from the
small variant of DINOv2 (layers 8 and 11) [41] and image
encoder features from SAM 2.1 [48]. This pairing leverages
the strong generalization capabilities of DINOv2 while inte-
grating SAM’s segmentation-oriented visual understanding.

<!-- page 5 -->
Novel
Views
Ava-256
NeRSemble
PSNR ↑
SSIM ↑
LPIPS ↓
DS ↓
ArcFace ↓
PSNR ↑
SSIM ↑
LPIPS ↓
DS ↓
ArcFace ↓
LGM
10.47
0.5912
0.5274
0.1849
0.7224
11.37
0.6619
0.5002
0.1889
0.7007
PanoHead
14.75
0.6865
0.3135
0.1060
0.3580
15.40
0.7495
0.2613
0.1062
0.3376
GAGAvatar
15.87
0.7428
0.2739
0.1143
0.3481
16.88
0.7979
0.2169
0.1066
0.2883
LAM
13.82
0.6947
0.3529
0.1430
0.4064
14.54
0.7577
0.3033
0.1340
0.3529
Ours
16.08
0.7003
0.2666
0.0927
0.2935
18.04
0.7755
0.1854
0.0758
0.2559
Table 1. Novel View Reconstruction Performance on Ava-256 and NeRSemble Datasets.
Extreme
Views
Ava-256
NeRSemble
PSNR ↑
SSIM ↑
LPIPS ↓
DS ↓
ArcFace ↓
PSNR ↑
SSIM ↑
LPIPS ↓
DS ↓
ArcFace ↓
LGM
9.88
0.5859
0.5651
0.2615
0.7367
9.94
0.6406
0.5631
0.2430
0.7576
PanoHead
14.16
0.6575
0.3492
0.1308
0.4163
14.44
0.7158
0.3098
0.1191
0.3727
GAGAvatar
13.54
0.6938
0.3643
0.1829
0.5228
14.55
0.7501
0.3169
0.1615
0.4757
LAM
11.36
0.6338
0.4545
0.2186
0.5680
11.58
0.6821
0.4326
0.2039
0.5510
Ours
15.58
0.6902
0.2866
0.1061
0.2812
17.33
0.7545
0.2154
0.0807
0.2588
Table 2. Extreme View Reconstruction Performance on Ava-256 and NeRSemble Datasets.
The perceptual loss is computed by aligning these fea-
ture representations between rendered and target images us-
ing cosine distance on ℓ2-normalized feature vectors from
both DINOv2 and SAM 2.1.
We train our method on a hybrid dataset composed of:
(1) real multi-view images from Nersemble [29], (2) multi-
view images of artificial personas from Cafca [5], and (3)
real single-view images from FFHQ [26]. This blend pro-
vides a strong balance between identity preservation, 3D
consistency, and visual diversity.
The training process occurs in two phases: First, we train
the base lifting model without the CNN sharpening module
for 70 hours on a single RTX 3090 GPU. Next, we freeze
the 3D reconstruction pipeline and train the CNN module
separately for 24 hours. During this second stage, the CNN
module’s sole task is to enhance 2D sharpness and remove
potential rendering artifacts. For this purpose, we supervise
the CNN module with a combination of DINOv2, L1 and
LPIPS losses, where LPIPS in particular is effective at en-
forcing perceptual sharpness.
3.5. 3D Editing Model
The editing model builds on the base lifting pipeline, re-
placing only the Lift Encoder with an Editing Encoder.
This ViT-based encoder takes 19-channel FaRL [64] seg-
mentation maps as input to guide geometry and integrates
a frozen CLIP [47] module to extract style features from
images. During inference, it enables both image- and zero-
shot text-driven stylization. Our setup disentangles geome-
try and style: the CLIP feature lacks sufficient spatial detail
for geometry, while the segmentation map carries no style
cues.
All other components remain unchanged from the base
model.
Fine-tuning the 3D editing model takes 30
hours, using our perceptual losses on Nersemble [29] and
FFHQ [26]. We exclude synthetic Cafca [5] data to avoid
biasing the model toward unrealistic reconstructions.
4. Results
4.1. Setup
We evaluate our model on 11 identities from the completely
unseen dataset Ava-256 [40] and on 5 held-out identities
from NeRSemble [29].
Evaluation Tasks. We assess 3D consistency under:
• Novel Views: Novel view synthesis from a single image
(5 views each for Ava-256, 16 views each for NeRSem-
ble).
• Extreme Views:
Most challenging viewpoint pairs
(left-right flips for Ava-256; plus vertical extremes for
NeRSemble).
Baselines. We compare against:
• LGM [54]: Diffusion-based multi-view Gaussian model.
• PanoHead [1]: 3D GAN with tri-grid NeRF and neural
upsampler, PTI Inversion [49] used for 3D reconstruction.
• GAGAvatar [8]: One-shot Gaussian lifting from 2D fea-
tures with 2D neural rendering.
• LAM [24]: Transformer-based Gaussian avatar genera-
tion from FLAME canonical points.
4.2. Quantitative Results
For our quantitative evaluation in Table 1 and Table 2, we
report PSNR and SSIM [57] as standard reconstruction met-

<!-- page 6 -->
Figure 3. Qualitative Evaluation on Samples From Ava-256 and NeRSemble.
Figure 4. 3D Reconstructions Across Video Frames. Our model
maintains consistent geometry and appearance across time, en-
abling coherent 3D avatar lifting while capturing subtle expression
changes like mouth, eye, and eyelid movements.
rics, LPIPS [63] and DreamSim (DS) [14] as perceptual
metrics, and ArcFace [9] distance to assess identity preser-
vation. All metrics are computed between the generated and
target view images.
Across both datasets and in both evaluation setups, our
method consistently outperforms all baselines in terms of
PSNR, LPIPS, DreamSim, and ArcFace, demonstrating
strong reconstruction fidelity, perceptual realism, and iden-
tity consistency. While GAGAvatar achieves slightly higher
SSIM, our method achieves superior results in all other met-
rics, particularly those better aligned with human perceptual
quality and recognition robustness.
The performance gap becomes even more pronounced
in the extreme views scenario, where input and target im-
ages are taken from drastically different angles (e.g., side or
top views). This setting simulates real-world use cases such
as immersive avatar creation from single-side views. Here,
other methods degrade significantly, especially in percep-
tual and identity metrics, while our method remains remark-
ably consistent.
These results indicate that our model generalizes well
across challenging viewpoints and maintains 3D-consistent
realism and identity preservation in extreme viewing condi-
tions.
4.3. Qualitative Results
Figure 3 presents a visual comparison between methods
under both novel-view and extreme-view conditions. Our

<!-- page 7 -->
Figure 5. Text-Based 3D Editing. Given a fixed segmentation map and varying text prompts, our model generates diverse 3D heads with
consistent geometry. Styles are guided by text, enabling low-level (e.g., hair color) and high-level (e.g., age) edits. Despite no text-specific
training, our model achieves zero-shot editing via the vision-aligned CLIP text encoder.
Figure 6. Conditional 3D Head Generation from Geometry
and Style. Our method disentangles geometry and style, enabling
diverse style transfer on fixed geometry and consistent appearance
across varying geometries for a given style.
model consistently produces realistic and 3D-consistent re-
constructions, maintaining detail and structural coherence
even under wide viewpoint changes.
It accurately com-
pletes unseen regions without introducing artifacts or de-
grading identity, highlighting its strong generalization and
geometric understanding. In contrast, PanoHead frequently
exhibits mirroring artifacts, generating implausible comple-
tions in occluded areas. GAGAvatar struggles with frontal-
to-side transitions, losing detail and geometric fidelity, and
often collapses under more extreme conditions. LAM fails
to maintain 3D consistency in its predicted Gaussians, lead-
ing to structural distortions when rendered from challenging
views. LGM also degrades in novel views, often hallucinat-
ing implausible and distorted head geometry and adding a
blue tint.
Another common downstream task is generating 3D
avatars from video, which is essential for applications like
telepresence, animation, and immersive media. Although
our model is trained on one-frame images, we show that it
generalizes to this setting by applying it frame-by-frame to
video input. Using two VFHQ [59] sequences, we evalu-
ate its ability to handle changes in facial expressions across
time. As shown in Fig. 4, our model produces reconstruc-
tions that are faithful to each frame while remaining consis-
tent in geometry and appearance across the entire sequence.
This cross-frame coherence is crucial for stable 3D avatars,
ensuring robustness to subtle expression changes—such as
mouth motion, eye dynamics, or blinking—without intro-
ducing temporal drift or identity shifts.
Notably, this is
achieved without any additional fine-tuning. Generated se-
quences are included in the supplementary material.
4.4. Disentangled 3D Editing
As shown in Fig. 6, our framework can generate 3D heads
by combining structural guidance from a segmentation map
with appearance cues extracted from a reference image.
This setup allows the geometry and style to be controlled
independently, enabling precise manipulation of head shape
and facial structure while freely varying texture, color, and
other stylistic attributes. The model preserves the input ge-
ometry when applying diverse styles, and conversely, main-
tains stylistic coherence when transferring the same style
across different geometries. Such behavior reflects a strong
disentanglement of the two factors, which is essential for
flexible and predictable editing.
Building on this capability, Fig. 5 demonstrates that our
model also supports text-driven 3D head editing in a zero-
shot manner. While it is trained only on image-based style
conditioning, we leverage the vision-aligned CLIP text en-
coder to map text prompts into the same latent space used
for style control. This allows our approach to interpret and
apply textual edits without any task-specific fine-tuning.
Geometry remains explicitly controlled via the input seg-

<!-- page 8 -->
Figure 7. Ablation Study on Data and Loss Variants. We compare 3D head reconstruction results for models trained with: (1) 2D data
only, (2) 3D multi-view data only, (3) LPIPS + L1 loss, (4) DINOv2 loss, (5) SAM2.1 loss, and (6) our full configuration.
Variant
PSNR ↑
SSIM ↑
LPIPS ↓
DS ↓
ArcFace ↓
2D
6.42
0.5362
0.6451
0.4230
0.7565
Multi-View
15.39
0.6898
0.2931
0.1092
0.3121
LPIPS+L1
15.72
0.7054
0.2877
0.1174
0.3030
DINOv2
15.26
0.7171
0.4085
0.1634
0.3479
SAM2.1
15.66
0.6742
0.3074
0.1063
0.3114
Full
15.58
0.6902
0.2866
0.1061
0.2812
LPIPS+L1 w.o. CNN
15.96
0.7228
0.2897
0.1179
0.2979
Full w.o. CNN
15.94
0.7089
0.3010
0.1161
0.2821
Table 3. Ablation Studies on Extreme Views for Ava-256.
mentation map, while style is modulated by varying the text
prompt. This disentanglement is particularly valuable for
iterative refinement, enabling users to explore a wide range
of appearance variations without affecting the underlying
structure. The model responds robustly to both low-level
visual edits — such as altering hair color or texture (e.g.,
curly hair) — and high-level semantic attributes, including
age, where it adapts skin and hair characteristics accord-
ingly.
A demo of our interactive web-application for this 3D
editing task is available in our supplementary materials.
4.5. Ablations
We study the effect of training data and loss formulation
on 3D reconstruction performance. Quantitative results are
reported in Table 3, and a qualitative comparison is shown
in Figure 7. All models use the same CNN module, which
was trained on a previous version even to our full model.
Training data.
Training only on 2D FFHQ samples
(2D) severely harms 3D consistency and reduces perfor-
mance across all metrics. In contrast, training only on multi-
view 3D data improves geometric consistency but still un-
derperforms, especially in identity preservation. This high-
lights that diverse single-image data provides strong appear-
ance priors that complement multi-view geometric supervi-
sion.
CNN module and standard loss.
The CNN refine-
ment mainly sharpens visuals, improving LPIPS, DS, and
ArcFace while slightly lowering PSNR and SSIM. With
CNN, our model also outperforms the LPIPS+L1 base-
line in LPIPS. Without CNN, LPIPS+L1 achieves the best
PSNR, SSIM, and LPIPS, but our model still surpasses it
in DS and ArcFace. This demonstrates (a) our method out-
performs a standard loss formulation overall, and (b) even
without the CNN — which is trained with LPIPS — super-
vision solely from generalized perceptual models can re-
main stronger than conventional losses.
Generalized Perceptual Losses The DINOv2-only
model (layers 8 and 11) preserves high-frequency de-
tails such as hair but produces less sharp overall results.
SAM2.1-only supervision achieves strong overall recon-
structions but underperforms in preserving details, such as
long hair strands. In contrast, our Full model — combin-
ing both DINOv2 and SAM2.1 supervision — achieves the
most realistic and identity-consistent outputs.
4.6. Limitations
While our method achieves high-quality single-image 3D
head reconstruction and editing, it also has certain limi-
tations.
Firstly, it lacks support for dynamic expression
transfer — an essential requirement for tasks such as fa-
cial reenactment and avatar animation. Secondly, the over-
all pipeline is not yet optimized for real-time inference; the
computational overhead involved in 3D lifting, rendering,
and refinement makes it impractical for applications that
require immediate feedback, such as live video conferenc-
ing or AR/VR environments. Lastly, our method currently
bakes lighting into the reconstructed scene, which limits the
model’s ability to generalize across different lighting condi-
tions or adapt to novel illumination setups.
5. Conclusions
We presented a new state-of-the-art approach for view-
angle robust 3D head reconstruction from a single image.
Our method introduces a novel and surprisingly effective
loss formulation that relies on perceptual training signals
from foundational models. This challenges the prevailing
assumption that conventional reconstruction losses (e.g.,
pixel-wise L1) or perceptual metrics (e.g., LPIPS) are nec-
essary for high-quality 3D supervision. Instead, we demon-
strate that generalized, task-agnostic features from powerful
foundation models enable consistent and realistic 3D geom-
etry and appearance.
In addition to strong reconstruction performance, our

<!-- page 9 -->
model enables a disentangled editing interface where
geometry is guided by segmentation maps and style
by text or reference images, opening new possibilities
for intuitive and semantically grounded 3D face editing.
References
[1] Sizhe An, Hongyi Xu, Yichun Shi, Guoxian Song, Umit
Ogras, and Linjie Luo. Panohead: Geometry-aware 3d full-
head synthesis in 360◦, 2023. 2, 5, 1, 3
[2] Shivangi Aneja, Justus Thies, Angela Dai, and Matthias
Nießner. Clipface: Text-guided editing of textured 3d mor-
phable models. In SIGGRAPH ’23 Conference Proceedings,
2023. 3
[3] Haoran Bai, Di Kang, Haoxian Zhang, Jinshan Pan, and Lin-
chao Bao. Ffhq-uv: Normalized facial uv-texture dataset for
3d face reconstruction. In IEEE Conference on Computer
Vision and Pattern Recognition, 2023. 2
[4] Ananta R. Bhattarai, Matthias Nießner, and Artem Sev-
astopolsky.
Triplanenet: An encoder for eg3d inversion.
2024. 2
[5] Marcel C. Buehler, Gengyan Li, Erroll Wood, Leonhard
Helminger, Xu Chen, Tanmay Shah, Daoye Wang, Stephan
Garbin, Sergio Orts-Escolano, Otmar Hilliges, Dmitry La-
gun, J´er´emy Riviere, Paulo Gotardo, Thabo Beeler, Abhim-
itra Meka, and Kripasindhu Sarkar.
Cafca: High-quality
novel view synthesis of expressive faces from casual few-
shot captures. In ACM SIGGRAPH Asia 2024 Conference
Paper. 2024. 2, 5
[6] Eric Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and
Gordon Wetzstein. pi-gan: Periodic implicit generative ad-
versarial networks for 3d-aware image synthesis. In arXiv,
2020. 2
[7] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki
Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo,
Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero
Karras, and Gordon Wetzstein. Efficient geometry-aware 3D
generative adversarial networks. In arXiv, 2021. 2
[8] Xuangeng Chu and Tatsuya Harada. Generalizable and an-
imatable gaussian head avatar.
In The Thirty-eighth An-
nual Conference on Neural Information Processing Systems,
2024. 2, 3, 5, 1
[9] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos
Zafeiriou. Arcface: Additive angular margin loss for deep
face recognition.
In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
4690–4699, 2019. 6
[10] Yu Deng, Duomin Wang, Xiaohang Ren, Xingyu Chen, and
Baoyuan Wang.
Portrait4d: Learning one-shot 4d head
avatar synthesis using synthetic data.
In IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, 2024.
2
[11] Yu Deng, Duomin Wang, and Baoyuan Wang. Portrait4d-v2:
Pseudo multi-view data creates better 4d head synthesizer.
arXiv preprint arXiv:2403.13570, 2024. 2
[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale.
arXiv preprint
arXiv:2010.11929, 2020. 2, 3
[13] Yao Feng, Haiwen Feng, Michael J. Black, and Timo
Bolkart.
Learning an animatable detailed 3D face model
from in-the-wild images. 2021. 2
[14] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy
Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dream-
sim: Learning new dimensions of human visual similarity
using synthetic data, 2023. 3, 6
[15] Baris Gecer, Stylianos Ploumpis, Irene Kotsia, and Stefanos
Zafeiriou. Ganfit: Generative adversarial network fitting for
high fidelity 3d face reconstruction. In The IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), 2019.
2
[16] Baris Gecer, Stylianos Ploumpis, Irene Kotsia, and Ste-
fanos P Zafeiriou. Fast-ganfit: Generative adversarial net-
work for high fidelity 3d face reconstruction. IEEE Trans-
actions on Pattern Analysis and Machine Intelligence, 2021.
2
[17] Dimitrios
Gerogiannis,
Foivos
Paraperas
Papantoniou,
Rolandos Alexandros Potamias, Alexandros Lattas, and Ste-
fanos Zafeiriou.
Arc2avatar:
Generating expressive 3d
avatars from a single image via id guidance. arXiv preprint
arXiv:2501.05379, 2025. 3
[18] Simon Giebenhain, Tobias Kirschstein, Martin R¨unz, Lour-
des Agapito, and Matthias Nießner. Npga: Neural paramet-
ric gaussian avatars. In SIGGRAPH Asia 2024 Conference
Papers, pages 1–11. ACM, 2024. 3
[19] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt.
Stylenerf:
A style-based 3d aware generator for high-
resolution image synthesis. In International Conference on
Learning Representations, 2022. 2
[20] Jiatao Gu, Qingzhe Gao, Shuangfei Zhai, Baoquan Chen,
Lingjie Liu, and Josh Susskind.
Control3Diff: Learning
Controllable 3D Diffusion Models from Single-view Images
.
In 2024 International Conference on 3D Vision (3DV),
pages 685–696, Los Alamitos, CA, USA, 2024. IEEE Com-
puter Society. 3
[21] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo
Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector
quantized diffusion model for text-to-image synthesis. arXiv
preprint arXiv:2111.14822, 2021. 3
[22] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong
Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffu-
sion training via min-snr weighting strategy. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion (ICCV), pages 7441–7451, 2023. 3
[23] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr
Doll´ar, and Ross Girshick. Masked autoencoders are scalable
vision learners. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 16000–
16009, 2022. 3, 4
[24] Yisheng He, Xiaodong Gu, Xiaodan Ye, Chao Xu, Zhengyi
Zhao, Yuan Dong, Weihao Yuan, Zilong Dong, and Liefeng
Bo. Lam: Large avatar model for one-shot animatable gaus-
sian head. In SIGGRAPH, 2025. 2, 3, 5, 1

<!-- page 10 -->
[25] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet,
Mohammad Norouzi, and Tim Salimans. Cascaded diffu-
sion models for high fidelity image generation. Journal of
Machine Learning Research, 23(47):1–33, 2022. 3
[26] Tero Karras, Samuli Laine, and Timo Aila. A style-based
generator architecture for generative adversarial networks.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2019. 2, 5
[27] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 3, 4
[28] Taras Khakhulin, Vanessa Sklyarova, Victor Lempitsky, and
Egor Zakharov. Realistic one-shot mesh-based head avatars.
In European Conference of Computer vision (ECCV), 2022.
2
[29] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim
Walter, and Matthias Nießner. Nersemble: Multi-view ra-
diance field reconstruction of human heads.
ACM Trans.
Graph., 42(4), 2023. 2, 5, 3
[30] Tobias Kirschstein,
Simon Giebenhain,
Jiapeng Tang,
Markos Georgopoulos, and Matthias Nießner.
Gghead:
Fast and generalizable 3d gaussian heads.
arXiv preprint
arXiv:2406.09377, 2024. 3
[31] Yushi Lan, Xuyi Meng, Shuai Yang, Chen Change Loy, and
Bo Dai. Self-supervised geometry-aware encoder for style-
based 3d gan inversion. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 20940–20949, 2023. 3
[32] Alexandros Lattas, Stylianos Moschoglou, Baris Gecer,
Stylianos
Ploumpis,
Vasileios
Triantafyllou,
Abhijeet
Ghosh, and Stefanos Zafeiriou. Avatarme: Realistically ren-
derable 3d facial reconstruction ”in-the-wild”. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2020. 2
[33] Alexandros
Lattas,
Stylianos
Moschoglou,
Stylianos
Ploumpis, Baris Gecer, Abhijeet Ghosh, and Stefanos P
Zafeiriou. Avatarme++: Facial shape and brdf inference with
photorealistic rendering-aware gans. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 2021. 2
[34] Jianhui Li, Jianmin Li, Haoji Zhang, Shilong Liu, Zhengyi
Wang, Zihao Xiao, Kaiwen Zheng, and Jun Zhu. Preim3d:
3d consistent precise image attribute editing from a single
image.
In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
8549–8558, 2023. 3
[35] Jianhui Li, Shilong Liu, Zidong Liu, Yikai Wang, Kaiwen
Zheng, Jinghui Xu, Jianmin Li, and Jun Zhu.
Instruct-
pix2nerf: Instructed 3d portrait editing from a single image,
2024. 3
[36] Tianye Li, Timo Bolkart, Michael. J. Black, Hao Li, and
Javier Romero. Learning a model of facial shape and ex-
pression from 4D scans. ACM Transactions on Graphics,
(Proc. SIGGRAPH Asia), 36(6):194:1–194:17, 2017. 4
[37] Zhanfeng Liao, Yuelang Xu, Zhe Li, Qijing Li, Boyao Zhou,
Ruifeng Bai, Di Xu, Hongwen Zhang, and Yebin Liu. Hha-
vatar: Gaussian head avatar with dynamic hairs. arXiv e-
prints, pages arXiv–2312, 2023. 3
[38] Jiangke Lin, Yi Yuan, Tianjia Shao, and Kun Zhou.
To-
wards high-fidelity 3d face reconstruction from in-the-wild
images using graph convolutional networks. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 5891–5900, 2020. 2
[39] Xin Lin, Jingtong Yue, Kelvin C. K. Chan, Lu Qi, Chao
Ren, Jinshan Pan, and Ming-Hsuan Yang. Multi-task image
restoration guided by robust dino features, 2024. 3
[40] Julieta Martinez, Emily Kim, Javier Romero, Timur Bagaut-
dinov, Shunsuke Saito, Shoou-I Yu, Stuart Anderson,
Michael Zollh¨ofer, Te-Li Wang, Shaojie Bai, Chenghui Li,
Shih-En Wei, Rohan Joshi, Wyatt Borsos, Tomas Simon,
Jason Saragih, Paul Theodosis, Alexander Greene, Anjani
Josyula, Silvio Mano Maeta, Andrew I. Jewett, Simon Ven-
shtain, Christopher Heilman, Yueh-Tung Chen, Sidi Fu, Mo-
hamed Ezzeldin A. Elshaer, Tingfang Du, Longhua Wu,
Shen-Chi Chen, Kai Kang, Michael Wu, Youssef Emad,
Steven Longay, Ashley Brewer, Hitesh Shah, James Booth,
Taylor Koska, Kayla Haidle, Matt Andromalos, Joanna Hsu,
Thomas Dauer, Peter Selednik, Tim Godisart, Scott Ardis-
son, Matthew Cipperly, Ben Humberston, Lon Farr, Bob
Hansen, Peihong Guo, Dave Braun, Steven Krenn, He Wen,
Lucas Evans, Natalia Fadeeva, Matthew Stewart, Gabriel
Schwartz, Divam Gupta, Gyeongsik Moon, Kaiwen Guo,
Yuan Dong, Yichen Xu, Takaaki Shiratori, Fabian Prada,
Bernardo R. Pires, Bo Peng, Julia Buffalini, Autumn Trim-
ble, Kevyn McPhail, Melissa Schoeller, and Yaser Sheikh.
Codec Avatar Studio: Paired Human Captures for Complete,
Driveable, and Generalizable Avatars.
NeurIPS Track on
Datasets and Benchmarks, 2024. 2, 5
[41] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V.
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Rus-
sell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-
Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nico-
las Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou,
Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bo-
janowski. Dinov2: Learning robust visual features without
supervision, 2023. 1, 2, 3, 4
[42] Roy
Or-El,
Xuan
Luo,
Mengyi
Shan,
Eli
Shecht-
man, Jeong Joon Park, and Ira Kemelmacher-Shlizerman.
StyleSDF: High-Resolution 3D-Consistent Image and Ge-
ometry Generation. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 13503–13513, 2022. 2
[43] Wamiq Reyaz Para, Abdelrahman Eldesokey, Zhenyu Li,
Pradyumna Reddy, Jiankang Deng, and Peter Wonka.
Avatarmmc: 3d head avatar generation and editing with
multi-modal conditioning, 2024. 3
[44] Foivos Paraperas Papantoniou, Alexandros Lattas, Stylianos
Moschoglou, Jiankang Deng, Bernhard Kainz, and Stefanos
Zafeiriou. Arc2face: A foundation model for id-consistent
human faces. In Proceedings of the European Conference on
Computer Vision (ECCV), 2024. 3
[45] Dustin
Podell,
Zion
English,
Kyle
Lacey,
Andreas
Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and
Robin Rombach. Sdxl: Improving latent diffusion models
for high-resolution image synthesis, 2023. 3

<!-- page 11 -->
[46] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide
Davoli, Simon Giebenhain, and Matthias Nießner.
Gaus-
sianavatars: Photorealistic head avatars with rigged 3d gaus-
sians. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 20299–20309,
2024. 3
[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever.
Learning transferable visual
models from natural language supervision, 2021. 2, 5, 1
[48] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junt-
ing Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-
Yuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feicht-
enhofer. Sam 2: Segment anything in images and videos.
arXiv preprint arXiv:2408.00714, 2024. 1, 2, 3, 4
[49] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel
Cohen-Or. Pivotal tuning for latent-based editing of real im-
ages. ACM Trans. Graph., 2021. 2, 5
[50] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models, 2021. 3
[51] Shoaib Meraj Sami, Md Mahedi Hasan, Jeremy Dawson, and
Nasser Nasrabadi. Hf-diff: High-frequency perceptual loss
and distribution matching for one-step diffusion-based image
super-resolution, 2024. 3
[52] Wenzhe Shi, Jose Caballero, Ferenc Husz´ar, Johannes Totz,
Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan
Wang. Real-time single image and video super-resolution
using an efficient sub-pixel convolutional neural network. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 1874–1883, 2016. 4
[53] Jingxiang Sun, Xuan Wang, Yichun Shi, Lizhen Wang, Jue
Wang, and Yebin Liu. Ide-3d: Interactive disentangled edit-
ing for high-resolution 3d-aware portrait synthesis.
ACM
Transactions on Graphics (TOG), 41(6):1–10, 2022. 3
[54] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang,
Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian
model for high-resolution 3d content creation. arXiv preprint
arXiv:2402.05054, 2024. 2, 3, 5, 1
[55] Lizhen Wang, Zhiyua Chen, Tao Yu, Chenguang Ma, Liang
Li, and Yebin Liu.
Faceverse: a fine-grained and detail-
controllable 3d face morphable model from a hybrid dataset.
In IEEE Conference on Computer Vision and Pattern Recog-
nition (CVPR2022), 2022. 2
[56] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin
Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang
Wen, Qifeng Chen, and Baining Guo. Rodin: A generative
model for sculpting 3d digital avatars using diffusion, 2022.
3
[57] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 5
[58] Jiaxin Xie, Hao Ouyang, Jingtan Piao, Chenyang Lei, and
Qifeng Chen.
High-fidelity 3d gan inversion by pseudo-
multi-view optimization. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 321–331, 2023. 3
[59] Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong,
and Ying Shan. Vfhq: A high-quality dataset and bench-
mark for video face super-resolution.
In The IEEE Con-
ference on Computer Vision and Pattern Recognition Work-
shops (CVPRW), 2022. 2, 7
[60] Xu Yinghao, Shi Zifan, Yifan Wang, Chen Hansheng, Yang
Ceyuan, Peng Sida, Shen Yujun, and Wetzstein Gordon.
Grm: Large gaussian reconstruction model for efficient 3d
reconstruction and generation, 2024. 4
[61] Picosson Yong and Wiliem. Mtred: 3d reconstruction dataset
for fly-over videos of maritime domain. In MaCVi, 2024. 3
[62] Bowen Zhang, Yiji Cheng, Chunyu Wang, Ting Zhang, Jiao-
long Yang, Yansong Tang, Feng Zhao, Dong Chen, and Bain-
ing Guo. Rodinhd: High-fidelity 3d avatar generation with
diffusion models. arXiv preprint arXiv:2407.06938, 2024. 3
[63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 3, 6
[64] Yinglin Zheng, Hao Yang, Ting Zhang, Jianmin Bao,
Dongdong Chen, Yangyu Huang, Lu Yuan, Dong Chen,
Ming Zeng, and Fang Wen.
General facial representa-
tion learning in a visual-linguistic manner. arXiv preprint
arXiv:2112.03109, 2021. 2, 5, 1

<!-- page 12 -->
PercHead: Perceptual Head Model
for Single-Image 3D Head Reconstruction & Editing
Supplementary Material
6. Evaluation Subjects and Processing
Subjects used for quantitative evaluation:
• NeRSemble: 059, 070, 370, 373, 374
• Ava-256:
– 20220809--1034--BJM420
– 20220815--1307--BMP511
– 20220831--0751--CMS162
– 20230224--1359--CMZ386
– 20230308--1352--BDF920
– 20230316--1103--BHK376
– 20230324--0820--AEY864
– 20230328--0800--BLY735
– 20230405--1635--AAN112
– 20230810--1630--ANX726
– 20230914--1105--BXQ083
These subjects were used for all evaluated methods, in-
cluding our model, PanoHead [1], Large Avatar Model
(LAM) [24], Large Gaussian Model (LGM) [54], and
GAGAvatar [8].
Cropping Alignment
We observed that PanoHead [1]
uses the tightest (smallest) image crops among all com-
pared methods. To ensure fair and consistent evaluation
across models, we applied the same PanoHead cropping to
all methods for qualitative and quantitative comparison.
GAGAvatar Processing
For GAGAvatar [8], their de-
fault rendering pipeline produces images with a black back-
ground. To standardize appearance and ensure comparabil-
ity across methods, we replace the black background with
white using their official GAGAvatar Track [8] prepro-
cessing pipeline.
7. Decoder Visualization Protocol
To understand the information flow in our 3D lifting de-
coder, we visualize intermediate outputs after each decoder
layer. For each visualization, we run a full forward pass,
but control the activation of the cross-attention mechanisms.
Specifically, to visualize the output after decoder layer i,
we keep all cross-attention layers active up to and including
layer i, while disabling cross-attention for all subsequent
layers. Importantly, we retain the MLP blocks and skip con-
nections in all layers, ensuring that feature propagation and
refinement still occur. This setup allows us to isolate the
contribution of 2D feature retrieval up to a specific depth in
the decoder.
8. 3D Editing Web Application
Our 3D editing web application allows users to extract a
segmentation map from an input image and interactively
modify it via drawing.
For stylization, users can either
upload a reference image or provide a text prompt.
In
our supplementary demo video, extracting a segmentation
map from an image takes 25 seconds, as it involves both
GAGAvatar’s preprocessing [8] and FARL [64] for seman-
tic segmentation. This step is only required when uploading
a new segmentation image—not when editing an existing
one. Stylization with a reference image takes 28 seconds,
which includes CLIP-based [47] feature extraction and a
forward pass through our model. In contrast, stylization us-
ing a text prompt is significantly faster, requiring only 10
seconds.
9. Supplementary Video
We highly recommend watching our supplementary video,
which showcases additional 3D reconstruction orbit views,
frame-by-frame 3D video generation, 3D edit orbit se-
quences, and a live demo of our interactive 3D editing web
application.

<!-- page 13 -->
Figure 8. Additional Results on Ava-256 [40] and Nersemble [29]. We present reconstructions across diverse viewpoint pairs: side-
to-frontal, frontal-to-side, side-to-side, and vertical angle changes. Competing methods often struggle with side and vertical viewpoints,
whereas our method consistently produces realistic and geometrically coherent results.

<!-- page 14 -->
Figure 9. Qualitative Comparison on Reconstruction to Different Target Angles on a Nersemble [29] Sample. We compare recon-
structions from a frontal input view across multiple target angles. While methods like GAGAvatar [8], PanoHead [1], and LAM [24] excel
at preserving identity in the frontal view, they degrade significantly under large view changes. In contrast, our method maintains high
quality and consistent identity across all target views — crucial for immersive 3D applications — while remaining competitive even in the
frontal case.
