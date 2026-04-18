<!-- page 1 -->
SEMANTIC-GUIDED 3D GAUSSIAN SPLATTING FOR TRANSIENT OBJECT REMOVAL
Aditi Prabakaran
Department of Computer Science
SRM University
Chennai, India
ap0232@srmist.edu.in
Priyesh Shukla
Computer Systems Group
International Institute of Information Technology
Hyderabad, India
priyesh.shukla@iiit.ac.in
ABSTRACT
Transient objects in casual multi-view captures cause ghost-
ing artifacts in 3D Gaussian Splatting (3DGS) reconstruction.
Existing solutions relied on scene decomposition at signif-
icant memory cost or on motion-based heuristics that were
vulnerable to parallax ambiguity. A semantic filtering frame-
work was proposed for category-aware transient removal us-
ing vision-language models. CLIP similarity scores between
rendered views and distractor text prompts were accumulated
per-Gaussian across training iterations. Gaussians exceeding
a calibrated threshold underwent opacity regularization and
periodic pruning. Unlike motion-based approaches, seman-
tic classification resolved parallax ambiguity by identifying
object categories independently of motion patterns. Exper-
iments on the RobustNeRF benchmark demonstrated consis-
tent improvement in reconstruction quality over vanilla 3DGS
across four sequences, while maintaining minimal memory
overhead and real-time rendering performance.
Threshold
calibration and comparisons with baselines validated seman-
tic guidance as a practical strategy for transient removal in
scenarios with predictable distractor categories.
Index Terms— 3D Gaussian Splatting, Vision-language
models, CLIP, Semantic filtering, Transient objects
1. INTRODUCTION
3D Gaussian Splatting (3DGS) [1] turned out to be an ef-
ficient alternative to Neural Radiance Fields (NeRF) [2]
because of it’s ability to perform efficient novel view syn-
thesis. By modeling scenes explicitly as 3D Gaussians which
were optimized through differentiable rasterization, 3DGS
achieved real-time rendering and was much faster to train in
comparison to implicit radiance fields employed in methods
such as Mip-NeRF 360 [3]. Despite such advancements, both
implicit and explicit neural rendering frameworks assumed
static views in scenes observed. When multi-view captures
consisted of transient objects such as people walking or items
being moved, observations were inconsistent across views
which led to ghosting artifacts in place of the transient object
in the reconstructed scene.
Several approaches addressed the issue of handling tran-
sient objects in frameworks that involved radiance field. Ro-
bustNeRF [5] proposed loss formulations to reduce the im-
pact distractors had on a scene during optimization. NeRF in
the Wild [6] incorporated per-image appearance embeddings
to account for capture conditions that were not constrained.
NeRF On-the-Go [7] made use of uncertainty estimation to
suppress dynamic content in captures of the real-world. Al-
though such approaches proved to be effective, these meth-
ods operated within implicit volumetric representations and
required excess computational resources to train.
Filtering methods based on motion and visibility had am-
biguity as parallax observed in static geometry or transient
objects led to low visibility. Static scene boundaries appeared
inconsistently across views due to camera motion, leading to
pruning more than what is needed when visibility alone was
used as the filtering factor.
In this work, a semantic-guided framework for transient
object removal in 3DGS was proposed. Instead of relying on
motion patterns for detecting transience, CLIP was used to
classify rendered training views against predefined distractor
categories.
Semantic scores for images were noted at the
Gaussian level while optimizing through iterations, which
in turn provided estimates for normalized per-Gaussian se-
mantic, thus depicting category consistency rather than view
frequency. Gaussians which appeared to be associated with
transient categories were progressively suppressed by opacity
regularization and pruning periodically, while static geome-
try was preserved. Experiments performed on the Robust-
NeRF benchmark [5] demonstrated that semantic guidance
effectively resolved motion-parallax ambiguity and improved
reconstruction quality over vanilla 3DGS.
2. RELATED WORK
2.1. 3D Gaussian Splatting
Kerbl et al. [1] introduced 3D Gaussian Splatting, optimized
by differentiable rasterization for 3D scene representation.
Real-time rendering was achieved while maintaining qual-
ity comparable to NeRF [2]. In comparison to Mip-NeRF
arXiv:2602.15516v1  [cs.CV]  17 Feb 2026

<!-- page 2 -->
360 [3], training time was significantly reduced.
Recent
extensions addressed anti-aliasing [13] and geometric accu-
racy [12].
2.2. Transient Object Handling in Neural Rendering
RobustNeRF [5] proposed loss formulation to reduce the im-
pact of distractors while training. NeRF in the Wild [6] used
per-image appearance embeddings to reduce transient objects
in captures. NeRF On-the-Go [7] used uncertainty estima-
tion to suppress distractors while structured representations
and appearance modeling were explored by Scaffold-GS and
Gaussian Shader [14], [15]. Yet, long training and volumetric
representations remained limitations.
2.3. Vision-Language Models for 3D Understanding
Vision-language models were integrated into 3D representa-
tions to provide semantic grounding. CLIP [4], trained on
large-scale image-text pairs, demonstrated strong zero-shot
classification capabilities. LERF [9] embedded CLIP-aligned
features into radiance fields to enable open-vocabulary scene
querying. DINOv2 [11] and SAM [8] demonstrated semantic
understanding that could guide 3D scene analysis.
Unlike prior methods that maintained dense semantic
embeddings throughout the rendering pipeline, the proposed
framework employed CLIP only during training to guide
structural pruning. This preserved the lightweight and real-
time properties of 3DGS while enabling semantically in-
formed transient suppression.
3. METHODOLOGY
3.1. Overview
Given a set of multi-view images that had transient objects,
the goal was to reconstruct static scene geometry while sup-
pressing distractors. The scene was represented as a collec-
tion of 3D Gaussians, following the 3DGS representation.
The framework extended on the baseline 3DGS optimization
with semantic filtering with scoring of rendered views using
CLIP, per-Gaussian accumulation of semantic features, and
category-aware pruning. The overall pipeline is illustrated in
Fig. 1.
3.2. CLIP-Based Semantic Scoring
For each training iteration t, view It was rendered from cam-
era pose Ct by splatting [1]:
It = Render(G, Ct).
(1)
The rendered image It ∈R3×H×W was passed to the CLIP
ViT-B/32 vision encoder to obtain image features:
fI = CLIPvision(It) ∈R512.
(2)
Two classes of text prompts were defined: distractor prompts
D for transient categories and objects and static prompts S
for permanent and stationary scene elements. For the Robust-
NeRF dataset, the following prompts were used:
D = {“a photo of a person”, “a photo of people”,
“a photo of pedestrians”, “a photo of hands”,
“a photo of a balloon”},
(3)
S = {“a photo of a building”, “a photo of a wall”,
“a photo of furniture”}.
(4)
Each prompt p was encoded through the CLIP text encoder:
fp = CLIPtext(p) ∈R512.
(5)
Both image and text features were L2-normalized. Cosine
similarity between the rendered image and each distractor
prompt was computed as:
sim(It, p) =
fI · fp
∥fI∥∥fp∥.
(6)
The distractor score for iteration t was taken as the maximum
similarity across all distractor prompts:
s(t)
d
= max
p∈D sim(It, p).
(7)
Since cosine similarity ranged between [−1, 1], scores were
normalized to [0, 1]:
ˆs(t)
d
= s(t)
d + 1
2
.
(8)
High distractor scores (ˆs(t)
d
> 0.5) indicated that the rendered
view might contain transient elements. A static scene score
ˆs(t)
s
was computed using prompts from S (Eq. 4).
3.3. Per-Gaussian Score Accumulation
Image-level scores from Eq. 7 indicated if a view had distrac-
tors but did not directly identify the responsible Gaussians.
Semantic evidence was therefore accumulated at the Gaus-
sian level based on visibility across training iterations.
For each Gaussian Gj, two metrics were maintained: ac-
cumulated score ˜sj and view count nj. At iteration t, visibil-
ity was found by rasterization. For v(t)
j
∈{0, 1}, indicating
whether Gaussian j contributed to the rendered image. The
accumulated score was updated as:
˜s(t)
j
=



˜s(t−1)
j
+ β · max

0, ˆs(t)
d −0.5

,
if v(t)
j
= 1,
˜s(t−1)
j
,
otherwise,
(9)
where β = 0.1 controlled the accumulation rate. Accumula-
tion occurred only when the view’s distractor score exceeded

<!-- page 3 -->
3D Scene
Gaussians
Rendering
CLIP Scoring
Vision Encoder
fI ∈R512
Per-Gaussian
Accumulation
Category-Aware
Pruning
Clean
Gaussians
Preserved static geometry
Training View
Reconstruction
Gaussian Splats
Static (blue) and transient (red)
3D Gaussian representations
Fig. 1. Overview of the CLIP-GS framework. Training views were rendered from 3D Gaussians, semantically scored using
CLIP against distractor prompts, aggregated into per-Gaussian semantic scores, and applied for opacity regularization and
periodic pruning over iterative optimization to remove transient objects while preserving static geometry.
the neutral threshold of 0.5, preventing evidence from clean
views from influencing semantic scores. The view count was
updated as:
n(t)
j
= n(t−1)
j
+ v(t)
j .
(10)
After T iterations, the normalized per-Gaussian semantic
score was:
sj =
˜s(T )
j
n(T )
j
.
(11)
Normalization by view count in Eq. 11 ensured that seman-
tic scores reflected average category consistency rather than
visibility frequency, preventing high-frequency viewpoint re-
gions from accumulating disproportionately large absolute
scores.
3.4. Category-Aware Pruning
Transient suppression was performed through two comple-
mentary mechanisms: continuous opacity regularization and
periodic discrete pruning.
3.4.1. Opacity Regularization
After initial geometry stabilization, a semantic regularization
term was incorporated into the standard photometric loss:
L = Lphoto + λc LCLIP,
(12)
where Lphoto corresponded to the original 3DGS photometric
loss [1]. The semantic regularization term was:
LCLIP = 1
N
N
X
j=1
sj αj,
(13)
with N denoting the current number of Gaussians.
This
term penalized the opacity of Gaussians with high seman-
tic scores, encouraging progressive suppression of transient
elements throughout optimization.
3.4.2. Periodic Pruning
At fixed intervals during training, Gaussians were removed
according to:
 sj > τ

∨
 (nj < nmin) ∧(αj < αmin)

,
(14)
where τ was the semantic score threshold, nmin = 10 was
the minimum view count, and αmin = 0.1 was the mini-
mum opacity. The first condition in Eq. 14 removed seman-
tically identified distractors, while the second removed geo-
metrically unstable Gaussians with insufficient visibility and
low opacity. The threshold τ was calibrated based on the nor-
malized score distribution, as analyzed in Section 4.
3.5. Handling Dynamic Gaussian Count
Since 3DGS performed densification and pruning during op-
timization, the number of Gaussians varied over time. When
new Gaussians were introduced through splitting or cloning,
their semantic statistics were initialized to zero. When Gaus-
sians were removed, corresponding entries were discarded
from the tracking arrays. This ensured consistent accumu-
lation of semantic statistics throughout optimization without
introducing bias from uninitialized primitives.
3.6. Implementation Details
The proposed method was implemented as an extension of the
original 3DGS framework [1]. CLIP ViT-B/32 was employed

<!-- page 4 -->
Table 1. Quantitative comparison on RobustNeRF sequences. CLIP-GS consistently improved over all baselines under identical
training settings while incurring minimal memory overhead.
Statue
Android
Yoda
Crab(2)
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Vanilla 3DGS [1]
20.04
0.79
0.25
25.20
0.81
0.31
26.20
0.76
0.45
24.50
0.76
0.45
Mip-NeRF 360 [3]
19.74
0.79
0.24
25.80
0.81
0.32
26.12
0.76
0.43
25.80
0.76
0.44
CLIP-GS (Ours)
21.98
0.78
0.25
26.12
0.83
0.28
26.80
0.77
0.44
24.18
0.78
0.40
in inference mode without parameter updates. Rendered im-
ages were resized to 224 × 224 prior to CLIP encoding. Op-
timization followed the standard 3DGS schedule of 20,000
iterations with adaptive density control. Semantic score accu-
mulation was activated from iteration 500, opacity regulariza-
tion from iteration 2,000, and periodic pruning from iteration
5,000 at intervals of 1,000 iterations. Hyperparameters were
set to: β = 0.1, λc = 0.01, τ ∈[0.015, 0.02], nmin = 10,
and αmin = 0.1. Memory overhead relative to vanilla 3DGS
remained minimal, as only two additional per-Gaussian scalar
arrays were maintained.
4. EXPERIMENTS
4.1. Experimental Setup
The proposed method was evaluated on the RobustNeRF
dataset [5], using the Statue, Android, Yoda, and Crab(2)
sequences.
All methods were initialized using COLMAP
camera poses and trained under identical optimization set-
tings for fair comparison. The proposed CLIP-GS was com-
pared against vanilla 3DGS [1] and Mip-NeRF 360 [3] as
competitive baselines. Distractor prompts included descrip-
tions of people, pedestrians, hands, and moving objects as
specified in Eq. 3. Reconstruction quality was evaluated us-
ing peak signal-to-noise ratio (PSNR), structural similarity
index (SSIM), and learned perceptual image patch similarity
(LPIPS) [10].
4.2. Quantitative Results
Table 1 presents the quantitative comparison across all four
RobustNeRF sequences.
CLIP-GS achieved the highest
PSNR on three of four sequences, with gains of up to
+1.94 dB over vanilla 3DGS (Statue) and +0.92 dB over
Mip-NeRF 360 (Android).
Consistent SSIM and LPIPS
improvements were also observed, indicating improved per-
ceptual quality in addition to pixel-level fidelity.
Threshold calibration was found to be critical for effective
pruning. An initial threshold of τ = 0.3 resulted in negligible
pruning and yielded degraded reconstruction quality. Analy-
sis of the normalized score distribution from Eq. 11 revealed
that scores were distributed within the range [0.01, 0.03] af-
ter view-count normalization, necessitating thresholds in the
interval τ ∈[0.015, 0.02]. The optimal threshold τ = 0.015
achieved the best reconstruction quality, with 3.8% of Gaus-
sians removed. More aggressive pruning at τ = 0.01 led to
over-suppression, removing 8.1% of Gaussians and degrading
reconstruction quality.
Opacity regularization alone yielded a +0.5 dB improve-
ment, while periodic pruning alone yielded +0.8 dB; the full
combined framework achieved the maximum +1.3 dB gain.
Removing dataset-specific prompts reduced performance by
0.6 dB, though even generic prompts maintained a +0.7 dB
improvement over the vanilla baseline.
4.3. Qualitative Results
Fig. 2 presents visual comparisons on held-out test views.
Vanilla 3DGS produced ghosting artifacts where transient ob-
jects appeared semi-transparently due to inconsistent multi-
view observations. Mip-NeRF 360 exhibited similar ghost-
ing patterns, as neither method incorporated explicit transient
suppression.
CLIP-GS successfully removed distractor ar-
tifacts while preserving static scene boundaries. Walls ob-
served in as few as 15% of views were correctly retained
through semantic classification as static elements, rather than
incorrectly pruned based on low visibility. Residual imper-
fections were observed for small or distant transient objects,
where reduced image resolution degraded CLIP confidence,
suggesting that patch-level scoring could further improve lo-
calization in future work.
5. DISCUSSION
5.1. Advantages of Semantic Guidance
The proposed approach addressed a fundamental limita-
tion of motion-based filtering through semantic reasoning.
Visibility-based methods suffered from inherent parallax am-
biguity: a Gaussian observed in few views could correspond
either to a genuine transient object or to static geometry under
strong viewpoint variation. Semantic classification resolved
this by assigning category labels independently of geomet-
ric cues. Explicit category specification proved effective in
distinguishing static surfaces from transients: walls visible
in only 15% of views in the Statue sequence were correctly
identified as “building” and preserved, while pedestrians were
reliably removed despite similar visibility profiles.

<!-- page 5 -->
Input
Vanilla 3DGS
Mip-NeRF 360
CLIP-GS (Ours)
Yoda
Statue
Android
Crab (2)
Fig. 2. Qualitative comparison for scene reconstruction on RobustNeRF sequences for Vanilla 3DGS, Mip-NeRF 360 and
CLIP-GS.
5.2. Practical Deployment Considerations
While approaches such as scene decomposition achieved
higher absolute reconstruction quality, they incurred more
memory overhead. CLIP-GS achieved consistent improve-
ments over vanilla 3DGS, preserving real-time rendering.
This efficiency-quality trade-off made the framework suit-
able for resource-constrained deployment scenarios where
memory constraint and rendering speed were critical.
5.3. Limitations
The current implementation has three practical limitations.
First, users must specify distractor categories before train-
ing, which requires knowing what transient objects appear
in the scene. Generic categories like “person” still worked
well across different scenes (+0.7 dB improvement). Second,
CLIP performed worse on small objects (fewer than 50 pix-
els), leading to incomplete removal of distant people. Further
exploration could employ patch-level classification to better
handle small objects. Third, the filtering threshold τ needed
adjustment for each dataset, though values stayed within a
small range (τ ∈[0.015, 0.02]) across all sequences tested in
Section 4.
6. CONCLUSION
A semantic-guided framework for transient object removal
in 3D Gaussian Splatting was presented.
CLIP-based se-
mantic scoring was accumulated per-Gaussian across training
iterations to enable category-aware pruning through opacity
regularization and periodic removal.
The proposed CLIP-
GS demonstrated consistent improvement in reconstruction
quality over vanilla 3DGS and Mip-NeRF 360 across four
RobustNeRF sequences, while maintaining minimal memory
overhead and preserving real-time rendering performance.
Threshold calibration analysis confirmed that normalized se-
mantic scores required dataset-specific tuning, and ablation
studies validated the complementary contributions of both
suppression mechanisms.
Future work will investigate patch-level semantic scoring
to improve localization of small transients, learned prompt
generation to reduce manual category specification, and adap-
tive thresholding strategies to improve generalization across
diverse capture conditions.

<!-- page 6 -->
7. REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Dret-
takis, “3D Gaussian Splatting for Real-Time Radiance
Field Rendering,” ACM Trans. Graph., vol. 42, no. 4,
pp. 139:1–139:14, 2023.
[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Bar-
ron, R. Ramamoorthi, and R. Ng, “NeRF: Represent-
ing Scenes as Neural Radiance Fields for View Synthe-
sis,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020,
pp. 405–421.
[3] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan,
and P. Hedman, “Mip-NeRF 360: Unbounded Anti-
Aliased Neural Radiance Fields,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022,
pp. 5470–5479.
[4] A. Radford et al., “Learning Transferable Visual Mod-
els From Natural Language Supervision,” in Proc. Int.
Conf. Mach. Learn. (ICML), 2021, pp. 8748–8763.
[5] S. Sabour et al., “RobustNeRF: Ignoring Distractors
with Robust Losses,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. (CVPR), 2023, pp. 20626–20636.
[6] R. Martin-Brualla et al., “NeRF in the Wild: Neural Ra-
diance Fields for Unconstrained Photo Collections,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), 2021, pp. 7210–7219.
[7] W. Ren et al., “NeRF On-the-Go:
Exploiting Un-
certainty for Distractor-Free NeRFs in the Wild,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), 2024, pp. 8931–8940.
[8] A. Kirillov et al., “Segment Anything,” in Proc.
IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023,
pp. 4015–4026.
[9] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and
M. Tancik, “LERF: Language Embedded Radiance
Fields,” in Proc. IEEE/CVF Int. Conf. Comput. Vis.
(ICCV), 2023, pp. 19729–19739.
[10] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and
O. Wang, “The Unreasonable Effectiveness of Deep
Features as a Perceptual Metric,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018,
pp. 586–595.
[11] M. Oquab et al., “DINOv2: Learning Robust Visual
Features without Supervision,” Trans. Mach. Learn.
Res. (TMLR), 2024.
[12] B. Huang et al., “2D Gaussian Splatting for Geomet-
rically Accurate Radiance Fields,” in Proc. ACM SIG-
GRAPH, 2024.
[13] Z. Yu et al., “Mip-Splatting: Alias-Free 3D Gaussian
Splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pat-
tern Recognit. (CVPR), 2024, pp. 19447–19456.
[14] T. Lu et al., “Scaffold-GS: Structured 3D Gaussians
for View-Adaptive Rendering,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024,
pp. 20654–20664.
[15] Y. Jiang et al., “GaussianShader: 3D Gaussian Splat-
ting with Shading Functions for Reflective Surfaces,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), 2024, pp. 5322–5332.
