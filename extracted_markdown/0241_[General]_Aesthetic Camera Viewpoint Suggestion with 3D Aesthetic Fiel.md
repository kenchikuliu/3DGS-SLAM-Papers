<!-- page 1 -->
Aesthetic Camera Viewpoint Suggestion with 3D Aesthetic Field
Sheyang Tang1, Armin Shafiee Sarvestani1, Jialu Xu1, Xiaoyu Xu2, Zhou Wang1
1University of Waterloo, 2City University of Hong Kong
{sheyang.tang,a5shafie,jialu.xu,zhou.wang}@uwaterloo.ca
Figure 1. Given sparse scene captures (left), our method learns a 3D aesthetic field that encodes spatially varying aesthetic cues. This field
enables geometry-grounded aesthetic reasoning in 3D, allowing efficient discovery of appealing camera viewpoints (right).
Abstract
The aesthetic quality of a scene depends strongly on camera
viewpoint. Existing approaches for aesthetic viewpoint sug-
gestion are either single-view adjustments, predicting lim-
ited camera adjustments from a single image without un-
derstanding scene geometry, or 3D exploration approaches,
which rely on dense captures or prebuilt 3D environments
coupled with costly reinforcement learning (RL) searches.
In this work, we introduce the notion of 3D aesthetic field
that enables geometry-grounded aesthetic reasoning in 3D
with sparse captures, allowing efficient viewpoint sugges-
tions in contrast to costly RL searches. We opt to learn
this 3D aesthetic field using a feedforward 3D Gaussian
Splatting network that distills high-level aesthetic knowl-
edge from a pretrained 2D aesthetic model into 3D space,
enabling aesthetic prediction for novel viewpoints from only
sparse input views. Building on this field, we propose a
two-stage search pipeline that combines coarse viewpoint
sampling with gradient-based refinement, efficiently identi-
fying aesthetically appealing viewpoints without dense cap-
tures or RL exploration. Extensive experiments show that
our method consistently suggests viewpoints with superior
framing and composition compared to existing approaches,
establishing a new direction toward 3D-aware aesthetic
modeling.
1. Introduction
Framing and composition play a central role in image aes-
thetics. The same 3D scene can appear engaging or awk-
ward depending solely on the camera’s viewpoint, as spa-
tial relationships and perspective vary with viewing po-
sition.
This makes aesthetics inherently 3D-dependent.
When framing a shot, experienced photographers intu-
itively reason about spatial layout by observing from mul-
tiple angles and forming an internal sense of scene aes-
thetics to anticipate how visual appeal changes with view-
point [5].
This can be viewed as developing a mental
map of aesthetic variation across viewpoints.
Enabling
machines to reason in a similar way—inferring spatially-
varying scene aesthetics from a few observations to sug-
gest appealing viewpoints—would benefit not only personal
photography [11, 23] but also view selection/planning in
VR/AR [28, 37] and autonomous systems [1, 32].
Existing approaches primarily focus on single-view ad-
justments [11, 14, 23] by predicting limited camera move-
ments to refine a single image.
Although they can en-
hance framing locally, they lack awareness of the under-
lying scene geometry, leaving their reasoning confined to
a narrow neighborhood around the anchor view.
Recent
works attempt to incorporate broader viewpoint changes
into aesthetic modeling [25, 35] using generative models.
However, they rely on hallucinated content from a single
view and thus cannot ensure geometric consistency with
arXiv:2602.20363v1  [cs.CV]  23 Feb 2026

<!-- page 2 -->
real scenes. This highlights the need for a 3D-aware solu-
tion grounded in scene geometry for reliable aesthetic rea-
soning beyond observed views in 3D space.
In another line of work, 3D exploration approaches
explicitly operate in spatial environments,
employing
reinforcement-learning (RL) or genetic algorithms to search
for appealing viewpoints [1, 22, 30, 32]. However, they as-
sume access to dense, high-quality visual inputs from either
real or readily available virtual 3D environments (e.g. sim-
ulations, pretrained NeRFs [16]), which are costly to con-
struct and require extensive data collection. Moreover, their
RL-based solutions involve costly step-by-step exploration
in the environment. These limitations call for a solution that
alleviates the dependency on dense captures and enables ef-
ficient aesthetic reasoning without iterative physical adjust-
ments in real world.
To address these limitations, we propose to unify aes-
thetic perception with 3D geometry understanding by learn-
ing a 3D aesthetic field that enables efficient inference of
scene aesthetics from sparse observations. This field intrin-
sically encodes spatially varying aesthetic cues, supporting
geometry-grounded aesthetic reasoning in 3D. Much like
a skilled photographer who observes the scene from a few
angles and develops a mental map of where the best angles
lie, our system learns a 3D aesthetic field from sparse ob-
servations to reason about aesthetic variation across view-
points and find appealing camera poses, as shown in Fig. 1.
Specifically, we distill knowledge from a pretrained 2D aes-
thetic model [29] into a feedforward 3D Gaussian Splatting
network [34], predicting per-Gaussian aesthetic features di-
rectly from sparse input views. Rendering aesthetic features
from this field allows evaluation of aesthetic quality at novel
viewpoints. The learned aesthetic field transforms view-
point suggestion into a differentiable optimization problem,
which we address by an efficient two-stage search pipeline:
(1) we sample candidate viewpoints, and (2) locally refine
them through gradient-based updates. By unifying aesthetic
perception with geometric understanding, our method rea-
sons beyond the observed views to efficiently identify op-
timal viewpoints within the scene, without the overhead of
RL or dense captures for reconstruction. Extensive experi-
ments demonstrate the superiority of our framework in aes-
thetic viewpoint suggestion. Our main contributions are:
• We introduce the task of 3D-aware aesthetic view-
point suggestion with sparse observations, accounting for
3D-dependency in aesthetic modeling without requiring
dense captures.
• We propose a novel 3D aesthetic field that unifies 2D aes-
thetic perception with 3D geometric understanding, mod-
eling aesthetic variation across viewpoints.
• We develop an efficient two-stage search pipeline that
combines coarse sampling with gradient-based refine-
ment to efficiently discover appealing viewpoints.
• Extensive experiments on diverse datasets and input set-
tings demonstrate the effectiveness of our framework for
3D-aware aesthetic viewpoint suggestion.
2. Related Works
2.1. Single-view Adjustment Methods
Early aesthetic optimization works focus on image crop-
ping [6, 26, 29, 38] to reframe an existing photo to im-
prove visual appeal.
They predict cropping windows to
maximize the aesthetic score but function only as post-
processing once an image is captured. Later works extend
aesthetic adjustment to virtual camera motion, reusing crop-
ping datasets for learning camera adjustment that emulate
crop movements.
Su et al. [23] predict in-plane camera
shifts for interactive composition refinement, while Li et
al. [11] suggest camera rotations through an iterative feed-
back loop. Liu et al. [14] employ outpainting to suggest
cropping windows beyond the image border, but their model
assumes a fixed camera with enlarged field of view, lacking
true parallax changes. More recent efforts attempt to incor-
porate broader viewpoint variation. Uchida et al. [25] com-
bine outpainting with single-image 3D reconstruction to en-
large the search space, and Yao et al. [35] synthesize poor-
to-better perspectives via image-to-video generation. How-
ever, they rely on hallucinated content and cannot ensure ge-
ometric consistency with real scenes. Lacking awareness of
the underlying 3D structure, single-view adjustment meth-
ods are restricted to local composition refinement around
the original viewpoint. They struggle to discover more aes-
thetic angles that involve excluding or introducing scene
elements, which requires 3D reasoning beyond observed
views. In contrast, our method explicitly models spatially
varying scene aesthetics, supporting geometry-aware aes-
thetic reasoning in 3D.
2.2. 3D-exploration Methods
3D-exploration methods reason directly in 3D environ-
ments. They employ RL or search algorithms to actively ex-
plore candidate viewpoints in real or simulated scenes. Au-
toPhoto [1] uses RL to navigate through 3D to find aesthetic
views, while GAIT [32] trains an RL agent to generate in-
door tours in simulation. ViewActive [30] uses geometric
and semantic cues to optimize viewpoint selection for mesh
objects. Skartados et al. [22] adopt a genetic algorithm to
find aesthetic views in a pretrained NeRF scene. However,
they rely on dense captures or prebuilt 3D assets that are
costly to obtain, and their iterative RL procedures requires
physical adjustments in real world. In contrast, our frame-
work benefits from efficient inference of 3D scene aesthet-
ics from sparse observations, and virtually reasons within
the aesthetic field.

<!-- page 3 -->
2.3. Feature Distillation in 3D Gaussian Splatting
Recent advances in 3D Gaussian Splatting [3, 9, 34] have
enabled efficient, differentiable rendering and provided a
versatile representation for encoding scene geometry and
appearance. Building on this, several studies have explored
distilling 2D semantic features [12, 18, 24, 40] into 3D
Gaussian fields for segmentation tasks. These developments
suggest the potential of transferring 2D knowledge into 3D
representations. However, unlike semantic features that are
largely viewpoint-invariant, aesthetic information is inher-
ently viewpoint-dependent and remains largely unexplored
in this context. Our work extends this paradigm by distill-
ing aesthetic perception into 3D Gaussian fields to enable
viewpoint-aware aesthetic modeling.
3. Proposed Method
Problem Statement.
We aim to identify aesthetically
pleasing viewpoints for a given 3D scene observed from
sparse input views. We formulate this as finding the cam-
era pose P∗that maximizes the predicted aesthetic score
score(P) over the continuous view space:
P∗= arg max
P score(P).
(1)
Since direct optimization is intractable as it requires back-
and-forth 2D projection and evaluation, we propose to ap-
proximate this objective by learning a 3D aesthetic field,
which defines a continuous mapping from camera pose to
aesthetic quality while being grounded in scene geometry.
To learn this field, we distill aesthetic knowledge from a
pretrained 2D aesthetic model into a feedforward Gaussian
Splatting framework [3, 34] to reconstruct per-Gaussian
geometry and aesthetic features from sparse input views.
The distilled field enables rendering of aesthetic features at
novel viewpoints, which are evaluated by an aesthetic de-
coder to assess framing and composition. To efficiently ex-
plore the 3D scene for viewpoint suggestion, we introduce
a two-stage search pipeline: (1) coarse sampling of candi-
date viewpoints along and around the input trajectory fol-
lowed by aesthetic scoring to select top candidates, and (2)
gradient-based optimization for local refinement. The over-
all framework is illustrated in Fig. 2.
3.1. Distilling the 3D Aesthetic Field
Feedforward 3D Gaussian Splatting. We build our frame-
work upon a feedforward Gaussian Spatting model [34] that
predicts per-pixel Gaussian representation from sparse in-
put views in a single forward pass. Specifically, given in-
put views {Ii ∈RH×W ×3}N
i=1 and corresponding camera
poses {Pi ∈R3×4}N
i=1, a multi-view transformer extracts
multi-view features {Fi
mv}N
i=1. The features are then fused
across neighboring views using plane-sweep [4, 33] aggre-
gation and enriched with monocular depth cues to predict
per-pixel depth, which determines the center µ of each 3D
Gaussian. A DPT head [19] finally regresses the remain-
ing per-Gaussian parameters including covariance Σ, opac-
ity α, and color c. Novel view images can be rendered from
the 3D Gaussians via Gaussian Splatting [9]. This backbone
provides a differentiable and efficient 3D representation that
we extend to build the aesthetic field.
Direct RGB Scoring and Its Instability. Given the feed-
forward Gaussian Splatting backbone, a straightforward so-
lution for viewpoint suggestion is to render novel views
and directly score them using a pretrained aesthetic model.
However, this naive approach faces two key challenges.
First, aesthetic models are highly sensitive to small pixel
variations, producing fluctuating scores even for adjacent
views (see Fig. 3). This is because existing datasets lack an-
notations across nearby viewpoints in 3D, hence the models
haven’t seen such variations in training. Second, render-
ing artifacts at novel views can bias aesthetic predictions
and misguide optimization (see Fig. 3). To address them,
we shift from pixel-level scoring to feature-level reason-
ing by distilling aesthetic representations into the 3D Gaus-
sians. Operating in this latent feature space enhances ro-
bustness to low-level artifacts and enforces multi-view spa-
tial consistency, leading to smoother aesthetic variations
across nearby views. See Sec. 4.2 for further details.
Aesthetic Feature Distillation. We distill high-level aes-
thetic representations from a pretrained aesthetic teacher
model [29]. Specifically, we select an intermediate layer
as the distillation target and train our model to predict a
per-Gaussian aesthetic embedding faes, which can be de-
coded into aesthetic scores through the remaining layers of
the teacher after rasterization.
As shown in Fig. 2, on top of the backbone, we intro-
duce lightweight aesthetic modules: a CNN aesthetic en-
coder, an aesthetic DPT head [19], and a transformer down-
sampler. The aesthetic encoder is from the teacher’s fea-
ture extraction layers [29] and produces multi-scale aes-
thetic features {Fi
aes}N
i=1 from the input images. These fea-
tures are fused with multi-view features {Fi
mv}N
i=1 from the
backbone, and processed by the aesthetic DPT head [19]
to regress per-Gaussian aesthetic embeddings faes. Com-
bined with other Gaussian attributes (µ, Σ, α), these em-
beddings are rendered into aesthetic feature maps bFpred
at novel views through the same rasterization pipeline as
RGB rendering. To reduce storage and rasterization over-
head, each faes is a compact 32-dim vector instead of 512-
dim as the teacher feature. Since the teacher feature maps
Fgt are much smaller and deeper than bFpred, we employ
a lightweight transformer downsampler to align them and
obtain the final prediction Fpred.
Furthermore, we condition the model on camera poses at
both input and novel views, since aesthetic representations
are inherently viewpoint-dependent. Incorporating pose in-

<!-- page 4 -->
Figure 2. We distill aesthetic features into a feedforward Gaussian Splatting network (top). At inference, to search for aesthetic viewpoints,
we adopt a two-stage pipeline: coarse sampling to find good candidates (bottom left) and local refinement by gradient ascent (bottom right).
formation allows the model to capture these dependencies.
Training Objective.
The backbone multi-view trans-
former, DPT head, and aesthetic encoder are kept frozen to
preserve consistent geometry predictions and 2D aesthetic
perception, while the additional modules are trained end-to-
end. We use mean-squared error (MSE) loss on the rendered
feature maps Fpred at hold-out views, where ground-truth
Fgt are obtained by feeding ground-truth images into the
pretrained aesthetic model [29].
3.2. Viewpoint Search Pipeline
The learned 3D aesthetic field provides a continuous and
differentiable mapping from camera pose to aesthetic score,
enabling efficient search and optimization in viewpoint
space. To efficiently explore this space, we adopt a two-
stage coarse-to-fine search pipeline. In the first stage, can-
didate viewpoints are heuristically sampled around the in-
put trajectories and scored to identify high-quality candi-
dates. In the second stage, we perform gradient ascent on
the selected candidates to locally refine their poses. The
distilled aesthetic field yields a smooth score landscape (see
Fig. 3 (a)), allowing stable gradient-based optimization (see
Sec. 4.4). The final refined viewpoints are returned as aes-
thetic suggestions. The entire process is illustrated in the
bottom of Fig. 2. We now describe each stage in detail.
Stage 1: Coarse Sampling. The goal of this stage is to effi-
ciently identify promising viewpoint candidates for further
refinement. We first connect the input views into a con-
tinuous camera trajectory by interpolating both camera po-
sitions and orientations, yielding a smooth path that cov-
ers the major observed viewpoints of the scene. Candidate
viewpoints are then linearly sampled along this trajectory.
For each sample, we further generate a local neighborhood
of perturbed cameras with small in-plane shifts and direc-
tional jitters, allowing local exploration around the trajec-
tory while maintaining scene focus. Each candidate view-
point is rendered through the aesthetic field to obtain its
aesthetic features, which are evaluated by the aesthetic de-
coder to produce a score. Finally, we select the top-K high-
scoring candidates {Pk
cand}K
k=1 for the subsequent refine-
ment stage. To enhance diversity, near-identical candidates
are filtered out via a distance-based duplication check, en-
suring that the selected set covers distinct viewpoints.
Stage 2: Gradient-based Refinement. Starting from the
candidate viewpoints obtained in Stage 1, we perform local
optimization directly on camera poses to further maximize
their aesthetic scores. Specifically, each camera pose Pk
cand
is parameterized by 3D translation and orientation, and up-
dated through gradient ascent on the aesthetic score:
Pt+1 = Pt + η∇Pscore(Pt),
(2)
where η is the step size. In practice, we optimize a 5-dim
vector comprising 3D translation and two rotation variables
(yaw and pitch), as roll is rarely adjusted in typical scene
captures. We use a fixed number of iterations for each can-

<!-- page 5 -->
didate and retain the top-scoring refined viewpoints as the
final suggestions.
4. Experiments
4.1. Experiment Setup
Datasets.
We use RealEstate10k (RE10k) [41] and
DL3DV [13] for training and evaluation. RE10k mainly
contains indoor videos, while DL3DV involves more di-
verse scenes. Both have camera parameters at each frame.
More details can be found in the Supplementary Material.
Implementation Details. We implement our framework
in Pytorch [17], using DepthSplat [34] as the feedfor-
ward Gaussian Splatting backbone. We use the VEN [29]
model as the teacher for distilling aesthetic representations.
VEN is a CNN model, and we use features from the 23rd
layer(14 × 14 × 512) as the distillation target. Following
Xu et al. [34], we use image resolutions of 256 × 256 for
RE10k and 256 × 448 for DL3DV. During training, we ran-
domly sample 2 input views for RE10k and 2 to 6 input
views for DL3DV, following the same protocol as Xu et
al. [34]. More implementation and training details can be
found in the Supplementary Material.
Search Pipeline Configurations. At inference, we search
for optimal views via a two-stage approach. In Stage 1,
within each segment of the interpolated trajectory, we uni-
formly sample 16 camera poses, and around each we further
sample 8 neighboring poses. We take top-2 candidates as
input to the next stage. In Stage 2, we use the Adam [10]
optimizer with step size 0.01 and update for 25 steps.
4.2. Aesthetic Prediction at Novel Views
We first evaluate our model’s ability to predict aesthetic
quality at unseen viewpoints to validate the learned 3D
aesthetic field before applying it to viewpoint suggestion.
Specifically, we compare our predicted aesthetic scores with
ground-truth scores from the teacher aesthetic model [29],
and benchmark against the RGB-scoring baseline. For each
scene in the RE10k and DL3DV test splits, we use 2, 4,
and 6 input views to build the aesthetic field and predict
aesthetic scores for the remaining frames, using the entire
video clips (about 250 frames each) for RE10k and 100
frames for DL3DV. Following Xu et al. [34], we train at
256×256 in RE10k and 256×448 in DL3DV, and evaluate
under both resolutions to assess generalization.
Correlation to Teacher Scores. We compute the correla-
tion between the predicted and teacher aesthetic scores at
novel views using two standard metrics: the Pearson lin-
ear correlation coefficient (PLCC) and the Spearman rank-
order correlation coefficient (SRCC), which are computed
for each scene and then averaged.
As shown in Tab. 1,
across all settings, our aesthetic field achieves significantly
higher correlation with the teacher than the RGB-scoring
Methods
#Views
RE10k [41]
DL3DV [13]
PLCC
SRCC
PLCC
SRCC
Baseline
2
0.657
0.628
0.326
0.307
Ours
0.780
0.740
0.509
0.477
Baseline
4
0.657
0.633
0.513
0.481
Ours
0.796
0.758
0.722
0.682
Baseline
6
0.745
0.701
0.580
0.553
Ours
0.836
0.794
0.753
0.719
(a) Image resolution 256 × 256.
Methods
#Views
RE10k [41]
DL3DV [13]
PLCC
SRCC
PLCC
SRCC
Baseline
2
0.634
0.611
0.299
0.286
Ours
0.733
0.704
0.479
0.445
Baseline
4
0.596
0.578
0.552
0.535
Ours
0.753
0.722
0.700
0.668
Baseline
6
0.711
0.656
0.646
0.625
Ours
0.821
0.764
0.737
0.707
(b) Image resolution 256 × 448.
Table 1. Novel view aesthetic score correlation between predicted
and ground-truth scores.
c
b
(a) Aesthetic score prediction.
(c) Nearby GT frames with fluctuation aesthetic scores.
(b) GT & predicted frames.
Score: 0.23
Score: -1.12
Figure 3. (a) Aesthetic score predictions over consecutive frames.
Our method produces scores closer to the ground truth while being
smoother and more consistent across nearby views. The dashed
line and box mark the regions visualized in (b) and (c), respec-
tively. (b) Given the same aesthetic model and viewpoint, render-
ing artifacts in the predicted view (bottom) bias the RGB-scoring
approach toward lower scores. (c) Ground truth scores fluctuate
noticeably across nearly identical nearby views.
baseline, indicating that the distilled field produces more
stable and faithful aesthetic predictions. As discussed ear-

<!-- page 6 -->
RE10k [41]
Methods
2 Input Views
4 Input Views
6 Input Views
VEN↑
SAMPNet↑
VEN↑
SAMPNet↑
VEN↑
SAMPNet↑
Baseline
1.48
2.29
1.79
2.26
2.01
2.29
In-plane Shift∗
1.52
2.31
1.81
2.36
2.10
2.41
Rotation∗
1.78
2.38
1.95
2.42
2.13
2.45
UNIC [14]†
1.15
2.17
1.61
2.33
1.82
2.35
Uchida et al. [25]†
1.58
2.32
1.89
2.37
2.13
2.42
Ours
1.89
2.40
2.03
2.45
2.20
2.49
DL3DV [13]
Baseline
2.08
2.23
2.31
2.26
2.47
2.30
In-plane Shift∗
2.16
2.28
2.45
2.29
2.59
2.33
Rotation∗
2.52
2.30
2.67
2.32
2.85
2.37
UNIC [14]†
2.07
2.21
2.35
2.27
2.48
2.30
Uchida et al. [25]†
2.34
2.25
2.68
2.32
2.81
2.36
Ours
2.56
2.33
2.76
2.36
2.91
2.41
Table 2. Aesthetic viewpoint suggestion evaluation on different datasets and with different numbers of input views.
† indicates open-
sourced methods adapted to our setting. ∗denotes approximations of non–open-sourced single-view methods in our setting. Note that such
approximations may overestimate the actual results that those methods could deliver.
lier, direct RGB scoring is highly sensitive to rendering arti-
facts. Moreover, teacher scores themselves fluctuate across
nearby ground-truth views with almost identical composi-
tions, so these quantitative results are best interpreted as
relative indicators of stability and fidelity rather than abso-
lute measures of accuracy. To better illustrate these effects
and the advantages of our feature-distilled field, we present
qualitative comparisons below.
Qualitative Comparisons. Figure 3 shows representative
examples to illustrate the problems of RGB-scoring base-
line and the superiority of our distilled aesthetic field. First,
rendering artifacts in predicted novel-view images (e.g.
noise, blurriness) can mislead the aesthetic model (Fig. 3
(b)). Second, ground-truth views with nearly identical con-
tent often yield fluctuating teacher scores (Fig. 3 (c)), re-
vealing the inherent sensitivity of aesthetic models to pixel-
level variations. As shown in Fig. 3 (a), these factors cause
large oscillations in the RGB-scoring predictions (orange),
whereas our aesthetic field (green) produces smoother and
more consistent scores that closely follow the teacher trend
(blue). Importantly, this stability is not achieved by explicit
score smoothing, which would require arbitrary choices of
window size or smoothness strength and lacks a princi-
pled basis without human ground-truth. Instead, our model
implicitly enforces smoothness through feature distillation,
where learning in the aesthetic feature space naturally reg-
ularizes the score landscape across neighboring viewpoints.
These qualitative and quantitative results confirm that our
distilled aesthetic field enables stable and consistent pre-
diction of viewpoint-dependent aesthetics. More examples
are provided in the Supplementary Material. The success
in novel view aesthetic prediction forms the foundation for
aesthetic viewpoint suggestion, which we evaluate next.
4.3. Aesthetic Viewpoint Suggestion
To evaluate our method on aesthetic viewpoint suggestion
given sparse input views, we vary the number of input views
between [2, 4, 6] and assess the aesthetic quality of the sug-
gested viewpoints.
Since no existing benchmarks account for viewpoint-
dependent aesthetics, we adopt the test splits of RE10k and
DL3DV to provide input views and use two aesthetic mod-
els (VEN [29] and SAMPNet [39]) to quantify framing and
composition quality. Because the suggested views may not
have corresponding ground-truth images, we reconstruct the
scenes with dense inputs to obtain pseudo ground-truths for
evaluation only, ensuring a fair and artifact-mitigated com-
parison without affecting the viewpoint suggestion process
itself. As no prior work operates in this sparse-view setting,
we compare against the baseline and single-view methods
adapted to our evaluation protocol as detailed below.
Comparison Approaches. We first compare our method
with the direct RGB-scoring baseline, which shares the
same search pipeline as ours.
To adapt single-view ad-
justment methods [11, 14, 21, 23, 25, 35] comparable in
our setting, we treat each sparse input view as an anchor,
around which we obtain a candidate based on their sug-
gestions. This yields N suggestions for the N input an-
chors per scene, where we take the best as the final out-
put.
Among single-view methods, only UNIC [14] and
Uchida et al. [25] are publicly available.
For fair com-
parison, we adapt both by fixing their crop sizes as input
resolution and map their outpainted suggestions to corre-
sponding real viewpoints. See Supplementary Material for

<!-- page 7 -->
Figure 4. Aesthetic viewpoint suggestion examples with 4 input views in RE10k (top 2 rows) and DL3DV (bottom 3 rows). Red boxes in
the last row show that single-view methods fail to remove distracting objects due to limited adjustment range. Zoom in for more details.
additional details. For other methods that predict minor in-
plane shift [21, 23] or rotation [11] around the anchor view-
point, we approximate their behavior by exhaustively sam-
pling and scoring nearby views with the aesthetic model.
Yao et al. [35] cannot be compared as their video genera-
tion model is not available. 3D-exploration methods are not
comparable as they need dense captures of the scene.
Quantitative Results. Table 2 reports the aesthetic view-
point suggestion results. Our method consistently recom-
mends views with higher aesthetic scores than all compar-
ison approaches in different metrics. It also maintains su-
perior performance across different input numbers, show-
ing robustness under varying observation sparsity.
Even
with only two views, our model produces significantly bet-
ter suggestions, showing its strong ability to reason about
the underlying 3D scene aesthetics from minimal observa-
tions. As the number of input views increases, the suggested
viewpoints achieve progressively higher scores, indicating
that our model effectively exploits broader scene coverage
to discover more aesthetically optimal perspectives. With
more input views, the performance gap over single-view-
based methods (e.g. rotation) narrows, likely because they
can also explore sufficient viewpoint variations within their
local neighborhoods. However, note that inplane-shift and
rotation serve as an upper bound for approximating single-
view methods in our setting, as we directly maximize target
scores rather than relying on learned guidance.
Qualitative Comparisons.
Figure 4 presents qualitative
examples of the suggested viewpoints.
Our framework
identifies visually balanced and well-composed views that
align with human aesthetic preferences. In contrast, single-
view adjustment methods are restricted to small neighbor-
hood around the anchor views and often fail to capture op-
timal perspectives. For example, they may break the struc-
tural continuity of objects (e.g. bed frames in the first row)
or struggle to remove distracting objects (e.g. marked by red
boxes in the last row) from the view while our method can
freely place the camera to enhance composition.
3D Visualization of Viewpoint Suggestions. Finally, we
visualize the viewpoint sampling strategy in Fig. 5. Can-
didate viewpoints are sampled along and around the trajec-
tory of input views, with colors indicating the predicted aes-
thetic scores. The results reveal how aesthetic quality varies
across 3D space, and the corresponding renderings confirm
strong alignment with human perceptual preferences.
4.4. Aesthetic Field Gradient Optimization Analysis
We analyze the learned aesthetic field through the lens of
gradient-based optimization, and show that the distilled
field enables stable and effective gradient ascent compared
with the RGB-scoring baseline. We randomly select starting
viewpoints and perform gradient ascent on both methods.

<!-- page 8 -->
Score: 1.168
Score: 2.598
Score: 2.259
Score: 1.052
Figure 5. Visualization of sampled viewpoints colored by aesthetic
score, with representative renderings shown alongside.
Methods
RE10k [41]
DL3DV [13]
∆VEN↑
∆VEN↑
Baseline
0.20
0.18
Ours
0.46
0.43
Table 3. Aesthetic score improvement with gradient ascent.
Initial
Step 7
Step 14
Step 21
Ours
Baseline
Initial
Step 7
Step 14
Step 21
Figure 6. Gradient ascent results. Zoom in for a closer look.
For each scene in the test split, both approaches receive the
same sparse input views, and the same random novel view is
chosen as the initial camera pose for both. We run optimiza-
tion for 25 steps and report the average score improvement
∆VEN in Tab. 3. Our method achieves consistent score
increases in different datasets, confirming that the learned
aesthetic field provides a well-behaved optimization land-
scape. In contrast, the RGB-scoring baseline suffers from
unstable updates that often degrade the results. Figure 6
further illustrate this behavior. Our method converges to-
ward balanced and aesthetically pleasing viewpoints, while
RGB-based refinement fails to update reliably toward better
views. See Supplementary Material for more examples.
4.5. Ablations
We conduct ablation studies to validate key architectural
and configuration choices in our framework.
View-conditioning.
We first examine the role of view-
conditioning in aesthetic prediction for novel viewpoints.
Tab. 4 shows that view-conditioning significantly improves
aesthetic prediction accuracy at unseen views. This con-
firms that explicitly modeling viewpoint dependence is cru-
cial for capturing aesthetic cues across diverse perspectives.
Methods
RE10k [41]
DL3DV [13]
PLCC
SRCC
PLCC
SRCC
w/o. view-cond.
0.732
0.695
0.658
0.625
w. view-cond.
0.796
0.758
0.700
0.668
Table 4. Ablation of view-conditioning on novel view aesthetic
prediction with 4 input views.
# Candidates
RE10k [41]
DL3DV [13]
VEN↑
VEN↑
1
1.96
2.57
2
2.03
2.76
3
2.05
2.78
# Steps
∆VEN↑
∆VEN↑
15
0.21
0.15
20
0.32
0.28
25
0.46
0.43
30
0.49
0.45
Table 5. Ablation of candidate numbers and refinement steps.
Search Configurations.
We analyze the number K of
candidates returned in coarse sampling and gradient ascent
steps in refinement. As shown in Tab. 5, performance gains
saturate around K = 2 and 25 refinement steps, which are
therefore adopted as default in all experiments. Additional
ablations are provided in the Supplementary Material.
5. Discussion & Conclusion
We present a novel framework for 3D-aware aesthetic view-
point suggestion that learns a 3D aesthetic field via feedfor-
ward 3D Gaussian Splatting, enabling geometry-grounded
aesthetic reasoning and efficient aesthetic viewpoint discov-
ery from sparse input views. Despite its effectiveness, our
framework still has room for improvement. First, it relies
on camera poses to build the aesthetic field. While such
information can be obtained from COLMAP [20] and is of-
ten available in robots/drones and smartphones [7], a pose-
free variant would broaden the applicability of our method.
This could be achieved by incorporating recent advances
in pose-free methods [8, 36]. Second, the quality of the
aesthetic field depends on the accuracy of the reconstructed
geometry, which is affected by both the backbone’s ability
and the input coverage. While the former can be improved
by stronger geometry backbones [27], the latter can be ad-
dressed by selecting views to ensure sufficient scene cover-
age [2, 31]. Third, our viewpoint search is constrained to
regions supported by the initial observations. A promising
extension is an active-perception loop that acquires addi-
tional views in promising directions to expand the aesthetic
field and enlarge the feasible search space.

<!-- page 9 -->
Aesthetic Camera Viewpoint Suggestion with 3D Aesthetic Field
Supplementary Material
S1. Datasets & Implementation Details
We conduct experiments on subsets of the RE10k [41] and
DL3DV [13] datasets. For RE10k, we use approximately
6, 000 scene clips as the training split and 749 clips as the
test split. For DL3DV, we adopt its 1K-5K subsets for train-
ing, which contain roughly 5, 000 scenes, and test on the
140 benchmark scenes.
During training, we sample 2 input views to reconstruct
the aesthetic features of 12 target views in RE10k, and sam-
ple 2 to 6 input views to predict aesthetic features for 24 tar-
get views in DL3DV. We train all models for 50,000 steps
with a batch size of 8. We use AdamW [15] with a learning
rate of 0.0005 and a cosine annealing scheduler.
S2. Aesthetic Prediction at Novel Views
We provide additional qualitative examples of novel-view
aesthetic prediction to further illustrate the two limitations
of the RGB-based baseline approach discussed in the main
paper. As shown in Fig. S1, the aesthetic model is biased by
the rendering artifact, and exhibits high sensitivity to minor
pixel-level variations across nearby views that are nearly
identical. In contrast, our distilled aesthetic field faithfully
follows the teacher’s underlying trend, producing smoother
and more consistent predictions across nearby frames.
S3. Aesthetic Viewpoint Suggestion
We first describe how existing single-view baselines are
adapted to our evaluation setting, followed by additional
qualitative comparisons of the suggested viewpoints.
UNIC [33] predicts crops with varying sizes in an ex-
trapolated image plane rather than real camera motions.
For fair comparison, we fix its crop size as input resolu-
tion (256 × 256 in RE10k and 256 × 448 in DL3DV) and
map the crop-center shifts to in-plane camera translations,
thereby obtaining corresponding real viewpoints where we
measure aesthetic qualities.
Uchida et al. [25] first out-
paint an image, reconstruct 3D point clouds from it, and
then optimize camera poses and image aspect ratios in this
outpainted 3D scene. We therefore also fix their output res-
olution and transform their suggestions to equivalent real
viewpoints for evaluation.
As shown in Fig. S2, our method consistently suggests
viewpoints with superior framing and composition than
comparison methods.
Finally, additional 3D visualizations of the viewpoint
sampling results are shown in Fig. S3. The results demon-
strate the variation of aesthetic quality across 3D space, and
(a) Aesthetic score prediction.
(c) Nearby GT frames with fluctuation aesthetic scores.
(b) GT & predicted frames.
Score: 0.68
Score: -0.20
b
c
Sequence 1.
Score: 1.41
Score: 0.36
b
c
(a) Aesthetic score prediction.
(c) Nearby GT frames with fluctuation aesthetic scores.
(b) GT & predicted frames.
Sequence 2.
Figure S1. Additional novel view aesthetic prediction results. In
both examples: (a) Aesthetic score predictions over consecutive
frames. The dashed line and box mark the regions visualized in
(b) and (c), respectively. (b) Given the same aesthetic model and
viewpoint, rendering artifacts in the predicted view (e.g., wrong
color prediction as marked in red boxes in top, noisy predictions
marked in blue boxes in bottom) bias the RGB-scoring approach.
(c) Ground-truth scores fluctuate noticeably across nearly identical
nearby views.
the corresponding renderings confirm strong alignment be-

<!-- page 10 -->
Figure S2. Additional results of aesthetic viewpoint suggestion.
tween our model’s evaluations and human perceptual pref-
erences.
S4. Aesthetic Field Gradient Optimization
We present additional results comparing gradient ascent op-
timization using the baseline approach and our method. As
shown in Fig. S4, our method consistently converges toward
more aesthetically pleasing viewpoints, whereas the base-
line approach often fails to make meaningful progress and
can sometimes degenerate.
S5. Ablations
We provide additional ablation experiments of the Stage 1
search configuration, focusing on two parameters: the num-
ber of samples along each input-view segment (S) and the
number of neighbors further drawn around each segment
sample (N). The total number of candidate viewpoints is
approximately proportional to S × N. We measure the aes-
thetic qualities of final suggestions via VEN [29]. As shown
in Tab. S1, our method remains stable across a wide range
S = 4
S = 8
S = 16
S = 32
N = 4
1.98
2.00
1.98
1.98
N = 8
1.98
1.94
2.03
2.02
N = 16
2.01
2.00
2.02
2.04
N = 32
2.02
2.03
2.04
2.06
Table S1. Ablation of the sampling configuration of search stage
1 on novel view aesthetic prediction. S denotes the number of
samples along each segment of the interpolated input-view trajec-
tory, and N denotes the number of neighboring viewpoints sam-
pled around each segment-sample. All experiments use 4 input
views on RE10k [41], and the aesthetic quality is measured by
VEN [29].
of sampling densities, while generally achieving better per-
formance with more samples. We adopt S = 16 and N = 8
as our default configuration, as this setting offers a good bal-
ance between suggestion quality and total number of sam-
ples being evaluated.

<!-- page 11 -->
References
[1] Hadi AlZayer, Hubert Lin, and Kavita Bala.
Autophoto:
Aesthetic photo capture using reinforcement learning.
In
2021 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pages 944–951. IEEE, 2021. 1,
2
[2] Xiao Chen, Quanyi Li, Tai Wang, Tianfan Xue, and Jiang-
miao Pang. Gennbv: Generalizable next-best-view policy for
active 3d reconstruction. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 16436–16445, 2024. 8
[3] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In European Conference on Computer
Vision, pages 370–386. Springer, 2024. 3
[4] Robert T Collins. A space-sweep approach to true multi-
image matching. In Proceedings CVPR IEEE computer so-
ciety conference on computer vision and pattern recognition,
pages 358–363. Ieee, 1996. 3
[5] Michael Freeman. The Photographer’s Eye Digitally Remas-
tered 10th Anniversary Edition: Composition and Design for
Better Digital Photos. Routledge, 2017. 1
[6] Chaoyi Hong, Shuaiyuan Du, Ke Xian, Hao Lu, Zhiguo Cao,
and Weicai Zhong. Composing photos like a photographer.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 7057–7066, 2021. 2
[7] Apple Inc. Apple arkit. https://developer.apple.
com/augmented-reality/arkit/, 2025. Accessed:
2025-11-13. 8
[8] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren,
Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng
Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting
from unconstrained views. arXiv preprint arXiv:2505.23716,
2025. 8
[9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 3
[10] Diederik P Kingma. Adam: A method for stochastic opti-
mization. arXiv preprint arXiv:1412.6980, 2014. 5
[11] Jiawan Li, Fei Zhou, Zhipeng Zhong, Jiongzhi Lin, and
Guoping Qiu. Towards smart point-and-shoot photography.
In Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 28242–28251, 2025. 1, 2, 6, 7
[12] Qijing Li, Jingxiang Sun, Liang An, Zhaoqi Su, Hongwen
Zhang, and Yebin Liu.
Semanticsplat: Feed-forward 3d
scene understanding with language-aware gaussian fields.
arXiv preprint arXiv:2506.09565, 2025. 3
[13] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,
Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,
et al.
Dl3dv-10k: A large-scale scene dataset for deep
learning-based 3d vision. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 22160–22169, 2024. 5, 6, 8, 1
[14] Xiaoyu Liu, Ming Liu, Junyi Li, Shuai Liu, Xiaotao Wang,
Lei Lei, and Wangmeng Zuo. Beyond image borders: Learn-
ing feature extrapolation for unbounded image composition.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 13023–13032, 2023. 1, 2, 6
[15] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. arXiv preprint arXiv:1711.05101, 2017. 1
[16] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[17] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An im-
perative style, high-performance deep learning library. Ad-
vances in neural information processing systems, 32, 2019.
5
[18] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
Hanspeter Pfister. Langsplat: 3d language gaussian splatting.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 20051–20060, 2024.
3
[19] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 12179–12188, 2021. 3
[20] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 8
[21] Nan Sheng, Yongzhen Ke, Shuai Yang, Yong Yang, and Lim-
ing Chen. View adjustment: helping users improve photo-
graphic composition. Multimedia Systems, 30(5):293, 2024.
6, 7
[22] Evangelos Skartados, Mehmet Kerim Yucel, Bruno Man-
ganelli, Anastasios Drosou, and Albert Sa`a-Garriga. Finding
waldo: Towards efficient exploration of nerf scene spaces.
In Proceedings of the 15th ACM Multimedia Systems Con-
ference, pages 155–165, 2024. 2
[23] Yu-Chuan Su, Raviteja Vemulapalli, Ben Weiss, Chun-Te
Chu, Philip Andrew Mansfield, Lior Shapira, and Colvin
Pitts. Camera view adjustment prediction for improving im-
age composition. arXiv preprint arXiv:2104.07608, 2021. 1,
2, 6, 7
[24] Qijian Tian, Xin Tan, Jingyu Gong, Yuan Xie, and Lizhuang
Ma. Uniforward: Unified 3d scene and semantic field re-
construction via feed-forward gaussian splatting from only
sparse-view images. arXiv preprint arXiv:2506.09378, 2025.
3
[25] Taichi Uchida, Yoshihiro Kanamori, and Yuki Endo. 3d view
optimization for improving image aesthetics.
In ICASSP
2025-2025 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pages 1–5. IEEE,
2025. 1, 2, 6
[26] Chao Wang, Li Niu, Bo Zhang, and Liqing Zhang. Image
cropping with spatial-aware feature and rank consistency. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 10052–10061, 2023. 2

<!-- page 12 -->
[27] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 8
[28] Zhan Wang, Qian Zhu, David Yip, Fugee Tsung, and Wei
Zeng. Cinefolio: Cinematography-guided camera planning
for immersive narrative visualization.
Visual Informatics,
page 100259, 2025. 1
[29] Zijun Wei, Jianming Zhang, Xiaohui Shen, Zhe Lin,
Radomir Mech, Minh Hoai, and Dimitris Samaras. Good
view hunting: Learning photo composition from dense view
pairs. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 5437–5446, 2018. 2,
3, 4, 5, 6
[30] Jiayi Wu, Xiaomin Lin, Botao He, Cornelia Fermuller, and
Yiannis Aloimonos. Viewactive: Active viewpoint optimiza-
tion from a single image. arXiv preprint arXiv:2409.09997,
2024. 2
[31] Wenhui
Xiao,
Rodrigo
Santa
Cruz,
David
Ahmedt-
Aristizabal, Olivier Salvado, Clinton Fookes, and Leo Le-
brat. Nerf director: Revisiting view selection in neural vol-
ume rendering. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20742–
20751, 2024. 8
[32] Desai Xie, Ping Hu, Xin Sun, Soren Pirk, Jianming Zhang,
Radom´ır Mech, and Arie E Kaufman.
Gait: Generating
aesthetic indoor tours with deep reinforcement learning. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 7409–7419, 2023. 1, 2
[33] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi,
Fisher Yu, Dacheng Tao, and Andreas Geiger.
Unifying
flow, stereo and depth estimation.
IEEE Transactions on
Pattern Analysis and Machine Intelligence, 45(11):13941–
13958, 2023. 3, 1
[34] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth.
In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 16453–16463, 2025. 2, 3, 5
[35] Lujian Yao, Siming Zheng, Xinbin Yuan, Zhuoxuan Cai, Pu
Wu, Jinwei Chen, Bo Li, and Peng-Tao Jiang. Photography
perspective composition: Towards aesthetic perspective rec-
ommendation. arXiv preprint arXiv:2505.20655, 2025. 1, 2,
6, 7
[36] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys,
Ming-Hsuan Yang, and Songyou Peng. No pose, no problem:
Surprisingly simple 3d gaussian splats from sparse unposed
images. arXiv preprint arXiv:2410.24207, 2024. 8
[37] Faisal Zaman, Craig Anslow, and Taehyun James Rhee. Vi-
carious: Context-aware viewpoints selection for mixed re-
ality collaboration. In Proceedings of the 29th ACM Sym-
posium on Virtual Reality Software and Technology, pages
1–11, 2023. 1
[38] Hui Zeng, Lida Li, Zisheng Cao, and Lei Zhang. Reliable
and efficient image cropping: A grid anchor based approach.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 5949–5957, 2019. 2
[39] Bo Zhang, Li Niu, and Liqing Zhang. Image composition
assessment with saliency-augmented multi-pattern pooling.
arXiv preprint arXiv:2104.03133, 2021. 6
[40] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3dgs: Supercharging
3d gaussian splatting to enable distilled feature fields. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21676–21685, 2024. 3
[41] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely.
Stereo magnification:
Learning
view synthesis using multiplane images.
arXiv preprint
arXiv:1805.09817, 2018. 5, 6, 8, 1, 2

<!-- page 13 -->
Score: 1.708
Score: 1.612
Score: -0.501
Score: 0.807
Score: 2.264
Score: 1.040
Score: 0.462
Score: 0.270
Figure S3. Additional visualizations of sampled viewpoints colored by aesthetic score, with representative renderings shown alongside.

<!-- page 14 -->
Figure S4. Additional gradient ascent results. Zoom in for a closer look.
