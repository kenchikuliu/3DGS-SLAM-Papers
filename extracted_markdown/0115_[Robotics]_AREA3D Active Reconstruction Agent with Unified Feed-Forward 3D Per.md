<!-- page 1 -->
AREA3D: Active Reconstruction Agent with Unified Feed-Forward 3D
Perception and Vision-Language Guidance
Tianling Xu1,2
Shengzhe Gan1
Leslie Gu2
Yuelei Li3
Fangneng Zhan4
Hanspeter Pfister2
1Southern University of Science and Technology
2Harvard University
3California Institute of Technology
4Massachusetts Institute of Technology
High-level 
Guidance from 
VLM
Feed Forward 
Perception
Active View Selection Strategy
1. Selected 
2. Selected 
1. Observed
2. Observed
Initial 
Observation
Simulated Scenario
AREA3D
Active 
Observation
Figure 1. Overview of our approach. We propose AREA3D, an active reconstruction agent, which unifies two complementary signals
of feed-forward 3D perception and vision-language guidance to decide the next best views under tight view budgets. AREA3D efficiently
reconstructs high-fidelity geometry from sparse observations by actively choosing the most informative viewpoints.
Abstract
Active 3D reconstruction enables an agent to autonomously
select viewpoints to build accurate and complete scene
geometry efficiently, rather than passively reconstructing
scenes from pre-collected images. Existing active recon-
struction methods often rely on geometric heuristics, which
may result in redundant observations without improving re-
construction quality. To address this, we propose AREA3D,
an active reconstruction agent for 3D reconstruction by
leveraging feed-forward 3D models and vision-language
guidance. The framework decouples view uncertainty mod-
eling from feed-forward reconstruction, enabling precise
uncertainty estimation without online optimization. More-
over, the integrated Vision-Language Model provides high-
level semantic guidance that guides exploration beyond
purely geometric cues.
Extensive experiments on both
scene-level and object-level benchmarks demonstrate that
AREA3D achieves state-of-the-art reconstruction accuracy,
especially in sparse views. The code will be available at
https://github.com/TianlingXu/AREA3D
1. Introduction
Active and autonomous exploration constitutes a corner-
stone of intelligent embodied agents, enabling them to per-
ceive, interpret, and interact with complex environments.
Within this broader context, active reconstruction represents
a critical subtask that demands the integration of 3D percep-
tion, viewpoint selection, and embodied execution.
Traditional active reconstruction methods [8, 12, 18, 38]
rely on handcrafted criteria, such as surface coverage, voxel
occupancy, or view overlap heuristics, to guide view selec-
tion. However, these handcrafted metrics provide no direct
awareness of under-reconstructed or missing regions.
As a result, they often overestimate areas that are al-
ready well-covered but poorly reconstructed, while failing
to identify holes or unseen surfaces caused by occlusion or
limited viewpoints. This mismatch leads to redundant ob-
servations and incomplete scene geometry. To overcome
1
arXiv:2512.05131v1  [cs.CV]  28 Nov 2025

<!-- page 2 -->
this, recent work [7, 12, 15, 20, 26] uses high-fidelity neu-
ral representations, such as 3D Gaussian Splatting [17] and
Neural Radiance Field [22], to model uncertainty or in-
formation gain. However, without robust data-driven pri-
ors, their performance degrades under sparse observations,
while the computational cost becomes prohibitive when
dense observations are available. Consequently, current ap-
proaches must carefully balance reconstruction quality and
efficiency, limiting their scalability and generalization to
real-world scenes.
We propose AREA3D, an active reconstruction agent,
which unifies two complementary signals of feed-forward
3D perception and vision-language guidance to decide the
next best views under tight view budgets.
As shown in
Figure 1, our framework features a dual-field uncertainty
modeling mechanism. Unlike per-scene optimized NeRF
or 3DGS, a data-driven feed-forward 3D model can directly
provide geometric confidence to build a 3D geometric field
about what is already well perceived, for both sparse views
and dense views.
On the other hand, a vision–language
model reasons about what is likely missing, highlighting
occluded or unseen content. We combine these cues into a
single, visibility-aware “where-to-look” field that guides the
agent to a small number of high-value views, progressively
closing coverage gaps. This design is lightweight, budget-
aware, and can handle both object-centric tabletop scenes
and room-scale scenes, yielding precise geometry with ac-
tive exploration.
In general, we conclude our three contributions as fol-
lows:
• We propose AREA3D, an agent for active 3D recon-
struction that unifies feed-forward 3D modeling and vi-
sion–language reasoning into a dual-field framework for
uncertainty-aware perception and view planning.
• We construct a unified active reconstruction benchmark
covering both object-centric and scene-level regimes, en-
abling consistent evaluation across scales and demon-
strating strong robustness and generalization.
• Extensive experiments demonstrate state-of-the-art accu-
racy and efficiency under tight budgets, with ablations
confirming the complementary value of feed-forward 3D
perception and high-level guidance from Vision Lan-
guage Model.
2. Related work
2.1. Active Reconstruction
Active reconstruction is a critical problem that lies at the in-
tersection of active perception, 3D vision, and robotics. Tra-
ditional methods often formulate it as an active view plan-
ning task, where the agent selects informative viewpoints
to minimize reconstruction uncertainty or maximize surface
coverage. Table 1 provides an overview of the uncertainty
proxies employed by existing active reconstruction meth-
ods.
Early Active View Selection.
Early approaches select
observation viewpoints based on handcrafted geometric
heuristics, such as voxel occupancy[5], view overlap, or
frontier-based exploration VLFM [10, 24, 39, 40] . While
effective in simple scenes, these metrics often rely on
known geometry or occupancy grids and cannot capture
complex scene ambiguity or perceptual uncertainty, fre-
quently leading to redundant observations.
Learning-Based & Neural-Field–Driven Methods. More
recent learning-based methods employ policy networks or
reinforcement learning [4, 23, 38] to predict the next best
view (NBV) from image features or partial reconstructions.
A significant body of work has focused on leveraging the
internal state of neural radiance fields (NeRF) [22] to guide
exploration. For example, some methods utilize uncertainty
derived directly from the NeRF model’s density field or ren-
dering variance to optimize view trajectories [7, 14, 18, 23].
Beyond neural radiance fields, 3D Gaussian representations
are increasingly explored for active reconstruction. Rep-
resentative methods include [15, 20, 26]. These methods
are inherently coupled to the online optimization, requiring
costly gradient-based updates to assess information gain.
Information-Theoretic Approaches.
In parallel to such
uncertainty-driven and RL-based strategies, information-
theoretic approaches aim to reason about view utility in a
more principled manner.
FisherRF [12] exemplifies this
direction by estimating the Fisher Information Matrix of
radiance-field parameters
[17] and selecting views via
Expected Information Gain (EIG). While principled and
ground-truth–free, it remains fundamentally tied to a high-
density differentiable representation and inherits similar
computational burdens, especially under sparse-view con-
ditions where the field has not yet converged.
2.2. Feed-forward 3D Reconstruction Model
Feed-forward 3D reconstruction models aim to infer 3D
structure from images in a single forward pass, avoiding
costly iterative optimization. Unlike traditional multi-view
stereo or differentiable implicit methods [22], these mod-
els provide fast reconstruction and can generalize across
different objects or scenes. Representative approaches in-
clude [33], [19], [32], and [42], which use convolutional
or transformer-based architectures to predict depth, point
clouds, or meshes from single or multiple images. The key
advantage of these models for active perception is their abil-
ity to provide not only a 3D prediction but also a fast, as-
sociated uncertainty estimate for that prediction. This un-
certainty, derived from the model’s internal confidence, can
be computed in a single forward pass without requiring a
partially converged 3D model or gradient-based optimiza-
tion. We adopt VGGT[31] as our backbone feed-forward
2

<!-- page 3 -->
model. It leverages priors learned from large-scale data and
provides robust uncertainty predictions that are directly ex-
ploited for active viewpoint selection, enabling an efficient
and scalable active reconstruction pipeline.
Table 1. Comparison of Different Uncertainty Proxies for Active
Reconstruction.
Uncertainty Proxy
Key Disadvantages
Geometric
Heuristics
[10, 24, 39, 40]
Ignores all perceptual and scene
uncertainty; highly inefficient
sampling with no guarantee of
coverage.
NeRF Rendering
Variance
[7, 14, 18, 23]
Requires costly backpropagation
through volumetric rendering;
impractical for real-time planning;
converges slowly from sparse views.
3D Gaussian
Splatting
[12, 15, 20, 26],
Degrades severely under
sparse-view inputs; still coupled to
online optimization to assess
information gain.
VLM Semantic
Reasoning
[13, 21, 26, 28, 43]
Lacks fine-grained geometric
precision; reasoning is high-level
and non-metric; planning can be
slow and costly.
2.3. Vision-Language Models for Robotics Planning
Large Vision-Language Models (VLMs) have rapidly ad-
vanced in recent years, driven by the emergence of general-
purpose multimodal foundation models
[1, 2, 34, 45].
Building on this progress, recent works have explored using
large Vision-Language Models (VLMs) and multi-modal
models to support high-level task planning in robotics
[6, 46].
These models can take visual observations and
language instructions as input to generate goal-directed
actions or action sequences, enabling flexible decision-
making and generalization. VLM-based approaches have
been applied to navigation, complex manipulation [11, 26],
and multi-step planning, often serving as a high-level pol-
icy that guides downstream low-level controllers. Beyond
task planning, VLMs are increasingly explored for embod-
ied perception and exploration. Approaches such as [9, 13,
21, 28, 43] investigate how language-grounded reasoning
can inform exploration strategies, memory-based scene un-
derstanding, or active information gathering. These meth-
ods highlight the potential of VLMs to capture semantic pri-
ors that are difficult to encode using purely geometric cri-
teria. In the context of active 3D reconstruction, VLMs of-
fer complementary semantic reasoning to geometry-driven
NBV planners. For example, AIR-Embodied [26] leverages
VLMs to infer occlusions or explore semantically meaning-
ful regions. Such high-level reasoning provides orthogonal
guidance to purely geometric or uncertainty-based planners.
In our experiments, we include a VLM-based planner as a
baseline to examine this trade-off.
3. Method
3.1. Overview
Problem Definition. We consider active 3D reconstruc-
tion with a strict view budget. Let T be the total budget,
and let O0 = {(I0, p0), . . . } denote the initial observations
available to the agent, where each observation consists of an
image Iv, its pose pv, and optional depth Dv. The remain-
ing view budget is thus T −|O0|.
Selecting an additional view set S induces the posed ob-
servations
O(S) = O0 ∪{ (Iv, pv) }v∈S.
A reconstructor R maps O(S) to a scene estimate
ˆG(S) = R
 O(S)

.
Given a task metric Q against the ground-truth scene G, the
budgeted view selection problem is formulated as
S⋆∈arg
max
|S|≤T −|O0| Q

ˆG(S), G

.
In practice we solve this budgeted set selection by op-
timizing a principled surrogate objective that scores candi-
date views for their expected reconstruction benefit; the sur-
rogate and scoring are detailed in the subsequent sections.
System Overview.
Our system follows a Dual-Field ac-
tive reconstruction system that unifies semantic cues from a
Vision-Language Model (VLM) with geometric cues from a
feed-forward neural reconstructor [31]. the semantic stream
uses a VLM[45] to produce a complementary semantic un-
certainty field. Both streams are fused on a shared voxel
grid to produce a unified 3D uncertainty field, which guides
the active viewpoint selection strategy. The entire pipeline
is shown in Figure 2.
Specifically, we present our system from three comple-
mentary components: a Feed-forward Confidence Mod-
eling module for geometric uncertainty field estimation, a
Vision Language Model Understanding module for high-
level semantic uncertainty field estimation, and an Active
Viewpoint Selection strategy for efficient view selection
and scene exploration.
3.2. Neural Feed Forward Confidence Modeling
Neural feed-forward reconstruction, trained on large-scale
datasets, exhibits outstanding reliability and robustness,
even under sparse observations, while delivering rapid in-
ference speed. By adopting a feed-forward 3D model, we
3

<!-- page 4 -->
Agentic API
3D Geometric 
Uncertainty Map
Embodied 
Planning & 
Execution 
Embodied agent
With sparse 
observations 
3D Gaussian Splatting
Next-Best-Views
Spatial Weight Map
Candidate centers priority queue 
3D Voxel Dual-Field Uncertainty
Ray Sampling
AREA3D Active Reconstruction System
2.  Geometric Uncertainty Field
1. Semantic Uncertainty Field
3.1 Active Centers Selection 
While views < view budget
1. Greedy selected viewpoint
2. Frustum-based decay
3. Updated priority queue
Select Viewpoint
Q: Structured Prompts
“Your task is to identify regions with HIGH 
UNCERTAINTY that need more observations
1. OCCLUSION & VISIBILITY 
2. GEOMETRIC AMBIGUITY 
3. LIGHT & TEXTURE…
VLM
Images
Parse 
Uncertainty 
Region
Semantic 
Region 
Analysis
Feed-
Forward 
Neural 3D 
Encoder
DPT 
head
3D Perception Module
Dual-Field
Fusion
Back-projection 
to 3D space
FOV masks 
pre-computing
Active Selected Viewpoints
3.2 Active Viewpoints Selection 
Aleatoric 
uncertainty
Figure 2. Overview of the AREA3D pipeline. The framework integrates feed-forward 3D perception and vision-language guidance to
actively select informative viewpoints and to reconstruct high-fidelity geometry via Gaussian Splatting, even under sparse observations.
introduce a geometric uncertainty field derived from the 3D
backbone [31]. Such a representation enables downstream
modules to reason about reconstruction reliability and to
guide active viewpoint selection effectively.
Feed Forward Neural 3D Perception Backbone.
We
adopt VGGT [31] as a transformer-based feed-forward ge-
ometry lifter that maps RGB tokens to dense pixel-level
predictions in a single pass without online optimization.
VGGT outputs a per-pixel depth confidence, which we in-
terpret as predictive precision.
We normalize this confi-
dence into a [0,1] score and splat it onto a common voxel
grid together with the corresponding depth.
Pixel x =
(x, y) with homogeneous coordinate ˜x = [x, y, 1]⊤is
back-projected using intrinsics K and camera pose Ti ∈
SE(3):
Xi(x) = Ti
  ˆDi(x) K−1˜x

,
and the per-view uncertainty scores are aggregated across
frames on the voxel grid to yield the geometric uncertainty
field that drives selection.
Aleatoric Uncertainty Modeling. The pretrained VGGT
encoder is trained with a heteroscedastic objective that
learns a per-pixel precision ci(x) to modulate the depth
residual. In its simplified form[16],
Ldepth =
X
x

ci(x) ℓi(x) −α log ci(x)

,
where ℓi(x) denotes the depth discrepancy. This objective
encourages high confidence ci in reliable regions and lower
values in ambiguous areas, allowing the pretrained confi-
dence map to act as a natural proxy for aleatoric, input-
dependent uncertainty during inference.
We apply a simple monotonic normalization and 3D
back-projection to integrate this uncertainty for fusion and
view selection.
Leveraging this uncertainty-aware feed-
forward design, our VGGT backbone provides robust 3D
perception from sparse observations while maintaining effi-
cient inference for active view planning.
3.3. Vision Language Model for High-Level Rea-
soning
Geometric confidence from the vision backbone cap-
tures areas of high reconstruction ambiguity, but purely
geometry-driven signals may miss semantically important
regions that are difficult to reconstruct.
To provide this
complementary guidance, we induce a semantic uncertainty
field that leverages high-level cues from a Vision Language
Model[45] and modulates them with vision-feature variabil-
ity. The resulting per-image map is then lifted into 3D and
fused across views to guide efficient viewpoint selection.
Uncertainty Region Analysis.
We leverage the Vi-
sion–Language Model as a semantic prior to identify re-
gions where additional observations are likely to improve
reconstruction quality—such as occluded areas, thin struc-
tures, reflective or textureless surfaces. To obtain spatially
grounded and consistent predictions, we design a structured
prompt that divides the image into fixed coarse grids and
asks the VLM to output a small number of region tuples,
4

<!-- page 5 -->
each associated with a category, including OCCLUSION,
GEOMETRIC, LIGHTING, BOUNDARY, TEXTURE and
a corresponding priority score. This constrained format re-
duces the free-form variability of large language models and
allows the responses to be mapped deterministically to im-
age coordinates.
Parsing Uncertainty Regions.
Each VLM-predicted re-
gion is converted into a soft spatial mask Mk(u) ∈[0, 1]
with Gaussian edge tapering and mild dilation for recall.
We assign each region a calibrated weight based on its cat-
egory and priority:
Wi(u) =
K
X
k=1
αtypek βpriok Mk(u),
where α and β are fixed coefficients for region type and
priority, respectively. The aggregated map Wi(u) is nor-
malized per image to [0, 1] for stability.
Let σi(u) denote the feature-level uncertainty from the
vision backbone. The final semantic-modulated uncertainty
is
U sem
i
(u) = Norm
 σi(u) [1 + λ Wi(u)]

,
where λ controls modulation strength.
The resulting
U sem
i
(u) serves as a dense semantic field that integrates
with geometric confidence in 3D to guide viewpoint selec-
tion. By projecting this semantic field into 3D and fusing it
with the feed-forward geometric confidence, our system ob-
tains a unified uncertainty map that guides the agent toward
the most informative viewpoints.
Meanwhile, to ensure that the reconstruction process
does not remain confined to the initially observed views,
we assign a global uncertainty weight to all voxels.
3.4. Active Viewpoint Selection Strategy
Problem Formulation.
With scores from the dual fields,
we could form the active viewpoint selection problem as
an information gain problem. We characterize selection as
maximizing the expected reduction in the fused uncertainty
field within the cone of vision of a candidate pose.
Visibility Gate.
Visibility is a pose-conditioned opera-
tor shared by the semantic and geometric fields. We re-
move out-of-frustum voxels with a deterministic frustum
test. To capture occlusions, we precompute a probabilis-
tic FOV mask at each seed over a coarse yaw/pitch grid.
The mask is generated using Monte Carlo ray sampling with
first-hit termination within the cone of vision, following the
approach of [37]. During selection we reuse these cached
masks instead of reshooting rays. The mask gates all scor-
ing and fusion so that utility is assigned only to potentially
observable content.
Frustum-based Uncertainty Decay.
After commit-
ting a view, we multiplicatively reduce the fused uncer-
tainty within the corresponding precomputed frustum mask.
Algorithm 1: Active View Selection with Dual
Fields Uncertainty Guidance
Input : Workspace W, observations O, view budget N,
MC rays Rpre
Output: Next viewpoints Vnext
Precompute:
1. Voxelize W for candidate seeds.
2. Build dual fields (geometry Fg, semantics Fs); fuse to utility
map U.
3. Run Monte Carlo rays (Rpre) per seed and orientation bin to
estimate visibility; cache FOV masks.
4. Initialize a priority queue Q using mask–weighted utility
bounds (with distance prior).
Iterate: for t = 1 to N do
Pop top seed s⋆from Q;
Evaluate poses at s⋆via mask-only scoring on cached
masks;
Commit best pose to Vnext;
Apply frustum-based decay to U;
Update priority queue for local neighbors under
decayed U; reinsert if promising (with light NMS);
end
return Vnext
This couples selection with evidence accumulation, mak-
ing already explained regions less attractive, while residual
visibility-gated uncertainty guides the policy toward novel
surfaces. Seeds are re-evaluated under the decayed field,
and the same location is revisited only when alternative ori-
entations remain informative.
Viewpoint Candidates Generation. Selection is split into
center and direction stages. We first voxelize the workspace
and treat valid voxel centers as candidate camera seeds. For
each seed, visibility masks over a small set of view orien-
tations are precomputed via Monte Carlo ray sampling and
cached for efficient reuse. Seeds are first pre-processed to
be maintained in a max-priority queue, ranked by an upper
bound on expected information gain under the fused dual
uncertainty field, which enables efficient greedy selection
of the most informative candidate at each iteration. The se-
lection proceeds iteratively in a greedy fashion: at each iter-
ation, the top-ranked seed is popped, a compact fan of can-
didate orientations and ranges is instantiated, and each pose
is scored by fast accumulation over the cached visibility
masks. The highest-scoring view is selected, followed by
light non-maximum suppression to encourage spatial and
angular diversity. Uncertainty within the selected view’s
frustum is decayed by a constant factor, and affected seeds
are re-keyed and re-queued. This greedy loop continues un-
til the view budget is exhausted. The full procedure is sum-
marized in Algorithm 1.
Generally, We couple dual-field utility with precomputed
5

<!-- page 6 -->
Table 2. Scene-level results on the Replica dataset. We report PSNR↑, SSIM↑, and LPIPS↓for four representative scenes. Our method
consistently outperforms baselines across all metrics and scenes.
Method
room0
office0
office2
office4
PSNR↑SSIM↑LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑SSIM↑
LPIPS↓
Random
28.17
0.821
0.152
32.35
0.826
0.152
27.75
0.837
0.157
26.19
0.829
0.166
VLM-based
27.21
0.808
0.193
24.92
0.791
0.201
25.91
0.810
0.187
23.90
0.802
0.190
FisherRF
29.11
0.832
0.151
27.13
0.825
0.156
28.20
0.840
0.142
27.79
0.827
0.152
Ours w/o VLM
28.53
0.855
0.123
31.75
0.842
0.134
28.00
0.853
0.128
30.06
0.847
0.132
Ours
29.23
0.867
0.110
32.98
0.855
0.120
28.70
0.862
0.115
31.79
0.858
0.118
Table 3. Object-level results under different scene complexities. We report PSNR↑, SSIM↑, and LPIPS↓. Our method consistently
outperforms all baselines across varying object counts.
Method
Single-object
5-objects
7-objects 1
7-objects 2
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
Random
31.37
0.870
0.088
29.66
0.882
0.105
29.61
0.853
0.143
32.24
0.892
0.099
Uniform
32.15
0.880
0.088
30.69
0.884
0.109
29.86
0.861
0.146
32.25
0.894
0.099
VLM-based
26.29
0.859
0.125
21.80
0.759
0.218
22.44
0.769
0.191
24.93
0.819
0.169
Air-Embodied
30.35
0.885
0.102
30.35
0.887
0.101
28.35
0.823
0.197
29.85
0.795
0.121
Ours w/o VLM
29.74
0.873
0.096
31.43
0.890
0.101
32.39
0.891
0.083
31.48
0.880
0.107
Ours
31.59
0.893
0.093
31.86
0.893
0.098
33.44
0.899
0.081
32.69
0.902
0.085
visibility. Voxelization bounds the search, and cached frus-
tum masks with a priority queue focus scoring on observ-
able high-gain seeds. Monte Carlo is used for mask con-
struction or occasional refresh. Frustum-based uncertainty
decay turns evidence into reduced utility, yielding a budget-
aware policy that balances exploration and exploitation.
These design choices enable efficient, robust view selection
even under sparse visual observations.
4. Experiments
4.1. Data Generation
We construct a unified active reconstruction benchmark
covering both object-centric and scene-level regimes. We
conduct experiments at both object-level and scene-level
within simulated environments, using CoppeliaSim and
Habitat [25, 27, 30] simulators, respectively.
Scene-level.
For scene-level experiments, we focus on
single-room indoor environments. We adopt the Replica
dataset [29] as our testing scenario and follow the data gen-
eration protocol of Semantic-NeRF [44]. Specifically, we
generate the dataset by replaying the camera trajectories and
split the rendered frames into an initial observation dataset
and test sets for fair evaluation.
Object-level.
For object-level experiments, we focus on
single and multi-object tabletop environments. We adopt
the OmniObject3D dataset [36]. We craft single-object and
multi-object scenarios specifically.The initial observation
dataset and test set are uniformly sampled on a hemisphere
centered on the object. Due to the increased complexity of
the 7-object case, we create two distinct scenarios for eval-
uation.
4.2. Experiment Setup
We evaluate active 3D reconstruction under a fixed view
budget. Let O0 denote the initial observations and T the
total view budget. At each episode, the agent can acquire
up to T −|O0| additional views. We consider two settings:
object-level with |O0| = 4 and T = 25, and scene-level
with |O0| = 15 and T = 40. Reconstruction quality is
evaluated on the resulting scene estimate ˆG(S) using 3D
Gaussian Splatting; specifically, we adopt PGSR [3] as the
downstream representation.
4.3. Metrics
We quantitatively evaluate the reconstruction quality of 3D
Gaussian Splatting using PSNR, SSIM [35], and LPIPS
[41], which jointly measure pixel-level accuracy, structural
consistency, and perceptual similarity. These metrics are
widely adopted in image reconstruction and novel view syn-
thesis tasks, and have been proven to reliably reflect percep-
tual and structural fidelity.
6

<!-- page 7 -->
Figure 3. PSNR as the number of input frames increases under different view-selection policies in the scene-level setting..
Figure 4. PSNR as the number of input frames increases under different view-selection policies in the object-level setting.
4.4. Baselines
We compare our proposed method against a set of repre-
sentative active reconstruction strategies. Several baselines
are common to both experimental levels, while others are
specialized for either the object-level or scene-level task, as
detailed below.
Common Baselines.
Common baselines are launched in
both scene-level and object-level.
• Random Sampling: A non-active baseline where a fixed
number of views are collected by sampling poses randomly.
• Uniform Sampling: A non-active baseline that samples
poses uniformly.
• VLM-based (Naive): An active baseline employing a
VLM. (1) Reasoning: The VLM is prompted with the cur-
rent view. (2) Planning: An LLM interprets the VLM’s out-
put and issues a simple, non-metric movement command.
Task-Specific Baselines. In addition to the common base-
lines, we include specialized planners relevant to each do-
main:
• AIR-Embodied (Object-Level): Air-Embodied is a 3D
Gaussian-based active reconstruction framework. We im-
plement a version of the AIR-Embodied framework [26].
To isolate its planning strategy from its interaction capabil-
ities, we disable the object manipulation module.
• FisherRF (Scene-Level): FisherRF[12] selects explo-
ration paths by maximizing the Expected Information Gain
(EIG), which is approximated from the Fisher Information
of the 3DGS parameters.
4.5. Results
Cross-scale Experiment.
To evaluate the generalization
ability of our model across different spatial scales, we con-
duct cross-scale experiments in both single-room indoor
and tabletop scenarios. This setting examines whether the
proposed framework can adapt to variations in scene scale,
geometry complexity, and object distribution. In the table-
top setup, we include both single-object and multi-object
configurations further to test robustness under different lev-
els of scene complexity.The results are summarized in Ta-
ble 2 and Table 3. Our model achieves the best performance
across all settings according to standard reconstruction met-
rics, including PSNR, SSIM, and LPIPS.
Figure 5 provides visualizations of novel view synthesis
for different policies, highlighting that our approach selects
viewpoints that effectively capture regions left unobserved
by other strategies.
Effectiveness Compared with Baselines.
To verify the
effectiveness of our method, we compare it with represen-
tative baselines under identical experimental setups.
We
additionally plot PSNR as a function of the number of ac-
quired viewpoints to illustrate the efficiency of each pol-
7

<!-- page 8 -->
Figure 5. PSNR comparison as frames increase in scene-level.
Table 4. Ablation study of Feed-Forward Perception and VLM
Guidance on both object-level and scene-level settings. We report
PSNR↑, SSIM↑, and LPIPS↓.
Components
Performance
Feed-Forward
VLM
PSNR↑
SSIM↑
LPIPS↓
Object-level
✗
✓
29.02
0.844
0.202
✓
✗
31.56
0.896
0.091
✓
✓
32.09
0.886
0.102
Scene-level
✗
✓
29.10
0.839
0.115
✓
✗
31.26
0.884
0.097
✓
✓
32.40
0.897
0.089
icy in achieving high-quality reconstruction under a limited
view budget. As shown in Figures 3 and 4, our method gen-
erally achieves higher PSNR with a comparable or smaller
number of viewpoints, for example, 10 frames in the object-
level experiments and 25 frames in the scene-level experi-
ments, demonstrating efficient reconstruction. Notably, the
superiority of our policy is more evident in multi-object sce-
narios (e.g., the 7-object setting), suggesting that it more ef-
fectively captures complex geometries, which is a capability
enabled by the synergy of 3D feed-forward perception and
semantic-level guidance.
4.6. Ablation
We conduct ablations on the two main components of our
framework: Feed-Forward Perception Field and Vision-
Language Guidance Field.
To evaluate their individual
contributions, we disable each component by setting its
corresponding weight to zero while keeping the rest of
the pipeline unchanged. We perform experiments on both
scene-level and object-level scenarios, and in both settings,
the results consistently validate the effectiveness of our
framework design. As shown in Table 4, removing either
component leads to a noticeable performance drop, demon-
strating that the two modules play complementary roles in
improving reconstruction quality.
5. Conclusion
In this paper, we present AREA3D, an active 3D recon-
struction agent that integrates a feed-forward 3D percep-
tion model with vision-language guidance.
By leverag-
ing the complementary strengths of pretrained feed-forward
3D models and vision-language models, AREA3D enables
intelligent and efficient active view selection without re-
quiring optimization-based policies. Extensive experiments
demonstrate that our framework achieves state-of-the-art
performance in novel view synthesis using 3D Gaussian re-
construction, particularly under sparse-view settings.
8

<!-- page 9 -->
Acknowledgements
This work was done while Tianling Xu was an intern at
Harvard University.
References
[1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,
Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei
Huang, et al.
Qwen technical report.
arXiv preprint
arXiv:2309.16609, 2023. 3
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, et al. Qwen2. 5-vl technical report. arXiv preprint
arXiv:2502.13923, 2025. 3
[3] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie,
Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and
Guofeng Zhang. Pgsr: Planar-based gaussian splatting for ef-
ficient and high-fidelity surface reconstruction. IEEE Trans-
actions on Visualization and Computer Graphics, 2024. 6
[4] Xiao Chen, Quanyi Li, Tai Wang, Tianfan Xue, and Jiang-
miao Pang. Gennbv: Generalizable next-best-view policy for
active 3d reconstruction. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 16436–16445, 2024. 2
[5] Brian Curless and Marc Levoy. A volumetric method for
building complex models from range images. In Proceedings
of the 23rd annual conference on Computer graphics and
interactive techniques, pages 303–312, 1996. 2
[6] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch,
Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson,
Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An
embodied multimodal language model. 2023. 3
[7] Ziyue Feng, Huangying Zhan, Zheng Chen, Qingan Yan, Xi-
angyu Xu, Changjiang Cai, Bing Li, Qilun Zhu, and Yi Xu.
Naruto: Neural active reconstruction from uncertain target
observations. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21572–
21583, 2024. 2, 3
[8] Lily Goli, Cody Reading, Silvia Sell´an, Alec Jacobson,
and Andrea Tagliasacchi. Bayes’ rays: Uncertainty quan-
tification for neural radiance fields.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20061–20070, 2024. 1
[9] Mobin Habibpour and Fatemeh Afghah. History-augmented
vision-language models for frontier-based zero-shot object
navigation. arXiv preprint arXiv:2506.16623, 2025. 3
[10] Lei Hou, Xiaopeng Chen, Kunyan Lan, Rune Rasmussen,
and Jonathan Roberts. Volumetric next best view by 3d oc-
cupancy mapping using markov chain gibbs sampler for pre-
cise manufacturing. IEEE Access, 7:121949–121960, 2019.
2, 3
[11] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li,
Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value
maps for robotic manipulation with language models. arXiv
preprint arXiv:2307.05973, 2023. 3
[12] Wen Jiang, Boshu Lei, and Kostas Daniilidis. Fisherrf: Ac-
tive view selection and mapping with radiance fields using
fisher information. In European Conference on Computer
Vision, pages 422–440. Springer, 2024. 1, 2, 3, 7
[13] Wen Jiang, Boshu Lei, Katrina Ashton, and Kostas Dani-
ilidis. Multimodal llm guided exploration and active map-
ping using fisher information.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 5392–5404, 2025. 3
[14] Liren Jin, Xieyuanli Chen, Julius R¨uckin, and Marija
Popovi´c. Neu-nbv: Next best view planning using uncer-
tainty estimation in image-based neural rendering. In 2023
IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), pages 11305–11312. IEEE, 2023. 2,
3
[15] Liren Jin, Xingguang Zhong, Yue Pan, Jens Behley, Cyrill
Stachniss, and Marija Popovi´c. Activegs: Active scene re-
construction using gaussian splatting. IEEE Robotics and
Automation Letters, 2025. 2, 3
[16] Alex Kendall and Yarin Gal. What uncertainties do we need
in bayesian deep learning for computer vision? Advances in
neural information processing systems, 30, 2017. 4
[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2
[18] Soomin Lee, Le Chen, Jiahao Wang, Alexander Liniger,
Suryansh Kumar, and Fisher Yu. Uncertainty guided pol-
icy for active robotic 3d reconstruction using neural radiance
fields. IEEE Robotics and Automation Letters, 7(4):12070–
12077, 2022. 1, 2, 3
[19] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Ground-
ing image matching in 3d with mast3r. In European Confer-
ence on Computer Vision, pages 71–91. Springer, 2024. 2
[20] Yuetao Li, Zijia Kuang, Ting Li, Qun Hao, Zike Yan,
Guyue Zhou, and Shaohui Zhang. Activesplat: High-fidelity
scene reconstruction through active gaussian splatting. IEEE
Robotics and Automation Letters, 2025. 2, 3
[21] Zhichen Lou, Kechun Xu, Zhongxiang Zhou, and Rong
Xiong.
Explorevlm: Closed-loop robot exploration task
planning with vision-language models.
arXiv preprint
arXiv:2508.11918, 2025. 3
[22] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[23] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang. Ac-
tivenerf: Learning where to see with uncertainty estimation.
In European Conference on Computer Vision, pages 230–
246. Springer, 2022. 2, 3
[24] Alexandru Pop and Levente Tamas. Next best view estima-
tion for volumetric information gain. IFAC-PapersOnLine,
55(15):160–165, 2022. 2, 3
[25] Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dal-
laire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai,
Alexander William Clegg, Michal Hlavac, So Yeon Min,
et al.
Habitat 3.0: A co-habitat for humans, avatars and
robots. arXiv preprint arXiv:2310.13724, 2023. 6
9

<!-- page 10 -->
[26] Zhenghao Qi, Shenghai Yuan, Fen Liu, Haozhi Cao,
Tianchen Deng, Jianfei Yang, and Lihua Xie. Air-embodied:
An efficient active 3dgs-based interaction and reconstruc-
tion framework with embodied large language model. arXiv
preprint arXiv:2409.16019, 2024. 2, 3, 7
[27] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets,
Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia
Liu, Vladlen Koltun, Jitendra Malik, et al.
Habitat: A
platform for embodied ai research.
In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 9339–9347, 2019. 6
[28] Venkatesh Sripada, Samuel Carter, Frank Guerin, and Amir
Ghalamzan. Scene exploration by vision-language models,
2025. 3
[29] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl
Ren, Shobhit Verma, et al. The replica dataset: A digital
replica of indoor spaces. arXiv preprint arXiv:1906.05797,
2019. 6
[30] Andrew Szot, Alexander Clegg, Eric Undersander, Erik Wi-
jmans, Yili Zhao, John Turner, Noah Maestre, Mustafa
Mukadam, Devendra Singh Chaplot, Oleksandr Maksymets,
et al. Habitat 2.0: Training home assistants to rearrange their
habitat. Advances in neural information processing systems,
34:251–266, 2021. 6
[31] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 2, 3, 4
[32] Qianqian
Wang,
Yifei
Zhang,
Aleksander
Holynski,
Alexei A Efros, and Angjoo Kanazawa. Continuous 3d per-
ception model with persistent state. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
10510–10522, 2025. 2
[33] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20697–
20709, 2024. 2
[34] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long
Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong
Ye, Jie Shao, et al.
Internvl3. 5: Advancing open-source
multimodal models in versatility, reasoning, and efficiency.
arXiv preprint arXiv:2508.18265, 2025. 3
[35] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 6
[36] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren,
Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian,
et al. Omniobject3d: Large-vocabulary 3d object dataset for
realistic perception, reconstruction and generation. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 803–814, 2023. 6
[37] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai
Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-
term consistent world simulation with memory.
arXiv
preprint arXiv:2504.12369, 2025. 5
[38] Dongyu Yan, Jianheng Liu, Fengyu Quan, Haoyao Chen,
and Mengmeng Fu. Active implicit object reconstruction us-
ing uncertainty-guided next-best-view optimization. IEEE
Robotics and Automation Letters, 8(10):6395–6402, 2023.
1, 2
[39] Zike Yan, Haoxiang Yang, and Hongbin Zha. Active neu-
ral mapping. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 10981–10992, 2023.
2, 3
[40] Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang,
and Bernadette Bucher.
Vlfm: Vision-language frontier
maps for zero-shot semantic navigation. In 2024 IEEE In-
ternational Conference on Robotics and Automation (ICRA),
pages 42–48. IEEE, 2024. 2, 3
[41] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 6
[42] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue,
Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gor-
don Wetzstein. Flare: Feed-forward geometry, appearance
and camera estimation from uncalibrated sparse views. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 21936–21947, 2025. 2
[43] Xinxin Zhao, Wenzhe Cai, Likun Tang, and Teng Wang.
Imaginenav: Prompting vision-language models as embod-
ied navigator through scene imagination.
arXiv preprint
arXiv:2410.09874, 2024. 3
[44] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and An-
drew J Davison. In-place scene labelling and understanding
with implicit scene representation.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 15838–15847, 2021. 6
[45] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shen-
glong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su,
Jie Shao, et al. Internvl3: Exploring advanced training and
test-time recipes for open-source multimodal models. arXiv
preprint arXiv:2504.10479, 2025. 3, 4
[46] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted
Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker,
Ayzaan Wahid, et al. Rt-2: Vision-language-action models
transfer web knowledge to robotic control. In Conference on
Robot Learning, pages 2165–2183. PMLR, 2023. 3
10

<!-- page 11 -->
AREA3D: Active Reconstruction Agent with Unified Feed-Forward 3D
Perception and Vision-Language Guidance
Supplementary Material
6. Dataset and Benchmark
In Sec. 4.1 of the main paper, we introduce our unified
benchmark for active 3D reconstruction. Here we provide
the complete details of the dataset configuration and scene
construction.
As illustrated in Fig. 6, we include eight
scenes in total: four single-room scenes that capture diverse
indoor layouts, and four tabletop scenes featuring object-
centric setups with rich geometric details and occlusions.
These scenes are used consistently across baselines and ab-
lations, enabling fair comparison under the same camera
budget.
7. Implementation Details
Systematic Prompt for VLM.
In Sec. 3.3 we describe
how the VLM output is fused with geometric uncertainty.
Here we detail the concrete prompt used in practice. At
the beginning of each episode, the agent collects O0 ini-
tial RGB views, and we query the VLM once with all
O0 frames. For each image, the field of view is divided
into a coarse grid (horizontal: left, center-left, center-right,
right; vertical:
top, middle, bottom), and the VLM is
asked to return a ranked list of regions, each with a lo-
cation, an uncertainty type (OCCLUSION, GEOMETRIC,
LIGHTING, BOUNDARY, or TEXTURE), a priority level
(HIGH, MEDIUM, LOW), and a short natural-language jus-
tification. The textual output is then parsed into per-pixel
importance maps and lifted into a 3D, visibility-aware un-
certainty field via 2D-to-3D unprojection.
For completeness, we show the instruction given to the
VLM. We provide all O0 initial images together with the
following text:
You are an expert visual analyzer for active 3D recon-
struction. You will be given several RGB images (ini-
tial observations) from the same scene. For each image,
independently identify regions that require additional
viewpoints for complete 3D reconstruction.
Coordinate system. Divide each image into a 4 × 3
grid: horizontal positions are left, center-left, center-
right, and right; vertical positions are top, middle, and
bottom. Example locations include “left-top”, “center-
left-middle”, “right-bottom”, and “center-right-top”.
Uncertainty categories (ranked by priority).
• OCCLUSION (high): hidden or blocked surfaces;
regions behind furniture, walls, or large objects; back
faces only visible from narrow viewing angles.
• GEOMETRIC (high): thin structures, surfaces at
grazing angles, complex curved shapes, reflective or
transparent materials.
• LIGHTING (medium):
deep shadows, overex-
posed areas, strong highlights, blur, very low-
contrast regions.
• BOUNDARY (medium): objects cut by image bor-
ders, incomplete views, extreme tangential angles.
• TEXTURE (low): textureless, repetitive, or very
low-contrast regions.
Output format. For each image, list 5–8 regions in
decreasing order of importance. Each region should be
summarized in one line with the following fields:
• REGION: location using the grid notation (e.g.,
“center-left-middle”).
• TYPE:
one
of
OCCLUSION,
GEOMETRIC,
LIGHTING, BOUNDARY, TEXTURE.
• PRIORITY:
HIGH
(must
observe),
MEDIUM
(should observe), or LOW (nice to observe).
• SIZE: small (< 10%), medium (10–25%), or
large (> 25%) of the image.
• REASON: 1–2 sentences explaining why extra view-
points are needed and what 3D information is cur-
rently missing.
Parsing Uncertainty Regions.
In Sec. 3.3 of the main
paper we define the 2D spatial weight map
Wi(u) =
X
k
αtypek βpriok Mk(u),
(1)
and the semantic modulation;
U sem
i
(u) = Norm
 σi(u) [1 + λ Wi(u)]

.
(2)
In our implementation, we employ a fixed weighting
scheme that reflects the relative importance of different pri-
orities and remains constant across all experiments.
We set the priority and size-dependent coefficients to
fixed values summarized in Table 5; the same settings are
used for all experiments. After aggregating over regions,
Wi(u) is normalized per image to [0, 1].
Frustum-based Uncertainty Decay.
In Sec. 3.4 of the
main paper we state that, after committing a view, we mul-
tiplicatively reduce the fused uncertainty inside the corre-
sponding frustum. Here we detail the decay rule used in our
implementation.
Let ut(v) denote the fused 3D uncertainty at voxel cen-
ter v at step t. Given a committed camera pose T c
w, we
1

<!-- page 12 -->
Single-object
5-objects
7-objects 1
7-objects 2
room 0
office 0
office 2
office 4
Figure 6. Four single-room scenes that capture diverse indoor layouts, and four tabletop scenes featuring object-centric setups with rich
geometric details and occlusions
Table 5. Coefficients for VLM region priority, size, and modula-
tion.
Symbol
Meaning
Value
βHIGH
priority = HIGH
3.0
βMED
priority = MEDIUM
1.5
βLOW
priority = LOW
0.5
ssmall
size = small
0.8
smedium
size = medium
1.0
slarge
size = large
1.2
λ
modulation strength
1.0
first determine the set of voxels whose centers fall inside
the viewing frustum Frustum(T c
w), using the camera for-
ward direction, a field-of-view threshold, and a depth range
consistent with view rendering). The uncertainty is then up-
dated by
ut+1(v) =
(
(1 −η) ut(v),
v ∈Frustum(T c
w),
ut(v),
otherwise,
(3)
i.e., all voxels inside the frustum are scaled by a constant
decay factor while others remain unchanged.
The hyperparameters used for this frustum-based decay
are summarized in Table 6 and are kept fixed for all experi-
ments.
Table 6. Hyperparameters for frustum-based uncertainty decay.
Symbol
Meaning
Value
η
decay factor
0.3
FOV
field of view
90◦
max depth
maximum depth
5 m
8. More Quantitative Results
Overall Aggregate Performance.
To summarize per-
formance on our benchmark, we aggregate the per-scene
PSNR, SSIM, and LPIPS reported in the main paper, sep-
arately for the object-level and scene-level configurations.
Averaged over all object-level scenes, our policy attains
32.09 PSNR, 0.886 SSIM, and 0.102 LPIPS. On the scene-
level benchmark, the corresponding averages are 32.40
PSNR, 0.897 SSIM, and 0.089 LPIPS. These aggregated
scores provide a compact summary of our behavior on both
parts of the benchmark and are consistent with the per-scene
comparisons in the main paper, where our policy generally
performs on par with or better than competing methods un-
der a fixed view budget.
Ablation on Global Initial Weight.
In Sec. 3.3 of the
main paper we state that, to prevent the agent from being
confined to the initially observed views, we assign a global
initial uncertainty weight to all voxels. Here we describe
the exact form used in implementation and compare it with
2

<!-- page 13 -->
Table 7. Ablation study of the global initial weight on both object-
level and scene-level benchmarks.
Setting
PSNR↑
SSIM↑
LPIPS↓
Object-level
γ = 0
29.212
0.859
0.120
γ = 0.01 (ours)
29.661
0.870
0.111
Scene-level
γ = 0
27.845
0.837
0.153
γ = 0.005 (ours)
28.265
0.848
0.112
a variant that removes this term.
Let ˆU(v) denote the fused 3D uncertainty projected from
the 2D semantic-modulated field. Before view selection be-
gins, each voxel is assigned a small additive initial weight
˜U(v) = ˆU(v) + γ,
(4)
where γ is a constant offset that ensures non-zero uncer-
tainty for voxels not covered by the initial observation set.
This additive form is preserved throughout the reconstruc-
tion process and the global initial weight undergoes the
same frustum-based decay as the other uncertainty compo-
nents. In practice, we use different values for the two bench-
marks: γ = 0.01 for the object-level setting and γ = 0.005
for the scene-level setting.
We compare two configurations: (i) a baseline without
global initial weight, where γ = 0; and (ii) our default set-
ting with a non-zero initial weight (γ = 0.01 for object-
level, γ = 0.005 for scene-level), which preserves a min-
imal amount of residual uncertainty in unseen regions and
encourages the policy to explore outside the initially ob-
served frustum.
Quantitative results on both object-level and scene-level
benchmarks are reported in Table 7. Using a non-zero initial
weight consistently improves viewpoint coverage and leads
to better long-range reconstruction quality.
9. More Visualization Results
Due to space constraints in the main paper, we only show
three qualitative examples of novel view synthesis results
obtained with 3D Gaussian Splatting under our active re-
construction policy. Here we provide additional visualiza-
tions covering both scene-level and object-level settings.
Each row compares our method with baselines on the same
target view, as illustrated in Fig. 7 and Fig. 8.
3

<!-- page 14 -->
Figure 7. Novel View Synthesis Results of different policies in scene-level.
4

<!-- page 15 -->
Figure 8. Novel View Synthesis Results of different policies in object-level.
5
