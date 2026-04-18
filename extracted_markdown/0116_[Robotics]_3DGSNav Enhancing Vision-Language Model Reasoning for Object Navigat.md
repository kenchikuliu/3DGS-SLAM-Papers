<!-- page 1 -->
3DGSNav: Enhancing Vision-Language Model Reasoning for
Object Navigation via Active 3D Gaussian Splatting
Wancai Zheng 1 Hao Chen 2 Xianlong Lu 1 Linlin Ou 1 Xinyi Yu 1
Abstract
Object navigation is a core capability of embod-
ied intelligence, enabling an agent to locate target
objects in unknown environments. Recent ad-
vances in vision–language models (VLMs) have
facilitated zero-shot object navigation (ZSON).
However, existing methods often rely on scene
abstractions that convert environments into se-
mantic maps or textual representations, causing
high-level decision making to be constrained by
the accuracy of low-level perception.
In this
work, we present 3DGSNav, a novel ZSON
framework that embeds 3D Gaussian Splatting
(3DGS) as persistent memory for VLMs to en-
hance spatial reasoning.
Through active per-
ception, 3DGSNav incrementally constructs a
3DGS representation of the environment, en-
abling trajectory-guided free-viewpoint rendering
of frontier-aware first-person views. Moreover,
we design structured visual prompts and inte-
grate them with Chain-of-Thought (CoT) prompt-
ing to further improve VLM reasoning. During
navigation, a real-time object detector filters po-
tential targets, while VLM-driven active view-
point switching performs target re-verification,
ensuring efficient and reliable recognition. Ex-
tensive evaluations across multiple benchmarks
and real-world experiments on a quadruped
robot demonstrate that our method achieves
robust and competitive performance against
state-of-the-art approaches. The Project Page:
https://aczheng-cai.github.io/3dgsnav.github.io/
1. Introduction
Object navigation (ObjectNav) is a core capability of em-
bodied intelligence, requiring an agent to efficiently localize
1Zhejiang University of Technology, Hangzhou, China.
2Zhejiang University, Hangzhou, China. Correspondence to: Xinyi
Yu <yuxy@zjut.edu.cn>.
Preprint. February 13, 2026.
Figure 1. 3DGSNav Real-World Demonstration. The robot
successfully navigates to locate the toilet. The reasoning process is
simplified for clarity. The manipulator can move freely to support
active perception.
a target in unknown environments. This task requires an
agent to jointly understand spatial structures and semantic
information in complex scenes, as well as to perform long-
horizon planning based on continuous observation. Recent
works (Zhang et al., 2024a; Zhu et al., 2025; Wen et al.,
2025) have incorporated vision–language models (VLMs)
to support decision-making in zero-shot object navigation
(ZSON), leveraging their strong contextual understanding
and reasoning abilities. However, these approaches often
fail to fully exploit the visual–spatial reasoning potential
inherent in VLMs. Instead, they typically abstract environ-
ments into textual descriptions (Zhou et al., 2024; Qiao et al.,
2025) or construct semantic value maps (Zhang et al., 2025;
Yokoyama et al., 2024) from current observations, thereby
discretizing the continuous environment into semantic-level
representations. While such formulations simplify the plan-
ning process, they weaken the spatial and structural rela-
tionships among objects. To mitigate this limitation, meth-
ods (Yin et al., 2024; Chen et al., 2023) introduce 3D scene
graphs to enhance relational modeling among objects. Nev-
ertheless, the perception module remains a bottleneck for
overall performance, preventing VLMs from fully leverag-
ing their inherent spatial reasoning capabilities.
”The ability of VLMs to generate travel guides, summarize
lengthy documents, and edit images is truly impressive.” –
VLM users
1
arXiv:2602.12159v1  [cs.RO]  12 Feb 2026

<!-- page 2 -->
3DGSNav
This observation highlights the strong planning, textual rea-
soning, and visual inference capabilities of VLMs. With
the rapid advancement of VLMs, an important open prob-
lem is how to maximize their potential in ZSON planning
while avoiding bottlenecks from low-level perceptual con-
straints. In this work, we propose 3DGSNav, a novel frame-
work that constructs environmental representations using
3DGS through active perception, which are leveraged as
persistent memory for VLMs. An opacity-driven viewpoint
selection mechanism is introduced to enable the agent to
actively acquire missing perceptual information, thereby im-
proving overall sensing efficiency. Leveraging the efficient
novel-view synthesis (NVS) capability of 3DGS, histori-
cal observations are rendered from memory. A trajectory-
guided free-viewpoint optimization strategy is then em-
ployed to generate informative first-person views (FPVs)
that support the reasoning of the planner VLM (Google,
2025) used in our system.
Furthermore, structured vi-
sual prompts are created from FPVs and a bird’s-eye view
(BEV), with online annotations integrated into Chain-of-
Thought (CoT) prompting to enhance the VLM spatial rea-
soning and long-horizon planning, enabling analysis and
decision-making over candidate frontier points. To maintain
lightweight and efficient navigation, a real-time object de-
tector (Wang et al., 2025a) identifies potential targets, while
the action-decision VLM (Team et al., 2026) performs target
verification or selects appropriate actions when necessary.
Selected actions are projected into the 3DGS memory to
render novel viewpoints, enabling active viewpoint-based re-
verification of potential targets and improving recognition
reliability and accuracy. Our contributions include:
• We embed 3DGS as a memory representation for VLM,
enabling efficient memory construction and retrieval
via active perception and free-viewpoint optimization,
and providing analyzable visual observations to sup-
port reasoning.
• We design an annotation-empowered visual prompting
framework that integrates structured visual prompts
with CoT prompting to effectively elicit the spatial
reasoning capabilities of VLMs for efficient ZSON,
without relying on scene abstraction.
• We conduct evaluations in real environments and on
several datasets, and the experiments verify the advan-
tages and efficacy of our approach.
2. Related work
2.1. Object Navigation
Object-goal navigation is a fundamental task in embod-
ied intelligence. Prior work primarily focuses on end-to-
end methods, such as reinforcement learning (Dang et al.,
2023; Ye et al., 2021; Mousavian et al., 2019) and imitation
learning (Ramrakhya et al., 2023; 2022), which learn direct
mappings from sensory observations to actions in closed
datasets or simulation. However, these approaches are often
limited by sim-to-real gap and substantial computational
costs, making real-world generalization challenging.
2.2. Zero-Shot Object Navigation
Benefiting from pretraining on large-scale datasets, large
models have achieved remarkable progress in zero-shot un-
derstanding. To enable robots to recognize unseen target cat-
egories in unknown environments, prior work has explored
incorporating large models into ObjectNav. Representative
approaches (Shi et al., 2025; Yu et al., 2023; Kuang et al.,
2024) construct semantic maps that textualize object infor-
mation around frontier points and employ large language
models (LLMs) to compute semantic similarity between
candidate frontiers and the target object, thereby guiding
exploration. Other methods (Yin et al., 2024; Chen et al.,
2023; Werby et al., 2024) build 3D scene graphs from se-
mantic segmentation results to enrich scene representations
and leverage LLMs to infer the next exploration location by
reasoning over spatial and semantic relationships. Beyond
LLM-based methods, VLFM (Yokoyama et al., 2024) inte-
grates a VLM (Li et al., 2023) with online observations to
continuously update a target-related similarity value map,
improving exploration efficiency. ApexNav (Zhang et al.,
2025) further enhances the robustness of semantic value
map construction through rule-based optimization and the
integration of additional auxiliary models. Recent advances
in 3D scene representations have also opened new directions
for embodied navigation. BeliefMapNav (Zhou et al., 2025)
adopts voxel-based maps with multi-level spatial semantics
and LLM reasoning to estimate target confidence, while the
work (Zhang et al., 2023) leverages point clouds to build
3D semantic maps that capture fine-grained geometry and
object relationships.
Despite these advances, most existing methods rely on se-
mantic or textual scene abstractions that inevitably lose
continuous spatial structure and fine-grained geometric de-
tails. This forces high-level decision making to depend pri-
marily on semantic modules and symbolic representations,
restricting effective navigation. To address this limitation,
we replace scene abstractions with 3DGS-based represen-
tations to model the environment, enabling VLMs to fully
exploit their inherent spatial understanding and reasoning
capabilities for ZSON.
2.3. LMMs for Spatial Reasoning
Recent advancements in 3D large multimodal models
(LMMs) have increasingly incorporated explicit geometric
or reconstructive priors to enhance spatial reasoning capa-
2

<!-- page 3 -->
3DGSNav
bilities, with applications spanning tasks such as 3D visual
question answering (VQA) and 3D grounding (Hong et al.,
2023; Wang et al., 2025b). Notably, 3D LLaVA (Deng et al.,
2025) and LL3DA (Chen et al., 2024a) focus on efficient
modality bridging, utilizing superpoint pooling for com-
pact 3D tokenization and query-based transformer aggrega-
tion for prompt-conditioned 3D understanding, respectively.
To capture scene geometry and object relationships more
effectively, Scene-LLM (Fu et al., 2025) adopts a hybrid
3D visual feature representation with scene state updates,
while another approach (Yu et al., 2025) explicitly mod-
els instance-level spatial relations. Recent video-centric
methods further enhance spatiotemporal representations
by integrating depth-derived or reconstruction-based 3D
cues (Zheng et al., 2025; Fan et al., 2025). Evaluations on
VSI-Bench (Yang et al., 2025b) indicate that recent video-
language models (Li et al., 2024; Liu et al., 2024; Chen
et al., 2024b) still face persistent challenges in visual spatial
reasoning.
However, these approaches focus on short-horizon interac-
tion, instruction following, or scene-level reasoning, where
the model passively analyzes a given scene snapshot. ZSON
necessitates long-horizon sequential decision-making under
partial observability. In this dynamic setting, the agent must
continuously integrate observations while preserving visual
spatial consistency. To the best of our knowledge, we are the
first to systematically incorporate 3DGS representations into
VLMs to enable long-horizon, complex spatial reasoning.
3. Approach
Task Definition. In ZSON, given a natural language in-
struction I (e.g., “find a chair”), the agent is required to
autonomously explore an unknown environment and nav-
igate to a location within a distance ds of the target ob-
ject and remaining there, all within a maximum of δmax
steps. At each time step t, the agent receives an observa-
tion, which includes the RGB image It, depth map Dt, and
pose T t
cw ∈SE(3). The navigation policy then predicts an
action At.
Overview. Figure 2 provides an overview of 3DGSNav.
Given an agent pose and an RGB-D observation, environ-
mental information is acquired via active perception, and
navigation maps are constructed using 3DGS. In the plan-
ning stage, a trajectory-guided free-viewpoint optimization
method is introduced to render information-dense FPVs of
candidate frontier points, which are augmented with explicit
visual annotations. These structured visual prompts are
then fed into the VLM, which performs spatial and seman-
tic reasoning to select optimal exploration targets via CoT
prompting. During navigation, a real-time object detector
is employed to efficiently detect potential targets. When
detection results are ambiguous, the VLM is invoked to
guide action selection by actively adjusting the rendering
viewpoint, enabling the acquisition of more discriminative
observations for target re-verification.
3.1. Perception
3.1.1. PRELIMINARIES
The 3D map is represented by a set of anisotropic Gaussian
primitives G:
G = {Gi : (µi, oi, ci, Σi)|i = 1, ..., N},
(1)
where each Gaussian primitive is defined by its position
µ ∈R3, opacity o ∈[0, 1], color c ∈R3, and covariance
matrix Σi = RSST RT , where S is a scale matrix and R is
a rotation matrix. Each Gaussian primitive Gi is projected
onto the image plane via the world-to-camera transforma-
tion Tcw for rendering. The projected covariance matrix is
computed as follows:
Σ′
i = JTcwΣiT T
cwJT ,
(2)
where J denotes the Jacobian matrix of the affine approxima-
tion to the projective function. Therefore, the rendered color
˜I, depth ˜D, and opacity ˜O are computed as follows (Kerbl
et al., 2023):
{˜I, ˜D, ˜O} =
N
X
i=1
{ci, zi, 1}σi
i−1
Y
j=1
(1 −σj),
(3)
where the color of the i-th 3D Gaussian is given by ci,
σi is the density term determined jointly by the Gaussian
distribution and the learned opacity oi, and zi denotes the
corresponding depth value in the camera coordinate.
The 3D map is optimized in the backend under the following
supervision:
Lg = λ1
 λ∥˜I −I∥2
2 + (1 −λ)(1 −SSIM(˜I, I))

+ λ2∥˜D −D∥2
2,
(4)
where SSIM is the structural similarity index mea-
sure (Wang et al., 2004), and {λ∗} are hyperparameters.
3.1.2. ACTIVE PERCEPTION
To improve perception efficiency, an active perception
method is proposed to replace the inefficient passive ro-
tational scanning. This approach leverages the flexibility of
a virtual camera by rendering a panoramic opacity field to
estimate observation completeness (details in Appendix B).
Subsequently, the DBSCAN algorithm Γ is applied to clus-
ter low-opacity regions, identifying representative discrete
target viewpoints in the continuous viewpoint space. Since
roll does not affect the observation process, the target view-
point is computed from the region S with the largest low-
opacity area and parameterized by pitch θ and yaw ϕ. These
3

<!-- page 4 -->
3DGSNav
Figure 2. System overview. The system builds navigation-oriented environment representations from robot poses and RGB-D observations
via active perception. Free-viewpoint optimization and structured visual prompts guide VLM-based zero-shot navigation planning, while
online object detection and viewpoint re-verification enable efficient target localization.
angles are then used to guide the motion of the real camera,
enabling it to actively cover visually missing regions.
(θ∗, ϕ∗) = Centroid
 
arg max
C∈Γ({(θ,ϕ), ˜
O(θ,ϕ)<τ})
S(C)
!
, (5)
where τ denotes the threshold used to measure low-opacity.
This strategy not only avoids localization errors and redun-
dant observations caused by mechanical rotation, but also
significantly enhances exploration efficiency and robustness
in long-term navigation.
3.1.3. FRONTIERS
In the 3DGS space, we construct an exploration map by
integrating opacity-based rendering and agent height infor-
mation from a top-down perspective (details are provided in
Appendix C). Frontier points P, defined as the boundaries
between explored and unexplored regions, are extracted
from the explored map following (Yu et al., 2023). To en-
sure that the extracted frontier points lie within observed
regions, their positions are further refined based on Cheby-
shev distance.
Given the definition of frontiers, rendering FPVs from dis-
crete frontier points often yields incomplete or local obser-
vations, whereas frontiers within the same region jointly
capture the more environmental context. To effectively
model the shared environmental information across multiple
frontiers, we propose a spatial-structure adaptive frontier
point clustering method. Specifically, a distance field is
constructed from the geometric information encoded in the
explored map, and local distance maxima are detected to
identify core skeleton points. These skeleton points serve
as adaptive seeds for a watershed segmentation algorithm,
and the centroid of each segmented region is selected as
a representative frontier point. This process enables uni-
fied modeling of redundant frontier information, producing
higher-quality observations during rendering while reduc-
ing the computational overhead associated with analyzing
isolated frontier points.
3.1.4. GUIDANCE TRAJECTORY
The guidance trajectory provides a principled reference
for free-viewpoint optimization and helps establish spatial
awareness for VLM-based planning. We design a cost func-
tion C(n) based on the distance transform to guide the
Dijkstra algorithm toward the shortest path to each frontier
point, preventing the trajectory from closely adhering to
obstacle boundaries. The total cost of node n is defined as:
C(n) = C(n −1) + ∆L + Ω(n),
(6)
where C(n −1) denotes the cost of the parent node, ∆L
represents the step length that controls search efficiency,
and Ω(n) is an exponential penalty term related to obstacle
proximity. To encourage paths to stay away from high-risk
regions, the penalty term is activated when the distance to
the nearest obstacle d(n) falls below a safety threshold ds:
Ω(n) = ω · exp(2 · (ds −d(n))),
(7)
where ω is a weight. By constructing a virtual artificial po-
tential field with a non-linear gradient, this function causes
the cost to increase exponentially as the distance to ob-
stacles decreases. The optimal guidance trajectory to the
i-th frontier point Pi is obtained as Ti = {n1, n2, . . . , nj},
4

<!-- page 5 -->
3DGSNav
where each node ni satisfies the criterion of minimizing the
accumulated cost C(n).
3.2. Planning
3.2.1. VIRTUAL VIEWPOINT INITIALIZATION
Since the guidance trajectory T represents the shortest path,
turns along the trajectory that are closest to a frontier point
generally provide greater visibility of the frontier region.
To focus on the frontier while capturing the surrounding
environment, we design a point-based initialization strategy
that leverages curvature ˆκi and distance ˆdi to determine
the initial virtual viewpoint position p∗by maximizing the
weighted score:
p∗= arg max
pi∈Pv
h
αˆκi + (1 −α) ˆdi
i
,
(8)
where α is a weighting factor that balances the contribu-
tions of curvature and distance. The valid candidate set Pv
is constrained by a minimum safety distance rmin and a
maximum look-back distance rmax:
Pv = {pi ∈T | rmin < ∥pi −P∥≤rmax} .
(9)
This approach ensures that the initial position is neither too
close, which could impair optimization efficiency, nor too
distant, maintaining a manageable and stable solution space.
Finally, the initial position p∗and the orientation toward the
i-th frontier point Pi are combined to determine the initial
pose ξ∗
i .
3.2.2. FREE-VIEWPOINT OPTIMIZATION
The initial pose cannot handle all scenarios, with some
viewpoints occluded and the VLM unable to reason about
frontier information. To resolve this, camera pose estimation
is formulated as a multi-constraint optimization problem to
determine the optimal viewpoint. Given an initial pose ξ∗
k,
the guidance trajectory Tk, and the frontier point Pk, the
camera rotation quaternion q and translation t are iteratively
optimized by minimizing a composite loss function:
L = λoLopa + λvLvis + λcLcos + λtLtraj.
(10)
Opacity loss. The rendered FPV ˜Ik should balance ob-
served and unobserved regions, as excessive unobserved
areas can hinder analysis of the frontier environment. To
quantify and regulate this balance, we introduce an opacity
loss that measures the proportion of unobserved regions:
Lopa = E[1 −˜O],
(11)
the E represents the averaging operation.
Ray occlusion loss. To ensure that the rendered FPV di-
rectly observes the frontier point, we model occlusion using
a depth consistency constraint:
Lvis = E[1 −Prv], Prv = σ( ˜Ds −zPk).
(12)
Rays are cast from the camera center toward the frontier
point and sampled at N points along the ray. The occlusion
probability is determined by a sigmoid σ function, which
compares the sampled depth ˜Ds, obtained from the frontier
point depth in the virtual camera, with the frontier point
depth zPk.
View alignment loss. To guarantee that the camera consis-
tently gazes toward the frontier point Pk, we compute the
cosine similarity between the camera forward vector and the
direction vector from the camera to the frontier. This loss
term is defined as:
Lcos = (1 −cos2(θ)) · E[Prv],
(13)
where θ is the angle between the two vectors. By introducing
the occlusion probability Prv, the weight of this constraint
is automatically reduced when the target is occluded.
Guidance trajectory loss. To enable the virtual camera to
adjust its position near the guidance trajectory, we design a
guidance trajectory loss. This loss calculates the minimum
smoothed distance between the current camera position t
and the trajectory point set pi ∈Tk:
Ltraj = −1
β ln
X
i
(w exp(−βdi)) , di = ∥t −pi∥2,
(14)
where w is a weight factor based on the distance dPk from
the camera to the frontier point, encouraging the camera to
move closer during optimization. The weight is adaptively
determined using the opacity loss:
w = exp(−(1 −Lopa)dPk
2
).
(15)
To prevent gradient backpropagation, the opacity loss Lopa
is detached from Ltraj. The parameter β serves as a smooth-
ing factor, with further derivation details provided in the
Appendix D.
3.2.3. ANNOTATION
To enhance the VLM’s understanding of spatial structure
and temporal evolution, online visual annotations are ap-
plied to both FPVs and the BEV, as illustrated in Fig-
ure 1. FPVs provide intuitive first-person visual observa-
tions, while the BEV supplies the VLM with global spatial
context and historical information.
A gaze point is overlaid in the FPV to clearly indicate the
region of interest of the current frontier point and to enhance
the VLM spatial understanding, highlighting the navigation
goal and its surrounding environment. Simultaneously, the
5

<!-- page 6 -->
3DGSNav
BEV explicitly annotates the spatial layout of the frontier
point along with the guidance trajectory of the agent, estab-
lishing a correspondence between the FPV and BEV and
providing global spatial context.
To further support spatial memory and temporal reasoning,
the BEV depicts the historical trajectory of the agent in blue
and its current position as a purple triangle, providing con-
tinuous motion context. In addition, as the environment is
gradually explored in ZSON tasks, unobserved regions are
marked in purple in the FPV, guiding the VLM to differenti-
ate between known and unknown areas and make informed
exploration decisions.
3.2.4. NAVIGATION POLICY
We integrate the FPV and BEV into a single composite
image X, which is combined with CoT C and fed into
the VLM (Google, 2025) to determine the next frontier
point: P∗= F(I, X, C).
The Fast Marching Method
(FMM) (Sethian, 1996) is then used to navigate the agent
from its current position to the long-term goal.
3.3. Re-verification
Detection accuracy is critical, as it provides the agent with
the terminal signal for task completion. Methods (Yin et al.,
2024; Cao et al., 2025; Zhu et al., 2025) typically address
uncertainty in detection accuracy by driving the agent to
change its pose in order to acquire new observations. How-
ever, this strategy often incurs additional exploration costs
and degrades overall navigation efficiency.
Therefore, as illustrated in Figure 3, we adopt a real-
time detector (Wang et al., 2025a) that supports open-
vocabulary detection via text prompts to perceive po-
tential targets during navigation.
The image contain-
ing detected potential targets is provided to the action-
decision VLM (Team et al., 2026), which reasons over
the environmental structure and object occlusions to de-
cide whether further actions should be taken from the ac-
tion space at ∈{forward (0.25 m), backward(0.25 m),
turn left(10◦), turn right(10◦), leave}. Instead of physi-
cally moving the robot, these selected actions are projected
into the 3DGS space to render novel views for further target
Figure 3. Visualization of the action-decision VLM target re-
verification. The green circular bounding boxes indicate detec-
tions from the real-time detector, while the red bounding boxes
denote the final detection results.
verification.
4. Experiments
4.1. Experiment Setup
Datasets. We conduct experiments on three datasets in the
Habitat (Puig et al., 2023) simulator. HM3Dv1 (Ramakrish-
nan et al., 2021) contains 2000 episodes across 20 scenes
with 6 goal categories. HM3Dv2 (Yadav et al., 2023) in-
cludes 1000 episodes over 36 scenes with 6 goal categories.
MP3D (Chang et al., 2017) consists of 2195 episodes span-
ning 11 scenes with 21 goal categories.
Metrics. On all benchmarks, the δmax is set to 500 steps
and ds is set to 1. We report Success Rate (SR(%)) and
Success weighted by inverse Path Length (SPL(%)), which
measures navigation efficiency relative to the optimal path
length. The best results are highlighted in bold.
Implementation Details. In the Habitat setup, the robot
can choose from a discrete action set At: move forward
(0.2 m), turn left (30°), turn right (30°), look up (40°), look
down (40°), and stop. The agent’s camera is mounted at
a height of 0.88 m. The navigation planning visual lan-
guage model is Gemini3-Pro (Gemini3) (Google, 2025).
In the comparative experiments, we use Qwen3-VL-32b-
Thinking (Qwen32b) (Yang et al., 2025a) and Qwen3-VL-
235b-A22B-Thinking (Qwen235b) (Yang et al., 2025a).
For real-time object detection, we employ YOLOE (Wang
et al., 2025a) with yoloe-11m-seg weights. The action-
decision VLM is GLM-4.5v (Team et al., 2026). All simu-
lation experiments are conducted on a system equipped
with an NVIDIA GeForce RTX 4060 Ti 16 GB GPU
and an Intel i7-12700KF CPU. The 3DGS optimization
uses {λ, λ1, λ2} = {0.2, 1.0, 0.7}. The active perception
opacity threshold is set to τ = 0.3. Guidance trajectory
parameters are {∆L, ds, ω} = {4, 10, 5}. Virtual view-
point initialization uses {α, rmin, rmax} = {0.7, 1.0, 4.0}.
Free-viewpoint optimization adopts {λo, λv, λc, λt, β} =
{0.01, 1.0, 0.01, 0.1, 5} and runs for 40 iterations.
4.2. Benchmark results
Table 1 reports the quantitative results of 3DGSNav on
the HM3Dv1, HM3Dv2, and MP3D datasets with base-
lines. Compared with state-of-the-art methods, 3DGSNav
achieves an average improvement of 13.5% in SR and
32.08% in SPL. Furthermore, compared with the RL-based
ZSON (Majumdar et al., 2022) method, which also does
not rely on scene abstractions, 3DGSNav improves SR by
an average of 203.01% and SPL by 320.11% on HM3Dv1
and MP3D, demonstrating the clear advantage of the VLM-
based reasoning. In addition, relative to BeliefMapNav,
which employs voxel representations and constructs 3D se-
mantic maps, 3DGSNav achieves gains of 25.25% in SR and
6

<!-- page 7 -->
3DGSNav
Table 1. Benchmark Comparisons. Comparison of success rate of different methods on HM3Dv1, HM3Dv2, MP3D. Scene abstraction
is defined as the representation of the environment in textual representations or semantic maps.
Method
Zero-shot
Scene
Unsupervised HM3Dv1 HM3Dv2
MP3D
abstraction
SR↑SPL↑SR↑SPL↑SR↑SPL↑
Habitat-Web (Ramrakhya et al., 2022)
%
!
%
41.5 16.0
-
-
31.6
8.5
SGM (Zhang et al., 2024b)
%
!
%
60.2 30.8
-
-
37.7 14.7
PSL (Sun et al., 2024)
!
!
%
42.4 19.2
-
-
18.9
6.4
ZSON (Majumdar et al., 2022)
!
%
%
25.5 12.6
-
-
15.3
4.8
ESC (Zhou et al., 2023)
!
!
!
39.2 22.3
-
-
28.7 14.2
OpenFMNav (Kuang et al., 2024)
!
!
!
54.9 24.4
-
-
37.2 15.7
L3MVN (Yu et al., 2023)
!
!
!
50.4 23.1 36.3 15.7 34.9 14.5
VLFM (Yokoyama et al., 2024)
!
!
!
52.5 30.4 63.6 32.5 36.4 17.5
SG-NAV (Yin et al., 2024)
!
!
!
54.0 24.9 49.6 25.5 40.2 16.0
BeliefMapsNav (Zhou et al., 2025)
!
!
!
61.4 30.6
-
-
37.3 17.6
ApexNav (Zhang et al., 2025)
!
!
!
59.6 33.0 76.2 38.0 39.2 17.8
3DGSNav(Our)
!
%
!
80.0 51.79 75.0 44.19 43.63 21.31
Figure 4. Comparison of self-explanations of Gemini3-Pro and Qwen3-235b-Thinking on ZSON tasks. The left and right images
correspond to Gemini3-Pro and Qwen3-235b-Thinking, respectively. The target object is highlighted by a red bounding box.
Table 2. Comparison of Different VLMs. Random refers to a
strategy where no VLM planning is performed, and frontier points
are selected randomly.
Method
HM3Dv1
HM3Dv2
MP3D
SR↑
SPL↑
SR↑
SPL↑
SR↑
SPL↑
Random
50.0
16.85 64.81 32.67
25.0
4.91
Qwen32b
61.67 32.47 66.67 34.44 29.54 13.95
Qwen235b
65.0
37.6
68.51 37.15 31.81 14.61
3DGSNav(Our)
80.0
51.79
75.0
44.19 43.63 21.31
51.65% in SPL. These results indicate that using 3DGS as
a VLM accessible memory representation is more effective
for spatial reasoning and navigation than scene abstractions.
4.3. Ablation Study
Due to space limitations, detailed analyses of VLM per-
formance on ZSON tasks under our paradigm, as well as
key findings and insights, are provided in the Appendix.
Specifically, a bottleneck analysis of VLM performance
on the ZSON task is presented in Appendix E, the role of
Table 3. Ablation study on viewpoints and annotations. The
virtual viewpoint is defined as a view directly oriented towards the
frontier point from the agent current position, without viewpoints
initialization or optimization.
Virtual
Free-viewpoint Historical Gaze
HM3Dv1
viewpoints
optimization
trajectory point SR↑SPL↑
%
!
!
!
76.66 38.54
!
%
!
!
78.33 47.7
!
!
%
!
77.96 48.47
!
!
!
%
78.0 48.59
!
!
!
!
80.0 51.79
viewpoints and annotations in VLM reasoning is discussed
in Appendix F, and the analysis of target re-verification is
included in Appendix G.
4.3.1. ANALYSIS OF DIFFERENT VLMS
To evaluate the effectiveness of VLMs in ZSON tasks,
Table 2 reports results from replacing the planning
7

<!-- page 8 -->
3DGSNav
VLM (Google, 2025) with a random frontier selection strat-
egy and two open-source VLMs. Gemini3 significantly
outperforms the open-source Qwen235b model, achieving a
20.14% improvement in SR and a 31.25% improvement in
SPL.
To
better
understand
this
performance
gap,
self-
explanations (Yang et al., 2025b) are employed to
analyze the VLM reasoning process. In Figure 4, Gemini3
(left) exhibits advanced image and spatial understanding, as
evidenced by the strong reasoning of ”the space is narrow
and does not match an environment suitable for placing
a couch”. Furthermore, it can infer spatial relationships
between different frontier points through the BEV, rather
than performing isolated analysis or making inaccurate
inferences. (e.g., Points 0 and 1 correspond to different
FPV viewpoints but can be inferred to belong to the same
room based on the BEV.) Moreover, the final prioritization
of the summary output further demonstrates that the current
model possesses strong planning capabilities.
Compared to random frontier point selection, 3DGSNav
improves SR by 42.07% and SPL by 115.48%, while
Qwen235b improves SR by 18.24% and SPL by 64.17%.
However, its performance gains are smaller than those of
Gemini3, mainly due to deficiencies in fundamental image
understanding (e.g., ”FPV clearly shows a couch and a
large purple region.”) that lead to deviations in subsequent
reasoning. Moreover, Figure 5 presents the distribution of
failure types, showing that reasoning failures account for
only 9.5%, the lowest among all failure types. This find-
ing further demonstrates the effectiveness of our paradigm
in unlocking the potential of VLMs for ZSON tasks. For
a more detailed analysis of the VLM, please refer to the
Appendix E.
4.3.2. VIEWPOINTS AND ANNOTATIONS
In Table 3, we conduct an ablation study to evaluate the
effectiveness of two key components: free viewpoints and
visual annotations. Without virtual viewpoints, frontier per-
ception is more likely to be occluded by walls or other
obstacles, resulting in a 13.25% decrease in SPL. The anno-
tated historical trajectory provides spatiotemporal guidance
that reduces redundant backtracking, thereby improving nav-
Table 4. Ablation study on re-verification, planner CoT, and
frontier clustering.
Re-verifi. Re-verifi. Planner
Frontier
HM3Dv1
model
action
CoT
clustering SR↑SPL↑
%
!
!
!
53.75 33.19
!
%
!
!
75.0 47.34
!
!
%
!
77.5 47.14
!
!
!
%
62.5 37.68
!
!
!
!
80.0 51.79
Figure 5. Failure cause statistics. Distribution of different failure
types.
igation efficiency. More detailed analyses of viewpoints and
annotations are provided in Appendix F.
4.3.3. RE-VERIFICATION
In Table 4, we conduct an ablation study to evaluate the
effectiveness of the re-verification module, focusing on both
the integration of the VLM and its action decision mech-
anism. Removing the re-verification VLM (i.e., relying
solely on the real-time detector) results in a 32.81% drop
in SR. Compared to the variant without VLM-driven ac-
tion decisions, enabling action selection allows the VLM
to perform more fine-grained reasoning over detection re-
sults, effectively exhibiting a CoT-like reasoning process.
Moreover, by projecting the selected actions into the 3DGS
representation to render novel viewpoints of the target, the
VLM can mitigate misrecognition caused by partial or lo-
cal observations, leading to more robust target verification.
Additional visualizations are provided in Appendix G.
4.3.4. PLANNER COT AND FRONTIER CLUSTERING
In Table 4, the results show that CoT is essential for effective
reasoning in ZSON tasks. Moreover, frontier point cluster-
ing aggregates dispersed frontiers to enrich free-viewpoint
rendering, yielding to a 21.87% improvement in SR. Details
of the CoT prompts are provided in Appendix J.
4.4. Real-robot Experiments
Our method achieves a 69.44% SR on a quadruped robot in
real-world office and hotel environments. More details and
results are provided in Appendix I.
5. Conclusion
In this work, we presented 3DGSNav, a novel framework for
ZSON that leverages 3DGS as a memory representation to
enhance VLM reasoning. By combining active perception,
trajectory-guided free-viewpoint rendering, and structured
visual prompts with CoT prompting, our method enables
8

<!-- page 9 -->
3DGSNav
efficient spatial reasoning and planning. 3DGSNav achieves
strong performance on both real-world robotic platforms and
public benchmarks. Future work will focus on improving
generality to support a broader range of tasks.
References
Cao, Y., Zhang, J., Yu, Z., Liu, S., Qin, Z., Zou, Q., Du,
B., and Xu, K. Cognav: Cognitive process modeling for
object goal navigation with llms. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pp. 9550–9560, 2025.
Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner,
M., Savva, M., Song, S., Zeng, A., and Zhang, Y. Matter-
port3d: Learning from rgb-d data in indoor environments.
arXiv preprint arXiv:1709.06158, 2017.
Chen, J., Li, G., Kumar, S., Ghanem, B., and Yu, F. How
to not train your dragon: Training-free embodied object
goal navigation with semantic frontiers. Proceedings of
Robotics: Science and Systems, 2023.
Chen, S., Chen, X., Zhang, C., Li, M., Yu, G., Fei, H., Zhu,
H., Fan, J., and Chen, T. Ll3da: Visual interactive in-
struction tuning for omni-3d understanding reasoning and
planning. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pp. 26428–
26438, 2024a.
Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E.,
Zhu, J., Ye, S., Tian, H., Liu, Z., et al. Expanding per-
formance boundaries of open-source multimodal models
with model, data, and test-time scaling. arXiv preprint
arXiv:2412.05271, 2024b.
Dang, R., Wang, L., He, Z., Su, S., Tang, J., Liu, C., and
Chen, Q. Search for or navigate to? dual adaptive thinking
for object navigation. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pp. 8250–
8259, 2023.
Deng, J., He, T., Jiang, L., Wang, T., Dayoub, F., and Reid,
I. 3d-llava: Towards generalist 3d lmms with omni su-
perpoint transformer. In Proceedings of the Computer
Vision and Pattern Recognition Conference, pp. 3772–
3782, 2025.
Fan, Z., Zhang, J., Li, R., Zhang, J., Chen, R., Hu, H., Wang,
K., Qu, H., Wang, D., Yan, Z., et al. Vlm-3r: Vision-
language models augmented with instruction-aligned 3d
reconstruction. arXiv preprint arXiv:2505.20279, 2025.
Fu, R., Liu, J., Chen, X., Nie, Y., and Xiong, W. Scene-llm:
Extending language model for 3d visual reasoning. In
2025 IEEE/CVF Winter Conference on Applications of
Computer Vision (WACV), pp. 2195–2206. IEEE, 2025.
Google.
A new era of intelligence with gemini 3,
2025. URL https://blog.google/products/
gemini/gemini-3.
Hong, Y., Zhen, H., Chen, P., Zheng, S., Du, Y., Chen,
Z., and Gan, C. 3d-llm: Injecting the 3d world into
large language models. Advances in Neural Information
Processing Systems, 36:20482–20494, 2023.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis, G. 3d
gaussian splatting for real-time radiance field rendering.
ACM Trans. Graph., 42(4):139–1, 2023.
Kuang, Y., Lin, H., and Jiang, M. Openfmnav: Towards
open-set zero-shot object navigation via vision-language
foundation models. arXiv preprint arXiv:2402.10670,
2024.
Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H.,
Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.
Llava-
onevision: Easy visual task transfer.
arXiv preprint
arXiv:2408.03326, 2024.
Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping
language-image pre-training with frozen image encoders
and large language models. In International conference
on machine learning, pp. 19730–19742. PMLR, 2023.
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines
with visual instruction tuning. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pp. 26296–26306, 2024.
Majumdar, A., Aggarwal, G., Devnani, B., Hoffman, J.,
and Batra, D. Zson: Zero-shot object-goal navigation
using multimodal goal embeddings. Advances in Neural
Information Processing Systems, 35:32340–32352, 2022.
Mousavian, A., Toshev, A., Fiˇser, M., Koˇseck´a, J., Wahid,
A., and Davidson, J. Visual representations for semantic
target driven navigation. In 2019 International Confer-
ence on Robotics and Automation (ICRA), pp. 8846–8852.
IEEE, 2019.
Puig, X., Undersander, E., Szot, A., Cote, M. D., Yang, T.-
Y., Partsey, R., Desai, R., Clegg, A. W., Hlavac, M., Min,
S. Y., et al. Habitat 3.0: A co-habitat for humans, avatars
and robots. arXiv preprint arXiv:2310.13724, 2023.
Qiao, Y., Lyu, W., Wang, H., Wang, Z., Li, Z., Zhang,
Y., Tan, M., and Wu, Q. Open-nav: Exploring zero-
shot vision-and-language navigation in continuous envi-
ronment with open-source llms. In 2025 IEEE Interna-
tional Conference on Robotics and Automation (ICRA),
pp. 6710–6717. IEEE, 2025.
Ramakrishnan,
S. K.,
Gokaslan,
A.,
Wijmans,
E.,
Maksymets, O., Clegg, A., Turner, J., Undersander,
9

<!-- page 10 -->
3DGSNav
E., Galuba, W., Westbury, A., Chang, A. X., et al.
Habitat-matterport 3d dataset (hm3d): 1000 large-scale
3d environments for embodied ai.
arXiv preprint
arXiv:2109.08238, 2021.
Ramrakhya, R., Undersander, E., Batra, D., and Das, A.
Habitat-web: Learning embodied object-search strategies
from human demonstrations at scale. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pp. 5173–5183, 2022.
Ramrakhya, R., Batra, D., Wijmans, E., and Das, A. Pirlnav:
Pretraining with imitation and rl finetuning for objectnav.
In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pp. 17896–17906,
2023.
Sethian, J. A. A fast marching level set method for mono-
tonically advancing fronts. proceedings of the National
Academy of Sciences, 93(4):1591–1595, 1996.
Shi, J., Yagi, S., Yamamori, S., and Morimoto, J. Llm-
guided zero-shot visual object navigation with building
semantic map. In 2025 IEEE/SICE International Sympo-
sium on System Integration (SII), pp. 1274–1279. IEEE,
2025.
Sun, X., Liu, L., Zhi, H., Qiu, R., and Liang, J. Prioritized
semantic learning for zero-shot instance navigation. In
European Conference on Computer Vision, pp. 161–178.
Springer, 2024.
Team, V., Hong, W., Yu, W., Gu, X., Wang, G., Gan, G.,
Tang, H., Cheng, J., Qi, J., Ji, J., Pan, L., Duan, S., Wang,
W., Wang, Y., Cheng, Y., He, Z., Su, Z., Yang, Z., Pan, Z.,
Zeng, A., Wang, B., Chen, B., Shi, B., Pang, C., Zhang,
C., Yin, D., Yang, F., Chen, G., Li, H., Zhu, J., Chen,
J., Xu, J., Xu, J., Chen, J., Lin, J., Chen, J., Wang, J.,
Chen, J., Lei, L., Gong, L., Pan, L., Liu, M., Xu, M.,
Zhang, M., Zheng, Q., Lyu, R., Tu, S., Yang, S., Meng,
S., Zhong, S., Huang, S., Zhao, S., Xue, S., Zhang, T.,
Luo, T., Hao, T., Tong, T., Jia, W., Li, W., Liu, X., Zhang,
X., Lyu, X., Zhang, X., Fan, X., Huang, X., Xue, Y.,
Wang, Y., Wang, Y., Wang, Y., An, Y., Du, Y., Huang,
Y., Niu, Y., Shi, Y., Wang, Y., Wang, Y., Yue, Y., Li,
Y., Liu, Y., Zhang, Y., Wang, Y., Zhang, Y., Xue, Z.,
Du, Z., Hou, Z., Wang, Z., Zhang, P., Liu, D., Xu, B.,
Li, J., Huang, M., Dong, Y., and Tang, J. Glm-4.5v
and glm-4.1v-thinking: Towards versatile multimodal
reasoning with scalable reinforcement learning, 2026.
URL https://arxiv.org/abs/2507.01006.
Wang, A., Liu, L., Chen, H., Lin, Z., Han, J., and Ding, G.
Yoloe: Real-time seeing anything, 2025a. URL https:
//arxiv.org/abs/2503.07465.
Wang, H., Zhao, Y., Wang, T., Fan, H., Zhang, X., and
Zhang, Z. Ross3d: Reconstructive visual instruction tun-
ing with 3d-awareness. arXiv preprint arXiv:2504.01901,
2025b.
Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P.
Image quality assessment: from error visibility to struc-
tural similarity. IEEE transactions on image processing,
13(4):600–612, 2004.
Wen, C., Huang, Y., Huang, H., Huang, Y., Yuan, S., Hao,
Y., Lin, H., Liu, Y.-S., and Fang, Y. Zero-shot object
navigation with vision-language models reasoning. In
International Conference on Pattern Recognition, pp. 389–
404. Springer, 2025.
Werby, A., Huang, C., B¨uchner, M., Valada, A., and Bur-
gard, W. Hierarchical open-vocabulary 3d scene graphs
for language-grounded robot navigation. In First Work-
shop on Vision-Language Models for Navigation and
Manipulation at ICRA 2024, 2024.
Yadav, K., Ramrakhya, R., Ramakrishnan, S. K., Gervet, T.,
Turner, J., Gokaslan, A., Maestre, N., Chang, A. X., Batra,
D., Savva, M., et al. Habitat-matterport 3d semantics
dataset. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 4927–
4936, 2023.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng,
B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu,
D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin,
H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang,
J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang,
K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang,
P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo,
S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang,
X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan,
Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and
Qiu, Z. Qwen3 technical report, 2025a. URL https:
//arxiv.org/abs/2505.09388.
Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L.,
and Xie, S. Thinking in space: How multimodal large
language models see, remember, and recall spaces, 2025b.
URL https://arxiv.org/abs/2412.14171.
Ye, J., Batra, D., Das, A., and Wijmans, E. Auxiliary tasks
and exploration enable objectgoal navigation. In Pro-
ceedings of the IEEE/CVF international conference on
computer vision, pp. 16117–16126, 2021.
Yin, H., Xu, X., Wu, Z., Zhou, J., and Lu, J. Sg-nav: Online
3d scene graph prompting for llm-based zero-shot object
navigation. Advances in neural information processing
systems, 37:5285–5307, 2024.
10

<!-- page 11 -->
3DGSNav
Yokoyama, N., Ha, S., Batra, D., Wang, J., and Bucher,
B. Vlfm: Vision-language frontier maps for zero-shot
semantic navigation. In 2024 IEEE International Con-
ference on Robotics and Automation (ICRA), pp. 42–48.
IEEE, 2024.
Yu, B., Kasaei, H., and Cao, M. L3mvn: Leveraging large
language models for visual target navigation. In 2023
IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), pp. 3554–3560. IEEE, 2023.
Yu, H., Li, W., Wang, S., Chen, J., and Zhu, J. Inst3d-lmm:
Instance-aware 3d scene understanding with multi-modal
instruction tuning. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pp. 14147–14157,
2025.
Zhang, J., Dai, L., Meng, F., Fan, Q., Chen, X., Xu, K., and
Wang, H. 3d-aware object goal navigation via simultane-
ous exploration and identification. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 6672–6682, June 2023.
Zhang, L., Zhang, Q., Wang, H., Xiao, E., Jiang, Z., Chen,
H., and Xu, R. Trihelper: Zero-shot object navigation
with dynamic assistance. In 2024 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS), pp.
10035–10042. IEEE, 2024a.
Zhang, M., Du, Y., Wu, C., Zhou, J., Qi, Z., Ma, J., and
Zhou, B. Apexnav: An adaptive exploration strategy for
zero-shot object navigation with target-centric semantic
fusion. arXiv preprint arXiv:2504.14478, 2025.
Zhang, S., Yu, X., Song, X., Wang, X., and Jiang, S. Imagine
before go: Self-supervised generative map for object goal
navigation. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 16414–
16425, 2024b.
Zheng, D., Huang, S., and Wang, L. Video-3d llm: Learning
position-aware video representation for 3d scene under-
standing. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pp. 8995–9006, 2025.
Zhou, G., Hong, Y., and Wu, Q. Navgpt: Explicit reasoning
in vision-and-language navigation with large language
models. In Proceedings of the AAAI Conference on Artifi-
cial Intelligence, volume 38, pp. 7641–7649, 2024.
Zhou, K., Zheng, K., Pryor, C., Shen, Y., Jin, H., Getoor,
L., and Wang, X. E. ESC: Exploration with soft com-
monsense constraints for zero-shot object navigation. In
Proceedings of the 40th International Conference on Ma-
chine Learning, volume 202 of Proceedings of Machine
Learning Research, pp. 42829–42842. PMLR, 23–29 Jul
2023.
Zhou, Z., Hu, Y., Zhang, L., Li, Z., and Chen, S. Beliefmap-
nav: 3d voxel-based belief map for zero-shot object navi-
gation. arXiv preprint arXiv:2506.06487, 2025.
Zhu, H., Li, Z., Liu, Z., Wang, W., Zhang, J., Francis, J., and
Oh, J. Strive: Structured representation integrating vlm
reasoning for efficient object navigation. arXiv preprint
arXiv:2505.06729, 2025.
11

<!-- page 12 -->
3DGSNav
A. Appendix Outline
In these supplementary materials, we provide:
• Panoramic Rendering B;
• Exploration Map Generation C;
• Role of β in Guidance Trajectory Loss D;
• Analysis of VLM Failure Cases E;
• Analysis of Viewpoints and Visual Annotations F;
• Analysis of Re-verification Cases G;
• Runtime Analysis H;
• Real-World Experimental Details I;
• Chain-of-Thought Prompt Design J;
B. Panoramic Rendering
We exploit the flexibility of the virtual camera, which allows arbitrary adjustment of intrinsic parameters, by setting the
horizontal field of view (hFoV) to 120° and the vertical field of view (vFoV) to 150°. To achieve full panoramic coverage,
we render 2π/hFoV evenly spaced views centered around the agent. Each rendered image has a resolution of 120 × 150
pixels. The corresponding camera intrinsic matrix is computed as follows:
K =


fx
0
cx
0
fy
cy
0
0
1

, fx = fy =
W
2 tan( θh
2 ), cx = cy = W
2
(16)
The panoramic image is obtained by horizontally concatenating all rendered views, as shown in Figure 2. The same
procedure is applied to generate panoramic depth fields and opacity fields.
C. Exploration Map Generation
We remove Gaussian primitives corresponding to the ceiling and the floor based on the height of the agent. A top down
rendering is then performed to obtain the global opacity map of the scene. Regions with opacity values below a threshold τ
are regarded as obstacles. Meanwhile, the floor is identified and treated as a traversable area. Finally, the obstacle map and
the traversable map are combined to construct the exploration map.
D. Role of β in Guidance Trajectory Loss
To better understand the role of β in the optimization process, we derive the formulation:
Ltraj = −1
β ln
X
i
 we−βdi
, di = ∥t −pi∥2.
(17)
For simplicity, we set w = 1, and the loss becomes:
Ltraj = −1
β ln
 X
i
e−βdi
!
.
(18)
We compute the gradient of the loss function with respect to the camera position t:
∂Ltraj
∂t
= −1
β ·
1
P
j e−βdj ·
X
i

e−βdi · (−β)∂di
∂t

.
(19)
12

<!-- page 13 -->
3DGSNav
Figure 6. Failure Case Statistics and Reasoning Failure Breakdown. Failure cases include reasoning failure, detection failure (e.g.,
ambiguous categories like gaming chair vs. chair, multiple targets like multiple chairs, and detection errors such as false positives/negatives),
different floor, and other issues such as sensor failure (invalid image/depth) and pose failure. The six categories in the reasoning failure
breakdown correspond to the target object classes in HM3Dv1 and HM3Dv2. The ratios denote the failure rate specifically for episodes
involving each target object.
Since ∂di
∂t = t−pi
di , which corresponds to the unit vector pointing from point pi to the camera position t, we can cancel the
factor β to obtain:
∂Ltraj
∂t
=
X
i
 
e−βdi
P
j e−βdj
!
t −pi
di
.
(20)
By defining αi =
e−βdi
P
j e−βdj , the gradient can be written as:
∂Ltraj
∂t
=
X
i
αi · ui,
(21)
where ui denotes the unit direction vector.
The coefficient αi follows a softmax distribution, where β controls the concentration of the distribution. The hyperparameter
β serves as a smoothing factor that modulates the sparsity of the influence of trajectory points. By adjusting β, we can
transition the optimization from a global centroid attraction (β →0) to a local point-wise alignment (β →∞). Therefore,
a moderate β value ensures a smooth optimization landscape, allowing the camera pose to converge robustly toward the
manifold defined by the reference trajectory T .
E. Analysis of VLM Failure Cases
A statistical analysis of different failure types is presented in Figure 6. The results indicate that the primary cause of task
failure lies in the final target detection stage, which can be further divided into three categories. Among these categories,
category ambiguity and multi-target ambiguity originate from the simulator’s decision mechanism.
Specifically, category ambiguity occurs when the issued instruction refers to an object category such as “chair.” Although the
environment contains valid subcategories, including wooden chairs and gaming chairs, and the agent correctly recognizes
them as semantically consistent with the instruction (as shown in Figure 12, middle), the simulator still labels the episode as
a failure.
In addition, multi-target ambiguity refers to scenarios involving multiple instances of the same object category (e.g., chairs
near a bar counter and chairs in a kitchen). In such cases, the simulator considers only the kitchen chair as the correct match,
while treating the chair near the bar counter as an incorrect target. Because the instruction provided to the simulator specifies
only the category “chair” without additional qualifiers, the task fails due to the inability to disambiguate among multiple
valid targets. Furthermore, detection errors account for 19% of failures, primarily arising from constrained local perception
13

<!-- page 14 -->
3DGSNav
Figure 7. VLM reasoning failure example 1. The target object is a plant and is highlighted with a red bounding box.
and degraded novel-view quality after action execution. This result further indicates that high-quality and robust novel-view
synthesis is not only critical for planning and decision-making, but also essential for ensuring accurate and reliable target
detection.
Notably, reasoning failures constitute the smallest proportion of overall failures, suggesting that our proposed paradigm
successfully unlocks the potential of the VLM for ZSON. Finally, because the navigation policy is based on two-dimensional
information, 3DGSNav is currently unable to perform cross-floor navigation, which remains an important direction for
future improvement.
E.1. Analysis of VLM Reasoning Failures
To quantify and identify the bottlenecks of the proposed paradigm in ZSON tasks, we conduct an in-depth analysis of several
failure cases.
E.1.1. WEAK TARGET PRIORS AND THE BACKTRACKING ISSUE
When target objects lack distinctive spatial priors (e.g., plants appearing in arbitrary locations, unlike beds confined to
bedrooms), the agent must rely on weak cues such as lighting and layout. Although this strategy is reasonable under
ambiguity (Fig. 7 A), historical observations (blue trajectory in Fig. 7) often dominate decision-making.
When multiple cues provide comparable evidence, backtracking becomes penalized by the reward mechanism. As a result,
the agent tends to avoid revisiting explored areas and continues forward, increasing the likelihood of failure. This trend is
corroborated by the reasoning failure breakdown in Fig. 6. Category-wise analysis further reveals that most failures occur
on weak-prior targets, such as plants, whereas objects with strong spatial and functional priors (e.g., TVs, beds, and toilets)
exhibit near-zero failures.
E.1.2. VISUAL PERCEPTION ISSUE
In Figure 8 C, a hallucination occurs at Frontier 5, which corresponds to Frontier 3 in Figure 8 B, where the agent infers a
potential presence of a chair.
14

<!-- page 15 -->
3DGSNav
Figure 8. VLM reasoning failure example 2. The target object is a chair and is highlighted with a red bounding box.
E.1.3. EXPLORATION ISSUE
Active exploration is a critical capability that agents must possess in ZSON tasks. However, extensive experimental evidence
suggests that VLMs tend to exhibit a conservative bias when exploring unknown environments, preferring regions with even
weak visual or semantic cues over proactively investigating entirely unexplored areas. The hallucination observed in Figure 8
may stem from insufficient environmental information provided by other frontier points, which can cause the VLM to
produce seemingly plausible yet ultimately incorrect justifications when selecting Frontier 5. Furthermore, Figures 9 and 10
further highlight the challenges VLMs face in leveraging BEV information to assess whether frontier points correspond to
previously unexplored spaces.
F. Analysis of Viewpoints and Visual Annotations
F.1. Viewpoint Analysis
Figure 11 presents an ablation study examining the impact of virtual viewpoints. For Frontiers 1 and 2, frontier information
is entirely occluded by walls when virtual viewpoints are not employed. In contrast, virtual viewpoint rendering produces
Figure 9. VLM reasoning failure example 3. The target object is a bed.
15

<!-- page 16 -->
3DGSNav
Figure 10. VLM reasoning failure example 4. The target object is a couch.
Table 5. Ablation study on viewpoints and annotations. The virtual viewpoint is defined as a view directly oriented towards the frontier
point from the agent current position, without viewpoints initialization or optimization.
Virtual
Free-viewpoint
Historical
Gaze
HM3Dv1
HM3Dv2
MP3D
viewpoints
optimization
trajectory
point
SR
SPL
SR
SPL
SR
SPL
%
!
!
!
76.66
38.54
72.22
38.26
30.3
16.68
!
%
!
!
78.33
47.7
74.07
36.72
39.39
18.32
!
!
%
!
77.96
48.47
72.22
37.51
36.36
17.32
!
!
!
%
78.0
48.59
72.22
38.52
42.42
20.06
!
!
!
!
80.0
51.79
75.0
44.19
43.63
21.31
FPVs that successfully reveal relevant frontier cues. With access to these additional visual information, the VLM can more
accurately infer the room type associated with each frontier point.
Furthermore, Table 5 indicates that removing virtual viewpoints results in the largest performance degradation in SPL among
all evaluated components, with a reduction of up to 25.47%. This substantial decline in navigation efficiency causes many
episodes to exceed the maximum step limit, ultimately leading to a 10.59% drop in SR.
Free-viewpoint optimization helps overcome constraints imposed by limited initial pose flexibility, thereby increasing
the likelihood of successfully observing informative frontier regions. Overall, these findings demonstrate that renderings
generated via 3DGS can be effectively interpreted and exploited by VLMs to support more accurate reasoning and efficient
navigation.
F.2. Annotation Visualization Analysis
F.2.1. GAZE POINT
As shown in Figure 12 (left), without gaze-point guidance, the VLM processes the entire image holistically. Distracted
by irrelevant objects such as the television and mirror, it incorrectly classifies Frontier 3 as part of a living room and
subsequently bases its object-likelihood reasoning on this erroneous room type. In contrast, the result in Figure 12 (right)
labels the room type as uncertain, preventing incorrect room-based inference. This comparison demonstrates that gaze
points effectively steer the VLM’s attention toward the spatial region surrounding the frontier, thereby improving its spatial
understanding.
Similarly, in Figure 13 (left), the absence of gaze points makes it difficult for the VLM to infer the correct navigation
direction from the FPV, causing it to mistakenly interpret Frontier 1 as leading into a room. By contrast, Figure 13 (right)
shows that gaze guidance enables the VLM to correctly reason that Frontier 1 corresponds to a hallway rather than an
adjacent room.
Overall, gaze points provide a focused spatial anchor for VLM analysis, replacing diffuse scene-level interpretation with
specific spatial reasoning. Table 5 further quantifies the benefits of gaze-point guidance. Moreover, without gaze points, the
VLM cannot reliably infer the corresponding focus region in the FPV from BEV frontier information alone, underscoring
the critical role of gaze-guided spatial grounding.
16

<!-- page 17 -->
3DGSNav
Figure 11. Ablation Study on Virtual Viewpoints. The left image shows rendering without virtual viewpoints, while the right image
shows rendering based on optimized virtual viewpoints. The target object is a chair and is highlighted with a red bounding box.
Figure 12. Presence of gaze point visualization example 1. The red point is the gaze point.
F.2.2. UNOBSERVED REGION REPRESENTATION
The representation of unobserved regions in FPVs has a substantial impact on VLM scene understanding. Empirically, using
black or gray to denote unseen areas consistently leads to the failure cases shown in Figure 14. In particular, the VLM
tends to misinterpret black regions as physical obstacles or hallucinate lighting-based spatial cues, resulting in unstable and
unreliable predictions.
To mitigate this issue, we represent unobserved regions in purple in our experiments. This color is salient to VLMs while
avoiding unintended associations with lighting variations or structural obstacles, thereby enabling more stable and accurate
scene interpretation.
G. Analysis of Re-verification Cases
To prevent the action-decision VLM from excessively influencing the agent’s behavior through overactive control, we
impose an upper limit of five action decisions per verification cycle.
Figures 15 and 16 illustrate the VLM-driven active target re-verification mechanism. Figure 15 presents representative cases
17

<!-- page 18 -->
3DGSNav
Figure 13. Presence of gaze point visualization example 2. The red point is the gaze point.
in which the VLM-based action decision module performs target re-verification and successfully corrects false positives
generated by the real-time detector. By reasoning over the target’s spatial location in the image, the VLM selects appropriate
actions that induce viewpoint transformations, thereby expanding target visibility and reducing misdetections caused by
limited local perception.
H. Runtime Analysis
Table 6 summarizes the runtime of the major system components. The cost of free-viewpoint optimization is evaluated per
optimization iteration, the guidance trajectory cost is computed per trajectory, and the frontier clustering cost corresponds to
the total time required to group all frontier points.
I. Real-World Experimental Details
I.1. Experiment Setup
We employ the Go2 quadruped robot as the mobile platform. The system is powered by an NVIDIA Jetson AGX Orin
embedded computing unit, featuring a 12-core ARM v8.2 64-bit CPU and 32GB of memory, running Ubuntu 20.04 and
the Robot Operating System 2 (ROS2). The robot is equipped with an Intel RealSense D455 RGB-D camera and a 6-DoF
robotic manipulator. The robot’s ego-motion is estimated using legged odometry.
To mitigate motion-induced jitter during locomotion, a pose-following control strategy is applied at the manipulator
end-effector, allowing it to adaptively track the quadruped’s body posture in real time. The navigation policy is implemented
within the Nav2 framework.
Figure 14. FPV unobserved representation.
18

<!-- page 19 -->
3DGSNav
Figure 15. Visualization of the action-decision VLM target re-verification. The green circular bounding boxes indicate detections from
the real-time detector, while the red bounding boxes denote the final detection results.
I.2. Real-World Experimental Results
We design six groups of experiments with progressively increasing difficulty, as illustrated in Figures 19, 18, and 17. Table 7
presents the quantitative results, reporting the number of successful trials out of six runs for each experimental group.
Overall, the results demonstrate the robustness and effectiveness of our proposed method in real-world settings.
I.3. Real-World Experimental Analysis
I.3.1. MOTION CONTROL AND SYSTEM STABILITY ISSUES
First, the quadruped robot’s native locomotion policy is not optimized to handle the additional payload introduced by the
manipulator. This results in a shifted center of mass, which in turn causes deviations in the walking trajectory. Second, severe
vibrations during locomotion lead to motion blur in the visual input. Although a posture-following strategy is employed to
stabilize the camera, its effectiveness remains limited under dynamic motion. Furthermore, the high latency of the VLM
API forces the motors to maintain static postures for prolonged periods, which can cause overheating and eventually trigger
system protection mechanisms.
To address these challenges, future work will focus on tightly integrating manipulator-based active perception with the
quadruped’s locomotion policy. This integration aims to achieve unified and coordinated control over perception and motion,
Table 6. System runtime. Average runtime of each component for a single run on an RTX 4060 Ti
Component
Free-viewpoint
Guidance
Frontier
optimization
trajectory
clustering
Time(ms)
115
2.4
6.3
19

<!-- page 20 -->
3DGSNav
Figure 16. Visualization of a long-horizon sequence of VLM-based action decisions for target re-verification. The green circular
bounding boxes indicate detections from the real-time detector, while the red bounding boxes denote the final detection results.
thereby mitigating payload imbalance, improving stability, and reducing the risk of overheating.
I.3.2. VISUAL PERCEPTION LIMITATIONS
Multiple experiments suggest that the RGB-D camera is prone to perception failures when encountering glass surfaces,
strong specular reflections, and highly hollow or open-structured environments. These failures degrade the rendering quality
of FPVs, thereby impairing the reasoning performance of the VLM and compromising navigation safety.
J. Chain-of-Thought prompt Design
Figures 20, 21, 22, 23, and 24 illustrate the key components of the CoT prompts used in our approach, including the input
formulation, intermediate reasoning steps, scoring mechanism, and final output generation.
Table 7. Real-World Experiments. Failure cases include planning errors, collisions, and detection errors.
Easy
Medium
Hard
Scenario
Office
Hotel
Hotel
Hotel
Hotel
Hotel
Target
chair
sofa
plant
chair
toilet
slippers
Success/Total
5/6
4/6
5/6
4/6
4/6
3/6
20

<!-- page 21 -->
3DGSNav
Figure 17. 3DGSNav Real-World Demonstration (Hard). The robot successfully navigates to locate the toilet. The reasoning process is
simplified in the figure.
Figure 18. 3DGSNav Real-World Demonstration (Medium). The robot successfully navigates to locate the sofa. The reasoning process
is simplified in the figure.
21

<!-- page 22 -->
3DGSNav
Figure 19. 3DGSNav Real-World Demonstration (Easy). The robot successfully navigates to locate the chair. The reasoning process is
simplified in the figure.
Figure 20. CoT input description.
22

<!-- page 23 -->
3DGSNav
Figure 21. CoT reasoning part 1.
Figure 22. CoT reasoning part 2.
23

<!-- page 24 -->
3DGSNav
Figure 23. CoT scoring part.
Figure 24. CoT output.
24
