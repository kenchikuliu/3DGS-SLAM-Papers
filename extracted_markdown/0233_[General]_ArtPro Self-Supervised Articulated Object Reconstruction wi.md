<!-- page 1 -->
ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive
Integration of Mobility Proposals
Xuelu Li1∗
Zhaonan Wang1∗
Xiaogang Wang2†
Lei Wu1
Manyi Li1†
Changhe Tu1
1Shandong University
2Southwest University
{xtluli, d wzn}@mail.sdu.edu.cn
wangxiaogang@swu.edu.cn
{i lily, manyili, chtu}@sdu.edu.cn
Abstract
Reconstructing articulated objects into high-fidelity dig-
ital twins is crucial for applications such as robotic manip-
ulation and interactive simulation. Recent self-supervised
methods using differentiable rendering frameworks like 3D
Gaussian Splatting remain highly sensitive to the initial
part segmentation. Their reliance on heuristic clustering
or pre-trained models often causes optimization to converge
to local minima, especially for complex multi-part objects.
To address these limitations, we propose ArtPro, a novel
self-supervised framework that introduces adaptive integra-
tion of mobility proposals. Our approach begins with an
over-segmentation initialization guided by geometry fea-
tures and motion priors, generating part proposals with
plausible motion hypotheses. During optimization, we dy-
namically merge these proposals by analyzing motion con-
sistency among spatial neighbors, while a collision-aware
motion pruning mechanism prevents erroneous kinematic
estimation. Extensive experiments on both synthetic and
real-world objects demonstrate that ArtPro achieves robust
reconstruction of complex multi-part objects, significantly
outperforming existing methods in accuracy and stability.
1. Introduction
Articulated objects such as cabinets, laptops, and scissors
are ubiquitous in our daily lives. Creating their high-fidelity
digital twins, including accurate reconstructions of part ge-
ometry, appearance, and kinematic structure, is the corner-
stone for enabling robotic manipulation, virtual reality, and
interactive scene simulation. However, this task remains
highly challenging, as it requires jointly solving a set of
intertwined sub-problems: part segmentation, motion pa-
rameter estimation, as well as high-quality appearance and
geometry reconstruction.
Existing reconstruction approaches largely fall into two
∗Equal contribution. †Corresponding author.
𝑅𝑒𝑐𝑜𝑛.
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝐺𝑇
𝑂𝑢𝑟𝑠
𝐴𝑟𝑡𝐺𝑆
𝑅𝑒𝑛𝑑𝑒𝑟𝑒𝑑
𝑆𝑡𝑎𝑟𝑡 𝐹𝑟𝑎𝑚𝑒
𝐸𝑛𝑑 𝐹𝑟𝑎𝑚𝑒
Figure 1. Existing methods like ArtGS [33] are highly sensitive
to initial part segmentation, leading to inaccurate motion and ge-
ometry. Our method, ArtPro, leverages a prior-guided mobility
initialization and adaptively merges mobility proposals during op-
timization, achieving robust reconstruction of complex multi-part
articulated objects.
paradigms. On one hand, the feed-forward reconstruction
methods [4, 5, 10, 19, 39, 41, 48] leverage powerful data-
driven priors from diffusion or vision-language models to
directly predict articulated structures from inputs like im-
ages or text, achieving remarkable inference speed. How-
ever, their generality is achieved either through coarse ge-
ometric abstractions or by being constrained to their train-
ing data. As a result, they often fail on unseen object cate-
gories or produce low-fidelity reconstructions, thus lacking
instance-specific details. On the other hand, per-instance
optimization methods [6, 32, 36, 47, 49], as represented by
frameworks like NeRF or 3D Gaussian Splatting (3DGS),
directly optimize the reconstruction for a specific instance
without relying on those pre-trained models, achieving su-
perior fidelity. Yet, they possess a critical weakness, i.e.
extreme sensitivity to the initialization of movable parts.
As shown in Figure 1, an initial segmentation from heuris-
tic clustering that fails on complex multi-part objects can
still yield a small rendering loss, causing the optimization
1
arXiv:2602.22666v2  [cs.CV]  26 Mar 2026

<!-- page 2 -->
to converge to an incorrect kinematic structure (a local min-
imum). This strong dependency is particularly detrimen-
tal for objects with multiple movable parts, which severely
hinders reconstruction efficiency and limits their practical
utility in real-world applications.
To overcome these limitations, we introduce ArtPro, a
robust and self-supervised framework for articulated object
reconstruction based on 3DGS. Our core idea is to forgo
the brittle “guess-the-segmentation-once” paradigm and in-
stead adopt a more intelligent and robust “propose-verify-
merge” pipeline. That is, we reframe part segmentation not
as a fixed starting point, but as a dynamic process that is
continuously refined during optimization.
The robustness and efficacy of ArtPro are driven by three
key designs that form a self-correcting optimization system:
• Prior-Guided Mobility Proposal Initialization:
We
generate over-segmented mobility proposals by integrat-
ing geometry features with motion priors, establishing a
robust initialization for subsequent optimization.
• Adaptive Proposal Integration: We dynamically merge
mobility part proposals during the 3DGS optimization
process by assessing motion consistency between adja-
cent regions, merging those that produce similar rendered
results under exchanged motion parameters.
• Collision-Aware Motion Pruning: We introduce an ac-
tive motion pruning mechanism that monitors inter-part
collisions during optimization and calibrates problematic
motion parameters to prevent local minima and ensure
stable kinematic estimation.
Across extensive synthetic and real-world experiments,
ArtPro demonstrates strong robustness and accuracy, con-
sistently generating digital twins with precise geometry and
kinematics. The method excels particularly on complex ob-
jects with multiple adjacent parts, significantly outperform-
ing existing state-of-the-art alternatives.
2. Related Works
2.1. Understanding of 3D Articulated Objects
Prior research on articulated object understanding aims to
infer part-level mobility and kinematic structures from sen-
sory inputs, spanning several key directions. A substan-
tial body of work focuses on detecting movable parts, ei-
ther from single static images [17, 42] or by discover-
ing articulation in videos [38].
Another major direction
is joint parameter estimation, which recovers kinematic
parameters such as joint axes, pivot point, and articula-
tion angles. Further extending this line of work, methods
have been developed for various inputs: some use point
clouds [8, 18, 23, 28, 44, 51], others rely on RGB-D im-
ages from single or multiple views [1, 3, 13, 15, 16, 27], and
some leverage videos [30] or 4D dynamic point clouds [31].
To overcome the limitations of passive perception, which
often suffer from ambiguous or insufficient viewpoints, ac-
tive perception methods have been developed. These meth-
ods employ strategies that enable robots to actively in-
teract with the environment or plan optimal viewing tra-
jectories, thereby disambiguating the articulated structure
and facilitating a more accurate estimation of its kinematic
parameters [2, 52, 54, 55].
More recently, the field has
increasingly leveraged the powerful reasoning capabilities
of Vision-Language Models (VLMs) for articulated object
understanding. By fine-tuning VLMs on these tasks, re-
searchers have enabled them to directly estimate joint pa-
rameters and perform kinematic reasoning from a combi-
nation of visual and textual inputs [14, 39, 43]. However,
despite producing shapes with plausible structures and joint
parameters, these methods rely heavily on object datasets
with extensive part-level annotations and consequently ex-
hibit limited generalization ability.
2.2. Reconstruction of 3D Articulated Objects
Current methodologies for 3D articulated object reconstruc-
tion fall into two paradigms: feed-forward reconstruction
and per-instance optimization, each presenting a unique set
of trade-offs between efficiency, generality, and fidelity.
The first paradigm employs feed-forward models to di-
rectly predict articulated structures from inputs such as sin-
gle/multiple images [4, 5, 10, 12, 19, 25, 35, 37, 48, 56],
text prompts [39, 41], or articulation graphs [26]. Lever-
aging the powerful data-driven priors from diffusion mod-
els [9, 22, 34] or vision-language models [21, 39, 43], these
methods achieve remarkable inference speed. However, this
efficiency comes at a significant cost. To ensure generality,
they often rely on the coarse geometric abstractions, such as
bounding boxes, primitive shapes, or parts retrieved from
limited databases, which fundamentally limit their ability
to reconstruct fine-grained, instance-specific geometry and
high-fidelity appearances. Therefore, their output quality is
bounded by the scope of their training data and the simplic-
ity of their output representations.
In contrast, the second paradigm employs per-instance
optimization, bypassing pre-trained feedforward models to
directly optimize a reconstruction for a specific object in-
stance. The populer strategy is to rely on implicit repre-
sentations, Neural Radiance Fields (NeRF) [6, 36, 49] and,
more recently, 3D Gaussian Splatting [33, 47]. However,
the scalability of these methods is severely limited by their
strong dependency on the initial part segmentation. This
is particularly problematic for complex multi-part objects,
where an imperfect initialization often causes the optimiza-
tion to converge to incorrect kinematic structures, making
them largely ineffective in practice. To address these lim-
itations, we introduce ArtPro, a novel framework designed
to overcome the aforementioned challenges by adaptively
initializing and integrating mobility proposals.
2

<!-- page 3 -->
2D RGB-D
2D RGB-D
Point Cloud 𝓟𝟏
Point Cloud 𝓟𝟎
𝑺𝒕𝒂𝒕𝒆 𝟎
𝑺𝒕𝒂𝒕𝒆 𝟏
OverSeg Part
✅
Stage 1. xxx Motion Optimization
Over Segment
Proposal Initialization
Multi-View Input
Optimization
Mobility Proposal
3D Gaussians 𝓖
Candidate Pruning
Prune
×
Detect
Neighbors
Proposal Integration
Merge
Merged Part 
update
×
×
×
×
Cycle #1
𝐶𝑦𝑐𝑙𝑒 #1
Cycle #i
𝐶𝑦𝑐𝑙𝑒 #𝑛
Search
Figure 2. ArtPro reconstructs articulated objects from multi-view RGBD images of two states. The pipeline begins by initializing part-
motion proposals through over-segmentation and a mixed-variable search. These proposals are then refined via a self-supervised optimiza-
tion that updates the parts and their motion parameters using transformable 3D Gaussians. The optimization incorporates motion pruning
and proposal integration operations to calibrate motions and merge proposals into coherent movable parts. Finally, a post-processing re-
finement stabilizes the appearance, geometry, and motion parameters of the reconstruction.
3. Method
3.1. Problem Formulation
Our goal is to reconstruct realistic digital twins of articu-
lated objects from captured multi-view images. As shown
in Figure 2, the input comprises two motion states of an ar-
ticulated object, each state represented by a set of RGBD
images. We parameterize the object states using a continu-
ous variable t, and consider the two given states as the start
(t = 0) and end (t = 1), respectively. Thus, any t ∈(0, 1)
represents an intermediate state. Assuming the object con-
tains one static part and M movable parts, the output of our
approach consists of the segmented parts and their motion
parameters, enabling realistic motion simulation of the re-
constructed object.
The segmented parts are encoded using the 3D Gaussian
Splatting (3DGS) representation. Specifically, in addition to
the vanilla Gaussian parameters, we define part-aware prob-
ability fields and assign probability values for the Gaussian
centered at xi: Ps(xi) indicating its likelihood of belonging
to the static part, and {Pm(xi)}M
m=1 for the movable parts.
In this way, we assign the Gaussians to different parts based
on the conditional probability
ˆPm(xi) =
 1 −Ps(xi)

Pm(xi)
PM
k=1 Pk(xi)
,
Gm =
n
gi ∈G
 ˆPm(xi) > ϵ
o
,
(1)
where Gm denotes the m-th movable part, ϵ = 0.01 is the
threshold, xi is the center of Gaussian gi ∈G. The union of
all the Gaussians G represents the whole object.
The part motion parameters are defined according to the
joint types. For prismatic joints, the motion of the movable
part is parameterized by a translation vector tm ∈R3. For
revolute joints, the motion parameter is defined as a 6DoF
rotation rm ∈R6 [57] and a center position cm ∈R3. Put
them together, we can apply the rigid transformation of part
m to a Gaussian gi centered at xi:
ˆxi = Rm(xi −cm) + cm + tm,
ˆqi = Rm ⊗qi,
(2)
where Rm is the rotation matrix of rm and qi is the quater-
nion parameter of the Gaussian gi.
With the above representation, we can apply the motion
parameters to transform the articulated object G
ˆG = T (G).
(3)
To render the transformed articulated object ˆG with 3DGS
differentiable rasterizer, we assign the adjusted opacity val-
ues ˆαi = ˆPm(xi)αi and ˆαi = Ps(xi)αi to the Gaussians
of movable parts and static part, respectively. This allows
us to optimize the parts and their motion parameters of the
articulated object based on the multi-view RGBD images.
3.2. Mobility Proposal Initialization
Since each part is attached with its own motion parame-
ters, the part initialization significantly impacts subsequent
optimization. To achieve robust reconstruction of complex
multi-part objects, our strategy is to initialize with an over-
segmentation of mobility proposals, which are then progres-
sively merged during the optimization process.
The process begins with the over-segmentation of point
clouds Pt = {P0, P1}, which are constructed from input
multi-view depth maps. The movable set ˆP0 is obtained by
selecting points in P0 whose nearest-neighbor distance to
P1 exceeds a threshold τ. Next, n seed points are selected
3

<!-- page 4 -->
from ˆP0 via farthest point sampling. We then utilize point-
wise features from a pre-trained segmentation model [29] to
grow n over-segmented parts around these seeds. Finally,
we merge parts with more than 80% overlap to produce M
movable part proposals { ˆPm}. Note that the proposal count
M does not necessarily equal to the ground-truth part num-
ber of the object.
We initialize the motion parameters for these proposals
based on the heuristic that joint axes are typically perpen-
dicular to part surfaces or aligned with their edges. For each
part Pm, we extract its oriented bounding box (OBB) and
three principal component axes {Xm, Ym, Zm}. We then
select the most reliable axis to initialize the motion param-
eters by solving a mixed-variable optimization problem:
arg min
ˆa,ϕ,d CD

Rm(ˆa, ϕ)( ˆPm −c) + c + d · ˆa →P1
,
(4)
where we jointly solve for the discrete joint axis choice
ˆa ∈{Xm, Ym, Zm}, the rotation angle ϕ, and the trans-
lation magnitude d. Here, Rm(ˆa, ϕ) denotes the rotation
about axis ˆa by angle ϕ, and c is the center of the OBB face
closest to P1. The objective is to minimize the one-sided
Chamfer distance CD(→) between the transformed mov-
able parts and the target point cloud P1. The variables are
constrained to d ∈[−0.5, 0.5] meters and ϕ ∈[−80◦, 80◦].
After obtaining the optimal solution, the motion param-
eters are initialized to an intermediate state as
Rm ←Rm(ˆa, ϕ/2),
tm ←(d/2) · ˆa,
cm ←c,
(5)
which helps stabilize the subsequent optimization by reduc-
ing the transformation magnitude to half.
3.3. Adaptive Proposal Integration Optimization
We propose an adaptive optimization algorithm to progres-
sively merge initial mobility proposals, ultimately deter-
mining the coherent movable parts of the articulated ob-
ject. As in Figure 2, the algorithm begins by establishing
associations between the Gaussian primitives and the ini-
tialized proposals. It then refines these proposals through
an adaptive optimization cycle, which consists of three key
components: (1) Gaussian primitive optimization, which is
the main body of the cycle, refines the segmentation and
motion parameters of the proposals by iteratively updating
the Gaussian primitives. (2) Motion pruning operation cal-
ibrates the motion parameters of the part proposals based
on their motion trajectories. (3) Proposal integration oper-
ation merges pairs of proposals that belong to the same part
and updates the associated Gaussian primitives. The cycle
repeats until no proposals can be merged.
Gaussian-Proposal Association.
We initialize two sets
of Gaussian primitives from the point clouds P0 and P1.
Following [33], we assign the part-assignment probabil-
ity for each Gaussian primitive using a Gaussian Mixture
Model (GMM). Specifically, the region of the m-th mov-
able part proposal is represented by a Gaussian component
N(µm, Σm) with a positive mixture weight ωm ∈R+. µm
and Σm are initialized from the mean and variance of the
point set ˆPm. The likelihood of a Gaussian primitive lo-
cated at x belonging to movable part m is then given by
Pm(x) = ωm · N(x; µm, Σm).
(6)
And that of the static part Ps(x) is initialized as zero. This
probabilistic association allows us to assign the Gaussians
to the parts as described in Section 3.1 and transform Gaus-
sian primitives according to the motion parameters.
Gaussian Primitive Optimization. The goal of this opti-
mization is to iteratively refine the parameters of Gaussian
primitives and their associated proposal motion parameters.
In each iteration, we transform the Gaussians from the start
state to the end state as described in Section 3.1 and render
them with the differentiable rasterizer. Therefore, we obtain
the Gaussian sets G, their centers {xi}, and the rendered im-
ages of the transformed articulated object.
We define a series of loss terms as the optimization ob-
jectives. The RGBD image loss LI of end state and one-
sided chamfer distance loss Lcd encourage the transformed
Gaussians to align with the underlying surface of the end
state, in order to ensure the reconstruction fidelity and en-
able the following proposal integration by checking the spa-
tial relation of Gaussian centers. The part contrastive loss
Lpc encourages the assignment of each Gaussian to a single
dominant part by suppressing the probabilities of all non-
maximum part proposals. The local smoothness loss Lls
enforces spatial smoothness on the static part probability
field by encouraging similar values for neighboring Gaus-
sian primitives. The regularization term Lreg encourages
the compactness of each part by regularizing the part prob-
abilities to align with a Gaussian distribution.
Finally, the objective function of the optimization is
L = LI + λcdLcd + λpcLpc + λlsLls + λregLreg.
(7)
The detailed formulation of each loss term is presented in
the supplementary material.
Motion Pruning Strategy. We calibrate the joint axes ev-
ery 100 iterations during the Gaussian primitive optimiza-
tion by measuring the overlap volume between the proposal
pairs along their motion trajectories. It prevents the opti-
mization from becoming stuck in local minima.
Specifically, for parts i and j, we compute their oriented
bounding boxes (OBBs) in both the original (i.e. b0
i , b0
j)
and transformed states (i.e. b1
i , b1
j). The overlap volumes
between the two parts in these states are calculated as v0
i,j =
Vol(b0
i ∩b0
j) and v1
i,j = Vol(b1
i ∩b1
j). A collision is flagged
if the overlap volume of the end state is larger than that of
the start state, i.e. ∆vi,j = v1
i,j −v0
i,j > τv with τv =
4

<!-- page 5 -->
10−4. Once a collision is detected, the colliding axis acol is
identified as the principal axis of the OBB that exhibits the
largest projection of the relative translation vector between
the two parts. The motion parameters are then calibrated as
follows to resolve the collision.
For a prismatic joint, the component of the translation
vector tm along the colliding axis acol is pruned by project-
ing it onto the plane orthogonal to acol:
tm ←tm −(tm · acol) · acol.
(8)
For a revolute joint, the proposals with near-identity ro-
tation angles (e.g., less than 5◦) are suppressed by resetting
the rotation axis to align with the nearest OBB edge and
halving the rotation angle to avoid the mutual influence be-
tween the two joint types.
In addition, we enforce a hard constraint on the joint type
after 4K iterations. Thus, we set and fix tm as zero for rev-
olute joints and Rm as identity matrix for prismatic joints.
Proposal Integration Operation. We identify and merge
proposal pairs that are both spatially adjacent and kinemat-
ically consistent. This is conducted every 5K iterations of
the Gaussian primitive optimization.
The spatial adjacency between two proposals, Gi and Gj,
is determined by performing the k-nearest neighbor search
(with k = 8) over the union of all movable Gaussian centers
∪{Gm}. Specifically, for each point x in Gi, if any of its
eight nearest neighbors belongs to a different movable part
Gj, then Gi and Gj are considered adjacent.
To evaluate motion consistency between an adjacent pair
(Gi, Gj), we define a score function based on the L1 distance
between rendered depth maps. The score S(T ) for a given
motion parameters T is formulated as the sum of L1 errors
across all training views v of the end state:
S(T ) =
X
Dv∈{D1
i }
|Dv(T (G)) −Dv|,
(9)
where Dv(T (G)) is the rendered depth map from viewpoint
v, and Dv is the corresponding ground-truth depth map at
the end state (t = 1). For a pair (Gi, Gj), we compute
two scores: the score with their current motion parameters,
S(T ), and the score when the motion of part i is replaced
by that of part j, denoted as S(T (j→i)). The integration cri-
terion is that their score variation falls below a predefined
threshold τ, i.e., |S(T (i)
i
) −S(T (j)
i
)| < τmerge = 10−3.
If a part Gi has multiple adjacent parts eligible for merging,
it is merged with the neighbor proposal Gj for which the
score variation is the smallest. This strategy ensures that
we integrate only spatially adjacent parts with empirically
consistent motions, effectively preventing spurious merges
caused by minor rendering score jitter.
When integrating two part proposals, we merge Gi and
Gj, and use the union of their Gaussian centers to initialize
the updated OBB and part probability field Pnew(x). The
motion of the integrated part is updated as that of Gj.
3.4. Post-Processing Refinement
After repeating the adaptive optimization cycle process sev-
eral times to progressively integrate the proposals into mov-
able parts, we conduct the Gaussian primitive optimization
again as the post-processing refinement to obtain the accu-
rate reconstruction. We follow the procedures of the Gaus-
sian primitive optimization as described in Section 3.3 with
a modified objective function.
The objective function includes the RGBD loss at both
states t = {0, 1}. And we introduce a collision loss Lcol to
avoid the collision between the movable and static parts of
transformed object T (G). Thus the objective function is
L = LI(G) + LI(T (G)) + λcLcol
(10)
Detailed formulations are in the supplementary material.
4. Experiments
4.1. Settings
Datasets. We evaluate all the methods with articulated ob-
jects collected from the datasets of existing works [24, 33],
including two-part objects and multi-part objects. In ad-
dition, we select 8 multi-part objects with 3-11 parts from
the PartNet-Mobility dataset [50]. These articulated objects
contain multiple parts with diverse motion structures and
close locations, which poses particular challenges for re-
constructing their accurate articulation structures.
Metrics. Following the evaluation protocol of ArtGS [33],
we assess performance with the metrics of both mesh recon-
struction and articulation estimation. For mesh reconstruc-
tion, we uniformly sample 10K points on the reconstructed
mesh and the ground-truth mesh, then compute the chamfer
distance for the whole object (CD-w), the static parts (CD-
s), and the movable parts (CD-m). For articulation estima-
tion, we compute the angular error (Axis Ang.) and dis-
tance (Axis Pos.) between the predicted and ground-truth
joint axes, with the latter only for revolute joints. We also
report part motion error (Part Motion), measuring rotational
geodesic distance error (in degrees) for revolute joints and
Euclidean distance error (in meters) for prismatic joints.
4.2. Comparisons
4.2.1. Comparison on Two-part Objects
The quantitative comparison with PARIS [24], Articulat-
edGS [11], DTA [46], ArtGS [33], is presented in the
supplementary material, showing that the 3DGS-based ap-
proaches produce competitive reconstruction results. Fig-
ure 3 provides the visual results of two-part objects using
ArtGS [33] and our approach.
It demonstrates that the
5

<!-- page 6 -->
Table 1. Quantitative evaluation on our dataset. Lower(↓) is better on all metrics. We mark all joint type prediction errors and movable part
count prediction errors with F. Window-103238 and Table-34610 are prismatic with no Axis Pos.
Table 34178
(5 parts)
Storage 40417
(7 parts)
Window 103238
(3 parts)
Table 23372
(5 parts)
Storage 45759
(5 parts)
Table 33116
(4 parts)
Table 34610
(6 parts)
Storage 47585
(11 parts)
All
Axis
Ang
DTA [46]
46.62
20.40
4.60
33.58
6.10
2.17
34.19
31.74
22.42
ArtGS [33]
4.64
F
0.02
39.52
16.68
0.04
F
0.97
8.70 F
Ours
0.05
0.07
0.06
0.06
0.03
0.02
0.09
0.16
0.07
Axis
Pos
DTA [46]
21.68
3.37
-
2.90
0.66
0.01
-
1.15
4.96
ArtGS [33]
1.42
F
-
1.46
0.05
0.00
-
0.00
0.32 F
Ours
0.00
0.00
-
0.00
0.00
0.00
-
0.00
0.00
Part
Motion
DTA [46]
25.65
30.17
0.26
12.79
32.92
0.11
0.10
2.51
13.07
ArtGS [33]
11.42
F
0.20
15.21
13.01
0.01
F
27.65
9.50 F
Ours
0.09
0.08
0.00
0.02
0.05
0.04
0.00
0.00
0.04
CD-s
DTA [46]
1.38
4.37
7.52
1.75
3.70
1.95
2.89
5.15
3.59
ArtGS [33]
1.87
0.90
6.61
2.81
1.83
1.60
2.55
2.97
2.64
Ours
0.42
0.36
0.11
0.47
0.37
0.65
1.15
3.21
0.84
CD-m
DTA [46]
F
F
65.60
F
F
0.85
48.06
F
38.17 F
ArtGS [33]
16.91
1,316.00
330.98
541.48
29.37
1.33
253.04
273.40
345.31
Ours
3.58
0.26
0.14
2.20
0.48
0.15
13.01
11.11
3.86
CD-w
DTA [46]
1.06
0.84
0.44
0.96
0.85
1.47
0.91
2.36
1.11
ArtGS [33]
2.05
1.00
0.64
2.69
1.67
1.97
2.02
4.08
2.02
Ours
0.45
0.33
0.12
0.41
0.36
0.30
1.16
2.13
0.65
Table 2. ArtGS-Multi dataset [33] results. Lower(↓) is better on
all metrics. Table-25493 is prismatic with no Axis Pos.
Table 25493
(4 parts)
Table 31249
(5 parts)
Storage 45503
(4 parts)
Storage 47468
(7 parts)
Oven 101908
(4 parts)
All
Axis
Ang
DTA
24.35
20.62
51.18
19.07
17.83
26.61
ArtGS
1.16
0.04
0.02
0.14
0.04
0.28
Ours
0.08
0.04
0.03
0.07
0.01
0.05
Axis
Pos
DTA
-
4.20
2.44
0.31
6.51
3.37
ArtGS
-
0.00
0.00
0.02
0.01
0.01
Ours
-
0.00
0.00
0.00
0.01
0.00
Part
Motion
DTA
0.12
30.80
43.77
10.67
31.80
23.43
ArtGS
0.00
0.01
0.03
0.62
0.23
0.18
Ours
0.00
0.02
0.03
0.09
0.09
0.05
CD-s
DTA
0.59
1.39
5.74
0.82
1.17
1.94
ArtGS
0.74
1.22
0.75
0.67
1.08
0.89
Ours
0.20
0.32
0.24
0.18
0.55
0.30
CD-m
DTA
104.38
230.38
246.63
476.91
359.16
283.49
ArtGS
3.53
3.09
0.13
3.70
0.25
2.14
Ours
0.14
1.88
0.08
2.26
0.11
0.89
CD-w
DTA
0.55
1.00
0.88
0.71
1.01
0.83
ArtGS
0.74
1.16
0.88
0.70
1.03
0.90
Ours
0.18
0.39
0.25
0.17
0.47
0.29
optimization results of ArtGS (and other 3DGS-based ap-
proaches) relies heavily on the initialization. By contrast,
our method adaptively merge the over-segmented part pro-
posals during optimization, thus achieving more robust re-
construction results.
It is worth noting that our method doesn’t necessarily
merge the disconnected components together, such as the
Blade-103706 shown in second row of Figure 3. For this
case, although the slider and blade are assigned as different
parts, our method still successfully estimates their accurate
motion parameters due to the adaptive proposal integration.
𝐵𝑙𝑎𝑑𝑒103706 𝑆𝑐𝑖𝑠𝑠𝑜𝑟11100
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝐴𝑟𝑡𝐺𝑆
𝑂𝑢𝑟𝑠
𝐺𝑇
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
Figure 3.
The part initialization and reconstruction results of
ArtGS [33] and ours on two-part objects. Although our approach
doesn’t merge the two disadjacent movable parts (second row), we
still obtain accurate motion and geometry reconstruction.
4.2.2. Comparison on Multi-part Objects
We further evaluate the performance of multi-part object re-
construction methods, i.e. DTA [46] and ArtGS [33]. Ta-
ble 2 and Table 1 reports the quantitative results. Since DTA
suffers from movable part identification and axis prediction
as the number of parts increases, it leads to significant de-
viations in the motion axes. This in turn results in greater
errors for part motion estimation and movable part recon-
struction. The other two methods, ArtGS and ours, both
achieve accurate reconstruction and motion estimation in
many cases. However, for complex objects with adjacent
movable parts, as reported in Table 1, ArtGS leads to in-
correct motion estimations and thus larger reconstruction
errors, while our method constantly achieves robust estima-
tion for all the movable parts.
Figure 4 shows the reconstructed objects at different
6

<!-- page 7 -->
𝑡= 0
𝑡= 0.5
𝑡= 1
𝑀𝑜𝑡𝑖𝑜𝑛
𝑡= 0
𝑡= 0.5
𝑡= 1
𝑀𝑜𝑡𝑖𝑜𝑛
𝑇𝑎𝑏𝑙𝑒34610
𝑆𝑡𝑜𝑟𝑎𝑔𝑒47585
𝑇𝑎𝑏𝑙𝑒31249
𝑇𝑎𝑏𝑙𝑒23372
𝑂𝑢𝑟𝑠
𝐴𝑟𝑡𝐺𝑆
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝐺𝑇
Figure 4. The reconstructed articulated objects in different motion states (t = {0, 0.5, 1}) and their part-motion structures.
𝑆𝑐𝑖𝑠𝑠𝑜𝑟11100
𝐶𝑦𝑐𝑙𝑒 #1
𝐶𝑦𝑐𝑙𝑒 #2
𝐶𝑦𝑐𝑙𝑒 #3
𝐹𝑖𝑛𝑎𝑙
𝑇𝑎𝑏𝑙𝑒34178
𝐶𝑦𝑐𝑙𝑒 #1
𝐶𝑦𝑐𝑙𝑒 #2
𝐹𝑖𝑛𝑎𝑙
𝐹𝑟𝑖𝑑𝑔𝑒10905
𝐶𝑦𝑐𝑙𝑒 #1
𝐶𝑦𝑐𝑙𝑒 #2
𝐹𝑖𝑛𝑎𝑙
𝑆𝑡𝑜𝑟𝑎𝑔𝑒47585
𝐶𝑦𝑐𝑙𝑒 #1
𝐶𝑦𝑐𝑙𝑒 #2
𝐶𝑦𝑐𝑙𝑒 #3
𝐹𝑖𝑛𝑎𝑙
Figure 5. The intermediate results during our adaptive proposal integration optimization. We show the optimized Gaussians (with their
center points) and the estimated motions before the proposal integration operation at each cycle.
states and their motion structures. Comparing these multi-
part articulated objects, ArtGS often fails and falls trapped
in a local optimum early during optimization, since the
clustering-based algorithm cannot provide a stable initial-
ization. By contrast, our over-segmentation-based initial-
ization can segment potential part proposals from the object,
and effectively merge them into complete parts during the
adaptive integration optimization. Thus, our method pro-
duces robust reconstructions for all the multi-part objects.
In Figure 5, we show the intermediate results of movable
parts being adaptively integrated under over-segmentation
initialization. For complex inputs, our method can merge
similar parts based on motion parameters optimized after
some iteration steps, resulting in complete movable parts
with only a few cycles.
4.2.3. Results on Real-World Objects
We additionally evaluate our approach on real-world artic-
ulated objects. Specifically, we collected 200 RGB images,
100 for each state, around the target object and utilized the
pre-trained depth-anything-V2 model [53] to estimate the
calibrated depth. Then, we use SAM2 [40] to obtain the
object masks. Finally, we take the processed multi-view
RGBD images of two states as the input of our approach.
Figure 7 presents the reconstruction results of the real-
world multi-part objects. Our method achieves robust and
accurate results for them. More importantly, the superior
mobility reconstruction ensures the correct articulated ob-
ject with high-quality parts and motion parameters.
4.3. Ablation Studies
As listed in Table 3, we conduct the following ablation ex-
periments to validate the effectiveness of our key designs:
• The baseline (#1) applies DBSCAN clustering [7] (with
the ground-truth part number M) to separate the movable
parts and assign identity transformation as initialization,
then optimize the Gaussians and motion parameters with
the same setting of our approach, but without the adap-
tive proposal integration and motion pruning strategy. As
shown in Figure 6, once the movable parts are incorrectly
7

<!-- page 8 -->
𝑆𝑡𝑜𝑟𝑎𝑔𝑒47585
𝑇𝑎𝑏𝑙𝑒34610
𝐼𝑛𝑖𝑡. 𝑃𝑎𝑟𝑡𝑠
#1 𝑉𝑎𝑛𝑖𝑙𝑙𝑎
#2 𝑤/ 𝑀𝑜𝐼
𝐼𝑛𝑖𝑡. 𝑃𝑎𝑟𝑡𝑠
#3 𝑤/ 𝑀𝑜𝐼
#4 𝑤/𝑜 𝑃𝑟𝑢𝑛𝑒
𝐹𝑢𝑙𝑙
𝑂𝑣𝑒𝑟𝑆𝑒𝑔
𝐶𝑙𝑢𝑠𝑡𝑒𝑟𝑖𝑛𝑔
Figure 6. Ablation study results. To compare with our OverSeg initialization, we use DBSCAN [7] as our clustering implementation,
denoted V anilla. We further add motion initialization to V anilla and OverSeg baseline, denoted w/ MoI. Finally, we removed the
pruning strategy from Full (w/o Prune) to verify its significant.
𝑆𝑡𝑜𝑟𝑎𝑔𝑒
𝑃𝑟𝑖𝑛𝑡𝑒𝑟
𝑡= 0
𝑡= 1
𝑀𝑜𝑡𝑖𝑜𝑛
𝑡= 0.5
Figure 7. Reconstructed results of real-world articulated objects.
segmented or located close to each other, it sometimes
causes missing parts or converges to local minima, lead-
ing to failed motion estimation and inaccurate geometry.
• We then add our proposal initialization algorithm (MoI)
into the baseline setting. Baseline (#2) refers to initializ-
ing our proposals with the given part number M, while
baseline (#3) leverages our over-segmentation initializa-
tion without knowing the ground-truth part number. In
Figure 6, the third col validates that our method leads to
more reasonable mobility initialization and thus more ac-
curate final reconstructions, while the fourth col exposes
the remaining problem with unstable part initialization.
• We further insert our adaptive proposal integration strat-
egy (Merge) in baseline (#4).
It achieves robustness
to the proposals of various complex articulated objects.
However, some of the objects still exhibit relatively larger
reconstruction errors, due to unstable motion initializa-
tion and the lack of collision-aware constraints to cali-
brate the parameters of adjacent parts, such as the middle
three drawers of Storage (w/o Prune) in Figure 6.
• By contrast, our full approach effectively prevents in-
correct motion estimation caused by collisions between
adjacent parts. By maximally leveraging motion priors
Table 3. Quantitative evaluation of the ablation study. Lower(↓) is
better on all metrics. To evaluate the results with arbitrary numbers
of movable parts, we compute CD between the merged movable
results and GT movable parts, denoted as over CD-m.
Case
OverSeg
MoI
Merge
Prune
CD-s
CD-m
CD-w
#1
1.46
9.27
0.93
#2
✓
1.04
6.97
0.67
#3
✓
✓
0.87
5.01
0.71
#4
✓
✓
✓
0.85
4.57
0.67
Full
✓
✓
✓
✓
0.84
3.63
0.65
of man-made objects, our method achieves robust recon-
struction performance for all the multi-part objects.
5. Conclusion
We introduced ArtPro, a self-supervised framework based
on 3D Gaussian Splatting for robustly reconstructing high-
fidelity digital twins of articulated objects. Our method be-
gins with an over-segmented set of part proposals and adap-
tively merges them during optimization through motion-
consistency analysis and collision-aware pruning.
This
strategy significantly improves motion estimation and re-
construction quality, especially for complex objects with
multiple parts and diverse articulations, as validated through
extensive experiments on synthetic and real-world datasets.
However, our approach currently relies primarily on mo-
tion cues between two object states, which may be insuffi-
cient for disambiguating parts with perfectly symmetric or
very subtle motions. Additionally, the reconstruction qual-
ity, particularly at part boundaries, remains sensitive to the
accuracy of input sensor data. Future work will explore
the integration of stronger semantic priors, geometry con-
straints, and multi-state tracking to address these challenges
and further expand the applicability of our method.
8

<!-- page 9 -->
6. Acknowledgement
This work is supported by the National Natural Science
Foundation of China (62302269), the Excellent Young
Scientists Fund Program (Overseas) of Shandong Province
(No.2023HWYQ-034), the Joint Funds of the National
Natural Science Foundation of China (U23A20312),
a grant from the Chongqing Natural Science Founda-
tion (CSTB2024NSCQ-MSX1026), the Natural Science
Foundation of Shandong Province (No.ZR2023QF077),
Shandong Province Key Research and Development Pro-
gram (2024TSGC0051). We also acknowledge the support
from the grant by Lejoin Intelligence (Shenzhen) Co.,Ltd.
References
[1] Ben Abbatematteo, Stefanie Tellex, and George Konidaris.
Learning to generalize kinematic models to novel objects. In
Proceedings of the 3rd Conference on Robot Learning, 2019.
2
[2] Ankan Kumar Bhunia, Changjian Li, and Hakan Bilen. In-
teractive anomaly detection for articulated objects via mo-
tion anticipation. In The Thirty-ninth Annual Conference on
Neural Information Processing Systems. 2
[3] Yuchen Che, Ryo Furukawa, and Asako Kanezaki. Op-align:
Object-level and part-level alignment for self-supervised
category-level articulated object pose estimation.
In Eu-
ropean Conference on Computer Vision, pages 72–88.
Springer, 2024. 2
[4] Zoey Chen, Aaron Walsman, Marius Memmel, Kaichun Mo,
Alex Fang, Karthikeya Vemuri, Alan Wu, Dieter Fox, and
Abhishek Gupta.
Urdformer: A pipeline for constructing
articulated simulation environments from real-world images.
arXiv preprint arXiv:2405.11656, 2024. 1, 2
[5] Tianyuan Dai, Josiah Wong, Yunfan Jiang, Chen Wang, Cem
Gokmen, Ruohan Zhang, Jiajun Wu, and Li Fei-Fei. Auto-
mated creation of digital cousins for robust policy learning.
arXiv preprint arXiv:2410.07408, 2024. 1, 2
[6] Jianning Deng, Kartic Subr, and Hakan Bilen.
Articulate
your nerf: Unsupervised articulated object modeling via con-
ditional view synthesis.
Advances in Neural Information
Processing Systems, 37:119717–119741, 2024. 1, 2
[7] Martin Ester, Hans-Peter Kriegel, J¨org Sander, Xiaowei Xu,
et al. A density-based algorithm for discovering clusters in
large spatial databases with noise. In kdd, pages 226–231,
1996. 7, 8
[8] Lian Fu, Ryoichi Ishikawa, Yoshihiro Sato, and Takeshi
Oishi. Capt: Category-level articulation estimation from a
single point cloud using transformer, 2024. 2
[9] Daoyi Gao, Yawar Siddiqui, Lei Li, and Angela Dai. Me-
shart: Generating articulated meshes with structure-guided
transformers. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 618–627, 2025. 2
[10] Mingju Gao, Yike Pan, Huan-ang Gao, Zongzheng Zhang,
Wenyi Li, Hao Dong, Hao Tang, Li Yi, and Hao Zhao.
Partrm: Modeling part-level dynamics with large cross-state
reconstruction model. In Proceedings of the Computer Vi-
sion and Pattern Recognition Conference, pages 7004–7014,
2025. 1, 2
[11] Junfu Guo, Yu Xin, Gaoyi Liu, Kai Xu, Ligang Liu, and
Ruizhen Hu.
Articulatedgs: Self-supervised digital twin
modeling of articulated objects using 3d gaussian splatting.
In Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 27144–27153, 2025. 5, 1, 2
[12] Nick Heppert, Muhammad Zubair Irshad, Sergey Zakharov,
Katherine Liu, Rares Andrei Ambrus, Jeannette Bohg, Ab-
hinav Valada, and Thomas Kollar. Carto: Category and joint
agnostic reconstruction of articulated objects. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 21201–21210, 2023. 2
[13] Ruizhen Hu, Wenchao Li, Oliver Van Kaick, Ariel Shamir,
Hao Zhang, and Hui Huang. Learning to predict part mo-
bility from a single static snapshot. ACM Transactions On
Graphics (TOG), 36(6):1–13, 2017. 2
[14] Siyuan Huang, Haonan Chang, Yuhan Liu, Yimeng Zhu,
Hao Dong, Peng Gao, Abdeslam Boularias, and Hongsheng
Li. A3vlm: Actionable articulation-aware vision language
model. arXiv preprint arXiv:2406.07549, 2024. 2
[15] Ajinkya Jain, Rudolf Lioutikov, Caleb Chuck, and Scott
Niekum.
Screwnet:
Category-independent articulation
model estimation from depth images using screw theory. In
2021 IEEE International Conference on Robotics and Au-
tomation (ICRA), pages 13670–13677. IEEE, 2021. 2
[16] Ajinkya Jain, Stephen Giguere, Rudolf Lioutikov, and Scott
Niekum. Distributional depth-based estimation of object ar-
ticulation models. In Conference on Robot Learning, pages
1611–1621. PMLR, 2022. 2
[17] Hanxiao Jiang, Yongsen Mao, Manolis Savva, and Angel X
Chang. Opd: Single-view 3d openable part detection. In
Computer Vision–ECCV 2022: 17th European Conference,
Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part
XXXIX, pages 410–426. Springer, 2022. 2
[18] Zhenyu Jiang, Cheng-Chun Hsu, and Yuke Zhu.
Ditto:
Building digital twins of articulated objects from interaction.
In Conference on Computer Vision and Pattern Recognition
(CVPR), 2022. 2
[19] Yuki Kawana and Tatsuya Harada.
Detection based part-
level articulated object reconstruction from single rgbd im-
age. Advances in Neural Information Processing Systems,
36:18444–18473, 2023. 1, 2
[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering.
ACM Transactions on Graphics
(TOG), 42:1 – 14, 2023. 1
[21] Long Le, Jason Xie, William Liang, Hung-Ju Wang, Yue
Yang, Yecheng Jason Ma, Kyle Vedder, Arjun Krishna, Di-
nesh Jayaraman, and Eric Eaton. Articulate-anything: Auto-
matic modeling of articulated objects via a vision-language
foundation model. arXiv preprint arXiv:2410.13882, 2024.
2
[22] Jiahui Lei, Congyue Deng, William B Shen, Leonidas J
Guibas, and Kostas Daniilidis. Nap: Neural 3d articulated
object prior.
Advances in Neural Information Processing
Systems, 36:31878–31894, 2023. 2
9

<!-- page 10 -->
[23] Xiaolong Li, He Wang, Li Yi, Leonidas J. Guibas, A. Lynn
Abbott, and Shuran Song. Category-level articulated object
pose estimation. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
3706–3715, 2020. 2
[24] Jiayi Liu, Ali Mahdavi-Amiri, and Manolis Savva. Paris:
Part-level reconstruction and motion analysis for articulated
objects. In Proceedings of the IEEE/CVF International Con-
ference on Computer Vision, pages 352–363, 2023. 5, 1, 2
[25] Jiayi Liu, Denys Iliash, Angel X Chang, Manolis Savva,
and Ali Mahdavi-Amiri. Singapo: Single image controlled
generation of articulated parts in objects.
arXiv preprint
arXiv:2410.16499, 2024. 2
[26] Jiayi Liu, Hou In Ivan Tam, Ali Mahdavi-Amiri, and Manolis
Savva. Cage: Controllable articulation generation. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 17880–17889, 2024. 2
[27] Liu Liu, Han Xue, Wenqiang Xu, Haoyuan Fu, and Cewu
Lu. Toward real-world category-level articulation pose esti-
mation. IEEE Transactions on Image Processing, 31:1072–
1083, 2022. 2
[28] Liu Liu, Jianming Du, Hao Wu, and Xun Yang. Category-
level articulated object 9d pose estimation via reinforcement
learning. Proceedings of the 31st ACM International Con-
ference on Multimedia, pages 728–736, 2023. 2
[29] Minghua Liu, Mikaela Angelina Uy, Donglai Xiang, Hao Su,
Sanja Fidler, Nicholas Sharp, and Jun Gao. Partfield: Learn-
ing 3d feature fields for part segmentation and beyond. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 9704–9715, 2025. 4, 3
[30] Qihao Liu, Weichao Qiu, Weiyao Wang, Gregory D Hager,
and Alan L Yuille. Nothing but geometric constraints: A
model-free method for articulated object pose estimation.
arXiv preprint arXiv:2012.00088, 2020. 2
[31] Shaowei Liu, Saurabh Gupta, and Shenlong Wang. Build-
ing rearticulable models for arbitrary 3d objects from 4d
point clouds. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21138–
21147, 2023. 2
[32] Yu Liu, Baoxiong Jia, Ruijie Lu, Chuyue Gan, Huayu Chen,
Junfeng Ni, Song-Chun Zhu, and Siyuan Huang. Videoartgs:
Building digital twins of articulated objects from monocular
video. arXiv preprint arXiv:2509.17647, 2025. 1
[33] Yu Liu, Baoxiong Jia, Ruijie Lu, Junfeng Ni, Song-Chun
Zhu, and Siyuan Huang. Artgs: Building interactable repli-
cas of complex articulated objects via gaussian splatting.
arXiv preprint arXiv:2502.19459, 2025. 1, 2, 4, 5, 6
[34] Rundong Luo, Haoran Geng, Congyue Deng, Puhao Li, Zan
Wang, Baoxiong Jia, Leonidas Guibas, and Siyuan Huang.
Physpart: Physically plausible part completion for inter-
actable objects.
In 2025 IEEE International Conference
on Robotics and Automation (ICRA), pages 12386–12393.
IEEE, 2025. 2
[35] Zhao Mandi, Yijia Weng, Dominik Bauer, and Shuran Song.
Real2code: Reconstruct articulated objects via code genera-
tion. arXiv preprint arXiv:2406.08474, 2024. 2
[36] Jiteng Mu, Weichao Qiu, Adam Kortylewski, Alan Yuille,
Nuno Vasconcelos, and Xiaolong Wang. A-sdf: Learning
disentangled signed distance functions for articulated shape
representation.
In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 13001–13011,
2021. 1, 2
[37] Akshay Gadi Patil, Yiming Qian, Shan Yang, Brian Jack-
son, Eric Bennett, and Hao Zhang.
Rosi: Recovering 3d
shape interiors from few articulation images. arXiv preprint
arXiv:2304.06342, 2023. 2
[38] Shengyi Qian, Linyi Jin, Chris Rockwell, Siyi Chen, and
David F. Fouhey. Understanding 3d object articulation in in-
ternet videos. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 1599–
1609, 2022. 2
[39] Xiaowen Qiu, Jincheng Yang, Yian Wang, Zhehuan Chen,
Yufei Wang, Tsun-Hsuan Wang, Zhou Xian, and Chuang
Gan. Articulate anymesh: Open-vocabulary 3d articulated
objects modeling. arXiv preprint arXiv:2502.02590, 2025.
1, 2
[40] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junt-
ing Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-
Yuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feicht-
enhofer. Sam 2: Segment anything in images and videos.
arXiv preprint arXiv:2408.00714, 2024. 7
[41] Jiayi Su, Youhe Feng, Zheng Li, Jinhua Song, Yangfan He,
Botao Ren, and Botian Xu. Artformer: Controllable genera-
tion of diverse 3d articulated objects. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
1894–1904, 2025. 1, 2
[42] Xiaohao Sun, Hanxiao Jiang, Manolis Savva, and An-
gel Xuan Chang.
Opdmulti: Openable part detection for
multiple objects.
arXiv preprint arXiv:2303.14087, 2023.
2
[43] Aditya Vora, Sauradip Nag, and Hao Zhang. Articulate that
object part (atop): 3d part articulation via text and motion
personalization. arXiv preprint arXiv:2502.07278, 2025. 2
[44] Xiaogang Wang, Bin Zhou, Yahao Shi, Xiaowu Chen, Qin-
ping Zhao, and Kai Xu. Shape2motion: Joint analysis of
motion parts and attributes from 3d shapes. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 8876–8884, 2019. 2
[45] Zehan Wang, Siyu Chen, Lihe Yang, Jialei Wang, Ziang
Zhang, Hengshuang Zhao, and Zhou Zhao. Depth anything
with any prior. arXiv preprint arXiv:2505.10565, 2025. 5
[46] Yijia Weng, Bowen Wen, Jonathan Tremblay, Valts Blukis,
Dieter Fox, Leonidas Guibas, and Stan Birchfield. Neural
implicit representation for building digital twins of unknown
articulated objects. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
3141–3150, 2024. 5, 6, 1, 2, 4
[47] Di Wu, Liu Liu, Zhou Linli, Anran Huang, Liangtu Song,
Qiaojun Yu, Qi Wu, and Cewu Lu. Reartgs: Reconstruct-
ing and generating articulated objects via 3d gaussian splat-
ting with geometric and motion constraints. arXiv preprint
arXiv:2503.06677, 2025. 1, 2
[48] Ruiqi Wu, Xinjie Wang, Liu Liu, Chunle Guo, Jiaxiong Qiu,
Chongyi Li, Lichao Huang, Zhizhong Su, and Ming-Ming
10

<!-- page 11 -->
Cheng. Dipo: Dual-state images controlled articulated ob-
ject generation powered by diverse data.
arXiv preprint
arXiv:2505.20460, 2025. 1, 2
[49] Tianhao Wu, Fangcheng Zhong, Andrea Tagliasacchi, For-
rester Cole, and Cengiz Oztireli. Dˆ 2nerf: Self-supervised
decoupling of dynamic and static objects from a monocular
video. Advances in neural information processing systems,
35:32653–32666, 2022. 1, 2
[50] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao
Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan,
He Wang, et al. Sapien: A simulated part-based interactive
environment. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 11097–
11107, 2020. 5
[51] Zihao Yan, Ruizhen Hu, Xingguang Yan, Luanmin Chen,
Oliver Van Kaick, Hao Zhang, and Hui Huang. Rpm-net:
recurrent prediction of motion and parts from point cloud.
arXiv preprint arXiv:2006.14865, 2020. 2
[52] Zihao Yan, Fubao Su, Mingyang Wang, Ruizhen Hu, Hao
Zhang, and Hui Huang. Interaction-driven active 3d recon-
struction with object interiors. ACM Transactions on Graph-
ics (TOG), 42(6):1–12, 2023. 2
[53] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any-
thing v2. Advances in Neural Information Processing Sys-
tems, 37:21875–21911, 2024. 7
[54] Hongliang Zeng, Ping Zhang, Chengjiong Wu, Jiahua Wang,
Tingyu Ye, and Fang Li. Mars: Multimodal active robotic
sensing for articulated characterization.
arXiv preprint
arXiv:2407.01191, 2024. 2
[55] Can Zhang and Gim Hee Lee. Iaao: Interactive affordance
learning for articulated objects in 3d environments. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference, pages 12132–12142, 2025. 2
[56] Ge Zhang, Or Litany, Srinath Sridhar, and Leonidas Guibas.
Strobenet: Category-level multiview reconstruction of artic-
ulated objects. arXiv preprint arXiv:2105.08016, 2021. 2
[57] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao
Li. On the continuity of rotation representations in neural
networks. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 5745–5753,
2019. 3
11

<!-- page 12 -->
ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive
Integration of Mobility Proposals
Supplementary Material
In this supplementary material, we first present the loss
term formulations of Gaussian primitive optimization in
Section A. Then, we present more two-part comparisons in
Section B. In Section C and Section D, we provide more
multi-part visualization and full initialization of our method
on multi-part dataset. In Section E and Section F, we report
additional ablation studies and computational costs. Finally,
in Section G, we report the failure cases of our method.
A. Loss Terms of Gaussian Optimization
Below we present the detailed formulation of the loss terms
used in the Gaussian primitive optimization, which are men-
tioned in Section 3.3 and 3.4 in the main paper.
The RGBD loss LI and the one-sided Chamfer Distance
loss Lcd encourage the transformed Gaussians to align with
the underlying surface of the end state. We have
LD = log(1+ ∥Dv(T (G)) −Dv ∥1)
LI = L3dgs(T (G)) + LD(T (G)),
Lcd = CD(∪{xm} →P1)
(11)
where Dv(·) denotes the rendered depth map of view v,
L3dgs is the RGB loss used in 3DGS [20], G is the Gaussian
sets, {xm} is the Gaussian centers of all movable {Gm},
and P1 is the point cloud of end state. We compute the loss
LI between the rendered images of transformed Gaussian
T (G) and GT RGBD of end state to optimize the appear-
ance, and use Lcd to further improve the geometric quality.
The part contrastive loss Lpc ensures each Gaussian be
dominated by one distinctive part
Lpc = E{xi∈Gm}M
h
1
M −1
M
X
k=1
k̸=m
Pk(xi)
i
,
(12)
where, Gm denotes the m-th movable part of {Gm}M
m=1,
Pk(xi) is the probability that the point xi belongs to the
k-th movable part.
The local smoothness loss Lls aims to maintain consis-
tent static part probability for the spatial neighbors xi, xj ∈
G. It is formulated as
Lls = Exi∈G, xj∈kNN(xi,G)|Ps(xi) −Ps(xj)|,
(13)
where kNN denote k-nearest neighbors with k = 20. To
avoid fragmented parts, we smooth the static probability
field on the kNN-based neighborhood graph to obtain a lo-
cally consistent segmentation field.
The regularization term aims to encourage the compact-
ness of the parts. It uses the mean {ˆµm} and variance scal-
ing {ˆsm} of initialization { ˆPm} to maintain the existence
of pm
Lreg = 1
M
M
X
m=1
h
λµ|µm −ˆµm|+λs ∥sm −ˆsm ∥1
i
(14)
where, sm is the scaling of variance Σm, and λµ
=
0.5, λs = 0.1 is loss weight.
In summary, the objective function of Gaussian primitive
optimization in the adaptive proposal integration stage is:
L = LI + λcdLcd + λpcLpc + λlsLls + λregLreg. (15)
We use λcd = 0.5, λpc = 0.1, λls = 0.02 and λreg = 1.0
in all the experiments presented in this paper.
We further define a collision loss Lcol, which avoids the
collisions between the movable and static parts of the trans-
formed object. It is formulated as
Lcol(Gi, Gj) = Ex∈Gi,y∈kNN(x,Gj)|ˆαy|,
Lcol = Lcol(T (Gm), Gs) + Lcol(Gs, T (Gm)),
(16)
where T (Gm) is the union of all movable parts of T (G),
Gs is the static part of T (G), ˆαy is the Gaussian opacity
corresponding to center position y, and kNN is the set of k
nearest neighbors with k = 32.
Therefore, the objective function of the post-processing
refinement optimization is
L = LI(G) + LI(T (G)) + λcLcol
(17)
We use λc = 0.02 in all the experiments presented in this
paper.
B. Additional Results on Two-Part Dataset
We compare with related methods including PARIS [24],
ArticulatedGS [11], DTA [46], ArtGS [33] on the two-part
object dataset. Since we take RGBD images as input, we
apply the depth loss to PARIS [24] and ArticulatedGS [11]
methods for a fair comparison. We report the quantitative
evaluation results in Table 4, showing that the 3DGS-based
methods, i.e. ArticulatedGS [11], ArtGS [33], and ours,
achieve competitive performance.
This is because static
and movable parts can be well initialized with spatial clus-
tering for two-part objects, which are further refined with
1

<!-- page 13 -->
Table 4. Results on the PARIS [24] dataset, including both synthetic and real data. Methods marked with an asterisk (*) denote versions
with added depth supervision for fair comparison. Specifically, ArticulatedGS* and PARIS* are trained with an additional depth loss.
Synthetic Objects
Real Objects
FoldChair Fridge Laptop Oven Scissor Stapler
USB
Washer
Blade
Storage
All
Fridge
Storage
All
Axis
Ang
PARIS* [24]
15.79
2.93
0.03
7.43
16.62
8.17
0.71
0.71
41.28
0.03
9.37
1.90
30.10
16.00
ArticulatedGS [11]
6.20
0.14
15.55
0.00
0.07
0.08
0.16
0.03
0.23
0.04
2.25
4.06
48.56
26.31
ArticulatedGS*
0.02
0.12
0.03
0.13
0.18
0.13
0.12
0.17
0.04
0.04
0.10
64.81
29.44
47.12
DTA [46]
0.03
0.09
0.07
0.22
0.10
0.07
0.11
0.36
0.20
0.09
0.13
2.08
13.64
7.86
ArtGS [33]
0.01
0.03
0.01
0.01
0.05
0.01
0.04
0.02
0.03
0.01
0.02
2.09
3.47
2.78
Ours
0.04
0.03
0.00
0.01
0.05
0.04
0.04
0.01
0.07
0.02
0.03
1.92
1.02
1.47
Axis
Pos
PARIS* [24]
0.25
1.13
0.00
0.05
1.59
4.67
3.35
3.28
-
-
1.79
0.50
-
0.50
ArticulatedGS [11]
4.93
0.00
0.16
0.03
0.00
0.02
0.63
0.00
-
-
0.72
1.71
-
1.71
ArticulatedGS*
0.00
0.02
0.00
0.02
0.00
0.02
0.64
0.02
-
-
0.09
2.71
-
2.71
DTA [46]
0.01
0.01
0.01
0.01
0.02
0.02
0.00
0.05
-
-
0.02
0.59
-
0.59
ArtGS [33]
0.00
0.00
0.01
0.00
0.00
0.01
0.00
0.00
-
-
0.00
0.47
-
0.47
Ours
0.00
0.00
0.00
0.00
0.00
0.00
0.00
0.00
-
-
0.00
0.34
-
0.34
Part
Motion
PARIS* [24]
127.34
45.26
0.03
9.13
68.36
107.76 96.93
49.77
0.36
0.30
50.52
1.58
0.57
1.08
ArticulatedGS [11]
59.08
0.55
28.16
0.13
0.05
0.07
0.27
0.00
0.04
0.00
8.84
15.11
0.53
7.82
ArticulatedGS*
0.26
0.12
0.07
0.31
0.13
0.09
0.10
0.09
0.00
0.00
0.12
39.58
0.32
19.95
DTA [46]
0.10
0.12
0.11
0.12
0.37
0.08
0.15
0.28
0.00
0.00
0.13
1.85
0.14
1.00
ArtGS [33]
0.03
0.04
0.02
0.02
0.04
0.01
0.03
0.03
0.00
0.00
0.02
1.94
0.04
0.99
Ours
0.06
0.03
0.00
0.03
0.04
0.05
0.04
0.04
0.00
0.00
0.03
2.76
0.04
1.40
CD-s
PARIS* [24]
10.20
8.82
0.16
3.18
15.58
2.48
1.95
12.19
1.40
8.67
6.46
11.64
20.25
15.95
ArticulatedGS [11]
3.17
2.02
4.27
2.31
0.37
1.84
1.88
5.17
0.44
2.74
2.42
36.05
75.82
55.94
ArticulatedGS*
0.58
1.09
1.72
1.88
0.64
1.45
1.24
3.58
0.24
1.93
1.44
230.04
46.78
138.41
DTA [46]
0.18
0.62
0.30
4.60
3.55
2.91
2.32
4.56
0.55
4.90
2.45
2.36
10.98
6.67
ArtGS [33]
0.26
0.52
0.63
3.88
0.61
3.83
2.25
6.43
0.54
7.31
2.63
1.64
2.93
2.29
Ours
0.48
0.31
0.12
1.41
0.19
0.67
0.96
2.69
0.21
1.81
0.89
1.96
2.54
2.25
CD-m
PARIS* [24]
17.97
7.23
0.15
6.54
16.65
30.46
10.17
265.27
117.99
52.34
52.48
77.85
474.57
276.21
ArticulatedGS [11]
36.61
2.29
24.90
0.96
0.35
1.64
1.02
3.88
1.86
5.49
7.90
107.96 2,459.45 1,283.71
ArticulatedGS*
0.33
0.63
2.07
1.08
0.57
1.97
1.17
0.41
0.78
0.83
0.98
69.23
1,578.02
823.63
DTA [46]
0.15
0.27
0.13
0.44
10.11
1.13
1.47
0.45
2.05
0.36
1.66
1.12
30.78
15.95
ArtGS [33]
0.54
0.21
0.13
0.89
0.64
0.52
1.22
0.45
1.12
1.02
0.67
0.66
6.28
3.47
Ours
0.11
0.28
0.07
0.33
0.18
0.83
0.37
0.08
F
0.35
0.28 F
28.31
35.95
32.13
CD-w
PARIS* [24]
4.37
5.53
0.26
3.18
3.90
5.27
1.78
10.11
0.58
7.80
4.28
8.99
32.10
20.55
ArticulatedGS [11]
1.06
2.12
8.52
2.13
0.35
1.61
1.88
4.79
0.23
2.62
2.53
77.53
995.99
536.76
ArticulatedGS*
0.43
0.96
0.80
1.60
0.59
1.48
1.00
3.08
0.20
1.75
1.19
51.22
965.80
508.51
DTA [46]
0.27
0.70
0.32
4.24
0.41
1.92
1.17
4.48
0.36
3.99
1.79
2.08
8.98
5.53
ArtGS [33]
0.43
0.58
0.50
3.58
0.67
2.63
1.28
5.99
0.61
5.21
2.15
1.29
3.23
2.26
Ours
0.10
0.31
0.09
1.32
0.18
0.62
0.50
2.40
0.18
1.60
0.73
1.03
2.03
1.53
the following deformable 3DGS optimization. This experi-
ment also acts as a sanity check for our method, validating
that the over-segmentation proposals from the same mov-
able part can be effectively merged based on their motion
similarities.
Figure 8 provides the full visualization results of ArtGS
and Ours on this dataset. The results demonstrate that our
method can robustly integrate the proposals and reconstruct
both the part probability field and motion parameters. For
incorrectly initialized mobility proposals, our optimization
can also obtain correct motion estimation and the final re-
constructions. For the real-world objects in this dataset, our
method’s performance on the geometry reconstruction, es-
pecially in terms of CD metric, is limited by the quality of
input depth maps. This is because our method tends to es-
timate the mobility parameters that align the transformed
results closer to the depth map of end state (t = 1). Large
discrepancies between the input depth maps of states t = 0
and t = 1 lead to our results containing a small number
of Gaussians originating from static part, such as the Real
Storage in Figure 8 and Figure 13.
C. Additional Results on Multi-Part Dataset
We present a comprehensive visual analysis to compare
ArtGS [33] and our method in Figure 9 and Figure 11,
which show the reconstructed articulated objects at different
motion states and their articulated joints of movable parts.
Compared to ArtGS, our method can robustly estimate the
movable parts and motion parameters, leading to consistent
and clean rendered images of the reconstructed objects.
2

<!-- page 14 -->
𝐴𝑟𝑡𝐺𝑆
𝑂𝑢𝑟𝑠
𝐺𝑇
𝐴𝑟𝑡𝐺𝑆
𝑂𝑢𝑟𝑠
𝐺𝑇
𝐹𝑜𝑙𝑑𝑐ℎ𝑎𝑖𝑟102255
𝐹𝑟𝑖𝑑𝑔𝑒10905
𝑂𝑣𝑒𝑛101917
𝑆𝑡𝑎𝑝𝑙𝑒𝑟103111
𝑈𝑆𝐵100109
𝑅𝑒𝑎𝑙 𝐹𝑟𝑖𝑑𝑔𝑒
𝐵𝑙𝑎𝑑𝑒103706
𝐿𝑎𝑝𝑡𝑜𝑝10211
𝑆𝑐𝑖𝑠𝑠𝑜𝑟11100
𝑆𝑡𝑜𝑟𝑎𝑔𝑒45135
𝑊𝑎𝑠ℎ𝑒𝑟103776
𝑅𝑒𝑎𝑙 𝑆𝑡𝑜𝑟𝑎𝑔𝑒
𝑅𝑒𝑠𝑢𝑙𝑡
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝐼𝑛𝑖𝑡.
Figure 8. Additional qualitative results on PARIS dataset, including the initializations and the final reconstructions. We show the Gaussians
with their center points for a better visualization of their segmentation and motion parameters.
D. More Visual Analysis on Multi-Part Dataset
We further show the visualizations of the initializations and
reconstructed results of ArtGS and ours in Figure 10 and
Figure 12. The results show that our mobility initializa-
tion can effectively extract the potential movable parts and
reasonable motion initialization, while ArtGS suffers from
unstable initialization. For over-segmented proposals, our
optimization can effectively estimate their motion parame-
ters and integrate adjacent proposals into an individual part.
We further report the visual quality metrics on our
dataset in Table 5.
Our method achieves better PSNR,
SSIM, and LPIPS scores compared to ArtGS, while pro-
ducing more accurate articulation.
Table 5. Visual quality on our dataset.
Method
PSNR
SSIM
LPIPS
ArtGS
36.10
0.980
0.037
Ours
48.02
0.998
0.003
Table 6. Additional ablation study on our dataset.
Case
CD-s
CD-m
CD-w
w/o PM
0.85
4.26
0.68
w/o MoI
0.89
5.62
0.70
w/o MoI&PI
0.87
5.81
0.71
Full
0.84
3.63
0.65
E. More Ablation Study Results
To strengthen the validation of careful initialization, we re-
port the results of “w/o MoI” (no principal axes for initial-
ization) and “w/o MoI&PI” (no principal axes for initial-
ization and pruning) in Table 6. We also add the results of
disabling the progressive merging after over-segmentation
(denoted as w/o PM) in Table 6.
F. Computation Overhead
We control the computational cost of ArtPro through two
key strategies. First, we use PartField [29] to produce well-
structured over-segmentation proposals rather than frag-
3

<!-- page 15 -->
𝑡= 0
𝑡= 0.5
𝑡= 1
𝑀𝑜𝑡𝑖𝑜𝑛
𝑡= 0
𝑡= 0.5
𝑡= 1
𝑀𝑜𝑡𝑖𝑜𝑛
𝑂𝑢𝑟𝑠
𝐴𝑟𝑡𝐺𝑆
𝐺𝑇
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝑇𝑎𝑏𝑙𝑒31249
𝑇𝑎𝑏𝑙𝑒25493
𝑆𝑡𝑜𝑟𝑎𝑔𝑒47648
𝑆𝑡𝑜𝑟𝑎𝑔𝑒45503
𝑂𝑣𝑒𝑛101908
Figure 9. Additional reconstruction results on ArtGS-Multi dataset. We render the reconstructed articulated objects in different motion
states (t ∈{0, 0.5, 1} and their motion structures for each result.
𝐴𝑟𝑡𝐺𝑆
𝑂𝑢𝑟𝑠
𝐺𝑇
𝐴𝑟𝑡𝐺𝑆
𝑂𝑢𝑟𝑠
𝐺𝑇
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝑇𝑎𝑏𝑙𝑒31249
𝑆𝑡𝑜𝑟𝑎𝑔𝑒47648
𝑂𝑣𝑒𝑛101908
𝑇𝑎𝑏𝑙𝑒25493
𝑆𝑡𝑜𝑟𝑎𝑔𝑒45503
Figure 10. Additional qualitative results on ArtGS-Multi dataset, including the initializations and the final reconstructions. We show the
Gaussians with their center points for a better visualization of their segmentation and motion parameters.
Table 7.
Computation overhead comparison on our multi-part
dataset. Time and memory are reported as mean±std.
Method
Time (min)
GPU Memory (GB)
DTA [46]
53.28±23.78
10.58±2.45
ArtGS [33]
10.41±0.89
2.94±0.04
Ours
16.93±2.37
2.69±0.23*
mented patches, and tend to merge the most relevant pro-
posals in the first optimization cycle, which limits the to-
tal number of proposals and cycles. Second, the optimiza-
tion of each cycle is monitored and stops early upon con-
vergence.
Table 7 reports the average computation time and GPU
memory usage across all objects in our dataset. Our method
achieves a practical balance between computational cost
and reconstruction quality.
Although ArtGS is faster, it
4

<!-- page 16 -->
𝑡= 0
𝑡= 0.5
𝑡= 1
𝑀𝑜𝑡𝑖𝑜𝑛
𝑡= 0
𝑡= 0.5
𝑡= 1
𝑀𝑜𝑡𝑖𝑜𝑛
𝑂𝑢𝑟𝑠
𝐴𝑟𝑡𝐺𝑆
𝐺𝑇
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝑊𝑖𝑛𝑑𝑜𝑤103238
𝑇𝑎𝑏𝑙𝑒34610
𝑇𝑎𝑏𝑙𝑒34178
𝑇𝑎𝑏𝑙𝑒33116
𝑇𝑎𝑏𝑙𝑒23372
𝑆𝑡𝑜𝑟𝑎𝑔𝑒47585
𝑆𝑡𝑜𝑟𝑎𝑔𝑒45759
𝑆𝑡𝑜𝑟𝑎𝑔𝑒40417
Figure 11. Additional reconstruction results on our dataset. We render the reconstructed articulated objects in different motion states
(t ∈{0, 0.5, 1} and their motion structures for each result.
cannot robustly reconstruct complex multi-part objects, as
shown in Tables 1–2 of the mainpage. DTA, on the other
hand, requires substantially more time and memory while
also failing on many multi-part cases.
G. Failure Cases and Robustness Analysis
Robustness to noisy depth and masks.
Fig. 14 shows
the robustness of our method to noisy depth maps and im-
precise segmentation masks.
For local geometry errors
(Fig. 14(a)) and imprecise mask boundaries near drawer
edges (Fig. 14(c)), our method mitigates the incorrect es-
timation from specific views by integrating multi-view in-
formation, achieving accurate reconstructions.
However,
our method would fail when the captured depth maps and
masks are highly fragmented and misaligned, as in the
real-scanned data provided by PARIS [24]. This can be
enhanced using prior-prompted depth predictor, e.g., Pri-
orDA [45], as we did to obtain our real-scanned data (the
printer and storage).
Part boundary sensitivity.
Although our approach
achieves much more robust motion estimation and recon-
struction of articulated objects than existing approaches, it
still suffers from the common issues of the self-supervised
3DGS reconstruction framework. We show two represen-
tative failure cases in Figure 13. First, since our method
lacks semantic part segmentation supervision for separat-
ing the adjacent movable parts, it sometimes causes the part
boundary to encroach on neighboring part regions, such as
the interior and the edge of the drawers in Figure 13.
Sensitivity to depth map quality. Second, the estimation
of mobility parameters in our method primarily relies on the
5

<!-- page 17 -->
𝐴𝑟𝑡𝐺𝑆
𝑂𝑢𝑟𝑠
𝐺𝑇
𝐴𝑟𝑡𝐺𝑆
𝑂𝑢𝑟𝑠
𝐺𝑇
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝐼𝑛𝑖𝑡𝑖𝑎𝑙𝑖𝑧𝑒
𝑅𝑒𝑠𝑢𝑙𝑡
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑠𝑢𝑙𝑡
𝐼𝑛𝑖𝑡𝑖𝑎𝑙𝑖𝑧𝑒
𝑅𝑒𝑠𝑢𝑙𝑡
𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒
𝑊𝑖𝑛𝑑𝑜𝑤103228
𝑇𝑎𝑏𝑙𝑒34178
𝑇𝑎𝑏𝑙𝑒23372
𝑆𝑡𝑜𝑟𝑎𝑔𝑒45759
𝑇𝑎𝑏𝑙𝑒34610
𝑇𝑎𝑏𝑙𝑒33116
𝑆𝑡𝑜𝑟𝑎𝑔𝑒47585
𝑆𝑡𝑜𝑟𝑎𝑔𝑒40417
Figure 12. Additional qualitative results on our dataset, including the initializations and the final reconstructions. We show the Gaussians
with their center points for a better visualization of their segmentation and motion parameters.
𝑅𝑒𝑐𝑜𝑛.
𝐼𝑛𝑖𝑡.
𝑅𝑒𝑐𝑜𝑛.
𝐼𝑛𝑖𝑡.
Part Boundary
Sensitivity to Depth Map Quality
Figure 13. Failure cases. We illustrate failure cases of low-quality part boundaries. For low-quality RGBD images input, our method
difficult to reconstruct the part boundaries.
(a) Noisy depth
(b) Reconstructed result 
(rendered Depth and RGB)
(c) Imprecise segmentation
Figure 14. Influence of noisy depth and mask.
depth component of the RGBD loss LI and the geometry-
aware Chamfer distance loss Lcd. Consequently, the accu-
racy of these estimations is highly dependent on the input
depth map quality. In scenarios where the acquired depth
maps contain significant inaccuracies, due to factors such
as sensor noise, occlusions, or varying illumination condi-
tions, the optimization process can be misled. As illustrated
in Figure 13 (using real-world data from the PARIS dataset),
this can cause our method to erroneously incorporate Gaus-
sians from the static part into the estimated movable part.
To achieve a lower loss value under these incorrect con-
straints, the optimization incorrectly transforms the mov-
able part into the interior of the object. It is worth noting
that in our real-world reconstruction experiments, the qual-
6

<!-- page 18 -->
ity of the mobility results has been significantly improved
by enhancing the input depth maps using the pre-trained
Depth-Anything-V2 model [53] which effectively mitigates
these issues.
Multi-DOF joints and non-rigid parts. Our current for-
mulation assumes that each movable part undergoes a sin-
gle rigid transformation, either revolute or prismatic. This
assumption does not hold for objects with multi-DOF joints
such as ball joints or compound hinges, or parts exhibiting
non-rigid deformations. In such cases, our motion parame-
terization cannot capture the full range of part motion, lead-
ing to inaccurate articulation estimation. Future work will
explore potential solutions including extending the motion
model to support composite transformations or integrating
learned deformation fields for non-rigid components.
7
