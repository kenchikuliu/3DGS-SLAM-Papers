<!-- page 1 -->
Multimodal LLM Guided Exploration and Active Mapping using
Fisher Information
Wen Jiang*1
Boshu Lei*1
Katrina Ashton1
Kostas Daniilidis1,2
1 University of Pennsylvania
1,2 Archimedes, Athena RC
Abstract
We present an active mapping system which plans for
both long-horizon exploration goals and short-term ac-
tions using a 3D Gaussian Splatting (3DGS) repre-
sentation.
Existing methods either do not take ad-
vantage of recent developments in multimodal Large
Language Models (LLM) or do not consider challenges
in localization uncertainty, which is critical in embod-
ied agents. We propose employing multimodal LLMs
for long-horizon planning in conjunction with detailed
motion planning using our information-based objec-
tive.
By leveraging high-quality view synthesis from
our 3DGS representation, our method employs a mul-
timodal LLM as a zero-shot planner for long-horizon
exploration goals from the semantic perspective.
We
also introduce an uncertainty-aware path proposal and
selection algorithm that balances the dual objectives of
maximizing the information gain for the environment
while minimizing the cost of localization errors. Experi-
ments conducted on the Gibson and Habitat-Matterport
3D datasets demonstrate state-of-the-art results of the
proposed method.
1. Introduction
Being able to autonomously explore and map an
environment while localizing within that map is a
core skill for a mobile robot. This ability could em-
power embodied artificial intelligence systems with ef-
fective 3D scene understanding through use in conjunc-
tion with vision-language features [60, 73, 94, 98] and
can be used as a basis for language-specified robotics
tasks [27, 28, 42, 68]. This task is challenging because
it requires general knowledge of the typical layout of
an environment to identify coarse targets for long-term
exploration, as well as the ability to plan each step to
ensure localization accuracy while maximizing informa-
tion gain or ‘surprise’ when traversing the environment.
�㻑
**Exploration Plan:**
1. **Current Situation Analysis:**
- The robot is currently located at the blue star marker.
...
2. **Observation of Candidates:**
- **Candidate 0:** Located towards the top of the scene...
...
3. **Selection Reasoning:**
- **Candidate 2** is favored for exploration as it is likely 
unvisited, near the previous frontier, and potentially opens up
new unexplored areas. It is a strategic choice for efficient 
area coverage without straying too far from the recently 
explored paths, reducing the risk of re-exploring already 
visited spaces.
```json
...
```
Please analyze the image and select a long-horizon goal... We 
are allowed to explore total of 2000 steps and this is the 
step 132... The current location of the robot is marked with 
the blue star(*) marker......
......
Please provide a detailed exploration plan and select an 
exploration target with reasons in the JSON format as shown 
below. 
···json ... ```
Figure 1. An overview about the interaction with a
multimodal LLM. The image on the left is a bird’s-eye
view rendering from our 3DGS representation. A detailed
example of the dialog can be found in the supplementary.
We use 3D Gaussian Splatting (3DGS) [36] as
scene representation and tackle this problem in two
phases: long-horizon planning with a Large Language
Model (LLM), and detailed motion planning with our
information-based approach by quantifying the Fisher
Information for 3DGS parameters and current local-
ization states. As shown in Fig. 1, we prompt a mul-
timodal LLM with our current map state, trajectory,
and frontiers [89], used as candidates for long-term ex-
ploration to exploit the LLM’s knowledge about scene
layouts. We then follow the suggestion from the LLM
by choosing paths that maximize the Expected Infor-
mation Gain (EIG) calculated using the Fisher Infor-
mation about both poses and the scene.
Classical active mapping methods often define ob-
jectives to reduce the uncertainty of the state estimate
based on its covariance [1, 7, 13, 21, 63, 85]. The co-
variance is readily available for classical systems which
use filters to update the state estimate. However, many
recent mapping systems use non-linear optimization to
update the state estimate, making the covariance dif-
ficult to obtain. In particular, systems with 3DGS for
the scene representation [29, 35, 49, 90] have been de-
1
arXiv:2410.17422v3  [cs.RO]  5 Sep 2025

<!-- page 2 -->
EIG 11.17 PSNR 21.07
EIG 32.57 PSNR 10.18
EIG 28.80, PSNR 29.22
EIG,20.21 PSNR 33.96
PSNR
EIG
Figure 2. Scatter Plot of EIG vs. PSNR We plot the
EIG and PSNR at sampled poses in the Cantwell scene of
the Gibson dataset. The figure corroborates the intuition
that the robot expects to gain little information (low EIG)
at well reconstructed region (high PSNR) and gain much
information (high EIG) at a poorly reconstructed region
(low PSNR).
veloped for high-fidelity rendering of novel views of the
scene.
Previous methods attempted to quantify the
uncertainty of radiance fields for reconstructing scenes
from given data [24, 71, 72, 77] for active view se-
lection [33, 41, 56, 77] and active object reconstruc-
tion [91] or mapping [33, 34, 39, 92] of scenes with
given localization, and for 3D reconstruction of small
scenes with an inward-facing camera [95]. However, all
the prior methods only model the uncertainty of the
scene representation, ignoring another primary source
of error for a practical mapping system: the risk of lo-
calization failures. Our proposed system could address
not only Information Gain during exploration but also
consider localization accuracy when the robot explores
unknown and texture-less regions.
In terms of long-horizon planning for robot explo-
ration, classical approaches such as frontier-based ex-
ploration [89] and A* algorithms are still used in ac-
tive mapping systems for their efficiency and sim-
plicity [6, 38, 75, 76, 84].
However, algorithms that
use only simple heuristics cannot determine the in-
formation gain, nor estimate 3D geometry, limiting
their objective to simply improving coverage or hav-
ing minimal travel distances.
To address this issue,
learnable approaches have been used for exploration
tasks [23, 43, 51, 64], but these methods are trained
for a limited scene distribution. Recently, LLMs and
Vision Language Models have demonstrated extraor-
dinary abilities in visual grounding and logical rea-
soning and have been studied with various robotic
tasks [10, 11, 16, 17, 28, 67, 81].
However, these
works do not focus on pure exploration or reconstruc-
tion tasks and many do not take advantage of photo-
realistic 3D representations for multimodal LLMs.
We argue that we leverage the best of both worlds by
using a multi-modal LLM to identify long-horizon ex-
ploration goals and planning paths to them using our
information-based algorithm. Our approach is seam-
lessly integrated with our 3DGS scene representation,
we utilize a high-quality bird’s-eye view rendering from
the scene representation to prompt the LLM and can
directly extract the EIG for path planning from it using
Fisher Information. As shown in Fig. 2, our EIG met-
ric reflects the ‘surprise’ and correlates well with the
rendering quality on candidate views, as measured by
the peak signal-to-noise ratio (PSNR) between the true
and rendered image, without actually taking a picture
at the candidate location. Notably, we compute the full
Fisher Information matrix for localization parameters
without any approximation with our efficient CUDA
implementation.
We also derived localization uncer-
tainty on candidate paths with the Cramér–Rao bound,
and we use this along with the EIG in our final ob-
jective to balance the dual problem of exploration and
localization. To validate our approach, we evaluate our
method on scenes from the Gibson [88] and Habitat-
Matterport 3D [61] datasets quantitatively and qual-
itatively. We show superior reconstruction quality in
various metrics compared to several baselines and re-
cent state-of-the-art methods [9, 23, 58, 89, 92].
Our contributions can be summarized as follows:
• We present an active mapping system for ground
robots that could autonomously explore the envi-
ronment and extensively compare our system with
previous approaches.
To the best of our knowl-
edge, we are the first active mapping system with
3D Gaussian representation that is not dependent
on ground truth camera pose readings.
• We provide a way to leverage the zero-shot long-
horizon planning ability of LLM into our active
mapping system seamlessly.
• We introduce localization uncertainty with active
mapping systems and effectively balance the infor-
mation gain for exploration and the cost of possible
localization errors.
2. Related work
Active Mapping and Localization
Efficiently ex-
ploring an environment in order to map it while being
able to localize in that map is a fundamental problem in
robotics. There are many methods that address aspects
of this problem separately – Simultaneous Localiza-
tion and Mapping (SLAM) methods [5, 18, 19, 32, 52–
54, 79, 82] address the mapping and localization. Many
exploration methods [12, 23, 33, 34, 39, 92] address the
exploration and mapping aspects while assuming poses
are provided.
Active SLAM methods [6, 75, 76, 84]
consider both of these problems. The exploration in
these systems is usually driven by a measure of uncer-
2

<!-- page 3 -->
tainty [4, 26, 44, 59]; specific utility functions are often
drawn from either Information Theory (IT) [70], or the
Theory of Optimal Experimental Design (TOED) [57].
Recently, learning-based approaches have been de-
veloped for active mapping and localization; Active
Neural SLAM [9] learns policies to drive exploration
and estimate the agent pose, more similarly to us
NARUTO [22] uses an SLAM backbone to estimate the
pose and uses uncertainty to drive exploration. How-
ever, they only consider the reconstruction uncertainty,
whereas we choose long-horizon planning goals and
consider the localization uncertainty as well as the re-
construction uncertainty when planning the best path
to the long-horizon goal.
Uncertainty quantification for radiance fields
The vast majority of previous work on uncertainty
quantification for radiance fields has been for post-
processing scenes [24, 71, 72, 77, 80], view selec-
tion [33, 41, 56, 77] or active view selection [33, 56, 77],
all of which assume the input images are posed. Active
neural mapping [39, 92] uses neural variability, that
is, the prediction robustness against random weight
perturbations, as an estimate of uncertainty to ac-
tively map a scene with ground truth poses provided.
Fisher-RF [33] also performs active scene mapping with
ground truth poses provided, based on an approxima-
tion of the Fisher Information of views along candidate
paths. Zhan et al. [95] performs active reconstruction
without ground truth camera poses.
However, they
only evaluate small-scale scenes and limit the camera
trajectories to be inwards facing and only model scene
uncertainty, not localization uncertainty, which is a key
consideration for active mapping and exploration.
Robot
Planning
with
Foundation
Models
LLMs and VLMs have been used widely in robotics as
for language-specified tasks, either as high-level plan-
ners in conjunction with other methods for low-level
control [3, 16, 17, 28, 30, 45, 62, 74], to create interme-
diate representations which can be planned over [31]
or directly outputting actions [46, 97].
Outside of
language-driven tasks, VLMs and LLMs have been
used in robotics for providing rewards to drive explo-
ration in Reinforcement Learning [83], and for visual
localization [50]. More similar to our task, LLMs and
VLMs have also been used for goal-driven navigation,
such as searching for specific objects [15, 65, 69, 93].
3. Method
We divide the scene exploration task into two
phases: (a) long-horizon planning for the coarse direc-
tion of exploration that leads to better coverage and
understanding (Sec. 3.2), and (b) detailed trajectory
proposal and path selection that aims to improve 3D re-
construction (Sec. 3.3) and localization (Sec. 3.4) from
a geometrical perspective. The former task relies more
on semantic information and prior knowledge about the
possible layout of the environment, which is suitable
for a generalist model with prior distributions on the
scene. The latter task, however, involves detailed mo-
tion planning and is better addressed by considering
information gain on fine-level geometries. Fig. 3 shows
an overview of our method.
3.1. Preliminary
In 3D Gaussian Splatting (3DGS) [36], the scene is
represented by a set of 3D Gaussians whose color and
opacity are learned via a rendering loss. An image can
be rendered by projecting the Gaussians to 2D and
using α-blending for the N ordered points on the 2D
splat that overlaps each pixel.
The Jacobian of the
current camera pose x with respect to the parameters
of 3D Gaussians can be computed by defining the (left)
partial derivative on the manifold [49]:
Df(x)
Dx
≜lim
τ→0
Log(f(Exp(τ) ◦x) ◦f(x)−1)
τ
,
(1)
where τ ∈se(3), ◦is a group composition, and Exp
and Log are the exponential and logarithmic mappings
between Lie algebra and Lie Group.
Fisher Information is a measurement of the infor-
mation that a random variable y carries about an un-
known parameter w of a distribution that models y.
In the problem of novel view synthesis, we are inter-
ested in measuring the observed information of a radi-
ance field with parameters w at a camera pose x using
the negative log-likelihood of the image observation y
taken from that pose:
−log p(y|x; w) = (y −f(x, w))T (y −f(x, w)),
(2)
where f(x, w) is the rendering model.
Under reg-
ularity conditions [66],
the Fisher Information of
−log p(y|x; w) is the Hessian of Eq. 2 with respect
to w, denoted H′′[y|x, w].
In our formulation y ∈
Rh×w×c is the RGB-D observation, and w is a tensor
for all the 3D Gaussian Parameters RN×14, where N is
the number of Gaussians and the parameters are the 3D
means, RGB colors, unormalized rotations represented
as quaternions, opacities and per-axis log-scales.
FisherRF [33] uses Fisher Information to address ac-
tive perception problems such as active view selection
and active mapping, although they do not consider lo-
calization uncertainty. Given a training set of views
Dtrain and a set of candidate trajectories {Pj} Fish-
3

<!-- page 4 -->
Scene Fisher Information
3D Gaussian Representation
Action
3D Gaussian 
SLAM Backend
RGBD Capture
𝐶!"#$! 𝐿𝑜𝑔𝑥%
Path 
Selection 
𝐸𝐼𝐺(𝑥&)
𝐸𝐼𝐺(𝑥')
𝐸𝐼𝐺(𝑥()
𝐸𝐼𝐺(𝑥))
𝐸𝐼𝐺(𝒫* )
Total EIG of a Path
LL M Guided Exploration
...
...
𝒫&
𝒫(
𝒫*
Path Proposal
Region of Interest
Localization Uncertainty
Long-term Planning
Short-term Planning
Figure 3. An Illustration of Our Active Mapping System Our method first identifies long-horizon targets with a
multimodal LLM by using novel-view synthesis from our 3DGS representation. Our information-based path proposal method
then proposes and selects short-term action sequences from the region of interest the multimodal LLM identifies. The best
path and action along the path is selected concerning both the information gain and localization accuracy.
erRF selects the path which maximizes the following:
tr




X
xacq
i
∈Pj
H′′[yacq
i
|xacq
i
, w]

H′′[w|Dtrain]−1


(3)
where w is the initial estimate of model parameters us-
ing the current training set. Crucially, the Fisher Infor-
mation H′′[yacq
i
|xacq
i
, w] does not depend on the label
yacq
i
of the acquisition sample xacq
i
, thus it is feasible to
compute it before visiting the potential view candidate
xacq
i
. However, the number of optimizable parameters
is typically more than 20 million, which means it is im-
possible to compute without sparsification or approx-
imation. In practice, FisherRF [33] applies a Laplace
approximation [14, 47] that approximates the Hessian
matrix with its diagonal values plus a log-prior regu-
larizer λI as follows
H′′[y|x, w] ≃diag(∇wf(x, w)T ∇wf(x, w)) + λI. (4)
3.2. Long Horizon Exploration with Foundation
Model
We propose using a multimodal LLM as a zero-
shot long-horizon planner and leaving the detailed path
planning to a closed-form uncertainty-aware motion
planning algorithm.
As we consider an agent moving in a 2D action plane
(e.g., a ground robot), we leverage the view synthe-
sis ability of 3DGS to create expressive bird’s-eye view
renderings that could provide an overview of the envi-
ronment. An occupancy grid on the motion plane of
our robot is created using our 3D Gaussian represen-
tation, which can be used to identify the frontiers of
the current environment. The frontiers are defined as
points a set of neighboring points on the action plane
on the boundary between free space and unobserved
space. The agent has many choices of frontiers, espe-
cially during the early stages of exploration. We ap-
ply Chain-of-Thought Prompting [87] to encourage the
multimodal LLM to provide analysis on the candidate
frontiers first before selection. We provide contextual
information such as the description of the task, the to-
tal steps allowed for exploration, and our current step
in the textual prompt. The LLM is also allowed to de-
cide whether the robot should not go to a frontier and
instead focus on improving existing regions. We anno-
tate the rendered map with the current location, pre-
viously visited trajectory and the frontiers which form
the candidate long-term navigation goals. We prompt
the LLM with both the contextual text information and
annotated map as shown in Fig. 1 and extract the long-
term goal from its output. By using a set of possible
candidates we avoid possible infeasible destinations for
motion planning. For example, the unexplored regions
we have no reconstruction for at this point might be
unreachable from the current space. We can also check
that the LLM’s output is actually one of the provided
options before proceeding, if it is not then we instead
select the largest frontier. We use GPT-4o [55] for our
long-horizon planning task, but our method is agnostic
to the underlying multimodal LLM as the inputs for the
4

<!-- page 5 -->
LLM are text prompts and a bird’s-eye view rendering.
Detailed examples can be found in the Supplementary.
3.3. Expected Information Gain for Path Proposal
and Selection
We use the Expected Information Gain (EIG) as
both a preliminary method to further refine our candi-
date poses and as part of our path selection criteria.
After identifying a long-horizon goal with the mul-
timodal LLM, we form an initial set of candidate poses
TI by sampling points in the coarse region of interest
from the LLM. If there are no unvisited boundaries or
the LLM suggests we do not need to explore the fron-
tiers, we sample poses across the free space to form
TI. We then evaluate the Expected Information Gain
(EIG) for each pose xacq
i
∈TI, given by
EIG(xacq
i
) = tr
 H′′[yacq
i
|xacq
i
, w] I(w)−1
,
(5)
as a preliminary selection metric to form our final can-
didate target poses set TF .
We calculate I(w) dif-
ferently to FisherRF [33], which approximates it us-
ing H′′[w|Dtrain] by computing the Hessians on the
training set.
This is also known as empirical Fisher
Information, whose limitations have been widely dis-
cussed [40, 48]. In most scenarios, this is a reluctant
design choice because the distribution of x ∼p(x) is
unknown (i.e., the distribution of all possible images).
However, x ∼p(x) in our case is tractable because it
represents the possible locations where we can take an
observation for the environment, i.e. the free space of
our map. Therefore, unlike FisherRF [33] as described
in Eq. 3, we propose to use Monte-Carlo sampling to
compute the Fisher Information of the current model
I(w) = Ex∼p(x) [H′′[y|x, w]] ≃
N
X
k=1
H′′[yk|xk, w], (6)
where xk is drawn from a uniform distribution of cam-
era poses in the free space of the current map which
we use to approximate p. We also uniformly initialize
3D Gaussians in the space, which will be subsequently
updated with rendering losses for visited regions.
After the set of final candidate poses TF has been
formed, we construct paths to each candidate pose with
the A* algorithm [25] using the occupancy map. The
path can be defined as an ordered set of camera poses
from the current location xt at exploration step t to
the frontier points xj
T from the long horizon planning.
Pj = {xj
t+1, . . . , xj
T }
(7)
To evaluate how beneficial a path Pj would be for
improving the reconstruction of the scene we consider
the EIG for the 3D Gaussian parameters along that
path, which can be computed as the sum over the path
of the following term [33]:
EIGPj,i(xi) = tr
 H′′[yi|xi, w] IPj,i(w)−1
(8)
where IPj,i(w) takes the mutual information along the
path into account as follows
IPj,i(w) = I(w) +
X
xt∈Pj,t<i
I(w|xt).
(9)
3.4. Localization Uncertainty for Path Selection
If solely maximizing the EIG, the robot will be more
likely to explore unvisited regions. However, explor-
ing regions that have not been well reconstructed also
means the agent would have the risk of worse localiza-
tion accuracy due to noise and ambiguities in the unre-
constructed regions during pose optimization. The cost
of localization must thus be considered during path
planning to balance the importance of exploring new
environments with maintaining localization accuracy.
We propose to use Fisher Information as a measure-
ment for localization uncertainty to address this. Dur-
ing optimization, we essentially optimize on the loga-
rithmic mapping τi ≜Log(xi) of our camera pose. By
the Cramér–Rao bound, the covariance of τi ∈se(3)
can be lower-bounded with the inverse of Fisher Infor-
mation matrix I(τi):
Cov(T(ˆτi)) ≥I(τi)−1
(10)
where T(τi) is an unbiased estimator for τi solved by
iteratively optimizing photo-metric loss. Hence, we can
define the localization cost Clocal at a pose xi in terms
of τi as:
Clocal(τi) = −log det(∇τif(τi, w)T ∇τif(τi, w)) (11)
Matsuki et al. [49] computed the Jacobians of the
camera pose with respect to the mean and covariances
of each gaussian ∂µI
∂x and ∂ΣI
∂x . However, we need to
compute the Jacobian of τi with respect to the render-
ing output:
∇τif(τi, w) = ∂f(τi, w)
∂τi
=
h
∂f(τi,w)
∂µI
∂f(τi,w)
∂ΣI
i "
DµC
Dτi
DW
Dτi
#
(12)
Unlike 3DGS parameters, our computation of the lo-
calization cost can be directly computed on the full
Fisher Information matrix without using the Laplace
approximation [14, 47]. That is because the full Jaco-
bian ∇τif(τi, w) is linear to the number of Gaussians.
5

<!-- page 6 -->
ANS [9]
Active-INR [92]
UPEN [23]
ExplORB [58]
FBE [89]
Ours
Figure 4. Qualitative Comparison for Final Scene Reconstruction on Gibson Dataset Greigsville (top) and Ribera
(bottom) scenes. We provide top-down rendering for different methods. Note that UPEN and Active-INR use GT pose in
this visualization.
UPEN [23]
FBE [89]
Ours
UPEN [23]
FBE [89]
Ours
Figure 5.
Qualitative Comparison for Final Scene Reconstruction on Habitat-Matterport 3D Dataset
oPj9qMxrDEa (left) and QKGMrurUVbk (right) scenes. We provide top-down rendering for different methods.
We select the best path by minimizing the total cost
for all viewpoints xi along a path Pj:
arg min
Pj
X
xi∈Pj
Clocal(Log(xi))−η log(EIGPj,i(xi)) (13)
where η is a hyper-parameter controlling the relative
importance between EIG and localization accuracy.
Our active mapping system constantly updates the
map, and we replan using our active path planning
algorithm if we detect the agent is getting close to a
possible obstacle or upon reaching the end of the pre-
viously selected path.
4. Experiments
4.1. Experimental Set-up
Dataset
Following previous methods [23, 92], our
algorithm is evaluated in the Habitat Simulator [78]
on
the
Gibson
[88]
and
Habitat-Matterport
3D
(HM3D) [61] datasets comprised of indoor scenes re-
constructed from scans of real houses. We adopt the
default start point in the Habitat Simulator as the
starting point for exploration in each scene. The to-
tal number of steps for each experiment is 2000. The
system takes color and depth images at the resolution
of 800x800 and outputs a discrete action at each step.
The action space consists of MOVE FORWARD by
5cm, TURN LEFT, and TURN RIGHT by 5°. The
field of view (FOV) is set to 90° vertically and horizon-
tally. Please refer to the supplement for more details
about the evaluation split and other hyper-parameters.
Metrics
We evaluated our method using the Peak-
signal-to-noise ratio (PSNR), Structural Similarity In-
dex Measure (SSIM) [86], Learned Perceptual Image
Patch Similarity (LPIPS) [96] for RGB rendering and
mean absolute error (MAE) for depth rendering as met-
rics for scene reconstruction quality. We calculate these
metrics using 2000 points uniformly sampled from the
movement plane of the agent in the scene, discard-
ing any points that are not navigable. We argue that
the rendering quality reflects both reconstruction qual-
ity and pose accuracy because high tracking accuracy
would help the training of the 3D Gaussian Splatting
model. Meanwhile, misaligned poses will lead to mis-
aligned rendering at test time thus leading to inferior
results. Following previous approaches [39, 92], we also
use the completion ratio as an evaluation metric. To
evaluate the pose estimation accuracy, we use the root
mean squared average tracking error (RMSE ATE),
but as the trajectories for each method are different,
the RMSE ATE should only be considered along with
other metrics such as completness.
6

<!-- page 7 -->
Method
PSNR ↑
SSIM ↑
LPIPS ↓
Depth MAE ↓
RMSE ATE ↓
Completeness (%) ↑
ANS [9]
16.34
0.6818
0.3923
0.3886
0.1105
35.33
UPEN [23]
16.44
0.6678
0.4134
0.4841
0.5158
22.66
ExplORB [58]
18.99
0.7175
0.3994
0.2664
0.2296
30.23
FBE [89]
21.45
0.7618
0.2126
0.1028
0.1680
55.87
Ours
23.28
0.8067
0.2507
0.0696
0.0226
84.38
Table 1. Quantitative Evaluation on Reconstruction Quality and Tracking Accuracy on Gibson dataset
Method
PSNR ↑
SSIM ↑
LPIPS ↓
Depth MAE ↓
RMSE ATE ↓
Completeness (%) ↑
UPEN [23]
12.23
0.4795
0.5157
0.7356
0.4393
17.48
ExplORB (gt) [58]
17.81
0.3694
0.6810
0.5071
-
31.92
FBE [89]
15.80
0.5952
0.4392
0.4085
1.2004
22.42
Ours
19.86
0.7127
0.4122
0.1666
0.0336
49.76
Table 2. Quantitative Evaluation on Reconstruction Quality and Tracking Accuracy on HM3D dataset
Baselines
We compare to two exploration meth-
ods which assume ground truth pose:
UPEN [23]
and Active Neural Mapping (active-INR) [92].
We
also compare our method with Active Neural SLAM
(ANS) [9], explORB [58] and Frontier Based Explo-
ration (FBE) [89] without ground truth pose provided.
Note that we do not run ANS on HM3D as it is not
trained on this dataset.
To compare the rendering quality fairly, we run all
the baselines using the MonoGS [49] backend for recon-
struction. We run UPEN and FBE online, but for ANS,
active-INR, and ExplORB, we record and playback tra-
jectories obtained using their source code. Because the
forward step size for ANS is much larger than for our
method, we interpolate the trajectory so that the for-
ward step size matches our method’s to make the steps
comparable.
For ExplORB, since the official imple-
mentation is based on MoveBase, which uses velocity
commands, we sample the trajectory at 5 Hz. We also
found that ANS, active-INR, and UPEN failed on some
scenes due to localization failure of the MonoGS back-
end. ANS produces a pose estimate (using information
from noisy pose sensors not provided to our pipeline),
so we set the pose estimate of the MonoGS backend to
the one from ANS. As active-INR and UPEN do not
produce a pose estimate, we evaluate them using the
ground-truth pose.
4.2. Comparison Against Previous Methods
Table 1 shows the results of our method and the
baselines for exploration in scenes from the Gib-
son dataset [88], and Table 2 shows the results on
HM3D [8]. Our method outperforms the baselines on
all metrics. We further qualitatively compare the re-
construction qualities after active exploration in Fig. 4
and Fig. 5, and the trajectories in Fig. 7. Our method
does not have major errors from failed localization, and
we have fewer gaps in the scenes than other meth-
ods.
For example, in the Ribera scene, all methods
except for us and FBE miss the bathroom at the bot-
tom left, and FBE misses more of the area around the
sofa than us. We show the trajectories taken for the
Cantwell scene from the Gibson dataset, which is a rel-
atively large and challenging scene. We show only a
few baselines to keep the figure legible.
We can see
that ANS [9] does not go to the region on the bottom
left, Active-INR [92] stays in the center region, and
FBE does not go to the region on the bottom right,
whereas our method efficiently visits most areas of the
scene.
4.3. Ablation Study and Runtime Analysis
To validate the effectiveness of localization uncer-
tainty, hyperparameters, and LLM, we performed an
ablative study of each component in our system on the
HM3D dataset in Table. 3. As can be seen, the aver-
age trajectory error is much lower with the localization
uncertainty than without.
In addition, after adding
LLM, the model performs better in all metrics.
We
also noticed the tracking error was lower after intro-
ducing multimodal LLM to the active mapping system.
We hypothesize this improvement is because improved
scene reconstruction can also help reduce tracking er-
rors.
We additionally provide a breakdown of the runtime
of our method, shown in Fig. 6.
7

<!-- page 8 -->
Method
PSNR ↑
SSIM ↑
LPIPS ↓
Depth MAE ↓
RMSE ATE ↓
Completeness (%) ↑
w.o. LLM & Localization Uncertainty
16.15
0.6550
0.6193
0.3409
0.2478
35.40
w.o. LLM
16.94
0.6799
0.5847
0.2887
0.1694
37.26
Ours (Llava-7b)
18.46
0.6805
0.4623
0.2033
0.0159
17.41
λ = 2 × 10−6 †
18.90
0.6976
0.4408
0.1966
0.0479
18.12
λ = 5 × 10−6 †
18.90
0.6946
0.4602
0.2145
0.0551
18.05
Ours†
19.86
0.7127
0.4122
0.1666
0.0336
49.76
Table 3. Ablation Study of Localization Uncertainty Term on Scenes from the HM3D Dataset. We compare
our method with and without the localization uncertainty term to validate that including it improves localization and
reconstruction. †: the multimodal LLM is GPT-4o in the experiment.
0
250
500
750
1000
1250
1500
1750
2000
Steps
10
2
10
1
100
101
102
Time/s
Time Analysis Across Steps
Action Evaluation
Action Planning
Global Planning
Mapping
Backend Overhead
Tracking
Action Evaluation
11.7%
Action Planning
6.8%
Global Planning
41.5%
Mapping
1.3%
Backend Overhead
2.0%
Tracking
36.7%
Total Time Distribution
Figure 6. Runtime Analysis of the Active Mapping System. We provide running time statistics by each step and by
the percentage of total time consumed on one episode.
ANS
Active-INR
  Ours
FBE
Figure 7. Qualitative Comparison for Trajectories on
Cantwell from Gibson Dataset.
5. Conclusion
We present an active mapping system that can au-
tonomously explore an environment, the first method
to do so using a 3D Gaussian representation without
ground truth camera poses.
By rendering maps us-
ing our scene representation, we can seamlessly query
a multimodal Large Language Model for long-horizon
planning to leverage its knowledge of scene layouts.
This allows the robot to efficiently explore the scene
while avoiding setting unreachable navigation goals.
We then plan an optimal path to the long-horizon goal
using our information-driven path proposal and selec-
tion algorithm, which balances the information gain
with respect to the acene against the localization un-
certainty, ensuring that the robot captures the geo-
metric details of the scene for a high-quality recon-
struction. We evaluate our method on scenes from the
Gibson [88] and Habitat-Matterport 3D [61] datasets
in terms of the rendering quality, completeness, and
average tracking error. To enable our method to sup-
port more robotics applications, future work could ex-
tend it to take advantage of our 3D scene represen-
tation and consider movement with higher degrees of
freedom (DOF) than the currently supported 3DOF.
Our method could also be extended to task-driven ex-
ploration such as finding objects [2, 37], leveraging the
LLM for efficient search. Incorporating semantic fea-
tures [60, 73, 98] to allow for grounding language to the
scene would also enable many robotics and computer
vision applications.
8

<!-- page 9 -->
Acknowledgements
The
authors
gratefully
ap-
preciate support through the following grants:
NSF
FRR 2220868,
NSF IIS-RI 2212433,
NSF TRIPODS
1934960,
NSF
CPS
2038873.
The
authors
appre-
ciate
the
help
from
Prof.
Pratik
Chaudhari
for
the
insightful
discussion
and
hardware
platforms.
References
[1] Arash Asgharivaskasi and Nikolay Atanasov. Semantic
octree mapping and shannon mutual information com-
putation for robot exploration. IEEE Transactions on
Robotics, 39(3):1910–1928, 2023. 1
[2] Dhruv Batra, Aaron Gokaslan, Aniruddha Kembhavi,
Oleksandr Maksymets,
Roozbeh Mottaghi,
Mano-
lis Savva,
Alexander Toshev,
and Erik Wijmans.
ObjectNav Revisited:
On Evaluation of Embod-
ied Agents Navigating to Objects.
arXiv preprint
arXiv:2006.13171, 2020. 8
[3] Anthony Brohan, Yevgen Chebotar, Chelsea Finn,
Karol Hausman, Alexander Herzog, Daniel Ho, Julian
Ibarz, Alex Irpan, Eric Jang, Ryan Julian, et al. Do
as i can, not as i say: Grounding language in robotic
affordances.
In Conference on robot learning, pages
287–318. PMLR, 2023. 3
[4] Adam Bry and Nicholas Roy. Rapidly-exploring ran-
dom belief trees for motion planning under uncer-
tainty.
In 2011 IEEE international conference on
robotics and automation, pages 723–730. IEEE, 2011.
3
[5] Carlos Campos, Richard Elvira, Juan J Gómez Ro-
dríguez, José MM Montiel, and Juan D Tardós. Orb-
slam3:
An accurate open-source library for visual,
visual–inertial, and multimap slam.
IEEE Transac-
tions on Robotics, 37(6):1874–1890, 2021. 2
[6] Luca Carlone, Jingjing Du, Miguel Kaouk Ng, Basilio
Bona, and Marina Indri. Active slam and exploration
with particle filters using kullback-leibler divergence.
Journal of Intelligent & Robotic Systems, 75:291–311,
2014. 2
[7] Henry Carrillo, Ian Reid, and José A Castellanos. On
the comparison of uncertainty criteria for active slam.
In 2012 IEEE International Conference on Robotics
and Automation, pages 2080–2087. IEEE, 2012. 1
[8] Angel Chang, Angela Dai, Thomas Funkhouser, Ma-
ciej Halber, Matthias Niessner, Manolis Savva, Shuran
Song, Andy Zeng, and Yinda Zhang. Matterport3d:
Learning from rgb-d data in indoor environments. In-
ternational Conference on 3D Vision, 2017. 7
[9] Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh
Gupta, Abhinav Gupta, and Ruslan Salakhutdinov.
Learning to explore using active neural slam. In ICLR,
2020. 2, 3, 6, 7, 16
[10] Jiaqi Chen, Bingqian Lin, Xinmin Liu, Lin Ma, Xi-
aodan Liang, and Kwan-Yee K Wong.
Affordances-
oriented planning using foundation models for con-
tinuous vision-language navigation.
arXiv preprint
arXiv:2407.05890, 2024. 2
[11] Jiaqi Chen, Bingqian Lin, Ran Xu, Zhenhua Chai, Xi-
aodan Liang, and Kwan-Yee Wong.
Mapgpt: Map-
guided prompting with adaptive path planning for
vision-and-language navigation. In Proceedings of the
62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
9796–9810, 2024. 2
[12] Tao Chen, Saurabh Gupta, and Abhinav Gupta.
Learning exploration policies for navigation.
In In-
ternational Conference on Learning Representations,
2019. 2
[13] Yongbo Chen, Shoudong Huang, and Robert Fitch.
Active slam for mobile robots with area coverage and
obstacle avoidance.
IEEE/ASME Transactions on
Mechatronics, 25(3):1182–1192, 2020. 1
[14] Erik Daxberger, Agustinus Kristiadi, Alexander Im-
mer, Runa Eschenhagen, Matthias Bauer, and Philipp
Hennig. Laplace redux–effortless Bayesian deep learn-
ing. In NeurIPS, 2021. 4, 5
[15] Vishnu Sashank Dorbala, James F Mullen Jr, and Di-
nesh Manocha. Can an embodied agent find your “cat-
shaped mug”? llm-based zero-shot object navigation.
IEEE Robotics and Automation Letters, 2023. 3
[16] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey
Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan
Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu,
Wenlong Huang, Yevgen Chebotar, Pierre Sermanet,
Daniel Duckworth, Sergey Levine, Vincent Vanhoucke,
Karol Hausman, Marc Toussaint, Klaus Greff, Andy
Zeng, Igor Mordatch, and Pete Florence.
Palm-e:
An embodied multimodal language model. In arXiv
preprint arXiv:2303.03378, 2023. 2, 3
[17] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia,
Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe
Yu, Pieter Abbeel, Joshua B Tenenbaum, et al. Video
language planning. arXiv preprint arXiv:2310.10625,
2023. 2, 3
[18] Jakob Engel, Thomas Schöps, and Daniel Cremers.
Lsd-slam: Large-scale direct monocular slam. In Eu-
ropean conference on computer vision, pages 834–849.
Springer, 2014. 2
[19] Jakob Engel, Vladlen Koltun, and Daniel Cremers.
Direct sparse odometry.
IEEE transactions on pat-
tern analysis and machine intelligence, 40(3):611–625,
2017. 2
[20] Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xi-
aowei Xu, et al.
A density-based algorithm for dis-
covering clusters in large spatial databases with noise.
In KDD, pages 226–231, 1996. 14
[21] Hans Jacob S Feder, John J Leonard, and Christo-
pher M Smith. Adaptive mobile robot navigation and
mapping. The International Journal of Robotics Re-
search, 18(7):650–668, 1999. 1
[22] Ziyue Feng, Huangying Zhan, Zheng Chen, Qingan
Yan, Xiangyu Xu, Changjiang Cai, Bing Li, Qilun Zhu,
9

<!-- page 10 -->
and Yi Xu. Naruto: Neural active reconstruction from
uncertain target observations.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 21572–21583, 2024. 3
[23] Georgios Georgakis, Bernadette Bucher, Anton Ara-
pin, Karl Schmeckpeper, Nikolai Matni, and Kostas
Daniilidis. Uncertainty-driven planner for exploration
and navigation. In ICRA, 2022. 2, 6, 7, 16, 17
[24] Lily Goli, Cody Reading, Silvia Sellán, Alec Jacobson,
and Andrea Tagliasacchi.
Bayes’ Rays: Uncertainty
quantification in neural radiance fields. arXiv, 2023.
2, 3
[25] Peter E Hart, Nils J Nilsson, and Bertram Raphael.
A formal basis for the heuristic determination of min-
imum cost paths. IEEE transactions on Systems Sci-
ence and Cybernetics, 4(2):100–107, 1968. 5
[26] Ruijie He, Emma Brunskill, and Nicholas Roy. Puma:
Planning under uncertainty with macro-actions.
In
Proceedings of the AAAI Conference on Artificial In-
telligence, pages 1089–1095, 2010. 3
[27] Jiawei Hou, Tianyu Wang, Tongying Pan, Shouyan
Wang, Xiangyang Xue, and Yanwei Fu.
TaMMa:
Target-driven multi-subscene mobile manipulation. In
8th Annual Conference on Robot Learning, 2024. 1
[28] Chenguang Huang, Oier Mees, Andy Zeng, and Wol-
fram Burgard. Visual language maps for robot nav-
igation.
In 2023 IEEE International Conference on
Robotics and Automation (ICRA), pages 10608–10615.
IEEE, 2023. 1, 2, 3
[29] Huajian Huang, Longwei Li, Cheng Hui, and Sai-Kit
Yeung. Photo-slam: Real-time simultaneous localiza-
tion and photorealistic mapping for monocular, stereo,
and rgb-d cameras. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recogni-
tion, 2024. 1
[30] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and
Igor Mordatch. Language models as zero-shot plan-
ners: Extracting actionable knowledge for embodied
agents. In International conference on machine learn-
ing, pages 9118–9147. PMLR, 2022. 3
[31] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu
Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable
3d value maps for robotic manipulation with language
models. arXiv preprint arXiv:2307.05973, 2023. 3
[32] Shahram Izadi, David Kim, Otmar Hilliges, David
Molyneaux,
Richard Newcombe,
Pushmeet Kohli,
Jamie Shotton, Steve Hodges, Dustin Freeman, An-
drew Davison, et al. Kinectfusion: real-time 3d recon-
struction and interaction using a moving depth cam-
era. In Proceedings of the 24th annual ACM sympo-
sium on User interface software and technology, pages
559–568, 2011. 2
[33] Wen Jiang, Boshu Lei, and Kostas Daniilidis. Fish-
errf:
Active view selection and mapping with radi-
ance fields using fisher information. In ECCV, page
422–440, 2024. 2, 3, 4, 5
[34] Rui Jin, Yuman Gao, Yingjian Wang, Yuze Wu, Hao-
jian Lu, Chao Xu, and Fei Gao.
Gs-planner:
A
gaussian-splatting-based planning framework for ac-
tive high-fidelity reconstruction. In 2024 IEEE/RSJ
International Conference on Intelligent Robots and
Systems (IROS), pages 11202–11209. IEEE, 2024. 2
[35] Nikhil Keetha, Jay Karhade, Krishna Murthy Jataval-
labhula, Gengshan Yang, Sebastian Scherer, Deva Ra-
manan, and Jonathon Luiten. Splatam: Splat, track &
map 3d gaussians for dense rgb-d slam. arXiv preprint
arXiv:2312.02126, 2023. 1
[36] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuh-
ler, and George Drettakis. 3d gaussian splatting for
real-time radiance field rendering. ACM Transactions
on Graphics, 42(4), 2023. 1, 3, 14
[37] Mukul Khanna, Ram Ramrakhya, Gunjan Chhablani,
Sriram Yenamandra,
Theophile Gervet,
Matthew
Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv
Batra, and Roozbeh Mottaghi. Goat-bench: A bench-
mark for multi-modal lifelong navigation.
In Pro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 16373–16383,
2024. 8
[38] Ayoung Kim and Ryan M Eustice. Active visual slam
for robotic area coverage: Theory and experiment. The
International Journal of Robotics Research, 34(4-5):
457–475, 2015. 2
[39] Zijia Kuang, Zike Yan, Hao Zhao, Guyue Zhou, and
Hongbin Zha.
Active neural mapping at scale.
In
2024 IEEE/RSJ International Conference on Intelli-
gent Robots and Systems (IROS), pages 7152–7159.
IEEE, 2024. 2, 3, 6
[40] Frederik Kunstner, Philipp Hennig, and Lukas Balles.
Limitations of the empirical fisher approximation for
natural gradient descent. Advances in neural informa-
tion processing systems, 32, 2019. 5
[41] Soomin Lee, Le Chen, Jiahao Wang, Alexander Lin-
iger, Suryansh Kumar, and Fisher Yu.
Uncertainty
guided policy for active robotic 3d reconstruction using
neural radiance fields. IEEE Robotics and Automation
Letters, 7(4):12070–12077, 2022. 2, 3
[42] Peiqi Liu, Yaswanth Orru, Chris Paxton, Nur Muham-
mad Mahi Shafiullah, and Lerrel Pinto.
Ok-robot:
What really matters in integrating open-knowledge
models for robotics. arXiv preprint arXiv:2401.12202,
2024. 1
[43] Xu Liu, Ankit Prabhu, Fernando Cladera, Ian D
Miller, Lifeng Zhou, Camillo J Taylor, and Vijay
Kumar.
Active metric-semantic mapping by multi-
ple aerial robots.
In 2023 IEEE International Con-
ference on Robotics and Automation (ICRA), pages
3282–3288. IEEE, 2023. 2
[44] Iker Lluvia, Elena Lazkano, and Ander Ansuategi. Ac-
tive mapping and robot exploration: A survey. Sen-
sors, 21(7):2445, 2021. 3
[45] Yuxing Long, Wenzhe Cai, Hongcheng Wang, Guanqi
Zhan, and Hao Dong. Instructnav: Zero-shot system
for generic instruction navigation in unexplored envi-
ronment. CoRL, 2024. 3
10

<!-- page 11 -->
[46] Yuxing Long, Xiaoqi Li, Wenzhe Cai, and Hao Dong.
Discuss before moving:
Visual language navigation
via multi-expert discussions.
In 2024 IEEE In-
ternational Conference on Robotics and Automation
(ICRA), pages 17380–17387. IEEE, 2024. 3
[47] David J. C. MacKay. Bayesian Interpolation. Neural
Computation, 4(3):415–447, 1992. 4, 5
[48] James Martens. New insights and perspectives on the
natural gradient method. Journal of Machine Learning
Research, 21(146):1–76, 2020. 5
[49] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and
Andrew J. Davison. Gaussian Splatting SLAM. In Pro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024. 1, 3, 5, 7, 14
[50] Reihaneh Mirjalili, Michael Krawez, and Wolfram Bur-
gard. Fm-loc: Using foundation models for improved
vision-based localization. In 2023 IEEE/RSJ Interna-
tional Conference on Intelligent Robots and Systems
(IROS), pages 1381–1387. IEEE, 2023. 3
[51] David Morilla-Cabello, Lorenzo Mur-Labadia, Ruben
Martinez-Cantin, and Eduardo Montijano.
Robust
fusion for bayesian semantic mapping.
In 2023
IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pages 76–81. IEEE, 2023.
2
[52] Raul Mur-Artal and Juan D Tardós. Orb-slam2: An
open-source slam system for monocular, stereo, and
rgb-d cameras. IEEE transactions on robotics, 33(5):
1255–1262, 2017. 2
[53] Raul Mur-Artal, Jose Maria Martinez Montiel, and
Juan D Tardos.
Orb-slam:
a versatile and accu-
rate monocular slam system.
IEEE transactions on
robotics, 31(5):1147–1163, 2015.
[54] Richard A Newcombe, Steven J Lovegrove, and An-
drew J Davison. Dtam: Dense tracking and mapping
in real-time. In 2011 international conference on com-
puter vision, pages 2320–2327. IEEE, 2011. 2
[55] OpenAI.
Gpt-4o
system
card,
2024.
https://openai.com/index/gpt-4o-system-card/.
4
[56] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang.
Activenerf: Learning where to see with uncertainty es-
timation. In ECCV, pages 230–246. Springer, 2022. 2,
3
[57] Andrej Pázman. Foundations of optimum experimental
design. Springer Dordrecht, 1986. 3
[58] Julio A Placed, Juan J Gómez Rodríguez, Juan D
Tardós, and José A Castellanos. Explorb-slam: Ac-
tive visual slam exploiting the pose-graph topology. In
Iberian Robotics conference, pages 199–210. Springer,
2022. 2, 6, 7, 16
[59] Julio A Placed, Jared Strader, Henry Carrillo, Nikolay
Atanasov, Vadim Indelman, Luca Carlone, and José A
Castellanos. A survey on active simultaneous localiza-
tion and mapping: State of the art and new frontiers.
IEEE Transactions on Robotics, 2023. 3
[60] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian
Wang, and Hanspeter Pfister. Langsplat: 3d language
gaussian splatting. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recogni-
tion, pages 20051–20060, 2024. 1, 8
[61] Santhosh Kumar Ramakrishnan,
Aaron Gokaslan,
Erik
Wijmans,
Oleksandr
Maksymets,
Alexander
Clegg, John M Turner, Eric Undersander, Wojciech
Galuba, Andrew Westbury, Angel X Chang, Mano-
lis Savva, Yili Zhao, and Dhruv Batra.
Habitat-
matterport 3d dataset (HM3d): 1000 large-scale 3d
environments for embodied AI.
In Thirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track, 2021. 2, 6, 8
[62] Krishan Rana, Jesse Haviland, Sourav Garg, Jad
Abou-Chakra, Ian D Reid, and Niko Suenderhauf. Say-
plan: Grounding large language models using 3d scene
graphs for scalable task planning. CoRR, 2023. 3
[63] María L Rodríguez-Arévalo, José Neira, and José A
Castellanos.
On the importance of uncertainty rep-
resentation in active slam.
IEEE Transactions on
Robotics, 34(3):829–834, 2018. 1
[64] Julius Rückin, Federico Magistri, Cyrill Stachniss, and
Marija Popović. An informative path planning frame-
work for active learning in uav-based semantic map-
ping.
IEEE Transactions on Robotics, 39(6):4279–
4296, 2023. 2
[65] Arjun P S, Andrew Melnik, and Gora Chand Nandi.
Cognitive planning for object goal navigation using
generative AI models. In NeurIPS 2024 Workshop on
Open-World Agents, 2024. 3
[66] M.J. Schervish.
Theory of Statistics.
Springer New
York, 2012. 3
[67] Pierre Sermanet,
Tianli Ding,
Jeffrey Zhao,
Fei
Xia,
Debidatta Dwibedi,
Keerthana Gopalakrish-
nan, Christine Chan, Gabriel Dulac-Arnold, Sharath
Maddineni, Nikhil J Joshi, et al.
Robovqa: Multi-
modal long-horizon reasoning for robotics.
In 2024
IEEE International Conference on Robotics and Au-
tomation (ICRA), pages 645–652. IEEE, 2024. 2
[68] Nur Muhammad Mahi Shafiullah, Chris Paxton, Lerrel
Pinto, Soumith Chintala, and Arthur Szlam. CLIP-
fields: Weakly supervised semantic fields for robotic
memory. In ICRA2023 Workshop on Pretraining for
Robotics (PT4R), 2023. 1
[69] Dhruv Shah, Michael Robert Equi, Błażej Osiński,
Fei Xia, Brian Ichter, and Sergey Levine. Navigation
with large language models: Semantic guesswork as a
heuristic for planning. In Conference on Robot Learn-
ing, pages 2683–2699. PMLR, 2023. 3
[70] Claude Elwood Shannon. A mathematical theory of
communication. The Bell system technical journal, 27
(3):379–423, 1948. 3
[71] Jianxiong Shen, Adria Ruiz, Antonio Agudo, and
Francesc Moreno-Noguer. Stochastic neural radiance
fields: Quantifying uncertainty in implicit 3d represen-
tations. CoRR, abs/2109.02123, 2021. 2, 3
[72] Jianxiong Shen, Antonio Agudo, Francesc Moreno-
Noguer, and Adria Ruiz. Conditional-flow nerf: Ac-
11

<!-- page 12 -->
curate 3d modelling with reliable uncertainty quantifi-
cation. In ECCV, 2022. 2, 3
[73] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-
Hua Guan. Language embedded 3d gaussians for open-
vocabulary scene understanding. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 5333–5343, 2024. 1, 8
[74] Chan Hee Song, Jiaman Wu, Clayton Washington,
Brian M Sadler, Wei-Lun Chao, and Yu Su.
Llm-
planner:
Few-shot grounded planning for embodied
agents with large language models. In Proceedings of
the IEEE/CVF International Conference on Computer
Vision, pages 2998–3009, 2023. 3
[75] Cyrill Stachniss, Dirk Hahnel, and Wolfram Burgard.
Exploration with active loop-closing for fastslam. In
2004 IEEE/RSJ International Conference on Intel-
ligent Robots and Systems (IROS)(IEEE Cat. No.
04CH37566), pages 1505–1510. IEEE, 2004. 2
[76] Zezhou Sun, Banghe Wu, Cheng-Zhong Xu, Sanjay E
Sarma, Jian Yang, and Hui Kong. Frontier detection
and reachability analysis for efficient 2d graph-slam
based active exploration. In 2020 IEEE/RSJ Interna-
tional Conference on Intelligent Robots and Systems
(IROS), pages 2051–2058. IEEE, 2020. 2
[77] Niko Sünderhauf,
Jad Abou-Chakra,
and Dimity
Miller.
Density-aware nerf ensembles:
Quantifying
predictive uncertainty in neural radiance fields.
In
2023 IEEE International Conference on Robotics and
Automation (ICRA), pages 9370–9376. IEEE, 2023. 2,
3
[78] Andrew Szot, Alexander Clegg, Eric Undersander,
Erik Wijmans, Yili Zhao, John Turner, Noah Maestre,
Mustafa Mukadam, Devendra Singh Chaplot, Olek-
sandr Maksymets, et al. Habitat 2.0: Training home
assistants to rearrange their habitat. Advances in neu-
ral information processing systems, 34:251–266, 2021.
6
[79] Yuezhan Tao, Xu Liu, Igor Spasojevic, Saurav Agar-
wal, and Vijay Kumar.
3d active metric-semantic
slam.
IEEE Robotics and Automation Letters, 9(3):
2989–2996, 2024. 2
[80] Yuezhan Tao, Dexter Ong, Varun Murali, Igor Spaso-
jevic, Pratik Chaudhari, and Vijay Kumar. Rt-guide:
Real-time gaussian splatting for information-driven ex-
ploration. arXiv preprint arXiv:2409.18122, 2024. 3
[81] Octo Model Team, Dibya Ghosh, Homer Walke, Karl
Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey
Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An
open-source generalist robot policy.
arXiv preprint
arXiv:2405.12213, 2024. 2
[82] Zachary Teed and Jia Deng. Droid-slam: Deep visual
slam for monocular, stereo, and rgb-d cameras. Ad-
vances in neural information processing systems, 34:
16558–16569, 2021. 2
[83] Eleftherios Triantafyllidis, Filippos Christianos, and
Zhibin Li.
Intrinsic language-guided exploration for
complex long-horizon robotic manipulation tasks. In
2024 IEEE International Conference on Robotics and
Automation (ICRA), pages 7493–7500. IEEE, 2024. 3
[84] Darko Trivun, Edin Šalaka, Dinko Osmanković, Jas-
min Velagić, and Nedim Osmić.
Active slam-based
algorithm for autonomous exploration with mobile
robot. In 2015 IEEE International Conference on In-
dustrial Technology (ICIT), pages 74–79. IEEE, 2015.
2
[85] Teresa
Vidal-Calleja,
Andrew
J
Davison,
Juan
Andrade-Cetto, and David William Murray.
Active
control for single camera slam.
In Proceedings 2006
IEEE International Conference on Robotics and Au-
tomation, 2006. ICRA 2006., pages 1930–1936. IEEE,
2006. 1
[86] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and
Eero P Simoncelli.
Image quality assessment: from
error visibility to structural similarity. IEEE transac-
tions on image processing, 13(4):600–612, 2004. 6
[87] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. Chain-of-thought prompting elicits reasoning in
large language models. Advances in neural information
processing systems, 35:24824–24837, 2022. 4
[88] Fei Xia, Amir R. Zamir, Zhi-Yang He, Alexander Sax,
Jitendra Malik, and Silvio Savarese. Gibson Env: real-
world perception for embodied agents. In Computer
Vision and Pattern Recognition (CVPR), 2018 IEEE
Conference on. IEEE, 2018. 2, 6, 7, 8
[89] Brian Yamauchi.
A frontier-based approach for au-
tonomous exploration. In Proceedings 1997 IEEE In-
ternational Symposium on Computational Intelligence
in Robotics and Automation CIRA’97.’Towards New
Computational Principles for Robotics and Automa-
tion’, pages 146–151. IEEE, 1997. 1, 2, 6, 7, 16, 17
[90] Chi Yan, Delin Qu, Dong Wang, Dan Xu, Zhigang
Wang, Bin Zhao, and Xuelong Li.
Gs-slam: Dense
visual slam with 3d gaussian splatting. arXiv preprint
arXiv:2311.11700, 2023. 1
[91] Dongyu Yan, Jianheng Liu, Fengyu Quan, Haoyao
Chen, and Mengmeng Fu. Active implicit object re-
construction using uncertainty-guided next-best-view
optimization. IEEE Robotics and Automation Letters,
2023. 2
[92] Zike Yan, Haoxiang Yang, and Hongbin Zha. Active
neural mapping. In ICCV, 2023. 2, 3, 6, 7, 14, 16
[93] Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang
Wang, and Bernadette Bucher. Vlfm: Vision-language
frontier maps for zero-shot semantic navigation.
In
2024 IEEE International Conference on Robotics and
Automation (ICRA), pages 42–48. IEEE, 2024. 3
[94] Justin Yu, Kush Hari, Kishore Srinivas, Karim El-
Refai, Adam Rashid, Chung Min Kim, Justin Kerr,
Richard Cheng, Muhammad Zubair Irshad, Ashwin
Balakrishna, et al.
Language-embedded gaussian
splats (legs): Incrementally building room-scale rep-
resentations with a mobile robot. In IROS, 2024. 1
[95] Huangying Zhan, Jiyang Zheng, Yi Xu, Ian Reid, and
12

<!-- page 13 -->
Hamid Rezatofighi.
Activermap: Radiance field for
active mapping and planning, 2022. 2, 3
[96] Richard Zhang, Phillip Isola, Alexei A Efros, Eli
Shechtman, and Oliver Wang. The unreasonable ef-
fectiveness of deep features as a perceptual metric. In
CVPR, pages 586–595, 2018. 6
[97] Gengze Zhou, Yicong Hong, and Qi Wu.
Navgpt:
Explicit reasoning in vision-and-language navigation
with large language models. Proceedings of the AAAI
Conference on Artificial Intelligence, 38(7):7641–7649,
2024. 3
[98] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen
Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya
You, Zhangyang Wang, and Achuta Kadambi.
Fea-
ture 3dgs:
Supercharging 3d gaussian splatting to
enable distilled feature fields.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 21676–21685, 2024. 1, 8
13

<!-- page 14 -->
6. Appendix
In this supplementary material, we discuss things we
left over in our main paper due to page constraints. We
provide more background information about 3DGS and
the derivative with respect to localization parameters
in Sec. 6.1. We discussed more implementation details
in Sec. 6.2 and provided the prompt we use for the
Multimodal LLM and an example of the interaction in
Sec. 6.5 to help reproduce our results. The source
code of this project will be made public soon.
We also include detailed qualitative results in Sec. 6.4.
6.1. Additional Background Information
For the completeness of our method, we also pro-
vide the key definition for the 3D Gaussian Splatting
backbone [36] and 3D Gaussian SLAM [49].
In 3D
Gaussian Splatting [36], the rendered pixel color is cal-
culated by composing all 3D Gaussians projected in a
tile.
ˆC(r) =
Ns
X
i=1
Ti (1 −exp(−σiδi)) ci
(14)
αi = exp

−
i−1
X
j=1
σjδj

(1 −exp(−σiδi))
(15)
δi = ti+1 −ti represents the distance between adja-
cent samples, and Ns indicates the number of samples.
ci is the color of each 3D Gaussian given the current
view direction d and σi is given by evaluating a 2D
Gaussian with covariance Σ.
The Jacobian of the localization parameters are de-
fined as:
DµC
Dx =

I
−µ×
C

and DW
Dx =


0
−W×
:,1
0
−W×
:,2
0
−W×
:,3

, (16)
where × denotes the skew symmetric matrix of a 3D
vector, and W:,i refers to the ith column of the matrix.
Unlike PSNR, the EIG can be computed without
ground truth images, making it possible to perform
view selection during exploration.
6.2. Implementation Details
The 2D occupancy map’s resolution is 5cm.
For
each single frontier pixel on the 2D map, we add 200
3D Gaussians, which are uniformly distributed in the
3D cube above each frontier pixel. Other parameters
like color, opacity, and scale are generated uniformly
between 0 and 1. When there are frontiers on the 2D
map, we choose the next frontier by querying LLM as
stated in the main paper. When no frontier exists, we
select the top 20% of Gaussians with the highest score.
These Gaussians are grouped using DBSCAN [20]. The
largest cluster is selected for candidate pose genera-
tion. Candidates are uniformly sampled in the range
between 0.3m to 1m, facing towards the selected posi-
tion. Only the poses in free space are kept for path-level
selection. The importance factor η in Eq. 13 is set to
5 across all experiments.
We compute the Expected Information Gain (EIG)
for each global candidate and use A* to plan a path
to each of them. In order to prevent a twisted path,
we consider locations 0.15m (3 pixels) away from the
current robot position as neighbors and set the robot
width to 3 pixels for collision check. However, the path
planned by A* might have redundant waypoints, caus-
ing unnecessary turns for the robot.
Therefore, we
smooth the path by finding shortcuts. Specifically, for
each waypoint wi, if the path between waypoint wi+2
and wi is collision-free, then we remove the interme-
diate waypoint wi+1 from the path.
Finally, we use
a greedy follower for motion planning.
If the angle
between the heading direction of the robot and the rel-
ative next waypoint is larger than 5°, then we turn left
or right to decrease the angle. Otherwise, we choose
the forward action to approach the next waypoint. In
such a way, we get a sequence of actions {ai}T
i=1 for
each path.
Given a sequence of actions {ai}T
i=1 for each path,
we use forward dynamics to compute the future cam-
era poses {ci}T
i=1.
Initially, we use an intermediate
variable H′′
obs ≜H′′[w∗] to help compute expected in-
formation gain along the path. For each camera pose
xi, we compute its pose Hessian H′′
pose and the current
model Hessian matrix H′′
cur ≜H′′[y|xi, w∗]. H′′
cur is
then accumulated, and we update H′′
obs to evaluate the
remaining poses on the path. We select the path that
minimizes the objective given by Eq. 13 for execution.
6.3. Scenes Used for Evaluation
Following
previous
literature
[92],
we
use
the
following
scenes
for
Gibson
Dataset:
Greigsville, Denmark, Cantwell, Eudora,
Pablo, Ribera, Swormville, Eastville,
Elmira.
For HM3D we use the following scenes:
DBjEcHFg4oq, mscxX4KEBcB, QKGMrurUVbk,
oPj9qMxrDEa, CETmJJqkhcK.
6.4. Detailed Version of Qualitative Results
We provide larger versions of the qualitative render-
ing comparisons from the main paper, Fig. 8 shows the
Gibson scenes and Fig. 9 shows the HM3D scenes.
We also present qualitative comparisons on testing
views from the Gibson dataset in Fig. 10 and HM3D
14

<!-- page 15 -->
in Fig. 11.
6.5. Example of Using Multimodal LLM
We provide an example of our interaction with the
multimodal LLM in Fig. 12 and additionally provide
our full text prompt to the LLM below.
1
2
<system> You are an AI assistant that can
analyze images and plan a long-term goal
for the exploration task of a ground
robot.
3
You will be given a bird-eye view image of a
scene.
4
The goal is to plan a long-term exploration
mission for a robot to traverse the area.
5
The robot's task is to explore the terrain
efficiently, identifying important
areas, potential obstacles, and
unvisited areas.
6
Please analyze the image and select a
long-term goal from the candidates for
the robot to explore the area.
7
Empty space doesn't always mean they are
unvisited regions, sometimes it's just
outside the floor plan of this scene.
8
We are allowed to explore a total of
<TOTAL\_STEPS> steps and this is step
<STEP\_ID>.
9
Therefore, it's better to select a space that
is close to the visited regions but
still unvisited and not behind the walls.
10
The current location of the robot is marked
with the blue star(*) marker.
11
The last frontier you selected is marked with
a yellow diamond shape.
12
The visited path is painted as green lines in
the image.
13
Note that you don't have to select the
closest point to the robot, but the
point that is most likely to be
unvisited and important to explore.
14
As you can see, there are <NUM\_FRAME>
candidate points to select from.
15
They are numbered from 0 to <NUM\_FRAME - 1>
in red color.
16
If you find all the goals are not necessary
to explore and we should instead focus
on improving existing reconstruction,
please give -1 in the `target` entry of
the JSON.
17
Please provide a detailed exploration plan
and select an exploration target with
reasons in the JSON format as shown
below.
18
19
20
```
21
{
22
"target": 2, "reason": "The target is located
at an unvisited region of the image and
seems to be an unvisited bedroom"
23
}
24
```
25
Do not cut off the JSON and generate the full
JSON.
26
</system>
27
28
29
<user>: I have a bird-eye view image of a
scene. The goal is to plan a long-term
exploration mission for a robot to
traverse the area. Please analyze the
attached image and provide the
exploration plan first and then an
exploration target in the specified JSON
format.
30
</user>
15

<!-- page 16 -->
ANS [9]
Active-INR [92]
UPEN [23]
ExplORB [58]
FBE [89]
Ours
ANS [9]
Active-INR [92]
UPEN [23]
ExplORB [58]
FBE [89]
Ours
Figure 8.
Qualitative Comparison for Final Scene Reconstruction on Gibson Dataset Greigsville (top) and
Ribera (bottom) scenes. We provide top-down rendering for different methods. Note that UPEN and Active-INR use GT
pose in this visualization.
16

<!-- page 17 -->
UPEN [23]
FBE [89]
Ours
Figure 9.
Qualitative Comparison for Final Scene Reconstruction on Habitat-Matterport 3D Dataset
mscxX4KEBcB (top), oPj9qMxrDEa (middle) and QKGMrurUVbk (bottom) scenes. We provide top-down rendering for
different methods.
17

<!-- page 18 -->
ANS
Active-INR (gt)
UPEN
ExplORB
FBE
Ours
Figure 10. Test Rendering Qualitative Visualization on Gibson Dataset All the renderings are from the test view
of the Gibson dataset.
18

<!-- page 19 -->
UPEN
FBE
Ours
Figure 11. Test Rendering Qualitative Visualization on Habitat-Matterport 3D Dataset All the renderings are
from the test view of the Habitat-Matterport 3D dataset.
19

<!-- page 20 -->
System: You are an AI assistant that can analyze images and plan a long-term 
goal for the exploration task of a ground robot.
You will be given a bird-eye view image of a scene.
The goal is to plan a long-term exploration mission for a robot to traverse 
the area.
The robot's task is to explore the terrain efficiently, identifying important 
areas, potential obstacles, and unvisited areas.
Please analyze the image and select a long-term goal from the candidates for 
the robot to explore the area.
Empty space doesn't always mean they are unvisited regions, sometimes it's 
just outside the floor plan of this scene.
We are allowed to explore a total of 2000 steps and this is step 2.
Therefore, it's better to select a space that is close to the visited regions 
but still unvisited and not behind the walls.
The current location of the robot is marked with the blue star(*) marker.
The last frontier you selected is marked with a yellow diamond(ç) shape. 
The visited path is painted as green lines in the image.
Note that you don't have to select the closest point to the robot, but the 
point that is most likely to be unvisited and important to explore.
As you can see, there are 4 candidate points to select from.
They are numbered from 0 to 3 in red color.
If you find all the goals are not necessary to explore and we should instead 
focus on improving existing reconstruction, please give -1 in the `target` 
entry of the JSON.
Please provide a detailed exploration plan and select an exploration target 
with reasons in the JSON format as shown below.
```
{
"target": 2, "reason": "The target is located at an unvisited region of the 
image and seems to be an unvisited bedroom"
}
```
Do not cut off the JSON and generate the full JSON.
User: I have a bird-eye view image of a scene. The goal is to plan a long-
term exploration mission for a robot to traverse the area. Please analyze the 
attached image and provide the exploration plan first and then an exploration 
target in the specified JSON format.
**Exploration Plan:**
1. **Current Situation Analysis:**
   - The robot is currently located at the blue star marker.
   - The previously selected frontier, marked by a yellow diamond, might still have unexplored neighboring areas.
   - So far, the robot has traversed 132 steps and can take a total of 2000 steps.
2. **Observation of Candidates:**
   - **Candidate 0:** Located towards the top of the scene. It appears to be in a separated area from the main path by walls and could be 
challenging to access directly.
   - **Candidate 1:** Positioned near the robot's previous path, to the left. The space seems to be partly blocked but may lead to a new room 
or area.
   - **Candidate 2:** Situated to the bottom right, close to the previously marked yellow diamond. This area seems unvisited with potential 
space for exploration.
   - **Candidate 3:** On the right side, aligned with the robot’s current heading, showing open space but appearing more distant from the 
recent paths.
3. **Selection Reasoning:**
   - **Candidate 2** is favored for exploration as it is likely unvisited, near the previous frontier, and potentially opens up new unexplored 
areas. It is a strategic choice for efficient area coverage without straying too far from the recently explored paths, reducing the risk of 
re-exploring already visited spaces.
```json
{
  "target": 2,
  "reason": "Candidate 2 is favored as it is near the previously marked frontier, indicating potential unexplored area relevant to the mission 
objectives. Continuing exploration here aligns with efficient coverage of the space."
}
```
�㻑
Figure 12. An Example about a Dialog with the Multi-modal LLM. The chat box on the top is the question from
our method, and the bottom is the response from GPT-4o, which we parsed to get a long-term goal for our path proposal.
We provided the contextual information in both the textual prompt and the image.
20
