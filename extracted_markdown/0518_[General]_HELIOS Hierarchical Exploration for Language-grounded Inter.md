<!-- page 1 -->
HELIOS: HIERARCHICAL EXPLORATION FOR LANGUAGE-
GROUNDED INTERACTION IN OPEN SCENES
Katrina Ashton1
Chahyon Ku2
Shrey Shah2
Saumit Vedula2
Tingrui Zhang2
Wen Jiang1
Kostas Daniilidis1
Bernadette Bucher2
1 University of Pennsylvania
2 University of Michigan
ABSTRACT
Language-specified mobile manipulation tasks in novel environments simultane-
ously face challenges interacting with a scene which is only partially observed,
grounding semantic information from language instructions to the partially ob-
served scene, and actively updating knowledge of the scene with new observa-
tions. To address these challenges, we propose HELIOS, a hierarchical scene
representation and associated search objective. We construct 2D maps containing
the relevant semantic and occupancy information for navigation while simulta-
neously actively constructing 3D Gaussian representations of task-relevant ob-
jects. We fuse observations across this multi-layered representation while explic-
itly modeling the multi-view consistency of the detections of each object using
the Dirichlet distribution. Planning is formulated as a search problem over our
hierarchical representation. We formulate an objective that jointly considers (i)
exploration of unobserved or uncertain regions of the environment and (ii) in-
formation gathering from additional observations of candidate objects. This ob-
jective integrates frontier-based exploration with the expected information gain
associated with improving semantic consistency of object detections. We evalu-
ate HELIOS on the OVMM benchmark in the Habitat simulator, a pick and place
benchmark in which perception is challenging due to large and complex scenes
with comparatively small target objects. HELIOS achieves state-of-the-art results
on OVMM. We demonstrate HELIOS performing language specified pick and
place in a real world office environment on a Spot robot. Our method leverages
pretrained VLMs to achieve these results in simulation and the real world without
any task specific training. Videos and code are available at our project website:
https://helios-robot-perception.github.io/
1
INTRODUCTION
Consider an autonomous robot tasked with bringing a mug from a coffee table to the kitchen counter
in a home. If that robot sees a coffee table but cannot currently detect a mug on it, should it go
closer to investigate if the mug is actually present? Or should it look in new parts of the home?
An autonomous robot should be able to efficiently reason through this question using environment
cues and the observations it accumulates during this search process. In order to perform mobile
manipulation which includes object search, this reasoning must occur simultaneously in both long
and short horizons. Low success rates on new benchmarks targeting language-specified mobile pick
and place tasks in novel environments have demonstrated that combining this long and short horizon
reasoning is still an open challenge [57].
Reasoning jointly over short and long spatio-temporal contexts requires very different policy ob-
jectives and scene representations. In general, object search methods explicitly manage local and
global search problems distinctly [29, 45, 63]. Search policies must then figure out when to switch
between local and global reasoning by deciding the likelihood of being close to the target object.
This is not always straightforward, and incorrect object detections have been identified as a ma-
jor cause of failure for this task [31, 35, 57]. While improvements in object detection can help
1
arXiv:2509.22498v2  [cs.RO]  26 Mar 2026

<!-- page 2 -->
to mitigate these issues, there are challenges which occur in robotics data which are much less
prevalent in other settings, such as unusual viewing angles and obstructed views. This aspect of
the exploration-exploitation tradeoff has been overlooked in recent works on mobile pick and place
tasks [31, 35, 57].
RGB 
Representation
Semantic 
Representation
Semantic 
Uncertainty
Occupancy 
map
Layered 
value 
map
Value maps 
scale
Uncertainty 
scale
Figure 1: Our hierarchical scene representa-
tion.
To address these challenges we propose HE-
LIOS, a framework for hierarchical scene rep-
resentation and decision-making for language-
specified mobile manipulation in novel en-
vironments.
Our key insight is that object
search benefits from task-driven representa-
tions that explicitly separate global exploration
from object-level reasoning.
We use aligned
coarse 2D grid maps encoding occupancy and
semantic likelihoods to support frontier-based
exploration and navigation globally while, lo-
cally, at the object level, we maintain a sparse
set of object-centric 3D Gaussians correspond-
ing to candidate task-relevant objects detected
during exploration. Each object instance is rep-
resented as a collection of 3D Gaussians whose
semantic class distribution is updated across
observations, thus facilitating an explicit en-
forcement of multi-view consistency and mod-
eling of the uncertainty in object identity. In
contrast to prior work we do not construct dense Gaussian representations of the entire scene [48,
51, 61, 65] but rather maintain a sparse object-centric representation only in regions of the scene
that are relevant to the task. Instead of accumulating high-dimensional vision-language embeddings
for each Gaussian, we directly maintain class distributions obtained from an open-vocabulary de-
tector, resulting in a semantic representation suitable for resource-constrained robotic systems. We
formulate a search objective defined over this hierarchical representation that explicitly trades off
exploration of unobserved regions with information gathering about candidate objects. This objec-
tive evaluates whether additional observations are expected to reduce uncertainty in object identity
or whether the robot should instead expand its search over unexplored regions of the environment.
Contributions.
• We present a hierarchical scene representation that enables reasoning across different spa-
tial resolutions and decision horizons. This is achieved by a coarse 2D occupancy and
semantic map that supports long-range exploration together with an object-centric 3D rep-
resentation enabling manipulation and semantic verification.
• We model objects via their 3D Gaussian instances and with their semantic probabilities
updated across views using a Dirichlet-based Bayesian update. This way, we equip the
robot with an explicit model of uncertainty and multi-view consistency of open-vocabulary
detections.
• We formulate a global search objective that combines semantic likelihood, predicted in-
formation gain, and navigation cost to decide whether to (i) explore unseen regions of the
environment or (ii) obtain additional views of candidate objects to disambiguate their iden-
tity.
• We verify that each of these components increases our method’s performance via an abla-
tion study on the HomeRobot Open-vocabulary Mobile Manipulation benchmark [56, 57].
We demonstrate HELIOS performing this task in a real world office environment on a Spot
robot, because our method does not involve any task-specific training we can do this with-
out needing to obtain and train on additional data.
2

<!-- page 3 -->
2
RELATED WORK
Language-grounded open world pick and place. Recent advancements in vision and language
have opened up challenges in natural language instruction following for robots in novel environ-
ments. Many methods focus on parsing complex or ambiguous language and accurately grounding
this language to observations made during task execution [4, 16, 41, 46, 58]. Others focus on im-
proving execution of language specified pick and place skills [46, 52, 53]. However, benchmarks
for targeted instantiations of this problem have identified that a major cause of failure in this task
is correctly finding and identifying objects for performing pick and place [31, 35, 57]. Our work
addresses this challenge by modeling the multi-view consistency of object detections, allowing us to
only interact with objects once we have obtained enough views that we are confident in the results
of the object detection.
Object search and detection. To find an object with an RGB camera, that camera needs to record
sufficient observations in the environment to correctly identify the object. Active object detec-
tion methods obtain additional views of a scene in order to capture an image from which a target
object can be correctly identified [2, 9, 15]. When these observations are accumulated in a map
of the environment, it enables a larger scale search problem in which the camera is systemati-
cally moved to possible locations in the map. Hierarchical object search methods explicitly per-
form global and local object search to ensure sensor coverage of the scene [29, 45, 63]. To per-
form object search efficiently, semantic information can be used as a prior about where objects are
more likely to be [1]. This semantic prior naturally yields an exploration and exploitation trade-
off [8, 12, 40, 54, 59, 60, 62]. In our work, we perform object search and detection as part of pick
and place mobile manipulation tasks. Therefore, we construct an objective for switching between
global object search and local object detection while simultaneously trading off exploration of the
scene and exploitation of semantic information.
3D Gaussians in robot perception. 3D Gaussians [24] have been used in a variety of robotics tasks
including SLAM [23, 34], active mapping [20, 21, 22], and table-top manipulation [32, 64]. These
methods all build a dense 3D representation of the entire scene. Many methods incorporate open-
vocabulary semantic features in 3D Gaussian representations [5, 48, 61, 65]. In contrast to previous
robot perception approaches, we only model target objects of interest with 3D Gaussians, building a
sparse 3D map which requires significantly less momeory than a full scene representation. We adapt
Wilson et al. [51] to perform semantic classification and estimate the associated uncertainty in our
sparse 3D Gaussian object map, which forms one layer of our scene representation.
Language-grounded scene representations.
Language-grounded scene representations can be
dense or sparse. Dense open-vocabulary 3D scene representations map vision-language features
which can be dynamically queried with language [19, 25, 27, 36, 42, 48]. However, these dense 3D
representations are not necessarily effective or efficient for performing planning and control. For se-
mantic navigation tasks, dense 2D language-grounded scene representations are more efficient and
have been shown to be effective [12, 13, 17, 59]. For language specified manipulation tasks, instance
level information about objects is important [38, 39, 47, 67]. To enable mobile manipulation, 3D
scene graphs build globally consistent maps of object centric representations needed for manipu-
lation [6, 14, 16, 18, 33, 43]. Our work builds on this direction in mobile manipulation by using
object instance information to construct a sparse map of 3D Gaussians. In our work, we combine
this information for manipulation in a hierarchical map with 2D value maps for semantic navigation.
3
METHOD
We address the problem of language specified pick and place mobile manipulation tasks in novel
environments. To carry out this task, the robot first needs to solve a search problem to find the target
object, including correctly identifying the target object. It must then navigate to a suitable grasp
position and grasp the object. Finally, it needs to solve another search problem in order to find the
place location, and then place the object there in a stable orientation. Note that all of these stages
need to be successful, and the robot must avoid collisions with the environment when navigating
and interacting with the objects, so this task is subject to compounding error rates. However the
robot can also use information collected in previous stages of the task to aid it later. For example,
the search to find the place location can be made more efficient by utilizing information collected
3

<!-- page 4 -->
when the robot was searching for the target object. In order to collate this information into a useful
and efficient format, we propose constructing a hierarchical task-driven map (see Section 3.1) with
2D map layers suitable for the search problems and 3D Gaussians to represent objects in the scene
relevant to manipulation. We detail how we explicitly reason over this map to solve a language
specified pick and place task in Section 3.2.
3.1
HIERARCHICAL TASK-DRIVEN MAP
We construct a hierarchical map with three layers, where each layer corresponds to the three primary
tasks that the robot needs to complete. First, to navigate around obstacles to a specified goal location,
the robot requires an occupancy map to perform collision free path planning. Second, to efficiently
search for objects, the robot can use semantic information in the environment to prioritize exploring
unobserved regions which are similar to target locations. Finally, in order to effectively manipulate
and perform robust detection of the objects of interest, we model the components of the scene where
we expect to perform pick and place with a sparse 3D representation using 3D Gaussians assigned
to instances of classes referenced in the instruction. Representing only the objects of interest using
3D Gaussians as opposed to the entire scene, as is done in prior works which use 3D Gaussians for
robotics, significantly improves the efficiency of our scene representation as shown in Section 4.3.
2D Occupancy Maps. We construct a 2D bird’s-eye view (BEV) occupancy map by ground pro-
jecting depth measurements. We use this map to perform collision-free path planning to navigate
around obstacles to goal locations. We also identify frontiers on the occupancy map, defined as
center-points of boundaries between explored and unexplored areas, which will enable us to search
unknown map regions.
2D Semantic Value Map. To choose between frontier points, we leverage semantic information
about the scene in order to search efficiently by going to areas more likely to contain the target
of interest first. We construct a layered semantic value map to enable this frontier-based approach
by extending prior work constructing semantic value maps [59] to incorporate multiple search tar-
gets. Each layer in our map is a 2D BEV value map constructed by using BLIP-2 [28] to score the
similarity of each observed RGB image to the prompt Seems like there is a (object)
ahead and fusing the results using a confidence based on the field-of-view cone for each observa-
tion. We construct one map layer for the pick location and one for the place location.
3D Gaussian representation for modeling objects. In order to enable reasoning about the multi-
view consistency of semantic classifications, we represent the objects of interest in the scene using
3D Gaussian Splatting (3DGS) [24]. To increase efficiency over prior applications of 3DGS to
robotic tasks [32, 64], instead of modeling the entire scene with 3D Gaussians we only use them to
model parts of the scene which have been detected as objects of interest. We assign Gaussians to
instances, allowing us to reason over objects in the scene instead of individual Gaussians. Our sparse
3DGS representation supports tracking the semantic class probability and semantic class uncertainty
for each Gaussian which we use to create an uncertainty-weighted object score for each instance.
Preliminaries – 3D Gaussian representation rendering. A 3D Gaussian x(µ, Σ; c, α) is defined
by its mean position µ, covariance Σ, color c and opacity α, these characteristics can be learned via
a rendering loss. A scene is rendered with many of these 3D Gaussians, the final number determined
by the task specific conditions in which Gaussians are added and removed. When an image is
rendered using 3DGS, the 3D Gaussians comprising the scene representation are first transformed
from the world frame to the camera frame and then projected into 2D Gaussians (splats) in the
image plane, x(µ, Σ; c, α) 7→˜x(˜µ, ˜Σ; c, α). Each pixel i’s color Qi is then calculated from the 2D
Gaussians using α-blending for the N ordered points on the 2D splats that overlap the pixel. For a
pixel with position pi and a 3D Gaussian xn, we first find the opacity ˜αn(pi) of the corresponding
2D Gaussian at that pixel position by weighting based on the pixel’s distance to the center of the
2D Gaussian with ˜αn(pi) = αn · k(pi, ˜xn), where k(pi, ˜xn) = exp

1
2(pi −˜µn)˜Σ−1
n (pi −˜µn)

.
Next, the N Gaussians are ordered based on depth, with ˜x1 being the closest to the camera, and
the final contribution for each Gaussian is calculated with α-blending to get the final pixel color
4

<!-- page 5 -->
Inconsistent
Consistent
Detection #1
Detection #2
Change in semantic 
probability for “knife”
Change in semantic 
probability for “table”
Positive 
change
Negative 
change
Figure 2: Example of multi-view fusion. We show two observations, in the first a toy rocket is
incorrectly identified as a knife and the table is correctly identified, in the second the table is again
correctly identified. Right of this we show the change in the semantic probability for each class in
the 3DGS part of our scene representation when it is updated with the second detection. We can see
that the incorrect detection of the object on the table as a knife is not multi-view consistent and so
the probability of this object being a knife goes down when we include the second detection. The
table is correctly detected across multiple frames so the probability goes up after fusion.
Qi = PN
n=1 cnκ(pi, ˜xn; {˜xj}j∈{1,...,N}) where
κ(pi, ˜xn; {˜xj}j∈{1,...,N}) := ˜αn(pi)
n−1
Y
j=1
(1 −˜αj(pi)).
(1)
Preliminaries – Semantic classes for 3D Gaussian representation. We represent the semantic
class scores with our 3DGS model in addition to color. Following Wilson et al. [51], we explicitly
model the distribution of semantic estimates of each Gaussian using the categorical distribution.
This distribution is then updated using its conjugate prior, the Dirichlet distribution. Note that this
method requires specifying number of object classes at the start of the episode. However, any amount
of classes can be specified, so this approach supports open-vocabulary mobile manipulation. The
probability density function (PDF) of the Dirichlet distribution is given by
f(θn|γn) =
1
B(γn)
C
Y
c=1
θγc
n−1
n,c
.
(2)
where B is the multivariate beta function and C is the number of classes. In our case, θn is the
categorical distribution for the Gaussian xn. The concentration parameters, γn = (γ1
n, ..., γC
n ), of
the Dirichlet distribution can be updated after each measurement using Bayesian Kernel Inference
as follows [51]
γc
n ←γc
n +
N
X
i=1
yc
i κ(pi, ˜xn; {˜xj}j∈{1,...,N}),
(3)
where yc
i is 1 if pi is of class c and 0 otherwise and κ(·) is defined in eq. (1).
Then, for a 3D Gaussian xn and class c, the expected probability of xn being of category c and its
variance is given by
E[θc
n] =
γc
n
PC
j=1 γj
n
, Var[θc
n] = E[θc
n](1 −E[θc
n])
1 + PC
j=1 γj
n
.
(4)
The variance can be considered a measure of the pixel-wise uncertainty of that class score based
on the multi-view consistency. During rendering we use E[θc
n] and
p
Var[θcn] in place of the color
parameter for rendering the semantic class scores and uncertainty, respectively. Figure 2 shows an
example of how the semantic class score is updated when we obtain a new measurement.
5

<!-- page 6 -->
Yes
Target 
object 
detected
No
Score over
threshold
Yes
Additional 
views useful
Pick 
succeeded
No
Place 
location
detected
Score over
threshold
Additional 
views useful
No
Yes
Yes
Yes
Remove target object
from consideration
Global search objective
Remove place location
from consideration
Search for place
location
Pick object
Yes
Obtain
additional
views
Obtain
additional
views
Place object
Go to candidate
selected by global
search objective
Go to highest-
value frontier or
explore
No
No
No
No
Yes
Figure 3: Method flow chart for HELIOS.
Preliminaries – Information gain. Using the Dirichlet distribution to model the semantic state of
the Gaussians allows us to find the entropy of the concentration parameters [30]
H(θn) =logB(γn) + (T(γn) −C)ψ(T(γn)) −
C
X
c=1
(γc
n −1)ψ(γc
n),
(5)
where T(γn) := PC
c=1 γc
n and ψ is the digamma function.
If we obtain a set of new observations, Y = {y1, ..., ym} at poses P = {p1, .., pm} then the infor-
mation gain is
IG(θn, Y |P) = H(θn) −H(θn|P, Y ).
(6)
Given P and Y , H(θn|P, Y ) can be found by updating θn and then calculating the updated entropy.
Instances for object-level reasoning. We assign 3D Gaussians to instances so we can reason about
objects. Because the objects are not always perfectly segmented this assignment is done by clus-
tering in 3D within Gaussians which have the same most likely semantic class. To prevent the
time requirements becoming intractable for large scenes, we detect which Gaussians are updated for
a new observation and only perform the clustering with these Gaussians and any other Gaussians
within the same instance.
Using these instances we can reason over the set of objects our representation is modeling, let us
call this set O. Each object in O consists of 3D Gaussians belonging to the same instance, and the
class of this object is given by the most common highest-probable class among the 3D Gaussians
belonging to that instance, i.e. for oi ∈O, its class is given by modeθ∈oi

argmaxc∈{classes}E[θo
n]

.
For each object oi ∈O we define the class score Sc :=
1
|oi|
P
θn∈oi E[θc
n], that is, the mean proba-
bility of the 3D Gaussians which make up the instance oi being of class c. Likewise, we define the
uncertainty Uc :=
1
|oi|
P
θn∈oi
p
Var[θcn].
Uncertainty-weighted object score. To determine whether we are confident in our estimate of an
object’s class we define our uncertainty-weighted object score, which takes into account both the
class score and uncertainty (balanced by a hyper-parameter αcs) for an object oi ∈O for class c:
Ψc(oi) := Sc(oi) −αcsUc(oi).
(7)
That is, the lower bound of the αcs-sigma estimate of oi.
3.2
HIERARCHICAL SEARCH
We plan over our hierarchical scene representation in a zero-shot manner, searching for the pick
location using our global search objective to balance between exploring new frontiers and exploiting
semantic information. Once we detect a target object we use our uncertainty-weighted object score
to decide whether we are confident enough in the classification to grasp it. Once the target object
has been grasped we perform a similar search procedure until we are confident we have found the
place location. Figure 3 shows the logical flow of our method.
Global search objective. Our global search objective balances exploring new frontiers with exploit-
ing detections of candidate pick locations. First we introduce some new notation, let A ⊂O be the
set of objects whose class is that of the pick location and let F be the set of frontiers.
6

<!-- page 7 -->
First, we will evaluate the benefit of searching for a detected object. We can work out whether
obtaining additional views Y from poses P of candidate pick location ai ∈A is likely to be in-
formative by considering the information gain (IG). We obtain the proposed poses as described in
the local search section, but we do not have the observations Y unless we move to these poses.
In the case of search, we prioritize avoiding false negatives more than false positives since ul-
timately an effective search policy should provide coverage of the full search space. Thus, we
propose an optimistic approach where we assume the best-case scenario that all the observations
in Y classify ai as the pick location a. Specifically, we define the estimated information gain as
IGa(ai|P, Y ∗) := P
θn∈ai H(θn) −H(θn|P, Y ∗), where Y ∗classifies ai as class a. We will drop
the condition and just write IGa(ai) for brevity. We can then combine the class score and the IG by
multiplying them, i.e. Sa(ai)IGa(ai), to get a measure of how much we want to search a candidate
pick location ai.
This information gain weighted object score allows us to compare candidate objects to each other,
but we want to be able to compare them to frontiers. When we choose a frontier fi ∈F, we store its
location and current score from our value map, denote this F0(fi). During global planning, the first
time each ai ∈A is detected we store the initial class score, Sa0(ai), the initial information gain,
IGa0(ai) as well as its initial center position. Then, we want to find the best candidate object while
taking into account the distance to the frontier. Explicitly, let F′ be the set of previously chosen
frontiers. Then we can calculate an estimated value for a previously chosen frontier f ′
i ∈F′ based
on its proximity to detected candidate objects as:
V0(f ′
i) := maxai∈A

Sa0(ai)IGa0(ai) −αddist(aj, f ′
i)

(8)
where αd is a hyper-parameter which controls the relative importance of candidate object score to
distance and dist(aj, f ′
i) is the Euclidean distance between the stored center of aj and f ′
i. Given
this association between previous frontiers and candidate object scores we can find an association
between frontier scores and candidate object scores by averaging the ratio of this new score to the
frontier score over all the previous frontiers:
F0 :=
1
|F′|
X
f ′
i∈Fp
V0(f ′
i)
F0(f ′
i)
(9)
This allows us to associate a frontier fi with a candidate object score by multiplying its score F(fi)
by F0. We take into account distance to form the following score function for ri ∈A ∪F:
V (ri) :=
Sa(ri)IGa(ri) −αddist(ri)
if ri ∈A
F(ri)F0 −αddist(ri)
if ri ∈F
(10)
where F(fi) is the current score from our value map for fi ∈F and dist(ri) is the Euclidean
distance from the agent to the center point of ri.
Local search. When local search is performed for a candidate pick location, we generate gaze
point positions in a contour around the 2D ground-projection of the 3D Gaussians representing that
location. We use our occupancy map to discard gaze points in occupied regions and remove any gaze
points where there is an occupied region over a certain height in the way between the gaze point and
the pick location. The orientation of a gaze point is set so that the agent will look towards the center
of the object in the ground-plane and the highest point on the object. The robot then goes to each
gaze point, starting from the closest. After performing local search for a candidate pick location we
mark it as visited and no longer consider it a candidate for local search.
When obtaining additional views for a candidate target object or place location, we generate gaze
points in the same way but only go to one. The robot goes the one that would result in the highest
uncertainty-weighted object score if the object was detected as the class it is believed to be from
that viewpoint (i.e. detected as the target object class when considering a target object candidate, or
as the place location class when looking for the place location). We choose to only visit one gaze
point at a time in this case because at this stage we solely focus on determining the object’s class,
whereas when searching the pick location the viewpoints need to provide good enough coverage of
the location to see if the target object is present and so we do not want to just choose the views which
are most informative about the object’s class. However these candidates are not marked as visited
and can be searched more than once.
7

<!-- page 8 -->
Table 1: Ablation study for components of our method, with comparison to using the Home-
Robot [57] baseline agents and recent method MoManipVLA [53] on the val split of the OVMM
challenge. For HomeRobot the results are included for different configurations of skills for naviga-
tion, gaze and place. E.g. R/N/H uses RL for navigation, no skill for gaze and heuristic skill for
place.
Method
FindObj
Pick
FindRec
Place
SR
HomeRobot H/N/H
28.7
15.2
5.3
-
0.4
HomeRobot H/R/R
29.4
13.2
5.8
-
0.5
HomeRobot R/N/H
21.9
11.5
6.0
-
0.6
HomeRobot R/R/R
21.7
10.2
6.2
-
0.4
MoManipVLA1
23.7
12.7
7.1
-
1.7
1 pick
Trusting agent
13.7 ± 1.0
12.3 ± 0.9
6.8 ± 0.7
2.1 ± 0.4
1.3 ± 0.3
W/o global search objective
16.8 ± 1.1
12.0 ± 0.9
6.8 ± 0.7
2.6 ± 0.5
1.7 ± 0.4
HELIOS
23.8 ± 1.2
17.2 ± 1.1
10.0 ± 0.9
3.3 ± 0.5
2.5 ± 0.5
5 picks
Trusting agent
20.4 ± 1.2
18.3 ± 1.1
10.2 ± 0.9
3.2 ± 0.5
1.8 ± 0.4
W/o global search objective
27.8 ± 1.3
21.2 ± 1.2
12.8 ± 1.0
4.9 ± 0.6
2.3 ± 0.4
HELIOS
39.2 ± 1.4
28.7 ± 1.3
17.4 ± 1.1
5.8 ± 0.7
3.1 ± 0.5
Unlim.
Trusting agent
21.9 ± 1.2
19.3 ± 1.1
10.8 ± 0.9
3.3 ± 0.5
1.8 ± 0.4
W/o global search objective
29.6 ± 1.3
22.0 ± 1.2
13.2 ± 1.0
5.0 ± 0.6
2.3 ± 0.4
HELIOS
42.3 ± 1.4
30.5 ± 1.3
18.6 ± 1.1
6.3 ± 0.7
3.2 ± 0.5
4
EXPERIMENTAL RESULTS
4.1
OPEN VOCABULARY MOBILE PICK AND PLACE IN A NOVEL ENVIRONMENT
Dataset and benchmark. We evaluate HELIOS on the validation split of the Home Robot OVMM
benchmark [56, 57] which uses scenes from the Habitat Synthetic Scenes Dataset (HSSD) [26]
in the Habitat simulator [50] and consists of 1199 episodes. In this benchmark, the robot must
carry out an instruction of the form “Move (object) from the (start receptacle) to the
(goal receptacle)” in an unknown environment. An oracle pick skill is provided, and we use
a simple heuristic skill that drops the object for placing.
Metrics. We report the following metrics from the OVMM benchmark [56, 57] indicating the suc-
cess of each phase of the task: FindObj if the robot is ever close enough to the object, Pick
if the robot successfully picks up the object, FindRec if the robot is ever close enough to a
goal receptacle after picking up the object. We additionally report Place which indicates
if the robot placed the object on the goal receptacle and the object remained stationary
on the goal receptacle after the set wait period. We also report the success rate (SR) as de-
fined in the OVMM benchmark – if all of these stages succeeded without collisions, then episode is
considered a success.
Baselines and ablations.
We evaluate the performance of HELIOS compared to the HomeR-
obot [57] baseline agents and MoManipVLA [53]. HomeRobot provides modular implementations
of the skills required to carry out the OVMM task, we compare to the results for their reported con-
figurations. Additionally, to isolate the effects of our hierarchical scene representation and global
search objective, we include the following ablations of our method:
• Trusting agent: this agent uses the same 2D maps and methods for local navigation and
place as our full method, but without the 3D portion of our hierarchical scene representa-
tion, our gaze points and global search objective. It goes to the frontier with the highest
value for the start receptacle until it detects an object (fully trusting the output
of the object detector), at which point it picks up the object. If the pick succeeds, it then
goes to the frontier with the highest value for the goal receptacle until it detects a
goal receptacle, at which point it places the object on it.
• W/o global search objective: this agent uses everything from our full method except for
the global search objective. Instead, it always prioritizes searching candidate objects over
going to frontiers.
• HELIOS: our full method, which uses our global search objective to balance when to
collect views of a detected start receptacle or go to a frontier.
8

<!-- page 9 -->
(a) Scene for hardware experiments with target
object visible.
(b) Scene for hardware experiments with target
object hidden.
(c) Target objects: bowl, coffee cup, drink (left-to-right).
Figure 4: Hardware experiments set-up.
Pick Attempts. In the OVMM benchmark, the agent is allowed an unlimited number of pick at-
tempts. We report results for our method and its ablations with limited numbers of pick attempts
as well as unlimited attempts. With limited attempts, if the agent exceeds the limit, we set all met-
rics for that episode to 0. A benefit of our hierarchical objective is the incorporation of retry logic
when we move back and forth between global and local reasoning. In contrast, the baselines do not
re-attempt picking. In Table 1, we see that allowing 5 pick attempts provides a significant improve-
ment over 1 pick attempt for HELIOS in all metrics. However, the further benefit of unlimited pick
attempts is marginal.
Note that the physical process of grasping the object is not modeled during pick attempts in the
OVMM benchmark. The pick action only fails when the target object is not in frame, revealing
ground truth information about the scene. Thus, our method has access to ground truth information
not accessed by the baselines (which in the real world only corresponds to our method making
additional observations) when attempting greater than 1 pick attempt.
Results. Table 1 shows the results of our benchmarking and ablation study. Our full method limited
to 1 pick outperforms the baselines on all metrics except for FindObj. Adding our hierarchical scene
representation and gaze points improves performance compared to our trusting agent, and adding
our global search objective results in further improvement for all metrics.
The place skill is a major cause of failure for our method. We used a simple approach of dropping
the object above the highest detected point in a region in front of the agent. Because we did
not adjust the orientation of the gripper before dropping, we qualitatively observed that the object
sometimes rolled off the the goal receptacle. Due to the modularity of HELIOS, we could
incorporate other modular solutions to picking without changing our novel contributions.
4.2
HARDWARE EXPERIMENTS
We deploy HELIOS on a Boston Dynamics Spot robot in a real-world office environment. In these
experiments, we utilize the Spot API to perform grasping and to navigate to the waypoints output
by our path planner. We utilize DepthPro [3] for monocular depth estimation.
We create an environment set-up (shown in Figure 4) which is used across 5 experimental scenarios.
In this set-up the robot is tasked with moving objects between the stool, filing cabinet and table.
For objects which start on the stool we conduct experiments both when the object starts in view and
when the view from the robot’s initial location is blocked by a whiteboard. We perform 10 trials
of each experimental scenario. Because methods to run on physical robots are embodiment-specific
we are limited to baselines which are designed for the Spot robot, thus we use the trusting agent
ablation of our method as a baseline. For all experiments the robot is stopped after 10 minutes.
9

<!-- page 10 -->
0
25
50
75
100
HELIOS
Trusting agent
FindObj
Pick
FindRec
Place
0
25
50
75
100
HELIOS
Trusting agent
(a) Move bowl from stool to table
(visible)
0
25
50
75
100
HELIOS
Trusting agent
(b) Move bowl from stool to table
(not visible)
0
25
50
75
100
HELIOS
Trusting agent
(c) Move drink from stool to fil-
ing cabinet (visible)
0
25
50
75
100
HELIOS
Trusting agent
(d) Move drink from stool to fil-
ing cabinet (not visible)
0
25
50
75
100
HELIOS
Trusting agent
(e) Move coffee cup from filing
cabinet to plastic box
0
25
50
75
100
HELIOS
Trusting agent
(f) Average over all 5 experimen-
tal set-ups
Figure 5: Hardware results. Success rates of subtask performance for HELIOS and trusting agent
baseline represented as stacked bar plots. The lowest bar in each column represents the rate of
successfully placing the object, which is the overall success at the task, while the other bars show
the success rate at the earlier subtasks.
(a) HELIOS
(b) Model full scene with 3DGS
Figure 6: GPU memory used to model just objects as in HELIOS verses to model the whole
scene with the rest of the method unchanged.
Figure 5 show the results for our method HELIOS and the baseline, if the robot successfully places
the object then the episode is considered a success. Some qualitative example videos are available
on our project website: https://helios-robot-perception.github.io/
10

<!-- page 11 -->
4.3
EFFICIENCY ANALYSIS
Figure 6 shows the GPU usage over time averaged across 50 episodes. We show this for both our
HELIOS and a version which models the full observed scene with 3D Gaussians (as opposed to
just the task-relevant objects) but is otherwise unchanged to show the difference in the memory
requirements for these scenarios.
We measure the allocated memory just after the scene update to give an indication of the require-
ments for just keeping the scene representation in GPU memory, and give the reserved memory
to give better indication of the peak memory usage. The GPU cache is cleared at the end of each
episode. The allocated and reserved memory are both obtained from PyTorch, thus they represent
the memory usage of our entire method not just the 3DGS portion of our scene representation. As
can be seen in Figure 6, modeling just the objects is more efficient than the full scene, especially
as time increases and the robot observes more of the scene. The maximum of the average allocated
memory is 4.1GB for HELIOS and 8.9GB for the full scene. For the average reserved memory the
maximum is 4.6GB for HELIOS and 24.3GB for the full scene. The maximum reserved memory is
7.1GB for HELIOS and 43.0GB for the full scene.
5
CONCLUSION
We present HELIOS, a hierarchical scene representation and associated search objective, to perform
language-specified pick and place mobile manipulation. We carefully design our novel scene rep-
resentation with associated objective for global and local search. HELIOS achieves state-of-the-art
results on the Open Vocabulary Mobile Manipulation (OVMM) benchmark [56, 57] and improve-
ment over a strong baseline in a real-world office environment with a Spot robot.
Limitations. The performance of HELIOS is limited by errors during execution of subskills includ-
ing collision avoidance and physical placing which can be improved by integrating better compo-
nent methods for physical subskills in future work. Another avenue for increasing performance is
by optimizing the choice of gaze points during local search. Filtering for informative gaze points
or considering the information gain when generating the gaze points could enable us to achieve im-
proved confidence during local search with fewer total gaze points. Reducing the number of gaze
points would allow additional time to enable exploration of more regions in the environment.
ACKNOWLEDGMENTS
The authors gratefully appreciate support through the following grants: NSF FRR 2220868, NSF
IIS-RI 2212433, ONR N00014-22-1-2677, and the Samsung 2025 LEAP-U Program.
REFERENCES
[1] P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Ma-
lik, R. Mottaghi, M. Savva, et al. On evaluation of embodied navigation agents. arXiv preprint
arXiv:1807.06757, 2018.
[2] N. Atanasov, B. Sankaran, J. Le Ny, T. Koletschka, G. J. Pappas, and K. Daniilidis. Hypothesis
testing framework for active object detection.
In 2013 IEEE International Conference on
Robotics and Automation, pages 4216–4222. IEEE, 2013.
[3] A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. R. Richter, and V. Koltun.
Depth pro: Sharp monocular metric depth in less than a second. In International Conference
on Learning Representations, 2025.
[4] A. Brohan, Y. Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho, J. Ibarz, A. Irpan, E. Jang,
R. Julian, et al. Do as i can, not as i say: Grounding language in robotic affordances. In
Conference on robot learning, pages 287–318. PMLR, 2023.
[5] J. Cen, X. Zhou, J. Fang, C. Wen, L. Xie, X. Zhang, W. Shen, and Q. Tian. Tackling view-
dependent semantics in 3d language gaussian splatting. ICML, 2025.
11

<!-- page 12 -->
[6] Y. Chang, L. Fermoselle, D. Ta, B. Bucher, L. Carlone, and J. Wang. Ashita: Automatic
scene-grounded hierarchical task analysis. CVPR, 2025.
[7] D. S. Chaplot, D. Gandhi, A. Gupta, and R. Salakhutdinov. Object goal navigation using
goal-oriented semantic exploration. In Proceedings of Neural Information Processing Systems
(NeurIPS), 2020.
[8] D. S. Chaplot, D. P. Gandhi, A. Gupta, and R. R. Salakhutdinov.
Object goal navigation
using goal-oriented semantic exploration. Advances in Neural Information Processing Systems,
33:4247–4258, 2020.
[9] W. Ding, N. Majcherczyk, M. Deshpande, X. Qi, D. Zhao, R. Madhivanan, and A. Sen. Learn-
ing to view: Decision transformers for active object detection. In 2023 IEEE International
Conference on Robotics and Automation (ICRA), pages 7140–7146. IEEE, 2023.
[10] M. Fey and J. E. Lenssen. Fast graph representation learning with PyTorch Geometric. In
ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.
[11] S. Garrido, L. Moreno, D. Blanco, and F. Martin. Fm2: A real-time fast marching sensor-based
motion planner. In Proceedings of the IEEE/ASME International Conference on Advanced
Intelligent Mechatronics, 2007.
[12] G. Georgakis, B. Bucher, K. Schmeckpeper, S. Singh, and K. Daniilidis. Learning to map for
active semantic goal navigation. arXiv preprint arXiv:2106.15648, 2021.
[13] G. Georgakis, K. Schmeckpeper, K. Wanchoo, S. Dan, E. Miltsakaki, D. Roth, and K. Dani-
ilidis. Cross-modal map learning for vision and language navigation. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 15460–15470, 2022.
[14] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera,
W. Paul, K. Ellis, R. Chellappa, et al. Conceptgraphs: Open-vocabulary 3d scene graphs for
perception and planning. In 2024 IEEE International Conference on Robotics and Automation
(ICRA), pages 5021–5028. IEEE, 2024.
[15] X. Han, H. Liu, F. Sun, and X. Zhang. Active object detection with multistep action prediction
using deep q-network. IEEE Transactions on Industrial Informatics, 15(6):3723–3731, 2019.
[16] D. Honerkamp, M. B¨uchner, F. Despinoy, T. Welschehold, and A. Valada. Language-grounded
dynamic scene graphs for interactive object search with mobile manipulation. IEEE Robotics
and Automation Letters, 2024.
[17] C. Huang, O. Mees, A. Zeng, and W. Burgard. Visual language maps for robot navigation.
In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 10608–
10615. IEEE, 2023.
[18] N. Hughes, Y. Chang, and L. Carlone. Hydra: A real-time spatial perception system for 3D
scene graph construction and optimization. Robotics: Science and Systems (RSS), 2022.
[19] K. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen, S. Li, G. Iyer, S. Saryazdi,
N. Keetha, A. Tewari, J. Tenenbaum, C. de Melo, M. Krishna, L. Paull, F. Shkurti, and A. Tor-
ralba. Conceptfusion: Open-set multimodal 3d mapping. Robotics: Science and Systems
(RSS), 2023.
[20] W. Jiang, B. Lei, K. Ashton, and K. Daniilidis. Multimodal llm guided exploration and active
mapping using fisher information. ICCV, 2025.
[21] L. Jin, X. Zhong, Y. Pan, J. Behley, C. Stachniss, and M. Popovi´c. Activegs: Active scene
reconstruction using gaussian splatting. IEEE Robotics and Automation Letters, 2025.
[22] R. Jin, Y. Gao, Y. Wang, Y. Wu, H. Lu, C. Xu, and F. Gao. Gs-planner: A gaussian-splatting-
based planning framework for active high-fidelity reconstruction. In 2024 IEEE/RSJ Inter-
national Conference on Intelligent Robots and Systems (IROS), pages 11202–11209. IEEE,
2024.
12

<!-- page 13 -->
[23] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten.
Splatam: Splat track & map 3d gaussians for dense rgb-d slam.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21357–21366,
2024.
[24] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.
[25] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik. Lerf: Language embedded ra-
diance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision,
pages 19729–19739, 2023.
[26] M. Khanna, Y. Mao, H. Jiang, S. Haresh, B. Shacklett, D. Batra, A. Clegg, E. Undersander,
A. X. Chang, and M. Savva. Habitat synthetic scenes dataset (hssd-200): An analysis of 3d
scene scale and realism tradeoffs for objectgoal navigation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 16384–16393, 2024.
[27] S. Kobayashi, E. Matsumoto, and V. Sitzmann. Decomposing nerf for editing via feature field
distillation. Advances in neural information processing systems, 35:23311–23330, 2022.
[28] J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with
frozen image encoders and large language models. In International conference on machine
learning, pages 19730–19742. PMLR, 2023.
[29] Y. Li, Y. Ma, X. Huo, and X. Wu. Remote object navigation for service robots using hi-
erarchical knowledge graph in human-centered environments. Intelligent Service Robotics,
15(4):459–473, 2022.
[30] J. Lin. On the dirichlet distribution. Department of Mathematics and Statistics, Queens Uni-
versity, 40, 2016.
[31] P. Liu, Y. Orru, C. Paxton, N. M. M. Shafiullah, and L. Pinto. Ok-robot: What really matters
in integrating open-knowledge models for robotics. arXiv preprint arXiv:2401.12202, 2024.
[32] G. Lu, S. Zhang, Z. Wang, C. Liu, J. Lu, and Y. Tang. Manigaussian: Dynamic gaussian
splatting for multi-task robotic manipulation. In European Conference on Computer Vision,
pages 349–366. Springer, 2024.
[33] D. Maggio, Y. Chang, N. Hughes, M. Trang, D. Griffith, C. Dougherty, E. Cristofalo,
L. Schmid, and L. Carlone. Clio: Real-time task-driven open-set 3d scene graphs. IEEE
Robotics and Automation Letters, 2024.
[34] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison. Gaussian splatting slam. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18039–
18048, 2024.
[35] A. Melnik, M. B¨uttner, L. Harz, L. Brown, G. C. Nandi, A. PS, G. K. Yadav, R. Kala, and
R. Haschke.
Uniteam: Open vocabulary mobile manipulation challenge.
arXiv preprint
arXiv:2312.08611, 2023.
[36] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, T. Funkhouser, et al. Openscene:
3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 815–824, 2023.
[37] X. Puig, E. Undersander, A. Szot, M. D. Cote, R. Partsey, J. Yang, R. Desai, A. W. Clegg,
M. Hlavac, T. Min, T. Gervet, V. Vondrus, V.-P. Berges, J. Turner, O. Maksymets, Z. Kira,
M. Kalakrishnan, J. Malik, D. S. Chaplot, U. Jain, D. Batra, A. Rai, and R. Mottaghi. Habitat
3.0: A co-habitat for humans, avatars and robots, 2023.
[38] J. Qian, Y. Li, B. Bucher, and D. Jayaraman. Task-oriented hierarchical object decomposition
for visuomotor control. In 8th Annual Conference on Robot Learning, 2024.
13

<!-- page 14 -->
[39] J. Qian, A. Panagopoulos, and D. Jayaraman. Recasting generic pretrained vision transform-
ers as object-centric scene encoders for manipulation policies. In 2024 IEEE International
Conference on Robotics and Automation (ICRA), pages 17544–17552. IEEE, 2024.
[40] S. K. Ramakrishnan, D. S. Chaplot, Z. Al-Halah, J. Malik, and K. Grauman.
Poni: Po-
tential functions for objectgoal navigation with interaction-free learning. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18890–18900,
2022.
[41] K. Rana, J. Haviland, S. Garg, J. Abou-Chakra, I. D. Reid, and N. Suenderhauf. Sayplan:
Grounding large language models using 3d scene graphs for scalable task planning. CoRR,
2023.
[42] A. Rashid, S. Sharma, C. M. Kim, J. Kerr, L. Y. Chen, A. Kanazawa, and K. Goldberg. Lan-
guage embedded radiance fields for zero-shot task-oriented grasping. In 7th Annual Conference
on Robot Learning, 2023.
[43] A. Rosinol, M. Abate, Y. Chang, and L. Carlone. Kimera: an open-source library for real-
time metric-semantic localization and mapping. In 2020 IEEE International Conference on
Robotics and Automation (ICRA), pages 1689–1696. IEEE, 2020.
[44] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun,
J. Malik, D. Parikh, and D. Batra. Habitat: A Platform for Embodied AI Research. In Pro-
ceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
[45] F. Schmalstieg, D. Honerkamp, T. Welschehold, and A. Valada. Learning hierarchical inter-
active multi-object search for mobile manipulation. IEEE Robotics and Automation Letters,
2023.
[46] R. Shah, A. Yu, Y. Zhu, Y. Zhu, and R. Mart´ın-Mart´ın. Bumble: Unifying reasoning and
acting with vision-language models for building-wide mobile manipulation. arXiv preprint
arXiv:2410.06237, 2024.
[47] J. Shi, J. Qian, Y. J. Ma, and D. Jayaraman. Plug-and-play object-centric representations from
“what” and “where” foundation models. In ICRA, 2024.
[48] J.-C. Shi, M. Wang, H.-B. Duan, and S.-H. Guan. Language embedded 3d gaussians for open-
vocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 5333–5343, 2024.
[49] M. Simonovsky and N. Komodakis. Dynamic edge-conditioned filters in convolutional neural
networks on graphs. In Proceedings of the IEEE conference on Computer Vision and Pattern
Recognition (CVPR), 2017.
[50] A. Szot, A. Clegg, E. Undersander, E. Wijmans, Y. Zhao, J. Turner, N. Maestre, M. Mukadam,
D. Chaplot, O. Maksymets, A. Gokaslan, V. Vondrus, S. Dharur, F. Meier, W. Galuba,
A. Chang, Z. Kira, V. Koltun, J. Malik, M. Savva, and D. Batra. Habitat 2.0: Training home
assistants to rearrange their habitat. In Advances in Neural Information Processing Systems
(NeurIPS), 2021.
[51] J. Wilson, M. Almeida, M. Sun, S. Mahajan, M. Ghaffari, P. Ewen, O. Ghasemalizadeh, C.-H.
Kuo, and A. Sen. Modeling uncertainty in 3d gaussian splatting through continuous semantic
splatting. arXiv preprint arXiv:2411.02547, 2024.
[52] Q. Wu, Z. Fu, X. Cheng, X. Wang, and C. Finn. Helpful doggybot: Open-world object fetching
using legged robots and vision-language models. arXiv preprint arXiv:2410.00231, 2024.
[53] Z. Wu, Y. Zhou, X. Xu, Z. Wang, and H. Yan. Momanipvla: Transferring vision-language-
action models for general mobile manipulation. Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2025.
[54] J. Ye, D. Batra, A. Das, and E. Wijmans. Auxiliary tasks and exploration enable objectnav.
ICCV, 2021.
14

<!-- page 15 -->
[55] V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye, J. Hu, M. Tancik,
and A. Kanazawa. gsplat: An open-source library for gaussian splatting. Journal of Machine
Learning Research, 26(34):1–17, 2025.
[56] S. Yenamandra, A. Ramachandran, M. Khanna, K. Yadav, D. S. Chaplot, G. Chhablani,
A. Clegg, T. Gervet, V. Jain, R. Partsey, R. Ramrakhya, A. Szot, T.-Y. Yang, A. Edsinger,
C. Kemp, B. Shah, Z. Kira, D. Batra, R. Mottaghi, Y. Bisk, and C. Paxton. The homerobot
open vocab mobile manipulation challenge. In Thirty-seventh Conference on Neural Informa-
tion Processing Systems: Competition Track, 2023.
[57] S. Yenamandra, A. Ramachandran, K. Yadav, A. S. Wang, M. Khanna, T. Gervet, T.-Y. Yang,
V. Jain, A. Clegg, J. M. Turner, Z. Kira, M. Savva, A. X. Chang, D. S. Chaplot, D. Batra,
R. Mottaghi, Y. Bisk, and C. Paxton. Homerobot: Open-vocabulary mobile manipulation. In
7th Annual Conference on Robot Learning, 2023.
[58] N. Yokoyama, A. Clegg, J. Truong, E. Undersander, T.-Y. Yang, S. Arnaud, S. Ha, D. Batra,
and A. Rai. Asc: Adaptive skill coordination for robotic mobile manipulation. IEEE Robotics
and Automation Letters, 9(1):779–786, 2023.
[59] N. H. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher. Vlfm: Vision-language frontier
maps for zero-shot semantic navigation. In 2nd Workshop on Language and Robot Learning:
Language as Grounding, 2023.
[60] B. Yu, H. Kasaei, and M. Cao. L3mvn: Leveraging large language models for visual target
navigation. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS), pages 3554–3560. IEEE, 2023.
[61] C. Zhang and G. H. Lee. econsg: Efficient and multi-view consistent open-vocabulary 3d
semantic gaussians. International Conference on Learning Representations, 2025.
[62] J. Zhang, L. Dai, F. Meng, Q. Fan, X. Chen, K. Xu, and H. Wang. 3d-aware object goal
navigation via simultaneous exploration and identification. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 6672–6682, 2023.
[63] K. Zheng, A. Paul, and S. Tellex. Asystem for generalized 3d multi-object search. In 2023
IEEE International Conference on Robotics and Automation (ICRA), pages 1638–1644. IEEE,
2023.
[64] Y. Zheng, X. Chen, Y. Zheng, S. Gu, R. Yang, B. Jin, P. Li, C. Zhong, Z. Wang, L. Liu, et al.
Gaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping. IEEE
Robotics and Automation Letters, 2024.
[65] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi.
Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
21676–21685, 2024.
[66] X. Zhou, R. Girdhar, A. Joulin, P. Kr¨ahenb¨uhl, and I. Misra. Detecting twenty-thousand classes
using image-level supervision. In European conference on computer vision, pages 350–368.
Springer, 2022.
[67] Y. Zhu, Z. Jiang, P. Stone, and Y. Zhu. Learning generalizable manipulation policies with
object-centric 3d representations. In 7th Annual Conference on Robot Learning, 2023.
15

<!-- page 16 -->
6
APPENDIX
We provide implementation details of our method including our hyper-parameter choices (Sec-
tion 6.1), compute details (Section 6.2), additional ablations using ground truth semantics (Sec-
tion 6.3) and on the choice of assumption when calculating the information gain (Section 6.4),
a sensitivity analysis of our hyper-parameters (Section 6.5), more detailed failure analysis (Sec-
tion 6.6) and details including licenses of existing assets we use in this work (Section 6.7).
6.1
IMPLEMENTATION DETAILS
Hyperparameters. The value of the hyper-parameters used in our experiments are given in Table 2.
Once the target object has been detected. First we introduce some new notion, let G ⊂O be the
set of objects whose class is that of the target object and A ⊂O be the set of objects whose class is
that of the place location.
Once a candidate target gi ∈G has been detected we check if it’s uncertainty-weighted object score
is over some threshold τg, and if Ψg(gi) ≥τg we will treat gi as the target object.
If Ψg(gi) < τg we can calculate the class score, S′(gi), and the uncertainty, U ′(gi), if we take m
observations Y from poses P and again assume the best-case scenario that each classified gi as class
g. Then we can obtain the class score this would give us as
Ψ′
g(gi) := S′(gi) −αcsU ′(gi).
(11)
When deciding where to obtain additional views, we consider both if obtaining these views could in-
crease the uncertainty-weighted object score to above the threshold and if the increase in uncertainty-
weighted object score is larger than a threshold τinc. This second condition is so that the agent can
obtain views of objects which have not been observed much, and so will have a lower uncertainty-
weighted object score due to higher uncertainty. If Ψ′
g(gi) ≥min(τg, Ψg(gi) + τinc) then we will
obtain the additional observations of gi, otherwise we return to global search.
After the target object has been grasped, we use the same formulation to decide whether something
is the correct class for a place location as we do for deciding whether to grasp a target object but
potentially with a different threshold. That is, if we have seen a candidate place location bi ∈B
we first check if Ψb(bi) ≥τb and if so we go there to place the target object, otherwise we check if
obtaining additional views satisfies Ψ′
b(bi) ≥min(τb, Ψb(bi) + τinc) and if so we obtain them. If
not or if there is no candidate bi we go to the frontier with the highest value for the place location.
Table 2: Hyperparameters. We provide a list of the hyper-parameters of our method with a de-
scription and the value used in our experiments. Some hyperparameters are only referenced in the
supplementary material and not in the main paper.
Name
Description
Value
αcs
Weighting of uncertainty for uncertainty-weighted object score
1
αd
Weighting of distance term for global search objective
0.001
τg
Threshold for uncertainty-weighted object score to pick up an object
0.5
τb
Threshold for uncertainty-weighted object score to place on a
0.5
goal receptacle
τinc
Minimum change in uncertainty-weighted object score that would
0.05
cause us to look at an object or goal receptacle
oda
Threshold for object detector confidence for start receptacle class
0.35
odg
Threshold for object detector confidence for object class
0.25
odb
Threshold for object detector confidence for goal receptacle class
0.45
csa
Class score for an object to be considered a candidate start receptacle
0.3
for the global search objective
csg
Class score for an object to be considered a candidate object for deciding
0.3
whether to obtain additional views
csb
Class score for an object to be considered a candidate goal receptacle
0.3
for deciding whether to obtain additional views
αcpa
Absolute concentration parameter update scaling
3
16

<!-- page 17 -->
Path Planner.
We modify the fast marching squared [11] motion planner from Home Robot
OVMM’s baseline [56] to generate navigation actions from the map and the goal pose. Similar
to the baseline, our planner also builds the arrival-time map with velocity directly proportional to
the distance from the closest obstacle, which balances the efficiency and safety of the motion plan.
However, to account for the fine navigation actions required for mobile manipulation, we make 3
modifications to the baseline: 1. Our planner doubles the resolution of the map at 2000 x 2000 cells
of 2.5cm x 2.5cm, as the map is directly derived from the depth observations instead of being pre-
dicted through a neural network as in the baseline [7]. 2. Our planner supports continuous actions of
moving forward [0.1m, 1.0m] or rotating [5◦, 30◦], as opposed to fixed actions of moving forward
0.3m or rotating 30◦from the baseline. 3. Our planner explicitly verifies that all intermediate po-
sitions for a forward move are collision-free, greatly improving safety around tighter choke-points
common in home environments.
Modifications to 3DGS semantic update. We apply a scaling αcpa directly to the concentration
parameter update to control the speed of this update, which corresponds to each observation being
repeated αcpa times.
Additional details of 3DGS instance creation. We spatially cluster Gaussians into instances by
putting the Gaussians in a voxel grid based on the Gaussian’s center, clustering them by connected
components of neighboring voxels, and assigning instance labels to the clusters based on previous
assignments. First, we put Gaussians of the same semantic label into a grid of 0.5m x 0.5m x
0.5m (adequate due to the spatial sparsity of relevant objects) voxels aligned with the odometry
coordinate frame. Then, we take the connected components on the graph of 26-connected voxels
containing Gaussians. Finally, we assign instance labels to each cluster by taking the minimum of
previous instance labels over all Gaussians in the cluster. If no Gaussian in a cluster previously
had an instance label, we assign (maximum instance label over all Gaussians) + 1. In practice,
this is implemented as a sequence of max object size=10m
voxel size=0.5m
= 20 min pooling operations on a voxel
grid neighborhood graph [49] using the pytorch geometric library [10]. Note we perform the above
procedure with only the Gaussians which were updated by the last measurement or which were
assigned to the same instance as any of these updated Gaussians.
Gaussian creation. We detect when a new observation represents data which is not already part
of our scene representation using the depth error. When an observation is taken, we first make a
mask of the pixels which have been detected as an object of interest. Within this mask, we calculate
the absolute difference between the measured depth and the rendered depth. We then mask this
difference again to keep only the parts where the measured depth is over 0. We find the parts of this
difference which are over 1m or over 0.001m and remain after an erosion operation, and create a
new Gaussian for each of them. Each Gaussian’s position is initialized using the measured depth
and camera pose to obtain it’s 3D location.
Re-observing previously detected parts of the scene. As we only model parts of the scene with 3D
Gaussians we need to detect when we are re-observing an area which is modeled with 3D Gaussians
versus looking towards such an area which is occluded. If we did not do this and only updated the
representation when an object is detected then we would not include any negative results (i.e. an
object not being detected) and thus we would become over-confident in the classes of objects. One
possibility would be to just update if there are any 3D Gaussians in the viewing direction as if they
are occluded the new Gaussians should be placed on the occluding object not on the original object,
however this is inefficient. Thus we render the depth of our 3D Gaussian scene representation in
the viewing direction and then find the pixels in the measured depth image with less than 0.5m of
difference to this rendering and finally perform a morphological transformation to close small holes.
We then only update the 3D Gaussians using the rendering which lies within this mask.
Expanded explanation of how we calculate information gain. When updating the global objective
score we use
IGo(oi|P, Y ∗) :=
X
θn∈oi
H(θn) −H(θn|P, Y ∗).
(12)
To obtain Y ∗, for each θn ∈oi we create a copy of the associated 3D Gaussian but with the se-
mantic class probabilities set to 1 for the class o and 0 for all other classes, then render using these
parameters at pose P – this rendered image is used as Y ∗. Then using Y ∗we update a copy of the
17

<!-- page 18 -->
Table 3: Ablation study for including ground-truth semantics. We show the performance in-
crease from using ground-truth semantics (with gt) for both our trusting agent, which does not
reason about the uncertainty of object detections, and our full method HELIOS, which does. We
show the results for our methods with unlimited picks. We also include results of the recent method
MoManipVLA [53] for additional comparison. The standard error of the mean is indicated.
Method
FindObj
Pick
FindRec
Place
SR
MoManipVLA
23.7
12.7
7.1
-
1.7
MoManipVLA with gt
66.1
62.6
53.1
-
15.8
Trusting agent
21.9 ± 1.2
19.3 ± 1.1
10.8 ± 0.9
3.3 ± 0.5
1.8 ± 0.4
Trusting agent with gt
57.5 ± 1.4
56.5 ± 1.4
44.7 ± 1.4
20.9 ± 1.2
12.8 ± 1.0
HELIOS
42.3 ± 1.4
30.5 ± 1.3
18.6 ± 1.1
6.3 ± 0.7
3.2 ± 0.5
HELIOS with gt
66.3 ± 1.4
58.3 ± 1.4
53.4 ± 1.4
29.8 ± 1.3
21.0 ± 1.2
concentration parameters using Eq. 3 and re-calculate the entropy using the updated concentration
parameters with Eq. 5 to obtain H(θn|P, Y ∗).
Object detector. We use the DETIC [66] object detector as implemented in the HomeRobot code-
base. We set separate thresholds for the detections for each class, with the thresholds for the object
and start receptacle a bit lower than the default used by HomeRobot (0.45) as our method is
designed to filter out false positives but does not address false negatives as shown in Table 2.
6.2
COMPUTATIONAL RESOURCES
The experiments presented in this paper ran on an internal cluster using a mix of 2080ti GPUs with
11GB of VRAM and L40 GPUs with 48GB of VRAM. Each full run of our method or its ablations
on the val split took around 288 GPU hours for 1199 episodes.
6.3
ABLATION USING GROUND TRUTH SEMANTICS
We perform an ablation study to show the effect of using ground-truth semantics on performance,
the results are shown in Table 3. We can see that our full method outperforms our trusting agent
when both use ground truth semantics, this may be due to fact that HELIOS performs local search
of detected pick locations whereas our trusting agent doesn’t. The gap between the pick success of
our trusting agent and our full method is much smaller with ground truth semantics (11.2% without
ground truth semantics and 1.8% with ground truth semantics). Likewise, the gap in pick success
with and without semantics is much higher for both MoManipVLA and our trusting agent than for
HELIOS (49.9% for MoManipVLA, 37.2% for our trusting agent and 27.8% for HELIOS). These
results indicate that our full method is less of an improvement when ground truth semantics are used.
This makes sense because alleviating issues from imperfect object detections is the main focus of
the components of HELIOS which are included in the full method but not in our trusting agent.
Addressing this challenge is not necessary when ground truth semantics are provided.
The relatively low overall success rates with ground truth semantics for both MoManipVLA and our
method indicate there is still more work required to increase search efficiency and the success rate
of physical subskills such as collision-free navigation and place. However the large gap between the
results with and without ground truth semantics for MoManipVLA and our trusting agent, especially
for the pick skill, still shows that robust object detection is a key bottleneck for this task. While
HELIOS still has a performance gap when not using ground truth semantics it takes a step towards
addressing this issue.
6.4
ABLATION STUDY ON INFORMATION GAIN ASSUMPTION
Table 4 shows an ablation of what assumption about the measurement we use to calculate the in-
formation gain. HELIOS uses the Optimistic update which assumes that the measurement will be
whatever object we are looking for (so if we think something might be a pick location we assume
the measurement will be the class of the pick location). We compare to using a more conservative
estimate (50-50) which assigns 50% probability of the measurement being whatever object we are
18

<!-- page 19 -->
Table 4: Ablation study for the information gain update assumption on the val split of the
OVMM challenge.
N picks
Method
FindObj
Pick
FindRec
Place
SR
1
50-50
14.5 ± 1.0
10.5 ± 0.9
6.6 ± 0.7
1.9 ± 0.4
1.5 ± 0.4
Optimistic
23.8 ± 1.2
17.2 ± 1.1
10.0 ± 0.9
3.3 ± 0.5
2.5 ± 0.5
5
50-50
23.1 ± 1.2
17.9 ± 1.1
12.0 ± 0.9
3.9 ± 0.6
2.2 ± 0.4
Optimistic
39.2 ± 1.4
28.7 ± 1.3
17.4 ± 1.1
5.8 ± 0.7
3.1 ± 0.5
Unlim.
50-50
26.0 ± 1.3
19.5 ± 1.1
12.7 ± 1.0
3.9 ± 0.6
2.2 ± 0.4
Optimistic
42.3 ± 1.4
30.5 ± 1.3
18.6 ± 1.1
6.3 ± 0.7
3.2 ± 0.5
Table 5: Sensitivity analysis results for 1 pick.
Param
Value
FindObj
Pick
FindRec
Place
SR
αcs
0.1
22.8 ± 1.2
15.3 ± 1.0
8.8 ± 0.8
2.6 ± 0.5
2.3 ± 0.4
1
23.8 ± 1.2
17.2 ± 1.1
10.0 ± 0.9
3.3 ± 0.5
2.5 ± 0.5
2
24.1 ± 1.2
16.6 ± 1.1
10.6 ± 0.9
3.4 ± 0.5
2.2 ± 0.4
αd
0.0001
22.8 ± 1.2
15.4 ± 1.0
9.0 ± 0.8
2.9 ± 0.5
1.8 ± 0.4
0.001
23.8 ± 1.2
17.2 ± 1.1
10.0 ± 0.9
3.3 ± 0.5
2.5 ± 0.5
0.01
23.4 ± 1.2
16.5 ± 1.1
9.8 ± 0.9
2.9 ± 0.5
2.2 ± 0.4
τg
0.3
20.0 ± 1.2
14.5 ± 1.0
7.8 ± 0.8
2.1 ± 0.4
1.1 ± 0.3
0.5
23.8 ± 1.2
17.2 ± 1.1
10.0 ± 0.9
3.3 ± 0.5
2.5 ± 0.5
0.7
25.0 ± 1.3
16.5 ± 1.1
10.2 ± 0.9
3.0 ± 0.5
1.8 ± 0.4
τinc
0.05
23.8 ± 1.2
17.2 ± 1.1
10.0 ± 0.9
3.3 ± 0.5
2.5 ± 0.5
0.1
21.4 ± 1.2
14.9 ± 1.0
7.8 ± 0.8
1.8 ± 0.4
1.0 ± 0.3
looking for and 50% it being the other/background class. As we can see the optimistic assumption
performs much better.
6.5
SENSITIVITY ANALYSIS
Table 5, Table 6 and Table 7 show the effect of some hyper-parameter changes in the 1-pick, 5-
picks and unlimited picks cases respectively. Only one hyper-parameter is modified at a time, all
unmodified hyper-parameters use the values given in Table 2. These experiments are all performed
for our method HELIOS without ground truth semantics.
Table 6: Sensitivity analysis results for 5 picks.
Param
Value
FindObj
Pick
FindRec
Place
SR
αcs
0.1
38.5 ± 1.4
27.6 ± 1.3
16.2 ± 1.1
5.8 ± 0.7
3.3 ± 0.5
1
39.2 ± 1.4
28.7 ± 1.3
17.4 ± 1.1
5.8 ± 0.7
3.1 ± 0.5
2
38.5 ± 1.4
26.8 ± 1.3
17.4 ± 1.1
6.4 ± 0.7
3.3 ± 0.5
αd
0.0001
36.2 ± 1.4
26.2 ± 1.3
15.6 ± 1.0
5.0 ± 0.6
2.4 ± 0.4
0.001
39.2 ± 1.4
28.7 ± 1.3
17.4 ± 1.1
5.8 ± 0.7
3.1 ± 0.5
0.01
38.5 ± 1.4
28.3 ± 1.3
17.3 ± 1.1
5.8 ± 0.7
3.0 ± 0.5
τg
0.3
34.7 ± 1.4
26.2 ± 1.3
14.8 ± 1.0
4.7 ± 0.6
1.8 ± 0.4
0.5
39.2 ± 1.4
28.7 ± 1.3
17.4 ± 1.1
5.8 ± 0.7
3.1 ± 0.5
0.7
36.6 ± 1.4
25.0 ± 1.3
15.9 ± 1.1
5.2 ± 0.6
2.5 ± 0.5
τinc
0.05
39.2 ± 1.4
28.7 ± 1.3
17.4 ± 1.1
5.8 ± 0.7
3.1 ± 0.5
0.1
37.1 ± 1.4
26.9 ± 1.3
14.7 ± 1.0
3.7 ± 0.5
1.3 ± 0.3
19

<!-- page 20 -->
Table 7: Sensitivity analysis results for unlimited picks.
Param
Value
FindObj
Pick
FindRec
Place
SR
αcs
0.1
42.0 ± 1.4
29.8 ± 1.3
17.1 ± 1.1
6.1 ± 0.7
3.3 ± 0.5
1
42.3 ± 1.4
30.5 ± 1.3
18.6 ± 1.1
6.3 ± 0.7
3.2 ± 0.5
2
39.9 ± 1.4
27.5 ± 1.3
17.9 ± 1.1
6.8 ± 0.7
3.4 ± 0.5
αd
0.0001
38.5 ± 1.4
27.6 ± 1.3
16.3 ± 1.1
5.3 ± 0.6
2.4 ± 0.4
0.001
42.3 ± 1.4
30.5 ± 1.3
18.6 ± 1.1
6.3 ± 0.7
3.2 ± 0.5
0.01
41.7 ± 1.4
30.4 ± 1.3
18.6 ± 1.1
6.2 ± 0.7
3.0 ± 0.5
τg
0.3
38.4 ± 1.4
28.6 ± 1.3
16.2 ± 1.1
4.9 ± 0.6
1.8 ± 0.4
0.5
42.3 ± 1.4
30.5 ± 1.3
18.6 ± 1.1
6.3 ± 0.7
3.2 ± 0.5
0.7
38.1 ± 1.4
25.7 ± 1.3
16.3 ± 1.1
5.4 ± 0.7
2.6 ± 0.5
τinc
0.05
42.3 ± 1.4
30.5 ± 1.3
18.6 ± 1.1
6.3 ± 0.7
3.2 ± 0.5
0.1
39.8 ± 1.4
28.2 ± 1.3
15.5 ± 1.0
3.8 ± 0.5
1.3 ± 0.3
6.6
MORE DETAILED FAILURE ANALYSIS
Figure 7 shows the failure cases breakdown for HELIOS and Figure 8 shows the failure cases when
using ground truth semantics, both in simulation. As can be seen, the collisions between the robot
and scene are a major cause of failure. This could be addressed by a better local path planner, how-
ever we expect that what works well in this simulation environment may not work well in the real
world or even in other simulators. Thus we consider improving this aspect of our method to be sec-
ondary to some other sources of failure, even if they are less significant in this setting. Failure to find
the target object is the second largest cause of failure, and the largest cause of failure without ground
truth semantics. Because it is still a large cause of failure even with ground truth semantics this fail-
ure seems to be mostly due to inefficient search rather than incorrect object detections. Improving
the efficiency of our local search could help with this, for example by optimizing the choice of gaze
points. It may also be possible to incorporate future improvements in semantic object navigation to
improve the overall efficiency of searching for objects. Finally we see that the place skill is a major
failure cause, in particular objects not being placed correctly onto the place location or rolling off
(our metrics cannot distinguish between these cases). Note that this remains a major cause of failure
when using ground truth semantics.
The difference in the collision rates between HELIOS with and without ground truth could be due
to the robot taking longer to find the relevant objects when ground truth is not provided, thus giving
more opportunities for it to collide with the environment.
Figure 9 shows the failure cases breakdown for the hardware experiments. The place skill is still a
major cause of failure, including objects being placed too close to the edge or bouncing off. The
local planner is also a significant source of failure, there is some noise from the depth sensors on
the robot that results in the occupancy map having multiple small obstacles in free space which can
result in the planner getting stuck. In addition the glass walls do not get picked up as obstacles so
sometimes the planner tries to path through them which also results in the robot getting stuck or
needing to be manually stopped. The grasp skill also sometimes failed to find angles to attempt
to grasp from, the main time this happened seems to be when the robot was in a position where it
couldn’t step backwards so it was too close to the object when attempting to grasp. When the grasp
skill was executed it occasionally failed due to knocking the object off the pick location.
20

<!-- page 21 -->
Figure 7: Failure mode break-down in simulation for HELIOS.
Figure 8: Failure mode break-down in simulation for HELIOS with ground truth semantics.
21

<!-- page 22 -->
Figure 9: Failure mode break-down in hardware for HELIOS.
Figure 10: Failure mode break-down in hardware for trusting agent baseline.
22

<!-- page 23 -->
6.7
DETAILS OF EXISTING ASSETS USED
Directly-used assets:
• Home Robot OVMM benchmark and code [56, 57]: MIT License, commit ede6a67a
(main branch as of submission).
https://github.com/facebookresearch/
home-robot
• Habitat
Synthetic
Scenes
Dataset
(HSSD)
[26]:
cc-by-nc-4.0,
obtained
using
Home Robot’s download script https://huggingface.co/datasets/hssd/
hssd-hab
• Habitat [37, 44, 50]: MIT License, for habitat-lab we used HomeRobot’s modified code,
for habitat-sim we use v0.2.5.
https://github.com/facebookresearch/habitat-lab
https://github.com/facebookresearch/habitat-sim
• VLFM [59]: MIT License
https://github.com/bdaiinstitute/vlfm
• gsplat [55]: Apache License 2.0
https://github.com/nerfstudio-project/gsplat
• SplaTAM [23]: BSD 3-Clause License, some code used with modifications rather than
directly importing
https://github.com/spla-tam/SplaTAM/
Key assets used in above works that we also use:
• BLIP2 [28]: BSD 3-Clause License, v1.0.2
https://github.com/salesforce/LAVIS
• DETIC [66]: Apache License 2.0, installed via HomeRobot
https://github.com/facebookresearch/Detic
23
