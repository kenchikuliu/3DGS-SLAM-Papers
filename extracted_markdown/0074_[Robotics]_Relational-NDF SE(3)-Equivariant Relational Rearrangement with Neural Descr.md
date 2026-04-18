<!-- page 1 -->
SE(3)-Equivariant Relational Rearrangement with
Neural Descriptor Fields
Anthony Simeonov∗,1,2, Yilun Du∗,2, Lin Yen-Chen2
Alberto Rodriguez3, Leslie Pack Kaelbling2, Tom´as Lozano-P´erez2, Pulkit Agrawal1,2
Massachusetts Institute of Technology
1 Improbable AI Lab, 2CSAIL, 3Department of Mechanical Engineering, ∗Equal Contribution
Abstract: We present a method for performing tasks involving spatial relations
between novel object instances initialized in arbitrary poses directly from point
cloud observations. Our framework provides a scalable way for specifying new
tasks using only ∼5-10 demonstrations. Object rearrangement is formalized as
the question of ﬁnding actions that conﬁgure task-relevant parts of the object in
a desired alignment. This formalism is implemented in three steps: assigning a
consistent local coordinate frame to the task-relevant object parts, determining
the location and orientation of this coordinate frame on unseen object instances,
and executing an action that brings these frames into the desired alignment. We
overcome the key technical challenge of determining task-relevant local coordinate
frames from a few demonstrations by developing an optimization method based on
Neural Descriptor Fields (NDFs) and a single annotated 3D keypoint. An energy-
based learning scheme to model the joint conﬁguration of the objects that satisﬁes
a desired relational task further improves performance. The method is tested on
three multi-object rearrangement tasks in simulation and on a real robot. Project
website, videos, and code: https://anthonysimeonov.github.io/r-ndf/
Keywords: Object Relations, Rearrangement, Manipulation, Neural Fields
1
Introduction
Many tasks we want robots to perform – e.g., stacking bowls and plates to declutter a table, putting
objects together to build an assembly, and hanging mugs on a rack with hooks – involve rearranging
objects relative to one another. Such tasks can be described in terms of spatial relations between parts
of a set of rigid objects. The desired relation can be achieved by ﬁrst attaching a local coordinate
frame to task-relevant parts of the object and then transforming the objects in a way that brings these
coordinate frames into the desired alignment. For example, hanging a mug on a rack is a relation
between the mug’s handle and the rack’s hook, while stacking a bowl on a mug involves aligning the
bottom of the bowl with the top of the mug (see Fig. 1).
Specifying and solving tasks in this way requires the ability to (i) assign a consistent local coordinate
frame to the task-relevant object parts, and (ii) detect the corresponding coordinate frames on new
object instances. Some prior works use large task-speciﬁc datasets with human-labeled keypoints
that identify the task-relevant parts [1, 2], but heavy dependence on manual annotation limits easy
deployment of such approaches for a wide diversity of tasks. Neural Descriptor Fields (NDFs) [3]
overcome the need for large-scale annotation by leveraging task-agnostic self-supervised pretraining,
followed by just a small set of task demonstrations (∼5-10) to both identify the task-relevant object
parts and assign each part an oriented local coordinate frame. NDFs have been shown to successfully
localize these local coordinate frames at the corresponding parts of new object instances.
While NDFs require less task-speciﬁc data, labeling the relevant object parts in a consistent fashion
can still be tedious – e.g., one must assign an orientation to the “handle” of multiple mugs and ensure
they are all consistent. Prior work [3] instead used demonstrations of the relation to associate a single
frame, assigned to the second object, with the task-relevant part of each manipulated object (e.g., label
a frame on the “hook” of a rack once, and associate this frame with each mug’s “handle” based on the
6th Conference on Robot Learning (CoRL 2022), Auckland, New Zealand.
arXiv:2211.09786v1  [cs.RO]  17 Nov 2022

<!-- page 2 -->
Initial configuration
(unseen objects in random poses)
Final configuration, with relation 
“bowl upright on mug” satisfied
Infer and execute relation “mug upright on table”
Infer and execute relation “bowl upright on mug”
Figure 1: Given a point cloud of a pair of unseen objects in arbitrary initial conﬁgurations (top left), Relational
Neural Descriptor Fields (R-NDFs) obtain relative transformations that satisfy a relational task objective, such as
“placing the mug upright on the table” (middle) and “stacking the bowl upright on top of the mug” (right). Our
framework obtains these transformations by inferring the 6D pose of local coordinate frames at the task-relevant
parts of the objects using just a small handful (∼5-10) of demonstrations of each relational task.
demonstrated interaction between the “handle” and the “hook”). However, this makes the limiting
assumption that the secondary object is known [3] – in the hanging example, the system generalizes
to unseen mugs, but fails if the rack is in a new pose or has a different shape. Our work addresses this
fundamental limitation of using NDFs for relational tasks. We present Relational Neural Descriptor
Fields (R-NDFs), a framework, using ∼5-10 demonstrations, that takes as input 3D point clouds of a
pair of unseen objects in arbitrary initial poses and outputs a relative transformation between them
that satisﬁes a relational task objective.
The central difﬁculty in applying NDFs to scenarios with changing pairs of objects is to assign a set
of consistent local coordinate frames to the task-relevant parts of the objects in the demonstrations,
which may be both unaligned and differently shaped. We propose an optimization method that uses
two NDFs (one per object) and a single 3D keypoint label in just one of the demonstrations, to assign
a set of local coordinate frames that are consistently posed relative to the task-relevant parts of the
objects. We then apply NDFs to localize the corresponding coordinate frames for unseen pairs of
objects presented in arbitrary initial poses, and solve for the relative transformation between them that
satisﬁes the desired relation. However, errors can accumulate when inferring a relative transformation
based on a pair of coordinate frames that have been independently localized. To mitigate this effect,
we also propose a learning approach that directly models the joint conﬁguration of the pair of objects
and helps reﬁne the transformation for satisfying the relation.
We validate R-NDFs on three relational rearrangement tasks in both simulation and the real world.
Our simulation results show that R-NDFs outperform a set of baseline approaches, and our proposed
optimization and learning-based reﬁnement schemes beneﬁt overall task success. Finally, our real
world results exhibit the effectiveness of R-NDFs on pairs of diverse real world objects in tabletop
pick-and-place, and highlight the potential for applying our approach to multi-step tasks.
2
Background: Neural Descriptor Fields
A Neural Descriptor Field (NDF) [3] represents an object using a function f that maps a 3D coordinate
x ∈R3 and an object point cloud P ∈R3×N to a spatial descriptor in Rd:
f(x|P) : R3 × R3×N →Rd.
(1)
The function f is parameterized as a neural network constructed to be SE(3)-equivariant, such that if
an object is subject to a rigid body transform T ∈SE(3) its spatial descriptors transform accordingly*:
f(x|P) ≡f(Tx|TP).
(2)
This enables NDFs to behave consistently for the same object, regardless of the underlying pose.
NDFs are also trained to learn correspondence over objects in the same category, so that points
near similar geometric features of different instances (e.g., a point near the handle of two different
mugs) are mapped to similar descriptor values. The equivariance property is obtained by using
SO(3)-equivariant neural network layers [4] and mean-centered point clouds, while the category-level
correspondence is obtained by training f on a category-level 3D reconstruction task [3, 5].
*We use homogeneous coordinates for ease of notation, i.e., Tx denotes Rx+t where T = (R, t) ∈SE(3).
2

<!-- page 3 -->
NDFs can also be redeﬁned to model a ﬁeld over full SE(3) poses, rather than individual points. This
is achieved by concatenating the descriptors of the individual points in a rigid set of query points
X ∈R3×Nq, i.e., a set of three or more non-collinear points xi, i = 1...Nq, that are constrained to
transform together rigidly. This construction allows NDFs to represent an SE(3) pose T via its action
on X, i.e., via the coordinates of the transformed query point cloud TX:
Z = F(T|P) =
M
xi∈X
f(Txi|P)
(3)
Thus, F maps a point cloud P and an SE(3) pose T to a category-level pose descriptor Z ∈Rd×Nq,
where F inherits the same SE(3)-equivariance from f.
3
General Problem Setup and Preliminaries
Our high-level goal is to enable a user to specify a task involving a geometric relationship between a
pair of rigid objects, and enable a robot to perform this task on unseen object instances presented in
arbitrary initial poses. Examples of relations we consider include “mug hanging on a rack”, “bowl
stacked upright on a mug”, and “bottle placed upright on a tray”.
Concretely, our goal is to build a system that takes as input two (nearly complete) 3D point clouds
PA and PB (each segmented out from the overall scene) of objects OA and OB, and outputs an
SE(3) transformation TB for transforming OB into a conﬁguration that satisﬁes a desired relation
between OA and OB. We represent the relation as an alignment between a pair of local coordinate
frames attached to task-relevant geometric features of the objects, and break down the problem of
obtaining TB into (i) assigning a set of consistent coordinate frames to the task-relevant local object
parts and (ii) localizing these coordinate frames on the relevant parts of the new objects.
Furthermore, we assume a user speciﬁes the relational task by providing a small handful of K task
demonstrations {Di}K
i=1, such that it’s intuitive and efﬁcient to specify a wide diversity of tasks with
minimal engineering effort. A demonstration D consists of point clouds ˆPA and ˆPB (of objects ˆOA
and ˆOB) and relation-satisfying transformation ˆTB.
NDFs for Encoding Single Unknown Object Relations. Prior work on NDFs may be applied to
a simpliﬁed version of this task, where the geometry and state of OA is known. Given that OA is
known, we can initialize a set of query points XA near the task-relevant part of OA and use the query
points to encode the relative pose ˆTB via Equation (3). Thus, a demonstration D is mapped to a
target pose descriptor ˆZ = F( ˆT−1
B |ˆPB) representing the (inverse of the) ﬁnal pose of ˆOB relative
to OA. In practice, pose descriptors from multiple demonstrations {Di}K
i=1 are averaged to obtain
an overall descriptor ˆZ =
1
K
PK
i=1 ˆZi for the whole set, which has important implications in the
version of the task with two unknown objects (see Section 4.1 for further discussion).
Given a novel object instance represented by point cloud PB, we can compute a transformation TB
such that transforming OB by TB satisﬁes the demonstrated relation between OA and OB. This is
achieved by minimizing the L1 distance to the target pose descriptor ˆZ:
T−1
B = argmin
T
∥F(T|PB) −ˆZ∥.
(4)
Intuitively, Equation (4) performs well across different objects due to the fact that NDFs are pretrained
to enable reconstruction across a large dataset of 3D shapes. As a result, shared descriptors are
discovered across different instances in a shape category. In contrast, training a model directly on the
few demonstrations (e.g., for regressing pose TB) would be more susceptible to overﬁtting.
4
Method
We now describe how we apply NDFs to infer relations between pairs of unknown objects. In
Section 4.1, we propose an iterative optimization method for assigning consistent task-relevant
coordinate frames to multiple objects. In Section 4.2, we discuss how we train a neural network on
top of NDF features to model the joint object conﬁguration and reﬁne an inferred transformation.
The system inputs consist of pretrained NDFs fA and fB for each object category, demonstra-
tions {Di}K
i=1 = {(ˆPA, ˆPB, ˆTB)i}K
i=1, and a single labeled 3D coordinate xAB for one of the
demonstrations, indicating approximately where the respective demonstration objects interact.
3

<!-- page 4 -->
(A)
(B)
(D)
(C)
Demonstration: 
Pose descriptor encoding
Inference (Step 1): 
Object A part localization 
Execution: 
Transformed Object B
Inference (Step 2): 
Object B part localization 
Figure 2: Method Overview. (A) A demonstration (ˆPA, ˆPB, ˆTB) of a relation is encoded into a pair of pose
descriptors by randomly sampling a set of query points XA at the origin and transforming it by ˆTXA to be
near the task-relevant interaction point xAB. NDFs fA and fB are then used to obtain descriptors ˆ
ZA and ˆ
ZB
representing coordinate frames near the task-relevant local parts on the objects. (B) Given point cloud PA of
a novel object, NDF fA, and pose descriptor ˆ
ZA, pose TXA of the corresponding coordinate frame on PA is
found. (C). This procedure is then repeated with PB, fB, and ˆ
ZB to ﬁnd pose T−1
B of the relevant parts of PB,
relative to pose TXA found in the ﬁrst inference step. (D) Transforming PB by TB satisﬁes the desired relation.
4.1
Multiple NDFs for Inferring Pairs of Task-Relevant Local Coordinate Frames
Consider a scenario where OA and OB have unknown underlying shapes and conﬁgurations. We
now show how NDFs can be used for inferring a pair of task-relevant local coordinate frames on both
objects and recovering a transformation TB that satisﬁes the relation. The key idea of our approach is
to formulate this problem as a bi-level optimization (illustrated in Figure 2), where we ﬁrst optimize
to ﬁnd a task-relevant portion of OA, and subsequently optimize a relative transform of a local part
of OB with respect to the local region of OA.
We begin with two pretrained NDFs, fA and fB, and query points XA in a canonical pose at
the world frame origin.
We obtain XA by sampling Nq points from a zero-mean Gaussian
and scaling such that XA has scale similar to the salient object parts.
We then use the key-
point xAB to transform XA near the task-relevant features in the demonstration associated with
xAB.
Denote this transformation as ˆTXA.
Finally, we encode world-frame pose ˆTXA into
a descriptor conditioned on ˆPA, as ˆZA = FA( ˆTXA|ˆPA), and relative pose ˆT−1
B
as ˆZB =
FB( ˆT−1
B ˆTXA|ˆPB), conditioned on ˆPB. At test-time, we optimize both the world-frame pose
of the query points TXA and the (inverse of) pose TB relative to the initial pose found in the ﬁrst step:
TXA = argmin
T
∥FA(T|PA) −ˆZA∥
(5)
T−1
B = argmin
T
∥FB(TTXA|PB) −ˆZB∥
(6)
Figure 2 shows an example of this pipeline, where the resulting TB is applied to the point cloud PB
of OB to satisfy the “hanging” relation.
Query points not aligned
Query points aligned
vs.
Demo 1
Demo 2
Figure 3:
Demo alignment.
We align
the query points by minimizing the variance
across the descriptor set before averaging.
Minimizing Descriptor Variance.
In practice, solv-
ing Equations (5) and (6) works better if pose descrip-
tors { ˆZi}K
i=1 from multiple demonstrations are averaged
together to obtain an overall target descriptor
ˆZ
=
1
K
PK
i=1 ˆZi (see Sec. 6.1 and [3]). The reason is that
a single demonstration underspeciﬁes which object parts
are relevant for the task, allowing ˆZ to be sensitive to
object features that are not relevant to the desired relation.
Instead, a set of demonstrations using slightly different
objects (e.g., with different scales) reveals regions near lo-
cal interactions that are shared across the demonstrations,
which helps disambiguate between parts that are critical
vs. irrelevant for the speciﬁed relation.
However, to avoid the pitfalls of averaging across a poten-
tially multimodal or disjoint set, we want descriptors in the set { ˆZi}K
i=1 to be sensitive to nearby
local geometry in a way that is consistent (i.e., unimodal) across the demos. This only occurs
if the query points used to obtain the descriptors are themselves consistently aligned relative to
4

<!-- page 5 -->
each respective object (see Figure 3). Therefore, we need to ﬁnd a transformation ˆTXA,i for each
demonstration Di that transforms the canonical query points XA into a conﬁguration that leads
the descriptors { ˆZi = FA( ˆTXA,i|ˆPA,i)}K
i=1 to be consistent with each other. We address this by
ﬁnding the set of transformations { ˆTXA,i}K
i=1 that minimizes the variance across the descriptor set
{ ˆZi = FA( ˆTXA,i|ˆPA,i)}K
i=1:
min
{ ˆTXA,i}K
i=1
Var({ ˆZi}K
i=1)
subject to ˆZi = FA( ˆTXA,i|ˆPA,i)
for i = 1, ..., K
(7)
where Var(·) denotes the sum of the per-element variance across a set of vectors. We perform this
minimization by applying NDFs in an alternating optimization procedure. Starting with an initial
reference pose (constructed using xAB) placing XA near the task-relevant object parts in one of the
demonstrations, we iteratively apply Equation (5) to obtain a descriptor for each demonstration that
matches the reference. At the outer level, we reﬁt the reference descriptor using the mean of the most
recently obtained individual descriptors, and repeat. More details can be found in the Appendix.
4.2
Capturing Joint Descriptor Alignment through Learned Energy Functions
The method in Section 4.1 proposes to infer a desired relation by sequentially localizing independent
coordinate frames for each object. While this approach is generally effective, small errors can
accumulate and cause slight misalignments that lead to failure in the execution. We thus propose to
learn a neural network which directly captures the joint conﬁguration of OA and OB that satisﬁes
the desired relation, and use this model to reﬁne predictions made by the method in Section 4.1.
Pairwise Energy Functions. We train an Energy-Based Model (EBM) Eθ(·) [6] to parameterize a
learned energy landscape over NDF encodings of relative poses between OA and OB (i.e,. Eθ acts
as a learned analogue for the L1 distance in Section 4.1). The energy function Eθ(·) is trained so
that the ground truth transform of OB with respect to OA is recovered given NDFs fA and fB (note
that f corresponds to descriptor evaluation at single coordinate x while F is deﬁned over sets of
coordinates). Explicitly, our energy function is trained so that:
TB = argmin
T
[Eθ(fB(·|TPB), fA(·|PA))] .
(8)
Since each NDF is a continuous ﬁeld, it is difﬁcult to input them directly into our energy function
Eθ(·). We represent the energy function as the sum of the point-wise evaluation of each NDF on a
set of different query points XE sampled from transformed pointcloud TPB.
Eθ(fB(·|TPB), fA(·|PA)) =
X
x∈XE
Eθ(fB(x|TPB), fA(x|PA))
(9)
At test-time, we use Equation (8) to reﬁne the transformation obtained using Equations (5) and (6).
4.3
Learning
NDF training. We represent NDFs fA and fB as two neural networks with identical architecture and
separate weights. Following [3], the architecture consists of a PointNet [7] point cloud encoder with
SO(3) equivariant Vector Neuron [4] layers, and a multi-layer perceptron (MLP) decoder. The NDF
is represented as a function mapping a 3D coordinate and a point cloud to the vector of concatenated
activations of the MLP. The models are trained end-to-end to reconstruct 3D shapes given object
point clouds. We use a dataset of ground truth 3D shapes and generate a corresponding set of 3D
point clouds in simulation. More architecture and training data details can be found in the Appendix.
Energy-Based Model Training. We supervise the EBM Eθ so that optimization over the learned
energy landscape recovers the relative transform between OA and OB. In particular, we follow
the training objective in [8] and train arg minT[Eθ(·)] to match a target pose using the following
procedure. We ﬁrst apply a small delta perturbation T∆to ˆTB ˆPB (i.e., the point cloud of OB
in its ﬁnal conﬁguration) to obtain ˆPB,∆= T∆ˆTB ˆPB. We then train Eθ to iteratively reﬁne
an initial random pose T0 with translation t0 and rotation R0 to undo the perturbation pose T∆.
We run n steps of optimization on t0 and R0, where an individual step is given by tk = tk−1 −
λ∇tEθ(fA(·|ˆPA), fB(·|TˆPB,∆)) and Rk = Rk−1 −λ∇REθ(fc(·|ˆPA), fB(·|TˆPB,∆)).
We may train the energy function so that Tn corresponds to the inverse of the perturbation pose T∆
using Ltrans = ∥tn −t−1
∆∥and Lrot = ∥Rn −R−1
∆∥. However, with symmetric objects, there are
multiple different rotations Rn which may satisfy the desired relation (e.g., a bowl is still “on” a
5

<!-- page 6 -->
mug, regardless of the angle about its radial axis). To account for these symmetries, we implicitly
enforce consistency between an optimized transform Tn and T−1
∆by enforcing that its application on
ˆPB,∆leads to a similar point cloud to ˆTB ˆPB. We achieve this by minimizing the Chamfer loss [9]
between the optimized transformed point cloud Tn ˆPB,∆and the demonstration point cloud ˆTB ˆPB.
5
Application to Tabletop Manipulation
Robot and Environment Setup. We apply the method in Section 4 to the problem of tabletop object
rearrangement using a Franka Panda robotic arm with a Robotiq 2F140 parallel jaw gripper. The
arm is used to collect the demonstrations and to execute the inferred transformation at test-time. Our
environment consists of the arm on a table with four calibrated depth cameras.
Providing and Encoding Demonstrations. When collecting a demonstration, initial object point
clouds ˆPA and ˆPB of objects ˆOA and ˆOB are obtained by fusing a set of back projected depth
images. The demonstrator moves the gripper to a pose ˆTgrasp, grasps ˆOB, and ﬁnally moves the
gripper to a pose ˆTplace that satisﬁes the desired relation between ˆOA and ˆOB. ˆTB is obtained as
ˆTplace ˆT−1
grasp. In one of the demonstrations, a 3D keypoint xAB is labeled near the parts of the objects
that interact with each other by moving the gripper to this region and recording its position.
Test-time Task Setup and Inference. At test time, we are given point clouds PA and PB of new
objects OA and OB. Equations (5), (6), and (8) are applied in sequence to obtain TB. TB is applied
to OB by transforming an initial grasp pose Tgrasp (obtained using a separate grasp generation
pipeline) by TB to obtain a placing pose Tplace = TBTgrasp, and off the shelf inverse kinematics and
motion planning is used to reach Tgrasp and Tplace.
6
Experiments and Results
Our experiments are designed to evaluate R-NDFs in executing relational rearrangement tasks with
unseen objects using only a few demonstrations. We seek to answer three questions: (1) How well do
R-NDFs predict transformations that satisfy a relational task? (2) How important is each component
in R-NDFs? (3) Can R-NDFs be used to perform multi-object pick-and-place tasks in the real world?
We also show additional results regarding (i) multi-step rearrangement via relation sequencing, (ii)
composing multiple energy terms in the optimization to achieve collision avoidance and multi-object
rearrangement, and (iii) applying R-NDF with partial point clouds in the Appendix.
Baselines. As existing rearrangement methods are not directly applicable with so few demonstrations,
we compare with two constructed baselines. The ﬁrst is to train an MLP to directly regress the
relative transformation between objects (“Pose Regression”). The MLP takes as input the point
cloud encodings obtained from the same PointNet [7] encoder with Vector Neuron [4] layers used in
NDFs, and is trained directly on the demonstrations. The second method is based on 3D point cloud
registration (“Patch Match”). We use a state-of-the-art registration method [10] to align the test-time
shapes to the demonstration shapes and then compute the resulting relative transformation.
Task Setup and Evaluation Metrics. We consider three relational rearrangement tasks for evalu-
ation: (1) Hanging a mug on the hook of a rack, (2) Stacking a bowl upright on top of a mug, and
(3) Placing a bottle upright inside of a box-shaped container. We provide 10 demonstrations of each
task and evaluate if each method, using the demonstrations, can infer a transformation that satisﬁes
the desired relation for unseen pairs of object instances with randomly sampled poses. Experiments
are conducted in both the real world and in simulation using PyBullet [11]. In simulation, the trans-
formation obtained by each method is directly applied by resetting the simulator to the transformed
object states. To quantify performance, we report the success rate over 100 trials, where we use the
ground truth simulator state to compute success (objects must be in contact, have the correct relative
orientation, and not interpenetrate).
6.1
Simulation Results
We begin by evaluating how well R-NDFs can infer the desired transformations in simulation. We
consider two settings of varying difﬁculty. First, the pair of unseen objects are positioned randomly on
6

<!-- page 7 -->
Bowl on Mug
Mug on Rack
Bottle in Container
Method
Upright Arbitrary Upright Arbitrary Upright
Arbitrary
Pose Regression
35.0
6.0
13.0
10.0
37.9
12.0
Patch Match
34.0
32.0
56.0
44.0
44.0
42.0
R-NDF
74.0
70.0
84.0
75.0
80.0
75.0
(a)
R-NDF (ours)
Patch Match 
Pose Regression
Bowl on
Mug
Mug on
Rack
Bottle in
Container
(b)
Figure 4: (a) Relation inference success rates in simulation. R-NDF performs better than the baseline
approaches. (b) Example predictions. Representative predictions made by each method in simulation
Multiple
Query Point
EBM
Upright Arbitrary
Demonstration
Alignment
Reﬁnement
Pose
Pose
No
No
No
39.3
43.6
Yes
No
No
66.0
60.0
Yes
Yes
No
78.0
72.0
Yes
Yes
Yes
84.0
75.0
(a)
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
1.6
Noise Standard Deviation (Normalized By Object Size)
0.00
0.25
0.50
0.75
1.00
Success Rate
Success Rate vs. Interaction Point Estimation Noise
(b)
Figure 5: (a) Ablations. R-NDF performance with different components ablated. Success rate is highest when
using multiple demonstrations, query point alignment, and EBM reﬁnement. (b) Success vs. Keypoint Noise.
Success rate vs. magnitude of noise (normalized by object size) added to the single labeled 3D keypoint xAB.
the table with a randomly sampled “upright” orientation (similar to those used in the demonstrations).
Second, the orientation of OB is randomly sampled from the full space of 3D rotations.
Results in Table 4a compare the performance of our approach to the baselines. On the other hand, the
registration-based method can sometimes ﬁnd transformations that correctly align the unseen shapes
to the demonstration objects, and thus achieves higher success rates than pose regression. However,
3D registration is susceptible to locally optimal results that align the task irrelevant parts of the
objects. Common failure modes of using 3D registration in the tasks we consider include aligning the
body of the mug but ignoring the handle, or aligning the racks to be upside down. Figure 4b illustrates
the ﬁnal simulator state after applying some of the representative predictions of each method.
In contrast, R-NDFs more accurately localize the task-relevant object parts and assign coordinate
frames to these parts that are consistent with the demonstrations, leading to the highest success rates.
Consistent with [3], the performance gap between the “upright” and “arbitrary” pose settings is small,
which can be attributed to the built-in equivariance of the features used in R-NDF.
6.2
Ablations
Next, we analyze the importance of the individual components of R-NDFs. We investigate ablations
on the simulated “mug on rack” task, again considering both “upright” and “arbitrary” pose settings.
The top row of Table 5a illustrates that R-NDF performs worse with a single demonstration. Since
there are multiple possible explanations for the alignment between two objects when given one
example of the desired relation, pose descriptors obtained from a single demonstration are more
sensitive to task-irrelevant object features. The second row of Table 5a investigates the effect of
averaging descriptors across the set of demonstrations without ﬁrst aligning the query points relative
to the objects in each demo. We modiﬁed the demonstrations to provide keypoints {xAB,i}K
i=1 near
the relevant region in each demonstration, and then transform the query points to this region without
aligning their orientations. Removing the query point alignment reduces the performance. The third
row of Table 5a shows that removing the EBM reﬁnement also decreases the success rate.
We further examine the importance of accurately specifying the 3D keypoint xAB near the task-
relevant region on one of the demonstrations. We run the trials multiple times with Gaussian
distributed noise added to the labeled point. Figure 5b shows a plot of the success rate vs. the noise
magnitude normalized by the approximate size of the object. The plot indicates that with limited
noise perturbation, the success rate does not suffer signiﬁcantly, though we observe a steep decline
with more substantial perturbations. These larger perturbations shift the query points to regions near
geometric features that are less relevant to the desired relation.
7

<!-- page 8 -->
Bowl on Mug
Mug on Rack
Bottle in Container
Figure 6: Real Execution Results. Example executions of relational tasks on unseen mugs, bowls, bottles,
racks, and containers in the real world. Our framework enables inferring the relative transformation between
pairs of unseen objects in arbitrary initial poses from a small handful of unaligned demonstrations of each task.
6.3
Real Results
Finally, we validate that R-NDFs can be used to perform pick-and-place on pairs of unseen objects in
the real world. Figure 6 shows the execution on our three tasks. Our method successfully infers a
transformation between the objects that satisﬁes the relations, despite the objects being presented in a
challenging array of initial conﬁgurations. Figure 1 shows a multi-step rearrangement application
of R-NDFs for the “bowl on mug” task. First, a relation between the mug and the table is speciﬁed
and inferred for placing the mug upright. Then, the system executes the “stacking” relation between
the bowl and the upright mug. This highlights how R-NDFs can enable executing sequential chains
of relations to satisfy task objectives involving more than two objects. Please see our attached
supplemental video for additional real-world results.
7
Related Work
Novel Object Rearrangement. Several methods exist for novel object rearrangement [1, 12–29],
many of which don’t consider multiple varying objects that interact. CatBC [30] uses dense correspon-
dence models to achieve impressive pick-and-place policy generalization from a single demonstration
but assumes a known receptacle for placing. Neural shape mating [31], OmniHang [32], and kPAM
2.0 [2] generalize to pairs of unseen objects, but these approaches train on large task-speciﬁc datasets.
TransporterNets [33, 34] enables rearrangement with varying pick and place locations from a few
demonstrations, but focuses on top-down manipulation and struggles with out-of-plane reorientation.
In contrast, we focus on executing relations involving large 3D reorientations.
Neural Fields in Robotics. Neural ﬁelds use neural networks to parameterize functions over contin-
uous spatial or temporal coordinates [35]. They have been applied to model various signals and scene
properties, such as images [36], geometry [5, 37, 38], appearance [39, 40], tactile imprints [41], and
sound [42], with high ﬁdelity and memory efﬁciency. Neural ﬁelds have been applied to represent
objects for manipulation [3, 43–46] and environment states for dynamics and policy learning [47–
49]. They have also been used for pose estimation [50, 51], SLAM [52, 53], and representing object
geometry without depth cameras [54, 55].
8
Limitations and Conclusion
Limitations. R-NDFs require a pretrained NDF for each category used in the task, which can be
nontrivial to obtain for novel object categories without existing 3D model datasets. Our approach
also requires an annotated keypoint to localize task-relevant object parts. Future work could explore
automated discovery of task-relevant regions directly from a set of demonstrations. Our system uses
depth cameras, which often struggle with noise and objects with thin and transparent features. An
RGB-only approach offering a similar level of generalization would be interesting to investigate.
Finally, we require segmented object point clouds. While object instance segmentation is quite
mature, pretrained segmentation models regularly struggle when objects are in diverse orientations.
Conclusion. This work presents an approach for learning from a limited number of demonstrations to
rearrange novel objects into conﬁgurations satisfying a relational task objective. We develop methods
that build upon prior applications of neural ﬁelds for representing objects and increase the scope
of tasks they can achieve. Our results illustrate the general applicability of our framework across a
diverse range of relational tasks involving pairs of novel objects in arbitrary initial poses.
8

<!-- page 9 -->
Acknowledgments
This work is supported by Sony, NSF Institute for AI and Fundamental Interactions, DARPA Machine
Common Sense, NSF grant 2214177, AFOSR grant FA9550-22-1-0249, ONR grant N00014-22-
1-2740, MIT-IBM Watson Lab, MIT Quest for Intelligence. Anthony Simeonov and Yilun Du are
supported in part by NSF Graduate Research Fellowships. We thank members of the Improbable AI
Lab and the Learning and Intelligent Systems Lab for the helpful discussions and feedback.
Author Contributions
Anthony Simeonov developed the idea of minimizing descriptor variance for aligning multiple
demonstrations, set up the simulation and real robot experiments, played a primary role in paper
writing, and led the project.
Yilun Du came up with and implemented the energy-based modeling framework for relative pose
inference, helped develop the overall framework of using NDFs for relational rearrangement tasks,
ran simulated experiments, helped with writing the paper, and co-led the project.
Yen-Chen Lin participated in research discussions about different ways to approach 6-DoF pick-
and-place/rearrangement tasks, helped suggest improvements to the NDF training and optimization
procedure, and helped with editing the paper.
Alberto Rodriguez helped with early brainstorming on how multiple NDF models could be used for
multi-object rearrangement tasks and gave feedback on the tasks and real robot results.
Leslie Kaelbling helped develop the idea of chaining multiple pairwise relations together to perform
multi-step tasks, provided suggestions on interesting rearrangement tasks to solve, and helped write
and edit the paper.
Tom´as Lozano-Per´ez also helped suggest the application to multi-step tasks via sequencing relations,
reinforced the investigation of representations grounded in local interactions between object parts,
and provided valuable feedback on the paper.
Pulkit Agrawal was involved in early technical discussions about how to use multiple NDF models
for rearrangement tasks, helped clarify key technical insights regarding query point labeling in the
demonstrations, advised the overall project, and helped with paper writing and editing.
References
[1] L. Manuelli, W. Gao, P. Florence, and R. Tedrake. kpam: Keypoint affordances for category-
level robotic manipulation. In The International Symposium of Robotics Research, pages
132–157. Springer, 2019. 1, 8
[2] W. Gao and R. Tedrake. kpam 2.0: Feedback control for category-level robotic manipulation.
IEEE Robotics and Automation Letters, 6(2):2962–2969, 2021. 1, 8, 26
[3] A. Simeonov, Y. Du, A. Tagliasacchi, J. B. Tenenbaum, A. Rodriguez, P. Agrawal, and V. Sitz-
mann. Neural descriptor ﬁelds: Se (3)-equivariant object representations for manipulation. In
2022 International Conference on Robotics and Automation (ICRA), pages 6394–6400. IEEE,
2022. 1, 2, 4, 5, 7, 8, 14, 16, 18, 20, 24, 27
[4] C. Deng, O. Litany, Y. Duan, A. Poulenard, A. Tagliasacchi, and L. J. Guibas. Vector neurons:
A general framework for so(3)-equivariant networks. In ICCV, 2021. URL https://arxiv.
org/abs/2104.12229. 2, 5, 6, 16
[5] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger. Occupancy networks:
Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 4460–4470, 2019. 2, 8, 14, 16, 27
[6] Y. Du and I. Mordatch. Implicit generation and modeling with energy based models. In NeurIPS,
2019. 5
[7] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning on point sets for 3d
classiﬁcation and segmentation. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 652–660, 2017. 5, 6, 16
[8] Y. Du, S. Li, Y. Sharma, J. B. Tenenbaum, and I. Mordatch. Unsupervised learning of composi-
tional energy concepts. NeurIPS, 2021. 5
9

<!-- page 10 -->
[9] H. G. Barrow, J. M. Tenenbaum, R. C. Bolles, and H. C. Wolf. Parametric correspondence and
chamfer matching: Two new techniques for image matching. Technical report, SRI International
Menlo Park CA Artiﬁcial Intelligence Center, 1977. 6
[10] W. Gao and R. Tedrake. Filterreg: Robust and efﬁcient probabilistic point-set registration using
gaussian ﬁlter and twist parameterization. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 11095–11104, 2019. 6
[11] E. Coumans and Y. Bai. Pybullet, a python module for physics simulation for games, robotics
and machine learning. GitHub repository, 2016. 6, 15, 17
[12] S. Cheng, K. Mo, and L. Shao. Learning to regrasp by learning to place. In 5th Annual Confer-
ence on Robot Learning, 2021. URL https://openreview.net/forum?id=Qdb1ODTQTnL.
8
[13] R. Li, C. Esteves, A. Makadia, and P. Agrawal. Stable object reorientation using contact plane
registration. In 2022 International Conference on Robotics and Automation (ICRA), pages
6379–6385. IEEE, 2022.
[14] S. Thompson, L. P. Kaelbling, and T. Lozano-Perez. Shape-based transfer of generic skills. In
2021 IEEE International Conference on Robotics and Automation (ICRA), pages 5996–6002.
IEEE, 2021.
[15] D. Batra, A. X. Chang, S. Chernova, A. J. Davison, J. Deng, V. Koltun, S. Levine, J. Malik,
I. Mordatch, R. Mottaghi, et al. Rearrangement: A challenge for embodied ai. arXiv preprint
arXiv:2011.01975, 2020.
[16] A. Simeonov, Y. Du, B. Kim, F. R. Hogan, J. Tenenbaum, P. Agrawal, and A. Rodriguez.
A long horizon planning framework for manipulating rigid pointcloud objects. In Confer-
ence on Robot Learning (CoRL), 2020. URL https://anthonysimeonov.github.io/
rpo-planning-framework/.
[17] S. Lu, R. Wang, Y. Miao, C. Mitash, and K. Bekris. Online object model reconstruction and
reuse for lifelong improvement of robot manipulation. In 2022 International Conference on
Robotics and Automation (ICRA), pages 1540–1546. IEEE, 2022.
[18] M. Gualtieri and R. Platt. Robotic pick-and-place with uncertain object instance segmentation
and shape completion. IEEE robotics and automation letters, 6(2):1753–1760, 2021.
[19] P. Florence, L. Manuelli, and R. Tedrake. Self-supervised correspondence in visuomotor policy
learning. IEEE Robotics and Automation Letters, 2019.
[20] A. Curtis, X. Fang, L. P. Kaelbling, T. Lozano-P´erez, and C. R. Garrett. Long-horizon manipu-
lation of unknown objects via task and motion planning with estimated affordances. In 2022
International Conference on Robotics and Automation (ICRA), pages 1940–1946. IEEE, 2022.
[21] C. Paxton, C. Xie, T. Hermans, and D. Fox. Predicting stable conﬁgurations for semantic
placement of novel objects. In Conference on Robot Learning, pages 806–815. PMLR, 2022.
[22] W. Yuan, C. Paxton, K. Desingh, and D. Fox. Sornet: Spatial object-centric representations for
sequential manipulation. In 5th Annual Conference on Robot Learning, pages 148–157. PMLR,
2021.
[23] A. Goyal, A. Mousavian, C. Paxton, Y.-W. Chao, B. Okorn, J. Deng, and D. Fox.
Ifor:
Iterative ﬂow minimization for robotic object rearrangement. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 14787–14797, 2022.
[24] A. H. Qureshi, A. Mousavian, C. Paxton, M. Yip, and D. Fox. NeRP: Neural Rearrangement
Planning for Unknown Objects. In Proceedings of Robotics: Science and Systems, Virtual, July
2021. doi:10.15607/RSS.2021.XVII.072.
[25] D. Driess, J.-S. Ha, and M. Toussaint. Learning to solve sequential physical reasoning problems
from a scene image. The International Journal of Robotics Research, 40(12-14):1435–1466,
2021.
[26] D. Driess, J.-S. Ha, and M. Toussaint. Deep visual reasoning: Learning to predict action
sequences for task and motion planning from an initial scene image. In Robotics: Science and
Systems 2020 (RSS 2020). RSS Foundation, 2020.
[27] W. Liu, C. Paxton, T. Hermans, and D. Fox. Structformer: Learning spatial structure for
language-guided semantic rearrangement of novel objects. In 2022 International Conference on
Robotics and Automation (ICRA), pages 6322–6329. IEEE, 2022.
[28] W. Goodwin, S. Vaze, I. Havoutis, and I. Posner. Semantically grounded object matching
for robust robotic scene rearrangement. In 2022 International Conference on Robotics and
Automation (ICRA), pages 11138–11144. IEEE, 2022.
10

<!-- page 11 -->
[29] M. Danielczuk, A. Mousavian, C. Eppner, and D. Fox. Object rearrangement using learned
implicit collision functions. In 2021 IEEE International Conference on Robotics and Automation
(ICRA), pages 6010–6017. IEEE, 2021. 8, 25, 27
[30] B. Wen, W. Lian, K. Bekris, and S. Schaal. You Only Demonstrate Once: Category-Level
Manipulation from Single Visual Demonstration. In Proceedings of Robotics: Science and
Systems, New York City, NY, USA, June 2022. doi:10.15607/RSS.2022.XVIII.044. 8
[31] Y.-C. Chen, H. Li, D. Turpin, A. Jacobson, and A. Garg. Neural shape mating: Self-supervised
object assembly with adversarial shape priors. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 12724–12733, 2022. 8
[32] Y. You, L. Shao, T. Migimatsu, and J. Bohg. Omnihang: Learning to hang arbitrary objects
using contact point correspondences and neural collision estimation. In 2021 IEEE International
Conference on Robotics and Automation (ICRA), pages 5921–5927. IEEE, 2021. 8, 25
[33] A. Zeng, P. Florence, J. Tompson, S. Welker, J. Chien, M. Attarian, T. Armstrong, I. Krasin,
D. Duong, V. Sindhwani, and J. Lee. Transporter networks: Rearranging the visual world for
robotic manipulation. Conference on Robot Learning (CoRL), 2020. 8
[34] H. Huang, D. Wang, R. Walters, and R. Platt. Equivariant Transporter Network. In Proceedings
of Robotics: Science and Systems, New York City, NY, USA, June 2022. doi:10.15607/RSS.
2022.XVIII.007. 8
[35] Y. Xie, T. Takikawa, S. Saito, O. Litany, S. Yan, N. Khan, F. Tombari, J. Tompkin, V. Sitzmann,
and S. Sridhar. Neural ﬁelds in visual computing and beyond. Computer Graphics Forum, 2022.
ISSN 1467-8659. doi:10.1111/cgf.14505. 8, 27
[36] T. Karras, M. Aittala, S. Laine, E. H¨ark¨onen, J. Hellsten, J. Lehtinen, and T. Aila. Alias-free
generative adversarial networks. NeurIPS, 34, 2021. 8
[37] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove. Deepsdf: Learning continuous
signed distance functions for shape representation. In Proc. CVPR, 2019. 8, 14, 27
[38] Z. Chen and H. Zhang. Learning implicit ﬁelds for generative shape modeling. In Proc. CVPR,
pages 5939–5948, 2019. 8
[39] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf:
Representing scenes as neural radiance ﬁelds for view synthesis. In Proc. ECCV, 2020. 8
[40] V. Sitzmann, M. Zollh¨ofer, and G. Wetzstein. Scene representation networks: Continuous
3d-structure-aware neural scene representations. In NeurIPS, 2019. 8
[41] R. Gao, Y.-Y. Chang, S. Mall, L. Fei-Fei, and J. Wu. Objectfolder: A dataset of objects with
implicit visual, auditory, and tactile representations. In CoRL, 2021. 8
[42] A. Luo, Y. Du, M. J. Tarr, J. B. Tenenbaum, A. Torralba, and C. Gan. Learning neural acoustic
ﬁelds. arXiv preprint arXiv:2204.00628, 2022. 8
[43] J.-S. Ha, D. Driess, and M. Toussaint. Deep visual constraints: Neural implicit models for
manipulation planning from visual input. IEEE Robotics and Automation Letters, 7(4):10857–
10864, 2022. 8
[44] Y. Wi, P. Florence, A. Zeng, and N. Fazeli. Virdo: Visio-tactile implicit representations of
deformable objects. arXiv preprint arXiv:2202.00868, 2022.
[45] Z. Jiang, Y. Zhu, M. Svetlik, K. Fang, and Y. Zhu. Synergies between affordance and geometry:
6-dof grasp detection via implicit representations. Robotics: science and systems, 2021.
[46] Z. Jiang, C.-C. Hsu, and Y. Zhu. Ditto: Building digital twins of articulated objects from
interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5616–5626, 2022. 8
[47] Y. Li, S. Li, V. Sitzmann, P. Agrawal, and A. Torralba. 3d neural scene representations for
visuomotor control. In CoRL, 2021. 8
[48] D. Driess, Z. Huang, Y. Li, R. Tedrake, and M. Toussaint. Learning multi-object dynamics with
compositional neural radiance ﬁelds. arXiv preprint arXiv:2202.11855, 2022.
[49] D. Driess, I. Schubert, P. Florence, Y. Li, and M. Toussaint. Reinforcement learning with neural
radiance ﬁelds. arXiv preprint arXiv:2206.01634, 2022. 8
[50] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-Y. Lin. iNeRF: Inverting
neural radiance ﬁelds for pose estimation. In IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), 2021. 8
[51] M. Adamkiewicz, T. Chen, A. Caccavale, R. Gardner, P. Culbertson, J. Bohg, and M. Schwager.
Vision-only robot navigation in a neural radiance world. In RA-L, 2022. 8
11

<!-- page 12 -->
[52] A. Moreau, N. Piasco, D. Tsishkou, B. Stanciulescu, and A. de La Fortelle. Lens: Localization
enhanced by nerf synthesis. In Conference on Robot Learning, 2021. 8
[53] E. Sucar, S. Liu, J. Ortiz, and A. Davison. iMAP: Implicit mapping and positioning in real-time.
In ICCV, 2021. 8
[54] L. Yen-Chen, P. Florence, J. T. Barron, T.-Y. Lin, A. Rodriguez, and P. Isola. NeRF-Supervision:
Learning dense object descriptors from neural radiance ﬁelds. In ICRA, 2022. 8, 26
[55] J. Ichnowski*, Y. Avigal*, J. Kerr, and K. Goldberg. Dex-NeRF: Using a neural radiance ﬁeld
to grasp transparent objects. In Conference on Robot Learning (CoRL), 2021. 8
[56] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva,
S. Song, H. Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint
arXiv:1512.03012, 2015. 14, 15
[57] D. P. Kingma and J. Ba.
Adam: A method for stochastic optimization.
arXiv preprint
arXiv:1412.6980, 2014. 16, 17
[58] J. Sola, J. Deray, and D. Atchuthan. A micro lie theory for state estimation in robotics. arXiv
preprint arXiv:1812.01537, 2018. 17
[59] T. Chen, A. Simeonov, and P. Agrawal. AIRobot. https://github.com/Improbable-AI/
airobot, 2019. 17
[60] M. Ester, H.-P. Kriegel, J. Sander, X. Xu, et al. A density-based algorithm for discovering
clusters in large spatial databases with noise. In kdd, volume 96, pages 226–231, 1996. 18
[61] A. Ganapathi, P. Florence, J. Varley, K. Burns, K. Goldberg, and A. Zeng. Implicit kinematic
policies: Unifying joint and cartesian action spaces in end-to-end robot learning. arXiv, 2022.
22
[62] J. Urain, P. Liu, A. Li, C. D’Eramo, and J. Peters. Composable Energy Policies for Reactive
Motion Generation and Reinforcement Learning . In Proceedings of Robotics: Science and
Systems, Virtual, July 2021. doi:10.15607/RSS.2021.XVII.052. 22
[63] Y. Du, S. Li, and I. Mordatch. Compositional visual generation with energy based models. In
Advances in Neural Information Processing Systems, 2020. 22
[64] Y. Du, S. Li, Y. Sharma, B. J. Tenenbaum, and I. Mordatch. Unsupervised learning of composi-
tional energy concepts. In Advances in Neural Information Processing Systems, 2021.
[65] N. Liu, S. Li, Y. Du, A. Torralba, and J. B. Tenenbaum. Compositional visual generation with
composable diffusion models. ECCV, 2022.
[66] N. Liu, S. Li, Y. Du, J. Tenenbaum, and A. Torralba. Learning to compose visual relations.
Advances in Neural Information Processing Systems, 34, 2021. 22
[67] W. Gao and R. Tedrake. kpam-sc: Generalizable manipulation planning using keypoint af-
fordance and shape completion. In 2021 IEEE International Conference on Robotics and
Automation (ICRA), pages 6527–6533, 2021. doi:10.1109/ICRA48506.2021.9561428. 26
[68] P. R. Florence, L. Manuelli, and R. Tedrake. Dense object nets: Learning dense visual object
descriptors by and for robotic manipulation. In Conference on Robot Learning, pages 373–385.
PMLR, 2018.
[69] S. Duggal and D. Pathak. Topologically-aware deformation ﬁelds for single-view 3d reconstruc-
tion. CVPR, 2022. 27
[70] Y. Deng, J. Yang, and X. Tong. Deformed implicit ﬁeld: Modeling 3d shapes with learned
dense correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 10286–10296, 2021. 26, 27
[71] S. Peng, M. Niemeyer, L. Mescheder, M. Pollefeys, and A. Geiger. Convolutional occupancy
networks. In Proc. ECCV, 2020. 27
[72] K. He, G. Gkioxari, P. Doll´ar, and R. Girshick. Mask r-cnn. In Proceedings of the IEEE
international conference on computer vision, pages 2961–2969, 2017. 27
[73] Y. Xiang, C. Xie, A. Mousavian, and D. Fox. Learning rgb-d feature embeddings for unseen
object instance segmentation. In Conference on Robot Learning, pages 461–470. PMLR, 2021.
[74] S. Back, J. Lee, T. Kim, S. Noh, R. Kang, S. Bak, and K. Lee. Unseen object amodal instance
segmentation via hierarchical occlusion modeling. In 2022 International Conference on Robotics
and Automation (ICRA), pages 5085–5092. IEEE, 2022.
[75] C. Xie, Y. Xiang, A. Mousavian, and D. Fox. Unseen object instance segmentation for robotic
environments. IEEE Transactions on Robotics, 37(5):1343–1359, 2021. 27
[76] A. Mousavian, C. Eppner, and D. Fox. 6-dof graspnet: Variational grasp generation for object
manipulation. In Proceedings of the IEEE International Conference on Computer Vision, pages
12

<!-- page 13 -->
2901–2910, 2019. 27
[77] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox. Contact-graspnet: Efﬁcient 6-dof
grasp generation in cluttered scenes. In 2021 IEEE International Conference on Robotics and
Automation (ICRA), pages 13438–13444. IEEE, 2021. 27
[78] H. Ryu, J.-H. Lee, H.-i. Lee, and J. Choi. Equivariant descriptor ﬁelds: Se (3)-equivariant
energy-based models for end-to-end visual robotic manipulation learning. arXiv preprint
arXiv:2206.08321, 2022. 27
[79] E. Chatzipantazis, S. Pertigkiozoglou, E. Dobriban, and K. Daniilidis. Se (3)-equivariant
attention networks for shape reconstruction in function space. arXiv preprint arXiv:2204.02394,
2022.
[80] Y. Chen, B. Fernando, H. Bilen, M. Nießner, and E. Gavves. 3d equivariant graph implicit
functions. arXiv preprint arXiv:2203.17178, 2022. 27
[81] C. Jiang, A. Sud, A. Makadia, J. Huang, M. Nießner, and T. Funkhouser. Local implicit grid
representations for 3d scenes. In Proc. CVPR, pages 6001–6010, 2020. 27
[82] J. Chibane, T. Alldieck, and G. Pons-Moll. Implicit functions in feature space for 3d shape
reconstruction and completion. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 6970–6981, 2020.
[83] R. Chabra, J. E. Lenssen, E. Ilg, T. Schmidt, J. Straub, S. Lovegrove, and R. Newcombe. Deep
local shapes: Learning local sdf priors for detailed 3d reconstruction. In European Conference
on Computer Vision, pages 608–625. Springer, 2020. 27
13

<!-- page 14 -->
SE(3)-Equivariant Relational Rearrangement with Neural
Descriptor Fields: Supplementary Material
In Section A1, we present details on data generation, model architecture, and training for NDFs. In
Section A2 we detail the optimization method used to recover the pose of a local coordinate frame
by minimizing descriptor distances (as in Equations (4), (5), and (6)). Section A3 describes the
procedure for training the energy-based models used in relative transformation reﬁnement. In Section
A4, we describe more details about our experimental setup, Section A5 discusses more details on the
evaluation tasks and robot execution pipelines, and Section A6 describes our alternating minimization
method for aligning descriptors across a set of demonstrations. In section A7 we present an additional
set of qualitative results showing how R-NDF can be used to handle collision avoidance and additional
problem constraints and more complex rearrangement tasks, and in Section A8 we discuss applying
R-NDF to relational rearrangement with more than two objects. Section A9 shows an example of the
framework operating with partial point clouds and Section A10 contains additional visualizations
of the tasks and objects used in the evaluation. Finally, in Section A11, we provide more thorough
implementation details and an expansion on the limitations of the proposed approach.
A1
NDF Training
In this section, we present details on the data used for training NDFs, the neural network architectures
we used in the NDF implementation, and model training.
A1.1
Training Data Generation
3D shape data for training NDFs. NDFs are trained to perform category-level 3D reconstruction
from point cloud inputs. We supervise this training using ground truth 3D shape data obtained
from synthetic 3D object models. The three tasks we consider include objects from ﬁve categories:
mugs, bowls, bottles, racks, and containers. Our NDF training thus begins with obtaining a dataset
of varying 3D models for a diverse set of object instances from each of these categories. We use
ShapeNet [56] for the mugs, bowls, and bottles, and procedurally generate our own dataset of .obj
ﬁles for the racks and containers. See Figure A1 for representative samples of the 3D models from
each category.
NDFs based on regressing occupancy vs. signed-distance. Given an object dataset of 3D models,
we generate a dataset of inputs and outputs for training the neural networks used in NDFs. While in [3]
the underlying NDF decoder is trained to perform 3D reconstruction by representing an occupancy
ﬁeld [5], i.e., as an Occupancy Network (ONet) that predicts whether a point is inside or outside of a
shape, we ﬁnd performance improves by instead training the model to regress a signed-distance ﬁeld
(SDF) [37], i.e., as a DeepSDF that predicts the minimum distance from a point to the boundary of
a shape and assigns negative/positive values for points inside/outside of the shape. Details on our
pipeline for generating the data used to train the SDF decoder can be found in the next subsection.
The following paragraphs discuss two reasons we hypothesize for the performance gap between
occupancy ﬁeld and signed-distance ﬁeld training.
First, a signed-distance ﬁeld (whose zero level set represents the boundary of a 3D shape) contains
information about the underlying object geometry even at query points that are far away from the
surface of the object, whereas the underlying occupancy ﬁeld is ﬂat everywhere except exactly at the
crossing between the inside and outside of the shape. This feature of an SDF increases the likelihood
of unique descriptors at different coordinates (e.g., f(xi|P) and f(xj|P)). These factors appear
to shape the optimization landscape in Equations (4), (5), and (6) to enable smoother and more
consistent convergence, along with less sensitivity to poor initialization.
Second, in addition to an (empirically observed) improvement in global optimization convergence,
we also observe the SDF-based decoder leads to the optimization performing much better near the
surface of the shape. The intuition is that an occupancy ﬁeld has a sharp discontinuity at the boundary
of the shape. When optimizing a query point set pose using Equation (4)-(6), the gradients become
steep when the optimization reaches the region near the input point cloud. Some of the points in
X end up inside the shape and get stuck. In contrast, although it may still have high-frequency
ﬂuctuations near the shape boundary, an SDF varies much less rapidly near the surface, and we
14

<!-- page 15 -->
ShapeNet Mugs
ShapeNet Bottles
ShapeNet Bowls
Synthetic Racks
Synthetic Containers
Figure A1: Example 3D models used to train NDFs and deploy NDFs on our rearrangement tasks. Mugs,
bottles, and bowls are from ShapeNet [56] while we procedurally generated our own synthetic racks and
box-shaped containers.
observe a corresponding reduction in local minima when running the NDF optimization using the
SDF model.
Data generation. Based on the empirical observations discussed above, we convert each 3D model
dataset into a corresponding dataset of input/output pairs for signed-distance function regression.
For each shape, we normalize the object to a unit bounding box and use the 3D model to generate
the distance from M query points to the boundary of the shape, where points inside and outside
the shape are labeled with negative and positive sign, respectively. We use an open-source tool† for
generating the ground truth signed distance. We computed signed-distance values for M = 200, 000
query points per shape, where the query points are sampled inside a unit sphere and biased toward
being near the surface of the shape (using about M/2 points near the shape boundary).
We also need a point cloud of each shape to provide as input during training. To generate these point
clouds, we initialize the objects on a table in PyBullet [11] in random positions and orientations,
and render depth images with the object segmented from the background using multiple simulated
cameras. These depth maps are converted to 3D point clouds and fused into the world coordinate
frame using known camera poses. To obtain a diverse set of point clouds, we randomize the number
†https://github.com/marian42/shapegan and https://github.com/marian42/mesh_to_sdf
15

<!-- page 16 -->
of cameras (1-4), camera viewing angles, distances between the cameras and objects, object scales,
and object poses. Rendering point clouds in this way allows the model to see some of the occlusion
patterns that occur when the objects are in different orientations and cannot be viewed from below
the table.
A1.2
Architecture
Point cloud encoder. We follow the encoder/decoder architecture proposed in Vector Neurons [4] for
rotation equivariant occupancy networks [5], and replace the output occupancy probability prediction
with a signed distance regression.
The encoder E follows from the SO(3)-equivariant PointNet [7] model proposed in [4]. E takes
P ∈R3×N as input and outputs a matrix latent representation Z ∈R3×C (i.e., to which applying a
rotation R ∈SO(3) is a meaningful operation). The composition of layer operations in E is rotation
equivariant (see [4] for details on how this property is achieved). As a result, given a point cloud P
and rotation R, the following relationship holds by construction:
E(RP) = RE(P) = RZ.
(10)
Implicit function decoder. The decoder Φ consists of an MLP with residual connections that also
contains Vector Neuron layers:
Φ(x, E(P)) : R3 × R3×C →R
(11)
The decoder takes in a combination of features, computed using the point cloud embedding Z and a
3D query point x, that is designed to make the output prediction rotation invariant, i.e., if Z and x
have been rotated together, the distance prediction should not change. Speciﬁcally, following [4], the
decoder predicts the signed distance s ∈R from 3D coordinate x to the shape as:
s(x, Z) = ResNet(

⟨x, Z⟩, ||x||2, VN-In(Z)

)
(12)
where VN-In(Z) is rotation invariant, obtained as VN-In(Z) := VT Z, where V is the output of
another rotation equivariant function of Z‡.
Building descriptors from MLP activations. Following prior work [3], we deﬁne an NDF as a
function mapping an input coordinate and point cloud to the vector of concatenated activations of the
decoder Φ:
f(x|P) =
L
M
i=1
Φi(x, E(P))
(13)
where L denotes the total number of layers in Φ, Φi denotes the output activation of the ith layer, and
L denotes the concatenation operator.
A1.3
Training Details
Here we discuss details on training the NDF models using the data and model architecture described
in the sections above. Training samples consist of inputs (point cloud Ptrain ∈RNtrain×3, with
Ntrain = 1000) and a set of query points Xtrain ∈RNq,train×3, with Nq,train = 1500) and ground
truth outputs (distance between each point in Xtrain and the underlying shape of the object represented
by Ptrain). Based on the corresponding object scale and pose used to generate the respective point
cloud in simulation, we compute ground truth distances by transforming and scaling the distances
relative to the normalized object shapes. We train E and Φ end-to-end to minimize the mean-squared
error between the predicted and ground-truth distance values across the query points.
We generated a dataset of approximately 100,000 samples for each category and trained one NDF
model per category using each respective dataset. We trained the models for 150 thousand iterations
on a single NVIDIA 3090 GPU with a batch size of 16 and a learning rate of 1e-4, which takes about
half a day. We train the models using the Adam [57] optimizer.
‡Following the explanation from [4], this is because the product of one rotation equivariant feature Z ∈R3×C
with the transpose of another V ∈R3×C′ is rotation invariant: (RV)T (RZ) = VT RT RZ = VT Z. See Sec.
3.5 of [4] for more details
16

<!-- page 17 -->
A2
NDF Coordinate Frame Localization via Descriptor Distance
Minimization
This section describes how the optimization in Equation (4) is performed to recover a pose, relative
to an unseen shape, that matches a demonstration pose which has been encoded into a target pose
descriptor.
The inputs to the problem are the target pose descriptor ˆZ, NDF f, point cloud P, and query points
X (in their canonical pose). We randomly initialize an SE(3) pose T0 as an axis-angle 3D rotation
R0
aa ∈R3 and a 3D translation t0 ∈R3. We then perform Niter iterations of gradient descent to
update this pose.
On iteration j, we begin by constructing a pose Tj ∈SE(3) by converting Rj
aa into rotation matrix
Rj ∈SO(3) using the SO(3) exponential map [58] and combining with the translation tj. We then
obtain pose descriptor Zj by transforming the query points X by Tj and applying Equation (3)
using f. Finally, we compute the loss L using the L1 distance between ˆZ and Zj and backpropagate
gradients of this loss to update the rotation and translation:
Rj+1
aa
←Rj
aa −λ∇RaaL
(14)
tj+1 ←tj −λ∇tL
(15)
We use the Adam [57] optimizer with a learning rate λ of 1e-2 to run this procedure. The optimization
is run for a ﬁxed number Niter = 650 iterations to optimize T. We empirically observed this to be
enough iterations to allow the solution to converge. Furthermore, since the loss landscape for this
problem is non-convex, the solution is somewhat sensitive to the initialization. To help obtain some
diversity in the solutions, we run the optimization multiple times in parallel from different initial
values for the pose. We used a batch size of 10, which uses about 10GB of GPU memory when using
Nq = 500 query points and a point cloud downsampled to N = 1500 points.
A3
EBM Training
In this section, we present the training details of utilizing an EBM to reﬁne predictions of relative
transformation between objects.
Training Data. To train EBMs to capture each relation, we utilize the 10 demonstrations provided
for each task. To prevent overﬁtting, and to construct more diverse data to train EBM models, we
heavily data augment point clouds in each demonstration. In particular, we skew point clouds, apply
per-point Gaussian noise, and simulate different occlusion patterns on the demonstration point clouds.
Model Architecture. We utilize a three-layer MLP (described in Table 3) to parameterize an EBM
operating over the concatenation of descriptors at each point. We utilize a swish activation in the
EBM to enable fully continuous gradients with respect to inputs.
Training Details. When training EBMs, we corrupt ˆTB ˆPB by a transformation corresponding
translation sampled from [−0.05, 0.05] along each dimension and rotation perturbation of 15 degrees
along yaw, pitch and roll. We utilize a gradient descent step size of 10 for translation and 20
for rotation during optimization in training, and run 8 steps of optimization. Optimized rotations
are represented as Euler angles, as the perturbations of individual rotation components are small.
Each EBM is evaluated pointwise across 1000 points (with descriptors with respect to each object
concatenated). EBMs are trained with a batch size of 16.
Computational Resources. To train each model, we utilize a single Volta 32GB machine for 6 hours
and train models for 12000 iterations.
A4
Experimental Setup
This section describes the details of our experimental setup in simulation and the real world.
A4.1
Simulated Experimental Setup
We use PyBullet [11] and the AIRobot [59] library to setup the tasks in the simulation, provide
demonstrations, and perform quantitative evaluation experiments. The simulation environment
contains a Franka Panda arm with a Robotiq 2F140 gripper attached, a table, and a set of simulated
17

<!-- page 18 -->
RGB-D cameras. We obtain segmentation masks using the built-in segmentation abilities of the
simulated cameras to separate object point clouds from the overall scene.
A4.2
Real World Experimental Setup
Our real-world environment also contains a Franka Robot arm with a Robotiq 2F140 parallel jaw
gripper. We also use four Realsense D415 RGB-D cameras, with extrinsics calibrated relative to
the robot’s base frame. We use a combination of point cloud cropping and Euclidean clustering
to segment object point clouds from the scene, identify their class identities, and ﬁlter out noise.
Speciﬁcally, we crop the overall scene to the known region above the table, and then crop OA and
OB based on which side of the table each object is on (assumed to be known for this experiment, just
for the purposes of obtaining the segmentation). Finally, we run DBSCAN [60] clustering to remove
outliers and noise to obtain the ﬁnal point clouds PA and PB. To demonstrate R-NDF on objects in
diverse initial orientations, we present some of the objects on a 3D-printed stand with angled support
surfaces. When using the stand, we remove it from the point cloud by estimating its pose using 3D
registration and the known CAD model.
A5
Evaluation Details
This section presents further details on the three tasks we used in our experiments and notes on the
automatic detection methods used to obtain success rates over multiple simulated trials.
A5.1
Tasks and Evaluation Criteria
Task Descriptions. We consider three relational rearrangement tasks for evaluation: (1) Hanging
a mug on the hook of a rack, (2) Stacking a bowl upright on top of a mug, and (3) Placing a
bottle upright inside of a box-shaped container. For “Hanging a mug on a rack”, we ensure the
mug is oriented consistently relative to the single peg of the rack. This is achieved by providing
demonstrations in which the handle always points left relative to a front-facing peg, and the opening
of the mug is always points toward the top of the rack. Similarly for stacking bowls on mugs and
putting bottles in containers, we provide demonstrations in which the “stacked” object is always
upright, though in these tasks, the orientation about the radial axis of the bowls/bottles doesn’t affect
the relation result and is thus ignored. All objects are presented in an initial “upright” pose on the
table for the demonstrations.
Evaluation Metrics and Success Criteria. To quantify performance, we report the success rate
over 100 trials, where we use the ground truth simulator state to compute success. For a trial to be
successful, objects OA and OB must be in contact, OB must have the correct orientation relative to
OA, and OA and OB must not interpenetrate.
A5.2
Providing Demonstrations
Here we re-describe the procedure for obtaining task demonstrations using a robot arm, a set of depth
cameras, and a parallel jaw gripper, as outlined in Section 5.
When collecting a demonstration, initial object point clouds ˆPA and ˆPB of objects ˆOA and ˆOB
are obtained by fusing a set of back projected depth images. The demonstrator moves the gripper
to a pose ˆTgrasp, grasps ˆOB, and ﬁnally moves the gripper to a pose ˆTplace that satisﬁes the desired
relation between ˆOA and ˆOB. ˆTB is obtained as ˆTplace ˆT−1
grasp. In one of the demonstrations, a 3D
keypoint xAB is labeled near the parts of the objects that interact with each other by moving the
gripper to this region and recording its position.
Tuning the scale of the query point set XA. As mentioned in Section 4.1, XA is obtained by
sampling from a zero-mean Gaussian, with a variance that must be chosen by the user. This variance
will impact the scale of the query point set, and can be thought of as a hyperparameter. The original
NDF paper discusses the implications of different query point sizes (see Table III in [3]). We therefore
did not repeat the ablations performed in [3] showing that the scale of the query point cloud must
be tuned based on the rough scale of the shapes in the object set. We instead tuned the variance
parameter and settled on values that led to good task performance. For real-world objects, the value
we used typically falls between 0.015 and 0.025. The heuristic used in [3] to simplify this tuning
procedure was to use a bounding box around the whole shape OA.
18

<!-- page 19 -->
A5.3
Baselines
Pose Regression. The Pose Regression baseline method consists of an MLP trained to predict TB
using the concatenated embeddings of PA and PB obtained from a pretrained VNN encoder (the
one used for NDFs). We supervise transform prediction ˆTB using the Chamfer distance between
a transformed point cloud ˆTBPB and a ground truth point cloud TBPB. Such a loss captures the
symmetry in PB. A transformation is represented using vector of six dimensions, where the ﬁrst three
dimensions correspond to translation and the subsequent dimensions correspond to an axis-angle
parameterization of rotations The architecture for pose regression is provided in Table 4.
Patch Match. The Patch match baseline uses 3D registration to align the test-time point clouds
PA and PB to the point clouds of the corresponding shapes used in one of the demonstrations,
and then using the demonstrated pose ˆ
TB together with these registration results to compute the
resulting pose TB. Formally, for demonstration Di, registering source PA to target ˆPA produces
SE(3) transformation TA,reg, and similarly for PB and ˆPB to obtain TB,reg. TB is then obtained as
T−1
A,reg ˆTB.
A5.4
Automatic success detection and common failure modes
This section discusses the methods we used for checking each success criteria in the simulator, along
with some of the common failure modes for each method.
Correct Orienttion. We compare the angle between the radial axis of the bowls/bottles and the
positve z-axis in the world to check if the ﬁnal orientation is correct for the “bowl on mug” and
“bottle in container” tasks. We count the objects as ending in a valid “upright” orientation if this angle
difference is below 15 degrees once the physics has been turned on and the object has settled. This is
important because a common failure mode for the “bottle in container” task is to localize the top of
the bottle instead of the bottom and try to place it upside down on the container. This occurs for both
our method and the baselines. For “bowl on mug”, this failure mode is much less apparent for our
method but still occurs quite often with the baselines.
We did not explicitly check if the orientation is correct for the “mug on rack” task, as this would
require comparing the angle between the axis pointing along the cylindrical body of the mug and the
axis pointing along the rack’s peg to ensure the mug opening points in the right direction relative to
the rack. Since the pegs on the racks are all slightly different, obtaining this ground truth peg-axis
angle was too cumbersome to manually set up. We instead relied on the “OA and OB in contact”
criteria to imply the correct relative conﬁguration between the mug and the rack. This is an effective
method because for many incorrect relative orientations, the mug misses the rack and falls when the
physics are turned on, leading to a ﬁnal conﬁguration where the objects do not touch. However, there
may still be solutions found that satisfy the “hanging” criteria by passing this check but not being in
the intended orientation. We thus allow any ﬁnal conﬁgurations satisfying “hanging” (implied by
the mug and rack ending up “in contact” and “not interpenetrating”), where the mug opening points
down instead of up, to be counted as a success. We observe this happens much more frequently for
the “Patch Match” baseline than R-NDF, as “Patch Match” struggles in disambiguating between an
upright and an upside-down mug during 3D registration.
In-Contact vs. Not-Interpenetrating. Since we reset OA and OB in the simulator to their ﬁnal
conﬁguration after predicting the relative transformation, the objects can end up in physically-
/geometrically-infeasible poses that intersect, and we don’t want to count these conﬁgurations as
successful for any of the tasks. As described above, we use the “in-contact” criterion to implicitly
determine if the “mug on rack” relation is satisﬁed based on the objects’ relative orientation. Even
for “bottle in container” and “bowl on mug”, where we explicitly check to ensure OB ends in an
upright orientation, we might obtain a prediction that places OB too low relative to OA and causes an
infeasible intersection. Therefore, we can obtain many false positives if we don’t take care to ensure
objects that interpenetrate are not counted as being successfully “in-contact”. However, automatically
disambiguating the “in contact” criteria with the “not interpenetrating” criteria, both of which are
required for success, turns out to be slightly nontrivial. Here we discuss the method we used to check
these criteria in a disentangled fashion.
We ﬁrst check that OA and OB are in contact after transforming OB by TB (“in contact”) and
allowing the physics simulation to proceed for 2 seconds,. We then ensure the objects are not only
in contact because they are interpenetrating by checking whether or not OB can be easily removed
19

<!-- page 20 -->
from its ﬁnal conﬁguration under dynamic physical effects. To achieve this, we transform OA and
OB to maintain their relative conﬁguration, but make OA turn upside down in the world frame.
For each of our tasks, if the objects are not in interpenetration, OB should fall away from OA,
whereas we observe that they regularly get stuck together in a physically implausible way if they
are in interpenetration. We thus check whether or not the objects are still in contact after turning
them upside down and waiting while the physics simulation proceeds, and label the pair as “not
interpenetrating” if they are not in contact after this delay. The most common failure mode for
R-NDFs on the “bottle in container” and “bowl on mug” tasks is to predict transformations that nearly
satisfy the relation but cause the objects to interpenetrate (e.g., the bottle/bowl is too low, and thus
intersects with the container/mug).
A5.5
Task Execution
This section describes additional details about the pipelines used for executing the inferred relations
in simulation and the real world.
Simulated Execution Pipeline. The evaluation pipeline mirrors the demonstration setup. Objects
from the 3D model dataset for the respective categories are loaded into the scene with randomly
sampled position and orientation. In the “upright” pose case, a known upright orientation is obtained,
and then adjusted with a random top-down yaw angle. In the “arbitrary” pose case, we sample a
rotation matrix uniformly from SO(3), load the object with this orientation, and constrain the object
in the world frame to be ﬁxed in this orientation. We do not allow it to fall on the table under gravity,
as this would bias the distribution of orientations covered to be those that are stable on a horizontal
surface, whereas we want to evaluate the ability of each method to generalize over all of SO(3). In
both cases, we randomly sample a position on/above the table that are in view for the simulated
cameras. We also load the target pose descriptor ˆZ, obtained by following the procedure described in
Section 4, for use in inference.
After loading objects OA and OB and the target pose descriptor ˆZ, we obtain segmented point
clouds PA and PB and apply Equations (5), (6), and (8) to obtain a transformation TB. The output
transformation is applied to OB by transforming its initial pose TB,start by TB and resetting the
state of the object in the simulation to the resulting pose TB,ﬁnal = TBTB,start. Task success is then
checked based on the criteria described in the section above.
Real World Execution Pipeline. Here, we repeat the description of how we execute the inferred
transformation using a robot arm with additional details.
At test time, we are given point clouds PA and PB of new objects OA and OB, each with potentially
new shapes and poses, and Equations (5), (6), and (8) are applied in sequence to obtain TB. We
ﬁrst obtain an initial grasp pose Tgrasp. Our implementation uses NDF to generate these grasp
poses, following the pipeline described in [3] and Section 3 (where OA is the robot’s gripper),
but any generic grasp generation pipeline could be used instead. We then obtain a placing pose
Tplace = TBTgrasp, and plan a collision-free path between the grasp and place pose using MoveIt!§.
The path is executed by following the joint trajectory in position control mode and opening/closing
the ﬁngers at the correct respective steps. The whole pipeline can be run multiple times in case the
planner returns infeasibility, as the inference methods for both grasp and placement generation can
produce somewhat different solutions depending on how the NDF optimization is initialized.
Pipeline for Executing Multiple Relations in Sequence. Figure 1 shows a multi-step rearrangement
application of R-NDFs for the “bowl on mug” task, where a “mug upright on the table” relation is
executed before the “bowl upright on the mug” relation. This section describes the setup for chaining
these relations and executing them in sequence (see also Subsection A8 below for further discussion).
To specify the “mug upright on table” component of the task, we follow [3] and the steps described
in Section 3 on “NDFs for Encoding Single Unknown Object Relations”, where the table is the
known object OA. Using this prior knowledge, we initialize a set of query points near a known
placing region on the table, and use these points to obtain the target pose descriptor from a set of
demonstrations (i.e., demos of placing mugs upright on the table near the query point set location).
The “bowl upright on mug” part of the task is then encoded using the method described in Section 4.
During execution, both inference steps are run using the initial point clouds PA and PB, i.e., we
don’t re-observe the mug after executing the upright placement on the table. Thus, we ﬁnd T1
B which
§https://moveit.ros.org/
20

<!-- page 21 -->
transforms the mug relative to the table, and T2
B, which transforms the bowl relative to the mug in
its initial conﬁguration. Finally, we execute relative transformations T1
B,exec = T1
B to the mug and
T2
B,exec = T1
BT2
B to the bowl, using the pick-and-place operation described in the section above.
A6
Alternating Minimization to Obtain an Average Pose Descriptor From
Multiple Unaligned Demonstrations
This section describes details and further intuition behind our method for aligning descriptors across
a set of demonstrations, as proposed in Section 4.
Assuming a speciﬁed interaction point xAB in demonstration Di, we ﬁrst construct ˆT0
XA,i. We then
transform the canonical query points XA by ˆT0
XA,i to the region near the task-relevant features on
the object and obtain ˆZ0
ref = ˆZ0
i with Equation (3). Equation (5) is then used with ˆZ0
ref to solve
for a corresponding transformation ˆTXA,j and a resulting pose descriptor ˆZ0
j = FA( ˆTXA,j|ˆPA,j)
for the remaining demonstrations {Dj}K
j=1,j̸=i. After running this for each demonstration, we
compute an average pose descriptor ˆZ1
ref =
1
K
PK
i=1 ˆZ0
j . We then apply the same procedure to each
demonstration (including the demo used for providing the initial reference pose) again, now using
ˆZ1
ref as the target descriptor. We repeat this process Q times, where Q is a hyperparameter (we used
Q = 3 throughout the experiments).
Intuitively, this procedure starts with a target descriptor obtained from a single demonstration
(whichever demonstration Di corresponds to the one where xAB was provided). As shown in the top
row of Table 5a, some fraction of the poses obtained by matching a single demonstration correctly
match a target descriptor (∼40% success). This means that some of the individual descriptors in
{ ˆZk
j }K
j=1,j̸=i found on the kth iteration correctly align with the reference, and are somewhat near
each other in descriptor space. The mean of this set ˆZk+1
ref thus provides a new descriptor that is both
(on average) more similar to each of the descriptors than was the original reference, and still similar
to the original descriptor that was obtained using the manually speciﬁed keypoint. By resetting the
target using this mean, the next alignment round uses a target that is more sensitive to the part features
that are shared among some of the demonstrations, and makes it more likely that a consistent pose
will be found for more of the demonstrations than the previous iteration. The similarity among the
resulting descriptors continues to increase accordingly. Overall, multiple rounds of this procedure
lead to poses for each respective demonstration which are consistent with each other and map to
descriptors with relatively high similarity.
By encouraging the individual descriptors in the set to converge to values that are similar to the mean
among the whole set, this iterative procedure allows the ﬁnal target descriptor to capture the shared
parts of the demonstration objects in a consistent fashion.
A7
Modifying the R-NDF Optimization Objective for Collision Avoidance
This section shows how the optimization objective used for recovering coordinate frame poses can be
modiﬁed to take collision avoidance into account.
Our framework models rearrangement tasks in the form of SE(3)-transformations of rigid bodies. The
method described in Section 4 does not address other constraints like collision avoidance and robot
kinematics, and our real-world task executions depend on a separate motion planning module, since
our approach does not produce full paths/trajectories. However, because we perform optimization for
pose localization, it is straightforward to incorporate extra cost terms that capture additional factors
like collision avoidance.
In Figure A2, we highlight this ability with an example of grasping a mug while avoiding collisions.
The top row optimizes the standard descriptor matching objective and doesn’t consider nearby
obstacles. In contrast, the bottom row shows the result when the optimizer minimizes a combined
energy, composed of a descriptor-matching cost and a collision avoidance cost. The grasp pose
found in the top row collides with the nearby obstacle, since it only tries to match the target pose
descriptor. On the other hand, in the bottom row, the energy landscape for the optimization takes
21

<!-- page 22 -->
(A) Without collision 
avoidance term
(B) With collision 
avoidance term
(i) Top-down view of 
energy landscape
(ii) Optimized grasp 
pose (isometric view)
(iii) Optimized grasp 
pose (top-down view)
High
Low
High
Low
Figure A2: Energy composition for collision avoidance and descriptor matching. (Top) NDF grasp pose
inference without collision avoidance cost. We recorded a grasp pose along the rim of a demo mug and
recovered the corresponding pose on a new mug using NDF optimization. This procedure ignores collision
avoidance with nearby obstacles, and thus ﬁnds a solution that would be infeasible due to interpenetration with
the box-shaped obstacle next to the mug. (Bottom) NDF grasp pose inference with collision avoidance cost.
We modify the NDF optimization with an additional cost term. This extra term penalizes query points that
fall inside the obstacle. The resulting energy landscape takes both the collision avoidance and the descriptor
matching into account. The optimizer ﬁnds a new solution that stays outside of the obstacle, while still achieving
a grasp along the rim.
the nearby obstacle into account, such that a different point along the rim of the mug achieves the
new optimum. The new optimum still satisﬁes the desired grasp along the rim but also achieves the
secondary constraint of keeping the gripper outside of the box-shaped obstacle.
This highlights the ability to incorporate additional task constraints together with the pose-matching
objective, using collision avoidance as one such example. The same principle can apply to other task
constraints. For instance, recent work has shown that energy-based modeling is useful for taking robot
kinematics into account by setting up joint space decision variables and using a differentiable forward
kinematics module [61]. In this way, a user can add a similar additional cost term that penalizes
solutions that cause self-collisions and violate joint limits. Other recent work has set up similar
compositional energy optimization approaches for trajectory synthesis and motion generation [62].
In machine learning, energy-based models have been useful for compositional generative modeling
and reasoning [63–66].
A8
Extending Beyond Two-Object Rearrangement
This section expands on how R-NDF can be used for rearrangement tasks involving more than two
objects, via both sequential and compositional optimization.
Our approach models relational rearrangement via pair-wise relations, which facilitates a natural way
to go beyond two-object rearrangement and handle tasks involving more objects. Here we discuss
two approaches for achieving this with R-NDF. While we did not use the EBM for these examples,
the same type of compositionality and sequential inference is also directly applicable to using the
learned model, which also operates on pairs of object point clouds.
22

<!-- page 23 -->
(a) Demo
(b) Test
(c) Complete 3-body rearrangement result 
Initial mug
Final mug
Initial bottle
Final bottle
Step 1: Localize container
(with respect to world)
Step 2: Localize mug 
(with respect to container)
Step 3: Localize bottle 
(with respect to mug)
(d) Sequential coordinate frame localization
Figure A3: Three-object rearrangement by sequentially localizing multiple task-relevant coordinate
frames. (a) Demonstration of “mug in container” and “bottle in mug”. (b) Initial conﬁguration of test-time
container, bottle, and mug. (c) Final conﬁguration satisfying the desired set of relations among the three objects
after inferring task-relevant coordinate frames on each of them. (d) Independent coordinate frame localization
steps executed in sequence. First, the bottom of the container is found in the world frame. Then, the bottom of
the sideways blue mug is found. Using the frame found in Step 1., the initial mug (dark blue) can be transformed
to the ﬁnal upright mug (pink). Finally, the bottom of the initially sideways bottle (teal) can be found, and
transformed to the ﬁnal bottle (green) inside of the mug. Composing each of these steps together provides the
ﬁnal conﬁguration shown in (c).
A8.1
Three-body rearrangement via sequential optimization
If a set of multi-object relations forms a tree structure, each coordinate frame can be found indepen-
dently and then composed in sequence to satisfy the overall task. This example is highlighted in
Figure 1 (where the table can be considered a third static object). For additional clarity, Figure A3
shows another example of this behavior. The task is to place the “mug upright in the container and
the bottle in the mug”. At test time, the new container, mug, and bottle are all in different initial
poses. First, the container is localized. Then, the mug is localized relative to the container, and ﬁnally,
the bottle is localized relative to the mug. These relative transformations are composed to obtain the
overall transformation applied to each object in execution.
A8.2
Three-body rearrangement via energy composition
Solving for task-relevant coordinate frame poses using optimization makes it straightforward to
incorporate additional costs. We can apply this fact to the setting of three-object rearrangement, as
shown in Figure A4. In particular, instead of matching descriptors with respect to one object, we can
solve for a pose that attempts to match descriptors with respect to two objects simultaneously.
Here, the task is to place a mug upright both “next to the bowl” and “next to the bottle” (see
Figure A4, top left). After localizing a coordinate frame on the bottom of a mug, we run the NDF
optimization of the query points to recover a coordinate frame that satisﬁes both of the “next to”
relations simultaneously. This is achieved by computing the overall optimization loss as the sum of
the two separate “bowl” and “bottle” descriptor matching losses. Note that due to the radial symmetry
of the bowl and the bottle, if we only optimize the mug pose with respect to one object, the solution
can end up anywhere along a circle surrounding the respective object. However, when we optimize
with respect to both objects, the solver ﬁnds a unique solution that satisﬁes both “next to” objectives,
placing the mug in a location that is unambiguously between the bowl and the bottle.
23

<!-- page 24 -->
(a) Demo of task: “mug next to 
bottle” AND “mug next to bowl”
(b) Query points relative to each shape in demo
(c) Inference with bottle only
(d) Inference with bowl only
(e) Inference with bowl AND bottle
Initial mug
Final mug
Figure A4: Three-object rearrangement by composing multiple NDF descriptor distances. (a) Demon-
stration showing “mug next to bottle” and ‘mug next to bowl”. (b) Using a single set of query points at the
bottom of the demonstration mug, we obtain a set of descriptors for each shape, relative to the point cloud of
each shape shown in the demonstrations. (c) At test-time, we ﬁrst localize the world frame pose of the bottom of
the mug. If we then optimize the ﬁnal mug pose by matching the bottle NDF descriptors alone, a solution far
away from the bowl is found. This is because the bottle has a radial symmetry which allows multiple solutions
for descriptor matching. (d) Similarly, when ﬁnding the mug pose relative to the bowl on its own, the mug ends
up at a position away from the bottle. (e) By optimizing the mug pose relative to both the bowl NDF and the
bottle NDF simultaneously, the resulting solution is “next to” both the bowl and and the bottle.
A9
Can R-NDF work with partial point clouds?
In this section, we provide evidence that R-NDF can work with partial point clouds obtained from a
single viewpoint, and discuss the difﬁculties of handling the problem of rearrangement with heavy
occlusions in its full generality.
Similar to [3], we obtained point clouds from multiple cameras at different viewing angles to ensure
the point cloud was relatively complete. The purpose of this design choice was to focus the effort
on expanding the rearrangement task capabilities, rather than handling scenes with large occlusions.
However, we have observed the model’s ability to deal with partial point clouds, so long as (1) they
are included in the training distribution, and (2) the important part of the object for matching (i.e.,
the handle, or the peg, for the mug/rack task) is not entirely hidden from view. Figure A5 shows an
example of successful coordinate frame localization on point clouds obtained from a single camera.
Dealing with the partial point cloud issue in its entirety requires addressing the possibility that the
task-relevant part might be completely hidden from view. This difﬁcult scenario greatly complicates
matters. Techniques related to active perception, wherein the system can search for new viewpoints,
are perhaps the fundamentally correct way to deal with this problem. While this was beyond the
scope of our work, note that as these complementary methods improve, our proposed method will
become applicable to more general occlusion settings.
A10
Miscellaneous Visualizations
In this section, we show additional visualizations of the tasks used in our simulation experiments.
Figure A6 shows more snapshots of the ﬁnal simulator state for each of the tasks used in the
quantitative evaluation.
24

<!-- page 25 -->
(a) Initial scene 
viewed from one camera
(b) Partial point clouds 
resulting from self-occlusion
(c) Localized task-relevant 
coordinate frames
(d) Final state satisfying
“mug on rack”
Figure A5: NDFs can work with partial point clouds. (a) Initial scene for “mug on rack” task viewed from
only a single camera. (b) Alternate view of the resulting point clouds for each object, showing large missing
regions of each point cloud. (c) Task-relevant coordinate frames found on each object using the partial point
clouds. This shows that, as long as the occlusions aren’t too severe, NDF descriptor matching can still work for
recovering corresponding coordinate frames that match a demonstration. (d) Final simulator state after applying
the recovered relative transformation.
A11
Expanded Details on System Engineering, Implementation, and
Limitations
This section provides more details on our assumptions, how the overall system is engineered, and
the resulting implications and limitations. Section 8 discusses some of these considerations, and we
provide a more thorough treatment here.
A11.1
Real World Systems Engineering and Implementation
Motion planning and collision avoidance in real-world experiments. This work models relational
rearrangement in the form of relative SE(3) transformations. We focus on evaluating whether or not
the inferred transformation achieves a desired relation between pairs of unseen objects. In simulation,
we can directly evaluate this core capability by bypassing the physics and resetting the object state to
its predicted goal conﬁguration. However, in the real world, we must execute a feasible path to the
goal with the object held by the robot’s end-effector.
Naive trajectory generation (e.g., via linear interpolation in task or joint space) regularly fails to
achieve this because the robot or the grasped object collides with part of the environment. For
example, hanging the mug on the rack requires a particular path to be followed in the last few inches
to avoid moving the rack (e.g., by the mug colliding with the peg). Solving this motion planning and
collision avoidance issue in its full generality would require performing full collision-free planning
of the arm with the grasped object, which remains a challenging task in itself when dealing with
raw point clouds. Several recent works [29, 32] have dedicated speciﬁc effort to solving just this
problem, highlighting that it remains an outstanding challenge without simple off-the-shelf solutions.
As this was not the focus of our work, we used domain knowledge of the task and extra supervision
to simplify path planning. Our approach was to create multiple Cartesian waypoints for the arm to
reach along its path to the goal, each of which has a high likelihood of being collision-free (e.g., at a
position high above the center of the table).
Additional “offset” waypoint pose in demonstrations. We also annotated an extra “offset waypoint”
in the demonstrations, and used this extra waypoint annotation to solve for an offset relative to the
inferred placement pose at test-time. This offset pose could be reached more easily without using
ﬁne-grained collision detection. Once at this offset pose, the ﬁnal placement pose could be achieved
by moving the end-effector in a straight-line task-space path. For example, in the “mug on rack” task,
this was a pose where the mug’s handle was aligned with the rack’s peg, but at a position just in front
of the tip of the peg. We performed this offset pose annotation in the demonstrations by recording an
additional end-effector pose before moving the gripper to the ﬁnal placement pose ( ˆTplace). We then
solved for the relative transformation between this offset pose and the placement pose. Finally, at
test-time, we solved for the world-frame offset pose using this same relative transformation and the
recovered placing pose.
25

<!-- page 26 -->
Mug on rack 
examples
Bowl on mug
examples
Bottle in container
examples
Figure A6: Additional visualizations of unseen objects for evaluation tasks. Snapshot of the simulator ﬁnal
state for each evaluation task. Simulator state is reached after transforming OB into its relation-satisfying
conﬁguration using R-NDF. Each image shows a successful execution of the task.
A11.2
Limiting Assumptions, Resulting Implications, and Avenues for Future Work
A11.2.1
Modeling assumption: We model rearrangement tasks as SE(3)-transformation(s) of rigid bod-
ies. R-NDF does not deal directly with other constraints like collision avoidance and kinemat-
ics, and depends on a separate motion planner for path planning. As discussed in the subsection
above, full collision-free planning was not the focus of this work, and we therefore used heuristic
solutions, domain knowledge, and extra real-world supervision for the purpose of simplifying path
planning to avoid issues with additional constraints. We also describe in Section A7 how to add
extra costs that take more problem constraints into account. Finally, note that the design choice of
predicting relative transformations representing goal conﬁgurations allows our system to generalize
much more easily and efﬁciently than it would have if we tried to learn full trajectory generators or
closed-loop policies.
A11.2.2
Input/Task assumption: R-NDF performs category-level manipulation with known categories,
and depends on the availability of an ofﬂine dataset of 3D shapes from each category. Assuming
a known category does limit the ability to directly apply our method to brand new objects in
unseen categories. However, this assumption is also what helps support generalization to new shape
instances, as the model learns to associate the way different shapes are similar to each other. Several
“category-level” manipulation systems and dense correspondence models have been proposed in
the past [2, 54, 67–70], each with the reasoning that it’s useful to enable programming a speciﬁc
robot skill, or learning a category-speciﬁc concept, that works across all instances from a category.
26

<!-- page 27 -->
Obtaining a module that ignores global features of the object and focuses on local parts that may be
shared across categories is an exciting direction for future work.
Depending on an ofﬂine set of 3D objects is another limitation. Many of the advancements in 3D
vision and graphics have also used ground truth shapes for training [5, 37, 71], but it would be ideal
if we could obtain a system with similar properties that does not depend on the availability of ground
truth 3D shapes. We leave this for future work to address.
A11.2.3
Input assumption: R-NDFs use instance-segmented point clouds with known identity (i.e., the
system knows which point cloud corresponds to OA and OB). As the focus of this work is to
relax the “static secondary/environmental object” assumption from the original NDF work, and
expand the scope of achievable rearrangement tasks, we directly inherited the assumption that point
clouds have been accurately segmented from the background. Note that signiﬁcant progress has
been made on instance/semantic segmentation [72–75] with other robotic systems deploying these as
components to their overall pipeline [29, 76]. Based on this trend, it’s not unreasonable to assume
access to an off-the-shelf module that provides segmented point clouds. However, it’s a fair concern
that today, these systems are not robust enough to be entirely “plug and play” without substantial
engineering effort. This is especially true in new environments that cause a distribution shift from the
data they were trained on, which can negatively affect the downstream task performance.
Changing the underlying NDF formulation to reduce the dependence on performant off-the-shelf
perception modules also has the potential to expand the system’s capabilities. An analogous pro-
gression in 6-DoF grasp generation has occurred and shown meaningful improvements in overall
system capability (i.e,. see [76] and [77] where [77] does not require segmentation). However, this
improvement is roughly orthogonal to our proposed approach.
Recent works on using different neural network encoders for 3D data that use local features have
shown promising results in not requiring accurate segmentation [78–80]. The implications of global
vs. local encoding on generalization and robustness to diverse geometries have been discussed
at length in various recent works on neural ﬁelds [35, 71, 81–83]. In principle, transferring such
approaches to our setting within the proposed R-NDF framework ought to provide similar beneﬁts in
the multi-object rearrangement tasks we consider.
A11.2.4
General limitation: We use NDF as the core component of the framework. If the descriptors
learned in NDF pretraining don’t work well, then the downstream R-NDF framework also
won’t work well. What if they fail on more difﬁcult shapes? It’s a concern that if the underlying
NDF models have not learned meaningful descriptors that encode correspondence in a correct/useful
way, our approach will inherit these problems and suffer in performance. This could potentially occur
on more difﬁcult shape categories or with more difﬁcult tests of generalization.
Recent approaches in neural implicit modeling include components for learning deformation ﬁelds
with explicit correction terms [70] and latent spaces with higher-dimension [69] to recover cross-
instance correspondence for topologically varying shapes. These ideas could potentially be incorpo-
rated into future versions of the NDF to support improved performance more challenging objects.
Investigating such changes to the underlying NDF setup was beyond the scope of this work, but it’s
an important consideration for scaling the approach to enable more difﬁcult object categories.
A11.2.5
General limitation: This work only shows results in empty scenes with minimal clutter, no
distractor objects, and multiple cameras to help retrieve a relatively complete point cloud. We
separated out the issue of handling point clouds with heavy occlusions from the goal of relaxing
the “ﬁxed placement object” assumption from [3]. However, as discussed in Section A9 and shown
in Figure A5, the NDFs can work with partial point clouds under some speciﬁc settings. First,
similar occlusion patterns should be included in the training distribution for 3D reconstruction
pretraining. Second, the task-relevant part of the object that was used for obtaining the target pose
descriptor should not be entirely out of view. Handling partial point clouds with full generality may
27

<!-- page 28 -->
require incorporating active perception that searches in camera pose space, and we decided to avoid
complicating the system by factoring in this set of considerations.
New approaches to setting up the underlying NDF model (i.e., with point cloud encoders that
operate on more local features) also have the potential to improve upon this issue. See the above
paragraph regarding the quality of the underlying NDF models for further commentary on the potential
implications of global vs. local feature encoders.
A11.2.6
Modeling assumption: We model relational rearrangement via pair-wise relations. Can this
be extended to general N-body rearrangement tasks? R-NDF can be applied to rearrangement
tasks with more than two objects. Depending on the nature of the multi-object task speciﬁcation,
there are different approaches available, including sequential localization and composing multiple
NDF descriptor distance terms in the optimization objective. See Section A8 for further discussion
and examples for handling rearrangement with more than two objects.
A11.2.7
Task limitation: We only show pick-and-place results. Can NDFs be used for tasks other than
pick-and-place? NDFs can be used whenever it’s useful to localize a coordinate frame near a
task-speciﬁc local part of an object. Inferring SE(3) transformations for pick-and-place with rigid
objects is one instantiation of this. However, there maybe other settings where this is useful. For
example, in multi-ﬁnger manipulation, it may be useful to encode the pose of each ﬁngertip relative
to an object with a per-ﬁnger query point set and corresponding ﬁngertip-pose descriptor.
Another point to consider is that other tasks of interest to the community may potentially be recast as
a form of pick-and-place. For instance, tool use consists of grasping an object, and then using some
distal part of the grasped object to interact with the external environment. The geometric part of this
“distal object part / environment” interaction could be solved in a way that is directly analogous to our
“placing pose” inference. However, one fundamental limitation is that NDFs are a purely geometric
representation. It would be interesting to see how to bring in dynamic/material/physical properties
like mass, stiffness, friction, etc. and solve more dynamic tasks.
A11.2.8
Implications of these assumptions and limitations: There are many moving parts to the overall
R-NDF system and this creates multiple potential sources of error. Which considerations im-
pact the performance the most? There can be multiple sources of error in NDF descriptor matching.
A few common error sources, roughly in order of their impact on performance, include:
• Out-of-distribution/severe point cloud noise. This can come from depth sensor noise and bad
segmentation. For example, some of the real-world racks we experimented with were quite
thin, shiny, and reﬂective, causing the depth image to have holes. The background/table was
also sometimes not cleanly removed from the point cloud, which left regions containing
outliers that looked different from the data the PointNet encoder was trained on. A similar
issue can occur if the cameras are not accurately calibrated relative to each other.
• Inaccuracies in the demonstrations. For example, in the real world, if the object moves while
in the gripper, or the placement object is accidentally bumped, the ﬁnal point clouds we
record might be in a different location than the true real-world objects.
• NDFs that pick up on task-irrelevant features. For example, a common failure mode for
hanging the mug on the rack is to rotate the mug to almost exactly the correct orientation,
but use the wrong rotation about the cylindrical axis. In this case, the NDF matches the
demonstrations via a the side and the opening, rather than the handle. Similarly, a somewhat
common failure mode for “bottle in container” is to place the bottle upside down, which can
occur when the top and the bottom both have a locally planar geometry.
What are the most important steps to help mitigate these errors?. To complement the discussion
on multiple sources of error above, we list a set of key factors to pay particular attention to for
reducing the likelihood of errors occurring:
28

<!-- page 29 -->
• As discussed above, out-of-distribution point clouds are a common source of error that can
cause failure in descriptor matching. The best remedy for this is to train the NDF on diverse
point clouds that cover the distribution that is likely to be encountered at test time. This can
come from diverse random object scaling, point cloud masking, camera viewpoints, and
other forms of 3D data augmentation.
• To complement the above point, making sure that the underlying implicit representation can
ﬁt the training data well for 3D reconstruction is important to ensure that the descriptors
encode something meaningful. A useful debugging step when descriptor matching has errors
is to examine the reconstruction and make sure it looks reasonable.
• We have found performance to be most consistent when using a handful (∼10) of demon-
strations, where the demonstrations themselves are somewhat diverse. They can be diverse
in the shapes that are used and the pose of the objects in each demonstration. Using more
demonstrations provides a better opportunity for the average target descriptor to accurately
capture the task-relevant geometry, based on what is most saliently shared across the set of
diverse demos. The issue of using a single demonstration (or a set of demonstrations that
fail to disambiguate the task-relevant object parts) is discussed in Section 4.1 in the paper.
• With the current version of the framework, accurate segmentation, and point cloud outlier
removal are important. We notice a meaningful improvement when taking serious care to
completely remove everything in the background/nearby environment, and leave just the
object in the point cloud.
• It is important to run the NDF optimization from multiple diverse initializations. Because
the optimization objective is non-convex, there exist many local minima. We have somewhat
reduced the effects of this by moving toward a DeepSDF-based NDF (rather than the original
OccNet-based NDF, see Section A1), but successfully obtaining a global optimum is still
subject to running pose optimization from multiple initial values.
• Real-world execution requires collision-free motion planning to work properly. The whole
execution is prone to fail if the placement object is accidentally bumped by the grasped
object during placing. Providing intermediate waypoints to reach during the path execution
that are conservatively far away from any potential collisions increases the likelihood of
success. We also assume the object doesn’t move in the grasp while it’s moving to the
placing pose. More robustness could be achieved by tracking the motion of the object in the
grasp to take any unanticipated motion into account and adjust the ﬁnal placing pose.
29

<!-- page 30 -->
A12
Model Architecture Diagrams
VN-LinearLeakyReLU 128
Linear 128
VN-ResnetBlock FC 128
VN-ResnetBlock FC 128
VN-ResnetBlock FC 128
VN-ResnetBlock FC 128
VN-ResnetBlock FC 128
Global Mean Pooling
VN-LinearLeakyReLU →256
Table 1: Vector Neuron point cloud encoder
architecture
ResnetBlock FC 128
ResnetBlock FC 128
ResnetBlock FC 128
ResnetBlock FC 128
ResnetBlock FC 128
Linear →1
Table 2: Signed-distance function decoder
architecture
Dense →128
Dense →128
Dense →6
Table 3: Architecture of EBM capturing rela-
tions.
Dense →256
Dense →256
Dense →6
Table 4: Architecture of pose regression base-
line.
30
