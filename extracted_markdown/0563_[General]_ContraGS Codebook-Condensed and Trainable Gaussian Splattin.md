<!-- page 1 -->
ContraGS: Codebook-Condensed and Trainable Gaussian Splatting for Fast,
Memory-Efficient Reconstruction
Sankeerth Durvasula1†, Sharanshangar Muhunthan1, Zain Moustafa1, Richard Chen1, Ruofan Liang1
Yushi Guan1,
Nilesh Ahuja2,
Nilesh Jain2, Selvakumar Panneer2, Nandita Vijaykumar1
1University of Toronto
2Intel
{sankeerth,zain,ruofan,guanyushi,nandita}@cs.toronto.edu
{shangar.muhunthan,riixardo.chen}@mail.utoronto.ca
{nilesh.ahuja,nilesh.jain,selvakumar.panneer}@intel.com
Figure 1. ContraGS: A 3D scene is modeled using 3D Gaussian Splatting [25]. A codebook stores a compressed representation of the
3DGS scene as common vectors of 3DG parameters. ContraGS enables learning parameters of a codebook-compressed 3DG scene by
posing the parameter estimation as a Bayesian inference problem. ContraGS proposes split/merge/update on each codebook vector to
explore the state space of codebook-compressed models.
Abstract
3D Gaussian Splatting (3DGS) is a state-of-art tech-
nique to model real-world scenes with high quality and
real-time rendering. Typically, a higher quality represen-
tation can be achieved by using a large number of 3D
Gaussians. However, using large 3D Gaussian counts sig-
nificantly increases the GPU device memory for storing
model parameters.
A large model thus requires power-
ful GPUs with high memory capacities for training and
has slower training/rendering latencies due to the ineffi-
ciencies of memory access and data movement.
In this
work, we introduce ContraGS, a method to enable train-
ing directly on compressed 3DGS representations without
reducing the Gaussian Counts, and thus with a little loss
in model quality. ContraGS leverages codebooks to com-
pactly store a set of Gaussian parameter vectors throughout
the training process, thereby significantly reducing mem-
ory consumption.
While codebooks have been demon-
strated to be highly effective at compressing fully trained
3DGS models, directly training using codebook representa-
tions is an unsolved challenge. ContraGS solves the prob-
lem of learning non-differentiable parameters in codebook-
compressed representations by posing parameter estimation
as a Bayesian inference problem. To this end, ContraGS
provides a framework that effectively uses MCMC sampling
to sample over a posterior distribution of these compressed
representations. With ContraGS, we demonstrate that Con-
traGS significantly reduces the peak memory during train-
ing (on average 3.49×) and accelerated training and ren-
dering (1.36× and 1.88× on average, respectively), while
retraining close to state-of-art quality.
1. Introduction
3D Gaussian Splatting [25] (3DGS) is a state-of-art tech-
nique for modeling real-world scenes with high quality and
arXiv:2509.03775v1  [cs.GR]  3 Sep 2025

<!-- page 2 -->
Figure 2. Impact of number of Gaussians on representation quality
for the “Truck” scene of T&T [27], using 3DGS-MCMC [26]
real-time rendering. It is a powerful scene representation
method applicable to various tasks, including 3D/4D recon-
struction [13, 34, 40, 56, 64, 67, 68], simultaneous localiza-
tion and mapping (SLAM) [21, 22, 60–62], 3D object gen-
eration [14, 23, 53, 71] 3D editing [3, 4, 45, 49, 57, 63, 65]
simulation [1, 32, 58, 59], representing detailed human
avatars [24, 28, 42, 43, 48, 51, 52, 66] , etc. 3DGS rep-
resents scenes with a collection of 3D Gaussians associ-
ated with a view-dependent emitted radiance (color). The
parameters defining the position, shape and color of each
Gaussian are obtained by minimizing a photometric loss us-
ing gradient descent.
Learning very high quality scene representations with
3DGS, especially for complex scenes, typically requires
a large number of 3D Gaussians.
Fig. 2 shows the re-
construction quality (PSNR) vs. the number of Gaussians
used to represent the “Truck” scene [27], using the state-
of-art approach 3DGS-MCMC [26]. 2 million Gaussians
achieves a PSNR of 27.64, whereas 4 million Gaussians
achieves a PSNR of 28.15. Using more Gaussians, how-
ever, incurs much larger memory overheads during both
training/inference and for storage. For example, represent-
ing the Truck scene [27] with Gaussians requires over 2
GB of memory for the model alone. Efficiently learning
scenes with high Gaussian counts thus necessitates a pow-
erful GPU with large memory capacity. The peak memory
needed for training can be as much as 4× the model size in
order to save intermediate values during rendering and gra-
dient computation. In addition to requiring GPUs with large
memory capacities, this also significantly slows down the
training process due to the large data movement overheads.
Thus, training complex scenes with 3DGS is challenging on
environments such as mobile devices, web browsers, and
other resource-constrained platforms.
A range of recent works [10, 18, 31, 47, 55] propose
post-training compression methods to reduce the amount of
memory required to store a 3DGS model. However, these
works still incur the full memory cost during training. Other
prior works [15, 17, 36, 38] reduce model size during train-
ing by pruning redundant Gaussians. However, this results
in lower quality models than those that can be achieved with
a larger number of Gaussians (Section 5.2).
In this work, we aim to enable high-speed memory-efficient
training and rendering without sacrificing the quality of a
3DGS scene with high Gaussian counts. To this end, we
leverage codebook representations [7, 10, 47] that can sig-
nificantly reduce the memory required to store 3DGS pa-
rameters.
Codebooks reduce the model’s memory foot-
print by storing a small set of vectors containing parameters
shared among Gaussians, instead of storing an independent
set of parameters for each Gaussian. Each Gaussian maps to
one vector in the codebook. Codebooks have been demon-
strated to be highly effective at compressing fully trained
3DGS models [7, 10, 47] as a post-training optimization.
However, leveraging codebook compression during train-
ing is an unsolved challenge. The parameters in the code-
book compressed representation are (a) the codebook vec-
tors and (b) the mapping between Gaussians and their code-
book vectors. These non-differentiable indexes that map
Gaussians to codebook vectors cannot be learnt by SGD.
In this work, we solve the problem of directly learning
on codebook-compressed representations by using MCMC
sampling to sample over a posterior distribution of these
compressed representations. Posing the parameter estima-
tion as Bayesian inference over a posterior distribution [26],
instead of an optimization problem, offers the opportunity
to estimate the non-differentiable compressed-3DGS pa-
rameters.
However, using Stochastic Gradient Langevin
Dynamics (SGLD), a method for Bayesian inference as pro-
posed by prior work, 3DGS-MCMC [26] will not work
for codebook-compressed representations. This is because
SGLD also requires the differentiability of the log-posterior
distribution with respect to estimated parameters.
We propose ContraGS, a method for learning codebook-
compressed representations via Bayesian inference. We use
the Metropolis-Hastings algorithm, a Markov Chain Monte
Carlo sampling method, to sample from the posterior distri-
bution. An overview of ContraGS is shown in Fig. 1. Start-
ing from an initialized set of mappings between the Gaus-
sians and the codebook, ContraGS defines a proposal dis-
tribution that generates new candidate compressed 3DGS-
models with a different set of Gaussian-to-Codebook map-
pings. This new set of models is obtained by either split-
ting/cloning a set of codebook vectors into two vectors or
merging pairs of codebook vectors into one vector. Gaus-
sians are remapped to the new codebook vectors generated
by splitting/merging. Different sequences of splitting and
merging operations will generate all possible codebook-
compressed model representations. Some steps may simply
update the values of the codebook vectors without affecting
the mapping. Thus, proposals drawn from this distribution
enable exploration of the complete state space of codebook-
compressed models.
We
demonstrate
that
ContraGS
enables
high-speed

<!-- page 3 -->
memory-efficient training without sacrificing the model
quality significantly.
With 2 million Gaussians used to
represent the scene, ContraGS reduces the peak model
memory by 3.49× on average during training.
This
accelerates training by 1.36× on average and increases the
final model’s rendering FPS by 1.88× on average.
Our
contributions can be listed as follows:
• This is the first proposed method that enables 3DGS train-
ing directly on compressed representations.
• We propose a novel mathematical framework to jointly
learn codebook vectors and their mappings to model pa-
rameters, achieving efficient compression during training.
We formulate codebook compression as an MCMC sam-
pling problem over a posterior distribution. We use this
approach to learn non-differentiable codebook indices by
exploring the state space of possible codebook mappings
during MCMC sampling.
This framework can be ap-
plied to learn other codebook-compressed point-based
3D reconstruction methods to enable fast, memory effi-
cient training such as RadiantFoam [16], ADOP [50], De-
formable Beta Splatting [35] and LinPrim [54].
• We show how adopting a Bayesian inference formulation
over a posterior allows incentivising smaller codebooks
sizes for higher compression.
• We demonstrate that ContraGS significantly reduces peak
memory during training while retaining close to state-of-
art quality. For any memory capacity constraint (i.e, peak
memory utilization), ContraGS achieves the highest rep-
resentation quality compared to prior works.
2. Background
2.1. 3DGS Scene Representation
Point-based scene representation techniques,
such as
3DGS [25], represent 3D scenes with a collection of
anisotropic 3D Gaussian densities in space. The ith Gaus-
sian distribution Gi is given by:
Gi(r) = oiexp

−1
2(r −µi)T Σ−1
i (r −µi)

(1)
Where r is a 3D position, oi is the opacity, µi is the location
of its center, and Σi is the covariance. Σi is expressed as:
Σ = R(q)SST R(q)T
(2)
Where R(q) represents the rotation matrix described by the
quaternion q, and S is a diagonal matrix corresponding to
principal axis scales. Rendering the scene for a camera pose
P involves determining the color C of each pixel x, the
scene comprising Gaussians can be rendered using the fol-
lowing equations:
C(x) =
N
X
k=1
αk(x)ck
k−1
Y
i=1
(1 −αi(x))
(3)
Here, αi is the 2D Gaussian projected from the 3D Gaussian
density (Eq. 1) onto the camera plane. ci is the color of the
3D Gaussian as seen in the viewing direction.
From a set of M images IP1, IP2, IP3, ..., IPM of a 3D
scene taken from viewing directions P1, P2, P3, ..., PM, 3D
reconstruction aims to determine the 3D Gaussians param-
eters that capture the 3D scene. Parameters are determined
by minimizing a loss function Lrecon, which measures the
accuracy of reconstruction. Lrecon is expressed as:
Lrecon = (1 −λssim) L1 + λssimLSSIM
(4)
Here, L1 and LSSIM are the averaged L1 and Structural Sim-
ilarity (SSIM) losses between the rendered and ground-truth
images. The parameters that minimize the loss function are
determined using gradient descent.
2.2. Codebooks to Store 3DG Parameters
3D scenes typically consist of many Gaussians with sim-
ilar sets of parameters.
The memory needed for storing
these redundant Gaussian parameters can be reduced by us-
ing codebooks. A codebook is a small set of vectors storing
3D Gaussian parameters. Each Gaussian is mapped to a
vector in the codebook. A Gaussian’s parameters can be
obtained by looking up the mapped vector in the codebook.
Figure 3. Gaussians mapped to one vector in a codebook of quater-
nions (right hand side). One Gaussian among them, whose quater-
nions are derived from the codebook vector is shown on the left.
C3DGS [47] uses codebooks to compress 3DGS scene.
It uses one codebook to store spherical harmoics coeffi-
cients, and another codebook to store covariance parame-
ters (rotation quaternion and principal axis scales) as shown
in Fig. 3. Each Gaussian is mapped to one vector in the SH
feature codebook, and one vector in the covariance code-
book. Note that although we adopt a similar codebook in-
dexing scheme as C3DGS [47], our method does not restrict
us from choosing this specific way of choosing codebooks.
2.3. 3DGS Training as MCMC Sampling
3DGS-MCMC [26] demonstrated that the training process
of 3DGS can be interpreted as performing bayesian infer-
ence over the probability distribution function given by:
p (G) = 1
Z exp (−Lrecon(G))
(5)
Here, G is the set of parameters of all Gaussians in the
scene, Z is a normalizing constant to the probability dis-
tribution. Lrecon is the reconstruction loss given by Eq. 4,

<!-- page 4 -->
written as a function over Gaussian parameter G. Bayesian
inference is performed by sampling a set of Gaussian pa-
rameter values Grecon from the distribution p (denoted as
Grecon ∼p). Sampling from this high dimensional distribu-
tion function can be performed using Markov Chain Monte
Carlo (MCMC) sampling.
2.3.1. MCMC Sampling via Metropolis-Hastings
MCMC is used to sample from high dimensional probabil-
ity distributions. Consider a high-dimensional state space
X, and a probability distribution function pX defined over
it. Our goal is to generate samples x from this distribution,
denoted as x ∼pX. One such MCMC sampling algorithm
is the Metropolis-Hastings (MH) [19]. MH is implemented
by constructing two distributions:
• Proposal distribution: Denoted as q(xi →xj), the pro-
posal distribution q specifies the probability of transition-
ing from each state in the state space xi to state xj.
• Acceptance distribution: Denoted as A(xi →xj), the
acceptance distribution determines whether a proposed
transition is accepted. Specifically, it is defined as:
A(xi →xj) = min

1, pX(xj)q(xj →xi)
pX(xi)q(xi →xj)

(6)
A(xi →xj) represents the probability of accepting a
move from xi to xj, ensuring that the sampling process
converges to the target distribution pX.
The algorithm then proceeds as follows: Starting from a
randomly initialized state x0, MH generates a sequence of
states x0 →x1 →... →xT . At each step, a new state
xj is sampled from current state xi using the proposal dis-
tribution q(xi →xj). This proposal is then accepted with
probability A(xi →xj), resulting in the next state being
xj, or rejected, resulting in the next state remaining xi. If
the proposal distribution q ensures that the state transitions
are ergodic, MH is guaranteed to asymptotically converge
to a sample drawn from the distribution pX.
2.3.2. 3DGS Training with SGLD
MH algorithm can be used to draw a sample G from distri-
bution p in Eq. 5 to obtain 3D Gaussian parameters. Starting
from an random initial state G0, MH generates a sequence
of transitions G0 →G1 →G2..Gt.. →GT . The pro-
posal distribution for transition from state Gt to Gt+1 in the
sequence is given by:
q(Gt →Gt+1) = N

Gt + ϵ
2∇G log(p(Gt)), ϵI

(7)
where N is the standard normal distribution, ϵ is a step
size parameter, ∇G log(p(Gt)) is the gradient of the log-
likelihood of the parameters given the ground truth images,
and I is the identity matrix. In standard MH, an acceptance
distribution A(Gt →Gt+1) would be used to determine
whether to accept the proposed transition:
A(Gt →Gt+1) = min

1, p(Gt+1)q(Gt+1 →Gt)
p(Gt)q(Gt →Gt+1)

(8)
However, in the Stochastic Gradient Langevin Dynam-
ics (SGLD) approximation, this explicit acceptance step is
omitted. Specifically, the SGLD update rule is given by:
Gt+1 = Gt + ϵ
2∇G log(p(Gt)) + √ϵη
(9)
where η ∼N(0, I).
3. Related Work
Compressing Scenes Represented by 3DGS. A range of
prior works [5–7, 10, 11, 18, 44, 46, 47, 55, 69, 70] aim to
compress trained 3DGS models in order to reduce the stor-
age required or to accelerate rendering speeds. To do this,
these works propose several techniques such as pruning re-
dundant gaussians, quantization, reducing bit-widths of less
sensitive parameters, and cutting down higher order spher-
ical harmonics coefficients to reduce the storage footprint
needed to represent the scene. Other prior works [10, 31,
47] also propose using codebooks to compress the model
memory footprint. However, these approaches compress
the 3D model representation post-training. In this work, we
aim to enable both, efficient training and rendering, by re-
ducing the memory usage during the training process itself.
To this end, we propose a novel codebook-based compres-
sion mechanism that significantly reduces memory utiliza-
tion and model size during training.
Accelerating 3DGS Training.
Prior works propose a
range of techniques to accelerate 3DGS training. SpeedyS-
plat [17], TurboGS [36], and TamingGS [39] propose den-
sification control heuristic strategies to reduce the number
of gaussians and thus accelerate both training and render-
ing.
These approaches reduce the memory consumption
during training, however they incur some loss in represen-
tation quality. We quantitively compare against these works
in Section 5, and we demonstrate that our approach achieves
higher or similar compression rates without sacrificing rep-
resentation quality.1 Additionally, these approaches require
careful tuning of the heuristic hyperparameters that may
not generalize well across 3D scenes. EAGLES [15] and
Scaffold-GS [37] aim to accelerate training speeds by using
a compact representation that leverages neural networks to
store 3D gaussian parameters more efficiently. We compare
against EAGLES and Scaffold-GS in Section 5 and demon-
strate that it cannot achieve the same representation quality
as state-of-art approaches.
1We could not compare against TurboGS [36] due to lack of availability
of open-source code, but would incur the same quality challenges due to
the reduced number of Gaussians for representation.

<!-- page 5 -->
Recent works such as 3DGS-LM [20] and 3DGS2 [29] pro-
pose using second order optimization techniques that re-
quire fewer iterations for convergence, thus accelerating
the training process. These approaches are orthogonal to
our work and we note that 3DGS-LM only reports small
speedups over the baseline [25]. Other works [8, 12, 33, 39]
propose low-level optimizations to the renderer’s GPU code
implementation or hardware support to speed up 3D gaus-
sian splatting training [9, 30]. These training acceleration
approaches are orthogonal to our work.
4. Method
Our goal in this work is to enable training compressed
representations of 3D Gaussians for high-speed and mem-
ory efficient 3D reconstruction.
To this end, we lever-
age codebook representations that can significantly reduce
the amount of memory required to store 3DGS param-
eters.
While codebooks have been demonstrated to be
highly effective at compressing fully trained 3DGS mod-
els [10, 31, 47], directly training using codebook represen-
tations is an unsolved challenge. In this work, we solve the
problem of learning on codebook-compressed representa-
tions by using MCMC sampling to sample over a posterior
distribution of these compressed representations.
This requires (1) defining a state space for codebook-
compressed 3DGS parameters; (2) formulating a posterior
distribution over the codebook-based state representation
whose samples accurately reconstruct the 3D scene; and (3)
defining state transitions and deriving corresponding pro-
posal and acceptance probabilities for MCMC sampling.
We now describe these elements of our approach.
4.1. Compressed Model State Space
We use a codebook representation similar to C3DGS [47],
but note that ContraGS can also use a different codebook
implementation (see Section 2.2). We define the state of the
codebook-compressed 3DGS model as: S = {G, C}:
G = {g1, g2, g3, ...gN}
C = {SH, SR}
(10)
In this expression, C comprises the 3DGS codebooks: SH
and SR.
The SH codebook stores vectors of spherical
harmonics coefficients used to derive the view dependent
color. The SR codebook contains vectors, where each vec-
tor stores the scaling and quaternion parameters for each
Gaussian. G corresponds to the set of 3D Gaussians, where
gi corresponds to a Gaussian with the following parameters:
gi = {pi, oi, g2sri, g2shi}
(11)
Here, pi is the position of the mean, oi the opacity. g2sh
and g2sr are integers that are pointers to vectors in the SH
and SR codebook respectively. Thus, a single 3D Gaussian
in the compressed 3DGS model G is represented as si =
{gi, C}. The state space representation described above is
summarized in Fig. 4.
Figure 4. Codebook layout used for ContraGS. ContraGS al-
locates two codebooks: (1) SH stores spherical harmonics coef-
ficientes of RGB colors, and (2) SR stores scaling + quaternion
parameters concatenated together as vectors. Each Gaussian has
g2sh, g2sr to index the two codebooks.
4.2. Formulating The Posterior Distribution
We define a posterior probability distribution over the Con-
traGS model state space as follows:
p (G, C) ∝exp
 −L

IP(G, C), Igt
P

(12)
IP(G, C) represents the image rendered from camera pose
P using the ContraGS codebook-based Gaussian model
state (G, C), and Igt
P is the corresponding ground truth im-
ages. We define the reconstruction loss function L as:
L = Lrecon + λsr|SR| + λsh|SH|
(13)
Lrecon is the reconstruction loss (defined in Eq. 4).
The
terms |SR| and |SH| represent the number of vectors in
the SR and SH codebooks, respectively. λsr and λsh are hy-
perparameters used to control the size of these codebooks.
Bayesian inference over this probability function is per-
formed by sampling the state S = {G, C}. Due to the
high dimensionality of the distribution, we use MCMC sam-
pling. However, unlike recent 3DGS-MCMC [26] that uses
SGLD for MCMC sampling (see Section 2.3.2), SGLD can-
not be applied to learn a discrete set of parameters such
as the mapping/indexing between the Gaussians and code-
book vectors. Instead we use the MH algorithm to sample
from the posterior defined on codebook-compressed repre-
sentations. To apply MH, we must define a proposal dis-
tribution and derive an acceptance distribution (see Sec-
tion 2.3.1). The proposal distribution enables splitting and
merging codebook vectors to transition between states with
different mappings between Gaussians and codebook vec-
tors. This allows MCMC to explore the joint space of con-
tinuous parameters. We now define the possible set of state
transitions over codebook-based 3DGS representations, and
define proposal distributions between these transitions.
4.3. Model State Space Transitions
To sample from the probability distribution in Eq. 12 using
MH, we need to define a proposal and acceptance distribu-

<!-- page 6 -->
tion functions (see Section 2.3.1). To do this, we first define
a set of valid state transitions: (1) parameter update, (2)
merge and (3) split.
Figure 5. ContraGS performs one of the 3 steps for each 3DG: (1)
Updates Gaussian and codebook parameters, or (2) Splitting the
3DG’s corresponding vector, or (3) Merge to the vector it previ-
ously split from.
• Parameter Update: This involves changing the values
of the parameters of the SH, SR codebooks as well as
3D Gaussian parameters in gi. The parameters are deter-
mined by the proposal distribution qupdate. The mappings
of the Gaussians to codebook vectors remains unchanged.
• Split: A Gaussian mapped to a codebook vector is split
and allocated a new codebook vector, as shown in Fig. 6a.
The 3D Gaussian mapped to codebook vector c is allo-
cated a new codebook vector c′. The value of c′ is gener-
ated by the proposal distribution qsplit.
• Merge: A Gaussian mapping to a codebook vector c′ can
be merged with a codebook vector c from which it was
previously split from, as shown in Fig. 6b. The parame-
ters in the codebook vector are unchanged.
(a) Split transition: a codebook vector referred to by two or more 3D Gaussian
parameter is split into two vectors.
(b) Merge transition: two codebook vectors referred to by different 3D Gaus-
sian parameters gets merged into one.
Figure 6. Split and Merge transitions.
We now derive the proposal and acceptance distributions.
4.4. Proposal and Acceptance Distributions
We define the proposal distribution for MH sampling tran-
sition from state as:
q(S →S′) = 0.98qupdate + 0.01qsplit + 0.01qmerge
(14)
Where qupdate is the parameter update transition, qsplit is
the split transition and qmerge is the merge transition. In other
words, at each step, we randomly choose the parameter up-
date step 98% of the times, and the split, merge transitions
1% of the time. Now, we define the proposal distributions
for each type of transition, and derive the acceptance distri-
butions as in Section 2.3.1. Please refer to Section A.1 in
the Appendix for a detailed derivation.
Parameter update step The proposal distribution for a pa-
rameter update, qupdate, is identical to the SGLD parameter
update, as described in Section 2.3.2. Thus we express the
parameter update state transition using qupdate as follows:
qupdate(Sdiff) = N

Sdiff + ϵp
2 ∇Sdifflog(p(S)), ϵpI

(15)
Where ϵp is a small hyperparameters. Sdiff is the differen-
tiable set of parameters (all parameters except the indices to
the codebooks). The update step is equivalent to the SGLD
update step with acceptance A = 1.
Split Step: During the split transition, for each Gaussian,
we allocate a new codebook vector and remap. A Gaussian
mapped to codebook vector c is remapped to a new code-
book vector entry row c′′. The original codebook vector
becomes c′.
c′ = c
c′′ = c + u
(16)
qsplit(u) = N(u|µ = 0, σ = ϵsplitI)
(17)
Where ϵsplit is a hyperparameter.
Merge Step: Two codebook vectors c, c′ can be merged
into one codebook vector of value c. Two rows to be merged
are selected with a transition distribution defined by:
qmerge(S →Smerge) = N(c −c′|µ = 0, σ = ϵmergeI) (18)
Where ϵmerge is a hyperparameter.
Acceptance:
Acceptance probabilities of the split and
merge step can be derived as:
A(S →Ssplit) = min

1, e−λSH
1
qsm(c′ −c′′)

(19)
A(S →Smerge) = min
 1, eλSHqsm(c −c′)

(20)
Where qsm is given by:
qsm(u) = exp
 
−u2
2
 
1
ϵ2
split
−
1
ϵ2merge
!!
(21)
5. Results
5.1. Experiment Setup
Evaluation Platform, Hyperparameter Configuration.
We measure training and rendering speeds of ContraGS on a

<!-- page 7 -->
MipNerf360
Deep Blending
Tanks and Temples
PSNR SSIM LPIPS Peak Mem PSNR SSIM LPIPS Peak Mem PSNR SSIM LPIPS Peak Mem
3DGS*
29.09
0.867
0.183
2089.363
30.10
0.909
0.241
817.159
22.03
0.821
0.197
901.405
EAGLES*
28.70
0.867
0.194
159.473
30.38
0.913
0.251
159.473
22.34
0.798
0.237
57.962
SpeedySplat*
28.33
0.846
0.241
-
30.02
0.907
0.269
-
21.68
0.773
0.289
-
Reduced-GS* 29.03
0.870
0.184
1432.407
29.96
0.906
0.243
572.136
23.72
0.846
0.176
636.500
Taming-GS*
29.39
0.863
0.198
477.598
30.11
0.910
0.251
325.629
22.58
0.830
0.191
1189.745
Scaffold-GS*
29.22
0.869
0.190
304.827
30.89
0.913
0.244
95.107
22.54
0.829
0.190
192.211
MCMC-2M
30.71
0.908
0.161
473.000
33.99
0.929
0.243
473.000
24.32
0.864
0.182
473.000
Ours-2M
30.06
0.899
0.175
130.038
33.73
0.925
0.252
120.672
24.35
0.861
0.187
127.968
MCMC-5M
32.72
0.943
0.128
1182.000
34.54
0.939
0.209
1182.000
26.63
0.911
0.112
1182.000
Ours-5M
31.01
0.919
0.146
275.594
34.33
0.932
0.232
444.259
26.18
0.898
0.127
338.000
Table 1. Averaged PSNR, SSIM and LPIPS and peak model memory during training (SfM initialization for * approaches)
system with a Core i7-13700K CPU and an NVIDIA RTX-
4090 GPU. For our evaluation, we set λSH to 2.3, λSR = 3.
ϵsplit and ϵmerge are set to 0.1, 0.05 for both SH and SR
codebooks. At each training step, we choose to perform
the parameter update step with a probability of 98%, and
the split and merge transitions with a 1% probability. We
initialize training with 100000 Gaussians with random pa-
rameters and grow the number of Gaussians by 5% every
100 training iterations. We evaluate two versions of Con-
traGS, ContraGS-2M and ContraGS-5M in which we cap
the Gaussians count at 2 million and 5 million respectively.
Dataset and Metrics. We evaluate ContraGS on 3D re-
construction tasks using multi-view images. We measure
the PSNR, LPIPS and SSIM on evaluation views of 3D
reconstruction tasks. We consider the following datasets
that contain multiview images of real world and synthetic
3D scenes: MipNerf360 [2] (counter, stump, kitchen, bicy-
cle, bonsai, room and garden scenes), Blender [41] (chair,
drums, ficus, hotdog, lego, materials, mic and ship scenes),
Deep Blending’s Playroom scene and Tanks and Tem-
ples [27] (Truck and Train scenes) datasets.
Prior Work Comparisons. We then compare the recon-
struction quality with prior works.
We compare Con-
traGS approach against 3DGS [25], MCMC-2M [26], and
MCMC-5M (two configurations with 2 million and 5 mil-
lion Gaussians respectively); 3DGS-MCMC [26] achieves
state-of-art representation quality.
Both MCMC-5M and
ContraGS-5M incur higher memory cost than their re-
spective 2M configurations but provide higher representa-
tion qualities.
We also compare with other prior works
that enable faster and more memory efficient training
by pruning the number of Gaussians (Section 3):
EA-
GLES [15], SpeedySplat [17], Reduced-GS[46], Tam-
ingGS [39], Scaffold-GS [37]. For prior work, we initial-
ize the positions and RGB colors of the Gaussians using the
corresponding Structure-from-Motion point cloud.
5.2. Representation Quality & Peak Model Memory
Table 1 shows the average PSNR, SSIM, LPIPS and the
peak model memory during training (in MB) achieved
by ContraGS on multiview 3D scene reconstruction tasks.
PSNR, SSIM and LPIPS measured for individual scenes are
presented in Section B of Appendix along with qualitative
comparisons of generated images.
First, we observe that ContraGS is able to achieve simi-
lar representation quality as that of a state-of-art approach
3DGS-MCMC [26]: less than 0.3 PSNR on average with
2 million Gaussians, and less than 0.8 PSNR on average
with 5 million Gaussians. ContraGS-2M achieves PSNR
equivalent to MCMC-2M on the Deep Blending and Tanks
and Temples datasets, and incurs a 0.65 PSNR drop on
average on the MipNerf360 dataset.
ContraGS-5M also
achieves equivalent PSNR on the DeepBlending incurs a
sharper PSNR drop on MipNerf360 (1.5 PSNR). At the
same time, ContraGS requires significantly less peak model
memory (3.78× and 3.5× on average respectively). Sec-
ond, ContraGS-5M outperforms MCMC-2M on all qual-
ity metrics (PSNR, SSIM and LPIPS) while having smaller
model memory size (1.39× on average). Third, compared
to the prior work that aims to achieve memory efficient
training, ContraGS-2M achieves significantly higher repre-
sentation quality while having a similar peak memory us-
age compared to EAGLES, Reduced-GS, Taming-GS and
Scaffold-GS. Finally, compared to 3DGS, ContraGS-2M is
able to significantly reduce the peak model memory during
training, by 9.99× on average. We conclude that ContraGS
enables significantly reduced peak memory during training
while retaining close to state-of-art quality. Thus, for any
memory capacity constraint (i.e, peak memory utilization),
ContraGS achieves the highest representation quality com-
pared to prior works.
Random point cloud initialization.
We note that
ContraGS-2M and ContraGS-5M (and MCMC-2M, 5M)
are able to reconstruct the scene initialized on a random
point cloud). The results presented in this section for these
configurations are with randomly initialized Gaussian pa-
rameters. EAGLES, Taming-GS, SpeedySplat, Reduced-
GS require the SfM point cloud to initialize the Gaussian
parameters and the results in this section were collected us-
ing SfM. A fairer comparison between these methods would
assume a random initialization, but since certain scenes
in the MipNerf360, Deep Blending, Tanks and Temples
datasets could not be trained with random initialization for
these approaches, the results in Table 1 include SfM ini-

<!-- page 8 -->
(a) #Gaussians vs. PSNR
(b) #Gaussians vs. Model Memory
(c) #Gaussians vs. Training Time
Figure 7. The impact of Gaussian count for 3DGS-MCMC and ContraGS on training quality (PSNR), model memory, and training time.
tialization for all approaches except ContraGS and MCMC.
We were able to train scenes in the Blender dataset for all
approaches (except Taming-GS) with random initialization,
and we present these results in Table 2.
Table 2 shows the average quality metrics and peak mem-
ory during training of ContraGS-2M for synthetic scenes
in the Blender dataset.
ContraGS-2M achieves a sig-
nificantly higher reconstruction quality compared to prior
works. Although the peak model memory during training
of ContraGS-2M is higher in comparison, the peak mem-
ory is in the same range as that of real world scenes in Ta-
ble 1. We do not report results from MCMC here because
MCMC-2M fails to reconstruct all synthetic scenes when
setting the Gaussians count to 2M, due to instability in the
training process.
PSNR SSIM LPIPS Peak Mem
3DGS
31.07
0.959
0.051
83.00
EAGLES
32.55
0.964
0.039
5.17
SpeedySplat
32.43
0.960
0.050
47.89
Reduced-GS
33.80
0.970
0.030
49.72
Scaffold-GS
33.46
0.967
0.034
31.66
Ours-2M
36.840 0.984
0.023
257.87
Table 2.
PSNR, SSIM, LPIPS for different workloads on the
Blender dataset (GS parameters initialized randomly)
5.3. Training and Rendering Speeds
Table 3 shows the average training speed in terms of training
iterations per second for ContraGS. We observe that, when
compared to MCMC-2M, ContraGS-2M is able to speed up
training and rendering by 1.36× and 1.88× on average. We
also compare against Taming3DGS, the state-of-art for ef-
ficient training, in Section E of the Appendix. These works
achieve higher training and rendering speeds. However, the
representation quality is significantly lower.
MipNerf360
Deep Blending
T & T
Train It/s FPS Train It/s FPS Train It/s FPS
MCMC
23.65
102
31.66
161
30.34
133
Ours
32.02
216
46.33
319
40.75
207
MCMC-5M
11.71
65
14.06
114
12.95
80
Ours-5M
21.1
87
19.87
141
19.18
114
Table 3. Training and rendering speeds
5.4. Ablation Study
Quality vs. Gaussians Count. Figure 7a depicts the PSNR
achieved by ContraGS and 3DGS-MCMC on varying the
number of Gaussians used to represent the Truck scene of
the T&T dataset. We observe that as the Gaussian count
increases, the PSNR achieved by ContraGS incurs a small
degradation in quality compared to 3DGS-MCMC. At 5
million Gaussians, the difference in PSNR is about 0.46
on average. Similar to 3DGS-MCMC, ContraGS achieves
higher representation quality with a greater Gaussian count.
Model memory vs. Number of Gaussians. Fig. 7b depicts
how the post-training memory footprint of the model varies
with the Gaussians count for the Tanks and Temples dataset.
This memory footprint increases linearly with the Gaussian
count for 3DGS-MCMC. However, ContraGS is able to rep-
resent large Gaussian counts without significantly increas-
ing the memory footprint and requires a lot less memory
than 3DGS-MCMC (3.49× on average).
Training speed vs. Number of Gaussians. Fig. 7c de-
picts the wall clock time needed to train the playroom scene
of the Deep Blending dataset for 30000 iterations. With
3DGS-MCMC, the training time increases linearly with the
gaussian count, however, ContraGS enable more efficient
and scalable training at larger gaussian counts.
6. Conclusion
We introduce ContraGS, a novel method for learn-
ing codebook-compressed representations of 3D Gaussian
Splatting for high-speed memory-efficient scene recon-
struction.
ContraGS can effectively reduce the model’s
memory footprint during training while accelerating train-
ing and rendering. This makes ContraGS useful for recon-
structing complex scenes with 3DGS, which is challenging
on resource-constrained platforms such as mobile devices
and web browsers. ContraGS achieves the highest repre-
sentation quality for any memory capacity constraint (i.e,
peak memory utilization) compared to existing approaches.
ContraGS provides an extensible mathematical framework
that can be applied to do compressed training on a range of
point-based scene representation methods.

<!-- page 9 -->
References
[1] Jad Abou-Chakra, Krishan Rana, Feras Dayoub, and Niko
S¨underhauf. Physically embodied gaussian splatting: A re-
altime correctable world model for robotics. arXiv preprint
arXiv:2406.10788, 2024. 2
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5470–5479, 2022. 7
[3] Minghao Chen, Iro Laina, and Andrea Vedaldi. Dge: Di-
rect gaussian 3d editing by consistent multi-view editing.
In European Conference on Computer Vision, pages 74–92.
Springer, 2024. 2
[4] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xi-
aofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping
Liu, and Guosheng Lin. Gaussianeditor: Swift and control-
lable 3d editing with gaussian splatting. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 21476–21485, 2024. 2
[5] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai.
Hac: Hash-grid assisted context for 3d
gaussian splatting compression. In European Conference on
Computer Vision, pages 422–438. Springer, 2024. 4
[6] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai. Hac++: Towards 100x compression of 3d
gaussian splatting. arXiv preprint arXiv:2501.12255, 2025.
[7] Tianchen Deng, Yaohui Chen, Leyan Zhang, Jianfei Yang,
Shenghai Yuan, Jiuming Liu, Danwei Wang, Hesheng Wang,
and Weidong Chen. Compact 3d gaussian splatting for dense
visual slam. arXiv preprint arXiv:2403.11247, 2024. 2, 4
[8] Sankeerth Durvasula, Adrian Zhao, Fan Chen, Ruofan
Liang, Pawan Kumar Sanjaya, and Nandita Vijaykumar.
Distwar: Fast differentiable rendering on raster-based ren-
dering pipelines. arXiv preprint arXiv:2401.05345, 2023. 5
[9] Sankeerth Durvasula, Adrian Zhao, Fan Chen, Ruofan
Liang, Pawan Kumar Sanjaya, Yushi Guan, Christina Gian-
noula, and Nandita Vijaykumar. Arc: Warp-level adaptive
atomic reduction in gpus to accelerate differentiable render-
ing. In Proceedings of the 30th ACM International Confer-
ence on Architectural Support for Programming Languages
and Operating Systems, Volume 1, pages 64–83, 2025. 5
[10] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia
Xu, Zhangyang Wang, et al.
Lightgaussian: Unbounded
3d gaussian compression with 15x reduction and 200+ fps.
Advances in neural information processing systems, 37:
140138–140158, 2025. 2, 4, 5
[11] Guangchi Fang and Bing Wang.
Mini-splatting: Repre-
senting scenes with a constrained number of gaussians. In
European Conference on Computer Vision, pages 165–181.
Springer, 2024. 4
[12] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi
Wang, Tao Liu, Zhilin Pei, Hengjie Li, Xingcheng Zhang,
and Bo Dai.
Flashgs: Efficient 3d gaussian splatting for
large-scale and high-resolution rendering.
arXiv preprint
arXiv:2408.07967, 2024. 5
[13] Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wen-
chao Ma, Le Chen, Danhang Tang, and Ulrich Neumann.
Gaussianflow: Splatting gaussian dynamics for 4d content
creation. arXiv preprint arXiv:2403.12365, 2024. 2
[14] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur
Brussee,
Ricardo
Martin-Brualla,
Pratul
Srinivasan,
Jonathan T Barron, and Ben Poole. Cat3d: Create anything
in 3d with multi-view diffusion models.
arXiv preprint
arXiv:2405.10314, 2024. 2
[15] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. Ea-
gles: Efficient accelerated 3d gaussians with lightweight en-
codings. In European Conference on Computer Vision, pages
54–71. Springer, 2024. 2, 4, 7
[16] Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi,
and Andrea Tagliasacchi. Radiant foam: Real-time differen-
tiable ray tracing. arXiv preprint arXiv:2502.01157, 2025.
3
[17] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias
Zwicker, and Tom Goldstein. Speedy-splat: Fast 3d gaus-
sian splatting with sparse pixels and sparse primitives. arXiv
preprint arXiv:2412.00578, 2024. 2, 4, 7
[18] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana,
Matthias Zwicker, and Tom Goldstein. Pup 3d-gs: Principled
uncertainty pruning for 3d gaussian splatting. arXiv preprint
arXiv:2406.10219, 2024. 2, 4
[19] W. K. Hastings.
Monte carlo sampling methods using
markov chains and their applications. Biometrika, 57(1):97–
109, 1970. 4
[20] Lukas H¨ollein,
Aljaˇz Boˇziˇc,
Michael Zollh¨ofer,
and
Matthias Nießner.
3dgs-lm:
Faster gaussian-splatting
optimization with levenberg-marquardt.
arXiv preprint
arXiv:2409.12892, 2024. 5
[21] Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li,
Liangjing Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng
Cui.
Cg-slam: Efficient dense rgb-d slam in a consistent
uncertainty-aware 3d gaussian field. In European Confer-
ence on Computer Vision, pages 93–112. Springer, 2024. 2
[22] Yue Hu, Rong Liu, Meida Chen, Andrew Feng, and Peter
Beerel. Splatmap: Online dense monocular slam with 3d
gaussian splatting. arXiv preprint arXiv:2501.07015, 2025.
2
[23] Lutao Jiang, Xu Zheng, Yuanhuiyi Lyu, Jiazhou Zhou, and
Lin Wang.
Brightdreamer: Generic 3d gaussian genera-
tive framework for fast text-to-3d synthesis. arXiv preprint
arXiv:2403.11273, 2024. 2
[24] HyunJun Jung, Nikolas Brasch, Jifei Song, Eduardo Perez-
Pellitero, Yiren Zhou, Zhihao Li, Nassir Navab, and Ben-
jamin Busam. Deformable 3d gaussian splatting for animat-
able human avatars. arXiv preprint arXiv:2312.15059, 2023.
2
[25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 1, 3, 5, 7
[26] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Jeff Tseng, Hossam Isack, Abhishek Kar, An-
drea Tagliasacchi, and Kwang Moo Yi.
3d gaussian

<!-- page 10 -->
splatting as markov chain monte carlo.
arXiv preprint
arXiv:2404.09591, 2024. 2, 3, 5, 7
[27] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and temples: Benchmarking large-scale scene
reconstruction. ACM Transactions on Graphics, 36(4), 2017.
2, 7
[28] Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel,
Oncel Tuzel, and Anurag Ranjan. Hugs: Human gaussian
splats. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 505–515, 2024.
2
[29] Lei Lan, Tianjia Shao, Zixuan Lu, Yu Zhang, Chenfanfu
Jiang, and Yin Yang. 3dgs2: Near second-order converg-
ing 3d gaussian splatting. arXiv preprint arXiv:2501.13975,
2025. 5
[30] Junseo Lee, Seokwon Lee, Jungi Lee, Junyong Park, and Jae-
woong Sim. Gscore: Efficient radiance field rendering via
architectural support for 3d gaussian splatting. In Proceed-
ings of the 29th ACM International Conference on Architec-
tural Support for Programming Languages and Operating
Systems, Volume 3, pages 497–511, 2024. 5
[31] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park. Compact 3d gaussian representation for
radiance field. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21719–
21728, 2024. 2, 4, 5
[32] Xinhai Li, Jialin Li, Ziheng Zhang, Rui Zhang, Fan Jia,
Tiancai Wang, Haoqiang Fan, Kuo-Kun Tseng, and Ruiping
Wang. Robogsim: A real2sim2real robotic gaussian splatting
simulator. arXiv preprint arXiv:2411.11839, 2024. 2
[33] Weikai Lin, Yu Feng, and Yuhao Zhu. Rtgs: Enabling real-
time gaussian splatting on mobile devices using efficiency-
guided pruning and foveated rendering.
arXiv preprint
arXiv:2407.00435, 2024. 5
[34] Rong Liu, Rui Xu, Yue Hu, Meida Chen, and Andrew Feng.
Atomgs: Atomizing gaussian splatting for high-fidelity radi-
ance field. arXiv preprint arXiv:2405.12369, 2024. 2
[35] Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and An-
drew Feng.
Deformable beta splatting.
arXiv preprint
arXiv:2501.18630, 2025. 3
[36] Tao Lu, Ankit Dhiman, R Srinath, Emre Arslan, Angela
Xing, Yuanbo Xiangli, R Venkatesh Babu, and Srinath Srid-
har.
Turbo-gs: Accelerating 3d gaussian fitting for high-
quality radiance fields.
arXiv preprint arXiv:2412.13547,
2024. 2, 4
[37] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 4, 7
[38] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 2
[39] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl,
Markus Steinberger, Francisco Vicente Carrasco, and Fer-
nando De La Torre.
Taming 3dgs: High-quality radiance
fields with limited resources. In SIGGRAPH Asia 2024 Con-
ference Papers, pages 1–11, 2024. 4, 5, 7
[40] Marko Mihajlovic, Sergey Prokudin, Siyu Tang, Robert
Maier, Federica Bogo, Tony Tung, and Edmond Boyer.
Splatfields: Neural gaussian splats for sparse 3d and 4d re-
construction. In European Conference on Computer Vision,
pages 313–332. Springer, 2024. 2
[41] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
7
[42] Gyeongsik Moon, Takaaki Shiratori, and Shunsuke Saito.
Expressive whole-body 3d gaussian avatar.
In European
Conference on Computer Vision, pages 19–35. Springer,
2024. 2
[43] Arthur Moreau, Jifei Song, Helisa Dhamo, Richard Shaw,
Yiren Zhou, and Eduardo P´erez-Pellitero. Human gaussian
splatting: Real-time rendering of animatable avatars. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 788–798, 2024. 2
[44] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakoto-
saona, Michael Oechsle, Daniel Duckworth, Rama Gosula,
Keisuke Tateno, John Bates, Dominik Kaeser, and Federico
Tombari. Radsplat: Radiance field-informed gaussian splat-
ting for robust real-time rendering with 900+ fps.
arXiv
preprint arXiv:2403.13806, 2024. 4
[45] Francesco Palandra, Andrea Sanchietti, Daniele Baieri, and
Emanuele Rodol`a.
Gsedit:
Efficient text-guided edit-
ing of 3d objects via gaussian splatting.
arXiv preprint
arXiv:2403.05154, 2024. 2
[46] Panagiotis Papantonakis,
Georgios Kopanas,
Bernhard
Kerbl, Alexandre Lanvin, and George Drettakis. Reducing
the memory footprint of 3d gaussian splatting. Proceedings
of the ACM on Computer Graphics and Interactive Tech-
niques, 7(1):1–17, 2024. 4, 7
[47] Jiating Qian, Yiming Yan, Fengjiao Gao, Baoyu Ge,
Maosheng Wei, Boyi Shangguan, and Guangjun He. C3dgs:
Compressing 3d gaussian model for surface reconstruction
of large-scale scenes based on multi-view uav images. IEEE
Journal of Selected Topics in Applied Earth Observations
and Remote Sensing, 2025. 2, 3, 4, 5
[48] Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas
Geiger, and Siyu Tang.
3dgs-avatar: Animatable avatars
via deformable 3d gaussian splatting.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5020–5030, 2024. 2
[49] Yansong Qu,
Dian Chen,
Xinyang Li,
Xiaofan Li,
Shengchuan Zhang, Liujuan Cao, and Rongrong Ji. Drag
your gaussian:
Effective drag-based editing with score
distillation for 3d gaussian splatting.
arXiv preprint
arXiv:2501.18672, 2025. 2
[50] Darius R¨uckert, Linus Franke, and Marc Stamminger. Adop:
Approximate differentiable one-pixel point rendering. ACM
Transactions on Graphics (ToG), 41(4):1–14, 2022. 3
[51] Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang,
Xiangru Lin, Yu Zhang, Mingming Fan, and Zeyu Wang.

<!-- page 11 -->
Splattingavatar:
Realistic real-time human avatars with
mesh-embedded gaussian splatting. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 1606–1616, 2024. 2
[52] David Svitov, Pietro Morerio, Lourdes Agapito, and Alessio
Del Bue. Haha: Highly articulated gaussian human avatars
with textured mesh prior. In Proceedings of the Asian Con-
ference on Computer Vision, pages 4051–4068, 2024. 2
[53] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang
Zeng. Dreamgaussian: Generative gaussian splatting for effi-
cient 3d content creation. arXiv preprint arXiv:2309.16653,
2023. 2
[54] Nicolas von L¨utzow and Matthias Nießner. Linprim: Lin-
ear primitives for differentiable volumetric rendering. arXiv
preprint arXiv:2501.16312, 2025. 3
[55] Henan Wang, Hanxin Zhu, Tianyu He, Runsen Feng, Jia-
jun Deng, Jiang Bian, and Zhibo Chen.
End-to-end rate-
distortion optimized 3d gaussian representation. In European
Conference on Computer Vision, pages 76–92. Springer,
2024. 2, 4
[56] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 20310–20320, 2024. 2
[57] Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian
Reid, Philip Torr, and Victor Adrian Prisacariu. Gaussctrl:
Multi-view consistent text-driven 3d gaussian splatting edit-
ing. In European Conference on Computer Vision, pages 55–
71. Springer, 2024. 2
[58] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng,
Yin Yang, and Chenfanfu Jiang.
Physgaussian: Physics-
integrated 3d gaussians for generative dynamics. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 4389–4398, 2024. 2
[59] Xinli Xu, Wenhang Ge, Dicong Qiu, ZhiFei Chen, Dongyu
Yan, Zhuoyun Liu, Haoyu Zhao, Hanfeng Zhao, Shunsi
Zhang, Junwei Liang, et al.
Gaussianproperty: Integrat-
ing physical properties to 3d gaussians with lmms. arXiv
preprint arXiv:2412.11258, 2024. 2
[60] Yueming Xu, Haochen Jiang, Zhongyang Xiao, Jianfeng
Feng, and Li Zhang. Dg-slam: Robust dynamic gaussian
splatting slam with hybrid pose optimization. arXiv preprint
arXiv:2411.08373, 2024. 2
[61] Yansong Xu, Junlin Li, Wei Zhang, Siyu Chen, Shengyong
Zhang, Yuquan Leng, and Weijia Zhou. Fgs-slam: Fourier-
based gaussian splatting for real-time slam with sparse and
dense map fusion. arXiv preprint arXiv:2503.01109, 2025.
[62] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
19595–19604, 2024. 2
[63] Ziyang Yan, Lei Li, Yihua Shao, Siyu Chen, Wuzong Kai,
Jenq-Neng Hwang, Hao Zhao, and Fabio Remondino. 3dsce-
needitor: Controllable 3d scene editing with gaussian splat-
ting. arXiv preprint arXiv:2412.01583, 2024. 2
[64] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction.
In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 20331–20341, 2024. 2
[65] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
Gaussian grouping: Segment and edit anything in 3d scenes.
In European Conference on Computer Vision, pages 162–
179. Springer, 2024. 2
[66] Ye Yuan, Xueting Li, Yangyi Huang, Shalini De Mello, Koki
Nagano, Jan Kautz, and Umar Iqbal. Gavatar: Animatable
3d gaussian avatars with implicit mesh learning. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 896–905, 2024. 2
[67] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li,
Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d
gaussian splatting for unconstrained image collections. In
European Conference on Computer Vision, pages 341–359.
Springer, 2024. 2
[68] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao,
Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large recon-
struction model for 3d gaussian splatting. In European Con-
ference on Computer Vision, pages 1–19. Springer, 2024. 2
[69] Xinjie Zhang, Xingtong Ge, Tongda Xu, Dailan He, Yan
Wang, Hongwei Qin, Guo Lu, Jing Geng, and Jun Zhang.
Gaussianimage: 1000 fps image representation and compres-
sion by 2d gaussian splatting. In European Conference on
Computer Vision, 2024. 4
[70] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang,
Cheng Peng, Rama Chellappa, and Deliang Fan. Lp-3dgs:
Learning to prune 3d gaussian splatting. Advances in Neural
Information Processing Systems, 37:122434–122457, 2025.
4
[71] Hongliang Zhong, Can Wang, Jingbo Zhang, and Jing
Liao.
Generative object insertion in gaussian splat-
ting with a multi-view diffusion model.
arXiv preprint
arXiv:2409.16938, 2024. 2

<!-- page 12 -->
A. Acceptance Probability Derivation
A.1. Proposal and Acceptance Probabilities
To determine whether a split, merge or parameter update
step proposed is accepted to be taken or not - we compute
the acceptance probability as follows:
A(S →S′) = min

1, P(S′)q(S|S′)
P(S)q(S′|S)

(22)
For the parameter update transition that uses the SGLD up-
date, A = 1. For the merge and split transitions, the accep-
tance probability is given by:
A(S →Smerge) = min

1, P(Smerge)qsplit(Smerge →S)
P(S)qmerge(S →Smerge)

(23)
A(S →Ssplit) = min

1, P(Ssplit)qmerge(Ssplit →S)
P(S)qsplit(S →Ssplit)

(24)
We now derive the acceptance distribution.
A.1.1. Parameter update choice
The proposal distribution for a parameter update, qupdate,
is identical to the SGLD parameter update. Thus we ex-
press the parameter update state transition using qupdate as
follows:
pt+1 ∼N

pt + ϵp
2 ∇p log(p(S)), ϵpI

(25)
ot+1 ∼N

ot + ϵo
2 ∇o log(p(S)), ϵoI

(26)
SHt+1 ∼N

SHt + ϵSH
2 ∇SH log(p(G, C)), ϵSHI

(27)
SRt+1 ∼N

SRt + ϵSR
2 ∇SR log(p(G, C)), ϵSRI

(28)
Where ϵp, ϵo, ϵSH, ϵSR are small hyperparameters. The ac-
ceptance probability A = 1.
A.1.2. Split Codebook Vectors
During the split transition, we split its codebook vector
into a new entry. A 3DG mapped to codebook vector c is
remapped into a new codebook vector entry row c′′. The
original codebook vector becomes c′.
c′ = c
c′′ = c + u
(29)
qsplit(u) = N(0, ϵsplitI)
(30)
The acceptance probability of this state transition is given
by:
A(S →Ssplit) = min

1, p(Ssplit)qmerge(c −c′)
p(S)qsplit(c −c′)

(31)
The ratio p(Ssplit)
p(S)
≈e−λSH if we choose to split the SH code-
book, and e−λSR if we choose to split the SR codebook. We
thus have the acceptance probability given by:
A(S →Ssplit) = min
 1, e−λSH/qsm(u)

(32)
A.1.3. Merge Codebook Vectors
Two codebook vectors c, c′ can be merged into one code-
book vector of value c. Two rows to be merged are selected
with a transition distribution defined by:
qmerge(S →Smerge) = N(c −c′|µ = 0, σ = ϵmergeI) (33)
The acceptance probability of a merge transition is:
A(S →Smerge) =
(34)
min

1, p(Smerge)qsplit(c −c′)
p(S)qmerge(c −c′)

p(S)
p(Smerge) ≈eλSH if it leads to a reduction in the number of
rows, or 1 otherwise, as merging a small set of rows of code-
book vectors does not affect the overall accuracy.
A(S →Smerge) = min
 1, eλSHqsm(c −c′)

(35)
Where
qsm(u) = exp
 
u2
2
 
1
ϵ2merge
−
1
ϵ2
split
!!
(36)

<!-- page 13 -->
B. Reconstruction using an SfM Initialized Point Cloud
DeepBlending
MipNerf360
Tanks and Temples
playroom
bicycle bonsai counter garden kitchen room stump
train
truck
3DGS
30.10
25.18
31.98
29.13
27.38
31.54
31.77 26.67 22.03
25.39
EAGLES
30.38
25.02
31.45
28.42
26.94
30.79
31.64 26.67 22.34
25.04
Reduced-GS
29.96
25.12
32.10
29.13
27.28
31.33
31.68 26.58 22.01
25.42
Scaffold-GS
30.89
25.02
32.50
29.43
27.30
31.42
32.13 26.72 22.54
25.74
SpeedySplat
30.02
25.10
31.20
28.28
26.68
29.65
30.78 26.64 21.68
25.23
Taming3DGS
30.11
24.86
32.93
29.63
27.59
32.16
32.40 26.17 22.58
26.00
ContraGS-2M
33.73
27.02
31.59
30.60
27.49
30.54
32.34 30.83 24.35
26.93
Table 4. PSNR on evaluation datasets
DeepBlending
MipNerf360
Tanks and Temples
playroom
bicycle bonsai counter garden kitchen room stump
train
truck
3DGS
0.909
0.748
0.946
0.916
0.858
0.916
0.916 0.768 0.821
0.885
EAGLES
0.913
0.750
0.942
0.907
0.840
0.928
0.927 0.774 0.798
0.876
Reduced-GS
0.906
0.747
0.947
0.915
0.856
0.932
0.926 0.768 0.810
0.882
Scaffold-GS
0.913
0.740
0.948
0.917
0.850
0.929
0.931 0.766 0.829
0.887
SpeedySplat
0.907
0.747
0.926
0.877
0.815
0.891
0.904 0.764 0.773
0.868
Taming3DGS
0.910
0.706
0.950
0.922
0.856
0.937
0.934 0.738 0.830
0.893
ContraGS-2M
0.925
0.813
0.943
0.938
0.848
0.922
0.933 0.896 0.861
0.904
Table 5. SSIM on evaluation datasets
DeepBlending
MipNerf360
Tanks and Temples
playroom
bicycle bonsai counter garden kitchen room stump
train
truck
3DGS
0.241
0.242
0.180
0.182
0.122
0.116
0.196 0.242 0.197
0.141
EAGLES
0.251
0.244
0.191
0.199
0.154
0.127
0.200 0.243 0.237
0.164
Reduced-GS
0.243
0.244
0.180
0.183
0.123
0.117
0.197 0.243 0.206
0.147
Scaffold-GS
0.244
0.260
0.179
0.185
0.133
0.122
0.187 0.261 0.190
0.136
SpeedySplat
0.269
0.244
0.227
0.258
0.213
0.197
0.257 0.289 0.289
0.190
Taming3DGS
0.251
0.314
0.172
0.171
0.129
0.110
0.181 0.309 0.191
0.125
ContraGS-2M
0.252
0.230
0.186
0.150
0.160
0.148
0.187 0.163 0.187
0.124
Table 6. LPIPS on evaluation datasets
Blender
DeepBlending
MipNerf360
Tanks and Temples
chair
drums
ficus
hotdog
lego
materials
mic
ship
playroom
bicycle
bonsai counter
garden
kitchen
room
stump
train
truck
ContraGS-2M 158.30 127.17 236.28
-
103.08
149.54
257.87 115.33
134.57
130.04
101.38
99.03
139.62
135.20 128.97
131
128.07
96.07
3DGS
190.27 149.03
67.31
70.85
140.61
58.35
83.00
115.99
817.16
2,089.36 482.08 467.02 1,858.58 690.12 553.89 1,772.67 480.17
901.40
EAGLES
10.78
8.44
4.80
6.38
12.35
4.80
5.17
7.36
63.14
159.47
50.91
44.77
116.03
82.74
52.33
156.55
33.40
57.96
Reduced-GS
121.14
96.92
65.94
46.87
85.20
40.21
49.72
68.60
572.14
1,432.41 321.76 295.36 1,406.55 443.20 371.62 1,099.43 269.20
636.50
Scaffold-GS
31.66
31.66
31.66
31.66
31.66
31.66
31.66
31.96
95.11
304.83
135.45
91.52
246.28
107.76
89.25
252.44
114.89
192.21
SpeedySplat
72.30
57.09
43.51
32.94
61.88
32.71
32.45
50.27
393.28
895.20
314.86 264.14 1,043.39 406.80 286.05
803.08
188.92
494.39
Taming3DGS
-
-
-
-
-
-
-
-
325.63
477.60
-
631.04 1,221.08 797.98 756.10
282.02
743.60
1,189.74
Table 7. Peak memory during training

<!-- page 14 -->
C. Random Parameter Initialization
Tanks and Temples
MipNerf360
DeepBlending
Blender
train
truck
bicycle bonsai counter garden kitchen room stump
playroom
chair drums ficus hotdog
lego
materials
mic
3DGS
19.49
18.41
17.69
16.46
23.46
21.71
24.61
27.69 20.74
14.41
31.93 24.91 29.05
36.48
32.37
29.69
34.58
EAGLES
18.91
18.43
19.30
17.79
23.25
24.82
25.52
25.27 20.08
20.82
34.42 25.68 33.67
37.05
34.92
28.95
35.12
Reduced-GS
18.74
18.02
15.59
23.85
24.15
25.82
25.11 19.67
12.95
35.59 26.28 35.48
38.07
36.06
30.50
36.71
Scaffold-GS
19.69
18.32
20.58
17.77
21.06
18.45
23.08
23.35 18.73
13.99
34.85 26.17 35.04
37.81
35.42
30.60
36.69
SpeedySplat
19.24
18.15
15.90
19.54
23.84
0.00
0.00
24.40
0.00
13.39
33.95 25.96 35.18
36.13
32.12
29.33
35.86
Taming3DGS 19.78
18.46
19.64
20.08
-
-
26.73
-
20.86
-
-
-
-
-
-
-
-
ContraGS-2M 24.35
26.93
27.02
31.59
30.60
27.49
30.54
32.34 30.83
33.73
38.48 29.05 38.79
36.48
39.34
36.20
41.56
Table 8. PSNR measured on evaluation dataset
Tanks and Temples
MipNerf360
DeepBlending
Blender
train
truck
bicycle bonsai counter garden kitchen room stump
playroom
chair drums ficus hotdog
lego
materials
mic
3DGS
0.737
0.722
0.487
0.658
0.827
0.710
0.878
0.870 0.606
0.682
0.983 0.941 0.953
0.984
0.975
0.950
0.987
EAGLES
0.726
0.719
0.562
0.690
0.835
0.784
0.890
0.846 0.600
0.808
0.984 0.950 0.982
0.983
0.980
0.949
0.989
Reduced-GS
0.717
0.712
-
0.620
0.840
0.762
0.891
0.838 0.567
0.646
0.988 0.955 0.987
0.985
0.983
0.960
0.992
Scaffold-GS
0.727
0.696
0.467
0.697
0.780
0.455
0.830
0.803 0.396
0.669
0.985 0.948 0.985
0.984
0.980
0.960
0.992
SpeedySplat
0.706
0.701
0.422
0.713
0.816
-
-
0.817
-
0.657
0.979 0.949 0.985
0.975
0.958
0.947
0.990
Taming3DGS 0.743
0.722
0.525
0.735
-
-
0.895
0.000 0.588
-
-
-
-
-
-
-
-
ContraGS-2M 0.861
0.904
0.813
0.943
0.938
0.848
0.922
0.933 0.896
0.925
0.993 0.977 0.994
0.984
0.992
0.989
0.997
Table 9. SSIM measured on evaluation dataset
Tanks and Temples
MipNerf360
DeepBlending
Blender
train
truck
bicycle bonsai counter garden kitchen room stump
playroom
chair drums ficus hotdog
lego
materials
mic
3DGS
0.286
0.291
0.481
0.469
0.289
0.251
0.182
0.279 0.398
0.549
0.023 0.059 0.043
0.030
0.031
0.064
0.026
EAGLES
0.306
0.305
0.422
0.443
0.273
0.203
0.174
0.312 0.406
0.387
0.015 0.045 0.018
0.025
0.020
0.053
0.011
Reduced-GS
0.307
0.308
-
0.498
0.277
0.219
0.168
0.320 0.441
0.308
0.010 0.036 0.012
0.020
0.016
0.037
0.006
Scaffold-GS
0.291
0.317
0.505
0.456
0.353
0.508
0.262
0.368 0.571
0.573
0.014 0.047 0.014
0.023
0.019
0.041
0.008
SpeedySplat
0.349
0.359
0.567
0.427
0.327
-
-
0.352 0.000
0.581
0.023 0.048 0.014
0.044
0.060
0.061
0.011
Taming3DGS 0.282
0.292
0.454
0.404
-
-
0.161
-
0.406
-
-
-
-
-
-
-
-
ContraGS-2M 0.187
0.124
0.230
0.186
0.150
0.160
0.148
0.187 0.163
0.252
0.007 0.026 0.007
0.030
0.009
0.019
0.003
Table 10. LPIPS measured on evaluation dataset

<!-- page 15 -->
D. Training and Rendering Speeds
T and T
MipNerf360
DeepBlending
train
truck
bicycle bonsai counter garden kitchen
room
stump
playroom
Taming3DGS 263.40 240.66 224.62 211.51 170.02 178.84 163.10 174.79 374.40
468.09
Ours 5M
129.00
96.00
89.00
58.00
46.00
118.00 101.00 107.00
88.00
141.00
Ours
206.00 209.00 159.00 217.00 219.00 249.19 210.00 242.00 220.00
319.00
MCMC 5M
75.50
85.90
72.00
68.50
56.10
91.70
59.49
71.24
41.00
114.00
MCMC 2M
94.00
173.32
77.00
75.69
62.78
185.00
70.16
143.58 142.79
251.8
Table 11. Frames per second (FPS) measured on differnt datasets
DeepBlending
MipNerf360
T and T
playroom
bicycle bonsai counter garden kitchen room
stump
train
truck
MCMC 2M
31.66
24.00
24.00
20.00
27.00
22.87
23.70
24.00
29.19 31.49
MCMC 5M
13.10
11.99
11.99
10.09
13.08
10.81
11.28
12.76
12.25 13.65
Ours
46.33
31.63
27.07
32.82
36.33
29.48
34.80
32.00
38.50 43.00
Ours 5M
21.65
15.09
13.55
11.59
17.80
13.14
17.45
17.33
19.87 18.48
Taming3DGS
93.12
71.35
50.09
48.59
40.02
40.70
49.09 102.78 67.08 47.11
Table 12. Training Iterations per second
D.1. Comparison of Training Speeds with Taming-GS at Different Gaussian Counts
Table 13 compares training speeds with TamingGS.
Model-numGS
PSNR Training time Memory
ContraGS-2M
30.06
14 mins
130 MB
ContraGS-530K 29.31
6.5 mins
59 MB
TamingGS
29.39
8 mins
477 MB
Table 13. Performance comparison of TamingGS and ContraGS for the same number of Gaussians

<!-- page 16 -->
E. Qualitative Results
Ground Truth
Ours
Bicycle
Stump
Train
Room
Figure 8. Qualitative results of ContraGS compared to ground truth reconstruction
