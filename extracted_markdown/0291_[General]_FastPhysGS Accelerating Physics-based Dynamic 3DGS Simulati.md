<!-- page 1 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via
Interior Completion and Adaptive Optimization
Yikun Ma 1 Yiqing Li 1 Jingwen Ye 2 Zhongkai Wu 2 Weidong Zhang 2 Lin Gao 3 4 Zhi Jin 1 5 6
Abstract
Extending 3D Gaussian Splatting (3DGS) to 4D
physical simulation remains challenging. Based
on the Material Point Method (MPM), existing
methods either rely on manual parameter tuning
or distill dynamics from video diffusion models,
limiting the generalization and optimization effi-
ciency. Recent attempts using LLMs/VLMs suffer
from a text/image-to-3D perceptual gap, yielding
unstable physics behavior. In addition, they often
ignore the surface structure of 3DGS, leading to
implausible motion. We propose FastPhysGS, a
fast and robust framework for physics-based dy-
namic 3DGS simulation: (1) Instance-aware Parti-
cle Filling (IPF) with Monte Carlo Importance
Sampling (MCIS) to efficiently populate inte-
rior particles while preserving geometric fidelity;
(2) Bidirectional Graph Decoupling Optimiza-
tion (BGDO), an adaptive strategy that rapidly
optimizes material parameters predicted from a
VLM. Experiments show FastPhysGS achieves
high-fidelity physical simulation in 1 minute us-
ing only 7 GB runtime memory, outperforming
prior works with broad potential applications.
1. Introduction
Recently, remarkable progress in 3D reconstruction (Milden-
hall et al., 2020; Kerbl et al., 2023; Wang et al., 2025) and
generation (Poole et al., 2023; Wang et al., 2023; H¨ollein
1School of Intelligent Systems Engineering, Shenzhen Cam-
pus of Sun Yat-sen University, Shenzhen, Guangdong 518107,
P.R.China.
2Anonymous Institution.
3Institute of Computing
Technology, Chinese Academy of Sciences, Beijing 100190,
China.
4University of Chinese Academy of Sciences, Beijing
101408, China.
5Guangdong Provincial Key Laboratory of
Fire Science and Intelligent Emergency Technology, Shenzhen
518107, P.R.China.
6Guangdong Provincial Key Laboratory
of Robotics and Digital Intelligent Manufacturing Technology,
Guangzhou 510535, PR China.. Correspondence to: Zhi Jin
<jinzh26@mail.sysu.edu.cn>.
Preprint. February 3, 2026.
Input
Frames
Omni Dream
3D
Cog
PG Ours
Speed (min)
CLIP Score
10G 20G 
40G 
0.260
0.300
1
0.270
0.280
0.290
120
90
60
30
Ours
OmniPhysGS
ICLR’25
DreamPhysics
AAAI’25
Physics3D
arXiv’24
CogVideoX
ICLR’25
PhysGaussian
CVPR’24
Memory 
Aesthetic Score
Speed (min)
4.00
4.80
1
4.20
4.40
4.60
120
90
60
30
Ours
10G 20G 
40G 
Memory 
PhysGaussian
CVPR’24
CogVideoX
ICLR’25
DreamPhysics
AAAI’25
Physics3D
arXiv’24
OmniPhysGS
ICLR’25
Figure 1. We propose FastPhysGS, an efficient and robust physics-
based dynamic 3DGS simulation framework. Input a 3DGS scene,
our method requires only 7 GB of memory and completes com-
plex dynamic simulations within 1 minute, making it practical to
generate real-time 4D physics-aware dynamics.
et al., 2023; Ma et al., 2024) has significantly advanced
applications in gaming, virtual reality, and robotics. How-
ever, generating realistic, dynamically consistent, and spatio-
temporally coherent 4D contents remain challenging. With
the progress in video generation models (Brooks et al., 2024;
Yang et al., 2025; Kong et al., 2024), several studies have
explored constructing 4D content that reflects real-world dy-
namics (Ren et al., 2023; Zeng et al., 2024; Ren et al., 2024).
These methods typically leverage dynamic priors derived
from video diffusion models to predict 4D deformations and
subsequently reconstruct spatio-temporal scenes. Neverthe-
less, they inherently lack explicit physical constraints, often
resulting in motions that violate physical laws and exhibit
spatio-temporal inconsistencies of 4D content.
To address these issues, PhysGaussian (Xie et al., 2024) first
integrates the Material Point Method (MPM) (Stomakhin
et al., 2013) with 3D Gaussian Splatting (3DGS) (Kerbl
et al., 2023), leveraging an explicit Lagrangian-Eulerian
grid representation to simulate physically plausible dynam-
1
arXiv:2602.01723v1  [cs.CV]  2 Feb 2026

<!-- page 2 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
ics in 3DGS. However, it requires manually specifying the
physical material properties for each scene, which demands
substantial expert knowledge. To mitigate this, subsequent
methods (Liu et al., 2024; Lin et al., 2025; Huang et al.,
2025) adopt Score Distillation Sampling (SDS) (Poole et al.,
2023) to distill material properties from video diffusion mod-
els. However, this way is computationally expensive and
relies on the motion priors from video diffusion models. In
contrast, recent methods (Lin et al., 2024; Zhao et al., 2025;
Chen et al., 2025) employ LLMs/VLMs to predict physical
parameters due to their reasoning capabilities. Nevertheless,
a perceptual gap exists between text/images and native 3D
content, which leads to unstable simulation results. Further-
more, existing methods overlook the hollow characteristic
of 3DGS, resulting in unrealistic physical behaviors.
In summary, existing methods face three key challenges: (1)
Neglecting the hollow 3DGS structure, while lacking the
ability to distinguish instance-level filling. (2) Relying on
manually designed parameters or diffusion distillation, lead-
ing to inadequate generalization and inefficient simulation.
(3) Employing large models to predict physical parameters,
leading to a perception gap with the native 3DGS content.
To address the aforementioned challenges, we propose Fast-
PhysGS with two stages: (1) Considering the spatial struc-
ture of 3DGS, we propose Instance-aware Particle Filling
(IPF). Our approach begins with segmenting the 3DGS ob-
jects, and leverages ray-casting guided by an occupancy
field to estimate initial internal particles. To handle complex
geometries filling, such as irregular surfaces and concave re-
gions, we then design the Monte Carlo Importance Sampling
(MCIS). (2) For adaptive and efficient refinement of physical
parameters predicted by a VLM, we propose Bidirectional
Graph Decoupling Optimization (BGDO). By exploring the
stress-strain mechanics, BGDO leverages stress gradients
and deformation to efficiently optimize parameters. Exten-
sive experiments demonstrate that compared to other phys-
ical dynamic simulation methods, FastPhysGS generates
plausible motions rapidly and robustly. As shown in Figures
1 and 2, our method achieves state-of-the-art results across
a wide range of physical simulations and material types,
only requires 7 GB running memory in 1 minute, which
demonstrates outstanding practicality and broad potential
application. Our main contributions are as follows:
• Real-Time, Memory-Efficient Dynamic 3DGS Sim-
ulation: FastPhysGS establishes a novel paradigm for
physics-based dynamic 3DGS, achieving high-fidelity,
multi-materials simulation with remarkable efficiency
and physical plausibility.
• Instance-aware Complete Geometry Representa-
tion: We propose IPF with MCIS, which explicitly
fills the hollow 3DGS, providing a geometrically com-
Input 3DGS
Rendered Frames
Input
Time
Force
Figure 2. FastPhysGS supports various physical behaviors includ-
ing movement, collision, tearing, rotation, swaying across diverse
materials, such as sand, rubber, jelly, water and elastomers.
plete and stable foundation for physical simulation.
• Perception-to-Physics optimization:
We present
BGDO, a novel adaptive refinement mechanism to
rapidly correct VLM-predicted material parameters.
The rest of this paper is organized as follows: Sec. 2 reviews
the related work. Sec. 3 introduces the preliminaries of
3DGS and MPM. Sec. 4 presents the details of our proposed
method. Sec. 5 provides experimental analysis and results.
Finally, Sec. 6 concludes the paper.
2. Related Work
2.1. 4D Content Generation
4D content creation has grown significantly across various
applications. With the success of video generation models
(Brooks et al., 2024; Yang et al., 2025; Kong et al., 2024),
several methods propose to generate 4D content using video
priors. MAV3D (Singer et al., 2023) first employs tempo-
ral Score Distillation Sampling (SDS) (Poole et al., 2023)
from the text-to-video diffusion model. Animate124 (Zhao
et al., 2023) proposes an image-to-4D coarse-to-fine frame-
work, while DreamGaussian4D (Ren et al., 2023) utilizes
pre-generated videos to supervise the deformation of 3DGS.
However, the diffusion dynamics may be physically inaccu-
rate, and directly applying them leads to inplausible results.
2.2. Physics-Based Dynamic Simulation
To address the issues mentioned in Sec.2.1, recent works
(Xie et al., 2024; Lin et al., 2025; Zhang et al., 2024; Huang
et al., 2025; Liu et al., 2024) introduce physics-based simu-
lation. PhysGaussian (Xie et al., 2024) first coupled 3DGS
2

<!-- page 3 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
(Kerbl et al., 2023) with continuum mechanics simulation
via MPM (Stomakhin et al., 2013). However, it requires
manual configuration of physical parameters for each scene,
causing inconvenient simulation. Subsequent approaches fo-
cus on automating MPM material perception. For instance,
DreamPhysics (Huang et al., 2025) and Physics3D (Liu
et al., 2024) leverage video generation models to estimate
physical material parameters by SDS. However, it is time-
consuming and relies on generative priors. To accelerate
simulation, methods like PhysSplat (Zhao et al., 2025) and
Phys4DGen (Lin et al., 2024) utilize LLMs/VLMs to pre-
dict parameters. However, they rely on the output of large
models, creating a perception gap with the actual 3D envi-
ronment, leading to instable simulation. Moreover, existing
works often overlook the hollow interior of 3DGS, which
distorts the computation of particle stresses. Although Phys-
Gaussian (Xie et al., 2024) introduces particle filling, its
global processing makes it difficult to distinguish object
instances and different materials.
3. Preliminaries
3.1. 3D Gaussian Splatting
3DGS (Kerbl et al., 2023) represents a scene as a collection
of 3D gaussians, each defined by a center µ and a covariance
matrix Σ. The gaussian function at position x is given by:
G(x) = exp

−1
2(x −µ)⊤Σ−1(x −µ)

,
(1)
where the covariance matrix Σ is decomposed as Σ =
RSS⊤R⊤, where S is a diagonal scaling matrix and R is
a rotation matrix. During rendering, view transformation W
is applied, and the 2D projected covariance is computed us-
ing the Jacobian J: Σ′ = JW ΣW ⊤J⊤. The pixel color
Col is computed by blending N overlapping gaussians:
Col =
N
X
i=1
coliαi
Y
j<i
(1 −αj),
(2)
where coli and αi denote the color and opacity of the i-th
gaussian, respectively.
3.2. Material Point Method
MPM (Stomakhin et al., 2013) is a widely used Lagrangian-
Eulerian framework for physics-based simulation of contin-
uum materials. In MPM, each time step consists of three
sequential stages: (1) Particle-to-Grid (P2G), where particle
mass m and momentum are mapped to a background grid;
(2) Grid update, where the discretized momentum equa-
tions are solved under applied internal and external forces
f; (3) Grid-to-Particle (G2P), where updated velocities, po-
sitions, and deformation gradients are interpolated back to
the particles. We adopt MLS-MPM (Hu et al., 2018), a
MPM variant that improves momentum conservation via lo-
cal affine velocity fields, and follow PhysGaussian to define
each gaussian kernel as the time-dependent state:
xi(t) = ∆(xi, t),
Σi(t) = Fi(t)ΣiFi(t)⊤,
(3)
where ∆(·, t) and Fi(t) denote coordinate deformation and
deformation gradient at timestep t.
4. Method
The overall pipeline of FastPhysGS is illustrated in Figure 4,
consisting of two stages: IPF and BGDO. Specifically, IPF
populates the interior 3DGS particles, leveraging MCIS to
handle complex geometry filling. Subsequently, BGDO
employs the bidirectional graph decoupling strategy, which
performs forward MPM simulation and backward optimiza-
tion to refine physical parameters predicted from the VLM.
Details of IPF and BGDO are described in Sec. 4.1 and 4.2.
4.1. Instance-aware Particle Filling
3DGS essentially reconstructs the surface appearance with
empty internal structure. Therefore, it is prone to collapse
during simulation, as the internal stress contribution is mini-
mal. PhysGaussian (Xie et al., 2024) first utilizes the opacity
field to determine internal points by checking whether a ray
passes through grids with different opacities. However, this
global filling makes it difficult to assign distinct materials
to multiple objects (e.g., granting each object independent
physical properties), while lacking reasonable adaptation
for complex structures. For example, Figure 3 shows that
PhysGaussian incorrectly fills the hollow region of the bam-
boo basket, causing the mat sides to be erroneously lifted
upward.
To tackle these problems, we propose IPF to efficiently
populate 3DGS particles with instance-aware capability. As
shown in Figure 4, let the original 3DGS positions P be:
P = {pi ∈R3}N
i=1,
(4)
4、TPO消融——不同E对梯度的影响
PhysGaussian
Ours
Filled Points
Rendering
Input
Figure 3. We extract the filled points as meshes for better visu-
alization. PhysGaussian incorrectly fills the hollow region of the
wicker basket, causing the mat to warp upward, while our method
achieves accurate instance-aware filling.
3

<!-- page 4 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
3
Input
3DGS
Seg. 
Rendered Frames
No Grad MPM Simulation
With Grad 


RayCast
{User Prompt}: 
many pillows fall down 
to the wicker basket.
{Expert Knowledge}:
Below are some common 
material parameters: …
DBSCAN
IPF
QuickHull
{, , }
occ=0.2
occ=0.7

	
AABB
1
Resetting
P2G
G2P
Grid Update
2
3
4
∆(,)

 + 1
 = /
 = ∇∆(, )
 = η g −(1 −η)
Filled 
3DGS

g = 10"
log (1 + g ) = 10

10"&
10&
Outlier
Target
 + 2
∆(,  + 1)
("
)
*)

BGDO-Back
Hulls
Qwen3-VL
+", ,"
+
, ,
BGDO-Forward
LabelTrack
{Output Format}:
Predict the material 
parameters for {Each Objects} 
based on the {User Prompt} 
and {Expert Knowledge} for
the input image.
E:[E1, E2, …]
-:[.1,.2, …]
/:[1,2, …]
0:[11,12, …]
2{,,}
5{,,}
+{,,}
{,,}
Pseudo 
MPM
{g , }
Ori.
3DGS
1~2 Iters
Prompt
MCIS
,", ,6, …, ,
Figure 4. The pipeline of our method. The first stage IPF rapidly fills the interior 3DGS particles, while MCIS is designed to identify
crucial points and handle complex geometries. The second stage BGDO is proposed to adaptively optimize MPM parameters. Overall,
our method generates complete and realistic 4D physical dynamics in 1 minute, showcasing great potential in practical applications.
where pi denotes the mean of each gaussian, and N repre-
sents the total number of gaussians. Specifically, we apply
DBSCAN clustering (Ester et al., 1996) to obtain K object
instances for subsequent instance-aware MPM simulation:
{Ck, lk }K
k=1 = DBSCAN(P, r),
(5)
Ck = {c(k)
j
∈R3},
k = 1, 2, . . . , K,
(6)
where Ck denotes the k-th point cluster, lk indicates the la-
bel of each cluster, r denotes the cluster radius, cj represents
the j-th point within the cluster Ck.
To rapidly obtain the interior gaussian points of each cluster
Ck, we independently compute the convex hull utilizing
Quickhull (Barber et al., 1996): Hk = Quickhull(Ck),
where Hk are the triangular meshes enclosing the points in
Ck. For all possible candidate filling points Qk of each Ck,
we construct AABB bounding boxes Bk from Hk:
Qk = {qj ∼Uni(Bk)}M
j=1 ,
M ≫|Ck|,
(7)
Bk = [b(k)
min, b(k)
max] ⊂R3,
(8)
where Uni represents uniform random sampling within the
boxes, and b denotes the boundary value of the hull Hk.
To efficiently determine whether a candidate point q lies
inside the convex hull Hk, we compute a 3D occupancy
field occ(q; Hk) ∈[0, 1], which employs fast ray-casting
(Levoy, 1990) to coarsely select the interior points based on
occupancy probability threshold (experimentally set to 0.6):
Qinside
k
= {qj ∈Qk | occ(qj; Hk) > 0.6} .
(9)
However, relying solely on the occupancy field results in
incorrect filling for some irregularly shaped objects (e.g.
Figure 8 illustrates the bamboo basket below, whose center
Labels
3DGS
0
(𝒳0, ∑0, 𝑐𝑜𝑙0,𝛼0)
0
(𝒳1, ∑1, 𝑐𝑜𝑙1,𝛼1)
1
(𝒳2, ∑2, 𝑐𝑜𝑙2,𝛼2)
1
(𝒳3, ∑3, 𝑐𝑜𝑙3,𝛼3)
2
(𝒳4, ∑4, 𝑐𝑜𝑙4,𝛼4)
Label的存储占用固定，
每个label记录该点动态情况
3DGS属性值动态变化，
但被追踪的地址不变
Fixed
Variable 3DGS
Ori.
Ori.
Filled
Filled
Figure 5. We use a contiguous memory block to store labels for
tracking the dynamically varying 3DGS properties.
is hollow and should not be filled). To address this, we pro-
pose the Monte Carlo Importance Sampling (MCIS) (Kocsis
& Szepesv´ari, 2006) strategy:
Ep[f] =
Z
f(x) p(x) dx =
Z
f(x) p(x)
q(x) q(x) dx
≈1
n
n
X
j=1
f(xj) · p(xj)
q(xj) = 1
n
n
X
j=1
f(xj) · wj ,
(10)
this is a Monte Carlo expected value of any function f(x)
under distribution p(x). The key insight lies in designing
a non-uniform probability density function p(x) to assign
higher probabilities to important points from Qk. We sam-
ple xj ∼q(x), and provide the detailed importance sam-
pling steps of MCIS:
1) Proximity-aware distance metric. For each interior
candidate point qj ∈Qinside
k
, we compute its minimal
Euclidean distance to the observed surface proxy Ck (e.g.,
a set of anchor gaussians or boundary samples):
d(k)
j
= min
c∈Ck ∥qj −c∥2.
(11)
2) Gaussian-derived importance weighting. Considering
the distribution characteristic of 3DGS, we map distances
4

<!-- page 5 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
into importance weights via an gaussian kernel, which ex-
ponentially attenuates influence with increasing distance:
w(k)
j
= max
 
exp
 
−
(d(k)
j )2
2σ2
!
, ϵ
!
, ϵ = 10−6,
(12)
where σ controls the locality of influence (empirically set to
0.02, see Sec. 5.3 for more details).
3) Normalized importance distribution. The unnormal-
ized weights w(k)
j
are converted into a discrete probability
function over Qinside
k
:
p(k)
j
=
w(k)
j
P|Qinside
k
|
j′=1
w(k)
j′
.
(13)
4) Stochastic sampling. We obtain nk points via multino-
mial sampling according to p(k)
j , yielding a refined subset:
Sk = {s(k)
i
}nk
i=1,
with P(s(k)
i
= qj) = p(k)
j ,
(14)
where P represents the sampling probability, indicating that
each qj is sampled by an importance distribution.
5) Aggregation of filled points. The final outputs are con-
structed by uniting the original P with the sampled points:
Pfilled =
K
[
k=1
Sk,
Pall = P ∪Pfilled.
(15)
6) Appearance suppression for structural fidelity. To
prevent visual artifacts from Pfilled, we nullify their appear-
ance contribution by setting opacity α = 0, retaining only
positional occupancy.
Finally, to consistently retrieve dynamic gaussian parame-
ters from a fixed label memory lk, we propose label tracking
as illustrated in Figure 5. In conclusion, IPF with MCIS fills
the interior 3DGS particles, providing a geometrically com-
plete structure foundation for subsequent MPM simulation.
4.2. Bidirectional Graph Decoupling Optimization
Optimization Target Analysis. To conveniently obtain
the physical parameters for MPM simulation, we employ
Qwen3-VL (Bai et al., 2025) to obtain an initial prediction
{ρ, ν, Φ, E} for each object instance lk. Density ρ mainly
influences the inertial term and has a relatively smooth ef-
fect. Poisson’s ratio ν lies in a narrow range (0 ∼0.5) and
yields stable predictions thanks to rich pre-defined expert
priors provided to the VLM. The energy density model Φ
adopts a pre-defined form (e.g., Fixed Corotated Elasticity
(Stomakhin et al., 2012), StVK Elasticity (Barbiˇc & James,
2005), Drucker-Prager Plasticity (Drucker & Prager, 1952),
and Fluid Plasticity (Stomakhin et al., 2014)), and we clas-
sify it by VLM. Thus, we optimize Young’s modulus E, as
it dominantly governs material stiffness. However, a percep-
tual gap between model reasoning and the native 3D space
leads to unsuitable simulation, necessitating an optimization
method to address this issue.
Specifically, we first derive the computational graph from
E to the rendered frames I:
I ←{X, Σ, α, col}←{x, v, C, F }←τ ←Φ←E, (16)
where {X, Σ, α, col} are the 3DGS parameters, and
{x, v, C, F } represent the MPM particle states (position,
velocity, affine momentum tensor, and deformation gradi-
ent), τ is the first Piola–Kirchhoff stress. Since 3DGS
parameters are directly generated from the MPM simulation,
the rendered frames are merely visual representation of the
underlying physical states. Therefore, we further analyze
the numerical computations in the momentum exchange
stage of MPM:
mt
ivt
i =
X
p
N(xi −xt
p)
h
mpvt
p+

mpCt
p −
4
(∆x)2 ∆tVp
∂Φ
∂F F tT
p

(xi −xt
p)
i
+ f t
i , (17)
where ∂Φ/∂F denotes the particles stress τ, which captures
the material-level forces in MPM. To conveniently control
different material properties for multi-objects, we set the τ
as a variable term changing with different E in K lables:
τ(F , E) =
K
X
k=1
∂Φk(Fk, Ek)
∂Fk
.
(18)
Based on the above analysis, we aim to efficiently optimize
E by leveraging {τ, F } as the physical plausibility signal,
with two primary objectives: (1) Time consumption: avoid
optimization during MPM simulation process, as current
frame is determined by the previous frame, the full computa-
tion is highly time-consuming; (2) Memory accumulation:
exclude simulation process from the computation graph, as
it accumulates gradients across frames, leading to excessive
GPU memory usage and out-of-memory errors.
Optimization Algorithm. Therefore, we propose BGDO
to separate the forward simulation from backward optimiza-
tion. As shown in Figure 4, we first perform a gradient-free
forward MPM with a temporal and causal process, where
subsequent frames are only influenced by prior states. Thus,
we record only three key frames (initial frame f0, an inter-
mediate frame ft, and the final frame fn) with attributes
{x, v, C, F }. This provides effective priors for optimiza-
tion while minimizing storage costs.
Then, we enable gradient computation in backward graph,
and perform a pseudo simulation process, which preserves
5

<!-- page 6 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
only the gradient pathways without actual simulation. To
optimize E using τ and F , we leverage the stress gradient,
while supervising material deformation by measuring the
Frobenius norm (Van Loan & Golub, 1996) as guidance:
gτ = 1
3
X
i∈{0,t,n}
∂∥τi∥
∂Ei
, dF = 1
3
X
i∈{0,t,n}
∥Fi −I∥F .
(19)
However, the stress gradient exhibits an wide dynamic range
(∼108 to 1020), causing numerical instability if used di-
rectly. Thus, we compress its magnitude via a logarithmic
transform g = log(1 + gτ), and the optimization equation
G is formulated as a weighted combination of g and dF :
G = η·g−(1−η)·dF, η = min(1, 0.1·log(1+E)), (20)
where η is the guidance weight, which is proportional to
E. Intuitively, an overly large E amplifies abnormal stress
gradients, increasing the influence of the first term η · g,
while an excessively small E results in softer material and
strengthens the effect of the deformation gradient in the
second term (1 −η) · dF. The balance between these two
signals enables stable and adaptive material optimization.
Finally, the update of Young’s modulus E is defined as:
log(E) ←−log(E) −G .
(21)
Overall, BGDO is formulated from the perspective of nu-
merical computation within the simulation, allowing for
adaptive and rapid rectification of undesirable predicted pa-
rameters in only 1–2 iterations. IPF takes about 22s, BGDO
forward simulation takes 39s, and the backward optimiza-
tion occurs almost instantaneously, as shown in Table 1.
Table 1. The running time and memory across different stages.
The results are averaged over our entire experiment dataset.
Method
IPF
BGDO-Forward
BGDO-Back
Full
Time
22 s
39 s
<1 s
1 min
Memory
1.6 GB
7 GB
7 GB
7 GB
5. Experiment
5.1. Implementation Details
Experimental Setup. Our method is deployed in the Py-
Torch (Paszke et al., 2019) and Taichi (Hu et al., 2019) en-
vironments. We compare our method against four physics-
based 3DGS simulation methods using the same dataset
and hardware: PhysGaussian (Xie et al., 2024), Dream-
Physics(Huang et al., 2025), Physics3D (Liu et al., 2024)
and OmniPhysGS (Lin et al., 2025). Additionally, we also
compare with SOTA video generation model CogVideoX-
5B (Yang et al., 2025), and physics-aware large video model
Veo-3.1 (DeepMind, 2025). All experiments are conducted
on a single NVIDIA RTX A6000 GPU with 48 GB memory,
with each method simulating and rendering 150 frames.
Datasets. We evaluate on three public datasets from (Xie
et al., 2024; Liu et al., 2024; Zhang et al., 2024) with 3DGS
format. We additionally synthesize the multi-objects 3DGS
dataset (e.g., a bear and a can, a duck and a stone). For more
details, please refer to the Appendix.
5.2. Comparison with Baseline Methods
We conduct qualitative and quantitative comparisons against
other physics-based 3DGS simulation approaches and SOTA
video generation models. We compare the rendering results
under identical camera trajectories and force priors. The
motion patterns include translation, rotation, free fall, col-
lision, tearing, and wind swaying, with material properties
spanning rubber, sand, liquid, metal, and elastomers.
Quantitative Comparison. For quantitative evaluation, we
adopt the CLIP Score (CS) (Radford et al., 2021) to measure
the semantic consistency between the frames and prompts,
and the Aesthetic Score (AS) (hristoph Schuhmann., 2022)
to assess the visual fidelity. We further follow VideoPhy2
(Bansal et al., 2025), and adopt Semantic Adherence (SA)
to measure the video-text alignment fidelity, while Physi-
cal Commensense (PC) measures whether videos obey the
physics laws of the real-world. We also compare the capa-
bility of automatic parameter optimization, running mem-
ory consumption, and simulation time. As shown in Ta-
ble 2, compared to the CogVideoX, the physics-aware Veo
achieves higher visual quality with AS 4.90 and physical
realism with PC 0.49, yet it struggles to simulate dynamics
that align with user intention. Regarding dynamic 3DGS
simulation methods, although PhysGaussian achieves low
simulation costs, it requires manual parameter tuning, and
yields unsatisfactory text-image alignment and rendering
quality with CS 0.267 and AS 4.06. In contrast, existing
video distillation-based methods (Physics3D, DreamPhysics
and OmniPhysGS) achieve higher scores in physical real-
ism evaluation with SA 0.81 and PC 0.55, but incur high
memory and computational cost.
By comparison, our FastPhysGS achieves the best perfor-
mance with the lowest computational cost, requiring only 7
GB memory and 1 minute simulation time. These demon-
strate potential values for interactive performance.
Qualitative Comparison. As shown in Figure 6, given
the same prompt and 3DGS, along with identical motion
patterns and force priors, we compare the rendering results
under the same camera pose. We present results for single-
object, multi-objects, and scene-level scenarios, respectively.
CogVideoX struggles to produce physically plausible dy-
namics, while Veo generates high-fidelity results, it still
differs from user intention. For example, we aim to gen-
erate a vertical tearing motion of the bread, CogVideoX
and Veo fail to capture this interaction, resulting in incon-
sistent behaviors. Among physics-based 3DGS methods,
6

<!-- page 7 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
Table 2. Quantitative comparisons. The CLIP Score (CS) measures the semantic alignment between text and frames, while the Aesthetic
Score (AS) evaluates visual quality. Semantic Adherence (SA) measures the video-text alignment fidelity, while Physical Commensense
(PC) measures evaluates the compliance with real-world physics. The best results are highlighted in bold.
Method
Pub.
Auto Param.
CS↑
AS↑
SA↑
PC↑
Memory (GB)↓
Time (min)↓
CogVideoX1.5-5B
ICLR’25
-
0.269
4.04
0.44
0.21
33
24
Veo-3.1
Google’25
-
0.291
4.90
0.81
0.49
-
-
PhysGaussian
CVPR’24
✗
0.267
4.06
0.69
0.55
12
2
Physics3D
arXiv’24
✔
0.275
4.48
0.81
0.53
30
90
DreamPhysics
AAAI’25
✔
0.279
4.38
0.75
0.51
35
90
OmniPhysGS
ICLR’25
✔
0.265
4.35
0.63
0.45
40
120
Ours
-
✔
0.292
4.71
0.87
0.61
7
1
Ours
Phys-
Gaussian
Physics3D
Dream-
Physics
CogVideoX
Omni-
PhysGS
Input
Frame 10
Frame 50
Frame 100
“A toy bear crashes
into a metal can”
“The plane 
blades spinning”
Frame 10
Frame 50
Frame 100
Frame 10 Frame 50 Frame 100
“The bread is tearing 
apart vertically”
Veo-3.1
Force
Figure 6. Visual comparisons between FastPhysGS and other physics-based 3DGS-MPM simulation and video generation methods. Our
method achieves better dynamic visual performance while exhibiting the lowest simulation memory and the fastest execution speed.
OmniPhysGS lacks effective interior filling, leading to un-
realistic deformation during collision (e.g., the bear hits a
metal can but the can exhibits unnatural collapse). Other
manual-designed or distillation-based methods face chal-
lenges in achieving accurate instance-level simulation. For
example, when simulating the rotation of aircraft blades, the
static fuselage erroneously deforms.
In comparison, our method benefits from IPF, providing
a instance-level geometric foundation. BGDO allows for
correcting initial parameters from VLM with better stability.
5.3. Ablation Study
We conduct ablation experiments to evaluate the effective-
ness of our proposed modules:
W/o IPF. As shown in Figure 7(a) and Table 3, missing
particle filling causes internal collapse of 3DGS, leading to
visually implausible results. By contrast, our IPF achieves
7

<!-- page 8 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
(a) W/o IPF
(b) W/o BGDO
(c) Ours
Input
Frame 58
Frame 73
Input
Frame 49
Frame 62
Input
Frame 25
Frame 40
 = 3e10
 = 3e8
 = 3e1
Force
Figure 7. Visual comparisons of our ablation studies of FastPhysGS.
4、MCIS消融
(a) Input
(b) w/o MCIS
(c) w/ MCIS
Figure 8. Visual ablation study of MCIS.
interior filling, providing a geometrically complete 3D struc-
ture for stable and realistic MPM simulation.
W/o MCIS. Figure 8(b) and Table 3 show that without
MCIS, complex geometries like concave surfaces and irreg-
ular curved bodies are challenging to handle, resulting in
inaccurate filling. MCIS mitigates this issue by modeling
importance sampling, achieving correct completion.
σ of MCIS. We analyze the influence of different gaussian
standard deviations σ within MCIS. Specifically, we observe
the results for σ =1 (has no effect), 0.06, 0.04, and 0.02,
as shown in Figure 9. Smaller σ assigns higher weights to
points close to cluster centers and suppresses those farther
4、MCIS消融-标准差
(a) 𝜎= 1.0
(b) 𝜎= 0.06
(c) 𝜎= 0.04
(d) 𝜎= 0.02
Figure 9. Ablation study of standard deviations σ of MCIS.
4、TPO消融——不同E对梯度的影响
异常：E
正常：E
𝜏
𝐸
3𝑒8
5𝑒13
3𝑒2
6𝑒5
3𝑒4
8𝑒6
3𝑒6
3𝑒10
Figure 10. Variation of the particle stress τ with different initial E.
away. As a result, sampling becomes highly focused on
geometrically salient regions, such as boundaries and non-
concave regions. In contrast, larger σ values flatten the
weight distribution, making sampling nearly uniform and
diminishing the role of importance.
W/o BGDO. To assess the robustness of BGDO under cor-
rupted initial parameters, we set the initial Young’s modulus
to 3 × 1010, 3 × 108, and 3 × 101. As shown in Figure 7(b)
and Table 3, these settings lead to varying degrees of simu-
lation failure. In contrast, our BGDO stabilizes the system
within only two iterations, recovering a reasonable Young’s
modulus and thereby enabling robust 3DGS simulation.
Table 3. Quantitative metrics of ablation study. The full model
achieves the best performance, with negligible overhead in running
memory (GB) and time (min).
Method
CS↑
AS↑
SA↑
PC↑
Mem.
Time
w/o IPF
0.263
4.26
0.60
0.44
5
0.8
w/o MCIS
0.285
4.31
0.72
0.52
7
1
w/o BGDO
0.217
3.34
0.57
0.36
7
0.5
Ours
0.292
4.71
0.87
0.61
7
1
Optimization stability of BGDO. To further validate the
stability of BGDO, we evaluate the stress τ under different
initial parameters. As illustrated in Figure 10, we change
only the E while keeping other experiment conditions iden-
tical, and plot over 60 sets of results. It shows that when E
is small, the gradient effect weakens and deformation domi-
nates, while a large E causes a sharp gradient increase. This
phenomenon prevents the optimization from diverging into
8

<!-- page 9 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
false directions, further enhancing the robustness of BGDO.
For more ablation studies of DBSCAN clustering radius r
and material compatibility, please refer to the appendix.
6. Conclusion
We propose FastPhysGS for efficient and robust physics-
based 3DGS simulation. IPF addresses instability caused by
hollow 3DGS structures, while handling complex geometry
filling and diverse object instances. BGDO achieves rapid
material parameters optimization. Experiments show the
superior performance of FastPhysGS, enabling practical
real-time applications and interactive dynamic simulation.
References
Bai, S., Cai, Y., et al. Qwen3-vl technical report. arXiv
preprint arXiv:2511.21631, 2025.
Bansal, H., Peng, C., Bitton, Y., Goldenberg, R., Grover,
A., and Chang, K.-W. Videophy-2: A challenging action-
centric physical commonsense evaluation in video gener-
ation. arXiv preprint arXiv:2503.06800, 2025.
Barber, C. B., Dobkin, D. P., and Huhdanpaa, H. The quick-
hull algorithm for convex hulls. ACM Transactions on
Mathematical Software (TOMS), 22(4):469–483, 1996.
Barbiˇc, J. and James, D. L. Real-time subspace integra-
tion for st. venant-kirchhoff deformable models. ACM
transactions on graphics (TOG), 24(3):982–990, 2005.
Bonet, J. and Wood, R. D. Nonlinear continuum mechanics
for finite element analysis. Cambridge university press,
1997.
Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y.,
Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman,
E., et al. Video generation models as world simulators.
OpenAI Blog, 2024.
Chen, B., Jiang, H., Liu, S., Gupta, S., Li, Y., Zhao, H., and
Wang, S. Physgen3d: Crafting a miniature interactive
world from a single image. CVPR, 2025.
DeepMind, G. Veo 3, 2025. URL https://deepmind.
google/en/models/veo/.
Drucker, D. C. and Prager, W. Soil mechanics and plastic
analysis or limit design. Quarterly of applied mathemat-
ics, 10(2):157–165, 1952.
Ester, M., Kriegel, H.-P., Sander, J., Xu, X., et al. A density-
based algorithm for discovering clusters in large spatial
databases with noise. In kdd, volume 96, pp. 226–231,
1996.
H¨ollein, L., Cao, A., Owens, A., Johnson, J., and Nießner,
M. Text2room: Extracting textured 3d meshes from 2d
text-to-image models. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pp. 7909–
7920, 2023.
hristoph Schuhmann. Laion-aesthetics. In Online, 2022.
Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A., and
Jiang, C. A moving least squares material point method
with displacement discontinuity and two-way rigid body
coupling. ACM Transactions on Graphics (TOG), 37(4):
1–14, 2018.
Hu, Y., Li, T.-M., Anderson, L., Ragan-Kelley, J., and Du-
rand, F. Taichi: a language for high-performance com-
putation on spatially sparse data structures. ACM Trans.
Graph., 2019.
Huang, T., Zhang, H., Zeng, Y., Zhang, Z., Li, H., Zuo, W.,
and Lau, R. W. Dreamphysics: Learning physics-based
3d dynamics with video diffusion priors. In Proceed-
ings of the AAAI Conference on Artificial Intelligence,
volume 39, pp. 3733–3741, 2025.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis, G. 3d
gaussian splatting for real-time radiance field rendering.
ACM Trans. Graph., 42(4):139–1, 2023.
Kl´ar, G., Gast, T., Pradhana, A., Fu, C., Schroeder, C., Jiang,
C., and Teran, J. Drucker-prager elastoplasticity for sand
animation. ACM Transactions on Graphics (TOG), 35(4):
1–12, 2016.
Kocsis, L. and Szepesv´ari, C. Bandit based monte-carlo
planning. In European conference on machine learning,
pp. 282–293. Springer, 2006.
Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J.,
Xiong, J., Li, X., Wu, B., Zhang, J., et al. Hunyuan-
video: A systematic framework for large video generative
models. arXiv preprint arXiv:2412.03603, 2024.
Levoy, M. Display of surface from volume data. IEEE
Computer Graphics and Applications, 8(3), 1990.
Lin, J., Wang, Z., Xu, D., Jiang, S., Gong, Y., and Jiang,
M. Phys4dgen: Physics-compliant 4d generation with
multi-material composition perception. arXiv preprint
arXiv:2411.16800, 2024.
Lin, Y., Lin, C., Xu, J., and MU, Y. Omniphysgs: 3d con-
stitutive gaussians for general physics-based dynamics
generation. In The Thirteenth International Conference
on Learning Representations, 2025.
Liu, F., Wang, H., Yao, S., Zhang, S., Zhou, J., and Duan, Y.
Physics3d: Learning physical properties of 3d gaussians
via video diffusion. NeurIPS, 2024.
9

<!-- page 10 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
Ma, Y., Zhan, D., and Jin, Z. Fastscene: text-driven fast 3d
indoor scene generation via panoramic gaussian splatting.
In Proceedings of the Thirty-Third International Joint
Conference on Artificial Intelligence, pp. 1173–1181,
2024.
Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T.,
Ramamoorthi, R., and Ng, R. Nerf: Representing scenes
as neural radiance fields for view synthesis. In European
Conference on Computer Vision, pp. 405–421. Springer,
2020.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J.,
Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,
L., Desmaison, A., K¨opf, A., Yang, E., DeVito, Z., Rai-
son, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang,
L., Bai, J., and Chintala, S. Pytorch: An imperative style,
high-performance deep learning library. 2019.
Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. Dream-
fusion: Text-to-3d using 2d diffusion. In The Eleventh
International Conference on Learning Representations,
2023.
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark,
J., Krueger, G., and Sutskever, I. Learning transferable
visual models from natural language supervision.
In
Proceedings of the 38th International Conference on Ma-
chine Learning, volume 139, pp. 8748–8763, 2021.
Ren, J., Pan, L., Tang, J., Zhang, C., Cao, A., Zeng, G.,
and Liu, Z. Dreamgaussian4d: Generative 4d gaussian
splatting. arXiv preprint arXiv:2312.17142, 2023.
Ren, J., Xie, C., Mirzaei, A., Kreis, K., Liu, Z., Torralba,
A., Fidler, S., Kim, S. W., Ling, H., et al. L4gm: Large
4d gaussian reconstruction model. Advances in Neural
Information Processing Systems, 37:56828–56858, 2024.
Singer, U., Sheynin, S., Polyak, A., Ashual, O., Makarov,
I., Kokkinos, F., Goyal, N., Vedaldi, A., Parikh, D., John-
son, J., et al. Text-to-4d dynamic scene generation. In
Proceedings of the 40th International Conference on Ma-
chine Learning, pp. 31915–31929, 2023.
Stomakhin, A., Howes, R., Schroeder, C. A., and Teran,
J. M. Energetically consistent invertible elasticity. In
Symposium on Computer Animation, volume 1, 2012.
Stomakhin, A., Schroeder, C., Chai, L., Teran, J., and Selle,
A. A material point method for snow simulation. ACM
Transactions on Graphics (TOG), 32(4):1–10, 2013.
Stomakhin, A., Schroeder, C., Jiang, C., Chai, L., Teran,
J., and Selle, A. Augmented mpm for phase-change and
varied materials. ACM Transactions on Graphics (TOG),
33(4):1–11, 2014.
Van Loan, C. F. and Golub, G. Matrix computations (johns
hopkins studies in mathematical sciences). Matrix Com-
putations, 5:32, 1996.
Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht,
C., and Novotny, D. Vggt: Visual geometry grounded
transformer. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pp. 5294–5306, 2025.
Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu,
J. Prolificdreamer: High-fidelity and diverse text-to-3d
generation with variational score distillation. Advances
in neural information processing systems, 36:8406–8441,
2023.
Xie, T., Zong, Z., Qiu, Y., Li, X., Feng, Y., Yang, Y., and
Jiang, C. Physgaussian: Physics-integrated 3d gaussians
for generative dynamics. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pp. 4389–4398, 2024.
Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu,
J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.
Cogvideox: Text-to-video diffusion models with an ex-
pert transformer. In The Thirteenth International Confer-
ence on Learning Representations, 2025.
Zeng, Y., Jiang, Y., Zhu, S., Lu, Y., Lin, Y., Zhu, H., Hu, W.,
Cao, X., and Yao, Y. Stag4d: Spatial-temporal anchored
generative 4d gaussians. In European Conference on
Computer Vision, pp. 163–179. Springer, 2024.
Zhang, T., Yu, H.-X., Wu, R., Feng, B. Y., Zheng, C.,
Snavely, N., Wu, J., and Freeman, W. T. Physdreamer:
Physics-based interaction with 3d objects via video gen-
eration. In European Conference on Computer Vision, pp.
388–406. Springer, 2024.
Zhao, H., Wang, H., Zhao, X., Fei, H., Wang, H., Long, C.,
and Zou, H. Efficient physics simulation for 3d scenes
via mllm-guided gaussian splatting. ICCV, 2025.
Zhao, Y., Yan, Z., Xie, E., Hong, L., Li, Z., and Lee, G. H.
Animate124: Animating one image to 4d dynamic scene.
arXiv preprint arXiv:2311.14603, 2023.
10

<!-- page 11 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
A. Material Point Method
A.1. MPM Algorithm
The Material Point Method (MPM) simulates the materials behavior by discretizing the continuum into particles and
updating their properties in a hybrid Eulerian-Lagrangian way. MPM naturally handles large deformation and complex
interactions with high robustness. The algorithm is summarized as follows:
Particle to Grid Transfer. In this step, mass m and momentum mv are transferred from particles to grid nodes by
distributing the particle properties to nearby points. The accumulated mass and momentum at grid node i at time step n are
computed as:
mn
i =
X
p
wn
ipmp,
(22)
mn
i vn
i =
X
p
wn
ipmp
 vn
p + Cn
p(xi −xn
p)

,
(23)
where wn
ip denotes the weighting function between particle p and grid point i, mp is the particle mass, vn
p is the particle
velocity, xn
p is the particle position, and Cn
p represents the local gradient correction term.
Grid Update. The grid velocities are updated based on external force and interaction with neighboring particles:
vn+1
i
= vn
i −∆t
mi
X
p
τ n
p ∇wn
ipV 0
p + ∆tf,
(24)
where ∆t is the time step, τ n
p is the stress tensor at particle p, V 0
p is the reference volume, f is the external force, and mi is
the total mass accumulated at grid node i.
Grid to Particle Transfer. Velocities are interpolated back from the grid to the particles, updating their states:
vn+1
p
=
X
i
vn+1
i
wn
ip,
(25)
xn+1
p
= xn
p + ∆tvn+1
p
,
(26)
Cn+1
p
=
4
∆x2
X
i
wn
ipvn+1
i
(xn
i −xn
p)T ,
(27)
∇vn+1
p
=
X
i
vn+1
i
∇wn
ip
T ,
(28)
τ n+1
p
= τ(Fn+1
E
, Fn+1
N
),
(29)
where ∆x is the Eulerian grid spacing, FE and FN represent the elastic and normal deformation gradients, respectively, and
τ is the constitutive stress model.
A.2. Energy Constitutive Models
We integrate expert-designed constitutive models, which describe several representative materials including elasticity (e.g.,
rubber, branches, and cloth), plasticity (e.g., snow, metal, and clay), viscoelasticity (e.g., honey and mud), and fluidity (e.g.,
water, oil, and lava).
Fixed Corotated Elasticity
Following Stomakhin et al. (2012), we define fixed corotated elasticity as:
P(F) = 2µ(F −R) + λJ(J −1)F−T ,
(30)
where R is the rotation matrix obtained from the polar decomposition F = RS, J = det(F) is the volume change, and µ,
λ are the Lam´e parameters. This model is well-suited for simulating rubber-like materials due to its ability to capture large
rotations with minimal shear distortion.
11

<!-- page 12 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
Neo-Hookean Elasticity
Based on (Bonet & Wood, 1997), the Neo-Hookean elasticity is defined as:
P(F) = µ
 F −F−T 
+ λ log(J) F−T ,
(31)
this formulation assumes isotropic hyperelastic behavior and is particularly effective in modeling spring-like elastic responses,
such as those seen in soft tissues or idealized linear elastic solids.
StVK Elasticity
We adopt the St. Venant-Kirchhoff (StVK) model (Barbiˇc & James, 2005), which is given by:
P(F) = U
 2µΣ−1 ln Σ + λ tr(ln Σ) Σ−1
VT ,
(32)
where F = UΣVT is the singular value decomposition (SVD) of the deformation gradient. The matrices U, Σ, and V
represent the left stretch, singular values, and right rotation, respectively. The StVK model captures both elastic and plastic
behaviors and is suitable for simulating materials such as sand and metals, especially when combined with plasticity models.
Identity Plasticity
The identity plasticity model assumes no plastic deformation, meaning the material behaves purely elastically. This is
commonly used for idealized elastic materials:
ψ(F) = F,
(33)
this model serves as a baseline for comparison with more complex plasticity formulations.
Drucker-Prager Plasticity
Following (Drucker & Prager, 1952; Kl´ar et al., 2016), we define Drucker-Prager plasticity using a return mapping based
on the singular value decomposition (SVD) of the deformation gradient. Let F = UΣVT , where Σ contains the singular
values. Define ϵ = log(Σ), which represents the logarithmic strain. The return mapping is then given by:
ψ(F) = UZ(Σ)VT ,
(34)
Z(Σ) =







I,
if P(ϵ) > 0,
Σ,
if δγ ≤0 and P(ϵ) ≤0,
exp

ϵ −δγ
ˆϵ
∥ˆϵ∥

,
otherwise,
(35)
where δγ is the plastic update parameter, ˆϵ is the deviatoric part of ϵ, and P(ϵ) denotes the sum of the diagonal entries of ϵ.
This model is particularly suitable for simulating granular materials such as snow and sand, due to its ability to capture
dilatancy and yield under pressure.
von Mises Plasticity
Based on (Hu et al., 2018), von Mises plasticity is defined through a similar SVD-based return mapping:
ψ(F) = UZ(Σ)VT ,
(36)
Z(Σ) =
(
Σ,
if δγ ≤0,
exp

ϵ −δγ
ˆϵ
∥ˆϵ∥

,
otherwise.
(37)
this formulation captures isotropic yielding and is widely used to simulate ductile materials such as metals and clay, where
plastic flow occurs under shear stress.
Fluid Plasticity
We adopt the fluid plasticity model from Stomakhin et al. (2014), which treats the material as incompressible and fluid-like.
The return mapping is defined as:
12

<!-- page 13 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
ψ(F) = J1/3I,
(38)
where J = det(F) is the volume ratio. This model effectively removes all elastic deformation and enforces a uniform
deformation state, making it suitable for simulating highly fluid-like materials such as water and lava.
B. More Method Details of FastPhysGS
B.1. Elasticity Models and Stress Sensitivity to E
All elasticity models in our framework compute the first Piola-Kirchhoff stress τ = ∂Φ/∂F from the deformation gradient
F ∈R3×3, where Φ is the strain energy density.
Crucially, each model expresses τ as a linear function of the Lam´e parameters:
µ =
E
2(1 + ν),
λ =
Eν
(1 + ν)(1 −2ν),
(39)
which are themselves linear in Young’s modulus E. This linearity implies that the stress scales approximately linearly with
E, i.e., τ ∝E, making ∥τ∥a natural proxy for material stiffness.
We implement five models:
• SigmaElasticity (Hencky/Logarithmic Strain): Uses principal logarithmic strains ε = log(Σ) , where F = UΣV⊤.
The stress is:
τ = U diag(2µε + λ tr(ε)1)U⊤.
(40)
This model is energetically consistent and well-suited for large deformations.
• CorotatedElasticity: Removes rigid rotation via polar decomposition R = UV⊤:
τ = 2µ(F −R)F⊤+ λJ(J −1)I,
(41)
where J = det(F) . The deviatoric term captures shear, while the volumetric term enforces near-incompressibility.
• StVKElasticity: Based on Green–Lagrange strain E = 1
2(F⊤F −I) :
τ = 2µFE + λJ(J −1)I.
(42)
Accurate for moderate strains but may stiffen unrealistically under large compression.
• FluidElasticity: Sets µ = 0 , yielding purely volumetric response:
τ = λJ(J −1)I.
(43)
Suitable for liquids or highly dissipative materials.
• VolumeElasticity: Uses Mie–Gr¨uneisen equation of state:
τ = κ
 J −J−γ+1
I,
κ = 2
3µ + λ, γ = 2.
(44)
Better stability under extreme compression.
In all cases, ∂τ/∂E exists and is non-zero, enabling gradient-based optimization of E . However, numerical instabili-
ties—especially near J ≈0 or during plastic yielding—cause ∥∇E∥τ∥∥to span many orders of magnitude ( 108 – 1020 ),
necessitating robust normalization.
13

<!-- page 14 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
B.2. Plasticity Models and Their Impact on Optimization
Plasticity modifies F after the elastic stress computation, thereby indirectly affecting the stress-deformation relationship
used in BGDO. We support four models:
• IdentityPlasticity: No modification; purely elastic.
• SigmaPlasticity: Enforces unilateral incompressibility by clamping J = det(F) ∈[0.05, 1.2] and resetting F = J1/3I.
This suppresses unrealistic expansion and stabilizes simulation, but reduces sensitivity to E in highly compressed
regions.
• VonMisesPlasticity: Yields when deviatoric strain exceeds yield stress σy :
if ∥εdev∥> σy
2µ, then project ε onto yield surface.
(45)
Since µ ∝E , higher E raises the yield threshold, making the material appear more elastic—this nonlinearity is
captured by BGDO’s stress gradient signal.
• DruckerPragerPlasticity: Pressure-dependent yielding:
f(τ) = ∥τdev∥+ α tr(τ) −c ≤0,
(46)
where α depends on friction angle ϕ , and c is cohesion. Compression increases resistance to yielding. Because
tr(τ) ∝E , the yield condition itself depends on E , creating a complex but differentiable coupling that BGDO
exploits.
Critically, all plasticity corrections are applied after stress computation, so the gradient ∇E∥τ∥remains well-defined
and reflects pre-plastic elastic behavior—exactly the signal needed for stiffness calibration.
B.3. Deformation Guidance via Frobenius Norm
The Frobenius norm of the post-step deformation gradient deviation,
δ = ∥F′ −I∥F =
v
u
u
t
3
X
i=1
3
X
j=1
 F ′
ij −δij
2,
(47)
quantifies the magnitude of local strain accumulated over a single MPM substep, where F′ is the updated deformation
gradient after one explicit integration step and δij is the Kronecker delta. Although this measure is not fully rotation-invariant
(unlike invariants of C = F⊤F ), the small time step size ∆t in our simulation ensures that rotational components in F′
are close to identity, i.e., F′ ≈I + ∇u with displacement gradient ∇u small. Consequently, ∥F′ −I∥F approximates the
Euclidean norm of the infinitesimal strain tensor and serves as a practical proxy for visible deformation.
We set a target value δtarget = 0.1 , chosen empirically to balance visual responsiveness and physical plausibility:
• If δ ≪δtarget , the material exhibits negligible motion—indicating excessive stiffness—so we reduce the penalty on
softness (effectively encouraging a decrease in E );
• If δ ≫δtarget , the material may sag, collapse, or exhibit numerical instability—suggesting insufficient stiffness—so we
increase E .
This deformation-based signal resolves an inherent ambiguity in stress-only optimization: two materials with different E can
produce similar stress magnitudes under static loading but exhibit drastically different dynamic responses. By incorporating
δ , BGDO aligns parameter calibration with perceptual motion cues.
14

<!-- page 15 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
B.4. Gradient Compression and Log-Space Update
The raw gradient of the stress norm with respect to the log-scale modulus,
gτ = ∇log E∥τ∥= ∂∥τ∥
∂log E ,
(48)
exhibits extreme dynamic range due to the nonlinear dependence of τ on F and the exponential mapping E = exp(log E).
In regions near element inversion ( J = det(F) →0 ) or plastic yield, ∥τ∥can change by orders of magnitude with
tiny changes in E, leading to ∥gτ∥∈[108, 1020] in practice. Direct use of such gradients causes optimizer divergence or
dominance by outlier particles.
To stabilize training, we apply a smooth, monotonically increasing compression function:
˜gτ = log(1 + |gτ|).
(49)
This transformation:
• Compresses the dynamic range;
• Preserves gradient sign and monotonicity—larger stress sensitivity still yields larger update magnitude;
• Reduces the influence of extreme outliers during spatial averaging across particles or temporal averaging across key
frames.
The final update operates in log-space:
log E ←log E −G,
where G =
1
|K|
X
k∈K
h
η · ˜g(k)
τ
−(1 −η) · δ(k)i
,
(50)
with key frame set K, and balancing weight η ∈[0, 1]. Updating log E guarantees E = exp(log E) > 0 and induces
multiplicative updates on E , which are more natural for scale parameters than additive ones. This formulation enables
robust convergence within 1–2 iterations while using only three stored simulation snapshots, achieving minimal memory
overhead.
B.5. Algorithm and Limitation
The BGDO optimization flow, summarized in Algorithm 1, decouples the causal forward simulation from the non-causal
parameter update. By restricting gradient computation to three key frames and leveraging compressed stress gradients with
Frobenius-norm deformation guidance, it achieves stable and memory-efficient material calibration.
Limitation For highly fine-grained complex scenes, interior filling may be limited by segmentation accuracy, thereby
affecting material precision. Adopting a large segmentation model could alleviate this issue. In the future, we will focus on
more general representations and physical interactions.
C. More Experimental Results
C.1. More Setup Details
To synthesize the multi-objects dataset, we collect public single-object datasets and then modify the underlying 3DGS code
to enable the fusion of single-object scenes. In MPM, spatial normalization is applied to a 1 × 1 × 1 volume, and the object
center is placed at (0.5, 0.5, 0.5). The simulation runs for 150 frames, the prompt is generated by a large language model,
and all other parameters, such as camera poses and viewing angles are determined by the input 3DGS scene.
C.2. Ablation Study of DBSCAN
To more intuitively illustrate the impact of different radius values on DBSCAN clustering, we visualize the segmentation
results obtained with radii ranging from 0.01 to 0.06 in Figure 11. As can be observed, a radius of 0.01 exhibits almost
no discriminative capability, yielding very poor segmentation. As the radius increases, the segmentation quality gradually
improves. Starting from a radius of 0.05, the segmentation becomes accurate, albeit at the cost of increased computational
time. In practice, we typically select a radius in the range of 0.05 to 0.07.
15

<!-- page 16 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
Algorithm 1 Bidirectional Graph Decoupling Optimization (BGDO)
Require: Initial Young’s modulus log E, material labels {lk}, total frames T, key frames K = {0, t, n}, target deformation
magnitude δtarget, balancing weight w ∈[0, 1]
Ensure: Optimized log E
Stage 1: Gradient-Free Forward Simulation
1: Initialize particle states (x, v, C, F)
2: frame buffer ←∅
3: for frame = 0 to T −1 do
4:
if frame ∈K then
5:
Store snapshot: frame buffer ←frame buffer ∪{(x, v, C, F)}
6:
end if
7:
for substep = 1 to S do
▷S : MPM substeps per frame
8:
τ ←ELASTICITY(F, log E.detach())
9:
(x, v, C, F) ←MPMSTEP(x, v, C, F, τ)
10:
F ←PLASTICITY(F, log E.detach())
11:
end for
12: end for
Stage 2: Gradient-Based Backward Optimization
13: if frame buffer ̸= ∅then
14:
gtotal ←0
15:
for each (xi, vi, Ci, Fi) ∈frame buffer do
16:
Fi ←Fi.requires grad()
17:
τi ←ELASTICITY(Fi, log E)
18:
( , , , F′
i) ←MPMSTEP(xi, vi, Ci, Fi, τi)
19:
F′
i ←PLASTICITY(F′
i, log E)
// Stress gradient signal (prevent over-stiffness)
20:
si ←∥τi∥F
21:
gτ ←∇log E si
22:
˜gτ ←log(1 + |gτ|)
// Deformation guidance (prevent over-softness)
23:
δi ←∥F′
i −I∥F
24:
˜gδ ←δi −δtarget
25:
gtotal ←gtotal + w · ˜gτ −(1 −w) · ˜gδ
26:
end for
27:
log E ←log E −
1
|K|gtotal
28: end if
29: return log E
16

<!-- page 17 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
Table 4. List of Symbols and Their Meanings
Symbol
Type
Meaning
µ
Vector
Center of a 3D Gaussian
Σ
Matrix
Covariance matrix of a 3D Gaussian
Σ′
Matrix
Projected 2D covariance in screen space
αi
Scalar
Opacity of the i -th Gaussian
N
Integer
Number of Gaussians
m
Scalar
Particle mass in MPM
f
Vector
Force vector in MPM
∆(·, t)
Function
Time-dependent spatial deformation map
Fi(t)
Matrix
Deformation gradient of particle i at time t
t
Scalar
Simulation time step
P
Set
Original set of 3DGS particle positions
pi
Vector
Position of i -th original Gaussian ( = µi )
K
Integer
Number of object instances after clustering
Ck
Set
Point cluster for instance k
lk
Integer
Instance identifier for cluster k
r
Scalar
DBSCAN neighborhood radius
c(k)
j
Vector
j -th point in cluster Ck
Hk
Mesh
Convex hull of cluster k
Bk
AABB
Axis-aligned bounding box of Hk
b(k)
min, b(k)
max
Vectors
Min/max corners of AABB for instance k
Qk
Set
Candidate filling points
M
Integer
Number of candidate points
occ(q; Hk)
Scalar
Occupancy probability
Qinside
k
Set
Interior candidates
d(k)
j
Scalar
Minimal Euclidean distance
σ
Scalar
Gaussian kernel scale
ϵ
Scalar
Small constant ( 10−6 ) for numerical stability
w(k)
j
Scalar
Importance weight
p(k)
j
Scalar
Normalized sampling probability
nk
Integer
Number of filled points selected for instance k
Sk
Set
Final sampled interior points for instance k
Pfilled
Set
All filled points
Pall
Set
Full particle set
ρ
Scalar
Material density
ν
Scalar
Poisson’s ratio
Φ
Function
Energy density model
E
Scalar
Young’s modulus
I
Image
Rendered frames
X
Set
3D positions of all Gaussians
v
Vector
Particle velocity in MPM
C
Matrix
Affine velocity field
τ
Tensor
First Piola–Kirchhoff stress
f0, ft, fn
Frames
Key frames (initial, mid, final)
gτ
Scalar
Average stress gradient
dF
Scalar
Average Frobenius norm of F −I
g
Scalar
Log-compressed stress gradient
G
Scalar
Combined optimization signal
η
Scalar
Adaptive weight
17

<!-- page 18 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
C.3. More Visualization Results
We present more visualization results in Figures 12 to 17, showcasing a rich variety of material simulations, such as rubber,
elastomers, sand, liquids, flexible materials, soft bodies, and rigid bodies. These further demonstrate the superiority and
practicality of our FastPhysGS.
r = 0.06
r = 0.01
r = 0.02
r = 0.03
r = 0.04
r = 0.05
Figure 11. Ablation study of different r of DBSCAN.
18

<!-- page 19 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
Original 3DGS
Filled 3DGS by our IPF
Figure 12. More visualized results of our IPF, demonstrating its ability to achieve stable filling across diverse structures.
19

<!-- page 20 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
Frames
Ours
OmniPhysGS
DreamPhysics
Physics3D
PhysGaussian
Veo-3.1
CogVideoX
“a carnation flower is swaying”
Figure 13. More visual comparisons of FastPhysGS and other baseline methods. Given the same prompt and force prior, our method
produces more realistic dynamics. For example, this flower swings to a certain extent and then swings back.
20

<!-- page 21 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
Frames
Ours
OmniPhysGS
DreamPhysics
Physics3D
PhysGaussian
Veo-3.1
CogVideoX
“a rubber duck lands on a stone”
Figure 14. More visual comparisons of FastPhysGS and other baseline methods. Given the same text description and force priors, our
method generates more realistic dynamic effects. For example, the rubber duck lands on a stone and bounces back.
21

<!-- page 22 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
“a sand bear 
collapsing”
“a rubber bear 
falling”
“a sand ficus 
collapsing”
“a ficus swaying
in the wind”
Frames
Figure 15. We further demonstrate the capability to simulate different material properties of FastPhysGS. For instance, in the bear scene,
our method produces bouncy behavior for a rubber material, as well as collapse effects for sand or liquid materials. Meanwhile, ficus
achieves soft-body swinging and collapse.
22

<!-- page 23 -->
FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization
“a soccer ball 
and a basketball 
are bouncing”
“a soccer ball 
collides with 
a basketball”
Frames
“many pillows
fall down to the 
wicker basket”
“a pillow 
collides with 
another pillow”
Figure 16. We further demonstrate the ability to simulate diverse motion priors. For example, it can handle the free fall and multi-collisions
of objects, such as balls and pillows.
“a carnation 
flower is 
swaying”
“a rubber bear 
and a metal can 
are rotating”
Frames
“an alocasia
is swaying”
“a rubber duck 
lands on a 
liquid puddle.”
Figure 17. We further demonstrate the ability to simulate diverse motion priors and material properties of different scenes.
23
