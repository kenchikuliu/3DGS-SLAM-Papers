<!-- page 1 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
1
NuRF: Nudging the Particle Filter in Radiance
Fields for Robot Visual Localization
Wugang Meng, Tianfu Wu, Huan Yin and Fumin Zhang
Abstract—Can we localize a robot on a map only using
monocular vision? This study presents NuRF, an adaptive and
nudged particle filter framework in radiance fields for 6-DoF
robot visual localization. NuRF leverages recent advancements
in radiance fields and visual place recognition. Conventional
visual place recognition meets the challenges of data sparsity
and artifact-induced inaccuracies. By utilizing radiance field-
generated novel views, NuRF enhances visual localization per-
formance and combines coarse global localization with the fine-
grained pose tracking of a particle filter, ensuring continuous
and precise localization. Experimentally, our method converges
7 times faster than existing Monte Carlo-based methods and
achieves localization accuracy within 1 meter, offering an efficient
and resilient solution for indoor visual localization.
Index Terms—Visual localization, Particle filter, Neural radi-
ance fields, Mobile robot
I. INTRODUCTION
V
ISUAL localization on a map is a critical topic for
robot navigation, which is close to how humans perceive
the world. An ideal visual localization system is desired to
accurately localize robots not only globally but also continu-
ously [1]. Existing feature-based techniques, like perspective-
n-point (PnP) [2] offer stable and accurate continuous pose
tracking for robots, they struggle to directly address global
localization in cases of localization failure [3]. On the other
hand, learning-based approaches like visual place recognition
(VPR) [4]–[6] can regress or retrieve poses through trained
networks from scratch. These approaches fall short of pro-
viding a comprehensive and cost-effective solution for contin-
uous 6-degree-of-freedom (6-DoF) pose tracking. There is a
growing need for a unified framework that integrates global
localization and pose tracking for robot visual localization,
enabling the use of one localization method on one map for
two tasks.
Recent advancements in radiance field rendering, partic-
ularly neural radiance fields (NeRF) [7] and 3D Gaussian
splatting (3DGS) [8], have shown significant potential in
overcoming the limitations of traditional visual localization.
Radiance field-based maps can enhance visual-based local-
ization by providing dense, high-quality, and renderable 3D
reconstructions of environments [9]–[12], thereby improving
localization capabilities in texture-less or occluded regions.
All authors are with the Department of Electronic and Computer Engineer-
ing, Hong Kong University of Science and Technology, Hong Kong SAR.
The work described in this paper was supported by grants AoE/E-601/24-N,
16203223, and C6029-23G from the Research Grants Council of the Hong
Kong Special Administrative Region, China. (Corresponding author: Huan
Yin)
Fig. 1.
NuRF achieves monocular localization on images generated from
radiance fields. The image within the blue box is observed by our blimp
robot, while the image in the green box is a reference image rendered from
radiance fields..
Recent studies [10], [11] demonstrate the feasibility of uti-
lizing radiance fields for visual global localization. However,
casually captured radiance fields often suffer from unreal
artifacts, such as floaters, which can emerge and are not
part of the real view [13]. These ghostly artifacts critically
impact visual localization in terms of robustness and accuracy.
Existing regularizers [14]–[16] are ineffective in detecting or
eliminating floaters; alternative optimization methods based on
3D diffusion models [13] are too time-consuming for real-time
robot localization.
To address the limitations above, we propose an adaptive
nudged particle filter in radiance field (NuRF). NuRF en-
hances VPR through radiance field representations and nudges
VPR information to mitigate the impact of floaters. First, we
utilize the radiance field to generate novel views from new
perspectives, overcoming the challenge of sparseness for VPR.
Second, inspired by a previous work [17], the nudged particle
filter fuses the coarse global localization information from
VPR with the particle filter’s fine-grained pose estimation,
enabling continuous, global and efficient localization. Third,
this hybrid approach also ensures a smooth transition from
global localization to accurate pose tracking via an adaptive
mechanism. This approach effectively balances computational
efficiency and precision, addressing the challenges of robot
localization using NeRF as maps. Overall, the contributions
are summarized as follows:
• We achieve a monocular robot localization by using
radiance fields as maps and leveraging VPR techniques.
• We propose nudging the particle filter to enhance visual
localization performance and overcome the challenge of
arXiv:2406.00312v2  [cs.RO]  26 Mar 2025

<!-- page 2 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
2
artifacts in radiance fields.
• We design an adaptive scheme that switches between
global localization and position tracking for robust and
efficient visual localization.
• All experiments are conducted with our blimp robot in
the real world, with ablated tests.
The rest of our paper is organized as follows: Section II
introduces the related work on radiance field, visual place
recognition, and nudged particle filter; Section III presents
the problem formulation. We detail the proposed method in
Section IV and demonstrate its effectiveness with real-world
experiments in Section V. The conclusions and future studies
are summarized in Section VI.
II. RELATED WORK
A. Localization in Radiance Fields
A radiance field is a representation of how light is dis-
tributed in three-dimensional space, capturing the interactions
between light and surfaces in the environment [7]. It can be
seen as a low-dimensional sub-manifold that encompasses all
the rendered images of the scene in a high-dimensional pixel
space. In recent years, radiance field is a new wave for robotic
applications, as summarized in the survey paper [18].
Loc-NeRF by Maggio et al. [10] introduced a Monte Carlo
localization method for estimating the posterior probability of
individual camera poses in a given space by rendering mul-
tiple random images within a radiance field. These rendered
images are then compared to the target image in pixel space,
evaluating the pixel-space distances. The recent advancement
in explicit neural radiance field Gaussian splatting [8] has
achieved a rendering rate of 160 frames per second (fps)
per image while requiring less than 500MB of storage [19].
Moreover, this technology can be implemented using handheld
devices such as smartphones [20]. In the 3DGS-ReLoc by
Jiang et al. [12], Gaussian Splatting representation was uti-
lized for visual re-localization in urban scenarios, following a
coarse-to-fine manner for global localization. Specifically, the
camera was first located by similarity comparisons, and then
refined by the PnP technique. This indicates that the proposed
method might not be practically feasible for engineering
implementation.
B. Visual Place Recognition
VPR is a fundamental problem in robotics and computer
vision [4], aiming to provide a coarse pose estimation by
retrieving geo-referenced frames in close proximity. Modern
learning-based VPR solutions, such as AnyLoc [21], offer
comprehensive and accurate frame retrieval due to their robust
representation of both images and maps.
However, VPR achieves localization by considering the
reference pose approximating the query pose. This inher-
ent approximation limits the localization accuracy, especially
when there is a significant deviation between the current pose
and the queried poses in the database. Consequently, VPR is
typically employed as a coarse or initial step in high-precision
localization systems [22], which is also analyzed in the survey
paper for global LiDAR localization [3]. On the other hand,
constructing a database of landmark anchors for reference
purposes can be a time-consuming and labor-intensive process.
It involves collecting a reference image database from the
robot’s onboard camera and odometer data as it navigates
through the environment.
C. Nudged Particle Filter
Nudging refers to a sampling approach in particle filter-
ing that guides particles towards regions of high expected
likelihood [23]. This technique has been demonstrated to
effectively address issues with Bootstrap particle filters [24]
when modeling errors are present, and it’s particularly effective
when the posterior probabilities concentrate in relatively small
areas of the state space, making it well-suited for high-
resolution observation models, such as images. In the previous
work conducted by Lin et al. [17], conducted experiments
demonstrating that the utilization of nudging particles enabled
accurate tracking of a robot’s pose, even in cases where the
initial distribution was incorrect and had low variance.
The preliminary version of Nudged Particle Filter was
developed with the objective of rectifying erroneous kinetic
modeling assumptions through the use of nudging and did
not achieve successful global localization using the nudged
particle filter in that particular study. We initially proposed the
application of Nudged Particle Filter to address deficiencies in
observation models like Radiance Field and global localiza-
tion.
III. PROBLEM FORMULATION
The visual localization problem can be formulated as an
estimation of an unknown posterior distribution. For conve-
nience, we will use the special Euclidean group (SE(3)) to
present the camera pose or camera extrinsic, let the pose of a
camera-mounted robot at time τ be the same as the extrinsic
matrix of the camera Tτ ∈SE(3). From time τ −1 to τ,
the robot pose is integrated by relative transformations Hτ ∈
SE(3), described as follows:
Tτ = Tτ−1Hτ =
Rτ−1V τ
Rτ−1uτ + tτ−1
0T
1

(1)
in which t, u ∈R3 are the translation vectors and R, V ∈
R3×3 are rotation matrices. For two consecutive poses, we
assume the sensors equipped on the robot can capture two
images Iτ−1, Iτ and the motion ˆHτ. Mathematically, both
implicit and explicit radiance fields model can be represented
as a function {L : SE(3) 7→RW ×H}. It provides a mapping
of the extrinsic matrix Tτ to a W × H image ˆIτ that with
minimum render loss LOSS(·, ·).
L(Tτ) = arg min
ˆIτ
LOSS(Iτ, ˆIτ)
(2)
Then, the posterior probability density function of the robot
pose Tτ can be written as:
p(Tτ|I1:τ, ˆH2:τ, L) = p(I1:τ|L, Tτ)p(Tτ| ˆH2:τ)
p(I1:τ|L)
(3)

<!-- page 3 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
3
and the visual-based localization problem can be described as
an estimation problem for unknown distribution, described as
follows:
Problem 3.1: Given a sequence of RGB images I =
[I1, I2, . . . , Iτ], motion sequence ˆ
H = [ ˆH2, ˆH3, . . . , ˆHτ], and
the radiance field model L, how to approximate the posterior
distribution of the robot pose Tτ as described in Equation (3).
Problem 3.1 is a widely studied problem in the robotics
literature for unknown distribution estimation and has been
solved by Sequential Monte Carlo (SMC) commonly [25]. Let
Ξτ = {ζτ,i}N
i=1 be the set of particles at time τ, where each
particle ζτ,i = [wτ,i, Tτ,i] with weight wτ,i and pose Tτ,i.
Consequently, the posterior probability density function (3)
can be approximated as follows:
f(z; Ξτ) = η
N
X
i=1
wτ,iδ(z −Tτ,i)
(4)
in which δ is the Dirac delta function, and η is a normal-
ization constant. The state transition probability p(Tτ| ˆH2:τ)
is given by Equation (1), to minimize the diversity between
f(z; Ξτ) and p(Tτ|I1:τ, ˆH2:τ, L), the un-normalized impor-
tance weights based on the likelihood should satisfied [26]:
wτ,i = wτ−1,ip(Iτ|L, Tτ,i)
(5)
and the likelihood p(Iτ|L, Tτ,i) can be computed directly by
the pixel-wised render loss function LOSS(·, ·) in Equation (2).
As presented in Equation (2), the radiance field rendering
function L is optimized via the pixel-wise loss function
LOSS(·, ·), indicating that the weights derived from the loss
function could be affected by these artifacts. In this paper,
we consider it is possible to eliminate unrealistic artifacts
by representing images using advanced visual features (de-
scriptors) [21], thereby retaining only the visual descriptors
of the environment. Consequently, a further key focus of this
study is how to selectively adjust a small subset of particles to
decrease the feature loss based on VPR information during the
particle filter update process to enhance estimation accuracy.
This consideration leads to the following Problem 3.2, where
the VPR loss is defined by the cosine distance between the
visual feature vector of the rendered image and the observed
image.
Problem 3.2: Given a particle set Ξ and w is calculated by
pixel-wised loss. The enhancement of the particle set to better
approximate the p(Tτ|I1:τ, ˆH2:τ, L) involves minimizing the
discrepancy between true distribution and the particle-based
estimate f(z; Ξ+):
Ξ+ = arg min
Ξ+ ∥1 −p(Tτ|I1:τ, ˆH2:τ, L)
f(z; Ξ+)
∥
(6)
IV. METHODOLOGY
To address the outlined problems, we introduce the NuRF
framework as depicted in Figure 2. This framework integrates
anchor points in close proximity to the target image Iτ within
the feature space during the resampling phase. By imple-
menting this strategy, both Problem 3.2 and Problem 3.1 are
Algorithm 1: Anchor Setting in Radiance Fields
Input : G, L, K, N, S
Output: D
1 T ←∅;
2 F ←∅;
3 S ←[(x1, y1, ω1), . . . , (xN, yN, ωN)];
4 for s ∈S do
5
ξ ←[s[1], s[2], 0, s[3], 0, 0];
6
T ←exp (ξ∧) ;
// Equation (7)
7
I ←L(T) ;
// Equation (2)
8
F ←F ∪Encode(I);
9
T ←T ∪T;
10 end
11 D ←{F : T };
12 return D;
effectively addressed for real-world deployment. Additionally,
the utilization of nudging particles facilitates the monitoring
of particle dispersion, i.e., the variance of particles, which is
used to switch the global localization and poser tracking in
the radiance fields.
A. Radiance Fields-enhanced VPR Anchors
Pixel-wised similarity is sensitive to translational move-
ment, as shown in Figure 3. Even if the orientation error is 0,
only images rendered in a very small spatial neighborhood of
the target image location can obtain a high similarity response.
In global localization, when the number of particles is limited,
the probability of randomly generated particles falling into this
neighborhood is also very small, and the particle filter might
fail to converge to the ground truth pose.
In order to overcome these problems and build an efficient
NuRF (Problem 3.1), we introduce the VPR on anchor poses.
As shown in Figure 4, VPR technology requires a set of anchor
points in the space and stores the images generated at these
anchor points. Specifically, we detail the anchor setting in
Algorithm 1, the process initiates by sampling a set of 3-DoF
anchors, denoted as s = (x, y, ω), from a 2D sub-manifold of
SE(3). Subsequently, each anchor s is converted into a 6-DoF
pose ξ = (η1, η2) by incorporating zero values for the z-axis,
pitch, and roll angles. This conversion employs the exponential
map for ξ to SE(3), as follows:
exp (ξ∧) =
∞
X
n=0
1
n!(ξ∧)n =

R
t
0
1

(7)
Following this, the rendered image I is embedded into a
feature space F using a pre-trained embedding function [27].
Both the image feature F and the pose T are subsequently
stored in a database D.
B. Nudging by Visual Place Recognition
After generating the VPR anchors, we need to incorporate
the reference information provided by the VPR results into the
pixel-wised weighting particle filter to solve Problem 3.2. We
adopt the definition of the nudging step provided by Akyildiz

<!-- page 4 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
4
Fig. 2. The pipeline of our designed NuRF framework. We first use radiance fields to generate images on anchor poses, store these images in a database, and
vectorize them for retrieval. Then, a particle filter is built for robot localization in radiance fields. The nudging step uses retrieved results (from VPR) to guide
the particles toward more confident states. The measurement model updates particle weights based on images that are observed and rendered by particles. The
motion model adjusts particles according to robot motion. Our adaptive workflow enables switching between global localization and pose tracking to address
the robot kidnapping problem.
Pixel-wised
Feature-wised
Original
Rendered
Comparison
Similarity along X-axis
Fig. 3.
We show the original image and the rendered image in pixel space and feature space correspondingly. In the comparison column, the difference
between the original image and the rendered image is highlighted in blue. We assess the similarity between a query image and images rendered in the same
orientation but shifted along the X-axis using pixel-wise similarity and feature-wise similarity.
[23], illustrated in Definition 4.1. In this definition, xτ ∈X
represents the system state at time τ, and yτ denotes the
observations at time τ. The state after the nudging step is
denoted as x+.
Definition 4.1: A nudging operator αyτ
τ
: X →X associated
with the likelihood function gτ(x) is a map such that if x+ =
αyτ
τ (x), then gτ(x+) ≥gτ(x).
Intuitively, the nudging step adapts the generation of parti-
cles in a manner that is not compensated by the importance
of weights but enhances the likelihood of accurate estimation.
To achieve this, we have developed a novel technique termed
VPR nudging.
Figure 5 demonstrates the VPR nudging step, as illustrated
in Definition 4.1. The algorithm inputs include the reference
image database D at anchors, the observed image I, the
number of nudging particles M, the radiance field [G, L], and
the low-resolution camera intrinsic K−. The objective is to
identify the top M images in the database D that are most
similar to I and to render low-resolution images Im at each
corresponding pose Tm. If the weight of nudged particles
wm exceeds the current average weight ¯w, the pose Tm is
incorporated into the nudging particle set Ξ+.
The VPR nudging step modifies the stochastic generation
of particles in a manner that is not offset by the importance
weights. This strategic alteration enables the integration of
valuable feature information without necessitating computa-
tionally intensive embedding operations on every image ren-
dered by particles. This approach enhances both the accuracy
and efficiency of the global localization.
C. Adaptive Scheme in NuRF
We introduce an adaptive scheme within the particle filter
framework to switch the global localization and pose tracking.
Initially, in global localization, we lower the resolution of
both the input and rendered images. This reduces localization
accuracy slightly but expands the area where high similarity
responses are detected, lessening sensitivity to small move-
ments. As the process iterates and the particles begin to
converge, we maintain the original input image quality and
enhance the resolution of the rendered images to improve pose

<!-- page 5 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
5
Fig. 4.
We display 504 anchor poses in 2D sub-manifold S, along with
images rendered at these specific anchors for visualization.
tracking accuracy. This adaptive strategy dynamically adjusts
the resolution based on the discrepancy between nudged and
original particles. This approach effectively addresses the
issues of particle deficiency in global localization and aids
in automatic recovery from kidnapping scenarios.
Specifically, after each nudging step, we calculate the
weighted variance of the current particles. This variance indi-
cates how close the particle filter is to converging and reflects
the diversity of particle distribution. It directly influences how
we adjust the resolution in the image rendering process to
optimize accuracy and performance. The weight variance is
defined as follows:
¯T = 1
N
N
X
i=1
wiTi
(8)
ΣT = 1
N
PN
i=1 wi(Ti −¯T)(Ti −¯T)T
PN
i=1 wi
(9)
in which ΣT is a metric for assessing the convergence of
the particle filter. When the weighted variance of the par-
ticles drops below a set threshold λ, the system shifts to
a higher resolution setting (K+), improving the accuracy
of pose estimation. If the variance exceeds this threshold,
the algorithm switches to a lower resolution setting (K−),
reducing computational load and increasing the efficiency of
robot localization. The strategy is described as:
K =
 K+
ΣT ≤λ
K−
ΣT > λ
(10)
The system pipeline, as depicted in Figure 2, seamlessly
integrates VPR nudging and adaptive resolution rendering
techniques. Particle adjustments span extensive areas using
low-resolution rendering for the global localization phase at
a lower resolution (K = K−). The resampling weights for
this phase are calculated as follows:
wτ,j =
LOSS(Iτ, L(Tτ,j))
PN
i=1 LOSS(Iτ, L(Tτ,i))
(11)
Conversely, during the local pose tracking phase at a higher
resolution (K = K+), the particle updates are refined in
Algorithm 2: Global Localization
Input: Ξτ−1, Iτ, G, L, D, M
1 for ζτ−1,i ∈Ξτ−1 do
2
Tτ,i ←Tτ−1Hτ ;
// Equation (1)
3
ˆIτ,i ←L(Tτ,i) ;
// Equation (2)
4
wτ,i ←wτ,i−1 ;
// Equation (5)
5 end
6 Ξ+
τ ←α(D, Iτ, M, G, L, K−) ;
// Nudging
7 Ξ†
τ ←∅
8 for ζτ,i ∈Ξτ do
9
ζ†
τ,i ←resampling(f(z; Ξ+))
10
Ξ†
τ,i ←ζ†
τ,i
11 end
12 return Ξ†
τ;
Algorithm 3: Pose Tracking
Input: Ξτ−1, Iτ, G, L, D, M
1 for ζτ−1,i ∈Ξτ−1 do
2
Tτ,i ←Tτ−1Hτ ;
// Equation (1)
3
ˆIτ,i ←L(Tτ,i) ;
// Equation (2)
4
wτ,i ←wτ,i−1 ;
// Equation (5)
5 end
6 Ξ+
τ ←α(D, Iτ, M, G, L, K+) ;
// Nudging
7 µτ ←E(Ξ+
τ ) ;
// Equation (8)
8 Στ ←Var(Ξ+
τ ) ;
// Equation (9)
9 Ξ†
τ ←∅
10 for ζτ,i ∈Ξτ do
11
ζ†
τ,i ←resampling(N(µτ, Στ))
12
Ξ†
τ,i ←ζ†
τ,i
13 end
14 return Ξ†
τ;
more confined areas using high-resolution rendering. The
resampling weights in this phase are determined by:
wτ,j = N(Tτ,j; µτ, Στ)
(12)
where µτ and Στ represent the weighted mean and variance,
respectively, detailed in Equations (8) and (9). The Gaussian
approximate resampling technique, often used in this phase,
is particularly suited for environments expected to exhibit
a single-peak distribution. This method is crucial in high-
resolution settings, where even minor distortions such as
rotations and translations can significantly impact particle
convergence. The strategies for both global localization and
local pose tracking, thoroughly described in Algorithm 2
and Algorithm 3, ensure robust performance across varying
environmental complexities.
V. EXPERIMENTS AND EVALUATION
We validate the proposed NuRF in a motion capture room,
which is a typical indoor environment with cluttered objects.
The experiments are designed to assess the performance in

<!-- page 6 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
6
Captured Image
Images rendered 
at all anchors 
Vector 
database
Vector
Query
Images at top 𝑘nearest anchors
Add as nudging particles 
…
encoder
Encoded Feature 
Maps
encoder
Fig. 5.
Pipeline of nudging particles using the feature information from VPR. Upon the observation of a new image, the ViT encoder transforms it into
a feature map. This map is then vectorized and subjected to cosine distance calculations with the vectors stored in the VPR database. The camera pose
corresponding to the k vectors with the shortest cosine distance are retrieved and add into the particle set.
(a) Experimental setup
(b) Rendered image
Fig. 6.
(a) A Blimp robot in our motion capture room for experimental
evaluation. (b) A rendered image in the motion capture room that closely
mimics the real-world scenario.
terms of global localization and pose tracking, i.e., the con-
vergence from scratch and local tracking accuracy. All the
experiments are conducted on a blimp robot [28].
A. Experimental Setup
1) Radiance Field in Indoor Environments: Our method
relies on a pre-trained 3D radiance field as a pre-built map for
robot localization. We directly utilize the 3DGS [8], a real-
time radiance field rendering technique known for its state-of-
the-art performance. The input of the 3DGS is a trajectory
consisting of continuous camera poses and corresponding
images at each pose, which can be easily obtained on our
blimp robot.
Our blimp robot is equipped with one camera for monocular
localization. The robot was operated remotely within a 4-
meter by 5-meter fly arena, as shown in Figure 6(a). A motion
capture system is installed. to obtain the ground truth poses. To
generate the 3DGS of the room, 534 images with a 1440×1920
resolution are captured, and the rendered image is shown in
Figure 6(b).
2) Baseline: We compare NuRF with Loc-NeRF [10] in
both global visual localization and pose tracking tasks to
demonstrate its superior accuracy and speed. NuRF utilizes
3DGS as the map, which renders much faster than NeRF [8] in
the original Loc-NeRF. Therefore, to ensure a fair comparison,
we replace the NeRF with the 3DGS map in the Loc-NeRF
system.
3) Implementation Details: The experiments are conducted
with Pytorch-based implementation. The computing hardware
comprises an Intel CPU i7-13700KF and a GeForce RTX4080
GPU. The input image size was set to 680 × 800, with a
render resolution of 64×80 for global localization and a render
resolution of 680 × 800 for pose tracking. The thresholds for
global localization and pose tracking switching were set to
λ+ = 5.
B. Global Visual Localization
To evaluate the NuRF method’s performance in global
localization, we conducted experiments with 20 randomly
chosen sets of image sequences. Each set contained 100 frames
depicting continuous motion.
A demonstration for global visual localization in radiance
fields with NuRF is shown in Figure 7, the first column
illustrates the utilization of 400 particles. The initial positions
of these particles were obtained by uniformly perturbing the
dimensions of the room, which has a height of 1.8m, a length
of 5m, and a width of 4m. For orientation, the yaw angles
were uniformly initialized within the range of [−180◦, 180◦],
while the pitch and roll angles for all particles were set to

<!-- page 7 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
7
Fig. 7. A case study for global visual localization in radiance fields. Top: Sequential observed images of the robot at frame 1, frame 25, frame 50, and frame
100. Middle: Particle filter states for each frame are shown. The larger blue pyramid indicates the estimated pose, while the black pyramid represents the
ground truth pose. Smaller pyramids symbolize the particles, colored according to the visible spectrum, with red indicating the highest similarity and purple
the lowest. Bottom: Rendered images at the estimated pose are shown, with the rendering resolution adapting from low to high based on the variance of
particles.
Fig. 8. Average localization errors of NuRF and comparisons over 20 trials.
0◦. Once the experiment begins, the motion model provides a
6-DoF (six degrees of freedom) motion matrix to update the
pose of the particle swarm (depicted in the second to fourth
columns of Figure 7). The resampling process also occurs in
the 6-DoF space, meaning that the algorithm optimizes the
particles in a complete 6-DoF state and performs only one
update per image.
We conducted a test to compare the global localization
capabilities of the Loc-NeRF algorithm with ours using the
same image sequence. Additionally, to validate the modules
in NuRF, we performed ablation experiments by omitting the
nudging step and utilizing only the bootstrap particle filter
(BPF) with pixel-wise weights for global localization perfor-
mance. The experimental results are displayed in Figures 8
Fig. 9. Average time costs of NuRF and comparisons over 20 trials.
and 9. In the box plot, the upper and lower edges indicate
the maximum and minimum localization errors from the 20
experiments, while the box length represents the interquartile
range. The median is denoted by a blue dashed line in the
center of the box. Figure 8 shows that the median localization
error for NuRF is approximately 0.5 m, which is lower than
that of Loc-NeRF and the BPF when the VPR nudging
step is removed. Additionally, NuRF exhibits the most stable
localization error across the 20 experiments, with significantly
lower variance compared to the other two algorithms. Figure 4
further highlights the operational efficiency of NuRF, illustrat-
ing that its convergence time is notably shorter than that of
Loc-NeRF and the standard BPF without the nudging step.

<!-- page 8 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
8
Fig. 10. A case study of pose tracking. Top: observed images of the robot at
key frames: frame 100, frame 110, and frame 120. Middle: The particle filter
states at each frame, following the same color settings in Figure 7. Bottom: the
rendered images at the estimated poses, with changes in rendering resolution
based on the variance of the particles.
C. Pose Tracking in Radiance Fields
In this section, we evaluate the performance of the NuRF
method in a pose tracking task using images captured from a
blimp. We initiate the tracking process with the pose estima-
tion obtained in Section V-B. We employ 200 particles for the
pose tracking task and initialize their positions by sampling
from a normal distribution centered around the estimated
position. The variances for the position dimensions are set to
[0.2, 0.2, 0.1] respectively. Similarly, we sample the yaw, pitch,
and roll angles from normal distributions centered around
the estimated rotation. The variances for the angles are set
to [5, 1, 1] respectively. Figure 10 illustrates a case study of
tracking process, where only one update is performed per
image.
The demonstration presented in the first column of Figure
10 shows the re-generation of the particles around the initially
estimated position when transitioning from global localization
to positional tracking. In the second column of Figure 10,
we observe the convergence of the particles after running the
NuRF algorithm for 10 frames. In the pose tracking phase, the
VPR system continues to generate particles. However, these
particles are solely utilized to detect potential abduction issues
and are not involved in the resampling process.
To validate the effectiveness of the NuRF, we conduct
experiments on 20 randomly selected trajectories consisting
of 40 frames each and repeat the same experiment on Loc-
NeRF and BPF with these data. The results of the experiment
are presented in Table I, which also shows the mean attitude
tracking error and position error for each algorithm after 40
time steps. It can be seen in Figure 11 that the algorithm NuRF
has smaller tracking error than Loc-NeRF in pitch, yaw, and
3D spatial position.
(a) Translation
(b) Roll
(c) Pitch
(d) Yaw
Fig. 11.
Average translational and orientational error of the robot position
over 40 frames of 20 trials.
D. Ablated Experiments
To validate the modules in NuRF, this section conducts
ablated experiments on visual place recognition and explores
the effect of the number of anchor points on global localization
performance and the image resolution on global pose tracking
performance.

<!-- page 9 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
9
TABLE I
AVERAGE TRACKING ERROR OF VARIOUS ALGORITHMS FOR 20 TRIALS.
Average orientation
tracking error (degree)
Average position
tracking error (m)
Roll
Pitch
Yaw
Loc-NeRF
3.51
12.93
55.91
1.72
NuRF (ours)
2.17
7.81
12.18
0.64
TABLE II
RESULT OF ABLATED EXPERIMENT FOR 20 TRIALS.
Average position
tracking error (m)
Average run
frequncy (Hz)
Pixel-wised BPF
1.204
0.0909
504-NuRF
0.79
0.0769
2502-NuRF
0.64
0.0706
The ablated experiment is designed to evaluate global lo-
calization capabilities under two specific conditions: without
VPR nudging (which is degenerated to pixel-wised BPF) and
with attenuated VPR nudging (where the number of nudging
particles was halved and the VPR anchor point count was
reduced to 504). The results show in Table II that if there
are no anchored points, i.e., no nudging step is performed,
there is a noticeable decline in the accuracy. Specifically,
the translational mean square error of position estimation
decreases from 0.642 m to 1.204 m when using NuRF for
global localization. In terms of efficiency, the experimental
results demonstrate that the nudging step leads to a slight
increase in the time required for a single particle update,
approximately 1 second. However, it significantly reduces the
number of iteration steps needed for the convergence.
VI. CONCLUSIONS AND FUTURE STUDY
We present NuRF, a nudged particle filter designed for
robot visual localization in radiance fields. Our approach in-
corporates key insights, including visual place recognition for
nudging and an adaptive scheme for both global localization
and pose tracking. To evaluate the effectiveness of the NuRF
framework, we perform real-world experiments and provide
comparisons and ablated studies. These experiments serve to
validate the effectiveness of NuRF in indoor environments.
The proposed NuRF still has certain limitations for practical
use. One key limitation is its deployment in texture-poor envi-
ronments, where visual place recognition might fail due to the
lack of sufficient visual features. Another limitation lies in the
operational efficiency of NuRF. The main efficiency bottleneck
is the rendering speed of images using NeRF. Currently, NuRF
achieves a localization frequency of nearly 0.1 Hz. While this
frequency is acceptable for low-speed ground mobile robots
and indoor blimps, it poses challenges for integration into
high-speed robotic platforms such as drones or self-driving
cars.
We consider several promising directions to advance NuRF.
One promising direction is the use of incremental 3D Gaussian
Splatting [29], which could help adapt and refine the NuRF
system for larger-scale environments. The use of Gaussian
Splatting could also improve the operational efficiency com-
pared to the original NeRF. Another direction involves inte-
grating planning capabilities within radiance fields to enable
full navigation. Additionally, future research will focus on
engineering efforts to achieve real-time robot localization,
which is crucial for deploying NuRF in high-speed robotic
platforms and dynamic environments.
REFERENCES
[1] W. Burgard, A. Derr, D. Fox, and A. B. Cremers, “Integrating global
position estimation and position tracking for mobile robots: the dynamic
markov localization approach,” in Proceedings. 1998 IEEE/RSJ Inter-
national Conference on Intelligent Robots and Systems. Innovations in
Theory, Practice and Applications (Cat. No. 98CH36190), vol. 2. IEEE,
1998, pp. 730–735.
[2] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm
for model fitting with applications to image analysis and automated
cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395,
1981.
[3] H. Yin, X. Xu, S. Lu, X. Chen, R. Xiong, S. Shen, C. Stachniss, and
Y. Wang, “A survey on global lidar localization: Challenges, advances
and open problems,” International Journal of Computer Vision, pp. 1–
33, 2024.
[4] S. Lowry, N. S¨underhauf, P. Newman, J. J. Leonard, D. Cox, P. Corke,
and M. J. Milford, “Visual place recognition: A survey,” ieee transac-
tions on robotics, vol. 32, no. 1, pp. 1–19, 2015.
[5] S. Schubert, P. Neubert, S. Garg, M. Milford, and T. Fischer, “Visual
place recognition: A tutorial,” IEEE Robotics & Automation Magazine,
2023.
[6] J. Miao, K. Jiang, T. Wen, Y. Wang, P. Jia, B. Wijaya, X. Zhao,
Q. Cheng, Z. Xiao, J. Huang et al., “A survey on monocular re-
localization: From the perspective of scene map representation,” IEEE
Transactions on Intelligent Vehicles, 2024.
[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[8] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, pp. 1–14, 2023.
[9] J. Liu, Q. Nie, Y. Liu, and C. Wang, “Nerf-loc: Visual localization
with conditional neural radiance field,” in 2023 IEEE International
Conference on Robotics and Automation (ICRA), 2023, pp. 9385–9392.
[10] D. Maggio, M. Abate, J. Shi, C. Mario, and L. Carlone, “Loc-nerf:
Monte carlo localization using neural radiance fields,” in 2023 IEEE
International Conference on Robotics and Automation (ICRA).
IEEE,
2023, pp. 4018–4025.
[11] Y. Sun, X. Wang, Y. Zhang, J. Zhang, C. Jiang, Y. Guo, and F. Wang,
“icomma: Inverting 3d gaussians splatting for camera pose estimation
via comparing and matching,” arXiv preprint arXiv:2312.09031, 2023.
[12] P. Jiang, G. Pandey, and S. Saripalli, “3dgs-reloc: 3d gaussian splat-
ting for map representation and visual relocalization,” arXiv preprint
arXiv:2403.11367, 2024.
[13] F. Warburg, E. Weber, M. Tancik, A. Holynski, and A. Kanazawa,
“Nerfbusters: Removing ghostly artifacts from casually captured nerfs,”
in Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 18 120–18 130.
[14] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, “Barf: Bundle-adjusting
neural radiance fields,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2021, pp. 5741–5751.
[15] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Doso-
vitskiy, and D. Duckworth, “Nerf in the wild: Neural radiance fields
for unconstrained photo collections,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2021, pp. 7210–
7219.
[16] S. Sabour, S. Vora, D. Duckworth, I. Krasin, D. J. Fleet, and
A. Tagliasacchi, “Robustnerf: Ignoring distractors with robust losses,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 20 626–20 636.
[17] T. X. Lin, S. Coogan, D. A. Sofge, and F. Zhang, “A particle fusion
approach for distributed filtering and smoothing,” Unmanned Systems,
pp. 1–15, 2024.
[18] G. Wang, L. Pan, S. Peng, S. Liu, C. Xu, Y. Miao, W. Zhan,
M. Tomizuka, M. Pollefeys, and H. Wang, “Nerf in robotics: A survey,”
arXiv preprint arXiv:2405.01333, 2024.

<!-- page 10 -->
ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS
10
[19] A. Hamdi, L. Melas-Kyriazi, G. Qian, J. Mai, R. Liu, C. Vondrick,
B. Ghanem, and A. Vedaldi, “Ges: Generalized exponential splatting
for efficient radiance field rendering,” arXiv preprint arXiv:2402.10128,
2024.
[20] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer,
D. Ramanan, and J. Luiten, “Splatam: Splat, track & map 3d gaussians
for dense rgb-d slam,” arXiv preprint arXiv:2312.02126, 2023.
[21] N. Keetha, A. Mishra, J. Karhade, K. M. Jatavallabhula, S. Scherer,
M. Krishna, and S. Garg, “Anyloc: Towards universal visual place
recognition,” IEEE Robotics and Automation Letters, vol. 9, no. 2, pp.
1286–1293, 2023.
[22] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, “From coarse to
fine: Robust hierarchical localization at large scale,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2019, pp. 12 716–12 725.
[23]
¨O. D. Akyildiz and J. M´ıguez, “Nudging the particle filter,” Statistics
and Computing, vol. 30, pp. 305–330, 2020.
[24] N. J. Gordon, D. J. Salmond, and A. F. Smith, “Novel approach to
nonlinear/non-gaussian bayesian state estimation,” in IEE proceedings
F (radar and signal processing), vol. 140, no. 2.
IET, 1993, pp. 107–
113.
[25] A. Doucet, A. M. Johansen et al., “A tutorial on particle filtering and
smoothing: Fifteen years later,” Handbook of nonlinear filtering, vol. 12,
no. 656-704, p. 3, 2009.
[26] A. Doucet, S. Godsill, and C. Andrieu, “On sequential monte carlo
sampling methods for bayesian filtering,” Statistics and computing,
vol. 10, pp. 197–208, 2000.
[27] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khali-
dov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y.
Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran,
N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut,
A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features
without supervision,” 2023.
[28] Q. Tao, J. Wang, Z. Xu, T. X. Lin, Y. Yuan, and F. Zhang, “Swing-
reducing flight control system for an underactuated indoor miniature
autonomous blimp,” IEEE/ASME Transactions on Mechatronics, vol. 26,
no. 4, pp. 1895–1904, 2021.
[29] K. Minamida and J. Rekimoto, “Incremental gaussian splatting: Gradual
3d reconstruction from a monocular camera following physical world
changes,” in SIGGRAPH Asia 2024 Posters, 2024, pp. 1–2.
