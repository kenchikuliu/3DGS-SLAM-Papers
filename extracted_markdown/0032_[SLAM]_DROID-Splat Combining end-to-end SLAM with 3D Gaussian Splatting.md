<!-- page 1 -->
DROID-Splat
Combining end-to-end SLAM with 3D Gaussian Splatting
Christian Homeyer1
Leon Begiristain1
Christoph Schnörr1
1Image and Pattern Analysis Group, Heidelberg University, Germany
homeyer@math.uni-heidelberg.de, lbegirri7@alumnes.ub.edu, schnoerr@math.uni-heidelberg.de
Abstract
Recent progress in scene synthesis makes standalone SLAM
systems purely based on optimizing hyperprimitives with a
Rendering objective possible [24]. However, the tracking
performance still lacks behind traditional [27] and end-
to-end SLAM systems [41]. An optimal trade-off between
robustness, speed and accuracy has not yet been reached,
especially for monocular video. In this paper, we introduce
a SLAM system based on an end-to-end Tracker and extend
it with a Renderer based on recent 3D Gaussian Splatting
techniques. Our framework DroidSplat achieves both SotA
tracking and rendering results on common SLAM bench-
marks. We implemented multiple building blocks of modern
SLAM systems to run in parallel, allowing for fast inference
on common consumer GPU’s. Recent progress in monoc-
ular depth prediction and camera calibration allows our
system to achieve strong results even on in-the-wild data
without known camera intrinsics. Code will be available at
https://github.com/ChenHoy/DROID-Splat.
1. Introduction
Simultaneous Localization and Mapping (SLAM) has been
a longstanding problem in Computer Vision, fundamental to
applications in robotics, autonomous driving and augmented
reality. While traditional systems focus on reconstruction of
accurate odometry and geometry from hand-crafted features,
they usually result in sparse or semi-dense representations
of the environment. End-to-end SLAM systems [21, 41, 42]
improved robustness and accuracy by using learned features
and a dense reconstruction objective, however they often
lack the ability to optimize a photo-realistic scene. Recent
progress in scene synthesis makes standalone SLAM sys-
tems purely based on optimizing hyperprimitives with a Ren-
dering objective possible [24]. However, the tracking per-
formance still lacks behind traditional [27] and end-to-end
SLAM systems [41]. We aim to close this gap by combining
the best of both worlds.
𝑮𝟏
𝑮𝒊−𝟐
𝑮𝒊−𝟏
𝑮𝒊
Figure 1.
DROID-Splat allows to reconstruct a scene with
known/unknown intrinsics. By combining an optical flow tracking
objective and a fast, dense Renderer, we can achieve photo-realistic
Reconstructions while optimizing accurate odometry.
In this paper, we introduce DROID-Splat: A SotA SLAM
system based on dense, end-to-end optical flow and a dense
Rendering objective using 3D Gaussian Splatting [18]. Our
system offers the same flexibility as it’s parent system [41]:
We support monocular and rgbd inference for different cam-
era models (since we focus on single camera reconstruction,
we neglect the stereo [41] or multi-view case [34]). By
combining the best of both worlds, we achieve fast tracking
inference on consumer GPU’s and can quickly optimize a
photo-realistic scene reconstruction. Our framework con-
sists of a i) local frontend ii) global backend iii) loop closure
detector iv) dense renderer. With this work we aim to system-
atically analyze the interplay of individual components and
optimization objectives in more detail than previous work.
A lot of recent SLAM frameworks have emerged, that focus
on a single component. Our work aims to provide a compre-
hensive tool, which allows to easily reconstruct a scene from
a video.
Monocular Video is notoriously difficult to reconstruct.
For this reason we additionally allow integration of SotA
1
arXiv:2411.17660v2  [cs.CV]  29 Nov 2024

<!-- page 2 -->
monocular depth prediction [2, 49, 51] priors similar to [56,
58] and concurrent work [33]. We show that with recent
advances, it is possible to robustly handle in-the-wild data
with unknown camera intrinsics. Using a depth prior and
an additional camera calibration objective [10], we achieve
strong reconstruction performance even on cellphone videos.
Our contributions are:
• We propose a dense SLAM system, which combines a
dense end-to-end tracker with dense hyperprimitives.
• We combine common building blocks of modern SLAM
systems in a fast parallel implementation. Our comprehen-
sive ablations show which components really matter.
• We show SotA results on common SLAM benchmarks for
both tracking and rendering in near real-time.
• Our framework is flexible with regards to input and works
even on in-the-wild data with unknown intrinsics.
2. Related Work
Visual SLAM. Traditional SLAM systems can be catego-
rized into direct or indirect [8] systems depending on their
intermediate representation and objective function. Indi-
rect approaches [27] make use of sparse feature descriptors
for matching and then solve a geometric bundle adjustment
problem. Direct approaches [7, 8] optimize a photometric
error directly and operate on semi-dense pixel representa-
tions. However, direct approaches usually lead to more
difficult optimization problems. Overcoming the limita-
tion of both hand-crafted features and ill-behaved optimiza-
tion, end-to-end SLAM systems [21, 41, 42] were proposed
which allow a dense representation with well-behaved track-
ing. Long-term tracking requires a loop closure mechanism
[21, 22, 27, 62]. Common frameworks memoize features of
past frames to find similarities of new incoming frames to
start a loop closure optimization, e.g. a Pose Graph Opti-
mization [20]. On top of good odometry, common systems
are concerned with a dense scene reconstruction. Tradi-
tional SLAM approaches either relied on voxel [3, 14, 28]
or point [17, 35, 46] based map representations. These rep-
resentations can allow a dense reconstruction, but are not
photo-realistic.
Differentiable rendering.
Neural Radiance Fields
(NeRFs) [25] opened the gate to achieve photorealistic vol-
ume rendering, but training was initially slow. Using a multi-
resolution hash encoding [26] enabled for the first time to
use neural radiance fields in a SLAM context [30]. Recently,
3D Gaussian Splatting (3DGS) [18] has revolutionized the
field. The real-time rendering and training quickly enabled
numerous works to taylor a direct SLAM system based on
the Rendering objective [16, 24, 48, 54]. However tracking
remains to be behind the traditional counterparts. Combined
hybrid systems [13, 33, 56] resolve this issue by combining
the best of both worlds. Similar in fashion, we make use of
a dense, robust end-to-end system [41] and combine it with
a renderer. We refer to [43] about more details on Rendering
in SLAM.
Numerous works improve the original GS with new tech-
niques [6, 12, 13, 19, 39, 44, 52, 53, 56, 60], some including
better densification and pruning strategies [6, 52, 53] or bet-
ter loss supervision [44]. We aim to leverage recent advances
in diff. rendering in a SLAM context.
Concurrent Work. The most related work to us is [33],
which similar to us, is based off DROID-SLAM [41] and
3DGS [18]. In the same manner, they make use of a previ-
ously proposed monocular prior integration [58]. However,
we go beyond the monocular use case and analyze different
input modes, renderer and additional camera calibration [10].
Moreover, we use a different loop closure mechanism and
dive deeper into the interplay of tracker and renderer.
3. Our Approach
Since our goal is a photo-realistic dense scene reconstruction,
we use a dense end-to-end tracker, which can provide reliable
depth (or disparity) for each pixel. After filtering this map
for only covisible points or areas of high confidence, we
feed it into a rendering module, which optimizes Gaussian
hyperprimitives for each pixel and densifies the scene based
on a rendering objective. Due to the lightweight nature of
Gaussian Splatting [18], we can run this rendering objective
in real-time in parallel to our tracking system. An overview
of our system can be seen in Figure 2. We systematically
build up our system from common SLAM components. By
unifying these techniques under one umbrella, we can reach
state-of-the-art online photo-realistic reconstruction.
3.1. End-to-end Tracking
We base our tracker on online end-to-end system DROID-
SLAM [41]. A frame-graph (V, E) is build from an incoming
ordered stream of images {I1, · · · , In} ∈RN×H×W ×3.
This structure is in practice a keyframe buffer, storing our
tracking state variables disparity di ∈RH×W and camera
pose gi ∈SE (3). Dense optical flow is estimated by a
recurrent neural network [40]. Given enough motion in the
scene, a keyframe is inserted into the graph. An edge (i, j)
signifies covisibility between the frames i and j. As this
graph is dynamically build and maintained over the incoming
stream, we perform differentiable Bundle Adjustment over
the graph. Given the current state of poses and disparity, we
can compute correspondences
 \l a be
l
 {p r oje
c
tion } p
_{ij} = \Pi _{c}\left ( G_{ij} \circ \Pi _{c}^{-1}\left ( p_{i},\; d_{i} \right ) \right ) 
(1)
with the camera projection function Π. We use a pinhole
camera model in all of our experiments, however similar to
[10], we support multiple camera models in theory for this
function. A correlation volume as in [40] can be indexed
given pij, so we retrieve correlation features along an edge
2

<!-- page 3 -->
Keyframe
Buffer
Feature DB
RGB (D) Video
Loop  
Detector
∞
Input Filter
DBA
Feature
Extraction& 
Correlation
Conv
GRU
𝐺1
𝑍1
𝜃
Update State
Δ𝑝𝑖𝑗
𝑤𝑖𝑗
𝑟12 𝑟13 𝑟21
𝑟23 𝑟31
𝑟32
Edges 𝑖𝑗
𝐺2
𝐺3
𝑠1
𝑜1
𝑍2
𝑠2
𝑜2
𝑍3
𝑠3
𝑜3
Rendering
Local BA
Global BA
𝑮𝒊
2D / 3D 
Gaussians
(Feedback)
Figure 2. DROID-Splat. We make use of an end-to-end SLAM system with an optical flow based objective to perform tracking and
reconstruct odometry and a dense initial map. The tracking objective is flexible, which allows us to optimize intrinsics or prior scale and
shift as well if wanted. We make use of SotA Gaussian Splatting techniques to learn a photo-realistic reconstruction based on a Rendering
objective. Since all components are differentiable and run in parallel, we can let parts interact flexibly.
(i, j). The features, along with image context and a hidden
state are input to a convolutional GRU to produce an update.
The GRU produces i) the residual field rij ∈RH×W ×2 and
an associated confidence wij ∈RH×W ×2. The residuals
guide the current correspondences as p∗
ij = rij + pij. To-
gether with the learned pose estimation confidence this pow-
ers a differentiable bundle adjustment optimization. Track-
ing is based on the reprojection based loss:
 \lab el {
e
qn:obje
ctiv
e}  \math
bf  E(\
m athb f 
{G}
', 
\ma
thb f  {d} ') 
&= \sum _{(i,j) \in \mathcal {E}} \norm {\mathbf {p}_{ij}^* - \Pi _c(\mathbf {G}'_{ij} \circ \Pi _c^{-1}(\mathbf {p}_i, \mathbf {d}'_i)) }_{\Sigma _{ij}}^2 \nonumber \\ \Sigma _{ij} &= \diag \mathbf {w}_{ij} \quad .
(2)
This generic loss function can be used flexibly to not only
supervise disparity d′ and pose G′, but as shown in [10], we
can also directly optimize the calibration of the camera with
intrinsics θ:
 \lab el {e q
n
:calib}
 \ma
th b f 
E
(\
ma t hbf  
{G}'
, \
math
bf  {d}
'
, \m at
hb f {\theta }) &= \sum _{(i,j) \in \mathcal {E}} \norm {\mathbf {p}_{ij}^* - \Pi _c\left ( \mathbf {G}'_{ij} \circ \mathbf {P}_{i}, \mathbf {\theta } \right ) }_{\Sigma _{ij}}^2 \nonumber \\ \text {with} \quad \mathbf {P}_{i} &= \Pi _c^{-1} \left ( \mathbf {p}_i, \mathbf {d}'_i, \mathbf {\theta } \right )
(3)
Now [41] supports RGBD-SLAM by regularizing this
with a prior term:
 \la bel {eq n
:
pri
or}
 \ ma
thbf E_{reg}\left (\mathbf {d}^{*}, \mathbf {d}'\right ) = \sum _{i \in \mathcal {V}} \norm { \mathbf {d}_{i}^{*} - \mathbf {d}'_{i} }^2 
(4)
over a given input depth d∗from an external sensor. Since
we want to reconstruct any video, we make use of monocular
depth prediction priors [2, 49, 51]. Even though monocular
depth prediction has made progress to predict accurate metric
predictions [2, 51], there is considerable temporal fluctua-
tions across the board of SotA monocular models. For this
reason we optimize in what we call the Pseudo-RGBD mode,
similar as [33, 56, 58]:
 \labe l {e qn: mon o_p r
i
or}
 \m
a t hbf
 E _{
r e g,
m}\left (\mathbf {d}^{*}, \mathbf {d}', \mathbf {s}', \mathbf {o}'\right ) = \sum _{i \in \mathcal {V}} \norm { \mathbf {d}_{i}^{*} - \left (\mathbf {s}'_{i} \cdot \mathbf {d}'_{i} + \mathbf {o}'_{i}\right ) }^2 
(5)
After solving this bundle adjustment problem for a fixed
number of iterations over the graph, we can update our state
variables and continue until the next recurrence. In P-RGBD
mode, we must be careful as an ambiguity between s, o and
G exists. For this reason like [58], we perform this in a
block-coordinate descent manner, where we first fix scales
and offsets and optimize poses. Afterwards we fix the pose
graph and optimize structure, scales and offsets. We observe
a similar ambiguity between intrinsics θ and the monocular
variables. For this reason we operate in two stages on in-
the-wild video inspired by [23]: 1. Fix the prior and use
Eq.3 together with Eq. 4 to calibrate the camera. 2. Use the
calibrated camera to run in P-RGBD mode with Eq. 5.
Modern SLAM systems [7, 27, 41] perform bundle adjust-
ment normally on different parts of the map: i) A local fron-
tend optimizes small-scale graphs for incoming keyframe
windows ii) A global backend optimizes large-scale graphs
with long-term connections over the whole map. While the
original implementation [41] performs this on two separate
3

<!-- page 4 -->
GPU’s, we run both Processes on a single GPU and perform
these two optimizations synchronized in paralell. Monocular
prior integration is performed on local frontend windows
before the adjusted map is put into the backend. Camera
intrinsics θ are treated as a global variable, that is optimized
in the backend.
3.2. Loop Closure
We observe, that Visual Odometry accuracy and robustness
depends not only on the optimization itself, but in particular
on the graph structure of front- and backend. Accumulated
drift can be compensated by running the Update operator on
long-term connections of potential loop candidates. While
[33, 59] detect candidates based on low apparent motion
detected by the recurrent flow network [41], we had more
success by using direct visual similarity. While systems
as [21, 22, 27] rely on hand-crafted ORB features [31], we
leverage recent end-to-end features from place recognition
tasks [1]. For each incoming keyframe, we compute it’s
visual features and insert them in a FAISS [4] database
on the CPU. We then check for nearest neighbors in all
past frames. Similar to [58], we only consider a frame pair
(i, j) a loop candidate if i) The feature distance is small
enough df,ij < τf ii) The camera orientation distance is
small enough d (Ri, Rj) < τr and iii) the frames are far
apart enough |ti −tj| > τt. If a candidate pair is found, we
augment the graph by adding a bi-directional edge to the
backend. This Process runs in parallel on the CPU with a
marginal additional cost.
3.3. Differentiable Rendering
Similar to previous works [13, 16, 24, 33] we utilize Gaus-
sian hyperprimitives defined as a set of points P ∈R3 asso-
ciated to our dense tracking map. Each Gaussian possesses
a rotation r ∈SO (3), scaling s ∈R3, density σ ∈R1
and spherical harmonic coefficients SH ∈R16. We ini-
tialize the Gaussians similar to [24] by downsampling the
map by a constant factor after triangulation. Gaussians are
optimized via backpropagation on a dense Rendering loss.
The rendering process [18] is defined as:
 \la be l
 
{eq
n:re
nde
r
ing
} C \le ft ( \mathbf {R}, \mathbf {t} \right ) = \sum _{i\in \mathcal {N}} \mathbf {c}_{i} \alpha _{i} \prod _{j=1}^{i-1} \left ( 1 - \alpha _{i} \right ), 
(6)
where c denotes the color converted from SH and αi =
σi · G (R, t, Pi, ri, si). This allows us to render our map at
given keyframe Gi to produce both an image I′
i and depth
Z′
i. We follow [24] for median depth rendering. Gaussian
Splatting [16, 18, 24] utilizes a mixed rendering loss
 \ l abel { r nd _ los s} L_{
i } &= \ lam bda
 _ {1
}  L _{ rg b } + \le
ft  (
1  - 
\ la m bda  _{
1 }  \
r ight ) L_{depth} \nonumber \\ &= \lambda _{1} \left [ \left (1 - \lambda _{2}\right ) \norm {I'_{i} - I^{*}_{i}} + \lambda _{2} \left ( 1 - SSIM\left (I'_{i}, I^{*}_{i}\right ) \right ) \right ] \nonumber \\ &+ \left ( 1 - \lambda _{1} \right ) \norm {Z'_{i} - Z^{*}_{i}}
(7)
which allows us to perform backpropagation by comparing
with a reference I∗, Z∗. Each time we update our renderer,
we optimize over a batch of cameras to improve our scene
reconstruction. Since every component is differentiable, we
can in theory optimize our keyframe poses with the rendering
objective and feed them back into the tracker. We therefore
want to research the questions: Which objective is better
suited for tracking? Can we improve our system further by
finetuning with a dense rendering objetive?
Since we only improve the map by covering the whole 3D
space with Gaussians, the original adaptive density control
[18] strategy splits and clones Gaussians based on their size
and gradient. This strategy was also used in any succesful
SLAM application [13, 16, 24, 33, 54]. It was recently
observed, that this strategy is suboptimal and by guiding
this process with a Monte Carlo Chain Markov (MCMC)
model [19], we can improve performance. At the same time,
this provides a preset upper limit of the total number of
primitives. We compare these different strategies for our
system and compare the 3D hyperprimitives themselves with
the recently proposed 2D surfel Gaussians [12]. 2D Gaussian
Splatting approximates surfaces by collapsing the primitives
to flat surface disks, which result in more accurate geometry.
4. Experiments
We combine our components in a flexible way and ablate
these choices in the following. During inference, we synchro-
nize frontend, backend and renderer based on fixed frequen-
cies, i.e. we run backend and renderer for every k1, k2 calls
of the leading frontend process. The loop detector is con-
stantly run in the background. If we detect a large tracking
map update, we record the rel. transformations to reanchor
our hyperprimitives. With mostly a stable map, we simply
use a rigid body transformation G ∈SE (3) for this purpose.
Gaussians are then typically in a position where they will
quickly reconverge upon a new rendering optimization. For
our monocular experiments with a prior, we use Metric3D
[51] as it gave the most temporally consistent predictions
without any scale optimization. We ablate this choice in the
supplementary against multiple SotA models. We run our
system on a NVIDIA RTX 4090. Similar to [24, 33] we do
a refinement stage after running online tracking. We refine
our map for 2k iterations and report the refined results for
our final numbers. As benchmark metrics generally favor
slower methods for this task, we report the detailed scaling
of speed and performance in Fig. 4. We give more details
on our system configuration, loss balancing and experiment
settings in the supplementary.
Datasets. We evaluate our method on common SLAM
benchmarks Replica [37] and TUM-RGBD [38]. We addi-
tionally showcase the ability of our system on self-recorded
outdoor cellphone video.
Baselines. We compare ourselves to SotA pure Splatting
4

<!-- page 5 -->
Components
ATE RMSE
KF [cm]
ATE RMSE
All [cm]
ATE RMSE
KF [cm]
ATE RMSE
All [cm]
TUM RGBD
Replica
Frontend +
Backend
4.88
5.22
2.51
2.47
+ scale opt.
1.92
1.80
0.273
0.273
+ Loop Detection
1.88
1.78
0.269
0.268
w Loop BA [59]
Backend
3.91
3.61
0.53
0.52
Table 1. Ablation Tracking. We compare in P-RGBD mode, but
observe that this is mostly consistent across input modes. Optimiz-
ing the scale of priors [51] is still important even when they are
metric predictions. Using visual cues to find loop edges improves
tracking.
Technique
PSNR↑
LPIPS↓
L1↓
PSNR↑
LPIPS↓
L1↓
KF
Non-KF
3DGS [18]
23.25
0.228
0.089
22.49
0.244
0.089
+ Covis.
Pruning [24]
23.26
0.227
0.091
22.46
0.245
0.092
MCMC [19]
23.80
0.211
0.082
22.84
0.232
0.0843
+ Covis.
Pruning [24]
23.78
0.214
0.82
22.81
0.234
0.0841
2DGS [12]
20.67
0.313
0.103
19.822
0.329
0.103
+ Covis.
Pruning [24]
20.71
0.310
0.102
19.838
0.329
0.103
Table 2. Ablation Rendering. We compare a selection of recent
advancements in rendering within our framework. Results are aver-
aged over TUM RGBD [38], as this is one of the most challenging
benchmarks. We test this in P-RGBD mode. See the supplementary
for more techniques we have tried.
based SLAM frameworks [16, 24, 54] and hybrid systems
[13, 33, 56] like ours. Finally, we also compare to systems
based on volume rendering [59] or NeRF’s [29, 30].
Evaluation Metrics. For Rendering we report PSNR,
SSIM [45] and LPIPS [57] on the rendered keyframe images
against the groundtruth. For geometry, we compare the
rendered depth L1 [cm] error to the groundtruth sensor as
in [30]. In case of monocular depth, we compare the scale-
aligned depth [2]. For tracking we compare the ATE RMSE
[cm] [38] error on the estimated trajectory. Reported results
are averaged over 5 runs for statistical significance.
Tracking Ablation. Table 1 shows the importance of
individual tracking components in P-RGBD mode. These
results are mostly consistent across input modes and datasets,
see supplementary. We make the observation that the factor
graph building process is of most importance.
Integrating monocular priors with the scale optimization
is crucial. We did not have success with the Loop BA pro-
posed in [59] in our Backend. Instead, we achieve the best
results when adding visually similar loop candidates into our
graph. We also want to highlight, that we achieve SotA re-
sults by simply utilizing a more conservative graph building
strategy. See the supplementary for more details.
Renderer Ablation. Table 2 shows an ablation of re-
cently proposed Gaussian Splatting techniques. For this
Method
fr1/
desk
fr2/
xyz
f3/
off
f1/
desk2
f1/
room
Avg.
Mono
DPV-SLAM [21]
1.8
1.0
-
2.9
9.6
GlORIE-SLAM [56]
1.6
0.2
1.4
2.8
4.2
2.1
GO-SLAM [59]
1.6
0.6
1.5
2.8
5.2
2.3
MonoGS [24]
4.2
4.8
4.4
-
-
MoD-SLAM [61]
1.5
0.7
1.1
-
-
Photo-SLAM [13]
1.54
0.98
1.26
-
-
Ours Mono
1.6
0.2
1.6
8.3
5.8
3.5
Ours P-RGBD
1.6
0.2
1.7
2.3
3.3
1.8
RGBD
SplaTAM [16]
3.4
1.2
5.2
6.5
11.1
5.5
GS-SLAM [54]
1.5
1.6
1.7
-
-
GO-SLAM [59]
1.5
0.6
1.3
2.8
5.2
2.3
Photo-SLAM [13]
2.6
0.35
1.0
-
-
Ours RGBD
1.6
1.4
1.4
2.2
2.7
1.9
Table 3. Tracking Performance TUM-RGBD [38] (ATE RMSE↓
[cm]). Results are from respective papers. We achieve SotA track-
ing with our framework. Best results are highlighted as first ,
second , third .
ablation, we compare results without a refinement stage. We
detail additional experiments in the supplementary. We want
to highlight, that common comparisons should always factor
in the total number of Gaussians used. We use 120k Gaus-
sians on average on TUM-RGBD. Of course, using more
primitives will improve photo-realism at the cost of memory
and compute. We observe, that the covisibility pruning from
[24] is not necessarily effective on indoor datasets. Naive
map building can perform better, however at the price of
a few thousand more Gaussians. By far the most effective
improvement in our experiments is the MCMC guided den-
sification strategy [19], which gives a consistent boost in
rendering metrics compared to the naive gradient based den-
sification strategy [18]. In order to make a fair comparison
we match the total number of Gaussians to be equal with
both strategies. 2D Gaussian Splatting turned out to be inef-
fective on very cluttered indoor scenes such as TUM-RGBD,
either over-smoothing details or not building correct surfaces.
We also observe, that optimizing geometry typically comes
at the cost of worse rendering performance. We can control
this trade-off by tuning λ1.
4.1. Comparison with the State-of-the-Art
We evaluate on synthetic and real-world scenes and com-
pare with the State-of-the-Art. As can be seen in Table 3,
we achieve competitive tracking performance on real world
scenes. We want to highlight, that mostly fr1/desk2 and
fr1/room are challenging and therefore account biggest in the
average. The performance of most frameworks seems similar
on easier ones. We are also SotA on Replica across different
modes in Table 4 and 5. It can be seen, that traditional and
end-to-end tracking systems are still the best once perfect
supervision is missing. However with perfect synthetic data,
direct methods achieve strong results. Monocular methods,
that utilize a depth prior [33, 56, 61] generally perform better
on rendering and tracking due to the extra information. Our
5

<!-- page 6 -->
Metric
NeRf-
SLAM [30]
GO-
SLAM [59]
NICER-
SLAM [64]
MoD-
SLAM [61]
Photo-
SLAM [13]
Mono-
GS [24]
GlORIE-
SLAM [56]
Q-SLAM
[29]
Splat-
SLAM∗[33]
Ours
mono
Ours
P-RGBD
PSNR ↑
41.40
22.13
25.41
27.31
33.30
31.22
31.04
32.49
36.45
39.47
39.66
SSIM ↑
-
0.73
0.83
0.85
0.93
0.91
0.97
0.89
0.95
1.0
1.0
LPIPS ↓
-
-
0.19
-
-
0.21
0.12
0.17
0.06
0.03
0.03
L1 ↓
4.49
4.39
-
3.23
-
27.24
-
2.76
2.41
3.33
3.34
ATE RMSE ↓
-
0.39
1.88
0.35
1.09
14.54
0.35
-
0.35
0.27
0.27
Table 4. Avg. Rendering and Tracking Results on Replica [37] for monocular methods. Results are from [43] and the respective papers.
Metric
Vox-
Fusion [50]
NICE-
SLAM [63]
Mono-
GS [24]
Point-
SLAM [32]
SplatAM
[16]
Gaussian
SLAM [54]
Photo-
SLAM [13]
GO-
SLAM [59]
Ours
PSNR ↑
24.42
26.16
38.94
35.17
34.11
42.08
34.96
-
39.66
SSIM ↑
0.81
0.83
0.97
0.98
0.97
1.0
0.94
-
1.0
LPIPS ↓
0.23
0.23
0.07
0.12
0.10
0.02
0.06
-
0.03
L1 ↓
-
-
-
-
-
-
-
3.38
0.55
ATE RMSE ↓
3.09
2.35
0.32∗/0.58
0.53
0.36
0.31
0.60
0.34
0.29
Table 5. Avg. Rendering and Tracking Results on Replica [37] for RGB-D methods. Results are from the respective papers.∗result comes
from the slower single-process implementation.
method consistently ranks across the highest in rendering
due to the dense representation both in tracker and renderer.
Even though Photo-SLAM [13] utilizes a robust tracking
system [27], the sparse hyperprimitive optimization does not
allow indistinguishable renders. Figure 3 shows rendered im-
ages and depth maps. We achieve highly detailed geometry
on monocular video. Our monocular prior provides dense
guidance even when laser sensors have holes. Table 6 show-
GlORIE-SLAM [56]
MonoGS [24]
DROID-Splat (Ours)
Ground Truth
fr3
office
fr1
desk
Photo-SLAM [13]
MonoGS [24]
DROID-Splat (Ours)
Ground Truth
Figure 3. Rendering Results on TUM-RGBD [38]. We show views, that were not in the training set, i.e. our keyframe buffer. Top two
rows show monocular methods, bottom shows RGBD (We show the results with prior for ours). We achieve a higher rendering and depth
quality than [13, 24, 56] due to initializing with a dense tracking system and using dense hyperprimitives. Using a monocular prior can even
improve upon a sparse laser sensor.
6

<!-- page 7 -->
Method
Metric
f1/desk
f2/xyz
f3/off
f1/desk2
f1/room
Avg.
RGB-D
SplaTaM [16]
PSNR↑
22.00
24.50
21.90
-
-
SSIM ↑
0.86
0.95
0.88
-
-
LPIPS ↓
0.23
0.10
0.20
-
-
Gaussian-SLAM [54]
PSNR↑
24.01
25.02
26.13
23.15
22.98
24.26
SSIM ↑
0.92
0.92
0.94
0.91
0.89
0.92
LPIPS ↓
0.18
0.19
0.14
0.20
0.24
0.19
Photo-SLAM [13]
PSNR↑
20.87
22.09
22.74
-
-
SSIM ↑
0.74
0.77
0.78
-
-
LPIPS ↓
0.24
0.17
0.15
-
-
Ours
PSNR↑
26.45
28.45
27.83
25.13
26.16
26.81
SSIM ↑
0.99
0.99
0.99
0.99
0.99
0.99
LPIPS ↓
0.12
0.07
0.10
0.15
0.14
0.12
Mono
Photo-SLAM [13]
PSNR↑
20.97
21.07
19.59
-
-
SSIM ↑
0.74
0.73
0.69
-
-
LPIPS ↓
0.23
0.17
0.24
-
-
MonoGS [24]
PSNR↑
19.67
16.17
20.63
19.16
18.41
18.81
SSIM ↑
0.73
0.72
0.77
0.66
0.64
0.70
LPIPS ↓
0.33
0.31
0.34
0.48
0.51
0.39
GlORIE-SLAM [56]
PSNR↑
20.26
25.62
21.21
19.09
18.78
20.99
SSIM ↑
0.79
0.72
0.72
0.92
0.73
0.77
LPIPS ↓
0.31
0.09
0.32
0.38
0.38
0.30
Splat-SLAM∗[33]
PSNR↑
25.61
29.53
26.05
23.98
24.06
25.85
SSIM ↑
0.84
0.90
0.84
0.81
0.80
0.84
LPIPS ↓
0.18
0.08
0.20
0.23
0.24
0.19
Ours Mono
PSNR↑
26.72
29.35
27.92
24.58
25.64
26.84
SSIM ↑
0.99
0.99
0.99
0.99
0.99
0.99
LPIPS ↓
0.12
0.07
0.11
0.18
0.17
0.13
Ours P-RGBD
PSNR↑
26.42
28.08
27.84
25.21
25.11
26.53
SSIM ↑
0.99
0.99
0.99
0.99
0.99
0.99
LPIPS ↓
0.12
0.08
0.11
0.17
0.18
0.13
Table 6. Rendering Performance on TUM-RGBD [38]. Numbers are from respective papers. Our method performs competitively or better
than related RGB-D methods with monocular input. We achieve SotA results across all modes.
cases the SotA on TUM-RGBD. We consistently achieve
strong rendering metrics even on more challenging scenes
like fr1/room. We provide a more detailed overlook in the
supplementary with an evaluation protocol of non-training
frames. We observe in our experiments, that the benefit of
both monocular and sensor depth priors can be observed
mainly in the L1 reconstruction metric and on non-training
frames. The current evaluation protocol rewards overfitting
the scene. We also want to highlight from Table 5, that
the synthetic Replica [37] benchmark is saturated in RGBD
mode. Predicted images and groundtruth are already indis-
tinguishable at a PSNR of ≥39[dB]. The same can be said
about the predicted depths at L1 ≤0.6[cm]. We did not
achieve the same geometry quality on Replica as related
work [29, 33, 61], however the comparison is not fair since
the number of hyperprimitives and inference time was not
published in addition. We believe, that with more refinement
iterations, different input filter thresholds and loss hyperpa-
rameters, we could reach the same metrics. More details can
be found in the supplementary.
Can our renderer improve tracking? We can feedback
outputs of our rendering pipeline back into the tracking sys-
tem and verify this idea on Replica. We can backpropagate
gradients from rendering into our pose graph by finetuning
poses during the render updates. We then use the finetuned
poses in our next tracking update during bundle adjustment.
Feedback
ATE RMSE
KF [cm]
ATE RMSE
All [cm]
PSNR↑
LPIPS↓
L1↓
Replica
RGBD
None
0.293
0.289
36.03
0.06
0.0076
Disparity
0.294
0.304
35.99
0.06
0.0076
Poses
0.277
0.273
36.26
0.06
0.0074
Both
0.28
0.289
36.19
0.06
0.0075
P-RGBD
None
0.269
0.268
32.92
0.134
0.0374
Poses
0.356
0.348
32.95
0.134
0.0394
Both
0.34
0.349
32.9
0.135
0.04
TUM RGBD
RGBD
None
1.94
1.87
23.76
0.194
0.054
Disparity
1.99
1.92
23.65
0.199
0.055
Poses
1.98
1.89
23.72
0.197
0.056
Both
2.0
2.12
23.61
0.199
0.056
Table 7. Feedback. Since all components are differentiable, we can
finetune the pose graph based on the rendering objective and feed
this back into the tracking system. This requires a well initialized
rendering system and good dense groundtruth supervision. We
did not achieve stable results on more noisy monocular data. We
compare the average result across the dataset over 5 runs without
refinement.
We can also go one step further and use the rendering
objective to densify the disparity state of the tracker. In
7

<!-- page 8 -->
2
4
6
8
10
12
14
16
18
Speed [fps]
15
20
25
30
35
40
PSNR [dB]
TUM-RGBD (Online)
Replica (Online)
TUM-RGBD (Refined)
Replica (Refined)
TUM-RGBD (T
racker)
Replica (T
racker)
Figure 4. Compute-Performance Trade-off. We take the average
across TUM RGBD [38] and Replica [37] in RGBD mode. We
added the baseline Tracker at the bottom for perspective, this does
not have a meaningful Metric attached to it.
practice we perform a check to make sure that the difference
between rendered depth and tracker disparity is not too large
in order to not confuse the update network. Table 7 shows
different variations of this experiment in both RGBD mode
with perfect groundtruth and in P-RGBD mode. We observe,
that this actually works as long as perfect groundtruth is
available. Results worsen when using a monocular prior or
trying this idea on real data. We do not report monocular
experiments on TUM-RGBD, because the stability of the
tracker was severely affected.
Runtime analysis. The rendering performance heav-
ily depends on how much compute is spend on optimizing
the Gaussian primitives. We can give a scaling curve by
adjusting the optimization iterations of the renderer or the
frequency with which we use it. We also distinguish between
online mode and offline refinement and give a baseline speed
of our standalone tracker in Figure 4. We report results
on the rendered keyframes. We want to note that in case of
monocular depth priors, we are bottlenecked by the inference
speed of the depth prediction network.
In-the-wild Reconstruction. For in the-wild reconstruc-
tion, we tested both 3DGS [18] and 2DGS [12] on challeng-
ing outdoor scenes. We qualitatively analyzed both methods
on self-recorded videos. The difference in rendering qual-
ity seems to prevail on unbounded outdoor scenes with no
360-degree camera trajectory. Due to the challenging light-
ing conditions and much more unreliable monocular depth
priors, we accumulate many floater Gaussians. 2D Gaussian
Splatting does not suffer as much from these, as it generates
smooth surfaces. See Figure 5 for a visual comparison.
Failure Cases. Our method fails to handle challenging
lighting changes and lens flares without additional modi-
fications. In general, we perform much worse in sparser
scenarios or when our priors are unreliable. We also ob-
serve, that although our tracking system is robust, it can
Figure 5. Results on hand-captured cellphone videos. In-the-
wild outdoor scenes pose different challenges than benchmarks.
Left: 3D Gaussian Splatting. Right: 2D Gaussian Splatting. While
2DGS is more resistant to floaters due to its surface optimization, it
struggles with rendering quality. Both methods cannot deal well
with strong lighting changes and reflections without extensions.
still drift on much more challenging scenes and trajectories.
Since we optimize our hyperprimitives in batches, we are
prune to catastrophic forgetting as other methods and need
to reoptimize.
5. Conclusion
We combined a dense end-to-end SLAM system with a
photo-realistic renderer. We systematically ablated common
design choices and achieve SotA results with our framework
on common benchmarks. The integration of recent monocu-
lar depth priors allowed to close the gap between monocular
and RGBD SLAM both for odometry and rendering. Our
experiments show, that photorealistic rendering and accu-
rate geometry can be complementary objectives at this level,
where improving rendering performance comes at a cost of
worse geometry. At the same time, we did not see an im-
provement of our tracker based on the rendering objective for
natural scenes. Our framework is flexible and can seamlessly
reconstruct even in-the-wild video with unknown intrinsics.
Outlook. We hope that our Python framework enables
rapid experimentation and further research in combining
neural networks and SLAM. Recent foundation models [55]
allow to infer 3D scenes from images directly without test-
time optimization. The integration of such models poses an
exciting avenue for future research. Extending the system to
larger complex scenes would be another interesting direction.
8

<!-- page 9 -->
ACKNOWLEDGMENT.
We would like to say a spe-
cial thank you to Andrei Prioteasa for helping setup the
experiments and our colleagues at the Institute of Applied
Mathematics, Heidelberg for fruitful discussions.
References
[1] Gabriele Berton, Gabriele Trivigno, Barbara Caputo, and
Carlo Masone.
Eigenplaces: Training viewpoint robust
models for visual place recognition. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 11080–11090, 2023. 4, 12
[2] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter
Wonka, and Matthias Müller. Zoedepth: Zero-shot trans-
fer by combining relative and metric depth. arXiv preprint
arXiv:2302.12288, 2023. 2, 3, 5, 17, 18
[3] Angela Dai, Matthias Nießner, Michael Zollhöfer, Shahram
Izadi, and Christian Theobalt. Bundlefusion: Real-time glob-
ally consistent 3d reconstruction using on-the-fly surface rein-
tegration. ACM Transactions on Graphics (ToG), 36(4):1,
2017. 2
[4] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff John-
son, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria
Lomeli, Lucas Hosseini, and Hervé Jégou. The faiss library.
arXiv preprint arXiv:2401.08281, 2024. 4
[5] Xiaobiao Du, Yida Wang, and Xin Yu. Mvgs: Multi-view-
regulated gaussian splatting for novel view synthesis, 2024.
19
[6] Xiaobiao Du, Yida Wang, and Xin Yu. Mvgs: Multi-view-
regulated gaussian splatting for novel view synthesis, 2024.
2
[7] Jakob Engel, Thomas Schöps, and Daniel Cremers. Lsd-slam:
Large-scale direct monocular slam. In European conference
on computer vision, pages 834–849. Springer, 2014. 2, 3
[8] Jakob Engel, Vladlen Koltun, and Daniel Cremers. Direct
sparse odometry. IEEE transactions on pattern analysis and
machine intelligence, 40(3):611–625, 2017. 2
[9] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we
ready for autonomous driving? the kitti vision benchmark
suite.
In 2012 IEEE conference on computer vision and
pattern recognition, pages 3354–3361. IEEE, 2012. 17
[10] Annika Hagemann, Moritz Knorr, and Christoph Stiller. Deep
geometry-aware camera self-calibration from video. In Pro-
ceedings of the IEEE/CVF International Conference on Com-
puter Vision, pages 3438–3448, 2023. 2, 3, 18
[11] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li,
Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and Ying-
Cong Chen.
Lotus:
Diffusion-based visual foundation
model for high-quality dense prediction.
arXiv preprint
arXiv:2409.18124, 2024. 17, 18
[12] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In ACM SIGGRAPH 2024 Conference
Papers, pages 1–11, 2024. 2, 4, 5, 8, 12, 16
[13] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung.
Photo-slam: Real-time simultaneous localization and photo-
realistic mapping for monocular stereo and rgb-d cameras.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 21584–21593, 2024.
2, 4, 5, 6, 7, 14
[14] Olaf Kähler, Victor Adrian Prisacariu, Carl Yuheng Ren, Xin
Sun, Philip H. S. Torr, and David William Murray. Very
high frame rate volumetric integration of depth images on
mobile devices. IEEE Trans. Vis. Comput. Graph., 21(11):
1241–1250, 2015. 2
[15] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Met-
zger, Rodrigo Caye Daudt, and Konrad Schindler. Repurpos-
ing diffusion-based image generators for monocular depth
estimation. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 9492–9502,
2024. 18
[16] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat track & map 3d gaussians
for dense rgb-d slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
21357–21366, 2024. 2, 4, 5, 6, 7
[17] Maik Keller, Damien Lefloch, Martin Lambers, Shahram
Izadi, Tim Weyrich, and Andreas Kolb. Real-time 3d recon-
struction in dynamic scenes using point-based fusion. In 2013
International Conference on 3D Vision-3DV 2013, pages 1–8.
IEEE, 2013. 2
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and
George Drettakis. 3d gaussian splatting for real-time radiance
field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 1, 2,
4, 5, 8, 12, 13, 16, 18, 19
[19] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Jeff Tseng, Hossam Isack, Abhishek Kar, Andrea
Tagliasacchi, and Kwang Moo Yi. 3d gaussian splatting as
markov chain monte carlo, 2024. 2, 4, 5, 12, 13, 16
[20] Rainer Kümmerle, Giorgio Grisetti, Hauke Strasdat, Kurt
Konolige, and Wolfram Burgard. g 2 o: A general framework
for graph optimization. In 2011 IEEE international confer-
ence on robotics and automation, pages 3607–3613. IEEE,
2011. 2
[21] Lahav Lipson, Zachary Teed, and Jia Deng. Deep patch visual
slam. arXiv preprint arXiv:2408.01654, 2024. 1, 2, 4, 5
[22] Lorenzo Liso, Erik Sandström, Vladimir Yugay, Luc
Van Gool, and Martin R Oswald. Loopy-slam: Dense neural
slam with loop closures. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 20363–20373, 2024. 2, 4
[23] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng,
Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf,
and Jia-Bin Huang. Robust dynamic radiance fields. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 13–23, 2023. 3
[24] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18039–18048, 2024. 1, 2, 4, 5, 6, 7, 12,
14
[25] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
9

<!-- page 10 -->
Representing scenes as neural radiance fields for view synthe-
sis, 2020. 2
[26] Thomas Müller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. ACM transactions on graphics (TOG),
41(4):1–15, 2022. 2
[27] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D
Tardos. Orb-slam: a versatile and accurate monocular slam
system. IEEE transactions on robotics, 31(5):1147–1163,
2015. 1, 2, 3, 4, 6
[28] Matthias Nießner, Michael Zollhöfer, Shahram Izadi, and
Marc Stamminger. Real-time 3d reconstruction at scale using
voxel hashing. ACM Transactions on Graphics (TOG), 32,
2013. 2
[29] Chensheng Peng, Chenfeng Xu, Yue Wang, Mingyu Ding,
Heng Yang, Masayoshi Tomizuka, Kurt Keutzer, Marco
Pavone, and Wei Zhan. Q-slam: Quadric representations
for monocular slam. arXiv preprint arXiv:2403.08125, 2024.
5, 6, 7
[30] Antoni Rosinol, John J Leonard, and Luca Carlone. Nerf-
slam: Real-time dense monocular slam with neural radiance
fields. In 2023 IEEE/RSJ International Conference on Intel-
ligent Robots and Systems (IROS), pages 3437–3444. IEEE,
2023. 2, 5, 6
[31] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary
Bradski. Orb: An efficient alternative to sift or surf. In 2011
International conference on computer vision, pages 2564–
2571. Ieee, 2011. 4
[32] Erik Sandström, Yue Li, Luc Van Gool, and Martin R Os-
wald. Point-slam: Dense neural point cloud-based slam. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 18433–18444, 2023. 6
[33] Erik Sandström, Keisuke Tateno, Michael Oechsle, Michael
Niemeyer, Luc Van Gool, Martin R Oswald, and Federico
Tombari. Splat-slam: Globally optimized rgb-only slam with
3d gaussians. arXiv preprint arXiv:2405.16544, 2024. 2, 3,
4, 5, 6, 7, 13
[34] Aron Schmied, Tobias Fischer, Martin Danelljan, Marc Polle-
feys, and Fisher Yu. R3d3: Dense 3d reconstruction of dy-
namic scenes from multiple cameras. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 3216–3226, 2023. 1
[35] Thomas Schops, Torsten Sattler, and Marc Pollefeys. Bad
slam: Bundle adjusted direct rgb-d slam. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 134–144, 2019. 2
[36] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob
Fergus. Indoor segmentation and support inference from rgbd
images. In Computer Vision–ECCV 2012: 12th European
Conference on Computer Vision, Florence, Italy, October 7-
13, 2012, Proceedings, Part V 12, pages 746–760. Springer,
2012. 17
[37] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl
Ren, Shobhit Verma, et al. The replica dataset: A digital
replica of indoor spaces. arXiv preprint arXiv:1906.05797,
2019. 4, 6, 7, 8, 12, 13, 15, 18
[38] Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the evalua-
tion of rgb-d slam systems. In 2012 IEEE/RSJ international
conference on intelligent robots and systems, pages 573–580.
IEEE, 2012. 4, 5, 6, 7, 8, 12, 13, 14, 16, 18
[39] Shuo Sun, Malcolm Mielle, Achim J Lilienthal, and Martin
Magnusson. High-fidelity slam using gaussian splatting with
rendering-guided densification and regularized optimization.
arXiv preprint arXiv:2403.12535, 2024. 2, 19
[40] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field
transforms for optical flow. In Computer Vision–ECCV 2020:
16th European Conference, Glasgow, UK, August 23–28,
2020, Proceedings, Part II 16, pages 402–419. Springer, 2020.
2
[41] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam
for monocular, stereo, and rgb-d cameras. Advances in neural
information processing systems, 34:16558–16569, 2021. 1, 2,
3, 4, 12, 19
[42] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch
visual odometry. Advances in Neural Information Processing
Systems, 36, 2024. 1, 2
[43] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandström,
Stefano Mattoccia, Martin R Oswald, and Matteo Poggi. How
nerfs and 3d gaussian splatting are reshaping slam: a survey.
arXiv preprint arXiv:2402.13255, 4, 2024. 2, 6
[44] Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto
Seiskari, Esa Rahtu, and Juho Kannala. Dn-splatter: Depth
and normal priors for gaussian splatting and meshing, 2024.
2
[45] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 5, 12
[46] Thomas Whelan, Stefan Leutenegger, Renato F Salas-
Moreno, Ben Glocker, and Andrew J Davison. Elasticfusion:
Dense slam without a pose graph. In Robotics: science and
systems, page 3. Rome, Italy, 2015. 2
[47] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay,
Pradyumna Chari, and Achuta Kadambi. Sparsegs: Real-time
360° sparse view synthesis using gaussian splatting. Arxiv,
2023. 19
[48] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with
3d gaussian splatting. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
19595–19604, 2024. 2
[49] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi
Feng, and Hengshuang Zhao. Depth anything: Unleashing
the power of large-scale unlabeled data. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10371–10381, 2024. 2, 3, 17, 18
[50] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian
Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and
mapping with voxel-based neural implicit representation. In
2022 IEEE International Symposium on Mixed and Aug-
mented Reality (ISMAR), pages 499–507. IEEE, 2022. 6
[51] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaix-
uan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d:
10

<!-- page 11 -->
Towards zero-shot metric 3d prediction from a single image.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 9043–9053, 2023. 2, 3, 4, 5, 13,
16, 17, 18
[52] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 19447–19456,
2024. 2
[53] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes, 2024. 2
[54] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Os-
wald. Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting. arXiv preprint arXiv:2312.10070, 2023. 2, 4,
5, 6, 7
[55] Chubin Zhang, Hongliang Song, Yi Wei, Yu Chen, Jiwen
Lu, and Yansong Tang. Geolrm: Geometry-aware large re-
construction model for high-quality 3d gaussian generation.
arXiv preprint arXiv:2406.15333, 2024. 8
[56] Ganlin Zhang, Erik Sandström, Youmin Zhang, Manthan
Patel, Luc Van Gool, and Martin R Oswald. Glorie-slam:
Globally optimized rgb-only implicit encoding point cloud
slam. arXiv preprint arXiv:2403.19549, 2024. 2, 3, 5, 6, 7,
14
[57] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In Proceedings of the IEEE
conference on computer vision and pattern recognition, pages
586–595, 2018. 5
[58] Wei Zhang, Tiecheng Sun, Sen Wang, Qing Cheng, and Nor-
bert Haala. Hi-slam: Monocular real-time dense mapping
with hybrid implicit fields. IEEE Robotics and Automation
Letters, 2023. 2, 3, 4
[59] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo
Poggi. Go-slam: Global optimization for consistent 3d instant
reconstruction. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 3727–3737, 2023. 4,
5, 6
[60] Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and
Hengshuang Zhao.
Pixel-gs: Density control with pixel-
aware gradient for 3d gaussian splatting.
arXiv preprint
arXiv:2403.15530, 2024. 2
[61] Heng Zhou, Zhetao Guo, Shuhong Liu, Lechen Zhang, Qihao
Wang, Yuxiang Ren, and Mingrui Li. Mod-slam: Monocular
dense mapping for unbounded 3d scene reconstruction. arXiv
preprint arXiv:2402.03762, 2024. 5, 6, 7
[62] Liyuan Zhu, Yue Li, Erik Sandström, Konrad Schindler, and
Iro Armeni. Loopsplat: Loop closure by registering 3d gaus-
sian splats. arXiv preprint arXiv:2408.10154, 2024. 2
[63] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun
Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys.
Nice-slam: Neural implicit scalable encoding for slam. In
Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 12786–12796, 2022. 6
[64] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui,
Martin R Oswald, Andreas Geiger, and Marc Pollefeys. Nicer-
slam: Neural implicit scene encoding for rgb slam. In 2024
International Conference on 3D Vision (3DV), pages 42–52.
IEEE, 2024. 6
11

<!-- page 12 -->
Supplementary Material
In this supplementary material, we provide more details on
our approach, experiment settings and further experimental
results. We encourage readers to take a look at our open-
sourced implementation upon publication for more detailed
configuration. All experiments are performed on a desktop
computer with an AMD Ryzen Threadripper PRO 5955WX
CPU and NVIDIA RTX4090 GPU.
1. Inference settings and hyperparameters
We run our system at resolution 320 × 432 on TUM-
RGBD [38] and 360 × 640 on Replica [37].
1.1. Tracking
Tracking is configured by the frontend and backend parame-
ters for graph building, optimization and our loop detector.
Since the configuration system is quite complex, we will
only brush across the critical parameter settings. We will
release full configurations upon publication.
Frontend.
We use a motion threshold of 3.0 for adding
new keyframes. During scale optimization we use the objec-
tive
 \ l a bel {eq:reg_weight} E = E + \alpha E_{reg} 
(8)
with α = 0.001. We found it important to keep keyframes
longer in the frontend bundle adjustment window. This can
be controlled with the max age variable in [41]. We increase
this value from 25 to 30. We use a weight β = 0.5 on TUM-
RGBD [38] and β = 0.7 on Replica [37] for measuring
frame distance, see [41].
Backend.
We run the backend every 8 frontend passes in
our experiments. We build our global graph more conserva-
tively by using a window of max. 150 frames, using up to
1500 edges.
Loop Detector.
We compute visual features by using the
EigenPlaces [1] ResNet50 network. We found qualitatively,
that a feature threshold τf = 0.5 works well in practice.
We make the assumption that loop candidates are at least
τt = 10 frames apart. For our orientation threshold, we set
τr = 15◦. This assumes that during a loop closure we have
a very similar orientation, but due to drift a very distinct
translation.
1.2. Rendering
We run our mapper every 20 frontend calls and optimize for
100 iterations at a small delay of 5 frames. We found that in
practice, this can be arbitrarily tuned, i.e. we could also run
more frequent with less training iterations. We anneal a 3D
positional learning rate between [1e −4, 1e −6] during our
optimization, the other parameters are similar to [24]. During
each iteration, we optimize newly added frames and add the
10 last frames and 20 random past frames simmilar to [24].
Since we run a test-time optimization, we are also prone
to catastrophic forgetting. Using enough random frames
ensures that this does not happen. Our system is not yet
designed to handle large-scale scenes or unbounded scenes,
where smarter strategies may be needed.
After filtering the tracking map with a covisibility check
[41], we downsample the point cloud with factor 64 on
Replica [37] and 16 on TUM-RGBD [38]. We use the same
thresholds for densification across datasets. We made the ex-
perience that balancing these parameters can result in similar
results as long as the total number of Gaussians is simi-
lar. We weight L1-error and SSIM [45] in our appearance
loss with λ2 = 0.2. We balance depth and appearance su-
pervision with λ1 = 0.9 on TUM-RGBD [38] and 0.8 on
Replica [37]. Balancing these two terms, can shift metrics
slightly in favor of either appearance or geometry. Similar
to [24] we encourage isotropic Gaussians with their scale
regularization term. Our officially reported metrics are for
dense depth supervision when a prior exists. For monocular
video, we use the filtered tracking map as depth guidance.
Monte
Carlo
Markov
Chain
Gaussian
Splatting.
MCMC Gaussian Splatting [19] has additional hyperparam-
eters for the noise level and the max. number of Gaussians
in a scene. This poses an upper limit beyond which cannot
be densified. We use lrnoise = 1e4 and use a slightly lower
number of Gaussians from the runs with vanilla 3D Gaussian
Splatting [18].
2D Gaussian Splatting
2D Gaussian Splatting has a
slightly different objective function [12]. On top of the de-
fault rendering objective, we also have a normal consistency
Lnormal and depth distortion loss Ldist. We also found, that
this representation has a different learning dynamic than 3D
Gaussians. We therefore tuned the weighting to the best of
our ability (without extensive parameter sweeps).
1.3. Feedback
In our feedback experiment, we perform backpropagation
on the local pose graph of our rendering batch as is done
in [24]. We used vanilla 3D Gaussian Splatting with the
adaptive density control densification strategy [18]. As a
sanity check, we only feedback the poses and/or disparity
of rendered frames that have a decently similar disparity to
the tracking map. The reason for this lies in the fact that
our renderer has a small delay behind the leading tracker.
If the rendering map is yet not covering enough pixels for
some reason, we could potentially feedback a much sparser
frame than we initially used during tracking. This could
potentially disturb the update network [41]. We therefore
12

<!-- page 13 -->
Technique
#Gaussians
PSNR↑
LPIPS↓
L1↓
PSNR↑
LPIPS↓
L1↓
KF
Non-KF
TUM-RGBD
Monocular
118 889
26.84
0.129
16.67
24.62
0.156
17.37
P-RGBD
119 100
26.53
0.131
8.50
24.81
0.155
8.38
RGBD
123 232
26.81
0.110
4.26
24.89
0.144
4.63
Replica
Monocular
246 637
39.47
0.031
3.33
38.42
0.032
3.47
P-RGBD
248 175
39.66
0.029
3.34
38.27
0.031
3.53
RGBD
235 825
39.66
0.028
0.55
38.87
0.029
0.61
Table 8. Full Rendering results. We report our overall best results with MCMC densification [19] averaged over 5 runs on TUM-RGBD [38]
and Replica [37] with refinement.
Supervision
#Gaussians
PSNR↑
LPIPS↓
L1↓
PSNR↑
LPIPS↓
L1↓
KF
Non-KF
TUM-RGBD
dense
88 280
25.98
0.140
8.2
24.48
0.161
8.2
sparse
100 156
26.67
0.129
15.6
24.37
0.155
16.7
Replica
dense
264 343
38.79
0.0361
3.55
37.84
0.0371
3.64
sparse
275 997
38.95
0.0347
2.96
37.82
0.0361
3.11
Table 9. Sparse vs. dense supervision of vanilla 3D Gaussian Splatting [18] with monocular prior. Geometry reconstruction depends
heavily on the degree and quality of supervision. TUM-RGBD [38] does not have enough redundancy in frames for the filtered map to cover
the scene. Replica [37] on the other hand will produce enough reliable covisible 3D points, such that the filtered tracking map provides
strong supervision for each Gaussian. For this reason, we can achieve better results when using the sparser, filtered map on Replica. This
closes the gap to related work [33]. We believe that with different priors and hyperparameters, we would achieve the same L1 error.
check that the abs. rel. error between rendered disparity and
tracking disparity is ≤0.2. If at least 50% of pixels satisfy
this condition, then the frame is considered good.
2. Extended Evaluation
In this section, we want to provide more insights into how our
system performs quantitatively and show more qualitative
results. The reported rendering metrics for our comparison
with related work are computed on the keyframe images
based on the estimated poses, as is standard. However, this
can give a warped view on the quality of a method. We want
to highlight several key points:
• Every method has a different keyframe management or
builds their graph based on different thresholds.
• Not all metrics are reportedly available on all datasets.
We omitted an extensive evaluation of related work due
to time constraints. Example: L1 metric is only readily
available for Replica [37], however due to being a virtual
dataset this metric is already quite saturated. The TUM-
RGBD [38] benchmark is much more interesting.
• Performance should be measured both on training and
other frames! Generalization of our test-time optimization
is what normally counts, which is why we report results
on non-training frames.
• The difference between modes only becomes apparent
when considering both geometry and predicted images
for both training and other frames.
We show our full evaluation metrics of the overall best
results in Table 8. We can only see a clean progression
from monocular to RGBD inputs on the challenging TUM-
RGBD [38] benchmark. We want to highlight, that strict
monocular methods can overfit the appearance of training
frames very well independent of tracking accuracy or ge-
ometric accuracy. However, we can generalize better and
achieve much more accurate geometry when using additional
depth priors. The benefit of a monocular prior [51] seems
to be much smaller on Replica [37]. We found out in Table
9, that depending on the depth supervision signal this result
changes. We also suspect concurrent work [33] to supervise
with a filtered depth map for this reason. Figure 6 and 7 show
qualitative examples on top to get a feeling for how good
methods work. We specifically chose non-training frames,
which might put us at a disadvantage. We can observe clear
improvements on fine-structured details, such as the lamp or
background.
Due to our dense map both in tracking and rendering we
13

<!-- page 14 -->
GlORIE-SLAM [56]
MonoGS [24]
DROID-Splat (Ours)
Ground Truth
fr1
room
fr2
xyz
fr1
room
fr2
xyz
Photo-SLAM [13]
MonoGS [24]
DROID-Splat (Ours)
Ground Truth
Figure 6. More Rendering Results on TUM-RGBD [38]. Top four rows are from monocular input, bottom from RGBD.
14

<!-- page 15 -->
Mono
P-RGBD
RGBD
Ground Truth
office 2
room 0
room 1
room 2
Figure 7. Rendering Results on Replica [37]. We show non-training frames in multiple input modalities. Note how visually close the
predictions are to the groundtruth.
15

<!-- page 16 -->
Technique
# Gaussians
PSNR↑
LPIPS↓
L1↓
PSNR↑
LPIPS↓
L1↓
KF
Non-KF
no refinement
2DGS [12]
173 309
20.71
0.31
10.2
19.84
0.33
10.3
3DGS [18]
111 878
23.26
0.23
9.1
22.46
0.25
9.2
+ MCMC [19]
113 060
23.78
0.21
8.2
22.81
0.23
8.4
with refinement
2DGS [12]
131 576
22.87
0.21
8.8
21.73
0.23
8.7
3DGS [18]
88 280
25.98
0.14
8.2
24.47
0.16
8.2
+ MCMC [19]
119 100
26.53
0.13
8.5
24.81
0.15
8.4
Table 10. Ablation Rendering Techniques. We report results averaged over 5 runs on TUM-RGBD [38] in P-RGBD mode using [51] as a
prior. We show a small progression with and without refinement. While 2D Gaussian Splatting [12] quickly produces smooth surfaces, this
is not rewarded in the L1 error metric.
can achieve better reconstructions than related work. For
monocular reconstructions, we specifically show our results
with a depth prior [51], which achieves much more accurate
geometric reconstruction and better photo-realism on non-
training frames than the monocular counter-part. This holds
true even for slightly worse L1 metrics on Replica, as can
be seen in the qualitative images. Results on Replica are
already so accurate, that slight scale differences across time
can create slightly non-flat walls. Table 10 shows a detailed
ablation of Rendering techniques. We did not combine 2D
Gaussian Splatting with the improved densification strategy
[19], however we expect this to gain a similar improvement.
We did not succeed in achieving better reconstructions for
2D Gaussian Splatting on TUM-RGBD [38]. However, we
3D Gaussian Splatting
2D Gaussian Splatting
Reference
17.53 dB
20.56 dB
24.06 dB
22.29 dB
21.64 dB
20.24 dB
Figure 8. Geometry vs. appearance. We found, that 2D Gaussian Splatting [12] can quickly create smooth surfaces and does not accumulate
many floaters in outdoor scenes. However, the rendering quality lacks behind 3D Gaussian Splatting [18] and as long as good supervision
exists we can achieve better L1 metrics with 3D Gaussian Splatting.
16

<!-- page 17 -->
observe a clear benefit of this representation similar to the
results in the respective paper, see examples in Figure 8. We
can quickly converge to flat surfaces, which helps to avoid
many floaters in outdoor-scenarios. On the used indoor
datasets, vanilla 3D Gaussians perform better.
3. Monocular Depth Prediction
Monocular depth prediction is a longstanding task with very
impressive in-the-wild results of recent SotA models [2, 11,
49, 51]. We show some qualitative comparisons between
selected models in Figure 9. Due to training on massive
datasets, current single-image depth predictions can recover
fine-structured details. Nonetheless, the accuracy of rel.
depth on a single frame is not the only thing that matters for
SLAM. We want to highlight:
• The rel. depth error on a single image should be minimal.
This is obvious, however most recent models are only
evaluated on specific benchmarks such as e.g. KITTI [9]
or NYU [36]. Even though model predictions can look
qualitatively very different, their abs. rel. error does not
seem to be that different on untypical depth prediction
benchmarks.
• Temporal consistency matters a lot. Even though we op-
Lotus [11]
DepthAnything [49]
Metric3D [51]
Reference
Figure 9. Monocular depth prediction in-the-wild. Models exhibit very clear differences w.r.t captured details and scale consistency on
self-captured video. Problems can arise in particular for reflective surfaces or paintings.
17

<!-- page 18 -->
Prior
ATE RMSE↓
PSNR↑
LPIPS↓
L1↓
ATE RMSE↓
PSNR↑
LPIPS↓
L1↓
KF
Non-KF
TUM-RGBD
Metric3D [51]
1.93
23.27
0.226
0.091
1.83
22.48
0.242
0.089
ZoeDepth [2]
1.97
23.21
0.233
0.132
1.87
22.34
0.249
0.136
DepthAnything [49]
1.91
23.24
0.229
0.098
1.79
22.43
0.246
0.099
Lotus [11]
2.45
22.84
0.256
0.297
2.39
21.84
0.273
0.313
Replica
Metric3D [51]
0.269
32.92
0.134
0.037
0.268
32.62
0.134
0.038
ZoeDepth [2]
0.266
33.24
0.123
0.088
0.265
32.89
0.123
0.091
DepthAnything [49]
0.268
33.06
0.131
0.063
0.268
32.73
0.131
0.066
Lotus [11]
0.275
32.23
0.116
0.295
0.278
31.72
0.118
0.318
Table 11. Ablation Prior Depth on Replica [37] and TUM-RGBD [38]. Recent SotA depth prediction networks [2, 11, 49, 51] have
different qualities for SLAM. Good temporal consistency allows accurate geometry reconstruction. However, rendering quality and tracking
does not necessarily correlate with it. Results are after online mapping without any refinement using vanilla 3D Gaussian Splatting [18] and
averaged over 5 runs.
timize scale si and shift oi parameters to match our per-
ceived optical flow, models result in differently consistent
integrated maps. It is still very beneficial to have high
temporal scale consistency in a depth model.
Recent diffusion models [11, 15] can leverage billion-
scale text-to-image pretraining to achieve strong depth pre-
diction results with little finetuning. As can be seen in Figure
9, the qualitative difference and recovered fine-structured
details compared to models trained only on million-scale
depth prediction datasets seems obvious. However, diffusion
models exhibit strong scale differences across a video. This
seems to create a lot of floaters, in part enhanced due to the
high-frequency details. We did not see an improvement for
SLAM by integrating these models for this reason. Table
11 shows the performance of our system with vanilla 3D
Gaussian Splatting [18]. We observe that Metric3D [51]
consistently optimizes the best geometry. However, other
metrics are not always consistent.
4. How important is camera calibration really?
In this section we want to show some qualitative examples
of in-the-wild footage with unknown intrinsics. As stated in
the main paper, we perform a two-stage reconstruction:
1. Run the system without scale-optimization and optimize
the camera intrinsics θ.
2. Use the now calibrated camera to run in P-RGBD mode
and additionally optimize si and oi
Since we need an initial estimate of the intrinsics, we assume
a heuristic where for a pinhole camera
  f x =  f y  =  \
le f t (
 H  + W
 \right ) / 2 \nonumber \\ cx = W / 2 \quad cy = H / 2 \quad .
(9)
The benefit of camera calibration was quantitatively shown
in [10]. We report qualitative results on self-recorded scenes
and show the robustness when initializing from a heuristic. It
can be seen in Figure 10, that both intrinsics calibration and
scale optimization are beneficial for in-the-wild reconstruc-
tion. With wrong intrinsics, we observe distorted odometry
and structure. With scale optimization, we can generate glob-
ally consistent maps. All together forms a good basis for
rendering.
18

<!-- page 19 -->
wo calibration
w calibration
w scale
w calibration
wo calibration
wo  calibration
wo  calibration
Figure 10. Camera calibration and prior integration matter. Distortion effects and artifacts both on the map and camera odometry can
be observed without calibration. Using our strategy, we can get rid of distortions. The scale-optimized prior integration allows accurate
structure reconstruction. Outdoor scenes require all together due to scale inconsistencies of common depth prediction models.
5. Failure Cases
Due to the challenging unbounded outdoor setting on uncal-
ibrated cameras, we quickly observed common limitations
of our framework. We notice that even though monocular
depth prediction networks allow highly detailed single-frame
predictions, their usage on in-the-wild video is limited. Scale
inconsistencies and inaccurate predictions make us accumu-
late floaters over time. We therefore have to use the fol-
lowing: We limit depth supervision to consistent 3D points
using the covisibility check [41] and pixels with confidence
σi ≥0.1. This removes the sky and many floaters, but can
also underconstraint the scene.
6. What did not work?
We tried the following things unsuccessfully:
• Multi-View Gaussian Splatting [5] backprojects crops of
2D appearance error into 3D by using the camera ray.
We can then perform an intersection test to carve out a
3D volume across multiple views. This test identified
new Gaussians, that cause a high 2D error, but were not
identified in the original densification strategy [18]. How-
ever, we did not manage to improve densification this way
within our framework.
• [39] uses a regularization term to battle catastrophic for-
getting. We did not succeed on improving our metrics
this way. We further tried to simply scale the gradients of
optimized Gaussians by the number of times its frame has
been already optimized by the renderer.
Bad Priors
Challenging
Illumination
Figure 11. Common failure cases. Since we are heavily dependent
on depth priors on in-the-wild video, our method can fail when
priors get unreliable. Similarly, if the geometry supervision is not
good enough, we accumulate floaters on outdoor scenes. Challeng-
ing lighting conditions can enhance this effect, since our model
will overfit the scene and create additional Gaussians for modeling
lighting effects (see Gaussians surrounding object).
• Sparse GS [47] uses a softmax for rendering depths. We
can identify floaters on outdoor scenes by analyzing the
modality of the depth distribution. Since we compute an
integrated absolute depth and supervise with priors, we
were not able to converge quickly to the correct values
due to the used logarithm function. Since Sparse GS was
created with rel. depth supervision, we did not pursue this
further.
19
