<!-- page 1 -->
AirGS: Real-Time 4D Gaussian Streaming for
Free-Viewpoint Video Experiences
Zhe Wang, Jinghang Li and Yifei Zhu∗
Global College, Shanghai Jiao Tong University
Email: 123369423@sjtu.edu.cn, sjtulijinghang@sjtu.edu.cn, yifei.zhu@sjtu.edu.cn
Abstract—Free-viewpoint video (FVV) enables immersive view-
ing experiences by allowing users to view scenes from arbitrary
perspectives. As a prominent reconstruction technique for FVV
generation, 4D Gaussian Splatting (4DGS) models dynamic
scenes with time-varying 3D Gaussian ellipsoids and achieves
high-quality rendering via fast rasterization. However, existing
4DGS approaches suffer from quality degradation over long
sequences and impose substantial bandwidth and storage over-
head, limiting their applicability in real-time and wide-scale de-
ployments. Therefore, we present AirGS, a streaming-optimized
4DGS framework that rearchitects the training and delivery
pipeline to enable high-quality, low-latency FVV experiences.
AirGS converts Gaussian video streams into multi-channel 2D
formats and intelligently identifies keyframes to enhance frame
reconstruction quality. It further combines temporal coherence
with inflation loss to reduce training time and representation size.
To support communication-efficient transmission, AirGS models
4DGS delivery as an integer linear programming problem and
design a lightweight pruning level selection algorithm to adap-
tively prune the Gaussian updates to be transmitted, balancing
reconstruction quality and bandwidth consumption. Extensive
experiments demonstrate that AirGS reduces quality deviation in
PSNR by more than 20% when scene changes, maintains frame-
level PSNR consistently above 30, accelerates training by 6×,
reduces per-frame transmission size by nearly 50% compared to
the SOTA 4DGS approaches.
Index Terms—Free-viewpoint video, 4D Gaussian Splatting,
video streaming
I. INTRODUCTION
Free-viewpoint video (FVV) allows users to explore a
dynamic scene from arbitrary viewpoints, offering an immer-
sive and interactive visual experience. To generate the FVV,
the straightforward way is to reconstruct the scene into 3D
model from captured multi-view 2D image sequences. Existing
dynamic scene reconstruction methods either explicitly model
scene geometry using volumetric or mesh-based primitives [1],
[2], or synthesize novel views through image-based interpola-
tion [3]–[5]. However, these methods often struggle to achieve
high reconstruction quality in real-world scenes with complex
geometry and rich appearance variations [6].
Recently, 3D Gaussian Splatting (3DGS) [8] has emerged
as a highly efficient method for novel view synthesis, offering
superior rendering quality and speed. The core idea is to rep-
resent a static scene as a set of 3D Gaussian ellipsoids learned
via optimization, and to render them efficiently using fast ras-
terization. Building on this breakthrough, several efforts have
This work is supported by the National Key R&D Program of China (Grant
No. 2024YFC3017100) and NSFC No. 62302292. ∗Corresponding author.
extended 3DGS to 4D dynamic scene modeling [9], [10]. How-
ever, these methods typically use a fixed number of Gaussian
ellipsoids across all frames, which limits reconstruction quality
over long sequences and increases per-frame storage overhead.
To address this, 3DGStream [6] introduces a keyframe-based
design, where a reference frame (anchor) is represented with
full 3D Gaussians, and subsequent frames capture dynamics
via a lightweight MLP that predicts Gaussian updates in the
anchor’s space. To reduce the runtime cost associated with
frequent MLP queries, V 3 [7] further stores precomputed MLP
outputs as multi-channel 2D images, enabling faster decoding
with minimal computational overhead.
These advancements demonstrate the potential of 4DGS-
based FVV generation. However, streaming 4DGS content in
practical network environments remains largely unexplored.
Fundamentally, several key challenges persist towards achiev-
ing real-time deployment. First, while the introduction of
keyframe and the auxiliary MLP significantly reduces com-
munication cost, keyframe selection remains an open problem.
Most of existing studies fix the set of Gaussian primitives
in the anchor space after initialization. Consequently, simply
selecting the first frame as a keyframe often leads to noticeable
quality degradation and loss of image details, demonstrated in
Fig. 1, since early frames may lack information about later-
appearing objects or large motions. Second, training a full
3DGS model for each keyframe is computationally expen-
sive. Naively increasing the number of keyframes improves
reconstruction quality, but significantly increases training time
and resource usage. Both highlight the need for an efficient
and scalable training strategy. Third, transmitting a full 3DGS
for every frame incurs substantial bandwidth and latency
overhead, making it difficult to satisfy real-time requirements
in practical fluctuating networks. Therefore, a communication-
efficient streaming framework is also needed to ensure a
smooth and high-quality FVV experience.
To address these challenges, we propose AirGS, a Gaussian-
based 4D video streaming framework designed for high-
quality reconstruction over long-sequence and smooth render-
ing under fluctuating network conditions. AirGS builds upon
the idea of converting the dynamic Gaussian sequence into
multi-channel long-sequence 2D Gaussian frames, enabling
compatibility with existing video codecs and facilitating ef-
ficient streaming. It systematically redesigns two fundamental
components of the 4DGS pipeline: the training strategy and the
streaming framework. In training, AirGS intelligently selects
arXiv:2512.20943v1  [cs.GR]  24 Dec 2025

<!-- page 2 -->
(a) Dynamic scene input
PSNR Dropping:
3.01
PSNR: 17.35
(b) 3DGStream [6]
PSNR Dropping:
2.38
PSNR: 28.71
(c) V 3 [7]
PSNR Dropping:
-2.23
PSNR: 31.41
(d) The proposed method: AirGS
Fig. 1.
We present AirGS, a novel framework for real-time, high-quality 4DGS-based FVV generation and streaming in practical networks. The PSNR
dropping refers to the difference between the reconstruction quality of the first frame and that of the current frame (the 25th frame in this example). The red
PSNR value indicates the quality of the region highlighted by the red box.
Input images
Training
Rendering
3DGS
User view
Server
Independently for frame 0, 1, ..., n
User
Streaming
(a) 3DGS
Key frame
Training
Canonical
3DGS
Subsequent
frames' 3DGS 
Server
Frame 0
Estimate
User
Streaming
User view
Rendering
(b) Typical 4DGS
Key frame
Canonical
3DGS
Training
Estimate
Transform
Server
Subsequent
frames' 3DGS 
Transformed
Data
Frame 0
User
Rendering
Reconstruct
User client
User view
Streaming
(c) Streamable 4DGS
Fig. 2. Illustration of 3DGS, typical 4DGS and streamable 4DGS for FVV
generation
frames with quality degradation as keyframes to initiate new
canonical spaces. To reduce retraining overhead, it exploits
temporal continuity by incorporating the neighboring frame
of the keyframe and introduces a novel inflation loss to
accurately reconstruct new objects and large motions while
minimizing additional Gaussian primitives and training costs.
For streaming, AirGS transmits only Gaussian deltas be-
tween consecutive frames, leveraging the inter-frame similarity
and sparsity we observe in 4DGS. It further incorporates a
frequency-based importance evaluator and a network-aware
Gaussian-pruner to adapt to the fluctuating networks, reducing
transmission overhead while preserving rendering fidelity. To
our knowledge, AirGS is the first streaming-optimized 4DGS-
based free-viewpoint video generation framework. In sum-
mary, our contributions are as follows:
• We propose AirGS, a novel streaming-optimized 4DGS
framework that enables high-quality real-time FVV gen-
eration and streaming over dynamic networks.
• We propose a quality-driven keyframe selection strategy
to capture abrupt object appearances and large motions.
We further exploit temporal continuity and introduce a
Gaussian inflation loss to further reduce training cost and
representation size.
• We design an adaptive pruning strategy to dynamically
remove low contribution Gaussian primitives based on
network conditions.
• Extensive experiments demonstrate that AirGS reduces
PSNR deviation by more than 20% when scene changes,
maintains frame-level PSNR consistently above 30, accel-
erates training by 6×, and reduces per-frame transmission
size by nearly 50% compared to the state-of-the-art 4DGS
approaches.
The rest of this paper is organized as follows. We present
background on 3DGS and existing 4DGS efforts towards FVV
in Section II. Section III details the system design of AirGS.
Section IV presents a comprehensive evaluation of AirGS.
Finally, we discuss related work in Section V and conclude
the paper in Section VI.
II. BACKGROUND
3D Gaussian Splatting (3DGS) [8] provides high-fidelity
3D reconstruction and novel view synthesis by integrating the
strengths of both implicit and explicit rendering. It reconstructs
3D scenes from point clouds generated from Structure-from-
Motion (SfM) process [11] by modeling each point as an
anisotropic Gaussian ellipsoid. Each ellipsoid or primitive is
characterized by a full 3D covariance matrix Σ, its center
spatial position µ = (x, y, z), its opacity α, the color c, and its
spherical harmonics, SH, to reflect the view-dependent effect.
The covariance matrix is used to control its geometry, which
can be further divided into a scaling matrix S and a rotation
matrix R. Thus, a Gaussian primitive i can be defined as
Gi = {µi, αi, ci, SHi, Ri, Si}. The value of these variables
can be obtained through machine learning with a differentiable
splatting renderer. The training loss is a combination of mean
absolute loss L1 and structural dissimilarity loss LD−SSIM:
L3DGS = (1 −λ)L1 + λLD−SSIM
(1)

<!-- page 3 -->
where λ is a weighting hyperparameter.
An intuitive approach towards free-viewpoint video con-
struction is to independently train all frames as independent
3DGS models, as illustrated in Fig. 2(a). However, this leads
to excessive computational cost. In contrast, to avoid the
complexity and high cost of performing full 3DGS training
for every frame, most of existing 4DGS studies construct
an anchor space as a reference and then reconstruct the
subsequent frames using motion estimation, as illustrated in
Fig. 2(b). Specifically, given a sequence of frames, a 3DGS
is first trained for the initial frame, forming the canonical
Gaussian space GC0 for the entire video. The motion of
Gaussian attributes di between consecutive frames (i−1) and
i is then estimated, allowing the Gaussian representation of
any frame t to be derived by the following equation.
GSt =
n
X
i=0
Gi
t = GC0 +
t
X
i=1
di
(2)
Streamable 4DGS methods, such as V 3 [7] in Fig. 2(c), encode
Gaussian primitives into streaming-friendly data formats, such
as 2D images. Specifically, all Gaussians for the current frame
reconstruction are represented by a set of images, with each
image encoding a single attribute, such as opacity, scale, etc.
The different attributes of the same Gaussian are stored at the
same pixel location across these images. After streaming all
these images to the client, the GSt, which has n Gaussian
primitives, can be easily obtained by Equ. 3 and subsequently
employed for local rendering.
GSt =
n
X
i=0
Gi(t) =
n
X
i=0
m
X
j=0
aj(i)
(3)
where ai(j) is the value of pixel i in image j, and m is the
number of attributes, which also corresponds to the number of
images transmitted for that frame.
III. SYSTEM DESIGN
A. System overview
The system overview of AirGS is shown in Fig. 3. AirGS
redesigns both the training and streaming processes of Gaus-
sian primitives. During training, AirGS first groups frames
by reconstruction quality, designating any frame below the
quality threshold as the keyframe of a new group to capture
major scene changes. For the keyframe in the first group,
AirGS simply follows the traditional 3DGS training strategy.
For keyframe reconstruction in subsequent groups, AirGS
introduces Gaussian inflation loss combined with Gaussian
densification to enable fast, high-quality initialization with
little memory overhead. For frame training within a group,
AirGS builds on existing canonical Gaussian primitive space
and leverages inter-frame similarity to estimate Gaussian mo-
tion of the canonical space for reconstruction. In streaming,
rather than transmitting complete Gaussian ellipsoids GSt at
each timestep, AirGS streams the Gaussian differences be-
tween adjacent frames, forming a sparse, compression-friendly
matrix that significantly reduces transmission cost. To handle
000
Input Frames
Key
Frame
at t
3DGS at t/GCt
Multi-channel
2D Images
Gaussian Motion
Estimation
+
3DGS at t-1
dt+1
dt+2
dt+n
Gaussian Attributes Motions 
Adaptivly
Pruning
d't+1
d't+2
d't+n
Gaussian
Densification
Inflation
Loss
Pruned Attributes Motions 
Gaussian
Transform
Construct
Render
Frame t+n
Rendering
Training
Streaming
Quality-aware
frame grouping
Group 0 Group t
Group t+n
Keyframe Fast
Reconstruction
Intra-group
Frame Training
Input
2
3
4
5
1
Fig. 3.
System overview of AirGS. Each frame group is identified by its
keyframe index.
fluctuating bandwidth, AirGS applies a usage-frequency-based
adaptive pruning strategy to select an appropriate pruning level
so that the reconstruction quality can be optimized under the
network constraint.
B. Efficient 4DGS Training for High-Quality Representation
The quality-based frame grouping. As discussed in the
background, once the canonical space is established, the
total number of Gaussian primitives remains fixed in existing
3DGS-based solutions for dynamic scene rendering. Conse-
quently, frames containing regions with large gradients during
training cannot be fitted by generating new Gaussians, leading
to a significant quality decrease. Therefore, as illustrated
in Fig. 4, we first propose a quality-based frame grouping
strategy to iteratively partition entire video sequences into
finer-grained groups, each with its own canonical space, to
enable high-quality FVV construction over long sequence.
Unlike prior methods, AirGS first initializes the canonical
Gaussian space using the first frame, and subsequent frames
are reconstructed by transforming this canonical space. It then
employs a quality threshold τ, e.g., value adopted in prior
work [12] [13], to monitor the quality discrepancy between the
rendered frame and its ground truth frame during the initial
reconstruction. When the reconstruction quality falls below
this threshold, indicating that the fixed canonical Gaussians
can no longer adequately model the current frame, the frame
is designated as another new keyframe and used to initiate a
new canonical space for the following frames.
Frame training within a group. In 3DGS, obtaining the
initial point cloud with SfM method for Gaussian recon-
struction of each frame often incurs substantial computational

<!-- page 4 -->
Key-Frame:
t
3DGS
reconstruction:
GCt
MLPs
GSt+1
MLPs
 
 
GSt+2
MLPs
Frame:
t+n
GSt+n
MLPs
GSt+n-1
t=
 t + n
Frame:
t+1
Frame:
t+2
Frame:
t+n-1
 
 
 
 
 
 
Fig. 4. The quality-based frame grouping from group t to the following group
t + n based on the key-frame t and quality threshold τ.
overhead. Benefited from the temporal continuity brought by
our grouping strategy, AirGS instead choose to reconstruct
new frames within a group by fine-tuning the results of
the preceding frame, with reconstruction cost significantly
reduced. Specifically, for two consecutive frames at timestep
t and t+1 within the same group, the optimized reconstruc-
tion GSt is used to initialize GSt+1. Leveraging the shared
canonical space, AirGS predicts inter-frame variations using
six MLPs, one for each Gaussian attribute (µ, α, c, SH, R,
S), eliminating full retraining for each frame, ensuring spatial
consistency, and accelerating training. Given the GSt for frame
t, the motion dt+1 can be obtained by:
dt+1 = {MLP1(GSt), MLP2(GSt), ..., MLP6(GSt)} (4)
Then, as shown in Fig. 4, by applying the estimated motion
to the GSt, we can efficiently get the Gaussian reconstruction
for frame t + 1.
GSt+1 = GSt + dt+1
(5)
For the training of the MLP, AirGS further introduces a
temporal loss Ltemp as follows to enhance the temporal
coherence between adjacent frames.
Ltemp =
6
X
i=1
||MLPi(GSt)||1
(6)
The loss is designed to prevent disruptions in inter-frame
similarity that could lead to perceptual discontinuities. Then,
combined with the conventional 3DGS loss L3DGS in Equ.
(1), we can get the total loss function for the reconstruction
of frames within the same group:
L = L3DGS + λtLtemp,
(7)
where λt is the weight of our temporal loss.
Keyframe fast reconstruction. Keyframe candidates of-
ten contain newly emerged objects or large motions, which
indicates that the previous 3DGS canonical space can no
longer support high-quality reconstruction of the current frame
through transformations based on the fixed number of Gaus-
sian primitives alone. Consequently, a new 3DGS canonical
Gaussian
Densification
Inflation
Loss  
Fig. 5. Training process of keyframe t. The 3DGS for frame t −1 (GSt−1)
initializes the model. Gaussian densification adds more primitives in high-
gradient regions (e.g., red box) for better fitting. An iterative optimization
with inflation loss prunes redundancies, yielding the final reconstruction GSt,
which serves as the canonical space GCt.
space must be re-initialized for accurate reconstruction. How-
ever, in existing methods [7] [14], these spaces can only be
obtained through a full reconstruction pipeline, which incurs
substantial computational and time overhead.
Fortunately, we observe that newly emerged objects or large
motions typically occupy only a small portion of the entire
scene, around 20% based on our measurements among the
HiFi4G dataset [15], while the majority of the frame, such as
the background, remains highly consistent with the previous
frame. Therefore, only a few additional Gaussians are required
to capture regions inadequately represented by the previous
canonical space.
Therefore, based on this insight, we still use the trained
3DGS of the preceding frame as an initialization for generating
the keyframe’s 3DGS in the new group. To accurately capture
areas with new objects or large motions, which often exhibit
high gradients during training, we apply Gaussian densification
by dynamically cloning or splitting primitives. The increased
Gaussian aims to represent the scene more precisely, enabling
high-quality reconstruction.
However, indiscriminately increasing the number of Gaus-
sian primitives imposes significant computational and stor-
age overhead, particularly for streaming applications, where
transmission cost can rise substantially. To prevent excessive
growth of Gaussian primitives and maximize the rendering
efficiency of each one, AirGS introduces an inflation loss
during keyframe construction and discards Gaussians with low
opacity throughout the training process. The loss progressively
reduces the opacity of redundant Gaussian primitives, leading
to their eventual elimination as they become visually insignif-
icant. This process is also illustrated in Fig. 5. The inflation
loss Linf is defined as:
Linf = max(0, N −U),
(8)
where N is the number of current Gaussian primitives and
U denotes a predefined soft constraint on the maximum
number of Gaussian primitives. In our work, the value of U is
determined based on the number of Gaussian primitives in the
last canonical space, which is derived from the last keyframe.
This design minimizes the risk of exceeding the predefined
image capacity, which would otherwise require extra image
allocation and cause abrupt increases in data size.

<!-- page 5 -->
0
10
20
30
40
50
60
70
The Pruning Level (%)
30.0
31.0
32.0
33.0
34.0
35.0
36.0
PSNR
PSNR
SSIM
0.9850
0.9865
0.9880
0.9895
0.9910
0.9925
0.9940
SSIM
Fig. 6. The 3DGS rendering quality with different level of usage-frequency-
based Gaussian primitives pruning.
For the training of keyframes, it is also essential to maintain
temporal consistency; however, as the MLPs no longer partic-
ipate in the reconstruction of the keyframes, the temporal loss
of key frames Lk
temp is defined as
Lk
temp =
X
i
||Gi
t −Gi
t−1||1,
(9)
where i is bounded by the value of U. The inflation loss
only removes newly added redundant Gaussians. Since the
Gaussians from frame t−1 are already well-optimized and just
insufficient to capture the new content in frame t, they are
preserved during the keyframe reconstruction, while only re-
dundant additions are pruned. Therefore, for the first frame of
the video, its loss follows the conventional 3DGS loss L3DGS
in Equation (1), while for the other following keyframes
training, the total loss is defined to be:
Lk
total = L3DGS + λtLk
temp + λinfLinf,
(10)
where λinf is the weight of Linf, and k is the keyframe’s
index.
C. Adaptive Pruning for 4DGS Streaming
After finishing the training process in remote servers, the
data required for rendering, i.e., the attributes of each frame’s
3DGS, is transmitted to the client side. In AirGS, two types
of data are retained on the server for each video: the multi-
channel 2D images for keyframes, where each image stores
one Gaussian attribute and each pixel corresponds to one prim-
itive, enabling fast, low-overhead decoding by naturally lever-
aging hardware codecs for efficient streaming and decoding;
the difference tensor Dt = Pt
i=k di represents the variation
between frame t and its corresponding group’s canonical space
GCk. This design enables fast reconstruction of any frame
within a group, thereby supporting efficient frame skipping.
To further reduce bandwidth usage during sequential play-
back, we adopt a differential transmission strategy, wherein
only dt is transmitted for frame t if frame t −1 was the last
viewed frame. This approach leverages inter-frame similarity,
as dt tends to be sparse matrices, thus reducing both storage
and transmission cost. However, under fluctuating network
bandwidth, this strategy alone remains insufficient. As more
keyframes are accumulated, the number of Gaussian primitives
Algorithm 1 Pruning level selection for frame t
Input:
Dense
pruning
level
space
Sd
t ,
a
list
of
(quality, size) tuples, each corresponding to a pruning
level; the "cliff" multiplier, β; the available bandwidth,
B; the target frame rate, R
Output: The pruning level, p
1: previous ←Sd
t [1][0] −Sd
t [0][0]
2: Initialize an empty array S
3: S ←[0]
4: for i from 1 to len(Sd
t ) −1 do
5:
drop ←Sd
t [i][0] −Sd
t [i −1][0]
6:
if drop/previous > β then
7:
p ←i
8:
break
9:
end if
10:
S.append(i)
11:
previous ←drop
12: end for
13: C ←B/R
14: low ←0
15: high ←len(S) −1
16: while low ≤high do
17:
mid ←floor((low + high)/2)
18:
tmp ←S[mid]
19:
if Sd
t [tmp][1] ≤C then
20:
p ←tmp
21:
high ←mid −1
22:
else
23:
low ←mid + 1
24:
end if
25: end while
26: return p
in the canonical space continuously increases, leading to a pro-
gressively larger amount of data dt that must be transmitted.
To address this, AirGS further adopts an adaptive pruning
strategy based on the Gaussian primitive’s usage frequency
(p(·)) to compress the transmission data, which is d′
t = p(dt).
The goal of the adaptive pruning is to figure out a Gaussian
primitives pruning level that maintains streaming smoothness
while minimizing degradation in rendering quality under the
fluctuation network bandwidth. To prevent cumulative quality
degradation caused by pruning-induced variation loss, we
maintain a record D′
t = P
t d′
t on the server, which tracks the
total amount of data actually transmitted up to frame t from
the keyframe. Thus, the data to be sent for frame t within the
group k should be:
d′
t = p(Dt −D′
t−1)
(11)
and the reconstruction follows the following equation:
GSt = GCK +
t
X
i=0
d′
t = GCk +
t
X
i=0
p(Dt −D′
t−1)
(12)
Pruning level selection problem formulation. Let n be
the total number of frames, and S the set of available pruning

<!-- page 6 -->
levels. For a pruning level j ∈S, let stj and qtj denote
the pruned data size and reconstruction quality of frame t,
respectively. Given the bandwidth constraint Bt for frame t
and a target transmission rate R. The pruning level selections
for a group of frames can be formulated as follows:
maximize
n
X
t=1
X
j∈S
qtjxtj
subject to
X
j∈S
stjxtj ≤Bt/R
X
j∈S
xtj = 1, ∀t
xtj ∈{0, 1}, j ∈S
t ∈{0, 1, . . . , n}
(13)
where xtj indicates whether pruning level j is selected for
streaming the frame t. The first constraint limits the data size
to be transmitted for frame t under Bt/R to achieve the target
frame rate R. The second ensures only one pruning level per
frame is selected. The last two define the selection variable
and total frame count.
Algorithm design To efficiently determine an appropriate
pruning level, AirGS adopts a stepwise fallback strategy,
which selects the lowest pruning level whose post-pruning size
cost does not exceed the bandwidth requirement. The next step
is to construct the pruning level space S. As shown in the
Fig. 6, pruning generally correlates with quality degradation,
with a tipping point where further pruning leads to a sharp
quality drop. Therefore, for frame t, we first perform fine-
grained pruning based on the rendering usage frequency of
each Gaussian primitive. The pruning ratios are varied across
a wide range, thereby constructing a dense pruning level space
Sd
t . Due to the high rendering efficiency of 3DGS, we can
rapidly evaluate the reconstruction quality at each pruning
level. When the quality degradation between two adjacent
levels significantly exceeds β times that of previous steps,
the latter is identified as a quality cliff, and all preceding
levels are retained as the final pruning space S. Leveraging
the monotonic relationship between pruning levels and quality
degradation, we employ binary search to efficiently identify
the lowest pruning level that satisfies the bandwidth constraint,
thereby maximizing the quality of each frame. Thus, the
detailed algorithm is shown in the Algorithm 1.
Complexity Analysis This algorithm can be divided into
two parts, the first part (line 4 to line 11) scans the dense
pruning level space to construct a reduced candidate set S.
Pruning levels in Sd
t that meet the quality requirement are
inserted into S in increasing pruning order, ensuring a mono-
tonically decreasing size sequence. The time complexity for
this candidate set construction part is O(n). The second part
aims to identify a feasible pruning level that maximizes the
reconstruction quality under the constraints of Equ. 13. Thanks
to the monotonically decreasing size order of S maintained in
the first part, a binary search with a time complexity of O(log
n) can be efficiently applied to locate a valid pruning level.
IV. EVALUATION
A. Evaluation setup
Dataset. We evaluate AirGS on the HiFi4G dataset [15],
which includes seven dynamic sequences captured at 30 fps
from 81 viewpoints with a resolution of 3840*2160. For each
sequence, 10 viewpoints are held out for testing, while the
remaining views are used for training. To assess streaming
performance under practical network conditions, we adopt the
driving scenario in LTE traces [16], which signifies limited
bandwidth and significant network fluctuation.
Baselines. We compare AirGS with several state-of-the-
art methods for dynamic scene reconstruction and rendering.
For neural radiance field–based approaches, we select NeuS2
[14], which leverages multi-resolution encoding and progres-
sive training for efficient rendering but may produce blurred
details in fine-grained regions. Among 3DGS-based methods,
in addition to the original 3DGS [8], we also compare AirGS
with three other state-of-the-art benchmarks: 3DGStream [6],
which uses a neural transformation cache to capture inter-
frame changes for fast reconstruction; 4DGS [9], which fuses
Gaussians with 4D neural voxel representations for high-
quality rendering; and V 3 [7], which adopts canonical space
and improves streaming efficiency by projecting each frame’s
3DGS into 2D images
Metrics. We evaluate performance across multiple dimen-
sions that critically impact the user’s viewing experience.
• Rendering visual quality: We utilize two widely adopted
metrics to evaluate the rendering performance of AirGS.
PSNR: This metric quantifies the difference between two
images by calculating the mean squared error (MSE)
of their pixel values. This metric provides a numerical
assessment of the differences between the reconstructed
frame and the ground truth [17].
SSIM: This metric evaluates the similarity between two
images by jointly considering luminance, contrast, and
structural information. An SSIM score of 1 indicates that
the images are perfectly identical [18].
• Transmission data size (MB). This metric measures the
amount of data required to render a frame, which also
reflects the bandwidth consumption per reconstruction.
• Training time. This metric reflects the efficiency of train-
ing on dynamic scenes using different methods.
• Rendering smoothness. Frames per second (FPS) mea-
sures rendering fluency and reflects the computational
burden of each method, serving as a key indicator of
smooth user experience.
• Transmission time. This metric measures the time needed
to transmit frame data from the server to the client.
Higher transmission times can cause playback stalls or
interruptions, degrading the viewing experience.
Hyperparameter Settings. All training and evaluation are
conducted on a single NVIDIA GeForce RTX 3090 GPU. For
all baselines, the entire video is treated as a single group,
and keyframes are selected according to each method’s default
strategy. For AirGS, the quality threshold for frame grouping

<!-- page 7 -->
3DGS
4DGS
Ground Truth
NeuS2
3DGStream
 
 
a s
AirGS
PSNR:
38.95
PSNR:
28.09
PSNR:
33.02
PSNR:
33.31
PSNR:
37.13
PSNR:
33.04
a+1 s
PSNR:
38.93
PSNR:
27.06
PSNR:
31.56
PSNR:
33.76
PSNR:
29.87
PSNR:
36.11
Fig. 7. Rendering results comparison of the dynamic scene against recent SOTA methods, showing the reconstruction at time step a and a+1.
TABLE I
COMPARISON WITH BASELINES ON DYNAMIC SCENES RECONSTRUCTION. CELLS HIGHLIGHTED IN RED INDICATE THE
BEST
RESULTS, WHILE THOSE
IN YELLOW DENOTE THE
SECOND-BEST .
Method
PSNR ↑
SSIM ↑
Size (MB) ↓
Training Time (Min) ↓
Rendering Smoothness (FPS) ↑
NeuS2 [14]
29.82
0.961
24.5
3.7
3
Naive 3DGS [8]
33.94
0.987
18.3
9.8
200
4DGS [9]
29.95
0.971
12.0
9
60
3DGStream [6]
25.33
0.944
7.8
0.75
215
V 3 [7]
27.69
0.949
2.0
1.5
387
AirGS
32.29
0.987
1.1
1.4
400
0
5
10
15
20
25
Transmission Data (MB)
0
50
100
150
200
250
300
350
400
Rendering Smoothness (FPS)
3DGStream
V3
NeuS2
4DGS
AirGS
3DGS
PSNR - Method
25.33 - 3DGStream
27.69 - V3
29.82 - NeuS2
29.95 - 4DGS
32.29 - AirGS
33.94 - 3DGS
Fig. 8.
Compared with other baselines, AirGS achieves high-quality and
smoother rendering with a small data size required. The point size refers to
PSNR.
is set to 30, commonly considered indicative of high-quality
reconstruction [12] [13] [19]. Loss hyper-parameters are set
as follows: λ = 0.2, consistent with prior work [6] [8], λt =
1e −3, and λinf = 1e −5. The dense pruning level space Sd
t
ranges from 0% to 100% in 10% increments, β is set to 2.
B. Evaluation Results
Overall Performance. We first demonstrate visual results
rendered by different methods in Fig. 7. These results illus-
trate that AirGS effectively reconstructs fine-grained details
and avoids the blurring, detail loss, and artifacts common
when modeling large motions, while others show noticeable
degradation. We next present the overall comparison results in
quantifiable metrics in Table I and Fig. 8, where all reported
values represent the average performance. AirGS ranks second
only to naive 3DGS [8] in quality, with a PSNR gap of 1.65 dB
and comparable SSIM, and outperforms other baselines by up
to 5 dB PSNR and 0.043 SSIM. Moreover, AirGS significantly
outperforms all baselines, including the naive 3DGS, in other
two key metrics: transmission size and rendering smoothness.
Specifically, AirGS reduces the per-frame transmission data
from 18.5 MB in naive 3DGS [8] to just 1.1 MB. Compared
to the second-best method V 3 [7], AirGS archives over 40%
data size reduction from 1.9 MB to 1.1 MB. In terms of
rendering smoothness, AirGS delivers an average rendering
speed twice that of naive 3DGS [8]. These improvements
are enabled by the combination of inflation loss and usage-
frequency-based pruning, which effectively control the number
of Gaussians per frame and enhance rendering efficiency. In
terms of training time, AirGS takes approximately 1.4 minutes
per frame, slightly slower than 3DGStream [6], which requires
only 0.75 minutes per frame, and ranks second overall, yet it
outperforms 3DGStream [6] in all other metrics.
Table II further demonstrates the keyframe training time
comparison for the keyframe-based methods. Due to the
reliance on implicit neural rendering, NeuS2 [14] incurs much
higher training cost and requires 7.6 times the reconstruction

<!-- page 8 -->
0
10
20
30
40
50
Frame Index
10
15
20
25
30
35
PSNR
AirGS
3DGS
V3
NeuS2
4DGS
3DGStream
(a) Quality fluctuation in PSNR for
different methods.
V3
GSt. NeuS2 4DGS 3DGS AirGS
0
5
10
15
20
25
PSNR Deviation
(b) Average
quality
deviation
in
PSNR for different methods. The
term "GSt." refers to the 3DGStream.
Fig. 9. Comparison of quality fluctuation and average quality deviation.
TABLE II
COMPARISON OF AVERAGE KEYFRAME TRAINING TIME
Method
Keyframe training time(min) ↓
NeuS2 [14]
4.62
V 3 [7]
2.98
AirGS
0.61
time of AirGS, while V 3 [7] takes 4.9 times longer. This effi-
ciency gain primarily stems from AirGS’s leverage of strong
inter-frame similarity, which avoids redundant retraining on
static regions and substantially reduces overall computation.
However, in Table I, AirGS achieves only a modest reduction
in average per-frame training time compared with it. This is
due to AirGS selecting multiple keyframes to better capture
significant variations, whereas V 3 [7] uses a single keyframe.
Quality Consistency. Fig. 9(a) demonstrates the detailed
frame-wise PSNR trends across different methods, indicating
that AirGS maintains frame-level PSNR above 31, avoids
sudden quality drops and accurately captures newly appearing
objects and large motions, ensuring high-quality reconstruc-
tion. Fig. 9(b) shows the average maximum per-frame recon-
struction deviation across different video sequences. AirGS
achieves reconstruction quality comparable to per-frame 3DGS
reconstruction, with average PSNR fluctuations below 5 dB,
demonstrating that by identifying hard-to-fit frames and as-
signing them new canonical spaces, AirGS’s quality-based
keyframe strategy maintains consistently high-fidelity recon-
struction over long sequences and prevents the degradation.
Transmission Performance. Fig. 10 and Table III present
the data transmission performance of different methods in
realistic networks. The cumulative distribution in Fig. 10(a)
shows that, thanks to its adaptive pruning strategy, AirGS can
transmit the necessary data for around 95% of frames within
25 ms, even under poor bandwidth. In contrast, some methods
require up to 350 ms per frame. Fig. 10(b) further demonstrates
the maximum transmission time per frame for each method.
Compared to all other baselines, AirGS reduces the maximum
transmission time by up to 37 times, and even compared
to the streaming-optimized V 3 [7], it requires only half the
time. As shown in Table III, AirGS achieves an average per-
frame transmission time of just 12ms. Compared to the second
0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35
Transmission Time (s)
0.0
0.2
0.4
0.6
0.8
1.0
CDF
AirGS
4DGS
NeuS2
V3
3DGStream
3DGS
(a) CDF of frame transmission time
among different methods.
V3
GSt. NeuS2 4DGS 3DGS AirGS
0.0
0.2
0.4
0.6
0.8
1.0
Max. Transmission Time (s)
(b) Maximum transmission time per
frame among different methods. The
term "GSt." refers to the 3DGStream.
Fig. 10.
CDF and the maximum value of transmission time of different
methods.
TABLE III
COMPARISON OF AVERAGE TRANSMISSION PERFORMANCE
Method
Transmission Time (s) ↓
Transmission Rate ↑
NeuS2 [14]
0.139
7.18
Naive 3DGS [8]
0.122
8.18
4DGS [9]
0.131
8.24
3DGStream [6]
0.063
15.90
V 3 [7]
0.021
48.38
AirGS
0.011
92.74
fastest method, V 3 [7], which requires 23 ms per frame, AirGS
delivers nearly a twofold speedup. Relative to the transmission
time of naive 3DGS [8] and the implicit rendering approach
NeuS2 [14], AirGS requires only 8.8% and 7.7% of their
time, respectively. The second column of Table III reports the
average transmission rates (frames/min). AirGS is the only
method exceeding 90, substantially higher than the next-best
rate of just over 40. Maintaining this high transmission rate is
essential for a seamless user viewing experience.
Impact of Pruning. In Fig. 11, we demonstrate the impact
of pruning on rendering quality. Fig. 11(a) and 11(c) show
the rendering results of AirGS without pruning, representing
the viewing experience under favorable network conditions. In
contrast, Fig. 11(b) and 11(d) depict the visual quality after
applying maximum pruning, which serves as the lower bound
within the AirGS framework under poor network conditions.
Fig. 11(e) further illustrates the impact: usage-frequency-based
pruning results in an average quality drop of only 2.3, while
reducing per-frame transmission size from 1.9 MB to 1.1 MB,
nearly a 50% reduction.
In Fig. 12, we further illustrate the impact of our pruning
strategy on transmission time performance. For a 175-frame
dynamic scene, pruning reduces the average per-frame trans-
mission time to approximately 0.012s, effectively mitigating
large fluctuations caused by network variability. Even in the
worst case, transmission remains twice as fast compared to the
unpruned setting. These findings demonstrate that our usage-
frequency-based pruning strategy effectively preserves high
rendering quality, smooth and stable transmission even under
adverse network conditions.
Overhead Analysis. We next evaluate the overhead of
AirGS when rendering a frame. The process consists of three

<!-- page 9 -->
(a) PSNR = 35.67
(b) PSNR = 34.31
(c) PSNR = 37.73
(d) PSNR = 35.51
Without Pruning
Maximum Pruning
30
32
34
36
38
PSNR
0.0
0.5
1.0
1.5
2.0
Transmission Size (MB)
(e) Average quality and transmis-
sion data size comparison
Fig. 11. Impact of pruning on reconstruction quality. The subfigures on the left (a and c) show the rendering results of AirGS without pruning, while those
on the right (b and d) illustrate the results after maximum pruning.
0
25
50
75
100
125
150
175
Frame Index
0.00
0.01
0.02
0.03
0.04
0.05
Transmission Time(s)
AirGS
AirGS (w/o Pruning)
Fig. 12.
Impact of pruning on transmission time per frame under the LTE
network conditions for 175 frames sequences.
phases. The first is streaming, where the data required to
render the target frame is transferred. Our measurements show
this phase takes approximately 11 ms per frame, accounting
for 58.2% of total processing time. The second phase is
decoding, which processes the data into Gaussian differences
between the current frame and the target frame, requiring about
25% of total time. Finally, the differences are applied to the
Gaussian primitives to finish the rendering of the target frame.
Overall, AirGS achieves an average end-to-end rendering time
of around 18.9 ms per frame, supporting reconstruction at over
50 FPS and delivering a smooth, seamless viewing experience.
V. RELATED WORK
Novel view synthesis. Novel view synthesis aims to gener-
ate unseen views from multi-view images. Early approaches
based on explicit geometric proxies [20]–[23] suffered from
limited fidelity and high storage costs. NeRF [24] [25]
achieved photorealistic results through implicit rendering but
required expensive per-pixel inference, prompting acceleration
methods that cache intermediate features in voxels or meshes
[26]–[29]. Inspired by the hybrid rendering, 3D Gaussian
Splatting [8] employs learned Gaussian primitives for high-
quality rendering with low computational overhead.
Free-Viewpoint Videos of Dynamic Scenes. Novel view
synthesis for dynamic scenes poses additional challenges due
to temporal changes. Typical methods rely on time-varying
primitives [1] [4] [30] or interpolation [3]. NeRF-based ap-
proaches model dynamics via canonical templates with time-
conditioned deformation [31]–[34], scene decomposition [35],
or flow estimation [36] [37]. However, these methods require
long training times and suffer from slow rendering. When
applied to long video sequences, they often produce artifacts
and blur. In recent years, 3DGS-based reconstruction emerges
as a promising technique for fast and high-quality construction.
In order to generate real-time FVV, 4DGS [9] integrates
3D Gaussian primitives with 4D neural voxel representations
to achieve high-resolution, real-time synthesis of dynamic
scenes. 3DGStream [6] uses keyframes, an MLP-based neural
transformation cache, and incremental Gaussian splats to warp
and update a base scene to the target scene of the specific
timestep. V 3 [7] encodes per-frame Gaussian attributes into 2D
images for full Gaussian primitives streaming, predicts motion
deltas with residual entropy and temporal consistency losses
based on fixed default frames.
However, these methods struggle to handle scene changes
over long sequences because they cannot reliably identify
frames with significant variations, causing the default canoni-
cal space to become invalid. They are also impractical for real-
world network environments, as each frame requires transmit-
ting all Gaussian primitives. In contrast, AirGS systematically
explores training and streaming stages for 4DGS-based FVV,
and proposes a multi-keyframe based training framework with
a communication-efficient streaming framework.
VI. CONCLUSION
In this paper, we present AirGS, a streaming-optimized
4DGS framework that enables high-quality and efficient FVV
experience. AirGS integrates a multi-keyframe training frame-
work to preserve reconstruction fidelity in dynamic scenes
and an adaptive streaming framework to ensure smooth view-
ing through dynamic pruning. For training, AirGS employs
a quality-driven keyframe selection strategy to initiate new
canonical spaces, and leverages inter-frame similarity along-
side a set of designed loss functions to facilitate model
training. For streaming, AirGS converts 3DGS data into 2D
image representations and adaptively prunes Gaussians to
balance reconstruction quality with bandwidth usage under
varying network conditions, enabling smooth and responsive
rendering. Comprehensive evaluations demonstrate that AirGS
consistently outperforms SOTA methods across key metrics,
including reconstruction quality, visual consistency, rendering
smoothness, training efficiency, and transmission cost.

<!-- page 10 -->
REFERENCES
[1] M. Dou, P. Davidson, S. R. Fanello, S. Khamis, A. Kowdle, C. Rhe-
mann, V. Tankovich, and S. Izadi, “Motion2fusion: Real-time volumetric
performance capture,” ACM Transactions on Graphics (ToG), vol. 36,
no. 6, pp. 1–16, 2017.
[2] R. A. Newcombe, D. Fox, and S. M. Seitz, “Dynamicfusion: Recon-
struction and tracking of non-rigid scenes in real-time,” in Proceedings
of the IEEE conference on computer vision and pattern recognition,
2015, pp. 343–352.
[3] M. Broxton, J. Flynn, R. Overbeck, D. Erickson, P. Hedman, M. Duvall,
J. Dourgarian, J. Busch, M. Whalen, and P. Debevec, “Immersive light
field video with a layered mesh representation,” ACM Transactions on
Graphics (TOG), vol. 39, no. 4, pp. 86–1, 2020.
[4] C. L. Zitnick, S. B. Kang, M. Uyttendaele, S. Winder, and R. Szeliski,
“High-quality video view interpolation using a layered representation,”
ACM transactions on graphics (TOG), vol. 23, no. 3, pp. 600–608, 2004.
[5] Z. Li, S. Niklaus, N. Snavely, and O. Wang, “Neural scene flow fields
for space-time view synthesis of dynamic scenes,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2021, pp. 6498–6508.
[6] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-
the-fly training of 3d gaussians for efficient streaming of photo-realistic
free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 20 675–20 685.
[7] P. Wang, Z. Zhang, L. Wang, K. Yao, S. Xie, J. Yu, M. Wu, and L. Xu,
“Vˆ 3: Viewing volumetric videos on mobiles via streamable 2d dynamic
gaussians,” ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp.
1–13, 2024.
[8] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[9] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and
X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2024, pp. 20 310–20 320.
[10] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-time photorealistic
dynamic scene representation and rendering with 4d gaussian splatting,”
arXiv preprint arXiv:2310.10642, 2023.
[11] N. Snavely, S. M. Seitz, and R. Szeliski, “Photo tourism: exploring photo
collections in 3d,” in ACM siggraph 2006 papers, 2006, pp. 835–846.
[12] Y. Lin, Z. Dai, S. Zhu, and Y. Yao, “Gaussian-flow: 4d reconstruction
with dynamic 3d gaussian particle,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 136–21 145.
[13] A. K. Jain, Fundamentals of digital image processing.
Prentice-Hall,
Inc., 1989.
[14] Y. Wang, Q. Han, M. Habermann, K. Daniilidis, C. Theobalt, and
L. Liu, “Neus2: Fast learning of neural implicit surfaces for multi-
view reconstruction,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2023, pp. 3295–3306.
[15] Y. Jiang, Z. Shen, P. Wang, Z. Su, Y. Hong, Y. Zhang, J. Yu, and
L. Xu, “Hifi4g: High-fidelity human performance rendering via compact
gaussian splatting,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2024, pp. 19 734–19 745.
[16] D. Raca, J. J. Quinlan, A. H. Zahran, and C. J. Sreenan, “Beyond
throughput: A 4g lte dataset with channel and context metrics,” in
Proceedings of the 9th ACM multimedia systems conference, 2018, pp.
460–465.
[17] A. Hore and D. Ziou, “Image quality metrics: Psnr vs. ssim,” in 2010
20th international conference on pattern recognition.
IEEE, 2010, pp.
2366–2369.
[18] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,” IEEE
transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[20] X. Ma, V. Hegde, and L. Yolyan, 3D Deep Learning with Python: Design
and develop your computer vision model with 3D data using PyTorch3D
and more.
Packt Publishing Ltd, 2022.
[21] A. Dalal, D. Hagen, K. G. Robbersmyr, and K. M. Knausgård, “Gaussian
splatting: 3d reconstruction and novel view synthesis, a review,” IEEE
Access, 2024.
[22] E. Penner and L. Zhang, “Soft 3d reconstruction for view synthesis,”
ACM Transactions on Graphics (TOG), vol. 36, no. 6, pp. 1–11, 2017.
[23] Y. Duan, F. Wei, Q. Dai, Y. He, W. Chen, and B. Chen, “4d-rotor
gaussian splatting: towards efficient novel view synthesis for dynamic
scenes,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.
[24] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” in ECCV, 2020.
[25] Z. Wang and Y. Zhu, “Nerflex: Resource-aware real-time high-quality
rendering of complex scenes on mobile devices,” in Proc. IEEE ICDCS,
2025.
[26] P. Hedman, P. P. Srinivasan, B. Mildenhall, J. T. Barron, and P. De-
bevec, “Baking neural radiance fields for real-time view synthesis,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 5875–5884.
[27] S. J. Garbin, M. Kowalski, M. Johnson, J. Shotton, and J. Valentin,
“Fastnerf: High-fidelity neural rendering at 200fps,” in Proceedings of
the IEEE/CVF international conference on computer vision, 2021, pp.
14 346–14 355.
[28] Z. Chen, T. Funkhouser, P. Hedman, and A. Tagliasacchi, “Mobilenerf:
Exploiting the polygon rasterization pipeline for efficient neural field
rendering on mobile architectures,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2023, pp.
16 569–16 578.
[29] L. Wang, J. Zhang, X. Liu, F. Zhao, Y. Zhang, Y. Zhang, M. Wu, J. Yu,
and L. Xu, “Fourier plenoctrees for dynamic radiance field rendering in
real-time,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2022, pp. 13 524–13 534.
[30] A. Collet, M. Chuang, P. Sweeney, D. Gillett, D. Evseev, D. Calabrese,
H. Hoppe, A. Kirk, and S. Sullivan, “High-quality streamable free-
viewpoint video,” ACM Transactions on Graphics (ToG), vol. 34, no. 4,
pp. 1–13, 2015.
[31] E. Tretschk, A. Tewari, V. Golyanik, M. Zollhöfer, C. Lassner, and
C. Theobalt, “Non-rigid neural radiance fields: Reconstruction and
novel view synthesis of a dynamic scene from monocular video,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 12 959–12 970.
[32] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. Nießner, and
Q. Tian, “Fast dynamic radiance fields with time-aware neural voxels,”
in SIGGRAPH Asia 2022 Conference Papers, 2022, pp. 1–9.
[33] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “D-
nerf: Neural radiance fields for dynamic scenes,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2021,
pp. 10 318–10 327.
[34] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz,
and R. Martin-Brualla, “Nerfies: Deformable neural radiance fields,”
in Proceedings of the IEEE/CVF international conference on computer
vision, 2021, pp. 5865–5874.
[35] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y. Xu, and
A. Geiger, “Nerfplayer: A streamable dynamic scene representation with
decomposed neural radiance fields,” IEEE Transactions on Visualization
and Computer Graphics, vol. 29, no. 5, pp. 2732–2742, 2023.
[36] X. Guo, J. Sun, Y. Dai, G. Chen, X. Ye, X. Tan, E. Ding, Y. Zhang, and
J. Wang, “Forward flow for novel view synthesis of dynamic scenes,”
in Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 16 022–16 033.
[37] F. Tian, S. Du, and Y. Duan, “Mononerf: Learning a generalizable
dynamic radiance field from monocular videos,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2023, pp.
17 903–17 913.
