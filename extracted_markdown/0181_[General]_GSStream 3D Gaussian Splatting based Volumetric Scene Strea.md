<!-- page 1 -->
1
GSStream: 3D Gaussian Splatting based Volumetric
Scene Streaming System
Zhiye Tang, Qiudan Zhang, Member, IEEE, Lei Zhang, Member, IEEE, Junhui Hou, Senior Member, IEEE, You
Yang, Senior Member, IEEE, Xu Wang, Member, IEEE
Abstract—Recently, the 3D Gaussian splatting (3DGS) tech-
nique for real-time radiance field rendering has revolutionized the
field of volumetric scene representation, providing users with an
immersive experience. But in return, it also poses a large amount
of data volume, which is extremely bandwidth-intensive. Cutting-
edge researchers have tried to introduce different approaches
and construct multiple variants for 3DGS to obtain a more
compact scene representation, but it is still challenging for
real-time distribution. In this paper, we propose GSStream,
a novel volumetric scene streaming system to support 3DGS
data format. Specifically, GSStream integrates a collaborative
viewport prediction module to better predict users’ future be-
haviors by learning collaborative priors and historical priors
from multiple users and users’ viewport sequences and a deep
reinforcement learning (DRL)-based bitrate adaptation module
to tackle the state and action space variability challenge of
the bitrate adaptation problem, achieving efficient volumetric
scene delivery. Besides, we first build a user viewport trajectory
dataset for volumetric scenes to support the training and stream-
ing simulation. Extensive experiments prove that our proposed
GSStream system outperforms existing representative volumetric
scene streaming systems in visual quality and network usage.
Demo video: https://youtu.be/3WEe8PN8yvA.
Index Terms—Volumetric Scene Streaming, 3D Gaussian Splat-
ting, Deep Reinforcement Learning
I. INTRODUCTION
Lately, 3DGS has significantly transformed the landscape
of volumetric scene representation, enabling the achievement
of exceptionally high-fidelity real-time rendering [1]. 3DGS
explicitly models a volumetric scene through the aggregation
of numerous 3D Gaussians, offering enhanced flexibility and
detail compared to traditional representations. However, this
substantial advantage in rendering quality is accompanied by
a notable increase in data volume, which poses challenges
for practical applications, particularly in scenarios requiring
efficient storage and transmission [2]. To mitigate this issue,
This work was supported in part by the National Natural Science Foundation
of China under Grant 62371310, in part by the Shenzhen Science and Tech-
nology Program (JCYJ20241202124415021), in part by the Guangdong Basic
and Applied Basic Research Foundation under Grant 2023A1515011236.
(Corresponding author: Xu Wang.)
Zhiye Tang, Qiudan Zhang, Lei Zhang and Xu Wang are with the
College of Computer Science and Software Engineering, Shenzhen Univer-
sity, Shenzhen, 518060, China. Email: (tangzhiye2022@foxmail.com, qiu-
danzhang@szu.edu.cn, leizhang@szu.edu.cn, wangxu@szu.edu.cn).
Junhui
Hou is
with Department
of Computer
Science, City Uni-
versity
of
Hong
Kong,
Kowloon,
Hong
Kong
SAR,
China.
Email:
(jh.hou@cityu.edu.hk).
You Yang is with School of Electronic Information and Communications,
Huazhong University of Science and Technology, Wuhan 430074, China.
Email: (yangyou@hust.edu.cn).
extensive research efforts have been directed towards opti-
mizing various aspects of the 3DGS framework. In a recent
advancement, Lu et al. introduced Scaffold-GS [3], which
innovatively employs learnable anchors instead of directly
training on individual 3D Gaussians. In this approach, each
anchor is initially derived from a structure-from-motion (SfM)
point cloud and aligned to a voxel grid, establishing a struc-
tured foundation for the scene representation. During both
the training and rendering phases, these anchors dynamically
generate multiple surrounding neural 3D Gaussians on the
fly through the use of multi-layer perceptrons (MLPs). This
process results in significant memory savings, as it reduces
the need to store vast numbers of individual Gaussians. For
example, representing an indoor scene measuring 3 meters by
8 meters in high quality using Scaffold-GS requires only a
few hundred megabytes of storage, which is a considerable
improvement over previous methods. Despite these advance-
ments, the storage requirements still present a substantial ob-
stacle for streaming such high-quality volumetric scenes under
the constraints of current network infrastructure, necessitating
further optimization and innovation in streaming technologies.
From the perspective of data formatting, 3DGS-derived
representations (e.g., Scaffold-GS) can be viewed as an evolu-
tion of traditional point clouds. A conventional point cloud
typically represents a 3D scene as a collection of discrete
points, where each point encompasses attributes such as 3D
spatial coordinates (x, y, z), color information, and normal
vectors, among others [6]. Similarly, Scaffold-GS organizes
its data by storing anchors as clusters of vectors, with each
vector encapsulating both feature and geometric attributes,
including center coordinates and scaling coefficients. This sim-
ilarity in storage structure implies that established point cloud
scene streaming methodologies are inherently compatible with
3DGS scenes, allowing for the adaptation of existing streaming
techniques to the 3DGS framework. Nonetheless, the challenge
of efficiently streaming point cloud content has been a focal
point of extensive research, exploring multiple dimensions and
strategies. The prevailing approach in this domain involves
the simultaneous application of viewport prediction and tile-
based bitrate adaptive streaming to minimize the volume of
data transmitted [7]. In practice, this involves partitioning a
volumetric scene into a grid of non-overlapping tiles, each
of which is individually encoded or downsampled into mul-
tiple representations corresponding to different quality levels.
Rather than transmitting all tiles at the highest quality, a bitrate
adaptation strategy is employed to selectively determine the
quality level of each tile based on its relevance to the user’s
arXiv:2603.09718v1  [cs.CV]  10 Mar 2026

<!-- page 2 -->
2
(a) Original
SSIM: 0.183
(b) CaV3 [4]
SSIM: 0.210
(c) GS3D [5]
SSIM: 0.676
(d) Ours
Fig. 1: Visualization effect of different volumetric scene
streaming systems. Our proposed GSStream achieves the best
visual performance among the state-of-the-art (SOTA) systems
with the help of the collaborative viewport prediction module
and the DRL-based bitrate adaptation module. Note that these
examples are rendered on the stump scene under a network
constraint of 120Mbps, at the 10th second (300th frame) of
the viewing process.
current and predicted future viewports. For instance, Zhang et
al. [8] proposed Trans-RL and adopted a sequence prediction
paradigm where a single Transformer model is utilized to
predict the user’s viewport trajectories. Subsequently, DRL
is introduced to address the bitrate adaptation challenge,
determining the optimal quality level for each tile to balance
visual fidelity and bandwidth usage. However, a significant
limitation of this approach is the assumption that all volumetric
scenes are divided into an identical number of tiles, which
is rarely the case in practical applications. Consequently,
existing methods encounter three primary challenges: (1) As
a novel form of volumetric scene representation, there is
a dearth of comprehensive user viewport trajectory datasets
specifically tailored for 3DGS scenes, hindering the devel-
opment of effective viewport prediction models; (2) Current
viewport prediction techniques often treat viewport trajectories
across different users in a uniform manner, failing to account
for individual differences in viewing behavior and personal
preferences, which can lead to suboptimal prediction accuracy;
and (3) The full potential of DRL algorithms in addressing the
bitrate adaptation problem has yet to be realized, as existing
approaches have not fully exploited the capabilities of DRL to
dynamically and intelligently manage bitrate allocation based
on varying network conditions and user behaviors.
In this paper, we introduce GSStream, a comprehensive 3D
Gaussian Splatting-based volumetric scene Streaming system
designed to tackle the aforementioned challenges head-on. The
GSStream system harnesses the capabilities of Scaffold-GS [3]
to represent volumetric scenes, thereby enhancing the visual
quality experienced by users through head-mounted devices
(HMDs). Recognizing that users exhibit diverse behavioral
inertia when engaging with splatted Gaussians, GSStream
incorporates a collaborative viewport prediction algorithm.
This algorithm not only predicts user viewports but also
extracts and utilizes user-specific embeddings derived from the
collected dataset in a collaborative manner, thereby leveraging
individual viewing patterns to improve prediction accuracy and
relevance. Additionally, GSStream addresses the variability
in the number of tiles across different volumetric scenes
by treating the tiles decomposed from each scene as an
unordered set. To effectively manage bitrate adaptation under
these conditions, GSStream models the problem as a Markov
decision process (MDP) [9] and employs a deep deterministic
policy gradient (DDPG)-based [10], [11] algorithm to derive
optimal bitrate allocation policies. For validation, we have
constructed a realistic user viewport trajectory dataset tailored
for 3DGS content, encompassing 15 diverse volumetric scenes
that include both indoor and outdoor scenarios. We devel-
oped a specialized user viewport data collection platform to
gather behavioral data from 32 subjects representing various
industries, ensuring a broad and representative dataset. Our
experimental results demonstrate that the proposed GSStream
system outperforms existing SOTA volumetric scene streaming
systems in terms of both visual quality and network efficiency,
underscoring the effectiveness of our integrated approach
in enhancing the user experience while optimizing resource
utilization.
The contributions of this work can be summarized as
follows:
• We present the first 3DGS-based volumetric scene
streaming system, GSStream, which seamlessly inte-
grates a Collaborative Viewport Prediction (CVP) mod-
ule with a DRL-based Bitrate Adaptation (DBA) module.
This integration allows for a more accurate estimation of
users’ future viewing behaviors and effectively addresses
the challenges posed by variability in state and action
spaces inherent in the bitrate adaptation problem.
• We develop the first comprehensive user viewport trajec-
tory dataset specifically for 3DGS content, encompassing
15 diverse volumetric scenes that include a range of in-
door and outdoor scenarios. This dataset was constructed
by designing and implementing a user viewport data
collection platform, which was utilized to capture detailed
behavioral data from 32 participants across different
industries, thereby ensuring the dataset’s diversity and
applicability.
• Through rigorous experimental evaluations, we demon-
strate that our proposed GSStream system significantly
enhances visual quality and optimizes network usage
compared to existing SOTA volumetric scene streaming
systems. These results validate the efficacy of our system
in delivering superior performance in real-world stream-
ing scenarios.

<!-- page 3 -->
3
II. RELATED WORKS
A. 3D Scene Representations and Rendering
There have been various explorations on representing vol-
umetric scenes, e.g., mesh, point cloud. Point cloud format
is the most popular among them since it is flexible, similar,
and easy for reconstruction [12]. Its adaptability to capture fine
geometric details, as well as its straightforward generation pro-
cess from common 3D scanning or multi-view reconstruction
pipelines, makes the point-based format a fundamental choice
for many computer vision tasks. Therefore, the topic of point-
based rendering has long been discussed since 1985 [13], in
which points are used as primitives for detailed rendering.
Modern methods usually replace points with surfels estimated
based on point cloud data [14] and use splatting operations
to rasterize and combine multiple overlapping surfels either
linearly or using neural networks [15], [16] to obtain the final
rendering result. Such approaches efficiently handle geometry
and appearance while maintaining a level of simplicity that
benefits many real-time applications.
With the emergence of differentiable rendering techniques,
neural radiance field (NeRF) [17] and 3DGS [1] have both
attracted increasing interest. NeRFs are neural network models
trained to predict novel views in 3D scenes according to certain
given views, effectively learning a volumetric representation
that captures scene geometry and appearance. Each pre-trained
model can be seen as an implicit representation of one certain
scene, in which color and density are output by a network
given a spatial coordinate and viewing direction. Recently,
3DGS has been notable for its pure explicit representation
and real-time rendering. It proposed using 3D Gaussians as
rendering primitives, maintains the storage structure as a point
cloud, and has brought new possibilities to the field of volu-
metric scene representation. This explicit formulation enables
direct manipulation of the scene’s geometry without relying on
a learned neural model, offering users more interpretability
and flexibility when editing or processing 3D data while
facilitating real-time rendering.
Starting from a SfM point cloud, the position µ of each
point i in a 3DGS is treated as the position of a 3D Gaussian
Gi:
Gi(x) = e−1
2 (x−µi)T Σ−1
i
(x−µi),
(1)
where x is a random position within a 3D space (i.e., domain
of definition), while Σi denotes the covariance matrix describ-
ing the pose and scale of the 3D Gaussian, formulated by a
scaling matrix Si and a rotation matrix Ri as
Σi = RiSiST
i RT
i ,
(2)
ensuring Σi is a positive semi-definite matrix. During train-
ing supervised by image datasets, 3DGS learns an opacity
value αi and a group of Spherical Harmonics coefficients
for each 3D Gaussian Gi to model its color. The above-
mentioned variables will then be rasterized onto a 2D plane
by employing α-blending to ensure high-fidelity rendering.
Following the framework proposed by [1], Scaffold-GS [3]
introduced a more storage-efficient anchor-based framework
by replacing 3D Gaussians as anchors. Specifically, each
anchor i consists of a location xi
∈R3 and attributes
Ai =
n
fi ∈RDf , si ∈RDs, oi ∈RDoo
, with each represent-
ing anchor feature, scaling and offsets respectively. In practical
rendering, fi is inputted into MLPg to generate a group of
neural Gaussians surrounding anchor i, whose shapes and
locations are determined by anchor location xi, scaling si
and offsets oi. Such a design demonstrates how volumetric
primitives can be learned and manipulated in a way that is
both memory-efficient and highly adaptable to diverse scene
structures.
Compared with the raw point cloud format, the 3DGS for-
mat is significantly different. With extra attributes, 3DGS can
provide rendered images with substantially higher fidelity and
is streaming-friendly, with far-reaching application prospects.
By explicitly encoding information within 3D Gaussians, the
pipeline enables a wider range of scene manipulations, such
as more precise control over geometry and appearance, as
well as the potential for advanced lighting or material effects.
However, such an advantage comes at the expense of a
larger data volume. In traditional point clouds, each point is
typically described by spatial coordinates, color, and normal
vector, occupying 27 bytes of memory [6]. 3DGS represents
a volumetric scene as a set of 3D Gaussians, with attribute
information such as center coordinates and covariance matrix,
thus consuming 248 bytes per point. For a typical indoor
scene with millions of 3D Gaussians, the data volume can
be up to 800MB, which is challenging for efficient delivery
under current network conditions. Such memory demands can
be particularly prohibitive in scenarios where fast, real-time
updates or remote streaming of high-fidelity 3D content is
necessary.
To this end, some researchers have also focused on devel-
oping compact representations for 3DGS. In [18], Lee et al.
designed a learnable mask strategy to reduce the amount of 3D
points in a 3DGS scene, greatly restricting the data volume.
Morgenstern et al. [19] proposed a highly parallel algorithm to
arrange Gaussian parameters into 2D grids for a more compact
representation, thereby enabling more efficient memory usage
and potentially faster data transfer. Sun et al. [20], from
another perspective, proposed 3DGStream, utilizing a neural
transformation cache to model the dynamic information of ob-
jects in 3DGS scenes, which is a streaming-friendly dynamic
3DGS representation. This method reduces the bandwidth
required to continuously update dynamic scenes, offering a
pathway to practical applications such as virtual reality or
telepresence systems. This paper mainly focuses on stream-
ing optimization using the anchor-based Scaffold-GS format,
aiming to balance storage efficiency and rendering quality for
real-time and remote 3D experiences.
B. Volumetric Scene Streaming
To efficiently transmit volumetric scenes, a common ap-
proach is to segment the scene content into multiple tiles, as a
viewer typically only focuses on the region of the point cloud
scene that lies within the field of view (FoV). By dividing
the scene into these manageable chunks, it becomes more
flexible to address diverse network conditions and personalized
user requirements. The decomposed tiles can be independently

<!-- page 4 -->
4
① Pre-Processing
Binary Representations
Volumetric Scene
tiling
down-
sampling
④ DRL-based Bitrate Adaptation (DBA)
② Volumetric Scene Playback
rendering
display
③ Collaborative Viewport Prediction (CVP)
Critic
Head
Actor
Head
Shared
Feature 
Extraction
Module
Critic
Head
Actor
Head
Shared
Feature 
Extraction
Module
selecting
Quality Level 𝐿 
Tile Number 𝐾 
Binary Representations
Received 3DGS Data
Tile Number 𝐾 
Quality Level 𝐿 
𝒙 
𝒚 
𝒛 
𝒙 
𝒚 
𝒛 
Historical Viewport
Future Viewport
Collaborative Priors 
Extraction Module
Historical Priors 
Extraction Module
Collaborative Priors 
Extraction Module
Historical Priors 
Extraction Module
Server Side
Client Side
① Pre-Processing
Binary Representations
Volumetric Scene
tiling
down-
sampling
④ DRL-based Bitrate Adaptation (DBA)
② Volumetric Scene Playback
rendering
display
③ Collaborative Viewport Prediction (CVP)
Critic
Head
Actor
Head
Shared
Feature 
Extraction
Module
selecting
Quality Level 𝐿 
Tile Number 𝐾 
Binary Representations
Received 3DGS Data
Tile Number 𝐾 
Quality Level 𝐿 
𝒙 
𝒚 
𝒛 
𝒙 
𝒚 
𝒛 
Historical Viewport
Future Viewport
Collaborative Priors 
Extraction Module
Historical Priors 
Extraction Module
Server Side
Client Side
Fig. 2: Overview of the proposed GSStream system. ①In the pre-processing phase, the volumetric scene is tiled into K tiles,
and each is downsampled into representations in L quality levels. ②Representations are selectively transmitted to the client
side to reconstruct a 3DGS scene for display. ③User’s viewports captured by HMD are utilized to predict future viewports.
④The predicted future viewports are then utilized to generate the representation selection strategy for the next time slot.
encoded or downsampled and then selectively transmitted
according to various conditions, such as the user’s dynamically
changing viewports, the inherent characteristics of each tile
(e.g., size, volume or number of Gaussians), and etc. to
optimize the user’s quality of experience (QoE) under stringent
bandwidth constraints. By adopting this selective transmission
approach, streaming systems can effectively avoid transmit-
ting unnecessary data outside a viewer’s FoV, thus reducing
bandwidth consumption and promoting more efficient resource
utilization.
Numerous attempts have been made to stream volumetric
data. These efforts can be broadly classified into two cate-
gories. The first category is dedicated to building more effi-
cient volumetric content delivery strategies based on simulated
viewports, which help researchers anticipate user behavior
under various usage scenarios or network conditions. For
instance, Van der Hooft et al. [21] modified dynamic adaptive
streaming over HTTP (DASH) [22] for volumetric video
streaming and incorporated bitrate adaptation for different
objects, effectively separating foreground and background con-
tent. In [23], [24], researchers created tile-wise metrics based
on quality levels and user viewports for each representation,
then applied greedy algorithms to select the most suitable
representation that balanced bandwidth demands and visual
quality. Building on this idea, Li et al. treated point cloud
streaming as a time-bound optimization problem by using
branch-and-bound techniques in [25], and later in [26] they
developed another hybrid visual saliency streaming method
to further improve the perceptual quality. More recently, [27],
[28] proposed DRL-empowered approaches for advanced point
cloud content delivery.
Another development direction is the design of comprehen-
sive streaming systems that integrate viewport prediction and
bitrate adaptation into one framework. For example, ViVo [7]
employed linear regression for viewport prediction and intro-
duced a streaming approach that considers tile distance, occlu-
sion, and visibility, thereby reducing the wasteful transmission
of low-priority regions. More recently, CaV3 [4] leveraged
Transformers [29] to fuse priors learned from historical user
behavior patterns and volumetric scene representations, aiming
to better predict users’ future viewports. In tandem with this
prediction method, they also introduced the approach from
Yusuf et al. [30], which formulated the volumetric scene
streaming problem as a cache-assisted multi-armed bandit
(MAB) [31] problem and tackled it with a reinforcement
learning algorithm. In [8], Zhang et al. proposed Trans-
RL, which forecasted user viewports via a Transformer and
formulated the bitrate adaptation problem as an MDP. Despite
these advances, certain assumptions, such as each scene having
the same number of tiles, simplify the problem space and leave
unresolved issues regarding variable state and action spaces.
In conclusion, among the mainstream volumetric scene
streaming methods, the user viewport information and the po-
tential of DRL are not fully leveraged. Existing schemes either
rely on simplistic viewport assumptions or use constrained
strategies that do not adequately account for the dynamics
of user interaction in immersive environments. To this end,
we propose GSStream, a novel 3DGS-based volumetric scene
streaming system that can efficiently transmit volumetric me-
dia content while delivering superior performance and better
QoE compared to other state-of-the-art systems.
III. SYSTEM DESIGN
As shown in Figure 2, our proposed GSStream system
can be decomposed into four decoupled parts, each of which
addresses a distinct stage in the volumetric streaming pipeline
while working cooperatively to ensure high performance and
user immersion. During pre-processing, the volumetric scenes

<!-- page 5 -->
5
Subject ID 05
Subject ID 07
Subject ID 16
Subject ID 22
Fig. 3: Viewport trajectories captured from four of the subjects.
Different subjects have different behavioral characteristics in
viewing volumetric scenes. For example, subjects like 07 and
16 prefer to surround the scenes randomly, while subjects
like 05 and 22 tend to stand in fixed positions with fewer
movements.
are meticulously subdivided into non-overlapping cubic tiles,
resulting in downsampled representations at L different quality
levels. When the user watches volumetric scenes via an HMD,
the instant viewport information is collected and routed to the
CVP module, which aggregates these real-time viewing pa-
rameters and prepares them for prediction. GSStream predicts
the user’s future viewports by leveraging both the historical
viewport sequence and learned user-specific embeddings. The
predicted viewport is then used to generate a representation
selection strategy, ensuring that each cubic tile is streamed
at a suitable quality level aligned with the user’s immediate
and near-future needs. Afterward, the server-side transmits
the selected representations in accordance with the prediction,
thereby achieving progressive streaming while optimizing net-
work usage and preserving an immersive viewing experience.
A. Pre-Processing
In the pre-processing phase, the entire volumetric scene is
divided into several non-overlapping cubic spaces (referred to
as tiles), which are subsequently downsampled into multiple
representations for the purpose of selective transmission. Due
to the inherently uneven distribution of 3D Gaussians in
3D space, the number of points contained within each tile
can vary. Similarly, the total number of tiles into which
the volumetric scene is divided is also not fixed and can
change depending on the characteristics of the scene. To
facilitate computations, we treat each tile as a 3D region and
represent the position of the k-th tile by its center coordinates,
denoted as (xk, yk, zk). For convenience, let us assume that
the entire volumetric scene is divided into a total of K tiles. A
voxel grid filter is then applied to downsample each tile into
representations at L different quality levels. Specifically, for
any two quality levels l1 and l2, where 1 ≤l1 < l2 ≤L, the
following relationships hold:
bk,l1 ≤bk,l2,
pk,l1 ≤pk,l2,
s.t. 1 ≤l1 < l2 ≤L,
(3)
[Concatenate]
Cross
Attention
𝑥 𝑡+1:𝑡+𝐻
𝑢
𝑦 𝑡+1:𝑡+𝐻
𝑢
𝑧 𝑡+1:𝑡+𝐻
𝑢
 
𝑝 𝑡+1:𝑡+𝐻
𝑢
𝑞 𝑡+1:𝑡+𝐻
𝑢
𝑟 𝑡+1:𝑡+𝐻
𝑢
 
𝑄 
𝐾 
𝑉 
𝑥 𝑡−𝐻+1:𝑡
𝑢
𝑦 𝑡−𝐻+1:𝑡
𝑢
𝑧 𝑡−𝐻+1:𝑡
𝑢
 
𝑝 𝑡−𝐻+1:𝑡
𝑢
𝑞 𝑡−𝐻+1:𝑡
𝑢
𝑟 𝑡−𝐻+1:𝑡
𝑢
  
MLP
Two-Headed
MLPs
iTransformer
User ID
Query
Collaborative Priors Extraction (CPE) Module
Historical Priors Extraction (HPE) Module
Viewport Sequence
Predicted Future 
Viewport Sequence
Cross
Attention
𝑄 
𝐾 
𝑉 
𝑢 
𝐄 𝐄 
𝐄 𝐄 
𝐞𝑢 
𝐞𝑢 
[Concatenate]
Cross
Attention
𝑥 𝑡+1:𝑡+𝐻
𝑢
𝑦 𝑡+1:𝑡+𝐻
𝑢
𝑧 𝑡+1:𝑡+𝐻
𝑢
 
𝑝 𝑡+1:𝑡+𝐻
𝑢
𝑞 𝑡+1:𝑡+𝐻
𝑢
𝑟 𝑡+1:𝑡+𝐻
𝑢
 
𝑄 
𝐾 
𝑉 
𝑥 𝑡−𝐻+1:𝑡
𝑢
𝑦 𝑡−𝐻+1:𝑡
𝑢
𝑧 𝑡−𝐻+1:𝑡
𝑢
 
𝑝 𝑡−𝐻+1:𝑡
𝑢
𝑞 𝑡−𝐻+1:𝑡
𝑢
𝑟 𝑡−𝐻+1:𝑡
𝑢
  
MLP
Two-Headed
MLPs
iTransformer
User ID
Query
Collaborative Priors Extraction (CPE) Module
Historical Priors Extraction (HPE) Module
Viewport Sequence
Predicted Future 
Viewport Sequence
Cross
Attention
𝑄 
𝐾 
𝑉 
𝑢 
𝐄 
𝐄 
𝐞𝑢 
Fig. 4: Illustration of the CVP module. The network utilizes
both historical information and collaborative information.
where bk,l denotes the number of bits and pk,l represents
the number of points contained in the l-th level representation
of tile k. The pre-processed tile representations are then
selectively transmitted to the user based on the predicted
viewport sequence generated by the CVP module and the
corresponding output produced by the DBA module. This
selective transmission ensures that the system can maintain
a smooth data flow while still delivering an immersive and
satisfactory viewing experience to the user.
B. Collaborative Viewport Prediction
1) Motivation: As a nascent form of volumetric scene
representation, there is a notable lack of real user viewing
behavior datasets specifically for scenes derived from 3DGS
(3D Graphics Systems). This scarcity poses challenges for
accurately modeling and understanding user interactions. To
bridge this gap, we developed a user behavior data collection
platform and successfully constructed a comprehensive dataset
from 32 subjects with diverse backgrounds. When compared
to existing user behavior datasets based on point cloud repre-
sentations, our findings reveal that the high-fidelity rendering
supported by 3DGS-derived scenes leads to more active user
engagement. This increased activity better captures the unique
behavioral characteristics of each user. For instance, compared
to the point cloud video-based user behavior dataset introduced
by [32], the mean-variance of users’ positional distribution in
our dataset is 9× higher (0.45 vs. 0.08). This significant differ-
ence highlights the enhanced variability and richness of user
interactions facilitated by 3DGS-derived representations. The
qualitative analysis further supports these results, as shown in
Fig. 3, which illustrates the distinct behavioral patterns of each
subject. These differences underscore the effectiveness of our
dataset in reflecting individual user behaviors. Consequently,
we raise an important question: Can we better predict users’
future behavior based on their historical behavior sequences
by proposing a method for collaboratively learning user group
behavior? The answer is yes. We introduce a Collaborative
Visual Prediction (CVP) module, depicted in Fig. 4, which
effectively leverages both historical and collaborative infor-
mation to accurately predict the future behavior of the current
user.

<!-- page 6 -->
6
2) Overview: As depicted in Figure 4, the proposed CVP
module is designed to enhance the accuracy of viewport
predictions by integrating both user-specific characteristics and
their historical interaction data. It is important to recognize that
individual users exhibit varying degrees of behavioral inertia
when navigating through volumetric scenes, meaning that
their viewing patterns and preferences can differ substantially.
Despite this variability, existing viewport prediction methods
often process viewport trajectories from diverse users in a
uniform manner, thereby overlooking the intricate relation-
ships between unique user attributes and their corresponding
viewing behaviors. To address this gap, the proposed net-
work architecture is composed of two primary components:
the Collaborative Priors Extraction (CPE) module and the
Historical Priors Extraction (HPE) module. The CPE module
is responsible for collaboratively learning user embeddings
by leveraging data from multiple users, thereby capturing
shared patterns and individual differences. Meanwhile, the
HPE module focuses on extracting insights from the histor-
ical viewport sequences of each user, enabling the model to
understand temporal dynamics and predict future viewports
based on past behavior. The entire network takes as input the
user identifier u along with the associated historical viewport
sequence vu
t−H+1:t. Specifically, a viewport vu
t for user u at
a given time slot t is represented by a vector defined as
vu
t = (xu
t , yu
t , zu
t , pu
t , qu
t , ru
t )T ,
(4)
where the components xu
t , yu
t , and zu
t correspond to the spatial
position coordinates of the user’s virtual camera, while pu
t , qu
t ,
and ru
t denote the orientation angles of the virtual camera
at time t. At each discrete time slot t, the instantaneous
viewport vu
t , as captured by the HMD, is centralized on the
client side to construct a comprehensive historical viewport
sequence. Utilizing this sequence, the network is then capable
of predicting future viewports with higher precision, thereby
providing a more personalized and responsive user experience.
3) Collaborative Priors Extraction: The CPE module aims
to extract collaborative priors by learning from the viewport
data of multiple users simultaneously. This collaborative learn-
ing approach enables the module to identify common viewing
patterns and unique user-specific behaviors, thereby enriching
the overall prediction model. The module incorporates two dis-
tinct attention mechanisms, each designed to capture different
aspects of user interactions. The attention mechanisms operate
based on the standard attention function, which involves a
query q, a key k, and a value v, mathematically expressed
as
Attention (Q, K, V) = softmax
QKT
√dK

V,
(5)
where the transformed matrices are given by
Q = qWq,
K = kWk,
V = vWv.
(6)
In this context, Wq, Wk, and Wv are learnable weight
matrices that project the query, key, and value vectors into
a suitable feature space. To facilitate user-specific learning, an
embedding matrix E is established, where each column vector
eu serves as a learnable embedding unique to user u. This
embedding matrix is initially fed into the first cross-attention
module as the query, while the historical viewport sequence
acts as both the key and the value. This configuration allows
the model to capture temporal attention, effectively learning
how each user’s viewing behavior evolves over time. The
output from this first attention layer is subsequently passed
to a second cross-attention module. In this stage, the query
is specifically the user embedding eu, which enables the
model to learn inter-user attention, thereby understanding how
a particular user’s behavior relates to the behaviors of other
users. This two-tiered attention mechanism ensures that the
CPE module can effectively leverage both temporal dynamics
and collaborative information across multiple users to generate
robust user embeddings that are instrumental for accurate
viewport prediction.
4) Historical Priors Extraction: The HPE module is dedi-
cated to extracting historical priors from the input sequence of
past viewports, thereby providing the model with contextual
information that is crucial for predicting future viewports.
This module comprises two main components: an MLP and
an iTransformer module [33]. The MLP serves as an initial
projection layer, transforming each viewport vector at a given
time slot into a higher-dimensional feature space. Following
the MLP, the iTransformer module is employed to model
the temporal dependencies within the historical viewport
sequence, which has been demonstrated to achieve SOTA
performance in temporal sequence prediction tasks. In the
context of the proposed network, the iTransformer leverages
the enhanced feature representations from the MLP to predict
future viewport behaviors in the time domain. By doing so,
the HPE module not only encapsulates the sequential nature
of user interactions but also enhances the model’s capacity
to anticipate future viewing directions and positions based
on past behavior, thereby contributing to more accurate and
personalized viewport predictions.
Feature Merge: To synthesize the rich information learned
from both the CPE and HPE modules, we employ a series
of MLPs to effectively merge these distinct priors. Specifi-
cally, the outputs generated by the CPE and HPE modules
are first concatenated, creating a unified feature vector that
encapsulates both collaborative and historical insights. This
concatenated vector is then passed through an MLP, which
serves to integrate and transform the combined features into
a deep viewport representation. Following this integration,
another MLP is utilized to decode the deep viewport features
into actionable predictions by bifurcating into two separate
MLPs: one dedicated to predicting future camera positions
and the other focused on predicting camera rotations.
The CVP module is trained using the captured user view-
port trajectory dataset. The training process employs a loss
function defined as the mean absolute error (MAE) between
the predicted viewports ˆvu
t+1:t+H and the actual ground truth
viewports vu
t+1:t+H. By minimizing the MAE during training,
the CVP module is able to produce highly accurate viewport
predictions that closely align with the users’ true viewing
behaviors.

<!-- page 7 -->
7
Set
Abstraction
Feature
Propagation
Set
Abstraction
Feature
Propagation
[Concatenate]
[Concatenate]
State 𝐬𝑡 
Actor Head 𝜋𝜃 𝐬𝑡 ↦𝒜 
Critic Head 𝑄𝜔 𝐬𝑡, 𝐚𝑡 ↦ℝ 
Set
Abstraction
Action 𝐚𝑡 
Shared Feature Extraction (SFE) Module 
Output
Repeat
Set
Abstraction
Feature
Propagation
Set
Abstraction
Feature
Propagation
Viewport
Temporal Feature
Tile-wise
Spatial Feature
Self
Attention
𝑄 
𝐾 
𝑉 
Action Value 𝑄 𝐬𝑡, 𝐚𝑡  
ℒ𝜃 
ℒ𝜔 Replay
Buffer
Bandwidth Throughput 𝑏𝑡 
Viewport Sequence 𝐯 𝑡+1:𝑡+𝐻 
Tile Feature 𝐅𝑡 
Cross
Attention
Legend
Train
Inference
Train
Inference
Train
Train
Set
Abstraction
Feature
Propagation
[Concatenate]
[Concatenate]
State 𝐬𝑡 
Actor Head 𝜋𝜃 𝐬𝑡 ↦𝒜 
Critic Head 𝑄𝜔 𝐬𝑡, 𝐚𝑡 ↦ℝ 
Set
Abstraction
Action 𝐚𝑡 
Shared Feature Extraction (SFE) Module 
Output
Repeat
Set
Abstraction
Feature
Propagation
Viewport
Temporal Feature
Tile-wise
Spatial Feature
Self
Attention
𝑄 
𝐾 
𝑉 
Action Value 𝑄 𝐬𝑡, 𝐚𝑡  
ℒ𝜃 
ℒ𝜔 Replay
Buffer
Bandwidth Throughput 𝑏𝑡 
Viewport Sequence 𝐯 𝑡+1:𝑡+𝐻 
Tile Feature 𝐅𝑡 
Cross
Attention
Legend
Train
Inference
Train
Fig. 5: The illustration of the whole architecture of the proposed DRL-based bitrate adaptation algorithm. The proposed
algorithm is based on DDPG, which is under the Actor-Critic framework [34], with the actor head π(st) outputting an action
at, and the critic head outputting the corresponding action value Q (st, at).
C. DRL-based Bitrate Adaptation
To achieve efficient 3DGS content delivery, we propose a
novel DBA module that adaptively determines the optimal
transmitted representations at each discrete time slot t based on
comprehensive tile statistics, the available network bandwidth
of the current user, and the user’s anticipated future viewports
as predicted by the CVP module. In this section, our proposed
bitrate adaptation strategy is specifically defined and analyzed
within a single-user scenario, allowing us to omit the super-
script u that denotes the user in previous sections, thereby
facilitating a more concise and streamlined presentation.
1) MDP Formulation: To address the bitrate adaptation
challenge across discrete time slots utilizing DRL methods, we
model the problem as an MDP. Specifically, we characterize
the state space by integrating several key components: the
bandwidth throughput bt at each discrete time slot t, the
predicted sequence of future viewports denoted as ˆvt+1:t+H,
and the comprehensive feature information of all tiles present
within the volumetric scene, represented by Ft. As a result,
the DBA module within the GSStream system functions as
an intelligent decision-maker. It is responsible for selecting an
appropriate action at ∈A based on the current state st ∈S,
with the objective of maximizing the received reward rt ∈R.
Here, S and A denote the state and action spaces, respectively.
In greater detail, the definitions of the state space, action
space, and the corresponding reward function are elaborated
as follows:
State Space: The observed state st at each time slot t is
defined as a combination of the current bandwidth throughput
bt (measured in bits per time slot), the predicted future user
viewport sequence ˆvt+1:t+H and the feature information of
tiles within the volumetric scene Ft =
 f 1
t , f 2
t , . . . , f K
t
T with
K denoting the number of tiles within a certain volumetric
scene, as
st = (bt, ˆvt+1:t+H, Ft) .
(7)
Notably, f k
t is a feature vector that describes tile k at time slot
t. For each tile k, the corresponding f k
t is defined as
f k
t =
 
xk, yk, zk,
pprev
k,t
pk,L
,
sprev
k,t
sk,L
,
lprev
k,t
L
!T
.
(8)
in which sk,l is an importance score of the l-th representation
of tile k, while pprev
k,t , sprev
k,t and lprev
k,t denote the number of points,
the importance score and the quality level of the tiles that have
already been transmitted (or selected) in previous time slots
respectively, with L denoting the highest quality level. Aiming
at the splatting nature of the rendering technique of Scaffold-
GS, the importance of each anchor is indeed represented by the
base scaling factors. To this end, we implement the importance
score for each tile as a combination:
sk,l =
nk,l
X
i=1
Y
(Si),
(9)
where nk,l is the number of anchors within the l-level repre-
sentation of tile k, while Q(Si) denoting the production of the
scaling factors of anchor i as the base volume of the associated
neural Gaussians.
Action Space: In our formulation, the action space is de-
fined as A = RK×L. At each time slot, entry ak,l
t
∈at denotes
the preference score of tile k to be transmitted in selected level
l at time slot t. Rather than determining the quality level of
each tile directly by the DRL method, GSStream selects the
representations according to the action at in a greedy way,
forming a set
Xt =
n
xk,l
t
o
s.t. xk,l
t
∈{0, 1},
(10)
in which each item xk,l
t
= 1 denotes that tile k is selected to
transmitted in level l at time slot t, otherwise no. The set Xt
is constructed by greedily setting xk,l
t
= 1 with the highest
ak,l
t
until
K
X
k
L
X
l

bk,l −bprev
k,t

· xk,l
t
> bt,
(11)
to ensure a smooth and progressive streaming. The set Xt also
satisfies
L
X
l
xk,l
t
≤1, ∀1 ≤k ≤K.
(12)
Reward Function: A reward function is a metric that can
evaluate each action at according to the state st. In our design,
the DRL algorithm participates in decision-making indirectly

<!-- page 8 -->
8
by determining the set Xt. Therefore, the reward function
rt (st, at) is strongly associated with the set Xt, as
rt(st, at) = ω1
K
X
k
yk
t
L
X
l
 
pk,l −pprev
k,t
pk,L −pprev
k,t
!
· xk,l
t
+ ω2
K
X
k
yk
t
L
X
l
 
sk,l −sprev
k,t
sk,L −sprev
k,t
!
· xk,l
t
−ω3 max {tcost −∆t, 0} ,
(13)
in which yk
t ∈Yt satisfying
Yt =

yk
t
	
s.t. yk
t ∈{0, 1}.
(14)
In the set Yt, each item yk
t
= 1 denotes that tile k will
appear in the user’s FoV within H time slots in the future,
according to the predicted viewport sequence ˆvt+1:t+H, vice
versa. Moreover, ω1, ω2 and ω3 are hyperparameters. In the
reward function, term pk,l −pprev
k,t
is the number of points
in tile k that will be incrementally transmitted at time slot
t, and term pk,L −pprev
k,t
is the number of the remaining
untransmitted points. Inspired by the idea of the Signed
Distance Field (SDF) [35], in which the sign of a distance
indicates the collision relationship between a tile and the user’s
view frustum, the first term in r(st, at) acts as a progressive
streaming reward, encouraging the network to transmit the
tile representations closer to the user’s FoV according to the
value of yk
t and with higher incremental content. Similarly,
the second term promotes the willingness of the network to
transmit tiles with higher incremental importance score, while
the last term is a delay time penalty with ∆t as the time slot
duration, and
tcost = 1
bt
K
X
k
L
X
l

bk,l −bprev
k,t

· xk,l
t .
(15)
Therefore, the state transition function can be formulated as
P (st+1, rt|st, at).
2) DDPG-based Algorithm: The proposed DDPG network
architecture is depicted in Figure 5. It is important to note
that the MDP defined in Section III-C1 operates within a
continuous action space, making it well-suited for resolution
using a DDPG-based approach. DDPG is a DRL algorithm that
effectively integrates the Actor-Critic framework as introduced
by Konda and Tsitsiklis [34]. Instead of employing two
separate networks to independently learn the policy function
π and the action value function Q, our design incorporates a
Shared Feature Extraction (SFE) block, along with dedicated
actor and critic heads. This shared feature extraction promotes
efficiency and coherence between the policy and value estima-
tions.
The SFE block is a pivotal component of our DDPG
network, designed to process and integrate various features ef-
fectively. It includes a self-attention module that captures tem-
poral features from the predicted viewport sequence ˆvt+1:t+H.
Additionally, the block utilizes a Set Abstraction (SA) module,
and a Feature Propagation (FP) module, as proposed by Qi et
al. [36], to extract tile-wise spatial features. The roles of the
SA and FP modules are further elaborated in Section III-C3.
Subsequently, the SFE block merges the temporal and spatial
features using a cross-attention module, where the tile-wise
spatial feature acts as the query and the viewport temporal
feature serves as both the key and value, as described in
Equation (5). The resulting feature is spatially aligned and
concatenated with the bandwidth throughput bt, thereby gen-
erating a high-level feature for each tile k.
The high-level features produced by the SFE block are then
fed into the actor and critic heads. In the actor head, we again
apply the SA and FP modules to ensure accurate estimation of
the tile-wise preference scores at. Each entry ak,l
t
represents
the preference score for transmitting tile k at level l during
time slot t. In contrast, the critic head focuses on estimating
an overall action value function Q(st, at) across all tiles in
the scene. To achieve this, the critic head employs a single
SA module for global feature abstraction and omits the FP
module, streamlining the evaluation of action values.
The training process involves constructing two DDPG
networks: the primary DDPG network {πθ(st), Qω(st, at)},
which interacts with the environment, and the target network
{πθ′(st), Qω′(st, at)}, which aids in stabilizing training. Both
networks are initialized with identical parameters. As the
primary network interacts with the environment, experiences
are stored in a replay buffer R as tuples (st, at, rt, s′
t), where
s′
t is the next state resulting from action at in state st. During
training, batches B ⊆R are sampled to update the networks.
The critic network’s gradient is computed as
∇ωLω = 1
|B|
|B|
X
i=1
2 (ri + γQω′ (s′
i, a′
i) −Qω (si, ai)) , (16)
where a′
i = πθ′ (s′
i) is the action chosen by the target
network in state s′
i, and γ is the discount factor. Simultane-
ously, the policy gradient for updating the actor network is
approximated by
∇θLθ ≈1
|B|
|B|
X
i=1
∇θQω (si, ai) .
(17)
This training regimen ensures that the actor improves its
policy based on the feedback from the critic, leading to more
effective decision-making over time.
In summary, the pseudo-code is shown in Algorithm 1, with
τ as the soft update factor and N as a random Gaussian noise
to encourage the exploration.
3) Space Variability: In various volumetric scenes, the
distribution of points within the 3D space is often non-
uniform, and the bounding boxes do not entirely overlap.
This inconsistency results in a variable number of tiles across
different scenes. Moreover, this variability causes the state and
action spaces in the MDP to fluctuate accordingly. Within DRL
algorithms, this issue of spatial variability manifests itself in
the form of inputs and outputs for the neural network that
consists of tensors with varying dimensions.
One potential solution to this challenge is to implement
neural network modules specifically designed to handle tensors
with varying dimensions. For instance, consider the input tile
feature Ft. A straightforward approach would be to employ a
Transformer or an attention-based module [29] to effectively

<!-- page 9 -->
9
Algorithm 1 DDPG-based Algorithm
Input :
{πθ (s) , Qω (s, a)}, {πθ′ (s) , Qω′ (s, a)},
P (st+1, rt|st, at)
1: initialization θ′ ←θ, ω′ ←ω, N, R;
2: for episode e = 1 →E do
3:
s1 ∼P (s1);
4:
for step t = 1 →T do
5:
at ←πθ (st) + N;
6:
st+1, rt ∼P (st+1, rt|st, at);
7:
R ←R ∪{(st, at, rt, st+1)};
8:
end for
9:
sample batch B ⊆R;
10:
update ω w.r.t. Equation (16);
11:
update θ w.r.t. Equation (17);
12:
θ′ ←τθ + (1 −τ)θ′;
13:
ω′ ←τω + (1 −τ)ω′;
14: end for
capture the features of each tile, treating Ft as a sequence
with a variable length. However, this approach introduces two
significant issues. Firstly, from a data structure perspective,
each tile is represented by a vector f k
t , and Ft essentially
constitutes an unordered set of vectors with varying lengths.
As a result, the inherent ordering of Ft can interfere with the
accurate extraction of features. Secondly, conventional Trans-
formers and attention modules are not adept at emphasizing
the spatial relationships between tiles, as they typically treat
the spatial coordinates of each tile on par with other attributes,
thereby neglecting the spatial dependencies. To overcome
these limitations, we incorporate the SA and FP modules
from [36] to effectively manage the variability in the state
space, as depicted in Figure 5.
Within the SFE block, we consider Ft as an unordered
collection of vectors. The sampling and grouping layers of
the SA module are responsible for hierarchically aggregating
features extracted from sampled tiles and their neighboring
tiles, thereby constructing more sophisticated, high-level rep-
resentations. Subsequently, the feature propagation mechanism
ensures that these high-level features are accurately learned
with respect to each individual tile. Consequently, the tile-
wise spatial features illustrated in Figure 5 form an unordered
set where each element is a vector representing the high-level
embedding of its corresponding tile. This embedding reflects
not only the intrinsic features of the tile but also its local
and global relationships within the overall volumetric scene.
In a similar fashion, we apply the SA and FP modules within
both the actor head and the critic head to enhance the overall
streaming performance of the system.
IV. EXPERIMENTS
To assess the performance of GSStream against existing
streaming systems, we conduct a series of experiments. These
experiments were designed to simulate real user behaviors
and adapt to dynamic network conditions, ensuring that our
evaluation reflects practical usage scenarios.
Scene
bicycle
bonsai
chtf
counter
drjohnson
#Tiles
2,130
529
2,736
180
59
Scene
flowers
garden
kitchen
playroom
room
#Tiles
1,340
693
480
124
197
Scene
sculpture
stump
train
treehill
truck
#Tiles
1,471
11,601
1,267
2,326
1,741
TABLE I: The numbers of tiles of each scene. The 3 scenes
marked red construct the testing set while the remaining 12
scenes form the training set.
bicycle
bonsai
chtf
counter
drjohnson
flowers
garden
kitchen
playroom
room
sculpture
stump
train
treehill
truck
Fig. 6: Thumbnails of the volumetric scenes, with scenes
marked red forming the testing set, and the others forming
the training set.
A. Dataset
We initiated our study by developing a user viewport trajec-
tory collection platform specifically for 3DGS-based streaming
tasks. A total of 32 subjects were recruited to provide viewport
trajectory data. The dataset includes 15 volumetric scenes,
with a balanced distribution of 8 outdoor and 7 indoor envi-
ronments. Each subject was instructed to freely explore each
volumetric scene for one minute, resulting in a dataset that
captured over 864,000 frames. This extensive data collection
ensures a realistic and comprehensive dataset, enhancing the
validity of our experimental results. All volumetric scenes
utilized in the experiments are presented as thumbnails in
Figure 6, offering a visual overview of the scenes included
in our study.
1) Scene Setup: Out of the 15 scenes utilized, 13 are
selected from public-available datasets. Specifically, 2 from
Tanks&Temples [37], 2 from Deep Blending [38], and 9 from
Mip-NeRF 360 [39]. The remaining 2 scenes were captured by
our team to ensure a diverse set of environments. Each scene
is pre-trained following the pipeline from [3] and divided into
tiles of size 3.2m×3.2m×3.2m. The number of tiles per scene
is detailed in Table I. We create L = 6 quality levels through
voxel grid downsampling with voxel sizes of 16cm, 8cm, 4cm,
2cm, and 1cm, respectively, while the highest quality level
retains all unsampled points.
2) Dataset Setup: A total of 32 subjects, aged between
18 and 25 with an equal gender distribution, viewed these
volumetric scenes for 1 minute each in a controlled 4m×4m
area. The dataset is split into training and testing sets, with
12 scenes and their corresponding viewport data forming the
training set and the remaining 3 scenes used for testing,

<!-- page 10 -->
10
as shown in Figure 6. Additionally, Figure 11 displays the
viewport trajectories collected from the subjects.
3) Dataset Capture: User viewport data were captured
using the HTC VIVE Pro Eye HMD. We developed a custom
Unreal Engine 5 (UE5) project to record viewport data from
the HMD at a rate of 30 frames per second (every 1/30
seconds).
B. Implementation Details
During training, the CVP and DBA modules were trained
separately before integration into the GSStream system. Key
implementation details are as follows:
1) Collaborative Viewport Prediction: The CVP module
was set with a prediction horizon H = 30. Both viewport and
user embeddings were 16-dimensional. The attention modules
and the integrated iTransformer each use 1 head. The learning
rate is set to 1 × 10−5.
2) DRL-based Bitrate Adaptation: In the DBA module, the
temporal discount factor γ is 0.9. The reward function weights
are ω1 = ω2 = 10 and ω3 = 1. Each minibatch contains 60
samples, with a soft update factor τ = 0.005. The time slot
duration ∆t is 1 second.
3) Bandwidth Trace Simulation: To mimic a real-world
streaming condition, we utilize bandwidth traces from cellular
networks [40]. We select 10 traces and scale them to average
bandwidths of 40Mbps, 80Mbps, and 120Mbps. During train-
ing, bandwidth conditions are randomly selected from these
scaled traces to ensure robustness across different network
scenarios.
C. Baselines
To evaluate GSStream’s performance, we compare it against
two types of baseline methods.
1) Baselines for Volumetric Video Streaming: The first set
of baselines includes prominent methods in volumetric video
streaming, adapted for static 3DGS scene streaming:
• ViVo [7]: ViVo is a rule-based method that selects tile
representations based on predicted user viewports. For
static scenes, we masked previously selected representa-
tions to prevent duplicate transmissions. Originally sup-
porting 5 quality levels, we interpolated to accommodate
6 levels in our setup.
• CaV3 [4]: CaV3 uses a cache-assisted strategy for effi-
cient streaming with effective tile caching and eviction.
For static scenes, we retained only the caching mecha-
nism to focus on tile selection based on priority, ensuring
fair comparison with GSStream.
2) Baseline for 3DGS Scene Streaming: Due to the limited
research on 3DGS scene streaming, we compared GSStream
with the open-source 3DGS web viewer GS3D:
• GS3D [5]: GS3D streams representations by sorting 3D
Gaussians based on their distance from the world origin,
prioritizing closer tiles. In a tile-based context, this means
selecting tiles closer to the origin first. Unlike GSStream,
GS3D relies solely on distance without incorporating
viewport predictions or adaptive bitrate strategies.
Scene
Average Viewport SSIM ↑
System
40Mbps
80Mbps
120Mbps
chtf
187MB
ViVo
0.10 ± 0.08
0.15 ± 0.12
0.18 ± 0.13
CaV3
0.56 ± 0.43
0.77 ± 0.38
0.84 ± 0.34
GS3D
0.61 ± 0.31
0.77 ± 0.31
0.85 ± 0.28
Ours
0.64 ± 0.37
0.81 ± 0.32
0.84 ± 0.31
garden
224MB
ViVo
0.49 ± 0.33
0.54 ± 0.32
0.58 ± 0.31
CaV3
0.67 ± 0.31
0.79 ± 0.29
0.89 ± 0.23
GS3D
0.66 ± 0.30
0.74 ± 0.29
0.79 ± 0.27
Ours
0.72 ± 0.29
0.87 ± 0.23
0.92 ± 0.18
stump
224MB
ViVo
0.40 ± 0.32
0.43 ± 0.32
0.47 ± 0.31
CaV3
0.54 ± 0.34
0.72 ± 0.33
0.90 ± 0.23
GS3D
0.60 ± 0.29
0.75 ± 0.29
0.82 ± 0.26
Ours
0.71 ± 0.29
0.89 ± 0.22
0.91 ± 0.21
TABLE II: The average viewport SSIM of different methods.
D. Results
1) Comparison on Viewport Prediction Task: We compared
the viewport prediction accuracy (measured in MAE) of
the proposed CVP module against ViVo [7], CaV3 [4] and
GS3D [5]. As illustrated in Figure 9, the proposed method
achieves SOTA among the compared representative systems.
2) Comparison on Bitrate Adaptation Task:
We com-
pare GSStream with three representative systems: ViVo [7],
CaV3 [4] and GS3D [5] with the above-mentioned bandwidth
simulation. To evaluate the performance of each streaming sys-
tem comprehensively, we construct comparison experiments
from three perspectives.
Average Viewport SSIM: As quantified in Table II, the
proposed GSStream framework achieves 118.9%, 9.4%, and
10.9% higher mean viewport SSIM than ViVo, CaV3, and
GS3D respectively, where viewport SSIM measures struc-
tural similarity between original and transmitted volumet-
ric content. This superiority derives from overcoming three
key limitations: ViVo’s rigid rule-based transmission, CaV3’s
binary quality selection restricting adaptation to bandwidth
extremes, and GS3D’s origin-anchored spatial prioritization
that degrades early-stage quality during viewpoint exploration.
By synergizing continuous quality spectrum selection with
temporal-aware adaptation, GSStream demonstrates compre-
hensive advantages in both perceptual quality maintenance and
adaptive efficiency, conclusively outperforming all baseline
methods for progressive volumetric streaming.
Temporal Visual Quality: Figure 7 depicts the tempo-
ral visual quality evolution across four streaming systems
under stochastic network dynamics, revealing three critical
observations. (1) Systemic adaptability: While all systems
exhibit progressive quality enhancement over time under
bandwidth-constrained scenarios, ViVo’s static prioritization
rules constrain its responsiveness to dynamic network con-
ditions. (2) Phase-sensitive divergence: Though CaV3 tem-
porarily achieves marginal SSIM advantages during initial
exploration phases through RL-based discretized adaptation, it
experiences early-stage quality fluctuation from binary quality
decisions. (3) Spatial-temporal optimization: GS3D’s origin-
biased transmission strategy unintentionally penalizes mid-
stage viewing experiences during coordinate-distant viewpoint
transitions, whereas GSStream continuously refines recon-

<!-- page 11 -->
11
0
10
20
30
40
50
60
Time (s)
0.4
0.6
0.8
SSIM
40 Mbps
0
10
20
30
40
50
60
Time (s)
0.4
0.6
0.8
1.0
80 Mbps
0
10
20
30
40
50
60
Time (s)
0.4
0.6
0.8
1.0
120 Mbps
ViVo
CaV3
GS3D
Ours
Fig. 7: The temporal viewport SSIM of the four volumetric scene streaming systems at each time slot t, under three typical
network conditions.
0
10
20
30
40
50
60
Time (s)
0
10
20
30
40
Data Transmitted (Mbits)
40 Mbps
0
10
20
30
40
50
60
Time (s)
0
20
40
60
80
80 Mbps
0
10
20
30
40
50
60
Time (s)
0
20
40
60
80
100
120
120 Mbps
Bandwidth
ViVo
CaV3
GS3D
Ours
Fig. 8: The accumulative transmitted data of the four volumetric scene streaming systems at each time slot t, under three
typical network conditions.
(a) The MAE of the future posi-
tion prediction.
(b) The MAE of the future rota-
tion prediction.
Fig. 9: Results of the comparison experiments on viewport
prediction task. Note that the compared baselines have dif-
ferent lengths in the prediction time window; therefore, the
experimental results have limited significance.
struction fidelity through temporal prediction-accelerated LOD
switching, ultimately achieving 25.9%/80.3% higher late-stage
visual detail retention than CaV3/GS3D respectively. These
empirical measurements confirm our framework’s capability
to synergize structural preservation with adaptive streaming
efficiency, fundamentally overcoming the spatial-temporal de-
coupling limitations inherent in baseline architectures.
Bandwidth Throughput Usage: Figure 8 reveals dis-
tinct bandwidth utilization patterns across four volumetric
streaming architectures. The results show that the proposed
GSStream system consistently outperforms ViVo, CaV3, and
(a) The MAE of the future posi-
tion prediction.
(b) The MAE of the future rota-
tion prediction.
Fig. 10: Results of the ablation experiments.
GS3D in terms of data transmission efficiency and bandwidth
utilization. While other methods like ViVo and CaV3 exhibit
significant fluctuations in throughput and fail to effectively
utilize available bandwidth, GSStream maintains stable and
high data throughput across all tested bandwidth conditions.
In contrast, GS3D, although improving throughput to some ex-
tent, still lags behind Ours in efficiently utilizing the available
bandwidth. Overall, Ours demonstrates superior performance
in static 3D scene streaming by ensuring stable and optimal
bandwidth utilization.
3) Ablation Experiments: We construct ablation experi-
ments for our proposed CVP module by masking the historical
prior information and the collaborative prior information re-
spectively, denoted as CVP w/o HPE and CVP w/o CPE

<!-- page 12 -->
12
in Figure 10, where CVP denotes the proposed collaborative
viewport prediction algorithm. As we can see, CVP fuses
the two signals and achieves the lowest average MAE in
both position and rotation prediction. Without the collaborative
priors, CVP w/o HPE still achieves higher performance than
CVP w/o CPE in position prediction, which reflects that the
viewing routes of different users when watching volumetric
scenes are affected by their inherent habits significantly. On the
other hand, without historical priors, CVP w/o HPE performs
weaker than CVP w/o CPE in rotation prediction, which indi-
cates that the rotation angles changed within the time window
H have a stronger sequential nature. However, the proposed
collaborative viewport prediction algorithm succeeded in fus-
ing all two signals, achieving the best performance among all
three settings.
V. LIMITATIONS
Dynamic media support: The proposed GSStream sys-
tem currently only considers the streaming of volumetric
scenes. However, with the rapid development of virtual reality
technology, the topic of volumetric video streaming is also
widely developed. Recently, pioneer works have explored the
possibility of using 3DGS-related technologies as a novel
form of volumetric video implementation, and have also made
amazing breakthroughs [20], [41]–[43]. In the future, one of
our key work directions is to expand GSStream to support
dynamic volumetric data streaming.
More optimization: Recently, 3DGS has attracted a lot of
interest from researchers. Numerous cutting-edge works are
exploring whether a more compact representation of 3DGS can
be given to save the data scale of volumetric scene represen-
tation [18], [19]. Besides, some researchers are committed to
developing codecs for 3DGS data to generate representations
in different quality levels, to achieve efficient and progressive
encoding and decoding [44], [45] as developed in the field
of point cloud compression [46]. However, in our proposed
GSStream system, representations are downsampled from the
original tiles. One of our future directions is to introduce
efficient 3DGS codecs, achieving greater savings on bandwidth
throughput.
VI. CONCLUSION
In this paper, we propose GSStream, a novel 3DGS-based
volumetric scene-streaming system. We utilize 3DGS as a new
form of volumetric scene representation, greatly enhancing
the display visual quality, compared with existing volumetric
scene streaming systems. The proposed GSStream system
integrates a CVP module and a DBA module. The CVP
module utilizes the user-specific embeddings learned from
multiple users collaboratively alongside the historical priors
learned from the historical viewport sequences, promoting the
viewport prediction accuracy. The DBA module tackles the
variability challenge of state and action space in the bitrate
adaptation problem by introducing the SA and FP modules,
then solves it via a DDPG-based algorithm. Experiments
demonstrate the efficiency and effectiveness of the proposed
volumetric scene streaming system among existing SOTA
streaming systems.
Subject 01
Subject 02
Subject 03
Subject 04
Subject 05
Subject 06
Subject 07
Subject 08
Subject 09
Subject 10
Subject 11
Subject 12
Subject 13
Subject 14
Subject 15
Subject 16
Subject 17
Subject 18
Subject 19
Subject 20
Subject 21
Subject 22
Subject 23
Subject 24
Subject 25
Subject 26
Subject 27
Subject 28
Subject 29
Subject 30
Subject 31
Subject 32
Fig. 11: The collected viewport trajectories from 32 subjects.
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, 2023.
[2] Y. Bao, T. Ding, J. Huo, Y. Liu, Y. Li, W. Li, Y. Gao, and J. Luo, “3d
gaussian splatting: Survey, technologies, challenges, and opportunities,”
IEEE Transactions on Circuits and Systems for Video Technology, pp.
1–1, 2025.
[3] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.
[4] J. Liu, B. Zhu, F. Wang, Y. Jin, W. Zhang, Z. Xu, and S. Cui, “Cav3:
Cache-assisted viewport adaptive volumetric video streaming,” in 2023
IEEE Conference Virtual Reality and 3D User Interfaces (VR).
IEEE,
2023, pp. 173–183.
[5] M.
Kellogg,
“3d
gaussian
splatting
for
three.js,”
2024,
https://github.com/mkkellogg/GaussianSplats3D.
[6] S. Schwarz, M. Preda, V. Baroncini, M. Budagavi, P. Cesar, P. A. Chou,
R. A. Cohen, M. Krivoku´ca, S. Lasserre, Z. Li et al., “Emerging mpeg
standards for point cloud compression,” IEEE Journal on Emerging and
Selected Topics in Circuits and Systems, vol. 9, no. 1, pp. 133–148,
2018.
[7] B. Han, Y. Liu, and F. Qian, “Vivo: Visibility-aware mobile volumetric
video streaming,” in Proceedings of the 26th annual international
conference on mobile computing and networking, 2020, pp. 1–13.
[8] C. Zhang, Y. Cao, Z. Liu, R. Yin, Y. Zhu, and X. Chen, “Trans-rl: A
prediction-control approach for qoe-aware point cloud video streaming,”
in GLOBECOM 2022-2022 IEEE Global Communications Conference.
IEEE, 2022, pp. 1899–1904.
[9] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction.
MIT press, 2018.
[10] D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. Riedmiller,

<!-- page 13 -->
13
“Deterministic policy gradient algorithms,” in International conference
on machine learning.
Pmlr, 2014, pp. 387–395.
[11] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa,
D. Silver, and D. Wierstra, “Continuous control with deep reinforcement
learning,” arXiv preprint arXiv:1509.02971, 2015.
[12] Z. Liu, Q. Li, X. Chen, C. Wu, S. Ishihara, J. Li, and Y. Ji, “Point cloud
video streaming: Challenges and solutions,” IEEE Network, vol. 35,
no. 5, pp. 202–209, 2021.
[13] M. Levoy and T. Whitted, “The use of points as a display primitive,”
Technical Report, The University of North Carolina at Chapel Hill, Dept.
of Computer Science, 1985.
[14] H. Pfister, M. Zwicker, J. Van Baar, and M. Gross, “Surfels: Surface
elements as rendering primitives,” in Proceedings of the 27th annual
conference on Computer graphics and interactive techniques, 2000, pp.
335–342.
[15] G. Bui, T. Le, B. Morago, and Y. Duan, “Point-based rendering enhance-
ment via deep learning,” The Visual Computer, vol. 34, pp. 829–841,
2018.
[16] M. Meshry, D. B. Goldman, S. Khamis, H. Hoppe, R. Pandey,
N. Snavely, and R. Martin-Brualla, “Neural rerendering in the wild,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2019, pp. 6878–6887.
[17] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[18] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 719–21 728.
[19] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, “Compact 3d
scene representation via self-organizing gaussian grids,” in European
Conference on Computer Vision.
Springer, 2024, pp. 18–34.
[20] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-
the-fly training of 3d gaussians for efficient streaming of photo-realistic
free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 20 675–20 685.
[21] J. Van Der Hooft, T. Wauters, F. De Turck, C. Timmerer, and H. Hell-
wagner, “Towards 6dof http adaptive streaming through point cloud
compression,” in Proceedings of the 27th ACM International Conference
on Multimedia, 2019, pp. 2405–2413.
[22] T. Stockhammer, “Dynamic adaptive streaming over http– standards and
design principles,” in Proceedings of the second annual ACM conference
on Multimedia systems, 2011, pp. 133–144.
[23] J. Park, P. A. Chou, and J.-N. Hwang, “Rate-utility optimized streaming
of volumetric media for augmented reality,” IEEE Journal on Emerging
and Selected Topics in Circuits and Systems, vol. 9, no. 1, pp. 149–162,
2019.
[24] L. Wang, C. Li, W. Dai, S. Li, J. Zou, and H. Xiong, “Qoe-driven
adaptive streaming for point clouds,” IEEE Transactions on Multimedia,
2022.
[25] J. Li, C. Zhang, Z. Liu, W. Sun, and Q. Li, “Joint communication
and computational resource allocation for qoe-driven point cloud video
streaming,” in ICC 2020-2020 IEEE International Conference on Com-
munications (ICC).
IEEE, 2020, pp. 1–6.
[26] J. Li, C. Zhang, Z. Liu, R. Hong, and H. Hu, “Optimal volumetric
video streaming with hybrid saliency based tiling,” IEEE Transactions
on Multimedia, 2022.
[27] J. Li, H. Wang, Z. Liu, P. Zhou, X. Chen, Q. Li, and R. Hong, “Toward
optimal real-time volumetric video streaming: A rolling optimization
and deep reinforcement learning based approach,” IEEE Transactions on
Circuits and Systems for Video Technology, vol. 33, no. 12, pp. 7870–
7883, 2023.
[28] X. Wang, W. Liu, S. Gong, Z. Liu, J. Xu, and Y. Fang, “Spatial
quality oriented rate control for volumetric video streaming via deep
reinforcement learning,” IEEE Transactions on Circuits and Systems for
Video Technology, pp. 1–1, 2024.
[29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in
neural information processing systems, vol. 30, 2017.
[30] F. B. Yusuf, V. Stebliankin, G. Vietri, and G. Narasimhan, “Cache
replacement as a mab with delayed feedback and decaying costs,” arXiv
preprint arXiv:2009.11330, 2020.
[31] A. Slivkins et al., “Introduction to multi-armed bandits,” Foundations
and Trends® in Machine Learning, vol. 12, no. 1-2, pp. 1–286, 2019.
[32] K. Hu, H. Yang, Y. Jin, J. Liu, Y. Chen, M. Zhang, and F. Wang,
“Understanding user behavior in volumetric video watching: Dataset,
analysis and prediction,” in Proceedings of the 31st ACM International
Conference on Multimedia, 2023, pp. 1108–1116.
[33] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itrans-
former: Inverted transformers are effective for time series forecasting,”
in The Twelfth International Conference on Learning Representations,
2024.
[34] V. Konda and J. Tsitsiklis, “Actor-critic algorithms,” Advances in neural
information processing systems, vol. 12, 1999.
[35] H. Oleynikova, A. Millane, Z. Taylor, E. Galceran, J. Nieto, and
R. Siegwart, “Signed distance fields: A natural representation for both
mapping and planning,” in RSS 2016 workshop: geometry and beyond-
representations, physics, and scene understanding for robotics.
Uni-
versity of Michigan, 2016.
[36] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical
feature learning on point sets in a metric space,” Advances in neural
information processing systems, vol. 30, 2017.
[37] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics, vol. 36, no. 4, 2017.
[38] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Bros-
tow, “Deep blending for free-viewpoint image-based rendering,” ACM
Transactions on Graphics (ToG), vol. 37, no. 6, pp. 257:1–257:15, 2018.
[39] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” CVPR,
2022.
[40] J. van der Hooft, S. Petrangeli, T. Wauters, R. Huysegems, P. Rondao Al-
face, T. Bostoen, and F. De Turck, “HTTP/2-based adaptive streaming
of HEVC video over 4G/LTE networks,” IEEE Communications Letters,
vol. 20, no. 11, pp. 2177–2180, 2016.
[41] Z. Xu, S. Peng, H. Lin, G. He, J. Sun, Y. Shen, H. Bao, and X. Zhou,
“4k4d: Real-time 4d view synthesis at 4k resolution,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2024, pp. 20 029–20 040.
[42] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and
X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2024, pp. 20 310–20 320.
[43] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, “Motion-aware 3d gaussian
splatting for efficient dynamic scene reconstruction,” IEEE Transactions
on Circuits and Systems for Video Technology, pp. 1–1, 2024.
[44] K. L. Navaneet, K. Pourahmadi Meibodi, S. Abbasi Koohpayegani, and
H. Pirsiavash, “Compgs: Smaller and faster gaussian splatting with vec-
tor quantization,” in Computer Vision – ECCV 2024, A. Leonardis,
E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds. Cham:
Springer Nature Switzerland, 2025, pp. 330–349.
[45] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: Efficient accelerated
3d gaussians with lightweight encodings,” in European Conference on
Computer Vision.
Springer, 2024, pp. 54–71.
[46] P. Zhang, X. Wang, L. Ma, S. Wang, S. Kwong, and J. Jiang, “Pro-
gressive point cloud upsampling via differentiable rendering,” IEEE
Transactions on Circuits and Systems for Video Technology, vol. 31,
no. 12, pp. 4673–4685, 2021.
Zhiye Tang received the B.E. degree from Shen-
zhen University, Shenzhen, China, in 2022. He is
currently working toward the M.S. degree with the
College of Computer Science and Software Engi-
neering, Shenzhen University, China. His research
interests include compact representation and efficient
transmission for 3D Gaussian Splatting.
-0.5in

<!-- page 14 -->
14
Qiudan Zhang received the B.E. and M.S. degrees
in the College of Computer Science and Software
Engineering from Shenzhen University, China in
2015 and 2018, respectively. She received her Ph.D.
degree from the Department of Computer Science,
City University of Hong Kong, China (Hong Kong
SAR) in 2021. She is currently an Assistant Profes-
sor in the College of Computer Science and Soft-
ware Engineering, Shenzhen University, China. Her
research interests include computer vision, visual
attention, 3D vision and deep learning.
Lei Zhang (Member, IEEE) received the B.Eng.
degree from the Advanced Class of Electronics and
Information Engineering, Huazhong University of
Science and Technology, Wuhan, China, in 2011,
and the M.S. and Ph.D. degrees from Simon Fraser
University, Burnaby, BC, Canada, in 2013 and 2019,
respectively. He is currently an Assistant Professor
with the College of Computer Science and Software
Engineering, Shenzhen University, Shenzhen, China.
His research interests include multimedia systems
and applications, mobile cloud computing, edge
computing, social networking, and the Internet of Things. He was the recipient
of the C.D. Nelson Memorial Graduate Scholarship (2013) and Best Paper
Finalist at IEEE/ACM IWQoS (2016).
Junhui Hou (Senior member, IEEE) is an Asso-
ciate Professor with the Department of Computer
Science, City University of Hong Kong. He holds a
B.Eng. degree in information engineering (Talented
Students Program) from the South China Univer-
sity of Technology, Guangzhou, China (2009), an
M.Eng. degree in signal and information processing
from Northwestern Polytechnical University, Xi’an,
China (2012), and a Ph.D. degree from the School
of Electrical and Electronic Engineering, Nanyang
Technological University, Singapore (2016). His re-
search interests are multi-dimensional visual computing. Dr. Hou received the
Early Career Award (3/381) from the Hong Kong Research Grants Council in
2018 and the NSFC Excellent Young Scientists Fund in 2024. He has served
or is serving as an Associate Editor for IEEE Transactions on Visualization
and Computer Graphics, IEEE Transactions on Image Processing, IEEE
Transactions on Multimedia, and IEEE Transactions on Circuits and Systems
for Video Technology.
You Yang (M’10-SM’16) received the Ph.D. degree
in computer science from the Institute of Computing
Technology, Chinese Academy of Sciences, Beijing,
China, in 2009. He worked as a Postdoctoral Fellow
with the Automation Department, Tsinghua Univer-
sity, from 2009 to 2011. He was a Senior Research
Scientist in Sumavision Research from 2011 to 2013.
He currently heads the Department of Information
Engineering, Huazhong University of Science and
Technology, Wuhan, China. He has authored and
co-authored more than 60 technical papers and au-
thorized 17 patents. His research interests include three-dimensional (3D)
vision system and its applications, including the multi-view imaging system,
3D/VR/AR content processing and broadcasting, human-machine interaction
techniques, and interactive visual applications.
Dr. Yang has been an Academic Editor of the PLoS ONE since 2015,
the Guest Editor of Neurocomputing in 2014, the Committee/TPC Member
or Session Chair of over 30 international conferences, and Reviewers of
28 prestigious international journals from IEEE, ACM, OSA and other
associations. He was elected to be a Fellow of the Institute of Engineering
and Technology in 2018.
Xu Wang (M’15) received the B.S. degree from
South China Normal University, Guangzhou, China,
in 2007, and M.S. degree from Ningbo University,
Ningbo, China, in 2010. He received his Ph.D.
degree from the Department of Computer Science,
City University of Hong Kong in 2014. In 2015,
he joined the College of Computer Science and
Software Engineering, Shenzhen University, where
he is currently an Associate Professor. His research
interests are video coding and 3D vision.
