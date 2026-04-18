<!-- page 1 -->
GS-Share: Enabling High-fidelity Map Sharing with
Incremental Gaussian Splatting
Xinran Zhang1
Hanqi Zhu1
Yifan Duan1
Yanyong Zhang1,2†
1 University of Science and Technology of China, China
2 Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, China
Figure 1: Comparison between previous methods and our GS-Share. In map-sharing scenarios where input images arrive progressively,
previous methods typically reconstruct the entire map from scratch at each stage, requiring full transmission whenever a user updates their
map. In contrast, GS-Share adopts an incremental map update strategy that learns only the changes and transmits only the map increments,
effectively eliminating redundant data transfer. Beyond reducing transmission overhead, GS-Share also leverages virtual-view synthesis to
enhance mapping accuracy. Together, these techniques form a complete map-sharing framework tailored to the Gaussian representation.
Abstract
Constructing and sharing 3D maps is essential for many applications, including autonomous driving and augmented reality.
Recently, 3D Gaussian splatting has emerged as a promising approach for accurate 3D reconstruction. However, a practical
map-sharing system that features high-fidelity, continuous updates, and network efficiency remains elusive. To address these
challenges, we introduce GS-Share, a photorealistic map-sharing system with a compact representation. The core of GS-Share
includes anchor-based global map construction, virtual-image-based map enhancement, and incremental map update. We eval-
uate GS-Share against state-of-the-art methods, demonstrating that our system achieves higher fidelity, particularly for extrap-
olated views, with improvements of 11%, 22%, and 74% in PSNR, LPIPS, and Depth L1, respectively. Furthermore, GS-Share
is significantly more compact, reducing map transmission overhead by 36%.
CCS Concepts
• Computing methodologies →Rendering; Shape modeling; • Information systems →Data compression;
† Corresponding author.
1. Introduction
Exploring and mapping uncharted environments has long been an
enduring pursuit of humanity. Imagine a bustling shopping mall
submitted to Arxiv
arXiv:2510.02884v1  [cs.GR]  3 Oct 2025

<!-- page 2 -->
2 of 11
Zhang et al. / GS-Share
where a high-fidelity 3D map can seamlessly guide visitors to their
destinations, offering dense and detailed reconstructions that signif-
icantly enhance user experiences. To realize this vision, 3D Gaus-
sian Splatting (3DGS) [KKLD23] has emerged as one of the most
promising approaches. By explicitly modeling each element in the
map as a Gaussian ellipsoid, 3DGS achieves remarkable results.
However, utilizing 3DGS at the scene level, as opposed to the
object level, typically involves massive data requirements. Conse-
quently, crowd-sourcing is typically employed for data collection,
enabling dispersed participants to share their map data [CFK∗19],
which is managed through a map-sharing framework. Typically,
such a map-sharing framework involves two categories of partici-
pants [ZZD∗24]: contributors and users. Contributors are individu-
als who traverse previously uncharted areas and upload their obser-
vations to the server. In contrast, as shown in Fig. 1, users download
the map from the server and convert it into a more human-readable
format, such as rendering it into images.
Albeit inspiring, many viewpoints inevitably remain unobserved
even with crowd-sourcing. Exhaustively covering every possible
view would be prohibitively expensive in terms of time and re-
sources. This introduces a critical challenge: maintaining high vi-
sual quality from multiple viewpoints, many of which have never
been directly observed by contributors – a problem known as novel
view synthesis (NVS). NVS can be further categorized into two
groups based on the similarity between the input view and the novel
view: interpolation and extrapolation [YLZ∗23,HKK∗24]. Interpo-
lation focuses on synthesizing novel views that are similar to the
training views collected from contributors, while extrapolation ad-
dresses the synthesis of views that are significantly different from
training views, as shown in Fig. 2, and is more challenging. In map-
sharing systems, extrapolation is particularly common, as contribu-
tors often only partially cover the scene. A few studies have sought
to address this issue: some focus on novel view synthesis with
few-shot learning [BLZ∗24, ZFJW24, COL24, LZB∗24, SZLP24],
while others explore view generalization across different scenar-
ios [CLTS24,ZZL∗25,CXZ∗24,WRI∗24,SLL∗25]. Although these
works improve general NVS performance, they still suffer from
limited rendering quality and can only handle a small number of
input images. In parallel, some prior works have investigated view
extrapolation using implicit representations [YLZ∗23, YJZ∗23],
but these methods are not applicable to Gaussian splatting due to
their different representations. As a result, a high-fidelity Gaussian-
splatting-based map-sharing system remains elusive.
In addition to the need for view extrapolation, a practical map-
sharing system has another important feature: it must deliver maps
to users as efficiently as possible. Given that 3D Gaussian maps are
typically large – e.g., 0.3 GB for a small single room in the Replica
dataset [SWM∗19] – transmitting raw Gaussians can consume sig-
nificant traffic and reduce user experiences. The situation becomes
even more challenging when the system must incorporate pro-
gressively collected observations from contributors. Such a system
must continuously update the global map and efficiently distribute
these updates to users with minimal overhead. To address the
transmission bottleneck, several approaches have been developed,
including anchor-based methods [LYX∗24, CWL∗24b, RJL∗24,
CWL∗25], pruning-based methods [FWW∗24, NMR∗24, FW24,
ZSL∗24], and quantization-based methods [KBKK15, CWL∗24b,
Figure 2: Illustration of view interpolation and extrapolation. In-
terpolated views closely resemble the training views, whereas ex-
trapolated views differ significantly from them.
LRS∗24, NMKP23, FWW∗24, GGS24, WZH∗24]. However, these
methods are primarily designed for a single, static map, the chal-
lenge of continuously updating the global map and distributing it to
users remains largely unexplored.
Given the above challenges – specifically, how can we design
a practical framework to construct a high-fidelity Gaussian map
collected from contributors, and efficiently share the continuously
updating map with users? In this paper, we address this problem
by proposing GS-Share. Following the server-client architecture
used in Map++ [ZZD∗24], GS-Share maintains a global map on the
server, with each user retaining a local copy. Users can volunteer to
become contributors. As contributors explore their surroundings,
they upload images from various perspectives to the server, en-
abling the global map to be enhanced and shared continuously. To
facilitate such a pipeline, GS-Share employs a modular design that
accurately registers both contributors and users to the system, pre-
serves a compact yet expressive representation of the global map,
and enhances map quality through customized training strategies.
The proposed pipeline enables high-quality map sharing and effi-
cient distribution to users, fulfilling the core goals of GS-Share. To
further enhance the view extrapolation capability of the global map,
GS-Share constructs an auxiliary virtual map from the input data,
from which various virtual images are sampled. Each virtual image
is associated with predicted per-pixel confidence scores. Together,
the generated virtual images and confidence scores serve as pseudo
ground truth, augmenting the training data and ultimately improv-
ing map quality. Once the global map is constructed, GS-Share
supports on-demand map access: users upload their current view,
which the server matches against contributor images to localize the
user. Once matched, the corresponding map segment is transmitted.
For the initial transmission, a full map will be delivered. For contin-
uous map updates, GS-Share employs a lightweight representation
named a map increment. Instead of retraining and transmitting the
entire global map to replace the outdated map, GS-Share maintains
a database of staged maps and updates the map database by train-
ing and transmitting only the incremental changes. In this way, the
cached map on user devices can be effectively reused. By integrat-
ing the received increment with the cached data, users can recon-
struct the up-to-date global map without performance degradation.
We summarize our contributions as follows:
• To the best of our knowledge, GS-Share is the first Gaussian
map-sharing framework that supports both continuous map up-
dates and novel view synthesis. As more contributor data is in-
submitted to Arxiv

<!-- page 3 -->
Zhang et al. / GS-Share
3 of 11
tegrated, the quality of the shared map progressively improves
with minimal transmission overhead.
• To enable high-fidelity and compact map sharing, we develop a
complete and modular pipeline with customized representations
and training strategies. In addition to the pipeline, we also in-
corporate a set of key techniques, including virtual-image-based
map enhancement for improved novel view synthesis and incre-
mental map updates for low-overhead synchronization between
the global map and user-side local maps.
• We evaluate GS-Share on a benchmark built from the Replica
dataset [SWM∗19], and the results show that, compared to the
state-of-the-art method, GS-Share improves PSNR, LPIPS, and
Depth L1 by an average of 11%, 22%, and 74% on extrapolated
views, respectively, while also being more compact, reducing
map transmission overhead by 36%.
2. Related Work
2.1. Map Sharing
Several recent studies have explored map sharing among multi-
ple users. Map++ [ZZD∗24] adopts a server-client-based frame-
work, where a global map is constructed on the server, and
each user maintains a partial copy of the global map. SLAM-
Share [DRW∗22] offloads most of the SLAM computation to the
server, enabling users to efficiently leverage the shared map. Pair-
Navi [DXW∗19, XDM∗21] constructs a trajectory map on the
server, and assists subsequent users by reusing the experience of
previous travelers. RecNet [SSKN24] represents map data as range
images and latent vectors for efficient transmission. Google AR-
Core [Goo22] maintains a sparse anchor map on the server, en-
abling localization through anchor sharing. These methods primar-
ily focus on map sharing for localization or sparse reconstruction,
while GS-Share is designed for dense reconstruction with 3D Gaus-
sians.
2.2. Novel View Synthesis
Several recent efforts have been made in novel view syn-
thesis. For instance, LoopSparseGS [BLZ∗24] and DNGaus-
sian [LZB∗24] address the challenge of sparse observations by in-
troducing a depth-aware regularization loss and leveraging geomet-
ric cues. Some works focus on view generalization cross scenar-
ios [CLTS24, ZZL∗25, CXZ∗24, WRI∗24, SLL∗25, JMX∗25]. For
example, PixelSplat [CLTS24] estimates the parameters of a 3D
Gaussian for each scenario using a single image pair in a single
forward pass, while MVSplat [CXZ∗24] adopts a similar approach
but extends it to multi-view images. Although these works improve
general novel view synthesis performance, they still suffer from
limited rendering quality and can only handle a small number of
input images. For view extrapolation, NerfVS [YLZ∗23] proposes
leveraging depth priors and view coverage priors to guide opti-
mization in NeRF, while PARF [YJZ∗23] fuses semantic, primi-
tive, and radiance information into a single framework, enabling
high-fidelity and high-speed rendering. Although effective, these
methods are primarily designed for radiance fields, which are not
directly applicable to Gaussian splatting.
2.3. Compact 3D Gaussian Splatting
To compress the Gaussian map, several approaches have been de-
veloped, which can be classified into three categories: (1) Anchor-
based methods [LYX∗24,CWL∗24b,CWL∗25,RJL∗24,LWM∗24,
CWL∗24a, CLW∗25], which encodes high-dimensional Gaussian
attributes, such as spherical harmonics, positions, colors, and opac-
ities into low-dimensional anchor features; (2) Pruning [CWL∗24b,
FWW∗24, NMR∗24, FW24, ZSL∗24], which filters out less crit-
ical Gaussians based on importance scores. and (3) Quantiza-
tion [KBKK15,CWL∗24b,LRS∗24,NMKP23,FWW∗24,GGS24,
WZH∗24, CWL∗24a], which maps high-precision Gaussian pa-
rameters to lower-precision discrete levels to reduce storage and
bandwidth. Among these works, HAC [CWL∗24b] is most closely
related to ours, it incorporates all three techniques within a uni-
fied framework, guided by a context model that dynamically ad-
justs quantization step sizes. Although these methods effectively
reduce the size of 3D Gaussians, they are primarily designed for
single-user scenarios with static maps. A suitable representation for
continuously updated, multi-user map-sharing systems remains an
open challenge.
3. Preliminary on Anchor-based 3D Gaussian Splatting
Recently, 3D Gaussian Splatting (3DGS) [KKLD23] has gained
considerable attention as an effective solution for 3D reconstruc-
tion. It models the scene using a collection of anisotropic 3D Gaus-
sian ellipsoids, enabling high-quality rendering. A 3D Gaussian el-
lipsoid is formulated as:
G(x) = e−1
2 (x−µ)T Σ−1(x−µ),Σ = RSST RT ,
(1)
where x ∈R3 is a spatial point, µ ∈R3 is the Gaussian center, and
R, S denote rotation and scaling, respectively. Gaussians are then
rendered onto the image plane via α-blending:
C = ∑
i∈I
ciσi
i−1
∏
j=1
(1−σj),
(2)
where C is the rendered pixel color along a ray, I is the set of Gaus-
sians that intersect the ray. ci and σi denote the color and opac-
ity of the i-th Gaussian, respectively. To further compress 3DGS,
an anchor-based representation [CWL∗24b] encodes color, opac-
ity, and rotation into a low-dimensional feature embedding Femb.
Assuming each anchor is responsible for K nearby Gaussians, it
can be defined as follows:
Fanc = {Fs,Fo,Femb},
(3)
where Fs ∈RK×3 and Fo ∈RK×3 represent the scales and offsets
of K corresponding Gaussians, respectively, and Femb ∈R50 de-
notes the feature embedding. The compressed anchor Fanc can be
decoded back into 3D Gaussians:
G = Decode(Fanc),
(4)
where G represents the recovered Gaussians. A two-layer MLP is
employed to decode Femb back into Gaussian attributes, and Fo is
added to the anchor position V to restore Gaussians’ 3D positions.
submitted to Arxiv

<!-- page 4 -->
4 of 11
Zhang et al. / GS-Share
Figure 3: Overview of GS-Share. GS-Share takes RGB-D images as input and aligns them to a unified coordinate system using
COLMAP [SF16], followed by a global map construction process. Since the global map must be transmitted to users, its size is critical,
motivating a compact representation. To further enhance map quality, GS-Share generates an auxiliary virtual map and renders virtual im-
ages from it. After post-processing, these virtual images serve as pseudo ground truth to augment the training data. Once the global map is
reconstructed, users register with the server to access it. GS-Share transmits the full map during the initial stage and the map increments in
subsequent updates. By leveraging the received increments, users reconstruct the latest Gaussian map with minimal transmission overhead.
4. Method
This section details the design of GS-Share. As shown in Fig. 3,
GS-Share takes observations from contributors as input and con-
structs a global map and a virtual map. The design goals of GS-
Share are two-fold: first, the constructed map should be of high ac-
curacy; Second, it should be continuously updated and efficiently
distributed to users. To achieve these goals, GS-Share incorpo-
rates three modules: anchor-based global map construction, virtual-
image-based map enhancement, and incremental map update.
4.1. Anchor-based Global Map Construction
We first present a comprehensive pipeline for global map construc-
tion, designed to support high-fidelity, compact map sharing across
multiple contributors. To initiate this process, GS-Share collects
multi-view images from contributors who explore the environment
and upload their observations to the server. To align data from dif-
ferent contributors into a unified global coordinate system, GS-
Share employs COLMAP [SF16], a Structure-from-Motion (SfM)
module, to estimate camera poses from the collected images. Based
on these poses, a compact global map is constructed using an
anchor-based Gaussian representation. Specifically, GS-Share first
lifts all the input images into 3D space and generates colored point
clouds, which are then voxelized as follows:
V = {|Praw
ϵ
|}·ϵ,
(5)
where V represents the anchor positions, Praw is the point cloud
coordinate, and ϵ is the voxel resolution (e.g., 3cm in our case).
These anchors are then used to initialize the global map follow-
ing HAC [CWL∗24b]. However, naive 3D Gaussian splatting ex-
hibits significant multi-view inconsistency [HYC∗24]: rendering a
3D Gaussian ellipsoid from different perspectives yields different
results. We observed that such inconsistency can severely degrade
map quality, especially for view extrapolation. To resolve this, we
replace 3D ellipsoids with view-consistent 2D flats and incorporate
a normal loss, following [HYC∗24]. Experiments show that this ap-
proach improves map quality and reduces overall map size. While
beneficial, we further observe that this strategy alone is insuffi-
cient: inaccurate normal estimates near object boundaries can still
degrade view extrapolation, particularly when input data is insuffi-
cient. To mitigate this, GS-Share identifies regions with a depth L1
error exceeding a threshold (e.g., 0.1m) as unreliable and excludes
them from the normal loss computation, thereby further improving
the performance. Finally, the loss function for the training/input
views is defined as follows:
Lt = L1t
obs +Lt
SSIM +Lt
reg +Lt
d +Lt
nm +Lt
mask,
(6)
where L1t
obs and Lt
SSIM denote the L1 and SSIM loss [WBSS04]
between the rendered image and the ground truth image, respec-
tively. Ltreg denotes the regularization term for scale [LSS∗21], Lt
d
is the L1 loss between the rendered depth and ground truth depth,
Ltnm accounts for the normal loss [HYC∗24] after edge filtering,
Lt
mask refers to the learning-based Gaussian pruning loss [LRS∗24].
Please note that each loss term is associated with a weight, the
specifics of which are provided in the implementation section.
Once training completes, the global map becomes available for
use. During inference, GS-Share performs image-based registra-
tion [Bra00] by matching a user’s uploaded view against contribu-
tor images with known poses. Based on the best match, the system
estimates the user’s location and transmits the corresponding map
segment. Altogether, GS-Share offers a unified framework for con-
structing and sharing 3D Gaussian maps, laying the foundation for
subsequent refinement and incremental updates.
4.2. Virtual-image-based Map Enhancement
As map-sharing systems demand high reusability, they place
stricter requirements on novel view synthesis. To further enhance
the quality of the shared map, we introduce an auxiliary virtual map
to augment the global map. Similar to the global map, the virtual
map is constructed from the input observations and can be rendered
submitted to Arxiv

<!-- page 5 -->
Zhang et al. / GS-Share
5 of 11
Figure 4: The process of virtual-image-based map enhancement. It
uses Lpred as the loss for training the predictor and Lv for training
the global map. During the training of the global map, the param-
eters of the predictor are frozen.
into images. These rendered images act as pseudo ground truth to
guide the training of the global map. Although both maps are de-
rived from the same input, the virtual map is solely responsible for
enhancing the global map, allowing it to be explicitly optimized for
preserving fine-grained geometric cues from the raw sensor data. In
contrast, the global map – trained using the standard 3DGS pipeline
– tends to lose such detail due to overfitting to the input views. To
construct such virtual map, we first lift the RGB-D images into 3D
space to generate colored point clouds. These point clouds are then
converted into 3D Gaussians. The attributes of each Gaussian are
computed as follows: the position and color of each Gaussian are
directly derived from the point cloud; the scale is determined based
on the point cloud density [KKLD23]; the opacity is set to 1; and
all Gaussians are isotropic, with rotation not considered. We inten-
tionally avoid using 2D Gaussians here, as they require extensive
training data for reliable rotation estimation, whereas a properly
initialized isotropic 3DGS can generalize well to novel views even
without training. The resulting virtual map can then be rendered.
Since interpolated views closely resemble the input images and
require little augmentation, we render virtual images only for ex-
trapolated views to serve as pseudo ground truth. Specifically, af-
ter obtaining camera poses via COLMAP [SF16], virtual poses are
randomly sampled across the entire scene. We retain only the poses
that qualify as extrapolated views: those that are sufficiently differ-
ent from all input poses are kept, while any pose that is too close
(e.g., with a rotation difference less than 10 degrees) is discarded.
Based on the selected poses Pslct, virtual images Iv
virt are rendered,
where the superscript v indicates that the image is rendered from
the virtual map. The rendered virtual images Iv
virt, however, require
further post-processing to serve as reliable pseudo ground truth.
We devise two techniques to improve the quality of Iv
virt: hole fill-
ing and confidence prediction. We observe that when input data is
insufficient, certain regions are not properly covered by Gaussians,
resulting in visible holes. To address this, we render an additional
opacity image alongside Iv
virt to identify those poorly covered ar-
eas. Regions where the opacity falls below a given threshold are
marked as holes and filled using the Navier-Stokes inpainting algo-
rithm [BBS01]. Even after hole filling, the virtual image may still
be inaccurate. To account for this, we incorporate a confidence pre-
diction step to assess the reliability of each pixel and guide its use
in supervision. Those areas with a low confidence score will be as-
signed a smaller weight when supervising the global map. To train
the confidence predictor, we render calibration images Iv
calib from
the virtual map at the training view (input pose Pobs), where accu-
rate ground truth is available for supervision. The predictor takes
Iv
calib as input and predicts a dense per-pixel confidence map. The
loss function is defined as follows:
Lpred =
Pred(Iv
calib)−τ−L1v
calib
τ
,
(7)
where Lpred is the loss for training the predictor, Pred(Iv
calib) is the
estimated confidence on calibration image Iv
calib, L1v
calib denotes
the accuracy of Iv
calib, which is the L1 error between the calibration
image Iv
calib and the input image Iobs, and τ is the maximum L1v
calib
of all calibration images in the dataset. As shown in Fig. 4, the
predictor is built upon U-Net [RFB15], consisting of an encoder-
decoder structure with skip connections to preserve spatial details.
Once trained on calibration images from the training views, the
predictor is applied to virtual images Iv
virt rendered from extrapo-
lated viewpoints. The global map can be then enhanced using the
following auxiliary loss:
Lv = L1v
virt ·Pred(Iv
virt),
(8)
where L1v
virt denotes the L1 error between the virtual image Iv
virt
and the corresponding rendered image Ivirt from the global map,
and Pred(Iv
virt) is the predicted confidence map for Iv
virt. This confi-
dence map encourages the global map to focus on high-confidence
regions. The total loss for global map construction is thus:
Ltotal = Lt +Lv,
(9)
where Lt is the loss for input views, detailed in Eq. (6) and Lv is the
auxiliary loss for virtual views. The associated loss weights will be
provided in the implementation section.
4.3. Incremental Map Update
Next, we describe how the shared map is efficiently transmitted
to the user. Following Map++ [ZZD∗24], GS-Share categorizes the
global map into seen and unseen areas. For unseen areas, a full map
segment is initially transmitted. For seen areas, the map is progres-
sively refined as the system progresses. To support lightweight syn-
chronization and efficient updates, GS-Share proposes to update the
global map in the form of map increments. A map increment FInc
consists of the same components as Fanc in Eq. (3), but is explicitly
designed to be more compact: the feature embedding dimension in
FInc is reduced by half compared to Fanc. Moreover, the process of
translating map data into 3D Gaussians is also different, as illus-
trated in Fig. 5. Specifically, FInc is first quantized into FInc′, which
is then decoded into a Gaussian increment ∆GS:
∆GS = Decode(FInc
′),
(10)
where FInc′ is the quantized map increment, S denotes the stage ID
of the map update, which is initialized to 0 and incremented by one
for each stage. In GS-Share, each stage advances by incorporating
additional observations, specifically those collected from a single
contributor. Decode denotes the decoding operation, which follows
submitted to Arxiv

<!-- page 6 -->
6 of 11
Zhang et al. / GS-Share
Figure 5: The process of training and transmitting map data. (a)
and (b) represent the procedures without and with incremental
map update, respectively. AE refers to arithmetic coding [WNC87],
while AD denotes the corresponding decoding procedure.
the same design as described in Sec. 3. Finally, the decoded Gaus-
sian increment ∆GS is integrated with the outdated map GS-1 from
the previous stage to derive the updated map GS. The process is
formulated as follows:
GS = GS−1 +∆GS.
(11)
Here, the addition denotes element-wise operations across the cor-
responding attributes of each Gaussian.
Next, we describe the training procedure for the map increment
FInc, as illustrated in Fig. 5. To enable adaptive quantization, FInc
is initialized as an empty embedding, along with a learnable quan-
tization step st [CWL∗24b]. To allow differentiable training of st,
we follow the strategy proposed in [BMS∗18], adding noise during
training and applying rounding only at inference time. Specifically,
we inject uniform noise sampled from the interval [- 1
2st,+ 1
2st] into
each element of FInc, simulating the effect of quantization and get
FInc′ as an intermediate result. The quantized feature FInc′ is then
decoded into Gaussian maps and supervised using the following
rate-distortion loss:
Lupdt = λqLentropy +Ltotal,
(12)
where Lentropy is the bit rate estimated using the entropy model fol-
lowing HAC [CWL∗24b], λq is the corresponding loss weight, and
Ltotal denotes the distortion term defined in Eq. (9). After training,
we obtain both the optimized quantization step st and the trained
map increment FInc. To efficiently transmit FInc to users, we per-
form a series of post-training steps, collectively referred to as the
inference stage. Unlike the noise injection used during training,
quantization in the inference stage is performed by directly round-
ing each element of FInc as follows:
Fσ
Inc = {|FInc
st |}·st,
(13)
where Fσ
Inc denotes the final quantized map increment. This re-
sult is further compressed into a bitstream using arithmetic coding
(AE) [WNC87]. The compressed bitstream, along with the map in-
crement decoder, is then transmitted to users and translated into the
updated Gaussian map.
5. Experimental Evaluation
We first describe our experimental setup. We compare GS-
Share against two baselines: the default 3DGS [KKLD23] and
HAC [CWL∗24b], a state-of-the-art compression method. Other
methods evaluated in HAC [CWL∗24b] are not included, as their
performance has already been thoroughly analyzed. To ensure a
fair comparison, we implement a server-client-based map-sharing
framework following Map++ [ZZD∗24], and port both 3DGS and
HAC into this framework as baselines.
5.1. Implementation
We implement GS-Share in PyTorch [PGM∗19] and train it on
a single NVIDIA RTX 3090Ti GPU. The voxel size ϵ is set to
0.03m, with each anchor representing 10 Gaussians. We adopt the
SfM-based COLMAP [SF16] for pose estimation. Compared to the
SLAM-based method [YLGO23], COLMAP yields more accurate
poses due to its longer optimization time. For each stage, the pre-
dictor is trained for 3,000 iterations (approximately 15 minutes),
and the global map is trained for 12,000 iterations (around 30 min-
utes). Training stops once the map quality, measured by PSNR,
converges. The predictor follows an encoder-decoder architecture,
downsampling the image by a factor of 8 before restoring it to the
original resolution. During the training of the global map, we set
the loss weights to: λt
obs = 1, λt
SSIM = 0.05, λtreg = 0.1, λt
d = 1.0,
λtnm = 0.1, λt
mask = 0.005, λt = 1.0, λv = 0.1, and λq = 0.0025.
5.2. Replica-Share Dataset
We conduct our experiments on a custom dataset, Replica-
Share. Unlike existing object-centric datasets such as TUM RGB-
D [SEE∗12] or single-user datasets such as ScanNet [DCS∗17],
Replica-Share is specifically designed for map sharing and evalu-
ating view extrapolation. To the best of our knowledge, no publicly
available dataset exists that is tailored for this purpose. Replica-
Share is recorded using the Replica simulator [SWM∗19] and spans
eight diverse indoor scenes. For each scene, we collect 2,000 obser-
vations along the trajectories of three different contributors. To save
bandwidth, only one out of every 20 observations is uploaded to the
server, while all 2,000 observations are retained for evaluating the
view interpolation performance of the global map. For view ex-
trapolation evaluation, we generate additional ground-truth data by
uniformly sampling 100 positions per scene and pairing each po-
sition with 10 random rotations, producing 1,000 novel views per
scene, most of which are extrapolated.
5.3. Metrics
We evaluate PSNR, LPIPS, SSIM, and Depth L1 for both inter-
polated and extrapolated views under GS-Share, HAC, and 3DGS.
Additionally, we report the size of the data transmitted from the
server to the user. For seen areas, this includes the bitstream of
the map increment FInc and its corresponding decoder, as shown in
Fig. 3. For unseen areas, it includes the bitstream of the full map
Fanc, its corresponding decoder, and additional anchor positions V.
We calculate the size of the accumulated transmission data across
all stages as the map size. The compression ratio is then determined
by comparing this map size to that of the original 3DGS.
submitted to Arxiv

<!-- page 7 -->
Zhang et al. / GS-Share
7 of 11
Table 1: Quantitative results. We evaluated the performance of HAC [CWL∗24b], 3DGS [KKLD23], and GS-Share across 8 scenes in
Replica-Share on extrapolated views. The best results are highlighted in red .
Method
Metric
Office0
Office1
Office2
Office3
Office4
Room0
Room1
Room2
Average
HAC [CWL∗24b]
extrapolation
PSNR [dB]↑
31.91
33.43
27.36
26.82
28.95
26.10
27.64
25.79
28.50
LPIPS↓
0.121
0.117
0.161
0.134
0.148
0.184
0.179
0.159
0.150
SSIM ↑
0.953
0.970
0.929
0.941
0.953
0.898
0.918
0.924
0.936
Depth L1 [cm]↓
1.76
2.25
3.66
5.98
2.99
2.80
1.82
9.87
3.89
Size [MB]↓
3.15
1.47
4.33
4.23
3.98
5.81
4.54
2.64
3.76
3DGS [KKLD23]
extrapolation
PSNR [dB]↑
31.46
33.58
27.86
27.06
29.18
26.13
27.67
27.21
28.77
LPIPS ↓
0.093
0.088
0.132
0.115
0.120
0.164
0.154
0.124
0.124
SSIM ↑
0.948
0.970
0.926
0.935
0.949
0.897
0.911
0.935
0.934
Depth L1 [cm]↓
1.50
1.82
2.68
6.54
5.00
2.37
1.57
11.53
4.13
Size [MB] ↓
184.99
93.60
238.59
261.26
291.83
310.85
242.61
140.12
220.48
GS-Share
extrapolation
PSNR [dB]↑
33.66
34.90
31.60
30.53
32.95
28.83
29.18
30.69
31.54
LPIPS↓
0.103
0.107
0.102
0.092
0.110
0.145
0.159
0.113
0.117
SSIM↑
0.965
0.978
0.975
0.968
0.974
0.937
0.929
0.966
0.962
Depth L1 [cm]↓
0.76
0.65
0.51
1.60
1.14
1.08
0.85
1.40
1.00
Size [MB]↓
1.93
0.94
2.78
2.60
2.47
3.72
2.86
2.04
2.42
Table 2: Quantitative results. We evaluated the performance of HAC [CWL∗24b], 3DGS [KKLD23], and GS-Share across 8 scenes in
Replica-Share on interpolated views. The best and second-best results are highlighted in red and orange , respectively.
Method
Metric
Office0
Office1
Office2
Office3
Office4
Room0
Room1
Room2
Average
HAC [CWL∗24b]
interpolation
PSNR [dB]↑
38.63
39.92
34.49
34.77
35.19
31.62
35.23
35.34
35.65
LPIPS↓
0.091
0.083
0.123
0.070
0.101
0.094
0.112
0.089
0.095
SSIM↑
0.987
0.994
0.984
0.990
0.985
0.978
0.982
0.989
0.986
Depth L1 [cm]↓
0.48
0.31
0.46
0.88
0.79
0.76
0.43
0.97
0.64
3DGS [KKLD23]
interpolation
PSNR [dB]↑
40.76
42.01
36.14
36.67
37.48
33.33
37.44
37.33
37.64
LPIPS ↓
0.036
0.028
0.041
0.026
0.042
0.044
0.048
0.036
0.038
SSIM ↑
0.993
0.996
0.992
0.995
0.992
0.985
0.990
0.994
0.992
Depth L1 [cm]↓
0.35
0.22
0.28
0.52
0.49
0.62
0.25
0.68
0.43
GS-Share
interpolation
PSNR [dB]↑
38.54
39.73
34.70
34.98
35.97
31.60
34.96
35.77
35.78
LPIPS↓
0.082
0.078
0.095
0.057
0.081
0.082
0.105
0.075
0.082
SSIM↑
0.986
0.992
0.987
0.992
0.989
0.979
0.981
0.990
0.987
Depth L1 [cm]↓
0.49
0.33
0.42
0.93
0.68
0.77
0.37
1.06
0.63
5.4. Results
Overall Performance. We compare the performance of GS-Share,
HAC [CWL∗24b], and the original 3DGS [KKLD23] on both ex-
trapolated and interpolated views, as shown in Tab. 1 and Tab. 2,
with a particular focus on the performance of the third stage. For
HAC and 3DGS, the full map is transmitted across all three stages.
In contrast, GS-Share transmits the full map only for unseen areas,
and updates seen regions using compact map increment features
along with a corresponding decoder. The results reveal that GS-
Share achieves significantly higher fidelity and compression ratios
compared to the other methods across all scenes in view extrapo-
lation. Taking the scene Office2 as an example, GS-Share achieves
a PSNR that is 4.24 dB higher than HAC and 3.74 dB higher than
3DGS, representing improvements of 15% and 13%, respectively.
Additionally, the transmission overhead of GS-Share is reduced by
36% compared to HAC, with a compression ratio of 86 times rel-
ative to the original 3DGS. Across all eight scenarios, the PSNR,
LPIPS, SSIM, and Depth L1 of GS-Share improve by 11%, 22%,
2.8%, and 74% compared to the state-of-the-art method HAC, re-
spectively, while the transmission overhead is reduced by 36%. The
improvements in compression efficiency are attributed to two key
design choices. First, the initial global map is transmitted in an
implicit anchor-based representation, which is significantly more
compact than explicitly storing all Gaussians, as in 3DGS. Sec-
ond, for map updates, we transmit only map increment information,
effectively reusing the client-side cached map to eliminate redun-
dant data transfer, offering a more efficient update mechanism than
HAC. For interpolated views, we observed that GS-Share performs
similarly to HAC, while 3DGS excels. This is primarily because
both GS-Share and HAC employ compression, which inevitably
introduces some degradation, whereas 3DGS uses a full, uncom-
pressed representation. The high performance of 3DGS, however,
comes at the cost of a significantly larger map size. On extrapo-
lated views, GS-Share outperforms both baselines, which can be
attributed to the proposed virtual-image-based map enhancement
module. Since interpolated views are already accurate, we did not
observe significant improvements in those cases.
Performance across Different Compression Ratios. To illustrate
submitted to Arxiv

<!-- page 8 -->
8 of 11
Zhang et al. / GS-Share
Figure 6: Performance compar-
ison under different compression
ratios on extrapolated views.
Figure 7: Performance compari-
son of GS-Share across different
stages on extrapolated views.
Figure 8: Impact of fraction of
training views, evaluated on ex-
trapolated views.
Figure 9: Impact of Gaussian
representation, evaluated on ex-
trapolated views.
the impact of compression ratios of HAC and GS-Share, we con-
ducted experiments at different compression ratios on Replica-
Share Room0. As shown in Fig. 6, GS-Share maintains high fidelity
even at higher compression ratios, for instance, at a compression ra-
tio of 143 times, the PSNR is 28.25 dB. In contrast, for HAC, the
results indicate a significant degradation in the PSNR curve when
the compression ratio is approximately 82 times.
Performance across Different Stages. Fig. 7 illustrates the sys-
tem’s performance across different stages and compression ratios.
Experiments are conducted on the Replica-Share Room0. We con-
trol the compression ratio by setting λq from 0.0005 to 0.0205 in in-
crements of 0.002. Please note that in this experiment, the compres-
sion rate is computed on a per-stage basis to enable clearer com-
parison, rather than being aggregated across all stages. As shown
in Fig. 7, we observe that within the same stage, PSNR declines
rapidly as the compression ratio increases. For example, as the
compression ratio increases from 28 to 86 times in stage 0, from
69 to 214 times in stage 1, and from 81 to 457 times in stage 2, the
PSNR decreases by 0.92 dB, 1.01 dB, and 1.00 dB, respectively.
However, the highest achieved PSNR steadily increases from stage
0 to stage 2, indicating that additional observations contribute to
improved map quality. In addition to PSNR, the compression ratio
also increases with the stage ID, demonstrating the effectiveness of
the map increment.
Ablation Study of the Main Components. As shown in Tab. 3,
we conduct an ablation study at stage 2 on three main components:
virtual-image-based map enhancement (VIRT), modified anchor-
based representation (ANCR), and the representation of map in-
crement (INCR). All performances are reported for extrapolated
views. The “Base” setting refers to HAC. The results indicate that
the use of VIRT improves the PSNR by 2.07 dB over the base-
line, while LPIPS, SSIM, and Depth L1 also show improvements
of 0.016, 0.030, and 1.05 cm, respectively. Building upon VIRT, the
use of our modified anchor-based representation further improves
Table 3: Ablation study of three main components.
BaseVIRTANCRINCR PSNR
[dB]↑LPIPS↓SSIM↑Depth L1
[cm]↓
Size
[MB]↓
✓
26.10 0.184
0.898
2.80
5.81
✓
✓
28.17 0.168
0.928
1.75
6.06
✓
✓
✓
28.58 0.145
0.935
1.08
5.56
✓
✓
✓
✓
28.83 0.145
0.937
1.08
3.72
the PSNR by 0.41 dB, LPIPS by 0.023, and Depth L1 by 0.67 cm,
while also resulting in a slight map size reduction of 8.3%. Finally,
the introduction of INCR improves the PSNR by an additional 0.25
dB and, more importantly, reduces the map size from 5.56 MB to
3.72 MB, representing a 33% reduction. To further quantify the
effect of the number of input views, we conduct an extensive ex-
periment. Specifically, we control the fraction of training views –
ranging from 20× downsampling to using the full set – and show the
PSNR values for extrapolated views in Fig. 8. These results show
that GS-Share consistently improves performance on extrapolated
views under various settings, validating its effectiveness.
Ablation Study of Additional Techniques. Except for the main
components, we also conduct an ablation study on several addi-
tional techniques, including edge filtering discussed in Sec. 4.1,
hole filling, and confidence prediction discussed in Sec. 4.2. Re-
sults for stage 0 are reported to better elucidate the effects of these
modules. As shown in Tab. 4, the inclusion of edge filtering in-
creases the PSNR by 0.3 dB and reduces Depth L1 by 0.15 cm,
as normal estimation is closely related to the object’s geometry.
Additionally, both the hole filling and confidence prediction mod-
ules show improvements in PSNR, LPIPS, SSIM, and Depth L1,
as they are designed to enhance the quality of the pseudo ground
truth. Finally, all techniques have a negligible impact on the map
size, which is in line with expectations.
Ablation Study of Different Gaussian Representations. Next,
we examine the effect of Gaussian representation under varying
compression ratios. To ensure a fair comparison, these experiments
are conducted without virtual image enhancement or incremental
map updates. As shown in Fig. 9, the proposed ANCR module us-
ing modified 2D Gaussians outperforms its 3D Gaussian counter-
part in terms of PSNR. At a compression ratio of 98 times, the 2D
Gaussian variant achieves a PSNR of 26.55 dB, compared to 24.57
dB for the 3D Gaussian representation. Moreover, the performance
gap widens as the compression ratio increases, demonstrating that
Table 4: Ablation study of additional techniques.
Experiment
PSNR
[dB]↑LPIPS↓SSIM↑Depth L1
[cm]↓
Size
[MB]↓
GS-Share
27.30 0.155
0.917
1.28
2.35
w/o Edge Filtering
27.00 0.160
0.914
1.43
2.35
w/o Hole Filling
25.97 0.173
0.907
1.32
2.36
w/o Confidence Prediction 26.88 0.163
0.915
1.38
2.33
submitted to Arxiv

<!-- page 9 -->
Zhang et al. / GS-Share
9 of 11
Figure 10: Visualization results of GS-Share and HAC [CWL∗24b]. We present the ground truth alongside the rendering results of GS-Share
and HAC. The PSNR values are labeled in the bottom right corner of each image.
ANCR is better suited for view extrapolation under compact Gaus-
sian splatting constraints.
Visualization Results of GS-Share and HAC. We visualize the
qualitative results of HAC [CWL∗24b] and GS-Share on extrapo-
lated views at stage 2 in Fig. 10. Regions with significant differ-
ences are highlighted with orange boxes, and the PSNR is labeled
in the bottom right corner of each image. We observe that in areas
with insufficient observations, view extrapolation under HAC often
leads to several negative effects, such as artifacts, distortion, and in-
correctly rendered colors and textures. In contrast, the rendered im-
ages from GS-Share exhibit higher fidelity and substantially reduce
these issues, showcasing the effectiveness of the designed modules.
Visualization Results of GS-Share at Different Stages. We
present the visualization results of GS-Share across multiple stages
in Fig. 11. Regions with significant differences are highlighted with
orange boxes. As the stage progresses, the reconstruction quality
consistently improves. This is attributed to the increasing amount
of input data, which provides richer information for reconstruc-
tion. This highlights the importance of timely map updates in map-
sharing systems. The improvement is especially noticeable in areas
with fine textures, such as object boundaries and detailed structures
like window blinds.
6. Conclusion
In this work, we propose GS-Share, the first incremental map-
sharing system based on Gaussian splatting. GS-Share enables
high-fidelity rendering, efficient map sharing, and continuous map
updating with a compact representation. Our experiments demon-
submitted to Arxiv

<!-- page 10 -->
10 of 11
Zhang et al. / GS-Share
Figure 11: Visualization results of GS-Share at different stages. The PSNR values are labeled in the bottom right corner of each image.
strate that GS-Share outperforms state-of-the-art methods in both
view extrapolation and transmission overhead. Moving forward,
we will continue to explore more practical challenges that a real-
world map-sharing system may encounter, including the integration
of various types of sensors, each with differing qualities.
7. ACKNOWLEDGMENTS
This work was supported by the National Natural Science Founda-
tion of China (No. 62332016) and the Key Research Program of
Frontier Sciences, CAS (No. ZDBS-LY-JSC001).
References
[BBS01]
BERTALMIO M., BERTOZZI A. L., SAPIRO G.: Navier-stokes,
fluid dynamics, and image and video inpainting. In Proceedings of the
2001 IEEE Computer Society Conference on Computer Vision and Pat-
tern Recognition (CVPR) (2001), vol. 1, IEEE, pp. I–I.
[BLZ∗24]
BAO Z., LIAO G., ZHOU K., LIU K., LI Q., QIU G.:
Loopsparsegs: Loop based sparse-view friendly gaussian splatting. arXiv
preprint arXiv:2408.00254 (2024).
[BMS∗18]
BALLÉ J., MINNEN D., SINGH S., HWANG S. J., JOHN-
STON N.: Variational image compression with a scale hyperprior. arXiv
preprint arXiv:1802.01436 (2018).
[Bra00]
BRADSKI G.: The opencv library. Dr. Dobb’s Journal: Software
Tools for the Professional Programmer 25, 11 (2000), 120–123.
[CFK∗19]
CAPPONI A., FIANDRINO C., KANTARCI B., FOSCHINI L.,
KLIAZOVICH D., BOUVRY P.: A survey on mobile crowdsensing sys-
tems: Challenges, solutions, and opportunities. IEEE communications
surveys & tutorials 21, 3 (2019), 2419–2465.
[CLTS24]
CHARATAN D., LI S. L., TAGLIASACCHI A., SITZMANN V.:
pixelsplat: 3d gaussian splats from image pairs for scalable generalizable
3d reconstruction. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR) (2024), pp. 19457–19467.
[CLW∗25]
CHEN Y., LI M., WU Q., LIN W., HARANDI M., CAI J.:
Pcgs: Progressive compression of 3d gaussian splatting. arXiv preprint
arXiv:2503.08511 (2025).
[COL24]
CHUNG J., OH J., LEE K. M.: Depth-regularized optimiza-
tion for 3d gaussian splatting in few-shot images.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) (2024), pp. 811–820.
[CWL∗24a]
CHEN Y., WU Q., LI M., LIN W., HARANDI M., CAI
J.: Fast feedforward 3d gaussian splatting compression. arXiv preprint
arXiv:2410.08017 (2024).
[CWL∗24b]
CHEN Y., WU Q., LIN W., HARANDI M., CAI J.: Hac:
Hash-grid assisted context for 3d gaussian splatting compression.
In
European Conference on Computer Vision (ECCV) (2024), Springer,
pp. 422–438.
[CWL∗25]
CHEN Y., WU Q., LIN W., HARANDI M., CAI J.: Hac++:
Towards 100x compression of 3d gaussian splatting.
arXiv preprint
arXiv:2501.12255 (2025).
[CXZ∗24]
CHEN Y., XU H., ZHENG C., ZHUANG B., POLLEFEYS M.,
GEIGER A., CHAM T.-J., CAI J.: Mvsplat: Efficient 3d gaussian splat-
ting from sparse multi-view images. In European Conference on Com-
puter Vision (ECCV) (2024), Springer, pp. 370–386.
[DCS∗17]
DAI
A.,
CHANG
A.
X.,
SAVVA
M.,
HALBER
M.,
FUNKHOUSER T., NIESSNER M.: Scannet: Richly-annotated 3d recon-
structions of indoor scenes. In Proceedings of the IEEE conference on
computer vision and pattern recognition (CVPR) (2017), pp. 5828–5839.
[DRW∗22]
DHAKAL A., RAN X., WANG Y., CHEN J., RAMAKRISH-
NAN K.: Slam-share: visual simultaneous localization and mapping for
real-time multi-user augmented reality. In Proceedings of the 18th In-
ternational Conference on emerging Networking EXperiments and Tech-
nologies (CoNEXT) (2022), pp. 293–306.
[DXW∗19]
DONG E., XU J., WU C., LIU Y., YANG Z.:
Pair-navi:
Peer-to-peer indoor navigation with mobile visual slam. In IEEE INFO-
COM 2019-IEEE conference on computer communications (INFOCOM)
(2019), IEEE, pp. 1189–1197.
[FW24]
FANG G., WANG B.: Mini-splatting: Representing scenes with a
constrained number of gaussians. In European Conference on Computer
Vision (ECCV) (2024), Springer, pp. 165–181.
[FWW∗24]
FAN Z., WANG K., WEN K., ZHU Z., XU D., WANG Z.,
ET AL.: Lightgaussian: Unbounded 3d gaussian compression with 15x
reduction and 200+ fps. Advances in neural information processing sys-
tems (NeurIPS) 37 (2024), 140138–140158.
[GGS24]
GIRISH S., GUPTA K., SHRIVASTAVA A.: Eagles: Efficient
accelerated 3d gaussians with lightweight encodings. In European Con-
ference on Computer Vision (ECCV) (2024), Springer, pp. 54–71.
[Goo22]
GOOGLE:
Cloud Anchors allow different users to share
AR experiences.
https://developers.google.com/ar/develop/
cloud-anchors, 2022.
[HKK∗24]
HWANG S., KIM M.-J., KANG T., KANG J., CHOO J.: Vegs:
View extrapolation of urban scenes in 3d gaussian splatting using learned
priors. In European Conference on Computer Vision (ECCV) (2024),
Springer, pp. 1–18.
[HYC∗24]
HUANG B., YU Z., CHEN A., GEIGER A., GAO S.: 2d gaus-
sian splatting for geometrically accurate radiance fields. In ACM SIG-
GRAPH 2024 Conference Papers (2024), pp. 1–11.
submitted to Arxiv

<!-- page 11 -->
Zhang et al. / GS-Share
11 of 11
[JMX∗25]
JIANG L., MAO Y., XU L., LU T., REN K., JIN Y., XU X.,
YU M., PANG J., ZHAO F., ET AL.: Anysplat: Feed-forward 3d gaussian
splatting from unconstrained views. arXiv preprint arXiv:2505.23716
(2025).
[KBKK15]
KUMAR G., BRAR E. S. S., KUMAR R., KUMAR A.: A
review: Dwt-dct technique and arithmetic-huffman coding based image
compression. International Journal of Engineering and Manufacturing
5, 3 (2015), 20.
[KKLD23]
KERBL B., KOPANAS G., LEIMKÜHLER T., DRETTAKIS G.:
3d gaussian splatting for real-time radiance field rendering. ACM Trans.
Graph. (TOG) 42, 4 (2023), 139–1.
[LRS∗24]
LEE J. C., RHO D., SUN X., KO J. H., PARK E.:
Com-
pact 3d gaussian representation for radiance field.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) (2024), pp. 21719–21728.
[LSS∗21]
LOMBARDI S., SIMON T., SCHWARTZ G., ZOLLHOEFER M.,
SHEIKH Y., SARAGIH J.: Mixture of volumetric primitives for efficient
neural rendering. ACM Transactions on Graphics (TOG) 40, 4 (2021),
1–13.
[LWM∗24]
LI H., WU Y., MENG J., GAO Q., ZHANG Z., WANG
R., ZHANG J.:
Instancegaussian: Appearance-semantic joint gaus-
sian representation for 3d instance-level perception.
arXiv preprint
arXiv:2411.19235 (2024).
[LYX∗24]
LU T., YU M., XU L., XIANGLI Y., WANG L., LIN D., DAI
B.: Scaffold-gs: Structured 3d gaussians for view-adaptive rendering.
In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) (2024), pp. 20654–20664.
[LZB∗24]
LI J., ZHANG J., BAI X., ZHENG J., NING X., ZHOU J., GU
L.: Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with
global-local depth normalization. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition (CVPR) (2024),
pp. 20775–20785.
[NMKP23]
NAVANEET K., MEIBODI K. P., KOOHPAYEGANI S. A.,
PIRSIAVASH H.:
Compact3d: Compressing gaussian splat radiance
field models with vector quantization. arXiv preprint arXiv:2311.18159
(2023).
[NMR∗24]
NIEMEYER M., MANHARDT F., RAKOTOSAONA M.-J.,
OECHSLE M., DUCKWORTH D., GOSULA R., TATENO K., BATES J.,
KAESER D., TOMBARI F.: Radsplat: Radiance field-informed gaussian
splatting for robust real-time rendering with 900+ fps. arXiv preprint
arXiv:2403.13806 (2024).
[PGM∗19]
PASZKE A., GROSS S., MASSA F., LERER A., BRADBURY
J., CHANAN G., KILLEEN T., LIN Z., GIMELSHEIN N., ANTIGA L.,
ET AL.: Pytorch: An imperative style, high-performance deep learning
library. Advances in neural information processing systems (NeurIPS)
32 (2019).
[RFB15]
RONNEBERGER O., FISCHER P., BROX T.: U-net: Convolu-
tional networks for biomedical image segmentation.
In International
Conference on Medical image computing and computer-assisted inter-
vention (MICCAI) (2015), Springer, pp. 234–241.
[RJL∗24]
REN K., JIANG L., LU T., YU M., XU L., NI Z., DAI B.:
Octree-gs: Towards consistent real-time rendering with lod-structured 3d
gaussians. arXiv preprint arXiv:2403.17898 (2024).
[SEE∗12]
STURM J., ENGELHARD N., ENDRES F., BURGARD W.,
CREMERS D.: A benchmark for the evaluation of rgb-d slam systems.
In 2012 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS) (2012), pp. 573–580.
[SF16]
SCHÖNBERGER J. L., FRAHM J.-M.: Structure-from-motion re-
visited.
In Conference on Computer Vision and Pattern Recognition
(CVPR) (2016).
[SLL∗25]
SONG Q., LI C., LIN H., PENG S., HUANG R.: Adgaussian:
Generalizable gaussian splatting for autonomous driving with multi-
modal inputs. arXiv preprint arXiv:2504.00437 (2025).
[SSKN24]
STATHOULOPOULOS N., SAUCEDO M. A., KOVAL A.,
NIKOLAKOPOULOS G.:
Recnet: An invertible point cloud encoding
through range image embeddings for multi-robot map sharing and re-
construction. In 2024 IEEE International Conference on Robotics and
Automation (ICRA) (2024), IEEE, pp. 4883–4889.
[SWM∗19]
STRAUB J., WHELAN T., MA L., CHEN Y., WIJMANS E.,
GREEN S., ENGEL J. J., MUR-ARTAL R., REN C., VERMA S., ET AL.:
The replica dataset: A digital replica of indoor spaces. arXiv preprint
arXiv:1906.05797 (2019).
[SZLP24]
SMART B., ZHENG C., LAINA I., PRISACARIU V. A.:
Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs.
arXiv preprint arXiv:2408.13912 (2024).
[WBSS04]
WANG Z., BOVIK A. C., SHEIKH H. R., SIMONCELLI E. P.:
Image quality assessment: from error visibility to structural similarity.
IEEE transactions on image processing (TIP) 13, 4 (2004), 600–612.
[WNC87]
WITTEN I. H., NEAL R. M., CLEARY J. G.: Arithmetic cod-
ing for data compression. Communications of the ACM 30, 6 (1987),
520–540.
[WRI∗24]
WEWER C., RAJ K., ILG E., SCHIELE B., LENSSEN J. E.:
latentsplat: Autoencoding variational gaussians for fast generalizable 3d
reconstruction. In European Conference on Computer Vision (ECCV)
(2024), Springer, pp. 456–473.
[WZH∗24]
WANG H., ZHU H., HE T., FENG R., DENG J., BIAN J.,
CHEN Z.: End-to-end rate-distortion optimized 3d gaussian represen-
tation. In European Conference on Computer Vision (ECCV) (2024),
Springer, pp. 76–92.
[XDM∗21]
XU J., DONG E., MA Q., WU C., YANG Z.: Smartphone-
based indoor visual navigation with leader-follower mode. ACM Trans-
actions on Sensor Networks (TOSN) 17, 2 (2021), 1–22.
[YJZ∗23]
YING H., JIANG B., ZHANG J., XU D., YU T., DAI Q., FANG
L.: Parf: Primitive-aware radiance fusion for indoor scene novel view
synthesis. In Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV) (2023), pp. 17706–17716.
[YLGO23]
YUGAY V., LI Y., GEVERS T., OSWALD M. R.: Gaussian-
slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint
arXiv:2312.10070 (2023).
[YLZ∗23]
YANG C., LI P., ZHOU Z., YUAN S., LIU B., YANG X.,
QIU W., SHEN W.: Nerfvs: Neural radiance fields for free view syn-
thesis via geometry scaffolds. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR) (June 2023),
pp. 16549–16558.
[ZFJW24]
ZHU Z., FAN Z., JIANG Y., WANG Z.: Fsgs: Real-time few-
shot view synthesis using gaussian splatting. In European conference on
computer vision (ECCV) (2024), Springer, pp. 145–163.
[ZSL∗24]
ZHANG Z., SONG T., LEE Y., YANG L., PENG C., CHEL-
LAPPA R., FAN D.: Lp-3dgs: Learning to prune 3d gaussian splatting.
arXiv preprint arXiv:2405.18784 (2024).
[ZZD∗24]
ZHANG X., ZHU H., DUAN Y., ZHANG W., SHANGGUAN
L., ZHANG Y., JI J., ZHANG Y.: Map++: Towards user-participatory
visual slam systems with efficient map expansion and sharing. In Pro-
ceedings of the 30th Annual International Conference on Mobile Com-
puting and Networking (MobiCom) (2024), pp. 633–647.
[ZZL∗25]
ZHANG C., ZOU Y., LI Z., YI M., WANG H.: Transplat:
Generalizable 3d gaussian splatting from sparse multi-view images with
transformers. In Proceedings of the AAAI Conference on Artificial Intel-
ligence (AAAI) (2025), vol. 39, pp. 9869–9877.
submitted to Arxiv
