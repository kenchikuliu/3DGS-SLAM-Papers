<!-- page 1 -->
TokenSplat: Token-aligned 3D Gaussian Splatting for Feed-forward
Pose-free Reconstruction
Yihui Li1,2, Chengxin Lv1,2, Zichen Tang1,3, Hongyu Yang3*, Di Huang1,2
1 State Key Laboratory of Complex and Critical Software Environment, Beijing, China
2 School of Computer Science and Engineering, Beihang University, China
3 School of Artificial Intelligence, Beihang University, China
{kidleyh, chengxinlv, zctang, hongyuyang, dhuang}@buaa.edu.cn
Abstract
We present TokenSplat, a feed-forward framework for joint
3D Gaussian reconstruction and camera pose estimation
from unposed multi-view images. At its core, TokenSplat
introduces a Token-aligned Gaussian Prediction module
that aligns semantically corresponding information across
views directly in the feature space. Guided by coarse to-
ken positions and fusion confidence, it aggregates multi-
scale contextual features to enable long-range cross-view
reasoning and reduce redundancy from overlapping Gaus-
sians. To further enhance pose robustness and disentangle
viewpoint cues from scene semantics, TokenSplat employs
learnable camera tokens and an Asymmetric Dual-Flow
Decoder (ADF-Decoder) that enforces directionally con-
strained communication between camera and image tokens.
This maintains clean factorization within a feed-forward
architecture, enabling coherent reconstruction and stable
pose estimation without iterative refinement. Extensive ex-
periments demonstrate that TokenSplat achieves higher re-
construction fidelity and novel-view synthesis quality in
pose-free settings, and significantly improves pose estima-
tion accuracy compared to prior pose-free methods. Project
page: https://kidleyh.github.io/tokensplat/.
1. Introduction
3D Gaussian Splatting (3DGS) [19] has recently emerged
as an efficient alternative to neural radiance fields (NeRFs)
[26], enabling high-quality rendering with significantly
faster performance.
Despite this progress, most existing
3DGS-based reconstruction pipelines [5, 21, 23, 25, 39, 45]
rely on per-scene optimization, which restricts their scala-
bility and limits generalization across diverse scenes. Re-
cent feed-forward variants [1, 4, 17, 40, 50] alleviate this
by predicting 3D Gaussians directly from input images, but
their applicability remains constrained by the requirement
for accurate camera poses. Pose estimation via structure-
from-motion (SfM) [29] is computationally expensive and
prone to failure in challenging environments, significantly
impacting reconstruction stability.
To reduce dependence on external poses, several ap-
proaches [3, 12, 31] jointly optimize camera poses and re-
construction through alternating pipelines. While effective
in alleviating strict pose requirements, these methods still
struggle to maintain reconstruction fidelity when many or
densely captured images are involved. More recent pose-
free frameworks [14, 30, 35, 44] attempt to infer both cam-
era poses and 3D structure directly from sparse unposed im-
ages. However, these approaches typically encode scene in-
formation and viewpoint cues within the same feature em-
beddings, leading to representation entanglement that cam-
era parameters become difficult to disentangle from scene
content, and errors in pose estimation propagate to 3D re-
construction.
Furthermore, most existing pose-free methods rely on
pixel-aligned 3DGS heads that generate Gaussians at pixel-
level granularity. As the number of views increases, this
pixel-centric prediction induces redundant and overlapping
Gaussians, causing geometric blurring, color inconsistency,
and reduced scene fidelity. Although some works [15, 37]
attempt multi-view 3DGS attribute fusion to suppress re-
dundancy, their local aggregation scope limits their ability
to form globally coherent structure, often resulting in frag-
mented or unstable reconstructions.
To address these challenges, we propose TokenSplat, a
feed-forward 3D Gaussian splatting framework that recon-
structs 3D scenes from an arbitrary number of unposed im-
ages while jointly estimating camera poses.
At its core
lies the Token-aligned Gaussian Prediction module, which
aligns semantically corresponding information across view-
points directly in the feature space. Guided by coarse token
positions and fusion confidence, this module adaptively ag-
gregates multi-scale contextual features across views, en-
arXiv:2603.00697v1  [cs.CV]  28 Feb 2026

<!-- page 2 -->
abling long-range cross-view reasoning and reducing re-
dundancy from overlapping Gaussians.
The aggregated
tokens are subsequently decoded by a 3D Gaussian pre-
diction head, which maps each token to multiple Gaus-
sians. This one-to-many mapping decouples Gaussian den-
sity from pixel resolution, yielding denser and more expres-
sive splats while preserving both structural integrity and se-
mantic coherence.
To jointly optimize 3D reconstruction and camera pose
estimation within a feed-forward architecture, we intro-
duce learnable camera tokens and an Asymmetric Dual-
Flow Decoder (ADF-Decoder) that regulates the interaction
between pose reasoning and scene reconstruction. Without
iterative refinement loops traditionally used to disentangle
pose and geometry, maintaining clean factorization between
camera parameters and scene features becomes crucial. Un-
like symmetric attention mechanisms that allow bidirec-
tional mixing and risk entangling viewpoint-specific cues
with scene semantics, the ADF-Decoder imposes direction-
ally constrained information flow between camera tokens
and image tokens. Camera tokens extract robust geometric
cues from image tokens to support accurate pose prediction,
while only stabilized low-frequency pose alignment signals
are propagated back to image tokens. Through disentan-
gled yet mutually supportive representations, TokenSplat
achieves stronger cross-view alignment, more stable pose
estimation, and higher-fidelity reconstruction.
In summary, our main contributions are as follows:
• We propose TokenSplat, a feed-forward pose-free recon-
struction framework that jointly estimates camera poses
and 3D Gaussian scenes from unposed multi-view im-
ages, exhibiting strong generalization.
• We introduce Token-aligned Gaussian Prediction module
that enables relatively long-range multi-view feature ag-
gregation and high-quality Gaussian generation with re-
duced redundancy and fragmentation.
• We design an Asymmetric Dual-Flow Decoder that dis-
entangles pose reasoning from scene encoding while en-
abling mutual reinforcement, leading to more accurate
pose estimation and improved reconstruction fidelity.
2. Related Work
2.1. Feed-forward 3D Reconstruction
Neural radiance fields (NeRF) [26] and 3D Gaussian Splat-
ting (3DGS) [19] have achieved impressive results in novel
view synthesis.
However, their scene-specific optimiza-
tion restricts scalability and generalization across multi-
ple scenes.
To overcome this limitation, recent works
[1, 2, 4, 32, 33, 41, 43, 47] have explored feed-forward ap-
proaches. These methods can reconstruct scenes without
per-scene optimization and generalize to unseen environ-
ments. Some works [37, 42] further improve model effi-
ciency and capability by incorporating lightweight cost vol-
ume modules or introducing depth priors. However, most
of these methods are still only applicable under sparse-view
conditions, and their broader applicability remains limited
due to the requirement for precise image poses as input.
2.2. Pose-Free 3D Scene Reconstruction
Feed-forward methods that rely on camera parameters [4,
36, 46] typically obtain them via structure-from-motion
(SfM) tools such as COLMAP [29], a process that can be
complex and error-prone. Some approaches jointly opti-
mize camera poses and scene representations, but they often
require inefficient pipelines. For instance, PF3Splat [11]
relies on off-the-shelf feature descriptors (LightGlue [24])
and RANSAC-based [9] pose initialization, and SelfSplat
uses cross-view U-Nets [28] to predict depth and pose sep-
arately. More recently, pose-free methods [13, 14, 35, 44]
reconstruct scenes directly in a canonical space without ex-
plicit pose inputs, leveraging point or feature matching.
While these methods improve robustness with sparse in-
puts, they still rely on primitive pixel-aligned 3D Gaussians,
which limits multi-view reconstruction fidelity and pose es-
timation accuracy.
To reduce redundancy and improve reconstruction con-
sistency, FreeSplat [37] maintains global Gaussians and
continuously fuses new view-specific Gaussians, while
AnySplat [15] follows the VGGT [34] paradigm and di-
rectly performs voxel fusion on the predicted Gaussians.
However, their purely local fusion of 3DGS attributes in
3D space lacks broader contextual reasoning, often result-
ing in fragmented or inconsistent scenes. In contrast, our
method performs multi-view feature aggregation at the to-
ken level, integrating semantically corresponding features
across views based on coarse token positions and confi-
dence scores.
This alleviates redundancy in overlapping
regions, and produces structurally coherent and complete
reconstructions.
3. Method
3.1. Problem Formulation
We aim to learn a feed-forward network that jointly re-
constructs 3D Gaussians and predicts camera poses from
a sequence of N unposed images {Ii}N
i=1, where Ii ∈
RH×W ×3. For 3D reconstruction, we predict a set of 3D
Gaussians in a canonical 3D space:
{(µg, σg, rg, sg, cg)}G
g=1 ,
(1)
where each Gaussian is parameterized by its center µg ∈
R3, opacity σg ∈R+, rotation quaternion rg ∈R4, scale
sg ∈R3, and spherical harmonics (SH) coefficients cg ∈
R3×(k+1)2 of degree k, following [18]. For camera pose es-
timation, the network predicts per-view poses Pi that trans-

<!-- page 3 -->
Figure 1. Overview of TokenSplat. TokenSplat performs feed-forward 3D Gaussian reconstruction and camera pose estimation from
unposed images. A shared ViT encoder extracts image tokens, which are processed by the Canonical Scene Decoder and the Asymmetric
Dual-Flow Decoder (ADF-Decoder). The fused tokens are then used by the Token-aligned Gaussian Prediction module and the camera
pose head to generate dense 3D Gaussians and accurate poses.
form each input image Ii into the canonical reference view
I1. Formally, the model learns a mapping:
fθ : {Ii}N
i=1 7−→{(µg, σg, rg, sg, cg)}G
g=1 ∪{Pi}N
i=1. (2)
During training, both the 3D Gaussian attributes and
camera poses are jointly optimized.
3.2. Architecture
As illustrated in Fig. 1, our method is an Transformer-based
architecture for feed-forward 3D reconstruction from un-
posed images. Each input view is first encoded into image
tokens via a shared ViT Encoder. The reference view I1
is processed by a Canonical Scene Decoder, which uses
cross-attention to integrate information from other views
and establish a canonical scene representation.
The re-
maining views are processed by the Asymmetric Dual-Flow
Decoder (ADF-Decoder), which refines both image and
learnable camera tokens through directionally constrained
communication, effectively disentangling camera parame-
ters from image features. The outputs of these decoders are
then utilized in two parallel branches: the Camera Pose Es-
timation Head predicts per-view camera transformations,
while the Token-aligned Gaussian Prediction module ag-
gregates multi-view tokens in the feature space and gener-
ates dense, coherent 3D Gaussians through its prediction
head. This unified feed-forward design enables consistent
multi-view reconstruction and robust pose estimation with-
out any iterative refinement.
ViT Encoder. For each input view, the RGB image is di-
vided into patches and embedded as a sequence of image
tokens. Optionally, following [44], camera intrinsics are en-
coded as an additional token via a linear layer and concate-
nated with the image tokens along the spatial dimension to
mitigate scale ambiguity. Each view’s tokens are indepen-
dently processed by a weight-shared ViT encoder, without
cross-view interaction.
Canonical Scene Decoder. To establish a canonical scene
representation, the reference view I1 is decoded using a ViT
decoder with cross-attention to other views. This ensures
that multi-view context is efficiently captured for the refer-
ence view and provides a consistent basis for reconstructing
the 3D scene. The canonical representation facilitates the
subsequent refinement of camera and scene features for the
remaining views.
3.3. Asymmetric Dual-Flow Decoder
The ADF-Decoder is designed to jointly refine image and
camera representations while explicitly disentangling scene
content from camera parameters.
To prevent entangling
viewpoint-specific cues with scene semantics, it employs
an asymmetric update scheme that image tokens primarily
aggregate scene context, whereas camera tokens extract ge-
ometric cues from image tokens and propagate only stabi-
lized pose signals back. This design enables accurate pose
estimation and high-fidelity 3D reconstruction in a single
feed-forward pass.
Token Initialization. Image tokens are generated by the
shared ViT encoder.
Camera tokens are initialized from
learnable embeddings and duplicated for the N −1 non-
reference views to provide per-view pose representations.
At each decoder layer, camera tokens interact with the cor-
responding image tokens through directionally constrained
attention, allowing them to extract view-specific geometric
cues while preventing unnecessary feature mixing.
Self-Attention. The ADF-Decoder consists of 12 decoder
blocks. Within each block, self-attention is applied sepa-

<!-- page 4 -->
rately to image and camera tokens. Image tokens integrate
intra-view context to capture local scene structures and en-
hance reconstruction. Camera tokens attend to their corre-
sponding image tokens to extract geometric cues for pose
estimation. Formally, for view i, let ˆtI
i denote the image to-
kens and ˆtc
i denote the camera token. They are projected to
queries, keys, and values QI
i (Qc
i), KI
i , and V I
i . The updates
are computed as:
ˆtI
i ←Softmax

QI
iKI
i
⊤/
√
d

V I
i ,
(3)
ˆtc
i ←Softmax

Qc
iKI
i
⊤/
√
d

V I
i ,
(4)
where d is the corresponding dimensionality of the queries
and keys. By updating camera tokens through their inter-
action with image tokens of the same view, the decoder ex-
tracts robust geometric features to support pose prediction.
Image Token Cross-View Attention. To capture global
multi-view context and enforce inter-view consistency, each
view performs cross-attention over tokens from other views
while explicitly avoiding interactions with its own tokens.
This prevents information leakage across views and en-
sures complementary cues are integrated from the rest of
the scene. For the image tokens of view Ii, the keys and
values from other views are concatenated along the spatial
dimension:
[KI
j]j̸=i = KI
1|| · · · ||KI
i−1||KI
i+1|| · · · ||KI
N,
(5)
[V I
j ]j̸=i = V I
1 || · · · ||V I
i−1||V I
i+1|| · · · ||V I
N.
(6)
The image tokens are then updated via cross-attention:
ˆtI
i ←Softmax

QI
i[KI
j]j̸=i
⊤/
√
d

[V I
j ]j̸=i.
(7)
To reduce computation with many views, a hyperparameter
pnv limits each view’s attention to its pnv −1 neighbors,
balancing context and efficiency.
Camera Token Cross-Attention. Updating camera tokens
solely via other camera tokens or solely via image tokens
is insufficient to capture global geometry. Therefore, each
camera token interacts with both image and camera tokens
from other views, allowing it to aggregate multi-view geo-
metric cues for more accurate camera pose estimation.
For view Ii, let ˆtc
i be its camera token. All camera to-
kens are projected to obtain Kc and V c. Each ˆtc
j for j ̸= i
is replicated to match the number of image tokens of other
views, and then summed with the image token keys and val-
ues to form the cross-attention inputs:
Kc
crossi = [KI
j]j̸=i + [repeat(Kc
j)]j̸=i,
(8)
V c
crossi = [V I
j ]j̸=i + [repeat(V c
j )]j̸=i.
(9)
The camera token is updated as:
ˆtc
i ←Softmax

Qc
iKc
crossi
⊤/
√
d

V c
crossi.
(10)
Pre- and Post-Modulation. Due to differences in token
counts and information content between image and cam-
era tokens, image tokens are modulated with corresponding
camera tokens both before and after attention. This stabi-
lizes updates, reinforces disentanglement between pose and
scene features, and ensures that pose cues guide reconstruc-
tion without contaminating scene semantics.
3.4. Token Fusion for Scene Reconstruction
The Token-aligned Gaussian Prediction module aggre-
gates multi-view features and predicts dense 3D Gaussians
for scene reconstruction. It comprises a Token Fusion op-
eration and a Gaussian Prediction Head.
Unlike tradi-
tional pixel-aligned Gaussian methods, which directly pre-
dict dense Gaussians for each pixel and suffer from redun-
dancy and geometric blurring in dense multi-view setups,
our method fuses features across tokens to produce more
consistent and efficient 3D Gaussians.
Token Fusion. The token fusion operation first predicts
coarse positions and fusion confidence for each token. To-
kens are then grouped based on spatial proximity with a spa-
tial grouping size ϵ, and fused using softmax-normalized
confidence scores. This produces a set of merged tokens
with consolidated features and coarse positions. The fusion
network adopts a DPT-based architecture [27], similar to
[44], with the number of predicted parameters adjusted.
Gaussian Prediction Head.
For each fused token, the
Gaussian prediction heads generate denser Gaussians, in-
cluding positional offsets relative to the token and Gaussian
attributes. Multi-scale features {Fi}nl
i=1 from different lay-
ers of the Transformer decoder corresponding to the fused
tokens are first upsampled and linearly projected:
ˆFi = Proji(Fi),
i = 1, . . . , nl.
(11)
Channel dimensions are unified, and the features are upsam-
pled according to layer scale (larger upsampling for lower
layers, smaller for higher layers). Features are then progres-
sively fused from deep to shallow layers using the residual
fusion module RF composed of residual blocks and upsam-
pling:
F fusion
nl
= RFnl( ˆFnl),
(12)
F fusion
i
= RFi( ˆFi, F fusion
i+1
).
(13)
Residual blocks enhance feature expressiveness, while up-
sampling spatially aligns semantic and detailed informa-
tion. The resulting fused features combine fine-grained de-
tails with rich semantic context.
Finally, the fused high-resolution features are passed
through a prediction block consisting of upsampling and
linear layers, producing continuous tensors representing the
final Gaussian predictions. For full architectural details of
the token-aligned Gaussian prediction heads, please refer to
the appendix.

<!-- page 5 -->
3.5. Camera Pose Estimation Head
The camera pose estimation head predicts the extrinsic pa-
rameters for each input view (excluding the reference view
I1). It takes the per-view camera tokens produced by the de-
coder as input and applies a linear projection to regress the
camera transformation P for each view. Jointly optimiz-
ing camera poses alongside the token-aligned Gaussian re-
construction encourages the network to learn geometrically
consistent 3D features, which improves both pose estima-
tion and scene reconstruction accuracy.
3.6. Loss Functions
Image Rendering Loss. We supervise the model using the
ground-truth target RGB images. The rendering loss com-
bines an L2 loss and a perceptual LPIPS loss [48]:
Lrender = L2(I,ˆI) + λlpipsLlpips(I,ˆI),
(14)
where I and ˆI denote the ground-truth and rendered images,
and λlpips balances the perceptual term.
Camera Pose Loss. The camera pose head predicts each
view’s extrinsic parameters relative to the reference view I1.
We supervise the predictions with a combination of MSE
and Unit Dual Quaternion (DQ) alignment loss [6, 8, 16],
which jointly represents rotation and translation to avoid in-
consistencies from separate predictions. The alignment loss
is defined as:
Lalign = ∥pI −pˆp∗∥+ ∥pI −ˆpp∗∥,
(15)
where p and ˆp are the predicted and ground-truth unit DQs,
and p∗denotes the conjugate. The overall camera pose loss
is:
Lpose = LMSE(P, ˆP) + Lalign.
(16)
Total Loss. The model is trained end-to-end with the total
loss:
L = Lrender + λcLpose,
(17)
where λc balances the contribution of the camera pose su-
pervision.
4. Experiment
We evaluate our method on novel view synthesis (NVS) and
camera pose estimation across sparse and long-sequence
real-world datasets.
4.1. Experimental Settings
Datasets.
We conduct experiments on ScanNet [7] and
RealEstate10K (RE10K) [49]. For ScanNet, we adopt the
training/testing splits from [37], and for RE10K we follow
[44]. Both settings ensure a direct and consistent compari-
son with prior work.
Baselines and Metrics. We evaluate NVS performance us-
ing PSNR, SSIM [38], and LPIPS [48]. For camera pose es-
timation, we report Absolute Translation Error (ATE), Rel-
ative Translation Error (RPE-t), and Relative Rotation Error
(RPE-r). We compare against generalizable NVS methods,
including (1) Pose-required: MVSplat [4] and FreeSplat
[37]; and (2) Pose-free: NoPoSplat [44], VicaSpalt [22],
SPFSplat [13], and AnySplat [15]. Pose estimation com-
parisons are conducted only with pose-free methods.
Implementation Details. Our method is implemented in
PyTorch.
The encoder follows [44] and adopts a ViT-
Large backbone with a patch size of 16. We initialize the
encoder-decoder and Gaussian center head with MASt3R
[20] weights, while the ADF-Decoder and remaining heads
are randomly initialized.
Following [22], we train and evaluate on RE10K un-
der both 4-view and 8-view reference settings, and further
perform cross-dataset generalization tests on ScanNet. For
ScanNet, we adopt the 3-view and 10-view settings from
[37], where the 10-view configuration is obtained by aug-
menting the 3-view setup with additional reference views.
To stress-test multi-view scalability on longer sequences,
we also design a 28-view evaluation on ScanNet.
For fair comparisons, we follow [4, 13, 22] and report
all quantitative results at a fixed resolution of 256 × 256.
Additional details are provided in the appendix.
4.2. Experimental Results
Performance on Multi-view NVS. The results are shown
on the left side of Tab. 1 and Tab. 2. Here, AnySplat refers to
zero-shot results trained on other datasets, while AnySplat∗
denotes the results we achieved after fine-tuning on the cor-
responding dataset.
As can be seen, TokenSplat consis-
tently outperforms state-of-the-art pose-free methods, in-
cluding those specifically designed for multi-view input
such as VicaSplat and AnySplat, which leverage cross-
neighbor attention and pixel-aligned Gaussian fusion, re-
spectively. In dense-view scenarios, TokenSplat even sur-
passes pose-dependent methods like FreeSplat, achieving
0.95 dB higher PSNR on RE10K with 8 views.
For the 28-view evaluation, we use models trained with
10 views. Despite the difference in view counts, Token-
Splat maintains stable reconstruction quality, while compet-
ing methods, including AnySplat, which fuses pixel-aligned
Gaussians by predicting fusion confidence, and FreeSplat,
show degradation. We attribute this to the Gaussian-level
fusion strategy used in these methods, which accumulates
inconsistencies and redundancies as more views are incor-
porated. In contrast, TokenSplat performs fusion in feature
space, enhancing cross-view reasoning and reducing redun-
dancy in long-sequence, high-density scenarios. Moreover,
as the number of input images increases, our model achieves
a higher SSIM of 0.061 over FreeSplat, while also show-

<!-- page 6 -->
Table 1. Quantitative results of NVS on RE10K with varying reference views (left) and cross-dataset generalization to ScanNet (right).
The best and second-best values are highlighted.
Method
RE10K (4 views)
RE10K (8 views)
ScanNet (4 views)
ScanNet (8 views)
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
Pose-
required
MVSplat
23.82
0.792
0.201
23.96
0.802
0.164
24.48
0.854
0.225
23.11
0.761
0.296
FreeSplat
24.99
0.814
0.162
25.20
0.829
0.179
27.23
0.873
0.196
24.64
0.819
0.241
Pose-
free
NoPoSplat 24.87
0.813
0.169
25.01
0.832
0.163
27.10
0.864
0.189
24.26
0.796
0.269
VicaSplat
24.65
0.795
0.206
24.50
0.806
0.164
26.46
0.837
0.214
22.95
0.765
0.306
SPFSplat
24.98
0.819
0.166
25.18
0.828
0.169
27.17
0.873
0.185
24.15
0.793
0.263
Anysplat∗
20.28
0.657
0.251
20.93
0.746
0.284
22.74
0.768
0.242
21.83
0.673
0.358
Anysplat
15.55
0.475
0.424
16.23
0.520
0.401
17.07
0.580
0.436
16.72
0.577
0.457
Ours
25.14
0.834
0.156
26.15
0.858
0.135
28.15
0.884
0.178
25.15
0.824
0.233
Table 2. Quantitative results of NVS on ScanNet under varying numbers of views. The best and second-best values are highlighted.
Method
3 views
10 views
28 views
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Pose-
required
MVSplat
24.98
0.784
0.228
23.12
0.768
0.236
22.26
0.714
0.336
FreeSplat
25.63
0.804
0.218
25.38
0.815
0.189
24.30
0.798
0.249
Pose-
free
NoPoSplat
25.55
0.803
0.213
24.21
0.811
0.198
23.21
0.764
0.268
VicaSplat
24.54
0.786
0.226
23.73
0.760
0.268
22.37
0.709
0.316
SPFSplat
25.85
0.788
0.229
24.63
0.773
0.226
23.54
0.775
0.261
Anysplat∗
19.84
0.658
0.314
22.06
0.687
0.353
21.17
0.680
0.357
Anysplat
15.05
0.415
0.416
16.73
0.522
0.437
16.58
0.561
0.439
Ours
26.57
0.841
0.189
26.83
0.851
0.179
26.87
0.859
0.173
ing improved novel view synthesis quality. These results
demonstrate the scalability of our approach to a larger num-
ber of views.
In Fig. 2, our method consistently produces clearer and
more realistic novel view renderings in challenging cases.
Ours better preserves fine structural details and reduces ar-
tifacts and blurring. See more visualizations in appendix.
Pose Estimation. The results are shown in the left side
of Tab. 3 and Tab. 4. For NoPoSplat, poses are estimated
via PnP [10] with RANSAC [9] from predicted Gaussian
centers, while other methods directly output camera poses.
TokenSplat consistently surpasses pose-free baselines on
RE10K, reducing RPE-R (lower is better) by 0.335 and
0.147 compared to VicaSplat and AnySplat under the 8-
view setting. This gain stems from our directionally con-
strained ADF-Decoder, which enforces disentangled inter-
action between camera and image tokens, leading to more
stable pose learning. On ScanNet, the model maintains ac-
curate pose estimation under the 28-view setting, reducing
ATE by 0.018 over AnySplat, confirming both robustness
and scalability of TokenSplat in multi-view scenarios.
Cross-Dataset Generalization. To assess zero-shot gener-
alization, we train TokenSplat on RE10K and directly eval-
uate it on ScanNet. As shown in the right halves of both
Tab. 1 and Tab. 3, which report cross-dataset results, our
method surpasses all SOTA approaches on both novel view
synthesis and pose estimation. Compared to FreeSplat, To-
kenSplat achieves up to 0.92/0.51 dB higher PSNR, and re-
duces ATE by 0.013/0.033 over SPFSplat. In Fig. 3, To-
kenSplat preserves coherent geometry and clearly recon-
structed scene structures. Notably, fine details such as fur-
niture boundaries are better preserved than in competing
methods, demonstrating the effectiveness of our method
across unseen scenes. The camera pose head further ensures
reliable pose estimation under novel camera configurations,
jointly contributing to superior cross-dataset performance.
Scene-level Visualization.
To further illustrate the re-
construction quality, Fig. 4 presents scene-level visualiza-
tions, where the camera is pulled back to reveal the global
3D Gaussian structures, along with several rendered novel
views. AnySplat, which directly fuses 3D Gaussians, ex-
hibits noticeable misalignment artifacts and visible fusion
cracks across viewpoints.
FreeSplat generates numerous
scattered Gaussians, while NoPoSplat and SPFSplat show
poor scalability and fail to generalize to unseen distant
viewpoints. In contrast, our method preserves coherent and
geometrically consistent reconstructions, even when ren-
dered from viewpoints far beyond the training range.
4.3. Ablation Analysis
We perform ablation studies on RE10K (8 views), summa-
rized in Tab. 5. Compared to our full model (a), (b) re-
placing the Token-aligned Gaussian Prediction with a pixel-
aligned Gaussian head degrades both reconstruction and

<!-- page 7 -->
Figure 2. Qualitative comparison on RE10K and ScanNet under varying numbers of reference views.
Figure 3. Cross-dataset generalization from RE10K to ScanNet.
Table 3. Quantitative results of pose prediction on RE10K with diverse views (left) and cross-dataset generalization to ScanNet (right).
Method
RE10K (4 views)
RE10K (8 views)
ScanNet (4 views)
ScanNet (8 views)
ATE↓RPE-t↓RPE-r↓ATE↓RPE-t↓RPE-r↓ATE↓RPE-t↓RPE-r↓ATE↓RPE-t↓RPE-r↓
NoPoSplat 0.032
0.070
1.923
0.123
0.135
1.757
0.139
0.259
2.135
0.208
0.268
4.098
VicaSplat
0.027
0.057
1.392
0.021
0.031
0.793
0.095
0.234
2.735
0.163
0.197
3.162
SPFSplat
0.036
0.078
3.113
0.037
0.047
2.117
0.075
0.201
2.658
0.121
0.173
3.676
Anysplat∗
0.028
0.051
1.515
0.020
0.025
0.578
0.092
0.238
1.902
0.098
0.150
2.016
Anysplat
0.033
0.055
1.624
0.024
0.029
0.605
0.099
0.268
1.995
0.101
0.157
2.114
Ours
0.016
0.034
1.054
0.012
0.019
0.458
0.062
0.159
1.162
0.088
0.130
1.672

<!-- page 8 -->
Table 4. Quantitative results of pose prediction on ScanNet with diverse views. The best and second-best values are highlighted.
Method
3 views
10 views
28 views
ATE↓
RPE-t↓
RPE-r↓
ATE↓
RPE-t↓
RPE-r↓
ATE↓
RPE-t↓
RPE-r↓
NoPoSplat
0.055
0.218
2.041
0.167
0.175
5.537
0.187
0.197
4.551
VicaSplat
0.075
0.268
1.803
0.171
0.231
3.768
0.243
0.189
3.725
SPFSplat
0.104
0.336
2.351
0.155
0.167
3.249
0.207
0.216
4.428
Anysplat∗
0.057
0.198
1.567
0.135
0.181
1.165
0.097
0.094
1.239
Anysplat
0.062
0.213
1.611
0.148
0.196
1.192
0.098
0.091
1.362
Ours
0.041
0.147
1.252
0.093
0.135
0.799
0.080
0.075
0.709
Figure 4. Scene-level visualizations and multiple novel viewpoints renderings.
Table 5. Component ablations on RE10K (8 view).
Method
PSNR↑SSIM↑LPIPS↓ATE↓RPE-t↓RPE-r↓
(a) Ours
26.15
0.858
0.135 0.012 0.019
0.458
(b) w Pixel Head
25.33
0.832
0.159 0.018 0.028
0.496
(c) w Any. Fusion.
25.77
0.847
0.148 0.020 0.027
0.489
(d) w/o ADF-Dec.
25.88
0.845
0.146 0.023 0.029
0.504
(e) w/o intrin. emb. 25.54
0.835
0.157 0.016 0.022
0.471
pose estimation, with SSIM dropping by 0.026 and RPE-
r increasing by 0.038; (c) using AnySplat-style Gaussian
fusion, which fuses pixel-aligned Gaussians by predicting
per-Gaussian fusion confidence, improves over the pixel-
aligned head but still underperforms our token-based fu-
sion, with PSNR 0.38 dB lower; (d) replacing the ADF-
decoder with a standard ViT decoder that concatenates cam-
era and image tokens for attention increases pose entan-
glement, leading to 0.046 higher RPE-r and 0.011 higher
LPIPS (lower is better), highlighting the benefit of direc-
tionally constrained communication for disentangling pose
and image features; (e) removing the known intrinsic cam-
era embedding reduces reconstruction quality, as the model
struggles to capture scale, though pose estimation remains
competitive. Overall, these results demonstrate the effec-
tiveness of our model designs for accurate multi-view re-
construction and camera pose estimation.
5. Conclusion
We present TokenSplat, a feed-forward framework for joint
3D reconstruction and camera pose estimation from un-
posed multi-view images. By leveraging token-based multi-
view fusion and ADF-Decoder, it achieves clean feature dis-
entanglement and stable aggregation, enabling coherent re-
construction without iterative refinement. TokenSplat con-
sistently improves reconstruction and pose accuracy and
generalizes well, achieving competitive zero-shot perfor-
mance when trained on a single RE10K-scale dataset.

<!-- page 9 -->
References
[1] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 19457–19467, 2024. 1, 2
[2] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang,
Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast general-
izable radiance field reconstruction from multi-view stereo.
In Proceedings of the IEEE/CVF international conference on
computer vision, pages 14124–14133, 2021. 2
[3] Yu Chen and Gim Hee Lee. Dbarf: Deep bundle-adjusting
generalizable neural radiance fields.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 24–34, 2023. 1
[4] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In European Conference on Computer
Vision, pages 370–386. Springer, 2024. 1, 2, 5
[5] Shin-Fang Chng, Sameera Ramasinghe, Jamie Sherrah, and
Simon Lucey. Gaussian activated neural radiance fields for
high fidelity reconstruction and pose estimation.
In Eu-
ropean Conference on Computer Vision, pages 264–280.
Springer, 2022. 1
[6] Clifford. Preliminary sketch of biquaternions. Proceedings
of the London Mathematical Society, 1(1):381–395, 1871. 5
[7] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes.
In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 5828–5839, 2017. 5
[8] Konstantinos Daniilidis.
Hand-eye calibration using dual
quaternions.
The International Journal of Robotics Re-
search, 18(3):286–298, 1999. 5
[9] Martin A Fischler and Robert C Bolles.
Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartography. Communications
of the ACM, 24(6):381–395, 1981. 2, 6
[10] Richard Hartley and Andrew Zisserman. Multiple view ge-
ometry in computer vision.
Cambridge university press,
2003. 6
[11] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han,
Jiaolong Yang, Chong Luo, and Seungryong Kim. Pf3plat:
Pose-free feed-forward 3d gaussian splatting. arXiv preprint
arXiv:2410.22128, 2024. 2
[12] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jiaolong
Yang, Seungryong Kim, and Chong Luo.
Unifying cor-
respondence pose and nerf for generalized pose-free novel
view synthesis. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20196–
20206, 2024. 1
[13] Ranran Huang and Krystian Mikolajczyk. No pose at all:
Self-supervised pose-free 3d gaussian splatting from sparse
views. arXiv preprint arXiv:2508.01171, 2025. 2, 5
[14] Hanwen Jiang, Zhenyu Jiang, Yue Zhao, and Qixing Huang.
Leap: Liberate sparse-view 3d modeling from camera poses.
arXiv preprint arXiv:2310.01410, 2023. 1, 2
[15] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren,
Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng
Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting
from unconstrained views. arXiv preprint arXiv:2505.23716,
2025. 1, 2, 5
[16] Ladislav Kavan, Steven Collins, Carol O’Sullivan, and Jiri
Zara.
Dual quaternions for rigid transformation blending.
Trinity College Dublin, 5:4, 2006. 5
[17] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger,
Lorenzo Porzi,
Yuchen Zhang,
Tobias Fischer,
Arno
Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes,
et al. Mapanything: Universal feed-forward metric 3d re-
construction. arXiv preprint arXiv:2509.13414, 2025. 1
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2
[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 1, 2
[20] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Ground-
ing image matching in 3d with mast3r. In European Confer-
ence on Computer Vision, pages 71–91. Springer, 2024. 5
[21] Yihui Li, Chengxin Lv, Hongyu Yang, and Di Huang. Micro-
macro wavelet-based gaussian splatting for 3d reconstruc-
tion from unconstrained images.
In Proceedings of the
AAAI Conference on Artificial Intelligence, pages 5057–
5065, 2025. 1
[22] Zhiqi Li, Chengrui Dong, Yiming Chen, Zhangchi Huang,
and Peidong Liu. Vicasplat: A single run is all you need for
3d gaussian splatting and camera estimation from unposed
video frames. arXiv preprint arXiv:2503.10286, 2025. 5, 1
[23] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Si-
mon Lucey. Barf: Bundle-adjusting neural radiance fields.
In Proceedings of the IEEE/CVF international conference on
computer vision, pages 5741–5751, 2021. 1
[24] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Polle-
feys. Lightglue: Local feature matching at light speed. In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 17627–17638, 2023. 2
[25] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 1
[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
2
[27] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 12179–12188, 2021. 4

<!-- page 10 -->
[28] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-
net: Convolutional networks for biomedical image segmen-
tation. In International Conference on Medical image com-
puting and computer-assisted intervention, pages 234–241.
Springer, 2015. 2
[29] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 1, 2
[30] Brandon Smart, Chuanxia Zheng, Iro Laina, and Vic-
tor Adrian Prisacariu.
Splatt3r:
Zero-shot gaussian
splatting from uncalibrated image pairs.
arXiv preprint
arXiv:2408.13912, 2024. 1
[31] Cameron Smith, Yilun Du, Ayush Tewari, and Vincent Sitz-
mann. Flowcam: Training generalizable 3d radiance fields
without camera poses via pixel-aligned scene flow. arXiv
preprint arXiv:2306.00180, 2023. 1
[32] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang,
Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian
model for high-resolution 3d content creation. In European
Conference on Computer Vision, pages 1–18. Springer, 2024.
2
[33] Shengji Tang, Weicai Ye, Peng Ye, Weihao Lin, Yang Zhou,
Tao Chen, and Wanli Ouyang. Hisplat: Hierarchical 3d gaus-
sian splatting for generalizable sparse-view reconstruction.
arXiv preprint arXiv:2410.06245, 2024. 2
[34] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 2
[35] Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan,
Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai
Zhang.
Pf-lrm:
Pose-free large reconstruction model
for joint pose and shape prediction.
arXiv preprint
arXiv:2311.12024, 2023. 1, 2
[36] Weijie Wang, Yeqing Chen, Zeyu Zhang, Hengyu Liu,
Haoxiao Wang, Zhiyuan Feng, Wenkang Qin, Zheng Zhu,
Donny Y Chen, and Bohan Zhuang. Volsplat: Rethinking
feed-forward 3d gaussian splatting with voxel-aligned pre-
diction. arXiv preprint arXiv:2509.19297, 2025. 2
[37] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee
Lee. Freesplat: Generalizable 3d gaussian splatting towards
free view synthesis of indoor scenes. Advances in Neural
Information Processing Systems, 37:107326–107349, 2024.
1, 2, 5
[38] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 5
[39] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Vic-
tor Adrian Prisacariu. Nerf–: Neural radiance fields without
known camera parameters. 2021. 1
[40] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele,
and Jan Eric Lenssen.
latentsplat: Autoencoding varia-
tional gaussians for fast generalizable 3d reconstruction. In
European conference on computer vision, pages 456–473.
Springer, 2024. 1
[41] Haofei Xu, Anpei Chen, Yuedong Chen, Christos Sakaridis,
Yulun Zhang, Marc Pollefeys, Andreas Geiger, and Fisher
Yu. Murf: multi-baseline radiance fields. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20041–20050, 2024. 2
[42] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth.
In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 16453–16463, 2025. 2
[43] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen,
Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wet-
zstein. Grm: Large gaussian reconstruction model for ef-
ficient 3d reconstruction and generation. In European Con-
ference on Computer Vision, pages 1–20. Springer, 2024. 2
[44] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys,
Ming-Hsuan Yang, and Songyou Peng. No pose, no problem:
Surprisingly simple 3d gaussian splats from sparse unposed
images. arXiv preprint arXiv:2410.24207, 2024. 1, 2, 3, 4, 5
[45] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19447–19456,
2024. 1
[46] Cheng Zhang, Haofei Xu, Qianyi Wu, Camilo Cruz Gam-
bardella, Dinh Phung, and Jianfei Cai.
Pansplat:
4k
panorama synthesis with feed-forward gaussian splatting. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 11437–11447, 2025. 2
[47] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao,
Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large recon-
struction model for 3d gaussian splatting. In European Con-
ference on Computer Vision, pages 1–19. Springer, 2024. 2
[48] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 5
[49] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely.
Stereo magnification: learning view
synthesis using multiplane images. ACM Transactions on
Graphics (TOG), 37(4):1–12, 2018. 5
[50] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li,
Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane
meets gaussian splatting: Fast and generalizable single-view
3d reconstruction with transformers.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10324–10335, 2024. 1

<!-- page 11 -->
TokenSplat: Token-aligned 3D Gaussian Splatting for Feed-forward
Pose-free Reconstruction
Supplementary Material
A1. More Implementation Details of Experi-
ment
More Training Details.
Our model is implemented us-
ing PyTorch, and all models are trained on a single A800
GPU. Comparable results can also be achieved on a single
A6000 or A100 GPU by adjusting to a smaller batch size
and increasing the number of training iterations. Follow-
ing strategies similar to those in [44] and [22], we train
with a batch size of 16, a learning rate of 2 × 10−5 for
the backbone, and a learning rate of 2 × 10−4 for the other
components for 30,000 iterations under the 2-view settings.
Subsequently, for multi-view scenarios, we fine-tune the
model for 100,000 iterations with a backbone learning rate
of 4 × 10−6, a learning rate of 4 × 10−5 for the other com-
ponents, and a batch size of 1. For the 28-view test on Scan-
Net, we use the model trained under the 10-view setting to
evaluate the model’s ability to generalize to a larger number
of views. All other models, except for AnySplat, adopt a
similar setup.
More Details of MVSplat and FreeSplat Results. All set-
tings follow those specified in the official FreeSplat repos-
itory. For the 28-view results of FreeSplat on ScanNet, we
use the model trained with the “fvt” setting (with the maxi-
mum number of views during training still set to 10), as this
configuration offers better scalability and leads to higher
performance for FreeSplat.
More Details of NoPoSplat Results. We use the official
2-view checkpoint from the repository and further fine-tune
NoPoSplat with the exact same settings as our method to
ensure a fair comparison. Additionally, all methods use the
same view alignment protocol as NoPoSplat for evaluation
to ensure complete fairness.
More Details of SPFSPlat Results. According to the offi-
cial training configuration provided for SPFSplat, 200,000
training iterations are required. We use the official 2-view
checkpoint from the repository and further fine-tune them
for 200,000 iterations on the multi-view setting to ensure
sufficient training of SPFSplat.
More Details of VicsSplat Results. For VicaSplat, the offi-
cial code specifies the training requirements for each step on
RE10K, and we strictly follow the training and fine-tuning
procedures provided by the official code. For training on
ScanNet, we follow our training and fine-tuning settings,
and, as specified by the official guidelines, we perform point
distillation before training.
More Details of AnySplat Results.
Since the training
dataset of AnySplat is approximately four times larger than
the number of scenes in our RE10K training set and 95
times larger than the number of images in ScanNet. There-
fore, we report both the zero-shot results of the generic
model provided by AnySplat (denoted as AnySplat), as well
as the results obtained by training on RE10K and ScanNet
following the official instructions (denoted as AnySplat∗).
Since AnySplat itself is trained on images with a resolution
of 448, using different input resolutions leads to severe arti-
facts. Therefore, we use 448-resolution inputs for zero-shot
testing with AnySplat. It is worth noting that the input im-
age content is kept exactly the same as in other models, with
only the resolution being different. For novel view synthesis
tests, we render the 3DGS outputs of the models at a reso-
lution of 256×256, ensuring that all models are evaluated
at the same resolution. (Rendering at 448 resolution and
then resampling to 256 yields even worse results.) More-
over, since AnySplat uses VGGT as its backbone and only
supports inputs at 14× resolution, we train AnySplat∗at a
resolution of 266×266, and adopt a testing strategy similar
to that of AnySplat.
Results show that while the visual quality of AnySplat
is acceptable, its reconstructed scenes exhibit some distor-
tions and deformations. This distortion is not easily notice-
able when viewing AnySplat results in isolation, but be-
comes much more apparent when test views are aligned
with ground-truth views for evaluation (following the pro-
tocol in NoPoSplat). In many scenes, such distortions are
clearly observed after this alignment. Even after fine-tuning
on the corresponding scenes, achieving improvements and
reaching the performance reported by the official imple-
mentation, there remains a significant gap compared to our
method.
A2. More Implementation Details of Modula-
tion in Asymmetric Dual-Flow Decoder
Inspired by the Adaptive Layer Normalization in DiT, we
adopt a similar modulation strategy. Specifically, an MLP
predicts three modulation parameters, i.e. scale, shift, and
gate, from the camera token at each step. Since the camera
tokens are continuously updated within the decoder blocks,
after self-attention layer, a separate MLP predicts a new set
of modulation parameters from the updated camera token.
The normalization parameters (scale and shift) and an addi-
tional scaling parameter (gate) are applied as follows: be-
fore attention, we perform ˆtI
i ←ˆtI
i × (1 + scale) + shift;
after attention, we update as ˆtI
i ←(1 + gate) × ˆtI
i.

<!-- page 12 -->
Figure A1. Structure of our Gaussian Prediction Head.
Figure A2. Network details in our Gaussian Prediction Head.
A3. More Implementation Details of Gaussian
Prediction Heads
In this section, we further elaborate on the detailed design
of our Gaussian Prediction heads, as illustrated in Fig. A1.
our Gaussian Prediction heads take the multi-scale features
from the fused token decoder outputs as input. In our im-
plementation, we follow the DPT head by selecting features
from the 0-th, 6-th, 9-th, and 12-th layers of the decoder as
inputs F1, F2, F3, F4 to the Gaussian Prediction head.
The multi-scale features are first individually processed
by a projection module, denoted as Proj(·), as illustrated
in the upper part of Fig. A2. This module consists of a
1×1 Conv1D layer for channel transformation, followed by
a deconvolution layer for upsampling. Subsequently, two
fully connected layers are used for linear mapping to get
ˆFi. Ultimately, all features are unified to the same number
of channels, which is set to 256 in our implementation.
Subsequently, we perform deep-to-shallow fusion from
feature ˆF4 to feature ˆF1, as shown in the lower part of Fig.
A2. For feature ˆF4, we directly apply a residual module
composed of two fully connected layers for further feature
transformation. For shallower features ˆFi, we first add the
fused feature from the deeper layer F fusion
i+1
after passing
it through a residual module consisting of two fully con-
nected layers, and then apply another residual module with
two fully connected layers for feature transformation. Fi-
nally, the output is upsampled using a deconvolution mod-
ule and further processed by a fully connected layer to ob-
tain F fusion
i
.
Figure A3. Number of Gaussians with increasing input views for
pixel-aligned GS, Gaussian fusion methods, and our Token Fusion
operation.
Figure A4. Inference time with increasing number of input views.
A4. More Implementation Details of Pose
Heads
The pose head takes the camera token output from the
Asymmetric Dual-Flow Decoder as input and consists of
activation functions and fully connected layer. Specifically,
we use a ReLU activation function and a fully connected
layer with dimensions (768, 7) in this work.
The pose
head outputs camera rotation (as a quaternion) and trans-
lation, which are then converted to unit dual quaternions for
loss computation. Our pose head is designed with a sim-
ple structure, as the Asymmetric Dual-Flow Decoder effec-
tively disentangles camera features, allowing accurate pre-
dictions with minimal complexity.
A5. More Experimental Analysis
A5.1. Efficiency
We investigate the impact of different fusion methods on
the number of Gaussian primitives and show the result in

<!-- page 13 -->
Figure A5. Visualization Results with increasing number of input views.
Fig. A3. As the number of input views increases, the num-
ber of pixel-aligned Gaussian primitives grows linearly. In
contrast, using Gaussian fusion modules in 3D space di-
rectly or our Token Fusion operation helps stabilize the
growth of Gaussians. It can be observed that as the num-
ber of views increases, our Token Fusion operation leads to
a much slower growth in the number of Gaussians, mak-
ing the count more stable compared to Gaussian fusion.
We investigate how the model’s inference time changes as
the number of input views increases and show the result in
Fig. A4. As other models experience excessive growth in
model size and memory usage with increasing views, we
mainly compare with FreeSplat and AnySplat. Since AnyS-
plat only supports up to 14× resolution, we use a resolution
of 252 × 252, while FreeSplat and ours use 256 × 256. Our
method maintains more stable inference time as the num-
ber of views increases. This is because, unlike Gaussian
fusion methods where computation in the Gaussian predic-
tion stage remains proportional to the number of input im-
ages, our approach performs fusion first, fully decoupling
the number of Gaussians from the number of images, re-
sulting in more stable inference time.
A5.2. Visualization Results with Increasing Num-
ber of Views
We further demonstrate the impact of increasing the num-
ber of input views on novel view synthesis. Starting with
the first and last views, we gradually add more intermediate
views. As shown in Fig. A5, additional input views pro-
gressively enhance scene completeness and visual details,
especially in challenging regions such as railings.
A6. More Visualization Results
We present additional qualitative comparisons with base-
lines on the RE10K and ScanNet datasets in Fig. A6 to
Fig. A10, further demonstrating the effectiveness and im-
provements of our method. Our approach achieves stable
and superior performance across varying numbers of input
images and diverse input data.
A7. Video Demo
The project webpage includes comparison videos between
our approach and state-of-the-art methods. Whether in ren-
dered videos or when observing the entire scene from a
distance, our method consistently exhibits more complete
and coherent structures, whereas existing methods suffer
from blurring or defocus-like artifacts, and some even ex-
hibit severe structural issues. For example, when running
SPFSplat and VicaSplat with more input views than used
during training, we observe significant scene compression
at the boundaries, making it difficult to effectively extend
outward. FreeSplat projects Gaussians into 3D space using
depth maps, resulting in a large number of floating artifacts.
In the left-side close-up rendering video of AnySplat, no-
ticeable cracks caused by unsuccessful Gaussian fusion can
be observed near the mirror at the beginning, and multiple
issues arising from 3D Gaussian fusion are also clearly vis-
ible in the long-range scene. Furthermore, our method ex-
hibits remarkable generalization ability, as evidenced by the
reconstruction results on scenes from different datasets and
in-the-wild data casually captured with mobile phones.

<!-- page 14 -->
Figure A6. More qualitative comparisons on RE10K with 4 input images.

<!-- page 15 -->
Figure A7. More qualitative comparisons on RE10K with 8 input images.

<!-- page 16 -->
Figure A8. More qualitative comparisons on ScanNet with 3 input images.

<!-- page 17 -->
Figure A9. More qualitative comparisons on ScanNet with 10 input images.

<!-- page 18 -->
Figure A10. More qualitative comparisons on ScanNet with 28 input images.
