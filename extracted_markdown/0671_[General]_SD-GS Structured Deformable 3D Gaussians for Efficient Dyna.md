<!-- page 1 -->
SD-GS: Structured Deformable 3D Gaussians for Efficient Dynamic Scene
Reconstruction
Wei Yao1⋆, Shuzhao Xie1⋆, Letian Li1, Weixiang Zhang1,
Zhixin Lai2, Shiqi Dai3, Ke Zhang4, Zhi Wang1†
1SIGS, Tsinghua University
2Google
3Department of CST, Tsinghua University
4Soochow University
{yaow21, xsz24, lilt24, zhang-wx22}@mails.tsinghua.edu.cn, zhixinlai@google.com,
daisq99@gmail.com, kzhang19@suda.edu.cn, wangzhi@sz.tsinghua.edu.cn
Figure 1. Our method successfully achieves photorealistic quality and high resolution rendering in real time while maintaining a compact
model size. (a) Our approach can be rendered at high resolution with 82 FPS on an Nvidia RTX 3090 GPU. (b) Quantitative comparisons
of rendering quality, speed, and storage requirements with several state-of-the-art baselines on the N3DV Dataset.
Abstract
Current 4D Gaussian frameworks for dynamic scene re-
construction deliver impressive visual fidelity and render-
ing speed, however, the inherent trade-off between storage
costs and the ability to characterize complex physical mo-
tions significantly limits the practical application of these
methods. To tackle these problems, we propose SD-GS, a
compact and efficient dynamic Gaussian splatting frame-
work for complex dynamic scene reconstruction, featuring
two key contributions. First, we introduce a deformable an-
chor grid, a hierarchical and memory-efficient scene rep-
resentation where each anchor point derives multiple 3D
⋆Equal contribution.
†Corresponding author.
Gaussians in its local spatiotemporal region and serves as
the geometric backbone of the 3D scene. Second, to en-
hance modeling capability for complex motions, we present
a deformation-aware densification strategy that adaptively
grows anchors in under-reconstructed high-dynamic re-
gions while reducing redundancy in static areas, achieving
superior visual quality with fewer anchors. Experimental
results demonstrate that, compared to state-of-the-art meth-
ods, SD-GS achieves an average of 60% reduction in model
size and an average of 100% improvement in FPS, signifi-
cantly enhancing computational efficiency while maintain-
ing or even surpassing visual quality.
arXiv:2507.07465v1  [cs.GR]  10 Jul 2025

<!-- page 2 -->
1. Introduction
Dynamic scene reconstruction from multi-view videos is
an important task in 3D computer vision, with tremen-
dous applications in AR, VR, and 3D content creation [31].
While Neural Radiance Fields (NeRFs) [1, 17, 24, 26, 32]
have made notable progress in dynamic scene reconstruc-
tion, 3D Gaussian Splatting (3DGS)-based methods [14]
have emerged as the dominant approach. This advantage
stems from two key factors: first, 3DGS employs explicit
geometric representations that naturally facilitate dynamic
modeling; second, its highly optimized CUDA rasteriza-
tion pipeline eliminates the need for intensive sampling and
querying of neural fields [4, 43, 44], significantly accelerat-
ing both rendering and training.
Recent 3DGS-based dynamic scene representations fall
into two categories: 1) Explicit approaches, which extend
the 3D Gaussian to 4D Gaussian primitives by adding a
temporal dimension to approximate the spatiotemporal 4D
volume of dynamic scenes [9, 39].
Although 4D Gaus-
sians achieve higher visual quality and faster rendering
speed, these works suffer from substantial storage require-
ments for numerous Gaussians and their 4D parameters as
they cannot leverage inherent cross-spatiotemporal correla-
tions. 2) Implicit approaches, which employ consistent de-
formable 3D Gaussians as the underlying structure to char-
acterize scenes, interpreting motion at each timestamp as a
deformation of the underlying structure of different mag-
nitudes [2, 35, 38]. These deformation-based methods of-
fer a more compact representation than explicit methods by
leveraging intrinsic correlations across spatial and temporal
conditions through deformations, enabling effective cross-
spatiotemporal information sharing.
However, these im-
plicit representations struggle to handle complex real-world
motions and generally exhibit slower rendering speeds.
Motivated by these challenges, we introduce SD-GS for
dynamic scene reconstruction to address the balance be-
tween storage efficiency and the capability to model com-
plex real-time motions. Our framework extends the anchor-
based scaffold representation from static scenes to de-
formable 3D Gaussian-based dynamic scene reconstruction.
Since local 3D Gaussians in spatiotemporal domains typi-
cally possess similar feature information, we propose to use
deformable 3D structured anchors initialized from a sparse
grid of SfM points as the underlying scene representation,
achieving a more compact structure compared to explicit
methods. The attributes of local neural 3D Gaussians can
be predicted from anchor features adapted to various times-
tamps and viewing angles.
Furthermore, previous densification strategies in static
scene reconstruction accumulated Gaussian gradients and
applied growth in underfit regions based on predefined gra-
dient thresholds. When applied to dynamic scene recon-
struction, this approach tends to generate numerous redun-
dant Gaussians in static regions, accompanied by increased
deformation calculation overhead and lower FPS, while
producing insufficient anchors in dynamic regions, leading
to poor reconstruction quality in these areas. To overcome
this limitation, we propose a deformation-aware densifica-
tion strategy to achieve adaptive and efficient anchor allo-
cation, directing new anchors toward poorly reconstructed
high-dynamic regions instead of static scene parts.
In summary, our contributions are as follows: 1) We
introduce structured 3D Gaussians and a meticulously de-
signed time-aware architecture to model dynamic scenes,
significantly reducing the model size.
2) We propose a
deformation-aware densification method to further enhance
the representation ability of the deformation grid in com-
plex dynamics, which also reduces storage costs. 3) Ex-
tensive experiments demonstrate that our approach signif-
icantly reduces storage requirements by 60% on average
and achieves 2x faster rendering speed, while maintaining
or even exceeding state-of-the-art visual quality.
2. Related Work
2.1. Dynamic 3D Gaussians
Based on the formulation of the movement of the objects,
the existing Dynamic 3D Gaussians can be divided into
explicit and implicit methods based on the representation
of time.
The explicit methods [9, 18, 39] are built on
the 4D Gaussians, with one more dimension represent-
ing the timestamp, which requires substantial memory for
training and rendering. In contrast, the implicit methods
[2, 13, 22, 35, 38] employ the deformation grid to model the
movement, which greatly utilize the spatiotemporal corre-
lations to reduce the memory and storage requirement. For
instance, D-3DGS [22] models dynamic scenes by allowing
the positions and rotation matrices of 3DGS to change over
time. Deformable 3DGS [38] uses an MLP to model a de-
formation field based on time and the canonical Gaussian
space. SC-GS [13] bounds dense 3DGS with sparse control
points, calculating the movement of Gaussians in a coarse-
to-fine manner. However, these implicit methods ignore the
inner redundancy of the canonical model, 3D Gaussians.
Thus, we introduce a scaffold representation to replace the
3D Gaussians, which further reduces the memory require-
ment. Besides, we carefully design a deformation grid to
enable the high-quality reconstruction for each timestamp.
Although contemporary works [7, 16] also utilize scaf-
fold representations, they employ a different deformation
strategy from us. For example, Scaffold4D [7] still keeps
the temporal dimension of the 4D Gaussians, while our
method utilizes a memory-efficient deformation grid.

<!-- page 3 -->
2.2. Gaussian Densification
Gaussian densification plays a pivotal role in recovering ac-
curate scene geometry. The vanilla 3DGS approach initial-
izes sparse points from structure-from-motion (SfM) [29]
and employs adaptive density control [14], first select-
ing Gaussians based on image-space gradients and then
cloning or splitting them according to their scale.
Intu-
itively, a well-designed densification strategy can enhance
optimization convergence, accelerating the overall training
process. Most existing methods, with limited exceptions
such as [10, 15, 23], prioritize improving rendering quality,
often at the expense of increased computational overhead.
Recent studies, such as [15, 23, 28, 40, 45], have refined
this strategy by incorporating image-space priors and intrin-
sic Gaussian properties for more informed selection. Other
approaches integrate multi-view constraints [6, 8, 19], lever-
age advanced optimization techniques, or analyze point
cloud geometry [10]. However, densification strategies for
dynamic Gaussians remain underexplored. In this work, we
propose a deformation-aware densification strategy tailored
to our dynamic Gaussian representation, which not only re-
duces storage consumption but also improves reconstruc-
tion quality for complex motion patterns.
3. Preliminary
3.1. 3D Gaussian Splatting
3DGS [14] is an explicit 3D representation in the form of
point clouds, utilizing Gaussians to model the points. Each
Gaussian is characterized by a covariance matrix Σ and a
center point µ, which is referred to as the mean value of
the Gaussian: G(x) = e−1
2 (x−µ)⊤Σ−1(x−µ). To maintain
the positive definiteness of the covariance matrix Σ, 3DGS
decomposes Σ into a scaling matrix S = diag(s), s ∈R3
and a rotation matrix R: Σ = RSS⊤R⊤. The rotation
matrix R is parameterized by a rotation quaternion r ∈R4.
The backpropagation process is illustrated in [14].
When rendering novel views, the technique of splatting
[41, 46] is employed for the Gaussians within the camera
planes. As introduced by [47], using a viewing transform
denoted as W and the affine transform J, the covariance
matrix Σ′ in camera coordinates system can be computed
by Σ′ = JWΣW⊤J⊤. Specifically, for each pixel, the
color and opacity of Gaussians are computed using G(x).
The blending of N ordered points that overlap the pixel is
given by: C = P
i∈N ciαi
Qi−1
j=1(1 −αj). Here, ci and
αi represent the density and color of this point computed
by a Gaussian with covariance Σ multiplied by a per-point
opacity and SH color coefficients.
3.2. Scaffold-GS
Scaffold-GS [21] is a variant of 3DGS, widely adopted in
3DGS compression [5, 27, 33, 36, 37] due to its low stor-
age requirements. It introduces anchor points to capture
common attributes of local 3D Gaussians. Specifically, the
anchor points are initialized from neural Gaussians by vox-
elizing the 3D scenes. Each anchor point has a context fea-
ture f ∈R32, a location x ∈R3, a scaling factor l ∈R6
and k learnable offset O ∈Rk×3. Given a camera at xc,
anchor points are used to predict the view-dependent neural
Gaussians in their corresponding voxels as follows,
{ci, ri, si, αi}k
i=0 = MLP(f, σc, ⃗dc),
(1)
where σc = ||x −xc||2, ⃗dc =
x−xc
||x−xc||2 , the superscript i
represents the index of neural Gaussian in the voxel, si, ci ∈
R3 are the scaling and color respectively, and ri ∈R4 is the
quaternion for rotation. The positions of neural Gaussians
are then calculated as
{µ0, ..., µk−1} = x + {O0, ..., Ok−1} · l:3,
(2)
where x is the learnable positions of the anchor and l:3 is
the base scaling of its associated neural Gaussians.
Af-
ter decoding the properties of neural Gaussians from an-
chor points, the remaining steps are the same as 3DGS [14].
By predicting the properties of neural Gaussians from the
anchor features and saving the properties of anchor points
only, Scaffold-GS greatly eliminates the redundancy among
3D neural Gaussians and decreases the storage demand.
3.3. Gaussian Deformation Field Network
The Gaussian Deformation Field Network is a core compo-
nent of 4DGS [35] designed to model dynamic 3D scenes
across space and time.
It consists of a spatial-temporal
structure encoder H and a multi-head Gaussian deformation
decoder D. Given a 3DGS model G and a timestamp t, the
network predicts deformations ∆G = F(G, t) to generate
temporally coherent 4D Gaussians G′ = G + ∆G.
The encoder H leverages a memory-efficient multi-
resolution HexPlane decomposition [3, 12], which projects
4D spatiotemporal features onto six 2D planes:
(x, y),
(x, z), (y, z), (x, t), (y, t), and (z, t). Each plane employs
bilinear interpolation to aggregate multi-scale voxel fea-
tures fh, followed by a lightweight MLP ϕd to fuse these
features into a unified deformation embedding fd. This hi-
erarchical encoding captures localized spatial-temporal cor-
relations among neighboring Gaussians while minimizing
computational overhead.
The decoder D utilizes separate MLP heads to predict
deformation parameters for position (∆X), rotation (∆r),
and scaling (∆s) as:
(∆X, ∆r, ∆s) = (ϕx(fd), ϕr(fd), ϕs(fd)),
(3)
yielding deformed Gaussians G′ = {X +∆X, r+∆r, s+
∆s, σ, C}. The framework preserves the differential split-
ting mechanism, enabling efficient novel view synthesis by

<!-- page 4 -->
Anchor Deformation Field
MLP
View direction
Neural Gaussian Generation Network
Neural Gaussians
Timestamp t
Feature
∆𝑙
∆𝑞
Canonical Model
Space-Time
Timestamp t
New anchors
Deformation-aware Densification
∆𝑥
Position
Rotation
Scaling
Rendered Image
Growing
∇!"
#
f
f
Figure 2. Overview of SD-GS. We introduce the Canonical Gaussian Model M as the geometric structure of dynamic scenes. Given
the Canonical Gaussian Model M and timestamp t, the Anchor Deformation Field F transforms the original Canonical Gaussian Model
M into the Deformed Gaussian Model M′. Neural Gaussians at the specific timestamp are then generated through the Neural Gaussians
Generation Network N. These neural Gaussians are subsequently splatted to produce rendered images using a 3D Gaussian splatting
pipeline. To better model complex real-world dynamics, we propose a deformation-aware densification strategy that encourages new
anchors to grow efficiently in under-reconstructed high-dynamic regions while reducing redundancy in static areas.
rendering deformed Gaussians through G′. This approach
balances expressiveness and efficiency, making it suitable
for dynamic scene modeling.
4. Method
Overview. We introduce a novel compact representation for
dynamic scenes, which represents the scene using a set of
deformable 3D Gaussians in an anchor-based framework.
In this section, we will describe each component and its
corresponding optimization process. In Sec. 4.1, we intro-
duce the overall framework, including the Canonical Gaus-
sian Model M, and the method of obtaining the Canonical
Gaussian Model at a specific timestep via Anchor Defor-
mation Field F. Additionally, to model motion with finer
details, we incorporate additional temporal information to
supervise Neural Gaussians Generation Network N.
In
Sec. 4.2, we elaborate on how our anchors efficiently grow
in complex dynamic regions with reconstruction deficien-
cies, achieving better visual performance with fewer anchor
points. Furthermore, the optimization framework will be
introduced in Sec. 4.3.
4.1. Model Architecture
The overview of our framework is illustrated in Fig. 2,
which consists of three main components: the Canonical
Gaussian Model M, the Anchor Deformation Fields F, and
the Neural Gaussian Generation Network N.
Canonical Gaussian Model M. To reduce memory re-
quirements, we replace the 3D Gaussians with Scaffold-GS
[21], which significantly decreases memory usage through
its anchor structure. The Canonical Gaussian Model M
represents the entire scene’s geometric structure by com-
bining anchor points with a set of local neural Gaussians.
As mentioned in Sec. 3.2, each anchor point is character-
ized by a local context feature vector fv ∈R32, a 3D po-
sition x ∈R3, a scaling factor l ∈R6, rotation quaternion
q ∈R4 and k learnable offsets O ∈Rk×3. The last three
dimensions of the scaling factor l enable anisotropic scaling
of the neural Gaussians, while the first three dimensions, to-
gether with learnable offsets O, determine the positions of
k neural Gaussians. The rotation quaternion q primarily in-
fluences the view frustum computation because we restrict
the prediction of the neural Gaussian to the anchors within
the view frustum during inference. To better learn high-
quality anchor distributions in dynamic regions, we initial-
ize the anchors using the sparse points from COLMAP [29],
and then train a static Canonical Gaussian Model without
anchor deformation using all multi-view images from the
dynamic video during the coarse stage.
Anchor Deformation Field F. Given the Canonical Gaus-
sian Model M and timestamp t, the Anchor Deformation
Field F transforms the original Canonical Gaussian Model
M into Deformed Gaussian Model M′ = ∆M + M.
The deformation ∆M is introduced by F(M, t), where
a spatial-temporal structure encoder H encodes both tem-
poral and spatial features of anchors fd = H(M, t), and
a multi-head anchor deformation decoder D predicts the
deformation ∆M = D(fd).
The deformation ∆M =
{∆x, ∆l, ∆q}, where ∆x, ∆l, and ∆q represent the defor-
mation of anchor’s 3D position, scaling factor, and rotation
quaternion, respectively. These are computed by separate
MLPs:
(∆x, ∆l, ∆q) = (ϕx(fd), ϕl(fd), ϕq(fd)).
(4)
Our strategy quantifies the deformation of anchors, encour-
aging new anchors to grow in dynamic regions that lack re-
construction rather than static regions. The details of this
strategy are further discussed in Sec. 4.2.

<!-- page 5 -->
Neural Gaussians Generation Network N. To enhance
the model’s temporal perception, we incorporate temporal
information into Neural Gaussians Generation Network N.
Specifically, given the Deformed Gaussian Model M′ un-
der a certain moment, we determine the positions for k neu-
ral Gaussians as follows:
{µ0, . . . , µk−1} = x′ + {O0, . . . , Ok−1} · l1:3.
(5)
where x′ represents the deformed position of each visi-
ble anchor. The attributes of each neural Gaussian (opac-
ity αi ∈R, color ci ∈R3, rotation ri ∈R4 and scal-
ing si ∈R3) are predicted through individual MLP de-
coders.
Specifically, we construct a conditional vector
[fv, dvc, ϕ(t)] by combining the anchor feature fv, viewing
direction dvc, and temporal embedding ϕ(t). The condi-
tional vector is fed into four independent fully-connected
networks Fα, Fc, Fr, Fs, which decode all attributes of the
neural Gaussians:







αi = Fα(fv, dvc, ϕ(t))
ci = Fc(fv, dvc, ϕ(t))
ri = Fr(fv, dvc, ϕ(t))
si = Fs(fv, dvc, ϕ(t))
(6)
After obtaining the 3D neural Gaussians at the given
timestamp, we render them using the existing efficient 3D
Gaussian splatting method [14], which applies to neural
Gaussians within the view frustum and with opacity greater
than a certain threshold.
4.2. Deformation-aware Densification Strategy
The approach represents temporal variations through the de-
formation of 3D Gaussian tends to produce relatively lower
visual quality in dynamic regions. The key factor to achiev-
ing high-quality results lies in efficiently growing new an-
chors in under-reconstructed dynamic regions while reduc-
ing redundant anchors in static areas.
Scaffold-GS [21] introduced an error-based anchor ex-
pansion strategy for static scenes, which grows new anchors
where neural Gaussians find significant. This approach col-
lects gradients of neural Gaussians by averaging over N
iterations, denoted as ∇g. Then, new anchors are placed
in underfitting regions based on predefined gradient thresh-
olds. However, when directly applied to dynamic scene re-
construction, the transient motions occurring in dynamic re-
gions, due to their short duration, fail to acquire sufficient
∇g to generate anchors for motion modeling, as they are in-
evitably penalized by the denominator N, regardless of their
actual errors. Consequently, this method fails to achieve sat-
isfactory visual results.
To address the issue, we propose a deformation-aware
densification strategy that quantifies anchor deformations
and assigns larger gradient weights to active neural Gaus-
sians. This enables anchors to adaptively grow based on
motion dynamics. We formulate ∇g as:
∇g =
PN
i=1 wi∥∇i
2D∥
PN
i=1 wi
,
(7)
where ∇i
2D is the 2D positional gradient of neural Gaus-
sians in the i-th iteration, and the weight term wi is deter-
mined by the anchor’s deformation magnitude as follows:
wi = α∥∆xi∥
sx
+ β ∥∆li∥
sl
+ γ ∥∆qi∥
sq
,
(8)
where ∆xi, ∆li, and ∆qi represent the anchor’s deforma-
tion in position, scaling, and rotation, respectively. sx, sl
and sq are normalization reference values for each deforma-
tion magnitude. To effectively highlight anchor points with
significant deformation, we use the 90th percentile value of
each deformation type in each iteration as the normaliza-
tion reference values. The weighting coefficients α, β, and
γ control the relative contribution of each deformation com-
ponent. Specifically, ∥∆xi∥and ∥∆li∥are computed as the
Euclidean distance between the anchor positions and scal-
ing parameters before and after deformation. The rotation
deformation amplitude ∥∆qi∥is:
∥∆qi∥= 2·arccos
 
clip
 
4
X
k=1
q(k)
orig,i · q(k)
def,i
 , 0.0, 1.0
!!
,
(9)
where qorig, qdef ∈RN×4 represent the unit quaternions be-
fore and after anchor deformation. This formula calculates
the absolute value of the quaternion dot product to mea-
sure rotation difference, where the absolute value operation
ensures that the direction of rotation remains unchanged.
The dot product result is clipped to the [0,1] range to avoid
floating-point errors causing numerical instability. Then the
2·arccos(·) function maps this to the rotation angle ranging
from [0, π].
Our strategy precisely identifies anchor points in dy-
namic regions with significant deformation, enabling these
anchors to receive greater gradient weight rewards. This
mechanism encourages the growth of new anchors in under-
reconstructed dynamic areas rather than static backgrounds,
thereby optimizing the anchor allocation mechanism to
achieve adaptive spatial deployment of anchors and improve
the efficiency of dynamic scene reconstruction.
4.3. Optimization Framework
Loss Design. We select L1 loss over rendered pixel col-
ors, SSIM loss LSSIM, a grid-based spatiotemporal total-
variation loss Ltv [3, 11, 30] and volume regularization Lvol
[20]. The learnable parameters of anchors and MLPs are
co-optimized by minimizing the rendering discrepancy. The
entire training loss function is as follows:
L = L1 + λSSIMLSSIM + λtvLtv + λvolLvol,
(10)

<!-- page 6 -->
Figure 3. Qualitative results on N3DV dataset. The white boxes highlight under-reconstructed regions. Our method demonstrates
superior fidelity across both dynamic and static areas of the scene.
Table 1. Quantitative results on N3DV dataset. We computed
the average metrics across all six scenes.
The best and the
second best results are denoted by pink and yellow.
Model
Metrics
Computational cost
PSNR↑
SSIM↑
LPIPS↓
Training time↓
FPS↑
Storage↓
4DGS [35]
30.94
0.936
0.056
45min
34.7
59MB
E-D3DGS [2]
30.86
0.938
0.048
3h20min
45.4
64MB
Realtime-4DGS [39]
31.11
0.939
0.050
7h50min
45.9
7970 MB
Ours
31.35
0.942
0.047
80min
82.1
22MB
where the weighting coefficients λSSIM = 0.2, λtv = 0.01,
and λvol = 0.01.
5. Experiment
In Sec. 5.1, we introduce the datasets, metrics, and base-
lines. In Sec. 5.2, we present the performance of our method
on two different datasets and compare it with state-of-the-
Table 2. Quantitative results on HyperNeRF dataset. The best
and the second best results are denoted by pink and yellow.
1
uses the metric from the original paper.
Model
Metrics
Computational cost
PSNR↑
SSIM↑
LPIPS↓
Training time↓
FPS↑
Storage↓
4DGS [35]
25.72
0.744
0.230
30min
34.1
74MB
E-D3DGS [2]1
25.43
0.697
0.231
2h5min
73.4
50MB
Deformable 3DGS [38]
24.91
0.705
0.243
1h30min
14.2
172 MB
Ours
25.79
0.737
0.221
28min
79.7
43MB
art methods based on 3D Gaussian Splatting. Subsequently,
in Sec. 5.3, we conduct ablation studies to demonstrate the
effectiveness of each module.
5.1. Experimental Settings
Datasets. We evaluate our method on two representative
real-world dynamic scene datasets: 1) Neural 3D Video

<!-- page 7 -->
Figure 4. Qualitative results on HyperNeRF dataset. This figure presents qualitative comparisons on the HyperNeRF dataset. The white
boxes highlight under-reconstructed regions. Our method demonstrates superior fidelity across both dynamic and static areas of the scene.
dataset (N3DV) [17] contains 6 real-world scenes. These
scenes feature relatively long durations and diverse mo-
tions, some containing multiple moving objects. Each scene
has approximately 20 synchronized videos. Except for the
flame salmon scene, which consists of 1200 frames, all
other scenes comprise 300 frames. For each scene, we se-
lect one camera view for testing while using the remaining
views for training. 2) HyperNeRF [24] is captured with
1-2 cameras, following straightforward camera motion. It
contains complex dynamic variations, such as human move-
ments and object deformations.
Baselines. We compare our method against several state-
of-the-art works in dynamic scene reconstruction, includ-
ing deformation-based methods like 4DGS [35] and De-
formable 3DGS [38], as well as 4D Gaussian-based meth-
ods like Real-time 4DGS [39]. We utilized their official
code to test their performance.
Metrics.
To evaluate reconstruction quality, we employ
peak-signal-to-noise ratio (PSNR), structural similarity in-
dex (SSIM) [34], and perceptual quality measure LPIPS
[42] with an AlexNet Backbone to assess the rendered im-
ages. Additionally, we evaluate storage efficiency by calcu-
lating the output file size as storage (MB), including point
cloud files, MLP weights, and other relevant components.
We also measured the rendering speed (FPS).
Implementation Details. To provide better anchor initial-
ization, we first warm up by training a static ScaffoldGS
without any deformation for 3000 iterations during the
coarse stage. Subsequently, in the fine stage of 140k iter-
ations, we train the anchor deformation field network along
with the learnable parameters of the anchors. Our imple-
mentation is based on the PyTorch [25] framework and runs
on a single NVIDIA RTX 3090 GPU. The optimization
parameters of the entire framework are appropriately fine-
tuned with reference to Scaffold-GS [21] and 4DGS [35].
We set α = 0.8, β = 0.1, and γ = 0.1 for Eq. 8.
5.2. Results Analysis
N3DV. We deliver quantitative results on the N3DV dataset
in Table 1. While deformation-based methods [2, 35] offer

<!-- page 8 -->
Table 3. Ablation studies on each component of our method.
“DAD” refers to deformation-aware densification strategy. “TIN”
refers to temporal injection in Neural Gaussians Generation Net-
work N. Experiments are conducted on the flame steak scene of
the Neural 3D Video dataset.
Model
PSNR↑SSIM↑LPIPS↓Anchors↓Storage↓
Ours w/o DAD 32.61
0.951
0.037
53K
26.28
Ours w/o TIN
31.15
0.949
0.041
40K
22.13
Ours w/o ∆x
30.20
0.947
0.042
41K
22.47
Ours w/o ∆l
32.43
0.955
0.038
41K
22.55
Ours w/o ∆q
32.70
0.956
0.036
41K
22.53
Ours
33.09
0.957
0.036
41K
22.54
compact memory usage, their FPS performance is signif-
icantly constrained due to the computational overhead of
deformation calculations for numerous Gaussians. In con-
trast, our method not only delivers superior image quality
but also achieves approximately 100% higher FPS while re-
ducing storage costs by around 65%. Notably, compared to
the 4D Gaussian-based approach [39], our method achieves
a remarkable 362× reduction in storage requirements and
6.4× faster training time while simultaneously improving
FPS by approximately 82%. Overall, our method outper-
forms state-of-the-art Gaussian Splatting-based methods in
both visual quality and rendering efficiency while maintain-
ing an exceptionally compact model size.
To further evaluate the model performance, we provide
qualitative comparisons with baselines in 3. As highlighted
in the boxed areas, existing methods often introduce arti-
facts and blurriness, particularly struggling with dynamic
region reconstruction. In contrast, our method generates
sharp and high-fidelity rendering results in both static and
dynamic regions.
HyperNeRF. Table 2 presents the quantitative results on
the HyperNeRF dataset. The results demonstrate that our
method achieves highly competitive reconstruction perfor-
mance while achieving the fastest rendering speed, short-
est training time, and minimal storage requirements. Fig-
ure 4 presents qualitative comparisons with Gaussian-based
methods on the HyperNeRF dataset.
Previous methods
struggle to reconstruct regions with rapid motion, often pro-
ducing blurry artifacts in dynamic areas, such as the moving
hands and knife in the boxed areas. In contrast, our method
achieves high visual fidelity in both static and dynamic re-
gions, effectively mitigating motion-related distortions.
5.3. Ablation Studies
Effects of deformation aware densification strategy. To
demonstrate the effectiveness of our proposed deforma-
tion aware densification strategy, we conduct a compara-
tive analysis of anchors spatial distribution. As illustrated in
Figure 5. Effectiveness of deform-aware densification strategy.
Without this strategy, static and dynamic regions exhibit similar
anchor density, as shown in (a). By adopting the deform-aware
densification strategy, the anchor distribution in dynamic regions
becomes significantly denser compared to the static regions, as
illustrated in (b).
Fig. 5, traditional densification approaches maintained sim-
ilar anchor point densities across both static backgrounds
and dynamic regions. In contrast, our method selectively
increases anchor density in under-reconstructed dynamic ar-
eas, enhancing motion representation while minimizing re-
dundant anchors in static regions. This adaptive allocation
improves the modeling of complex motion while reducing
computational overhead in static areas.
Understanding the temporal injection in Neural Gaus-
sians Generation Network N. To achieve more refined
motion modeling, we incorporate temporal information not
only into the Anchor Deformation Field F but also into
the MLPs within the Neural Gaussians Generation Network
N. This temporal injection in N enables the generation
of neural gaussians with time-varying properties, allowing
our model to capture dynamic scene characteristics more
effectively. As demonstrated in Table 3, the models with
dual temporal information injection outperform the model
without temporal injection in Neural Gaussians Generation
Network N.
Analysis of the deformation of each part. We introduce
separate MLPs (ϕx, ϕl, ϕq) in the Anchor Deformation
Field F to model the temporal changes of anchors, includ-
ing 3D position, scaling factor, and rotation quaternion. As
shown in Table 3, anchor movement plays the most critical
role in fitting dynamic scenes. The scaling factor primar-
ily modulates the anisotropic scaling of neural Gaussians
and the scaling of offsets, serving to simulate the stretching
and twisting of surfaces at the microscopic level during non-
rigid motion processes. The rotation quaternion of anchors
mainly determines the visibility of anchors in the view frus-
tum, which is used to model changes in anchor visibility
during movement.

<!-- page 9 -->
6. Conclusion
This paper introduces SD-GS, an innovative and com-
pact framework designed to represent dynamic scenes. It
achieves high visual quality while significantly reducing
storage costs and improving FPS. Additionally, our densifi-
cation strategy effectively optimizes anchor point distribu-
tion in dynamic scenes to address the challenges of complex
motion reconstruction. Extensive experiments demonstrate
that our model achieves competitive reconstruction qual-
ity on challenging real-world dynamic scene datasets while
substantially reducing model size.
References
[1] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael
Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil
Kim.
Hyperreel:
High-fidelity 6-dof video with ray-
conditioned sampling.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 16610–16620, 2023. 2
[2] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee,
Gun Bang, and Youngjung Uh.
Per-gaussian embedding-
based deformation for deformable 3d gaussian splatting. In
European Conference on Computer Vision, pages 321–335.
Springer, 2024. 2, 6, 7
[3] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 130–141, 2023. 3, 5
[4] Guikun Chen and Wenguan Wang. A survey on 3d gaussian
splatting, 2025. 2
[5] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai.
Hac: Hash-grid assisted context for 3d
gaussian splatting compression. In European Conference on
Computer Vision, 2024. 3
[6] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin,
Yuexin Ma, Wenping Wang, and Xuejin Chen. Gaussianpro:
3d gaussian splatting with progressive propagation. In Forty-
first International Conference on Machine Learning, 2024. 3
[7] CWoong Oh Cho, In Cho, Seoha Kim, Jeongmin Bae,
Youngjung Uh, and Seon Joo Kim.
4d scaffold gaussian
splatting for memory efficient dynamic scene reconstruction,
2024. 2
[8] Xiaobiao Du, Yida Wang, and Xin Yu. Mvgs: Multi-view-
regulated gaussian splatting for novel view synthesis. arXiv
preprint arXiv:2410.02103, 2024. 3
[9] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wen-
zheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting:
towards efficient novel view synthesis for dynamic scenes.
In ACM SIGGRAPH 2024 Conference Papers, pages 1–11,
2024. 2
[10] Guangchi Fang and Bing Wang.
Mini-splatting: Repre-
senting scenes with a constrained number of gaussians. In
European Conference on Computer Vision, pages 165–181.
Springer, 2024. 3
[11] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-
aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.
Fast dynamic radiance fields with time-aware neural vox-
els. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9,
2022. 5
[12] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
Explicit radiance fields in space, time, and appearance. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 12479–12488, 2023. 3
[13] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu,
Yan-Pei Cao, and Xiaojuan Qi.
Sc-gs: Sparse-controlled
gaussian splatting for editable dynamic scenes. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 4220–4230, 2024. 2
[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering.
ACM Transactions on Graphics
(ToG), 42(4):1–14, 2023. 2, 3, 5
[15] Sieun Kim, Kyungjin Lee, and Youngki Lee.
Color-cued
efficient densification method for 3d gaussian splatting. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 775–783, 2024. 3
[16] Sangwoon Kwak, Joonsoo Kim, Jun Young Jeong, Won-
Sik Cheong, Jihyong Oh, and Munchurl Kim. Modec-gs:
Global-to-local motion decomposition and temporal interval
adjustment for compact dynamic 3d gaussian splatting, 2025.
2
[17] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 5521–5531, 2022. 2, 7
[18] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 8508–8520, 2024. 2
[19] Zhuoxiao Li, Shanliang Yao, Yijie Chu, Angel F Garcia-
Fernandez, Yong Yue, Eng Gee Lim, and Xiaohui Zhu.
Mvg-splatting: Multi-view guided gaussian splatting with
adaptive quantile-based geometric consistency densification.
arXiv preprint arXiv:2407.11840, 2024. 3
[20] Stephen Lombardi,
Tomas Simon,
Gabriel Schwartz,
Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. Mix-
ture of volumetric primitives for efficient neural rendering.
ACM Transactions on Graphics (ToG), 40(4):1–13, 2021. 5
[21] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 3, 4, 5, 7
[22] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In 2024 International Con-
ference on 3D Vision (3DV), pages 800–809. IEEE, 2024. 2
[23] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl,
Markus Steinberger, Francisco Vicente Carrasco, and Fer-
nando De La Torre.
Taming 3dgs: High-quality radiance

<!-- page 10 -->
fields with limited resources. In SIGGRAPH Asia 2024 Con-
ference Papers, pages 1–11, 2024. 3
[24] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M Seitz.
Hypernerf:
A higher-
dimensional representation for topologically varying neural
radiance fields. arXiv preprint arXiv:2106.13228, 2021. 2, 7
[25] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An im-
perative style, high-performance deep learning library. Ad-
vances in neural information processing systems, 32, 2019.
7
[26] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer.
D-nerf: Neural radiance fields
for dynamic scenes. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
10318–10327, 2021. 2
[27] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. arXiv
preprint arXiv:2403.17898, 2024. 3
[28] Samuel Rota Bul`o, Lorenzo Porzi, and Peter Kontschieder.
Revising densification in gaussian splatting.
In European
Conference on Computer Vision, pages 347–362. Springer,
2024. 3
[29] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 3, 4
[30] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel
grid optimization: Super-fast convergence for radiance fields
reconstruction. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 5459–
5469, 2022. 5
[31] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang
Zeng. Dreamgaussian: Generative gaussian splatting for effi-
cient 3d content creation. arXiv preprint arXiv:2309.16653,
2023. 2
[32] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei
Song, and Huaping Liu. Mixed neural voxels for fast multi-
view video synthesis. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, pages 19706–
19716, 2023. 2
[33] Yufei Wang, Zhihao Li, Lanqing Guo, Wenhan Yang, Alex C
Kot, and Bihan Wen. Contextgs: Compact 3d gaussian splat-
ting with anchor level context model. In Advances in neural
information processing systems (NeurIPS), 2024. 3
[34] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 7
[35] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 20310–20320, 2024.
2, 3, 6, 7
[36] Shuzhao Xie, Jiahang Liu, Weixiang Zhang, Shijia Ge,
Sicheng Pan, Chen Tang, Yunpeng Bai, and Zhi Wang.
Sizegs: Size-aware compression of 3d gaussians with hier-
archical mixed precision quantization. arXiv, 2024. 3
[37] Shuzhao Xie, Weixiang Zhang, Chen Tang, Yunpeng Bai,
Rongwei Lu, Shijia Ge, and Zhi Wang.
Mesongs: Post-
training compression of 3d gaussians via efficient attribute
transformation. In European Conference on Computer Vi-
sion. Springer, 2024. 3
[38] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin.
Deformable 3d gaussians for
high-fidelity monocular dynamic scene reconstruction. arXiv
preprint arXiv:2309.13101, 2023. 2, 6, 7
[39] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li
Zhang. Real-time photorealistic dynamic scene representa-
tion and rendering with 4d gaussian splatting. arXiv preprint
arXiv:2310.10642, 2023. 2, 6, 7, 8
[40] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong
Dou. Absgs: Recovering fine details in 3d gaussian splat-
ting. In Proceedings of the 32nd ACM International Confer-
ence on Multimedia, pages 1053–1061, 2024. 3
[41] Wang Yifan, Felice Serena, Shihao Wu, Cengiz ¨Oztireli,
and Olga Sorkine-Hornung. Differentiable surface splatting
for point-based geometry processing. ACM Transactions on
Graphics (TOG), 38(6):1–14, 2019. 3
[42] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 7
[43] Weixiang Zhang, Shuzhao Xie, Shijia Ge, Wei Yao, Chen
Tang, and Zhi Wang. Expansive supervision for neural radi-
ance field, 2024. 2
[44] Weixiang Zhang, Shuzhao Xie, Chengwei Ren, Siyi Xie,
Chen Tang, Shijia Ge, Mingzi Wang, and Zhi Wang. Evos:
Efficient implicit neural training via evolutionary selector,
2024. 2
[45] Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and Heng-
shuang Zhao.
Pixel-gs: Density control with pixel-aware
gradient for 3d gaussian splatting. In European Conference
on Computer Vision, pages 326–342. Springer, 2024. 3
[46] M. Zwicker, H. Pfister, J. van Baar, and M. Gross. Ewa vol-
ume splatting. In Proceedings Visualization, 2001. VIS ’01.,
pages 29–538, 2001. 3
[47] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and
Markus Gross.
Surface splatting.
In Proceedings of the
28th annual conference on Computer graphics and interac-
tive techniques, pages 371–378, 2001. 3
