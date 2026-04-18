<!-- page 1 -->
Tessellation GS: Neural Mesh Gaussians for Robust Monocular Reconstruction
of Dynamic Objects
Shuohan Tao1, Boyao Zhou2, Hanzhang Tu2, Yuwang Wang2, and Yebin Liu2*
1University of Cambridge, 2Tsinghua University
Abstract
3D Gaussian Splatting (GS) enables highly photoreal-
istic scene reconstruction from posed image sequences but
struggles with viewpoint extrapolation due to its anisotropic
nature, leading to overfitting and poor generalization, par-
ticularly in sparse-view and dynamic scene reconstruction.
We propose Tessellation GS, a structured 2D GS approach
anchored on mesh faces, to reconstruct dynamic scenes
from a single continuously moving or static camera. Our
method constrains 2D Gaussians to localized regions and
infers their attributes via hierarchical neural features on
mesh faces. Gaussian subdivision is guided by an adaptive
face subdivision strategy driven by a detail-aware loss func-
tion. Additionally, we leverage priors from a reconstruc-
tion foundation model to initialize Gaussian deformations,
enabling robust reconstruction of general dynamic objects
from a single static camera, previously extremely challeng-
ing for optimization-based methods. Our method outper-
forms previous SOTA method, reducing LPIPS by 29.1%
and Chamfer distance by 49.2% on appearance and mesh
reconstruction tasks.
1. Introduction
Reconstruction of observed scenes has always been a ma-
jor challenge in computer vision. With the spark of dif-
ferentiable rendering, the task has shifted from solely rely-
ing on special equipments like depth cameras and LiDARs
to leveraging more accessible multi-camera or monocular
video setup. However, most methods [18, 29, 30] focus on
static scenes due to the inherent under-determinacy of dy-
namic scene reconstruction and non-rigid deformation.
Early approaches [4, 31] employed image warping to
preserve observed information for novel view synthesis or
physical plausibility regularization in unobserved regions
for geometry reconstruction. Recent monocular differen-
tiable rendering methods incorporate geometric constraints
*Corresponding author (liuyebin@mail.tsinghua.edu.cn).
alongside photometric loss with implicit NeRF [11, 15, 28,
47] or explicit Gaussian Splatting [24, 25, 44] represen-
tation. Such ”4D” methods leverages spectral-biasedness
of MLP [5, 44, 47] or spatial-temporal factorization [3, 6,
33, 44] but relies on explicit or implicit (camera move-
ment) multi-view input.
As pointed out by [7], existing
differentiable rendering approaches are highly susceptible
to view overfitting, and using unrealistic camera motions
in datasets D-NeRF [36] and DG-Mesh [25]. On the other
hand, avatar-like methods [10, 13, 22, 49] enable per-pose
Gaussian attributes modeling for human performance re-
construction, which heavily rely on category-specific tem-
plate, e.g. SMPL [26]. Thus, such methods are not able
to handle topological changes and category-agnostic recon-
struction.
In terms of template-free method, DG-Mesh [25],
TiNeuVox [6], and HexPlane [3] pose various structural and
loss designs. DG-Mesh utilized Laplacian regularization to
ensure smoothness of the reconstructed geometry. TiNeu-
Vox and HexPlane factorizes spatiotemporal information
into several feature planes that are decoded by MLP. Al-
though effective for input video with enough camera move-
ment, they do not explicitly regularize view-overfitting, and
as a result, they can not recover dynamic details from gen-
eral monocular video input and degrade synthesis quality
for novel views far from input view, due to the lack global
cue of moving object.
To bridge this gap, we propose to use a template-free ge-
ometry prior to anchor our surfel-like Gaussian attributes.
To obtain such prior, we apply large reconstruction model
(LRM) to extract coarse geometry frame by frame. Such
geometry is in a relatively low resolution without high-
frequency details and correspondence between them is un-
known. Therefore, we propose to learn a deformation field
with control points whose movements are defined by a mo-
tion MLP to build the correspondence between reference
frame and other frames and to preserve temporal consis-
tency by doing a sequential optimization. In this process,
we design a robust Chamfer loss to overcome the flicker-
ing geometry and floating artifacts from LRM output. We
1
arXiv:2512.07381v1  [cs.CV]  8 Dec 2025

<!-- page 2 -->
only optimize deformation field in this stage, while further
anchor 2D Gaussian attributes for joint optimization of de-
formation and appearance in stage 2. Since the deforma-
tion field is established in stage 1, we define all Gaussian
points on the geometry surface of reference frame and learn
Gaussian attributes with appearance decoders and Gaussian
features defined on mesh vertices. To further enhance the
high-frequency appearance modeling, we propose a novel
hierarchical structure of Gaussian points, which follows a
structural densification mechanism instead of the gradient-
based splitting in original Gaussian Splatting. Attaching
surfel-like Gaussians on geometry triangles, our appearance
decoders are able to optimize both geometry and appear-
ance with solely input images. Finally, we achieve accurate
motion reconstruction, temporally consistent geometry, and
photo-realistic appearance modeling within 90 minutes for
500 frames.
In conclusion, our main contributions are as follows:
• Our method is capable of reconstructing dynamic object
from a single monocular video under challenging camera
setup, for category-agnostic objects.
• We leverage LRM to prepare per-frame coarse geometry
prior and further propose a deformation MLP to build the
correspondence of frames and to faithfully match the dy-
namic information of the input video.
• We proposed a novel mesh Gaussian structure that pro-
vides higher fidelity appearance, lower memory burden,
and higher training speed. We also proposed two con-
straints for mesh Gaussians to avoid view-overfitting.
2. Related Work
Differentiable Rendering NeRF [23] was proposed to rep-
resent a static scene with density and color volumes de-
fined by an MLP. The training process is time-consuming
as it performs costly numerical integration along camera
rays at every training iteration. Recently, a more efficient
method, 3D Gaussian Splatting [18] was proposed. It rep-
resents a scene as many anisotropic 3D Gaussian volumes
with tractable integrations. It achieved great speedup com-
pared to NeRF. However, due to the anisotropic nature of
3D Gaussians, they are prone to overfitting to camera views,
often elongating along the camera ray. This results in arti-
facts when rendered from novel viewpoints, particularly in
sparse-view regions where multi-view supervision is weak.
Scaffold GS [27] partially solved the overfitting issue by
encoding local geometric structures in compact neural fea-
tures. 2D GS [14] proposed to set one of the axis of 3D
GS to have almost zero scale in order to model geometri-
cal details more accurately. Building on top of that, mesh
based GS [8, 9] anchor Gaussian Splats to mesh faces, with
the Gaussian normal direction aligned with the mesh faces
either through a loss term or a hard constraint. An underly-
ing geometric representation makes them less susceptible to
overfitting. Although they thrive in multi-view reconstruc-
tion tasks, they still suffer from view-overfitting in monoc-
ular reconstruction tasks, even worse on dynamic scenes.
Dynamic Reconstruction Recent works have used both
NeRF and 3D GS for reconstructing dynamic scenes. NeRF
based methods [3, 6, 21, 33, 34, 36] usually uses a time-
conditioned or per-frame embedding conditioned NeRF to
model time-varying appearance of scenes. 3D GS based
methods [28, 45, 47] also uses time-conditioned deforma-
tion models to offset a canonical set of 3D GS points to
respective timesteps to match the captured images. These
works have all achieved incredible results on existing
monocular video dataset of dynamic objects.
Nonetheless, most of these methods are not robust
against novel-view rendering, and most existing datasets
either provide effectively multi-view input or use testing
views that’s not too far from the training view, as pointed
out by [7]. There are two levels of difficulties here:
• Lack of geometric grounding. Without an explicit ge-
ometric representation, photometric appearances remain
unconstrained, reducing the reconstruction to mere col-
ored points in space.
• Weak coupling between geometry and appearance.
Even when a geometric representation is present, exist-
ing appearance models are not strictly constrained by it,
leading to potential view-overfitting.
As a result, most successful and deployable monocular re-
construction methods either rely on class-specific templates
(e.g., SMPL) [16, 20, 32, 35, 37–39, 43] or require addi-
tional depth input to compensate for missing multi-view in-
formation.
3. Method
Reconstructing both geometry and appearance of objects
from slowly moving cameras is challenging due to the am-
biguity in motion of unobserved region, view overfitting
tendencies of differentiable rendering methods, and lack of
geometric information from camera movement. Our model
is able to solve the challenge by performing a 2-stage opti-
mization process. In the first stage shown in Fig. 1 (a) and
(b), we designed a robust framework to extract motion and
geometry information from unstable LRM output defined
by a canonical mesh and a control-point based deformation
model. In the second stage in Fig. 1 (c), structured 2D GS
will be initialized on the canonical mesh. We express 2D GS
as functions of vertex neural features to keep an expressive
and compact representation. Robustness to view-overfitting
is achieved by avoiding GS occlusion and constraining GS
influence to local mesh faces. We also include a carefully
designed adaptive GS subdivison mechanism to automati-
cally add new GS to regions with fine photometric or geo-
metric details through a mesh-Gaussian quad tree.
2

<!-- page 3 -->
Optimization
Subdivision
(c) Mesh Neuaral Gaussian Derivation
Natural Input Camera Motion
(a) Mesh Initialization
rgb
xy
cov
Mesh Face
Mesh Face
Subdivision
AB
ABA
ABB
ABC
ABD
AC
ACA
ACB
ACC
ACD
z offset
Child
1
Blended Opacity
Parent 
Opacity
1
0
Parent
Reconstruction
Foundation 
Model
Initialize Control Points
Anchor 
Neural Gaussians
...
Interpolated Vertex Normal
Deform
AB
AC
AD
AE
A
Physical
Mesh Face
Parent Neural
Gaussian Face
Child Neural
Gaussian Face
...
(b) Template Extraction
Neural
Gaussian Face
Canonical Mesh
Robust Novel View Rendering
Stage 1
Stage 2
Figure 1. Illustration of pipeline. In stage one, we get per-frame mesh sequence from LRM by querying each frame. We fix the mesh
by Taubin smoothing [40] and subdivide faces or collapse edges until the number of faces reaches our desired initial number of Gaussians.
In stage two, we initialize 2D Gaussians defined by neural features on the canonical mesh. We train the neural Gaussians jointly with the
deformation model. The resulting Gaussians are extremely robust to view-overfitting.
…
→
→
(a)
(b)
(c)
…
Figure 2. Adaptive densification via mesh-Gaussian quad tree
on a single mesh face. Red triangles are leaf nodes of whose as-
sociated Gaussians will not further subdivide. Blue triangles are
non-leaf nodes with no associated Gaussians. (a) and (b): the tree
allows for adaptive density of Gaussians. (c): learnable subdivi-
sion ratio further improves the expressiveness.
3.1. Stage One: Data Driven Template
In the first stage, we extract 3D template and motion infor-
mation from LRM [12, 42] to form a 4D template. We have
LRM output corresponding meshes to each frame of the in-
put video and choose one (in our experiments simply the
first mesh) to be canonical mesh. We then perform ICP (It-
erative Closest Point) between canonical mesh and each of
the meshes in the sequence. Then similar to BANMo [46],
our deformation model initializes control points and opti-
mize their positions and a time-conditioned MLP to drive
their motions. During this process, we jointly optimize the
skinning weight of mesh vertices to drive the mesh.
wn,k = MLPweight,k(Ck −vn,t0) + ϕ(||Ck −vn,t0||)
ck,t = MLPmotion,k(t), vn,t = vn,t0 +
K
X
k=1
ck,twn,k
(1)
In the above equation where we derived the final per vertex
location vn,t at time t, MLPweight,k is a per control point
MLP that output weights taking as input displacement from
control point to canonical vertices, ϕ is an isotropic Gaus-
sian kernel for radial basis function whose scales are deter-
mined by the average nearest neighbor distance of the con-
trol points at initialization, wn,k is the composite skinning
weight between the nth vertex vn,t0 and the kth control point
Ck. MLPmotion,k is a per control point MLP that takes as
input time and output control point displacement ck,t, and
vn,t0 is per vertex location at canonical time.
LRCD =
1
|V |
X
v∈V
min

d2, min
u∈U ∥v −u∥2

+ 1
|U|
X
u∈U
min

d2, min
v∈V ∥u −v∥2

Ltotal = wlapLlap + wnLn + LRCD
(2)
In the above equation, we proposed robust Chamfer loss
LRCD, where U and V are sets of vertices in target and
source meshes, d is truncation distance, Llap is mesh Lapla-
cian regularization, Ln is normal consistency regulariza-
tion, and LRCD is our proposed robust Chamfer loss; wlap
and wn are weights for respective loss terms. We optimize
the deformation model against Ltotal. Please refer to the
supplementary material for more detail about stage one.
3.2. Stage Two: Tessellation Gaussian Splatting
In the second stage, we solely use the ground-truth frames to
jointly optimize motion information and appearance defined
by GS. Traditional GS doesn’t perform well when trained
on sparse views as they could overfit to training cameras.
We have identified 2 major reasons. First, Gaussians could
elongate along the camera rays freely without hurting pho-
tometric performance. In addition, when large scale defor-
mation happens on non-visible regions from training views,
3

<!-- page 4 -->
unsupervised occluded Gaussians could appear due to the
separation of the occluding Gaussians, resulting in arti-
facts. We show the examples of these two types of artifacts
in our ablation study shown in Fig. 6. We therefore pro-
pose a novel 2D GS architecture, mesh-Gaussian quad tree,
that both avoids Gaussian occlusion and scale overfitting by
constraining their locations and scales to local structured
triangles on mesh faces to minimize Gaussian overlapping.
3.2.1
Structured 2D Gaussians
We build mesh-Gaussian quad tree as demonstrated in Fig. 1
(c) and Fig. 2.
At initialization, each mesh face is also
a parent Gaussian face. Each parent face is divided into
4 smaller child faces connecting its 3 edge centers similar
to Mesh-GS [8], forming a quad tree shown in Fig. 2 (a)
and (b). However, instead of fixing the subdivision points
like MeshGS, we introduce learnable edge point ratios s1,
s2, and s3 to allow these points to slide along the edges to
match local textures, as in Fig. 2 (c). A large parent Gaus-
sian is spawned at the center of the parent face, while 4
child Gaussians are spawned at the center of child faces.
To maintain clarity, we refer to both parent faces and child
faces as Gaussian faces, distinguishing them from the un-
derlying physical mesh faces, whose structure remains un-
changed throughout the training. This hierarchical struc-
ture enables adaptive refinement of Gaussians, resulting in
a compact and expressive representation.
Each parent face has 6 learnable features fi ∈R128: 3 par-
ent features are defined on the vertices, and 3 edge features
on the edges. Each Gaussian has a learnable barycentric
weight r ∈R3 to interpolate from the 3 vertices to decide
their barycentric coordinate. They have a separate set of
barycentric weights c ∈R3 to interpolate from the 3 ver-
tex features to get their neural features. For both parent and
child Gaussians, features are interpolated as follows:
finterpolated = softmax(r) · [f1||f2||f3]
(3)
where f1, f2, and f3 are vertex features of the faces each
Gaussian belong to, and || is the concatenation operator. For
parent faces, vertex features are directly stored. For child
faces, for example the yellow triangular face in Fig. 1, its
vertex features are calculated as:
f1 = fparent1 · (1 −s1) + fparent3 · s1 + fedge2
f2 = fparent2 · s2 + fparent3 · (1 −s2) + fedge3
(4)
and f3 equals fparent3. We interpolate parent vertex fea-
tures fparent along edges by our learnable shape proportion
s and add the edge features fedgei on top of that. We allow
child faces to share vertex features with parent face where
they share a common vertex. We decode Gaussian features
with 3 MLP appearance decoders:
MLPqs([finterpolated||e2
e1||e3
e1]) = [q2D||s2D]
MLPc([f||p]) = rgb
MLPoffset(finterpolated) = zoffset
(5)
where q2D||s2D is the 2D GS’s rotation and scale concate-
nated. zoffset is the GS offset along the face normal di-
rection, e1, e2, and e3 are the edge lengths of each Gaus-
sian face. To model pose dependent apperance, we encode
the location of the 30 control points into a pose embedding
p, similar to Gaussian Avatar [13] and Animatable Gaus-
sians [22]. 2D GS’s normal direction is interpolated from
vertex normals, and we similarly interpolate vertex colors
to add to the decoded neural colors. Further GS constraints
are described in Sec. 3.2.3.
3.2.2
Competitive Gaussian Opacities
We utilized Gaussian opacities for gradient-free Gaussian
subdivision. Specifically, only parent Gaussians have opti-
mizable opacities as in Fig. 3, while the four child Gaus-
sians compete their opacities with their parents, as com-
puted in Eq. (6), where we have heuristically set β to be 0.9.
Importantly, β ensures the sum of a parent Gaussian’s opac-
ity and any of its child Gaussians’ opacities never equals 1
except when the parent Gaussian’s opacity is either 0 or 1,
as plotted in Fig. 1. In such cases, the child Gaussians’
opacities become either 0 or 1, effectively forcing binary
opacity assignments.
αchild = (1 −αβ
parent)
1
β
(6)
With this setup, each parent Gaussian’s opacity serves as
an indicator of local detail level. The regularization term
Lα in Eq. (11), which reaches its minimum only when all
parent Gaussians are fully opaque, competes with the pho-
tometric and geometric loss terms. When the other loss
terms exceed a certain threshold, indicating the presence
of finer local details, parent Gaussians’ opacities tend to
zero, allowing child Gaussians to emerge and model these
finer details. This mechanism adaptively allocates Gaus-
sian population, ensuring that regions with higher complex-
ity receive higher-resolution representations, while simpler
areas remain efficiently represented by fewer and larger par-
ent Gaussians, thereby maintaining an adaptive and efficient
Gaussian hierarchy.
3.2.3
Gaussian Constraints
We propose two constraints on the Gaussians to avoid view-
overfitting. First, Gaussian scales are constrained to a maxi-
mum of one-fourth of the base and height of their respective
4

<!-- page 5 -->
+
→
+
or
or
Figure 3. Learnable subdivision ratio fits boundary better. Yel-
low Gaussian is a parent Gaussian, blue Gaussians are child Gaus-
sians. Color boundary denoted by red and blue regions can be
better modeled by child Gaussians than parent. Child Gaussians’
opacities will naturally become one through optimization.
triangles by applying a sigmoid activation to the decoded
s2D in Eq. (5). Gaussians are also allowed to rotate around
normals by the decoded rotation offset q2D. This way, the
Gaussians are initialized with just enough scale to span the
surface which naturally minimizes overlapping. As the sub-
division mechanism progresses, new Gaussians are dynam-
ically introduced to fill regions where the initial Gaussians
fail to provide sufficient coverage.
In addition, for Gaussian offset along the face normal
direction, rather than employing a soft Gaussian anchor-
ing constraint as in DG-Mesh [25], we introduce a scale-
dependent anchoring strategy. Gaussians associated with
smaller neural faces are permitted to have larger offsets
while initial Gaussians with parent being mesh faces have
zero offset, judged by the value:
u = tanh(ep
eg
)
(7)
where ep and eg are the mean edge length of the root mesh
face and the mean edge length of the Gaussian face. How-
ever, no offset is allowed to exceed the mean edge length of
the root physical mesh face, ensured by
wbar = (1 −w1)(1 −w2)(1 −w3)
offset = wbar · u · ep · tanh(zoffset)
(8)
where w1, w2, and w3 are the barycentric coordinates of
Gaussian points with respect to their root mesh faces.
This ensures that finer geometric details are only carved
after the object’s motion has already been well optimized,
which prevents premature deformations that could lead to
inconsistent geometry.
This constraint also ensures that
strong gradient will flow to mesh vertices to optimize ge-
ometric details through photometric losses. The effective-
ness of the contraints are shown in Sec. 4.3.
3.2.4
Adaptive Gaussian Population Control
Thanks to our proposed adaptive mesh subdivision process,
our neural GS can model very fine level of details with
our mesh-Gaussian quad tree, as illustrated by Fig. 1. Our
model subdivides a parent Gaussian by turning its 4 child
Gaussians into 4 new parent Gaussians, making each child
face a new parent face. During this process, we remove the
original parent Gaussian, compute its three edge features,
and reassign them as new vertex features. The newly in-
troduced edge features are initialized to zero, minimizing
disruption to the optimization process and allowing us to
constantly introduce new Gaussians in a stable manner dur-
ing training. Edge features of each new parent faces are
initialized to zero, minimizing the impact on optimization
process and incrementally introducing new Gaussians in a
stable manner during training. Our subdivision mechanism
is implemented as follows:
• Parent Gaussian Subdivision:
All parent Gaussians
with opacities below 0.1 are subdivided every 5000 itera-
tions except for the initial and last 5000 iterations, ensur-
ing denser Gaussians in finer regions.
• Child Gaussian Deactivation: Excessive child Gaus-
sians whose parent opacities remain above 0.9 in 90% of
the iterations since the last subdivision are turned off, and
we also exclude them when calculating opacity regular-
ization Lα.
3.2.5
Loss Terms
Our losses are as follows:
Lpho = L1 + Lssim
(9)
where we used L1 and SSIM loss between GT image and
rendered image to supervise photometric appearance,
Ledge = 1
|E|
X
(i,j)∈E
(∥vi,t0 −vj,t0∥−∥vi,t −vj,t∥)2 (10)
where we penalize edge length changes with respect to
canonical mesh, vi,t0 is the canonical vertex coordinate and
vi,t is its coordinate at timestep t. E is the set of neighbor-
ing vertices.
Lα = 1
N
N
X
i
αi
(11)
The above Lα is the mean opacity of all Gaussians. Due to
our parent-child opacity coupling, its minimum is when all
parents are fully opaque. Our final loss LT GS is:
Lreg = Llap + wedgeLedge + wαLα
Lprior = wflowLflow + wnormalLnormal
LT GS = Lpho + wregLreg + wpriorLprior
(12)
where Lflow is the difference between the rendered opti-
cal flow and the optical flow predicted by RAFT [41]. Llap
is mesh Laplacian regularization further explained in sup-
plementary material. We use normal supervision Lnormal
provided by DSINE [2] only for static cameras.
5

<!-- page 6 -->
Ours
TiNeuVox
DG-Mesh
HexPlane
Input






	




Figure 4. Test results on Smooth D-NeRF. (a) and (b): input images at two different timesteps. (c), (e), (g), and (i): rendering results at
the first timestep. (d), (f), (h), and (j): rendering results at the second timestep. Our results are visually better than all other methods. The
second best results are produced by DG-Mesh [25]. Their 3D Gaussians’ unconstrained scales result in foggy appearance caused by large
Gaussian floaters.
(a)
(b)
(c)
Figure 5. Unbiased4d [17] results. (a): input image. (b): ren-
dered novel view. (c): extracted mesh.
4. Experiments
4.1. Datasets
Original D-NeRF dataset, which employs teleporting cam-
eras to simulate multi-view supervision, is impractical for
real-world applications. Thus we re-render D-NeRF’s hu-
manoid characters1 with naturally moving cameras.
We
rotate the camera around the character at constant speed
1Available on Adobe’s Mixamo.
and return camera to the starting position as the animation
stops. We place 2 test cameras rotating with the training
camera, both looking at the character but are 45 degrees
away from the training camera. Further illustration of the
setup is available in the supplementary material. In addi-
tion to blender rendered dataset, we qualitatively tested our
model on Unbiased4D [17], including one video of a de-
forming cactus toy captured using a hand-held camera. We
also tested on People-Snapshot [1] comparing against Gaus-
sian Avatar [13], featuring self-rotating actors captured with
a static camera. We train our model on a single 3090 GPU
with Adam [19] and PyTorch. The whole training pipeline
takes about 90 minutes.
4.2. Main Results
Our main results on Smooth D-NeRF are demonstrated in
Fig. 4, Fig. 8, and Sec. 4.
We achieved better results
than all compared methods.
Since TiNeuVox and Hex-
Plane lack an underlying geometric representation, they ex-
hibit severe overfitting to training views. DG-Mesh, while
more robust due to mesh-anchored Gaussians, still suffers
from mild view-overfitting due to its use of vanilla 3D GS.
However, the ”mutant” data features a relatively stationary
motion, and our model’s flexible motion representation is
disadvantaged but still achieved better perceptive quality
measured by LPIPS. Shown in Fig. 8, DG-Mesh generates
noisy meshes due to its reliance on SfM to initialize meshes,
6

<!-- page 7 -->
Current View
Closest Training View
Novel View
w./o.
w./
w./o.
w./
Novel View
Current View
Closest Training View
w./o.
w./o.
w./o.
w./o.
t1
t3
t2
t4
l.c.
l.c.
l.c.
l.c.
l.c.
l.c.
l.c.
l.c.
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(h)
Figure 6. Ablation comparison of locality constraints (l.c.). (a) and (e): ablation rendering results from training pose at current timestep.
(b) and (f): ablation rendering results from test pose. (c) and (g): rendering results from test pose without ablation. (d) and (h): spatially
closest training views to test views at t1 and t2. (b) demonstrated view-overfitting caused by occluded Gaussians not being optimized, as
shown by the large red Gaussian. Comparison between (f) and (g) shows our pipeline avoids 2D GS scales overfitting to the training view.
(a) and (e) shows that these artifacts are not be visible to training views but will appear in novel test views.
Method
Hook
Mutant
PSNR ↑
SSIM ↑
LPIPS ↓
CD ↓
PSNR ↑
SSIM ↑
LPIPS ↓
CD ↓
TiNeuVox-B
13.996
0.859
0.1803
-
16.380
0.914
0.1801
-
HexPlane
27.836
0.966
0.0297
-
17.632
0.880
0.1126
-
DG-Mesh
26.123
0.968
0.0506
2
18.506
0.879
0.1057
2.8
Ours
32.302
0.981
0.0208
0.8
17.104
0.882
0.1017
3.8
Method
Standup
Jumping Jacks
PSNR ↑
SSIM ↑
LPIPS ↓
CD ↓
PSNR ↑
SSIM ↑
LPIPS ↓
CD ↓
TiNeuVox-B
12.583
0.839
0.2446
-
16.417
0.915
0.1799
-
HexPlane
21.256
0.898
0.0936
-
24.977
0.940
0.0770
-
DG-Mesh
22.481
0.937
0.0875
1.1
25.433
0.962
0.0665
7
Ours
23.464
0.942
0.0572
0.9
25.995
0.964
0.0404
1.1
Table 1. Quantitative comparison against previous works on Smooth D-NeRF. The best method in each metric is bolded. Chamfer distance
(CD) is reported in scale 1e−3 and is unitless, same as training data of LRM.
(a)
(b)
(c)
(d)
(e)
(f)
Figure 7. Comparison with LRM direct output. (a) and (d):
ground-truth. (b) and (e): LRM direct output. (c) and (f): our
pipeline’s output. Further comparison available in supplementary
material.
which performs poorly when object motion dominates over
camera motion, and its direct use of time-conditioned MLPs
for deformation, which is under-constrained. Our method
produces accurate meshes with sharp details both because
we initialize meshes directly from 3D priors and our use of
merely 30 learnable control points to drive our meshes, pro-
viding strong local rigidity guarantee. We also show side-
by-side comparison with LRM’s direct rendering result in
Fig. 7, and our model refines both geometry and appearance
significantly. In addition, our results shown in Fig. 5 and
Fig. 9 shows our model is robust against in-the-wild camera
motion and static camera, achieving similar performance to
Gaussian Avatar [13].
4.3. Ablation Studies
4.3.1
Effectiveness of Pruning Strategy
We tested our pipeline without child Gaussian pruning
mechanism.
As shown in Sec. 4.3.2, the improvement
in PSNR is minimal, suggesting excessive Gaussians that
don’t contribute much to the photometric quality have in-
deed be pruned. The number of Gaussians, training time,
and GPU memory usage also become much worse without
child Gaussian pruning.
4.3.2
Effectiveness of Locality Constraints
We allow the 2D Gaussians to freely optimize their at-
tributes, only keeping their normal direction aligned to the
interpolated face normal and scale proportional to face size,
7

<!-- page 8 -->
GT
DG-Mesh
Ours
GT
DG-Mesh
Ours
Figure 8. Qualitative comparison of mesh reconstruction. Our meshes contain much more geometric details due to the strong correlation
between the photometric appearance and geometries brought by the Gaussian offset constraint.
similar to Mesh-GS [8].
Shown in Fig. 6, two types of
artifacts occur: occluded Gaussians not being optimized,
shown by the large red Gaussian in Fig. 6(b) that’s not vis-
ible to any training view, and thin long Gaussians shown in
Fig. 6(f) being Gaussians freely elongate along viewing di-
rection. Our model is able to suppress both artifacts with
our scale locality constraints.
PSNR
CD
Num GS
Time
Memory
w./o. prune
24.860
-
∼750k
∼240 mins
23GB
w./o. offset
24.537
2.53
-
-
-
Full model
24.716
1.65
∼40k
∼80 mins
6GB
Table 2. Ablation comparison with and without offset constraint
and child Gaussian pruning. CD reported in 1e−3 scale.
4.3.3
Effectiveness of Offset Constraints
To test the effectiveness of the offset constraint discussed in
Sec. 3.2.3, we remove the constraint and let the Gaussians
freely offset along the face normal. We tested the average
chamfer distance of the mesh sequences under this setup,
and as shown in Sec. 4.3.2, the chamfer distance is higher.
As shown in Fig. 10, the generated mesh seems coarser
and lacks fine details, which suggests our offset constraint
indeed allows for more accurate geometric reconstruction
from photometric supervision.
5. Conclusion
We present Tessellation GS in this work, a pipeline to recon-
struct dynamic objects using mesh and novel structured neu-
ral Gaussians with robust view extrapolation performance
from natural monocular videos. Tessellation GS distributes
2D Gaussians on mesh faces in a structured way with min-
imal overlapping and decode their attributes from neural
features on the vertices.
Their population is adaptively
controlled through carefully designed mesh-Gaussian quad
trees linked by Gaussian opacities that adaptively add more
Gaussians in regions with finer details. With scales strongly
linked and constrained by mesh face shapes, Tessellation
(a)
(b)
(c)
(d)
Figure 9. People-Snapshot results. (a): input view rendering re-
sult. (b): our method’s novel view rendering result. (c): Gaussian
Avatar’s novel view rendering result. (d): our extracted mesh.
(a)
(b)
Figure 10. Comparison of meshes extracted with and without
offset constraint. (a): mesh with Gaussian offset constraint. (b):
mesh without Gaussian offset constraint.
GS has shown excellent robustness to view-overfitting and
achieved SOTA performance on monocular reconstruction
task using natural camera motion in terms of both photo-
metric performance and mesh quality.
Limitation
Our method relies on the quality of canonical mesh to
handle topological changes. Further studies towards mesh
merge and division with consistent mesh face correspon-
dence could consider incorporating Tessellation GS for ap-
pearance representation.
8

<!-- page 9 -->
References
[1] Thiemo Alldieck, Marcus Magnor, Weipeng Xu, Christian
Theobalt, and Gerard Pons-Moll. Video based reconstruction
of 3d people models. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 8387–
8397, 2018. 6
[2] Gwangbin Bae and Andrew J Davison. Rethinking induc-
tive biases for surface normal estimation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 9535–9545, 2024. 5
[3] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In CVPR, pages 130–141, 2023. 1,
2
[4] Shenchang Eric Chen and Lance Williams. View interpo-
lation for image synthesis. In SIGGRAPH, pages 279–288,
1993. 1
[5] Devikalyan Das, Christopher Wewer, Raza Yunus, Eddy
Ilg, and Jan Eric Lenssen. Neural parametric gaussians for
monocular non-rigid object reconstruction. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 10715–10725, 2024. 1, 11
[6] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-
aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.
Fast dynamic radiance fields with time-aware neural voxels.
In SIGGRAPH Asia, pages 1–9, 2022. 1, 2
[7] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell,
and Angjoo Kanazawa. Monocular dynamic view synthesis:
A reality check. Advances in Neural Information Processing
Systems, 35:33768–33780, 2022. 1, 2
[8] Lin Gao, Jie Yang, Bo-Tao Zhang, Jia-Mu Sun, Yu-Jie Yuan,
Hongbo Fu, and Yu-Kun Lai. Mesh-based gaussian splatting
for real-time large-scale deformation. CoRR, 2024. 2, 4, 8
[9] Antoine Gu´edon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh reconstruc-
tion and high-quality mesh rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5354–5363, 2024. 2
[10] Chen Guo, Tianjian Jiang, Xu Chen, Jie Song, and Otmar
Hilliges. Vid2avatar: 3d avatar reconstruction from videos in
the wild via self-supervised scene decomposition. In CVPR,
pages 12858–12868, 2023. 1
[11] Zhiyang Guo, Wengang Zhou, Li Li, Min Wang, and
Houqiang Li. Motion-aware 3d gaussian splatting for effi-
cient dynamic scene reconstruction. IEEE Transactions on
Circuits and Systems for Video Technology, 2024. 1
[12] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou,
Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao
Tan. Lrm: Large reconstruction model for single image to
3d. In The Twelfth International Conference on Learning
Representations. 3
[13] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao
Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie.
Gaussianavatar: Towards realistic human avatar modeling
from a single video via animatable 3d gaussians. In CVPR,
pages 634–644, 2024. 1, 4, 6, 7
[14] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 2
[15] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu,
Yan-Pei Cao, and Xiaojuan Qi.
Sc-gs: Sparse-controlled
gaussian splatting for editable dynamic scenes. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 4220–4230, 2024. 1
[16] Wei Jiang, Kwang Moo Yi, Golnoosh Samei, Oncel Tuzel,
and Anurag Ranjan. Neuman: Neural human radiance field
from a single video. In ECCV, pages 402–418, 2022. 2
[17] Erik Johnson, Marc Habermann, Soshi Shimada, Vladislav
Golyanik, and Christian Theobalt. Unbiased 4d: Monocular
4d reconstruction with a neural deformation model. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 6598–6607, 2023. 6
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM TOG, 42(4):1–14, 2023. 1, 2
[19] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization.
arXiv preprint arXiv:1412.6980,
2014. 6
[20] Youngjoong Kwon, Dahun Kim, Duygu Ceylan, and Henry
Fuchs. Neural human performer: Learning generalizable ra-
diance fields for human performance rendering. NeurIPS,
34:24741–24752, 2021. 2
[21] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
and Zhaoyang Lv. Neural 3d video synthesis from multi-
view video. In CVPR, pages 5521–5531, 2022. 2
[22] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Ani-
matable gaussians: Learning pose-dependent gaussian maps
for high-fidelity human avatar modeling. In CVPR, pages
19711–19722, 2024. 1, 4
[23] Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai,
Hujun Bao, and Xiaowei Zhou.
Efficient neural radiance
fields for interactive free-viewpoint video. In SIGGRAPH
Asia, pages 1–9, 2022. 2
[24] Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao.
Gaussian-flow: 4d reconstruction with dynamic 3d gaus-
sian particle. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21136–
21145, 2024. 1
[25] Isabella Liu, Hao Su, and Xiaolong Wang. Dynamic gaus-
sians mesh: Consistent mesh reconstruction from monocular
videos. CoRR, 2024. 1, 5, 6, 12
[26] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard
Pons-Moll, and Michael J Black. Smpl: A skinned multi-
person linear model. ACM TOG, 34(6):1–16, 2015. 1
[27] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 2
[28] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In 3DV, 2024. 1, 2
9

<!-- page 10 -->
[29] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, pages 405–421, 2020. 1
[30] Richard A Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J. Davison, Push-
meet Kohli, Jamie Shotton, Steve Hodges, and Andrew
Fitzgibbon. Kinectfusion: Real-time dense surface mapping
and tracking. IEEE international symposium on mixed and
augmented reality., pages 127–136, 2011. 1
[31] Richard A Newcombe, Dieter Fox, and Steven M Seitz.
Dynamicfusion: Reconstruction and tracking of non-rigid
scenes in real-time. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 343–352,
2015. 1
[32] Xiao Pan, Zongxin Yang, Jianxin Ma, Chang Zhou, and Yi
Yang. Transhuman: A transformer-based human represen-
tation for generalizable neural human rendering. In ICCV,
pages 3544–3555, 2023. 2
[33] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien
Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
Martin-Brualla. Nerfies: Deformable neural radiance fields.
In Proceedings of the IEEE/CVF international conference on
computer vision, pages 5865–5874, 2021. 1, 2
[34] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M Seitz.
Hypernerf:
a higher-
dimensional representation for topologically varying neural
radiance fields. ACM Transactions on Graphics (TOG), 40
(6):1–12, 2021. 2
[35] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang,
Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body:
Implicit neural representations with structured latent codes
for novel view synthesis of dynamic humans.
In CVPR,
pages 9054–9063, 2021. 2
[36] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer.
D-nerf: Neural radiance fields
for dynamic scenes. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
10318–10327, 2021. 1, 2
[37] Ruizhi Shao, Liliang Chen, Zerong Zheng, Hongwen Zhang,
Yuxiang Zhang, Han Huang, Yandong Guo, and Yebin Liu.
Floren: Real-time high-quality human performance render-
ing via appearance flow using sparse rgb cameras. In SIG-
GRAPH Asia, pages 1–10, 2022. 2
[38] Ruizhi Shao, Hongwen Zhang, He Zhang, Mingjia Chen,
Yan-Pei Cao, Tao Yu, and Yebin Liu. Doublefield: Bridging
the neural surface and radiance fields for high-fidelity human
reconstruction and rendering. In CVPR, pages 15872–15882,
2022.
[39] Xin Suo, Yuheng Jiang, Pei Lin, Yingliang Zhang, Minye
Wu, Kaiwen Guo, and Lan Xu. Neuralhumanfvv: Real-time
neural volumetric human performance rendering using rgb
cameras. In CVPR, pages 6226–6237, 2021. 2
[40] G. Taubin. Curve and surface smoothing without shrinkage.
In Proceedings of IEEE International Conference on Com-
puter Vision, pages 852–857, 1995. 3
[41] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field
transforms for optical flow. In ECCV, pages 402–419, 2020.
5
[42] Dmitry Tochilkin, David Pankratz, ZeXiang Liu, Zixuan
Huang, Adam Letts, Yangguang Li, Ding Liang, Christian
Laforte, Varun Jampani, and Yan-Pei Cao. Triposr: Fast 3d
object reconstruction from a single image. CoRR, 2024. 3
[43] Chung-Yi Weng,
Brian Curless,
Pratul P Srinivasan,
Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Hu-
mannerf: Free-viewpoint rendering of moving people from
monocular video. In CVPR, pages 16210–16220, 2022. 2
[44] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In CVPR, pages 20310–20320, 2024. 1
[45] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 20310–20320, 2024. 2
[46] Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ra-
manan, Andrea Vedaldi, and Hanbyul Joo. Banmo: Building
animatable 3d neural models from many casual videos. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 2863–2873, 2022. 3
[47] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction.
In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 20331–20341, 2024. 1, 2
[48] Yuxin Yao, Siyu Ren, Junhui Hou, Zhi Deng, Juyong Zhang,
and Wenping Wang. Dynosurf: Neural deformation-based
temporally consistent dynamic surface reconstruction.
In
European Conference on Computer Vision, pages 271–288.
Springer, 2024. 11
[49] Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J
Black, and Otmar Hilliges. Pointavatar: Deformable point-
based head avatars from videos. In CVPR, pages 21057–
21067, 2023. 1
10

<!-- page 11 -->
Tessellation GS: Neural Mesh Gaussians for Robust Monocular Reconstruction
of Dynamic Objects
Supplementary Material
Camera Setup
0
1
2
3
0
1
(a)
(b)
Figure 11. Camera setup for SDNF and PSS.(a): SDNF cam-
eras. (b): PSS cameras.
We use blender coordinate system, and the character’s up
direction aligns with the positive Z-axis.
For Smooth D-NeRF (SDNF), shown in Fig. 11 (a), camera
0 is a static validation view; camera 1, 2, and 3 rotate around
the character (z-axis) but are mutually rigid. Camera 3 is
training view, and camera 1 and 2 are test views, both 45 de-
grees azimuthal angles away from camera 3. Video supple-
mentary material’s ”smooth dnerf validation”plementary
material compares training and validation videos.
For People Snapshot dataset (PSS) shown in Fig. 11 (b),
both training camera 0 and test camera 1 are static, and on
the opposite sides of the character, 180 degree azimuthal an-
gle to each others. We solely used video captured by camera
0 as input, and tested the results with camera 1. For the one
cactus video from Unbiased4D (Ub4D), we could not pro-
vide rendering result from similarly novel views since only
the front of the toy is captured in the training video.
Stage One Loss Setup
For our proposed robust Chamfer loss LRCD in Eq. (2), U
and V are sets of vertices in target and source meshes, and
d is truncation distance. This setup helps the deformation
model to not be interfered by large and sudden deformations
between consecutive meshes, which are usually caused by
temporal inconsistency and floating artifact of LRM. In ad-
dition, we define Llap as follows:
Llap = 1
N
N
X
i=1
∥vi −
1
|Ei|
X
j∈Ei
vj∥2
(13)
where vi is location of the ith vertex and Ei is the set of
neighboring vertices of the ith vertex. Ln is normal consis-
tency regularization defined as follows:
Ln = 1
|E|
X
(i,j)∈E
∥ni −nj∥
(14)
where ni is vertex normal vector of the ith vertex and E is
the set of all neighboring vertex pairs.
T1
T2
T3
T4
Figure 12. Illustration of distribution of control points. From
T1 to T4, each next temperature value is halved with T4 being 1.
Deformation Model Setup
NPGS [5] uses a set of low-rank deformation basis to de-
form points. A fixed set of K learnable deformation basis
is used for all timesteps, while each point has an individ-
ual learnable time-varying weight used to weighted sum the
K deformation basis to get the displacement at timestep t.
We observed that although it keeps the motion of the object
low-rank, it doesn’t inherently keep local rigidity. Inspired
by DynoSurf [48], we decided to make things opposite: K
deformation control points provide time-varying deforma-
tion to drive deformation of mesh vertices. We express lo-
cations of each control point Ck as weighted sum over all
vertices in the canonical mesh with learnable weights mk
as in Eq. (15), where Vt0 is a matrix whose columns are
positions of all vertices in the canonical mesh, this keeps
11

<!-- page 12 -->
Loss Term
Weight
Explanation
Stage One
LRCD
1
Our proposed robust Chamfer distance that truncate loss to zero above a threshold.
Llap
0.5
Mesh Laplacian regularization ensures smoothness of mesh surface and uniformity of vertices.
Ln
0.001
Normal consistency regularization ensures smoothness of mesh surface.
Stage Two
L1
0.8
Rendered frames vs. input frames in L1 norm.
Lssim
0.2
SSIM loss between rendered frames and input frames.
Ledge
0.2
L1 norm loss to penalize change in edge length of meshes after deformation.
Llap
0.03
Same as in stage one.
Lα
0.002
Mean opacity of all Gaussians to encourage the model to use fewer Gaussians.
Lnormal
0.1 or 0
Predicted normal by DSINE [2] vs. rendered normal in L2 norm. Only used for static camera cases.
Lflow
0.01
Predicted optical flow by RAFT [41] vs. rendered optical flow in L1 norm.
Table 3. Parameter weights for stage one and stage two.
the control points within the convex hull of the mesh.
Ck = Vt0softmax(mk
T )
=
N
X
i=1
 
e
mk,i
T
PN
j=1 e
mk,j
T
!
vi,t0
(15)
We used hierarchical temperatures in softmax when decid-
ing control point locations. High temperatures keep the con-
trol points close to the large-scale structures of the mesh,
while low temperatures allow control points to spread out
onto finer geometries of the mesh, analogous to SMPL’s tree
structure. As illustrated in Fig. 12, we use 4 levels of gran-
ularities of control points, composed of in total 30 control
points to drive the canonical mesh. To initialize the con-
trol nodes, we use farthest point sampling (FPS) to select
30 points, then we set the corresponding weight mk to a
large value to keep the result from FPS. Each vertex has a
skinning weight over the 30 control points predicted by per
control point MLP taking as input displacement from vertex
to control point.
Further Results
(a)
(b)
(c)
(d)
Figure 13. People Snapshot result. (a): input image. (b): novel
view rendering result. (c): extracted mesh. (d): tracking result.
We have included an additional result on People Snap-
shot in Fig. 13. For complete tracking result, please refer
to video results. We have also conducted further ablation
study on the effectiveness of our stage two pipeline. We
incorporated stage one of our pipeline with DG-Mesh [25],
PSNR: 24.7
PSNR: 13.6
(d)
PSNR: 23.2
(c)
(b)
(a)
Figure 14. Ablation and quantitative comparison on Tessella-
tion GS. (a): input image. (b): our method’s rendered novel view.
(c): direct LRM rendering result. (d): DG-Mesh’s result after in-
corporating LRM prior.
equipping DG-Mesh with LRM prior. Shown in Fig. 14
(d), DG-Mesh still exhibits view-overfitting due to their rel-
atively more free setup of mesh Gaussians. This proves that
both stage one and stage two are crucial for the success of
our pipeline.
Optimization
In stage one, we used an exponentially decaying learning
rate from 1e-3 to 1e-5 except for mk in Eq. (15), where we
used a constant learning rate of 1e-2. We train for 20000
steps until it converges. Together with LRM generation of
mesh sequence, this step takes around 30 minutes. In stage
two, we used an exponentially decaying learning rate for all
MLPs, including motion MLPs, appearance decoders, and
pose encoders, from 1e-3 to 1e-5. We set constant learning
rate of 1e-3 for Gaussian features on mesh vertices, Gaus-
sian scales, and parent Gaussian opacities. We keep the con-
stant learning rate of 1e-2 for mk. We train 400000 steps
for the reconstruction to converge, which usually takes 60
minutes. In the initial 5000 steps, we train the model with-
out Gaussian density control. After the initial 5000 steps,
we use our adaptive density control technique mentioned
in Sec. 3.2.4 to introduce new Gaussians and delete exces-
sive Gaussians every 2000 steps until the final 5000 steps,
when we stop the density control again.
12
