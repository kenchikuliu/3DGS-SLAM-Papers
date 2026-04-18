<!-- page 1 -->
1
RP-SLAM: Real-time Photorealistic SLAM with
Efficient 3D Gaussian Splatting
Lizhi Bai, Chunqi Tian*, Jun Yang, Siyu Zhang, Masanori Suganuma, Takayuki Okatani
Abstract—3D Gaussian Splatting has emerged as a promising
technique for high-quality 3D rendering, leading to increasing
interest in integrating 3DGS into realism SLAM systems. How-
ever, existing methods face challenges such as Gaussian primitives
redundancy, forgetting problem during continuous optimization,
and difficulty in initializing primitives in monocular case due
to lack of depth information. In order to achieve efficient and
photorealistic mapping, we propose RP-SLAM, a 3D Gaussian
splatting-based vision SLAM method for monocular and RGB-
D cameras. RP-SLAM decouples camera poses estimation from
Gaussian primitives optimization and consists of three key
components. Firstly, we propose an efficient incremental mapping
approach to achieve a compact and accurate representation of
the scene through adaptive sampling and Gaussian primitives
filtering. Secondly, a dynamic window optimization method
is proposed to mitigate the forgetting problem and improve
map consistency. Finally, for the monocular case, a monocular
keyframe initialization method based on sparse point cloud is
proposed to improve the initialization accuracy of Gaussian
primitives, which provides a geometric basis for subsequent
optimization. The results of numerous experiments demonstrate
that RP-SLAM achieves state-of-the-art map rendering accuracy
while ensuring real-time performance and model compactness.
Index Terms—SLAM, 3D Gaussian splatting, photorealistic
mapping.
I. INTRODUCTION
Visual Simultaneous Localization and Mapping (vSLAM)
has long served as a foundational technology in robotics
and computer vision, with decades of research advancing its
applications across diverse fields [1]–[4]. The advent of sophis-
ticated, real-world applications such as autonomous driving,
augmented and virtual reality, and embodied intelligence has
given rise to a multitude of novel demands that extend be-
yond the conventional real-time tracking and mapping. These
advancements require SLAM systems not only to provide
precise spatial localization but also to deliver photorealistic
scene reconstruction, where achieving high-fidelity visual rep-
resentation is crucial.
Traditional vSLAM systems, exemplified by ORB-SLAM3
[1], LSD-SLAM [4] and DSO [3], rely on sparse feature
extraction methods and are primarily concerned with pro-
viding accurate but visually simplistic representations of the
surrounding environment. Consequently, these approaches are
Lizhi Bai, Chunqi Tian, Jun Yang and Siyu Zhang are with Department
of Computer Science and Technology, Tongji University, Shanghai, 201804,
China (e-mail: {bailizhi, tianchunqi, junyang, zsyzsy}@tongji.edu.cn).
Lizhi Bai, Masanori Suganuma and Takayuki Okatani are with Graduate
School of Information Sciences, Tohoku University, Sendai, 980-0845, Japan
(e-mail: {suganuma, okatani}@vision.is.tohoku.ac.jp).
* Corresponding author
not sufficient to meet the demands of photorealistic scene
reconstruction.
Recent studies have investigated the potential of implicit
scene representations, in particular Neural Radiance Field
(NeRF) [5], as a solution for dense and realistic 3D re-
construction. A number of approaches have been developed
which integrate NeRF into SLAM system, with the objective
of optimizing camera poses and map representation through
neural rendering. To illustrate, iMap [6] employs a multi-
layer perceptron (MLP) to represent the scene and optimize
the camera poses. NICE-SLAM [7] utilizes a hierarchical grid
to represent the 3D scene and optimize the implicit features
stored in the grid nodes. Additionally, there are many tech-
niques [8]–[10] that achieve enhanced performance through
the utilization of multi-resolution hash grids [11]. Despite the
considerable potential of NeRF in the generation of realistic
scenes, it is confronted with a number of challenges when in-
tegrated into SLAM systems, including the high computational
cost, prolonged training time, and vulnerability to catastrophic
forgetting.
A recently developed explicit scene representation technique
based on 3D Gaussian Sputtering (3DGS) [12] has demon-
strated the potential to provide a compelling solution for high-
quality 3D rendering. In comparison to NeRF, 3DGS has
been shown to achieve comparable rendering quality while
exhibiting significantly superior rendering speed.
In light of the enhancements in rendering efficiency and
optimization of 3DGS, there has been a amount of research
conducted into the integration of 3DGS into dense and
photorealistic SLAM systems. Notably, SplaTAM [13] and
MonoGS [14] are the inaugural approaches to utilisze 3DGS
for coupling estimation of camera poses with optimization
of scene representation. These coupled approaches require
tedious iterations to optimize the camera poses based on the
scene representation, making them challenging to execute in
real time.
Concurrent works [15]–[17] decouple the estimation of
camera poses from the optimization of Gaussian primitives.
They employ conventional techniques [1], [18], [19] to esti-
mate the camera poses, circumventing the time-consuming it-
erations inherent to the coupled approaches, thereby enhancing
real-time performance. For incremental mapping, these meth-
ods typically rely on dense pixel sampling when initializing
new Gaussian primitives, leading to redundant primitives and
significant storage overhead. In order to optimize the scene
representation, the prevailing method involves the selection
of a specific number of keyframes from the keyframe set,
which are then merged with the newly added keyframe into
arXiv:2412.09868v1  [cs.RO]  13 Dec 2024

<!-- page 2 -->
2
a fixed keyframe window. Only keyframes located within this
fixed window are used for map optimization. However, this
approach may potentially lead to erroneous local minima and
forgetting problems during successive iterations.
The absence of depth information in the monocular case
makes it challenging to accurately add new primitives.
MonoGS [14] employs random depths with no geometric basis
for initialization, and its complete dependence on the mapping
process to optimize the initial primitives makes it difficult
to obtain an accurate representation of the scene. Photo-
SLAM [16] and CaRtGS [20] employ only the spatial gradient-
based densification method in the original 3DGS [12] to add
new primitives. However, the limited number of iterations is
insufficient to optimize the new primitives to accurate posi-
tions, resulting in artifacts due to the inconsistency between
the geometric and photometric properties of the Gaussian
primitives.
In order to address these challenges, we propose a real-
time photorealistic SLAM with efficient 3D Gaussian splatting
for RGB-D and monocular cameras, namely RP-SLAM. RP-
SLAM is a decoupled system that utilizes feature-based SLAM
for camera tracking and 3DGS for optimizing the photoreal-
istic scene representation.
Our method comprises three main components. Firstly, We
propose an efficient incremental mapping method that incor-
porates image sampling and Gaussian primitives filtering. In
contrast to previous dense image sampling, we propose adap-
tive sampling using image gradients to focus the sampling and
computational resources on texture-rich regions and reduce the
generation of redundant samples. In conjunction with Gaussian
primitives filtering, redundant primitives are further eliminated
to achieve an efficient and compact scene representation.
Secondly, to achieve a consistent scene representation, we
propose dynamic keyframe window optimization. Compared
to the fixed keyframe window, this method dynamically adjusts
the keyframes to be optimized at each iteration based on
the covisibility between keyframes, alleviating the forgetting
problem during the continuous optimization process. Finally,
in order to add new Gaussian primitives in the monocular
case where geometric information is absent, we propose a
monocular keyframe initialization method based on sparse
point cloud. In comparison to alternative methods with un-
certainty in creating new primitives, this method can create
new primitives relatively accurately using the initial geometric
information and provide a geometric basis for subsequent
optimization.
To summarize, our contributions are as follows:
• Efficient Incremental Mapping: By using adaptive sam-
pling guided by local image gradients and Gaussian prim-
itives filtering, our method reduces redundant Gaussian
primitives while maintaining high rendering quality.
• Robust Mapping with Dynamic Keyframe Window:
Dynamic keyframe window optimization mitigates the
impact of forgetting, enhancing mapping consistency.
• Improved Monocular Keyframe Initialization: For
monocular cameras, our keyframe initialization approach
enables relatively accurate placement of Gaussian prim-
itives, improving rendering quality and reducing redun-
dancy.
• We validate our approach on standard RGB-D and
monocular datasets (TUM [21], Replicia [22] and Scan-
Net++ [23]), demonstrating its superior efficiency, render-
ing quality, and robustness compared to existing methods.
II. RELATED WORK
A. Classical Visual SLAM
Classical SLAM methods typically employ factor graphs
to model the intricate optimization challenges between vari-
ables and measurements. In order to achieve real-time opera-
tion, these methods incrementally update the estimated pose,
thereby avoiding computationally expensive processes. To
illustrate, the ORB-SLAM series [1], [24], [25] are based on
the extraction and tracking of lightweight geometric features
across sequential frames, with bundle adjustment conducted
locally rather than globally. Direct SLAM approaches, such as
LSD-SLAM [4] and DSO [3], operate directly with raw image
intensities, circumventing the necessity for geometric feature
extraction. They are capable of maintaining a sparse or semi-
dense map represented by point cloud, even in the context
of resource-constrained systems. DTAM [26], an early dense
SLAM system, employs photometric consistency across pixels
to track the camera, leveraging multi-view stereo constraints to
update the dense scene model and represent it as a cost volume.
KinectFusion [18] employs RGB-D cameras to facilitate real-
time camera pose estimation and scene geometry updates
through iterative closest point (ICP) and TSDF-Fusion [27].
Subsequent studies have introduced a range of data structures
to enhance the scalability of SLAM systems, including Vox-
elHash [28]–[30] and Octrees [31], [32].
Although classical SLAM techniques offer real-time perfor-
mance and efficiency, they often fall short of the level of detail
and visual fidelity required to accurately represent complex
environments. Furthermore, they are inadequate for achieving
realistic scene reconstruction in advanced applications.
B. Neural Implicit Visual SLAM
iNeRF [33] represents a pioneering attempt to demonstrate
that camera pose regression is feasible within a trained NeRF
model for a specific scene. However, rather than being a
complete SLAM system, iNeRF addresses a localization task
based on a existing model. Barf [34] further presents a method
to simultaneously fit NeRF while refining camera poses, even
with an initially rough estimate, by establishing a theoretical
bridge between classical image alignment and joint alignment
and reconstruction within neural radiance fields. iMAP [6],
on the other hand, introduces the application of NeRF to re-
construct accurate 3D geometry from RGB-D images without
requiring initial pose information. By using a single MLP
to represent the global scene, iMAP jointly optimizes both
the scene map and camera poses. However, this single MLP
architecture struggles to capture fine geometric details and
to scale effectively in larger environments. NICE-SLAM [7]
addresses this scalability limitation by partitioning the world
coordinate frame into uniform grids, thereby enhancing both

<!-- page 3 -->
3
Fig. 1. Overview of our RP-SLAM. Keyframes and sparse point cloud are provided the feature-based SLAM, where the point cloud are used for monocular
keyframe initialization. Afterwards, the dense point cloud is obtained by efficient incremental mapping, which in turn initializes the Gaussian primitives.
Finally, the scene representation is optimized by dynamic keyframe window, where the geometric loss is used only for the RGB-D case. The dashed arrows
are for the monocular case only
inference speed and accuracy. NeRF-SLAM [8] integrates
Droid-SLAM [2] with Instant-NGP [11], utilizing Droid-
SLAM for camera pose estimation, dense depth mapping, and
uncertainty measurement, and leveraging this data to optimize
Instant-NGP’s scene representation. GO-SLAM [35] further
improves global scene consistency by introducing loop closure
and global bundle adjustment, while Co-SLAM [10] achieves
high-quality scene reconstruction by combining coordinate
encoding with sparse grid representations.
Despite the significant potential of NeRF in the generation
of realistic scenes, a number of challenges emerge when inte-
grating this approach into SLAM systems. These include the
high computational cost, the prolonged training time required,
and the vulnerability to catastrophic forgetting.
C. 3D Gaussian Splatting Visual SLAM
3DGS [12] has rapidly gained interest within the field of
SLAM research due to its rapid rendering capabilities and
explicit scene representation. MonoGS [14] and SplaTAM
[13] represent foundational efforts in coupled GS-SLAM algo-
rithms, pioneering an approach that jointly optimizes Gaussian
primitives and camera poses through gradient backpropaga-
tion. The concept of sub-maps is introduced by Gaussian-
SLAM [36] as a means of mitigating the issue of catastrophic
forgetting. LoopSplat [37] builds upon the foundations of
Gaussian-SLAM [36] through the utilization of a Gaussian
splatting-based loop closure strategy, thereby enhancing the
accuracy of poses through the implementation of improved
registration techniques. However, a significant obstacle with
these methods is that the time-consuming process of esti-
mating the camera pose for each frame using 3DGS places
a considerable computational burden, which limits its ability
to achieve real-time performance. In order to address this
challenge, decoupled 3DGS-based SLAM approaches have
been introduced. Splat-SLAM [38] and IG-SLAM [39] employ
pre-trained dense bundle adjustment [2] for the purpose of
camera pose tracking, while simultaneously utilising proxy
depth maps for optimizing map representation. RTG-SLAM
[15] integrates frame-to-model ICP [18] for pose tracking and
employ the opaque Gaussian primitives for depth rendering.
GS-ICP-SLAM [17] attains remarkable speeds by capitalising
on the shared covariance between G-ICP [19] and 3DGS.
Photo-SLAM [16] and CaRtGS [20] deploy ORB-SLAM3
[1] for tracking and incorporates a coarse-to-fine approach in
map optimization, thereby enhancing system robustness and
overall performance. For incremental mapping, these methods
often rely on dense pixel sampling to initialize new Gaussian
primitives, resulting in redundancy and high storage costs.
To optimize scene representation, a common approach selects
a fixed number of keyframes, merging them with the latest
keyframe into a fixed keyframe window for map optimization.
However, this strategy risks local minima and forgetting issues
over iterations.
In the monocular case, the lack of depth information com-
plicates the accurate addition of new primitives. MonoGS
[14] initializes primitives using random depths without a
geometric foundation, relying entirely on the mapping process
for optimization, which hinders precise scene representation.
Photo-SLAM [16] and CaRtGS [20] adopt the spatial gradient-
based densification method from the original 3DGS [12] to
add primitives. However, the limited iterations fail to optimize
these primitives, leading to artifacts caused by mismatches
between the geometric and photometric properties of the
Gaussian primitives.
III. METHODS
The overview of RP-SLAM is illustrated in Fig. 1. Feature-
based SLAM is primarily utilized for camera poses estimation,
the generation of keyframes and their covisibility, as well as
the provision of a sparse initial point cloud for monocular
mode. Then, the received keyframes is used for efficient
incremental mapping to generate Gaussian primitives. The

<!-- page 4 -->
4
(a) c = 4. No. of sampling points: 9898.
(b) c = 8. No. of sampling points: 2872.
(c) c = 16. No. of sampling points: 826.
Fig. 2. Quadtree-based adaptive image sampling guided by image local gradients at different minimum cell sizes: 4, 8, 16. The method is capable of adaptively
focusing sampling on regions that are rich in texture. A smaller minimum cell size allows for a more detailed sampling, but this is accompanied by an increased
need for processing of the resulting data.
Gaussian primitives are optimized using the dynamic keyframe
window based on the covisibility. In the monocular mode,
sparse point cloud must be employed for monocular keyframe
initialization.
A. 3D Gaussian Splatting
In our method, 3DGS is employed for the purpose of map-
ping the scene, which comprises a set of Gaussian primitives
G and renders photorealistic images by tile-based rasterisa-
tion. Each Gaussian primitive Gi is characterised by both
optical and geometric attributes. The former include color ci
and opacity αi, whereas the latter are defined in the world
coordinate, comprising a mean µw
i indicative of its position
and a covariance Σw
i indicative of its ellipsoidal shape. And
the covariance Σw
i is decomposed into a scale vector si and
a quaternion qi. Given n Gaussians, the color of a pixel can
be obtained by rendering:
C =
n
X
i=1
ciαi
i−1
Y
j=1
(1 −αj).
(1)
Following [14], per-pixel depth can also be rasterised by alpha-
blending:
D =
n
X
i=1
ziαi
i−1
Y
j=1
(1 −αj),
(2)
where zi is the distance to the mean µw
i of Gaussian Gi along
the camera ray.
In contrast to marching along the camera rays, the rasteriza-
tion process iterates over the Gaussians. As such, free space is
disregarded during the rendering process. The contributions of
α are decayed during rasterization by the Gaussian functions
based on the 2D Gaussians N(µ, Σ) formed by splatting the
3D Gaussians N(µw, Σw):
µ = π(Tcwµw),
Σ = JRΣwR⊤J⊤,
(3)
where π is the projection operator, and Tcw represents the
camera pose in the world. J refers to the Jacobian of the
projective transformation, and R is the rotation of Tcw. This
formulation makes the 3D Gaussian differentiable, and the
blending operation provides the Gaussian with a gradient flow.
To optimize the scene representation, photometric loss Lpho
and geometric loss Lgeo are considered, where geometric
loss is only used for RGB-D case. Following MonoGS [14],
isotropic regularization Liso is introduced with the aim of
reducing the generation of artifacts by penalize the scaling
parameters s. As a result, the final monocular loss Lmono and
RGB-D loss LRGB−D are as follows:
Lpho = ||I(G, Tcw) −¯I||1,
Lgeo = ||D(G, Tcw) −¯D||1,
Liso =
|G|
X
i=1
||si −˜si · 1||1,
Lmono = Lpho + λisoLiso.
LRGB−D = λphoLpho + (1 −λpho)Lgoe + λisoLiso.
(4)
where I(G, Tcw) and D(G, Tcw) are the rendered rgb image
and depth map from the given Gaussian primitives G and
camera pose Tcw, and ¯I and ¯D are the ground truth rgb image
and depth map. ˜si is the mean of si. λpho and λiso are the
hyperparameters.
B. Efficient Incremental Mapping
The efficient scene representation in 3DGS-based SLAM is
reliant upon a meticulous balance between the number of map
parameters (number of Gaussian primitives) and map fidelity.
The majority of 3DGS-based methods frequently encounter
difficulties in maintaining this balance, as uniformly dense
sampling may result in superfluous computation and storage of
simple regions, whereas sparse sampling may prove inadequate
for capturing crucial details in complex textures. In order to
address this challenge, we propose the efficient incremental
mapping which consists of quadtree-based adaptive image
sampling and KNN-based Gaussian primitive filtering.
For each keyframe from feature-based SLAM, the quadtree-
based adaptive image sampling is implemented through a
recursive subdivision of this keyframe into smaller quadtree
cells. Subdivision of the image continues until the adaptive
minimum cell size cth = ηc or the variance in the gradient
of each cell falls below an adaptive threshold τth = ητ, with
each cell evaluated in turn. Here, η =
√H×W
512
, c and τ are
the predefined thresholds, and H and W are the image sizes.
Formally, the gradient variance V of a cell C is defined as:
V (C) =
1
|C|
X
p∈C
(∇I(p) −µC)2,
(5)
where ∇I(p) is the gradient magnitude at pixel p, and µC
is the mean gradient magnitude within cell C. Subdivision

<!-- page 5 -->
5
(a)
(b)
(c)
Fig. 3. Rendered depths in the monocular case. (a) Initial depth obtained by our RP-SLAM from a sparse point cloud, which describes the initial geometry
at the viewpoint. (b) Depth obtained by initial iterations using the depth of (a) based on the dense point cloud obtained in Sec. III-B. (c) Depth obtained
according to MonoGS’s [14] monocular initial method. Following preliminary iterations, the Gaussian primitives obtained by our method in the monocular
case have been found to describe the scene structure with reasonable accuracy. This is in comparison to the result obtained by MonoGS [14], which is less
satisfactory in this regard.
occurs if V (C) > τ. Fig. 2 depicts the sampling of the
image at different minimum cell sizes. The quadtree-based
adaptive image sampling dynamically adjusts the sampling
density based on the gradient information of the input image.
This method guarantees that image sampling is concentrated
on regions of high structural complexity, thereby ensuring an
efficient sampling process.
Once the sampling pixels are selected, they are back-
projected into 3D space using the camera pose and intrinsic
parameters, resulting in a set of candidate Gaussian primitives.
For a sampled pixel p in the image, the corresponding 3D
position P in the world coordinate frame is computed as:
P = Twcπ−1(p, d),
(6)
where π−1 represents the inverse projection using the cor-
responding depth d, and Twc is the transformation from
camera to world coordinate frame. To circumvent redundancy
and guarantee an efficacious map representation, KNN-based
filtering step is implemented for the newly generated Gaus-
sian primitives. For each candidate primitive, the method
assesses its spatial proximity to existing primitives in the
map. Specifically, For a newly generated Gaussian primitive
Gnew, a KNN [40] search is performed to find its closest
neighbors G = {G1, G2, ..., Gk} and k = 3 in our setting. The
new Gaussian Gnew is considered redundant if the distance
dis(Gnew, Gi) to every neighbor Gi satisfies:
dis(Gnew, Gi) < λri ∀Gi ∈G,
(7)
where λ is a predefined scaling factor that adjusts the effective
influence of the neighbor’s radius, ri, which is the minimum
value in the scale vector si corresponding to the Gaussian
primitive Gi. If this condition is met, Gnew is deemed
redundant and discarded. Otherwise, Gnew is added to the
map as a new Gaussian primitive. The KNN-based Gaussian
primitive filtering eliminates redundant primitives based on
local Gaussian primitive relations, thereby ensuring an efficient
scene representation and, consequently, the efficient utilization
of computational and storage resources.
By combining quadtree-based adaptive sampling and KNN-
based filtering, the proposed method achieves efficient incre-
mental mapping. Quadtree-based adaptive sampling ensures
that computational resources are concentrated on regions with
significant structural details, while KNN-based filtering elim-
inates redundant Gaussians, maintaining a compact and non-
redundant map representation.
C. Dynamic Keyframe Window
The addition of a new keyframe typically signifies the
necessity for optimization in previously unexplored areas. The
prevailing methods entail the selection of a specific number of
keyframes from the keyframe set, which are then incorporated
into a fixed window of keyframes with the newly added
keyframe. Only those keyframes situated within this fixed
window are employed for map optimization. Nevertheless,
during the iterative process, this approach may give rise to
severe forgetting and overfitting issues, ultimately leading to
a deterioration in the quality of the final map representation.
To address this issue, we propose the optimization of a
dynamic keyframe window at each iteration. Specifically, for
each new keyframe Knew generated by feature-based SLAM,
the system categorizes all keyframes K into two distinct
sets: co-visible keyframes K1 and other keyframes K2. This
categorization is based on the degree of visibility of other
keyframes in relation to the current keyframe Knew. Co-
visible keyframes K1 are defined as keyframes that observe the
same or near-neighbouring regions as the current keyframe.
In contrast, other keyframes K2 represent those that do not
observe the same region as the current keyframe. At each
iteration, a random selection of k1 and k2 keyframes is made
from the two sets, respectively. These keyframes are then
combined with the current keyframe, thus forming a dynamic
keyframe window W:
W = {Knew} ∪S1 ∪S2,
S1 ⊆K1,
|S1| = k1,
S2 ⊆K2,
|S2| = k2.
(8)
Subsequently, the dynamic keyframe window is employed for
the purpose of optimizing the map representation, specifically
the parameters of the Gaussian primitives.
This approach ensures that the optimization process is
focused on both local consistency, with consideration given
to keyframes that are closely related to Knew, and global
consistency, with the inclusion of non-co-visible keyframes
that are critical for maintaining long-term map accuracy. By

<!-- page 6 -->
6
TABLE I
QUANTITATIVE RESULTS FOR REPLICA [22] DATASET IN THE
MONOCULAR CASE. MS DENOTES MODEL SIZE (MB)
Methods
Metrics
o0
o1
o2
o3
o4
r0
r1
r2
Avg.
MonoGS
[14]
PSNR↑30.58 32.49 32.65 26.34 24.69 23.76 24.99 23.57 27.38
SSIM↑0.892 0.904 0.836 0.851 0.871 0.755 0.785 0.833 0.841
LPIPS↓0.249 0.191 0.302 0.22 0.315 0.322 0.352 0.353 0.288
ATE↓
25.4
24.8
48.9
11.8
61.6
13.6
55.4
26.1 33.45
FPS↑
1.4
1.6
1.3
1.3
1.3
1.3
1.3
1.3
1.5
MS↓
6.5
4.6
7.2
8.4
5.9
8.5
7.3
5.2
6.7
Photo-
SLAM
[16]
PSNR↑36.98 37.59 31.79 31.62 34.16 29.77 31.3 33.18 33.30
SSIM↑0.955 0.950 0.929 0.920 0.941 0.871 0.911 0.934 0.927
LPIPS↓0.061 0.062 0.091 0.086 0.072 0.106 0.083 0.067 0.079
ATE↓
0.32
0.45
2.53 0.39
0.61
0.43
0.68
0.27 0.71
FPS↑
35.9
35.4
34.1 34.8
36.1
34.1 36.4
34.9
35.2
MS↓
19.1
18.4
25.4
18.4
19.8
26.8
32.1 21.1 23.14
CaRtGS
[20]
PSNR↑35.49 36.22 33.54 32.82 35.31 31.92 32.44 34.53 34.03
SSIM↑0.946 0.947 0.938 0.936 0.944 0.913 0.915 0.953 0.934
LPIPS↓0.078 0.052 0.076 0.077 0.061 0.074 0.071 0.051 0.068
ATE↓
0.21
0.23
1.31 0.13
0.18
0.17 0.42
0.19
0.36
FPS↑
36.5
36.4
33.6 34.7
36.9
34.5 37.5
34.6
35.6
MS↓
9.9
11.4
16.1
14.1
13.4
23.6
18.4 15.5 15.3
RP-SLAM
(ours)
PSNR↑37.74 40.35 34.29 32.78 35.92 31.95 33.61 35.85 35.31
SSIM↑0.958 0.961 0.941 0.937 0.942 0.921 0.925 0.935 0.940
LPIPS↓0.062 0.058 0.071 0.075 0.065 0.071 0.067 0.062 0.067
ATE↓
0.35
0.31
2.24
0.44 0.57
0.35 0.46
0.38
0.64
FPS↑
18.0
18.3
16.5
17.2
17.8
16.3
18.2 17.0 17.3
MS↓
11.8
8.1
12.3 10.9
11.8
12.7 11.9
11.1
11.3
dynamically adapting the keyframe set, this method prevents
overfitting to the newly added keyframe and mitigates forget-
ting of previously observed regions, thereby ensuring a more
balanced and robust map optimization process.
D. Monocular Keyframe Initialization
In the context of monocular case, the initialization of a dense
scene representation is a challenging yet indispensable step for
the generation of high-quality mapping. Feature-based SLAM
is capable of providing a sparse point cloud representation
of the scene. However, this representation is inadequate for
tasks that necessitate high-density and realistic reconstruction,
given the limitations of sparse point cloud. To address this
issue, we propose a monocular keyframe initialization method
that employs the sparse point cloud to generate an initial set
of Gaussian primitives, and subsequently achieves a dense and
accurate scene representation through a refinement process.
For each newly added keyframe, the sparse point cloud
generated from the feature-based SLAM corresponding to the
new keyframe are first extracted. These point cloud provide a
simple description of the geometric information corresponding
to the new keyframe. They are subsequently used to initialize
some of the new Gaussian primitives, which are combined
with the existing Gaussian primitives to provide a preliminary
representation of the scene. The initialised Gaussian primitives
are then optimised using the new keyframe in accordance with
Eq. 4, and an initial depth map is generated for the new
keyframe in terms of Eq. 2. The initial depth map, shown
in Fig. 3 (a), provides a foundational estimation of the new
scene’s structure, enabling efficient image sampling and Gaus-
sians generation as described in Sec. III-B, which facilitates
TABLE II
QUANTITATIVE RESULTS FOR REPLICA [22] DATASET IN THE RGB-D
CASE. MS DENOTES MODEL SIZE (MB).
Methods
Metrics
o0
o1
o2
o3
o4
r0
r1
r2
Avg.
MonoGS
[14]
PSNR↑40.08 41.22 35.57 34.34 33.74 32.78 35.87 36.61 36.78
SSIM↑0.971 0.974 0.956 0.952 0.936 0.934 0.953 0.972 0.956
LPIPS↓0.058 0.049 0.065 0.063 0.099 0.074 0.081 0.071 0.070
ATE↓
0.43 0.51 0.23
0.21
2.36 0.44 0.26
0.35
0.60
FPS↑
1.2
1.3
1.1
1.0
1.1
1.1
1.1
1.0
1.1
MS↓
16.2 15.2 26.6 24.7 23.1 25.2
18.5
22.1
21.5
Photo-
SLAM
[16]
PSNR↑38.47 39.08 33.03 33.78 36.02 30.71 33.51 35.12 34.97
SSIM↑0.964 0.961 0.938 0.938 0.952 0.899 0.934 0.951 0.942
LPIPS↓0.05 0.047 0.077 0.066 0.054 0.075 0.057 0.043 0.059
ATE↓
0.48 0.45 1.32 0.79 0.57 0.49 0.45
0.29
0.61
FPS↑
32.3
32.2 29.2
28.8
28.2 28.2
29.6
26.3 29.3
MS↓
19.9 20.1 31.4 24.3 30.9 51.8 39.7
38.2
32.0
SplaTAM
[13]
PSNR↑37.97 38.95 32.92 29.81 31.96 32.56 33.61 35.11 34.11
SSIM↑0.981 0.981 0.966 0.951 0.948 0.955 0.961 0.980 0.965
LPIPS↓0.089 0.097 0.098 0.117 0.155 0.071 0.097 0.075 0.100
ATE↓
0.41
0.25 0.32 0.35 0.57 0.29 0.51
0.33 0.38
FPS↑
0.2
0.2
0.3
0.3
0.2
0.2
0.2
0.2
0.2
MS↓
323.6 294.2 237 273.3 276.7 265.8 348.7 298.4 289.7
RTG-
SLAM
[15]
PSNR↑39.09 39.22 33.45 33.33 35.53 30.79 34.52 35.65 35.07
SSIM↑0.987 0.986 0.981 0.951 0.984 0.945 0.977 0.981 0.974
LPIPS↓0.098 0.135 0.075 0.298 0.116 0.154 0.131 0.137 0.143
ATE↓
0.15
0.14 0.22
0.26
0.25 0.21
0.19
0.12 0.19
FPS↑
8.3
8.4
8.1
7.9
8.1
7.9
7.8
7.8
8.0
MS↓
51.2 56.5 53.6 63.3 62.4 61.9 78.3
66.8
61.8
CaRtGS
[20]
PSNR↑35.52 37.85 33.58 34.04 35.34 30.02 33.58 36.65 34.57
SSIM↑0.953 0.955 0.946 0.943 0.956 0.847 0.936 0.964 0.938
LPIPS↓0.061 0.044 0.066 0.061 0.051 0.072 0.058 0.043 0.057
ATE↓
0.49 0.39 1.03 0.42
0.47 0.32 0.36 0.19 0.46
FPS↑
31.5
32.5 29.8
28.6
27.1 27.8
29.5
26.1 29.1
MS↓
10.3
14.2 20.1
16.6
20.2 29.9 25.3
27.9 20.6
RP-SLAM
(ours)
PSNR↑41.26 41.01 36.27 34.98 36.05 33.74 36.71 36.96 37.12
SSIM↑0.982 0.985 0.975 0.972 0.958 0.951 0.963 0.982 0.971
LPIPS↓0.055 0.057 0.064 0.048 0.051 0.064 0.062 0.049 0.056
ATE↓
0.43 0.38 0.53 0.36 0.56 0.43 0.25
0.23
0.40
FPS↑
19.1 19.5 17.8 17.6 18.5 17.6 18.7
17.9
18.3
MS↓
10.1
7.3
10.9
9.5
10.7 11.7
10.9
9.2
10.0
the creation of a dense scene representation corresponding
to the new keyframe in monocular mode. Once the dense
Gaussian primitives for the new keyframe have been generated
by the described method, the scene representation is optimized
through the implementation of the dynamic keyframe window,
as detailed in Sec. III-C. The rendered depths before and after
initialization are illustrated in Fig. 3. MonoGS [14] employs
a random value initialization process for Gaussian primitives,
which results in depth map that is challenging to describe the
scene geometry after initial optimization as shown in Fig. 3
(c). For MonoGS, the inherent uncertainty in the geometry
further necessitates additional steps to eliminate outliers and
to assign more primitives to new viewpoints, with the objective
of enhancing the rendering quality.
This monocular keyframe initialization approach effectively
bridges the gap between sparse and dense representations
in monocular case. By leveraging the sparse point cloud in
ORB-SLAM3 and combining it with adaptive sampling and
redundancy removal methods, our approach ensures that the
monocular initialization process is both efficient and capable
of capturing fine scene details.

<!-- page 7 -->
7
TABLE III
QUANTITATIVE RESULTS FOR TUM [21] DATASET. MS DENOTES MODEL
SIZE (MB).
Mode
Methods
Metrics
fr1
fr2
fr3
Avg.
ATE↓
MS↓
Mono
MonoGS
[14]
PSNR↑16.71 15.59 19.19 17.16
SSIM↑
0.634 0.665 0.727 0.675
4.16
2.3
LPIPS↓0.411 0.338 0.351 0.367
Photo-
SLAM
[16]
PSNR↑20.97 21.07 19.59 20.54
SSIM↑
0.743 0.726 0.692 0.720
1.22
18.2
LPIPS↓0.228 0.166 0.239 0.211
CaRtGS
[20]
PSNR↑20.61 21.53 20.08 20.74
SSIM↑
0.726 0.718 0.717 0.720
1.13
12.9
LPIPS↓0.248 0.159 0.236 0.214
RP-SLAM
(ours)
PSNR↑21.87 22.48 21.32 21.89
SSIM↑
0.751 0.738 0.775 0.755
1.21
6.3
LPIPS↓0.235 0.162 0.231 0.209
RGB-D
MonoGS
[14]
PSNR↑18.67 15.94 19.23 17.95
SSIM↑
0.708 0.798 0.742 0.749
1.45
2.7
LPIPS↓0.317 0.311 0.325 0.318
Photo-
SLAM
[16]
PSNR↑20.87 22.09 22.74 21.90
SSIM↑
0.743 0.765 0.780 0.763
1.07
17.1
LPIPS↓0.235 0.169 0.154 0.186
SplaTAM
[13]
PSNR↑21.88 23.35 20.61 21.94
SSIM↑
0.831 0.852 0.744 0.809
3.34
140.4
LPIPS↓0.238 0.154 0.211 0.201
RTG-
SLAM
[15]
PSNR↑19.38 17.53 18.86 18.59
SSIM↑
0.716 0.686 0.761 0.721
1.05
76.5
LPIPS↓0.465 0.464 0.438 0.456
CaRtGS
[20]
PSNR↑20.59 22.75 22.99 22.11
SSIM↑
0.729 0.748 0.773 0.750
0.99
13.9
LPIPS↓0.253 0.158 0.151 0.187
RP-SLAM
(ours)
PSNR↑22.89 23.32 23.07 23.09
SSIM↑
0.783
0.85
0.776 0.805
0.98
3.8
LPIPS↓0.228 0.161 0.213 0.200
IV. EXPERIMENTS
To validate the effectiveness of the proposed RP-SLAM,
we conduct extensive experiments designed to evaluate its
performance in both monocular and RGB-D settings across
a range of both real and synthetic datasets. In addition, we
perform ablation studies to justify our design choices.
A. Experimental Setup
1) Datesets: For our quantitative analysis, we evaluate
our method on three datesets: TUM-RGBD dataset [21] (3
sequences), Replica dataset [22] (8 sequences) and ScanNet++
dataset [23] (4 sequences). The Replica [22] benchmark is the
simplest because it contains synthetic scenes, highly accurate
and complete (synthetic) depth maps, and small displacements
between consecutive camera poses. In contrast, the TUM-
RGBD [21] benchmark is more difficult because it uses
older, lower-quality cameras with poor-quality RGB and depth
images. The depth images are quite sparse with a lot of missing
information, while the color images have a very high degree
of motion blur. ScanNet++ [23] displays remarkable color
and depth image quality in comparison to other benchmarks.
Additionally, a supplementary trajectory is provided for each
scene, allowing for the assessment of the rendering capabilities
TABLE IV
QUANTITATIVE RESULTS FOR SCANNET++ [23] DATASET IN THE RGB-D
CASE. MS DENOTES MODEL SIZE (MB).
Views
Methods
Metrics
S1
S2
S3
S4
Avg.
Training View
SplaTAM
[13]
PSNR↑
27.82
23.41
26.69
27.39
25.97
SSIM↑
0.946
0.885
0.891
0.941
0.907
LPIPS↓
0.119
0.263
0.214
0.121
0.199
MS↓
194.3
210.7
313.4
150.1
242.8
RTG-SLAM
[15]
PSNR↑
20.56
21.23
24.88
23.24
22.48
SSIM↑
0.872
0.981
0.906
0.901
0.893
LPIPS↓
0.244
0.228
0.232
0.238
0.236
MS↓
150.8
149.8
168.1
158.4
156.8
RP-SLAM
(ours)
PSNR↑
29.73
27.59
28.96
29.77
29.01
SSIM↑
0.941
0.863
0.945
0.931
0.920
LPIPS↓
0.101
0.197
0.122
0.112
0.133
MS↓
25.3
32.1
22.8
26.8
26.8
Novel View
SplaTAM
[13]
PSNR↑
23.31
22.08
24.22
23.07
23.17
SSIM↑
0.839
0.849
0.882
0.889
0.865
LPIPS↓
0.288
0.276
0.257
0.196
0.266
RTG-SLAM
[15]
PSNR↑
19.32
19.14
19.98
20.03
19.62
SSIM↑
0.803
0.801
0.812
0.798
0.802
LPIPS↓
0.288
0.297
0.268
0.259
0.284
RP-SLAM
(ours)
PSNR↑
25.41
25.62
25.83
26.15
25.75
SSIM↑
0.879
0.835
0.878
0.894
0.872
LPIPS↓
0.208
0.241
0.214
0.212
0.219
in a new viewpoint. We evaluate four scenes (8b5caf3398,
39f36da05b, b20a261fdf, f34d532901) as used in [15].
2) Implementation Details: All experimental evaluations
are executed on a desktop PC with an Intel Core i9-14900KF
CPU and NVIDIA RTX 4090 GPU. In our implementation,
we use ORB-SLAM3 [1] for camera tracking due to its
robustness in multiple environments. The 3DGS module from
MonoGS [14] is utilized, and Gaussian primitives parameters
initialization method and the original hyperparameters are
retained. Additionally, the Gaussians densification and pruning
methods present in MonoGS are also maintained. The spheri-
cal harmonics degree is set to 0 as in MonoGS. The iterations
of optimization for each keyframe is set to 100 and the initial
iterations is set to 50 in monocular case. We follow MonoGS
by setting λpho and λiso to 0.9 and 10, respectively. For other
hyperparameters, we set τ = 15, λ = 1.0, k1 = 5, k2 = 3.
In addition, we set c to 8 on Replica [22] and TUM [21]
datasets and to 4 on ScanNet++ [23] dataset based on its high-
resolution details.
3) Metrics: In regard to the precision of camera tracking,
we present the Root Mean Square Error (RMSE) of the
Absolute Trajectory Error (ATE) of the keyframes in centime-
ter. In order to evaluate the quality of map rendering, three
commonly used metrics are considered: PSNR, SSIM, and
LPIPS. Furthermore, the computational and storage efficiency
are evaluated by measuring the system frame rate and the
model size after mapping. The average of three runs is reported
for all evaluations. Best results are highlighted as first and
second .
4) Baseline Methods: We compare our method with several
state-of-the-art Gaussian SLAM approaches such as MonoGS
[14], SplaTAM [13], Photo-SLAM [16], RTG-SLAM [15]
and CaRtGS [20]. Among these methods, MonoGS, Photo-

<!-- page 8 -->
8
TABLE V
ABLATION STUDY FOR MONOCULAR KEYFRAME INITIALIZATION (MKI),
EFFICIENT INCREMENTAL MAPPING (EIM) AND DYNAMIC KEYFRAME
WINDOW (DKW) ON REPLICA OFFICE0 [22] IN MONOCULAR CASE. MS
DENOTES MODEL SIZE (MB).
MKI
EIM
DKW
PSNR↑
MS↓
✗
✗
✗
34.87
19.3
✓
✗
✗
36.67
16.8
✗
✓
✗
34.23
15.6
✗
✗
✓
35.28
18.9
✓
✓
✓
37.74
11.8
TABLE VI
ABLATION STUDY FOR MINIMUM CELL SIZE ON REPLICA OFFICE0 [22] IN
RGB-D CASE. MS DENOTES MODEL SIZE (MB).
minimum cell size
PSNR↑
MS↓
4
42.08
29.4
8
41.26
10.1
16
37.12
5.8
SLAM and CaRtGS can be applied to monocular and RGB-
D cases, whereas others are exclusively utilized for RGB-D
data. We reproduce the results using the official code and run
all experiments on the same desktop computer. Since only
SplaTAM [13] and RTG-SLAM [15] support ScanNet++ [23]
dataset, we compare with these two methods on ScanNet++
in the RGB-D case.
B. Camera Tracking Accuracy
The evaluation of camera tracking accuracy across Tab. I, II
and III demonstrates the robust performance of RP-SLAM in
both monocular and RGB-D modes. As RP-SLAM builds upon
ORB-SLAM3 [1] for camera tracking, it inherently inherits
the well-established precision of ORB-SLAM3’s feature-based
pose estimation. This foundation ensures that RP-SLAM pro-
vides accurate camera pose estimation across diverse datasets,
as evidenced by the consistently low Absolute Trajectory Error
(ATE) values reported.
For monocular mode, RP-SLAM achieves an ATE of 0.63
on the Replica dataset (Tab. I) and 1.21 on the TUM dataset
(Tab. III). These results are comparable to or slightly higher
than state-of-the-art methods such as Photo-SLAM [16] and
CaRtGS [20] but significantly outperform the coupled ap-
proach like MonoGS [14]. The accuracy of camera tracking
in the coupled approach is contingent upon the availability of
a high-quality representation of the scene. In the monocular
case, MonoGS is unable to acquire accurate scene geometry
in a timely manner, resulting in a low-quality representation
of the scene. This, in turn, leads to poor camera tracking
accuracy. In the RGB-D mode, coupled methods, such as
MonoGS [14] and SplaTAM [13], are capable of achieving
relatively accurate scene representations through the utilization
of the depth map. Consequently, they are able to attain camera
tracking accuracies on the Replica [22] dataset that are compa-
rable to those of decoupled methods, as evidenced in Tab. II.
In challenging real-world scenarios, however, the decoupled
approaches do prove to be more effective, as evidenced by
c = 4
c = 8
Fig. 4. Effect of different minimum cell sizes on rendering high-resolution
image in ScanNet++ [23] dataset. When c = 4, the handwriting on the
whiteboard is observed to be more discernible. Zoom in for a clearer view.
the results presented in Tab. III, which demonstrate that our
approach achieves the highest ATE.
By decoupling camera tracking from Gaussians optimiza-
tion, RP-SLAM achieves a key benefit: real-time performance.
As illustrated in Tab. II, the system frame rate of RP-SLAM
is approximately 20 times higher than that of MonoGS [14]
and nearly 100 times higher than that of SplaTAM [13].
The decoupled design ensures that the computational load
associated with rendering and optimizing Gaussian primitives
does not interfere with the responsiveness of the camera
tracking pipeline. This approach allows RP-SLAM to maintain
competitive tracking accuracy while also achieving real-time
system frame rates.
C. Rendering Quality Results
The results of the rendering quality of RP-SLAM on the
Replica [22], TUM [21], and ScanNet++ [23] datasets in RGB-
D mode demonstrate that RP-SLAM is consistently effective in
achieving realistic reconstructions. With regard to the Replica
dataset (Tab. II), RP-SLAM achieves the highest average
PSNR (37.12) and the lowest LPIPS (0.056) while maintaining
a high SSIM (0.971). This is a notable improvement over
the results obtained by MonoGS [14], Photo-SLAM [16],
SplaTAM [13], and CaRtGS [20]. While RTG-SLAM [15]
achieves the highest SSIM, RP-SLAM produces results that
are nearly identical, with only one-sixth of its model size.
Moreover, CaRtGS attains the highest LPIPS in several scenar-
ios, yet its model size is approximately twice that of our own.
Similarly, on the TUM dataset (Tab. III), RP-SLAM achieves
a PSNR of 23.09, an SSIM of 0.805, and an LPIPS of 0.200.
In comparison to MonoGS, RP-SLAM demonstrates superior
performance in terms of PSNR, while exhibiting a model size
that is only 1 Mb larger. While SplaTAM and Photo-SLAM
achieve higher SSIM and LPIPS, respectively, their model
sizes exceed ours by a factor of 35 and 4, respectively. Finally,
on the ScanNet++ dataset (Tab. IV), RP-SLAM consistently
outperforms all baseline methods in both the training views
and the novel views, and maintains the lowest model size.
The superior rendering quality observed in these datasets
can be attributed to the combined effect of Efficient Incremen-
tal Mapping (EIM) and Dynamic Keyframe Window (DKW)
optimization. Adaptive sampling in EIM, guided by the image
gradient, ensures that regions with high-frequency textures
receive sufficient detail while reducing redundancy in regions
of lesser importance and ensures the efficient allocation of

<!-- page 9 -->
9
GT
MonoGS [14]
Photo-SLAM [16]
CaRtGS [20]
RP-SLAM (ours)
(a)
(b)
(c)
(d)
(e)
Fig. 5. Qualitative comparisons on Replica [22] dataset in the monocular case. The green dashed boxes in our method mark areas where RP-SLAM outperforms
other methods, such as sharper textures and fewer artefacts. Zoom in for a clearer view.
computational resources. Furthermore, the DKW employs the
use of co-visible keyframes to maintain consistency across
keyframes, thus ensuring the robustness of the reconstructed
scene and, to a certain extent, addressing the issue of forget-
ting that can occur during successive optimization processes.
Collectively, these two modules enable RP-SLAM to generate
realistic and consistent scene representations.
Furthermore, for the monocular case (Tab. I and III), RP-
SLAM attains a PSNR of 35.31, an SSIM of 0.940, and an
LPIPS of 0.067 on the Replica dataset, and on the TUM
dataset it achieves a PSNR of 21.89, an SSIM of 0.755
and an LPIPS of 0.209, consistently outperforming state-of-
the-art monocular methods. These results demonstrate that
accurate initialization of Gaussian primitives using the sparse
point cloud generated by ORB-SLAM3 provides a reliable
foundation for subsequent optimization. Concurrently, EIM
and DKW also facilitate efficient scene representation and
consistent scene optimization for the monocular case. This
combination allows RP-SLAM to overcome the inherent lim-
itations of monocular SLAM, such as geometric ambiguity
and lack of depth cues, to produce detailed and realistic
reconstructions.
The rendering results for all datasets demonstrate the ef-
ficacy of the core modules of RP-SLAM. The efficiency of
image sampling and map representation is enhanced through
the implementation of EIM, while ensuring the retention of
essential details within complex regions. Concurrently, the
DKW addresses the issue of forgetting and ensures the con-
sistency of the reconstructed scene, thus enabling RP-SLAM
to maintain robust performance over longer sequences. In
monocular case, the MKI ensures accurate initialization of
Gaussian primitives, thereby providing the requisite geometric
basis for high-fidelity mapping. Collectively, these modules
enable RP-SLAM to achieve state-of-the-art rendering quality
in both RGB-D and monocular cases.
D. Computational and Storage Efficiency
The computational and storage efficiency of RP-SLAM
is demonstrated through a comprehensive evaluation of the
Replica [22], TUM [21], and ScanNet++ [23] datasets, as
detailed in Tab. I to IV. The results demonstrate that RP-
SLAM ensures photorealistic reconstruction while maintaining
a compact model size and achieving a high frame rate in
comparison to state-of-the-art methods.
The high frame rates achieved by RP-SLAM can be at-
tributed to the decoupling of camera tracking from Gaussian
primitives optimization. By employing ORB-SLAM3 [1] for
direct camera pose estimation, RP-SLAM circumvents the
arduous iterative procedure necessitated for the optimization
of the camera pose with respect to the scene representation.
This separation permits the system to allocate computational
resources in a more efficient manner, thereby achieving real-
time performance. As illustrated in Tab. I, RP-SLAM attains
an average frame rate of 17.3 FPS in monocular mode on
the Replica dataset, which is considerably superior to that
of MonoGS, which exhibits a lower frame rate due to the
overhead of coupled optimization. Similarly, in RGB-D mode
(Tab. II), RP-SLAM achieves an average frame rate of 18.3
FPS, which outperforms MonoGS [14], RTG-SLAM [15], and
SplaTAM [13].

<!-- page 10 -->
10
GT
MonoGS [14]
Photo-SLAM [16]
RTG-SLAM [15]
CaRtGS [20]
RP-SLAM (ours)
(a)
(b)
(c)
(d)
(e)
Fig. 6. Qualitative comparisons on Replica [22] dataset in the RGBD case. The green dashed boxes in our method mark areas where RP-SLAM outperforms
other methods, such as sharper textures and fewer artefacts. Zoom in for a clearer view.
The compact model size of RP-SLAM is attributed to EIM,
which represents the scene in an efficient manner by reducing
redundancy without compromising the rendering quality. EIM
employs quadtree-based adaptive sampling oriented to the
image gradient, which ensures that regions with a high level
of detail are adequately sampled while avoiding unnecessary
repetitive sampling in regions with a more even texture.
Furthermore, the implementation of KNN-besed filtering facil-
itates the refinement of the Gaussian primitives set through the
elimination of redundancy, thereby enhancing the efficiency of
the map representation. As illustrated in Tab. I, the average
model size of RP-SLAM in monocular mode on the Replica
dataset is 11.3 Mb, which is notably smaller than that of Photo-
SLAM [16] (23.14 Mb) and that of CaRtGS [20] (15.3 Mb). In
RGB-D mode (Tab. II), the average model size of RP-SLAM is
10.0 Mb, which is significantly smaller than that of all baseline
methods. On the TUM dataset (Tab. III), RP-SLAM maintains
a compact model size of 6.3 Mb in monocular mode and 3.8
Mb in RGB-D mode, which outperforms all methods except
MonoGS [14]. Similarly, the smallest model size was obtained
on the ScanNet++ dataset (Tab. IV).
The combined impact of the decoupled sampling method
and EIM is further emphasized when considering the bal-
ance between computational efficiency and rendering qual-
ity. Despite achieving higher frame rates and smaller model
sizes, RP-SLAM does not compromise the quality of the
reconstructed scene. This balance is critical for applications
that require both real-time and high-fidelity reconstruction. In
conclusion, the computational and storage efficiency of RP-
SLAM is markedly enhanced by the decoupled method for
camera pose estimation and scene representation optimization,
in addition to EIM. Collectively, these approaches empower
RP-SLAM to attain real-time operation at high frame rates
while maintaining a compact model size, and furthermore,
provide realistic and consistent scene reconstruction.
E. Ablation Study
1) Different Modules Ablation: The ablation study pre-
sented in Tab. V evaluates the contributions of the Monocular
Keyframe Initialization (MKI), Efficient Incremental Map-
ping (EIM), and Dynamic Keyframe Window (DKW) in the
monocular case of RP-SLAM on Replica [22] Office0.
The analysis focuses on PSNR and model size (MS) met-
rics, demonstrating the roles of these modules in improving
rendering quality and model representation efficiency.
In order to establish a baseline for comparison, each of the
three modules of RP-SLAM is replaced with the corresponding
module of MonoGS [14], including random depth initializa-
tion, random sampling and fixed keyframe window. When all
three modules are disabled, the baseline achieves a PSNR of

<!-- page 11 -->
11
GT
SplaTAM [13]
RTG-SLAM [15]
RP-SLAM (ours)
(a)
(b)
(c)
(d)
Fig. 7. Qualitative comparisons on ScanNet++ [23] dataset in the RGBD case. The green dashed boxes in our method mark areas where RP-SLAM outperforms
other methods, such as sharper textures and fewer artefacts. Zoom in for a clearer view.
34.87 and a model size of 19.3 Mb. The lack of structured
initialization, adaptive sampling, and consistency optimization
results in suboptimal performance, with limited reconstruction
quality and inefficient map representation.
With only MKI enabled, the PSNR is increased to 36.67 and
the model size is moderately reduced to 16.8 Mb. MKI ensures
accurate placement of Gaussian primitives by leveraging the
sparse point cloud generated by ORB-SLAM3 [1], which
reduces the generation of redundant primitives to a certain
extent. This module provides a solid geometric foundation
for subsequent optimization, thereby improving reconstruction
quality while maintaining reasonable storage efficiency.
Enabling only EIM drastically reduces the model size to
15.6 Mb and achieves a comparable result of 34.23 in PSNR
to the baseline method. EIM employs quadtree-based adaptive
sampling to focus computational resources on regions with
high texture complexity, while minimizing redundant sampling
of smooth regions and further eliminating redundant primitives
through KNN filtering. Although the PSNR is not improved,
the representation efficiency improvement can be clearly seen
from the reduction of model size.
Activating only the DKW improves PSNR to 35.28 while
slightly reducing the model size to 18.9 Mb. The DKW
employs the use of co-visible keyframes to construct the
aforementioned window during the iteration process, thereby
mitigating the issue of forgetting and ensuring temporal con-
sistency during the mapping process, which in turn improves
the quality of the scene rendering.
RP-SLAM exhibits optimal performance with a PSNR of
37.74 and a model size of 11.8 Mb when all three mod-
ules (MKI, EIM, and DKW) are enabled. This configuration
demonstrates the efficacy of the intermodule synergy. MKI
ensures accurate initialization of Gaussian primitives, EIM
minimizes redundancy while concentrating resources on crit-
ical regions, and DKW maintains temporal consistency and
robustness against forgetting. Collectively, these modules em-
power RP-SLAM to attain the state-of-the-art rendering quality
while markedly enhancing storage and model representation
efficiency.
2) Different Mininum Cell Sizes Ablation: The ablation
study presented in Tab. VI examines the impact of minimum
cell size in the quadtree-based adaptive sampling strategy on
the performance of RP-SLAM in the RGB-D case. The min-
imum cell size determines the granularity of sampling during
the mapping process, which directly affects the reconstruction
quality (as measured by PSNR) and model efficiency (as
measured by model size). The results demonstrate a clear
trade-off between these two factors as the minimum cell size
is varied.
When the minimum cell size is set to 4, the system exhibits

<!-- page 12 -->
12
the highest PSNR of 42.08, indicating that the reconstruction is
of superior quality, with texture-rich regions being represented
in detail. This configuration is particularly well-suited to
the rendering of complex textures with high resolution in
ScanNet++ [23], thereby ensuring a high level of detail in
the reconstructed scene, as illustrated in Fig. 4. However,
for common scenes, particularly those of low resolution, this
results in an increase in the model size to 29.4 Mb, which
reflects overly dense sampling in highly detailed sections, as
illustrated in Fig.2 (a). Increasing the minimum cell size to 16
leads to a drastic reduction in the model size to just 5.8 Mb,
showcasing the efficiency of the adaptive sampling mechanism
in reducing redundancy. However, this configuration sacrifices
rendering quality, with the PSNR dropping to 37.12. The
coarser sampling reduces the system’s ability to capture fine-
grained details in complex regions, as shown in Fig.2 (c),
resulting in lower fidelity reconstructions. The configuration
with a minimum cell size of 8 exhibits a balanced performance
for common scenes, as evidenced by a PSNR of 41.26 and
a model size of 10.1 MB. This setup provides high-quality
reconstruction while maintaining the scene’s efficiency and
compactness.
By modifying the minimum cell size, the system can be
adapted to suit different scenarios, with the option of pri-
oritizing either reconstruction quality or model efficiency in
accordance with the specific requirements of the application.
F. Qualitative Results
The qualitative results presented in Fig. 5, Fig. 6 and Fig. 7
compare RP-SLAM with several state-of-the-art methods, in-
cluding MonoGS [14], Photo-SLAM [16], RTG-SLAM [15],
SplaTAM [13] and CaRtGS [20], in both monocular and RGB-
D scenarios on Replica [22] and ScanNet++ [23] datasets.
In the monocular case (Fig. 5), RP-SLAM demonstrates
superior performance in terms of texture clarity when com-
pared to the baseline methods. To illustrate, in column (a), the
remaining methods are largely unable to capture the intricate
textures of the sofa, floor and table. In contrast, RP-SLAM
accurately reconstructs the texture and edge shapes of these
objects, closely aligning with the ground truth. A similar trend
is observed in other scenes, where RP-SLAM produces clearer
and more complete reconstructions. In the RGB-D case (Fig. 6
and Fig. 7), RP-SLAM once again demonstrates a notable
enhancement in rendering quality relative to other comparable
methods. For instance, in Fig. 6 (c), RP-SLAM is capable of
rendering with greater clarity, as evidenced by the sharpness
of the clocks. Similarly, as illustrated in Fig. 7 (b), RP-SLAM
is capable of acquiring more detailed information, such as the
handwriting on the whiteboard, while simultaneously reducing
the occurrence of artefacts, such as the windows.
In both monocular and RGB-D cases, RP-SLAM achieves
superior quality performance by effectively balancing recon-
struction fidelity and efficiency. The integration of MKI, EIM
and DKW allows RP-SLAM to address the major limitations
of the other methods. MKI provides robust initialization of
Gaussian primitives, enabling accurate reconstruction even in
monocular mode, while EIM dynamically adapts the sampling
to focus on regions of interest, ensuring high quality recon-
struction with minimal redundancy. DKW ensures temporal
consistency and reduces artefacts caused by forgetting prob-
lems in sequential optimization. In conclusion, the qualitative
results substantiate the efficacy of our RP-SLAM in attaining
optimal rendering quality.
V. CONCLUSION
This paper introduces RP-SLAM, a 3D Gaussian splatting-
based visual SLAM system designed to achieve real-time
photorealistic scene reconstruction in monocular and RGB-D
settings. Through the decoupling of camera poses estimation
and scene optimization, RP-SLAM leverages feature-based
SLAM for efficient and robust tracking while utilizing adaptive
sampling and Gaussian primitives filtering to maintain high-
quality scene representation with minimal redundancy. The
dynamic keyframe window optimization ensures temporal
consistency, addressing the forgetting problem during sequen-
tial mapping. For monocular configurations, the proposed
Gaussian primitives initialization method provides a strong
geometric foundation, enabling accurate reconstructions. Ex-
tensive evaluations validate the effectiveness of RP-SLAM in
achieving state-of-the-art rendering quality and compact model
size.
Feature Work: Although this method is effective in balanc-
ing photorealistic reconstruction and model size, it is currently
unable to accommodate dynamic scenes. Our future research
will focus on extending the method to address this limitation.
REFERENCES
[1] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and J. D.
Tard´os, “Orb-slam3: An accurate open-source library for visual, visual–
inertial, and multimap slam,” IEEE Transactions on Robotics, vol. 37,
no. 6, pp. 1874–1890, 2021.
[2] Z. Teed and J. Deng, “Droid-slam: Deep visual slam for monocular,
stereo, and rgb-d cameras,” Advances in neural information processing
systems, vol. 34, pp. 16 558–16 569, 2021.
[3] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE
transactions on pattern analysis and machine intelligence, vol. 40, no. 3,
pp. 611–625, 2017.
[4] J. Engel, T. Sch¨ops, and D. Cremers, “Lsd-slam: Large-scale direct
monocular slam,” in European conference on computer vision. Springer,
2014, pp. 834–849.
[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[6] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “imap: Implicit mapping and
positioning in real-time,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2021, pp. 6229–6238.
[7] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and
M. Pollefeys, “Nice-slam: Neural implicit scalable encoding for slam,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022, pp. 12 786–12 796.
[8] A. Rosinol, J. J. Leonard, and L. Carlone, “Nerf-slam: Real-time
dense monocular slam with neural radiance fields,” in 2023 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS).
IEEE, 2023, pp. 3437–3444.
[9] L. Bai, C. Tian, J. Yang, S. Zhang, and W. Liang, “Neb-slam: Neural
blocks-based salable rgb-d slam for unknown scenes,” arXiv preprint
arXiv:2405.15151, 2024.
[10] H. Wang, J. Wang, and L. Agapito, “Co-slam: Joint coordinate and
sparse parametric encodings for neural real-time slam,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 13 293–13 302.

<!-- page 13 -->
13
[11] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM transactions on
graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.
[12] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Trans. Graph.,
vol. 42, no. 4, July 2023.
[13] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer,
D. Ramanan, and J. Luiten, “Splatam: Splat track & map 3d gaussians
for dense rgb-d slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 21 357–21 366.
[14] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian splatting
slam,” in Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2024, pp. 18 039–18 048.
[15] Z. Peng, T. Shao, Y. Liu, J. Zhou, Y. Yang, J. Wang, and K. Zhou, “Rtg-
slam: Real-time 3d reconstruction at scale using gaussian splatting,” in
ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.
[16] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-slam: Real-
time simultaneous localization and photorealistic mapping for monocular
stereo and rgb-d cameras,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 21 584–21 593.
[17] S. Ha, J. Yeon, and H. Yu, “Rgbd gs-icp slam,” in European Conference
on Computer Vision.
Springer, 2025, pp. 180–197.
[18] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim,
A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon,
“Kinectfusion: Real-time dense surface mapping and tracking,” in 2011
10th IEEE international symposium on mixed and augmented reality.
Ieee, 2011, pp. 127–136.
[19] A. Segal, D. Haehnel, and S. Thrun, “Generalized-icp.” in Robotics:
science and systems, vol. 2, no. 4.
Seattle, WA, 2009, p. 435.
[20] D. Feng, Z. Chen, Y. Yin, S. Zhong, Y. Qi, and H. Chen, “Cartgs:
Computational alignment for real-time gaussian splatting slam,” arXiv
preprint arXiv:2410.00486, 2024.
[21] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A
benchmark for the evaluation of rgb-d slam systems,” in 2012 IEEE/RSJ
international conference on intelligent robots and systems. IEEE, 2012,
pp. 573–580.
[22] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel,
R. Mur-Artal, C. Ren, S. Verma et al., “The replica dataset: A digital
replica of indoor spaces,” arXiv preprint arXiv:1906.05797, 2019.
[23] C. Yeshwanth, Y.-C. Liu, M. Nießner, and A. Dai, “Scannet++: A high-
fidelity dataset of 3d indoor scenes,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 12–22.
[24] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “Orb-slam: a versatile
and accurate monocular slam system,” IEEE transactions on robotics,
vol. 31, no. 5, pp. 1147–1163, 2015.
[25] R. Mur-Artal and J. D. Tard´os, “Orb-slam2: An open-source slam
system for monocular, stereo, and rgb-d cameras,” IEEE transactions
on robotics, vol. 33, no. 5, pp. 1255–1262, 2017.
[26] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “Dtam: Dense
tracking and mapping in real-time,” in 2011 international conference on
computer vision.
IEEE, 2011, pp. 2320–2327.
[27] B. Curless and M. Levoy, “A volumetric method for building complex
models from range images,” in Proceedings of the 23rd annual confer-
ence on Computer graphics and interactive techniques, 1996, pp. 303–
312.
[28] M. Nießner, M. Zollh¨ofer, S. Izadi, and M. Stamminger, “Real-time
3d reconstruction at scale using voxel hashing,” ACM Transactions on
Graphics (ToG), vol. 32, no. 6, pp. 1–11, 2013.
[29] O. K¨ahler, V. A. Prisacariu, C. Y. Ren, X. Sun, P. Torr, and D. Murray,
“Very high frame rate volumetric integration of depth images on mobile
devices,” IEEE transactions on visualization and computer graphics,
vol. 21, no. 11, pp. 1241–1250, 2015.
[30] J. Chen, D. Bautembach, and S. Izadi, “Scalable real-time volumetric
surface reconstruction.” ACM Trans. Graph., vol. 32, no. 4, pp. 113–1,
2013.
[31] M. Zeng, F. Zhao, J. Zheng, and X. Liu, “Octree-based fusion for
realtime 3d reconstruction,” Graphical Models, vol. 75, no. 3, pp. 126–
136, 2013.
[32] E. Vespa, N. Nikolov, M. Grimm, L. Nardi, P. H. Kelly, and S. Leuteneg-
ger, “Efficient octree-based volumetric slam supporting signed-distance
and occupancy mapping,” IEEE Robotics and Automation Letters, vol. 3,
no. 2, pp. 1144–1151, 2018.
[33] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-Y.
Lin, “inerf: Inverting neural radiance fields for pose estimation,” in 2021
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS).
IEEE, 2021, pp. 1323–1330.
[34] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, “Barf: Bundle-adjusting
neural radiance fields,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2021, pp. 5741–5751.
[35] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, “Go-slam: Global
optimization for consistent 3d instant reconstruction,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 3727–3737.
[36] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-slam:
Photo-realistic dense slam with gaussian splatting,” arXiv preprint
arXiv:2312.10070, 2023.
[37] L. Zhu, Y. Li, E. Sandstr¨om, K. Schindler, and I. Armeni, “Loop-
splat: Loop closure by registering 3d gaussian splats,” arXiv preprint
arXiv:2408.10154, 2024.
[38] E. Sandstr¨om, K. Tateno, M. Oechsle, M. Niemeyer, L. Van Gool, M. R.
Oswald, and F. Tombari, “Splat-slam: Globally optimized rgb-only slam
with 3d gaussians,” arXiv preprint arXiv:2405.16544, 2024.
[39] F. A. Sarikamis and A. A. Alatan, “Ig-slam: Instant gaussian slam,”
arXiv preprint arXiv:2408.01126, 2024.
[40] N. Ravi, J. Reizenstein, D. Novotny, T. Gordon, W.-Y. Lo, J. Johnson,
and G. Gkioxari, “Accelerating 3d deep learning with pytorch3d,”
arXiv:2007.08501, 2020.
