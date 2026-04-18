<!-- page 1 -->
SIREN: Semantic, Initialization-Free Registration of
Multi-Robot Gaussian Splatting Maps
Ola Shorinwa, Jiankai Sun, Mac Schwager, Anirudha Majumdar
Submap 𝑅!
View 1
View 2
Submap 𝑅$
View 1
View 2
𝑅!
𝑅$
Registration
Robust
SIREN
Initialization-Free
Semantics-Grounded
𝑅!
𝑅$
Box
Egg
Multi-Robot Gaussian 
Splatting Mapping
Fig. 1: SIREN enables robust registration (i.e., fusion) of multi-robot Gaussian Splatting maps, with no access to camera poses,
images, and inter-map relative poses, via semantics-grounded optimization centered on feature-rich regions of each map.
Abstract—We present SIREN for registration of multi-robot
Gaussian Splatting (GSplat) maps, with zero access to camera
poses, images, and inter-map transforms for initialization or fusion
of local submaps. To realize these capabilities, SIREN harnesses
the versatility and robustness of semantics in three critical ways to
derive a rigorous registration pipeline for multi-robot GSplat maps.
First, SIREN utilizes semantics to identify feature-rich regions
of the local maps where the registration problem is better posed,
eliminating the need for any initialization which is generally
required in prior work. Second, SIREN identifies candidate
correspondences between Gaussians in the local maps using
robust semantic features, constituting the foundation for robust
geometric optimization, coarsely aligning 3D Gaussian primitives
extracted from the local maps. Third, this key step enables
subsequent photometric refinement of the transformation between
the submaps, where SIREN leverages novel-view synthesis in
GSplat maps along with a semantics-based image filter to compute
a high-accuracy non-rigid transformation for the generation of a
high-fidelity fused map. We demonstrate the superior performance
of SIREN compared to competing baselines across a range of
real-world datasets, and in particular, across the most widely-
used robot hardware platforms, including a manipulator, drone,
and quadruped. In our experiments, SIREN achieves about 90x
smaller rotation errors, 300x smaller translation errors, and
44x smaller scale errors in the most challenging scenes, where
competing methods struggle. We will release the code and provide
a link to the project page after the review process.
Index Terms—Multi-Robot Mapping, Gaussian Splatting, Map
Registration.
I. INTRODUCTION
In robotics, traditional map representations such as point-
cloud and voxel maps constitute a critical component of
the robotics stack, enabling downstream behavior prediction,
planning, and control across many problem domains, e.g., nav-
igation and manipulation. However, these map representations
often lack the expressiveness required to capture high-fidelity
visual details and semantics [20], limiting their applications in
fine-grained robotics tasks, e.g., in dexterous open-vocabulary
manipulation [41]. To address these fundamental limitations,
recent robotics research has adopted radiance fields as flexible,
high-fidelity 3D scene representations, e.g., in robot navigation
[7, 34] and manipulation [36, 43]. Radiance fields, e.g., neural
radiance fields (NeRFs) [27] and Gaussian Splatting (GSplat)
[20], are trained entirely from monocular images, typically
collected by a single robot on a single deployment.
However, practical real-world robot mapping requires multi-
ple deployments and multiple robot platforms, especially when
mapping large-scale areas. For example, mobile robots have
a limited battery life, while fixed-base robotic manipulators
have a limited workspace, making map registration a necessity
for covering large-scale areas. Fusing map information across
multiple robot platforms and deployments remains a key
challenge, particularly with radiance field maps. Prior work
has explored fusing multiple radiance field maps [6, 52];
however, these methods either require a good initialization
of inter-map correspondences or access to the camera poses
and images, which is often unavailable in many practical
situations. Moreover, these methods often fail in unstructured
real-world environments, an important operational domain for
robots. To address these challenges, we introduce SIREN, a
semantic, initialization-free registration algorithm for multi-
robot Gaussian Splatting maps.
Although often unexploited, many real-world scenes contain
rich semantic information, e.g., associated with objects such
as vehicles, people, utensils, and vegetation. Leveraging this
key insight, SIREN trains a semantic GSplat to directly embed
semantic features in GSplat maps and subsequently uses the
arXiv:2502.06519v1  [cs.RO]  10 Feb 2025

<!-- page 2 -->
inherent semantics in the local maps to identify feature-rich
regions of the local maps, providing a more reliable set of
Gaussians for the identification of candidate correspondences.
This critical design choice underpins the superior performance
of SIREN. Specifically, the core challenge in map registration
can be largely attributed to the difficulty in identifying accurate
correspondences between points [44]. In fact, given accurate
correspondences, the map registration problem can be solved
efficiently in closed-form. By centering the registration problem
on feature-rich regions of the local maps to derive a reliable
set of correspondences, SIREN addresses this core challenge.
Subsequently, SIREN harnesses the robustness of semantic
features to formulate a geometric optimization problem for
a coarse non-rigid relative transformation aligning Gaussian
primitives across the local maps. We solve the geometric
optimization problem efficiently in closed-form. Although the
coarsely aligned map may be satisfactory in certain scenarios,
the fused map often lacks the photorealism afforded by GSplat
maps. To address this weakness, SIREN leverages novel-view
synthesis of GSplat maps to render images across the local
maps, used as a supervision signal for computing a high-
accuracy relative transformation. To guard against the impacts
of inaccurate renderings from the local GSplats, we utilize
a semantics-based image filter to identify reliable candidate
images, which we use for supervision. Consequently, SIREN
generates photorealistic fused GSplat maps from the local
multi-robot maps, illustrated in Figure 1, where we show the
key components of SIREN.
We demonstrate the superior effectiveness of SIREN com-
pared to both existing GSplat registration methods and classical
point-cloud registration methods across different real-world
datasets, including standard benchmarks for radiance fields and
data collected across three different robot hardware platforms:
a quadruped, drone, and fixed-base manipulator. In almost all
settings, SIREN achieves lower rotation, translation, and scale
errors compared to all baselines, especially in the quadruped
mapping task, where SIREN achieves about 90x lower rotation
error, 300x lower translation error, and 44x lower scale errors.
We summarize our contributions:
• We introduce a semantics-grounded feature extraction and
matching method for GSplat map registration, centering
the registration problem on feature-rich regions, addressing
a key challenge of registration algorithms.
• We derive a Gaussian-to-Gaussian registration procedure
for coarse alignment, utilizing semantic correspondence
to identify and mitigate the effects of outliers in the
registration process.
• We present a photometric registration procedure, leverag-
ing novel-view synthesis of GSplats and a semantics-
based image filter to compute high-accuracy relative
transformations, generating photorealistic fused maps.
• Together, these components constitute SIREN, enabling
the registration of multi-robot GSplat maps, with zero
access to source images or poses and no inter-map relative
pose initialization.
II. RELATED WORK
Radiance Fields. Neural radiance fields (NeRFs) [27] signifi-
cantly outperform traditional 3D scene reconstruction methods,
such as those based on point clouds and voxels, generating
photorealistic renderings, which capture intricate levels of
geometric and visual details. NeRFs represent a scene using
volumetric density and color fields over a 5D input space,
comprising a 3D location and a 2D viewing direction. NeRFs
parameterize each field using multi-layer perceptrons (MLPs)
trained through gradient descent. Although NeRFs achieve
remarkable high-fidelity reconstructions, NeRFs are limited by
significant training time and slow rendering speeds [53, 50, 2].
Gaussian Splatting [20] was introduced to address these
limitations. GSplats represent the scene using ellipsoidal
primitives, each with a mean and covariance (spatial and
geometric parameters) and opacity and spherical harmonic
parameters (visual-related parameters). GSplats generate high-
fidelity scene renderings at real-time speeds with generally
faster training times compared to NeRFs. Recent work has
improved the geometric accuracy of GSplats [14, 16], in
addition to eliminating high-frequency artifacts [51, 23].
Semantic Radiance Fields. Large vision-language models,
e.g., CLIP [35] and DINO [5, 30] have demonstrated the
effectiveness of large-scale pretraining in learning robust visual
and language features, enabling object detection [13, 28], object
segmentation [49, 25], and image captioning [29, 26]. Prior
work has examined grounding the 2D image-language features
from vision-language foundation models in 3D radiance fields.
CLIP-NeRF [47], DFF [22], and LERF [21] train NeRFs with
CLIP image-language features, enabling open-vocabulary object
segmentation and scene-editing. Similarly, subsequent work has
enabled distillation of semantic features into GSplats [32, 55],
with similar open-vocabulary object segmentation quality, albeit
at much faster rendering rates [42]. Moreover, prior work
has leveraged semantic radiance fields to enable GSplat-based
world models [24] and open-vocabulary robotic manipulation
in NeRFs [41, 36] and GSplat environments [43, 17]. In this
work, we leverage semantic radiance fields for registration of
3D maps, which has not been explored in prior work, to the
best of our knowledge.
Point Cloud Registration. The Iterative Closest Point (ICP)
algorithm [3] has proven to be notably effective for point cloud
registration, despite its simplicity. However, ICP generally
requires a good initial solution, which is often computed using
global registration techniques, e.g., RANSAC [10, 15] and FGR
[54]. Many variants of ICP have been introduced to improve
its robustness [8, 37, 4, 31], leveraging the local color and
geometry of the constituent points for faster convergence. More
recently, learning-based methods [48, 11, 33] have emerged for
point cloud registration, utilizing convolutional neural networks
(CNNs) and transformers for feature extraction and feature
matching to compute the correspondences between points.
Registration of Radiance Fields. Training large-scale radiance
fields is often infeasible, due to computational resource

<!-- page 3 -->
constraints. Consequently, Nerf2nerf [12] aligns individually-
trained NeRFs with different frames into a shared reference
frame, by extracting the geometry of the scene from the
NeRF as a surface field. Nerf2nerf requires human annotation
of keypoints within each NeRF for registration, posing a
practical challenge. Similarly, the NeRF registration methods
in [18, 9] computes spatial features of NeRFs using learned
feature descriptors 3D primitives and subsequently estimates
the transformation between the source and target NeRFs using
the Kabsch-Umeyama algorithm [46] or RANSAC.
More recent work has explored the registration of GSplats.
LoopSplat [56] and PhotoReg [52] compute the optimal trans-
formation between GSplat maps by minimizing the rendering
loss but require access to the set of camera poses (keyframes)
of each GSplat. In contrast, GaussReg [6] computes a coarse
transformation between two GS-maps using a geometric trans-
former [33], which is refined with a 2D convolutional neural
network augmented with a geometric transformer, without
access to the camera poses. In contrast to these existing
methods, SIREN leverages the semantics inherent to GSplat
maps to identify regions of overlap and to coarsely align GSplat
maps, eliminating the need for access to camera poses or
images. Moreover, unlike GaussReg, SIREN does not require a
separate training procedure for the learned CNN and geometric
transformer models.
III. PRELIMINARIES
We present relevant notation used in this paper. We denote
the trace of a function by trace and the determinant and
variance of a matrix by det and var, respectively. Further,
we denote the strictly positive orthant by R++. Next, we
provide a brief introduction to Gaussian Splatting. Gaussian
Splatting represents non-empty space in a scene using a
set of ellipsoidal primitives, each parameterized by a mean
µ ∈R3, a covariance Σ ∈R3×3 defined by a rotation matrix
H ∈SO(3) and a diagonal scaling matrix Λ ∈R3×3, an opac-
ity parameter α ∈[0, 1], and spherical harmonic parameters.
These attributes are optimized via gradient descent on the
loss function: Lgs = (1 −λ) P
I∈D
I −ˆI

1 + λLD−SSIM,
over the training dataset D, where λ ∈(0, 1) represents the
relative weight term and LD−SSIM represents the differentiable
structural similarity loss index measure. The first term in the
rendering loss represents the photometric loss between the
ground-truth image and the rendered image, generated via a
tile-based rasterization procedure, given a camera pose.
IV. SIREN FOR GSPLAT MAP REGISTRATION
We introduce SIREN, our semantics-grounded, initialization-
free registration algorithm for GSplat maps. At its core,
SIREN leverages open-vocabulary semantics within a prin-
cipled optimization-based framework to enable the robust
registration of multi-robot GSplat maps. SIREN utilizes an
SCF procedure, composed of: (a) Semantic feature extraction
and matching, (b) Coarse Gaussian-to-Gaussian geometric
registration, and (c) Fine photometric registration, illustrated
in Figure 2. Here, for simplicity, we discuss the registration
pipeline in the problem setting with two multi-robot maps,
where we seek to register a source GSplat map to a target
GSplat map. However, the discussion applies to the registration
of multiple local multi-robot maps.
In the first step, SIREN identifies corresponding pairs of
Gaussians in a pair of GSplat maps by examining the similarity
between the semantic features of the ellipsoids. Subsequently,
given the set of corresponding Gaussians, SIREN solves a
Gaussian-to-Gaussian optimization problem to compute the
optimal transformation aligning the pair of multi-robot maps
with a robust objective function, which leverages the semantic
similarity between each pair of ellipsoids to guard against the
impacts of outliers. In the last step, SIREN harnesses the novel-
view synthesis capabilities of Gaussian Splatting to render
candidate images for image-to-image registration, enabling fine
registration of both maps via a structure-from-motion-based
approach. In this stage, SIREN utilizes image-level semantic
features to identify pairs of corresponding images, critical to
the robust matching of local features such as corners, edges,
and blobs between the images. We discuss each of these steps
in greater detail.
A. Semantic Feature Extraction and Matching
Noting that semantics underpin SIREN, we begin with a
discussion of the semantic distillation procedure utilized by
SIREN in grounding 2D semantic information from the vision-
language model in GSplat maps, where we associate semantic
embeddings with each ellipsoid in the GSplat map.
Semantic Gaussian Splatting. Existing methods for training
semantic GSplat models generally train auxiliary models, e.g.,
autoencoders or CNNs, for dimensionality reduction of the
semantic features from vision-language models to compute
lower-dimensional semantic features, which are distilled into
the GSplat model [32, 55]. These methods require relatively
significant computation time and GPU memory, which we
seek to avoid in SIREN. Consequently, we take a different
approach to semantic distillation. In SIREN, we simultaneously
train a semantic field ψ alongside the GSplat model. The
semantic field ψ : R3 7→Rd maps 3D points to d-dimensional
semantic features, where d is determined by the vision-
language foundation model, e.g., d = 512 or 1024 in CLIP. We
parameterize ψ with a multi-resolution neural hashgrid, trained
with the GSplat model, using the loss function:
L = Lgs + γ
X
I∈D
If −ˆIf

2
F −β
X
I∈D
ϕ(If, ˆIf),
(1)
where If ∈RW ×H×d and
ˆIf ∈RW ×H×d represent the
ground-truth and predicted semantic feature maps associated
with each image in the training dataset D, γ and β represent
relative weight terms, and ϕ represents the cosine-similarity
function between each semantic feature in If and ˆIf.
To predict the semantic feature map associated with each
training image, we leverage a key insight of Gaussian Splatting:
Gaussian Splatting provides highly-accurate depth estimates,
even without any depth supervision [16]. This key insight

<!-- page 4 -->
Semantics Extraction and Matching
Submap 𝑅!
Submap 𝑅"
Semantically
Similar 
Gaussians
Candidate Matches
Coarse Gaussian Registration
Aligning…
Optimal 
Rotation
1
2
Optimal 
Scale 
and 
Translation
Optimization
Fine Photometric Registration
Submap 𝑅!
Submap 𝑅"
Bundle 
Adjustment
𝑃!
𝑃"
𝑃"
𝑃!
SIREN
SIREN
𝑃"
𝑃!
Fig. 2: SIREN consists of three steps: (a) semantic feature extraction and matching of Gaussians across the local maps, (b)
coarse Gaussian-to-Gaussian registration for coarsely aligning the local maps, (c) fine photometric registration for high-accuracy
fusion of the local maps, through image-to-image registration and bundle adjustment.
enables us to avoid training proposal networks (as required in
NeRFs) that generate samples of the termination points of rays
associated with each pixel in the rendered image of a camera,
ultimately enabling SIREN to avoid significant compute and
memory overhead associated with training proposal networks.
As such, given a camera pose, we back-project points from
the image plane to the 3D world and pass these points into
ψ to predict the semantic feature associated with these points.
We augment each pixel in the image plane with its semantic
features to obtain the semantic feature map. Training the GSplat
with the semantic component does not adversely impact the
photometric performance of the GSplat, enabling us to utilize
the same hyperparameters and adaptive densification procedure
used in the original GSplat work [20].
Feature Extraction. In the feature extraction and matching step,
SIREN identifies feature-rich areas of the scene via semantic
localization, to improve the robustness of the subsequent
optimization-based registration steps, as the feasibility and
convergence of the optimization problems significantly depend
on the presence of informative features. Given a trained
semantic GSplat model, we augment each Gaussian with a
semantic attribute, computed by querying the semantic field ψ
at the mean µ of each Gaussian. Subsequently, from a set of
open-vocabulary queries, we compute the semantic relevancy
score between each Gaussian and the natural-language query
by taking the pairwise softmax over the cosine-similarity
between the semantic feature of each Gaussian and the semantic
embedding associated with the text query and the cosine-
similarity between the semantic feature of each Gaussian and
the semantic embedding associated with a generic or null text
query (i.e., a text query for a generic object or an object a user
does not want to localize) [21]. Depending on the quality of
the local multi-robot maps, we can post-process the resulting
set of Gaussians to either inflate the set by incorporating other
Gaussians in close proximity to the initial set (based on the
geometric or semantic distance) or deflate the set by removing
Gaussians considered to be outliers (based on statistics, e.g.,
the standard deviation of the distance between neighboring
Gaussians).
Feature Matching. Given the set of extracted Gaussians for
each map, we match Gaussians from the source GSplat map
to the target GSplat map, resulting in a set of correspondences
E where (i, j) ∈E indicates that Gaussian i in the source
GSplat map corresponds to Gaussian j in target GSplat map.
To identify the candidate matches in E, we compute the cosine-
similarity between the semantic embeddings of the Gaussians
in the source and target maps. We match each Gaussian in
the source map to a random set of M Gaussians in the target
map, sampling among the target Gaussians (i.e., Gaussians
in the target map) that are within a specified distance from
the source Gaussian (i.e., Gaussian in the source map). In the
sampling step, SIREN can utilize a uniform distribution or
a distribution where the probability values are proportional
to the cosine-similarity values between the source and target
Gaussians. For computational reasons, this operation can be
performed using efficient data structures such as KD-trees. In
addition, we can repeat the procedure to match each Gaussian
in the target GSplat map to a set of Gaussians in the source
GSplat map, taking care to ensure that E does not contain
any duplicate entry. Moreover, the matching process can be
augmented with geometric information, by selecting candidate
matches using geometric descriptors, such as the Fast Point
Feature Histograms (FPFH) descriptors [38]. We denote the
set of Gaussians in the source map present in E by P and the
set of Gaussians in the target map present in E by Q.
B. Coarse Gaussian-to-Gaussian Registration
We use the output from the feature matching step to compute
an initial non-rigid transformation, consisting of a scale sc ∈R,
a rotation matrix R ∈SO(3), and a translation vector t ∈R3,
aligning the Gaussians in the source GSplat map to the
Gaussians in the target GSplat map. Since computing this
transformation using all the Gaussians is intractable in general,
we solve for this transformation using the Gaussians in P

<!-- page 5 -->
and Q, a feature-dense, much smaller set of Gaussians, a
design choice that not only reduces the computational cost,
but also improves the feasibility and convergence properties of
the resulting optimization problem. Specifically, we formulate
the coarse Gaussian-to-Gaussian registration problem as an
optimization problem over the transformation parameters, given
by:
minimize
sc∈R++,R∈SO(3),t∈R3
1
2
X
(i,j)∈E
wij

∥scRpi + t −qj∥2
2
+
s2
cRΣpiRT −Σqj
2
F

,
(2)
where pi ∈R3 denotes the mean of Gaussian i in the source
map, qj ∈R3 denotes the mean of Gaussian j in the target map,
and Σpi and Σqj denote the covariance of Gaussian i in the
source map and Gaussian j in the target map, respectively. We
introduce the weight wij in (2) to increase the robustness of the
optimization problem to outliers (i.e., false correspondences),
by reducing the influence of outliers. We define wij to be
proportional to the cosine-similarity between the semantic
embeddings of Gaussian i in P and Gaussian j in Q. The
optimization problem in (2) is challenging to solve in general,
necessitating the derivation of a less challenging formulation.
Noting that the covariance of the Gaussians in the source and
target maps can be expressed in the form Σpi = HpiΛpiΛT
piHT
pi
and Σqj = HqjΛqjΛT
qjHT
qj, where Hpi and Hqj denote the
orientation of Gaussian i in the source map and Gaussian j in
the target map, respectively, and Λpi and Λqj denote the scale
of the Gaussians, we express the problem in (2) in the form:
minimize
sc∈R++,R∈SO(3),t∈R3
1
2
X
(i,j)∈E
wij

∥scRpi + t −qj∥2
2
+
scRHpiΛpi −HqjΛqj
2
F

,
(3)
which can be solved efficiently in closed-form, which we show
in Appendix A, with:
R⋆
c = UcΘcV ⊤
c ,
(4)
s⋆
c =
trace(ΘcΣ)
trace

W ˇP ⊤ˇP + P
(i,j)∈E wij ˇH⊤
pi ˇHpi
,
(5)
t⋆
c = ˜µQ −s⋆
cR⋆˜µP,
(6)
where UcΣcV ⊤
c
= ˇQW ˇP ⊤+ P
(i,j)∈E wij ˇHqj ˇH⊤
pi, computed
via
the
singular
value
decomposition
(SVD),
and
Θc = diag(1, 1, det(UcV ⊤
c )). We define ˜µP
and ˜µQ as
the weighted average of the means of the Gaussians in P
and Q, with weights wij for Gaussian i in P and Gaussian
j in Q. Further, ˇP ∈R3×N and ˇQ ∈R3×N represent the
zero-centered Gaussians in P and Q, respectively, with the
ith column of ˇP given by ˇPi = pi −˜µP and similarly for the
jth column of ˇQ. We introduce the terms ˇHpi ∈R3×3 and
ˇHqj ∈R3×3 to simplify notation, with: ˇHpi = HpiΛpi and
ˇHqj = HqjΛqj. In addition, W ∈RN×N denotes the diagonal
weight matrix, Wkk = wk, with wk = wij, ∀k = (i, j) ∈E.
Although the resulting solution is optimal for the problem in
(3), the solution of (3) might not be optimal for the registration
of the two sets of Gaussians, i.e., P and Q, given that C might
contain spurious correspondences. To improve the robustness of
SIREN to spurious correspondences, we utilize RANSAC [10]
when solving the optimization problem in (3). With RANSAC,
we iteratively update the correspondences in C to remove
false correspondences and compute an optimal transformation
associated with the resulting set of correspondences.
C. Fine Photometric Registration
In the preceding coarse registration step, SIREN computes a
transformation aligning the source and target maps using only
the geometric attributes of the Gaussians in each map. The
coarse registration step fails to leverage the highly-informative
visual features inherent in the GSplat maps, effectively limiting
the accuracy of the estimated transformation. To overcome
this limitation, SIREN harnesses the novel-view synthesis
capability of GSplat maps to generate photorealistic images
and optimizes over the resulting set of rendered images
to compute a transformation consistent with the rendered
images from the source and target maps. The fine photometric
registration procedure employs a lightweight structure-from-
motion framework to minimize the computation costs, while
improving the fidelity of the registered maps. This procedure
consists of the following steps: (i) image generation, (ii) image
registration and triangulation, and (iii) bundle adjustment,
which we discuss in the rest of this section.
Image Generation and Matching. The fine registration
procedure begins with the identification of a set of images
with common features across the source and target maps,
constituting arguably the most important step of the fine
registration procedure. In particular, the feasibility of the
fine registration procedure hinges on matching corresponding
features across all images in the set. In general, identifying
good candidate images for the matching process is challenging,
especially without any prior knowledge of the region of overlap
between the source and target maps. To address this challenge,
we leverage the semantic submap extracted in the first stage of
SIREN to identify a region of overlap between the source and
target maps. Subsequently, we exploit novel-view synthesis in
Gaussian Splatting to render images at corresponding poses
in both maps, by transforming the camera pose in one map
to the associated camera pose in the other map, utilizing the
coarse registration result to compute the corresponding pose.
With this approach, not only do the resulting images contain
common features from the overlapping region, the images also
contain a dense set of features, associated with the semantic
submap. However, the pair of rendered images may not contain
sufficient matches, which could degrade the accuracy of the
fine registration procedure. To mitigate this risk, we harness
image semantics in vision foundation models to evaluate the
similarity between each pair of rendered images, retaining
only sufficiently similar images. In this work, we use CLIP
along with the cosine-similarity metric, given that the image
embeddings of CLIP were trained with a cosine-similarity loss

<!-- page 6 -->
function; however, other vision foundation can also be used,
e.g., [5].
Image Registration and Triangulation. Following the gen-
eration of corresponding images, we extract features from all
images using the learned feature extractors NetVLad [1] for
global image-level descriptors and SuperPoint [39] for local
features, which we found to be more robust compared to
classical feature extractors, e.g., SIFT [19]. Subsequently, we
match features across all images using [39]. From correspond-
ing features, we estimate the relative pose of the camera and
the estimated 3D locations of the feature points via image
registration and triangulation, yielding an initial estimate of
the camera pose associated with each image in a common
reference frame.
Bundle Adjustment. The image registration step does not
always provide high-accuracy camera pose estimates. Hence,
we refine the estimated camera poses via bundle adjustment,
i.e., we optimize over the camera pose and the 3D locations
of the feature points jointly through non-linear optimization.
For brevity, we do not discuss the bundle adjustment problem
in greater detail, noting its extensive discussion in prior work,
e.g., [40]. Although non-convex, the optimization problem
can be solved efficiently via iterative methods, such as the
Levenberg-Marquardth method, which we employ in this work.
From the bundle adjustment optimization problem, we compute
the camera poses associated with each image in an arbitrary
common frame B. Given the camera poses expressed in A and
the corresponding poses in the source and target maps, we can
compute an optimal transformation for registering A to either
the source frame (frame Bs) or the target frame (frame Bt)
from the following registration problem in SE(3):
minimize
sf ∈R++,R∈SO(3),t∈R3
1
2
X
(i,j)∈V

∥sfRai + t −bj∥2
2
+βij
RRci −Rdj
2
F

,
(7)
where sf, R, and t denote the scale, rotation, and translation
parameters, respectively, V denotes the set of edges between
the camera poses expressed in A and the corresponding
poses in either the source or target frame, with ai ∈R3 and
bj ∈R3 denoting the origin of the camera in A and the
origin of the camera in the frame Bs or Bt, respectively, and
Rai and Rbj denoting the associated rotation matrices. We
introduce the weight parameter βij ∈R++, which determines
the contribution of the rotation-error component. In general,
the optimization problem in (7) cannot be solved in closed-
form. Solving (7) generally requires an iterative optimization
method, e.g., sequential convex programming methods or
Riemannian optimization methods. However, as βij approaches
zero, ∀(i, j) ∈V, the optimal solution (7) approaches a limit
point, with:
R⋆
f →UfΘfV ⊤
f , s⋆
f →trace(ΘfΣf)
trace
  ˇA⊤ˇA
,
t⋆
f →µB −s⋆
fR⋆
fµA,
(8)
where UfΣfV ⊤
f = ˇB ˇA⊤, Θf = diag(1, 1, det(UfV ⊤
f )), µA
and µB denote the mean of the camera origins in frames
A and B, respectively, and the ith column of ˇA ∈R3×N
and the jth column of ˇB ∈R3×N are given by ai −µA
and bj −µB, respectively. The limit point follows from the
derivation in Section IV-B and [46]. We can compose the
pairwise transformations between frame A and frames Bs and
Bt to compute a transformation from Bs and Bt. We apply
the resulting transformation to the source map to express the
source and target maps in a common frame and subsequently
merge the resulting maps to obtain a composite GSplat map.
Following the registration procedures, the composite map can
be finetuned with new or existing data, which we explore in our
experiments in Appendix B-D. We summarize the procedures
in SIREN in Algorithm 1.
Algorithm 1: SIREN: Multi-Robot Map Registration
Input: Local GSplat Maps G1, G2;
Output: Fused GSplat Map Gf;
// Semantic Feature Extraction and
Matching
Correspondence Set C ←GetCorrespondence(G1, G2);
// Coarse Registration
// Compute the Optimal Rotation
R⋆
c ←Procedure (4);
// Compute the Optimal Scale
s⋆
c ←Procedure (5);
// Compute the Optimal Translation
t⋆
c ←Procedure (6);
// Fine Registration
// Get Images
Ds ←Render(G1, G2, R⋆
c, s⋆
c, t⋆
c);
// Refine Transformation
(R⋆
f, s⋆
f, t⋆
f) ←Procedure (8);
// Fuse Local Maps
Gf ←Fuse(G1, G2, R⋆
f, s⋆
f, t⋆
f);
V. EXPERIMENTS
We examine the performance of SIREN in comparison to
existing registration methods for Gaussian Splatting and point
clouds. Specifically, we compare two variants of SIREN—
i.e., SIREN-NR, which solves the optimization problem (3) in
closed-form without RANSAC, and SIREN-R, which utilizes
RANSAC for coarse registration—to the GSplat registration
methods GaussReg [6] and PhotoReg [52], in addition to
RANSAC-based global registration (RANSAC-GR) [10, 15],
Fast Global Registration (FGR) [54], and variants of the
Iterative Closest Point (ICP) [37, 31]. We evaluate each method
not only on standard benchmark datasets for radiance fields,
but also on real-world data collected by heterogeneous robot
platforms, including a quadruped, drone, and manipulator (in
the case of SIREN). In all our experiments, we only require the
trained GSplat models as input; however, some of the baselines
require access to the set of camera poses, which we provide

<!-- page 7 -->
when evaluating these methods. Further, we ablate the different
components of SIREN, to quantify the relative improvements
in performance provided by each component, and examine
the gains in visual fidelity afforded by finetuning the fused
model. We provide these results in Appendix B, as well as
additional discussion of the results presented in this section.
Lastly, we demonstrate SIREN in collaborative multi-robot
mapping, where the mapping task cannot be accomplished by
a single robot, necessitating mapping with multiple robots for
task success.
Experimental Setup and Metrics. For the real-world robot
data, we utilize the Unitree Go1 Quadruped and a Modal
AI drone with an onboard camera and the Franka Panda
manipulator with a wrist camera to collect RGB images. In
addition, we evaluate all methods on the real-world scenes
in the Mip-NeRF360 dataset [2], a state-of-the-art benchmark
dataset for neural rendering. We train the GSplat models using
the original implementation provided by the authors of [20] for
baselines which require this pipeline and utilize Nerfstudio [45]
for SIREN. We execute SIREN on a desktop computer with a
24GB NVIDIA GeForce RTX 3090 GPU and the baselines on
an H20 GPU after training the GSplat maps for 30000 iterations.
We note that in robotics, the geometric fidelity of robot’s map is
of significant importance for effective localization and collision
avoidance. Hence, we compare all methods in terms of the
rotation error (RE) [deg.], translation error (TE), and scale error
(SE) [in non-metric units] attained by each method, in addition
to the computation time (CT) [sec.]. Moreover, we examine
the photometric quality of the fused maps generated by each
method, computing the peak signal-to-noise ratio (PSNR), the
structural similarity index measure (SSIM), and the learned
perceptual image patch similarity (LPIPS), standard metrics in
the computer vision community for assessing visual fidelity.
We provide color-coded results for each metric with the red
shade denoting the top-performing statistic, the yellow shade
denoting the second-best, and the green shade denoting the
third-best. In all the registration methods, we do not pre-process
the individual submaps to remove floaters (i.e., non-existing
geometry). Consequently, floaters present in these submaps are
retained in the fused map.
A. Mip-NeRF360 Dataset
We utilize the Playroom, Truck, and Room scenes in the Mip-
NeRF360 Dataset. These real-world scenes were all collected in
realistic settings with natural lighting effects, both indoors and
outdoors. While the Playroom and Room scenes were captured
indoors, the Truck scene was captured outdoors. We split the
datasets into two subsets with varying overlap. Specifically,
the first subset of the Truck scene captures the left side of the
truck, while the second subset captures the right side of the
truck. The only overlap between both subsets occurs at the
front and rear of the truck. We split the Room scene into two
subsets following the same procedure. In the Playroom scene,
we allow for greater overlap, with the density of images per
subregion of the scene varying between both subsets. We train
independent GSplat maps for each scene-subset pair.
Geometric Evaluation. In Table I, we report the geometric
errors of each registration method across the three scenes.
SIREN-R, our method, achieves the lowest rotation and
translation errors in two of the three scenes (Playroom and
Truck): with about 1.14x to 8.89x lower rotation errors and
about 6x to 46x lower translation errors compared to the
baseline methods. Meanwhile, in the Room scene, SIREN-NR
achieves the lowest translation and scale error, with SIREN-R
achieving the second-best performance on these metrics. In
summary, SIREN achieves the lowest geometric errors (i.e.,
rotation, translation, and scale errors) across all scenes, except
the rotation error in the Room scene.
Photometric Evaluation. Now, we examine the photometric
performance of the GSplat registration methods reported in
Table III in Appendix B. SIREN-R achieves the best photomet-
ric performance in the Playroom scene, with the highest mean
PSNR and SSIM and lowest mean LPIPS scores. Similarly,
in the Room scene, SIREN-NR achieves the best photometric
performance across all metrics, followed by SIREN-R. In the
Truck scene, RANSAC-GR achieves the best mean PSNR and
SSIM scores. Although this finding may appear inconsistent
with the geometric results presented in Table I, the high standard
deviation of each of the scores achieved by RANSAC-GR
(about 2x to 3x larger than that of SIREN) suggests that the
geometric and photometric performance metrics for this scene
might be consistent, indicating that the fused map generated
by RANSAC-GR warrants further examination. We provide
rendered images from the fused map generated by RANSAC-
GR compared to the ground-truth images in Figure 3 to examine
the registration results of RANSAC-GR. From Figure 3, we
note that RANSAC-GR fails to accurately register the left
and right sides of the truck. In fact, the left side of the truck
is missing in the bottom panel associated with RANSAC-
GR in Figure 3. However, this failure mode is not fully
captured by the mean score of the photometric performance
metrics, since the rendered images of the right side of the truck
(shown in the top panel in Figure 3) look quite similar to the
corresponding ground-truth images. In conclusion, RANSAC-
GR does not accurately register the individual GSplat maps,
despite achieving the highest mean PSNR and SSIM scores in
the Truck scene. In Figure 4, we show the rendered images
from the fused GSplat maps generated by the registration
methods from different viewpoints compared to the ground-
truth images. We visualize a pair of images from the Playroom,
Truck, and Room scenes, restricting our visualizations to
PhotoReg, GaussReg, Colored-ICP, and SIREN-R due to space
considerations.
B. Mobile-Robot Mapping
We utilize a quadruped and a drone to map three envi-
ronments, depicted in Figure 5. The quadruped maps the
Kitchen and Workshop environments, while the drone maps
an Apartment scene, with multiple partitioned room-like areas.
The robots create submaps in each environment individually,
containing different regions of the scene. The submaps in the
Kitchen and Workshop scenes have minimal overlap, while the

<!-- page 8 -->
TABLE I: Geometric performance of the registration algorithms on the Mip-NeRF360 dataset (see Section V for a description
of the metrics).
Playroom
Truck
Room
Methods
RE ↓
TE ↓
SE ↓
CT ↓
RE ↓
TE ↓
SE ↓
CT ↓
RE ↓
TE ↓
SE ↓
CT ↓
PhotoReg [52]
6.036
18806
841.3
2177
177.3
2856
444.0
1814
0.161
4983
452.7
1409
GaussReg [6]
0.766
55.50
0.364
15.06
21.10
316.3
16.76
5.174
7.464
628.3
91.97
6.932
RANSAC-GR [10, 15]
4.835
56.22
17.85
0.996
46.72
2642
13.64
2.569
8.139
194.7
152.5
0.517
FGR [54]
2.988
18.83
14.37
0.887
3.778
2231
79.45
3.480
4.869
265.6
219.6
0.511
ICP [37]
2.362
19.11
14.37
2.127
3.672
2232
79.45
3.805
5.154
266.1
219.6
1.579
Colored-ICP [31]
0.194
12.28
14.37
3.951
4.043
2250
79.45
6.392
2.256
232.7
219.6
3.815
SIREN-NR [Ours]
0.348
4.860
0.282
41.16
0.511
8.07
9.581
53.42
0.381
2.648
1.016
40.24
SIREN-R [Ours]
0.170
1.933
0.170
39.73
0.413
6.845
2.548
52.47
0.237
3.289
2.673
39.71
Ground-Truth
RANSAC-GR
Right
Left
Fig. 3: Although RANSAC-GR achieves the highest mean
PSNR and SSIM scores and the lowest LPIPS score in the
Truck scene, RANSAC-GR does not accurately register the
individual GSplat maps. While the right side of the truck in
the RANSAC-GR fused map looks similar to the ground-truth
image (shown in the top panel), the left side of the truck is
missing (shown in the bottom panel). The standard deviation of
the PSNR, SSIM, and LPIPS scores achieved by RANSAC-GR
reflects the actual registration performance of the method.
submaps in the Apartment scene have greater overlap. Since
each submap is trained independently in different reference
frames, fusing the submaps requires registration of the maps.
Here, we examine the performance of GaussReg, PhotoReg,
and two variants of SIREN: SIREN-NR and SIREN-R, in
registering the submaps in each scene to obtain a composite
map of the entire scene.
Geometric Performance. Table II summarizes the geometric
errors of each algorithm, showing that SIREN achieves the
best geometric performance across all scenes, with the top-two-
performing methods being the variants of SIREN. Specifically,
in the Kitchen scene, SIREN-NR achieves the lowest rotation,
translation, and scale errors by a factor of about 160x, 465x, and
488x, respectively, compared to the best-performing baseline.
The performance of SIREN-R closely follows that of SIREN-
NR. Similarly, in the Workshop scene, SIREN-R achieves the
lowest rotation and translation errors by a factor of 415x
and 1287x, respectively, compared to the best-performing
baseline, followed by SIREN-NR, while SIREN-NR achieves
the lowest scale error by a factor of 2962x, followed by SIREN-
R. Lastly, SIREN-R achieves the lowest rotation, translation,
and scale errors in the Apartment scene, followed by SIREN-
NR. GaussReg requires the least computation time across
all scenes, while PhotoReg requires the greatest computation
time. Although compared to GaussReg SIREN requires a
notably greater computation time, SIREN requires much lower
computation times compared to PhotoReg.
Photometric Performance. Further, we examine the pho-
tometric quality of the fused map generated by the GSplat
registration methods across the three scenes. In line with the
geometric results, SIREN outperforms all the baseline methods,
as reported in Table IV. While SIREN-R achieves the best
photometric scores (i.e., the highest PSNR and SSIM scores
and lowest LPIPS scores) in the Workshop scene, SIREN-NR
attains the best-performing PSNR, SSIM, and LPIPS scores
in the Kitchen scene, followed by SIREN-R. In the Apartment
scene, SIREN-R achieves the best PSNR score and LPIPS
(tied with SIREN-NR), while SIREN-NR also achieves the best
SSIM score. GaussReg outperforms PhotoReg in all scenes.
In addition to the results in Table IV, we provide rendered
images from each of the fused map in Figure 6 for qualitative
evaluation of the performance of each method.
C. Tabletop Mapping with Multiple Manipulators
We demonstrate the effectiveness of SIREN in tabletop
robotics tasks with fixed-base manipulators, which often
require the robots to map the scene prior to the task, e.g.,
in manipulation [41, 43]. In Figure 7, we provide an example
with two Franka robots, each with a wrist camera. Due to
the limited workspace of each robot, visualized in Figure 7,
mapping often requires the assistance of a human-operator [43]
or ad-hoc solutions such as hardware improvisation, e.g., using
selfie sticks [41]. By enabling the fusion of GSplat maps trained
individually by each robot, SIREN effectively eliminates these
limitations. In other words, with SIREN, each robot can train
a submap within its reachable workspace and still recover
the global map via registration with SIREN. In Figure 8, we
show the submaps trained by each robot. As expected, each
robot has a high-fidelity submap within the confines of its
reachable workspace, evident in the first-two images in the

<!-- page 9 -->
PhotoReg
GaussReg
SIREN-R
Colored-ICP
Playroom
Truck
Room
Ground-Truth
Fig. 4: Rendered images from the fused GSplat maps of the Playroom, Truck, and Room scenes. SIREN generates high-fidelity
fused GSplat maps, evidenced by the precise geometric detail in the images, visible in the regions indicated by the green
squares. Inaccurate registration of GSplat maps generally result in artifacts in the rendered images.
left robot’s map and the last-two images in the right’s robot
map in Figure 8. In areas outside of its reachable workspace,
the robot’s map fails to represent the real world accurately,
visible in the last-two images in the left robot’s map and the
first-two images in the right’s robot map. With SIREN, each
robot obtains a higher-fidelity map over a much broader region
of the environment. However, floaters present in the submaps
can degrade the quality of the fused map in certain regions.
To address this challenge, we finetune the fused map for about
70.98 secs using images generated entirely from the GSplat
maps, i.e., we do not require any real-world data. We provide
rendered images from the finetuned fused map in Figure 8,
showing near-perfect reconstruction of the global scene. We
explore the finetuning procedure in Appendix B-D.
VI. CONCLUSION
We present SIREN, a semantics-grounded registration al-
gorithm for multi-robot GSplat maps that neither requires
access to camera poses or images nor initialization of inter-
map relative transforms. SIREN harnesses the robustness of
semantics to: (a) identify candidate matches between Gaussians
across the input maps, (b) compute a coarse transformation
aligning both maps from a Gaussian-to-Gaussian registration
problem posed as an optimization program, and (c) refine
the coarse registration result for high-accuracy fusion of local
submaps into a high-fidelity global map. We demonstrate the
versatility of SIREN across maps constructed by robots of
different embodiments, including a quadruped, drone, and
manipulator, highlighting the superior performance of SIREN
compared to GSplat registration algorithms and classical point-
cloud registration methods.

<!-- page 10 -->
Kitchen
Workshop
Apartment
Fig. 5: Stillshots of a quadruped mapping different areas of a kitchen and workshop and a drone mapping an apartment-like
scene. Each robot trains independent GSplat submaps of the areas it mapped. The submaps of each scene are registered to
obtain a composite map covering the entirety of the scene.
TABLE II: Geometric performance of GSplat registration algorithms in mobile-robot mapping.
Kitchen
Workshop
Apartment
Methods
RE ↓
TE ↓
SE ↓
CT ↓
RE ↓
TE ↓
SE ↓
CT ↓
RE ↓
TE ↓
SE ↓
CT ↓
PhotoReg [52]
40.49
2350
413.37
1042
140.5
10052
4310
934.2
24.09
4433
260.2
801.0
GaussReg [6]
40.89
1477
171.8
11.33
55.66
9531
4305
5.491
3.114
102.6
13.59
5.4983
SIREN-NR [Ours]
0.253
3.173
0.352
59.22
0.518
11.77
1.453
67.98
0.148
1.758
0.605
35.91
SIREN-R [Ours]
0.430
4.795
3.849
56.14
0.134
7.400
10.88
55.16
0.119
1.495
0.102
34.22
VII. LIMITATIONS AND FUTURE WORK
Our registration algorithm relies on semantics for robust,
initialization-free registration, and thus requires that the input
maps have embedded semantic codes. A GSplat map may lack
semantic information if the map was not trained with semantics
or if the scene lacks any semantically-relevant features, which
would be a tail event in practical situations. We can post-train
GSplats to embed semantics in 3D into the map or leverage
2D vision foundation models to directly extract semantic
information from RGB images rendered from the GSplat by
back-projecting 2D pixels into the 3D world. Radiance fields
are prone to generate floaters in areas of the scene with little
to no supervision, which can degrade the fidelity of the map.
The resulting floaters are retained in the fused map, which
could ultimately reduce the accuracy of the map. However,
by finetuning the fused map with synthetic data, i.e., images
rendered from the map as opposed to real-world images, floaters
in the map can be removed for high-fidelity mapping.
ACKNOWLEDGMENTS
This work was supported in part by NSF grant 2342246, NSF
CAREER Award 2044149, Office of Naval Research N00014-
23-1-2148, and Princeton SEAS Innovation Award from The
Addy Fund for Excellence in Engineering. Toyota Research
Institute provided funds to support this work. Jiankai Sun
is partially supported by Stanford Interdisciplinary Graduate
Fellowship.

<!-- page 11 -->
PhotoReg
GaussReg
SIREN-R
Ground-Truth
Kitchen
Workshop
Apartment
Fig. 6: Rendered images from the fused GSplat maps of the Kitchen, Workshop, and Apartment scenes mapped by a quadruped
and drone. Unlike other competing methods, SIREN generates fused GSplat maps of high visual fidelity, e.g., in the regions
indicated by the green squares.
Tabletop Task
𝒲!
𝒲"
Limited Workspace
Multi-Robot Mapping
Fig. 7: Tabletop robotics tasks, e.g., manipulation, generally require robots to map the scene prior to completing the task.
However, the limited workspace of each robot often demands assistance from a human-operator or improvised hardware, e.g.,
selfie sticks. SIREN eliminates these challenges, via registration of the local maps trained by each robot to construct a global
map consistent with the real-world.

<!-- page 12 -->
Individual Submaps
Left
Right
Fused Map
Pre-Finetuned
Post-Finetuned
Fig. 8: Rendered images of the local maps of a tabletop scene trained by two manipulators. The maps provide high-fidelity
reconstructions within the workspace of each robot, but fail to represent the real-world in regions outside the workspace. SIREN
fuses the local maps to generate a high-fidelity global map consistent with the entirety of the scene, especially after finetuning
on data rendered directly from the GSplat to remove floaters, without any interaction with the real-world, as indicated by the
green squares.

<!-- page 13 -->
REFERENCES
[1] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas
Pajdla, and Josef Sivic. Netvlad: Cnn architecture for
weakly supervised place recognition. In Proceedings of
the IEEE conference on computer vision and pattern
recognition, pages 5297–5307, 2016.
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pages 5470–5479, 2022.
[3] Paul J Besl and Neil D McKay. Method for registration
of 3-d shapes. In Sensor fusion IV: control paradigms
and data structures, volume 1611, pages 586–606. Spie,
1992.
[4] Sofien Bouaziz, Andrea Tagliasacchi, and Mark Pauly.
Sparse iterative closest point. In Computer graphics forum,
volume 32, pages 113–123. Wiley Online Library, 2013.
[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin.
Emerging properties in self-supervised vision transformers.
In Proceedings of the IEEE/CVF international conference
on computer vision, pages 9650–9660, 2021.
[6] Jiahao Chang, Yinglin Xu, Yihao Li, Yuantao Chen,
Wensen Feng, and Xiaoguang Han.
Gaussreg: Fast
3d registration with gaussian splatting.
In European
Conference on Computer Vision, pages 407–423. Springer,
2025.
[7] Timothy Chen, Ola Shorinwa, Joseph Bruno, Javier Yu,
Weijia Zeng, Keiko Nagami, Philip Dames, and Mac
Schwager. Splat-nav: Safe real-time robot navigation in
gaussian splatting maps. arXiv preprint arXiv:2403.02751,
2024.
[8] Yang Chen and G´erard Medioni. Object modelling by
registration of multiple range images. Image and vision
computing, 10(3):145–155, 1992.
[9] Yu Chen and Gim Hee Lee. Dreg-nerf: Deep registra-
tion for neural radiance fields. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 22703–22713, 2023.
[10] Martin A Fischler and Robert C Bolles.
Random
sample consensus: a paradigm for model fitting with
applications to image analysis and automated cartography.
Communications of the ACM, 24(6):381–395, 1981.
[11] Kexue Fu, Shaolei Liu, Xiaoyuan Luo, and Manning
Wang. Robust point cloud registration framework based
on deep graph matching. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 8893–8902, 2021.
[12] Lily Goli, Daniel Rebain, Sara Sabour, Animesh Garg,
and Andrea Tagliasacchi. nerf2nerf: Pairwise registration
of neural radiance fields. In 2023 IEEE International
Conference on Robotics and Automation (ICRA), pages
9354–9361. IEEE, 2023.
[13] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui.
Open-vocabulary object detection via vision and language
knowledge distillation. arXiv preprint arXiv:2104.13921,
2021.
[14] Antoine Gu´edon and Vincent Lepetit. Sugar: Surface-
aligned gaussian splatting for efficient 3d mesh recon-
struction and high-quality mesh rendering. In Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 5354–5363, 2024.
[15] Dirk Holz, Alexandru E Ichim, Federico Tombari, Radu B
Rusu, and Sven Behnke. Registration with the point cloud
library: A modular framework for aligning in 3-d. IEEE
Robotics & Automation Magazine, 22(4):110–124, 2015.
[16] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger,
and Shenghua Gao. 2d gaussian splatting for geometrically
accurate radiance fields.
In ACM SIGGRAPH 2024
Conference Papers, pages 1–11, 2024.
[17] Mazeyu Ji, Ri-Zhao Qiu, Xueyan Zou, and Xiaolong
Wang. Graspsplats: Efficient manipulation with 3d feature
splatting. arXiv preprint arXiv:2409.02084, 2024.
[18] Han Jiang, Ruoxuan Li, Haosen Sun, Yu-Wing Tai, and
Chi-Keung Tang. Registering neural radiance fields as 3d
density images. arXiv preprint arXiv:2305.12843, 2023.
[19] Ebrahim Karami, Siva Prasad, and Mohamed Shehata.
Image matching using sift, surf, brief and orb: perfor-
mance comparison for distorted images. arXiv preprint
arXiv:1710.02726, 2017.
[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-
time radiance field rendering. ACM Trans. Graph., 42(4):
139–1, 2023.
[21] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik. Lerf: Language embed-
ded radiance fields. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages
19729–19739, 2023.
[22] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitz-
mann. Decomposing nerf for editing via feature field
distillation. Advances in Neural Information Processing
Systems, 35:23311–23330, 2022.
[23] Byeonghyeon Lee, Howoong Lee, Xiangyu Sun, Usman
Ali, and Eunbyung Park. Deblurring 3d gaussian splatting.
In European Conference on Computer Vision, pages 127–
143. Springer, 2025.
[24] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu,
Jiwen Lu, and Yansong Tang. Manigaussian: Dynamic
gaussian splatting for multi-task robotic manipulation. In
European Conference on Computer Vision, pages 349–366.
Springer, 2025.
[25] Timo L¨uddecke and Alexander Ecker. Image segmentation
using text and image prompts. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pages 7086–7096, 2022.
[26] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei,
Nan Duan, and Tianrui Li. Clip4clip: An empirical study
of clip for end to end video clip retrieval and captioning.
Neurocomputing, 508:293–304, 2022.

<!-- page 14 -->
[27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view
synthesis. Communications of the ACM, 65(1):99–106,
2021.
[28] Matthias Minderer, Alexey Gritsenko, Austin Stone,
Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy,
Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani,
Zhuoran Shen, et al.
Simple open-vocabulary object
detection. In European Conference on Computer Vision,
pages 728–755. Springer, 2022.
[29] Ron Mokady, Amir Hertz, and Amit H Bermano. Clip-
cap: Clip prefix for image captioning. arXiv preprint
arXiv:2111.09734, 2021.
[30] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby,
et al. Dinov2: Learning robust visual features without
supervision. arXiv preprint arXiv:2304.07193, 2023.
[31] Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Colored
point cloud registration revisited. In Proceedings of the
IEEE international conference on computer vision, pages
143–152, 2017.
[32] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang,
and Hanspeter Pfister. Langsplat: 3d language gaussian
splatting. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 20051–
20060, 2024.
[33] Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo,
Yuxing Peng, Slobodan Ilic, Dewen Hu, and Kai Xu.
Geotransformer: Fast and robust point cloud registration
with geometric transformer. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 45(8):9806–9821,
2023.
[34] Ri-Zhao Qiu, Yafei Hu, Ge Yang, Yuchen Song, Yang
Fu, Jianglong Ye, Jiteng Mu, Ruihan Yang, Nikolay
Atanasov, Sebastian Scherer, et al. Learning generalizable
feature fields for mobile manipulation. arXiv preprint
arXiv:2403.07563, 2024.
[35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al.
Learning transferable visual models from natural language
supervision.
In International conference on machine
learning, pages 8748–8763. PMLR, 2021.
[36] Adam Rashid, Satvik Sharma, Chung Min Kim, Justin
Kerr, Lawrence Yunliang Chen, Angjoo Kanazawa, and
Ken Goldberg.
Language embedded radiance fields
for zero-shot task-oriented grasping.
In 7th Annual
Conference on Robot Learning, 2023.
[37] Szymon Rusinkiewicz and Marc Levoy. Efficient variants
of the icp algorithm. In Proceedings third international
conference on 3-D digital imaging and modeling, pages
145–152. IEEE, 2001.
[38] Radu Bogdan Rusu, Nico Blodow, and Michael Beetz.
Fast point feature histograms (fpfh) for 3d registration.
In 2009 IEEE international conference on robotics and
automation, pages 3212–3217. IEEE, 2009.
[39] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,
and Andrew Rabinovich. Superglue: Learning feature
matching with graph neural networks. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 4938–4947, 2020.
[40] Johannes
L
Schonberger
and
Jan-Michael
Frahm.
Structure-from-motion revisited.
In Proceedings of
the IEEE conference on computer vision and pattern
recognition, pages 4104–4113, 2016.
[41] William Shen, Ge Yang, Alan Yu, Jansen Wong,
Leslie Pack Kaelbling, and Phillip Isola. Distilled feature
fields enable few-shot language-guided manipulation.
arXiv preprint arXiv:2308.07931, 2023.
[42] Ola Shorinwa, Jiankai Sun, and Mac Schwager. Fast-
splat: Fast, ambiguity-free semantics transfer in gaussian
splatting. arXiv preprint arXiv:2411.13753, 2024.
[43] Ola Shorinwa, Johnathan Tucker, Aliyah Smith, Aiden
Swann, Timothy Chen, Roya Firoozi, Monroe David
Kennedy, and Mac Schwager.
Splat-mover: Multi-
stage, open-vocabulary robotic manipulation via editable
gaussian splatting. In 8th Annual Conference on Robot
Learning, 2024.
[44] Gary KL Tam, Zhi-Quan Cheng, Yu-Kun Lai, Frank C
Langbein, Yonghuai Liu, David Marshall, Ralph R Martin,
Xian-Fang Sun, and Paul L Rosin. Registration of 3d point
clouds and meshes: A survey from rigid to nonrigid. IEEE
transactions on visualization and computer graphics, 19
(7):1199–1217, 2012.
[45] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li,
Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake
Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A
modular framework for neural radiance field development.
In ACM SIGGRAPH 2023 Conference Proceedings, pages
1–12, 2023.
[46] Shinji Umeyama. Least-squares estimation of transfor-
mation parameters between two point patterns. IEEE
Transactions on Pattern Analysis & Machine Intelligence,
13(04):376–380, 1991.
[47] Can Wang, Menglei Chai, Mingming He, Dongdong
Chen, and Jing Liao. Clip-nerf: Text-and-image driven
manipulation of neural radiance fields. In Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 3835–3844, 2022.
[48] Yue Wang and Justin M Solomon. Deep closest point:
Learning representations for point cloud registration. In
Proceedings of the IEEE/CVF international conference
on computer vision, pages 3523–3532, 2019.
[49] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yan-
dong Guo, Mingming Gong, and Tongliang Liu. Cris:
Clip-driven referring image segmentation. In Proceedings
of the IEEE/CVF conference on computer vision and
pattern recognition, pages 11686–11695, 2022.
[50] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo
Kanazawa. pixelnerf: Neural radiance fields from one or

<!-- page 15 -->
few images. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 4578–
4587, 2021.
[51] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler,
and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian
splatting. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 19447–
19456, 2024.
[52] Ziwen Yuan, Tianyi Zhang, Matthew Johnson-Roberson,
and Weiming Zhi.
Photoreg: Photometrically regis-
tering 3d gaussian splatting models.
arXiv preprint
arXiv:2410.05044, 2024.
[53] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen
Koltun. Nerf++: Analyzing and improving neural radiance
fields. arXiv preprint arXiv:2010.07492, 2020.
[54] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Fast
global registration.
In Computer Vision–ECCV 2016:
14th European Conference, Amsterdam, The Netherlands,
October 11-14, 2016, Proceedings, Part II 14, pages 766–
782. Springer, 2016.
[55] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen
Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya
You, Zhangyang Wang, and Achuta Kadambi. Feature
3dgs: Supercharging 3d gaussian splatting to enable
distilled feature fields. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 21676–21685, 2024.
[56] Liyuan Zhu, Yue Li, Erik Sandstr¨om, Shengyu Huang,
Konrad Schindler, and Iro Armeni.
Loopsplat: Loop
closure by registering 3d gaussian splats. arXiv preprint
arXiv:2408.10154, 2024.

<!-- page 16 -->
APPENDIX A
COARSE GAUSSIAN-TO-GAUSSIAN REGISTRATION
We discuss the derivation of the closed-form solution to (3).
Let the objective function of (3) be denoted by J. From the
first-order optimality conditions:
∇tJ =
X
(i,j)∈E
(wijscRpi + t −qj) = 0,
(9)
yielding the optimal translation:
t⋆
c = ˜µQ −s⋆
cR⋆˜µP,
(10)
where ˜µP and ˜µQ denote the weighted average of the means
of the Gaussians in P and Q, with weights wij for Gaussian
i in P and Gaussian j in Q. By substituting the optimal value
of t (10) in (3), we obtain the following optimization problem
over sc and R:
minimize
sc∈R++,R∈SO(3)
1
2
(scR ˇP −ˇQ)W
1
2

2
F
+ 1
2
X
(i,j)∈E
wij
scR ˇHpi −ˇHqj
2
F ,
(11)
where ˇP ∈R3×N and ˇQ ∈R3×N represent the zero-centered
Gaussians in P and Q, respectively, with the ith column
of ˇP given by ˇPi = pi −˜µP and similarly for the jth col-
umn of ˇQ, and ˇHpi = HpiΛpi and ˇHqj = HqjΛqj. Lastly,
W ∈RN×N denotes the diagonal weight matrix, Wkk = wk,
with wk = wij, ∀k = (i, j) ∈E. Now, we can reformulate (11)
as a trace-minimization problem, by leveraging the relation:
∥A∥2
F = trace(A⊤A) for any real-valued matrix A ∈Rm×n.
Reformulating the problem as a trace-minimization problem
enables us to decompose the norm-minimization problem (11)
into a nested pair of subproblems: an outer subproblem over
sc and an inner subproblem over R. We can simplify the inner
subproblem into the form:
minimize
R∈SO(3)
−trace

R⊤

ˇQW ˇP ⊤+
X
(i,j)∈E
wij ˇHqj ˇH⊤
pi



,
(12)
which affords a closed-form optimal solution, with
R⋆
c = UcΘcV ⊤
c ,
(13)
where UcΣcV ⊤
c
= ˇQW ˇP ⊤+ P
(i,j)∈E wij ˇHqj ˇH⊤
pi, computed
via
the
singular
value
decomposition
(SVD),
and
Θc = diag(1, 1, det(UcV ⊤
c )). Using the first-order optimality
condition, we can compute the optimal scale after computing
the optimal rotation from the outer subproblem given by:
minimize
sc∈R++
1
2s2
ctrace

W ˇP ⊤ˇP +
X
(i,j)∈E
wij ˇH⊤
pi ˇHpi


−sctrace

R⊤

ˇQW ˇP ⊤+
X
(i,j)∈E
wij ˇHqj ˇH⊤
pi



,
(14)
yielding the optimal solution:
s⋆
c =
trace(ΘcΣ)
trace

W ˇP ⊤ˇP + P
(i,j)∈E wij ˇH⊤
pi ˇHpi
.
(15)
For brevity, we omit further analysis of the optimality of the
solution and refer interested readers to [46] for the proof of a
related problem, which applies to the problem considered in
this work.
APPENDIX B
EXPERIMENTS
We report the photometric performance of each registration
method with the Mip-NeRF360 dataset and the data collected
by the robots in our experiments and provide further discussion
of the experimental results. In addition, we present ablations,
examining the different components of SIREN. Lastly, we
explore finetuning the resulting composite maps for higher
visual fidelity.
A. Mip-NeRF360 Dataset
Geometric Evaluation. In the Room scene, PhotoReg achieves
the lowest rotation error by a factor of about 1.47x but also
achieves the largest translation and scale errors. Based on the
results across all scenes, SIREN almost always consistently
outperforms competing methods. From Table I, RANSAC-
GR and FGR achieve the fastest computation times; however,
RANSAC-GR and FGR do not generally achieve consistently
low geometric errors. Although SIREN is slower than the
classical point-cloud registration algorithms and GaussReg,
SIREN generally outperforms these methods in accuracy by
significant margins. Moreover, about 40% to 50% of the
total computation time of SIREN is spent on the semantics
extraction procedure. Hence, the total computation time can be
significantly improved by utilizing faster semantics distillation
methods, e.g., [42].
Photometric Evaluation. From Figure 4, in the Playroom
scene, PhotoReg fails to sufficiently register the individual
maps to obtain photorealistic renderings. In contrast, GaussReg,
Colored-ICP, and SIREN-R generate high-fidelity renderings.
However, the fused maps generated by GaussReg and Colored-
ICP are not accurately aligned, compared to that of SIREN-
R, as evidenced in insets in the images. The fused map in
GaussReg and Colored-ICP contain duplicate objects due to in-
accurate registration of the individual maps. In contrast, SIREN-
R provides greater accuracy. Likewise, SIREN-R achieves the
highest-fidelity rendering in the Truck scene with consistent
geometry, whereas Colored-ICP fails to register the left and
right sides of the truck. Although GaussReg fuses both sides of
the truck, GaussReg fails to compute a high-accuracy transform,
resulting in the artifacts visible in Figure 4. Although PhotoReg
registers the cargo bed of the truck in both maps, PhotoReg
fails to align the truck accurately in terms of the rotation
transform, with the front end of the truck in one map registered
to the rear end of the truck in the other map. Finally, in the
Room scene, whereas SIREN-R generates high-fidelity rendered

<!-- page 17 -->
images, other methods fail to accurately register the individual
maps. In particular, Colored-ICP generates a fused map with
duplicate objects, e.g., the piano and the table, indicated by the
green squares, while PhotoReg and GaussReg generate fused
maps with notable artifacts.
B. Mobile-Robot Mapping
Photometric Evaluation. From Figure 6, as highlighted
by the green squares, SIREN-R generates composite maps
that are consistent with the ground-truth, unlike GaussReg
and PhotoReg. The fused maps generated by GaussReg and
PhotoReg contain conspicuous artifacts due to inaccurate
registration of the individual maps created by the robots,
especially in the Kitchen scene. PhotoReg fails to sufficiently
register the individual maps, resulting in blurry renderings,
with few recognizable features, e.g., in the Workshop scene.
In the Apartment scene, the rendered images from GaussReg
contain duplicate objects, unlike those of SIREN-R, which
have accurate geometric detail.
C. Ablations
We examine the constituent registration steps in SIREN,
namely: the coarse Gaussian-to-Gaussian and fine photometric
registration procedures, assessing the accuracy of the registra-
tion result generated by each procedure. We denote the variant
of SIREN with coarse registration performed without RANSAC
and fine registration by SIREN-CNR. Likewise, we denote the
variant of SIREN with coarse registration performed using
RANSAC but without fine registration by SIREN-CR. We
compute the geometric and photometric performance metrics
for each of these variants and report the results in Table V
and Table VI, respectively. We also report the performance of
SIREN-NR and SIREN-R from Table I and Table III for easy
reference. From Table V, we note that the fine registration
step in SIREN notably improves the rotation error to sub-
degree errors, achieving about 2x smaller translation errors and
in some cases, 100x smaller translation errors. Likewise, the
fine registration step generally results in much smaller scale
errors, although not necessarily in all cases, as reflected in
the Truck scene. Similarly, the variants of SIREN with fine
registration (i.e., SIREN-NR and SIREN-R) achieve notably
higher photometric performance, especially in the Playroom
and Room scenes, reported in Table VI. In general, the coarse
registration step brings corresponding objects in both GSplat
maps into close proximity in the fused map. However, the
resulting fused map lacks precise geometric detail, degrading
its visual fidelity. After the coarse registration step, the fine
registration procedure refines the transformation parameters for
precise alignment of the individual maps, ultimately generating
a photorealistic fused map.
Although SIREN-CR does not always outperform RANSAC-
GR in Table I, we observed empirically that the performance
of RANSAC-GR has a high variance, posing a challenge for
the fine registration step, which requires a sufficient number
of corresponding features between rendered frames across the
individual maps to compute a solution. Moreover, ICP and
its variants tend to converge to a local optimum, close to the
solution used for initialization. As a result, these methods
generally fail to provide a sufficiently good initialization for
the fine registration procedure. The coarse registration step
in SIREN relies significantly on the semantics extracted from
the map to overcome these limitations, leveraging the inherent
semantics to register corresponding objects at a sufficient level
of accuracy for fine registration.
D. Finetuning
SIREN does not pre-process the local GSplat maps before
registration of the maps, resulting in the retention of floaters
in the fused map whenever floaters exist in the local maps.
Here, we examine finetuning the fused map with rendered
images from the local maps to remove visual artifacts, without
requiring access to the data used in the training the local GSplat
maps, i.e., we do not require access to the real-world camera
images and poses. To finetune the fused map without access
to the original dataset, we select camera poses expressed in
the local frames of the local GSplat maps (e.g., randomly
or via an informed approach) and render images from these
maps at these camera poses. Subsequently, we transform the
set of camera poses from their associated local frames to the
frame of the fused map using the transformation parameters
computed by SIREN. We construct a finetuning dataset from
the set of images and associated camera poses, which we use
in finetuning the fused map.
In Table VII, we provide the photometric scores of the fused
GSplat map from SIREN-R before and after finetuning and the
ground-truth GSplat map. We train the ground-truth GSplat map
using the combined training datasets used in training the local
GSplat maps (i.e., the real-world camera images and poses,
not the set of rendered images generated from the local GSplat
maps), representing the ideal composite GSplat model. The
computation time in Table VII represents the total training time
for the ground-truth map and the total time used in finetuning
the fused map. Table VII indicates that finetuning the fused map
improves the PSNR, SSIM, and LPIPS scores compared to that
of the pre-finetuned map. Specifically, in less than 90 seconds,
finetuning reduces the gap between the photometric scores
of the ground-truth map and the photometric scores of the
fused map by about 20% to 40%. The relative improvements
provided by finetuning the fused map depend on the finetuning
data used, an area for future research. We provide rendered
images from the fused GSplat map computed by SIREN-R,
before and after finetuning, and the corresponding images in
the ground-truth fused map in Figure 9. Across all three scenes,
finetuning the fused map removes floaters and other artifacts,
e.g., in the regions indicated by the green squares, ultimately
resulting in higher PSNR and SSIM scores, as reported in
Table VII.

<!-- page 18 -->
TABLE III: Photometric performance of registration algorithms for GSplat maps from the Mip-NeRF360 dataset.
Playroom
Truck
Room
Methods
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PhotoReg [52]
11.5 ± 2.3
0.68 ± 0.11
0.67 ± 0.12
10.6 ± 0.9
0.39 ± 0.07
0.72 ± 0.08
10.2 ± 1.2
0.46 ± 0.07
0.78 ± 0.04
GaussReg [6]
23.7 ± 3.3
0.86 ± 0.06
0.22 ± 0.08
13.4 ± 1.9
0.54 ± 0.12
0.53 ± 0.13
13.4 ± 3.6
0.61 ± 0.11
0.55 ± 0.15
RANSAC-GR [10, 15]
17.9 ± 3.2
0.77 ± 0.09
0.37 ± 0.09
18.9 ± 7.0
0.66 ± 0.24
0.32 ± 0.22
14.2 ± 2.4
0.66 ± 0.09
0.46 ± 0.11
FGR [54]
22.2 ± 3.2
0.85 ± 0.06
0.24 ± 0.08
13.0 ± 2.6
0.57 ± 0.19
0.43 ± 0.20
17.2 ± 2.3
0.77 ± 0.08
0.33 ± 0.11
ICP [37]
22.7 ± 3.3
0.85 ± 0.06
0.24 ± 0.08
12.9 ± 2.6
0.57 ± 0.19
0.43 ± 0.20
16.8 ± 2.2
0.76 ± 0.09
0.35 ± 0.11
Colored-ICP [31]
26.2 ± 3.1
0.89 ± 0.04
0.17 ± 0.06
13.4 ± 2.4
0.58 ± 0.19
0.40 ± 0.18
15.4 ± 2.1
0.71 ± 0.10
0.41 ± 0.13
SIREN-NR [Ours]
26.3 ± 3.1
0.87 ± 0.05
0.17 ± 0.06
15.4 ± 1.7
0.52 ± 0.12
0.35 ± 0.05
24.8 ± 3.3
0.83 ± 0.04
0.22 ± 0.06
SIREN-R [Ours]
28.3 ± 2.9
0.90 ± 0.04
0.15 ± 0.06
16.4 ± 2.4
0.57 ± 0.13
0.31 ± 0.07
24.1 ± 3.1
0.82 ± 0.05
0.23 ± 0.06
TABLE IV: Photometric performance of GSplat registration algorithms for mobile-robot mapping.
Kitchen
Workshop
Apartment
Methods
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PhotoReg [52]
0.75 ± 0.05
11.6 ± 1.3
0.53 ± 0.06
0.78 ± 0.08
11.3 ± 1.3
0.48 ± 0.05
13.4 ± 1.5
0.61 ± 0.05
0.75 ± 0.04
GaussReg [6]
14.1 ± 1.9
0.62 ± 0.06
0.60 ± 0.11
17.0 ± 2.8
0.61 ± 0.05
0.53 ± 0.11
14.3 ± 1.6
0.62 ± 0.05
0.62 ± 0.04
SIREN-NR [Ours]
19.3 ± 3.2
0.63 ± 0.05
0.40 ± 0.09
19.9 ± 2.5
0.60 ± 0.04
0.40 ± 0.08
15.3 ± 1.5
0.64 ± 0.04
0.55 ± 0.03
SIREN-R [Ours]
18.8 ± 2.8
0.62 ± 0.05
0.41 ± 0.08
20.3 ± 2.7
0.62 ± 0.04
0.38 ± 0.09
15.3 ± 1.4
0.63 ± 0.04
0.55 ± 0.03
TABLE V: Geometric Performance: Ablation of the Coarse Gaussian-to-Gaussian and Fine Photometric Registration in SIREN.
Playroom
Truck
Room
Methods
RE ↓
TE ↓
SE ↓
CT ↓
RE ↓
TE ↓
SE ↓
CT ↓
RE ↓
TE ↓
SE ↓
CT ↓
SIREN-CNR
22.72
454.2
482.1
20.20
49.98
355.4
55.06
24.17
20.50
474.0
371.9
17.27
SIREN-CR
21.15
324.2
51.94
20.47
0.804
7.691
7.744
26.32
24.07
381.8
155.1
17.58
SIREN-NR
0.348
4.860
0.282
41.16
0.511
8.07
9.581
53.42
0.381
2.648
1.016
40.24
SIREN-R
0.170
1.933
0.170
39.73
0.413
6.845
2.548
52.47
0.237
3.289
2.673
39.71
TABLE VI: Photometric Performance: Ablation of the Coarse Gaussian-to-Gaussian and Fine Photometric Registration in
SIREN.
Playroom
Truck
Room
Methods
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
SIREN-CNR
12.0 ± 3.2
0.67 ± 0.11
0.65 ± 0.17
11.4 ± 2.4
0.52 ± 0.11
0.63 ± 0.22
6.9 ± 5.9
0.72 ± 0.11
0.39 ± 0.15
SIREN-CR
13.7 ± 3.4
0.65 ± 0.14
0.58 ± 0.19
16.2 ± 2.2
0.56 ± 0.14
0.31 ± 0.07
11.7 ± 1.9
0.48 ± 0.12
0.64 ± 0.10
SIREN-NR
26.3 ± 3.1
0.87 ± 0.05
0.17 ± 0.06
15.4 ± 1.7
0.52 ± 0.12
0.35 ± 0.05
24.8 ± 3.3
0.83 ± 0.04
0.22 ± 0.06
SIREN-R
28.3 ± 2.9
0.90 ± 0.04
0.15 ± 0.06
16.4 ± 2.4
0.57 ± 0.13
0.31 ± 0.07
24.1 ± 3.1
0.82 ± 0.05
0.23 ± 0.06
TABLE VII: Photometric performance after finetuning SIREN-R.
Playroom
Truck
Room
Methods
PSNR ↑
SSIM ↑
LPIPS ↓
CT ↓
PSNR ↑
SSIM ↑
LPIPS ↓
CT ↓
PSNR ↑
SSIM ↑
LPIPS ↓
CT ↓
Ground-Truth
36.3 ± 3.5
0.96 ± 0.03
0.09 ± 0.05
721.1
26.4 ± 1.4
0.89 ± 0.02
0.10 ± 0.01
601.7
34.1 ± 1.7
0.94 ± 0.02
0.12 ± 0.04
840.1
Pre-Finetuning
29.1 ± 3.3
0.91 ± 0.04
0.15 ± 0.06
N/A
16.8 ± 2.5
0.61 ± 0.10
0.30 ± 0.07
N/A
22.5 ± 2.5
0.79 ± 0.05
0.26 ± 0.06
N/A
Post-Finetuning
30.8 ± 2.6
0.92 ± 0.04
0.14 ± 0.06
72.69
21.1 ± 1.8
0.69 ± 0.1
0.23 ± 0.04
86.26
26.0 ± 3.6
0.83 ± 0.09
0.22 ± 0.08
79.78

<!-- page 19 -->
Ground-Truth
Pre-Finetuning
Post-Finetuning
Playroom
Truck
Room
Fig. 9: Rendered images from the fused GSplat maps generated by SIREN-R before and after finetuning, in the Playroom,
Truck, and Room scenes. Finetuning improves the visual fidelity of the fused map, removing floaters and other artifacts.
