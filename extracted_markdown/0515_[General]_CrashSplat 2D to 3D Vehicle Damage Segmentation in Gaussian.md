<!-- page 1 -->
CrashSplat: 2D to 3D Vehicle Damage
Segmentation in Gaussian Splatting
Dragos¸-Andrei Chileban, Andrei-S¸tefan Bulzan, Cosmin Cernˇazanu-Glˇavan
Politehnica University of Timis¸oara
dragos-andrei.chileban@student.upt.ro, stefan.bulzan@student.upt.ro, cosmin.cernazanu@cs.upt.ro
Abstract—Automatic car damage detection has been a topic of
significant interest for the auto insurance industry as it promises
faster, accurate, and cost-effective damage assessments. However,
few works have gone beyond 2D image analysis to leverage 3D re-
construction methods, which have the potential to provide a more
comprehensive and geometrically accurate representation of the
damage. Moreover, recent methods employing 3D representations
for novel view synthesis, particularly 3D Gaussian Splatting (3D-
GS), have demonstrated the ability to generate accurate and
coherent 3D reconstructions from a limited number of views.
In this work we introduce an automatic car damage detection
pipeline that performs 3D damage segmentation by up-lifting 2D
masks. Additionally, we propose a simple yet effective learning-
free approach for single-view 3D-GS segmentation. Specifically,
Gaussians are projected onto the image plane using camera
parameters obtained via Structure from Motion (SfM). They are
then filtered through an algorithm that utilizes Z-buffering along
with a normal distribution model of depth and opacities.
Through experiments we found that this method is particularly
effective for challenging scenarios like car damage detection,
where target objects (e.g., scratches, small dents) may only be
clearly visible in a single view, making multi-view consistency
approaches impractical or impossible. The code is publicly
available at: https://github.com/DragosChileban/CrashSplat.
Index Terms—Vehicle damage, 3D gaussian splatting, 3D
segmentation
I. INTRODUCTION
Vehicle damage detection involves using image analysis to
automatically detect damage to car body components. Recent
advances in deep neural networks have led to solutions for this
task based on supervised learning, which require large-scale
labeled datasets. Some studies proposed manually-annotated
datasets [1], [2] while others rely on synthetically generated
samples [3]. Most of the existing methods experiment with
different detection and segmentation networks in order to
compare training results on publicly available damage datasets.
However, by focusing solely on 2D image analysis, they
hamper the vehicle visualization and inspection process, hence
restricting the overall understanding of the damage produced.
In order to overcome this limitation, we propose a solution
that segments damaged parts in 2D and projects the masks
onto a 3D reconstruction of the vehicle.
For generating the reconstruction we use 3D Gaussian
Splatting (3D-GS) [4], a novel-view synthesis method that
allows us to build a 3D car model from a limited number
of images in just a few minutes. Consequently, our goal is
to find the 3D Gaussians that correspond to the segmented
Fig. 1:
Pipeline diagram. (a) Instance segmentation module
that generates 2D damage masks from input frames. (b)
View-synthesis generation module that computes SfM camera
parameters and builds a 3D Gaussian Splatting of the vehicle.
(c) 3D-GS segmentation module that projects the 2D mask
onto the 3D reconstruction.
damage masks. Current 3D-GS segmentation solutions intro-
duce learnable Gaussian parameters that are optimized during
training, requiring additional processing time [5]–[9]. Some
other studies propose learning-free approaches that do not
require scene-specific training [10], [11]. Using segmentation
foundation models like Segment Anything Model (SAM) [12]
they generate multiple masks that are used for global align-
ment and multi-view consistency. However such an approach
would not be reliable for some use-cases (e.g. car damage
detection) where consistent masks would be hard to pre-
dict across multiple views. Thus, we propose a single-view
segmentation method inspired by the rasterization algorithm
of 3D-GS. The Gaussians are projected on the image plane
and traversed in ascending order of depth while cumulating
their weights in a grid-like buffer, in the same manner as Z-
buffering. This way, foreground gaussians will progressively
cover the ones in the background. Remaining outliers will be
filtered out using a statistical filtering of depths and opacities.
Our complete pipeline is described in Fig. 1. We first predict
2D damage masks using an instance segmentation model
based on YOLO [13] architecture. Then, we compute camera
parameters using Structure from Motion (SfM) [14] in order
to generate a 3D Gaussian Splatting of the scene. Finally, we
use the camera parameters to project the Gaussians onto the
1
arXiv:2509.23947v1  [cs.CV]  28 Sep 2025

<!-- page 2 -->
images and select only the ones that correspond to the detected
mask.
In summary, our work offers the following two main con-
tributions:
• We propose an end-to-end pipeline for automatic dam-
age detection by treating this task from a 3D perspective.
Our approach improves the damage evaluation process by
combining the accurate visualization offered by images with
the superior geometrical representation of the damage in 3D
reconstructions. We believe that the various applications of
vehicle damage detection could benefit from our solution.
• We introduce a low-complexity single-view 3D Gaussian
Splatting segmentation method for up-lifting 2D masks to
3D. Extensive experiments demonstrate the robustness of our
method on car damage detection and similar tasks, where
multi-view segmentation masks are difficult to obtain. Our ap-
proach is particularly effective in these challenging scenarios,
generating segmentation masks that successfully capture the
target object and are consistent across multiple views.
II. RELATED WORK
A. Car automatic damage detection
Automatic damage detection is strongly related to the ex-
istence of high-quality datasets for car damage classification,
detection and segmentation. To our knowledge, there are a
few publicly available datasets that provide detection and
segmentation labels for vehicle damage. CarDD [1] is a
dataset that contains 4000 images with over 9000 instances
of damaged parts. The labels cover 6 damage classes that
are well-balanced in the number of samples and pixel size.
VehiDE [2] provides over 14000 pictures with damaged car
parts from 8 categories. Another related dataset is Tartesi-
aDS [15], which contains over 100 images with labels for
scratches and deformations, designed for insurance company
use cases. The larger of these datasets, CarDD and VehiDE,
have been used by many automatic damage detection studies,
being an important contribution to the topic. Recently, Crash-
Car101 [3] proposed a procedural damage generation pipeline
by introducing augmented damage into 3D car models, thus
obtaining synthetic 2D images with labels for car parts and
damage categories. This synthetic approach can overcome the
inefficiencies of manual annotation, enabling the creation of
large-scale damage datasets. However, it may fail to accurately
reproduce vehicle damage due to the discrepancy between real
and synthetic generated data.
The improvement of deep neural networks has revolu-
tionized the field of computer vision, enabling machines to
interpret visual data with remarkable accuracy. Over the past
decade, advances in architectures such as Convolutional Neu-
ral Networks and, more recently, transformer-based models
have significantly improved performance in tasks like image
classification, object detection, and semantic segmentation,
paving the way for reliable automatic damage detection sys-
tems in real-world applications.
Dwivedi et al. [16] used a web-scraped dataset of 2000
samples for both damage classification and detection, em-
ploying YOLO alongside various encoder architectures. This
approach demonstrated the ability to identify both the class
and location of the damage efficiently. Furthermore, Chen et
al. [17] applied MaskR-CNN [18] for road damage detection
and instance segmentation, emphasizing its ability to produce
accurate per-pixel masks of damage. Chen H. [19] uses Mask
R-CNN for detecting damaged regions and image alignment
in order to identify the modifications produced by the incident.
Lee D. [20] is introducing a three quarter view dataset with
annotations for vehicle orientation and damage type, com-
paring results from multiple backbone networks. Experiments
using YOLO architecture are performed by [21]–[23] on
existing or newly introduced damaged car parts datasets, using
various data augmentation techniques. The vast majority of
these mentioned studies provide 2D detection solutions by
training multiple detection architectures on existing datasets
and comparing their results. In contrast, our work goes beyond
2D analysis by up-lifting segmentation masks to 3D car
reconstructions, providing a more comprehensive visualization
of the damage.
B. 3D vehicle reconstruction
DreamCar [24] reconstructs a complete 3D model using
a limited number of frames (one to five). They use a 3D-
GS approach and prior-knowledge of car shapes from a
generative model. Their method is useful for autonomous
driving scenarios, where accurate representation of 3D data
is needed for bridging the gap between the 2D and 3D world
and using the existing 2D large scale datasets. Car-GS [25]
is using 3D-GS as well, but this time focusing on eliminating
the reflecting and transparent surface artifacts by introducing
other useful learnable parameters to each Gaussian. This work
can help in scenarios where high quality representations are
necessary, thereby mitigating the effects of imperfect, noisy, or
low-quality visual input data. Another 3D car reconstruction
approach is introduced by [26] that uses Structure-from-
Motion on video frames capturing vehicles in translation.
Wang et al. [27] use a 3D wireframe library and car structure
prior knowledge in order to build a 3D car model from a single
view. The solution introduced by [28] uses 3D pose estimation
in order to identify damaged parts by comparing 3D CAD
models of damaged and undamaged vehicles. Similar to our
method, [29] projects detected damage in 2D and projects the
result on 3D car reconstructions. In comparison, they project
2D bounding boxes using ray-tracing on precomputed 3D
car meshes. Car meshes are time consuming to compute and
even impossible to recreate from a limited number of vehicle
images. Additionally, we project segmentation masks instead
of bounding boxes, which allow us to perform the projection
at pixel-level accuracy.
C. 3D segmentation in Gaussian Splatting
The task of creating multi-view synthesis from a limited
number of pictures of an object is quite recent and has
become a central problem in 3D computer vision and graphics.
This challenge involves generating novel views of a scene
2

<!-- page 3 -->
by relying on learned representations that infer geometry,
appearance, and lighting. Recent breakthroughs such as Neural
Radiance Fields (NeRF) [30], 3D Gaussian Splatting [4], and
other neural rendering approaches have dramatically improved
the quality and realism of synthesized views. These methods
leverage deep learning to model scenes as continuous volu-
metric functions or compact point-based structures, enabling
high-fidelity reconstructions with minimal supervision.
Segmenting 3D objects in this kind of representations is a
difficult task due to the lack of available 3D datasets that offer
ground truth segmentation masks. Thus, the evaluation metrics
are, in general, based on comparing the 2D back-projection of
the 3D mask with the ground truth 2D mask.
Recently, most of the works on 3D-GS segmentation are
using training-based methods. Here, the semantics of the scene
are learned by using newly introduced Gaussian parameters,
supervised by 2D masks that are generated by 2D foundation
models (SAM [12]). These methods [5]–[9] develop a good
understanding of the scene semantics and could perform multi-
object real-time segmentation, but require learnable parameters
that take additional time to train. GaussianCut [31] builds a
graph that corresponds to the 3D scene and uses a graph-
cut algorithm to do interactive segmentation. FlashSplat [32]
formulates 2D to 3D segmentation as a linear programming
problem and solves it in closed form. Other methods [10], [11]
start from already trained 3D Splats and assign some labels
(IDs) to each Gaussian. Using multi-view consistency scores
or by training neural networks, they later query these labels to
perform interactive 3D segmentation. This approach requires
additional processing time and also accurate 2D segmentation
masks from multiple view, which are not always available
nor necessary (some objects require less viewing angles to
be segmented than others). Our method does not require any
additional global pretraining or postprocessing of the scene.
We use a single-view 2D mask that we up-lift to 3D by
projection and find the corresponding 3D Gaussians using
a Z-buffering algorithm and statistical filtering. This way, a
robust 3D segmentation, consistent across multiple views, is
generated from a single view.
III. METHOD
A. Preliminaries
For the vehicle damage instance segmentation we use
a pretrained version of the YOLO11 segmentation network
from Ultralytics [33]. The model is trained on CarDD [1]
and VehiDE [2] datasets that contain samples with annotated
masks for diverse damage categories. We run the inference
on multiple vehicle frames and generate 2D segmentation
masks that will be later used by the projection algorithm.
More detailed explanations and a comparison for the instance
segmentation module are also included in Sec. IV-A.
The view-synthesis generation is achieved using multiple
images capturing the vehicle from different angles. We are
first using these images to compute the camera parameters by
running COLMAP, which is a general-purpose Structure-from-
Motion (SfM) and Multi-View Stereo (MVS) pipeline [14],
[34]. Here, features are extracted and matched in multiple
images in order to compute the camera intrinsic and extrinsic
parameters alongside a sparse point-cloud. Intrinsic parameters
define how the camera maps 3D points in its local coordinate
system to 2D image coordinates and are made of the focal
length (fx, fy) meaning how strongly the camera lens con-
verges the light, and the optical center of the image (cx, cy)
(considered here to be the center of the image). These two
form the intrinsic matrix:
K =


fx
0
cx
0
fy
cy
0
0
1


(1)
Extrinsic parameters describe the camera’s position and orien-
tation in 3D space relative to the scene (world coordinates)
and are made of the translation vector t ∈R1×3 and the
rotation matrix R ∈R3×3. These are used to bring the world
origin into camera’s origin and to rotate the world coordinate
system into the camera’s orientation. The complete projection
equation from 3D world coordinates to 2D image coordinates
used for our Gaussian projection is formulated:
ximage ∼K[R | t]Xworld
(2)
where ximage and Xworld are the image and world point
coordinates.
3D Gaussian Splatting starts from the sparse point cloud
generated by SfM and initializes a set of 3D Gaussians. By
optimization, they are refined to better capture the geometry
and appearance of the scene, resulting in a densified set of
Gaussians. These are parameterized by their position µ ∈R3,
scale, orientation (represented through a quaternion), opacity
and color. 3D-GS also uses an efficient rasterization technique
for generating 2D views. First, all Gaussians are projected
onto the image plane, the screen is split into 16x16 tiles and
the Gaussians are distributed to the tiles they are intersecting.
Next, the Gaussians are sorted based on depth and a separate
thread for each tile is launched resulting in fast parallel
processing. Then, for each pixel in the tile, the sorted Gaussian
list is traversed from front to back. The accumulated opacity,
αacc, is updated with the opacity αi of each subsequent
Gaussian i until saturation is reached. This process can be
formulated as:
αacc ←αacc + (1 −αacc) · αi
if α ≥T, stop
(3)
where T is a chosen threshold, usually 1. Finally, a backward
pass is performed, from the last Gaussian that contributed to
the pixel to the front, and alpha-blending is used:
Cout ←αiCi + (1 −αi)Cprev
(4)
in order to compute the final pixel’s color Cout. Here, Ci
represents the color of the current Gaussian, and Cprev is the
color accumulated from the Gaussians located behind it.
3

<!-- page 4 -->
Fig. 2: Visual comparison of the 2D back-projected Gaussian segmentation and the ground truth mask. (a) The 3D to 2D
projection of the segmented Gaussians. (b) The polygon resulted from the Convex/Concave Hull of (a). (c) The ground truth
mask. (d) The view used for segmentation with the ground truth mask. (e) and (f) Rendering result of the segmentation by
coloring the Gaussians. (g) and (h) The segmentation result projected in other views. Scenarios shown (top to bottom): row 1:
a right-rear car scratch, row 2: a front-right wheel, row 3: a front-right car headlight.
B. Problem definition
Having a 2D mask from a single view, the goal is to lift
it to 3D in order to produce a consistent and accurate 3D
segmentation of the Gaussian Splatting. Our approach uses
an algorithm that projects the shapes of Gaussians in 2D,
keeps those that fall inside mask boundaries and cumulates
their weights in a Z-buffer. This way, the Gaussians in the
foreground will progressively cover the outliers from the
background. Additionally, we do a statistical filtering based
on normal distributions of opacity and depth in order to
further remove noise from floating Gaussians or any remaining
background outliers.
C. 2D to 3D mask up-lifting
We will first project the centers of the Gaussians on the
image plane and filter out those that fall outside the mask’s
boundaries. Considering a 3D point Xworld = (Xw, Yw, Zw) in
world coordinates, its coordinates in the camera frame Xcam =
(Xc, Yc, Zc) can be formulated as:
Xcam = R(Xworld −t),
(5)
where t ∈R3 and R ∈R3×3 are the camera’s translation
vector and rotation matrix, respectively. To project this 3D
camera point into 2D image coordinates (x, y), we use the
following equations:
x = fx · Xc
Zc
+ w
2 ,
y = fy · Yc
Zc
+ h
2 ,
(6)
where (fx, fy) are the camera’s focal lengths and (w, h) is the
image size in pixels.
Having the Gaussians inside the 2D mask, we will start
the Z-buffering algorithm by sorting them in ascending order
of the distance from the camera. For each Gaussian, we will
check if its center position in the buffer is lower than β,
a threshold that is calculated as the mean opacity of the
Gaussians already in the buffer.
For the 2D shape projection we begin with a 3D Gaussian
represented by a mean position p ∈R3, a rotation (orientation)
matrix Rg ∈SO(3) derived from a quaternion, and a scale
vector s ∈R3. The scale is stored in logarithmic form for
numerical stability, so we define the scale matrix as:
S = diag(exp(sx), exp(sy), exp(sz))
(7)
The 3D covariance matrix of the Gaussian in world space is
constructed by rotating the scaled ellipsoid:
Σworld = RgS2R⊤
g
(8)
4

<!-- page 5 -->
To project this Gaussian into the camera frame, we apply the
camera’s rotation matrix Rcam:
Σcam = RcamΣworldR⊤
cam
(9)
Next, we project the 3D covariance into 2D image space using
the Jacobian of the perspective projection. For a point pcam =
(x, y, z) in camera coordinates, the Jacobian J is:
J =
 fx
z
0
−fxx
z2
0
fy
z
−fyy
z2

(10)
where fx and fy are the camera focal lengths in pixels. The
2D image-space covariance becomes:
Σimage = JΣcamJ⊤
(11)
This covariance describes an elliptical Gaussian in the 2D
image. To evaluate the spatial extent of this Gaussian, we
compute the Mahalanobis distance for each pixel within a
region of interest:
D2(x) = (x −µ)⊤Σ−1
image(x −µ)
(12)
where µ = (xproj, yproj) is the projected mean. The final
Gaussian weight at each pixel is computed as:
w(x) = α · exp

−1
2D2(x)

(13)
where α is the opacity. This weight value represents the
contribution of the Gaussian to that pixel.
As previously mentioned, the weights are summed into
a buffer resulting in a subset of 3D Gaussians. Artifacts
caused by poor reconstruction quality may be present in the
scene. Thus, we remove Gaussians that do not fall within two
standard deviations (2σ) of the mean depth and opacity values
of the current subset.
Fig. 3 shows us a comparison of segmentations with and
without the statistical filtering for opacity and depth. In case of
depth, we can see that there are visible background Gaussians
that were improperly segmented, mostly because of failures
in the Z-buffering algorithm. These outliers are successfully
filtered out by using the normal distribution of depth and
only keeping Gaussians within 2 standard deviations from the
mean. The same thing is done with the opacity value. Floating
Gaussians between the camera and the object caused by noise
are wrongly segmented but filtered out by applying the same
approach.
During depth-buffering, some Gaussians may be wrongly
filtered out due to overlapping in the grid, causing an in-
complete or sparse mask. Thus, this algorithm may not be
enough for obtaining our final segmentation, but very effective
for identifying the boundaries of the 3D mask by suppressing
background artifacts. Consequently, we traverse again the list
of Gaussian and fill the mask with those that are in between
2 standard deviations away from the mask’s mean depth.
Fig. 3: The effect of statistical filtering on segmentation qual-
ity. Our method removes background outliers based on depth
and eliminates noisy floating Gaussians based on opacity.
IV. EXPERIMENTS AND RESULTS
A. 2D Car damage segmentation
For the instance segmentation of car damage, we conducted
experiments on both existing car damage datasets, namely
CarDD [1] and VehiDE [2]. These datasets contain segmen-
tation labels for various damage classes. Table I presents a
comparison including the number of images, instances and
damage categories of these datasets.
TABLE I: Overview of publicly available car damage detection
datasets.
Dataset
Images
Instances
Damage Categories
CarDD [1]
4000
9000
scratch,
dent,
glass
shatter,
crack, tire flat, and lamp bro-
ken
VehiDE [2]
13945
32000
dents, broken glass, scratch,
lost parts, punctured, torn, bro-
ken lights, and non-damage
In both dataset papers, authors introduce experiments with
different Recurrent Convolutional Neural Network architec-
tures and conclude that the best result is obtained by DCN
(Deformable convolutional networks) [35]. Consequently, we
decided to run experiments using a different type of archi-
tecture, namely YOLO [13]. We trained two versions of the
YOLOv11 network from Ultralytics [33].The datasets train/test
split was made using the original structure provided by the
authors. CarDD has 2816 samples in the training set and 810
in the validation set. VehiDE provides 11621 training samples
and 2324 validation samples, with the rest of the samples being
in the test sets.
5

<!-- page 6 -->
(a) Left-view mask
(b) Center-view mask
(c) Right-view mask
Ours(left)
SAGD(left)
Ours(center)
SAGD(center)
Ours(right)
SAGD(right)
Fig. 4: Qualitative comparison with the multi-view method SAGD [11]. SAGD’s approach requires consistent masks from
multiple views (a, b, c) to produce its segmentation. In contrast, our method achieves a visually comparable result using
only a single mask from the center view (b), demonstrating its effectiveness for scenarios where obtaining reliable multi-view
annotations is impractical or impossible.
TABLE II: Car damage instance segmentation training results
in terms of box (b) and mask (m) Average Precision.
Dataset
Network
mAP50 (b/m)
mAP50 −95 (b/m)
CarDD [1]
YOLO11-l
76.7 / 75.8
61.8 / 58.5
YOLO11-x
76.2 / 75.5
62.0 / 58.6
VehiDE [2]
YOLO11-l
53.9 / 51.2
36.1 / 29.4
YOLO11-x
55.4 / 52.3
37.5 / 30.2
Table II shows a summary of mAP metrics for both datasets
and YOLO models. In summary, the YOLO11-x version
performs slightly better than YOLO11-l, but having more than
double parameters, making us choose the lighter version. The
inference time for YOLO11-l on our system (Apple M3 Pro)
is under 100ms for one image of size 640x640.
B. 3D Gaussian Splatting Segmentation
In order to demonstrate the robustness of our method, we
conducted experiments on self-recorded data of damaged cars
but also on samples from public 3D-GS datasets.
In order to check the consistency of our projected mask,
we run the algorithm and back-project the segmentation re-
sult onto the original image that was used as input for the
segmentation. The resulting 2D mask is a set of 2D Gaussian
shapes that may have an irregular boundary or may not be
fully connected. In order to compare this mask with the ground
truth one, we use convex and concave hulls, depending on the
resulting shape (we use this step only for the comparisons on
self-recorded data). Now, the generated polygon is compared
with ground truth in terms of Intersection over Union, F1 score
and Accuracy. Some qualitative results of this experiment
are shown in Fig. 2. We demonstrate the consistency of our
segmentation results by visualizing the generated masks from
different angles, on the rendering and projected on vehicle
images.
The mentioned metrics used for the comparison are pre-
sented in Table III, together with the processing time required
for the segmentation. Besides the comparison with the view
used as input, we manually labeled segmentation masks for
two more views that capture the damage, (g) and (h) from
Fig. 2 (the masks displayed in the figures are the projection
of our result, not these ground truth masks). We calculate the
mean metrics over these three views in order to show that our
results are consistent across multiple unseen views.
In order to compare our method with other studies, we
use the SPIn-NeRF [36] which include 2D segmentation
masks of some objects from multiple public 3D reconstruction
datasets. For the evaluation, we use only one ground truth
mask that we up-lift using our method. The resulting 3D
segmentation is then projected in all the views from the scene
and compared with the ground truth mask in order to compute
IoU and Accuracy. These results are compared with other 3D
segmentation methods in Table IV. The results reported for the
other methods are taken directly from the original publications,
namely SA3D [6] and SAGD [11]. Also, for generating the
3DGS scenes, we used OpenSplat [37] and optimized for 7k
steps. The underline results mark the scenes where our single-
view approach outperformed multi-view methods.
We could observe that although only one view was used for
the segmentation, the results are comparable with multi-view
approaches, some of them requiring scene-specific training.
More than this, we also outperformed the baseline multi-view
segmentation method (Single View [6]) on some scenes, ob-
taining a similar mean for the IoU and Accuracy on the SPIn-
6

<!-- page 7 -->
fortress
lego
orchids
pinecone
Fig. 5: Qualitative comparison of our method on scenes from SPIn-NeRF [36].
NeRF scenes. Despite its name, this baseline method uses
a multi-view approach. It starts from an initial ground-truth
mask that mapped to the 3D space using depth information in
order to obtain an initial 3D segmentation. This result is then
back-projected into other views, resulting in 2D prompt points
that are passed to SAM in order to further generate 2D masks
into the corresponding views.
TABLE III: Quantitative results of our segmentation method.
All metrics are calculated compared to the input ground truth
masks used for segmentation - left / mean across three different
views (the input one and two unseen views) - right.
Object
IoU(%)
F1(%)
Acc(%)
Time(s)
Scratch
(Fig. 2 - top)
65.69 / 52.44
79.29 / 67.82
99.30 / 99.28
0.04
Flat tire
(Fig. 2 - center)
88.15 / 87.16
93.70 / 93.13
99.01 / 99.01
0.12
Broken lamp
(Fig. 2 - bottom)
82.41 / 67.07
90.36 / 78.75
97.93 / 97.70
0.31
Another important comparison that we conduct is on the
commonly used ”Truck” scene from the Tanks and Tem-
ples [38] benchmark dataset. This scene is presented in the
visual results from most of the 3D-GS segmentation studies.
In Fig. 4 we compare our method with SAGD [11], which
introduces a very intuitive learning-free segmentation approach
based on a multi-view label voting mechanism. Our method
obtained similar results by using only a single view for the
segmentation and without any needed preprocessing.
We report end-to-end, per-instance overhead, excluding re-
construction (which all methods require), as a practical mea-
sure for comparison. Our single-view, training-free pipeline
uses one 2D mask and a lightweight up-lifting step, which
results in sub-second CPU latency per instance in our exper-
iments. We don’t include kernel timing comparisons across
papers, as fair benchmarking would require matching hardware
setups and numbers of views.
For additional context, in the “Truck” wheel scene, our
CPU-side steps (sorting, projection, segmentation) took under
3 seconds. In comparison, we ran SAGD [11], a multi-view
segmentation method, with its projection and voting stages
taking under a second on an A100/40GB GPU, but its total
runtime depends heavily on the number of views, since it
requires generating a mask per view (each usually taking
around 0.4 seconds, as specified in [11]). There also exist
segmentation methods capable of achieving millisecond-level
performance. However, they typically require scene-specific
optimization, which increases preprocessing time and reduces
general applicability. We therefore emphasize the practical
advantage: a computationally efficient single-view method that
avoids the need for multi-view setups or reliance on large
foundation models like SAM, which, in addition to high
7

<!-- page 8 -->
TABLE IV: Quantitative evaluation on SPIn-NeRF [36] scenes, comparing our method with existing multi-view approaches.
Underlined entries highlight scenarios where our single-view method outperforms the multi-view baselines (Single View [6]
is a multi-view method despite its name (see Section IV-B).
Scenes
Single View [6]
MVSeg [36]
SA3D [6]
SAGD [11]
Ours
IoU
Acc
IoU
Acc
IoU
Acc
IoU
Acc
IoU
Acc
Orchids
79.4
96.0
92.7
98.8
83.6
96.9
85.4
97.5
72.1
93.9
Ferns
95.2
99.3
94.3
99.2
97.1
99.6
92.0
98.9
76.0
96.0
Horns
85.3
97.1
92.8
98.7
94.5
99.0
91.1
98.4
89.1
98.0
Fortress
94.1
99.1
97.7
99.7
98.3
99.8
95.6
99.5
83.2
97.4
Pinecone
57.0
92.5
93.4
99.2
92.9
99.1
92.6
99.0
81.3
97.3
Lego
76.0
99.1
74.9
99.2
92.2
99.8
90.2
99.7
78.0
99.2
Mean
81.1
97.1
90.9
99.1
93.1
99.0
91.1
98.8
79.9
96.9
processing time, can struggle with fine-grained or niche-class
segmentation.
In Fig. 5, we present qualitative results of our method
on several scenes from the SPIn-NeRF dataset. Although the
segmentation is performed using only a single view as input,
the objects in the scenes are successfully identified and seg-
mented. Moreover, the resulting masks maintain consistency
and visual quality when rendered from multiple viewpoints,
demonstrating the robustness of our approach.
V. DISCUSSIONS AND LIMITATIONS
Our automatic vehicle damage detection solution can be im-
plemented in multiple real-world applications. Car insurance
companies could benefit from it by introducing this approach
in software applications for visualizing and analyzing the
vehicle 3D model in order to approximate the severity and
cost of the damage. Car mechanic shops could generate
interactive reports and send tailored offers based on the vehicle
reconstruction generated from a video taken by customers.
Car-selling websites can include this as a feature in order for
the potential buyers to have the most accurate visualization of
the vehicle condition.
Limitations of our approach can occur in multiple modules
that we rely on. First, the instance segmentation network fails
frequently, either by not fully covering the damage with the
mask or by giving false positive damage predictions caused by
shadows or reflections. The only publicly available datasets
for damage detection, that we also used in this work, have
only a few thousand samples, which has been demonstrated
to be insufficient for robust object detection. This limitation
can be solved in the future by either using a synthetic dataset
approach like [3] or by enhancing the detection architecture.
Regarding the 3D-GS module, the quality of our final seg-
mentation is inherently linked to the underlying reconstruction,
which depends on the quantity and quality of input images.
This dependency is most apparent on challenging surfaces,
such as glass, where reflections and transparency can lead to
noisy or low-opacity Gaussians, causing segmentation errors.
While refinement methods like those proposed by [25] can
mitigate these artifacts, these reconstruction challenges high-
light a broader issue in the field: the scarcity of comprehensive
3D vehicle damage datasets. This scarcity necessitated our
reliance on back-projected 2D metrics for evaluation. A crucial
next step for validating this and similar methods involves the
creation of such datasets, which would enable the use of more
robust 3D-specific metrics like 3D IoU and Chamfer distance.
Furthermore, expanding the evaluation to include more diverse
real-world and synthetic datasets, alongside a broader com-
parative analysis against other 3D-GS segmentation baselines,
remains an important avenue for future work to fully establish
the robustness of single-view approaches.
As mentioned previously, our method is similar to the
rasterization algorithm from 3D-GS, given that both traverse
and process a list of 2D-projected Gaussians. In future work,
we could implement our algorithm using an approach similar
to their rasterization. By splitting the mask region into multiple
tiles, we could perform depth-buffering using multiple threads,
significantly reducing the processing time.
VI. CONCLUSION
This work presents a complete pipeline for automatic vehi-
cle damage detection that combines 2D and 3D segmentation
in order to visualize damaged car parts on a 3D reconstruction
of the vehicle. We conducted several experiments with instance
segmentation models on two public damage datasets. We
generated the 3D reconstruction of the vehicle using 3D Gaus-
sian Splatting and showed the benefits of this view-synthesis
method for our task. Extensive 3D segmentation experiments
demonstrate the robustness of our single-view method, achiev-
ing results comparable to multi-view approaches without the
associated preprocessing overhead. By eliminating the need
for complex multi-view annotations, our approach paves the
way for practical, scalable 3D damage analysis and other fine-
grained 3D instance segmentation tasks directly from casual
video captures.
REFERENCES
[1] X. Wang, W. Li, and Z. Wu, “Cardd: A new dataset for vision-based
car damage detection,” IEEE Transactions on Intelligent Transportation
Systems, vol. 24, no. 7, pp. 7202–7214, 2023.
[2] N. T. Huynh, N. N. Tran, A. T. Huynh, V.-D. Hoang, and H. D. Nguyen,
“Vehide dataset: New dataset for automatic vehicle damage detection in
car insurance,” in 2023 15th International Conference on Knowledge
and Systems Engineering (KSE).
IEEE, 2023, pp. 1–6.
8

<!-- page 9 -->
[3] J. Parslov, E. Riise, and D. P. Papadopoulos, “Crashcar101: Procedural
generation for damage assessment,” in Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision (WACV), January
2024.
[4] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions
on
Graphics,
vol.
42,
no.
4,
July
2023.
[Online].
Available:
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
[5] J. Cen, J. Fang, C. Yang, L. Xie, X. Zhang, W. Shen, and Q. Tian,
“Segment any 3d gaussians,” in Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 39, 2025, pp. 1971–1979.
[6] J. Cen, J. Fang, Z. Zhou, C. Yang, L. Xie, X. Zhang, W. Shen, and
Q. Tian, “Segment anything in 3d with radiance fields,” International
Journal of Computer Vision, pp. 1–23, 2025.
[7] S. Choi, H. Song, J. Kim, T. Kim, and H. Do, “Click-gaussian:
Interactive segmentation to any 3d gaussians,” in European Conference
on Computer Vision.
Springer, 2024, pp. 289–305.
[8] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari,
S. You, Z. Wang, and A. Kadambi, “Feature 3dgs: Supercharging 3d
gaussian splatting to enable distilled feature fields,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 21 676–21 685.
[9] R. Zhu, S. Qiu, Z. Liu, K.-H. Hui, Q. Wu, P.-A. Heng, and C.-W. Fu,
“Rethinking end-to-end 2d to 3d scene segmentation in gaussian splat-
ting,” in Proceedings of the Computer Vision and Pattern Recognition
Conference, 2025, pp. 3656–3665.
[10] Y. Guo, J. Hu, Y. Qu, and L. Cao, “Wildseg3d: Segment any 3d objects
in the wild from 2d images,” arXiv preprint arXiv:2503.08407, 2025.
[11] X. Hu, Y. Wang, L. Fan, J. Fan, J. Peng, Z. Lei, Q. Li, and Z. Zhang,
“Sagd: Boundary-enhanced segment anything in 3d gaussian via gaus-
sian decomposition,” arXiv preprint arXiv:2401.17857, 2024.
[12] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,”
in Proceedings of the IEEE/CVF international conference on computer
vision, 2023, pp. 4015–4026.
[13] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look
once: Unified, real-time object detection,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2016, pp. 779–
788.
[14] J. L. Sch¨onberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Conference on Computer Vision and Pattern Recognition (CVPR),
2016.
[15] S. A. P´erez-Zarate, D. Corzo-Garc´ıa, J. L. Pro-Mart´ın, J. A. ´Alvarez-
Garc´ıa, M. A. Mart´ınez-del Amor, and D. Fern´andez-Cabrera, “Auto-
mated car damage assessment using computer vision: Insurance com-
pany use case,” Applied Sciences, vol. 14, no. 20, p. 9560, 2024.
[16] M. Dwivedi, H. S. Malik, S. Omkar, E. B. Monis, B. Khanna, S. R.
Samal, A. Tiwari, and A. Rathi, “Deep learning-based car damage
classification and detection,” in Advances in artificial intelligence and
data engineering: Select proceedings of AIDE 2019.
Springer, 2021,
pp. 207–221.
[17] Q. Chen, X. Gan, W. Huang, J. Feng, and H. Shim, “Road damage
detection and classification using mask r-cnn with densenet backbone,”
Computers, Materials, Continua, vol. 65, pp. 2201–2215, 01 2020.
[18] K. He, G. Gkioxari, P. Doll´ar, and R. Girshick, “Mask r-cnn,” in
Proceedings of the IEEE international conference on computer vision,
2017, pp. 2961–2969.
[19] H. Chen, “Car damage detection and patch-to-patch self-supervised
image alignment,” arXiv preprint arXiv:2403.06674, 2024.
[20] D. Lee, J. Lee, and E. Park, “Automated vehicle damage classification
using the three-quarter view car damage dataset and deep learning ap-
proaches,” Heliyon, vol. 10, no. 14, p. e34016, 2024. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S2405844024100473
[21] S.
A.
P´erez-Zarate,
D.
Corzo-Garc´ıa,
J.
L.
Pro-Mart´ın,
J.
A.
´Alvarez Garc´ıa, M. A. Mart´ınez-del Amor, and D. Fern´andez-Cabrera,
“Automated car damage assessment using computer vision: Insurance
company use case,” Applied Sciences, vol. 14, no. 20, 2024. [Online].
Available: https://www.mdpi.com/2076-3417/14/20/9560
[22] M. R. S. Ramazhan, A. Bustamam, and R. A. Buyung, “Smart
car damage assessment using enhanced yolo algorithm and image
processing techniques,” Information, vol. 16, no. 3, 2025. [Online].
Available: https://www.mdpi.com/2078-2489/16/3/211
[23] Y. S, R. J. J, and R. Vasanthi, “Yolov8-powered real-time car damage
detection,” in 2025 3rd International Conference on Intelligent Data
Communication Technologies and Internet of Things (IDCIoT), 2025,
pp. 2223–2228.
[24] X. Du, H. Sun, M. Lu, T. Zhu, and X. Yu, “Dreamcar: Leveraging car-
specific prior for in-the-wild 3d car reconstruction,” IEEE Robotics and
Automation Letters, 2024.
[25] C. Li, J. Wang, X. Wang, X. Zhou, W. Wu, Y. Zhang, and T. Cao, “Car-
gs: Addressing reflective and transparent surface challenges in 3d car
reconstruction,” arXiv preprint arXiv:2501.11020, 2025.
[26] A. Auclair, L. Cohen, and N. Vincent, “A robust approach for 3d cars
reconstruction,” vol. 4522, 06 2007, pp. 183–192.
[27] B. Wang, Q. Wu, H. Wang, L. Hu, and B. Li, “3d surface reconstruction
of car body based on any single view,” IEEE Access, vol. 12, pp. 74 903–
74 914, 2024.
[28] S. Jayawardena, “Image based automatic vehicle damage detection,”
Ph.D. dissertation, Australian National University, 11 2013.
[29] R. E. van Ruitenbeek and S. Bhulai, “Multi-view damage inspection
using single-view damage projection,” Machine Vision and Applications,
vol. 33, no. 3, p. 46, 2022. [Online]. Available: https://doi.org/10.1007/
s00138-022-01295-w
[30] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[31] U. Jain, A. Mirzaei, and I. Gilitschenski, “Gaussiancut: Interactive
segmentation via graph cut for 3d gaussian splatting,” in The Thirty-
eighth Annual Conference on Neural Information Processing Systems,
2024.
[32] Q. Shen, X. Yang, and X. Wang, “Flashsplat: 2d to 3d gaussian splatting
segmentation solved optimally,” in European Conference on Computer
Vision.
Springer, 2024, pp. 456–472.
[33] G. Jocher, J. Qiu, and A. Chaurasia, “Ultralytics yolo,” 2023, page:
https://github.com/ultralytics/ultralytics, Version: 8.0.0, License: AGPL-
3.0.
[34] J. L. Sch¨onberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, “Pixel-
wise view selection for unstructured multi-view stereo,” in European
Conference on Computer Vision (ECCV), 2016.
[35] J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei, “Deformable
convolutional networks,” in Proceedings of the IEEE international
conference on computer vision, 2017, pp. 764–773.
[36] A. Mirzaei, T. Aumentado-Armstrong, K. G. Derpanis, J. Kelly, M. A.
Brubaker, I. Gilitschenski, and A. Levinshtein, “SPIn-NeRF: Multiview
segmentation and perceptual inpainting with neural radiance fields,” in
CVPR, 2023.
[37] P. Toffanin, “Opensplat,” 2024, page: https://github.com/pierotofy/
OpenSplat, Version: 1.1.4, License: AGPL-3.0.
[38] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics, vol. 36, no. 4, 2017.
9
