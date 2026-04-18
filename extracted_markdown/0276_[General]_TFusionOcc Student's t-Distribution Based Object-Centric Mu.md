<!-- page 1 -->
1
TFusionOcc: Student’s t-Distribution Based
Object-Centric Multi-Sensor Fusion Framework for
3D Occupancy Prediction
Zhenxing Ming, Julie Stephany Berrio, Mao Shan, and Stewart Worrall
Abstract—3D
semantic
occupancy
prediction
enables
au-
tonomous vehicles (AVs) to perceive fine-grained geometric and
semantic structure of their surroundings from onboard sensors,
which is essential for safe decision-making and navigation. Recent
models for 3D semantic occupancy prediction have successfully
addressed the challenge of describing real-world objects with
varied shapes and classes. However, the intermediate repre-
sentations used by existing methods for 3D semantic occu-
pancy prediction rely heavily on 3D voxel volumes or a set
of 3D Gaussians, hindering the model’s ability to efficiently
and effectively capture fine-grained geometric details in the
3D driving environment. This paper introduces TFusionOcc, a
novel object-centric multi-sensor fusion framework for predicting
3D semantic occupancy. By leveraging multi-stage multi-sensor
fusion, Student’s t-distribution, and the T-Mixture model (TMM),
together with more geometrically flexible primitives, such as the
deformable superquadric (superquadric with inverse warp), the
proposed method achieved state-of-the-art (SOTA) performance
on the nuScenes benchmark. In addition, extensive experiments
were conducted on the nuScenes-C dataset to demonstrate
the robustness of the proposed method in different camera
and lidar corruption scenarios. The code will be available at:
https://github.com/DanielMing123/TFusionOcc
Index Terms—Autonomous driving, 3D semantic occupancy
prediction, Multi-sensor fusion, Environment perception
I. INTRODUCTION
Accurate and efficient modeling of the scene around au-
tonomous vehicles (AV) is essential for safe and collision-
free navigation. As technology advances, the introduction
of a 3D semantic occupancy representation has successfully
addressed the limitations of traditional 3D object detection and
2D bird’s-eye-view (BEV) segmentation networks, particularly
for detecting irregular objects and predicting occupied space
in 3D. Thus, 3D semantic occupancy provides a suitable
representation of the vehicle’s surrounding environment. This
representation inherently ensures geometric consistency of the
AV’s driving scene and can accurately describe occluded areas.
In addition, a multi-sensor fusion technique for 3D semantic
occupancy prediction further enhances an AVs’ ability to
model the 3D world.
Recent advances in voxel-based multi-sensor fusion meth-
ods [1]–[7] have achieved remarkable progress by integrat-
ing complementary information from different sensors, such
as surround-view cameras, surround-view radars, and lidars.
This work has been supported by the Australian Centre for Robotics
(ACFR). The authors are with the ACFR at the University of Sydney
(NSW, Australia). E-mails: zmin2675@uni.sydney.edu.au, {j.berrio, m.shan,
s.worrall}@acfr.usyd.edu.au
Demonstrating superior robustness and overall performance
in several challenging light and weather conditions, these
methods heavily rely on volumetric grid calculations (Fig 1,
voxel-based approach), which hinders real-time application
and makes them unsuitable for edge device deployment. Mean-
while, recent advances in object-centric multi-sensor fusion
methods [8], [9] aim to address the limitations of the voxel-
based approach by utilizing a set of 3D Gaussian primitives
to only model the occupied area in the driving scene (Fig
1, 3D Gaussian-based approach), reducing the computational
overhead caused by empty regions by concentrating represen-
tational capacity where it matters most. However, their model-
ing performance is inferior to that of voxel-based approaches.
This performance gap is mainly due to the limited geometric
shape representation capability of primitives and their weaker
robustness to outliers.
To address the aforementioned limitations, we propose TFu-
sionOcc (Fig 1, our approach), a novel Student’s t-distribution-
based object-centric multi-sensor fusion framework. We adopt
a multi-stage feature-level fusion strategy, including early-
stage, middle-stage, and late-stage fusion, to better integrate
complementary information across modalities and mitigate
the drawbacks of each modality. For object-centric modeling,
we adopt a Student’s t-distribution and the T-Mixture model
(TMM) as a probability kernel, combined with several types of
primitives, such as the general T-Primitive, superquadric, and
deformable superquadric (superquadric with inverse-warp),
to achieve greater geometric shape representation flexibility
and robustness against outliers. The extensive experiments
conducted on nuScenes [10] and nuScenes-C [11] datasets
demonstrate that our proposed approach not only achieves
state-of-the-art (SOTA) performance but also exhibits supe-
rior robustness against several sensor corruptions. The main
contributions of this paper are summarized as follows:
• We propose TFusionOcc, a novel object-centric multi-
stage multi-sensor fusion framework that leverages dif-
ferent types of T primitives to efficiently and effectively
model the 3D driving scene for 3D semantic occupancy
prediction.
• We propose a multi-gate cross-attention fusion module
(MGCAFusion), which consists of a weighted summation
and gated concatenation fusion modules, to combine the
multi-modal features in an effective way.
• We propose a set of Student’s t-distribution-based
primitives, including the general T-Primitive, the T-
arXiv:2602.06400v1  [cs.CV]  6 Feb 2026

<!-- page 2 -->
2
Cam 3D Volume
Lidar 3D Volume
Feature 
Fusion
3D Fused Volume
2D Backbone
3D Backbone
Surround-View Images
Lidar Point Cloud
Visual Feats
VT
Lidar 3D Volume
3D Fused Volume
2D Backbone
3D Backbone
Surround-View Images
Lidar Point Cloud
Visual Feats
3D-Gaussian 
Primitives
Feature 
Sampling
3D-Gaussian 
Splatting
Semantic Aware Lidar 
Cylindrical Volume
3D Fused Volume
2D Backbone
3D Backbone
Surround-View Images
Lidar Point Cloud
Depth-Aware Visual Feats
T-Primitives
T-Splatting
Early-Fusion
Middle-Fusion
Transformer
Voxel-Based 
Approach
3D-Gaussian-Based 
Approach
Our Approach
Camera Flow
Lidar Flow
Feature-Fusion Flow
Refined-T-Primitives
VT: 2D-to-3D View-Transformation
Arrow Explain:
Fig. 1: Pipeline of three approaches: Voxel-based approach (top-left), 3D-Gaussian-Primitive-based object-centric approach (top-right),
and our approach (bottom-left).
Superquadric primitive, and the T-Superquadric with in-
verse warp primitive, to effectively model 3D driving
scenes.
• We thoroughly investigated the robustness of the pro-
posed method with respect to the impact of different per-
ception ranges and different lidar and camera corruption
scenarios.
The remainder of this paper is structured as follows. Section
II provides an overview of related research and identifies the
key differences between this study and previous publications.
Section III outlines the general framework of TFusionOcc and
offers a detailed explanation of the implementation of each
module. Section IV presents the results of our experiments.
Finally, Section V provides the conclusion of our work.
II. RELATED WORK
A. Voxel-Based Approach
In OccFusion [1], the authors extract a set of 3D voxel vol-
umes from each different modality sensor, such as surround-
view cameras, surround-view radars and lidar, followed by
the feature fusion operation through the proposed dynamic
feature fusion module to perform multi-sensor fusion-based
3D semantic occupancy prediction. The model exhibits strong
robustness under challenging rain and night scenarios. In Co-
Occ [2], the authors leverage the volume rendering tech-
nique, which originally derived from NeRF [12], to refine
the intermediate 3D feature volumes. They also proposed
a GSFusion module for efficient multimodal feature fusion,
achieving SOTA performance. In OccCylindrical [4], the au-
thors proposed a multi-sensor fusion-based framework that
uses a cylindrical partition to perform voxelisation of a lidar-
captured 3D point cloud and a pseudo 3D point cloud predicted
by surround-view cameras, resulting in better preservation of
3D geometric information. The feature fusion and further
refinement operations are performed under the cylindrical
coordinate system. In FusionOcc [3], the authors proposed
a cross-modality fusion module to generate a high-quality
depth-aware visual feature, followed by a 2D to 3D view
transformation, resulting in a depth-aware visual feature vol-
umes under the Cartesian coordinate system. The lidar branch
conducts voxelization and voxel encoding, resulting in a lidar
feature volume also under the Cartesian coordinate system.
Different modality feature volumes were fused via feature
channel concatenation and further refined by a 3D Encoder. In
DAOcc [5], the authors leverage multi-task learning to intro-
duce an additional 3D supervision signal to guide intermediate
features, leading to robust generalization and overall SOTA
performance. Similarly, in Inverse++ [13], the authors leverage
multi-task learning by incorporating an extra auxiliary 3D
object detection branch. In addition, SDGOcc [6] incorporates
a multi-task head to introduce a 2D semantic mask into
the model training procedure, leveraging an additional 2D
semantic mask to enhance the precision and densification of
depth maps obtained from lidar point cloud projection, thereby
boosting the model’s overall performance.
Despite achieving remarkable performance, these voxel-
based methods incur redundant calculations for empty voxels,
which are often dominant in 3D driving scenarios, resulting
in inefficient computation. Our study presents a novel object-
centric approach, aiming to address the aforementioned con-
straints by leveraging T-Primitives to model only occupied
space in driving scenarios.
B. Object-Centric Modelling Approach
In the GaussianFormer series [14], [15], the authors pro-
posed a 3D Gaussian-based object-centric modeling approach
to improve computational efficiency. They further proposed
a Gaussian-mixture model (GMM) to improve the model’s
robustness. The 3D-to-3D Gaussian splatting operator, initially
proposed in [16] for 3D-to-2D splatting, is further proposed

<!-- page 3 -->
3
to boost the model’s efficiency. In QuadricFormer [17], the
authors proposed leveraging a superquadric primitive com-
bined with a Gaussian probability kernel to achieve object-
centric modeling and benefit from the geometric shape repre-
sentation’s flexibility, thereby improving model performance.
In GaussianFormer3D [8], the authors extend the Gaussian-
Former work into a multi-sensor fusion scenario. By utilizing
a set of 3D Gaussian primitives as an intermediate feature
representation, the authors designed a pipeline that enables
each 3D Gaussian primitive to aggregate features from differ-
ent modalities and fuse them to perform 3D primitive-to-3D
voxel volume Gaussian splatting. Benefiting from the object-
centric modeling technique and multi-modal fusion, the model
achieves significant performance and efficiency improvements.
In GaussianFusionOcc [9], the authors proposed a Gaussian-
FusionBlock module that performs efficient and high-quality
feature aggregation and fusion across different modalities,
allowing high-quality 3D Gaussian primitive modeling and
leading to SOTA performance.
Though our proposed approach belongs to the object-centric
modeling category, the key difference of our work compared
to previous ones is that we explored Student’s t-distribution
and T-mixture model to enhance the model robustness. Fur-
thermore, we proposed a multi-stage feature fusion pipeline
to mitigate the drawbacks of each modality, resulting in better
fused features. Lastly, we enhanced the primitive’s geomet-
ric flexibility to provide a better intermediate representation,
thereby achieving SOTA performance.
III. TFUSIONOCC
This work leverages a set of Student-T-distribution-based
primitives combined with a 3D to 3D splatting operation
to generate a dense 3D semantic occupancy grid of the
surrounding scene by integrating information from surround-
view cameras and lidar. Thus, the problem can be formulated
as follows:
P T
refined = F
 Cam1, Cam2, ..., CamN, Lidar, P T 
(1)
Occ = Splat
 P T
refined

(2)
where F refers to a multi-sensor fusion framework, P T
refers to a set of primitives based on T-distribution P T =

pT
1 , pT
2 , ..., pT
K
	
, along with a set of query vectors Q =
{q1, q2, ..., qK}, P T
refined refers to a set of T-distribution-based
primitives refined through multi-sensor fusion framework,
N refers to the total number of surround-view images and
Splat (·) refers to the splatting operation. In the final predicted
3D semantic occupancy Occ ∈R{X×Y ×Z}, each grid is
assigned a semantic property ranging from 0 to C, where C
refers to the total amount of semantic classes. In our case, a
class value of 0 corresponds to an empty grid.
A. Overview
The overall architecture is shown in Figure 2. For the
camera branch, given

img1, img2, ..., imgN	
surround-view
images, a 2D encoder and a decoder are leveraged to ex-
tract V Cam
=
n
V l
n
	N
n=1 ∈RC×Hl×Wl
oL=3
l=1
multi-scale
visual features. Meanwhile, a DepthNet is leveraged, re-
sulting in a camera-based pseudo 3D point cloud Ptcam
and a multi-scale camera-based dense depth maps V cam
depth =
n
V l
depth n
oN
n=1 ∈RD×Hl×Wl
L=3
l=1
, where D refers to the
total amount of depth-bins. The pseudo 3D point cloud
Ptcam is further divided into cylindrical voxels, resulting
in a camera-based 3D voxel volume V olCam
Cylin defined in
cylindrical coordinates. For the lidar branch, given a 3D
point cloud Ptlidar from lidar, the cylindrical partition is
first applied, resulting in a lidar-based 3D voxel volume
V olLidar
Cylin. Then a 3D encoder (e.g., Cylinder3D [18]) is used
to extract lidar features V Lidar in cylindrical coordinates.
Meanwhile, PtLidar is projected onto each surround-view
camera frame, resulting in sparse multi-scale lidar-based depth
maps V lidar
depth =
n
V l
depth n
oN
n=1 ∈RD×Hl×Wl
L=3
l=1
. Then, a
multi-stage feature fusion strategy is adopted. For the camera
branch side, V cam
depth ⊕V lidar
depth results in multi-scale fused dense
depth maps V fuse
depth. Then V fuse
depth is encoded by a multi-layer
perceptron (MLP) and added to V Cam, resulting in multi-
scale depth-aware visual features F Cam
depth. In addition, an outer
product operation F Cam
depth ⊗V fuse
depth is conducted, resulting
in a multi-scale lifted depth-aware visual feature V olCam
depth.
For the lidar branch side, each occupied voxel center of
V olLidar
Cylin serves as an anchor to project onto V Cam, thereby
aggregating semantic information. The aggregated semantic
information is further fused with the lidar feature V olLidar
Cylin
through an early-fusion module, resulting in a semantic-aware
lidar feature V olLidar
sem . The projection operation begins with a
transition from a cylindrical coordinate system to a Cartesian
coordinate system, followed by a transition to a new coordinate
frame. The V olCam
Cylin and V olLidar
Cylin are fused through the
skeleton merge module, resulting in a fused voxel volume
V olF used
Cylin under cylindrical coordinates. The V olF used
Cylin is then
used as primitive anchors to initialize T-Primitives P T under
cylindrical coordinates. The P T , V olCam
depth and V olLidar
sem
are
fed to the transformer module, which stacks the N× blocks
to obtain a set of refined T-Primitives. Lastly, the refined set
of T-Primitives performs a splatting operation, resulting in the
final 3D semantic occupancy grid.
B. Image 2D Encoder and Decoder for Surround-View Images
The purpose of the image encoder is to capture both the
spatial and semantic features of the surround-view images.
These features serve as the foundation for the subsequent
feature fusion procedure, and lastly, the task of 3D semantic
occupancy prediction. In our approach, we first use a 2D back-
bone network (e.g. ResNet50 [19]) to extract visual features
at multiple scales V Cam
Encode =
n
V l
n
	N
n=1 ∈RCl×Hl×Wl
oL=3
l=1 .
Subsequently, these features are fused using feature-pyramid
networks (FPN), resulting in visual features V Cam
=
n
V l
n
	N
n=1 ∈RC×Hl×Wl
oL=3
l=1
that have the same feature
channel dimmension and resolutions that are 1
8,
1
16, and
1
32
of the input image resolution, respectively. The deeper visual
feature, with a smaller resolution, contains more semantic

<!-- page 4 -->
4
Skeleton 
Merge
Transformer
EarlyFusion
Camera FLow
Lidar FLow
Feature 
Fusion FLow
MGCAFusion
3D Deform-Atten
FFN & Add & Norm
FFN & Add & Norm
Refine Module
Sparse 
Self-Atten
×N Blocks
Transformer Inner Structure
Outer Product
Feature Add
Surround-View Images
2D-Encoder
Lidar Point Cloud
Lidar-Cylindrical 
Partition
3D-Encoder
Lidar Feature 
Semantic-Aware 
Lidar Feature 
T-Primitives
Multi-Scale Lidar Sparse 
Depth-Maps
Multi-Scale Cam Dense 
Depth-Maps
Multi-Scale-Fused Dense 
Depth-Maps
2D-Decoder Multi-Scale Visual Feats
Primitive Anchors
DepthNet
Depth-Aware Visual 
Feats
Cam-Cylindrical 
Partition
T-Splatting
Lifted Depth-Aware Visual Feats
Refined-T-Primitives
3D Semantic Occupancy
Visual Feats 
Sampling
Fig. 2: Overall architecture of TFusionOcc. The pipeline comprises two different modality branches and a multi-stage feature fusion
branch. The camera branch extracts multi-scale visual features and predicts a pseudo 3D point cloud from surround-view images. The
pseudo 3D point cloud is further projected and cylindrically partitioned, resulting in camera-based, multi-scale, dense depth maps and a
voxel volume defined under cylindrical coordinates. The lidar branch applies a cylindrical partition followed by a 3D encoder to extract the
lidar feature. Meanwhile, the lidar point cloud is also projected to generate lidar-based, multi-scale, sparse depth maps. The feature fusion
branch adopts a multi-stage fusion strategy to merge all outputs from the two-modality branch and leverages a proposed transformer to refine
the T-primitives property through fused features.
information. In contrast, the relatively shallower visual feature,
with larger resolutions, provides richer spatial details.
C. Camera-based Pseudo 3D Point Cloud and Multi-Scale
Cam Dense Depth Maps Generation
MLP
D
D
D Down Sampling
⅛ Scale Visual Features
Multi-Scale Cam Dense Depth Maps
1/8 Scale Depth Map
 1/16  Scale Depth Map
 1/32  Scale Depth Map
Image-based Point Cloud
Pixel-Ray
Probability
Fig. 3: Inner Structure of DepthNet. The 1/8-scale visual features
are first used to generate a 1/8-scale depth map. Then, bilinear
interpolation is leveraged to generate 1/16- and 1/32-scale depth
maps. Meanwhile, the image-based pseudo-point cloud is generated
solely from a 1/8-scale depth map.
The pseudo 3D point cloud Ptcam and camera-based multi-
scale dense depth maps V cam
depth are generated through the
DepthNet module as shown in Figure 3. The DepthNet module
takes V Cam
Encode as input and leverages a SECONDFPN [20] to
generate a single-scale visual feature V cam
sec
= {Vn}N
n=1 ∈
RC×Hl×Wl, whose resolution is
1
8 of the input image. The
V cam
sec
is further passed to an MLP to estimate the ray depth
per-pixel, resulting in a single-scale depth map Vdepth n ∈
RN×D×Hl×Wl. Bilinear interpolation with a 0.5 downsam-
pling factor is applied iteratively to Vdepth n, resulting in
two additional depth maps that have
1
16 and
1
32 resolution of
the input image. Instead of performing depth estimation on
each scale visual feature, our approach saves a certain amount
of computation overhead while still maintaining a relatively
high depth estimation precision. The predicted depth map
and the two extra downsampled depth maps form V cam
depth =
n
V l
depth n
oN
n=1 ∈RD×Hl×Wl
L=3
l=1
. Meanwhile, the pseudo
3D point cloud Ptcam is acquired only from 1
8 scale depth map
Vdepth n. Each pixel ray has a set of depth bins separated by
0.5m intervals. We followed the approach proposed in [15]
to use 3D semantic occupancy annotations to supervise depth
estimation. For each pixel ray, we take the highest probability
depth bins of K to form Ptcam.
D. Skeleton Merge Module
The primitive anchor initialization process is shown in
Figure 4. Given the pseudo 3D point cloud derived from
the camera Ptcam and the real 3D point cloud of the lidar

<!-- page 5 -->
5
Cam-Cylindrical 
Partition
Lidar-Cylindrical 
Partition
Overlap Removal
Reference
Range Filtering
Primitive Anchors
Main Skeleton
Augment
Scatter Sampling
Scatter Sampling
Reference
Lidar Point Cloud
Camera Point Cloud
Fig. 4: Skeleton Merge Module. The upper lidar-branch serves as
the main skeleton to provide a foundation structure for the 3D scene,
and the bottom camera-branch serves as an augmentation based on the
main skeleton to provide more detailed local structure to compensate
for the fine-grained geometry of the main skeleton.
Ptlidar, we first apply the cylindrical partition to both point
clouds, resulting in two volumes of voxels V olCam
Cylin and
V olLidar
Cylin under cylindrical coordinates. We then predefine the
total primitive anchors of M + N, where the anchors M are
sampled from V olLidar
Cylin and the anchors N are sampled from
V olCam
Cylin, with a ratio of M : N = 3 : 1. Following [15],
[21], we adopt the farthest point sampling strategy to obtain M
Lidar anchors from V olLidar
Cylin, which act as the main structural
skeleton. Using these Lidar anchors as reference, any voxel
in V olCam
Cylin that spatially overlaps with the LiDAR anchors is
removed. In addition, we impose a radius constraint r = 5m to
perform range filtering. Any voxel in V olCam
Cylin whose closest
Lidar anchor is farther away than r is also discarded. Then, N
Camera anchors are obtained using the farthest point sampling
strategy on the remaining voxels in V olCam
Cylin. Based on the
aforementioned filtering and sampling strategy, we acquire a
final set of M + N anchors whose locations will be used to
initialize a set of T-primitives.
The skeleton merge module takes advantage of the lidar
strength in preserving the fine-grained 3D geometry of the
scene and takes advantage of its V olLidar
Cylin to extract the main
skeleton. Meanwhile, the module also leverages the camera’s
strength in handling occlusion scenes and uses its V olCam
Cylin to
extract the augmented skeleton surrounding the main skeleton.
The fused skeleton preserves the finer-grained 3D geometry of
the scene by combining the advantages of both modalities.
E. Multi-scale-Fused Dense Depth-Maps Generation
Given multi-scale camera dense depth maps V cam
depth and
multi-scale lidar sparse depth maps V lidar
depth, we first go through
per-pixel level value addition followed by a value clamp,
resulting in the final multi-scale fused dense depth maps as
shown in Figure 5 upper. The detail of the addition of the
per-pixel ray value is shown in Figure 5 bottom. For the
camera side, each pixel ray has D number of depth bins, and
the interval between each depth bin is 0.5m. The DepthNet
predicts the probability that each depth bin being occupied
with a value in the range of [0, 1]. On the lidar side, each
pixel ray also has D depth bins, but only one depth bin has
Value Clamp
Multi-Scale Cam Dense Depth Maps
Multi-Scale Lidar Sparse Depth Maps
Multi-Scale Fused Dense Depth Maps
Pixel-Ray
Probability
Pixel-Ray
Probability
Pixel-Ray
Probability
Value Clamp
Pixel-Ray
Probability
Cam Pixel Ray
Lidar Pixel Ray
Fused Pixel Ray
Clampled Pixel Ray
Depth Map Perspective
Per-Pixel Ray Perspective
Fig. 5: Multi-Scale Fused Dense Depth Maps Generation. The
upper part exhibits general multi-modality depth map fusion from a
depth map perspective, and the bottom part demonstrates the detailed
multi-modality depth map fusion from a per-pixel ray perspective.
been assigned a probability value of 1.0, while all other depth
bins are assigned a probability value of 0. When adding the
camera pixel ray to the lidar pixel ray, the resulting fused pixel
ray may have some depth-bin probability values exceeding 1.0;
therefore, these values need to be clamped within the range
[0, 1].
This multi-modality depth map fusion, on the one hand,
solves the sparsity problem of lidar-based depth maps, and
on the other hand, mitigates the inaccurate depth estimation
problem of camera-based depth maps, resulting in denser and
highly precise fused depth maps.
F. Early-Fusion Module
Lidar-Cylindrical Partition
Coordinate 
Conversion
Lidar Feat Per Anchor
Projection
MLP
MLP
C
1/8 Visual Feature
1/16 Visual Feature
1/32 Visual Feature
W1
W2
W3
Weighted-Summed 
Visual Feature
Semantic-Aware 
Lidar Feature
C
Element-wise Multiplication
Feature Addition
Feature Channel Concatenation
Fig. 6: Early-Fusion Module. Each occupied voxel centre in
V olLidar
Cylin serves as an anchor and is projected onto multi-scale visual
features to aggregate semantic information, yielding a semantic-aware
lidar feature.
The detailed inner structure of the Early-Fusion module is
shown in Figure 6. Given the sparse volume of the voxel
V olLidar
Cylin in cylindrical coordinates, we leverage each occupied
voxel center as an anchor and convert each anchor position
from cylindrical coordinates to Cartesian coordinates. Then,
each anchor is projected onto multi-scale visual features V Cam
to aggregate semantic information at three scales. The aggre-
gated three-scale visual features are weighted, summed, and
concatenated back to the corresponding voxel feature, then fed

<!-- page 6 -->
6
through an MLP layer to perform the feature fusion, resulting
in the last semantic-aware lidar feature V olLidar
sem .
G. T-Primitive Representation
Following the GaussianFormer [14], [15], [17] series, we
propose the probabilistic Student’s T-distribution superposition
as an efficient and effective 3D scene representation. Specifi-
cally, we propose three types of primitives that are enhanced
by the inner kernel of T-distribution or T-distribution-like. The
first is the Student-T-distribution-based primitive, which has
the property as follows:
P T
i (mi, si, ri, αi, ci)
(3)
where mi, si, ri, αi and ci refer to the mean, scale value
for the x, y, z axes, quaternion vector, opacity and semantic
vector of the i-th primitive, respectively. The second is the
Student-T-distribution-based superquadric primitive, which has
the property as follows:
P T
SQi(mi, si, ri, αi, ϵ1 i, ϵ2 i, ci)
(4)
where ϵ1 and ϵ2 are the shape exponents of the superquadric.
The third is the Student-T-distribution-based superquadric with
inverse-warp primitive that has the property as follows:
P T
SQIW i(mi, si, ri, αi, ϵ1 i, ϵ2 i, wi, ci)
(5)
where ωi is the learnable weighting scaler valued between -1
and 1.
Similarly to [15], we decompose the 3D modeling target into
geometry and semantics predictions, and adopt the multiplica-
tion theorem of probability and the T-mixture model to address
them probabilistically, respectively. What is more, compared to
the Gaussian mixture model, the T-mixture model significantly
improves robustness against outliers.
1) Student-T-Distribution Based Geometry Prediction: To
restrict T-Primitives to represent only occupied regions for
geometry prediction, we interpret the T-distribution primitives
P T =

pT
i
	K
i=1 as the probability of their surrounding space
being occupied. In detail, we assign a probability value of
100% to the centers of the T-distribution, which decays with
respect to the distance from the centers m:
α
 x; pT
i

=

1 + 1
ν (x −m)T Σ−1 (x −m)
−ν+3
2
(6)
where ν refers to the degree of freedom, it is obtained
by calculating how many voxels are intercepted by each T-
primitive minus 1, and α
 x; pT
i

denotes the probability of
the point x being occupied induced by the i-th T-primitive.
Equation 6 assigns a high probability of occupancy when
the point x is close to the center m of the i-th T-primitive,
which prevents any T-primitive from describing an empty area.
To further derive the overall probability of occupancy, we
assume that the probabilities of a point being occupied by
different T-primitives are mutually independent, and thus we
can aggregate them according to the multiplication theorem of
probability:
α (x) = 1 −
K
Y
i=1
 1 −α
 x; pT
i

(7)
where α(x) represents the overall probability of occupancy at
point x.
2) T-Mixture Model Based Semantic Prediction: In addition
to the Student-T-distribution-based Geometry modeling, we
also need to perform the Student-T-distribution-based seman-
tics prediction. Following [15], we interpret the set of T-
primitives as a T-mixture model, where the semantics pre-
diction could be formulated as calculating the expectation of
semantics given the probabilistic T-mixture model.
To be specific, we take the original opacity properties α
as the prior distribution of the T-primitives, which is normal-
ized by l1. Then, the T-primitive probabilistic distribution is
adopted as a conditional probability. In addition, we normalize
each T-primitive’s original semantics properties ci with a soft-
max activation function, yielding ˜ci to ensure the boundness
of predicted semantics. Lastly, the expectation calculation of
the T-mixture model can be formulated as:
e
 x; P T 
=
K
X
i=1
p
 P T
i |x

˜ci =
PK
i=1 p
 x|P T
i

ai˜ci
PK
j=1 p
 x|P T
j

aj
(8)
where p
 P T
i |x

, p
 x|P T
i

and ˜ci denote the posterior proba-
bility of point x belonging to the i-th T-primitive distribution,
the conditional probability of point x given the i-th T-primitive
distribution, and the softmax-normalized semantic properties,
respectively.
Lastly, based on the geometry and semantic predictions,
we combine the two prediction results together, resulting in
the final 3D semantic occupancy prediction. The combination
process is formulated as follows.
Occ =

1 −α (x) ; α (x) · e
 x; P T 
(9)
where 1 −α (x) refers to the probability of the empty class
and α (x) · e
 x; P T 
refers to the final semantic probabilities.
The α (x) is used to weight the initial semantic prediction.
3) Student-T-Distribution Based Primitive: The straightfor-
ward T-distribution primitive p
 x|P T
i

can be formulated as:
p
 x|P T
i

=
Γ
  ν+3
2

Γ
  ν
2

· (νπ)3/2 · |Σ|1/2 ·

1 + 1
ν (x −m)T Σ−1 (x −m)
−ν+3
2
(10)
where ν refers to the degree of freedom, the same as in
Equation 6, it is obtained by calculating how many voxels
are intercepted by each T-primitive minus 1. Γ refers to
the gamma function, and performs the Γ (n) = (n −1)!
operation. Compared to Gaussian-primitive, this T-primitive
is highly tailed in its probability distribution, leading to better
robustness against outliers.
4) Student-T-Distribution Based SuperQuadric Primitive:
Following the quadricformer [17], we use the T-probabilistic
modeling mechanism to convert superquadrics into occu-
pancy probabilities. We first transform the 3D point position
(X, Y, Z), which is occupied by the T-Superquadric primitive,
into its local coordinate system as:


XS
YS
ZS

= R ×


X −mx
Y −my
Z −mz


(11)

<!-- page 7 -->
7
where (XS, YS, ZS) refers to the local coordinate under the
T-Superquadric primitive, and R is the rotation matrix of each
T-Superquadric primitive. Then, the occupancy probability of
the 3D point (X, Y, Z) associated with each T-Superquadric
primitive can be computed as:
f (XS, YS, ZS) =
 XS
Sx
 2
ϵ2
+
YS
Sy
 2
ϵ2
! ϵ2
ϵ1
+
ZS
Sz
 2
ϵ1
(12)
p
 x|P T
SQi

=

1 + 1
ν · f (XS, YS, ZS)
−ν+3
2
(13)
where (SX, SY , SZ) refers to the scale parameters along the
X, Y, Z axes, ϵ1 and ϵ2 are the shape exponents of the
superquadric.
5) Student-T-Distribution
Based
SuperQuadric
With
Inverse-Warp Primitive: So far, thanks to the T-distribution
kernel enhancement, the T-Superquadric primitive is robust to
outliers. Nevertheless, the T-Superquadric primitive has a very
limited shape variety. It can only exhibit convex, symmetric
shapes and cannot capture non-convex, asymmetric shapes,
which are common in the daily 3D driving scene. To solve this
limitation, we make the T-Superquadric primitive deformable
through inverse-warp enhancement. Specifically, we choose
24 basis field functions {Bi(u, v, w)}24
i=1, where u, v, w refers
to three inputs, respectively, to deform the local coordinates
(XS, YS, ZS) as follows:
u = XS
Sx
, v = YS
Sy
, w = ZS
Sz
(14)


˜XS
˜YS
˜ZS

=


XS
YS
ZS

−
24
X
i=1
ωi · Bi (u, v, w)
(15)
where ωi is the learnable weighting scaler valued between
-1 and 1. Then, the occupancy probability of the 3D point
associated with each T-Superquadric-Inverse-Warp primitive
can be calculated as follows:
f

˜XS, ˜YS, ˜ZS

=


 ˜XS
Sx
! 2
ϵ2
+
 ˜YS
Sy
! 2
ϵ2


ϵ2
ϵ1
+
 ˜ZS
Sz
! 2
ϵ1
(16)
p
 x|P T
SQIW i

=

1 + 1
ν · f

˜XS, ˜YS, ˜ZS
−ν+3
2
(17)
Each basis field function is listed in Table I. B1 to B9 provide
constants, linear, basic shears, and taper deformation. B10
to B18 enable full shear, twist, and bending deformation.
Moreover, B19 to B24 provide quadratics, radial bulge, and
smooth corner deformation.
H. Transformer
The general inner structure of the proposed transformer
is shown in Figure 2. It takes a set of T-primitives, multi-
scale lifted depth-aware visual features, and a semantic-aware
lidar feature as input, and proceeds through a fused depth-
map-guided 3D deformable attention module, a multi-gate
cross-attention fusion module, two MLP-based feed-forward
Basis Field
u
v
w
B1
1.0
0.0
0.0
B2
0.0
1.0
0.0
B3
0.0
0.0
1.0
B4
u
0.0
0.0
B5
0.0
v
0.0
B6
0.0
0.0
w
B7
v
0.0
0.0
B8
w
0.0
0.0
B9
0.0
w
0.0
B10
0.0
u
0.0
B11
0.0
0.0
u
B12
0.0
0.0
v
B13
−w ∗v
w ∗u
0.0
B14
0.0
−u ∗w
u ∗v
B15
v ∗w
0.0
−v ∗u
B16
w2
0.0
0.0
B17
0.0
w2
0.0
B18
0.0
0.0
u2 + v2
B19
u2
0.0
0.0
B20
0.0
v2
0.0
B21
0.0
0.0
w2
B22
(u2 + v2) ∗u
(u2 + v2) ∗v
0.0
B23
u ∗v
u ∗v
0.0
B24
u ∗v2
u2 ∗v
0.0
TABLE I: The 24 basis field functions make the T-Superquadric de-
formable. Those functions enable full shears, twists, bends, quadrat-
ics, radial bulge and smooth corners.
modules, a 3D sparse convolution-based self-attention mod-
ule, and a T-primitive properties refine module iteratively N
times, resulting in the last refined T-primitives. The following
sections will introduce the key implementation details of each
module.
u
v
d
…
Dui,vj,0
Vui,vj
Vui,vj
Vui,vj
Vui,vj
Vui,vj
Vui,vj
Vui,vj
Dui,vj,1 Dui,vj,2 Dui,vj,3 Dui,vj,4 Dui,vj,5 Dui,vj,6
×
×
×
×
×
×
×
Depth Prob
0.0
1.0
Lifted Depth-Aware 
Visual Features
T-Primitives
Fig. 7: 3D Deformable Attention For the same visual feature
at location (ui, vj), several different depth values ranging between
[0, 1.0] along the depth-axis are used to weight the same pixel feature,
resulting in the depth encoded visual feature to mitigate the projection
ambiguity problem stemming from the projection process.
1) Fused Depth Map based 3D Deformable Attention:
Taking multi-scale lifted depth-aware visual features and a set
of T-primitive as input, we use a 3D deformable attention oper-
ator, namely DFA3D [22], which performs feature aggregation
on lifted visual features to mitigate the 3D to 2D projection
ambiguity problem, to aggregate visual features for each T-
primitive. As shown in Figure 2 camera branch and Figure
7, the multi-scale lifted depth-aware visual features V olCam
depth
are obtained through outer product operation F Cam
depth ⊗V fuse
depth,

<!-- page 8 -->
8
this operation gives an accurate depth encoding for each
visual feature in the feature map. For each T-primitive feature
sampling pT
i , we first use the mean location mi of each pT
i
as the center, convert the center location from the cylindrical
coordinate to the cartesian coordinate, and feed its associate
query feature qi to an MLP layer, resulting in 13 offsets ∆mj
with respect to the center to form a set of 3D sampling points
msamp = {msamp
i
= mi + ∆mj | j = 1, ..., 13}. Then, we
leverage extrinsic and intrinsic matrices to project each 3D
sampling point msamp
i
onto the lifted depth-aware visual
features, resulting in the location of the sampling of the pixel
frame ¯msamp
i
= (ui, vi, di). Lastly, a 3D deformable attention
operator DFA3D takes ¯msamp
i
as a sampling location to per-
form cross-attention visual feature aggregation on V olCam
depth,
resulting in an updated query feature qupdated
i
. The overall
sampling process can be formulated as follows:
qupdated
i
=
13
X
i=1
DFA3D
 qi, ¯msamp
i
, V olCam
depth

(18)
After the 3D cross-attention process, qupdated
i
already aggre-
gates sufficient depth-aware visual features and is passed to
the MGCAFusion module for feature fusion. Meanwhile, the
set of 3D sampling points msamp for each T-primitive is also
passed on to the MGCAFusion module for semantic-aware
lidar feature aggregation.
2) Multi-Gate
Cross-Attention
Fusion
(MGCAFusion)
Module: Given a set of 3D sampling points msamp for each
T-primitive, we apply bilinear interpolation on V olLidar
sem
to
aggregate a semantic-aware lidar feature for each T-primitive.
The detailed implementation of the multi-gate cross attention
fusion module (MGCAFusion) is demonstrated in Figure
8. It consists of two central parts, the weighted summation
fusion part, as highlighted in the green rectangle on the
left of Figure 8 and the gated concatenation fusion part, as
highlighted in the pink rectangle on the right of Figure 8.
Both parts take the same inputs, but perform different feature
fusion strategies.
For the weighted summation fusion part, the semantic-aware
lidar feature and the depth-aware visual feature are summed in
elements first, then passed through an MLP layer followed by
a softmax layer, resulting in two weight values, WLidar and
WV is. The WLidar and WV is are multiplied again with the
corresponding semantic-aware lidar feature and depth-aware
visual feature, respectively, resulting in the weighted lidar
feature and the weighted visual feature. Lastly, the weighted
lidar feature and the weighted visual feature are summed and
passed through an MLP layer, yielding the summation fusion
feature.
For the gated concatenation fusion part, the semantic-aware
lidar feat and the depth-aware visual feat are passed through
an MLP layer, followed by a sigmoid function, respectively,
resulting in a visual gate vector GV is and a lidar gate vector
GLidar. GV is is element-wise multiplied with the semantic-
aware lidar feature, followed by a skip connection that results
in the gated lidar feature. GLidar is element-wise multiplied
with the depth-aware visual feature, followed by a skip con-
nection that results in the gated visual feature. The gated Lidar
and visual features are concatenated along the feature channel
dimension, followed by an MLP layer that reduces the fused
feature channel dimension, yielding the concatenation fusion
feature.
Lastly, the summation and concatenation fusion features are
combined via feature-channel concatenation, followed by an
MLP layer, yielding the final fused feature.
3) 3D Sparse Conv-based Self-Attention Module: The self-
attention module is demonstrated in Figure 9. Each T-primitive
center mi is treated as a 3D point, and we can acquire a
set of 3D points based on T-primitive. Then, this T-primitive-
based sparse 3D point cloud undergoes a coordinate trans-
formation from Cartesian to cylindrical coordinates, followed
by voxelization, yielding a sparse voxel volume in cylindrical
coordinates. Lastly, the 3D sparse convolution applies to this
sparse voxel volume to perform the self-attention operation.
Since our T-primitives are initialized in cylindrical coordinates,
we perform the self-attention operation in the same coordinates
to better preserve the scene’s fine-grained 3D geometry.
4) T-Primitive Property Refinement Module:
Since T-
primitive queries have aggregated sufficient semantic and 3D
geometry information in the previous 3D deformable attention,
self-attention, and MGCAFusion modules, we use each query
vector to update its associated T-primitive properties. Specif-
ically, for three types of T-primitives, we feed each type of
T-primitive query to an MLP layer to obtain the intermediate
properties, and we combine the intermediate properties with
the initial properties to achieve the refinement of the T-
primitive property. The property refinement details for three
types of T-primitives are formulated as:
MLP(QPi) =⇒( ˆmi, ˆsi, ˆri, ˆci)
MLP(QSQi) =⇒( ˆmi, ˆsi, ˆri, ˆϵ1i, ˆϵ2i, ˆci)
MLP(QSQIWi) =⇒( ˆmi, ˆsi, ˆri, ˆϵ1i, ˆϵ2i, ˆwi, ˆci)
P T
Newi = (m + ˆmi, ˆsi, ˆri, ˆci)
P T
NewSQi = (m + ˆmi, ˆsi, ˆri, ˆϵ1i, ˆϵ2i, ˆci)
P T
NewSQIWi = (m + ˆmi, ˆsi, ˆri, ˆϵ1i, ˆϵ2i, ˆwi, ˆci)
(19)
IV. EXPERIMENTAL RESULTS
A. Implementation Details
The TFusionOcc model takes six surround-view images and
10 lidar sweeps per data sample and leverages ResNet50-
DCN [19] as a 2D Backbone, initialized with weights from
FCOS3D [23]. The transformer layer is stacked four times
iteratively. AdamW optimizer is used, with an initial learning
rate of 2e-4 and a weight decay of 0.01. The learning rate is
decayed using a multistep scheduler. The predicted occupancy
has a resolution of 200 × 200 × 16 for full-scale evaluation
on nuScenes-Surroundocc, nuScenes-Occ3D and nuScenes-
C datasets. For data augmentation, we employ photometric
distortions and a grid mask to the input images during training.
Model training is conducted on two H100 GPUs with 94GB
of memory.
B. Loss Function
To train TFusionOcc, we take advantage of the Lov´asz-
Softmax loss [24] Llovasz and the binary cross-entropy loss

<!-- page 9 -->
9
MLP
Softmax
MLP
MLP
Sigmoid
MLP
Sigmoid
MLP
C
C
MLP
Semantic-Aware 
Lidar Feat
Depth-Aware 
Visual Feat
Semantic-Aware 
Lidar Feat
Depth-Aware 
Visual Feat
W-Lidar
W-Vis
Weighted Lidar 
Feat
Weighted Visual 
Feat
Weighted Summation Fusion Part
Gate-Vis
Gate-Lidar
Gated Lidar Feat
Gated Visual Feat
Gated Concatenation Fusion Part
Summation 
Fusion Feat
Concatenation 
Fusion Feat
Fused Feature
Fig. 8: Multi-Gate Cross Attention Fusion Module. The left part exhibits the multi-modality weighted summation fusion part, and the
right part exhibits the multi-modality gated concatenation fusion part. The outputs of the two major parts are further fused to produce the
final fused feature.
T-Primitives
Sparse 
Conv
Primitive-based 
3D Point Cloud
Cylindrical 
Voxelization
3D Sparse 
Convolution
Self-Attention Pipeline
Fig. 9: Sparse-Convolution-Based Self-Attention Pipeline. The
center of T-primitives is treated as a 3D point to form a primitive-
based 3D point cloud. We apply a cylindrical partition to the obtained
3D point cloud, followed by a 3D sparse convolution to realise self-
attention.
LBCE to optimize the 3D semantic occupancy task. For depth
supervision, a binary cross-entropy loss Ldepth
BCE is used to refine
the depth distribution feature. The final loss is formulated as
follows.
LT otal = Llovasz + λ · LBCE + Ldepth
BCE
(20)
where λ balance the loss weight, in our case, we set λ = 10.
The final loss is applied only to the output of the last iterated
transformer.
C. Dataset
The public nuScenes dataset [10], specifically designed
for autonomous driving purposes, serves as the primary data
source for our experiments. Furthermore, nuScenes-C [11],
which not only augments the existing nuScenes dataset with
additional simulated challenging scenarios, such as snow,
fog, lowlight and brightness, etc, but also simulates several
engineering failure cases, such as frame lost and camera crash,
etc, is employed as a supplementary dataset to further evaluate
the robustness of the proposed algorithm. Each corruption
scenario in nuScenes-C has three severities, ranked easy, mid,
and hard, respectively.
To perform the 3D semantic occupancy prediction task, we
use dense labels obtained from SurroundOcc [25] and Occ3D
[26], respectively. Since the test set lacks semantic labels, we
train our model on the training set and evaluate its performance
using the validation set. For annotations from SurroundOcc,
we set the range of the X and Y axes to [-50, 50] meters
and the Z axis to [-5, 3] meters under lidar coordinates. For
annotations from Occ3D, we set the range of the X and Y
axes to [-40, 40] meters and the Z axis to [-1, 5.4] meters
under the ego coordinates. The input images have a resolution
of 1600 × 900 pixels, while the final output of the semantic
occupancy prediction branch is represented with a resolution
of 200×200×16. The annotations from SurroundOcc contain a
total of 17 semantic classes, while the annotations from Occ3D
comprise 18 semantic classes. Label 0 refers to free voxels in
both annotations.
Additionally, following the methodology proposed in [1],
we conduct an in-depth analysis of our model performance in
challenging scenarios, specifically in rainy and night condi-
tions. This evaluation is carried out using the annotation file
provided by [1].
D. Performance Evaluate Metrics
To assess the performance of various SOTA algorithms
and compare them with our approach in the 3D semantic
occupancy prediction task, we utilize the intersection over
union (IoU) to evaluate each semantic class. Moreover, we
employ the mean IoU over all semantic classes (mIoU) as a
comprehensive evaluation metric:
IoU =
TP
TP + FP + FN
(21)
and
mIoU =
1
Cls
Cls
X
i=1
TPi
TPi + FPi + FNi
(22)
where TP, FP, and FN represent the counts of true pos-
itives, false positives, and false negatives in our predictions,
respectively, while Cls denotes the total class number.
E. Model Performance Analysis
To demonstrate the performance of our proposed TFu-
sionOcc, we evaluated it on the Surroundocc-nuScenes and

<!-- page 10 -->
10
Occ3D-nuScenes benchmarks. For each benchmark, we eval-
uate our model across three primitive settings, including T-
P (general T-primitive), T-SQ (T-superquadric), and T-SQ-IW
(deformable T-superquadric or T-superquadric with inverse-
warp), with 12800 and 25600 total primitives, respectively.
The results of the experiment on the surroundocc-nuScenes
and Occ3D-nuScenes benchmarks are shown in Table II and
Table III, respectively.
In
both
benchmarks,
TFusionOcc
(T-SQ-IW)
settings
achieve the best and second-best performance, demonstrating
the importance of the primitives’ shape variety and robustness.
Specifically, our proposed method excels at detecting back-
ground objects, including drivable surfaces, other flat areas,
sidewalks, terrain, man-made structures, and vegetation. In
addition, our model performs significantly well on relatively
small foreground objects, such as barriers, bicycles, cars,
motorcycles, pedestrians, and traffic cones.
Our method, which only adopts a single 3D supervision sig-
nal, achieved par-performance in comparison with algorithms
that leverage multi-task learning to receive dual supervision
signals, such as DAOcc [5] in the Surroundocc-nuScenes
benchmark that incorporates a 3D object detection head and
a 3D semantic occupancy prediction head during training,
and SDGOcc [6] in the Occ3D-nuScenes benchmark, which
incorporates a 2D semantic segmentation head and a 3D
semantic occupancy prediction head during training.
F. Challenging Scenarios Performance Analysis
To further examine the performance and robustness of the
proposed algorithm, we follow [1] to evaluate TFusionOcc
with different T-primitive representations under challenging
rainy and night-time scenarios and the experiment results are
shown in Table IV and Table V, respectively.
In the rainy scenario, raindrops on the camera lens cause
focus blur and degrade the quality of semantic information
provided by the camera branch. Furthermore, laser beam
reflections during rainy conditions degraded the quality of
the geometry information provided by the lidar branch. As a
result, all algorithms in the benchmark exhibit varying degrees
of performance degradation. However, our proposed method
benefits from a multi-stage feature fusion design and the flex-
ible T-Primitive representation, exhibits the least performance
degradation in the benchmark, demonstrating high robustness
against challenging rainy conditions.
In the nighttime scenario, due to the nature of the camera
sensor, which is sensitive to the ambient lightning conditions,
the dim lighting condition during the nighttime leads to
semantic information loss in the camera branch. Meanwhile,
the depth-estimation accuracy from the camera branch drops.
As a result, all algorithms in the benchmark experience severe
performance degradation around 1.56% to 11.10% under the
IoU metric and 8.89% to 12.50% under the mIoU metric.
Among all performance degradation algorithms, our method
still achieves SOTA performance, demonstrating superior fea-
ture fusion and information filtering capabilities.
G. Performance Analysis On nuScenes-C Dataset
To further assess the robustness of the proposed algo-
rithm in extreme conditions, such as snow, fog, and sensor
malfunctions, we use the nuScenes-C dataset and evaluate
TFusionOcc at different levels of corruption and severity. The
results of the experiment for several camera corruption settings
are shown in Tables VI, VII, and X, and the results for
several lidar corruption settings are shown in Tables VIII,
IX, and XI. Note that since the nuScenes-C dataset only
provides single-lidar-sweep corruption data, and our model
leverages 10 lidar sweeps. Hence, when evaluating our model
across different lidar corruption settings, we use only a single
corrupted lidar sweep from the nuScenes-C dataset, and the
model performance under the clean setting differs from that in
Table II due to a much sparser lidar point cloud. In the camera
corruption scenario, our model shows strong robustness against
camera crashes, frame loss, brightness, and fog, as measured
by the mIoU metric. Nevertheless, the motion blur and snow
scenarios cause significant performance degradation in our
model. With around 5.04% to 8.43% and 4.91% to 6.64%
degradation under the IoU metric, 10.00% to 10.63% and
14.02% to 14.55% degradation under the mIoU metric for
the motion blur and snow corruption case, respectively. In the
lidar corruption scenario, our model is sensitive to lidar point
cloud density and quality degradation, demonstrating its firm
reliance on lidar point cloud quality.
H. Performance Analysis On Varying Distance
1) Performance Variation Trend At Different Distances.:
Following [1], we take the ego vehicle as the origin and R
as the radius. By adjusting the length of R, we evaluate our
proposed method against other SOTA algorithms at varying
perception distance radius [20m, 25m, 30m, 35m, 40m, 45m,
50m], including challenging rainy and night scenarios. The
trend of performance variation is shown in Figure 10.
The trend of performance variation across the whole vali-
dation set, as shown in Figures 10a and 10d, aligns with the
performance benchmark in Table II. In the rainy scenario, the
performance variation trend of our proposed method is similar
to the results of the whole validation set experiment, except
that the performance gap between different T-Primitive settings
further narrows down to 0.11% to 2.90% under the mIoU
metric and 0.44% to 1.74% under the IoU metric due to the
semantic and geometric information loss from both sensors.
In the nighttime scenario, we discover that our T-SQ-25600
setting outperforms the second-best T-SQ-IW-12800 setting
within a 35-meter perception range under the mIoU metric,
as shown in Figure 10c. This phenomenon may stem from the
fact that more primitives can better capture close-range details,
leading to better robustness against challenging dim-lighting
conditions.
2) Performance In Different Sectors Ranges: To thoroughly
study the effect of distance on the proposed method, we further
evaluate our TFusionOcc against other SOTA algorithms in
different sector ranges [0∼10m, 10∼20m, 20∼30m, 30∼40m,
40∼50m] and present the evaluation results in Figure 11.

<!-- page 11 -->
11
Method
2D-Backbone
Modality
IoU
mIoU
• barrier
• bicycle
• bus
• car
• const. veh.
• motorcycle
• pedestrian
• traffic cone
• trailer
• truck
• drive. surf.
• other flat
• sidewalk
• terrain
• manmade
• vegetation
MonoScene [27]
R101-DCN
C
23.96
7.31
4.03
0.35
8.00
8.04
2.90
0.28
1.16
0.67
4.01
4.35
27.72
5.20
15.13
11.29
9.03
14.86
Atlas [28]
-
C
28.66
15.00
10.64
5.68
19.66
24.94
8.90
8.84
6.47
3.28
10.42
16.21
34.86
15.46
21.89
20.95
11.21
20.54
BEVFormer [29]
R101-DCN
C
30.50
16.75
14.22
6.58
23.46
28.28
8.66
10.77
6.64
4.05
11.20
17.78
37.28
18.00
22.88
22.17
13.80
22.21
TPVFormer [30]
R101-DCN
C
30.86
17.10
15.96
5.31
23.86
27.32
9.79
8.74
7.09
5.20
10.97
19.22
38.87
21.25
24.26
23.15
11.73
20.81
C-CONet [31]
R101
C
26.10
18.40
18.60
10.00
26.40
27.40
8.60
15.70
13.30
9.70
10.90
20.20
33.00
20.70
21.40
21.80
14.70
21.30
InverseMatrixVT3D [32]
R101-DCN
C
30.03
18.88
18.39
12.46
26.30
29.11
11.00
15.74
14.78
11.38
13.31
21.61
36.30
19.97
21.26
20.43
11.49
18.47
OccFormer [33]
R101-DCN
C
31.39
19.03
18.65
10.41
23.92
30.29
10.31
14.19
13.59
10.13
12.49
20.77
38.78
19.79
24.19
22.21
13.48
21.35
FB-Occ [34]
R101
C
31.50
19.60
20.60
11.30
26.90
29.80
10.40
13.60
13.70
11.40
11.50
20.60
38.20
21.50
24.60
22.70
14.80
21.60
RenderOcc [35]
R101
C
29.20
19.00
19.70
11.20
28.10
28.20
9.80
14.70
11.80
11.90
13.10
20.10
33.20
21.30
22.60
22.30
15.30
20.90
GaussianFormer [14]
R101-DCN
C
29.83
19.10
19.52
11.26
26.11
29.78
10.47
13.83
12.58
8.67
12.74
21.57
39.63
23.28
24.46
22.99
9.59
19.12
Co-Occ [2]
R101
C
30.00
20.30
22.50
11.20
28.60
29.50
9.90
15.80
13.50
8.70
13.60
22.20
34.90
23.10
24.20
24.10
18.00
24.80
GaussianFormer2-256 [15]
R101-DCN
C
31.74
20.82
21.39
13.44
28.49
30.82
10.92
15.84
13.55
10.53
14.04
22.92
40.61
24.36
26.08
24.27
13.83
21.98
SurroundOcc [25]
R101-DCN
C
31.49
20.30
20.59
11.68
28.06
30.86
10.70
15.14
14.09
12.06
14.38
22.26
37.29
23.70
24.49
22.77
14.89
21.86
Inverse++ [13]
R101-DCN
C
31.73
20.91
20.90
13.27
28.40
31.37
11.90
17.76
15.39
13.49
13.32
23.19
39.37
22.85
25.27
23.68
13.43
20.98
LMSCNet [36]
-
L
36.60
14.90
13.10
4.50
14.70
22.10
12.60
4.20
7.20
7.10
12.20
11.50
26.30
14.30
21.10
15.20
18.50
34.20
L-CONet [31]
-
L
39.40
17.70
19.20
4.00
15.10
26.90
6.20
3.80
6.80
6.00
14.10
13.10
39.70
19.10
24.00
23.90
25.10
35.70
Co-Occ [2]
-
L
42.20
22.90
22.00
6.90
25.70
32.40
14.50
13.50
21.00
10.50
18.00
22.50
36.60
21.80
24.60
25.70
31.20
39.90
Co-Occ [2]
R101-DCN
C+L
41.10
27.10
28.10
16.10
34.00
37.20
17.00
21.60
20.80
15.90
21.90
28.70
42.30
25.40
29.10
28.60
28.20
38.00
GaussianFormer3D [8]
R101-DCN
C+L
43.30
27.10
26.90
15.80
32.70
36.10
18.60
21.70
24.10
13.00
21.30
29.00
40.60
23.70
27.30
28.20
32.60
42.30
OccFusion [1]
R101-DCN
C+R
33.97
20.73
20.46
13.98
27.99
31.52
13.68
18.45
15.79
13.05
13.94
23.84
37.85
19.60
22.41
21.20
16.16
21.81
OccFusion [1]
R101-DCN
C+L
43.53
27.55
25.15
19.87
34.75
36.21
20.03
23.11
25.25
17.50
22.70
30.06
39.47
23.26
25.68
27.57
29.54
40.60
OccFusion [1]
R101-DCN
C+L+R
43.96
28.27
28.32
20.95
35.06
36.84
20.33
26.22
25.86
19.17
21.27
30.64
40.08
23.75
25.56
27.63
29.82
40.82
GaussianFusionOcc [9]
R101-DCN
C+L
45.16
30.21
30.22
18.70
35.91
39.57
22.67
27.36
30.10
18.59
24.45
31.25
43.06
25.76
29.12
29.33
34.65
42.70
GaussianFusionOcc [9]
R101-DCN
C+L+R
45.20
30.37
30.43
18.54
36.23
39.66
22.57
27.35
30.30
19.14
24.56
31.95
42.60
25.82
29.48
29.70
34.78
42.95
OccCylindrical [4]
Res-50
C+L
44.94
28.67
26.17
22.12
31.47
36.84
17.95
27.77
29.85
23.90
20.64
28.27
43.00
23.14
27.99
27.81
30.81
40.95
DAOcc [5]
Res-50
C+L
45.00
30.50
30.80
19.50
34.00
38.80
25.00
27.70
29.90
22.50
23.20
31.60
41.00
25.90
29.40
29.90
35.20
43.50
TFusionOcc (T-SQ-12800)
Res-50
C+L
46.70
30.76
30.67
21.27
33.12
38.96
22.09
27.54
30.47
22.87
21.65
29.86
42.63
27.18
29.80
30.91
36.81
46.26
TFusionOcc (T-SQ-IW-12800)
Res-50
C+L
46.92
31.21
31.65
20.13
34.75
39.70
23.08
27.00
30.89
23.33
22.02
30.40
43.11
27.78
30.25
31.47
37.15
46.60
TFusionOcc (T-P-25600)
Res-50
C+L
45.49
30.04
30.15
19.58
34.20
38.64
22.10
26.60
29.85
22.24
22.44
29.65
41.08
25.19
28.43
29.71
35.77
45.01
TFusionOcc (T-SQ-25600)
Res-50
C+L
46.14
30.88
30.87
21.26
33.68
39.25
22.54
28.34
31.45
23.61
22.80
30.16
41.76
26.12
29.23
30.26
36.77
46.06
TFusionOcc (T-SQ-IW-25600)
Res-50
C+L
47.01
31.47
31.96
20.78
34.57
39.73
24.02
27.82
31.19
23.82
23.05
30.67
43.17
27.18
30.13
31.18
37.52
46.71
TABLE II: 3D semantic occupancy prediction results on SurroundOcc-nuScenes validation set. Our approach outperforms
other existing methods with the same input modality. For readers’ reference, the bottom of the table presents results from three
additional methods using different input modalities. All methods are trained with dense occupancy labels from SurroundOcc
[25]. Notion of modality: Camera (C), Lidar (L), Radar (R). The best and second-best performance is indicated in bold red
and bold blue, respectively.
Method
Mask
Backbone
Modality
mIoU
• others
• barrier
• bicycle
• bus
• car
• const. veh.
• motorcycle
• pedestrian
• traffic cone
• trailer
• truck
• drive. surf.
• other flat
• sidewalk
• terrain
• manmade
• vegetation
MonoScene [27]
%
Res-101
C
6.06
1.75
7.23
4.26
4.93
9.38
5.67
3.98
3.01
5.90
4.45
7.17
14.91
6.32
7.92
7.43
1.01
7.65
BEVDet [37]
%
Res-101
C
11.73
2.09
15.29
0.0
4.18
12.97
1.35
0.0
0.43
0.13
6.59
6.66
52.72
19.04
26.45
21.78
14.51
15.26
BEVFormer [29]
%
Res-101
C
23.67
5.03
38.79
9.98
34.41
41.09
13.24
16.50
18.15
17.83
18.66
27.70
48.95
27.73
29.08
25.38
15.41
14.46
BEVStereo [38]
%
Res-101
C
24.51
5.73
38.41
7.88
38.70
41.20
17.56
17.33
14.69
10.31
16.84
29.62
54.08
28.92
32.68
26.54
18.74
17.49
TPVFormer [30]
%
Res-101
C
28.34
6.67
39.20
14.24
41.54
46.98
19.21
22.64
17.87
14.54
30.20
35.51
56.18
33.65
35.69
31.61
19.97
16.12
OccFormer [33]
%
Res-101
C
21.93
5.94
30.29
12.32
34.40
39.17
14.44
16.45
17.22
9.27
13.90
26.36
50.99
30.96
34.66
22.73
6.76
6.97
RenderOcc [35]
%
Swin-B
C
26.11
4.84
31.72
10.72
27.67
26.45
13.87
18.20
17.67
17.84
21.19
23.25
63.20
36.42
46.21
44.26
19.58
20.72
CTF-Occ [26]
%
Res-101
C
28.53
8.09
39.33
20.56
38.29
42.24
16.93
24.52
22.72
21.05
22.98
31.11
53.33
33.84
37.98
33.23
20.79
18.00
Inverse++ [13]
%
Res-101
C
31.04
9.56
41.91
23.53
42.38
46.35
18.61
28.03
26.61
24.77
25.93
34.81
60.10
33.23
37.62
34.83
19.20
20.26
RadOcc [39]
!
Swin-B
C+L
49.38
10.90
58.20
25.00
57.90
62.90
34.00
33.50
50.10
32.10
48.90
52.10
82.90
42.70
55.30
58.30
68.60
66.00
OccFusion [1]
!
R101-DCN
C+L+R
46.67
12.40
50.30
31.50
57.60
58.80
34.00
41.00
47.20
29.70
42.00
48.00
78.40
35.70
47.30
52.70
63.50
63.30
OccFusion [1]
!
R101-DCN
C+L
48.74
12.40
51.80
33.00
54.60
57.70
34.00
43.00
48.40
35.50
41.20
48.60
83.00
44.70
57.10
60.00
62.50
61.30
GaussianFormer3D [8]
!
R101-DCN
C+L
46.40
9.80
50.00
31.30
54.00
59.40
28.10
36.20
46.20
26.70
40.20
49.70
79.10
37.30
49.00
55.00
69.10
67.60
RM2Occ [40]
!
R101-DCN
C+L
47.82
13.34
54.53
33.81
58.30
59.97
34.45
34.89
29.43
31.53
39.91
42.44
96.01
49.76
61.28
67.62
53.71
51.96
SDGOcc [6]
!
Res-50
C+L
51.66
13.20
57.80
24.30
60.30
64.30
36.20
39.40
52.40
35.80
50.90
53.70
84.60
47.50
58.00
61.60
70.70
67.70
EFFOcc [7]
!
Res-50
C+L
52.82
12.09
59.67
33.39
61.76
64.98
35.46
46.01
57.09
41.04
47.87
54.59
82.76
43.95
56.37
60.23
71.12
69.60
TFusionOcc (T-SQ-12800)
!
Res-50
C+L
52.34
14.82
58.85
39.48
55.10
62.24
35.42
44.19
58.61
47.17
44.92
48.43
82.54
42.87
54.65
59.52
71.05
69.96
TFusionOcc (T-SQ-IW-12800)
!
Res-50
C+L
52.97
14.90
60.59
40.93
56.48
63.68
35.25
45.96
58.37
48.62
44.54
49.84
83.42
43.85
55.09
59.67
69.86
69.51
TFusionOcc (T-P-25600)
!
Res-50
C+L
51.51
14.75
58.10
37.73
51.21
62.03
35.55
44.82
55.18
43.70
44.33
49.24
82.71
42.59
54.83
59.27
70.21
69.41
TFusionOcc (T-SQ-25600)
!
Res-50
C+L
51.70
14.89
59.19
37.64
53.96
62.40
33.88
44.74
58.41
47.17
41.08
47.23
82.46
43.25
54.80
59.04
70.34
68.39
TFusionOcc (T-SQ-IW-25600)
!
Res-50
C+L
53.35
14.66
60.25
40.80
58.39
63.67
34.81
45.97
59.04
48.72
45.32
50.43
83.21
44.70
55.55
60.22
71.41
69.75
TABLE III: 3D semantic occupancy prediction results on Occ3D-nuScenes benchmark. All methods are trained with dense
occupancy labels from Occ3D [26]. The camera visibility mask is not used during the training phase. Notion of modality:
Camera (C), Lidar (L), Radar (R). The best and second-best performance is indicated in bold red and bold blue, respectively.
In the whole validation set, the T-SQ-IW-25600 setting dom-
inates performance, followed by the T-SQ-IW-12800 setting
under the mIoU metric, as shown in Figure 11a. However,
once the perception sector range reaches 30∼40 m or 40∼50
m, the geometric prediction capability under the settings T-
SQ-IW-25600 and T-SQ-IW-12800 is outperformed by DAOcc
and OccCylindrical, as shown in Figure 11d. In the rainy
scenario, our T-SQ-IW-25600 and T-SQ-IW-12800 achieved
par performance with DAOcc in the 0∼10m sector range, but
in the remaining sector range, the vast majority of our settings
outperform DAOcc and OccCylindrical in mIoU metrics as
presented in Figure 11b. Regarding the geometry prediction
capability, our TFusionOcc, in all settings, outperforms DAOcc
and OccCylindrical for the sector ranges 0∼10m, 10∼20m
and 20∼30m. However, as the perception sector range reaches
30∼40 m and 40∼50 m, our proposed approach is slightly
left behind other SOTA algorithms, as shown in Figure 11e.
In the nighttime scenario, under the mIoU criterion, since
the perception sector range is very close, like 0∼10m and
10∼20m, our T-SQ-IW-25600 and T-SQ-25600 achieved the
best and second-best performance. As the perception sector
range reaches 30∼40m and 40∼50m, our TFusionOcc in all

<!-- page 12 -->
12
Method
Backbone
Modality
IoU
mIoU
• barrier
• bicycle
• bus
• car
• const. veh.
• motorcycle
• pedestrian
• traffic cone
• trailer
• truck
• drive. surf.
• other flat
• sidewalk
• terrain
• manmade
• vegetation
InverseMatrixVT3D [32]
R101-DCN
C
29.72 (-0.31%)
18.99 (+0.11%)
18.55
14.29
22.28
30.02
10.19
15.20
10.03
9.71
13.28
20.98
37.18
23.47
27.74
17.46
10.36
23.13
GaussianFormer [14]
R101-DCN
C
27.37 (-2.46%)
16.96 (-2.14%)
18.16
9.58
21.09
26.83
8.04
10.13
7.80
5.84
12.66
18.24
35.53
18.51
27.79
19.23
11.04
20.85
Co-Occ [2]
R101
C
28.90 (-1.10%)
19.70 (-0.60%)
22.10
17.60
26.30
30.80
10.90
9.90
8.20
9.70
11.40
19.30
39.00
22.20
32.60
23.00
11.50
21.30
GaussianFormer2-256 [15]
R101-DCN
C
31.14 (-0.60%)
20.36 (-0.46%)
19.84
13.52
26.89
31.65
10.82
15.16
9.04
8.41
13.72
21.84
40.51
24.57
32.21
20.65
12.64
24.33
SurroundOcc [25]
R101-DCN
C
30.57 (-0.92%)
19.85 (-0.45%)
21.40
12.75
25.49
31.31
11.39
12.65
8.94
9.48
14.51
21.52
35.34
25.32
29.89
18.37
14.44
24.78
Inverse++ [13]
R101-DCN
C
31.32 (-0.41%)
20.66 (-0.25%)
22.52
13.79
25.49
31.80
11.70
16.72
11.14
10.12
12.29
22.25
38.78
23.93
31.62
21.14
12.65
24.61
Co-Occ [2]
R101
C+L
40.30 (-0.80%)
26.60 (-0.50%)
26.60
19.10
37.60
37.20
15.90
20.30
16.30
12.30
23.30
27.00
41.00
22.80
35.20
24.60
27.80
39.30
OccFusion [1]
R101-DCN
C+L
42.67 (-0.86%)
26.68 (-0.87%)
20.91
18.39
35.26
36.19
17.69
19.05
19.40
17.08
23.86
28.86
38.28
26.37
31.44
21.35
29.48
43.22
OccFusion [1]
R101-DCN
C+L+R
42.67 (-1.29%)
27.39 (-0.88%)
27.82
21.10
36.00
37.10
17.23
21.67
20.34
17.46
20.93
28.57
38.99
24.72
31.96
21.26
29.64
43.53
GaussianFusionOcc [9]
R101-DCN
C+L
44.28 (-0.88%)
29.19 (-1.02%)
28.10
19.84
36.28
38.90
18.11
21.13
26.14
17.95
25.79
29.92
41.72
27.35
34.99
22.85
33.84
44.10
GaussianFusionOcc [9]
R101-DCN
C+L+R
44.36 (-0.84%)
29.86 (-0.51%)
28.40
19.88
38.87
39.33
20.61
26.05
25.66
17.97
26.07
31.02
41.70
24.94
35.27
24.08
33.90
44.00
OccCylindrical [4]
Res-50
C+L
44.08 (-0.86%)
28.07 (-0.60%)
24.50
22.66
33.47
36.79
16.96
23.63
23.79
23.49
22.70
28.14
41.67
22.32
33.55
21.44
30.58
43.45
DAOcc [5]
Res-50
C+L
44.51 (-0.49%)
29.65 (-0.85%)
28.50
19.63
38.01
38.98
21.42
20.19
24.63
21.52
23.60
31.12
40.56
26.68
35.05
24.94
34.29
45.28
TFusionOcc (T-SQ-12800)
Res-50
C+L
45.79 (-0.91%)
30.28 (-0.48%)
29.32
22.06
35.67
38.47
18.43
23.29
26.19
24.65
23.34
28.40
41.79
28.40
34.99
26.16
35.75
47.54
TFusionOcc (T-SQ-IW-12800)
Res-50
C+L
45.81 (-1.11%)
30.74 (-0.47%)
30.54
21.85
37.99
39.11
20.43
21.56
26.44
24.82
23.39
28.74
41.99
28.68
35.70
26.86
36.08
47.63
TFusionOcc (T-P-25600)
Res-50
C+L
44.77 (-0.72%)
29.22 (-0.82%)
27.86
21.11
36.24
38.33
17.79
18.33
25.25
23.53
25.50
28.59
40.27
24.95
33.54
24.82
34.90
46.57
TFusionOcc (T-SQ-25600)
Res-50
C+L
45.40 (-0.74%)
30.38 (-0.50%)
29.18
22.52
36.87
38.61
19.41
22.19
26.98
24.84
24.78
28.70
41.21
27.39
34.74
25.57
35.69
47.44
TFusionOcc (T-SQ-IW-25600)
Res-50
C+L
46.25 (-0.76%)
30.82 (-0.65%)
30.38
21.70
38.23
39.28
20.29
19.78
26.75
25.35
24.88
29.05
42.35
28.14
35.91
26.52
36.49
48.01
TABLE IV: 3D semantic occupancy prediction results on SurroundOcc-nuScenes validation rainy scenario subset. All
methods are trained with dense occupancy labels from SurroundOcc [25]. Notion of modality: Camera (C), Lidar (L), Radar
(R). The best and second-best performance is indicated in bold red and bold blue, respectively.
Method
Backbone
Modality
IoU
mIoU
• barrier
• bicycle
• bus
• car
• const. veh.
• motorcycle
• pedestrian
• traffic cone
• trailer
• truck
• drive. surf.
• other flat
• sidewalk
• terrain
• manmade
• vegetation
InverseMatrixVT3D [32]
R101-DCN
C
22.48 (-7.55%)
9.99 (-8.89%)
10.40
12.03
0.00
29.94
0.00
9.92
4.88
0.91
0.00
17.79
29.10
2.37
10.80
9.40
8.68
13.57
GaussianFormer [14]
R101-DCN
C
20.30 (-9.53%)
9.07 (-10.03%)
6.11
8.70
0.00
25.75
0.00
10.44
2.85
0.55
0.00
17.26
30.65
2.95
12.53
9.94
6.65
10.71
Co-Occ [2]
R101
C
18.90 (-11.10%)
9.40 (-10.90%)
4.50
9.30
0.00
29.50
0.00
8.40
3.50
0.00
0.00
15.10
29.40
0.60
12.40
11.50
10.70
15.50
GaussianFormer2-256 [15]
R101-DCN
C
21.19 (-10.55%)
10.14 (-10.68%)
5.25
9.29
0.00
29.33
0.00
13.65
5.80
0.90
0.00
20.22
31.80
1.94
14.83
10.48
5.96
12.72
SurroundOcc [25]
R101-DCN
C
24.38 (-7.11%)
10.80 (-9.50%)
10.55
14.60
0.00
31.05
0.00
8.26
5.37
0.58
0.00
18.75
30.72
2.74
12.39
11.53
10.52
15.77
Inverse++ [13]
R101-DCN
C
23.70 (-8.03%)
10.93 (-9.98%)
8.87
10.19
0.00
32.62
0.00
11.77
7.46
0.72
0.00
22.20
32.95
2.15
13.01
9.79
8.61
14.48
Co-Occ [2]
R101
C+L
35.60 (-5.50%)
14.60 (-12.50%)
8.40
16.40
0.00
37.20
0.00
13.80
10.90
0.40
0.00
24.30
36.40
2.20
14.60
17.00
19.90
31.70
OccFusion [1]
R101-DCN
C+L
40.87 (-2.66%)
15.87 (-11.68%)
13.28
17.53
0.00
36.42
0.00
16.16
11.37
1.42
0.00
25.71
32.64
0.63
16.06
20.87
24.52
37.27
OccFusion [1]
R101-DCN
C+L+R
41.01 (-2.95%)
16.61 (-11.66%)
15.70
16.26
0.00
38.09
0.00
22.18
13.24
0.08
0.00
25.92
33.15
1.57
16.08
21.09
24.51
37.83
GaussianFusionOcc [9]
R101-DCN
C+L
42.78 (-2.38%)
18.66 (-11.55%)
16.09
12.27
0.00
39.82
0.00
27.66
13.68
0.07
0.00
38.25
40.10
2.07
19.64
19.82
29.55
39.61
GaussianFusionOcc [9]
R101-DCN
C+L+R
42.51 (-2.69%)
18.45 (-11.92%)
12.15
11.47
0.00
39.77
0.00
29.58
15.27
0.04
0.00
37.08
37.13
2.58
19.94
20.63
29.42
40.10
OccCylindrical [4]
Res-50
C+L
43.38 (-1.56%)
17.79 (-10.88%)
16.19
10.04
0.00
37.84
0.00
25.63
13.28
0.23
0.00
32.93
39.59
3.89
17.89
21.61
26.99
38.47
DAOcc [5]
Res-50
C+L
42.85 (-2.15%)
18.53 (-11.97%)
17.18
15.07
0.00
39.05
0.00
25.77
14.70
0.07
0.00
32.51
33.95
4.02
19.78
22.58
30.91
40.87
TFusionOcc (T-SQ-12800)
Res-50
C+L
44.74 (-1.96%)
18.81 (-11.95%)
16.01
17.51
0.00
39.84
0.00
23.82
12.76
1.13
0.00
32.79
37.67
3.48
19.70
21.42
31.41
43.40
TFusionOcc (T-SQ-IW-12800)
Res-50
C+L
44.82 (-2.10%)
19.07 (-12.14%)
16.81
15.53
0.00
40.36
0.00
21.27
13.33
3.79
0.00
34.89
38.04
3.92
19.86
21.89
31.67
43.72
TFusionOcc (T-P-25600)
Res-50
C+L
43.25 (-2.24%)
17.83 (-12.21%)
14.12
10.10
0.00
38.45
0.00
21.99
13.10
1.16
0.00
32.05
34.60
4.61
18.91
23.14
30.76
42.35
TFusionOcc (T-SQ-25600)
Res-50
C+L
43.86 (-2.28%)
18.95 (-11.93%)
17.10
16.09
0.00
40.17
0.00
23.21
15.13
2.97
0.00
33.83
36.18
3.75
20.12
20.48
31.30
42.94
TFusionOcc (T-SQ-IW-25600)
Res-50
C+L
44.97 (-2.04%)
19.67 (-11.80%)
17.98
19.96
0.00
40.69
0.00
23.30
14.76
3.85
0.00
34.49
38.35
3.43
19.70
22.42
31.95
43.79
TABLE V: 3D semantic occupancy prediction results on SurroundOcc-nuScenes validation night scenario subset. All
methods are trained with dense occupancy labels from SurroundOcc [25]. Notion of modality: Camera (C), Lidar (L), Radar
(R). The best and second-best performance is indicated in bold red and bold blue, respectively.
Method
Metric
Clean
Cam Crash
Frame Lost
Color Quant
Motion Blur
Bright
Low Light
Fog
Snow
OccCylindrical
IoU
44.94
41.76 (-3.18%)
40.84 (-4.10%)
43.64 (-1.30%)
41.45 (-3.49%)
44.78 (-0.16%)
43.48 (-1.46%)
44.59 (-0.35%)
38.92 (-6.02%)
DAOcc
IoU
45.00
42.75 (-2.25%)
42.16 (-2.84%)
43.88 (-1.12%)
42.75 (-2.25%)
43.72 (-1.28%)
41.25 (-3.75%)
43.92 (-1.08%)
40.72 (-4.28%)
TFusionOcc (T-P-25600)
IoU
45.49
42.59 (-2.90%)
41.97 (-3.52%)
43.14 (-2.35%)
40.45 (-5.04%)
44.81 (-0.68%)
41.24 (-4.25%)
44.14 (-1.35%)
40.58 (-4.91%)
TFusionOcc (T-SQ-25600)
IoU
46.14
42.59 (-3.55%)
41.93 (-4.21%)
43.03 (-3.11%)
38.36 (-7.78%)
45.41 (-0.73%)
41.99 (-4.15%)
44.90 (-1.24%)
40.09 (-6.05%)
TFusionOcc (T-SQ-IW-25600)
IoU
47.01
42.61 (-4.40%)
41.77 (-5.24%)
43.97 (-3.04%)
38.58 (-8.43%)
46.31 (-0.70%)
42.19 (-4.82%)
45.95 (-1.06%)
40.37 (-6.64%)
TABLE VI: 3D semantic occupancy prediction IoU results on nuScenes-C Camera Corruption benchmark. All methods
are evaluated with different corruption type input images provided by nuScenes-C [11] and ground-truth occupancy labels from
SurroundOcc [25].
Method
Metric
Clean
Cam Crash
Frame Lost
Color Quant
Motion Blur
Bright
Low Light
Fog
Snow
SurroundOcc
mIoU
20.30
11.60 (-8.70%)
10.00 (-10.30%)
14.03 (-6.27%)
12.41 (-7.89%)
19.18 (-1.12%)
12.15 (-8.15%)
18.42 (-1.88%)
7.39 (-12.91%)
Inverse++
mIoU
20.91
11.76 (-9.15%)
10.22 (-10.69%)
14.88 (-6.03%)
12.84 (-8.07%)
19.71 (-1.20%)
12.05 (-8.86%)
18.86 (-2.05%)
7.62 (-13.29%)
OccCylindrical
mIoU
28.67
23.38 (-5.29%)
21.87 (-6.80%)
25.38 (-3.29%)
22.57 (-6.10%)
27.88 (-0.79%)
22.17 (-6.50%)
27.38 (-1.29%)
18.29 (-10.38%)
DAOcc
mIoU
30.50
22.35 (-8.15%)
20.78 (-9.72%)
27.46 (-3.04%)
26.25 (-4.25%)
26.78 (-3.72%)
21.15 (-9.35%)
26.87 (-3.63%)
19.42 (-11.08%)
TFusionOcc (T-P-25600)
mIoU
30.04
22.34 (-7.70%)
20.78 (-9.26%)
24.42 (-5.62%)
20.04 (-10.00%)
29.03 (-1.01%)
18.62 (-11.42%)
27.43 (-2.61%)
15.62 (-14.42%)
TFusionOcc (T-SQ-25600)
mIoU
30.88
23.50 (-7.38%)
22.04 (-8.84%)
25.65 (-5.23%)
20.25 (-10.63%)
29.92 (-0.96%)
20.98 (-9.90%)
28.85 (-2.03%)
16.33 (-14.55%)
TFusionOcc (T-SQ-IW-25600)
mIoU
31.47
24.39 (-7.08%)
23.03 (-8.44%)
26.14 (-5.33%)
21.06 (-10.41%)
30.58 (-0.89%)
21.67 (-9.80%)
29.69 (-1.78%)
17.45 (-14.02%)
TABLE VII: 3D semantic occupancy prediction mIoU results on nuScenes-C Camera Corruption benchmark. All methods
are evaluated with different corruption type input images provided by nuScenes-C [11] and ground-truth occupancy labels from
SurroundOcc [25].
Method
Metric
Clean
Beam Missing
Cross Sensor
Cross Talk
Incomplete Echo
Motion Blur
Wet Ground
Fog
Snow
OccCylindrical
IoU
35.89
30.73 (-5.16%)
25.01 (-10.88%)
33.29 (-2.60%)
35.37 (-0.52%)
32.44 (-3.45%)
35.04 (-0.85%)
25.79 (-10.10%)
34.00 (-1.89%)
DAOcc
IoU
42.39
38.17 (-4.22%)
33.79 (-8.60%)
33.68 (-8.71%)
42.01 (-0.38%)
38.05 (-4.34%)
41.67 (-0.72%)
31.47 (-10.92%)
37.01 (-5.38%)
TFusionOcc (T-P-25600)
IoU
40.29
32.75 (-7.54%)
25.65 (-14.64%)
36.25 (-4.04%)
39.67 (-0.62%)
34.79 (-5.50%)
38.17 (-2.12%)
26.44 (-13.85%)
36.88 (-3.41%)
TFusionOcc (T-SQ-25600)
IoU
40.53
33.00 (-7.53%)
25.89 (-14.64%)
36.60 (-3.93%)
39.90 (-0.63%)
34.67 (-5.86%)
38.74 (-1.79%)
26.54 (-13.99%)
36.88 (-3.65%)
TFusionOcc (T-SQ-IW-25600)
IoU
41.53
34.09 (-7.44%)
26.82 (-14.71%)
37.50 (-4.03%)
40.94 (-0.59%)
35.54 (-5.99%)
39.84 (-1.69%)
27.10 (-14.43%)
37.70 (-3.83%)
TABLE VIII: 3D semantic occupancy prediction IoU results on nuScenes-C Lidar Corruption benchmark. All methods
are evaluated with different corruption type input images provided by nuScenes-C [11] and ground-truth occupancy labels from
SurroundOcc [25].

<!-- page 13 -->
13
Method
Metric
Clean
Beam Missing
Cross Sensor
Cross Talk
Incomplete Echo
Motion Blur
Wet Ground
Fog
Snow
OccCylindrical
mIoU
24.06
19.83 (-4.23%)
16.20 (-7.86%)
22.60 (-1.46%)
20.93 (-3.13%)
20.85 (-3.21%)
23.31 (-0.75%)
17.32 (-6.74%)
22.38 (-1.68%)
DAOcc
mIoU
27.73
22.96 (-4.77%)
18.41 (-9.32%)
22.86 (-4.87%)
24.45 (-3.28%)
23.66 (-4.07%)
27.25 (-0.48%)
19.73 (-8.00%)
24.53 (-3.20%)
TFusionOcc (T-P-25600)
mIoU
26.13
20.49 (-5.64%)
15.38 (-10.75%)
23.76 (-2.37%)
22.69 (-3.44%)
21.39 (-4.74%)
25.05 (-1.08%)
17.44 (-8.69%)
23.15 (-2.98%)
TFusionOcc (T-SQ-25600)
mIoU
27.03
21.07 (-5.96%)
15.61 (-11.42%)
24.73 (-2.30%)
23.30 (-3.73%)
22.15 (-4.88%)
26.06 (-0.97%)
18.40 (-8.63%)
24.37 (-2.66%)
TFusionOcc (T-SQ-IW-25600)
mIoU
27.50
21.81 (-5.69%)
16.49 (-11.01%)
24.94 (-2.56%)
24.11 (-3.39%)
22.62 (-4.88%)
26.57 (-0.93%)
18.63 (-8.87%)
24.37 (-3.13%)
TABLE IX: 3D semantic occupancy prediction mIoU results on nuScenes-C Lidar Corruption benchmark. All methods
are evaluated with different corruption type input lidar point cloud provided by nuScenes-C [11] and ground-truth occupancy
labels from SurroundOcc [25].
Corruption
Severity
IoU
mIoU
• barrier
• bicycle
• bus
• car
• const. veh.
• motorcycle
• pedestrian
• traffic cone
• trailer
• truck
• drive. surf.
• other flat
• sidewalk
• terrain
• manmade
• vegetation
CameraCrash
Easy
43.16
26.44
23.34
16.62
28.80
35.79
19.34
21.30
28.47
19.01
18.46
24.14
36.89
19.19
25.07
27.42
35.32
43.94
CameraCrash
Mid
42.27
23.37
17.66
11.56
26.79
33.91
14.58
16.88
26.48
16.44
13.69
19.75
35.87
16.54
22.43
24.70
34.07
42.56
CameraCrash
Hard
42.39
23.35
20.30
11.23
26.54
34.40
15.27
13.93
25.80
17.87
11.15
21.39
35.24
14.59
22.44
25.18
34.52
43.77
FrameLost
Easy
44.33
27.42
25.40
16.12
30.44
36.89
19.15
21.10
28.59
20.72
18.97
26.33
38.65
21.48
26.20
28.11
35.81
44.84
FrameLost
Mid
41.33
22.19
15.55
10.88
24.67
33.26
13.95
14.55
25.62
15.66
11.70
19.94
33.63
13.21
21.59
24.17
33.98
42.73
FrameLost
Hard
39.66
19.48
9.91
9.09
22.02
31.26
11.74
11.63
23.98
12.70
8.77
15.30
30.99
9.11
18.80
21.75
33.03
41.52
ColorQuant
Easy
46.55
30.61
31.18
19.09
33.46
39.63
22.86
26.15
30.57
23.20
22.87
30.29
42.26
25.40
28.71
30.42
37.19
46.44
ColorQuant
Mid
44.19
26.96
25.94
14.03
30.21
38.26
19.43
18.13
29.03
22.00
18.56
26.63
38.54
19.68
24.39
26.26
35.72
44.54
ColorQuant
Hard
41.18
20.85
13.87
8.24
22.61
34.82
12.93
8.22
26.70
18.66
9.86
20.29
33.70
9.37
19.14
33.79
42.30
20.85
MotionBlur
Easy
41.43
26.89
28.45
15.39
31.42
37.33
20.59
19.98
29.74
23.25
19.18
25.99
36.16
17.03
22.63
25.86
34.73
42.48
MotionBlur
Mid
37.26
18.80
15.08
8.51
21.43
33.12
11.83
13.43
22.58
17.93
6.36
13.80
29.91
3.61
12.99
20.31
31.56
38.34
MotionBlur
Hard
37.06
17.48
12.18
7.54
19.65
32.08
9.80
12.42
20.60
16.37
4.50
12.63
29.47
2.63
11.26
19.46
31.17
37.93
Brightness
Easy
46.90
31.35
31.83
20.44
34.29
39.73
23.55
27.70
31.19
23.89
23.07
30.52
43.06
27.37
29.94
30.92
37.45
46.57
Brightness
Mid
46.45
30.67
31.12
19.86
32.69
39.37
22.29
26.38
31.08
23.73
22.70
29.40
42.35
27.07
28.96
30.20
37.26
46.31
Brightness
Hard
45.59
29.71
30.29
19.08
31.22
38.80
20.78
25.06
30.92
23.76
21.97
28.14
40.90
25.49
27.29
28.93
36.92
45.88
LowLight
Easy
43.86
25.38
19.52
14.11
30.88
37.84
16.55
20.19
28.43
17.38
13.94
25.64
37.39
15.11
23.22
25.82
35.57
44.48
LowLight
Mid
42.12
21.69
11.25
11.54
27.65
36.35
11.88
18.08
26.75
12.07
5.65
22.09
34.43
8.43
19.90
23.02
34.45
43.44
LowLight
Hard
40.60
17.93
3.52
6.49
23.37
34.05
7.87
15.49
23.81
5.33
1.57
17.02
32.01
3.56
16.69
20.34
33.18
42.59
Fog
Easy
46.37
30.57
30.82
19.55
34.30
39.40
21.91
25.99
30.65
23.35
22.12
30.06
42.13
26.77
28.91
29.86
37.10
46.27
Fog
Mid
46.03
29.89
30.01
18.93
33.72
39.29
20.89
25.12
30.42
22.91
21.21
29.32
41.37
25.36
27.89
28.93
36.87
46.07
Fog
Hard
45.44
28.61
29.00
17.43
32.02
38.90
18.92
24.15
30.04
22.41
19.61
27.90
40.02
22.29
25.69
27.37
36.47
45.58
Snow
Easy
41.26
20.46
12.34
8.18
24.01
34.48
14.53
13.38
26.04
20.12
6.13
19.37
32.52
6.58
12.73
21.68
33.02
42.31
Snow
Mid
39.61
15.81
7.13
4.17
16.34
30.79
7.63
11.86
22.00
16.87
4.93
9.97
27.63
2.30
7.69
15.13
30.04
38.46
Snow
Hard
40.23
16.08
5.29
4.19
16.00
31.17
8.68
11.28
22.39
16.59
4.58
10.08
29.80
2.64
9.17
14.27
31.40
39.78
TABLE X: The performance of TFusionOcc(T-SQ-IW-25600) on the nuScene-C [11] dataset under different camera
corruption and severity settings.
Corruption
Severity
IoU
mIoU
• barrier
• bicycle
• bus
• car
• const. veh.
• motorcycle
• pedestrian
• traffic cone
• trailer
• truck
• drive. surf.
• other flat
• sidewalk
• terrain
• manmade
• vegetation
BeamMissing
Light
38.03
24.83
25.39
12.35
30.38
33.20
17.80
20.84
23.97
16.98
19.35
26.19
37.83
21.87
23.66
24.77
26.92
35.79
BeamMissing
Moderate
34.28
21.94
22.45
9.84
28.26
29.26
15.39
17.79
20.83
12.97
17.61
23.96
35.44
19.69
20.99
22.06
23.07
31.41
BeamMissing
Heavy
29.97
18.67
18.85
6.93
25.51
24.95
13.02
13.23
15.80
11.40
15.58
21.18
32.63
17.35
18.16
18.45
19.03
26.71
CrossSensor
Light
31.71
20.08
19.55
7.95
27.85
27.18
14.63
13.75
17.86
11.99
17.34
23.06
34.50
19.05
19.57
19.78
20.08
27.19
CrossSensor
Moderate
26.52
16.07
15.50
5.00
23.42
21.71
12.15
7.42
13.75
9.30
14.33
19.55
31.52
15.85
16.60
16.43
14.97
19.66
CrossSensor
Heavy
22.23
13.32
11.50
3.77
21.31
16.56
6.96
9.04
10.43
7.01
13.48
15.76
28.99
14.64
14.00
13.66
11.28
14.64
CrossTalk
Light
39.77
26.33
28.50
14.08
31.14
35.44
18.51
21.00
25.73
17.29
19.97
27.30
39.52
24.07
26.08
27.46
29.48
35.63
CrossTalk
Moderate
37.47
24.93
27.82
12.62
30.10
34.11
16.63
19.01
24.07
15.91
19.14
26.14
38.85
23.78
25.07
26.82
28.03
30.82
CrossTalk
Heavy
35.26
23.57
26.75
11.55
29.19
32.80
15.39
16.61
22.60
14.60
18.47
25.09
38.09
23.34
23.41
25.86
26.44
26.93
IncompleteEcho
Light
41.18
25.74
28.97
12.08
28.39
30.45
15.70
19.21
27.68
19.35
17.63
24.15
39.80
23.88
26.37
27.75
30.51
39.94
IncompleteEcho
Moderate
41.01
24.66
28.96
10.54
25.72
26.81
13.15
17.27
27.67
19.26
15.40
21.58
39.74
23.90
26.36
27.73
30.52
39.93
IncompleteEcho
Heavy
40.62
21.93
28.92
8.66
17.25
16.89
7.41
13.52
27.62
19.08
8.97
14.64
39.57
23.92
26.34
27.71
30.53
39.91
MotionBlur
Light
38.10
24.89
25.61
12.99
30.89
33.74
19.39
20.91
24.15
14.99
19.40
26.77
36.30
21.25
23.10
24.59
26.09
38.00
MotionBlur
Moderate
35.44
22.56
22.40
11.01
29.50
31.09
18.24
18.10
20.87
11.53
18.13
25.43
33.88
18.98
20.68
22.26
22.74
36.10
MotionBlur
Heavy
33.07
20.42
19.24
9.15
28.01
28.37
17.26
15.57
17.81
8.80
16.49
23.64
32.07
17.44
18.77
20.20
19.85
34.03
WetGround
Light
40.53
26.99
28.86
15.18
31.66
36.33
19.80
23.18
27.60
19.65
20.73
28.03
37.07
21.62
23.82
27.84
30.50
39.99
WetGround
Moderate
39.84
26.57
28.86
15.12
31.65
36.23
19.77
22.70
27.53
19.86
20.71
27.91
35.13
19.21
21.98
28.01
30.50
39.98
WetGround
Heavy
39.14
26.14
28.80
14.79
31.57
36.14
19.74
22.69
27.47
19.87
20.68
27.85
33.08
17.10
20.01
27.99
30.49
39.98
Fog
Light
36.86
24.38
27.36
14.09
28.96
33.78
15.06
20.04
23.85
18.70
17.81
23.49
38.44
22.32
24.71
23.95
25.84
31.69
Fog
Moderate
28.39
20.05
22.93
12.52
26.29
29.36
12.23
17.41
19.34
16.90
11.32
16.80
34.83
20.18
21.50
19.90
16.92
22.37
Fog
Heavy
16.04
11.46
13.35
6.10
15.92
19.10
7.93
8.75
9.78
10.17
3.34
7.81
26.48
14.66
12.86
11.56
6.17
9.35
Snow
Light
39.38
24.78
28.29
14.52
29.78
34.99
14.71
19.91
23.43
17.00
18.80
25.34
39.33
23.29
25.23
26.10
27.60
28.23
Snow
Moderate
37.52
24.42
28.34
14.26
29.77
34.75
14.91
19.96
22.36
15.96
18.51
24.98
38.93
23.35
25.34
26.44
27.15
25.73
Snow
Heavy
36.20
23.91
28.11
13.47
29.74
34.20
15.34
18.98
21.07
13.80
18.17
25.16
38.42
23.08
25.28
26.47
25.68
25.65
TABLE XI: The performance of TFusionOcc(T-SQ-IW-25600) on the nuScene-C [11] dataset under different lidar
corruption and severity settings.

<!-- page 14 -->
14
20
25
30
35
40
45
50
Radius Range (M)
28
30
32
34
36
38
40
42
mIoU
mIoU Performance Variation Trend on Validation Set
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
OccFusion (C+L+R)
OccFusion (C+L)
Co-Occ (C+L)
(a)
20
25
30
35
40
45
50
Radius Range (M)
27.5
30.0
32.5
35.0
37.5
40.0
42.5
mIoU
mIoU Performance Variation Trend on Rainy Scenario Subset
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
OccFusion (C+L+R)
OccFusion (C+L)
Co-Occ (C+L)
(b)
20
25
30
35
40
45
50
Radius Range (M)
14
16
18
20
22
24
26
mIoU
mIoU Performance Variation Trend on Night Scenario Subset
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
OccFusion (C+L+R)
OccFusion (C+L)
Co-Occ (C+L)
(c)
20
25
30
35
40
45
50
Radius Range (M)
45
50
55
60
IoU
IoU Performance Variation Trend on Validation Set
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
OccFusion (C+L+R)
OccFusion (C+L)
Co-Occ (C+L)
(d)
20
25
30
35
40
45
50
Radius Range (M)
40
45
50
55
60
65
IoU
IoU Performance Variation Trend on Rainy Scenario Subset
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
OccFusion (C+L+R)
OccFusion (C+L)
Co-Occ (C+L)
(e)
20
25
30
35
40
45
50
Radius Range (M)
35
40
45
50
55
60
IoU
IoU Performance Variation Trend on Night Scenario Subset
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
OccFusion (C+L+R)
OccFusion (C+L)
Co-Occ (C+L)
(f)
Fig. 10: Performance variation trend for 3D semantic occupancy prediction task. (a) mIoU performance variation trend on the whole
SurroundOcc-nuScenes validation set, (b) IoU performance variation on the whole SurroundOcc-nuScenes validation set, (c) mIoU
performance variation trend on the SurroundOcc-nuScenes validation rainy scenario subset, (d) IoU performance variation on the SurroundOcc-
nuScenes validation rainy scenario subset, (e) mIoU performance variation on the SurroundOcc-nuScenes validation night scenario subset,
and (f) IoU performance variation on the SurroundOcc-nuScenes validation night scenario subset. Better viewed when zoomed in.
settings achieved par performance with DAOcc and OccCylin-
drical, as shown in Figure 11c. Regarding the geometry predic-
tion capability measured under the IoU criterion, our approach
in all settings outperforms the others in the sector ranges
0∼10m, 10∼20m and 20∼30m. Nevertheless, our approach
lags behind other SOTA algorithms when the perception sector
range reaches 30∼40m and 40∼50m as shown in Figure 11f.
I. Qualitative Study
1) Qualitative Results On SurroundOcc-nuScenes Dataset:
To qualitatively study the difference in each setting of the
proposed TFusionOcc, we demonstrate the visualization result
of each setting in daytime, rainy, and night-time scenarios
in Figure 12, with the main discrepancy area highlighted.
In the daytime scenarios as shown at the top of Figure 12,
in the orange square highlighted area, except for the T-P-
25600 setting, all other settings successfully detect those two
pedestrians. What is more, T-SQ-IW Primitive preserves better
the geometric contour shape of background buildings, such as
sharper building edges and more accurate building boundaries,
than T-P primitive and T-SQ primitive, demonstrating the
importance of the geometric shape flexibility of intermediate
primitives. In rainy scenarios, as shown in the middle of
Figure 12, in the dark red square highlighted area. The better
geometric shape representation capability of the primitives
could more accurately capture the geometric structure of the
object, leading to better detection performance; T-SQ-IW-
25600 achieved the best performance due to its best geometric
shape representation capability, and T-P-25600 the worst,
stemming from the ellipsoid geometric shape prior. Similarly,
in the nighttime scenarios, as shown in the bottom of Figure
12, in the square highlighted area. Even with severe occlusions
and a vehicle headlight glare pollution problem, we observed
that greater geometric shape flexibility of primitives leads to
better detection performance, which aligns with rainy and
daytime phenomena.
Furthermore, to demonstrate the effectiveness and robust-
ness of our method, we compare TFusionOcc in three different
T-Primitive settings, such as T-P-25600, T-SQ-25600 and T-
SQ-IW-25600, against other SOTA algorithms in challenging
daytime, rainy and nighttime scenarios, as shown in Figure 13,
with the primary difference area highlighted in each scenario.
In the daytime scenario, as shown at the top of Figure 13,
in the orange square highlighted area, benefit from multi-
task learning, which provides an additional supervision signal,
DAOcc successfully detects distant pedestrians and achieves
similar performance as TFusionOcc. However, in the dark-red-
square-highlighted area, DAOcc fails to detect the distant ve-
hicle, demonstrating the superior performance of our proposed
method. In the rainy scenario, as shown in the middle of Figure
13, in the purple square highlighted area, our proposed method
not only successfully detects all dynamic objects, but also
preserves their general contours well. In the highlighted dark-
blue-square area, our approach achieves par performance with
other SOTA algorithms for detecting challenging pedestrians.
In the nighttime scenario, as shown at the bottom of Figure 13,
our proposed method successfully detects very distant dynamic
objects in the highlighted dark-red-square area, demonstrating
its superior long-range dynamic object perception capability.

<!-- page 15 -->
15
0 ~ 10m
10 ~ 20m
20 ~ 30m
30 ~ 40m
40 ~ 50m
Sector Range (m)
0
10
20
30
40
mIoU
mIoU Performance on Validation Set (Whole)
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
(a)
0 ~ 10m
10 ~ 20m
20 ~ 30m
30 ~ 40m
40 ~ 50m
Sector Range (m)
0
10
20
30
40
mIoU
mIoU Performance on Validation Set (Rainy)
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
(b)
0 ~ 10m
10 ~ 20m
20 ~ 30m
30 ~ 40m
40 ~ 50m
Sector Range (m)
0
5
10
15
20
25
30
mIoU
mIoU Performance on Validation Set (Night)
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
(c)
0 ~ 10m
10 ~ 20m
20 ~ 30m
30 ~ 40m
40 ~ 50m
Sector Range (m)
0
10
20
30
40
50
60
70
IoU
IoU Performance on Validation Set (Whole)
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
(d)
0 ~ 10m
10 ~ 20m
20 ~ 30m
30 ~ 40m
40 ~ 50m
Sector Range (m)
0
10
20
30
40
50
60
70
IoU
IoU Performance on Validation Set (Rainy)
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
(e)
0 ~ 10m
10 ~ 20m
20 ~ 30m
30 ~ 40m
40 ~ 50m
Sector Range (m)
0
10
20
30
40
50
60
70
mIoU
mIoU Performance on Validation Set (Night)
T-SQ-IW-12800
T-SQ-12800
T-SQ-IW-25600
T-SQ-25600
T-P-25600
DAOcc
OccCylindrical
(f)
Fig. 11: Performance at different sector ranges for the 3D semantic occupancy prediction task. (a) mIoU performance at different sector
ranges on the whole SurroundOcc-nuScenes validation set, (b) IoU performance at different sector ranges on the whole SurroundOcc-
nuScenes validation set, (c) mIoU performance at different sector ranges on the SurroundOcc-nuScenes validation rainy scenario subset,
(d) IoU performance at different sector ranges on the SurroundOcc-nuScenes validation rainy scenario subset, (e) mIoU performance at
different sector ranges on the SurroundOcc-nuScenes validation night scenario subset, and (f) IoU performance at different sector ranges on
the SurroundOcc-nuScenes validation night scenario subset. Better viewed when zoomed in.
2) Qualitative Results On nuScenes-C Dataset: Since the
nuScenes dataset does not contain fog and snow scenarios,
which are common driving conditions in the real world, we
adopted the nuScenes-C dataset to evaluate our model across
three settings within the same scene under fog and snow
conditions, with the severity of each condition set to hard, and
compare the visualization results with other SOTA algorithms.
The visualization results are shown in Figure 14, with the main
area of the discrepancy highlighted in dark blue and dark red
squares.
Under foggy conditions, the surround-view images suffer
significant visibility degradation due to strong atmospheric
scattering, leading to reduced illumination, low contrast, and
severely blurred object boundaries, especially in distant re-
gions of the scene. Compared to baseline SOTA algorithms,
our method, especially the T-SQIW-25600 setting, successfully
detects all dynamic pedestrian objects in the scene (as shown
by the dark blue squares) and also accurately captures the
background objects’ contour as well (as shown by the dark
red squares), demonstrating the robustness of the proposed
algorithm under foggy conditions. Nevertheless, in the snow
condition, the snowfall captured in the surround-view image
appears to introduce salt-and-pepper noise to the image, and
snowflakes occlude some objects’ contours, even the main
body parts. In this case, our proposed method is less tolerant
of salt-and-pepper noise than the baseline SOTA algorithms.
J. Model Efficiency
We evaluated the real-time performance of our model in var-
ious settings on an RTX 4090 GPU and presented the results in
Table XII. The experiment results show the trade-off between
the total number of primitives and the model inference speed.
Method
Modality
Latency
(ms) (↓)
Memory
(GB) (↓)
Params
SurroundOcc* [25]
C
472
5.98
180.51M
InverseMatrixVT3D* [32]
C
447
4.41
67.18M
OccFusion(C+R)* [1]
C+R
588
5.56
92.71M
OccFusion(C+L)* [1]
C+L
591
5.56
92.71M
OccFusion(C+L+R)* [1]
C+L+R
601
5.78
114.97M
OccCylindrical* [4]
C+L
398
10.63
111.62M
TFusionOcc (T-SQ-12800)
C+L
278.46
3.68
95.62M
TFusionOcc (T-SQ-IW-12800)
C+L
281.47
3.88
95.96M
TFusionOcc (T-P-25600)
C+L
385.49
4.73
97.51M
TFusionOcc (T-SQ-25600)
C+L
388.05
4.13
97.59M
TFusionOcc (T-SQ-IW-25600)
C+L
391.06
4.04
98.24M
TABLE XII: Efficiency comparison of multi-modal 3D semantic
occupancy prediction on the nuScenes validation set. Algorithm
results marked with * were taken from the paper that introduced
the model and were measured on a different GPU.
More primitives require more rendering operations, leading to
higher latency. Surprisingly, all five settings have similar total
model parameters. Based on the main performance benchmark
shown in Table II, the T-SQ-IW-12800 setting delivers the best
balance between detection and real-time performance.
K. Ablation Study
Main-Skele
Augment
Overlap Remove
Range Filter
IoU (↑)
mIoU (↑)
!
!
!
!
46.24 (Baseline)
30.79 (Baseline)
!
!
%
!
46.20 (-0.04%)
30.70 (-0.09%)
!
!
!
%
46.07 (-0.17%)
30.76 (-0.03%)
!
!
%
%
46.08 (-0.16%)
30.69 (-0.1%)
!
%
%
%
41.96 (-4.28%)
29.92 (-0.87%)
%
!
%
%
29.35 (-16.89%)
22.31 (-8.48%)
TABLE XIII: Ablation study on the Skeleton Merge Module.
Main-Skele: Lidar-side main skeleton branch. Augment: Camera-side
augment skeleton branch. ↑:the higher, the better.

<!-- page 16 -->
16
FRONT LEFT
FRONT 
FRONT RIGHT BACK LEFT
BACK 
BACK RIGHT
TFusionOcc (T-P-25600)
TFusionOcc (T-SQ-IW-12800)
TFusionOcc (T-SQ-IW-25600)
Ground Truth
TFusionOcc (T-SQ-12800)
TFusionOcc (T-SQ-25600)
TFusionOcc (T-P-25600)
TFusionOcc (T-SQ-IW-12800)
TFusionOcc (T-SQ-IW-25600)
Ground Truth
TFusionOcc (T-SQ-12800)
TFusionOcc (T-SQ-25600)
TFusionOcc (T-P-25600)
TFusionOcc (T-SQ-IW-12800)
TFusionOcc (T-SQ-IW-25600)
Ground Truth
TFusionOcc (T-SQ-12800)
TFusionOcc (T-SQ-25600)
Fig. 12: Qualitative results under different settings for daytime, rainy, and nighttime scenarios displayed in the upper, middle,
and bottom sections, respectively. Better viewed when zoomed in.

<!-- page 17 -->
17
FRONT LEFT
FRONT 
FRONT RIGHT BACK LEFT
BACK 
BACK RIGHT
TFusionOcc (T-P-25600)
OccCylindrical
DAOcc
Ground Truth
TFusionOcc (T-SQ-25600)
TFusionOcc (T-SQ-IW-25600)
TFusionOcc (T-P-25600)
OccCylindrical
DAOcc
Ground Truth
TFusionOcc (T-SQ-25600)
TFusionOcc (T-SQ-IW-25600)
TFusionOcc (T-P-25600)
OccCylindrical
DAOcc
Ground Truth
TFusionOcc (T-SQ-25600)
TFusionOcc (T-SQ-IW-25600)
Fig. 13: Qualitative results compare against other SOTA algorithms for daytime, rainy, and nighttime scenarios displayed in
the upper, middle, and bottom sections, respectively. Better viewed when zoomed in.

<!-- page 18 -->
18
FRONT LEFT
FRONT 
FRONT RIGHT BACK LEFT
BACK 
BACK RIGHT
TFusionOcc (T-P-25600)
Ground Truth
TFusionOcc (T-SQ-25600)
TFusionOcc (T-SQIW-25600)
OccCylindrical
DAOcc
TFusionOcc (T-P-25600)
Ground Truth
TFusionOcc (T-SQ-25600)
TFusionOcc (T-SQIW-25600)
OccCylindrical
DAOcc
Fig. 14: Qualitative results compare against other SOTA algorithms under fog and snow scenarios, displayed in the top and
bottom sections, respectively. Better viewed when zoomed in.
Range Filter r
IoU (↑)
mIoU (↑)
5 m
46.24 (Baseline)
30.79 (Baseline)
4 m
46.21 (-0.03%)
30.75 (-0.04%)
3 m
46.09 (-0.15%)
30.76 (-0.03%)
2 m
45.72 (-0.52%)
30.69 (-0.10%)
1 m
44.64 (-1.60%)
30.32 (-0.47%)
0.5 m
43.60 (-2.64%)
30.04 (-0.75%)
TABLE XIV: Ablation study on range filter hyperparameter r. ↑:the
higher, the better.
1) Ablation Study On Skeleton Merge Module: We con-
ducted an ablation study for each component of the skeleton
merge module and demonstrated the results of the experiment
in Table XIII. The experiment results prove the significant
impact of the main skeleton structure, which supports the
fundamental 29.92% mIoU and 41.96% IoU performance,
and the auxiliary impact of the augment structure, which
further boosts 0.87% mIoU and 4.28% IoU performance. The
main skeleton, based on lidar, provides detailed 3D geometric
information of the surrounding scene, and the augmented
skeleton, based on the camera, provides additional information
to mitigate occlusion impact in the scene. Furthermore, we
studied the impact of the range filter hyperparameter r on the
overall performance of the model and present the results in
Table XIV. The results show that a smaller filter range forces
the augmented skeleton to get too close to the main skeleton,

<!-- page 19 -->
19
Vis-Depth Fusion
Depth-Fusion
Vis-Depth
Lidar-Depth
IoU (↑)
mIoU (↑)
!
!
!
!
46.24 (Baseline)
30.79 (Baseline)
!
%
!
%
46.06 (-0.18%)
30.57 (-0.22%)
!
%
%
!
39.80 (-6.44%)
25.28 (-5.51%)
%
!
!
!
44.01 (-2.23%)
28.98 (-1.81%)
%
%
!
%
43.99 (-2.25%)
28.93 (-1.86%)
%
%
%
!
40.33 (-5.91%)
24.71 (-6.08%)
TABLE XV: Ablation study on Depth-Fusion-Based 3D Deformable
Attention Module. ↑:the higher, the better. Vis-Depth Fusion: add the
fused depth feature back to the visual feature, resulting in a depth-
aware visual feature. Depth-Fusion: fuse the visual depth feature with
the lidar depth feature. Vis-Depth: visual depth feature. Lidar-Depth:
lidar depth feature. ↑:the higher, the better.
Skip-Connect
WSFusion
GCFusion
IoU (↑)
mIoU (↑)
!
!
!
46.24 (Baseline)
30.79 (Baseline)
!
!
%
38.60 (-7.64%)
18.18 (-12.61%)
!
%
!
39.39 (-6.85%)
18.57 (-12.22%)
%
!
!
24.20 (-22.04%)
19.91 (-10.88%)
TABLE XVI: Ablation study on the MGCAFusion Module. Skip-
Connet: skip-connection structure at the last feature fusion stage.
WSFusion: weighted summation fusion part. GCFusion: gated con-
catenation fusion part. ↑:the higher, the better.
preventing it from overcoming the lidar sensor’s occlusion
limitation.
2) Ablation Study On Depth-Fusion-Based 3D Deformable
Attention Module:
We conducted an ablation study with
respect to each component of the depth-fusion-based 3D
deformable attention module and presented the experiment
results in Table XV. In practice, Vis-Depth provides dense but
less accurate depth maps, while Lidar-Depth provides sparse
but more accurate depth maps. The experiment shows that
dense depth maps contribute significantly to the performance,
even though they are less accurate. Relying solely on sparse
but accurate depth maps from lidar results in a significant
performance degradation of 5.5%-6.0%. Furthermore, adding
fused depth maps back to the original visual features results
in depth-aware visual features contributing the second-largest
performance improvement, around 2.2%. The experiment re-
sults demonstrate the necessity of multi-stage multi-modality
feature fusion to boost model performance.
3) Ablation Study On MGCAFusion Module: We conducted
an ablation study for each major component of the MGCAFu-
sion module and present the experiment results in Table XVI.
The experiment results show that the weighted summation
and gated concatenation fusion components contribute almost
equally to the model’s final performance, with the former
slightly more important. Missing each part will cause 12.22%
to 12.61% mIoU and 6.85% to 7.64% IoU performance
degradation. The skip-connection structure plays an important
role during the feature fusion.
V. CONCLUSION AND FUTURE WORK
This paper presents TFusionOcc, a novel Student’s t-
distribution-based object-centric multi-stage multi-sensor fu-
sion framework. Our framework in the sensor fusion part
adopts a multi-stage feature-level fusion strategy, including an
early-stage, a middle-stage, and a late-stage fusion, to better
integrate complementary information across modalities and
mitigate the drawbacks of each modality. We also adopt an
object-centric modeling strategy by leveraging the Student’s
t-distribution and the T-Mixture model as a probability kernel,
combined with several typical primitives, such as the general
T-Primitive and the superquadric, to achieve greater robustness
against outliers. We further propose a primitive called the
deformable superquadric primitive (superquadric with inverse-
warp), which offers greater geometric shape representation
flexibility, thereby boosting the model’s 3D geometric shape-
capturing capability in the driving scene. The extensive ex-
periments conducted on the nuScenes [10] and nuScenes-C
[11] datasets, including challenging cases of rainy, night, fog,
snow and sensor-malfunction, demonstrate that our proposed
approach not only achieves SOTA performance but also ex-
hibits superior robustness. In our future research endeavors, we
will investigate how to utilize a much smaller set of primitives
and fewer rendering operations to achieve similar performance.
REFERENCES
[1] Z. Ming, J. S. Berrio, M. Shan, and S. Worrall, “Occfusion: Multi-
sensor fusion framework for 3d semantic occupancy prediction,” IEEE
Transactions on Intelligent Vehicles, 2024.
[2] J. Pan, Z. Wang, and L. Wang, “Co-occ: Coupling explicit feature
fusion with volume rendering regularization for multi-modal 3d semantic
occupancy prediction,” IEEE Robotics and Automation Letters, 2024.
[3] S. Zhang, Y. Zhai, J. Mei, and Y. Hu, “Fusionocc: Multi-modal fusion for
3d occupancy prediction,” in Proceedings of the 32nd ACM International
Conference on Multimedia, 2024, pp. 787–796.
[4] Z. Ming, J. S. Berrio, M. Shan, Y. Huang, H. Lyu, N. H. K. Tran,
T.-Y. Tseng, and S. Worrall, “Occcylindrical: Multi-modal fusion with
cylindrical representation for 3d semantic occupancy prediction,” arXiv
preprint arXiv:2505.03284, 2025.
[5] Z. Yang, Y. Dong, J. Wang, H. Wang, L. Ma, Z. Cui, Q. Liu, H. Pei,
K. Zhang, and C. Zhang, “Daocc: 3d object detection assisted multi-
sensor fusion for 3d occupancy prediction,” IEEE Transactions on
Circuits and Systems for Video Technology, 2025.
[6] Z. Duan, C. Dang, X. Hu, P. An, J. Ding, J. Zhan, Y. Xu, and J. Ma,
“Sdgocc: Semantic and depth-guided bird’s-eye view transformation for
3d multimodal occupancy prediction,” in Proceedings of the Computer
Vision and Pattern Recognition Conference, 2025, pp. 6751–6760.
[7] Y. Shi, K. Jiang, J. Miao, K. Wang, K. Qian, Y. Wang, J. Li, T. Wen,
M. Yang, Y. Xu et al., “Effocc: Learning efficient occupancy net-
works from minimal labels for autonomous driving,” arXiv preprint
arXiv:2406.07042, 2024.
[8] L. Zhao, S. Wei, J. Hays, and L. Gan, “Gaussianformer3d: Multi-modal
gaussian-based semantic occupancy prediction with 3d deformable at-
tention,” arXiv preprint arXiv:2505.10685, 2025.
[9] T. Pavkovi´c, M.-A. N. Mahani, J. Niedermayer, and J. Betz, “Gaus-
sianfusionocc: A seamless sensor fusion approach for 3d occupancy
prediction using 3d gaussians,” arXiv preprint arXiv:2507.18522, 2025.
[10] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu,
A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A
multimodal dataset for autonomous driving,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2020,
pp. 11 621–11 631.
[11] S. Xie, L. Kong, W. Zhang, J. Ren, L. Pan, K. Chen, and Z. Liu,
“Benchmarking and improving bird’s eye view perception robustness
in autonomous driving,” IEEE Transactions on Pattern Analysis and
Machine Intelligence, 2025.
[12] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[13] Z. Ming, J. S. Berrio-Perez, M. Shan, and S. Worrall, “Inverse++:
Vision-centric 3d semantic occupancy prediction assisted with 3d object
detection,” Neurocomputing, p. 132162, 2025.
[14] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, “Gaussianformer:
Scene as gaussians for vision-based 3d semantic occupancy prediction,”
in European Conference on Computer Vision. Springer, 2024, pp. 376–
393.

<!-- page 20 -->
20
[15] Y. Huang, A. Thammatadatrakoon, W. Zheng, Y. Zhang, D. Du, and
J. Lu, “Gaussianformer-2: Probabilistic gaussian superposition for effi-
cient 3d occupancy prediction,” in Proceedings of the Computer Vision
and Pattern Recognition Conference, 2025, pp. 27 477–27 486.
[16] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[17] S. Zuo, W. Zheng, X. Han, L. Yang, Y. Pan, and J. Lu, “Quadricformer:
Scene as superquadrics for 3d semantic occupancy prediction,” arXiv
preprint arXiv:2506.10977, 2025.
[18] H. Zhou, X. Zhu, X. Song, Y. Ma, Z. Wang, H. Li, and D. Lin,
“Cylinder3d: An effective 3d framework for driving-scene lidar semantic
segmentation,” arXiv preprint arXiv:2008.01550, 2020.
[19] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in Proceedings of the IEEE conference on computer vision
and pattern recognition, 2016, pp. 770–778.
[20] Y. Yan, Y. Mao, and B. Li, “Second: Sparsely embedded convolutional
detection,” Sensors, vol. 18, no. 10, p. 3337, 2018.
[21] H. Zhao, L. Jiang, J. Jia, P. H. Torr, and V. Koltun, “Point transformer,”
in Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 16 259–16 268.
[22] H. Li, H. Zhang, Z. Zeng, S. Liu, F. Li, T. Ren, and L. Zhang, “Dfa3d:
3d deformable attention for 2d-to-3d feature lifting,” in Proceedings of
the IEEE/CVF International Conference on Computer Vision, 2023, pp.
6684–6693.
[23] T. Wang, X. Zhu, J. Pang, and D. Lin, “Fcos3d: Fully convolutional one-
stage monocular 3d object detection,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021, pp. 913–922.
[24] M. Berman, A. R. Triki, and M. B. Blaschko, “The lov´asz-softmax loss:
A tractable surrogate for the optimization of the intersection-over-union
measure in neural networks,” in Proceedings of the IEEE conference on
computer vision and pattern recognition, 2018, pp. 4413–4421.
[25] Y. Wei, L. Zhao, W. Zheng, Z. Zhu, J. Zhou, and J. Lu, “Surroundocc:
Multi-camera 3d occupancy prediction for autonomous driving,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 21 729–21 740.
[26] X. Tian, T. Jiang, L. Yun, Y. Mao, H. Yang, Y. Wang, Y. Wang, and
H. Zhao, “Occ3d: A large-scale 3d occupancy prediction benchmark
for autonomous driving,” Advances in Neural Information Processing
Systems, vol. 36, pp. 64 318–64 330, 2023.
[27] A.-Q. Cao and R. de Charette, “Monoscene: Monocular 3d semantic
scene completion,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2022, pp. 3991–4001.
[28] Z. Murez, T. Van As, J. Bartolozzi, A. Sinha, V. Badrinarayanan, and
A. Rabinovich, “Atlas: End-to-end 3d scene reconstruction from posed
images,” in Computer Vision–ECCV 2020: 16th European Conference,
Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16. Springer,
2020, pp. 414–431.
[29] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Y. Qiao, and J. Dai,
“Bevformer: Learning bird’s-eye-view representation from multi-camera
images via spatiotemporal transformers,” in ECCV. Springer, 2022, pp.
1–18.
[30] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, “Tri-perspective view
for vision-based 3d semantic occupancy prediction,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 9223–9232.
[31] X. Wang, Z. Zhu, W. Xu, Y. Zhang, Y. Wei, X. Chi, Y. Ye, D. Du,
J. Lu, and X. Wang, “Openoccupancy: A large scale benchmark for
surrounding semantic occupancy perception,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2023, pp.
17 850–17 859.
[32] Z. Ming, J. S. Berrio, M. Shan, and S. Worrall, “Inversematrixvt3d: An
efficient projection matrix-based approach for 3d occupancy prediction,”
in 2024 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS).
IEEE, 2024, pp. 9565–9572.
[33] Y. Zhang, Z. Zhu, and D. Du, “Occformer: Dual-path transformer for
vision-based 3d semantic occupancy prediction,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2023, pp.
9433–9443.
[34] Z. Li, Z. Yu, D. Austin, M. Fang, S. Lan, J. Kautz, and J. M. Alvarez,
“Fb-occ: 3d occupancy prediction based on forward-backward view
transformation,” arXiv preprint arXiv:2307.01492, 2023.
[35] M. Pan, J. Liu, R. Zhang, P. Huang, X. Li, H. Xie, B. Wang, L. Liu,
and S. Zhang, “Renderocc: Vision-centric 3d occupancy prediction with
2d rendering supervision,” in 2024 IEEE International Conference on
Robotics and Automation (ICRA).
IEEE, 2024, pp. 12 404–12 411.
[36] L. Roldao, R. de Charette, and A. Verroust-Blondet, “Lmscnet:
Lightweight multiscale 3d semantic completion,” in 2020 International
Conference on 3D Vision (3DV).
IEEE, 2020, pp. 111–119.
[37] J. Huang, G. Huang, Z. Zhu, Y. Ye, and D. Du, “Bevdet: High-
performance multi-camera 3d object detection in bird-eye-view,” arXiv
preprint arXiv:2112.11790, 2021.
[38] Y. Li, H. Bao, Z. Ge, J. Yang, J. Sun, and Z. Li, “Bevstereo: Enhancing
depth estimation in multi-view 3d object detection with temporal stereo,”
in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37,
no. 2, 2023, pp. 1486–1494.
[39] H. Zhang, X. Yan, D. Bai, J. Gao, P. Wang, B. Liu, S. Cui, and
Z. Li, “Radocc: Learning cross-modality occupancy knowledge through
rendering assisted distillation,” in Proceedings of the AAAI Conference
on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7060–7068.
[40] Y. Ren, L. Wang, M. Li, H. Jiang, Z. Cui, M. Yang, H. Yu, and
D. Yang, “Rm 2 occ: Re-projection multi-task multi-sensor fusion for
autonomous driving 3d object detection and occupancy perception,”
IEEE Transactions on Intelligent Transportation Systems, 2025.
