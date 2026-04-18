<!-- page 1 -->
Aerial-Ground Image Feature Matching via 3D
Gaussian Splatting-based Intermediate View Rendering
Jiangxue Yua, Hui Wanga, San Jiangb,c,d,∗, Xing Zhangb,c, Dejin Zhangb,c,
Qingquan Lib,c
aSchool of Computer Science, China University of Geosciences, Wuhan, 430074, China
bGuangdong Key Laboratory of Urban Informatics, Shenzhen University, Guangdong
Shenzhen, 518060, China
cMNR Key Laboratory for Geo-Environmental Monitoring of Great Bay Area, Shenzhen
University, Guangdong Shenzhen, 518060, China
dEngineering Research Center of Natural Resource Information Management and Digital
Twin Engineering Software, Ministry of Education, Wuhan, 430074, China
Abstract
The integration of aerial and ground images has been a promising solution
in 3D modeling of complex scenes, which is seriously restricted by finding
reliable correspondences. The primary contribution of this study is a feature
matching algorithm for aerial and ground images, whose core idea is to gen-
erate intermediate views to alleviate perspective distortions caused by the
extensive viewpoint changes. First, by using aerial images only, sparse mod-
els are reconstructed through an incremental SfM (Structure from Motion)
engine due to their large scene coverage.
Second, 3D Gaussian Splatting
is then adopted for scene rendering by taking as inputs sparse points and
oriented images. For accurate view rendering, a render viewpoint determi-
nation algorithm is designed by using the oriented camera poses of aerial
images, which is used to generate high-quality intermediate images that can
bridge the gap between aerial and ground images. Third, with the aid of
intermediate images, reliable feature matching is conducted for match pairs
from render-aerial and render-ground images, and final matches can be gener-
ated by transmitting correspondences through intermediate views. By using
real aerial and ground datasets, the validation of the proposed solution has
been verified in terms of feature matching and scene rendering and com-
∗Corresponding author.
Email address: jiangsan@szu.edu.cn (San Jiang)
Preprint submitted to Elsevier
September 25, 2025
arXiv:2509.19898v1  [cs.CV]  24 Sep 2025

<!-- page 2 -->
pared comprehensively with widely used methods. The experimental results
demonstrate that the proposed solution can provide reliable feature matches
for aerial and ground images with an obvious increase in the number of initial
and refined matches, and it can provide enough matches to achieve accurate
ISfM reconstruction and complete 3DGS-based scene rendering.
Keywords:
3D reconstruction, 3D Gaussian Splatting, feature matching,
unmanned aerial vehicle, mobile mapping system, structure from motion
1. Introduction
3D reconstruction has been extensively explored within the domain of
photogrammetry and remote sensing (RS), and it plays a key role in ur-
ban scene modeling and management (Ge et al., 2024; Wang et al., 2023).
In recent years, low-altitude RS platforms, such as unmanned aerial vehicle
(UAV), can provide images with high spatial and time resolutions for the
detailed 3D modeling of urban scenes, except for satellite images that aim
at covering large-scale scenes.
Integrated with geometry-aware trajectory
planning techniques, such as the optimized views photogrammetry and snap-
to-the-object photogrammetry, UAV can achieve flexible recording of ground
objects and collect multi-view high-resolution images (Li et al., 2023a). How-
ever, 3D modeling in urban canyons still faces great challenges because of
the limited observation space of aerial RS platforms and serious occlusions
of urban high buildings. Compared with aerial RS platforms, ground mo-
bile mapping systems (MMS) have the advantage of recording near-ground
scenes, which could be a well compensation to provide complete observation
(Jhan et al., 2021). Thus, the integration of aerial and ground images has
been a promising solution in 3D modeling of complex urban scenes (Gao
et al., 2019).
The integration of aerial and ground images mainly relies on their reli-
able registration. In the literature, there are two major groups of solutions to
achieve this purpose (Gao et al., 2019). For the first one, by using the POS
(Positioning and Orientation System) data from RS platforms, sparse 3D
models can be separately created from aerial and ground images and roughly
geo-referenced to the same coordinate system. The integration of aerial and
ground images is then converted as the registration of point clouds of two
sparse models (Gao et al., 2018a,b; Shan et al., 2014), which can be achieved
by using a reliable geometric transformation estimation algorithm, such as
2

<!-- page 3 -->
ICP (Iterative Closest Point) (Besl and McKay, 1992). However, the reg-
istration accuracy of the fused model would be seriously degenerated due
to the limited overlap and existing outliers in the sparse models. For the
second group, feature matching is further conducted to establish correspon-
dences between aerial and ground images, which are then used as tie-points
for the combined BA (bundle adjustment) optimization. The perspective
distortions and scale differences are the main issues that cause the failure of
feature matching between aerial and ground images. According to the used
strategy for alleviating these issues, existing methods can be divided into two
categories, i.e., image rectification-based methods (Li et al., 2023b; Liu et al.,
2023; Song et al., 2019; Wu et al., 2018) and view rendering-based methods
(Gao et al., 2018a,b; Zhu et al., 2020). The former uses either rough POS
data (Hu et al., 2015; Jiang and Jiang, 2017) or exploits common planes,
such as building facades (Liu et al., 2023; Wu et al., 2018), for image global
rectification. These methods can obviously decrease appearance differences
and increase feature repeatability. However, their performance heavily relies
on auxiliary data and scene structures. Thus, view rendering-based meth-
ods are further exploited, which mainly aim to render intermediate views
to bridge the gap between aerial and ground images. Compared with image
rectification-based methods, view rendering-based methods do not depend on
specific scene structures. However, view rendering-based methods are degen-
erated by the high computational costs required to build dense and textured
models and large distortions in rendered images.
In contrast to the above-mentioned hand-crafted methods, deep learning-
based methods have also gained extensive attention in feature matching (Xu
et al., 2024). Existing methods on one hand use the representation learning
ability of CNN (convolutional neural network) for feature detection and de-
scription, which can be observed from the earlier networks for image patch
description (Luo et al., 2019, 2018; Mishchuk et al., 2017; Tian et al., 2017)
to recent end-to-end networks for joint feature detection and description
(DeTone et al., 2018; Dusmanu et al., 2019; Luo et al., 2020). On the other
hand, to enhance the discriminability of descriptors, GNN (graph neural
network) has also been used in recent studies for feature matching, which
obviously promotes the development of feature matching in challenging con-
ditions, such as high texture-less regions and low illumination scenes (Chen
et al., 2022; Edstedt et al., 2024; Sarlin et al., 2020; Sun et al., 2021; Wang
et al., 2022). However, large perspective distortions are still the main chal-
lenge for feature matching of aerial and ground images. In recent years, deep
3

<!-- page 4 -->
learning has also promoted the development of novel view synthesis, such as
the cutting-edge technique NeRF (Neural Radiance Field) (Mildenhall et al.,
2021) and 3DGS (3D Gaussian Splatting) (Kerbl et al., 2023). Compared
with classical view rendering-based methods, these techniques have high ren-
dering efficiency and synthesis quality because they can render views from
sparse models without the time-consuming step for building dense models.
Inspired by these recent studies, this study proposes a reliable feature
matching solution for aerial and ground images. Similar to view rendering-
based methods, the core idea is to render intermediate views to alleviate
perspective distortions and scale differences and build transmit matches be-
tween aerial and ground images. The main contributions of this study are
summarized as follows: (1) we propose a view rendering algorithm via 3D
Gaussian Splatting, which uses POS data for the coarse alignment of sparse
models and calculation of intermediate view position and orientation; (2)
we design a feature matching algorithm between aerial and ground images,
which combines a deep learning-based feature matching network and tran-
sits aerial-ground feature matches through the rendered intermediate views;
(3) we further verify the performance of the proposed solution by using real
aerial and ground datasets in terms of feature matching and SfM (Structure
from Motion) based 3D reconstruction.
This paper is organized as follows. Section 2 presents the related work in
the literature. Section 3 gives the details of the proposed methodology for
aerial and ground image matching, which is followed by experimental tests
in terms of feature matching and 3D Gaussian Splatting scene rendering in
Section 4. Finally, Section 5 presents conclusions and future studies.
2. Related work
This study aims at feature matching between aerial and ground images.
In the literature, existing methods can be grouped into three categories, i.e.,
image rectification-based methods, view rendering-based methods, and deep
learning-based methods. Therefore, the following literature review will be
conducted from these three aspects.
2.1. Image rectification-based methods
The main difficulty in the feature matching of aerial and ground images
is the perspective distortions and scale differences. Image rectification is the
most direct strategy to decrease their geometric inconsistency. According
4

<!-- page 5 -->
to the used rectification strategy, existing methods can be divided into two
groups, i.e., image horizontal rectification and image vertical rectification.
The first one uses the POS data to rectify images onto geographic surfaces.
By using onboard POS and ground elevation data, Hu et al. (2015) pro-
posed rectifying aerial oblique images onto the same elevation plane, which
decreases image deformations.
In the work of Jiang and Jiang (2017), a
comparative evaluation has been conducted for UAV oblique images, which
compares feature matching performance with and without image rectifica-
tion. The tests demonstrate its validation in feature matching. In contrast
to oblique image matching, the rectification plane is usually vertical consid-
ering the observation direction of aerial and ground images.
The second one is image vertical rectification. Instead of using an ele-
vation plane, these methods exploit vertical primitives as projection planes.
In urban environments, building facades are the most obvious plane prim-
itives that have been exploited frequently in studies. Based on this idea,
Wu et al. (2018) proposed an approach for feature matching of aerial oblique
imagery and terrestrial imagery. To address perspective distortion and scale
variation, base planes are first selected from urban building facades that can
be observed by both aerial and ground images. Image rectification is then
executed to make building facades geometrically consistent in both images.
The proposed feature matching method has been applied for high-quality
3D modeling of urban scenes. In their subsequent study (Li et al., 2023b),
the proposed solution has been further adopted for the integration of aerial,
MMS, and backpack images and 3 mapping in urban areas. Instead of the
above-mentioned image global rectification, Liu et al. (2023) proposed an
image patch rectification algorithm that rectifies image patches in one local
normalized plane (Wu et al., 2008), which has been verified in feature match-
ing of aerial and ground images. Compared with image global rectification,
more time costs are consumed in generating and rectifying image patches.
Generally, image rectification-based methods on the one hand depend on on-
board POS data for geometric projection and coarse alignment; on the other
hand, they rely on specific structures to support primitive plane selection.
2.2. View rendering-based methods
View rendering-based methods aim to render intermediate views that are
used to bridge the gap between aerial and ground images, which can reduce
their perspective distortions and scale differences.
Compared with image
rectification-based methods, view rendering-based methods have two major
5

<!-- page 6 -->
advantages. On the one hand, they do not rely on specific structures to se-
lect primitive planes, such as building facades; on the other hand, they can
render new images observing from any viewpoints. View rendering-based
methods have gained attention in recent studies. As the earlier work, Shan
et al. (2014) proposed a viewpoint-dependent matching method that uses
depth maps and camera poses to warp ground images onto aerial images,
and experiments for model geo-localization have been conducted to verify
the feature matching solution. Based on the same idea, Gao et al. (2018a)
designed an accurate algorithm to align aerial and ground 3D models, whose
core idea is to render aerial images from ground models at properly selected
viewpoints. Considering outliers and noises in ground models, a view ren-
dering algorithm has been designed to avoid artifacts and increase render
quality.
In contrast to ground-to-aerial view rendering, Zhu et al. (2020) proposed
an aerial-to-ground view rendering algorithm that renders ground images
from aerial models at ground views by using dense reconstructed mesh mod-
els and conducted feature matching between rendered images and ground
images. With the aid of depths and normal vectors of rendered ground im-
ages, initial matches are then transmitted to aerial images. The proposed so-
lution has been evaluated by using five open-sourced aerial-ground datasets.
Due to fact that dense matching and texture mapping are extremely high
time-consuming, the authors further designed a new view rendering strategy
from sparse mesh models (Gao et al., 2018b). Co-visible meshes were first
extracted from ground mesh models, which were used to render aerial im-
ages by using triangle-induced homograph between aerial and ground images.
The proposed solution has been verified by 3D modeling of ancient Chinese
architectures. However, artifacts inevitably exist in the rendered images due
to the usage of sparse model. In conclusion, although view rendering-based
methods can bridge the gap between aerial and ground images, high time
consumptions and low rendering quality are their major drawbacks.
2.3. Deep learning-based methods
Due to the high representation learning ability, deep learning-based meth-
ods have gained rapid development in feature matching. Pioneer methods
use CNN to extract high-level feature maps and implement feature detection
and description. Existing networks can be categorized according to the steps
of image matching, i.e., feature detection, feature description, and feature
matching. The earlier networks aim at generating feature descriptors with
6

<!-- page 7 -->
high discriminative ability, such as L2-Net (Tian et al., 2017), GeoDesc (Luo
et al., 2018), and ContextDesc (Luo et al., 2019), which feed into image
patches and generate feature descriptors for the subsequent feature match-
ing. Considering feature matching in difficult conditions, e.g., low-texture
and low-illumination environments, joint feature detection and description
networks have been proposed, which use image pairs as inputs and gener-
ate feature points and descriptors. The well-known networks include D2-Net
(Dusmanu et al., 2019), ASLFeat (Luo et al., 2020), and SuperPoint (DeTone
et al., 2018). Joint feature detection and description networks use feature
maps to detect keypoints and create descriptors. These methods have been
extensively utilized for oblique image feature matching (Ji et al., 2023; Jiang
et al., 2021).
Except for feature detection and description networks,
GNN and
Transformer-based attention mechanism have also been exploited in recent
feature matching networks. SuperGlue (Sarlin et al., 2020) is the most well-
known network that adopts attentional GNN to enhance feature keypoints
and descriptors into high-dimension vectors. Compared with L2 distance-
based feature matching strategy, SuperGlue analogs human to find corre-
spondences between image pairs and identify false matches. This work has
greatly inspired following research. To cope with feature matching in non-
texture regions, recent years have witnessed the arising of detector-free fea-
ture matching networks, e.g., LoFTR (Sun et al., 2021), ASpanFormer (Chen
et al., 2022), and MatchFormer (Wang et al., 2022). The core idea of these
networks is to iteratively use self and cross attention layers in Transformer
to obtain enhanced feature maps. For feature matching of aerial and ground
images, detector-free networks have gain lots of attention in the field of pho-
togrammetry (Li et al., 2023b; Liang et al., 2023; Zhu et al., 2024).
3. Methodology
Figure 1 shows the overall workflow of the proposed feature matching
algorithm for aerial and ground images. UAV images and their rough POS
data are the only inputs, and three major steps are conducted to achieve
feature matching. First, a sparse model is constructed from aerial images
through a well-known SfM pipeline, and the corresponding scene is rendered
by using the 3DGS algorithm. Second, based on the SfM sparse model, a
set of render viewpoints can be calculated according to aerial viewpoints and
the pre-defined configurations, which are then used to guide the rendering
7

<!-- page 8 -->
of intermediate views from the 3DGS model. Third, by using the rendered
images, feature matching is conducted among aerial, ground, and rendered
images. That is, a set of feature matches can be obtained between rendered
and aerial images as well as between rendered and ground images because
of their relatively small perspective deformation. Finally, feature matches
between aerial and ground images can be induced from the above results by
using rendered images as the transistor.
Figure 1: The workflow of the proposed algorithm.
3.1. The principle of 3D Gaussian Splatting
The core idea of the proposed feature matching algorithm is using 3D
Gaussian Splatting for photorealistic scene rendering and intermediate view
generation. Compared with classical scene rendering solutions such as MVS
(Multi-View Stereo) based dense matching and texture mapping, 3D Gaus-
sian Splatting has the advantage of low-memory consumption, high-speed
and high-quality rendering. Thus, the principle of 3D Gaussian Splatting is
first presented.
3D Gaussian Splatting is an explicit radiance field representation tech-
nique. It represents a scene by using a large set of 3D anisotropic balls, i.e.,
3D Gaussians that capture the desirable properties of continuous volumetric
8

<!-- page 9 -->
radiance fields for scene optimization (Kerbl et al., 2023). Each 3D Gaussian
G(x), as presented by Equation 1, is characterized by its mean vector µ,
covariance matrix P, opacity α, and color c that is calculated from spher-
ical harmonics as view-dependent appearance.
The covariance matrix P
can be decomposed into the rotation matrix R ∈R3×3 and the scale matrix
S ∈R3×1, i.e., P = RSSTRT.
G(x) = e−1
2 (x−µ)T Σ−1(x−µ)
(1)
When blended, 3D Gaussians create a full model that can be rendered
from any viewpoint, which can support novel view rendering. The input of 3D
Gaussian Splatting is a sparse model with point clouds and oriented images
generated from SfM. Each 3D point in the point cloud is converted into a 3D
Gaussian, with the mean and covariance of the Gaussian corresponding to
the point’s position and uncertainty. This results in a set of Gaussians that
are then optimized during training to minimize the difference between the
rasterized images and the real images. For view rendering, these Gaussians
are first projected onto the 2D image plane, and their color is calculated by
using spherical harmonics. After sorted according to depth, the color of a
pixel x is calculated by the α-blending of opacity and color of Gaussians that
cover the pixel by depth order, as presented by Equation 2, where ci is the
learned color; α′
i is calculated by the learned opacity αi and the Gaussian,
i.e., α′
i = αiG(X).
C(x) =
X
i∈N
α′
ici
i−1
Y
j=1
 1 −α′
j

(2)
3D Gaussian Splatting does not depend on dense sampling, which is very
computationally expensive and memory-intensive, as in NeRF (Mildenhall
et al., 2021).
Instead, 3D Gaussian Splatting uses a tile-based rasterizer
for efficient volume rendering, which can quickly sort and handle occlusions
while limiting the number of Gaussians that receive gradients. It allows for
high-quality novel view synthesis with only a few input views and has been
shown to achieve real-time rendering speeds with competitive training times
and visual quality. Therefore, 3D Gaussian Splatting is adopted in this study
for intermediate view rendering to bridge the gap between aerial and ground
images.
9

<!-- page 10 -->
3.2. Intermediate view rendering for feature matching
According to the procedure of 3D Gaussian Splatting, this section
presents the proposed intermediate view rendering algorithm for feature
matching of aerial and ground images. The overall workflow consists of three
major steps, including aerial image SfM reconstruction, view rendering be-
tween aerial-ground images, and aerial-ground image feature matching.
3.2.1. Aerial image SfM reconstruction
3D Gaussian Splatting uses a sparse model as its input, which includes
sparse point clouds and oriented images. The former is used to initialize
Gaussians; the latter is utilized to calculate training loss and refine rendering
scenes. In the literature, incremental SfM (ISfM) has become the golden
standard technique for sparse model generation, especially for UAV images
(Jiang et al., 2020), because of its high robustness to feature matching outliers
and low dependency on initial values of unknown parameters. Thus, as in 3D
Gaussian Splatting (Kerbl et al., 2023), the widely used open-source library
ColMap (Schonberger and Frahm, 2016) with its default configurations has
been used for generating sparse models.
(a)
(b)
Figure 2: The generated sparse model by using (a) aerial images and (b) ground images.
Image planes are rendered as red rectangles, and 3D points are rendered with image colors.
For the proposed aerial-ground feature matching workflow, there are two
options to select images for generating sparse models, i.e., aerial images and
ground images. Generally, aerial images are captured from high altitudes
with complete scene coverage, as shown in Figure 2(a); on the contrary,
ground images are collected with near distances, which can record more scene
detail, as presented in Figure 2(b). To ensure that the subsequently rendered
10

<!-- page 11 -->
images have large and valid overlap regions with existing images, aerial im-
ages are selected to conduct the ISfM reconstruction and generate the sparse
model.
3.2.2. View rendering between aerial-ground images
Photorealistic scene rendering can be achieved based on the reconstructed
sparse model. In the literature, classical SfM-MVS based workflows sequen-
tially execute dense matching for point cloud generation, point meshing for
Delaunay triangulation construction, and texture mapping for real scene ren-
dering. These solutions have the disadvantages of time-consuming and low
quality due to the long procedure and high computational burden. Thus,
this study uses 3D Gaussian Splatting for scene rendering.
The purpose of scene rendering is to rapidly generate high-quality inter-
mediate images. Considering the high resolution of aerial images, the train-
ing time and required GPU memory are substantial. However, 3D Gaussian
Splatting can still produce high-quality 3D models even under low-resolution
conditions. In this study, the original images are first down-sampled with
the scaling width and height proportionally to a width of less than 1,500
pixels. Taking as input the camera poses and 3D point clouds from the SfM
sparse model, 3D Gaussian Splatting is used for iterative training to render
the entire scene, as shown in Figure 3.
(a)
(b)
Figure 3: The visualization of (a) 3D Gaussians in the rendered scene and (b) novel view
image rendering based on 3D Gaussian Splatting.
Render viewpoint determination is critical for aerial-ground intermedi-
ate view rendering as it mainly affects the perspective deformations between
11

<!-- page 12 -->
rendered images and aerial-ground images. By using the SfM reconstruction,
this study designs an algorithm to determine render viewpoints, as illus-
trated in Figure 4. Assuming that the camera pose of an aerial viewpoint
is represented as {Ra, Ca}, in which Ra and Ca = {Xa, Ya, Za} indicate the
rotation matrix and projection center, respectively. Based on the imaging
geometry, the coordinate {Xt, Yt, Zt} of the ground target observed through
the principle point of the aerial viewpoint is calculated via ray intersection.
The core idea of the proposed algorithm is that the rendered images must
have large overlap regions with aerial and ground images and have as less as
possible perspective deformations. Thus, the render viewpoint is determined
by moving the position of the aerial viewpoint along the vertical direction
and rotating the camera around X axis until the ground target is observed
again via the principle point of the rendered viewpoint. According to above
definition, the camera pose {Rr, Cr} of the render view points is calculated
via Equations 3 and 4 , in which β is the intersection angle of vectors ⃗a and
⃗b, i.e., cos(β) = ⃗a∗⃗b/∥⃗a∥∥⃗b∥. By using these render viewpoints, intermediate
images can be generated from the 3D GS model.



Xr = Xa
Yr = Ya
Zr = 1
2Za
(3)
Rr = Ra ∗RX(−β)
(4)
3.2.3. Aerial-ground image feature matching
Aerial and ground images have significant differences in viewing angles
and low overlaps, which leads to a large variance in feature descriptors and
low repeatability of corresponding points. It in turn reduces the reliability
and accuracy of direct feature matching. The principle of the proposed algo-
rithm in this paper is to introduce intermediate images between aerial and
ground images. The intermediate images reduce perspective differences and
increase overlap regions between aerial-ground images, which decreases the
difficulty of direct aerial-ground feature matching and serves as a bridge to
assist in feature matching.
By using the rendered intermediate images, feature matching can then be
executed using a well-known algorithm. In this study, we have tested using
SIFT with default configurations. It is shown that the default SIFT method
resulted in very few initial matches among the three sets of images, which
12

<!-- page 13 -->
Figure 4: The illustration of render viewpoint determination based on the oriented camera
poses of aerial images from SfM reconstruction.
could not meet the requirements of subsequent experiments. Therefore, re-
cent deep learning-based methods, i.e., SuperPoint & SuperGlue (DeTone
et al., 2018; Sarlin et al., 2020), have been selected for feature extraction
and matching. First, SuperPoint is used for local feature extraction from all
images to obtain all local features. Second, SuperGlue is adopted to execute
feature matching between intermediate images and aerial-ground images. Fi-
nally, after outlier removal based on fundamental matrix estimation, final
matches can be obtained for aerial-intermediate and ground-intermediate im-
age pairs. The results reveal that the feature matching solution can achieve
a larger match number and higher precision.
Feature matches between aerial-ground images should be extracted after
obtaining feature matching results of aerial-intermediate images and ground-
intermediate images. The process of merging feature matching results be-
tween intermediate images and aerial-ground images is shown in Figure 5.
The main procedure consists of the steps: (1) for one intermediate image,
reading its matching pairs with aerial and ground images, i.e., one pair of
aerial-intermediate image matches and one pair of ground-intermediate im-
age matches; (2) extracting match data from these two image pairs, and
merging them to generate aerial-ground image matching pairs; (3) checking
if the new aerial-ground image pair whether or not exists; if it does, merging
the new and old feature matching results and ensure their uniqueness; if not,
directly establishing a new image pair; (4) repeating the above steps for all
13

<!-- page 14 -->
intermediate images. Especially, step (2) involves reading match data corre-
sponding to aerial-intermediate image pairs and ground-intermediate image
pairs and storing them in Array 1 and Array 2, respectively. Then, finding
the points in the intermediate image that exist in both Array 1 and Array
2, and storing them as common points, which are the matching points of
aerial and ground images. According to these processing steps, final matches
between aerial-ground images can be obtained.
Figure 5: The procedure of aerial-ground feature matching and merging.
3.3. Algorithm implementation
According to the proposed image matching workflow, this study has im-
plemented the algorithm by using open-source and self-developed packages.
Especially, the widely used ISfM toolkit ColMap with the default configu-
rations has been used for generating sparse models, in which 3D points and
oriented images are used for 3D Gaussian Splatting based scene rendering.
14

<!-- page 15 -->
For 3DGS-based scene rendering, we set the total number of training itera-
tions as 30k; we also update densification every 100 times and reset opacity
every 3,000 times. In addition, we set opacity, scale, and rotation to 0.05,
0.005, and 0.001, respectively. For aerial-ground image feature matching, we
set the input image size to 1440 pixels in SuperPoint & SuperGlue and set
the maximum keypoint number to 8192, which is similar to SIFTGPU in
ColMap.
Table 1: The source list of used software packages.
Software package
Website source
ColMap
https://github.com/colmap/colmap
3DGS
https://github.com/graphdeco-inria/gaussian-splatting
SuperPoint
https://github.com/rpautrat/SuperPoint
SuperGlue
https://github.com/magicleap/SuperGluePretrainedNetwork
4. Experimental results
In the experiments, three aerial-ground datasets are used to evaluate the
performance of the proposed algorithm.
First, we compare the efficiency
and quality of intermediate views that are rendered by classical SfM-MVS
and the 3DGS-based solutions. Second, feature matching is then conducted
to evaluate the performance of the aerial-ground image matching algorithm,
which is compared with other two feature matching methods. Third, the
matching results are utilized in ISfM reconstruction to validate its practical
applications. In this study, the tests are conducted on a Windows platform
that is equipped with 64 GB memory, a 3.0 GHz Intel Core i9-13900K CPU
and a 24 GB NVIDIA GeForce RTX 4090 graphic card.
4.1. Test sites and datasets
For performance evaluation, three widely used aerial-ground datasets have
been used in this study. The details of these datasets are listed in Table 2.
The first and second datasets, i.e., CENTER and ZECHE, are collected and
prepared as the “ISPRS benchmark for multi-platform photogrammetry”,
which serves as a multi-sensor benchmark to evaluate image orientation and
dense matching algorithms (Nex et al., 2015). Both aerial images and ground
images in these two datasets are recorded by using Sony NEX-7 cameras
15

<!-- page 16 -->
with dimensions of 6,000 by 4,000 pixels. For the dataset Center, there are
146 and 204 images collected from aerial and ground sensors with the GSD
(ground sampling distance) of 1.10 cm and 0.53 cm, respectively; for the
dataset ZECHE, there are 172 and 147 aerial and ground images, whose GSD
respectively reaches to 0.56 cm and 0.28 cm due to lower flight altitudes. The
third dataset is collected from a campus library building (Zhu et al., 2020)
by using Sony ICLE 510 and Cannon EOS M6 cameras for aerial and ground
image collection, respectively. There are 123 and 78 images recorded from the
aerial and ground sensors, whose GSD is 1.69 cm and 1.06 cm, respectively.
For these three datasets, a total number of 12 image pairs have been selected
to evaluate the performance of feature matching, as shown in Figure 6.
Table 2: Detailed information of the three datasets.
Item Name
CENTER
ZECHE
SWJTU-LIB
Aerial
Ground
Aerial
Ground
Aerial
Ground
Camera mode
Sony
NEX-7
Sony
NEX-7
Sony
NEX-7
Sony
NEX-7
Sony
ICLE 510
Cannon
EOS M6
GSD (cm)
1.10
0.53
0.56
0.28
1.69
1.06
Number of images
146
203
172
147
123
78
With POS data
Yes
Yes
Yes
Yes
Yes
Yes
Image size (pixel)
6,000×4,000
6,000×4,000
6,000×4,000
4.2. Evaluation and metrics
The performance of the proposed algorithm would be evaluated in feature
matching and ISfM reconstruction. The former uses 12 image pairs that are
selected from the three datasets to evaluate and compare its performance in
feature matching; the latter feeds as input feature matching results and con-
ducts ISfM reconstruction for aerial and ground images, which can then be
used to generate complete 3D models. For the evaluation of feature match-
ing, three metrics are used, including NCM (number of correct matches),
NIM (number of initial matches), and NMP (number of match pairs). NCM
indicates the number of true matches that are retained after outlier removal;
NIM indicates the number of matches that pass through nearest neighbor
searching and ratio test; NMP is the number of match pairs that have at
least 15 true matches. For the evaluation of ISfM reconstruction, two metrics
are also selected, i.e., NRI (number of registered images) and N3P (number
16

<!-- page 17 -->
(a) CENTER, pairs 1-4
(b) ZECHE, pairs 5-8
(c) SWJTU-LIB, pairs 9-12
Figure 6: The selected 12 image pairs from the three test datasets. The left and right
columns severally show aerial and ground images from their corresponding datasets.
of 3D points). NRI is the number of images that are successfully registered
in image orientation; N3P represents the number of resumed 3D points. The
details of conducted evaluation and used metrics are shown in Table 3.
Table 3: The details of conducted evaluation and used metrics.
Evaluation
Metric
Description
Feature matching
NCM
The number of true matches.
NIM
The number of matches.
NMP
The number of match pairs.
ISfM reconstruction
NRI
The number of registered images.
N3P
The number of resumed 3D points.
4.3. Motivation for Rendering Intermediate Views with 3DGS
This study opts to utilize 3DGS for rendering intermediate views rather
than directly employing 3DGS representations for pose optimization. This
decision is grounded in an in-depth analysis of existing pose-free 3DGS meth-
ods. Specifically, there are significant viewpoint discrepancies between aerial
and ground images, encompassing aspects such as altitude, lighting, and tex-
ture distribution. These differences complicate direct pose optimization on
17

<!-- page 18 -->
implicit representations. In contrast, rendering intermediate views simplifies
the problem to a scenario more akin to 2D image matching, thereby handling
these discrepancies more effectively.
Despite the existence of some excellent pose-free 3DGS methods, such as
CF3dgs (Fu et al., 2024), NoPoSplat (Ye et al., 2024), PF3plat (Hong et al.,
2024), etc., they exhibit limitations when dealing with open area tasks. These
methods perform well in smaller scenes, for instance, capturing hundreds of
images of a single object can yield satisfactory results. However, in the large
open area scenes involved in this paper, due to the low overlap between
images, these methods struggle to complete effective reconstructions. More-
over, these methods are computationally demanding and have primarily been
tested on low-resolution images; for example, NoPoSplat and PF3plat mainly
use image datasets of 256×256, and even CF3dgs only handles images up to
960×540.
To further verify the performance of pose-free 3DGS methods in pro-
cessing open area scenes, we tested the SWJTU-LIB dataset. The dataset’s
original resolution is 6000×4000, which we downscaled by a factor of eight
to 750×500 for testing with the CF3dgs method. The test results indicate
that the average Peak Signal-to-Noise Ratio (PSNR) of the images rendered
using the CF3dgs method is only 15. This value is significantly below the
image quality standards typically considered acceptable, indicating a con-
siderable discrepancy between the rendered images and the original images.
Additionally, visually inspecting the rendered results in Figure 7, we can ob-
serve noticeable blurring and distortion in the images, particularly around
edges and texture details. This further confirms the numerical results that
CF3dgs may not achieve the desired rendering quality when handling open
area scenes. Therefore, this study adopts the method of rendering interme-
diate views with 3DGS to overcome the aforementioned challenges.
Figure 7: The illustration of CF3dgs scene rendering.
18

<!-- page 19 -->
4.4. Analysis of the performance for image rendering
Image rendering is the critical step in the workflow of the proposed aerial-
ground feature matching algorithm. In general, image rendering can be im-
plemented by using the MVS-based pipeline that sequentially conducts image
orientation, dense matching, point meshing, and texture mapping. Main is-
sues arise from the high time cost of dense matching and low quality of scene
rending.
On the contrary, 3DGS exploits neural radiance field for scene
representation, which can achieve efficient scene rendering with high-quality
details. Thus, this section will compare the performance of these two image
rendering solutions.
By using these three datasets, MVS and 3DGS are utilized for perfor-
mance evaluation, and these two solutions take as input sparse models that
are reconstructed using an ISfM pipeline. Noticeably, only aerial images are
used in intermediate view rendering in this test. The metric efficiency is used
for performance evaluation, which is the time cost of scene rendering. The
results are listed in Table 4. Wen can see that 3DGS consumes 20.0 min,
25.0 min, and 28.0 min in scene rendering for the three datasets, respectively.
However, the time cost of MVS is about 2.75, 2.16, and 3.07 times of that
in 3DGS. In addition, the quality of scene rendering of MVS is much lower
than 3DGS, as illustrated in Figure 8, in which the left column indicates the
results of MVS-based rendering; the right column represents the results of
3DGS-based rendering. It is clearly shown that there are many erroneous
regions observed from the results of MVS, e.g., building roofs with texture-
less regions and repetitive patterns and building facades that have trans-
parent glasses. These regions pose challenges on MVS-based scene rendering
method. On the contrary, 3DGS can give photorealistic rendering results and
can facilitate the subsequent feature matching of aerial and ground images
via intermediate view rendering.
Table 4: The efficiency of different rendering solutions (unit in minutes).
Dataset
MVS
3DGS
CENTER
55.0
20.0
ZECHE
54.0
25.0
SWJTU-LIB
86.0
28.0
To further evaluate the practical performance of the two image rendering
solutions in feature matching, we selected rendered images from the SWJTU-
LIB dataset and the image pair 9 for feature matching tests. We employed
19

<!-- page 20 -->
MVS-based rendering
3DGS-based rendering
(a) CENTER
(b) ZECHE
(c) SWJTU-LIB
Figure 8: The illustration of intermediate view rendering for the three datasets.
20

<!-- page 21 -->
two feature matching methods: SIFTGPU and SuperPoint & SuperGlue.
Figure 9 shows the visualization results of the two rendering methods, while
Table 5 lists the specific quantitative outcomes, including NCM (Number of
Correct Matches) and NIM (Number of Initial Matches).
The visualization results in Figure 9 clearly demonstrate the advantages
of the 3DGS-based rendering method in feature matching. Specifically, the
images rendered with 3DGS provide a greater number of correct matches
(NCM) and initial matches (NIM) during the feature matching process, which
is beneficial for enhancing the accuracy and efficiency of subsequent feature
matching. In contrast, although the MVS-based rendering method can also
perform feature matching, it falls short of the 3DGS method in both the
quantity and quality of matching points. The quantitative results in Table 5
further confirm this. These results indicate that the 3DGS rendering method
has a significant advantage in feature matching, capable of providing richer
and more accurate feature match information for subsequent 3D reconstruc-
tion. This not only verifies the superiority of 3DGS in rendering quality and
feature matching performance but also provides strong support for its future
application in complex scenarios.
(a) MVS-based rendering
(b) 3DGS-based rendering
Figure 9: The illustration of different rendering solutions in feature matching.
4.5. Evaluation of aerial and ground feature matching
With the aid of intermediate views, feature matching can then be con-
ducted for aerial and ground images, which alleviates viewpoint changes and
transmits feature correspondences. By using 12 image pairs selected from
three datasets as shown in Figure 6, this section analyzes the performance
21

<!-- page 22 -->
Table 5: Quantitative results on image pair 9. The values indicate NCM and NIM.
Image rendering
SIFTGPU
SuperPoint & SuperGlue
Aerial
Ground
Aerial
Ground
MVS-based
40/67
20/35
1213/1580
490/603
3DGS-based
928/965
59/78
3159/3214
984/1065
of the proposed feature matching algorithm in terms of NCM and NIM. For
performance comparison, two widely used solutions, i.e., the classical algo-
rithm SIFTGPU and the deep learning network SuperPoint & SuperGlue,
have also been evaluated.
Table 6 presents the quantitative results on the selected image pairs, in
which there are 4 images selected from each dataset. It is clearly shown that
because of large viewpoint changes, SIFTGPU failed in feature matching
for many image pairs, e.g., 1, 2, and 4 in CENTER and 5, 6, and 7 in
ZECHE, and 10 in SWJTU-LIB. The main reason is that there are very
few initial matches that can be found from aerial and ground images with
large appearance differences, as illustrated in Figure 10(a) and Figure 11(a).
By using deep learning-based feature detection and description, SuperPoint
& SuperGlue obviously increases the performance of feature matching for
aerial and ground images, especially for CENTER and SWJTU-LIB, when
compared with SIFTGPU. However, for dataset ZECHE, feature matching
still failed in image pairs 5, 6, and 7. The main reason is the extremely
large viewpoint changes, including the perspective distortion and appearance
rotation, as shown in Figure 11(b).
On the contrary, for the proposed algorithm, reliable feature matching
can be achieved for all image pairs. When compared with SuperPoint &
SuperGlue, the metrics NCM and NIM are obviously better, especially for
dataset ZECHE. The main reason is the usage of intermediate views. For
further analysis, Figure 10(c), Figure 11(c), and Figure 12(c) show rendered
images, feature matching results of render-aerial and render-ground image
pairs, respectively, and the final matches are presented in Figure 10(d), Fig-
ure 11(d), and Figure 12(d), respectively. It is shown that intermediate views
can bridge the gap between aerial and ground images and increase their fea-
ture matching performance. Noticeably, the dimension of rendered images
is almost the same as the image size used in SuperPoint for feature detec-
tion. In conclusion, the proposed algorithm can obviously increase feature
matching of aerial and ground images.
22

<!-- page 23 -->
(a) SIFTGPU (0/27)
(b) SuperPoint & SuperGlue (143/236)
(c) Intermediate view aided feature matching
(d) Ours (845/2,074)
Figure 10: The illustration of aerial-ground matching results (image pair 1) for dataset
CENTER. The values in the bracket indicate the metric NCM and NIM, respectively.
23

<!-- page 24 -->
(a) SIFTGPU (0/20)
(b) SuperPoint & SuperGlue (0/8)
(c) Intermediate view aided feature matching
(d) Ours (108/924)
Figure 11: The illustration of aerial-ground matching results (image pair 5) for dataset
ZECHE. The values in the bracket indicate the metric NCM and NIM, respectively.
24

<!-- page 25 -->
(a) SIFTGPU (146/173)
(b) SuperPoint & SuperGlue (820/1,074)
(c) Intermediate view aided feature matching
(d) Ours (2,547/13,057)
Figure 12: Aerial-ground matching results (image pair 9) for dataset SWJTU-LIB. The
values in the bracket indicate the metric NCM and NIM, respectively.
25

<!-- page 26 -->
(a)
(b)
Figure 13: The statistic of aerial-ground feature matching results in terms of NCM and
NMP.
26

<!-- page 27 -->
Table 6: Quantitative results on selected image pairs. The values indicate NCM and NIM.
Image pair
SIFTGPU
SuperPoint & SuperGlue
Ours
1
0/27
143/236
845/2,074
2
0/0
105/144
223/626
3
24/45
240/438
714/2,001
4
0/35
0/0
174/404
5
0/20
0/8
108/924
6
0/0
0/0
216/1,813
7
0/0
0/0
183/856
8
79/109
145/3,054
326/558
9
146/173
820/1,074
2,547/13,057
10
0/16
57/101
182/663
11
42/68
400/483
1,268/4,035
12
35/50
225/333
669/1,282
To evaluate the overall performance of the proposed algorithm, feature
matching is then conducted for aerial-ground image pairs of these three
datasets.
Because of the large viewpoint changes, it is hard to retrieval
correct match pairs based on image retrieval. In this evaluation, image pairs
are selected by using an exhaustive way.
In addition, two metrics NCM
and NMP are used for performance comparison. The former indicates the
number of correct matches after RANSAN-based outlier removal; the latter
represents the number of match pairs that have at least 15 correct matches,
which would be used to guide subsequent ISfM reconstruction.
Figure 13 presents the statistical results of aerial-ground feature match-
ing for the three datasets. It is clearly shown that for these two metrics, the
proposed algorithm achieves better performance when compared with other
two methods. For the metric NCM, the increase ratio of the proposed algo-
rithm is 3.72, 5.35, and 0.81 for the three datasets, respectively, compared
with SuperPoint & SuperGlue. It indicates that many more matches can be
found as shown in Figure 13(a), which further leads to more validated match
pairs in Figure 13(b). Therefore, the proposed algorithm can provide reliable
matches for aerial and ground images.
4.6. Application of matching result in 3DGS rendering
ISfM reconstruction can be executed to create sparse models for scene
rendering based on 3D Gaussian Splatting using feature matching results of
27

<!-- page 28 -->
aerial and ground images. The input of 3D Gaussian Splatting is a sparse
model with point clouds for Gaussian initialization and oriented images for
scene training. In this section, ISfM is utilized to resume camera poses and
3D points. Table 7 presents the statistical results of sparse reconstruction in
the terms of NRI and N3P, which indicates the number of registered images
and resumed 3D points, respectively. We can see that by using the matching
results, all images in three datasets can be successfully registered to create
entire sparse models, as shown in Figure 14.
Table 7: Results for the three datasets on the sparse reconstruction.
Dataset
NRI
N3P
Aerial
Ground
CENTER
146/146
203/203
245,623
ZECHE
172/172
147/147
324,683
SWJTU-LIB
123/123
78/78
142,408
(a) CENTER
(b) ZECHE
(c) SWJTU-LIB
Figure 14: SfM reconstruction of the three datasets. The red rectangles indicate oriented
images, and 3D points are rendered by image colors.
By using the reconstructed sparse models, 3D Gaussian Splatting can be
utilized to render photorealistic scenes. Compared with classical SfM-MVS
based pipeline, 3D Gaussian Splatting can render scenes with high precision
and efficiency. To compare the effect of integrating aerial and ground im-
ages, two individual tests are conducted in this section. The first one uses
aerial images only for scene rendering; the second one uses both aerial and
ground images for scene rendering. Both time cost and render quality are
evaluated in this section. Table 8 lists the time cost of scene rendering for
two configurations. It is clearly shown that less time costs are consumed
by the configuration with aerial-ground images when compared with that
28

<!-- page 29 -->
with aerial images only, which is 14.0 min, 23.0 min, and 21.0 min for the
three datasets. The main reason is that by using both aerial and ground
images, 3D Gaussian Splatting can converge rapidly with required precision,
especially for the near-ground regions that can not be observed enough from
aerial images. Figure 15 compares the render effects from the scene gener-
ated with varying configurations. We can see that photorealistic scenes can
be created by integrating both aerial and ground images based on 3D Gaus-
sian Splatting. Thus, the proposed algorithm can provide reliable feature
matches to achieve accurate and complete scene rendering.
Table 8: The time cost of scene rendering for two configurations (unit in minutes).
Dataset
Aerial images only
Aerial-ground images
CENTER
20.0
14.0
ZECHE
25.0
23.0
SWJTU-LIB
28.0
21.0
In addition to rendering, we also evaluated the geometric accuracy. Us-
ing cloud-to-cloud comparison, we assessed the relative Euclidean distance
between the generated point clouds and the reference data, providing a quan-
titative measure of the accuracy of different reconstruction pipelines. Consid-
ering that the CENTER and ZECHE datasets contain LiDAR data, we used
them as the ground truth data (GT), while the SWJTU-LIB dataset does
not include LiDAR data and thus could not be evaluated. For each dataset,
we generated dense point clouds using the SfM-MVS workflow (ColMap), as
well as the previously obtained 3DGS point clouds, and then registered them
to the GT (Figure 16). Finally, as shown in Table 9, we extracted metrics. In
the illustrations, the left side shows the results from the MVS pipeline, and
the right side shows those from the 3DGS pipeline. The color scale indicates
the distance metric, with warmer colors representing greater distances from
the reference data. The table lists the quantitative results, including the
maximum distance, mean distance, and standard deviation for each dataset
and pipeline. These metrics are crucial for assessing the precision and con-
sistency of the reconstruction methods.
Specifically, Figure 16, the visualization from the MVS pipeline shows
more blue areas, indicating smaller distances, and the Table 9 also shows
that the MVS mean distances are 0.76 cm and 0.63 cm, lower than 3DGS’s
1.12 cm and 0.78 cm. This is because the original 3DGS only constrains color,
29

<!-- page 30 -->
Aerial images only
Aerial-ground images
(a) CENTER
(b) ZECHE
(c) SWJTU-LIB
Figure 15: The illustration of 3DGS scene rendering with varying configurations.
Table 9: Metrics of cloud-to-cloud comparisons (unit in cm).
Dataset
Pipeline
Max Distance
Mean distance
Standard deviation
CENTER
MVS
29.66
0.76
0.96
3DGS
11.39
1.12
0.88
ZECHE
MVS
46.45
0.63
1.12
3DGS
16.58
0.78
0.67
30

<!-- page 31 -->
MVS
3DGS
(a) CENTER
(b) ZECHE
Figure 16: The illustration of cloud-to-cloud comparisons for both MVS and 3DGS meth-
ods with respect to the ground truth data (unit in cm).
31

<!-- page 32 -->
so it is not as geometrically accurate as MVS overall. However, 3DGS has
a clear advantage in the maximum distance and standard deviation metrics,
with the MVS pipeline’s visualization showing very noticeable red areas.
For the CENTER dataset, the MVS pipeline’s maximum distance is 29.66
cm, with a standard deviation of 0.96; while the 3DGS pipeline reduces
this to 11.39 cm and 0.88.
On the ZECHE dataset, the MVS pipeline’s
maximum distance is 46.45 cm, with a standard deviation of 1.12; in contrast,
the 3DGS pipeline significantly reduces this to 16.58 cm and 0.67.
This
indicates that the 3DGS method is more robust when dealing with datasets
that have complex structures or varying lighting conditions, and in the future,
geometric constraints can be added to 3DGS to achieve more accurate 3D
modeling.
5. Discussions
(1) Advantages of Integrating Aerial and Ground Images
Compared with using aerial images alone, the integration of aerial and
ground images offers advantages in computational time, reconstruction ac-
curacy, and completeness of scene coverage. Aerial images provide broad
coverage but lack detail, performing well for flat areas yet poorly for ver-
tical surfaces. Ground images supply high-resolution details, especially for
vertical structures, compensating for the deficiencies of aerial images. This
combination enables high-precision, all-around reconstruction while reducing
costs and improving data acquisition efficiency in small areas.
(2) Challenges of 3D Gaussian Splatting in Large-Scale Reconstructions
Though efficient, 3D Gaussian Splatting faces challenges in large-scale
reconstructions due to computational complexity and memory usage. Large
scenes require numerous Gaussian points for detail representation, con-
strained by GPU memory. Global optimization demands extensive iterations,
which are time-consuming and prone to instability. Complex lighting and im-
age variations further complicate the process. Future research will explore
strategies like block-wise reconstruction, incorporating prior knowledge, and
parallel processing to address these issues.
(3) Error Sources in the Reconstruction Process
Errors in the reconstruction process are complex, especially regarding
feature matching reliability and depth estimation accuracy when rendering
intermediate views. For feature matching, intermediate views may not fully
32

<!-- page 33 -->
eliminate perspective differences, leading to inaccurate feature point extrac-
tion or mismatches, especially in areas with sparse or repetitive textures.
Rendering quality issues can also affect feature point positioning. For depth
estimation, information loss, lighting differences, and accumulated errors dur-
ing rendering can result in inaccurate depth estimation.
6. Conclusions
For reliable feature matching between aerial and ground images, this
study proposes an intermediate view rendering based solution. Similar to
view rendering-based methods, the core idea is to use intermediate views to
alleviate viewpoint changes of aerial and ground images. The main workflow
consists of: (1) SfM reconstruction from aerial images; (2) intermediate view
rendering using 3D Gaussian Splatting; and (3) reliable feature matching
with the aid of intermediate view. The validation of the proposed solution
has been verified by using real datasets and compared with other widely used
methods. The results demonstrate that the proposed algorithm can provide
reliable feature matches for aerial and ground images and can achieve accu-
rate and complete scene rendering for 3D Gaussian Splatting. In future work,
two major tests can be further conducted. On the one hand, the improved
version of 3DGS can be used and verified in aerial-ground image matching;
on the other hand, mesh models can be extracted from rendered scenes for
quantitative precision evaluation.
Disclosure statement
No potential conflict of interest was reported by the author(s).
Data availability statement
Data is not available due to ethical and legal restrictions.
Acknowledgement
This research was funded by the National Natural Science Foundation of
China (Grant No. 42371442), the Hubei Provincial Natural Science Founda-
tion of China (Grant No. 2023AFB568).
33

<!-- page 34 -->
References
Besl, P.J., McKay, N.D., 1992. Method for registration of 3-d shapes, in:
Sensor fusion IV: control paradigms and data structures, Spie. pp. 586–
606.
Chen, H., Luo, Z., Zhou, L., Tian, Y., Zhen, M., Fang, T., Mckinnon, D.,
Tsin, Y., Quan, L., 2022. Aspanformer: Detector-free image matching
with adaptive span transformer, in: European Conference on Computer
Vision, Springer. pp. 20–36.
DeTone, D., Malisiewicz, T., Rabinovich, A., 2018.
Superpoint:
Self-
supervised interest point detection and description, in: Proceedings of the
IEEE conference on computer vision and pattern recognition workshops,
pp. 224–236.
Dusmanu, M., Rocco, I., Pajdla, T., Pollefeys, M., Sivic, J., Torii, A., Sattler,
T., 2019.
D2-net: A trainable cnn for joint description and detection
of local features, in: Proceedings of the ieee/cvf conference on computer
vision and pattern recognition, pp. 8092–8101.
Edstedt, J., Sun, Q., B¨okman, G., Wadenb¨ack, M., Felsberg, M., 2024. Roma:
Robust dense feature matching, in: Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pp. 19790–19800.
Fu, Y., Liu, S., Kulkarni, A., Kautz, J., Efros, A.A., Wang, X., 2024. Colmap-
free 3d gaussian splatting, in: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 20796–20805.
Gao, X., Hu, L., Cui, H., Shen, S., Hu, Z., 2018a. Accurate and efficient
ground-to-aerial model alignment. Pattern Recognition 76, 288–302.
Gao, X., Shen, S., Hu, Z., Wang, Z., 2019. Ground and aerial meta-data inte-
gration for localization and reconstruction: A review. Pattern Recognition
Letters 127, 202–214.
Gao, X., Shen, S., Zhou, Y., Cui, H., Zhu, L., Hu, Z., 2018b. Ancient chinese
architecture 3d preservation by merging ground and aerial point clouds.
ISPRS Journal of Photogrammetry and Remote Sensing 143, 72–84.
34

<!-- page 35 -->
Ge, Y., Guo, B., Xu, G., Liu, Y., Jiang, X., Peng, Z., 2024.
Rapid 3d
modelling: Clustering method based on dynamic load balancing strategy.
The Photogrammetric Record 39, 67–86.
Hong, S., Jung, J., Shin, H., Han, J., Yang, J., Luo, C., Kim, S., 2024.
Pf3plat:
Pose-free feed-forward 3d gaussian splatting.
arXiv preprint
arXiv:2410.22128 .
Hu, H., Zhu, Q., Du, Z., Zhang, Y., Ding, Y., 2015. Reliable spatial rela-
tionship constrained feature point matching of oblique aerial images. Pho-
togrammetric Engineering & Remote Sensing 81, 49–58.
Jhan, J.P., Kerle, N., Rau, J.Y., 2021. Integrating uav and ground panoramic
images for point cloud analysis of damaged building. IEEE geoscience and
remote sensing letters 19, 1–5.
Ji, S., Zeng, C., Zhang, Y., Duan, Y., 2023. An evaluation of conventional
and deep learning-based image-matching methods on diverse datasets. The
Photogrammetric Record 38, 137–159.
Jiang, S., Jiang, C., Jiang, W., 2020. Efficient structure from motion for
large-scale uav images: A review and a comparison of sfm tools. ISPRS
Journal of Photogrammetry and Remote Sensing 167, 230–251.
Jiang, S., Jiang, W., 2017. On-board gnss/imu assisted feature extraction
and matching for oblique uav images. Remote Sensing 9, 813.
Jiang, S., Jiang, W., Guo, B., Li, L., Wang, L., 2021. Learned local features
for structure from motion of uav images: A comparative evaluation. IEEE
Journal of Selected Topics in Applied Earth Observations and Remote
Sensing 14, 10583–10597.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., Drettakis, G., 2023. 3d gaussian
splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
139–1.
Li, Q., Huang, H., Yu, W., Jiang, S., 2023a. Optimized views photogram-
metry: Precision analysis and a large-scale case study in qingdao. IEEE
Journal of Selected Topics in Applied Earth Observations and Remote
Sensing 16, 1144–1159.
35

<!-- page 36 -->
Li, Z., Wu, B., Li, Y., Chen, Z., 2023b. Fusion of aerial, mms and backpack
images and point clouds for optimized 3d mapping in urban areas. ISPRS
Journal of Photogrammetry and Remote Sensing 202, 463–478.
Liang, Y., Yang, Y., Mu, Y., Cui, T., 2023. Robust fusion of multi-source
images for accurate 3d reconstruction of complex urban scenes. Remote
Sensing 15, 5302.
Liu, J., Yin, H., Liu, B., Lu, P., 2023. Tie point matching between terrestrial
and aerial images based on patch variational refinement. Remote Sensing
15, 968.
Luo, Z., Shen, T., Zhou, L., Zhang, J., Yao, Y., Li, S., Fang, T., Quan,
L., 2019. Contextdesc: Local descriptor augmentation with cross-modality
context, in: Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pp. 2527–2536.
Luo, Z., Shen, T., Zhou, L., Zhu, S., Zhang, R., Yao, Y., Fang, T., Quan,
L., 2018.
Geodesc: Learning local descriptors by integrating geometry
constraints, in: Proceedings of the European conference on computer vision
(ECCV), pp. 168–183.
Luo, Z., Zhou, L., Bai, X., Chen, H., Zhang, J., Yao, Y., Li, S., Fang, T.,
Quan, L., 2020. Aslfeat: Learning local features of accurate shape and
localization, in: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 6589–6598.
Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R.,
Ng, R., 2021. Nerf: Representing scenes as neural radiance fields for view
synthesis. Communications of the ACM 65, 99–106.
Mishchuk, A., Mishkin, D., Radenovic, F., Matas, J., 2017. Working hard to
know your neighbor’s margins: Local descriptor learning loss. Advances
in neural information processing systems 30.
Nex, F., Gerke, M., Remondino, F., Przybilla, H.J., B¨aumker, M., Zurhorst,
A., 2015. Isprs benchmark for multi-platform photogrammetry. ISPRS
Annals of the Photogrammetry, Remote Sensing and Spatial Information
Sciences 2, 135–142.
36

<!-- page 37 -->
Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A., 2020. Superglue:
Learning feature matching with graph neural networks, in: Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 4938–4947.
Schonberger, J.L., Frahm, J.M., 2016. Structure-from-motion revisited, in:
Proceedings of the IEEE conference on computer vision and pattern recog-
nition, pp. 4104–4113.
Shan, Q., Wu, C., Curless, B., Furukawa, Y., Hernandez, C., Seitz, S.M.,
2014. Accurate geo-registration by ground-to-aerial image matching, in:
2014 2nd International Conference on 3D Vision, IEEE. pp. 525–532.
Song, W.H., Jung, H.G., Gwak, I.Y., Lee, S.W., 2019. Oblique aerial im-
age matching based on iterative simulation and homography evaluation.
Pattern Recognition 87, 317–331.
Sun, J., Shen, Z., Wang, Y., Bao, H., Zhou, X., 2021. Loftr: Detector-free lo-
cal feature matching with transformers, in: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pp. 8922–8931.
Tian, Y., Fan, B., Wu, F., 2017. L2-net: Deep learning of discriminative
patch descriptor in euclidean space, in: Proceedings of the IEEE conference
on computer vision and pattern recognition, pp. 661–669.
Wang, Q., Zhang, J., Yang, K., Peng, K., Stiefelhagen, R., 2022. Match-
former: Interleaving attention in transformers for feature matching, in:
Proceedings of the Asian Conference on Computer Vision, pp. 2746–2762.
Wang, X., Xiang, H., Niu, W., Mao, Z., Huang, X., Zhang, F., 2023. Oblique
photogrammetry supporting procedural tree modeling in urban areas. IS-
PRS Journal of Photogrammetry and Remote Sensing 200, 120–137.
Wu, B., Xie, L., Hu, H., Zhu, Q., Yau, E., 2018. Integration of aerial oblique
imagery and terrestrial imagery for optimized 3d modeling in urban areas.
ISPRS journal of photogrammetry and remote sensing 139, 119–132.
Wu, C., Clipp, B., Li, X., Frahm, J.M., Pollefeys, M., 2008. 3d model match-
ing with viewpoint-invariant patches (vip), in: 2008 IEEE Conference on
Computer Vision and Pattern Recognition, IEEE. pp. 1–8.
37

<!-- page 38 -->
Xu, S., Chen, S., Xu, R., Wang, C., Lu, P., Guo, L., 2024. Local feature
matching using deep learning: A survey. Information Fusion 107, 102344.
Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.H., Peng, S., 2024.
No pose, no problem: Surprisingly simple 3d gaussian splats from sparse
unposed images. arXiv preprint arXiv:2410.24207 .
Zhu, B., Ye, Y., Dai, J., Peng, T., Deng, J., Zhu, Q., 2024. Vdft: Robust
feature matching of aerial and ground images using viewpoint-invariant
deformable feature transformation.
ISPRS Journal of Photogrammetry
and Remote Sensing 218, 311–325.
Zhu, Q., Wang, Z., Hu, H., Xie, L., Ge, X., Zhang, Y., 2020. Leveraging
photogrammetric mesh models for aerial-ground feature point matching
toward integrated 3d reconstruction. ISPRS Journal of Photogrammetry
and Remote Sensing 166, 26–40.
38
