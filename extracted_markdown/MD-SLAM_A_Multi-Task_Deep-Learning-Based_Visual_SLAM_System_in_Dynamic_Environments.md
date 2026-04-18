<!-- page 1 -->
11882
IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 8, AUGUST 2025
MD-SLAM: A Multi-Task Deep-Learning-Based
Visual SLAM System in Dynamic Environments
Ying He
, Senior Member, IEEE, Xinyu Zeng, F. Richard Yu
, Fellow, IEEE, Zhengjun Zhong,
Zhiquan Liu
, Senior Member, IEEE, and Guang Zhou
Abstract—With the development of self-driving vehicles and
intelligent robots, visual simultaneous localization and mapping
(SLAM) has attracted signiﬁcant attention. In dynamic environ-
ments, non-stationary objects can cause performance degradation
of visual SLAM systems. The inaccurate pose and lack of seman-
ticinformation in feature points can lead to incorrect differentiation
between static and dynamic points, resulting in a degradation of
the visual SLAM system’s performance. In this paper, we propose
a novel visual SLAM system based on multi-task deep neural
networks to address this issue. Speciﬁcally, we apply multi-task
deep neural networks to extract higher quality and more robust
oriented local features and perceive dynamic semantic regions,
which are used to better remove dynamic points spatially and
potential dynamic points semantically in the SLAM system. We
evaluate our method on public datasets, and the results show that
our method outperforms existing visual SLAM systems.
Index
Terms—Visual
SLAM,
deep
learning,
dynamic
environments, multi-task neural network.
I. INTRODUCTION
S
IMULTANEOUS localization and mapping (SLAM) is a
crucial technology for self-driving vehicles and intelligent
robots to plan and navigate autonomously in an unknown en-
vironment. In recent years, there has been a notable surge in
the development and adoption of camera-based visual SLAM
systems. These systems offer a compelling alternative to tra-
ditional laser-based SLAM systems, which often necessitate
costly equipment. Examples of such camera-based visual SLAM
systems include ORB-SLAM2 [1] and DSO [2].
A predominant assumption underlying most current visual
SLAM frameworks is the static nature of the environment. This
Received 20 October 2024; revised 21 December 2024; accepted 6 Jan-
uary 2025. Date of publication 18 March 2025; date of current version
15 August 2025. This work was supported in part by Shenzhen Science
and Technology Program under Grant KQTD20210811150132001 and Grant
ZDSYS20220527171400002, in part by the National Natural Science Founda-
tion of China (NSFC) under Grant 62271324, Grant 62231020, Grant 62371309,
and Grant 62394335, and in part by Guangdong Basic and Applied Basic
Research Foundation under Grant 2023A1515011979. The review of this article
was coordinated by Dr. Bo Yu. (Corresponding author: Guang Zhou.)
Ying He, Xinyu Zeng, and Zhengjun Zhong are with the College of Computer
Science and Software Engineering, Shenzhen University, Shenzhen 518060,
China (e-mail: heying@szu.edu.cn).
F. Richard Yu is with the School of Information Technology, Carleton Uni-
versity, Ottawa, ON K1S 5B6, Canada (e-mail: richard.yu@carleton.ca).
Zhiquan Liu is with Jinan University, Guangzhou 510632, China (e-mail:
zqliu@jnu.edu.cn).
Guang Zhou is with DeepRoute Inc., Shenzhen 518116, China (e-mail:
maxwell@deeproute.ai).
Digital Object Identiﬁer 10.1109/TVT.2025.3552541
assumption plays a critical role in stabilizing the localization and
mapping processes, simplifying real-world settings’ dynamic
complexities. However, this static environment hypothesis often
falls short of reﬂecting the dynamic and ever-changing nature
of real-world scenarios, posing challenges to the robustness and
adaptability of SLAM systems in practical applications. There-
fore, it is crucial to identify and process dynamic information in
the environment.
The methods of visual SLAM for perceiving environmental
information can be broadly divided into two approaches: direct
methods and feature-based methods. direct methods leverage the
raw pixel data from entire images, aiming for a more comprehen-
sive utilization of available visual information. Direct methods
are sensitive to lighting changes and image noise, which can
reduce their accuracy in challenging environments. On the other
hand, feature-based methods emphasize the extraction of sparse,
distinctive features from the environment, employing algorithms
designed to be computationally efﬁcient. These methods often
utilize handcrafted features—such as ORB [3], which, despite
their computational speed, may lack the nuanced capability to
characterize local features thoroughly. For feature-based SLAM
systems, a common thread is their reliance on the handcrafted
features. This reliance underscores a trade-off between com-
putational efﬁciency and the depth of keypoint characteriza-
tion. ORB, for example, is praised for its balance of speed
and effectiveness in feature detection and matching but has
limitations in capturing the full complexity of local features
in diverse environments. In recent years, researchers have tried
to introduce deep learning into visual SLAM systems. Several
works have demonstrated that learning-based systems, such as
SuperPoint [4] and Hf-Net [5], can improve the localization
ability of SLAM systems.
Although some excellent works have been done on tradi-
tional and deep-learning-based visual SLAM systems, most
of existing works only consider the reliability of each pixel
and do not consider the pixel orientation, which is critical for
identifying dynamic points in the environment. Additionally, the
local features are not differentiated in the face of the dynamic
environment, and the assumption of the static environment is
still maintained. Furthermore, they also fail to identify potential
dynamic information in the environment, which can similarly
affect the robustness of the SLAM system.
In order to perceive and process dynamic points in the envi-
ronment, in this paper, we design a multi-task neural network
and propose a novel visual SLAM system, MD-SLAM, based on
0018-9545 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artiﬁcial intelligence and similar technologies.
Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 2 -->
HE et al.: MD-SLAM: A MULTI-TASK DEEP-LEARNING-BASED VISUAL SLAM SYSTEM IN DYNAMIC ENVIRONMENTS
11883
this network. The multi-task neural network can perform both
semantic segmentation and feature extraction tasks. Introducing
multi-task neural network can not only extract higher quality and
more robust local features for visual SLAM systems, but also
better identify dynamic points spatially and potential dynamic
points semantically. Our contributions can be summarized as
follows:
r We propose a novel visual SLAM system based on a
multi-task neural network to address the problems faced
by traditional visual SLAM systems, including the decline
in local feature quality and the difﬁculty in recognizing
dynamic information in dynamic environments.
r To address the issue of traditional visual SLAM systems
being unable to identify dynamic objects, we design a
semantic segmentation task branch in the multi-task neural
network to extract semantic categories at the pixel level
of the image to identify dynamic objects and eliminate
them, thereby improving the pose estimation accuracy of
the system.
r To address the drift issue caused by the self-motion of
visual SLAM systems, we design a feature extraction task
branch within the multi-task neural network that extracts
features with orientation information. This branch is re-
sponsible for removing a collection of feature points from
images, thereby enhancing the feature-matching accuracy
of the visual SLAM system.
r We evaluate MD-SLAM on the public datasets, and the
results illustrate the superior performance of our approach
compared with traditional visual SLAM systems.
In the rest of this paper, we introduce the related work about
deep learning in visual SLAM and dynamic SLAM systems in
Section II. Then, we present our multi-task neural network in
Section III and visual SLAM system in Section IV. Section V
discusses the experimental results of our method on public
datasets. Section VI concludes this study with future work.
II. RELATED WORK
In this section, we ﬁrst introduce the development of tradi-
tional visual SLAM systems. Then, we discuss the application
of deep learning in visual SLAM systems. Lastly, we explore the
related work on visual SLAM systems in dynamic environments.
A. Traditional Visual SLAM
Previous work [1], [6], [7], [8], [9] suggested a number
of feature-based approaches. A representative work is ORB-
SLAM3 [9], which uses ORB features to signiﬁcantly speed up
local feature extraction compared to the FAST corner detector. It
integrates support for RGBD and stereo cameras, and promotes
the fusion of visual and inertial data through the Inertial Mea-
surement Unit (IMU), enhancing performance and robustness in
dynamic environments. Additionally, ORB-SLAM3 introduced
a novel multiple-map concept, enabling the creation of inde-
pendent maps in various areas that could be merged, enhancing
its suitability for large-scale environments. In addition to local
feature-based keyframe methods [2], [10], [11], visual SLAM
systems incorporate approaches like the direct method and op-
tical ﬂow to enhance performance. A representative work is the
VINS-MONO [10] system from the Hong Kong University of
Science and Technology exempliﬁes this by integrating optical
ﬂow data and IMU readings for more accurate trajectory esti-
mation.
B. Enhancing Local Features With Deep Learning
The rapid evolution of deep learning has signiﬁcantly im-
pacted various ﬁelds [12], [13], [14], with its integration into
visual SLAM systems marking a pivotal advancement. Deep
learning’s robust capabilities in data preprocessing and abstrac-
tion have made it an invaluable tool for addressing some of the
most challenging problems in SLAM, such as data association.
The quest for more robust local feature extraction in feature-
based SLAM systems has led researchers to explore and in-
tegrate advanced deep learning models. This exploration is
driven by the need to enhance SLAM performance, especially
in challenging environments where traditional feature extraction
methods may falter. GCNv2-SLAM [15] represents a signiﬁcant
leap forward by employing a CNN network designed to learn
the coordinates of local features along with their descriptors in
a binary format, reminiscent of BRIEF [16] descriptors. This
method offers an innovative solution by combining the efﬁ-
ciencyandcompactnessoftraditionalbinarydescriptorswiththe
learning capability of CNNs. The result is a system that retains
the real-time performance required for applications like mobile
unmanned aerial vehicles (UAVs) and improves the robustness
and accuracy of feature extraction and matching in varied en-
vironments. RWT-SLAM [17] addresses a different yet equally
challenging aspect of feature-based SLAM: feature matching
in environments with low texture, such as plain white walls.
By incorporating LoFTR [18], a Transformer-based method,
RWT-SLAM enhances the SLAM system’s ability to perform
accurate feature matching under conditions where traditional
methods struggle. Transformer [19], known for its effectiveness
in capturing long-range dependencies in data, offers a signiﬁcant
advantage in identifying and matching local features across
images, even in texture-sparse scenes.
RI-LBD [20] employs a learning-based method to obtain
local binary descriptors. It jointly learns the projection matrix
and each pattern’s orientation to obtain rotationinvariant local
binary descriptors. SuperPoint [21] presents a self-supervised
framework for training interest point detectors and descriptors
suitable for a large number of multiple-view geometry problems
in computer vision. RoRD [22] proposes rotation-robust local
descriptors learned through training data augmented with rota-
tion homographies. This approach enhances place recognition
accuracy across drastically different viewpoints and maintains
high performance even under extreme viewpoint changes. Fea-
tureBooster [23] introduce a lightweight network to improve
descriptors of keypoints within the same image. The network
takes the original descriptors and the geometric properties of
keypoints as the input, and uses an MLP-based self-boosting
stage and a Transformer-based cross-boosting stage to enhance
the descriptors.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 3 -->
11884
IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 8, AUGUST 2025
Fig. 1.
The structure of our proposed multi-task deep neural network. We employ a shared rotation-invariant encoder to extract features from pyramid images.
Then we design a semantic segmentation task branch to identify potential dynamic points. And in the deep local feature extraction task branch, we extract features’
coordinate information and corresponding orientation information simultaneously.
C. Visual SLAM in Dynamic Environments
One assumption of the current mainstream SLAM systems is
that the objects are stationary in static environments. However,
this assumption often does not hold in the real world, and the
presence of dynamic objects in the scene can affect the localiza-
tion and the construction of SLAM systems. It is necessary to
remove the dynamic objects to address this issue.
CoSLAM [24] introduces camera pose estimation and cam-
era mapping using multiple cameras to handle dynamic en-
vironments and calculates the reconstruction projection error
of local features in a sliding frames window to distinguish
between dynamic and static key points. Alcantarilla et al. [25]
propose to improve the robustness of the visual SLAM system
by introducing optical ﬂow for dense scenes. The pixel-level
motion estimation of the images can be performed to classify
the dynamic and static local features using the optical ﬂow
map. However, mathematical motion estimation of local features
cannot accurately determine the nature of local features, such as
semantic information. Thus, a more accurate classiﬁer is needed.
DynaSLAM [26], implemented on ORB-SLAM2, uses an
instance segmentation detector, Mask-RCNN [27], with a back-
ground inpainting module to reduce the impact of dynamic
objects. It uses two methods to detect the moving regions:
multi-view geometry and deep learning. DS-SLAM [28], also
built on ORB-SLAM2, uses a semantic segmentation network
SegNet [29] with a moving consistency check to remove the
dynamic objects and generate a semantic octree map. All the
above methods use a blocked module, and they must wait for the
semantic map after extracting the local features, which costs a lot
oftime.RDS-SLAM[30]addsaseparatesemanticsegmentation
thread to the ORB-SLAM3 [9], which adds support for IMU
devices to ORB-SLAM2 with a more accurate map module.
DP-SLAM [31] combines the results of geometry constraints
and semantic segmentation to track the dynamic keypoints in a
Bayesian probability estimation framework. Blitz-SLAM [32]
removes noise blocks in the local point cloud by leveraging the
advantages of semantic and geometric information from mask,
RGB, and depth images. This approach enables the generation
of a global point cloud map by merging the local point clouds
effectively. These methods all focus on processing dynamic
environments using classic ORB features, and the extraction of
local features is decoupled from dynamic object perception.
III. THE PROPOSED MULTI-TASK NEURAL NETWORK
In this section, we ﬁrst introduce the structure of the proposed
multi-task neural network. Then, we describe the loss functions
usedintrainingthenetwork.Thestructureofourproposedmulti-
task deep neural network as shown in Fig. 1.
A. Shared Rotation-Invariant Encoder
In existing vision SLAM systems that utilize deep local fea-
ture extractor [15], [21], [33], traditional CNNs are employed.
While traditional CNNs can handle the majority of static images
effectively, they face challenges with rotated images due to
the lack of rotational invariance in their sliding window-based
feature extraction mechanism. When the input image is rotated,
even the same object may result in signiﬁcantly different ex-
tracted features, thereby affecting the overall performance of
the model.
To address this, we choose a modiﬁed CNN based on rota-
tional invariance, E2CNN [34], as the shared feature encoder for
our multi-task neural network. E2CNN ensures that the feature
extractionprocessisunaffectedbyimagerotationbyintroducing
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 4 -->
HE et al.: MD-SLAM: A MULTI-TASK DEEP-LEARNING-BASED VISUAL SLAM SYSTEM IN DYNAMIC ENVIRONMENTS
11885
equivariance to the Euclidean group. The key idea of E2CNN
in 3D is to ensure that the convolution operation respects and
leverages the inherent symmetries in the data, such as rota-
tions and reﬂections, by using group convolutions. It replaces
traditional convolution operations with group convolutions that
simultaneously perform convolutions across spatial, rotational,
and reﬂection transformation dimensions. This approach en-
ables the encoder to consider features in different directions
and positions simultaneously, thereby preserving the orientation
information of the image.
In the speciﬁc implementation, we ﬁrst subject the image to
multi-scale scaling transformations to obtain pyramid images
of various sizes. This approach enhances the model’s scale
invariance and understanding of context. Following this, we
employ a shared rotation-invariant encoder to extract features
from these pyramid images, resulting in a feature pyramid.
This feature pyramid is then utilized in subsequent branches
for dynamic detection and depth feature extraction.
B. Dynamic Detection Task Branch
In the dynamic realms of real-world environments, the pres-
ence of moving objects poses signiﬁcant challenges to the accu-
racy and reliability of front-end feature matching and map point
determination in visual SLAM systems. For effective navigation
and mapping, it becomes imperative to identify and exclude
points associated with these moving entities. Semantic segmen-
tation, with its pixel-level precision, emerges as a particularly apt
solution for isolating dynamic areas and facilitating pixel-level
keypoint extraction, thereby enhancing the system’s capability
to discern static from dynamic elements.
The design and implementation of semantic segmentation
networks typically follow the standard encoder-decoder archi-
tecture, known for its efﬁciency in capturing and reconstructing
detailed semantic information. Within this framework, the de-
coder plays a crucial role in managing computational complexity
and ensuring the ﬁdelity of the semantic output. We employ
Depth-Wise and Point-Wise modules to construct the encoder.
These modules effectively simplify the computations in the con-
volution process without sacriﬁcing performance. Additionally,
we introduce an expansion module to enhance the receptive ﬁeld
oftheencoder,allowingforabroaderunderstandingofthescene.
A key challenge in semantic segmentation networks is the
potential loss of ﬁne-grained information due to the convolution
depth. Our decoder integrates a two-stage convolution within its
upsampling module. Initially, the output of the ﬁrst convolution
isenrichedbyaddingfeaturemapsfromthecorrespondingupper
layer of the decoder, effectively reintroducing lost details. Then,
this enriched output undergoes a second convolution, further
reﬁning the feature map and reducing information loss. Finally,
a convolutional network is used to output the ﬁnal classiﬁcation
results, and the Softmax function is applied to convert the
predictive output into a probability distribution.
C. Deep Feature Extraction Task Branch
In the deep local feature extraction task branch, we aim
to extract features’ coordinate information and corresponding
orientation information simultaneously. Therefore, this branch
includes two sub-modules: a coordinate extraction module and
an orientation extraction module.
In the coordinate extraction module, we ﬁrst perform group
pooling on the image feature pyramid features extracted by the
shared rotation-invariant encoder. This step helps reduce the
data’s dimensionality while preserving crucial spatial informa-
tion. Subsequently, these features are resized through interpola-
tion to match the resolution of the original image and effectively
fused along the channel dimension. Then, a convolutional layer
with a kernel size of 1 is used to reduce the dimensionality of the
fused features to obtain the feature point coordinate score map
S, where the values determine the quality of the corresponding
coordinate points.
Intheorientationextractionmodule,weﬁrstperformchannel-
wise pooling on the image feature pyramid features extracted by
the shared rotation-invariant encoder to reduce redundant infor-
mation and extract key orientation features. Then, interpolation
is used to ensure the size consistency of all features before per-
forming an accumulation operation. Finally, a CNN processes
thefusedfeaturedata,andtheSoftmaxmethodisusedtoproduce
the orientation feature map O. The number of channels in the O
corresponds to the system’s predetermined range of orientations.
For instance, setting each directional interval at 30◦divides
the 360◦spectrum into 12 distinct sectors, implying that the
orientation feature vector comprises 12 channels. Upon applying
Softmax, the channel within each feature point that exhibits the
highest value signiﬁes the deﬁnitive orientation information of
said point.
After extracting the oriented features, we also use a learning-
based method, HardNet [35], to extract the descriptors for each
feature. HardNet performs excellently in several vision tasks due
to its fast computation and unique triplet metric loss function,
which is as good as the SIFT [36] descriptors. HardNet can
generate 128-dimensional dense descriptor tensors for each
keypoint, reducing the computation time for subsequent feature
matching. Fig. 1 shows the multi-task DNN structure of our
proposed method.
D. Loss Function
1) Semantic Segmentation Task: We use different loss func-
tions for training based on different task contents. Cross-entropy
lossisusedfortraininginthesemanticsegmentationtaskbranch.
It is not only easy to optimize but it also can solve the problem
of category imbalance. Cross-entropy loss can also enhance the
model’s conﬁdence in correct classiﬁcation by maximizing the
tangent class probability and improving the prediction accuracy
of semantic segmentation models. Suppose the model’s output
is a probability distribution expressed as the probability of each
category as P(c|x), where c is the category, and x is the input
pixel or image region. For a single sample in a problem with C
categories, the cross-entropy loss can be deﬁned as:
Lce = −
C

c=1
yc log(P(c|x))
(1)
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 5 -->
11886
IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 8, AUGUST 2025
where yc is an indicator variable that equals 1 when the sample
belongs to class c and 0 otherwise; P(c|x) is the probability
predicted by the model that sample x belongs to class c. In image
segmentation tasks, it’s necessary to account for all pixels in the
image. For an image containing Np pixels, the cross-entropy
loss is calculated by aggregating the losses for each pixel across
the image:
Lse = −1
Np
Np

i=1
C

c=1
yi,c log(P(c|xi)
(2)
2) Deep Local Feature Coordinate Extraction Task: We con-
duct self-supervised for the deep local feature coordinate ex-
traction task, drawing on existing work [37]. During dataset
preparation, a random afﬁne transformation Tgt is applied to
image Ia to perform rotation and scaling, producing image Ib.
These two images, Ia and Ib, along with the transformation
matrix Tgt, are used as the input for network training. A sliding
window strategy is adopted for the coordinate extraction module
to accelerate training. By performing sliding sampling on the
score map S with a window size of Nl, and applying the Softmax
operation to each window, the response value of each pixel
within every window is obtained:
mi
u,v =
ewi
u,v
ci+Nl
j=ci
ci+Nl
k=ci ewi
j,k
(3)
where wi refers to the score values corresponding to the ith
Nw × Nw window, ci is the coordinate of the top-left corner
of the window, and u, v, j, k are the subscript indices for the
coordinates of the pixels involved. Subsequently, the response
value mi
u,v is used as a weight to perform a weighted sum of
all pixel values within the window, yielding a soft keypoint
coordinate for the corresponding window:
[xi, yi]T =

[u,v]∈wi
mi
u,v · [u, v]T
(4)
After extracting the soft coordinates, for each window, the
coordinates corresponding to the maximum score value are
directly selected as the hard coordinates [ˆx, ˆy] of that window.
Two sets of soft and hard coordinate collections are obtained
by performing the same operations on the score maps extracted
from the two input images. Subsequently, the transformation
matrix T is used to calculate the distance between all soft and
hard coordinates in the two images as index proposal loss Lip.
Finally, the coordinates extraction loss consists of the Lip for
different window sizes:
Lkp

Ia, Ib, Tgt

=
L

l=1
λl(Lip(Ia, Ib, Tgt, Nl)
+ Lip(Ib, Ia, T −1
gt , Nl))
(5)
whereNl isthewindowsize,Listhenumberofdifferentwindow
size and λi is a balance hyper parameters for each window level.
3) DeepLocalFeatureOrientationExtractionTask: Wetreat
the deep local feature orientation extraction task as a classiﬁca-
tion task for training. First, the obtained rotation feature map Ob
and Ob from Ia and Ib are aligned with each other using Tgt
and T ′
gt to get ˆOa and ˆOb, respectively. Then, the cross-entropy
calculation method is used to compute the distance in orientation
for all pixels as the optimization objective:
Lori = −
W

i=1
H

j=1
G

k=1
( ˆOa
kij) log( ˆOb
kij)
(6)
where G is the group number used in shared rotation-invariant
encoder. The ﬁnal loss function calculation for the deep feature
extraction task is as follows:
Lfeat = Lkp + Lori
(7)
IV. THE PROPOSED MULTI-TASK DEEP-LEARNING-BASED
VISUAL SLAM SYSTEM
In this section, we ﬁrst provide an overview of the proposed
multi-task deep-learning-based visual SLAM system. Then, we
describe the system’s tracking and loop closure modules.
A. Overview
Our SLAM system, MD-SLAM, is based on the ORB-
SLAM2 system, and Fig. 2 shows the proposed SLAM system
pipeline. We use a multi-task DNN extractor to give both local
keypoints, dense local descriptors and dynamic regions instead
of a traditional ORB detector with no dynamic objects outliers
rejection. Unlike other learning-based local features, the key-
point detector we used extracts oriented keypoints which greatly
help the feature matching between two frames. The orientation
outlierrejectionimprovestheperformanceofourSLAMsystem.
While extracting local features, the image will also be input
into the semantic segmentation branch to obtain the semantic
information of dynamic objects in the image to adapt to the
dynamic environments.
B. Tracking
The tracking module is the front-end part of the visual SLAM
system, and its performance is directly related to the system’s
adaptability in dynamic environments as well as its ability to
handle fast motion and complex scenes.
In MD-SLAM, we replace the original ORB feature extractor
with the multi-task neural network described in Section III to
extract local features and semantic labels from image frames.
During the initialization phase of the tracking module, the strat-
egy varies depending on the operation mode. For the monocular
mode, since the system cannot perceive depth information,
it performs feature extraction and matching on consecutive
frames and uses epipolar geometry to calculate the fundamental
matrix and the homography matrix between images, followed
by triangulation for pose initialization. In the RGBD mode,
depth information is directly used to initialize the local fea-
tures extracted from the ﬁrst frame. It is worth noting that
in the monocular mode, to facilitate rapid initialization when
the number of extracted local features is insufﬁcient, semantic
information ﬁltering is not performed.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 6 -->
HE et al.: MD-SLAM: A MULTI-TASK DEEP-LEARNING-BASED VISUAL SLAM SYSTEM IN DYNAMIC ENVIRONMENTS
11887
Fig. 2.
The pipeline of our proposed visual SLAM system. The colored part is our main improvement part. A shared Multi-Task DNN Extractor extracts dense
descriptors, keypoints, and dynamic regions from input images. In the Tracking phase, dynamic objects are removed, features are matched by dense descriptors,
and orientation outliers are rejected for accurate pose prediction. In the Loop Closing module, the retrained FBoW is used for searching.
After the initialization is completed, the system will track
and match subsequent input image frames one by one. At this
stage, MD-SLAM will perform additional outlier removal for
the matched local features. First, the input image undergoes fea-
ture extraction and semantic segmentation using the multi-task
neural network in the tracking module, resulting in a collection
of image local features P(x, o) along with the semantic labels
K for each local feature, where x represents the horizontal and
vertical coordinates of each local feature in the image while o
denotes the orientation information of that local feature. Based
on the semantic labels, the category to which a local feature
belongs can be determined. Deﬁne the label set of dynamic
scenes in the current scenario as κ, it is possible to directly
assess whether the category of a local feature belongs to κ to
identify whether the local feature is dynamic.
di =

1,
Ki ∈κ
0,
Ki /∈κ
(8)
If di = 1, then the local feature is considered an outlier and is
removed.
Subsequently, we match the sets of local features from the
two images after dynamic outlier removal. If the system is in
monocular mode and has not been initialized, the BFMatcher
algorithm is used for feature matching. If the system has already
been initialized, a constant velocity motion model is used to
construct local feature projection matching to accelerate the
matching speed of features. After obtaining the initial pairs of
matched local features, we apply orientation constraints to these
pairs using the orientation information of local features extracted
by the multi-task neural network. We calculate the orientation
differencebetweenlocalfeaturesineachpair,takingtheabsolute
value of this difference as the orientation discrepancy of the pair.
Subsequently,allthesedifferencesarestoredinahashtable,with
the key being the angle difference and the value being the list of
matching pairs. We then iterate through this hash table, selecting
the three directions that contain the most matching pairs as the
correct local feature pairs, and consider other matching pairs
that do not meet this criterion as outliers and remove them.
After obtaining the matching relationship between the current
frame and the keyframe, we use the method of reprojecting
errors to solve the PnP problem and optimize to obtain the pose
information of the current frame.
C. Loop Closing
The loop closure module plays a critical role in mitigating the
challenge of long-term positional drift in visual SLAM systems.
This drift often stems from sensor inaccuracies, the accumu-
lation of errors in pose estimation by the tracking module,
among other contributory factors. The essence of this module
is to equip the SLAM system with a mechanism for recognizing
whether its current location has been previously visited within
the established map. This recognition facilitates the fusion of
poses, enhancing the system’s accuracy and reliability over time.
In the original ORB-SLAM2 system, the loop closure module
incorporates the DBoW2 [38] algorithm. However, it’s impor-
tant to note that DBoW2 is not ideally suited for handling
ﬂoating-point type descriptors, resulting in less than optimal
efﬁciency. This limitation is particularly pertinent given that
descriptors derived from multi-task neural network typically
exhibit ﬂoating-point characteristics, necessitating a more com-
patible algorithmic approach.
To address this requirement, we utilize the FBoW [39] al-
gorithm model in lieu of DBoW2. This strategic update lever-
ages FBoW’s inherent strengths, notably its compatibility with
ﬂoating-point types and its ability to harness underlying hard-
ware acceleration features. Additionally, FBoW’s proﬁciency
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 7 -->
11888
IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 8, AUGUST 2025
TABLE I
TRAINING SERVER HARDWARE
TABLE II
QUANTITATIVE COMPARISON OF THE REPEATABILITY (%) AND PRECISION (%)
ON HPATCHES
in processing data in binary form translates to a marked im-
provement in computational speed, especially in the context of
calculating matching similarities within the bag of words model.
V. RESULTS AND DISCUSSIONS
In this section, we ﬁrst conduct simulations on feature points,
and then conduct comparative simulations with other visual
SLAM systems on two datasets. Finally, we perform an ablation
experiment.
A. Datasets
We utilize the HPatches [40] dataset to conduct experiments
on the feature extractor task within the multi-task neural net-
work. Additionally, for the localization and mapping experi-
ments, two widely used datasets in the industry, KITTI [41]
and TUM [42], are selected to test the MD-SLAM. These three
datasets hold signiﬁcant positions in the ﬁeld of computer vision,
with speciﬁc descriptions as follows:
The HPatches dataset evaluates image pair matching and
feature detection algorithms, with two key segments: illumi-
nation and viewpoint changes. The illumination segment tests
algorithm robustness against varying light conditions, while the
viewpoint segment assesses resilience to perspective changes.
This setup evaluates feature detection and matching algorithm
adaptability and resilience under realistic conditions.
The KITTI dataset, a collaboration between the Karlsruhe
Institute of Technology and the Toyota Technological Institute
at Chicago, features diverse sensor data, including cameras,
LiDAR, GPS, and IMU, supporting a wide array of computer
vision and autonomous driving tasks such as stereo vision,
object detection and tracking, visual SLAM, and vehicle per-
ception. Its high-quality visuals and precise annotations make
it ideal for evaluating visual SLAM systems and object detec-
tion and tracking algorithms, setting a benchmark in the ﬁeld
and aiding in the development and comparison of innovative
solutions.
The TUM dataset, created by the Technical University of Mu-
nich’s Computer Vision and Artiﬁcial Intelligence Laboratory,
is tailored for visual SLAM and 3D reconstruction research. It
includes visual, Inertial Measurement Unit (IMU), depth map,
and annotated data, supporting research in visual SLAM, 3D
reconstruction, pose, and motion estimation. The depth map
data, notable for its high precision, is especially valuable for
depth-reliant applications, making the TUM dataset a crucial
tool for developing and evaluating algorithms in these advanced
research areas. Its varied and detailed data signiﬁcantly aids in
crafting robust and accurate computer vision systems.
B. Metrics
In evaluating keypoint accuracy, we assess the model using
keypoint repeatability and precision. Repeatability measures the
proportion of local features that the feature extractor can still
detect after an image has transformed, evaluating the feature
extractor’s stability. The formula for calculating repeatability is
as follows:
Rep = 1
N
N

i=1
Nmatchedi
Min(NimageAi, NimageBi)
(9)
where N represents the total number of image pairs tested in the
dataset, Nmatchedi is the number of repeated (matched) local
features for the ith image pair, NimageAi and NimageBi are the
numbers of keypoints extracted from the ﬁrst and second images
of the ith pair, respectively.
Furthermore, we calculate the accuracy of the matched local
feature pairs to measure the precision of correct matches by the
model. The speciﬁc calculation process involves using the trans-
formation matrix applied during training to transfer the feature
points extracted from one training image to another coordinate
system. We calculate the Euclidean distance between the local
feature and its matched counterpart in this coordinate system.
The matched local feature pairs with distances exceeding a
certain threshold are considered qualiﬁed pairs. We calculate
the ﬁnal accuracy as follows:
MMAθ = 1
N
N

i=1
Ci
Ti
(10)
where Ci denotes the number of correct local feature matches
in the ith image pair, Ti is the total number of matched keypoint
pairs in the ith image pair, N represents the total number of
image pairs tested in the dataset, and θ denotes the threshold.
We use absolute position error (APE) and relative position
error (RPE) to evaluate the positioning accuracy of the SLAM
system. APE calculates the direct difference between the esti-
mated and ground-truth poses, reﬂecting the system’s precision
and the global consistency of the trajectory. RPE calculates the
accuracy of the pose difference between two frames of images
separated by a ﬁxed time difference, reﬂecting the system’s local
accuracy. In practical calculations, We use the open source tool
evo1 to calculate root mean square error (RMSE) of APE and
1https://github.com/MichaelGrupp/evo
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 8 -->
HE et al.: MD-SLAM: A MULTI-TASK DEEP-LEARNING-BASED VISUAL SLAM SYSTEM IN DYNAMIC ENVIRONMENTS
11889
TABLE III
QUANTITATIVE COMPARISON OF THE RMSE OF THE APE [M] AND THE AVERAGE OF THE RPE [M] ON KITTI AND TUM DATASETS
TABLE IV
QUANTITATIVE COMPARISON OF THE RMSE OF THE APE [M] AND OF OUR
SYSTEM AND OTHER SYSTEMS ON TUM DATASET
TABLE V
QUANTITATIVE COMPARISON OF THE RMSE OF THE APE [M] OF OUR SYSTEM
AND OTHER SYSTEMS ON KITTI DATASET
TABLE VI
ABLATION STUDY ON TUM AND KITTI DATASETS
RPE indicators to quantify the experimental results and generate
visual images of the poses. In addition, we use (11) to calculate
the improvement rate on the APE and RPE metrics:
δ = o −m
o
× 100%
(11)
where o is the comparison method, and m is the method to be
compared.
C. Implementation Details
We use a pre-trained model, ﬁne-tuned for speciﬁc cases, to
enhance the deep feature extractor’s performance and speed up
training. Speciﬁcally, for outdoor autonomous driving scenarios,
the network is ﬁne-tuned on the Cityscapes [43] dataset, employ-
ing a unique self-supervised method that doesn’t depend heavily
on annotated data. For indoor scenarios with multiple people, we
ﬁne-tune the COCO [44] dataset, leveraging its diverse, dynamic
object and human labels to train the model for accurate keypoint
extraction and dynamic change management in both outdoor
and indoor settings, thus boosting performance and versatility.
The training involves a phased approach, where during keypoint
extractor ﬁne-tuning, dynamic semantic network parameters are
frozen to focus on improving feature extraction. Conversely,
when training the dynamic semantic network, all but the seman-
tic segmentation task parameters are frozen, ensuring keypoint
extractor robustness. This method stabilizes the network while
optimizing each component.
The multi-task neural network model is constructed using
PyTorch and trained on a high-performance server. The server’s
speciﬁc hardware is shown as Table I. In other feature extraction
networks, SuperPoint uses a Titan X GPU, processing one frame
in 13 ms, while DX-SLAM uses a GTX 1070 GPU, processing
one frame in 19.7 ms. In the actual training process, the shared
orientation-invariant encoder parameter G is set to 36, and the
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 9 -->
11890
IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 8, AUGUST 2025
Fig. 3.
Visualization of trajectories from ORB-SLAM2 and ours compare to ground-truth on fr3_sitting_half and fr3_walking_half sequences. Left is ORB-
SLAM2 and right is MD-SLAM. (a) Trajectories on fr3_sitting_half sequence. (b) Trajectories on fr3_walking_half sequence.
number of epochs for ﬁne-tuning across different datasets is
determined to be 10, with a batch training size of 8. During the
phased training of the network, the Adam optimizer is employed
for gradient descent across all stages, with learning rates set at
0.001 and 0.00001, respectively. When training the semantic
network, we terminated the training early based on the mIoU
metric’s convergence conditions to prevent over-ﬁtting.
D. Deep Feature Test
We quantitatively tested the deep local feature extractor on
HPatches. We used SIFT and ORB, frequently used in traditional
visual SLAM, as baseline comparison methods and SuperPoint
as a comparison method for deep network extraction. When
quantifying accuracy, the calculation thresholds are selected as
3px and 5px, and the comparison data is shown in Table II.
The data in Table II shows that our method surpasses other
approaches in both repeatability and precision. Compared to
the ORB feature extractor used in traditional visual SLAM, the
multi-task network feature extractor employed by MD-SLAM
achieves a 21.3% improvement in matching repeatability and an
average precision increase of 56.6% across different thresholds.
This indicates that feature extractors based on multi-task neural
network are well-suited for the feature extraction module within
the tracking component of visual SLAM systems. Moreover,
compared to the SuperPoint feature extractor, which also relies
on deep learning, the directional information extracted by our
method signiﬁcantly enhances feature matching performance,
with a 9.94% improvement in repeatability and an average
precision increase of 12.81%.
E. Localization Test
We conduct different deep feature-based SLAM systems on
KITTI and TUM datasets, where we suppress the output of
the dynamic semantic task to evaluate our deep local features
performance. Our baseline method is ORB-SLAM2, which uses
the classic features. We also use SuperPoint and HF-Net features
based on deep networks for comparison. We use the ofﬂine
method when evaluating the deep local features. We extract
1000 features for each frame and only use one layer on the
pyramid scale of the system. For convenience and fairness,
we use different features on each test sequence to extract and
train the required bag-of-worlds (BOW) model with 10 depths
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 10 -->
HE et al.: MD-SLAM: A MULTI-TASK DEEP-LEARNING-BASED VISUAL SLAM SYSTEM IN DYNAMIC ENVIRONMENTS
11891
Fig. 4.
Visualization of APE and RPE from ORB-SLAM2 and MD-SLAM on KITTI. Left is ORB-SLAM2 and right is MD-SLAM. (a) APE comparison on
KITTI-07. (b) RPE comparison on KITTI-07. (c) APE comparison on KITTI-10. (d) RPE comparison on KITTI-10.
and 5 wideness parameters. Table III shows the experimental
results.
As we can see from Table III, our system can make a bet-
ter performance in most sequences. In the term of APE, the
improvement values reach up to 74.07%. In terms of RPE,
the improvement can reach 68.96%. Also, our system’s APE
and RPE metrics in most sequences are smaller than other
learning-based features. All methods based on learned local
features have smaller APE and RPE than the baseline SLAM,
ORB-SLAM2. This shows that the oriented keypoint provided
byourkeypointnetworkcanimprovetherobustnessandstability
of the visual SLAM system. For the narrow indoor ﬁeld of view
and frame size, we only extracted 500 local features, and the rest
of the visual SLAM settings, such as pyramid level and BOW
parameters, were the same as the KITTI evaluation. The exper-
imental results are also shown in Table III. The data in the table
indicatesthattheexperimentalresultsareverysimilartothoseon
the KITTI dataset. Deep local features can effectively improve
the robustness of the visual SLAM system, and our deep local
features can signiﬁcantly improve the stability of the SLAM
system because of the orientation extraction. Compared with the
traditional method, our method has an average APE difference
of 4.23 m over all sequences and an average improvement of
8.25% in RPE. Our deep local features also perform better than
the traditional method in facing scenes with dynamic objects.
F. Dynamic Environments Test
In this subsection, we focus on the performance of our system
in dynamic environments. The experimental settings are the
same as the above, and the difference is that our system will
use the semantic branch for dynamic region detection.
We ﬁrst compare our SLAM system with the TUM dataset’s
baseline ORB-SLAM2 and other similar methods. As we can see
from the data in Table IV, MD-SLAM has lower APE indicators
in most sequences, and the average increase is 63.84%. The
results show that our method of deep features with orientation
and dynamic point removal by semantic map can produce lower
positioning errors in the face of dynamic environments. We can
see from Fig. 3 that the trajectories estimated by our SLAM are
closer to the ground truth. We also compare our system with
other approaches for handling outdoor dynamic environments.
We also ﬁnd from the Fig. 4 that the APE and RPE indicators
of MD-SLAM are better than ORB-SLAM2 in terms of min-
imum, maximum and RMSE values, and it has a more stable
pose estimation performance. Table V shows that our system
outperforms the others on most sequences.
Thanks to the assistance of the semantic map, the method
using dynamic object removal is superior to the method without
using the semantic map in APE. At the same time, our slam
outperforms the method of semantic segmentation alone in most
scenarios, which shows that deep keypoints and dynamic points
removal and orientation constraint can get better performance.
G. Ablation Study
We control the variables into four stages: without using a
semantic map and orientation information which means only
usingthekeypointcoordinatelikeotherlearning-basedmethods;
without using the semantic map but using oriented keypoint;
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 11 -->
11892
IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 8, AUGUST 2025
using the semantic map but not using the orientation information
and the complete system. Table VI shows the experimental
ablation results.
We can see that the semantic information of the keypoints
plays a critical role in our system, and the method constrained
by semantic information shows a signiﬁcant advantage over the
method based only on orientation constraints in almost all test
sequences in terms of absolute pose error terms. In summary,
the semantic information of keypoints is necessary in dynamic
environments, while incorporating the orientation information
of keypoints in open ﬁeld environments can further improve the
performance of the system.
VI. CONCLUSION
In this paper, we studied the challenges posed by dynamic
objects and self-movement rotation in dynamic environments
on visual SLAM systems. We designed a visual SLAM system
based on a multi-task neural network, MD-SLAM. The neural
network integrated into the MD-SLAM system is capable of
simultaneously extracting image keypoints and identifying dy-
namic areas, as well as performing outlier removal within the
tracking module. Simulations conducted across multiple dataset
sequences indicate that the MD-SLAM system surpasses base-
line methods and other advanced approaches in both accuracy
and robustness, showcasing its efﬁcacy in handling dynamic
environments within the realm of visual SLAM. In the future,
we plan to enhance the computational efﬁciency of MD-SLAM,
ensuring it can perform in real-time and meet the demands
of resource-constrained environments. Additionally, we aim to
deploy our method in practical applications, such as autonomous
vehicles and robotics, where robust and accurate SLAM systems
are critical for navigation and environment mapping in dynamic,
real-world conditions.
REFERENCES
[1] R. Mur-Artal and J. D. Tardós, “ORB-SLAM2: An open-source SLAM
system for monocular, stereo, and RGB-D cameras,” IEEE Trans. Robot.,
vol. 33, no. 5, pp. 1255–1262, Oct. 2017.
[2] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE
Trans. Pattern Anal. Mach. Intell., vol. 40, no. 3, pp. 611–625, Mar. 2018.
[3] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efﬁcient
alternative to SIFT or SURF,” in Proc. Int. Conf. Comput. Vis. (ICCV),
2011, pp. 2564–2571.
[4] D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superpoint: Self-
supervised interest point detection and description,” in Proc. IEEE Conf.
Comput. Vis. Pattern Recognit. Workshops (CVPR), 2018, pp. 224–236.
[5] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, “From coarse
to ﬁne: Robust hierarchical localization at large scale,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2019,
pp. 12716–12725.
[6] A. J. Davison, I. D. Reid, N. D. Molton, and O. Stasse, “MonoSLAM:
Real-time single camera SLAM,” IEEE Trans. Pattern Anal. Mach. Intell.,
vol. 29, no. 6, pp. 1052–1067, Jun. 2007.
[7] G. Klein and D. Murray, “Parallel tracking and mapping for small AR
workspaces,” in Proc. IEEE Int. Symp. Mixed Augmented Reality (ISMAR),
IEEE, 2007, pp. 225–234.
[8] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM: A
versatile and accurate monocular SLAM system,” IEEE Trans. Robot.,
vol. 31, no. 5, pp. 1147–1163, Oct. 2015.
[9] C. Campos, R. Elvira, J. J. G. Rodríguez, J. M. Montiel, and J. D. Tardós,
“orb-slam3: An accurate open-source library for visual, visual–inertial,
and multimap slam,” IEEE Trans. Robot., vol. 37, no. 6, pp. 1874–1890,
Dec. 2021.
[10] T. Qin, P. Li, and S. Shen, “Vins-mono: A robust and versatile monoc-
ular visual-inertial state estimator,” IEEE Trans. Robot., vol. 34, no. 4,
pp. 1004–1020, Aug. 2018.
[11] J. Engel, T. Schöps, and D. Cremers, “LSD-SLAM: Large-scale direct
monocular SLAM,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2014,
pp. 834–849.
[12] Y. He, J. Fang, F. R. Yu, and V. C. M. Leung, “Large language mod-
els (LLMs) inference ofﬂoading and resource allocation in cloud-edge
computing: An active inference approach,” IEEE Trans. Mobile Comput.,
vol. 23, pp. 11253–11264, Dec. 2024.
[13] Y. He, N. Zhao, and H. Yin, “Integrated networking, caching, and
computing for connected vehicles: A deep reinforcement learning
approach,” IEEE Trans. Veh. Technol., vol. 67, no. 1, pp. 44–55,
Jan. 2018.
[14] Y. Ren, H. Zhang, F. R. Yu, W. Li, P. Zhao, and Y. He, “Industrial Internet
of Things with large language models (LLMs): An intelligence-based
reinforcement learning approach,” IEEE Trans. Mobile Comput., vol. 24,
pp. 4136–4152, May 2025.
[15] J. Tang, L. Ericson, J. Folkesson, and P. Jensfelt, “GCNv2: Efﬁcient
correspondence prediction for real-time SLAM,” IEEE Robot. Automat.
Lett., vol. 4, no. 4, pp. 3505–3512, Oct. 2019.
[16] M. Calonder, V. Lepetit, C. Strecha, and P. Fua, “BRIEF: Binary robust in-
dependent elementary features,” in Proc. Eur. Conf. Comput. Vis. (ECCV),
2010, pp. 778–792.
[17] Q. Peng, X. Zhao, R. Dang, and Z. Xiang, “RWT-SLAM: Robust visual
SLAM for weakly textured environments,” in Proc. IEEE Intell. Vehicles
Symp. (IV), Jeju Island, Korea, 2024.
[18] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, “LoFTR: Detector-free
local feature matching with transformers,” in Proc. IEEE Conf. Comput.
Vis. Pattern Recognit. (CVPR), 2021, pp. 8922–8931.
[19] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf.
Process. Syst., 2017, vol. 30, pp. 5998–6008.
[20] Y. Duan, J. Lu, J. Feng, and J. Zhou, “Learning rotation-invariant lo-
cal binary descriptor,” IEEE Trans. Image Process., vol. 26, no. 8,
pp. 3636–3651, Aug. 2017.
[21] C. Deng, K. Qiu, R. Xiong, and C. Zhou, “Comparative study of deep
learning based features in SLAM,” in Proc. 4th Asia-Paciﬁc Conf. Intell.
Robot Syst., 2019, pp. 250–254.
[22] U. S. Parihar et al., “RoRD: Rotation-robust descriptors and orthographic
views for local feature matching,” in Proc. IEEE/RSJ Int. Conf. Intell.
Robots Syst. (IROS), 2021, pp. 1593–1600.
[23] X. Wang, Z. Liu, Y. Hu, W. Xi, W. Yu, and D. Zou, “Featurebooster:
Boosting feature descriptors with a lightweight neural network,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023,
pp. 7630–7639.
[24] D. Zou and P. Tan, “CoSLAM: Collaborative visual SLAM in dynamic
environments,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 2,
pp. 354–366, Feb. 2013.
[25] P. F. Alcantarilla, J. J. Yebes, J. Almazán, and L. M. Bergasa, “On
combining visual SLAM and dense scene ﬂow to increase the robustness
of localization and mapping in dynamic environments,” in Proc. IEEE Int.
Conf. Robot. Automat. (ICRA), 2012, pp. 1290–1297.
[26] B. Bescos, J. M. Fácil, J. Civera, and J. Neira, “DynaSLAM: Tracking,
mapping, and inpainting in dynamic scenes,” IEEE Robot. Automat. Lett.,
vol. 3, no. 4, pp. 4076–4083, Oct. 2018.
[27] K. He, G. Gkioxari, P. Dollár, and R. Girshick, “Mask R-CNN,” in Proc.
IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2961–2969.
[28] C. Yu et al., “DS-SLAM: A semantic visual SLAM towards dynamic
environments,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS),
2018, pp. 1168–1174.
[29] V. Badrinarayanan, A. Kendall, and R. Cipolla, “SegNet: A deep con-
volutional encoder-decoder architecture for image segmentation,” IEEE
Trans. Pattern Anal. Mach. Intell., vol. 39, no. 12, pp. 2481–2495,
Dec. 2017.
[30] Y. Liu and J. Miura, “RDS-SLAM: Real-time dynamic SLAM using
semantic segmentation methods,” IEEE Access, vol. 9, pp. 23772–23785,
2021.
[31] A. Li, J. Wang, M. Xu, and Z. Chen, “DP-SLAM: A visual SLAM with
moving probability towards dynamic environments,” Inf. Sci., vol. 556,
pp. 128–142, 2021.
[32] Y. Fan, Q. Zhang, Y. Tang, S. Liu, and H. Han, “Blitz-SLAM: A semantic
SLAM in dynamic environments,” Pattern Recognit., vol. 121, 2022,
Art. no. 108225.
[33] D. Li et al., “DXSLAM: A robust and efﬁcient visual SLAM system with
deep features,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS),
2020, pp. 4958–4965.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 12 -->
HE et al.: MD-SLAM: A MULTI-TASK DEEP-LEARNING-BASED VISUAL SLAM SYSTEM IN DYNAMIC ENVIRONMENTS
11893
[34] M. Weiler and G. Cesa, “General E(2)-Equivariant steerable CNNs,” in
Proc. Conf. Neural Inf. Process. Syst. (NeurIPS), 2019, pp. 13003–13019.
[35] A. Mishchuk, D. Mishkin, F. Radenovic, and J. Matas, “Working hard to
know your neighbor’s margins: Local descriptor learning loss,” in Proc.
Adv. Neural Inf. Process. Syst., 2017, vol. 30, pp. 4826–4837.
[36] D. G. Lowe, “Object recognition from local scale-invariant features,”
in Proc. 7th IEEE Int. Conf. Comput. Vis. (ICCV), 1999, vol. 2,
pp. 1150–1157.
[37] J. Lee, B. Kim, and M. Cho, “Self-supervised equivariant learning for
oriented keypoint detection,” in Proc. IEEE Conf. Comput. Vis. Pattern
Recognit. (CVPR), 2022, pp. 4837–4847.
[38] D. Gálvez-López and J. D. Tardós, “Bags of binary words for fast
place recognition in image sequences,” IEEE Trans. Robot., vol. 28,
pp. 1188–1197, Oct. 2012.
[39] R. Munoz-Salinas and R. Medina-Carnicer, “UcoSLAM: Simultaneous
localization and mapping by fusion of keypoints and squared planar
markers,” Pattern Recognit., vol. 101, 2020, Art. no. 107193.
[40] V. Balntas, K. Lenc, A. Vedaldi, and K. Mikolajczyk, “HPatches: A bench-
mark and evaluation of handcrafted and learned local descriptors,” in Proc.
IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 5173–5182.
[41] A. Geiger, P. Lenz, and R. Urtasun, “Are We Ready for Autonomous
Driving? The KITTI Vision Benchmark Suite,” in Proc. IEEE Conf.
Comput. Vis. Pattern Recognit. (CVPR), 2012, pp. 3354–3361.
[42] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A
benchmark for the evaluation of RGB-D SLAM systems,” in Proc. 2012
IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2012, pp. 573–580.
[43] M. Cordts et al., “The cityscapes dataset for semantic urban scene under-
standing,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR),
2016, pp. 3213–3223.
[44] T.-Y. Lin et al., “Microsoft COCO: Common objects in context,” in Proc.
Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 740–755.
[45] P. Su, S. Luo, and X. Huang, “Real-time dynamic SLAM algorithm based
on deep learning,” IEEE Access, vol. 10, pp. 87754–87766, 2022.
[46] H. Guan, C. Qian, T. Wu, X. Hu, F. Duan, and X. Ye, “A dynamic
scene vision SLAM method incorporating object detection and object
characterization,” Sustainability, vol. 15, no. 4, 2023, Art. no. 3048.
[47] L. Chen, Z. Ling, Y. Gao, R. Sun, and S. Jin, “A real-time semantic visual
SLAM for dynamic environment based on deep learning and dynamic
probabilistic propagation,” Complex Intell. Syst., vol. 9, pp. 5653–5677,
Oct. 2023.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:23:32 UTC from IEEE Xplore.  Restrictions apply.
