<!-- page 1 -->
Highlights
Enhancing annotations for 5D apple pose estimation through 3D
Gaussian Splatting (3DGS)
Robert van de Ven, Trim Bresilla, Bram Nelissen, Ard Nieuwenhuizen, Eldert
J. van Henten, Gert Kootstra
• Novel pipeline simplifying pose annotation
• Novel method to quantify the occlusion rate was developed
• 99.6% reduction in the amount of manual annotations
• Training with an occlusion rate ≤95% for the labels lead to the best
performance
• Improved fruit detection and similar pose estimation as state of the art
arXiv:2512.20148v1  [cs.CV]  23 Dec 2025

<!-- page 2 -->
Enhancing annotations for 5D apple pose estimation
through 3D Gaussian Splatting (3DGS)
Robert van de Vena, Trim Bresillab, Bram Nelissena, Ard Nieuwenhuizenb,
Eldert J. van Hentena, Gert Kootstraa
aAgricultural Biosystems Engineering, Wageningen University &
Research, Droevendaalsesteeg 1, Wageningen, 6708 PB, the Netherlands
bAgrosystems Research, Wageningen University & Research, Droevendaalsesteeg
1, Wageningen, 6708 PB, the Netherlands
Abstract
Automating tasks in orchards is challenging because of the large amount of
variation in the environment and occlusions. One of the challenges is apple
pose estimation, where key points, such as the calyx, are often occluded.
Recently developed pose estimation methods no longer rely on these key
points, but still require them for annotations, making annotating challeng-
ing and time-consuming. Due to the abovementioned occlusions, there can
be conflicting and missing annotations of the same fruit between different im-
ages. Novel 3D reconstruction methods can be used to simplify annotating
and enlarge datasets. We propose a novel pipeline consisting of 3D Gaussian
Splatting to reconstruct an orchard scene, simplified annotations, automated
projection of the annotations to images, and the training and evaluation of
a pose estimation method. Using our pipeline, 105 manual annotations were
required to obtain 28,191 training labels, a reduction of 99.6%. Experimental
results indicated that training with labels of fruits that are ≤95% occluded
resulted in the best performance, with a neutral F1 score of 0.927 on the
original images and 0.970 on the rendered images. Adjusting the size of the
training dataset had small effects on the model performance in terms of F1
score and pose estimation accuracy. It was found that the least occluded
fruits had the best position estimation, which worsened as the fruits became
more occluded. It was also found that the tested pose estimation method
was unable to correctly learn the orientation estimation of apples.
Email address: robert.vandeven@wur.nl (Robert van de Ven)
Preprint submitted to Smart Agricultural Technology
December 24, 2025

<!-- page 3 -->
Keywords:
Fruit pose, Gaussian Splatting, 3D detection, Fruit occlusions
1. Introduction
Due to reduced labor availability and increased labor costs, there has
been an increasing interest in automation in orchards through the use of
robotics [1]. This is in particular the case for harvesting, as it requires a
large amount of manual labor in a short period of time. Automating tasks
in orchards, however, is challenging because of the large amount of variation
in the environment and the frequent occurrence of occlusions [2]. For the
harvesting of apples, a specific challenge is the correct detachment of the
fruits, which requires a specific rotational motion for optimal detachment at
the abscission layer [3, 4, 5]. This requires information about the position
and orientation, the pose, of the fruit.
Due to the rotational symmetry around the center axis, the pose of an
apple, as for many other fruits, can be defined as 5-dimensional (5D), con-
sisting of the 3D position and the 2D orientation of the center axis. Previous
work on apple-pose estimation used multi-stage [6, 7] or end-to-end detectors
[8]. Multi-stage detectors separate fruit detection and pose estimation, often
utilizing separate neural networks for the detection of the fruit and the de-
tection of key points for the calyx or peduncle, based on which a fruit pose is
calculated. With multi-stage detectors, the fruits can be detected quite well,
while the smaller calyx or peduncle is much more challenging to detect. Niu
et al. [7] achieved an Average Precision (AP) between IoU thresholds of 0.5
to 0.95 (AP@0.5:0.95) of 0.944 for the apple, but only 0.583 for the calyx. In
a lab environment, it was found that the average orientation error was 12.3°,
with increasing error when the fruits were less visible. In a real orchard,
Kok and Chen [6] achieved a median orientation error of 17.6°. If neither the
calyx nor peduncle is visible or detected, these multi-stage detectors cannot
estimate the orientation, which is the case in a substantial amount of situa-
tions. In comparison, end-to-end detectors use a single network to perform
both the detection and pose estimation in a single pass. These detectors do
not rely on the detection of specific key points to determine the orientation
of the fruit, allowing them, in principle, to estimate the orientation of fruits
without the key points visible. Son et al. [8] developed a deep neural network,
FRESHNet, which predicted 3D oriented bounding boxes per apple instance
in RGB-D images, including a custom stem direction loss, which accounted
2

<!-- page 4 -->
for the rotational symmetry to improve orientation estimation. They used a
dataset with 312 unique apples captured in 3,381 images, resulting in 9,044
apple instances in the images. FRESHNet achieved an mAP@0.5 of 0.783,
and a pitch and yaw error of 23.9°, and 34.4° respectively. This method was
not yet able to improve the performance compared to the multi-stage ap-
proaches. A potential reason is the limited size of the dataset in comparison
to that used in Rukhovich et al. [9], on which FRESHNet was based.
High quality training data is essential for the correct functioning of the
end-to-end method. While it is straightforward to annotate instance masks
for fruit detection, it is much more challenging to accurately annotate the
pose, specifically the orientation, of fruits in a camera image.
Especially
when neither the peduncle nor calyx is visible, the orientation cannot reliably
be annotated. In addition, as in Son et al. [8], datasets typically consist
of a large set of images taken from a smaller set of fruits under different
viewpoints. Annotating all instances in the images is very time-consuming,
and the consistency of labeling the orientation of the same apples in multiple
images is typically low, resulting in conflicting annotations, which limit the
performance of the trained model. Furthermore, there is an increased risk of
data leakage, as the same apple occurs in multiple images, which should not
end up in the training and the test set.
In this paper, we propose the use of 3D scene reconstruction from cam-
era images to deal with the above-mentioned challenges in data collection,
data annotation, and model training. In the 3D reconstruction, the data is
more complete through the integration of multiple viewpoints. This allows
better labeling of the fruit pose, as the annotator can rotate the 3D scene in
the annotation process. A 3D scene reconstruction, furthermore, allows the
rendering of new camera views.
However, the quality of the rendered images relies on the quality of the
reconstruction. Traditional methods, like Structure from Motion (SfM) and
multi-view stereo (MVS) lack density to achieve a high quality reconstruction
[10]. As a result, the technique was used to render depth for existing images
instead of rendering additional images, such as in Gen´e-Mola et al. [11, 12].
However, recent deep learning methods resulted in a major leap forward
in the quality of the reconstruction [13]. Especially 3D Gaussian Splatting
(3DGS) is promising, with reduced computational intensity for rendering,
enabling real-time rendering of high quality images [13].
These novel 3D
scene reconstruction methods can be used to address challenges in fruit pose
estimation by viewing around occlusions to improve annotations and render
3

<!-- page 5 -->
additional images from novel viewpoints to enlarge and enrich the dataset.
In this paper, we propose a novel pipeline, consisting of 3DGS to recon-
struct an orchard scene, simplified annotations, automated projection of the
annotations to images, and the training and evaluation of a pose-estimation
method. The annotation of the apple poses is done using the 3DGS. These
annotations are then projected onto the images to automatically create an-
notated image data.
This greatly simplifies the process of obtaining 3D
annotations, as only the unique apples need to be labeled instead of all ap-
ples in all images. Moreover, labels are now available for all fruits regardless
of possible occlusions in a given image, while a human annotator would be
unable to annotate a fruit in a 2D image if the occlusions are above a certain
level. In addition, the 3DGS enables rendering additional training data from
novel viewpoints to enlarge and enrich the dataset. The generated image
data is then used to train and test an apple detection and pose estimation
method. Through experiments, we investigated the effect of occlusion rate
on the model performance, and the effect of dataset size – including rendered
data – on model performance.
2. Materials & Methods
In this section, we describe the proposed pipeline, the dataset, and the
experiments.
2.1. Proposed pipeline
The proposed pipeline1 used in this study is shown in Figure 1. In the
pipeline, the first step is to acquire a dataset of RGB images, described in
section 2.1.1. Then, the scene is reconstructed in 3D using Structure from
Motion (SfM) and 3D Gaussian Splatting (3DGS), as described in section
2.1.2. Once the scene has been reconstructed, the fruit poses can be anno-
tated in the reconstruction and transformed to image space, as described in
section 2.1.3. Then, the data is split and images rendered or processed, as
described in section 2.1.4. Lastly, the created datasets can be used to train
the pose estimation method, as described in section 2.2.
1The pipeline is available at https://github.com/WUR-ABE/Gaussian-Splatting_
pose-estimation
4

<!-- page 6 -->
Structure 
from Motion 
(SfM)
Training 3D 
Gaussian 
Splatting (3D GS)
Manual 
annotation in 3D 
reconstruction
Training pose 
estimation 
method
3D reconstruction
Automated 
transfer to 
image space
Process 
original 
images
Rendering 
images from 
splat
Create data 
splits 
Dataset of 
generated 
images with 
labels
Dataset of 
real images 
with labels
Annotating fruit poses
Processing images
Data 
acquisition
Figure 1: Flowchart showing the proposed pipeline for pose estimation.
2.1.1. Dataset acquisition
The dataset collected for this paper consisted of images taken of 13
trees in an apple orchard located in Randwijk, the Netherlands. The im-
ages were taken 2 October 2024 between 16:25 and 17:05, during over-
cast weather conditions.
The images were taken using a Nikon Z6 sys-
tem camera. The exact settings are provided in Appendix
A. To ensure
that the quality of the reconstruction is as high as possible, images were
taken at varying heights while moving around the set of trees, and mul-
tiple trees were included in each image.
In addition, the pitch and yaw
of the camera were adjusted at each height to ensure complete coverage of
the trees. In total, 367 images were collected. The dataset is available at
https://doi.org/10.4121/976c94f2-028f-4291-adfd-20eb82b0f647
2.1.2. 3D reconstruction through Gaussian Splatting
In our study, the scene was reconstructed using 3D Gaussian Splatting
(3DGS) [13]. A well reconstructed scene enables the rendering of realistic im-
ages to train a pose estimation algorithm. In 3DGS, the scene is represented
by optimized 3D Gaussians. Each 3D Gaussian has a center (position) µ,
opacity α, 3D covariance matrix (size and orientation) Σ, and color c. The
color across the surface of the Gaussian is represented using spherical har-
monics to enable view-dependent appearance.
The placement of these 3D Gaussians is determined through an optimiza-
tion process utilizing back projection, which both optimizes the properties
5

<!-- page 7 -->
of the 3D Gaussians present and optimizes the number of 3D Gaussians
present [13]. The number of Gaussians is initialized using the sparse points
from Structure from Motion (SfM), as a good initialization is essential for
high reconstruction quality. The properties of the 3D Gaussians are opti-
mized through back-propagation, with the covariance matrix Σ represented
as a quaternion q and a 3D scale vector s to prevent creating a non-positive
semi-definite covariance matrix. The Gaussians can be densified by splitting
a large 3D Gaussian into two scaled down 3D Gaussians or copying a small
3D Gaussian and shifting the copy to a less well constructed area. In addi-
tion, the Gaussians can be pruned by removing Gaussians that are virtually
transparent, i.e., with low opacity α, and removing Gaussians that are too
large in view-space or world-space.
In our case, the software Agisoft Professional Photoscan using the COLMAP
algorithm [14] was used to obtain the initial sparse point cloud. In total, 364
images were matched. The settings are shown in Appendix
B. Next, the
camera poses were refined using the SO3xR3 optimizer from Nerfstudio [15].
Lastly, the 3DGS was trained using the software Jawset Postshot, using the
settings shown in Appendix C. The trained 3DGS is shown in Fig. 2. The
thirteen trees that were focused on are visible in high detail, while the general
structure of other trees can also be observed.
2.1.3. Annotating fruit poses
The trained 3DGS, in combination with the camera poses, enables the
annotation of fruit poses in the 3D reconstruction, providing a single anno-
tation per fruit and projecting these annotations to image space for training
a pose estimation method. In the following sections, we will describe the
process of annotating the global fruit pose and projecting these fruit poses
to image space.
Manual annotation in 3D reconstruction. The steps for manual annotation
are shown in Fig. 3. Fig. 3a shows the 3DGS. This 3DGS was converted to a
high density point cloud using the 3DGS-to-PC toolbox [16], using 50,000,000
points. The resulting point cloud is shown in Fig. 3b. In this point cloud,
each fruit was manually segmented by cropping the point cloud from multiple
viewpoints until only points belonging to the fruit remained. An example
of the resulting point cloud for a single apple is shown in Fig. 3c. Next,
the fruit orientation was determined by selecting the point in the fruit point
cloud that lies at the center of the calyx. In some cases, the color in the
6

<!-- page 8 -->
Figure 2: Image from the trained 3DGS. The camera poses are shown using frustums in
white. Not all camera poses are visible from this viewpoint.
point cloud was not clear enough to determine the calyx. In these cases,
the trained 3DGS and the original images were used to determine where the
calyx was located. The selected point of the calyx location is circled in blue
in Fig. 3d.
Automated transfer to image space. As the camera intrinsics and extrinsics
are known, the fruit point clouds can be transformed into each image. For the
fruits that are present in an image, the required label for the pose estimation
method is calculated. In our case, this consists of a 2D bounding box in the
image and a 3D oriented bounding box, using the calyx to determine the
orientation. In addition, the calyx key point or a mask could also be created.
However, being within the camera frustum does not mean that a fruit
is clearly visible, as there could be leaves in front of the fruit, or it could
be on the other side of the tree. Therefore, the visibility of the fruits also
needs to be determined. The visibility of each fruit within the frustum was
determined by first rendering a depth image of the fruit point cloud using the
camera intrinsics and extrinsics. As the point cloud only contains a single
7

<!-- page 9 -->
(a) 3DGS
(b) Point cloud
(c) Individual apple
(d) Calyx point
Figure 3: Steps to obtain fruit pose annotations.
(a) 98%
(b) 80%
(c) 60%
(d) 40%
(e) 20%
(f) 16%
Figure 4: Examples of occlusion rates of different apples
fruit, there are no occluding objects and all pixels with a depth belong to
the fruit, resulting in a total area of sT pixels. Next, the depth image was
rendered using the 3DGS as the source. As this includes the whole scene,
the occlusions are also present. Then, for each pixel that had a depth in
the image from the fruit point cloud, the difference in depth with the depth
image from the 3DGS was determined. If this difference was greater than 15
mm, the pixel was counted as occluded, resulting in a total occluded area of
sO pixels. The occlusion rate was then determined as:
o = sO
sT
× 100[%]
(1)
where o indicates the occlusion rate of a given fruit in a given image. Fig. 4
shows fruits with different amounts of occlusion. As sT is determined from a
point cloud, there can be small areas without depth on the front of the fruit
but with depth on the back of the fruit. As the difference in depth was used
to determine occlusions, these areas would be falsely counted as occluded.
Because of this, very few fruits had occlusion rates lower than 16%. For
example, the fruit shown in Fig. 4f does not seem to have any occlusions.
2.1.4. Processing images
To create datasets from the original images and the 3DGS, several steps
are required. First, the data splits need to be introduced to create clean
8

<!-- page 10 -->
(a) Full 3DGS
(b) Cropped 3DGS
Figure 5: Figure showing the full 3DGS and the cropped 3DGS when using the bounding
box of a tree.
splits and prevent data leakage. Then, novel images can be rendered using
these splits and the original images can be processed to follow the data splits.
The pose estimation method used in this pipeline uses RGB images and a
depth image, so depth images need to be rendered as well.
Creating data splits. After the 3DGS was trained, a bounding box was drawn
around each tree, separating the 3DGS into 13 smaller 3DGS, each containing
only a single tree. As the trees did not exactly fall within a bounding box,
the emphasis was put on ensuring the fruits of each tree were in the correct
bounding box, and some leaves were allowed to fall outside the bounding
box. Fig. 5 shows a bounding box in the 3DGS, where it can be seen that
some leaves from another tree are included in the bounding box.
9

<!-- page 11 -->
Rendering novel images. Images are created by projecting all 3D Gaussians
to the image space, the process called ”splatting”, resulting in 2D Gaussians,
with an expected depth for each 2D Gaussian.
Next, the Gaussians are
filtered to only include the Gaussians present in the image and sorted on
depth. Next, the color for each pixel is determined based on the color, depth,
and opacity of each Gaussian present, where the opacity of the Gaussian with
the lowest depth is used to determine the influence of the Gaussians with
higher depth. Simultaneous with the RGB, the depth image is rendered,
using the opacity normalized expected depth [17].
Processing original images. The original images included multiple fruits and
lacked depth information. In order to achieve clean data splits, each fruit
should only occur in a single split. This can be achieved by splitting the
dataset using the trees, where the edge of the trees generally does not con-
tain fruits, allowing clean splits without fruits on the boundary. Therefore,
each image should only include a single tree. In addition, a depth image is
required for the pose estimation method. Therefore, additional processing
was required to achieve a usable dataset.
A depth image can be rendered in the same way as when rendering novel
images, but using the intrinsics and extrinsics of the original image. However,
the original images generally contain multiple trees, which can result in data
leakage. In the middle of Fig. 6, part of an original image is shown, where
three different trees can be seen. To create clean data splits, the images
were split into separate images for each tree present, with the pixels showing
another tree masked out.
To achieve this, two depth images were rendered. First, a depth image
was rendered using the complete 3DGS, as shown in Fig. 5a. Next, a depth
image was rendered using the 3DGS containing only the desired tree, as
shown in Fig. 5b.
The cropped depth image only shows the desired tree but lacks occlusions
of other trees, which can occlude part of the leaves and fruits. Therefore, the
depth in the cropped image was compared with the depth in the complete
image. For any pixel where the depth in the complete image was lower than
in the cropped image, the pixel in the cropped image was set to zero. This
indicates that there is an occlusion and therefore the desired tree was not
visible, and there could not be a depth measurement.
This masked and
cropped depth image shows the depth of a single tree while accounting for
the occlusions caused by other trees.
10

<!-- page 12 -->
Tree 1
Tree 2
Tree 1
Tree 2
Figure 6: Masking of a single image to obtain multiple images that show only a single
tree. On the left, the depth image
Then, the masked and cropped depth image was used to mask the original
RGB image, converting the pixels without depth to black in order to hide
the other trees. This process is shown in Fig. 6. The outlines of the removed
trees can be observed by the outlines of the black areas.
2.2. Pose estimation method
In order to perform pose estimation, FRESHNet [8] was used. FRESH-
Net uses multimodal input, where the RGB-D image is converted to a colored
point cloud and a separate RGB image to enable the detection of 3D oriented
bounding boxes around the fruits. FRESHNet is a multimodal detector that
combines geometric information from a colored point cloud with visual fea-
tures from an RGB image. Architecturally it uses parallel encoders for the
point cloud and the RGB view, fuses intermediate features, and predicts
object-level outputs, specifically 3D oriented bounding boxes, which contain
information about location, orientation, and size. Instead of regressing a full
6-DoF orientation, the network enforces the bounding-box X-axis to align
11

<!-- page 13 -->
with the vector from the fruit center to the calyx; this reduces orientation
prediction to a unit stem direction vector and eliminates the redundant ro-
tation around that axis.
Training uses a stem-direction loss that encourages the predicted unit
vector ˆv to point toward the annotated calyx. This loss is combined with
standard localization and size losses for the bounding box and with classifi-
cation losses.
2.3. Dataset
The dataset acquisition was described in Section 2.1.1.
Through the
proposed pipeline, we obtained two datasets, one containing the processed
original images and the other containing the novel rendered images. For both
datasets, the data was split into a train, validation and test split. The train
split contained ten trees and 79 unique apples, the validation split two trees
and 19 unique apples, and the test split one tree and seven unique apples.
2.3.1. Original images dataset
The original images were 6048 by 4024 pixels.
In order to maintain
high quality images for the pose estimation method, these were patched
into smaller images of 1300 by 1300 pixels.
As this does not scale down
exactly from the original image size, each patched image overlapped slightly
with the other images. Each original image was converted into 20 smaller
patched images. Fig. 7 shows an example of the patching of images, showing
the horizontal and vertical overlap between the patched images. In order to
maintain correct camera intrinsics of the patched images, the center point
was shifted according to the patch location.
These depth and RGB images were combined with the labels transformed
to each image and stored as the dataset containing the original images. This
resulted in a dataset containing around 800 images per tree, combining to a
total of 10,757 images and 28,191 labels of fruits. Fig. 8 shows details on the
distribution of fruit orientations and occlusions for the training, validation
and testing splits.
2.3.2. Rendered images dataset
When rendering additional images, any camera extrinsics and intrinsics
can be used. In order to keep the images similar to the original images, the
same camera intrinsics as used for the patched images were used. As the
images can be rendered for each tree, a similar pattern was used for each
12

<!-- page 14 -->
Figure 7: Patches obtained from a single image. Along the vertical axis, four rows of
patches were created. Along the horizontal axis, five columns of patches were created. In
the large image, the overlap between patches can be observed.
Table 1: Settings for rendering images around a tree. Yaw rotation and height translation
are shown in blue. Camera roll is shown in green and camera pitch is shown in red.
Setting
Steps
Range
Height
3
−0.5 to 0.7 m
Roll
7
−1
2π to 1
2π radians
Pitch
3
−1
4π to 1
4π radians
Yaw
32
0 to 2π radians
Distance from tree
2
2.7 to 3.2 m
tree, rotating the camera around the tree at several heights and distances
from the tree while adjusting roll and pitch at each camera position. Figure
9 shows the adjustment of the camera pose around the tree origin. The tree
origin is shown with the wire frame, with the X-axis represented with red, the
Y-axis with green and the Z-axis with blue. First, the camera was translated
to the correct height. Next, the camera was rotated around the roll, pitch,
and yaw axes to the desired orientation. Lastly, it was moved backwards
along the rotated X-axis to the desired distance from the tree. Table 1 shows
the number of steps and range for each camera pose setting.
From each camera pose, an RGB and a depth image were created. These
were combined with the labels transformed to each image and stored as the
dataset containing rendered images. This resulted in a dataset containing
13

<!-- page 15 -->
−50
0
50
Pitch (degrees)
−100
0
100
Yaw (degrees)
(a) Train orientations
−50
0
50
Pitch (degrees)
−100
0
100
Yaw (degrees)
(b) Validation orientations
−50
0
50
Pitch (degrees)
−100
0
100
Yaw (degrees)
(c) Test orientations
0.1
0.2
0.3
0.4
0.5
0.6
0.7
Density
(d) 3D view of orientations
0
1000
2000
Count
Train
0
200
400
Count
Val
0.0
0.2
0.4
0.6
0.8
1.0
Occlusion rate
0
100
200
Count
Test
(e) Occlusions
Figure 8: Original data distribution.
4,032 images per tree, combining to a total of 52,416 images and 270,732
labels. Fig. 10 shows details on the distribution of fruit orientations and
occlusions for the training, validation and testing splits.
2.4. Experiments
With these datasets, we designed two experiments. First, a label occlu-
sion experiment was set up, described in section 2.4.1. Next, a dataset size
experiment was set up, described in section 2.4.2.
2.4.1. Label occlusion experiment
With this experiment, we aim to determine what occlusion rate in the
training labels results in the best model performance. Because the fruits are
labeled in the 3DGS, even heavily occluded fruits can be provided with labels
14

<!-- page 16 -->
(a) Side view
(b) Top view
Figure 9: Figure showing the camera movement for rendering images around a tree. The
wireframe at the center of the tree indicates the origin. Camera roll rotation is shown
with a curved red arrow, pitch rotation with a curved green arrow, and yaw rotation with
a curved blue arrow. Camera height translation is shown with a straight blue arrow, and
distance from tree translation is shown with a straight gray arrow.
when training the pose estimation algorithm, and the occlusion rate of each
label is known, as described in section 2.1.3. In practice, it is desired to detect
all fruits, including heavily occluded fruits. However, heavily occluded fruits
can be too challenging for a pose estimation algorithm to successfully learn
pose estimation. The opposite is also true, as clearly visible fruits without
a label can be detected, resulting in false positives. The examples in Fig.
4 show different occlusion rates for fruits. At 98%, the fruit could easily
be missed, both by the annotator and the network. However, at 60%, the
detection of the fruit is relatively easy, but orientation estimation remained
challenging, as the calyx and peduncle are not visible. Therefore, we de-
signed an experiment to evaluate the effect of occlusion rate on the model
performance. In addition, model performance is evaluated on a dataset con-
taining labels with a certain occlusion rate. Therefore, we evaluated each
model multiple times to determine the effect of occlusion rate not only in
model training but also during model evaluation.
This resulted in training the model with different upper limits of the oc-
clusion rate. This limit was set at 100%, 95%, 85%, 75%, 65% and 55%.
For this experiment, the model was either trained and tested on the origi-
nal images or on rendered images. As the dataset with rendered images is
15

<!-- page 17 -->
−50
0
50
Pitch (degrees)
−100
0
100
Yaw (degrees)
(a) Train orientations
−50
0
50
Pitch (degrees)
−100
0
100
Yaw (degrees)
(b) Validation orientations
−50
0
50
Pitch (degrees)
−100
0
100
Yaw (degrees)
(c) Test orientations
0.1
0.2
0.3
0.4
0.5
0.6
0.7
Density
(d) 3D view of orientations
0
10000
20000
Count
Train
0
1000
2000
Count
Val
0.0
0.2
0.4
0.6
0.8
1.0
Occlusion rate
0
500
1000
Count
Test
(e) Visibility
Figure 10: rendered data distribution.
much larger than the dataset with original images, the rendered dataset was
subsampled to contain the same number of labels as the original dataset.
Next, the validation labels were also selected with different upper limits of
the occlusion rate. This limit was set at 100%, 99.9%, 99%, 95%, 90%, 85%,
80%, 75%, 70%, 65%, 60%, 55% and 50%.
The models were evaluated on the F1 score, a commonly used metric for
object detection. The F1 score was calculated using the intersection over
union (IoU) between the ground truth 3D bounding box and the predicted
3D bounding box, using an IoU threshold of 0.5. If the threshold was above
0.5, the sample was a true positive (TP). Else, it was a false positive (FP).
A ground truth box without a predicted box was counted as a false negative
(FN).
16

<!-- page 18 -->
Neutral F1 score. Changing the upper limit of the occlusion rate changed
which ground truth labels were present. If the pose estimation method was
able to detect a fruit that was not provided with a label, this would be
counted as an FP, despite it being a correct prediction. This effect influences
the precision. Therefore, we calculated the precision using all labels, remov-
ing the effect of predicting fruits without labels based on their occlusion.
Then, the F1 score can be calculated using this recalculated precision and
the original recall, resulting in a neutralized F1 score. This neutralized F1
score gives a more realistic representation of the detection performance, as
the metric is not reduced for detecting fruits that are present but were not
given a label.
2.4.2. Dataset size experiment
With this experiment, we aimed to determine the effect of dataset size
on the performance. In general, more input data improves the performance
of neural networks. However, the number of original images is limited, as
annotating these images takes effort. Our proposed pipeline can render ad-
ditional images, without requiring additional annotation effort. In addition,
any camera pose can be used, increasing the variation of fruit poses.
Our experiment was to change the dataset size and test the effect on
apple detection and pose estimation. In this experiment, the models were
trained with the best max occlusion rate, as found in the experiment de-
scribed in 2.4.1. The complete training sets were randomly sampled to cre-
ate the datasets. As the rendered images had a different amount of apples
present per image than the original images, the dataset sizes were equal-
ized using the number of instances in the dataset relative to the number of
instances in the dataset containing the original images. Using the original
images, the model was trained with datasets containing 12.5%, 25%, 50%,
and 100% of the number of instances. Using rendered images, the model was
trained with datasets containing 12.5%, 25%, 50%, and 100% of the number
of instances. Lastly, the datasets of original images and rendered images
were mixed. Using mixed images, the model was trained with datasets con-
taining 25%, 50%, 100%, and 200% of the number of instances. At these
sizes, both sources accounted for half the training instances. At 200%, all
original images were used. The dataset was further expanded using only ren-
dered images, resulting in datasets containing 300%, 400%, and 500% of the
number of instances.
Each model was evaluated on the original images test set. The models
17

<!-- page 19 -->
were evaluated using the F1 score and the neutralized F1 score. In addition,
the pose estimation was evaluated.
The fruit position was evaluated by
looking at the Euclidean distance between the center of the ground truth
3D bounding box and that of the predicted 3D bounding box, as shown in
equation 2. The fruit orientation was evaluated by calculating the angular
difference between the fruit pose vector of the ground truth instance and that
of the predicted instance, as shown in equation 3.
d (p, ˆp) =
q
(px −ˆpx)2 + (py −ˆpy)2 + (pz −ˆpz)2
(2)
θ (v, ˆv) = arccos (v · ˆv)
(3)
3. Results
In this section, the results of the experiments are presented. In section
3.1, the results of the label occlusion experiment are presented. In section
3.2, the results of the dataset size experiment are presented.
3.1. Label occlusion experiment
Through our novel pipeline, we were able to annotate even the heavily
occluded fruits, something that is impossible in the conventional way of an-
notation. This raises the question of what the maximum occlusion rate is for
training of the object detector and pose estimator. This experiment, there-
fore, aimed to determine what occlusion rate in the training labels results
in the best model performance. Fig. 11 and Fig. 12 show the detection
performance for the real and rendered images respectively, measured using
the F1 and neutralized F1 score, as a function of the upper limit of the oc-
clusion rate in the test dataset, with separate lines for each upper limit of
the occlusion rate in the train dataset.
Fig. 11 shows the performance when training and testing on the original
images. With a high test occlusion rate, the F1 and neutral F1 scores are
relatively low, as there is a high amount of false negatives, as fruits that are
heavily occluded have a label but are not detected. With a low test occlusion
rate, the F1 score (Fig. 11a) was relatively low, as there is a high amount
of false positives because fruits that are relatively visible have no label but
are detected. In contrast, the neutral F1 score, shown in Fig. 11b, increased
when lowering the test occlusion rate, as the detection of occluded fruits was
18

<!-- page 20 -->
≤50
≤60
≤70
≤80
≤90
≤100
Test occlusion [%]
0.00
0.25
0.50
0.75
1.00
F1 Score
Train occlusion [%]
≤55
≤65
≤75
≤85
≤95
≤100
(a) F1 score
≤50
≤60
≤70
≤80
≤90
≤100
Test occlusion [%]
0.00
0.25
0.50
0.75
1.00
Neutral F1 Score
Train occlusion [%]
≤55
≤65
≤75
≤85
≤95
≤100
(b) Neutral F1 score
Figure 11: Performance of each train setting on each test setting, using the original im-
ages as train and test dataset. Vertical bars indicate the 95%-interval determined using
bootstrapping.
≤50
≤60
≤70
≤80
≤90
≤100
Test occlusion [%]
0.00
0.25
0.50
0.75
1.00
F1 Score
Train occlusion [%]
≤55
≤65
≤75
≤85
≤95
≤100
(a) F1 score
≤50
≤60
≤70
≤80
≤90
≤100
Test occlusion [%]
0.00
0.25
0.50
0.75
1.00
Neutral F1 Score
Train occlusion [%]
≤55
≤65
≤75
≤85
≤95
≤100
(b) Neutral F1 score
Figure 12: Performance of each train setting on each test setting, using the rendered
images as the train and test dataset. Vertical bars indicate the 95%-interval determined
using bootstrapping.
19

<!-- page 21 -->
not counted as false positives, better reflecting the ability to detect the fruits
present. For each of the different train occlusion rates, the highest F1 score
was achieved when it resembled the test occlusion rate. This indicates that
the neural network not only learned to detect the fruits but also learned at
what maximum level of occlusion they should be detected. Training with
an occlusion rate of ≤95% or ≤100%, using all fruit instances, resulted
in the highest neutral F1 score, but the differences between these two train
settings are small, which further reduced when lowering the test occlusion
rate. This shows that providing labels with a higher occlusion rate adds value
up to 95%. Adding the last 5% of the most occluded fruits did not have a
significant effect on the performance of the model.
Fig. 12 shows the performance when training and testing on the dataset
containing rendered images. Here, the same effects can be observed as when
using the original images. However, the F1 score and neutral F1 score are
slightly higher compared to when using the original images, indicating that
the detection of fruits in the rendered images is somewhat easier than in the
original images. Using rendered images, there was a smaller reduction in
performance in the case of high test occlusion rates, compared to using the
original images.
In order to better analyze the performance, the best train occlusion rate
for each test occlusion rate was determined. Table 2 shows the train occlusion
rate that achieved the best neutral F1 score on each test occlusion rate, for
the rendered and original images.
If there were no significant differences
between multiple train occlusion rates for a given test occlusion rate, all
train occlusion rates in the best performing group are shown. The significant
groups were calculated using Tukey’s HSD test. Here, it can be seen that as
the test occlusion rate is reduced, the best train occlusion rate was lower as
well, although not as much. This effect was greater in the rendered images
than in the original images.
From the table, it can be seen that a train
occlusion rate of ≤100% performs best when the test occlusion rate is also
high. From a test occlusion rate of ≤95%, a train occlusion rate of ≤95%
starts to perform best. This shows that annotating the 5% of most occluded
fruits only matters when it is important to detect these fruits. However, when
only the least occluded fruits need to be detected, it is beneficial to annotate
more occluded fruits, which can be observed at low test occlusion rates, where
the best train occlusion rate is ≤75% or ≤85%, for the rendered and original
images respectively. The difference between the best train occlusion rate for
a given test occlusion rate is bigger when using the original images than when
20

<!-- page 22 -->
Table 2: Table showing the group of train settings that performed best on each test setting.
Multiple train settings are shown if p > 0.05.
Test occlusion [%]
Train occlusion [%]
Rendered
Original
≤100
≤100
≤100
≤99.9
≤100
≤100
≤99
≤100
≤100
≤95
≤100, ≤95
≤95
≤90
≤95
≤95
≤85
≤95
≤95
≤80
≤95
≤95
≤75
≤95
≤95
≤70
≤95
≤95
≤65
≤95, ≤75
≤85, ≤95
≤60
≤75
≤95, ≤85
≤55
≤75, ≤65
≤85, ≤95
≤50
≤75
≤85
using the rendered images. When using the original images, a train occlusion
rate of ≤95% was in the best performing group still when the test occlusion
rate was ≤55%. To determine which train setting performed the best across
all test settings, the number of occurrences in the group that performed best
was counted for all levels of the test occlusion rate. This resulted in the
highest count for models trained with an occlusion rate of ≤95%, which
occurred in 7 of the 13 test settings when using the rendered images and in
9 of 13 test settings when using the original images.
In Table 3, the performance when averaging across all test settings is
shown for each train setting. The neutral F1 score is calculated by averag-
ing the performance across all test settings, and significant differences are
determined using Tukey’s HSD test. The rank is determined by ordering the
train settings on the neutral F1 score, using Tukey’s HSD test to assign the
same rank if there are no significant differences between train settings. In
this table, a similar trend in the results was visible when using the original
or rendered images. It can be seen that the models trained with an occlusion
rate of ≤95% and ≤100% achieved similar neutral F1 scores and outper-
formed all other train settings. When looking at the average rank, training
21

<!-- page 23 -->
Table 3: Table showing the neutralized F1 score and average rank of train settings across all
test settings. Significantly different groups are indicated with different letters if p < 0.05.
The best performance is highlighted in bold.
Train occlusion [%]
Rendered
Original
Neutral F1
Rank
Neutral F1
Rank
≤100
0.969a
2.53
0.924a
2.15
≤95
0.970a
1.54
0.927a
1.31
≤85
0.954b
2.46
0.910b
2.23
≤75
0.928c
2.92
0.878c
3.61
≤65
0.888d
4.15
0.834d
4.61
≤55
0.808e
5.46
0.769e
5.69
90:100
80:90
70:80
60:70
50:60
40:50
30:40
0:30
Occlusion [%]
0.00
0.25
0.50
0.75
1.00
Recall
(a) Recall
90:100
80:90
70:80
60:70
50:60
40:50
30:40
0:30
Occlusion [%]
0.0
2.5
5.0
7.5
10.0
Euclidean Distance Error(mm)
(b) Position error
90:100
80:90
70:80
60:70
50:60
40:50
30:40
0:30
Occlusion [%]
0
20
40
60
Vector Angle Error (°)
(c) Orientation error
Figure 13: Performance as a function of different levels of fruit occlusion, using the original
images as the train and test dataset. Vertical bars indicate the 95% confidence interval.
with an occlusion rate of ≤95% was the best, achieving nearly 1 rank higher
on average than the next best train setting. The models trained with ≤100%
and ≤85% are very similar in average rank, but there is a significant dif-
ference in neutral F1 score. This difference can be attributed to the large
differences in neutral F1 score at high test max occlusion rates. Here, the
models trained with an occlusion rate of ≤100% achieved a relatively large
difference in neutralized F1 score, while the ranking did not take the absolute
difference into account.
As training with an occlusion rate of ≤95% resulted in the best perfor-
mance, we used these models to analyze the performance for multiple levels
of occlusion, by dividing the test instances into bins. Eight bins were created,
in steps of 10% from fully occluded. Fruits with an occlusion rate of less than
22

<!-- page 24 -->
90:100
80:90
70:80
60:70
50:60
40:50
30:40
0:30
Occlusion [%]
0.00
0.25
0.50
0.75
1.00
Recall
(a) Recall
90:100
80:90
70:80
60:70
50:60
40:50
30:40
0:30
Occlusion [%]
0
5
10
Euclidean Distance Error(mm)
(b) Position error
90:100
80:90
70:80
60:70
50:60
40:50
30:40
0:30
Occlusion [%]
0
20
40
60
Vector Angle Error (°)
(c) Orientation error
Figure 14: Performance as a function of different levels of fruit occlusion, using the ren-
dered images as the train dataset and the original images as the test dataset. Vertical
bars indicate the 95% confidence interval.
30% were contained in a single bin, as there were relatively few fruits with
these low occlusion rates. Fig. 13 shows the performance as a function of
the occlusion rate of instances in the test dataset. The models were trained
on a dataset of original images with an occlusion rate of ≤95% and tested
on the original images. The detection performance is shown with the re-
call, shown in Fig. 13a. In this case, the performance remained consistent
as the occlusion rate increased. There was a large decrease in performance
for the bin with the most occluded fruits, indicating that a large amount
of these fruits were not detected. The position estimation performance is
shown with the Euclidean distance, shown in Fig.
13b.
In this case, it
can be seen that the position estimation accuracy reduced when the fruits
became more occluded. The orientation estimation performance is shown
with the vector angle error, shown in Fig. 13c. There was minimal effect
of the occlusion rate on the orientation estimation. For the least and most
occluded fruits, the orientation estimation was slightly worse. Fig. 14 shows
the performance as a function of the occlusion rate of instances in the test
dataset. The models were trained on a dataset of rendered images with an
occlusion rate of ≤95% and tested on the original images. The detection
performance is shown with the recall, shown in Fig. 14a. In this case, the
performance remained consistent as the occlusion rate increased. There was
a large decrease in performance for the bin with the most occluded fruits,
indicating that a large amount of these fruits were not detected. The posi-
tion estimation performance is shown with the Euclidean distance, shown in
Fig. 14b. In this case, it can be seen that the position estimation accuracy
23

<!-- page 25 -->
reduced when the fruits became more occluded. The orientation estimation
performance is shown with the vector angle error, shown in Fig. 14c. There
was minimal effect of the occlusion rate on the orientation estimation. The
orientation estimation became slightly worse when the fruits became more
occluded.
3.2. Dataset size experiment
With this experiment, we aimed to determine the effect of dataset size
on the performance. For this experiment, multiple training datasets were
evaluated on the test split of the dataset consisting of the original images.
Fig.
15 shows the detection performance as a function of the number of
labeled apple instances in the training set for original, rendered and mixed
training sets. The models were trained on a dataset with an occlusion rate
of ≤95% and tested on a dataset containing only the original images, with
an occlusion rate of ≤85%. Here it can be seen that the models trained
on just rendered data perform a lot worse than the models trained on the
original data. Using mixed data resulted in similar performance to using the
original images. Expanding the dataset beyond the amount of instances in
the dataset containing original images by adding more rendered images had
no significant effect on the detection performance.
Fig. 16 shows the pose estimation performance as a function of the num-
ber of labeled apple instances in the training set for original, rendered and
mixed training sets. The models were trained on a dataset with an occlusion
rate of ≤95% and tested on a dataset containing only the original images,
with an occlusion rate of ≤85%. For the position estimation, shown in Fig.
16a, there was no significant effect of the dataset size on position estimation
accuracy when using the dataset containing the original images or mixed im-
ages. When using the dataset containing the rendered images, there was a
significant improvement in position estimation accuracy when increasing the
dataset size. For the orientation estimation, shown in Fig. 16b, the smallest
datasets resulted in the best orientation estimation accuracy, while the oppo-
site was expected. Increasing the dataset size up to a normalized label count
of 1.0 reduced the orientation estimation accuracy, which stabilized for nor-
malized label counts greater than 1.0. To obtain an improved understanding
of the reduction in orientation estimation accuracy when the dataset size was
increased, the distribution of the estimated orientations for different dataset
settings is shown in Fig. 17. Here, it can be observed that the models trained
with the smallest datasets do not predict variation in the orientation at all.
24

<!-- page 26 -->
0
2
4
Normalized label count
0.0
0.2
0.4
0.6
0.8
1.0
F1 Score
Dataset
Mixed
Rendered
Original
(a) F1 score
0
2
4
Normalized label count
0.0
0.2
0.4
0.6
0.8
1.0
Neutral F1 Score
Dataset
Mixed
Rendered
Original
(b) Neutral F1 score
Figure 15: Detection performance as a function of the number of labeled apple instances.
Different colors indicate the type of images used in the train dataset. The number of
labeled apple instances was normalized relative to the number of labeled apple instances in
the original images, which was 15,419. Vertical bars indicate the 95%-interval determined
using bootstrapping.
This corresponds to always predicting no orientation, which can be used to
determine whether a model is predicting a correct orientation. While the
models trained with larger datasets resulted in more variation in the pre-
dicted orientation, this did not improve the orientation estimation.
This
indicates that the model did not correctly learn to predict the fruit orien-
tation but was likely randomly guessing a fruit orientation. This effect also
occurs when training with the datasets containing the original and rendered
images, for which the results are shown in Appendix D.
3.3. Comparison of annotation effort and performance
Table 4 shows the annotation effort and performance of multiple apple
pose estimation methods.
The methods are compared on the number of
annotations required to obtain the datasets used for training, the resulting
number of instances available for training, the detection performance, and the
pose estimation performance. Through our proposed pipeline, a fraction of
the annotation effort was required, while more instances were in the dataset.
While both other methods needed as many annotations as instances in the
dataset, our proposed pipeline reduced the annotation effort by 99.6%, when
considering only the original images. This if further reduced when rendering
25

<!-- page 27 -->
0
2
4
Normalized label count
0
2
4
6
8
10
Euclidean Distance Error (mm)
Dataset
Mixed
Rendered
Original
(a) Position error
0
2
4
Normalized label count
0
20
40
60
Vector Angle Error (°)
Dataset
Mixed
Rendered
Original
(b) Orientation error
Figure 16: Pose estimation performance as a function of the number of labeled apple
instances. Different colors indicate the type of images used in the train dataset. The
number of labeled apple instances was normalized relative to the number of labeled apple
instances in the original images, which was 15,419. Vertical bars indicate the 95%-interval
determined using bootstrapping.
additional images. In terms of detection performance, our proposed pipeline
improved the detection performance over both compared methods. In terms
of pose estimation performance, our proposed pipeline performed worse than
the multi-stage method presented by Kok and Chen [6], which achieved a
much lower median angle error.
Compared to FRESHNet [8] trained on
their dataset, our pitch and yaw errors were similar. Both other methods did
not report the position accuracy.
26

<!-- page 28 -->
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(a) Normalized label count of 0.125
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(b) Normalized label count of 0.25
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(c) Normalized label count of 0.5
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(d) Normalized label count of 1.0
Figure 17: Figure showing the orientations predicted for different training dataset sizes,
using only the original images. The test dataset consisted of the original images. The
distribution of orientations in this split is shown in Fig. 8c.
27

<!-- page 29 -->
Table 4: Table showing the annotation effort and performance of apple pose estimation
methods. Values indicated with 1 were not directly reported in their related work and
were calculated based on their reported precision and recall.
Model
Kok and Chen [6]
FRESHNet [8]
Ours
Annotation effort
3D Annotations
N/A
9,044
105
2D Annotations
1,000
N/A
N/A
Instances in dataset
1,000
9,044
28,191
Detection performance
mAP
0.8911
0.782
–
F1
0.9011
0.7841
0.943
Pose estimation performance
Pitch Error
–
23.88°
25.76°
Yaw Error
–
34.43°
32.99°
Vector Angle Error
17.6°
–
48.1°
Euclidean Distance
–
–
7.13 mm
4. Discussion
4.1. Discussion of the 3D annotation pipeline
3D scene reconstruction has been used for agricultural tasks, such as
fruit detection and segmentation, in related work [10, 11, 12]. These works
exploited the reconstruction mainly to generate depth data or to augment
datasets. Our proposed pipeline extends this by using 3DGS to reconstruct
an orchard scene, enabling the manual annotation in the 3DGS, projecting
the 3D annotations to the 3D image, and allowing the collection of a consis-
tent and large-scale dataset for 5D fruit pose estimation to train and evaluate
a pose-estimation method.
Using the 3DGS, the number of manual annotations was reduced by 99.6%
compared to annotating the data on a 2D image level. Moreover, as the
3DGS gives a more complete view of a fruit compared to individual images,
the pose could be annotated more accurately and the projection resulted in
consistent annotation of the image data even in situations with low visibility
due to occlusions. The pipeline furthermore enabled the calculation of the
occlusion rate of the fruit instances in the images. This allowed us to study
the effect of occlusions. Through this experiment, we were able to determine
28

<!-- page 30 -->
that the best performance could be achieved when fruits with an occlusion
rate of ≤95% were included in the train dataset.
The occlusion rate of an apple in an image was determined using the 3D
point cloud of that apple. However, the point cloud does not completely
cover the area of the fruit, leaving small holes, as can be seen in Fig. 3c.
In addition, the depth determined from the 3DGS contains errors if small
obstacles were in front of the fruit. These two effects resulted in the estimated
occlusion rate of fully visible fruits being approximately 13% instead of the
expected 0%. However, this affects the occlusion rate of all fruits in a similar
way, and does not affect the comparison between different train settings.
Therefore, we believe the conclusion remains valid.
The 3DGS that was created contained a single set of trees. This resulted
in a relatively homogeneous dataset, which does not represent the variety
in fruit growing systems. This limits the conclusions to similar conditions
as in this orchard. Including more variation in the dataset is valuable to
improve the generalizability of the pose-estimation method but the value of
the pipeline remains, as it simplifies creating new datasets.
4.2. Discussion of pose estimation
With our experiments, the effect of adjusting the upper limit of the oc-
clusion rate of the instances in the training set on model performance was
shown.
For the detection performance, it was found that the highest F1
score can be achieved when the test occlusion rate was similar to the train
occlusion rate. The trained models learned that apples occluded more than
the upper limit should not be detected. As detecting a fruit that is actually
there but occluded more than the upper limit should not be considered an
error by the detection model, we introduced the neutral F1 score. This led
to the finding that using more occluded fruits in the training data improved
the detection performance.
Concerning the performance of pose estimation, fruit occlusions reduced
the accuracy of estimating the position, with higher error for more occluded
instances.
For orientation estimation this relationship was not found.
It
should be noted, however, that orientation estimation showed a high error
between 50° and 60° for all occlusion rates. If instead of using FRESHnet,
the orientation angles were always set to 0.0, we obtained a lower orientation
error. From this, we have to conclude that FRESHnet was unable to estimate
the orientation of the apples. Our results are similar to the orientation esti-
mation accuracy reported in [8]. While they drew positive conclusions from
29

<!-- page 31 -->
the results, we believe that we have to conclude that FRESHnet is unable to
estimate the orientation of the apples.
4.3. Future work
In this work, we created a 3DGS of a single set of trees. This resulted in a
relatively homogeneous dataset, which does not represent the variety in fruit
growing systems. Therefore, future work should focus on collecting datasets
in different fruit growing systems, allowing for better generalization.
Although we focused on the evaluation of one pose-estimation method,
the proposed pipeline can be used to benchmark more pose estimation meth-
ods. Some pose estimation methods define the pose not as an oriented 3D
bounding box, but by a set of keypoints, e.g. Kok and Chen [6], for which
the labels cannot be generated by our pipeline. Therefore, expanding our
pipeline to generate other label types is valuable to enable evaluating multi-
ple methods under the same circumstances.
Given the poor performance of orientation estimation, focus should be
on improving the orientation estimation, for instance by tuning the weights,
improving the loss function on the orientation, or adjusting the network
architecture.
In addition, the occlusion rate can be incorporated during
training, to prevent the model from being punished for detecting fruits with
high occlusion rates, that would normally not have gotten a label.
Given the effect that the occlusion rate had on the performance of the
model, a comparison with manually providing labels for each image is inter-
esting. It is not known until what occlusion rate annotators provide labels
for fruits, and how the occlusion rate affects the accuracy of the annota-
tion. Therefore, it is recommended to research up to what occlusion rate
annotators provide annotations and how accurate these annotations are. In
addition, it is interesting to research how consistent multiple annotators are,
both for our method and for annotating each instance.
5. Conclusion
In this study, we presented a pipeline that utilized 3DGS to reconstruct an
orchard scene, simplified the manual annotation process, automated projec-
tion of the annotations to images, and performed the training and evaluation
of a pose-estimation method. To obtain labels for the original images, only
105 annotations were required to obtain 28,191 training labels for 10,757
images, a reduction of 99.6%.
30

<!-- page 32 -->
Through the 3DGS, the occlusion rate of each fruit in each image was
determined. To determine the effect of the occlusion rate on model perfor-
mance, we trained the model with multiple upper limits for the occlusion
rate of labels. It was found that each model performed best when tested on
labels with a similar upper limit of occlusion rate. It was found that training
with an occlusion rate up to ≤95% resulted in the best performance, with
a neutral F1 score of 0.927 on the original images and 0.970 on the rendered
images. Adjusting the size of the training dataset had small effects on the
model performance in terms of F1 score and pose-estimation accuracy. It
was found that the position of fruits could be determined with an average
error of 7.13 mm. The least occluded fruits had the best position estimation,
which worsened as the fruits became more occluded. It was also found that
the tested pose estimation method was unable to correctly learn the orienta-
tion estimation of apples, with an average vector angle error of 48.1°, coming
from a pitch error of 25.76° and a yaw error of 32.99°.
6. Acknowledgments
This work was partially funded by the Dutch Ministry of LVVN (Agricul-
ture, Fisheries, Food Security, and Nature), under the project code KB-38-
001-005, and by the Netherlands Organization for Scientific Research (NWO
grant 17626, project Synergia).
31

<!-- page 33 -->
References
[1] J. Lowenberg-DeBoer, I. Y. Huang, V. Grigoriadis, S. Blackmore, Eco-
nomics of robots and automation in field crop production,
Precision
Agriculture 21 (2020) 278–299.
[2] G. Kootstra, X. Wang, P. M. Blok, J. Hemming, E. Van Henten, Selec-
tive harvesting robotics: current research, trends, and future directions,
Current Robotics Reports 2 (2021) 95–104.
[3] J. Li, M. Karkee, Q. Zhang, K. Xiao, T. Feng, Characterizing apple
picking patterns for robotic harvesting, Computers and Electronics in
Agriculture 127 (2016) 633–640.
[4] P. Fan, B. Yan, M. Wang, X. Lei, Z. Liu, F. Yang, Three-finger grasp
planning and experimental analysis of picking patterns for robotic ap-
ple harvesting, Computers and Electronics in Agriculture 188 (2021)
106353.
[5] J. Davidson, S. Bhusal, C. Mo, M. Karkee, Q. Zhang,
Robotic ma-
nipulation for specialty crop harvesting: A review of manipulator and
end-effector technologies, Global Journal of Agricultural and Allied Sci-
ences 2 (2020) 25–41.
[6] E. Kok, C. Chen, Occluded apples orientation estimator based on deep
learning model for robotic harvesting, Computers and Electronics in
Agriculture 219 (2024) 108781.
[7] J. Niu, M. Bi, Q. Yu,
Apple pose estimation based on sch-yolo11s
segmentation, Agronomy 15 (2025) 900.
[8] G. Son, S. Lee, Y. Choi, Fresh: Fusion-based 3d apple recognition via
estimating stem direction heading, Agriculture 14 (2024) 2161.
[9] D. Rukhovich, A. Vorontsova, A. Konushin, Tr3d: Towards real-time
indoor 3d object detection, in: 2023 IEEE International Conference on
Image Processing (ICIP), IEEE, 2023, pp. 281–285.
[10] J. Li, X. Qi, S. H. Nabaei, M. Liu, D. Chen, X. Zhang, X. Yin, Z. Li,
A survey on 3d reconstruction techniques in plant phenotyping: from
classical methods to neural radiance fields (nerf), 3d gaussian splatting
(3dgs), and beyond, arXiv preprint arXiv:2505.00737 (2025).
32

<!-- page 34 -->
[11] J. Gen´e-Mola, R. Sanz-Cortiella, J. R. Rosell-Polo, A. Escola, E. Gre-
gorio, In-field apple size estimation using photogrammetry-derived 3d
point clouds: Comparison of 4 different methods considering fruit oc-
clusions, Computers and electronics in agriculture 188 (2021) 106343.
[12] J. Gen´e-Mola, M. Ferrer-Ferrer, E. Gregorio, P. M. Blok, J. Hemming,
J.-R. Morros, J. R. Rosell-Polo, V. Vilaplana, J. Ruiz-Hidalgo, Looking
behind occlusions: A study on amodal segmentation for robust on-tree
apple fruit size estimation, Computers and Electronics in Agriculture
209 (2023) 107854.
[13] G. Chen, W. Wang, A survey on 3d gaussian splatting, arXiv preprint
arXiv:2401.03890 (2024).
[14] J. L. Schonberger, J.-M. Frahm, Structure-from-motion revisited, in:
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 4104–4113.
[15] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, T. Wang, A. Kristoffersen,
J. Austin, K. Salahi, A. Ahuja, et al., Nerfstudio: A modular frame-
work for neural radiance field development, in: ACM SIGGRAPH 2023
conference proceedings, 2023, pp. 1–12.
[16] L. A. Stuart, M. P. Pound, 3dgs-to-pc: Convert a 3d gaussian splatting
scene into a dense point cloud or mesh, arXiv preprint arXiv:2501.07478
(2025).
[17] V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye,
J. Hu, M. Tancik, gsplat: An open-source library for gaussian splatting,
Journal of Machine Learning Research 26 (2025) 1–17.
[18] S. Kheradmand, D. Rebain, G. Sharma, W. Sun, Y.-C. Tseng, H. Isack,
A. Kar, A. Tagliasacchi, K. M. Yi, 3d gaussian splatting as markov
chain monte carlo, Advances in Neural Information Processing Systems
37 (2024) 80965–80986.
33

<!-- page 35 -->
Appendix A. Camera settings when taking images
Table A.5: Settings of camera for taking images
Setting
Value
Camera type
Nikon Z6
Lens type
NIKKOR Z 24-70mm f/2.8s
Image size
6048 by 4024
White balancing
Cloudy, approx. 6000K
F-stop
f/6.3
ISO
1600
Exposure
1/160
Focal length
24.0 mm
Appendix B. Settings for Structure from Motion
Table B.6: Settings used for camera pose estimation and sparse point cloud generation
Step
Setting
Value
Camera pose estimation
Accuracy
High
Key point limit
40,000
Tie point limit
4,000
Point cloud generation
Quality
High
Depth-filtering
Mild
Appendix C. Settings for training 3DGS
Table C.7: Settings used for training the 3DGS in the software Jawset Postshot
Setting
Value
Model profile
3DGS MCMC [18]
Sample images
3200 pixels
Max number of Gaussians
7,000,000
Max training steps
37,000
Antialiasing
Disabled
Create Sky Model
Disabled
34

<!-- page 36 -->
Appendix D. Additional results on orientation prediction
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(a) Normalized label count of 0.125
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(b) Normalized label count of 0.25
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(c) Normalized label count of 0.5
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(d) Normalized label count of 1.0
Figure D.18: Figure showing the orientations predicted for different training dataset sizes,
using only the rendered images. The test dataset consisted of the original images. The
distribution of orientations in this split is shown in Fig. 8c.
35

<!-- page 37 -->
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(a) Normalized label count of 0.25
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(b) Normalized label count of 0.5
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(c) Normalized label count of 1.0
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(d) Normalized label count of 2.0
Figure D.19: Figure showing the orientations predicted for different training dataset sizes,
using both original and rendered images up to a normalized label count of 2.0. The test
dataset consisted of the original images. The distribution of orientations in this split is
shown in Fig. 8c.
36

<!-- page 38 -->
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(a) Normalized label count of 3.0
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(b) Normalized label count of 4.0
−50
0
50
Pred Pitch (deg)
−100
0
100
Pred Yaw (deg)
(c) Normalized label count of 5.0
Figure D.20: Figure showing the orientations predicted for different training dataset sizes,
using both original and rendered images from a normalized label count of 3.0. The test
dataset consisted of the original images. The distribution of orientations in this split is
shown in Fig. 8c.
37
