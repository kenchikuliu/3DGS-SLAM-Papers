<!-- page 1 -->
DensifyBeforehand: LiDAR-assisted Content-aware Densification
for Efficient and Quality 3D Gaussian Splatting
Phurtivilai Patt
Leyang Huang
Yinqiang Zhang
Yang Lei
The University of Hong Kong
Abstract
This paper addresses the limitations of existing 3D Gaus-
sian Splatting (3DGS) methods, particularly their reliance
on adaptive density control, which can lead to floating
artifacts and inefficient resource usage.
We propose a
novel “densify beforehand” approach that enhances the
initialization of 3D scenes by combining sparse LiDAR
data with monocular depth estimation from correspond-
ing RGB images. Our ROI-aware sampling scheme prior-
itizes semantically and geometrically important regions,
yielding a dense point cloud that improves visual fidelity
and computational efficiency. This “densify beforehand”
approach bypasses the adaptive density control that may
introduce redundant Gaussians in the original pipeline, al-
lowing the optimization to focus on the other attributes of
3D Gaussian primitives, reducing overlap while enhanc-
ing visual quality. Our method achieves comparable re-
sults to state-of-the-art techniques while significantly low-
ering resource consumption and training time. We vali-
date our approach through extensive comparisons and ab-
lation studies on four newly collected datasets, showcas-
ing its effectiveness in preserving regions of interest in
complex scenes.
Keywords: 3DGS, adaptive density control, ROI-aware
sampling, point cloud generation, visual quality, resource
efficiency
1
Introduction
Neural Radiance Fields [18, 1] that can render novel view
images of a scene have attracted increasing research in-
terest. Among all relevant works, 3D Gaussian Splatting
(3DGS) [14] uses the sparse points from structure-from-
motion (SfM) [22] when tracking the camera poses of the
input multi-view images to initialize a 3D scene and em-
ploys a point-based approach that drastically speeds up
the rendering efficiency, paving the way for various com-
puter vision and robotic applications. Based on this recent
breakthrough, recent works [13, 17, 11, 9, 15, 23] accom-
pany the RGB camera with a ranging sensor (e.g., RGB-
Depth cameras or LiDAR device) to utilize 3DGS for si-
Figure 1: Our performance comparing to other SOTA
methods on wetlab dataset (a) and corner dataset (b). Our
method can achieve comparable performance in terms of
rendering quality and use fewer Gaussians to represent the
same scene.
multaneous localization and mapping (SLAM) and fast or
large-scale scene reconstruction. Compared to using the
sparse point cloud produced by SfM, LiDAR sensors can
provide a readily measurable point cloud for many down-
stream tasks, such as construction inspection or progress
monitoring [25].
With the increasing accessibility of miniaturized Li-
DAR sensors on personal mobile devices [2], such as the
iPhone Pro series, a promising application would be how
to use the miniaturized LiDAR sensor to facilitate model-
ing a scene with 3DGS. Despite their advantages of being
accessible and less sensitive to lighting conditions (com-
pared to RGB-Depth cameras), miniaturized LiDAR sen-
sors usually provide very sparse ranging data [21, 3] and
are with a limited ranging distance, thereby restricting
their performance in modeling complex 3D scenes, par-
ticularly in regions with fine details or thin objects.
Previous LiDAR-based works optimize their 3DGS
scenes with the adaptive density control (ADC) strategy
proposed in [14] to grow Gaussian primitives by cloning,
splitting, and pruning. However, ADC faces several lim-
itations, such as floating artifacts [26] and an excessive
number of Gaussian primitives [7], as discussed in Sec. 2,
potentially compromising the overall visual fidelity.
In this paper, we aim to propose a novel, streamlined
1
arXiv:2511.19294v1  [cs.CV]  24 Nov 2025

<!-- page 2 -->
approach to 3DGS scene representation. This approach
takes as input from a consumer-grade mobile device: 1)
the tracked poses of the device, 2) sparse point clouds
by a miniaturized LiDAR sensor, and 3) high-resolution
RGB images captured by an RGB camera.
As shown
in Fig. 2, it then resorts to the monocular depth estima-
tion (MDE) approach (e.g., [10]) for initial point cloud
densification. This “densify-beforehand” approach uses a
(nearly) constant number of Gaussian primitives from the
initialization, offering a practical controllability of the op-
timization process. It also achieves high-quality rendering
results without using several million Gaussian primitives,
yielding a resource-aware and efficient implementation by
leveraging LiDAR input.
We prioritize computational resources by defining the
regions of interest (ROI) in the RGB images (such as
texture-rich regions or foreground objects and texts/signs)
while downsampling other regions. This yields a dense
point cloud to represent the ROIs in the given scene. As
we will show in our results, our method can model a scene
with 3DGS using a much smaller number of Gaussian
primitives yet achieving a comparable rendering quality.
Comparison with state-of-the-art methods shows that
our “densify-beforehand” approach can achieve results
comparable to the adaptive density control of the origi-
nal 3DGS [14] or an improved version proposed in Pixel-
GS [26] in terms of visual quality, while drastically re-
ducing resource consumption in terms of the final num-
ber of Gaussians and training time. We also compare our
method with Taming 3DGS [16] and LightGaussian [8]
to demonstrate our effectiveness in a resource-constrained
set-up. Ablation studies are conducted to validate our de-
sign choices. Furthermore, our MDE-based densification
pipeline generates dense point clouds comparable to those
produced by depth sensors, validating its effectiveness.
We collected six new datasets featuring scenes with thin
geometry objects and rich textual content, to validate the
proposed method. We also showcase some applications
where our reconstructed scenes can preserve regions of
interest specified by users (such as using a verbal descrip-
tion).
In summary, our technical contributions are:
1) A novel offline, content-aware densification ap-
proach that bypasses the adaptive density control in the
existing 3DGS pipeline.
2) A practical strategy that leverages sparse LiDAR
points to refine the monocular depth estimation results for
3DGS scene representation.
3) Six datasets demonstrating a diverse range of appear-
ances captured using a camera and a miniaturized LiDAR
sensor.
2
Related works
LiDAR-assisted 3DGS. The accessibility of LiDAR sen-
sors and recent progress in multi-sensor fusion [27, 9] lead
to increasing adoption of visual-LiDAR systems in recon-
structing 3D scenes [9, 15, 23, 6]. In contrast to these
previous works that utilize high-end LiDAR sensors, our
method focuses on a miniaturized LiDAR equipped on a
commodity-grade mobile device.
Since a miniaturized
LiDAR sensor produces a very sparse point cloud, we
propose to leverage the recent advancement in monocu-
lar depth estimation (MDE), such as [10, 24, 5] to name
just a few, to densify the sparse LiDAR point cloud. We
show that this approach can bypass the error-prone yet ex-
pensive adaptive density control, allowing our method to
achieve SOTA performance in terms of both visual quality
and efficiency.
Adaptive Density Control.
Original 3DGS employs
adaptive density control to grow the 3D scene initialized
from SfM. However, this adaptive density control strat-
egy suffers from several limitations that may yield blurry
rendered images of the scene. Large Gaussian primitives
can be produced due to the sparse, unevenly distributed
initial points produced by SfM. Pixel-GS [26] addresses
this limitation by taking into account the number of pixels
a Gaussian primitive covers in each view in the optimiza-
tion. This weighted average formulation encourages the
point growth in regions covered by large Gaussian primi-
tives, thus avoiding the blurry results due to the presence
of large Gaussian primitives. Unlike these methods that
rely on the sparse point cloud from SfM or the adap-
tive density control (ADC), we leverage a mobile visual-
LiDAR device (i.e., an iPhone Pro 16) and propose a novel
approach that densifies the scene before optimization. In
particular, we replace the cloning operation, one of the
three major components of the ADC that increases Gaus-
sians in the scene, with the proposed DensifyBeforehand
point cloud. We validate our design choice by compari-
son with Pixel-GS [26] and the original 3DGS [14] that
use the same LiDAR point cloud for initialization.
Efficiency.
Foreground objects are contextually im-
portant for the scene and may require a large number of
3D Gaussian primitives to model in the original pipeline.
However, representing a scene with an excessive number
of Gaussian primitives requires substantial GPU mem-
ory for rendering and impairs real-time rendering per-
formance. Tamining 3DGS [16] proposes a score-based
densification approach that modifies the original adaptive
density control to be resource-aware.
Our method adopts a simple importance sampling ap-
proach to distribute more sampled points in the region of
interest (where color variance is high by default). This
approach allows a 3DGS scene to be trained with a con-
trollable resource in minutes while maintaining the visual
2

<!-- page 3 -->
quality of foreground objects or even the texts/signs on
them during runtime rendering. We show that our method
can be comparable with Tamining 3DGS [16] in terms of
the visual quality of the rendering outputs while achieving
higher training and rendering efficiency.
Observing the 3DGS scene representation contains
many Gaussian primitives with very low opacity, recent
works also proposed different approaches to cull these
low-opacity Gaussians [8, 19]. As shown in the experi-
ments, our 3DGS scenes can be pruned straightforwardly
by thresholding the opacity of Gaussians, yet achieve
comparable results as the state-of-the-art.
3
Methodology
3D Gaussian Splatting (3DGS) represents a scene as a
collection of 3D Gaussian primitives, centered at µ with
a scaling S and rotation R [14]. The color rendered on
the image plane is blended from multiple Gaussian primi-
tives depending on the opacity. The optimization pipeline
consists of three major operations: cloning, splitting, and
pruning. We propose a densify-beforehand approach in-
stead of performing cloning operations during the opti-
mization.
The overview of our method is shown in Fig. 2. Given
a set of posed RGB images and a LiDAR point cloud, our
method first fuses the sparse LiDAR data and the monoc-
ular depth estimation (MDE) results from [10] to den-
sify the scene. We sample an ROI-aware subset from this
densified point cloud, then initialize and train 3DGS with
these ROI-aware samples, bypassing the adaptive density
control in the original pipeline.
3.1
LiDAR-assisted Estimated Depth Re-
finement
Given the huge progress in monocular depth estima-
tion [10, 5, 4, 24], accurately estimating the metric depth
of a given RGB image is still challenging. Therefore, to
provide a dense point cloud to initialize the 3DGS, we
leverage sparse point clouds from a LiDAR sensor to re-
fine the estimated depth from MDE.
The input to our method contains a set of RGB im-
ages {Ik} with corresponding poses {Tk} derived from
LiDAR-based camera tracking, and a point cloud X =
{xi}. We adopt Metric3D-V2-vits-giant [10] to provide
a per-pixel depth map D for each RGB image I. Then,
we propose a global-local approach to refine the estimated
depth maps {Dk} as described.
First, we project the LiDAR points X to each image
plane defined by a Tk with the use of hidden point re-
moval [12] to avoid projecting occluded points onto the
image plane, denoted Ak. Value at each pixel of Ak is
either 0 or the depth from the projected point and 1k is
an indicator function of the non-zero elements of Ak. We
term the non-zero pixels in Ak as anchors.
Global scaling operation. We then calculate a global
rescaling factor for each Dk based on the anchors in Ak:
gk = MEDIAN
 1k ∗Ak
1k ∗Dk

,
where MEDIAN is the median operator and ∗denotes the
element-wise multiplication. Thus, the scaled depth map
is calculated as
Dt
k = gt
kDt−1
k
.
(1)
We perform this global scaling iteratively to ensure the
convergence of the scaled depth map, which is usually re-
quired no more than 5 iterations, and denote the resulting
rescaled depth maps as ˜Dk.
Local cluster-based scaling operation. Upon the con-
vergence of the iterative global scaling operations, multi-
ple scans stack in the scene and point cloud consolitation
is need. In order to address this issue, we propose a local
cluster-based scaling operation to consolidate the depth
maps from MDE.
We use the anchor pixels pi in Ak to perform cluster-
ing on the corresponding monocular depth image ˜Dk. The
clustering process assigns each pixel qj in ˜D to the near-
est anchor pixel from pi based on their Euclidean distance
d(pi, qj) =
(pi, A(pi)) −(qj, ˜D(qj))

2.
Since LiDAR points are distributed in the near scene and
cannot capture far-away regions, we intentionally filter
out any pixels qj whose distance to pi is larger than a
threshold τ to avoid wrongly rescaling far-away points in
˜D. Thus, a cluster at pi is denoted Ci = {qj|d(pi, qj) <
τ}. Within each cluster Ci, we first compute a local scal-
ing factor γi as γi = A(pi)/ ˜D(pi) and then apply γi to
all pixels qj in Ci:
ˆD(Ci) = γi ˜D(qj ∈Ci).
(2)
3.2
ROI-aware Sampling for 3DGS
Dense depth maps of the input RGB frames generated
in the previous stage not only densify the sparse LiDAR
points X in the foreground scene but also the far-away
background scene that cannot be captured by the miniatur-
ized LiDAR. However, converting all dense depth maps1
into a point cloud for training would entail a huge com-
putational overhead. Therefore, we subsample a subset of
pixels from each image and convert them into the initial
point cloud (several hundred thousand points) for 3DGS
training.
12.7 million pixels per frame for a 1920 × 1440 RGB image
3

<!-- page 4 -->
Figure 2: Our “densify beforehand” approach. Our method takes as input a set of posed RGB frames real-time tracked
from a visual-LiDAR device and the resulting sparse point cloud. We adopt a monocular depth estimation (MDE)
method to derive dense depth maps from RGB frames and utilize the sparse LiDAR points to rescale the MDE results.
To prepare dense initial points for training 3DGS, we adopt an ROI-aware importance sampling strategy. Finally we
train the dense input points with 3DGS splitting and pruning to yield an efficient and compact scene.
We employ an importance sampling approach to better
reconstruct the region of interest of the scene. Specifi-
cally, we aim to define the importance scores to holisti-
cally reflect the importance of 3D regions instead of their
2D projections in each RGB frame. Therefore, we ac-
cumulate the spatial importance score s(X) from pixels.
Given the definition of a per-pixel importance score, we
back-project the per-pixel importance scores s(qj) to the
LiDAR point cloud X for establishing the spatial impor-
tance in the 3D scene. First, we compute each anchor’s
importance score s(pi) by averaging the importance score
of the associated pixels in its corresponding cluster Ci.
Recalling that anchors are 2D projections of the LiDAR
points, we average the importance scores from different
anchors of the same LiDAR point to be the importance
score s(xi ∈X) of these LiDAR points. Thus, the spatial
importance is established. By default, we use the color
variance with a kernel size of 9 to compute the per-pixel
importance.
When sampling a dense set of points from 3D, we
project the importance scores from the LiDAR point to its
corresponding anchor in a certain RGB frame and perform
the importance sampling based on this importance score.
This treatment ensures a consistent important score within
a small spatial neighborhood, avoiding a view-dependent
importance scoring scheme and respect both the texture
and geometry of the 3D scene. This approach naturally
concentrates computing resources (i.e., the samples) on
the foreground scene as far-away regions have fewer Li-
DAR points.
Given the spatial importance score s(x) in 3D, we
threshold it with a pre-defined ratio to the median of the
importance score, producing an importance mask.
We
then project it to all RGB frames Ik with their poses Tk.
This projected mask is denoted Mk for k-th RGB frame.
We generate a point cloud P ∈RN×3 from ˆDk using
their camera poses Tk. The density of the point cloud
from ROI or non-ROI regions are controlled by the mask
Mk. To achieve this, we apply a density parameter ρ that
controls the number of points generated in ROI and non-
ROI regions. Specifically, for ROI regions, we use a high
sampling density ρROI to ensure dense point coverage; on
the other hand, we use a low sampling density ρnon-ROI
to maintain computational efficiency. This sampling den-
sity ratio (ρROI/ρnon−ROI) is set to 30 and the masking
threshold for defining the ROI is set to the median of the
spatial importance scores. We found that putting more
points in the ROI regions, or the texture-rich regions, is
beneficial to obtain higher reconstruction quality metrics
(e.g., PSNR).
The resulting point cloud P is used as input for train-
ing the 3D Gaussian model. By preprocessing the scene
to prioritize ROIs, we ensure that the Gaussian model fo-
cuses its resources on ROI areas, leading to more efficient
training and accurate reconstruction in these areas. This
approach avoids the cloning operation in the original ADC
pipeline to excessively introduce nearby Gaussians that
may be redundant at this end.
4

<!-- page 5 -->
W1
R2
C3
P4
M5
S6
#Frames
191
321
564
156
285
249
#Points
148,414
111,710
177,747
184,685
123,247
218,659
Table 1: Statistics of the collected datasets. W1: Wet Lab;
R2: Reception; C3: Corner; P4: Pantry; M5: Machines;
S6: Staircase
4
Experimental results
In this section, we first introduce our self-collected Li-
DAR dataset in Sec. 4.1. Sec. 4.2 compares our Den-
sifyBeforehand approach with state-of-the-art methods,
demonstrating its comparable performance in visual qual-
ity and superiority in training efficiency, storage, and the
peak number of Gaussian primitives during optimization.
These results demonstrate the practical usefulness of our
method. Ablation studies are conducted in Sec. 4.3 to an-
alyze the contribution of each proposed component.
4.1
Dataset preparation
Six real-world datasets (see Fig. 3) were collected to fa-
cilitate a comprehensive evaluation of our work.
The
datasets include Wetlab (W1), Reception (R2), Corner
(C3), Pantry (P4), Machines (M5), and Staircase (S6). An
iPad Pro with a miniaturized LiDAR sensor was used for
data collection. ARKit 2 was used to acquire the pose of
the sensors and the data. RGB frames are at a resolution
of 1920 × 1440. See Tab. 1 for more statistics about the
datasets.
4.2
Comparison with SOTA
Comparing methods.
To validate our approach, we
compared our method against two groups of state-of-
the-art (SOTA) methods. The first group is the original
3DGS [14] and Pixel-GS [26] that uses a huge number of
Gaussians to represent the scene. The other group con-
tains LightGaussian [8] and Taming 3DGS [16] that im-
prove 3DGS’s computational overhead.
Implementation details. All experiments were con-
ducted using PyTorch [20] on a desktop computer
equipped with an NVIDIA RTX4090 GPU (24GB) and
an Intel Core i9-12900K CPU running on Linux Ubuntu
22.04.
We implemented our work based on the original 3DGS
framework [14]. All comparing methods [16, 26, 8] were
implemented with their open-sourced codes and default
hyperparameters. We downsampled the input images to a
resolution of 1600 × 1200 to allow all methods, includ-
ing the more computationally intensive ones, to be trained
with an NVIDIA RTX4090 GPU. All timings start at load-
2https://developer.apple.com/augmented-reality/arkit/
ing the dataset and conclude once the final checkpoint is
saved.
Metrics. We report established metrics from previous
studies, i.e., PSNR, SSIM, and LPIPS to evaluate the qual-
ity of the rendering results. Additionally, we also compare
the training time in seconds (s), as well as the final and
peak numbers of Gaussian primitives during training (de-
noted #G and Peak #G, respectively). These metrics serve
as indicators of computational efficiency and overhead of
the comparing methods.
Results.
Tab. 2 quantitatively compares the novel-
view rendering quality of our methods to those of the
SOTA methods. We also facilitate the readers to appre-
ciate our method’s advantages and limitations by a quali-
tative comparison with the SOTA methods in Fig. 3.
First, we compare our results with those produced by
3DGS [14] on the datasets we collected. We observe that
our results are comparable to, if not better than, the 3DGS
results. Fig. 3 shows that with our densify-beforehand ap-
proach, our 3DGS scene can eventually capture the puller
in the scene which is thin and with a complicated shape.
On the other hand, the original 3DGS pipeline can cap-
ture only the upper part of the puller which is straight;
the lower part with a triangular shape is challenging for
the original pipeline as cloning and splitting the initial
points (sparse LiDAR points) cannot effectively move ex-
isting Gaussians to this complicated region.
Compar-
ing our method with 3DGS in terms of the computa-
tional overhead also demonstrates our advantages. Our
method requires less training time and produces consis-
tently fewer Gaussians at the end across different scenes
than the 3DGS original pipeline with cloning. Despite
using less resources, our method can represent the scene
with a slightly higher reconstruction quality on Wetlab
(W1) and Corner (C3) datasets and achieve comparable
performance on Staircase (S6) and Pantry(P4) datasets.
Similar to 3DGS, PixelGS [26] also entails a huge num-
ber of Gaussians (around 2.5 million) to represent a scene
with comparable visual quality (see Fig. 3(b) and Tab. 2
Staircase), entailing a prolonged training time.
Our approach can be seen as an alternative to the orig-
inal 3DGS pipeline and Taming3DGS where a moderate
size of Gaussian primitives are initialized in a scene for
the optimizer to grow. With the monocular depth estima-
tion results (rescaled by the LiDAR input as reference),
we approach the 3DGS initialization more aggressively
by putting a large number of points at the beginning and
letting the optimizer prune them. We show that this pro-
posed approach can achieve comparable results with Tam-
ing3DGS, with better performance on Wetlab (W1) and
Corner (C3) and slightly failing behind Taming3DGS on
Reception (R2) and Staircase (S6). In terms of compu-
tational overhead, our method consistently outperforms
Taming3DGS in the training time and our final results can
5

<!-- page 6 -->
PSNR ↑
SSIM ↑
LPIPS ↓
Time (s) ↓
#G (k) ↓
Peak #G (k)↓
POpaq≤0.1 ↓
Wetlab
3DGS [14]
28.8505
0.9387
0.1459
407.61
918.23
918.23
60.1
Pixel-GS [26]
29.6315
0.9369
0.1500
2019.86
1439.35
1439.35
54.17
Taming-GS [16]
28.7854
0.9387
0.1456
496.04
406.86
406.86
57.91
LightGS [8]
28.9727
0.9362
0.1580
2736.09
285.42
739.44
37.83
Ours
29.7187
0.9389
0.1454
402.87
252.90
630.71
5.52
Reception
3DGS [14]
25.9346
0.9017
0.1643
465.68
975.12
975.12
55.55
Pixel-GS [26]
25.9772
0.9016
0.1657
1621.12
1678.7
1678.7
41.94
Taming-GS [16]
26.0844
0.9029
0.1641
524.18
474.12
474.12
51.64
LightGS [8]
25.8042
0.8964
0.1877
2820.87
474.14
745.09
41.91
Ours
25.8380
0.8997
0.1656
440.01
333.59
649.81
4.28
Corner
3DGS [14]
28.2961
0.9096
0.1553
559.92
1847.52
1847.52
62.51
Pixel-GS [26]
28.4304
0.9134
0.146
2076.66
3629.32
3636.69
60.93
Taming-GS [16]
28.145
0.9043
0.1648
720.00
521.24
985.17
57.56
LightGS [8]*
27.4819
0.8949
0.1850
3316.4
1095.78
3025.00
22.82
Ours
28.4400
0.9110
0.1500
577.74
666.20
1498.87
4.54
Machines
3DGS [14]
24.422
0.8111
0.2696
480.81
1349.9
1349.9
56.52
Pixel-GS [26]
24.5386
0.8143
0.2594
1927.67
2970.21
2970.21
45.81
Taming-GS [16]
24.6384
0.8154
0.2633
661.42
999.96
999.96
59.65
LightGS [8]
23.8544
0.7965
0.3044
2758.96
1000.00
1104.82
57.27
Ours
24.4118
0.8103
0.2667
454.88
532.57
1099.45
3.95
Pantry
3DGS [14]
27.2479
0.9063
0.1734
382.11
863.09
863.09
56.29
Pixel-GS [26]
27.5212
0.9078
0.1710
1524.78
1554.57
1554.57
41.38
Taming-GS [16]
27.3060
0.9060
0.1739
448.44
446.25
446.25
54.05
LightGS [8]
27.2005
0.9048
0.1864
2638.26
446.26
723.69
34.08
Ours
26.9792
0.9023
0.1777
379.42
330.57
699.79
6.14
Staircase
3DGS [14]
27.8365
0.8904
0.1912
524.63
1349.62
1349.62
57.75
Pixel-GS [26]
27.8628
0.891
0.1875
1869.42
2404.83
2404.83
54.41
Taming-GS [16]
28.0503
0.8927
0.1915
542.22
581.38
581.38
52.02
LightGS [8]
27.2810
0.8809
0.2157
2572.47
581.40
1125.02
33.22
Ours
27.7073
0.8875
0.1940
433.73
385.42
877.40
5.30
Table 2:
Quantitative comparison with SOTA on six self-collected datasets. All comparing methods are initialized
with the same LiDAR point cloud for fair comparison. Our method densifies the LiDAR point cloud with the use of
a monocular depth estimation method. Besides the metrics for rendering quality, we also report Time (s) for training,
the number of final/peak Gaussians in thousands (#G / Peak#G), and the percentage of Gaussians with opacity lower
than 0.1. *LightGS cannot fit in NVIDIA RTX4090 GPU and thus is trained on L20 (with 48GB GPU memory) which
is slower.
6

<!-- page 7 -->
Figure 3: Qualitative comparisons of our results and those produced by the SOTA methods. Our method can capture
the puller in the Wetlab scene (in the first row) and reconstruct both the text (enclosed) and the far-away scene of the
Corner scene (in the second row). We achieve comparable visual quality with the comparing methods in the Pantry
(third) and Staircase (fourth) where PixelGS excels in capturing the texts.
be as compact as 200,000 Gaussians.
LightGS [8] is designed to derive a more compact
3DGS representation, which requires a trained 3DGS as
input. We followed their source codes for implementa-
tion and observed that LightGS sometimes incurs a high
usage of GPU memory and failed to train the scene Cor-
ner (C3) on the RTX4090 GPU. We have to report results
regarding this particular scene on a GPU server. In con-
trast, with our DensifyBeforehand approach, our method
can strictly control the number of Gaussians and thus the
GPU memory usage.
From Tab. 2, we can see that our method constantly
outperforms LightGS with higher reconstruction metrics
with a lower number of less Gaussian primitives.
The difference between our method and Taming3DGS
or LightGS lies in how the scoring function is imple-
mented. While they incorporate a scoring function in the
optimization loop to compress the final 3DGS scenes, we
opt to spatially score the scene at the initialization. There-
fore, we can furnish as many initial candidates as possible
for the splitting and pruning operations to optimize the ini-
tial Gaussians. Since the dense initialization, we observed
that our scenes usually contain a large number of very
low-opacity Gaussians and can be straightforwardly re-
moved by setting a threshold. Thus, our approach avoids
a sophisticated algorithmic design and parameter search
for reaching to a certain number of final Gaussians as
LightGS does. We can apply a straightforward pruning at
the 25,000-th iteration of our optimization to remove any
Gaussians with an opacity lower than 0.1. This simplic-
ity demonstrates the practical advantage of our proposed
method.
The proposed method can consistently avoid excessive
floating artifacts. This is exemplified in Figure 4, where
a non-test view is rendered by LightGS [8] (left) and our
method (right) are compared. We highlight the floaters
(in yellow) above the machines in LightGS and we do not
observe floaters around the tabletop.
7

<!-- page 8 -->
Figure 4: Novel-view renderings to showcase floating ar-
tifacts. Left: LightGS, Right: Our result.
PSNR ↑
SSIM ↑
LPIPS ↓
#G (k) ↓
Ours (ROI-aware)
28.5410
0.9109
0.1490
666
Uniform Sampling
28.3167
0.9095
0.1490
698
No Pruning at 25k
28.5463
0.9113
0.1467
1,344
Table 3: Ablation study on Beforehand Densification on
the Corner (C2) dataset.
4.3
Ablation and discussion
We first perform an ablation study on the ROI-aware im-
portance sampling. We report the results on Corner (C2)
dataset in Tab. 3. Our method and the ablated version us-
ing uniform sampling are given around 1.5 million Gaus-
sians as the budget. The proposed ROI-aware importance
sampling generates a dense point cloud from the depth
maps, which can lead to higher performance in all three
metrics. This validates the use of ROI-aware importance
sampling.
We also ablate our method without the late
pruning at 25,000 iterations and show that this straight-
forward pruning operation does not bring any adversary
effect to our method.
We also examine how the estimated depth influences
our results. Therefore, we compare our results produced
by the ground-truth depth maps and depth maps from
different sizes of the Metric3D-V2 (small, large, or gi-
ant) on the ICL living room dataset where the ground-
truth is available. Tab. 4 demonstrates that the render-
ing quality in term of the three metrics produced by each
method varies little, showing the robustness of the pro-
posed method in leveraging MDE methods’ results for
building 3DGS scenes.
The runtime for Metric3Dv2-
small and giant to estimate the depth maps from 200 RGB
images at resolution 1920 × 1440 is around 1.5 mins and
5 mins, respectively.
We also showcase an application in Fig. 5 where the
SAM can be applied to specify an ROI which further im-
PSNR ↑
SSIM ↑
LPIPS ↓
Time (s) ↓
#G (k) ↓
POpaq≤0.1 ↓
100%
31.726
0.9204
0.2479
258.45
445
5.13
20% + M3D-S
31.8126
0.9212
0.2471
263.77
375
4.88
20% + M3D-L
31.8459
0.9213
0.2483
262.31
370
4.83
20% + M3D-G
31.8018
0.9212
0.2478
261.02
372
4.73
Table 4: Validation of our method using different sizes of
Metric3D-V2 models and ground-truth depth on the syn-
thetic ICL living room dataset.
Figure 5: User-specified semantic mask and improved
readability of the poster in the Staircase scene.
proves the readability of the poster shown in the Staircase
scene.
4.4
Limitations.
While our method achieves comparable results with
SOTA methods with a lower budget of Gaussian primi-
tives, we also point out some limitations of the proposed
method. First, our method relies on the monocular depth
estimation algorithms which may provide inaccurate re-
sults, especially for texture-rich objects (e.g., posters) in
our datasets. While our global-local scaling operations
can refine the depth maps in a degree, this will lead to
floaters. Second, we currently utilize the color variance
as the indicator of default ROI. In the future we will ex-
plore a better way to support users to define the ROI and
automatically distribute the samples on-the-fly during
5
Conclusion
In this paper, we propose a Densify-Beforehand approach
to prepare dense initial point cloud for 3DGS represen-
tation by leveraging the monocular depth estimation and
the sparse LiDAR point cloud provided by a miniaturized
LiDAR. This is to circumvent the limitations of existing
adaptive density control used in the original 3D Gaussian
Splatting, in particular, to avoid excessively introducing
points by the cloning operation. We compare the pro-
posed method to state-of-the-art methods, such as 3DGS,
PixelGS, Taming3DGS, and LightGaussian, and demon-
strate the effectiveness of the proposed method in recon-
structing the scene as well as advantage in computational
efficiency. We ablated the proposed method to validate
our design choices. We showcase our results in six di-
verse scenarios.
8

<!-- page 9 -->
References
[1] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and
P. Hedman. Mip-nerf 360: Unbounded anti-aliased neu-
ral radiance fields. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
5470–5479, 2022.
[2] G. Baruch, Z. Chen, A. Dehghan, T. Dimry, Y. Fei-
gin, P. Fu, T. Gebauer, B. Joffe, D. Kurz, A. Schwartz,
et al. Arkitscenes: A diverse real-world dataset for 3d in-
door scene understanding using mobile rgb-d data. arXiv
preprint arXiv:2111.08897, 2021.
[3] N. Behari, A. Young, S. Somasundaram, T. Klinghoffer,
A. Dave, and R. Raskar. Blurred lidar for sharper 3d: Ro-
bust handheld 3d scanning with diffuse lidar and rgb. arXiv
preprint arXiv:2411.19474, 2024.
[4] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. M¨uller.
Zoedepth: Zero-shot transfer by combining relative and
metric depth. CoRR, abs/2302.12288, 2023.
[5] A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos,
Y. Zhou, S. R. Richter, and V. Koltun. Depth pro: Sharp
monocular metric depth in less than a second.
CoRR,
abs/2410.02073, 2024.
[6] J. Cui, J. Cao, F. Zhao, Z. He, Y. Chen, Y. Zhong, L. Xu,
Y. Shi, Y. Zhang, and J. Yu. Letsgo: Large-scale garage
modeling and rendering via lidar-assisted gaussian primi-
tives. ACM Transactions on Graphics (TOG), 43(6):1–18,
2024.
[7] X. Deng, C. Diao, M. Li, R. Yu, and D. Xu.
Efficient
density control for 3d gaussian splatting. arXiv preprint
arXiv:2411.10133, 2024.
[8] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang.
Lightgaussian: Unbounded 3d gaussian compression with
15x reduction and 200+ fps, 2023.
[9] S. Hong, J. He, X. Zheng, C. Zheng, and S. Shen. Liv-
gaussmap: Lidar-inertial-visual fusion for real-time 3d ra-
diance field map rendering. IEEE Robotics and Automa-
tion Letters, 2024.
[10] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen,
K. Wang, G. Yu, C. Shen, and S. Shen. Metric3d v2: A
versatile monocular geometric foundation model for zero-
shot metric depth and surface normal estimation.
IEEE
Trans. Pattern Anal. Mach. Intell., 46(12):10579–10596,
2024.
[11] C. Jiang, R. Gao, K. Shao, Y. Wang, R. Xiong, and
Y. Zhang. Li-gs: Gaussian splatting with lidar incorporated
for accurate large-scale reconstruction. IEEE Robotics and
Automation Letters, 2024.
[12] S. Katz, A. Tal, and R. Basri. Direct visibility of point sets.
In ACM SIGGRAPH 2007 papers, pages 24–es. 2007.
[13] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang,
S. Scherer, D. Ramanan, and J. Luiten. Splatam: Splat
track & map 3d gaussians for dense rgb-d slam. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21357–21366, 2024.
[14] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis.
3d gaussian splatting for real-time radiance field rendering.
ACM Trans. Graph., 42(4):139–1, 2023.
[15] X. Lang, L. Li, H. Zhang, F. Xiong, M. Xu, Y. Liu, X. Zuo,
and J. Lv.
Gaussian-lic:
Photo-realistic lidar-inertial-
camera slam with 3d gaussian splatting.
arXiv preprint
arXiv:2404.06926, 2024.
[16] S. S. Mallick, R. Goel, B. Kerbl, M. Steinberger, F. V. Car-
rasco, and F. De La Torre.
Taming 3dgs: High-quality
radiance fields with limited resources. In SIGGRAPH Asia
2024 Conference Papers, pages 1–11, 2024.
[17] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison.
Gaussian splatting slam. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 18039–18048, 2024.
[18] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron,
R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as
neural radiance fields for view synthesis. Communications
of the ACM, 65(1):99–106, 2021.
[19] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and
G. Drettakis. Reducing the memory footprint of 3d gaus-
sian splatting.
Proceedings of the ACM on Computer
Graphics and Interactive Techniques, 7(1):1–17, 2024.
[20] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury,
G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga,
A. Desmaison, A. K¨opf, E. Z. Yang, Z. DeVito, M. Raison,
A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai,
and S. Chintala.
Pytorch: An imperative style, high-
performance deep learning library.
In H. M. Wallach,
H. Larochelle, A. Beygelzimer, F. d’Alch´e-Buc, E. B. Fox,
and R. Garnett, editors, Advances in Neural Information
Processing Systems 32: Annual Conference on Neural In-
formation Processing Systems 2019, NeurIPS 2019, De-
cember 8-14, 2019, Vancouver, BC, Canada, pages 8024–
8035, 2019.
[21] X. Ren, M. Turkulainen, J. Wang, O. Seiskari, I. Melekhov,
J. Kannala, and E. Rahtu. Ags-mesh: Adaptive gaussian
splatting and meshing with geometric priors for indoor
room reconstruction using smartphones.
arXiv preprint
arXiv:2411.19271, 2024.
[22] J. L. Sch¨onberger and J.-M. Frahm. Structure-from-motion
revisited. In Conference on Computer Vision and Pattern
Recognition (CVPR), 2016.
[23] R. Xiao, W. Liu, Y. Chen, and L. Hu. Liv-gs: Lidar-vision
integration for 3d gaussian splatting slam in outdoor envi-
ronments. IEEE Robotics and Automation Letters, 2024.
[24] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng,
and H. Zhao.
Depth anything V2.
In A. Globersons,
L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tom-
czak, and C. Zhang, editors, Advances in Neural Informa-
tion Processing Systems 38: Annual Conference on Neu-
ral Information Processing Systems 2024, NeurIPS 2024,
Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
[25] Y. Zhang, L. Lu, X. Luo, and J. Pan. Global bim-point
cloud registration and association for construction progress
monitoring.
Automation in Construction, 168:105796,
2024.
[26] Z. Zhang, W. Hu, Y. Lao, T. He, and H. Zhao. Pixel-gs:
Density control with pixel-aware gradient for 3d gaussian
splatting. In European Conference on Computer Vision,
pages 326–342. Springer, 2024.
9

<!-- page 10 -->
[27] C. Zheng, W. Xu, Z. Zou, T. Hua, C. Yuan, D. He, B. Zhou,
Z. Liu, J. Lin, F. Zhu, et al. Fast-livo2: Fast, direct lidar-
inertial-visual odometry. IEEE Transactions on Robotics,
2024.
10
