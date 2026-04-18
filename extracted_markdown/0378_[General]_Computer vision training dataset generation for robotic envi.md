<!-- page 1 -->
Computer vision training dataset generation for
robotic environments using Gaussian splatting
Patryk Niżeniec
1* and Marcin Iwanowski
1
1Institute of Engineering and Technology, Faculty of Physics,
Astronomy and Informatics, Nicolaus Copernicus University in Toruń,
87-100, Toruń, Poland.
*Corresponding author(s). E-mail(s): pnizeniec@umk.pl;
Contributing authors: iwanowski@fizyka.umk.pl;
Abstract
This paper introduces a novel pipeline for generating large-scale, highly realistic,
and automatically labeled datasets for computer vision tasks in robotic environ-
ments. Our approach addresses the critical challenges of the domain gap between
synthetic and real-world imagery and the time-consuming bottleneck of manual
annotation. We leverage 3D Gaussian Splatting (3DGS) to create photorealis-
tic representations of the operational environment and objects. These assets are
then used in a game engine where physics simulations create natural arrange-
ments. A novel, two-pass rendering technique combines the realism of splats with
a shadow map generated from proxy meshes. This map is then algorithmically
composited with the image to add both physically plausible shadows and sub-
tle highlights, significantly enhancing realism. Pixel-perfect segmentation masks
are generated automatically and formatted for direct use with object detection
models like YOLO. Our experiments show that a hybrid training strategy, com-
bining a small set of real images with a large volume of our synthetic data, yields
the best detection and segmentation performance, confirming this as an optimal
strategy for efficiently achieving robust and accurate models.
Keywords: dataset generation, computer vision, Gaussian splatting, synthetic
images, object detection, segmentation
1
arXiv:2512.13411v1  [cs.CV]  15 Dec 2025

<!-- page 2 -->
1 Introduction
Object detection and semantic segmentation play a crucial role in modern com-
puter vision. It allows fast and accurate identification of position, size, and type of
objects present within the digital image, which makes it applicable to many computer-
vision-based engineering tasks [1, 2], including robotics [3]. The rapid advancement
of robotics and industrial automation heavily relies on robust and accurate percep-
tion systems. Object detection and segmentation are fundamental tasks that enable
robots to understand and interact with their environment. Modern solutions, par-
ticularly those based on deep learning architectures like YOLO (You Only Look
Once) [4, 5], have achieved remarkable performance. However, their effectiveness is
directly proportional to the quality and quantity of the training data.
Object detection and semantic segmentation tasks belong to supervised learning,
meaning they require annotated training data – images with appropriate labeling.
Creating large-scale, precisely annotated datasets is a significant bottleneck in devel-
oping robotic vision systems. Manual labeling is a labor-intensive, time-consuming,
and error-prone process. Synthetic data generation emerges as a promising alterna-
tive, offering the potential to create vast, diverse, and perfectly labeled datasets at
a fraction of the cost. However, a significant challenge with synthetic data is the
"domain gap" – the discrepancy between the appearance of synthetic images and their
real-world counterparts, which can degrade model performance.
This paper introduces a novel pipeline for generating highly realistic, automatically
labeled images – training data for robotic environments. Our approach leverages 3D
Gaussian Splatting (3DGS), a state-of-the-art neural rendering technique, to create
photorealistic representations of operational environment and individual objects. To
further bridge the domain gap, we introduce a unique method for simulating and
compositing physically-based shadows, a crucial visual cue often missing in other
synthetic data generation methods.
To validate the effectiveness of our proposed pipeline, we conduct a series of exper-
iments. We train several YOLO11 models of varying sizes on three distinct datasets:
one consisting of real-world images , a second generated entirely by our pipeline , and
a hybrid set that combines both data types. The performance of each model is then
evaluated on a consistent, held-out test set of real images. This comparison is quanti-
fied using standard mean Average Precision (mAP) metrics for both object detection
and segmentation.
Our main contributions are:
1. a complete, end-to-end pipeline for generating synthetic datasets using Gaussian
Splatting,
2. a novel technique for isolating object splats and integrating them with physics-
enabled meshes, and
3. a hybrid rendering approach that combines photorealistic splats with simulated
shadows to enhance realism.
We demonstrate through experiments that a model trained on a hybrid dataset,
combining a small number of real images with our synthetic data, outperforms models
trained on either real or synthetic data alone.
2

<!-- page 3 -->
This paper consists of 6 sections. Section 2 summarizes related works. In section
3, preliminary issues are introduced. Section 4 focuses on the details of the pro-
posed approach. Section 5 summarizes the results of experiments. Finally, section
6 concludes the paper. The project page is available at: https://patrykni.github.io/
UnitySplat2Data/.
2 Related Works
Large annotated datasets remain a critical bottleneck for deep learning in vision and
robotics. Synthetic data generation offers a scalable solution, and early efforts relied on
3D rendering engines such as Blender or Unity, or simulation platforms like Falcon [6].
While powerful, these methods require significant manual effort to create high-quality
3D assets and accurately model environments to bridge the "domain–gap" to real-
world conditions. Simpler strategies like “Cut-and-Paste” [7] expand datasets by
compositing object crops onto backgrounds, but suffer from limited viewpoint diversity
and contextual artifacts.
The IJCV 2018 editorial [8] articulated the trade-offs between realism, scalability,
and the simulation-to-reality gap. Subsequent works confirmed the value of synthetic
data: [9] showed foggy-scene synthesis improves detection under adverse conditions;
[10] presented SyDog-Video, demonstrating gains in temporal pose estimation from
synthetic training; and [11] proposed SynthStab, coupling a new dataset with a RAFT-
based stabilizer. In recent paper [12], the leveraged multifaceted synthetic corpora to
enhance compositional object detection was proposed. Procedural pipelines such as
BlenderProc automate photorealistic rendering with labels [13], while domain random-
ization [14] and benchmarks like Syn2Real [15] highlight both successes and remaining
gaps in transfer.
Neural radiance fields (NeRF) [16] inspired interest in learned scene represen-
tations, but rendering costs limit scalability. 3D Gaussian Splatting (3DGS) [17]
overcomes this with real-time, high-fidelity novel view synthesis, making it attractive
for dataset pipelines. Several concurrent works explore its potential.
The advent of 3D Gaussian Splatting (3DGS) has introduced a new paradigm
for synthetic data generation, enabling the creation of photorealistic representations
directly from a set of posed multi-view images. This has inspired numerous works to
develop data generation pipelines leveraging this technology. Moreover, the broader
capture-reconstruct-generate architecture is increasingly recognized as an effective
framework in robotics, a trend highlighted in recent surveys [18] and demonstrated by
specific applications like SparseGrasp, which employs 3DGS for scene reconstruction
to facilitate robotic grasping [19].
The PEGASUS system [20] presents an architecture that is conceptually the clos-
est to ours. It separately reconstructs both environments and objects using 3DGS
and leverages a physics engine for natural object placement. While PEGASUS pro-
vides a robust framework for scene composition, our method introduces a significant
enhancement in visual fidelity through a dedicated post-processing pipeline for shadow
and light integration. Our approach generates a separate shadow pass from object
meshes, which is then algorithmically processed—involving steps like Gaussian blur
3

<!-- page 4 -->
and non-linear value mapping—before being composited with the splat render to cre-
ate geometrically-correct, soft shadows. This directly addresses a known limitation
of the PEGASUS system, which lacks realistic shadow rendering, thereby further
improving visual fidelity.
Other works integrate 3DGS into different frameworks. The "Cut-and-Splat"
method [21] composites 3DGS objects onto 2D background images, ingeniously using
monocular depth estimation to find plausible support surfaces for placement. However,
their approach does not simulate lighting or shadows, relying instead on appearance
augmentation to create visual variance. A different strategy is seen in the work [22],
where import 3DGS objects into a traditionally modeled 3D environment within
Unreal Engine. While our method also leverages a game engine (Unity), its role is fun-
damentally different. It relies entirely on Unreal Engine’s built-in renderer for their
final image output. In contrast, our approach uses a hybrid rendering technique. We
utilize Unity’s lighting system to generate a separate shadow pass by rendering the
object meshes. This shadow pass is then composited with a render of the photorealistic
splats. This final compositing step, which combines elements from different rendering
passes, allows us to create highly realistic images with plausible shadows, giving us
unique control over the final appearance.
A key limitation in many synthetic pipelines is the treatment of shadows and global
illumination, which strongly affect realism. Traditional techniques addressed seamless
compositing [23], but recent methods improve controllability: [24] generate shadows
from pixel height maps, and [25] harmonized inserted shadows with scene illumination
However, most 3DGS-based systems do not yet integrate physically plausible shad-
ows. PEGASUS [20] and Cut-and-Splat [21] lack shadow simulation, while [22] rely
on Unreal’s built-in renderer. In contrast, our approach introduces a hybrid rendering
strategy: we isolate splats for photorealism and combine them with a separate, physics-
based shadow pass generated in Unity. This compositing step yields controllable,
physically grounded shadows that improve both realism and detector robustness.
Orthogonal to rendering pipelines, some research improves the fidelity of source
assets. For instance, Moghadam et al. [26] demonstrated a method for simulating
manufacturing flaws by directly modifying 3D models to include realistic cracks and
imperfections. Such methods complement advances in rendering by ensuring that the
input assets themselves better approximate real-world variability.
Related to the above articles, our work is uniquely positioned at the intersec-
tion of these emerging techniques. Our pipeline addresses key realism gaps found in
related works by combining a fully 3DGS-based representation for both objects and
environments with a novel, explicit shadow simulation layer. This hybrid approach,
orchestrated within the Unity engine, aims to generate highly realistic, physically
plausible, and automatically annotated data for training robust robotic perception
models.
4

<!-- page 5 -->
3 Preliminaries
3.1 Object detection and segmentation
Object detection is a fundamental computer vision task that involves identifying and
localizing objects within an image, typically by drawing bounding boxes around them.
Instance segmentation is a more advanced task that goes a step further by predicting
a pixel-level mask for each detected object instance, providing a much more detailed
understanding of the object’s shape and boundaries [1, 2]. In our experiments, we
have chosen the YOLO (You Only Look Once) family of models, particularly the
recent versions developed by Ultralytics, which have become a de facto standard for
real-time object detection due to their high speed and accuracy [4, 5]. A key feature
of this architecture is its scalability; it is offered in several variants of increasing
size and complexity, typically denoted as nano (n), small (s), medium (m), large (l),
and extra-large (x). This range provides a direct trade-off between inference speed
and detection accuracy, allowing practitioners to select the optimal model for their
specific constraints, whether for real-time applications on resource-constrained devices
or for achieving maximum performance on high-end hardware. Notably, the training
pipeline for YOLO’s detection mode can accept data in a segmentation format, which
consists of a text file for each image [27]. In this format, each line represents a single
object instance and follows the structure: ‘<class_index> <x1> <y1> <x2> <y2>
...‘. The ‘class_index‘ is an integer identifying the object’s class, and the subsequent
pairs of numbers are the coordinates of the polygon vertices that define the object’s
segmentation mask. A key aspect of this format is that all coordinates are normalized
to a range of [0, 1] by dividing them by the image’s width and height, respectively.
Our proposed method leverages this capability by generating these precise, normalized
segmentation coordinates automatically.
3.2 Evaluation Metrics
The evaluation of object detection and segmentation models hinges on classifying each
prediction against a ground-truth annotation. The primary mechanism for this is the
Intersection over Union (IoU) score, which quantifies the degree of overlap between
the predicted and actual object boundaries. Based on a predefined IoU threshold, a
prediction is classified as a True Positive (TP) if it correctly identifies an object, a
False Positive (FP) if it is an incorrect detection, and a False Negative (FN) is noted
for any ground-truth object the model fails to find.
From these counts, two fundamental metrics are derived: Precision, which measures
the accuracy of the predictions (TP/(TP + FP)), and Recall, which measures the
model’s ability to find all relevant objects (TP/(TP + FN)). A trade-off typically
exists between these two; improving recall by detecting more objects can often lower
precision by introducing more errors. This relationship is captured by the Precision-
Recall (P-R) curve. The Average Precision (AP) provides a single, comprehensive score
for a class by calculating the area under this P-R curve, summarizing the model’s
performance across all confidence thresholds.
5

<!-- page 6 -->
The final metric, mean Average Precision (mAP), is the average of AP scores across
all object classes. In our experiments, we use two standard variants of this metric. The
mAP50 score is calculated using a lenient IoU threshold of 50%, rewarding models
for general object localization. The mAP50-95 score is a stricter evaluation, averaging
mAP scores over ten IoU thresholds from 50% to 95%, demanding high precision in
both location and shape. We report both Box mAP and Mask mAP, which apply these
principles to bounding boxes and pixel-level masks, respectively. Reporting both scores
thus provides a more comprehensive assessment, distinguishing between the model’s
ability to correctly identify objects and its capacity for their precise localization.
3.3 Gaussian splatting
3D Gaussian Splatting (3DGS) is a rasterization-based method for real-time rendering
of photorealistic virtual views from a set of input images – novel views synthesis [17].
In contrast to Neural Radiance Fields (NeRFs), which rely on continuous, implicit
scene representations and computationally expensive volumetric ray-marching [16],
3DGS utilizes an explicit and unstructured representation based on a collection of 3D
Gaussians. This approach achieves state-of-the-art visual quality while maintaining
competitive training times and enabling real-time rendering at high resolutions.
The core primitive of the method is a 3D Gaussian, which is defined by several
key attributes: a 3D position (mean) µ, an anisotropic 3D covariance matrix Σ (rep-
resenting its scale and rotation), a color (represented by Spherical Harmonics (SH)
coefficients to model view-dependent effects), and an opacity value α. This explicit
representation avoids the need for costly neural network evaluations during rendering,
which is a primary reason for its high performance [17].
The process, illustrated in Figure 1, begins with data acquisition, as depicted in
the "Input: Image Set" block of the diagram. This initial step involves capturing
photographs of a static scene from various viewpoints. To enhance geometric accuracy,
this input dataset can be augmented with depth maps corresponding to the input
images. Subsequently, in the "Sparse Point Cloud & Camera Poses" stage, Structure-
from-Motion (SfM) software, such as COLMAP, analyzes the input images [28] to
yield the precise 3D position and orientation of the camera for each photograph and
a sparse 3D point cloud that provides a preliminary geometric scaffold of the scene.
The third stage is the "Initialization of 3D Gaussians". In this step, the sparse point
cloud generated by SfM is transformed into an initial set of 3D Gaussians. Each point
from the cloud is converted into a single Gaussian primitive. The initial 3D position
and the base color of each Gaussian are directly inherited from the corresponding
point’s attributes in the SfM output. This provides a coarse but geometrically aligned
starting point for the subsequent optimization phase.
The core of the method is an iterative optimization loop, represented by the inter-
connected "Differentiable Rasterization" and "Optimization & Adaptive Control"
blocks of a diagram. The loop begins with rasterization, a process that projects all
3D Gaussians onto a 2D image plane to create a rendered image. This rasterizer is
differentiable, with is a crucial property allowing the calculation of gradients for every
pixel with respect to the parameters of the Gaussians that influenced it. The rendered
image is then compared against the corresponding ground-truth photograph, and a
6

<!-- page 7 -->
loss value is computed. If depth maps were provided as input, an additional depth loss
term is calculated by comparing the rendered depth with the input depth map. This
depth-regularized loss provides a strong geometric guide, helping to reduce artifacts
and improve the accuracy of the reconstruction.
This loss value then drives the "Optimization & Adaptive Control" stage. Using
the gradients obtained from the differentiable rasterization, an optimization algo-
rithm updates all the parameters of the 3D Gaussians to minimize the loss. This
includes not only their 3D position µ, but also their anisotropic covariance Σ (which
defines their 3D scale and rotation), their opacity α, and their Spherical Harmonics
(SH) coefficients, which model complex, view-dependent color effects. Concurrently,
an adaptive control mechanism refines the geometry. As shown on the left of the opti-
mization block, Gaussians in poorly represented areas are either cloned to add detail
(under-reconstruction) or split into smaller Gaussians to refine complex surfaces (over-
reconstruction). This entire loop of rasterization, loss calculation, and optimization is
repeated thousands of times.
Fig. 1: The 3D Gaussian Splatting pipeline.
Once the optimization process converges and the loss is minimized, the final set of
optimized 3D Gaussians constitutes the "Final Photorealistic Representation". This
output is essentially a dense, highly detailed point cloud where each point contains rich
information about position, shape, color, and opacity. This final representation can
be saved and loaded into compatible viewers that utilize the same fast rasterization
technique to enable real-time, high-quality navigation and rendering of novel views of
the captured scene.
7

<!-- page 8 -->
4 Proposed method
Our proposed pipeline is an end-to-end system designed to transform a series
of photographs into a large-scale, accurately labeled dataset suitable for training
deep-learning-based object-detection and segmentation models. The entire process,
illustrated in Figure 2, is divided into two main stages: "Asset Acquisition and Prepa-
ration", and "Synthetic Scene Generation and Rendering". The process begins with
image acquisition, followed by the creation of 3D assets (splats and meshes). Scene
composition and physics simulation take place in the Unity engine. The final image is
created through hybrid rendering of splats and shadows, and labels are generated auto-
matically, creating a complete dataset for training object detection and segmentation
models, such as YOLO.
Fig. 2: Diagram of the proposed method for generating synthetic datasets.
4.1 Asset Acquisition and Preparation
The first stage involves creating high-fidelity digital representations of the visions
system environment and the objects of interest. The quality of these assets is funda-
mentally dependent on the initial set of captured images, the number and nature of
which are dictated by the requirements of the Structure-from-Motion (SfM) algorithm.
Successful SfM reconstruction is contingent upon providing a sufficient number of
images with significant visual overlap, allowing the algorithm to match corresponding
features and accurately determine camera poses. The acquisition technique involves
capturing images from a multitude of viewpoints to ensure complete coverage. The
8

<!-- page 9 -->
required number of images depends on the subject’s size and complexity; a large,
detailed environment necessitates more images than a single object. While extracting
frames from video is an option, we found that individual photos yield higher-quality
results, avoiding issues like motion blur that can degrade reconstruction.
A counterintuitively critical aspect of this stage, particularly for individual objects,
is the nature of the background. A feature-rich, textured background provides a wealth
of stable keypoints essential for robust camera pose estimation, especially when the
object itself is smooth. The ideal background should have a non-repetitive, high-
frequency texture.
Our specific process, as detailed in the left panel of Figure 2, begins by capturing
a comprehensive set of images of the target environment. These images, along with
corresponding depth maps generated using the Depth Anything v2 model [29], are
processed to create a photorealistic 3D Gaussian Splat of the entire static scene.
For each individual object, we perform several parallel tasks. First, camera poses
are determined from the original, unmodified images using a Structure-from-Motion
implementation (COLMAP [28]). These poses are a prerequisite for generating an
initial 3DGS of the object with its feature-rich background intact. Concurrently, clean
object masks are created by programmatically removing the background from the
same image set, for which we utilized the REMBG library [30]. Finally, a precise
mesh of the object is generated using these background-free images and the previously
calculated camera poses. This is achieved through a differentiable inverse graphics
approach, implemented with the nvdiffrec tool [31]. Finally, to isolate the object’s
splat representation, we perform the ’Background removal from splat’ step shown
on the diagram by projecting the generated masks into the 3D scene and filtering
the initial Gaussian point cloud. A crucial final step is to align the generated mesh
with its corresponding splat representation. The nvdiffrec tool automatically applies
a transformation to the input camera poses to center and scale the object, which
simplifies the mesh generation process. However, this results in a coordinate system
mismatch between the output mesh and the original Gaussian Splat. To rectify this,
we apply the ’Inverse transform’, ensuring both assets are perfectly co-located. The
outcome of this process is a database of asset pairs, where each object is represented
by both its optimized splat representation and its corresponding aligned mesh.
4.2 Synthetic Scene Generation and Rendering
With all assets prepared, we move to the "Synthetic Scene Generation and Rendering"
stage shown on the right of the diagram. We use the Unity engine, enhanced with the
UnityGaussianSplatting library [32], to generate the final dataset. A new scene is cre-
ated for each environment, where we load the environment splat and augment it with
simple 3D shapes. These shapes serve multiple purposes: some are visible elements
with colliders, such as a tabletop, while others are completely invisible colliders that
form an enclosure around the workspace, preventing objects from falling off during
the physics simulation. The previously generated assets for each object—the photore-
alistic Gaussian Splat and its corresponding collision mesh from nvdiffrec—are then
loaded into the scene. A script ensures that each splat’s transform matches its mesh
9

<!-- page 10 -->
after physics simulation. For each generation series, object meshes are randomly posi-
tioned above the table and dropped, allowing the ’Physics simulation’ step to create
a natural resting pose. The scene’s camera and light sources are also animated along
predefined paths to ensure variety, as indicated by the "Camera and light movement"
block.
Fig. 3: Hybrid compositing pipeline
A key innovation of our method is a two-pass hybrid rendering approach, shown in
Figure 3. For each frame, a photorealistic render of the Gaussian Splats is generated
(the "Appearance pass") alongside a separate pass from the simple object meshes (the
"Shadow pass"). Importantly, this pass includes meshes of both the objects and key
scene elements (e.g., the tabletop), allowing the system to naturally handle complex
interactions such as shadows cast by one object onto another and by the environment
onto the objects. This shadow map undergoes a multi-stage algorithmic process: it
is first normalized and softened using a Gaussian blur, then its tonal transitions are
refined with a sigmoid curve. The final composition multiplicatively applies these pro-
cessed shadows to the splat render. Furthermore, the same map is used to create a
highlight mask, which is additively blended with the image to simulate light reflec-
tions. This advanced compositing technique combines the photorealism of Gaussian
Splatting with physically plausible shadows and lighting. To further increase data
10

<!-- page 11 -->
variance, random post-processing effects such as adjustments to hue, exposure, and
noise are applied.
4.3 Automated Label Generation
Leveraging the controlled environment of the game engine, we can generate pixel-
perfect labels automatically. This process is visualized in Figure 4. For each final
image, we create a corresponding instance segmentation mask. This is achieved by
rendering each object’s mesh to an in-memory buffer with a unique solid color, respect-
ing the render order to handle occlusions correctly. The resulting multi-colored "ID
map" is then processed; by identifying pixels of a specific color, we extract the precise
contour of each visible object part. Small, noisy mask fragments are filtered out. The
final output for each generated scene is a pair: the composite image with shadows and
a text file containing the class and segmentation coordinates in required data format
e.g. the YOLO standard.
Fig. 4: The automated labeling process. From left to right: the final rendered image,
the generated instance segmentation masks (ID map), and the final image with bound-
ing boxes and segmentation contours applied.
5 Experiments
Our proposed pipeline is primarily designed for robotics applications, specifically for
perception systems operating within an industrial robot’s workspace. To validate
our approach in its intended context, all experiments were conducted within such a
scenario. The setup involved various objects of interest being placed within the oper-
ational area of a robotic workstation, simulating a typical environment for tasks like
robotic pick-and-place or assembly (Figure 5).
To evaluate our proposed data generation pipeline, we conducted a series of exper-
iments. The first step involved applying the acquisition method described in Section
4.1 to capture the source images for our 3D assets. Through empirical evaluation, we
found that approximately 300 images were sufficient for a complex environment (a
robotic workstation), while around 60 images yielded a high-quality reconstruction for
a single object. These values represent a practical balance, ensuring sufficient feature
coverage for a high-quality SfM reconstruction while avoiding the diminishing returns
associated with a significantly larger image set. Examples of the captured images are
shown in Figure 6.
11

<!-- page 12 -->
Fig. 5: Objects used in the experiments (top row) and example scenes showing their
example placement in the robotic workstations.
With the 3D assets prepared, we proceeded to generate our synthetic dataset. To
evaluate its effectiveness, we designed a series of experiments to compare the perfor-
mance of a YOLO11 model [27] trained on different data configurations. Specifically,
we trained separate models on a purely real dataset, our purely synthetic dataset, and
a hybrid combination of both. All models were then benchmarked on a consistent,
held-out test set of real images. The primary goal was to quantify the performance of
a model trained on our synthetic data against one trained on real-world data and to
assess the benefits of a hybrid training strategy.
5.1 Datasets
We prepared several distinct datasets for our experiments to thoroughly test different
training strategies. All models were ultimately evaluated on the same real test set to
ensure a fair and consistent comparison.
• Real Training Set: This dataset consists of manually captured and annotated
images from our robotic lab environments. To analyze the impact of dataset size,
we created two versions: a smaller one with 25 training images and a larger one with
75 training images. Both versions share a validation set of 30 images. These images
were captured across three different robotic workstations to provide a variety of
background contexts. Example images are shown in Figure 7.
• Synthetic Training Set: This dataset was generated entirely using our proposed
pipeline and contains 900 training images and 300 validation images. The significant
disparity in size compared to the real dataset highlights a key advantage of our
method: the ability to rapidly generate large volumes of perfectly labeled data with
minimal effort. Example images are shown in Figure 8.
12

<!-- page 13 -->
Fig. 6: Example images from the data acquisition stage, showing the capture process
for both a single object (top) and the environment (bottom).
• Hybrid Training Set: To evaluate the benefits of combining real and synthetic
data, we created a hybrid set. It consists of the larger real training set (75 images)
merged with the synthetic set (900 images), for a total of 975 training images.
The validation set is similarly combined, resulting in 330 images (30 real + 300
synthetic).
• Real Test Set: This is a separate, held-out dataset of 93 real-world images, which
was not used during the training of any model. To properly test the model’s general-
ization capabilities, this data was collected from two different robotic workstations
that were not part of the training set. One of these workstations included two
distinct table setups, effectively creating three unique test environments. Example
images are shown in Figure 9.
Fig. 7: Example images from the Real Training Set.
13

<!-- page 14 -->
Fig. 8: Example images from the Synthetic Training Set.
Fig. 9: Example images from the Real Test Set.
5.2 Experimental Setup and Results
For the experiments, we trained several YOLO models of varying sizes on each of the
prepared training sets. A batch size of 8 was used for all training runs. This value
was selected as the maximum feasible size constrained by the available GPU memory
for our largest model variant; it was kept consistent across all models to ensure a
fair and controlled comparison. All models were trained for 150 epochs, a duration
determined empirically to provide a balance between achieving model convergence
and avoiding significant overfitting, while also maintaining computational efficiency.
Other hyperparameters were kept at their default values. The performance of each
trained model was then evaluated on the real test set using the mean Average Precision
(mAP) metric, focusing on the mAP50 and mAP50-95 scores for both detection (Box
mAP) and segmentation (Mask mAP).
The detailed results are presented in Table 1, with a graphical representation in
Figures 10 through 13. The Hybrid Training Set consistently achieves the highest
performance across all model sizes and on all metrics. This strongly suggests that
our synthetic data serves as a powerful augmentation to a smaller real dataset. The
hybrid approach combines the two main advantages of the other sets: the high domain
fidelity of the real images, which were captured with the same camera as the test
set, and the vast variation in object poses, lighting, and backgrounds provided by the
large synthetic dataset.
The model trained on the larger Real Training Set (75 images) generally outper-
forms the model trained purely on synthetic data, especially on the stricter mAP50-95
metric. This is expected, as it minimizes the domain gap related to camera sensor
characteristics. However, its performance is still notably lower than that of the hybrid
14

<!-- page 15 -->
Table 1: Detailed performance metrics for all trained models on the test set. The best results for
each metric and model size are shown in bold.
Training Set
Model size
(M)
Box mAP50
(%)
Box mAP50-95
(%)
Mask mAP50
(%)
Mask mAP50-95
(%)
YOLO11n
Hybrid
2.84
96.08
85.73
96.25
83.25
Real
2.84
91.64
75.36
91.19
73.95
Synthetic
2.84
88.80
71.18
89.04
69.08
YOLO11s
Hybrid
9.43
98.99
88.69
98.41
86.55
Real
9.43
98.23
85.24
97.02
82.51
Synthetic
9.43
94.27
77.12
91.62
74.23
YOLO11m
Hybrid
20.06
98.59
87.73
97.85
85.24
Real
20.06
97.89
85.01
96.54
82.09
Synthetic
20.06
94.88
79.25
92.55
76.88
YOLO11l
Hybrid
25.31
98.84
88.05
98.11
85.73
Real
25.31
98.15
85.33
96.98
82.47
Synthetic
25.31
95.01
79.58
92.98
77.01
YOLO11x
Hybrid
56.88
98.77
87.93
98.05
85.58
Real
56.88
98.02
85.19
96.88
82.13
Synthetic
56.88
95.23
80.12
93.12
77.54
Fig. 10: Performance comparison for detection (Box mAP) at IoU threshold 0.5.
15

<!-- page 16 -->
Fig. 11: Performance comparison for segmentation (Mask mAP) at IoU threshold 0.5.
Fig. 12: Performance comparison for detection (Box mAP) averaged over IoU thresh-
olds from 0.5 to 0.95.
model, highlighting the limitations of relying solely on a small number of real-world
examples.
The Synthetic Training Set on its own achieves a respectable performance, vali-
dating the quality of our data generation pipeline. While it is outperformed by the
other sets, it establishes a strong baseline that could be crucial in scenarios where
collecting any real data is impractical. The performance gap can be attributed to the
domain shift between the synthetic renders and the real test images.
The trends observed for both detection (Box mAP) and segmentation (Mask mAP)
are remarkably similar, confirming that our pipeline generates accurate segmentation
masks that effectively teach the model the precise shapes of the objects, leading to
high-quality instance segmentation.
16

<!-- page 17 -->
Fig. 13: Performance comparison for segmentation (Mask mAP) averaged over IoU
thresholds from 0.5 to 0.95.
6 Conclusions
In this paper, we have presented a comprehensive, end-to-end pipeline for generat-
ing high-fidelity, automatically labeled synthetic datasets for training object detectors
in robotic applications. Our method successfully leverages the photorealism of 3D
Gaussian Splatting and introduces a novel hybrid rendering technique that compos-
ites physically-plausible shadows onto the scene. This approach effectively addresses
the critical challenge of the domain gap between synthetic and real-world imagery,
while the automated nature of the pipeline eliminates the time-consuming manual
annotation bottleneck.
Our experiments systematically demonstrated the value of this approach. The
results unequivocally show that the hybrid training strategy yields the best detection
and segmentation performance across all tested models. This confirms that our syn-
thetic data acts as a powerful augmentation, combining the sheer volume and variety
of generated images with the high domain fidelity of real ones. It is important to note
that the real training images were captured with the same camera as the test set, which
gives them an inherent advantage. Consequently, the purely synthetic dataset, while
achieving respectable results, was consistently outperformed by datasets containing
real images. Nevertheless, its strong standalone performance validates our pipeline as
a viable tool for bootstrapping perception systems in scenarios where collecting real
data is impractical. Our findings confirm that combining a small set of real images
with a large volume of our generated data is the optimal strategy for achieving robust
and accurate models.
Statements and Declarations
Funding
The authors did not receive support from any organization for the submitted work.
17

<!-- page 18 -->
Competing Interests
The authors have no relevant financial or non-financial interests to disclose.
Data, Material, and Code Availability
The datasets and source code generated during the current study are available from
the corresponding author on reasonable request for the purpose of peer review. Upon
publication, the materials will be made publicly and permanently available via a
GitHub Pages site.
References
[1] Liu, L., Ouyang, W., Wang, X., Fieguth, P., Chen, J., Liu, X., Pietikäi-
nen, M.: Deep learning for generic object detection: A survey. International
Journal of Computer Vision 128, 261–318 (2020) https://doi.org/10.1007/
s11263-019-01247-4
[2] Ren, J., Bi, Z., Niu, Q., Liu, J., Peng, B., Zhang, S., Pan, X., Wang, J., Chen, K.,
Yin, C.H., Feng, P., Wen, Y., Wang, T., Chen, S., Li, M., Xu, J., Liu, M.: Deep
Learning and Machine Learning – Object Detection and Semantic Segmentation:
From Theory to Applications (2024). https://arxiv.org/abs/2410.15584
[3] Manakitsa, N.: A review of machine learning and deep learning for object
detection, semantic segmentation, and human action recognition in machine
and robotic vision. Technologies 12(2), 15 (2024) https://doi.org/10.3390/
technologies12020015
[4] Kotthapalli, e.a.: YOLOv1 to YOLOv11: A Comprehensive Survey of Real-Time
Object Detection Innovations and Challenges (2025). https://doi.org/10.48550/
arXiv.2508.02067
[5] Ramos, L.T., Sappa, A.D.: A Decade of You Only Look Once (YOLO) for Object
Detection (2025). https://arxiv.org/html/2504.18586v1
[6] Rebekah Bogdanoff, M.S.: Training YOLOv8 with Synthetic Data from Falcon.
https://www.duality.ai/blog/training-yolov8-with-synthetic-data.
Accessed:
2025-08-25 (2025)
[7] Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T.-Y., Cubuk, E.D., Le, Q.V.,
Zoph, B.: Simple copy-paste is a strong data augmentation method for instance
segmentation. 2021 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2917–2927 (2020) https://doi.org/10.1109/CVPR46437.
2021.00294
[8] Gaidon, A., Lopez, A.M., Perronnin, F.: The reasonable effectiveness of synthetic
visual data. International Journal of Computer Vision 126, 899–901 (2018) https:
//doi.org/10.1007/s11263-018-1108-0
18

<!-- page 19 -->
[9] Sakaridis, C., Dai, D., Van Gool, L.: Semantic foggy scene understanding with
synthetic data. International Journal of Computer Vision 126, 973–992 (2018)
https://doi.org/10.1007/s11263-018-1072-8
[10] Shooter, M., Malleson, C., Hilton, A.: Sydog-video: A synthetic dog video dataset
for temporal pose estimation. International Journal of Computer Vision 132,
1986–2002 (2024) https://doi.org/10.1007/s11263-023-01946-z
[11] Souza, M., Almeida Maia, H., Pedrini, H.: Naft and synthstab: A raft-based
network and a synthetic dataset for digital video stabilization. International
Journal of Computer Vision 133(5), 2345–2370 (2025) https://doi.org/10.1007/
s11263-024-02264-8
[12] Park, K., An, S., Lee, Y.J., Kim, D.: Learning compositionality from multi-
faceted synthetic data for language-based object detection. International Journal
of Computer Vision (2025) https://doi.org/10.1007/s11263-025-02554-9
[13] Denninger, M., Winkelbauer, D., Sundermeyer, M., et al.: Blenderproc2: A pro-
cedural pipeline for photorealistic rendering. Journal of Open Source Software
8(82), 4901 (2023) https://doi.org/10.21105/joss.04901
[14] Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., Abbeel, P.: Domain
Randomization for Transferring Deep Neural Networks from Simulation to the
Real World (2017). https://arxiv.org/abs/1703.06907
[15] Peng, X., Usman, B., Saito, K., Kaushik, N., Hoffman, J., Saenko, K.: Syn2Real:
A New Benchmark for Synthetic-to-Real Visual Domain Adaptation (2018).
https://doi.org/10.48550/arXiv.1806.09755
[16] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R.,
Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis.
In: Proceedings of the European Conference on Computer Vision (ECCV), pp.
405–421. Springer, Cham (2020). https://doi.org/10.1007/978-3-030-58452-8_24
. https://arxiv.org/abs/2003.08934
[17] Kerbl, B., Kopanas, G., Leimkuehler, T., Drettakis, G.: 3d gaussian splatting
for real-time radiance field rendering. ACM Trans. Graph. 42(4) (2023) https:
//doi.org/10.1145/3592433
[18] Irshad, M.Z., Comi, M., Lin, Y.-C., Heppert, N., Valada, A., Ambrus, R., Kira, Z.,
Tremblay, J.: Neural fields in robotics: A survey. arXiv preprint arXiv:2410.20220
(2024) https://doi.org/10.48550/arXiv.2410.20220
[19] Yu, J., Ren, X., Gu, Y., Lin, H., Wang, T., Zhu, Y., Xu, H., Jiang, Y.-G., Xue,
X., Fu, Y.: Sparsegrasp: Robotic grasping via 3d semantic gaussian splatting
from sparse multi-view rgb images. arXiv preprint arXiv:2412.02140 (2024) https:
//doi.org/10.48550/arXiv.2412.02140
19

<!-- page 20 -->
[20] Meyer, L., Erich, F., Yoshiyasu, Y., Stamminger, M., Ando, N., Domae, Y.: Pega-
sus: Physically enhanced gaussian splatting simulation system for 6dof object
pose dataset generation. In: 2024 IEEE/RSJ International Conference on Intel-
ligent Robots and Systems (IROS), pp. 10710–10715 (2024). https://doi.org/10.
1109/IROS58592.2024.10802037
[21] Vanherle, B., Zoomers, B., Put, J., Van Reeth, F., Michiels, N.: Cut-and-Splat:
Leveraging Gaussian Splatting for Synthetic Data Generation (2025). https://
doi.org/10.48550/arXiv.2504.08473
[22] Deogan, A., Beks, W., Teurlings, P., Vos, K., Brand, M., Molengraft, R.: Synthetic
Dataset Generation for Autonomous Mobile Robots Using 3D Gaussian Splatting
(2025). https://doi.org/10.48550/arXiv.2506.05092
[23] Chuang, Y.-Y., Goldman, D.B., Curless, B., Salesin, D.H., Szeliski, R.: Shadow
matting and compositing. ACM Transactions on Graphics 22(3), 494–500 (2003)
https://doi.org/10.1145/882262.882298
[24] Sheng, Y., Liu, Y., Zhang, J., Yin, W., Oztireli, A.C., Zhang, H., Lin, Z., Shecht-
man, E., Benes, B.: Controllable shadow generation using pixel height maps. In:
Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel,
October 23–27, 2022, Proceedings, Part XXIII, pp. 240–256. Springer, Berlin,
Heidelberg (2022). https://doi.org/10.1007/978-3-031-20050-2_15 . https://doi.
org/10.1007/978-3-031-20050-2_15
[25] Valença, L., Zhang, J., Gharbi, M., Hold-Geoffroy, Y., Lalonde, J.-F.: Shadow
harmonization for realistic compositing. In: SIGGRAPH Asia 2023 Con-
ference Papers. SA ’23. Association for Computing Machinery, New York,
NY, USA (2023). https://doi.org/10.1145/3610548.3618227 . https://doi.org/10.
1145/3610548.3618227
[26] Moghadam, A., Bhatia, S., Kakhki, F.D., Ichikawa, H.: Integrating synthetic data
and deep learning for enhanced defect detection and quality assurance in manu-
facturing processes. Preprints (2025) https://doi.org/10.20944/preprints202501.
0204.v1
[27] Ultralytics Inc.: Ultralytics YOLO11: Real-Time Object Detection, Segmenta-
tion, and More. https://docs.ultralytics.com/models/yolo11/. Accessed: 2025-08-
26 (2024)
[28] Schönberger, J.L., Zheng, E., Pollefeys, M., Frahm, J.-M.: Pixelwise view selec-
tion for unstructured multi-view stereo. In: European Conference on Computer
Vision (ECCV), pp. 501–518. Springer, Cham (2016). https://doi.org/10.1007/
978-3-319-46487-9_31 . https://colmap.github.io/
[29] Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth
Anything V2 (2024). https://arxiv.org/abs/2406.09414
20

<!-- page 21 -->
[30] Dovahcrow:
Rembg:
Remove
Image
Background.
https://github.com/
danielgatis/rembg. Accessed: 2025-08-26 (2020)
[31] Munkberg, J., Hasselgren, J., Shen, H., Gao, J., Chen, W., Evans, A., Müller, T.,
Fidler, S.: Extracting triangular 3d models, materials, and lighting from images.
In: ACM SIGGRAPH Asia 2021 Conference Proceedings. Association for Com-
puting Machinery, New York, NY, USA (2021). https://doi.org/10.48550/arXiv.
2111.12503 . https://nvlabs.github.io/nvdiffrec
[32] Pranckevičius, A.: UnityGaussianSplatting: A Gaussian Splatting playground in
Unity. https://github.com/aras-p/UnityGaussianSplatting. Accessed: 2025-09-10
(2023)
21
