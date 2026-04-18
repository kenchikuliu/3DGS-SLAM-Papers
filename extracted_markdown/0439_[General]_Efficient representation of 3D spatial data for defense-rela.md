<!-- page 1 -->
Efficient representation of 3D spatial data for defense-related
applications
Benjamin Kahl, Marcus Hebel, and Michael Arens
Fraunhofer IOSB, Ettlingen, Fraunhofer Institute of Optronics, System Technologies and
Image Exploitation. Fraunhofer Center for Machine Learning.
Gutleuthausstr. 1, 76275 Ettlingen, Germany
ABSTRACT
Geospatial sensor data is essential for modern defense and security, offering indispensable 3D information for
situational awareness. This data, gathered from sources like lidar sensors and optical cameras, allows for the
creation of detailed models of operational environments.
In this paper, we provide a comparative analysis of traditional representation methods, such as point clouds,
voxel grids, and triangle meshes, alongside modern neural and implicit techniques like Neural Radiance Fields
(NeRFs) and 3D Gaussian Splatting (3DGS). Our evaluation reveals a fundamental trade-off: traditional models
offer robust geometric accuracy ideal for functional tasks like line-of-sight analysis and physics simulations, while
modern methods excel at producing high-fidelity, photorealistic visuals but often lack geometric reliability.
Based on these findings, we conclude that a hybrid approach is the most promising path forward.
We
propose a system architecture that combines a traditional mesh scaffold for geometric integrity with a neural
representation like 3DGS for visual detail, managed within a hierarchical scene structure to ensure scalability
and performance.
Keywords: Geospatial data, lidar, 3D data representation, neural fields, Gaussian splatting
1. INTRODUCTION
A critical differentiator in modern defense is the ability to perceive and act upon information faster and more
accurately than an adversary, a trend underscored by the rise of open source and social media intelligence.1,2
While many battle management systems still rely on traditional 2D interfaces that can limit spatial awareness,
emerging systems are increasingly incorporating algorithms from computer graphics and AI to build richer models
of the battlespace.3
A well-integrated 3D operational picture offers considerable benefits over 2D counterparts, enabling more in-
tuitive analysis for tasks like route optimization, line-of-sight assessments, and terrain masking, as can be seen in
Fig. 1. The primary technical challenge, however, lies in integrating, processing, and updating massive, hetero-
geneous data sets from disparate sensors in real-time. Raw sensor data is often too voluminous or unstructured
for immediate tactical use, making its transformation into a unified model a significant hurdle.
This paper examines the key stages of the data pipeline for creating such operational models. We analyze
common data sources, formats, and models and compare approaches for transforming raw data into coherent,
actionable 3D tactical representations. The goal is to identify best practices and future directions for enabling
effective decision-making through advanced digital battlespace visualization.
Further author information:
Benjamin Kahl: benjamin.kahl@iosb.fraunhofer.de, phone: +49 7243 992-393, ORCID: 0009-0009-3423-2823
Marcus Hebel: marcus.hebel@iosb.fraunhofer.de, phone: +49 7243 992-323, ORCID: 0000-0003-4301-7286
Michael Arens: michael.arens@iosb.fraunhofer.de, phone: +49 7243 992-147, ORCID: 0000-0002-7857-0332
Proc. of SPIE Vol. 13679 1367911-1
This is an author-prepared version.
The original publication can be found in the SPIE Digital
Library:
Benjamin Kahl, Marcus Hebel, and Michael Arens "Efficient representation of 3D spa-
tial data for defense-related applications", Proc.
SPIE 13679, Artificial Intelligence for Se-
curity and Defence Applications III, 1367911 (27 October 2025); https://doi.org/10.1117/12.3069693
Systematic reproduction and distribution, duplication of any material in this paper for a fee
or for commercial purposes, or modification of the content of the paper are prohibited.
Artificial Intelligence for Security and Defence Applications III, edited by Hugo J. Kuijf,
Radhakrishna Prabhu, Yitzhak Yitzhaky, Proc. of SPIE Vol. 13679, 1367911 2025
Published by SPIE · 0277-786X · doi: 10.1117/12.3069693
arXiv:2511.05109v1  [cs.GR]  7 Nov 2025

<!-- page 2 -->
Figure 1.
Screenshot from the popular geospatial information software WinTAK4 (left), and ambient-light lidar-scan of
the same area (right). A 3D representation can provide richer information and enhance decision-making.
1.1 Previous Work
The digital representation of 3D scenes has evolved from traditional explicit methods like point clouds, voxel
grids, and meshes to powerful implicit and neural representations.
Machine learning innovations have introduced powerful implicit and neural representations that can capture
complex scenes with remarkable fidelity from sensor data.
The concept of representing geometry implicitly
through a learned continuous function was notably advanced by Park et al.
(2019) with DeepSDF,5 which
leverages deep-learning techniques to encode a shape’s signed distance function into a neural field.
This idea was later expanded upon by Mildenhall et al. through the introduction of Neural Radiance Fields
(NeRF),6 which encode a scene’s direction-based radiance. A concept which has been adapted and implemented
into standardized tools, such as the notable Nerfstudio by Tancik et al. in 2022.7
In 2023, Kerbl et al.8 introduced 3D Gaussian Splatting (3DGS), a method that achieves real-time rendering
of high-fidelity scenes by representing them as a set of optimized 3D Gaussians.
Many of these techniques have already been extensively compared and analyzed. A 2024 survey by Wang9
provides a qualitative assessment of these various representation forms, from classic meshes to modern innovations
like NeRFs and Gaussian Splatting, and lists notable datasets for each one. Similarly, other surveys have focused
on more specific applications of machine learning to 3D data.
For instance, surveys by Shi et al.
(2023)10
and Li et al.
(2024)11 explore the burgeoning field of generative models and their transfer from 2D to 3D
representations. Earlier, a survey by Ahmed et al. (2019)12 reviewed the use of other deep learning techniques,
such as segmentation and recognition, on various 3D data representations.
In this paper, we assess many of the same representation forms, but remain focused on their applicability for
use-cases beyond rendering. Subsequently, we will propose a hybrid solution that addresses their shortcomings.
2. OVERVIEW
The flow of information within Command-and-Control (C2) systems is commonly separated into an OODA-loop
(observe, orient, direct and act) for decision making, and an Intelligence Cycle (IC) for assessments.13 ICs consist
of various continuous stages that typically involve a stage of directing and planning a data collection effort, the
collection of data itself, the subsequent processing of said data into a congruent package and the dissemination
of that package to the relevant end users.13
For the purposes of this paper, we restrict our scope to the stage of processing all available data into a
digestible representation form and the underlying methodology thereof.
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 3 -->
This process, which takes us from the raw data delivered by sensors to a usable, congruent application can
broadly be subdivided into the four stages of Collection, Fusion, Aggregation and Usage (as depicted in Fig. 2).
We will outline the processes of each stage below.
Figure 2. Generalized steps of a sensor-to-representation pipeline. Raw sensor data is collected, fused, then an aggregate
model is formed, which is then queried and visualized by the end-user application.
2.1 Collection Stage
Before a coherent 3D tactical map can be created, an acquisition of relevant data is necessary. Data can be
acquired through different kinds of sensors mounted to various platforms such as UAVs, ground vehicles, fixed
installations, satellites or otherwise.
Despite the broad diversity of platforms, the sensors involved can generally be classified into a few functional
categories:14
• Camera Sensors are passive sensors that capture 2D images by measuring electromagnetic radiation on
a photosensor array. While most cameras exclusively produce imagery in the visible light spectrum, more
specialized variants operate in other spectra, such as the infrared or ultraviolet. The resulting data is
typically delivered as a 2D array or texture, often geotagged and timestamped.
• Lidar Sensors are active sensors that emit laser pulses and measure their return times to determine the
distance to a reflecting surface. Common lidar architectures include a rotating array of emitters and sensors.
Each rotation produces a data-set including a measured distance (as well as if a distance was measured) for
each pulse. Some lidar sensors also capture additional data, such as the reflectivity or ambient illumination
of the surface (see Fig. 5).
This data is often managed in the form of 3D point clouds containing all laser hits. However, it is important
to note that, while point clouds are an effective way to represent solid surfaces, lidar systems also implicitly
measure volumetric information of a negative space of where lidar pulses passed through, confirming the
space as empty.
Depth-enabled cameras can serve a similar purpose to lidar sensors.
• Geolocation and Positional Sensors like intertial navigation systems (INS) often combine measure-
ments from multiple sub-sensors such as GNSS receivers, IMUs and odometry sensors to produce an
estimate on the measurement platform’s pose in some shared spatial reference frame. Raw data is often
delivered in a position and rotation in WGS84 or ECEF coordinates alongside a timestamp. Without
accurate positioning, even the most detailed image or point cloud has limited value.
• Miscellaneous Environmental Sensors include any auxiliary sensors that can provide contextual in-
formation about local conditions, such as temperature, humidity, atmospheric pressure, sound levels, or
radiation. While these are not always essential for constructing 3D spatial models, they may be crucial for
specific mission types or for enhancing situational awareness. These sensors usually produce scalar values
tagged with a time.
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 4 -->
2.2 Fusion Stage
The fusion stage is responsible for transforming heterogeneous sensor outputs into a unified spatiotemporal
format. This involves aligning data across different coordinate systems, synchronizing timestamps, and applying
necessary sensor-specific corrections. Effective fusion is a prerequisite for constructing a coherent model of the
environment, as it allows data from disparate sources to be meaningfully combined (see Fig. 3).
2.2.1 Processing
A fundamental aspect of sensor fusion is the normalization of positional and temporal references. Lidar mea-
surements and camera images need to be lined up and converted into a consistent global or mission-specific
coordinate space. Temporal synchronization is equally crucial; even small discrepancies in timestamp alignment
can lead to significant registration errors, especially for fast-moving platforms.
With an adequate positional alignment, the data from miscellaneous environmental sensors, such as tem-
perature, can be interpreted as a 3D vector field (or scalar field), where locations without measurements are
interpolated to the nearest available values.
Figure 3. Overview of the first two stages. Derivative data combines and filters raw data from multiple sensors.
Camera images need to be undistorted, which effectively means removing any distortion coming from lenses
and turning the image into something an equivalent pinhole camera would produce, where lines that are straight
in reality also appear as straight lines in the image.
This allows for an easier projection of points into the
image via a simple frustum-projection, as well as other operations. Furthermore, multiple camera images can
be stitched to form a unified panoramic image. If a camera’s pose is unknown, structure-from-motion (SfM)
methods, such as COLMAP,15,16 can be used to find out camera poses, as well as create additional point-clouds
from feature-matching.
2.2.2 Post-Processing
Once sensor data is unified, it undergoes various post-processing operations.
For camera images, this often
includes anonymization or adjustments to brightness and contrast.
For lidar data, common operations are
outlier removal and motion filtering.
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 5 -->
As shown in Fig. 3, a key advantage is the ability to perform cross-modality enrichment, where data from
one sensor enhances another. For example, camera images can be used to apply color to lidar point clouds, and
point clouds can be used to generate depth images.
The final, fused output for each timestamp is a coherent data bundle that combines images (including stitched
360° views), lidar point clouds, and any additional measurements as vector fields. This comprehensive dataset
serves as the foundation for creating coherent operational pictures.
2.3 Aggregation Stage
The datasets resulting from the fusion stage are detailed and cohesive, but too large to be queried, analyzed or
visualized efficiently. Thus, an aggregate model needs to be constructed that captures essential spatial structures
and semantics within a single structure, while minimizing redundancy and resource demands.
We will outline and compare five different forms of representation in the next chapter and evaluate these for
their individual strengths and weaknesses. Their adequacy is given by their performance based on the use-cases
required by the last stage of our pipeline.
2.4 Usage Stage
The final stage of the tactical-map pipeline centers on the utilization of the aggregated scene representations.
Once sensor data has been fused and compressed into a coherent model, its ultimate value is measured by how
well it supports the needs of the end user.
The exact details and requirements depend heavily on the nature of the user application and it’s purpose,
but some common procedures include (see Fig. 4):
• Line-of-sight analysis: Assessing visibility from a given location, accounting for terrain, obstacles, and
occlusions.
• Visualization: Rendering the environment in a way that is intuitive, photorealistic, or semantically
informative for human operators or decision systems. The computational complexity of visualization and
line-of-sight analysis are closely linked.
• Route planning: Identifying viable paths through the environment, often requiring integration with
semantic information (e.g., traversable surfaces, hazards).
• Physics/collision simulations: Virtual environments can be leveraged to insert new objects into them
and see how they interact with the surrounding environment.
• Change detection and monitoring: Comparing updated scans or mission data with existing maps to
detect structural changes, new objects, or intrusions.
• Simulation and synthetic data generation: Using the map as a foundation for simulating agents,
sensors, or physical phenomena in controlled environments.
2.5 Evaluation Criteria
Many of the computational processes required for the above listed use-cases rely on a similar set of operations.
For the sake of our evaluation, we will regard the following:
• Write Performance: The efficiency and ease of creating the representation from raw sensor data, as well
as updating or modifying an existing model.
• Memory Footprint: The amount of memory required to store the representation, which dictates hardware
constraints and scalability.
• Fidelity: The ability of the representation to accurately preserve different aspects of the source data. We
distinguish between:
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 6 -->
Figure 4.
Examples of common use cases for a 3D representation model.
– Surface Fidelity: How accurately the model represents surface geometry, typically captured by lidar
sensors.
– Visual Fidelity: How well the model preserves photorealistic appearance and texture information
from camera sensors.
– Volumetric Fidelity: The capacity to represent non-surface data, such as atmospheric measurements
or other phenomena within a volume.
• Computational Performance: The efficiency of the representation when used for common downstream
tasks, such as visibility estimation, rendering, route planning, or physics simulations.
In the following chapter, we assess how the various aggregation techniques perform across these dimensions.
It’s important to note that, in isolation, these criteria don’t provide a complete picture on the advantages and
disadvantages of each representation form.
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 7 -->
3. EVALUATION OF AGGREGATE MODELS
3.1 Point Clouds
A point cloud is a fundamental 3D representation consisting of a set of points P = {pi | pi ∈R3}N
i=1 in Euclidean
space. Each point pi can be augmented with additional attributes, such as color, intensity, or semantic labels
(see Fig.
5).
As a minimally processed output from lidar sensors, point clouds offer a simple and detailed
representation that remains faithful to the captured data. However, their data is often sparse, unstructured,
and unordered, lacking any explicit connectivity information.
This representation is also inherently limited;
it captures only the surface points where lidar pulses terminate, discarding valuable volumetric data about
scanned-through areas (free space) versus unscanned regions (unknown space), and often neglects rich contextual
information available from co-located cameras.
Figure 5.
An example of a multi-layer point-cloud captured by a lidar sensor,17 consising of points (top-left), pulse-
intensity (top-right), ambient light (bottom-left) and RGB colors captured by a complementary camera rig (bottom-
right).
Traditional algorithms for handling point clouds almost invariably rely on spatial acceleration structures, such
as k-d trees or octrees, to enable efficient spatial queries18,19 like nearest-neighbor searches or ray intersections.
Even with these structures, operations that depend on surface or volume information, such as line-of-sight
analysis, remain inefficient without inferring additional structure. Similarly, rendering point clouds often relies
on Level-of-Detail (LOD) systems and custom compute shaders due to regular GPU pipelines not being optimized
for point-rasterization.20,21
The unstructured nature of point clouds also poses a fundamental challenge for deep learning models, as the
convolutional kernels used in standard CNNs are not applicable to unordered sets. A seminal work addressing this
was PointNet,22 which introduced a neural network architecture capable of processing raw point clouds directly.
PointNet achieves permutation invariance by applying a shared MLP onto the input points independently and
then pooling the resulting feature vectors into a single, global feature vector using a symmetrical max-pooling
operation (see Fig. 6). Its successor, PointNet++,23 extended this concept by introducing a hierarchical archi-
tecture to capture local geometric features at multiple scales, effectively creating a multi-scale understanding of
the point cloud’s geometry. Despite these advances, extracting robust and meaningful features from sparse point
data remains an active area of research.10
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 8 -->
3.1.1 Surfels
By augmenting each point pi with a normal vector ni, it can be promoted into a surfel, si = (pi, ni).24 The key
innovation is that the rendering process for these surfels can be made differentiable.25,26
Given a depth-sorted set of surfels {si}N
i=1 intersected by a pixel ray, the final color C(x) of the pixel can be
calculated through following rendering equation:6,26
C(r) =
N
X
i=1
Ti(1 −exp(−σi))ci
(1)
where ci and σi are the color (or radiance) and relative volume density of the respective surfels at their
intersection points, and Ti is the accumulated transmittance along the ray Ti = exp(−Pi−1
j=1 σj). An example of
this process can be seen in Fig. 6.
Because this blending process is differentiable, gradients can be backpropagated from a rendering loss (e.g.,
the difference between the rendered image and a ground-truth photo) to the surfel parameters. This enables
gradient-based optimization to refine the 3D scene directly from 2D images.25,26
This principle forms the
theoretical basis for 3D Gaussian Splatting, which we evaluate in Section 3.5.
Figure 6.
a) Example of a point-cloud captured with a double-lidar setup using the MODISSA platform.17
b) The
network architecture behind PointNet’s pre-processing. Points undergo individual feature extraction. A set of learnable
TNet matrices ensure transformation invariance, while a symmetrical max-pooling layer ensures permutation invariance.
c) A rendition of the Stanford Bunny27 using surfels and d), An illustration of the differential rendering function of
surfels.26
3.2 Voxel Grids
A voxel grid discretizes a continuous 3D space into a regular lattice of cubic elements, or voxels. Formally, a
grid G of resolution Nx × Ny × Nz can be defined as a 3D tensor where each voxel vi,j,k ∈G stores a particular
value associated with its spatial location. This value can represent various properties, such as occupancy, color,
or signed distance values. Like point clouds, voxel grids are trivial to construct or modify and are superb for
storing volumetric information such as sensor coverage.
While primitive voxel grids have a cubic memory complexity (O(N 3)), in practice, most scenes are sparse,
meaning the majority of voxels represent empty space.
This observation motivates the use of sparse data
structures such as octrees.28 Although octrees can significantly alleviate memory requirements for sparse envi-
ronments, their irregular, pointer-based structure disrupts the contiguous memory layout that makes dense grids
computationally efficient.29
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 9 -->
Figure 7.
a) An example of how voxel grids can be constructed from lidar point-clouds and sensor coverage encoded as
a per-voxel certainty function. b) An example of how differences in this certainty function can easily be used to detect
changes in the environment. c) Illustration of the differentiable, volumentric rendering function of plenoxels.30 A ray
traversing a voxel yields a color and volume density value that is tri-linearly interpolated to the spherical harmonics on
the edges of the voxel. d) The concept behind voxel hash encoding. Instead of containing the scene itself, the voxel map
only points to learnable feature vectors in a hash-grid. Note that during read-out, the feature-vectors in the voxel-map
are typically tri-linearly interpolated on the queried position.
Depending on the acceleration structure used, the predictable, Euclidean structure of voxel grids allows for
rapid neighborhood lookups, making them adequate for operations like 3D convolution. On the other hand,
queries that require traversing the space, such as line-of-sight or ray casting, can be inefficient. A ray must be
”marched” through the grid cell by cell, and the precision of any intersection test is fundamentally limited by
the voxel size.
In the context of deep learning, their similarity to 2D images allowed for the straightforward application of
3D Convolutional Neural Networks (CNNs). Early works like DeepVoxels31 for representation, 3D-R2N232 for
reconstruction and 3D-GAN33 for shape generation demonstrated the viability of this approach. However, the
aforementioned cubic memory complexity proved a significant hindrance on achievable resolution and training
times.10 As with other operations, acceleration structures like octrees can reduce this cost, but their non-regular
structure does not lend itself well to processing by neural networks.10
Subsequent research to overcome these limitations diverged into two main paradigms: The first paradigm
shifted focus from using grids as an input to a large network, to treating the grid itself as the set of optimizable
parameters. A prominent example is Plenoxels,30 which utilizes a sparse voxel grid where each cell stores not
only density but also coefficients for spherical harmonics (SH) to model view-dependent color. The entire set
of parameters is then optimized directly against a collection of training images using a differentiable volumetric
rendering process (see Fig. 7c). This philosophy of directly optimizing explicit geometric primitives has been
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 10 -->
further advanced by methods like 3D Gaussian Splatting,8 which replaces voxels with anisotropic 3D Gaussians
(see Section 3.5).
The second paradigm focused on using grids to accelerate implicit neural representations. Here, each grid cell
holds a learnable, high-dimensional feature vector rather than a final value. To query a continuous point p, the
feature vectors from its neighboring voxels are interpolated, and this interpolated feature is processed by a very
small multi-layer perceptron (MLP) to output the final density and color. This hybrid explicit-grid/implicit-
network design was powerfully demonstrated by Instant-NGPs34 (see Section 3.4.2).
3.3 Triangle Mesh
3D triangle meshes are among the most ubiquitous representations for 3D shapes. A mesh defines the surface
of an object as a collection of vertices, edges, and polygonal faces. The vertices specify coordinates in 3D space,
while the faces describe the surface connectivity between them. The widespread adoption of meshes in computer
graphics has resulted in a mature ecosystem of tools, algorithms, and hardware acceleration, making them a
compelling representation for virtual environments.
Modern Graphics Processing Units (GPUs) are highly optimized for rasterizing extremely large meshes and
can perform on-the-fly operations like morphing and tessellation in real-time. This hardware support facilitates
efficient computational operations. For instance, visibility analysis is handled through established techniques
like z-buffering and shadow mapping, which can be further accelerated by dedicated ray-tracing hardware on
modern GPUs.35 Furthermore, the application of 2D textures allows meshes to achieve a high degree of visual
fidelity at a very low memory cost.
Figure 8. Mesh of the same point-cloud with three different LODs (left one being lowest, right one being highest) and their
respective polygon-counts. In this example, mesh coloring is performed on a per-vertex level, and could be significantly
improved through the use of textures.
Meshes do, however, lack volumetric expressiveness and can only describe the boundary of a shape. This
limits them to binary volumetric distinctions (inside vs. outside) and prevents the representation of smooth
interior properties that would be achievable with voxel grids or neural fields. Additionally, reliably constructing
a topologically correct and accurate mesh from raw sensor data is a non-trivial challenge, often complicated by
noise and incomplete data.36
Applying deep learning methods to 3D meshes has historically been difficult due to their irregular, non-
Euclidean structure. Unlike images, which are organized in a regular pixel grid, meshes have complex and variable
vertex connectivity. Consequently, standard models like Convolutional Neural Networks (CNNs), designed for
grid-like data, cannot be directly applied.10 Generating a new mesh is also complex, as it requires synthesizing
not just vertex positions but also a plausible topological structure for the faces.10
To bridge this gap, differentiable rendering techniques have been developed. These methods enable a mesh
to ”learn” its own shape by comparing its rendered output to a set of target images, in much a similar fashion
as surfels, plenoxels or Gaussian Splatting. Among the most notable implementations is the Neural 3D Mesh
Renderer by Kato et al.37 It uses standard, non-differentiable rasterization for the forward pass, but employs a
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 11 -->
proxy function in the backward pass to approximate the gradient of the rasterization step. This approximation
is often sufficient to guide the optimization of the mesh’s geometry. In contrast, the Soft Rasterizer by Liu
et al.38,39 introduces a fully differentiable rendering pipeline. It reformulates rasterization as a probabilistic
process where every triangle contributes to the final color of each pixel, enabling a more direct and accurate flow
of gradients.
Beyond mesh generation and optimization, other methods focus on applying deep learning directly to the
analysis of existing mesh structures. A seminal work in this area is MeshCNN by Hanocka et al.40 MeshCNN
allows certain meshes to be used as inputs for traditional CNNs by treating the geometric properties of the meshes
edges analogously to pixels on an image and performing convolutions on rings of four neighboring edges in an
order consistent with the respective face normal.40 This allows fully convolution tasks, such as segmentation, to
be performed with traditional CNNs directly on meshes.
3.4 Neural Fields
At their simplest level, neural fields are multilayer perceptrons (MLPs, often with fully connected layers and no
external memory) that take spatial coordinates as an input and output a value associated with that location.
The concept was popularized by Mescheder et al. in 2019 with Occupancy Networks,41 where a 3D shape is
represented by an MLP that learns its occupancy function (i.e., whether a point is inside or outside the shape)
based on input coordinates. Park et al. later expanded this concept with DeepSDF,42 which improves rendering
performance and enables the inference of surface normals by encoding the space as a signed distance function
(SDF), which represents the distance to the nearest surface for every point in the represented space.
However, these early methods suffered from a significant spectral bias, meaning they were inherently biased
towards learning low-frequency functions and struggled to represent fine, high-frequency details like sharp edges,
textures, or intricate geometry. The breakthrough addressing this issue came when Mildenhall et al. intro-
duced Neural Radiance Fields (NeRFs).6 With NeRF, they borrowed a concept first introduced by Vaswani
et al.43 in the realm of natural language processing and applied it to the completely different domain of 3D
representations. This concept, known as positional encoding, maps the low-dimensional input coordinates into a
higher-dimensional feature space, making it easier for the network to learn high-frequency variations.
The specific implementation in NeRF is a deterministic mapping that creates a set of axis-aligned, exponen-
tially spaced frequency features. A point p = (x, y, z) in 3D space is transformed into a higher-dimensional point
by applying the following function to each component of the coordinate vector:
γ(p) = (sin(20πp), cos(20πp), ..., sin(2L−1πp), cos(2L−1πp))
(2)
Where L is the number of chosen frequencies. The final vector is the concatenation of these features for x,
y, and z, resulting in a (3 ∗2L)-dimensional vector. This mapping allows the MLP to more easily represent
high-frequency functions. Despite being popularized by NeRF, this technique is broadly applicable to improve
detail in any neural field.
A more general approach, analyzed in detail by Tancik et al.,44 uses Random Fourier Features (RFF, see Fig.
9a). This method relies on a matrix Ωof randomly sampled frequency vectors, where each vector defines both
a direction and frequency:
γ(p) = (cos(2πΩp), sin(2πΩp))
(3)
While the RFF approach is theoretically powerful, the original NeRF paper found great success with the
first, simpler method using a fixed set of axis-aligned frequencies.
With these tools, neural fields have proven to be a powerful, continuous representation that can capture
complex volumetric data within a memory-efficient network. The optimization process naturally encourages the
network to use its limited capacity to represent the most important aspects of the environment.
However, neural fields are not trivial to train on raw sensor data, as the loss function often requires sampling
points in known empty space, not just on measured surfaces. Furthermore, they are typically cumbersome to
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 12 -->
modify. A ”targeted” update to a small area is difficult, as a local change can affect the global representation,
usually requiring a full re-training of the network.
Similarly, performing traditional operations such as visibility determination or physics simulations on a neural
field is computationally expensive, as these operations require repeatedly querying the network by ray-marching
through the represented volume.
Figure 9.
a) The concept behind Random Fourier Features (RFF) in 2D: The input coordinates are decomposed into a
set of fourier features with randomized frequencies and directions. b) The differentiable rendering process of NeRFs: A
ray is marched through the field, accumulating transmittance. For each sampling point, an inference of the network is
performed. c) The concept behind multi-resolution hash encoding used in Instant-NGPs: The input coordinates sample
a number of hashgrids with different resolutions for feature vectors that are tri-linearly interpolated for each grid, then
concatenated alongside the view direction. A small MLP interprets the combined feature vector to output the radiance.
3.4.1 Neural Radiance Fields (NeRFs)
In addition to the aforementioned positional encoding, Mildenhall et al.’s NeRFs distinguished themselves from
regular Neural Fields by introducing the viewing direction to the input of the network.6 Instead of just a position
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 13 -->
(x, y, z), a NeRF takes two angles encoding the viewing direction (x, y, z, θ, ϕ) for inputs. Thus, the network
produces not color, but a view-dependent radiance cr and volume density ωr, and that allows the representation
to mimic phenomena such as specular reflections or refractions. The network thus acts as a function of five
parameters and four outputs:
M(x, y, z, θ, ϕ) −→(r, g, b, σ)
(4)
The rendering of a NeRF image is performed using classical volume rendering (see Fig. 9b). For each pixel,
a camera ray r (given by θ and ϕ) is marched through the scene, and the MLP is queried at N sample points
along it. The final color C(r) is then computed by integrating the color and density values:
C(r) =
N
X
i=0
Ti,rci,r(1 −exp(−σi,rδi))
(5)
where ci,r and σi,r are the radiance and volume density at point i in direction r, given by the output of the
network, δi is the distance to the next sample point and Ti,r is the transmittance, which tracks the accumulated
occlusion of the ray through its sample points, given by Ti,r = exp(−Pi−1
j=1 σj,rδj).
Like with plenoxels or surfels, this process is differentiable, meaning the difference between a rendered image
and its original counterpart can be backpropagated all the way back through the volume rendering formula. This
allows NeRFs to be trained directly on a collection of camera images, enabling them to capture photorealistic
detail far beyond what could be reconstructed from geometric data like a lidar point cloud alone.
With these adaptations, NeRFs produce highly photorealistic results and have become a foundational tech-
nique in 3D computer graphics.9
Despite the impressive visual quality, the original NeRF architecture does not solve many of the drawbacks
inherent to neural fields. Namely, training times being extremely slow, targeted modifications being difficult,
and because the geometry is stored implicitly, querying the scene for tasks like collision detection or physics
simulations remains prohibitively expensive.
3.4.2 Instant-NGPs
The slow training times of NeRFs were ameliorated when M¨uller et al.
introduced Instant-NGPs (Neural
Graphics Primitives with a Multiresolution Hash Encoding).34 Instant-NGPs employ a hybrid explicit-implicit
representation with a learnable multi-resolution hash grid that maps input coordinates to sets of rich, multi-
dimensional feature vectors that encode features on various resolutions (see Fig. 9c).
Given a position p = (x, y, z), the explicit multi-resolution hash-grid (effectively a series of voxel maps pointing
to a limited set of learnable vectors), produces a combined set of vectors for each of the N resolutions:
H(x, y, z) −→(F0, F1, ...FN)
(6)
This combined feature vector, which now encodes multi-scale information about that point in space, is passed
to a small MLP. The viewing direction (θ, ϕ) is encoded separately and concatenated to an intermediate feature
layer within the MLP, just before the color is predicted. The network then produces the final radiance and
volume density:
M(F0, F1, ...FN, θ, ϕ) −→(r, g, b, σ)
(7)
By offloading the bulk of the scene representation from the slow, large MLP to the fast, explicit hash grid,
the network’s task is simplified from representing the entire scene to just interpreting the rich features from the
grid. This allows for a much smaller MLP, which dramatically reduces training and inference times from hours
to mere seconds or minutes, all while producing results comparable to regular NeRFs.
While the improved training times make creation and modification operations more feasible, Instant-NGPs
are still ill-suited as representations for operations requiring many visibility determinations or collision/physics
simulations.
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 14 -->
3.5 Gaussian Splatting
3D Gaussian Splatting (3DGS), introduced by Kerbl et al. in 2023,8 is the newest major evolution in the line of
self-learning 3D representations. While NeRFs or Instant-NGPs store the scene representation implicitly within
the weights of their network, 3DGS is an explicit representation, where the trained parameters directly define a
set of discrete objects that make up the rendered geometry.
Similarly to surfels (see Section 3.1), a 3DGS representation consists of a set of primitives called Gaussians
which take the shape of anisotropic, 3-dimensional ellipsoids that have a position, covariance (shape and scale),
color and opacity. In more advanced implementations they are also outfitted with spherical harmonics parameters,
allowing them to adopt different colors when viewed from different directions.
A Gaussian Gi at position µi and covariance matrix Σi is given by the following equation:
Gi(p) = exp

−1
2(p −µi)T Σ−1
i (p −µi)

(8)
In contrast to NeRFs, the rendering (ie. ”splatting”) of Gaussians does not require ray-marching and re-
peated sampling. While exact implementation details can differ, the most common approach involves sorting
the Gaussians back-to-front and then performing a simple alpha-blend.8 The final color of a pixel affected by N
Gaussians (sorted by depth) is given by
C(p′) =
N
X
i=1
ciG′
i(p′)
i−1
Y
j=1
(1 −G′
j(p′))
(9)
where ci is the color of the i’th Gaussian and G′
i is the 2D projection of the Gaussian onto the viewing plane
(see Fig. 10). This entire process is differentiable, allowing the properties of the Gaussians to be optimized
directly to match a set of training images.
Figure 10. The process of training a 3DGS representation: 3D Gaussians are rasterized (”splatted”) onto a camera with
the same perspective as a reference image. The loss for each tile/pixel is then back-propagated through the projection to
the Gaussians’ parameters, as well as towards adaptive density control.
In their implementation, Kerbl et al. use a Structure-from-Motion process to initialize the Gaussians from a
sparse point cloud. Throughout training, these Gaussians are optimized using a process called adaptive density
control, which can split, clone, or prune Gaussians to better represent the scene’s geometry.
The results are very impressive (see Fig. 11). Renders of trained 3DGS representations are as photorealistic
as NeRFs, but can be generated in real-time. The explicit model can also be expanded, modified, or selectively
reduced with greater ease. Further improvements to this method have introduced distance-based LOD function-
ality or anti-aliasing, allowing for Gaussian Splatting to be used even on very large, complex environments.45,46
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 15 -->
Figure 11. A 3DGS render with the Gaussians scaled to 1% (left), 40% (center), and 100% (right) of their size respectively.
Unlike with point clouds, 3DGS only allocates a higher amount of Gaussians in locations with a coarse texture, acting as
a kind of ”compressed point cloud”.
Despite 3DGS’ superb visual quality, its primary weakness stems from the fact that the optimization is driven
purely by 2D image reconstruction. This can lead to geometrically inaccurate or ”hollow” representations that
look correct from the training viewpoints but may not reflect the true underlying scene structure47 (see Fig.
12). Consequently, operations that rely on accurate geometry, such as route planning, physics simulations, or
visibility calculations, can yield unreliable results. Furthermore, volumetric information, such as measurement
coverage, cannot be encoded as naturally as it could be in a voxel grid or a true volumetric neural field.
Figure 12. The same 3DGS representation rendered from different camera angles. Angles that are similar (top right) or
the same (top left) to the camera poses of the reference/training images are highly convicing. Diverging from the angles
of the training data (bottom) showcases the unreliable geometric properties of 3DGS. Bottom left shows the road being
partially transparent due to no reference image existing along that direction. Bottom right shows how coarse patterns in
the sky and ground are being mimicked through Gaussians that don’t accurately reflect the real surfaces.
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 16 -->
4. VERDICT
The preceding sections have surveyed a range of 3D representation paradigms, from traditional geometric prim-
itives like meshes and voxels to modern, self-learning implicit and explicit neural models.
In Table 1 we have listed estimated ratings for each representation archetype based on the evaluation criteria
outlined in Section 2.5 and illustrated their differences in Fig. 14. Whilst a true comparison would depend highly
on implementation specifics and parameters, the overall trend reveals that the visual fidelity of modern methods
comes at a cost in performance in other criteria.
As such, we believe a praxis-oriented implementation requires moving away from a monolithic, single-purpose
representation and toward a practical, hybrid system. In this section, we explore tentative ideas and concepts
for how such a hybrid system may be structured.
Table 1. A Comparison of 3D Representation Methods
Metric
Point Cloud
Voxel Grid
Mesh
Neural Field
3DGS
Write Performance
*****
*****
**
*
***
Memory Footprint
*
**
****
*****
*****
Computational Performance
*
***
*****
*
*
Surface Fidelity (Lidar)
*****
***
*****
**
**
Visual Fidelity (Camera)
**
*
**
*****
*****
Volumetric Fidelity (Misc.)
*
*****
**
****
*
4.1 Observations and Possible Solutions
Based on the foregoing survey, we can make several observations on the status quo of 3D representation method-
ology. Below, we list each of these observations in turn, alongside possible approaches to address them.
4.1.1 Scene Management requires Hierarchy
Regardless of the chosen representation, enabling efficient processing and rendering of large environments will
remain a challenge. While it may seem ”elegant” to contain an entire environment within a single neural field or
Gaussian splat, it is unlikely to be practical. Instead, a hierarchical acceleration structure, such as a Bounding
Volume Hierarchy (BVH), is essential for efficient culling, instancing, and managing Levels of Detail (LODs).
Within such a ”super-structure,” the environment would consist of a hierarchy of instances with bounding
volumes and transform information (see Fig. 13). The data within these instances could then be loaded and
rendered dynamically on demand, or even swapped for lower-fidelity models based on distance. For instance, an
object could be displayed as a high-detail Gaussian splat up close but be reduced to a simple mesh or even a
basic impostor/billboard at larger distances.
This concept would also handle dynamic scenes more effectively, as instances can be moved, added, or removed
with relative ease. Furthermore, annotative information such as semantic labels could be stored on a per-instance
level, allowing a user to query and interact with the system through these labels, perhaps even with the aid of
Large Language Models (LLMs).
Due to their widely employed nature within game engines and graphics frameworks,35 we believe some form
of bounding volume hierarchy is the most viable contender for a dynamic and manageable acceleration structure.
4.1.2 Functionality requires Geometric Grounding
Traditional models (particularly meshes) offer robust geometric guarantees and computational efficiency for tasks
like physics and visibility but lack the ability to capture photorealistic appearance from images (see Fig. 14).
Conversely, modern methods like NeRFs and Gaussian Splatting excel at photorealism but are often geometrically
unreliable and computationally prohibitive for tasks other than rendering.47
As such, we expect that any system leveraging these technologies for more than just image synthesis would
benefit greatly from an explicit geometric scaffold. This ”traditional” mesh could be used for all physics, collision,
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 17 -->
Figure 13. Example of a 3-level BVH applied to a scene. An area-division acts as a top-level acceleration structure, while
the data is stored within instances that are trained and managed individually. The LOD of instances can be downscaled
or their underlying model swapped out depending on need or distance.
and visibility calculations, while the modern representation is employed for photorealism, thus retaining the
strengths of both models.
For instance, rendering a NeRF could be accelerated by first ray-tracing against an underlying mesh and
then querying the NeRF only at the point of intersection. Similarly, inserting physics-based objects into a scene
represented by 3DGS would be far more stable if the underlying physics calculations were performed on an
encompassing mesh rather than the Gaussians themselves.
We do not expect these geometric doubles to be modeled manually, but rather to be generated by one of the
various neural mesh reconstruction methods such as those outlined in Section 3.3.
4.1.3 Fusing Sensor Modalities
Modern approaches such as 3DGS and NeRFs primarily focus on data from camera sensors, providing great
photorealistic detail but with potentially flimsy geometry. In contrast, voxel grids and point clouds make far
more use of data recorded by lidar sensors, providing a more solid geometric construct but without the same
level of visual detail.
A hybrid model should leverage data from both sensor types: lidar for robust shape and cameras for rich
visual appearance.
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 18 -->
Figure 14.
Comparison of different representation forms of the same scene.
A direct qualitative comparison is not
possible, as each format uses different parameters, memory and software. But the results highlight the focus of newer
methods on camera data, while more traditional models are more reliant on geometric shape.
4.2 Proposal for a Hybrid Model
Based on these observations, we propose a hybrid, instance-based representation pipeline designed for both
functional utility and visual fidelity.
As a first step, the pipeline would perform instance segmentation on incoming lidar point cloud data to
create the initial instances for the BVH. Each instance would be associated with relevant camera images and any
generated semantic labels. This step could be performed by point-cloud segmenters like PointNet, or by methods
that use complementary camera data for segmentation, such as Better Call SAL,48 LDLS,49 or LIF-Seg.50
Each instance would then undergo a training process for both its realistic 3DGS representation and its
geometric mesh scaffold (see Fig. 15). This process would involve a combined loss function: a geometric loss
would penalize deviations between the scaffold and the source lidar data, ensuring accuracy, while a photometric
loss would optimize the properties of the Gaussians to ensure that renders match the source camera images. This
dual-objective optimization ensures the final model is both geometrically sound and visually accurate.
Alternatively, the scaffold and 3DGS could be trained sequentially. One of the many surface reconstruction
techniques, such as Poisson51 or Marching Cubes,52 could create an initial mesh, which is then refined by a
differentiable mesh renderer like Soft Rasterizer38 or Neural 3D Mesh Renderer.37 This final shape would then
provide positional constraints for the 3DGS representation trained thereafter. Finally, the trained Gaussians
could be used to bake a high-fidelity texture for the mesh, providing a more lightweight asset.
It is important to note that this proposal provides a rough, theoretical basis. We expect there to be a number
of challenges and limitations that must be addressed in a practical implementation.
4.2.1 Related Work
This proposed hybrid model aligns with a significant trend in recent computer graphics research that seeks
to unify the benefits of classical and neural representations. Several state-of-the-art methods could serve as a
foundation for this approach:
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 19 -->
Figure 15. Theoretical overview of the proposed pipeline-architecture. After data fusion, the combined point clouds and
images are segmented and instanced, then a mesh scaffold and shape-constrained 3DGS representation is generated for
each instance.
• NeuS and VolSDF: Methods like NeuS53 and VolSDF54 are designed to learn a neural Signed Distance
Function (SDF) from images using volume rendering. Because an SDF implicitly defines a surface, a high-
quality mesh can easily be extracted. These models learn both a robust geometry and a neural radiance
field simultaneously, providing both a functional mesh and a high-fidelity appearance model from a single
training process.
• SuGaR (Surface-Aligned Gaussian Splatting): This approach is perhaps most closely aligned with
our proposal. SuGaR47 explicitly optimizes a set of 3D Gaussians to conform to a coarse geometric mesh.
It uses the mesh as a prior to regularize the Gaussians, resulting in a clean, well-defined surface while
leveraging the rendering speed and quality of Gaussian Splatting.
This reframes the Gaussians as an
incredibly advanced neural texture for the mesh, where the mesh provides the stable geometric foundation
and the Gaussians provide the fine visual detail.
4.3 Conclusions
Based on our evaluation of existing representation forms and the observations we were able to draw from them
in the sections above, we can draw the following core conclusions:
• Conclusion #1: While modern representations like 3DGS or NeRFs outpace traditional models in visual
quality, they are often inadequate for functional tasks such as visibility calculations, route planning, or
physics simulations.
• Conclusion #2: Although irregular representations like point clouds and meshes require specialized deep
learning architectures, the advent of differentiable renderers makes it possible to leverage modern optimiza-
tion techniques to generate and refine these traditional geometric formats.
• Conclusion #3: In our estimation, a hybrid model is the most promising path forward. Such a system
would use a BVH for scene management, a ”scaffold” mesh for functional operations, and a surface-aligned
representation like 3DGS for visual detail, with a differentiable renderer bridging the gap between these
components.
Looking ahead, the success of the hybrid model depends on continued research in key areas. Methods for
simultaneous training of the mesh scaffold and surface-constrained 3DGS must be refined to optimize geometric
accuracy and visual fidelity together. As components are added and challenges arise, the architecture will require
ongoing adjustment. This work is not a final solution but a framework to guide the next steps toward a functional,
versatile 3D battlespace visualization system.
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 20 -->
REFERENCES
[1] Zi´o lkowska, A., “Open source intelligence (OSINT) as an element of military recon,” Security and Defence
Quarterly 19(2), 65–77 (2018).
[2] Sufi, F., “Social media analytics on Russia–Ukraine cyber war with natural language processing: Perspectives
and challenges,” Information 14(9) (2023).
[3] Schubert, J., Brynielsson, J., Nilsson, M., and Svenmarck, P., “Artificial intelligence for decision support
in command and control systems,” in [23rd International Command and Control Research & Technology
Symposium “Multi-Domain C], 2, 18–33 (2018).
[4] “WinTAK - Team Awareness Kit, Tactical Assault Kit.” https://www.civtak.org/tag/wintak/.
[5] Park, J. J., Florence, P., Straub, J., Newcombe, R., and Lovegrove, S., “DeepSDF: Learning continuous
signed distance functions for shape representation,” in [The IEEE Conference on Computer Vision and
Pattern Recognition (CVPR)], (June 2019).
[6] Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R., “NeRF: Repre-
senting scenes as neural radiance fields for view synthesis,” in [Proceedings of the European Conference on
Computer Vision (ECCV)], (2020).
[7] Tancik, M., Weber, E., Ng, E., Li, R., Yi, B., Kerr, J., Wang, T., Kristoffersen, A., Austin, J., Salahi, K.,
Ahuja, A., McAllister, D., and Kanazawa, A., “Nerfstudio: A modular framework for neural radiance field
development,” in [ACM SIGGRAPH 2023 Conference Proceedings], SIGGRAPH ’23 (2023).
[8] Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis, G., “3D Gaussian splatting for real-time radiance
field rendering,” ACM Transactions on Graphics 42 (July 2023).
[9] Wang, Z., “3D representation methods: A survey,” arXiv (2410.06475) (2024).
[10] Shi, Z., Peng, S., Xu, Y., Geiger, A., Liao, Y., and Shen, Y., “Deep generative models on 3D representations:
A survey,” arXiv (2210.15663) (2023).
[11] Li, X., Zhang, Q., Kang, D., Cheng, W., Gao, Y., Zhang, J., Liang, Z., Liao, J., Cao, Y.-P., and Shan, Y.,
“Advances in 3D generation: A survey,” arXiv (2401.17807) (2024).
[12] Ahmed, E., Saint, A., Shabayek, A. E. R., Cherenkova, K., Das, R., Gusev, G., Aouada, D., and Ottersten,
B., “A survey on deep learning advances on different 3D data representations,” arXiv (1808.01462) (2019).
[13] Biermann, J., “Understanding military information processing — an approach to support intelligence in
defence and security,” in [Harbour Protection Through Data Fusion Technologies], Shahbazian, E., Rogova,
G., and DeWeert, M. J., eds., 127–137, Springer Netherlands, Dordrecht (2009).
[14] Elhashash, M., Albanwan, H., and Qin, R., “A review of mobile mapping systems: From sensors to appli-
cations,” Sensors 22(11) (2022).
[15] Sch¨onberger, J. L. and Frahm, J.-M., “Structure-from-motion revisited,” in [Conference on Computer Vision
and Pattern Recognition (CVPR)], (2016).
[16] Sch¨onberger, J. L., Zheng, E., Pollefeys, M., and Frahm, J.-M., “Pixelwise view selection for unstructured
multi-view stereo,” in [European Conference on Computer Vision (ECCV)], (2016).
[17] Borgmann, B., Schatz, V., Hammer, M., Hebel, M., Arens, M., and Stilla, U., “MODISSA: a multipur-
pose platform for the prototypical realization of vehicle-related applications using optical sensors,” Applied
Optics 60, F50–F65 (06 2021).
[18] Sch¨utz, M., Ohrhallinger, S., and Wimmer, M., “Fast out-of-core octree generation for massive point clouds,”
Computer Graphics Forum 39, 1–13 (Nov. 2020).
[19] Kivi, P. E. J., M¨akitalo, M. J., ˇZ´adn´ık, J., Ikkala, J., Vadakital, V. K. M., and J¨a¨askel¨ainen, P. O., “Real-
time rendering of point clouds with photorealistic effects: A survey,” IEEE Access 10, 13151–13173 (2022).
[20] G¨unther, C., Kanzok, T., Linsen, L., and Rosenthal, P., “A GPGPU-based pipeline for accelerated rendering
of point clouds,” Journal of WSCG 21, 153 (06 2013).
[21] Sch¨utz, M., Kerbl, B., and Wimmer, M., “Rendering point clouds with compute shaders and vertex order
optimization,” Computer Graphics Forum 40(4), 115–126 (2021).
[22] Qi, Charles, R., Su, H., Mo, K., and Guibas, L. J., “PointNet: Deep learning on point sets for 3D classifica-
tion and segmentation,” in [The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)],
77–85 (2017).
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 21 -->
[23] Qi, C. R., Yi, L., Su, H., and Guibas, L. J., “PointNet++: Deep hierarchical feature learning on point sets
in a metric space,” in [Proceedings of the 31st International Conference on Neural Information Processing
Systems], NIPS’17, 5105–5114, Curran Associates Inc., Red Hook, NY, USA (2017).
[24] Pfister, H., Zwicker, M., van Baar, J., and Gross, M., “Surfels: Surface elements as rendering primitives,” in
[Proceedings of the 27th Annual Conference on Computer Graphics and Interactive Techniques], SIGGRAPH
’00, 335–342, ACM Press/Addison-Wesley Publishing Co., USA (2000).
[25] Yifan, W., Serena, F., Wu, S., ¨Oztireli, C., and Sorkine-Hornung, O., “Differentiable surface splatting for
point-based geometry processing,” ACM Trans. Graph. 38 (Nov. 2019).
[26] Gao, Y., Cao, Y.-P., and Shan, Y., “SurfelNeRF: Neural surfel radiance fields for online photorealistic
reconstruction of indoor scenes,” in [Conference on Computer Vision and Pattern Recognition (CVPR)],
108–118 (2023).
[27] “The Stanford 3D scanning repository.” URL: http://graphics.stanford.edu/data/3Dscanrep/.
[28] Crassin, C., Neyret, F., Sainz, M., Green, S., and Eisemann, E., “Interactive indirect illumination using
voxel cone tracing,” Computer Graphics Forum 30(7), 1921–1930 (2011).
[29] Museth, K., “VDB: High-resolution sparse volumes with dynamic topology,” ACM Trans. Graph. 32, 27:1–
27:22 (07 2013).
[30] Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., and Kanazawa, A., “Plenoxels: Radiance fields
without neural networks,” in [2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR)], 5491–5500 (2022).
[31] Sitzmann, V., Thies, J., Heide, F., Nießner, M., Wetzstein, G., and Zollh¨ofer, M., “DeepVoxels: Learning
persistent 3D feature embeddings,” in [Proc. Computer Vision and Pattern Recognition (CVPR), IEEE],
(2019).
[32] Choy, C. B., Xu, D., Gwak, J., Chen, K., and Savarese, S., “3D-R2N2: A unified approach for single
and multi-view 3D object reconstruction,” in [Proceedings of the European Conference on Computer Vision
(ECCV)], (2016).
[33] Wu, J., Zhang, C., Xue, T., Freeman, W. T., and Tenenbaum, J. B., “Learning a probabilistic latent space
of object shapes via 3D generative-adversarial modeling,” in [Advances in Neural Information Processing
Systems], 82–90 (2016).
[34] M¨uller, T., Evans, A., Schied, C., and Keller, A., “Instant neural graphics primitives with a multiresolution
hash encoding,” ACM Trans. Graph. 41, 102:1–102:15 (July 2022).
[35] Kahl, B., “Hardware acceleration of progressive refinement radiosity using Nvidia RTX,” arXiv (2303.14831)
(2023).
[36] Berger, M., Tagliasacchi, A., Seversky, L., Alliez, P., Guennebaud, G., Levine, J., Sharf, A., and Silva, C.,
“A survey of surface reconstruction from point clouds,” Computer Graphics Forum 36, 301–329 (03 2017).
[37] Kato, H., Ushiku, Y., and Harada, T., “Neural 3D mesh renderer,” in [The IEEE Conference on Computer
Vision and Pattern Recognition (CVPR)], 3907–3916 (2018).
[38] Liu, S., Chen, W., Li, T., and Li, H., “Soft rasterizer: A differentiable renderer for image-based 3D reason-
ing,” in [The IEEE International Conference on Computer Vision (ICCV)], 7707–7716 (2019).
[39] Liu, S., Li, T., Chen, W., and Li, H., “A general differentiable mesh renderer for image-based 3D reasoning,”
IEEE Transactions on Pattern Analysis and Machine Intelligence (2020).
[40] Hanocka, R., Hertz, A., Fish, N., Giryes, R., Fleishman, S., and Cohen-Or, D., “MeshCNN: A network with
an edge,” ACM Transactions on Graphics (TOG) 38(4), 90:1–90:12 (2019).
[41] Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., and Geiger, A., “Occupancy networks: Learning 3D
reconstruction in function space,” in [The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR)], 4455–4465 (2019).
[42] Park, J. J., Florence, P., Straub, J., Newcombe, R., and Lovegrove, S., “DeepSDF: Learning continuous
signed distance functions for shape representation,” in [The IEEE Conference on Computer Vision and
Pattern Recognition (CVPR)], 165–174 (2019).
[43] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin,
I., “Attention is all you need,” in [Proceedings of the 31st International Conference on Neural Information
Processing Systems], NIPS’17, 6000–6010, Curran Associates Inc., Red Hook, NY, USA (2017).
Proc. of SPIE Vol. 13679 1367911-1

<!-- page 22 -->
[44] Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoor-
thi, R., Barron, J. T., and Ng, R., “Fourier features let networks learn high frequency functions in low
dimensional domains,” NeurIPS (2020).
[45] Kulhanek, J., Rakotosaona, M.-J., Manhardt, F., Tsalicoglou, C., Niemeyer, M., Sattler, T., Peng, S.,
and Tombari, F., “LODGE: Level-of-detail large-scale Gaussian splatting with efficient rendering,” arXiv
(2505.23158) (2025).
[46] Windisch, F., Radl, L., K¨ohler, T., Steiner, M., Schmalstieg, D., and Steinberger, M., “A LoD of Gaus-
sians: Unified training and rendering for ultra-large scale reconstruction with external memory,” arXiv
(2507.01110) (2025).
[47] Gu´edon, A. and Lepetit, V., “SuGaR: Surface-aligned Gaussian splatting for efficient 3D mesh reconstruction
and high-quality mesh rendering,” in [Conference on Computer Vision and Pattern Recognition (CVPR)],
5354–5363 (2024).
[48] Oˇsep, A., Meinhardt, T., Ferroni, F., Peri, N., Ramanan, D., and Leal-Taix´e, L., “Better call SAL: Towards
learning to segment anything in lidar,” in [Computer Vision – ECCV 2024: 18th European Conference,
Milan, Italy, September 29–October 4, 2024, Proceedings, Part XXXIX], 71–90, Springer-Verlag, Berlin,
Heidelberg (2024).
[49] Wang, B., Chao, W.-L., Wang, Y., Hariharan, B., Weinberger, K., and Campbell, M., “LDLS: 3-D object
segmentation through label diffusion from 2-D images,” arXiv (1910.13955) (2019).
[50] Zhao, L., Zhou, H., Zhu, X., Song, X., Li, H., and Tao, W., “LIF-Seg: LiDAR and camera image fusion for
3D LiDAR semantic segmentation,” arXiv (2108.07511) (2021).
[51] Kazhdan, M., Bolitho, M., and Hoppe, H., “Poisson surface reconstruction,” in [Proceedings of the Fourth
Eurographics Symposium on Geometry Processing], SGP ’06, 61–70, Eurographics Association, Goslar, DEU
(2006).
[52] Lorensen, W. E. and Cline, H. E., “Marching cubes: A high resolution 3D surface construction algorithm,”
SIGGRAPH Comput. Graph. 21, 163–169 (Aug. 1987).
[53] Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., and Wang, W., “NeuS: Learning neural implicit
surfaces by volume rendering for multi-view reconstruction,” NeurIPS (2021).
[54] Yariv, L., Gu, J., Kasten, Y., and Lipman, Y., “Volume rendering of neural implicit surfaces,” in [Thirty-
Fifth Conference on Neural Information Processing Systems], (2021).
Proc. of SPIE Vol. 13679 1367911-1
