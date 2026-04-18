<!-- page 1 -->
GS-VERSE: MESH-BASED GAUSSIAN SPLATTING FOR
PHYSICS-AWARE INTERACTION IN VIRTUAL REALITY
Anastasiya Pechko, Piotr Borycki, Joanna Waczy´nska, Daniel Barczyk, Agata Szyma´nska
Jagiellonian University
anastasiya.pechko@doctoral.uj.edu.pl
Sławomir Tadeja
University of Cambridge
Przemysław Spurek
Jagiellonian University
IDEAS Research Institute
przemyslaw.spurek@uj.edu.com
Figure 1: The three scenes used in our study: (top) dark room, (middle) toy room, and (bottom)
garden as seen from VR. The three columns on the right present a handful from a large range
of possible physics-aware 3D object manipulations (e.g., moving, swinging, stretching, pulling,
twisting, shaking, crushing, tipping).
ABSTRACT
As the demand for immersive 3D content grows, the need for intuitive and effi-
cient interaction methods becomes paramount. Current techniques for physically
manipulating 3D content within Virtual Reality (VR) often face significant limi-
tations, including reliance on engineering-intensive processes and simplified ge-
ometric representations, such as tetrahedral cages, which can compromise visual
fidelity and physical accuracy. In this paper, we introduce GS-Verse (Gaussian
Splatting for Virtual Environment Rendering and Scene Editing), a novel method
designed to overcome these challenges by directly integrating an object’s mesh
1
arXiv:2510.11878v2  [cs.GR]  4 Nov 2025

<!-- page 2 -->
with a Gaussian Splatting (GS) representation. Our approach enables more pre-
cise surface approximation, leading to highly realistic deformations and interac-
tions. By leveraging existing 3D mesh assets, GS-Verse facilitates seamless con-
tent reuse and simplifies the development workflow. Moreover, our system is de-
signed to be physics-engine-agnostic, granting developers robust deployment flex-
ibility. This versatile architecture delivers a highly realistic, adaptable, and intu-
itive approach to interactive 3D manipulation. We rigorously validate our method
against the current state-of-the-art technique that couples VR with GS in a com-
parative user study involving 18 participants. Specifically, we demonstrate that
our approach is statistically significantly better for physics-aware stretching ma-
nipulation and is also more consistent in other physics-based manipulations like
twisting and shaking. Further evaluation across various interactions and scenes
confirms that our method consistently delivers high and reliable performance,
showing its potential as a plausible alternative to existing methods. Code is avail-
able at https://github.com/Anastasiya999/GS-Verse
1
INTRODUCTION
The creation of immersive 3D environments has traditionally been the domain of skilled experts and
graphic artists Huang et al. (2025). Conventional workflows, reliant on meticulous mesh modeling
and texture mapping, are labor-intensive, time-consuming, and require a high level of technical
expertise Yuan et al. (2025); Barron et al. (2022). This complexity poses a significant barrier to the
widespread adoption and creation of rich, interactive 3D content, particularly for emerging platforms
like Virtual Reality (VR) Tadeja (2020); Huang et al. (2025).
An alternative paradigm has recently emerged with 3D Gaussian Splatting (3DGS) Kerbl et al.
(2023), a technique that enables the creation of photorealistic 3D assets and entire scenes, which
can be used to populate immersive environments Qiu et al. (2025). This approach enables the recon-
struction of detailed 3D objects and scenes from video recordings, typically captured with a standard
smartphone. This breakthrough allows experts and non-experts alike to develop VR environments or
capture and import individual real-world objects into existing digital scenes, significantly lowering
the barrier to entry for content creators Qiu et al. (2025).
GS has already been adopted for VR applications, with recent work demonstrating its integration
into popular development platforms such as Unity game engine Jiang et al. (2024); Franke et al.
(2025); Tu et al. (2025). VR-Splatting introduces a foveated rendering framework that combines
splats with neural points to reduce computational cost by adapting rendering quality to the user’s
gaze while preserving fidelity in the foveal region Franke et al. (2025). Similarly, VRSplat presents
a fast and robust pipeline tailored for VR, emphasizing low-latency and stable performance to en-
able interactive experiences Tu et al. (2025). VR-GS incorporates physics-aware dynamics into GS,
allowing interactive manipulation of splatted objects in immersive environments Jiang et al. (2024).
However, this approach has a notable limitation–it relies on simplified geometric proxies, such as
low-resolution tetrahedral cages, to approximate the object’s physics. This simplification, although
computationally efficient, compromises the accuracy of physical simulations and can result in visu-
ally unconvincing deformations that do not accurately represent the object’s true surface geometry.
In this paper, we introduce GS-Verse which addresses the key challenges of crafting, editing, and
interacting with digital assets by leveraging the strengths of GS within an immersive environment,
see Fig. 1. While previous work has made strides in integrating physical simulations with GS, they
often rely on simplified geometric representations (e.g., tetrahedral cages), which can compromise
the visual fidelity of the final rendering and the plausibility of physical interactions.
Our model represents a novel approach that directly employs the object’s surface mesh as the pri-
mary representation for physical simulation, see Fig. 2. In practice, we can estimate high-quality
meshes directly from GS representation Guédon & Lepetit (2024); Waczy´nska et al. (2024); Huang
et al. (2024a); Li et al. (2025), which allows us to model large 3D scenes. For content generation,
we can utilize similar tools to convert 2D images into GS representations and extract meshes. Ad-
ditionally, objects can also be obtained using specialized generative models, which create meshes
with GS representations Xiang et al. (2025). The use of high-quality surface meshes offers several
key advantages. First, it ensures a more accurate approximation of the object’s geometry, resulting
2

<!-- page 3 -->
Figure 2: Overview of our proposed method, GS-Verse. The approach enables real-time interaction
in VR by generating mesh-based Gaussian Splatting assets. The processing pipeline begins with
multiview image scene reconstruction, followed by mesh extraction using SuGaR or Trellis. A
subsequent segmentation step optimizes the scene by dividing it into dynamic and static components.
Thanks to mesh-based parameterization via GaMeS, the resulting representations can be seamlessly
integrated into physics-aware engines such as Unity, enabling efficient and physically consistent VR
interactions.
in more realistic deformations and interactions, see Fig. 3. Second, by utilizing a standard mesh, our
system is compatible with existing 3D models and assets, allowing for the seamless reuse of a vast
array of interactive VR applications, see Fig. 4.
Furthermore, our system is designed to be engine-agnostic, providing a flexible framework that can
be integrated with any off-the-shelf physics engine. This versatility allows developers to choose
the simulation tool best suited for their specific needs, without being locked into a proprietary or
specialized solution. The ability to use a wide range of physics engines not only simplifies the
development workflow but also broadens the potential applications of our system. By combining
these unique advantages, our GS-Verse method offers a robust, flexible, and highly realistic solution
for immersive 3D content manipulation.
The following constitutes a list of our key contributions:
1. We propose GS-Verse pipeline that uses surface meshes with GS representation in VR,
ensuring more accurate object geometry approximation and backward compatibility with
the majority of existing game engine systems.
2. GS-Verse is engine-agnostic and works with existing physics simulation engines to provide
optimized, physics-aware interaction and manipulation, contributing to more natural and
immersive VR experiences.
3. GS-Verse offers highly realistic, adaptable 3D manipulation, providing a statistically sig-
nificant advancement in physics-aware stretching and consistent perceived performance
across diverse manipulation scenarios.
3

<!-- page 4 -->
Method
stretch
twist
shake
static
VR-GS
GS-Verse
Figure 3: Visual example of a participant performing the requested manipulations during the first,
closed-ended task: stretch, twist, shake in VR-GS (first row) and GS-Verse method (second row).
2
RELATED WORKS
Here, we review the key areas of research that provide the foundation for our work. We structure
our discussion into two main parts. First, we provide an overview of radiance fields, focusing on
the evolution from Neural Radiance Fields (NeRF) to modern point-based representations like 3D
GS. Second, we examine the specific challenges and existing solutions for using Radiance Fields in
Virtual Reality.
Radiance Fields
The challenge of synthesizing novel views from a collection of images has long
been a central focus in computer graphics. A significant breakthrough in this area occurred with the
introduction of NeRFs Mildenhall et al. (2021), which combine the principles of classical volume
rendering with the expressive power of multilayer perceptrons (MLPs) to produce images of remark-
able quality. Despite their visual fidelity, NeRFs are inherently limited by substantial computational
costs, as their reliance on per-pixel ray marching results in prolonged training and rendering times.
In response, a significant body of research has emerged to address these performance bottlenecks
Fridovich-Keil et al. (2022); Müller et al. (2022); Steiner et al. (2024); Zimny et al. (2025), extend
the framework to large, unbounded scenes Barron et al. (2022), and adapt it for manual editing and
physical engines Chen et al. (2023); Wang et al. (2023); Zieli´nski et al. (2025).
To overcome the NeRFs’ performance limitations, 3D Gaussian Splatting (3DGS) Kerbl et al. (2023)
introduced a paradigm shift by representing scenes with an explicit set of 3D Gaussians. This
approach leverages highly efficient software rasterization, resulting in a significant acceleration in
rendering speed. The success of 3DGS has spurred a new wave of research, with efforts focused on
enhancing the technique’s robustness and efficiency. This includes developing methods for artifact-
free rendering Huang et al. (2024b); Radl et al. (2024); Yu et al. (2024), strategies for reducing
the number of primitives required to represent a scene without compromising quality Fan et al.
(2024); Fang & Wang (2024), and scaling the representation to handle massive datasets for large-
scale exploration Lin et al. (2024a). GS representations are well-suited for manual editing and
physical engines Waczy´nska et al. (2024); Guédon & Lepetit (2024); Borycki et al. (2024); Tobiasz
et al. (2025)
Radiance Fields for VR
Immersive interfaces such as VR serve as a natural and compelling appli-
cation for radiance fields, offering users an unparalleled sense of immersion when exploring captured
scenes. Early efforts to integrate NeRFs into VR, such as FoV-NeRF Deng et al. (2022), utilized
gaze-contingent neural representations and foveated rendering to achieve real-time performance.
Another approach, VR-NeRF Xu et al. (2023), focuses on architectural solutions, distributing the
computational load across multiple GPUs and utilizing occupancy grids to accelerate rendering.
However, these methods still struggle to meet the demanding performance targets of VR without
significant compromises in visual quality.
4

<!-- page 5 -->
Example modification
VR-GS
GS-Verse
Figure 4:
Artifacts during maximum object stretching from the user perspective. In VR-GS (top
row), overstretching caused very large splats that overlapped the object. In addition, users were able
to detach individual splats, making the interaction unsuccessful. In contrast, our GS-Verse (bottom
row) remained robust even under extreme stretching, showing minimal artifacts.
The advent of 3DGS has opened new avenues for VR applications. For instance, VR-GS Jiang
et al. (2024) demonstrated how an interactive layer can be built upon the 3DGS framework to enable
physics-based manipulation of objects within the scene. While innovative, its reliance on simplified
proxy geometries for simulation is a limitation that our proposed method directly addresses, posi-
tioning our work as a potential high-quality replacement. Other concurrent research has explored
foveated rendering for GS on mobile hardware Lin et al. (2024b), proposing a level-of-detail strategy
and a training regimen designed to reduce the overlap between Gaussians. More recently, VRSplat
Tu et al. (2025) introduced a fast and robust pipeline specifically designed for VR, featuring a novel
anti-aliasing method to mitigate visual artifacts like popping and flickering. VR-Splatting Franke
et al. (2025) proposed a hybrid foveated system that elegantly combines the strengths of neural point
rendering for the sharp foveal region with the smoothness of 3DGS for the periphery, successfully
meeting VR’s performance demands. While these works significantly advance rendering efficiency,
our research diverges by focusing on the core challenge of high-fidelity physical interaction in VR
Gaussian Splatting, which remains underexplored and is rarely supported by publicly available im-
plementations.
3
MESH-BASED GS FOR PHYSICS-AWARE INTERACTION IN VR (GS-VERSE)
Gaussian Splatting
GS represents a 3D scene using a collection of spatially distributed 3D Gaus-
sian primitives. Each Gaussian is defined by its mean position, covariance matrix, opacity value, and
color encoded through spherical harmonics (SH) Fridovich-Keil et al. (2022); Müller et al. (2022).
The GS approach constructs a radiance field by iteratively optimizing the positions, covariances,
opacities, and SH-based color coefficients of the Gaussians. One of the key advantages of GS lies
in its rendering pipeline, which efficiently projects 3D Gaussian components onto 2D image planes,
resulting in a highly efficient rendering pipeline.
The 3D scene is modeled as a dense set of Gaussians:
G = {(N(mi, Σi), σi, ci)}n
i=1 ,
where mi is the mean (position), Σi is the covariance matrix, σi represents the opacity, and ci
denotes the SH color of the i-th Gaussian.
Training involves rendering from the current set of Gaussians and comparing the synthesized views
to ground-truth images. Because projecting from 3D to 2D may introduce geometric errors, the
optimization is capable of adding, removing, or relocating Gaussians to correct such inaccuracies.
To adaptively refine the scene representation, a densification strategy is employed during training.
New Gaussians are created by splitting existing ones based on heuristics such as gradient magnitude,
5

<!-- page 6 -->
Method
Simulation Mesh
Original Render
VR View
VR-GS
GS-Verse
Figure 5: Comparison of VR-GS and GS-Verse across three visualization stages: the simulation
mesh (left), the original render (middle), and the render inside VR (right). Unlike VR-GS, which
relies on a cage mesh, GS-Verse employs a more geometry-accurate simulation mesh.
visibility, and accumulated opacity. This allows the model to allocate more representational capacity
in regions with high complexity, while avoiding unnecessary overhead in simpler areas.
The optimization process minimizes a photometric reconstruction loss, typically defined as the mean
squared error (MSE) between rendered images and the ground-truth views. This is often combined
with additional regularization terms that encourage spatial compactness, control Gaussian growth,
and stabilize the covariance matrices during training.
Mesh Reconstruction for GS
An essential step in extending GS beyond photorealistic rendering
is the recovery of a consistent and geometrically accurate mesh representation. Recent methods
demonstrate that imposing surface constraints and geometric regularization significantly improve
the fidelity of reconstructed meshes. For instance, SuGaR Guédon & Lepetit (2024) introduces
surface-aligned splats that facilitate efficient and accurate mesh extraction, while 2DGS Huang et al.
(2024a) leverages 2D Gaussians embedded in the 3D space to capture local geometric structures with
higher precision. Complementarily, GeoSVR Li et al. (2025) refines sparse voxel representations by
enforcing geometry-aware regularization, which improves surface continuity and global consistency.
In our work, we employ one of these approaches to reconstruct a high-quality triangular mesh from
the GS representation, which we denote as
M = (V, E, F),
where V = {vi ∈R3}Nv
i=1 is the set of vertices, E ⊆V ×V is the set of edges, and F ⊆V ×V ×V
represents the triangular faces. This mesh serves as a geometry-aware proxy that can be directly
integrated into our VR framework.
Segmentation
To enable object-level manipulation within VR environments, it is necessary to
perform segmentation of the GS representation. In our system, we adopt a strategy similar to that
employed in VR-GS Jiang et al. (2024), where segmentation is achieved by associating subsets of
Gaussians with individual object instances. Each segmented region corresponds to a coherent group
of Gaussians that can be independently manipulated, deformed, or subjected to physics-based in-
teractions. This approach not only supports efficient rendering and editing but also ensures that
interactions are localized and consistent with object boundaries, thereby enhancing realism in im-
mersive VR scenes. Alternatively, the Gaussians can be segmented as part of post-processing in
cases where objects are visibly separated.
GS-based generative models
Beyond reconstruction, GS also provides a foundation for genera-
tive 3D object synthesis, which is particularly useful for populating interactive VR environments.
6

<!-- page 7 -->
Example modification
Lamp
Fox
Figure 6: Visual example of a participant performing the manipulations requested during the second,
goal-directed action task: (1) Turn on the light by pointing at the lamp and swinging it to make it
oscillate, and (2) Pull the fox by the ears.
The TRELLIS framework Xiang et al. (2025) introduces a structured latent representation that can
be decoded into multiple 3D formats, including 3D Gaussians and explicit meshes. TRELLIS gen-
erates small objects with both appearance and geometric detail. In our VR system, these generated
mesh-augmented GS objects are employed as interactive props or scene elements, allowing users to
grasp, manipulate, or collide with them while maintaining consistency with the underlying Gaus-
sian representation. Formally, the generative model samples a latent z ∼p(z) and decodes it into
a Gaussian Splatting plus mesh pair (G, M), where M = (V, E, F) is the triangular mesh and G
is the associated set of Gaussians. This dual output supports both efficient rendering via GS and
geometry-aware interactions in VR.
GS-Verse: Mesh-based Gaussian Splatting representation
Our model (see Fig. 2) builds upon
external mesh reconstruction tools to obtain an explicit triangular mesh representation, which
subsequently serves as the basis for training Gaussian components. Following the approach of
GaMeS Waczy´nska et al. (2024), we introduce a mesh-guided Gaussian parameterization, where
splats are directly anchored to the mesh faces.
Given a single triangular face with vertices
V = {v1, v2, v3} ⊂R3, the mean of a Gaussian component is defined as a convex combination of
the vertices:
mV (α1, α2, α3) = α1v1 + α2v2 + α3v3,
where α1 + α2 + α3 = 1 and αi are trainable parameters. This guarantees that the Gaussian centers
remain consistently located within the face interior. To model the covariance, we construct a rotation
matrix RV aligned with the triangle (using its normal and edge directions) and a diagonal scaling
matrix SV proportional to the face dimensions. The covariance is then expressed as:
ΣV = RV SV ST
V RT
V ,
ensuring that each Gaussian adapts to the local face geometry. For a given face V , we place k ∈N
Gaussians, parameterized as:
GV =

N
 mV (αi
1, αi
2, αi
3), ρiΣV
	k
i=1 ,
where ρi ∈R+ controls the trainable scale. This parameterization tightly couples Gaussians with
the underlying mesh, ensuring that geometric transformations of the mesh are directly propagated to
the associated Gaussians. As a result, our representation enables consistent rendering and physical
interaction, while preserving the structural properties of the original mesh, see Fig. 5.
Physical simulations
A key advantage of our mesh-based GS representation is that it en-
ables direct integration with existing physics simulation engines. Since our framework explicitly
parametrizes Gaussians on top of a triangular mesh M = (V, E, F), we can leverage the vast
ecosystem of physics solvers developed for mesh-based geometry. In practice, this enables us to
employ standard techniques, such as mass-spring systems Liu et al. (2013), finite element methods
7

<!-- page 8 -->
Example modification
Pillow
Ballon
Figure 7: Visual example of a participant performing the requested manipulations during the second,
goal-directed action task: (1) Shake the pillow and throw it, and (2) Inflate the balloon so that it rises.
Example modification
Balls
Can
Figure 8: Visual example of a participant performing the instructed manipulations requested during
the second, goal-directed action task: (1) Tip the marbles out of the bowl, and (2) Crush the can
and throw it into the trash can. The marbles were simulated using Unity rigid bodies with a physics
material, while the can was simulated with a custom mesh deformation driven by applied forces.
Marinkovic & Zehn (2019), or position-based dynamics Müller et al. (2007), to simulate deforma-
tions, collisions, and rigid-body interactions in a VR environment. Unlike approaches that approxi-
mate physical behavior through simplified proxies (e.g., low-resolution cages), our method ensures
that physical forces act directly on the mesh, thereby indirectly propagating to the Gaussian splats
anchored on its faces. This tight coupling provides physically consistent object behavior while main-
taining the rendering efficiency of GS. As a result, our system achieves immersive and interactive
dynamics in VR without requiring the development of specialized physics models.
4
EVALUATION
To evaluate our interface, we designed a user study with non-expert participants, as GS-Verse method
should be used as a generalized tool allowing for generating various 3D assets that can be used to
populate VR environments. When preparing our experimental design and tasks, we considered the
prior evaluation approach that relied on user studies Jiang et al. (2024); Franke et al. (2025); Tu
et al. (2025). During each task, we collected a range of subjective data to assess the perceived qual-
ity of rendered 3D assets. This included usability, measured with the System Usability Scale (SUS)
Brooke (1996), cognitive task load, measured with the NASA Task Load Index (TLX) Hart & Stave-
8

<!-- page 9 -->
land (1988), and flow, measured with the Short Flow Scale (SFS) Engeser & Rheinberg (2008). We
also administered the Simulation Sickness Questionnaire (SSQ) Kennedy et al. (1993) to exclude
participants experiencing “severe’ symptoms before using VR. These standardized questionnaires
were presented to participants in their native language, using available translations from the litera-
ture. Finally, we used a Likert-like scale to ask participants how natural they rate the reconstructions
generated by our method. All the statistical tests relied on a conservative α = 0.05 level.
4.1
EXPERIMENTAL SETUP
We deployed the GS systems on a desktop PC equipped with Intel(R) Core(TM) i7-14700K
(3.40 GHz), 32,0 GB RAM, NVIDIA GeForce RTX 4070 SUPER, running under Win-
dows 11 Home OS and featuring CUDA compilation tools, release 12.4, V12.4.99 Build
cuda_12.4.r12.4/compiler.33961263_0. As the VR platform, we selected the Meta Quest Pro head-
set. For interacting with the VR environment and the GS models, we relied on Quest handheld
controllers. The exact controls are shown in Tab. 1.
Button
Function Description
Example
Primary Button
Used for stretch deformation of virtual
objects. When pressed and held, par-
ticipants could stretch and elongate de-
formable objects to interact with their
shape.
‘Pull the fox by the ears’ (Task 2, dark
room, see Fig. 6).
Grab Button
Used for object grasping and manipu-
lation. Participants could grab, lift, and
reposition objects within the virtual en-
vironment.
‘Tip the marbles out of the bowl’ (Task
2, garden, see Fig. 8).
Select Button
Used for scaling, pressing, and adding
movement. This button allowed partic-
ipants to scale object size, apply pres-
sure, or introduce movement, depend-
ing on the task context.
‘Inflate the balloon so that it rises’(Task
2, toy room, see Fig. 7).
Table 1: The mapping of the handheld controller button to manipulation/interaction methods.
4.2
3D SCENES
For our evaluation study, we prepared three immersive scenes. Each of these scenes consisted of a
large background model and three GS-based models generated with GS-Verse that could be manip-
ulated and interacted with by the study participants. Furthermore, to ensure plausible manipulation
results, the interaction with all these models was physics-aware, meaning that stretching, twisting,
or shaking closely resembled real-life manipulations.
Scene: dark room
We created this scene using the Armoury 3D model from the Real World
Textured Things (RWTT) dataset Maggiordomo et al. (2020). The first interactable object, a fox, was
reconstructed from the Instant-NGP dataset Müller et al. (2022). We implemented soft-body logic
in Unity to simulate realistic deformation and dynamic movement in response to controller input.
For the second and third objects, a chair and a lamp, we used TRELLIS to obtain the 3D meshes,
and then Blender to generate the NeRF Synthetic dataset. The chair was configured as a rigid body
with an assigned physics material and equipped with the XR Grab Interactable component from the
XR Interaction Toolkit, allowing it to be physically manipulated and moved by the user in virtual
space. For the lamp, we used the XR Simple Interactable component to listen for controller events
and apply corresponding transformations for oscillation and turning the light on. The scene is shown
in Fig. 6.
Scene: toy room
We prepared this scene using the Playroom scene from the Deep Blending
dataset Hedman et al. (2018). The images were used to create the coloured mesh using the mesh
processing tool MeshLab Cignoni et al. (2008). The three interactable objects included: a balloon, a
car, and a pillow. These objects were also reconstructed using TRELLIS to generate the 3D meshes,
followed by Blender to prepare the dataset. For the balloon, we used a Unity Compute shader to
9

<!-- page 10 -->
Manipulation Median
VR-GS
Median
GS-Verse
Mean ± SD
VR-GS
Mean ± SD
GS-Verse
W (Wilcoxon) p-value Effect size (r)
stretching
-0.5
2.0
-0.33 ± 2.00 1.33 ± 1.28
4.0
0.009
0.62
twisting
0.5
1.0
-0.28 ± 2.19 0.22 ± 1.90
29.0
0.425
0.19
shaking
2.0
2.0
1.50 ± 1.47
1.33 ± 1.61
15.0
0.673
0.10
Table 2: The descriptive statistics and Wilcoxon Signed-Ranks Test results comparing the perceived
“naturalness” VR-GS and GS-Verse across three manipulations conducted during the first, close-
ended task. The results suggest statistically significant difference for the stretching manipulation
between the two methods.
Questionnaire Median
VR-GS
Median
GS-Verse
Mean ± SD
VR-GS
Mean ± SD
GS-Verse
W (Wilcoxon) p-value Effect size (r)
SUS ↑
77.50
77.50
73.6 ± 16.7 78.9 ± 11.6
68.0
1.0000
-0.000
TLX ↓
7.085
6.665
9.81 ± 10.1 8.47 ± 6.12
55.0
0.5012
0.159
Table 3: Wilcoxon Signed-Ranks Test results comparing VR-GS and GS-Verse across usability and
induced cognitive load during the first, close-ended task. The higher SUS [0-100] and lower TLX
[0-100] are better. While the differences are statistically insignificant, our GS-Verse method led to
higher SUS and lower TLX.
simulate inflation and configured it as a rigid body with upward force so that it naturally rises. For
the car, we configured it as a rigid body and added the XR Grab Interactable component to listen
for controller events and apply translation/grab transformations in response. The pillow was imple-
mented as a soft body with logic to simulate stretching and deformation. In addition, it was equipped
with the XR Grab Interactable component to allow grabbing and throwing. The scene is shown in
Fig. 7.
Scene: garden
This scene was reconstructed using the Mip-NeRF 360 dataset Barron et al. (2022).
For the first of three interactable objects, marbles inside a bowl, we prepared the dataset using
TRELLIS and Blender, and each marble was configured as a rigid body with a physics material
that caused them to bounce and jump naturally. The second object, a flower, was segmented from
the original garden scene and given the XR Grab Interactable component to enable grabbing and
throwing interactions. The last object, a can, was reconstructed using TRELLIS and Blender and
designed as a soft body with logic to simulate deformation under press forces applied via raycast.
The scene is shown in Fig. 8.
4.3
USER STUDY
4.3.1
PARTICIPANTS
We recruited 18 participants using opportunistic sampling. The group consisted of 2 females and 16
males, ranging in age from 19 to 31 years (m = 25.2(2), SD = 3.21).
4.3.2
EXPERIMENTAL DESIGN AND TASKS
Inspired by earlier work Jiang et al. (2024); Tu et al. (2025), we designed two separate tasks to evalu-
ate our approach with users. The first closed-ended task had a specific objective intended to compare
our method with previous work concerning the use of GS in a VR environment Jiang et al. (2024).
At the time this study was initiated, we selected VR-GS as the baseline for comparison because, to
the best of our knowledge, it was the only publicly available implementation that combined Gaus-
sian Splatting with physics-based object interaction in VR and provided example scenes, allowing
for a direct and reproducible evaluation. Other concurrent works primarily focused on rendering ef-
ficiency or foveated visualization and did not offer publicly accessible implementations suitable for
interactive assessment. Whereas the second, goal-directed action task served as a means of evaluat-
ing perceived quality of 3D scenes populated with 3D assets generated with our GS-Verse method.
In both tasks, the manipulation with the 3D assets took into account physics-based constraints and
10

<!-- page 11 -->
Questionnaire
t
df p-value Mean ± SD
VR-GS
Mean ± SD
GS-Verse
Mean
difference
95% CI
lower
95% CI
upper
Cohen’s d
FLOW ↑
-1.5 17
0.152
4.80 ± 1.07 5.03 ± 0.86
-0.233
-0.538
0.071
-0.354
Table 4: Paired t-test comparing results of the flow experienced by the participants during the first,
close-ended task execution using VR-GS and GS-Verse. Sample size (N), t-statistic, degrees of
freedom (df), p-value, mean difference, 95% confidence interval, and effect size (Cohen’s d) are
reported. The higher FLOW [1-7] values are better.
affordances. Additionally, we administered the SSQ Kennedy et al. (1993) questionnaire before each
task to exclude participants who reported having “severe” symptoms before commencing any task.
Figure 9: Participants’ rating of three physics-
aware manipulations of the GS 3D objects: (i)
stretching, (ii) twisting, and (iii) shaking in task 1.
The comparison revealed a statistically significant
preference for our system in the stretching ma-
nipulation, as well as higher consistency across
the other two physics-aware manipulations (**p
< 0.01, ns = not significant).
All participants initially experienced the test
scene to familiarize themselves with available
manipulation techniques. Afterward, they were
exposed to the first closed-ended and second
goal-directed action tasks. Such an approach
allowed them to first learn about the available
manipulation techniques (e.g., stretching, twist-
ing, and shaking) and their limitations before
freely experiencing the VR environment popu-
lated with 3D assets generated with GS-Verse
method.
The first closed-ended task allowed us to di-
rectly compare GS-Verse with VR-GS Jiang
et al. (2024). During this task, we asked the par-
ticipants to (i) stretch, (ii) twist, and (iii) shake
a 3D object reconstructed in the same 3D scene
by either GS-Verse or VR-GS presented in bal-
anced, randomized order (see Fig. 3). Partici-
pants were able to deform the 3D asset as much
or as little as they wanted in under 2 minutes.
After experiencing each of the two reconstruc-
tion methods, we administered the SUS Brooke
(1996), TLX Hart & Staveland (1988), and SFS
Engeser & Rheinberg (2008) questionnaires. In
addition, we asked the participants to assess the
“naturalness” of the interaction with the 3D objects on a 7-point Likert scale ranging from non-
natural to very natural.
The second task had a goal-directed action form and was also inspired by previous work coupling
VR and GS Tu et al. (2025). In this task, the participants were immersed in three different scenes,
namely, (A) dark room, (B) toy room, and (C) garden. In each scene, we placed three interactive 3D
assets that can be deformed using techniques available in an earlier task (see Fig. 1). Furthermore,
we asked the participant to perform three swift interactions, adjusted to be similar actions that could
be taken in the particular real-life context and scene. In the case of the dark room, these were: (1)
Move the chair into the corner (2) Turn on the light by pointing at the lamp, and swing it (or make
it sway/oscillate), and (3) Pull the fox by the ears (see Fig. 6). When immersed in the toy room,
the participants were tasked to: (1) Inflate the balloon so that it rises, (2) Start the toy car so that it
crashes into the wall, and (3) Shake the pillow and throw it (see Fig. 7). Whereas, when exposed to
the garden scene, we ask them to: (1) Tip the marbles out of the bowl, (2) Knock the flowerpot off the
table onto the grass, and (3) Crush the can and throw it into the trash can (see Fig. 8). Similarly to
the first task, after experiencing each of the three scenes, we administered the SUS Brooke (1996),
TLX Hart & Staveland (1988), and SFS Engeser & Rheinberg (2008) questionnaires. In addition, we
asked the participants to assess the “naturalness” of the interaction with the 3D objects on a 7-point
Likert scale ranging from non-natural to very natural. This approach allowed us to gain insight into
how participants perceived the quality of the renders and physics-aware interaction provided with
GS-Verse method under a diverse range of contexts.
11

<!-- page 12 -->
SUS
TLX
FLOW
Figure 10: Comparison of VR-GS and GS-Verse methods for the close-ended task across three ques-
tionnaires: SUS [0-100]↑, TLX [0-100]↓, and FLOW [1-7]↑. Non-significance of paired compar-
isons between methods is indicated above each plot and calculated using the Wilcoxon Signed-Rank
Test for SUS and TLX, and the paired t-test for FLOW (ns = not significant).
SUS
TLX
FLOW
Figure 11: Comparison of participants’ provided scores for goal-directed action task across three
evaluation methods: SUS [0-100]↑, TLX [0-100]↓, and FLOW [1-7]↑. For SUS and TLX, a Fried-
man test was used to assess differences between conditions (scenes), while FLOW was analyzed us-
ing a repeated-measures ANOVA. Statistically significant difference was observed within the FLOW
metric between the dark room and the toy room (**p < 0.01, ns = not significant).
5
EVALUATION RESULTS
To compare our GS-Verse method with prior work and assess the overall quality of our approach for
populating interactive, immersive environments across various contexts, we employed a qualitative
evaluation strategy. We decided on this evaluation because the primary goal of our work is to provide
plausible, realistic, physics-aware manipulations of 3D assets within a VR environment populated
with GS-based objects. Therefore, we employed a Likert-like scale to capture participants’ own
assessment of the “naturalness” of the three distinguishable, physics-based manipulations, as well
as three standardized, well-established surveys.
The latter included the SUS Brooke (1996) questionnaire for assessing the perceived usability of
interacting with and manipulating 3D assets generated with our GS-Verse method. We also ad-
ministered the NASA-TLX questionnaire Hart & Staveland (1988), a widely adopted approach for
quantifying participants’ subjective cognitive load during the execution of a specific task in VR
Tadeja et al. (2021); Tadeja et al.. Finally, we used the SFS questionnaire Engeser & Rheinberg
12

<!-- page 13 -->
(2008) to track perceived flow levels during VR interaction, which can serve as an indicator of user
engagement and skill during task performance Engeser & Rheinberg (2008); Laakasuo et al. (2022).
5.1
CLOSE-ENDED TASK
Perceived Naturalness
The Shapiro-Wilk test indicated that the data were not normally dis-
tributed. Therefore, we used the nonparametric Wilcoxon Signed-Rank Test. We present in Tab.
2 results comparing medians and test statistics between VR-GS and GS-Verse (our) across three
physics-aware manipulations, i.e., stretching, (ii) twisting, and (iii) shaking. For the latter two, the
Wilcoxon Signed-Rank Test revealed no statistically significant difference (twisting W = 29.00,
p = .425, r = .19; shaking: W = 15.00, p = .673, r = .10) (see Fig. 9).
However, in
the case of stretching manipulation, the median performance was higher for the GS-Verse method
(Mdn = 2.00) than for VR-GS (Mdn = ˘0.50). A Wilcoxon Signed-Rank Test indicated that this
difference was statistically significant, W = 4.00, p = .009, with a large effect size (r = .62).
This suggests that participants perceived the stretching manipulation as significantly more natural
when using our GS-Verse, with VR-GS showing visible artifacts such as large overlapping splats
and detached parts (see Fig. 4). Moreover, as indicated in Fig. 9, they were also more consistent in
their assessment of the remaining two manipulations.
Perceived Usability
For the SUS questionnaire, median performance was identical between VR-
GS (Mdn = 77.50) and GS-Verse (Mdn = 77.50). On average, the VR-GS method scored 73.61±
16.68, while GS-Verse method scored 78.89±11.58, which is above average in both cases, with our
method achieving a slightly higher result than the overall mean for graphical user interfaces (GUIs)
Bangor (2009). As the Shapiro-Wilk test indicated that the data were not normally distributed,
we used the Wilcoxon Signed-Rank Test, which showed that the difference was not statistically
significant (W = 68.0, p = 1.000), with a negligible effect size (r = −0.000) (see Tab. 3).
Perceived Taskload
With respect to TLX results, median workload scores were slightly higher
for VR-GS (Mdn = 7.085) than for GS-Verse (Mdn = 6.665). The mean workload for VR-GS was
9.81±10.11, compared to 8.47±6.13 for GS-Verse. Again, due to the non-normality of the data, we
relied on the nonparametric Wilcoxon Signed-Rank Test, which showed no statistically significant
difference (W = 55.0, p = 0.501), with a small effect size (r = 0.159)(see Tab.
3). The mean
scores in both groups can be considered to belong to the “low” category Prabaswari et al. (2019);
Tadeja et al., as expected given the ease of the manipulation task and the highly usable interface, as
shown by the high SUS scores.
Perceived Flow
For the FLOW method, participants’ scores were slightly lower for VR-GS (M =
4.80, SD = 1.07) compared to GS-Verse (M = 5.03, SD = 0.86). A paired-samples t-test
indicated that this difference, ¯d = −0.233, 95% CI [−0.538, 0.071], was not statistically significant,
t(17) = −1.50, p = .152, with a small effect size (Cohen’s d = −0.354) (see Tab. 4).
Observed Manipulation Issues
We summarize in Tab. 5 the number and percentage of partici-
pants who experienced interaction issues in VR-GS and GS-Verse. Interaction latency issues refer to
delays in the system’s response to participant-invoked manipulations, while interaction range aware-
ness issues indicate participants’ difficulty judging the effective interaction distance. In VR-GS, 5
out of 18 participants (27.8%) reported latency delays during task performance, despite all users op-
erating under the same conditions and hardware. Furthermore, 11 participants (61.1%) experienced
difficulties in accurately perceiving the interaction range from the outset, without guidance or hints.
In contrast, with GS-Verse, only 3 participants (16.7%) reported latency issues, and just 2 partici-
pants (11.1%) faced interaction range awareness difficulties, indicating that our system provides a
more stable and natural interaction experience.
Overall Assessment
We observed no statistically significant differences between VR-GS and GS-
Verse in terms of usability (SUS), taskload (TLX), and (FLOW) (see Fig. 10). However, our ap-
proach led to perceiving physics-aware manipulations as significantly more natural for stretching
3D objects and to a more consistent assessment of other manipulations, such as twisting and shaking
(see Fig. 9).
13

<!-- page 14 -->
Issue Type
VR-GS (n = 18)
GS-Verse (n = 18)
Interaction latency issue ↓
5 (27.78%)
3 (16.67%)
Interaction range awareness issue ↓
11 (61.1%)
2 (11.1%)
Table 5: Number and percentage of participants experiencing interaction issues in VR-GS and GS-
Verse. “Interaction latency issues” indicate moments of delayed system response, while “interaction
range awareness issues” indicate participants’ difficulty judging the effective interaction distance.
5.2
GOAL-DIRECTED ACTION TASK
Perceived Usability
Participants’ usability scores were assessed across three scenes using the Sys-
tem Usability Scale (SUS). The median SUS scores were the same for the dark room and toy room
(86.25), and slightly lower for the garden scene (85.00). Because the Shapiro-Wilk test indicated
that the data were not normally distributed, we used a nonparametric Friedman test to examine
differences in SUS scores across the three scenes. The analysis showed no statistically significant
difference, χ2(2) = 2.81, p = 0.245 (see Fig. 11), while, at the same time, all the results are high
and well above the mean for GUIs Bangor (2009).
Perceived Taskload
Median workload scores were also high and similar across the three scenes,
though slightly lower for the toy room scene, i.e., 5.83, 5.00, and 5.83, respectively. The mean
taskload scores were 7.41 ± 6.20, 6.71 ± 6.57, and 7.68 ± 6.63 for dark room, toy room, and garden
scenes. As the data were non-normal, we conducted a Friedman test, which revealed no statistically
significant differences across the rooms, χ2(2) = 4.33, p = 0.115 (see Tab. 6). Similarly to the
close-ended task, the mean scores fall within the “low” workload category Prabaswari et al. (2019);
Tadeja et al..
Perceived Flow
For the FLOW method, the scores were slightly lower in the dark room scene
(M = 5.14, SD = 1.06) compared to the toy room (M = 5.49, SD = 0.73) and garden
(M = 5.28, SD = 0.81). A one-way repeated-measures ANOVA indicated that these differ-
ences were statistically significant, F(2, 34) = 5.47, p = .009, η2
p = .244. Post hoc comparisons
with Bonferroni correction showed that scores associated with the toy room scene were significantly
higher than in dark room, p = .024, 95% CI [−0.60, −0.10] (see Tab. 7). While the differences
between garden and dark room (p = .602, 95% CI [−0.37, 0.08]) and toy room (p = .122, 95% CI
[0.01, 0.40]) scenes were not statistically significant (see Tab. 6).
A possible explanation for the observed statistical significance between dark room and toy room
scenes is a disruption of the balance between task difficulty and participants’ skills. The flow state
occurs when a task provides an optimal level of challenge, i.e., when it is engaging but not overly
demanding Csikszentmihalyi (1990). In the dark room environment, reduced lighting, lower object
visibility, and higher visual complexity may have made orientation and task performance more diffi-
cult, partially disturbing this balance. Additionally, the high number of visual details, combined with
lower illumination, may have influenced this result, as the environment could be considered more
perceptually demanding, potentially negatively affecting the intensity of the flow experienceHassan
et al. (2020). Moreover, the lack of clearly defined goals in the dark room condition—for instance,
slightly ambiguous instructions such as “move the chair to another corner of the room,” without
specifying which chair or which corner—may have further hindered participants’ sense of control
and clarity, thereby potentially reducing the experienced flow.
Overall Assessment
Our analysis revealed no statistically significant differences between scenes
in terms of usability (SUS) or taskload (TLX). However, a statistically significant difference was
found for the FLOW measure between the dark room and toy room scenes. This finding suggests that
the dark room environment may have disrupted the balance between task difficulty and participants’
skills. The dark room environment’s reduced lighting, greater visual complexity, and less clearly
defined goals likely increased perceptual and cognitive demands, potentially reducing the overall
intensity of the flow experience.
14

<!-- page 15 -->
NASA TLX
SUS
FLOW
Scene
Median Mean ± SD
Scene
Median Mean ± SD
Scene
Median Mean ± SD
dark room
5.83
7.41 ± 6.20
dark room
86.25 84.44 ± 11.17
dark room
5.30
5.14 ± 1.06
toy room
5.00
6.71 ± 6.57
toy room
86.25
85.42 ± 8.80
toy room
5.40
5.49 ± 0.73
garden
5.83
7.68 ± 6.63
garden
85.00 81.94 ± 14.00
garden
5.25
5.28 ± 0.81
Friedman Test
Friedman Test
ANOVA Test
Chi-Square p-value
df
Chi-Square p-value
df
Mean Square
F
df
4.33
0.115
2
2.81
0.245
2
0.663
5.47
2
Sig.
Partial η2
0.009
0.244
Table 6: Descriptive statistics and results of the Friedman tests for NASA TLX (left) and SUS
(middle), and results of the repeated-measures ANOVA for FLOW (right), across the three scenes
— dark room, toy room, and garden — used in the second, goal-directed action task. No statistically
significant differences were observed for NASA TLX and SUS; however, significant differences
were found for Flow.
Comparison
Mean Diff Std. Error
t
df p (Bonferroni) 95% CI Lower 95% CI Upper
dark room vs toy room
-0.350
0.116
-3.01 17
0.024
-0.596
-0.104
dark room vs garden
-0.144
0.109
-1.33 17
0.602
-0.373
0.085
toy room vs garden
0.206
0.093
2.22 17
0.122
0.010
0.401
Table 7: The results of pairwise comparisons (with Bonferroni correction) of FLOW scores across
three scenes, i.e., dark room, toy room, and garden), used in the second, goal-directed action task.
This outcome suggests a statistically significant difference in experienced flow between the dark
room and the toy room.
6
DISCUSSION
We propose a unified, physics-aware interactive system that achieves real-time interactions with 3D
GS representations in VR. Our approach was guided by three core principles: (i) achieving immer-
sive, realistic generative dynamics; (ii) while ensuring high-quality and resource-efficient physics-
aware manipulation across various contexts and use case scenarios; and (iii) offering a unified frame-
work that links visual representation and physical simulation.
Immersive and Realistic Generative Dynamics
The primary goal of our system is to provide
users with an immersive experience where virtual objects behave and deform in a manner that closely
mirrors the real world Kalawsky (1999). This necessitates a move beyond purely geometric defor-
mation towards a system grounded in physical principles. Unlike methods that reconstruct motion
from time-dependent datasets or use generative machine learning, our system simulates dynamics
based on physical laws.
A key limitation of a handful of existing prior works, such as VR-GS Jiang et al. (2024), is their
reliance on a simplified tetrahedral mesh (cage) reconstructed from Gaussian kernels to drive the
simulation. While computationally efficient, this cage only approximates the true object surface.
These core characteristics may lead to object manipulations that users do not perceive as natural.
As noted in the VR-GS, the occurrence of artifacts during large deformation necessitates a more
robust approach than the current simple embedding strategy Jiang et al. (2024). To mitigate this,
they introduced a more complex two-level interpolation scheme.
In contrast, our system, GS-Verse, addresses this problem at its root. By directly integrating the ob-
ject’s mesh with the GS representation, we eliminate the need for such approximations and correc-
tive workarounds. This direct binding ensures that the physical forces are applied to a geometrically
accurate surface, resulting in deformations that are not only physically plausible but also visually
faithful to the object’s intricate details. To evaluate our method, we carried out a comparative user
study involving three different types of physics-aware object manipulations in VR. Our approach led
to perceiving physics-aware manipulations as significantly more natural when stretching 3D objects
and to more consistent results for twisting and shaking (see Fig. 9). Further examination of our
method in different scenes and contexts (see Fig. 1) shows its high usability (≥85.0/100.0), low
15

<!-- page 16 -->
taskload (≤10.0/100.0), and high flow (≥5.0/7.0) when manipulating 3D objects generated with
our method and populating a VR environment (see Tab. 6).
Physics-Aware Resource-Efficient Manipulation
Efficient use of available computing resources
is crucial for interactive VR systems to prevent user discomfort and simulation sickness Deng et al.
(2022); Sutcliffe et al. (2019). While fundamental transformations like rotation, scaling, and trans-
lation are standard, our system enables real-time, physics-based interactions that allow for direct
manipulation and deformation of objects. The challenge lies in achieving this without exceeding
VR’s strict frame budget. The authors of VR-GS correctly identify that a per-Gaussian simulation,
as seen in PhysGaussian Xie et al. (2024), is computationally prohibitive for real-time applications.
Their choice of a reduced tetrahedral mesh and an XPBD-based simulation was a pragmatic com-
promise to meet performance targets.
Our system builds on this insight but takes a different path to optimization. By leveraging the origi-
nal mesh, we can employ well-established and highly optimized algorithms for mesh-based physics
simulation. Crucially, we introduce a novel mapping algorithm that efficiently propagates deforma-
tions from mesh vertices to the thousands of associated Gaussian kernels in a single, parallel-friendly
computational pass. This allows us to handle complex and accurate physical representation while
still operating comfortably within the real-time frame rate, thus achieving the goal of physically
realistic editing without the performance penalty of per-primitive calculations.
Unified Framework.
Our design philosophy adheres strictly to the principle of what you see is
what you simulate Müller et al. (2016). In contrast to the hybrid, two-part approach of VR-GS
Jiang et al. (2024), where the visible Gaussian shells are driven by an invisible cage mesh, our
framework can be considered as truly unified. The surface that the user sees and interacts with–
defined by the GS representation–is directly coupled to the same underlying mesh that the physics
engine simulates. This tight integration ensures that every interaction, no matter how subtle, has
a direct and accurate visual and physical consequence. This unified approach not only enhances
realism but also simplifies the content creation pipeline, as there is no need to generate or manage a
separate physical proxy.
7
CONCLUSION
In this paper, we introduced GS-Verse, a novel mesh-based Gaussian Splatting (GS) system that en-
ables physics-aware interaction with 3D content in VR. Unlike prior approaches that rely on simpli-
fied proxy geometries or tetrahedral cages Jiang et al. (2024), our method directly integrates surface
meshes with Gaussian primitives, providing a unified, geometrically consistent representation. Such
an approach supports both realistic rendering and physically plausible manipulation.
Through a comprehensive user study involving 18 participants and three distinct immersive scenes,
we demonstrated that GS-Verse method delivers high usability, low cognitive workload, and strong
engagement levels. Notably, participants rated the stretching manipulation as significantly more
natural than the state-of-the-art VR-GS baseline Jiang et al. (2024), while maintaining consistent
performance across other physics-aware manipulation types, such as twisting and shaking. These
results confirm the robustness of our approach and its ability to deliver stable, realistic, and intuitive
interactions in complex VR environments.
REFERENCES
Aaron Bangor. Determining What Individual SUS Scores Mean: Adding an Adjective Rating Scale.
4(3), 2009.
Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf
360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pp. 5470–5479, 2022.
Piotr Borycki, Weronika Smolak, Joanna Waczy´nska, Marcin Mazur, Sławomir Tadeja, and Prze-
mysław Spurek.
Gasp:
Gaussian splatting for physic-based simulations.
arXiv preprint
arXiv:2409.05819, 2024.
16

<!-- page 17 -->
John Brooke. SUS: A ’Quick and Dirty’ Usability Scale. In P. W. Jordan, B. Thomas, B. A. Weerd-
meester, and A. L. McClelland (eds.), Usability Evaluation in Industry, pp. 189–194. London:
Taylor and Francis, November 1996.
Jun-Kun Chen, Jipeng Lyu, and Yu-Xiong Wang. Neuraleditor: Editing neural radiance fields via
manipulating point clouds. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 12439–12448, 2023.
Paolo Cignoni, Marco Callieri, Massimiliano Corsini, Matteo Dellepiane, Fabio Ganovelli, and
Guido Ranzuglia.
MeshLab: an Open-Source Mesh Processing Tool.
In Vittorio Scarano,
Rosario De Chiara, and Ugo Erra (eds.), Eurographics Italian Chapter Conference. The Eu-
rographics Association, 2008.
ISBN 978-3-905673-68-5.
doi: 10.2312/LocalChapterEvents/
ItalChap/ItalianChapConf2008/129-136.
Mihaly Csikszentmihalyi.
Flow: The Psychology of Optimal Experience.
Harper Perennial,
New York, NY, March 1990. ISBN 0060920432. URL http://www.amazon.com/gp/
product/0060920432/ref=si3_rdr_bb_product/104-4616565-4570345.
Nianchen Deng, Zhenyi He, Jiannan Ye, Budmonde Duinkharjav, Praneeth Chakravarthula, Xubo
Yang, and Qi Sun. Fov-nerf: Foveated neural radiance fields for virtual reality. IEEE Transactions
on Visualization and Computer Graphics, 28(11):3854–3864, 2022.
Stefan Engeser and Falko Rheinberg. Flow, performance and moderators of challenge-skill bal-
ance.
Motiv. Emot., 32(3):158–172, September 2008.
ISSN 1573-6644.
doi:
10.1007/
s11031-008-9102-4.
Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaus-
sian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural
information processing systems, 37:140138–140158, 2024.
Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of
gaussians. In European Conference on Computer Vision, pp. 165–181. Springer, 2024.
Linus Franke, Laura Fink, and Marc Stamminger. Vr-splatting: Foveated radiance field rendering
via 3d gaussian splatting and neural points. Proceedings of the ACM on Computer Graphics and
Interactive Techniques, 8(1):1–21, 2025.
Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo
Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pp. 5501–5510, 2022.
Antoine Guédon and Vincent Lepetit. Sugar: Surface-aligned gaussian splatting for efficient 3d
mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pp. 5354–5363, 2024.
Sandra Hart and Lowell Staveland. Development of NASA-TLX (Task Load Index): Results of
Empirical and Theoretical Research. Advances in Psychology, pp. 139–183, 1988.
Lobna Hassan, Henrietta Jylhä, Max Sjöblom, and Juho Hamari. Flow in vr: a study on the relation-
ships between preconditions, experience and continued use. In Hawaii International Conference
on System Sciences, pp. 1196–1205, 2020.
Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Bros-
tow. Deep blending for free-viewpoint image-based rendering. ACM Trans. Graph., 37(6), De-
cember 2018. ISSN 0730-0301. doi: 10.1145/3272127.3275084. URL https://doi.org/
10.1145/3272127.3275084.
Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting
for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pp.
1–11, 2024a.
Letian Huang, Jiayang Bai, Jie Guo, Yuanqi Li, and Yanwen Guo. On the error analysis of 3d
gaussian splatting and an optimal projection strategy. In European conference on computer vision,
pp. 247–263. Springer, 2024b.
17

<!-- page 18 -->
Nan Huang, Prashant Goswami, Veronica Sundstedt, Yan Hu, and Abbas Cheddad. Personalized
smart immersive XR environments: a systematic literature review.
The Visual Computer, 41
(11):8593–8626, September 2025. ISSN 1432-2315. doi: 10.1007/s00371-025-03887-9. URL
https://doi.org/10.1007/s00371-025-03887-9.
Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau,
Feng Gao, Yin Yang, et al. Vr-gs: A physical dynamics-aware interactive gaussian splatting
system in virtual reality. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–1, 2024.
Roy S Kalawsky.
Vruse—a computerised diagnostic tool:
for usability evaluation of vir-
tual/synthetic environment systems. Applied ergonomics, 30(1):11–25, 1999.
Robert S. Kennedy, Norman E. Lane, Kevin S. Berbaum, and Michael G. Lilienthal. Simulator sick-
ness questionnaire: An enhanced method for quantifying simulator sickness. The International
Journal of Aviation Psychology, 3(3):203–220, 1993. doi: 10.1207/s15327108ijap0303\_3.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splat-
ting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023.
URL https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.
Michael Laakasuo, Jussi Palomäki, Sami Abuhamdeh, Otto Lappi, and Benjamin Ultan Cow-
ley. Psychometric analysis of the flow short scale translated to Finnish. Scientific Reports, 12
(1), November 2022.
ISSN 2045-2322.
doi: 10.1038/s41598-022-24715-3.
URL https:
//www.nature.com/articles/s41598-022-24715-3. Number: 1 Publisher: Nature
Publishing Group.
Jiahe Li, Jiawei Zhang, Youmin Zhang, Xiao Bai, Jin Zheng, Xiaohan Yu, and Lin Gu.
Geosvr: Taming sparse voxels for geometrically accurate surface reconstruction. arXiv preprint
arXiv:2509.18090, 2025.
Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu,
Songcen Xu, Youliang Yan, et al. Vastgaussian: Vast 3d gaussians for large scene reconstruction.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
5166–5175, 2024a.
Weikai Lin, Yu Feng, and Yuhao Zhu. Rtgs: Enabling real-time gaussian splatting on mobile devices
using efficiency-guided pruning and foveated rendering. arXiv e-prints, pp. arXiv–2407, 2024b.
Tiantian Liu, Adam W Bargteil, James F O’Brien, and Ladislav Kavan. Fast simulation of mass-
spring systems. ACM Transactions on Graphics (TOG), 32(6):1–7, 2013.
Andrea Maggiordomo, Federico Ponchio, Paolo Cignoni, and Marco Tarini. Real-world textured
things: A repository of textured models generated with modern photo-reconstruction tools. Com-
puter Aided Geometric Design, 83:101943, November 2020. ISSN 0167-8396. doi: 10.1016/j.
cagd.2020.101943. URL http://dx.doi.org/10.1016/j.cagd.2020.101943.
Dragan Marinkovic and Manfred Zehn. Survey of finite element method-based real-time simula-
tions. Applied Sciences, 9(14):2775, 2019.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications
of the ACM, 65(1):99–106, 2021.
Matthias Müller, Bruno Heidelberger, Marcus Hennix, and John Ratcliff. Position based dynamics.
Journal of Visual Communication and Image Representation, 18(2):109–118, 2007.
Matthias Müller, Nuttapong Chentanez, and Miles Macklin. Simulating visual geometry. In Pro-
ceedings of the 9th International Conference on Motion in Games, pp. 31–38, 2016.
Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics prim-
itives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15,
2022.
18

<!-- page 19 -->
Atyanti Dyah Prabaswari, Chancard Basumerda, and Bagus Wahyu Utomo. The mental workload
analysis of staff in study program of private educational organization. IOP Conference Series:
Materials Science and Engineering, 528(1):012018, may 2019. doi: 10.1088/1757-899X/528/1/
012018. URL https://dx.doi.org/10.1088/1757-899X/528/1/012018.
Shi Qiu, Binzhu Xie, Qixuan Liu, and Pheng-Ann Heng. Advancing extended reality with 3d gaus-
sian splatting: Innovations and prospects. In 2025 IEEE International Conference on Artificial
Intelligence and eXtended and Virtual Reality (AIxVR), pp. 203–208. IEEE, 2025.
Lukas Radl, Michael Steiner, Mathias Parger, Alexander Weinrauch, Bernhard Kerbl, and Markus
Steinberger. Stopthepop: Sorted gaussian splatting for view-consistent real-time rendering. ACM
Transactions on Graphics (TOG), 43(4):1–17, 2024.
Michael Steiner, Thomas Köhler, Lukas Radl, and Markus Steinberger. Frustum volume caching
for accelerated nerf rendering. Proceedings of the ACM on Computer Graphics and Interactive
Techniques, 7(3):1–22, 2024.
Alistair G Sutcliffe, Charalambos Poullis, Andreas Gregoriades, Irene Katsouri, Aimilia Tzanavari,
and Kyriakos Herakleous. Reflecting on the design process for virtual reality applications. Inter-
national Journal of Human–Computer Interaction, 35(2):168–179, 2019.
Slawomir Tadeja.
Exploring Engineering Applications of Visual Analytics in Virtual Reality.
PhD thesis, Apollo - University of Cambridge Repository, 2020.
URL https://www.
repository.cam.ac.uk/handle/1810/326870.
Sławomir K. Tadeja, Luca O. Solari Bozzi, Kerr D. G. Samson, Sebastian W. Pattinson, and Thomas
Bohné. Exploring the repair process of a 3d printer using augmented reality-based guidance.
Computers & Graphics. ISSN 0097-8493. doi: 10.1016/j.cag.2023.10.017. URL https://
www.sciencedirect.com/science/article/pii/S0097849323002546.
Sławomir Konrad Tadeja, Wojciech Rydlewicz, Yupu Lu, Tomasz Bubas, Maciej Rydlewicz, and
Per Ola Kristensson. Measurement and Inspection of Photo-Realistic 3-D VR Models. IEEE
Computer Graphics and Applications, 41(6):143–151, 2021. doi: 10.1109/MCG.2021.3114955.
Rafał Tobiasz, Grzegorz Wilczy´nski, Marcin Mazur, Sławomir Tadeja, and Przemysław Spurek.
Meshsplats:
Mesh-based rendering with gaussian splatting initialization.
arXiv preprint
arXiv:2502.07754, 2025.
Xuechang Tu, Lukas Radl, Michael Steiner, Markus Steinberger, Bernhard Kerbl, and Fernando
de la Torre. Vrsplat: Fast and robust gaussian splatting for virtual reality. Proceedings of the
ACM on Computer Graphics and Interactive Techniques, 8(1):1–22, 2025.
Joanna Waczy´nska, Piotr Borycki, Sławomir Tadeja, Jacek Tabor, and Przemysław Spurek. Games:
Mesh-based adapting and modification of gaussian splatting. arXiv preprint arXiv:2402.01459,
2024.
Yuze Wang, Junyi Wang, Yansong Qu, and Yue Qi. Rip-nerf: Learning rotation-invariant point-
based neural radiance field for fine-grained editing and compositing. In Proceedings of the 2023
ACM international conference on multimedia retrieval, pp. 125–134, 2023.
Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen,
Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation.
In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21469–21480,
2025.
Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang.
Physgaussian: Physics-integrated 3d gaussians for generative dynamics. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4389–4398, 2024.
Linning Xu, Vasu Agrawal, William Laney, Tony Garcia, Aayush Bansal, Changil Kim, Samuel
Rota Bulò, Lorenzo Porzi, Peter Kontschieder, Aljaž Božiˇc, et al. Vr-nerf: High-fidelity virtual-
ized walkable spaces. In SIGGRAPH Asia 2023 Conference Papers, pp. 1–12, 2023.
19

<!-- page 20 -->
Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-
free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 19447–19456, 2024.
Jinyan Yuan, Bangbang Yang, Keke Wang, Panwang Pan, Lin Ma, Xuehai Zhang, Xiao Liu,
Zhaopeng Cui, and Yuewen Ma. Immersegen: Agent-guided immersive world generation with
alpha-textured proxies. arXiv preprint arXiv:2506.14315, 2025.
Mikołaj Zieli´nski, Krzysztof Byrski, Tomasz Szczepanik, and Przemysław Spurek. Genie: Gaussian
encoding for neural radiance fields interactive editing. arXiv preprint arXiv:2508.02831, 2025.
Dominik Zimny, Artur Kasymov, Adam Kania, Jacek Tabor, Maciej Zi˛eba, Marcin Mazur, and
Przemysław Spurek. Multiplanenerf: Neural radiance field with non-trainable representation.
Expert Systems with Applications, 279:127350, 2025.
20
