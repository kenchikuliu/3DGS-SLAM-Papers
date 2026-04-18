<!-- page 1 -->
SketchRodGS: Sketch-based Extraction of Slender Geometries for
Animating Gaussian Splatting Scenes
Haato Watanabe
The University of Tokyo
Japan
heart.watanabe.research@gmail.com
Nobuyuki Umetani
The University of Tokyo
Japan
n.umetani@gmail.com
(a) 2D user sketch 
(b) Constructed 3D polyline
(c) Interactive deformation
(d) Editing a complicated object
Figure 1: Given a user’s sketch (a) on a Gaussian splatting model, the system constructs a 3D polyline for elastic rod simulation (b),
where the user can deform it by pulling it (c). It is possible to capture and edit complex shapes such as chains (d).
Abstract
Physics simulation of slender elastic objects often requires dis-
cretization as a polyline. However, constructing a polyline from
Gaussian splatting is challenging as Gaussian splatting lacks con-
nectivity information and the configuration of Gaussian primitives
contains much noise. This paper presents a method to extract a
polyline representation of the slender part of the objects in a Gauss-
ian splatting scene from the user’s sketching input. Our method
robustly constructs a polyline mesh that represents the slender
parts using the screen-space shortest path analysis that can be effi-
ciently solved using dynamic programming. We demonstrate the
effectiveness of our approach in several in-the-wild examples.
CCS Concepts
• Computing methodologies →Point-based models; Graphics
systems and interfaces; Mesh geometry models.
Keywords
Sketch-based modeling, Elastic rod simulation, Gaussian splatting
1
Introduction
Recently, the Gaussian splatting [Kerbl et al. 2023] has attracted
significant attention as an efficient and high-quality novel view
synthesis method. The Gaussian splatting represents the radiance
field with a set of Gaussian primitives; hence, it is possible to dy-
namically animate the scene by changing the properties such as
the position and shape of the primitives. Since there are numer-
ous small Gaussian primitives in the scene, it is not practical to
animate the primitives individually. Instead, the primitives are typ-
ically animated by embedding in the mesh deformation [Gao et al.
2024], tetrahedral mesh [Jiang et al. 2024], skinning transforma-
tions, or spatial interpolation from points in the point-based simu-
lation [Feng et al. 2025]. Physics simulation or rigged animation is
performed on such an embedded deformation space.
When constructing a deformation embedding (i.e., parametric
deformation), the deformation modes need to resolve the objects’
geometric detail for detailed animation. This is particularly true for
thin or slender structures such as cloth and rods, where the neigh-
boring geometries (e.g., body and cloth) move independently. The
animation of the thin or slender scene typically requires discretiza-
tion using a mesh that conforms to the geometry. Furthermore, the
mesh is required to be high quality (i.e., clean topology and uniform
edge length) to run a physics simulation on it. Constructing such
a simulation-ready mesh from a model of novel view synthesis is
challenging since the novel view synthesis is optimized for view
interpolation and thus its geometric representation is noisy and
inconsistent with the actual objects. In Gaussian splatting, one can
possibly construct a mesh by connecting the 3D center positions
of Gaussian primitives based on their proximity. However, this ap-
proach tends to be slow and not robust since Gaussian Splatting
prioritizes visual consistency over geometric accuracy.
To address this limitation, we propose a new screen-space algo-
rithm for constructing 3D polylines suitable for elastic rod simula-
tions. Our method lets the user sketch upon the slender part, then it
generates the polyline mesh suitable for elastic rod simulation. Our
method leverages dynamic programming to enable fast, interactive
construction of rod meshes, and introduces a novel procedure to
handle self-intersections in screen space. The proposed approach
allows slender objects to be reconstructed and edited efficiently,
making it a practical tool for simulation-driven content creation.
Our contributions are summarized as
arXiv:2601.02072v1  [cs.GR]  5 Jan 2026

<!-- page 2 -->
Watanabe et al.
• a novel screen-space algorithm for capturing slender struc-
ture from a sketch stroke from a Gaussian splatting scene,
efficiently leveraging the dynamic programming,
• handling partially occluded objects, which is challenging
for screen-space methods,
• several Gaussian splatting datasets for slender objects.
2
Related Work
Neural Deformation & Segmentation. Lan et al. [2024] segment
the Gaussian splatting scene based on the semantic segmentation on
the projected 2D image. Qu et al. [2025] introduced the method to
drag the Gaussian splatting objects by specifying target positions.
However, the use of the score distillation model in moving the
Gaussian primitive is computationally expensive. Our work is based
on the geometric analysis of the projected Gaussian splatting; hence,
our method is lightweight and does not depend on prior knowledge.
Rigged Deformation. Characters represented using Gaussian splat-
ting are often animated using linear blend skinning (e.g., [Kocabas
et al. 2024]). Ma et al. [2024] combine skinning with the blendshape
to construct a high-quality, animatable head model. Our method
also constructs skinning to animate the slender objects in real-time.
While the rigging for character is typically constructed by tem-
plate fitting or neural model (e.g., [Xu et al. 2020]), we construct the
rigging based on the polyline extracted from the Gaussian splatting.
Mesh Embedding. To animate Gaussian splatting with elastic
simulation, VR-GS [Jiang et al. 2024] introduces a two-level embed-
ding of Gaussian primitives inside tetrahedral meshes. However, it
is costly to build a tetrahedral mesh. SuGaR [Guédon and Lepetit
2024] extracts a surface mesh from Gaussian splatting by aligning
the Gaussian primitive to the surface. To achieve large deformation,
Gao et al. [Gao et al. 2024] embed the Gaussian primitives in the
faces of a surface mesh that can be adaptively subdivided. Given
a surface mesh, we can technically construct a polyline from the
surface mesh using skeleton extraction (e.g., [Au et al. 2008]), but it
is still costly to construct a mesh from Gaussian splatting.
Meshless Embedding. PhysGaussian [Xie et al. 2024] and Gauss-
ian splashing [Feng et al. 2025] simulate deformation using point-
based discretization. Elastic deformation using point-based dis-
cretization is suitable for bulky objects, but not suitable for slender
objects such as cloth and rods. The deformation field needs to con-
form to the geometry on which we run the simulation. The elastic
rods need a polyline mesh that aligns with the slender structure.
Sketch-based 3D Scene Manipulation. Our work is greatly inspired
by the sketch-based modeling studies in the computer graphics
community. Specifically, 3-sweep [Chen et al. 2013] provides an
interface to manipulate the image from sketch input. Skippy [Krs
et al. 2017] is a sketching interface for designing 3D curves that
wraps around a 3D object. Please refer to [Liu and Bessmeltsev
2025] for the recent survey. It is still challenging to incorporate an
interactive sketch-based interface for Gaussian splatting because
shape analysis is complex for point-based representation.
3
Method
Challenge. The core of our system lies in the robust connectivity
finding among the Gaussian primitives. Naïve approach finds the
connectivity based on the proximity between Gaussian primitives,
but it is both slow and memory-intensive. Furthermore, the Gauss-
ian primitives in the slender part typically take a highly elongated
shape, making it challenging to analyze proximity. Our approach
finds the connectivity in the 2D screen space instead of the 3D object
space (see Fig. 2). Our key observation is that even the Gaussian
primitives are not geometrically connected in the 3D space; they are
connected once projected onto the screen. After all, the Gaussian
splatting is optimized for the loss based on the projected image.
Image of Primitive Index. For each pixel, we store the index of the
Gaussian primitive if a primitive can be seen in the pixel. Multiple
Gaussian primitives can be seen in a pixel since each Gaussian
primitive has an opacity channel, and the final image is obtained
by alpha blending. Thus, we store the index of the primitive that
contributes the most to the pixel. The primitive can be efficiently
found by slightly modifying the rasterization procedure of the
Gaussian splatting [Kerbl et al. 2023]. We denote the primitives
index as K [𝑖] that is associated with the pixel 𝑖.
User Inputs. Let the user’s stroke be a sequence of coordinates
on the screen s = {𝑝1, 𝑝2, . . . , 𝑝𝑁}. We denote 𝐼: R2 →N as
the function that returns the index of the pixel given the screen
coordinates. We assume that the stroke’s first vertex and the last
vertex are on the object that we want to extract. We denote the
Gaussian primitives corresponding to the first and the last vertices
as 𝑘𝑓𝑖𝑟𝑠𝑡= K [𝐼(𝑝1)], 𝑘𝑙𝑎𝑠𝑡= K [𝐼(𝑝𝑁)]. However, our algorithm,
which we describe below, allows the points in the middle that are
not precisely on top of the object we want to extract. The user also
gives the radius 𝑅∈R of the slender object.
Path Optimization. We find the path on the screen by connecting
pixels from 𝐼(𝑝1) to 𝐼(𝑝𝑁). Our method finds the path on the screen,
minimizing the length of the 3D path that connects Gaussian prim-
itives from 𝑘𝑓𝑖𝑟𝑠𝑡to 𝑘𝑙𝑎𝑠𝑡while following the user’s input stroke.
Given pixel 𝑖and 𝑗are connected, we compute the weight 𝑤𝑖𝑗as
𝑤𝑖𝑗=
®𝑐K[𝑖] −®𝑐K[𝑗]
 + 𝛼𝐷s
 𝐼−1(𝑖)2 ,
(1)
where ®𝑐𝑘∈R3 is the center of the Gaussian primitive 𝑘, 𝐷s is the
minimum distance to the polyline s and a point on the screen, and
𝐼−1 : N →R2 is the function that returns the pixel center given the
pixel index. Note that, in (1), the first term evaluates the length of
the polyline in 3D and the second term evaluates the distance to the
input stroke while 𝛼∈R is a user-defined parameter controlling the
tolerance of the deviation of the path from the stroke (𝛼= 1 in this
paper). A pixel on the screen can connect to adjacent eight pixels
that share an edge or a vertex. Dynamic programming efficiently
finds the path connecting adjacent pixels that minimizes the sum of
the weights (1). We are inspired by the use of dynamic programming
in the sketch snapping [Su et al. 2014].
Occlusion. The algorithm we described so far works well if all the
slender parts are visible, i.e., not occluded by themselves or other
objects. However, as shown in Fig. 3-left, the shortest path may
skip the looped part at the self-intersection or the path may jump

<!-- page 3 -->
SketchRodGS: Sketch-based Extraction of Slender Geometries from 3DGS
(a) Inputs
(b) Screen-space path optimization
(d) Elastic rod sim. 
     & linear blend skinning
(c) Smoothing, resampling
     & segmentation
Polyline
for elastic rod 
simulation
Figure 2: Workflow. (a) The user inputs a stroke s in a viewpoint, and the index of the Gaussian primitive for each pixel K is
computed for that viewpoint. (b) The shortest path connecting pixels is computed by dynamic programming. (c) We smooth
and resample the polyline connecting the center of Gaussian primitives to obtain a polyline for simulation. We segment the
primitives inside the cylinder with radius 𝑅. (d) We animate the Gaussian primitives using the linear blend skinning based on
the elastic rod simulation on the polyline.
to the occluding object. In both cases, the shortest path contains a
discontinuous depth change between adjacent pixels. Thus, in the
dynamic programming, we exclude such pixels from the candidates
when the distance is greater than the 3𝑅.
By eliminating the discontinuous adjacent pixels from the can-
didate, our algorithm may stop where there is no candidate (see
Fig. 3-right). To skip the occluded part, we check the distances
between the vertex of the stroke and the path connecting pixels and
flag covered if it is less than a threshold (10 pixels here). Then, we
resume the path finding from the not-covered vertex of the stroke
with the smallest index whose depth value is similar to the pixel
where the connection is lost. Note that our occlusion handling has
some heuristics and sometimes fails. However, since our algorithm
is very fast, the user can always try again by sketching a more
accurate stroke or sketching from a viewpoint with less occlusion.
Post Process. After computing the shortest path connecting pixels
on the screen, we construct a 3D polyline mesh by connecting the
centers of the Gaussian primitives that are associated with the
pixels. Since the centers of the Gaussian primitives and the centers
of the pixels are not aligned perfectly, the initial 3D polyline is
jagged. We apply a few steps of Laplacian smoothing, and then we
resample the polyline with a user-specified edge length. Once the
Low depth
(occluder)
Stroke
Object to 
segment
Naïve path
Depth discontinuites
High depth
(background)
Med. depth
(target)
Resume
Stop Skip
Figure 3: Left: Naively connecting the shortest path on the
screen may result in discontinuities when there are occlu-
sions. Right: to overcome this issue, we do not connect pixels
with a large depth gap. In addition, we skip the vertices in
the stroke if there is a depth gap.
polyline mesh is computed, we segment each Gaussian primitive if
its center’s shortest distance to the polyline is less than 𝑅.
Finally, we animate Gaussian primitives using the linear blend
skinning. The skinning weight is computed by finding the nearest
point on the polyline to the center of the Gaussian. For the physics
simulation of elastic rods, we use the discrete elastic rod where the
rotation frames are locally updated [Bergou et al. 2010]. The frame
on a vertex is computed by averaging ones on adjacent edges.
4
Results
Implementation detail. The dynamic programming and the elastic
rod simulation are implemented in C++ and Rust, respectively,
while the interface is implemented using Python 1. Because our
method focuses on polyline mesh construction for skinny object,
We collected in-the-wild 10 datasets for 3D Gaussian splatting that
has slender objects (e.g., chain, tube and pole). The training images
are Full HD, taken by author’s iPhone 15 Pro. All the experiments
are conducted with an image resolution of 960 x 540 on a machine
with a GeForce RTX 3090 GPU and an Intel Core i9 CPU. Fig. 4
shows examples of the editing results using our system. For all the
examples, our algorithm roughly takes 0.5 second for processing
the sketch and 50 milliseconds for animation and rendering. Please
refer to the accompanying video for the animation results.
Comparison to the baseline. Fig. 5 shows the comparison against
the naïve method, where the connection between the primitives is
optimized with dynamic programming with the same weight (1),
but the candidates are selected from the eight nearest neighbors
of the primitives’ center based on 3D Euclidean distance. While
the naïve method stops computing the path in the middle due to
the highly elongated anisotropic Gaussian primitives. The naïve
method, implemented in Python, takes 13 seconds to compute until
it stops, excluding the time to construct the acceleration structure.
1The code is available at https://github.com/haato-w/sketch-rod-gs

<!-- page 4 -->
Watanabe et al.
Figure 4: Examples of manipulating slender objects using our system. Each pair images shows the undeformed object (inset)
and resulting real-time deformation by pulling the 3D polyline.
(a) Input sketch
(b) Path finding using 
3D k-nearest neighbour
(c) Our path finding
on the screen 
Gaussian primitives
The method
stops here
Figure 5: Comparison against a baseline using 3D k-nearest
neighbor path finding. Given a stroke (a), the naïve method
stops working in the middle due to noisy, highly elongated
Gaussian primitives (b). On the other hand, our screen space
approach finds the path covering the entire stroke (c).
5
Conclusions & Future Work
In this work, we present an interactive system to construct a poly-
line mesh for elastic rod simulation from a Gaussian splatting model.
We demonstrate the efficiency and robustness of our screen space
algorithm that leverages dynamic programming.
However, when a smooth polyline bends and forms a cusp, the
individual needle-like elongated Gaussian primitives might be visi-
ble as an artifact. This artifact can be prevented by subdividing the
Gaussian primitives beforehand [Gao et al. 2024]. Currently, the
user needs to set the radius of the slender object for the segmenta-
tion. It is left as future work to recognize the shape of the slender
part as a generalized cylinder [Chen et al. 2013] by extracting the
medial axis and cross-section shape. Finally, an interesting future
research avenue is to construct high-quality surface meshes capable
of thin-shell or cloth simulation from screen-space analysis.
Acknowledgments
This work was financially supported by I. Meisters inc.
References
Oscar Kin-Chung Au, Chiew-Lan Tai, Hung-Kuo Chu, Daniel Cohen-Or, and Tong-Yee
Lee. 2008. Skeleton extraction by mesh contraction. ACM Trans. Graph. 27, 3 (2008).
Miklós Bergou, Basile Audoly, Etienne Vouga, Max Wardetzky, and Eitan Grinspun.
2010. Discrete viscous threads. ACM Trans. Graph. 29, 4, Article 116 (July 2010).
Tao Chen, Zhe Zhu, Ariel Shamir, Shi-Min Hu, and Daniel Cohen-Or. 2013. 3-Sweep:
extracting editable objects from a single photo. ACM Trans. Graph. 32, 6 (2013).
Yutao Feng, Xiang Feng, Yintong Shang, Ying Jiang, Chang Yu, Zeshun Zong, Tianjia
Shao, Hongzhi Wu, Kun Zhou, Chenfanfu Jiang, and Yin Yang. 2025. Gaussian
Splashing: Unified Particles for Versatile Motion Synthesis and Rendering. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). 518–529.
Lin Gao, Jie Yang, Bo-Tao Zhang, Jia-Mu Sun, Yu-Jie Yuan, Hongbo Fu, and Yu-Kun
Lai. 2024. Real-time Large-scale Deformation of Gaussian Splatting. ACM Trans.
Graph. 43, 6, Article 200 (Nov. 2024), 17 pages. doi:10.1145/3687756
Antoine Guédon and Vincent Lepetit. 2024. SuGaR: Surface-Aligned Gaussian Splatting
for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. In 2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li,
Henry Lau, Feng Gao, Yin Yang, and Chenfanfu Jiang. 2024. VR-GS: A Physical
Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality. In ACM
SIGGRAPH 2024 Conference Papers (SIGGRAPH ’24).
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 2023.
3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph.
42, 4, Article 139 (July 2023), 14 pages. doi:10.1145/3592433
Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel, Oncel Tuzel, and Anurag
Ranjan. 2024. HUGS: Human Gaussian Splats. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR). 505–515.
Vojtěch Krs, Ersin Yumer, Nathan Carr, Bedrich Benes, and Radomír Měch. 2017. Skippy:
single view 3D curve interactive modeling. ACM Trans. Graph. 36, 4 (2017).
Kun Lan, Haoran Li, Haolin Shi, Wenjun Wu, Lin Wang, and Yong Liao. 2024. 2D-
Guided 3D Gaussian Segmentation. In 2024 Asian Conference on Communication and
Networks (ASIANComNet). 1–5. doi:10.1109/ASIANComNet63184.2024.10811031
Chenxi Liu and Mikhail Bessmeltsev. 2025. State-of-the-art Report in Sketch Processing.
Computer Graphics Forum 44, 2 (2025).
Shengjie Ma, Yanlin Weng, Tianjia Shao, and Kun Zhou. 2024. 3D Gaussian Blendshapes
for Head Avatar Animation. In ACM SIGGRAPH 2024 Conference Papers (SIGGRAPH
’24).
Yansong Qu, Dian Chen, Xinyang Li, Xiaofan Li, Shengchuan Zhang, Liujuan Cao, and
Rongrong Ji. 2025. Drag Your Gaussian: Effective Drag-Based Editing with Score
Distillation for 3D Gaussian Splatting. In Proceedings of the Special Interest Group
on Computer Graphics and Interactive Techniques Conference Conference Papers
(SIGGRAPH Conference Papers ’25). Article 80, 12 pages.
Qingkun Su, Wing Ho Andy Li, Jue Wang, and Hongbo Fu. 2014. EZ-sketching: three-
level optimization for error-tolerant image tracing. ACM Trans. Graph. 33, 4, Article
54 (July 2014), 9 pages. doi:10.1145/2601097.2601202
Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu
Jiang. 2024. PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dy-
namics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR). 4389–4398.
Zhan Xu, Yang Zhou, Evangelos Kalogerakis, Chris Landreth, and Karan Singh. 2020.
RigNet: neural rigging for articulated characters. ACM Trans. Graph. 39, 4, Article
58 (Aug. 2020), 14 pages. doi:10.1145/3386569.3392379
