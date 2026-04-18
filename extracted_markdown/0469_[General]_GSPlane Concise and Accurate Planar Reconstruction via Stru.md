# GSPLANE: CONCISE AND ACCURATE PLANAR RE-CONSTRUCTION VIA STRUCTURED REPRESENTATION

Ruitong Gan1,6, Junran Peng2,6, Yang Liu3,4,6, Chuanchen Luo5,6, Qing Li1, & Zhaoxiang Zhang3,4B   
1 The Hong Kong Polytechnic University 2 University of Science and Technology Beijing   
3 NLPR, MAIS, Institute of Automation, Chinese Academy of Sciences   
4 University of Chinese Academy of Sciences 5 Shandong University 6 Linketic   
ruitong.gan@connect.polyu.hk, jrpeng4ever@126.com   
{liuyang2022, zhaoxiang.zhang}@ia.ac.cn   
chuanchen.luo@sdu.edu.cn, qing-prof.li@polyu.edu.hk

<!-- image-->  
Figure 1: We introduce GSPlane, which adopts 2D planar priors to constrain planar Gaussian distributions on corresponding planes. The structured representation for planes not only empowers the layout refinement of the mesh, resulting in topological correctness with notable less vertices, but also demonstrates the potential in decoupling objects and sealing contact regions on supportive planes.

## ABSTRACT

Planes are fundamental primitives of 3D sences, especially in man-made environments such as indoor spaces and urban streets. Representing these planes in a structured and parameterized format facilitates scene editing and physical simulations in downstream applications. Recently, Gaussian Splatting (GS) has demonstrated remarkable effectiveness in the Novel View Synthesis task, with extensions showing great potential in accurate surface reconstruction. However, even stateof-the-art GS representations often struggle to reconstruct planar regions with sufficient smoothness and precision. To address this issue, we propose GSPlane, which recovers accurate geometry and produces clean and well-structured mesh connectivity for plane regions in the reconstructed scene. By leveraging off-theshelf segmentation and normal prediction models, GSPlane extracts robust planar priors to establish structured representations for planar Gaussian coordinates, which help guide the training process by enforcing geometric consistency. To further enhance training robustness, a Dynamic Gaussian Re-classifier is introduced to adaptively reclassify planar Gaussians with persistently high gradients as nonplanar, ensuring more reliable optimization. Furthermore, we utilize the optimized planar priors to refine the mesh layouts, significantly improving topological structure while reducing the number of vertices and faces. We also explore applications of the structured planar representation, which enable decoupling and flexible manipulation of objects on supportive planes. Extensive experiments demonstrate that, with no sacrifice in rendering quality, the introduction of planar priors significantly improves the geometric accuracy of the extracted meshes across various baselines.

## 1 INTRODUCTION

Planes are commonly witnessed in our daily environments, forming the foundation of many scenes: the streets and building facades outdoors, and the floors and ceilings indoors. When manually constructing digital assets, artists can easily leverage their priors knowledge to accurately model textures and geometric distributions in these areas. Establishing accurate planar structures not only enables concise meshes with considerably fewer vertices and faces, but also support downstream tasks such as physical simulation Qi et al. (2024). In contrast, for example, if a reconstructed table is uneven or lacks geometric consistency, objects like cups would struggle to rest stably on its surface. Naturally, a question arises: in the context of creating digital twins through 3D reconstruction, can the introduction of planar priors help achieve more normal-consistent and accurate geometric reconstructions? Unfortunately, despite recent advancements in 3D reconstruction, there has been limited exploration of how to effectively leverage planar priors to address these challenges.

Recently, 3D Gaussian Splatting (3DGS) Kerbl et al. (2023) introduced an explicit representation capable of achieving high-fidelity novel view synthesis in real time. Rather than relying on neural networks, 3DGS employs Gaussians characterized by parameters such as position, scale, rotation, color, and opacity. Its highly optimized rasterization pipeline enables fast rendering speed. Following 3DGS, several notable works Huang et al. (2024a); Guedon & Lepetit (2024); Zhang et al. Â´ (2024); Yu et al. (2024); Chen et al. (2024) focused on improving the Gaussian representation and depth regularization strategies to gain higher mesh quality in 3D surface reconstruction tasks, which has been extensively studied in the field of computer vision and graphics. Building on these advancements, methods such as GaussianRoom Xiang et al. (2024), AGS-mesh Ren et al. (2024) further integrate prior information to enhance geometric accuracy. However, through our experiments, we observed that the prior knowledge in these methods is typically used as a supervisory signal to minimize regularization losses during training, and the Gaussian representations generated are not strictly constrained to lie on a single plane. Additionally, the meshing strategies adopted in these approaches tend to produce overly dense distributions of vertices and faces, especially for planar regions, leading to high-resolution demands that can be costly and less practical for downstream applications.

To address the aforementioned challenges, we propose GSPlane, a novel method that leverages planar priors from 2D images to generate meshes with consistent normal and coherent topology in planar regions. Our approach begins by estimating surface normal maps Hu et al. (2024) for each posed image and identifying potential planar regions using subpart mask proposals generated by SAM Kirillov et al. (2023). These 2D planar priors are then projected into 3D space to cluster the initial 3D Gaussians into plane-specific groups. We introduce a structured representation for planar Gaussians by re-parameterizing their xyz coordinates into a normalized weighted combination of three non-collinear basis points defining the plane. During training, both the basis pointsâ coordinates and the normalized weights for each planar Gaussian are optimized to refine the planeâs orientation and position. To further improve accuracy, we incorporate a Dynamic Gaussian Reclassifier (DGR), which dynamically corrects false-positive planar Gaussians during training. The extracted mesh will be further refined by leveraging the optimized planar priors, enhancing the surface topology and layout in planar regions. Additionally, we explore Supportive Plane Correction (SPC), an applications of our structured planar representation, demonstrating its ability to improve mesh realism by preserving planar integrity and enabling flexible object manipulation across supportive planes.

To thoroughly evaluate the effectiveness of 2D planar priors, we take both the indoor dataset Scan-NetV2 Dai et al. (2017) and outdoor Tanks and Temples Dataset Knapitsch et al. (2017) as benchmarks. Extensive experiments demonstrate that GSPlane achieves significantly better performance in planar regions, producing meshes with a unified layout and consistent normalsâwhile maintaining rendering quality without any degradation. To summarize, the main contributions of the paper are:

â¢ We propose GSPlane, a powerful method that lifts 2D planar priors into 3D space and establishes a structured representation for planar Gaussians. Additionally, we incorporate optimized planar information during mesh layout refinement, ensuring topological correctness and consistency in the planar regions of the mesh.

â¢ We present Supportive Plane Correction, an application of our structured planar representation that preserves planar integrity when decoupling objects from their supportive planes, enabling accurate planar geometry and facilitating flexible object manipulation.

â¢ Extensive experiments validate our SOTA surface reconstruction performance, showcasing promising benefits of 2D planar prior in 3D reconstruction.

## 2 RELATED WORKS

## 2.1 GAUSSIAN SPLATTING

Extracting accurate surfaces from unordered and discrete 3DGS is both a challenging and fascinating task. Numerous algorithms have been developed to extract high-quality surfaces while ensuring smoothness and managing outliers. The pioneering SuGaR Guedon & Lepetit (2024) approach Â´ pretrains 3DGS and integrates it with the extracted mesh for fine-tuning, utilizing the Poisson reconstruction algorithm for rapid mesh extraction. Techniques like 2DGS Huang et al. (2024a) and GaussianSurfels Dai et al. (2024) reduce the original 3D Gaussian primitives to 2D to avoid ambiguous depth estimation. During GS training, the estimated normals derived from rendering and depth maps are aligned to ensure smooth surfaces. GOF Yu et al. (2024) focuses on unbounded scenes, using ray-tracing-based volume rendering to achieve a contiguous opacity distribution. RaDeGS Zhang et al. (2024) introduces a novel definition of ray intersection with Gaussian structures, deriving curved surfaces and depth distributions. Furthermore, recent works Xiang et al. (2024); Ren et al. (2024); Turkulainen et al. (2024); Wang et al. (2024); Dai et al. (2024); Chen et al. (2024); Zanjani et al. (2025); Li et al. (2025); Sun et al. (2025) incorporate surface normal and monocular depth information predicted from off-the-shelf models as additional supervision in the training process, resulting in improved surface reconstruction quality and geometrical consistency. However, these mesh surfaces are still composed of overly dense distributions of vertices and faces, resulting in topological inaccuracies when compared to real-world structures. This excessive density not only leads to significantly larger file sizes but also poses challenges for subsequent editing and processing tasks.

## 2.2 TRADITIONAL 3D PLANE RECONSTRUCTION

Traditional methods for 3D plane reconstruction often focus on identifying potential plane areas within a scene using RGB-D images Salas-Moreno et al. (2014); Silberman et al. (2012); Huang et al. (2017) or sparse 3D point clouds Borrmann et al. (2011); Sommer et al. (2020). By utilizing sets of points with 3D coordinates, either obtained from point clouds or derived from depth information, robust estimators such as PCA or RANSAC Fischler & Bolles (1981) can be employed to fit geometric representations of planes. Other approaches Gallup et al. (2010); Argiles et al. (2011) tackle the planar reconstruction problem through multi-view image segmentation, where each pixel is assigned to planar proposals represented in Markov Random Fields (MRF). In our research, we propose leveraging planar priors from 2D images to reconstruct target scenes. In earlier attempts, we proposed to directly post-process the reconstructed mesh Barda et al. (2023) via 2D planar priors, which led to significant errors in plane distribution. To address this, we introduced a structured planar representation that is optimized during training, allowing us to leverage learned plane equations to refine the reconstruction.

## 2.3 LEARNABLE 3D PLANE RECONSTRUCTION

With the increasing availability of large-scale datasets containing both 2D images and 3D point clouds, learning-based methods have become the mainstream for extracting planar information from single images or videos. This capability facilitates the reconstruction of potential planes within a scene. Classical approaches, such as PlaneNet Liu et al. (2018), PlaneRecover Yang & Zhou (2018), and PlaneRCNN Liu et al. (2019), segment possible plane distributions from a single image and optimize plane parameters using depth features to achieve a final reconstructed scene. PlanarRecon Xie et al. (2022) is the first method to predict the planar representation of a scene from a sequence of images before reconstruction. Building on previous methods, Airplanes Watson et al. (2024) proposes estimating 3D-consistent plane embeddings and grouping them into scene instances. Uniplane Huang et al. (2024b) uses sparse attention to query per-object embeddings for the scene.

Alphatablets He et al. (2024) employs off-the-shelf surface normal and depth information to initialize small planes, which are further optimized to align with the sceneâs geometry and texture. While these methods show significant promise in reconstructing planar regions, they often produce less detailed and realistic geometric structures in non-planar areas. In contrast, out model well balance the performance in both planar and non-planar areas, achieving high quality for both rendering and surface reconstruction.

## 3 METHODS

<!-- image-->  
Figure 2: Pipeline of GSPlane. Given a set of posed images as input, our method first extracts 2D planar priors from each view, align them with point cloud to obtain plane distributions in 3D space, and re-parameterize the related coordinates of Gaussians. During training, our Dynamic Gaussian Re-classifier continues to correct false-positive planar gaussian by reverting their representation back to xyz. The layout of the mesh extracted from training will also be refined with the optimized planar information from the structured representation.

Figure 2 illustrates the overall pipeline of GSPlane. Starting with posed input images, GSPlane initially extracts planar prior information from each specific view, integrates them into the 3D point cloud, and establishes structured representations for 3D planar points before and during training (Sec. 3.1). A Dynamic Gaussian Re-classifier (DGR) is then employed to refine the optimization process by identifying and correcting false-positive planar Gaussians (Sec. 3.2). Finally, the extracted mesh is refined using the learned planar distributions to enhance surface topology and layout (Sec. 3.3). Additionally, we propose Supportive Plane Correction (SPC), an application incubated from planar prior to improve realism by preserving planar integrity and enabling flexible object manipulation in reconstructed scenes (Sec. 3.4).

## 3.1 STRUCTURED REPRESENTATION FOR PLANES

Given a set of posed images $I _ { p } = \{ I _ { 1 } , I _ { 2 } , \ldots , I _ { n } \}$ , potential planes are detected in each image using surface normal predictions. For each image $I _ { i } ,$ Metric3Dv2 Hu et al. (2024) generates a surface normal map $\hat { N _ { i } } = ( n _ { x } , n _ { y } , n _ { z } ) \in \mathbb { R } ^ { H \times W \times 3 }$ and Segment-Anything-Model (SAM) Kirillov et al. (2023) produces subpart masks $M _ { i } = \{ M _ { i , 1 } , M _ { i , 2 } , . . . , M _ { i , j } \}$ of the scene, where i denotes the i-th image and j for j-th mask. For each mask region $M _ { i , j } , \mathrm { w e }$ compute the cosine similarity between the normals of individual pixels and the average normal of the region. If more than 70% of the pixels in the region exceed the similarity threshold Î±, these pixels are identified as a planar region. Overlapping planar regions are then merged into larger planar masks $P = \{ P _ { 1 } , P _ { 2 } , . . . , P _ { n } \}$ due to normal consistency.

Given an initial point cloud or COLMAP reconstruction of the scene, the coordinates of points in the point cloud are used to initialize the positions of the Gaussians. To incorporate planar priors into training, we establish planar relationships across different Gaussian units by projecting 2D planar masks from multiple views into 3D space. we then construct an undirected graph $G ( V , E )$ where each node $V _ { i }$ corresponds to a point in the point cloud. An edge $E ( V _ { i } , V _ { j } )$ is established between two nodes if the two corresponding points appear together on the same projected planar mask. The weight of the edge represents the frequency of these two points appearing on the same planar mask. Background points can be filtered out using depth information, and planar relationships are aggregated across all views on the graph G. The Leiden algorithm Traag et al. (2019), which is designed to detect communities in weighted graph, clusters nodes in G into planar groups, and will be served as constraints for training Gaussians. More details can be found in Appendix Sec. A.

We assume that if a group of points $V _ { P }$ in the point cloud is determined to lie on a plane, their corresponding Gaussian centers should also reside on the same plane. To impose this constraint, we introduce planar priors to re-parameterize Gaussian coordinates, replacing the direct optimization of $x y z$ positions with normalized weight parameters. Specifically, for a planar cluster $V _ { P }$ RANSAC Fischler & Bolles (1981) is employed to estimate the plane, onto which all Gaussian centers are projected to obtain $V _ { P } ^ { \prime }$ . From $\bar { V } _ { P } ^ { \prime }$ , three non-collinear points $F _ { 1 } , F _ { 2 } , F _ { 3 }$ are randomly selected to serve as basis points defining the plane function. Each projected coordinate in $V _ { P } ^ { \prime }$ is then expressed as a normalized linear combination of the basis points:

$$
V _ { P } ^ { \prime } = \omega _ { 1 } F _ { 1 } + \omega _ { 2 } F _ { 2 } + \omega _ { 3 } F _ { 3 } , \quad \mathrm { s . t . } \quad \omega _ { 1 } + \omega _ { 2 } + \omega _ { 3 } = 1 .\tag{1}
$$

These weights $\omega _ { 1 } , \omega _ { 2 } , \omega _ { 3 }$ are optimized during training to enforce planar constraints on the planar Gaussians.

## 3.2 DYNAMIC GAUSSAIN RE-CLASSIFIER

Building upon the structured representation, planar Gaussians are optimized during training to adhere to planar constraints. The coordinates of planar Gaussians, whether initialized directly from the point cloud or derived through densification, are represented using basis points and normalized weights (Eq. 1). While the coordinates of the basis points are optimized as well, they are assigned a lower learning rate to allow for adjustments in plane orientation and position.

The accuracy of the planar Gaussian relations and the effectiveness of the planar priors are closely tied to the performance of SAM and Metric3Dv2. However, in cases where a Gaussian is misclassified as planar (i.e., a false-positive planar Gaussian), it cannot be correctly optimized according to the planar coordinate formulation in Eq. 1. To address this issue, we propose the Dynamic Gaussian Re-classifier (DGR) to identify and reclassify such false-positive planar Gaussians. During the DGR phases, gradients for both planar and non-planar Gaussians are collected and averaged for evaluation. The top 5% of planar Gaussians, based on their average gradients, are then compared to the average gradient magnitude of the top 20% of non-planar Gaussians. If the gradient magnitude of a planar Gaussian exceeds the average gradient magnitude of the top 20% non-planar Gaussians, the coordinates of that planar Gaussian are re-formulated back into the xyz coordinate format. DGR operates iteratively between Gaussian densification and after the final densification step. The implementation details are provided in Sec. C in the Appendix.

## 3.3 MESH LAYOUT REFINEMENT

Traditional mesh generation methods applied after Gaussian Splatting often produce overly dense meshes with redundant vertices and faces, which not only reduce geometric accuracy but also compromise storage efficiency.To address this, we introduce a mesh layout refinement procedure that leverages planar priors to optimize mesh structure in planar regions. This refinement improves normal consistency, topological coherence, and reduces vertex density, facilitating object decoupling from supportive planes like floors or tables.

Starting from an initial mesh O (e.g., generated via TSDF Curless & Levoy (1996), Marching Tetrahedra Shen et al. (2021), etc.), we first identify clusters of mesh vertices that correspond to known planar regions. These planar relationships are precomputed from the sparse point cloud P cd as sets $V _ { P } ^ { i }$ , where i indexes the i-th detected plane. We transfer these planar relationships from the point cloud to the mesh by assigning mesh vertices to planes using a spatial proximity criterion based on the voxel size Î´. Specifically, for a given plane A, a mesh vertex $v _ { x } \in O$ is considered to belong to plane A if:

$$
\{ v _ { x } \mid \exists v _ { y } \in V _ { P } ^ { A } , | v _ { y } - v _ { x } | < 1 . 5 \delta \land \forall \bar { v } _ { y } \notin V _ { P } ^ { A } , | \bar { v } _ { y } - v _ { x } | > 0 . 5 \delta \} .\tag{2}
$$

This ensures that each mesh vertex is matched to a unique planar region with sufficient spatial confidence.

Once planar vertex clusters are identified, we refine each planar region individually. We begin by removing all mesh faces formed by three vertices lying on the same plane, retaining only the associated vertices. Each planar vertex cluster is then classified into two categories. Boundary vertices, which are connected via mesh edges to vertices outside the planar cluster and thus form the perimeter of the planar region. Interior vertices, which are fully enclosed within the planar region and have no direct connections to non-cluster vertices. Both boundary and interior vertices are projected onto their corresponding planes, defined by the optimized basis points. To regularize the interior structure, we replace interior vertices with a set of uniformly distributed 2D grid points on the plane. These grid points serve as candidates for reconstructing the triangulated surface of the planar region. However, we observe that planar regions in meshes often have irregular shapes, which can cause misalignment between the grid layout and the actual geometry. To mitigate this, we compute the minimum enclosing rectangle (MER) of the projected vertices. The MER provides a consistent 2D bounding frame aligned with the local plane axes, enabling uniform placement of grid points along the x- and y-directions. Considering the actual region of the plane in mesh, grid points falling outside the projected planar region are discarded. The remaining grid points, together with the projected boundary vertices, form a 2D point setthat is triangulated using Delaunay triangulation Lee & Schachter (1980).This produces a set of triangular faces that seamlessly connect the planar interior to its boundary. Finally, the 2D grid coordinates and their associated faces are mapped back into 3D space using the plane basis, and the resulting vertices and faces are integrated into the original mesh. This results in a refined planar region with consistent normals, reduced redundancy, and improved geometric structure. The complete mesh refinement algorithm is detailed in Alg. 2 in the Appendix.

## 3.4 SUPPORTIVE PLANE CORRECTION

Conventional mesh reconstruction methods often merge individual objects and structural elements into a single, overly connected surface. This results in unrealistic geometry, particularly in regions where objects are in contact. For instance, when attempting to digitally separate an object - such as removing a cup from a table - the reconstructed mesh may exhibit gaps or voids in the contact area, failing to preserve the original physical continuity of the supporting surface. To address this challenge, we propose leveraging planar priors to refine mesh representations within designated planar regions. This approach, referred to as Supportive Plane Correction (SPC), is an optional refinement step in our method designed to handle planar surfaces that serve as object-supporting structures, such as tables, shelves, or floors. To address this issue, we introduce an optional refinement step termed Supportive Plane Correction (SPC), which leverages planar priors to improve mesh representations of object-supporting surfaces, such as tables, shelves, or floors. Unlike general planar regions, supportive planes typically exhibit structural incompleteness â characterized by multiple internal voids (e.g., holes within the plane) or missing boundary regions (e.g., incomplete edges). SPC builds upon the mesh layout refinement process described in Sec. 3.3, with key modifications tailored to preserve the integrity of supportive planes. Specifically, during grid point sampling, points that fall outside the initially projected planar region are retained rather than discarded. In contrast, boundary vertices that define voids or holes are excluded from the Delaunay triangulation step.This ensures that the resulting triangulated surface spans the full extent of the plane while avoiding reintroducing known discontinuities. Beyond structural refinement, SPC enables flexible and physically plausible object manipulation. By isolating and sealing the contact regions between objects and their supporting surfaces, individual objects can be repositioned or removed without affecting the geometry of the underlying plane. This capability enhances both the visual realism and editability of the reconstructed scene by preserving planar surface continuity while enabling object-level interaction.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTINGS

Dataset. We conduct extensive experiments on both the indoor dataset ScanNetV2 Dai et al. (2017) and outdoor dataset Tanks and Temples Dataset Knapitsch et al. (2017). Both datasets provides ground-truth mesh for evaluation. We evaluate scenes in terms of geometric accuracy, plane-wise geometric accuracy, and rendering quality compared with previous methods.

Metrics. To evaluate the scene-wise geometric reconstruction performance, we follow the protocol of PlanarRecon Xie et al. (2022) and report metrics including Accuracy, Completion, Precision, Recall, and F-score. Additionally, we adopt the approach from Airplanes Watson et al. (2024) to report planar-wise metrics such as fidelity, completion, and L1 chamfer. These metrics are evaluated on the k = 20 and k = 30 largest planes sampled from ground truth mesh using PlaneRCNN Liu et al. (2019). Note that planar-wise metrics can only be assessed on meshes produced through our Planar-Guided Mesh Extraction, as baseline methods do not incorporate planar information in the extracted mesh. Please refer to airplanes Watson et al. (2024) for more details. To comprehensively evaluate performance, we also provide metrics about rendering quality, including PSNR, SSIM, and LPIPS, as done in 3DGS Kerbl et al. (2023).

<table><tr><td></td><td colspan="5">Geometry</td><td colspan="3">NVS</td><td>Mesh</td></tr><tr><td>Method</td><td>Accâ</td><td>Compâ</td><td>Precâ</td><td>Recallâ</td><td>F-scoreâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>Vertices</td></tr><tr><td>GaussianRoom Xiang et al. (2024)</td><td>0.084</td><td>0.062</td><td>0.602</td><td>0.621</td><td>0.611</td><td>0.779</td><td>23.89</td><td>0.36</td><td>3.01M</td></tr><tr><td>Alphatablets He et al. (2024)</td><td>00.094</td><td>0.219</td><td>0.501</td><td>0.446</td><td>00.459</td><td>-</td><td>-</td><td>-</td><td>139.4K</td></tr><tr><td>3DGS Kerbl et al. (2023)</td><td>0.083</td><td>0.099</td><td>0.453</td><td>0.429</td><td>0.436</td><td>0.849</td><td>23.494</td><td>0.321</td><td>2.24M</td></tr><tr><td>3DGS + Ours-train</td><td>088</td><td>0.097</td><td>0.459</td><td>0.438</td><td>0.446</td><td>0.853</td><td>23.718</td><td>0.320</td><td>2.00M</td></tr><tr><td>3DGS + Ours-full</td><td>0077</td><td>0080</td><td>0.471</td><td>0.656</td><td>0.548</td><td>-</td><td>-</td><td>-</td><td>1.23M</td></tr><tr><td>2DGS Huang et al. (2024a)</td><td>0.066</td><td>0.078</td><td>0.603</td><td>0.568</td><td>0.583</td><td>0.845</td><td>22.673</td><td>0.346</td><td>1.73M</td></tr><tr><td>2DGS + Ours-train</td><td>0.063</td><td>0073</td><td>0.650</td><td>0.620</td><td>0.6333</td><td>00.847</td><td>23.263</td><td>0..37</td><td>1.60M</td></tr><tr><td>2DGS + Ours-full</td><td>00.058</td><td>00.062</td><td>0.664</td><td>0.716</td><td>0.689</td><td>-</td><td></td><td>-</td><td>946.1K</td></tr><tr><td>GOF (Tetra.) Yu et al. (2024)</td><td>0.120</td><td>0.111</td><td>0.413</td><td>0.484</td><td>0.444</td><td>0.810</td><td>21.444</td><td>0.357</td><td>41.7M</td></tr><tr><td>GOF (TSDF) + Ours-train</td><td>0.100</td><td>0.091</td><td>0.477</td><td>0.598</td><td>0.5528</td><td>0.828</td><td>22.460</td><td>0.359</td><td>1.89M</td></tr><tr><td>GOF (TSDF) + Ours-full</td><td>.086</td><td>080</td><td>00.482</td><td>0.686</td><td>0.566</td><td></td><td></td><td>-</td><td>1.02M</td></tr><tr><td>RaDe-GS Zhang et al. (2024)</td><td>0.101</td><td>0.104</td><td>0.480</td><td>0.507</td><td>0.491</td><td>0.829</td><td>22.334</td><td>0.348</td><td>1.49M</td></tr><tr><td>RaDe-GS + Ours-train</td><td>00.096</td><td>0.101</td><td>00.507</td><td>0.558</td><td>0.528</td><td>00.832</td><td>22.394</td><td>0..351</td><td>1.45M</td></tr><tr><td>RaDe-GS + Ours-full</td><td>0.082</td><td>0086</td><td>00.520</td><td>0.674</td><td>0.587</td><td>-</td><td>-</td><td>-</td><td>794.3K</td></tr><tr><td>PGSR Chen et al. (2024)</td><td>0.079</td><td>0.085</td><td>0.581</td><td>0.571</td><td>0.573</td><td>0.847</td><td>25.350</td><td>0.274</td><td>5.3M</td></tr><tr><td>PGSR + Ours-train</td><td>0.065</td><td>0063</td><td>0.633</td><td>0.640</td><td>0.63</td><td>0.852</td><td>25.494</td><td>0.261</td><td>5.2M</td></tr><tr><td>PGSR + Ours-full</td><td>0.062</td><td>0.059</td><td>0.636</td><td>.658</td><td>0.646</td><td>-</td><td>-</td><td>-</td><td>2.9M</td></tr></table>

Table 1: Quantitative evaluations including both the overall geometric scores and novel view synthesis (NVS) metrics on ScanNetV2 Dai et al. (2017) scenes. âOurs-trainâ denotes applying structured representation for planes and DGR. âOurs-fullâ denotes additionally applying mesh layout refinement after training.

<table><tr><td>Metric</td><td>3DGS</td><td>3DGS + Ours</td><td>2DGS</td><td>2DGS + Ours</td><td>GOF</td><td>GOF + Ours</td><td>RaDe-GS</td><td>RaDe-GS + Ours</td><td>PGSR</td><td>PGSR + Ours</td></tr><tr><td>F-scoreâ</td><td>0.09</td><td>0.17</td><td>0.32</td><td>0.34</td><td>0.46</td><td>0.47</td><td>0.40</td><td>0.42</td><td>0.52</td><td>0.52</td></tr><tr><td>Planar Vertices</td><td>317.5K</td><td>4.53K</td><td>609.3K</td><td>6.94K</td><td>3.04M</td><td>41.26K</td><td>503.1K</td><td>6.89K</td><td>2.39M</td><td>29.27K</td></tr><tr><td>Overall Mesh Vertices</td><td>1.86M</td><td>1.55M</td><td>3.75M</td><td>3.03M</td><td>57.82M</td><td>53.58M</td><td>2.39M</td><td>1.76M</td><td>14.69M</td><td>12.04M</td></tr></table>

Table 2: Quantitative evaluations on Tanks and Temples Dataset Knapitsch et al. (2017).

Implementation Details We implement our GSPlane method on five representative GS-based methods, including 3DGS Kerbl et al. (2023), 2DGS Huang et al. (2024a), GOF Yu et al. (2024), RaDe-GS Zhang et al. (2024), and PGSR Chen et al. (2024). The initial mesh is extracted with the proposed process from the baseline, with the voxel size as 0.005. Note that the Marching Tetrahedral used in GOF closes all boundaries, including the ceilings of indoor scenes and empty plane regions, which violate the actual mesh distribution. Thus, when introducing our strategy to GoF, we abort this technique and turn to TSDF fusion for mesh extraction, so as to avoid mesh in actually empty areas. During the experiment, we set the threshold of cosine similarity Î± to 0.98.

## 4.2 OVERALL PERFORMANCE

The indoor quantitative results of the overall metrics are presented in Tab. 1. Specifically, Ourstrain denotes applying structured representation of planes and Dynamic Gaussain Re-classifier in the training stage, while Ours-full further incorporates mesh layout refinement in the post-training stage. Note that the Supportive Plane Correction (SPC) step is excluded from the performance evaluation. For a fair comparison, we also report results from GaussianRoom Xiang et al. (2024) and AlphaTablets He et al. (2024), which leverage normal maps, depth, and edge information as priors for reconstruction. Compared with the methods that adopt off-the-shelf predictions for direct supervision, our GSPlane demonstrates the effectiveness of incorporating planar priors. The results highlight that the structured plane representation consistently improves both geometric and rendering quality across baselines, while the proposed mesh layout refinement enables more accurate and complete surface estimation. Ours-train achieves a slight reduction in vertex count compared to baseline methods because it produces tighter and more compact planar distribution of Gaussians, while Ours-full significantly reduces the number of vertices in the final mesh. Notably, the structured Gaussian planar representation also contributes to enhanced rendering quality, see Sec. D in Appendix for rendering visualizations.

<!-- image-->  
Figure 3: Visualizations of the mesh performance on both indoor and outdoor scenes. We provide comparisons on four baseline methods. It can be seen from the refined normal map and wireframes that our method can reduce the number of vertices by large margin, while maintaining consistent normal and topology across different planes. More examples can be found in Appendix.

The outdoor quantitative results are displayed in Tab. 2, where we report the F-score as the reconstruction metric, along with the number of planar and total vertices for comparison. Ours in Tab. 2 corresponds to the Ours-full configuration in Tab. 1. As seen in the table, our method improves reconstruction performance in outdoor scenes while significantly reducing the number of vertices in the mesh. However, the geometric improvements are less pronounced compared to indoor scenes, primarily because the TNT dataset contains fewer planar regions in some scenarios compared to ScanNetV2. Nevertheless, our method still achieves substantial reductions in mesh vertex count, demonstrating its efficiency in outdoor settings. Visualizations for both indoor and outdoor scenes can be found in Fig. 3 and Fig. 7 in the Appendix.

## 4.3 PLANAR-WISE GEOMETRY

The planar metrics, including Fidelity, Accuracy, and L1-Chamfer Distance, are presented in Tab.3. Our proposed planar-guided mesh extraction demonstrates significant potential for improving the reconstruction of planar regions across various Gaussian Splatting baselines. More visualizations on processing planar priors and mesh quality comparison can be found in Appendix Sec. E.

## 4.4 ABLATION STUDY

We conduct an ablation study to evaluate the effectiveness of different modules in GSPlane, including the optimization of basis points, the Dynamic Gaussian Re-classifier (DGR), and the postrefinement of the mesh layout. The results are presented in the left table of Fig. 4. Compared to the baseline performance of 2DGS, our GSPlane significantly enhances the quality of the generated mesh. Additionally, we perform experiments on 2DGS and RaDe-GS, both of which estimate normal maps during the rasterization process. Our goal is to analyze the differences between our proposed structured representation and directly using off-the-shelf normal maps to supervise the estimated normals. As shown in the right table of Fig. 4, adopting our structured representation leads to better geometric performance in the reconstructed mesh. For ablation studies on hyperparameters, please refer Tab. 4 and Tab. 5 in Appendix Sec. A.

<table><tr><td>Method</td><td>Fidelityâ</td><td>Accâ</td><td>CDâ</td></tr><tr><td>PlanarRecon Xie et al. (2022)</td><td>18.86</td><td>16.21</td><td>17.53</td></tr><tr><td>AirPlanes Watson et al. (2024) PlanarSplatting Tan et al. (2025)</td><td>8.76 6.64</td><td>7.98 11.76</td><td>8.37 9.2</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td> $3 \mathrm { D G S } + \mathrm { O u r s } { \cdot } \mathrm { f u l l }$ </td><td>6.21 / 6.75</td><td>7.95 / 8.15</td><td>7.08 / 7.35</td></tr><tr><td> $2 \mathrm { D G S } + \mathrm { O u r s } { - } \mathrm { f u l l }$ </td><td>5.49 / 5.82</td><td>7.32 / 7.71</td><td>6.41 / 6.77</td></tr><tr><td> $\mathrm { G O F + O u r s â f u l l }$ </td><td>8.25 / 8.74</td><td>9.50 / 9.93</td><td>8.88 / 9.34</td></tr><tr><td>RaDe-GS + Ours-full</td><td>7.57 / 7.83</td><td>6.34 / 6.60</td><td>6.96 / 7.22</td></tr><tr><td>PGSR + Ours-full</td><td>5.24 / 5.39</td><td>6.58 / 6.65</td><td>5.91 / 6.02</td></tr></table>

Table 3: Planar-wise metrics evaluated on $k = 2 0 / k = 3 0$ largest plane regions from gt mesh in ScanNetV2, following Airplanes Watson et al. (2024). The results from methods displayed in grey are evaluated with k = 20 from the papers.

<table><tr><td></td><td>Precâ</td><td>Recallâ</td><td>F-scoreâ</td></tr><tr><td>2DGS</td><td>0.603</td><td>0.568</td><td>0.583</td></tr><tr><td>+ Train w/o basis points</td><td>0.637</td><td>0.596</td><td>0.616</td></tr><tr><td>+ Train w/o DGR</td><td>0.648</td><td>0.613</td><td>0.630</td></tr><tr><td>+ Train</td><td>0.650</td><td>0.620</td><td>0.633</td></tr><tr><td>+ Train + Mesh Ref.</td><td>0.664</td><td>0.716</td><td>0.689</td></tr></table>

<table><tr><td>Setting</td><td>Accâ</td><td>Compâ</td><td>Precâ</td><td>Recallâ</td><td>F-scoreâ</td></tr><tr><td>2DGS Huang et al. (2024a)</td><td>0.0661</td><td>0.0782</td><td>0.6035</td><td>0.5676</td><td>0.5834</td></tr><tr><td>2DGS + normal</td><td>0.0645</td><td>0.0764</td><td>0.6396</td><td>0.5972</td><td>0.6177</td></tr><tr><td>2DGS + Ours-train</td><td>0.0630</td><td>0.0733</td><td>0.6501</td><td>0.6197</td><td>0.6330</td></tr><tr><td>RaDe-GS Zhang et al. (2024)</td><td>0.1008</td><td>0.1041</td><td>0.4805</td><td>0.5069</td><td>0.4914</td></tr><tr><td>RaDe-GS + normal</td><td>0.0947</td><td>0.1024</td><td>0.5179</td><td>0.5388</td><td>0.5281</td></tr><tr><td>RaDe-GS + Ours-train</td><td>0.0960</td><td>0.1016</td><td>0.5069</td><td>0.5576</td><td>0.5283</td></tr></table>

Figure 4: Ablation study results on GSPlane. The left table shows the effectiveness of different modules in GSPlane, and the right table compares our structured representation with off-the-shelf normal map supervision for mesh geometry reconstruction.

## 4.5 APPLICATION ON SUPPORTIVE PLANE

To validate the effectiveness of Supportive Plane Correction (SPC), we conducted experiments demonstrating its ability to accurately reconstruct supportive planes and decouple objects resting on them. As shown in the left of Fig. 5, the default result of mesh layout refinement can provide unified grid points on plane, but the boundaries of the placed object are connected with the grid points to maintain wholeness of the structure. By fully utilizing the optimized planar priors, it is possible to infer the real shape and structure of the supportive plane - desk, and objects placed on the desk can also be removed from the desk. This ensures that the reconstructed supportive plane remains continuous and free of artifacts, even in the presence of complex void geometries. The hole of the objects at the contact area can also be sealed using the supportive plane function, and are further free to manipulate across the supportive plane or within the scene.

<!-- image-->  
Mesh Layout Refinement (w/o SPC)

<!-- image-->  
Reconstruction of Desk (Supportive)

<!-- image-->  
Decoupled Objects  
Figure 5: Visualizations of Supportive Plane Correction. When running SPC, the object boundaries are excluded from plane reconstruction, leading to an intact plane with complete shape like in reality. The objects are decoupled from the supportive plane surface, and can be further moved or manipulated freely.

## 5 CONCLUSION

In this paper, we highlight the potential of incorporating plane prior knowledge into Gaussian Splatting for improved reconstruction of planar regions. By leveraging segmentation and surface normal estimation, GSPlane generates structured planar representations, improving the geometric accuracy and topological consistency of meshes while reducing the density of vertices and faces. Additional discussion on supportive plane demonstrates that our structured planar representation enables realistic plane completion and decouples objects from planes, allowing further object manipulation. Our experiments demonstrate that leveraging this prior significantly enhances the geometric accuracy and topological consistency of extracted meshes, reducing the complexity of the mesh structure.

## REFERENCES

Alberto Argiles, Javier Civera, and Luis Montesano. Dense multi-planar scene estimation from a sparse set of images. In 2011 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 4448â4454. IEEE, 2011.

Amir Barda, Yotam Erel, Yoni Kasten, and Amit H Bermano. Roar: robust adaptive reconstruction of shapes using planar projections. arXiv preprint arXiv:2307.00690, 2023.

Dorit Borrmann, Jan Elseberg, Kai Lingemann, and Andreas Nuchter. The 3d hough transform for Â¨ plane detection in point clouds: A review and a new accumulator design. 3D Research, 2(2):1â13, 2011.

Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics, 2024.

Brian Curless and Marc Levoy. A volumetric method for building complex models from range images. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pp. 303â312, 1996.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828â5839, 2017.

Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu, Huamin Wang, and Weiwei Xu. High-quality surface reconstruction using gaussian surfels. In ACM SIGGRAPH 2024 Conference Papers, pp. 1â11, 2024.

Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24 (6):381â395, 1981.

David Gallup, Jan-Michael Frahm, and Marc Pollefeys. Piecewise planar and non-planar stereo for urban scene reconstruction. In 2010 IEEE computer society conference on computer vision and pattern recognition, pp. 1418â1425. IEEE, 2010.

Antoine Guedon and Vincent Lepetit. Sugar: Surface-aligned gaussian splatting for efficient Â´ 3d mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5354â5363, 2024.

Yuze He, Wang Zhao, Shaohui Liu, Yubin Hu, Yushi Bai, Yu-Hui Wen, and Yong-Jin Liu. Alphatablets: A generic plane representation for 3d planar reconstruction from monocular videos. arXiv preprint arXiv:2411.19950, 2024.

Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pp. 1â11, 2024a.

Jingwei Huang, Angela Dai, Leonidas J Guibas, and Matthias NieÃner. 3dlite: towards commodity 3d scanning for content creation. ACM Trans. Graph., 36(6):203â1, 2017.

Yuzhong Huang, Chen Liu, Ji Hou, Ke Huo, Shiyu Dong, and Fred Morstatter. Uniplane: Unified plane detection and reconstruction from posed monocular videos. arXiv preprint arXiv:2407.03594, 2024b.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splat- Â¨ ting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015â4026, 2023.

Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4):1â13, 2017.

Der-Tsai Lee and Bruce J Schachter. Two algorithms for constructing a delaunay triangulation. International Journal of Computer & Information Sciences, 9(3):219â242, 1980.

Deqi Li, Shi-Sheng Huang, and Hua Huang. Mpgs: Multi-plane gaussian splatting for compact scenes rendering. IEEE Transactions on Visualization and Computer Graphics, 2025.

Chen Liu, Jimei Yang, Duygu Ceylan, Ersin Yumer, and Yasutaka Furukawa. Planenet: Piecewise planar reconstruction from a single rgb image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2579â2588, 2018.

Chen Liu, Kihwan Kim, Jinwei Gu, Yasutaka Furukawa, and Jan Kautz. Planercnn: 3d plane detection and reconstruction from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4450â4459, 2019.

Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma. Shapellm: Universal 3d object understanding for embodied interaction. In European Conference on Computer Vision, pp. 214â238. Springer, 2024.

Xuqian Ren, Matias Turkulainen, Jiepeng Wang, Otto Seiskari, Iaroslav Melekhov, Juho Kannala, and Esa Rahtu. Ags-mesh: Adaptive gaussian splatting and meshing with geometric priors for indoor room reconstruction using smartphones. arXiv preprint arXiv:2411.19271, 2024.

Renato F Salas-Moreno, Ben Glocken, Paul HJ Kelly, and Andrew J Davison. Dense planar slam. In 2014 IEEE international symposium on mixed and augmented reality (ISMAR), pp. 157â164. IEEE, 2014.

Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems, 34:6087â6101, 2021.

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer VisionâECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pp. 746â760. Springer, 2012.

Christiane Sommer, Yumin Sun, Leonidas Guibas, Daniel Cremers, and Tolga Birdal. From planes to corners: Multi-purpose primitive detection in unorganized 3d point clouds. IEEE Robotics and Automation Letters, 5(2):1764â1771, 2020.

Jingxiang Sun, Cheng Peng, Ruizhi Shao, Yuan-Chen Guo, Xiaochen Zhao, Yangguang Li, Yanpei Cao, Bo Zhang, and Yebin Liu. Dreamcraft3d++: Efficient hierarchical 3d generation with multiplane reconstruction model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Bin Tan, Rui Yu, Yujun Shen, and Nan Xue. Planarsplatting: Accurate planar surface reconstruction in 3 minutes. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1190â1199, 2025.

Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. From louvain to leiden: guaranteeing well-connected communities. Scientific reports, 9(1):1â12, 2019.

Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto Seiskari, Esa Rahtu, and Juho Kannala. Dn-splatter: Depth and normal priors for gaussian splatting and meshing. arXiv preprint arXiv:2403.17822, 2024.

Jiepeng Wang, Yuan Liu, Peng Wang, Cheng Lin, Junhui Hou, Xin Li, Taku Komura, and Wenping Wang. Gaussurf: Geometry-guided 3d gaussian splatting for surface reconstruction. arXiv preprint arXiv:2411.19454, 2024.

Jamie Watson, Filippo Aleotti, Mohamed Sayed, Zawar Qureshi, Oisin Mac Aodha, Gabriel Brostow, Michael Firman, and Sara Vicente. Airplanes: Accurate plane estimation via 3d-consistent embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5270â5280, 2024.

Haodong Xiang, Xinghui Li, Xiansong Lai, Wanting Zhang, Zhichao Liao, Kai Cheng, and Xueping Liu. Gaussianroom: Improving 3d gaussian splatting with sdf guidance and monocular cues for indoor scene reconstruction. arXiv preprint arXiv:2405.19671, 2024.

Yiming Xie, Matheus Gadelha, Fengting Yang, Xiaowei Zhou, and Huaizu Jiang. Planarrecon: Real-time 3d plane detection and reconstruction from posed monocular videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6219â6228, 2022.

Fengting Yang and Zihan Zhou. Recovering 3d planes from a single image via convolutional neural networks. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 85â100, 2018.

Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (TOG), 43(6):1â13, 2024.

Farhad G Zanjani, Hong Cai, Hanno Ackermann, Leila Mirvakhabova, and Fatih Porikli. Planar gaussian splatting. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 8905â8914. IEEE, 2025.

Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. Radegs: Rasterizing depth in gaussian splatting. arXiv preprint arXiv:2406.01467, 2024.

## A ADDITIONAL ILLUSTRATION ON STRUCTURED REPRESENTATION FOR PLANES

This section provide additional illustrations on how the structured representation for planes are obtained from 2D images. After obtaining the normal maps and subpart mask proposals from off-theshelf models, we first multiply the normal map $N _ { i }$ with each mask proposal $\bar { M _ { i , j } }$ , where i denotes the i-th image and $j \cdot$ -th mask, to isolate the normal distribution $N _ { m a s k }$ within each instance region. To determine if a region is planar, we take cosine similarity to measure the distance between each pixel normal to the average normal within the instance. Empirically, if more than 70% of the pixels have a similarity larger than a certain threshold $\alpha ,$ , we regard these pixels as a single plane. The largest connected region of these valid pixels is then selected as a plane proposal. In case multiple planes are mistakenly segmented into a single mask and do not meet the previous condition, we apply K-means clustering to the normals in this region with pixel number bigger than $\sigma .$ We then evaluate each cluster using the 70% criterion to identify all potential planes. If none of the clusters meet the criterion, the mask proposal is considered non-planar. In our experience, setting the target number of clusters to 2 yields good results. By following these steps, we can identify all the plane proposals $M _ { p l a n e } ^ { i }$ in image $I _ { i }$

To address the potential intersections among the obtained plane proposals $\mathcal { M } _ { p l a n e } ^ { i }$ , we implement a series of steps to resolve conflicts in these overlapping areas. We first define an empty list $\mathcal { M } _ { r } ^ { i }$ merge to store the exclusive planar masks after the process. We iteratively select each element $M _ { p l a n \epsilon } ^ { i , k }$ in $\mathcal { M } _ { p l a n e } ^ { i } .$ , and compute normal vector cosine similarity with all other proposals $M _ { p l a n e } ^ { i , l } .$ e. If any proposals matches through aforementioned 70% criteria, they are merged together with $M _ { p l a n e } ^ { i , k }$ and pop out from $\mathcal { M } _ { p l a n e } ^ { i } .$ . The final $M _ { p l a n \epsilon } ^ { i , k }$ will be stored in $\mathcal { M } _ { m e r g e } ^ { i } .$ After completing all the planar proposals in $\mathcal { M } _ { p l a n e } ^ { i } ,$ we achieve a collection of mutually exclusive planar masks $\mathcal { M } _ { m e r g e } ^ { i }$ . By assigning each element with an index, we are able to obtain the final planar mask $P _ { i }$ . The overall algorithm is detailed in Alg. 1.

<!-- image-->

<!-- image-->  
Figure 6: Illustration of 2 possible situations when encountering occlusion. Here red region and yellow region are denoted as occluded points as they are not visible in the camera. In both situation, the red region will be filtered by clustering the depth information.

When lifting 2D priors into 3D space, given a planar instance map $P _ { i }$ with corresponding extrinsics $[ R _ { i } , t _ { i } ] ,$ , and the intrinsic matrix $K ,$ , we begin by projecting all nodes $V$ back into 2D camera coordinates. For each plane instance indicated in $P _ { i } ,$ there is a group of points $V _ { G }$ projected onto this region. We perform K-means clustering on projected depths with $K = 2$ to coarsely filter out occluded points that may not appear in the image. An illustration figure of this process is shown in Fig. 6. The occluded points will be projected onto the plane region together with the foreground points. We only consider the closest as plane-related points in each camera pose, so filtering out points with larger depth is necessary. Points with similar depths in one camera can be further distinguished through other views. The filtered point set is denoted as $V _ { G } ^ { \prime }$ , and the edge $E ( V _ { x } , V _ { y } \in V _ { G } ^ { \prime } )$ will be established among these points, as they are considered to be in the same plane from the plane instance $P _ { i }$ . For every two nodes $V _ { x } , V _ { y } \in V _ { G } ^ { \prime }$ , the edge $E ( V _ { x } , V _ { y } )$ will be created with the weight of 1 if it doesnât exist before. Otherwise, the weight will be incremented by 1. Using Leiden Algorithm to divide different communities, we identify the Gaussians distributed across each plane in the scene.

Algorithm 1 2D Planar Perception   
Require: normal map $N _ { i }$ , mask proposals $\{ M _ { i , j } \}$   
1: for each $M _ { i , j }$ do   
2: $N _ { \mathrm { m a s k } }  \tilde { N } _ { i } \odot M _ { i , j } ,$   
3: $d \gets c o s \_ s i m ( N _ { \mathrm { m a s k } } , \overline { { N _ { \mathrm { m a s k } } } } )$   
4: if rati $) ( d > \alpha ) < 0 . 3$ then   
5: $M _ { p l a n e } ^ { i , j }  M _ { i , j } [ d > \alpha ]$   
6: else if $\mathrm { A r e a } ( N _ { m a s k } ) > \sigma$ then   
7: $N _ { c l u s t e r 1 } , N _ { c l u s t e r 2 } \gets \mathrm { K \mathrm { - } m e a n s } ( N _ { m a s k } )$   
8: Repeat Step 2-5 on $N _ { c l u s t e r 1 } , N _ { c l u s t e r 2 }$   
9: end if   
10 : end for   
11: $\mathcal { M } _ { p l a n e } ^ { i }  [ M _ { p l a n e } ^ { i , j } ]$   
12: $\mathcal { M } _ { m e r g e } ^ { i } $ empty list   
13: while $\mathcal { M } _ { p l a n e } ^ { i }$ not empty do   
14: $M _ { \mathrm { p l a n e } } ^ { i , k }  { \mathcal { M } } _ { p l a n e } ^ { i } [ 0 ]$   
15: for each $l \neq$ k do   
16: $d ^ { \prime } \gets c o s \_ s i m ( \overline { { M _ { \mathrm { p l a n e } } ^ { i , k } } } , \overline { { M _ { p l a n e } ^ { i , l } } } )$   
17: if $d ^ { \prime } > \alpha$ then   
18: $M _ { \cap } \gets M _ { \mathrm { p l a n e } } ^ { i , k } \cap M _ { p l a n e } ^ { i , l }$   
19: $M _ { \mathrm { p l a n e } } ^ { i , k }  \mathrm { \hat { \it M } } _ { \mathrm { p l a n e } } ^ { i , k } + \mathrm { \hat { \it M } } _ { p l a n e } ^ { i , l } - M _ { \cap }$   
20: $\mathcal { M } _ { p l a n e } ^ { i } . \mathrm { p o p } ( M _ { \mathrm { p l a n e } } ^ { i , l } )$   
21: end if   
22: end for   
23: $\mathcal { M } _ { m e r g e } ^ { i } . p u s h ( M _ { \mathrm { p l a n e } } ^ { i , k } )$   
24: end while   
25: Pi â assign instance ID with $\mathcal { M } _ { m e r g e } ^ { i }$   
26: return $P _ { i }$

The ablation studies for hyperparameters Î±, Ï are displayed in Tab. 4 and Tab. 5. Here, we choose RaDe-GS as the baseline method, and run full settings of GSPlane. When implementing our experiments, we choose $\alpha = 0 . 9 8$ and $\sigma = 2 0 0$ as our settings.

<table><tr><td>Î±</td><td>Accâ</td><td>Compâ</td><td>Precâ</td><td>Recallâ</td><td>F-scoreâ</td><td>num_plane</td></tr><tr><td>0.95</td><td>0.0821</td><td>0.0861</td><td>0.5168</td><td>0.672</td><td>0.5842</td><td>35.57</td></tr><tr><td>0.98</td><td>0.0824</td><td>0.0855</td><td>0.5197</td><td>0.6738</td><td>0.5868</td><td>34.43</td></tr><tr><td>0.99</td><td>0.0831</td><td>0.0829</td><td>0.5214</td><td>0.6654</td><td>0.5846</td><td>31.29</td></tr></table>

Table 4: Ablation on the cosine similarity threshold $\alpha .$

<table><tr><td>Ï</td><td>Accâ</td><td>Compâ</td><td>Precâ</td><td>Recallâ</td><td>F-scoreâ</td><td>num_plane</td></tr><tr><td>100</td><td>0.0824</td><td>0.0855</td><td>0.5197</td><td>0.6738</td><td>0.5868</td><td>34.43</td></tr><tr><td>200</td><td>0.0824</td><td>0.0855</td><td>0.5197</td><td>0.6738</td><td>0.5868</td><td>34.43</td></tr><tr><td>500</td><td>0.0827</td><td>0.0874</td><td>0.5175</td><td>0.6699</td><td>0.5839</td><td>32.14</td></tr></table>

Table 5: Ablation on the minimum pixel number Ï of K-means clustering.

## B ALGORITHMIC ILLUSTRATION ON MESH LAYOUT REFINEMENT

Algorithm 2 Mesh Layout Refinement   
Require: Extracted mesh O, Initial sparse point cloud P cd, voxel size Î´, precomputed planar rela  
tionships $V _ { P } ^ { i } \in P c d$   
1: for each plane $A \in V _ { P }$ do   
2: for each vertex $v _ { x } \in O$ do   
3: $\mathbf { i f } \exists v _ { y } \in V _ { P } ^ { A } , | \bar { v } _ { y } - v _ { x } | < 1 . 5 \delta$ and $\forall \bar { v } _ { y } \notin V _ { P } ^ { A } , | \bar { v } _ { y } - v _ { x } | > 0$ .5Î´ then   
4: Assign $v _ { x }$ to plane A in $O \colon v _ { x }  \hat { V } _ { P } ^ { A } \in O$   
5: end if   
6: end for   
7: end for   
8: for each $\hat { V } _ { P } ^ { A }$ do   
9: Remove planar faces: $\{ f \in O | f = ( v _ { 1 } , v _ { 2 } , v _ { 3 } ) , v _ { 1 } , v _ { 2 } , v _ { 3 } \in \hat { V } _ { P } ^ { A } \}$   
10: Categorize vertices: Boundary $\hat { V } _ { B } ^ { A }$ , Interior $\hat { V } _ { I } ^ { A }$   
11: Project $\hat { V } _ { B } ^ { A } , \hat { V } _ { I } ^ { A }$ onto plane A: ${ \hat { V } } _ { B } ^ { \bar { A } } \to { \hat { V } } _ { B } ^ { a } , { \hat { V } } _ { I } ^ { \bar { A } } \to { \hat { V } } _ { I } ^ { a }$   
12: Compute bounding rectangle $R _ { A }$ covering $( \hat { V } _ { B } ^ { a } , \hat { V } _ { I } ^ { a } )$ and generate grid points $G _ { A }$ within $R _ { A }$   
13: Exclude $G _ { A }$ points outside the projected region $( \hat { V } _ { B } ^ { a } , \hat { V } _ { I } ^ { a } )$   
14: Perform Delaunay triangulation: TA = Delaunay $( V _ { B } ^ { A } \cup G _ { A } )$   
15: Map $T _ { A }$ and $G _ { A }$ back to 3D space   
16: Integrate $T _ { A } , G _ { A }$ into O   
17: end for   
18: return Refined mesh $O ^ { \prime }$

## C IMPLEMENTATION OF DYNAMIC GAUSSAIN RE-CLASSIFIER

This section we provide some implementation details of our Dynamic Gaussian Re-classifier (DGR). The DGR is designed to identify and reclassify Gaussians that are mistakenly regarded as planar Gaussians. According to the general design of Gaussian training process, the distribution of Gaussians will be densified from Iteration 500 to 15,000 in each 100 iteration, and the whole training process will end at Iteration 30,000. Our DGR phase will be operating for the latter 50 iterations between every densification step, and for 100 iterations at Iteration 20,000.

During the DGR phase, gradients of both planar Gaussians and non-planar Gaussians before finally proceeding to back-propagation will be stored and averaged for evaluation. The top 5% of the planar gradients are selected and compared with the average magnitude of top 20% non-planar gradients. Those with higher gradient magnitudes, the coordinates of their corresponding planar Gaussians will be re-formulated back to xyz format. The DGR design can correct those mistaken planar Gaussians, and it will not influence the training for non-planar Gaussians. Thus, even if the true-positive planar Gaussians are processed, they will still be supervised with the baseline design.

## D ADDITIONAL QUALITATIVE RESULTS

In this section we first provide additional qualitative results on the overall reconstruction of the mesh in Fig. 7. We also provide examples in both rendering effects of GSPlane and baseline methods in Novel View Synthesis. The visualizations are shown in Fig. 8. According to the quantitative results in Tab. 1, GSPlane also provides comparable results with small improvements, up to 0.018 and 1.02 for GOF in the SSIM and PSNR, respectively.

<!-- image-->  
Figure 7: Visualization of reconstructed mesh performance.

## E VISUALIZATION OF PLANAR PRIOR EXTRACTION AND PERFORMANCE

In this section, we provide visualizations starting from 2D planar prior to the final refinement results in Fig.9. Before training, we first establish planar priors by aggregating both subparts proposals from SAM Kirillov et al. (2023) and normal maps from Metric3Dv2 Hu et al. (2024). After structured representation for 3D planes are established, given a unrefined mesh with densely distributed vertices, GSPlane can create refined planar regions that exhibit consistent normals and topology, along with unified edges and a reduced number of vertices and faces, resulting in a more efficient and structured representation.

<!-- image-->  
Figure 8: Visualization of NVS results.

<!-- image-->  
Figure 9: Visualization of an example from kitchen corner. The left shows the normal map and aggregated planar mask proposals of 2D views. The middle and right of the figure are the target mesh before & after the layout refinement via structured representation of planes.

## F LIMITATION

Though GSPlane is able to provide concise and accurate geometry with satisfied topology and unified normal in planar region, there are still some issues before acquiring a desired and satisfied scene mesh. Currently, our focus is on planar regions, and the structured representation of non-planar regions remains an open challenge, which we leave as future work. A possible direction for addressing this issue could involve developing alternative representations tailored to complex surfaces. Additionally, the accuracy of planar priors are constrained by foundation models of masks and normals.

## G LARGE LANGUAGE MODEL USAGE

Large Language Models (LLMs) are used for polishing writing in this manuscript. The prompt is used as follows:

Assume you are a native English speaker, a senior researcher in the area of computer vision and graphics. Please help me polish the following content: