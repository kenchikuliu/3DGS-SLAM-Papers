# DynamicGSG: Dynamic 3D Gaussian Scene Graphs for Environment Adaptation

Luzhou Ge1â, Xiangyu Zhu1â, Zhuo Yang1 and Xuesong Li1,â 

Abstractâ In real-world scenarios, environment changes caused by human or agent activities make it extremely challenging for robots to perform various long-term tasks. Recent works typically struggle to effectively understand and adapt to dynamic environments due to the inability to update their environment representations in memory according to environment changes and lack of fine-grained reconstruction of the environments. To address these challenges, we propose DynamicGSG, a dynamic, high-fidelity, openvocabulary scene graph construction system leveraging Gaussian splatting. DynamicGSG builds hierarchical scene graphs using advanced vision language models to represent the spatial and semantic relationships between objects in the environments, utilizes a joint feature loss we designed to supervise Gaussian instance grouping while optimizing the Gaussian maps, and locally updates the Gaussian scene graphs according to real environment changes for long-term environment adaptation. Experiments and ablation studies demonstrate the performance and efficacy of our proposed method in terms of semantic segmentation, language-guided object retrieval, and reconstruction quality. Furthermore, we validate the dynamic updating capabilities of our system in real laboratory environments. The source code and supplementary experimental materials will be released at: https://github.com/GeLuzhou/Dynamic-GSG.

## I. INTRODUCTION

Future intelligent robots are supposed to execute diverse long-term complex tasks in dynamic environments based on intricate natural language instructions from humans. To achieve this, agents must possess dynamic environment perception and comprehension capabilities. Prior studies [1], [2], [3], [4] construct static open-vocabulary scene graphs from sensor data to capture the topological structures of environments at specific moments, which enhances robotsâ understanding of complex instructions and facilitates task completion. However, static scene graphs are of limited use in real-world scenarios, as robot workspaces typically change due to human activities or other agentsâ operations. Additionally, the inherent latency between the scene graph stored in memory (which guides task planning and execution) and the actual environment state significantly hinders successful task completion.

Recent advancements in 3D Gaussian splatting [5], [6], [7], [8] have attracted significant attention within the robotics community. These developments have applications in various domains, including high-quality reconstruction [5], [6], [9], [10] robotic manipulation [11], semantic embedding [12], [13], and 3D environment understanding [14], [15].

<!-- image-->  
Fig. 1: The dynamic high-fidelity multi-layer Gaussian scene graphs we constructed can adapt to environment changes, represent the spatial and semantic relationships of the objects, and support various forms of language-guided object retrieval.

The explicit point-based representation of 3D Gaussians effectively facilitates semantic information integration from advanced vision language models, enabling the construction of topological scene representations. Based on fast differentiable rendering and explicit representation, we find 3D Gaussians are particularly suitable for locally rapid updates of reconstructed scenes. Moreover, the high-fidelity environment reconstruction provided by 3D Gaussians naturally supports the development of dynamic, high-quality scene graph construction systems.

Most previous works on scene graph construction primarily rely on point clouds [1], [2], [4], [16], [17] These methods often struggle to promptly respond to dynamic environment changes and fail to capture fine-grained details of 3D scenes due to the inherent limitations of traditional representation techniques.

In this paper, we propose DynamicGSG, a framework that utilizes 3D Gaussian Splatting to construct dynamic, high-fidelity, open-vocabulary scene graphs. Advanced vision models such as Yolo-World [18], Segment Anything [19], CLIP [20] are employed to detect objects and extract their semantic features. Subsequently, we analyze the spatial and semantic relationships of objects to build hierarchical scene graphs using Large Vision Language Model (LVLM). Through incorporating additional semantic supervision, we improve the accuracy of instance-level

Gaussian grouping and the overall reconstruction quality. Moreover, the rapid training and differentiable rendering of 3D Gaussians facilitate efficient scene updates to accommodate environment changes.

In summary, our contributions are as follows:

â¢ We propose Dynamic 3D Gaussian Scene Graphs, combining instance-level rendering with VLM semantic information to achieve 3D-2D object association and building multi-layer scene graphs with LVLM.

â¢ We design a joint loss function that ensures accurate intra-instance Gaussian grouping and high-fidelity scene reconstruction.

â¢ We utilize the fast differentiable rendering of Gaussians to update the scene graphs, enabling our system to adapt to dynamic environment changes.

â¢ We deployed DynamicGSG in real-lab environments, demonstrating its capability to construct 3D Gaussian scene graphs and perform dynamic updates for effective environment adaptation.

## II. RELATED WORK

## A. 3D Scene Graphs

3D scene graph [21], [17] offers a hierarchical graphstructured representation of the environment with nodes representing spatial concepts at multiple levels of abstraction (e.g., objects, places, rooms, buildings.) and edges preserving the spatial and semantic relationships between nodes. Taking advantage of the generalization abilities of vision foundation models [18], [19], [22], [23] and cross-modal grounding capabilities of vision-language models [20], [24] enables the construction of 3D scene graphs at the open-vocabulary level. Recent methods [1], [2], [4], [3], [25] construct 3D scene graphs embedded with VLM semantic features, facilitating object retrieval, robot manipulation and navigation.

However, most of these methods [1], [2], [4] rely on static environment assumptions and typically lack mechanisms for handling dynamic updates. RoboEXP [25] conducts dynamic scene updates based on object spatial relationships and the scene modifications resulting from robot operations. DovSG [3] extends dynamic scene graphs construction to mobile agents and utilizes Large Language Models (LLMs) [26] for task decomposition and planning, enabling robots to accomplish complex tasks in dynamic environments over the long term. To enhance computational efficiency, these methods often employ aggressive point cloud downsampling strategies, which significantly limits their capacity for highfidelity geometric reconstruction of scene details.

## B. Gaussian-based Open-Vocabulary Scene Understanding

Recent progress in Gaussian Splatting [5], [6], [7], [27], [8] demonstrates outstanding performance in photo-realistic reconstruction with promising efficiency. The gradient-based optimization has also been applied in an online setting to incrementally construct the Gaussian map [9], [10], [28], [29] through differentiable rendering.

Concurrently, advancements in vision foundation models (such as SAM [19], CLIP [20], DINO [30]) have motivated the exploration of integrating 2D semantic features into 3D Gaussians. LangSplat [14] learns hierarchical semantics using SAM and trains an autoencoder to distill highdimensional CLIP features into low-dimensional semantic attributes of Gaussians. GaussianGrouping [12] implements joint 3D Gaussian optimization based on 2D pre-matched SAM masks and 3D spatial consistency, enabling highquality scene reconstruction and open-vocabulary object segmentation. OpenGaussian [15] augments 3D Gaussian Splatting with point-level open-vocabulary understanding through SAM-based instance feature training, a twolevel codebook for discretization, and instance-level 3D-2D feature association for geometric-semantic alignment. However, Most of the above works require embedding semantic information into pre-trained Gaussian scenes or conducting data preprocess prior to gradient-based optimization. Such offline pipelines inherently conflict with robotsâ operational paradigms in dynamic settings.

To address these deficiencies, we incrementally construct the semantic Gaussian map using posed RGB-D sequences from public datasets or live cameras running Visual-Inertial Odometry (VIO) [31], [32], [33], [34] frameworks, enabling it to achieve high-fidelity reconstruction and integrate rich semantic information to build a topological scene graph. Leveraging the rapid training and differentiable rendering capabilities of 3D Gaussians, our system can dynamically update both the Gaussian maps and scene graphs to adapt to changes in the real-world environments.

## III. METHODOLOGY

The proposed system constructs dynamic high-fidelity Gaussian scene graphs by splatting Gaussians onto 2D image planes in various forms, building semantic scene graphs, optimizing Gaussian maps, and making dynamic updates. An overview of DynamicGSG is presented in Fig. 2.

## A. Gaussian Preliminaries

We densely represent the scene using isotropic Gaussian, which is an explicit representation parameterized by RGB color c, center position $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , radius $\mathbf { r } \in \mathbb { R } ^ { + }$ , and opacity $\mathbf { o } \in [ 0 , 1 ]$ . The influence of each Gaussian on 3D space point $\mathbf { x } \in \mathbb { R } ^ { 3 }$ is defined as:

$$
f ( \mathbf { x } ) = \mathbf { o } \cdot \exp \left( { \frac { \| \mathbf { x } - { \pmb { \mu } } \| ^ { 2 } } { 2 \mathbf { r } ^ { 2 } } } \right)\tag{1}
$$

The view synthesis and Gaussian parameters optimization are implemented through differentiable rendering with the Gaussian map and a camera pose $T \in S E { ( 3 ) }$ . The color, depth, and visibility (accumulated opacity) of each pixel u at camera pose $T _ { t }$ is determined by Î±-blending contributions from depth-ordered projections of 3D Gaussians:

$$
\hat { C } _ { t } ( \mathbf { u } ) = \sum _ { i = 1 } ^ { n } \mathbf { c } _ { i } f _ { i } ( \mathbf { u } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \mathbf { u } ) \right) .\tag{2}
$$

$$
{ { \hat { D } } _ { t } } ( { \mathbf { u } } ) = \sum _ { i = 1 } ^ { n } { { d _ { i } } { f _ { i } } ( \mathbf { u } ) \prod _ { j = 1 } ^ { i - 1 } { { ( 1 - { f _ { j } } ( \mathbf { u } ) ) } } } ,\tag{3}
$$

<!-- image-->  
Fig. 2: Overview of DynamicGSG: Our system processes posed RGB-D sequences, utilizes open-vocabulary object detection and segmentation models to obtain 2D masks, and extracts corresponding semantic features. In parallel, we employ instance-level rendering to get 2D masks and semantic features of objects in the map for object fusion. Subsequently, we perform Gaussian initialization and joint optimization to incrementally create a high-fidelity object-centric Gaussian map. Based on the spatial relationship of objects and the capabilities of LVLM, we construct a hierarchical scene graph to provide a structured description of the scene. In dynamic realworld scenarios, after refining the initial camera poses obtained from VINS-Fusion[33], we detect local changes and make corresponding modifications in the Gaussian map and scene graph for environment adaptation.

$$
\hat { S } _ { t } ( \mathbf { u } ) = \sum _ { i = 1 } ^ { n } f _ { i } ( \mathbf { u } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \mathbf { u } ) \right) ,\tag{4}
$$

where $d _ { i }$ is the depth of the i-th Gaussian center in the camera coordinate.

We preserve the basic properties of Gaussians while introducing two additional parameters: a low-dimensional instance feature $\mathbf { e _ { \lambda } \in \mathbb { R } ^ { 3 } }$ and an identifier idx $\in \mathbb { N } ^ { + }$ . We render the 2D instance feature $\hat { E } \in \mathbb { R } ^ { 3 \times H \times W }$ for each pixel in a differentiable manner similar to color blending:

$$
\hat { E } _ { t } ( \mathbf { u } ) = \sum _ { i = 1 } ^ { n } \mathbf { e } _ { i } f _ { i } ( \mathbf { u } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \mathbf { u } ) \right) .\tag{5}
$$

## B. Gaussian Objects Association

Our system processes posed RGB-D sequences $\begin{array} { r l } { I _ { t } } & { { } = } \end{array}$ $\langle C _ { t } , D _ { t } , T _ { t } \rangle \in \{ I _ { 1 } , I _ { 2 } , \ldots , I _ { t } \}$ to construct a Gaussian scene graph: $\textbf { G } = \{ O _ { t } , E _ { t } \}$ , where $O _ { t } ~ = ~ \{ o _ { j } \} _ { j = 1 } ^ { J }$ and $E _ { t } ~ =$ $\{ e _ { k } \} _ { k = 1 } ^ { K }$ represent the sets of objects and edges, respectively. Each object $o _ { j }$ is characterized by a set of Gaussians indexed by idx and a semantic feature vector $f _ { o _ { j } }$

a) Object Recognition: We employ various advanced vision models to obtain instance-level semantic information from RGB images. For the current frame $I _ { t } ,$ , we first use an open-vocabulary object detection model (YOLO-World [18]) to obtain object bounding boxes $\{ \mathbf { b } _ { t , i } \} _ { i = 1 } ^ { M }$ . These detected proposals are subsequently processed by Segment Anything [19] to generate corresponding masks $\{ \mathbf { m } _ { t , i } \} _ { i = 1 } ^ { M }$ Post-processing is then performed to ensure these masks do not overlap with each other. Finally, we use CLIP [20] to obtain each detectionâs semantic feature. Since CLIPâs visual descriptors are image-aligned, each mask $\mathbf { m } _ { t , i }$ is passed to CLIP to extract an instance-level semantic feature $f _ { t , i }$

<!-- image-->  
Fig. 3: 3D-2D Gaussian Object Association.

b) 3D-2D Object Association: For all detections $D _ { t } =$ $\{ \mathbf { d } _ { t , i } \} _ { i = 1 } ^ { M }$ in $I _ { t } ,$ , we associate them with the map objects $O _ { t - 1 }$ through joint geometric and semantic similarity matching. As shown in we first render all Gaussians into the current camera view at instance-level to obtain object masks $\{ \mathbf { m } _ { t , o _ { j } } \} _ { j = 1 } ^ { J }$ Objects $O _ { t - 1 }$ whose projected masks contain more than $ { \delta _ { \mathrm { p i x } } }$ pixels in the image plane are considered visible in the current frame. The geometric similarity is then formulated by the Intersection over Union (IoU) between all 2D detection masks $\{ \mathbf { m } _ { t , i } \} _ { i = 1 } ^ { M }$ and projected masks of visible 3D objects

$$
\{ \mathbf { m } _ { t , o _ { j } } \} _ { j = 1 } ^ { J } \mathrm { : }
$$

$$
s _ { \mathrm { g e o } } ( i , j ) = \frac { m _ { t , i } \cap m _ { t , o _ { j } } } { m _ { t , i } \cup m _ { t , o _ { j } } }\tag{6}
$$

The semantic similarity is calculated as the normalized cosine similarity between detection CLIP descriptors $\{ f _ { t , i } \} _ { i = 1 } ^ { M }$ and visible object semantic features $\{ f _ { o _ { j } } \} _ { j = 1 } ^ { J }$ . The joint similarity $s ( i , j )$ combines geometric and semantic similarity through weighted summation:

$$
s _ { \mathrm { s e m } } ( i , j ) = ( f _ { t , i } ) ^ { \top } f _ { o _ { j } } / 2 + 1 / 2\tag{7}
$$

$$
s ( i , j ) = w _ { g } s _ { \mathrm { g e o } } ( i , j ) + w _ { s } s _ { \mathrm { s e m } } ( i , j )\tag{8}
$$

where $w _ { g } + w _ { s } = 1$ . Object association is performed through a greedy algorithm that assigns each detection to a visible object with the maximum similarity score. A new object is initialized when no visible object with similarity exceeding the threshold $\delta _ { s i m }$ is matched.

c) Object Fusion: If a detection $d _ { t , i }$ is associated with a map object $o _ { j }$ , we fuse this detection with the corresponding object by updating the objectâs semantic feature as:

$$
f _ { o _ { j } } = ( n _ { o _ { j } } f _ { o _ { j } } + f _ { t , i } ) / ( n _ { o _ { j } } + 1 )\tag{9}
$$

where $n _ { o _ { j } }$ denotes the number of $o _ { j }$ that has been associated.

## C. Gaussian Map Optimization

To incrementally construct the object-centric Gaussian map, new Gaussians must be initialized and subsequently optimized with observations to ensure our map is realistic and accurately group Gaussians belonging to each object.

a) Densification based on opacity: Firstly, new Gaussians will be dynamically initialized to cover newly observed regions and objects. We create a newly observed mask following [9], which identifies pixels exhibiting either insufficient transparency accumulation or emerging geometry occluding the existing scene structure:

$$
M _ { t } = ( \hat { S } _ { t } < \lambda _ { s } ) \cup ( ( D _ { t } < \hat { D } _ { t } ) ( | D _ { t } - \hat { D } _ { t } | ) > \lambda _ { \mathrm { M D E } } )\tag{10}
$$

where $\lambda _ { s } = 0 . 5 , \lambda _ { M D E }$ equals 50 times median depth error. For each pixel in $I _ { t } ,$ we add a new Gaussian characterized by the pixelâs color, aligned depth, and an opacity of 0.5. Each new Gaussianâs idx is assigned by the objectâs index after object association, and its instance feature e is initialized using the value corresponding to idx in the codebook, which contains 200 instance colors from ScanNet dataset [35]. As illustrated in the lower-right corner of Fig. 2, these instance features are then projected onto the 2D mask regions where the corresponding objects is located in $C _ { t }$ , forming a groundtruth instance feature $E _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$

b) Intra-instance Gaussian Regularization: During gradient-based optimization, each attribute of Gaussians will be refined according to differentiable rendering. As shown in Fig. 4, we observe that Gaussians initially grouped by object indexes tend to progressively encroach into regions occupied by other objects, resulting in segmentation artifacts and reduced boundary precision. To address this limitation, we propose a novel feature consistency loss with GT instance feature $E _ { t }$ to improve reconstruction fidelity and instancelevel grouping accuracy:

$$
{ L } _ { \mathrm { f e a t u r e } } = \lambda _ { 1 } | E _ { t } - \hat { E } _ { t } | + \lambda _ { 2 } ( 1 - S S I M ( E _ { t } , \hat { E } _ { t } ) )\tag{11}
$$

where $\lambda _ { 1 } ~ = ~ 0 . 8 , \lambda _ { 2 } ~ = ~ 0 . 2$ . As demonstrated in Fig. 4, our feature loss enforces intra-instance Gaussian consistency while boosting semantic segmentation precision.

c) Gaussian map optimization: We minimize the joint loss, including color, depth, and features, to optimize the Gaussian map:

$$
{ \cal L } _ { \mathrm { c o l o r } } = \lambda _ { 3 } | C _ { t } - \hat { C } _ { t } | + \lambda _ { 4 } ( 1 - S S I M ( C _ { t } , \hat { C } _ { t } ) )\tag{12}
$$

$$
L _ { \mathrm { d e p t h } } = | D _ { t } - \hat { D } _ { t } |\tag{13}
$$

$$
L _ { \mathrm { m a p p i n g } } = w _ { c } L _ { \mathrm { c o l o r } } + w _ { d } L _ { \mathrm { d e p t h } } + w _ { f } L _ { \mathrm { f e a t u r e } }\tag{14}
$$

where $\lambda _ { 3 } = 0 . 8 , \lambda _ { 4 } = 0 . 2 , w _ { c } = w _ { f } = 0 . 5 , w _ { d } = 1 . 0 .$

In the process of optimizing the Gaussian map, we perform multi-view scene optimization by collecting a list of keyframes to improve the quality of 3D reconstruction. A keyframe is selected and stored every n-th frame, and m keyframes are chosen for multi-view optimization based on temporal distance and geometric constraints. Furthermore, we prune redundant Gaussians with near-zero opacity or large covariances as [5].

## D. Multi-layer Scene Graph Construction

With the well-reconstructed object-centric Gaussian map, we construct a hierarchical scene graph $\mathbf { G } _ { t } = \{ O _ { t } , E _ { t } \}$ that reflects the structured description of the environment. As illustrated on the right side of Fig. 2, objects are categorized into asset, ordinary, and standalone objects.

a) Asset objects: We use a Large Vision Language Model (GPT-4o) to identify asset objects in indoor environments. The expected asset objects typically include furniture on the ground, such as chairs, tables, or cabinets, rather than small portable containers or decorative items (e.g., baskets, vases, or lamps). Our system takes the labels generated by Yolo-World[18] for objects in $O _ { t }$ as input to GPT-4o. The asset objects, denoted as $\tilde { O } _ { t } \subseteq O _ { t }$ , are classified through a specific prompt.

b) Ordinary objects: We define ordinary objects $\bar { O } _ { t }$ through spatial relationships between an object and asset objects ${ \tilde { O } } _ { t }$ . If the center position of an object is located above the spatial range of the asset object $\tilde { o } _ { j } ~ \in ~ \tilde { O } _ { t }$ , we determine this object as $\bar { o } _ { i } ~ \in ~ \bar { O } _ { t }$ and establish an edge $e _ { t } ( i , j ) \in E _ { t }$ between them. This edge not only represents the spatial relationship but also indicates that $\bar { o } _ { i }$ is carried by ${ \tilde { o } } _ { j }$

c) Standalone objects: Standalone objects are defined as the complement of asset and ordinary objects, represented as $\begin{array} { r c l } { \hat { O } _ { t } } & { \subseteq } & { ( O _ { t } \ - \ \tilde { O } _ { t } \ - \ \bar { O } _ { t } ) } \end{array}$ Typical instances include stools, windows, and similar entities that exhibit inherent independence in operational environments.

Additionally, asset and standalone objects are directly connected to the room node within the scene graph $\mathbf { G } _ { t }$

## E. Dynamic Scene Update

In dynamic human-robot collaborative environments, the changes in object positions and their spatial interrelationships present significant challenges for long-term task execution. Leveraging high-fidelity reconstruction and fast differentiable rendering of 3D Gaussian, we implement a dynamic update mechanism to detect changes and locally update the Gaussian map and scene graph at the instance level with real-time RGB-D observations, ensuring temporal consistency between the environment representations in the robotâs memory and the physical workspace, as shown in Fig. 1 and 6.

a) Refine Camera Tracking: In public datasets or simulation environments, our system can directly utilize RGB-D frames with ground-truth poses as input. In realworld scenarios, the lack of reliable pose priors introduces fundamental constraints on the construction and modification of Gaussian scene graphs. To address this sensing gap, we deploy a validated and efficient VIO[31] framework to acquire an initial estimate $T _ { t , e s t }$ and execute an iterative pose refinement by minimizing the L1 loss between aligned color $C _ { t }$ and depth $D _ { t }$ from the camera and their rendered views to obtain precise camera pose $T _ { t }$

$$
L _ { \mathrm { t r a c k i n g } } = ( \hat { S } _ { t } > \lambda _ { r } ) ( \lambda _ { 5 } | C _ { t } - \hat { C } _ { t } | + \lambda _ { 6 } | D _ { t } - \hat { D } _ { t } | )\tag{15}
$$

where $\lambda _ { r } = 0 . 9 9 , \lambda _ { 5 } = 0 . 5 , \lambda _ { 6 } = 1 . 0 .$ . During this stage, the parameters of the Gaussians are fixed.

b) Update Local Gaussian Map: Using the refined pose $T _ { t } ,$ , our system begins to detect local changes, including the disappearance, displacement, and emergence of objects. Subsequently, we will modify the map and alter the scene graph at the instance level.

To address the disappearance and movement of objects, we render all objects where over 50% of their Gaussians appear within the current camera frustum to acquire visible object RGB masks $\{ m _ { t , o _ { j } } ^ { r g b } \} _ { j = 1 } ^ { J }$ and compute structural similarity (SSIM) between $\{ m _ { t , o _ { j } } ^ { r g b } \} _ { j = 1 } ^ { J }$ and the corresponding regions within RGB observation $\operatorname { \bar { \it C } } _ { t } \mathrm { : }$

$$
S ( j ) = S S I M ( m _ { t , o _ { j } } ^ { r g b } , C _ { t } ( m _ { t , o _ { j } } ) )\tag{16}
$$

If $\begin{array} { r l r } { S ( j ) } & { { } < } & { \delta _ { \mathrm { c h a n g e } } . } \end{array}$ , it indicates that object $o _ { j }$ has either been moved or substituted by another object. We categorize all disappeared objects as $O _ { \mathrm { d e l e t e } }$ and remove their associated obsolete Gaussians. Next, new objects $O _ { \mathrm { a p p e a r } }$ will be instantiated following Sec. III-B, based on the recalculated similarity matrix $s ( i , j )$ between $\{ \mathbf { d } _ { t , i } \} _ { i = 1 } ^ { M }$ and $\{ \mathbf { 0 } _ { t - 1 , j } \} _ { j = 1 } ^ { J ^ { \prime } } \subseteq \{ O _ { t - 1 } - O _ { \mathrm { d e l e t e } } \}$ . The union of deleted and new objects constitutes the update set $\begin{array} { r l } { O _ { \mathrm { u p d a t e } } } & { { } = } \end{array}$ $\{ O _ { \mathrm { a p p e a r } } , O _ { \mathrm { d e l e t e } } \}$

Finally, if $O _ { \mathrm { u p d a t e } } \neq \emptyset$ , our system will clear the keyframe list to prevent outdated keyframes from participating in the subsequent Gaussian map optimization detailed in Sec. III-C. Otherwise, we shall skip this stage.

c) Update Scene graph: Following the local updates of the Gaussian map, the scene graph $\mathbf { G } _ { t - 1 }$ requires corresponding adjustments based on the categories of objects in $O _ { \mathrm { u p d a t e } }$ . For the deleted objects $O _ { \mathrm { d e l e t e } }$ , if an object is marked as ordinary or standalone, it is sufficient to delete it along with the parent edge $e _ { k }$ from $\mathbf { G } _ { t - 1 } ;$ in contrast, if the object is an asset object, the removal must extend to all its child nodes and their associated edges. For the newly appeared objects $O _ { \mathrm { a p p e a r } } ,$ we determine their type using the method described in Sec. III-D. Subsequently, we insert nodes of corresponding types and establish relational edges within $\mathbf { G } _ { t - 1 }$ to construct the updated scene graph $\mathbf { G } _ { t }$

<table><tr><td>Methods</td><td colspan="3">Metrics</td></tr><tr><td></td><td>mAccâ</td><td>mIoUâ</td><td>F-mIoUâ</td></tr><tr><td>ConceptGraphs [1]</td><td>39.43</td><td>25.57</td><td>44.06</td></tr><tr><td>ConceptGraphs-Detector [1]</td><td>41.18</td><td>26.82</td><td>42.28</td></tr><tr><td>HOV-SG [2]</td><td>39.95</td><td>27.52</td><td>46.79</td></tr><tr><td>DynamicGSG</td><td>54.04</td><td>31.06</td><td>46.21</td></tr><tr><td>DynamicGSG w/o feature loss</td><td>52.94</td><td>25.97</td><td>37.32</td></tr></table>

TABLE I: 3D Open-vocabulary Semantic Segmentation on Replica [36]: Attributed to 3D-2D Gaussian Object Association, DynamicGSG significantly outperforms the baselines in terms of both mAcc and mIOU. And the joint feature loss also effectively improve the mIOU and F-mIOU of semantic segmentation.  
<!-- image-->  
Fig. 4: Visualization of Feature Loss Ablation Experiments.

## IV. EXPERIMENT

## A. Experiment Setups

To comprehensively evaluate DynamicGSG, we conduct a series of experiments using data sourced from Replica[36], ScanNet++[37], and real laboratory environments: (1) A quantitative comparison of 3D open-vocabulary semantic segmentation on the Replica dataset, contrasting our results with recent open-vocabulary scene graph construction methods, accompanied by an ablation study to investigate the contribution of joint feature loss. (2) A language-guided object retrieval experiment to evaluate the effectiveness of multi-layer scene graphs generated by DynamicGSG in capturing spatial-semantic object relationships. (3) Quantitative evaluation of scene reconstruction quality on Replica and ScanNet++ datasets. (4) Within our laboratory, we manually introduce environment changes to validate DynamicGSGâs capability for dynamic updating of Gaussian scene graphs.

All experiments are conducted on a desktop computer equipped with an Intel Core i7-14700KF CPU, an NVIDIA RTX 4090D GPU, and 32GB RAM. In all experiments, we set thresholds $\delta _ { \mathrm { p i x } } = 2 0 0 , \delta _ { \mathrm { s i m } } = 0 . 5 5$ and $\delta _ { \mathrm { c h a n g e } } = 0 . 1 5$

<table><tr><td>Methods</td><td>Query</td><td>Match</td><td>R@1</td><td>R@2</td><td>R@3</td></tr><tr><td rowspan="3">ConceptGraphs [1]</td><td>Descriptive</td><td>CLIP LLM HSG</td><td>0.52 0.40</td><td>0.65 0.55</td><td>0.71 0.62</td></tr><tr><td>Affordance</td><td>CLIP LLM</td><td>0.60 0.60</td><td>0.63 0.69</td><td>0.69</td></tr><tr><td>Negation</td><td>CLIP LLM</td><td>0.17 0.77</td><td>0.49 0.91</td><td>0.80 0.60 0.97</td></tr><tr><td></td><td>Descriptive</td><td>CLIP LLM</td><td>0.64 0.41</td><td>0.74 0.57</td><td>0.76</td></tr><tr><td rowspan="3">DynamicGSG</td><td></td><td>HSG</td><td>0.71</td><td>0.81</td><td>0.64 0.82</td></tr><tr><td>Affordance</td><td>CLIP LLM</td><td>0.60 0.65</td><td>0.66 0.74</td><td>0.66 0.77</td></tr><tr><td>Negation</td><td>CLIP LLM</td><td>0.34 0.77</td><td>0.54 0.89</td><td>0.71 0.94</td></tr></table>

TABLE II: Language-guided Object retrieval on Replica[36]. CLIP, LLM, and HSG refer to Semantic-based match, LLM-based match, and Hierarchical scene graph-based match, respectively.

(2) This is a coiled cord on the table.  
<!-- image-->  
Fig. 5: Qualitative Results of Object Retrieval: DynamicGSG effectively locates objects that ConceptGraphs [1] cannot retrieve through Hierarchical scene graph-based match.

## B. 3D Open-vocabulary Semantic Segmentation

To evaluate the quality of semantic embeddings in DynamicGSG and investigate how joint feature loss supervision influences Gaussian instance grouping, we perform an ablation experiment of the 3D open-vocabulary semantic segmentation on 8 scenes from the Replica dataset [36] and quantitatively compare our results with recent scene graph construction methods. The primary baseline methods used for comparison are ConceptGraphs [1] and HOV-SG [2]. For the ablation analysis, we include a variant of DynamicGSG without feature loss. All compared methods consistently adopt the ViT-H-14 CLIP backbone for semantic feature extraction.

To generate the semantic segmentation, we first calculate the CLIP text description vector for class-specific prompt formatted as âan image of {class label}â corresponding to each class in the Replica dataset. For each scene, we compute the cosine similarity between the semantic feature of each object within the scene graph and the text description vector of each class. Each objectâs points or Gaussians are allocated to the class with the highest similarity score. Finally, the point clouds or Gaussians generated by all methods are transformed to the same coordinate as the ground-truth semantic point clouds. Quantitative evaluation is performed through standardized metrics, including mAcc, mIoU, and

<table><tr><td rowspan="2">Methods</td><td colspan="3">Metrics</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>HOV-SG [2]</td><td>19.69</td><td>0.821</td><td>0.284</td></tr><tr><td>ConceptGraphs [1]</td><td>23.24</td><td>0.910</td><td>0.204</td></tr><tr><td>SGS-SLAM [13]</td><td>35.43</td><td>0.978</td><td>0.077</td></tr><tr><td>DynamicGSG</td><td>35.62</td><td>0.979</td><td>0.068</td></tr></table>

TABLE III: Quantitative Reconstruction Performance on Replica [36]: DynamicGSG is comparable to the Gaussian-based semantic SLAM method [13] while significantly outperforming methods based on point cloud [1], [2].

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Metrics</td><td colspan="3">Train View</td><td colspan="3">Novel View</td></tr><tr><td>S1</td><td>S2</td><td>Avg.</td><td>S1</td><td>S2</td><td>Avg.</td></tr><tr><td rowspan="3">SplaTAM [9]</td><td>PSNR â</td><td>27.78</td><td>28.40</td><td>28.09</td><td>24.50</td><td>25.56</td><td>25.03</td></tr><tr><td>SSIM â</td><td>0.946</td><td>0.944</td><td>0.945</td><td>0.896</td><td>0.892</td><td>0.894</td></tr><tr><td>LPIPS â</td><td>0.121</td><td>0.129</td><td>0.125</td><td>0.210</td><td>0.255</td><td>0.233</td></tr><tr><td rowspan="3">DynamicGSG</td><td>PSNR â</td><td>27.86</td><td>28.47</td><td>28.17</td><td>24.81</td><td>25.65</td><td>25.19</td></tr><tr><td>SSIM â</td><td>0.945</td><td>0.946</td><td>0.946</td><td>0.902</td><td>0.893</td><td>0.898</td></tr><tr><td>LPIPS â</td><td>0.120</td><td>0.125</td><td>0.123</td><td>0.202</td><td>0.247</td><td>0.225</td></tr></table>

TABLE IV: Novel & Train View Synthesis Performance on ScanNet++ [37]: DynamicGSG not only provides photorealistic reconstruction on training views but also enables high-fidelity novel view synthesis at any camera pose.

frequency-weighted mIoU.

As shown in Tab. I, our method performs better than all baselines on mAcc and mIoU while achieving comparable performance to HOV-SG on F-mIoU. Object association in [1], [2] relies on the overlap ratio between point clouds which suffers from a critical limitation: the potential association of small objects into nearby large objects due to high overlap ratios of 3D point clouds. Our method employs 3D-2D object association which effectively prevents spurious merging, as the small masks and large objects do not exhibit abnormal geometric similarity, ultimately yielding a significant enhancement in mAcc. And the joint feature loss also effectively regularizes the Gaussian instance grouping to improve the mIOU and F-mIOU. The qualitative results in Fig. 4 further demonstrate that joint feature loss significantly enhances the regularization of intra-instance Gaussian grouping.

## C. Language-guided Object Retrieval

To validate whether the multi-layer scene graphs constructed by DynamicGSG can effectively capture spatial and semantic object relationships, we conduct a languageguided object retrieval experiment employing diverse query types across three semantic complexity levels (Descriptive: E.g., âThese are some books and they are on the table.â; Affordance: E.g., âSomething I can open with my keys.â; Negation: E.g., âSomething to sit on other than a chair.â) provided by ConceptGraphs [1] on the Replica dataset [36]. Following ConceptGraphs, we select 20 Descriptive, 5 Affordance and Negation queries for each scene, ensuring each query corresponds to at least one ground-truth object.

We employ three distinct object retrieval methods: (1) Semantic-based match: Objects in the scene graph are

Before Update

After Update

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
(c)

<!-- image-->  
(d)  
Fig. 6: Visualization of Dynamic Updates: (a) The backpack on the sofa and the bin are removed. (b) Holistic position update of the table and its contents (books and bottle). (c) The books and bottle exchange positions, and the backpack is moved to the chair. (d) The teacup disappears from the tea table and some fruits appear on the computer table.

selected based on the cosine similarity between their semantic features and the CLIP text embedding of the query. The object exhibiting the highest similarity is chosen. (2) LLM-based match: LLM (GPT-4o) is utilized to identify the object node within the scene graph that optimally corresponds to the query statement. (3) Hierarchical scene graph-based match: For descriptive queries incorporating inter-object relationships, DynamicGSG leverages multilayer scene graphs and object semantic features to perform hierarchical matching.

Analysis of top-1, top-2 and top-3 recall across diverse query types in Tab. II indicates that hierarchical matching substantially enhances accuracy for descriptive queries involving inter-object relationships, exhibiting a significant improvement over ConceptGraphs [1]. This performance improvement is qualitatively illustrated in Fig. 5, where our hierarchical matching approach successfully localizes target objects that baseline fails to identify. Futhermore, LLMs demonstrate superior instruction comprehension for Affordance and Negation queries, which more closely approximate natural human language. DynamicGSG also facilitates earlier object localization within these queries.

## D. Scene Reconstruction Quality

To evaluate reconstruction fidelity of DynamicGSG relative to recent scene graph construction methods, we establish an evaluation protocol across three metrics (PSNR, SSIM, and LPIPS) following SplaTAM [9]. For a fair comparison, all methods utilize the ground-truth camera poses provided by the datasets.

The quantitative comparisons on the Replica dataset [36], presented in Table III, demonstrate that our method significantly outperforms point cloud based methods [1], [2] while achieving comparable results to the advanced Gaussian semantic SLAM method SGS-SLAM [13], which utilizes ground-truth semantic annotations during optimization. To further validate the reconstruction quality of DynamicGSG, we extend our evaluation to novel view synthesis with SplaTAM on two scenes (8b5caf3398, b20a261fdf) from the ScanNet++ dataset [37]. The results in Tab. IV show that our method marginally outperforms SplaTAM in challenging scenarios.

<table><tr><td>Types of Change</td><td>Success Rate (%)</td></tr><tr><td>Object Disappearance</td><td>27 / 30 90.0</td></tr><tr><td>Object Relocation 25 / 30</td><td>83.3</td></tr><tr><td>Novel Object Emergence</td><td>19 / 20 95.0</td></tr><tr><td>Total</td><td>71 / 80 88.8</td></tr></table>

TABLE V: Success Rate of Dynamic Updates in Real-world.

## E. Real-world Dynamic Update

To assess DynamicGSGâs capability in adapting to dynamic environments, we establish 30 scenes in our lab with a total of 80 manual environment modifications, including object disappearance (30 instances), object relocation (30 instances), and novel object appearance (20 instances) and employ VINS-Fusion [33] integrated with an Intel RealSense D455 to acquire aligned RGB-D streams at a resolution of 640Ã480, along with initial pose estimation.

As detailed in Tab. V, DynamicGSG, while leveraging the method detailed in Sec. III-E to perform initial pose refinement and incrementally construct Gaussian scene graphs, successfully detects environment changes and executes corresponding dynamic updates, ensuring temporal consistency between the scene graphs and the real-world environments. Some visualization results of the experiment, presented in Fig. 1 and 6, demonstrate DynamicGSG detects three types of environment changes and effectively performs instance-level local updates to construct dynamic, highfidelity Gaussian scene graphs.

## V. CONCLUSION

In this paper, we introduce DynamicGSG, a novel system designed to construct dynamic high-quality 3D Gaussian scene graphs. Utilizing fast differentiable rendering of 3D Gaussians, our system alleviates key problems in 3D scene graphs, such as the absence of mechanisms for dynamic environment adaptation and poor reconstruction quality. Extensive experimental results demonstrate that our system can perform dynamic updates in scene graphs according to real environment changes, effectively represent the spatial and semantic relationships between objects, and accurately capture intricate geometric details of scenes. These capabilities enable our system to assist agents in performing long-term navigation and mobile manipulation within indoor environments.

## REFERENCES

[1] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa et al., âConceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 5021â5028.

[2] A. Werby, C. Huang, M. Buchner, A. Valada, and W. Burgard, Â¨ âHierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation,â in Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.

[3] Z. Yan, S. Li, Z. Wang, L. Wu, H. Wang, J. Zhu, L. Chen, and J. Liu, âDynamic open-vocabulary 3d scene graphs for long-term languageguided mobile manipulation,â arXiv preprint arXiv:2410.11989, 2024.

[4] S. Linok, T. Zemskova, S. Ladanova, R. Titkov, and D. Yudin, âBeyond bare queries: Open-vocabulary object retrieval with 3d scene graph,â arXiv preprint arXiv:2406.07113, 2024.

[5] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[6] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in ACM SIGGRAPH 2024 conference papers, 2024, pp. 1â11.

[7] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, âMip-splatting: Alias-free 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 447â19 456.

[8] K. Cheng, X. Long, K. Yang, Y. Yao, W. Yin, Y. Ma, W. Wang, and X. Chen, âGaussianpro: 3d gaussian splatting with progressive propagation,â in Forty-first International Conference on Machine Learning, 2024.

[9] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[10] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGsslam: Dense visual slam with 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 595â19 604.

[11] Y. Zheng, X. Chen, Y. Zheng, S. Gu, R. Yang, B. Jin, P. Li, C. Zhong, Z. Wang, L. Liu et al., âGaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping,â arXiv preprint arXiv:2403.09637, 2024.

[12] M. Ye, M. Danelljan, F. Yu, and L. Ke, âGaussian grouping: Segment and edit anything in 3d scenes,â in European Conference on Computer Vision. Springer, 2024, pp. 162â179.

[13] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, âSgs-slam: Semantic gaussian splatting for neural dense slam,â in European Conference on Computer Vision. Springer, 2024, pp. 163â 179.

[14] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, âLangsplat: 3d language gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 051â20 060.

[15] Y. Wu, J. Meng, H. Li, C. Wu, Y. Shi, X. Cheng, C. Zhao, H. Feng, E. Ding, J. Wang et al., âOpengaussian: Towards point-level 3d gaussian-based open vocabulary understanding,â arXiv preprint arXiv:2406.02058, 2024.

[16] N. Hughes, Y. Chang, and L. Carlone, âHydra: A real-time spatial perception system for 3D scene graph construction and optimization,â 2022.

[17] N. Hughes, Y. Chang, S. Hu, R. Talak, R. Abdulhai, J. Strader, and L. Carlone, âFoundations of spatial perception for robotics: Hierarchical representations and real-time systems,â The International Journal of Robotics Research, 2024.

[18] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, âYolo-world: Real-time open-vocabulary object detection,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16 901â16 911.

[19] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., âSegment anything,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015â4026.

[20] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PMLR, 2021, pp. 8748â8763.

[21] A. Rosinol, A. Gupta, M. Abate, J. Shi, and L. Carlone, â3d dynamic scene graphs: Actionable spatial perception with places, objects, and humans,â arXiv preprint arXiv:2002.06289, 2020.

[22] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., âGrounding dino: Marrying dino with grounded pre-training for open-set object detection,â in European Conference on Computer Vision. Springer, 2024, pp. 38â55.

[23] Y. Zhang, X. Huang, J. Ma, Z. Li, Z. Luo, Y. Xie, Y. Qin, T. Luo, Y. Li, S. Liu et al., âRecognize anything: A strong image tagging model,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1724â1732.

[24] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, âSigmoid loss for language image pre-training,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 11 975â 11 986.

[25] H. Jiang, B. Huang, R. Wu, Z. Li, S. Garg, H. Nayyeri, S. Wang, and Y. Li, âRoboexp: Action-conditioned scene graph via interactive exploration for robotic manipulation,â arXiv preprint arXiv:2402.15487, 2024.

[26] OpenAI, âGpt-4 technical report,â 2024.

[27] A. Guedon and V. Lepetit, âSugar: Surface-aligned gaussian splatting Â´ for efficient 3d mesh reconstruction and high-quality mesh rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5354â5363.

[28] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[29] J. Wei and S. Leutenegger, âGsfusion: Online rgb-d mapping where gaussian splatting meets tsdf fusion,â IEEE Robotics and Automation Letters, 2024.

[30] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., âDinov2: Learning robust visual features without supervision,â arXiv preprint arXiv:2304.07193, 2023.

[31] T. Qin, P. Li, and S. Shen, âVins-mono: A robust and versatile monocular visual-inertial state estimator,â IEEE Transactions on Robotics, vol. 34, no. 4, pp. 1004â1020, 2018.

[32] T. Qin and S. Shen, âOnline temporal calibration for monocular visual-inertial systems,â in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 3662â 3669.

[33] T. Qin, J. Pan, S. Cao, and S. Shen, âA general optimization-based framework for local odometry estimation with multiple sensors,â 2019.

[34] T. Qin, S. Cao, J. Pan, and S. Shen, âA general optimization-based framework for global pose estimation with multiple sensors,â 2019.

[35] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieÃner, âScannet: Richly-annotated 3d reconstructions of indoor scenes,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828â5839.

[36] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[37] C. Yeshwanth, Y.-C. Liu, M. NieÃner, and A. Dai, âScannet++: A highfidelity dataset of 3d indoor scenes,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 12â22.