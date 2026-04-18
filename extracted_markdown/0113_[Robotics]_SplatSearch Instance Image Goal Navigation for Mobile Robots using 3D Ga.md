# SplatSearch: Instance Image Goal Navigation for Mobile Robots using 3D Gaussian Splatting and Diffusion Models

Siddarth Narasimhan, Student Member, IEEE, Matthew Lisondra, Student Member, IEEE, Haitong Wang, Student Member, IEEE, Goldie Nejat, Member, IEEE

Abstractâ The Instance Image Goal Navigation (IIN) problem requires mobile robots deployed in unknown environments to search for specific objects or people of interest using only a single reference goal image of the target. This problem can be especially challenging when: 1) the reference image is captured from an arbitrary viewpoint, and 2) the robot must operate with sparse-view scene reconstructions. In this paper, we address the IIN problem, by introducing SplatSearch, a novel architecture that leverages sparse-view 3D Gaussian Splatting (3DGS) reconstructions. SplatSearch renders multiple viewpoints around candidate objects using a sparse online 3DGS map, and uses a multi-view diffusion model to complete missing regions of the rendered images, enabling robust feature matching against the goal image. A novel frontier exploration policy is introduced which uses visual context from the synthesized viewpoints with semantic context from the goal image to evaluate frontier locations, allowing the robot to prioritize frontiers that are semantically and visually relevant to the goal image. Extensive experiments in photorealistic home and real-world environments validate the higher performance of SplatSearch against current state-of-the-art methods in terms of Success Rate and Success Path Length. An ablation study confirms the design choices of SplatSearch. Our project page can be found at https://splat-search.github.io/.

## I. INTRODUCTION

Mobile robots have been increasingly deployed in humancentered environments, where many of their tasks involve locating and navigating to people or objects of interest. For example, in healthcare settings, robots need to search for and deliver medical supplies to specific nurses [1], locate particular patients in a hospital [2], or guide visitors in longterm care homes [3]. In domestic environments, mobile robots find and retrieve household items needed for meal preparation [4], laundry [5], or cleaning [6]. In many of these applications, reference images of the target object/person are provided a priori, enabling a robot to visually search for the object/person during navigation [7].

The aforementioned problem is known as Instance Image Goal Navigation (IIN). IIN is defined as a mobile robot requiring to navigate an unknown static environment to search for a person or object, given a reference image [8]. In IIN, the following challenges need to be addressed: 1) the viewpoint of the provided goal image may differ from the robotâs viewpoint, increasing the difficulty of recognizing the goal object during navigation [9]; and 2) a robot must leverage visual and semantic context from earlier exploration to improve the efficiency of future navigation actions [10], [11].

To-date, existing IIN approaches have primarily used deep learning [7], [8], [12]-[23], and large foundation models [10], [24]-[26]. Deep learning methods have used convolutional encoders to extract features from visual observations and the goal image, capturing information such as scene geometry and visual appearance (e.g. color, texture and object shape). These features are used to predict the robotâs next action using actor-critic models [12]-[15], transformers [7], [16], diffusion models [17], recurrent neural networks [19] or inverse dynamics models [20]. Large foundation model approaches use pre-trained encoders from vision transformers to convert image observations and the goal image into semantic embeddings [10], [24]-[26]. The semantic embeddings are then used by scene graphs [10], [26], or reinforcement learning architectures [24], to direct the robot towards the goal. However, these existing methods assume that the goal image is captured from a viewpoint that is similar to the robotâs viewpoint. For example, if the goal image is captured from a Birdâs Eye View (BEV), the robot may fail to recognize it from its own frontal view perception due to feature differences across the two viewpoints [9].

<!-- image-->  
Fig. 1. An overview of SplatSearch. SplatSearch builds a sparse-view 3DGS map and uses semantic and visual context scores to select frontier locations during exploration. To recognize goal images provided from arbitrary viewpoints, SplatSearch renders novel viewpoints around the objects and inpaints incomplete regions using a multi-view diffusion model.

To address the aforementioned limitation, recent research has explored the use of alternative scene representations that render the environment from novel viewpoints. In particular, 3D Gaussian Splatting (3DGS) [27] enables photorealistic 3D scene reconstruction of an environment using collections of 3D Gaussian ellipsoids. With respect to the IIN task, 3DGS, has been used for viewpoint rendering allowing a robot to localize a goal image within the map of an environment and then navigate to the goal using waypoint-based planning [9], [28], [29], or model predictive control [30]. However, these approaches assume that a detailed 3DGS map is available prior to robot deployment, which can limit their usability in unknown environments [31].

In this paper, we propose SplatSearch, a novel 3DGS-based architecture to address the IIN problem of a mobile robot searching for a static object or person in an unknown environment with sparse reconstruction, using only a single reference image from an arbitrary viewpoint, Fig. 1. The main contributions of SplatSearch are:

1) the integration of a viewpoint synthesis method that renders multiple views for candidate objects using a sparse online 3DGS map.

2) a View-Consistent Image Completion Network (VCICN) multi-view diffusion model that we have developed which is trained on sparse-view 3DGS renders. The VCICN inpaints missing regions of viewpoints for feature matching with the goal object. This unique approach enables viewpointinvariant goal recognition by combining geometry-based novel view rendering with diffusion-based inpainting.

3) A novel frontier exploration strategy for goal-directed search that is, to the authorsâ knowledge, the first to use: i) semantic context scores between RGB observations and the goal image, and ii) visual context scores from the inpainted novel viewpoints used to evaluate and select frontiers. This fusion allows a mobile robot to prioritize frontiers that are both semantically and visual contextually similar to the target object/person.

## II. RELATED WORKS

We define existing IIN methods into: 1) deep learningbased methods [7], [8], [12]-[23], 2) large foundation model methods [10], [24]-[26], and 3) 3DGS methods [9], [28]-[30], [32].

## A. Deep Learning-Based Methods

Deep learning-based methods for the IIN task have mainly used deep reinforcement learning (DRL) [7], [8], [12]-[16], [19], [21]-[23] or imitation learning (IL) [17], [18], [20].

DRL-based methods have used convolutional encoders such as ResNet [7], [8], [12]-[15], [21], [22], graph convolutional networks [16], or vision transformers [19] to generate joint embeddings from RGB image observations and the goal image to inform the navigation policy. DRL architectures including actor-critic networks [13]-[15], [22], proximal policy optimization (PPO) [8], [12], [21], [23], and transformers [7], [16], have been used to generate future navigation actions given the current and history of image observations. In [23], an additional navigation policy was introduced to alternate between exploring new frontiers and verifying whether the current observation matches the goal image. To enable multi-goal IIN tasks, navigation policies were memory-augmented using attention networks [12], [16], topological graphs [22], or feature-extraction encoders [13].

IL-based methods have generated robot navigation actions by learning from expert demonstrations by other robots using indoor navigation [17], [20], or IIN datasets [18]. In [17], IL was used to train a diffusion model to enable the generation of collision-free navigation trajectories, while being conditioned on the goal image. In [18], robots learned from demonstrations to maintain a topological graph of an environment, allowing them to generate subgoals towards the goal image. In [20], generative IL was used by training a variational encoder to predict the next observation from expert demonstrations, to reach the goal image.

## B. Large Foundation Model Methods

Large foundation models have either used vision-language models (VLMs) to generate semantic priors with a robotâs current RGB observations in order to locate a goal object [10] [24], or have queried a large language model (LLM) directly to inform the navigation policy [25], [26].

With respect to using VLMs, in [10], [24], RGB observations were converted into a semantic embedding space using contrastive language image pretraining (CLIP) [33], while the goal image was processed through a ResNet encoder to align semantic features and locate the goal. For example, in [10], a semantic map of indoor environments was incrementally constructed using CLIP to keep a memory of instances of object categories. In [24], the embeddings from the observations and goal were concatenated and trained with an actor-critic policy network to determine robot actions.

Methods that have directly queried LLMs to generate navigation actions for robots have prompted the LLMs with an online scene graph of the environments, describing spatial and semantic relationships between objects [26], [25]. The scene graphs were used to generate plans towards candidate objects, and a local feature matching method, such as LightGlue [26] or SBERT [25], was used to identify the goal.

## C. 3DGS Methods

3DGS methods have leveraged novel view-synthesis to preserve detailed visual information across diverse viewpoints of a goal image [9], [28]-[30], [32]. Prior to navigation, these methods built high-fidelity 3DGS maps and embedded semantic features into the maps using CLIP [9] [29] or created topological graphs [28] to enable querying of goal images. In [9], multiple viewpoints around candidate objects were generated, then a feature matcher was used to identify the instance of the object. In [29], an improvement was made to [9], by estimating optimal viewpoints using hierarchical scoring. After an object instance is identified, navigation policies used to reach the goal object include diffusion [28], model predictive control [30] or the Fast Marching Method [9], [29]. IGL-Nav, [32], builds an online 3DGS of the environment instead of using a pre-built 3DGS representation. The 3DGS map is used for navigation by localizing the 6DoF pose of the goal image, and iteratively navigating towards this location.

## D. Summary of Limitations

DRL-based methods used pre-trained encoders to extract and compare features between a current observation and a goal image of an object of interest from the robot perspective [7], [8], [12]-[16], [19], [21]-[23]. However, these encoders are not able to recognize goal objects from arbitrary viewpoints, resulting in degraded performance [9]. IL methods, [17], [18], [20], rely on trained policies from datasets, thus, their recognition ability is limited to the set of objects included in these datasets and they cannot generalize to new IIN scenarios [20]. The large foundation model methods, [10], [24]-[26], are unable to distinguish between visually distinct but semantically similar objects (e.g., two different chairs) [34]. 3DGS methods, [9], [28]-[30], [32], assume that a dense 3DGS map is available to the robot prior to deployment, thus, cannot be applied to unknown environments where scene reconstructions are unavailable.

<!-- image-->  
Fig. 2. The SplatSearch architecture, consisting of: 1) Map Generation Module (MGM) which generates a 3DGS map for photorealistic mapping and occupancy map for frontier exploration, 2) Semantic Object Identification Module (SOIM) detects if a current RGB image contains an object with the same class as the goal image, 3) Novel Viewpoint Synthesis Module (NVSM) generates multiple viewpoints around the object, 4) View-Consistent Image Completion Network (VCICN) inpaints the sparse viewpoint images from NVSM, 5) Feature Extraction Module (FEM) which uses the inpainted images to match features with the goal image, and 6) Exploration Planner (EP) which selects and generates feasible paths towards new frontiers.

To address the above limitations, we propose SplatSearch, the first online 3DGS-based architecture to address the problem of IIN with goal images provided at arbitrary viewpoints in unknown and sparsely reconstructed environments. By synthesizing novel views of candidate objects using the 3DGS representation, SplatSearch achieves viewpoint-invariant recognition, overcoming the limitations of DRL encoders. Its reliance on geometric reconstruction and feature matching instead of learning from fixed datasets allows it to generalize to novel objects and environments. We also uniquely integrate both semantic and visual context for frontier exploration using the goal image to enable instancelevel recognition. By constructing sparse 3DGS maps and inpainting the viewpoints using a multi-view diffusion model, SplatSearch eliminates the dependence on pre-built reconstructions, enabling deployment in unseen settings.

## III. INSTANCE IMAGE GOAL NAVIGATION IN UNKNOWN ENVIRONMENTS

Robot IIN addresses the problem of a mobile robot, at an initial random pose in an unknown static 3D environment that needs to navigate to a goal object or person represented by an RGB image, $I ^ { g } \in \mathbb { R } ^ { \breve { H } \times W \times 3 }$ located at an unknown goal position $x ^ { g }$ . At each timestep, ??, the robot receives an RGB image and depth observation, $O _ { t } = ( I _ { t } , D _ { t } )$ , from its onboard RGB-D camera, and executes an action, $a _ { t } = \pi ( O _ { t } )$ , according to a navigation policy, ??. The total path length of the robot is given by:

$$
d ( \pi ) = \sum _ { t = 0 } ^ { T - 1 } \lVert x _ { t + 1 } - x _ { t } \rVert _ { 2 } ,\tag{1}
$$

where $x _ { t }$ is the robot position after action $a _ { t } ,$ and $T$ is the termination timestep. Robot navigation is considered successful if the robot reaches a distance within radius, $\tau _ { s u c c } ,$ of the true goal position. The objective is to execute a navigation policy that minimizes the expected path length of the mobile robot subject to the following success condition:

$$
\pi ^ { * } = \underset { \pi } { \operatorname { a r g m i n } } \ : \mathbb { E } [ d ( \pi ) \ : | \ : \| x _ { T } - x ^ { g } \| _ { 2 } \leq \tau _ { s u c c } ] .\tag{2}
$$

## IV. SPLATSEARCH ARCHITECTURE

The proposed SplatSearch architecture consists of six main modules, Fig. 2: 1) Map Generation Module (MGM), 2) Semantic Object Identification Module (SOIM), 3) Novel Viewpoint Synthesis Module (NVSM), 4) View-Consistent Image Completion Network (VCICN), 5) Feature Extraction Module (FEM), and 6) Exploration Planner (EP). Inputs are RGB-D observations with corresponding poses from the robotâs onboard camera and a single goal image. These are first processed by the MGM, which incrementally constructs a sparse-view 3DGS map and a value map. The RGB-D observations and the 3DGS map are used as inputs into the SOIM, which analyzes the current RGB frame to detect candidate objects of the same class as the goal image. Detected objects are segmented and assigned unique identifiers, producing a set of candidate object locations. These centroids are used by the NVSM to render multiple candidate views of each detected object from novel poses around the centroid using the 3DGS map. Since these renders are often incomplete due to sparse reconstructions, they are refined by the VCICN by inpainting missing regions to produce complete views. The completed renders are compared against the goal image by the FEM using feature matching, outputting visual context scores that represent the likelihood of a detected instance matching the goal object. The EP uses the visual context scores and semantic scores obtained from a pre-trained image encoder to select frontiers for navigation. If a goal object is recognized, the EP plans a direct navigation path to its location; otherwise, it selects the next frontier that maximizes semantic and feature matching scores. Each module is discussed in detail below.

## A. Map Generation Module (MGM)

The MGM utilizes RGB images, $I _ { t } ,$ depth images, $D _ { t } ,$ , and robot position, $\boldsymbol { x } _ { t } ,$ to incrementally construct: 1) a sparseview 3DGS map of the environment for novel viewpoint rendering around objects, and 2) a value map, which is used to generate semantic context scores to evaluate frontiers during exploration.

## 1) 3DGS Map

A 3DGS map of the environment at timestep $t , \pmb { \mathcal { M } } _ { t } ^ { G S }$ , is represented by anisotropic 3D Gaussians, $G _ { i }$ , each parameterized by color $\mathbf { } c _ { i } ,$ , position $\pmb { \mu } _ { i } \in \ \mathbb { R } ^ { 3 }$ , covariance $\pmb { \Sigma } _ { i } \in$ $\mathbf { \mathbb { R } } ^ { 3 \times 3 }$ , and opacity $\pmb { \mathscr { o } } _ { i } . \ \mathcal { P } : \ \mathbb { R } ^ { 2 }  \mathbb { R } ^ { 3 }$ denotes the mapping from a 2D pixel coordinate in the RGB image, $u _ { k } \in \mathbb { R } ^ { 2 }$ , to its corresponding 3D point in the scene. The weight of 3D Gaussian, $G _ { i } ,$ , at a pixel ${ \pmb u } _ { k }$ in the rendered image is given by [35]:

$$
w _ { i } ( { \pmb u } _ { k } ) = \pmb { o } _ { i } \exp \left( - \frac { 1 } { 2 } ( \mathcal { P } ( { \pmb u } _ { k } ) - { \pmb \mu } _ { i } ) ^ { T } { \pmb \Sigma } _ { i } ^ { - 1 } ( \mathcal { P } ( { \pmb u } _ { k } ) - { \pmb \mu } _ { i } ) \right) ,\tag{3}
$$

where $i \in \{ 1 , \ldots , N _ { G } \}$ represents indices for the set of 3D Gaussians and $k \in \{ 1 , \ldots , N _ { P } \}$ represents the indices for the set of pixels in the RGB image. The rendered RGB color $\hat { I } _ { t } ,$ depth $\widehat { D } _ { t }$ and opacity $\hat { O } _ { t }$ images, are generated by alphacompositing the 2D projection of each Gaussian into the image plane [36]:

$$
\hat { I } _ { t } ( \pmb { u } _ { k } ) = \sum _ { i = 1 } ^ { N _ { G } } \pmb { c } _ { i } w _ { i } ( \pmb { u } _ { k } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \pmb { u } _ { k } ) \right) .\tag{4}
$$

$$
\widehat { D } _ { t } ( \pmb { u } _ { k } ) = \sum _ { i = 1 } ^ { N _ { G } } z _ { i } w _ { i } ( \pmb { u } _ { k } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \pmb { u } _ { k } ) \right) ,\tag{5}
$$

$$
\hat { O } _ { t } ( \pmb { u } _ { k } ) = \sum _ { i = 1 } ^ { N _ { G } } w _ { i } ( \pmb { u } _ { k } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \pmb { u } _ { k } ) \right) ,\tag{6}
$$

where $z _ { i }$ is the depth of $G _ { i }$ in the robotâs camera frame. The 3DGS map is optimized online by minimizing both the photometric loss, $\mathcal { L } _ { p } .$ , and geometric loss, $\mathcal { L } _ { g } ,$ between the rendered and the observed RGB-D frames at timestep ?? [35]:

$$
\mathcal { L } _ { p } = \left| I _ { t } - \hat { I } _ { t } \right| + \left( 1 - \mathrm { S S I M } \big ( I _ { t } , \hat { I } _ { t } \big ) \right) \mathrm { , }\tag{7}
$$

$$
\begin{array} { r } { \mathcal { L } _ { g } = \big | D _ { t } - \widehat { D } _ { k } \big | , } \end{array}\tag{8}
$$

$$
\begin{array} { r } { \mathcal { L } _ { t } = \lambda _ { p } \mathcal { L } _ { p } + \lambda _ { g } \mathcal { L } _ { g } , } \end{array}\tag{9}
$$

where $\lambda _ { p }$ and $\lambda _ { g }$ are weighting coefficients determined empirically through expert domain tuning to balance the relative importance of photometric fidelity and geometric accuracy. The Structural Similarity Index (SSIM) is used within the photometric loss to minimize perceptual differences between the rendered and observed RGB images, ensuring that the reconstructed views maintain visual consistency with the actual camera observations. The 3DGS map is updated online at each timestep in regions which are not previously mapped or regions with low accumulated opacity:

$$
M _ { t } ( u _ { k } ) = \left( \hat { O } _ { t } < \tau _ { o } \right) \cup \left( ( D _ { t } < \widehat { D } _ { t } ) \cap \left( \left| D _ { t } - \widehat { D } _ { t } \right| > \tau _ { M } \right) \right) ,\tag{10}
$$

where $\tau _ { o }$ is the opacity threshold tuned empirically from simulated experiments to optimize for reconstruction quality, and $\tau _ { M }$ is the median depth error. Pixels satisfying $\boldsymbol { M } _ { t } ( \boldsymbol { u } _ { k } ) =$ 1 are used to update the corresponding Gaussians in the 3DGS map.

## 2) Value Map

The depth images, $D _ { t }$ , are used to build an online occupancy map, $\mathbf { \mathcal { M } } _ { t } ^ { o c c }$ , consisting of occupied, free and unknown cells. Frontier points, $\mathbf { \mathcal { F } } _ { t } .$ , are extracted as the boundary locations between the free and unknown cells.

The value map, $\mathbf { \mathcal { M } } _ { t } ^ { v a l }$ , is constructed by assigning each free-space cell in $\mathbf { \mathcal { M } } _ { t } ^ { o c c }$ , with a semantic context score quantifying its visual and semantic relevance to $I ^ { g }$ . To obtain the semantic context scores, we use only the image encoder of a pre-trained BLIP-2 VLM [37] to generate semantic embeddings for the goal image, $\begin{array} { r } { { \boldsymbol { e } } _ { G } , } \end{array}$ and the current RGB observation, $e _ { t }$ . The semantic score is determined through the cosine similarity between the embeddings:

$$
v _ { t } = { \frac { e _ { t } \cdot e _ { g } } { \lVert e _ { t } \rVert \lVert e _ { g } \rVert } } ,\tag{11}
$$

where a higher value represents a higher semantic relevance to the goal image. The value $v _ { t }$ is stored in all of the free space pixels within the current field-of-view (FOV) of the robot.

We also generate a confidence map, $\pmb { \mathscr { M } } _ { t } ^ { c o n f }$ , to determine how to update a pixelâs semantic value if it is already assigned a value and is within the current FOV of the robot. The confidence of a pixel ?? within the FOV of the robot is determined as $c _ { t , i } = \cos ^ { 2 } ( \pi \theta _ { i } / \theta _ { f o v } )$ , where $\theta _ { f o v }$ is the FOV of the robot, and $\theta _ { i }$ is the angle of the pixel relative to the optical axis of the robot. At each timestep, the new value, $v _ { t , i } ^ { n e w }$ , and confidence score, $c _ { t , i } ^ { n e w }$ , for all pixels in the robotâs FOV are recomputed [11]:

$$
v _ { t , i } ^ { n e w } = \frac { c _ { t , i } v _ { t } + c _ { t , i } ^ { p r e v } v _ { t , i } ^ { p r e v } } { c _ { t , i } + c _ { t , i } ^ { p r e v } } , c _ { t , i } ^ { n e w } = \frac { c _ { t } ^ { 2 } + \left( c _ { t , i } ^ { p r e v } \right) ^ { 2 } } { c _ { t , i } + c _ { t , i } ^ { p r e v } } ,\tag{12}
$$

where $v _ { t , i } ^ { p r e v }$ and $c _ { t , i } ^ { p r e v }$ are the previous value and previous confidence score of the pixel, respectively. For each frontier, $f \in \mathcal { F } _ { t } ,$ we compute a normalized average semantic score, $\overline { { V _ { t } } } ( f )$ as the mean of $\mathbf { \mathcal { M } } _ { t } ^ { v a l }$ within a circle of radius $r _ { v a l }$ around the frontier point, ??. These scores are used by the EP to guide frontier selection towards regions that are semantically similar to the goal image.

## B. Semantic Object Identification Module (SOIM)

The SOIM determines whether the current RGB image, $I _ { t } ,$ contains an instance belonging to the image goal class, and assigns a unique identifier to each detected object.

We use YOLOv7 [38] to generate bounding boxes around candidate objects and MobileSAM [39] to generate instance masks $\left\{ m _ { t , n } \right\} _ { n = 0 } ^ { n _ { t } }$ , where $n _ { t }$ denotes the number of masks detected at timestep ??. For each mask $m _ { t , n }$ , the pixels are mapped to their corresponding 3D Gaussians in the 3DGS map, $\pmb { \mathcal { M } } _ { t } ^ { G S }$ , using ??. The corresponding set of Gaussians is denoted as $\pmb { \mathcal { G } } ^ { ( i ) } = \{ \pmb { \mu } _ { i } ^ { ( i ) } \}$ , which are aggregated and averaged to estimate the 3D centroid of the object, $\overline { { \pmb { \mu } } } ^ { ( i ) }$ ), with identification ??. Each object is stored in an object database $\pmb { \mathcal { D } } _ { O }$ as:

$$
{ \pmb { \mathcal { O } } } ^ { ( i ) } = \big \{ { \pmb { \mathcal { G } } } ^ { ( i ) } , { \pmb { \overline { { \mu } } } } ^ { ( i ) } , { \pmb { \mathcal { X } } } ^ { ( i ) } \big \} ,\tag{13}
$$

where $\pmb { \mathscr { X } } ^ { ( i ) }$ represents the set of positions from which object ?? has been observed. When a new object is detected, its centroid $\overline { { \pmb { \mu } } } ^ { n e w }$ is compared to existing object centroids in $\pmb { \mathcal { D } } _ { O }$ If no match is found, it is appended as a new entry to ${ \pmb { \mathcal D } } _ { O } ;$ otherwise, the matched object is updated with the new viewpoint. The centroid, $\overline { { \pmb { \mu } } } ^ { ( i ) }$ , is passed to the NVSM each time an object in $\pmb { \mathcal { D } } _ { O }$ has been observed in $f _ { O }$ new distinct viewpoints (i.e. when $| \boldsymbol { x } ^ { ( i ) } |$ mod $f _ { O } = 0 )$ . This ensures that the NVSM is only executed when increasingly refined object information is obtained.

## C. Novel Viewpoint Synthesis Module (NVSM)

The NVSM generates camera viewpoints using the 3DGS map and the input centroid, $\overline { { \pmb { \mu } } } ^ { ( i ) }$ , to render novel images around the object. Around the centroid, $\overline { { \pmb { \mu } } } ^ { ( i ) }$ , viewpoints are sampled along a spherical surface of radius $r _ { v }$ , restricted to the upper hemisphere. A total of $N _ { R }$ viewpoints are generated by rotating at fixed angular steps, with each camera-to-world transformation matrix, $\mathbf { T } _ { c  w } ,$ positioned tangentially to the sphere and oriented towards the centroid. Using these transformation matrices, the set of novel images, $\left\{ \hat { I } _ { i } ^ { ( i ) } \right\} _ { i = 0 } ^ { N _ { R } }$ are rendered using the 3DGS map, and are subsequently provided to the VCICN.

## D. View-Consistent Image Completion Network (VCICN)

The VCICN is used to complete rendered images by inpainting pixels that are unobserved or partially observed in the 3DGS map. It uses a multi-view diffusion network to generate structurally consistent and complete images that can be reliably compared by the FEM, since the 3DGS map is only sparsely generated. The input to the VCICN consists of $N _ { R }$ rendered viewpoints, $\left\{ \hat { I } _ { i } ^ { ( i ) } \right\} _ { i = 0 } ^ { N _ { R } } ,$ , and their corresponding binary masks $\left\{ \widehat { M } _ { i } ^ { \left( i \right) } \right\} _ { i = 0 } ^ { N _ { R } }$ , representing the observed and missing regions of the image. We set $\widehat { M } _ { 0 } ^ { ( i ) } = \mathbf { 0 }$ , as the first viewpoint that is unmasked for training stability.

Each rendered image, is encoded into a latent feature map using a variational autoencoder (VAE):

$$
\begin{array} { r } { z _ { 0 } ^ { i } = \mathrm { V A E } ( \hat { I } _ { i } ^ { ( i ) } ) \in \mathbb { R } ^ { H \times W \times 4 } . } \end{array}\tag{14}
$$

The binary mask $\widehat { M } _ { i } ^ { ( i ) }$ is down-sampled using the same VAE network to match the latent resolution. For each latent $z _ { 0 } ^ { i }$ , we simulate the forward diffusion process at a randomly chosen timestep ?? to produce a noisy latent vector:

$$
z _ { t } ^ { i } = \sqrt { \alpha _ { t } } z _ { 0 } ^ { i } + \sqrt { 1 - \alpha _ { t } } \epsilon ,\tag{15}
$$

where $\alpha _ { t }$ follows a consistent noise schedule, defining the amount of noise added at each timestep, and $\epsilon { \sim } \mathcal { N } ( 0 , \mathbf { I } )$ . The sequence of inputs at timestep ?? is:

$$
\begin{array} { r } { \pmb { x } _ { t } = \big [ \boldsymbol { z } _ { t } ^ { 0 : N _ { R } } , \boldsymbol { z } ^ { 0 : N _ { R } } \big ( 1 - \widehat { M } ^ { 0 : N _ { R } } \big ) , \widehat { M } ^ { 0 : N _ { R } } \big ] , } \end{array}\tag{16}
$$

where $\widehat { M }$ is the down sampled mask. The VCICN predicts the noise $\epsilon _ { \theta } ( x _ { t } , \tau _ { \phi } ( c ) , t )$ where ?? is the CLIP-encoded text prompt corresponding to the goal object. The network is trained with an MSE loss, to encourage accurate noise prediction and enable reconstruction of complete, viewconsistent images [40]:

$$
\mathcal { L } _ { D } = \mathbb { E } _ { x _ { 0 } , \epsilon , t , c } \left[ \left\| \epsilon - \epsilon _ { \theta } \big ( \pmb { x } _ { t } , \tau _ { \phi } ( c ) , t \big ) \right\| _ { 2 } ^ { 2 } \right] .\tag{17}
$$

After reverse diffusion denoising, the VCICN outputs the completed viewpoint set of images, $\left\{ \mathbf { I } _ { i } ^ { ( i ) } \right\} _ { i = 0 } ^ { N _ { R } }$ , which are then used by the FEM for feature extraction against the goal image.

## E. Feature Extraction Module (FEM)

The FEM determines whether the current object instance corresponds to the goal image, ????. The input is the set of completed novel viewpoints $\left\{ \mathbf { I } _ { i } ^ { ( i ) } \right\} _ { i = 0 } ^ { N _ { R } }$ , generated from the VCICN.

The object score is computed using EfficientLoFTR, denoted by $\pmb { { \cal E } } ( \cdot , \cdot )$ which returns the number of feature correspondences between two images. For each viewpoint, $\mathbf { I } _ { i } ^ { ( i ) }$ , the viewpoint with the highest match is computed as:

$$
i ^ { * } = \arg \operatorname* { m a x } _ { i } { \cal E } \big ( I ^ { g } , \mathbf { I } _ { i } ^ { ( i ) } \big ) .\tag{18}
$$

If the object score satisfies $\begin{array} { r } { \pmb { s } _ { m a x } ^ { ( i ) } = \pmb { E } \big ( I ^ { g } , \mathbf { I } _ { i ^ { * } } ^ { ( i ) } \big ) \geq \tau _ { m } . } \end{array}$ where $\tau _ { m }$ is the feature-match threshold determined through simulated experiments, the object is declared to match the goal instance and is passed into the EP for selection of the final frontier. If $s _ { m a x } ^ { ( i ) } < \tau _ { m }$ , the object is not considered a match and the score is used by the EP to determine the next frontier.

## F. Exploration Planner (EP)

As previously mentioned, the EP utilizes a frontier-based exploration strategy, where unexplored frontiers are chosen as intermediate navigation points during exploration. Our contribution to traditional frontier-based exploration is the combined use of: 1) semantic context from the value map to enable the EP to prioritize frontiers that are semantically relevant to the goal image, to guide the robot when it is not within in the goalâs vicinity, and 2) visual context from the FEM to select frontiers close to object instances with higher feature-matching scores, to enable instance-level recognition. Namely, the EP determines the next frontier location to navigate to using $\overline { { V _ { t } } } ( f )$ and $\pmb { \mathcal { F } } _ { t }$ from the MGM, and $\left\{ \pmb { s } _ { m a x } ^ { ( i ) } \right\} _ { i = 0 } ^ { | \pmb { \mathcal { D } } _ { O } | }$ from the FEM.

During exploration, if the FEM has not detected the goal object, the EP selects from the set of frontiers, $f \in \mathcal { F } _ { t } .$ , that maximizes the utility function:

$$
f _ { t } ^ { * } = \underset { f \in \mathcal { F } _ { t } } { \operatorname { a r g m a x } } [ \alpha \cdot d _ { t } ( f ) + \beta \cdot \mathcal { E } _ { t } ( f ) + \delta \cdot \bar { V _ { t } } ( f ) ] ,\tag{19}
$$

where ??, $\beta , \delta$ are the weighting coefficients, $d _ { t } ( f )$ is the normalized inverse distance between $x _ { t }$ and $f . \ \mathcal { E } _ { t } ( f )$ is the normalized object score determined as:

$$
\mathcal { E } _ { t } ( f ) = \frac { 1 } { C } \operatorname* { m a x } _ { i } \frac { s _ { m a x } ^ { ( i ) } } { \tau _ { m } \lVert f - \overline { { \pmb { \mu } } } ^ { ( i ) } \rVert } ,\tag{20}
$$

where ?? is the normalization constant. $\mathbf { \mathcal { M } } _ { t } ^ { o c c }$ is then used by the Navigation Controller, which utilizes the Fast Marching Method (FMM) [41] to generate the shortest feasible path to navigate from the current location, $x _ { t } ,$ , to $x _ { T } = f _ { t } ^ { * }$ . If the FEM confirms a match with the goal image, FMM plans a path to $\overline { { \pmb { \mu } } } ^ { ( i ) * }$ . When the goal location is reached, the EP executes a stop action and the episode is completed.

## V. MULTI-VIEW DIFFUSION POLICY

The details of data collection and training for the VCICN are explained below.

## A. Dataset Collection

We used the Habitat simulator with the HM3DSem-v0.2 dataset [42], which provides 145 training, 36 validation and 35 testing environments for the IIN task. To construct the training data for VCICN, we executed SplatSearch (without VCICN) on the 145 environments in the training set, spanning six object classes (TV, sofa, chair, table, plant, monitor) [8]. When the SOIM detects an object instance, we applied the NVSM to capture $N _ { R } = 8$ viewpoints around the object, with their corresponding masks representing regions that are not constructed. This process produced a dataset of approximately 2,000 objects, each with 8 sparse-views, corresponding to 16,000 images in total. We augment this dataset with 400 people each with 8 viewpoints from the EgoBody dataset [43], resulting in a total of 19,200 images.

## B. Training

Training was completed on an RTX 4090 GPU with 24GB of VRAM. The model was fine-tuned for 10k steps with a batch size of 64, learning rate of 0.0001. Training was completed in 46 hours using early stopping, Namely, the epoch with the lowest validation loss was selected as the final checkpoint to prevent overfitting to the data.

## VI. EXPERIMENTS

We evaluated the performance of SplatSearch by conducting: 1) a comparison study with state-of-the-art (SOTA) learning methods in a photorealistic simulator, 2) an ablation study to investigate the design choices of SplatSearch, and 3) real-world experiments to evaluate the generalizability of SplatSearch in real-world environments.

## A. Comparison Study

We evaluated SplatSearch in the Habitat simulator across 100 episodes using two validation sets, Fig. 3: 1) the standard HM3DSem-v0.2 validation dataset (HM3D-val) [8], and 2) a modified dataset designed to evaluate robustness to viewpoint invariance, HM3D-val-hard. We construct HM3D-val-hard by re-sampling goal images from HM3D-val in spherical coordinates around the object centroid with radial distance $r \in [ 0 . 5 , 2 . 5 ] m$ , azimuth $\theta \in ( 0 , \pi ]$ elevations, $r \in$ [1.0,2.0]??, and camera angle further perturbed between $[ - 1 0 ^ { o } , + 1 0 ^ { o } ]$ The performance metrics used to evaluate each benchmark are: 1) the Success Rate (SR); measures the percentage of episodes where the robot successfully reaches the goal object, and 2) the success path length (SPL); measures the success rate with the robotâs path length against the shortest path length from the start location to the goal object location:

$$
S P L = \frac { 1 } { N _ { T } } \sum _ { i = 1 } ^ { N _ { T } } S _ { i } \frac { d ( \pi ^ { * } ) } { \operatorname* { m a x } \bigl ( d ( \pi ) , d ( \pi ^ { * } ) \bigr ) } ,\tag{21}
$$

where $N _ { T }$ is the number of trials.

## 1) Comparison Methods:

We compared SplatSearch with the following SOTA methods:

GaussNav [9]: Uses a pre-built 3DGS representation of an environment to generate candidate viewpoints around goalclass instances, using a local feature matcher to locate the goal, and a classical planner to navigate to the goal. This was selected as a representative 3DGS approach.

IEVE [23]: A DRL-based approach that uses a switch policy to decide at every timestep to either explore a new frontier using DRL or verify current observation against the goal image using classical feature matching. IEVE was selected as it is the SOTA learning-based approach for IIN.

UniGoal [26]: Uses an LLM to build an online scene graph in semantic space, and a classical feature matcher to compare the goal image against the RGB observation. UniGoal uses only semantic context provided by the scene graph and LLM to evaluate frontiers for navigation.

2) Results: The SR and SPL of SplatSearch and the SOTA comparison methods are presented in Table I. SplatSearch achieved the highest SR (0.7000, 0.6300) and SPL (0.3740, 0.2675) with respect to the SOTA methods for both datasets. In general, the overall performance of all the methods degraded for HM3D-val-hard dataset due to the increased challenge of recognizing the goal image from BEV perspectives. However, SplatSearch was able to render objects from any viewpoint using the NVSM and VCICN, resulting in robust feature matching correspondences for recognition of the specific instance of the goal image. As IEVE does not generate a 3D scene representation of the environment and is unable to match features from the goal image to new unseen viewpoints, it had lower SRs and SPLs. UniGoal had lower SPL and SR than SplatSearch, as it converts the goal image into semantic labels and must physically navigate to candidate objects to perform RGBbased feature matching. In contrast, SplatSearch uses the NVSM to match features against the goal image without requiring navigation to the object, and also incorporates semantic/visual context scores resulting in higher path efficiency. GaussNav achieved the lowest SRs and SPLs as it requires constructing the 3DGS map of the full environment prior to search. This affects the SPL as navigation is required for generating the map and the SR as full scene reconstructions can exceed memory limits on the robotâs GPU. In contrast, SplatSearch avoids these issues by incrementally generating the 3DGS map and simultaneously searching for the goal object, avoiding redundant coverage.

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 3: (a) Sample goal images used in the HM3D_val dataset (top row) and corresponding images with BEV viewpoints in the HM3D_val_hard dataset (bottom row); and (b) Sample photorealistic indoor environments contained in the HM3D dataset.

<!-- image-->  
Fig. 4: SplatSearch was deployed in a real-world indoor environment with goal images of five new objects including a person, chair, laptop, TV and backpack.

TABLE I: COMPARISON BETWEEN SPLATSEARCH AND IIN SOTAS
<table><tr><td rowspan="3">Method</td><td colspan="2">HM3D-val</td><td colspan="2">HM3D-val-hard</td></tr><tr><td>SRâ</td><td>SPL â</td><td>SRâ</td><td>SPL â</td></tr><tr><td>GaussNav</td><td>0.5400</td><td>0.0673</td><td>0.4900</td><td>0.0428</td></tr><tr><td>IEVE</td><td>0.6600</td><td>0.2735</td><td>0.5800</td><td>0.1932</td></tr><tr><td>UniGoal</td><td>0.6900</td><td>0.2837</td><td>0.5900</td><td>0.1738</td></tr><tr><td>SplatSearch</td><td>0.7000</td><td>0.3740</td><td>0.6300</td><td>0.2675</td></tr></table>

## B. Ablation Study

We conducted an ablation study with different variants of SplatSearch on the HM3D-val and HM3D-val-hard datasets to investigate the impact of the specific modules in the architecture. These included:

SplatSearch without (w/o) NVSM: Uses the RGB observations, $I _ { t } ,$ to compute visual context scores instead of rendering novel viewpoints from the 3DGS map. Evaluates the impact of viewpoint synthesis on goal recognition.

SplatSearch w/o VCICN: Uses the NVSM to generate novel viewpoints, however, does not use multi-view diffusion to inpaint the missing regions of the images. Determines the effect of inpainting on feature matching within the FEM.

SplatSearch w/o Semantic Context Scores (SCS): Does not include SCS, namely $\overline { { V _ { t } } } ( f )$ in Equation 19. This determines the effect of semantic context scores from the value map on frontier selection.

SplatSearch w/o Visual Context Scores (VCS): Does not include VCS, namely $\mathcal { E } _ { t } ( f )$ in Equation 19. Determines the effect of visual context scores on frontier selection.

1) Results: The results for the ablation study are presented in Table II. Overall, SplatSearch achieved the highest SR and SPL across all of the variants. SplatSearch w/o VCICN had a lower SR and SPL, as the rendered images contained large missing regions, which degraded the performance of FEM to recognize the goal image. SplatSearch w/o NVSM also had a lower SR and SPL as it does not use the 3DGS map to render candidate objects from arbitrary viewpoints. SplatSearch w/o SCS achieved a lower SPL, which highlights the importance of the value map in identifying semantic relationships between objects to select frontiers. SplatSearch w/o VCS did not select frontiers that were nearby objects with high feature matches, resulting in redundant exploration. These results highlight the importance of the NVSM and VCICN for identifying goal images from arbitrary viewpoints, and the SCS and VCS for frontier selection.

TABLE II: ABLATION STUDY
<table><tr><td rowspan="2">Variants</td><td colspan="2">HM3D-val</td><td colspan="2">HM3D-val-hard</td></tr><tr><td>SR â</td><td>SPL â</td><td>SRâ</td><td>SPL â</td></tr><tr><td>SplatSearch w/o NVSM</td><td>0.6200</td><td>0.2863</td><td>0.4300</td><td>0.1712</td></tr><tr><td>SplatSearch w/o VCICN</td><td>0.6500</td><td>0.3174</td><td>0.5800</td><td>0.2204</td></tr><tr><td>SplatSearch w/o SCS</td><td>0.6400</td><td>0.3155</td><td>0.5600</td><td>0.2489</td></tr><tr><td>SplatSearch w/o VCS</td><td>0.6700</td><td>0.3407</td><td>0.5900</td><td>0.2476</td></tr><tr><td>SplatSearch</td><td>0.7000</td><td>0.3740</td><td>0.6300</td><td>0.2675</td></tr></table>

## C. Real-World Experiments

We conducted real-world experiments in a 3-room indoor space (size 35m x 14m) at the University of Toronto, Fig. 4. A Jackal robot with an onboard ZED2 stereo camera was used. No additional training was used for the new real-world deployment. We evaluated five distinct image goals to test the generalizability of SplatSearch: a person, TV, chair, backpack, and laptop â which included objects in the simulation dataset (TV, chair) and new objects not in the simulation dataset (person, backpack, laptop). We randomized the reference image viewpoints across frontal, side, and BEV perspectives, for a total of 25 trials. SplatSearch achived an SR = 0.77 and SPL = 0.48 across all trials, consistent with the simulation results. A video of our results is shown on our project webpage, https://splatsearch.github.io/.

## VII. CONCLUSION

In this paper, we presented SplatSearch, a novel architecture to address the IIN problem for mobile robots in unknown environments. SplatSearch uniquely uses a combination of sparse-view 3DGS reconstructions to synthesize candidate viewpoints around objects, and a multiview diffusion model to inpaint missing regions of the images for goal recognition across arbitrary viewpoints of candidate objects or people. A frontier policy combines visual and semantic context from onboard RGB observations to guide the robot towards the goal image. Extensive photorealistic experiments validate SplatSearchâs performance with respect to SR and SPL when compared to state-of-the-art baselines. An ablation study validated our design choices for the NVSM for novel viewpoint rendering and the VCICN for diffusionbased image inpainting in our architecture. Real-world experiments show the ability of SplatSearch to perform IIN for new and unseen objects from arbitrary viewpoints in a new environment. Future work will extend our SplatSearch architecture to consider dynamic obstacles in environments.

## REFERENCES

[1] A. A. Morgan et al., âRobots in Healthcare: a Scoping Review,â Curr Robot Rep, vol. 3, no. 4, pp. 271â280, Oct. 2022.

[2] A. Fung, A. H. Tan, H. Wang, B. Benhabib, and G. Nejat, âMLLM-Search: A Zero-Shot Approach to Finding People Using Multimodal Large Language Models,â Robotics, vol. 14, no. 8, p. 102, Jul. 2025.

[3] S. Narasimhan, A. H. Tan, D. Choi, and G. Nejat, âOLiVia-Nav: An Online Lifelong Vision Language Approach for Mobile Robot Social Navigation,â in 2025 IEEE International Conference on Robotics and Automation (ICRA), IEEE, May 2025, pp. 9130â9137.

[4] P. Bovbel and G. Nejat, âCasper: An Assistive Kitchen Robot to Promote Aging in Place1,â Journal of Medical Devices, vol. 8, no. 3, p. 030945, Sep. 2014.

[5] K. Black et al., â\$Ï_0\$: A Vision-Language-Action Flow Model for General Robot Control,â Nov. 13, 2024, arXiv: arXiv:2410.24164.

[6] J. Wu et al., âTidyBot: Personalized Robot Assistance with Large Language Models,â Auton Robot, vol. 47, no. 8, pp. 1087â1102, Dec. 2023.

[7] H. Wang, A. H. Tan, and G. Nejat, âNavFormer: A Transformer Architecture for Robot Target-Driven Navigation in Unknown and Dynamic Environments,â IEEE Robot. Autom. Lett., vol. 9, no. 8, pp. 6808â6815, Aug. 2024.

[8] J. Krantz, S. Lee, J. Malik, D. Batra, and D. S. Chaplot, âInstance-Specific Image Goal Navigation: Training Embodied Agents to Find Object Instances,â Nov. 29, 2022, arXiv: arXiv:2211.15876.

[9] X. Lei, M. Wang, W. Zhou, and H. Li, âGaussNav: Gaussian Splatting for Visual Navigation,â IEEE Trans. Pattern Anal. Mach. Intell., pp. 1â14, 2025.

[10] M. Chang et al., âGOAT: GO to Any Thing,â Nov. 10, 2023, arXiv: arXiv:2311.06430.

[11] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, âVLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation,â in 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan: IEEE, May 2024, pp. 42â48.

[12] L. Mezghan et al., âMemory-Augmented Reinforcement Learning for Image-Goal Navigation,â in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Kyoto, Japan: IEEE, Oct. 2022, pp. 3316â3323.

[13] P. Li, K. Wu, J. Fu, and S. Zhou, âREGNav: Room Expert Guided Image-Goal Navigation,â AAAI, vol. 39, no. 5, pp. 4860â4868, Apr. 2025.

[14] A. Devo, G. Mezzetti, G. Costante, M. L. Fravolini, and P. Valigi, âTowards Generalization in Target-Driven Visual Navigation by Using Deep Reinforcement Learning,â IEEE Trans. Robot., vol. 36, no. 5, pp. 1546â1561, Oct. 2020.

[15] Y. Zhu et al., âTarget-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning,â 2016, arXiv.

[16] H. Li et al., âMemoNav: Working Memory Model for Visual Navigation,â Mar. 28, 2024, arXiv: arXiv:2402.19161.

[17] A. Sridhar, D. Shah, C. Glossop, and S. Levine, âNoMaD: Goal Masked Diffusion Policies for Navigation and Exploration,â in 2024

IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan: IEEE, May 2024, pp. 63â70.

[18] D. Shah, B. Eysenbach, G. Kahn, N. Rhinehart, and S. Levine, âRapid Exploration for Open-World Navigation with Latent Goal Models,â Oct. 11, 2023, arXiv: arXiv:2104.05859.

[19] K. Yadav et al., âOVRL-V2: A simple state-of-art baseline for ImageNav and ObjectNav,â Mar. 14, 2023, arXiv: arXiv:2303.07798.

[20] Q. Wu et al., âTowards Target-Driven Visual Navigation in Indoor Scenes via Generative Imitation Learning,â IEEE Robot. Autom. Lett., vol. 6, no. 1, pp. 175â182, Jan. 2021.

[21] K. Yadav et al., âOffline Visual Representation Learning for Embodied Navigation,â Apr. 27, 2022, arXiv: arXiv:2204.13226.

[22] D. S. Chaplot, R. Salakhutdinov, A. Gupta, and S. Gupta, âNeural Topological SLAM for Visual Navigation,â May 28, 2020, arXiv: arXiv:2005.12256.

[23] X. Lei, M. Wang, W. Zhou, L. Li, and H. Li, âInstance-aware Exploration-Verification-Exploitation for Instance ImageGoal Navigation,â Mar. 22, 2024, arXiv: arXiv:2402.17587.

[24] A. Majumdar, G. Aggarwal, B. Devnani, J. Hoffman, and D. Batra, âZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings,â Oct. 13, 2023, arXiv: arXiv:2206.12403.

[25] Y. Tang et al., âOpenIN: Open-Vocabulary Instance-Oriented Navigation in Dynamic Domestic Environments,â IEEE Robot. Autom. Lett., vol. 10, no. 9, pp. 9256â9263, Sep. 2025.

[26] H. Yin et al., âUniGoal: Towards Universal Zero-shot Goal-oriented Navigation,â Mar. 18, 2025, arXiv: arXiv:2503.10630.

[27] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, â3D Gaussian Splatting for Real-Time Radiance Field Rendering,â Aug. 08, 2023, arXiv: arXiv:2308.04079.

[28] K. Honda, T. Ishita, Y. Yoshimura, and R. Yonetani, âGSplatVNM: Point-of-View Synthesis for Visual Navigation Models Using Gaussian Splatting,â Mar. 10, 2025, arXiv: arXiv:2503.05152.

[29] Y. Deng et al., âHierarchical Scoring with 3D Gaussian Splatting for Instance Image-Goal Navigation,â Jun. 10, 2025, arXiv: arXiv:2506.07338.

[30] W. Meng, T. Wu, H. Yin, and F. Zhang, âBEINGS: Bayesian Embodied Image-Goal Navigation With Gaussian Splatting,â in 2025 IEEE International Conference on Robotics and Automation (ICRA), Atlanta, GA, USA: IEEE, May 2025, pp. 5252â5258.

[31] K. Chen et al., âSLGaussian: Fast Language Gaussian Splatting in Sparse Views,â Dec. 11, 2024, arXiv: arXiv:2412.08331.

[32] W. Guo et al., âIGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation,â Aug. 01, 2025, arXiv: arXiv:2508.00823.

[33] A. Radford et al., âLearning Transferable Visual Models From Natural Language Supervision,â Feb. 26, 2021, arXiv: arXiv:2103.00020.

[34] S. Zhu, G. Wang, X. Kong, D. Kong, and H. Wang, â3D Gaussian Splatting in Robotics: A Survey,â Dec. 19, 2024, arXiv: arXiv:2410.12262.

[35] Y. Li et al., âActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting,â IEEE Robot. Autom. Lett., vol. 10, no. 8, pp. 8099â8106, Aug. 2025.

[36] N. Keetha et al., âSplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM,â Apr. 16, 2024, arXiv: arXiv:2312.02126.

[37] J. Li, D. Li, S. Savarese, and S. Hoi, âBLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models,â Jun. 15, 2023, arXiv: arXiv:2301.12597.

[38] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, âYOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors,â Jul. 06, 2022, arXiv: arXiv:2207.02696.

[39] C. Zhang et al., âFaster Segment Anything: Towards Lightweight SAM for Mobile Applications,â Jul. 01, 2023, arXiv: arXiv:2306.14289.

[40] C. Cao, C. Yu, F. Wang, X. Xue, and Y. Fu, âMVInpainter: Learning Multi-View Consistent Inpainting to Bridge 2D and 3D Editing,â Nov. 19, 2024, arXiv: arXiv:2408.08000.

[41] S. Garrido, L. Moreno, and D. Blanco, âExploration of a cluttered environment using Voronoi Transform and Fast Marching,â Robotics and Autonomous Systems, vol. 56, no. 12, pp. 1069â1081, Dec. 2008.

[42] K. Yadav et al., âHabitat-Matterport 3D Semantics Dataset,â Oct. 12, 2023, arXiv: arXiv:2210.05633.

[43] S. Zhang et al., âEgoBody: Human Body Shape and Motion of Interacting People from Head-Mounted Devices,â in Computer Vision â ECCV 2022, vol. 13666, S. Avidan, G. Brostow, M. CissÃ©, G. M. Farinella, and T. Hassner, Eds., in Lecture Notes in Computer Science, vol. 13666. , Cham: Springer Nature Switzerland, 2022, pp. 180â200.