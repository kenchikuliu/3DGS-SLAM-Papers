# GSplatVNM: Point-of-View Synthesis for Visual Navigation Models Using Gaussian Splatting

Kohei Honda1,2, Takeshi Ishita1, Yasuhiro Yoshimura1, Ryo Yonetani1

Abstractâ This paper presents a novel approach to imagegoal navigation by integrating 3D Gaussian Splatting (3DGS) with Visual Navigation Models (VNMs), a method we refer to as GSplatVNM. VNMs offer a promising paradigm for image-goal navigation by guiding a robot through a sequence of point-of-view images without requiring metrical localization or environment-specific training. However, constructing a dense and traversable sequence of target viewpoints from start to goal remains a central challenge, particularly when the available image database is sparse. To address these challenges, we propose a 3DGS-based viewpoint synthesis framework for VNMs that synthesizes intermediate viewpoints to seamlessly bridge gaps in sparse data while significantly reducing storage overhead. Experimental results in a photorealistic simulator demonstrate that our approach not only enhances navigation efficiency but also exhibits robustness under varying levels of image database sparsity.

## I. INTRODUCTION

Efficient robot navigation relies on the availability of sufficient environmental information; however, the associated data collection costs cannot always be justified. For example, imagine an exhibition hall or retail store. The appearance and structure of such environments may change day by day, necessitating continuous monitoring and updates. Moreover, regular site surveys can only be conducted during the limited hours when the facility is closed. Therefore, reducing overall data collection costs is essential for deploying robots in these settings.

Vision-based navigation offers a promising solution. In particular, recent advances in image-to-waypoints Visual Navigation Models (VNMs) [1]â[3] have demonstrated zeroshot navigation capabilities without precise self-localization using dense point clouds [4], [5], thereby reducing the need for extensive data collection compared to conventional localization-based navigation frameworks. Nevertheless, these approaches have not completely eliminated the need for data collection; VNMs plan a path based on the robotâs point-of-view images of the goal, and a continuous sequence of such imagesâfrom the start to the goalâis necessary for long-range navigation. Existing methods construct an Image-based Topological Graph (ITG) to describe the traversability between randomly sampled images in the environment [6], which necessitates collecting a sufficient number of images to ensure robust navigation performance.

<!-- image-->  
Fig. 1. GSplatVNM employs 3DGS as a compact and renderable environment representation for the VNM, which imagines future point-of-view images to efficiently navigate the robot from the given start to the goal.

In this work, we propose GSplatVNM, a new visionbased navigation framework that requires reduced data collection. As illustrated in Fig. 1, we integrate VNMs with 3D Gaussian splatting (3DGS) [7] as a compact environment representation. 3DGS is a neural model that enables high-quality 3D reconstruction of the environment from a pre-collected image database (DB) and can further synthesize novel images for arbitrary viewpoints not present in the original database. We leverage these capabilities of 3DGS to localize the start and goal poses from a given pair of images and connect them by synthesizing a continuous sequence of traversable point-of-view images. These images are then used by the VNM to control the robot.

We validate the proposed GSplatVNM framework in photorealistic simulation environments [8], [9]. Our experiments demonstrate that GSplatVNM outperforms conventional ITG-based methods [3], [10] in terms of navigation efficiency and robustness, particularly when using sparse image databases. Notably, GSplatVNM can even navigate to a point-of-view that has been seen but not visited, a task that has proven difficult for ITG-based methods. These results demonstrate that GSplatVNM can significantly reduce data collection costs while maintaining a continuous set of feasible target point-of-view images.

## II. RELATED WORK

Since this paper proposes a novel environment representation for VNMs as an alternative to ITG, we first provide an overview of ITG, which is a commonly used environment representation for visual navigation. We then review recent studies that use neural rendering models as a prior map for visual navigation, which are closely related to our proposed method.

## A. ITG-based Visual Navigation

As shown in Fig. 2, ITG is a graph representation of the environment, where each node represents a point-of-view image and edges encode traversability [6]. Unlike point cloud maps, ITG does not contain absolute position information; it is defined by the images and the relative relationships between them. Typically, the robot localizes itself within the ITG and navigates by following a path computed using graph search algorithms [11]. Because the robot only needs to estimate its current node in the ITG, high-precision localization and mapping are not required, which enhances portability and scalability. Furthermore, ITG-based visual navigation has demonstrated strong generalization performance [11] to unseen environments compared to pure learning-based approaches that do not leverage prior knowledge of the environment [9], [12].

## B. Construction of Environment-Covering ITG

Although ITG-based environment representations are a promising approach for visual navigation, they have a critical limitation: while the robot moves continuously in space, the ITG is represented as a discrete graph. As a result, the ITG alone may provide a sparse and insufficient spatial representation, potentially degrading navigation performance. For example, the start and goal images may not always be present in the ITG, and suboptimal connectivity estimates between nodes can lead the robot to take unnecessarily long paths. Therefore, building an ITG that efficiently covers the environment is a challenging task.

One straightforward approach to constructing an ITG is to physically collect images through robot exploration and then link nodes that are known to be physically traversable [13], [14]. However, this approach requires extensive sequences of images across the environment. Other methods relax this physical requirement by estimating traversability between nodes based on image similarity, thereby allowing edges between nodes that have not been observed through direct physical transitions [10], [11], [15]. Although this approach reduces the need for a continuous image sequence, it often struggles in scenarios where the collected image database is spatially sparse, making it difficult to generate a feasible and efficient view sequence to reach the goal. To mitigate this issue, some studies have proposed online updating and expansion of the ITG using simultaneous localization and mapping techniques [16], [17] as well as neural image generation models [1], [18]. However, as the number of nodes increases, the boundary between ITG and metric maps becomes blurred, and the computational and storage costs increase significantly.

## C. Visual Navigation with Neural Rendering Models

To overcome the limitations of ITGs, a promising direction is to use neural rendering models as a prior map for visual navigation. Some studies generate target point-ofview images online using world models [19], [20], but these methods require significant computation and lack geometric information about the environment. In contrast, Neural Radiance Fields (NeRF) [21] and 3DGS [22] provide geometric information and can quickly generate novel point-of-view images, making them well-suited for navigation tasks. In particular, 3DGS relies on lightweight 3D Gaussians as its physical representation, which have been used as a geometric prior map for visual navigation [22]â[25].

While such recent studies [22]â[25] also use 3DGS for navigation, their approaches typically require online and presize self-localization within the 3DGS map. In contrast, our method uses 3DGS as an offline environment model solely to synthesize a sequence of target viewpoints. A powerful, pre-trained VNM then follows these images without needing to explicitly track its pose against the map. Our core contribution is therefore the integration of 3DGS as a viewpoint generator to guide a localization-free policy, rather than using it as a map for online localization.

## III. 3DGS AS ENVIRONMENT REPRESENTATION

## A. Representation by 3D Gaussians

3DGS is an advanced technique for the reconstruction and rendering of environments. Instead of using traditional mesh or voxel-based approaches, environments are encoded as a set of 3D Gaussians, each associated with color, opacity, and covariance parameters. An environment is represented by 3D Gaussians, denoted as $\{ ( \mu _ { i } , \Sigma _ { i } , \mathbf { c } _ { i } , \alpha _ { i } ) \} _ { i = 1 } ^ { N } ,$ where N is the number of Gaussians, $\mu _ { i } \in \mathbb { R } ^ { 3 }$ is the mean, $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ is the covariance matrix, $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ is the color, and $\alpha _ { i } \in [ 0 , 1 ]$ is the opacity of the i-th Gaussian. Each Gaussian can be expressed as a continuous function over the 3D position space $\mathbf { x } \in \mathbb { R } ^ { 3 }$ as follows: $G _ { i } ( \mathbf { x } ) = \mathrm { e x p } \left( - \textstyle { \frac { 1 } { 2 } } ( \mathbf { x } - { \boldsymbol { \mu } } _ { i } ) ^ { \top } { \boldsymbol { \Sigma } } _ { i } ^ { - 1 } ( \mathbf { x } - { \boldsymbol { \mu } } _ { i } ) \right)$ . To render a view from a given camera pose, we splat the Gaussians onto the 2D image plane and determine the pixel colors using $\mathbf { c } _ { i }$ and $\alpha _ { i }$ . By training the parameters of 3DGS to minimize a reprojection error across multiple views, we obtain a compact and renderable representation of the environment.

## B. Distance Estimation between Robot and 3DGS

3DGS can not only render novel views but also be employed to represent the geometry of the environment [22]â [24]. In this work, following the approach proposed in [24], we convert the 3DGS into a set of ellipsoids $\mathcal { E } _ { i }$ to estimate the distance between the robot and the environment, where each ellipsoid $\mathcal { E } _ { i }$ is defined as $\mathcal { E } _ { i } = \{ \mathbf { x } \in \mathbb { R } ^ { 3 } \mid ( \mathbf { x } - { \boldsymbol { \mu } } _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( \mathbf { x } -$ $\mu _ { i } ) \leq 1 \}$ }. Thus, the ellipsoid-to-sphere distance between the i-th ellipsoid and the robot centroid $\widehat { \mathbf { p } } = F \pmb { R } ^ { \top } ( \mathbf { p } - \mu _ { i } )$ can be formulated as the following quadratic program:

$$
d _ { i } ( \mathbf { p } ) = \operatorname* { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { 3 } } \left\| \mathbf { x } - \hat { \mathbf { p } } \right\| ^ { 2 } , \quad \mathrm { s . t . } \mathbf { x } ^ { \top } S ^ { - 2 } \mathbf { x } \leq 1 ,\tag{1}
$$

where $\mathbf { p } \in \mathbb { R } ^ { 3 }$ is the robot 3D position approximated by a sphere, F is a reflection matrix (a diagonal matrix with entries of either 1 or â1 used to reflect points across the primary planes), R is the rotation matrix of the ellipsoid, and S is a scaling matrix determined by the confidence of the Gaussian, which serves as a hyperparameter.

<!-- image-->  
Fig. 2. Overview of the proposed GSplatVNM. In a conventional ITG-based approach, the environment is represented by ITG, and the target point-of-views given to the VNM (NoMaD [3]) are limited to a set of collected image DB. On the other hand, GSplatVNM represents the environment as 3DGS to synthesize dense and traversable point-of-view images from the start to the goal, thereby guiding NoMaD. As a result, GSplatVNM can achieve efficient image-goal navigation by seamlessly bridging gaps in the image DB, even when the image DB is sparse.

## IV. VISUAL NAVIGATION WITH 3DGS

Our aim is to achieve efficient image-goal navigation with VNMs using a compact and spatially aware representation of the navigation environment. Our proposed framework uses 3DGS as a prior map for the VNM to reduce the number of images that need to be collected beforehand.

## A. Overview of the Proposed Framework

Figure 2 shows an overview of our proposed navigation framework integrating 3DGS with a VNM. Given the start and goal images, we first estimate the robotâs start and goal poses in 3DGS and then plan a global trajectory within the 3DGS representation. We subsequently render target pointof-view images along the planned trajectory to condition the VNM, i.e., NoMaD [3], which is a state-of-the-art VNM. During navigation, NoMaD generates spatial waypoints from the observation images and the synthesized target point-of-view images, and the robot follows these waypoints using Model Predictive Path Integral control (MPPI) [26].

## B. Start and Goal Pose Estimation in 3DGS

In the proposed framework, we first estimate the start and goal poses in 3DGS from the given start and goal images. We assume that the robot moves on a 2D plane; hence, the robotâs pose is represented as $\mathbf { q } = ( x , y , \theta _ { \mathrm { y a w } } ) \in \mathrm { S E } ( 2 )$ , where x and y denote the 2D position and $\theta _ { \mathrm { y a w } }$ the yaw angle. Although some studies have addressed accurate and online pose estimation in 3DGS [22], we require only a one-shot, rough estimation of the global pose.

In this work, we optimize the global start and goal poses, qstart and $\mathbf { q } _ { \mathrm { g o a l } } .$ , by minimizing the following loss function $\mathcal { L } \colon$

$$
\begin{array} { r l } & { \mathcal { L } ( \mathbf { q } _ { * } ) = \mathcal { L } _ { \mathrm { i m g } } ( I _ { * } , I _ { \mathrm { r e n d e r e d } } ( \mathbf { q } _ { * } ) ) + \mathbf { 1 } _ { \mathrm { c o l l i s i o n } } [ d _ { \mathrm { m i n } } ( \mathbf { q } _ { * } ) \leq r ] , } \\ & { d _ { \mathrm { m i n } } ( \mathbf { q } _ { * } ) = \displaystyle \operatorname* { m i n } _ { i \in \{ 0 , \dots , N - 1 \} } d _ { i } ( \mathbf { q } _ { * } ) , \quad * \in \{ \mathrm { s t a r t } , \mathbf { g } \mathrm { o a l } \} , } \end{array}\tag{2}
$$

where $I _ { * }$ is the given start or goal image, $I _ { \mathrm { r e n d e r e d } } ( \mathbf { q } _ { * } )$ is the image rendered from 3DGS at pose qâ, and r is the radius of the robot. The first term, ${ \mathcal { L } } _ { \mathrm { i m g } } ,$ in (2) is an image similarity metric between the given and rendered images from 3DGS at the pose qâ. Specifically, we use the Learned Perceptual

Image Patch Similarity (LPIPS) metric [27], which is computed from the feature maps of AlexNet [28] and ranges from 0 to 1. The second term is a collision penalty to avoid the infeasibility of global planning. Here, $\mathbf { 1 } _ { \mathrm { c o l l i s i o n } }$ denotes an indicator function, and $d _ { i }$ is the distance between the robot and the i-th ellipsoid-approximated 3DGS defined in (1).

To render the image and calculate the distance, in addition to the robotâs 2D pose $\mathbf { q } _ { * }$ , the z position $z _ { * } .$ , roll $\theta _ { \mathrm { r o l l } } ^ { * } .$ , and pitch $\theta _ { \mathrm { p i t c h } } ^ { * }$ angles are required. We assume that the image DB was captured with fixed $( z , \theta _ { \mathrm { r o l l } } , \theta _ { \mathrm { p i t c h } } )$ values, and we use the mean of those values obtained from the estimated camera poses during the training of 3DGS.

Although the loss function (2) is theoretically differentiable, its complexity may lead to convergence at local optima. Given that the optimization is over a three-dimensional space, we employ a black-box optimization method known as the Tree-structured Parzen Estimator [29]. The entire optimization process is implemented using Optuna [30].

## C. Global Trajectory Planning in 3DGS

We then plan a traversable global trajectory in 3DGS from the estimated start to goal poses, $\mathbf { q } _ { \mathrm { s t a r t } }$ and $\mathbf { q } _ { \mathrm { g o a l } }$ . In this work, considering both the guarantee of optimality and ease of implementation, we plan a global path $\{ ( x _ { \mathrm { s t a r t } } , y _ { \mathrm { s t a r t } } ) , ( x ^ { 0 } , y ^ { 0 } ) , \ldots , ( x _ { M - 1 } , y _ { M - 1 } ) , ( x _ { \mathrm { g o a l } } , y _ { \mathrm { g o a l } } ) \}$ using $\mathbf { A } ^ { * }$ search [31] without considering yaw angles, and then assigning yaw angles along the path. That is, we compute the traversable trajectory $\mathbb { T }$ as follows:

$$
\mathcal { T } = \{ \mathbf { q } _ { \mathrm { s t a r t } } , ( x _ { 0 } , y _ { 0 } , \theta _ { \mathrm { y a w } } ^ { 0 } ) , \dots , ( x _ { M - 1 } , y _ { M - 1 } , \theta _ { \mathrm { y a w } } ^ { M - 1 } ) , \mathbf { q } _ { \mathrm { g o a l } } \} ,\tag{3}
$$

$$
\begin{array} { r l } & { \big \{ ( x _ { 0 } , y _ { 0 } ) , \dotsc , ( x _ { M - 1 } , y _ { M - 1 } ) \big \} = \mathbf { A } ^ { * } ( \mathbf { q } _ { \mathrm { s t a r t } } , \mathbf { q } _ { \mathrm { g o a l } } ) , } \\ & { \theta _ { \mathrm { y a w } } ^ { i } = \mathrm { a t a n } 2 ( y _ { i + 1 } - y _ { i } , x _ { i + 1 } - x _ { i } ) , \quad i \in \{ 0 , \dotsc , M - 1 \} . } \end{array}
$$

Here, M denotes the number of poses in the trajectory. $\mathbf { A } ^ { * }$ search considers collisions between the robot and the 3DGS as well as the loss function (2). Although the above method guarantees optimality in terms of distance from the start to the goal position, the yaw angles do not account for the robotâs dynamics or NoMaDâs tracking capability, which may result in trajectories that are not always trackable. We also experimented with Hybrid $\mathbf { A } ^ { * }$ search [32], which can account for the differential two-wheel model, but we found that its performance was not significantly different from (3). Additionally, since 3DGS operates on a different scale than the physical world, it is challenging to accurately account for kinodynamics.

After the global planning, we render a set of the target point-of-views $\mathcal { V } _ { r } = \{ I _ { 0 } ^ { r } ( \mathbf { q } _ { 0 } ) , \ldots , I _ { M - 1 } ^ { r } ( \mathbf { q } _ { M - 1 } ) , I _ { \mathrm { g o a l } } \}$ along the planned trajectory T using 3DGS. These rendered images are then used to condition the NoMaD policy to generate spatial waypoints.

## D. Zero-shot Local Planning and Control with NoMaD

NoMaD [3] is a visual subgoal-conditioned policy that generates spatial waypoints from a sequence of observation images at time (t), $\mathcal { O } _ { \mathrm { o b s } } = \{ I _ { t - p } , \ldots , I _ { t } \}$ (with p=3 in No-MaD), as well as from the target subgoal point-of-view image $I _ { \mathrm { t a r g e t } }$ . NoMaD consists of three networks:

â¢ A subgoal image-conditioned vision encoder, $\mathbf { c } _ { t } ~ =$ $f _ { \mathrm { e n c } } ( \mathcal { O } _ { \mathrm { o b s } } , I _ { \mathrm { t a r g e t } } )$ , that extracts context features from the observation $\mathrm { \odot _ { o b s } }$ and target subgoal image $I _ { \mathrm { t a r g e t } } .$

â¢ A diffusion model [33]-based policy, $\pi ( \mathbf { c } _ { t } )$ , that generates a set of waypoints ${ \mathcal W } _ { t } = \{ \mathbf w _ { 0 } , . . . , \mathbf w _ { L - 1 } \}$ (with L=8 in NoMaD) from the encoded features ct, where each waypoint represents a 2D physically-scaled position in the robotâs coordinate frame.

â¢ A distance-estimation network, $\phi \left( \mathbf { c } _ { t } \right)$ , that predicts the physical distance between the current image position and the target image position.

These networks are trained in a supervised manner on large, heterogeneous datasets collected across a diverse set of environments and robotic platforms over 100 hours, resulting in zero-shot navigation capability in unseen environments and on unseen robot setups [3].

In our navigation scheme, after synthesizing a set of target point-of-view images toward the goal, $\mathcal { V } _ { r } .$ , we first estimate the nearest image $I _ { t } ^ { r } \in \mathcal { V } _ { r }$ to the current observation. This estimation is performed using the distance-estimation network, $\phi ,$ which is conditioned on both the current observation $\mathrm { \mathcal { O } o b s }$ and the target images $I _ { i } ^ { r } \in \mathcal { V } _ { r }$ (with $I _ { 0 } ^ { r } = I _ { \mathrm { s t a r t } } )$ . Then, using the subsequent image in the sequence, $I _ { t + 1 } ^ { r }$ , as the next subgoal, we generate point-of-view-conditioned waypoints ${ \mathcal { W } } _ { t }$ according to the policy $\pi ( \mathbf { c } _ { t } )$ , where $\mathbf { c } _ { t } = f _ { \mathrm { e n c } } ( \mathcal { O } _ { \mathrm { o b s } } , I _ { t + 1 } ^ { r } )$ . The robot follows these waypoints using MPPI [26], a samplingbased model predictive control method that minimizes future deviations from the waypoints while taking into account the robotâs differential two-wheel dynamics.

## V. EXPERIMENTS

We comprehensively evaluate the proposed GSplatVNM on photorealistic 3D indoor environments using the AI Habitat simulator platform [9]. In our experiments, we compare the proposed method with conventional methods in terms of success rate, path efficiency, and robustness with respect to the number of pre-collected images in the image DB.

## A. Simulation Setup

1) Robot Setup: We simulate a circular wheeled robot (radius: 0.5 m) that navigates the environment using the Habitat simulator API, with state updates every 0.5 seconds. The action space comprises continuous linear and angular velocities, with a maximum linear velocity of 0.25 m/s and a maximum angular velocity of 0.5 rad/s. The robot is equipped with a monocular camera mounted at a height of 0.4 m from the ground. The camera has a resolution of 1280Ã720 pixels and a field of view of 120 degrees.

2) Simulation Environments: To evaluate performance across different environment sizes, we conduct experiments in three indoor environments: Greigsville (small) and Ribera (medium) from the Gibson dataset [8], and skokloster-castle (large) from the Habitat Test dataset1. Their sizes are 43.6, 50.6, and approximately 190 $\mathrm { m } ^ { 2 } .$ , respectively.

## B. Experiment Setup

1) Image-Goal Navigation Task: The task is image-goal navigation, where the robot must move from a start pose to a goal position at which the given goal image is visible, while remaining within the traversable area provided by the simulator. A trial is considered successful if the robot reaches the goal within a specified distance (0.5 m for Greigsville and Ribera, and 1.0 m for skokloster-castle) within 500 steps (i.e., 250 s).

In our experiments, we assume that the robot is equipped with a collision avoidance system independent of NoMaD. Consequently, the simulator restricts the robot from leaving the traversable area, and collision avoidance performance is not evaluated2.

2) Pre-Collection of Image DB: For the imagegoal navigation task, the robot utilizes a pre-collected image DB as prior knowledge of the environment. To evaluate robustness with respect to the size of the image DB, we prepare multiple image DBs for each environment. Specifically, we collect {300, 500, 1000} images for Greigsville, {500,800,1000} images for Ribera, and {300,1000,2000,3000} images for skokloster-castle. Before the navigation task, the robot explores the environment using an exploration policy based on NoMaD [3] and collects images at regular intervals of 0.5 s until these numbers of images are collected.

3) Evaluation Metrics: We evaluate navigation performance using the Success weighted by Path Length (SPL) metric [34], which is commonly used to assess the efficiency of navigation tasks. SPL is defined as the ratio of the shortest path length to the path length taken by the robot, weighted by the success rate. The definition of SPL is as follows:

$$
\mathrm { S P L } = \frac { 1 } { N _ { \mathrm { t r i a l s } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t r i a l s } } } { \bf 1 } _ { \mathrm { s u c c e s s } } \frac { L _ { i } } { \operatorname* { m a x } ( L _ { i } , L _ { \mathrm { m i n } } ) } ,\tag{4}
$$

<!-- image-->  
(a) Greigsville (small)

<!-- image-->  
(b) Ribera (medium)

<!-- image-->  
(c) skokloster-castle (large)  
Fig. 3. Comparison of the SPL on each environment.

where $N _ { \mathrm { t r i a l s } }$ is the number of trials, $L _ { i }$ is the path length of the i-th trial, $L _ { \mathrm { m i n } }$ is the shortest path length, and ${ \bf 1 } _ { \mathrm { s u c c e s s } }$ is an indicator function that returns 1 if the robot reaches the goal. The SPL ranges from 0 to 1, with 1 indicating that the robot reached the goal via the shortest possible path. The shortest path length is computed using the Dijkstra algorithm on the traversable area map provided by the simulator.

## C. Implementation Details

1) 3DGS Construction: We construct the 3DGS for each image DB using the Splatfacto implementation in Nerfstudio [35], with the initial Gaussian means provided by a point cloud generated by COLMAP [36]. Note that the robotâs radius used in (1) is set to 0.01 in the 3DGS due to the scale difference between the 3DGS and the actual environment.

2) Baseline Methods: To evaluate the effectiveness of our 3DGS-based target point-of-view generation for NoMaD, we compare our approach with the following baseline methods:

â¢ NoMaD w/o ITG: NoMaD is conditioned solely on physically reachable point-of-view images [3]. Specifically, the sequence of images traversed (either in order or in reverse) during the image collection, from the estimated start image to the goal image, is used as the sequence of target point-of-views.

â¢ NoMaD w/ ITG: NoMaD is conditioned on the shortest target point-of-view sequence computed on a preconstructed ITG using the Dijkstra algorithm. The ITG is built from an image DB by connecting nodes based on image similarity; two nodes are connected if the estimated distance Ï between them is within a specified threshold (5 m) [10] (see Section IV-D for details).

In both cases, the start and goal nodes are determined from the image DB as those with the smallest estimated distance $\phi$ to the given start and goal images.

Our goal is to specifically assess the impact of the underlying environment representation (3DGS vs. ITG) on a stateof-the-art zero-shot navigation policy, rather than conducting a broad comparison of different navigation policies such as GNM [1] or ViNT [2]. Therefore, we utilize variants of the same NoMaD policy for all methods to ensure a fair comparison of the representation itself.

3) Configurations: For all methods, including the proposed GSplatVNM, we utilize the same image DB and pretrained NoMaD weights provided by the authors [3]. Note that the pre-trained datasets do not include data from the AI Habitat simulator.

## D. Experimental Results

1) Comparisons of the Navigation Efficiency: Figure 3 presents the SPL distribution over $N _ { \mathrm { t r i a l s } } = 2 0$ trials for each environment and image DB size. Overall, the proposed GSplatVNM achieves higher SPL values compared to the baseline methods across all environments and image DB sizes.

We observe that the baseline methods exhibit varying trends depending on the image DB size. For NoMaD w/o ITG, the SPL is generally lower than that of the other methods, particularly when the image DB is sufficiently large, because it merely traverses the pre-collected image sequence, leading to frequent detours. Interestingly, when the image DB is very sparse, SPL can be higher than in denser cases, as an untrackable sequence of point-ofviews may cause the robot to deviate from the intended path, sometimes resulting in a fortunate shortcut. In contrast, NoMaD w/ ITG shows significant degradation in SPL as the image DB size decreasesâespecially in the Ribera and skokloster-castle environmentsâdue to a reduced number of nodes in the ITG, which may lead to cases where the start and goal images are absent or detours occur.

In contrast, GSplatVNM demonstrates robustness with respect to the image DB size in terms of SPL. Specifically, for image DB sizes of 500 and 1000 in Greigsville, 500, 800, and 1000 in Ribera, and 1000, 2000, and 3000 in skokloster-castle, GSplatVNM achieves consistently high SPL values. For very small image DBs (300 images for Greigsville and skokloster-castle), SPL decreases slightly due to a decline in the quality of images rendered by 3DGS, which adversely affects pose estimation accuracy and tracking performance of NoMaD.

2) Qualitative Comparisons: Figure 4 shows selected navigation results for both the baseline methods and GSplatVNM. In Figures 4a and 4c, GSplatVNM successfully interpolates point-of-view images in the gaps of the image collection trajectory, allowing the robot to reach the goal along a shorter path compared to the baselines. In Figure 4b (3), the goal is located in a bottom-left room that is inaccessible using only the image DB, as the robot did not enter that room during image collection. In this case, while the baseline methods failed, GSplatVNM succeeded by synthesizing the missing point-of-view images for areas that were observed but not visited.

<!-- image-->  
(1) Trajectories of Image Collection

(2) Navigation Result  
<!-- image-->  
(3) Navigation Result

(a) A simulation result on Greigsville (image DB size: 500)  
<!-- image-->  
(1) Trajectories of Image Collection  
(2) Navigation Result

(b) A simulation result on Ribera (image DB size: 800)  
<!-- image-->  
(c) A simulation result on skokloster-castle (image DB size: 1000)  
Fig. 4. Trajectories of the image collection and selected navigation results for each environment. GSplatVNM can generate point-of-view images that are not included in the pre-collected image DB, enabling the robot to reach the goal via a shorter path than the baseline methods. Notably, GSplatVNM is able to reach a goal located in a room that was not visited during image collection (see Fig. 4b (3)).

3) Performance Analysis of Individual Modules: Table I summarizes the overall navigation success rate (SR) along with the success rates of individual modules: the Pose Estimation Success Rate (PE-SR) for the start and goal poses, and the target point-of-view Following Success Rate (VF-SR). The PE-SR reflects the accuracy of the estimated start and goal poses (manually verified), while the VF-SR measures whether NoMaD successfully reaches the final node in the sequence of target point-of-view images, indicating the ease of following by NoMaD.

TABLE I  
SUCCESS RATE OF THE NAVIGATION TASK AND INDIVIDUAL MODULES
<table><tr><td>Env. ID</td><td>Method</td><td>SR</td><td>PE-SR</td><td>VF-SR</td></tr><tr><td rowspan="3">Greigsville</td><td>NoMaD w/o ITG</td><td>0.47</td><td>0.27</td><td>0.22</td></tr><tr><td>NoMaD w/ ITG</td><td>0.42</td><td>0.27</td><td>0.48</td></tr><tr><td>GSplatVNM</td><td>0.55</td><td>0.39</td><td>0.38</td></tr><tr><td rowspan="3">Ribera</td><td>NoMaD w/o ITG</td><td>0.30</td><td>0.34</td><td>0.05</td></tr><tr><td>NoMaD w/ ITG</td><td>0.27</td><td>0.34</td><td>0.51</td></tr><tr><td>GSplatVNM</td><td>0.40</td><td>0.47</td><td>0.33</td></tr><tr><td rowspan="3">skokloster-castle</td><td>NoMaD w/o ITG</td><td>0.34</td><td>0.20</td><td>0.18</td></tr><tr><td>NoMaD w/ ITG</td><td>0.39</td><td>0.20</td><td>0.59</td></tr><tr><td>GSplatVNM</td><td>0.59</td><td>0.62</td><td>0.37</td></tr></table>

Across all environments, GSplatVNM consistently achieves the highest SR, with particularly notable performance in the skokloster-castle environment, where its SR is 20% higher than that of NoMaD w/ ITG. Additionally, GSplatVNM exhibits a higher PE-SR than the baseline methods, demonstrating its ability to optimize point-of-view images despite lower image similarity on the 3DGS. Notably, in the skokloster-castle environment, where the image DB has limited spatial coverage, GSplatVNM achieves a high PE-SR by effectively interpolating point-of-view images. However, regarding VF-SR, NoMaD w/ ITG attains the highest success rates in the Ribera and skokloster-castle environments. This is because the point-of-view images generated by ITG are closer to the actual traversed views, facilitating easier tracking by NoMaD. Improving NoMaDâs tracking performance along these generated views represents a promising direction for future research.

TABLE II  
COMPARISON OF STORAGE USAGE (SKOKLOSTER-CASTLE)
<table><tr><td>Image Num.</td><td>ITG Storage (MB)</td><td>3DGS Storage (MB)</td></tr><tr><td>300</td><td>322</td><td>508</td></tr><tr><td>1000</td><td>840</td><td>530</td></tr><tr><td>2000</td><td>2300</td><td>442</td></tr><tr><td>3000</td><td>3400</td><td>223</td></tr></table>

4) Storage Usage Comparison: Table II compares the storage usage of ITG and 3DGS for various image DB sizes in the skokloster-castle environment. The ITG requires significantly more storage than 3DGS, with its storage usage increasing linearly with the number of images in the DB. In contrast, the storage usage of 3DGS is independent of the number of images, as it compresses image information into a compact representation based on Gaussians; thus, the storage depends solely on the number of Gaussians. These results confirm that 3DGS is a more storage-efficient representation for VNMs.

## VI. CONCLUSION

This paper proposes a novel environment representation for VNMs using 3DGS instead of the traditional ITG. With 3DGS, spatially arbitrary point-of-view images can be generated between the start and the goal, enabling high navigation performance even when ITG lacks sufficient spatial coverage. As a result, the robot is capable of reaching locations that have been observed but not previously visited. Furthermore, 3DGS offers improved storage efficiency compared to ITG, which is particularly advantageous for robotics applications.

Despite these promising results, our work has several limitations:

â¢ Real-World Validation: The current validation is confined to simulation environments. A crucial next step is to conduct experiments on a physical robot to evaluate the sim-to-real transferability and robustness of our framework under real-world conditions.

â¢ Planar Motion Assumption: Our framework assumes the robot operates on a 2D plane, making it unsuitable for environments with significant elevation changes, such as ramps or multiple floors. While our approach could theoretically be extended by incorporating 3D pose estimation and global planning, we leave this extension for future work.

â¢ Collision Checking with 3DGS: The collision checking in our method lacks geometric precision. 3DGS is optimized for rendering quality rather than metrically accurate reconstruction, which can cause a scale discrepancy between the model and the physical world. Consequently, the robotâs radius for collision checking within the 3DGS space is treated as a hyperparameter in our experiment. However, this limitation is mitigated by the fact that the VNM policy follows rendered target images instead of precise world coordinates, reducing the criticality of exact geometric collision checking.

Looking forward, several technical challenges remain to be addressed. One is improving the tracking performance of the VNM on the generated point-of-view images. Potential approaches include fine-tuning the VNM itself [2] or replanning the global trajectory [37]. Another challenge is handling dynamic obstacles. Although the current VNM can avoid obstacles, online reconstruction of 3DGS for dynamic environments [38] could provide a more robust solution. Finally, enhancing the global position estimation accuracy within the 3DGS representation is important. Since image similarity-based methods can sometimes confuse visually similar locations, incorporating instance segmentation [23], [39] may help overcome this limitation.

## REFERENCES

[1] D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine, âGNM: A general navigation model to drive any robot,â in International Conference on Robotics and Automation. IEEE, 2023, pp. 7226â 7233.

[2] D. Shah, A. Sridhar, N. Dashora, K. Stachowicz, K. Black, N. Hirose, and S. Levine, âViNT: A foundation model for visual navigation,â in Conference on Robot Learning. PMLR, 2023, pp. 711â733.

[3] A. Sridhar, D. Shah, C. Glossop, and S. Levine, âNoMaD: Goal masked diffusion policies for navigation and exploration,â in International Conference on Robotics and Automation. IEEE, 2024, pp. 63â70.

[4] J. Thoma, D. P. Paudel, A. Chhatkuli, T. Probst, and L. V. Gool, âMapping, localization and path planning for image-based navigation using visual features and map,â in Conference on Computer Vision and Pattern Recognition. IEEE/CVF, 2019, pp. 7383â7391.

[5] O. Kwon, J. Park, and S. Oh, âRenderable neural radiance map for visual navigation,â in Conference on Computer Vision and Pattern Recognition. IEEE/CVF, 2023, pp. 9099â9108.

[6] F. Fraundorfer, C. Engels, and D. Nister, âTopological mapping, Â´ localization and navigation using image collections,â in International Conference on Intelligent Robots and Systems. IEEE/RSJ, 2007, pp. 3872â3877.

[7] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, pp. 139â1, 2023.

[8] F. Xia, A. R. Zamir, Z. He, A. Sax, J. Malik, and S. Savarese, âGibson Env: real-world perception for embodied agents,â in Computer Vision and Pattern Recognition. IEEE/CVF, 2018.

[9] Manolis Savva\*, Abhishek Kadian\*, Oleksandr Maksymets\*, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik, D. Parikh, and D. Batra, âHabitat: A Platform for Embodied AI Research,â in International Conference on Computer Vision. IEEE/CVF, 2019.

[10] D. Shah, B. Osinski, S. Levine, Â´ et al., âLM-Nav: Robotic navigation with large pre-trained models of language, vision, and action,â in Conference on Robot Learning. PMLR, 2023, pp. 492â504.

[11] N. Savinov, A. Dosovitskiy, and V. Koltun, âSemi-parametric topological memory for navigation,â in International Conference on Learning Representations, 2018.

[12] E. Wijmans, A. Kadian, A. Morcos, S. Lee, I. Essa, D. Parikh, M. Savva, and D. Batra, âDD-PPO: Learning near-perfect pointgoal navigators from 2.5 billion frames,â in International Conference on Learning Representations, 2019.

[13] D. Shah and S. Levine, âViKiNG: Vision-Based Kilometer-Scale Navigation with Geographic Hints,â in Proceedings of Robotics: Science and Systems, 2022.

[14] D. Shah, B. Eysenbach, N. Rhinehart, and S. Levine, âRapid exploration for open-world navigation with latent goal models,â in Conference on Robot Learning. PMLR, 2022, pp. 674â684.

[15] D. Shah, B. Eysenbach, G. Kahn, N. Rhinehart, and S. Levine, âViNG: Learning open-world navigation with visual goals,â in International Conference on Robotics and Automation. IEEE, 2021, pp. 13 215â 13 222.

[16] R. R. Wiyatno, A. Xu, and L. Paull, âLifelong topological visual navigation,â IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 9271â9278, 2022.

[17] D. S. Chaplot, R. Salakhutdinov, A. Gupta, and S. Gupta, âNeural topological slam for visual navigation,â in Conference on Computer Vision and Pattern Recognition. IEEE/CVF, 2020, pp. 12 875â12 884.

[18] X. Cui, Q. Liu, Z. Liu, and H. Wang, âFrontier-enhanced topological memory with improved exploration awareness for embodied visual navigation,â in European Conference on Computer Vision. Springer, 2024, pp. 296â313.

[19] J. Y. Koh, H. Lee, Y. Yang, J. Baldridge, and P. Anderson, âPathdreamer: A world model for indoor navigation,â in International Conference on Computer Vision. IEEE/CVF, 2021, pp. 14 738â 14 748.

[20] A. Bar, G. Zhou, D. Tran, T. Darrell, and Y. LeCun, âNavigation world models,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 15 791â15 801.

[21] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[22] T. Chen, O. Shorinwa, J. Bruno, A. Swann, J. Yu, W. Zeng, K. Nagami, P. Dames, and M. Schwager, âSplat-Nav: Safe real-time robot navigation in gaussian splatting maps,â arXiv preprint arXiv:2403.02751, 2024.

[23] X. Lei, M. Wang, W. Zhou, and H. Li, âGaussNav: Gaussian splatting for visual navigation,â IEEE Transactions on Pattern Analysis & Machine Intelligence, no. 01, pp. 1â14, 2025.

[24] T. Chen, A. Swann, J. Yu, O. Shorinwa, R. Murai, M. Kennedy III, and M. Schwager, âSAFER-Splat: A control barrier function for safe navigation with online gaussian splatting maps,â arXiv preprint arXiv:2409.09868, 2024.

[25] M. Adamkiewicz, T. Chen, A. Caccavale, R. Gardner, P. Culbertson, J. Bohg, and M. Schwager, âVision-only robot navigation in a neural

radiance world,â IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4606â4613, 2022.

[26] G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou, âInformation-theoretic model predictive control: Theory and applications to autonomous driving,â IEEE Transactions on Robotics, vol. 34, no. 6, pp. 1603â1622, 2018.

[27] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Conference on Computer Vision and Pattern Recognition. IEEE/CVF, 2018, pp. 586â595.

[28] A. Krizhevsky, I. Sutskever, and G. E. Hinton, âImagenet classification with deep convolutional neural networks,â Advances in Neural Information Processing Systems, vol. 25, 2012.

[29] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kegl, âAlgorithms for Â´ hyper-parameter optimization,â Advances in Neural Information Processing Systems, vol. 24, 2011.

[30] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, âOptuna: A next-generation hyperparameter optimization framework,â in ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019.

[31] P. E. Hart, N. J. Nilsson, and B. Raphael, âA formal basis for the heuristic determination of minimum cost paths,â IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100â107, 1968.

[32] D. Dolgov, S. Thrun, M. Montemerlo, and J. Diebel, âPractical search techniques in path planning for autonomous driving,â ann arbor, vol. 1001, no. 48105, pp. 18â80, 2008.

[33] J. Ho, A. Jain, and P. Abbeel, âDenoising diffusion probabilistic models,â Advances in Neural Information Processing Systems, vol. 33, pp. 6840â6851, 2020.

[34] P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva, et al., âOn evaluation of embodied navigation agents,â arXiv preprint arXiv:1807.06757, 2018.

[35] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, T. Wang, A. Kristoffersen, J. Austin, K. Salahi, A. Ahuja, et al., âNerfstudio: A modular framework for neural radiance field development,â in ACM SIGGRAPH 2023 conference proceedings, 2023, pp. 1â12.

[36] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â Â¨ in Conference on Computer Vision and Pattern Recognition, 2016.

[37] K. Honda, R. Yonetani, M. Nishimura, and T. Kozuno, âWhen to replan? an adaptive replanning strategy for autonomous navigation using deep reinforcement learning,â in IEEE International Conference on Robotics and Automation. IEEE, 2024, pp. 6650â6656.

[38] Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and S. Peng, âStreet Gaussians: Modeling dynamic urban scenes with gaussian splatting,â in European Conference on Computer Vision. Springer, 2024, pp. 156â173.

[39] S. Garg, K. Rana, M. Hosseinzadeh, L. Mares, N. Sunderhauf, Â¨ F. Dayoub, and I. Reid, âRoboHop: Segment-based topological map representation for open-world visual navigation,â in International Conference on Robotics and Automation. IEEE, 2024, pp. 4090â 4097.