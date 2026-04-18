# RT-GuIDE: Real-Time Gaussian Splatting for Information-Driven Exploration

Yuezhan Tao, Dexter Ong, Varun Murali, Igor Spasojevic, Pratik Chaudhari and Vijay Kumar

AbstractâWe propose a framework for active mapping and exploration that leverages Gaussian splatting for constructing dense maps. Further, we develop a GPU-accelerated motion planning algorithm that can exploit the Gaussian map for realtime navigation. The Gaussian map constructed onboard the robot is optimized for both photometric and geometric quality while enabling real-time situational awareness for autonomy. We show through viewpoint selection experiments that our method yields comparable Peak Signal-to-Noise Ratio (PSNR) and similar reconstruction error to state-of-the-art approaches, while being orders of magnitude faster to compute. In closed-loop physicsbased simulation and real-world experiments, our algorithm achieves better map quality (at least 0.8dB higher PSNR and more than 16% higher geometric reconstruction accuracy) than maps constructed by a state-of-the-art method, enabling semantic segmentation using off-the-shelf open-set models. Experiment videos and more details can be found on our project page: https://tyuezhan.github.io/RT GuIDE/

Index TermsâView Planning for SLAM; Mapping; Perception-Action Coupling

## I. INTRODUCTION

A CTIVE mapping is a problem of optimizing the trajectoryof an autonomous robot in an unknown environment to of an autonomous robot in an unknown environment to construct an informative map in real-time. It is a critical component of numerous real-world applications such as precision agriculture, infrastructure inspection, and search and rescue missions. While nearly all tasks rely on recovering accurate metric information to enable path planning, many also require more fine-grained information. Recent advances in learned map representations from the computer vision and graphics communities [1, 2] have opened up new possibilities for active mapping and exploration while maintaining both geometrically and visually accurate digital twins of the environment.

While prior work effectively solves the problem of information-driven exploration [3, 4] or frontier-based exploration [5, 6], in this work we consider the additional problem of generating radiance fields while also performing autonomous navigation. Prior work has also proposed information metrics using novel learned scene representations that are capable of high quality visual reconstruction but are incapable of running in real-time onboard a robot [7]. To enable efficient mapping and planning in these novel representations, we consider approximation techniques for computing the information gain. Further, we consider the problem of generating highquality maps that are capable of novel-view synthesis that can be used for downstream applications.

<!-- image-->  
Figure 1: Key elements of our proposed approach. [A] Robot building a Gaussian map onboard in real-time and using it to avoid obstacles in the environment. Synthesized color and depth images from the Gaussian map are presented next to the corresponding observations from the RGBD sensor. [B] Robot navigating to unobserved areas (right) with high information gain while maximizing information along the trajectory. Map regions colored light to dark in increasing estimated information gain.

Fig. 1 shows the elements of our approach. We use the Gaussian splatting approach proposed in SplaTAM [8] to generate maps. We propose an efficient information gain metric and use a hierarchical planning framework to plan highlevel navigation targets that yield maximal information in the environment and low-level paths that are dynamically feasible, collision-free and maximize the local information of the path. We utilize 3DGS as a unified representation for both mapping and planning. We avoid the cost of maintaining additional volumetric representations â achieving the same granularity of a 3DGS map with a voxel map would require around 19.2 GB of RAM for a 40mÃ40mÃ3m space at 1 cm resolution. With our GPU-based trajectory planner, we achieve an 18x speedup compared to a CPU-based implementation, as detailed in Sec. V-B. This allows us to generate collision-free trajectories on millions of Gaussians efficiently. Our system runs real-time onboard a fully autonomous unmanned ground vehicle (UGV) to explore an unknown environment while generating a highfidelity visual representation of the environment.

In summary, the contributions of this paper are:

1) A unified framework for online mapping and planning built only on Gaussian splatting, eliminating the need for maintaining additional volumetric representations.

2) An information gain heuristic that is easy to compute and capable of running in real-time onboard a robot.

3) A real-time exploration system built on Gaussian splatting, comprehensively validated in both simulation and real-world experiments across diverse indoor and outdoor environments. We will release the full autonomy framework as open-source.

## II. RELATED WORK

Map Representation. To effectively construct a map of the environment, numerous map representations have been proposed in the robotics community. The most intuitive but effective volumetric representation has been widely used. Voxel-based representation could maintain information such as occupancy [9] or signed distance [10]. With the recent application of semantic segmentation, semantic maps that contain additional task-relevant information have been proposed [11]â [13]. With the recent advances in learned map representations in the computer vision community, Neural Radiance Fields (NeRF) [1] and 3D Gaussian Splatting (3DGS) [2] have become popular representations for robotic motion planning [7, 14, 15]. In this work, we study the problem of active mapping with the 3DGS representation.

Active Mapping. The problem of exploration and active Simultaneous Localization and Mapping (SLAM) has been widely studied in the past decade. The classical exploration framework uses a model-based approach to actively navigate towards frontiers [5] or waypoints that have the highest information gain [3, 4, 7]. Some recent work combines the idea of frontier exploration and the information-driven approach to further improve efficiency [16]â[18]. However, most of the existing work developed their approaches based on classical map representations. In this work, we instead consider the active mapping problem with a 3DGS representation. Bayesian neural networks [19, 20] and deep ensembles [7] are common approaches for estimating uncertainty in learned representations. Radiance field representations provide additional possibilities for estimating uncertainty through the volumetric rendering process [21]. NARUTO [22] uses an uncertaintylearning module to quantify uncertainty for active reconstruction. CG-SLAM [23] uses the difference between rendered depth and observations, and the alignment of Î±-blended and median depth as measures of uncertainty. FisherRF [24] leverages Fisher information to compute pixel-wise uncertainty on rendered images. Where prior work in implicit and radiance field representations use indirect methods like ensembles and rendering for estimating uncertainty in the representation, a map represented by Gaussians encodes physical parameters of the scene, which motivates estimating information from the Gaussian parameters directly without rendering.

Planning in Radiance Fields. Prior work has also considered planning directly in radiance fields. GS-planner [15] plans trajectories in a Gaussian map and uses observability coverage and reconstruction loss stored in a voxel grid as an approximation of information gain for exploration. Sim-to-real approaches leverage radiance fields as a simulator for imitation learning [25]. Adamkiewicz et al. [26] utilize the geometric fidelity of the representation to first map the environment and then perform trajectory optimization to plan paths in these environments. Splat-nav [14] uses a pre-generated Gaussian map to compute safe polytopes to generate paths. The probabilistic representations of free space has also been utilized for motion planning [27] where authors use uncertainties in the learned representation to provide guarantees on collision-free paths. While prior work considers planning in radiance fields, they typically maintain a separate volumetric representation for planning which requires additional memory usage onboard the robot. On the other hand, maintaining a secondary representation may lead to inconsistencies in collision checks for planning. Existing work that performs collision checking with 3DGS requires subsampling to provide soft constraints. In this work, our proposed planning module performs dense checks directly against all Gaussians for collision avoidance.

## III. PROBLEM SPECIFICATION & PRELIMINARIES

Our goal is to construct an estimate $\hat { G }$ of the true map of the environment $G ^ { * }$ , which is a priori unknown. The quality of the constructed map $\hat { G }$ is evaluated using a hidden test set $\tau$ of tuples of poses and corresponding noise-free measurements, $( x _ { t e s t } ^ { * } , y _ { t e s t } ^ { * } ) _ { t e s t \in \mathcal { T } }$ . We aim to solve the following problem

$$
\begin{array} { r l } & { \arg \operatorname* { m i n } \displaystyle \sum _ { \begin{array} { c } { t e s t \in \mathcal { T } } \\ { s . t \hat { G } = \Phi ( x _ { 1 : T } , y _ { 1 : T } ) , } \end{array} } \mathcal { L } ( y _ { t e s t } ^ { * } , h ( x _ { t e s t } ^ { * } , \hat { G } ) ) } \end{array}\tag{1}
$$

where $\mathcal { L } ( \cdot )$ captures the difference between a synthesized and a true measurement, $\Phi ( \cdot )$ is the function that constructs $\hat { G }$ from the set of measurements $y _ { 1 : T }$ obtained at poses $x _ { 1 : T } , h ( \cdot )$ is the rendering function that synthesizes a measurement given a pose and a map, and $T$ is the time budget for exploration.

The problem above is ill-posed, as the set of test poses and measurements $\tau$ is hidden. We approach this challenge by solving the following two problems in a receding horizon scheme. At time $k \leq T$ , we first solve a mapping problem that involves minimizing the difference between the observed measurements $y _ { 1 : k }$ and synthesized measurements $\hat { y } _ { 1 : k } \colon$

$$
\Phi ( x _ { 1 : k } , y _ { 1 : k } ) : = \underset { \hat { G } } { \arg \operatorname* { m i n } } \sum _ { s = 1 } ^ { k } \mathcal { L } ( y _ { s } , h ( x _ { s } , \hat { G } ) ) .\tag{2}
$$

Given the posterior estimate of the map ${ \hat { G } } ,$ we find the subsequent viewpoint $x _ { k + 1 }$ that maximizes the information gain $\boldsymbol { \mathcal { T } } ( \cdot )$ given by the mutual information between the map $\hat { G }$ and the measurement $y _ { k + 1 }$ at the corresponding pose:

$$
\operatorname* { m a x } _ { x _ { k + 1 } } \mathcal { T } ( \hat { G } ; y _ { k + 1 } \mid x _ { k + 1 } ) .\tag{3}
$$

Gaussian Splatting. We use 3D Gaussian Splatting (3DGS) [2] to represent the environment as a collection of isotropic Gaussians ${ \hat { G } } .$ Each Gaussian is parameterized with 8 values representing RGB intensities $( c ) .$ , 3D position $( \mu )$ radius (r), and opacity (Î±). To generate the measurement at a given camera pose of the estimated map ${ \hat { G } } ,$ 3D Gaussians are sorted in increasing order of depth relative to the camera pose and are then splatted onto the image plane. In particular, each splatted Gaussian will have a projected 2D mean $\left( \mu _ { 2 D } \right)$ and radius $( r _ { 2 D } )$ on the image plane,

<!-- image-->  
Figure 2: The proposed active mapping framework. The proposed framework contains two major components, the planning module and the mapping module. As can be seen in the figure, the Mapping module ([A]) takes in RGB, depth and pose measurements, and updates the map representation at every step and computes the utility of cuboidal regions (Sec. IV-B1). The utility of each region is then passed to the planning module which comprises the topological graph and motion primitive library ([B]). The trajectory planner in turn attempts to plan a path to goal that maximizes information gathering (queried from the mapper). The planned trajectory is executed by the robot to get a new set of observations.

$$
\mu _ { 2 D } = \frac { K R _ { k } ^ { \mathsf { T } } ( \mu - T _ { k } ) } { d } , \ r _ { 2 D } = \frac { f r } { d } , \ d = e _ { 3 } ^ { \mathsf { T } } R _ { k } ^ { \mathsf { T } } ( \mu - T _ { k } ) ,
$$

where K is the intrinsic matrix, $R _ { k }$ and $T _ { k }$ are the rotational and translational components of the camera pose in the world frame at time $k , f$ is the focal length, and $e _ { 3 } = [ 0 , 0 , 1 ] ^ { \intercal }$

For a given pixel $p$ in the image, its color $\mathcal { C } ( \cdot )$ can be obtained as

$$
\mathcal { C } ( p ) = \sum _ { i = 1 } ^ { n } c _ { i } f _ { i } ( p ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f _ { j } ( p ) ) , f _ { i } ( p ) = \alpha _ { i } \exp \left\{ - \frac { | | p - \mu _ { 2 D , i } | | ^ { 2 } } { 2 r _ { 2 D , i } ^ { 2 } } \right\} .
$$

Similarly, the depth $\mathcal { D } ( \cdot )$ can be obtained by replacing $c _ { i }$ with $d _ { i } .$ . At every iteration, we render color image $\hat { y } _ { c }$ and depth image ${ \hat { y } } _ { d }$ . We set the function $\mathcal { L }$ (from eq. (2)) to

$$
\mathcal { L } = \frac { 1 } { N _ { p } } \sum _ { p \in y } ( | \hat { y } _ { d } - y _ { d } | + \lambda _ { 1 } | \hat { y } _ { c } - y _ { c } | ) + \lambda _ { 2 } ( 1 - S S I M ( \hat { y } _ { c } , y _ { c } ) )\tag{4}
$$

which is a weighted combination of the L1 loss on depth and rendered pixel colors and the structured similarity index measure (SSIM) to update the parameters of the Gaussians, and $N _ { p }$ is the number of pixels in the image.

## IV. METHOD

Our proposed framework comprises a mapping module and a planning module, as illustrated in Fig. 2. The mapping module (Sec. IV-A) accumulates measurements and poses to reconstruct the environment and computes the uncertainty of Gaussians in the map. This is then passed to the planning module for planning guidance paths (Sec. IV-B1) and trajectories (Sec. IV-B2).

## A. Mapping & Uncertainty Estimation

We build the 3DGS mapping module upon SplaTAM [8] with isotropic Gaussians. As in [8], we set $\lambda _ { 1 } = 0 . 4 , \lambda _ { 2 } = 0 . 1$ At the beginning of each mapping iteration, we initialize new Gaussians at positions corresponding to the current color and depth measurements. The parameters of these Gaussians are refined through backpropagation. As in [2], we prune Gaussians with low opacity or large radius.

Uncertainty Estimation. The objective for finding the nextbest-view can then be expressed as maximizing the mutual information

$$
\begin{array} { r l } & { x _ { k + 1 } ^ { * } = \underset { x _ { k + 1 } } { \arg \operatorname* { m a x } } \mathcal { T } ( y _ { k + 1 } ; \hat { G } \mid x _ { k + 1 } ) , } \\ & { \qquad \doteq H ( y _ { k + 1 } \mid x _ { k + 1 } ) - H ( y _ { k + 1 } \mid \hat { G } , x _ { k + 1 } ) } \end{array}\tag{5}
$$

where $H ( \cdot )$ is the Shannon entropy. We have

$$
\begin{array} { r l } & { H ( y _ { k + 1 } \mid x _ { k + 1 } ) = - \displaystyle \int p ( y _ { k + 1 } \mid x _ { k + 1 } ) \log p ( y _ { k + 1 } \mid x _ { k + 1 } ) ) d y _ { k + 1 } } \\ & { \mathrm { w i t h ~ } p ( y _ { k + 1 } \mid x _ { k + 1 } ) = \displaystyle \int p ( y _ { k + 1 } \mid \hat { G } , x _ { k + 1 } ) p ( \hat { G } ) d \hat { G } \mathrm { ~ a n d ~ } } \\ & { H ( y _ { k + 1 } \mid \hat { G } , x _ { k + 1 } ) = - \int p ( \hat { G } ) } \\ & { ~ \displaystyle \left[ \int p ( y _ { k + 1 } \mid \hat { G } , x _ { k + 1 } ) \log p ( y _ { k + 1 } \mid \hat { G } , x _ { k + 1 } ) d y _ { k + 1 } \right] d \hat { G } } \end{array}
$$

We make the following assumptions in this work to approximate the information gain efficiently: (a) $\hat { G } \sim$ ${ \mathcal { N } } ( \mathrm { E } [ G ] , \mathrm { V a r } [ G ] )$ , where G is the random variable that denotes the map; (b) $p ( y _ { k + 1 } \mid \hat { G } ) = \mathcal { N } ( h ( x _ { k + 1 } , \hat { G } ) , \Sigma _ { y } ) ;$ ; with (c) constant isotropic variance $\Sigma _ { y } .$ . For the marginal entropy $H ( y _ { k + 1 } | \mathbf { \alpha } | \mathbf { \alpha } x _ { k + 1 } )$ , given assumptions (a) and (b), and linearization of $h ( \cdot )$ around E[G], we can approximate the distribution $\begin{array} { r } { p ( y \mid x ) = \mathcal { N } ( h ( x , \dot { \mathrm { E } } [ \dot { G } ] ) , J _ { k + 1 } \mathrm { V a r } [ G ] J _ { k + 1 } ^ { \top } ) } \end{array}$ where $J _ { k + 1 } = \partial h ( x _ { k + 1 } , \hat { G } ) / \partial G$ . Under assumptions (b) and (c), the conditional entropy does not depend on G. Recall the entropy of a $d _ { Z }$ dimensional Gaussian distribution $\mathcal { N } ( \mu _ { Z } , \Sigma _ { Z } )$ is $\begin{array} { r } { \frac { d _ { Z } } { 2 } \log ( 2 \pi e ) + \frac { 1 } { 2 } \log \operatorname* { d e t } ( \Sigma _ { Z } ) } \end{array}$ . Dropping the constant terms, the information gain can be simplified to

$$
\mathcal { T } ( y _ { k + 1 } ; \hat { G } \mid x _ { k + 1 } ) = \frac { 1 } { 2 } \log \operatorname* { d e t } ( I + \Sigma _ { y } ^ { - 1 } J _ { k + 1 } \mathrm { V a r } [ G ] J _ { k + 1 } ^ { \top } ) .\tag{6}
$$

Using the first order approximation of log det about the identity, we can define a proxy metric for scoring next best view as

$$
\boldsymbol { x } _ { k + 1 } ^ { * } = \underset { \boldsymbol { x } _ { k + 1 } } { \arg \operatorname* { m a x } } \mathrm { T r } ( \Sigma _ { \boldsymbol { y } } ^ { - 1 } \boldsymbol { J } _ { k + 1 } \mathrm { V a r } [ G ] \boldsymbol { J } _ { k + 1 } ^ { \intercal } ) .\tag{7}
$$

As in [24], assumption (c) allows us to simplify the optimization by ignoring the effect of $\Sigma _ { y } ^ { - 1 }$ . Evaluating $J _ { k + 1 }$ is computationally expensive because it typically requires rendering at new poses. To achieve real-time evaluation of multiple candidate viewpoints, we replace the Jacobian term $J _ { k + 1 }$ with a binary matrix that corresponds to its sparsity pattern. Intuitively, this captures the set of Gaussians within the field of view at pose $x _ { k + 1 }$ . The key insight behind these approximations is that we estimate information gain using the current uncertainty, based on the assumption that local measurements lead to precise local map estimates. We show empirically that the exploration performance based on our metric (in Sec. V-A2) is similar to the performance based on eq. (7). We propose a heuristic for estimating uncertainty $\mathrm { V a r } [ G ]$ based on the magnitude of the change in means $( \mu )$ of the Gaussians, associating larger displacements with higher uncertainty. The motivation behind this is that $\mathrm { V a r } [ G ]$ can be approximated with the empirical Fisher Information matrix [28], Var $\mathbf { \nabla } [ G ] \approx \nabla _ { G } \mathcal { L } ( x _ { k } , y _ { k } ) \nabla _ { G } \mathcal { L } ( x _ { k } , y _ { k } ) \nabla$ , which in turn can be directly linked to the square of the updates of Gaussian parameters tuned by the gradient descent algorithm. In practice, we observe that the square of the magnitude of the updates outlined above is sensitive to measurement noise (in Sec. V-A3), so we use the magnitude of the change instead.

Empirically, we observe that the means of Gaussians located near boundaries between observed and unobserved space exhibit significant changes over successive updates, indicating high uncertainty, as visualized in Fig. 2. Consequently, directing exploration toward these high-uncertainty Gaussians is analogous to following frontiers in grid-based exploration, making this heuristic effective for autonomous exploration. Through careful bookkeeping, Gaussians that remain unchanged in the most recent update retain their previously computed displacement values, ensuring that our heuristic provides a consistent measure of each Gaussianâs movement over time. In addition to the boundaries between observed and unobserved space, areas within observed space that have insufficient measurements also exhibit relatively high uncertainty. This further directs the robot to explore these areas and gather more information, improving the overall map quality.

## B. Hierarchical Planning

In spite of the approximations introduced in the previous section to compute information gain obtained by navigating to a given area in space in a computationally efficient way, the task of finding such a region is still a challenging optimization problem. For this reason, we address the latter task through a hierarchical planning framework. The high-level planner provides guidance to map particular regions of the environment and a path to that region from the known space in the environment. The low-level (trajectory) planner finds a trajectory that is dynamically-feasible (i.e. obeys the robotâs physical constraints), collision-free and locally maximizes the information gain along the path.

1) High-level planner: In contrast to traditional mapping representations, the Gaussian map does not encode occluded and free space. Instead of computing (geometric) frontiers, we use the Gaussian uncertainty estimates to identify regions of the map that should be visited next. We evenly partition the a priori known enclosing space of the desired map into cuboidal regions. At time $k ,$ for a region $^ { O , }$ denote the cardinality of the Gaussians in the region by $M _ { k }$ . We compute the mean uncertainty of that region $\begin{array} { r } { \Omega _ { k } = 1 / M _ { k } \left( \sum _ { \mu \in o } \| \mu _ { k } - \mu _ { k - 1 } \| _ { 2 } \right) } \end{array}$

We formulate a high-level guidance path that allows us to (i) navigate to regions of high uncertainty; and (ii) utilize the known traversable space to plan long-range trajectories without computationally expensive collision checks. At the high level of our planner, we construct a tree by incrementally adding nodes along the traveled path. The tree consists of two types of nodes: odometry nodes and viewpoint nodes. As the robot moves, it verifies the presence of a nearby odometry node; if none exist, a new odometry node is added to the tree and connected to the nearest existing odometry nodes. At each planning iteration, we sample a fixed number of viewpoints around the identified regions. We compute the shortest viewing distance from the Gaussian region centroid to the optical center of the camera, given the camera intrinsics. Each sampled viewpoint is then assigned the utility computed for that region and connected to the closest odometry node in the tree as a viewpoint node. We then use Dijkstraâs algorithm to find the shortest path from the current robot location in the tree to all the viewpoint nodes in the tree. Finally, we compute the cost-benefit of a path using $\Omega / e ^ { d }$ where â¦ is the estimated information gain and d is the distance to the node [29]. The maximal cost-benefit path is sent to the trajectory planner.

2) Trajectory planner: We partition the current map $\hat { G }$ into three disjoint subsets: $\hat { G } _ { H }$ , which consists of Gaussians with high uncertainty, $\hat { G } _ { L }$ , which consists of Gaussians with low uncertainty, and $\hat { G } _ { O }$ for the rest. For a Gaussian $g ~ \in ~ { \hat { G } }$ and pose $x ,$ we define the binary visibility function $v ( x , g )$ capturing whether $g$ is in the field of view of $x .$ To avoid rendering each view to check for occlusions, we cull the Gaussians that are beyond the perception range of the sensor.

To maximize information gain while exploring the environment, we aim to obtain viewpoints with high utility $\xi$ that maximize the number of high-uncertainty Gaussians and minimize the number of low-uncertainty Gaussians in the field of view. Trajectories are evaluated based on the sum of the utilities of the viewpoints in each trajectory given by

$$
\xi ( x , \hat { G } ) = \sum _ { g \in { \hat { G } } _ { H } } v ( x , g ) - \lambda _ { \xi } \sum _ { g \in { \hat { G } } _ { L } } v ( x , g ) .\tag{8}
$$

The first term in eq. (8) encourages additional observations of parts of the scene with high uncertainty. The second term penalizes observation of stable Gaussians in explored areas and encourages exploration of unseen parts of the environment. The two terms are weighted by $\lambda _ { \xi } .$

Given the path generated by the high-level planner, we select the furthest point $p _ { g o a l }$ along the path that lies within a local planning range $R _ { H }$ , and set it as the center of the goal region $\mathcal { X } _ { g o a l }$ . We use the unicycle model as the robot dynamics $f ( \cdot )$ The robot state $x = [ p , \theta ] \in \mathbb { R } ^ { 2 } \times S ^ { 1 }$ consists of its position (p) and heading (Î¸). The control inputs $u \ = \ [ v , \omega ] \ \in \ \mathbb { R } ^ { 2 }$ consist of linear velocity (v) and angular velocity (Ï). We solve the trajectory planning problem in 2 steps: (i) finding feasible, collision-free trajectory candidates, and (ii) selecting a trajectory that maximizes the information gain. The first problem is defined as follows:

Problem 1. Trajectory Candidate Generation. Given an initial robot state $x _ { 0 } \in \mathcal { X } _ { f r e e } ,$ and goal region $\mathcal { X } _ { g o a l } ,$ , find the control inputs u(Â·) defined on [0, Ï ] that solve:

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m i n } _ { u ( \cdot ) , \tau } ~ \lambda _ { t } \tau + \int _ { 0 } ^ { \tau } u ( t ) ^ { T } u ( t ) d t } \\ & { \displaystyle s . t . ~ \forall t \in [ 0 , \tau ] , ~ x ( 0 ) = x _ { 0 } , ~ x ( \tau ) \in \mathcal { X } _ { g o a l } , } \\ & { \displaystyle \dot { x } ( t ) = f ( x ( t ) , u ( t ) ) , ~ x ( t ) \in \mathcal { X } _ { f r e e } , } \\ & { \displaystyle | | v ( t ) | | _ { 2 } \leq v _ { m a x } , ~ | \omega ( t ) | \leq \omega _ { m a x } } \end{array}\tag{9}
$$

where $\lambda _ { t }$ weights the time cost with the control effort, $v _ { m a x }$ and $\omega _ { m a x }$ are actuation constraints, and $x _ { 0 }$ is the initial state.

Motion Primitive Tree Generation. Inspired by [30], we solve problem 1 by performing a tree search on the motion primitives tree. Motion primitives are generated with fixed control inputs over a time interval with known initial states. Given the actuation constraints $v _ { m a x }$ and $\omega _ { m a x }$ of the robot, we uniformly generate $N _ { v } \times N _ { \omega }$ samples from $[ 0 , v _ { m a x } ] \times [ - \omega _ { m a x } , \omega _ { m a x } ]$ as the finite set of control inputs. Subsequently, motion primitives are constructed given the dynamics model, the controls $u ,$ and time discretization. Collision Checking. We sample a fixed number of points on each motion primitive to conduct collision checks. Since we have an uncertain map, we relax $x \ \in \ X _ { f r e e }$ to a chance constraint. Let $d _ { \kappa }$ be the minimum distance between a test point and the set of Gaussians. Our constraint then amounts to the probability of the distance between the test point and the set of Gaussians in the scene being less than some allowed tolerance $\gamma$ with probability Î· i.e. $P ( d _ { \kappa } < \gamma ) \leq \eta$ . When checking for collisions, each sampled point is bounded with a sphere of radius $r _ { r o b o t }$ and the radius r of each Gaussian is scaled by a factor of $\lambda _ { g }$ . The truth value of the test is determined by comparing the distance to all Gaussians with $\gamma : = r _ { r o b o t } + \lambda _ { g } r$ , as illustrated in Fig. 2. Note that setting $\lambda _ { g } = 3$ is equivalent to the test proposed by [15]. However, unlike the continuous trajectory optimization approach in [15], which requires subsampling to provide soft constraints, our search-based method allows dense checks. We are implicitly assuming that 3DGS accurately captures geometry of the scene and we notice this holds empirically since we initialize Gaussians in the map using the depth measurements from the RGB-D sensor. In the case of collision checking against nonisotropic Gaussians, the principal axis of the Gaussian can be taken as the radius. We develop a GPU-accelerated approach for testing collisions of all sampled points with all Gaussians from the map at once while growing the search tree. This allows the real-time expansion of the search tree.

Tree Search. The cost of each valid motion primitive is defined by Prob. 1. Since the maximum velocity of the robot is bounded by $v _ { m a x }$ , we consider the minimum time heuristic as $h ( p ) : = | | p _ { g o a l } - p | | _ { 2 } / v _ { m a x }$ . We use $\mathbf { A } ^ { * }$ to search through the motion primitives tree and keep top $N _ { t r a j }$ candidate trajectories for the information gain maximization.

Information Maximization. A state sequence $\{ x _ { i } ( l \tau / L ) \mid l \in$ $\{ 0 , \ldots , L \} \}$ containing the end state of each of L segments in the i-th trajectory is used for evaluating the information along the trajectory. The information of each state is evaluated according to eq. (8). Finally, the trajectory with the highest information

$$
\underset { i \in \{ 1 , \dots , N _ { t r a j } \} } { \arg \operatorname* { m a x } } \sum _ { l = 0 } ^ { L } \xi \left( x _ { i } \left( \frac { l \tau } { L } \right) \right) , l \in \{ 0 , \dots , L \}\tag{10}
$$

is selected and executed by the robot.

## C. Implementation Details

We implemented the proposed method using PyTorch2.4 and the mapping module following [8]. We note that our framework is compatible with other Gaussian splatting approaches if they can meet the compute and latency requirements. The parameters of the mapper were set to the default configuration of [8] for the TUM dataset. For viewpoint selection experiments, we used 50 mapping iterations. For realworld experiments, to enable real-time mapping and collision avoidance, we reduced the number of mapping iterations to 10 and pruned Gaussians once per mapping sequence. The mapping and planning modules are both set to 1Hz onboard the robot. For trajectory viewpoint evaluation, we set the threshold for high and low uncertainty Gaussians at half a standard deviation above and below the average magnitude of parameter updates, respectively. We found Gaussians on the ground plane to be particularly noisy in indoor environments due to reflections off the floor. This necessitated the removal of Gaussians on the ground plane for planning. We set the Gaussian region size to 2.5m, $\lambda _ { \xi } ~ = ~ 1$ and $\lambda _ { g } ~ = ~ 3$ . For planning, we set $N _ { v } = 3 , v _ { m a x } = 0 . 6 , N _ { \omega } = 5 , \omega _ { m a x } = 0 . 9$ and time discretization 1s.

## V. RESULTS

## A. Information Metric Evaluation & Ablation

1) Experiment setup & Baselines: To evaluate our proposed heuristic on viewpoint selection, we conducted simulation experiments with the AI2-THOR simulator [31] on the iTHOR dataset. In these experiments, time budget is not imposed, and we run experiments on all test scenes with 100 steps each. The iTHOR dataset consists of indoor scenes including kitchens, bedrooms, bathrooms and living rooms. At each step, the agent is allowed to translate 0.5m in both x and y positions and a discrete set of possible orientations. These candidate positions, along with their corresponding yaw angles, define the full set of potential viewpoints. Each viewpoint is evaluated using an uncertainty metric, and the one with the highest uncertainty is selected as the next action. The simulations were run on a desktop computer with an AMD Ryzen Threadripper PRO 5975WX and NVIDIA RTX A4000 (16GB).

For each scene, we synthesized novel views from the generated maps from a set of uniformly sampled test poses and computed the image metrics against the ground truth images. We evaluate the Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) [32], Learned Perceptual

Table I: Information metric benchmark experiments on iTHOR
<table><tr><td>Methods</td><td>PSNR [dB] â</td><td>SSIM â</td><td>LPIPS â</td><td>RMSE [m] â</td><td>t/step [s] â</td></tr><tr><td>Ensemble</td><td>21.489</td><td>0.762</td><td>0.360</td><td>0.350</td><td>2.667</td></tr><tr><td>FisherRFâ </td><td>24.171</td><td>0.849</td><td>0.270</td><td>0.285</td><td>1.299</td></tr><tr><td>RTGuIDE</td><td>22.946</td><td>0.820</td><td>0.305</td><td>0.343</td><td>0.013</td></tr></table>

â  Viewpoint selected using information metric from FisherRF.

Table II: Information metric ablation experiments on iTHOR
<table><tr><td>Methods</td><td>PSNR [dB] â</td><td>SSIM â</td><td>LPIPS â</td><td>RMSE [m] â</td><td>t/step [s] â</td></tr><tr><td>RTGuIDE</td><td>22.946</td><td>0.820</td><td>0.305</td><td>0.343</td><td>0.013</td></tr><tr><td>RTGuIDE-sum</td><td>20.279</td><td>0.740</td><td>0.371</td><td>0.512</td><td>0.007</td></tr><tr><td>RTGuIDE-sq</td><td>22.489</td><td>0.802</td><td>0.308</td><td>0.343</td><td>0.014</td></tr><tr><td>RTGuIDE w. noise</td><td>22.541</td><td>0.801</td><td>0.329</td><td>0.344</td><td>0.013</td></tr><tr><td>RTGuIDE-sq w. noise</td><td>21.445</td><td>0.779</td><td>0.351</td><td>0.386</td><td>0.013</td></tr></table>

Image Patch Similarity (LPIPS) [33] on the RGB images and Root Mean-Square-Error (RMSE) on the depth images. We evaluate the proposed method (denoted as Ours) on viewpoints following eq. (8) against two other view selection methods. For FisherRF, we follow the original implementation to evaluate viewpoints based on the Fisher information. For the Ensemble baseline, we train an ensemble of 5 models with the leave-one-out procedure at each step. We compute the patchwise variance across the rendered images for each sampled viewpoint and ensemble model, and select the viewpoint with the highest variance as the next action.

2) Benchmark experiments: The averaged results over all test scenes are presented in Tab. I. Our heuristic performs comparably with FisherRF and outperforms Ensemble across all metrics, while being more than an order of magnitude faster in computation time. FisherRF renders images at every sampled viewpoint to approximate the information gain with the full map parameters, while we avoid such computational cost by computing the uncertainty directly on the parameters of the Gaussians. This trades off the approximation accuracy with efficiency. This efficiency in computation is crucial for real-time operation where we evaluate potentially hundreds of viewpoints in each planning iteration.

3) Ablation experiments: For ablation experiments, we implemented RTGuIDE-sum, which solely considers the sum of the uncertainty of Gaussians in the camera view. RTGuIDE-sq evaluates the uncertainty of Gaussians as the squared L2 norm following the original derivation in eq. (7) and using eq. (8) to evaluate the uncertainty of viewpoints. To simulate real-world noisy measurements, we added Gaussian noise $( \mu = 0 , \sigma ^ { 2 } = 0 . 1 )$ to depth measurements in the simulation to evaluate RTGuIDE w. noise and RTGuIDE-sq w. noise.

As shown in Tab. II, the ablation experiments on different metrics demonstrate that our proposed approach outperforms the simple uncertainty summation (RTGuIDE-sum), which focuses purely on exploitation. Furthermore, the results show that both the squared (RTGuIDE-sq) and standard L2 norms (RTGuIDE) achieve comparable performance when perfect depth measurements are available. However, when noise is introduced, the experiments confirm that our chosen metric is more robust and better suited for real-world scenarios where measurement noise is inevitable.

<!-- image-->  
Figure 3: Time spent to plan a trajectory with a 5m horizon (including 129 collision checking points) versus the number of Gaussians in the map.

Table III: Closed-loop Experiments on MP3D in Unity Simulator
<table><tr><td>Methods</td><td>PSNR [dB] â</td><td>SSIM â</td><td>LPIPS â</td><td>RMSE [m] â</td><td>Coverage [%] â</td></tr><tr><td>Ensemble</td><td>10.71</td><td>0.24</td><td>0.76</td><td>2.14</td><td>38.93</td></tr><tr><td>FisherRF</td><td>15.37</td><td>0.49</td><td>0.59</td><td>1.06</td><td>66.40</td></tr><tr><td>RTGuIDE</td><td>16.32</td><td>0.56</td><td>0.54</td><td>0.71</td><td>92.36</td></tr><tr><td>GGT </td><td>18.13</td><td>0.65</td><td>0.47</td><td>0.50</td><td></td></tr></table>

## B. Planner Performance

We evaluate the necessity of our proposed GPU-based planning approach in enabling real-time planning. In particular, we conducted experiments on trajectory planning with a 5-meter horizon, evaluating both GPU-accelerated collision checking and a fully CPU-based implementation. As illustrated in Fig. 3, the planning time of CPU-based planner increases significantly with the number of Gaussians. For example, a 40m Ã 40m outdoor parking lot can contain around $3 \times 1 0 ^ { 6 }$ Gaussians. In this case, the CPU-based planner requires 11.65 seconds, whereas the GPU-based planner takes 0.62 seconds, achieving 18Ã speedup. The GPU-based collision testing enables parallel growth of the search tree at each layer, along with simultaneous and efficient collision checks between all test points and Gaussians directly on the GPU.

## C. Closed-loop Simulation Experiments

1) Experiment setup & Baselines: We conducted experiments in a Unity-based simulator with ROS and a simulated Clearpath Jackal robot with an RGBD sensor and ground truth odometry. This set of experiments was designed to evaluate all methods with real-time and closed-loop operation, considering robot dynamics and collision avoidance.

The original implementation of FisherRF [24] does not model dynamics or include real-world experimental results, so we implement a closed-loop version of it. We first construct both a voxel map and a Gaussian splatting map online. We detect and cluster frontiers from the voxel map and use A\* to generate reference paths to frontier clusters. Camera views along each path are evaluated to compute information gain. To generate collision-free and dynamically-feasible trajectories, we generate trajectories along waypoints in the path through MPL [30] in the voxel map. We similarly implement an Ensemble method as described in Sec. V-A1.

We obtained the ground truth (GT) by tele-operating the robot to uniformly survey the environment. The collected data was used to build a 3DGS map for GT and also served as test poses for the other methods. This represents the upper bound of the evaluation metrics with the Gaussian splatting method. The exploration budget for all methods is set to approximately 5 times the duration needed to fully survey the environment with tele-operation. For the baseline methods, we set the voxel resolution to 5cm. We rendered 500 novel views at the ground truth test poses for evaluation. The same image metrics on the rendered RGB and depth images are used as in Sec. V-A. We evaluated all methods on 5 MP3D [34] scenes [17DRP5sb8fy, HxpKQynjfin, 2t7WUuJeko7, 8194nk5LbLH, YVUC4YcDtcY] with a time budget of 1500s, 480s, 1500s, 1200s and 1800s respectively. These scenes consist of multi-room indoor environments with varying size and clutter.

Table IV: Quantitative Results of Real-world Experiments.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Env.</td><td rowspan=1 colspan=1>Budget (Time) [s]</td><td rowspan=1 colspan=1>PSNR [dB] â</td><td rowspan=1 colspan=1>SSIM</td><td rowspan=1 colspan=1>LPIPS</td><td rowspan=1 colspan=1>RMSE [m] â</td><td rowspan=1 colspan=1>Coverage [%] â</td><td rowspan=1 colspan=1>%] â</td><td rowspan=1 colspan=1>mIoUa â</td></tr><tr><td rowspan=3 colspan=1>EnsembleFisherRFRTGuIDEGT</td><td rowspan=3 colspan=1>Indoor 1 $7 2 \mathrm { m } ^ { 2 }$ </td><td rowspan=3 colspan=1>300</td><td rowspan=2 colspan=1>6.2716.8017.83</td><td rowspan=1 colspan=1>0.2040.665</td><td rowspan=1 colspan=1>0.8390.393</td><td rowspan=1 colspan=1>2.0510.242</td><td rowspan=1 colspan=2>98.4899.86</td><td rowspan=1 colspan=1>0.0710.314</td></tr><tr><td rowspan=1 colspan=1>0.737</td><td rowspan=1 colspan=1>0.334</td><td rowspan=1 colspan=1>0.202</td><td rowspan=1 colspan=2>99.95</td><td rowspan=1 colspan=1>0.338</td></tr><tr><td rowspan=1 colspan=1>20.56</td><td rowspan=1 colspan=1>0.805</td><td rowspan=1 colspan=1>0.237</td><td rowspan=1 colspan=1>0.158</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.420</td></tr><tr><td rowspan=3 colspan=1>EnsembleFisherRFRTGuIDEGT</td><td rowspan=3 colspan=1>Indoor 2189m2</td><td rowspan=3 colspan=1>600</td><td rowspan=2 colspan=1>10.4114.8816.40</td><td rowspan=2 colspan=1>0.2690.5880.711</td><td rowspan=1 colspan=1>0.7870.451</td><td rowspan=1 colspan=1>2.4101.170</td><td rowspan=2 colspan=2>84.1072.80100.00</td><td rowspan=2 colspan=1>0.1560.2480.391</td></tr><tr><td rowspan=1 colspan=1>0.299</td><td rowspan=1 colspan=1>0.689</td></tr><tr><td rowspan=1 colspan=1>18.39</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.454</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.422</td></tr><tr><td rowspan=3 colspan=1>EnsembleFisherRFRTGuIDEGT</td><td rowspan=3 colspan=1>Outdoor 1 $2 5 2 \mathrm { m ^ { 2 } }$ </td><td rowspan=3 colspan=1>600</td><td rowspan=2 colspan=1>10.5917.5820.00</td><td rowspan=2 colspan=1>0.3920.6870.744</td><td rowspan=2 colspan=1>0.6370.3490.296</td><td rowspan=1 colspan=1>1.7050.519</td><td rowspan=2 colspan=2>48.3085.81100.00</td><td rowspan=2 colspan=1>0.5490.6000.622</td></tr><tr><td rowspan=1 colspan=1>0.369</td></tr><tr><td rowspan=1 colspan=1>22.95</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.241</td><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.63T</td></tr><tr><td rowspan=2 colspan=1>EnsembleFisherRFRTGuIDE</td><td rowspan=3 colspan=1>Outdoor 2 $4 8 0 \mathrm { m ^ { 2 } }$ </td><td rowspan=3 colspan=1>900</td><td rowspan=2 colspan=1>16.7919.4420.27</td><td rowspan=2 colspan=1>0.5380.6800.750</td><td rowspan=1 colspan=1>0.5760.439</td><td rowspan=1 colspan=1>1.4220.773</td><td rowspan=2 colspan=2>49.5960.9598.35</td><td rowspan=1 colspan=1>0.1590.177</td></tr><tr><td rowspan=1 colspan=1>0.345</td><td rowspan=1 colspan=1>0.399</td><td rowspan=1 colspan=1>0.174</td></tr><tr><td rowspan=1 colspan=1>GT</td><td rowspan=1 colspan=1>24.19</td><td rowspan=1 colspan=1>0.835</td><td rowspan=1 colspan=1>0.285</td><td rowspan=1 colspan=1>0.227</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.246</td></tr></table>

a Indoor 1 classes: [floor, chair, table, refrigerator, cabinet, backpack, plants]. Indoor 2 classes: [floor, chair, table, couch, cushion, trashcan, television, plants]. Outdoor 1 classes: [pavement, plants, barrel, traffic cone]. Outdoor 2 classes: [pavement, grass, tree, bench, lamppost].

2) Results: The averaged statistics of the experiments are presented in Table III. We note that there are signifcant errors in the meshes in MP3D which affect the reconstruction results for all methods including the ground truth. Our proposed method achieves 0.95dB and 5.61dB higher PSNR than FisherRF and Ensemble. The significantly higher coverage demonstrates the importance of efficient viewpoint selection in real-time missions with limited operational budget.

## D. Real-world Experiments & Benchmarks

1) Experiment setup & Baselines: To evaluate the effectiveness of our proposed framework in real-world settings, we compared it with two baselines onboard the robot. We followed the same setup as in the closed-loop simulation experiments for the baselines and the ground truth GT.

We deployed our method and the baselines in four different real-world environments on a Clearpath Jackal robot outfitted with an AMD Ryzen 5 3600 and RTX 4000 Ada SFF. In addition, the platform is equipped with an Ouster OS1 LiDAR for state estimation and a ZED 2i stereo camera for mapping. We used [35] to provide odometry and truncated the depth measurements at 5 meters for all methods in all of our experiments. The same evaluation setup from Sec. V-C is employed. To verify the usefulness of the generated Gaussian map representation for downstream tasks in robotics, we also performed an evaluation on the task of semantic segmentation. We used Grounded SAM 2 [36] to obtain segmentation masks of the rendered and groundtruth images and computed the mean Intersection over Union (mIoU).

<!-- image-->  
k=1/8T

<!-- image-->  
k=3/8T

<!-- image-->  
k=5/8T

<!-- image-->  
k=8/8T

Figure 4: Visualization of onboard constructed map  
<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Rendered RGB image from onboard Gaussian Splatting map

<!-- image-->

<!-- image-->

<!-- image-->  
Ground truth RGB image from camera

<!-- image-->  
Figure 5: Qualitative results

2) Qualitative results: We evaluated the quality of the Gaussian splatting maps that were constructed in real-time onboard the robot by rendering novel views with a set of test poses. A visualization of the onboard constructed map is shown in Fig. 4, and examples of rendered color images are presented in Fig. 5. Constructed in real time onboard the robot, our Gaussian splatting map provides a detailed representation of the environment. Being able to render photorealistic images from the map benefits downstream tasks such as semantic segmentation, which are analyzed in the next section.

3) Quantitative results: As shown in Tab. IV, in real-world scenarios when the operational budget is limited, our proposed framework constructs maps of higher quality than the baselines onboard the robot. To validate that our approach generates maps of reasonable quality, we also present maps generated with the ground truth samples. Results on novel view image rendering show that the maps constructed with our approach achieve 0.8-2.4 higher PSNR than FisherRF and 3.5-11.6 higher PSNR than Ensemble. Furthermore, when evaluated on depth reconstruction, our method reduces the RMSE of the rendered depth by 16.5%-48% compared to FisherRF, and by 71.4%-90.2% compared to Ensemble. These results reflect that our method outperforms the baselines in achieving good coverage of the environments while also optimizing for map quality during exploration.

For semantic segmentation, our approach achieves better mIoU scores for most experiments, indicating higher fidelity of the rendered images compared to the baselines.

4) Discussion: We attribute the performance of our approach to two key factors. First, the use of an information metric that is efficient to compute onboard the robot enables smooth and continuous operation. Second, since our method does not rely on frontiers extracted from the voxel map for geometric coverage, it allows for revisiting of areas in the environment to further improve map quality.

## VI. LIMITATIONS AND FUTURE WORK

In this work, we present a framework for real-time active exploration and mapping with Gaussian splatting. A future direction is to consider semantic features together with the Gaussian splatting maps to perform complex tasks in the environment like object search and represent dynamic scenes in our framework. In this work, we study the active mapping problem which assumes perfect state estimation. In future work, we aim to incorporate the state estimation uncertainty and formulate this entire framework using a single sensor.

## REFERENCES

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, no. 1, pp. 99â106, 2021.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Trans. Graph., vol. 42, no. 4, pp. 1â14, 2023.

[3] L. Schmid, M. Pantic, R. Khanna, L. Ott, R. Siegwart, and J. Nieto, âAn efficient sampling-based method for online informative path planning in unknown environments,â IEEE Robot. Automat. Lett., vol. 5, no. 2, pp. 1500â1507, 2020.

[4] M. Dharmadhikari, T. Dang, L. Solanka, J. Loje, H. Nguyen, N. Khedekar, and K. Alexis, âMotion primitives-based path planning for fast and agile exploration using aerial robots,â in Proc. IEEE Int. Conf. Robot. Automat., 2020, pp. 179â185.

[5] B. Yamauchi, âA frontier-based approach for autonomous exploration,â in Proc. IEEE Int. Symp. Comput. Intell. Robot. Automat. IEEE, 1997, pp. 146â151.

[6] J. Yu, H. Shen, J. Xu, and T. Zhang, âEcho: An efficient heuristic viewpoint determination method on frontier-based autonomous exploration for quadrotors,â IEEE Robot. Automat. Lett., vol. 8, no. 8, pp. 5047â 5054, 2023.

[7] H. Siming, C. D. Hsu, D. Ong, Y. S. Shao, and P. Chaudhari, âActive perception using neural radiance fields,â in 2024 American Control Conference (ACC). IEEE, 2024, pp. 4353â4358.

[8] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat, track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[9] A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss, and W. Burgard, âOctomap: An efficient probabilistic 3d mapping framework based on octrees,â Auton. Robots, vol. 34, pp. 189â206, 2013.

[10] H. Oleynikova, Z. Taylor, M. Fehr, R. Siegwart, and J. Nieto, âVoxblox: Incremental 3d euclidean signed distance fields for on-board mav planning,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2017, pp. 1366â1373.

[11] N. Hughes, Y. Chang, and L. Carlone, âHydra: A real-time spatial perception system for 3D scene graph construction and optimization,â Robot.: Sci. Syst., 2022.

[12] A. Asgharivaskasi and N. Atanasov, âSemantic octree mapping and shannon mutual information computation for robot exploration,â IEEE Trans. Robot., vol. 39, no. 3, pp. 1910â1928, 2023.

[13] X. Liu, J. Lei, A. Prabhu, Y. Tao, I. Spasojevic, P. Chaudhari, N. Atanasov, and V. Kumar, âSlideslam: Sparse, lightweight, decentralized metric-semantic slam for multi-robot navigation,â arXiv preprint arXiv:2406.17249, 2024.

[14] T. Chen, O. Shorinwa, J. Bruno, A. Swann, J. Yu, W. Zeng, K. Nagami, P. Dames, and M. Schwager, âSplat-nav: Safe real-time robot navigation in gaussian splatting maps,â IEEE Trans. Robot., 2025.

[15] R. Jin, Y. Gao, Y. Wang, Y. Wu, H. Lu, C. Xu, and F. Gao, âGs-planner: A gaussian-splatting-based planning framework for active high-fidelity reconstruction,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. IEEE, 2024, pp. 11 202â11 209.

[16] B. Zhou, Y. Zhang, X. Chen, and S. Shen, âFuel: Fast uav exploration using incremental frontier structure and hierarchical planning,â IEEE Robot. Automat. Lett., vol. 6, no. 2, pp. 779â786, 2021.

[17] Y. Tao, Y. Wu, B. Li, F. Cladera, A. Zhou, D. Thakur, and V. Kumar, âSEER: Safe efficient exploration for aerial robots using learning to predict information gain,â in Proc. IEEE Int. Conf. Robot. Automat. IEEE, 2023, pp. 1235â1241.

[18] Y. Tao, X. Liu, I. Spasojevic, S. Agarwal, and V. Kumar, â3d active metric-semantic slam,â IEEE Robot. Automat. Lett., vol. 9, no. 3, pp. 2989â2996, 2024.

[19] X. Pan, Z. Lai, S. Song, and G. Huang, âActivenerf: Learning where to see with uncertainty estimation,â in Proc. Eur. Conf. Comput. Vision. Springer, 2022, pp. 230â246.

[20] S. Lee, K. Kang, and H. Yu, âBayesian nerf: Quantifying uncertainty with volume density in neural radiance fields,â arXiv preprint arXiv:2404.06727, 2024.

[21] S. Lee, L. Chen, J. Wang, A. Liniger, S. Kumar, and F. Yu, âUncertainty guided policy for active robotic 3d reconstruction using neural radiance fields,â IEEE Robot. Automat. Lett., vol. 7, no. 4, pp. 12 070â12 077, 2022.

[22] Z. Feng, H. Zhan, Z. Chen, Q. Yan, X. Xu, C. Cai, B. Li, Q. Zhu, and Y. Xu, âNaruto: Neural active reconstruction from uncertain target observations,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 21 572â21 583.

[23] J. Hu, X. Chen, B. Feng, G. Li, L. Yang, H. Bao, G. Zhang, and Z. Cui, âCg-slam: Efficient dense rgb-d slam in a consistent uncertainty-aware 3d gaussian field,â in Proc. Eur. Conf. Comput. Vision. Springer, 2024, pp. 93â112.

[24] W. Jiang, B. Lei, and K. Daniilidis, âFisherrf: Active view selection and mapping with radiance fields using fisher information,â in Proc. Eur. Conf. Comput. Vision. Springer, 2024, p. 422â440.

[25] V. Murali, G. Rosman, S. Karamn, and D. Rus, âLearning autonomous driving from aerial views,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. IEEE, 2024.

[26] M. Adamkiewicz, T. Chen, A. Caccavale, R. Gardner, P. Culbertson, J. Bohg, and M. Schwager, âVision-only robot navigation in a neural radiance world,â IEEE Robot. Automat. Lett., vol. 7, no. 2, pp. 4606â 4613, 2022.

[27] T. Chen, P. Culbertson, and M. Schwager, âCatnips: Collision avoidance through neural implicit probabilistic scenes,â IEEE Trans. Robot., vol. 40, pp. 2712â2728, 2024.

[28] J. Martens, âNew insights and perspectives on the natural gradient method,â J. Mach. Learn. Res., vol. 21, no. 146, pp. 1â76, 2020.

[29] C. Gomez, M. Fehr, A. Millane, A. C. Hernandez, J. Nieto, R. Barber, and R. Siegwart, âHybrid topological and 3d dense mapping through autonomous exploration for large indoor environments,â in Proc. IEEE Int. Conf. Robot. Automat., 2020, pp. 9673â9679.

[30] S. Liu, N. Atanasov, K. Mohta, and V. Kumar, âSearch-based motion planning for quadrotors using linear quadratic minimum time control,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. IEEE, 2017, pp. 2872â2879.

[31] E. Kolve, R. Mottaghi, W. Han, E. VanderBilt, L. Weihs, A. Herrasti, M. Deitke, K. Ehsani, D. Gordon, Y. Zhu, et al., âAi2-thor: An interactive 3d environment for visual ai,â arXiv preprint arXiv:1712.05474, 2017.

[32] Z. Wang, E. Simoncelli, and A. Bovik, âMultiscale structural similarity for image quality assessment,â in The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003, vol. 2, 2003, pp. 1398â 1402 Vol.2.

[33] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2018.

[34] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Niessner, M. Savva, S. Song, A. Zeng, and Y. Zhang, âMatterport3d: Learning from rgb-d data in indoor environments,â Int. Conf. 3D Vision, 2017.

[35] C. Bai, T. Xiao, Y. Chen, H. Wang, F. Zhang, and X. Gao, âFasterlio: Lightweight tightly coupled lidar-inertial odometry using parallel sparse incremental voxels,â IEEE Robot. Automat. Lett., vol. 7, no. 2, pp. 4861â4868, 2022.

[36] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, Z. Zeng, H. Zhang, F. Li, J. Yang, H. Li, Q. Jiang, and L. Zhang, âGrounded sam: Assembling open-world models for diverse visual tasks,â 2024.