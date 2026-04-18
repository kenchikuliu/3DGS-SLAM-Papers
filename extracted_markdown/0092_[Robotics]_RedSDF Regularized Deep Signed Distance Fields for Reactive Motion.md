# Regularized Deep Signed Distance Fields for Reactive Motion Generation

Puze Liu, Kuo Zhang, Davide Tateo, Snehal Jauhri, Jan Peters and Georgia Chalvatzaki

Abstractâ Autonomous robots should operate in real-world dynamic environments and collaborate with humans in tight spaces. A key component for allowing robots to leave structured lab and manufacturing settings is their ability to evaluate online and real-time collisions with the world around them. Distance-based constraints are fundamental for enabling robots to plan their actions and act safely, protecting both humans and their hardware. However, different applications require different distance resolutions, leading to various heuristic approaches for measuring distance fields w.r.t. obstacles, which are computationally expensive and hinder their application in dynamic obstacle avoidance use-cases. We propose Regularized Deep Signed Distance Fields (ReDSDF), a single neural implicit function that can compute smooth distance fields at any scale, with fine-grained resolution over high-dimensional manifolds and articulated bodies like humans, thanks to our effective data generation and a simple inductive bias during training. We demonstrate the effectiveness of our approach in representative simulated tasks for whole-body control (WBC) and safe Human-Robot Interaction (HRI) in shared workspaces. Finally, we provide proof of concept of a real-world application in a HRI handover task with a mobile manipulator robot.

## I. INTRODUCTION

Safety is an essential prerequisite for the real-world deployment of robots and their interaction with the dynamic world and the humans in it. The study of safety constraints in robotics that can filter dangerous actions is as long as the field itself. A common way of approaching the problem is to define constraint functions that apply to specific applications and tasks. The imposed constraints can ensure safety by excluding unsafe parts of the state and action space to avoid collisions and self-collision, harming humans in shared workspaces, or for specifying task-specific workspaces, e.g., in human-robot collaboration, robot-assisted feeding, dressing, etc. Each of these tasks may demand a different set of constraints to be specified and satisfied during deployment.

Despite their importance, constraint functions may be difficult to hand-design, while ill-specified constraints may often make it challenging for robots to plan [1], optimize [2], learn [3], or act reactively [4]. One way of specifying a constraint is to define its boundaries as a level curve k of an implicit function f(X) = k, where X can be any query point in space. This formulation is particularly convenient because it defines the constraint over a surface intuitively. Second, it is easy to reason about constraint violations/slack, allowing for easier optimization. Furthermore, if the constraint is also differentiable, it is also possible to use gradient-based projection algorithms to enforce them.

<!-- image-->

<!-- image-->  
(a) Robot view

<!-- image-->  
(b) ReDSDF of the human  
Fig. 1: Real-world evaluation of our learned ReDSDF model in a HRI scenario. The mobile manipulator robot TIAGo++ has to approach the human and deliver him a water bottle without intervening in the safety boundary. (a) The perceived instance in the robotâs field of view. (b) The perceived human manifold, along with the 20cm level set (red point-cloud), i.e. the set of points at 20cm distance to the human.

It is common in robotics to formulate constraints in terms of distance w.r.t. a given object. Measuring distances w.r.t. to manifolds is necessary for most robot applications for collision avoidance [5], but also for HRI, where the robot should position itself at a certain distance from the surrounding humans to interact with them safely while respecting social norms [6]â[8]. Signed Distance Function (SDF) is a prominent representation for expressing distance w.r.t. a given surface, by defining a function that precomputes the distance of an arbitrary point in the space w.r.t. a fixed surface. Every point on the surface has zero distance, while points outside it have a signed distance. The sign is chosen by convention to be positive in the line of sight w.r.t. the sensor detecting the surface. If a surface represents a closed region of space, e.g., an object or a human, we can consider every point inside this region as a negative distance. Different variants of SDF were introduced in particular for robotics to counteract the high computational complexity and the non-smooth representations that arise from querying multiple points from numerous static or even dynamic obstacles [9].

In this paper, we present ReDSDF, an approach that extends the concept of SDFs for arbitrary articulated objects such as robotic manipulators and humans (Fig. 1). ReDSDF allows us to define complex distance functions to be used as constraints or energy functions to avoid collisions and achieve safe interactions. Our method provides a single deep model that not only preserves structural information in the proximity of articulated bodies, but also effectively provides a signed distance field at any scale. Notably, we define a simple yet effective inductive bias, i.e. L2-norm, for regularizing our deep signed distance model. This regularization allows us to obtain expressive deep SDFs that can provide precise distance computation over the objectâs manifold when operating close to it and meaningful level curves when the robot is far from it.

Our deep implicit distance fields give us three significant benefits: first, we can learn and generalize the distance function from data, making it possible to reliably approximate complex manifolds like humans. Second, we can generate smooth and differentiable distance fields that are particularly well-suited for reactive control or trajectory optimization. Finally, deep neural networks allow fast queries of many points in parallel, allowing precise collision avoidance. Our method can be applied to a wide range of robotic applications, from reactive WBC and HRI, to collision-free navigation of aerial robots, as ReDSDF can satisfy any-scale demands depending on the task at hand.

We demonstrate the capabilities of ReDSDF when employed in reactive motion generation tasks. In particular, we provide a qualitative comparison of the produced distance fields of ReDSDF against representative deep-based baselines, showcasing the advantages of our method. Next, we integrate ReDSDF in reactive motion generation. Namely, we design controllers for the WBC of the bimanual mobile manipulator TIAGo++, and for safe HRI in a shared workspace scenario in a novel simulated task that integrates real-human demonstrations. Finally, thanks to the advances in real-time human skeleton tracking and 3D shape detection, we demonstrate the real-world performance of ReDSDF for computing safety distances when interacting with a human. Our results show the potential of ReDSDF to become a major component of various robotic applications, where precise safety constraints are needed.

## II. RELATED WORK

SDFs have been studied extensively in the field of computer graphics [10] to reconstruct meshes of surrounding objects given some range information from lasers or cameras. The benefits of SDFs are very well related to problems of robot motion planning and control, collision-checking, and obstacle avoidance [11]. However, while SDFs are locally accurate due to truncation effects, they are challenging to construct from partial observations, i.e., when the whole shape of an object cannot be determined from a single viewpoint. Curless et al. [10] proposed a volumetric integration method that sacrifices the full coverage of the space for improving the local updates based on partial observations, preserving the representation of positions and orientations on the surface of objects. Truncating the field at small negative and positive values produces the Truncated Signed Distance Function (TSDF), in which a point outside the truncated region is located in a narrowband that embeds the surface of the object [12], [13], which allows for better modeling of sensor noise [14]. TSDFs are used overwhelmingly in simultaneous localization and mapping applications [15], [16]. Differently from TSDFs that limit the representation capabilities on points close to the surface, the Euclidean Signed Distance Function (ESDF) provides a method for assessing the free space rather than a fine obstacle area, which is needed for aerial robot mapping and planning [17]. The most popular trajectory optimization methods rely on ESDF to represent the environment [2], [18], [19], as they are effective in static environments, but suffers high computation time for real-time deployment in dynamic environments.

The deep learning revolution in computer graphics and vision led to the emergence of novel methods for object instance segmentation and reconstruction [20], [21] with immediate applications to robot mapping [22]â[24]. Deep implicit functions provide a novel way of training and learning to approximate meshes from point clouds and voxelized data from large simulated datasets [25]. DeepSDF refers to a shape representation method that employs a SDF for training a surface reconstruction network by classifying points as belonging to the surface of a mesh or not [26]. The Equality Constraint Manifold Neural Network (ECoMaNN) approach [27] extends the application of implicit functions to approximate generic equality constraints for robot motion planning. However, these results only apply to simple scenes, and the distance field estimates are accurate only locally. Other deep learning advances allow for 3D human shape reconstruction from point-clouds [28], [29] and RGB frames [30], opening up new possibilities for more precise pose estimation of highly articulated meshes, like humans [31]. While implicit representations have been studied in reconstructing the geometry of articulated objects [32]â[35], distances for query points that are not on the object surface may be illdefined for robot manipulation.

In the context of reactive motion control, the use of SDFs is challenging due to high computation times that hinder the necessary high control frequency for operating in dynamic environments. To this end, composite SDFs were proposed in [36] for propagating SDFs of moving objects by tracking the occupancy box of those objects in the environment. GPU-accelerated voxels combined with whole-body motion planning were explored in [9]. SDFs are specified across every link of a robot in [37] for obstacle avoidance during manipulation, while a specific distance function is presented in [38] for self-collision avoidance in WBC.

<!-- image-->  
Fig. 2: The Network Structure of ReDSDF

For HRI, detecting human poses and understanding the effective safe workspace of the human is essential. Parting from computer graphics, the modeling of highly articulated meshes and the simulation of natural biomechanical properties of human motion [39] became increasingly interesting for robotics. Human-centric knowledge was naturally transferred to human-robot simulated tasks, where the need for specifying the effective workspace of a human partner when interacting with a robot can be considered as a manipulability area around the human skeleton [40]. A common way of representing safety constraints in humans for HRI tasks is through learning the contour around a human pose projected in a low-dimensional space [41], [42] with primary applications in social human-aware navigation. For closer interactions, many methods represent human links with bounding spheres, from which proximity queries can be made for adapting robot strategies [43], [44].

## III. LEARNING REDSDF AND ROBOT CONTROL

In what follows, we present our unified vision of distance fields through our method ReDSDF. We show how to learn deep-distance fields for complex articulated objects, such as robotic manipulators and humans. Next, we describe the integration of the learned distance fields in reactive motion generation for robot WBC and reactive HRI. For our method, we assume a target-centric coordinate system centered at the center of mass of the object or on any relevant fixed point, e.g. the pelvis position of a human.

## A. Regularized Deep Signed Distance Fields

We exploit two key assumptions to derive a unified vision of any-scale deep signed distance fields. First, we assume that the scale of an arbitrary object can be defined using one of its characteristic dimensions, e.g., the radius of the bounding sphere $r .$ Thus, for any point x where the distance to the center of object $\pmb { x } _ { c } \mathrm { ~ i s ~ } \| \pmb { x } - \pmb { x } _ { c } \| _ { 2 } \gg r ,$ , the distance is approximated by $d ( \pmb { x } ) ~ \approx ~ \| \pmb { x } - \pmb { x } _ { c } \| _ { 2 }$ . Exploiting this assumption makes it possible to build a distance function that unifies most definitions of âdistanceâ to a given target. The second key assumption is the dependence of the distance function on the configuration q of the considered target. For example, q can be the joint positions of a manipulator or the human poses detected by a human tracking system.

We define ReDSDF w.r.t. a given (articulated) target with configuration q at any query point x as the distance

$$
d _ { \pmb { q } } ( \pmb { x } ) = \left( 1 - \sigma _ { \pmb { \theta } } ( \pmb { x } , \pmb { q } ) \right) f _ { \pmb { \theta } } ( \pmb { x } , \pmb { q } ) + \sigma _ { \pmb { \theta } } ( \pmb { x } , \pmb { q } ) \| \pmb { x } - \pmb { x } _ { c } \| _ { 2 } ,\tag{1}
$$

where Î¸ is the vector of learnable parameters, q is the targetâs state, $f _ { \pmb \theta } ( \pmb x , \pmb q )$ is a neural network approximator, and the mode switching function $\sigma _ { \pmb { \theta } } ( \pmb { x } , \pmb { q } )$ is defined as

$$
\sigma _ { \pmb { \theta } } ( \pmb { x } , \pmb { q } ) = \mathrm { s i g m o i d } \left( \alpha _ { \pmb { \theta } } \left( \| \pmb { x } - \pmb { x } _ { c } \| _ { 2 } - \rho _ { \pmb { \theta } } \right) \right)\tag{2}
$$

where $\alpha _ { \pmb { \theta } }$ and $\rho _ { \pmb { \theta } }$ are shaping functions, implemented as neural networks. Here, the $\rho _ { \pmb { \theta } }$ function defines a (soft) threshold for switching from a distance w.r.t. the closest point on the targetâs surface to the distance w.r.t. the center of the target. The $\alpha _ { \pmb { \theta } }$ function regulates the sharpness of the change between the two regimes.

The network of ReDSDF, depicted in Figure 2, is composed of 5 fully-connected hidden layers. All hidden layers are 512-dimensional. In addition, we connect the features from the second layer with a 32-dimensional layer to compute $\alpha _ { \pmb { \theta } }$ and $\rho _ { \pmb { \theta } }$ . We use ReLU for all layers except for the output layers, which have their specific activation functions. We limit the range of $\rho _ { \pmb { \theta } } \in ( 0 . 5 , 1 . 5 )$ by applying a Sigmoid function and a bias on the output unit. $\alpha _ { \pmb { \theta } }$ is constrained to be positive using a Softplus activation. The distance output $f _ { \theta }$ is linear.

a) Data generation and augmentation: To train ReDSDF w.r.t. a given (articulated) target, we need to generate an appropriate dataset. We first obtain a point cloud of the object of interest from different viewpoints and at different configurations $\pmb q ^ { [ k ] }$ . Using the Open3D library [45], we estimate the normal direction $\bar { n } _ { i } ^ { [ k ] }$ for each point $\bar { \mathbf { \Phi } } _ { \mathbf { \Phi } } ^ { [ k ] }$ in the point cloud. The normals are used to augment the original point cloud to different distance levels and to regularize the gradient direction of the ReDSDF as shown in the loss function (4). However, the estimation of normal directions may be wrong: training can be unstable if the estimated normals donât match the ones of the neighborhood. This issue can be solved by removing such points from the dataset. Using the estimated normals, we augment the data following the procedure of [27], i.e., we augment the data points to M different distance levels by $\pmb { x } _ { i j } ^ { [ \tilde { k } ] } = \pmb { x } _ { i } ^ { [ k ] } + \bar { \pmb { n } } _ { i } ^ { [ \tilde { k } ] } \bar { d } _ { j } , j =$ $1 , 2 , \cdots , M$ . For each augmented point, we find its closest point from the original point cloud. If the closest point is not the same as the one from which the augmentation starts $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } _ { i } ^ { [ k ] } }$ , we reject this augmented point. Differently from [27], we do not use a fixed step size, but we heuristically select a set of step sizes, considering the scale of the object. Finally, we assign to every generated point $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } _ { i j } ^ { [ k ] } }$ a weight $\omega _ { i } ^ { [ k ] } = M / M _ { s , i } ^ { [ k ] }$ , where $M _ { s , i } ^ { [ \bar { k } ] }$ is the number of the augmented points starting from $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } _ { i } ^ { [ k ] } }$ that are not rejected. The set of points generated by this method can be unnecessarily large to train a good distance field. To reduce the dataset without losing precision, we uniformly down-sample a subset of the augmented dataset. Finally, we obtain the dataset

$$
\begin{array} { r } { \mathcal { D } = \{ \mathbf { x } _ { i j } ^ { [ k ] } , \mathbf { q } ^ { [ k ] } , \bar { d } _ { j } , \bar { \mathbf { n } } _ { i } ^ { [ k ] } , \omega _ { i } ^ { [ k ] } \} . } \end{array}\tag{3}
$$

b) Human-based data generation: An important factor, when considering distance to humans, is the high variability of body shapes. This is particularly important when we need to perform close quarter interaction, e.g., for handovers and shared workspace interactions. Instead of using skeletonbased human reconstruction, we leverage the realistic 3D model of the human body, A Skinned Multi-Person Linear Model (SMPL) [31]. Human 3D meshes are obtained directly from the SMPL model, given the human poses. The points and normals are extracted from the triangular mesh-elements by computing the center of each triangle and the cross product of the edges. We generate the augmented data as mentioned in previous section. We use the AMASS [46] dataset to build up a comprehensive dataset that contains various human poses. In this work, we choose 10, 000 human configurations from the AMASS dataset.

c) Training: We train the model by optimizing the following loss function

$$
\begin{array} { r l r } {  { \mathcal { L } ( \mathcal { D } ) = \sum _ { \mathcal { D } } \omega _ { q } ( x ) ( \bar { d } _ { q } ( x ) - d _ { q } ( x ) ) ^ { 2 } } } \\ & { } & { + ( \| D _ { q } ( x ) \bar { n } _ { q } ( x ) \| _ { 2 } ^ { 2 } + \| N _ { q } ( x ) \nabla _ { x } d _ { q } ( x ) \| _ { 2 } ^ { 2 } ) } \\ & { } & { + \gamma \rho _ { \theta } ( x , q ) ^ { 2 } , ~ ( 4 , . . , d ) } \end{array}\tag{}
$$

where $D _ { q } ( \pmb { x } ) = \mathrm { n u l l } \left( \nabla _ { \pmb { x } } d _ { \pmb { q } } ( \pmb { x } ) \right)$ and ${ \cal N } _ { \pmb q } ( { \pmb x } ) = \mathrm { n u l l } \left( \bar { \pmb n } _ { \pmb q } ( { \pmb x } ) \right)$ are, respectively, the null-space of the gradient of the distance field i.e., the space tangent to the isolines at every point x, and the null space of the normals i.e., the tangent planes to the object. The first component of the loss is computing the squared error of the network distance prediction w.r.t. the target value. The second component of the loss is similar to the one proposed by [27] to align the estimated normals $\bar { n } _ { q } ( { \pmb x } )$ with the normals of the ReDSDF model. The last component of the loss is a regularization term, where $\gamma$ is a regularization coefficient, set in our experiments to 0.02. This regularization is trying to impose the inductive bias of the distance to every point of the state-space. Reducing the output of the learnable function $\rho _ { \pmb { \theta } } ( \pmb { x } , \pmb { q } )$ has the effect of switching the distance regime of the network as soon as possible. This regularization is important in particular where the dataset is sparse. The dataset is split into training, validate and test datasets with the ratio 0.8, 0.1, 0.1.

## B. Robot Motion Control with ReDSDF

ReDSDF can be readily employed within any control and planning framework to provide real-time constraintsâ inference. Reactive motion generation provides a nice framework to showcase the benefit of our deep-distance field for robot control. Namely, we will describe the integration of ReDSDF in a framework for whole-body motion control based on Artificial Potential Fields (APF) [47]. Note that ReDSDF can be integrated to any other type of reactive motion generation, such as Riemannian Motion Policies (RMP) [4], CEP [48]. ReDSDF is not restricted to reactive motion generation, but can be employed by any constrained planner, both as a differentiable constraint and as an energy function. For the sake of simplicity, we focus on APF-based reactive control, as it directly allows showing the benefits of ReDSDF.

We consider the settings where the APFs compute a velocity field for the robotâs end-effector. We define a simple PID controller for tracking the end-effectorâs velocity, but more sophisticated options can be used to generate the task-oriented velocity signal. On top of this, we add an obstacle avoidance velocity signal. Since our method exploits deep neural networks, it is possible to rapidly compute the whole set of collision points in a batch. To perform precise obstacle avoidance on a bimanual mobile manipulator robot of Fig. 5.a, we evaluate the ReDSDF over multiple points sampled on the shell of the arms w.r.t. the obstacle. It is noteworthy that the target obstacle can as well be the other arm or any other part of the articulated body of the robot, allowing us to perform self-collision avoidance. For avoiding self-collisions with the other arm of a bimanual robot, we learn a ReDSDF model of the robot with only a single arm. The learned ReDSDF can be used to avoid obstacle for the other arm. We leverage the symmetry of the robot to construct the ReDSDF for the untrained arm.

For each target obstacle o, we compute the obstacle avoidance energy field as

$$
E _ { o } ( { \pmb x } ) = \left\{ \begin{array} { l l } { 0 } & { d _ { \pmb q } ( { \pmb x } ) > \kappa } \\ { \frac { \bar { \ b v } } { 2 \kappa } \left( d _ { \pmb q } ( { \pmb x } ) - \kappa \right) ^ { 2 } } & { 0 \leq d _ { \pmb q } ( { \pmb x } ) \leq \kappa } \end{array} \right. ,\tag{5}
$$

where $\kappa$ is the maximum obstacle avoidance distance, and vÂ¯ is the repulsive velocity coefficient. For $0 \leq d _ { q } ( x ) \leq \kappa ,$ we obtain the obstacle avoidance velocity as follows

$$
\dot { \pmb { x } } _ { o } = - \nabla _ { \pmb { x } } E _ { o } ( { \pmb x } ) = - \frac { \bar { v } } { \kappa } \left( d _ { \pmb { q } } ( { \pmb x } ) - \kappa \right) \nabla _ { \pmb { x } } d _ { \pmb { q } } ( { \pmb x } ) .\tag{6}
$$

While we could also define the energy for negative distance, we choose not to do it as we are considering the rigid obstacle avoidance task, as a negative distance implies a collision with the obstacle. In case of collision, we stop the robot from taking any other action.

The velocity at each Points of Interest (PoI) xi is computed by combining the repulsive velocity contribution from the set of obstacles O in the environment:

$$
\dot { \pmb { x } } _ { i } = \frac { 1 } { | \mathcal { O } | } \sum _ { o \in \mathcal { O } } \dot { \pmb { x } } _ { i , o } .\tag{7}
$$

To transform this velocity from task-space velocity to the robot configuration space velocity $\mathbf { \delta } \mathbf { q } _ { r } ,$ , we can use the Jacobian matrix $J _ { i } ( { \pmb q } _ { r } )$ computed in the coordinate system defined in $\mathbf { x } _ { i } .$ . While the most common solution is to project the velocity into the configuration space using the Jacobian pseudoinverse $J ^ { \dag } ( q _ { r } )$ , in this work we use the Jacobian transpose $\dot { \pmb q } _ { r , i } ~ = ~ J _ { i } ^ { \top } ( { \pmb q } _ { r } ) \dot { \pmb x } _ { i }$ . This choice is well-known in the literature [49], [50], and it is often used to avoid issues with singularities without losing the convergence guarantee. Finally, we compute the joint velocity by combining the obstacle avoidance velocity with the task velocity ${ \dot { \mathbf { x } } } _ { t }$ as

$$
\dot { \pmb q _ { r } } = J _ { t } ^ { \top } ( \pmb q _ { r } ) \dot { \pmb x } _ { t } + \sum _ { i } ^ { N } \dot { \pmb q } _ { r , i } ,\tag{8}
$$

where N is the number of interest points and $J _ { t } ( { \pmb q } _ { r } )$ is the Jacobian computed in the end-effector frame. Instead of using clipping to enforce the joint velocity limits, we rescale all jointsâ velocities with an appropriate constant, maintaining the same task-space direction while keeping every joint inside its limit.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
0-level mesh

<!-- image-->  
y plane

Fig. 3: Reconstruction of the table distance function  
<!-- image-->  
z plane

<!-- image-->  
z plane (zoom)

## IV. EXPERIMENTS

To evaluate the effectiveness and applicability of ReDSDF to produce smooth distance fields for robot control, we perform qualitative and quantitative comparisons with various baseline methods in a set of simulated control tasks. First, we evaluate the quality of the produced distance fields of ReDSDF against state-of-the-art methods. Next, we test our method as a component of reactive motion generation for WBC and robot control in a shared human-robot workspace. Last, we empirically evaluate the performance of ReDSDF in a realistic HRI scenario, as seen in Fig. 1, whose results can be found in the attached video and in our project site https://irosalab.com/2022/02/28/redsdf/.

## A. Quality of the distance field

We compare ReDSDF against the state-of-the-art methods ECoMaNN [27] and DeepSDF [26]. As both baselines were introduced for approximating the manifold for static objects or sampled points cloud, we compare against them with a static object, e.g., a table in Fig. 3 and a human mesh with fixed pose Fig. 4. For the baseline approaches, we directly apply the available source code provided by the authors. We use the same databases for all methods, but the training data are generated using the approach-specific data augmentation technique. As the computation of an accurate metric for this task is challenging, we will only provide a qualitative evaluation. However, we argue that the performance difference w.r.t. other state-of-the-art methods is clear, and thus this evaluation is already sufficiently convincing that ReDSDF can provide more rational distance fields for robotics environments.

Fig. 3 depicts the results for learning the table manifold. As we can see, the ECoMaNN reconstruction contains artifacts in the estimation of the 0-level curve of the constraint, while our approach reconstructs the mesh without issues. The distance field estimation of ECoMaNN is problematic due to wrong normal estimations on the manifold. Instead, DeepSDF reconstructs the manifold well. However, DeepSDF normalizes every mesh file individually and restricts the networkâs output by a tanh, limiting the distance fieldâs output range. Contrarily, we can observe the superiority of ReDSDF in generating not only a clear geometric manifold of the object, but also smoother and cleaner distance fields at any scale. Next, we compare the distance field reconstruction from a human mesh file in Fig. 4. All methods reconstruct roughly the human shape. However, ReDSDF provides more precise details like the human fingers and ears. The weighting technique significantly improves the reconstruction of the points in cluttered areas. Furthermore, when looking at the shape of the distance field for points far from the center of mass, we can see that ECoMaNN unnaturally deforms the space, e.g., stretching the distance along the hands. DeepSDF, on the other hand, suffers the same scaling problem, while our approach exploits the inductive bias to enforce a well-behaved, even if approximate, distance field.

For ReDSDF, we generate augmented fine-grained data points in the proximity of the target-surface and more sparse data points as we go away from it. This augmentation allows us to have points at any scale, improving the smoothness of the function and obtaining a precise reconstruction of the 0-level constraint. The inductive bias we use to regularize the distance field plays a major role in the quality of the reconstruction of the distance function at any scale. Indeed, this is one of the reasons why we do not need to generate augmentation points in the whole state space, and we can make them sparser going away from the manifold without losing the smoothness of the approximated field.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
0-level mesh (closeup)

<!-- image-->  
x plane (zoom)

<!-- image-->  
y plane  
Fig. 4: Reconstruction of the human distance function

## B. Whole-body Control

For evaluating the applicability of ReDSDF for robot control, we devise a WBC for the bimanual mobile manipulator TIAGo++ robot (Fig. 5a). For our simulated experiment, we focus on the control of the two arms and torso (15 degrees of freedom) but assume a stationary base. However, our approach can be trivially extended to generate a base velocity. We generate the augmented dataset of 10, 000 configurations that only contains one arm (left in our experiment). The distance field of the other arm can be computed by mirroring the PoI through the symmetric plane. We defined 66 PoI distributed on the surface of each arm and computed the distance using ReDSDF w.r.t. the other arm and the robot body. Note that the model is learned by excluding each controlled arm from the robot itself; otherwise, every point would be considered in collision. The learned ReDSDF and PoI of the other arms is illustrated in Fig. 5a. We use a PID controller to provide the control velocity of the endeffector to reach the target. The repulsive velocity that avoids the collision is computed as described in (6). We compare our WBC to one without obstacle avoidance and to another one that uses APFs but computes the distance using spheres to approximate the robot model, a common approach in the literature. We group the spheres into 3 subgroups, i.e., body (18 spheres), left and right arm (6 spheres each). The distances are computed between the spheres from different groups. By adjusting the distance threshold of Îº and the velocity coefficient vÂ¯, we can implement a task-oriented controller that cares more about the end-task, or a cautious controller that places more emphasis on collision avoidance.

Our experimental results are summarized in the first part of Table I. We conducted 1, 000 experiments with different reaching targets for each arm. To generate the target points, we first sample a random point in space for the first arm, and then we generate a second point for the second arm in the vicinity of the first point. The target points that are too close to the robot are rejected. Each task is simulated for 30 s with a control frequency of 60 Hz. The episode terminates once the simulator detects a collision based on the meshes. We regard the task as successful when the end-distance of the end-effector to the target is less than 3 cm. We report the success rate percentage, the absolute number of collisions, and the mean and confidence interval of the final position error, the reach time, smoothness of the generated trajectory Ï as: $\begin{array} { r } { s ( \tau ) = \sum _ { \pmb { q } _ { r } ( i ) \in \tau } \| \ddot { \pmb { q } } _ { r } ( i ) \| ^ { 2 } } \end{array}$ , and of the time spent for computing collisions (C. time).

Due to our design choices, both the sphere-based distance computation and ReDSDF are comparable in terms of success rate when the controller is task-oriented, but ReDSDF only has seven collisions. For the cautious controller, ReDSDF provides zero collisions and satisfactory performance, while the heuristic sphere-based method requires strong repulsive forces to eliminate collisions, hurting the task performance. For the smoothness metric, we consider both changes in the direction of the velocity and changes in magnitude, and for a fair comparison, we only compute the smoothness for the collision-free episodes. ReDSDF generates a continuous distance field, which results in a much smoother controller. For the heuristic, the distance and normal directions are chosen based on the closest sphere of each PoI. The normals may change drastically when the closest sphere varies, which results in jittery movements. The distance computation time is evaluated on an AMD Ryzen 7 3700X 8-Core Processor with a GeForce RTX 2080 SUPER. We compute the distance of the 132 PoI (66 PoI per arm) with ReDSDF in batch on the GPU using PyTorch. For the sphere-based method, 288 distance queries are requested at each time in batch on the GPU to speed up its performance.

<!-- image-->  
(a)

<!-- image-->  
(b)

Fig. 5: (a) ReDSDF for WBC. Left: PoI of the controlled arm that we check to avoid collisions with the rest of the robot. Right: The distance field of the robot, excluding the controlled arm. (b) Human-robot shared-workspace simulated task: human and robot execute sequential pick-and-place-like actions in a tight workspace, where the robot should avoid collisions with the human and the table. The red point cloud represents the 0.1-level ReDSDF of the human and is used for collision-avoidance by the robot controller.  
WBC results
<table><tr><td></td><td>No avoidance</td><td>Sphere-based (task-oriented)</td><td></td><td>ReDSDF (task-oriented) Sphere-based (cautious)</td><td>ReDSDF (cautious)</td></tr><tr><td>Success rate</td><td>44.8%</td><td>80.9%</td><td>82.6%</td><td>38.3%</td><td>73.2%</td></tr><tr><td># collisions</td><td> $5 4 8 / 1 0 0 0$ </td><td>49/1000</td><td>7/1000</td><td>0/1000</td><td>0/1000</td></tr><tr><td>Final err. (cm)</td><td> $1 0 . 4 2 \pm 0 . 5 9$ </td><td> $1 . 4 2 \pm 0 . 1 4$ </td><td> $1 . 2 5 \pm 0 . 1 1$ </td><td> $7 . 2 1 \pm 0 . 4 2$ </td><td> $1 . 9 5 \pm 0 . 1 5$ </td></tr><tr><td>Reach time (s)</td><td> $6 . 1 7 \pm 0 . 4 2$ </td><td> $1 2 . 5 8 \pm 0 . 5 8$ </td><td> $1 2 . 7 4 \pm 0 . 5 8$ </td><td> $2 3 . 9 8 \pm 0 . 5 5$ </td><td> $1 5 . 5 6 \pm 0 . 6 5$ </td></tr><tr><td>Smoothness</td><td> $1 0 . 5 0 \pm 0 . 0 6$ </td><td> $8 1 1 . 8 8 \pm 1 8 3 . 4 0$ </td><td> $1 1 . 6 2 \pm 0 . 1 1$ </td><td> $3 5 5 7 . 5 1 \pm 2 8 4 . 1 5$ </td><td> $1 2 . 3 4 \pm 0 . 2 0$ </td></tr><tr><td>C. time (ms)</td><td> $0 . 1 0 \pm 5 . 4 9 \mathrm { e } . 5$ </td><td> $5 . 7 9 \pm 6 . 2 8 \mathrm { e } { \cdot } 3$ </td><td> $3 . 2 7 \pm 5 . 6 1 \mathrm { e } { - 3 }$ </td><td> $5 . 9 7 \pm 5 . 6 8 \mathrm { e } { - 3 }$ </td><td> $3 . 2 3 \pm 3 . 1 8 \mathrm { e } \cdot 3$ </td></tr><tr><td colspan="6">Shared Workspace results</td></tr><tr><td></td><td>No avoidance</td><td>Sphere-based (task-oriented)</td><td>ReDSDF (task-oriented)</td><td>Sphere-based (cautious)</td><td>ReDSDF (cautious)</td></tr><tr><td># collisions</td><td>935/1000</td><td> $1 7 1 / 1 0 0 0$ </td><td>95/1000</td><td>86/1000</td><td>27/1000</td></tr><tr><td># targets</td><td> $2 . 1 7 2 \pm 0 . 1 9$ </td><td> $6 . 3 5 \pm 0 . 1 8$ </td><td> $5 . 7 8 \pm 0 . 1 6$ </td><td> $4 . 4 4 \pm 0 . 1 4$ </td><td> $4 . 7 3 \pm 0 . 1 6$ </td></tr><tr><td>Smoothness</td><td> $7 1 . 9 5 \pm 0 . 8 5$ </td><td> $2 1 2 0 . 0 0 \pm 1 9 8 . 3 5$ </td><td> $1 3 5 . 5 6 \pm 3 . 6 7$ </td><td> $2 2 8 6 . 6 9 \pm 1 7 4 . 8 2$ </td><td> $6 2 4 . 1 4 \pm 4 3 . 5 0$ </td></tr></table>

TABLE I: Results of WBC and shared workspace experiments in simulation

The computation time for ReDSDF is 50% less than the heuristic. We want to point out that the computation load of ReDSDF grows linearly w.r.t. the number of PoI, while for the sphere-based method it grows quadratically. This advantage allows us to achieve more query points and finer computations of the distance field.

## C. Shared Human-Robot workspace

To test the applicability of ReDSDF for reactive motion generation for challenging HRI tasks, we built a simulation task where a human and a robot are performing sequential pick-and-place-like actions in a shared workspace. To simulate realistic movements, we record a set of trajectories from an actual human performing the task, and we infer the SMPL trajectories with the help of VIBE [30]. We replay these trajectories in our PyBullet simulator, constituting a novel way of building simulation tasks for HRI with the robot dynamically interacting with the human (Fig. 5b). Instead of reaching the fixed targets as in WBC task, we generate 9 targets for each task to mimic the pick-and-place scenario, a new target will be activated once the old target has been reached. We assume that the human is not responsive to the robot in the current setting. While this assumption is unrealistic, we want to test the reactiveness of the robot controller while interacting in a tight workspace with a human. If the human actively avoids the robot, it would be impossible to test if our controller successfully avoids collisions. Indeed, we can see the proposed scenario as an approximation of a human not paying attention to the robot while performing a shared-workspace task.

Table I contains the results of this experiment. We compared the collision avoidance behavior based on ReDSDF with two simple baselines. The first, trivial baseline is to plan without considering the human at all. The second baseline is a prevalent approach in the literature that considers only the human handâs position and reacts to hand movements. We developed similar task-oriented and cautious controller as in Sec. IV-B. For all methods, we also include a potential field to prevent collision with the table. Inspecting the results, we can observe that our controllers with ReDSDF can provide smoother distance fields that result in smoother robot movements, with significantly fewer collisions compared to the baseline methods. Even though in the target-oriented case the heuristic achieves more targets, it suffers from a higher number of collisions. Overall, our results validate our assumption that ReDSDF is effective and smooth when used in real-time as a constrained function for safe and reactive HRI. As proof of concept, Fig. 1 and the accompanying video provides a real-world experiment in a realistic HRI handover scenario, where we can inspect the any-scale smooth performance of ReDSDF.

## V. CONCLUSIONS

Safety is a critical property for autonomous robots, both for planning actions that respect their hardware and the surrounding world and when robots interact with humans in shared workspaces. In this work, we presented the Regularized Deep Signed Distance Fields (ReDSDF) framework, which generalizes the concept of SDF to arbitrary articulated objects. We regularize the training of the distance function with an inductive bias to ensure smooth distance fields that can be used at any scale. Using the proposed technique, we obtain high-quality, smooth distance fields for any point in the space. Our method can be easily deployed to generate reactive motions both in the context of a whole-body control (WBC) and in a human-robot shared workspace scenario, using real human data. The reactive motion can reliably avoid obstacles while reducing the interference considerably with the main task, allowing us to outperform baselines methods by a significant margin in many metrics. The ReDSDF structure can be easily deployed in real robot applications, as it can provide accurate distance fields in different resolutions, from rough distance estimates for obstacle avoidance during navigation to fine-grained resolutions in close interactions with humans. Though we demonstrated the applicability of ReDSDF to reactive motion generation, thanks to its smooth differentiable nature, it can be easily integrated into different pipelines, such as constrained task and motion planning or constrained reinforcement learning.

## REFERENCES

[1] M. da Silva Arantes, C. F. M. Toledo, B. C. Williams, and M. Ono, âCollision-free encoding for chance-constrained nonconvex path planning,â IEEE TRO, 2019.

[2] N. Ratliff, M. Zucker, J. A. Bagnell, and S. Srinivasa, âChomp: Gradient optimization techniques for efficient motion planning,â in ICRA, 2009.

[3] P. Liu, D. Tateo, H. B. Ammar, and J. Peters, âRobot reinforcement learning on the constraint manifold,â in CoRL, 2022.

[4] N. D. Ratliff, J. Issac, D. Kappler, S. Birchfield, and D. Fox, âRiemannian motion policies,â arXiv:1801.02854, 2018.

[5] F. S. Barbosa, B. Lacerda, P. Duckworth, J. Tumova, and N. Hawes, âRisk-aware motion planning in partially known environments,â IEEE CDC, 2021.

[6] C.-P. Lam, C.-T. Chou, K.-H. Chiang, and L.-C. Fu, âHuman-centered robot navigationâtowards a harmoniously humanârobot coexisting environment,â IEEE TRO, 2010.

[7] P. T. Singamaneni, A. Favier, and R. Alami, âHuman-aware navigation planner for diverse human-robot interaction contexts,â in IROS, 2021.

[8] F. Ferraguti, C. T. Landi, S. Costi, M. Bonfe, S. Farsoni, C. Secchi, and \` C. Fantuzzi, âSafety barrier functions and multi-camera tracking for humanârobot shared environment,â Robotics and Autonomous Systems, vol. 124, p. 103388, 2020.

[9] M. N. Finean, W. Merkt, and I. Havoutis, âSimultaneous scene reconstruction and whole-body motion planning for safe operation in dynamic environments,â in IROS, 2021.

[10] B. Curless and M. Levoy, âA volumetric method for building complex models from range images,â in SIGGRAPH, 1996.

[11] N. Ratliff, M. Toussaint, and S. Schaal, âUnderstanding the geometry of workspace obstacles in motion optimization,â in ICRA, 2015.

[12] D. R. Canelhas, âTruncated signed distance fields applied to robotics,â Ph.D. dissertation, Orebro University, 2017. Â¨

[13] H. Oleynikova, A. Millane, and Z. Taylor et al., âSigned distance fields: A natural representation for both mapping and planning,â in RSS Workshops, 2016.

[14] K. Saulnier, N. Atanasov, G. J. Pappas, and V. Kumar, âInformation theoretic active exploration in signed distance fields,â in ICRA, 2020.

[15] S. Izadi, D. Kim, O. Hilliges, and D. Molyneaux et al., âKinectfusion: real-time 3d reconstruction and interaction using a moving depth camera,â in ACM Symp. on User interface software and tech., 2011.

[16] E. Vespa, N. Nikolov, and M. Grimm et al., âEfficient octree-based volumetric slam supporting signed-distance and occupancy mapping,â IEEE RA-L, 2018.

[17] L. Han, F. Gao, B. Zhou, and S. Shen, âFiesta: Fast incremental euclidean distance fields for online motion planning of aerial robots,â in IROS, 2019.

[18] M. Mukadam, J. Dong, X. Yan, F. Dellaert, and B. Boots, âContinuoustime gaussian process motion planning via probabilistic inference,â IJRR, 2018.

[19] J. Schulman, Y. Duan, and J. Ho et al., âMotion planning with sequential convex optimization and convex collision checking,â IJRR, 2014.

[20] L. Han, T. Zheng, L. Xu, and L. Fang, âOccuseg: Occupancy-aware 3d instance segmentation,â in CVPR, 2020.

[21] W. Xu, H. Wang, F. Qi, and C. Lu, âExplicit shape encoding for realtime instance segmentation,â in ICCV, 2019.

[22] B. Xu, W. Li, and D. Tzoumanikas et al., âMid-fusion: Octree-based object-level multi-instance dynamic slam,â in ICRA, 2019.

[23] M. Grinvald, F. Furrer, and T. Novkovic et al., âVolumetric instanceaware semantic mapping and 3d object discovery,â IEEE RA-L, 2019.

[25] S. Peng, M. Niemeyer, L. Mescheder, M. Pollefeys, and A. Geiger, âConvolutional occupancy networks,â in ECCV, 2020.

[26] J. J. Park and P. Florence et al., âDeepsdf: Learning continuous signed distance functions for shape representation,â in CVPR, 2019.

[27] G. Sutanto, I. M. R. Fernandez, and P. Englert et al., âLearning equalityÂ´ constraints for motion planning on manifolds,â in CoRL, 2020.

[28] H. Jiang, J. Cai, and J. Zheng, âSkeleton-aware 3d human shape reconstruction from point clouds,â in CVPR, 2019.

[29] B. L. Bhatnagar, C. Sminchisescu, C. Theobalt, and G. Pons-Moll, âCombining implicit function learning and parametric models for 3d human reconstruction,â in ECCV, 2020.

[30] M. Kocabas, N. Athanasiou, and M. J. Black, âVibe: Video inference for human body pose and shape estimation,â in CVPR, 2020.

[31] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, âSmpl: A skinned multi-person linear model,â ACM TOG, 2015.

[32] B. Deng, J. P. Lewis, T. Jeruzalski, G. Pons-Moll, G. Hinton, M. Norouzi, and A. Tagliasacchi, âNasa neural articulated shape approximation,â in ECCV. Springer, 2020.

[33] J. Mu, W. Qiu, A. Kortylewski, A. Yuille, N. Vasconcelos, and X. Wang, âA-sdf: Learning disentangled signed distance functions for articulated shape representation,â in ICCV, 2021.

[34] A. Gropp, L. Yariv, N. Haim, M. Atzmon, and Y. Lipman, âImplicit geometric regularization for learning shapes,â in ICML. PMLR, 2020.

[35] M. Atzmon and Y. Lipman, âSald: Sign agnostic learning with derivatives,â in ICLR, 2020.

[36] M. N. Finean, W. Merkt, and I. Havoutis, âPredicted composite signeddistance fields for real-time motion planning in dynamic environments,â in ICAPS, 2021.

[37] S. Xu, G. Li, and J. Liu, âObstacle avoidance for manipulator with arbitrary arm shape using signed distance function,â in ROBIO, 2018.

[38] J. J. Quiroz-Omana and B. V. Adorno, âWhole-body control with (self) Ë collision avoidance using vector field inequalities,â IEEE RA-L, 2019.

[40] N. Vahrenkamp, H. Arnst, and M. Wachter et al., âWorkspace analysis Â¨ for planning human-robot interaction tasks,â in Humanoids, 2016.

[41] P. Papadakis, A. Spalanzani, and C. Laugier, âSocial mapping of human-populated environments by implicit function learning,â in IROS, 2013.

[42] B. Lewandowski, T. Wengefeld, and S. Muller et al., âSocially Â¨ compliant human-robot interaction for autonomous scanning tasks in supermarket environments,â in RO-MAN, 2020.

[43] J. Corrales, F. Candelas, and F. Torres, âSafe humanârobot interaction based on dynamic sphere-swept line bounding volumes,â Robotics and Computer-Integrated Manufacturing, 2011.

[44] P. Svarny, M. Tesar, J. K. Behrens, and M. Hoffmann, âSafe physical hri: Toward a unified treatment of speed and separation monitoring together with power and force limiting,â in IROS, 2019.

[45] Q.-Y. Zhou, J. Park, and V. Koltun, âOpen3D: A modern library for 3D data processing,â arXiv:1801.09847, 2018.

[46] N. Mahmood, N. Ghorbani, N. F. Troje, and et al., âAMASS: Archive of motion capture as surface shapes,â in ICCV, 2019.

[47] O. Khatib, âReal-time obstacle avoidance for manipulators and mobile robots,â in ICRA, 1985.

[48] J. Urain, A. Li, P. Liu, C. Dâeramo, and J. Peters, âComposable energy policies for reactive motion generation and reinforcement learning,â in Robotics: Science and Systems (RSS), 2021.

[49] S. Chiaverini, L. Sciavicco, and B. Siciliano, âControl of robotic systems through singularities,â in Advanced Robot Control, 1991.

[50] S. R. Buss, âIntroduction to inverse kinematics with jacobian transpose, pseudoinverse and damped least squares methods,â IEEE Journal of Robotics and Automation, 2004.