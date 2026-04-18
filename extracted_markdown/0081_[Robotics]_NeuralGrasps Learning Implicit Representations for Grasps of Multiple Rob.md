# NeuralGrasps: Learning Implicit Representations for Grasps of Multiple Robotic Hands

Ninad Khargonkar1 Neil Song2 Zesheng Xu1 Balakrishnan Prabhakaran1 Yu Xiang1

1The University of Texas at Dallas 2St. Markâs School of Texas

{ninadarun.khargonkar,zesheng.xu,bprabhakaran,yu.xiang}@utdallas.edu 23songn@smtexas.org

Abstract: We introduce a neural implicit representation for grasps of objects from multiple robotic hands. Different grasps across multiple robotic hands are encoded into a shared latent space. Each latent vector is learned to decode to the 3D shape of an object and the 3D shape of a robotic hand in a grasping pose in terms of the signed distance functions of the two 3D shapes. In addition, the distance metric in the latent space is learned to preserve the similarity between grasps across different robotic hands, where the similarity of grasps is defined according to contact regions of the robotic hands. This property enables our method to transfer grasps between different grippers including a human hand, and grasp transfer has the potential to share grasping skills between robots and enable robots to learn grasping skills from humans. Furthermore, the encoded signed distance functions of objects and grasps in our implicit representation can be used for 6D object pose estimation with grasping contact optimization from partial point clouds, which enables robotic grasping in the real world1.

Keywords: Robot Grasping, Neural Implicit Representations, Grasp Transfer, Grasping Contact Modeling, 6D Object Pose Estimation

## 1 Introduction

Robot manipulation is a fundamental research problem in robotics. If we want to have robots that can perform tasks to assist humans autonomously, we need to enable robots to grasp and manipulate objects and use tools to perform tasks. Robots have different grippers ranging from two-finger parallel grippers to five-finger anthropomorphic robotic hands. Usually, manipulation research typically focuses on one type of robot gripper. For instance, two-finger grippers are widely studied due to their simplicity in planning and control. Most commercial robots designed for research adopt twofinger grippers such as the Franka Emika Panda arm, the Fetch mobile manipulator, and the Baxter robot. As a result, manipulation skills learned from these robots are limited to two-finger grippers. It is unclear how to transfer or share these manipulation skills to robots with different grippers, especially for skills learned from imitating humans or reinforcement learning [1, 2]. Similarly, skills learned for multi-finger grippers are not transferable to two-finger grippers [3].

In this work, we study how to transfer grasps between different robot grippers. First, grasp transfer enables robots with different grippers to share grasping skills. Second, we consider the human hand to be a special robot gripper, so we can transfer human grasps to robot grippers. This will enable robots to learn grasping skills from human demonstrations. Learning from human demonstrations is very valuable in semantic grasping [4] or task-orientated grasping [5] where robots need to grasp objects according to the tasks. To achieve this goal, we introduce a novel implicit representation of multiple robotic hands learned using deep neural networks. Our representation learning is motivated by DeepSDF [6] and Grasping Fields [7] which learn continuous Signed Distance Functions (SDFs) for 3D shapes of objects and human hands. We learn continuous SDFs for objects and multiple robotic hands. Our novelty lies in learning a latent space of grasps from multiple robotic hands, where a latent vector in this space encodes a grasp from a specific robotic hand. Importantly, the distance metric in the latent space encodes the similarity between grasps across different grippers.

<!-- image-->  
Figure 1: Barnes-Hut t-SNE visualization [9] using our learned embeddings of grasps from five different robotic hands.

Therefore, given a grasp from one gripper, we can retrieve the closest grasp from another gripper using the distance metric in the latent space. This property enables grasp transfer between different grippers including human hands. Fig. 1 shows the t-SNE visualization of grasps from five different robotic hands using our learned latent vectors for two objects. We demonstrate grasp transfer from human hands to a Fetch mobile manipulator in our experiments.

Specifically, we use a deep neural network trained to encode an object and a number of grasps from multiple grippers. We use an object-centric view where all the grasps are defined with respect to the object coordinate frame. Given a latent vector z and a 3D point $\bar { \bf x } \in \mathcal { R } ^ { 3 }$ as input, the network predicts the SDF values of the 3D point to the object and to the grasp of a gripper corresponding to z. During training, in addition to the loss function on predicted SDF values, we utilize a triplet loss function [8] to learn the distance metric in the latent space. In order to measure similarity between grasps from different grippers, we consider the contact maps of the grippers on the object, and use the similarity between contact maps to measure the grasp similarity. The triplet loss function extends the distance metric between contact maps to the latent space. In this way, the learned latent space preserves the similarity between grasps across different grippers.

Another advantage of our implicit representation is its use for real-world grasping. The learned SDFs of objects and grasps are fully differentiable functions. This property enables us to solve an optimization problem to estimate the 6D pose of object given a partial point cloud of the object as input. During this optimization, we also consider the contact of a grasp and the point cloud of the object. With the estimated 6D object pose, a robot can use the encoded grasps to grasp the object. We demonstrated that optimizing contact between a grasp and the input point cloud can improve grasping success rate in the real world.

In summary, our contributions in this work are i) Presenting a novel implicit representation for grasps of multiple robotic hands; ii) Our representation enables grasp transfer between different robot grippers and human hands; iii) The representation can be used to solve object pose and optimize contact between grasps and observed point clouds in real-world grasping; iv) Novel grasping dataset with multiple robotic hands is introduced for multi-gripper transfer and to encourage future research.

## 2 Related Work

Neural Implicit Representations. Neural implicit representations for scenes and objects have attracted widespread attention recently. Occupancy networks [10] learn continuous occupancy functions of 3D shapes using neural networks. DeepSDF [6] trains neural networks to encode the Signed Distance Functions (SDFs) of 3D shapes, while Neural Radiance Fields (NeRFs) [11] encode both the geometry and color information of scenes into neural networks. The advantages of such implicit representations are: i) they are compact. For instance, a single DeepSDF network can encode a couple thousands of 3D shapes from the same category. NeRFs can encode large scenes with details with a single network. ii) The implicit representations are differentiable. So they can be easily used in gradient-based optimization to solve downstream tasks. For example, DeepSDF can be used for

3D reconstruction from partial observations. Grasping Fields [7] can synthesize humans grasps and reconstruct hands and objects jointly. The recently proposed Neural Descriptor Fields [12] learn feature descriptors of 3D shapes for robot manipulation. The key difference between our work and related work is the implicit representation learning of grasps from multiple robotic hands and using the representation for both grasp transfer across grippers and object pose estimation for grasping.

Grasping with Multiple Robotic Hands. Traditional grasp planning methods such as the Graspit! simulator [13] use analytic approaches to access the grasp quality. These grasp quality measurements usually employ task wrench space analysis [14, 15] or force closure analysis [16, 17]. Analytic grasp planning methods can deal with different objects and different robotic hands. However, the main limitation of these traditional approaches is that they require full state information about objects such as shape and pose. They cannot work with partial observations of objects, e.g., point clouds from RGB-D cameras. Recent data-driven approaches for grasp planning utilize large-scale datasets with planned grasps [18, 19] and machine learning techniques to learn models that can work with partial observations [20, 21, 22, 23, 24]. However, majority of these works only focus on one type of robotic hand, especially, the two-finger parallel gripper. An exception is the UniGrasp [25] that considers gripper attributes in learning to detect grasping contact points on objects with a neural network. It can handle multiple grippers in detecting contact points for grasping. Our work differs from UniGrasp in that our implicit representations of objects and grasps enable grasp transfer between grippers and object pose estimation with grasp contact optimization. Another work closely related to ours is the ContactGrasp [26] that uses explicit contact maps to transfer grasps between human hands and robotic grippers. In contrast, our approach learns contact maps implicitly and uses a common latent space of grasps from human hands and robotic grippers for grasp transfer.

## 3 NeuralGrasps

Our goal is to learn representations over different robotic hands that can be used for downstream tasks such as grasping and grasp transfer. We leverage recent progress in neural implicit representations for representing 3D shapes. Specifically, DeepSDF [6] models the geometry of an object o by learning the signed distance function $f _ { \mathrm { S D F } } ( { \bf \bar { x } } ; o ) : \mathcal { R } ^ { 3 }  \mathcal { R }$ over the object. Consequently, the signed distance function implicitly represents the surface points via the zero level set. DeepSDF models the 3D shapes of a set of objects by introducing a latent code for each object: $f _ { \mathrm { S D F } } ( \mathbf { x } , \mathbf { z } _ { i } )$ , where $\mathbf { z } _ { i } \in \mathcal { R } ^ { d }$ with d the dimension of the latent space and $i = 1 , 2 , \dots , N$ for N shapes.

## 3.1 Grasp Encoding Network

Given a dataset of grasps from multiple grippers over an object o, we represent each grasp between a gripper and the object via the signed distance functions of both the gripper and the object in a normalized, object-centric 3D space. Motivated by DeepSDF [6] and Grasping Fields [7], we jointly model the two signed distance functions using a grasp encoding network. We utilize an autodecoder based formulation as in DeepSDF where the network output is conditioned upon a latent vector $\mathbf { z } \in \mathcal { R } ^ { d }$ corresponding to each grasping scene in the dataset. So the grasp encoding network models the following function: $f _ { \mathrm { S D F } } ( \mathbf { \check { x } } , \mathbf { z } ; \theta _ { o } ) : \mathcal { R } ^ { 3 } \times \mathcal { R } ^ { d }  \mathcal { R } ^ { 2 }$ where $\theta _ { o }$ denotes the network parameters corresponding to the model for object o. The output of the network is two SDF values for the query point x: one for the object $f _ { \mathrm { S D F } } ^ { o } ( \mathbf { x } ; \mathbf { z } , \theta _ { o } )$ and the other one for the gripper $f _ { \mathrm { S D F } } ^ { g } ( \mathbf { x } ; \mathbf { z } , \theta _ { o } )$ The network architecture is illustrated in $\operatorname { F i g } . 2$ . Note that for a given grasp encoded by its latent code $\mathbf { z } ,$ the contact point set C between the gripper and the object is implicitly represented by the zero level set: $\mathcal { C } = \{ \mathbf { x } \in \mathcal { R } ^ { 3 } \mid f _ { \mathrm { S D F } } ( \mathbf { x } , \mathbf { z } ; \theta _ { o } ) = \mathbf { \bar { 0 } } \}$ . Because these are 3D points both on the surface of the gripper and on the surface of the object.

## 3.2 Grasp Similarity

We aim to constrain the latent vector z for a grasp to be close to the latent vectors of similar grasps. Since robot grippers have different kinematics, it is non-trivial to measure grasp similarity from different grippers. We consider an object-centric formulation here. We rely on the contact regions on the object to establish a similarity/distance measure over the grasp set. This follows the intuition that two similar grasps probably interact with similar regions of the object.

<!-- image-->  
Figure 2: Left) Our network architecture with the training losses. Right) Illustration of the grasp transfer process during inference. Yellow modules indicate trainable parameters.

Contact Map Representation. To compare grasps on the basis of their contact regions on the object, we compute a contact map Ï over the object points given a grasp. Given a grasping scene with two point clouds $P _ { o } , P _ { q } \subset \mathcal { R } ^ { 3 }$ representing the object points and the gripper points, respectively. Let $d ( p _ { o } , P _ { g } )$ denote the distance between a point $p _ { o } ~ \in ~ P _ { o }$ to the closest point in $\begin{array} { r } { \overbar { P } _ { g } , \mathrm { i . e . , } \ \overbar { d } ( p _ { o } , P _ { g } ) = \operatorname* { m i n } _ { p _ { g } \in P _ { g } } \ d ( p _ { o } , p _ { g } ) } \end{array}$ , where we use the standard $L _ { 2 }$ distance for $d ( p _ { o } , p _ { g } )$ in our experiments. To obtain the object-centric contact map Ï, we simply take $\phi \in \mathcal { R } ^ { | O | }$ with each dimension corresponding to an object point $p _ { o } ~ \in ~ P _ { o }$ . Then we define the contact map as $\begin{array} { r } { \phi ( p _ { o } ; p _ { o } \in P _ { o } ) = \exp ( \frac { - d ( p _ { o } , P _ { g } ) } { \alpha } ) } \end{array}$ where $\alpha = 0 . 0 5 m$ is a constant to penalize and score down object points with $d ( \cdot , \bar { P _ { g } } )$ value higher than Î±. Using this formulation, two grasps $g _ { 1 } , g _ { 2 }$ are considered to be similar if $\phi _ { g _ { 1 } } \sim \phi _ { g _ { 2 } }$ Note that the contact map is only assumed to be known at training time, and hence the unseen (test) grasps do not require a contact map for inference. A few representative visualization of contact maps across different grippers are shown in Fig. 3.

<!-- image-->  
Figure 3: Illustration of several ground truth and inferred contact maps. Brighter regions correspond to the contact regions.

## 3.3 Learning the Implicit Representations

For a given object $^ { O , }$ we train the network parameters $\theta _ { o }$ on a dataset of grasps across multiple grippers. Each training sample $T _ { i } = \{ \phi _ { i } , X _ { i } \bar  \} , i = 1 , . . . N$ , contains the contact map Ïi for a grasp $\phi _ { i }$ and a set of (query, SDF) pairs: $X _ { i } = \{ ( \mathbf { x } _ { i } ^ { j } , S D F ( \mathbf { x } _ { i } ^ { j } ) ) _ { i = 1 } ^ { M } \mid \mathbf { x } _ { i } ^ { j } \in \mathcal { R } ^ { 3 } ; S D F ( \mathbf { x } _ { i } ^ { j } ) \in \mathcal { R } ^ { 2 } \}$ , where M indicates the number of query points and $S D F ( \mathbf { x } _ { i } ^ { j } ) = ( S D F _ { o } ( \mathbf { x } _ { i } ^ { j } ) , S D F _ { g } ( \mathbf { x } _ { i } ^ { j } ) )$ consists of the ground truth signed distances to the object and the gripper, respectively. Similar to the auto-decoder formulation in DeepSDF [6], the training proceeds by pairing each grasp (training sample) with a latent vector $\mathbf { z } _ { i }$ and minimizing the loss function which consists of the two terms described below. We assume a zero-mean Gaussian as the prior over the latent vectors.

Reconstruction Loss. The reconstruction loss $\mathcal { L } _ { \mathrm { S D F } }$ is the $L _ { 1 }$ distance between the ground truth and the predicted SDF values for the gripper and the object over an input query point. A standard practice in learning SDFs is to clamp both the ground truth and predicted distances to be within a specific range [âÎ´, Î´] with a clamp function: clamp $( x , \delta ) : = \operatorname* { m i n } ( \delta , \operatorname* { m a x } ( - \delta , x ) )$ , where Î´ is a parameter used to control the learning of SDF to be within a certain distance of the surface $( \delta = 0 . 0 5 m$ in our experiments). In this way, the learning focuses on the regions of the shape boundary. The reconstruction loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S D F } } ( \mathbf { x } , \mathbf { z } , \theta _ { o } ) = | \mathrm { c l a m p } ( f _ { \mathrm { S D F } } ( \mathbf { x } , \mathbf { z } ; \theta _ { o } ) , \delta ) - \mathrm { c l a m p } ( S D F ( \mathbf { x } ) , \delta ) | . } \end{array}\tag{1}
$$

Triplet Loss in the Latent Space. We constrain the latent vectors to represent a notion of similarity between the encoded grasps by utilizing the contact map $\phi .$ . Our goal is to encourage ${ \bf z } _ { g _ { 1 } } , { \bf z } _ { g _ { 2 } }$ to be close to each other in the latent space if grasps $g _ { 1 } , g _ { 2 }$ are similar to each other on the basis of their contact maps $\phi _ { g _ { 1 } } , \phi _ { g _ { 2 } }$ . We explicitly model such a constraint over the latent vectors during training since we use an encoder-less architecture. We make use of the triple loss [8] with a variable margin for each triplet of latent vectors $\left( \mathbf { z } _ { a } , \mathbf { z } _ { p } , \mathbf { z } _ { n } \right)$ . The triplet $( a , p , n )$ represents the anchor, the positive and the negative training samples. We use the similarity between contact maps to define positive and negative examples w.r.t an anchor example. The triplet loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r i p l e t } } ( \mathbf { z } _ { a } , \mathbf { z } _ { p } , \mathbf { z } _ { n } ) = \operatorname* { m a x } \{ d ( \mathbf { z } _ { a } , \mathbf { z } _ { p } ) - d ( \mathbf { z } _ { a } , \mathbf { z } _ { n } ) + m ( a , p , n ) \ , \ 0 \} , } \end{array}\tag{2}
$$

where $d ( \cdot , \cdot )$ is the $L _ { 2 }$ distance and $m ( a , p , n )$ is the margin. In the original triplet loss function, the margin is a constant. In our case, we define the margin according to the similarity between grasp contact maps. Therefore, we can induce the grasp similarity into the latent space. The variable margin is defined as $m ( a , p , n ) = | | \phi _ { a } - \phi _ { n } | | \overline { { 2 } } - \overline { { | } } | | \phi _ { a } - \phi _ { p } \overline { { | } } | | _ { 2 }$ using $L _ { 2 }$ distance between contact maps. Without loss of generality, given an anchor and two other grasps, we can select the one with smaller contact map distance as the positive example and the other one as negative example. Therefore, the margin $m ( a , p , n )$ is always greater than 0 for every sampled triplet. For each training batch, we randomly sample a fixed number of triplets and compute the mean of the triplet losses.

Training Optimization. Overall, we solve the following optimization problem to estimate the network parameters $\theta _ { o }$ and latent codes $\{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N }$ for a set of N grasps from multiple grippers:

$$
\theta _ { o } ^ { * } , \{ \mathbf { z } _ { i } ^ { * } \} _ { i = 1 } ^ { N } = \arg \operatorname* { m i n } _ { \theta _ { o } , \{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N } } \Big [ \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } \mathcal { L } _ { \mathrm { S D F } } ( \mathbf { x } _ { i } ^ { j } , \mathbf { z } _ { i } , \theta _ { o } ) + \sum _ { ( a , p , n ) } \mathcal { L } _ { \mathrm { t r i p l e t } } ( \mathbf { z } _ { a } , \mathbf { z } _ { p } , \mathbf { z } _ { n } ) + \frac { 1 } { \sigma ^ { 2 } } \sum _ { i = 1 } ^ { N } \| \mathbf { z } _ { i } \| _ { 2 } ^ { 2 } \Big ] ,\tag{3}
$$

Here, each grasping scene contains M points and Ï is the standard deviation of a zero-mean multivariate-Gaussian prior distribution on the latent codes $\{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N }$

## 3.4 Solving Downstream Tasks with Our Implicit Representations

Shape Reconstruction. Given a latent vector z corresponding to a grasp, we can perform inference over the learned network to reconstruct the shape of the object and the shape of the gripper. This is achieved by querying a set of 3D points and then obtaining the object surface points $P _ { o } = \{ \mathbf { x } \in$ $\mathcal { R } ^ { 3 } \mid f _ { \mathrm { S D F } } ^ { o } ( \mathbf { x } , \mathbf { \bar { z } } ; \mathbf { \bar { \theta } } _ { o } ) = 0 \}$ } and the gripper surface $P _ { g } = \{ { \bf x } \in { \bf \bar { \mathcal { R } } } ^ { 3 } \mid f _ { \mathrm { S D F } } ^ { \tilde { g } } ( { \bf x } , { \bf z } ; \theta _ { o } ) \ \stackrel { . } { = } 0 \}$ . The contact points between the grasp and the object are intersection of the two sets of points. In cases when the latent vector z is unknown but we observe a set of points with their SDF values $( \mathbf { x } ^ { j } , S D F ( \mathbf { x } ^ { j } ) ) _ { j = 1 } ^ { M } ,$ we can solve the following optimization problem to estimate the latent code:

$$
\mathbf { z } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { z } } \Big [ \sum _ { j = 1 } ^ { M } \mathcal { L } _ { \mathrm { S D F } } ( \mathbf { x } ^ { j } , \mathbf { z } , \theta _ { o } ) + \frac { 1 } { \sigma ^ { 2 } } \| \mathbf { z } \| _ { 2 } ^ { 2 } \Big ] .\tag{4}
$$

In the real world, the query points $\mathbf { x } ^ { j }$ are usually from depth sensors. We can approximate the SDF samples using a similar scheme as shown in [6] by sampling points at a small distance away from surface points along the normal. On obtaining the latent code $\mathbf { z } ^ { \ast }$ , we can reconstruct the shapes of the object and the gripper.

Grasp Transfer. Due to the triplet loss imposed over the latent space, the inferred latent code $\mathbf { z } ^ { \ast }$ for an unseen grasp also follows the notion of grasp similarity. This metric learning constraint allows us to retrieve a most similar grasp in the dataset on the basis of the nearest neighbor in the latent code space as shown in Fig. 2. The nearest neighbor grasp is not constrained to be from the input gripper, and hence such a framework makes it useful in a grasp transfer scenario where the demo and target grippers might differ. For example, the input grasp can be from a human as grasping demonstration.

6D Object Pose Estimation with Grasp Contact. Since our implicit representation encodes the SDFs of an object and its grasps, we can utilize it to estimate the 6D pose of the object, i.e., 3D rotation R and translation t, given camera observations in the real world. Suppose we can segment a set of 3D points of the object in the camera frame $\{ \mathbf { x } _ { c } ^ { j } \} _ { j = 1 } ^ { M }$ . If we transform these points into the object coordinate frame using $( R , \mathbf { t } )$ , they should have SDF values of the object close to zeros, since the transformed points are on the object surface. In addition, if the final goal is to execute a grasp g to grasp the object, we can optimize the object pose such that the contact points of grasp g are close to the observed 3D points. Let $\{ \mathbf { x } _ { g } ^ { k } \} _ { k = 1 } ^ { L } \subset \{ \mathbf { x } _ { c } ^ { \bar { j } } \} _ { j = 1 } ^ { M }$ be the contact points of grasp g in the camera frame. We solve the following optimization problem to estimate the object pose:

<!-- image-->  
Figure 4: Objects and robotic hands in our multi-hand grasping dataset.

$$
R ^ { * } , \mathbf { t } ^ { * } = \arg \operatorname* { m i n } _ { R , \mathbf { t } } \Big [ \sum _ { j = 1 } ^ { M } | f _ { \mathrm { S D F } } ^ { o } ( R \mathbf { x } _ { c } ^ { j } + \mathbf { t } , \mathbf { z } ; \theta _ { o } ) | + \sum _ { k = 1 } ^ { L } | f _ { \mathrm { S D F } } ^ { g } ( R \mathbf { x } _ { g } ^ { k } + \mathbf { t } , \mathbf { z } ; \theta _ { o } ) | \Big ] ,\tag{5}
$$

where $f _ { \mathrm { S D F } } ^ { o }$ and $f _ { \mathrm { S D F } } ^ { g }$ are the SDF predictions of the object and the gripper from the network and z is the latent code for grasp g. Starting from an initial object pose, we can apply gradient descent to optimize object pose estimation. More details on object pose estimation are provided in the supplementary materials.

## 4 Multi-Hand Grasping Dataset

Most of the current datasets for object manipulation exclusively focus either on parallel jaw grippers [19] or human hand-object interaction [27]. Such datasets lack the data needed for learning neural implicit representations of multiple grippers. In order to train our proposed model, we generated a synthetic dataset of common robot grippers grasping objects. Creating a synthetic dataset of grasps over 3D models of real world objects allows for high detail grasping scenes containing the dense point clouds, contact regions, and signed distance function values.

Grippers and Objects. We selected five gripper models and seven objects from the YCB object set [28]. The gripper set includes four robot grippers across a range of the number of fingers: 2-finger Fetch and Franka-Panda grippers, 3-finger Barrett gripper, 4-finger Allegro finger, and finally a human hand model. The Fetch gripper was also used in our real-world experiments. The motivation to include a diverse set of multi-fingered grippers was to investigate how well such an encoding scheme works across grippers with different kinematic configurations. The seven objects from the YCB set are cracker box, tomato soup can, mustard bottle, pudding box, gelatin box, potted meat can, and bleach cleanser. Fig. 4 shows the objects and grippers grasps in our dataset.

Grasp Generation. To generate diverse multi-hand grasps over the YCB objects, we utilized the Graspit! simulator [13]. We initially generated a large number of grasps from GraspIt! and later sampled the initial set to generate a fixed amount of grasps for each gripper using the farthest point sampling algorithm which ensures diversity of the 3D position in the sampled grasp set. After the sampling stage, each grasping scene, i.e., an object with a gripper in a grasping pose, was loaded in the Pybullet [29] environment, and dense point clouds were rendered for the grasping scene using multiple viewpoints. Using the rendered point clouds, we generated the SDF values and the contact maps in the training samples as described in Section 3.3. We followed DeepSDF [6] to sample majority of SDF values on query points close to the gripper and object surface. Our pipeline for data generation can be easily extended with additional sets of objects and grippers.

Table 1: Chamfer distance (Ã1e-3) for shape reconstruction and similarity scores for grasp retrieval
<table><tr><td>YCB Model</td><td colspan="6">Shape Reconstruction</td><td colspan="4">Grasp Retrieval</td></tr><tr><td></td><td>Object</td><td>Fetch</td><td>Barrett</td><td>Human</td><td>Allegro</td><td>Panda</td><td>Near Z</td><td>Near GT</td><td>Far Z</td><td>Far GT</td></tr><tr><td>Cracker Box</td><td>1.33</td><td>1.44</td><td>5.52</td><td>2.86</td><td>3.03</td><td>2.48</td><td>0.81</td><td>0.88</td><td>0.19</td><td>0.15</td></tr><tr><td>Soup Can</td><td>2.60</td><td>4.95</td><td>4.06</td><td>3.21</td><td>3.97</td><td>4.20</td><td>0.73</td><td>0.84</td><td>0.13</td><td>0.09</td></tr><tr><td>Mustard Bottle</td><td>1.05</td><td>2.38</td><td>3.39</td><td>2.05</td><td>2.09</td><td>4.14</td><td>0.80</td><td>0.87</td><td>0.14</td><td>0.10</td></tr><tr><td>Pudding Box</td><td>1.49</td><td>5.27</td><td>8.27</td><td>5.45</td><td>3.80</td><td>5.40</td><td>0.77</td><td>0.84</td><td>0.25</td><td>0.16</td></tr><tr><td>Gelatin Box</td><td>1.20</td><td>5.42</td><td>7.04</td><td>4.72</td><td>3.74</td><td>3.72</td><td>0.75</td><td>0.81</td><td>0.22</td><td>0.14</td></tr><tr><td>Potted Meat Can</td><td>2.13</td><td>3.63</td><td>3.78</td><td>3.06</td><td>4.45</td><td>1.78</td><td>0.76</td><td>0.83</td><td>0.21</td><td>0.12</td></tr><tr><td>Bleach Cleanser</td><td>7.82</td><td>2.45</td><td>5.63</td><td>7.22</td><td>5.41</td><td>14.08</td><td>0.86</td><td>0.91</td><td>0.16</td><td>0.13</td></tr></table>

<!-- image-->  
Figure 5: Grasp Retrieval: Query grasps from test set with close/far grasps from training set

## 5 Experiments

## 5.1 Representation Learning on the Multi-Hand Grasping Dataset

To the best of our knowledge, there is no previous method that encodes grasps from multiple robotic grippers using neural implicit representation and enforces a notion of grasp similarity in a unified fashion. Hence, to validate our representation, we focus on the tasks of (1) shape reconstruction from a given latent code and, (2) grasp retrieval using the latent codes.

Shape Reconstruction. For this task, we consider a set of unseen test grasps with inferred latent codes from Eqn. (4). For each test grasp, query points are randomly sampled and passed as inputs to the network with the specific latent code. Using the predicted SDF values on the set of query points, the object and gripper points are separated out $( \bar { P } _ { o } , \bar { P _ { g } }$ in Section 3.4). We then compute the Chamfer distance against their ground truth point clouds and report the results in Table 1. The results shown in the table indicate a good performance even though reconstruction is not the primary goal of the method and is only included as a validation step for the representation learning. Furthermore, the inferred contact maps for such reconstructions align closely with the ground truth as seen in Fig. 3.

Grasp Retrieval. To test the effectiveness of the metric learning loss on the latent space, we perform an experiment on grasp retrieval for an input query grasp. We use the latent vector of an input query grasp and retrieve the closest and farthest grasps using (a) the distance between the latent vectors as a similarity measure, and (b) the distance between the contact maps which is considered as the ground truth (GT in Table 1). We report the similarity score $s = 1 \bar { \bf \Phi } - d { \bf \Phi } ; d \in [ 0 , 1 ]$ between the query and retrieved grasps for both (a) and (b) in Table 1. d is the normalized $L _ { 1 }$ distance between the contact maps of the pairs of grasps. As seen in Table 1, the learned latent codes have similarity scores aligned with those for the contact maps based method, and this shows the effectiveness of the metric learning constraint. Fig. 5 shows some grasp retrieval examples using the latent codes.

## 5.2 Real-World Experiments on Grasping and Grasp Transfer

Grasping with Object Pose Estimation. To validate our implicit representation in the real world, we first conducted a grasping experiment with the 7 YCB objects in our dataset on a tabletop. A Fetch mobile manipulator is employed for grasping. For each trial, we simply put one object on a table and we use the table height to segment the point cloud of the object from the Fetch head camera. Using the segmented point cloud, we estimate the object pose according to Eqn. (5). For comparison, we also tested a baseline model that only uses the SDF of the object without grasp contact optimization, i.e., no $f _ { \mathrm { S D F } } ^ { g }$ in Eqn. (5). Since Eqn. (5) requires a grasp for optimization, we first use the baseline model to estimate the object pose, and then select the closest grasp from the grasp set to the current gripper location for optimization. Using the estimated pose, we attempt to grasp and lift the object after finding a suitable motion plan using MoveIt. We did 5 trials for each object. The results of this study are shown in Table 2 where we can see the relative improvement in grasp success rate for the method using the contact-based pose estimation. This also shows the importance of the implicit contact point modeling in the framework since contact is a critical component of grasping. Please see the supplementary materials for the grasping videos.

<!-- image-->  
Figure 6: Our pipeline for grasp transfer from human demonstrations to a Fetch mobile manipulator.

Human-to-Robot Grasp Transfer. We consider the task of transferring human grasps to our Fetch robot via the proposed implicit representation. We first conduct object pose estimation as described above. After the pose estimation, a person demonstrates a grasp on the object. Using the RGB-D images from the Fetch camera, we estimate the 3D hand joints using A2J [30] and then utilize Pose2Mesh [31] to reconstruct the 3D hand mesh from the predicted 3D hand joints. We combine the hand points from the 3D hand mesh with the object point cloud and infer a latent code for the demonstrated grasp via Eqn. (4). Using the inferred latent code, we query for the closest Fetch gripper grasp in the encoding space of training data grasps and execute it in the real world. We show a qualitative result of such a grasp transfer from a human demonstration in Fig. 6. More examples can be found in the supplementary materials.

Table 2: Grasp success over 5 trials: baseline vs. grasp contact optimization
<table><tr><td>YCB Model</td><td>Baseline</td><td>With Contact</td></tr><tr><td>Cracker Box</td><td>5</td><td>5</td></tr><tr><td>Soup Can</td><td>2</td><td>4</td></tr><tr><td>Mustard</td><td>2</td><td>4</td></tr><tr><td>Pudding Box</td><td>4</td><td>4</td></tr><tr><td>Gelatin Box</td><td>3</td><td>5</td></tr><tr><td>Potted Meat</td><td>3</td><td>4</td></tr><tr><td>Bleach</td><td>3</td><td>4</td></tr><tr><td>#Success</td><td>22</td><td>30</td></tr><tr><td>Success Rate</td><td>0.628</td><td>0.857</td></tr></table>

## 6 Conclusion and Future Work

We introduce NeuralGrasps, a framework for learning implicit representations for grasps of multiple robotic hands. In our framework, grasps across different robotic hands are embedded into a shared latent space, where the distance metric in the latent space is learned to preserve grasp similarity. Therefore, NeuralGrasps enables grasp transfer between grippers in the latent space. In addition, a latent code can be decoded to the signed distance functions of an object and a gripper in a grasping pose. This property enables us to perform 6D object pose estimation with grasping contact optimization using partial observations in the real world. With the estimated object pose, robot grasping can be achieved using the encoded grasps in NeuralGrasps.

In the future, we plan to address several limitations in NeuralGrasps. First, we train a single network for each object in our method. We will study learning representations of multiple objects and multiple grasps with a unified network. Second, NeuralGrasps cannot synthesize novel grasps in the latent space well. We will extend it to grasp synthesis in the future. Lastly, we will train NeuralGrasps with a large number of objects and make it be able to deal with unseen objects during inference.

## Acknowledgments

This work was supported in part by the DARPA Perceptually-enabled Task Guidance (PTG) Program under contract number HR00112220005. B. Prabhakaranâs work was supported by (while serving at) the National Science Foundation.

## References

[1] A. Mandlekar, Y. Zhu, A. Garg, J. Booher, M. Spero, A. Tung, J. Gao, J. Emmons, A. Gupta, E. Orbay, et al. Roboturk: A crowdsourcing platform for robotic skill learning through imitation. In Conference on Robot Learning, pages 879â893. PMLR, 2018.

[2] L. Wang, Y. Xiang, W. Yang, A. Mousavian, and D. Fox. Goal-auxiliary actor-critic for 6d robotic grasping with point clouds. In Conference on Robot Learning, pages 70â80. PMLR, 2022.

[3] O. M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, et al. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1):3â20, 2020.

[4] H. Dang and P. K. Allen. Semantic grasping: Planning robotic grasps functionally suitable for an object manipulation task. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 1311â1317. IEEE, 2012.

[5] M. Kokic, D. Kragic, and J. Bohg. Learning task-oriented grasping from human activity datasets. IEEE Robotics and Automation Letters, 5(2):3352â3359, 2020.

[6] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 165â174, 2019.

[7] K. Karunratanakul, J. Yang, Y. Zhang, M. J. Black, K. Muandet, and S. Tang. Grasping field: Learning implicit representations for human grasps. In 2020 International Conference on 3D Vision (3DV), pages 333â344. IEEE, 2020.

[8] F. Schroff, D. Kalenichenko, and J. Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815â823, 2015.

[9] L. Van Der Maaten. Accelerating t-sne using tree-based algorithms. The Journal of Machine Learning Research, 15(1):3221â3245, 2014.

[10] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4460â4470, 2019.

[11] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European conference on computer vision, pages 405â421. Springer, 2020.

[12] A. Simeonov, Y. Du, A. Tagliasacchi, J. B. Tenenbaum, A. Rodriguez, P. Agrawal, and V. Sitzmann. Neural descriptor fields: Se (3)-equivariant object representations for manipulation. arXiv preprint arXiv:2112.05124, 2021.

[13] A. T. Miller and P. K. Allen. Graspit! a versatile simulator for robotic grasping. IEEE Robotics & Automation Magazine, 11(4):110â122, 2004.

[14] C. Ferrari and J. F. Canny. Planning optimal grasps. In ICRA, volume 3, page 6, 1992.

[15] C. Borst, M. Fischer, and G. Hirzinger. Grasp planning: How to choose a suitable task wrench space. In IEEE International Conference on Robotics and Automation, 2004. Proceedings. ICRAâ04. 2004, volume 1, pages 319â325. IEEE, 2004.

[16] V.-D. Nguyen. Constructing force-closure grasps. The International Journal of Robotics Research, 7(3):3â16, 1988.

[17] D. Berenson and S. S. Srinivasa. Grasp synthesis in cluttered environments for dexterous hands. In Humanoids 2008-8th IEEE-RAS International Conference on Humanoid Robots, pages 189â196. IEEE, 2008.

[18] C. Goldfeder, M. Ciocarlie, H. Dang, and P. K. Allen. The columbia grasp database. In 2009 IEEE international conference on robotics and automation, pages 1710â1716. IEEE, 2009.

[19] C. Eppner, A. Mousavian, and D. Fox. Acronym: A large-scale grasp dataset based on simulation. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 6222â6227. IEEE, 2021.

[20] J. Bohg, A. Morales, T. Asfour, and D. Kragic. Data-driven grasp synthesisâa survey. IEEE Transactions on robotics, 30(2):289â309, 2013.

[21] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea, and K. Goldberg. Dexnet 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics. arXiv preprint arXiv:1703.09312, 2017.

[22] F.-J. Chu, R. Xu, and P. A. Vela. Real-world multiobject, multigrasp detection. IEEE Robotics and Automation Letters, 3(4):3355â3362, 2018.

[23] A. Mousavian, C. Eppner, and D. Fox. 6-dof graspnet: Variational grasp generation for object manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2901â2910, 2019.

[24] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox. Contact-graspnet: Efficient 6-dof grasp generation in cluttered scenes. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 13438â13444. IEEE, 2021.

[25] L. Shao, F. Ferreira, M. Jorda, V. Nambiar, J. Luo, E. Solowjow, J. A. Ojea, O. Khatib, and J. Bohg. Unigrasp: Learning a unified model to grasp with multifingered robotic hands. IEEE Robotics and Automation Letters, 5(2):2286â2293, 2020.

[26] S. Brahmbhatt, A. Handa, J. Hays, and D. Fox. Contactgrasp: Functional multi-finger grasp synthesis from contact. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2386â2393. IEEE, 2019.

[27] Y. Hasson, G. Varol, D. Tzionas, I. Kalevatykh, M. J. Black, I. Laptev, and C. Schmid. Learning joint reconstruction of hands and manipulated objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11807â11816, 2019.

[28] B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, and A. M. Dollar. The ycb object and model set: Towards common benchmarks for manipulation research. In 2015 international conference on advanced robotics (ICAR), pages 510â517. IEEE, 2015.

[29] E. Coumans and Y. Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning. 2016.

[30] F. Xiong, B. Zhang, Y. Xiao, Z. Cao, T. Yu, J. T. Zhou, and J. Yuan. A2j: Anchor-to-joint regression network for 3d articulated pose estimation from a single depth image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 793â802, 2019.

[31] H. Choi, G. Moon, and K. M. Lee. Pose2mesh: Graph convolutional network for 3d human pose and mesh recovery from a 2d human pose. In European Conference on Computer Vision, pages 769â787. Springer, 2020.

[32] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448â456. PMLR, 2015.

[33] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15 (1):1929â1958, 2014.

[34] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[35] S. Chitta, I. Sucan, and S. Cousins. Moveit![ros topics]. IEEE Robotics & Automation Magazine, 19(1):18â19, 2012.

[36] Z. Tian, C. Shen, H. Chen, and T. He. Fcos: Fully convolutional one-stage object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9627â9636, 2019.

[37] D. Shan, J. Geng, M. Shu, and D. F. Fouhey. Understanding human hands in contact at internet scale. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9869â9878, 2020.

[38] Y.-W. Chao, W. Yang, Y. Xiang, P. Molchanov, A. Handa, J. Tremblay, Y. S. Narang, K. Van Wyk, U. Iqbal, S. Birchfield, et al. Dexycb: A benchmark for capturing hand grasping of objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9044â9053, 2021.

[39] C. Zimmermann, D. Ceylan, J. Yang, B. Russell, M. Argus, and T. Brox. Freihand: A dataset for markerless capture of hand pose and shape from single rgb images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 813â822, 2019.

## Appendix

## A Details about the Multi-Hand Grasping Dataset

We use the Graspit! [13] simulation software to generate a set of initial grasps for each gripperobject pair in a similar fashion as the ObMan dataset [27]. Graspit assumes setting contact points on the gripper links which we manually set for grippers not present by default in Graspit (Franka Panda and Allegro grippers). We also ensure to not select a proposed grasp if there is no contact with the object. On the initial set of grasps generated by Graspit, we use farthest point sampling to obtain a final, fixed set of diverse grasps. In our experiments, the multi-hand grasping dataset has 60 grasps for each gripper-object pair i.e across the five grippers and the seven objects. The relevant grasp information (griper pose and configuration for the joints) generated by Graspit were stored as a json file which was later utilized in the PyBullet rendering. Once the grasps were generated, the object and gripper models were loaded in PyBullet environment with the pose and the joint configuration corresponding to the grasp applied on the gripper. We used 128 virtual Pybullet camera viewpoints around the object to ensure complete scene information and render a dense point cloud for each grasping scene.

Upon generation of the point clouds, the SDF values for the gripper and the object were generated using the scheme outlined in [6] with the caveat that we computed the SDFs over surface point clouds instead of 3D meshes as we had access to the surface points for the gripper and object via the dense point cloud rendering. For each object, we ensured that every grasping scene is scaled to a unit sphere using a constant scaling factor across all grippers before the computing the SDF training data to ensure that the relative differences in gripper sizes stay the same. Given a desired number of points for the SDF generation (we used 40,000 points), there was an equal division between points having (at least one of) positive and negative SDF values to either the gripper or the object. As seen in other related implicit representation works, a majority (95%) of the sampled points were close to the gripper and object surface to encourage learning of SDF near their surfaces. The SDF values were also utilized in the contact map generation to find the object contact points that were within 5cm of a gripper surface point.

<!-- image-->  
Figure 7: Network architecture used in our experiments. The 12 MLP blocks are shown along with the final layer tanh activation on the 2 output SDF values.

## B Network Architecture and Training Details

The auto-decoder architecture shown in Fig. 7 is used in all of our experiments. The decoder component consists of 12 fully connected layers with each having 512 dimensions, and finally the two SDF values (for gripper and object) as the output. We use Batch Normalization [32] and Dropout [33] after each layer with a dropout probability of 0.2. The query point and latent code for a specific grasping scene are concatenated and passed as input to the auto-decoder. The latent code size (d) for all experiments was 128 and the latent code distribution had a zero-mean Gaussian prior $\mathcal { N } ( 0 , \sigma ^ { 2 } \mathcal { T } _ { d } )$ with $\sigma = 0 . 0 1$ . We used the Adam [34] optimizer to the train the network parameters and the latent codes for a total of 2,000 epochs with a step learning rate schedule for the optimizer. From the 60 grasps for every object-gripper pair in the dataset, we choose 50 of them across each object to construct the training set for the neural network model training. For the loss function on the SDF values, we have equal weight between the individual components on the gripper, object SDFs and the triplet loss. Each component was given a weight of 1.0. The SDF clamping distance Î´ is set to be 0.05 and for each grasping scene, we sample 20,000 points from the corresponding SDF samples in the dataset.

## C Experiment Details

## C.1 Shape Reconstruction and Grasp Retrieval

For the experiments on unseen test grasps mentioned in the paper, we pick a set of held-out grasps not used during the model training. For an object, we select 5 grasping scenes from every gripper as the test set. The latent code for each test grasp is inferred using the SDF samples prepared during dataset construction and doing the MAP optimization for 800 iterations. Once the latent code is inferred, we try to predict the SDF values (query it via the model) on a randomly sampled set of query points inside the unit sphere. For all our experiments, we sample 100,000 points inside the unit sphere. Using the predicted SDF values on the query points, we obtain the estimated point clouds for the object and the gripper as the query points having the SDFs less than zero for object and gripper respectively. Chamfer distance is then computed between the inferred and ground truth point clouds for the object and gripper in the specific test grasping scene.

In the grasp retrieval experiments, for an input query grasp, the $L _ { 1 }$ distance matrices for 1) the inferred latent code of the query grasp and training grasp latent codes and 2) the ground truth contact map of the query grasp and training set contact maps, each has the distances normalized to be between 0 and 1 so that the similarity score (computed as 1-distance) also remains between 0 and 1. We opted to use the simple 1-distance measure since any monotonically decreasing function of the distance works as a notion of similarity. Once we have the distances between the query and training set grasps, we simply sort them to obtain the closest and farthest examples.

We present the Chamfer distances and grasp retrieval similarity scores on the grasps in the training set in Table 3. The training set experiments were used as a sanity check on the model for representing grasps it had already seen. Although reconstruction is not the primary goal, we see low values for Chamfer distance and a high degree of match between the ground truth contact map-based and latent code-based grasp retrieval results. The Barnes-Hut t-SNE visualizations [9] of the learned embeddings of the different grasps from five robotic hands across the different objects are shown in Figs. 8, 9, 10.

<!-- image-->  
Figure 8: t-SNE visualization of the grasp embeddings on tomato soup can (left) and mustard bottle (right).

<!-- image-->  
Figure 9: t-SNE visualization of the grasp embeddings on pudding box (left)and gelatin box (right).

<!-- image-->  
Figure 10: t-SNE visualization of the grasp embeddings on potted meat can.

Table 3: Chamfer distance (x1e-3) for shape reconstruction and contact map similarity for grasp retrieval on training set.
<table><tr><td>Ycb Model</td><td colspan="6">Shape Reconstruction</td><td colspan="4">Grasp Retrieval</td></tr><tr><td rowspan="2">Cracker Box</td><td>Object</td><td>Fetch</td><td>Barrett</td><td>Human</td><td>Allegro</td><td>Panda</td><td>Near Z</td><td>Near GT</td><td>Far Z</td><td>Far GT</td></tr><tr><td>1.28</td><td>4.06</td><td>5.89</td><td>6.36</td><td>4.11</td><td>7.84</td><td>0.87</td><td>0.88</td><td>0.26</td><td>0.24</td></tr><tr><td>Soup Can</td><td>1.29</td><td>3.77</td><td>4.84</td><td>2.91</td><td>3.30</td><td>3.12</td><td>0.78</td><td>0.79</td><td>0.12</td><td>0.10</td></tr><tr><td>Mustard Bottle</td><td>0.99</td><td>1.91</td><td>3.22</td><td>2.14</td><td>1.98</td><td>2.48</td><td>0.83</td><td>0.84</td><td>0.19</td><td>0.16</td></tr><tr><td>Pudding Box</td><td>1.22</td><td>5.14</td><td>8.16</td><td>6.55</td><td>5.25</td><td>6.11</td><td>0.83</td><td>0.84</td><td>0.24</td><td>0.23</td></tr><tr><td>Gelan Box</td><td>1.12</td><td>5.53</td><td>7.73</td><td>4.66</td><td>4.99</td><td>4.73</td><td>0.80</td><td>0.81</td><td>0.23</td><td>0.21</td></tr><tr><td>Potted Meat Can</td><td>1.96</td><td>3.90</td><td>5.23</td><td>3.78</td><td>3.77</td><td>3.05</td><td>0.79</td><td>0.80</td><td>0.14</td><td>0.10</td></tr><tr><td>Bleach Cleanser</td><td>7.09</td><td>4.87</td><td>6.12</td><td>7.52</td><td>5.50</td><td>8.85</td><td>0.72</td><td>0.73</td><td>0.18</td><td>0.17</td></tr></table>

## C.2 Robotic Grasping

For the object pose estimation and grasping experiments, we assume an uncluttered tabletop scene with a single object placed on the table. Since the goal of the experiments is not to showcase detection or segmentation on object point clouds, we adopt a simple heuristic of using the table height to segment out the real world point cloud of the target object. We only use a single-view (partial) point cloud of the object in the real-world experiments.

Baseline Model for 6D Object Pose Estimation. The pose optimization assumes that the given points lie on the object surface and hence the predicted SDF values for them should be close to zero (first term of Equation 7). We solve the following optimization problem to estimate the 3D rotation R and the 3D translation t of the object pose:

$$
R ^ { * } , \mathbf { t } ^ { * } = \arg \operatorname* { m i n } _ { R , \mathbf { t } } \Big [ \sum _ { j = 1 } ^ { M } | f _ { \mathrm { S D F } } ^ { o } ( R \mathbf { x } _ { c } ^ { j } + \mathbf { t } , \mathbf { z } ; \theta _ { o } ) | \Big ] ,\tag{6}
$$

where $\{ \mathbf { x } _ { c } ^ { j } \} _ { j = 1 } ^ { M }$ 1 denotes the points in the camera frame. z is the latent code of an arbitrary grasp in the training set since we do not use the SDF of the gripper in this optimization. We name this model the baseline model for object pose estimation since it does not utilize the grasp contact. To start the optimization with a good initial pose, we initialize the 3D translation with the center of the point cloud. For 3D rotation, we discretize the SO(3) space by yaw, pitch and roll angles with 45-degree intervals, then evaluate and randomly select a 3D rotation within the top-5 minimum objective values according to Eq. (6) as the initial rotation. After the optimization, we accept the estimated pose if the objective value is smaller than a pre-defined threshold. Otherwise, we re-run the optimization until a good pose is obtained. These steps are useful to avoid local minimums in SDF-based optimization for object pose estimation.

6D Object Pose Estimation with Contact. Letâs denote the estimated object pose using the baseline model as $( R _ { 0 } , \mathbf { t } _ { 0 } )$ . Using this initial pose estimation, we find a Fetch gripper grasp from the training set closest to the current gripper position for optimization. Because all the grasps are defined in the coordinate frame of the object. We need to use the object pose to transform the grasps to the camera frame, and use the pose between camera and robot base to transform them into the robot base frame. After that, we can compare them with the current gripper pose in the robot base frame. Assume the select grasp has a latent code $\mathbf { z } ^ { \ast }$ and contact map Ï. We can first find a set of contact points on the object surface $\{ \mathbf { x } _ { o } ^ { k } \} _ { k = 1 } ^ { L }$ using the contact map Ï. Note that these points are defined in the object coordinate frame. Using the initial pose estimation $( R _ { 0 } , \mathbf { t } _ { 0 } )$ , we can transform these points to the camera frame and then find the nearest neighbors of the transformed points from the object point cloud in the camera frame. Letâs denote these nearest neighbor points in the camera frame as $\{ \mathbf { x } _ { g } ^ { k } \} _ { k = 1 } ^ { L }$ We consider these points to be the contact points in the camera frame for the grasp encoded by $\mathbf { z } ^ { \ast }$ . Therefore, we can perform the following optimization to refine the initial pose:

$$
R ^ { * } , \mathbf { t } ^ { * } = \arg \operatorname* { m i n } _ { R , \mathbf { t } } \Big [ \sum _ { j = 1 } ^ { M } \big | f _ { \mathrm { S D F } } ^ { o } ( R \mathbf { x } _ { c } ^ { j } + \mathbf { t } , \mathbf { z } ^ { * } ; \theta _ { o } ) \big | + \sum _ { k = 1 } ^ { L } \big | f _ { \mathrm { S D F } } ^ { g } ( R \mathbf { x } _ { g } ^ { k } + \mathbf { t } , \mathbf { z } ^ { * } ; \theta _ { o } ) \big | \Big ] ,\tag{7}
$$

This is our algorithm for 6D object pose estimation with contact using our implicit representations.

Grasp Execution. In order to execute the chosen grasp with optimized object pose, we first compute a standoff position and then sample 10 way points from the the standoff to the final grasp position for smooth execution. In the videos, it can be seen that sometimes the motion from standoff to final pose is very fast resulting in some momentum transferred to the object. We add the table on which the object is placed as an obstacle and then utilize the Moveit [35] package to do the motion planning from an initial stowed position for the Fetch gripper. Lastly, on arriving in the final position, the Fetch gripper attempts to grasp the object by closing its fingers and then lifting the object from the table. The pose estimation method is not fail-proof as seen from the results of Table 2 in the paper and attached video containing the grasping clips. A few failures are a consequence of the pose estimation getting stuck in a local minima. In some cases, the pose estimation for the âwith contactâ method actually performed worse due to an incorrect closest point matching, and likely resulting from a bad initial pose estimate.

## C.3 Human Demonstrations and Grasp Transfer

We utilize existing methods for hand pose estimation in our human-to-robot grasp transfer experiments. The same RGB-D robot head camera is used to observe the human grasp demonstrations and infer the hand pose and 3D mesh for it on a per frame basis. Following a demo of the desired grasp by a human hand, we estimate the human hand grasps with the following pipeline. Given a RGB-D video of the human grasp demo, the Region of Interest (RoI) around the hand is detected using the Fully Convolutional One-Stage Object Detector (FCOS [36]) trained on 100 Days of Hands [37] dataset. Next, the RoI of the hand is cropped from the depth image, and the 3D hand joints are estimated using the Anchor to Joint (A2J [30]) model trained on the DexYCB [38] dataset. Lastly, we use the Pose2Mesh [31] model trained on FreiHAND [39] dataset to recover a 3D mesh of the hand from the predicted 3D hand joints. The 3D mesh consequently also gives the hand surface point cloud required for the latent code inference (here the hand is considered as a gripper). The sup-

<!-- image-->  
Figure 11: Closest Fetch gripper grasps for the human demonstrations on the gelatin box (009). The grasps in the dataset generated from Graspit are not similar to a pincer like grasp.

plementary video also shows how the grasp transfer process looks like with demos for each object. Once we have the closest Fetch gripper grasps, the Moveit [35] motion planning package is used as before to find a motion plan to the closest grasp. We note that the closest retrieved Fetch grasp might not have a feasible motion plan associated with it and hence we iterated along the ordered list of closest grasps and execute the one for which a successful motion plan is found. This set of executed grasps are shown in the video and hence for a few cases, the execution slightly differs from the demonstration.

The grasp transfer for the gelatin box (009) demo was not successful due to the motion planning finding no feasible plan to the inferred closest Fetch gripperâs grasp. This is likely due to the fact that the inferred closest Fetch grippers grasps have the gripper to grasp the box using the top and bottom rather than the corner as see in Fig. 11. Additional qualitative results on closest inferred grasps across the entire gripper set are shown in Fig. 12, 13, and 14. It is possible that the transfer process is sometimes not feasible for such scenarios where the mapping to the Fetch gripper is not perfect. It can be a consequence of the dataset not containing a specific type of Fetch gripper grasp and the existing grasps being very different from the human demo even though the contact regions may be close on a small object relative to the grippers like the gelatin box. Another avenue to

Hand Demonstration & Inferred Combined Point Cloud

Allegro

Barrett

Fetch

Panda

Human Hand

<!-- image-->  
Figure 12: Closest grasps for all grippers for the human demonstrations on the cracker box (top) and tomato soup can (bottom)

consider in the grasp transfer process is utilizing reliable object pose estimates of real world scans. For example in Fig. 12, for the first row of cracker box demonstration, the closest Panda gripper grasp is diagonally opposite to others. This result is due to the fact that the latent code inference does not take texture information into account and hence we have a flipped result.

<!-- image-->

Allegro

Barrett

Fetch

Panda

Human Hand

<!-- image-->  
Figure 13: Closest grasps for all grippers for the human demonstrations on the mustard bottle (top), pudding box (middle), and gelatin box (bottom).

Hand Demonstration & Inferred Combined Point Cloud

<!-- image-->

<!-- image-->

<!-- image-->

Allegro

Barrett

Fetch

Panda

<!-- image-->

Human Hand

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

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Hand Demonstration & Inferred Combined Point Cloud

<!-- image-->

<!-- image-->  
Barrett

<!-- image-->  
Fetch

<!-- image-->  
Panda

<!-- image-->  
Human Hand

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

Figure 14: Closest grasps for all grippers for the human demonstrations on the potted meat can (top) and bleach cleanser (bottom).