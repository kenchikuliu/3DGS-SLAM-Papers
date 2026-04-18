# GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics

Qianzhong Chenâ1, Naixiang Gaoâ1, Suning Huang2, JunEn Low1, Timothy Chen2, Jiankai Sun2, and Mac Schwager2

Abstractâ Autonomous drones capable of interpreting and executing high-level language instructions in unstructured environments remain a long-standing goal. Yet existing approaches are constrained by their dependence on hand-crafted skills, extensive parameter tuning, or computationally intensive models unsuitable for onboard use. We introduce GRaD-Nav++, a lightweight VisionâLanguageâAction (VLA) framework that runs fully onboard and follows natural-language commands in real time. Our policy is trained in a photorealistic 3D Gaussian Splatting (3DGS) simulator via Differentiable Reinforcement Learning (DiffRL), enabling efficient learning of low-level control from visual and linguistic inputs. At its core is a Mixture-of-Experts (MoE) action head, which adaptively routes computation to improve generalization while mitigating forgetting. In multi-task generalization experiments, GRaD-Nav++ achieves a success rate of 83% on trained tasks and 75% on unseen tasks in simulation. When deployed on real hardware, it attains 67% success on trained tasks and 50% on unseen ones. In multi-environment adaptation experiments, GRaD-Nav++ achieves an average success rate of 81% across diverse simulated environments and 67% across varied real-world settings. These results establish a new benchmark for fully onboard Vision-Language-Action (VLA) flight and demonstrate that compact, efficient models can enable reliable, languageguided navigation without relying on external infrastructure.

## I. INTRODUCTION

In recent years, autonomous drones have made remarkable progress in navigating complex environments, driven by advances in modular âperception-planning-controlâ pipelines [1], [2], imitation learning (IL) [3], and reinforcement learning (RL) [4]â[6]. These approaches have enabled drones to perform a broad range of tasks, ranging from basic waypoint following to sophisticated multi-agent coordination in dynamic environments. Despite these successes, existing methods are often restricted to executing narrowly defined tasks within highly structured settings. They typically rely on extensive manual tuning, task-specific reward shaping, or large-scale expert demonstrations, which limit their flexibility and scalability. As a result, there remains a significant gap between current autonomous drone systems and truly intelligent agents that can understand abstract human intentions and complete complex tasks specified through natural language instructions.

<!-- image-->  
Fig. 1: Our GRaD-Nav++ architecture.

The emergence of large language models (LLMs) [7], vision-language models (VLMs) [8], [9], has opened new possibilities for bridging the gap between natural language instructions and autonomous drone control. By leveraging the semantic understanding and reasoning capabilities of these large models [10], drones are now able to interpret high-level human commands and adapt their behaviors accordingly, enabling more flexible and intuitive human-robot interaction.

Several recent works have explored the use of LLMs to direct drones in performing complex tasks and engaging in interactive behaviors [11], [12]. However, these approaches typically adopt a layered architecture, where the LLM selects from a predefined set of task-specific skills or interacts with the system through API calls. Thus, the resulting gap between high-level decision-making and low-level control remains a critical bottleneck. Moreover, due to their size and computational demands, LLMs cannot be deployed onboard, requiring ground station support and reliable communication linksâconditions that are often impractical in real-world deployments.

Vision-Language-Action (VLA) models [13]â[15] offer a promising solution by enabling end-to-end policies that directly map natural language instructions and visual inputs to low-level actions. This unified architecture allows better use of pretrained model priors while avoiding the reliance on predefined skills or external APIs. Recent efforts have applied VLA models to drone flight tasks. RaceVLA [16] achieves impressive performance in competitive drone racing scenarios, but its applicability is limited to highly constrained domains and lacks generalization to more diverse navigation tasks. CognitiveDrone [17] demonstrates embodied reasoning capabilities and instruction-grounded control, yet it is trained and evaluated entirely in simulation, with no evidence of sim-to-real transfer. In addition, both methods still rely on large-scale models with billions of parameters, making them unsuitable for real-time onboard deployment on computationally limited aerial platforms.

A major barrier in training such models is the need for both high-fidelity visual environments and sample-efficient policy optimization. To address this, we leverage two recent advances: (i) 3D Gaussian Splatting (3DGS) [18] enables high-quality photorealistic rendering at interactive rates, providing rich visual input without the need for costly meshbased simulation or NeRF rendering; (ii) Differentiable Deep Reinforcement Learning (DiffRL) [19], a framework that uses differentiable physics simulation [20], [21] to accelerate policy learning via gradient-based optimization. Built on this foundation, we further integrate a lightweight Mixtureof-Experts (MoE) action head, which improves generalization by routing computation across specialized sub-policies. Together, these components enable fast, high-fidelity, and sample-efficient training of VLA models, paving the way for real-time, onboard deployment of instruction-following drones. The key contributions of our approach are as follows:

â¢ We propose a novel light-weighted drone flight VLA framework that operates entirely on the droneâs onboard computing hardware.

â¢ Our VLA policy enables the drone to accomplish complex tasks based on high-level natural language instructions, demonstrating both generalization to previously unseen tasks and adaptability across diverse environmental conditions.

â¢ We develop a novel multi-headed MoE action module for our VLA policy, trained using 3DGS and DiffRL. Our approach achieves state-of-the-art performance in terms of sample efficiency and task success rate.

## II. BACKGROUND

## A. Partially Observable Markov Decision Process (POMDP)

A POMDP is represented by the tuple $( S , O , A , P , r , \gamma )$ In this formulation, S denotes the true state space, O corresponds to the visual observation space, and A is the action space of the robot. The transition dynamics are defined by $P : S \times A \to S$ , and the reward function is given by $r : S \times A  \mathbb { R }$ . The discount factor $\gamma \in ( 0 , 1 ]$ determines the relative importance of future rewards. The objective is to learn an optimal policy $\pi _ { \boldsymbol { \theta } } ( a _ { t } \ | \ o _ { t } )$ that maximizes the expected cumulative reward $\begin{array} { r } { \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \right] } \end{array}$

## B. GRaD-Navâs Joint Simulation Pipeline with 3DGS and Differentiable Drone Dynamics

1) 3D Gaussian Splatting: 3D Gaussian Splatting (3DGS) represents a scene using a set of anisotropic Gaussian prim-

itives, each defined by position $\mu _ { i } ,$ , covariance $\Sigma _ { i } .$ , color $\mathbf { } c _ { i } .$ and opacity $\alpha _ { i }$ [18]. These Gaussians are projected onto a 2D plane to compute pixel colors:

$$
\mathbf { C } ( \mathbf { p } ) = \sum _ { i = 1 } ^ { N } \mathcal { N } ( \mathbf { p } ; \mu _ { i } , \Sigma _ { i } ) \cdot T _ { i } \cdot \pmb { c } _ { i } ,\tag{1}
$$

where $\begin{array} { r } { T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } \end{array}$ accounts for transmittance and occlusion.

2) Differentiable Quadrotor Dynamics: We derived from GRaD-NAvâs [4] PyTorch based differentiable quadrotor simulator to support gradient-based learning. The control input $\pmb { u } _ { t } = ( \omega _ { t } ^ { d } , c _ { t } )$ commands body rates and normalized thrust. The angular acceleration is:

$$
\begin{array} { r } { \dot { \boldsymbol { \omega } } = \boldsymbol { I } ^ { - 1 } [ K _ { p } ( \boldsymbol { \omega } ^ { d } - \boldsymbol { \omega } ) - K _ { d } \dot { \boldsymbol { \omega } } - \boldsymbol { \omega } \times ( \boldsymbol { I } \boldsymbol { \omega } ) ] , } \end{array}\tag{2}
$$

and orientation $\pmb q \in \mathbb S ^ { 3 }$ evolves as:

$$
\pmb { q } _ { t + 1 } = \mathrm { n o r m } ( \pmb { q } _ { t } + \frac { \Delta t } { 2 } \pmb { q } _ { t } \otimes [ 0  ] ^ { \top } ) .\tag{3}
$$

Linear acceleration is:

$$
\pmb { a } = \frac { 1 } { m } \pmb { R } ( \pmb { q } ) [ 0 \mathrm { ~ } 0 \mathrm { ~ } T ] ^ { \top } + \pmb { g } , \quad T = c T _ { \operatorname* { m a x } } .\tag{4}
$$

3) Hybrid Simulation Pipeline: At each simulation step, the drone pose ${ \pmb T } = ( { \pmb p } , { \pmb q } )$ is fed into the 3DGS renderer to generate first-person RGB observations. A precomputed point cloud also supports reward shaping, reference planning, and collision detection during DiffRL training.

## C. Mixture-of-Experts (MoE)

The Mixture-of-Experts (MoE) architecture, initially proposed by Jacobs et al. and Jordan & Jacobs, is a versatile model that employs a set of specialized expert networks, each focusing on different aspects of the input data [22] [23]. A central gating network determines which experts are activated for each input, dynamically selecting the most suitable ones. This design has been further refined by Shazeer et al., introducing a sparse MoE approach where only a subset of experts is utilized for each input [24]. Our MoE implementation is derived from MENTOR [25]:

$$
F ^ { \mathrm { M o E } } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { N } w _ { i } ( \mathbf { x } ) \cdot \mathrm { F F N } _ { i } ( \mathbf { x } ) ,\tag{5}
$$

$$
w _ { i } ( \mathbf { x } ) = \frac { \exp ( h _ { i } ( \mathbf { x } ) ) \cdot \mathbb { I } [ i \in \mathcal { K } ( \mathbf { x } ) ] } { \sum _ { j \in \mathcal { K } ( \mathbf { x } ) } \exp ( h _ { j } ( \mathbf { x } ) ) } ,\tag{6}
$$

where N is the total number of expert networks. $\mathrm { F F N } _ { i }$ denotes the i-th expert, which is typically implemented as a feedforward neural network. The gating weight $w _ { i } ( \mathbf { x } )$ determines the contribution of the i-th expert to the final output for input x, and is computed by applying a softmax function over the scores $h _ { i } ( \mathbf { x } )$ produced by a routing network. Only the top-k experts (indexed by $\kappa ( \mathbf { x } ) )$ are selected for each input x, and the remaining experts receive zero weight. The indicator function $\mathbb { I } [ i \in { \cal K } ( { \bf x } ) ]$ ensures sparsity by masking out non-selected experts.

## III. METHOD

We present a novel VLA-styled drone autonomous flight framework that can follow natural language instructions and conduct safe navigation in different environments.

## A. Task Definition

We posit that enabling VLA-based drone flight requires two key capabilities:

i. Multi-task generalization: We train our VLA policy on a large set of two-stage tasks within a single environment to promote generalization and zero-shot transfer to untrained tasks. Each task involves (1) selecting a correct direction from {through, left, right, above} to pass a gate, and (2) identifying and flying to a target object among {ladder, cart, monitor}. For example, the instruction âGO THROUGH gate then FLY to LADDER baseâ requires sequential spatial reasoning and object grounding. We construct 12 such tasks, training on 8 with guidance from a reference trajectory, and reserving 4 for zero-shot evaluation. All directions and objects are evenly represented in training.

ii. Multi-environment adaptation: We also train the policy on a smaller task set across two distinct 3DGS environments, each with different gate placements and distractors. These single-stage tasks require directional decisions (e.g., âFLY past the LEFT side of the gateâ) based on visionlanguage input. Importantly, we do not signal environment changesârobust adaptation emerges from the VLM-based visual grounding.

Table VII summarizes the task instructions. For each trained task, we generate a reference trajectory using $n =$ 4 key waypoints defined on the 3DGS point cloud and connected via $A ^ { * }$ planning. Further details are provided in Section III-C.

## B. Model Architecture

An overview of our model architecture is demonstrated in Figure 1.

1) Vision-Language Model (VLM): We use a pretrained Contrastive Language-Image Pretraining (CLIP) model [8] for high-level scene understanding and vision-instruction matching. CLIP is a multi-headed pretrained VLM developed by OpenAI, which can be used for zero-shot image-text matching. We encode both natural language instruction and droneâs first person perspective RGB image using CLIPâs text and visual encoders, respectively. The CLIP model is frozen during training, but we fine-tune a liner layer that fuses and downsamples visual and textual embeddings that generated by CLIP encoders to a common latent space. The final VLM feature vector $\mathbf { e } _ { t } \in \mathbb { R } ^ { 5 1 2 }$ is a 512-dimensional vector, which is used as the input to the policy network. Due to the limitation of dorneâs onboard computer and to ensure runtime model frequency, we run the CLIP in an asynchronous manner, i.e. the CLIP model is run in parallel with the policy network, and the VLM feature vector $\mathbf { e } _ { t }$ is updated every 10 time steps. In summary, we leverage VLM for high-level scene understanding and instruction matching, which offers the down stream policy network an informative and semantically rich representation of the environment, helping the policy network in decision making and controlling.

2) Policy Network: We employ a MoE policy network comprising two expert subnetworks. At each time step, the router activates the top-k experts with k=2, ensuring that all experts are utilized while maintaining sparse computation. The number of experts is intentionally kept small to enable efficient inference and real-time deployment on resourceconstrained onboard hardware. Each expert $\pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t + 1 } \big | \mathbf { o } _ { t } , \mathbf { e } _ { t } \big )$ is a multi-headed network parameterized by Î¸, which takes the VLM feature vector $\mathbf { e } _ { t }$ and the observation ${ \mathbf o } _ { t } \in \mathbb { R } ^ { 7 2 }$ as input. Each head of the single expert is a 3-layer MLP and each layer has 512, 256, 128 neurons, respectively. The processed VLM feature and observation are fused and generate action $\mathbf { a } _ { t + 1 }$ using another 3-layer MLP, each layer has 256, 128, 64 neurons, respectively. The observation is defined as following:

$$
{ \bf o } _ { t } = \left[ { d _ { t } \mathrm { ~  ~ \nabla ~ } z _ { t } \mathrm { ~ \bf ~ r } _ { t } } \right] ^ { T } ,\tag{7}
$$

where $\ b { d } _ { t } \in \mathbb { R } ^ { 3 2 }$ is the minimum-pooled depth image that used for collision avoidance. $z _ { t } \in \mathbb { R } ^ { 2 4 }$ is the output of the context estimator network (will be introduced in Section III-B.4), which is used for mitigating sim-to-real gap and improving policy robustness. $\mathbf { r } _ { t }$ is the raw dynamics observation from drone differentiable simulator and can be defined as:

$$
{ \bf r } _ { t } = \left[ { \pmb h } _ { t } \quad { \pmb q } _ { t } \quad { \pmb v } _ { t } \quad { \pmb a } _ { t } \quad { \pmb a } _ { t - 1 } \right] ^ { T } ,\tag{8}
$$

where $\boldsymbol { h } _ { t } , ~ \boldsymbol { q } _ { t } , ~ \boldsymbol { v } _ { t }$ are drone bodyâs height, quaternion, and linear velocity, respectively. $\mathbf { } \mathbf { a } _ { t }$ and $\mathbf { \delta } _ { a _ { t - 1 } }$ are current action and previous action. We do not need to explicitly estimate droneâs $_ { \mathrm { X ^ { - } y } }$ positions. Aligned with our simulation setting Section II-B.2, the action $\pmb { a } _ { t } = ( \omega _ { t } ^ { d } , c _ { t } )$ is a 4-dimensional vector, including body rate $\omega _ { t } ^ { d } \in \mathbb { R } ^ { 3 }$ and normalized thrust $c _ { t } \in \mathbb { R }$ . It is important to note that we do not implement a standalone convolution neural network (CNN) for RGBD visual perception. Instead, we directly utilize the VLM feature vector $\mathbf { e } _ { t }$ as a high-level representation of visual semantics. This design choice is primarily motivated by the need to optimize inference speed during real-world deployment on robotâs onboard computer. Additionally, the depth feature vector $\mathbf { } d _ { t }$ is derived via a simple minimum pooling operation and is used exclusively for low-level collision avoidance, rather than for high-level perceptual reasoning.

3) Value Network: The value network $V _ { \phi } ( \mathbf { s } _ { t } , \mathbf { e } _ { t } )$ shares the same multi-headed structure as policy network ÏÎ¸, parameterized by Ï and is used to estimate the state value. Except for the observation $\mathbf { o } _ { t }$ , the privileged observation $\mathbf { s } _ { t } \in \mathbb { R } ^ { 7 4 }$ used by the value network also has access to bodyâs x-y positions $\pmb { p } _ { t } \in \mathbb { R } ^ { 2 }$ defined as:

$$
\mathbf { s } _ { t } = \left[ \mathbf { o } _ { t } \quad p _ { t } \right] ^ { T } .\tag{9}
$$

The value function is not needed at runtime, hence the access to privileged information from the simulator is not a practical limitation.

4) Context Estimator Network: Similar to [4], we incorporated a Î²-variational autoencoder (Î²-VAE) [26] based CENet [27] with visual perception data. CENet is designed for encoding the droneâs surrounding environment, especially the spatial relationship with obstacles, to a latent vector $\mathbf { z } _ { t }$ to enable runtime adaptivity to the environment. We used a history observation of the last 5 time steps as CENetâs input.

## C. Training Procedure

Similar to oringinal GRad-Nav [4], our DiffRL training procedure centers on a reward function designed for smooth, stable dynamic control and efficient, safe navigation, as detailed in Table I. This function, $\begin{array} { r } { r _ { t } ( \mathbf { s } _ { t } , \mathbf { a } _ { t } ) = \sum r _ { i } w _ { i } } \end{array}$ , is intentionally kept simple to enhance transferability across different environments and agents. For guiding the drone along desired trajectories, as described in Table I, we incorporate several reward terms. A waypoint reward $r _ { \mathrm { w p } } =$ $\left( e ^ { - \| \pmb { p } - \pmb { w } _ { \mathrm { n e x t } } \| ^ { 2 } } \right)$ encourages the drone (at position p) to approach the next waypoint ${ \pmb w } _ { \mathrm { n e x t } }$ (i.e., the closest waypoint with $x _ { \mathrm { w p } } > x _ { \mathrm { d r o n e } } )$ on a precomputed reference trajectory. To further ensure adherence to this path, a reference trajectory tracking reward $\begin{array} { r } { r _ { \mathrm { t r a j } } = \left\| \frac { \boldsymbol { v } } { \| \boldsymbol { v } \| } - \frac { { \boldsymbol { v } } _ { \mathrm { d e s } } } { \| { \boldsymbol { v } } _ { \mathrm { d e s } } \| } \right\| } \end{array}$ penalizes deviations between the droneâs normalized velocity $\frac { \pmb { v } } { \Vert \pmb { v } \Vert }$ and the desired velocity direction $\frac { \boldsymbol { v } _ { \mathrm { d e s } } } { \lVert \boldsymbol { v } _ { \mathrm { d e s } } \rVert }$ from the reference trajectory. Safe navigation is promoted by an obstacle avoidance reward $r _ { \mathrm { o b s } } ~ = ~ d _ { \mathrm { o b s } }$ when the minimum distance to the nearest obstacle $d _ { \mathrm { o b s } }$ within the droneâs field of view is less than a predefined threshold $d _ { \mathrm { t h } }$ (set to 0.5m in our method). The specific weighting factors $\omega _ { i }$ for these components are listed in Table I.

To ensure balanced task exposure during training, we initialize each agent in the batch with a task sampled uniformly at random. Furthermore, upon each environment reset, agents are reassigned new tasks, also randomly selected from a uniform distribution. This strategy dynamically redistributes task assignments across the batch throughout training, thereby mitigating potential overfitting to specific tasks. For task generalization experiment, we train the policy within a single environment for 800 epochs, which requires approximately 7 hours on a single Nvidia RTX 4090 GPU. For environment adaptation experiment, we alternate training between two distinct environments represented by 3DGS models. Specifically, the policy is trained for 200 epochs in one environment before switching to the other, with this cycle repeated until 800 epochs are completed in each environment. The total training duration under this multi-environment setting is approximately 14 hours on the same desktop mentioned above. Due to the additional computational overhead introduced by VLM inference during forward simulation, our approach incurs a 5% wall-clock time increase during training compared to the original GRaD-Nav framework.

## IV. EXPERIMENTAL RESULTS

In this section, we aim to address the following research questions through empirical evaluation:

TABLE I: Reward function terms and their respective weights. Here, early terminations includes (i) droneâs height exceeds the ceilingâs height (3m), (ii) droneâs linear velocity exceeds the threshold (20m/s), (iii) drone has been out of bound for more than 3m, i.e. $\pmb { x } _ { o . b . }$ , $y _ { o . b . } \ \geq \ 3 ,$ , q0 is the droneâs initial quaternion, $h _ { \mathrm { t a r g e t } }$ is the target height of the drone (also serving as the initial hover height), $\hat { y } _ { \mathrm { y a w } }$ is the normalized heading direction of the drone, $d _ { \mathrm { w p } }$ is the distance to the next waypoint, $\pmb { d } _ { \mathrm { o b s t } }$ is the distance to the closest obstacle in the droneâs field of view (FOV), $\mathbf { \bot } \mathbf { x } _ { 0 . \mathrm { b . } }$ and ${ \mathbf { 3 0 . 6 . } }$ are the distances from the 3DGS map boundary, and ${ \pmb v } _ { \mathrm { d e s } }$ is the desired velocity direction from the pre-planned reference trajectory.
<table><tr><td>Reward</td><td>Equation  $( r _ { i } )$  Weight (wi)</td></tr><tr><td>Survival Linear velocity Pose</td><td>Safe Control Rewards ! {early terminations} 8.0  $\| \pmb { v } \| ^ { 2 }$  -0.5  $\left\| q - q _ { 0 } \right\|$  -0.5</td></tr><tr><td>Height Action</td><td> $( h - h _ { \mathrm { t a r g e t } } ) ^ { 2 }$  â2.0  $\| \pmb { a } _ { t } \| ^ { 2 }$  -1.0 â1.0</td></tr><tr><td>Action rate</td><td> $\| \pmb { a } _ { t } - \pmb { a } _ { t - 1 } \| ^ { 2 }$   $\| \pmb { a } _ { t } - 2 \pmb { a } _ { t - 1 } + \pmb { a } _ { t - 2 } \| ^ { 2 }$  â1.0</td></tr><tr><td>Smoothness</td><td></td></tr><tr><td></td><td>Efficient Navigation Rewards</td></tr><tr><td></td><td> $\frac { \overline { { { \bf { v } } } } _ { x \overline { { { y } } } } } { \| { \bf { v } } _ { x y } \| } \cdot \overline { { \hat { { y } } } } _ { \mathrm { { y a w } } }$ </td></tr><tr><td>Yaw alignment</td><td></td></tr><tr><td></td><td> $\exp ( - d _ { \mathrm { w p } } )$ </td></tr><tr><td>Waypoint</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Obstacle avoidance</td><td> $\mathbf { \Delta } d _ { \mathrm { o b s t } }$ </td></tr><tr><td></td><td></td></tr><tr><td>Out-of-map</td><td> ${ \pmb x } _ { \mathrm { o . b . } } ^ { 2 } + { \pmb y } _ { \mathrm { o . b . } } ^ { 2 }$ </td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Ref. traj. tracking</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td> $\begin{array} { r } { \| \frac { \pmb { v } } { \| \pmb { v } \| } - \frac { \pmb { v } _ { \mathrm { d e s } } } { \| \pmb { v } _ { \mathrm { r e f } } \| } \| } \end{array}$ </td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

1) What is the functional contribution of the VLM within our VLA drone flight framework?

2) In what ways does the MoE architecture facilitate multi-task, multi-environment generalization and mitigate catastrophic forgetting?

3) Why is the use of DiffRL essential to the effectiveness of our proposed approach?

We conduct two experiments that focusing on new tasks generalization and multi-environment adaptation in both simulation and on a real drone, respectively. For simulation, the droneâs initial positions are randomized in a cube with side length of 1m, centered at (0,0,1.4)m; their initial poses are also randomized, $\phi , \theta , \psi \in [ - 0 . 2 5 , 0 . 2 5 ]$ . For real hardware experiments, our drone is mounted with a Pixracer low-level flight controller and an Intel Realsense D435 camera. All the models are deployed and running onboard using a Nvidia Jetson Orin Nano companion computer. As mentioned in Section III-B.1, the VLM runs once every 10 nominal time steps, updating the VLM feature vector $e _ { t }$ that is saved in the buffer. The overall onboard policy inference frequency is 25 Hz.

## A. Multi Tasks Generalization Experiment

In this experiment, we evaluate the effectiveness of our VLA-based drone flight framework in executing multiple long-horizon tasks, with a particular focus on its generalization capability to unseen tasks. To investigate the functional contribution of the VLM, we conduct an ablation study in which the VLM module is removed and task instructions are encoded using one-hot vectors. Additionally, we compare our approach with a Proximal Policy Optimization (PPO) [28] baseline that employs the same VLM module, in order to highlight the benefits of using DiffRL in scenarios where sample efficiency is critical. As defined in Section III-A, we construct a set of two-stage long-horizon tasks that emphasize varying degrees of spatial reasoning and object recognition. Each task consists of a natural language instruction paired with a reference trajectory. We train our policy on 8 tasks and evaluate its performance on 4 held-out, untrained tasks. The training hyperparameters are summarized in Table VI; all methods are trained under identical conditions in terms of the total number of simulation steps.

TABLE II: Simulation results of multi task generalization experiment. For 8 trained tasks and 4 untrained tasks, the evaluation is conducted on 6 trials for each task, with the average rewards and success rates (SR) reported.
<table><tr><td rowspan="2">Method</td><td colspan="8">Evaluation Results (Trained | Untrained)</td></tr><tr><td colspan="2">Reward</td><td colspan="2">Stage 1 SR</td><td colspan="2">Stage 2 SR</td><td colspan="2">Overall SR</td></tr><tr><td>PPO</td><td>1272.8</td><td>1168.3</td><td>0/24</td><td>0/12</td><td>0/24</td><td>N/A</td><td>0/24</td><td>0/12</td></tr><tr><td>w/o VLM</td><td>3725.7</td><td>3246.5</td><td>3/24</td><td>1/12</td><td>1/3 |</td><td>|0/1</td><td>1/24</td><td>0/12</td></tr><tr><td>Proposed</td><td>5068.8</td><td>4447.3</td><td>24/24</td><td>10/12</td><td>20/24</td><td> | 9/10</td><td>20/24</td><td>9/12</td></tr></table>

In the following, we present the results of our multitask generalization experiments conducted both in simulation and on a real drone platform. For each setting, we report the success rates (SR) of Stage 1 and Stage 2 separately, along with the overall task success rate. Since Stage 2 is contingent upon the successful completion of Stage 1, we only evaluate Stage 2 SR for trials where Stage 1 was successful. The success criteria for each stage go beyond avoiding early termination (as specified in Table I) and preventing collisions. Specifically, Stage 1 requires the drone to navigate past a gate in the correct direction as indicated by the instruction. In Stage 2, the drone must correctly identify and approach the target object referred to in the instruction. A Stage 2 trial is considered successful if the drone ends up closer to the target object than to either of the two distractor objects. Formally, Stage 2 is successful if:

$$
\left\| \mathbf { p } - \mathbf { p } _ { \mathrm { t } } \right\| < \operatorname* { m i n } \left( \left\| \mathbf { p } - \mathbf { p } _ { \mathrm { 1 } } \right\| , \ \left\| \mathbf { p } - \mathbf { p } _ { \mathrm { 2 } } \right\| \right)\tag{10}
$$

where p denote the final position of the drone, and $\mathbf { p } _ { \mathrm { t } } , \mathbf { p } _ { 1 } ,$ and p2 denote the positions of the target object and the two distractor objects, respectively. Figure 2 demonstrates examples of policy generalizing to untrained long horizon tasks in simulator.

The results are summarized in Table II and Table III for simulation and real-world experiments, respectively.

The results show that: (i) standard model-free RL (e.g., PPO) fails to learn effective policies for our VLA-based multi-task drone flight within $\mathrm { \bar { 1 } \times 1 0 ^ { 7 } }$ steps; (ii) the VLM module is criticalânot just as a text encoder, but for grounding instructions to visual context, enabling task understanding; (iii) our method achieves 83%/75% success on trained/unseen tasks in simulation, and over 50% on both in real-world tests, indicating strong sim-to-real transfer; (iv) performance drops more in Stage 2 than Stage 1 on hardware, likely due to challenges in detection, fine control, and error accumulation.

<!-- image-->  
Fig. 2: Example trajectories of untrained long horizon tasks. The instructions are âGO THROUGH gate then STOP over CARTâ (left) and âFLY past the RIGHT side of the gate then STOP over MONITORâ (right).

TABLE III: Real hardware experiment results of multi task generalization experiment. For 8 trained tasks and 4 untrained tasks, the evaluation is conducted on 3 trials for each task, with the average rewards and success rates (SR) reported.
<table><tr><td>Method</td><td colspan="4">Success Rate (Trained Untrained</td></tr><tr><td>PPO</td><td>Stage 1 0/24</td><td>0/12</td><td>Stage 2 0/24 | N/A</td><td>Overall 0/24 0/12</td></tr><tr><td>w/o VLM</td><td>1/24</td><td>0/12</td><td>0/1 | N/A</td><td>0/24 0/12</td></tr><tr><td>Proposed</td><td>21/24</td><td>9/12</td><td>16/21 | 6/9</td><td>16/24 6/12</td></tr></table>

## B. Multi Environment Adaptation Experiment

In this experiment, we evaluate the effectiveness of drone flight VLA framework in following human instructions and executing multiple tasks across multiple environments. The goal is to investigate the functional contribution of MoE architecture in enabling multi-task, multi-environment generalization while mitigating catastrophic forgetting.

To this end, we conduct ablation studies with two alternative settings, each replacing the MoE architecture with a conventional multi-layer perceptron (MLP) actor network: (i) A baseline actor network with the same architecture as each individual expert in the MoE. We refer to this configuration as the single expert (SE) baseline in the following sections. (ii) A larger actor network that horizontally combines two experts (corresponding to top k = 2), ensuring the total number of parameters matches that of the MoE-based actor. We refer to this configuration as the large network (LN) baseline in the following sections.

The tasks used in this experiment are defined in Section III-A, focusing on spatial reasoning and correct gate traversal, i.e. the first stage of the tasks introduced in Section IV-A. We evaluate performance in both simulation and real-world deployments, reporting training reward, evaluation reward, and task-specific success rates (SR) across environments. The success criterion is identical to that of Stage 1 as defined in Section IV-A: the drone must traverse the designated gate in the correct direction as specified by the instruction, without incurring collisions or early termination. The training hyperparameters are provided in Table VI. The simulation and real hardware experiments results are demonstrated in Table IV and Table V, respectively. Figure 3 shows real-world hardware demonstrations of the drone executing tasks in multiple environments.

Task 1: Left  
Task 2: Through  
Task 3: Above  
Task 4: Right  
<!-- image-->  
Fig. 3: Demonstration of multi-environment adaptation in real-world experiments using video frame overlay visualization. The top row shows the drone flying to the left of, through, above, and to the right of the gate in the middle-gate environment. The bottom row shows the drone executing the same directional tasks in the left-gate environment. Red arrowed curves illustrate the approximate flight trajectories. The learned policy demonstrates robust generalization to varying environments, adapting to changes in gate positions and the presence of distractor objects.

TABLE IV: Simulation results of the multi-environment adaptation experiment are presented. For each of the 4 tasks in the 2 different surrounding environments, evaluation is conducted over 6 trials per task. The reported metrics include the average evaluation rewards and success rates.  
TABLE V: Real hardware experiment results of the multienvironment adaptation experiment. For each of the 4 tasks across 2 different surrounding environments, evaluation is performed over 3 trials per task. The reported metrics include the average evaluation rewards and success rates.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Reward</td><td colspan="8">Success Rate (Left | Mid.)</td></tr><tr><td colspan="2">Through</td><td colspan="2">Right</td><td colspan="2">Left</td><td colspan="2">Over</td></tr><tr><td>w/o MoE, SE</td><td>3566.7</td><td>5/6</td><td>1/6</td><td>5/6</td><td>0/6</td><td>5/6</td><td>| 3/6</td><td>4/6</td><td> |2/6</td></tr><tr><td>w/o MoE, LN</td><td>3764.6</td><td>6/6</td><td> | 2/6</td><td>6/6</td><td>s | 1/6</td><td>5/6</td><td> | 3/6</td><td>5/6</td><td> | 2/6</td></tr><tr><td>Proposed</td><td>4169.5</td><td>6/6</td><td>| 5/6</td><td>5/6</td><td>4/6</td><td>6/6</td><td>4/6</td><td>5/6</td><td>4/6</td></tr></table>

The proposed MoE policy generalizes well across environments. In the left gate setting, all methodsâSE, LN, and MoEâperform comparably. However, in the mid gate environment, SE and LN degrade sharply (SE: 25% sim / 8% real, LN: 33% sim / 25% real), while MoE maintains 70% in simulation and 67% on hardware. This is due to catastrophic forgetting in SE and LN, which overfit to the last trained (left gate) task. In contrast, MoE avoids forgetting by dynamically routing through experts, preserving environmentspecific knowledge and enabling robust performance across tasks.

<table><tr><td rowspan="2">Method</td><td colspan="8">Success Rate (Left | Mid.)</td></tr><tr><td colspan="2">Through</td><td colspan="2">Right</td><td colspan="2">Left</td><td colspan="2">Over</td></tr><tr><td>w/o MoE, SE</td><td>3/3</td><td>0/3</td><td>3/3</td><td>|0/3</td><td>1/3</td><td>1/3</td><td>2/3</td><td>0/3</td></tr><tr><td>w/o MoE, LN</td><td>2/3</td><td>0/3</td><td>3/3</td><td>1/3</td><td>1/3</td><td>2/3</td><td>2/3</td><td>0/3</td></tr><tr><td>Proposed</td><td>3/3</td><td>2/3</td><td>3/3</td><td>2/3</td><td>2/3</td><td>| 3/3</td><td>2/3</td><td>1/3</td></tr></table>

Figure 4 illustrates the expert utilization patterns during the execution of the same task (âGO THROUGH gateâ) under different surrounding environments. The results demonstrate that the MoE architecture adaptively allocates expert resources according to the specific demands of each environment. In both plots, a noticeable shift in expert activation occurs between time steps 200 and 300, which corresponds to the phase when the drone approaches the gate. This transition indicates that the gating network dynamically adjusts expert weights in response to changing environmental context and task phase. These observations further support the effectiveness of the MoE architecture in enabling efficient resource allocation and facilitating robust generalization across multiple tasks and environments.

<!-- image-->  
Fig. 4: Expertsâ usage intensity when executing the same task (âGO THROUGH gateâ) at different surrounding environments (top-left gate, bottom-middle gate).

## C. Task Shift Experiment

To further validate and analyze the functional contribution of the VLM within our VLA-based drone flight framework, we design an experiment in which the task instruction is modified during the execution of a long-horizon task. This experiment aims to assess the VLMâs ability to re-ground a new instruction within the current visual context and evaluate whether the learned policy can successfully adapt mid-flight to a new task objective.

During execution, we record the cosine similarity between the VLMâs text and visual embeddings every 10 time steps, matching the update frequency of the VLM module. The cosine similarity is computed as:

$$
{ \mathrm { c o s i n e } } _ { \mathrm { - } \mathrm { } \mathrm { { s i m i l a r i t y } } } ( e _ { \mathrm { t e x } } , e _ { \mathrm { v i s } } ) = { \frac { e _ { \mathrm { t e x } } \cdot e _ { \mathrm { v i s } } } { \left\| e _ { \mathrm { t e x } } \right\| \left\| e _ { \mathrm { v i s } } \right\| } }\tag{11}
$$

where $e _ { \mathrm { t e x } }$ and $e _ { \mathrm { v i s } }$ denote the text and visual embeddings produced by the VLMâs text and visual encoders, respectively. At time step 100, the task instruction is changed from âFLY past the LEFT side of the gate then FLY to LADDER baseâ to âFLY past the RIGHT side of the gate then STOP over MONITOR.â The resulting drone trajectory and the corresponding VLM similarity scores over time are visualized in Figure 5.

As shown in Figure 5, the VLMâs normalized match score exhibits a significant drop shortly after the task instruction is changed at step 100, reflecting a temporary mismatch between the new instruction and the droneâs current visual context. Following this drop, the score gradually recovers as the drone adjusts its behavior to align with the new instruction. This dynamic trend confirms that the VLM generates meaningful, context-aware similarity scores that reflect the semantic alignment between perception and language. Importantly, despite the abrupt mid-flight task switch, the policy is able to successfully complete the new task, demonstrating the robustness and adaptability of our framework under taskshift scenarios.

<!-- image-->

<!-- image-->  
Fig. 5: Task-switching experiment with instruction change at step 100. Left: Experiment scene showing the drone flying past the gate. Right: Normalized cosine similarity between text and visual embeddings over time, reflecting the VLMâs ability to re-ground the new instruction.

## V. CONCLUSION

In this paper, we present GRaD-Nav++, a fully onboard VLA framework that turns high-level natural-language commands directly into low-level drone controls. Experiments reveal that the Vision-Language Model plays a critical role in interpreting high-level natural language commands and linking them to visual observations, thereby enabling the drone to make task-relevant decisions in complex environments. The MoE action head, through sparse routing, dedicates capacity to the most relevant behaviors and thus boosts generalization while preventing catastrophic forgetting. Finally, training with DiffRL inside the 3DGS model supplies smooth endto-end gradients, improves sample efficiency, and produces policies that transfer to real hardware without additional tuning. The possible future directions including: coupling the policy with learned world models for long-horizon predictive planning and even greater sample efficiency.

Limitations: Due to limitations in model capacity and the size of the task training set, our policy is capable of generalizing to novel combinations of sub-tasks that were individually encountered during training. For example, after training on tasks such as âGO THROUGH gate then STOP over MONITORâ and âFLY past the LEFT side of the gate then STOP over CART,â the policy successfully completes âGO THROUGH gate then STOP over CARTâ by recombining familiar components. However, it fails on instructions like âFLY OUT the LEFT door then LAND on the table,â where both sub-tasks are entirely unseen during training. As such, we do not claim our framework to be a general solution for open-vocabulary embodied AI.

[1] F. Gao, L. Wang, B. Zhou, X. Zhou, J. Pan, and S. Shen, âTeachrepeat-replan: A complete and robust system for aggressive flight in complex environments,â IEEE Transactions on Robotics, vol. 36, no. 5, pp. 1526â1545, 2020.

[2] D. Hanover, A. Loquercio, L. Bauersfeld, A. Romero, R. Penicka, Y. Song, G. Cioffi, E. Kaufmann, and D. Scaramuzza, âAutonomous drone racing: A survey,â IEEE Transactions on Robotics, 2024.

[3] J. Low, M. Adang, J. Yu, K. Nagami, and M. Schwager, âSous vide: Cooking visual drone navigation policies in a gaussian splatting vacuum,â arXiv preprint arXiv:2412.16346, 2024.

[4] Q. Chen, J. Sun, N. Gao, J. Low, T. Chen, and M. Schwager, âGrad-nav: Efficiently learning visual drone navigation with gaussian radiance fields and differentiable dynamics,â arXiv preprint arXiv:2503.03984, 2025.

[5] Z. Xu, X. Han, H. Shen, H. Jin, and K. Shimada, âNavrl: Learning safe flight in dynamic environments,â IEEE Robotics and Automation Letters, 2025.

[6] Y. Hu, Y. Zhang, Y. Song, Y. Deng, F. Yu, L. Zhang, W. Lin, D. Zou, and W. Yu, âSeeing through pixel motion: Learning obstacle avoidance from optical flow with one camera,â IEEE Robotics and Automation Letters, 2025.

[7] B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal et al., âLanguage models are few-shot learners,â arXiv preprint arXiv:2005.14165, vol. 1, p. 3, 2020.

[8] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PmLR, 2021, pp. 8748â8763.

[9] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, âSigmoid loss for language image pre-training,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 975â11 986.

[10] J. Sun, C. Zheng, E. Xie, Z. Liu, R. Chu, J. Qiu, J. Xu, M. Ding, H. Li, M. Geng et al., âA survey of reasoning with foundation models: Concepts, methodologies, and outlook,â ACM Computing Surveys, 2023.

[11] G. Chen, X. Yu, N. Ling, and L. Zhong, âTypefly: Flying drones with large language model,â arXiv preprint arXiv:2312.14950, 2023.

[12] W. Wang, Y. Li, L. Jiao, and J. Yuan, âGsce: A prompt framework with enhanced reasoning for reliable llm-driven drone control,â in 2025 International Conference on Unmanned Aircraft Systems (ICUAS). IEEE, 2025, pp. 441â448.

[13] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn et al., âRt-2: Visionlanguage-action models transfer web knowledge to robotic control,â arXiv preprint arXiv:2307.15818, 2023.

[14] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., âOpenvla: An open-source vision-language-action model,â arXiv preprint arXiv:2406.09246, 2024.

[15] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter et al., âÏ0: A vision-languageaction flow model for general robot control, 2024,â URL https://arxiv. org/abs/2410.24164.

[16] V. Serpiva, A. Lykov, A. Myshlyaev, M. H. Khan, A. A. Abdulkarim, O. Sautenkov, and D. Tsetserukou, âRacevla: Vla-based racing drone navigation with human-like behaviour,â arXiv preprint arXiv:2503.02572, 2025.

[17] A. Lykov, V. Serpiva, M. H. Khan, O. Sautenkov, A. Myshlyaev, G. Tadevosyan, Y. Yaqoot, and D. Tsetserukou, âCognitivedrone: A vla model and evaluation benchmark for real-time cognitive task solving and reasoning in uavs,â arXiv preprint arXiv:2503.01378, 2025.

[18] B. Kerbl, G. Rainer, Z. Lahner et al., â3d gaussian splatting for real-time radiance field rendering,â Advances in Neural Information Processing Systems (NeurIPS), 2023.

[19] J. Xu, V. Makoviychuk, Y. Narang, F. Ramos, W. Matusik, A. Garg, and M. Macklin, âAccelerated policy learning with parallel differentiable simulation,â arXiv preprint arXiv:2204.07137, 2022.

[20] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem, âBraxâa differentiable physics engine for large scale rigid body simulation,â arXiv preprint arXiv:2106.13281, 2021.

[21] T. A. Howell, S. Le Cleacâh, J. Z. Kolter, M. Schwager, and Z. Manchester, âDojo: A differentiable simulator for robotics,â arXiv preprint arXiv:2203.00806, vol. 9, no. 2, p. 4, 2022.

[22] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, âAdaptive mixtures of local experts,â Neural Computation, vol. 3, no. 1, pp. 79â 87, 1991.

[23] M. I. Jordan and R. A. Jacobs, âHierarchical mixtures of experts and the em algorithm,â Neural Computation, vol. 6, no. 2, pp. 181â214, 1994.

[24] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, âOutrageously large neural networks: The sparsely-gated mixture-of-experts layer,â arXiv preprint, vol. arXiv:1701.06538, 2017. [Online]. Available: https://arxiv.org/abs/1701.06538

[25] S. Huang, Z. Zhang, T. Liang, Y. Xu, Z. Kou, C. Lu, G. Xu, Z. Xue, and H. Xu, âMentor: Mixture-of-experts network with task-oriented perturbation for visual reinforcement learning,â 2024. [Online]. Available: https://arxiv.org/abs/2410.14972

[26] I. Higgins, L. Matthey, A. Pal, C. P. Burgess, X. Glorot, M. M. Botvinick, S. Mohamed, and A. Lerchner, âbeta-vae: Learning basic visual concepts with a constrained variational framework.â ICLR (Poster), vol. 3, 2017.

[27] I. M. A. Nahrendra, B. Yu, and H. Myung, âDreamwaq: Learning robust quadrupedal locomotion with implicit terrain imagination via deep reinforcement learning,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 5078â5084.

[28] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, âProximal policy optimization algorithms,â arXiv preprint arXiv:1707.06347, 2017.

## VI. APPENDIX

TABLE VI: Hyper-parameters table of different training methods.
<table><tr><td>Parameters</td><td>Ours</td><td>PPO</td><td>w/o MoE</td></tr><tr><td>Number of envs</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Discount factor Î³</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>Actor learning rate</td><td>3e-4</td><td>3e-4</td><td>3e-4</td></tr><tr><td>Critic learning rate</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>CENet learning rate</td><td>5e-4</td><td>5e-4</td><td>5e-4</td></tr><tr><td>GAE Î»</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td>Horizon length</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Critic updates</td><td>16</td><td>-</td><td>16</td></tr><tr><td>Clipping parameter â¬</td><td>-</td><td>0.1</td><td>-</td></tr><tr><td>Entropy coefficient</td><td>-</td><td>1e-3</td><td>-</td></tr><tr><td>MoE Aux. loss weight</td><td>0.5</td><td>-</td><td>-</td></tr><tr><td>MoE balance weight</td><td>10</td><td>-</td><td></td></tr><tr><td>MoE entropy weight</td><td>0.1</td><td>-</td><td></td></tr></table>

TABLE VII: High level natural language instructions table for drone flight VLA tasks in Section III-A.
<table><tr><td>Stage #1</td><td>Stage #2</td></tr><tr><td>GO THROUGH gate</td><td>STOP over MONITOR</td></tr><tr><td>FLY past the RIGHT side of the gate FLY past the LEFT side of the gate</td><td>STOP over CART</td></tr><tr><td>FLY ABOVE gate</td><td>FLY to LADDER base</td></tr></table>