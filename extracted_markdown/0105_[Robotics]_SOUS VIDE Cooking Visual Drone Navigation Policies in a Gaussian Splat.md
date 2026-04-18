# SOUS VIDE: Cooking Visual Drone Navigation Policies in a Gaussian Splatting Vacuum

JunEn Low, Maximilian Adang, Javier Yu, Keiko Nagami, and Mac Schwager

AbstractâWe propose a new simulator, training approach, and policy architecture, collectively called SOUS VIDE, for end-toend visual drone navigation. Our trained policies exhibit zeroshot sim-to-real transfer with robust real-world performance using only onboard perception and computation. Our simulator, called FiGS, couples a computationally simple drone dynamics model with a high visual fidelity Gaussian Splatting scene reconstruction. FiGS can quickly simulate drone flights producing photorealistic images at up to 130 fps. We use FiGS to collect 100k-300k image/state-action pairs from an expert MPC with privileged state and dynamics information, randomized over dynamics parameters and spatial disturbances. We then distill this expert MPC into an end-to-end visuomotor policy with a lightweight neural architecture, called SV-Net. SV-Net processes color image, optical flow and IMU data streams into low-level thrust and body rate commands at 20 Hz onboard a drone. Crucially, SV-Net includes a learned module for low-level control that adapts at runtime to variations in drone dynamics. In a campaign of 105 hardware experiments, we show SOUS VIDE policies to be robust to 30% mass variations, 40 m/s wind gusts, 60% changes in ambient brightness, shifting or removing objects from the scene, and people moving aggressively through the droneâs visual field. Code, data, and experiment videos can be found on our project page: https://stanfordmsl.github.io/SousVide/.

## I. INTRODUCTION

L EARNED visuomotor policies offer a compelling alter-native to traditional drone navigation stacks by unifying native to traditional drone navigation stacks by unifying perception and control into a streamlined framework. Unfortunately, training policies with human-like agility and collision avoidance requires a large corpus of visual and state data, making behavior cloning from human pilot demonstrations impractical. Simulation provides a promising alternative, but the sim-to-real gap has remained a persistent obstacle to realworld deployment. Recent work has demonstrated that in controlled environments and with carefully built digital twins in simulation, learned policies can achieve highly agile, superhuman performance [1]â[3]. However, this raises the question: can we train a visuomotor policy to navigate unstructured realworld environments with minimal human curation?

We address this challenge with FiGS (Flying in Gaussian Splats), a photorealistic drone simulator combining a Gaussian Splat (GSplat) [4] scene model with a lightweight 10- dimensional drone dynamics model using thrust and body rate inputsâequivalent to the Acro mode used by expert human pilots. FiGS reconstructs scenes from video captures, like publicly available footage or smartphone recordings, processed with standard tools [5] and generates realistic image sequences and state estimation data for a drone at up to 130 fps on a standard GPU, all within an hour of acquiring video data. This contrasts with the current practice of approximating each real scene with a handcrafted simulation instance [6]â[11] that can take weeks of laborious sim-to-real transfer to perfect.

<!-- image-->  
Fig. 1. SOUS VIDE overview: We train our FiGS simulator from a hand held camera. We use FiGS to generate flight demonstrations (image/stateaction pairs) from an MPC expert with privileged information randomized over dynamics parameters and positional disturbances. We use this data to train our policy, SV-Net, which operates solely with onboard observations.

Building on FiGS, we develop SOUS VIDEâ  (Scene Optimized Understanding via Synthesized Visual Inertial Data from Experts), a behavior cloning pipeline that produces a robust drone navigation policy capable of zero-shot sim-toreal transferâtrained entirely within simulation without realworld demonstrations or fine-tuning (Fig. 1). Specifically, we use FiGS to generate 100k-300k image/state-action pairs from an expert MPC policy following a desired nominal trajectory within a GSplat. The MPC has access to the ground truth state in the simulator and is therefore able to demonstrate high-quality, collision-free trajectories. To obtain stable and robust flight, we randomize dynamics parameters and spatial perturbations and record the MPCâs response. We use these expert demonstrations to train a student policy without privileged information (no lateral position data).

The learned policy produced by SOUS VIDE, SV-Net, is a novel architecture designed to process both images and observable state history while remaining efficient enough to run onboard the drone. The policy ingests images using a SqueezeNet [12], re-trained on our data, and outputs a feature vector that is fused with observable data across several small Multi-Layer Perceptrons (MLPs) to produce low-level bodyrate commands. Within these MLPs, we also implement a form of the Rapid Motor Adaptation (RMA) concept from [13], which takes a history of observable data from a sliding window and produces a latent code that captures evolving flight dynamics in real time. We find this RMA module is crucial for robust flight, adapting online to variations such as battery drain, rotor downwash effects, and wind gusts.

To summarize, we make the following contributions:

1) Flying in Gaussian Splats (FiGS): A simulator coupling GSplat scene models with drone dynamics for efficient and photorealistic visual-inertial data generation.

2) Scalable Visuomotor Policy Generation: We use FiGS to generate large synthetic datasets to train visuomotor policies that transfer zero-shot to real-world flight.

3) SV-Net: An onboard policy that fuses image and observable states to infer thrust and body rates while continuously adapting to changing flight conditions.

We evaluate SOUS VIDE policies across 105 hardware flights in 4 different scenes, testing 9 different experimental conditions. We demonstrate our policyâs robustness to 30% mass variations, 40 m/s wind gusts, 60% changes in ambient brightness, shifting or removing objects from the scene, and people moving aggressively through the droneâs visual field.

The paper is organized as follows: Section II reviews related work, Section III describes the FiGS simulator, Section IV details the MPC-based data synthesis, and Section V presents the SV-Net policy. Hardware experiments are in Section VI, with conclusions, limitations, and future work in Section VII.

## II. RELATED WORK

Drone Polices Trained with GSplats: Learned representations like GSplats have proven effective in training visuomotor policies across many domains, from manipulation [14], [15] to bipedal locomotion [16] to aerial robotics [17], [18]. Closest to our approach, [17] uses a learned representation and an MPC expert to train a trajectory-following policy but requires an initial sample of real-world expert flight demonstrations to be collected via motion capture. Additionally, its 45â¦ downwardfacing camera focuses on ground features, missing the spatial information of a forward view needed for obstacle avoidance, as seen in drone racing [2]. Meanwhile, [18], treats the GSplat reconstruction as a background, relying instead on colored spheres as visual markers injected into both the simulation and real-world scenes as the basis for decision making. Moreover, it relies on velocity commands that encode only high-level approach and turn behaviors, delegating low-level control to a manufacturer-supplied autonomy stack. To the best of our knowledge, SOUS VIDE is the first method to leverage GSplats for generating low-level drone navigation policies for unstructured environments without assistive infrastructure or real-world expert flight data.

Training Drone Policies in Simulation: Simulators offer scalability and safety in collecting training data, but they introduce a sim-to-real gap, making it difficult to transfer policies to the real world. Many existing works use domain randomization [19] to robustify policies, as we also do. For drones, another common strategy is to enhance the fidelity of the drone dynamics model with drag and other effects [20]â[22], while another is to improve rendering pipelines and graphics assets [6]â[11]. However, none of these approaches can match the speed and visual fidelity of GSplat scene reconstructions. Another prevalent solution involves visual abstractions, such as depth maps [1], [23] or learned feature embeddings [2], which aim to distill visual information into a domain-invariant representation. However, this discards information encoded in raw pixel data that could otherwise improve task performance.

High-performance simulation-trained policies have been demonstrated for drone racing [2], [3], marking an impressive technological achievement. However, these methods blend real and simulation flight data, physics and learning-based models, and hand-engineered visual features cued into racing gates. In contrast, our method can train a policy using video clips of the scene and can transfer zero-shot to the real-world with only minimal tuning of easily measurable parameters.

Rapid Motor Adaptation: RMA, a technique originally developed for quadruped locomotion policies in [13], can be viewed as a pre-trained alternative to online parameter estimation [24] where an encoder is trained to take in a sensing history to produce a latent vector that captures runtime operating conditions (e.g., terrain for a quadruped, or flight dynamics for a drone). RMA has been adapted for drones in [25], [26] where they have been show to achieve stable flight with impressive robustness. However, they are not designed for visual navigation. Our lightweight RMA implementation is crucial for real-world robustness, addressing variations in both modeled and unmodeled drone dynamics.

Generalist Collision Avoidance Policies: Some existing works train policies to steer a drone through environments not seen at training time, often focusing on a particular scene domain like forests [27], office buildings [28], or urban roadways [29]. Such policies have been trained both with Reinforcement Learning (RL) [28] and with Behavior Cloning (BC) [27], [29], using both simulated [23], [28] and real-world [27], [30]â[32] data. Recent examples strive toward policies that can operate across different robot embodiments [31], [32]. While impressive for their generality, they often treat the drone as a pseudo-static ground robot by using a finely tuned onboard Visual-Inertial Odometry (VIO) stack to constrain the dynamics to planar velocities and yaw. This fails to exploit the droneâs full agility when navigating cluttered indoor environments with complex 3D trajectories. In contrast, SOUS VIDE directly commands thrust and body rates, mirroring the capabilities of expert human pilots.

## III. FLYING IN GAUSSIAN SPLATS (FIGS)

FiGS, our lightweight GSplat-based flight simulator, consists of a GSplat model trained from video captures of the scene, within which a drone is simulated using a simplified 10-dimensional drone dynamics model.

Gaussian Splats: 3D Gaussian Splatting [4] is a learned representation approach that approximates the geometry and appearance of real-world scenes using a large collection of Gaussiansâpotentially millionsâeach parameterized by its position, covariance, color, and opacity. They leverage highspeed, projection-based differentiable rasterization and are trained from sparse RGB images by backpropagating through the rasterization to minimize photometric error. This approach enables photorealistic reconstructions and full-resolution renders at over 100 fps on a standard desktop GPU.

In this work, we generate GSplats from short video recordings (2-3 minutes) of scene walk-throughs with a handheld camera. From the video we extract a set of training images and use the open-source tool Nerfstudio [5], [33] to train the GSplat model. The resulting model ${ \mathcal { G S } } _ { \phi }$ , where $\phi$ are its parameters, can render photorealistic images from a virtual camera placed at any pose within the region covered by the training images. Given a camera pose $( p , q )$ , where p represents the position and q the orientation in quaternion form, the rendered image is given by ${ \cal I } = \mathcal { G } S _ { \phi } ( p , { \bf q } )$ . To obtain metric scale and align the GSplat frame to a known global frame in the scene, we start the video recording with an ArUco tag marker in frame.

Drone Dynamics Model: Our model operates in the world, body, and camera frames (W, B, C) and uses a 10-dimensional semi-kinematic state vector, $\pmb { x } = \left[ \pmb { p } _ { \mathcal { W } } , \pmb { v } _ { \mathcal { W } } , \pmb { q } _ { B \mathcal { W } } \right] ^ { T }$ , representing position $\pmb { p } _ { \mathcal { W } } = ( p _ { x } , p _ { y } , p _ { z } )$ , velocity $\boldsymbol { v } _ { \mathcal { W } } = ( v _ { x } , v _ { y } , v _ { z } )$ and orientation qBW $\mathbf { \Psi } = \left( q _ { x } , q _ { y } , q _ { z } , q _ { w } \right)$ . The control inputs, $\mathbf { \boldsymbol { u } } = \left[ f _ { t h } , \omega _ { B } \right] ^ { T }$ , include normalized thrust $f _ { t h }$ and angular velocity $\boldsymbol { \omega } _ { B } = ( \omega _ { x } , \omega _ { y } , \omega _ { z } )$ . This produces model dynamics

$$
\begin{array} { c } { { p _ { \mathcal { W } } = v _ { \mathcal { W } } , } } \\ { { \dot { v } _ { \mathcal { W } } = g z _ { \mathcal { W } } - k _ { t h } \frac { f _ { t h } } { m _ { d r } } z _ { \mathcal { B } } , } } \\ { { \dot { q } _ { \mathcal { B W } } = \displaystyle \frac { 1 } { 2 } W ( \boldsymbol { \omega } _ { \mathcal { B } } ) q _ { \mathcal { B W } } , } } \end{array}\tag{1}
$$

where g is gravitational acceleration, $W ( \omega _ { B } )$ is the quaternion multiplication matrix, and zW , zB are the z-axis unit vectors of the world and body frames. The thrust coefficient and mass, $( k _ { t h } , m _ { d r } )$ , are stored in the drone parameter vector Î¸.

Thrust and body rate commands are the standard low-level input for most flight controllers [1]â[3], [21], providing robust tracking through high-rate gyroscope feedback. This choice also enhances platform agnosticism in a cost-effective manner and is widely favored by expert human pilots. Moreover, for our use case, it offers the significant advantage of omitting the rotational acceleration equations (Eulerâs equations) in (1).

We forward integrate these equations of motion using ACA-DOS [34], a highly efficient trajectory optimizer that provides direct access to its dynamics update function, to obtain the state trajectory $\mathbf { X } = \{ \pmb { x } _ { 0 } , \dots , \pmb { x } _ { K } \}$ and input trajectory ${ \bf U } =$ $\{ { \pmb u } _ { 0 } , \dots , { \pmb u } _ { K - 1 } \}$ , where K denotes the number of discrete time steps. Applying the body-camera transform $T _ { c } ^ { B }$ to the pose variables within X, we can render the image sequence $\pmb { \mathcal { T } } = \{ \pmb { I } _ { 0 } , \dots , \pmb { I } _ { K } \}$ as seen by the onboard camera from the GSplat. This data can be used in an RL or BC framework, and can supervise the training of either state-feedback or imagefeedback policies. For SOUS VIDE, we use an BC framework for image/state feedback, which we will describe next.

## IV. MPC EXPERT AND DATA SYNTHESIS

SOUS VIDE generates visuomotor policies in two steps. First, it programmatically synthesizes a large dataset of demonstrations from an MPC expert policy with privileged state information using our simulator, FiGS. Then, it distills these demonstrations into a policy deployed on the drone.

<!-- image-->  
Fig. 2. Dynamic rollout of 50 data samples. At each time step, the update function $f _ { d }$ simulates the solution from the MPC expert, while the transform $T _ { c } ^ { B }$ is used to extract the corresponding camera image I from the GSplat.

Many drone navigation frameworks are designed around a desired trajectory, whether to encode complex paths [17], [35], [36], race courses over a sequence of gates [2], [3], or even optimal approaches for perching [37]. Given the complexity and variety of motion planning objectives, this abstraction facilitates a decoupled approach where high-level goals can be achieved by a higher-level task planner that generates a desired trajectory for a low-level navigation policy to execute. For instance, if the goal is obstacle avoidance, one could use the already existing GSplat to generate collision-free waypoints [38] that could then be turned into a desired trajectory.

In this work we are interested in the ability to navigate tight spaces and so we handpick a sequence of waypoints that intentionally guides the drone near or through obstacles. From these we use [35] to compute a dynamically feasible spline which we then sample at our desired control frequency $\nu _ { \mathrm { c t l } }$ to extract an $N _ { d } { \mathrm { - s t e p } }$ state and input desired trajectory $( \mathbf { X } ^ { d } , \mathbf { U } ^ { d } )$ , parametrized by Î¸. We can then apply one of a variety of trajectory optimization techniques, such as the oneshot sampling methods in [36] or even an iterative form of DAgger [39], to guide a drone towards the desired trajectory in simulation. We opt for the simplest approach, domain randomization, as described in Algo. 1 and illustrated in Fig. 2. This leverages the strength of FiGS in quickly producing large volumes of photorealistic image data while leaving the door open to more sophisticated techniques.

Given a desired number of samples per time-step $( N _ { s } )$ and rollout duration $( t _ { s } )$ , the demonstration dataset comprises of $N _ { s } \cdot N _ { d }$ dynamic rollout samples, each containing $\nu _ { c t l } \cdot t _ { s }$ timesteps of state $( \mathbf { X } ^ { s } )$ , input $( \mathbf { U } ^ { s } )$ and image $( \pmb { \mathcal { Z } } ^ { s } )$ data for a drone with parameters $\theta _ { s }$ . Each rollout begins by sampling $\theta _ { s }$ and xs0 from a uniform distribution parametrized by $( \theta _ { \mathrm { m i n } } , \theta _ { \mathrm { m a x } } , \Delta x )$ $\theta _ { s }$ is then passed to GenerateDynamics to instantiate the dynamics update function $( f _ { d } )$ encoding (1). This enables us to run MPC, the expert policy which uses privileged information to guide the drone toward $\mathbf { X } ^ { d }$ from $\pmb { x } _ { 0 } ^ { s }$ by solving

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m i n } _ { \pmb { u } } \sum _ { k = 0 } ^ { N - 1 } ( \delta \pmb { x } _ { k } ^ { T } Q _ { k } \delta \pmb { x } _ { k } + \delta \pmb { u } _ { k } ^ { T } R _ { k } \delta \pmb { u } _ { k } ) + \delta \pmb { x } _ { N } ^ { T } Q _ { N } \delta \pmb { x } _ { N } } \\ & { \mathrm { s . t . } \quad \pmb { x } _ { k + 1 } ^ { s } = f _ { d } ( \pmb { x } _ { k } ^ { s } , \pmb { u } _ { k } ^ { s } ) , \quad g _ { c } ( \pmb { u } _ { k } ^ { s } ) \leq \mathbf { 0 } } \end{array} .\tag{2}
$$

in an N-step receding horizon manner, subject to dynamics update $f _ { d }$ and the control limits constraint $g _ { c }$ . The stage-wise weights, $\left( Q _ { k } , R _ { k } \right)$ , and the terminal weights, $Q _ { N }$ , are applied to the difference between the rollout and the closest segment of the desired trajectory, defined by $\delta \mathbf { x } _ { k } = ( \mathbf { x } _ { k } ^ { s } - \mathbf { x } _ { k } ^ { \bar { d } } )$ and similarly for $\delta \mathbf { { u } } _ { k }$ . The resulting state trajectory is then fed into GenerateImages to render first-person-view (FPV) images $( \pmb { \mathcal { Z } } ^ { s } )$ from ${ \mathcal { G S } } _ { \phi }$ using the body to camera transform $( T _ { C } ^ { B } )$

Algorithm 1 FiGS Domain Randomization   
Require: $\mathcal { G S } _ { \phi } , \mathbf { X } ^ { d } , \mathbf { U } ^ { d } , N _ { d } , \theta _ { \mathrm { m i n } } , \theta _ { \mathrm { m a x } } , \Delta \mathbf { x } , N _ { s } , t _ { s } , T _ { \mathcal { C } } ^ { B }$   
1: Initialize dataset $\mathcal { D } = \emptyset$   
2: for $i = 0$ to $N _ { d }$ do   
3: for $j = 0$ to $N _ { s }$ do   
4: $\theta _ { s } \sim U ( \theta _ { \operatorname* { m i n } } , \theta _ { \operatorname* { m a x } } ) , x _ { 0 } ^ { s } \sim ( x _ { i } ^ { d } - \Delta x , x _ { i } ^ { d } + \Delta x )$   
5: fd â GenerateDynamics(Î¸s)   
6: $\mathbf { X } ^ { s } , \mathbf { U } ^ { s } = \mathrm { M P C } ( \boldsymbol { \mathbf { x } } _ { 0 } ^ { s } , f _ { d } , t _ { s } , \mathbf { X } ^ { d } , \mathbf { U } ^ { d } )$   
7: $\pmb { \mathcal { T } } ^ { s } = \mathsf { G e n e r a t e I m a g e } ( \mathbf { X } ^ { s } , T _ { \mathcal { C } } ^ { \mathcal { B } } , \mathcal { G } S _ { \phi } )$   
8: $\mathcal { D }  \mathcal { D } \cup \{ ( \mathbf { X } ^ { s } , \mathbf { U } ^ { s } , \pmb { \mathcal { T } } ^ { s } , \pmb { \theta } _ { s } ) \}$

## V. SV-NET POLICY ARCHITECTURE

Our policy architecture, SV-Net, runs on an Orin Nano onboard the drone at 20 Hz. To output thrust and body rate commands, $\pmb { u } = ( f _ { t h } , \omega )$ , the policy relies solely on onboard data: (i) images from the onboard camera and (ii) height, velocity, and orientation estimates $( p _ { z } , \pmb { v } _ { \mathcal { W } } , \pmb { q } _ { B \mathcal { W } } )$ provided by an Extended Kalman Filter (EKF), which fuses data from an IMU, a downward-facing time-of-flight sensor, and an optical flow sensor. These inexpensive, compact sensors are common on hobby-grade drones, providing state estimates that, while not pinpoint precise, are useful for controlâespecially since most height-sensitive applications occur over reasonably level surfaces. Notably, SV-Net performs better with $( p _ { z } , \pmb { v } _ { \mathcal { W } } )$ even when overflying obstacles, than without them. Beyond serving as direct inputs to SV-Net, these estimates, along with timestamps, are used to compute the history data:

$$
\begin{array} { r l r } { \delta t ^ { k - 1 } = t ^ { k } - t ^ { k - 1 } , } & { } & { \delta p _ { \mathcal { W } } ^ { k - 1 } = \delta t ^ { k - 1 } \cdot \pmb { v } _ { \mathcal { W } } ^ { k } , } \\ { \delta \pmb { v } _ { \mathcal { W } } ^ { k - 1 } = \pmb { v } _ { \mathcal { W } } ^ { k } - \pmb { v } _ { \mathcal { W } } ^ { k - 1 } , } & { } & { \delta \pmb { q } _ { \mathcal { B W } } ^ { k - 1 } = \pmb { q } _ { \mathcal { W B } } ^ { k } \cdot \pmb { q } _ { \mathcal { B W } } ^ { k - 1 } . } \end{array}\tag{3}
$$

We use ${ \pmb v } _ { \mathcal { W } }$ to infer $\delta p$ as the drone cannot observe its lateral position. For brevity, we use superscript time indices.

SV-Net comprises three components: a feature extractor, a history network and a command network (Fig. 3). The architecture uses SqueezeNet [12] as a vision encoder, augmenting its output with estimated height and orientation before passing it through an MLP to create a pose-aware feature extractor. The history network, inspired by RMA, uses the sliding time-step window of history data to generate a latent vector encoding the evolving flight dynamics of the drone at that instant. The policy ingests the latent vector to adapt its output to current flight conditions. The command network combines the outputs of the feature extractor and history network with the observable states and an objective vector, $\mathcal { O } ^ { k }$ , which encodes the change in position, initial and final velocity, initial and final orientation (quaternion), and total trajectory time. We use this to facilitate training and deployment across different trajectories when a single SV-Net is encoded with multiple trajectories (Section VI-C).

<!-- image-->  
Fig. 3. SV-Net consists of three components: a feature extractor that processes visual information from color images, a history network that uses an RMA technique to adapt to variations in dynamics through a history of observable states, and a command network that integrates the outputs of these components with observable states to generate body-rate commands.

We train SV-Net on the demonstration dataset D in two stages. In the first stage, we train the history network to estimate $\theta _ { s }$ given history data extracted from $\mathbf { X } ^ { s } , \mathbf { U } ^ { s }$ through (3). Once trained, the history network is frozen and the remaining components of the policy are trained end-to-end (including the SqueezeNet image encoder) to predict the body rate commands (Us) given the observable states within $\mathbf { X } ^ { s }$ and the images $( \pmb { \mathcal { Z } } ^ { s } )$ . In hardware testing, we find the best performance is achieved by using the second-to-last layer of the history network as input to the command network MLP, rather than the explicit estimate of Î¸.

## A. Analysis of RMA Module

A property of (1) is that if we allow the drone parameters, $k _ { t h }$ and $m ,$ to be variables that can be adjusted online, we can use them to compensate for a wide range of model inaccuracies that are not limited to the thrust and weight of the drone. For simplicity, let $\begin{array} { r } { c = \frac { k _ { t h } } { m } } \end{array}$ . Given an additional force vector $f _ { a d d }$ in the world frame, to account not only for model inaccuracies within c but also for external forces such as aerodynamic drag and ground effect, we can compute an equivalent cË in an augmented form of the velocity equation in (1),

$$
g z _ { \mathcal { W } } - \hat { c } f _ { t h } z _ { B } = g z _ { \mathcal { W } } - c f _ { t h } z _ { B } + f _ { a d d } .\tag{4}
$$

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 4. Clockwise from top left: 1) Desired trajectory in the sceneâs GSplat with corresponding real-world First-Person-View (FPV) of key objects. 2) Drone hardware and frames (W, B, C). We use an Orin Nano and PixRacer Pro for control, while sensing is handled by the PixRacerâs IMU, an ARK Flow sensor, and the D435âs monocular camera. Motion capture markers provide ground truth. 3) 3D position and velocity performance of the policies in Section VI-A.

Then, taking the least-squares estimate of cË, we get:

$$
\operatorname* { m i n } _ { \hat { c } } | | ( { \hat { c } } - c ) f _ { t h } z _ { \boldsymbol { \{ \mathcal { B } } }  + f _ { a d d } | | ^ { 2 } \ \Rightarrow \ { \hat { c } } \approx c - \frac { z _ { \boldsymbol { \mathcal { B } } } ^ { T } f _ { a d d } } { f _ { t h } } .\tag{5}
$$

As evidenced in (5), the capacity of cË to accurately approximate additional forces hinges on near-collinearity of $_ { z _ { B } }$ and $f _ { a d d }$ . This constraint is acceptable for most drone applications. For instance: (i) by definition, thrust-related additional forces align with the $_ { z _ { B } }$ axis, (ii) much of the droneâs operational envelope is near-hover, where $_ { z _ { B } }$ aligns closely with primarily vertical forces, such as those due to changes in mass and ground effect, and (iii) at higher speeds, aerodynamic drag aligns with $z _ { B } ,$ as the motor thrust vector must follow the flight direction. Hence, the RMA module can account for variations in flight dynamics the drone encounters during flight.

## VI. EXPERIMENTS

In this section, we evaluate our SOUS VIDE policies across three fronts: efficacy of the proposed policy architecture, robustness to dynamic and visual disturbances, and generalization to novel scenarios. We demonstrate that the SV-Net policy, equipped with the RMA module, achieves stateof-the-art performance in zero-shot sim-to-real transfer. We emphasize that in all experiments, the policy does not observe the lateral position $( p _ { x } , p _ { y } )$ . However, it does observe $p _ { z }$ through the onboard time-of-flight sensor input.

We perform all experiments using a quadrotor drone equipped with a PixRacer Pro for low-level body-rate tracking control and an Orin Nano for policy execution, as shown in Fig. 4. The onboard sensing suite consists of an IMU, an ARK Flow sensor, and a monocular camera, with the first two fused via an EKF. The motion capture markers visible in our images and videos are used for diagnostics, enabling trajectory plotting in comparison to the ground truth.

To evaluate performance, we consider four key metrics. Completion: Categorizes trajectories along a discrete spectrumâ (ââ) indicates a fully successful position and orientation tracking with no collisions, (â) allows for minor collisions with successful recovery, (â¼) signifies completion of the position component but not the orientation, (â) denotes failure due to an unrecovered collision, and (ââ) corresponds to failure due to drifting off-course. Collision Rate (CR): Quantifies the number of collisions per meter traveled. Trajectory Tracking Error (TTE): Measures the â2-norm of the position error relative to the closest point in the desired trajectory. Finally, Proximity Percentile (PP): Represents the fraction of the trajectory that remains within 30 cm of the intended path. Together, these metrics provide a comprehensive evaluation of trajectory accuracy, robustness, and recovery behavior.

## A. Policy Architecture Ablations

We evaluated our main policy versus three ablations on a 15-second trajectory that guides the drone through a gate and under a ladder before ending facing a monitor. The desired trajectory is visualized in the GSplat in Fig. 4, along with flights from each policy ablation. All policies were trained on the same expert MPC dataset (180k observation-action pairs) using PyTorch, the Adam optimizer (learning rate 1e-4), for approximately 12 hours on a desktop machine (i9-13900K, RTX 4090, 64GB RAM). The policy ablations are:

â¢ SV-Net: Our proposed architecture with a locked pretrained RMA network and 2nd-to-last layer latent code to the command network.

â¢ SV no RMA: A minimal variant comprising only the feature extractor and command network. This serves as our approximation of a zero-shot transfer counterpart to the few-shot transfer described in [17].

â¢ SV no pre-train: A variant of SV-Net that skips the RMA network pre-training and goes directly to training the entire network (with the history network unlocked).

â¢ SV no latent: Same as SV-Net, but uses the RMAâs explicit estimate of Î¸ instead of the 2nd-to-last layer.

<table><tr><td rowspan="2">Ablation Experiments</td><td colspan="5">Completion</td><td rowspan="2">CR (c/m)</td><td rowspan="2">TTE (m)</td><td rowspan="2">PP (%)</td></tr><tr><td>vv</td><td>v</td><td>~</td><td>Ã</td><td>xx</td></tr><tr><td>SV-Net (ours)</td><td>5/5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.00</td><td>0.17</td><td>96.0</td></tr><tr><td>SV no RMA</td><td>3/5</td><td>-</td><td>1/5</td><td>-</td><td>1/5</td><td>0.00</td><td>0.65</td><td>37.7</td></tr><tr><td>SV no pre-train</td><td>-</td><td>-</td><td>1/5</td><td>4/5</td><td>-</td><td>0.10</td><td>0.39</td><td>51.4</td></tr><tr><td>SV no latent</td><td>1/5</td><td>2/5</td><td>1/5</td><td>1/5</td><td>-</td><td>0.05</td><td>0.43</td><td>42.0</td></tr></table>

TABLE I

TABLE COMPARING PERFORMANCE OF ABLATIONS OF SV-NET  
<!-- image-->  
Fig. 5. SV-Net history networkâs estimate of cË with mean $\mu _ { \hat { c } }$ overlaid for Section VI-A flights in simulation (left) and real-world (right).

As shown in Table I, SV-Net outperformed all other architectures, achieving a success rate of 100% with no collisions, a TTE of 0.17m and a PP of 96%, more than doubling the performance of SV no RMA.

To study the history networkâs performance, we acquired a ground truth estimate of $c = 6 . 0 3$ by measuring the mass of the drone and recording the throttle command at hover. We found that when pre-trained (SV-Net and SV no latent), the RMA module maintained an estimated cË value that stayed close to this in both simulation and real-world flights. SV-Net demonstrated the least deviation, with real-world $\mu _ { \hat { c } } = 6 . 0 5 , \sigma _ { \hat { c } } =$ 0.25 (illustrated in Fig. 5). In contrast, SV no latentâs estimate is more unstable with $\mu _ { \hat { c } } ~ = ~ 6 . 3 9 , \sigma _ { \hat { c } } ~ = ~ 1 . 0 7$ across its five flights. We hypothesize that using the 2nd-to-last layer of the history network improves performance as its higherdimensional latent code outweighs the minor information loss from skipping the final layer. Consequently, SV no latent suffers from a feedback loop, where poor estimates degrade policy performance, further amplifying estimation errors. We also note that SV no pre-train, which does have a history network but is instead trained directly on control commands, exhibits a highly unstable signal with $( \mu = - 2 . 8 1 , \sigma = 9 . 5 6 )$

Lastly, we observe that using larger datasets, while cheap to synthesize, offers little performance gain while increasing the training time.

## B. Robustness Experiments

Using the SV-Net result from Section VI-A as a baseline, we conduct five additional experiments, each introducing a distinct disturbance (illustrated in Fig. 6):

â¢ Lighting: Scene brightness was reduced to 40% of original lumens.

â¢ Dynamic: Four people actively moving within the field of view along the entire trajectory.

â¢ Static: The gate, ladder, and monitor (present at train time) were removed at runtime while the pillars adjacent to the gate were occluded with white cloth.

â¢ Payload: A rigid 350g payload (30% increase in drone weight) was attached below the center-of mass.

â¢ Wind: The drone was exposed to a 40 m/s wind gust using a leaf blower.

<!-- image-->

<!-- image-->

Fig. 6. Visualization of disturbances and the corresponding position and velocity performance of SV-Net. Lighting: illumination reduced by 60%, Dynamic: human activity in the scene, Static: key objects in training removed at runtime, Payload: 30% increase in mass, Wind: 40 m/s gust from leaf blower. SV-Net maintains adequate performance in all cases.
<table><tr><td rowspan="2">Robustness Experiments</td><td colspan="5">Completion</td><td rowspan="2">CR (c/m)</td><td rowspan="2">TTE (m)</td><td rowspan="2">PP (%)</td></tr><tr><td>vv</td><td>v</td><td>~</td><td>x</td><td>xx</td></tr><tr><td>Baseline</td><td>5/5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.00</td><td>0.17</td><td>96.0</td></tr><tr><td>Lighting</td><td>4/5</td><td>-</td><td>1/5</td><td></td><td>-</td><td>0.00</td><td>0.24</td><td>64.2</td></tr><tr><td>Dynamic</td><td>5/5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.00</td><td>0.19</td><td>86.3</td></tr><tr><td>Static</td><td>-</td><td>-</td><td>4/5</td><td>-</td><td>1/5</td><td>0.06</td><td>0.49</td><td>25.3</td></tr><tr><td>Payload</td><td>5/5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.00</td><td>0.20</td><td>80.1</td></tr><tr><td>Wind</td><td>5/5</td><td></td><td>-</td><td></td><td>-</td><td>0.00</td><td>0.20</td><td>81.7</td></tr></table>

TABLE II

TABLE PRESENTING SV-NET PERFORMANCE UNDER DISTURBANCES.  
<!-- image-->  
Fig. 7. SV-Net estimate of cË with mean ÂµcË overlaid for Section VI-B flight: payload (left) and wind (right). Wind disturbance region highlighted in green.

The results in Table II show SV-Net consistently demonstrated resilience to dynamic disturbances, payload variations, and wind gusts, maintaining near-baseline performance with minimal impact across all metrics.

When lighting was degraded, the policy struggled to distinguish dark objects from the dim background, particularly near the end, where it drifted away from keeping the (black) monitor in frame. We also tested the policy with less than 40% of the original lumens and the policy consistently failed by drifting off-course from the start location. While the policy handled dynamic scene changes with ease, static changes posed the greatest challenge: i) it underflew waypoints and experienced minor collisions with the occluded pillars, and (ii) it consistently flew through the space where the ladder rungs would have been. Despite these difficulties, the policy reliably tracked the overall trajectory shape, recovered from collisions, and successfully reached the final position in 4 out of 5 flights.

These results suggest that the policy is able to retain essential scene information that would otherwise be lost in approaches relying on visual abstractions.

As shown in Fig. 7, the RMA module maintains a stable cË under wind and payload disturbances, performing nearly identically to the baseline. In the wind disturbance flight, we see a downward spike in cË that correlates to when the drone passes the leafblower (which is effecting a positive $f _ { a d d }$ on the drone). Interestingly, the estimated cË during the payload flight is perceptibly different from the ground truth estimate updated with the additional mass $( c = 4 . 6 2 )$ . Given its overall trajectory performance, we believe the RMA module is in fact compensating for inaccuracies in the thrust model in (1), itself a simplified approximation of rotor dynamics.

## C. Skill-Testing Experiments

In our last set of experiments, we trained three different SV-Net policies, one for each of the following novel scenarios:

â¢ Multi-Objective: One policy was trained to execute three distinct trajectories within the same scene distinguished by unique objective inputs $O ^ { k }$ . Two trajectories used identical positional splines but traversing in opposite directions, while the third followed a climbing orbit.

â¢ Extended Trajectory: The drone navigated a trajectory that is double the length and duration of the trajectory in previous sections.

â¢ Cluttered Environment: The policy was deployed close to the ground in a highly cluttered workshop with obstacles spaced as close as 1.0 m apart.

<!-- image-->

Extended Trajectory  
<!-- image-->

<!-- image-->

Cluttered Environment  
<!-- image-->  
Fig. 8. Position and velocity plots for the Multi-Objective (top) and Extended Trajectory (middle) experiments, with the latterâs desired trajectory in its GSplat. We also show a time-lapse of a Cluttered Environment flight (bottom).

<table><tr><td rowspan="2">Skill-Testing Experiments</td><td colspan="5">Completion</td><td rowspan="2">CR (c/m)</td><td rowspan="2">TTE (m)</td><td rowspan="2">PP (%)</td></tr><tr><td>vv</td><td>v</td><td>â¼</td><td>Ã</td><td>xx</td></tr><tr><td>Multi-Objective</td><td>15/15</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.00</td><td>0.20</td><td>83.4</td></tr><tr><td>Extended Trajectory</td><td>23/30</td><td>4/30</td><td></td><td>3/30</td><td>-</td><td>0.02</td><td>0.24</td><td>72.5</td></tr><tr><td>Cluttered Environment</td><td>13/15</td><td>1/15</td><td>-</td><td>-</td><td>1/15</td><td>0.01</td><td>n/a</td><td>n/a</td></tr></table>

TABLE III

TABLE PRESENTING SV-NET PERFORMANCE OVER NOVEL TRAJECTORIES.  
<!-- image-->

<!-- image-->  
--desired successful failed  
Fig. 9. Quaternion rotation from the IMU during Cluttered Environment flights. The orange outlier trajectory is from the single failed flight.

Visualizations are shown in Fig. 8, with results reported in Table III. The multi-objective policy had mixed success, indicating the need for more robust objective encodings in future work. Though it achieved a 100% collision-free success rate, we observed significant degradation in one of the three tasks, where the drone consistently under-flew the desired trajectory and overshot its end-point In the extended trajectory, the policy performed comparably to the baseline in Section VI-B, with a low CR of 0.02 c/m, TTE of 0.24 m and a PP of 72.5% over 30 attempts. Finally, in the cluttered environment, the policy achieved a 93.3% success rate on a 20 s trajectory through a visually complex scene, demonstrating its robustness in real-world, unstructured environments. Exactly because it is an unstructured environment, there is no motion capture system available for measuring TTE and PP. Instead, we present a time-lapse (bottom of Fig. 8) and the orientation reported by the onboard IMU (Fig. 9).

## VII. CONCLUSIONS

This work introduces the SOUS VIDE approach for training end-to-end visual drone navigation policies. SOUS VIDE comprises the FiGS simulator based on a Gaussian Splat scene model, data generation from a simulated MPC expert, and distillation into a lightweight visuomotor policy architecture. By coupling high-fidelity visual data synthesis with online adaptation mechanisms, SOUS VIDE achieves zero-shot simto-real transfer, demonstrating robustness to variations in mass, thrust, lighting, and dynamic scene changes. Our experiments underscore the policyâs ability to generalize across diverse scenarios, including complex and extended trajectories, with graceful degradation under extreme conditions. Notably, the integration of a streamlined adaptation module enables the policy to overcome limitations of prior visuomotor approaches, offering a computationally efficient yet effective solution for addressing model inaccuracies. These findings highlight the potential of SOUS VIDE as a foundation for future advancements in autonomous drone navigation.

Limitations and Future Work: While its robustness and versatility are evident, challenges such as inconsistent performance in multi-objective tasks suggest opportunities for improvement through more sophisticated objective encodings.

SOUS VIDE has been used to train policies that are highly optimized for a single real-life environment. Future work will explore training policies with the same tools across multiple environments in FiGS to enable generalist skills, like general collision avoidance, and scene-agnostic navigation. We will also explore augmenting SOUS VIDE policies with semantic goal understanding, so goals can be given by a human operator in the form of natural language commands. Ultimately, this work paves the way for deploying learned visuomotor policies in real-world applications, bridging the gap between simulation and practical autonomy in drone operations.

## REFERENCES

[1] A. Loquercio, E. Kaufmann, R. Ranftl, M. Muller, V. Koltun, and Â¨ D. Scaramuzza, âLearning high-speed flight in the wild,â Science Robotics, vol. 6, no. 59, p. eabg5810, 2021.

[2] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Muller, V. Koltun, and Â¨ D. Scaramuzza, âChampion-level drone racing using deep reinforcement learning,â Nature, vol. 620, no. 7976, pp. 982â987, 2023.

[3] I. Geles, L. Bauersfeld, A. Romero, J. Xing, and D. Scaramuzza, âDemonstrating agile flight from pixels without state estimation,â in Robotics: Science and Systems, 2024.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[5] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, J. Kerr, T. Wang, A. Kristoffersen, J. Austin, K. Salahi, A. Ahuja, D. McAllister, and A. Kanazawa, âNerfstudio: A modular framework for neural radiance field development,â in ACM SIGGRAPH 2023 Conference Proceedings, ser. SIGGRAPH â23, 2023.

[6] S. Shah, D. Dey, C. Lovett, and A. Kapoor, âAirsim: High-fidelity visual and physical simulation for autonomous vehicles,â in Field and Service Robotics: Results of the 11th International Conference. Springer, 2018, pp. 621â635.

[7] W. Guerra, E. Tal, V. Murali, G. Ryou, and S. Karaman, âFlightgoggles: Photorealistic sensor simulation for perception-driven robotics using photogrammetry and virtual reality,â in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019, pp. 6941â 6948.

[8] Y. Song, S. Naji, E. Kaufmann, A. Loquercio, and D. Scaramuzza, âFlightmare: A flexible quadrotor simulator,â in Conference on Robot Learning. PMLR, 2021, pp. 1147â1157.

[9] J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, and A. P. Schoellig, âLearning to flyâa gym environment with pybullet physics for reinforcement learning of multi-agent quadcopter control,â in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021, pp. 7512â7519.

[10] M. Jacinto, J. Pinto, J. Patrikar, J. Keller, R. Cunha, S. Scherer, and A. Pascoal, âPegasus simulator: An isaac sim framework for multiple aerial vehicles simulation,â in 2024 International Conference on Unmanned Aircraft Systems (ICUAS), 2024, pp. 917â922.

[11] B. Xu, F. Gao, C. Yu, R. Zhang, Y. Wu, and Y. Wang, âOmnidrones: An efficient and flexible platform for reinforcement learning in drone control,â 2023.

[12] F. N. Iandola, âSqueezenet: Alexnet-level accuracy with 50x fewer parameters andÂ¡ 0.5 mb model size,â arXiv preprint arXiv:1602.07360, 2016.

[13] A. Kumar, Z. Fu, D. Pathak, and J. Malik, âRma: Rapid motor adaptation for legged robots,â arXiv preprint arXiv:2107.04034, 2021.

[14] A. Zhou, M. J. Kim, L. Wang, P. Florence, and C. Finn, âNerf in the palm of your hand: Corrective augmentation for robotics via novel-view synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 907â17 917.

[15] M. N. Qureshi, S. Garg, F. Yandun, D. Held, G. Kantor, and A. Silwal, âSplatsim: Zero-shot sim2real transfer of rgb manipulation policies using gaussian splatting,â arXiv preprint arXiv:2409.10161, 2024.

[16] T. Haarnoja, B. Moran, G. Lever, S. H. Huang, D. Tirumala, J. Humplik, M. Wulfmeier, S. Tunyasuvunakool, N. Y. Siegel, R. Hafner et al., âLearning agile soccer skills for a bipedal robot with deep reinforcement learning,â Science Robotics, vol. 9, no. 89, p. eadi8022, 2024.

[17] A. Tagliabue and J. P. How, âTube-nerf: Efficient imitation learning of visuomotor policies from mpc via tube-guided data augmentation and nerfs,â IEEE Robotics and Automation Letters, 2024.

[18] A. Quach, M. Chahine, A. Amini, R. Hasani, and D. Rus, âGaussian splatting to real world flight navigation transfer with liquid networks,â in Proc. of the Conference on Robot Learning (CoRL), 2024.

[19] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, âDomain randomization for transferring deep neural networks from simulation to the real world,â in 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2017, pp. 23â30.

[20] G. M. Hoffmann, H. Huang, S. L. Waslander, and C. J. Tomlin, âPrecision flight control for a multi-vehicle quadrotor helicopter testbed,â Control engineering practice, vol. 19, no. 9, pp. 1023â1036, 2011.

[21] M. Faessler, A. Franchi, and D. Scaramuzza, âDifferential flatness of quadrotor dynamics subject to rotor drag for accurate tracking of highspeed trajectories,â IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 620â626, 2017.

[22] E. Tal and S. Karaman, âAccurate tracking of aggressive quadrotor trajectories using incremental nonlinear dynamic inversion and differential flatness,â IEEE Transactions on Control Systems Technology, vol. 29, no. 3, pp. 1203â1218, 2020.

[23] A. Bhattacharya, N. Rao, D. Parikh, P. Kunapuli, N. Matni, and V. Kumar, âVision transformers for end-to-end vision-based quadrotor obstacle avoidance,â arXiv preprint arXiv:2405.10391, 2024.

[24] G. Loianno, C. Brunner, G. McGrath, and V. Kumar, âEstimation, control, and planning for aggressive flight with a small quadrotor with a single camera and imu,â IEEE Robotics and Automation Letters, vol. 2, no. 2, pp. 404â411, 2016.

[25] D. Zhang, A. Loquercio, X. Wu, A. Kumar, J. Malik, and M. W. Mueller, âLearning a single near-hover position controller for vastly different quadcopters,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 1263â1269.

[26] D. Zhang, A. Loquercio, J. Tang, T.-H. Wang, J. Malik, and M. W. Mueller, âA learning-based quadcopter controller with extreme adaptation,â arXiv preprint arXiv:2409.12949, 2024.

[27] S. Ross, N. Melik-Barkhudarov, K. S. Shankar, A. Wendel, D. Dey, J. A. Bagnell, and M. Hebert, âLearning monocular reactive uav control in cluttered natural environments,â in 2013 IEEE international conference on robotics and automation. IEEE, 2013, pp. 1765â1772.

[28] F. Sadeghi and S. Levine, âCad2rl: Real single-image flight without a single real image,â arXiv preprint arXiv:1611.04201, 2016.

[29] A. Loquercio, A. I. Maqueda, C. R. Del-Blanco, and D. Scaramuzza, âDronet: Learning to fly by driving,â IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 1088â1095, 2018.

[30] D. Gandhi, L. Pinto, and A. Gupta, âLearning to fly by crashing,â in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017, pp. 3948â3955.

[31] D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine, âGnm: A general navigation model to drive any robot,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 7226â7233.

[32] R. Doshi, H. Walke, O. Mees, S. Dasari, and S. Levine, âScaling crossembodied learning: One policy for manipulation, navigation, locomotion and aviation,â arXiv preprint arXiv:2408.11812, 2024.

[33] V. Ye, M. Turkulainen, and the Nerfstudio team, âgsplat.â [Online]. Available: https://github.com/nerfstudio-project/gsplat

[34] R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. van Duijkeren, A. Zanelli, B. Novoselnik, T. Albin, R. Quirynen, and M. Diehl, âacados â a modular open-source framework for fast embedded optimal control,â Mathematical Programming Computation, 2021.

[35] D. Mellinger and V. Kumar, âMinimum snap trajectory generation and control for quadrotors,â in 2011 IEEE international conference on robotics and automation. IEEE, 2011, pp. 2520â2525.

[36] A. Tagliabue, D.-K. Kim, M. Everett, and J. P. How, âEfficient guided policy search via imitation of robust tube mpc,â in 2022 International Conference on Robotics and Automation. IEEE, 2022, pp. 462â468.

[37] J. Thomas, G. Loianno, M. Pope, E. W. Hawkes, M. A. Estrada, H. Jiang, M. R. Cutkosky, and V. Kumar, âPlanning and control of aggressive maneuvers for perching on inclined and vertical surfaces,â in International Design Engineering Technical Conferences and Computers and Information in Engineering Conference, vol. 57144. American Society of Mechanical Engineers, 2015, p. V05CT08A012.

[38] T. Chen, O. Shorinwa, J. Bruno, J. Yu, W. Zeng, K. Nagami, P. Dames, and M. Schwager, âSplat-nav: Safe real-time robot navigation in gaussian splatting maps,â arXiv preprint arXiv:2403.02751, 2024.

[39] S. Ross, G. Gordon, and D. Bagnell, âA reduction of imitation learning and structured prediction to no-regret online learning,â in Proceedings of the fourteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2011, pp. 627â635.