# ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation with Moving Obstacles

Seungyeon Yoo, Youngseok Jang, Dabin Kim, Youngsoo Han, Seungwoo Jung, and H. Jin Kim

Abstractâ Visual navigation models often struggle in realworld dynamic environments due to limited robustness to the sim-to-real gap and the difficulty of training policies tailored to target deployment environments (e.g., households, restaurants, and factories). Although real-to-sim navigation simulation using 3D Gaussian Splatting (GS) can mitigate these challenges, prior GS-based works have considered only static scenes or non-photorealistic human obstacles built from simulator assets, despite the importance of safe navigation in dynamic environments. To address these issues, we propose ReaDy-Go, a novel real-to-sim simulation pipeline that synthesizes photorealistic dynamic scenarios in target environments by augmenting a reconstructed static GS scene with dynamic human GS obstacles, and trains navigation policies using the generated datasets. The pipeline provides three key contributions: (1) a dynamic GS simulator that integrates static scene GS with a human animation module, enabling the insertion of animatable human GS avatars and the synthesis of plausible human motions from 2D trajectories, (2) a navigation dataset generation framework that leverages the simulator along with a robot expert planner designed for dynamic GS representations and a human planner, and (3) robust navigation policies to both the sim-to-real gap and moving obstacles. The proposed simulator generates thousands of photorealistic navigation scenarios with animatable human GS avatars from arbitrary viewpoints. ReaDy-Go outperforms baselines across target environments in both simulation and real-world experiments, demonstrating improved navigation performance even after sim-to-real transfer and in the presence of moving obstacles. Moreover, zero-shot sim-to-real deployment in an unseen environment indicates its generalization potential. Project page: https://syeon-yoo.github.io/ready-go-site/.

## I. INTRODUCTION

Visual navigation policies that rely solely on an RGB camera provide practical advantages for robotic systems, including reduced hardware complexity, lower sensor costs, and lighter payloads. Furthermore, RGB-only navigation modules can be seamlessly integrated with other vision-based perception and decision-making components. Despite these benefits, their performance in real-world dynamic environments often lacks robustness and efficiency, limiting reliable deployment in practical settings.

The fundamental challenge lies in achieving robust navigation in environment-specific, dynamic real-world settings. RGB-only navigation models typically learn nonlinear visuomotor policies from high-dimensional monocular observations, where depth ambiguity complicates scene understanding. Therefore, most learning-based approaches are trained extensively in simulation, as collecting large-scale real-world navigation data is impractical. However, the resulting simto-real distribution gap significantly degrades performance during deployment. Moreover, robots are commonly operated in environment-specific settings such as households, restaurants, or factories, where scene layouts are unique. General navigation models [1]â[3] often fail to fully exploit environment-specific characteristics, which leads to reduced success rates [4], [5], while constructing digital twins for targeted dataset generation remains expensive [6].

<!-- image-->  
Fig. 1: The proposed real-to-sim dynamic environment simulation pipeline for visual navigation. ReaDy-Go generates photorealistic navigation datasets for dynamic scenarios and trains environment-specific visual navigation policies from these datasets. The resulting policies demonstrate robustness to the sim-to-real gap and moving obstacles.

To mitigate these limitations, recent works [4], [7]â[12] have proposed real-to-sim simulation pipelines based on 3D Gaussian Splatting (GS) [13]. By reconstructing environments from RGB videos, GS enables high-fidelity rendering at fast frame rates, novel view synthesis, and simulation with an explicit 3D scene representation. GS-based pipelines leverage these advantages for navigation dataset generation and policy training to reduce the sim-to-real gap in target environments. However, existing GS-based navigation frameworks are limited to static scenes or introduce dynamic elements using non-photorealistic simulation assets, such as pre-defined human meshes. Such limitations make it difficult to learn safe navigation in the presence of dynamic obstacles and to render photorealistic human appearances within reconstructed real-world environments. As a result, the generation of photorealistic navigation datasets for dynamic environments remains underexplored.

Addressing this issue is essential for robot deployment in real-world environments with moving elements. In particular, there is currently no GS-based real-to-sim pipeline that jointly (1) models animatable human GS obstacles within reconstructed static GS scenes, (2) generates navigation datasets tailored to environment-specific training in dynamic settings, and (3) trains visual navigation policies that are robust under sim-to-real transfer and to moving obstacles.

Motivated by these limitations, we propose ReaDy-Go, a photorealistic Real-to-Sim Dynamic 3D Gaussian Splatting Simulation pipeline for environment-specific RGB-only visual navigation with moving obstacles (Fig. 1). The proposed framework consists of three key components: (1) a dynamic GS simulator that integrates a static scene GS, an animatable human GS obstacle, and a human motion generation module, enabling the placement of a human in the scene and the synthesis of plausible motions conditioned on 2D trajectories; (2) a photorealistic dataset generation pipeline for dynamic environments that is composed of the dynamic simulator, a robot expert planner designed for dynamic GS representations, and a human planner; and (3) training an RGB-only visual navigation policy using imitation learning with the generated datasets. We focus on goal navigation as the downstream task, as it is a fundamental building block extensible to broader navigation problems.

To the best of our knowledge, ReaDy-Go is the first GSbased real-to-sim framework for dynamic scenes that enables photorealistic dataset generation with animated human GS obstacles and environment-specific navigation policy training. Our contributions are threefold.

â¢ Dynamic GS Simulator: We develop a photorealistic real-to-sim dynamic 3D Gaussian Splatting simulator with human GS obstacles. The simulator represents both the scene and the human as GS representations and generates human motion, i.e., body root locations and joint configurations, from a given 2D trajectory.

â¢ Photorealistic Dynamic Dataset Generation Pipeline: We propose a new pipeline, ReaDy-Go, that generates photorealistic dynamic scenario datasets and trains environment-specific visual navigation policies, without requiring mesh extraction and physics engine integration. The pipeline incorporates the dynamic GS simulator, a robot expert planner designed for dynamic GS representations, and a human planner.

â¢ Robust Sim-to-Real Performance in Dynamic Environments: The proposed method achieves robust RGB-only visual navigation performance in dynamic environments in both simulation and sim-to-real transfer experiments. Furthermore, it shows generalization via zero-shot simto-real deployment in an unseen environment.

## II. RELATED WORK

## A. RGB-Only Visual Navigation Policies

RGB-only navigation often uses learning-based methods to learn nonlinear visuomotor policies that map highdimensional observations to actions while mitigating monocular depth ambiguity. Since learning-based navigation policies usually require large amounts of data, previous works have used simulation datasets to train policies by imitation learning or reinforcement learning [5], [14]â[18]. To leverage the fact that simulation can easily capture multi-modal data, cross-modal learning that uses heterogeneous data during training while employing only RGB for inference has also been proposed to distill multi-modal information into RGBonly policies [19], [20]. Despite these advances, policies trained in simulation often suffer from sim-to-real performance degradation. Techniques such as domain randomization [15], [16] may not fully eliminate distribution gaps.

An alternative direction to address the sim-to-real gap for visual navigation policies is to train General Navigation Models (GNMs) on large-scale real-world navigation datasets [1]â[3]. GNMs show the potential to handle diverse robot embodiments and environments in a zero-shot manner using a single model. However, they typically require a prebuilt topological map (e.g., sequences of goal images) during inference and may underperform compared to environmentspecific policies in target deployment settings [4], [5], as they cannot fully exploit scene-specific structural information.

To address these limitations, real-to-sim simulation approaches reconstruct target environments and generate photorealistic datasets tailored for environment-specific policy training. Scene reconstructions can be achieved using asset retrieval [21], [22] or 3D Gaussian Splatting, which provides explicit scene representations for rendering and planning.

## B. GS-Based Real-to-Sim Simulation for Navigation Policies

Real-to-sim simulation based on 3D Gaussian Splatting (GS) has recently gained attention in robotics for reducing the sim-to-real gap in both navigation [4], [7]â[12] and manipulation tasks [23]â[25]. GS offers high-fidelity rendering at fast frame rates, novel view synthesis, and geometrically interpretable primitives that are useful for simulation.

For navigation, the first type of work is rule-based planning methods using GS maps to exploit their geometric consistency and rendering quality [26], [27]. While effective in static environments, these methods require access to the GS map at inference time and are not well suited for resourceconstrained robotic platforms. In the second type of work, learning-based navigation policies are trained in GS-based real-to-sim environments and transferred to actual robot hardware such as drones [8]â[10], [12], wheeled robots [4], [11], and legged robots [7]. These approaches demonstrate improved sim-to-real transfer by leveraging photorealistic reconstructions of deployment environments.

Despite this advantage, existing GS-based real-to-sim navigation frameworks predominantly assume static scenes, whereas dynamic environments are essential for real-world applications. Moreover, some works simplify tasks by using predefined trajectories or strong goal cues (e.g., colored targets), reducing the task difficulty. Although Vid2Sim [11] introduces dynamic obstacles via simulation assets, it may limit photorealism compared to fully GS-based pipelines and requires additional mesh extraction and physics engine integration to generate observations from heterogeneous representations (GS and meshes).

<!-- image-->  
Fig. 2: ReaDy-Go overview. The proposed photorealistic simulation pipeline for visual navigation in dynamic environments consists of three main components: (1) a real-to-sim dynamic 3D Gaussian Splatting (GS) simulator with animatable human GS avatars, (2) photorealistic navigation dataset generation for dynamic scenarios, and (3) visual navigation policy training.

In contrast, our method, ReaDy-Go, develops a pipeline for navigation policies that is composed of a photorealistic real-to-sim dynamic GS simulation using GS for both the scene and dynamic human obstacles, together with a robot expert planner and a human planner. We show that ReaDy-Go improves sim-to-real transfer performance of environment-specific navigation policies in dynamic environments and demonstrates policy generalization.

## III. METHOD

To improve RGB-only navigation performance in dynamic environments, ReaDy-Go proposes a cost-effective and scalable pipeline to train environment-specific visual navigation policies using photorealistic datasets with dynamic obstacles in deployment scenes, thereby mitigating the sim-toreal gap. It requires only a single video per environment. Given a video of a static target deployment environment, our pipeline generates photorealistic navigation datasets with moving human obstacles and trains an environment-specific navigation policy, as shown in Fig. 2. The pipeline consists of three main components: (1) a real-to-sim dynamic 3D Gaussian Splatting (GS) simulator, (2) dynamic navigation dataset generation using the simulator and planners, and (3) navigation policy training. Each component is explained in detail in this section.

## A. A Photorealistic Real-to-Sim Dynamic GS Simulator

Given images from a monocular video with corresponding camera poses estimated in metric scale using COLMAP [28] and an ArUco marker in the initial frames, ReaDy-Go builds a dynamic GS simulator by integrating a static GS scene for a target deployment environment with a human animation module that places pre-extracted human GS models in the scene and animates them along desired 2D trajectories. The simulator is the core component for generating dynamic scenarios in the target environment using various human GS models and user-defined 2D trajectories.

1) Static GS scene reconstruction: The background scene in which the robot will be deployed is reconstructed using

GS. GS is a representation that enables 3D geometry reconstruction, high-fidelity novel view synthesis, and fast training and rendering by fitting positions, rotations, scales, opacities, and colors of 3D Gaussian primitives to the training set images [13]. Specifically, we employed PGSR [29] for 3D scene reconstruction, which achieves high-quality surface reconstruction and rendering by compressing 3D Gaussians into flat planes and using geometric regularization loss terms in addition to a photometric loss. Improved geometric accuracy compared to the vanilla GS reduces Gaussian noise in the scene and improves multi-view consistency. This is important when extracting scene voxels and 2D occupancy grids for planners of the robot and dynamic obstacles.

2) Human Animation Module: We set humans as dynamic obstacles in the real-to-sim GS scene. The human animation module places an animatable human GS model in the scene and then generates plausible human motion along a desired obstacle trajectory.

Animatable human GS avatars are extracted from the NeuMan dataset [30] using HUGS [31]. HUGS disentangles a dynamic human and a static scene from a video and parameterizes the human GS in a canonical space initialized with the SMPL body model [32], together with triplane features. Leveraging these parameters, fully connected layers estimate human Gaussian attributes and linear blend skinning (LBS) weights to animate the human GS under novel poses given SMPL joint parameters. The extracted human GS models can be placed, animated, and rendered in novel GS scenes and viewpoints.

Subsequently, to generate natural human motion along desired 2D trajectories of dynamic obstacles, PriorMDM [33] is adopted to predict the body root trajectory and joint angles in the SMPL parameters, which are used to animate human GS models. It enables fine-grained trajectory-level control over human motion using a motion diffusion model (MDM) as a generative prior. Given a 2D trajectory, we convert it into body root linear and rotation velocities, normalize them to match the HumanML3D [34] representation as the model input, and feed them into PriorMDM. The predicted 3D body joint positions are then fitted to SMPL parameters through SMPLify [35] and transformed into the world coordinate frame in the target environment, which makes plausible human animation along a desired trajectory possible.

## B. Navigation Data Generation for Dynamic Environments

To generate photorealistic navigation datasets to mitigate the sim-to-real gap for scenarios in dynamic environments, ReaDy-Go proposes a pipeline that integrates our dynamic GS simulator, a robot expert planner designed for dynamic GS representations, and a human planner. By leveraging the simulator and planners, the pipeline collects RGB observations, actions, and relative goal positions as training samples for a navigation policy. This data generation process does not require onerous procedures such as scene mesh extraction and physics engine integration.

1) Static scene voxelization with opacity filtering for planners: For planning modules, we first voxelize static environments by marking a voxel as occupied if it contains all or part of the 1Ï Gaussian ellipsoid, following [26]. However, reconstructed static GS scenes can contain spurious Gaussians around the ground, although they are suppressed by geometric regularization. Such noise hinders robot and human planning by reducing free space in the scene even though rendered images appear high-quality. For this reason, we filter out noise based on the accumulated opacities in each voxel during scene voxelization. If the sum of Gaussian opacities in a voxel does not exceed an opacity threshold, the voxel is classified as free space. This leads to more accurate free space regions around the ground, especially for weakly textured scenes. Then, the filtered voxel map is converted to 2D occupancy maps for planners by projecting occupancy across height ranges: from near the ground to the robot height for the robot navigable map and from near the ground to the human height for the human walkable map. Occupied grids in each map are inflated by a safety margin, such as the robot or human radius, for safe planning.

2) Robot expert planner for dynamic GS representations: Collecting expert navigation data for policy training requires an expert planning algorithm capable of navigating dynamic GS environments while generating kinematically feasible trajectories for ground vehicles in our setting. However, prior planning algorithms designed for GS representations, such as Splat-Nav [26], are restricted to static environments and rely on quadrotor-specific planning limited to holonomic vehicles. Similarly, GaussNav [27] operates solely in static GS environments with a discrete action space, failing to utilize the full kinematic capabilities of ground vehicles. To overcome these limitations, we design an expert planner that can handle GS-based moving obstacles and generate feasible trajectories for ground vehicles by leveraging the Hybrid Aâ planner [36] augmented with a motion primitive library.

In the proposed planner, the robot navigable map is updated at each timestep with a dynamic obstacle if the human appears within the camera field of view (FOV) and comes close to the robotâs trajectory within a safety margin. To obtain a cleaner set of human Gaussian primitives that better corresponds to the actual human-occupied region for 2D projection, we remove spurious primitives by discarding ellipsoids with high uncertainty, defined as those with a covariance trace exceeding the 75th percentile. We further downsample the remaining primitives based on their mean positions using voxel-grid downsampling. Finally, these valid points are projected onto the 2D robot navigable map, where the human-occupied regions are then inflated to consider the humanâs movement based on a safety buffer derived from a constant velocity prediction over a 2 s horizon.

<!-- image-->

<!-- image-->  
(a) Initial planned path  
(b) Replanned path  
Fig. 3: Visualization of the robot expert planner. (a) The robot follows a collision-free path (red) from start (green) to goal (blue). (b) When a dynamic obstacle (human point cloud in red; inflated region in magenta) makes the path unsafe, the robot follows a replanned path (yellow).

During the Hybrid Aâ expansion phase, each node transitions to its neighbors via a set of discrete motion primitives. These primitives are generated by combining three velocity scales (1/3, 2/3, 1 of maximum velocity) with three steering commands (left, straight, right). The cost for each node is computed as a weighted sum of the primitiveâs arc length, steering penalty, and the heuristic cost-to-go to the goal. The expansion process terminates once a feasible path to the goal is identified. However, because the initial plan relies on short-horizon constant velocity predictions for human motion, the resulting trajectory may become unsafe during execution. To address this, we implement a reactive replanning mechanism. A replan is triggered either when a defined replanning period elapses or immediately if the minimum distance between a dynamic obstacle and the look-ahead segment of the currently executed trajectory falls below a safety margin. Fig. 3 illustrates the resulting path generated by the expert planner, highlighting the replanning behavior. With the proposed expert planner, we can generate safe robot trajectories in dynamic environments and gather expert demonstrations for navigation datasets.

3) Human planner: The human planner generates trajectories for dynamic obstacles using the 2D human walkable map. Firstly, to encourage human-robot interaction-rich scenarios, we sample pairs of human start and goal positions such that the resulting trajectories either cross or run parallel to the line connecting the robotâs start and goal positions. A seed path is computed on the 2D map using a graphbased search algorithm, e.g., Aâ. Then, a smooth trajectory is planned by spline-based optimization. Given this final 2D human trajectory, the human animation module generates SMPL motion parameters that enable a human to walk or run along the trajectory.

4) Data generation: With the planners, we can generate diverse dynamic scenarios in the proposed simulator. With a sufficient number of trials, the scenarios cover most regions in the scene and include various cases of static and dynamic obstacle avoidance. For each scenario, we record a set of tuples consisting of photorealistic RGB observations, corresponding expert outputs as actions (linear velocity v and angular velocity w), and relative goal positions.

## C. Training Visual Navigation Policy

Utilizing generated photorealistic expert navigation datasets, we aim to train end-to-end visual navigation policies using imitation learning to improve robustness to the sim-toreal gap and dynamic obstacles. To focus on validating the effect of ReaDy-Go simulation on navigation performance, we use a relatively simple and lightweight architecture for the navigation policy, as described in Section IV-B.2. The RGB encoder receives three consecutive RGB frames to incorporate ego history and the movement of a dynamic obstacle for safe and efficient obstacle detouring. Three consecutive intermediate latent vectors from the RGB encoder are concatenated with the action history of the previous timestep $( v _ { t - 1 } , w _ { t - 1 } )$ and relative goal positions in the robot local coordinate frame $( \Delta x _ { t , r } , \Delta y _ { t , r } )$ . The action history offers ego movement information, and the relative goal positions are used for goal navigation. The concatenated vector passes through shallow fully connected layers and outputs the action $( \boldsymbol { v } _ { t } , \ \boldsymbol { w } _ { t } )$ . The action outputs are supervised to match the expert actions from the dataset using mean squared error.

## IV. EXPERIMENTS

In this section, we evaluate the visual navigation performance in dynamic environments and the robustness to sim-to-real transfer of ReaDy-Go visual navigation policies trained in simulation. First, the qualitative results of ReaDy-Go simulation are examined to show the photorealistic realto-sim dynamic GS simulation. Then, simulation and realworld experiments are conducted in static and dynamic tasks to compare its effective and robust navigation performance for target deployment environments against baselines. Finally, generalization of ReaDy-Go policies is investigated through navigation experiments in an unseen environment.

## A. Experimental Setup

1) Task description: The task is goal navigation, where a robot should navigate from a start to a goal. For each episode, the start/goal pairs are sampled randomly in the free space of the environment. The robot should reach the goal without collisions within the scenario time limit. In the Static task, the agent should avoid static obstacles in the scene. In the Dynamic task, the agent should also detour around one or two dynamic obstacles in addition to the static obstacles. For each task and environment, we evaluate 100 episodes in simulation and 10 episodes in real-world experiments.

The robot used for real-world experiments was a differential wheeled robot equipped with a forward-facing fixed ZED2 camera. The robot was modeled as a unicycle and used an NVIDIA Jetson Orin NX for on-board inference. The camera obtained image observations for model input and wheel odometry was used to estimate relative goal positions.

2) Evaluation metrics: We evaluate navigation performance using Success Rate (SR) and Average Reaching Time (ART). SR is the proportion of successful scenarios among the total test scenarios, reflecting how the agent safely navigates in environments. A scenario is successful if the robot reaches the goal within 1 m before the maximum scenario length of 50 seconds. ART is the average time to finish scenarios across total scenarios, indicating how the agent effectively navigates to the goals. For failed scenarios, we set the reaching time to the maximum scenario length.

3) Baselines: We compare the following baselines against ReaDy-Go visual navigation policies to evaluate the effect of photorealistic dynamic GS simulation data for target deployment environments.

â¢ Vid2Sim [11] generates real-to-sim navigation data by combining a GS scene, mesh, and a physics engine. Although it also generates dynamic environment scenarios, Vid2Sim uses a human simulation asset from the Unity engine [37] as a dynamic obstacle, which is less photorealistic. To isolate the effect of photorealistic dynamic obstacles on navigation policies, we employ the same policy architecture for both Vid2Sim and ReaDy-Go, differing only in the data generation method.

â¢ GNM [1] is a general navigation model that is trained on large-scale navigation data. It can be deployed in diverse environments and across embodiments in a zeroshot manner. It uses image encoders and fully connected layers for goal-conditioned trajectory prediction.

â¢ ViNT [2] extends GNM with a transformer architecture that improves cross-embodiment generalization and downstream adaptability.

â¢ NoMaD [3] extends ViNT by employing a diffusion model decoder to obtain a highly expressive policy for both goal-conditioned navigation and exploration.

ReaDy-Go and Vid2Sim are trained using datasets of dynamic scenarios and deployed in both Static and Dynamic tasks. For a fair comparison with image-goal navigation baselines (GNM, ViNT, and NoMaD), we provide them goal images captured at goal positions within 10 m of the start, with the camera oriented along the start-to-goal direction, which matches the robotâs initial heading.

## B. Implementation Details

1) Dataset: We selected three target environments, Outside, Lobby, and Library, as shown in Figs. 4 (simulation) and 5 (real-world). Each environment was reconstructed as a GS scene using a monocular video recorded for about six minutes, yielding 1,000â1,500 images. We used six human GS avatars (Section III-A.2) into the navigation datasets. During ReaDy-Go simulation, we generated 400 training episodes for each environment, which corresponds to approximately 80kâ120k data samples. The validation scenarios consist of 50 episodes for each environment, and we selected the checkpoint with the best validation performance, which is used for testing in simulation and the real world.

<!-- image-->

<!-- image-->  
(a) Outside environment

<!-- image-->

<!-- image-->

<!-- image-->  
(b) Lobby environment

<!-- image-->

<!-- image-->

<!-- image-->  
(c) Library environment

<!-- image-->  
Fig. 4: Qualitative novel-view synthesis results from the proposed dynamic GS simulation pipeline across diverse viewpoints and environments. ReaDy-Go generates photorealistic, geometrically consistent dynamic scenarios with natural human motion from novel viewpoints, enabling navigation dataset generation for target deployment environments.

TABLE I: Visual Navigation Performance in Simulation
<table><tr><td rowspan="3">Method</td><td rowspan="3">Params</td><td colspan="4">Outside</td><td colspan="4">Lobby</td><td colspan="4">Library</td></tr><tr><td colspan="2">Static</td><td colspan="2">Dynamic</td><td colspan="2">Static</td><td colspan="2">Dynamic</td><td colspan="2">Static</td><td colspan="2">Dynamic</td></tr><tr><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td></tr><tr><td>Expert</td><td>-</td><td>100%</td><td></td><td>91%</td><td>-</td><td>100%</td><td>-</td><td>83%</td><td>-</td><td>100%</td><td>-</td><td>78%</td><td>-</td></tr><tr><td>GNM</td><td>9M</td><td>50%</td><td>28.69</td><td>43%</td><td>31.89</td><td>50%</td><td>28.12</td><td>50%</td><td>28.65</td><td>74%</td><td>17.01</td><td>58%</td><td>23.48</td></tr><tr><td>NoMaD</td><td>19M</td><td>35%</td><td>35.04</td><td>36%</td><td>34.64</td><td>25%</td><td>38.89</td><td>21%</td><td>40.47</td><td>67%</td><td>19.74</td><td>48%</td><td>27.94</td></tr><tr><td>ViNT</td><td>30M</td><td>50%</td><td>28.72</td><td>36%</td><td>34.48</td><td>60%</td><td>25.31</td><td>55%</td><td>27.82</td><td>75%</td><td>15.85</td><td>62%</td><td>21.72</td></tr><tr><td>Vid2Sim</td><td>2M</td><td>89%</td><td>14.96</td><td>57%</td><td>26.88</td><td>98%</td><td>9.83</td><td>70%</td><td>21.06</td><td>91%</td><td>9.82</td><td>68%</td><td>19.92</td></tr><tr><td>ReaDy-Go</td><td>2M</td><td>90%</td><td>13.43</td><td>78%</td><td>18.68</td><td>98%</td><td>8.93</td><td>78%</td><td>17.59</td><td>86%</td><td>11.08</td><td>80%</td><td>13.98</td></tr></table>

2) Navigation policy training: The navigation policy consists of ten convolutional layers with residual connections, which take an RGB input and output a 20-dimensional encoded vector, and three MLP layers with non-linear activations, which take a 64-dimensional concatenated vector (i.e., encoded vectors from three consecutive RGB input images, the previous action, and the relative goal position). The policy predicts the action output (v, w) and is trained with the Adam optimizer with a learning rate of 10â4. The image resolution is 144 Ã 256, and the time interval between consecutive input images is 0.5 s.

## C. Visualization of ReaDy-Go Dynamic GS Simulation

Qualitative results demonstrate that ReaDy-Go produces photorealistic real-to-sim dynamic environments, as shown in Fig. 4. First, the proposed human animation module generates plausible body motions for human GS avatars within static GS scenes along given 2D trajectories, without relying on a physics engine. Second, the framework supports scalable generation of thousands of photorealistic navigation scenarios via expert and human planners. Finally, novel view synthesis results confirm that geometric consistency can be preserved from arbitrary viewpoints in dynamic settings.

## D. Visual Navigation Performance in Simulation

The impact of photorealistic dynamic GS simulation datasets on visual navigation policies is examined through simulation tests comparing ReaDy-Go against baselines, as summarized in Table I. We observe two key takeaways from the experiments. First, training environment-specific navigation policies using real-to-sim simulation is crucial for safe and efficient navigation. ReaDy-Go and Vid2Sim, both trained in real-to-sim target environments, achieve higher success rates and lower average reaching times than general navigation models (GNM, NoMaD, and ViNT) across all tasks and environments. This performance is particularly notable given that the general models requiring up to 15Ã more parameters. These indicate that real-to-sim simulation with GS is a cost-effective and scalable approach to achieve fewer collisions and faster task completion with only a video.

TABLE II: Real-World Visual Navigation Performance
<table><tr><td rowspan="2">Method</td><td rowspan="2">Params</td><td colspan="4">Outside</td><td colspan="4">Lobby Dynamic</td><td colspan="4">Library</td></tr><tr><td>Static</td><td></td><td colspan="2">Dynamic</td><td></td><td colspan="3">Static</td><td>Static</td><td></td><td>Dynamic</td><td></td></tr><tr><td></td><td></td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td><td>SR â</td><td>ART (s) â</td></tr><tr><td>ViNT</td><td>30M</td><td>50%</td><td>33.54</td><td>30%</td><td>39.22</td><td>60%</td><td>27.83</td><td>20%</td><td>42.34</td><td>80%</td><td>20.60</td><td>40%</td><td>34.96</td></tr><tr><td>Vid2Sim</td><td>2M</td><td>90%</td><td>17.72</td><td>60%</td><td>28.98</td><td>70%</td><td>25.35</td><td>40%</td><td>36.50</td><td>90%</td><td>15.44</td><td>60%</td><td>28.54</td></tr><tr><td>ReaDy-Go</td><td>2M</td><td>100 %</td><td>17.45</td><td>90%</td><td>20.68</td><td>90%</td><td>18.93</td><td>70%</td><td>26.49</td><td>100%</td><td>11.12</td><td>80%</td><td>19.57</td></tr></table>

<!-- image-->  
(a) Outside

<!-- image-->  
(b) Lobby

<!-- image-->  
(c) Library

<!-- image-->  
(d) Zero-shot sim-to-real deployment  
Fig. 5: Real-world experiments. ReaDy-Go demonstrates robust real-world visual navigation performance after simto-real transfer in target environments (aâc) and an unseen environment (d), in both Static and Dynamic.

Second, photorealistic dynamic obstacles in our simulation are a key factor in maintaining visual navigation performance in dynamic environments. While ReaDy-Go and Vid2Sim perform comparably in static environments, their performance differs significantly in dynamic settings. ReaDy-Go maintains high success rates and low average reaching times relatively well, whereas Vid2Sim exhibits a substantial performance drop. Since the two methods differ only in the training data, i.e., photorealistic human GS dynamic obstacles for ReaDy-Go versus human assets in a physics engine for Vid2Sim, these results suggest that the proposed photorealistic dynamic GS simulation helps improve navigation performance in dynamic scenarios, which is a practical advantage for robot deployment.

## E. Real-World Visual Navigation Performance

To investigate the effectiveness of the proposed pipeline in mitigating the sim-to-real gap, we compare the real-world navigation performance of ReaDy-Go and baselines in the target environments shown in Fig. 5. For baselines in realworld experiments, Vid2Sim and ViNT are used, as they achieve better performance among the four baseline models in the simulation experiments.

Table II reports three main findings. First, the photorealistic real-to-sim simulation of ReaDy-Go facilitates sim-to-real transfer of visual navigation policies trained solely in simulation. ReaDy-Go achieves comparable success rates in both Static and Dynamic tasks in the real world, consistent with its simulation results across all environments, even though the real-world test environments exhibit minor appearance changes between data collection and deployment (e.g., floor texture, lighting, or background variations). This indicates that photorealistic datasets for dynamic scenarios improve the robustness of end-to-end visual navigation policies to the sim-to-real gap.

TABLE III: ReaDy-Go Generalization to an Unseen Env.
<table><tr><td>Task</td><td>SR â</td><td>ART (s) â</td></tr><tr><td>Static</td><td>70%</td><td>30.60</td></tr><tr><td>Dynamic</td><td>50%</td><td>35.45</td></tr></table>

Second, environment-specific policies (ReaDy-Go and Vid2Sim) trained on real-to-sim datasets for target environments achieve higher success rates and lower average reaching times than the general navigation model (ViNT), although ViNT requires 15Ã more parameters. This validates the practicality of ReaDy-Go for robots operating in specific environments, such as households, restaurants, and factories.

Lastly, photorealistic dynamic obstacles in ReaDy-Go are important for maintaining navigation performance in dynamic environments. In Dynamic, ReaDy-Go shows the highest success rate and the lowest average reaching time, with only a slight performance degradation compared to Static. In contrast, Vid2Sim exhibits a larger performance drop in Dynamic compared to ReaDy-Go.

## F. Zero-Shot Sim-to-Real Generalization

We further assess the generalization potential of ReaDy-Go by measuring zero-shot sim-to-real performance in an unseen environment. For this experiment, the policy is trained on the combined datasets from three environments, i.e., a total of 1,200 episodes from Outside, Lobby, and Library. We then deploy the policy to an unseen environment, as shown in Fig. 5â(d). The policy achieves over a 50% success rate in both Static and Dynamic tasks, with the higher average reaching time compared to the results in the training environments, as depicted in Table III. These results suggest that the policy learns general navigation behaviors, such as detouring around static and dynamic obstacles while reaching the goal, but requires a longer time to complete the task, as the policy does not have prior training in the unseen scene. ReaDy-Go can provide a scalable approach toward general navigation if trained on larger navigation datasets from diverse environments, which can be easily reconstructed from a few minutes of videos using the proposed photorealistic real-to-sim dynamic GS simulation pipeline.

## V. CONCLUSION

In this work, we propose ReaDy-Go, a real-to-sim dynamic GS simulation pipeline for training visual navigation policies. By integrating a GS scene, a human animation module, an expert planner, and a human planner, it generates photorealistic navigation datasets for dynamic scenarios in target environments. The navigation policy trained on the datasets achieves better navigation performance in target environments compared to baseline methods, including a prior real-to-sim simulation method and general navigation models, in both simulation and the real world. The results confirm that ReaDy-Go facilitates robust sim-to-real transfer even in dynamic environments and highlights the advantages of environment-specific policies and photorealistic dynamic obstacles. Furthermore, ReaDy-Go shows generalization through zero-shot sim-to-real deployment in an unseen environment. Future work includes expanding training environments to strengthen generalization or incorporating reinforcement learning to enhance navigation policies.

## REFERENCES

[1] D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine, âGNM: A General Navigation Model to Drive Any Robot,â in International Conference on Robotics and Automation (ICRA), 2023.

[2] D. Shah, A. Sridhar, N. Dashora, K. Stachowicz, K. Black, N. Hirose, and S. Levine, âViNT: A foundation model for visual navigation,â in 7th Annual Conference on Robot Learning, 2023.

[3] A. Sridhar, D. Shah, C. Glossop, and S. Levine, âNomad: Goal masked diffusion policies for navigation and exploration,â in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 63â70.

[4] G. Chhablani, X. Ye, M. Z. Irshad, and Z. Kira, âEmbodiedsplat: Personalized real-to-sim-to-real navigation with gaussian splats from a mobile device,â in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 25 431â 25 441.

[5] A. Majumdar, K. Yadav, S. Arnaud, J. Ma, C. Chen, S. Silwal, A. Jain, V.-P. Berges, T. Wu, J. Vakil, P. Abbeel, J. Malik, D. Batra, Y. Lin, O. Maksymets, A. Rajeswaran, and F. Meier, âWhere are we in the search for an artificial visual cortex for embodied intelligence?â in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 655â677.

[6] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. M. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang, M. Savva, Y. Zhao, and D. Batra, âHabitat-matterport 3d dataset (HM3d): 1000 large-scale 3d environments for embodied AI,â in Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

[7] S. Zhu, L. Mou, D. Li, B. Ye, R. Huang, and H. Zhao, âVrrobo: A real-to-sim-to-real framework for visual robot navigation and locomotion,â IEEE Robotics and Automation Letters, vol. 10, no. 8, pp. 7875â7882, 2025.

[8] A. Quach, M. Chahine, A. Amini, R. Hasani, and D. Rus, âGaussian splatting to real world flight navigation transfer with liquid networks,â in 8th Annual Conference on Robot Learning, 2024.

[9] J. Low, M. Adang, J. Yu, K. Nagami, and M. Schwager, âSous vide: Cooking visual drone navigation policies in a gaussian splatting vacuum,â IEEE Robotics and Automation Letters, vol. 10, no. 5, pp. 5122â5129, 2025.

[10] Y. Miao, E. Yuceel, G. Fainekos, B. Hoxha, H. Okamoto, and S. Mitra, âPerformance-guided refinement for visual aerial navigation using editable gaussian splatting in falcongym 2.0,â arXiv preprint arXiv:2510.02248, 2025.

[11] Z. Xie, Z. Liu, Z. Peng, W. Wu, and B. Zhou, âVid2sim: Realistic and interactive simulation from video for urban navigation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 1581â1591.

[12] S. Norelius, A. O. Feldman, and M. Schwager, âSketchplan: Diffusion based drone planning from human sketches,â arXiv preprint arXiv:2510.03545, 2025.

[13] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023.

[14] V. Tolani, S. Bansal, A. Faust, and C. Tomlin, âVisual navigation among humans with optimal control as a supervisor,â IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2288â2295, 2021.

[15] A. Loquercio, E. Kaufmann, R. Ranftl, A. Dosovitskiy, V. Koltun, and D. Scaramuzza, âDeep drone racing: From simulation to reality with domain randomization,â IEEE Transactions on Robotics, vol. 36, no. 1, pp. 1â14, 2020.

[16] F. Sadeghi and S. Levine, âCad2rl: Real single-image flight without a single real image,â arXiv preprint arXiv:1611.04201, 2016.

[17] K. Yadav, R. Ramrakhya, A. Majumdar, V.-P. Berges, S. Kuhar, D. Batra, A. Baevski, and O. Maksymets, âOffline visual representation learning for embodied navigation,â in Workshop on Reincarnating Reinforcement Learning at ICLR 2023, 2023.

[18] K. Yadav, A. Majumdar, R. Ramrakhya, N. Yokoyama, A. Baevski, Z. Kira, O. Maksymets, and D. Batra, âOvrl-v2: A simple state-of-art baseline for imagenav and objectnav,â arXiv preprint arXiv:2303.07798, 2023.

[19] S. Yoo, S. Jung, Y. Lee, D. Shim, and H. J. Kim, âMono-camera-only target chasing for a drone in a dense environment by cross-modal learning,â IEEE Robotics and Automation Letters, vol. 9, no. 8, pp. 7254â7261, 2024.

[20] R. Bonatti, R. Madaan, V. Vineet, S. Scherer, and A. Kapoor, âLearning visuomotor policies for aerial navigation using cross-modal representations,â in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020, pp. 1637â1644.

[21] M. Liu, H. He, E. Ricci, W. Wu, and B. Zhou, âUrbanverse: Scaling urban simulation by watching city-tour videos,â arXiv preprint arXiv:2510.15018, 2025.

[22] M. Deitke, R. Hendrix, A. Farhadi, K. Ehsani, and A. Kembhavi, âPhone2proc: Bringing robust robots into our chaotic world,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9665â9675.

[23] M. N. Qureshi, S. Garg, F. Yandun, D. Held, G. Kantor, and A. Silwal, âSplatsim: Zero-shot sim2real transfer of rgb manipulation policies using gaussian splatting,â arXiv preprint arXiv:2409.10161, 2024.

[24] J. Yu, L. Fu, H. Huang, K. El-Refai, R. A. Ambrus, R. Cheng, M. Z. Irshad, and K. Goldberg, âReal2render2real: Scaling robot data without dynamics simulation or robot hardware,â arXiv preprint arXiv:2505.09601, 2025.

[25] H. Lou, Y. Liu, Y. Pan, Y. Geng, J. Chen, W. Ma, C. Li, L. Wang, H. Feng, L. Shi, L. Luo, and Y. Shi, âRobo-gs: A physics consistent spatial-temporal model for robotic arm with hybrid representation,â in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 15 379â15 386.

[26] T. Chen, O. Shorinwa, J. Bruno, A. Swann, J. Yu, W. Zeng, K. Nagami, P. Dames, and M. Schwager, âSplat-nav: Safe real-time robot navigation in gaussian splatting maps,â IEEE Transactions on Robotics, vol. 41, pp. 2765â2784, 2025.

[27] X. Lei, M. Wang, W. Zhou, and H. Li, âGaussnav: Gaussian splatting for visual navigation,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 5, pp. 4108â4121, 2025.

[28] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â Â¨ in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104â4113.

[29] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, âPgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction,â IEEE Transactions on Visualization and Computer Graphics, vol. 31, no. 9, pp. 6100â 6111, 2025.

[30] W. Jiang, K. M. Yi, G. Samei, O. Tuzel, and A. Ranjan, âNeuman: Neural human radiance field from a single video,â in Proceedings of the European conference on computer vision (ECCV), 2022.

[31] M. Kocabas, J.-H. R. Chang, J. Gabriel, O. Tuzel, and A. Ranjan, âHUGS: Human gaussian splatting,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. [Online]. Available: https://arxiv.org/abs/2311.17910

[32] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, âSMPL: A skinned multi-person linear model,â ACM Trans. Graphics (Proc. SIGGRAPH Asia), vol. 34, no. 6, pp. 248:1â248:16, Oct. 2015.

[33] Y. Shafir, G. Tevet, R. Kapon, and A. H. Bermano, âHuman motion diffusion as a generative prior,â in The Twelfth International Conference on Learning Representations, 2024.

[34] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and L. Cheng, âGenerating diverse and natural 3d human motions from text,â in

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 5152â5161.

[35] F. Bogo, A. Kanazawa, C. Lassner, P. Gehler, J. Romero, and M. J. Black, âKeep it SMPL: Automatic estimation of 3D human pose and shape from a single image,â in Computer Vision â ECCV 2016, ser. Lecture Notes in Computer Science. Springer International Publishing, Oct. 2016.

[36] D. Dolgov, S. Thrun, M. Montemerlo, and J. Diebel, âPractical search techniques in path planning for autonomous driving,â ann arbor, vol. 1001, no. 48105, pp. 18â80, 2008.

[37] J. K. Haas, âA history of the unity game engine,â 2014.