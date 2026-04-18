# SketchPlan: Diffusion Based Drone Planning From Human Sketches

Sixten Norelius1, Aaron O. Feldman2, Mac Schwager2

Abstractâ We propose SketchPlan, a diffusion-based planner that interprets 2D hand-drawn sketches over depth images to generate 3D flight paths for drone navigation. SketchPlan comprises two components: a SketchAdapter that learns to map the human sketches to projected 2D paths, and DiffPath, a diffusion model that infers 3D trajectories from 2D projections and a first person view depth image. Our model achieves zeroshot sim-to-real transfer, generating accurate and safe flight paths in previously unseen real-world environments. To train the model, we build a synthetic dataset of 32k flight paths using a diverse set of photorealistic 3D Gaussian Splatting scenes. We automatically label the data by computing 2D projections of the 3D flight paths onto the camera plane, and use this to train the DiffPath diffusion model. However, since real human 2D sketches differ significantly from ideal 2D projections, we additionally label 872 of the 3D flight paths with real human sketches and use this to train the SketchAdapter to infer the 2D projection from the human sketch. We demonstrate SketchPlanâs effectiveness in both simulated and real-world experiments, and show through ablations that training on a mix of human labeled and auto-labeled data together with a modular design significantly boosts its capabilities to correctly interpret human intent and infer 3D paths. In real-world drone tests, SketchPlan achieved 100% success in low/medium clutter and 40% in unseen high-clutter environments, outperforming key ablations by 20â60% in task completion. The code for SketchPlan is publicly available at https://github.com/ sixnor/SketchPlan.

## I. INTRODUCTION

Instructing robots can be difficult for humans. Typical solutions include either low level teleoperation or high-level task specifications tailored for specific domains. This leaves a gap for loose structure interaction in a modality natural to humans. Natural language instruction has been successfully deployed in a variety of contexts [1], [2], [3], [4] but may become cumbersome to prompt as execution requirements become more nuanced. For example, consider trying to guide a robot towards a specific patch of a monochromatic floor. Because this task lacks features that are easy to articulate in natural language, it would be hard to accomplish. In contrast, communicating instructions via a hand-drawn sketch can offer fine-grained spatial intent while still remaining intuitive and quick to formulate for humans. Such approaches have found success in manipulation based tasks [5], [6] with an overhead or oblique camera view with few occlusions between objects. However, in contexts such as Unmanned Aerial Vehicle (UAV) navigation, the camera is typically egocentric, yielding a less informative perspective and making sketches harder to interpret via feature matching [6] or direct unprojection [5], [7] of the sketch into 3D. To overcome this challenge, imitation learning is a promising approach, but unfortunately, no dataset of real-world flights with matching sketches, egocentric images, and trajectories to learn from exists. Conducting a sufficient number of flights for a diverse set of environments would pose an exhausting endeavor.

<!-- image-->  
Fig. 1: SketchPlan Architecture. SketchAdapter takes a 2D human sketch and predicts the intended 2D path projection, trained with a small set of human-labeled data. DiffPath infers a 3D path from its 2D projection and an FPV depth image, trained with a large volume of auto-labeled 3D paths from diverse 3DGS scenes. Dashed arrows indicate loss computation for training.

Motivated by this, we choose to forgo real-world flight data altogether and instead assemble a dataset composed entirely of simulated drone paths matched with depth images using 3D Gaussian Splatting (3DGS) [8] in a diverse set of environments. This allows for rapid generation of highfidelity visual and spatial data, which we then label with hand-drawn sketches after the fact for a subset of the flights, creating a partially human-labeled dataset.

Utilizing this dataset, we propose SketchPlan, a flexible human-sketch to 3D path planner. SketchPlan is trained by dividing the planner into two separate components, SketchAdapter and DiffPath, allowing for exploitation of drone paths both with and without matching human sketches by training each component on different subsets of the data. SketchPlan takes as input a depth image with a hand-drawn human sketch overlaid and outputs a drone path composed of a sequence of 3D waypoints. Lending from the scale and flexibility of our approach, SketchPlan achieves zero-shot transfer from simulated data to entirely unseen real-world environments.

The main contributions of our approach are:

â¢ We introduce SketchPlan, a learned planner that translates a human-drawn sketch, drawn on top of a depth image, to a 3D flight path in 0.4s, enabling live humanin-the-loop navigation.

â¢ We train SketchPlan on a dataset of 32k 3D flight paths synthesized using photorealistic indoor and outdoor Gaussian Splatting scenes, where less than a thousand are human-labeled with matching sketches.

â¢ We show that SketchPlan achieves zero-shot generalizability, solving real-world drone navigation tasks in cluttered, unseen environments on hardware.

## II. RELATED WORK

In this section, we highlight some prior work in sketchbased methods for robotic instruction, 3DGS for UAV navigation, diffusion-based robot planning, and learning from partially-labeled data.

Sketching has been used in navigation tasks by drawing an approximate overhead map of the scene with a desired path for the robot to follow [4], [9], [10]. To match features between the map and the real-world to enable navigation, in [4], this is done via use of pretrained Visual Language Models (VLMs), whereas in [10], an occupancy map predictor is learned. Critically, the relation between the drawn sketch and path is inferred and not a priori specified, allowing for more nuance in interpretation. While sketching a full map enables precise spatial reasoning in cluttered environments, its complexity precludes on-the-fly sketching as in our approach.

As in our work, other approaches have considered sketching a robot trajectory, particularly for manipulation tasks [5], [6], [11], [7], [12]. Approaches like [5], [6], [12] utilize hindsight trajectory sketching, where they annotate a preexisting dataset with synthetic sketches in post, in order to train a policy that can then go from trajectory sketch to actions. In contrast, [11], [7] try to leverage sketching as a means for quicker collection of demonstration data for manipulation policies. Amongst these, [5], [11], [7] generate sketch training data by projecting 3D demonstration trajectories into 2D. However, projection may be a poor representation for how a human would draw sketches for the 3D demonstrations, particularly if given an uninformative camera perspective [7] such as a first-person view in our case. This highlights two key challenges for our problem of sketchbased drone navigation. Due to the egocentric camera view, we cannot rely on projections as proxies for hand-drawn sketches. Thus, we must learn a general, image-dependent association between sketch and trajectory, necessitating a novel dataset annotated with human sketches.

To collect this dataset, we rely on 3DGS [8] to generate photorealistic interpretable 3D representations of real-world scenes. In robotics, 3DGS has enabled large scale dataset generation [13], [14], [15]. In [14], [15], 3DGS is leveraged as the scene representation for a drone dynamics simulator, which is then used to train policies that achieve low sim-toreal gap. 3DGSâs photorealism also enables low sim-to-real gap for training UAV perception tasks like human recognition [16]. The 3DGS representation has also been directly used for robot planning. To generate collision-free paths, [2], [17] incorporate the gaussian splats into trajectory optimization.

Using the 3DGS data, we train a diffusion model to infer 3D trajectories from their 2D projections on a camera plane, conditioned on a depth image. Diffusion models have recently been adapted for robotic planning and control [18], [19], offering a powerful and flexible framework for modeling complex, multimodal trajectories. They can be trained via behavior cloning or reinforcement learning, and have shown success in tasks like visuomotor manipulation [18], [20], [21] and goal-conditioned planning [22], [23]. Beyond training, inference-time guidance has been proposed to better align outputs with human intent [24]. Despite their strengths, diffusion models are prone to overfitting on small datasets [25], motivating the generation of a large auto-labeled dataset for greater generalization.

We operate in a partially-labeled setting by using only a small subset of human-labeled data while training on a much larger auto-labeled dataset. We train a lightweight model to adapt from human-sketches to auto-labeled 2D projections. Recent works have applied similar strategies to enhance behavior cloning and skill transfer, such as mining positives from unlabeled data [26], and generating synthetic labels using model confidence, predictors, or image alignment [27], [6], [28], [29].

## III. DATASET

In this section, we describe how the training data is collected without requiring real-world examples of drone flights. This is enabled by exploiting 3D Gaussian Splatting (3DGS) models of several real-world environments. By using 3DGS environments, we can obtain accurate representations of both visual and geometric aspects of the scene. This accuracy results in minimal sim-to-real gap, enabling us to train using simulated data but then directly deploy our sketch-conditioned planner for real-world testing.

Each labeled sample of the dataset consists of a path $w ^ { ( j ) } \in \mathbf { R } ^ { T \times 3 }$ represented by a sequence of 3D waypoints in the droneâs body frame, a depth image $I ^ { ( j ) }$ , a 2D projection of the 3D path onto the camera plane $p ^ { ( j ) } \in \mathbf { R } ^ { T \times 2 }$ and for a subset of samples, a human sketch $\boldsymbol { s } ^ { ( j ) } \in \mathbf { R } ^ { T \times 2 }$ drawn on top of I(j). A relatively small subset of the data is humanlabeled, i.e., has associated human sketches, but we also use the remaining large number of 3D path samples auto-labeled with 2D projections in our modular training approach.

## A. Environments

We use the Zip-NeRF [30] dataset, which contains highquality images with registered poses in diverse environments, to train our model. Specifically, we use environments alameda, berlin, london and nyc, which feature a mix of indoor and outdoor scenes. Because the poses from these environments had different non-metric scalings, they were scaled to a shared metric frame by using common household objects with known measurements as reference (e.g., a video game console). This is crucial because the planner returns 3D paths in a metric frame. From the image and pose pairs, we produce 3DGS representations of the scenes using [31]. Although the raw images in the dataset contain no depth information, the 3DGS novel view synthesis capability enables the rendering of photo-realistic depth images from arbitrary poses of the drone camera. Moreover, the collision geometry obtained from 3DGS is high-quality, so that 3D paths which are collision-free with respect to the 3DGS will also be so in the ground-truth scene. We also reserve a test 3DGS environment (flight) similar to the real environments we use in hardware testing to assess how SketchPlan generalizes to an unseen environment.

## B. Path and View Generation

Given a 3DGS of an environment, we wish to automatically generate a large dataset of collision-free 3D trajectories in this environment to then label with sketches. To do so, we sample collision-free start and end points densely within the gaussian splat. To find a feasible trajectory between the two points, we employ Splatnav [2], a motion planning framework that generates collision-free, spline-parameterized 3D trajectories using a 3DGS environment representation. For data regularity, trajectories are clipped to a fixed arc length of 3 meters. A sequence of $T ~ = ~ 1 0 0$ discrete waypoints $w ^ { ( j ) }$ is obtained by selecting uniformly in arc length within the clipped trajectory. In addition, the depth image $I ^ { ( j ) }$ is rendered by orienting with the trajectory start i.e., the camera is placed at the first 3D waypoint while facing the second and has the world vertical axis aligned with the image vertical. We also collect the 2D projections $p ^ { ( j ) }$ of the paths $w ^ { ( j ) }$ onto the camera plane (for details see [32]).

In general, we find that our selection of environments yields good path diversity. The indoor scenes nyc and berlin feature sharp turns around corners with poor visibility, while the outdoor scenes london and alameda have less abrupt turns and more weakly structured geometry such as plants, bushes and trees.

## C. Sketching

Intuitively, it might be expected that for a given path $w ^ { ( j ) }$ and depth image $\bar { I ^ { ( j ) } }$ , the projection of the path $p ^ { ( j ) }$ and a corresponding human sketch $\bar { s } ^ { ( j ) }$ should be similar enough that collecting $s ^ { ( j ) }$ would be redundant. However, because of our egocentric camera viewpoint, $p ^ { ( j ) }$ and $s ^ { ( j ) }$ are often very different. For instance, as shown in Fig. 2 an almost entirely straight 3D path gets projected to nearly a single point in the middle of the image. Thus for the dataset to accurately represent human intent in going from $s ^ { ( j ) }$ to $w ^ { ( j ) }$ , directly using $p ^ { ( j ) }$ as a proxy for $s ^ { \bar { ( j ) } }$ does not suffice.

TABLE I: Dataset metrics for the number of 3D paths and human sketches in each environment. \*: Extrapolated from 100 sketches.
<table><tr><td>Env.</td><td>#Paths</td><td>#Sketches</td><td>Plan Time</td><td>Sketch Time*</td></tr><tr><td>alameda</td><td>9711</td><td>242</td><td>50 min</td><td>40 min</td></tr><tr><td>berlin</td><td>7488</td><td>203</td><td>47 min</td><td>30 min</td></tr><tr><td>london</td><td>8813</td><td>223</td><td>48 min</td><td>33 min</td></tr><tr><td>nyc</td><td>6550</td><td>204</td><td>37 min</td><td>26 min</td></tr><tr><td>Total</td><td>32562</td><td>872</td><td>3 h 2 min</td><td>2 h 9 min</td></tr><tr><td>flight</td><td>â</td><td>151</td><td>â</td><td></td></tr></table>

Therefore, after producing the dataset of 3D trajectories in the various environments, we obtain corresponding 2D sketches by querying a human user/labeler. Given a sequence of waypoints $w ^ { ( j ) }$ and depth image $I ^ { ( j ) }$ , the human draws a sketch matching $w ^ { ( j ) }$ . For user convenience, $w ^ { ( j ) }$ was displayed in a 3D interactive viewer of the 3DGS environment, allowing the sketcher to accurately gauge how $w ^ { ( j ) }$ interacts with the scene geometry. The user simply draws on top of $I ^ { ( j ) }$ using a computer mouse or stylus to produce $s ^ { ( \hat { j } ) }$ (by resampling their drawing to a fixed number of points). At first, our approach to data generation may seem counterintuitive; although we wish to predict from input sketch $s ^ { ( j ) }$ the corresponding path $w ^ { ( j ) }$ , the human labels in reverse, taking input $\bar { \boldsymbol { w } } ^ { ( j ) }$ and producing $s ^ { ( j ) }$ . However, first sampling a 3D trajectory and then drawing the corresponding 2D sketch allows for quicker labeling and the generation of auto-labeled data (paths with projections but without sketches) with no human interaction. Thus, we can generate a large and diverse auto-labeled dataset of path and depth image pairs for model training. Since labeling all paths is time-prohibitive, we use a modular approach to train. All 32k paths are used in training, while the human only sketches labels for a small subset, taking just 2 hours, as shown in Table I. However, a resulting limitation of this approach is that the human may desire paths never produced in automatic path generation i.e., there is mismatch between the types of paths humans would like to execute and ones that can arise from the data generation method. We counteract this pitfall by using a flexible path planner and choosing diverse environments to promote path diversity.

## IV. SKETCHPLAN

The sketch-to-path model, SketchPlan, is composed of two separate, independently trained modules. An overview of the architecture is presented in Fig. 1. The first, SketchAdapter, ingests human sketches and infers the 2D camera projection of the 3D path. The second, DiffPath, aims to then reconstruct the 3D paths from the 2D camera projections. Since 2D projections are available for both the labeled and autolabeled data, we train DiffPath on the full dataset. In contrast,

SketchAdapter is trained only with the human labeled data. Thus, taken together SketchAdapter and DiffPath use both labeled and auto-labeled data in a partially-labeled learning approach. At inference, these components are chained together to go from human sketch to 3D path. Lastly, the paths generated by DiffPath tend to avoid collisions, but are not guaranteed to do so. We therefore include a reactive collision avoidance controller on the drone as a safety filter on the paths generated by SketchPlan.

## A. SketchAdapter

The human sketch $s ^ { ( j ) }$ and projection $p ^ { ( j ) }$ are often not very similar. However, we still expect them to be highly correlated. To bridge the gap between sketch and projection, we train our adaptation module, SketchAdapter, to go from $s ^ { ( j ) } \tan p ^ { ( j ) }$ , outputting predictions $\hat { p } ^ { ( j ) }$ on the human-labeled portion of the dataset. To avoid overfitting to the limited amount of labeled data, we opt to learn an affine transform between the two using Partial Least Squares (PLS) [33], a method which maximizes covariance between the input and output features in a latent space with a fixed number of components. We choose the number of components to be 5, resulting in a low-dimensional yet surprisingly expressive latent for the SketchAdapter model.

<!-- image-->

<!-- image-->  
Fig. 2: Comparison of human sketch and projection (left) for a 3D path (right). Note that the camera projection of the 3D path differs significantly from the human sketch. The projection maps the straight part of the path to a tight pixel cluster in the middle of the image while the human instead sketches a straight vertical line.

## B. DiffPath

Our path generation model, DiffPath, ingests at inference the depth image $I ^ { ( j ) }$ and approximate projection $\hat { p } ^ { ( j ) }$ produced from SketchAdapter and returns a suitable 3D path $\hat { w } ^ { ( j ) }$ . There exist many suitable 3D paths for a particular 2D human sketch and image due to ill-posedness of the problem. Rather than explicitly/structurally biasing the model toward a class of suitable paths, DiffPath learns the distribution of 3D paths from training examples. The model can thus learn human-sketching biases through data. We use a diffusion model as the DiffPath backbone to allow for expressive path distribution modeling.

<!-- image-->  
Fig. 3: DiffPath Encoder. Our encoder combines the depth image and 2D projection into a compact embedding $z ^ { ( j ) }$ by sampling patches along the projection on a granular feature map. Convolutional kernels denoted as $K \times K \times C / S$ , where K is the kernel size, C the channels and S the stride.

Encoder. Our vision encoder is based on a convolutional neural network (CNN). We rescale the depth images to size $2 2 4 \times 2 2 4$ , normalize the pixel values to be zero mean with unit variance and clip the maximum depth at 15 m. We first pass the depth image $I ^ { ( j ) }$ through 6 convolutional layers to produce an intermediate feature map. Then, we sample from the produced feature map along $\hat { p ^ { ( j ) } }$ (which has pixel coordinates) using bilinear interpolation, to obtain a set of the local feature embeddings. Notably, these local feature embeddings serve as intermediates that simultaneously capture information from the projection $\hat { p } ^ { ( j ) }$ and the depth image. Finally, an additional convolutional layer is applied to the full feature map to obtain a coarse global embedding of the image. After this, the local, global and $\hat { p } ^ { ( j ) }$ features are flattened and concatenated, to serve as inputs for an MLP that produces a final combined embedding $z ^ { ( j ) }$ of $\hat { p } ^ { ( j ) }$ and $I ^ { ( j ) }$

Denoising Network. We utilize a 1D UNet with Featurewise Linear Modulation (FiLM) [34] conditioning as the denoising network as in [18]. To condition the denoising network, we use the embedding $z ^ { ( j ) }$ produced by our encoder. Thus, the diffusion denoising process depends on the provided projection and depth image. We utilize a 1D UNet with Feature-wise Linear Modulation (FiLM) [34] conditioning, as in [18], to model the denoising process. At inference time, DiffPath starts from a purely noisy path sample $\hat { w } _ { T } ^ { ( j ) } \sim \mathcal { N } ( 0 , I )$ and by conditioning on $z ^ { ( j ) }$ iteratively refines it toward a coherent 3D trajectory through a sequence of denoising steps:

$$
\hat { w } _ { t - 1 } ^ { ( j ) } = F _ { \theta } \left( \hat { w } _ { t } ^ { ( j ) } , t , z ^ { ( j ) } \right)\tag{1}
$$

where $F _ { \theta }$ is the stochastic denoising function parametrized by Î¸ and t is the denoising step. Each step incrementally shapes the path to be consistent with the scene and user intent encoded in $z ^ { ( j ) }$ , eventually reaching a denoised predicted path $\hat { w } _ { 0 } ^ { ( j ) } = \hat { w } ^ { ( j ) }$

During training, we jointly optimize both encoder and denoising networks. The latent representation $z ^ { ( j ) }$ is constructed using the auto-labeled 2D projection $p ^ { ( j ) }$ (instead of $\hat { p } ^ { ( j ) } )$ . The denoising network is trained to reconstruct the original 3D path $w ^ { ( j ) }$ in the dataset from a perturbed version by minimizing mean squared error (see [18] for details).

## C. Collision Filter

Although DiffPath conditions on the depth image and usually avoids collisions, the generated path waypoints are not guaranteed to be collision free. To remedy this issue at test time, we postprocess the generated waypoints with a simple collision filter. If a generated waypoint $\hat { w } _ { i }$ is in collision with the depth image, the filter finds a modified collision-free waypoint $\tilde { w } _ { i }$ close to $\hat { w } _ { i }$ . To identify collision, we use the depth images to create a live 3D point cloud at runtime. More specifically, we retain the last 10 s of depth images, unproject the pixels in each depth image into a 3D point cloud with points $\{ p _ { i } \} _ { i = \cdot } ^ { N }$ and combine and filter for outliers. To tractably find a collision-free $\tilde { w } _ { i } ,$ , we first find a collision-free convex polytope around the current robot position x in which to search.

Algorithm 1: Safe Polytope Construction   
Input: Current position x, point cloud $\{ p _ { i } \} _ { i = 1 } ^ { N }$ , robot   
radius r   
Output: Set of halfspace constraints H   
$\mathcal { P }  \{ p _ { i } \} _ { i = 1 } ^ { N } \qquad / /$ Init unchecked points   
H â â   
while $\mathcal { P } \neq \emptyset$ do   
pclose â NearestNeighbor $_ { \mathcal { P } } ( x )$   
$a  ( \underline { p } _ { \mathrm { c l o s e } } - x ) / \| p _ { \mathrm { c l o s e } } - x \|$   
$b \gets a ^ { \top } p _ { \mathrm { c l o s e } } - r$   
${ \mathcal { H } }  { \mathcal { H } } \cup \{ ( a , b ) \}$   
${ \mathcal { P } }  \{ p \in { \dot { \mathcal { P } } } \mid a ^ { \top } p \leq a ^ { \top } p _ { \mathrm { c l o s e } } \}$ // Prune   
end

For each obstacle point $p _ { i } ,$ , we define a supporting hyperplane orthogonal to the vector $p _ { i } - x$ and tangent to the r-ball centered at $p _ { i }$ . A waypoint $\tilde { w } _ { i }$ is guaranteed to be collisionfree if it lies outside all such half-spaces, but some can be pruned while remaining safe, as we detail in Algorithm 1.

$$
\begin{array} { r l } { \underset { \tilde { w } _ { i } } { \mathrm { m i n } } } & { \| \tilde { { w } } _ { i } - \hat { w } _ { i } \| _ { 2 } ^ { 2 } + \alpha \| t \| _ { 1 } } \\ { \mathrm { s u b j e c t ~ t o } } & { a _ { i } ^ { T } \tilde { w } _ { i } \leq b _ { i } + t _ { i } , \quad ( a _ { i } , b _ { i } ) \in \mathcal { H } } \\ & { t _ { i } \geq 0 } \end{array}\tag{2}
$$

Once this collision-free polytope has been formed, we solve for $\tilde { w } _ { i }$ by minimizing the distance to $\hat { w } _ { i }$ , while adhering to the half-space constraints but with slack t to obtain a solution if this is infeasible. This yields a quadratic program that can be solved at least 50 Hz on the drone, shown in (2).

## V. SIMULATED EXPERIMENTS

We test SketchPlan on the unseen simulation environment flight to see how well it performs on a novel 3DGS scene, like those in the training dataset.

## A. Setup

The test setup in simulation is composed of an interactive sketching interface, where novel views are rendered from a 3DGS representation of flight upon which the user can draw a sketch to generate a path using SketchPlan. The collision filter is disabled in order to gauge how well the model implicitly plans collision-free paths. For evaluation, we utilize both new sketches without ground-truth 3D paths and 3D paths labeled with sketches, to gather a mix of qualitative and quantitative results.

## B. Metrics

For the simulated testing, we define two quantitative metrics. Path Distance (PD), measures how much the predicted path $\hat { w } ^ { ( j ) }$ differs from the ground truth $w ^ { ( j ) }$ , defined as

$$
\mathrm { P D } = \sum _ { i = 1 } ^ { T } \| \boldsymbol { \hat { w } } _ { i } ^ { ( j ) } - \boldsymbol { w } _ { i } ^ { ( j ) } \| _ { 2 } .\tag{3}
$$

To gauge how well our model implicitly achieves collision avoidance, we check whether the robot would have collided with the 3DGS environment (leveraging the geometric interpretation of 3DGS as a union of ellipsoids as in [2]). We define the Collision Rate (CR) as the fraction of paths that collide with the environment.

## C. Ablations

To gauge the contribution of each component in our pipeline, we conduct a series of ablations to isolate the effect of individual design choices on overall performance.

w MLP SketchAdapter. The PLS SketchAdapter is replaced by an MLP with two hidden layers of dimension 256 each trained with BatchNorm and a dropout rate of 0.7.

w/o Diffusion Model. This ablation consists of the encoder directly predicting $\hat { w } ^ { ( j ) }$ without passing through the diffusion model. This is accomplished by adding an additional affine layer for regression after the encoder has produced the embedding $\bar { z ^ { ( j ) } }$ of the input features.

w/o SketchAdapter. The PLS SketchAdapter is removed entirely and the human sketch $s ^ { ( j ) }$ is fed directly to DiffPath and treated as if it were the projection $p ^ { ( j ) }$

w/o Auto-labeled data. We train DiffPath on the raw human sketches from the labeled portion of the dataset, forgoing SketchAdapter and auto-labeled 2D projection data entirely. Inference-Time Stochastic Sampling. To test whether it is valuable to condition the diffusion explicitly on the predicted projection $\hat { p } ^ { ( j ) }$ , we test only conditioning on the depth image and instead utilize cost-guided Stochastic Sampling (SS) diffusion [35], [24]. At test time, we add an additional term to the denoising process (Eq. 1) to guide the generated 3D path $\hat { w } ^ { ( j ) }$ to align with $\hat { p } ^ { ( j ) }$ i.e., to minimize the MSE between the 2D projection of $\hat { w } ^ { ( j ) }$ and $\hat { p } ^ { ( j ) }$

## D. Results

We find that SketchPlan takes on average 0.4s to generate a path in simulation on an NVIDIA RTX 4090, allowing it to be used in real-time human-in-the-loop operation.

<!-- image-->

Fig. 4: Left: Sample Sketches â Right: Resulting 3D Paths. Note that more curvature in the sketch results in greater side translation in the generated 3D path.  
<!-- image-->  
Fig. 5: Rollout of the same 2D sketch in two different poses. Note how the curvature of the 3D path (bottom) changes depending on the background scene (top). Arrows in bottom panels denote path direction.

In Figures 4 and 5, we qualitatively test how SketchPlan adapts to changes in both environment and sketch. In Fig. 4 while keeping the depth image fixed, we varied the drawn sketchâs curvature. Observe that as the curvature of the sketch increases, so does that of the path, reflecting responsiveness to the human sketch. However, SketchPlan can still struggle with sketches where the intent is to generate paths with sharp turns or complex curvature, as these are rarely generated by SplatNav [2] and thus are rare in our dataset. This can can particularly be seen in Fig. 5, where the sketch intent was to generate an almost 90Â°-turn, which SketchPlan fails to do.

Meanwhile, keeping the sketch fixed as in Fig. 5 but changing the environment, and the associated depth image presented to SketchPlan, also yields different paths. Specifically, we observe that the generated 3D path curvature changes based on the obstacle locations. This finding showcases SketchPlanâs capabilities to use the environment to infer more granular human intent and implicitly avoid collision. Together, the qualitative results from Figures 4 and 5 showcase that our model learns behavior from both the environment and sketch, and that it would not suffice to use just one of them to predict the path.

We also quantitatively assessed our method and compared against several ablations on withheld human-labeled validation data.

From Table II, we note that SketchAdapter is critical as without it (w/o SketchAdapter) both PD and CR are significantly higher, confirming that the human sketch and projections are indeed different with an egocentric camera view. The exact form of SketchAdapter seems to be less important, as we achieve similar metrics when using a shallow MLP (w MLP SketchAdapter) instead of PLS. We expect that as the number of sketches increases, deeper models for SketchAdapter could perform better.

TABLE II: Quantitative metrics for flight environment
<table><tr><td>Method</td><td>PD</td><td>Collision Rate</td></tr><tr><td>Ours</td><td>6.38</td><td>10.7%</td></tr><tr><td>w/o SketchAdapter</td><td>17.4</td><td>27.8%</td></tr><tr><td>w MLP SketchAdapter</td><td>6.18</td><td>12.9%</td></tr><tr><td>w/o Auto-labeled data</td><td>10.99</td><td>17.4%</td></tr><tr><td>w/o Diffusion Model</td><td>6.93</td><td>16.2%</td></tr><tr><td>Inference-Time SS [24]</td><td>10.81</td><td>14.3%</td></tr></table>

Training without auto-labeled data (w/o Auto-labeled data) also degrades both PD and CR, which we suspect is due to the diffusion model overfitting. In addition, we note that removing the diffusion model and directly using the encoder (w/o Diffusion Model) to predict the path yields a higher CR.

Finally, we find that using guidance instead of conditioning the model on the sketch (Inference-Time SS) does not fail catastrophically but lags behind our approach in both PD and CR. This finding suggests that training DiffPath with 2D projection data is beneficial, instead of only using the inferred 2D projection $\hat { p } ^ { ( j ) }$ at test time.

## VI. DRONE HARDWARE EXPERIMENTS

We also evaluated our method on unseen real-world environments, demonstrating successful sketch-based navigation with a quadrotor drone. We trial on multi-sketch navigation tasks in three different environments of varying difficulty. Although our method is trained purely with 3DGS simulation, we find that it can be successfully applied zero-shot, overcoming the sim-to-real gap.

In each trial, the task involves navigating from a 3D start to a goal position, by only drawing sketches and having the drone track the corresponding waypoints generated by SketchPlan. To accomplish this, the user draws a sketch, SketchPlan is inferenced and the drone executes the path to then hover in place awaiting another sketch, repeating until the task is completed or failed. A trial is concluded if the goal is reached or the drone enters a state from which the user cannot recover using SketchPlan.

## A. Setup

We construct three environments to reveal how our approach fares under different levels of obstacle clutter, height, and shape. Our Easy environment is composed of three triangular foam prisms arranged sparsely, acting as a test of SketchPlanâs capabilities in a simple uncluttered scene. For Medium, more obstacles are added and arranged more densely. We also add obstacles with a lower height to test the capabilities of the planner to traverse overhead and not only to the side of obstacles. Hard features more intricate geometry such as a ladder and flight gate and is densely occupied. Hard was designed to be out of distribution from the training dataset, which generally contains sparsely occupied environments with simple geometry and limited verticality, to test the limits of SketchPlan.

<!-- image-->  
Fig. 6: Sample drone hardware trials for the three environments. Green â Ours, Blue â w/o Diffusion, Red â w/o SketchAdapter. The wireframe ball indicates the goal region.

Our hardware setup consists of a quadrotor drone equipped with a Pixracer flight controller, Nvidia Jetson Orin Nano and ZED Mini stereo camera, from which we obtain a depth image of the current egocentric quadrotor view. All computation and model inference is off-board by transmitting the depth images over Wi-Fi, and sending waypoints back to the drone. Since our model requires connection to a host device for sketching, conducting inference off-board does not induce an additional need for computational resources.

TABLE III: Real-World Testing Metrics Across Difficulties and Models
<table><tr><td>Env.</td><td>Model</td><td>Finish</td><td>Alt.</td><td>Avoid</td><td>Side</td><td>Sketches</td></tr><tr><td rowspan="3">Easy</td><td>Ours</td><td>5/5</td><td>4/5</td><td>17 cm</td><td>5/5</td><td>2</td></tr><tr><td>w/o DP</td><td>2/5</td><td>1/2</td><td>11 cm</td><td>2/2</td><td>2</td></tr><tr><td>w/o SA</td><td>3/5</td><td>1/3</td><td>23 cm</td><td>2/3</td><td>2.33</td></tr><tr><td rowspan="3">Medium</td><td>Ours</td><td>5/5</td><td>4/5</td><td>30 cm</td><td>4/5</td><td>3</td></tr><tr><td>w/o DP</td><td>2/5</td><td>2/2</td><td>39 cm</td><td>1/2</td><td>3</td></tr><tr><td>w/o SA</td><td>3/5</td><td>2/3</td><td>33 cm</td><td>1/3</td><td>3</td></tr><tr><td rowspan="3">Hard</td><td>Ours</td><td>2/5</td><td>2/2</td><td>29 cm</td><td>2/2</td><td>3</td></tr><tr><td>w/o DP</td><td>1/5</td><td>1/1</td><td>36 cm</td><td>0/1</td><td>4</td></tr><tr><td>w/o SA</td><td>1/5</td><td>1/1</td><td>23 cm</td><td>1/1</td><td>3</td></tr></table>

## B. Metrics

For each task attempt, the user is assigned a goal position to reach using sketch-controlled navigation. They then interact with the drone using continuous sketching to draw a sequence of sketches to attempt to reach the goal.

At the most basic level, we measure whether the task was completed at all and how many sketches it took using continuous sketching. We define Finish to be when the drone enters within 1 m of the goal location, which is a predetermined point in 3D space, visualized by wireframe balls in Fig. 6. Then, Sketches is the number of sketches it took in total to go from start to goal.

For the trials where the task was successfully completed, we record additional metrics related to whether human intent was mirrored in the executed paths, both in terms of topology and distance deviation. After the user had drawn a sketch but before the drone had started executing the predicted path, the human noted their intent according to the follow criteria:

Side On which sides should obstacles be passed? right / left / above / below   
Altitude What is the intended altitude? ascend / level / descend

Following path generation and tracking for all sketches in a run, these metrics were then scored to see whether user intent was fulfilled. For Altitude, we define ascend (descend) to be whether the droneâs final position has increased (decreased) by 30 cm or more compared to its starting position.

We also collect a metric Avoid to see whether collision avoidance noticeably modified the original path generated by DiffPath by calculating the maximum deviation between the predicted waypoints and the collision avoidance adjusted ones for each sketch.

## C. Results

The results of the trials are noted in Table III. The metrics beyond task completion (Altitude, Avoid, Side) are only presented for the trials that finished successfully. The results indicate that our full method outperforms the baseline ablations. SketchPlan manages to complete all trials successfully in the easy and medium environments. While it successfully reaches the goal less often in the hard environment, it still manages to satisfy the Altitude and Side metrics in cases where it does succeed. The typical failure mode for our method in the hard environment was to get stuck in front of the ladder and flight gate. We hypothesize this issue is because these obstacles have small openings surrounded by thin edges, which donât occur in the training environments. In contrast, without the diffusion model, the drone tended to generate highly noisy paths and veer from the sketch intension. Without SketchAdapter, the quadrotor tended to go out of bounds, especially towards the ceiling.

This indicates that it is misinterpreting the intent of the user, as also reflected in the Side and Altitude metrics.

## VII. CONCLUSION

We presented SketchPlan, a diffusion-based drone path planner capable of converting human sketches drawn on depth images into viable 3D trajectories. By leveraging 3D Gaussian Splatting environments, we sidestep collecting realworld flight data, enabling scalable and diverse synthetic training. Our two-module design of SketchAdapter and Diff-Path enables mixed-data training, using both human-labeled and auto-labeled data, and robust generalization. Simulated and real-world evaluations confirm that SketchPlan captures nuanced human intent and adapts to environmental context, outperforming ablations in path accuracy and task success. Our results demonstrate the feasibility and advantages of using sketch-conditioned diffusion models for intuitive UAV navigation in complex settings.

Limitations. Our method is trained on sketches drawn to match pre-generated paths, which may not reflect the full range of paths that humans could intend to generate. This limits the types of paths SketchPlan can generate, such as correctly interpreting sketches with complex obstacle interaction or curvature. Moreover, all sketches in our dataset are drawn by the authors, which may not be representative for a broader set of users. While our collision filter reactively helps ensure safety, it cannot replan globally, and thus may get stuck in situations from which the user cannot sketch to recover. Future work could address these concerns by 1) taking human intent into account when generating the dataset, 2) collecting a more diverse set of sketches and 3) implementing a more adaptive replanning solution.

## REFERENCES

[1] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, âVoxposer: Composable 3d value maps for robotic manipulation with language models,â arXiv preprint arXiv:2307.05973, 2023.

[2] T. Chen, O. Shorinwa, J. Bruno, A. Swann, J. Yu, W. Zeng, K. Nagami, P. Dames, and M. Schwager, âSplat-nav: Safe real-time robot navigation in gaussian splatting maps,â IEEE Transactions on Robotics, 2025.

[3] C. Lynch, A. Wahid, J. Tompson, T. Ding, J. Betker, R. Baruch, T. Armstrong, and P. Florence, âInteractive language: Talking to robots in real time,â IEEE Robotics and Automation Letters, 2023.

[4] A. H. Tan, A. Fung, H. Wang, and G. Nejat, âMobile robot navigation using hand-drawn maps: A vision language model approach,â arXiv preprint arXiv:2502.00114, 2025.

[5] J. Gu, S. Kirmani, P. Wohlhart, Y. Lu, M. G. Arenas, K. Rao, W. Yu, C. Fu, K. Gopalakrishnan, Z. Xu et al., âRt-trajectory: Robotic task generalization via hindsight trajectory sketches,â arXiv preprint arXiv:2311.01977, 2023.

[6] P. Sundaresan, Q. Vuong, J. Gu, P. Xu, T. Xiao, S. Kirmani, T. Yu, M. Stark, A. Jain, K. Hausman et al., âRt-sketch: Goal-conditioned imitation learning from hand-drawn sketches,â in 8th Annual Conference on Robot Learning, 2024.

[7] S. A. Mehta, H. Nemlekar, H. Sumant, and D. P. Losey, âL2d2: Robot learning from 2d drawings,â arXiv preprint arXiv:2505.12072, 2025.

[8] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[9] M. Skubic, C. Bailey, and G. Chronis, âA sketch interface for mobile robots,â in SMCâ03 Conference Proceedings. 2003 IEEE International Conference on Systems, Man and Cybernetics. Conference Theme-System Security and Assurance (Cat. No. 03CH37483), vol. 1. IEEE, 2003, pp. 919â924.

[10] C. Xu, C. Amato, and L. L. Wong, âRobot navigation in unseen environments using coarse maps,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 2932â2938.

[11] P. Yu, A. Bhaskar, A. Singh, Z. Mahammad, and P. Tokekar, âSketchto-skill: Bootstrapping robot learning with human drawn trajectory sketches,â arXiv preprint arXiv:2503.11918, 2025.

[12] W. Zhi, H. Tang, T. Zhang, and M. Johnson-Roberson, âTeaching periodic stable robot motion generation via sketch,â IEEE Robotics and Automation Letters, 2024.

[13] J. Mao, B. Li, B. Ivanovic, Y. Chen, Y. Wang, Y. You, C. Xiao, D. Xu, M. Pavone, and Y. Wang, âDreamdrive: Generative 4d scene modeling from street view images,â arXiv preprint arXiv:2501.00601, 2024.

[14] J. Low, M. Adang, J. Yu, K. Nagami, and M. Schwager, âSous vide: Cooking visual drone navigation policies in a gaussian splatting vacuum,â IEEE Robotics and Automation Letters, 2025.

[15] Q. Chen, J. Sun, N. Gao, J. Low, T. Chen, and M. Schwager, âGrad-nav: Efficiently learning visual drone navigation with gaussian radiance fields and differentiable dynamics,â arXiv preprint arXiv:2503.03984, 2025.

[16] J. Choi, D. Jung, Y. Lee, S. Eum, D. Manocha, and H. Kwon, âUavtwin: Neural digital twins for uavs using gaussian splatting,â 2025. [Online]. Available: https://arxiv.org/abs/2504.02158

[17] R. Jin, Y. Gao, Y. Wang, H. Lu, and F. Gao, âGs-planner: A gaussian-splatting-based planning framework for active high-fidelity reconstruction,â 2024. [Online]. Available: https://arxiv.org/abs/2405. 10142

[18] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, âDiffusion policy: Visuomotor policy learning via action diffusion,â The International Journal of Robotics Research, p. 02783649241273668, 2023.

[19] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine, âPlanning with diffusion for flexible behavior synthesis,â arXiv preprint arXiv:2205.09991, 2022.

[20] D. Wang, S. Hart, D. Surovik, T. Kelestemur, H. Huang, H. Zhao, M. Yeatman, J. Wang, R. Walters, and R. Platt, âEquivariant diffusion policy,â arXiv preprint arXiv:2407.01812, 2024.

[21] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, â3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations,â arXiv preprint arXiv:2403.03954, 2024.

[22] A. Sridhar, D. Shah, C. Glossop, and S. Levine, âNomad: Goal masked diffusion policies for navigation and exploration,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 63â70.

[23] M. Reuss, M. Li, X. Jia, and R. Lioutikov, âGoal-conditioned imitation learning using score-based diffusion policies,â arXiv preprint arXiv:2304.02532, 2023.

[24] Y. Wang, L. Wang, Y. Du, B. Sundaralingam, X. Yang, Y.-W. Chao, C. Perez-DâArpino, D. Fox, and J. Shah, âInference-time policy steering through human interactions,â arXiv preprint arXiv:2411.16627, 2024.

[25] C. He, X. Liu, G. S. Camps, G. Sartoretti, and M. Schwager, âDemystifying diffusion policies: Action memorization and simple lookup table alternatives,â arXiv preprint arXiv:2505.05787, 2025.

[26] Q. Wang, R. McCarthy, D. C. Bulens, K. McGuinness, N. E. OâConnor, F. R. Sanchez, N. Gurtler, F. Widmaier, and S. J. Redmond, Â¨ âImproving behavioural cloning with positive unlabeled learning,â in Conference on robot learning. PMLR, 2023, pp. 3851â3869.

[27] Q. Zheng, M. Henaff, B. Amos, and A. Grover, âSemi-supervised offline reinforcement learning with action-free trajectories,â in International conference on machine learning. PMLR, 2023, pp. 42 339â 42 362.

[28] J. Park, Y. Seo, J. Shin, H. Lee, P. Abbeel, and K. Lee, âSurf: Semi-supervised reward learning with data augmentation for feedback-efficient preference-based reinforcement learning,â 2022. [Online]. Available: https://arxiv.org/abs/2203.10050

[29] Z. Rozsypalek, G. Broughton, P. Linder, T. Rou Â´ cek, K. Kusumam, Ë and T. KrajnÂ´Ä±k, âSemi-supervised learning for image alignment in teach and repeat navigation,â in Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing, 2022, pp. 731â738.

[30] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âZip-nerf: Anti-aliased grid-based neural radiance fields,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 697â19 705.

[31] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, T. Wang, A. Kristoffersen, J. Austin, K. Salahi, A. Ahuja et al., âNerfstudio: A modular frame-

work for neural radiance field development,â in ACM SIGGRAPH 2023 conference proceedings, 2023, pp. 1â12.

[32] R. Hartley, Multiple view geometry in computer vision. Cambridge university press, 2003, vol. 665.

[33] H. Abdi, âPartial least squares regression and projection on latent structure regression (pls regression),â Wiley interdisciplinary reviews: computational statistics, vol. 2, no. 1, pp. 97â106, 2010.

[34] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, âFilm: Visual reasoning with a general conditioning layer,â in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[35] Y. Du, C. Durkan, R. Strudel, J. B. Tenenbaum, S. Dieleman, R. Fergus, J. Sohl-Dickstein, A. Doucet, and W. S. Grathwohl, âReduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc,â in International conference on machine learning. PMLR, 2023, pp. 8489â8510.