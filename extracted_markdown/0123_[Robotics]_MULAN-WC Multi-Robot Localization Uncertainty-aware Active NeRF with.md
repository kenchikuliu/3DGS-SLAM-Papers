# MULAN-WC: Multi-Robot Localization Uncertainty-aware Active NeRF with Wireless Coordination

Weiying Wangâ, Victor Caiâ , Stephanie Gilâ

âJohn A. Paulson School of Engineering and Applied Sciences

Harvard University

Email: weiyingwang, sgil@g.harvard.edu â victorcai@college.harvard.edu

<!-- image-->  
Fig. 1: Overview: We propose a collaborative, localization uncertainty-aware NeRF framework for a team of robots, employing wireless coordination and active best-next-view selection for novel view finding.

AbstractâThis paper presents MULAN-WC, a novel multirobot 3D reconstruction framework that leverages wireless signalbased coordination between robots and Neural Radiance Fields (NeRF). Our approach addresses key challenges in multi-robot 3D reconstruction, including inter-robot pose estimation, localization uncertainty quantification, and active best-next-view selection. We introduce a method for using wireless Angle-of-Arrival (AoA) and ranging measurements to estimate relative poses between robots, as well as quantifying and incorporating the uncertainty embedded in the wireless localization of these pose estimates into the NeRF training loss to mitigate the impact of inaccurate camera poses. Furthermore, we propose an active view selection approach that accounts for robot pose uncertainty when determining the best-next-views to improve the 3D reconstruction, enabling faster convergence through intelligent view selection. Extensive experiments on both synthetic and realworld datasets demonstrate the effectiveness of our framework in theory and in practice. Leveraging wireless coordination and localization uncertainty-aware training, MULAN-WC can achieve high-quality 3D reconstruction that is close to applying the ground truth camera poses. Furthermore, the quantification of the information gain from a novel view enables consistent

rendering quality improvement with incrementally captured images by commanding the robot to the novel view position. Our hardware experiments showcase the practicality of deploying MULAN-WC to real robotic systems.

## I. INTRODUCTION

Vision-based 3D reconstruction in previously unseen environments is pivotal in a broad spectrum of robotics applications, ranging from autonomous navigation [1] to mapping and localization [2] to scene understanding [3]. The conventional process typically involves: 1) collecting multi-modal sensory information from onboard sensors such as RGB-D cameras and inertial measurement units, 2) extracting geometric features to compute relative pose information, and 3) applying pose graph optimization to produce a 3D environment representation using geometrically constrained spatial feature information [4, 5, 6]. Scaling up this capability to a fleet of robots could enable better coverage and faster exploration in large-scale environments. Nevertheless, it is nontrivial to scale up conventional methods to a fleet of robots. This is due to challenges in effectively obtaining relative poses between robots, which are needed to align inter-robot frames and form a global understanding of the scene. Another problem lies in how to actively command the robot to acquire visual information so as to maximize the information gain in the 3D reconstruction [7, 8]. Active image acquisition is even more critical in a multi-robot setting to fully leverage the advantages of the fleet over a single robot. To address these challenges, we introduce a multi-robot collaborative framework utilizing Neural Radiance Fields (NeRF) for reconstruction, and using on-board wireless signal-based coordination to provide relative positional information between robots.

Firstly, to address the need for photometrically and geometrically accurate 3D reconstruction, a large number of works in NeRF [9, 10, 11] offer a revolutionary technique in synthesizing photorealistic 3D implicit representations from sparse 2D images. This is attributed to NeRFâs unique capability to model the volumetric density and color of light in a scene, enabling highly detailed and accurate reconstructions from diverse viewpoints. A crucial input that enables real-time NeRF in [11] is the camera pose corresponding to each image in the same frame. Using optical flow-based feature tracking described in [12], the relative camera poses can be computed in real-time and fed into the NeRF training. However, acquiring accurate relative positional information in a multi-robot system is nontrivial, especially in the absence of global localization systems like GNSS or motion capture systems. In the traditional multi-robot coordination or localization approach, multirobot SLAM is often dependent on the alignment of individual maps and subsequent pose estimation from overlapped appearance-based feature observations [5, 13], namely loop closure. However, inter-robot loop closure brings significant complications and computational overhead [14, 5, 15]. To satisfy the need for relative positional information, we instead use phased array-based wireless sensing between robots, building upon our previous work in [16, 17]. Here, the off-the-shelf WiFi chip, which is native for communication on most robotics platforms, can be used to obtain inter-robot positional information independent of appearance-based environmental features. This positional information thus can be utilized to compute the relative translation between any pair of robots and thus their image frames. Inspired by other works in incorporating depth information from the SLAM pipeline [12] where depth information is used as supervision, we also design our training loss to be aware of which regions of data are more certain than others based on the uncertainty in wireless sensing. Like other perception modalities, wireless sensing also encodes probabilistic perception due to environmental and hardware noise. This work develops a method to quantify the wireless localization uncertainty based on Angle of Arrival (AoA) profile reconstruction and correlation with the received AoA profile. Integrating this quantified uncertainty information into the NeRF training process allows us to bias the training loss and ensure that the training loss is informed about data regions with higher localization uncertainty, enhancing the accuracy and reliability of the 3D NeRF-generated reconstructions.

A multi-robot system offers advantages beyond merely improving the efficiency of scene coverage from different viewpoints [18]; we can also enable active image acquisition by determining the most informative next view for the NeRF model and controlling robots to acquire these additional images. Most works in applying NeRF to 3D reconstruction only passively process the images given by the perception pipeline. In resource-constrained large-scale deployment, it is beneficial to actively plan robots to acquire the most informative next image. Some works [19, 20] acquire images that can maximally cover the scene of interest, leading to higher information gain or better quality of reconstruction. However, to the best of our knowledge, this is the first time that active information acquisition has been applied to a coordinated multi-robot system for NeRF-based 3D reconstruction. The work in [19] proposes a promising approach to evaluate the potential information gained from a novel view by quantifying the reduction of the variance for the rendering. However, this quantification does not consider the localization uncertainty of the camera, which is particularly necessary in multi-robot or multiple-camera setups, making it inefficient when dealing with localization uncertainty across robots that use wireless coordination. We address this specifically by considering the inter-robot camera pose uncertainty in the characterization of the color posterior in 3D space. Our work integrates localization uncertainty quantification into the evaluation of novel-view information gain by deriving the reduction of the variance. Subsequently, we can direct the robots to actively capture the best images from a set of feasible next positions, for the team of robots to achieve the highest information gain in the NeRF model.

In summary, this work makes three main contributions to multi-robot 3D reconstruction integrated with NeRF:

â¢ Framework for integrating SAR-based wireless coordination for multi-robot NeRF: We present a framework that leverages multi-robot collaboration and SAR-based wireless coordination to enable multi-robot localization uncertainty-aware NeRF.

â¢ Collaborative active image acquisition: Our system introduces a framework for active image acquisition, utilizing uncertainty quantification and novel-view location sampling to direct robots for optimal data collection, maximizing information gain for NeRF.

â¢ Extensive hardware experiments: We conducted experiments on our customized hardware robot demonstrating that our framework does not only effectively achieve the same quality of rendering faster and higher converged quality, but also can actively command the robot to an unvisited place that reduces the variance of rendering.

## II. PROBLEM FORMULATION

In this section, we briefly review some background knowledge of NeRF and introduce the wireless coordination from our previous work [17] as a basis for our approach.

## A. NeRF Formulation

NeRF implicitly represents a scene using a fully connected neural network. In the ideal propagation ray-tracing model, the scene is modeled as a continuous function that maps any viewing angle D of 5D input coordinates, consisting of the position in Cartesian coordinates (x, y, z) and the viewing angle $( \theta , \phi )$ , to a color $c ( r , g , b )$ and a volume density Ï. NeRF renders the color of the sampling ray passing through the environment with classical volume rendering. Suppose we sample a ray from a position o in direction d. The points along the ray can be parameterized by

$$
r ( t ) = o + t d
$$

The color projection of the ray back to the projection plane is

$$
\mathcal { C } ( \boldsymbol { r } ) = \int _ { t _ { n } } ^ { t _ { f } } T ( t ) \sigma ( r ( t ) ) c ( D ) d t ,
$$

where $\begin{array} { r } { T ( t ) = e x p ( - \int _ { t _ { n } } ^ { t } \sigma r ( s ) d s ) } \end{array}$ is the accumulated transmittance along the sampling ray, and $t _ { n }$ and $t _ { f }$ are the artificial sampling box. In a realistic setup, the computation of the full integral of the color through the ray can be intractable. Instead, [21] discretizes the integral as the linear combination of multiple sample points along the ray. NeRF optimizes the approximated discrete function by minimizing the squared reconstruction error between the ground truth color of each pixel captured in training RGB images and the reconstructed rendering pixel colors. The loss function is then defined as

$$
\mathcal { L } = \sum _ { i } | | \mathcal { C } ( \boldsymbol { r } _ { i } ) - \bar { \mathcal { C } } ( \boldsymbol { r } _ { i } ) | | _ { 2 } ^ { 2 }\tag{1}
$$

where $\bar { \mathcal { C } } ( \boldsymbol { r } _ { i } )$ is the captured color from images.

## B. Collaborative NeRF

To achieve 3D reconstruction with more than one robot, one of the fundamental requirements is having a common frame of reference from a known camera extrinsic or relative transformation between cameras even if the data is collected from different robots from different views. Instead of being given a set of poses $\tau$ in the same frame of reference, we instead focus on the problem of having the sets of poses from all the robots $\alpha , \beta , \ldots$ . in the team in $\mathcal { T } _ { \alpha } , \ \mathcal { T } _ { \beta } , \ldots$ . Without loss of generality, we only focus on the observation from two robots Î± and $\beta .$ In order to align a pose $T _ { k } ^ { \alpha }$ of robot Î± at local time k and another pose $T _ { p } ^ { \beta }$ of robot $\beta$ at local time p, we need to obtain the inter-robot camera extrinsic $T _ { k _ { \alpha } } ^ { p \beta } = ( t _ { k _ { \alpha } } ^ { p \beta } , \theta _ { k _ { \alpha } } ^ { p \beta } )$ , which is the distance and Angle of Arrival (AoA) between two cameras on different robots.

## C. Wireless Coordination

In our previous work [14], we extract the AoA information between any two robots by measuring the phase difference in the Wi-Fi channel. Suppose we have two robots Î± and $\beta$ in communicating range at time t and their poses $T _ { \alpha }$ and $T _ { \beta }$ in local frames. We can measure relative position between two robots using ranging from the ultra-wideband (UWB) as well as AoA from our SAR-based framework output [17] with a probability density function of ranging and AoA annotated by $f _ { u w b } ( d \vert T ^ { \alpha } , T ^ { \beta } )$ and $f _ { a o a } ( \phi | T ^ { \alpha } , T ^ { \beta } )$ respectively, defined as:

$$
f _ { u w b } ( d | T ^ { \alpha } , T ^ { \beta } ) = c _ { 1 } \exp \left( \sigma _ { k , p } ^ { - 2 } ( d - \| t _ { k \alpha } ^ { p \beta } \| _ { 2 } ) ^ { 2 } \right)\tag{2}
$$

$$
f _ { a o a } ( \phi | T ^ { \alpha } , T ^ { \beta } ) = c _ { 2 } \exp \left( \kappa _ { k , p } ^ { 2 } c o s ( \phi _ { k \alpha } ^ { p \beta } - \theta _ { k \alpha } ^ { p \beta } ) \right)\tag{3}
$$

where $\begin{array} { r } { c _ { 1 } = \frac { 1 } { \sqrt { 2 \pi \sigma _ { k , p } ^ { 2 } } } } \end{array}$ and $c _ { 2 } = \frac { 1 } { 2 \pi I _ { 0 } ( \kappa _ { k , p } ) }$ . Here, $\sigma _ { k , p } ^ { 2 }$ and $t _ { k \alpha } ^ { p \beta }$ are the variance and mean of the distance measurement; and $\kappa _ { k , p } ^ { 2 }$ and $\theta _ { k \alpha } ^ { p \beta }$ are the concentration parameters computed as the inverse of the AoA variance and the mean of the AoA distribution.

## III. APPROACH

In this section, we present our multi-robot NeRF framework that addresses the challenges of inter-robot pose localization, uncertainty quantification, and active best-next-view finding. Our approach leverages wireless signals, specifically ranging and Angle of Arrival (AoA) measurements, to estimate the relative poses between robots. We develop a novel method to quantify the uncertainty of AoA estimates by reconstructing the AoA profile and correlating it with the received AoA profile. This uncertainty quantification is then integrated into the NeRF training process to mitigate the impact of inaccurate poses on the reconstruction quality. Furthermore, we propose an active view-finding approach that accounts for the position uncertainty of the robots when selecting the most informative views for NeRF training. By incorporating localization uncertainty into the novel view selection process, our framework can more accurately determine the best next views for each robot, even in the presence of pose uncertainty arising from wireless coordination.

A. Inter-robot Pose Localization and Uncertainty Quantification

As described in Section II-B, accurate multi-robot NeRF reconstruction relies on obtaining the transformation between cameras on different robots. In a multi-robot setup, we propose using wireless signals leveraging UWB ranging and WiFi AoA measurements to obtain accurate inter-robot poses. Suppose we have a pose $T _ { k } ^ { \alpha }$ in SE(3) of robot Î± at local time k and another pose $T _ { p } ^ { \beta }$ of robot $\beta$ at local time p. We then can obtain a wireless measurement of range and AoA between the two robots using onboard UWB and WiFi annotated by a tuple $( t _ { k _ { \alpha } } ^ { p \beta } , \theta _ { k _ { \alpha } } ^ { p \beta } )$ . If we aim to use robot $\alpha \mathrm { { s } }$ frame as the global frame, then the extrinsic or the rigid transformation of robot $\beta ^ { \bullet } { \bf s }$ camera pose can be represented by $T _ { k p } ^ { \alpha \beta } \oplus T _ { p } ^ { \beta }$ ; where the annotation â denotes rigid transformation. However, the accuracy of the resulting pose estimate is subject to the uncertainties in the ranging and AoA measurements as described in Eq 2 and Eq 3. To mitigate the impact of inaccurate poses on the NeRF training process, we propose applying a weight to each training example based on the uncertainty of the associated robot poseâs ranging and AoA measurements from the other robots.

Quantifying the uncertainty of AoA estimates is particularly challenging, since there is a lack of standard error quantification methods applicable from previous works. To address this, we propose a novel approach as follows. The ideal channel on wavelength Î» at robot Î± from robot $\beta$ over distance $d _ { \alpha \beta } ( t )$ is

$$
h _ { \alpha \beta } ( t ) = \frac { 1 } { d _ { \alpha \beta } ( t ) } e x p \left( \frac { - 2 \pi \sqrt { - 1 } } { \lambda } d _ { \alpha \beta } ( t ) \right)\tag{4}
$$

Suppose over $t = t _ { k } , \ldots , t _ { l }$ , robot Î± receives the measured channel $\overline { { h } } _ { \alpha \beta } ( t )$ and also collects its local pose information $\overline { { T } } _ { \alpha } ( t )$ containing the displacement distance, azimuth, and zenith of robot Î± from the center of its frame. Then the measured AoA profile is constructed as in [16] to be $\overline { { F } } _ { \alpha \beta } ( \phi , \theta )$ over a sample space in $( \phi , \theta )$ and the measured AoA is chosen as $( \overline { { \phi } } , \overline { { \theta } } ) = \arg \operatorname* { m a x } _ { ( \phi , \theta ) } \left\{ \overline { { F } } _ { \alpha \beta } ( \phi , \theta ) \right\}$ at the tallest peak.

Now, given $( { \overline { { \phi } } } , { \overline { { \theta } } } )$ and pose information $\overline { { T } } _ { \alpha } ( t _ { k } ) , \ldots , \overline { { T } } _ { \alpha } ( t _ { l } )$ we reconstruct the channel over $t = t _ { k } , \ldots , t _ { l }$ as

$$
h _ { \alpha \beta } ^ { \prime } ( t ) = e x p \left( \frac { - 2 \pi \sqrt { - 1 } } { \lambda } f _ { \alpha } ( \overline { { T } } _ { \alpha } ( t ) , \overline { { \phi } } , \overline { { \theta } } ) + \nu ( t ) \sqrt { - 1 } \right)\tag{5}
$$

where $f _ { \alpha }$ is the displacement of robot Î± projected along the measured AoA direction in the local frame, relative to the first observation at $t _ { k } ,$ , and $\nu ( t )$ is a zero-mean real random variable that injects Gaussian phase noise into each element of the channel to add a small realistic amount of error tolerance in the measured profile. The same AoA algorithm is run to obtain the reconstructed profile $F _ { \alpha \beta } ^ { \prime } ( \phi , \theta )$ and its tallest peak $( \phi ^ { \prime } , \theta ^ { \prime } )$ Since $F _ { \alpha \beta } ^ { \prime } ( \phi , \theta )$ is constructed from $( { \overline { { \phi } } } , { \overline { { \theta } } } )$ , making the noise $\nu ( t )$ small ensures that the reconstruction has $\left( \phi ^ { \prime } , \theta ^ { \prime } \right) = \left( \overline { { \phi } } , \overline { { \theta } } \right)$ when the $\phi$ and Î¸ sample spaces are discretized during profile computation. Figure 2 simulates the profiles for illustration.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 2: Simulation of the AoA variance methodology. (a) Measured AoA profile $\overline { { F } } _ { \alpha \beta } ( \phi , \theta )$ from a simulated measured wireless channel $\overline { { h } } _ { \alpha \beta }$ with 0.7 radians standard deviation injected phase noise. (b) Reconstructed profile $F _ { \alpha \beta } ^ { \prime } ( \phi , \theta )$ from reconstructed channel $h _ { \alpha \beta } ^ { \prime } ,$ with 0.5 radians standard deviation phase noise for tolerance. Both tallest peaks align at $( \overline { { \phi } } = 4 5 . 6 ^ { \circ } , \overline { { \theta } } = 9 0 ^ { \circ } )$ . (c) and (d) are respective top views.

The AoA uncertainty is then calculated as follows. Both profiles are cropped to a small rectangle R with $\Delta \phi = \Delta \theta = 1 0 ^ { \circ }$ around the measured AoA $( { \overline { { \phi } } } , { \overline { { \theta } } } )$ to ignore distant multipath components. Then define AoA uncertainty $\kappa _ { k , p }$ via

$$
\frac { 1 } { \kappa _ { k , p } } = \sum _ { ( \phi , \theta ) \in R } \overline { { F } } _ { \alpha \beta } ( \phi , \theta ) F _ { \alpha \beta } ^ { \prime } ( \phi , \theta )\tag{6}
$$

This measures how concentrated the received profile is around its peak $( { \overline { { \phi } } } , { \overline { { \theta } } } )$ , such that a lower AoA uncertainty $\kappa _ { k , p }$ corresponds to a more reliable AoA measurement. This AoA variance can be used in Equation 3 to understand the variability in the pose estimates, as described in the next section.

## B. Uncertainty-aware NeRF Training

As described in our problem formulation, if we have two sets of camera poses $T _ { \alpha }$ and $T _ { \beta }$ in their own local frames, we want to find the relative transformation between them to align the poses in the same frame of reference. For each wireless measurement between poses $T _ { k } ^ { \alpha }$ and $T _ { p } ^ { \beta }$ , we can quantify the uncertainties $\sigma _ { k , p }$ and $\kappa _ { k , p }$ respectively of the ranging and AoA estimates using the methods described in the previous section. We propose incorporating these uncertainty measures into the NeRF training process by modifying the standard pixel loss function $\mathcal { L }$ given in Eq 1 by re-scaling the loss for each training sample. For brevity of the annotation in this section, we omit robot indices $\alpha , \beta ,$ and all local time frames p and k from now on. We apply uncertainty propagation to compute the error ellipse of uncertain measurements. Since the ranging and AoA measurements are taken using different sensing modalities, they can be treated as independent measurements with the error ellipseâs axes aligning with x-y axes. The semimajor and semi-minor axes a and b of the error ellipse can be derived as:

$$
a ^ { 2 } = \sigma ^ { 2 } c o s ^ { 2 } ( \theta ) + t ^ { 2 } s i n ^ { 2 } ( \theta ) \kappa ^ { 2 }
$$

$$
b ^ { 2 } = \sigma ^ { 2 } s i n ^ { 2 } ( \theta ) + t ^ { 2 } c o s ^ { 2 } ( \theta ) \kappa ^ { 2 }\tag{7}
$$

(8)

where t is the ranging measurement and Î¸ is the AoA estimate. Then the scale factor with confidence interval CI is given by

$$
k = \sqrt { - 2 l o g ( 1 - C I ) }\tag{9}
$$

Hence, the uncertainty of this wireless localization $\gamma$ can be represented by the area of the error ellipse $\gamma = k ^ { 2 } \pi a b$ Then the new loss function can be re-scaled with $\gamma$ that is normalized by sigmoid function

$$
\mathcal { L } _ { u n c e r t a i n t y } = S I G M O I D ( \gamma ) \mathcal { L }\tag{10}
$$

By incorporating the uncertainty-aware scaling factor into the NeRF loss function, our multi-robot NeRF system can effectively learn to reconstruct the 3D scene while accounting for the varying reliability of the pose estimates obtained through wireless coordination. This approach results in a more robust and accurate 3D reconstruction, especially in scenarios where the pose estimates may be subject to significant uncertainties.

## C. Active Best-View Finding with Position Uncertainty

In a multi-robot NeRF system, actively selecting the best views for each robot to capture can significantly improve the efficiency and quality of the 3D reconstruction. However, the uncertainty in robot poses obtained through wireless coordination can impact the effectiveness of the view selection process. When a robot attempts to find the best next view location by proposing and evaluating potential new positions, the uncertainty in its current pose can lead to inaccurate assessments of the information gain at novel view locations.

To address this challenge, we propose an active view finding approach that incorporates the positional uncertainty of the robots to guide the selection of the most informative views for NeRF training. Building upon the approach proposed in [19] for evaluating the potential information gain from novel views by quantifying the reduction of variance in rendering, we extend this method to account for the uncertainty in the robotâs current position and its propagation to the novel view locations being evaluated. By considering localization uncertainty during the novel view selection process, we can more accurately determine the most informative next views for each robot, even in the presence of pose uncertainty arising from wireless coordination.

We adopt the assumption that the radiance color of any location along the ray $r ( t )$ can be parameterized by a Gaussian distribution with mean $\bar { c } ( r ( t ) )$ and variance $\bar { \beta } ( \boldsymbol { r } ( t ) )$ . To incorporate this uncertainty, we model the origin o of each ray r following a Gaussian distribution with 0 mean and the variance Ï, representing the localization uncertainty.

$$
\mathbf { \sigma } _ { \mathbf { o } } \sim \mathcal { N } ( 0 , \sigma )\tag{11}
$$

Assuming a NeRF model M has been trained on an initial collection of data D, the prior distribution $P ( c ( r ( t _ { k } ) ) | D )$ of the color c at location $r ( t )$ follows a Gaussian distribution $\mathcal { N } \sim ( \bar { c } ( r ( t ) ) , \bar { \beta } ^ { 2 } ( r ( t ) ) )$ . The accumulated color from a new ray r passing through can also be modeled as a Gaussian distribution:

$$
p ( C ( r ) | c ( r ( t ) ) , o )\tag{12}
$$

where $C ( r )$ is the color of the rendered pixel accumulated from the ray r, and $c ( r ( t ) )$ is the color of the location in 3D space. Then if we marginalize over o,

$$
p ( C ( r ) | c ( r ( t ) ) ) = \int p ( C ( r ) | c ( r ( t _ { k } ) ) ) * p ( o )  d o\tag{13}
$$

$$
p ( C ( r ) | c ( r ( t _ { k } ) ) ) \sim \mathcal { N } ( \sum _ { i = 1 } ^ { N } \alpha _ { i } \bar { c } ( r ( t ) ) , \sum _ { i = 1 } ^ { N } \alpha _ { i } * \sigma ^ { 2 } + \bar { \beta } ^ { 2 } ( r ( t _ { k } ) ) )\tag{14}
$$

Then apply Bayesâ rule to get the posterior:

$$
\begin{array} { r l } & { P ( C ( r ) | D , r ( t _ { k } ) , o ) } \\ & { \propto P ( C ( r ) | c ( r ( t _ { k } ) ) ) * P ( c ( r ( t _ { k } ) ) | D ) * P ( o ) } \end{array}
$$

$$
\propto e x p \left( - \frac { 1 } { 2 } \left( c ( r ( t _ { k } ) ) - \left( \frac { \omega C ( r ) } { \alpha _ { k } } + ( 1 - \omega ) \right) * \bar { c } ( r ( t _ { k } ) ) \right) \right.
$$

$$
\left( \frac { \alpha _ { k } ^ { 2 } } { \alpha _ { k } ^ { 2 } \sigma ^ { 2 } + \bar { \beta } ^ { 2 } ( r ) } + \frac { 1 } { \bar { \beta } ^ { 2 } ( r ( t _ { k } ) ) } \right) ^ { - 1 } \bigg )\tag{â}
$$

$$
w h e r e \ \omega = \frac { \alpha _ { k } ^ { 2 } \bar { \beta } ^ { 2 } ( r ( t _ { k } ) ) } { \alpha _ { k } ^ { 2 } \bar { \beta } ( r ( t _ { k } ) ) ^ { 2 } + \alpha _ { k } ^ { 2 } \sigma ^ { 2 } + \bar { \beta } ^ { 2 } ( r ) }\tag{15}
$$

Then extract the variance of the posterior distribution:

$$
\left( \frac { \alpha _ { k } ^ { 2 } } { \alpha _ { k } ^ { 2 } \sigma ^ { 2 } + \bar { \beta } ^ { 2 } ( r ) } + \frac { 1 } { \bar { \beta } ^ { 2 } ( r ( t _ { k } ) ) } \right) ^ { - 1 }\tag{16}
$$

As the localization variance increases, the uncertainty of the radiance field also increases accordingly. To select the best view, the metric will prefer the novel view with lower localization uncertainty. Since we only need to consider the variance reduction given multiple rays from a sampled novelview position, we can then command the robot to move to the location with highest variance reduction using Equation 16.

## IV. RESULTS

In this section, we present a comprehensive evaluation of our methodology through both synthetic datasets collected in synthetic environments and real-world datasets collected on our hardware robots. Our findings validate the effectiveness of our algorithm in integrating perspectives from multiple robots within an active acquisition framework, showcasing significant improvements in data capture and processing. We implemented our algorithm based on a Pytorch implementation of the SDF and NeRF part described in Instant-NGP [11] with CUDA-accelerated ray marching. We modified the loss function to implement the localization uncertainty-aware loss described in Eq 1. Further, for the active image collection, we re-implemented the rendering variance reduction described in [19], incorporating the localization uncertainty from Eq 16. We use a desktop with NVIDIA RTX 6000 for all evaluations.

## A. Wireless Variance

First, we present the performance benchmark with our proposed wireless variance metric as formulated in Section III-A. With a simulated trajectory, the metric is tested over 16 random trials with various amounts of injected Gaussian channel phase noise as defined in Equation 5, ranging from 0.01 to 3 radians standard deviation. As shown in Figure 3, the AoA error from the ground truth scales quickly and becomes unstable as our proposed AoA variance metric increases. To further demonstrate the relationship between the variance of the AoA error and our proposed profile variance, Figure 4 clearly shows that the variance of the AoA error will increase superlinearly as our metric increases. This result indicates that our AoA uncertainty quantification is well aligned as an indicator of the variance of AoA measurements, which can be further used to quantify the uncertainty of the camera position.

## B. Simulation Experiment

Our localization uncertainty-aware framework described in Sec III-B is first assessed using a synthetic dataset lego released with the original NeRF work [21] that is commonly used for evaluating NeRF frameworks. The dataset is partitioned into two subsets to simulate data acquired from two robots, allowing us to mimic the real-world scenario of capturing images from different angles and positions, thereby testing the robustness and adaptability of our algorithm in synthesizing and analyzing data from varied viewpoints. Three setups are evaluated with results in Table I:

<!-- image-->  
Fig. 3: Absolute AoA error from ground truth plotted against our AoA uncertainty metric. We see that nonzero AoA error grows as our AoA uncertainty metric grows, indicating that our metric successfully captures true error in measured AoA.

<!-- image-->  
Fig. 4: The variance of the AoA error here as a function of AoA uncertainty is calculated empirically by finding the variance of the AoA error on the y-axis within a sliding window of $\Delta \kappa _ { k , p } = 8 . 4 \times 1 0 ^ { 5 }$ along the x-axis. It is fit with a power curve of the form $y = a x ^ { b }$ , with $r ^ { 2 } = 0 . 8 9 4 2$

A [Oracle]: Camera poses from the dataset in the global frame (known as extrinsic between cameras) and images from the dataset.

B [Normalized camera poses with AOA and ranging simulated]: Camera poses from the dataset but normalized by the first pose in each partition. Then the AOA and ranging are simulated with noise whose standard deviation are 0.05 meters and 5 degrees respectively.

C [Normalized camera poses with AOA and ranging simulated with variance as supervision]: The same setup as B but the training loss is incorporated with the localization variance.

As predicted, training using localization variance produces better performance closer to using ground truth poses.

<table><tr><td>A</td><td>PSNR LPIPS</td><td>30.47 0.062</td></tr><tr><td>B</td><td>PSNR LPIPS</td><td>26.48 0.092</td></tr><tr><td>C</td><td>PSNR LPIPS</td><td>28.69 0.071</td></tr></table>

TABLE I: Performance comparison between different setups where larger PSNR values are better, and smaller LPIPS values indicate better quality, illustrating that applying uncertaintyaware loss can effectively improve the quality of the model.

## C. Hardware Experiment

For a real-world application, we deployed our algorithm on two customized Locobot PX100 robots. These robots were equipped with Oak-D Pro cameras, operating at 1080p 20Hz, along with DWM1001 UWB modules, 5dBi Antennas, and Intel NUC 10 computers for onboard processing. The experimental setup places a drone as a test object central relative to the two robots, which are programmed to navigate curved paths around the object to complete data capture. The wireless AOA measurements are computed by deploying [17] which only requires very small communication bandwidth at around 5 kB/s.

Both robots utilize onboard Visual Inertial Odometry (VIO) to estimate local camera displacement within their respective frames. At the onset of the experiment, Angle of Arrival (AoA) and ranging measurements are taken to establish an initial estimate of the relative positioning between the robots. Subsequently, the covariance of the VIO data was monitored to identify optimal intervals for refreshing wireless data collection. In the meantime, the testbed is equipped with the Optitrack motion capture (mocap) system providing the ground truth camera poses for each robot.

The experiments are conducted using five setups:

A [Oracle]: Camera poses captured by motion system in the global frame (known extrinsic between cameras) and images from the onboard camera.

B [Best case for our system]: Camera poses from motion capture system with wireless coordination and images from the onboard camera. The poses are normalized in each robotâs local frame.

C [Our system âin-the-wildâ (no mocap)]: Camera poses from onboard VIO, wireless coordination, and images from the onboard camera.

D [Our system âin-the-wildâ with variance as supervision]: Camera poses from onboard VIO, wireless perception for coordination, and uncertainty-aware training loss scaled by localization variance.

E [Benchmark comparison]: Camera poses from onboard VIO; COLMAP [22] is used for computing interrobot relative camera pose extraction.

All five setups are evaluated using standard metrics for NeRF: Peak Signal-to-Noise Ratio (PSNR) and Learned Perceptual Image Patch Similarity (LPIPS) [23]. Each metric is evaluated from samples in the test set ground truth images, along with camera poses. For each setup, there are a total of 100 images with camera poses that are collected continuously from each robot while robots are moving around the drone subject. drone-1 an drone-2 are different images in the testing dataset.

<table><tr><td></td><td></td><td>drone-1</td><td>drone-2</td></tr><tr><td>A</td><td>PSNR LPIPS</td><td>26.4 0.351</td><td>24.5 0.384</td></tr><tr><td>B</td><td>PSNR LPIPS</td><td>25.45 0.382</td><td>23.4 0.378</td></tr><tr><td>C</td><td>PSNR LPIPS</td><td>23.32 0.41</td><td>22.3 0.405</td></tr><tr><td>D</td><td>PSNR LPIPS</td><td>25.04 0.389</td><td>23.03 0.395</td></tr><tr><td>E</td><td>PSNR LPIPS</td><td>11.5 0.79</td><td>12.5 0.85</td></tr></table>

TABLE II: Performance comparison between different setups where larger PSNR values are better, and smaller LPIPS values indicate better quality. The comparison between setup A and setup B demonstrates that applying wireless coordination can effectively achieve close performance to having a global coordinate system. Results from setup C show the realistic performance of our system using fully onboard VIO for local positioning, which is degraded but still relatively robust. Setup D shows the scaling with localization uncertainty quantification can improve the quality almost to the best-case scenario in setup B. Setup E fails to produce a coherent 3D rendering.

As shown in Table II which provides our quantitative results, setup A shows the best performance we can achieve in a two-robot team since it is based on the ground truth camera poses provided by the motion capture system. In setup B, the poses are normalized by the starting pose of each robot captured in the motion capture system. Then wireless coordination is incorporated to provide inter-robot camera extrinsic. Setup C shows the realistic setup, which applies the wireless localization between robots to the local VIO poses and is effectively close to the result in setup B. Setup D shows that our framework can achieve better results than C by using the variance-aware loss function defined in Equation 10 with the corresponding localization variance proposed in Equation 16. The benchmark comparison setup E fails to produce a cohesive 3D rendering due to discrepancies in the relative camera pose estimation using COLMAP [22], which is commonly used for estimating the relative camera pose given two frames from different views. This is mainly due to the drastic translation change in camera view from different robots. This results also suggests that COLMAP wonât be a proper solution for a multirobot setup.

Many robotics applications require a quickly-converging view of the environment before the model training fully converges. In our experiment, we also validate that our methods not only deliver better rendering but also achieve faster PSNR improvement as shown in Fig 5.

## D. Active Image Capturing

For evaluating active best next view, following the initial phase of data gathering, a waiting period was observed until the Neural Radiance Field (NeRF) modelâs loss stabilized.

<!-- image-->  
Fig. 5: PSNR improvement over epochs, with different setups.

<!-- image-->

<!-- image-->  
(b) Re-rendered drone using wireless coordination and uncertaintyaware loss(PSNR:25.04).  
Fig. 6: An example of the drone we reconstructed in the testbed. The left figure is the ground truth image, right figure is the re-rendered image from a trained model.

The robots then execute a series of maneuvers, sampling eight different directions at 0.5-meter intervals. Our evaluation focused on minimizing the variance of the rendering posterior, employing Equation 16 to identify positions yielding the most significant reduction in variance. The application of our algorithm in a hardware setting demonstrates its practical feasibility. Moreover, it underscores the potential of our method to optimize the data capture process through strategic robot positioning and movement.

After selecting the location with the highest variance reduction using our proposed method, the robot is commanded to the new location and observes the environment again. We then let the model train until the loss stabilizes and repeat the process four times to evaluate the efficacy of our method. For comparison, we also randomly selected accessible locations around the robots and controlled the robots to move to those locations. The evaluation-maneuver-training loop was conducted on both our policy and the random policy and the results are reported in Table III. These results demonstrate that our approach provides a principle metric that can improve the quality of the rendering consistently.

## V. CONCLUSION

This work presents MULAN-WC, a multi-robot 3D reconstruction method that uses wireless signal-based coordination between robots. This work presents i) a framework for multirobot NeRF that uses SAR-based wireless relative position measurements to stitch together views of the environment from multiple robots, ii) uncertainty-based weighting of samples in the NeRF training as a supervision technique, where samples with greater wireless measurement noise are weighted less, leading to better accuracy of the combined rendering, and iii) collaborative active next-image acquisition, where novelview location sampling incorporates wireless pose uncertainty, and is used to direct robots to better sampling locations that reduce variance during NeRF training. We demonstrate the performance of the multi-robot framework in hardware, where our results show good quality of rendering according to the standard NeRF error metrics of PSNR and LPIPS, and consistent improvement when we additionally use the uncertainty of the AoA measurement as supervision in the NeRF training. Lastly, we show that AoA measurements can be used to select the best-next-view based on regions of better position accuracy and that this results in incremental rendering quality improvement.

<table><tr><td>observation#</td><td></td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td></tr><tr><td>Our</td><td>PSNR</td><td>19.66</td><td>19.80</td><td>20.04</td><td>20.08</td></tr><tr><td>algorithm</td><td>LPIPS</td><td>0.407</td><td>0.398</td><td>0.394</td><td>0.381</td></tr><tr><td>Random</td><td>PSNR</td><td>19.53</td><td>19.63</td><td>19.60</td><td>19.63</td></tr><tr><td>Exploration</td><td>LPIPS</td><td>0.422</td><td>0.419</td><td>0.421</td><td>0.418</td></tr></table>

TABLE III: Performance comparison between different setups, demonstrating that our method improves the rendering quality metric with consecutive views.

## ACKNOWLEDGMENTS

The authors gratefully acknowledge partial funding through NSF grant #CNS-2114733, Amazon ARA, and the Sloan award #FG-2020-13998.

## REFERENCES

[1] M. Meilland, A. I. Comport, and P. Rives, âDense omnidirectional rgb-d mapping of large-scale outdoor environments for real-time localization and autonomous navigation,â Journal of Field Robotics, vol. 32, no. 4, pp. 474â503, 2015.

[2] Y.-J. Yeh and H.-Y. Lin, â3d reconstruction and visual slam of indoor scenes for augmented reality application,â in 2018 IEEE 14th International Conference on Control and Automation (ICCA), 2018, pp. 94â99.

[3] Y. Nie, J. Hou, X. Han, and M. NieÃner, âRfd-net: Point scene understanding by semantic instance reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4608â4618.

[4] M. Cao, W. Jia, Y. Li, Z. Lv, L. Li, L. Zheng, and X. Liu, âFast and robust local feature extraction for 3d reconstruction,â Computers & Electrical Engineering, vol. 71, 2018.

[5] Y. Chang, Y. Tian, J. P. How, and L. Carlone, âKimera-multi: a system for distributed multi-robot metric-semantic simultaneous localization and mapping,â in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021, pp. 11 210â11 218.

[6] J. Yin, L. Carlone, S. Rosa, and B. Bona, âGraph-based robust localization and mapping for autonomous mobile robotic navigation,â in 2014 IEEE International Conference on Mechatronics and Automation. IEEE, 2014, pp. 1680â1685.

[7] Y. Gao, L. Su, and H. Liang, âMc-nerf: Multi-camera neural radiance fields for multi-camera image acquisition systems,â 8 2023.

[8] W. Wang, N. Jadhav, P. Vohs, N. Hughes, M. Mazumder, and S. Gil, âActive rendezvous for multi-robot pose graph optimization using sensing over wi-fi,â in Robotics Research, T. Asfour, E. Yoshida, J. Park, H. Christensen, and O. Khatib, Eds. Cham: Springer International Publishing, 2022, p. 832.

[9] J. Kerr, L. Fu, H. Huang, Y. Avigal, M. Tancik, J. Ichnowski, A. Kanazawa, and K. Goldberg, âEvo-nerf: Evolving nerf for sequential robot grasping of transparent objects,â in 6th annual conference on robot learning, 2022.

[10] M. Adamkiewicz, T. Chen, A. Caccavale, R. Gardner, P. Culbertson, J. Bohg, and M. Schwager, âVision-only robot navigation in a neural radiance world,â IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4606â4613, 2022.

[11] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural Â¨ graphics primitives with a multiresolution hash encoding,â ACM transactions on graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[12] A. Rosinol, J. J. Leonard, and L. Carlone, âNerf-slam: Real-time dense monocular slam with neural radiance fields,â 2022.

[13] H. Do, S. Hong, and J. Kim, âRobust loop closure method for multi-robot map fusion by integration of consistency and data similarity,â IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 5701â5708, 2020.

[14] W. Wang, A. Kemmeren, D. Son, J. Alonso-Mora, and S. Gil, âWi-closure: Reliable and efficient search of inter-robot loop closures using wireless sensing,â in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023.

[15] W. Wang, N. Jadhav, P. Vohs, N. Hughes, M. Mazumder, and S. Gil, âActive rendezvous for multi-robot pose graph optimization using sensing over wi-fi,â in Robotics Research, T. Asfour, E. Yoshida, J. Park, H. Christensen, and O. Khatib, Eds. Cham: Springer International Publishing, 2022, pp. 832â 849.

[16] N. Jadhav, W. Wang, D. Zhang, O. Khatib, S. Kumar, and S. Gil, âA wireless signal-based sensing framework for robotics,â The International Journal of Robotics Research, vol. 41, no. 11-12, pp. 955â992, 2022.

[17] N. Jadhav, W. Wang, D. Zhang, S. Kumar, and S. Gil, âToolbox release: A wifi-based relative bearing framework for robotics,â in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022, pp. 13 714â13 721.

[18] W. Chen, X. Wang, S. Gao, G. Shang, C. Zhou, Z. Li, C. Xu, and K. Hu, âOverview of multi-robot collaborative slam from the perspective of data fusion,â Machines, vol. 11, no. 6, 2023.

[19] X. Pan, Z. Lai, S. Song, and G. Huang, âActivenerf: Learning where to see with uncertainty estimation,â in European Conference on Computer Vision, 2022.

[20] K. Lee, S. Gupta, S. Kim, B. Makwana, C. Chen, and C. Feng, âSo-nerf: Active view planning for nerf using surrogate objectives,â arXiv preprint arXiv:2312.03266, 2023.

[21] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â 2020.

[22] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion Â¨ revisited,â in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[23] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, jun 2018, pp. 586â595.