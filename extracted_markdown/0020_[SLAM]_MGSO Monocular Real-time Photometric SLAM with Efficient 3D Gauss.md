# MGSO: Monocular Real-time Photometric SLAM with Efficient 3D Gaussian Splatting

Yan Song Hu1, Nicolas Abboud1,2, Muhammad Qasim Ali1, Adam Srebrnjak Yang1, Imad Elhajj2, Daniel Asmar2, Yuhao Chen1, John S. Zelek1

<!-- image-->  
Fig. 1: Qualitative renders of the TUM-RGBD dataset [1] with input point clouds. By initializing 3D Gaussian Splatting (3DGS) [2] with dense, structured point clouds, MGSO produces reconstructions that are memory-efficient and high quality.

Abstractâ Real-time SLAM with dense 3D mapping is computationally challenging, especially on resource-limited devices. The recent development of 3D Gaussian Splatting (3DGS) offers a promising approach for real-time dense 3D reconstruction. However, existing 3DGS-based SLAM systems struggle to balance hardware simplicity, speed, and map quality. Most systems excel in one or two of the aforementioned aspects but rarely achieve all. A key issue is the difficulty of initializing 3D Gaussians while concurrently conducting SLAM. To address these challenges, we present Monocular GSO (MGSO), a novel real-time SLAM system that integrates photometric SLAM with 3DGS. Photometric SLAM provides dense structured point clouds for 3DGS initialization, accelerating optimization and producing more efficient maps with fewer Gaussians. As a result, experiments show that our system generates reconstructions with a balance of quality, memory efficiency, and speed that outperforms the state-of-the-art. Furthermore, our system achieves all results using RGB inputs. We evaluate the Replica, TUM-RGBD, and EuRoC datasets against current live dense reconstruction systems. Not only do we surpass contemporary systems, but experiments also show that we maintain our performance on laptop hardware, making it a practical solution for robotics, A/R, and other real-time applications.

## I. INTRODUCTION

Simultaneous localization and mapping (SLAM) is a fundamental task in autonomous robot navigation. It is the process by which a robot constructs a map of an environment while concurrently keeping track of its own location. Accurate self-localization is an essential precursor for advanced mobile robot tasks. Traditionally, SLAM systems provide semantically-poor map representations that are efficient for localization and basic navigation but lack the details needed for complex tasks. For example, sparse point clouds are efficient for localization but lack the surface detail needed for robotic grasping. For these complex robotic tasks, dense, high-fidelity spatial data is increasingly important.

To meet this demand, SLAM systems have evolved to generate dense 3D maps while still simultaneously performing localization. Dense SLAM systems are categorized into two approaches: decoupled and coupled. Decoupled approaches separate tracking from reconstruction, using a traditional SLAM system to provide outputs for a dense reconstruction process. Coupled approaches integrate dense reconstruction with both mapping and tracking, improving map quality but often facing speed bottlenecks, as accurate localization depends on building a high-quality map, which takes time.

A key challenge in decoupled systems is the lack of synergy between SLAM and dense reconstruction components. SLAM algorithms often fail to provide optimal data for high-quality dense reconstruction, compromising overall system performance. To address this challenge, we tailored our SLAM system to meet the specific needs of 3D Gaussian Splatting (3DGS) [2]. 3DGS typically requires an initial point cloud to begin reconstruction, with denser, well-structured initial point clouds leading to improved and faster results [3]. However, traditional feature-based SLAM methods produce sparse point clouds that are not optimal for 3DGS initialization. While RGB-D data could provide dense and accurate point clouds, using a monocular camera is preferable for wider applicability.

In this paper, we introduce Monocular-GSO (MGSO), a dense visual SLAM system that performs high-quality online 3D reconstruction in real-time using a single monocular camera. MGSO is a decoupled system that employs photometric SLAM to initialize a 3D Gaussian Splatting (3DGS) module running in parallel, enabling live dense scene reconstruction. The MGSO acronym is a blend of Direct Sparse Odometry (DSO) [4], the photometric SLAM system we built upon, and Gaussian Splatting (GS). In contrast to conventional featurebased SLAM methods that generate sparse point clouds, MGSO is designed to track a dense set of pixels, yielding a denser and well-structured point cloud output. We leverage this dense, structured point cloud to initialize 3D Gaussian Splatting (3DGS) in unmapped areas. Initializing with a high-quality set of points accelerates 3DGS optimization, guiding it toward more compact reconstructions with fewer artifacts and redundancies. As a result, our approach leads to real-time reconstruction with dense 3D maps with high quality and memory compactness.

The main contributions of MGSO are as follows:

â¢ A real-time dense SLAM system that harnesses the synergy between photometric SLAM and 3DGS.

â¢ Our system only requires a monocular camera.

â¢ Experiments show that our system has a combination of speed, map quality, and memory efficiency unmatched by other dense SLAM systems.

## II. RELATED WORK

Dense SLAM research has long explored various 3D representations such as signed distance functions [5], dense point clouds [6], and surfel clouds [7][8]. Despite these advancements, efficiently generating high-quality maps for realtime applications remains challenging. Recent innovations in Neural Radiance Fields (NeRFs) [9] and 3D Gaussian Splatting [2] show promise in addressing the issue. These approaches offer high-quality representations that are easy to create and render. Consequently, this section will focus on systems based on these two techniques.

NeRFs represent scenes using a neural network that outputs novel views based on the input cameraâs position and rotation. They also allow for the incremental learning and updating of their 3D representations through gradient-based optimization [9]. This capability has been effectively applied in pioneering works like iMap [10], and further improved by subsequent systems such as NICE-SLAM [11], Orbeez-SLAM [12], and NeRF-SLAM [13]. However, NeRF-based SLAM systems face two notable challenges: they require predefined scene bounds, which is often impractical in exploratory environments; and their implicit scene representations can be difficult to integrate with other systems. These limitations have spurred the adoption of 3DGS as a more suitable dense reconstructor.

3DGS is an approach to scene representation that models the environment as a large set of 3D Gaussians, which resemble blurry overlapping clouds. Rendering images involves projecting Gaussians onto the camera plane, depthsorting, and blending them front-to-back. Similar to NeRF,

3DGS allows for gradient-based optimization of parameters by minimizing the discrepancy between rendered and input images. The method will also heuristically clone or prune Gaussians over time. 3DGS offers a boundless and fast-torender representation compared to NeRF, making it highly suitable for real-time SLAM applications.

Early 3DGS-based SLAM systems, like MonoGS [14], SplaTAM [15], GS-SLAM [16], and Gaussian-SLAM [17], utilize a one-stage approach where tracking and mapping are tightly coupled. This approach introduces a dependency on map refinement before tracking can proceed, which results in slow performance, as shown in Table I. Even newer coupled systems such as CG-SLAM [18], RTG-SLAM [19], and SplatSLAM [20] struggle to run at speeds faster than 20 fps. To allow dense 3DGS-SLAM to operate faster, twostage systems like Photo-SLAM [21], IG-SLAM [22] and GS-ICP [23] emerged, decoupling the tracking and mapping functions. Furthermore, the majority of current 3DGS systems heavily rely on depth data to perform 3D reconstruction (Table I), making them dependant on RGB-D sensors.

TABLE I: Existing 3DGS SLAM Systems
<table><tr><td>Name</td><td>Type</td><td>Possible Sensors</td><td>FPS</td></tr><tr><td>MonoGS [14]</td><td>Coupled</td><td>RGB,RGB-D</td><td>&lt;5</td></tr><tr><td>SplaTAM [15]</td><td>Coupled</td><td>RGB-D</td><td>&lt;5</td></tr><tr><td>GS-SLAM [16]</td><td>Coupled</td><td>RGB-D</td><td>&gt;5, &lt;10</td></tr><tr><td>Gaussian-SLAM [17]</td><td>Coupled</td><td>RGB-D</td><td>&lt;5</td></tr><tr><td>CG-SLAM [18]</td><td>Coupled</td><td>RGB-D</td><td>&gt;15, &lt;20</td></tr><tr><td>RTG-SLAM [19]</td><td>Coupled</td><td>RGB-D</td><td>&gt;15, &lt;20</td></tr><tr><td>SplatSLAM [20]</td><td>Coupled</td><td>RGB</td><td>&lt;5</td></tr><tr><td>GS-ICP [23]</td><td>Decoupled</td><td>RGB-D</td><td>&gt;30</td></tr><tr><td>Photo-SLAM [21]</td><td>Decoupled</td><td>RGB*,RGB-D</td><td>&gt;30</td></tr><tr><td>IG-SLAM [22]</td><td>Decoupled</td><td>RGB</td><td>&gt;5, &lt;10</td></tr><tr><td>MGSO (ours)</td><td>Decoupled</td><td>RGB</td><td>&gt;30</td></tr></table>

\*Both monocular and stereo

Our system, MGSO, improves on existing two-stage 3DGS-based SLAM systems while utilizing only RGB data. It operates at 30 fps or higher, a performance matched only by Photo-SLAM and GS-ICP (see Table I). While MGSO is most similar to Photo-SLAM, which combines 3DGS with ORBSLAM3, we address Photo-SLAMâs tendency to create large, memory-inefficient maps. GS-ICP offers exceptional speed but requires depth data to initialize its iterative closest point tracking, whereas our system operates using only RGB data. Unlike IG-SLAM [22], which uses pseudo-depth RGB-D data at the cost of slower performance, MGSO maintains real-time speeds while generating accurate, compact maps.

## III. METHOD

MGSO integrates two core components that operate concurrently: a SLAM module responsible for accurate pose estimation, and a 3D dense reconstruction module for mapping.

## A. SLAM module:

The tracking backbone of our system is built upon a lineage of visual SLAM approaches originating from Direct Sparse Odometry (DSO) [4]. DSOâs key innovation is demonstrating that selective pixel sampling for photometric tracking enhances localization accuracy compared to using all pixels in an image. We chose to build upon DSO because we found that its pixel selection strategy also aligns well with initializing 3DGS. DSO tracks a set of pixels across consecutive frames i and j, optimizing the camera pose (p) by minimizing the photometric loss equation below for each pixel tracked,

$$
E = \left\| { ( I _ { j } [ { \pmb p } _ { j } ] - b _ { j } ) - \frac { s _ { j } a _ { j } } { s _ { i } a _ { i } } ( I _ { i } [ { \pmb p } _ { i } ] - b _ { i } ) } \right\|\tag{1}
$$

where I queries the pixel intensity, a and b are variables to account for lighting changes, and s is the camera exposure. The basic principle of this loss equation is to identify pose changes that best match the pixel intensity variation between consecutive frames i and j. The equation is applied at both tracking and mapping levels.

At each frame, our systemâs tracking process calculates pose changes relative to the latest keyframe, assuming a fixed map. The map of tracked pixels is only adjusted when a keyframe is inserted. A new keyframe is a reference frame that captures a distinct view of the scene relative to existing keyframes. When mapping is done, all current keyframe poses and the map, which consists of the tracked pixel points, are adjusted. Our system then converts the map of tracked pixels into a point cloud map and adds it, along with the keyframe poses, to the dense reconstruction module. We adopt DSOâs windowed keyframe management strategy, which generates keyframes when significant changes in the field of view, rotation, or lighting are detected. Older keyframes are removed if the number of keyframes exceeds the window size, which by default is eight, using a distancebased score to ensure a well-distributed set of keyframes in 3D space.

<!-- image-->  
Fig. 3: Comparison of 3DGS point clouds from MGSO and Photo-SLAM on Replica room0. Top left: original frame; top right: Map from original 3DGS after 10,240 iterations with Gaussian size set to 0.1. Bottom: MGSO vs. Photo-SLAM point clouds.

The inspiration for our method is from analyzing the final 3DGS Gaussian position of the original 3DGS (Figure 3). We realized that the final position, colour, and distribution of Gaussians of the final map resembled the point cloud output from DSO (Figure 3). From this observation, we conjectured that initializing 3DGS with photometric SLAM would enhance 3DGS optimization because it would reduce the required optimizations.

<!-- image-->  
Fig. 4: Tracked points from ORBSLAM3 (left) compared to our system (right). Our system tracks much more points than ORBSLAM3, which results in denser point clouds outputs.

A major aspect of DSOâs well-structured dense point cloud is itâs pixel selection strategy. DSO does pixel selection by dividing the image into blocks and selecting the highestgradient pixel above a gradient threshold in each block. It then repeats the process with a lower threshold and larger blocks. This approach not only tracks more pixels in complex areas but also ensures pixel selection in simpler regions. It differs from traditional methods, which typically only track easily recognizable features such as corners and edges. The differences between the two approaches can be observed in Figure 4. This is important because we observed that while completed 3DGS maps have more Gaussians in complex areas, they still maintain some Gaussians in noncomplex areas. Furthermore, DSO tracks pixels with high gradients, which are much more common than trackable feature points. Consequently, DSOâs output point cloud more closely matches the density of completed 3DGS maps. Our experiments revealed that whileDSOâs pixel selection density is optimal for tracking, increasing the pixel selection density enhances 3DGS performance, particularly in low-gradient areas that are challenging for tracking. To address this issue, we modified DSO to include additional tracked pixels not used for pose estimation to increase the output point cloud density (Figure 5). This modification allows the system to have the optimal pixel density for both tracking and 3DGS. Despite these enhancements, flat regions with minimal to no gradients remain sparsely populated with tracked pixels. This is because DSOâs pixel tracking system requires at least some gradient for tracking, and as a result, pixels in areas with no gradient are never tracked.

<!-- image-->  
Fig. 5: Original DSO Point cloud (left) compared to our systemâs point cloud (right). Our system has much more output points, especially in flat low-gradient regions.

We observed that 3DGS performs better with slightly misplaced initialized points in flat areas than with none at all. Therefore, we implemented an interpolation method that estimates point locations in low-gradient regions based on nearby tracked pixels. Our method employs the Delaunay triangulation algorithm [24] to divide the image into a series of triangles using tracked pixels as vertices. The depth of each interpolated point is calculated as the average depth of the triangleâs vertices, which generally provides accurate results for pixels on flat surfaces. While feature-based systems like Photo-SLAM also interpolate inactive 2D feature points, our method outperforms theirs due to a higher initial point count and by focusing interpolation on flat areas where itâs most accurate, which can be observed in Figure 3.

## B. Dense Reconstruction

MGSO employs 3DGS as its dense reconstruction method. Following the original 3DGS, we map the scene using a set of anisotropic Gaussians G. Each Gaussian $G _ { i }$ is modeled with an opacity, rotation, location, scale, and color. We follow Mono-GSâ [14] technique of representing color using RGB instead of spherical harmonics because Mono-GS showed this increased speed for minimal impact on reconstruction quality. We render RGB images of the map using the original differentiable tile-based rasterization introduced in 3DGS.

The parameters of each Gaussian are optimized using gradient descent to minimize the photometric loss L:

$$
L = | I _ { r } - I _ { g t } | ( 1 - \lambda ) + S S I M ( I _ { r } , I _ { g t } ) \lambda\tag{2}
$$

where $I _ { r }$ denotes the rendered image, $I _ { g t }$ refers to the captured image, Î» is a weighting factor and SSIM [25] represents the structural similarity metric.

In order to improve the speed of our system, we employ Gaussian-pyramid based learning introduced in Photo-SLAM [21] to progressively train the Gaussian map. The pyramid helps accelerate training for live video scenarios. A multiscale Gaussian pyramid is created by repeatedly smoothing and down-sampling ground-truth image captured by the camera. The photometric loss calculation progresses from using the highest pyramid level for $I _ { g t }$ in initial iterations to lower levels as training advances. Furthermore, we use an optimized version of the 3DGS CUDA back-end [26] that is faster than the original.

Our adaptive control strategy periodically densifies and prunes Gaussians every 1000 training iterations to improve map quality over time. We design our strategy around the point clouds returned by our SLAM module, similar to how Photo-SLAM tailored their strategy to ORBSLAM3 [27]. The point clouds generated by our SLAM system are characterized by their high density and uniform coverage. They adapt to the sceneâs complexity, concentrating points in intricate areas while maintaining representation in simpler regions. When a novel keyframe is processed, we initialize new Gaussians with location and color taken from the point cloud created by the SLAM system.

However, we noticed the emergence of floaters when utilizing the 3DGSâs densification and pruning strategies. To mitigate the presence of floaters, we utilized the adaptive control strategies in AbsGS [28]. Thus, as part of our adaptive control strategy we periodically densify Gaussians with high homo-directional view-space position gradients by splitting or cloning them. Large, high-variance Gaussians are split, while small Gaussians in under-reconstructed regions are cloned. Furthermore, we periodically prune Gaussians with low opacity to remove transparent floaters. We use the same splitting and cloning parameters as original 3DGS.

## IV. EXPERIMENTS AND DISCUSSIONS

We evaluate MGSO against the latest state-of-the-art 3DGS dense SLAM systems: MonoGS [14], GlORIE-SLAM [20], Splat-SLAM [15], IG-SLAM [22], and Photo-SLAM [21], to demonstrate our systemâs combination of highquality reconstruction, efficient runtime, and compact maps.

## A. Implementation and Setup

Datasets: Evaluations are done on the sythetic Replica [29] dataset and real-life EuRoC MAV [30] and TUM-RGBD [1] datasets. These datasets are commonly used to evaluate dense SLAM systems.

Hardware: Results for Replica and re-testing Photo-SLAM were done on an Intel i9-14900K with a NVIDIA RTX 4090. The laptop runs for Replica were done on an Intel i7-12700H with a NVIDIA GeForce RTX 3080 mobile. The results for EuRoC and TUM were done on an Intel i5- 12600KF with a NVIDIA RTX 3090.

Experimental Setup: We utilize the default optimization configuration of 3DGS with the exception of adjusting the densification interval to 1000. We configure the SLAM module (DSO [4]) to the default tracking settings. The parameters for increasing SLAM output point density through the inclusion of untracked points were determined through iterative testing.

For our experiments on the EuRoC MAV dataset, we implemented a preprocessing step involving undistorting and cropping the images before inputting them into the SLAM systems. This procedure was necessary to resolve the challenge of aligning poses between the undistorted 3DGS map and the distorted ground truth images. We also re-evaluated Photo-SLAM using this modified dataset, and notably, our new tests showed significant improvements to previously reported performance (Table IV).

Evaluation: To ensure a fair comparison, we evaluated our system by inputting the output 3DGS maps and pose estimation data into the original 3DGS rendering and metric scripts. We evaluate our reconstructions with the standard image quality metrics: PSNR, SSIM [25], and LPIPS [31]. Using an third-party evaluation system rather than built-in metrics offers a more realistic assessment, accounting for real-world factors like potential misalignment between the poses and map. Consistent with other dense SLAM systems, we evaluate on every fifth frame. Photo-SLAMâs evaluations are updated using this methodology to ensure consistency. We did ten runs for the Replica and EuRoC dataset and five runs for the TUM-RGBD dataset. Results for other systems were obtained from their respective publications, with the exception of Mono-GS, whose results were sourced from Splat-SLAM [20].

TABLE II: Reconstruction Results on Replica (cm)
<table><tr><td>Method</td><td>metric</td><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>r0</td><td>r1</td><td>r2</td><td>Avg.</td><td>Map Size</td><td>FPS</td></tr><tr><td rowspan="3">Photo-SLAM</td><td>PSNR[dB] â</td><td>35.22</td><td>34.35</td><td>29.58</td><td>28.55</td><td>32.05</td><td>26.75</td><td>27.78</td><td>29.43</td><td>30.46</td><td rowspan="3">22.5 MB</td><td rowspan="3">&gt;30</td></tr><tr><td>SSIM â</td><td>0.94</td><td>0.93</td><td>0.91</td><td>0.89</td><td>0.92</td><td>0.79</td><td>0.84</td><td>0.89</td><td>0.89</td></tr><tr><td>LPIPSâ</td><td>0.21</td><td>0.23</td><td>0.26</td><td>0.26</td><td>0.22</td><td>0.31</td><td>0.28</td><td>0.24</td><td>0.25</td></tr><tr><td rowspan="3">MGSO</td><td>PSNR[dB]</td><td>35.85</td><td>37.15</td><td>29.19</td><td>30.44</td><td>30.08</td><td>27.71</td><td>29.50</td><td>31.33</td><td>31.41</td><td></td><td rowspan="3">30</td></tr><tr><td>SSIM â</td><td>0.94</td><td>0.94</td><td>0.90</td><td>0.90</td><td>0.91</td><td>0.79</td><td>0.86</td><td>0.91</td><td>0.89</td><td>4.6 MB</td></tr><tr><td>LPIPSâ</td><td>0.22</td><td>0.25</td><td>0.29</td><td>0.26</td><td>0.26</td><td>0.33</td><td>0.27</td><td>0.24</td><td>0.27</td><td></td></tr><tr><td rowspan="3">MGSO (laptop)</td><td>PSNR[dB]</td><td>36.34</td><td>38.20</td><td>28.90</td><td>30.27</td><td>31.41</td><td>28.11</td><td>30.04</td><td>31.89</td><td>31.90</td><td></td><td rowspan="3">30</td></tr><tr><td>SSIM â</td><td>0.95</td><td>0.96</td><td>0.90</td><td>0.91</td><td>0.93</td><td>0.82</td><td>0.87</td><td>0.92</td><td>0.91</td><td>5.2 MB</td></tr><tr><td>LPIPSâ</td><td>0.24</td><td>0.25</td><td>0.31</td><td>0.27</td><td>0.25</td><td>0.35</td><td>0.29</td><td>0.26</td><td>0.28</td><td></td></tr></table>

However, the evaluation process tends to favor slower systems, as 3DGS performs better with extended training times, which may create bias against faster systems. Therefore, readers should consider the differences in speed when interpreting results. Because our system inherits real-time constraint handling from DSO, we decided to constrain our speed to the real-time speeds of videos to enhance the realism of the results. Replica and TUM-RGBD were run at 30 fps while EuRoC was run at 20 fps.

## B. Discussion

TABLE III: Absolute Trajectory Error of Tracking on Replica (RMSE in cm)
<table><tr><td>Method</td><td>r0</td><td>r1</td><td>r2</td><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>Avg.</td></tr><tr><td>Photo-SLAM</td><td>0.58</td><td>0.32</td><td>5.03</td><td>0.47</td><td>0.58</td><td>0.35</td><td>1.18</td><td>0.23</td><td>1.09</td></tr><tr><td>MGSO</td><td>0.35</td><td>1.02</td><td>5.93</td><td>0.22</td><td>0.54</td><td>0.28</td><td>0.34</td><td>0.2</td><td>1.11</td></tr></table>

1) Localization: While we include tracking results in Table III, tracking performance is not our systemâs focus. We did not modify the localization aspect of DSO and should inherit its performance. Our systemâs comparable tracking to Photo-SLAM suggests any rendering differences are not due to localization.

2) Reconstruction Quality: MGSO consistently achieves high PSNR and SSIM across all datasets. In the Replica dataset (Table II), MGSO outperforms Photo-SLAM with a PSNR of 31.406 dB and a much smaller map size of 4.618 Mb, with its mobile version showing even better results (31.896 dB PSNR, 0.906 SSIM). On the EuRoC dataset (Table IV), MGSO further demonstrates superior performance with 22.10 dB PSNR and 0.80 SSIM, compared to Photo-SLAMâs 19.68 dB and 0.75 SSIM. Similar trends are observed on the TUM dataset (Table V), where MGSO achieves a higher PSNR and SSIM than Photo-SLAM. MGSOâs key advantage is its ability to generate dense, well-structured point clouds, requiring less refinement and resulting in more compact mapsâhalf the size of Photo-SLAMâs. This efficient initialization reduces the need for extensive operations like cloning and pruning, leading to faster convergence and fewer reconstruction artifacts. Figure 6 further highlights

MGSOâs improved rendering of flat surfaces, fewer floating artifacts, and better preservation of edges and thin features, showcasing its capacity to handle complex scenes with more accurate and detailed reconstructions.

TABLE IV: Reconstruction Results on EuRoC
<table><tr><td>Method</td><td>metric</td><td>MH</td><td>V1</td><td>V2</td><td>Avg.</td><td>Mem.</td></tr><tr><td rowspan="2">Photo- SLAM</td><td>PSNR[dB] â</td><td>18.60</td><td>18.30</td><td>17.94</td><td>18.28</td><td rowspan="2">111.8</td></tr><tr><td>SSIM â</td><td>0.65</td><td>0.73</td><td>0.65</td><td>0.68</td></tr><tr><td rowspan="4">MGSO</td><td>LPIPSâ PSNR[dB] â</td><td>0.39</td><td>0.44</td><td>0.53</td><td>0.46</td><td></td></tr><tr><td></td><td>20.75</td><td>20.26</td><td>20.31</td><td>20.44</td><td rowspan="3">8.3</td></tr><tr><td>SSIM â</td><td>0.72</td><td>0.79</td><td>0.75</td><td>0.76</td></tr><tr><td>LPIPSâ</td><td>0.36</td><td>0.39</td><td>0.39</td><td>0.38</td></tr></table>

Mem. is average map size in Mb

TABLE V: Reconstruction Results on TUMâs
<table><tr><td>Method</td><td>metric</td><td>fr1</td><td>fr2</td><td>fr3</td><td>Avg.</td></tr><tr><td rowspan="3">Photo-SLAM</td><td>PSNR[dB] â</td><td>18.01</td><td>16.93</td><td>17.11</td><td>17.35</td></tr><tr><td>SSIM â</td><td>0.65</td><td>0.60</td><td>0.62</td><td>0.63</td></tr><tr><td>LPIPSâ</td><td>0.41</td><td>0.41</td><td>0.42</td><td>0.41</td></tr><tr><td rowspan="3">MGSO</td><td>PSNR[dB] â</td><td>18.07</td><td>24.10</td><td>21.61</td><td>21.26</td></tr><tr><td>SSIM â</td><td>0.66</td><td>0.80</td><td>0.75</td><td>0.74</td></tr><tr><td>LPIPSâ</td><td>0.45</td><td>0.33</td><td>0.38</td><td>0.39</td></tr></table>

3) Resource Efficiency and Real-Time Performance: MGSO excels in low memory usage and real-time FPS. On the EuRoC dataset (Table IV), MGSO requires only 8.32 MB, significantly less than Photo-SLAMâs 109.73 MB, and just 2.85 MB on the TUM dataset (Table V), compared to Photo-SLAMâs 17 MB. All the while, MGSO maintains realtime performance (Tables II,VI) MGSOâs structured point clouds allow it to create compact maps with minimal redundant elements, resulting in lower memory consumption. This contrasts with Photo-SLAMâs larger map sizes, which require more refinement. Figure 7 underscores MGSOâs balance of high FPS with low map size, we are the only system capable of real-time performance with compact maps.

## C. Ablations

ted experiments to evaluate the robustness of our system to the frequency of densification and pruning. As shown in table VII, increasing the rate of densification does not improve reconstruction results and instead reduces the compactness of the final 3DGS map. In fact, at high densification rates, the reconstruction quality diminishes. The observed robustness suggests that our system generates spatially accurate point clouds that effectively capture both complex and simple areas of the scene, without requiring significant adjustments from densification or pruning.

<!-- image-->  
(a) MGSO renders flat surfaces well

<!-- image-->  
(b) MGSO has less floaters and artifacts

<!-- image-->  
(c) MGSO has better edges on difficult scene

<!-- image-->  
(d) MGSO renders features better

<!-- image-->  
(e) MGSO has less floaters and artifacts  
Fig. 6: Comparison of difficult novel view renders between MGSO (top) and Photo-SLAM (bottom). Captions describe how MGSO performs better.

TABLE VI: Replica Aggregated Results
<table><tr><td>Method</td><td>PNSR[dB]</td><td>Map Size</td><td>FPS</td><td>GPU Usage</td></tr><tr><td>GIORIE-SLAM (GIS)</td><td>31.04</td><td>114 Mb</td><td>0.23</td><td>15.22</td></tr><tr><td>Mono-GS (MGS)</td><td>31.22</td><td>6.8 Mb</td><td>0.32</td><td>14.62</td></tr><tr><td>Splat-SLAM (SpS)</td><td>36.45</td><td>6.8 Mb</td><td>1.24</td><td>17.57</td></tr><tr><td>IG-SLAM (IGS)</td><td>36.21</td><td>14.8 Mb</td><td>9.94</td><td>16.20</td></tr><tr><td>Photo-SLAM (PhS)</td><td>30.46</td><td>22.5 Mb</td><td>&gt;30*</td><td>3.62</td></tr><tr><td>MGSO (MGSO)</td><td>31.41</td><td>4.3 Mb</td><td>30*</td><td>7.98</td></tr></table>

\*System processed data as fast as inputted video stream

<!-- image-->  
Fig. 7: Plot of Table VI. The âxâ in the legend represents the frames per second. We consider fps>24 (cinema fps standard) as real-time.

Table VIII demonstrates the importance of dense, wellstructured inputs to 3D Gaussians. We deliberately reduced the density of our point clouds by utilizing only half of the points from our tracking system and excluding additional untracked points. This intentional sparsification resulted in a marked decline in reconstruction quality.

TABLE VII: Densify Iteration Ablation
<table><tr><td>Scene</td><td>metric</td><td>1024</td><td>512</td><td>256</td><td>128</td><td>64</td><td>32</td></tr><tr><td>00</td><td>PSNR[dB]â</td><td>37.06</td><td>37.10</td><td>37.02</td><td>37.15</td><td>37.40</td><td>36.11</td></tr><tr><td></td><td>Memory(Mb) PSNR[dB]â</td><td>6.4 28.84</td><td>7.2 28.73</td><td>9.0 28.86</td><td>13.0 28.76</td><td>21.4 28.49</td><td>38.1 27.73</td></tr><tr><td>r0</td><td>Memory(Mb)</td><td>4.5</td><td>5.3</td><td>7.7</td><td>12.4</td><td>22.2</td><td>38.5</td></tr></table>

Ablation Evaluations done on training images instead of test images

TABLE VIII: Additional Dense Points Ablation
<table><tr><td>Dataset</td><td>metric</td><td>00</td><td>01</td><td>02</td><td>r0</td><td>r1</td></tr><tr><td>Base</td><td>PSNR[dB]â Memory(Mb)</td><td>37.06 6.4</td><td>38.37 4.2</td><td>29.81 6.0</td><td>28.84 4.5</td><td>30.68 5.3</td></tr><tr><td>Halved</td><td>PSNR[dB]â Memory(Mb)</td><td>33.48 2.0</td><td>33.93 1.7</td><td>27.87 2.2</td><td>27.30 2.1</td><td>29.13 2.0</td></tr></table>

Ablation evaluations done on training images instead of test images

## V. CONCLUSIONS

MGSO integrates real-time photometric SLAM with 3D Gaussian Splatting (3DGS) to achieve dense, high-quality, and memory efficient 3D reconstruction using only a monocular camera. Our approach addressed several challenges in order to harness the natural compatibility of these two techniques. Its proven versatility across various environments without the use of depth sensors makes it optimal for robotics, AR/VR, and digital twin applications. Future research could explore implementing loop closure for global consistency and real-time re-rendering for adaptive scene reconstruction, enhancing MGSOâs precision and efficiency in complex, large-scale environments.

## ACKNOWLEDGMENT

This work was supported by the Natural Sciences and Engineering Research Council (NSERC) (grant-number, www.nserc-crsng.gc.ca) and the DIDYMOS-XR Horizon Europe project (grant number 101092875âDIDYMOS-XR,www.didymos-xr.eu).

[1] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of rgb-d slam systems,â in Proc. of the International Conference on Intelligent Robot Systems (IROS), Oct. 2012.

[2] B. Kerbl, G. Kopanas, T. Leimk Â¨uhler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[3] Y. S. Hu, D. Mao, Y. Chen, and J. Zelek, âTowards real-time gaussian splatting: Accelerating 3dgs through photometric slam,â 2024. [Online]. Available: https://arxiv.org/abs/2408.03825

[4] J. Engel, V. Koltun, and D. Cremers, âDirect sparse odometry,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 3, pp. 611â625, 2018.

[5] S. Izadi, D. Kim, O. Hilliges, D. Molyneaux, R. Newcombe, P. Kohli, J. Shotton, S. Hodges, D. Freeman, A. Davison, et al., âKinectfusion: real-time 3d reconstruction and interaction using a moving depth camera,â in Proceedings of the 24th annual ACM symposium on User interface software and technology, 2011, pp. 559â568.

[6] C. Kerl, J. Sturm, and D. Cremers, âDense visual slam for rgb-d cameras,â in 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2013, pp. 2100â2106.

[7] T. Sch Â¨ops, T. Sattler, and M. Pollefeys, âBad slam: Bundle adjusted direct rgb-d slam,â in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 134â144.

[8] T. Whelan, R. F. Salas-Moreno, B. Glocker, A. J. Davison, and S. Leutenegger, âElasticfusion: Real-time dense slam and light source estimation,â The International Journal of Robotics Research, vol. 35, no. 14, pp. 1697â1716, 2016.

[9] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â in ECCV, 2020.

[10] E. Sucar, S. Liu, J. Ortiz, and A. Davison, âiMAP: Implicit mapping and positioning in real-time,â in Proceedings of the International Conference on Computer Vision (ICCV), 2021.

[11] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[12] C.-M. Chung, Y.-C. Tseng, Y.-C. Hsu, X.-Q. Shi, Y.-H. Hua, J.-F. Yeh, W.-C. Chen, Y.-T. Chen, and W. H. Hsu, âOrbeez-slam: A real-time monocular visual slam with orb features and nerf-realized mapping,â in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 9400â9406.

[13] A. Rosinol, J. J. Leonard, and L. Carlone, âNerf-slam: Real-time dense monocular slam with neural radiance fields,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 3437â3444.

[14] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, âGaussian Splatting SLAM,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[15] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[16] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGsslam: Dense visual slam with 3d gaussian splatting,â in CVPR, 2024.

[17] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, âGaussian-slam: Photorealistic dense slam with gaussian splatting,â 2023.

[18] J. Hu, X. Chen, B. Feng, G. Li, L. Yang, H. Bao, G. Zhang, and Z. Cui, âCg-slam: Efficient dense rgb-d slam in a consistent uncertainty-aware 3d gaussian field,â arXiv preprint arXiv:2403.16095, 2024.

[19] Z. Peng, T. Shao, Y. Liu, J. Zhou, Y. Yang, J. Wang, and K. Zhou, âRtg-slam: Real-time 3d reconstruction at scale using gaussian splatting,â in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1â11.

[20] E. Sandstr Â¨om, K. Tateno, M. Oechsle, M. Niemeyer, L. V. Gool, M. R. Oswald, and F. Tombari, âSplat-slam: Globally optimized rgb-only slam with 3d gaussians,â 2024. [Online]. Available: https://arxiv.org/abs/2405.16544

[21] H. Huang, L. Li, C. Hui, and S.-K. Yeung, âPhoto-slam: Real-time simultaneous localization and photorealistic mapping for monocular, stereo, and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[22] F. A. Sarikamis and A. A. Alatan, âIg-slam: Instant gaussian slam,â 2024. [Online]. Available: https://arxiv.org/abs/2408.01126

[23] S. Ha, J. Yeon, and H. Yu, âRgbd gs-icp slam,â ArXiv, vol. abs/2403.12550, 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:268531141

[24] C. L. Lawson, âTransforming triangulations,â Discrete Mathematics, vol. 3, no. 4, pp. 365â372, 1972. [Online]. Available: https://www.sciencedirect.com/science/article/pii/0012365X72900933

[25] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, 2004.

[26] J. Patas, âGaussian splatting cuda,â https://github.com/MrNeRF/gaussian-splatting-cuda, 2023.

[27] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. M. Montiel, and J. D. Tard Â´os, âOrb-slam3: An accurate open-source library for visual, visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[28] Z. Ye, W. Li, S. Liu, P. Qiao, and Y. Dou, âAbsgs: Recovering fine details for 3d gaussian splatting,â 2024.

[29] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, A. Clarkson, M. Yan, B. Budge, Y. Yan, X. Pan, J. Yon, Y. Zou, K. Leon, N. Carter, J. Briales, T. Gillingham, E. Mueggler, L. Pesqueira, M. Savva, D. Batra, H. M. Strasdat, R. D. Nardi, M. Goesele, S. Lovegrove, and R. Newcombe, âThe Replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[30] M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. W. Achtelik, and R. Siegwart, âThe euroc micro aerial vehicle datasets,â The International Journal of Robotics Research, vol. 35, no. 10, pp. 1157â1163, 2016. [Online]. Available: https://doi.org/10.1177/0278364915620033

[31] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in CVPR, 2018.