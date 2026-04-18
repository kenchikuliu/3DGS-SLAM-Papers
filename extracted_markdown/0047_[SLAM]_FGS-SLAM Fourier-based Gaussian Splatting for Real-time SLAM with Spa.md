# FGS-SLAM: Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map Fusion

Yansong Xu1,2, Junlin Li1, Wei Zhang1, Siyu Chen1,2, Shengyong Zhang1, Yuquan Leng3â, Weijia Zhou1â

Abstractâ 3D gaussian splatting has advanced simultaneous localization and mapping (SLAM) technology by enabling realtime positioning and the construction of high-fidelity maps. However, the uncertainty in gaussian position and initialization parameters introduces challenges, often requiring extensive iterative convergence and resulting in redundant or insufficient gaussian representations. To address this, we introduce a novel adaptive densification method based on Fourier frequency domain analysis to establish gaussian priors for rapid convergence. Additionally, we propose constructing independent and unified sparse and dense maps, where a sparse map supports efficient tracking via Generalized Iterative Closest Point (GICP) and a dense map creates high-fidelity visual representations. This is the first SLAM system leveraging frequency domain analysis to achieve high-quality gaussian mapping in realtime. Experimental results demonstrate an average frame rate of 36 FPS on Replica and TUM RGB-D datasets, achieving competitive accuracy in both localization and mapping. The source code is publicly available at https://github.com/ 3DV-Coder/FGS-SLAM.

## I. INTRODUCTION

With the rapid development of fields such as robotics [1], augmented reality (AR), and drones, there is an increasing demand for efficient and accurate 3D environmental perception. As a result, the importance of Simultaneous Localization and Mapping (SLAM) technology in these applications has grown significantly. The main challenge of SLAM is achieving both autonomous localization and 3D map construction in unknown environments. However, traditional SLAM methods still face significant challenges in balancing real-time performance and scene accuracy.

In recent years, sparse and dense SLAM methods have emerged as key approaches. Sparse SLAM methods are computationally efficient and offer advantages in real-time performance [2], [3]. However, they represent the scene with fewer features, limiting the ability to reconstruct detailed environments. Dense SLAM methods [4], [5], [6], [7], on the other hand, create fine-grained environmental representations using high-density point clouds and voxel models.

<!-- image-->  
Fig. 1. FGS-SLAM adopts a map-sharing mechanism, jointly maintaining both a 3D gaussian dense map and a sparse map. The gaussian map provides excellent rendering performance from new viewpoints. The system operates at a speed at least one order of magnitude faster than other methods, achieving real-time frame rates, while ensuring accurate localization and high-fidelity map reconstruction quality.

However, their high computational complexity makes them less suitable for real-time applications. To overcome these limitations, researchers have explored neural network-based implicit dense map construction, such as Neural Radiance Fields (NeRF) SLAM. These methods use neural radiance fields to enhance scene detail [8], [9], [10], [11]. However, the high computational cost of volumetric rendering makes it difficult to achieve real-time performance, and the maps lack interpretability.

3D gaussian splatting (3DGS) is an explicit dense mapping method that represents the environment with gaussian distributions [12], [13]. This approach offers high rendering speed and better interpretability. SLAM systems based on 3DGS have significantly improved pose estimation efficiency and map quality, making it an important direction in explicit dense SLAM [14], [15], [13]. However, current 3DGS methods often struggle with the uncertainty in gaussian point initialization, leading to redundancy or under representation, which impacts mapping efficiency and accuracy. While existing 3DGS-based methods predominantly rely on spatial domain information for map construction, FreGS [16] mitigates over-reconstruction by introducing frequency regularization to supervise the gaussian optimization process in the spectral domain. In contrast to FreGS, our approach focuses on optimizing gaussian initialization and distribution strategies directly in the frequency domain.â

To address these challenges, this paper introduces FGS-SLAM, a method that adapts gaussian densification based on frequency domain analysis. For the first time, we use Fourier domain analysis for gaussian initialization, reducing redundant gaussian points and accelerating parameter convergence. Additionally, we propose an independent yet unified approach to constructing sparse and dense maps. The sparse map is used for efficient camera tracking, while the dense map enables high-fidelity scene representation. This method enables a SLAM system that balances real-time performance with high accuracy.

The main contributions of this paper are as follows:

â¢ Gaussian Initialization Strategy Guided by Frequency Domain: We propose a novel frequencydomain analysis-based gaussian initialization method, which accelerates the convergence of gaussian parameters and reduces redundant points, thus enhancing the efficiency of dense map construction.

â¢ Independent Unified Framework for Sparse and Dense Maps: We achieved efficient localization with sparse maps and high-fidelity reconstruction with dense maps, effectively balancing the real-time performance and detailed expression capabilities of the SLAM system.

â¢ Adaptive gaussian Density Distribution Strategy: By leveraging frequency domain analysis, we adaptively assign gaussian density and radius to different regions of the scene, effectively reducing computational complexity while ensuring scene accuracy.

## II. RELATED WORKS

## A. Sparse map SLAM

Sparse map SLAM typically focuses on real-time performance and computational efficiency by selecting a limited number of key feature points for camera tracking and pose estimation. Representative methods include ORB-SLAM2[2] and ORB-SLAM3 [3]. These methods reduce computational cost by extracting feature points to form sparse maps but are limited in their ability to express scene details. The Generalized Iterative Closest Point (GICP) algorithm [17] is a widely used sparse SLAM tracking method that relies on point cloud matching for accurate pose estimation, making it suitable for sparse map construction. Despite the broad application of ICP-based algorithms in sparse map SLAM, their accuracy is highly dependent on the density and quality of the map, which makes it challenging to achieve precise scene reconstruction in complex environments. The FGS-SLAM method proposed in this paper combines the efficiency of sparse maps with the high-fidelity scene representation of dense maps. By addressing the redundancy issue with frequencydomain-guided gaussian initialization, FGS-SLAM achieves a SLAM system that balances both real-time performance and precision.

## B. Dense map SLAM

Dense map SLAM, on the other hand, provides a complete representation of the environment through high-density point clouds or voxel modeling [11], [18]. Methods like ElasticFusion [5] and KinectFusion [19] construct dense maps using depth sensors, enabling high-quality scene reconstruction. However, these methods suffer from poor real-time performance and high computational resource requirements. While dense SLAM has made significant advancements in scene detail representation, it still faces challenges in realtime applications. This paper addresses these challenges by introducing an adaptive gaussian density distribution strategy, achieving efficient dense map construction while maintaining both high-fidelity scene representation and the real-time demands of the SLAM system.

## C. NeRF SLAM

The introduction of Neural Radiance Fields has allowed SLAM systems to achieve implicit dense scene representations, thereby enhancing the level of detail in 3D map reconstruction. iMAP [20] was the first real-time NeRF SLAM system that jointly optimized the 3D scene and camera poses through implicit neural networks. NICE-SLAM [8] further employed feature grids and multi-resolution strategies to accelerate the optimization of dense SLAM scenes while retaining NeRFâs fine-grained representation. Co-SLAM [9] improved NeRFâs performance in both detail expression and efficient optimization by using a multi-resolution hash grid structure. While NeRF-based SLAM methods enable highquality scene reconstruction, the computational burden of volume rendering and ray tracing leads to poor real-time performance, and the implicit representation reduces the interpretability of the map.

## D. 3DGS SLAM

3D gaussian splatting method was introduced to enable explicit dense map construction. 3DGS uses gaussian points to directly represent the scene, offering faster rendering speeds and better interpretability. Methods such as SplaTAM [14] and GS-SLAM [21] leverage the fast rendering capabilities of 3DGS to construct high-fidelity dense maps, significantly improving efficiency. However, the uncertainty in the initialization of gaussian points in 3DGS often leads to redundant or under-expressed gaussian points, which impacts mapping efficiency and accuracy. The FGS-SLAM method proposed in this paper improves on 3DGS by using frequency-domain information for gaussian initialization, allowing the parameters to converge more quickly, reducing redundancy, and improving the quality of the dense map, ultimately enhancing both localization and mapping accuracy in SLAM systems.

## III. METHOD

Fig. 2 illustrates the system framework. Both the dense and sparse maps are composed of 3D gaussians. The gaussian set possesses the following properties: $G \{ \mu _ { i } , S _ { i } , R _ { i } , \alpha _ { i } , c _ { i } \} , ( i = 1 , \ldots , N )$ . Each gaussian consists of a position $\mu _ { i }$ , scale $S _ { i }$ , rotation $R _ { i }$ , opacity $\alpha _ { i }$ , and color $\mathrm { c } _ { i } .$ The relationship between the scale $S$ and covariance C is given by $C \overset { \cdot } { = } R S S ^ { \mathrm { T } } R ^ { \mathrm { T } }$ . The 3D gaussian splatting rendering process is based on alpha blending, which achieves the 2D projection. In this paper, we describe the rendering process using the following equation:

$$
\mathcal { F } _ { p } = \sum _ { i = 1 } ^ { n } \gamma _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{1}
$$

where $\mathcal { F } p$ represents the rendered value of the pixel formed by the combination of n gaussians splats, and $\gamma _ { i }$ denotes the contribution coefficient of the i-th gaussian to the pixel rendering. This equation can be interpreted differently based on the type of rendering: 1) Color Rendering: Here, $\gamma _ { i }$ indicates the color of each gaussian $c _ { i } ,$ , with the equation capturing the cumulative color contribution. 2) Depth Rendering: For depth rendering, $\gamma _ { i }$ denotes the depth of each gaussian $d _ { i } ,$ and the process accumulates depth values across gaussians. 3) Opacity Rendering: By setting $\gamma _ { i } = 1$ to represent opacity, the equation simplifies to model the accumulation of opacity, which is essential for computing visibility under a specific viewpoint.

Gaussian distribution assumptions in GICP and 3DGS share a common foundation, utilizing a shared gaussian set $G \{ \mu _ { i } , S _ { i } \} , ( i = 1 , \ldots , N )$ . This shared representation enables efficient tracking via GICP, which estimates poses based on sparse gaussian maps, and high-quality mapping via 3DGS, which optimizes and updates the map using these poses. The integration of GICP and 3DGS leverages 3D gaussian representations to achieve fast and accurate SLAM performance.

Functionally, the sparse and dense maps operate independently: the sparse map facilitates efficient camera tracking, while the dense map supports map reconstruction. Both maps are unified through shared gaussian attributes and joint optimization. Consistency in gaussian attribute optimization is maintained via joint rendering of points from both maps. The dense map adaptively initializes new gaussians, shares missing-region masks with the sparse map, and filters tracking points, ensuring resource sharing and optimization coherence.

## A. Mapping

Adaptive Gaussian Densification. The color change gradient corresponds to the frequency. The scene is composed of color change gradients at different frequencies. We observe that regions with small color change gradients, representing low-frequency areas, are expected to use sparse, large gaussian representations, while dense, small gaussian representations are more suitable for high-frequency areas. To address this, we propose adaptive gaussian densification. The Fourier transform is used to convert the spatial domain of the image frame $I ( x , y )$ into the frequency domain. The Fourier domain representation of the image is expressed as:

$$
F ( u , v ) = \sum _ { x = 0 } ^ { W - 1 } \sum _ { y = 0 } ^ { H - 1 } I ( x , y ) \cdot e ^ { - j 2 \pi ( \frac { u \cdot x } { W } + \frac { v \cdot y } { H } ) } ,\tag{2}
$$

where $F _ { c } ( u , v )$ represents the complex-valued function in the frequency domain at $( u , v )$ , and $I ( x , y )$ denotes the pixel values in the spatial domain. W and H are the width and height of the image, and u and v are the horizontal and vertical coordinates in the frequency domain. The Fourier centering is defined as:

$$
F _ { c } ( u , v ) = F ( u , v ) \cdot ( - 1 ) ^ { u + v } .\tag{3}
$$

The magnitude spectrum is given by:

$$
\left| F _ { c } ( u , v ) \right| = \sqrt { R e ( u , v ) ^ { 2 } + I m ( u , v ) ^ { 2 } } ,\tag{4}
$$

where $R e ( u , v )$ and $I m ( u , v )$ denote the real and imaginary parts of the complex-valued function $F _ { c } ( u , v )$ . Additionally, a gaussian filter is applied to the frequency domain image for frequency separation. The frequency domain transfer function of the gaussian high-pass filter is:

$$
H ( u , v ) = 1 - e ^ { - \frac { D ^ { 2 } ( u , v ) } { 2 D _ { 0 } ^ { 2 } } } ,\tag{5}
$$

where $D ( u , v )$ is the Euclidean distance from the pixel position $( u , v )$ to the filter center. $D _ { 0 }$ is the cutoff frequency of the filter, determining the strength of the high-pass filter.

After applying the gaussian high-pass filter, the frequency domain function $F _ { h } ( u , v )$ is:

$$
F _ { h } ( u , v ) = F _ { c } ( u , v ) \cdot H ( u , v ) .\tag{6}
$$

The Fourier inverse transform of the gaussian high-pass filter is also a gaussian function. This means that the inverse Fourier transform (IDFT) of the equation above results in a spatial gaussian filter that avoids ringing effects. This filter sets the low-frequency direct current component to zero, meaning the filtered result depends only on the sceneâs color gradient changes and is not influenced by the sceneâs color.

The high-frequency component-dominated image is reconstructed in the spatial domain after inverse Fourier transform from the filtered frequency domain image:

$$
\tilde { I _ { h } } ( x , y ) = \frac { 1 } { H \cdot W } \sum _ { x = 0 } ^ { H - 1 } \sum _ { y = 0 } ^ { W - 1 } F _ { h } ( u , v ) \cdot e ^ { j 2 \pi ( u \frac { x } { W } + v \frac { y } { H } ) } .\tag{7}
$$

Typically, the energy from low to high frequencies decreases overall. Therefore, the energy values obtained after gaussian high-pass filtering decrease from low to high, and the frequency histogram of the high-frequency image $\tilde { I _ { h } } ( x , y )$ presents a unimodal shape. Based on this observation, we use a triangular thresholding method to construct a triangle between the highest peak of the histogram and its endpoints, finding the point furthest from the baseline as the threshold. The high-frequency region is then segmented, and the low-frequency region is obtained by complement. We use equidistant sampling points with varying spacings in different frequency domains as gaussian sampling points, with the sampling interval in high-frequency regions being $m ,$ and in low-frequency regions being n where $( m < n )$ The resulting high-frequency region position mask is $M _ { h }$ The low-frequency region position mask is $M _ { l }$ after the inversion of $M _ { h }$

<!-- image-->  
Fig. 2. System Overview. The proposed method uses RGB-D data as input to the system. Mapping: The spatial domain is transformed into the frequency domain through Fourier transforms. New gaussians are adaptively initialized based on high and low frequency regions, thereby constructing a gaussian dense map. Resource Sharing: The system simultaneously constructs both sparse and dense maps, with map points stored using gaussian attributes. Gaussian attributes and the mask of missing gaussian regions are shared between the maps. Tracking: GICP performs rapid registration using the 3D gaussian point cloud of the sparse map, and supplements the gaussian points in the sparse map. The system selects co-visibility and random keyframes, and jointly optimizes the gaussian map through gaussian rasterization rendering.

regions:

Gaussian Missing Region Check Strategy. Incorporating all gaussians blindly into the map would result in gaussian explosion and redundancy. Therefore, we propose a gaussian missing region check strategy. The gaussian missing region refers to areas that were not observed in previous keyframes or regions where the gaussian map representation is insufficient. Using equation (1), we perform alpha blending for opacity rendering (equivalent to the accumulation of gaussian opacity). If the final rendering opacity is lower than the threshold, the area is considered insufficient in gaussian representation, resulting in the gaussian expression deficit mask $M _ { i }$ in the current frame. Additionally, considering the presence of foreground and background in the scene, if the background has already been mapped in previous keyframes but the foreground is present in the current frameâs viewpoint, simply using opacity rendering cannot detect the missing foreground. Thus, we introduce a depth mask $M _ { d }$ and a color mask $M _ { c }$ . The depth mask is generated by comparing the gaussian depth rendering with the current frameâs depth. Areas with an abnormally large depth difference are considered foreground. Similarly, the color mask compares the gaussian color rendering with the current frameâs color. Areas with significant color differences are treated as foreground. The final gaussian missing region mask $M _ { m }$ is:

$$
M _ { m } = M _ { i } \cup M _ { d } \cup M _ { c } .\tag{8}
$$

The final regions where gaussians are added are:

$$
\begin{array} { c } { { M _ { h } = M _ { h } \cap M _ { m } } } \\ { { M _ { l } = M _ { l } \cap M _ { m } } } \end{array} .\tag{9}
$$

Different gaussian radii are set for high and low-frequency

$$
r = \left\{ \begin{array} { l l } { \alpha _ { h } \cdot \frac { d } { f } } & { \mathrm { i f } \ G \in M _ { h } } \\ { \alpha _ { l } \cdot \frac { d } { f } } & { \mathrm { i f } \ G \in M _ { l } } \end{array} , \right.\tag{10}
$$

where the ratio of depth d to focal length f represents the gaussian scale based on the 3D gaussian projection onto a 2D image, with a radius of 1 pixel. $\alpha _ { h }$ and $\alpha _ { l }$ represent the gaussian scale factors for high-frequency and low-frequency regions, respectively, where $\alpha _ { h } \ < \ \alpha _ { l }$ means the gaussian scale for high-frequency regions is smaller than that for lowfrequency regions. Instead of performing gaussian splitting after gaussian initialization, new gaussians are adaptively added in the regions where gaussians are missing.

Gaussian Pruning. The gaussian pruning strategy includes two components. First, gaussian overgrowth in any dimension may cause artifacts in the gaussian map. Thus, we prune excessively large gaussians and use the mapping strategy to re-supplement them. Secondly, gaussians with low opacity contribute weakly to scene representation. To reduce redundancy, we prune gaussians with opacity below the threshold.

## B. Tracking

We construct a sparse gaussian map as the target point cloud for efficient tracking via Generalized Iterative Closest Point (GICP) alignment. To prevent tracking accuracy and speed degradation from excessive point clouds, we apply uniform downsampling on keyframes to build the sparse map in 3D space. The subsampling point of the current frame serves as the source point cloud, while the sparse gaussian map is the target point cloud for ICP tracking.

The sparse map is updated only on tracking keyframes. The gaussian addition strategy is similar to that of the dense map. The source point cloud is filtered through the gaussian missing region mask. It is then added to the sparse map.

Tracking is achieved through the GICP method, which optimizes a transformation matrix T to align the source point cloud $P ~ = ~ \{ p _ { 0 } , p _ { 1 } , . . . , p _ { M } \}$ with the target point cloud $Q = \{ q _ { 0 } , q _ { 1 } , . . . , q _ { N } \}$ . GICP models the local surfaces of the source point $p _ { i }$ and the target point $q _ { i }$ as gaussian distributions: $p _ { i } \sim \mathcal N ( \hat { p _ { i } } , C _ { p } ^ { i } )$ , $q _ { i } \sim \mathcal { N } ( \hat { q } _ { i } , C _ { q } ^ { i } )$ , where $C _ { p } ^ { i }$ and $C _ { q } ^ { i }$ i are the covariance matrices of the local regions of $p _ { i }$ and $q _ { i }$ . The registration error between the two point clouds is defined as:

$$
\begin{array} { r } { \hat { d } _ { i } = \hat { q } _ { i } - T \hat { p } _ { i } , } \end{array}\tag{11}
$$

Using the properties of gaussian distributions, the error $d _ { i }$ is derived to follow:

$$
d _ { i } \sim \mathcal { N } ( 0 , C _ { q } ^ { i } + T C _ { p } ^ { i } T ^ { \mathrm { T } } ) ,\tag{12}
$$

To find the optimal transformation matrix T , we maximize the probability distribution of each paired point $\mathbf { p } ( d _ { i } )$ (maximum log-likelihood estimation):

$$
\begin{array} { l } { { \displaystyle T = \arg \operatorname* { m a x } _ { \mathbf { T } } \prod _ { i } \mathbf { p } ( d _ { i } ) } } \\ { { \displaystyle \quad = \arg \operatorname* { m a x } _ { \mathbf { T } } \sum _ { i } \log ( \mathbf { p } ( d _ { i } ) ) } } \\ { { \displaystyle \quad = \arg \operatorname* { m i n } _ { T } \sum _ { i } d _ { i } ^ { \mathrm { T } } \left( C _ { q } ^ { i } + T C _ { p } ^ { i } T ^ { \mathrm { T } } \right) ^ { - 1 } d _ { i } } } \end{array} ,\tag{13}
$$

## C. Keyframe Selection

Tracking keyframes are selected based on the overlap ratio between the observed point cloud in the current frame and the sparse gaussian map. For points within a permissible distance, they are considered overlapping. If the ratio of overlapping points to the total observed point cloud is below a threshold, the current frame is chosen as a tracking keyframe. If the current frame differs from the previous tracking keyframe by ten frames, it is marked as a mappingonly keyframe. Tracking keyframes serve both tracking and mapping purposes, while mapping keyframes are used solely for mapping.

To further optimize the map, additional filtering is applied to the keyframes. As shown in Fig. 3, we select keyframes with a co-visibility greater than 70% with the current frame as co-visible keyframes. Additionally, to mitigate the effects of scene forgetting and gaussian artifacts, we randomly sample 30% of the remaining keyframes as random keyframes.

## D. Map Optimization

Our system optimizes both sparse and dense gaussian maps through local and global optimization. The co-visible keyframes are used for local optimization of the neighboring region observed in the current frame, while global optimization is achieved jointly by the co-visible keyframes and randomly selected keyframes.

To ensure stability and avoid uncontrolled gaussian scaling, we introduce a regularization loss $\mathcal { L } _ { r e g } \{$

$$
\mathcal { L } _ { r e g } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| S _ { i , 1 : 2 } - \bar { S } _ { 1 : 2 } \right| + \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | S _ { i , 3 } - \varepsilon | ,\tag{14}
$$

<!-- image-->  
Fig. 3. Illustration of Keyframe Selection Strategy. The red frustum and its sector represent the current frameâs observed field of view. The blue frustums indicate co-visibility keyframes with a partially overlapping field of view with the current frame. The black frustums represent normal keyframes, while the green frustums are randomly selected keyframes from other frames.

where $S \in \mathbb { R } ^ { n \times 3 }$ represents the gaussian scale parameter matrix, where n is the sample count, and $\varepsilon ~  ~ 0 . ~ \bar { S } _ { 1 : 2 }$ denotes the mean of the first two scale parameters. This regularization loss $\mathcal { L } _ { { r e g } }$ encourages consistency of gaussian scales across dimensions and promotes flattened alignment along the surface normal direction in the neighborhood.

The mapping loss combines color loss, depth loss, structural similarity (SSIM) loss, and regularization loss:

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } o s s = \lambda _ { i } | | I - I _ { g t } | | _ { 1 } + \lambda _ { d } | | D - D _ { g t } | | _ { 1 } } \\ { + ( 1 - \lambda _ { i } ) ( 1 - s s i m ( I , I _ { g t } ) ) + \lambda _ { r } { \mathcal { L } _ { r e g } } } \end{array} , } \end{array}\tag{15}
$$

where $\lambda _ { i }$ and $\lambda _ { d }$ represent the weights for color loss and depth loss, respectively. I and $I _ { g t }$ denote the RGB rendered image and the ground truth image, respectively. D and $D _ { g t }$ represent the depth rendered image and the ground truth depth image. ssim(Â·) denotes the structural similarity calculation, and $\lambda _ { r }$ is the weight for the regularization loss.

Mapping optimization and tracking run in parallel threads, significantly enhancing speed and ensuring real-time SLAM performance. The dense map updates on tracking keyframes, and both sparse and dense maps share computational resources, leading to unified optimization without additional computational cost.

## IV. EXPERIMENTS

## A. Experimental Setup

Datasets. We evaluate our method using the Replica and TUM RGB-D datasets. The Replica dataset provides synthetic data with precise RGB images and depth maps. TUM RGB-D dataset have relatively lower image quality, requiring greater robustness and accuracy in dense mapping and tracking methods.

Implementation Details. Our experiments were conducted on an NVIDIA GeForce RTX 4090 24GB GPU. The mapping optimization and tracking processes run in parallel threads, sharing gaussian data across threads for efficiency.

Evaluation Metrics. For map reconstruction performance, we assess rendering quality using PSNR, SSIM, and LPIPS metrics. For camera tracking accuracy, we use the Root Mean Square Error of Absolute Trajectory Error (ATE RMSE) as the main evaluation metric. For evaluating system real-time performance, we use the overall system frame rate (FPS) rather than the rendering speed or the speed of individual components.

<!-- image-->  
Fig. 4. Qualitative result comparison on the Replica dataset. Detail zoom-ins from three scenes are presented. Our method outperforms other frameworks in the reconstruction of map details.

<!-- image-->  
Fig. 5. Performance Comparison of gaussian missing region check strategies. Our method effectively and promptly fills in gaussians in regions observed in very few frames.

Baseline Methods. We compare our approach with NeRFbased neural implicit SLAM methods, including NICE-SLAM [8], Point-SLAM [22], and Co-SLAM [9]. Additionally, we compare with state-of-the-art 3DGS-based explicit SLAM methods like GS-SLAM [21], SplaTAM [14], MonoGS [15], CG-SLAM [23], and GS-ICP-SLAM [24].

## B. Camera Tracking Accuracy Evaluation

The tables in this paper contains three colors, representing the best , second best , and third best , respectively. As shown in Table I, our method demonstrates excellent tracking performance across eight scenes in the Replica dataset. This success is due to the shared gaussian distribution assumptions between GICP and our mapâs gaussian representation. GICP achieves tracking by leveraging the 3D gaussian distribution within the sparse map. In contrast, other baseline methods typically track camera poses by minimizing 2D image rendering differences with ground truth images, which demands maintaining dense maps. The upkeep of dense maps is computationally expensive, and their quality greatly affects tracking accuracy, as mapping errors can propagate into tracking. Our method, by maintaining a lightweight sparse map for GICP tracking, avoids the need to project 3D maps into 2D, achieving direct tracking within the 3D space.

TABLE I  
CAMERA POSE ESTIMATION RESULTS ON THE REPLICA DATASET (ATE RMSEâ[CM]). OUR METHOD DEMONSTRATES SUPERIOR PERFORMANCE ACROSS ALL 8 SCENES, OUTPERFORMING THE CURRENT STATE-OF-THE-ART (SOTA) BASELINES. SOME BASELINE DATA IS SOURCED FROM [23].
<table><tr><td>Method</td><td>R0</td><td>R1</td><td>R2</td><td>OF0</td><td>OF1</td><td>OF2</td><td>OF3</td><td>OF4</td><td>Avg.</td></tr><tr><td>NICE-SLAM [8]</td><td>0.97</td><td>1.31</td><td>1.07</td><td>0.88</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>1.06</td></tr><tr><td>Point-SLAM [22]</td><td>0.56</td><td>0.47</td><td>0.30</td><td>0.35</td><td>0.62</td><td>0.55</td><td>0.72</td><td>0.73</td><td>0.54</td></tr><tr><td>Co-SLAM [9]</td><td>0.77</td><td>1.04</td><td>1.09</td><td>0.58</td><td>0.53</td><td>2.05</td><td>1.49</td><td>0.84</td><td>0.99</td></tr><tr><td>GS-SLM[21]</td><td>0.48</td><td>0.53</td><td>0.33</td><td>0.52</td><td>0.41</td><td>0.59</td><td>0.46</td><td>0.70</td><td>0.50</td></tr><tr><td>SplaTAM [14]â </td><td>0.27</td><td>0.31</td><td>0.63</td><td>0.49</td><td>0.22</td><td>0.30</td><td>0.35</td><td>0.52</td><td>0.39</td></tr><tr><td>MonoGS (RGB-D) [15]â </td><td>0.35</td><td>0.26</td><td>0.27</td><td>0.41</td><td>0.40</td><td>0.22</td><td>0.14</td><td>2.10</td><td>0.52</td></tr><tr><td>CG-SLAM [23]</td><td>0.29</td><td>0.27</td><td>0.25</td><td>0.33</td><td>0.14</td><td>0.28</td><td>0.31</td><td>0.29</td><td>0.27</td></tr><tr><td>Ours</td><td>0.14</td><td>0.17</td><td>0.10</td><td>0.16</td><td>0.13</td><td>0.16</td><td>0.16</td><td>0.20</td><td>0.15</td></tr></table>

â  denotes the reproduced results by running officially released code.

In Table II, the tracking performance of various methods on the TUM RGB-D dataset is presented. Due to lower image quality and motion blur, tracking accuracy on the TUM dataset is generally reduced compared to Replica. Our method achieves tracking performance comparable to advanced NeRF-based SLAM and 3DGS-based SLAM approaches but offers a significantly faster system speed, with at least an order of magnitude advantage. While both GS-ICP-SLAM [24] and our method operate in real time, our method achieves higher tracking accuracy.

## C. Mapping Quality Evaluation

Table III presents our methodâs mapping quality and system speed. Our approach achieves state-of-the-art or nearoptimal performance in mapping quality across most metrics while maintaining a speed advantage of one to two orders of magnitude. This efficiency is due to our shared sparse and dense maps, where sparse map tracking and shared gaussians conserve computational resources, and frequency-domaininformed adaptive gaussian densification ensures high map quality.

Fig. 4 presents a qualitative comparison of results, using three scenes as examples to demonstrate the high demands

TABLE II

CAMERA POSE ESTIMATION RESULTS ON THE TUM-RGBD DATASET (ATE RMSEâ[CM]). OUR METHOD ACHIEVES TRACKING ACCURACY ON PAR WITH THE STATE-OF-THE-ART (SOTA) IN THIS DATASET, WITH THE SYSTEM RUNNING AT A FRAME RATE (FPSâ) THAT DEMONSTRATES ADVANCED PERFORMANCE. SOME BASELINE DATA IS SOURCED FROM [24], [23].

<table><tr><td rowspan=1 colspan=2>Method               fr1/desk</td><td rowspan=1 colspan=4>fr2/xyz fr3/office Avg. FPSâ</td></tr><tr><td rowspan=3 colspan=2>NICE-SLAM [8]          2.8Point-SLAM [22]         2.7Co-SLAM [9]            2.7</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>0.08</td></tr><tr><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>3.9</td><td rowspan=1 colspan=1>2.6</td><td rowspan=1 colspan=1>0.22</td></tr><tr><td rowspan=1 colspan=1>1.9</td><td rowspan=1 colspan=1>2.6</td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>GS-SLAM [21]</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>6.6</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>SplaTAM [14]â </td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>3.2</td><td rowspan=1 colspan=1>0.22</td></tr><tr><td rowspan=2 colspan=1>MonoGS (RGB-D) [15]â CG-SLAM [23]</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>1.6</td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=1>1.6</td><td rowspan=1 colspan=1>1.64</td></tr><tr><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>GS-ICP-SLAM [24]â </td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>1.8</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>29.96</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>44.09</td></tr></table>

TABLE III

RENDERING RESULTS ON THE REPLICA DATASET. OUR METHOD ACHIEVES AN OPTIMAL BALANCE BETWEEN SYSTEM SPEED AND MAPPING QUALITY. SOME BASELINE DATA IS SOURCED FROM [22], [24].
<table><tr><td>Method</td><td>Metrics</td><td>R0</td><td>R1</td><td>R2</td><td>OF0</td><td>OF1</td><td>OF2</td><td>OF3</td><td>OF4</td><td>Avg.</td><td>FPS â</td></tr><tr><td>NICE-SLAM [8]</td><td>PSNR[dB]â</td><td>22.12</td><td>22.47</td><td>24.52</td><td>29.07</td><td>30.34</td><td>19.66</td><td>22.23</td><td>24.94</td><td>24.42</td><td></td></tr><tr><td></td><td>SIM</td><td>0.689</td><td>0.757</td><td>0.814</td><td>0.874</td><td>.886</td><td>0.797</td><td>0.801</td><td>0.856</td><td>0.809</td><td></td></tr><tr><td></td><td>LPIPS â</td><td>0.330</td><td>0.271</td><td>0.208</td><td>0.229</td><td>0.181</td><td>0.235</td><td>0.209</td><td>0.198</td><td>0.233</td><td></td></tr><tr><td>Point-SLAM [22]</td><td>PSNR[dB]â</td><td>33.38</td><td>34.10</td><td>36.32</td><td>38.72</td><td>39.31</td><td>34.22</td><td>34.10</td><td>34.82</td><td>35.62</td><td></td></tr><tr><td></td><td>SS âM</td><td>0.979</td><td>0.977</td><td>0.985</td><td>0.985</td><td>0.987</td><td>0.962</td><td>0.963</td><td>0.981</td><td>0.977</td><td>0.3</td></tr><tr><td></td><td>LPIPS â</td><td>0.097</td><td>0.115</td><td>0.101</td><td>0.089</td><td>0.110</td><td>0.152</td><td>0.119</td><td>0.131</td><td>0.114</td><td></td></tr><tr><td>GS-SLAM [21]</td><td>PSNR[dB]â</td><td>31.56</td><td>32.86</td><td>32.59</td><td>38.70</td><td>41.17</td><td>32.36</td><td>32.03</td><td>32.92</td><td>34.27</td><td></td></tr><tr><td></td><td>SSIM â</td><td>0.968</td><td>0.973</td><td>0.971</td><td>0.986</td><td>0.993</td><td>0.978</td><td>0.970</td><td>0.968</td><td>0.975</td><td>8.34</td></tr><tr><td></td><td>L PPIPS </td><td>0.094</td><td>0.075</td><td>0.093</td><td>0.050</td><td>0.033</td><td>0.094</td><td>0.110</td><td>0.112</td><td>0.082</td><td></td></tr><tr><td>SplaTAM [14]â </td><td>PSNR[dB]â</td><td>32.60</td><td>33.63</td><td>34.91</td><td>38.15</td><td>39.05</td><td>31.89</td><td>30.18</td><td>32.01</td><td>34.05</td><td></td></tr><tr><td></td><td>SSIM *</td><td>0.975</td><td>0.969</td><td>0.982</td><td>0.981</td><td>0.981</td><td>0.966</td><td>0.951</td><td>0.948</td><td>0.969</td><td>0.18</td></tr><tr><td></td><td>LPIPS </td><td>00.070</td><td>00.097</td><td>00.073</td><td>0.088</td><td>0.094</td><td>0.100</td><td>0.118</td><td>0.154</td><td>0.099</td><td></td></tr><tr><td>MonoGS (RGB-D) [15]â </td><td>PSNR[dB]â</td><td>33.21</td><td>35.88</td><td>36.86</td><td>40.49</td><td>41.39</td><td>35.62</td><td>35.48</td><td>33.65</td><td>36.57</td><td></td></tr><tr><td></td><td>SSIM L PIPS </td><td>0.937</td><td>0.954</td><td>0.961</td><td>0.974</td><td>0.975</td><td>0.958</td><td>0.957</td><td>0.940</td><td>0.957</td><td>0.81</td></tr><tr><td></td><td></td><td>0.081</td><td>0.092</td><td>0.075</td><td>0.061</td><td>0.053</td><td>0.071</td><td>0.059</td><td>0.112</td><td>0.076</td><td></td></tr><tr><td>GS-ICP-SLAM [24]â </td><td>PSNR[dB]â</td><td>35.11</td><td>37.28</td><td>38.11</td><td>42.38</td><td>42.76</td><td>36.77</td><td>36.80</td><td>38.54</td><td>38.55</td><td></td></tr><tr><td></td><td>SSIM </td><td>0.960</td><td>0.968</td><td>0.973</td><td>0.984</td><td>0.982</td><td>0.971</td><td>0.968</td><td>0.967</td><td>0.970</td><td>29.95</td></tr><tr><td></td><td>L PIPS </td><td>0.053</td><td>0.051</td><td>0.053</td><td>0.032</td><td>0.036</td><td>0.048</td><td>0.047</td><td>0.049</td><td>0.045</td><td></td></tr><tr><td>Ours</td><td>PSNR[dB]â</td><td>35.27</td><td>38.05</td><td>38.63</td><td>42.73</td><td>43.18</td><td>36.42</td><td>37.04</td><td>38.66</td><td>38.75</td><td></td></tr><tr><td></td><td>SIM </td><td>0.961</td><td>0.972</td><td>0.975</td><td>0.984</td><td>0.984</td><td>0.973</td><td>0.969</td><td>0.972</td><td>0.974</td><td>32.75</td></tr><tr><td></td><td>LLPPIPS </td><td>0.045</td><td>0.043</td><td>0.045</td><td>0.028</td><td>0.035</td><td>0.045</td><td>0.040</td><td>0.046</td><td>0.041</td><td></td></tr></table>

for fine-grained mapping in the system. In the zoomedin view of the iron mesh detail in Room1, the SplaTAM result shows chaotic textures, MonoGS fails to reconstruct the iron mesh texture, and GS-ICP-SLAM only reconstructs part of the texture. In Office3, the chair backrest is semitransparent and consists of rows of black dots forming lines. GS-ICP-SLAM fails to reconstruct the black dots and instead reconstructs them as black lines. For the floor texture details in Office4, our method outperforms the others. In contrast, our method captures the fine texture differences in these scenes, with minimal deviation from the ground truth. The successful reconstruction of these details is attributed to our Fourier frequency segmentation mechanism, which initializes a dense distribution of small gaussians at high-frequency locations.

We employ a gaussian missing region check strategy for each new keyframe. Fig. 5 provides an intuitive comparison between our strategy and GS-ICP-SLAM. GS-ICP-SLAM adds new gaussians by populating new keyframes with uniformly and equidistantly sampled sparse gaussian points. While this strategy is simple and efficient to execute, it leads to insufficient gaussian representation in regions observed in few frames. This issue can result in holes in the map. Our deficient region inpainting strategy can promptly fill these holes and correct regions with significant map discrepancies.

<!-- image-->  
Fig. 6. Ablation comparison of gaussian regularization loss.  
TABLE IV

EXECUTION TIME OF EACH MODULE IN THE SYSTEM ON REPLICA / OFFICE0. [MS Ã IT] DENOTES THE SINGLE EXECUTION TIME AND THE NUMBER OF ITERATIONS. THE BASELINE DATA IS SOURCED FROM [23].
<table><tr><td>Method</td><td>Tracking [ms Ã it]â</td><td> $\mathrm { M a p p i n g }$   $[ \mathrm { m s } ~ \times \mathrm { i t } ] \downarrow$ </td><td>System FPâS </td></tr><tr><td>Vox-Fusion [11]</td><td> $2 3 . 6 1 \times 3 0$ </td><td> $8 6 . 5 5 \times 1 0$ </td><td>1.1</td></tr><tr><td>NICE-SLAM [8]</td><td> $6 . 1 9 \times 1 0$ </td><td> $9 1 . 5 9 \times 6 0$ </td><td>0.98</td></tr><tr><td>Co-SLAM [9]</td><td> $4 . 4 5 \times 1 0$ </td><td> $1 0 . 9 \times 1 0$ </td><td>14.2</td></tr><tr><td>Point-SLAM [22]</td><td> $6 . 1 4 \times 4 0$ </td><td> $2 2 . 2 5 \times 3 0 0$ </td><td>0.48</td></tr><tr><td>GS-SLAM [21]</td><td> $1 1 . 9 \times 1 0$ </td><td> $1 2 . 8 \times 1 0 0$ </td><td>8.34</td></tr><tr><td>SplaTAM [14]</td><td> $4 1 . 7 \times 4 0$ </td><td> $5 0 . 1 \times 6 0$ </td><td>0.21</td></tr><tr><td>CG-SLAM [23]</td><td> $7 . 8 9 \times 1 5$ </td><td> $1 2 . 2 \times 6 0$ </td><td>8.5</td></tr><tr><td>Ours</td><td> $2 9 . 8 \times 1$ </td><td> $1 0 . 7 \times 2 . 8 $ </td><td>33.04</td></tr></table>

As shown in the Table IV, compared to other methods, our system has the shortest tracking time, and the single mapping time is the lowest. Surprisingly, the iteration count is fewer than three, which is 5 to 100 times fewer than the mapping iteration count of other methods. This demonstrates that the frequency-domain guided SLAM method proposed in this paper can converge quickly with very few iterations.

## D. Ablation Study

Keyframe Selection. We evaluated the impact of different keyframe selection strategies on mapping performance using eight Replica dataset scenes. The first approach randomly selects keyframes, the second optimizes only co-visible keyframes, and the third combines both co-visible and randomly selected keyframes. Table V shows that optimizing only random keyframes reduces optimization opportunities in recently observed areas, while focusing solely on covisible keyframes can over-stretch certain gaussians and lead to unobserved scene regions being forgotten. Combining both strategies provides optimal results by balancing recent and global scene coverage.

Adaptive Gaussian Densification. To assess the effect of adaptive densification based on frequency-domain analysis, we conducted ablation studies varying gaussian density and radius. Table VI shows results in Replica Office0, where equidistant sparse mapping yields lower quality than equidistant dense and adaptive dense methods. However, sparse mapping uses the fewest gaussians and has lower memory requirements. Large-radius gaussians yield weaker maps than small-radius gaussians, suggesting the latterâs advantage in capturing details. However, small-radius gaussians in equidistant dense initialization lead to increased gaussian count and memory use. Our adaptive densification balances mapping quality and memory usage by applying sparse largeradius gaussians in low-frequency regions and dense smallradius gaussians in high-frequency areas.

TABLE V  
ABLATION STUDY OF KEYFRAME SELECTION ON THE REPLICA DATASET. THE RESULTS REPRESENT THE AVERAGE ACROSS 8 SCENES.
<table><tr><td>Method</td><td>PSNR[dB]â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>w/o co-visible keyframes</td><td>37.67</td><td>0.968</td><td>0.049</td></tr><tr><td>w/o random keyframes</td><td>34.55</td><td>0.949</td><td>0.077</td></tr><tr><td>Ours</td><td>38.75</td><td>0.974</td><td>0.041</td></tr></table>

TABLE VI  
ABLATION STUDY OF ADAPTIVE GAUSSIAN DENSIFICATION ON THE REPLICA / OFFICE0 DATASET.
<table><tr><td colspan="2">Method</td><td>PSNR[dB]â</td><td>SSIM â</td><td>LPIPS â</td><td>Memory Usageâ</td></tr><tr><td rowspan="2">Sparse equidistant</td><td>Large gaussians</td><td>39.78</td><td>0.973</td><td>0.064</td><td>29.6 M</td></tr><tr><td>Small gaussians</td><td>41.94</td><td>0.981</td><td>0.037</td><td>54.4 M</td></tr><tr><td rowspan="2">Dense equidistant</td><td>Large gaussians</td><td>40.62</td><td>0.978</td><td>0.044</td><td>112.4M</td></tr><tr><td>Small gaussians</td><td>42.68</td><td>0.984</td><td>0.022</td><td>157.4 M</td></tr><tr><td colspan="2">Adaptive gaussian densification</td><td>42.73</td><td>0.984</td><td>0.028</td><td>68.1 M</td></tr></table>

Regularization. As shown in Fig. 6, adding gaussian regularization loss improves the mapping quality. This is because the regularization constrains the gaussian distribution, preventing it from being excessively elongated in any particular direction. Experimental results demonstrate that the regularization constraint leads to higher mapping quality.

## V. CONCLUSION

We introduce a novel SLAM system based on frequencydomain analysis that initializes gaussians according to highand low-frequency regions, resulting in detailed mapping in high-frequency areas. By combining a gaussian missing region check strategy, the system effectively avoids incomplete reconstructions in regions visible in few frames. The proposed shared-resource mechanism for dense and sparse gaussian maps significantly enhances system speed. This framework achieves state-of-the-art mapping quality while maintaining real-time operational speed. The proposed method opens up a new research avenue for analyzing 3DGS SLAM from the frequency domain perspective. We believe that the rich information in the frequency domain will drive the further development of SLAM.

## REFERENCES

[1] C. Li, Y. Zhang, Z. Yu, X. Liu, and Q. Shi, âA robust visual slam system for small-scale quadruped robots in dynamic environments,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 321â326.

[2] R. Mur-Artal and J. D. Tardos, âOrb-slam2: An open-source slam Â´ system for monocular, stereo, and rgb-d cameras,â IEEE transactions on robotics, vol. 33, no. 5, pp. 1255â1262, 2017.

[3] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âOrb-slam3: An accurate open-source library for visual, Â´ visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[4] A. Dai, M. NieÃner, M. Zollhofer, S. Izadi, and C. Theobalt, âBundle- Â¨ fusion: Real-time globally consistent 3d reconstruction using onthe-fly surface reintegration,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, p. 1, 2017.

[5] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison, âElasticfusion: Dense slam without a pose graph.â in Robotics: science and systems, vol. 11. Rome, Italy, 2015, p. 3.

[6] T. Whelan, M. Kaess, M. F. Fallon, H. Johannsson, J. J. Leonard, and J. B. McDonald, âKintinuous: Spatially extended kinectfusion,â in AAAI Conference on Artificial Intelligence, 2012.

[7] X. Wang, Y. Zhang, Z. Zhang, M. Wang, Z. Li, and X. Chen, âFi-slam: Feature fusion and instance reconstruction for neural implicit slam,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 527â532.

[8] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 12 786â12 796.

[9] H. Wang, J. Wang, and L. Agapito, âCo-slam: Joint coordinate and sparse parametric encodings for neural real-time slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 293â13 302.

[10] M. M. Johari, C. Carta, and F. Fleuret, âEslam: Efficient dense slam system based on hybrid representation of signed distance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 408â17 419.

[11] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVoxfusion: Dense tracking and mapping with voxel-based neural implicit representation,â in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2022, pp. 499â507.

[12] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[13] L. C. Sun, N. P. Bhatt, J. C. Liu, Z. Fan, Z. Wang, T. E. Humphreys, and U. Topcu, âMm3dgs slam: Multi-modal 3d gaussian splatting for slam using vision, depth, and inertial measurements,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 10 159â10 166.

[14] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[15] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[16] J. Zhang, F. Zhan, M. Xu, S. Lu, and E. Xing, âFregs: 3d gaussian splatting with progressive frequency regularization,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 424â21 433.

[17] A. Segal, D. Haehnel, and S. Thrun, âGeneralized-icp.â in Robotics: science and systems, vol. 2, no. 4. Seattle, WA, 2009, p. 435.

[18] E. Hourdakis, S. Piperakis, and P. Trahanias, âroboslam: Dense rgb-d slam for humanoid robots,â in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 2224â2231.

[19] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon, âKinectfusion: Real-time dense surface mapping and tracking,â in 2011 10th IEEE international symposium on mixed and augmented reality. Ieee, 2011, pp. 127â136.

[20] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, âimap: Implicit mapping and positioning in real-time,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 6229â6238.

[21] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGsslam: Dense visual slam with 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 595â19 604.

[22] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, âPoint- Â¨ slam: Dense neural point cloud-based slam,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 433â18 444.

[23] J. Hu, X. Chen, B. Feng, G. Li, L. Yang, H. Bao, G. Zhang, and Z. Cui, âCg-slam: Efficient dense rgb-d slam in a consistent uncertaintyaware 3d gaussian field,â in European Conference on Computer Vision. Springer, 2025, pp. 93â112.

[24] S. Ha, J. Yeon, and H. Yu, âRgbd gs-icp slam,â in European Conference on Computer Vision. Springer, 2024, pp. 180â197.