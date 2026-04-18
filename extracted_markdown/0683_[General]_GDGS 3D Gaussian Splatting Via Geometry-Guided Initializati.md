# GDGS: 3D GAUSSIAN SPLATTING VIA GEOMETRY-GUIDED INITIALIZATION AND DYNAMIC DENSITY CONTROL

Xingjun Wang, Lianlei Shan

Tsinghua University

## ABSTRACT

We propose a method to enhance 3D Gaussian Splatting (3DGS) [1], addressing challenges in initialization, optimization, and density control. Gaussian Splatting is an alternative for rendering realistic images while supporting real-time performance, and it has gained popularity due to its explicit 3D Gaussian representation. However, 3DGS heavily depends on accurate initialization and faces difficulties in optimizing unstructured Gaussian distributions into ordered surfaces, with limited adaptive density control mechanism proposed so far. Our first key contribution is a geometry-guided initialization to predict Gaussian parameters, ensuring precise placement and faster convergence. We then introduce a surface-aligned optimization strategy to refine Gaussian placement, improving geometric accuracy and aligning with the surface normals of the scene. Finally, we present a dynamic adaptive density control mechanism that adjusts Gaussian density based on regional complexity, for visual fidelity. These innovations enable our method to achieve high-fidelity real-time rendering and significant improvements in visual quality, even in complex scenes. Our method demonstrates comparable or superior results to state-of-the-art methods, rendering high-fidelity images in real time.

Index Termsâ 3D Gaussian Splatting (3DGS), Novel view synthesis, Real-time rendering, Structure-from-Motion (SfM)

## 1. INTRODUCTION

Novel view synthesis is a fundamental task in computer vision and graphics. 3D Gaussian Splatting (3DGS) [1]has emerged as a cutting-edge approach for capturing and rendering 3D scenes from novel perspectives. Unlike NeRFs [2], which rely on MLPs which are computationally intensive and resource-demanding, 3DGS directly models scenes using 3D Gaussians. This method optimizes Gaussian positions, orientations, appearances, and alpha blending to represent the sceneâs geometry and appearance efficiently.

Current 3DGS methods encode scene geometry and appearance by optimizing parameters such as position, covariance, and color of 3D Gaussians. Despite their flexibility, these methods face challenges in aligning unstructured Gaussian distributions into ordered surfaces. Additionally, uniform treatment of all image regions leads to inefficiencies, as highdetail or close-up areas demand finer sampling, while simpler areas incur unnecessary computational costs.

We introduce three key innovations. First, an improved geometric initialization strategy generates a structured and reliable point cloud, outperforming point cloud from Structurefrom-Motion. Second, surface normals are aligned with planes to further improve geometric accuracy. Third, a novel adaptive density control (ADC) mechanism leverages dynamic resolution to determine regions requiring additional Gaussians. Unlike current approaches that delete overly transparent or camera-proximal points and clone large highgradient Gaussians, this method uses fixed region segmentation to assess detail needs based on Gaussian density and gradient magnitude. In regions requiring adjustment, evenly distributed Gaussians are increased via cloning, while uneven distributions are refined by modifying regional loss functions to improve Gaussian allocation.

Our proposed approach results in images with minimal pixel-level distortion, successfully preserving overall structural integrity. The method ensures a higher degree of structural similarity while reducing pixel-wise discrepancies, achieving superior accuracy in capturing intricate details of the scene, especially in high-detail regions. This leads to improved rendering quality and enhanced performance for realtime rendering tasks.

## 2. RELATED WORKS

Our research builds on 3D Gaussian Splatting (3DGS) [1]. We discuss related works in traditional scene reconstruction, neural rendering, and point-based rendering.

Early scene reconstruction methods leveraged light fields for novel-view synthesis [3, 4], progressing to unstructured captures [5]. Structure-from-Motion (SfM) [6]introduced sparse point clouds for visualizing 3D space, further enhanced by Multi-View Stereo (MVS) [7] for dense reconstructions. These methods achieved compelling results in tasks such as re-projection-based view synthesis [8, 9, 10, 11]. However, challenges remained with artifacts like unreconstructed regions and over-reconstructed geometry.

Neural rendering algorithms [12] have significantly reduced these issues, offering superior performance without the overhead of storing all input images on the GPU. These methods have established neural rendering as a robust alternative to traditional approaches for diverse applications.

Point-based rendering [13, 14, 15, 16] provides an efficient way to handle unstructured geometry but often suffers from discontinuities and aliasing. Differentiable pointbased techniques [17, 18] have incorporated neural features for improved performance, but their reliance on MVS-derived geometry limits their robustness in complex scenes. Pulsar [19]introduced fast sphere rasterization, inspiring the efficient rasterization techniques used in 3DGS.

NeRF [2] marked a significant advancement in novelview synthesis by rendering 3D views through ray integration of 2D data. NeRF encodes positional information using positional encoding to improve spatial understanding and utilizes hierarchical volume sampling for enhanced rendering through multi-level sampling. NeRF trains a MLP to predict the density and radiance at any 3D point.

While subsequent models have extended NeRFâs capabilities for dynamic scenes, reduced data requirements, and accelerated training using external tools like hash grids [20], its reliance on computationally expensive ray-based querying limits its rendering speed, making it unsuitable for real-time applications.

3DGS [1] introduces an explicit scene representation using 3D Gaussian primitives, which offers substantial advantages for real-time novel-view synthesis. Unlike NeRFâs implicit volumetric representation, 3DGS directly models scenes with explicit 3D Gaussians. These are rasterized into image space using a fast and differentiable CUDAbased algorithm, enabling real-time rendering at high resolutions. SuGaR [21] aligns Gaussian splats with surface normals, enhancing the fidelity of 3D mesh reconstruction and enabling high-quality rendering. Multi-Scale 3D Gaussian Splatting [22] introduces a multi-scale approach to Gaussian splatting, ensuring anti-aliased rendering by dynamically adapting splat density and scale based on scene complexity. And others also do some meaningful works in segmentation [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53].

## 3. METHOD

## 3.1. Overview

Our research builds upon the 3D Gaussian Splatting (3DGS) framework [1] by introducing additional steps to enhance its initialization, optimization, and adaptive density control as shown in Figure 1. Starting with input images and camera parameters calibrated using Structure-from-Motion (SfM) [6], we use the resulting sparse point cloud as input to a Multi-Layer Perceptron (MLP), which predicts initial Gaussian parametersâpositions, covariance matrices, and opacity. This data-driven initialization replaces random parameter generation, ensuring a more precise starting point and accelerating convergence. The initialized Gaussians are then optimized to align with surface normals extracted from mesh data, ensuring they remain external to the object while capturing fine surface details. This optimization is guided by a composite loss function that balances distance, direction, and surface fitting losses. Additionally, our dynamic Adaptive Density Control (ADC) divides the scene into fixed 3D grid regions, dynamically refining Gaussian density based on regional complexity. High-complexity regions are enriched by cloning Gaussians, while simpler regions undergo loss-based dispersion to maintain uniformity and avoid redundancy. The optimized Gaussians are rendered through a tile-based rasterizer achieving real-time performance.

<!-- image-->  
Fig. 1. Optimization process begins with Structure-from-Motion (SfM) points, points are utilized in a geometry-guided initialization phase to accurately position 3D Gaussians according to the sceneâs geometry. The initialized Gaussians, undergo further optimization to ensure they align with the surface normals, enhancing geometric accuracy. Following this, dynamic region density control is applied, adjusting the Gaussiansâ density across the scene to improve rendering efficiency and quality.

## 3.2. Geometry Guided Adaptive Optimization

<!-- image-->  
Fig. 2. Following initialization, positioning logic is applied to adjust the point orientation and distance, by setting training objectives and defining a loss function specifically for the orientation and position convergence criteria.

Our goal is to achieve high-quality initialization and optimization of 3D Gaussians for novel-view synthesis by leveraging structured data. This is accomplished in two distinct stages: an adaptive initialization stage, which ensures precise placement of Gaussians, and a dynamic optimization stage, which refines their positions and orientations for accurate reconstruction as shown in Figure 2. Together, these stages address the challenges of reconstructing 3D scenes with both precision and efficiency.

The adaptive initialization process begins by utilizing structured outputs such as camera intrinsics, extrinsics, and a sparse point cloud. These outputs provide the foundation for predicting Gaussian centers (Âµ) using a Multi-Layer Perceptron (MLP). Sparse point clouds capture the geometric structure of the scene but lack reliable surface normals. To overcome this limitation, the scene is modeled as a set of 3D Gaussians, each defined by a center position $( \mu )$ and covariance matrix (Î£). The MLP, trained on camera calibration and point cloud data, predicts Gaussian centers that closely align with the scene geometry, significantly improving initialization accuracy and optimization convergence.

The MLP architecture takes 3D coordinates from the point cloud as input and processes them through multiple layers with nonlinear activations (e.g., ReLU) to model complex relationships. The output layer directly predicts the Gaussian center positions $( \mu )$ , which serve as the starting points for subsequent optimization. During training, a loss function evaluates the deviation of predicted Gaussian centers from their ground truth:

$$
\mathcal { L } _ { \mathrm { i n i t } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \mu _ { i } ^ { \mathrm { p r e d } } - \mu _ { i } ^ { \mathrm { g t } } \| ^ { 2 } ,\tag{1}
$$

where $\mu _ { i } ^ { \mathrm { p r e d } }$ and $\mu _ { i } ^ { \mathrm { g t } }$ represent the predicted and ground truth centers, respectively. The training process employs normalized input features to ensure uniform contributions across dimensions, preventing gradient saturation. Data augmentation, such as Gaussian noise addition, enhances robustness and generalization.

Following initialization, dynamic optimization refines Gaussian placement to better align with surface geometry and capture scene complexity. Each Gaussian i is iteratively adjusted to position it slightly outside the object surface, perpendicular to its normal vector. The target position is:

$$
\mu _ { i } ^ { \mathrm { t a r g e t } } = \mu _ { i } ^ { \mathrm { m e s h } } + d \cdot \mathbf { N } ,\tag{2}
$$

where $\mu _ { i } ^ { \mathrm { m e s h } }$ is the nearest point on the surface mesh, d is a positive offset, and N is the surface normal. The iterative update rule adjusts Gaussian centers toward their targets:

$$
\mu _ { i }  \mu _ { i } - \eta \nabla _ { \mu _ { i } } { \mathcal { L } } ,\tag{3}
$$

where $\eta$ is the learning rate, and $\mathcal { L }$ represents the total loss.

The optimization process is governed by a composite loss function:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { d } \mathcal { L } _ { \mathrm { d i s t } } + \lambda _ { a } \mathcal { L } _ { \mathrm { a l i g n } } + \lambda _ { s } \mathcal { L } _ { \mathrm { s u r f a c e } } , } \end{array}\tag{4}
$$

where $\lambda _ { d } , \lambda _ { a } , \lambda _ { s }$ are hyperparameters balancing the components. The distance loss evaluates deviations from the target:

$$
\mathcal { L } _ { \mathrm { d i s t } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \mu _ { i } - \mu _ { i } ^ { \mathrm { t a r g e t } } \| ^ { 2 } .\tag{5}
$$

The alignment loss ensures Gaussian orientations align with surface normals:

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( 1 - \cos \theta _ { i } ) ,\tag{6}
$$

where cos $\begin{array} { r } { \theta _ { i } = \frac { \mathbf { U } _ { i } \cdot \mathbf { N } _ { i } } { \| \mathbf { U } _ { i } \| \| \mathbf { N } _ { i } \| } } \end{array}$ , and $\mathbf { U } _ { i }$ is the Gaussianâs orientation vector. The surface fitting loss measures alignment with the mesh:

$$
\mathcal { L } _ { \mathrm { s u r f a c e } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j \in \mathcal { N } ( i ) } \| \mu _ { i } - \mu _ { j } ^ { \mathrm { m e s h } } \| ^ { 2 } ,\tag{7}
$$

where $\mathcal { N } ( i )$ denotes neighboring mesh points of Gaussian i.

Optimization continues until the total loss falls below a predefined threshold, ensuring both efficiency and accuracy. Dynamic learning rates adapt to the convergence rate, further stabilizing the process. By integrating adaptive initialization with dynamic optimization, this framework ensures precise Gaussian placement and enhances the fidelity of 3D scene reconstructions.

## 3.3. Dynamic Region Density Control

<!-- image-->  
Fig. 3. Loss adjustments in uniform regions enhance Gaussian distributions, improving flexibility and computational efficiency. By restricting cloning to the Non-Uniform areas, the system increases visual details in high density area only

The original Adaptive Density Control (ADC) system adjusts Gaussian density through a two-step strategy. First, transparency-based pruning identifies Gaussians with low transparency and close proximity to the camera as redundant and removes them. This reduces foreground clutter, optimizes memory usage, and enhances rendering clarity without compromising critical visual details. Mathematically, a Gaussian at position x near the camera position c is pruned if

$$
\alpha ( \mathbf { x } ) < \delta _ { d } \quad \mathrm { a n d } \quad d ( \mathbf { x } , \mathbf { c } ) < \delta _ { d } ,\tag{8}
$$

where $\alpha ( \mathbf { x } )$ represents transparency, $d ( \mathbf { x } , \mathbf { c } )$ is the Euclidean distance to the camera, and $\delta _ { d }$ is the pruning threshold. Second, gradient-based cloning addresses regions requiring enhanced resolution by cloning Gaussians with high gradients $\nabla I ( \mathbf { x } )$ and large sizes (defined by covariance matrix Î£). Cloning increases density in these areas, preserving fidelity without overloading simple regions. Cloning occurs if

$$
\nabla I ( \mathbf { x } ) > \delta _ { g } \quad \mathrm { a n d } \quad \mathrm { s i z e } ( \Sigma ) > \delta _ { s } ,\tag{9}
$$

where $\delta _ { g }$ and $\delta _ { s }$ are thresholds for gradients and Gaussian sizes, respectively. While effective, this approach applies uniform adjustments across regions, neglecting spatial variations in detail requirements, leading to uneven Gaussian distributions, visual artifacts, or reduced computational efficiency.

The Dynamic ADC system builds upon these principles by incorporating Top 20 Loss to improve region-based Gaussian adjustments and distribution control as shown in Figure 3. The scene is divided into fixed regions $\mathcal { R } _ { k } ~ ( k ~ =$ $1 , 2 , \ldots , N )$ , enabling localized evaluation and adjustment of Gaussian distributions. Using Top 20 Loss, the system prioritizes high-density regions, defined by the density ratio of the Top 20 densest to Bottom 20 sparsest sectors, ensuring critical areas receive more precise adjustments. Each region contains a subset of high-gradient Gaussians:

$$
\mathcal { G } _ { k } = \{ \mathbf x \in \mathcal { R } _ { k } \mid \nabla I ( \mathbf x ) > \delta _ { g } \} .\tag{10}
$$

Regions with significant density variance $\sigma _ { k } ^ { 2 }$ exceeding the non-uniformity threshold $\delta _ { u }$ are classified as non-uniform. These regions are adjusted by cloning Gaussians, while uniform regions $( \sigma _ { k } ^ { 2 } \le \delta _ { u } )$ are optimized through Top 20 Loss, promoting even distributions and reducing redundancy.

For non-uniform regions $( \sigma _ { k } ^ { 2 } > \delta _ { u } )$ , Gaussians are cloned to ensure adequate coverage. New Gaussians are positioned with small perturbations to maintain appropriate spatial distribution, targeting areas identified by the Top 20 Loss mechanism. For uniform regions $( \sigma _ { k } ^ { 2 } \le \delta _ { u } )$ , adjustments are applied to the loss function $\mathcal { L } _ { k }$ , specifically the dispersion term $\mathcal { L } _ { \mathrm { t o p } 2 0 } .$ , to encourage more even Gaussian distributions. The dispersion term is defined as:

$$
\mathcal { L } _ { \mathrm { t o p } 2 0 } = \frac { 1 } { | \mathcal { G } _ { \mathrm { t o p } 2 0 } | } \sum _ { \mathbf { x } _ { i } , \mathbf { x } _ { j } \in \mathcal { G } _ { \mathrm { t o p } 2 0 } } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| ^ { 2 } ,\tag{11}
$$

where $\mathcal { G } _ { \mathrm { t o p } 2 0 }$ represents the set of Gaussians within the Top 20 densest regions. This term minimizes clustering, ensuring better spatial coverage.

The Dynamic ADC system significantly enhances both efficiency and fidelity. High-gradient regions prioritized by

Top 20 Loss receive finer Gaussian distributions, improving fidelity in detailed areas while avoiding redundant placement in simpler regions. By leveraging density ratios as a guiding metric, the system dynamically adapts to scene complexities, balancing high-frequency detail in critical areas with efficient resource allocation across the scene.

This updated ADC framework transitions from a uniform gradient-transparency model to a region-sensitive control mechanism.

The Dynamic ADC system incorporates a composite loss function to achieve dynamic Gaussian distribution control, defined as:

$$
\mathcal { L } _ { \mathrm { A D C } } = \mathcal { L } _ { \mathrm { r e c o n } } ( \mathcal { R } _ { k } ) + \lambda _ { \mathrm { t o p } 2 0 } \mathcal { L } _ { \mathrm { t o p } 2 0 } ,\tag{12}
$$

where $\mathcal { L } _ { \mathrm { r e c o n } } ( \mathcal { R } _ { k } )$ represents the reconstruction loss for region $\mathcal { R } _ { k } ,$ and $\lambda _ { \mathrm { t o p } 2 0 }$ adjusts the weight of Top 20 Loss based on the regionâs detail requirements.

By incorporating region segmentation, gradient-sensitive adjustments, and Top 20 Loss, the framework achieves a refined balance between image quality and computational performance, ensuring robust adaptability for high-fidelity rendering in complex 3D scenes.

## 4. IMPLEMENTATION, RESULTS, AND EVALUATION

Ground Truth  
ours-30K  
<!-- image-->  
3DGS-30K  
Fig. 4. Qualitative comparison of reconstructed scenes for Ground Truth, ours-30K, and 3DGS-30K. The insets highlight fine details in both indoor and outdoor scenes, showing the superior reconstruction fidelity of our approach in preserving structural details

## 4.1. Implementation

All models are optimized on a single A800 GPU with 80 GB of memory. Training is divided into three stages across 30,000 iterations for efficient and accurate Gaussian splatting. Training spans 30,000 iterations divided into three phases. During the initial phase from 0 to 5,000 iterations, no regularization is applied, allowing Gaussians to adapt freely while L1 and SSIM loss ensure alignment. In the regularization phase from 5,000 to 30,000 iterations, a distance and orientation loss is applied every 100 iterations to refine alignment, with depth regularization included when depth maps are available. The density control phase from 10,000 to 30,000 iterations clones additional Gaussians in high-density areas to maintain fidelity. Link to code is available(https://github.com/ssssour/gd-3dgs).

Table 1. Quantitative evaluation of our method compared to previous work, computed across Indoor, Outdoor, All Mip360 scenes, Tanks & Temples (T&T), and Deep Blending (DB) scenes.
<table><tr><td></td><td colspan="3">Indoor Scenes</td><td colspan="3">Outdoor Scenes</td><td colspan="3">Average Across All Scenes</td><td colspan="3">Tanks &amp; Temples (T&amp;T)</td><td colspan="3">Deep Blending (DB)</td></tr><tr><td>Method</td><td>PSNR</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td><td>LPIPS</td><td>SSIM</td></tr><tr><td>Plenoxels [54]</td><td>24.83</td><td>0.426</td><td>0.766</td><td>22.02</td><td>0.465</td><td>0.542</td><td>23.62</td><td>0.443</td><td>0.670</td><td>21.08</td><td>0.379</td><td>0.719</td><td>23.06</td><td>0.510</td><td>0.795</td></tr><tr><td>INGP-Base [20]</td><td>28.65</td><td>0.281</td><td>0.840</td><td>23.47</td><td>0.416</td><td>0.571</td><td>26.43</td><td>0.339</td><td>0.725</td><td>21.72</td><td>0.330</td><td>0.723</td><td>23.62</td><td>0.423</td><td>0.797</td></tr><tr><td>INGP-Big [20]</td><td>29.14</td><td>0.242</td><td>0.863</td><td>23.57</td><td>0.375</td><td>0.602</td><td>26.75</td><td>0.299</td><td>0.751</td><td>21.92</td><td>0.305</td><td>0.745</td><td>24.96</td><td>0.390</td><td>00.817</td></tr><tr><td>Mip-NeRF360 [55]</td><td>31.58</td><td>0.182</td><td>0.914</td><td>25.79</td><td>0.247</td><td>0.746</td><td>29.09</td><td>0.210</td><td>0.842</td><td>22.22</td><td>0.257</td><td>0.759</td><td>29.40</td><td>0.245</td><td>0.901</td></tr><tr><td>GS-7K</td><td>28.95</td><td>0.222</td><td>0.901</td><td>23.70</td><td>0.321</td><td>0.668</td><td>26.32</td><td>0.272</td><td>0.785</td><td>21.20</td><td>0.280</td><td>0.767</td><td>27.78</td><td>0.317</td><td>0.875</td></tr><tr><td>3DGS-30K</td><td>31.05</td><td>0.186</td><td>0.925</td><td>24.69</td><td>0.239</td><td>0.729</td><td>27.87</td><td>0.213</td><td>0.827</td><td>23.14</td><td>0.183</td><td>0.841</td><td>29.41</td><td>0.243</td><td>0.903</td></tr><tr><td>ours-7K</td><td>30.06</td><td>0..196</td><td>00.911</td><td>24.07</td><td>0.269</td><td>0.697</td><td>26.71</td><td>0.238</td><td>0.794</td><td>22.7</td><td>0.220</td><td>0.804</td><td>28.65</td><td>0.277</td><td>0.885</td></tr><tr><td>ours-30K</td><td>31.72</td><td>0.179</td><td>0.925</td><td>25.16</td><td>0.221</td><td>0.739</td><td>28.36</td><td>0.191</td><td>0.834</td><td>24.39</td><td>0.151</td><td>0.854</td><td>30.28</td><td>0.230</td><td>0.903</td></tr></table>

Table 2. Ablation Results for Different Scenarios Across
<table><tr><td>Dataset</td><td>Mip360</td><td>Deep Blending</td><td>Tanks &amp; Temples</td></tr><tr><td>Method</td><td>PSNR SSIM LPIPS</td><td>PSNR SSIM LPIPS</td><td>PSNR SSIM LPIPS</td></tr><tr><td>NoInit</td><td>27.93 0.829 0.197</td><td>29.43 0.902 0.237</td><td>23.62 0.841 0.175</td></tr><tr><td>NoGloss</td><td>28.29 0.832 0.196</td><td>30.13 0.894 0.241</td><td>24.37 0.848 0.168</td></tr><tr><td>NoDynADC</td><td>28.18 0.832 0.193</td><td>29.89 0.896 0.243</td><td>23.74 0.852 0.171</td></tr></table>

## 4.2. Results and Evaluation

We evaluated several variations of our method on the same datasets as 3DGS, including Mip-NeRF360 [55], Tanks & Temples [56], Deep Blending [10], and the synthetic Blender dataset [2]. The tests were carried out with consistent hyperparameters. Visual quality was assessed using standard metrics (PSNR, SSIM, and LPIPS), and our method demonstrated an improved degree of structural similarity while reducing pixel-wise discrepancies, achieving superior accuracy in capturing intricate details of the scene, especially in highdetail regions than 3DGS and other state-of-the-art methods.

As shown in Table 1, our approach achieved high-quality results in as little as 7K iterations, with further improvements observed after 30K iterations. As shown in Figure 4, visual comparisons showed that our method reduced background artifacts and enhanced fine details, such as straight lines and distant window in outdoor scenes. Our method demonstrates high pixel-wise accuracy and structural integrity with minimal perceptual distortion. Compared to 3DGS-30K, it achieves improved perceptual quality, as seen in the higher SSIM and lower LPIPS values across diverse datasets, including Tanks & Temples and Deep Blending. As shown in the quantitative tables, it excels in high-detail indoor scenarios with superior PSNR and SSIM scores and exhibits strong adaptability across diverse scenes, effectively handling varying complexities. Furthermore, it illustrates that our approach rivals or outperforms leading methods, particularly in perceptual quality, structural preservation, and robustness in dynamic scenarios.

## 4.3. Ablations

Ablation studies were conducted to evaluate the contributions of our innovations. As shown in Table 2, the absence of geometric initialization led to degraded quality, particularly in background regions, highlighting the importance of structured initialization for stability as it impacts all three metrics. This underscores its critical role in maintaining scene quality across datasets. The removal of region-aware adjustments in adaptive density control resulted in uneven Gaussian distributions, mainly decreasing PSNR due to suboptimal density allocation but slightly improving SSIM and LPIPS, enhancing perceptual quality in simpler areas. Disabling surface-aligned optimization primarily worsened LPIPS while leaving PSNR and SSIM largely unaffected, demonstrating its significance for perceptual rendering quality, though its absence did not affect visual results significantly. These findings collectively highlight the importance of each component in optimizing high-complexity areas, maintaining uniformity in simpler regions, and ensuring overall scene stability and quality.

## 5. CONCLUSIONS

Our enhanced 3D Gaussian Splatting method addresses limitations in initialization, optimization, and density control, maintaining real-time performance. By using SfM data and an MLP for initialization, we achieve improved convergence and accuracy over the original 3DGS. Our approach refines Gaussian placement for higher fidelity reconstructions and introduces adaptive density control to optimize resource allocation. This method challenges continuous representations, showing explicit methods can achieve high-quality rendering with reduced training times and greater efficiency. Despite advancements, our research identifies areas for improvement, such as reducing GPU memory usage and potential for mesh reconstruction applications.

## 6. REFERENCES

[1] Bernhard Kerbl, Pavel Korshunov, Alexander Keller, Florian Bernard, Jan Eric Lenssen, and Carsten Dachsbacher, â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics (TOG), vol. 42, no. 4, pp. 1â16, Jul. 2023.

[2] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â European Conference on Computer Vision (ECCV), vol. 12347, pp. 405â421, Aug. 2020.

[3] Steven J. Gortler, Radek Grzeszczuk, Richard Szeliski, and Michael F. Cohen, âThe lumigraph,â ACM SIG-GRAPH Proceedings, vol. 30, pp. 43â54, Aug. 1996.

[4] Marc Levoy and Pat Hanrahan, âLight field rendering,â ACM SIGGRAPH Proceedings, vol. 30, pp. 31â42, Aug. 1996.

[5] Chris Buehler, Michael Bosse, Leonard McMillan, Steven Gortler, and Michael Cohen, âUnstructured lumigraph rendering,â ACM SIGGRAPH Proceedings, vol. 20, pp. 425â432, Aug. 2001.

[6] Noah Snavely, Steven M. Seitz, and Richard Szeliski, âPhoto tourism: Exploring photo collections in 3d,â ACM SIGGRAPH Proceedings, vol. 25, pp. 835â846, Aug. 2006.

[7] Michael Goesele, Noah Snavely, Brian Curless, Hugues Hoppe, and Steven M. Seitz, âMulti-view stereo for community photo collections,â International Conference on Computer Vision (ICCV), vol. 1, pp. 1â8, Oct. 2007.

[8] Gaurav Chaurasia, Sylvain Duchene, Olga Sorkine-Hornung, and George Drettakis, âDepth synthesis and local warps for plausible image-based navigation,â ACM Transactions on Graphics (TOG), vol. 32, no. 3, pp. 1â 12, Aug. 2013.

[9] Martin Eisemann, Bert De Decker, Marcus Magnor, Philippe Bekaert, Edilson De Aguiar, Naveed Ahmed, Christian Theobalt, and Anita Sellent, âFloating textures,â Computer Graphics Forum, vol. 27, no. 2, pp. 409â418, Jun. 2008.

[10] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow, âDeep blending for free-viewpoint image-based rendering,â ACM Transactions on Graphics (TOG), vol. 37, no. 6, pp. 257:1â257:15, Dec. 2018.

[11] Georgios Kopanas, Julien Philip, Thomas Leimkuhler, Â¨ and George Drettakis, âPoint-based neural rendering with per-view optimization,â Computer Graphics Forum, vol. 40, no. 4, pp. 29â43, Aug. 2021.

[12] Ayush Tewari, Justus Thies, Ben Mildenhall, Pratul P. Srinivasan, Edgar Tretschk, Christoph Lassner, Vincent Sitzmann, Ricardo Martin-Brualla, Stephen Lombardi, et al., âAdvances in neural rendering,â Computer

Graphics Forum, vol. 41, no. 2, pp. 703â735, May. 2022.

[13] Jeff P. Grossman and William J. Dally, âPoint sample rendering,â Rendering Techniques, vol. 1, pp. 181â192, Jun. 1998.

[14] Mario Botsch, Alexander Hornung, Matthias Zwicker, and Leif Kobbelt, âHigh-quality surface splatting on todayâs gpus,â Symposium on Point-Based Graphics (SPBG), vol. 10, pp. 17â24, Aug. 2005.

[15] Hanspeter Pfister, Matthias Zwicker, Jeroen Van Baar, and Markus Gross, âSurfels: Surface elements as rendering primitives,â ACM SIGGRAPH Proceedings, vol. 12, pp. 335â342, Jul. 2000.

[16] Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and Markus Gross, âSurface splatting,â ACM SIG-GRAPH Proceedings, vol. 20, pp. 371â378, Jul. 2001.

[17] Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lempitsky, âNeural pointbased graphics,â European Conference on Computer Vision (ECCV), vol. 12363, pp. 696â712, Aug. 2020.

[18] Darius Ruckert, Linus Franke, and Marc Stamminger, Â¨ âAdop: Approximate differentiable one-pixel point rendering,â ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 99:1â99:14, Jul. 2022.

[19] Christoph Lassner and Michael Zollhofer, âPulsar: Effi- Â¨ cient sphere-based neural rendering,â Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1440â1449, Jun. 2021.

[20] Thomas Muller, Alex Evans, Christoph Schied, and Â¨ Alexander Keller, âInstant neural graphics primitives with a multiresolution hash encoding,â ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 102:1â 102:15, Jul. 2022.

[21] Antoine Guedon and Vincent Lepetit, âSugar: Surface- Â´ aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 5354â5363.

[22] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee, âMulti-scale 3d gaussian splatting for anti-aliased rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[23] Lianlei Shan and Weiqiang Wang, âDensenet-based land cover classification network with deep fusion,â IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1â5, 2021.

[24] Lianlei Shan, Minglong Li, Xiaobin Li, Yang Bai, Ke Lv, Bin Luo, Si-Bao Chen, and Weiqiang Wang, âUhrsnet: A semantic segmentation network specifically for ultra-high-resolution images,â in 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021, pp. 1460â1466.

[25] Lianlei Shan, Weiqiang Wang, Ke Lv, and Bin Luo, âClass-incremental learning for semantic segmentation in aerial imagery via distillation in all aspects,â IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1â12, 2021.

[26] Lianlei Shan, Xiaobin Li, and Weiqiang Wang, âDecouple the high-frequency and low-frequency information of images for semantic segmentation,â in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 1805â1809.

[27] Lianlei Shan and Weiqiang Wang, âMbnet: A multiresolution branch network for semantic segmentation of ultra-high resolution images,â in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 2589â2593.

[28] Lianlei Shan, Weiqiang Wang, Ke Lv, and Bin Luo, âClass-incremental semantic segmentation of aerial images via pixel-level feature generation and task-wise distillation,â IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1â17, 2022.

[29] Minglong Li, Lianlei Shan, Xiaobin Li, Yang Bai, Dengji Zhou, Weiqiang Wang, Ke Lv, Bin Luo, and Si-Bao Chen, âGlobal-local attention network for semantic segmentation in aerial images,â in 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021, pp. 5704â5711.

[30] Weijia Wu, Yuzhong Zhao, Zhuang Li, Lianlei Shan, Hong Zhou, and Mike Zheng Shou, âContinual learning for image segmentation with dynamic query,â IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 6, pp. 4874â4886, 2023.

[31] Leo Shan, Wenzhang Zhou, and Grace Zhao, âIncremental few shot semantic segmentation via classagnostic mask proposal and language-driven classifier,â in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 8561â8570.

[32] Leo Shan Wenzhang Zhou Grace Zhao, âBoosting general trimap-free matting in the real-world image,â arXiv preprint arXiv:2405.17916, 2024.

[33] Lianlei Shan, Guiqin Zhao, Jun Xie, Peirui Cheng, Xiaobin Li, and Zhepeng Wang, âA data-related patch proposal for semantic segmentation of aerial images,â IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1â 5, 2023.

[34] Guiqin Zhao, Lianlei Shan, and Weiqiang Wang, âEndto-end remote sensing change detection of unregistered bi-temporal images for natural disasters,â in International Conference on Artificial Neural Networks. Springer, 2023, pp. 259â270.

[35] Lianlei Shan, Wenzhang Zhou, Wei Li, and Xingyu Ding, âLifelong learning and selective forgetting via

contrastive strategy,â arXiv preprint arXiv:2405.18663, 2024.

[36] Lianlei Shan, Shixian Luo, Zezhou Zhu, Yu Yuan, and Yong Wu, âCognitive memory in large language models,â arXiv preprint arXiv:2504.02441, 2025.

[37] Weijun Meng, Lianlei Shan, Sugang Ma, Dan Liu, and Bin Hu, âDlnet: A dual-level network with self-and cross-attention for high-resolution remote sensing segmentation,â Remote Sensing, vol. 17, no. 7, pp. 1119, 2025.

[38] Bingyun Du, Lianlei Shan, Xiaoyu Shao, Dongyou Zhang, Xinrui Wang, and Jiaxi Wu, âTransform dualbranch attention net: Efficient semantic segmentation of ultra-high-resolution remote sensing images,â Remote Sensing, vol. 17, no. 3, pp. 540, 2025.

[39] Xiaobin Li, Lianlei Shan, and Weiqiang Wang, âFusing multitask models by recursive least squares,â in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 3640â3644.

[40] Yuyang Ji and Lianlei Shan, âLdnet: Semantic segmentation of high-resolution images via learnable patch proposal and dynamic refinement,â in 2024 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2024, pp. 1â6.

[41] Lianlei Shan, Wenzhang Zhou, Wei Li, and Xingyu Ding, âOrganizing background to explore latent classes for incremental few-shot semantic segmentation,â arXiv preprint arXiv:2405.19568, 2024.

[42] Lianlei Shan, Weiqiang Wang, Ke Lv, and Bin Luo, âEdge-guided and class-balanced active learning for semantic segmentation of aerial images,â arXiv preprint arXiv:2405.18078, 2024.

[43] Xingyu Ding, Lianlei Shan, Guiqin Zhao, Meiqi Wu, Wenzhang Zhou, and Wei Li, âThe binary quantized neural network for dense prediction via specially designed upsampling and attention,â arXiv preprint arXiv:2405.17776, 2024.

[44] Xiaobin Li, Lianlei Shan, Minglong Li, and Weiqiang Wang, âEnergy minimum regularization in continual learning,â in 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021, pp. 6404â 6409.

[45] Lianlei Shan, Weiqiang Wang, Ke Lv, and Bin Luo, âBoosting semantic segmentation of aerial images via decoupled and multilevel compaction and dispersion,â IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1â16, 2023.

[46] Ruochen Pi and Lianlei Shan, âSynthetic lung x-ray generation through cross-attention and affinity transformation,â arXiv preprint arXiv:2503.07209, 2025.

[47] Xirui Zhou, Lianlei Shan, and Xiaolin Gui, âDynrslvlm: Enhancing autonomous driving perception with

dynamic resolution vision-language models,â arXiv preprint arXiv:2503.11265, 2025.

[48] Ellen Yi-Ge and Leo Shawn, âFlexdataset: Crafting annotated dataset generation for diverse applications,â in Proceedings of the AAAI Conference on Artificial Intelligence, 2025, vol. 39, pp. 9481â9489.

[49] Chengsong Sun, Weiping Li, Xiang Li, Yuankun Liu, and Lianlei Shan, âGmm-based comprehensive feature extraction and relative distance preservation for few-shot cross-modal retrieval,â arXiv preprint arXiv:2505.13306, 2025.

[50] Hailong Luo, Bin Wu, Hongyong Jia, Qingqing Zhu, and Lianlei Shan, âLlm-cot enhanced graph neural recommendation with harmonized group policy optimization,â arXiv preprint arXiv:2505.12396, 2025.

[51] Shixian Luo, Zezhou Zhu, Yu Yuan, Yuncheng Yang, Lianlei Shan, and Yong Wu, âGeogrambench: Benchmarking the geometric program reasoning in modern llms,â arXiv preprint arXiv:2505.17653, 2025.

[52] Qiang Yi and Lianlei Shan, âGeolocsft: Efficient visual geolocation via supervised fine-tuning of multimodal foundation models,â arXiv preprint arXiv:2506.01277, 2025.

[53] Hengzhi Chen, Liqian Feng, Wenhua Wu, Xiaogang Zhu, Shawn Leo, and Kun Hu, âF2net: A frequencyfused network for ultra-high resolution remote sensing segmentation,â arXiv preprint arXiv:2506.07847, 2025.

[54] David B. Fridovich-Keil, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa, âPlenoxels: Radiance fields without neural networks,â CVPR Proceedings, vol. 41, pp. 5491â5500, Jun. 2022.

[55] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â CVPR Proceedings, pp. 5670â5679, Jun. 2022.

[56] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun, âTanks and temples: Benchmarking large-scale scene reconstruction,â ACM Transactions on Graphics (TOG), vol. 36, no. 4, pp. 1â13, Jul. 2017.