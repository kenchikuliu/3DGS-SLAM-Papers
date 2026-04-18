# From Slices to Structures: Unsupervised 3D Reconstruction of Female Pelvic Anatomy from Freehand Transvaginal Ultrasound

Max Krahenmann, Sergio Tascon-Morales, Fabian Laumer, Julia E. Vogt, and Ece Ozkan Â¨

Abstractâ Volumetric ultrasound has the potential to significantly improve diagnostic accuracy and clinical decision-making, yet its widespread adoption remains limited by dependence on specialized hardware and restrictive acquisition protocols. In this work, we present a novel unsupervised framework for reconstructing 3D anatomical structures from freehand 2D transvaginal ultrasound sweeps, without requiring external tracking or learned pose estimators. Our method, TVGS, adapts the principles of Gaussian Splatting to the domain of ultrasound, introducing a slice-aware, differentiable rasterizer tailored to the unique physics and geometry of ultrasound imaging. We model anatomy as a collection of anisotropic 3D Gaussians and optimize their parameters directly from image-level supervision. To ensure robustness against irregular probe motion, we introduce a joint optimization scheme that refines slice poses alongside anatomical structure. The result is a compact, flexible, and memory-efficient volumetric representation that captures anatomical detail with high spatial fidelity. This work demonstrates that accurate 3D reconstruction from 2D ultrasound images can be achieved through purely computational means, offering a scalable alternative to conventional 3D systems and enabling new opportunities for AI-assisted analysis and diagnosis.

## I. INTRODUCTION

Three-dimensional (3D) ultrasound (US) imaging plays an essential role in obstetrics and gynecology, providing clinicians with a spatial context that is crucial for diagnosis, surgical planning, and treatment monitoring [1], [2]. Transvaginal ultrasound (TVS) is widely used for high-resolution imaging of pelvic organs due to its affordability, safety, and minimally invasive nature, but extending 2D TVS to 3D imaging remains technically challenging. Existing 3D US systems typically rely on external tracking mechanisms or mechanically swept probes. However, while 3D ultrasound imaging is commonly performed both transabdominally and transvaginally, especially in obstetric applications such as fetal imaging, most commercial 3D systems are designed for abdominal or general freehand use, not specifically for transvaginal gynecological contexts [3], [4]. In addition to being physically cumbersome and difficult to integrate into routine clinical workflows, these systems often suffer from lower resolution compared to conventional 2D imaging. The reliance on mechanical scanning or external tracking can introduce motion artifacts, particularly in dynamic or less controlled scanning environments. Furthermore, the need for specialized hardware increases system complexity and cost, which hinder widespread adoption.

Recent advances have explored machine learning-based methods for reconstructing 3D volumes from freehand 2D ultrasound sweeps. These approaches include pose estimation using speckle correlation, deep neural networks, and selfsupervised trajectory inference [5], [6], [7]. While promising in specific domains, such methods are highly dependent on the training data, sensitive to imaging artifacts, and often lack generalizability across different acquisitions or anatomical regions. Current approaches still rely on generating pose information prior to reconstruction, which introduces an additional layer of complexity [8].

In this work, we propose TVGS: an unsupervised method for 3D US reconstruction that is free from the need for any external tracking or prior learned pose estimation. Our method is specifically designed for the gynecological domain and operates on standard 2D TVS sweeps acquired freehand. It leverages recent advances in volumetric rendering, adapting Gaussian Splatting [9] to the constraints and physics of US imaging. Our approach uses sensorless geometric modeling of probe motion to establish spatial coherence between slices, and reconstructs anatomical structures using anisotropic Gaussian primitives optimized directly from US data.

Our framework represents anatomy as a collection of anisotropic Gaussians, optimized via a custom CUDAoptimized differentiable rasterizer. This continuous representation mitigates the aliasing and staircase artifacts inherent to discrete voxel-based methods, particularly in off-axis slices, while achieving convergence speeds significantly faster than implicit neural representations. Importantly, our method requires no calibration data and does not rely on training with large datasets.

We validate our method on real-world clinical TVS data, reconstructing 3D volumes from two orthogonal freehand sweeps. Despite being unsupervised and model-free, our approach produces anatomically accurate volumes that clearly capture key structures such as the uterus and endometrium, with strong visual and spatial fidelity. Our method also demonstrates robustness under realistic acquisition conditions and shows potential for real-time performance. In summary, our contributions are:

â¢ An unsupervised 3D reconstruction method from 2D TVS sweeps, requiring no external tracking or learned pose estimators.

â¢ A custom volumetric framework using Gaussian Splatting, adapted for US-specific imaging physics.

â¢ A CUDA-optimized differentiable rasterizer tailored for slice-based rendering under highly anisotropic sampling (dense in-plane, sparse through-plane).

â¢ A joint optimization strategy that refines slice poses alongside scene parameters, increasing robustness to irregular probe motion typical in freehand acquisition.

## II. RELATED WORK

2D to 3D US reconstruction has been approached using a variety of strategies. These approaches differ significantly in their reliance on external tracking hardware, training data, and assumptions about probe motion, each carrying implications for their suitability in clinical workflows, particularly in TVS.

Early systems using electromagnetic sensors [10] provided accurate positioning but are ill-suited for TVS due to bulk and patient discomfort [5]. Consequently, sensorless methods emerged, leveraging signal processing techniques like speckle decorrelation [11], [12], [13] and spatial correlation [14] to infer probe movement. While foundational, these approaches often struggle with noise and complex multi-axis motion. Later statistical and graph-based optimizations [15] improved robustness but remained too fragile for widespread clinical adoption.

With the rise of deep learning, supervised learning methods for pose estimation gained popularity. These approaches train convolutional or recurrent neural networks to predict either inter-frame transformations or directly reconstruct 3D volumes. Early models demonstrated that CNNs could infer relative probe motion from US frames, enabling trackerless 3D reconstruction [6]. Later methods like DC2-Net [16] introduced contrastive and temporal losses to improve trajectory consistency across sequences. Another noteworthy model is PLPPI, which incorporates domain-specific physics constraints into the learning pipeline, enabling improved handling of outof-plane motion [17]. However, a shared limitation of these models is their dependency on ground truth data from tracked acquisitions for training and evaluation. In many clinical workflowsâparticularly for TVSâsuch tracking data is unavailable or infeasible to collect. Some recent works like RecON [18] explore online learning, updating models in real-time as new frames are acquired. While promising for reducing drift and latency, these methods still presuppose access to prior 3D volumes or strong priors, making them unsuitable for fully unsupervised clinical settings.

In parallel, implicit representations like Neural Radiance Fields (NeRF) [19] and subsequent developments (e.g., ImplicitVol [4]) model 3D structures as continuous functions learned by neural networks. These models offer resolution independence and are memory-efficient but typically require extensive training and known camera posesâconditions that are generally not met in US unless a tracking device is present. Hybrid methods such as RapidVol [7] attempt to blend implicit and explicit approaches, using tri-planar maps refined by lightweight networks to accelerate training and inference. However, even these models typically assume access to 3D volumes for training or initialization.

Beyond motion estimation, the choice of volumetric representation plays a central role in reconstruction quality and efficiency. Traditional explicit methods such as voxel grids offer high interpretability and easy integration into existing medical imaging pipelines. However, they suffer from cubic memory growth with resolution and are thus computationally intensive for fine anatomical detail. More recently, techniques like Gaussian Splatting have emerged as a powerful alternative. Originally developed for real-time rendering in computer vision [9], Gaussian Splatting represents scenes using parameterized Gaussian primitives whose locations, shapes, and radiance are optimized to match rendered views. UltraGauss [8] adapted this to fetal US imaging by incorporating anisotropic Gaussians and integrating pose predictions from QAERTS [20]. This approach improves rendering quality and speed but still depends on learned pose models, which hinders generalization and deployment in tracking-free environments.

In contrast to these existing approaches, our method introduces the first application of Gaussian Splatting for 3D US that is entirely unsupervised and sensorless. We eliminate the need for pose estimation models, supervision, or tracking hardware by leveraging geometric assumptions on sweep motion and optimizing anisotropic Gaussian primitives directly from 2D TVS slices. This enables anatomically accurate, scalable 3D reconstruction using only standard 2D US equipment, significantly lowering the barrier to clinical deployment and advancing the capabilities of low-cost US imaging.

## III. DATASETS

To develop and evaluate our unsupervised 3D reconstruction framework, we utilize both real and synthetic TVS data. Synthetic data offers controlled conditions with ground-truth pose labels for benchmarking, while real clinical sweeps provide representative anatomical complexity and scanning variability.

## A. Synthetic Data

To support development and controlled evaluation, we construct a synthetic dataset. Starting from a 3D uterus mesh in STL format, we voxelized the model into a segmentation volume with three labels: interior (0.5), border (1), and exterior (0). Using 3D Slicer [21], slices are generated by simulating probe rotations through angular sweeps, ranging from â60â¦ to $+ 6 0 ^ { \circ }$ in sagittal and transversal directions, as shown in Fig. 1. To reflect realistic acquisition scenarios, the slices are sparsely sampled along each sweep, with a limited number of angular steps per direction, resulting in a coarse and uneven spatial coverage of the volume.

<!-- image-->  
Fig. 1. Top: Simulated sagittal sweep. Bottom: Simulated transversal sweep. The image only shows some of the generated probe rotations.

Each slice is associated with a known 6D pose vector:

$$
\mathbf { y } = [ r _ { x } , r _ { y } , r _ { z } , t _ { x } , t _ { y } , t _ { z } ] ,\tag{1}
$$

where r and t denote Euler angles and translations with respect to a canonical coordinate system. This ground truth facilitates quantitative evaluation of slice pose and 3D reconstruction accuracy. Examples of generated frames are shown in Fig. 2 for sagittal (left) and transversal (right).

<!-- image-->  
Fig. 2. Synthetic ultrasound slices generated from a simulated uterus model using a renderer. Left: Sagittal view showing longitudinal anatomy. Right: Transversal view perpendicular to the sagittal plane. These slices can be sampled at arbitrary resolution.

## B. Real Data

20 TVS sweeps were acquired from 10 patients (2 sweeps per patient) using a GEHC Voluson Expert 22 ultrasound machine. Table I summarizes the main parameters of the data. Each sweep represents either a sagittal or transversal view, with probe motion approximated as a smooth angular trajectory. An example image for both modalities can be seen in Fig. 3. Each patient provided written informed consent prior to participation, and all procedures adhered to strict ethical standards to ensure confidentiality and privacy. Since real data lacks tracked probe poses, we assume consistent frame intervals and smooth probe motion. We initialize the pose of each frame using equal angular spacing. This assumption mirrors the angular sampling strategy used in generating synthetic data. Other than resizing and padding images when needed, no additional transformations or data augmentation techniques are applied, ensuring that the intrinsic properties of the images are preserved. Our method assumes (i) temporally ordered slices with approximately uniform angular spacing, and (ii) minimal tissue deformation during the sweep. These assumptions are critical for effective sensorless reconstruction, and we analyze their impact in Section IV.

<!-- image-->  
Fig. 3. Two frames sampled from the real data sweeps. On the left, a frame taken from the sagittal view. On the right, a frame taken from the transversal view.

## C. Test Set

For quantitative evaluation, we establish a distinct test set partition. Although the raw sweep lengths vary, we standardize our evaluation by subsampling 85 equidistant slices from each sweep. From these, we select 10 slices uniformly at random to serve as the test set and keep the remaining 75 for training (we analyze the impact of different slice counts in Section V-C.3). Crucially, because the ultrasound slices represent thin, planar cross-sections with sparse angular spacing, these held-out test slices do not spatially overlap with any of the remaining slices used for training. Consequently, this evaluation setup is particularly challenging; it effectively constitutes a task of novel view synthesis (NVS), requiring the model to accurately infer anatomical structures in the gaps between training views without direct supervision. As we do not have accurate poses for NVS, we perform a short optimization of them before evaluation to factor out pose error, consistent with standard practices for pose estimation in neural rendering frameworks [22].

## IV. UNSUPERVISED GAUSSIAN SPLATTING FOR ULTRASOUND

## A. Overview: From Gaussian Splatting to Ultrasound Imaging

Gaussian Splatting is a recent real-time rendering method that represents a 3D scene using volumetric Gaussian primitives [9]. Each Gaussian contributes to pixels in projected views based on its position, shape (covariance), opacity, and radiance. The method is differentiable and well-suited for learning-based optimization, particularly in settings where redundant multi-view image data is available (e.g., RGB images from moving cameras).

However, US imaging presents fundamental differences. Each image (or âsliceâ) corresponds to a narrow, planar cross-section of tissue, implying limited overlap between any two slices. In TVS, this limitation is even more pronounced as there is no overlap between slices in the sweep. As a result, there is no redundancy between views to help maintain scene consistency. Additionally, the physics of US introduces directionality and discontinuities that standard perspective projection does not capture. This stands in contrast to Gaussian Splatting for optical images, where we work with projections of the scene onto a camera sensor, inherently providing depth information. Since US does not have this projection component and instead captures direct cross-sections, approaches relying on projection geometry cannot be directly applied. Misaligned Gaussian primitivesâespecially in depthâcan significantly affect the rendering, since each slice is extremely thin. To address these challenges, we adapt Gaussian Splatting, building a slice-aware differentiable rasterizer specifically designed for US data. Our framework accounts for the sparse and largely non-overlapping slices while retaining the strengths of the Gaussian formulationâcompact, continuous scene representation and differentiability.

TABLE I  
MAIN PARAMETERS OF REAL DATA. VALUES IN PARENTHESES REPRESENT THE STANDARD DEVIATION.
<table><tr><td>Parameter</td><td>Value(s)</td></tr><tr><td>Frames per second</td><td>10-20</td></tr><tr><td>Duration</td><td>~10 seconds</td></tr><tr><td>Frames per sweep</td><td>100-300</td></tr><tr><td>Plane</td><td>Sagittal / Transversal</td></tr><tr><td>Original Resolution</td><td>1528 x 784</td></tr><tr><td>Age</td><td>41.9 (6.0)</td></tr><tr><td>Height [m]</td><td>1.67 (0.07)</td></tr><tr><td>Weight [kg]</td><td>68.49 (13.26)</td></tr><tr><td>Ethnicity</td><td>White/European</td></tr></table>

<!-- image-->  
Fig. 4. Overview of our differentiable Gaussian Splatting pipeline for ultrasound volume reconstruction. A set of 3D Gaussiansâeach defined by a position, shape, intensity, and opacityâis rendered onto input ultrasound slice planes via a custom rasterizer. A reconstruction loss drives gradient-based optimization of all Gaussian parameters. The framework includes strategies for initialization, pruning, and performance optimization.

An overview of our method is shown in Fig. 4. We represent anatomy as a set of volumetric 3D Gaussians, which are rendered onto slice planes using a custom differentiable rasterizer. This enables end-to-end optimization of all parameters using only image-level supervision from sparse slices.

## B. Slice-Based Differentiable Rasterization

Our rasterizer computes the intensity of each pixel in a slice by aggregating contributions from a set of 3D Gaussian primitives. We begin by describing the formulation in terms of its constituent steps. Each Gaussian g is defined by a set of parameters:

$\mu _ { g } \in \mathbb { R } ^ { 3 } \colon$ the center of the Gaussian,

$\bar { \Sigma _ { g } } \in \mathbb { R } ^ { 3 \times 3 }$ : the covariance matrix describing its shape and orientation. It is factored in the same way as in the original GS formulation [9]: $\Sigma _ { g } = ( S _ { g } R _ { g } ) ^ { \top } ( S _ { g } R _ { g } )$ , where $S _ { g }$ is a diagonal scale matrix and $R _ { g }$ is a rotation matrix derived from a unit quaternion,

$o _ { g } \in [ 0 , 1 ]$ ]: the opacity, controlling transparency,

$I _ { g } \in \mathbb { R }$ : the scalar intensity value associated with the Gaussian.

Each pixel p in a US slice is associated with a known 3D coordinate $\mathbf { c } _ { p } \in \mathbb { R } ^ { 3 }$ , obtained via slice pose modeling as described in Section III-B and the location of the pixel in the slice. Notably, $\mathbf { c } _ { p }$ is a differentiable function of the sliceâs rigid body transformation parameters. Consequently, gradients computed at the pixel level during the backward pass flow not only to the Gaussian parameters but also to the extrinsic matrix of the slice, enabling the correction of misalignment errors. For each Gaussian $^ { g , }$ we compute the offset vector from the Gaussian center to the pixel coordinate:

$$
\begin{array} { r } { \mathbf { d } _ { g p } = \mathbf { c } _ { p } - \mu _ { g } , } \end{array}\tag{2}
$$

This displacement indicates how far and in which direction pixel p lies from the Gaussian center. We use the Mahalanobis distance to compute how âcloseâ the pixel is to the Gaussian in the metric induced by the covariance:

$$
\begin{array} { r } { e _ { g p } = - \frac 1 2 \mathbf { d } _ { g p } ^ { \top } \Sigma _ { g } ^ { - 1 } \mathbf { d } _ { g p } . } \end{array}\tag{3}
$$

This term governs the spatial falloff of the Gaussianâs influence. A higher value of $e _ { g p }$ (closer to 0) means greater contribution. The influence of Gaussian $g$ at pixel $p$ is then:

$$
\alpha _ { g p } = o _ { g } \exp ( e _ { g p } ) ,\tag{4}
$$

where $o _ { g }$ scales the exponential according to the visibility (opacity) of the Gaussian. The final intensity $I _ { p }$ at pixel $p$ is computed as a weighted sum of intensities from all Gaussians:

$$
I _ { p } = \sum _ { g } \alpha _ { g p } I _ { g } .\tag{5}
$$

This formulation defines each pixel as the sum of soft and spatially-varying influences from 3D Gaussians. By using Gaussian functions instead of hard surface projections, the method can represent continuous, soft-tissue-like structures and remains differentiable for optimization. It is especially well-suited for sparse data with anisotropic spatial sampling such as US slices, where direct projection is unreliable and information across slices is limited.

To enable efficient end-to-end optimization of this representation, we implement the rasterizer as a high-performance CUDA kernel. The custom nature of this slice-accumulation process precludes the use of standard automatic differentiation frameworks like PyTorchâs autograd. Instead, we analytically derive the gradients for all Gaussian parameters and implement them directly within the kernel using the chain rule. This manual backward pass also avoids heavy automatic differentiation graphs, resulting in significant memory savings.

## C. Parameter Initialization

Effective initialization of Gaussian parameters is crucial to ensure stable and efficient optimization. This is particularly important because the contribution of each Gaussian to the rendered image, thus, the magnitude of its gradient updatesâdepends multiplicatively on both its intensity $I _ { g }$ and opacity $o _ { g } .$ If these parameters are poorly initialized (e.g., near zero), the Gaussian becomes effectively invisible and receives negligible gradient updates, rendering it âinactiveâ from the outset. If initialized randomly, we could potentially run into such issues; the following values were found to work well empirically for our setting:

â¢ For moderate initial opacity, $o _ { q } = \sigma ( 1 . 0 ) \approx 0 . 7 3 ,$

â¢ For mid-range intensity, $I _ { g } = 0 . 5 ,$ â¢ For moderate spatial extent, $\mathbf { s } _ { g } = \exp ( [ 0 . 5 , 0 . 5 , 0 . 5 ] ) ^ { \top }$ â [1.65, 1.65, 1.65]â¤

â¢ For identity rotation, $\mathbf { q } _ { g } = [ 1 , 0 , 0 , 0 ] ^ { \top }$

This choice of initialization ensures that each Gaussian has a non-negligible influence on the rendered slice at the start of training, gradients with respect to spatial and shape parameters are not suppressed, and training begins from a smooth, isotropic configuration, avoiding early instability from sharp or ill-posed Gaussians. Unlike opacity and intensity, the mean position $\mu _ { g } \in \mathbb { R } ^ { 3 }$ is highly sensitive to initialization. Gaussians must be spatially close to the slices they are meant to influence; otherwise, their contribution to the rendered image (and thus their gradient signal) is exponentially suppressed via the term $\exp ( e _ { g p } )$ . In practice, we found that poor initialization of $\mu _ { g }$ leads to Gaussians becoming permanently inactive before they can migrate toward meaningful regions. We consider two strategies for initializing $\mu _ { g } .$

- On-slice initialization: Given a set of slice posesâwhether inferred via our sensorless motion model or obtained from external trackingâwe distribute Gaussian centers uniformly across the image planes. This places primitives directly within the signal domain from the start, ensuring strong initial gradient updates and accelerating convergence.

- Uniform sampling in a bounding volume: Alternatively, we sample Gaussian centers uniformly within a predefined 3D bounding box enclosing the volume of interest. While this strategy is agnostic to the initial slice configuration, it typically requires additional optimization iterations for primitives to migrate from empty space toward informative anatomical regions.

## D. Inactive Gaussians and Density Control

During training, certain Gaussians may become effectively âinactiveââtypically those initialized in regions of the volume that are consistently dark or far from any informative US data. These Gaussians tend to receive low gradient signals, leading to rapid decay of their opacity $o _ { g }$ and intensity $I _ { g }$ toward zero. Once these values are sufficiently small, the Gaussianâs contribution to the image becomes negligible, which in turn suppresses all subsequent gradient updates to its position, shape, and rotation. In effect, these Gaussians are permanently frozen in an uninfluential state.

Such inactive Gaussians waste computational resources. To mitigate this, we incorporate a lightweight density control mechanism that periodically removes such Gaussians and replaces them with new ones better positioned to contribute meaningfully.

We define a simple but effective scalar activity metric:

$$
m _ { g } = | I _ { g } \cdot o _ { g } |
$$

Gaussians for which $m _ { g } \ < \ \epsilon$ (where Ïµ is a low predefined threshold) are considered inactive. These are removed from the parameter set every N optimization steps. This process is efficient to implement and avoids the overhead of evaluating gradient norms or rendering error sensitivity for Gaussians.

To maintain volumetric capacity, we re-seed an equivalent number of new Gaussians, sampling their means uniformly within the bounding box of currently active primitives $( \mu _ { \mathrm { n e w } } \sim$ $\mathcal { U } [ \mu _ { \mathrm { m i n } } , \mu _ { \mathrm { m a x } } ] )$ . This simple heuristic leverages the existing distribution of active Gaussians to guide new candidates toward potentially informative regions, without requiring a full analysis of image-space reconstruction error. Other parameters are reset to default values as described in Sec. IV-C.

As a complementary strategy, we apply a non-uniform learning rate schedule where the Gaussian means $\mu _ { g }$ are updated more aggressively than other parameters (e.g., opacity, intensity, and rotation). The motivation is to allow Gaussians to quickly migrate toward regions where they influence the rendered image before their visibility can be âshut offâ by the optimizer. In more extreme variants, we temporarily freeze non-positional parameters entirely during early training epochs. This prevents premature deactivation and leads to more effective exploration of the volume.

Both the pruningâreinitialization mechanism and learning rate scheduling are essential for maintaining a healthy and responsive population of Gaussians during training.

## E. Joint Pose and Scene Optimization

A core challenge in sensorless freehand ultrasound is the variability in probe motion. While we initialize our reconstruction assuming a smooth, linear angular sweep, human hand motion inevitably introduces irregularities. To address this, we decouple the slice poses from the fixed geometric prior. We treat the pose parameters $\mathbf { y } _ { i } ~ = ~ \left[ \mathbf { r } _ { i } , \mathbf { t } _ { i } \right]$ for each slice i as learnable parameters. During the backward pass, gradients are propagated through the rasterizer not only to the Gaussian primitives but also to the transformation matrices defining the slice positions. This allows the model to fine-tune the alignment of individual slices to maximize photometric consistency with the reconstructed volume. We apply a lower learning rate to pose parameters compared to scene parameters to prevent drift and maintain the global structure defined by the initialization.

## F. Practical Considerations for Implementation

To make training tractable at scale, we apply a series of CUDA-level optimizations across both the forward and backward passes of the differentiable rasterizer. In particular, we precompute invariant quantities such as inverse covariance matrices $( \Sigma _ { g } ^ { - 1 } )$ , which can then be reused across all affected pixels. We also adopt shared memory tiling, where Gaussian parameters are loaded once per tile and reused by all threads in a block, thereby minimizing global memory traffic. Further efficiency is gained through symmetry-aware matrix handling, which exploits the symmetric structure of covariance matrices to reduce both memory usage and compute overhead, and through matrix-free computation strategies that replace expensive general-purpose matrix operations with lightweight element-wise arithmetic. Together, these optimizations improve training throughput by an order of magnitude compared to a naive implementation and are critical for scaling to hundreds of thousands of Gaussians in high-resolution 3D volumes.

## V. EXPERIMENTS AND RESULTS

## A. Experimental Setup

We followed standard practices for image resolution and batch size, while empirically tuning remaining settings to ensure high fidelity. Unless otherwise specified, we train for 150 epochs with a batch size of 32. We use a resolution of 256 Ã 256 for synthetic data and 256 Ã 128 for real data, the latter selected to match the datasetâs native 2:1 form factor. The scenes are represented with approximately 50k Gaussians, and optimized using a hybrid loss $( \lambda _ { \mathrm { L 1 } } = 0 . 8 , \lambda _ { \mathrm { S S I M } } = 0 . 2 )$ with learning rates 0.2 (means), 0.03 (opacity), 0.01 (scale), and 0.008 (intensity). The selection of specific values is explained in more detail in Sec. V-B. We divide the experiments into optimizations (Sec. V-B) and reconstruction quality (Sec. V-C). To comprehensively evaluate model performance, we report standard image similarity metrics: Structural Similarity Index Measure (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Learned Perceptual Image Patch Similarity (LPIPS), using their conventional formulations.

## B. Optimizations

In this section, we evaluate the impact of various method configurations on reconstruction performance. First, we explored different training objectives, testing L1, L2, SSIM, PSNR, NCC, LPIPS and hybrid combinations. We found that a hybrid configuration $( \lambda _ { \mathrm { L 1 } } = 0 . 8 , \lambda _ { \mathrm { S S I M } } = 0 . 2 )$ yielded the best results, offering superior robustness across metrics compared to the other tested functions. Consequently, we select this loss for all subsequent experiments.

## 1) Results of Optimization Techniques:

a) Gaussian Mean initialization: To evaluate the suitability of each of the initialization approaches described in Sec. IV-C, we run experiments and report the SSIM, PSNR and LPIPS for each case, as shown in Table II. Fig. 5 shows an example of the resulting scene obtained through uniform random initialization.

TABLE II  
SSIM, PSNR AND LPIPS METRICS ON RUNS WITH DIFFERENTINITIALIZATION FOR SAGITTAL AND TRANSVERSAL VIEW OPTIMIZATION.
<table><tr><td>Group</td><td>Init. method</td><td>SSIM (â)</td><td>PSNR (â)</td><td>LPIPS(â)</td></tr><tr><td>Sagittal</td><td>Random</td><td>0.960</td><td>26.49</td><td>0.083</td></tr><tr><td></td><td>On-Slice</td><td>0.975</td><td>28.54</td><td>0.057</td></tr><tr><td>Transversal</td><td>Random</td><td>0.974</td><td>28.25</td><td>0.062</td></tr><tr><td></td><td>On-Slice</td><td>0.983</td><td>30.38</td><td>0.047</td></tr></table>

As shown in Table II, on-slice initialization yields consistently superior metrics and is selected as the default method. We attribute this performance gap to the sparse nature of US slice data. Random initialization frequently places primitives in empty space far from the image planes; these primitives contribute negligible density to the rendering, resulting in vanishing gradients that prevent them from migrating toward anatomical structures. In contrast, on-slice initialization guarantees that primitives spatially overlap with valid signal from the first iteration. This ensures immediate gradient flow and prevents primitives from becoming effectively âinactiveâ at the start of trainingâa challenge we analyze further in the following section on density control.

<!-- image-->  
Fig. 5. Example of a reconstructed 3D view scene after uniform random initialization of means.

<!-- image-->  
Fig. 6. Mean SSIM, PSNR and LPIPS vs. training time for different numbers of Gaussians representing the scene.

b) Number of Gaussians: The number of Gaussian primitives directly influences both reconstruction quality and computational cost. Increasing this number provides the model with more flexibility to capture fine anatomical details, potentially enhancing reconstruction fidelity. However, this improvement comes with a higher memory footprint and significantly longer training times.

To explore this trade-off, we systematically varied the number of Gaussians on a fixed set of real data sweeps. We monitored reconstruction quality metric alongside the total training time, as shown in Fig. 6. As expected, a higher number of Gaussians generally leads to improved reconstruction quality, reflected in lower L1 loss and higher SSIM scores. At the same time, training time increases substantially due to the added computational burden per iteration.

For the tested 256x128 resolution images, the results suggest an optimal range between approximately 60,000 Gaussians (about 600 per slice in the test setup) and 100,000 Gaussians (about 1000 per slice), which appears to strike a good balance between reconstruction quality and computational efficiency. Beyond roughly 100,000 Gaussians, the gains in L1 and SSIM begin to plateau, indicating diminishing returns, while the execution time continues to rise significantly. To strike a good balance between performance and execution time during experiments, we use approximately 50k Gaussians.

c) Inactive Gaussians: As described before, inactive Gaussians can limit the effect of a large number of primitives on the final reconstruction. We apply reinitialization of Gaussians with spatial awareness, as described in Sec. IV-D, using different thresholds for pruning $( m _ { g } = | I _ { g } o _ { g } |$ values of 0.2, 0.1, 0.05, and 0.01) and test several schedules for when to apply the removal and addition steps. The best results were obtained using a threshold of 0.05 and applying density control every 25 epochs, starting at epoch 50 and continuing until epoch 125.

We observed that this mechanism introduces transient spikes in the loss function immediately following each pruning and reseeding step, though the optimization stabilizes rapidly. Quantitatively, the strategy yields minor improvements, primarily when using random initialization where the risk of inactive primitives is higher. In this setting, applying density control improved SSIM from 0.974 to 0.978, PSNR from 28.25 dB to 28.82 dB, and LPIPS from 0.062 to 0.053. Thus, while not strictly necessary for convergenceâespecially given the robustness of our default on-slice initializationâit serves as a useful refinement to maximize performance.

2) Robustness to Motion Jitter: Real-world sweeps often contain âjitterââsmall deviations in speed and angle due to hand tremor or irregular motion. To evaluate our methodâs robustness, we introduced synthetic jitter to the simulated dataset. We define jitter as random noise added to the ground truth poses, scaled by a percentage of the total sweep extent (e.g., 2.5% jitter corresponds to deviations up to 2.5% of the total rotation range and image dimensions).

Since pose optimization is an optional component of our pipeline (Sec. IV-E), we study it as an ablation to assess how it improves robustness to motion jitter. Quantitative results for increasing jitter levels are summarized in Table III.

As expected, introducing motion jitter degrades the performance of the fixed-pose baseline. At 2.5% jitterâa level consistent with a reasonably steady handâthe test SSIM drops from 0.971 to 0.931.

Activating learnable poses allows the model to correct for these deviations. Pose optimization significantly recovers performance, boosting Test SSIM from 0.931 back to 0.952, increasing PSNR and reducing LPIPS error. This demonstrates that our method can self-calibrate small motion artifacts without external tracking.

At extreme jitter levels (5%), the optimization struggles to recover the correct geometry, likely getting trapped in local minima where the misalignment is too large for the gradients to guide the slices back to coherence. Interestingly, at this high noise level, the fixed-pose baseline outperforms the optimized version on the test set. This suggests that for highly erratic motion, the strong regularization of the smooth geometric prior is safer than attempting to optimize poses. However, for standard clinical sweeps, joint optimization provides a significant boost in fidelity.

## C. Reconstruction Quality

1) View-Specific Reconstructions on Synthetic Data: Based on insights gained from earlier experiments, we now evaluate the reconstruction performance of our method more closely. We first evaluate our method on the synthetic dataset to validate geometric accuracy in a controlled environment. Using the default configuration of approximately 50,000 Gaussians, the optimization converges rapidly, requiring approximately 8 minutes for each view (sagittal and transversal).

TABLE III  
EFFECT OF LEARNED POSES ON DATA AFFECTED BY DIFFERENT PERCENTAGES OF JITTER.
<table><tr><td></td><td colspan="3">Training</td><td colspan="3">Test</td></tr><tr><td></td><td>SSIM</td><td>PSNR LPIPS</td><td></td><td>SSIM</td><td>PSNR</td><td>LPIPS</td></tr><tr><td>No Jitter</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TVGS</td><td>0.978</td><td>28.36</td><td>0.053</td><td>0.972</td><td>28.51</td><td>0.058</td></tr><tr><td>TVGS w/o pose opt.</td><td>0.975</td><td>28.54</td><td>0.057</td><td>0.971</td><td>28.83</td><td>0.059</td></tr><tr><td>2.5% Jitter</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TVGS</td><td>0.950</td><td>23.06</td><td>0.099</td><td>0.952</td><td>23.88</td><td>0.089</td></tr><tr><td>TVGS w/o pose opt.</td><td>0.942</td><td>21.13</td><td>0.106</td><td>0.931</td><td>20.44</td><td>0.126</td></tr><tr><td>5% Jitter</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TVGS</td><td>0.927</td><td>19.47</td><td>0.122</td><td>0.899</td><td>18.62</td><td>0.159</td></tr><tr><td>TVGS w/o pose opt.</td><td>0.923</td><td>19.06</td><td>0.126</td><td>0.913</td><td>19.12</td><td>0.145</td></tr></table>

Qualitative Results. Fig. 7 illustrates the reconstruction fidelity. The model successfully captures the scene geometry, producing visually faithful renderings. We observe minor artifacts near the edges of structures, which we attribute to the sharp, non-smooth intensity transitions characteristic of binary segmentation masks used to generate the synthetic ground truth. Fig. 8 displays the resulting 3D volumetric representations. While individual sagittal or transversal sweeps capture the general structure, they exhibit cone-shaped artifacts due to the limited angular coverage of the ultrasound probe (spanning Â±60Â°). Combining both views minimizes these occlusions, resulting in a more complete 3D reconstruction.

Comparison with Implicit Representations. Benchmarking unsupervised 3D US reconstruction is inherently challenging due to the lack of standard baselines that operate without external tracking. We compare our approach against ImplicitVol [4], a state-of-the-art implicit neural representation method. To ensure a fair comparison in our tracking-free setting, we adapted ImplicitVol to use the slice poses estimated by our framework rather than pre-computed tracked poses.

Quantitative results are summarized in Table IV. Our method achieves high structural similarity and peak signalto-noise ratios, confirming the validity of the reconstruction. We observe that ImplicitVol achieves marginally higher quantitative scores (SSIM and PSNR) and lower perceptual error (LPIPS). This likely stems from the continuous nature of the MLPs in ImplicitVol, possessing an inductive bias towards smoothness well-suited for perfect, noise-free synthetic surfaces. In contrast, our discrete Gaussian representation may introduce minor high-frequency noise when approximating these perfectly smooth boundaries.

However, this slight difference in metric performance is outweighed by the computational efficiency of our approach. As shown in Table IV, ImplicitVol requires approximately one hour to converge. In contrast, our Gaussian Splatting framework achieves comparable fidelity in just 8 to 16 minutesârepresenting an improvement in training speed of nearly an order of magnitude.

2) Cross-View Generalization on Real Data: Real US data introduces challenges compared to synthetic environments, particularly the lack of clear segmentation masks. Consequently, we evaluate reconstruction quality through both visual cross-view generalization and quantitative benchmarking against the adapted ImplicitVol baseline.

Fig. 9 demonstrates the qualitative capabilities of our model.

<!-- image-->  
Fig. 7. Reconstruction results for sagittal and transversal sweeps. The top row shows results from sagittal training, the middle row from transversal training, and the bottom row shows the ground truth.

TABLE IV  
QUANTITATIVE COMPARISON OF OUR METHOD WITH DIFFERENT NUMBER OF GAUSSIANS AGAINST IMPLICITVOL ON SIMULATED DATA.
<table><tr><td></td><td colspan="3">Train</td><td colspan="3">Test</td><td></td></tr><tr><td></td><td>SSIM</td><td>PSNR</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td><td>LPIPS</td><td>Time</td></tr><tr><td colspan="8">Sagittal</td></tr><tr><td>ImplicitVol</td><td>0.994</td><td>31.35</td><td>0.018</td><td>0.994</td><td>34.67</td><td>0.013</td><td>56m</td></tr><tr><td>TVGS (Ours) - 50k</td><td>0.978</td><td>28.36</td><td>0.053</td><td>0.972</td><td>28.51</td><td>0.058</td><td>8m</td></tr><tr><td>TVGS (Ours) - 100k</td><td>0.981</td><td>29.00</td><td>0.047</td><td>0.974</td><td>29.02</td><td>0.051</td><td>16m</td></tr><tr><td colspan="8">Transversal</td></tr><tr><td>ImplicitVol</td><td>0.991</td><td>30.09</td><td>0.026</td><td>0.984</td><td>31.22</td><td>0.047</td><td>62m</td></tr><tr><td>TVGS (Ours) - 50k</td><td>0.983</td><td>30.33</td><td>0.046</td><td>0.970</td><td>28.25</td><td>0.065</td><td>8m</td></tr><tr><td>TVGS (Ours) - 100k</td><td>0.986</td><td>31.47</td><td>0.039</td><td>0.972</td><td>28.55</td><td>0.054</td><td>16m</td></tr></table>

Even when training is restricted to a single view (sagittal or transversal), the model generalizes well when tested on the orthogonal view. It captures anatomical features with high fidelity and handles difficult regions, such as areas obscured by fluid, with surprising accuracy.

To strictly quantify this performance, we compared our method against the ImplicitVol baseline on the real clinical data. As shown in Table V, our approachâs advantages are more pronounced here than in synthetic experiments. Our method outperforms the baseline across all metrics (SSIM, PSNR, LPIPS) for both sagittal and transversal sweeps. Notably, our approach achieves superior perceptual quality (indicated by lower LPIPS) and a drastic reduction in computational cost. While the implicit baseline requires over 3 hours to converge on real data, our method delivers superior results in just 11 to 16 minutes. This order-of-magnitude improvement in efficiency is critical for clinical adoption, where rapid feedback is essential.

Despite the strong individual view reconstruction, combining multiple views remains a challenge. We observed alignment discrepancies between sagittal and transversal reconstructions, which complicates training on both views simultaneously. Addressing this misalignment is a key direction for future work; proper integration of both views could further refine slice position estimation, potentially moving beyond the simplified assumption of evenly spaced angles. In summary, our method demonstrates strong potential for reconstructing consistent and anatomically meaningful 3D representations from US sweeps, offering a scalable and fast alternative to existing implicit representations

<!-- image-->  
Fig. 8. 3D reconstructions using different training settings. Left: sagittal only. Middle: transversal only. Right: both sagittal and transversal.

TABLE V  
QUANTITATIVE COMPARISON OF OUR METHOD WITH DIFFERENT NUMBER OF GAUSSIANS AGAINST IMPLICITVOL ON REAL PATIENT DATA.
<table><tr><td></td><td></td><td>Train</td><td></td><td></td><td>Test</td><td></td><td></td><td></td></tr><tr><td></td><td>SSIM PSNR LPIPS</td><td></td><td></td><td>SSIM PSNR</td><td></td><td>LPIPS</td><td>Time</td><td>Memory</td></tr><tr><td>Sagittal</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ImplicitVol</td><td>0.858</td><td>28.10</td><td>0.240</td><td>0.783</td><td>26.43</td><td>0.300</td><td>3.14h</td><td>10GB</td></tr><tr><td>TVGS (Ours) - 50k</td><td>0.901</td><td>31.38</td><td>0.188</td><td>0.845</td><td>28.51</td><td>0.223</td><td>11m</td><td>129MB</td></tr><tr><td>TVGS (Ours) - 100k</td><td>0.925</td><td>32.24</td><td>0.143</td><td>0.864</td><td>28.85</td><td>0.185</td><td>16m</td><td>160MB</td></tr><tr><td>Transversal</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ImplicitVol</td><td>0.855</td><td>27.38</td><td>0.247</td><td>0.790</td><td>26.21</td><td>0.295</td><td>3.07h</td><td>10GB</td></tr><tr><td>TVGS (Ours) - 50k</td><td>0.897</td><td>30.81</td><td>0.194</td><td>0.847</td><td>28.44</td><td>0.223</td><td>11m</td><td>129MB</td></tr><tr><td>TVGS (Ours) - 100k 0.923</td><td></td><td>31.21</td><td>0.145</td><td>0.823</td><td>27.80</td><td>0.222</td><td>16m</td><td>160MB</td></tr></table>

<!-- image-->  
Fig. 9. Reconstructions from real data. Top: trained on sagittal view. Middle: trained on transversal view. Bottom: ground truth for comparison.

3) Effect of Slice Density: We also analyze the impact of slice density and Gaussian count. The number of slices acquired during a sweep significantly affects reconstruction quality. We systematically varied slice densityâfrom 10 to 200 slicesâto assess how it influences our modelâs performance. As shown in Fig. 10, reconstructions with very few slices (10â25) are blurry and lack anatomical detail, while quality improves substantially with 50 or more slices. Beyond 100 slices, gains diminish, suggesting that ultra-dense sampling offers limited additional benefit. Quantitative trends in Fig. 11 confirm this: higher slice counts result in lower L1 loss and higher SSIM, and they also accelerate convergence. Conversely, sparse setups plateau at lower fidelity and slower training. Our Gaussian primitives naturally compensate for moderate sparsity by expanding spatial support across neighboring slices. However, this ability is limited when data is extremely sparse, leading to smooth, under-detailed outputs. In contrast, denser setups allow Gaussians to reinforce structural coherence through natural overlap across slices.

<!-- image-->

<!-- image-->  
Fig. 10. Reconstruction results for different amounts of slices.

<!-- image-->

<!-- image-->  
Fig. 11. SSIM and LPIPS across epochs for different numbers of slices, same number of per-slice Gaussians.

## D. Rendering Speed

Finally, we evaluate the rendering performance by measuring the average frame rate under varying scene complexities. Specifically, we benchmark scenes containing different numbers of Gaussians to assess how the rendering speed scales with model size. The results, summarized in Table VI, provide a quantitative comparison across these settings.

TABLE VI  
RENDERING SPEED DIFFERENT NUMBER OF GAUSSIAN PRIMITIVES, FOR 10,000 ITERATIONS.
<table><tr><td># Gaussians</td><td>Sweep type</td><td>Mean speed [ms]</td><td>FPS</td></tr><tr><td rowspan="2">â¼ 10,000</td><td>Transversal</td><td> $4 . 3 5 8 \pm 0 . 0 4 0$ </td><td>229.47</td></tr><tr><td>Sagittal</td><td> $4 . 3 9 7 \pm 0 . 0 3 1$ </td><td>227.40</td></tr><tr><td rowspan="2">â¼ 50,000</td><td>Transversal</td><td> $1 8 . 1 4 2 \pm 1 7 . 4 8 1$ </td><td>55.12</td></tr><tr><td>Sagittal</td><td> $1 8 . 1 8 2 \pm 1 7 . 6 1 3$ </td><td>55.00</td></tr></table>

## VI. DISCUSSION

Our experiments highlight the sensitivity of the method to initialization. Gaussians placed near actual data slices converged faster and contributed more effectively to the final reconstruction, while randomly initialized primitives were often slow to optimize or entirely deactivated. Assigning higher learning rates to the mean parameters, compared to opacity or scale, helped reduce early deactivation and encouraged convergence. Slice density was another important factor influencing performance. Denser sweeps led to higherquality reconstructions, with fewer gaps and artifacts. Although adding more Gaussians can partially compensate for sparse data, this increases computational cost. This trade-off between reconstruction fidelity, data acquisition effort, and runtime is important to consider for clinical deployment. The method also demonstrated promising generalization across orthogonal sweep directions. Volumes trained on sagittal sweeps could generalize well to transversal inputs and vice versa. However, combining multiple views simultaneously remains a challenge, particularly due to the need for accurate alignment. Our method does not currently perform registration, which can lead to visible inconsistencies when merging sweeps.

Practical Limitations. Our method relies on initialization from temporally ordered slices with uniform angular spacing. While our experiments on jitter (Table III) demonstrate that joint pose optimization can effectively compensate for moderate motion irregularities and hand tremor, the method remains sensitive to rapid, large-scale changes in probe orientation that violate the initialization prior. Furthermore, our rasterization framework utilizes a simplified additive model that neglects acoustic attenuation and shadowing. While this tradeoff enables faster performance required for operator feedback, it limits the physical realism of the rendering compared to computationally intensive wave-propagation models. Even with CUDA acceleration, the method remains computationally intensive. Near-real-time performance may be achievable with further optimization, but current runtimes are high, especially for large numbers of Gaussians or high-resolution data. Finally, while we demonstrated feasibility using both synthetic and real datasets, broader clinical validation is needed. Differences in patient anatomy, imaging conditions, and probe operators were not extensively studied, and generalizing to these variables will be important before the method can be adopted in practice.

Clinical Potential. Despite these limitations, the method shows potential for improving clinical US workflows. During TVS examination the ability to generate accurate 3D reconstructions from freehand sweeps could support better visualization and localization of subtle anatomical abnormalities and enhance surgical planning. Additionally, real-time feedback during scanning could guide clinicians in covering relevant regions more thoroughly, potentially reducing missed findings making scanning less operator dependent. The synthetic dataset and evaluation pipeline developed as part of this work may also serve as a foundation for future studies, offering reproducible benchmarks for US reconstruction research.

## VII. CONCLUSION

We have introduced a novel, unsupervised framework for reconstructing 3D anatomical volumes from freehand 2D transvaginal US sweeps, without requiring external tracking, ground truth trajectories, or pose supervision. By adapting Gaussian Splatting to the specific physics and geometry of US imaging, and designing a custom differentiable rasterizer, our method enables accurate, fast and memoryefficient volumetric reconstruction from sparse, uncalibrated slice data. Through extensive experiments on both synthetic and real-world datasets, we demonstrated that the approach can achieve high-fidelity reconstructions, generalizes well across sweep orientations, and remains robust to various acquisition conditions. We showed that initialization, slice density, adaptive pose refinement, and the number of Gaussians are critical factors influencing performance, and proposed effective strategies such as density control for optimizing these parameters. By eliminating the reliance on fixed geometric priors, the proposed learnable pose formulation proves that high-fidelity 3D US reconstruction can be achieved as a purely computational, self-correcting task. While challenges remainâparticularly regarding multi-view alignment and realtime performanceâresults suggest strong potential for clinical deployment. Our method offers a scalable, hardwarefree alternative to conventional 3D US systems and provides a foundation for future research in data-driven, volumetric reconstruction and AI-assisted gynecological imaging.

## REFERENCES

[1] J. Alcazar, âThree-dimensional ultrasound in gynecology: Current status and future perspectives,â Current Womenâs Health Reviews, 2005.

[2] J. A. Jensen, âMedical ultrasound imaging,â Progress in Biophysics and Molecular Biology, 2007.

[3] C. Apirakviriya, T. Rungruxsirivorn, V. Phupong, and W. Wisawasukmongchol, âDiagnostic accuracy of 3Dtransvaginal ultrasound in detecting uterine cavity abnormalities in infertile patients as compared with hysteroscopy,â Eur J Obstet Gynecol Reprod Biol, 2016.

[4] P.-H. Yeung, L. Hesse, M. Aliasi, M. Haak, W. Xie, A. I. Namburete, et al., âSensorless volumetric reconstruction of fetal brain freehand ultrasound scans with deep implicit representation,â Medical Image Analysis, 94, 103147, 2024.

[5] C. Peng, Q. Cai, M. Chen, and X. Jiang, âRecent advances in tracking devices for biomedical ultrasound imaging applications,â Micromachines, 2022.

[6] R. Prevost et al., â3D freehand ultrasound without external tracking using deep learning,â Med Image Anal, 2018.

[7] M. C. Eid, P.-H. Yeung, M. K. Wyburd, J. F. Henriques, and A. I. Namburete, âRapidvol: Rapid reconstruction of 3d ultrasound volumes from sensorless 2d scans,â in 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), 2025, 1â5. DOI: 10 . 1109 / ISBI60581.2025.10980994

[8] M. C. Eid, A. I. L. Namburete, and J. F. Henriques, Ultragauss: Ultrafast gaussian reconstruction of 3d ultrasound volumes, 2025. arXiv: 2505.05643 [eess.IV].

[9] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, Â¨ â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics (SIGGRAPH Conference Proceedings), 42, 4, Jul. 2023.

[10] T. Wen et al., âAn accurate and effective fmm-based approach for freehand 3D ultrasound reconstruction,â Biomed Signal Process Control, 2013.

[11] J.-F. Chen, J. B. Fowlkes, P. L. Carson, and J. M. Rubin, âDetermination of scan-plane motion using speckle decorrelation: Theoretical considerations and initial test,â Int J Imaging Syst Technol, 1997.

[12] A. H. Gee, R. James Housden, P. Hassenpflug, G. M. Treece, and R. W. Prager, âSensorless freehand 3D ultrasound in real tissue: Speckle decorrelation without fully developed speckle,â Med Image Anal, 2006.

[13] P.-C. Li, C.-Y. Li, and W.-C. Yeh, âTissue motion and elevational speckle decorrelation in freehand 3D ultrasound,â Ultrasonic Imaging, 2002.

[14] A. D. Gilliam, J. A. Hossack, and S. T. Acton, âFreehand 3d ultrasound volume reconstruction via sub-pixel phase correlation,â in 2006 International Conference on Image Processing, 2006, 2537â2540. DOI: 10.1109/ ICIP.2006.312958

[15] L. Tetrel, H. Chebrek, and C. Laporte, âLearning for graph-based sensorless freehand 3d ultrasound,â in Machine Learning in Medical Imaging, L. Wang, E. Adeli, Q. Wang, Y. Shi, and H.-I. Suk, Eds., Springer International Publishing, 2016, 205â212.

[16] H. Guo, H. Chao, S. Xu, B. J. Wood, J. Wang, and P. Yan, âUltrasound volume reconstruction from freehand scans without tracking,â IEEE Trans on Biomed Eng, 2023.

[17] Y. Dou, F. Mu, Y. Li, and T. Varghese, âSensorless end-to-end freehand 3-D ultrasound reconstruction with physics-guided deep learning,â IEEE Trans Ultrason Ferroelectr Freq Control, 2024.

[18] M. Luo et al., âRecON: Online learning for sensorless freehand 3D ultrasound reconstruction,â Med Image Anal, 2023.

[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â in ECCV, 2020, 405â421.

[20] J. Ramesh, N. Dinsdale, P.-H. Yeung, and A. I. L. Namburete, â Geometric Transformation Uncertainty for Improving 3D Fetal Brain Pose Prediction from Freehand 2D Ultrasound Videos,â in proceedings of Medical Image Computing and Computer Assisted Intervention â MICCAI 2024, vol. LNCS 15001, Springer Nature Switzerland, Oct. 2024, 419â429.

[21] A. Fedorov et al., â3D slicer as an image computing platform for the quantitative imaging network,â Magnetic Resonance Imaging, 2012.

[22] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, âBarf: Bundle-adjusting neural radiance fields,â in IEEE International Conference on Computer Vision (ICCV), 2021, 5741â5751.

## APPENDIX

## A. Gradient Derivations

This appendix provides the detailed derivations required for manually implementing the backward pass of the differentiable rasterizer. Since the forward pass is implemented using custom CUDA kernels, we cannot rely on autograd engines and must explicitly compute partial derivatives for each Gaussian parameter.

1) Notation Recap: For each pixel $p$ at coordinate $\mathbf { c } _ { p } \in \mathbb { R } ^ { 3 }$ and each Gaussian $^ { g , }$ we define:

$$
\begin{array} { l } { { \displaystyle { \bf d } _ { g p } = { \bf c } _ { p } - \mu _ { g } } , } \\ { { \displaystyle e _ { g p } = - \frac { 1 } { 2 } { \bf d } _ { g p } ^ { \top } \Sigma _ { g } ^ { - 1 } { \bf d } _ { g p } } , } \\ { { \displaystyle \alpha _ { g p } = o _ { g } \exp ( e _ { g p } ) } , } \\ { { \displaystyle I _ { p } = \sum _ { g } \alpha _ { g p } I _ { g } } } \end{array}
$$

The loss function L depends on rendered intensities $I _ { p } ,$ , and we are given $\frac { \partial L } { \partial I _ { p } }$ from upstream layers.

2) Gradient w.r.t. Gaussian Intensity $\mathrm { l } _ { \mathrm { g } } \dot { . }$ Since $\begin{array} { r l } { I _ { p } } & { { } = } \end{array}$ $\textstyle \sum _ { g } \alpha _ { g p } I _ { g }$ , differentiating w.r.t. $I _ { g }$ gives:

$$
\begin{array} { c } { { \displaystyle { \frac { \partial I _ { p } } { \partial I _ { g } } = \alpha _ { g p } } } } \\ { { \displaystyle { \Rightarrow \frac { \partial L } { \partial I _ { g } } = \sum _ { p } \frac { \partial L } { \partial I _ { p } } \frac { \partial I _ { p } } { \partial I _ { g } } = \sum _ { p } \frac { \partial L } { \partial I _ { p } } \alpha _ { g p } } } } \end{array}
$$

3) Gradient w.r.t. Opacity $\mathsf { o } _ { \mathsf { g } } .$ From $\alpha _ { g p } = o _ { g } \exp ( e _ { g p } )$ , we get:

$$
\begin{array} { l } { { \displaystyle { \frac { \partial \alpha _ { g p } } { \partial o _ { g } } = \exp ( e _ { g p } ) } } } \\ { { \displaystyle { \Rightarrow \frac { \partial L } { \partial o _ { g } } = \sum _ { p } \frac { \partial L } { \partial I _ { p } } I _ { g } \frac { \partial \alpha _ { g p } } { \partial o _ { g } } = \sum _ { p } I _ { g } \frac { \partial L } { \partial I _ { p } } \exp ( e _ { g p } ) } } } \end{array}
$$

4) Gradient w.r.t. Mean $\mu _ { \mathrm { g } } .$ We use:

$$
\frac { \partial e _ { g p } } { \partial { \bf d } _ { g p } } = - \Sigma _ { g } ^ { - 1 } { \bf d } _ { g p } , \quad \frac { \partial \alpha _ { g p } } { \partial { \bf d } _ { g p } } = o _ { g } \exp ( e _ { g p } ) \left( - \Sigma _ { g } ^ { - 1 } { \bf d } _ { g p } \right)
$$

Then,

$$
\frac { \partial L } { \partial \mathbf { d } _ { g p } } = \frac { \partial L } { \partial I _ { p } } I _ { g } \frac { \partial \alpha _ { g p } } { \partial \mathbf { d } _ { g p } } = - I _ { g } \frac { \partial L } { \partial I _ { p } } o _ { g } \exp ( e _ { g p } ) \Sigma _ { g } ^ { - 1 } \mathbf { d } _ { g p }
$$

Using $\mathbf { d } _ { g p } = \mathbf { c } _ { p } - \mu _ { g } ,$ , we get:

$$
\frac { \partial L } { \partial \mu _ { g } } = - \sum _ { p } \frac { \partial L } { \partial \mathbf { d } _ { g p } } , \quad \frac { \partial L } { \partial \mathbf { c } _ { p } } = \sum _ { g } \frac { \partial L } { \partial \mathbf { d } _ { g p } }
$$

5) Gradient w.r.t. Covariance $\Sigma _ { \mathrm { g } } .$ From:

$$
e _ { g p } = - \frac 1 2 { \bf d } _ { g p } ^ { \top } \Sigma _ { g } ^ { - 1 } { \bf d } _ { g p } ,
$$

we use the identity:

$$
\frac { \partial e _ { g p } } { \partial \Sigma _ { g } } = \frac { 1 } { 2 } \Sigma _ { g } ^ { - 1 } { \bf d } _ { g p } { \bf d } _ { g p } ^ { \top } \Sigma _ { g } ^ { - 1 }
$$

Combining with the chain rule:

$$
\begin{array} { c } { { \displaystyle { \frac { \partial { \cal L } } { \partial e _ { g p } } } = I _ { g } \frac { \partial { \cal L } } { \partial I _ { p } } o _ { g } \exp ( e _ { g p } ) } } \\ { { \displaystyle \Rightarrow \frac { \partial { \cal L } } { \partial \Sigma _ { g } } = \sum _ { p } \frac { \partial { \cal L } } { \partial e _ { g p } } \frac { \partial e _ { g p } } { \partial \Sigma _ { g } } } } \\ { { \displaystyle ~ = \frac { 1 } { 2 } \sum _ { p } I _ { g } \frac { \partial { \cal L } } { \partial I _ { p } } o _ { g } \exp ( e _ { g p } ) \Sigma _ { g } ^ { - 1 } { \bf d } _ { g p } { \bf d } _ { g p } ^ { \top } \Sigma _ { g } ^ { - 1 } } } \end{array}
$$

For covariance vector representation, we store only six unique elements:

$$
\mathrm { c o v 3 D } _ { g } = [ \sigma _ { 0 } , \sigma _ { 1 } , \sigma _ { 2 } , \sigma _ { 3 } , \sigma _ { 4 } , \sigma _ { 5 } ] \Rightarrow \Sigma _ { g } = \left[ { \sigma _ { 0 } } \quad \sigma _ { 1 } \quad \sigma _ { 2 } \right]
$$

To compute $\frac { \partial L } { \partial \sigma _ { i } }$ , we symmetrize the gradient:

$$
\frac { \partial L } { \partial \sigma _ { i } } = \left\{ \begin{array} { l l } { ( \frac { \partial L } { \partial \Sigma _ { g } } ) _ { i i } } & { \mathrm { f o r ~ d i a g o n a l ~ } \sigma _ { i } } \\ { \frac { 1 } { 2 } \left[ ( \frac { \partial L } { \partial \Sigma _ { g } } ) _ { c d } + ( \frac { \partial L } { \partial \Sigma _ { g } } ) _ { d c } \right] } & { \mathrm { f o r ~ o f f - d i a g o n a l ~ } ( c , d ) } \end{array} \right.
$$

6) Propagation to Scale and Rotation: We use:

$$
\begin{array} { c } { { \Sigma _ { g } = M _ { g } ^ { \top } M _ { g } , \quad M _ { g } = S _ { g } R _ { g } } } \\ { { \Rightarrow { \displaystyle \frac { \partial L } { \partial M _ { g } } } = 2 M _ { g } { \displaystyle \frac { \partial L } { \partial \Sigma _ { g } } } , \quad { \displaystyle \frac { \partial L } { \partial S _ { g } } } = { \displaystyle \frac { \partial L } { \partial M _ { g } } } R _ { g } ^ { \top } , \quad { \displaystyle \frac { \partial L } { \partial R _ { g } } } = S _ { g } ^ { \top } { \displaystyle \frac { \partial L } { \partial M _ { g } } } } } \end{array}
$$

7) Gradient w.r.t. Quaternion ${ \mathsf { q } } _ { \mathrm { \Delta } }$ : We represent rotation using quaternions $q _ { g } = [ q _ { r } , q _ { i } , q _ { j } , q _ { k } ]$ . The corresponding rotation matrix $R ( q _ { g } )$ is defined analytically. To differentiate, we compute:

$$
\frac { \partial L } { \partial q _ { m } } = \sum _ { u , v } \frac { \partial L } { \partial [ R _ { g } ] _ { u v } } \frac { \partial [ R ( q _ { g } ) ] _ { u v } } { \partial q _ { m } } , \quad m \in \{ r , i , j , k \}
$$

To backpropagate through quaternion normalization $q _ { g } =$ $\tilde { q } _ { g } / \lVert \tilde { q } _ { g } \rVert$ , we use:

$$
\frac { \partial L } { \partial \tilde { q } _ { g } } = \frac { 1 } { \Vert \tilde { q } _ { g } \Vert } \left( I - q _ { g } q _ { g } ^ { \top } \right) \frac { \partial L } { \partial q _ { g } }
$$

## B. Initialization Schemes

We define the two initialization strategies used for setting Gaussian means $\mu _ { g } \mathrm { . }$

1) On-slice initialization.: Assuming a known slice pose with rotation $R _ { S } \in \mathbb { R } ^ { 3 \times 3 }$ and translation $\mathbf { t } _ { S } \in \mathbb { R } ^ { 3 }$ , we define a 2D reference grid:

$$
\mathcal { P } = \{ ( x _ { i } , y _ { j } , 0 ) \in \mathbb { R } ^ { 3 } \mid x _ { i } \in [ a _ { x } , b _ { x } ] , \ y _ { j } \in [ a _ { y } , b _ { y } ] \}
$$

with uniform grid spacing. The 3D Gaussian centers are then computed via:

$$
\mu _ { i j } = R _ { S } ( p _ { i j } + \mathbf { t } _ { S } ) , \quad p _ { i j } \in \mathcal { P }
$$

2) Uniform sampling.: If no pose is known, we sample:

$$
\mu _ { g } = \mathbf { a } + ( \mathbf { b } - \mathbf { a } ) \odot \xi _ { g } , \quad \xi _ { g } \sim \mathcal { U } [ 0 , 1 ] ^ { 3 }
$$

where $\mathbf { a } , \mathbf { b } \in \mathbb { R } ^ { 3 }$ define the bounding box.

A visual comparison of these two initialization strategies is shown in Fig. 12. On-slice initialization produces structured, well-aligned Gaussians from the start, whereas uniform sampling yields a diffuse cloud of Gaussians that must be refined during training.

<!-- image-->  
Fig. 12. Left: On-Slice initialization for a sagittal sweep. Right: Random initialization for a sagittal sweep.

## C. Inactive Gaussians and Density Control

Our density control mechanism involves periodic pruning and reinitialization of Gaussians based on their contribution metric $m _ { g } ~ = ~ | I _ { g } \cdot o _ { g } |$ . Below are the key implementation details:

1) Pruning Schedule.: Every 10 epochs, we evaluate all Gaussians and mark those with:

$$
m _ { g } = \vert I _ { g } \cdot o _ { g } \vert < \epsilon , \quad \mathrm { w i t h } \ \epsilon = 1 0 ^ { - 4 }
$$

as inactive. These are removed from the parameter set.

2) Reinitialization.: For each removed Gaussian, a new one is inserted. The new mean $\mu _ { g }$ is sampled uniformly within the current bounding box of active Gaussian means:

$$
\mu _ { g } \sim \mathcal { U } [ \mu _ { \mathrm { m i n } } , \mu _ { \mathrm { m a x } } ]
$$

All other parameters are initialized using the defaults described in Section IV-C.

3) Optional Filters.: To prevent rapid re-deactivation, newly inserted Gaussians can be given a brief âgrace periodâ (e.g., skipped in the next one or two pruning steps). This lightweight mechanism stabilizes training and improves efficiency, particularly in early epochs where many Gaussians otherwise become stuck.

Tables VII and VIII show the search results for the alternative method proposed for inactive Gaussians.

TABLE VII  
L1 AND SSIM METRICS ON DIFFERENT LEARNING RATES FOR MEANS
<table><tr><td></td><td colspan="2">0.15</td><td colspan="2">0.2</td><td colspan="2">0.25</td></tr><tr><td>Metric</td><td>L1</td><td>SSIM</td><td>L1</td><td>SSIM</td><td>L1</td><td>SSIM</td></tr><tr><td>Means</td><td>0.0201</td><td>0.8642</td><td>0.0189</td><td>0.8692</td><td>0.201</td><td>0.8700</td></tr></table>

TABLE VIII

L1 AND SSIM METRICS ON DIFFERENT LEARNING RATES
<table><tr><td></td><td colspan="2">0.005</td><td colspan="2">0.01</td><td colspan="2">0.05</td></tr><tr><td>Metric</td><td>L1</td><td>SSIM</td><td>L1</td><td>SSIM</td><td>L1</td><td>SSIM</td></tr><tr><td>Opacities</td><td>0.0228</td><td>0.8451</td><td>0.0213</td><td>0.8485</td><td>0.0221</td><td>0.8518</td></tr><tr><td>Scales</td><td>0.0171</td><td>0.8671</td><td>0.0169</td><td>0.8680</td><td>0.0221</td><td>0.8518</td></tr><tr><td>Intensities</td><td>0.0194</td><td>0.8540</td><td>0.0195</td><td>0.8546</td><td>0.0221</td><td>0.8518</td></tr></table>

<!-- image-->  
Fig. 13. Distribution of Gaussian primitives for a scene.

## D. Spatial convergence

Fig. 13 visualizes the spatial distribution of Gaussians after optimization. For clarity, all opacities and intensities are set to 1.0, and Gaussian scales are reduced by a factor of 0.2. The visualizations show that Gaussians concentrate in high-intensity areas corresponding to meaningful anatomical structures, while low-intensity regions remain largely emptyâreflecting the modelâs ability to focus its representational capacity where it matters most.

Fig. 14 illustrates how Gaussian centers evolve during training, comparing a sparse configuration (â 24 Gaussians per slice) to a denser one (â 112 per slice). With fewer Gaussians, individual movements are larger and more distinct, as each primitive covers more space. In denser settings, movement is subtler due to increased redundancy.

Notably, Gaussians exhibit dynamic reallocation: some exit the visualized slice while others enter or converge toward high-intensity regions. This behavior suggests an adaptive mechanism that reallocates primitives toward regions of higher relevance or deactivates them when unnecessary, enabling efficient use of limited representational resources.

## E. Implementation Optimizations

Efficient training of the differentiable rasterizer requires high-performance implementation of both the forward and backward passes due to the dense, per-pixel evaluation of contributions from thousands of 3D Gaussians. This section describes the key CUDA-level optimizations applied to ensure computational feasibility and scalability.

1) Precomputation of Reusable Quantities: The backward pass relies heavily on values such as $\Sigma _ { g } ^ { - 1 }$ , which are reused across all pixels influenced by a given Gaussian. To avoid redundant computation, we launch a dedicated kernel that precomputes $\Sigma _ { g } ^ { - 1 }$ for all Gaussians and stores the uppertriangular components in a compact 6-element representation. These cached inverses are then accessed during both forward and backward passes to evaluate $e _ { g p } , \alpha _ { g p }$ , and their gradients.

<!-- image-->  
Fig. 14. Trajectory of Gaussian centers during training for the same slice, comparing different quantities of Gaussians. Grey traces indicate movement paths. Some Gaussians appear only at the start or end of the trajectory within the slice, suggesting they originated from or moved to adjacent slices.

Similarly, we precompute $\Sigma _ { g } ^ { - 1 } \mathbf { d } _ { g p }$ wherever possible to avoid recomputation during the accumulation of gradient terms. This significantly reduces arithmetic overhead in the main rasterization kernels.

2) Shared Memory Tiling: A major bottleneck in GPU rasterization is the repeated access to Gaussian parameters from global memory. To mitigate this, we divide the set of Gaussians into tiles of size T (e.g., 64). Each CUDA thread block loads one tile of Gaussian data (means, scales, quaternions, opacities, intensities) into shared memory. Threads within the block synchronize and reuse the shared memory tile for computing contributions and gradients for their assigned pixels. This results in reduction in global memory bandwidth usage, better memory coalescing and cache locality and increased parallel efficiency and occupancy.

3) Symmetric Covariance Storage and Update: Each 3D Gaussian covariance matrix $\Sigma _ { g }$ is symmetric. During the backward pass, we compute $\frac { \partial L } { \partial \Sigma _ { g } }$ as a full matrix. Each symmetric entry (e.g., $\sigma _ { 1 } .$ , appearing in both (0, 1) and (1, 0)) receives contributions from both locations and is updated using their average. This saves memory, reduces write operations, and simplifies gradient accumulation logic.

4) Matrix-Free Operations: To minimize expensive matrix multiplications and inversions, we unroll expressions like $\mathbf { d } _ { g p } ^ { \top } \bar { \Sigma } _ { g } ^ { - 1 } \mathbf { d } _ { g p }$ into explicit component-wise dot products using precomputed inverse covariances. Gradients involving matrixâvector products are similarly computed via direct arithmetic to avoid launching linear algebra kernels. This low-level optimization is especially effective for backward steps where every saved FLOP improves scalability.

<!-- image-->  
Fig. 15. Effect of cumulative optimization strategies on training time.

TABLE IX  
QUANTITATIVE RESULTS IN THE CASE OF SIMULATED DATA.
<table><tr><td>Pat. ID</td><td>SSIM</td><td>Training PSNR</td><td>LPIPS</td><td></td><td>Test PSNR</td><td>LPIPS</td><td>Time</td></tr><tr><td>Sag.</td><td></td><td></td><td></td><td>SSIM</td><td></td><td></td><td></td></tr><tr><td>ImplicitVol</td><td>0.9944</td><td>31.35</td><td>0.0182</td><td>0.9936</td><td>34.67</td><td>0.0132</td><td>56m</td></tr><tr><td>Ours - 50k</td><td>0.9760</td><td>28.61</td><td>0.0556</td><td>0.9301</td><td>28.35</td><td>0.1284</td><td>8m</td></tr><tr><td>Our - 100k</td><td>0.9792</td><td>29.41</td><td>0.0502</td><td>0.9218</td><td>28.53</td><td>0.1350</td><td>16m</td></tr><tr><td>Trans.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ImplicitVol</td><td>0.9910</td><td>30.09</td><td>0.0255</td><td>0.9835</td><td>31.22</td><td>0.0468</td><td>62m</td></tr><tr><td>Ours - 50k</td><td>0.9834</td><td>30.36</td><td>0.0461</td><td>0.8770</td><td>27.98</td><td>0.2185</td><td>8m</td></tr><tr><td>Ours - 100k</td><td>0.9865</td><td>31.75</td><td>0.0394</td><td>0.8378</td><td>27.77</td><td>0.2319</td><td>16m</td></tr></table>

5) Compiler-Level Optimizations: To further improve performance, we apply the following compiler flags during kernel compilation

â¢ -O3: Enables aggressive optimizations like loop unrolling, constant folding, and function inlining.

--use fast math: Activates fast approximations for transcendental functions (e.g., exp, sqrt), which are sufficient for training purposes and significantly faster.

In practice, these flags reduce kernel runtime by 20â40% without compromising training stability.

6) Quantitative Impact: Fig. 15 shows the training time required to fit a fixed US volume as each optimization is introduced cumulatively. The baseline PyTorch implementation (no tiling, no precomputation, no matrix unrolling) takes approximately 18 hours. Our optimized CUDA implementation reduces this to under 2 hours for a scene with over 50,000 Gaussians.

## F. Extended Results

This section presents a detailed quantitative breakdown of the reconstruction performance across individual subjects. While the main text reports aggregated statistics, the tables below illustrate the consistency of our method across different patient scans. We compare the baseline ImplicitVol approach against our differentiable rasterization method configured with 50,000 and 100,000 primitives. As shown in the per-patient breakdowns, our approach achieves competitive similarity metrics (SSIM, PSNR) and perceptual quality (LPIPS) while consistently reducing the computational burden from approximately 3 hours to under 20 minutes per volume.

QUANTITATIVE RESULTS FOR DIFFERENT PATIENTS IN THE CASE OF IMPLICITVOL ON REAL DATA.  
TABLE X
<table><tr><td>Pat. ID</td><td>SSIM</td><td>Training PSNR</td><td>LPIPS</td><td>SSIM</td><td>Test PSNR</td><td>LPIPS</td><td>Time</td></tr><tr><td>Sag.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>152</td><td>0.8459</td><td>27.11</td><td>0.2599</td><td>0.7730</td><td>25.62</td><td>0.3145</td><td>3.15h</td></tr><tr><td>154</td><td>0.8364</td><td>27.44</td><td>0.2634</td><td>0.7709</td><td>26.28</td><td>0.3081</td><td>3.34h</td></tr><tr><td>159</td><td>0.8480</td><td>27.06</td><td>0.2506</td><td>0.7731</td><td>25.61</td><td>0.3122</td><td>3.12h</td></tr><tr><td>231</td><td>0.8397</td><td>27.53</td><td>0.2533</td><td>0.7570</td><td>25.95</td><td>0.3135</td><td>3.03h</td></tr><tr><td>232</td><td>0.8970</td><td>28.90</td><td>0.2151</td><td>0.8229</td><td>27.18</td><td>0.2831</td><td>3.13h</td></tr><tr><td>238</td><td>0.8922</td><td>29.87</td><td>0.2029</td><td>0.8204</td><td>27.79</td><td>0.2720</td><td>3.02h</td></tr><tr><td>274</td><td>0.8600</td><td>28.66</td><td>0.2262</td><td>0.7806</td><td>26.53</td><td>0.2963</td><td>3.17h</td></tr><tr><td>306</td><td>0.8317</td><td>27.45</td><td>0.2566</td><td>0.7573</td><td>25.92</td><td>0.3028</td><td>3.13h</td></tr><tr><td>324</td><td>0.8924</td><td>29.15</td><td>0.2127</td><td>0.8126</td><td>26.87</td><td>0.2835</td><td>3.13h</td></tr><tr><td>326</td><td>0.8357</td><td>27.80</td><td>0.2611</td><td>0.7670</td><td>26.54</td><td>0.3156</td><td>3.16h</td></tr><tr><td>Trans.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>152</td><td>0.8537</td><td>26.81</td><td>0.2591</td><td>0.7868</td><td>25.73</td><td>0.3076</td><td>3.07h</td></tr><tr><td>154</td><td>0.8154</td><td>26.07</td><td>0.2810</td><td>0.7588</td><td>25.33</td><td>0.3144</td><td>3.13h</td></tr><tr><td>159</td><td>0.8595</td><td>27.57</td><td>0.2491</td><td>0.8080</td><td>26.33</td><td>0.2908</td><td>3.08h</td></tr><tr><td>231</td><td>0.8307</td><td>26.19</td><td>0.2660</td><td>0.7582</td><td>25.23</td><td>0.3157</td><td>3.09h</td></tr><tr><td>232</td><td>0.9144</td><td>28.90</td><td>0.2059</td><td>0.8521</td><td>27.12</td><td>0.2778</td><td>3.05h</td></tr><tr><td>238</td><td>0.8762</td><td>28.68</td><td>0.2218</td><td>0.8235</td><td>27.55</td><td>0.2629</td><td>3.06h</td></tr><tr><td>274</td><td>0.8556</td><td>27.96</td><td>0.2340</td><td>0.7898</td><td>26.87</td><td>0.2793</td><td>3.07h</td></tr><tr><td>306</td><td>0.8237</td><td>26.21</td><td>0.2644</td><td>0.7491</td><td>25.40</td><td>0.3056</td><td>3.09h</td></tr><tr><td>324</td><td>0.8879</td><td>29.18</td><td>0.2226</td><td>0.8128</td><td>27.25</td><td>0.2811</td><td>2.97h</td></tr><tr><td>326</td><td>0.8285</td><td>26.18</td><td>0.2652</td><td>0.7647</td><td>25.27</td><td>0.3120</td><td>3.11h</td></tr></table>

TABLE XI

QUANTITATIVE RESULTS FOR DIFFERENT PATIENTS IN THE CASE OF OURS-50K ON REAL DATA. TRAINED ON 150 EPOCHS
<table><tr><td>Pat. ID</td><td>SSIM</td><td>Training PSNR</td><td>LPIPS</td><td>SSIM</td><td>Test PSNR</td><td>LPIPS</td><td>Time</td></tr><tr><td>Sag.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>152</td><td>0.8903</td><td>30.40</td><td>0.2099</td><td>0.8303</td><td>27.81</td><td>0.2543</td><td>11m</td></tr><tr><td>154</td><td>0.8866</td><td>30.77</td><td>0.2088</td><td>0.8412</td><td>28.74</td><td>0.2299</td><td>11m</td></tr><tr><td>159</td><td>0.8899</td><td>30.61</td><td>0.1970</td><td>0.8378</td><td>28.28</td><td>0.2309</td><td>11m</td></tr><tr><td>231</td><td>0.8901</td><td>30.97</td><td>0.1991</td><td>0.8279</td><td>28.39</td><td>0.2428</td><td>11m</td></tr><tr><td>232</td><td>0.9248</td><td>32.24</td><td>0.1723</td><td>0.8657</td><td>28.82</td><td>0.2106</td><td>11m</td></tr><tr><td>238</td><td>0.9200</td><td>33.03</td><td>0.1603</td><td>0.8658</td><td>29.08</td><td>0.1964</td><td>11m</td></tr><tr><td>274</td><td>0.9040</td><td>31.83</td><td>0.1735</td><td>0.8484</td><td>28.61</td><td>0.2103</td><td>11m</td></tr><tr><td>306</td><td>0.8875</td><td>30.59</td><td>0.1910</td><td>0.8298</td><td>27.99</td><td>0.2207</td><td>11m</td></tr><tr><td>324</td><td>0.9260</td><td>32.68</td><td>0.1698</td><td>0.8616</td><td>28.54</td><td>0.2107</td><td>11m</td></tr><tr><td>326</td><td>0.8913</td><td>30.72</td><td>0.1934</td><td>0.8442</td><td>28.79</td><td>0.2209</td><td>11m</td></tr><tr><td>Trans.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>152</td><td>0.8967</td><td>30.16</td><td>0.2032</td><td>0.8431</td><td>27.94</td><td>0.2321</td><td>11m</td></tr><tr><td>154</td><td>0.8696</td><td>29.41</td><td>0.2196</td><td>0.8250</td><td>27.69</td><td>0.2453</td><td>11m</td></tr><tr><td>159</td><td>0.8998</td><td>30.88</td><td>0.1968</td><td>0.8609</td><td>28.83</td><td>0.2224</td><td>11m</td></tr><tr><td>231</td><td>0.8834</td><td>29.67</td><td>0.2039</td><td>0.8237</td><td>27.67</td><td>0.2352</td><td>11m</td></tr><tr><td>232</td><td>0.9329</td><td>32.44</td><td>0.1708</td><td>0.8854</td><td>29.31</td><td>0.2112</td><td>11m</td></tr><tr><td>238</td><td>0.9129</td><td>32.72</td><td>0.1748</td><td>0.8707</td><td>30.14</td><td>0.1931</td><td>11m</td></tr><tr><td>274</td><td>0.8981</td><td>31.01</td><td>0.1859</td><td>0.8470</td><td>28.67</td><td>0.2138</td><td>11m</td></tr><tr><td>306</td><td>0.8778</td><td>29.90</td><td>0.1978</td><td>0.8121</td><td>27.26</td><td>0.2307</td><td>11m</td></tr><tr><td>324</td><td>0.9227</td><td>32.76</td><td>0.1818</td><td>0.8628</td><td>28.80</td><td>0.2122</td><td>11m</td></tr><tr><td>326</td><td>0.8800</td><td>29.11</td><td>0.2041</td><td>0.8425</td><td>28.12</td><td>0.2357</td><td>11m</td></tr></table>

TABLE XII

QUANTITATIVE RESULTS FOR DIFFERENT PATIENTS IN THE CASE OF OURS-100K ON REAL DATA. TRAINED ON 150 EPOCHS
<table><tr><td>Pat. ID</td><td>SSIM</td><td>Training PSNR</td><td>LPIPS</td><td>SSIM</td><td>Test PSNR</td><td>LPIPS</td><td>Time</td></tr><tr><td>Sag.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>152</td><td>0.9157</td><td>31.21</td><td>0.1638</td><td>0.8471</td><td>28.12</td><td>0.2118</td><td>16m</td></tr><tr><td>154</td><td>0.9157</td><td>31.80</td><td>0.1590</td><td>0.8613</td><td>29.19</td><td>0.1879</td><td>16m</td></tr><tr><td>159</td><td>0.9161</td><td>31.48</td><td>0.1523</td><td>0.8588</td><td>28.68</td><td>0.1877</td><td>16m</td></tr><tr><td>231</td><td>0.9167</td><td>31.77</td><td>0.1526</td><td>0.8471</td><td>28.86</td><td>0.2055</td><td>16m</td></tr><tr><td>232</td><td>0.9415</td><td>32.88</td><td>0.1326</td><td>0.8792</td><td>29.00</td><td>0.1761</td><td>16m</td></tr><tr><td>238</td><td>0.9368</td><td>33.69</td><td>0.1253</td><td>0.8781</td><td>29.30</td><td>0.1656</td><td>16m</td></tr><tr><td>274</td><td>0.9281</td><td>32.60</td><td>0.1306</td><td>0.8644</td><td>28.86</td><td>0.1772</td><td>16m</td></tr><tr><td>306</td><td>0.9182</td><td>31.65</td><td>0.1406</td><td>0.8542</td><td>28.40</td><td>0.1791</td><td>16m</td></tr><tr><td>324</td><td>0.9439</td><td>33.63</td><td>0.1277</td><td>0.8771</td><td>28.82</td><td>0.1746</td><td>16m</td></tr><tr><td>326</td><td>0.9199</td><td>31.71</td><td>0.1446</td><td>0.8683</td><td>29.28</td><td>0.1800</td><td>16m</td></tr><tr><td>Trans.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>152</td><td>0.9215</td><td>30.97</td><td>0.1574</td><td>0.8602</td><td>28.26</td><td>0.1926</td><td>16m</td></tr><tr><td>154</td><td>0.9027</td><td>30.38</td><td>0.1679</td><td>0.8409</td><td>27.89</td><td>0.2080</td><td>16m</td></tr><tr><td>159</td><td>0.9227</td><td>31.69</td><td>0.1523</td><td>0.8802</td><td>29.34</td><td>0.1833</td><td>16m</td></tr><tr><td>231</td><td>0.9083</td><td>30.26</td><td>0.1623</td><td>0.8323</td><td>27.63</td><td>0.2066</td><td>16m</td></tr><tr><td>232</td><td>0.9448</td><td>32.80</td><td>0.1369</td><td>0.8908</td><td>29.40</td><td>0.1804</td><td>16m</td></tr><tr><td>238</td><td>0.9348</td><td>33.71</td><td>0.1333</td><td>0.8881</td><td>30.53</td><td>0.1578</td><td>16m</td></tr><tr><td>274</td><td>0.9224</td><td>31.87</td><td>0.1430</td><td>0.8620</td><td>28.93</td><td>0.1834</td><td>16m</td></tr><tr><td>306</td><td>0.9097</td><td>30.94</td><td>0.1506</td><td>0.8324</td><td>27.63</td><td>0.1886</td><td>16m</td></tr><tr><td>324</td><td>0.9399</td><td>33.48</td><td>0.1413</td><td>0.8719</td><td>28.93</td><td>0.1823</td><td>16m</td></tr><tr><td>326</td><td>0.9077</td><td>29.77</td><td>0.1610</td><td>0.8632</td><td>28.39</td><td>0.1921</td><td>16m</td></tr></table>