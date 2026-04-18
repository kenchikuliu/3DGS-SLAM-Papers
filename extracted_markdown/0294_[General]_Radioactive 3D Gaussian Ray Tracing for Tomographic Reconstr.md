# Radioactive 3D Gaussian Ray Tracing for Tomographic Reconstruction

Ling Chen Independent Researcher

Bao Yang Southern Medical University

## Abstract

3D Gaussian Splatting (3DGS) has recently emerged in computer vision as a promising rendering technique. By adapting the principles of Elliptical Weighted Average (EWA) splatting to a modern differentiable pipeline, 3DGS enables real-time, high-quality novel view synthesis. Building upon this, R2-Gaussian extended the 3DGS paradigm to tomographic reconstruction by rectifying integration bias, achieving state-of-the-art performance in computed tomography (CT). To enable differentiability, R2-Gaussian adopts a local affine approximation: each 3D Gaussian is locally mapped to a 2D Gaussian on the detector and composed via alpha blending to form projections. However, the affine approximation can degrade reconstruction quantitative accuracy and complicate the incorporation of nonlinear geometric corrections. To address these limitations, we propose a tomographic reconstruction framework based on 3D Gaussian ray tracing. Our approach provides two key advantages over splatting-based models: (i) it computes the line integral through 3D Gaussian primitives analytically, avoiding the local affine collapse and thus yielding a more physically consistent forward projection model; and (ii) the ray-tracing formulation gives explicit control over ray origins and directions, which facilitates the precise application of nonlinear geometric corrections, e.g., arc-correction used in positron emission tomography (PET). These properties extend the applicability of Gaussian-based reconstruction to a wider range of realistic tomography systems while improving projection accuracy.

## 1. Introduction

Tomographic imaging modalities, such as Computed Tomography (CT) [17, 20, 46] and Positron Emission Tomography (PET) [3, 5, 42], are widely used for medical diagnosis, non-destructive industrial testing, and scientific research. The primary objective is to accurately reconstruct the internal three-dimensional anatomical structure or functional distribution of a subject. For CT, classical analytical algorithms such as FeldkampâDavisâKress (FDK) [14] are computationally efficient but suffer severe quality degradation under sparse projections, high noise levels, or physical effects like beam hardening. For PET, statistical iterative methods, e.g., Ordered Subsets Expectation Maximization (OSEM) [18, 35], model the imaging physics and the Poisson measurement noise, yielding improved image quality as compared to analytic methods; however, their performance depends strongly on the accuracy of the system matrix, which can be difficult to obtain or expensive to store.

Recently, implicit neural representations from computer vision, such as Neural Radiance Fields (NeRF) [10, 27, 30, 48], introduced a new paradigm for 3D reconstruction by learning continuous mappings from spatial coordinates to radiance and density. However, these methods typically require long training and rendering times, limiting their practicality in clinical or industrial settings that demand fast turnaround. A major step toward practicality is 3D Gaussian Splatting (3DGS) [21, 23, 38, 41, 47], which replaces implicit MLPs with many explicit, optimizable 3D Gaussian primitives. Each primitive is parameterized by position, orientation, scale, opacity and view-dependent coefficients; a differentiable rasterization pipeline projects these Gaussians to the image plane and blends them via alpha compositing by combining classical Elliptical Weighted Average (EWA) splatting [52, 53] with gradient-based optimization. 3DGS achieves much faster training and real-time novel-view synthesis while retaining high visual quality.

Motivated by these capabilities, researchers have adapted Gaussian representations for tomographic reconstruction. R2-Gaussian [49] is a notable example that adapts 3DGS for CT reconstruction by addressing an integration bias in the original splatting-based projection. In that framework, the scene is represented by Gaussians whose attributes correspond to physical quantities, e.g., attenuation, and the Gaussian parameters are optimized so that rasterized projections match measured scan data. R2-Gaussian demonstrates promising high-resolution reconstructions on CT by introducing a projection model more consistent with respect to views. Similar to 3DGS, R2-Gaussian adopts a local affine approximation: a 3D Gaussian is locally collapsed into a 2D Gaussian on the detector via an affine mapping, and contributions are accumulated for standard alpha compositing. This approximation simplifies gradient computation and improves efficiency, but it also can degrade integration exactnessâan issue for quantitative imaging that demands high accuracy. Moreover, some real imaging systems exhibit geometric characteristics, e.g, cylindrical arrangement of PET detectors [43]. In PET, coincident gamma detections define Lines of Response (LORs) that may require arc correction [9] to reconcile the detector geometry with reconstruction assumptions. Methods relying on local affine splatting encounter difficulties in incorporating such non-uniform bin spacing in the sinogram precisely, which limits their applicability in systems where accurate geometric modeling is essential.

To address these limitations, we propose a tomographic reconstruction framework based on 3D Gaussian ray tracing [12, 31]. The framework is straightforward for tomographic reconstruction as shown in Fig. 1. Instead of projecting into 2D screen space via local affine approximation, our approach returns to the physical primitives of tomographic imagingâray tracing and exact line integralsâbringing two main advantages:

Accurate projective model. We derive and implement an analytical expression for the exact line integral of a ray through a 3D Gaussian primitive, eliminating the local affine approximation and improving numerical consistency. Compatibility with different scanner geometries. 3D Gaussian Ray tracing gives explicit control over ray origins and directions, allowing us to compatible with different scanner geometric characteristics precisely during ray setup and thus adapt the method to imaging systems such as PET.

The proposed framework is quantitatively and qualitatively evaluated on both PET and CT modalities. For PET, we perform (i) an analytical National Electrical Manufacturers Association (NEMA) phantom [4] simulation to evaluate reconstruction performance, (ii) a three-point-source Monte Carlo simulation to investigate the influence of arc versus non-arc correction, and (iii) qualitative assessment using realistic PET data. For CT, experiments are conducted on the same synthetic and real-world datasets as $\mathtt { R } ^ { 2 } \mathtt { - }$ Gaussian, with reconstruction quality compared using 3D PSNR and 3D SSIM metrics.

## 2. Related work

## 2.1. Tomographic Reconstruction

The core objective of tomographic reconstruction algorithms is to reconstruct images of an internal structure from externally acquired projection data, primarily categorized into analytical and iterative methods. Analytical methods [36], represented by FDK, are based on rigorous mathematical models and are computationally efficient, serving as the standard algorithm for scenarios with complete, high signalto-noise ratio data. Iterative methods [44] employ statistical models, e.g., OSEM, approaching an optimal solution through successive iterations, offering superior advantages when handling incomplete or low signal-to-noise ratio data.

The choice between reconstruction algorithms for CT and PET is profoundly influenced by their underlying physical principles and data characteristics. CT measures the deterministic attenuation of X-rays, resulting in high-quality data, which is why FDK has long been its mainstream algorithm; iterative methods are primarily utilized in low-dose CT scenarios to suppress noise. In contrast, PET measures stochastic coincidence counts that follow a Poisson distribution, meaning the data is inherently characterized by low signal-to-noise ratio. This makes iterative methods the absolute dominant reconstruction technique for PET, as they effectively handle noise and allow for the incorporation of physical corrections.

## 2.2. R2-Gaussian

R2-Gaussian introduces the concept of explicit Gaussian primitives, previously used in neural rendering, into the task of tomographic reconstruction. It proposes a Gaussian representation and forward projection pipeline adapted for the physics of X-ray imaging. Unlike the typical 3DGS, R2-Gaussian re-derives the projection relationship from 3D anisotropic Gaussians to the 2D detector plane by addressing integration bias. This constructs a radiative Gaussian kernel formulation that is more consistent with radiative transfer. Based on this formulation, Zha et al. [49] implemented a 3D Gaussian splatting frame work for tomographic reconstruction.

Regarding scaling integration bias, the core idea of R2- Gaussian is to recover the 3D density volume by 3D Gaussian splatting. This approach corrects the density inconsistency in 3DGS, ensuring the contribution of a single Gaussian to a pixel remains consistent across different projection angles. To maintain efficiency, R2-Gaussian is typically coupled with sparse acceleration strategies, such as frustum culling, to limit the number of primitives requiring alpha compositing.

## 2.3. 3D Gaussian Ray Tracing

3D Gaussian Ray Tracing treats the scene as a volumetric collection of 3D Gaussian primitives [15, 31, 50]. It directly computes ray-primitive intersections and evaluates the corresponding integrals along pixel rays, rather than splatting the Gaussians into screen space and then performing approximate blending. By constructing a Bounding Volume Hierarchy (BVH) [29] and leveraging high-performance GPU ray-tracing hardware, the method casts rays for each pixel and processes depth-ordered intersections in batches, while the sorting process can be waived if it applies to tomographic reconstruction.

<!-- image-->  
Figure 1. Overview of 3D Gaussian ray tracing for tomographic reconstruction. Given the scanner geometry, each rayâs origin and direction are defined. Along each ray, we compute the pixel value by analytically integrating the contributions of all 3D Gaussian primitives the ray encounters. By optimizing the Gaussian parameters so that the rendered projections match the measured projections, we obtain a tomographic reconstruction by voxelizing the optimized Gaussians.

Compared to splatting-based pipelines, ray tracing offers three key advantages: First, it can naturally and accurately evaluate the line integral and transmittance along the ray, thereby avoiding the assumption of non-overlapping Gaussians. Second, not only 3D Gaussian kernel but also a more compact kernel, e.g., Epanechnikov kernel [11], can be applied to ray tracing [12]. Third, we can define the ray origin and direction very flexible, which make this method compatible with different geometric characteristics of scanner. Thus, for application scenarios requiring more physically consistent projection or complex geometric correction, 3D Gaussian Ray Tracing demonstrates significant advantages over traditional 3D Gaussian Splatting in terms of numerical accuracy and flexibility.

## 3. Method

In this section, we outline the preliminaries of $\mathbf { R } ^ { 2 } .$ -Gaussian and the limitations we encounter in PET image reconstruction. After that, we present a detailed derivation of 3D Gaussian ray tracing for tomographic reconstruction.

R2-Gaussian. The core innovation in $\mathbf { R } ^ { 2 } .$ -Gaussian lies in its derivation of a density-consistent integral for Gaussians, which reveals a previously unidentified integration bias in the standard 3DGS formulation. This bias emerges from approximations in the Gaussian projection integral, leading to significant inconsistency in density retrieval. To address this, $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ -Gaussian introduces a covariance-related factor to scale the density and achieve accurate density retrieval. Nevertheless, the projection process still relies on the local affine approximation:

$$
I _ { r } ( \mathbf { r } ) \approx \sum _ { i = 1 } ^ { M } \int G _ { i } \left( \tilde { \mathbf { x } } \mid \rho _ { i } , \phi ( \mathbf { p } ) , \mathbf { J } _ { i } \mathbf { W } \Sigma _ { i } \mathbf { W } ^ { \top } \mathbf { J } _ { i } ^ { \top } \right) d x _ { 2 } ,\tag{1}
$$

where $G _ { i }$ denotes a 3D Gaussian kernel and $\begin{array} { r l } { \tilde { \mathbf { x } } } & { { } = } \end{array}$ $[ x _ { 0 } , x _ { 1 } , x _ { 2 } ]$ represents a coordinate point in the ray space. In addition, $\rho _ { i } , p _ { i } ,$ and $\Sigma _ { i }$ are learnable parameters corresponding to central density, position and covariance, respectively. Moreover, $\mathbf { J } _ { i }$ is the local approximation matrix and W is the viewing transformation matrix.

Inherent limitations. In applying $\mathbf { R } ^ { 2 } .$ -Gaussian to PET image reconstruction, several challenges arise from the unique physical characteristics of the modality. $\mathbf { R } ^ { 2 } .$ Gaussian was originally designed for cone-beam or parallelbeam CT reconstruction. In PET, cross-segment LORs [13, 25], namely, those with non-zero ring difference, produce oblique projections, so we need to update the projector with a shear transformation; this update is straightforward to implement. A key consideration is the need for arc correction, which is required by the cylindrical arrangement of detectors in PET scanners. This geometry produces nonuniform bin spacing in the sinogram, as shown in Fig. 2, requiring the forward projection model to accurately represent both the cylindrically arranged detectors and the varying spacings of LORs. Traditional 3DGS methods, which rely on local affine approximations, are difficult to adapt to such arc-corrected projections. The same limitation applies to the tomographic version of 3DGS, namely $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ -Gaussian. Although conservative resampling could be applied, it compromises the fidelity of the raw measured projections. Furthermore, PET is a quantitative imaging modality where reconstruction accuracy directly affects clinical measurements such as standardized uptake values (SUVs) [39], which are used to assess metabolic activity. Therefore, the projection model must maintain high quantitative accuracy and avoid approximations that could compromise diagnostic reliability or introduce bias in density estimation.

<!-- image-->  
Figure 2. An example of a cylindrical detector arrangement that produces non-uniform tangential bin spacing in the sinogram. The sinogram before arc correction corresponds to straightening the arc-shaped detectors into a line (top one). However, the physical spacing between bins, namely, measured along chord, is nonuniform (bottom one).

## 3.1. Overall Framework

These limitations motivate a tomographic reconstruction model that can not only provide more accurate line integrals but also be geometrically flexible for ray alignment. Thus, we employ 3D Gaussian ray tracing for tomographic reconstruction. By ray tracing, we can calculate the line integrals analytically without approximation and also define ray origins flexibly. Below we derive the analytic line integral of a 3D anisotropic Gaussian along a ray.

Let a single anisotropic 3D Gaussian, with the constant normalization absorbed into a density scalar, be written as:

$$
G ( \pmb { x } ) = \mathrm { e x p } \Big ( - \frac { 1 } { 2 } ( \pmb { x } - \pmb { \mu } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \pmb { x } - \pmb { \mu } ) \Big ) ,\tag{2}
$$

where $\textbf { \textit { x } } \in \ \mathbb { R } ^ { 3 }$ represents a coordinate point in the ray space, $\pmb { \mu }$ is the mean that controls the position of the 3D Gaussian ellipsoid, and Î£ is the covariance, which can be decomposed into a scale matrix S and a rotation matrix R: $\Sigma \stackrel { \cdot } { = } R S S ^ { \top } R ^ { \top }$ . Since a learnable density scalar $\rho ,$ which can be adjusted during optimization, multiplies the 3D Gaussian kernel during the line integral, there is no need to explicitly handle the normalization term $\frac { 1 } { ( 2 \pi ) ^ { 3 / 2 } | \pmb { \Sigma } | ^ { 1 / 2 } } .$

A ray $\mathbf { } _ { \pmb { r } ( t ) }$ is parameterized by the ray origin o and the ray direction d:

$$
r ( t ) = o + t d ,\tag{3}
$$

where $t \in \mathbb { R } , o \in \mathbb { R } ^ { 3 }$ , and $\mathbf { \Delta } d \in \mathbb { R } ^ { 3 }$ . We consider the line integral over all 3D Gaussians as:

$$
\begin{array} { c } { { \displaystyle I ( { \pmb r } ) = \int _ { - \infty } ^ { \infty } \displaystyle \sum _ { i = 1 } ^ { M } \rho _ { i } \cdot G _ { i } \left( { \pmb r } ( t ) \right) d t , } } \\ { { \displaystyle I ( { \pmb o } , { \pmb d } ) = \sum _ { i = 1 } ^ { M } \rho _ { i } \cdot \int _ { - \infty } ^ { \infty } G _ { i } \left( { \pmb o } + t { \pmb d } \right) d t , } } \end{array}\tag{4}
$$

where $I ( r )$ represents the rendered pixel value along ray r, M is the number of 3D Gaussians in the ray space, and $\rho _ { i }$ is the central density of the i-th 3D Gaussian.

## 3.2. Quadratic Exponent Along the Ray

We consider the line integral for a single Gaussian without density as:

$$
\begin{array} { r } { \dot { I } ( o , d ) = \int _ { - \infty } ^ { \infty } \exp \Bigl ( - \frac { 1 } { 2 } ( o + t d - \mu ) ^ { \top } \Sigma ^ { - 1 } ( o + t d - \mu ) \Bigr ) d t , } \end{array}\tag{5}
$$

Define $\delta = o - \mu$ . Substituting into the exponent yields:

$$
{ \begin{array} { r l } & { \operatorname { e x p o n e n t } = - { \frac { 1 } { 2 } } ( o + t d - \mu ) ^ { \top } \Sigma ^ { - 1 } ( o + t d - \mu ) } \\ & { \qquad = - { \frac { 1 } { 2 } } ( \delta + t d ) ^ { \top } \Sigma ^ { - 1 } ( \delta + t d ) } \\ & { \qquad = - { \frac { 1 } { 2 } } ( \delta ^ { \top } \Sigma ^ { - 1 } \delta + 2 t d ^ { \top } \Sigma ^ { - 1 } \delta + t ^ { 2 } d ^ { \top } \Sigma ^ { - 1 } d ) . } \end{array} }\tag{6}
$$

Since $\pmb { \Sigma } ^ { - 1 }$ is symmetric, $\delta ^ { \top } \Sigma ^ { - 1 } d = d ^ { \top } \Sigma ^ { - 1 } \delta$ . Here, we define:

$$
\begin{array} { r } { A = \pmb { d } ^ { \top } \pmb { \Sigma } ^ { - 1 } \pmb { d } , } \\ { B = \pmb { d } ^ { \top } \pmb { \Sigma } ^ { - 1 } \pmb { \delta } , } \\ { C = \pmb { \delta } ^ { \top } \pmb { \Sigma } ^ { - 1 } \pmb { \delta } . } \end{array}\tag{7}
$$

Substituting Eq. (7) into Eq. (6) gives the quadratic form of the exponent along the ray:

$$
\textstyle { \mathrm { e x p o n e n t } } = - { \frac { 1 } { 2 } } ( C + 2 B t + A t ^ { 2 } ) .\tag{8}
$$

## 3.3. Closed Form Integral

Substituting the quadratic expression into Eq. (5), the integral becomes:

$$
\begin{array} { r l } & { \dot { I } ( o , d ) = \displaystyle \int _ { - \infty } ^ { \infty } \exp \Big ( - \frac { 1 } { 2 } ( C + 2 B t + A t ^ { 2 } ) \Big ) d t , } \\ & { \qquad = \exp \Big ( - \frac { 1 } { 2 } C \Big ) \displaystyle \int _ { - \infty } ^ { \infty } \exp \Big ( - \frac { 1 } { 2 } ( 2 B t + A t ^ { 2 } ) \Big ) d t , } \\ & { \qquad = \exp \Big ( - \frac { 1 } { 2 } C \Big ) \displaystyle \int _ { - \infty } ^ { \infty } \exp \Big ( - \frac { 1 } { 2 } A ( t + \frac { B } { A } ) ^ { 2 } + \frac { B ^ { 2 } } { 2 A } \Big ) d t , } \\ & { \qquad = \exp \Big ( - \frac { 1 } { 2 } ( C - \frac { B ^ { 2 } } { A } ) \Big ) \displaystyle \int _ { - \infty } ^ { \infty } \exp \Big ( - \frac { 1 } { 2 } A ( t + \frac { B } { A } ) ^ { 2 } \Big ) d t } \end{array}\tag{t.}
$$

(9)

Let $\textstyle h = t + { \frac { B } { A } }$ . The remaining integral is the standard Gaussian integral:

$$
\int _ { - \infty } ^ { \infty } \exp \Bigl ( - \textstyle { \frac { 1 } { 2 } } A h ^ { 2 } \Bigr ) d h = \sqrt { \frac { 2 \pi } { A } } .\tag{10}
$$

Therefore, the line integral for a single Gaussian without density is:

$$
\begin{array} { r } { \dot { I } ( o , d ) = \sqrt { \frac { 2 \pi } { A } } \cdot \mathrm { e x p } \Big ( - \frac { 1 } { 2 } \big ( C - \frac { B ^ { 2 } } { A } \big ) \Big ) . } \end{array}\tag{11}
$$

Finally, the line integral along the ray for all Gaussians becomes:

$$
\begin{array} { r } { \boxed { I ( \pmb { r } ) = \displaystyle \sum _ { i = 1 } ^ { M } \rho _ { i } \cdot \sqrt { \frac { 2 \pi } { A _ { i } } } \cdot \exp \biggl ( - \frac { 1 } { 2 } \bigl ( C _ { i } - \frac { B _ { i } ^ { 2 } } { A _ { i } } \bigr ) \biggr ) } } \end{array}\tag{12}
$$

This expression gives an analytic, differentiable forward projection for the 3D Gaussian model and avoids the local affine projection approximation commonly used in splatting-based methods.

## 4. Experiment

## 4.1. Experimental Settings

PET dataset. The PET dataset contains three parts. First, we analytically simulate the NEMA phantom using the Software for Tomographic Image Reconstruction (STIR) v6.2 [40] for a Siemens Biograph mMR scanner and add Poisson noise to the sinogram afterward. STIR can simulate arc-corrected sinograms by resampling the raw projections using an overlap-interpolation method. Then, for the threepoint-source Monte Carlo simulation to investigate the influence of arc versus non-arc correction, we follow the protocol in the PET/CT Acceptance Testing and Quality Assurance Report from the American Association of Physicists in Medicine (AAPM) [28], using the GEANT4 Application for Tomographic Emission (GATE) v9.2 [37]. In the simulation, three point sources are located at (0, 1), (0, 10), and (0, 20) cm, respectively, where (0,0) cm is the isocenter of the scanner. The stopping condition for the simulation is acquisition of 5 million counts. The scanner used in this simulation was a General Electric Discovery 690. The raw output data are converted to sinograms for reconstruction by STIR. Finally, for qualitative assessment, a realistic brain data was collected from a collaborating institute, which was used under approval by the institutional review board, and informed consent was waived because the data were analyzed retrospectively and anonymized.

Table 1. Comparison of reconstruction times for the NEMA phantom.
<table><tr><td>Method</td><td colspan="2">OSEM R2-Gaussian Our method</td></tr><tr><td>Time (min)</td><td>17 12</td><td>35</td></tr></table>

CT dataset. To facilitate comparison with $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ -Gaussian, we employ the same synthetic and real CT datasets to evaluate performance differences. To establish a benchmark for sparse-view reconstruction, three scenarios with 75, 50, and 25 views are defined and applied to both synthetic and realworld data. The synthetic data were derived from 15 diverse CT volumes of natural and artificial objects, including human organs, animals and plants, and artificial objects. X-ray projections were computationally generated using the TI-GRE toolbox [7] and subsequently contaminated with models of electronic noise. For real-world data evaluation, three objects from the Finnish Inverse Problems Society (FIPS) dataset are utilized, each containing 721 real projections. In the absence of definitive ground truths, pseudo-groundtruth volumes were generated by applying the FDK algorithm [14] to the complete set of views.

Implementation details. The implementation of 3D Gaussian ray tracing for tomographic reconstruction is based on PyTorch [34] and the NVIDIA OptiX ray tracing engine v6.0 [33]. All experiments are run using the Adam optimizer [1] with the same Gaussian densify-and-prune strategy as $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ -Gaussian [49]. For ray tracing, an icosahedron is used to represent a Gaussian. Back-face culling is enabled to trigger our custom Anyhit program for accumulating the analytic integral of each Gaussian. To accelerate the training process, we also skip Gaussians whose contributions are negligible. All tasks are conducted on an RTX 3080 Ti GPU.

## 4.2. Experimental Results

The NEMA phantom contains six spheres of different sizes with a hollow cylinder in the middle. The six spheres are numbered 0 to 5 according to their inner diameter, which are 10 mm, 13 mm, 17 mm, 22 mm, 28 mm, and 37 mm, respectively. Since R2-Gaussian previously only worked for parallel-beam or cone-beam projections, we updated the projector to make it feasible for oblique projections, which is necessary for utilizing cross-segment LORs in PET reconstruction. Furthermore, we compare with the OSEM implementation in STIR as a baseline. We compare three reconstruction results: (1) noisy NEMA phantom reconstructed by OSEM; (2) noisy NEMA phantom reconstructed by $\mathrm { R } ^ { \dot { 2 } . }$ Gaussian; and (3) noisy NEMA phantom reconstructed by our method. Reconstruction performance is evaluated in terms of (i) signal-to-background ratio (SBR), (ii) recovered sphere diameters, and (iii) the standard deviation within each sphere. These metrics indicate quantitative accuracy, reconstruction fidelity, and noise level, respectively. The reconstruction times are summarized in Tab. 1. Owing to the highly parallel nature of splatting-based methods, $\mathtt { R } ^ { 2 } \mathtt { \Gamma }$ -Gaussian is the fastest of the three methods. OSEM is intermediate. Our method is the slowest, taking almost twice as long as $\mathbf { R } ^ { 2 } \mathbf { \mathrm { \Sigma } }$ -Gaussian.

Tab. 2 summarizes the reconstructed sphere diameters for the three methods. The diameter is obtained by the full width half maximum of a sphere. From this table, both $\mathbf { R } ^ { 2 } .$ Gaussian and OSEM have three out of six spheres with diameter error less than 5%. Notably, $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ -Gaussian perfectly reconstructed the smallest sphere (10 mm), which is generally the most challenging target. Our method has two out of six spheres with error less than 5%, suggesting that recovery of true diameters by our method is slightly inferior to $\mathrm { \tt R } ^ { \dot { 2 } . }$ -Gaussian and OSEM in these experiments.

The activity ratio between spheres and background is set to 4:1; an SBR closer to 4 indicates better quantitative accuracy. Tab. 3 summarizes the SBR results for all spheres.To calculate the SBR, we use the measured FWHM as the sphere diameter, compute the mean activity within that diameter, and obtain the SBR by dividing this mean by the mean activity of a 20Ã20Ã20 background volume.From the table, our method provides the best quantitative accuracy among the three methods: five out of six spheres have relative error smaller than 5%. $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ -Gaussian is the second best, with two out of six spheres within 5% error. OSEM performs worst in SBR accuracy, with only one sphere within 5% error.

Fig. 3 shows the standard deviation inside each sphere for the three methods to quantify noise level. OSEM exhibits the lowest standard deviations overall, indicating stronger noise suppression. Both our method and $\mathrm { \mathbf { R } } ^ { 2 } \mathrm { - }$ Gaussian show higher standard deviations, particularly for the smaller spheres, which indicates higher noise levels in those regions. Taken together, the results indicate a tradeoff: our method achieves the best quantitative accuracy across most spheres, while OSEM yields the lowest noise, and $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ -Gaussian provides a balance with good fidelity for small objects in some cases.

Fig. 4 plots the line profile along the central horizontal line of the NEMA phantom reconstructions produced by each method. Note that the central cylinder of the NEMA phantom is empty and therefore should have zero activity. From the line profiles on the right, we can directly compare how the three methods preserve the zero background in the hollow cylinder: OSEM shows a positive bias in reconstructed activity inside the hollow cylinder, as shown in Fig. 4a, indicating contamination or a background offset; $\mathbf { R } ^ { 2 } .$ -Gaussian and our method stay close to zero there, showing they better preserve the low-signal property of the hollow region. The positive bias in OSEM may come from incomplete correction for scatter and randoms. However, methods based on 3D Gaussians can better suppress such biases.

<!-- image-->  
Figure 3. Standard deviation within each NEMA phantom sphere for OSEM, $\mathbb { R } ^ { 2 }$ -Gaussian, and our method (lower values indicate better noise suppression).

Point sources are commonly used to measure the spatial resolution of PET systems [16], since spatial resolution is typically defined as the FWHM of the Point Spread Function (PSF) [2] and is calculated from the line profile through a reconstructed image of a point source in air. The results of the three-point-source Monte Carlo simulation are shown in Fig. 5. All reconstructions were produced by our method; Fig. 5a and Fig. 5b show results without and with arc correction, respectively. Here we also measured the FWHM of the three point sources. For the non-arc case, the FWHMs are 3.43, 3.69, and 5.56 pixel, respectively. For the arccorrected case, they are 3.45, 3.82, and 4.56 pixel, respectively. Since the three point sources were set to the same activity, their reconstructed values should be similar. The arccorrected reconstruction yields values that are closer to each other. The results show that applying arc correction can improve spatial resolution and quantitative accuracy [9, 19], especially for sources far from the center of field of view. Fig. 6 shows reconstructions of realistic brain data by (a)

Table 2. Measured diameters (mm) for each NEMA phantom sphere reconstructed by OSEM, $\mathbf { R } ^ { 2 } \cdot$ -Gaussian, and our method. Parentheses show relative error versus ground truth.
<table><tr><td>Method</td><td>Sphere0</td><td>Sphere1</td><td>Sphere2</td><td>Sphere3</td><td>Sphere4</td><td>Sphere5</td></tr><tr><td>OSEM</td><td>10.9 (8.9%)</td><td>13.7 (-21.4%)</td><td>17.2 (1.2%)</td><td>20.6 (-6.3%)</td><td>27.7 (-1.0%)</td><td>37.3 (0.7%)</td></tr><tr><td> $\mathbf { R } ^ { 2 } .$  Gaussian</td><td>10.0 (-0.4%)</td><td>10.8 (17.1%)</td><td>15.7 (-7.4%)</td><td>21.3 (-3.4%)</td><td>28.4 (1.3%)</td><td>39.2 (6.0%)</td></tr><tr><td>Our method</td><td>9.5 (-5.1%)</td><td>10.8 (16.6%)</td><td>15.1 (-11.1%)</td><td>21.2 (-3.7%)</td><td>27.6 (-1.3%)</td><td>39.7 (7.2%)</td></tr></table>

Table 3. Signal-to-background ratios for each NEMA phantom sphere reconstructed by OSEM, $\mathrm { R } ^ { 2 } \mathrm { - G a u s s i a n } .$ , and our method. Parentheses show percentage error relative to the reference activity ratio (4:1).
<table><tr><td>Method</td><td>Sphere0</td><td>Sphere1</td><td>Sphere2</td><td>Sphere3</td><td>Sphere4</td><td>Sphere5</td></tr><tr><td>OSEM</td><td>2.15 (-46.3%)</td><td>3.14 (-21.4%)</td><td>3.35 (-16.1%)</td><td>3.75 (-6.2%)</td><td>3.78 (-5.4%)</td><td>3.99 (-0.2%)</td></tr><tr><td>R2-Gaussian</td><td>2.99 (-25.2%)</td><td>4.38 (9.6%)</td><td>4.14 (3.4%)</td><td>4.24 (6.1%)</td><td>4.20 (5.1%)</td><td>4.06 (1.5%)</td></tr><tr><td>Our method</td><td>3.00 (-24.5%)</td><td>3.97 (-0.7%)</td><td> $3 . 9 0 \ : ( - 2 . 6 \% )$ </td><td>4.16 (3.9%)</td><td>4.13 (3.2%)</td><td>4.07 (1.8%)</td></tr></table>

<!-- image-->

<!-- image-->  
(a) OSEM.

<!-- image-->

<!-- image-->

<!-- image-->

(b) $\begin{array} { r } { \mathbf { R } ^ { 2 } . } \end{array}$ -Gaussian.  
<!-- image-->

<!-- image-->  
(c) Our method.  
Figure 4. NEMA phantom PET reconstructions by (a) OSEM, (b) $\mathrm { R } ^ { 2 } \cdot$ Gaussian, and (c) our method. Corresponding central horizontal line profiles are plotted on the right.

(a) Non-arc-correction.  
<!-- image-->  
(b) Arc correction.  
Figure 5. Three-point-source PET reconstructions: (a) without arc correction; (b) with arc correction. From left to right, sources are at (0,1), (0,10), and (0,20) cm. Central horizontal line profiles are shown on the right.

OSEM, (b) $\mathtt { R } ^ { 2 } \mathtt { - }$ -Gaussian, and (c) our method. Visually, our method provides the most detail among the three. In the left column, the sulci and gyri of the cerebral cortex are more clearly visible in our result. In the middle column, the putamen and caudate nucleus, which are important in the diagnosis of Parkinsonâs and Alzheimerâs diseases [8, 22], are most clearly depicted by our method. In the right column, the OSEM and ${ \bf \bar { R } } ^ { 2 } .$ -Gaussian reconstructions show artifacts in the lower edge, while our reconstruction does not. Overall, among the three methods our method preserves image details best while exhibiting greater robustness to data noise in realistic PET data reconstruction.

<!-- image-->

<!-- image-->  
(a) OSEM.

<!-- image-->

<!-- image-->

<!-- image-->  
(b) R2-Gaussian.

<!-- image-->

<!-- image-->

<!-- image-->  
(c) Our method.

<!-- image-->  
Figure 6. A realistic brain PET data reconstructions by (a) OSEM, (b) $\mathbb { R } ^ { 2 }$ -Gaussian, and (c) our method.

Table 4. Performance comparison on synthetic CT dataset.
<table><tr><td>Method</td><td>75 views</td><td>50 views</td><td>25 views</td><td></td></tr><tr><td></td><td>PSNR SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR SSIM</td></tr><tr><td>R2-Gaussian</td><td>38.30 0.937</td><td>35.02</td><td>0.902</td><td>32.95 0.866</td></tr><tr><td>Our method</td><td>39.36 0.937</td><td>35.13</td><td>0.902</td><td>32.99 0.865</td></tr></table>

Table 5. Performance comparison on real-world CT dataset.
<table><tr><td>Method</td><td>75 views</td><td>50 views</td><td>25 views</td></tr><tr><td></td><td>PSNR SSIM</td><td>PSNR SSIM</td><td>PSNR SSIM</td></tr><tr><td>R2-Gaussian</td><td>35.71 0.928</td><td>34.22 0.915</td><td>30.56 0.852</td></tr><tr><td>Our method</td><td>35.79 0.928</td><td>34.30 0.915</td><td>30.68 0.856</td></tr></table>

The CT reconstruction results on the synthetic and realworld datasets are summarized in Tab. 4 and Tab. 5, respectively. There is little difference in SSIM between $\mathtt { R } ^ { 2 } \mathtt { - }$ Gaussian and our method. In terms of PSNR, our method achieves higher values than $\mathbf { R } ^ { 2 } .$ -Gaussian in all tested cases. A paired t-test indicates that the PSNR improvement is statistically significant $\left( \mathtt { p } = 0 . 0 0 4 8 \right)$ .

In practice, clinical CT is often interpreted qualitatively, and quantitative accuracy is typically less critical than in PET. Considering the trade-off between reconstruction quality and runtime, $\mathbf { R } ^ { 2 } .$ -Gaussian may still be preferred for CT applications where speed is the dominant concern.

For PET reconstruction, our method demonstrates improved quantitative accuracy and greater flexibility to accommodate various scanner geometries, making it more suitable for quantitative PET tasks that require precise geometric modeling.

## 5. Limitations and Future Works

Limitations. Unlike CT, there are few publicly available PET datasets, and PET data cannot be synthesized as straightforwardly as CT data because PET detects pairs of coincident gamma photons emitted in nearly 180Â° opposite directions within the imaged subject. Hence, in this study we relied on analytical and Monte Carlo simulations to approximate realistic measurements. High-fidelity Monte Carlo is extremely computationally expensive and therefore limited in scale, while analytical phantom simulations are cheap but do not capture many real-world effects, e.g., scatter, randoms. In addition, although the analytic integrals of 3D Gaussian ray tracing improve physical fidelity, a raytracing implementation is still computationally slower than highly parallel splatting-based implementations. This increases reconstruction time and limits throughput for largescale experiments.

Future works. As a next step, we will extend the projection operator to explicitly model attenuation correction, scatter, and randoms within the ray-tracing pipeline, and to support Time-of-Flight (TOF) PET [45]. By incorporating these effects in a differentiable way, we expect this to move the method toward fully quantitative PET reconstruction [24]. We will also investigate practical acceleration strategies, e.g., better BVH construction and mixed-precision computation, to reduce run time [6].

Moreover, recent works show that integration with diffusion models or learned priors can further improve reconstruction quality [26, 32, 51]. It would be interesting to explore combining diffusion-based regularization or generative priors with 3D Gaussian ray tracing to improve noise suppression and perceptual quality while keeping quantitative fidelity.

## 6. Conclusion

In this study, we propose a 3D Gaussian ray tracing framework for tomographic reconstruction. Our method avoids the local affine approximation used in splatting-based approaches by evaluating an analytic 3D Gaussian line integral along each ray. It is also adaptable to different scanner geometries, enabling PET reconstruction without resampling the sinogram and thus without compromising the original count data. Owing to the direct analytic integral and more accurate geometric modeling, our method demonstrates improved quantitative accuracy in PET imaging.

## References

[1] Kingma DP Ba J Adam et al. A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 1412(6), 2014. 5

[2] Go Akamatsu, Katsuhiko Mitsumoto, Takafumi Taniguchi, Yuji Tsutsui, Shingo Baba, and Masayuki Sasaki. Influences of point-spread function and time-of-flight reconstructions on standardized uptake value of lymph node metastases in fdg-pet. European journal of radiology, 83(1):226â230, 2014. 6

[3] Dale L Bailey, Michael N Maisey, David W Townsend, and Peter E Valk. Positron emission tomography. Springer, 2005. 1

[4] Qinan Bao, Danny Newport, Mu Chen, David B Stout, and Arion F Chatziioannou. Performance evaluation of the inveon dedicated pet preclinical tomograph based on the nema nu-4 standards. Journal of Nuclear Medicine, 50(3):401â 408, 2009. 2

[5] Rachel Bar-Shalom, Ana Y Valdivia, and M Donald Blaufox. Pet imaging in oncology. In Seminars in nuclear medicine, pages 150â185. Elsevier, 2000. 1

[6] Louis Bavoil, Steven P Callahan, Aaron Lefohn, Joao LD Comba, and Claudio T Silva. Multi-fragment effects on theÂ´ gpu using the k-buffer. In Proceedings of the 2007 symposium on Interactive 3D graphics and games, pages 97â104, 2007. 8

[7] Ander Biguri, Manjit Dosanjh, Steven Hancock, and Manuchehr Soleimani. Tigre: a matlab-gpu toolbox for cbct image reconstruction. Biomedical Physics & Engineering Express, 2(5):055010, 2016. 5

[8] E Broussolle, C Dentresangle, P Landais, L Garcia-Larrea, P Pollak, B Croisile, O Hibert, F Bonnefoi, G Galy, JC Froment, et al. The relation of putamen and caudate nucleus 18fdopa uptake to motor and cognitive performances in parkinsonâs disease. Journal of the neurological sciences, 166(2): 141â151, 1999. 7

[9] Ralph Buchert, Karl H Bohuslavizki, Harald Fricke, Janos Mester, and Malte Clausen. Performance evaluation of pet scanners: testing of geometric arc correction by offcentre uniformity measurement. European journal of nuclear medicine, 27(1):83â90, 2000. 2, 6

[10] Yuanhao Cai, Jiahao Wang, Alan Yuille, Zongwei Zhou, and Angtian Wang. Structure-aware sparse-view x-ray 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11174â 11183, 2024. 1

[11] Chi-Yang Chu, Daniel J Henderson, and Christopher F Parmeter. On discrete epanechnikov kernel functions. Computational statistics & data analysis, 116:79â105, 2017. 3

[12] Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic, Simon Green, Piotr Didyk, and Adrian Jarabo. Donât splat your gaussians: Volumetric ray-traced primitives for modeling and rendering scattering and emissive media. ACM Transactions on Graphics, 44(1):1â17, 2025. 2, 3

[13] Michel Defrise and Paul Kinahan. Data acquisition and image reconstruction for 3d pet. In The theory and practice of 3D PET, pages 11â53. Springer, 1998. 3

[14] Lee A Feldkamp, Lloyd C Davis, and James W Kress. Practical cone-beam algorithm. Journal of the Optical Society of America A, 1(6):612â619, 1984. 1, 5

[15] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. Relightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing. In European Conference on Computer Vision, pages 73â89. Springer, 2024. 2

[16] Kuang Gong, Simon R Cherry, and Jinyi Qi. On the assessment of spatial resolution of pet systems with iterative image reconstruction. Physics in Medicine & Biology, 61(5):N193, 2016. 6 6

[17] Godfrey N Hounsfield. Computed medical imaging. Science, 210(4465):22â28, 1980. 1

[18] H Malcolm Hudson and Richard S Larkin. Accelerated image reconstruction using ordered subsets of projection data. IEEE transactions on medical imaging, 13(4):601â609, 1994. 1

[19] Dan J Kadrmas. Lor-osem: statistical pet reconstruction from raw line-of-response histograms. Physics in Medicine & Biology, 49(20):4731, 2004. 6

[20] Avinash C Kak and Malcolm Slaney. Principles of computerized tomographic imaging. SIAM, 2001. 1

[21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1

[22] J Koivunen, A Verkkoniemi, S Aalto, Anders Paetau, J-P Ahonen, M Viitanen, K Nagren, J Rokka, M Haaparanta, Ë Hannu Kalimo, et al. Pet amyloid ligand [11c] pib uptake shows predominantly striatal increase in variant alzheimerâs disease. Brain, 131(7):1845â1853, 2008. 7

[23] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Transactions on Graphics (ToG), 39(6):1â14, 2020. 1

[24] Adriaan A Lammertsma. Forward to the past: the case for quantitative pet imaging. Journal of Nuclear Medicine, 58 (7):1019â1024, 2017. 8

[25] Robert M Lewitt, Gerd Muehllehner, and Joel S Karp. Threedimensional image reconstruction for pet by multi-slice rebinning and axial image filtering. Physics in Medicine & Biology, 39(3):321, 1994. 3

[26] Xi Liu, Chaoyi Zhou, and Siyu Huang. 3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with viewconsistent 2d diffusion priors. Advances in Neural Information Processing Systems, 37:133305â133327, 2024. 8

[27] Yuxuan Long, Yulin Zhang, Hong Wang, Xiaodong Kuang, Hailiang Huang, Fan Rao, Huafeng Liu, Yefeng Zheng, and Wentao Zhu. PD-INR: Prior-Driven Implicit Neural Representations for TOF-PET Reconstruction . In proceedings of Medical Image Computing and Computer Assisted Intervention â MICCAI 2025. Springer Nature Switzerland, 2025. 1

[28] Benjamin P Lopez, David W Jordan, Brad J Kemp, Paul E Kinahan, Charles R Schmidtlein, and Osama R Mawlawi. Pet/ct acceptance testing and quality assurance: Executive

summary of aapm task group 126 report. Medical physics, 48(2):e31âe35, 2021. 5

[29] Daniel Meister, Shinji Ogaki, Carsten Benthin, Michael J Doyle, Michael Guthe, and JiËrÂ´Ä± Bittner. A survey on bounding volume hierarchies for ray tracing. In Computer Graphics Forum, pages 683â712. Wiley Online Library, 2021. 2

[30] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1

[31] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray tracing: Fast tracing of particle scenes. ACM Transactions on Graphics (TOG), 43(6):1â19, 2024. 2

[32] Yuxuan Mu, Xinxin Zuo, Chuan Guo, Yilin Wang, Juwei Lu, Xiaofeng Wu, Songcen Xu, Peng Dai, Youliang Yan, and Li Cheng. Gsd: View-guided gaussian splatting diffusion for 3d reconstruction. In European Conference on Computer Vision, pages 55â72. Springer, 2024. 8

[33] Steven G Parker, James Bigler, Andreas Dietrich, Heiko Friedrich, Jared Hoberock, David Luebke, David McAllister, Morgan McGuire, Keith Morley, Austin Robison, et al. Optix: a general purpose ray tracing engine. Acm transactions on graphics (tog), 29(4):1â13, 2010. 5

[34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 5

[35] E Rapisarda, V Bettinardi, Kris Thielemans, and MC Gilardi. Image-based point spread function implementation in a fully 3d osem reconstruction algorithm for pet. Physics in medicine & biology, 55(14):4131, 2010. 1

[36] Simon Rit, David Sarrut, and Laurent Desbat. Comparison of analytic and algebraic methods for motion-compensated cone-beam ct reconstruction of the thorax. IEEE transactions on medical imaging, 28(10):1513â1525, 2009. 2

[37] D Strulab, Giovanni Santin, Delphine Lazaro, Vincent Breton, and Christian Morel. Gate (geant4 application for tomographic emission): a pet/spect general-purpose simulation platform. Nuclear Physics B-Proceedings Supplements, 125:75â79, 2003. 5

[38] Chinmay Talegaonkar, Yash Belhe, Ravi Ramamoorthi, and Nicholas Antipa. Volumetrically consistent 3d gaussian rasterization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10953â10963, 2025. 1

[39] Joseph A Thie. Understanding the standardized uptake value, its methods, and implications for usage. Journal of Nuclear Medicine, 45(9):1431â1434, 2004. 4

[40] Kris Thielemans, Charalampos Tsoumpas, Sanida Mustafovic, Tobias Beisel, Pablo Aguiar, Nikolaos Dikaios, and Matthew W Jacobson. Stir: software for tomographic image reconstruction release 2. Physics in Medicine & Biology, 57(4):867, 2012. 5

[41] Christopher Thirgood, Oscar Mendez, Erin Ling, Jon Storey, and Simon Hadfield. Hypergs: Hyperspectral 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5970â5979, 2025. 1

[42] DW Townsend et al. Physical principles and technology of clinical pet imaging. Annals-Academy of Medicine Singapore, 33(2):133â145, 2004. 1

[43] Timothy G Turkington. Introduction to pet instrumentation. Journal of nuclear medicine technology, 29(1):4â11, 2001. 2

[44] Stefaan Vandenberghe, YVES DâAsseler, Rik Van de Walle, T Kauppinen, Michel Koole, Luc Bouwens, Koen Van Laere, Ignace Lemahieu, and RA Dierckx. Iterative reconstruction algorithms in nuclear medicine. Computerized medical imaging and graphics, 25(2):105â111, 2001. 2

[45] Stefaan Vandenberghe, Ekaterina Mikhaylova, Ester DâHoe, Pieter Mollet, and Joel S Karp. Recent developments in timeof-flight pet. EJNMMI physics, 3(1):3, 2016. 8

[46] Philip J Withers, Charles Bouman, Simone Carmignato, Veerle Cnudde, David Grimaldi, Charlotte K Hagen, Eric Maire, Marena Manley, Anton Du Plessis, and Stuart R Stock. X-ray computed tomography. Nature Reviews Methods Primers, 1(1):18, 2021. 1

[47] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024. 1

[48] Ruyi Zha, Yanhao Zhang, and Hongdong Li. Naf: neural attenuation fields for sparse-view cbct reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 442â452. Springer, 2022. 1

[49] Ruyi Zha, Tao Jun Lin, Yuanhao Cai, Jiwen Cao, Yanhao Zhang, and Hongdong Li. R2-gaussian: Rectifying radiative gaussian splatting for tomographic reconstruction. arXiv preprint arXiv:2405.20693, 2024. 1, 2, 5

[50] Chenxu Zhou, Lvchang Fu, Sida Peng, Yunzhi Yan, Zhanhua Zhang, Yong Chen, Jiazhi Xia, and Xiaowei Zhou. Lidar-rt: Gaussian-based ray tracing for dynamic lidar re-simulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1538â1548, 2025. 2

[51] Junsheng Zhou, Weiqi Zhang, and Yu-Shen Liu. Diffgs: Functional gaussian splatting diffusion. Advances in Neural Information Processing Systems, 37:37535â37560, 2024. 8

[52] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Surface splatting. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pages 371â378, 2001. 1

[53] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Ewa splatting. IEEE Transactions on Visualization and Computer Graphics, 8(3):223â238, 2002. 1