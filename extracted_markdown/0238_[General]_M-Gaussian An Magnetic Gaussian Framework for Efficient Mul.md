<!-- page 1 -->
M-GAUSSIAN: AN MAGNETIC GAUSSIAN FRAMEWORK FOR
EFFICIENT MULTI-STACK MRI RECONSTRUCTION ∗†‡§
Kangyuan Zheng, Xuan Cai, Jiangqi Wang, Guixing Fu, Zhuoshuo Li,
Yazhou Chen, Xinting Ge, Liangqiong Qu, Mengting Liu
ABSTRACT
Magnetic Resonance Imaging (MRI) is a crucial non-invasive imaging modality. In routine clinical
practice, multi-stack thick-slice acquisitions are widely used to reduce scan time and motion sensitiv-
ity, particularly in challenging scenarios such as fetal brain imaging. However, the resulting severe
through-plane anisotropy compromises volumetric analysis and downstream quantitative assessment,
necessitating robust reconstruction of isotropic high-resolution volumes. Implicit neural representa-
tion methods, while achieving high quality, suffer from computational inefficiency due to complex
network structures. We present M-Gaussian, adapting 3D Gaussian Splatting to MRI reconstruction.
Our contributions include: (1) Magnetic Gaussian primitives with physics-consistent volumetric
rendering, (2) neural residual field for high-frequency detail refinement, and (3) multi-resolution
progressive training. Our method achieves an optimal balance between quality and speed. On the
FeTA dataset, M-Gaussian achieves 40.31 dB PSNR while being 14 times faster, representing the
first successful adaptation of 3D Gaussian Splatting to multi-stack MRI reconstruction.
Keywords 3D Gaussian Splatting, implicit neural representation, medical imaging, MRI, slice-to-volume reconstruction
1
Introduction
Magnetic Resonance Imaging (MRI) stands as one of the most important non-invasive imaging modalities in modern
medicine, providing exceptional soft tissue contrast and multi-parametric information crucial for diagnosis and treatment
planning [1]. Ideally, MRI would yield high-resolution isotropic 3D volumes with uniform spatial resolution in all
dimensions, enabling arbitrary multi-planar reformatting and precise volumetric measurements. However, fundamental
physical constraints create an unavoidable trade-off between spatial resolution, signal-to-noise ratio (SNR), and scan
time [2]. Direct acquisition of isotropic high-resolution volumes requires prohibitively long scan times, making it
impractical for many clinical scenarios, particularly fetal imaging where unpredictable motion demands ultra-fast
acquisition protocols, and adult brain imaging where patient comfort and clinical workflow efficiency are essential.
Consequently, clinical practice has adopted a pragmatic compromise: acquiring multiple stacks of thick 2D slices from
different orientations [3]. This multi-stack acquisition strategy significantly reduces scan time while maintaining high
in-plane resolution, but introduces severe anisotropy with slice thickness often 3-5 times larger than in-plane resolution.
The resulting data suffers from partial volume effects [4], inter-slice gaps, and potential misalignment due to patient
motion [5]. These artifacts significantly compromise quantitative analysis capabilities, making the reconstruction of
high-quality isotropic volumes from such anisotropic multi-stack data a critical problem in medical imaging.
∗(Corresponding authors: Mengting Liu and Liangqiong Qu.)
†K. Zheng, X. Cai, J. Wang, G. Fu, Z. Li and M. Liu are with School of Biomedical Engineering, Shenzhen Campus of Sun
Yat-sen University, Shenzhen, 518107, China. (e-mail: {zhengky29, caix63, wangjq236, fugx3}@mail2.sysu.edu.cn; {lizhsh36,
liumt55}@mail.sysu.edu.cn).
‡Y. Chen and X. Ge are with the School of Information Science and Engineering, Shandong Normal University, Shandong, China.
(e-mail: yachauchen@gmail.com; xintingge@sdnu.edu.cn).
§L. Qu is with the School of Computing and Data Science, The University of Hong Kong, Hong Kong, China. (e-mail:
liangqqu@hku.hk).
arXiv:2603.00145v1  [cs.CV]  24 Feb 2026

<!-- page 2 -->
Zheng et al.: M-Gaussian
Figure 1: Overview of the M-Gaussian framework. (a) The training pipeline integrates Gaussian rendering with a
Neural Residual Field (NRF) refinement module. (b) Point cloud construction is performed via multi-stack registration
and devoxelization to initialize the Gaussians. (c) The inference stage generates the final volume through Gaussian
sampling and aggregation at the target resolution.
Traditional slice-to-volume reconstruction methods [6] have approached this problem through iterative optimization
frameworks with explicit regularization terms. While achieving clinical success, these approaches are computationally
expensive and scale poorly for high-resolution reconstruction.
The emergence of implicit neural representation (INR) methods has introduced a paradigm shift [7]. By parameterizing
the volume as a continuous function through coordinate-based neural networks, INR methods offer resolution-agnostic
reconstruction and the ability to learn complex tissue priors directly from data. However, INR relies on coordinate-based
multi-layer perceptrons (MLPs), creating a computational bottleneck where each spatial query necessitates a complete
forward pass through the network, resulting in inherently long training and inference times.
In contrast, recent advances in 3D Gaussian Splatting (3DGS) [8] have demonstrated that explicit primitive-based
representations can substantially accelerate volumetric reconstruction without sacrificing quality. By representing scenes
as collections of anisotropic 3D Gaussians with learnable parameters, 3DGS enables efficient rendering through direct
primitive evaluation and tile-based rasterization rather than substantial network queries. The remarkable success of
3DGS in computer vision—from novel view synthesis to dynamic scene reconstruction—strongly motivates exploring
similar benefits for MRI reconstruction from thick-slice acquisitions.
However, adapting 3DGS to MRI presents fundamental challenges. The imaging physics of MRI fundamentally differs
from optical imaging: MRI signals represent volumetric tissue properties rather than surface reflectance, requiring
complete redesign of the Gaussian properties and rendering pipeline. Furthermore, the volumetric nature of MRI
reconstruction requires evaluating millions of 3D points rather than projecting onto 2D image planes, making the
projection-based rendering of standard 3DGS computationally prohibitive.
We present MRI-tailored 3D Gaussian (M-Gaussian), the first successful adaptation of 3DGS for MRI reconstruction
from multi-stack thick-slice acquisitions. Our method addresses these fundamental challenges through a comprehensive
2

<!-- page 3 -->
Zheng et al.: M-Gaussian
framework combining the efficiency of explicit Gaussian representations with domain-specific innovations tailored for
MRI. Our key contributions include:
• Magnetic Gaussian primitives representing tissue-specific signal intensities with a volumetric rendering
pipeline consistent with MRI physics.
• Block-based spatial partitioning that restricts Gaussian evaluation to local neighborhoods for efficient querying.
• A neural residual field that complements the smooth Gaussian representation by capturing high-frequency
anatomical details.
• A multi-resolution progressive training strategy that ensures stable convergence for high-resolution reconstruc-
tion.
2
Related Work
2.1
Multi-Stack MRI Reconstruction
Reconstructing isotropic volumes from anisotropic multi-stack MRI data has been extensively studied. Early ap-
proaches [9] pioneered registration-based techniques using scattered data interpolation, establishing foundational
methodologies. While [10] formulated the problem as robust super-resolution with M-estimation for outlier handling.
Building upon these, [11] introduced complete outlier removal using robust statistics based on expectation maximization
with intensity matching for bias field correction. Subsequently, [3] proposed multi-level B-spline interpolation, and [6]
developed an automated framework integrating brain localization, segmentation, and slice-level outlier rejection.
Recent advances have incorporated GPU acceleration [12] and total variation regularization [13] to enhance computa-
tional efficiency and reconstruction quality. Nevertheless, these explicit approaches remain constrained by iterative
optimization schemes, exhibiting computational complexity that scales poorly with increasing output resolution.
2.2
Neural Representations in Medical Imaging
Neural Radiance Fields have emerged as a powerful paradigm for 3D reconstruction and rendering. Recent works have
successfully adapted Neural Radiance Fields [14] to various medical imaging modalities: MedNeRF [15] for X-ray
imaging and Ultra-NeRF [16] for ultrasound imaging. In MRI imaging, NeSVoR [17] demonstrates the potential of
INR for slice-to-volume reconstruction in fetal MRI, achieving resolution-agnostic reconstruction. IREM [18] similarly
applied INR to super-resolution reconstruction of adult brain MRI.
While achieving resolution-agnostic reconstruction, these coordinate-based MLP approaches require numerous network
evaluations, resulting in inherently slow training and inference without dedicated implementation optimizations.
2.3
3D Gaussian Splatting
3D Gaussian Splatting represents scenes as collections of 3D Gaussian primitives, enabling real-time rendering through
efficient rasterization while achieving superior reconstruction quality with significantly faster training compared to neural
radiance fields. The method has been successfully extended to various domains including dynamic scenes [19, 20, 21]
and surface reconstruction [22, 23].
More recently, researchers have begun exploring 3DGS applications in medical imaging. Several works have adapted
the framework for X-ray-based imaging: X-Gaussian [24] for X-ray novel view synthesis, R2-Gaussian [25] and
3DGR-CT [26] for tomographic reconstruction, and x2-Gaussian [27] for continuous-time tomographic reconstruction.
However, these methods are designed for X-ray imaging with its specific projection model and do not address the
unique challenges of MRI thick-slice acquisition and soft tissue contrast.
3
Method
3.1
Overall Pipeline
Fig. 1 depicts our training and inference pipeline. We construct a unified point cloud by registering input slice stacks
into a common RAS (Right-Anterior-Superior) anatomical space and sampling foreground pixels. Each sample consists
of a 3D coordinate normalized to the canonical [−1, 1]3 space and its corresponding intensity value.
We initialize a uniform grid of 3D Gaussians at low resolution and train through differentiable rendering. The model is
optimized using the Adam optimizer with point-wise, structural, and regularization losses. To capture fine anatomical
3

<!-- page 4 -->
Zheng et al.: M-Gaussian
details beyond the Gaussian representation’s capacity, a lightweight Neural Residual Field is incorporated in subsequent
training stages.
Upon convergence, the learned continuous representation—combining both Gaussian primitives and neural residuals—is
sampled at target voxel coordinates to produce the final high-resolution volume.
3.2
Magnetic 3D Gaussian Representation
MRI signal acquisition fundamentally differs from optical imaging in both physics and data characteristics. As shown
in Fig. 2, while optical imaging captures surface reflectance that varies with viewing direction, MRI measures intrinsic
tissue properties that remain constant regardless of the imaging plane orientation.
The original 3DGS framework uses spherical harmonics (SH) [28] to model view-dependent RGB color c,
c(v) =
L
X
l=0
l
X
m=−l
klmYlm(v)
(1)
where v is the view direction, L is the maximum degree of SH coefficients, Ylm are the SH basis functions and klm are
the learnable SH coefficients, requiring 3(L + 1)2 coefficients per primitive. While this is essential for optical rendering,
it introduces substantial memory overhead and computational complexity when applied to MRI reconstruction where
RGB color and view-dependency are unnecessary. Maintaining view-dependent appearance parameters not only
increases memory consumption but also complicates optimization without providing commensurate benefits.
Consequently, we eliminate view-dependent color representation entirely, replacing it with a single scalar intensity
value that directly models tissue-specific MRI signal properties. We propose Magnetic Gaussians, where each primitive
Gi is parameterized by:
Gi = {µi, Σi, αi}
(2)
where µi ∈R3 represents the spatial center in normalized coordinates, Σi ∈R3×3 is the covariance matrix encoding
the shape and orientation, and αi ∈[0, 1] is the normalized MRI signal intensity. By replacing view-dependent spherical
harmonics with a single intensity value, our parameterization reduces per-primitive parameters from 59 to 11, achieving
5.4 times memory reduction, which is critical for high-resolution brain imaging where millions of primitives are
required.
To ensure numerical stability and efficient gradient-based optimization, we employ a factored representation:
Σi = RiSiST
i RT
i
(3)
where Ri ∈SO(3) is the rotation matrix and Si = diag(exp(si)) ∈R3×3 is the diagonal scale matrix. The rotation
matrix Ri is parameterized using unit quaternions qi ∈R4 with ∥qi∥= 1, providing a singularity-free representation
for 3D rotations. During training, we optimize unnormalized quaternions and apply normalization before converting to
rotation matrices. For the scale matrix, we use learnable log-scale parameters si = [sx
i , sy
i , sz
i ]T ∈R3 with exponential
mapping Si = diag(exp(si)). This ensures positive scale values while providing numerically stable gradients.
3.3
MRI Volume Rendering with Spatial Query
We designed an efficient rendering pipeline for the proposed M-Gaussian representation. The volumetric nature of MRI
reconstruction demands volumetric sampling at arbitrary 3D locations throughout the imaging volume. Direct evaluation
of all NG Gaussians for each query point would result in O(NG × Nvoxels) complexity, becoming computationally
prohibitive for high-resolution reconstruction where both terms can reach millions. Our approach is a localized query
mechanism whose design is motivated by the physical locality of MRI signals and justified by the mathematically
compact support of Gaussian primitives.
3.3.1
Block-based Spatial Partitioning for Efficient Query
We introduce a spatial partitioning scheme to accelerate query operations. The normalized volume [−1, 1]3 is partitioned
into a uniform grid based on a given grid_resolution. Each Gaussian Gi is assigned to a grid cell according to its
center µi:
cell_index(µi) =

(µi + 1) · grid_resolution
2

(4)
For any query point x, its corresponding cell index, (i0, j0, k0), is first determined. A local neighborhood of cells,
Ncells(x), is then defined within a search radius of block_radius:
Ncells(x) = {celli,j,k | |i −i0|, |j −j0|, |k −k0| ≤block_radius}
(5)
4

<!-- page 5 -->
Zheng et al.: M-Gaussian
Figure 2: Comparison between original 3D Gaussian Splatting primitives and the proposed Magnetic Gaussian
primitives. The original 3DGS utilizes view-dependent spherical harmonic coefficients for color representation, whereas
M-Gaussian employs a single intensity value to model tissue-specific MRI signal properties, which significantly reduces
memory overhead.
The active set of Gaussians for rendering, Glocal, is subsequently defined as all primitives residing within this cell
neighborhood:
Glocal(x) = {Gi | cell_index(µi) ∈Ncells(x)}
(6)
This partitioning strategy provides fundamental computational advantages over projection-based methods like original
3DGS. While 3DGS requires complex splatting operations involving projection, sorting, and tile-based rasterization, our
approach achieves O(1) search time for finding relevant Gaussians through grid-based indexing. Our method eliminates
view-dependent computations entirely—traditional 3DGS must re-project and re-sort Gaussians for each viewing angle,
whereas our spatial partitioning remains constant regardless of query pattern. This is particularly advantageous for
volumetric reconstruction tasks requiring dense 3D space sampling rather than specific 2D view rendering.
3.3.2
MRI Volumetric Rendering
Unlike optical imaging where radiance is accumulated along viewing rays, MRI signal generation follows fundamentally
different physics. Each voxel in an MRI volume represents the aggregate magnetic resonance signal from tissue within
that spatial location. This distinction necessitates a specialized rendering approach that respects the volumetric nature
of MRI data while maintaining computational efficiency.
For each sampled point xsample from slice k, we apply the corresponding rigid transformation to obtain x = Tk(xsample)
in the reconstructed volume space. These transformations account for inter-slice motion and misalignment inherent in
multi-stack acquisitions. Signal intensity is computed by aggregating local Gaussian contributions:
I(x) =
X
i∈Glocal(x)
αi · exp

−1
2(x −µi)T Σ−1
i (x −µi)

(7)
The rigid transformations {Tk} are jointly optimized with Gaussian parameters during training to achieve accurate slice
alignment and volume reconstruction. This formulation differs fundamentally from alpha-compositing in NeRF or
projection-based rendering in 3DGS. Instead of accumulating along rays or projecting onto planes, we directly evaluate
the 3D Gaussian mixture at each query point, aligning with MRI’s physical signal formation process. The inherent
smoothness of Gaussian basis functions naturally models the Point Spread Function characteristic of MRI acquisition,
capturing gradual signal transitions without explicit PSF modeling. Our approach also naturally handles partial volume
effects critical in thick-slice reconstruction—when voxels contain multiple tissue types, overlapping Gaussians with
different intensities αi provide principled mixed signal representation, particularly important at tissue boundaries where
traditional methods struggle with smooth transitions.
3.4
Neural Residual Field Enhancement
Anatomical boundaries between tissues and fine-scale structures exhibit sharp intensity transitions that challenge the
inherently smooth Gaussian representation. To address this limitation, we augment the base Gaussian representation
5

<!-- page 6 -->
Zheng et al.: M-Gaussian
Table 1: Quantitative comparison with baseline slice-to-volume MRI reconstruction methods across three datasets.
Metrics include PSNR, SSIM, NCC, and NRMSE for reconstruction quality, and runtime for computational efficiency.
Best results are in bold, second-best are underlined.
Methods
PSNR (dB) ↑
SSIM ↑
NCC ↑
NRMSE ↓
Runtime (min) ↓
FeTA Dataset
NiftyMIC [6]
36.08
0.9857
0.9932
0.0157
79.12
SVRTK [11]
34.64
0.9861
0.9916
0.0185
14.75
NeSVoR [17]
31.35
0.9588
0.9546
0.0414
31.75
M-Gaussian (Ours)
40.31
0.9936
0.9975
0.0096
5.63
FaBiAN Dataset
NiftyMIC [6]
32.09
0.9673
0.9666
0.0249
66.15
SVRTK [11]
31.38
0.9332
0.9497
0.0270
12.38
NeSVoR [17]
30.15
0.9445
0.9622
0.0311
30.42
M-Gaussian (Ours)
32.26
0.9521
0.9668
0.0244
4.78
HCP Dataset
NiftyMIC [6]
33.34
0.9782
0.9917
0.0215
1353.48
SVRTK [11]
32.33
0.9699
0.9869
0.0242
42.52
NeSVoR [17]
32.79
0.9751
0.9876
0.0229
64.88
M-Gaussian (Ours)
33.10
0.9776
0.9970
0.0221
17.28
with a Neural Residual Field (NRF)—a lightweight MLP that captures high-frequency details beyond the capacity of
smooth basis functions.
The NRF employs Fourier positional encoding with 6 frequency bands to map input coordinates x to higher-dimensional
features, enabling the network to learn fine spatial patterns. The architecture consists of 4 hidden layers with 64 neurons
each, using SiLU activation functions. The output is bounded through scaled tanh to [−0.1, 0.1]:
Ifinal(x) = I(x) + r(x)
(8)
where r(x) is the NRF-predicted residual. This bounding ensures residual corrections remain small relative to base
Gaussian intensities, preventing overfitting to noise.
NRF is activated after initial Gaussian convergence through a delayed activation strategy. This allows Gaussians to first
capture coarse volumetric structures before NRF refines high-frequency details. The delayed activation is critical to
prevent the flexible MLP from fitting noise before a robust base representation is established.
3.5
Training
3.5.1
Multi-Resolution Progressive Training
We employ a coarse-to-fine training strategy where the Gaussian grid resolution increases progressively at predefined
iteration milestones. During resolution transitions, Gaussian parameters are interpolated from the previous grid
using trilinear interpolation for intensity and scale, and normalized linear interpolation for rotation quaternions.
This progressive densification provides strong initialization from coarse features and accelerates convergence toward
fine-grained anatomical details.
3.5.2
Loss Function
Our training objective comprises reconstruction and regularization terms:
L = Lrecon + Lreg
(9)
The reconstruction loss ensures data fidelity through smooth L1 loss for robust point-wise accuracy and SSIM loss for
perceptual quality preservation:
Lrecon = LL1 + λSSIMLSSIM
(10)
We adopt the smooth L1 loss, also known as the Huber loss, which combines the stability of L2 loss near zero with the
robustness of L1 loss for larger errors:
smoothL1(x) =
0.5x2
if |x| < 1
|x| −0.5
otherwise
(11)
6

<!-- page 7 -->
Zheng et al.: M-Gaussian
Figure 3: Qualitative comparison of slice-to-volume MRI reconstruction methods. Each row shows results from a
different dataset (FeTA, FaBiAN, HCP). The Input column displays one representative stack from the multi-stack
acquisition. Our M-Gaussian method produces reconstructions with sharper anatomical boundaries and fewer artifacts
compared to baseline methods.
where x = Ipred(x) −Igt(x) denotes the difference between predicted and ground truth intensities. This formulation
provides quadratic smoothing for small residuals while maintaining linear growth for outliers, making optimization less
sensitive to noise.
The regularization loss constrains model behavior:
Lreg = λanisoLaniso
(12)
Anisotropic regularization prevents excessive Gaussian elongation:
Laniso =
1
NG
X
i
max

0, max(si)
min(si) −λr

(13)
where λr controls the permissible degree of anisotropy, balancing expressive flexibility with numerical stability.
4
Experiments
4.1
Experimental Settings
4.1.1
Datasets
We evaluate our method on three datasets:
• FeTA [29]: Fetal brain MRI volumes with 0.5 mm isotropic resolution. We simulate clinical thick-slice
acquisitions by downsampling to 0.8 × 0.8 × 3 mm3 with added motion artifacts.
• FaBiAN [30]: Synthetic fetal brain dataset generated at 1.1 × 1.1 × 3 mm3 resolution with k-space noise and
stochastic inter-slice motion. Controlled ground truth is provided for quantitative evaluation.
• HCP [31]: Adult brain MRI from the Human Connectome Project at 0.7 mm isotropic resolution, downsampled
to 1.0 × 1.0 × 2 mm3 to simulate clinical acquisitions.
For all datasets, three orthogonal stacks (axial, coronal, sagittal) are generated per volume to simulate the clinical
multi-stack acquisition protocol. The selection of these diverse datasets allows for a comprehensive evaluation across
different anatomical scales, age groups, and acquisition characteristics, ensuring the robustness of the proposed method
in various clinical scenarios.
7

<!-- page 8 -->
Zheng et al.: M-Gaussian
4.1.2
Baselines
We compare against three representative methods: SVRTK [11], a toolkit employing iterative reconstruction with robust
statistics; NiftyMIC [6], a comprehensive pipeline integrating motion correction, bias field estimation, and intensity
standardization; and NeSVoR [17], an implicit neural representation method. For fair comparison, NeSVoR is evaluated
using its pure PyTorch [32] implementation without custom CUDA kernels.
4.1.3
Evaluation Metrics
Reconstruction quality is assessed using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM),
Normalized Cross-Correlation (NCC), and Normalized Root Mean Square Error (NRMSE). Runtime (in minutes) is
also reported to evaluate computational efficiency.
4.1.4
Implementation Details
M-Gaussian is implemented in PyTorch. Gaussian parameters are optimized using Adam with learning rates of 0.001,
0.05, 0.005, and 0.001 for position, intensity, scale, and rotation, respectively. The Neural Residual Field employs 6
Fourier frequency bands with 4 hidden layers of 64 neurons each, optimized with Adam at a learning rate of 0.0001. The
NRF is activated at iteration 2000 for FeTA/FaBiAN and iteration 4000 for HCP. Loss weights are set to λSSIM = 0.5,
λaniso = 0.1, and λr = 1.5. The default block radius for spatial querying is set to 5.
Target resolutions are 0.5 mm (FeTA), 1.1 mm (FaBiAN), and 0.7 mm (HCP). Progressive training starts from 703
grids, increasing to 2003 for FeTA/FaBiAN at iterations 500, 1000, 2000, and 3000. HCP follows an extended
schedule reaching 2803 at iteration 6000 to capture finer anatomical structures. Total training iterations are 4000 for
FeTA/FaBiAN and 8000 for HCP. All experiments are conducted on a system with 2 AMD EPYC 7352 CPUs and an
NVIDIA RTX A6000 GPU (48 GB memory).
4.2
Comparison with Baseline Methods
Table 1 presents quantitative comparisons across the three datasets, and Fig. 3 illustrates representative qualitative
results. Our method achieves an optimal balance between reconstruction quality and computational efficiency across all
evaluated scenarios.
On the FeTA dataset, M-Gaussian achieves a PSNR of 40.31 dB, outperforming the second-best method (NiftyMIC) by
a substantial margin of 4.23 dB. Notably, this quality improvement is accompanied by a 14× speedup in runtime (5.63
min vs. 79.12 min). The superior performance on fetal data demonstrates our method’s effectiveness in handling the
challenging scenario of small brain volumes with limited anatomical context.
The efficiency advantage becomes more pronounced on the high-resolution HCP dataset. M-Gaussian completes
reconstruction in just 17.28 minutes—a 78× acceleration compared to NiftyMIC (1353.48 min)—while maintaining
competitive accuracy with the highest NCC and second-best PSNR. This demonstrates the scalability of our approach
to high-resolution adult brain reconstruction.
On the FaBiAN dataset, M-Gaussian achieves the highest PSNR and NCC in under 5 minutes. While NiftyMIC achieves
slightly higher SSIM, our method provides a better trade-off considering the nearly 14× faster runtime.
Qualitatively, as shown in Fig. 3, M-Gaussian produces reconstructions with sharper tissue boundaries and more
coherent anatomical structures. The baseline methods exhibit varying degrees of blurring (NiftyMIC, SVRTK) or noise
(NeSVoR), particularly in regions with complex anatomy or near tissue interfaces.
4.3
Ablation Study
We conduct systematic ablation experiments to evaluate the contribution of each proposed component. Table 2 reports
quantitative results.
4.3.1
Progressive Resolution Training
We compare the proposed multi-resolution progressive training schedule against a baseline trained at fixed high
resolution. To isolate the effect of progressive resolution, both configurations are optimized without SSIM loss, as SSIM
evaluation requires sampling entire slices and would introduce confounding factors in analyzing convergence behavior.
As shown in Table 2, progressive training improves PSNR by 1.17 dB, 2.58 dB, and 1.20 dB on FeTA, FaBiAN, and
HCP respectively. Fig. 4 further demonstrates that progressive resolution training substantially accelerates convergence
8

<!-- page 9 -->
Zheng et al.: M-Gaussian
Table 2: Ablation study quantifying the contribution of structural similarity loss (SSIM), progressive resolution training
(P.R.), Neural Residual Field (NRF), and anisotropic regularization (A.R.).
Configuration
FeTA
FaBiAN
HCP
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
Full model
40.31
0.9936
32.26
0.9521
33.10
0.9776
w/o SSIM
32.24
0.9611
31.01
0.9388
32.56
0.9582
w/o SSIM & P.R.
31.07
0.9573
28.43
0.9272
31.36
0.9512
w/o NRF
39.11
0.9893
30.93
0.9387
31.96
0.9538
w/o A.R.
39.49
0.9902
30.02
0.9486
32.82
0.9606
and yields higher-quality reconstructions. Direct optimization of millions of primitives at full resolution leads to
unstable dynamics, whereas the coarse-to-fine strategy stabilizes training by gradually increasing representational
complexity.
4.3.2
Structural Similarity Loss
To assess the role of the SSIM loss, we contrast the full model with a variant trained using only L1 reconstruction loss.
As shown in Table 2, incorporating SSIM loss yields dramatic improvements: 8.07 dB on FeTA, 1.25 dB on FaBiAN,
and 0.54 dB on HCP. Fig. 4 demonstrates that SSIM loss enhances preservation of anatomical boundaries and tissue
contrast, maintaining clinically relevant morphology that pixel-wise losses alone cannot capture.
4.3.3
Neural Residual Field Enhancement
We evaluate the NRF contribution by comparing the full model against a variant without neural residual refinement.
Removing NRF results in PSNR drops of 1.20 dB (FeTA), 1.33 dB (FaBiAN), and 1.14 dB (HCP). Fig. 5 shows that
the variant without NRF exhibits increased noise, particularly near tissue boundaries. We attribute this to the inherent
smoothness of Gaussian basis functions, which struggle to model sharp intensity transitions. The NRF addresses this
limitation by providing a complementary representation well-suited for high-frequency patterns, enabling Gaussians to
focus on coarse-scale modeling while NRF refines fine details.
4.3.4
Anisotropic Regularization
Disabling anisotropic regularization during training results in PSNR degradation of 0.82 dB (FeTA), 2.24 dB (FaBiAN),
and 0.28 dB (HCP). As illustrated in Fig. 6, the absence of this regularizer leads to streaking and granular noise
artifacts. We attribute this to the emergence of needle-like Gaussians with extreme aspect ratios during optimization.
The regularization prevents such degenerate geometries by penalizing aspect ratios exceeding λr, ensuring Gaussians
maintain moderate anisotropy conducive to smooth volumetric reconstruction.
Figure 4: Qualitative ablation results on Progressive resolution (P.R.) training and SSIM loss. Progressive resolution
training stabilizes convergence, while the inclusion of SSIM loss significantly enhances the preservation of structural
integrity and contrast. The effect is particularly significant on FeTA dataset.
9

<!-- page 10 -->
Zheng et al.: M-Gaussian
Figure 5: Qualitative ablation results on Neural Residual Field (NRF). The NRF suppresses noise and refines high-
frequency details, which is prominent on the HCP dataset.
Figure 6: Qualitative ablation results on Anisotropic Regularization (A.R.). The absence of A.R. leads to needle-like
artifacts due to degenerate Gaussian geometries. These artifacts are effectively mitigated in the full model, which is
most pronounced on the FaBiAN dataset.
4.4
Hyperparameter Analysis
Beyond validating the necessity of each component, we investigate how key hyperparameters affect the quality-efficiency
trade-off.
4.4.1
Gaussian Grid Resolution
The Gaussian grid resolution directly determines the model’s representational capacity and computational requirements.
To enable direct comparison across different resolutions, we disable progressive training and train the model with fixed
resolutions ranging from 503 to 4003 on the HCP dataset. Resolutions beyond 4003 could not be evaluated due to GPU
memory constraints.
As shown in Table 3, reconstruction quality improves substantially as resolution increases from 503 to 1503, with PSNR
rising from 30.27 dB to 31.31 dB (+1.04 dB). At higher resolutions, however, the gains become marginal—from 1503 to
3503, PSNR improves by only 0.45 dB while training time increases by 65%. Notably, performance at 4003 (31.63 dB)
drops slightly below that of 3503 (31.76 dB), indicating that excessively high resolution without progressive training
may lead to optimization difficulties or overfitting.
These findings strongly motivate our progressive training strategy. Comparing with results in Table 1, our full model
with progressive training achieves 33.10 dB on HCP in 17.28 minutes—outperforming the best fixed-resolution result
(31.76 dB at 3503) by 1.34 dB while being 36% faster. The coarse-to-fine optimization path not only improves efficiency
but also enables better final reconstruction quality by providing stable gradient flow and strong initialization at each
resolution stage.
4.4.2
Effect of Block Radius
The block radius parameter controls the spatial extent of Gaussian contributions considered for each query point,
directly impacting the trade-off between computational efficiency and reconstruction accuracy. A smaller radius reduces
the number of Gaussians evaluated per query but may exclude relevant contributions, while a larger radius increases
computational cost and may cause over-smoothing by aggregating excessive Gaussian responses.
10

<!-- page 11 -->
Zheng et al.: M-Gaussian
Table 3: Impact of Gaussian grid resolution on reconstruction fidelity and training time using the HCP dataset. The
models were trained with fixed resolutions, excluding the effect of progressive training strategy.
Resolution
PSNR ↑
SSIM ↑
Time (min) ↓
503
30.27
0.9352
10.82
1503
31.31
0.9505
16.43
2503
31.50
0.9527
24.70
3503
31.76
0.9547
27.07
4003
31.63
0.9546
30.30
5
10
15
20
25
30
35
Run Time (minutes)
32.5
33.0
33.5
34.0
34.5
35.0
PSNR (dB)
BLock Radius: 3
BLock Radius: 4
BLock Radius: 5
BLock Radius: 6
BLock Radius: 7
Figure 7: Analysis of block radius on reconstruction qual-
ity and training time. A block radius of 5 achieves the
optimal balance, maximizing PSNR while maintaining
computational efficiency.
10
15
20
25
30
35
Run Time (minutes)
30
31
32
33
34
35
36
PSNR (dB)
Resolution: 50
Resolution: 150
Resolution: 250
Resolution: 350
Resolution: 400
Figure 8: Analysis of Gaussian grid resolution on recon-
struction quality and training time. While higher resolu-
tions improve quality, the performance plateaus beyond
3503.
Table 4 summarizes the results across block radius values from 3 to 7. Block radius 5 achieves the highest PSNR
(33.10 dB), surpassing both smaller and larger values. This non-monotonic behavior reflects the dual nature of the
radius parameter. When the radius is too small, relevant Gaussian contributions may be excluded, and the limited
number of Gaussians in the aggregation makes the rendered intensity more susceptible to fluctuations, introducing
noise. Conversely, when the radius is too large, distant Gaussians with negligible contributions are included, and the
excessive aggregation tends to over-smooth fine anatomical details.
The efficiency gap across configurations is also substantial: radius 3 completes training in 7.62 minutes while radius 7
requires 34.24 minutes—a 4.5× difference. Given that radius 5 achieves optimal quality while maintaining moderate
training time, we adopt it as the default setting. This choice balances reconstruction accuracy with computational
efficiency, making M-Gaussian practical for routine clinical deployment.
Table 4: Quantitative assessment of reconstruction performance under different block radius settings. The results
illustrate the trade-off between reconstruction quality and runtime.
Block Radius
PSNR ↑
SSIM ↑
Time (min) ↓
3
32.40
0.9733
7.62
4
32.67
0.9768
11.13
5
33.10
0.9776
17.28
6
32.77
0.9778
26.32
7
32.89
0.9780
34.24
4.5
Evaluation of Downstream Segmentation Tasks
To validate the clinical utility of the reconstructed volumes, we evaluated their performance in a downstream automated
brain segmentation task. While standard image quality metrics such as PSNR and SSIM quantify signal fidelity,
segmentation accuracy provides a more clinically meaningful assessment of the extent to which semantic anatomical
boundaries are preserved for subsequent analysis.
11

<!-- page 12 -->
Zheng et al.: M-Gaussian
Table 5: Quantitative evaluation of segmentation performance on the FeTA dataset. Metrics include Dice, HD95, and
ASSD. Best results are bold, second-best are underlined.
Structure
Dice ↑
HD95 (mm) ↓
ASSD (mm) ↓
NiftyMIC
SVRTK
NeSVoR
M-Gaussian
NiftyMIC
SVRTK
NeSVoR
M-Gaussian
NiftyMIC
SVRTK
NeSVoR
M-Gaussian
External CSF
0.859
0.883
0.888
0.905
1.43
1.42
1.10
0.97
0.55
0.53
0.46
0.41
Grey Matter
0.832
0.816
0.843
0.846
0.82
0.85
0.68
0.66
0.37
0.38
0.34
0.32
White Matter
0.945
0.937
0.951
0.953
0.95
1.11
0.80
0.69
0.39
0.46
0.36
0.34
Ventricles
0.914
0.818
0.906
0.917
0.64
1.23
0.65
0.63
0.28
0.52
0.30
0.28
Cerebellum
0.924
0.934
0.941
0.931
1.12
0.94
0.87
1.05
0.46
0.40
0.36
0.42
Deep GM
0.899
0.830
0.922
0.920
1.50
3.34
1.10
1.52
0.56
1.00
0.45
0.57
Brainstem
0.902
0.776
0.922
0.925
1.10
3.57
0.86
0.83
0.45
0.99
0.37
0.35
Mean
0.896
0.856
0.910
0.914
1.08
1.78
0.87
0.91
0.44
0.61
0.38
0.38
4.5.1
Experimental Protocol
We employed nnU-Net [33], a well-established deep learning framework for medical image segmentation widely
adopted as a benchmark in the field. Our evaluation follows a domain generalization protocol designed to rigorously
assess reconstruction quality: a 3D nnU-Net model is first trained exclusively on ground-truth isotropic high-resolution
volumes from the FeTA dataset, ensuring the segmentation network learns optimal anatomical features from artifact-free
reference data. This pre-trained model is then applied directly to volumes reconstructed by each method without any
fine-tuning. Since the segmentation network has not been exposed to reconstruction artifacts during training, any
performance degradation directly reflects geometric distortions, blurring, or boundary inconsistencies introduced by the
reconstruction process. Seven anatomically distinct structures are segmented: external cerebrospinal fluid (CSF), grey
matter, white matter, ventricles, cerebellum, deep grey matter (Deep GM), and brainstem.
4.5.2
Volumetric Overlap Analysis
Table 5 presents Dice score quantifying volumetric overlap between predicted and ground-truth segmentations. M-
Gaussian achieves the highest mean Dice, surpassing NeSVoR and substantially outperforming traditional methods
NiftyMIC and SVRTK. Performance analysis reveals distinct patterns correlating with anatomical scale. For large,
spatially extensive structures such as external CSF, white matter, and grey matter, M-Gaussian demonstrates particularly
strong performance, as these regions benefit from the Gaussian primitives’ capacity to efficiently model smooth
volumetric distributions. In contrast, for smaller, geometrically complex structures including cerebellum and deep grey
matter, the performance gap between M-Gaussian and NeSVoR narrows considerably, suggesting that implicit neural
representations may retain marginal advantages for highly localized structures requiring dense spatial sampling. Despite
these structure-specific variations, M-Gaussian maintains the highest overall mean Dice across all seven anatomical
regions.
4.5.3
Boundary Localization Accuracy
Beyond volumetric agreement, precise boundary localization is essential for clinical applications requiring geometric
measurements, such as cortical thickness analysis. Table 5 reports 95th percentile Hausdorff Distance (HD95) and
Average Symmetric Surface Distance (ASSD). M-Gaussian achieves the lowest boundary errors for the majority of
structures. Notably, for grey matter and white matter—regions characterized by highly convoluted cortical folding—our
method demonstrates notable improvements over NeSVoR in HD95. For external CSF, which demarcates the challenging
low-contrast interface between brain tissue and surrounding fluid, M-Gaussian also achieves considerably lower
HD95 compared to both NeSVoR and traditional methods. The superior boundary fidelity can be attributed to the
complementary architecture: while Gaussian primitives efficiently model smooth volumetric intensity distributions, the
Neural Residual Field captures high-frequency spatial variations localized at tissue boundaries.
Figure 9 provides qualitative validation across three orthogonal planes. The segmentation masks derived from M-
Gaussian reconstructions exhibit the closest visual correspondence with ground truth, particularly for large-scale
structures such as the cortical ribbon and ventricular system. For finer structures including the cerebellum and brainstem,
all learning-based methods achieve comparable delineation accuracy, consistent with the quantitative findings.
5
Discussion
The experimental results demonstrate that M-Gaussian achieves a favorable balance between reconstruction quality and
computational efficiency. Several aspects of our approach merit further discussion.
12

<!-- page 13 -->
Zheng et al.: M-Gaussian
Figure 9: Qualitative comparison of downstream segmentation results on the FeTA dataset. M-Gaussian reconstructions
yield segmentation masks with the closest correspondence to ground truth, particularly for large-scale structures such as
the cortical ribbon and ventricular boundaries.
The success of M-Gaussian can be attributed to the synergy between explicit Gaussian representations and neural
refinement. Gaussians efficiently capture smooth volumetric structures through direct evaluation, while the Neural
Residual Field complements this by modeling high-frequency details that challenge smooth Gaussian basis functions.
Block-based spatial partitioning reduces computational complexity from quadratic to near-linear, and progressive
training accelerates convergence through strong coarse-scale initialization.
The downstream segmentation evaluation provides additional validation beyond standard image quality metrics.
M-Gaussian achieves superior performance on large-scale anatomical structures while showing comparable results
to implicit methods on smaller, geometrically complex regions. Large structures with relatively smooth intensity
distributions align naturally with the Gaussian basis functions, whereas fine structures requiring dense local sampling
may benefit from the continuous querying capability of implicit representations. Nevertheless, the overall segmentation
accuracy demonstrates that M-Gaussian reconstructions preserve clinically relevant anatomical boundaries, supporting
its potential utility in automated analysis pipelines.
From a clinical workflow perspective, the substantial acceleration achieved by M-Gaussian addresses a critical bottleneck
in current reconstruction pipelines. Traditional optimization-based methods and recent implicit neural approaches often
require processing times measured in minutes to hours, limiting their applicability in time-sensitive clinical scenarios.
The reduced computational burden enables more practical integration into clinical workflows, potentially facilitating
real-time quality assessment during acquisition and rapid turnaround for diagnostic interpretation.
It is noteworthy that the current framework assumes rigid inter-slice motion, which may not hold in cases of severe
fetal movement or organ deformation. Future directions include extending the motion model to handle non-rigid
deformations and exploring joint optimization of acquisition and reconstruction protocols.
6
Conclusion
In this work, we introduced M-Gaussian, the first successful adaptation of 3D Gaussian Splatting for MRI slice-to-
volume reconstruction. By reformulating Gaussian primitives to model intrinsic tissue properties and developing a
volumetric rendering pipeline consistent with MRI physics, M-Gaussian successfully transformed 3DGS, originally
designed for optical imaging, into the field of MRI imaging. The proposed block-based spatial partitioning enables
efficient volumetric queries, while the Neural Residual Field captures fine anatomical details beyond the capacity of
13

<!-- page 14 -->
Zheng et al.: M-Gaussian
smooth Gaussian representations. Our multi-resolution progressive training strategy ensures stable convergence for
high-resolution reconstruction.
Experimental results on three diverse datasets demonstrate that M-Gaussian achieves an optimal balance between
reconstruction quality and runtime efficiency. On the FeTA dataset, our method achieves 40.31 dB PSNR while being
14× faster than competing methods. On the high-resolution HCP dataset, M-Gaussian provides 78× acceleration
compared to NiftyMIC while maintaining competitive accuracy. These results establish the potential of explicit
Gaussian-based representations as an efficient alternative for MRI image reconstruction, opening new avenues for
real-time clinical applications.
References
[1] V. Kuperman, Magnetic resonance imaging: physical principles and applications.
Elsevier, 2000.
[2] R. M. Heidemann, Ö. Özsarlak, P. M. Parizel, J. Michiels, B. Kiefer, V. Jellus, M. Müller, F. Breuer, M. Blaimer,
M. A. Griswold et al., “A brief review of parallel magnetic resonance imaging,” European radiology, vol. 13,
no. 10, pp. 2323–2337, 2003.
[3] S. Jiang, H. Xue, A. Glover, M. Rutherford, D. Rueckert, and J. V. Hajnal, “Mri of moving subjects using multislice
snapshot images with volume reconstruction (svr): application to fetal, neonatal, and adult brain studies,” IEEE
transactions on medical imaging, vol. 26, no. 7, pp. 967–980, 2007.
[4] J. Tohka, “Partial volume effect modeling for segmentation and tissue classification of brain magnetic resonance
images: A review,” World journal of radiology, vol. 6, no. 11, p. 855, 2014.
[5] K. Krupa and M. Bekiesi´nska-Figatowska, “Artifacts in magnetic resonance imaging,” Polish journal of radiology,
vol. 80, p. 93, 2015.
[6] M. Ebner, G. Wang, W. Li, M. Aertsen, P. A. Patel, R. Aughwane, A. Melbourne, T. Doel, S. Dymarkowski,
P. De Coppi et al., “An automated framework for localization, segmentation and super-resolution reconstruction of
fetal brain mri,” NeuroImage, vol. 206, p. 116324, 2020.
[7] X. Wang, Y. Chen, S. Hu, H. Fan, H. Zhu, and X. Li, “Neural radiance fields in medical imaging: A survey,” arXiv
e-prints, pp. arXiv–2402, 2024.
[8] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3d gaussian splatting for real-time radiance field
rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.
[9] F. Rousseau, O. A. Glenn, B. Iordanova, C. Rodriguez-Carranza, D. B. Vigneron, J. A. Barkovich, and
C. Studholme, “Registration-based approach for reconstruction of high-resolution in utero fetal mr brain images,”
Academic radiology, vol. 13, no. 9, pp. 1072–1081, 2006.
[10] A. Gholipour, J. A. Estroff, and S. K. Warfield, “Robust super-resolution volume reconstruction from slice
acquisitions: application to fetal brain mri,” IEEE transactions on medical imaging, vol. 29, no. 10, pp. 1739–
1758, 2010.
[11] M. Kuklisova-Murgasova, G. Quaghebeur, M. A. Rutherford, J. V. Hajnal, and J. A. Schnabel, “Reconstruction of
fetal brain mri with intensity matching and complete outlier removal,” Medical image analysis, vol. 16, no. 8, pp.
1550–1564, 2012.
[12] B. Kainz, M. Steinberger, W. Wein, M. Kuklisova-Murgasova, C. Malamateniou, K. Keraudren, T. Torsney-Weir,
M. Rutherford, P. Aljabar, J. V. Hajnal et al., “Fast volume reconstruction from motion corrupted stacks of 2d
slices,” IEEE transactions on medical imaging, vol. 34, no. 9, pp. 1901–1913, 2015.
[13] S. Tourbier, X. Bresson, P. Hagmann, J.-P. Thiran, R. Meuli, and M. B. Cuadra, “An efficient total variation
algorithm for super-resolution in fetal brain mri with adaptive regularization,” NeuroImage, vol. 118, pp. 584–597,
2015.
[14] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes
as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.
[15] A. Corona-Figueroa, J. Frawley, S. Bond-Taylor, S. Bethapudi, H. P. Shum, and C. G. Willcocks, “Mednerf:
Medical neural radiance fields for reconstructing 3d-aware ct-projections from a single x-ray,” in 2022 44th annual
international conference of the IEEE engineering in medicine & Biology society (EMBC).
IEEE, 2022, pp.
3843–3848.
[16] M. Wysocki, M. F. Azampour, C. Eilers, B. Busam, M. Salehi, and N. Navab, “Ultra-nerf: Neural radiance fields
for ultrasound imaging,” in Medical Imaging with Deep Learning.
PMLR, 2024, pp. 382–401.
14

<!-- page 15 -->
Zheng et al.: M-Gaussian
[17] J. Xu, D. Moyer, B. Gagoski, J. E. Iglesias, P. E. Grant, P. Golland, and E. Adalsteinsson, “Nesvor: implicit neural
representation for slice-to-volume reconstruction in mri,” IEEE transactions on medical imaging, vol. 42, no. 6,
pp. 1707–1719, 2023.
[18] Q. Wu, Y. Li, L. Xu, R. Feng, H. Wei, Q. Yang, B. Yu, X. Liu, J. Yu, and Y. Zhang, “Irem: High-resolution
magnetic resonance image reconstruction via implicit neural representation,” in International Conference on
Medical Image Computing and Computer-Assisted Intervention.
Springer, 2021, pp. 65–74.
[19] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian splatting for
real-time dynamic scene rendering,” in Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2024, pp. 20 310–20 320.
[20] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi, “Sc-gs: Sparse-controlled gaussian splatting for
editable dynamic scenes,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
2024, pp. 4220–4230.
[21] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, “Motion-aware 3d gaussian splatting for efficient dynamic scene
reconstruction,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.
[22] H. Chen, C. Li, and G. H. Lee, “Neusg: Neural implicit surface reconstruction with 3d gaussian splatting guidance,”
arXiv preprint arXiv:2312.00846, 2023.
[23] X. Lyu, Y.-T. Sun, Y.-H. Huang, X. Wu, Z. Yang, Y. Chen, J. Pang, and X. Qi, “3dgsr: Implicit surface
reconstruction with 3d gaussian splatting,” ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp. 1–12, 2024.
[24] Y. Cai, Y. Liang, J. Wang, A. Wang, Y. Zhang, X. Yang, Z. Zhou, and A. Yuille, “Radiative gaussian splatting for
efficient x-ray novel view synthesis,” in European Conference on Computer Vision.
Springer, 2024, pp. 283–299.
[25] R. Zha, T. J. Lin, Y. Cai, J. Cao, Y. Zhang, and H. Li, “R2-gaussian: Rectifying radiative gaussian splatting for
tomographic reconstruction,” arXiv preprint arXiv:2405.20693, 2024.
[26] Y. Li, X. Fu, H. Li, S. Zhao, R. Jin, and S. K. Zhou, “3dgr-ct: Sparse-view ct reconstruction with a 3d gaussian
representation,” Medical Image Analysis, p. 103585, 2025.
[27] W. Yu, Y. Cai, R. Zha, Z. Fan, C. Li, and Y. Yuan, “X2-gaussian: 4d radiative gaussian splatting for continuous-time
tomographic reconstruction,” arXiv preprint arXiv:2503.21779, 2025.
[28] R. Ramamoorthi and P. Hanrahan, “An efficient representation for irradiance environment maps,” in Proceedings
of the 28th annual conference on Computer graphics and interactive techniques, 2001, pp. 497–500.
[29] K. Payette, P. de Dumast, H. Kebiri, I. Ezhov, J. C. Paetzold, S. Shit, A. Iqbal, R. Khan, R. Kottke, P. Grehten
et al., “An automatic multi-tissue human fetal brain segmentation benchmark using the fetal tissue annotation
dataset,” Scientific data, vol. 8, no. 1, p. 167, 2021.
[30] P. de Dumast, T. Sanchez, H. Lajous, and M. Bach Cuadra, “Simulation-based parameter optimization for
fetal brain mri super-resolution reconstruction,” in International Conference on Medical Image Computing and
Computer-Assisted Intervention.
Springer, 2023, pp. 336–346.
[31] D. C. Van Essen, S. M. Smith, D. M. Barch, T. E. Behrens, E. Yacoub, K. Ugurbil, W.-M. H. Consortium et al.,
“The wu-minn human connectome project: an overview,” Neuroimage, vol. 80, pp. 62–79, 2013.
[32] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga
et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information
processing systems, vol. 32, 2019.
[33] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for
deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.
15
