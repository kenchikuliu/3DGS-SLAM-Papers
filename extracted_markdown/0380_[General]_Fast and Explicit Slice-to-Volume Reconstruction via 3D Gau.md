<!-- page 1 -->
Proceedings of Machine Learning Research – Under Review:1–17, 2026
Full Paper – MIDL 2026 submission
Fast and Explicit: Slice-to-Volume Reconstruction via 3D
Gaussian Primitives with Analytic Point Spread Function
Modeling
Maik Dannecker∗1
m.dannecker@tum.de
Steven Jia∗2
steven.jia@univ-amu.fr
Nil Stolt-Ans´o1
nil.stolt@tum.de
Nadine Girard2
nadine.girard@ap-hm.fr
Guillaume Auzias2
guillaume.auzias@univ-amu.fr
Fran¸cois Rousseau3
francois.rousseau@imt-atlantique.fr
Daniel Rueckert1,4
daniel.rueckert@tum.de
1 TUM University Hospital, Technical University of Munich, Munich, Germany
2 Institut de Neurosciences de la Timone, Aix-Marseille Universit´e, Marseille, France
3 IMT Atlantique, Brest, France
4 Department of Computing, Imperial College London, London, UK
Editors: Under Review for MIDL 2026
Abstract
Recovering high-fidelity 3D images from sparse or degraded 2D images is a fundamental
challenge in medical imaging, with broad applications ranging from 3D ultrasound recon-
struction to MRI super-resolution. In the context of fetal MRI, high-resolution 3D recon-
struction of the brain from motion-corrupted low-resolution 2D acquisitions is a prerequisite
for accurate neurodevelopmental diagnosis. While implicit neural representations (INRs)
have recently established state-of-the-art performance in self-supervised slice-to-volume re-
construction (SVR), they suffer from a critical computational bottleneck: accurately mod-
eling the image acquisition physics requires expensive stochastic Monte Carlo sampling to
approximate the point spread function (PSF). In this work, we propose a shift from neu-
ral network based implicit representations to Gaussian based explicit representations. By
parameterizing the HR 3D image volume as a field of anisotropic Gaussian primitives, we
leverage the property of Gaussians being closed under convolution and thus derive a closed-
form analytical solution for the forward model. This formulation reduces the previously
intractable acquisition integral to an exact covariance addition (Σobs = ΣHR + ΣP SF ), ef-
fectively bypassing the need for compute-intensive stochastic sampling while ensuring exact
gradient propagation. We demonstrate that our approach matches the reconstruction qual-
ity of self-supervised state-of-the-art SVR frameworks while delivering a 5×–10× speed-up
on neonatal and fetal data. With convergence often reached in under 30 seconds, our frame-
work paves the way towards translation into clinical routine of real-time fetal 3D MRI. Code
will be public at https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR.
Keywords: Slice-to-Volume Reconstruction, 3D Gaussian Splatting, Super-Resolution,
Fetal MRI, Neonatal MRI.
1. Introduction
The reconstruction of continuous 3D anatomical models from sparse or low-dimensional
measurements is a ubiquitous challenge across medical imaging, essential for tasks ranging
∗Contributed equally
© 2026 CC-BY 4.0, M. Dannecker, S. Jia, N. Stolt-Ans´o, N. Girard, G. Auzias, F. Rousseau & D. Rueckert.
arXiv:2512.11624v2  [cs.CV]  16 Dec 2025

<!-- page 2 -->
Dannecker Jia Stolt-Ans´o Girard Auzias Rousseau Rueckert
Figure 1: SVR via Gaussian Primitives. a) Clinical acquisition introduces through-
slice blurring due to the anisotropic thick-slice PSF. b) We parameterize the
unknown HR volume as a field of Gaussians GΣj. By approximating the PSF as a
Gaussian kernel ϕΣP SF , we exploit the closure property that the convolution of two
Gaussians is strictly another Gaussian. Consequently, the observed low-resolution
slice ˆIi is simulated analytically via exact covariance addition (Σobs = Σj+ΣPSF )
(b-1), replacing expensive Monte Carlo sampling (b-2) with a fast, deterministic
forward pass. Note, motion correction is omitted in this figure for clarity.
from 3D ultrasound rendering to image super-resolution. In the specific domain of perinatal
neuroradiology, the reconstruction of the moving fetal and neonatal brain from stacks of
thick, low-resolution (LR) 2D MRI slices is increasingly critical for the early diagnosis of
neurodevelopmental disorders (Uus et al., 2023). Consequently, slice-to-volume reconstruc-
tion (SVR) is required to recover a coherent, isotropic high-resolution volume from these
sparse, motion-corrupted stacks. Speed is a crucial factor in this pipeline; rapid reconstruc-
tion allows clinicians to assess scan quality in real-time and re-acquire corrupted stacks
during the same examination, avoiding the logistical burden of patient recall.
Classic SVR frameworks (Jiang et al., 2007; Tourbier et al., 2015; Ebner et al., 2020)
formulate reconstruction as a discrete optimization problem on a fixed voxel grid. While
effective, these methods are limited by discrete sampling resolutions and rely on scattered
data interpolation. Recently, coordinate-based networks or implicit neural representations
2

<!-- page 3 -->
Fast Explicit SVR with Gaussian Primitives
(INRs), such as NeSVoR (Xu et al., 2023), have advanced the state-of-the-art by modeling
the 3D image volume as a continuous function parameterized by a Multilayer Perceptron
(MLP)(Wu et al., 2021; McGinnis et al., 2023).
Recovering a 3D volume from 2D views is a fundamental task in computer graphics,
driving the rapid translation of Neural Radiance Fields (Mildenhall et al., 2022) to MRI.
Notably, NeSVoR successfully adapted the hash encodings of Instant-NGP (M¨uller et al.,
2022) to accelerate fetal brain SVR. Recently, however, 3D Gaussian Splatting (3DGS)
(Kerbl et al., 2023) has surpassed implicit ray-marching methods, demonstrating superior
training efficiency and rendering quality.
Motivated by these advances, we explore the
translation of this explicit Gaussian representation to medical reconstruction.
Although INRs offer resolution independence, they suffer from a severe computational
bottleneck in the forward model. To accurately simulate the physics of slice acquisition
with anisotropic resolution (high in-slice but low out-of-slice resolution), the network must
integrate the continuous volumetric signal over the slice profile, typically approximated by
a Gaussian point spread function (PSF) (Rousseau et al., 2005). For implicit networks,
this convolution is analytically intractable and requires Monte Carlo integration (stratified
sampling) with up to 256 samples per query point (Xu et al., 2023). As each sample consti-
tutes a forward pass, this stochastic approximation causes considerable compute overhead
in current SVR frameworks.
1.1. Contribution
We reformulate SVR from an implicit representation to an explicit Gaussian representation
for fetal MRI, as illustrated in Figure 1. Drawing inspiration from 3DGS and Image-GS
(Zhang et al., 2025), we formulate the HR 3D brain image as a sparse cloud of anisotropic
Gaussians. As our task is volumetric (3D →3D) rather than projective (3D →2D), we
eliminate the need for rasterization and projection matrices; we refer to this representation
as Gaussian Primitives. This formulation replaces the intractable slice acquisition integral
for PSF modeling with an exact, closed-form evaluation, thereby unlocking unprecedented
reconstruction speeds. Our specific contributions are:
• Gaussian Primitives. We introduce an explicit 3D representation for super-resolution
in medical imaging. This continuous, resolution-independent representation enables
rapid optimization without the geometric overhead of graphical projection matrices.
• Analytic PSF. Contrary to INRs which approximate the PSF through expensive
stochastic Monte Carlo sampling, we derive a closed-form analytical solution utilizing
the closure of Gaussians under convolution.
• Scale Regularization.
We introduce Gaussian scale regularization to enforce a
piecewise smooth prior, mitigating overfitting to noise and preventing high-frequency
artifacts.
• Efficiency. We demonstrate reconstruction speed-ups of factor 5 −10 compared to
INR baselines on neonatal data, with convergence often reached in under 30 seconds,
facilitating preview reconstructions during scan-time (Figure 2).
3

<!-- page 4 -->
Dannecker Jia Stolt-Ans´o Girard Auzias Rousseau Rueckert
… 
NeSVoR
GSVR (ours)
9s | 0.71                           18s | 0.88                          37s | 0.93
95s | 0.94 
8s | 0.59                          15s | 0.68                           39s | 0.79
470s | 0.95 
Time | 
SSIM
Reconstruction Time
… 
Figure 2: Convergence Speed. Comparison of SSIM and PSNR over time (seconds) of
the proposed GSVR vs. NeSVoR performed on one simulated subjected.
2. Methodology
We formulate the SVR problem as the inverse recovery of a canonical, HR, isotropic volume
V : R3 →R from a collection of motion-corrupted, anisotropic, LR slices I = {Ii}M
i=1, as
depicted in Figure 1.
2.1. Physical Forward Model
We adopt the continuous acquisition model of NeSVoR. In this model, an observed intensity
Ii(u) at a 2D pixel coordinate u ∈R2 on the i-th slice is the result of integrating the
underlying image volume V over the slice profile (PSF), subject to motion:
Ii(u) = σi
Z
R3 V (x) · ϕi(x −Ti(u)) dx

+ ϵi
(1)
Here Ti(u) = Riu + ti maps the 2D slice coordinate to 3D world space via a rigid transfor-
mation (rotation Ri, translation ti), σi represents the slice-specific intensity scaling, ϵi rep-
resents Gaussian noise, and ϕi denotes the 3D PSF oriented according to the slice geometry.
For INRs, Equation (1) has no closed-form solution. Existing methods rely on stochastic
Monte Carlo sampling to approximate the PSF, typically requiring > 100 queries/samples
per point, which is the primary limiting factor for reconstruction speed (Xu et al., 2023).
2.2. Explicit Gaussian Representation
We propose an explicit Gaussian representation to circumvent the sampling bottleneck
inherent to INRs. This shift enables a closed-form analytical solution to the acquisition
integral of Equation (1), which we detail in Section 2.3.
4

<!-- page 5 -->
Fast Explicit SVR with Gaussian Primitives
w/o PSF
w/ PSF
b)
Original Scale
γ = 0.5
γ = 0.25
a)
c)
λ!"# = 0
λ!"# = 2.5 × 10$%
λ!"# = 0.01
Figure 3: a) Gaussian Representation: Shrinking the Gaussians by factor γ during infer-
ence reveals content adaptation: isotropic primitives cover homogeneous regions,
while anisotropic ellipsoids capture boundaries. b) Analytic PSF: Explicitly
modeling the PSF resolves slice thickness, whereas its absence results in blur and
partial volume artifacts. c) Regularization: Increasing λreg prevents noise over-
fitting by penalizing very small Gaussians. Large λreg results in over-smoothing.
In contrast to standard 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) that relies
on 2D rasterization, we model the MRI signal as continuous volumetric field of Gaussian
primitives. Formally, we define the volume V (x) as a mixture of N 3D Gaussian primitives
(see Figure 3-a). Each primitive Gj is defined by 11 learnable parameters: mean position
µj ∈R3, a covariance matrix Σj ∈R3×3 factorized as Σ = RSST RT where S is a diagonal
scaling matrix and R is a rotation matrix expressed with 4 quaternions, and an intensity cj.
The evaluation of the signal at any 3D coordinate x is computed as a normalized weighted
sum:
V (x) =
P
j∈SK(x) cj · exp

−1
2(x −µj)T Σ−1
j (x −µj)

P
j∈SK(x) exp

−1
2(x −µj)T Σ−1
j (x −µj)

+ δ
(2)
where δ is a small constant for numerical stability, and SK(x) denotes the set of K Gaussians
nearest to x (see Section 2.5).
Our formulation differs from standard 3DGS by strictly operating in 3D world space.
Unlike graphical rendering, we do not project primitives onto a 2D imaging plane, elimi-
nating the need for the Jacobian projection matrix. Instead of ”splatting” or ”rasterizing”
Gaussians, we evaluate the continuous volumetric density field directly.
2.3. Closed-Form PSF Modeling via Convolution
To accurately resolve the acquisition physics, we must model the PSF, ϕ, which incorporates
the slice selection profile and in-plane blurring. Conceptually, the observed intensity Ii(u)
5

<!-- page 6 -->
Dannecker Jia Stolt-Ans´o Girard Auzias Rousseau Rueckert
is the result of convolving the underlying volume V at motion-corrected position Ti(u), with
the slice-oriented PSF ϕi:
Ii(u) = (V ∗ϕi)(Ti(u)) .
(3)
In standard approaches, including INRs, evaluating this convolution is analytically in-
tractable and typically approximated via compute-expensive Monte Carlo integration (Fig-
ure 1, b-2). We circumvent this bottleneck by leveraging the semi-group property of Gaus-
sians: the convolution of two Gaussians is strictly another Gaussian (Figure 1, b-1).
First, we model the physical PSF as anisotropic 3D Gaussian kernel ϕ = N(0, ΣPSF),
where ΣPSF is defined in the slice coordinate system to account for slice resolution and thick-
ness (Rousseau et al., 2010; Gholipour et al., 2010). Second, we parametrize the unknown
HR image volume V as a sum of Gaussians Gj(µj, Σj), as described by Equation (2). The
convolution in Equation (3) now reduces to a closed-form addition of their covariance ma-
trices. For a slice rotated by Ri, the effective observed covariance Σobs,j for each Gaussian
primitive is given by:
Σobs,j = Σj + RiΣPSFRT
i .
(4)
Substituting Equation (4) into the acquisition model transforms the integral from Equa-
tion (1) into a direct, exact evaluation. The final intensity Ii(u) is computed by replacing
the HR covariance Σj with the convolved covariance Σobs,j in the rendering equation:
Ii(u) = σi
P
j∈SK cj exp

−1
2vT
j Σ−1
obs,jvj

P
j∈SK exp

−1
2vT
j Σ−1
obs,jvj

+ δ
,
(5)
Here vj = Ti(u) −µj is the motion corrected coordinate vector relative to the j-th Gaus-
sian mean. This formulation evaluates the physically ”blurred” signal via a simple matrix
addition, ensuring fast and exact gradient propagation without stochastic sampling.
2.4. Motion Correction and Outlier Handling
We follow the robust optimization strategy of NeSVoR by jointly optimizing the Gaussian
parameters and slice-wise rigid transformations. For each slice i, we learn a rigid transfor-
mation Ti(u) = Riu + ti. Critically, the anisotropic PSF covariance ΣPSF is dynamically
rotated alongside the slice via Equation (4), ensuring the through-plane blur remains aligned
with the slice normal in 3D space.
To robustly handle outliers (e.g., signal drop, inconsistent contrast), we incorporate
two mechanisms into the forward model: a learnable intensity scalar σi to compensate for
global intensity shifts, and an aleatoric uncertainty weight ωi to down-weight corrupted
slices during loss computation. Together, this setup allows the framework to self-correct for
motion and artifacts in an end-to-end manner.
6

<!-- page 7 -->
Fast Explicit SVR with Gaussian Primitives
2.5. Optimization Strategy
Sparse Computation (Top-K): To ensure computational efficiency, we employ a Top-K
culling strategy inspired by (Zhang et al., 2025). For any motion corrected coordinate Ti(u),
we only compute contributions of the K nearest Gaussians (by L2 distance ∥Ti(u) −µj∥2).
Content-Adaptive Initialization:
To accelerate convergence, we utilize gradient
adaptive initialization (Zhang et al., 2025). We compute the gradient magnitude ∥∇Ii∥
of the input slices to construct a sampling probability map, and define the initialization
probability Pinit as a mixture of gradient-based and uniform sampling:
Pinit(Ti(u)) ∝(1 −λinit)∥∇Ii(u)∥+ λinit .
(6)
We initialize the means µj by sampling from Pinit, ensuring high Gaussian density in detailed
anatomical regions (tissue boundaries) and lower density in homogeneous regions.
Loss Function: We optimize the Gaussian parameters (µ, Σ, c) and motion parameters
(q, t) to minimize the L1 reconstruction loss combined with a scale regularization term:
L =
X
i
∥Ii −ˆIi∥1 + λreg
N
X
j=1
∥sj −starget∥2
2
(7)
where sj are the scaling factors of the covariance, and starget a hyperparameter defining the
desired mean scale (empirically set to 1.6mm isotropic). The regularization prevents the
Gaussians from becoming too small (overfitting to noise) or expanding excessively, thereby
ensuring smooth anatomical continuity.
3. Experiments and Results
3.1. Baselines
We compare our framework against two self-supervised state-of-the-art SVR methods:
1) SVRTK (Kuklisova-Murgasova et al., 2012a): The standard CPU-based iterative op-
timization toolkit, executed here on 16 parallel cores. We limited reconstruction to three
iterations, as convergence was observed at this stage. We noted that SVRTK is highly
sensitive to masking imperfections; despite manual segmentation corrections, final recon-
structions occasionally exhibited field-of-view cut-offs.
2) NeSVoR (Xu et al., 2023): A GPU-accelerated INR approach yielding state-of-the-art
reconstruction quality and runtime. We utilized default hyperparameters, which proved op-
timal for our experiments. Notably, NeSVoR optimizes ≈4.8 million parameters, roughly
an order of magnitude more than our ≈550, 000 parameters.
3.2. Data
Simulated Data: To ensure high-quality ground truth with minimal motion artifacts, we
selected isotropic (0.5 mm) neonatal scans from the dHCP dataset (Cordero-Grande et al.,
2019). We simulated clinical fetal acquisitions by generating three orthogonal stacks (0.5 ×
0.5×3 mm) (Mercier et al., 2025). To model the worst-case residual motion that is typically
expected after SVoRT pre-alignment, we applied random slice-wise rigid transformations
sampled from U(−6◦, 6◦) and U(−4, 4) mm.
7

<!-- page 8 -->
Dannecker Jia Stolt-Ans´o Girard Auzias Rousseau Rueckert
Table 1: Quantitative comparison of reconstruction accuracy and runtime on 10 neonatal
subjects with simulated motion corruption. Best metrics in bold.
Method
PSNR ↑
SSIM ↑
NCC ↑
Time (s)↓
SVRTK (Kuklisova-Murgasova et al., 2012b)
18.65∗± 1.22
0.641∗± 0.074
0.541∗± 0.126
573
NeSVoR (Xu et al., 2023)
28.38 ± 1.37
0.933 ± 0.030
0.955 ± 0.011
478
GSVR (Ours)
28.76 ± 1.52
0.936 ± 0.018
0.957 ± 0.012
79
*Statistically significant difference to the best performing method (p < 0.05, paired t-test).
dHCP T2w, Turbo Spin Echo
MarsFet T2w, (HASTE/TRUFISP)
GSVR (ours)             NeSVoR                     SVRTK
t=96s
t=484s
t=1038s
t=56s
t=483s
t=464s
t=91s
t=480s
t=1027s
t=50s
t=498s
t=467s
t=24s
t=479s
t=279s
t=35s
t=477s
t=233s
t=25s
t=485s
t=152s
t=14s
t=489s
t=187s
Figure 4: Qualitative comparison on dHCP and MarsFet cohorts. SVRTK exhibits
masking artifacts on limited FOVs. The proposed GSVR matches the high re-
construction quality of NeSVoR while achieving speed-ups of up to factor 10.
In-vivo dHCP Fetal Data: We selected four subjects from the fetal dHCP dataset
(Price et al., 2019) between 27 and 37 weeks gestational age (GA). Each subject includes
6 uniquely oriented motion-corrupted stacks centered on the fetal brain using a 3T Philips
Achieva with zoomed multiband single-shot Turbo Spin Echo sequence at an in-slice reso-
lution of 1.1 × 1.1mm2 and 2.2 mm slice thickness.
In-vivo MarsFet Fetal Data: We used two subjects from the clinical dataset named
MarsFet(Mihailov et al., 2025). Both subjects include two acquisition sequences, a half-
Fourier single-shot turbo spin-echo (HASTE) sequence (0.74 × 0.74 × 3.5mm) and a TRU-
FISP sequence (0.625 × 0.625 × 3.0mm) (Girard et al., 2003).
Pre-processing. We applied a unified pipeline across all methods. In-vivo scans underwent
bias field correction (Tustison et al., 2010) (plus denoising for MarsFet), automated brain
masking (Ranzini et al., 2021), and were co-registered to a common space via SVoRT (Xu
8

<!-- page 9 -->
Fast Explicit SVR with Gaussian Primitives
et al., 2022). For simulated data, we used ground-truth masks with no further processing.
Target resolution for all data and methods was set to 0.5mm isotropic.
3.3. Results
Quantitative Evaluation on Simulated Data: Table 1 summarizes performance on 10
simulated subjects. Our Gaussian-based SVR (GSVR) matches the state-of-the-art NeSVoR
across all metrics (PSNR 28.07, SSIM 0.936) without external slice pre-alignment, demon-
strating robust internal motion correction. SVRTK degrades significantly (18.20 dB) due to
sensitivity to large motion corruption (see Appendix Figure 5). Critically, GSVR completes
reconstruction in just 79s, a 6× and 7× speed-up over NeSVoR (478s) and SVRTK (573s),
respectively. As shown in Figure 2, our Gaussian representation with analytic PSF enables
rapid convergence, yielding usable previews (SSIM > 0.8) in under 20 seconds.
Qualitative Evaluation on In-vivo Data: Figure 4 shows reconstructions on real-
world clinical scans from the dHCP and MarsFet datasets. SVRTK reconstructions often
exhibit field-of-view (FOV) cut-offs, failing to recover the complete brain due to masking
failures or incomplete FOV in the raw acquisitions. Both NeSVoR and GSVR successfully
recover coherent, high-resolution isotropic 3D volumes. Our method resolves fine anatomical
structures, such as the cortex, with a level of sharpness comparable to NeSVoR. The visual
results further confirm the efficiency gains of GSVR demonstrating speed-ups of up to factor
10 over NeSVoR an SVRTK without compromising reconstruction quality.
3.4. Ablation Study
We analyze the impact of model components in Table 2.
Gaussian Density (N): Increasing the number of primitives from 10k to 50k improves
detail recovery considerably (+1.4 dB); further increases yield diminishing returns.
Sparsity (Top-K): Reducing neighbors to K = 10 degrades reconstruction (−2.7 dB)
by spatially limiting the gradient flow required for robust motion correction. K = 80 yields
the highest metrics, but increases runtime by 40%; we select K = 50 as the best trade-off.
PSF & Regularization: Disabling the analytic PSF causes a massive drop in PSNR
(−8.0 dB) due to partial volume effects. Scale regularization (+1.3 dB) and adaptive ini-
tialization (+1.3 dB) prove beneficial to prevent overfitting and ensure robust convergence.
4. Discussion
Analytic Tractability & Broader Impact: We demonstrated that the computational
bottleneck of MRI physics simulation—specifically PSF convolution—can be solved analyti-
cally by adopting an explicit Gaussian representation. Unlike INRs, which require expensive
stochastic Monte Carlo sampling to approximate slice integration, our approach transforms
this into a closed-form algebraic operation (Σobs = ΣHR + ΣPSF ). This finding has broad
implications beyond SVR: any inverse problem involving Gaussian kernels—such as decon-
volution microscopy, PET partial volume correction, or diffusion tensor estimation—could
benefit from this formulation, restoring the exact gradient propagation of classical signal
processing within a learning-based framework.
9

<!-- page 10 -->
Dannecker Jia Stolt-Ans´o Girard Auzias Rousseau Rueckert
Table 2: Ablation study analyzing the impact of Gaussian density (NGauss), sparsity (top-
K), and model components on reconstruction quality and runtime.
Configuration
Metrics
NGauss
Top-K
Initadapt
PSF
Reg
PSNR ↑
SSIM ↑
Time (s) ↓
10000
50
✓
✓
✓
27.41 ± 1.31
0.906 ± 0.029
64
20000
50
✓
✓
✓
28.38 ± 1.42
0.927 ± 0.020
69
50000
50
✓
✓
✓
28.76 ± 1.52
0.936 ± 0.018
79
80000
50
✓
✓
✓
28.60 ± 1.59
0.936 ± 0.018
82
50000
10
✓
✓
✓
26.07 ± 1.02
0.900 ± 0.018
28
50000
20
✓
✓
✓
27.57 ± 1.25
0.923 ± 0.016
40
50000
80
✓
✓
✓
28.88 ± 1.51
0.939 ± 0.018
111
50000
50
✗
✓
✓
27.50 ± 0.92
0.920 ± 0.019
80
50000
50
✓
✗
✓
20.78 ± 1.02
0.723 ± 0.049
69
50000
50
✓
✓
✗
27.42 ± 1.71
0.917 ± 0.032
78
Shaded row indicates model configuration used in this study.
Towards Real-Time SVR: By eliminating the sampling bottleneck, our framework
achieves usable reconstructions in under 30 seconds, enabling intra-session quality con-
trol. Clinicians can thus assess scan utility immediately, potentially reducing patient recall
rates. While our current implementation relies on generic PyTorch kernels, runtimes re-
ported in 2D Gaussian fitting benchmarks (Zhang et al., 2025) suggest that porting our
volumetric operations to dedicated CUDA backends could yield a further 2–4× speed-up.
Unlocking this engineering potential paves the way for real-time SVR, transforming recon-
struction from offline post-processing into an interactive clinical tool.
4.1. Limitations and Future Work
Bias Correction & Robustness: Future work could integrate bias field correction us-
ing low-frequency Gaussian primitives. Additionally, a systematic evaluation is needed to
analyze the robustness of the model to severely corrupted acquisitions.
Coarse-to-Fine Optimization: Implementing progressive densification of Gaussians
primitives (Kerbl et al., 2023)—starting with sparse, large primitives—would enable robust
and fast global gradient flow for motion correction even with small neighborhoods, followed
by adaptive splitting to reconstruct fine details.
Extension to Diffusion MRI: By adapting Spherical Harmonics (used in 3DGS for
view-dependence) to model direction-dependent signals, the framework could be extended
to reconstruct Diffusion Weighted Imaging data.
Acknowledgments
This research has been supported by the ERA-NET NEURON MULTI-FACT Project.
Data were provided by the developing Human Connectome Project, KCL-Imperial-Oxford
Consortium funded by the European Research Council under the European Union Seventh
10

<!-- page 11 -->
Fast Explicit SVR with Gaussian Primitives
Framework Programme (FP/2007-2013) / ERC Grant Agreement no. [319456]. We are
grateful to the families who generously supported this trial.
References
Lucilio Cordero-Grande, Anthony N Price, Emer J Hughes, and Joseph V. Hajnal. Automat-
ing fetal brain reconstruction using distance regression learning. International Society for
Magnetic Resonance in Medicine, 2019.
Michael Ebner, Guotai Wang, Wenqi Li, Michael Aertsen, Premal A Patel, Rosalind Augh-
wane, Andrew Melbourne, Tom Doel, Steven Dymarkowski, Paolo De Coppi, et al. An
automated framework for localization, segmentation and super-resolution reconstruction
of fetal brain MRI. NeuroImage, 206:116324, 2020.
Ali Gholipour, Judy A Estroff, Mustafa Sahin, Sanjay P Prabhu, and Simon K Warfield.
Maximum a posteriori estimation of isotropic high-resolution volumetric mri from or-
thogonal thick-slice scans. In International Conference on Medical Image Computing and
Computer-Assisted Intervention, pages 109–116. Springer, 2010.
Nadine Girard et al. MR imaging of acquired fetal brain disorders. Child’s nervous system
: ChNS : official journal of the International Society for Pediatric Neurosurgery, 19(7-8):
490–500, August 2003. ISSN 0256-7040. doi: 10.1007/s00381-003-0761-x.
Shuzhou Jiang, Hui Xue, Alan Glover, Mary Rutherford, Daniel Rueckert, and Joseph V
Hajnal. MRI of moving subjects using multislice snapshot images with volume reconstruc-
tion (SVR): application to fetal, neonatal, and adult brain studies. IEEE transactions on
medical imaging, 26(7):967–980, 2007.
Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billion-scale similarity search with GPUs.
IEEE Transactions on Big Data, 7(3):535–547, 2019.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D Gaus-
sian Splatting for Real-Time Radiance Field Rendering, August 2023.
URL http:
//arxiv.org/abs/2308.04079. arXiv:2308.04079 [cs].
Maria Kuklisova-Murgasova, Gerardine Quaghebeur, Mary A Rutherford, Joseph V Hajnal,
and Julia A Schnabel. Reconstruction of fetal brain MRI with intensity matching and
complete outlier removal. Medical image analysis, 16(8):1550–1564, 2012a.
Maria Kuklisova-Murgasova, Gerardine Quaghebeur, Mary A. Rutherford, Joseph V. Ha-
jnal, and Julia A. Schnabel. Reconstruction of fetal brain MRI with intensity match-
ing and complete outlier removal. Medical Image Analysis, 16(8):1550–1564, December
2012b. ISSN 13618415. doi: 10.1016/j.media.2012.07.004. URL https://linkinghub.
elsevier.com/retrieve/pii/S1361841512000965.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint
arXiv:1711.05101, 2017.
11

<!-- page 12 -->
Dannecker Jia Stolt-Ans´o Girard Auzias Rousseau Rueckert
Julian McGinnis, Suprosanna Shit, Hongwei Bran Li, Vasiliki Sideri-Lampretsa, Robert
Graf, Maik Dannecker, Jiazhen Pan, Nil Stolt-Ans´o, Mark M¨uhlau, Jan S Kirschke, et al.
Single-subject multi-contrast MRI super-resolution via implicit neural representations.
In International Conference on Medical Image Computing and Computer-Assisted Inter-
vention, pages 173–183. Springer, 2023.
Chloe Mercier, Sylvain Faisan, Alexandre Pron, Nadine Girard, Guillaume Auzias, Thierry
Chonavel, and Fran¸cois Rousseau. Intersection-based slice motion estimation for fetal
brain imaging. Computers in Biology and Medicine, 190:110005, May 2025. ISSN 0010-
4825. doi: 10.1016/j.compbiomed.2025.110005. URL https://www.sciencedirect.com/
science/article/pii/S0010482525003567.
Angeline Mihailov et al. Burst of gyrification in the human brain after birth. Communica-
tions Biology, 8(1):1–13, 2025. ISSN 2399-3642.
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ra-
mamoorthi, and Ren Ng. NeRF: representing scenes as neural radiance fields for view
synthesis. Communications of the ACM, 65(1):99–106, January 2022. ISSN 0001-0782,
1557-7317. doi: 10.1145/3503250. URL https://dl.acm.org/doi/10.1145/3503250.
Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller.
Instant Neural
Graphics Primitives with a Multiresolution Hash Encoding. ACM Transactions on Graph-
ics, 41(4):1–15, July 2022. ISSN 0730-0301, 1557-7368. doi: 10.1145/3528223.3530127.
URL http://arxiv.org/abs/2201.05989. arXiv:2201.05989 [cs].
Anthony N Price, Lucilio Cordero-Grande, Emer Hughes, Suzanne Hiscocks, Elaine Green,
Laura McCabe, Jana Hutter, Giulio Ferrazzi, Maria Deprez, Thomas Roberts, et al. The
developing human connectome project (dHCP): fetal acquisition protocol. In Proceedings
of the annual meeting of the International Society of Magnetic Resonance in Medicine
(ISMRM), volume 244, 2019.
Marta Ranzini, Lucas Fidon, S´ebastien Ourselin, Marc Modat, and Tom Vercauteren. MON-
AIfbs: MONAI-based fetal brain MRI deep learning segmentation, 2021.
Fran¸cois Rousseau, Orit Glenn, Bistra Iordanova, Claudia Rodriguez-Carranza, Daniel
Vigneron, James Barkovich, and Colin Studholme.
A novel approach to high resolu-
tion fetal brain mr imaging. In International Conference on Medical Image Computing
and Computer-Assisted Intervention, pages 548–555, Berlin, Heidelberg, 2005. Springer,
Springer Berlin Heidelberg. ISBN 978-3-540-32094-4.
Fran¸cois Rousseau, Kio Kim, Colin Studholme, Meriam Koob, and J-L Dietemann. On
super-resolution for fetal brain mri. In International conference on medical image com-
puting and computer-assisted intervention, pages 355–362. Springer, 2010.
S´ebastien Tourbier, Xavier Bresson, Patric Hagmann, Jean-Philippe Thiran, Reto Meuli,
and Meritxell Bach Cuadra. An efficient total variation algorithm for super-resolution in
fetal brain MRI with adaptive regularization. NeuroImage, 118:584–597, 2015.
12

<!-- page 13 -->
Fast Explicit SVR with Gaussian Primitives
Nicholas J Tustison, Brian B Avants, Philip A Cook, Yuanjie Zheng, Alexander Egan,
Paul A Yushkevich, and James C Gee.
N4ITK: improved N3 bias correction.
IEEE
transactions on medical imaging, 29(6):1310–1320, 2010.
Alena U Uus, Alexia Egloff Collado, Thomas A Roberts, Joseph V Hajnal, Mary A Ruther-
ford, and Maria Deprez. Retrospective motion correction in foetal MRI for clinical appli-
cations: existing methods, applications and integration into clinical practice. The British
journal of radiology, 96(1147):20220071, 2023.
Qing Wu, Yuwei Li, Lan Xu, Ruiming Feng, Hongjiang Wei, Qing Yang, Boliang Yu, Xi-
aozhao Liu, Jingyi Yu, and Yuyao Zhang. IREM: High-resolution magnetic resonance im-
age reconstruction via implicit neural representation. In Medical Image Computing and
Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Stras-
bourg, France, September 27–October 1, 2021, Proceedings, Part VI 24, pages 65–74.
Springer, 2021.
Junshen Xu, Daniel Moyer, P Ellen Grant, Polina Golland, Juan Eugenio Iglesias, and Elfar
Adalsteinsson. SVoRT: Iterative transformer for slice-to-volume registration in fetal brain
MRI. In International Conference on Medical Image Computing and Computer-Assisted
Intervention, pages 3–13. Springer, 2022.
Junshen Xu, Daniel Moyer, Borjan Gagoski, Juan Eugenio Iglesias, P Ellen Grant, Polina
Golland, and Elfar Adalsteinsson. NeSVoR: implicit neural representation for slice-to-
volume reconstruction in MRI. IEEE transactions on medical imaging, 42(6):1707–1719,
2023.
Yunxiang Zhang, Bingxuan Li, Alexandr Kuznetsov, Akshay Jindal, Stavros Diolatzis,
Kenneth Chen, Anton Sochenov, Anton Kaplanyan, and Qi Sun. Image-GS: Content-
Adaptive Image Representation via 2D Gaussians.
In Proceedings of the Special In-
terest Group on Computer Graphics and Interactive Techniques Conference Confer-
ence Papers, pages 1–11, August 2025.
doi: 10.1145/3721238.3730596.
URL http:
//arxiv.org/abs/2407.01866. arXiv:2407.01866 [cs].
Appendix A. Implementation
For initialization, we utilize the content-adaptive strategy described in Sec. 2.5. We set
the number of Gaussians to N = 50, 000 for all experiments.
We set the initialization
mixing parameter λinit = 0.0, favoring high-frequency edge regions entirely over uniform
distribution to capture anatomical detail rapidly.
A.1. Optimization Details
The model is trained for 500 epochs using the AdamW optimizer(Loshchilov and Hutter,
2017).
We utilize a large effective batch size (processing up to 107 points per pass) to
approximate full-batch gradient descent. The learning rates are set as follows: positions µ
and scaling s at 2.5 × 10−2, rotation quaternions q and color c at 1.0 × 10−2. The motion
correction parameters are optimized with a lower learning rate for rotations (2.5 × 10−3)
13

<!-- page 14 -->
Dannecker Jia Stolt-Ans´o Girard Auzias Rousseau Rueckert
compared to translations (1.0×10−1). A StepLR scheduler is applied, decaying the learning
rate by a factor of 0.5 every 200 epochs.
We employ the Top-K approximation with K = 50 neighbors. To balance speed and
accuracy, the neighbor index is updated every 50 epochs using the FAISS library (Johnson
et al., 2019) for efficient GPU-based nearest neighbor search.
The scale regularization
weight was empirically set to λreg = 2.5 × 10−3 with a target scale of 1.6mm to encourage
smoothness.
A.2. Acceleration
To eliminate Python overhead in the critical path, we implemented custom fused kernels
for:
• Fused Mahalanobis Distance: A single kernel computes the covariance inversion
and the quadratic form (x −µ)T Σ−1(x −µ) for the K nearest neighbors.
• Fused Motion Correction: We fuse the quaternion-to-matrix conversion, coordi-
nate transformation, and PSF covariance rotation RΣPSF RT into a single operation.
This optimized implementation allows for convergence in under 60 seconds for standard
fetal volumes.
14

<!-- page 15 -->
Fast Explicit SVR with Gaussian Primitives
Appendix B. Additional Results
Coronal
Sagittal
Axial
Ground Truth
LR Stack
SVRTK
NeSVoR
GSVR (ours)
Figure 5: Qualitative evaluation on simulated data. Visualizing from left to right the
ground truth neonaten reconstruction, the low-resolution (LR) motion corrupted
stacks, and the three evaluated methods. Whereas SVRTK struggles with motion
correction, NeSVoR and the proposed GSVR framework achieve high quality
reconstructions. Reconstruction times for the depicted neonatal subject are 763s
(SVRTK), 471s (NeSVoR), and 92s (GSVR).
15

<!-- page 16 -->
Dannecker Jia Stolt-Ans´o Girard Auzias Rousseau Rueckert
Axial
Coronal
Sagittal
GSVR
NeSVoR
SVRTK
GSVR
NeSVoR
SVRTK
dHCP Data, 4 Subjects, T2w, Turbo Spin Echo
MarsFet Data, 2 Subjects, T2w, HASTE | TRUFISP
Axial
Coronal
Sagittal
Figure 6: Qualitative comparison on dHCP and MarsFet cohorts (complete
view). SVRTK exhibits masking artifacts on limited FOVs. The proposed GSVR
matches the high reconstruction quality of NeSVoR while achieving speed-ups of
up to factor 10.
16

<!-- page 17 -->
Fast Explicit SVR with Gaussian Primitives
Appendix C. Nomenclature
Symbol
Description
Indices and Dimensions
N
Total number of Gaussian primitives (N = 50, 000).
K
Number of nearest neighbors for Top-K sparsity (K = 50).
i
Index for the 2D MRI slice (i ∈{1, . . . , M}).
j
Index for the Gaussian primitive (j ∈{1, . . . , N}).
u
2D coordinate in the slice image plane (u ∈R2).
x
3D coordinate in the canonical world volume (x ∈R3).
Gaussian Representation
V (x)
Continuous volumetric representation of the 3D image volume.
Gj
The j-th anisotropic 3D Gaussian primitive.
µj
Mean position vector of Gaussian j (∈R3).
Σj
Covariance matrix of Gaussian j (∈R3×3).
sj
Scaling vector of Gaussian j (log-space parameters).
cj
Learnable intensity/color parameter of Gaussian j.
Forward Model & Physics
Ii, ˆIi
Observed and reconstructed intensity for slice i.
σi
Learnable intensity scaling factor for slice i.
ωi
Aleatoric uncertainty weight for slice i (outlier handling).
Ti
Rigid motion transformation (u →x) defined by {Ri, ti}.
vj
Centered relative coordinate vector (Ti(u) −µj).
ϕ
The 3D Point Spread Function (PSF).
ΣPSF
PSF covariance in slice coordinates (thickness/in-plane).
Σobs,j
Effective ”observed” covariance (Σj + RiΣPSFRT
i ).
Optimization & Hyperparameters
λreg
Weight for scale regularization term (2.5 × 10−3).
starget
Target scale hyperparameter (isotropic 1.6mm).
λinit
Mixing parameter for gradient-adaptive initialization.
γ
Visualization shrink factor for Gaussian iso-surfaces.
Table 3: Summary of notation and symbols used in the forward model and optimization.
17
