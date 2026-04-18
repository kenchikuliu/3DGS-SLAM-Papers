<!-- page 1 -->
CRYOSPLAT: GAUSSIAN SPLATTING FOR CRYO-EM
HOMOGENEOUS RECONSTRUCTION
Suyi Chen & Haibin Ling
Department of Computer Science
Stony Brook University
Stony Brook, NY 11794, USA
{suychen, hling}@cs.stonybrook.edu
ABSTRACT
As a critical modality for structural biology, cryogenic electron microscopy (cryo-
EM) facilitates the determination of macromolecular structures at near-atomic res-
olution. The core computational task in single-particle cryo-EM is to reconstruct
the 3D electrostatic potential of a molecule from noisy 2D projections acquired
at unknown orientations. Gaussian mixture models (GMMs) provide a continu-
ous, compact, and physically interpretable representation for molecular density
and have recently gained interest in cryo-EM reconstruction. However, existing
methods rely on external consensus maps or atomic models for initialization, lim-
iting their use in self-contained pipelines. In parallel, differentiable rendering
techniques such as Gaussian splatting have demonstrated remarkable scalability
and efficiency for volumetric representations, suggesting a natural fit for GMM-
based cryo-EM reconstruction. However, off-the-shelf Gaussian splatting meth-
ods are designed for photorealistic view synthesis and remain incompatible with
cryo-EM due to mismatches in the image formation physics, reconstruction ob-
jectives, and coordinate systems. Addressing these issues, we propose cryoSplat,
a GMM-based method that integrates Gaussian splatting with the physics of cryo-
EM image formation. In particular, we develop an orthogonal projection-aware
Gaussian splatting, with adaptations such as a view-dependent normalization term
and FFT-aligned coordinate system tailored for cryo-EM imaging. These inno-
vations enable stable and efficient homogeneous reconstruction directly from raw
cryo-EM particle images using random initialization. Experimental results on real
datasets validate the effectiveness and robustness of cryoSplat over representative
baselines. The code will be released upon publication.
1
INTRODUCTION
Single particle cryogenic electron microscopy (cryo-EM) has emerged as a transformative tool in
structural biology, enabling visualization of macromolecular complexes at atomic or near-atomic
resolution without crystallization (K¨uhlbrandt, 2014; Nogales, 2016; Renaud et al., 2018). Central
to cryo-EM is the computational task of reconstructing a 3D volume from a large collection of
2D projection images, each corresponding to a different, unknown viewing direction of identical
particles embedded in vitreous ice.
This inverse problem is inherently ill-posed and computationally challenging. First, cryo-EM im-
ages are severely corrupted by noise due to the low electron dose required to prevent radiation dam-
age. For experimental datasets, the signal-to-noise (SNR) could be as low as around −20 dB (Ben-
dory et al., 2020; Bepler et al., 2020). Second, the orientations (poses) of individual particles are
unknown and must be inferred jointly with the 3D structure. Third, many biological samples exhibit
structural heterogeneity, with multiple conformational states coexisting in the same dataset.
These difficulties underscore two central objectives in cryo-EM image analysis: ab initio recon-
struction, which aims to estimate both the 3D structure and particle poses directly from raw data
without prior models, and heterogeneous reconstruction, which seeks to disentangle and reconstruct
multiple structural states from the dataset. Both objectives fundamentally rely on the availability of
1
arXiv:2508.04929v3  [eess.IV]  25 Sep 2025

<!-- page 2 -->
a robust and efficient homogeneous reconstruction method, which assumes all particles correspond
to a single structure and serves as a building block for more complex inference.
Approaches to homogeneous reconstruction include methods based on backprojection, iterative
expectation-maximization with voxel-based volumes (Tang et al., 2007; Scheres, 2012; Punjani
et al., 2017; Shekarforoush et al., 2024), and more recently, neural representation learning (Zhong
et al., 2021a;b), which models the 3D volume using coordinate-based networks. In parallel, Gaus-
sian mixture models (GMMs) have received attention for their continuous, compact, and physically
interpretable parameterization of molecular density (Chen & Ludtke, 2021; Chen et al., 2023a). No-
tably, GMMs offer a natural connection to atomic models and can represent fine structural details
using fewer parameters (Chen et al., 2023b; Li et al., 2024; Schwab et al., 2024; Chen, 2025).
Despite their conceptual appeal, existing GMM-based methods (Chen & Ludtke, 2021; Chen et al.,
2023a;b; Li et al., 2024; Schwab et al., 2024; Chen, 2025) for cryo-EM reconstruction rely on
nontrivial prerequisite steps. These approaches typically rely on consensus volumes from external
pipelines, or even atomic models, for initialization, and have not demonstrated stable convergence
when directly optimizing from experimental images. In fact, no prior method achieves reliable
GMM-based reconstruction even under known particle poses, due to the inherent difficulty of op-
timizing mixture parameters in extreme noise. As a result, GMMs lack a self-contained and stable
formulation that can serve as a backbone for broader reconstruction workflows.
In this work, we propose cryoSplat, a GMM-based homogeneous reconstruction method that fills
this foundational gap. Given known particle poses, cryoSplat performs stable and efficient recon-
struction directly from raw cryo-EM projection images, starting from random initialization and re-
quiring no external priors. Inspired by recent advances in 3D Gaussian Splatting (3DGS) by Kerbl
et al. (2023), we model the 3D density as a mixture of anisotropic Gaussians and project them into
2D using a novel differentiable orthographic splatting algorithm consistent with cryo-EM physics.
To support practical scalability and training efficiency, we develop a CUDA-accelerated real-space
renderer that enables fast rasterization and optimization of the GMM.
Our contributions can be summarized as follows:
• A self-contained GMM-based reconstruction method: We present cryoSplat as the first
method capable of performing cryo-EM homogeneous reconstruction from a randomly ini-
tialized Gaussian mixture model without an external prior, thereby establishing the missing
foundation needed to develop GMMs into standalone reconstruction tools.
• A physically accurate projection model: We design a splatting algorithm under orthogo-
nal projection tailored to cryo-EM image formation, enabling differentiable projection of
anisotropic Gaussians in real space.
• An efficient implementation: We implement a CUDA-accelerated real-space renderer that
enables fast optimization of GMMs with tens of thousands of Gaussians.
• Experimental validation: We demonstrate the effectiveness of cryoSplat on real datasets,
showing it converges reliably from random initialization and achieves reconstruction qual-
ity outperforming state-of-the-art methods.
2
RELATED WORK
2.1
VOLUME REPRESENTATION IN CRYO-EM
In cryo-EM experiments, purified biomolecules are rapidly frozen in a thin layer of vitreous ice,
where each particle adopts a random orientation. A high-energy electron beam passes through the
specimen, interacts with the electrostatic potential of the particles, and is recorded on a detector as a
2D projection image (Singer & Sigworth, 2020). The goal of cryo-EM reconstruction is to recover
the 3D electrostatic potential, i.e., the volume, from a large set of such noisy and randomly oriented
2D projections. Central to cryo-EM reconstruction is the choice of volume representation.
2.1.1
VOXEL-BASED REPRESENTATION
Voxel-based representations are the most widely used in conventional cryo-EM software, e.g., RE-
LION (Scheres, 2016b), cryoSPARC (Punjani et al., 2017) and EMAN2 (Tang et al., 2007). The 3D
2

<!-- page 3 -->
volume is discretized into a regular grid of density values, enabling fast projection and reconstruction
via FFT-based algorithms. Despite their practical success, voxel grids are memory-intensive and in-
herently discrete, which limits their compatibility with modern learning-based analysis frameworks.
2.1.2
NEURAL FIELD
Neural fields represent the volume as a continuous function parameterized by neural networks.
These methods (Zhong et al., 2021a;b; Levy et al., 2022a;b; 2025) offer differentiability, implicit
smoothness, and natural compatibility with learning-based heterogeneous analysis. However, the
implicit nature of neural fields often comes at the cost of interpretability, and such models are typi-
cally slow to train and difficult to constrain with biological priors.
2.1.3
GAUSSIAN MIXTURE MODEL
Gaussian mixture models have a long history in structural biology, with early uses for molecular ap-
proximation (Grant & Pickup, 1995; Grant et al., 1996; Kawabata, 2008). E2GMM (Chen & Ludtke,
2021) was among the first to apply GMMs to cryo-EM heterogeneous reconstruction. Like neural
fields, GMMs can approximate any smooth density function and support differentiable optimization.
More importantly, GMMs provide an explicit and interpretable representation that naturally links to
atomic structures. Recent studies (Chen et al., 2023a;b; Li et al., 2024; Schwab et al., 2024; Chen,
2025) have shown that GMMs can capture molecular flexibility by modeling atomic motion directly,
making them highly suitable for heterogeneous reconstruction and downstream structural analysis.
However, existing GMM-based methods typically require initialization from an externally recon-
structed consensus map or even an atomic model. Without such guidance, random initialization
leads to unstable optimization and poor reconstruction quality. Our work addresses this limitation
by introducing a GMM-based reconstruction pipeline that can be stably trained from scratch.
2.2
GAUSSIAN SPLATTING
3DGS (Kerbl et al., 2023) is a recent differentiable rendering technique developed for real-time
novel view synthesis. It represents a 3D scene as a collection of anisotropic Gaussians and ren-
ders images via rasterization-based accumulation and alpha blending (Zwicker et al., 2002). While
3DGS achieves high visual fidelity in synthetic and real-world RGB datasets, as a volume rendering
method, it is not a physically accurate model of natural image formation (Huang et al., 2024).
Although the original 3DGS formulation is not physically consistent with natural image formation,
its volume rendering framework closely aligns with the cryo-EM imaging model, where each image
arises from an orthographic line integral of electrostatic potential modulated by the contrast transfer
function (CTF). Leveraging this alignment, we adapt splatting to cryo-EM by rederiving the pro-
jection of anisotropic Gaussians under cryo-EM physics, replacing heuristic alpha blending with
physically accurate line integrals and incorporating CTF modulation.
3
METHOD
3.1
OVERVIEW
Our goal is to achieve physically accurate and computationally efficient cryo-EM reconstruction by
leveraging a Gaussian Mixture Model (GMM). To this end, we propose cryoSplat, a differentiable
framework that represents the 3D electrostatic potential of a specimen as a set of anisotropic Gaus-
sians and directly simulates the cryo-EM image formation process in real space, faithfully adhering
to the physics of transmission electron microscopy.
Building upon recent progress in differentiable volume rendering, particularly the Gaussian splatting
framework by Kerbl et al. (2023), we adopt a tile-based rasterization strategy for scalable and effi-
cient computation. However, the original 3DGS formulation is not directly applicable to cryo-EM
due to several fundamental mismatches: (i) it employs perspective projection consistent with a pin-
hole camera model, in contrast to the orthographic projection in cryo-EM imaging; (ii) it prioritizes
photorealistic novel view synthesis over physical reconstruction accuracy; and (iii) its coordinate
system is misaligned with the Fourier slice theorem, which underpins cryo-EM reconstruction.
3

<!-- page 4 -->
(b) Image Formation
(a) Volume
Micrograph
Particle
(c) Fourier Slice Theorem
Fourier
Transform
3D Fourier Space
Slice
Viewing
Transformation
CTF
Modulation
Orthogonal
Projection
Fast  Differentiable Rasterization
(d) CryoSplat
Gaussian Mixture Model
Simulated Projection
Operation Flow
Gradient Flow
CTF
Projection
View
Voxelization
Figure 1: Cryo-EM reconstruction aims to recover a 3D volume (a) from a large set of 2D particle
images (b). (b) Purified biomolecules with random orientations are embedded in a thin layer of
vitreous ice. The electrostatic potential of the sample interacts with transmitted electrons, forming
a micrograph that contains 2D projections of the molecules. Individual particle images are ex-
tracted from the micrograph; they are extremely noisy and modulated by highly oscillatory CTFs.
(c) Fourier slice theorem: the 2D Fourier transform of a particle image corresponds to a central slice
of the 3D Fourier transform of the volume. (d) Overview of cryoSplat. An anisotropic GMM repre-
senting the 3D volume is transformed to the projection direction, orthogonally projected onto a 2D
image plane using a fast differentiable rasterizer, and modulated by the oscillatory CTF to simulate a
physically accurate projection. The GMM parameters are optimized via gradients propagated from
the discrepancy between the simulated and observed particle images. The resulting GMM can be
voxelized to obtain the final 3D volume.
To address these issues, cryoSplat introduces several key adaptations: (i) we replace heuristic alpha
blending with physically grounded line integrals to reflect the transmission nature of electron imag-
ing; (ii) we fix the normalization between 3D-to-2D transformation and apply consistent learning
rates across all parameters to ensure stable optimization; and (iii) we align the rasterization coordi-
nate system with the FFT grid, allowing accurate gradient propagation through CTF modulation.
These modifications collectively enable cryoSplat to perform stable, end-to-end differentiable re-
construction from raw cryo-EM particle images, starting from random initialization without relying
on externally provided volumes or atomic models.
3.2
IMAGE FORMATION
As shown in Fig. 1(b), electrons traverse a vitrified specimen, and the transmitted wavefronts un-
dergo phase shifts due to the specimen’s electrostatic potential (Singer & Sigworth, 2020). Under the
weak phase approximation, the phase shifts are linearly related to the 3D potential (volume), and the
image formed at the detector is a line integral (projection) of this potential along the beam direction,
further convolved with H : R2 →R, the point spread function (PSF) of the imaging system.
In homogeneous reconstruction, it is assumed that all particle images Y : R2 →R correspond to
identical copies of a single 3D volume V : R3 →R, and that any conformational or compositional
heterogeneity is negligible. Under this assumption, the image formation model can be expressed as:
Y (rx, ry) = H(rx, ry) ∗
Z
R
V (W ⊤r + t)drz + ϵ,
(1)
where r = [rx, ry, rz]⊤are the 3D Cartesian coordinates in real space, W ∈SO(3) is the 3D pose
of the particle, and t = [tx, ty, 0]⊤is the in-plane translation, accounting for imperfect centering
during particle cropping. The noise term ϵ is modeled as independent, zero-mean Gaussian noise.
4

<!-- page 5 -->
3.3
CRYOSPLAT
3.3.1
ANISOTROPIC GMM
Anisotropic GMMs are developed to represent the volume, which can be written in the form
V (r) =
N
X
i=1
AiGi(r),
(2)
where N denotes the Guassian count and Ai is the amplitude of the i-th normalized Gaussian Gi(r).
By substituting Eq. (1), we obtain the full forward process of cryoSplat. Specifically, we apply a
viewing transformation to align the GMM to the target orientation, orthographically project each
Gaussian along the z-axis to form a 2D image, and convolve the result with the PSF:
X(rx, ry) = H(rx, ry) ∗
N
X
i=1
Ai
Z
R
Gi(W ⊤r + t)drz.
(3)
Since the integral is linear, each Gaussian contributes independently to the final image. We thus
focus on a single Gaussian and omit the subscript i in the following discussion. A normalized 3D
Gaussian is defined as:
G(r|µ, Σ) =
1
(2π)
3
2 |Σ|
1
2
exp

−1
2(r −µ)⊤Σ−1(r −µ)

,
(4)
where µ ∈R3 and Σ ∈R3×3 denote the mean (position) and the covariance matrix (shape),
respectively. The determinant |Σ| ensures proper normalization. Following Kerbl et al. (2023), to
guarantee the positive semidefinite property, we construct the covariance matrix as:
Σ = RSS⊤R⊤,
(5)
where S = diag(s) is a diagonal scaling matrix and R ∈SO(3) is a rotation matrix. In our
implementation, we store the scaling vector s = [sx, sy, sz]⊤and parameterize R using a quaternion
q = [qw, qx, qy, qz]⊤. To ensure positivity and stable gradients during optimization, we apply a
softplus function to both the amplitude A and the scaling vector s. The quaternion q is normalized
to ensure it represents a valid rotation. Altogether, each anisotropic Gaussian is parameterized by
the 11-dimensional set {µx, µy, µz, sx, sy, sz, qw, qx, qy, qz, A}.
3.3.2
VIEWING TRANSFORMATION
The viewing transformation is the first step in simulating image formation, aligning each Gaus-
sian with a given projection direction. Since the parameters µ and Σ describe Gaussians in world
coordinates, we must transform them into the image-space coordinates before projection.
According to the derivation in Zwicker et al. (2002), applying an affine transformation to a Gaussian
results in another Gaussian with appropriately transformed parameters. In our case, the transforma-
tion consists of a rotation W ∈SO(3) and a 2D in-plane translation t ∈R3, leading to:
˙G(r| ˙µ, ˙Σ) = G(W ⊤r + t|µ, Σ),
(6)
where the transformed mean and covariance are given by ˙µ = W (µ −t) and ˙Σ = W ΣW ⊤.
3.3.3
ORTHOGONAL PROJECTION
The orthogonal projection closely aligns with the physical principles of cryo-EM. Mathematically,
it corresponds to a line integral of a 3D Gaussian along the z-axis, resulting in a 2D Gaussian,
hereafter referred to as a splat, ˜G(˜r|˜µ, ˜Σ):
˜G(˜r|˜µ, ˜Σ) =
Z
R
˙G(r| ˙µ, ˙Σ)drz.
(7)
This operation effectively integrates the 3D Gaussian along the projection axis, preserving its Gaus-
sian form in 2D. The resulting closed-form expression is:
˜G(˜r|˜µ, ˜Σ) =
1
2π| ˜Σ|
1
2 exp

−1
2(˜r −˜µ)⊤˜Σ−1(˜r −˜µ)

,
(8)
5

<!-- page 6 -->
where ˜r = [rx, ry]⊤denotes the 2D Cartesian coordinates in real space.
In prior works, such as 3DGS, the normalization term 1/(2π| ˜Σ|
1
2 ) is often omitted, as their primary
focus is on photorealistic novel view synthesis rather than the physical fidelity of the underlying
3D representation. However, in cryo-EM reconstruction, the ultimate goal is to recover the correct
3D volume. Omitting this view-dependent normalization introduces bias in amplitude and leads to
incorrect reconstructions. Therefore, unlike 3DGS, we retain the normalization term to preserve the
quantitative correctness of the model.
After projection, the final image is constructed by summing the weighted contributions of all splats
and applying the PSF:
X(rx, ry) = H(rx, ry) ∗
N
X
i=1
Ai ˜Gi(˜r).
(9)
3.3.4
FAST DIFFERENTIABLE RASTERIZATION
We adopt the efficient tile-based rasterization framework from Kerbl et al. (2023), which enables
scalable and differentiable processing of tens of thousands of Gaussians via per-tile accumulation.
Unlike 3DGS, which uses alpha blending for photorealistic rendering, we modify the rasterization to
directly sum contributions of splats, in accordance with the physical transmission model in cryo-EM.
For an image X ∈RD×D, the original 3DGS implementation places the continuous coordinate cen-
ter at [(D −1)/2, (D −1)/2]⊤, i.e., halfway between two discrete pixels. In contrast, FFT-based
image formation assumes the origin is located at the integer grid point [⌊D/2⌋, ⌊D/2⌋]⊤. To ensure
compatibility with FFT-based forward and backward modeling, we shift the rasterization coordinates
by half a pixel so that the image center aligns with the FFT grid. This alignment eliminates phase
inconsistencies and enables accurate electron projection simulation, while preserving the compu-
tational efficiency of the 3DGS architecture. Let X, Y ∈RD×D be the matrices representing the
GMM-based projection X and the observed image Y after rasterization, respectively.
3.3.5
LOSS FUNCTION
Unlike previous GMM-based methods that rely on specially designed losses with complex regular-
ization or constraints to ensure stable optimization, we adopt a much simpler formulation. Specifi-
cally, we directly apply the mean squared error (MSE) loss between the GMM-based projection X
and the observed image Y : L = ∥X −Y ∥2
2.
Despite its simplicity, this loss formulation leads to stable and fast convergence in practice, without
requiring additional regularization terms.
4
EXPERIMENT
4.1
EXPERIMENTAL SETTINGS
Datasets. We evaluate our method on four publicly available cryo-EM datasets from the Electron
Microscopy Public Image Archive (EMPIAR) (Iudin et al., 2016): EMPIAR-10028 (Pf80S ribo-
some) (Wong et al., 2014), EMPIAR-10049 (RAG complex) (Ru et al., 2015), EMPIAR-10076 (E.
coli LSU assembly) (Davis et al., 2016), and EMPIAR-10180 (pre-catalytic spliceosome) (Plaschka
et al., 2017). These datasets span a range of structural complexity and image quality, from rigid
assemblies with high contrast to highly heterogeneous macromolecular machines. For each dataset,
we use the provided particle images, consensus pose estimates, and CTF parameters. All reconstruc-
tions are performed under the homogeneous assumption.
Evaluation metrics. Since ground truth volumes are unavailable for real datasets, we follow stan-
dard practice and assess reconstruction quality using the gold standard Fourier Shell Correlation
(FSC) (Van Heel & Schatz, 2005). Each dataset is split evenly into two halves, and the method is
applied independently to each. Reconstructed volumes are compared in Fourier space by comput-
ing FSC as a function of spatial frequency, quantifying their consistency across frequency shells.
Resolution is defined as the spatial frequency where the FSC curve drops below the 0.143 threshold.
6

<!-- page 7 -->
Backprojection
CryoDRGN
CryoSplat
1/resolution (Å⁻¹)
Backprojection
CryoDRGN
CryoSplat, Epoch=1
CryoSplat, Epoch=2
CryoSplat, Epoch=3
CryoSplat, Epoch=4
CryoSplat, Epoch=5
EMPIAR-10180
FSC
EMPIAR-10076
FSC
FSC
EMPIAR-10049
FSC
EMPIAR-10028
3.80 Å 
4.19 Å 
3.88 Å 
3.30 Å 
4.51 Å 
4.26 Å 
4.62 Å 
3.80 Å 
3.80 Å 
4.00 Å 
4.07 Å 
2.49 Å 
Figure 2: Qualitative and quantitative comparison of voxel-based, neural, and GMM-based represen-
tations. (Left) Final 3D reconstructions on four real datasets visualized with ChimeraX (Pettersen
et al., 2021). (Right) FSC curves are plotted for quantitative evaluation. Gray dashed lines indicate
the standard resolution thresholds of 0.5 and 0.143, reported in Angstroms ( ˚A). CryoSplat consis-
tently achieves higher resolution across all datasets.
Backprojection
CryoDRGN
CryoSplat, N=2,048
CryoSplat, N=3,072
CryoSplat, N=5,120
CryoSplat, N=10,000
CryoSplat, N=30,000
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
EMPIAR-10180
FSC
1/resolution (Å⁻¹)
1/resolution (Å⁻¹)
1/resolution (Å⁻¹)
1/resolution (Å⁻¹)
Figure 3: Reconstruction performance with varying numbers of Gaussians. Increasing the number
improves accuracy and robustness, with 10,000 Gaussians generally sufficient for most datasets.
Implementation details. For all experiments, particle images from EMPIAR-10028, 10076, and
10180 are downsampled to 256 × 256, while EMPIAR-10049 is used at its original 192 × 192
resolution. Published particle translations are applied to the observed images via phase shifting
in Fourier space prior to reconstruction, rather than through the GMM viewing transform. We do
not apply any windowing to the observed particle images during preprocessing. The 3D volume is
defined over the domain [−E, E]3, and each 2D projection is assumed to lie within [−E, E]2 in the
image plane, where E = 0.5 defines the spatial extent. Gaussians are initialized with random means
7

<!-- page 8 -->
Frame per Second
Backprojection, D=192
Backprojection, D=256
CryoDRGN, D=192
CryoDRGN, D=256
CryoSplat, D=192
CryoSplat, D=256
Number of Gaussians
Figure 4: Runtime efficiency across reconstruction methods at different resolutions (D = 192 and
D = 256). Frame rates (FPS) are measured under increasing numbers of Gaussians (log-scaled).
µ ∼N(0, 0.0752I), isotropic scales sx =sy =sz =0.0075, identity quaternion q=[1, 0, 0, 0]T , and
amplitude A = 1/(2N), where N is the number of Gaussians. All parameters are trainable. We use
Adam (Kingma & Ba, 2014) with batch size 1, learning rate 0.001, and exponential decay (γ = 0.1)
at each epoch. All GMMs are trained for 5 epochs. For the voxel-based backprojection baseline,
we use the implementation provided by Zhong et al. (2021a). For neural representation learning,
we follow cryoDRGN’s default configuration: three 1,024-node layers, trained for 50 epochs. All
experiments are run on a single NVIDIA GeForce RTX 3090.
4.2
EVALUATION ON REAL DATASETS
We evaluate the performance of different volume representations on real cryo-EM datasets under
a homogeneous reconstruction setting. To ensure a fair comparison focused solely on the choice
of volume representation, all methods reconstruct consensus maps using the same set of published
particle poses, without performing pose estimation. Our evaluation focuses on two aspects: (i)
the ability to reconstruct high-resolution consensus maps, and (ii) robustness to noise and imper-
fect pose assignments. Since related methods (Zhong et al., 2021a;b; Levy et al., 2022a;b; 2025)
adopt cryoDRGN’s neural field implementation and differ mainly in pose estimation, we focus our
comparison on cryoDRGN. Accordingly, we evaluate three representative approaches: voxel-based
backprojection, the neural representation method cryoDRGN (Zhong et al., 2021a), and our pro-
posed GMM-based method cryoSplat. Visualizations of the reconstructed volumes are shown in
Fig. 2, and spatial resolution is quantified using gold-standard FSC curves. Each volume of cryoS-
plat is represented using 30,000 Gaussians.
The Pf80S ribosome (EMPIAR-10028) is relatively easy to reconstruct due to its high-contrast im-
ages and structurally stable particles. All methods achieve high-resolution results (3.80 ˚A) and
strong FSC agreement across the spectrum. cryoDRGN yields slightly higher FSC values at inter-
mediate frequencies, while cryoSplat outperforms all baselines at high spatial frequencies, demon-
strating its ability to recover fine structural details.
The RAG complex (EMPIAR-10049) poses greater challenges due to imperfect published poses
and flexible regions such as the DNA elements and the nonamer binding domain (NBD), indicated
by arrows. CryoSplat significantly outperforms both backprojection and cryoDRGN, achieving a
resolution of 2.49 ˚A. Unlike the baselines, cryoSplat reconstructs the DNA elements and the NBD
with minimal artifacts. Its FSC curve remains consistently above those of other methods across all
spatial frequencies, highlighting its robustness to pose inaccuracies and structural variability.
This LSU assembly dataset (EMPIAR-10076) contains substantial compositional and conforma-
tional heterogeneity, making consensus reconstruction particularly challenging. Both FSC analysis
and visualization show that cryoSplat is more resilient under such conditions, achieving a resolution
of 3.30 ˚A with fewer artifacts than voxel-based or neural methods, as indicated by the red circle.
8

<!-- page 9 -->
The spliceosome dataset (EMPIAR-10180) features large-scale motions of the SF3b indicated by
the red circle, making consensus reconstruction particularly challenging. The reconstructions from
backprojection and cryoDRGN show pronounced artifacts in this region, while cryoSplat is more
robust to such motions and achieves a resolution of 4.26 ˚A. FSC analysis further confirms that
cryoSplat significantly outperforms the baselines across the frequency range.
CryoSplat consistently converges within 5 epochs, with FSC curves from the 4th and 5th epochs
tightly overlapping, indicating stable optimization and improved generalization.
4.3
ABLATION STUDIES
This section reports ablation studies of our approach. More results can be found in Appendix E.
Number of Gaussians. Fig. 3 shows the FSC curves for cryoSplat with varying numbers of Gaus-
sians. In general, increasing the number of Gaussians leads to improved FSC, as a denser GMM pro-
vides greater representational capacity. While cryoSplat performs well on most datasets, its relative
performance varies due to differences in structural complexity and dataset-specific challenges. On
EMPIAR-10028, cryoSplat reaches a resolution of 3.8 ˚A under all settings. While configurations
with fewer than 10,000 Gaussians exhibit lower FSC values than cryoDRGN and backprojection
across most frequencies, the curves intersect at the highest frequency, indicating comparable final
resolution. For EMPIAR-10049, all cryoSplat settings significantly outperform both backprojection
(4.00 ˚A) and cryoDRGN (4.07 ˚A), achieving a resolution of 2.49 ˚A. Moreover, the FSC curves of all
cryoSplat variants remain consistently above those of the two baselines across the entire frequency
range. For EMPIAR-10076, the 30,000-Gaussian model clearly outperforms other settings; even
with fewer Gaussians, cryoSplat still surpasses the baselines, reaching 3.3 ˚A. For EMPIAR-10180,
the models with 10,000 and 30,000 Gaussians achieve the best FSC, reaching 4.3 ˚A, while sparser
GMMs remain competitive at high spatial frequencies. Overall, we find that using 10,000 Gaussians
is sufficient to reconstruct high-resolution volumes that consistently outperform baseline methods
across most datasets. Associated qualitative comparisons are provided in Appendix E.
Runtime efficiency. We compare the runtime efficiency of cryoSplat with other representation
baselines, as shown in Fig. 4. Backprojection is the fastest, as it generates projections by directly
indexing and interpolating from a dense voxel grid, but it is incompatible with modern non-linear
heterogeneous analysis techniques. For such tasks, neural representations and GMMs offer greater
flexibility. Under commonly used settings in heterogeneous reconstruction (e.g., 2,048–3,072 Gaus-
sians (Chen & Ludtke, 2021; Chen et al., 2023a;b)), cryoSplat achieves 2–3× higher FPS than cry-
oDRGN. Moreover, cryoSplat typically converges within 5 epochs, compared to 50 epochs required
by cryoDRGN, providing an overall speedup up to 30×. As discussed above, using 10,000 Gaus-
sians allows cryoSplat to consistently outperform baseline methods across most datasets, while still
maintaining a higher FPS than cryoDRGN. Even with an extremely large number of Gaussians (e.g.,
30,000), cryoSplat provides reasonable runtime performance for orthogonal projection operations.
Overall, as shown in Fig. 4, cryoSplat demonstrates sub-linear time complexity with respect to the
number of Gaussians, offering a favorable trade-off between accuracy and efficiency.
5
CONCLUSION
We present cryoSplat, a novel GMM-based framework that integrates Gaussian splatting with the
physics of cryo-EM image formation. CryoSplat enables stable and efficient homogeneous recon-
struction directly from raw cryo-EM particle images, starting from random initialization without
relying on consensus volumes. Experimental results on real datasets demonstrate the effectiveness,
robustness, and faster convergence of cryoSplat compared to representative baselines.
Limitation and future work. While our current method assumes known poses and thus does not
qualify as an ab initio approach, cryoSplat establishes a principled foundation for future GMM-
based methods that aim to tackle ab initio and heterogeneous reconstruction. We believe cryoSplat
provides a missing piece in the broader goal of integrating GMMs into scalable and self-contained
cryo-EM reconstruction pipelines. These directions are left for future work.
9

<!-- page 10 -->
REFERENCES
Tamir Bendory, Alberto Bartesaghi, and Amit Singer. Single-particle cryo-electron microscopy:
Mathematical theory, computational challenges, and opportunities. IEEE signal processing mag-
azine, 37(2):58–76, 2020.
Tristan Bepler, Kotaro Kelley, Alex J Noble, and Bonnie Berger.
Topaz-denoise: general deep
denoising models for cryoem and cryoet. Nature communications, 11(1):5208, 2020.
Ronald N Bracewell. Strip integration in radio astronomy. Australian Journal of Physics, 9(2):
198–217, 1956.
Muyuan Chen. Building molecular model series from heterogeneous cryoem structures using gaus-
sian mixture models and deep neural networks. Communications Biology, 8(1):798, 2025.
Muyuan Chen and Steven J Ludtke. Deep learning-based mixed-dimensional Gaussian mixture
model for characterizing variability in cryo-EM. Nature Methods, 18(8):930–936, 2021. URL
https://doi.org/10.1038/s41592-021-01220-5.
Muyuan Chen, Michael F. Schmid, and Wah Chiu. Improving resolution and resolvability of single
particle cryoem structures using Gaussian mixture models. Nature Methods, 21:37–40, 2023a.
Muyuan Chen, Bogdan Toader, and Lederman Roy.
Integrating molecular models into cryoem
heterogeneity analysis using scalable high-resolution deep Gaussian mixture models. Journal of
Molecular Biology, 435(9):168014, 2023b.
Joseph H Davis, Yong Zi Tan, Bridget Carragher, Clinton S Potter, Dmitry Lyumkis, and James R
Williamson. Modular assembly of the bacterial large ribosomal subunit. Cell, 167(6):1610–1622,
2016.
J Andrew Grant and BT Pickup. A gaussian description of molecular shape. The Journal of Physical
Chemistry, 99(11):3503–3510, 1995.
J Andrew Grant, Maria A Gallardo, and Barry T Pickup. A fast method of molecular shape compari-
son: A simple application of a gaussian description of molecular shape. Journal of computational
chemistry, 17(14):1653–1666, 1996.
Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting
for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pp.
1–11, 2024.
Andrii Iudin, Paul K Korir, Jos´e Salavert-Torres, Gerard J Kleywegt, and Ardan Patwardhan. Em-
piar: a public archive for raw electron microscopy image data. Nature methods, 13(5):387–388,
2016.
Takeshi Kawabata. Multiple subunit fitting into a low-resolution density map of a macromolecular
complex using a gaussian mixture model. Biophysical journal, 95(10):4643–4658, 2008.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3D Gaussian splat-
ting for real-time radiance field rendering. ACM Transactions on Graphics (Proc. SIGGRAPH),
2023.
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. Int. Conf. on
Learning Representations (ICLR), 2014.
Werner K¨uhlbrandt. The Resolution Revolution: Advances in detector technology and image pro-
cessing are yielding high-resolution electron cryo-microscopy structures of biomolecules. Sci-
ence, 343(6178):1443–1444, 2014.
Axel Levy, Fr´ed´eric Poitevin, Julien Martel, Youssef Nashed, Ariana Peck, Nina Miolane, Daniel
Ratner, Mike Dunne, and Gordon Wetzstein. Cryoai: Amortized inference of poses for ab initio
reconstruction of 3d molecular volumes from real cryo-em images. In European Conference on
Computer Vision, pp. 540–557. Springer, 2022a.
10

<!-- page 11 -->
Axel Levy, Gordon Wetzstein, Julien NP Martel, Frederic Poitevin, and Ellen Zhong. Amortized
inference for heterogeneous reconstruction in cryo-EM. Advances in Neural Information Pro-
cessing Systems, 35:13038–13049, 2022b.
Axel Levy, Rishwanth Raghu, J Ryan Feathers, Michal Grzadkowski, Frederic Poitevin, Francesca
Vallese, Oliver B Clarke, Gordon Wetzstein, and Ellen D Zhong. Cryodrgn-ai: Neural ab initio
reconstruction of challenging cryo-em and cryo-et datasets. bioRxiv, pp. 2024–05, 2025.
Yilai Li, Yi Zhou, Jing Yuan, Fei Ye, and Quanquan Gu. Cryostar: leveraging structural priors and
constraints for cryo-em heterogeneous reconstruction. Nature Methods, 21(12):2318–2326, 2024.
Eva Nogales. The development of cryo-em into a mainstream structural biology technique. Nature
methods, 13(1):24–27, 2016.
Eric F Pettersen, Thomas D Goddard, Conrad C Huang, Elaine C Meng, Gregory S Couch, Tris-
tan I Croll, John H Morris, and Thomas E Ferrin. Ucsf chimerax: Structure visualization for
researchers, educators, and developers. Protein science, 30(1):70–82, 2021.
Clemens Plaschka, Pei-Chun Lin, and Kiyoshi Nagai. Structure of a pre-catalytic spliceosome.
Nature, 546(7660):617–621, 2017.
Ali Punjani, John L Rubinstein, David J Fleet, and Marcus A Brubaker. CryoSPARC: Algorithms
for rapid unsupervised cryo-em structure determination. Nature Methods, 14:290–296, 2017.
Jean-Paul Renaud, Ashwin Chari, Claudio Ciferri, Wen-ti Liu, Herv´e-William R´emigy, Holger
Stark, and Christian Wiesmann.
Cryo-em in drug discovery: achievements, limitations and
prospects. Nature reviews Drug discovery, 17(7):471–492, 2018.
Heng Ru, Melissa G Chambers, Tian-Min Fu, Alexander B Tong, Maofu Liao, and Hao Wu. Molec-
ular mechanism of v (d) j recombination from synaptic rag1-rag2 complex structures. Cell, 163
(5):1138–1152, 2015.
S. H. Scheres. Processing of structurally heterogeneous cryo-em data in RELION. Methods Enzy-
mol., 579:125–157, 2016a.
Sjors H. W. Scheres. RELION: Implementation of a Bayesian approach to cryo-em structure deter-
mination. Journal of Structural Biology, 180(3):519 – 530, 2012.
Sjors H W Scheres. Processing of structurally heterogeneous cryo-em data in RELION. In R A
Crowther (ed.), The Resolution Revolution: Recent Advances In cryoEM, volume 579 of Meth-
ods in Enzymology, pp. 125–157. Academic Press, 2016b. doi: https://doi.org/10.1016/bs.mie.
2016.04.012.
URL https://www.sciencedirect.com/science/article/pii/
S0076687916300301.
Johannes Schwab, Dari Kimanius, Alister Burt, Tom Dendooven, and Sjors H. W. Scheres. Dy-
naMight: Estimating molecular motions with improved reconstruction from cryo-em images. Na-
ture Methods, 21:1855–1862, 2024.
Shayan Shekarforoush, David B Lindell, Marcus A Brubaker, and David J Fleet. CryoSPIN: Im-
proving ab-initio cryo-em reconstruction with semi-amortized pose inference. NeurIPS, 2024.
Amit Singer and Fred J Sigworth.
Computational methods for single-particle electron cryomi-
croscopy. Annual review of biomedical data science, 3(1):163–190, 2020.
G. Tang, L. Peng, P. R. Baldwin, D. S. Mann, W. Jiang, I. Rees, and S. J. Ludtke. Eman2: an
extensible image processing suite for electron microscopy. J. Struct. Biol., 157:38–46, 2007.
Marin Van Heel and Michael Schatz. Fourier shell correlation threshold criteria. Journal of struc-
tural biology, 151(3):250–262, 2005.
Wilson Wong, Xiao-chen Bai, Alan Brown, Israel S Fernandez, Eric Hanssen, Melanie Condron,
Yan Hong Tan, Jake Baum, and Sjors HW Scheres. Cryo-em structure of the plasmodium falci-
parum 80s ribosome bound to the anti-protozoan drug emetine. elife, 3:e03080, 2014.
11

<!-- page 12 -->
Ellen D. Zhong, Tristan Bepler, Bonnie Berger, and Joseph H. Davis. CryoDRGN: Reconstruction
of heterogeneous cryo-em structures using neural networks. Nature Methods, 18:176–185, 2021a.
Ellen D Zhong, Adam Lerer, Joseph H Davis, and Bonnie Berger. CryoDRGN2: Ab initio neural
reconstruction of 3d protein structures from real cryo-EM images. In IEEE International Confer-
ence on Computer Vision, pp. 4066–4075, 2021b.
Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Ewa splatting. IEEE
Transactions on Visualization and Computer Graphics, 8(3):223–238, 2002.
12

<!-- page 13 -->
APPENDIX
A
DETAILS OF METHOD
A.1
REAL SPACE RECONSTRUCTION
According to the Fourier slice theorem (Bracewell, 1956), illustrated in Fig. 1(c), the 2D Fourier
transform of a projection corresponds to a central slice of the 3D Fourier transform of the volume,
orthogonal to the projection direction and passing through the origin. Based on this property, an
alternative and widely adopted formulation models reconstruction directly in the Fourier domain,
where the image formation model becomes:
bY (kx, ky) = bH(kx, ky) · bV (W ⊤k) · e−2πik⊤t +bϵ,
(10)
where k = [kx, ky, 0]⊤denotes the Cartesian coordinates in Fourier space, and the 2D spectrum bY ,
the CTF bH and the 3D spectrum bV denote the Fourier transform of Y , H and V , respectively. The
noise term bϵ is similarly modeled as independent, zero-mean Gaussian noise in the Fourier domain.
In this work, departing from most existing approaches that adopt Eq. (10), we instead build our
pipeline on Eq. (1), performing homogeneous reconstruction directly in real space.
A key reason we choose to operate in real space is that it allows us to fully exploit the fast rasteriza-
tion strategy from 3DGS. In high-resolution reconstructions, individual Gaussians in real space have
small spatial scales and affect only a few nearby tiles, as shown in Fig. 5(a). This locality means
that each GPU thread is responsible for a single pixel and only needs to process a small subset of all
Gaussians. In contrast, Gaussians in Fourier space become broad as resolution increases, leading to
near-global support. As a result, each pixel in the frequency domain must aggregate contributions
from nearly all Gaussians, making fast rendering impractical.
Real Space
Fourier Space
FFT-based
Image-centered
(b)
(a)
Figure 5: Rasterization details. (a) A Gaussian with small spatial scales in real space during high-
resolution reconstruction overlaps at most four tiles, while in Fourier space it exhibits nearly global
support. Tile boundaries are indicated by lines. (b) For a 4 × 4 image, the origin of the continu-
ous coordinate system during rasterization is defined differently: FFT-based coordinates place it at
[2, 2]⊤, whereas image-centered coordinates place it at [1.5, 1.5]⊤. The origin is marked by a dot.
Image-centered coordinates induce a phase error of −πk/D, up to ±π/2 at the Nyquist frequency,
degrading reconstruction at high frequency. The effect diminishes with larger D but never vanishes.
A.2
COMPUTATIONAL DETAILS
Section 3.3.3 describes how a 3D Gaussian is projected along the z-axis to form a splat. As discussed
in Zwicker et al. (2002), the splat can be computed analytically by removing the z-axis components
from the mean and covariance:







˙µ = [ ˙µx, ˙µy, ˙µz]⊤⇒[ ˙µx, ˙µy]⊤= ˜µ
˙Σ =


˙σxx
˙σxy
˙σxz
˙σxy
˙σyy
˙σyz
˙σxz
˙σyz
˙σzz

⇒
 ˙σxx
˙σxy
˙σxy
˙σyy

= ˜Σ
(11)
thereby enabling an efficient computation of Eq. (7).
13

<!-- page 14 -->
As discussed in Sec. 3.3.4 and shown in Fig. 5(b), when a continuous image X : R2 →R is
rasterized onto pixels X ∈RD×D, the origin of the continuous coordinate system should be aligned
with [⌊D/2⌋, ⌊D/2⌋]⊤to match the FFT-based coordinate convention used in cryo-EM. Formally,
Xi,j = X

(j −⌊D
2 ⌋)2E
D , −(i −⌊D
2 ⌋)2E
D

,
(12)
where Xi,j denotes (i, j)-th entry of matrix X. Note that the row and column indices correspond to
the y- and x-axes, respectively, with the y-axis flipped during this rasterization.
In practice, since the PSF corresponds to a large convolution kernel, we apply the contrast transfer
function (CTF) in the Fourier domain after rasterization for efficiency:
Xi,j = F−1
 
c
H ⊙F
 N
X
i=1
Ai ˜Gi
 (j −⌊D
2 ⌋) 2E
D , −(i −⌊D
2 ⌋) 2E
D

!!
,
(13)
where F(·) and F−1(·) denote the Fourier and inverse Fourier transform operators, respectively. c
H
is the rasterized CTF bH and ⊙denotes element-wise (Hadamard) product.
Before deriving the gradients, we first define
Q(rx, ry) =
N
X
i=1
Ai ˜Gi(˜r),
(14)
which is the pre-rasterization continuous image. For clarity, we omit the Gaussian index i in the
following derivations, as the gradients are computed independently for each Gaussian. We denote
by P the set of 2D coordinates corresponding to the centers of rasterized pixels. When ˜r ∈P, the
coordinate ˜r = [rx, ry]⊤refers to a discrete sampling location in the image plane. The gradients
used in the backward pass can be summarized as































∂L
∂A =
X
˜r∈P
∂L
∂Q(˜r)
∂Q(˜r)
∂A
∇µL =
X
˜r∈P
∂L
∂Q(˜r)∇µQ(˜r)
∇sL =
X
˜r∈P
∂L
∂Q(˜r)∇ΣQ(˜r) ◦∂Σ
∂s
∇qL =
X
˜r∈P
∂L
∂Q(˜r)∇ΣQ(˜r) ◦∂Σ
∂q
(15)
where ◦denotes the composition of Jacobian operators (chain rule). The derivation of gradients with
respect to the amplitude A and mean µ is trivial, which can be given directly by
∂Q
∂A = ˜G(˜r),
(16)
and
(
∇˜µQ = A ˜G(˜r) ˜Σ−1(˜r −˜µ)
∇µQ = W[∇˜µQ⊤0]⊤
(17)
where [∇˜µQ⊤0]⊤embeds the 2D gradient into 3D space by padding the z-component with zero.
For completeness, we provide the derivation of the covariance gradients ∇ΣQ, noting that our for-
mulation retains the normalization term, which is omitted in 3DGS (Kerbl et al., 2023). Remember
˜G(˜r|˜µ, ˜Σ) =
1
2π| ˜Σ|
1
2 exp(−1
2(˜r −˜µ)⊤˜Σ−1(˜r −˜µ))
= | ˜Σ−1|
1
2
2π
exp(−1
2(˜r −˜µ)⊤˜Σ−1(˜r −˜µ)).
(18)
14

<!-- page 15 -->
We can first compute
∇˜Σ−1Q = A exp(−1
2(˜r −˜µ)⊤˜Σ−1(˜r −˜µ)) 1
4π | ˜Σ−1|−1
2 | ˜Σ−1| ˜Σ⊤
+ A| ˜Σ−1|
1
2
2π
exp(−1
2(˜r −˜µ)⊤˜Σ−1(˜r −˜µ))(−1
2(˜r −˜µ)(˜r −˜µ)⊤)
= 1
2A ˜G(˜r)( ˜Σ −(˜r −˜µ)(˜r −˜µ)⊤),
(19)
and then ∇˜ΣQ = −˜Σ−⊤∇˜Σ−1Q ˜Σ−⊤. Finally,
∇˙ΣQ =
"∇˜ΣQ
0
0⊤
0
#
.
(20)
The subsequent derivations of ∇sL and ∇qL follow exactly the formulation in Kerbl et al. (2023).
B
DATASET DETAILS
We provide detailed statistics and characteristics of the cryo-EM datasets used in our experiments:
• EMPIAR-10028 (Plasmodium falciparum 80S (Pf80S) ribosome) (Wong et al., 2014):
105,247 particle images of size 360 × 360 pixels at a sampling rate of 1.34 ˚A/pixel. This
is a widely used benchmark with high-contrast images and a static structure.
• EMPIAR-10049 (RAG1-RAG2 complex) (Ru et al., 2015): 108,544 particles of size 192×
192 pixels at 1.23 ˚A/pixel. This dataset is considered more challenging due to its lower
contrast and flexibility in some regions.
• EMPIAR-10076 (E. coli large ribosomal subunit undergoing (LSU) assembly) (Davis et al.,
2016): 131,899 particles of size 320 × 320 pixels at 1.31 ˚A/pixel. This dataset contains
substantial conformational and compositional heterogeneity, which poses a challenge to
homogeneous modeling.
• EMPIAR-10180 (Pre-catalytic spliceosome) (Plaschka et al., 2017): 327,490 particles of
size 320×320 pixels at 1.69 ˚A/pixel. It samples a continuum of conformations, particularly
involving large-scale motions of the SF3b subcomplex.
• Synthetic 80S ribosome: We construct a synthetic dataset of the 80S ribosome with 100,000
particles using Relion (Scheres, 2016a), following the protocol of Levy et al. (2022a). The
electron scattering potential is derived in ChimeraX (Pettersen et al., 2021) at a resolution
of 6.0 ˚A/pixel, based on two atomic models: the small subunit (PDB 3J7A) and the large
subunit (PDB 3J79) (Wong et al., 2014). Each particle image is 128 × 128 pixels with a
pixel size of 3.77 ˚A/pixel. Orientations are uniformly sampled over SO(3), and all images
are centered without translations. Defocus values for the CTF are randomly drawn from
log-normal distributions following Levy et al. (2022a), and zero-mean white Gaussian noise
with varying signal-to-noise ratios (SNRs) is added.
C
FOURIER SHELL CORRELATION
To evaluate reconstruction quality on real datasets without ground truth volumes, we adopt the gold
standard Fourier Shell Correlation (FSC) (Van Heel & Schatz, 2005), following established proto-
cols. Each dataset is randomly split into two halves, and the reconstruction algorithm is applied
independently to each subset. Let the resulting volumes be bVa(k) and bVb(k), representing their
Fourier transforms. The FSC is computed as a function of frequency k using the following formula:
FSC(k) =
P
∥k∥2=k
bVa(k) · bVb(k)∗
s
P
∥k∥2=k
|bVa(k)|2

P
∥k∥2=k
|bVb(k)|2
.
(21)
This metric quantifies the correlation between two independently reconstructed volumes within con-
centric shells in Fourier space. The spatial resolution is defined as the frequency where the FSC
curve drops below the 0.143 threshold, indicating the limit of reproducible structural detail.
15

<!-- page 16 -->
D
MORE IMPLEMENTATION DETAILS
D.1
INTUITION BEHIND INITIALIZATION
The values used in initialization are fixed but grounded in straightforward statistical intuition. We
observe that most particles are concentrated within a spherical region of radius E/2, where E = 0.5
defines the spatial extent as mentioned in Sec. 4.1. To reflect this prior and accelerate convergence,
we initialize the Gaussian means within this region.
Moreover, based on the three-sigma rule for Gaussian distributions N(µ, σ2), where 99.7% of sam-
ples fall within [µ −3σ, µ + 3σ], we obtain σ = E/6 from 3σ = E/2. To slightly tighten the
spread, we apply a scaling factor and use σ = 0.9 · E/6 = 0.075 to initialize the means. The initial
scale of each Gaussian is set to 0.1 × 0.075 = 0.0075, encouraging localized support. Finally, to
maintain consistent overall energy across varying numbers of Gaussians, we initialize the amplitude
as A = 1/(2N), where N is the total number of Gaussians.
D.2
INTUITION BEHIND LEARNING RATE
In the original 3DGS (Kerbl et al., 2023), different learning rates are assigned to different types of
Gaussian parameters (means, scales, rotations, opacities). While this works well in novel view
synthesis, it introduces instability in cryo-EM reconstruction. Let the full parameter vector be
θ = [µx, µy, µz, sx, sy, sz, qw, qx, qy, qz, A]⊤. In gradient descent optimization, the direction of
parameter updates is determined by the gradient ∇θL. Unequal learning rates distort this direction
by scaling different components unequally, which can lead to divergence. We observe that such prac-
tice causes Gaussians to spread uncontrollably in early iterations and finally diverge. To avoid this,
we adopt a single unified learning rate across all parameter types, preserving the intended descent
direction and ensuring stable convergence.
D.3
OPTIMIZATION ALGORITHM
Our optimization algorithm is summarized in Algorithm 1. Unlike Kerbl et al. (2023), which uses
gradient magnitude as the criterion for splitting and cloning Gaussians, we observe that gradients
are not a reliable indicator for densification in cryo-EM reconstruction. Furthermore, elaborate
densification schemes are generally unnecessary, as our method seldom suffers from significant
local minima owing to its close consistency with cryo-EM imaging physics. Nevertheless, we retain
a simple densification option to balance efficiency and resolution: fewer Gaussians enable faster
training, whereas more Gaussians yield higher-resolution reconstructions, as demonstrated in Fig. 4.
D.4
DETAILS OF THE RASTERIZER
The details of the rasterizer are summarized in Algorithm 2. We follow the tile-based rasterization
framework of Kerbl et al. (2023), where the output image is divided into 16×16 pixel tiles, and each
splat is instantiated in every tile it overlaps. The splat instances are then assigned keys for sorting,
after which each tile can be processed efficiently by locating the corresponding ranges in the sorted
list. Since pixels are computed in parallel, the runtime is primarily determined by the maximum
number of Gaussians within any tile. For more details, we refer the reader to Kerbl et al. (2023).
E
ADDITIONAL RESULTS
E.1
MULTIMEDIA RESULTS
We provide videos of rotating reconstructed volumes in the supplementary material for all methods
and datasets to facilitate visual comparison.
16

<!-- page 17 -->
Algorithm 1 Optimization and Densification
N: number of Gaussians
D: side length of the observed particle images
Θ ←InitAttributes(N)
▷Positions, Scales, Quaternions, Amplitudes
i ←0
▷Epoch Count
while not converged do
for (Y , W , t, c
H) in Dataloader() do
▷Observed Image, Rotation, Translation, CTF
Y ←FourierShift(Y , t)
▷Center Alignment
Q ←Rasterize(Θ, W , D)
▷Algorithm 2
X ←ApplyCTF(Q, c
H)
▷Apply CTF
L ←Loss(X, Y )
▷Loss
Θ ←Adam(∇L)
▷Backprop and Step
end for
if IsDoubleGaussians(i) then
▷(Optional) Densification
for all Gaussian(µ, s, q, A) in Θ do
SplitGaussian(µ, s, q, A)
end for
end if
i ←i + 1
end while
Algorithm 2 CUDA-accelerated Rasterization
Θ: Gaussian parameters
W : viewing transformation matrix
D: side length of the observed particle images
function Rasterize(Θ, W , D)
µ, Σ, A ←BuildGaussians(Θ)
˙µ, ˙Σ ←ViewingTransform(µ, Σ, W )
▷Viewing Transformation
˜µ, ˜Σ ←Projection( ˙µ, ˙Σ)
▷Orthogonal Projection
T ←CreateTiles(D)
▷Tile Count
L, K ←DuplicateWithKeys(˜µ, T)
▷List of Indices and Keys
SortByKeys(K, L)
R ←IdentifyTileRanges(T, K)
Q ←0
▷Init Canvas
for all Tile t in Q do
for all Pixel p in t do
r ←GetTileRange(R, t)
Q(p) ←SumSplats(p, L, r, K, ˜µ, ˜Σ, A)
end for
end for
return Q
end function
E.2
MEMORY USAGE
Tab. 1 compares GPU memory usage across different reconstruction methods. CryoDRGN (Zhong
et al., 2021a) exhibits the highest memory footprint, exceeding 2.5 GiB at D = 192 and approaching
5 GiB at D = 256, primarily due to its deep neural decoder and a larger batch size of 8. Interestingly,
backprojection consumes more memory at D = 192 than at D = 256, which may be attributed to
implementation-specific factors such as padding overhead or kernel-level optimizations that favor
power-of-two dimensions. This anomaly appears method-specific and does not reflect a general
trend. In contrast, CryoSplat demonstrates consistently low memory usage across all configurations.
Even with as many as 30,000 Gaussians, CryoSplat maintains a memory footprint below 380 MiB,
with negligible variation across resolutions. This efficiency underscores the scalability and suitabil-
ity of CryoSplat for large-scale or memory-constrained cryo-EM reconstruction scenarios.
17

<!-- page 18 -->
Table 1: GPU memory usage across reconstruction methods at resolutions (D = 192, D = 256).
Methods
# Params
Settings
Batch Size
GPU Mem. (MiB)
D = 192
D = 256
Backprojection
(D + 1)3
—
1
508
396
CryoDRGN (Zhong et al., 2021a)
(6 · ⌊D/2⌋+ L + 3) · C
C = 1,024
1
680
1,008
+L · C2 + 2
L = 3
8
2,560
4,906
CryoSplat (Ours)
11 · N
N = 2,048
1
344
346
N = 3,072
344
348
N = 5,120
346
348
N = 10,000
348
350
N = 30,000
376
378
Isotropic
Anisotropic
1/resolution (Å⁻¹)
FSC
Anisotropic
Isotropic
1/resolution (Å⁻¹)
FSC
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
1/resolution (Å⁻¹)
FSC
Isotropic
Anisotropic
EMPIAR-10180
1/resolution (Å⁻¹)
FSC
Isotropic
Anisotropic
Figure 6: Qualitative and quantitative comparison of isotropic and anisotropic GMMs (N = 30,000)
on two datasets (EMPIAR-10028 and EMPIAR-10049). FSC curves show that anisotropic Gaus-
sians consistently achieve higher correlations across spatial frequencies, indicating improved recon-
struction accuracy. Volume visualizations further reveal that anisotropic GMMs better recover fine
structural details and elongated features, whereas isotropic Gaussians tend to fragment such regions.
E.3
ISOTROPIC VS. ANISOTROPIC
CryoSplat represents 3D volumes using anisotropic Gaussians while remaining fully compatible
with the isotropic formulation widely adopted in prior works (Chen & Ludtke, 2021; Chen et al.,
2023a;b; Schwab et al., 2024; Chen, 2025). When the scaling is isotropic, sx = sy = sz = σ, the
anisotropic Gaussian exactly reduces to the standard isotropic form:
G(r|µ, σ) =
1
(2π)
3
2 σ3 exp(−∥r −µ∥2
2
2σ2
),
(22)
allowing direct integration into existing isotropic GMM-based pipelines.
We investigate the impact of isotropic versus anisotropic Gaussians on reconstruction quality. As
shown in Fig. 6, anisotropic GMMs achieve higher FSC scores across spatial frequencies and pro-
duce sharper, more detailed structures. Subjectively, isotropic Gaussians struggle to capture elon-
gated features and are often captured by noise, which may contribute to the unstable convergence
from random initialization reported in previous methods. These results highlight the improved rep-
resentational capacity and reconstruction robustness enabled by anisotropic modeling.
E.4
NUMBER OF GAUSSIANS
We present visual comparisons of reconstruction results using different numbers of Gaussians. As
shown in Fig. 7, increasing the number of components yields progressively sharper and more de-
tailed structures. These qualitative observations align with the quantitative improvements in FSC
curves reported in Fig. 3. Red arrows highlight representative regions where the differences in re-
construction quality are especially pronounced, facilitating direct visual comparison across settings.
18

<!-- page 19 -->
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
EMPIAR-10180
N=2,048
N=3,072
N=5,120
N=10,000
N=30,000
Figure 7: Qualitative evaluation of reconstruction performance with different numbers of Gaussians.
Increasing the number of Gaussians leads to visibly improved reconstructions, with finer structural
details and enhanced sharpness. Red arrows mark representative regions that highlight the qualitative
differences for clearer comparison across settings.
E.5
SIGNAL-TO-NOISE RATIO
Noise Free
0 dB
-10 dB
-20 dB
-30 dB
GT
Example Particles
Signal-to-Noise Ratio
1/resolution (Å⁻¹)
FSC
Noise Free
30 dB
20 dB
10 dB
0 dB
-10 dB
-15 dB
-20 dB
-25 dB
-30 dB
Figure 8: Reconstruction performance under varying SNRs. (top) Ground-truth (GT) and recon-
structed volumes at different SNR levels. (bottom) Example synthetic particle images correspond-
ing to each SNR. (right) FSC curves between GT and reconstructed volumes across SNRs.
We study the effect of SNR levels on cryoSplat with 5,120 Gaussians using the synthetic 80S dataset
described in Sec. B. Figure 8 shows example synthetic particles, reconstructed volumes, and FSC
curves under varying SNRs. FSCs are computed between the ground truth (GT) and reconstructed
volumes. Overall, cryoSplat shows strong noise robustness: SNRs above 0 dB have little impact on
reconstruction; high resolution is preserved even under severe noise at −15 dB, and reconstructions
remain satisfactory at −20 dB, despite particles being barely visible.
19
